"""S7 pool8: overlap the exact RTR2 FP32 suffix with late GD weight stages.

Writes only box_out/s7/k3_pool8_v1.py and k3_pool8_v2.py. The new CUDA
template parameter defaults OFF (PIPE=0); these two generated candidates
explicitly select PIPE=1 (stages 8..15) / PIPE=2 (stages 4..11) for VF=2.
No new state, packs, module bank, launch, router precision or reduction order.

Premortem: (1) eight persistent accumulators could spill / lower residency;
(2) the earlier cp.async drain / shared loads could delay the weight stream.
The Linux gate requires zero spills and >=2-CTA resource capacity for VF=2.
Performance is UNMEASURED. The 17.4/17.9us baseline is kernel_baseline.md:76/87.
The extra router is 8 MiB / 4.64e6 = EST 1.808us of irreducible HBM service.
Only the suffix (not that service or an already-overlapped PDL edge) is moved.
An alternative, direct per-warp router loads, is rejected: it exposes dependent
global loads in the suffix and doubles weight-load requests (L1/L2 may combine
duplicates, so +8 MiB of requests is NOT evidence of +8 MiB of HBM traffic).

Usage: python -B tools/s7f_pool8_patches.py [--check] [--variant 1|2|all]
The report contains exact changed-line intervals and CPU proof/dry-run results.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import difflib
import hashlib
import json
from pathlib import Path
import time
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = {v: ROOT / f"box_out/s7/k3_pool8_v{v}.py" for v in (1, 2)}
TARGET = (0, 1, 1, 8, 16, 2, 1, 2, 1, 2, 0)
LIVE_GUARD = '''def _pool8_require(state):
 r=_gd_live(state,"pooler",8)
 if not _VFR["ok"]or r is None:
  raise RuntimeError("[pool8] fused VF=2 self-test/route failed; fallback is not a candidate")
 info=r["mod"].gdr_info(2)
 if info[1]!=0 or info[2]<2:
  raise RuntimeError(f"[pool8] requires zero local bytes and >=2 CTAs/SM: {list(info)}")
'''


def require(ok, why):
    if not ok:
        raise AssertionError(why)


def sub(text, old, new, what, count=1):
    # Same fail-closed, counted replacement convention as padz_patches.sub.
    require(text.count(old) == count, (what, text.count(old), count))
    return text.replace(old, new)


def node(tree, name):
    found = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name == name or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                                              t.id == name for t in n.targets)]
    require(len(found) == 1, (name, len(found)))
    return found[0]


def literal(src, name):
    return ast.literal_eval(node(ast.parse(src), name).value)


def cuda_edit(cu, mode):
    require(type(mode) is int and mode in (0, 1, 2), "invalid PIPE selector")
    template = ("template <int OUT, int MT, int WR, int WK, int KU, int ST, "
                "int S, int MB, int XP, int VF = 0, int CAT = 0>\n")
    cu = sub(cu, template, template.replace("int CAT = 0>", "int CAT = 0, int PIPE = 0>"), "default-off template")

    suffix_start = "    if constexpr (VF == 2) {   // router32_kernel<8>'s arithmetic"
    suffix_end = "    if constexpr (VF >= 1) {\n      if (A.marks) {"
    lo, hi = cu.index(suffix_start), cu.index(suffix_end)
    suffix = cu[lo:hi]
    start_rows = suffix.index("const int row_pair = warp >> 1, token_half = warp & 1;\n")
    start_acc = suffix.index("float racc0[4], racc1[4];\n")
    start_loop = suffix.index("#pragma unroll\nfor (int i = 0; i < 8; ++i) {\n")
    start_reduce = suffix.index("#pragma unroll\nfor (int j = 0; j < 4; ++j) {\nfloat a0")
    # Copy the proven arithmetic, rather than transcribe it. Drop only i's loop
    # wrapper: the unrolled GD loop supplies i=0..7 in the identical order.
    rows = suffix[start_rows:start_acc]
    chunk = suffix[start_loop:start_reduce]
    body = chunk[len("#pragma unroll\nfor (int i = 0; i < 8; ++i) {\n"):-2]
    require(chunk.endswith("}\n") and body.count("fmaf(") == 8, "RTR2 chunk changed")
    helper = ("// S7 POOL8: one unchanged RTR2 K128 chunk; all eight warps participate.\n"
              "__device__ __forceinline__ void pool8_step(int i, int lane, int warp,\n"
              "    float (&racc0)[4], float (&racc1)[4]) {\n" + rows + body + "}\n")
    cu = sub(cu, "#define GC_XP(S_)", helper + "#define GC_XP(S_)", "chunk helper")

    loop_anchor = "#pragma unroll\nfor (int s = 0; s < KU; ++s) {\n"
    init = """// S7 POOL8: the selector defaults off; only the explicit VF=2 instance opts in.
static_assert(PIPE >= 0 && PIPE <= 2, "POOL8 PIPE must be 0, 1 or 2");
static_assert(!PIPE || (VF == 2 && MT == 1 && WR == 1 && WK == 8 &&
    KU == 16 && ST == 2 && S == 1 && CAT == 0), "POOL8 exact fused route only");
float pool8_r0[PIPE ? 4 : 1], pool8_r1[PIPE ? 4 : 1];
if constexpr (PIPE) {
#pragma unroll
for (int j = 0; j < 4; ++j) { pool8_r0[j] = 0.f; pool8_r1[j] = 0.f; }
}
"""
    cu = sub(cu, loop_anchor, init + loop_anchor, "persistent router accumulators")
    chunk_hook = """    // S7 POOL8: weight reloads and the four HMMAs precede this independent work.
    if constexpr (PIPE) {
      constexpr int first = PIPE == 1 ? 8 : 4;
      if (s >= first && s < first + 8) {
        if (s == first) {
          asm volatile("cp.async.wait_all;\\n" ::: "memory");
          __syncthreads(); // all 256 threads: both original copy groups are complete
        }
        pool8_step(s - first, lane, warp, pool8_r0, pool8_r1);
      }
    }
"""
    hook = "    if constexpr (CAT > 0) {\n      if (s == KUP - 1) {"
    cu = sub(cu, hook, chunk_hook + hook, "interleaved router chunks")

    # Preserve the stock branch byte-for-byte. The new branch only reduces /
    # publishes the exact per-lane accumulators computed in the GD loop.
    reduce = suffix[start_reduce:]
    require(reduce.endswith("}\n}\n}\n"), "RTR2 suffix braces changed")
    reduce = reduce[:-4]  # leave j's closing brace; remove warp and VF braces
    reduce = reduce.replace("racc0", "pool8_r0").replace("racc1", "pool8_r1")
    new_suffix = ("    if constexpr (VF == 2 && PIPE) {\n"
                  "const int row_pair = warp >> 1, token_half = warp & 1;\n" + reduce +
                  "    } else {\n" + suffix + "    }\n")
    cu = sub(cu, suffix, new_suffix, "router suffix becomes reduction only")
    old_instance = "gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2>"
    new_instance = old_instance[:-1] + f", 0, {mode}>"
    cu = sub(cu, old_instance, new_instance, "VF=2 launch/attribute/info arm", 3)
    return cu


def generate(base, mode=0):
    require("\r" not in base and "POOL8" not in base, "base must be untouched LF source")
    old = literal(base, "GD_CUDA_SRC")
    new = cuda_edit(old, mode)
    out = sub(base, old, new, "GD source literal")
    if mode:
        out = sub(out, "def _vfr_prepare(state):\n", LIVE_GUARD + "def _vfr_prepare(state):\n", "fail closed on fused-route loss")
        out = sub(out, " _vfr_prepare(state)\n", " _vfr_prepare(state)\n _pool8_require(state)\n", "prepare live-route gate")
    lines = out.splitlines(keepends=True)
    lines[1] = f"# nonce pool8-v{mode} (exact RTR2 interleaved in GD; PIPE={mode}; tools/s7f_pool8_patches.py)\n"
    out = "".join(lines)
    require(len(out.encode()) < 1_000_000, "submission size")
    compile(out, f"pool8-v{mode}", "exec")
    return out


def builder_capture(src):
    """Execute the base's real GD builder / routes, mocking load_inline only.

    This follows the proven no-import CPU extraction used by local/gdesc_gate.sh;
    no layer is imported, no CUDA device is queried, no module handle is invented.
    """
    tree = ast.parse(src)
    names = ("HIDDEN", "D_EXPERT", "N_EXPERTS", "UNPOOL_SLICE", "SLICED_UNPOOL_MAX_TOKENS",
             "GD_SMS", "GD_ROUTE", "GD_CAT_ROUTE", "GD_CAT_UNITS", "GD_DIMS", "GD_CPP_SRC",
             "GD_CUDA_SRC", "_GD", "_gd_nk", "_gd_key", "_gd_cat_key", "_gd_cat_keys", "_gd_keys")
    captures, marker = [], object()
    def load_inline(**kw):
        captures.append(kw)
        return marker
    real_import = builtins.__import__
    def importer(name, *args, **kw):
        if name == "torch.utils.cpp_extension":
            return types.SimpleNamespace(load_inline=load_inline)
        return real_import(name, *args, **kw)
    ns = {"hashlib": hashlib, "time": time, "_pt1_wait": lambda _: None,
          "__builtins__": dict(vars(builtins), __import__=importer)}
    exec(compile(ast.Module(body=[node(tree, n) for n in names + ("_gd_build",)], type_ignores=[]),
                 "pool8-real-GD-builder", "exec"), ns)
    require(ns["_gd_build"]() is marker and ns["_gd_build"]() is marker and len(captures) == 1,
            "GD build/cache dry run")
    require("@INST@" not in captures[0]["cuda_sources"] and "@INFO@" not in captures[0]["cuda_sources"],
            "unexpanded instantiations")
    return captures[0], ns


def dry_run(base, src):
    """Exercise actual _vfr_prepare, dispatch and base route-record construction.

    Tensor arithmetic and CUDA calls are symbolic ONLY. This verifies Python
    paths and exact state layout, not GPU numerical equivalence.
    """
    spec, ns = builder_capture(src)
    tree, btree = ast.parse(src), ast.parse(base)
    logs, runs = [], []
    class Tensor:
        device = "cpu-symbolic"
        def __init__(self, shape, dtype="f32", value=None):
            self.shape, self.dtype, self.value = tuple(shape), dtype, value
        def is_contiguous(self): return True
        def contiguous(self): return self
        def __mul__(self, other): return Tensor(self.shape, self.dtype)
        def pow(self, n): return self
        def mean(self, *a, **kw): return self
        def to(self, dtype): return Tensor(self.shape, dtype)
        def data_ptr(self): return 16
        def dim(self): return len(self.shape)
    class Generator:
        def __init__(self, **kw): pass
        def manual_seed(self, seed): return self
    class Module:
        info = [112, 0, 2, 4224]  # MOCKED admission, never hardware evidence
        def gdr_info(self, vf): return list(self.info)
        def gdr_run(self, vf, w, esc, x, y, t, k, n, tb, nbx, pdl, z, w32, part, marks):
            require((k, n, nbx) == (8192, 4096, 256), "real packed route keys")
            require(w is packed_w and esc is packed_e and w32 is router, "wrong state accessor")
            require(y.shape == (t, n) and part.shape == (8, t, 256), "output dimensions")
            require(vf in (1, 2) and (vf != 2 or z.shape == (t, 8192)), "VF selector/input")
            y.value = ("pool", t)
            part.value = ("part", t) if vf == 2 else None
            runs.append((vf, t, pdl))
        def gd_run(self, inst, w, esc, x, y, t, k, n, tb, nbx, nz, pdl):
            y.value = ("pool", t)
        def v8_router32(self, z, w, part): part.value = ("part", z.shape[0])
    torch = types.SimpleNamespace(
        bfloat16="bf16", float32="f32", int64="i64", Generator=Generator,
        empty=lambda shape, dtype=None, **kw: Tensor(shape, dtype),
        zeros=lambda shape, dtype=None, **kw: Tensor(shape, dtype),
        full=lambda shape, value, dtype=None, **kw: Tensor(shape, dtype, value),
        randn=lambda shape, **kw: Tensor(shape), rsqrt=lambda x: x,
        equal=lambda x, y: x.shape == y.shape and x.value == y.value,
        cuda=types.SimpleNamespace(synchronize=lambda: None))
    ns.update(torch=torch, _SBB1={"mod": object()}, _nb64=lambda t: False,
              _GD_SEEN=set(), _k3_disable=lambda *args: (_ for _ in ()).throw(AssertionError(args)))
    ns["__builtins__"] = dict(vars(builtins), print=lambda *a, **kw: logs.append(a))
    names = ("GD_ENABLE", "GD_PDL", "GD_TRIG", "VF_GDR", "VF_GDM", "_VFR",
             "_gd_bucket", "_gd_live", "_gd", "_vfr_gdr", "_vfr_fused", "_vfr_marked_pooler", "_vfr_prepare")
    if "def _pool8_require(" in src:
        names += ("_pool8_require",)
    exec(compile(ast.Module(body=[node(tree, n) for n in names], type_ignores=[]),
                 "pool8-real-VFR-prepare", "exec"), ns)
    # Evaluate the real _gd_prepare route-record dictionary; no hand-written
    # substitute schema is allowed to conceal an accessor typo.
    prep = node(btree, "_gd_prepare")
    rexpr = next(n.value for n in ast.walk(prep) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "r" for t in n.targets) and isinstance(n.value, ast.Dict))
    packed_w, packed_e = object(), object()
    mod, router = Module(), Tensor((256, 8192))
    rt = ns["GD_ROUTE"]["pooler", 8]
    key = ns["_gd_key"](*ns["_gd_nk"]("pooler", 8), rt)
    require(key == TARGET[:9], "actual pooler route changed")
    route = eval(compile(ast.Expression(rexpr), "base-_gd_prepare-record", "eval"),
                 {"mod": mod, "ids": {key: 0}, "key": key,
                  "pk": {"w": packed_w, "esc": packed_e, "tb": 0, "nbx": 256},
                  "n": 4096, "k": 8192, "rt": rt, "proj": "pooler", "UNPOOL_SLICE": 1024,
                  "desc": "m1(1,8,16,1)s2x1 CPU dry run"})
    state = {"gd": {"route": {("pooler", 8): route}}, "router": router, "v8": {"mod": mod}}
    # Execute the actual prepare hook, not a copied substitute invocation.
    hook = next(n for n in node(tree, "prepare").body if isinstance(n, ast.Expr) and
                isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_vfr_prepare")
    ns["state"] = state
    exec(compile(ast.Module(body=[hook], type_ignores=[]), "real-prepare-hook", "exec"), ns)
    require(ns["_VFR"]["ok"] and len(runs) == 6, "real VFR self-test Python paths")
    if "_pool8_require" in ns:
        live_hook = next(n for n in node(tree, "prepare").body if isinstance(n, ast.Expr) and
                         isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and
                         n.value.func.id == "_pool8_require")
        exec(compile(ast.Module(body=[live_hook], type_ignores=[]), "actual-new-prepare-hook", "exec"), ns)
        # Test the strict added gate even where the old 64-byte allowance would
        # accept local storage. Do not time a silently de-fused fallback arm.
        for info in ([112, 4, 2, 4224], [112, 0, 1, 4224]):
            mod.info = info
            try:
                ns["_pool8_require"](state)
            except RuntimeError:
                pass
            else:
                raise AssertionError("strict pool8 resource gate did not reject")
        mod.info = [112, 0, 2, 4224]
    require([(v, t) for v, t, p in runs] == [(v, t) for t in (8, 5, 2) for v in (2, 1)], "self-tests kept")
    for t in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 128, 256):
        z, x = Tensor((t, 8192)), Tensor((t, 8192), "bf16")
        got = ns["_vfr_fused"](state, x, z)
        require((got is not None) == (2 <= t <= 8), ("shape selector", t))
    ns["VF_GDM"] = True
    require(ns["_vfr_marked_pooler"](state, Tensor((8, 8192), "bf16")) is not None, "marked arm")
    ns["VF_GDM"] = False
    require(ns["_vfr_fused"](state, Tensor((8, 8192), "bf16"), Tensor((8, 8192), "bf16")) is None,
            "router FP32 guard")
    # A typo/missing required route key must fail in this CPU dry run.
    for missing in ("mod", "w", "esc", "N", "K", "tb", "nbx", "T"):
        saved = route.pop(missing)
        try:
            ns["_vfr_gdr"](state, Tensor((8, 8192), "bf16"), Tensor((8, 8192)), 2)
        except KeyError as exc:
            require(exc.args == (missing,), ("wrong missing key", exc))
        else:
            raise AssertionError(("missing key hidden", missing))
        finally:
            route[missing] = saved
    # Existing fail-closed preparation remains live for unavailable/spilled arms.
    for info in ([128, 0, 1, 4224], [112, 128, 2, 4224]):
        mod.info = info
        ns["_vfr_prepare"](state)
        require(not ns["_VFR"]["ok"], "resource fallback")
    mod.info = [112, 0, 2, 4224]
    ns["_vfr_prepare"](state)
    require(ns["_VFR"]["ok"], "repeat prepare")
    ns["_vfr_prepare"](dict(state, gd={"route": {}}))
    require(not ns["_VFR"]["ok"], "missing-route fallback")
    if "_pool8_require" in ns:
        try:
            ns["_pool8_require"](state)
        except RuntimeError:
            pass
        else:
            raise AssertionError("silent VF fallback accepted")
    return {"status": "PASS", "actual_builder_cache": True, "actual_prepare_hook": True,
            "actual_route_dict": True, "selector_shapes": [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 128, 256],
            "missing_key_negative_tests": 8, "bank_construction": "N/A: no new bank/state",
            "GPU_numerics": "NOT RUN; symbolic tensor results are not numerical evidence",
            "mocked_resource_info": "not a hardware measurement", "calls": len(runs)}, spec


def geometry_gate(mode):
    first = 8 if mode == 1 else 4
    schedule = [(s, s - first) for s in range(16) if first <= s < first + 8]
    require([i for s, i in schedule] == list(range(8)), "router K iteration order")
    # Verify every output's ordered per-lane FP32 operand indices and XOR tree.
    # No floating-point model is needed: identical FMA sequences prove equality
    # for finite values, signed zeros and special values under identical flags.
    checked = 0
    for warp in range(8):
        for lane in range(32):
            for j in range(4):
                for r in range(2):
                    row, tok = 2 * (warp >> 1) + r, 4 * (warp & 1) + j
                    old = [(row, tok, 128 * i + 4 * lane + c) for i in range(8) for c in range(4)]
                    new = [(row, tok, 128 * i + 4 * lane + c) for s, i in schedule for c in range(4)]
                    require(old == new, "FP32 FMA order")
                    checked += len(old)
    for m in range(1, 9):
        parts, pool = set(), set()
        for cta in range(256):
            bx, sl = cta & 31, cta >> 5
            for tid in range(256):
                # Main combine uses i=tid<128, all eight K warps from red.
                for i in range(tid, 128, 256):
                    lane, c = i & 31, (i >> 5) & 3
                    row = cta * 16 + (lane >> 2) + 8 * (c >> 1)
                    tok = 2 * (lane & 3) + (c & 1)
                    if tok < m:
                        idx = (tok, row)
                        require(idx not in pool, "duplicate pool writer")
                        pool.add(idx)
                if tid % 32:
                    continue
                warp = tid >> 5
                for j in range(4):
                    tok = 4 * (warp & 1) + j
                    if tok < m:
                        for r in range(2):
                            idx = (sl, tok, bx * 8 + 2 * (warp >> 1) + r)
                            require(idx not in parts, "duplicate part writer")
                            parts.add(idx)
        require(len(parts) == 8 * m * 256 and len(pool) == m * 4096, "output coverage")
    # The unchanged pre-wait weight and post-wait x copy loops cover each
    # float4 once; the new uniform wait drains both groups before any LDS.
    owners = [i for t in range(256) for i in range(t, 2048, 256)]
    require(sorted(owners) == list(range(2048)), "copy coverage")
    return {"status": "PASS", "FMA_operand_indices_compared": checked,
            "router_stage_schedule": schedule, "M1_through_M8_write_coverage": True,
            "protocol": ["CTAs0..255 tid0..255: original two copy groups, wait_all and CTA barrier (moved once)",
                         "CTAs0..255 tid0..255: red[warp][0:4][lane] stores and original combine barrier",
                         "CTAs0..255 tid0..127: 128 pool combine/store indices, all 8 K warp partials read",
                         "CTAs0..255 all 8 warps: original XOR16,8,4,2,1 router reduction",
                         "CTAs0..255 tid=32*w, w0..7: 2 rows x 4 live tokens into A.part",
                         "marks only: all threads original final barrier; tid0 stores final timestamp",
                         "No exited CTAs, counters, flags, resets, combines or empty-barrier arrivals added/removed",
                         "Dynamic smem is read-only after copy completion; no reusable async slot"]}


def cpu_gate(base, generated, mode):
    require(literal(generated, "GD_CUDA_SRC") == cuda_edit(literal(base, "GD_CUDA_SRC"), mode), "CUDA edit mismatch")
    # Reverse exactly the CUDA literal, strict live guard and nonce. All base
    # accessors, self-tests, routes and every other CUDA module stay identical.
    restored = sub(generated, literal(generated, "GD_CUDA_SRC"), literal(base, "GD_CUDA_SRC"), "restore CUDA")
    restored = sub(restored, LIVE_GUARD, "", "restore live guard")
    restored = sub(restored, " _pool8_require(state)\n", "", "restore live hook")
    lines = restored.splitlines(keepends=True)
    lines[1] = base.splitlines(keepends=True)[1]
    require("".join(lines) == base, "unrelated source changed")
    require(literal(generated, "GD_CUDA_SRC").count("__trap") == literal(base, "GD_CUDA_SRC").count("__trap"), "new trap")
    dry, spec = dry_run(base, generated)
    matcher = difflib.SequenceMatcher(a=base.splitlines(), b=generated.splitlines(), autojunk=False)
    edits = [{"kind": op, "base_lines_inclusive": [a+1, b], "candidate_lines_inclusive": [c+1, d]}
             for op, a, b, c, d in matcher.get_opcodes() if op != "equal"]
    return {"status": "CPU_PASS", "variant": mode, "bytes": len(generated.encode()),
            "sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "identity_except_GD_CUDA_live_guard_and_nonce": True, "prepare_dry_run": dry,
            "geometry": geometry_gate(mode), "per_selector_changed_lines": {f"PIPE={mode}": edits},
            "baseline_MEASURED": {
                "v1s_us": 17.4, "v1f_us": 17.9,
                "source": "notes/study/evo/s7/kernel_baseline.md:76,87 (41-run medians)",
                "d1s_us": 14.0, "d1s_source": "notes/study/evo/s7/kernel_baseline.md:6",
                "previous_defused_T8_pooler_us": [13.6, 14.1],
                "previous_defused_source": "notes/reset/LOG.md:837 (single runs, XP6/7; not a controlled attribution)",
                "new_candidate_savings": "UNMEASURED; no exact attribution of the 3.4us gap"},
            "accounting_EST": {
                "pooler_weight_bytes_per_CTA_stage_before_after": [12288, 12288],
                "pooler_weight_bytes_total_before_after": [50331648, 50331648],
                "pooler_escape_bytes_total_before_after": [262144, 262144],
                "router_weight_bytes_per_CTA_before_after": [32768, 32768],
                "router_weight_bytes_total_before_after": [8388608, 8388608],
                "router_input_L2_read_bytes_per_CTA_before_after": [32768, 32768],
                "router_smem_read_bytes_per_CTA_before_after": [196608, 196608],
                "router_partial_output_bytes_total_before_after_T8": [65536, 65536],
                "weight_stream_growth": False,
                "GD_per_thread_per_unit_before_after": "3 LDG128 W + 2 LDG128 X + 36 decode operations + 4 MMA; identical variable escape work; last 2 units have no W refill",
                "router_per_thread_before_after": "48 LDS128 + 256 FMA + 40 SHFL + 40 FADD; identical. 16 cp.async16 + 2 commits + 1 wait_all; identical",
                "changed_8_units_per_thread_added": "each 6 LDS128 + 32 FMA (38 data operations); removed from suffix; address/control SASS must be measured",
                "barriers_before_after": "2 CTA barriers, plus 1 when marks enabled; unchanged count",
                "extra_persistent_register_values": 8,
                "saving_model": ("EST conditional v1s/v1f: 8 warps*2 CTAs*(48+256) issue slots/(4 SMSPs*1520 cycles/us)=0.800us; "
                                 + ("hide 7/8 => 0.700us." if mode == 1 else "hide 8/8 => 0.800us.")
                                 + " Four issue slots/SM/cycle is an architecture assumption, not measured achieved throughput; complete hiding is unverified."),
                "critical_path": "router suffix follows the final GD MMA and gates A.part/tail3; moved work can occupy stream stalls. Existing warp/PDL overlap may reduce carry to zero; no measured saving claimed",
                "value": ("EST 0.700*0.146=0.1022" if mode == 1 else "EST 0.800*0.146=0.1168") + " %-pts using standing T8 constant; no credited gain",
                "irreducible_router_HBM_us": "EST 8388608 / 4.64e6 = 1.808",
                "alternative_direct_router_loads": "REJECT: dependent global loads replace ready LDS; weight-load requests double from 8 to 16 MiB. L1/L2 reuse may avoid added HBM, so no time penalty or >=0.5us gain is established; not built"},
            "compile_SASS_GPU": "NOT RUN; tools/local/s7f_pool8_gate.sh"}, spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--variant", choices=("1", "2", "all"), default="all")
    args = ap.parse_args()
    base = BASE.read_bytes().decode()
    variants = (1, 2) if args.variant == "all" else (int(args.variant),)
    reports = []
    # Default-off is also generated and exercised, without writing another file.
    off = generate(base, 0)
    dry_run(base, off)
    for mode in variants:
        generated = generate(base, mode)
        result, _ = cpu_gate(base, generated, mode)
        if args.check:
            require(OUT[mode].read_bytes() == generated.encode(), "stale output")
        else:
            OUT[mode].parent.mkdir(parents=True, exist_ok=True)
            OUT[mode].write_bytes(generated.encode())
        result["path"] = str(OUT[mode].relative_to(ROOT))
        reports.append(result)
    print(json.dumps({"default_off_dry_run": "PASS", "variants": reports}, indent=2))


if __name__ == "__main__":
    main()

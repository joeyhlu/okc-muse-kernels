"""S7-KS: audited, default-off KS3 load-order experiment; no GPU work.

Only writes box_out/s7/k3_ks_v1.py and tools/s7_ks_gate.json. The immutable
input is kernels/muse_decode_layer/k3_tinys7_c1.py. Run with python -B -I.

S7_KS_BITS is a PREPARE-TIME selector, not a call-time arm:
  0: original CUDA prologue, byte for byte after preprocessing (default).
  1: <16,1>: issue positions/cos_sin/cache_loc ahead of the existing gather.
  2: <8,0>: wait for QKV, issue Q/new-KV/gate plus metadata ahead of gather.
  3: both. There is no extra module bank, state key, launch, byte, or wait.
The original launch_dependents location, chunk32, and every merge/reset stay.

Performance verdict: KILL for promotion under the requested >=1.5 us combined bar.
R-ks3g-0 (notes/lab.md:516) already puts the gather at its pattern ceiling.
Issue order is not evidence of exposed latency: SASS and GPU timing remain
necessary. This complete opt-in build is not a claimed performance winner.
Kill reasons: Q/metadata may already overlap the gather; longer register
lifetimes may spill or reduce useful overlap. In particular, an earlier PDL
wait can remove pre-wait gather overlap on nearby <8,0> fallback shapes.

No changed hot loop. Per-thread old-key gather: NS16 8 cp.async x16 B=128 B;
NS8 16 x16 B=256 B. Per CTA 32768/65536 B, unchanged. Each chunk32 still has
32 mma16816 calls, 64 fp8-pair conversions, 4 uint4 K LDS, 8 uint2 V LDS,
16 byte_perm, and 9 ex2 calls per thread. NCH=1/2; tail warps 6/7 do +1.
These are source operations, not a fabricated SASS instruction count.
New prologue performs the same loads, arithmetic, waits, and barriers;
the Linux gate records actual SASS counts before and after.

Protocol coverage (unchanged, checked below): all 256 threads finish the
cp.async wait, stage their Q row, and participate in the local merge. Tail
CTA threads 0..255 zero both 4096-B new-key buffers; threads 0..127 write
the two new 128-B cache rows. MG1: every CTA publishes 16*64 FP32 partials
and 16 denominators; thread 0 takes the ticket. Non-last CTAs may then exit.
The last CTA's threads 0..255 read/reset every float4, tid%16==0 read/reset
the 16 denominators, all threads write output, and thread 0 resets counter.
MG0: every CTA writes its own 16*64 partials and 16 float2 statistics before
the release ticket; the last CTA's 256 threads combine all NS slots, write
the output, and thread 0 resets counter. No MG0 partial reset is required:
all slots are overwritten before the next ticket can elect a last CTA.
"""

import argparse
import ast
import contextlib
import copy
import difflib
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "box_out/s7/k3_ks_v1.py"
REPORT = ROOT / "tools/s7_ks_gate.json"

PY_HELPER = '''S7_KS_BITS=0 # prepare-time only: 1=d16 metadata, 2=d32 Q+metadata
def _s7_ks_src():
 if type(S7_KS_BITS) is not int or not 0<=S7_KS_BITS<=3:
  raise ValueError("S7_KS_BITS must be an integer in 0..3 before prepare")
 return "#define S7_KS_BITS %d\\n"%S7_KS_BITS+KS1_CUDA_SRC
'''


def require(ok, why):
    if not ok:
        raise AssertionError(why)


def sub(s, old, new, expected=1):
    require(s.count(old) == expected, ("anchor count", old[:90], s.count(old), expected))
    return s.replace(old, new)


def region(s, begin, end):
    require(s.count(begin) == 1 and s.count(end) == 1, ("unique region", begin, end))
    a = s.index(begin)
    b = s.index(end, a)
    return s[a:b]


def functions(s):
    return {n.name: n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}


def literal(s, key):
    found = [n.value for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id == key]
    require(len(found) == 1, ("assignment", key, len(found)))
    return ast.literal_eval(found[0])


def patch(row, bits=0):
    require(type(bits) is int and 0 <= bits <= 3, "selector range")
    start = "KS3P(0);\n"
    finish = "{\nconst float2 a = norm_rope(qr0, cs, p.rope, p.eps)"
    old = region(row, start, finish)
    seq = region(old, "const int seq =", "{\nconstexpr int PER =")
    gather = region(old, "{\nconstexpr int PER =", "float2 cs =")
    cs = region(old, "if (p.rope) {\n", "long long loc =")
    zero = region(old, "reinterpret_cast<uint4*>(kn)[tid]", "KS3P(1);")
    # zero ends in the closing brace of the tail conditional; retain it.
    loads = old[old.index("const uint32_t qr0 ="):]
    qdecl = 'const __nv_bfloat16* xr = p.qkv + (size_t)b * p.stride_qkv;\n'
    rowdecl = 'const int mr = tid >> 4, mu = tid & 15;\n'
    loads = sub(loads, rowdecl, "")
    loads = sub(loads, "const uint32_t qr0 =", "qr0 =")
    loads = sub(loads, "const uint32_t qr1 =", "qr1 =")
    loads = sub(loads, "uint32_t nr = 0u;\n", "")
    loads = sub(loads, "const uint2 graw =", "graw =")
    metadata = cs + "if (tail) loc = __ldg(p.cache_loc + b);\n"
    new = (
        "KS3P(0);\n"
        "constexpr bool S7Q = (S7_KS_BITS & 2) && NS == 8 && MG == 0;\n"
        "constexpr bool S7M = S7Q || ((S7_KS_BITS & 1) && NS == 16 && MG == 1);\n"
        "float2 cs = make_float2(1.f, 0.f);\nlong long loc = 0;\n" + qdecl + rowdecl +
        "uint32_t qr0 = 0u, qr1 = 0u, nr = 0u;\nuint2 graw = make_uint2(0u, 0u);\n"
        "if constexpr (S7Q) {\npdl_wait();\n" + loads + "}\n" + seq +
        "if constexpr (S7M) {\n" + metadata + "}\n" + gather +
        "if constexpr (!S7M) {\n" + metadata + "}\n"
        "if (tail) {\n" + zero +
        "KS3P(1);\nif constexpr (!S7Q) pdl_wait();\n"
        "if (p.trig == 1) pdl_trigger();\nKS3P(2);\n"
        "if constexpr (!S7Q) {\n" + loads + "}\n")
    result = sub(row, old, "#if S7_KS_BITS == 0\n" + old + "#else\n" + new + "#endif\n")
    result = sub(result, "namespace ks3 {\n", "#ifndef S7_KS_BITS\n#define S7_KS_BITS 0\n#endif\nnamespace ks3 {\n")
    result = sub(result, "def _ks1_build():\n", PY_HELPER.replace("S7_KS_BITS=0", f"S7_KS_BITS={bits}") + "def _ks1_build():\n")
    result = sub(result, 'cuda=(KS1_CUDA_SRC.replace("@INSTS@",insts)', 'cuda=(_s7_ks_src().replace("@INSTS@",insts)')
    lines = result.splitlines(keepends=True)
    require(lines[1].startswith("# nonce tinys7-c1 "), "row nonce")
    lines[1] = f"# nonce ks-v1 (S7 KS3 load ordering; prepare selector {bits}; tools/s7_ks_patches.py)\n"
    result = "".join(lines)
    require("\r" not in result and result.isascii(), "ASCII LF candidate")
    require(len(result.encode("ascii")) < 1_000_000, "submission byte cap")
    compile(result, str(OUT), "exec")
    return result


def cuda_source(s, bits=None):
    cuda = literal(s, "KS1_CUDA_SRC")
    if bits is None:
        bits = literal(s, "S7_KS_BITS") if "def _s7_ks_src():" in s else 0
    return f"#define S7_KS_BITS {bits}\n" + cuda


def device_unit(s, bits=0):
    """Actual device code and launchers, with explicit KS3 instantiations; no torch/GPU."""
    cuda = cuda_source(s, bits)
    body = region(cuda, "namespace ks1 {", "#define KS1_INSTS @INSTS@")
    header = (
        "#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n"
        "#include <cuda_fp8.h>\n#include <stdint.h>\n#include <vector>\n#include <stdexcept>\n"
        "#define C10_CUDA_CHECK(x) do { if ((x) != cudaSuccess) throw std::runtime_error(\"cuda\"); } while (0)\n"
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)\n#define KS1_ARCH_OK 1\n#else\n#define KS1_ARCH_OK 0\n#endif\n"
        f"#define S7_KS_BITS {bits}\n")
    inst = "".join(f"template __global__ void ks3::ks3_kernel<{ns}, {mg}>(const ks3::Params3);\n"
                   for ns, mg in ((16, 1), (8, 0), (16, 0), (4, 0)))
    return header + body + inst


class Tensor:
    """Only metadata for running the REAL prepare/call Python without CUDA."""
    def __init__(self, shape, dtype="float32", device="cpu-dry-run"):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype, self.device = dtype, device

    def numel(self):
        return math.prod(self.shape)

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def stride(self, dim):
        return math.prod(self.shape[dim + 1:])

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            return Tensor(self.shape, args[0], self.device)
        return Tensor(args, self.dtype, self.device)


class Module:
    def __init__(self):
        self.calls = []

    def ks1_info(self, *args):
        return (0, 0, 0, 2)

    def ks3_info(self, *args):
        return (0, 0, 0, 2)

    def ks3_forward(self, *args):
        require(len(args) == 19, ("ks3 arity", len(args)))
        B = args[0].shape[0]
        ns = args[13] % 100
        mg = args[13] // 100
        require(args[11].shape == (B, 1024), "output shape")
        need_o = B * (ns if mg == 0 else 1) * 16 * 64
        need_ml = B * (ns * 16 * 2 if mg == 0 else 16)
        require(args[8].numel() >= need_o and args[9].numel() >= need_ml, "workspace size")
        self.calls.append((B, ns, mg))


def dry_run(candidate):
    tree = ast.parse(candidate)
    names = {"_s7_ks_src", "_ks1_build", "_ks1_prepare", "_vsl_prepare", "_ks1_counters",
             "_ks3_key", "_ks3_routed", "_ks3_call", "_ks1_run", "_vsl_ws", "_vsl_pick",
             "_vsl_call", "_vsl_variants", "_ks1_wanted", "_hya_plan"}
    defs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            if node.name == "_ks1_build":
                # Replace ONLY the unavailable CUDA extension import. Execute the
                # actual build function, argument assembly, hash, and cache paths.
                node.body = [ast.Pass() if isinstance(n, ast.ImportFrom) and
                             n.module == "torch.utils.cpp_extension" else n for n in node.body]
            defs.append(node)
    require({d.name for d in defs} == names, "dry-run real function set")
    code = compile(ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[])), str(OUT), "exec")
    rows, tags = [], []
    for bits in range(4):
        env = {"__builtins__": __builtins__}
        # Evaluate only the base file's exact literal/constant declarations.
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                key = n.targets[0].id
                if key.startswith(("KS1_", "KS2_", "KS3_", "VSL_")) or key in (
                        "_KS1", "_VSL", "NB_KS3_B", "S7_KS_BITS", "ROPE_THETA"):
                    exec(compile(ast.Module(body=[n], type_ignores=[]), str(OUT), "exec"), env)
        require(env["_KS1"] == {"mod": None, "cnt": {}, "build_s": 0.0}, "real module layout")
        require(env["_VSL"] == {"ws": {}, "seen": set()}, "real VSL layout")
        env["S7_KS_BITS"] = bits
        mod, builds = Module(), []

        def loader(**kwargs):
            builds.append(kwargs)
            src = kwargs["cuda_sources"]
            require(src.startswith(f"#define S7_KS_BITS {bits}\n"), "compile selector consumed")
            require(not any(x in src for x in ("@INSTS@", "@MINSTS@", "@M3INSTS@")), "instantiation expansion")
            require("KS3_XM(16, 1)" in src, "d16 instance built")
            require(kwargs["functions"] == ["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"], "exports")
            return mod

        torch = SimpleNamespace(**{x: x for x in ("float32", "bfloat16", "int32", "int64", "uint8")},
                                empty=lambda shape, dtype, device: Tensor(shape, dtype, device),
                                zeros=lambda shape, dtype, device: Tensor(shape, dtype, device))
        env.update(torch=torch, FP8="fp8", hashlib=hashlib, time=SimpleNamespace(time=lambda: 10.0),
                   load_inline=loader, _pt1_wait=lambda key: None,
                   _dg1_fold=lambda which: [] if which == "fns" else "",
                   _dg1_fold_on=lambda: False, _P1_MOD=None)
        # Device numeric tests are the ONLY prepare tests stubbed here; there is
        # deliberately no claim of GPU correctness from this metadata dry run.
        env["_ks1_self_test"] = lambda cs, dev, geos, verbose: set(geos)
        env["_ks2_self_test"] = lambda cs, dev, geos: set(geos)
        env["_ks3_self_test"] = lambda cs, dev: set(env["KS3_INSTS"])
        env["_vsl_self_test"] = lambda cs, dev: set(env["_vsl_variants"]())
        exec(code, env)
        state = {"router": Tensor((256, 4096)), "cos_sin": Tensor((8192, 64))}
        with contextlib.redirect_stdout(io.StringIO()):
            env["_ks1_prepare"](state, 0.0)
        require(state["ks1"] == set(env["KS1_INSTS"]) and state["ks3"] == {8}, "prepare state not silently disabled")
        require(("ks3", 16, 1) in state["vsl"], "real VSL prepare state")
        require(len(builds) == 1 and env["_KS1"]["mod"] is mod, "build completed")
        require(env["_ks1_build"]() is mod and len(builds) == 1, "module cache path")
        tags.append(builds[0]["name"])
        for B, expected in ((16, (16, 1)), (32, (8, 0)), (17, (8, 0)), (31, (8, 0)), (33, (8, 0)), (64, (8, 0))):
            cfg = dict(batch=B, q_len=1, window=2048, rope_theta=env["ROPE_THETA"],
                       page_size=1, kv_scale=1.0, head_dim=64, num_q_heads=16, num_kv_heads=2,
                       sm_scale=0.125, qk_scale_factor=1.0)
            qkv = Tensor((B, 2304), "bfloat16")
            data = dict(config=cfg, k_cache=Tensor((8192, 128), "fp8"),
                        v_cache=Tensor((8192, 128), "fp8"), page_table=Tensor((B, 4096), "int32"),
                        cache_seqlens=Tensor(B, "int32"), cache_loc=Tensor(B, "int64"), positions=Tensor(B, "int64"))
            require(env["_ks1_wanted"](qkv, data, cfg, state), ("wanted", bits, B))
            with contextlib.redirect_stdout(io.StringIO()):
                out = env["_ks1_run"](qkv, data, state)
            require(out is not None and out.shape == (B, 1024), "wrapper must return output")
            require(mod.calls[-1] == (B, *expected), ("exact route", mod.calls[-1], expected))
            if B == 16:
                old = state["vsl"]
                state["vsl"] = set()
                env["_ks1_run"](qkv, data, state)
                require(mod.calls[-1] == (16, 8, 0), "d16 fallback route")
                state["vsl"] = old
        for bad in (-1, 4, 1.0, True, "1", None):
            env["S7_KS_BITS"] = bad
            try:
                env["_s7_ks_src"]()
            except ValueError:
                pass
            else:
                raise AssertionError(("invalid selector accepted", bad))
        env["S7_KS_BITS"] = bits
        # Required-key negative test: real prepare cannot hide a missing router.
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                env["_ks1_prepare"]({"cos_sin": state["cos_sin"]}, 0.0)
            except KeyError as exc:
                require(exc.args == ("router",), "wrong missing-key failure")
            else:
                raise AssertionError("missing real state key did not fail")
        plans = [env["_hya_plan"](B, 1, 8, 2048, 4096, 1, 132) for B in (16, 32)]
        require([p["n_splits"] for p in plans] == [8, 8], "base HYA plan")
        # v32s's unchanged HYA_TUNED overrides 8 to 4 at launch.
        rows.append(dict(bits=bits, routes=mod.calls, metadata_only=True, new_bank="none"))
    require(len(set(tags)) == 4, "different compile arms must have distinct module hashes")
    return {"status": "PASS", "arms": rows, "distinct_build_tags": tags,
            "limitation": "real prepare/call Python with fake tensors/compiler; numeric GPU self-tests stubbed"}


def coverage_gate():
    totals = []
    for ns in (4, 8, 16):
        kmax = 2048 // ns
        copied = {(tid + 256*j) for tid in range(256) for j in range(kmax*8//256)}
        require(copied == set(range(kmax*8)), "all KV vectors copied exactly once")
        keys = {(s*kmax + (w//2)*(kmax//4) + 32*c + k)
                for s in range(ns) for w in range(8) for c in range(kmax//128) for k in range(32)}
        require(keys == set(range(2048)), "all old-key slots attended")
        for head in (0, 1):
            hkeys = [s*kmax + (w//2)*(kmax//4) + 32*c + k
                     for s in range(ns) for w in range(8) if w%2 == head
                     for c in range(kmax//128) for k in range(32)]
            require(sorted(hkeys) == list(range(2048)), "each KV head covers each key exactly once")
        # Both KV heads, all query heads, and every output dimension.
        outputs = {(tid//16, 4*(tid%16)+j) for tid in range(256) for j in range(4)}
        require(outputs == {(r, j) for r in range(16) for j in range(64)}, "merge/reset output coverage")
        require({tid//16 for tid in range(256) if tid%16 == 0} == set(range(16)), "denominator reset coverage")
        require({tid + off for tid in range(256) for off in (0, 256)} == set(range(512)), "tail clear coverage")
        for seqlen in (0, 1, 2, 31, 128, 2047, 2048, 2049, 3000, 80000):
            seq = max(seqlen-1, 0)
            nold = min(seq, 2047)
            attended = {seq-nold+k for k in keys if k < nold} | {seq}
            require(attended == set(range(max(0, seq-2047), seq+1)), "window including new key")
        totals.append(dict(ns=ns, kv_bytes_per_cta=2*kmax*128, cp_async_per_thread=2*kmax*8//256,
                           smem_bytes=2*kmax*128+8192+2048+512+64, threads=256))
    return {"status": "PASS", "geometries": totals, "protocol_changed": False}


def cpu_gates(row, candidate):
    require(len(row.encode("ascii")) == 965632, "immutable task row size changed")
    require("\r" not in row and "\r" not in candidate, "LF")
    before, after = functions(row), functions(candidate)
    require(set(after) - set(before) == {"_s7_ks_src"}, "only one new Python helper")
    require(set(before) <= set(after), "no removed functions")
    for name in before:
        if name != "_ks1_build":
            require(ast.dump(before[name], include_attributes=False) == ast.dump(after[name], include_attributes=False),
                    ("unrelated function changed", name))
    oldcu, newcu = literal(row, "KS1_CUDA_SRC"), literal(candidate, "KS1_CUDA_SRC")
    a = "template <typename F>\n__device__ __forceinline__ void chunk32("
    b = "template <int NS, int MG = 0>\n__global__"
    require(region(oldcu, a, b) == region(newcu, a, b), "exact same chunk math")
    a = "{\nconst float2 a = norm_rope(qr0, cs, p.rope, p.eps)"
    require(oldcu[oldcu.index(a):] == newcu[newcu.index(a):], "norm/new KV/chunks/merge/reset/launch suffix unchanged")
    original = region(oldcu, "KS3P(0);\n", a)
    require("#if S7_KS_BITS == 0\n" + original + "#else\n" in newcu, "default-off CUDA prologue is verbatim")
    # The gather body itself appears twice: original arm and reordered arm.
    gather = region(original, "{\nconstexpr int PER =", "float2 cs =")
    require(newcu.count(gather) == 2, "gather bytes/instructions unchanged")
    require(newcu.count("__trap") == oldcu.count("__trap"), "no new trap")
    changed = []
    a_lines, b_lines = row.splitlines(), candidate.splitlines()
    for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False).get_opcodes():
        if tag != "equal":
            changed.append(dict(kind=tag, base_lines=[a0+1, a1], candidate_lines=[b0+1, b1]))
    def line(text):
        hits = [i for i, value in enumerate(b_lines, 1) if value == text]
        require(len(hits) == 1, ("line-map anchor", text, hits))
        return hits[0]
    def block_lines(text):
        first = line(text)
        depth = 0
        for last in range(first, len(b_lines)+1):
            depth += b_lines[last-1].count("{") - b_lines[last-1].count("}")
            if depth == 0:
                return [first, last]
        raise AssertionError(("unclosed line-map block", text))
    bit_lines = {
        "bit1": {"predicate": line("constexpr bool S7M = S7Q || ((S7_KS_BITS & 1) && NS == 16 && MG == 1);"),
                 "early_metadata": block_lines("if constexpr (S7M) {"),
                 "original_metadata_guard": block_lines("if constexpr (!S7M) {")},
        "bit2": {"predicate": line("constexpr bool S7Q = (S7_KS_BITS & 2) && NS == 8 && MG == 0;"),
                 "early_wait_and_q_loads": block_lines("if constexpr (S7Q) {"),
                 "original_wait_guard": line("if constexpr (!S7Q) pdl_wait();"),
                 "original_q_load_guard": block_lines("if constexpr (!S7Q) {"),
                 "metadata": "shares bit1 early/original metadata blocks"},
        "common": {"exact_diff_hunks": changed, "build_source": line(' cuda=(_s7_ks_src().replace("@INSTS@",insts).replace("@MINSTS@",minsts).replace("@M3INSTS@",m3insts)')},
    }
    return dict(status="PASS", python_compile="PASS", unaffected_python_ast="PASS", unchanged_math_and_protocol="PASS",
                coverage=coverage_gate(), dry_run=dry_run(candidate), changed_lines=changed, per_bit_lines=bit_lines,
                byte_accounting={"weight_stream_growth_bytes": 0, "kv_cta_before_after": {"d16s": [32768, 32768], "d32s": [65536, 65536]},
                                 "q_gate_load_bytes_per_cta_before_after": [4096, 4096],
                                 "tail_new_kv_load_bytes_before_after": [512, 512],
                                 "mg1_partial_payload_bytes_per_cta_before_after": [4160, 4160],
                                 "mg0_partial_payload_bytes_per_cta_before_after": [4224, 4224],
                                 "removed_launches": 0, "removed_waits": 0},
                chunk32_source_ops_before_after={"mma16816": [32, 32], "fp8_pair_conversions": [64, 64],
                                                "K_LDS128": [4, 4], "V_LDS64": [8, 8], "byte_perm": [16, 16], "ex2": [9, 9]},
                candidate_bytes=len(candidate.encode("ascii")), candidate_sha256=hashlib.sha256(candidate.encode("ascii")).hexdigest(),
                base_bytes=len(row.encode("ascii")), base_sha256=hashlib.sha256(row.encode("ascii")).hexdigest(),
                selector=literal(candidate, "S7_KS_BITS"), timing_status="NOT RUN; no defensible >=1.5 us summed saving",
                cuda_compile="NOT RUN: Windows sandbox cannot reach WSL/NVCC", sass="NOT RUN", gpu_correctness="NOT RUN")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, choices=range(4), default=0)
    ap.add_argument("--check", action="store_true", help="check the existing candidate without rewriting it")
    args = ap.parse_args()
    raw = BASE.read_bytes()
    require(b"\r" not in raw, "row LF")
    row = raw.decode("ascii")
    candidate = patch(row, args.bits)
    if args.check:
        require(OUT.read_bytes() == candidate.encode("ascii"), "candidate differs from deterministic generator")
    result = cpu_gates(row, candidate)
    if not args.check:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(candidate.encode("ascii"))
    require(BASE.read_bytes() == raw, "base must be unchanged")
    REPORT.write_bytes((json.dumps(result, indent=2) + "\n").encode("ascii"))
    print(json.dumps({k: result[k] for k in ("status", "candidate_bytes", "candidate_sha256", "selector", "cuda_compile", "timing_status")}, indent=2))


if __name__ == "__main__":
    main()

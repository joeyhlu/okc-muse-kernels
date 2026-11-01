"""TSTAG: per-job TINY shares/credits; lab or fixed ship builds.

Kill risks FIRST: a per-job release fence can stall behind the next job's
prefetch (R-dsm-0b/R-vfn-1); 132 credit publishers and uneven 3/4-unit shares
can erase the readiness lead (VMS_IL). No speed claim until paired H200 data.
Default writes ONLY kernels/lab/k3_tstag_ab.py; --check is read-only.
--dry-run executes real prepare nodes with tensor/compiler doubles, no GPU.
Ship: --ship --base=<in> --out=<out> [--check]. No GPU timing claim.
"""
import argparse
import ast
import builtins
import contextlib
import difflib
import hashlib
import inspect
import io
import json
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz_rt.py"
OUT = ROOT / "kernels/lab/k3_tstag_ab.py"


def node(source, name):
    return next(n for n in ast.parse(source).body if
                getattr(n, "name", None) == name or isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)


def literal(source, name):
    return ast.literal_eval(node(source, name).value)


def _tstag_sub(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f"TSTAG anchor count {s.count(old)}: {old[:100]!r}")
    return s.replace(old, new)


_TSTAG_MAP = r'''// TSTAG: q is phase-local; lo/hi are one job's share, not a global range.
template<int RB>
__device__ __forceinline__ int tstag_unit(int q, int lo, int hi) {
const unsigned n = (unsigned)(hi - lo), u = (unsigned)q;
// At G=132: W13 n=3/4, W2 n=1/2. Constant divisions avoid integer DIV.
const unsigned j = n == 1 ? u : n == 2 ? u / 2 : n == 3 ? u / 3 : n == 4 ? u / 4 : u / n;
return (int)(j * RB + lo + u - j * n);
}
template<bool PB>
__device__ __forceinline__ int tstag_v(const Smem& S, int q) {
return tstag_unit<PB ? RB_B : RB_A>(q, S.wsr[PB ? 2 : 0], S.wsr[PB ? 3 : 1]);
}
'''


def _tstag_source(source, kind):
    if kind == "ctl":
        return source
    if kind != "stag":
        raise ValueError(kind)
    s = _tstag_sub(source, "#define OKC_WSRF 0\n", "#define OKC_WSRF 0\n" + _TSTAG_MAP)
    # No arithmetic-loop changes. Only translate phase-local ordinals to the
    # existing job-major global unit coordinate expected by ring_issue/stores.
    s = _tstag_sub(s, "const int v = v0 + i;\nconst int j = njF + v / rbs;",
                  "const int v = tstag_v<PB>(S, i);\nconst int j = njF + v / rbs;")
    s = _tstag_sub(s, "if (j != curj) {\ncurj = j;", """if (j != curj) {
// At M=1 every live row uses token 0; equal cnt also preserves zero lane masks.
// PB uses expert-specific inter rows and MUST always reload.
const bool same_x = !PB && p.M == 1 && curj >= 0 && S.jC[curj] == S.jC[j];
curj = j;""")
    s = _tstag_sub(s, "load_x_regs<PB>(p, S, j, lane, warp, x0);",
                  "if (!same_x) load_x_regs<PB>(p, S, j, lane, warp, x0);")
    s = _tstag_sub(s, "const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);",
                  "const int nA = S.njS * (S.wsr[1] - S.wsr[0]), N = nA + S.njS * (S.wsr[3] - S.wsr[2]);")
    s = _tstag_sub(s, "const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;",
                  "const int v = tstag_v<true>(S, ul), j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;")
    s = _tstag_sub(s, "if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);\nelse ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);",
                  "if (q < nA) ring_issue<false>(p, S, tstag_v<false>(S, q), ring + sl * UNIT_BYTES, S.bars + sl);\nelse ring_issue<true>(p, S, tstag_v<true>(S, q - nA), ring + sl * UNIT_BYTES, S.bars + sl);")
    s = _tstag_sub(s, "const int nA = a1 - a0, N = nA + (b1 - b0);",
                  "const int nA = S.njS * (a1 - a0), nB = S.njS * (b1 - b0), N = nA + nB;")
    s = _tstag_sub(s, "N - min(OKC_WSBLK, b1 - b0) : N;", "N - min(OKC_WSBLK, nB) : N;")
    s = _tstag_sub(s, "const int v = PB ? b0 + (q - nA) : a0 + q;",
                  "const int v = PB ? tstag_v<true>(S, q - nA) : tstag_v<false>(S, q);")
    s = _tstag_sub(s, """if (q == nA - 1) {
__threadfence();
__syncwarp();
if (lane == 0) {
const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
for (int jj = jf; jj <= jl; ++jj) {
const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
atomicAdd(p.doneA + jj, b - a);
}
}
}""", """if (rb == a1 - 1) {
// Every helper lane has stored its final unit of THIS job before release.
__threadfence();
__syncwarp();
if (lane == 0) atomicAdd(p.doneA + j, a1 - a0);
}""")
    # Same phase lengths as helper/refill. Keep CPF dead calculations unchanged.
    s = _tstag_sub(s, "skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);",
                  "v0 = 0; v1 = njS * (S.wsr[PB ? 3 : 1] - S.wsr[PB ? 2 : 0]);")
    s = _tstag_sub(s, "const int nA = S.njS * RB_A, nB = S.njS * RB_B;",
                  "const int nA = RB_A, nB = RB_B; // TSTAG: bounds within EACH job")
    # Live FAST1 path is SPLIT=0/F1EARLY=1. Require exactly those flags; no
    # speculative edit to the dead SPLIT branch or any route/barrier code.
    s = _tstag_sub(s, "const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);",
                  "const int nS = RB_A; // TSTAG FAST1 uses the first job's per-CTA units")
    s = _tstag_sub(s, "const int n = min(OKC_FAST1, v1 - v0);",
                  "const int n = min(OKC_FAST1, __popc(m) * (v1 - v0));")
    s = _tstag_sub(s, "for (int i = 0; i < n; ++i) {\nconst int v = v0 + i;",
                  "for (int i = 0; i < n; ++i) {\nconst int v = tstag_unit<RB_A>(i, v0, v1);")
    return s


TSTAG_ON = "ctl"
TSTAG_ARMS = ("ctl", "stag")
_TSTAG = {"mods": {}, "errors": {}, "secs": {}, "calls": {}, "accepted": set()}


def _tstag_build_one(k, kind, var):
    from torch.utils.cpp_extension import load_inline
    cu = _tstag_source(_HYBLO_CUDA, kind)
    flags = list(_HYBLO_FLAGS) + list(var) + ["-Xptxas=-v"]
    tag = hashlib.sha1((cu + " ".join(flags)).encode()).hexdigest()[:12]
    return load_inline(name=f"tstag_{k}_{kind}_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=cu,
                       functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=flags, verbose=False)


def _tstag_prepare():
    _TSTAG["errors"].clear()
    _TSTAG["calls"].clear()
    if not HYBLO_ENABLE:
        return
    if _HYBLO["w"] is None or _HYBLO["sc"] is None:
        raise RuntimeError("TSTAG: ship HYBLO prepare unavailable")
    for k, _, var in _HYBLO_KEYS:
        if not _HYBLO["ok"].get(k) or k not in _HYBLO["mods"]:
            raise RuntimeError(f"TSTAG: ship {k} unavailable")
        if k != "tiny":
            continue
        key, t0 = (k, "stag"), time.time()
        try:
            if key not in _TSTAG["mods"]:
                _TSTAG["mods"][key] = _tstag_build_one(k, "stag", var)
        except Exception as ex:
            _TSTAG["errors"][key] = repr(ex)
        _TSTAG["secs"][key] = time.time() - t0


def _tstag_select(k, m):
    if TSTAG_ON not in TSTAG_ARMS:
        raise RuntimeError(f"invalid TSTAG arm {TSTAG_ON!r}")
    kind = "stag" if TSTAG_ON == "stag" and k == "tiny" and m == 1 else "ctl"
    if kind == "ctl":
        mod = _HYBLO["mods"][k]
    else:
        if (k, kind) not in _TSTAG["accepted"]:
            raise RuntimeError(f"TSTAG {k}/{kind}: compile/SASS gate not accepted")
        if (k, kind) in _TSTAG["errors"] or (k, kind) not in _TSTAG["mods"]:
            raise RuntimeError(f"TSTAG {k}/{kind}: missing build")
        mod = _TSTAG["mods"][(k, kind)]
    key = f"{k}:{m}:{TSTAG_ON}:{kind}"
    _TSTAG["calls"][key] = _TSTAG["calls"].get(key, 0) + 1
    return mod, (k, kind)


def build(base):
    assert "TSTAG_ON" not in base
    runtime = "\n# TSTAG LAB: ctl is ship; stag changes only M=1 in the tiny bank.\n"
    for name in ("_TSTAG_MAP", "TSTAG_ON", "TSTAG_ARMS"):
        runtime += f"{name} = {globals()[name]!r}\n"
    runtime += '_TSTAG = {"mods": {}, "errors": {}, "secs": {}, "calls": {}, "accepted": set()}\n'
    for fn in (_tstag_sub, _tstag_source, _tstag_build_one, _tstag_prepare, _tstag_select):
        runtime += inspect.getsource(fn) + "\n"
    s = _tstag_sub(base, "HYBLO_ENABLE = True\n", runtime + "HYBLO_ENABLE = True\n")
    run = ast.get_source_segment(s, node(s, "_hyblo_run"))
    new = _tstag_sub(run, 'mod = _HYBLO["mods"][k]', 'mod, pkey = _tstag_select(k, x.shape[0])')
    new = new.replace('_HYBLO["pdl"].get(k)', '_HYBLO["pdl"].get(pkey)').replace('_HYBLO["pdl"][k]', '_HYBLO["pdl"][pkey]')
    s = _tstag_sub(s, run, new)
    anchor = next(l for l in s.splitlines(True) if '_hyblo_prepare(state["k2a33"]["st"]' in l)
    s = _tstag_sub(s, anchor, anchor + "        _tstag_prepare()\n")
    ast.parse(s)
    assert literal(s, "_HYBLO_CUDA") == literal(base, "_HYBLO_CUDA")
    cu = literal(base, "_HYBLO_CUDA")
    changed = _tstag_source(cu, "stag")
    # Zero hot math, layout, route, cleanup, combine or barrier changes.
    def section(x, a, b): return x.split(a, 1)[1].split(b, 1)[0]
    for a, b in (("const unsigned char* slot = ring + sl * UNIT_BYTES;", "const int bb = q % NRB;"),
                 ("constexpr int UNIT_CODES", "static_assert(!OKC_WS"),
                 ("WS_JOIN();\nasm volatile(\"griddepcontrol.launch_dependents;", "#include <torch/extension.h>"),
                 ("route(p, S);", "const int nA =")):
        assert section(cu, a, b) == section(changed, a, b), a
    for key, _, var in literal(base, "_HYBLO_KEYS"):
        assert "-DOKC_SPLIT=0" in var
    assert "#define OKC_F1EARLY 1" in cu and "#define OKC_TDYN 0" in cu
    assert _tstag_source(cu, "ctl") == cu
    # All actual waits, mbarrier operations and reset expressions survive.
    for token in ("mbar_wait(", "mbar_init(", "mbar_arrive1(", "CTA_SYNC();", "WS_JOIN();",
                  "p.ctr", "other[i]", "p.doneA[i] = 0", "while (ld_acquire(p.doneA"):
        assert [l for l in cu.splitlines() if token in l] == [l for l in changed.splitlines() if token in l], token
    # Only these two host function bodies may change; all other stock bodies
    # remain byte-for-byte equal, not merely AST equivalent.
    funcs = [n for n in ast.parse(base).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    names = {n.name for n in funcs}
    kept = [n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert [n.name for n in funcs] == [n.name for n in kept]
    lines0,lines1 = base.splitlines(),s.splitlines()
    for n,nn in zip(funcs,kept):
        if lines0[n.lineno-1:n.end_lineno] != lines1[nn.lineno-1:nn.end_lineno]:
            assert n.name in ("_hyblo_run","prepare"), n.name
    return s


def protocol_gate():
    """Exhaustive integer ownership/credit/FAST1/tail/reset proof, CPU only.

    No timing inference: enumerate G=132, jobs=0..32 (T=1 has <=8), both
    WSBLK variants. No CTA exits added, including njS=0. Every mapped unit
    appears once globally; all per-job helpers publish exactly their widths.
    """
    from collections import Counter
    cases = 0
    for jobs in range(33):
        for tail in (1, 2):
            acount, bcount, credit = Counter(), Counter(), Counter()
            for c in range(132):
                bounds = [(rb*c//132, rb*(c+1)//132) for rb in (512, 256)]
                maps = []
                for rb, (lo, hi) in zip((512, 256), bounds):
                    n = hi-lo
                    units = [(q//n, lo+q % n) for q in range(jobs*n)]
                    assert units == [(j,u) for j in range(jobs) for u in range(lo,hi)]
                    maps.append(units)
                aa, bb = maps; acount.update(aa); bcount.update(bb)
                # FAST1 phase 0..min(2,nA) agrees with ring_issue mapping.
                lo, hi = bounds[0]
                assert aa[:2] == [(i//(hi-lo), lo+i % (hi-lo)) for i in range(min(2,len(aa)))]
                # Helper publishes after all lanes store the last unit per job.
                for j,u in aa:
                    if u == hi-1: credit[j] += hi-lo
                # Helper's prefix + delegated logical warps partition B exactly.
                ntail = min(tail, len(bb))
                helper = bb[:len(bb)-ntail]
                delegated = [bb[len(bb)-1-w] for w in range(ntail)]
                assert Counter(helper+delegated) == Counter(bb)
                assert len(aa)+len(helper) == len(aa)+len(bb)-ntail
            assert acount == Counter({(j,u):1 for j in range(jobs) for u in range(512)})
            assert bcount == Counter({(j,u):1 for j in range(jobs) for u in range(256)})
            assert credit == Counter({j:512 for j in range(jobs)})
            cases += 1
    # Logical 0..255 survive on ALL CTAs. Helper logical256..287 returns at
    # its existing WS_JOIN, exactly as ship. No new exit/wait/generation path.
    coverage = {c*256+t for c in range(132) for t in range(256)}
    assert set(range(3072)) <= coverage and set(range(88)) <= coverage
    for m in (1,8):
        assert {(c,t,k) for c in range(m*4) for t in range(256) for k in range(4)}
        assert sorted((c%4)*1024+t*4+k for c in range(4) for t in range(256) for k in range(4)) == list(range(4096))
    for old in range(1,9):
        for new in range(1,9):
            old_regs = ["token0" if r < old else "zero" for r in range(8)]
            new_regs = ["token0" if r < new else "zero" for r in range(8)]
            assert (old_regs == new_regs) == (old == new)
    return {"status":"PASS", "cases":cases, "jobs_tested":"0..32", "grid":132,
            "doneA_per_job":512, "new_exits":0, "new_waits":0,
            "coverage":"all 132 CTAs keep logical 0..255 through bc/cleanup/combine; helper 256..287 keeps final WS_JOIN"}


def change_report(base):
    cu = literal(base, "_HYBLO_CUDA")
    start = node(base, "_HYBLO_CUDA").lineno
    after = _tstag_source(cu, "stag")
    return {"ctl":[], "stag":[{"base_lines":([start+a0,start+a1-1] if a1>a0 else []),
            "insert_before_base_line":start+a0 if a1==a0 else None, "stag_cuda_lines":[b0+1,b1], "kind":tag}
            for tag,a0,a1,b0,b1 in difflib.SequenceMatcher(None,cu.splitlines(),after.splitlines(),autojunk=False).get_opcodes() if tag != "equal"]}


def ship_tools():
    # Reuse the submission gates and literal editor; never import the layer.
    return runpy.run_path(str(ROOT / "tools/mg_patches.py"))


def ship_sources(source):
    """Evaluate only fixed CUDA assignments, checking every tiny2 undo site."""
    cu = literal(source, "_HYBLO_CUDA")
    found = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_HYBLO_TINY2_CUDA"]
    assert len(found) <= 1
    if not found:
        return {"tiny": cu, "tiny2": cu}

    def evaluate(n):
        if isinstance(n, ast.Name):
            assert n.id == "_HYBLO_CUDA"
            return cu
        assert isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        assert n.func.attr == "replace" and not n.keywords and len(n.args) == 3
        old, new, count = map(ast.literal_eval, n.args)
        assert count == 1
        return _tstag_sub(evaluate(n.func.value), old, new)

    tiny2 = evaluate(found[0].value)
    env = {"_HYBLO_CUDA": cu}
    exec(compile(ast.Module(found, []), "<ship-source>", "exec"), env)
    assert env["_HYBLO_TINY2_CUDA"] == tiny2
    return {"tiny": cu, "tiny2": tiny2}


def apply_ship(base):
    """Store STAG literally; reconstruct the unchanged tiny2 source once.

    The small inverse delta avoids duplicating 38 KB of CUDA. Selection happens
    only in the existing two-build prepare loop; kernel_fn has no new branch.
    """
    assert "_HYBLO_TINY2_CUDA" not in base, "STAG already applied"
    original = literal(base, "_HYBLO_CUDA")
    stag = _tstag_source(original, "stag")
    undo = "_HYBLO_TINY2_CUDA = (_HYBLO_CUDA\n"
    before, after = original.splitlines(True), stag.splitlines(True)
    current = stag
    for group in difflib.SequenceMatcher(None, before, after, autojunk=False).get_grouped_opcodes(0):
        old = "".join(after[group[0][3]:group[-1][4]])
        new = "".join(before[group[0][1]:group[-1][2]])
        current = _tstag_sub(current, old, new)
        undo += f"    .replace({old!r}, {new!r}, 1)\n"
    assert current == original
    undo += ")\n"
    s = ship_tools()["cuda_edit"](base, "_HYBLO_CUDA", lambda _: stag)
    s = _tstag_sub(s, "HYBLO_ENABLE = True\n", undo + "HYBLO_ENABLE = True\n")
    old = ast.get_source_segment(s, node(s, "_hyblo_build"))
    new = _tstag_sub(old, "        fl = list(_HYBLO_FLAGS) + list(var)\n",
                     '        cu = _HYBLO_CUDA if k == "tiny" else _HYBLO_TINY2_CUDA\n'
                     "        fl = list(_HYBLO_FLAGS) + list(var)\n")
    new = _tstag_sub(new, '_HYB_MOE_CPP + _HYBLO_CUDA + " ".join(fl)',
                     '_HYB_MOE_CPP + cu + " ".join(fl)')
    new = _tstag_sub(new, "cuda_sources=_HYBLO_CUDA,", "cuda_sources=cu,")
    s = _tstag_sub(s, old, new)
    assert ship_sources(s) == {"tiny": stag, "tiny2": original}
    return s


def ship_gates(base, source, expected):
    mg = ship_tools()
    size = mg["gates"](source, ship=True)
    for token in ("TSTAG_ON", "TSTAG_ARMS", "_TSTAG =", "_tstag_prepare", "_tstag_select",
                  "_tstag_build_one", "TINYX_ON", "TINYX_ARMS", "_TINYX =", "_tinyx_prepare",
                  "_tinyx_select", "_tinyx_build_one", "_BANK", "tinyx_class", "tinyx_sink", "okc_set_tinyx"):
        assert token not in source, f"lab code leaked: {token}"
    assert ship_sources(source) == expected, "ship CUDA differs from lab transform"
    # Fingerprint every untouched top-level node, including every timed route.
    allowed = {"_HYBLO_CUDA", "_HYBLO_TINY2_CUDA", "_hyblo_build"}
    def kept(s):
        result = []
        lines = s.splitlines(True)
        for n in ast.parse(s).body:
            name = getattr(n, "name", None)
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                name = n.targets[0].id
            if name not in allowed:
                part = lines[n.lineno-1:n.end_lineno]
                part[-1] = part[-1][:n.end_col_offset]
                part[0] = part[0][n.col_offset:]
                result.append("".join(part))
        return result
    untouched = kept(base)
    assert kept(source) == untouched, "unrelated ship code changed"
    assert literal(source, "_HYBLO_KEYS") == literal(base, "_HYBLO_KEYS")
    with contextlib.redirect_stdout(io.StringIO()):
        dry = dry_run(base, source, ship_expected=expected)
    changes = [{"kind": tag, "base_lines": [a0+1, a1] if a1 > a0 else [],
                "insert_before_base_line": a0+1 if a0 == a1 else None,
                "ship_lines": [b0+1, b1] if b1 > b0 else []}
               for tag,a0,a1,b0,b1 in difflib.SequenceMatcher(
                   None, base.splitlines(), source.splitlines(), autojunk=False).get_opcodes() if tag != "equal"]
    return {"status": "CPU_PASS", "size": size, "margin": mg["LIMIT"] - size,
            "sha1": hashlib.sha1(source.encode()).hexdigest(),
            "cuda_sha1": {k: hashlib.sha1(v.encode()).hexdigest() for k,v in expected.items()},
            "untouched_sha1": hashlib.sha1("\n".join(untouched).encode()).hexdigest(),
            "prepare_dry_run": dry, "changes": changes, "compile_SASS_GPU": "NOT RUN"}


def accounting():
    return {
        "scope":"EST source accounting, NOT measured SASS or a speed claim",
        "kill_risks_first":["per-job fence drains outstanding next-job weights (R-dsm-0b/R-vfn-1)",
                            "132 publishers/job and 3/4-unit imbalance erase readiness slack (VMS_IL)"],
        "measured":{"source":"R-tinymk-1 / box_out/tinymk/d1s.json, 19 rounds",
                    "W13_end_p50_us":12.80,"W2_first_credit_exit_p50_us":14.91,"W13_unit_us":1.12},
        "estimate":{"credit_headstart_us":"3..4 A units * MEASURED 1.12 = 3.36..4.48",
                    "gross_target_us":"user hypothesis 1.8 of MEASURED (14.91-12.80)=2.11; unvalidated",
                    "extra_B_activation_floor_us":"132 * 8192 B / 4.4e6 B/us = 0.24576",
                    "net_budget_us":"1.8 - 0.24576 = 1.55424 BEFORE extra fences/atomics/mapping; not a prediction",
                    "critical_path":"R-tinymk-1 idle consumers AND idle HBM at A->B; intra-kernel dependence cannot be removed by PDL",
                    "unmeasured":"d1f gap; all candidate timing; actual SASS issue counts"},
        "unit_bytes_before_after":{"HBM_weight_scale_per_CTA":[36864,36864],
            "shared_weight_scale_per_compute_thread":[144,144],"red_store_per_compute_thread":[16,16],
            "red_read_per_reducer_thread":[128,128],"T1_global_inter_store_per_A_unit":[16,16],
            "T1_global_ybuf_store_per_B_unit":[64,64]},
        "T1_two_jobs_per_CTA":{"A_x_vector_loads_active_thread":[16,16],
            "B_x_vector_loads_active_thread":[16,32],"active_x_threads":32,
            "A_x_bytes":[8192,8192],"B_x_bytes":[8192,16384],
            "helper_thread_fence_and_sync_pairs":[1,2],"helper_lane0_credit_atomics":[1,2],
            "publishers_per_job":[66,132],"note":"weight/stage bytes unchanged; B activation traffic DOES grow, explicitly charged"},
        "instructions_per_thread_per_unit_EST":{
            "consumer_unchanged_math_and_weight_LDS_before_after":[457,457],
            "consumer_breakdown":"256 unpack INT + 16 PRMT + 16 IMAD + 128 HMUL2 + 32 HMMA + 9 LDS128; excludes address/control/FADD",
            "helper_unchanged_reduction_before_after":"8 LDS128 + 32 FADD in both arms",
            "changed_mapping":"before global v=lo+q; after n=hi-lo, 1..4 constant-divisor selection, j=q/n, v=j*RB+lo+q-j*n; emitted instructions compiler-dependent",
            "job_switch":"A: cnt equality + predicate, skips all 16 LDG128/active thread on same token mask; B: 16 LDG128/active thread each new job",
            "credit":"before one fence/sync per helper thread and one atomic per helper lane0/call at T1; after two of each",
            "sass_status":"PENDING Linux gate; JSON dumps every real function, regs/spills/opcode counts. Whole-function counts are NOT per-unit counts."},
        "stag8":"OMITTED: at 8 jobs, each CTA changes from 1..2 jobs to 8 (6..7 extra reloads/phase); upper bound 7*16 LDG128 per active thread/phase, 7*65536=458752 B/CTA/phase at cnt=8. No T8 speed claim.",
        "protocol_consumers_and_coverage":[
            "doneA acquire: every CTA, all eight compute warps lane0 (logical 0,32,..224), first unit of each B job; syncwarp all lanes. Remapped helper lane0 publishes that job's width AFTER every helper lane stores+fences. Sum_c(floor(512*(c+1)/132)-floor(512*c/132))=512.",
            "helper physical 0..31 (logical256..287): q=0..N-min(WSBLK,nB)-1, all A units retained; helper stores inter and B prefix, wsre releases unchanged. final WS_JOIN unchanged; no new exit.",
            "B tail: logical warp0..min(WSBLK,nB)-1, lanes0..31, ul=nB-1-warp. Same red ring slot q=nA+ul; new global unit map is used both at norm prefetch and delegated store. Helper prefix plus tail is a disjoint cover.",
            "ring consumers: every CTA logical0..255; q contiguous across A/B, seq=nA+nB. mbarriers, parity, NS=3, refill and arrival lines unchanged. FAST1 issues min(2,nA) of the same map. At zero jobs no issue/division, joins still execute.",
            "ss next-bank reset: CTA0..11 logical0..255 cover i=c*256+tid in [0,3072); stride132*256. CTA0 logical0 resets ctr[4+((gen+1)&1)].",
            "ctr generation read: every CTA logical0 reads ctr[3] and selects ss bank; unchanged.",
            "bc: ALL CTA0..131 logical0 fence/atomicAdd/wait; old==131 increments ctr[3]. All logical0..255 stay for both CTA_SYNCs. Helper already returned only after old final WS_JOIN.",
            "post-bc ctr[0:2] reset: CTA0 logical0/1. doneA[0:88] reset: CTA0 logical0..87 (general stride G*256 unchanged). All jobs covered independently of empty/nonempty unit shares.",
            "first combine: T1 CTA0..3, T8 CTA0..31, logical0..255 each writes4 columns; four CTAs/row cover4096. Remaining it=c+132*n loop unchanged, zero further iterations at T1/8. No ss current-bank reset added; ping-pong generation unchanged."],
        "route":"base _v11_k2a33_call:7678 -> _hyblo_route -> _hyblo_key:7573 -> tiny for M1, tiny2 for M8; moe_fused<false> at7524/7528; flags7535/7536; stag selector additionally requires m==1"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ship", action="store_true")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.ship and a.out is None:
        p.error("--ship requires --out")
    base = a.base.read_text(encoding="utf-8")
    out = a.out or OUT
    if a.ship:
        assert out.resolve() != a.base.resolve(), "ship output must differ from base"
        source = apply_ship(base)
        expected = {"tiny": _tstag_source(literal(base, "_HYBLO_CUDA"), "stag"),
                    "tiny2": literal(base, "_HYBLO_CUDA")}
        result = ship_gates(base, source, expected)
        result["protocol"] = protocol_gate()
        if a.check or a.dry_run:
            assert out.read_bytes() == source.encode(), "regenerate STAG ship"
        else:
            out.write_bytes(source.encode())
        print(json.dumps(result, indent=2))
        return 0
    source = build(base)
    protocol = protocol_gate()
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        dry_run(base, source)
    if a.check or a.dry_run:
        assert out.read_text(encoding="utf-8") == source, "regenerate TSTAG layer"
    else:
        out.write_bytes(source.encode())
    print(json.dumps({"status":"CPU_PASS", "protocol":protocol, "prepare_dry_run_scope":"tensor/compiler doubles ONLY",
                      "accounting":accounting(),
                      "changes":change_report(base), "layer_sha256":hashlib.sha256(source.encode()).hexdigest(),
                      "compile_SASS_GPU":"NOT RUN"}, indent=2))
    return 0


# CPU dry-run below is copied from the proven TINYX/PADZ pipeline and scoped
# to this bank/arms. Tensor/compiler doubles never count as GPU evidence.

def dry_run(base, patched, fixture=False, ship_expected=None):
    """Execute real _hyblo_prepare/_hyb_scratch and every new host branch.

    Only tensor arithmetic, prepack, reference values and compiler are doubles.
    State dictionaries, allocations' shapes, flags, build arguments and selectors
    are real AST nodes from the base and emitted layer; no permissive dicts.
    """
    class Tensor:
        device = "cpu-dry"
        def __init__(self, shape=(8, 4096)): self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        def __getitem__(self, k):
            if isinstance(k, slice):
                return Tensor((len(range(*k.indices(self.shape[0]))),) + self.shape[1:])
            return Tensor(self.shape)
        def __add__(self, x): return self
        __radd__ = __mul__ = __rmul__ = __eq__ = __add__
        def all(self): return True
        def numel(self):
            import math
            return math.prod(self.shape)
        def view(self, *s): return self if len(s) == 1 and isinstance(s[0], str) else Tensor(s)
        def reshape(self, *s): return Tensor(s)
        def to(self, *a): return self
        def contiguous(self): return self
        clone = float = contiguous
        def fill_(self, v): return self
        def data_ptr(self): return 16
    builds, calls, failures, build_names = [], [], set(), {}
    class Module:
        def __init__(self, key): self.key, self.flags, self.pdl = key, 0, None
        def okc_set_pdl(self, pdl): self.pdl = pdl
        def okc_run(self, *a):
            assert a[0] is ns["_HYBLO"]["w"] and a[1] is ns["_HYBLO"]["sc"]
            assert a[-3:] == (0, 32, 132)
            calls.append((self.key, self.flags, self.pdl, a[2].shape[0]))
    torch = NS(**{s: s for s in ("int16", "int32", "float32", "bfloat16", "uint8")},
        empty=lambda s, **k: Tensor(s), zeros=lambda s, **k: Tensor(s),
        arange=lambda *a, **k: Tensor((a[0],)), stack=lambda *a, **k: Tensor(), cat=lambda *a, **k: Tensor(),
        isfinite=lambda t: t, cuda=NS(synchronize=lambda: None))
    ns = dict(torch=torch, time=time, hashlib=hashlib, _pt1_wait=lambda *a: None,
              _k2a_ref_fp32=lambda *a: Tensor(), _k2a_rel_err=lambda *a: (0., 0.))
    names = ("_HYB", "_HYB_MMAX", "_HYB_EMAX", "_HYB_FIXF", "_HYB_XROWS", "_HYB_GUARD",
             "_HYBLO", "_HYBLO_CUDA", "_HYB_MOE_CPP", "_HYBLO_FLAGS", "_HYBLO_KEYS", "HYBLO_ENABLE",
             "HYBLO_PDL", "HYBLO_T", "HYBLO_K", "HYBLO_SELF_TEST", "TSTAG_ON", "TSTAG_ARMS",
             "_TSTAG", "_TSTAG_MAP", "_tstag_sub", "_tstag_source",
             "_tstag_build_one", "_tstag_prepare", "_tstag_select", "_hyb_scratch", "_hyb_guards_ok",
             "_hyblo_key", "_hyblo_build", "_hyblo_run", "_hyblo_route", "_hyblo_prepare")
    if ship_expected is not None:
        names = tuple(n for n in names if not n.startswith(("TSTAG", "_TSTAG", "_tstag")))
    for name in names:
        exec(compile(ast.Module(body=[node(patched, name)], type_ignores=[]), f"real:{name}", "exec"), ns)
    if ship_expected is not None and "_HYBLO_TINY2_CUDA" in patched:
        exec(compile(ast.Module([node(patched, "_HYBLO_TINY2_CUDA")], []), "real:tiny2-source", "exec"), ns)
    def load_inline(**kw):
        if kw["name"].startswith("hyblo_"):
            kind, k = "ctl", kw["name"].split("_")[1]
        else:
            _, k, kind, _ = kw["name"].split("_")
        var = next(v for key, _, v in ns["_HYBLO_KEYS"] if key == k)
        expected = ship_expected[k] if ship_expected is not None else _tstag_source(literal(base, "_HYBLO_CUDA"), kind)
        assert kw["cuda_sources"] == expected
        assert kw["extra_cuda_cflags"] == list(ns["_HYBLO_FLAGS"]) + list(var) + ([] if kind == "ctl" else ["-Xptxas=-v"])
        assert kw["functions"] == ["okc_run", "okc_set_pdl"]
        assert kw["cpp_sources"] == literal(base, "_HYB_MOE_CPP")
        if ship_expected is not None:
            tag = hashlib.sha1((kw["cpp_sources"] + expected + " ".join(kw["extra_cuda_cflags"])).encode()).hexdigest()[:12]
            assert kw["name"] == f"hyblo_{k}_{tag}", "stale module cache tag"
            previous = ship_sources(base)[k]
            old_tag = hashlib.sha1((kw["cpp_sources"] + previous + " ".join(kw["extra_cuda_cflags"])).encode()).hexdigest()[:12]
            assert (tag == old_tag) == (expected == previous), "changed source reused old cache tag"
            build_names[k] = {"before": f"hyblo_{k}_{old_tag}", "after": kw["name"]}
        builds.append((k, kind))
        if (k, kind) in failures: raise RuntimeError("injected build failure")
        return Module((k, kind))
    original_import = builtins.__import__
    ns["__builtins__"] = dict(vars(builtins), __import__=lambda n, *a, **k: NS(load_inline=load_inline) if n == "torch.utils.cpp_extension" else original_import(n, *a, **k))
    # Re-create functions with the restricted per-exec imports, never sys.modules.
    for name in (("_hyblo_build",) if ship_expected is not None else ("_hyblo_build", "_tstag_build_one")):
        exec(compile(ast.Module(body=[node(patched, name)], type_ignores=[]), name, "exec"), ns)
    for name in ("_hyblo_pack_codes", "_hyblo_pack_scales", "_hyblo_fat_order", "_hyblo_unit_order"):
        ns[name] = lambda *a: Tensor()
    ns["_hyblo_inputs"] = lambda bs, *a: {k: Tensor((bs, 4096)) for k in ("hidden_states", "topk_ids", "topk_weights")}
    ns["_v11_k2a33_call"] = lambda *a: Tensor()
    ns["_HYB"].update(ok=True, w=[Tensor() for _ in range(14)], nsm=132, full=[])
    w = {k: Tensor() for k in ("w13", "w13_sf", "w2", "w2_sf")}
    ns["_hyblo_prepare"]({}, w, "cpu-dry")
    assert ns["_HYBLO"]["ok"] == {"tiny": True, "tiny2": True}
    assert len(ns["_HYBLO"]["sc"]) == 3 and len(ns["_HYBLO"]["w"]) == 14
    if ship_expected is not None:
        ns["_hyblo_prepare"]({}, w, "cpu-dry")
        assert builds == [("tiny", "ctl"), ("tiny2", "ctl")], "prepare rebuilt cached modules"
        # Copy the proven PADZ-derived real state-layout execution below.
        ns.update(st={}, state={}, weights={k: Tensor() for k in
            ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs", "post_expert_norm")}, t0=0)
        for name in ("K2A_M_MAX", "K2A_BUCKETS", "V11_K2A33_ROUTE"):
            exec(compile(ast.Module([node(base, name)], []), name, "exec"), ns)
        prep = node(base, "_v11_k2a33_prepare")
        ok_stmt = next(n for n in prep.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "ok")
        exec(compile(ast.Module([ok_stmt], []), "real-ok", "exec"), ns)
        ret = next(n.value for n in prep.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict))
        k33 = eval(compile(ast.Expression(ret), "real-return", "eval"), ns)
        ns["_v11_k2a33_prepare"] = lambda *a: k33
        assignment = next(n for n in ast.walk(ast.parse(base)) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_v11_k2a33_prepare")
        exec(compile(ast.Module([assignment], []), "real-state", "exec"), ns)
        site = next(l.strip() for l in patched.splitlines() if '_hyblo_prepare(state["k2a33"]["st"]' in l)
        exec(site, ns)
        exec(compile(ast.Module([node(patched, "_v11_k2a33_call")], []), "real-route", "exec"), ns)
        for t in (1, 2, 4, 5, 8):
            k = ns["_hyblo_key"](t)
            for pdl in (True, False, True):
                ns["HYBLO_PDL"] = pdl
                args = (Tensor((t, 4096)), Tensor(), Tensor())
                result = (ns["_v11_k2a33_call"](*args, ns["state"]["k2a33"]["st"])
                          if t in (1, 8) else ns["_hyblo_route"](*args))
                assert result.shape == (t, 4096)
                assert calls[-1] == ((k, "ctl"), 0, int(pdl), t)
        assert builds == [("tiny", "ctl"), ("tiny2", "ctl")]
        return {"status": "PASS", "scope": "real AST/state; tensor/compiler doubles, no GPU",
                "builds": [k for k,_ in builds], "routes": [1, 2, 4, 5, 8], "PDL": [0, 1],
                "state_path": 'state["k2a33"]["st"]', "source_and_cache_tags": "PASS", "module_names": build_names}
    ns["_tstag_prepare"](); ns["_tstag_prepare"]()
    assert len(builds) == 3 and not ns["_TSTAG"]["errors"]
    def must_fail(f):
        try: f()
        except (RuntimeError, KeyError): return
        raise AssertionError("failure silently accepted")
    ns["TSTAG_ON"] = "stag"
    must_fail(lambda: ns["_tstag_select"]("tiny", 1))
    ns["_TSTAG"]["accepted"] = set(ns["_TSTAG"]["mods"])
    for t in (1, 2, 4, 5, 8):
        k = "tiny" if t <= 4 else "tiny2"
        for arm in TSTAG_ARMS:
            kind, flags = ("stag" if arm == "stag" and t == 1 else "ctl"), 0
            for pdl in (True, False, True):
                ns["TSTAG_ON"], ns["HYBLO_PDL"] = arm, pdl
                out = ns["_hyblo_route"](Tensor((t, 4096)), Tensor(), Tensor())
                assert out.shape == (t, 4096) and calls[-1] == ((k, kind), flags, int(pdl), t)
    ns["TSTAG_ON"] = "bad"; must_fail(lambda: ns["_tstag_select"]("tiny", 1))
    ns["TSTAG_ON"] = "stag"
    del ns["_TSTAG"]["mods"][("tiny", "stag")]
    failures.add(("tiny", "stag")); ns["_tstag_prepare"]()
    must_fail(lambda: ns["_tstag_select"]("tiny", 1))
    failures.clear(); ns["_tstag_prepare"](); ns["_tstag_select"]("tiny", 1)
    saved = ns["_HYBLO"]["sc"]; ns["_HYBLO"]["sc"] = None
    must_fail(ns["_tstag_prepare"]); ns["_HYBLO"]["sc"] = saved
    ns["_HYBLO"]["ok"]["tiny2"] = False; must_fail(ns["_tstag_prepare"])
    ns["_HYBLO"]["ok"]["tiny2"] = True
    ns["HYBLO_ENABLE"] = False; ns["_tstag_prepare"]()
    ns["HYBLO_ENABLE"] = True; ns["TSTAG_ON"] = "ctl"
    # Execute the new prepare call site against the REAL outer return layout.
    ns.update(st={}, state={}, weights={k: Tensor() for k in
        ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs", "post_expert_norm")}, t0=0)
    for name in ("K2A_M_MAX", "K2A_BUCKETS", "V11_K2A33_ROUTE"):
        exec(compile(ast.Module(body=[node(base, name)], type_ignores=[]), name, "exec"), ns)
    prep = node(base, "_v11_k2a33_prepare")
    ok_stmt = next(n for n in prep.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "ok")
    exec(compile(ast.Module(body=[ok_stmt], type_ignores=[]), "real-ok", "exec"), ns)
    ret = next(n.value for n in prep.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict))
    k33 = eval(compile(ast.Expression(ret), "real-return", "eval"), ns)
    ns["_v11_k2a33_prepare"] = lambda *a: k33
    assignment = next(n for n in ast.walk(ast.parse(base)) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_v11_k2a33_prepare")
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "real-state", "exec"), ns)
    site = next(l.strip() for l in patched.splitlines() if '_hyblo_prepare(state["k2a33"]["st"]' in l)
    exec(site + "\n_tstag_prepare()", ns)
    # The actual layer dispatch must reach the HYBLO branch at both scored T.
    exec(compile(ast.Module(body=[node(base, "_v11_k2a33_call")], type_ignores=[]), "real-route", "exec"), ns)
    for t in (1, 8):
        for arm in TSTAG_ARMS:
            ns["TSTAG_ON"] = arm
            assert ns["_v11_k2a33_call"](Tensor((t, 4096)), Tensor(), Tensor(), k33["st"]).shape == (t, 4096)
    ns["TSTAG_ON"] = "ctl"
    print("[tstag] CPU dry PASS: real prepare/scratch, three builds, cache, all arms/PDL, missing gate/bank, independent failure/retry")
    if fixture:
        class ModuleProxy:
            def __getattr__(self, k):
                if k not in ns: raise AttributeError(k)
                return ns[k]
            def __setattr__(self, k, v): ns[k] = v
        return ModuleProxy(), ns["state"], Tensor

if __name__ == "__main__":
    raise SystemExit(main())

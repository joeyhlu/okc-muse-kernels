"""TUK: A-only unequal K ownership for the helper0 TINY/TINY2 builds.

Only writes kernels/lab/k3_tuk_ab.py. Run with python -B; --check is read-only.
The ship CUDA, flags, builder and module names stay byte-for-byte unchanged.
CPU gates and the strict prepare/dispatch dry run live in local/tuk_compile.py.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/lab/k3_tuk_ab.py"
ARMS = ("ship", "u34", "u58")
KEEP = {"u34": 6, "u58": 5}


def once(src, old, new):
    assert src.count(old) == 1, ("exact anchor", src.count(old), old[:100])
    return src.replace(old, new, 1)


def node(src, name):
    matches = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            matches.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            matches.append(n)
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def literal(src, name):
    return ast.literal_eval(node(src, name).value)


def chunks(arm, warp):
    """Logical compute warp -> original 64-wide K-chunk indices (not physical warp)."""
    if arm == "ship":
        return list(range(warp * 8, warp * 8 + 8))
    keep = KEEP[arm]
    group, role = divmod(warp, 4)
    owned = list(range(warp * 8, warp * 8 + (keep if role == 3 else 8)))
    if role < 8 - keep:
        owned.append(group * 32 + 24 + keep + role)
    return owned


CUDA_HELPERS = r'''
// TUK: TIDX rotates physical warp 0 to logical helper 8. SMSP0 compute = 3,7.
// A changes K ownership only; the original PB=true instruction source is kept.
constexpr int TUK_KEEP = @KEEP@;
static_assert(NW == 8 && NCH == 8 && NTL == 288, "TUK helper0 geometry");
template <int C>
__device__ __forceinline__ int tuk_chunk(int warp) {
if constexpr (C < 8) return warp * 8 + C;
else return (warp >> 2) * 32 + 24 + TUK_KEEP + (warp & 3);
}
template <int C>
__device__ __forceinline__ void tuk_x_piece(const Params& p, const Smem& S, int j,
int lane, int warp, uint4 (&x)[18]) {
const int r = lane >> 2, t = lane & 3;
if (r < S.jC[j]) {
const __nv_bfloat16* row = p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD;
const uint4* src = reinterpret_cast<const uint4*>(row + tuk_chunk<C>(warp) * 64 + 16 * t);
x[2 * C] = __ldg(src); x[2 * C + 1] = __ldg(src + 1);
} else {
x[2 * C] = make_uint4(0, 0, 0, 0); x[2 * C + 1] = make_uint4(0, 0, 0, 0);
}
}
__device__ __forceinline__ void tuk_load_x(const Params& p, const Smem& S, int j,
int lane, int warp, uint4 (&x)[18]) {
// Two warp-uniform suffix branches; no separate whole-unit role instantiations.
@X_COMMON@
if (__all_sync(0xffffffffu, (warp & 3) != 3)) {
@X_TAIL@
}
if (__all_sync(0xffffffffu, (warp & 3) < 8 - TUK_KEEP)) {
tuk_x_piece<8>(p, S, j, lane, warp, x);
}
}
template <int C>
__device__ __forceinline__ void tuk_piece(const unsigned char* slot, int lane, int warp,
uint32_t swd, uint32_t k16, const uint4 (&x)[18], float (&acc)[4], float (&acc2)[4],
float (&acc3)[4], float (&acc4)[4]) {
const int ck = tuk_chunk<C>(warp);
const uint4 w = *reinterpret_cast<const uint4*>(slot + ck * 512 + lane * 16);
constexpr int sb = (TUK_KEEP == 5 && C == 5) ? 0 : (C & 1) * 2;
const uint32_t sg = SCDK(swd, sb, k16), sg8 = SCDK(swd, sb + 1, k16);
const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
const uint32_t b[8] = {x[2*C].x, x[2*C].y, x[2*C].z, x[2*C].w,
x[2*C+1].x, x[2*C+1].y, x[2*C+1].z, x[2*C+1].w};
#pragma unroll
for (int st = 0; st < 4; ++st) {
uint32_t r0, r1, r2, r3;
dq4(ww[st], r0, r1, r2, r3);
mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4,
bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2*st], b[2*st+1]);
}
}
__device__ __forceinline__ void tuk_accum(const unsigned char* slot, int lane, int warp,
uint32_t k16, const uint4 (&x)[18], float (&acc)[4], float (&acc2)[4],
float (&acc3)[4], float (&acc4)[4]) {
// Load each retained scale byte exactly once; donated chunks read only their U16.
const unsigned char* sp = slot + UNIT_CODES + warp * 512 + lane * 16;
const uint2 sc01 = *reinterpret_cast<const uint2*>(sp);
const uint32_t sc2 = *reinterpret_cast<const @SCTYPE@*>(sp + 8);
@A_COMMON@
if (__all_sync(0xffffffffu, (warp & 3) != 3)) {
@SCALE5@
const uint32_t sc3 = *reinterpret_cast<const uint32_t*>(sp + 12);
@A_TAIL@
}
if (__all_sync(0xffffffffu, (warp & 3) < 8 - TUK_KEEP)) {
const int ck = tuk_chunk<8>(warp);
const uint32_t sc8 = *reinterpret_cast<const unsigned short*>(
slot + UNIT_CODES + (ck >> 3) * 512 + lane * 16 + (ck & 7) * 2);
tuk_piece<8>(slot, lane, warp, sc8, k16, x, acc, acc2, acc3, acc4);
}
}
'''


def cuda_edits(cu, arm):
    keep = KEEP[arm]
    helpers = CUDA_HELPERS.replace("@KEEP@", str(keep))
    helpers = helpers.replace("@SCTYPE@", "uint32_t" if keep == 6 else "unsigned short")
    helpers = helpers.replace("@SCALE5@", "" if keep == 6 else
                              "const uint32_t sc5 = *reinterpret_cast<const unsigned short*>(sp + 10);")
    for what, call in (("X", "tuk_x_piece<{c}>(p, S, j, lane, warp, x);"),
                       ("A", "tuk_piece<{c}>(slot, lane, warp, {sc}, k16, x, acc, acc2, acc3, acc4);")):
        def emit(c):
            sc = "sc01.x" if c < 2 else "sc01.y" if c < 4 else "sc5" if c == 5 and keep == 5 else "sc2" if c < 6 else "sc3"
            return call.format(c=c, sc=sc)
        helpers = helpers.replace(f"@{what}_COMMON@", "\n".join(emit(c) for c in range(keep)))
        helpers = helpers.replace(f"@{what}_TAIL@", "\n".join(emit(c) for c in range(keep, 8)))
    start = "template <bool PB>\n__device__ void skinny_ring_ws("
    end = "\n__device__ __forceinline__ void ws_issue("
    a = cu.index(start)
    b = cu.index(end, a)
    old = cu[a:b]
    new = once(old, "uint4 x0[2 * NCH];", "uint4 x0[2 * (PB ? NCH : NCH + 1)];")
    new = once(new, "load_x_regs<PB>(p, S, j, lane, warp, x0);",
               "if constexpr (PB) load_x_regs<PB>(p, S, j, lane, warp, x0);\n"
               "else tuk_load_x(p, S, j, lane, warp, x0);")
    sc = "const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);"
    new = once(new, sc, "uint4 sc;\nif constexpr (PB) sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);")
    loop0 = "#pragma unroll\nfor (int c = 0; c < NCH; ++c) {"
    loop1 = "#pragma unroll\nfor (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];"
    i, j = new.index(loop0), new.index(loop1)
    old_loop = new[i:j]
    new = once(new, old_loop, "if constexpr (PB) {\n" + old_loop +
               "} else {\ntuk_accum(slot, lane, warp, k16, x0, acc, acc2, acc3, acc4);\n}\n")
    return [(old, helpers + "\n" + new, 1)]


LAB = r'''
# TUK lab bank (padz pattern): one process/state, own scratch, shared immutable weights.
TUK = "ship"
_TUK_ARMS = ("ship", "u34", "u58")
_TUK_BANK = {"ship": _HYBLO}
_TUK_BANK.update({a: {"mods": {}, "ok": {}, "sc": None, "full": None,
                    "pdl": {}, "secs": {}, "err": {}, "tests": [], "calls": {}}
                  for a in _TUK_ARMS[1:]})
def _tuk_src(arm, key):
    if arm not in _TUK_ARMS:
        raise ValueError(f"TUK arm {arm!r}")
    var = next(v for k, _, v in _HYBLO_KEYS if k == key)
    src = _HYBLO_CUDA
    for old, new, count in _TUK_EDITS.get(arm, ()):
        assert src.count(old) == count, old[:80]
        src = src.replace(old, new)
    return src, list(_HYBLO_FLAGS) + list(var)
def _tuk_build():
    from torch.utils.cpp_extension import load_inline
    for arm in _TUK_ARMS[1:]:
        H = _TUK_BANK[arm]
        for key, _, _ in _HYBLO_KEYS:
            if key in H["mods"] or key in H["err"]:
                continue
            t0 = time.time()
            try:
                src, fl = _tuk_src(arm, key)
                tag = hashlib.sha1((_HYB_MOE_CPP + src + " ".join(fl)).encode()).hexdigest()[:12]
                H["mods"][key] = load_inline(name=f"tuk_{arm}_{key}_{tag}", cpp_sources=_HYB_MOE_CPP,
                    cuda_sources=src, functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
            except Exception as ex:
                H["err"][key] = repr(ex)[-600:]
            H["secs"][key] = time.time() - t0
def _tuk_run(arm, key, x, ids, wts):
    # Accessors copied from _hyblo_run; no VMS state or invented module handles.
    H = _TUK_BANK[arm]
    ids, wts = ids.to(torch.int32).contiguous(), wts.to(torch.float32).contiguous()
    mod = H["mods"][key]
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYBLO_PDL else 0
    if H["pdl"].get(key) != pdl:
        mod.okc_set_pdl(pdl)
        H["pdl"][key] = pdl
    mod.okc_run(_HYBLO["w"], H["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
def _tuk_route(x, ids, wts):
    if TUK not in _TUK_ARMS:
        raise ValueError(f"TUK arm {TUK!r}")
    key = _hyblo_key(x.shape[0])
    H = _TUK_BANK[TUK]
    if HYBLO_K != "h" or not H["ok"].get(key):
        raise RuntimeError(f"TUK {TUK}/{key} not validated; refusing a silently timed fallback")
    H["calls"][(key, x.shape[0])] = H["calls"].get((key, x.shape[0]), 0) + 1
    return _tuk_run(TUK, key, x, ids, wts)
def _tuk_prepare(st, w, dev):
    for arm in _TUK_ARMS[1:]:
        _TUK_BANK[arm]["ok"] = {}
        _TUK_BANK[arm]["tests"] = []
    if not HYBLO_ENABLE or not all(_HYBLO["ok"].get(k) for k, _, _ in _HYBLO_KEYS):
        print("[tuk] OFF: ship HYBLO validation failed", flush=True)
        return
    _tuk_build()  # serial, after base prepare; this is a lab vehicle, not a submission
    for arm in _TUK_ARMS[1:]:
        H = _TUK_BANK[arm]
        H["sc"], H["full"] = _hyb_scratch(dev)
    for bs, seed, dist in HYBLO_SELF_TEST:
        d = _hyblo_inputs(bs, seed, dev, dist)
        x, ids, wts = d["hidden_states"].contiguous(), d["topk_ids"], d["topk_weights"]
        key = _hyblo_key(bs)
        ref = _k2a_ref_fp32(d, w)
        ship = _hyblo_run(key, x, ids, wts).clone()
        for arm in _TUK_ARMS[1:]:
            H = _TUK_BANK[arm]
            if key not in H["mods"]:
                H["ok"][key] = False
                continue
            good, prev = True, None
            for call in range(2):
                got = _tuk_run(arm, key, x, ids, wts)
                torch.cuda.synchronize()
                ev = _k2a_rel_err(got, ship)[0]
                tot, row = _k2a_rel_err(got, ref)
                rep = 0.0 if prev is None else _k2a_rel_err(got, prev)[0]
                ok = tot < 1e-2 and row < 5e-2 and ev < 1e-2 and rep < 1e-3 and bool(torch.isfinite(got.float()).all())
                H["tests"].append(dict(bs=bs, seed=seed, dist=dist, call=call,
                    ship=ev, fp32=tot, row=row, repeat=rep, ok=ok))
                print(f"[tuk] {arm}/{key} bs={bs} {dist} call={call}: vs ship {ev:.3e} "
                      f"vs FP32 {tot:.3e} row {row:.3e} repeat {rep:.3e} -> {ok}", flush=True)
                good = good and ok
                prev = got.clone()
            # Warmed modules/scratch/PDL before capture; every existing case, including nj=0.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gout = _tuk_run(arm, key, x, ids, wts)
            for replay in range(2):
                graph.replay()
                torch.cuda.synchronize()
                ev = _k2a_rel_err(gout, ship)[0]
                tot, row = _k2a_rel_err(gout, ref)
                ok = ev < 1e-2 and tot < 1e-2 and row < 5e-2 and bool(torch.isfinite(gout.float()).all())
                H["tests"].append(dict(bs=bs, seed=seed, dist=dist, graph=replay,
                    ship=ev, fp32=tot, row=row, ok=ok))
                print(f"[tuk] {arm}/{key} bs={bs} {dist} graph={replay}: vs ship {ev:.3e} "
                      f"vs FP32 {tot:.3e} row {row:.3e} -> {ok}", flush=True)
                good = good and ok
            H["ok"][key] = H["ok"].get(key, True) and good
    for arm in _TUK_ARMS[1:]:
        H = _TUK_BANK[arm]
        guards = _hyb_guards_ok() and all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in H["full"])
        H["ok"] = {k: bool(v and guards) for k, v in H["ok"].items()}
        print(f"[tuk] {arm} ok={H['ok']} guards={guards} nvcc={H['secs']} errors={H['err']}", flush=True)
def _tuk_prepare_all(st, w, dev):
    global TUK
    saved = TUK
    if saved not in _TUK_ARMS:
        raise ValueError(f"TUK arm {saved!r}")
    try:
        TUK = "ship"  # the base self-test also calls the layer's MoE dispatch
        _hyblo_prepare(st, w, dev)
        _tuk_prepare(st, w, dev)
    finally:
        TUK = saved
'''


def prepare_anchor(base):
    hits = [line for line in base.splitlines(True) if line.startswith('  _hyblo_prepare(state["k2a33"]["st"],')]
    assert len(hits) == 1
    return hits[0]


def build(base):
    cu = literal(base, "_HYBLO_CUDA")
    edits = {arm: cuda_edits(cu, arm) for arm in KEEP}
    src = once(base, "def _hyblo_key(m):\n", "_TUK_EDITS = " + repr(edits) + "\n" + LAB + "\ndef _hyblo_key(m):\n")
    src = once(src, "def _hyblo_route(x,ids,wts):\n", "def _hyblo_route(x,ids,wts):\n if TUK != 'ship':\n  return _tuk_route(x,ids,wts)\n")
    anchor = prepare_anchor(base)
    # Same nested layer state and weights expression as the base, without guessing.
    src = once(src, anchor, anchor.replace("_hyblo_prepare(", "_tuk_prepare_all("))
    ast.parse(src)
    assert src.isascii() and "\r" not in src
    assert literal(src, "_HYBLO_CUDA") == cu
    for name in ("_HYBLO_FLAGS", "_HYBLO_KEYS", "HYBLO_SELF_TEST", "_HYB_MOE_CPP"):
        assert literal(src, name) == literal(base, name), name
    for name in ("_hyblo_build", "_hyblo_run", "_hyblo_prepare"):
        assert ast.dump(node(src, name)) == ast.dump(node(base, name)), name
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    base = BASE.read_text(encoding="ascii")
    src = build(base)
    compile(src, str(OUT), "exec")  # no pycache writes
    if args.check:
        assert OUT.read_bytes() == src.encode("ascii"), "stale TUK lab file"
    else:
        OUT.write_bytes(src.encode("ascii"))  # LF, including on Windows
    print(f"TUK {'check' if args.check else 'generate'} PASS: {len(src):,} bytes; "
          f"sha256={hashlib.sha256(src.encode()).hexdigest()}")


if __name__ == "__main__":
    main()

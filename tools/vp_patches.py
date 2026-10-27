"""vp_patches.py: VP = the bf16_proj vp_2 x ring (kernels/muse_bf16_proj/vp_2.py, notes/study/vp_proj.md) in the layer's GD
GEMVs (tools/gd_patches.py's gd_kernel), stacked on k3_vms_1k (tools/gen_k3_vms.py vms1_text(kl1=True)).

XP >= 2 in a GD route = each warp's x fragments of the next XP - 1 units are in flight in a per-warp shared-memory ring
(cp.async 16 B per lane, issued after griddepcontrol.wait like GD's x loads; each lane reads back only what it copied, so
only cp.async.wait_group orders it); XP = 0 / 1 are gd_kernel exactly. The ring is dynamic smem (NW x XP x MT KB per CTA).
GD_VP_ROUTE replaces GD_ROUTE entries; each falls back to the GD route it replaced (GD_VP_FALLBACK) if its build / pack / self-test
(GD's own: bit-exact unpack, judge recipe at every token class of the bucket, PDL + trigger, bitwise repeat) fails.
GD_VP_ENABLE False = k3_vms_1k's GD routes exactly.

    python -I tools/vp_patches.py [--n=1] [--check]   writes kernels/muse_decode_layer/k3_vp_<n>.py + kernels/lab/vp_k3_<n>_ab.py
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

CUDA_EDITS = [
    ("cp.async helpers", "__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n",
     "__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n"
     "__device__ __forceinline__ void cpa16(uint32_t d, const void* s) {\n"
     "  asm volatile(\"cp.async.ca.shared.global [%0], [%1], 16;\\n\" :: \"r\"(d), \"l\"(s) : \"memory\");\n}\n"
     "__device__ __forceinline__ void cpa_commit() { asm volatile(\"cp.async.commit_group;\\n\" ::: \"memory\"); }\n"
     "template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile(\"cp.async.wait_group %0;\\n\" :: \"n\"(N) : \"memory\"); }\n"),
    ("ring depth", "  constexpr int XB = XP ? 2 : 1;\n",
     "  constexpr int XB = XP == 1 ? 2 : 1;\n  constexpr int XR = XP >= 2 ? XP : 0;   // VP: x ring depth (units) per warp\n"
     "  extern __shared__ __align__(16) uint4 vp_xring[];   // [NW][XR][MT][2][32 lanes]\n"),
    ("ring prologue", """  uint4 X[XB][MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    X[0][mt][0] = ldx(xb + xo[mt]);
    X[0][mt][1] = ldx(xb + xo[mt] + 1);
  }
""", """  uint4 X[XB][MT][2];
  uint4* xr = vp_xring + (size_t)(threadIdx.x >> 5) * XR * MT * 64 + (threadIdx.x & 31);
  auto xissue = [&](int su) {   // VP: this lane's x of unit su -> ring slot su % XR, one commit group per unit
    if (su < KU) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) cpa16(smem_u32(xr + (((su % (XR ? XR : 1)) * MT + mt) * 2 + h) * 32), xb + xo[mt] + su * 8 + h);
    }
    cpa_commit();
  };
  if constexpr (XR > 0) {
#pragma unroll
    for (int i = 0; i < XR - 1; ++i) xissue(i);
  } else {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      X[0][mt][0] = ldx(xb + xo[mt]);
      X[0][mt][1] = ldx(xb + xo[mt] + 1);
    }
  }
"""),
    ("ring use", """    const int xs = XP ? (s & 1) : 0;
    if (XP) {""", """    const int xs = XP == 1 ? (s & 1) : 0;
    if constexpr (XR > 0) {
      xissue(s + XR - 1);
      cpa_wait<XR - 1>();
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) X[0][mt][h] = xr[(((s % XR) * MT + mt) * 2 + h) * 32];
    } else if (XP == 1) {"""),
    ("launch smem", """  cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];""", """  cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);
  constexpr int dyn = WR * WK * (XP >= 2 ? XP : 0) * MT * 1024;   // VP x rings
  static int vp_set = 0;   // first call = prepare()'s info / self-test, outside any capture
  if (dyn > 0 && !vp_set) {
    const int e = (int)cudaFuncSetAttribute(gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
    if (e) return e;
    vp_set = 1;
  }
  cfg.dynamicSmemBytes = (size_t)dyn;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];"""),
    ("info smem", """  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP>;
  int e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, 0);""",
     """  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP>;
  constexpr int dyn = WR * WK * (XP >= 2 ? XP : 0) * MT * 1024;
  int e = dyn > 0 ? (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn) : 0;
  if (e) return e;
  e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, (size_t)dyn);"""),
]

PREP_OLD = """    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        if not rt:
            continue
        n, k = _gd_nk(proj, b)"""
PREP_NEW = """    todo, routes0 = [], {}
    for kk, rt0 in sorted(GD_ROUTE.items(), key=str):
        fb = GD_VP_FALLBACK.get(kk) if GD_VP_ENABLE else None
        todo.append((kk, rt0, fb, False))
        if fb and GD_VP_AB:
            todo.append((kk, fb, None, True))   # the replaced GD route, kept live as the call-time arm GD_VP_ENABLE False
    for (proj, b), rt, fb, arm0 in todo:
        if not rt or (not arm0 and (proj, b) in routes):
            continue
        n, k = _gd_nk(proj, b)"""
PY_EDITS = [
    ("keys", """        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None
        if key is not None and key not in keys:
            keys.append(key)""", """        for r_ in (rt, GD_VP_FALLBACK.get((proj, b)) if GD_VP_ENABLE else None):
            key = _gd_key(*_gd_nk(proj, b), r_) if r_ else None
            if key is not None and key not in keys:
                keys.append(key)"""),
    ("arm store", """                routes[(proj, b)] = r
                live.append(f"{proj}/{b} {desc}")""", """                (routes0 if arm0 else routes)[(proj, b)] = r
                live.append(f"{proj}/{b}{' (vp off)' if arm0 else ''} {desc}")"""),
    ("gd state", """    state["gd"] = {"mod": mod, "route": routes} if routes else None""",
     """    state["gd"] = {"mod": mod, "route": routes, "route0": routes0} if routes else None"""),
    ("call-time arm", """    r = st["route"].get((proj, b))""",
     """    r = (None if GD_VP_ENABLE else st["route0"].get((proj, b))) or st["route"].get((proj, b))"""),
    ("fallback", """        if why is not None:
            print(f"[gd] FALLBACK {proj}/{b}: {why} -> today's route", flush=True)""",
     """        if why is not None and fb:
            print(f"[vp] FALLBACK {proj}/{b} {rt}: {why} -> GD route {fb}", flush=True)
            todo.append(((proj, b), fb, None, False))
        elif why is not None:
            print(f"[gd] FALLBACK {proj}/{b}: {why} -> today's route", flush=True)"""),
]
VP_PY = '''

# VP (tools/vp_patches.py): GD routes with x through a per-warp cp.async smem ring (xp >= 2; bf16_proj vp_2). VP_ROUTE replaces
# GD_ROUTE entries, each with its GD route as the prepare-time fallback. GD_VP_ENABLE False = k3_vms_1k's GD routes.
GD_VP_ENABLE = True       # read at prepare (route table) and at call time (False: the replaced GD routes, kept live)
GD_VP_AB = True           # prepare also builds / self-tests each replaced GD route (the call-time arm and A/B control)
GD_VP_ROUTE = @VPROUTE@
GD_VP_FALLBACK = {kk: GD_ROUTE.get(kk) for kk in GD_VP_ROUTE} if GD_VP_ENABLE else {}
GD_ROUTE.update(GD_VP_ROUTE if GD_VP_ENABLE else {})
'''
VP_PY_ANCHOR = "\n\ndef _gd_bucket(T):\n"

# (proj, gd bucket) -> (mt, wr, wk, ku, st, s, xp): the R-vp-2 clone winners (in-prepare net us, XR vs today's GD route):
# o m8 (1,8,2,1) s2 x2 4.00 vs 4.29; o m16 (2,8,2,1) s2 x2 4.29 vs (2,4,4,1) 4.83; unpooler m16 (2,4,16,1) s2 x3 12.51 vs 12.99.
VP_ROUTE = {("o", 8): (1, 1, 8, 2, 2, 1, 2), ("o", 16): (2, 2, 8, 2, 2, 1, 2), ("unpooler", 16): (2, 2, 4, 16, 2, 1, 3)}


def _repl(text, edits, what):
    for name, old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"vp patch {what} {name!r}: anchor found {text.count(old)}x (want 1)")
        text = text.replace(old, new)
    return text


def apply(text, route=None):
    route = VP_ROUTE if route is None else route
    if "GD_VP_ENABLE" in text:
        raise RuntimeError("vp patch: GD_VP_ENABLE already in the base")
    c = text.index("GD_CUDA_SRC = r'''")
    d = text.index("'''", c + 20) + 3
    cuda = _repl(text[c:d], CUDA_EDITS, "cuda")
    if "'''" in cuda[18:-3]:
        raise RuntimeError("vp patch: the kernel source holds '''")
    text = _repl(text[:c] + cuda + text[d:], PY_EDITS, "py")
    if text.count(PREP_OLD) != 1 or text.count(VP_PY_ANCHOR) != 1:
        raise RuntimeError("vp patch: _gd_prepare / _gd_bucket anchors")
    text = text.replace(PREP_OLD, PREP_NEW)
    return text.replace(VP_PY_ANCHOR, VP_PY.replace("@VPROUTE@", repr(dict(sorted(route.items())))) + VP_PY_ANCHOR)


def main():
    n = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--n=")), 1)
    sys.path.insert(0, str(ROOT / "tools"))
    argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_vms as GV  # noqa: E402
    sys.argv = argv
    t, base, n1 = GV.vms1_text(kl1=True)
    tag = f"k3-vp-{n}"
    nonce = f"# nonce {tag} (k3_vms_1k + VP: GD x ring (cp.async, xp >= 2) on routes {sorted(VP_ROUTE)}, GD fallback)"
    t = apply(t.replace(n1, nonce))
    rep = []
    s = GV.GM.finish(t, nonce, tag, base, rep)
    out = ROOT / f"kernels/muse_decode_layer/k3_vp_{n}.py"
    lab = ROOT / f"kernels/lab/vp_k3_{n}_ab.py"
    if "--check" in sys.argv:
        GV.GM.check(out.exists() and out.read_text(encoding="utf-8") == s, f"{out.name} on disk differs")
        print(f"[vp] --check: {out.name} up to date")
        return 0
    out.write_text(s, encoding="utf-8", newline="\n")
    lab.write_text(s, encoding="utf-8", newline="\n")
    base_len = len((ROOT / "kernels/muse_decode_layer/k3_vms_1k.py").read_bytes())
    print(f"[vp] wrote {out.relative_to(ROOT)} and {lab.relative_to(ROOT)} ({len(s.encode())} B = k3_vms_1k "
          f"{len(s.encode()) - base_len:+d} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}), VP_ROUTE {VP_ROUTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

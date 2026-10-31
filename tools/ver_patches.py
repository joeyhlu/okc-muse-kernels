"""ver_patches.py: VMS_ER = early ring-slot release in the layer VMS consumer (moe redesign, R-ver).
Ship (VMS_RO, per-unit CTA sync): each consumer warp reads its 8 weight words of a unit from the ring slot inside the compute
loop, and tid 0 refills that slot only after the post-compute CTA sync, so a slot stays pinned for the whole compute (MK2:
C 0.82 us of a ~1.09 us/unit HBM share; ~1 % slack, mb_52: any loop schedule change costs ~3 %). VMS_ER 1: every warp loads
its whole unit (8 uint4 weights + the scales) into registers first, the CTA syncs, tid 0 refills the slot, and the compute runs
from registers while the next copy is in flight (+~24 registers). Same bar.sync count per unit, same memory ordering
(generic smem reads -> bar.sync -> the async-proxy write that refills the slot), same numerics. At VMS_ER 0 the source
preprocesses to the base's. Applies to the ship source with or without tools/vt16_patches.py (its 2-tile path reads wv[]
under VMS_ER).
    python -I tools/ver_patches.py <file.py>
"""
import pathlib
import sys

DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = "#ifndef VMS_ER\n#define VMS_ER 0   // R-ver: 1 = consumer loads its unit into registers, releases the ring slot, then computes\n#endif\n"
O_SC = "      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16);\n"
N_SC = (O_SC + "#if VMS_ER && VMS_RO && !VMS_HI\n"
        "      uint4 wv[8];   // R-ver: this warp's whole unit in registers, then the slot goes back to the copy engine\n"
        "#pragma unroll\n"
        "      for (int cc = 0; cc < 8; ++cc) wv[cc] = *reinterpret_cast<const uint4*>(slot + (warp * 8 + cc) * 512 + lane * 16);\n"
        "      cta_sync_compute();\n"
        "      if (q + NS < N) {\n"
        "#if __CUDA_ARCH__ >= 900\n"
        "        if (tid == 0)\n"
        "#endif\n"
        "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);\n"
        "      }\n"
        "#endif\n")
O_W = "{\n        const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * 8 + cc) * 512 + lane * 16);\n"
N_W = ("{\n#if VMS_ER && VMS_RO && !VMS_HI\n        const uint4 w = wv[cc];   // R-ver\n#else\n" + O_W[2:] + "#endif\n")   # O_W[2:]: the load line without its brace context
O_SYNC = "#if !VMS_HI\n      cta_sync_compute();\n"
N_SYNC = "#if !VMS_HI && !VMS_ER\n      cta_sync_compute();\n"

# the layer's VMS (no VMS_HI variant): the RO-path CTA sync + refill issue right before the pempty wait
O_SYNC_L = ("      cta_sync_compute();\n#if VMS_MK2\n      { const long long c2 = clock64(); cS += c2 - ck; ck = c2; }\n#endif\n"
            "      if (q + NS < N) {\n#if __CUDA_ARCH__ >= 900\n        if (tid == 0)\n#endif\n"
            "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);\n      }\n"
            "      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));\n")
N_SYNC_L = ("#if !VMS_ER\n" + O_SYNC_L[:O_SYNC_L.index("      mbar_wait(&S.pempty")] + "#endif\n"
            + O_SYNC_L[O_SYNC_L.index("      mbar_wait(&S.pempty"):])
PATCHES = [("defines", DEF_ANCHOR, DEFS + DEF_ANCHOR, 1), ("early release", O_SC, N_SC, 1), ("w from registers", O_W, N_W, 1)]


def apply(s):
    for tag, old, _, n in PATCHES:
        assert s.count(old) == n, f"ver anchor '{tag}' found {s.count(old)} times (want {n})"
    alt = [(O_SYNC, N_SYNC), (O_SYNC_L, N_SYNC_L)]
    hits = [(o, n) for o, n in alt if s.count(o) == 1]
    assert len(hits) == 1, f"ver anchor 'post-compute sync': {[s.count(o) for o, _ in alt]} (want exactly one form once)"
    for _, old, new, _ in PATCHES:
        s = s.replace(old, new)
    return s.replace(hits[0][0], hits[0][1])


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[ver] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

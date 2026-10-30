"""vhi_patches.py: VMS_HI = the helper warp refills the ring; no per-unit CTA-wide sync in the compute warps (moe redesign, R-vhi).
Ship (VMS_RO 1): after each unit the 8 compute warps meet at bar.sync 1 (256 threads) only so tid 0 may refill the unit's slot with
unit q + NS; every unit then waits for the slowest warp, and warp 0 also pays the issue. VMS_HI 1: compute warps go straight from
a unit's MMAs to the partial-buffer handoff (pempty -> red -> pfull, as the ship) and on to the next unit; the helper, right after
its pfull wait for unit q (= all 8 warps' partials, which depend on every word they read from the slot), issues unit q + NS into
slot q % NS before its reduction / epilogue / credit fence. Same data, same math, same partial order (bitwise equal outputs).
sm_86 test path: full barriers count 32 cp.async arrivals; the prologue copies come from compute warp 0, the refills from the
helper (32 lanes each). Needs VMS_RO 1. At VMS_HI 0 the source preprocesses to the base's.
    python -I tools/vhi_patches.py <file.py>
"""
import pathlib
import sys

OLD_ISSUE_FN = "__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {\n"
NEW_ISSUE_FN = r"""#if VMS_HI
__device__ __forceinline__ void issue32(const unsigned char* src, unsigned char* dst, uint64_t* bar, int lane) {   // R-vhi: one warp
#if __CUDA_ARCH__ >= 900
  if (lane == 0) issue(src, dst, bar, 0);
#else
  for (int i = lane; i < UB / 16; i += 32)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst + i * 16)), "l"(src + i * 16) : "memory");
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(su32(bar)) : "memory");
#endif
}
#endif
__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {
"""
OLD_INIT = "    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], NT);\n"
NEW_INIT = "    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], VMS_HI ? 32 : NT);   // R-vhi: one warp per copy\n"
OLD_PRO = ("    for (int q = 0; q < VMS_PRE && q < N; ++q) {   // R-vpre: prologue fill depth\n"
           "#if __CUDA_ARCH__ >= 900\n"
           "      if (tid == 0)\n"
           "#endif\n"
           "      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);\n"
           "    }\n")
NEW_PRO = ("    for (int q = 0; q < VMS_PRE && q < N; ++q) {   // R-vpre: prologue fill depth\n"
           "#if VMS_HI\n"
           "      if (warp == 0) issue32(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], lane);\n"
           "#else\n"
           "#if __CUDA_ARCH__ >= 900\n"
           "      if (tid == 0)\n"
           "#endif\n"
           "      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);\n"
           "#endif\n"
           "    }\n")
OLD_RO = ("      cta_sync_compute();\n"
          "#if VMS_MK2\n"
          "      { const long long c2 = clock64(); cS += c2 - ck; ck = c2; }\n"
          "#endif\n"
          "      if (q + NS < N) {\n"
          "#if __CUDA_ARCH__ >= 900\n"
          "        if (tid == 0)\n"
          "#endif\n"
          "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);\n"
          "      }\n"
          "      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));\n")
NEW_RO = ("#if !VMS_HI\n" + OLD_RO.rsplit("      mbar_wait(&S.pempty", 1)[0] + "#endif\n"
          "      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));\n")
OLD_HW = ("      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));\n"
          "#if VMS_MK2\n"
          "      { const long long c2 = clock64(); cF += c2 - ck; cP = c2; }\n"
          "#endif\n")
NEW_HW = (OLD_HW + "#if VMS_HI\n"
          "      if (q + NS < N) issue32(unit_src(p, S, q + NS, nA, a0, b0), ring + (q % NS) * UB, &S.full[q % NS], lane);   // R-vhi\n"
          "#endif\n")
DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = ("#ifndef VMS_HI\n#define VMS_HI 0   // R-vhi: 1 = helper refills the ring, no per-unit CTA sync in the compute warps\n#endif\n"
        "#if VMS_HI && !VMS_RO\n#error \"VMS_HI needs VMS_RO 1\"\n#endif\n")


def apply(s):
    for old, tag in ((OLD_ISSUE_FN, "scd"), (OLD_INIT, "full init"), (OLD_PRO, "prologue"), (OLD_RO, "RO unit end"),
                     (OLD_HW, "helper pfull wait"), (DEF_ANCHOR, "defines")):
        assert s.count(old) == 1, f"vhi anchor '{tag}' found {s.count(old)} times"
    for old, new in ((OLD_ISSUE_FN, NEW_ISSUE_FN), (OLD_INIT, NEW_INIT), (OLD_PRO, NEW_PRO), (OLD_RO, NEW_RO), (OLD_HW, NEW_HW)):
        s = s.replace(old, new)
    return s.replace(DEF_ANCHOR, DEFS + DEF_ANCHOR)


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vhi] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

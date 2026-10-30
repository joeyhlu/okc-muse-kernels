"""vhp_patches.py: VMS_HP = helper-warp epilogue constants loaded one unit ahead on the layer VMS (moe redesign, R-vhp).
Ship: the helper waits unit q's partials (pfull), then loads that unit's constants (gsA / gsB per job, the post-norm weights nw for
the unit's 16 rows: a fresh line every B unit) -> one dependent global round trip per unit, behind the expert stream's bytes in
flight. Only the last unit's is on the critical path (bs1 marks: compute end -> helper end 0.9 us), plus the last A unit of a job
(doneA credit -> the A->B hop). VMS_HP 1 loads unit q + 1's constants before unit q's pfull wait (registers rotate at the end of
the iteration), the first unit's before the loop. Same values, same arithmetic (w = nw * gs) -> bitwise equal. At VMS_HP 0 the
source preprocesses to the base's.
    python -I tools/vhp_patches.py <file.py>
"""
import pathlib
import sys

OLD_FN = "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {\n"
NEW_FN = r"""#if VMS_HP
__device__ __forceinline__ void hp_cst(const Params& p, const Smem& S, int q, int N, int nA, int a0, int b0, int g, float& gs, float& w0,
                                       float& w1) {   // R-vhp: unit q's epilogue constants
  if (q >= N) return;
  if (q < nA) {
    gs = __ldg(p.gsA + S.jE[(a0 + q) >> 9]);
  } else {
    const int v = b0 + (q - nA), n0 = (v & 255) * 16 + g;
    gs = __ldg(p.gsB + S.jE[v >> 8]);
    w0 = __ldg(p.nw + n0);
    w1 = __ldg(p.nw + n0 + 8);
  }
}
#endif
__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {
"""
OLD_PRE = ("    float* ssc = p.ss + (S.gen & 1) * MAXP;\n"
           "    for (int q = 0; q < N; ++q) {\n"
           "      const int bq = q & (NRB - 1);\n")
NEW_PRE = ("    float* ssc = p.ss + (S.gen & 1) * MAXP;\n"
           "#if VMS_HP\n"
           "    float hg = 0.f, hw0 = 0.f, hw1 = 0.f;\n"
           "    hp_cst(p, S, 0, N, nA, a0, b0, g, hg, hw0, hw1);\n"
           "#endif\n"
           "    for (int q = 0; q < N; ++q) {\n"
           "      const int bq = q & (NRB - 1);\n"
           "#if VMS_HP\n"
           "      float ng = 0.f, nw0 = 0.f, nw1 = 0.f;\n"
           "      hp_cst(p, S, q + 1, N, nA, a0, b0, g, ng, nw0, nw1);   // R-vhp: next unit's, before this unit's pfull wait\n"
           "#endif\n")
OLD_GA = "        const float gs = __ldg(p.gsA + S.jE[j]);\n"
NEW_GA = "#if VMS_HP\n        const float gs = hg;\n#else\n        const float gs = __ldg(p.gsA + S.jE[j]);\n#endif\n"
OLD_GB = ("        const float gs = __ldg(p.gsB + S.jE[j]);\n"
          "        const int n0 = rb * 16 + g;\n"
          "        const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;\n")
NEW_GB = ("#if VMS_HP\n"
          "        const float gs = hg;\n"
          "        const int n0 = rb * 16 + g;\n"
          "        const float w0 = hw0 * gs, w1 = hw1 * gs;\n"
          "#else\n" + OLD_GB + "#endif\n")
OLD_END = ("          if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);\n"
           "        }\n"
           "      }\n")
NEW_END = OLD_END + "#if VMS_HP\n      hg = ng; hw0 = nw0; hw1 = nw1;\n#endif\n"
DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = "#ifndef VMS_HP\n#define VMS_HP 0   // R-vhp: 1 = helper epilogue constants loaded one unit ahead\n#endif\n"


def apply(s):
    for old, tag in ((OLD_FN, "kernel"), (OLD_PRE, "helper loop"), (OLD_GA, "gsA"), (OLD_GB, "gsB / nw"), (OLD_END, "loop end"),
                     (DEF_ANCHOR, "defines")):
        assert s.count(old) == 1, f"vhp anchor '{tag}' found {s.count(old)} times"
    for old, new in ((OLD_FN, NEW_FN), (OLD_PRE, NEW_PRE), (OLD_GA, NEW_GA), (OLD_GB, NEW_GB), (OLD_END, NEW_END)):
        s = s.replace(old, new)
    return s.replace(DEF_ANCHOR, DEFS + DEF_ANCHOR)


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vhp] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

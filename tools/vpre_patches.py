"""vpre_patches.py: VMS_PRE = number of ring slots the layer VMS fills in its prologue (moe redesign, R-vpre).
mb_24 measured deeper rings slower at every small bucket (NS 4 / 5: b1 +1.5 / +4.6 %, b8 +1.4 / +3.7 %, b16 +0.3 / +1.6 %): the
start burst of 132 CTAs x NS x 36 KB puts each CTA's first unit behind more queued bytes. VMS_PRE p < NS issues only units 0..p-1
before the loop; tid 0 issues units p..NS-1 right after unit 0's full barrier completed (then the ship's refill of q + NS after
unit q continues unchanged). Default VMS_PRE = VMS_NS: the source preprocesses to the base's.
    python -I tools/vpre_patches.py <file.py>
"""
import pathlib
import sys

OLD_PRO = "    for (int q = 0; q < NS && q < N; ++q) {\n"
NEW_PRO = "    for (int q = 0; q < VMS_PRE && q < N; ++q) {   // R-vpre: prologue fill depth\n"
OLD_WAIT = "      mbar_wait(&S.full[sl], par);\n"
NEW_WAIT = ("      mbar_wait(&S.full[sl], par);\n"
            "#if VMS_PRE < VMS_NS\n"
            "      if (q == 0) {   // R-vpre: the rest of the ring after the first unit landed\n"
            "        for (int r = VMS_PRE; r < NS && r < N; ++r) {\n"
            "#if __CUDA_ARCH__ >= 900\n"
            "          if (tid == 0)\n"
            "#endif\n"
            "          issue(unit_src(p, S, r, nA, a0, b0), ring + r * UB, &S.full[r], tid);\n"
            "        }\n"
            "      }\n"
            "#endif\n")
DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = "#ifndef VMS_PRE\n#define VMS_PRE VMS_NS   // R-vpre: ring slots filled before the unit loop\n#endif\n"


def apply(s):
    for old, tag in ((OLD_PRO, "prologue"), (OLD_WAIT, "full wait"), (DEF_ANCHOR, "defines")):
        assert s.count(old) == 1, f"vpre anchor '{tag}' found {s.count(old)} times"
    s = s.replace(OLD_PRO, NEW_PRO).replace(OLD_WAIT, NEW_WAIT)
    return s.replace(DEF_ANCHOR, DEFS + DEF_ANCHOR)


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vpre] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

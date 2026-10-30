"""vud_patches.py: VM_UD2 = wgmma B descriptor from a provably warp-uniform value, re-ported to the current VML (moe redesign;
R-gap-0 candidate 1, parent 2026-10-30). Every HGMMA of stage<N> reads its B descriptor from uniform registers, but the stage's
x-slot base dlo = d0 + xs * (XS >> 4) comes from `it` (not uniform to ptxas), so each HGMMA rebuilds lo (VIADD) + hi (IMAD.MOV)
and moves both with R2UR (~64 / stage; R-gap-0: our ~337 instr / stage vs alygg77's ~295). R-m32e-12 (Oct 28, bf2 builds, T 256
standalone) found no stage-time change; untested on today's w72 (CREG 112) on board routings, whose hot nt 5-8 jobs are
compute-bound. VM_UD2 3: dlo = __reduce_max_sync over the warp (a REDUX into a UR); 1: REDUX of xs only. Default 0 = the base.
    python -I tools/vud_patches.py <file.py>
"""
import pathlib
import sys

OLD = "const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);\n#if VM_UD\n"
NEW = ("#if VM_UD2 == 3   // R-vud: the stage's x-slot descriptor base through a REDUX -> uniform register\n"
       "const uint32_t dlo = __reduce_max_sync(0xffffffffu, (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4)), dhi = (uint32_t)(d0 >> 32);\n"
       "#elif VM_UD2 == 1\n"
       "const uint32_t dlo = (uint32_t)d0 + __reduce_max_sync(0xffffffffu, (uint32_t)xs) * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);\n"
       "#else\n"
       "const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);\n"
       "#endif\n#if VM_UD\n")
DEF_A = "#ifndef VM_UD\n#define VM_UD 0\n"
DEF_N = "#ifndef VM_UD2\n#define VM_UD2 0   // R-vud: 3 = REDUX-uniform x descriptor base per stage, 1 = REDUX of xs\n#endif\n" + DEF_A


def apply(s):
    for old, tag in ((OLD, "descriptor"), (DEF_A, "defines")):
        assert s.count(old) == 1, f"vud anchor '{tag}' found {s.count(old)} times"
    return s.replace(OLD, NEW).replace(DEF_A, DEF_N)


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vud] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

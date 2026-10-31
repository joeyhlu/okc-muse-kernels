"""vcb_patches.py (R-vcb): VMS_CB = batched combine loads in the layer VMS (moe lane, T = 1 / 8 line; Codex audit ext_codex_t1.md s2).

Ship: after the grid barrier, the M x 4 combine CTAs (the kernel's last work) sum each token's <= 8 local routes; ptxas keeps every
route's three loads (tw, ss, the ybuf float4) inside its own pr >= 0 branch and consumes them at once (SASS of the ship: LDG x3 ->
MUFU.RSQ -> FFMA x4 per route), so a token's 2-3 routes (R-rh-2: T = 1 has 2-3 local experts) pay 2-3 serial L2 round trips.
VMS_CB 1: the 8 route slots' loads are issued first (an invalid slot reads row 0 of ybuf / ss, a valid address, and is ignored),
then the same arithmetic runs in the same route order: the FP32 ops and their order are unchanged (bitwise-equal output).
At VMS_CB 0 the source preprocesses to the base's.
    python -I tools/vcb_patches.py <file.py>
"""
import pathlib
import sys

DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = "#ifndef VMS_CB\n#define VMS_CB 0   // R-vcb: 1 = the combine issues every route slot's loads before its arithmetic\n#endif\n"
O_CMB = ("#pragma unroll\n"
         "    for (int s = 0; s < 8; ++s) {\n"
         "      const int pr = S.pos[tk * 8 + s];\n"
         "      if (pr >= 0) {\n"
         "        const float cf = __ldg(p.tw + tk * 8 + s) * rsqrtf(__ldcg(ssc + pr) * (1.f / HD) + p.eps);\n"
         "        const float4 y = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)pr * HD + col));\n"
         "        a.x += cf * y.x; a.y += cf * y.y; a.z += cf * y.z; a.w += cf * y.w;\n"
         "      }\n"
         "    }\n")
N_CMB = ("#if VMS_CB\n"
         "    int cpr[8];   // R-vcb: every slot's loads first (invalid slots read row 0 and are skipped), then the ship's arithmetic in order\n"
         "    float ctw[8], css[8];\n"
         "    float4 cy[8];\n"
         "#pragma unroll\n"
         "    for (int s = 0; s < 8; ++s) cpr[s] = S.pos[tk * 8 + s];\n"
         "#pragma unroll\n"
         "    for (int s = 0; s < 8; ++s) {\n"
         "      const int pq = cpr[s] >= 0 ? cpr[s] : 0;\n"
         "      ctw[s] = __ldg(p.tw + tk * 8 + s);\n"
         "      css[s] = __ldcg(ssc + pq);\n"
         "      cy[s] = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)pq * HD + col));\n"
         "    }\n"
         "#pragma unroll\n"
         "    for (int s = 0; s < 8; ++s) {\n"
         "      if (cpr[s] >= 0) {\n"
         "        const float cf = ctw[s] * rsqrtf(css[s] * (1.f / HD) + p.eps);\n"
         "        const float4 y = cy[s];\n"
         "        a.x += cf * y.x; a.y += cf * y.y; a.z += cf * y.z; a.w += cf * y.w;\n"
         "      }\n"
         "    }\n"
         "#else\n" + O_CMB + "#endif\n")
PATCHES = [("defines", DEF_ANCHOR, DEFS + DEF_ANCHOR, 1), ("combine", O_CMB, N_CMB, 1)]


def apply(s):
    for tag, old, _, n in PATCHES:
        assert s.count(old) == n, f"vcb anchor '{tag}' found {s.count(old)} times (want {n})"
    for _, old, new, _ in PATCHES:
        s = s.replace(old, new)
    return s


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vcb] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

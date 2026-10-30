"""vsh_patches.py: VMS_SH = expert-strided VMS shares for few-job calls (moe redesign, R-vsh).

Shipped VMS cuts the job-major A list and the job-major B list into 132 contiguous ranges, so a CTA's A slice and B slice sit
on the same job band and every CTA's first B unit waits for the whole band's A (doneA hop ~2 us at T=1, fully exposed: the
board's bs1 has 2-4 local experts on 99 % of draws). VMS_SH n (compile-time) with 2 <= njobs <= n at runtime: every CTA takes
ka = (c+1)*512/G - c*512/G A units of EVERY job and kb = (c+1)*256/G - c*256/G B units of every job, in the order
A(j0) A(j1) .. B(j0) B(j1) ..: doneA(j0) completes while the CTAs run A(j1), doneA(j_last) while they run B(j0). Per-CTA
work is unchanged (same unit counts), credits are per job (count ka) after the CTA's last A unit of that job, and an A job
switch skips the x reload when this lane's token row is unchanged (every job at T=1). Outside 2..n jobs the ship's plan
runs. Numerics identical to the ship (same units, same per-unit math). Needs !(VMS_FX & 4) (its pre-route first copies use
the contiguous plan). At VMS_SH 0 the source preprocesses to the base's.
    python -I tools/vsh_patches.py <file.py>
"""
import pathlib
import sys


def _rep(s, old, new, tag):
    n = s.count(old)
    assert n == 1, f"vsh anchor '{tag}' found {n} times"
    return s.replace(old, new)


OLD_SRC = ("__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {\n"
           "  if (q < nA) { const int v = a0 + q; return p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB; }\n"
           "  const int v = b0 + (q - nA); return p.wB + ((size_t)S.jE[v >> 8] * RBB + (v & 255)) * UB;\n}\n")
NEW_SRC = ("#ifndef VMS_SH\n#define VMS_SH 0   // R-vsh: > 0 = expert-strided shares when 2 <= njobs <= VMS_SH\n#endif\n"
           "#if VMS_SH && (VMS_FX & 4)\n#error \"VMS_SH needs VMS_FX without bit 4 (pre-route first copies use the contiguous plan)\"\n#endif\n"
           "struct Plan { int a0, b0, nA, sh, ka, kaL, kb, kbL; };   // R-vsh: sh = strided shares\n"
           "__device__ __forceinline__ int vA(const Plan& P, int q) { return P.sh ? (q / P.ka) * RBA + P.kaL + q % P.ka : P.a0 + q; }\n"
           "__device__ __forceinline__ int vB(const Plan& P, int r) { return P.sh ? (r / P.kb) * RBB + P.kbL + r % P.kb : P.b0 + r; }\n"
           "__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, const Plan& P) {\n"
           "  if (q < P.nA) { const int v = vA(P, q); return p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB; }\n"
           "  const int v = vB(P, q - P.nA); return p.wB + ((size_t)S.jE[v >> 8] * RBB + (v & 255)) * UB;\n}\n"
           "__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {\n"
           "  const Plan P{a0, b0, nA, 0, 1, 0, 1, 0};\n  return unit_src(p, S, q, P);\n}\n")
OLD_PLAN = "  const int nA = a1 - a0, N = t1 - t0;\n"
NEW_PLAN = ("#if VMS_SH\n"
            "  const int shp = (njobs >= 2 && njobs <= VMS_SH) ? 1 : 0;\n"
            "  const int kaL = c * RBA / G, kbL = c * RBB / G;\n"
            "  const Plan PL{a0, b0, shp ? njobs * ((c + 1) * RBA / G - kaL) : a1 - a0, shp, (c + 1) * RBA / G - kaL, kaL,\n"
            "                (c + 1) * RBB / G - kbL, kbL};\n"
            "  const int nA = PL.nA, N = shp ? njobs * (PL.ka + PL.kb) : t1 - t0;\n"
            "#else\n"
            "  const Plan PL{a0, b0, a1 - a0, 0, 1, 0, 1, 0};\n"
            "  const int nA = a1 - a0, N = t1 - t0;\n"
            "#endif\n")
OLD_ISSUE0 = "      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);\n    }\n#endif\n"
NEW_ISSUE0 = "      issue(unit_src(p, S, q, PL), ring + q * UB, &S.full[q], tid);\n    }\n#endif\n"
OLD_ISSUE = "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);\n"
NEW_ISSUE = "        issue(unit_src(p, S, q + NS, PL), ring + sl * UB, &S.full[sl], tid);\n"
OLD_V = "      const int v = PB ? b0 + (q - nA) : a0 + q;\n      const int j = PB ? (v >> 8) + 4096 : (v >> 9);\n"
NEW_V = "      const int v = PB ? vB(PL, q - nA) : vA(PL, q);\n      const int j = PB ? (v >> 8) + 4096 : (v >> 9);\n"
OLD_CURJ = "    int curj = -1;\n    int sl = 0; uint32_t par = 0;\n"
NEW_CURJ = "    int curj = -1;\n#if VMS_SH\n    int xrow = -3;   // R-vsh: token row this lane's xr holds (-1 zeros, -2 B h, -3 none)\n#endif\n    int sl = 0; uint32_t par = 0;\n"
OLD_XL = ("        if (g < cnt) {\n"
          "          const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID)\n")
NEW_XL = ("#if VMS_SH\n"
          "        const int nrow = PB ? -2 : (g < cnt ? (S.ent[st + g] >> 3) : -1);\n"
          "        const bool xsame = !PB && nrow == xrow;   // R-vsh: same token row in this lane -> keep xr\n"
          "        xrow = nrow;\n"
          "        if (xsame) {\n"
          "        } else\n"
          "#endif\n" + OLD_XL)
OLD_HA = ("        const int v = a0 + q, j = v >> 9, rb = v & 511;\n")
NEW_HA = ("        const int v = vA(PL, q), j = v >> 9, rb = v & 511;\n")
OLD_CR = ("        if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units\n")
NEW_CR = ("        if (PL.sh ? (q % PL.ka == PL.ka - 1) : (q == nA - 1 || ((v + 1) >> 9) != j)) {     // this CTA's last unit of job j: credit its units\n")
OLD_CNT = "          if (lane == 0) atomicAdd(p.ctr + 8 + j, min(a1, (j + 1) * RBA) - max(a0, j * RBA));\n"
NEW_CNT = "          if (lane == 0) atomicAdd(p.ctr + 8 + j, PL.sh ? PL.ka : min(a1, (j + 1) * RBA) - max(a0, j * RBA));\n"
OLD_HB = "        const int v = b0 + (q - nA), j = v >> 8, rb = v & 255;\n"
NEW_HB = "        const int v = vB(PL, q - nA), j = v >> 8, rb = v & 255;\n"


def apply(s):
    s = _rep(s, OLD_SRC, NEW_SRC, "unit_src")
    s = _rep(s, OLD_PLAN, NEW_PLAN, "plan")
    s = _rep(s, OLD_ISSUE0, NEW_ISSUE0, "issue0")
    assert s.count(OLD_ISSUE) == 2, "issue refill"
    s = s.replace(OLD_ISSUE, NEW_ISSUE)
    s = _rep(s, OLD_V, NEW_V, "loop v")
    s = _rep(s, OLD_CURJ, NEW_CURJ, "curj")
    s = _rep(s, OLD_XL, NEW_XL, "x load")
    s = _rep(s, OLD_HA, NEW_HA, "helper A v")
    s = _rep(s, OLD_CR, NEW_CR, "credit cond")
    s = _rep(s, OLD_CNT, NEW_CNT, "credit count")
    s = _rep(s, OLD_HB, NEW_HB, "helper B v")
    return s


def layer_src(path):
    L = pathlib.Path(path).read_text(encoding="utf-8")
    a = L.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    c = L.index('VMS_CPP_SRC = r"""') + len('VMS_CPP_SRC = r"""')
    return L[a:L.index('"""', a)], L[c:L.index('"""', c)]


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    p = apply(t)
    print(f"[vsh] {sys.argv[1]}: +{len(p) - len(t)} B, all anchors applied")

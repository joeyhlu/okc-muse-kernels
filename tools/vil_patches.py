"""vil_patches.py (R-vil): VMS_IL = job-interleaved unit order for the T = 1 VMS (moe lane, T = 1 / 8 line).

Ship: the A list (njobs x 512 units) and the B list (njobs x 256) are cut into contiguous per-CTA ranges, so a CTA's A range sits in
~1 job and job j's A completes only when all of its ~G / njobs CTAs finish: B waits for the slowest A CTA (board b1 marks: doneA
wait ~1.9 us p50, ~1.2 us for the slowest CTA). R-rh-2: at T = 1 this GPU sees 2-3 local experts (1 token each) on every draw.
VMS_IL 1: every CTA takes the SAME row-block share of every job (A rows [c * 512 / G, (c + 1) * 512 / G), B rows [c * 256 / G, ...)),
job 0 first: job 0's A completes about half-way through every CTA's A list, so B of job 0 starts with no doneA wait and job 1's A
has a whole B unit pair of slack. At M == 1 every job is the same token, so A job switches keep the x registers (no reload).
a0 / b0 carry (row start | row count << 16) under IL; UVA / UVB map a unit index to the ship's v = job * 512 + rb (A) or
job * 256 + rb (B), so unit_src, the helper epilogues, doneA counters and the combine are unchanged. Correct for any M (slower
above M = 1: job switches reload x); meant as its own module for M == 1 calls. At VMS_IL 0 the source preprocesses to the base's
(the macros expand to the ship's expressions). Applies after tools/vt16_patches.py (or on the bare ship source).
    python -I tools/vil_patches.py <file.py>
"""
import pathlib
import sys

DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = ("#ifndef VMS_IL\n#define VMS_IL 0   // R-vil: 1 = job-interleaved unit order (every CTA: the same row share of every job, job 0 first)\n#endif\n"
        "#if VMS_IL && ((VMS_FX & 4) || VMS_XPF || VMS_HI)\n#error \"VMS_IL: FX 4 / XPF / HI index the ship's contiguous unit ranges\"\n#endif\n")
O_USRC = "__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {\n"
N_USRC = ("#if VMS_IL\n"
          "__device__ __forceinline__ int il_uv(int q, int ab, int per) {   // R-vil: q-th unit of a CTA whose share is rows [ab & 0xFFFF, + ab >> 16) of every job\n"
          "  const int l = ab >> 16, r = q / l;\n"
          "  return r * per + (ab & 0xFFFF) + (q - r * l);\n"
          "}\n"
          "#define UVA(q) il_uv((q), a0, RBA)\n"
          "#define UVB(q) il_uv((q) - nA, b0, RBB)\n"
          "#else\n"
          "#define UVA(q) (a0 + (q))\n"
          "#define UVB(q) (b0 + ((q) - nA))\n"
          "#endif\n" + O_USRC)
REPS = [
    ("unit_src A", "  if (q < nA) { const int v = a0 + q; return p.wA", "  if (q < nA) { const int v = UVA(q); return p.wA", 1),
    ("unit_src B", "  const int v = b0 + (q - nA); return p.wB", "  const int v = UVB(q); return p.wB", 1),
    ("consumer v", "      const int v = PB ? b0 + (q - nA) : a0 + q;", "      const int v = PB ? UVB(q) : UVA(q);", 1),
    ("helper A", "        const int v = a0 + q, j = v >> 9, rb = v & 511;", "        const int v = UVA(q), j = v >> 9, rb = v & 511;", 1),
    ("helper B", "        const int v = b0 + (q - nA), j = v >> 8, rb = v & 255;", "        const int v = UVB(q), j = v >> 8, rb = v & 255;", 1),
]
OPT = [("hp_cst A", "    gs = __ldg(p.gsA + S.jE[(a0 + q) >> 9]);", "    gs = __ldg(p.gsA + S.jE[UVA(q) >> 9]);"),
       ("hp_cst B", "    const int v = b0 + (q - nA), n0 =", "    const int v = UVB(q), n0 ="),
       ("vt16 helper vote", "S.jC[(q < nA) ? ((a0 + q) >> 9) : ((b0 + (q - nA)) >> 8)]", "S.jC[(q < nA) ? (UVA(q) >> 9) : (UVB(q) >> 8)]")]
O_LAST = "        if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units\n"
N_LAST = ("#if VMS_IL\n        if (q == nA - 1 || (UVA(q + 1) >> 9) != j) {   // R-vil: this CTA's last unit of job j\n#else\n"
          + O_LAST + "#endif\n")
O_CRED = "          if (lane == 0) atomicAdd(p.ctr + 8 + j, min(a1, (j + 1) * RBA) - max(a0, j * RBA));\n"
N_CRED = ("#if VMS_IL\n          if (lane == 0) atomicAdd(p.ctr + 8 + j, a0 >> 16);   // R-vil: la units of every job\n#else\n"
          + O_CRED + "#endif\n")
O_PLAN = ("  const int njobs = S.njobs;\n"
          "  const int NA = njobs * RBA, NN = njobs * (RBA + RBB);\n"
          "  const int a0 = c * NA / G, a1 = (c + 1) * NA / G;\n"
          "  const int t0 = c * NN / G, t1 = (c + 1) * NN / G;\n"
          "  const int b0 = t0 - a0;\n"
          "  const int nA = a1 - a0, N = t1 - t0;\n")
N_PLAN = ("#if VMS_IL\n"
          "  const int njobs = S.njobs;   // R-vil: the same A / B row-block share of every job, job 0 first\n"
          "  const int ra = c * RBA / G, la = (c + 1) * RBA / G - ra, rq = c * RBB / G, lq = (c + 1) * RBB / G - rq;\n"
          "  const int a0 = ra | (la << 16), b0 = rq | (lq << 16);\n"
          "  const int nA = njobs * la, N = nA + njobs * lq;\n"
          "#else\n" + O_PLAN + "#endif\n")
O_CURJ = "    int curj = -1;\n"
N_CURJ = "#if VMS_IL\n    bool il_ax = false;   // R-vil: the A token's x is in xr\n#endif\n" + O_CURJ
O_XL = ("        if (g < cnt) {\n"
        "          const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID)\n")
N_XL = ("#if VMS_IL\n"
        "        const bool il_keep = !PB && il_ax && p.m == 1;   // R-vil: M == 1: every A job is the same token\n"
        "        il_ax = !PB;\n"
        "        if (!il_keep)\n"
        "#endif\n" + O_XL)
PATCHES = ([("defines", DEF_ANCHOR, DEFS + DEF_ANCHOR, 1), ("uv", O_USRC, N_USRC, 1)] + REPS
           + [("plan", O_PLAN, N_PLAN, 1), ("curj", O_CURJ, N_CURJ, 1), ("x load", O_XL, N_XL, 1), ("last of job", O_LAST, N_LAST, 1),
              ("credit", O_CRED, N_CRED, 1)])


def apply(s):
    for tag, old, _, n in PATCHES:
        assert s.count(old) == n, f"vil anchor '{tag}' found {s.count(old)} times (want {n})"
    for _, old, new, _ in PATCHES:
        s = s.replace(old, new)
    for tag, old, new in OPT:
        assert s.count(old) in (0, 1), f"vil optional anchor '{tag}' found {s.count(old)} times"
        s = s.replace(old, new)
    return s


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vil] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

"""vt16_patches.py: VMS_T16 = 16-token VMS jobs with a weighted unit split (moe redesign, R-vt16; Muse hot-tile idea, ext_muse_review_xhigh).
Ship: VMS jobs hold <= 8 tokens (one m16n8k16 n-tile, x in registers), so an expert with 9-16 tokens is two jobs and its weights
stream twice: +20.9 % weight passes on the layer's T = 16 profiles (52 jobs / 43 experts; board bs16 +3.6 %, T <= 8 0 %). VML
instead of VMS at T = 16 is +9 % (mb_44). VMS_T16 1:
  - route() forms jobs of <= 16 tokens;
  - a 2-tile job's unit unpacks every weight word once and feeds two n-tiles: tokens 0-7 from the x registers (as the ship),
    tokens 8-15 from this lane's own smem slot x2s (loaded at the job switch, uint4 x2s[NW][16][32], 64 KB); each tile keeps 2 MMA
    chains (accumulators: 4 x 4 floats as the ship); its tile-2 partials go to red2[NRB][NW][32] (16 KB) before the pfull arrive;
  - the helper reduces red2 too (before freeing the buffer) and runs the A (SwiGLU -> inter) / B (ybuf + ss) epilogue for tokens
    8-15 of 2-tile units;
  - the A and B unit lists are split by weight (VMS_WT1 per 1-tile unit, VMS_WT2 per 2-tile unit; default 20 / 30) with one
    shared inverse t16_split(), so neighbouring CTAs' boundaries partition the lists exactly; warp 0 computes the four split
    points in route() from register prefixes (ballot counts + shuffles; mb_49: per-thread smem binary searches cost ~1.2 us).
Unit -> (job, row block) mapping, doneA counts (512 A units per job), unit_src, the combine are unchanged. Numerics: tokens 0-7 of
a 2-tile unit sum 2 chains instead of 4 (FP32 order only). SMEM ~209 KB (does not fit an RTX 3070: judge-GPU checks only).
At VMS_T16 0 the source preprocesses to the base's.
    python -I tools/vt16_patches.py <file.py>
"""
import pathlib
import sys

DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = ("#ifndef VMS_T16\n#define VMS_T16 0   // R-vt16: 1 = 16-token jobs (2 n-tiles per unit), weighted unit split\n#endif\n"
        "#ifndef VMS_T16F\n#define VMS_T16F 0   // R-vt16: 1 = no 2-tile job in the call -> the ship's closed-form split (no prefix / ballot plan)\n#endif\n"
        "#ifndef VMS_T16E\n#define VMS_T16E 0   // R-vt16: 1 = consumer 1-tile loop first (fall-through), the 2-tile path after it (layout)\n#endif\n"
        "#ifndef VMS_T16X\n#define VMS_T16X 0   // R-vt16 bisect (exact only without > 8-token jobs): 1 = no consumer 2-tile path, 2 = no helper tile 2, 4 = base plan\n#endif\n"
        "#ifndef VMS_LPAD\n#define VMS_LPAD 0   // R-vt16 probe: bytes of unused smem\n#endif\n"
        "#ifndef VMS_WT1\n#define VMS_WT1 20   // R-vt16: split weight of a 1-tile unit\n#endif\n"
        "#ifndef VMS_WT2\n#define VMS_WT2 30   // R-vt16: split weight of a 2-tile unit (> 8 tokens)\n#endif\n")
O_SMEM = "  float4 red[NRB][NW][32];\n};\n"
N_SMEM = ("  float4 red[NRB][NW][32];\n#if VMS_T16\n  float4 red2[NRB][NW][32];   // R-vt16: tile-2 partials\n"
          "  uint4 x2s[NW][16][32];       // R-vt16: tokens 8-15 x, one private uint4 column per lane\n"
          "  int pl[4];                    // R-vt16: this CTA's a0, a1, b0, b1 (warp 0, route)\n#endif\n"
          "#if VMS_LPAD > 0\n  unsigned char lpad[VMS_LPAD];   // R-vt16 probe: unused smem (L1 carve-out effect)\n#endif\n};\n")
O_SPLITFN = "__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {\n"
N_SPLITFN = r"""#if VMS_T16
// R-vt16 plan helpers, warp 0 only (all 32 lanes, uniform results): lane l holds s0 = jW[l + 1], s1 = jW[l + 33] (inclusive
// prefixes of the per-unit job weights) and w0 / w1 = the weights of jobs l / l + 32 (0 past the last job).
__device__ __forceinline__ int t16_jw(int s0, int s1, int j) {   // jW[j], 0 <= j <= 64
  const int u0 = __shfl_sync(0xffffffffu, s0, (j - 1) & 31), u1 = __shfl_sync(0xffffffffu, s1, (j - 33) & 31);
  return j == 0 ? 0 : (j <= 32 ? u0 : u1);
}
__device__ __forceinline__ int t16_wj(int w0, int w1, int j) {   // weight of job j
  const int u0 = __shfl_sync(0xffffffffu, w0, j & 31), u1 = __shfl_sync(0xffffffffu, w1, j & 31);
  return j < 32 ? u0 : (j < 64 ? u1 : 0);
}
__device__ __forceinline__ int t16_split(int s0, int s1, int w0, int w1, int nJ, int per, int x) {   // weight -> unit index
  const int lane = threadIdx.x & 31;   // lo = largest j <= nJ with per * jW[j] <= x (jW strictly rises: a ballot count)
  const int lo = __popc(__ballot_sync(0xffffffffu, lane < nJ && per * s0 <= x)) + __popc(__ballot_sync(0xffffffffu, lane + 32 < nJ && per * s1 <= x));
  const int dx = x - per * t16_jw(s0, s1, lo), wl = t16_wj(w0, w1, lo);
  return lo >= nJ ? nJ * per : lo * per + (wl == VMS_WT2 ? dx / VMS_WT2 : dx / VMS_WT1);
}
__device__ __forceinline__ int t16_wpre(int s0, int s1, int w0, int w1, int per, int v) {   // weight of the units before index v
  const int j = v / per;
  return per * t16_jw(s0, s1, j) + (v - j * per) * t16_wj(w0, w1, j);
}
#endif
__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {
"""
O_NJ = "    int inc = tot, nj = (tot + 7) >> 3, jinc = nj;\n"
N_NJ = ("#if VMS_T16\n    int inc = tot, nj = (tot + 15) >> 4, jinc = nj;   // R-vt16: <= 16 tokens per job\n#else\n" + O_NJ + "#endif\n")
O_JOBS = "    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + 8 * k; S.jC[joff + k] = min(8, tot - 8 * k); }\n"
N_JOBS = ("#if VMS_T16\n    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + 16 * k; S.jC[joff + k] = min(16, tot - 16 * k); }\n"
          "#else\n" + O_JOBS + "#endif\n")
O_NJOBS = "    if (lane == 31) S.njobs = jinc;\n    __syncwarp();\n"
N_NJOBS = (O_NJOBS + "#if VMS_T16 && !(VMS_T16X & 4)\n"
           "    {   // R-vt16: per-job unit weights, their prefix and this CTA's weighted A / B unit ranges, once per CTA in registers\n"
           "      const int nJ = __shfl_sync(0xffffffffu, jinc, 31);\n"
           "      const int w0 = lane < nJ ? (S.jC[lane] > 8 ? VMS_WT2 : VMS_WT1) : 0;\n"
           "      const int w1 = lane + 32 < nJ ? (S.jC[lane + 32] > 8 ? VMS_WT2 : VMS_WT1) : 0;\n"
           "      const bool flat = VMS_T16F && __ballot_sync(0xffffffffu, w0 == VMS_WT2 || w1 == VMS_WT2) == 0u;   // no 2-tile job\n"
           "      if (flat) {   // the ship's closed-form split\n"
           "        if (lane == 0) {\n"
           "          const int NA0 = nJ * RBA, NN0 = nJ * (RBA + RBB), fa0 = c * NA0 / G, fa1 = (c + 1) * NA0 / G;\n"
           "          S.pl[0] = fa0; S.pl[1] = fa1; S.pl[2] = c * NN0 / G - fa0; S.pl[3] = (c + 1) * NN0 / G - fa1;\n"
           "        }\n"
           "      } else {\n"
           "      int s0 = w0, s1 = w1;\n"
           "#pragma unroll\n"
           "      for (int d = 1; d < 32; d <<= 1) {\n"
           "        const int u0 = __shfl_up_sync(0xffffffffu, s0, d), u1 = __shfl_up_sync(0xffffffffu, s1, d);\n"
           "        if (lane >= d) { s0 += u0; s1 += u1; }\n"
           "      }\n"
           "      s1 += __shfl_sync(0xffffffffu, s0, 31);   // jW[lane + 33]\n"
           "      const int WJ = __shfl_sync(0xffffffffu, s1, 31);   // per-unit weights summed over jobs (A: 512 units / job, B: 256)\n"
           "      const int pa0 = t16_split(s0, s1, w0, w1, nJ, RBA, RBA * WJ * c / G), pa1 = t16_split(s0, s1, w0, w1, nJ, RBA, RBA * WJ * (c + 1) / G);\n"
           "      // B starts where this CTA's total (A + B) weight share begins: the ship's b0 = t0 - a0 coupling in weight space\n"
           "      const int T0 = (RBA + RBB) * WJ * c / G, T1 = (RBA + RBB) * WJ * (c + 1) / G;\n"
           "      const int pb0 = t16_split(s0, s1, w0, w1, nJ, RBB, T0 - t16_wpre(s0, s1, w0, w1, RBA, pa0));\n"
           "      const int pb1 = t16_split(s0, s1, w0, w1, nJ, RBB, T1 - t16_wpre(s0, s1, w0, w1, RBA, pa1));\n"
           "      if (lane == 0) { S.pl[0] = pa0; S.pl[1] = pa1; S.pl[2] = pb0; S.pl[3] = pb1; }\n"
           "      }\n"
           "    }\n"
           "#endif\n")
O_PLAN = ("  const int njobs = S.njobs;\n"
          "  const int NA = njobs * RBA, NN = njobs * (RBA + RBB);\n"
          "  const int a0 = c * NA / G, a1 = (c + 1) * NA / G;\n"
          "  const int t0 = c * NN / G, t1 = (c + 1) * NN / G;\n"
          "  const int b0 = t0 - a0;\n"
          "  const int nA = a1 - a0, N = t1 - t0;\n")
N_PLAN = ("#if VMS_T16 && !(VMS_T16X & 4)\n"
          "  const int njobs = S.njobs;\n"
          "  const int a0 = S.pl[0], a1 = S.pl[1], b0 = S.pl[2], b1 = S.pl[3];   // R-vt16: warp 0's weighted split (route)\n"
          "  const int nA = a1 - a0, N = nA + (b1 - b0);\n"
          "#else\n" + O_PLAN + "#endif\n")
O_XDECL = "    uint4 xr[16];\n    int curj = -1;\n"
N_XDECL = ("    uint4 xr[16];\n#if VMS_T16\n    uint4* x2p = &S.x2s[warp][0][lane];   // R-vt16: this lane's tokens 8-15 x\n    bool two = false;\n#endif\n"
           "    int curj = -1;\n")
O_XLOAD = ("        } else {\n"
           "#pragma unroll\n"
           "          for (int cc = 0; cc < 16; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);\n"
           "        }\n"
           "      }\n")
N_XLOAD = ("        } else {\n"
           "#pragma unroll\n"
           "          for (int cc = 0; cc < 16; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);\n"
           "        }\n"
           "#if VMS_T16\n"
           "        two = !(VMS_T16X & 1) && __all_sync(0xffffffffu, cnt > 8);   // R-vt16: a vote = provably warp-uniform (no collective guards on the mma.sync branch)\n"
           "        if (two) {   // R-vt16: tokens 8-15 of the job into this lane's x2s column\n"
           "          if (g + 8 < cnt) {\n"
           "            const uint4* src2 = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + 8 + g) * ID)\n"
           "                                   : reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + 8 + g] >> 3) * HD);\n"
           "            src2 += warp * 64 + 2 * t;\n"
           "#pragma unroll\n"
           "            for (int cc = 0; cc < 8; ++cc) {\n"
           "              x2p[(2 * cc) * 32] = PB ? __ldcg(src2 + cc * 8) : __ldg(src2 + cc * 8);\n"
           "              x2p[(2 * cc + 1) * 32] = PB ? __ldcg(src2 + cc * 8 + 1) : __ldg(src2 + cc * 8 + 1);\n"
           "            }\n"
           "          } else {\n"
           "#pragma unroll\n"
           "            for (int cc = 0; cc < 16; ++cc) x2p[cc * 32] = make_uint4(0u, 0u, 0u, 0u);\n"
           "          }\n"
           "        }\n"
           "#endif\n"
           "      }\n")
O_CONS = r"""      float acc[4][4];
#pragma unroll
      for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) {
"""
T2_BODY = r"""#pragma unroll
        for (int cc = 0; cc < 8; ++cc) {
#if VMS_ER
          const uint4 w = wv[cc];   // R-ver: the unit was copied to registers before the slot release
#else
          const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * 8 + cc) * 512 + lane * 16);
#endif
          const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
          const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
          const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
          const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
          const uint32_t bb[8] = {xr[2 * cc].x, xr[2 * cc].y, xr[2 * cc].z, xr[2 * cc].w,
                                  xr[2 * cc + 1].x, xr[2 * cc + 1].y, xr[2 * cc + 1].z, xr[2 * cc + 1].w};
          const uint4 y0 = x2p[(2 * cc) * 32], y1 = x2p[(2 * cc + 1) * 32];
          const uint32_t b2[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t qw = ww[s];
            const uint32_t M = 0x81C081C0u;
            const uint32_t r0 = qw & M;
            const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;
            const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;
            const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
            const uint32_t A0 = bmul(r0, sg), A1 = bmul(r1, sg8), A2 = bmul(r2, sg), A3 = bmul(r3, sg8);
            mma(acc[s & 1], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);
            mma(acc[2 + (s & 1)], A0, A1, A2, A3, b2[2 * s], b2[2 * s + 1]);
          }
        }
        o2.x = acc[2][0] + acc[3][0]; o2.y = acc[2][1] + acc[3][1]; o2.z = acc[2][2] + acc[3][2]; o2.w = acc[2][3] + acc[3][3];
        acc[2][0] = 0.f; acc[2][1] = 0.f; acc[2][2] = 0.f; acc[2][3] = 0.f; acc[3][0] = 0.f; acc[3][1] = 0.f; acc[3][2] = 0.f; acc[3][3] = 0.f;
"""   # R-vt16: the 2-tile unit (unpack once, 2 n-tiles), shared by both layouts
N_CONS = (r"""      float acc[4][4];
#pragma unroll
      for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
#if VMS_T16
      float4 o2 = make_float4(0.f, 0.f, 0.f, 0.f);
#if VMS_T16E
      if (!two)   // R-vt16 layout: the 1-tile loop falls through, the 2-tile path follows it
#else
      if (two) {   // R-vt16: unpack once, 2 n-tiles (2 MMA chains each)
""" + T2_BODY + r"""      } else
#endif
#endif
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) {
""")
O_CEND = "      float4 o;\n      o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];\n"
N_CEND = ("#if VMS_T16 && VMS_T16E\n      else {   // R-vt16: unpack once, 2 n-tiles (2 MMA chains each)\n" + T2_BODY + "      }\n#endif\n" + O_CEND)
O_RED = "      S.red[bq][warp][lane] = o;\n"
N_RED = "      S.red[bq][warp][lane] = o;\n#if VMS_T16\n      if (two) S.red2[bq][warp][lane] = o2;   // R-vt16\n#endif\n"
O_HRED = ("      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }\n"
          "      __syncwarp();\n"
          "      if (lane == 0) mbar_arrive(&S.pempty[bq]);\n")
N_HRED = ("      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }\n"
          "#if VMS_T16\n"
          "      float4 sm2 = make_float4(0.f, 0.f, 0.f, 0.f);\n"
          "      if (two) {\n"
          "#pragma unroll\n"
          "        for (int w = 0; w < NW; ++w) { const float4 u = S.red2[bq][w][lane]; sm2.x += u.x; sm2.y += u.y; sm2.z += u.z; sm2.w += u.w; }\n"
          "      }\n"
          "#endif\n"
          "      __syncwarp();\n"
          "      if (lane == 0) mbar_arrive(&S.pempty[bq]);\n")
O_HWAIT = "      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));\n"
N_HWAIT = ("#if VMS_T16\n"
           "      const bool two = !(VMS_T16X & 2) && __all_sync(0xffffffffu, S.jC[(q < nA) ? ((a0 + q) >> 9) : ((b0 + (q - nA)) >> 8)] > 8);   // R-vt16: tile 2 of this unit (warp-uniform vote, its smem read under the pfull wait)\n"
           "#endif\n" + O_HWAIT)
O_AEPI = ("            p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);\n"
          "          }\n"
          "        }\n")
N_AEPI = (O_AEPI + "#if VMS_T16\n"
          "        if (two) {   // R-vt16: tokens 8-15\n"
          "#pragma unroll\n"
          "          for (int h = 0; h < 2; ++h) {\n"
          "            const int tk = 8 + 2 * t + h;\n"
          "            if (tk < cnt) {\n"
          "              const float ga = (h ? sm2.y : sm2.x) * gs, up = (h ? sm2.w : sm2.z) * gs;\n"
          "              const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;\n"
          "              p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);\n"
          "            }\n"
          "          }\n"
          "        }\n"
          "#endif\n")
O_BEPI = ("          if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);\n"
          "        }\n")
N_BEPI = (O_BEPI + "#if VMS_T16\n"
          "        if (two) {   // R-vt16: tokens 8-15\n"
          "#pragma unroll\n"
          "          for (int h = 0; h < 2; ++h) {\n"
          "            const int tk = 8 + 2 * t + h;\n"
          "            const float y0 = (h ? sm2.y : sm2.x) * w0, y1 = (h ? sm2.w : sm2.z) * w1;\n"
          "            float sq = y0 * y0 + y1 * y1;\n"
          "            if (tk < cnt) {\n"
          "              p.ybuf[(size_t)(st + tk) * HD + n0] = y0;\n"
          "              p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;\n"
          "            }\n"
          "            sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n"
          "            sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n"
          "            sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n"
          "            if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);\n"
          "          }\n"
          "        }\n"
          "#endif\n")

PATCHES = [("defines", DEF_ANCHOR, DEFS + DEF_ANCHOR, 1), ("smem", O_SMEM, N_SMEM, 1), ("split fn", O_SPLITFN, N_SPLITFN, 1),
           ("nj", O_NJ, N_NJ, 1), ("jobs", O_JOBS, N_JOBS, 1), ("njobs", O_NJOBS, N_NJOBS, 1), ("plan", O_PLAN, N_PLAN, 1), ("x decl", O_XDECL, N_XDECL, 1),
           ("x load", O_XLOAD, N_XLOAD, 1), ("consumer", O_CONS, N_CONS, 1), ("consumer end", O_CEND, N_CEND, 1), ("red write", O_RED, N_RED, 2),
           ("helper vote", O_HWAIT, N_HWAIT, 1), ("helper red", O_HRED, N_HRED, 1), ("A epilogue", O_AEPI, N_AEPI, 1), ("B epilogue", O_BEPI, N_BEPI, 1)]


def apply(s):
    for tag, old, _, n in PATCHES:
        assert s.count(old) == n, f"vt16 anchor '{tag}' found {s.count(old)} times (want {n})"
    for _, old, new, _ in PATCHES:
        s = s.replace(old, new)
    return s


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vt16] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

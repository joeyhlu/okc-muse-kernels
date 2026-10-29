"""vocc_patches.py: VOCC (VMS consumer occupancy, 4 consumer warps per SMSP) compile-time variants of the layer's VMS source
(VMS_CUDA_SRC of k3_gc_20 = k3_sfx_2). Every change sits in #if VMS_OCC blocks; at VMS_OCC 0 the source compiles to the
base's SASS (checked by tools/vocc_local_test.py). OCC builds need the ship's VMS_RO=1 (and VMS_FX=0, VMS_MK2=0).
  VMS_OCC 1 (K16): 16 consumer warps, each takes half of a base warp's K range (4 chunks, x in 32 registers); 16 partials per
     unit for the helper. = R-vms-4's n16 on today's consumer (L2 unpack, RO). Per warp per unit the fixed cost (~134 of 582
     SASS instructions) is paid twice per SMSP.
  VMS_OCC 2 (U2): 16 consumer warps = two 8-warp groups on alternate units (group q & 1), each warp with the base's per-unit
     work and layout (8 chunks, x in 64 registers); per-group named barrier (bar 1 + g) gates the slot refill; each group owns
     the slots / reduction buffers of its parity (NS, NRB even: no parity aliasing between groups). Same instructions per SMSP
     per unit as the base, twice the warps to interleave. ptxas caps a 544-thread CTA at 96 registers (warpgroup-granular
     allocation on sm_90): x (64) + 4 accumulator chains spill, hence
  VMS_ACC n (OCC builds only; 4 = base): independent mma accumulator chains per warp (2 frees 8 registers; the FP32 sum
     order of a unit's 32 k16 steps changes, far below the bf16 output rounding).
Arithmetic per unit, job order and the helper epilogue are unchanged (the ss FP32 atomic order is not deterministic in the base).
"""
NL = chr(10)


def _rep(src, old, new, what):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"[vocc_patches] {what}: expected 1 match, got {n}")
    return src.replace(old, new)


def _wrap(src, old, new, what, cond="VMS_OCC"):
    return _rep(src, old, f"#if {cond}{NL}{new}#else{NL}{old}#endif{NL}", what)


CONSUMER_OCC = r"""  if (warp < NW) {
#if VMS_FX || VMS_MK2 || !VMS_RO
#error "VMS_OCC needs VMS_RO=1, VMS_FX=0, VMS_MK2=0"
#endif
    const int grp = (NGR == 2) ? (warp >> 3) : 0, wq = (NGR == 2) ? (warp & 7) : warp;
    const int gtid = tid - grp * (NWU * 32);
#if __CUDA_ARCH__ >= 900
    if (tid == 0)
      for (int q = 0; q < NS && q < N; ++q) issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], 0);
#else
    for (int q = grp; q < NS && q < N; q += NGR) issue_g(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], gtid);
#endif
    const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);   // opaque 16: the scale build's multiply stays an IMAD
    const int g = lane >> 2, t = lane & 3;
    uint4 xr[2 * NCH];
    int curj = -1;
    for (int q = grp; q < N; q += NGR) {
      const int sl = q % NS;
      const uint32_t par = (uint32_t)((q / NS) & 1);
      const bool PB = q >= nA;
      const int v = PB ? b0 + (q - nA) : a0 + q;
      const int j = PB ? (v >> 8) + 4096 : (v >> 9);
      if (j != curj) {
        curj = j;
        const int jj = PB ? j - 4096 : j;
        const int cnt = S.jC[jj], st = S.jS[jj];
        if (PB) {
          if (mk && tid == 0 && mrow[3] == 0) mrow[3] = gt();
          if (lane == 0) {
            int n = 0;
            while (ld_rlx(p.ctr + 8 + jj) < RBA) {
              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
              __nanosleep(64);
            }
          }
          __syncwarp();
          (void)ld_acq(p.ctr + 8 + jj);
          if (mk && tid == 0 && mrow[4] == 0) mrow[4] = gt();
        }
        if (g < cnt) {
          const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID)
                                : reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + g] >> 3) * HD);
          src += wq * (8 * NCH) + 2 * t;
#pragma unroll
          for (int cc = 0; cc < NCH; ++cc) {
            xr[2 * cc] = PB ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);
            xr[2 * cc + 1] = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);
          }
        } else {
#pragma unroll
          for (int cc = 0; cc < 2 * NCH; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
      mbar_wait(&S.full[sl], par);
      if (mk && tid == 0 && q == 0) mrow[2] = gt();
      const unsigned char* slot = ring + sl * UB;
#if VMS_OCC == 1
      const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UC + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
      const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + wq * 512 + lane * 16);
#endif
      float acc[NACC][4];
#pragma unroll
      for (int s = 0; s < NACC; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
#pragma unroll
      for (int cc = 0; cc < NCH; ++cc) {
        const uint4 w = *reinterpret_cast<const uint4*>(slot + (wq * NCH + cc) * 512 + lane * 16);
        const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
        const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
        const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        const uint32_t bb[8] = {xr[2 * cc].x, xr[2 * cc].y, xr[2 * cc].z, xr[2 * cc].w,
                                xr[2 * cc + 1].x, xr[2 * cc + 1].y, xr[2 * cc + 1].z, xr[2 * cc + 1].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t qw = ww[s];
          const uint32_t M = 0x81C081C0u;
          const uint32_t r0 = qw & M;
          const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;   // UNP L2 layout (tools/unp_patches.py)
          const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;
          const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
          mma(acc[s % NACC], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);
        }
      }
      float4 o;
#if VMS_ACC == 4
      o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
      o.y = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
      o.z = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
      o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
#elif VMS_ACC == 2
      o.x = acc[0][0] + acc[1][0]; o.y = acc[0][1] + acc[1][1]; o.z = acc[0][2] + acc[1][2]; o.w = acc[0][3] + acc[1][3];
#else
      o.x = acc[0][0]; o.y = acc[0][1]; o.z = acc[0][2]; o.w = acc[0][3];
#endif
      const int bq = q & (NRB - 1);
      asm volatile("bar.sync %0, %1;\n" :: "r"(1 + grp), "n"(NWU * 32) : "memory");   // the group's reads of the slot are done
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (gtid == 0) issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], 0);
#else
        issue_g(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], gtid);
#endif
      }
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
      S.red[bq][wq][lane] = o;
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pfull[bq]);
    }
    if (mk && tid == 0) mrow[5] = gt();
"""

CONSTS_OCC = (
    "#if VMS_OCC == 1" + NL +
    "constexpr int NW = 16, NWU = 16, NCH = 4, NGR = 1, NT = NW * 32, NTL = NT + 32, NRB = VMS_NRB, NS = VMS_NS, G = VMS_G;" + NL +
    "#elif VMS_OCC == 2" + NL +
    "constexpr int NW = 16, NWU = 8, NCH = 8, NGR = 2, NT = NW * 32, NTL = NT + 32, NRB = VMS_NRB, NS = VMS_NS, G = VMS_G;" + NL +
    "static_assert(NS % 2 == 0 && NRB % 2 == 0, \"VMS_OCC 2: each group owns the slots / buffers of its parity\");" + NL +
    "#endif" + NL + "#if VMS_OCC" + NL + "constexpr int NACC = VMS_ACC;   // independent accumulator chains per warp (base 4)" + NL +
    "static_assert(NACC == 1 || NACC == 2 || NACC == 4, \"VMS_ACC\");" + NL +
    "#endif" + NL)

ISSUE_G = (
    "#if VMS_OCC" + NL +
    "__device__ __forceinline__ void issue_g(const unsigned char* src, unsigned char* dst, uint64_t* bar, int gtid) {   // sm_86 path" + NL +
    "  for (int i = gtid; i < UB / 16; i += NWU * 32)" + NL +
    "    asm volatile(\"cp.async.cg.shared.global [%0], [%1], 16;\\n\" :: \"r\"(su32(dst + i * 16)), \"l\"(src + i * 16) : \"memory\");" + NL +
    "  asm volatile(\"cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\\n\" :: \"r\"(su32(bar)) : \"memory\");" + NL +
    "}" + NL + "#endif" + NL)


def apply(src):
    s = src
    s = _rep(s, "#ifndef VMS_G" + NL, "#ifndef VMS_OCC" + NL + "#define VMS_OCC 0" + NL + "#endif" + NL + "#ifndef VMS_ACC" + NL +
             "#define VMS_ACC 4" + NL + "#endif" + NL + "#ifndef VMS_G" + NL, "define")
    old = "constexpr int NW = 8, NT = NW * 32, NTL = NT + 32, NRB = VMS_NRB, NS = VMS_NS, G = VMS_G;" + NL
    s = _wrap(s, old, CONSTS_OCC, "consts")
    s = _wrap(s, "  float4 red[NRB][NW][32];" + NL, "  float4 red[NRB][NWU][32];" + NL, "red")
    old = "__device__ __forceinline__ uint32_t scd("
    s = _rep(s, old, ISSUE_G + NL + old, "issue_g")
    s = _wrap(s, "    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], NT);" + NL,
              "    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], NWU * 32);" + NL, "full init sm86")
    s = _wrap(s, "    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], NW); mbar_init(&S.pempty[i], 1); }" + NL,
              "    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], NWU); mbar_init(&S.pempty[i], 1); }" + NL, "pfull init")
    a = s.index("  if (warp < NW) {" + NL + "#if !((VMS_FX & 4) && __CUDA_ARCH__ >= 900)" + NL)
    e = s.index("  } else {" + NL + "    const int g = lane >> 2, t = lane & 3;" + NL + "    float* ssc", a)
    s = s[:a] + "#if VMS_OCC" + NL + CONSUMER_OCC + "#else" + NL + s[a:e] + "#endif" + NL + s[e:]
    hs = "      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }" + NL
    s = _wrap(s, hs, hs.replace("w < NW;", "w < NWU;"), "helper sum")
    s = _wrap(s, "    if (tid >= NT) break;" + NL, "    if (tid >= 256) break;   // one 1024-column item per 256 threads" + NL, "final pass")
    return s


if __name__ == "__main__":
    import pathlib
    import sys
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    L = (ROOT / (sys.argv[1] if len(sys.argv) > 1 else "kernels/muse_decode_layer/k3_gc_20.py")).read_text(encoding="utf-8")
    a = L.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    print(apply(L[a:L.index('"""', a)])[:200])

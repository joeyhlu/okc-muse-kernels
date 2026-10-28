"""vks_patches.py: VMS-KS J1 (notes/study/indep_mid.md s3, R-vks-1 GO for T = 16) as text edits on the layer's VMS source.

A local expert with 9..16 tokens becomes ONE 16-token job over two K-halves (instead of two 8-token full-K jobs):
  * job counts per phase are unchanged (512 A slots, 256 B slots), so CTA ranges, doneA (512 credits) and the final pass stay;
    slot u of a J1 job: A h = u >> 8, pair P = u & 255; B h = u >> 7, P = u & 127 (half-major: a CTA's range mostly holds one
    half, so x loads once per (job, half));
  * slot = K-half h of VMS units 2P, 2P + 1: [codes U0 h 16 KB | codes U1 h 16 KB | scales U0 h 2 KB | scales U1 h 2 KB],
    4 bulk copies (no new prepack);
  * compute warp w: rows of both units (2 m-tiles), K-slice 256 (VMS warp 4 h + w / 2, chunks 4 (w & 1) ..), 16 tokens
    (2 n-tiles), x in the same 16 uint4 (R-vks-1's consume1); 16 accumulators; per-warp partials staged in all of S.red
    (16 KB, free once slot q - 1 is consumed); compute warps 0-3 sum tile f = warp over the 8 warps in fixed order into a
    2 KB buffer per ring entry (S.rk, +8 KB smem), then pfull (R-vks-2: shared fp32 atomicAdd = ATOMS.CAST.SPIN loops);
  * helper: reads S.rk, frees the entry, stores the CTA partial to fx[pair][h] (scratch behind ybuf), fence, atomic
    on fcnt[pair] (behind the doneA counters); the second arriver adds the two halves in h0 + h1 order and runs VMS's A
    (SwiGLU -> inter, +2 doneA credits batched per run) or B (y rows + ss) epilogue, and resets fcnt. The finish of slot q is
    deferred to the next J1 slot of the same run (one fence serves q's release and q - 1's acquire), flushed at run end.
R-vks-4: two helper warps (10 warps keep the 168-register cap) take alternate slots; J0 doneA credits are counted
per helper; the J1 decode is the base's decode (UNP L2 on k3_gc_5).
Everything new sits under #if VMS_KS; VMS_KS = 0 preprocesses back to the input text (check_ks0).
"""

NL = "\n"


def _sub1(t, old, new):
    assert t.count(old) == 1, (old[:100], t.count(old))
    return t.replace(old, new)


DEFS = "#ifndef VMS_KS\n#define VMS_KS 0\n#endif\n"

CONSTS = """#if VMS_KS
constexpr int MAXK1 = 16, FXS = 384;   // J1 jobs per launch (T <= 16 has <= 14), fixup pairs per J1 job (256 A + 128 B)
#endif
"""

SMEM = """#if VMS_KS
  int jK[MAXJ];
  float4 rk[NRB][4][32];
#endif
"""

ISSUE_KS = r"""#if VMS_KS
__device__ __forceinline__ void bulk_ks(unsigned char* dst, const unsigned char* src, uint32_t n, uint64_t* bar, uint64_t pol) {
#if __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(su32(dst)), "l"(src), "r"(n), "r"(su32(bar)), "l"(pol) : "memory");
#endif
}
__device__ __forceinline__ void issue_ks(const Params& p, const Smem& S, int q, int nA, int a0, int b0, unsigned char* dst,
                                         uint64_t* bar, int tid) {
  const bool PB = q >= nA;
  const int v = PB ? b0 + (q - nA) : a0 + q;
  const int j = PB ? (v >> 8) : (v >> 9);
  if (S.jK[j] < 0) { issue(unit_src(p, S, q, nA, a0, b0), dst, bar, tid); return; }
  const int u = PB ? (v & 255) : (v & 511), h = PB ? (u >> 7) : (u >> 8), P = PB ? (u & 127) : (u & 255);
  const unsigned char* b = PB ? p.wB + ((size_t)S.jE[j] * RBB + 2 * P) * UB : p.wA + ((size_t)S.jE[j] * RBA + 2 * P) * UB;
  const unsigned char* s0 = b + h * 16384;
  const unsigned char* s1 = b + UB + h * 16384;
  const unsigned char* s2 = b + UC + h * 2048;
  const unsigned char* s3 = b + UB + UC + h * 2048;
#if __CUDA_ARCH__ >= 900
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(UB) : "memory");
  bulk_ks(dst, s0, 16384, bar, pol);
  bulk_ks(dst + 16384, s1, 16384, bar, pol);
  bulk_ks(dst + 32768, s2, 2048, bar, pol);
  bulk_ks(dst + 34816, s3, 2048, bar, pol);
#else
  for (int i = tid; i < UB / 16; i += NT) {
    const int d = i * 16;
    const unsigned char* src = d < 16384 ? s0 + d : d < 32768 ? s1 + (d - 16384) : d < 34816 ? s2 + (d - 32768) : s3 + (d - 34816);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst + d)), "l"(src) : "memory");
  }
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(su32(bar)) : "memory");
#endif
}
#endif
"""

ROUTE_OLD = """    int inc = tot, nj = (tot + 7) >> 3, jinc = nj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, inc, d), b = __shfl_up_sync(0xffffffffu, jinc, d);
      if (lane >= d) { inc += a; jinc += b; }
    }
    const int off = inc - tot, joff = jinc - nj;
    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + 8 * k; S.jC[joff + k] = min(8, tot - 8 * k); }
"""

ROUTE_NEW = """#if VMS_KS
    const int JT = tot > 8 ? 16 : 8;   // 9..16 tokens: one J1 job over two K-halves
    int inc = tot, nj = (tot + JT - 1) / JT, jinc = nj, nk = tot > 8 ? nj : 0, kinc = nk;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, inc, d), b = __shfl_up_sync(0xffffffffu, jinc, d);
      const int e = __shfl_up_sync(0xffffffffu, kinc, d);
      if (lane >= d) { inc += a; jinc += b; kinc += e; }
    }
    const int off = inc - tot, joff = jinc - nj, koff = kinc - nk;
    for (int k = 0; k < nj; ++k) {
      S.jE[joff + k] = lane; S.jS[joff + k] = off + JT * k; S.jC[joff + k] = min(JT, tot - JT * k);
      S.jK[joff + k] = (tot > 8 && koff + k < MAXK1) ? koff + k : -1;
      if (tot > 8 && koff + k >= MAXK1) atomicExch(p.ctr + 2, 3);
    }
#else
""" + ROUTE_OLD + "#endif\n"

ZERO_RK = None   # R-vks-3: rk is written by compute warps 0-3 (no accumulation), nothing to zero

ISS0_OLD = "      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);\n"
ISS0_NEW = ("#if VMS_KS\n      issue_ks(p, S, q, nA, a0, b0, ring + q * UB, &S.full[q], tid);\n#else\n" + ISS0_OLD + "#endif\n")
ISS1_OLD = "        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);\n"
ISS1_NEW = ("#if VMS_KS\n        issue_ks(p, S, q + NS, nA, a0, b0, ring + sl * UB, &S.full[sl], tid);\n#else\n" + ISS1_OLD + "#endif\n")

CURJ_OLD = "    int curj = -1;\n"
CURJ_NEW = "    int curj = -1;\n#if VMS_KS\n    int curk = 0, prevk = 0;\n#endif\n"

JCHG_OLD = """      if (j != curj) {
        curj = j;
"""
JCHG_NEW = """#if VMS_KS
      const int jk = S.jK[PB ? (v >> 8) : (v >> 9)];
      const int hk = jk < 0 ? 0 : PB ? ((v >> 7) & 1) : ((v >> 8) & 1);
      if (j * 2 + hk != curj) {
        curj = j * 2 + hk;
        curk = jk >= 0;
#else
""" + JCHG_OLD + "#endif\n"

XLD_OLD = "        if (g < cnt) {\n          const uint4* src = PB ?"
XLD_NEW = """#if VMS_KS
        if (curk) {   // J1: tokens g / 8 + g, K-half hk, VMS warp 4 hk + warp / 2, chunks 4 (warp & 1) ..
          const int wo = (4 * hk + (warp >> 1)) * 64 + (warp & 1) * 32 + 2 * t;
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            const int tk = 8 * n + g;
            if (tk < cnt) {
              const uint4* src = (PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + tk) * ID)
                                     : reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + tk] >> 3) * HD)) + wo;
#pragma unroll
              for (int c4 = 0; c4 < 4; ++c4) {
                xr[n * 8 + 2 * c4] = PB ? __ldcg(src + c4 * 8) : __ldg(src + c4 * 8);
                xr[n * 8 + 2 * c4 + 1] = PB ? __ldcg(src + c4 * 8 + 1) : __ldg(src + c4 * 8 + 1);
              }
            } else {
#pragma unroll
              for (int c4 = 0; c4 < 8; ++c4) xr[n * 8 + c4] = make_uint4(0u, 0u, 0u, 0u);
            }
          }
        } else
#endif
        if (g < cnt) {
          const uint4* src = PB ?"""

CMP_HEAD_OLD = "      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16);\n"
CMP_HEAD_NEW = r"""#if VMS_KS
      const int bq = q & (NRB - 1);
      if (curk) {   // J1 slot: 2 m-tiles (units 2P, 2P + 1) x K-slice 256 x 2 n-tiles; R-vks-1 consume1
        float acc[4][4];
#pragma unroll
        for (int f = 0; f < 4; ++f) { acc[f][0] = 0.f; acc[f][1] = 0.f; acc[f][2] = 0.f; acc[f][3] = 0.f; }
        const int wl = warp >> 1, ch = (warp & 1) * 4;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const uint2 sc = *reinterpret_cast<const uint2*>(slot + UC + mt * 2048 + wl * 512 + lane * 16 + 8 * (warp & 1));
#pragma unroll
          for (int cc = 0; cc < 4; ++cc) {
            const uint4 w = *reinterpret_cast<const uint4*>(slot + mt * 16384 + (wl * 8 + ch + cc) * 512 + lane * 16);
            const uint32_t sw = cc < 2 ? sc.x : sc.y;
            const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
            const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
            const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
              const uint32_t qw = ww[s];
              const uint32_t M = 0x81C081C0u;
@@DQ@@
              const uint32_t f0 = bmul(r0, sg), f1 = bmul(r1, sg8), f2 = bmul(r2, sg), f3 = bmul(r3, sg8);
#pragma unroll
              for (int nt = 0; nt < 2; ++nt) {
                const uint4 X0 = xr[nt * 8 + 2 * cc], X1 = xr[nt * 8 + 2 * cc + 1];
                const uint32_t bb[8] = {X0.x, X0.y, X0.z, X0.w, X1.x, X1.y, X1.z, X1.w};
                mma(acc[mt * 2 + nt], f0, f1, f2, f3, bb[2 * s], bb[2 * s + 1]);
              }
            }
          }
        }
        // stage = all of S.red as [warp][f][lane] (16 KB): free once the helper has consumed slot q - 1 (then every J0
        // partial before it is consumed, and warps 0-3 have read slot q - 1's stage before they released it)
        if (q > 0) mbar_wait(&S.pempty[(q - 1) & (NRB - 1)], (uint32_t)(((q - 1) / NRB) & 1));
        float4* stg = &S.red[0][0][0];
#pragma unroll
        for (int f = 0; f < 4; ++f) stg[(warp * 4 + f) * 32 + lane] = make_float4(acc[f][0], acc[f][1], acc[f][2], acc[f][3]);
      } else {
        if (prevk) mbar_wait(&S.pempty[(q - 1) & (NRB - 1)], (uint32_t)(((q - 1) / NRB) & 1));   // previous J1 stage read
#endif
""" + CMP_HEAD_OLD

CMP_TAIL_OLD = "      S.red[bq][warp][lane] = o;\n      cta_sync_compute();\n"
CMP_TAIL_NEW = "      S.red[bq][warp][lane] = o;\n#if VMS_KS\n      }\n#endif\n      cta_sync_compute();\n"

PFULL_OLD = "      if (tid == 0) mbar_arrive(&S.pfull[bq]);\n      if (++sl == NS) { sl = 0; par ^= 1u; }\n"
PFULL_NEW = r"""#if VMS_KS
      if (curk) {
        if (warp < 4) {   // K-slice sum of tile f = warp over the 8 warps' partials (fixed order) -> rk[bq][f]
          const float4* stg = &S.red[0][0][0];
          float4 s = stg[warp * 32 + lane];
#pragma unroll
          for (int w2 = 1; w2 < NW; ++w2) {
            const float4 u = stg[(w2 * 4 + warp) * 32 + lane];
            s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
          }
          mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
          S.rk[bq][warp][lane] = s;
          asm volatile("bar.sync 2, 128;\n" ::: "memory");
          if (tid == 0) mbar_arrive(&S.pfull[bq]);
        }
      } else if (tid == 0) mbar_arrive(&S.pfull[bq]);
      prevk = curk;
      if (++sl == NS) { sl = 0; par ^= 1u; }
#else
""" + PFULL_OLD + "#endif\n"

HLP_HEAD_OLD = """    float* ssc = p.ss + (S.gen & 1) * MAXP;
    for (int q = 0; q < N; ++q) {
      const int bq = q & (NRB - 1);
"""
HLP_HEAD_NEW = r"""    float* ssc = p.ss + (S.gen & 1) * MAXP;
#if VMS_KS
    float4* fxb = reinterpret_cast<float4*>(p.ybuf + (size_t)MAXP * HD);   // fixup partials [J1 job][pair][h][4][32]
    int* fcnt = p.ctr + 8 + MAXJ;                                          // arrivals per fixup pair (reset by the 2nd)
    int pend = 0, pold = 0, pfs = 0, ph = 0, pP = 0, pA = 0, pj = 0, cred = 0;
    float4 ppd[4];
    int c0n = 0;   // J0 A units this helper finished in the current job (credited per helper)
    for (int q = warp - NW; q < N; q += 2) {   // helper h = warp - NW takes slots h, h + 2, ... (R-vks-4)
#else
    for (int q = 0; q < N; ++q) {
#endif
      const int bq = q & (NRB - 1);
#if VMS_KS
      {
        const bool QA = q < nA;
        const int vq = QA ? a0 + q : b0 + (q - nA), jq = QA ? (vq >> 9) : (vq >> 8), kq = S.jK[jq];
        if (kq >= 0) {
          mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
          float4 pp[4];
#pragma unroll
          for (int f = 0; f < 4; ++f) pp[f] = S.rk[bq][f][lane];
          __syncwarp();
          if (lane == 0) mbar_arrive(&S.pempty[bq]);
          const int u = QA ? (vq & 511) : (vq & 255), h = QA ? (u >> 8) : (u >> 7), P = QA ? (u & 255) : (u & 127);
          const int fs = kq * FXS + (QA ? P : 256 + P);
#pragma unroll
          for (int f = 0; f < 4; ++f) __stcg(fxb + (size_t)fs * 256 + (h * 4 + f) * 32 + lane, pp[f]);
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");   // releases this partial; acquires the pending partner
          __syncwarp();
          int oq = 0;
          if (lane == 0) oq = atomicAdd(fcnt + fs, 1);
          bool last = q + 2 >= N || (QA && q + 2 >= nA);
          if (!last) {
            const bool QN = q + 2 < nA;
            const int vn = QN ? a0 + q + 2 : b0 + (q + 2 - nA);
            last = (QN ? (vn >> 9) : (vn >> 8)) != jq;
          }
          for (int pass = 0; pass < 2; ++pass) {
            if (pass == 1) {   // run end: finish this slot now
              if (!last) break;
              asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
            }
            if (pend && __shfl_sync(0xffffffffu, pold, 0) == 1) {   // second arriver of the pending pair: h0 + h1, epilogue
              float4 sm4[4];
#pragma unroll
              for (int f = 0; f < 4; ++f) {
                const float4 o = __ldcg(fxb + (size_t)pfs * 256 + ((1 - ph) * 4 + f) * 32 + lane);
                const float4 a = ph ? o : ppd[f], b = ph ? ppd[f] : o;
                sm4[f] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
              }
              if (lane == 0) fcnt[pfs] = 0;
              const int cnt = S.jC[pj], st = S.jS[pj];
              if (pA) {
                const float gs = __ldg(p.gsA + S.jE[pj]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                  const int rb = 2 * pP + mt;
#pragma unroll
                  for (int nt = 0; nt < 2; ++nt) {
                    const float4 sm = sm4[mt * 2 + nt];
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                      const int tk = 8 * nt + 2 * t + hh;
                      if (tk < cnt) {
                        const float ga = (hh ? sm.y : sm.x) * gs, up = (hh ? sm.w : sm.z) * gs;
                        const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;
                        p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);
                      }
                    }
                  }
                }
                cred += 2;
              } else {
                const float gs = __ldg(p.gsB + S.jE[pj]);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                  const int n0 = (2 * pP + mt) * 16 + g;
                  const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;
#pragma unroll
                  for (int nt = 0; nt < 2; ++nt) {
                    const float4 sm = sm4[mt * 2 + nt];
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                      const int tk = 8 * nt + 2 * t + hh;
                      const float y0 = (hh ? sm.y : sm.x) * w0, y1 = (hh ? sm.w : sm.z) * w1;
                      float sq = y0 * y0 + y1 * y1;
                      if (tk < cnt) {
                        p.ybuf[(size_t)(st + tk) * HD + n0] = y0;
                        p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;
                      }
                      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
                      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
                      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
                      if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);
                    }
                  }
                }
              }
            }
            pend = 1; pold = oq; pfs = fs; ph = h; pP = P; pA = QA; pj = jq;
#pragma unroll
            for (int f = 0; f < 4; ++f) ppd[f] = pp[f];
            if (pass == 1) pend = 0;
          }
          if (last) {
            if (QA && cred) {   // this CTA's doneA credits for job jq (2 per pair it finished)
              asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
              __syncwarp();
              if (lane == 0) atomicAdd(p.ctr + 8 + jq, cred);
            }
            cred = 0;
          }
          continue;
        }
      }
#endif
"""

NTL_OLD = "constexpr int NW = 8, NT = NW * 32, NTL = NT + 32, NRB = 4, NS = VMS_NS, G = VMS_G;\n"
NTL_NEW = ("#if VMS_KS\nconstexpr int NW = 8, NT = NW * 32, NTL = NT + 64, NRB = 4, NS = VMS_NS, G = VMS_G;   // 2 helper warps\n"
           "#else\n" + NTL_OLD + "#endif\n")

CRED_OLD = """        if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units
          __threadfence();
          __syncwarp();
          if (lane == 0) atomicAdd(p.ctr + 8 + j, min(a1, (j + 1) * RBA) - max(a0, j * RBA));
          if (mk && lane == 0 && q == nA - 1) mrow[6] = gt();
        }
"""
CRED_NEW = "#if VMS_KS\n" + r"""        ++c0n;
        if (q + 2 >= nA || ((v + 2) >> 9) != j) {     // this helper's last unit of job j: credit its own units
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
          __syncwarp();
          if (lane == 0) atomicAdd(p.ctr + 8 + j, c0n);
          c0n = 0;
          if (mk && lane == 0 && q + 2 >= nA) mrow[6] = gt();
        }
#else
""" + CRED_OLD + "#endif\n"

DQ_HEAD = "          const uint32_t M = 0x81C081C0u;\n"


def base_decode(src):
    """the base J0 decode lines r0..r3 (after the M line), re-indented for the J1 block"""
    i = src.index(DQ_HEAD) + len(DQ_HEAD)
    lines = src[i:].split("\n")[:4]
    assert [ln.strip().split(" =")[0] for ln in lines] == ["const uint32_t r0", "const uint32_t r1", "const uint32_t r2",
                                                          "const uint32_t r3"], lines
    return "\n".join("    " + ln for ln in lines) + "\n"


def apply_ks(src):
    """src: the layer's VMS CUDA source (k3_gc_4 text between VMS_CUDA_SRC = r\"\"\" and \"\"\")."""
    t = src
    t = _sub1(t, "#ifndef VMS_G\n#define VMS_G 132\n#endif\n", "#ifndef VMS_G\n#define VMS_G 132\n#endif\n" + DEFS)
    t = _sub1(t, "constexpr int WAIT_MAX = 1 << 22;\n", "constexpr int WAIT_MAX = 1 << 22;\n" + CONSTS)
    t = _sub1(t, "  float4 red[NRB][NW][32];\n", "  float4 red[NRB][NW][32];\n" + SMEM)
    t = _sub1(t, "__device__ __forceinline__ uint32_t scd(", ISSUE_KS + "__device__ __forceinline__ uint32_t scd(")
    t = _sub1(t, ROUTE_OLD, ROUTE_NEW)
    t = _sub1(t, ISS0_OLD, ISS0_NEW)
    t = _sub1(t, ISS1_OLD, ISS1_NEW)
    t = _sub1(t, CURJ_OLD, CURJ_NEW)
    t = _sub1(t, JCHG_OLD, JCHG_NEW)
    t = _sub1(t, XLD_OLD, XLD_NEW)
    t = _sub1(t, CMP_HEAD_OLD, CMP_HEAD_NEW)
    t = _sub1(t, CMP_TAIL_OLD, CMP_TAIL_NEW)
    t = _sub1(t, PFULL_OLD, PFULL_NEW)
    t = _sub1(t, HLP_HEAD_OLD, HLP_HEAD_NEW)
    t = _sub1(t, NTL_OLD, NTL_NEW)
    t = _sub1(t, CRED_OLD, CRED_NEW)
    assert t.count("@@DQ@@\n") == 1
    t = t.replace("@@DQ@@\n", base_decode(src))
    return t


def pp_flag(src, name, val):
    """Minimal preprocessor for one flag: resolve #if NAME / #if !NAME / #else / #endif blocks (nesting with other
    #if / #ifdef / #ifndef kept verbatim) and drop the #ifndef NAME default block."""
    src = src.replace(f"#ifndef {name}\n#define {name} 0\n#endif\n", "")
    out, stack = [], []          # stack entries: None (other #if) or [active_branch_bool]
    for ln in src.split(NL):
        s = ln.strip()
        if s == f"#if {name}" or s == f"#if !{name}":
            want = bool(val) if s == f"#if {name}" else not val
            stack.append([want])
            continue
        if s.startswith("#if"):
            stack.append(None)
        elif s == "#else" and stack and stack[-1] is not None:
            stack[-1][0] = not stack[-1][0]
            continue
        elif s.startswith("#endif"):
            top = stack.pop()
            if top is not None:
                continue
        if all(e is None or e[0] for e in stack):
            out.append(ln)
    assert not stack
    return NL.join(out)


def check_ks0(base_src, patched):
    return pp_flag(patched, "VMS_KS", 0) == base_src

"""VML probe 2 (L1 step-0, notes/study/beyond_vjeux.md s3 L1): can the RS-wgmma consumer overlap tensor time with dequant
issue and the weight stream? Same structure as kernels/lab/vml_probe.py (1 CTA/SM, 4 consumer WGs sharing one 36 KB
weight stage + one x stage, 2 producer warps), with compile-time arms:
  PG_GS  k16 steps per commit group (2 = R-vml-1 / public, 4)
  PG_NP  wgmma groups kept in flight (wait_group NP; NP + 1 A-fragment buffers; 1 = R-vml-1, 3)
  PG_RES 0 stream from HBM, 1 resident (weights + x slot 0 loaded once, no ring waits: consumer-only time),
         2 resident + tensor only (constant A fragments, no LDS / dequant: wgmma time alone)
  PG_PP  1: WGs 2 / 3 start half a stage after WGs 0 / 1 (named-barrier handshake on the first stage)
  R-vml-14 (notes/study/fable_review.md s2: is the N-growing stage excess F - C the x path?):
  PG_XR  1: x resident (the CTA's first unit's stage-0 x loaded once into slot 0, never refilled; weights streamed as F;
         results wrong by design, timing only)
  PG_XL  1: x slices by cp.async (LDGSTS, 16 B per lane, all 32 lanes of the x-producer warp, completion through
         cp.async.mbarrier.arrive.noinc) instead of the bulk-copy (TMA) unit; consumers fence.proxy.async after the wait
  PG_XSW 1: x in the 128-byte-swizzled K-major wgmma B layout ([i 4][n][128 B], 16-B chunk q at q ^ (n & 7)) with a
         SWIZZLE_128B descriptor (SBO 1024) instead of the no-swizzle core-matrix layout (SBO 4096, LBO 128)
  PG_WS  1: lockstep break (papers_oct28 #1): every weight stage split into 4 per-WG sub-stages (9 KB: the WG's 8 KB of codes
         + 1 KB of scales) with their own full / empty barriers, filled by lanes 0-3 of the weight-producer warp (one per WG),
         so a WG waits only for its own 9 KB; the x ring stays shared (NX slots of slack between WGs)
  R-vml-15 (per-stage tricks of the public WGS consumer, kernel_diff_oct28 / vm_large s6), compile-time:
  PG_DQ  1: pipe-balanced dequant: the two left shifts of deq4 as multiplies by opaque constants (IMAD.SHL on the FMA pipe
         instead of SHF on the ALU pipe; same bits) and the scale decode's x16 as an opaque IMAD
  PG_SC2 1: one PRMT + one IMAD build both rows' scales (row g low half, row g + 8 high half) and the multiplies use the
         half-broadcast forms (__low2bfloat162 / __high2bfloat162, folded into HMUL2 .H0_H0 / .H1_H1)
  PG_DS  1: x descriptor = slot-0 descriptor + slot offset (no per-stage rebuild)
  PG_CL  1: codes loaded one 4-step group ahead (8 live registers instead of 16), weight slot released after the last load
Stage time = kernel time x G / (units x 16).
"""
import hashlib
import time

import torch

import vml_probe as V1

HEAD = V1.VP_CUDA[:V1.VP_CUDA.index("// ---------------- consumer ----------------")]

BODY = r"""
// ---------------- consumer (probe 2) ----------------
#ifndef PG_GS
#define PG_GS 2
#endif
#ifndef PG_NP
#define PG_NP 1
#endif
#ifndef PG_RES
#define PG_RES 0
#endif
#ifndef PG_PP
#define PG_PP 0
#endif
#ifndef PG_RG
#define PG_RG 0
#endif
#ifndef PG_RC
#define PG_RC 112
#endif
#ifndef PG_W2
#define PG_W2 0
#endif
#ifndef PG_MMA
#define PG_MMA 0
#endif
#ifndef PG_W2NP
#define PG_W2NP 2
#endif
#ifndef PG_XR
#define PG_XR 0
#endif
#ifndef PG_XL
#define PG_XL 0
#endif
#ifndef PG_XSW
#define PG_XSW 0
#endif
#ifndef PG_WS
#define PG_WS 0
#endif
#ifndef PG_DQ
#define PG_DQ 0
#endif
#ifndef PG_SC2
#define PG_SC2 0
#endif
#ifndef PG_DS
#define PG_DS 0
#endif
#ifndef PG_CL
#define PG_CL 0
#endif
constexpr bool XFIX = PG_RES || PG_XR;   // x slot 0 for every stage, no x ring
__device__ __forceinline__ void deq4_dq(uint32_t w, uint32_t* o, uint32_t k64, uint32_t k4) {   // == deq4 (same bits)
  o[0] = w & 0x81C081C0u;
  o[1] = (w * k64) & 0x81C081C0u;                                  // rotl 6: the mask keeps no wrapped bit
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w * k4) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {   // a x (b.lo, b.lo)
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {   // a x (b.hi, b.hi)
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
// one dequant half group under the R-vml-15 switches: w0 = row g, w8 = row g + 8 (2 k16 steps each)
__device__ __forceinline__ void dec_half2(uint32_t w0, uint32_t w8, uint32_t sw2, uint32_t par, uint32_t* hh, uint32_t k16,
                                          uint32_t k64, uint32_t k4) {
  if constexpr (PG_DQ) { deq4_dq(w0, hh, k64, k4); deq4_dq(w8, hh + 4, k64, k4); }
  else { deq4(w0, hh); deq4(w8, hh + 4); }
  if constexpr (PG_SC2) {
    const uint32_t sb = 2u * par;
    const uint32_t sgp = __byte_perm(sw2, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16 + 0x75807580u;
#pragma unroll
    for (int j = 0; j < 4; ++j) { hh[j] = bmul2l(hh[j], sgp); hh[4 + j] = bmul2h(hh[4 + j], sgp); }
  } else {
    const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * k16 + 0x75807580u;
    const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * k16 + 0x75807580u;
#pragma unroll
    for (int j = 0; j < 4; ++j) { hh[j] = bmul2(hh[j], Sg); hh[4 + j] = bmul2(hh[4 + j], S8); }
  }
}
constexpr bool TRICKS = PG_DQ || PG_SC2 || PG_DS || PG_CL;
constexpr int WSB = 2 * (NS + NX);       // PG_WS barriers after the shared ones: full [q][slot], then empty [q][slot]
constexpr int SMEM2 = SMEM + (PG_WS ? 8 * 8 * NS : 0);
__device__ __forceinline__ void prod_w_ws(const Args& A, uint32_t ring, uint32_t bars, int u0, int u1, int q) {
  const uint64_t pol = policy_evict_first();
  int it = 0;
  for (int u = u0; u < u1; ++u) {
    const int2 un = A.unit[u];
    const int e = A.job[un.x].x;
    const uint8_t* wb = A.ws13 + ((size_t)e * 64 + 2 * un.y) * 16 * SW;
    for (int s = 0; s < 16; ++s, ++it) {
      const int sl = it % NS;
      if (it >= NS) mbar_wait(bars + 8 * (WSB + 4 * NS + q * NS + sl), ((it / NS) & 1) ^ 1, A.err);
      const uint32_t fb = bars + 8 * (WSB + q * NS + sl), dst = ring + sl * US + q * 9216;
      const uint8_t* blk = wb + (size_t)((q >> 1) * 16 + s) * SW;
      mbar_expect_tx(fb, 9216);
      bulk_g2s_ef(dst, blk + (q & 1) * 8192, 8192, fb, pol);
      bulk_g2s_ef(dst + 8192, blk + 16384 + (q & 1) * 1024, 1024, fb, pol);
    }
  }
}
__device__ __forceinline__ uint64_t wg_desc_sw(uint32_t a) {   // K-major SWIZZLE_128B: LBO 16 B (unused), SBO 1024 B
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024u >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(b) : "memory");
}
__device__ __forceinline__ void prod_x_l(const Args& A, uint32_t xring, uint32_t bars, int u0, int u1, int lane) {
  int it = 0;
  for (int u = u0; u < u1; ++u) {
    const int4 jb = A.job[A.unit[u].x];
    const uint32_t bytes = (uint32_t)jb.z * 512u;
    const uint8_t* xs = A.xp + (size_t)jb.w;
    for (int s = 0; s < 16; ++s, ++it) {
      const int sl = it % NX;
      if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
      const uint8_t* src = xs + (size_t)s * bytes;
      const uint32_t dst = xring + sl * XS;
      for (uint32_t c = lane; c < bytes / 16; c += 32) cp_async16(dst + 16 * c, src + 16 * c);
      cp_async_mbar_arrive(bars + 8 * (2 * NS + sl));
    }
  }
}
constexpr int NCW2 = PG_W2 ? 8 : NCW;   // W2: 2 consumer WGs x 128 rows (two m64 accumulator chains per WG)
constexpr int NTH2 = PG_W2 ? 32 * 8 + 64 : (PG_RG ? 32 * NCW + 128 : NTH);
constexpr int GS = PG_GS, NPD = PG_NP, NB = PG_NP + 1, GPS = 16 / PG_GS, RES = PG_RES;
static_assert(GPS % NB == 0, "groups per stage must be a multiple of the A-buffer count");
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }

__device__ __forceinline__ void dec_half(uint32_t w0, uint32_t w8, uint32_t Sg, uint32_t S8, uint32_t* hh) {
  deq4(w0, hh); deq4(w8, hh + 4);
#pragma unroll
  for (int j = 0; j < 4; ++j) { hh[j] = bmul2(hh[j], Sg); hh[4 + j] = bmul2(hh[4 + j], S8); }
}
template <int N, bool Z>
__device__ __forceinline__ void mma2(float* d, const uint32_t* hh, uint32_t dlo, uint32_t dhi, uint32_t step2 = 16) {
  if constexpr (Z) wgmma_zero<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  else wgmma_acc<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + step2, dhi);
}

template <int N>
__device__ __forceinline__ void stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
                                      uint32_t (&hb)[NB][4 * GS], int q, int wp, int lane, bool first, bool pp_arrive) {
  const int ws = RES ? 0 : it % NS, xs = XFIX ? 0 : it % NX;
  if constexpr (!RES) mbar_wait(bars + 8 * (PG_WS ? WSB + q * NS + ws : ws), (it / NS) & 1, A.err);
  const uint8_t* st = PG_WS ? smem + ws * US + q * 9216 : smem + ws * US + (q >> 1) * SW;
  const int tile = (q & 1) * 4 + wp;
  uint4 c[4];
  uint32_t sw0 = 0, sw1 = 0;
  const uint4* cp = reinterpret_cast<const uint4*>(st + (PG_WS ? wp : tile) * 2048) + lane;
  if constexpr (RES < 2) {
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + (PG_WS ? 8192 + wp * 256 : 16384 + tile * 256)) + lane;
    if constexpr (PG_CL) c[0] = cp[0];
    else {
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
    }
    sw0 = sp[0]; sw1 = sp[32];
  }
  const uint32_t zo = (uint32_t)(reinterpret_cast<uintptr_t>(A.inter) >> 63);   // 0, opaque to the compiler
  const uint32_t k16 = PG_DQ ? 16u + zo : 16u, k64 = 64u + zo, k4 = 4u + zo;
  if constexpr (!RES) {
    __syncwarp();
    if (!PG_CL && lane == 0) mbar_arrive(bars + 8 * (PG_WS ? WSB + 4 * NS + q * NS + ws : NS + ws));
    if constexpr (!PG_XR) mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
    if constexpr (PG_XL) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncwarp();
  }
  const uint64_t d0 = PG_XSW ? wg_desc_sw(sa(smem + NS * US + xs * XS))
                              : wg_desc_sa(sa(smem + NS * US + (PG_DS ? 0 : xs * XS)), 4096u);
  const uint32_t dlo = PG_DS ? (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4) : (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
  for (int gi = 0; gi < GPS; ++gi) {
    uint32_t* hh = hb[gi % NB];
    if constexpr (RES < 2 && TRICKS) {
      static_assert(GS == 2, "R-vml-15 switches: GS 2 only");
      const int i = gi >> 1, hf = gi & 1;
      if constexpr (PG_CL) {
        if (hf == 0 && i < 3) c[i + 1] = cp[32 * (i + 1)];   // the next 4-step group's codes, one group ahead
      }
      const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
      dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16, k64, k4);
      if constexpr (PG_CL && !RES) {
        if (gi == 6) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (PG_WS ? WSB + 4 * NS + q * NS + ws : NS + ws)); }
      }
    } else if constexpr (RES < 2) {
      const int i = (GS == 2) ? (gi >> 1) : gi;
      const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
      if constexpr (GS == 2) {
        const int hf = gi & 1;
        dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
      } else {
        dec_half(c[i].x, c[i].z, Sg, S8, hh);
        dec_half(c[i].y, c[i].w, Sg, S8, hh + 8);
      }
    } else {
#pragma unroll
      for (int k = 0; k < 4 * GS; ++k) hh[k] = 0x3C003C00u ^ (uint32_t)(lane + it + gi + k);
    }
    wg_fence();
#pragma unroll
    for (int h2 = 0; h2 < GS / 2; ++h2) {
      const uint32_t kk = (uint32_t)(gi * GS + 2 * h2);   // k16 step (even); SW128: k64 block kk >> 2, 32 B per step inside
      const uint32_t off = PG_XSW ? (kk >> 2) * (uint32_t)N * 8u + (kk & 3u) * 2u : 16u * kk;
      const uint32_t st2 = PG_XSW ? 2u : 16u;
      if (first && gi == 0 && h2 == 0) mma2<N, true>(d, hh + 8 * h2, dlo + off, dhi, st2);
      else mma2<N, false>(d, hh + 8 * h2, dlo + off, dhi, st2);
    }
    wg_commit();
    wg_wait<NPD>();
    if constexpr (!XFIX) {
      if (gi == NPD - 1 && prevx >= 0) {   // the previous stage's last group retired: its x slot is free
        if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
        prevx = -1;
      }
    }
    if constexpr (PG_PP) {
      if (pp_arrive && gi == GPS / 2 - 1) bar_arrive_n(9, 512);
    }
  }
  if constexpr (!XFIX) prevx = xs;
  ++it;
}

template <int N>
__device__ __forceinline__ void unit_run(const Args& A, uint8_t* smem, uint32_t bars, int& it, int u, int q, int wp, int lane,
                                         bool firstunit) {
  float d[N / 2];
  uint32_t hb[NB][4 * GS];
  int prevx = -1;
  if constexpr (PG_PP) {
    if (firstunit && q >= 2) bar_sync_n(9, 512);
  }
  stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, PG_PP && firstunit && q < 2);
#pragma unroll 1
  for (int s = 1; s < 16; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, false);
  wg_wait<0>();
  if constexpr (!RES) {
    if (!XFIX && lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
    const int2 un = A.unit[u];
    const int4 jb = A.job[un.x];
    const float gs = __ldg(A.gsc13 + jb.x);
    const int g = lane >> 2, t = lane & 3;
    const int col = 8 * ((2 * un.y + (q >> 1)) * 8 + (q & 1) * 4 + wp) + g;
    const int rb = A.jbase[un.x];
#pragma unroll
    for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nt + 2 * t + hh;
        if (n < jb.y)
          A.inter[(size_t)(rb + n) * KDIM + col] = __float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
      }
  } else {
    if (d[0] == 1234.5f) A.err[3] = 1;
  }
}

// ---------------- W2: 2 consumer WGs, each 128 rows of the 256-row stage as two independent m64 chains ----------------
template <int N>
__device__ __forceinline__ void stage_w2(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* dA, float* dB,
                                         uint32_t (&hA)[2][8], uint32_t (&hB)[2][8], int q, int wp, int lane, bool first) {
  const int ws = RES ? 0 : it % NS, xs = RES ? 0 : it % NX;
  if constexpr (!RES) mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  const uint8_t* st = smem + ws * US + q * SW;
  uint4 ca[4], cb[4];
  uint32_t sa0 = 0, sa1 = 0, sb0 = 0, sb1 = 0;
  if constexpr (RES < 2) {
    const uint4* pa = reinterpret_cast<const uint4*>(st + wp * 2048) + lane;
    const uint4* pb = reinterpret_cast<const uint4*>(st + (wp + 4) * 2048) + lane;
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(st + 16384 + wp * 256) + lane;
    const uint32_t* qb = reinterpret_cast<const uint32_t*>(st + 16384 + (wp + 4) * 256) + lane;
#pragma unroll
    for (int i = 0; i < 4; ++i) { ca[i] = pa[32 * i]; cb[i] = pb[32 * i]; }
    sa0 = qa[0]; sa1 = qa[32]; sb0 = qb[0]; sb1 = qb[32];
  }
  if constexpr (!RES) {
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
    mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
    __syncwarp();
  }
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);
  const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
  for (int gi = 0; gi < 8; ++gi) {
    const int i = gi >> 1, hf = gi & 1, par = i & 1;
    const uint32_t off = 16u * (uint32_t)(2 * gi);
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      uint32_t* hh = ch ? hB[gi & 1] : hA[gi & 1];
      float* d = ch ? dB : dA;
      if constexpr (RES < 2) {
        const uint32_t sw2 = ch ? ((i < 2) ? sb0 : sb1) : ((i < 2) ? sa0 : sa1);
        const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
        const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
        const uint4 cc = ch ? cb[i] : ca[i];
        dec_half(hf ? cc.y : cc.x, hf ? cc.w : cc.z, Sg, S8, hh);
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) hh[k] = 0x3C003C00u ^ (uint32_t)(lane + it + gi + k + 8 * ch);
      }
      wg_fence();
      if (first && gi == 0) mma2<N, true>(d, hh, dlo + off, dhi);
      else mma2<N, false>(d, hh, dlo + off, dhi);
      wg_commit();
    }
    wg_wait<PG_W2NP>();
    if constexpr (!RES) {
      if (gi == 0 && prevx >= 0) {
        if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
        prevx = -1;
      }
    }
  }
  if constexpr (!RES) prevx = xs;
  ++it;
}
template <int N>
__device__ __forceinline__ void unit_run_w2(const Args& A, uint8_t* smem, uint32_t bars, int& it, int u, int q, int wp, int lane) {
  float dA[N / 2], dB[N / 2];
  uint32_t hA[2][8], hB[2][8];
  int prevx = -1;
  stage_w2<N>(A, smem, bars, it, prevx, dA, dB, hA, hB, q, wp, lane, true);
#pragma unroll 1
  for (int s = 1; s < 16; ++s) stage_w2<N>(A, smem, bars, it, prevx, dA, dB, hA, hB, q, wp, lane, false);
  wg_wait<0>();
  if constexpr (!RES) {
    if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
    const int2 un = A.unit[u];
    const int4 jb = A.job[un.x];
    const float gs = __ldg(A.gsc13 + jb.x);
    const int g = lane >> 2, t = lane & 3;
    const int rb = A.jbase[un.x];
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const float* d = ch ? dB : dA;
      const int col = 8 * ((2 * un.y + q) * 8 + wp + 4 * ch) + g;
#pragma unroll
      for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int n = 8 * nt + 2 * t + hh;
          if (n < jb.y)
            A.inter[(size_t)(rb + n) * KDIM + col] = __float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
        }
    }
  } else {
    if (dA[0] == 1234.5f || dB[0] == 1234.5f) A.err[3] = 1;
  }
}

// ---------------- M: mma.sync m16n8k16 consumer (every warp its own 16-row tile, no warpgroup sync, B by ldmatrix) -------
__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
template <int N>
__device__ __forceinline__ void stage_m(const Args& A, uint8_t* smem, uint32_t bars, int& it, float (&d)[2][N / 2], int warp,
                                        int lane) {
  constexpr int NT = N / 8;
  const int ws = it % NS, xs = it % NX;
  mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  const uint8_t* st = smem + ws * US + (warp >> 3) * SW;
  const int tile = warp & 7;
  const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
  uint4 c[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
  const uint32_t sw0 = sp[0], sw1 = sp[32];
  __syncwarp();
  if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
  mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
  __syncwarp();
  // x slot [n8][i 4][q 8][r 8][16 B]: ldmatrix.x4 of core matrices q = 4 hf .. 4 hf + 3 of group i = steps 2 hf, 2 hf + 1
  const uint32_t xb = sa(smem + NS * US + xs * XS) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
    const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
    const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t hh[8];
      dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        uint32_t b0, b1, b2, b3;
        ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
        mma16816(&d[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);
        mma16816(&d[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);
      }
    }
  }
  __syncwarp();
  if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
  ++it;
}
template <int N>
__device__ __forceinline__ void unit_run_m(const Args& A, uint8_t* smem, uint32_t bars, int& it, int u, int warp, int lane) {
  float d[2][N / 2];
#pragma unroll
  for (int k = 0; k < N / 2; ++k) { d[0][k] = 0.f; d[1][k] = 0.f; }
#pragma unroll 1
  for (int s = 0; s < 16; ++s) stage_m<N>(A, smem, bars, it, d, warp, lane);
  const int2 un = A.unit[u];
  const int4 jb = A.job[un.x];
  const float gs = __ldg(A.gsc13 + jb.x);
  const int g = lane >> 2, t = lane & 3;
  const int col = 8 * ((2 * un.y + (warp >> 3)) * 8 + (warp & 7)) + g;
  const int rb = A.jbase[un.x];
#pragma unroll
  for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int n = 8 * nt + 2 * t + hh;
      if (n < jb.y)
        A.inter[(size_t)(rb + n) * KDIM + col] = __float2bfloat16_rn(swiglu((d[0][4 * nt + hh] + d[1][4 * nt + hh]) * gs,
                                                                           (d[0][4 * nt + 2 + hh] + d[1][4 * nt + 2 + hh]) * gs));
    }
}

__global__ void __launch_bounds__(NTH2, 1) vp2_kernel(const Args A) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = sa(smem + BAR_OFF);
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW2); }
    for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), PG_XL ? 32 : 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW2); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if constexpr (PG_WS) {
      for (int i = 0; i < 4 * NS; ++i) { mbar_init(bars + 8 * (WSB + i), 1); mbar_init(bars + 8 * (WSB + 4 * NS + i), 4); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (A.marks) A.marks[(size_t)blockIdx.x * 8] = gtime();
  }
  __syncthreads();
  const int2 rg = A.cta[blockIdx.x];
  if (warp >= NCW2) {
    if constexpr (PG_RG && !PG_W2) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
      if (warp >= NCW + 2) return;
    }
    if (lane == 0) {
      if constexpr (RES) {   // one weight stage + one x stage, loaded once
        const int2 un = A.unit[rg.x];
        const int4 jb = A.job[un.x];
        if (warp == NCW2) {
          const uint8_t* wb = A.ws13 + ((size_t)jb.x * 64 + 2 * un.y) * 16 * SW;
          mbar_expect_tx(bars, US);
          bulk_g2s(sa(smem), wb, SW, bars);
          bulk_g2s(sa(smem) + SW, wb + (size_t)16 * SW, SW, bars);
        } else {
          const uint32_t bytes = (uint32_t)jb.z * 512u;
          mbar_expect_tx(bars + 8 * (2 * NS), bytes);
          bulk_g2s(sa(smem + NS * US), A.xp + (size_t)jb.w, bytes, bars + 8 * (2 * NS));
        }
      } else if constexpr (PG_XR) {   // weights streamed, x stage 0 of the first unit once into slot 0
        if (warp == NCW2) { if constexpr (!PG_WS) prod_w(A, sa(smem), bars, rg.x, rg.y); }
        else {
          const int4 jb = A.job[A.unit[rg.x].x];
          const uint32_t bytes = (uint32_t)jb.z * 512u;
          mbar_expect_tx(bars + 8 * (2 * NS), bytes);
          bulk_g2s(sa(smem + NS * US), A.xp + (size_t)jb.w, bytes, bars + 8 * (2 * NS));
        }
      } else {
        if (warp == NCW2) { if constexpr (!PG_WS) prod_w(A, sa(smem), bars, rg.x, rg.y); }
        else if constexpr (!PG_XL) prod_x(A, sa(smem + NS * US), bars, rg.x, rg.y);
      }
    }
    if constexpr (PG_WS && !RES) {
      if (warp == NCW2 && lane < 4) prod_w_ws(A, sa(smem), bars, rg.x, rg.y, lane);
    }
    if constexpr (PG_XL && !RES) {
      if (warp == NCW2 + 1) prod_x_l(A, sa(smem + NS * US), bars, rg.x, rg.y, lane);
    }
    return;
  }
  if constexpr (PG_RG && !PG_W2) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(PG_RC) : "memory");
  if constexpr (RES) {
    mbar_wait(bars, 0, A.err);
    mbar_wait(bars + 8 * (2 * NS), 0, A.err);
    __syncwarp();
  } else if constexpr (PG_XR) {
    mbar_wait(bars + 8 * (2 * NS), 0, A.err);
    __syncwarp();
  }
  const int q = warp >> 2, wp = warp & 3;
  int it = 0;
  if constexpr (PG_MMA) {
    for (int u = rg.x; u < rg.y; ++u) {
      switch (A.job[A.unit[u].x].z >> 3) {
        case 1: unit_run_m<8>(A, smem, bars, it, u, warp, lane); break;
        case 2: unit_run_m<16>(A, smem, bars, it, u, warp, lane); break;
        default: unit_run_m<32>(A, smem, bars, it, u, warp, lane); break;
      }
    }
  } else if constexpr (PG_W2) {
    for (int u = rg.x; u < rg.y; ++u) {
      switch (A.job[A.unit[u].x].z >> 3) {
        case 1: unit_run_w2<8>(A, smem, bars, it, u, q, wp, lane); break;
        case 2: unit_run_w2<16>(A, smem, bars, it, u, q, wp, lane); break;
        case 4: unit_run_w2<32>(A, smem, bars, it, u, q, wp, lane); break;
        case 6: unit_run_w2<48>(A, smem, bars, it, u, q, wp, lane); break;
        default: unit_run_w2<64>(A, smem, bars, it, u, q, wp, lane); break;
      }
    }
  } else
  for (int u = rg.x; u < rg.y; ++u) {
    const int np = A.job[A.unit[u].x].z;
    const bool fu = u == rg.x;
    switch (np >> 3) {
      case 1: unit_run<8>(A, smem, bars, it, u, q, wp, lane, fu); break;
      case 2: unit_run<16>(A, smem, bars, it, u, q, wp, lane, fu); break;
      case 3: unit_run<24>(A, smem, bars, it, u, q, wp, lane, fu); break;
      case 4: unit_run<32>(A, smem, bars, it, u, q, wp, lane, fu); break;
      case 6: unit_run<48>(A, smem, bars, it, u, q, wp, lane, fu); break;
      default: unit_run<64>(A, smem, bars, it, u, q, wp, lane, fu); break;
    }
  }
  if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 8 + 1 + q] = gtime();
}
}  // namespace vp

std::vector<int64_t> vp2_attrs() {
  cudaFuncSetAttribute(vp::vp2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vp::SMEM2);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, vp::vp2_kernel);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vp::vp2_kernel, vp::NTH2, vp::SMEM2);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vp::SMEM2, occ};
}

void vp2_run(torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor job, torch::Tensor jbase, torch::Tensor unit,
             torch::Tensor cta, torch::Tensor xp, torch::Tensor inter, torch::Tensor err, torch::Tensor marks, int64_t G) {
  vp::Args a;
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>();
  a.job = reinterpret_cast<const int4*>(job.data_ptr());
  a.jbase = jbase.data_ptr<int>();
  a.unit = reinterpret_cast<const int2*>(unit.data_ptr());
  a.cta = reinterpret_cast<const int2*>(cta.data_ptr());
  a.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr());
  a.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  a.err = err.data_ptr<int>();
  a.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  cudaFuncSetAttribute(vp::vp2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vp::SMEM2);
  vp::vp2_kernel<<<(int)G, vp::NTH2, vp::SMEM2, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
CUDA = HEAD + BODY
CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> vp2_attrs();
void vp2_run(torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor job, torch::Tensor jbase, torch::Tensor unit,
             torch::Tensor cta, torch::Tensor xp, torch::Tensor inter, torch::Tensor err, torch::Tensor marks, int64_t G);
"""
ARMS = {"F": {}, "PP": {"PG_PP": 1}, "C": {"PG_RES": 1},
        "W2": {"PG_W2": 1, "PG_W2NP": 2}, "W2a": {"PG_W2": 1, "PG_W2NP": 1}, "M": {"PG_MMA": 1},
        "XR": {"PG_XR": 1}, "XL": {"PG_XL": 1}, "XSW": {"PG_XSW": 1}, "XLSW": {"PG_XL": 1, "PG_XSW": 1},
        "WS": {"PG_WS": 1}, "WSXR": {"PG_WS": 1, "PG_XR": 1},
        "DQ": {"PG_DQ": 1}, "SC2": {"PG_SC2": 1}, "DS": {"PG_DS": 1}, "CL": {"PG_CL": 1},
        "ALL": {"PG_DQ": 1, "PG_SC2": 1, "PG_DS": 1, "PG_CL": 1}, "CALL": {"PG_DQ": 1, "PG_SC2": 1, "PG_DS": 1, "PG_RES": 1}}
NOCHECK = ("PG_RES", "PG_XR")   # timing-only arms (results wrong by design)


def to_sw128(xp, npad):
    """K2A core-matrix x stages ([s][n8][i 4][q 8][r 8][16 B]) -> the SW128 K-major layout of PG_XSW per stage:
    [s][i 4][n = 8 n8 + r][chunk p 8][16 B] with chunk p holding core matrix q = p ^ r (bytes only, lossless)."""
    X = xp.view(-1, 16, npad // 8, 4, 8, 8, 16).permute(0, 1, 3, 2, 5, 4, 6)   # [e][s][i][n8][r][q][16]
    r = torch.arange(8, device=xp.device)
    qidx = (torch.arange(8, device=xp.device)[None, :] ^ r[:, None])           # [r][p] -> q
    Y = X[:, :, :, :, r[:, None], qidx, :]                                       # [e][s][i][n8][r][p][16]
    return Y.contiguous().view(torch.uint8).reshape(-1)


def flags_of(arm):
    return V1.FLAGS + [f"-D{k}={v}" for k, v in ARMS[arm].items()]


def build(arm, verbose=False):
    from torch.utils.cpp_extension import load_inline
    fl = flags_of(arm)
    tag = hashlib.sha1((CPP + CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vml_probe2_{arm}_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["vp2_run", "vp2_attrs"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0

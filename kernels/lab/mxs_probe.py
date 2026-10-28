"""mxs_probe.py: step-0 probe for MXS (notes/study/indep_mid.md s4), mixed hot / cold VML stages.

Question. VML runs every 36 KB stage (two 128-row K2A blocks x 256 K) for ONE job: a hot job (N 40-64) is tensor-bound
(stage 1.7-2.0 us), a cold job (N 8, mma.sync L1b) is stream-bound (1.12 us), one after the other. If the two 128-row blocks
of a stage come from a hot job (WGs 0-1, RS-wgmma m64nNk16) and a cold job (WGs 2-3, N 8), the hot pair's tensor time can
overlap the cold pair's dequant and the stream inside the same SM. Is a mixed stage cheaper than the mean of a pure hot
and a pure cold stage?

Structure: the vml_probe.py consumer (R-vml-1, kernels/lab/vml_probe.py MODE 0) re-written torch-free; 1 CTA/SM, 576
threads = 4 consumer WGs + weight producer warp + x producer warp; NS = 3 weight slots (2 x 18,432 B: block of pair 0 at
offset 0, block of pair 1 at 18,432) and NX = 3 x slots of 64 rows (pair 0's job x at offset 0, pair 1's after it when the
jobs differ). WG q works on block q >> 1, tile (q & 1) * 4 + warp (as VML). The cold pair runs RS-wgmma N 8
(MXS_COLD_MMA 0) or mma.sync m16n8k16 with ldmatrix B from the x slot (MXS_COLD_MMA 1, = VML's L1b stage_m).
Unit = (hot job, hot block, cold job, cold block) x 16 stages; pure units have both blocks from the same job.
Arms (plans, same builds): PH = pure hot units, PC = pure cold units, MIX = one hot + one cold block per unit, all with
the same G x U units of distinct weight blocks; FP32 check of every unit's SwiGLU output; per-CTA clock marks.
Decision: r = MIX / ((PH + PC) / 2) on the faster cold build; GO <= 0.87, KILL >= 0.95 (s4.4).
"""
from __future__ import annotations

import hashlib
import statistics
import time


def _wg_spec(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
    body = ("{\\n.reg .b64 dd;\\nmov.b64 dd, {%%%d, %%%d};\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 "
            "{%s}, {%s}, dd, %d, 1, 1, 0;\\n}\\n")
    s = ""
    for nm, sd, cons in (("wgmma_acc", 1, plus), ("wgmma_zero", 0, eq)):
        s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
              "uint32_t a3, uint32_t lo, uint32_t hi) {\n  asm volatile(\"" % (nm, n)
              + body % (na + 4, na + 5, n, outs, a, sd)
              + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}\n" % cons)
    return s


DEV_SRC = r"""
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef MXS_COLD_MMA
#define MXS_COLD_MMA 1
#endif
namespace mxs {
constexpr int KDIM = 4096;
constexpr int SW = 18432;                  // K2A ws13 stage of one 128-row block: [tile 8][kb 4][lane][16 B] + 2 KB scales
constexpr int US = 2 * SW;
constexpr int NS = 3, NX = 3;
constexpr int XS = 64 * 512;               // x slot: <= 64 rows x 256 K bf16 (both jobs of a mixed unit)
constexpr int NCW = 16;
constexpr int NTH = 32 * NCW + 64;
constexpr int BAR_OFF = NS * US + NX * XS;
constexpr int SMEM = BAR_OFF + 8 * 2 * (NS + NX);
constexpr long long WAIT_BUDGET = 1ll << 31;

struct Args {
  const uint8_t* ws13; const float* gsc13;
  const int4* job;     // expert, ntok, npad, x byte offset
  const int* jbase;    // first inter row of the job
  const int4* unit;    // (hot job, hot block, cold job, cold block); pure units: the same job twice
  const int2* cta;     // [u0, u1) per CTA
  const uint8_t* xp; __nv_bfloat16* inter; int* err; long long* marks;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {   // bounded, no trap (R9)
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  int n = 0;
  while (!mbar_try(b, par)) {
    if ((++n & 255) == 0) {
      if (*reinterpret_cast<volatile int*>(err) != 0) return;
      if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); return; }
    }
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_acc(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <int N> __device__ __forceinline__ void wgmma_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
@WGSPECS@
__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // K2A BITPOS3 decode (vml_probe)
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
__device__ __forceinline__ void dec_half(uint32_t w0, uint32_t w8, uint32_t Sg, uint32_t S8, uint32_t* hh) {
  deq4(w0, hh); deq4(w8, hh + 4);
#pragma unroll
  for (int j = 0; j < 4; ++j) { hh[j] = bmul2(hh[j], Sg); hh[4 + j] = bmul2(hh[4 + j], S8); }
}
template <int N, bool Z>
__device__ __forceinline__ void mma2(float* d, const uint32_t* hh, uint32_t dlo, uint32_t dhi) {
  if constexpr (Z) wgmma_zero<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  else wgmma_acc<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + 16, dhi);
}

// ---------------- producers (lane 0 of warps 16 / 17)
__device__ __forceinline__ void prod_w(const Args& A, uint32_t ring, uint32_t bars, int u0, int u1) {
  const uint64_t pol = policy_evict_first();
  int it = 0;
  for (int u = u0; u < u1; ++u) {
    const int4 un = A.unit[u];
    const uint8_t* wh = A.ws13 + ((size_t)A.job[un.x].x * 64 + un.y) * 16 * SW;
    const uint8_t* wc = A.ws13 + ((size_t)A.job[un.z].x * 64 + un.w) * 16 * SW;
    for (int s = 0; s < 16; ++s, ++it) {
      const int sl = it % NS;
      if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
      const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
      mbar_expect_tx(fb, US);
      bulk_g2s_ef(dst, wh + (size_t)s * SW, SW, fb, pol);
      bulk_g2s_ef(dst + SW, wc + (size_t)s * SW, SW, fb, pol);
    }
  }
}
__device__ __forceinline__ void prod_x(const Args& A, uint32_t xring, uint32_t bars, int u0, int u1) {
  int it = 0;
  for (int u = u0; u < u1; ++u) {
    const int4 un = A.unit[u];
    const int4 jh = A.job[un.x], jc = A.job[un.z];
    const uint32_t bh = (uint32_t)jh.z * 512u, bc = un.x == un.z ? 0u : (uint32_t)jc.z * 512u;
    for (int s = 0; s < 16; ++s, ++it) {
      const int sl = it % NX;
      if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
      const uint32_t fb = bars + 8 * (2 * NS + sl), dst = xring + sl * XS;
      mbar_expect_tx(fb, bh + bc);
      bulk_g2s(dst, A.xp + (size_t)jh.w + (size_t)s * bh, bh, fb);
      if (bc) bulk_g2s(dst + bh, A.xp + (size_t)jc.w + (size_t)s * bc, bc, fb);
    }
  }
}

// ---------------- consumers: RS-wgmma stage (vml_probe MODE 0, x at slot offset xoff)
template <int N>
__device__ __forceinline__ void stage_w(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
                                        uint32_t (&hb)[2][8], int q, int wp, int lane, bool first, uint32_t xoff) {
  const int ws = it % NS, xs = it % NX;
  mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  const uint8_t* st = smem + ws * US + (q >> 1) * SW;
  const int tile = (q & 1) * 4 + wp;
  const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
  uint4 c[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
  const uint32_t sw0 = sp[0], sw1 = sp[32];
  __syncwarp();
  if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));      // codes + scales in registers: weight slot free
  mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
  __syncwarp();
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS + xoff), 4096u);
  const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
    const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
    const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t* hh = hb[hf];
      dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
      wg_fence();
      if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);
      else mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
      wg_commit();
      wg_wait<1>();
      if (i == 0 && hf == 0 && prevx >= 0) {             // the previous stage's last group retired: its x slot is free
        if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
        prevx = -1;
      }
    }
  }
  prevx = xs;
  ++it;
}
// mma.sync stage (VML L1b stage_m, x at slot offset xoff): per warp its 16-row tile, 2 accumulator chains
template <int N>
__device__ __forceinline__ void stage_m(const Args& A, uint8_t* smem, uint32_t bars, int& it, float (&d2)[2][N / 2], int q,
                                        int wp, int lane, uint32_t xoff) {
  const int ws = it % NS, xs = it % NX;
  mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  const uint8_t* st = smem + ws * US + (q >> 1) * SW;
  const int tile = (q & 1) * 4 + wp;
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
  const uint32_t xb = sa(smem + NS * US + xs * XS + xoff) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
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
      for (int nt = 0; nt < N / 8; ++nt) {
        uint32_t b0, b1, b2, b3;
        ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
        mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);
        mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);
      }
    }
  }
  __syncwarp();
  if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
  ++it;
}

template <int N>
__device__ __forceinline__ void pair_unit(const Args& A, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane,
                                          uint32_t xoff, int4 jb, int blk, int jidx) {
  float d[N / 2];
  if constexpr (MXS_COLD_MMA && N <= 16) {
    float d2[2][N / 2];
#pragma unroll
    for (int k = 0; k < N / 2; ++k) { d2[0][k] = 0.f; d2[1][k] = 0.f; }
#pragma unroll 1
    for (int s = 0; s < 16; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane, xoff);
#pragma unroll
    for (int k = 0; k < N / 2; ++k) d[k] = d2[0][k] + d2[1][k];
  } else {
    uint32_t hb[2][8];
    int prevx = -1;
    stage_w<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, xoff);
#pragma unroll 1
    for (int s = 1; s < 16; ++s) stage_w<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, xoff);
    wg_wait<0>();
    if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
  }
  const float gs = __ldg(A.gsc13 + jb.x);
  const int g = lane >> 2, t = lane & 3;
  const int col = 8 * (blk * 8 + (q & 1) * 4 + wp) + g;
  const int rb = A.jbase[jidx];
#pragma unroll
  for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int n = 8 * nt + 2 * t + hh;
      if (n < jb.y)
        A.inter[(size_t)(rb + n) * KDIM + col] = __float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
    }
}

__global__ void __launch_bounds__(NTH, 1) mxs_kernel(const Args A) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = sa(smem + BAR_OFF);
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
    for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (A.marks) { A.marks[(size_t)blockIdx.x * 10] = gtime(); A.marks[(size_t)blockIdx.x * 10 + 1] = clock64(); }
  }
  __syncthreads();
  const int2 rg = A.cta[blockIdx.x];
  if (warp >= NCW) {
    if (lane == 0) {
      if (warp == NCW) prod_w(A, sa(smem), bars, rg.x, rg.y);
      else prod_x(A, sa(smem + NS * US), bars, rg.x, rg.y);
    }
    return;
  }
  const int q = warp >> 2, wp = warp & 3, pair = q >> 1;
  int it = 0;
  for (int u = rg.x; u < rg.y; ++u) {
    const int4 un = A.unit[u];
    const int my = pair ? un.z : un.x;
    const int4 jb = A.job[my];
    const uint32_t xoff = (pair && un.x != un.z) ? (uint32_t)A.job[un.x].z * 512u : 0u;
    const int blk = pair ? un.w : un.y;
    switch (jb.z >> 3) {
      case 1: pair_unit<8>(A, smem, bars, it, q, wp, lane, xoff, jb, blk, my); break;
      case 2: pair_unit<16>(A, smem, bars, it, q, wp, lane, xoff, jb, blk, my); break;
      case 3: pair_unit<24>(A, smem, bars, it, q, wp, lane, xoff, jb, blk, my); break;
      case 4: pair_unit<32>(A, smem, bars, it, q, wp, lane, xoff, jb, blk, my); break;
      case 5: pair_unit<40>(A, smem, bars, it, q, wp, lane, xoff, jb, blk, my); break;
      case 6: pair_unit<48>(A, smem, bars, it, q, wp, lane, xoff, jb, blk, my); break;
      case 7: pair_unit<56>(A, smem, bars, it, q, wp, lane, xoff, jb, blk, my); break;
      default: pair_unit<64>(A, smem, bars, it, q, wp, lane, xoff, jb, blk, my); break;
    }
  }
  if (wp == 0 && lane == 0 && A.marks) {
    A.marks[(size_t)blockIdx.x * 10 + 2 + 2 * q] = gtime();
    A.marks[(size_t)blockIdx.x * 10 + 3 + 2 * q] = clock64();
  }
}
}  // namespace mxs
""".replace("@WGSPECS@", "".join(_wg_spec(n) for n in (8, 16, 24, 32, 40, 48, 56, 64)))

HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

std::vector<int64_t> mxs_attrs() {
  cudaFuncSetAttribute(mxs::mxs_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mxs::SMEM);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, mxs::mxs_kernel);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mxs::mxs_kernel, mxs::NTH, mxs::SMEM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)mxs::SMEM, occ};
}

void mxs_run(torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor job, torch::Tensor jbase, torch::Tensor unit,
             torch::Tensor cta, torch::Tensor xp, torch::Tensor inter, torch::Tensor err, torch::Tensor marks, int64_t G) {
  mxs::Args a;
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>();
  a.job = reinterpret_cast<const int4*>(job.data_ptr());
  a.jbase = jbase.data_ptr<int>();
  a.unit = reinterpret_cast<const int4*>(unit.data_ptr());
  a.cta = reinterpret_cast<const int2*>(cta.data_ptr());
  a.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr());
  a.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  a.err = err.data_ptr<int>();
  a.marks = marks.numel() >= G * 10 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  cudaFuncSetAttribute(mxs::mxs_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mxs::SMEM);
  mxs::mxs_kernel<<<(int)G, mxs::NTH, mxs::SMEM, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> mxs_attrs();
void mxs_run(torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor job, torch::Tensor jbase, torch::Tensor unit,
             torch::Tensor cta, torch::Tensor xp, torch::Tensor inter, torch::Tensor err, torch::Tensor marks, int64_t G);
"""

FLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
BUILDS = {"w": ["-DMXS_COLD_MMA=0"], "m": ["-DMXS_COLD_MMA=1"]}


def build(b, verbose=False):
    from torch.utils.cpp_extension import load_inline
    flags = FLAGS + BUILDS[b]
    src = DEV_SRC + HOST_SRC
    tag = hashlib.sha1((CPP_SRC + src + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"mxs_probe_{b}_{tag}", cpp_sources=CPP_SRC, cuda_sources=src, functions=["mxs_run", "mxs_attrs"],
                      extra_cuda_cflags=flags, verbose=verbose)
    return mod, time.time() - t0


def pack_x(x, toks, npad):
    """x rows (zero-padded to npad) in K2A's xw_u32 wgmma B layout per 256-k stage (= vml_probe.pack_x):
    [s][n8][i 4][q 8][r 8][t 4][b 2] for k = 256 s + 64 i + 16 t + 2 q + b -> uint8 [16 * npad * 512]."""
    import torch
    X = torch.zeros((npad, 4096), dtype=torch.bfloat16, device=x.device)
    X[:len(toks)] = x[torch.as_tensor(toks, dtype=torch.long, device=x.device)]
    X = X.view(npad // 8, 8, 16, 4, 4, 8, 2).permute(2, 0, 3, 5, 1, 4, 6).contiguous()
    return X.view(torch.uint8).reshape(-1)


def plan(nh, nc, U, G, dev, seed=7):
    """Jobs: experts 0..15 hot (nh tokens), 16..31 cold (nc tokens). Arms PH / PC / MIX over G x U units of distinct
    128-row blocks: PH units = 2 hot blocks of one expert, PC = 2 cold blocks, MIX = 1 hot + 1 cold block."""
    import torch
    assert -(-nh // 8) * 8 + -(-nc // 8) * 8 <= 64, "x slot holds 64 rows"
    assert 2 * G * U <= 16 * 64, "not enough distinct blocks"
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((256, 4096), generator=g).to(torch.bfloat16).to(dev)
    jobs, jbase, xparts, toks, xo, rb = [], [], [], [], 0, 0
    for e in range(32):
        n = nh if e < 16 else nc
        npad = -(-n // 8) * 8
        tk = torch.randperm(256, generator=g)[:n].tolist()
        xparts.append(pack_x(x, tk, npad))
        jobs.append((e, n, npad, xo))
        jbase.append(rb)
        toks.append(tk)
        xo += 16 * npad * 512
        rb += n
    hb = [(e, b) for e in range(16) for b in range(64)]
    cb = [(e, b) for e in range(16, 32) for b in range(64)]
    units = {"PH": [(hb[2 * k][0], hb[2 * k][1], hb[2 * k + 1][0], hb[2 * k + 1][1]) for k in range(G * U)],
             "PC": [(cb[2 * k][0], cb[2 * k][1], cb[2 * k + 1][0], cb[2 * k + 1][1]) for k in range(G * U)],
             "MIX": [(hb[k][0], hb[k][1], cb[k][0], cb[k][1]) for k in range(G * U)]}
    t = lambda v: torch.tensor(v, dtype=torch.int32).to(dev).contiguous()  # noqa: E731
    cta = t([(c * U, (c + 1) * U) for c in range(G)])
    return {"x": x, "toks": toks, "job": t(jobs), "jbase": t(jbase), "units": {k: t(v) for k, v in units.items()},
            "ulist": units, "cta": cta, "xp": torch.cat(xparts).contiguous(), "rows": rb}


def check(P, arm, inter, w, dequant):
    """max relative error over the inter columns every unit of the arm produced (64 per 128-row block)."""
    import torch
    blocks = {}
    for (eh, bh, ec, bc) in P["ulist"][arm]:
        blocks.setdefault(eh, set()).add(bh)
        blocks.setdefault(ec, set()).add(bc)
    worst = 0.0
    jb = P["jbase"].tolist()
    for e, bs in blocks.items():
        xs = P["x"].float()[torch.as_tensor(P["toks"][e], dtype=torch.long, device=P["x"].device)]
        W = dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e])
        h = xs @ W.T
        ref = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        cols = torch.cat([torch.arange(64 * b, 64 * b + 64, device=ref.device) for b in sorted(bs)])
        got = inter[jb[e]:jb[e] + len(P["toks"][e])].float()[:, cols]
        r = ref[:, cols]
        worst = max(worst, float((got - r).norm() / r.norm().clamp(min=1e-30)))
    return worst


def probe(ws13, gsc13, w, dequant, timer, ov, G, log=print, nhs=(56, 40), nc=8, U=3, rounds=12):
    import torch
    import concurrent.futures as cf
    dev = ws13.device
    mods = {}
    with cf.ThreadPoolExecutor(len(BUILDS)) as ex:
        futs = {b: ex.submit(build, b) for b in BUILDS}
        for b, f in futs.items():
            mods[b], dt = f.result()
            log(f"[mxs] build {b} in {dt:.0f} s attrs (regs, local, smem, occ) {list(mods[b].mxs_attrs())}")
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    marks = torch.zeros((G * 10,), dtype=torch.int64, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    res = {}
    for nh in nhs:
        P = plan(nh, nc, U, G, dev)
        inter = torch.zeros((P["rows"] + 8, 4096), dtype=torch.bfloat16, device=dev)

        def fn(b, arm, mk=False, P=P, inter=inter):
            return lambda: mods[b].mxs_run(ws13, gsc13, P["job"], P["jbase"], P["units"][arm], P["cta"], P["xp"], inter,
                                           err, marks if mk else nomark, G)
        rel, gr = {}, {}
        for b in mods:
            for arm in ("PH", "PC", "MIX"):
                err.zero_()
                inter.zero_()
                fn(b, arm)()
                torch.cuda.synchronize()
                rel[f"{arm}{b}"] = check(P, arm, inter, w, dequant) if int(err[0]) == 0 else float("inf")
                f = fn(b, arm)
                with torch.cuda.stream(timer.stream):
                    f()
                    torch.cuda.synchronize()
                    g_ = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_, stream=timer.stream):
                        f()
                torch.cuda.synchronize()
                gr[f"{arm}{b}"] = g_
        names = list(gr)
        ts = {a: [] for a in names}
        for rd in range(rounds):
            k = rd % len(names)
            for a in names[k:] + names[:k]:
                ts[a].append(max(timer.time_replay(gr[a]) - ov, 1e-3))
        us = {a: statistics.median(v) for a, v in ts.items()}
        stage = {a: us[a] / (16 * U) for a in names}
        clk = {}
        for b in mods:
            for arm in ("PH", "PC", "MIX"):
                marks.zero_()
                with torch.cuda.stream(timer.stream):
                    timer.flush()
                    fn(b, arm, mk=True)()
                torch.cuda.synchronize()
                m = marks.view(G, 10).double()
                tend = m[:, 2:10:2].max(dim=1).values
                cend = m[:, 3:10:2].max(dim=1).values
                clk[f"{arm}{b}"] = float(((cend - m[:, 1]) / (tend - m[:, 0]).clamp(min=1.0)).median())
        r = {}
        for b in mods:
            mean = 0.5 * (stage[f"PH{b}"] + stage[f"PC{b}"])
            r[b] = stage[f"MIX{b}"] / mean
            log(f"[mxs] NH={nh} build {b}: stage us PH {stage[f'PH{b}']:.3f} PC {stage[f'PC{b}']:.3f} MIX "
                f"{stage[f'MIX{b}']:.3f} | MIX / mean(PH, PC) {r[b]:.3f} | SM GHz PH {clk[f'PH{b}']:.2f} PC "
                f"{clk[f'PC{b}']:.2f} MIX {clk[f'MIX{b}']:.2f} | rel " + " ".join(f"{a} {rel[a + b]:.1e}" for a in
                                                                                  ("PH", "PC", "MIX")))
        res[nh] = {"stage_us": stage, "us": us, "rel": rel, "ratio": r, "ghz": clk, "err": int(err[0])}
        del gr, inter, P
    return res


def decide(res, log=print):
    """Pre-registered rule (notes/study/indep_mid.md s4.4) on NH = 56 (the T = 128 hot-job size), best cold build."""
    nh = 56 if 56 in res else max(res)
    R = res[nh]
    if any(v > 1e-2 for v in R["rel"].values()):
        log("[mxs] INVALID: an arm failed the FP32 check")
        return "INVALID"
    b = min(R["ratio"], key=lambda k: R["stage_us"][f"MIX{k}"])
    r = R["ratio"][b]
    verdict = "STRONG GO" if r <= 0.80 else "GO" if r <= 0.87 else "KILL" if r >= 0.95 else "PARK"
    log(f"[mxs] NH={nh}: best cold build {b}, MIX / mean(PH, PC) = {r:.3f} -> {verdict}")
    return verdict

"""HUC0: hot-under-cold step-0 lab (notes/study/moe_hot_fresh.md). W13 phase only, timing + correctness.

Question: can a hot RS-wgmma warpgroup run on the same SM as a cold mma.sync byte stream without slowing it, so the
hot experts' tensor time hides under the cold stream instead of adding to it (K2A today: role-split CTAs, 2 per SM)?

One launch, 1 CTA/SM x 768 threads (launch 80 regs, setmaxnreg):
  WG0   warp 0 / 1 = cold producers (rings A / B), warp 2 = hot producer, warp 3 idle      (40 regs)
  WG1-4 16 cold consumer warps, 8 per ring: K2A v3 cold_item NC1 (deq4 / bmul2 + mma.sync m16n8k16, <= 8 tokens),
        one 128-row x full-K item per 8 warps, x rows staged by per-row bulk copies (K2A K1's X path)   (72 regs)
  WG5   hot consumer warpgroup: RS wgmma m64nNk16 (N 16 / 32 / 48), a unit = one 128-row block (two m64 tiles,
        K2A's prepacked stage layout) x full K x one token group of a hot expert (<= 48 tokens)        (152 regs)
Cold = experts with 1..8 tokens, hot = >= 9 tokens (split into token groups of <= 48). Each CTA gets a serpentine deal of
hot units (cost-sorted) and a contiguous run of cold items sized so every CTA moves the same bytes (hot + cold).
Plans and the hot X pack (wgmma core-matrix layout, K2A's xw_u32 order) are built on the host from the routing: lab
only, outside timing, so the arms exclude ~1-2 us of device-side planning and ~1-2 us of X pack.
mode: bit 0 null consumers, bit 1 no cold work, bit 2 no hot work.
"""
import hashlib
import math
import statistics
import time

import torch

HUC0_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace huc {
constexpr int KDIM = 4096;
constexpr int SW = 18432;
constexpr int XROW = 528;
constexpr int CNST = 3;
constexpr int CSLOT = SW + 8 * XROW;            // 22656
constexpr int HNST = 2;
constexpr int HNMAX = 48;
constexpr int HSLOT = SW + HNMAX * 512;         // 43008
constexpr int CRING = CNST * CSLOT;             // 67968
constexpr int HOFF = 2 * CRING;                 // 135936
constexpr int BOFF = HOFF + HNST * HSLOT;       // 221952
constexpr int NBAR = 2 * 2 * CNST + 2 * HNST;   // 16
constexpr int SMEM = BOFF + 8 * NBAR;           // 222080
constexpr int NTHR = 768;
constexpr unsigned FULL = 0xffffffffu;
constexpr long long WAIT_BUDGET = 1ll << 30;    // clock64 cycles (~0.5 s) per wait -> error flag, no trap

struct Args {
  const __nv_bfloat16* x; const uint8_t* ws13; const float* gsc13;
  const int* plan; const int4* hu; const int* hr; const int4* ci; const int* cl;
  const uint8_t* xp; __nv_bfloat16* inter; int* err; long long* marks; int mode;
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout (K2A v3)
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
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  int n = 0;
  while (!mbar_try(b, par)) {
    if ((++n & 255) == 0) {
      if (*reinterpret_cast<volatile int*>(err) != 0) return;
      if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }
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
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
// wgmma (K2A v3.2 / v4.0 forms): no-swizzle K-major core matrices, LBO 128 B, SBO as given
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_rs2(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <> __device__ __forceinline__ void wgmma_rs2<16>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%12, %13};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
template <> __device__ __forceinline__ void wgmma_rs2<32>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%20, %21};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
template <> __device__ __forceinline__ void wgmma_rs2<48>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%28, %29};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}

// One kb (16 B of codes) of this warp's tile, B from the staged x row (K2A v3 consume_kb<1>).
__device__ __forceinline__ void consume_kb1(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow,
                                            int kloc, int t, float* acc) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  const uint4* xq = reinterpret_cast<const uint4*>(xrow + kloc + 16 * t);
  const uint4 x0 = xq[0], x1 = xq[1];
  mma16816(acc, ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
  mma16816(acc, ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
  mma16816(acc, ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
  mma16816(acc, ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
}

// ---------------- cold ring: producer (one warp) ----------------
__device__ __forceinline__ void cold_producer(const Args& A, uint32_t ring_a, uint32_t full_a, uint32_t empty_a,
                                              int b, int e, int lane) {
  const uint64_t pol = policy_evict_first();
  int it = 0;
  for (int i = b; i < e; ++i) {
    const int4 c = A.ci[i];   // expert, row block, token-list offset (= first route), tokens
    const uint8_t* wb = A.ws13 + (size_t)(c.x * 64 + c.y) * 16 * SW;
    const int tok = lane < c.w ? A.cl[c.z + lane] : 0;
    for (int s = 0; s < 16; ++s, ++it) {
      const int slot = it % CNST;
      mbar_wait(empty_a + 8 * slot, ((it / CNST) & 1) ^ 1, A.err);
      const uint32_t dst = ring_a + slot * CSLOT, fb = full_a + 8 * slot;
      if (lane == 0) {
        mbar_expect_tx(fb, SW + c.w * 512);
        bulk_g2s_ef(dst, wb + (size_t)s * SW, SW, fb, pol);
      }
      __syncwarp();
      if (lane < c.w) bulk_g2s(dst + SW + lane * XROW, A.x + (size_t)tok * KDIM + s * 256, 512, fb);
      __syncwarp();
    }
  }
}
// ---------------- cold ring: consumer warp w (tile w of every stage) ----------------
__device__ __forceinline__ void cold_consumer(const Args& A, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                              int b, int e, int w, int lane, bool nullc) {
  const int g8 = lane >> 2, t = lane & 3;
  int it = 0;
  for (int i = b; i < e; ++i) {
    const int4 c = A.ci[i];
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < 16; ++s, ++it) {
      const int slot = it % CNST;
      mbar_wait(full_a + 8 * slot, (it / CNST) & 1, A.err);
      if (!nullc) {
        const uint8_t* st = ring + slot * CSLOT;
        const uint4* wp = reinterpret_cast<const uint4*>(st + w * 2048) + lane;
        const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + w * 256) + lane;
        const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW + g8 * XROW);
        uint4 wv[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) wv[k] = wp[k * 32];
        const uint32_t sv0 = sp[0], sv1 = sp[32];
        consume_kb1(wv[0], sv0, 0, xrow, 0, t, acc);
        consume_kb1(wv[1], sv0, 1, xrow, 64, t, acc);
        consume_kb1(wv[2], sv1, 0, xrow, 128, t, acc2);
        consume_kb1(wv[3], sv1, 1, xrow, 192, t, acc2);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
    if (!nullc) {   // SwiGLU -> inter (K2A flush_item<1, 1>, whole item)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k];
      const float gs = __ldg(A.gsc13 + c.x);
      const int col = 8 * (c.y * 8 + w) + g8;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 2 * t + hh;
        if (n < c.w) A.inter[(size_t)(c.z + n) * KDIM + col] = __float2bfloat16_rn(swiglu(acc[hh] * gs, acc[2 + hh] * gs));
      }
    }
  }
}
// ---------------- hot ring: producer (one warp) ----------------
__device__ __forceinline__ void hot_producer(const Args& A, uint32_t ring_a, uint32_t full_a, uint32_t empty_a,
                                             int b, int e, int lane) {
  const uint64_t pol = policy_evict_first();
  int it = 0;
  for (int u = b; u < e; ++u) {
    const int4 h = A.hu[u];   // expert, row block, X byte offset, tokens | padded tokens << 16
    const uint32_t xb = (uint32_t)(h.w >> 16) * 512u;
    const uint8_t* wb = A.ws13 + (size_t)(h.x * 64 + h.y) * 16 * SW;
    const uint8_t* xs = A.xp + (size_t)h.z;
    for (int s = 0; s < 16; ++s, ++it) {
      const int slot = it % HNST;
      mbar_wait(empty_a + 8 * slot, ((it / HNST) & 1) ^ 1, A.err);
      if (lane == 0) {
        const uint32_t dst = ring_a + slot * HSLOT, fb = full_a + 8 * slot;
        mbar_expect_tx(fb, SW + xb);
        bulk_g2s_ef(dst, wb + (size_t)s * SW, SW, fb, pol);
        bulk_g2s(dst + SW, xs + (size_t)s * xb, xb, fb);
      }
      __syncwarp();
    }
  }
}
template <int N>
__device__ __forceinline__ void hot_flush(const Args& A, const float (&acc)[N / 2], int tile, int rbase, int ntok,
                                          float gs, int g, int t) {
  const int col = 8 * tile + g;
#pragma unroll
  for (int q = 0; q < N / 8; ++q)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int n = 8 * q + 2 * t + hh;
      if (n < ntok)
        A.inter[(size_t)(rbase + n) * KDIM + col] =
            __float2bfloat16_rn(swiglu(acc[4 * q + hh] * gs, acc[4 * q + 2 + hh] * gs));
    }
}
// decode this warp's tile kb i into h[16] (K2A hot_item's exact BF16 dequant)
__device__ __forceinline__ void hot_decode(const uint4 w, const uint32_t sw2, int par, uint32_t* h) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  deq4(w.x, h); deq4(w.y, h + 4); deq4(w.z, h + 8); deq4(w.w, h + 12);
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) { h[jj] = bmul2(h[jj], Sg); h[8 + jj] = bmul2(h[8 + jj], S8); }
}
// One hot unit: 128 rows (tile pair hw / hw + 4 = two m64 accumulators) x 16 stages x N token columns.
// Per kb: tile A group (decode hA, 4 wgmma, commit), tile B group (decode hB, 4 wgmma, commit); wait_group 1 before a
// buffer is rewritten (the other tile's group may stay in flight), wait_group 0 before the slot is released.
template <int N>
__device__ __forceinline__ void hot_unit(const Args& A, const uint8_t* ring, uint32_t full_a, uint32_t empty_a, int& it,
                                         const int4 h, int rbase, int hw, int lane, bool nullc) {
  const int g = lane >> 2, t = lane & 3;
  float acc0[N / 2], acc1[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) { acc0[i] = 0.f; acc1[i] = 0.f; }
  uint32_t hA[16], hB[16];
  for (int s = 0; s < 16; ++s, ++it) {
    const int slot = it % HNST;
    mbar_wait(full_a + 8 * slot, (it / HNST) & 1, A.err);
    if (!nullc) {
      const uint8_t* st = ring + slot * HSLOT;
      const uint4* wpA = reinterpret_cast<const uint4*>(st + hw * 2048) + lane;
      const uint4* wpB = reinterpret_cast<const uint4*>(st + (hw + 4) * 2048) + lane;
      const uint32_t* spA = reinterpret_cast<const uint32_t*>(st + 16384 + hw * 256) + lane;
      const uint32_t* spB = reinterpret_cast<const uint32_t*>(st + 16384 + (hw + 4) * 256) + lane;
      const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);
      const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 wa = wpA[i * 32], wb = wpB[i * 32];
        const uint32_t sa2 = spA[(i >> 1) * 32], sb2 = spB[(i >> 1) * 32];
        wg_wait<1>();
        hot_decode(wa, sa2, i & 1, hA);
        wg_fence();
        wgmma_rs2<N>(acc0, hA[0], hA[8], hA[1], hA[9], dlo + 64 * i, dhi);
        wgmma_rs2<N>(acc0, hA[2], hA[10], hA[3], hA[11], dlo + 64 * i + 16, dhi);
        wgmma_rs2<N>(acc0, hA[4], hA[12], hA[5], hA[13], dlo + 64 * i + 32, dhi);
        wgmma_rs2<N>(acc0, hA[6], hA[14], hA[7], hA[15], dlo + 64 * i + 48, dhi);
        wg_commit();
        wg_wait<1>();
        hot_decode(wb, sb2, i & 1, hB);
        wg_fence();
        wgmma_rs2<N>(acc1, hB[0], hB[8], hB[1], hB[9], dlo + 64 * i, dhi);
        wgmma_rs2<N>(acc1, hB[2], hB[10], hB[3], hB[11], dlo + 64 * i + 16, dhi);
        wgmma_rs2<N>(acc1, hB[4], hB[12], hB[5], hB[13], dlo + 64 * i + 32, dhi);
        wgmma_rs2<N>(acc1, hB[6], hB[14], hB[7], hB[15], dlo + 64 * i + 48, dhi);
        wg_commit();
      }
      wg_wait<0>();
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  if (!nullc) {
    const float gs = __ldg(A.gsc13 + h.x);
    const int ntok = h.w & 0xffff;
    hot_flush<N>(A, acc0, h.y * 8 + hw, rbase, ntok, gs, g, t);
    hot_flush<N>(A, acc1, h.y * 8 + hw + 4, rbase, ntok, gs, g, t);
  }
}

__global__ void __launch_bounds__(NTHR, 1) huc0_kernel(const Args A) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + BOFF);
  // bars: ring r (0 / 1) full [6r .. 6r+2], empty [6r+3 .. 6r+5]; hot full [12, 13], empty [14, 15]
  if (tid == 0) {
    for (int r = 0; r < 2; ++r)
      for (int i = 0; i < CNST; ++i) { mbar_init(sa(bars + 6 * r + i), 1); mbar_init(sa(bars + 6 * r + 3 + i), 8); }
    for (int i = 0; i < HNST; ++i) { mbar_init(sa(bars + 12 + i), 1); mbar_init(sa(bars + 14 + i), 4); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (A.marks) A.marks[(size_t)blockIdx.x * 8 + 0] = gtime();
  }
  __syncthreads();
  const int* pl = A.plan + (size_t)blockIdx.x * 8;
  const bool nullc = (A.mode & 1) != 0, nocold = (A.mode & 2) != 0, nohot = (A.mode & 4) != 0;
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
    if (warp < 2) {
      if (!nocold)
        cold_producer(A, sa(smem + warp * CRING), sa(bars + 6 * warp), sa(bars + 6 * warp + 3), pl[2 + 2 * warp],
                      pl[3 + 2 * warp], lane);
    } else if (warp == 2) {
      if (!nohot) hot_producer(A, sa(smem + HOFF), sa(bars + 12), sa(bars + 14), pl[0], pl[1], lane);
    }
  } else if (wg <= 4) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 72;\n" ::: "memory");
    const int cw = warp - 4, r = cw >> 3, w = cw & 7;
    if (!nocold) {
      cold_consumer(A, smem + r * CRING, sa(bars + 6 * r), sa(bars + 6 * r + 3), pl[2 + 2 * r], pl[3 + 2 * r], w, lane,
                    nullc);
      if (w == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 8 + 1 + r] = gtime();
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 152;\n" ::: "memory");
    const int hw = warp - 20;
    if (!nohot) {
      const uint32_t full_a = sa(bars + 12), empty_a = sa(bars + 14);
      const uint8_t* ring = smem + HOFF;
      int it = 0;
      for (int u = pl[0]; u < pl[1]; ++u) {
        const int4 h = A.hu[u];
        const int np = h.w >> 16, rbase = A.hr[u];
        if (np <= 16) hot_unit<16>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
        else if (np <= 32) hot_unit<32>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
        else hot_unit<48>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
      }
      if (hw == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 8 + 3] = gtime();
    }
  }
}
}  // namespace huc

std::vector<int64_t> huc0_attrs() {
  cudaFuncSetAttribute(huc::huc0_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, huc::SMEM);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, huc::huc0_kernel);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, huc::huc0_kernel, huc::NTHR, huc::SMEM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + huc::SMEM), occ};
}

void huc0_run(torch::Tensor x, torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor plan, torch::Tensor hu,
              torch::Tensor hr, torch::Tensor ci, torch::Tensor cl, torch::Tensor xp, torch::Tensor inter,
              torch::Tensor err, torch::Tensor marks, int64_t G, int64_t mode) {
  huc::Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>();
  a.plan = plan.data_ptr<int>();
  a.hu = reinterpret_cast<const int4*>(hu.data_ptr());
  a.hr = hr.data_ptr<int>();
  a.ci = reinterpret_cast<const int4*>(ci.data_ptr());
  a.cl = cl.data_ptr<int>();
  a.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr());
  a.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  a.err = err.data_ptr<int>();
  a.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  a.mode = (int)mode;
  cudaFuncSetAttribute(huc::huc0_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, huc::SMEM);
  huc::huc0_kernel<<<(int)G, huc::NTHR, huc::SMEM, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

HUC0_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> huc0_attrs();
void huc0_run(torch::Tensor x, torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor plan, torch::Tensor hu,
              torch::Tensor hr, torch::Tensor ci, torch::Tensor cl, torch::Tensor xp, torch::Tensor inter,
              torch::Tensor err, torch::Tensor marks, int64_t G, int64_t mode);
"""

HUC0_HN = 48           # max tokens per hot unit (X slot)
HUC0_TH = 9            # hot iff >= 9 tokens (cold consumer = NC1, <= 8)
HUC0_EXPERT_W13 = 64 * 16 * 18432   # bytes of one expert's W13 stages


def huc0_build(verbose=False):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((HUC0_CPP + HUC0_CUDA).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"huc0_{tag}", cpp_sources=HUC0_CPP, cuda_sources=HUC0_CUDA,
                      functions=["huc0_run", "huc0_attrs"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"], verbose=verbose)
    return mod, time.time() - t0


def huc0_case(bs, seed, dev):
    """generate_input's normal / ep8 / busiest routing math (our own copy, seeds disjoint from the judge's)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def huc0_pack_x(x, toks, npad):
    """x rows of toks, zero-padded to npad, in K2A's wgmma B layout per 256-k stage: [s][n8][i 4][q 8][r 8][t 4][b 2]
    for k = 256 s + 64 i + 16 t + 2 q + b (xw_u32 order) -> uint8 [16 * npad * 512]."""
    X = torch.zeros((npad, 4096), dtype=torch.bfloat16, device=x.device)
    X[:len(toks)] = x[torch.tensor(toks, dtype=torch.long, device=x.device)]
    X = X.view(npad // 8, 8, 16, 4, 4, 8, 2).permute(2, 0, 3, 5, 1, 4, 6).contiguous()
    return X.view(torch.uint8).reshape(-1)


def huc0_plan(d, G, dev, th=HUC0_TH, hn=HUC0_HN):
    """Host plan (lab only): hot units dealt serpentine by cost, cold items as contiguous runs that equalize bytes."""
    ids = d["topk_ids"].long().cpu()
    M = ids.shape[0]
    toks = {e: [] for e in range(32)}
    for t in range(M):
        for j in range(8):
            e = int(ids[t, j]) & 255
            if e < 32:
                toks[e].append(t)
    cnt = {e: len(v) for e, v in toks.items()}
    hot = sorted([e for e in range(32) if cnt[e] >= th], key=lambda e: (-cnt[e], e))
    cold = [e for e in range(32) if 0 < cnt[e] < th]
    route = 0
    cl, citems_by_e = [], {}
    for e in cold:
        citems_by_e[e] = (route, cnt[e])
        cl += toks[e]
        route += cnt[e]
    xp_parts, xoff, units = [], 0, []   # unit: (cost, order, e, rb, xoff, nt, npad, rbase)
    route_tok = {}
    for e in cold:
        for n, t in enumerate(toks[e]):
            route_tok[citems_by_e[e][0] + n] = (t, e)
    order = 0
    hot_macs = 0
    for e in hot:
        tl = toks[e]
        N = len(tl)
        ng = -(-N // hn)
        base = -(-N // ng)
        base = -(-base // 8) * 8
        for k in range(ng):
            tk = tl[k * base:(k + 1) * base]
            if not tk:
                continue
            nt = len(tk)
            npad = -(-nt // 16) * 16
            xp_parts.append(huc0_pack_x(d["hidden_states"], tk, npad))
            for n, t in enumerate(tk):
                route_tok[route + n] = (t, e)
            # cost: hot WG time ~ max(tensor 128 x 4096 x npad at ~1400 MAC/clk, bytes (18432 + npad x 512) x 16 at ~24 B/clk)
            cost = max(128 * 4096 * npad / 1400.0, (18432 + npad * 512) * 16 / 24.0)
            for rb in range(64):
                units.append((cost, order, e, rb, xoff, nt, npad, route))
                order += 1
            hot_macs += 64 * 128 * 4096 * nt
            xoff += 16 * npad * 512
            route += nt
    R = route
    units.sort(key=lambda u: (-u[0], u[1]))
    per = [[] for _ in range(G)]
    for p, u in enumerate(units):
        r, i = divmod(p, G)
        per[i if r % 2 == 0 else G - 1 - i].append(u)
    hu, hr = [], []
    plan = torch.zeros((G, 8), dtype=torch.int32)
    hbytes = []
    for c in range(G):
        plan[c, 0] = len(hu)
        hb = 0
        for (_, _, e, rb, xo, nt, npad, rbase) in per[c]:
            hu.append((e, rb, xo, nt | (npad << 16)))
            hr.append(rbase)
            hb += 16 * (18432 + npad * 512)
        plan[c, 1] = len(hu)
        hbytes.append(hb)
    citems = []   # (e, rb, loff, ntok, bytes)
    for e in cold:
        loff, n = citems_by_e[e]
        for rb in range(64):
            citems.append((e, rb, loff, n, 16 * (18432 + n * 512)))
    ctot = sum(c[4] for c in citems)
    tot = ctot + sum(hbytes)
    target = tot / G
    quota = [max(0.0, target - hb) for hb in hbytes]
    qs = sum(quota) or 1.0
    quota = [q * ctot / qs for q in quota]
    bounds, acc = [], 0.0
    for q in quota:
        acc += q
        bounds.append(acc)
    assign = [[] for _ in range(G)]
    cum, c = 0.0, 0
    for it in citems:
        mid = cum + it[4] / 2
        while c < G - 1 and mid > bounds[c]:
            c += 1
        assign[c].append(it)
        cum += it[4]
    ci = []
    for c in range(G):
        a_items, b_items = assign[c][0::2], assign[c][1::2]
        plan[c, 2] = len(ci)
        ci += [it[:4] for it in a_items]
        plan[c, 3] = len(ci)
        plan[c, 4] = len(ci)
        ci += [it[:4] for it in b_items]
        plan[c, 5] = len(ci)
    to_dev = lambda v, n: (torch.tensor(v, dtype=torch.int32).reshape(-1, n) if v else
                           torch.zeros((1, n), dtype=torch.int32)).to(dev).contiguous()
    P = {
        "plan": plan.to(dev).contiguous(),
        "hu": to_dev(hu, 4), "hr": to_dev(hr, 1).reshape(-1), "ci": to_dev(ci, 4), "cl": to_dev(cl, 1).reshape(-1),
        "xp": (torch.cat(xp_parts) if xp_parts else torch.zeros(16, dtype=torch.uint8, device=dev)).contiguous(),
        "R": R, "route_tok": route_tok, "hot": hot, "cold": cold, "cnt": cnt,
        "hot_bytes": len(hot) * HUC0_EXPERT_W13, "cold_bytes": len(cold) * HUC0_EXPERT_W13,
        "hot_macs": hot_macs, "hunits": len(units), "citems": len(citems),
        "hb_max": max(hbytes) if hbytes else 0, "per_cta_bytes": tot / G,
    }
    # empty-work plan rows for the no-cold / no-hot arms are produced by the mode bits; plan stays shared
    return P


def huc0_ref(d, w, P, dequant):
    """FP32 reference of inter for every planned route: silu(x W_gate^T) * (x W_up^T), bf16."""
    x = d["hidden_states"].float()
    R = P["R"]
    ref = torch.zeros((max(R, 1), 4096), dtype=torch.float32, device=x.device)
    by_e = {}
    for r, (t, e) in P["route_tok"].items():
        by_e.setdefault(e, []).append((r, t))
    for e, lst in by_e.items():
        rr = torch.tensor([a for a, _ in lst], dtype=torch.long, device=x.device)
        tt = torch.tensor([b for _, b in lst], dtype=torch.long, device=x.device)
        W = dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e])
        h = x[tt] @ W.T
        ref[rr] = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        del W
    return ref


class HucTimer:
    """The judge clone's timer: 256 MB read before every replay (cold L2), events around graph.replay()."""

    def __init__(self):
        self.flush_buf = torch.ones((256 << 20) // 4, dtype=torch.float32, device="cuda")
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")
        self.stream = torch.cuda.Stream()

    def capture(self, fn):
        with torch.cuda.stream(self.stream):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                fn()
        torch.cuda.synchronize()
        return gr

    def time(self, gr):
        with torch.cuda.stream(self.stream):
            torch.sum(self.flush_buf, dim=0, out=self.sink)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    def overhead(self, n=60):
        tiny = torch.zeros(1, device="cuda")
        with torch.cuda.stream(self.stream):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                tiny.add_(0)
        torch.cuda.synchronize()
        return statistics.median(self.time(gr) for _ in range(n))


def huc0_lab(st, k_full, k1_only, k_reset, w, dequant, log=print, rounds=24, cases=((256, 6), (128, 4)),
             seed0=880001):
    """Arms per routing: K (shipped K2A, full call), S (HUC0 null consumers), C (cold only), H (hot only), CH (both);
    then K1 (shipped K2A's W13 kernel(s) alone, separate block, state reset after). Prints the pre-registered gate."""
    dev = st["ws13"].device
    mod, dt = huc0_build()
    a = list(mod.huc0_attrs())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    log(f"[huc0] built in {dt:.0f}s; attrs (regs, local, smem, occ) {a}; {sms} SMs")
    if a[0] != 80 or a[3] < 1:
        log(f"[huc0] ABORT: launch registers {a[0]} != 80 (setmaxnreg budget 40/72/152 assumes 80) or occupancy {a[3]}")
        return []
    G = sms
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    marks = torch.zeros((G * 8,), dtype=torch.int64, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    timer = HucTimer()
    ov = timer.overhead()
    log(f"[huc0] graph overhead {ov:.2f} us")
    rows = []
    for bs, ncase in cases:
        for r in range(ncase):
            d = huc0_case(bs, seed0 + 1000 * bs + r, dev)
            t0 = time.time()
            P = huc0_plan(d, G, dev)
            inter = torch.zeros((max(P["R"], 1), 4096), dtype=torch.bfloat16, device=dev)
            hot_cnt = [P["cnt"][e] for e in P["hot"]]
            log(f"[huc0] bs={bs} r={r}: hot {len(P['hot'])} experts {hot_cnt} ({P['hunits']} units, "
                f"{P['hot_macs'] / 1e9:.2f} GMAC), cold {len(P['cold'])} experts ({P['citems']} items); "
                f"W13 bytes hot {P['hot_bytes'] / 1e6:.0f} + cold {P['cold_bytes'] / 1e6:.0f} MB; plan {time.time() - t0:.1f}s")

            def hf(mode, mk=False, d=d, P=P, inter=inter):
                def f():
                    mod.huc0_run(d["hidden_states"], st["ws13"], st["gsc13"], P["plan"], P["hu"], P["hr"], P["ci"],
                                 P["cl"], P["xp"], inter, err, marks if mk else nomark, G, mode)
                return f
            # correctness (first routing of each bs): CH output vs FP32 reference
            err.zero_()
            hf(0)()
            torch.cuda.synchronize()
            e0 = [int(v) for v in err.tolist()]
            rel_h = rel_c = float("nan")
            if r == 0:
                ref = huc0_ref(d, w, P, dequant)
                got = inter.float()
                ncold = sum(P["cnt"][e] for e in P["cold"])
                if ncold:
                    rel_c = float((got[:ncold] - ref[:ncold]).norm() / ref[:ncold].norm().clamp(min=1e-30))
                if P["R"] > ncold:
                    rel_h = float((got[ncold:P["R"]] - ref[ncold:P["R"]]).norm() /
                                  ref[ncold:P["R"]].norm().clamp(min=1e-30))
                del ref, got
            log(f"[huc0]   check: rel(cold) {rel_c:.2e} rel(hot) {rel_h:.2e} err {e0}")
            if e0[0] != 0:
                log("[huc0]   error flag set: timing skipped for this routing")
                rows.append({"bs": bs, "bad": True})
                err.zero_()
                continue
            arms = {"K": lambda d=d: k_full(d), "S": hf(1), "C": hf(4), "H": hf(2), "CH": hf(0)}
            graphs = {nm: timer.capture(fn) for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            del graphs
            # K1 alone (shipped W13 kernel(s)); separate block, K2A state reset afterwards
            try:
                g1 = timer.capture(lambda d=d: k1_only(d))
                us["K1"] = statistics.median(max(timer.time(g1) - ov, 1e-3) for _ in range(max(8, rounds // 2)))
                del g1
            except Exception as ex:  # noqa: BLE001
                log(f"[huc0]   K1 arm failed: {ex!r}")
                us["K1"] = float("nan")
            k_reset()
            # one marked run of CH: per-CTA ends (us from the CTA's start)
            marks.zero_()
            with torch.cuda.stream(timer.stream):
                torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                hf(0, mk=True)()
            torch.cuda.synchronize()
            mv = marks.view(G, 8).tolist()
            t00 = min(m[0] for m in mv)
            ends = sorted((max(m[1], m[2], m[3]) - t00) / 1e3 for m in mv)
            cend = sorted((max(m[1], m[2]) - t00) / 1e3 for m in mv)
            hend = sorted((m[3] - t00) / 1e3 for m in mv if m[3] > 0)
            e1 = [int(v) for v in err.tolist()]
            B = P["hot_bytes"] + P["cold_bytes"]
            row = {"bs": bs, "us": us, "B": B, "cb": P["cold_bytes"], "hot_macs": P["hot_macs"], "err": e1, "rel_h": rel_h, "rel_c": rel_c,
                   "end": (statistics.median(ends), ends[-1]), "cend": (statistics.median(cend), cend[-1]),
                   "hend": (statistics.median(hend), hend[-1]) if hend else (float("nan"), float("nan"))}
            rows.append(row)
            log(f"[huc0]   us: " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items()) +
                f" | W13 bytes/S {B / us['S'] / 1e6:.2f} TB/s, /CH {B / us['CH'] / 1e6:.2f} TB/s; "
                f"hot {P['hot_macs'] * 2 / us['H'] / 1e6:.0f} TF in H, {P['hot_macs'] * 2 / us['CH'] / 1e6:.0f} TF in CH | "
                f"CH/max(C,H) {us['CH'] / max(us['C'], us['H']):.3f} CH/K1 {us['CH'] / us['K1']:.3f} err {e1}")
            log(f"[huc0]   CH marks p50/max: end {row['end'][0]:.1f}/{row['end'][1]:.1f} cold {row['cend'][0]:.1f}/"
                f"{row['cend'][1]:.1f} hot {row['hend'][0]:.1f}/{row['hend'][1]:.1f} us")
            err.zero_()
            del inter, P
    # pre-registered gate (notes/memory/gpu_requests.md R-hot-0); +3 us = device-side plan + X pack not in the arms
    log("[huc0] GATE (medians over routings): G2 (decides) (CH+3)/K1 <= 0.75 at bs256 and <= 0.90 at bs128, rel <= 1e-2, "
        "err 0; diagnostics G1 CH/max(C,H) (<= 1.10 = hot hides under cold), G3 cold-only rate on cold W13 bytes")
    g2s, valid = {}, True
    for bs in (256, 128):
        rr = [x for x in rows if x["bs"] == bs and not x.get("bad")]
        bad = [x for x in rows if x["bs"] == bs and x.get("bad")]
        if not rr or bad:
            log(f"[huc0]   bs={bs}: {len(rr)} valid / {len(bad)} with error flags -> INVALID")
            valid = False
            continue
        rels = [v for x in rr for v in (x["rel_h"], x["rel_c"]) if v == v]
        if any(v > 1e-2 for v in rels) or any(x["err"][0] for x in rr):
            log(f"[huc0]   bs={bs}: rel {rels} or err flags -> INVALID")
            valid = False
        g1 = statistics.median(x["us"]["CH"] / max(x["us"]["C"], x["us"]["H"]) for x in rr)
        g2 = statistics.median((x["us"]["CH"] + 3.0) / x["us"]["K1"] for x in rr)
        g3 = statistics.median(x["cb"] / x["us"]["C"] / 1e6 for x in rr)
        g2s[bs] = g2
        log(f"[huc0]   bs={bs}: G2 (CH+3)/K1 {g2:.3f} | G1 CH/max(C,H) {g1:.3f} | CH/K (full call) "
            f"{statistics.median(x['us']['CH'] / x['us']['K'] for x in rr):.3f} | S {statistics.median(x['us']['S'] for x in rr):.1f} "
            f"C {statistics.median(x['us']['C'] for x in rr):.1f} H {statistics.median(x['us']['H'] for x in rr):.1f} "
            f"CH {statistics.median(x['us']['CH'] for x in rr):.1f} K1 {statistics.median(x['us']['K1'] for x in rr):.1f} us | "
            f"G3 cold rate {g3:.2f} TB/s")
    if not valid or 256 not in g2s:
        dec = "INVALID (fix, rerun under the same rule)"
    elif g2s[256] <= 0.75 and g2s.get(128, 9) <= 0.90:
        dec = "GO (build HUC K1/K2 for bs128 + bs256)"
    elif g2s[256] <= 0.75:
        dec = "PARTIAL (build HUC for bs256 only)"
    elif g2s[256] <= 0.85:
        dec = "REVISE (one revision guided by S / C / H and the marks, one rerun)"
    else:
        dec = "KILL (HUC structure; add trap)"
    log(f"[huc0] DECISION {dec}")
    del timer
    torch.cuda.empty_cache()
    return rows

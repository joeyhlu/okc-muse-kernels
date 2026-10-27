"""Z1: slice-chained W13 -> W2 inside one CTA, T <= 8 (request R-moe1-2; notes/study/moe_structure_fresh.md s7).

No cross-CTA dependency between W13 and W2: CTA b of G = SMs (1 CTA/SM, 16 consumer warps) owns a contiguous, equal share
of the touched experts' intermediate dims in 16-dim granules (expert-major order; 1-2 experts per CTA at bs <= 8):
  1. X of the expert's routed tokens is loaded once into smem (warp 17);
  2. W13 for its dims: tile-major K2A chunks (16 rows x 256 K, K2A's own bytes permuted), one stage = one tile x 16 K-stages,
     warp w = K-stage w (K2A's consume_kb), sub-groups of SUB tiles reduced across the 16 warps in smem, SwiGLU, bf16 inter
     in smem (16-dim granules, k16 B-fragment order);
  3. W2[:, dims] from a slice-contiguous repack (per 16-row tile and 16-dim granule: one 32-bit word per lane in mma-A order
     via K2A's BITPOS3 + 16 scale bytes = 144 B), the tile's granule run is one bulk copy; partial y for all 4096 rows,
     red.add into y[route][4096] (FP32, zero at rest);
  4. finalize kernel behind PDL (one CTA per token): sums of squares from the full y rows, post-norm, weighted sum, y -> 0.
mode bits (lab arms): 1 null consumers (no math / reds), 8 no finalize kernel.
sm_90a: bulk copies + PDL. sm_86 (local correctness only): cp.async, 1-stage ring, <= 2 tokens, 2-tile sub-groups.
"""
import hashlib
import statistics
import time

import torch

Z1_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace z1 {
constexpr int KDIM = 4096;
constexpr int CH = 2304;              // W13 chunk: 16 rows x 256 K (2048 B codes + 256 B scales), K2A layout
constexpr int STB = 16 * CH;          // stage bytes: one W13 tile x 16 K-stages (36864)
constexpr int GB = 144;               // W2 (16-row tile, 16-dim granule): 32 lane words + 16 scale bytes
#ifndef Z1_NST
#define Z1_NST 3
#endif
#ifndef Z1_XR
#define Z1_XR 8
#endif
#ifndef Z1_SUB
#define Z1_SUB 4
#endif
constexpr int NST = Z1_NST, XR = Z1_XR, SUB = Z1_SUB;
constexpr int XRB = 8208;             // X row stride in smem (8192 + 16 pad: conflict-free B reads)
constexpr int NCW = 16;               // consumer warps
constexpr int NTH = 32 * (NCW + 2);   // + producer (16) + X loader (17)
constexpr int MAXG = 64;              // granules per CTA (inter buffer)
constexpr int IGB = 256;              // inter bytes per granule: 8 tokens x 16 dims bf16
constexpr int RED = NCW * SUB * 32 * 16;
constexpr int PMAX = 64;
constexpr unsigned FULL = 0xffffffffu;
constexpr long long WAIT_BUDGET = 1ll << 30;
#if __CUDA_ARCH__ >= 900
constexpr int FULLCNT = 1, XCNT = 1;
#else
constexpr int FULLCNT = 32, XCNT = 32;
#endif
constexpr int OFF_X = NST * STB, OFF_RED = OFF_X + XR * XRB, OFF_INT = OFF_RED + RED, OFF_BAR = OFF_INT + MAXG * IGB;
constexpr int SMEM = OFF_BAR + (2 * NST + 2) * 8;

struct Args {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint8_t* ws13t; const uint8_t* ws2s;
  const float* gsc13; const float* gsc2; const float* nw;
  float* y; int* meta; int* err; __nv_bfloat16* out; long long* marks;
  int M, lo, mode;
};

struct Plan {
  int n, R, nseg;
  int grp[32], cnt[32], off[32];
  float gs13[32], gs2[32];
  int se[4], sj0[4], sj1[4], sgb[4];
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // K2A BITPOS3: o[j] = codes (2j, 2j + 1)
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
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
// K2A consume_kb<1>: one kb (16 B of codes per lane) of a 16-row W13 chunk, B from X row g
__device__ __forceinline__ void consume_kb1(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow,
                                            int kloc, int t, float* acc) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  const uint4* xp = reinterpret_cast<const uint4*>(xrow + kloc + 16 * t);
  const uint4 x0 = xp[0], x1 = xp[1];
  mma16816(acc, ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
  mma16816(acc, ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
  mma16816(acc, ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
  mma16816(acc, ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
}

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
#endif
  return ok != 0;
}
__device__ __forceinline__ void set_err(int* err, int code) {
  if (atomicExch(err, 1) == 0) atomicExch(err + 1, code);
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, int code) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return; }
  }
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void red_f32(float* p, float v) {
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n" :: "l"(p), "f"(v) : "memory");
}
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
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
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
#else
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void warp_copy(int lane, uint32_t dst, const void* src, uint32_t bytes) {
  const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
  for (uint32_t o = 16u * lane; o < bytes; o += 512u) cpa16(dst + o, s + o);
}
__device__ __forceinline__ void pdl_launch() {}
__device__ __forceinline__ void pdl_wait() {}
#endif
// position of dim d (0..15) inside a granule's 16-dim B row: lane t reads dims (2t, 2t+1, 2t+8, 2t+9) at 4t..4t+3
__device__ __forceinline__ int iperm(int d) { return 4 * ((d & 7) >> 1) + (d & 1) + 2 * (d >> 3); }

__global__ void __launch_bounds__(NTH, 1) z1_kernel(const Args A) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int16_t s_list[PMAX];
  __shared__ int16_t s_rt[PMAX];
  __shared__ Plan S;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = blockIdx.x, G = gridDim.x;
  const int NP = 8 * A.M;
  const int mode = A.mode;
  long long* mk = A.marks ? A.marks + (size_t)b * 16 : nullptr;
  const uint32_t ring_a = sa(smem), x_a = sa(smem + OFF_X);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST), xfull_a = sa(bars + 2 * NST), xempty_a = sa(bars + 2 * NST + 1);
  float* red_s = reinterpret_cast<float*>(smem + OFF_RED);
  uint8_t* inter_s = smem + OFF_INT;

  if (tid == 0) {
    if (mk) mk[0] = gtime();
    pdl_launch();
    for (int i = 0; i < NST; ++i) { mbar_init(full_a + 8 * i, FULLCNT); mbar_init(empty_a + 8 * i, NCW); }
    mbar_init(xfull_a, XCNT); mbar_init(xempty_a, 1);
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
  if (warp == 1) { S.gs13[lane] = __ldg(A.gsc13 + lane); S.gs2[lane] = __ldg(A.gsc2 + lane); }
  if (warp == 0) {
    // ballot routing (Q2): lane holds id positions p = lane, lane + 32
    int e0 = -1, e1 = -1;
    if (lane < NP) { const int v = __ldg(A.ids + lane) - A.lo; e0 = (v >= 0 && v < 32) ? v : -1; }
    if (lane + 32 < NP) { const int v = __ldg(A.ids + lane + 32) - A.lo; e1 = (v >= 0 && v < 32) ? v : -1; }
    int fh = 0, c = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e) {
      const unsigned b0 = __ballot_sync(FULL, e0 == e), b1 = __ballot_sync(FULL, e1 == e);
      if (lane == e) { fh = __popc(b0); c = fh + __popc(b1); }
    }
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    const unsigned msk = __ballot_sync(FULL, c > 0);
    S.cnt[lane] = c; S.off[lane] = off;
    if (c > 0) S.grp[__popc(msk & ((1u << lane) - 1u))] = lane;
    const unsigned lt = (1u << lane) - 1u;
    const unsigned m0 = __match_any_sync(FULL, e0), m1 = __match_any_sync(FULL, e1);
    const int off0 = __shfl_sync(FULL, off, e0 & 31), off1 = __shfl_sync(FULL, off, e1 & 31);
    const int fh1 = __shfl_sync(FULL, fh, e1 & 31);
    if (lane < NP) {
      int r = -1;
      if (e0 >= 0) { r = off0 + __popc(m0 & lt); s_list[r] = (int16_t)lane; }
      s_rt[lane] = (int16_t)r;
    }
    if (lane + 32 < NP) {
      int r = -1;
      if (e1 >= 0) { r = off1 + fh1 + __popc(m1 & lt); s_list[r] = (int16_t)(lane + 32); }
      s_rt[lane + 32] = (int16_t)r;
    }
    if (lane == 31) {
      const int n = __popc(msk);
      S.n = n; S.R = v;
      // this CTA's equal share of the n * 256 granules, cut into per-expert segments
      const long long ng = (long long)n * 256;
      const int g0 = (int)(ng * b / G), g1 = (int)(ng * (b + 1) / G);
      int ns = 0, gb = 0;
      for (int k = g0 / 256; g1 > g0 && k <= (g1 - 1) / 256 && ns < 4; ++k) {
        const int j0 = max(g0, 256 * k) - 256 * k, j1 = min(g1, 256 * (k + 1)) - 256 * k;
        S.se[ns] = k; S.sj0[ns] = j0; S.sj1[ns] = j1; S.sgb[ns] = gb;
        gb += j1 - j0; ++ns;
      }
      S.nseg = ns;
      if (gb > MAXG) { set_err(A.err, 9); S.nseg = 0; }
    }
  }
  __syncthreads();
  const int nseg = S.nseg;

  if (warp == NCW) {
    // ---------------- producer: W13 tile stages, then W2 tile-piece stages, per segment ----------------
#if __CUDA_ARCH__ >= 900
    const uint64_t pol = policy_evict_first();
#endif
    int it = 0;
    bool first = true;
    for (int k = 0; k < nseg; ++k) {
      const int e = S.grp[S.se[k]], j0 = S.sj0[k], j1 = S.sj1[k], ngr = j1 - j0;
      for (int tl = 2 * j0; tl < 2 * j1; ++tl, ++it) {
        const int slot = it % NST;
        if (it >= NST) mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, A.err, 2);
        const uint8_t* src = A.ws13t + ((size_t)(e * 512 + tl) * 16) * CH;
#if __CUDA_ARCH__ >= 900
        if (lane == 0) { mbar_expect_tx(full_a + 8 * slot, STB); bulk_g2s_ef(ring_a + slot * STB, src, STB, full_a + 8 * slot, pol); }
#else
        warp_copy(lane, ring_a + slot * STB, src, STB);
        cpa_arrive(full_a + 8 * slot);
#endif
        if (first) { first = false; if (mk && lane == 0) mk[1] = gtime(); }
      }
      const int piece = ngr * GB;
      const int tps = min(32, STB / piece);
      for (int r0 = 0; r0 < 256; r0 += tps, ++it) {
        const int nt = min(tps, 256 - r0);
        const int slot = it % NST;
        if (it >= NST) mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, A.err, 2);
        const uint8_t* src = A.ws2s + ((size_t)(e * 256 + r0) * 256 + j0) * GB;
#if __CUDA_ARCH__ >= 900
        if (lane == 0) mbar_expect_tx(full_a + 8 * slot, (uint32_t)(nt * piece));
        __syncwarp();
        if (lane < nt) bulk_g2s_ef(ring_a + slot * STB + lane * piece, src + (size_t)lane * 256 * GB, piece, full_a + 8 * slot, pol);
#else
        for (int i = 0; i < nt; ++i) warp_copy(lane, ring_a + slot * STB + i * piece, src + (size_t)i * 256 * GB, piece);
        cpa_arrive(full_a + 8 * slot);
#endif
      }
    }
    if (mk && lane == 0) mk[10] = gtime();
    return;
  }

  if (warp == NCW + 1) {
    // ---------------- X loader: the segment's routed token rows, once per segment; route table for the finalize ----------------
    if (b == 0) {
      for (int p = lane; p < NP; p += 32) A.meta[p] = s_rt[p];
    }
    for (int k = 0; k < nseg; ++k) {
      const int e = S.grp[S.se[k]], te = S.cnt[e], c0 = S.off[e];
      if (k > 0) mbar_wait(xempty_a, (k - 1) & 1, A.err, 3);
      const int nrow = min(te, XR);
#if __CUDA_ARCH__ >= 900
      if (lane == 0) mbar_expect_tx(xfull_a, (uint32_t)(nrow * 8192));
      __syncwarp();
      if (lane < nrow) {
        const int tok = ((int)s_list[c0 + lane]) >> 3;
        bulk_g2s(x_a + lane * XRB, A.x + (size_t)tok * KDIM, 8192, xfull_a);
      }
#else
      for (int n = 0; n < nrow; ++n) {
        const int tok = ((int)s_list[c0 + n]) >> 3;
        warp_copy(lane, x_a + n * XRB, A.x + (size_t)tok * KDIM, 8192);
      }
      cpa_arrive(xfull_a);
#endif
    }
    return;
  }

  // ---------------- consumers: warps 0-15 ----------------
  const bool nullc = (mode & 1) != 0;
  const int g = lane >> 2, t = lane & 3;
  int it = 0;
  for (int k = 0; k < nseg; ++k) {
    const int e = S.grp[S.se[k]], j0 = S.sj0[k], j1 = S.sj1[k], ngr = j1 - j0, gbase = S.sgb[k];
    const int te = S.cnt[e], c0 = S.off[e];
    mbar_wait(xfull_a, k & 1, A.err, 4);
    // rows >= the routed tokens hold stale data (their mma columns are discarded); clamp so the read stays in smem
    const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(smem + OFF_X + min(g, XR - 1) * XRB);
    const float gs13 = S.gs13[e];
    // ---- W13: sub-groups of SUB tiles; warp = K-stage; cross-warp reduction; SwiGLU -> inter (bf16, granule order) ----
    for (int t0 = 2 * j0; t0 < 2 * j1; t0 += SUB) {
      const int ntl = min(SUB, 2 * j1 - t0);
      float acc[SUB][4], acc2[SUB][4];
#pragma unroll
      for (int q = 0; q < SUB; ++q)
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[q][i] = 0.f; acc2[q][i] = 0.f; }
#pragma unroll
      for (int q = 0; q < SUB; ++q) {
        if (q < ntl) {
          const int slot = it % NST;
          mbar_wait(full_a + 8 * slot, (it / NST) & 1, A.err, 1);
          if (!nullc) {
            const uint8_t* ch = smem + slot * STB + warp * CH;
            const uint4* wp = reinterpret_cast<const uint4*>(ch) + lane;
            const uint32_t* sp = reinterpret_cast<const uint32_t*>(ch + 2048) + lane;
            uint4 wv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
            const uint32_t sv0 = sp[0], sv1 = sp[32];
            const int kb = 256 * warp;
            consume_kb1(wv[0], sv0, 0, xrow, kb, t, acc[q]);
            consume_kb1(wv[1], sv0, 1, xrow, kb + 64, t, acc[q]);
            consume_kb1(wv[2], sv1, 0, xrow, kb + 128, t, acc2[q]);
            consume_kb1(wv[3], sv1, 1, xrow, kb + 192, t, acc2[q]);
          }
          __syncwarp();
          if (lane == 0) mbar_arrive(empty_a + 8 * slot);
          ++it;
        }
      }
      if (!nullc) {
#pragma unroll
        for (int q = 0; q < SUB; ++q)
          if (q < ntl)
            *reinterpret_cast<float4*>(red_s + ((warp * SUB + q) * 32 + lane) * 4) =
                make_float4(acc[q][0] + acc2[q][0], acc[q][1] + acc2[q][1], acc[q][2] + acc2[q][2], acc[q][3] + acc2[q][3]);
      }
      bar_sync(1, 32 * NCW);
      if (!nullc && tid < SUB * 64) {
        const int q = tid >> 6, gg = (tid >> 3) & 7, n = tid & 7;
        if (q < ntl && n < te) {
          const int ln = 4 * gg + (n >> 1), el = n & 1;
          float sg = 0.f, su = 0.f;
#pragma unroll
          for (int w = 0; w < NCW; ++w) {
            const float* pr = red_s + ((w * SUB + q) * 32 + ln) * 4;
            sg += pr[el]; su += pr[2 + el];
          }
          const int tile = t0 + q;
          const int jl = gbase + (tile >> 1) - j0;
          const int d = 8 * (tile & 1) + gg;
          reinterpret_cast<__nv_bfloat16*>(inter_s + jl * IGB + n * 32)[iperm(d)] = __float2bfloat16_rn(swiglu(sg * gs13, su * gs13));
        }
      }
      bar_sync(1, 32 * NCW);
    }
    if (tid == 0) mbar_arrive(xempty_a);   // X rows of this segment no longer read
    if (mk && tid == 0) mk[2] = gtime();
    // ---- W2: per stage, tps tiles' granule runs; warp w takes tiles w, w + 16; red.add partial y ----
    const int piece = ngr * GB;
    const int tps = min(32, STB / piece);
    const float gs2 = S.gs2[e];
    for (int r0 = 0; r0 < 256; r0 += tps, ++it) {
      const int nt = min(tps, 256 - r0);
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, A.err, 1);
      if (!nullc) {
        for (int ti = warp; ti < nt; ti += NCW) {
          const uint8_t* pp = smem + slot * STB + ti * piece;
          float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
          int jl = 0;
          for (; jl + 1 < ngr; jl += 2) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint8_t* gp = pp + (jl + h) * GB;
              const uint32_t wd = reinterpret_cast<const uint32_t*>(gp)[lane];
              const uint32_t sl = reinterpret_cast<const uint32_t*>(gp + 128)[g >> 2];
              const uint32_t sh = reinterpret_cast<const uint32_t*>(gp + 128)[(g >> 2) + 2];
              const uint32_t bg = (sl >> (8 * (g & 3))) & 0xFFu, b8 = (sh >> (8 * (g & 3))) & 0xFFu;
              const uint32_t Sg = (bg | (bg << 16)) * 16u + 0x75807580u, S8 = (b8 | (b8 << 16)) * 16u + 0x75807580u;
              const uint2 bq = *reinterpret_cast<const uint2*>(inter_s + (gbase + jl + h) * IGB + g * 32 + 8 * t);
              uint32_t o[4];
              deq4(wd, o);
              o[0] = bmul2(o[0], Sg); o[1] = bmul2(o[1], Sg); o[2] = bmul2(o[2], S8); o[3] = bmul2(o[3], S8);
              mma16816(h ? acc2 : acc, o[0], o[2], o[1], o[3], bq.x, bq.y);
            }
          }
          if (jl < ngr) {
            const uint8_t* gp = pp + jl * GB;
            const uint32_t wd = reinterpret_cast<const uint32_t*>(gp)[lane];
            const uint32_t sl = reinterpret_cast<const uint32_t*>(gp + 128)[g >> 2];
            const uint32_t sh = reinterpret_cast<const uint32_t*>(gp + 128)[(g >> 2) + 2];
            const uint32_t bg = (sl >> (8 * (g & 3))) & 0xFFu, b8 = (sh >> (8 * (g & 3))) & 0xFFu;
            const uint32_t Sg = (bg | (bg << 16)) * 16u + 0x75807580u, S8 = (b8 | (b8 << 16)) * 16u + 0x75807580u;
            const uint2 bq = *reinterpret_cast<const uint2*>(inter_s + (gbase + jl) * IGB + g * 32 + 8 * t);
            uint32_t o[4];
            deq4(wd, o);
            o[0] = bmul2(o[0], Sg); o[1] = bmul2(o[1], Sg); o[2] = bmul2(o[2], S8); o[3] = bmul2(o[3], S8);
            mma16816(acc, o[0], o[2], o[1], o[3], bq.x, bq.y);
          }
          const int row = 16 * (r0 + ti) + g;
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int n = 2 * t + hh;
            if (n < te) {
              float* yp = A.y + (size_t)(c0 + n) * KDIM + row;
              red_f32(yp, (acc[hh] + acc2[hh]) * gs2);
              red_f32(yp + 8, (acc[2 + hh] + acc2[2 + hh]) * gs2);
            }
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
    if (mk && tid == 0) mk[3] = gtime();
  }
  if (mk && tid == 0) mk[6] = gtime();
}

// finalize: one CTA per token (PDL secondary): sums of squares from the full y rows, post-norm + weighted sum, y -> 0
__global__ void __launch_bounds__(256) z1_fin(const Args A) {
  __shared__ float s_red[8][8];
  __shared__ float s_sc[8];
  __shared__ float s_w8[8];
  __shared__ int s_c[8];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tk = blockIdx.x;
  if (tid < 8) s_w8[tid] = __ldg(A.wts + tk * 8 + tid);
  float4 nwv[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) nwv[k] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * k + 4 * tid));
  pdl_wait();
  if (tid < 8) s_c[tid] = A.meta[tk * 8 + tid];
  __syncthreads();
  float ssp[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    ssp[j] = 0.f;
    const int c = s_c[j];
    if (c >= 0) {
      float4 v[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) v[k] = __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)c * KDIM + 1024 * k + 4 * tid));
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float zz;
        zz = bf16r(v[k].x) * nwv[k].x; a += zz * zz;
        zz = bf16r(v[k].y) * nwv[k].y; a += zz * zz;
        zz = bf16r(v[k].z) * nwv[k].z; a += zz * zz;
        zz = bf16r(v[k].w) * nwv[k].w; a += zz * zz;
      }
      ssp[j] = a;
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float a = ssp[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(FULL, a, o);
    if (lane == 0) s_red[warp][j] = a;
  }
  __syncthreads();
  if (tid < 8) {
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) a += s_red[w][tid];
    s_sc[tid] = (s_c[tid] >= 0) ? rsqrtf(a * (1.f / 4096.f) + 1e-8f) * s_w8[tid] : 0.f;
  }
  __syncthreads();
  float o[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i] = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int c = s_c[j];
    if (c >= 0) {
      const float r = s_sc[j];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float4* yp = reinterpret_cast<float4*>(A.y + (size_t)c * KDIM + 1024 * k + 4 * tid);
        const float4 v = __ldcg(yp);
        o[4 * k + 0] += bf16r(v.x) * nwv[k].x * r; o[4 * k + 1] += bf16r(v.y) * nwv[k].y * r;
        o[4 * k + 2] += bf16r(v.z) * nwv[k].z * r; o[4 * k + 3] += bf16r(v.w) * nwv[k].w * r;
        *yp = make_float4(0.f, 0.f, 0.f, 0.f);   // self-reset for the next call
      }
    }
  }
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint2 u;
    u.x = pack_bf2(o[4 * k], o[4 * k + 1]); u.y = pack_bf2(o[4 * k + 2], o[4 * k + 3]);
    *reinterpret_cast<uint2*>(A.out + (size_t)tk * KDIM + 1024 * k + 4 * tid) = u;
  }
}

void launch(const Args& a, int G) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(z1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaFuncSetAttribute(z1_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  z1_kernel<<<G, NTH, SMEM, st>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (a.mode & 8) return;
#if defined(Z1_NO_PDL)
  z1_fin<<<a.M, 256, 0, st>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st; cfg.attrs = at; cfg.numAttrs = 1;
  cfg.gridDim = dim3(a.M); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, z1_fin, a));
#endif
}
std::vector<int64_t> attrs() {
  cudaFuncSetAttribute(z1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaFuncSetAttribute(z1_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, z1_kernel);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, z1_kernel, NTH, SMEM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SMEM), occ};
}
}  // namespace z1

void z1_run(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13t, torch::Tensor ws2s,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor y, torch::Tensor meta,
            torch::Tensor err, torch::Tensor out, torch::Tensor marks, int64_t M, int64_t lo, int64_t G, int64_t mode) {
  TORCH_CHECK(M >= 1 && M <= z1::XR, "z1: M out of range");
  z1::Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ids = ids.data_ptr<int>(); a.wts = wts.data_ptr<float>();
  a.ws13t = reinterpret_cast<const uint8_t*>(ws13t.data_ptr()); a.ws2s = reinterpret_cast<const uint8_t*>(ws2s.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>(); a.gsc2 = gsc2.data_ptr<float>(); a.nw = nw.data_ptr<float>();
  a.y = y.data_ptr<float>(); a.meta = meta.data_ptr<int>(); a.err = err.data_ptr<int>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.marks = marks.numel() >= G * 16 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  a.M = (int)M; a.lo = (int)lo; a.mode = (int)mode;
  z1::launch(a, (int)G);
}
std::vector<int64_t> z1_attrs() { return z1::attrs(); }
"""

Z1_CPP = r"""
#include <torch/extension.h>
#include <vector>
void z1_run(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13t, torch::Tensor ws2s,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor y, torch::Tensor meta,
            torch::Tensor err, torch::Tensor out, torch::Tensor marks, int64_t M, int64_t lo, int64_t G, int64_t mode);
std::vector<int64_t> z1_attrs();
"""

Z1_EXPERT_BYTES = 768 * 36864


def z1_build(arch="sm_90a", verbose=False):
    from torch.utils.cpp_extension import load_inline
    gen = (["-gencode=arch=compute_90a,code=sm_90a"] if arch == "sm_90a"
           else ["-gencode=arch=compute_86,code=sm_86", "-DZ1_NST=1", "-DZ1_XR=2", "-DZ1_SUB=2", "-DZ1_NO_PDL"])
    tag = hashlib.sha1((Z1_CPP + Z1_CUDA + arch).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"z1_moe_{tag}", cpp_sources=Z1_CPP, cuda_sources=Z1_CUDA, functions=["z1_run", "z1_attrs"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills"] + gen, verbose=verbose)
    return mod, time.time() - t0


def z1_repack(st, w2, w2_sf, reencode, pack_words, bitpos):
    """ws13t: K2A's ws13 [32][64 rb][16 s][8 x 2048 codes | 8 x 256 scales] -> tile-major [32][512][16][2304] (same bytes).
    ws2s [32][256 tiles][256 granules][144]: per (16-row tile r, 16-dim granule j) the 32 lane words (lane 4g + t holds
    codes rows g / g+8 x k 2t, 2t+1, 2t+8, 2t+9 in K2A's BITPOS3 order) + 16 re-encoded scale bytes (rows 0..15)."""
    ws13 = st["ws13"]
    E = ws13.shape[0]
    codes13 = ws13[..., :16384].reshape(E, 64, 16, 8, 2048)
    sc13 = ws13[..., 16384:].reshape(E, 64, 16, 8, 256)
    ws13t = torch.cat((codes13, sc13), dim=-1).permute(0, 1, 3, 2, 4).reshape(E, 512, 16, 2304).contiguous()
    ws2s = torch.empty((E, 256, 256, 144), dtype=torch.uint8, device=ws13.device)
    for e in range(E):
        w = w2[e]
        sfb = w2_sf[e].view(torch.uint8) if w2_sf.dtype != torch.uint8 else w2_sf[e]
        bp, neg, zero, bad = reencode(sfb)
        if bad:
            raise ValueError("NaN block scale in W2")
        codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(4096, 4096)
        if bool(neg.any()) or bool(zero.any()):
            cb = codes.view(4096, 256, 16)
            cb = torch.where(neg.unsqueeze(-1), cb ^ 8, cb)
            cb = torch.where(zero.unsqueeze(-1), torch.zeros_like(cb), cb)
            codes = cb.reshape(4096, 4096)
        # [r 256][hi 2][g 8][j 256][kh 2][t 4][kl 2] -> [r][j][g][t][hi][kh][kl]
        c = codes.view(256, 2, 8, 256, 2, 4, 2).permute(0, 3, 2, 5, 1, 4, 6).reshape(256, 256, 32, 8)
        words = pack_words(c, bitpos)                                          # int32 [256][256][32]
        sc = bp.view(256, 16, 256).permute(0, 2, 1).contiguous()               # [r][j][rho]
        ws2s[e] = torch.cat((words.contiguous().view(torch.uint8).view(256, 256, 128), sc), dim=-1)
    return ws13t, ws2s.contiguous()


def z1_bufs(dev, G):
    return {
        "y": torch.zeros((64, 4096), dtype=torch.float32, device=dev),
        "meta": torch.zeros((64,), dtype=torch.int32, device=dev),
        "err": torch.zeros((8,), dtype=torch.int32, device=dev),
        "marks": torch.zeros((G * 16,), dtype=torch.int64, device=dev),
        "nomark": torch.zeros((0,), dtype=torch.int64, device=dev),
    }


def z1_fn(mod, st, lay, bufs, data, G, mode=0, mark=False):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    lo = int(data["config"]["local_start"])
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    mk = bufs["marks"] if mark else bufs["nomark"]

    def f():
        mod.z1_run(x, ids, wts, lay["ws13t"], lay["ws2s"], st["gsc13"], st["gsc2"], st["nw"], bufs["y"], bufs["meta"],
                   bufs["err"], out, mk, m, lo, G, mode)
        return out
    return f


def z1_case(bs, seed, dev, dist="normal"):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    if dist == "normal":
        z = torch.randn((256,), generator=g, device=dev)
        u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    else:
        scores = scores + torch.rand((256,), generator=g, device=dev).pow(6) * 4
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def z1_touched(data):
    ids = data["topk_ids"].long() - int(data["config"]["local_start"])
    return int(torch.unique(ids[(ids >= 0) & (ids < 32)]).numel())


def z1_rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


class Z1Timer:
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
                out = fn()
        torch.cuda.synchronize()
        return gr, out

    def time(self, gr):
        with torch.cuda.stream(self.stream):
            torch.sum(self.flush_buf, dim=0, out=self.sink)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    def overhead(self, n=100):
        tiny = torch.zeros(1, device="cuda")
        with torch.cuda.stream(self.stream):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                tiny.add_(0)
        torch.cuda.synchronize()
        return statistics.median(self.time(gr) for _ in range(n))


def z1_marks(bufs, G):
    mv = bufs["marks"].view(-1, 16)[:G].tolist()
    t0 = min(r[0] for r in mv)

    def p(k):
        v = sorted((r[k] - t0) / 1e3 for r in mv if r[k] > 0)
        return (statistics.median(v), v[-1]) if v else (float("nan"), float("nan"))
    return {nm: p(k) for nm, k in (("first_copy", 1), ("w13_end", 2), ("w2_end", 3), ("end", 6), ("prod_done", 10))}


def z1_lab(st, w, k2a_fn, k2a_state, reencode, pack_words, bitpos, log=print, rounds=60, inputs=4, seeds0=919001,
           sizes=(1, 4, 8), arch="sm_90a"):
    """R-moe1-2: arms K (shipped K2A), Z (Z1), Z0 (null consumers), S (null consumers, no finalize). Same routings as R-moe1-0/1."""
    dev = st["ws13"].device
    t0 = time.time()
    ws13t, ws2s = z1_repack(st, w["w2"], w["w2_sf"], reencode, pack_words, bitpos)
    lay = {"ws13t": ws13t, "ws2s": ws2s}
    torch.cuda.synchronize()
    mod, dt = z1_build(arch)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    a = list(mod.z1_attrs())
    log(f"[z1] repack {time.time() - t0 - dt:.1f}s, build {dt:.0f}s; attrs (regs, local, smem, occ) {a}; {sms} SMs")
    if a[3] < 1:
        log("[z1] cannot be resident: lab skipped")
        return []
    G = sms
    timer = Z1Timer()
    ov = timer.overhead()
    log(f"[z1] graph overhead {ov:.2f} us; G {G}")
    rows = []
    for bs in sizes:
        for r in range(inputs):
            d = z1_case(bs, seeds0 + 1000 * bs + r, dev)
            ne = z1_touched(d)
            B = ne * Z1_EXPERT_BYTES
            spec = {"Z": 0, "Z0": 1, "S": 9}
            bufs = {nm: z1_bufs(dev, G) for nm in spec}
            arms = {"K": lambda d=d: k2a_fn(d, k2a_state)}
            arms.update({nm: z1_fn(mod, st, lay, bufs[nm], d, G, md) for nm, md in spec.items()})
            ref = arms["K"]().float().clone()
            torch.cuda.synchronize()
            o1 = arms["Z"]().float().clone()
            o2 = arms["Z"]().float().clone()
            torch.cuda.synchronize()
            rel = (z1_rel(o1, ref), z1_rel(o2, ref), bool(torch.isfinite(o1).all()))
            graphs = {nm: timer.capture(fn)[0] for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            o3 = arms["Z"]().float()
            torch.cuda.synchronize()
            rel3 = z1_rel(o3, ref)
            tl = {}
            for nm in ("Z", "Z0"):
                bufs[nm]["marks"].zero_()
                f = z1_fn(mod, st, lay, bufs[nm], d, G, spec[nm] | 8, mark=True)
                with torch.cuda.stream(timer.stream):
                    torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                    f()
                torch.cuda.synchronize()
                tl[nm] = z1_marks(bufs[nm], G)
                bufs[nm]["y"].zero_()   # the marked run skipped the finalize (y self-reset)
            e_all = {nm: [int(v) for v in bufs[nm]["err"].tolist()[:2]] for nm in spec}
            rows.append({"bs": bs, "ne": ne, "B": B, "us": us, "rel": rel, "rel3": rel3, "err": e_all, "tl": tl})
            log(f"[z1] bs={bs} r={r} experts {ne} ({B / 1e6:.1f} MB): " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items())
                + " | " + " ".join(f"{nm}/K {us[nm] / us['K']:.3f}" for nm in us if nm != "K"))
            log(f"[z1]   rel Z vs K {rel[0]:.1e}/{rel[1]:.1e}/{rel[2]} after-timing {rel3:.1e} | err {e_all}")
            for nm, s in tl.items():
                log(f"[z1]   TL {nm} (no fin): " + " ".join(f"{k} {a2:.1f}/{b2:.1f}" for k, (a2, b2) in s.items()))
            del graphs, bufs
            torch.cuda.empty_cache()
    log("[z1] GATE (R-moe1-2: median Z/K bs1 <= 0.85 AND bs8 <= 0.92, rel <= 1e-3, err 0; else stop)")
    verdict = True
    for bs in sizes:
        rr = [x for x in rows if x["bs"] == bs]
        if not rr:
            continue
        mr = statistics.median(x["us"]["Z"] / x["us"]["K"] for x in rr)
        ok_c = all(x["rel"][0] <= 1e-3 and x["rel"][1] <= 1e-3 and x["rel"][2] and x["rel3"] <= 1e-3
                   and x["err"]["Z"][0] == 0 for x in rr)
        lim = 0.85 if bs == 1 else 0.92
        okb = ok_c and mr <= lim
        if bs in (1, 8):
            verdict &= okb
        log(f"[z1]   bs={bs}: median Z/K {mr:.3f} (limit {lim}{'' if bs in (1, 8) else ', info'}), correctness "
            f"{'ok' if ok_c else 'FAIL'} -> {'pass' if okb else 'FAIL'}")
    log(f"[z1] GATE {'GO' if verdict else 'NO-GO'}")
    del timer, lay, ws13t, ws2s
    torch.cuda.empty_cache()
    return rows

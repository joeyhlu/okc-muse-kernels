"""fac_bench.py (R-fac-1): VA1 (decode full, q_len 1) consumer microbench -- ship tile_compute1<4> vs the FAC variants, in a
standalone copy of va1_main's ring (TILE 64, ST 4, 16 KB stages, 2 CTAs/SM, 4 consumer + 4 producer warps, full/empty
mbarriers, VA's swizzles and loader), without the segment / merge logic.

Consumer variants (CV):
  0 ship : tile_compute1<4> verbatim (S = Q K^T, M = 8 heads padded to 16; O = P V padded; stage released after PV).
  1 TP   : transposed MMAs. QK as S^T = K Q^T (M = keys, N = 8 heads: 8 HMMA instead of 16), P^T by 4 movmatrix.trans,
           PV as O^T = V^T P^T (M = dims, N = heads: 8 HMMA instead of 16); same K / V smem reads, same F2FP count,
           16 accumulators instead of 32.
  2 TP+ER: TP with early release: all K (4 LDS.128) and V (8 LDS.64 + PRMT) fragments are read at tile start, the
           stage's empty arrive follows at once (address-dependent on every load, R-sfx2 style), compute from registers.
  3 ER   : ship math with early release.
Modes: gather (VA's random-row cp.async gather from a page-size-1 cache, cold L2) and nocopy (the producer copies the
first ST tiles, then only arrives: tiles already in smem, consumer-only time). --stamp adds per-warp clock64 sums:
hold = full-wait return -> empty arrive, wait = full-wait time.

  local (RTX 3070, correctness):  python kernels/lab/fac_bench.py --local
  box (H200):                     python3 -u kernels/lab/fac_bench.py --out=box_out/fac/bench.json
"""
import argparse
import json
import math
import os
import statistics
import time

import torch
from torch.utils.cpp_extension import load_inline

CPP = r"""
#include <torch/extension.h>
void fac_run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor q, torch::Tensor o, torch::Tensor ml,
             torch::Tensor st, int64_t N, double cl2, int64_t stamp, int64_t cv, int64_t nocopy, int64_t grid, int64_t stn);
int64_t fac_occ(int64_t cv, int64_t stn);
"""

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace fac {
constexpr int TILE = 64, ROWB = 128, STAGE = 2 * TILE * ROWB;
template <int ST> constexpr int smem_of() { return 75872 + (ST - 4) * STAGE; }   // ST 4 = k3_gc_21 va_smem<4>(): 2 CTAs/SM
constexpr unsigned FULL = 0xffffffffu;
constexpr float P_OFFSET = 8.0f;

struct Prm {
  const uint8_t* kc; const uint8_t* vc; const int* pt; const uint32_t* q;
  float* o; float* ml; long long* st; int N; float cl2; int stamp; uint32_t zero;
};

__device__ __forceinline__ uint32_t sa(const void* x) { return (uint32_t)__cvta_generic_to_shared(x); }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t v) {
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xFFFFu), __NV_E4M3);
  return (uint32_t)r.x | ((uint32_t)r.y << 16);
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t movt(uint32_t x) {
  uint32_t y;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ void mwait(uint64_t* b, int par) {
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra W;\n}\n" ::"r"(sa(b)), "r"(par) : "memory");
#else
  uint32_t ok = 0;
  do asm volatile("{\n.reg .pred P;\nmbarrier.test_wait.parity.shared.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}\n"
                  : "=r"(ok) : "r"(sa(b)), "r"(par) : "memory");
  while (!ok);
#endif
}
__device__ __forceinline__ void marrive(uint32_t a) {
  asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(a) : "memory");
}
__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa(smem_ptr)), "l"(gptr), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ int kswz(int x, int ch) { return ch ^ (4 * (x & 1)); }
__device__ __forceinline__ int vswz(int x, int ch) { return ch ^ (2 * ((x >> 1) & 3)); }

__device__ __forceinline__ uint2 vld(const uint8_t* Vbuf, int x, int h, int g) {
  return *reinterpret_cast<const uint2*>(Vbuf + x * ROWB + 16 * ((4 * h + (g >> 1)) ^ (2 * ((x >> 1) & 3))) + 8 * (g & 1));
}
__device__ __forceinline__ void v_frags(const uint8_t* Vbuf, int x0, int h, int g, uint32_t (&w01)[4], uint32_t (&w23)[4]) {
  const uint2 va = vld(Vbuf, x0, h, g), vb = vld(Vbuf, x0 + 1, h, g), vc = vld(Vbuf, x0 + 8, h, g), vd = vld(Vbuf, x0 + 9, h, g);
  w01[0] = __byte_perm(va.x, vb.x, 0x5140); w01[1] = __byte_perm(va.x, vb.x, 0x7362);
  w01[2] = __byte_perm(va.y, vb.y, 0x5140); w01[3] = __byte_perm(va.y, vb.y, 0x7362);
  w23[0] = __byte_perm(vc.x, vd.x, 0x5140); w23[1] = __byte_perm(vc.x, vd.x, 0x7362);
  w23[2] = __byte_perm(vc.y, vd.y, 0x5140); w23[3] = __byte_perm(vc.y, vd.y, 0x7362);
}
__device__ __forceinline__ uint32_t kword(const uint4& v, int s) { return s == 0 ? v.x : s == 1 ? v.y : s == 2 ? v.z : v.w; }

// ---- CV 0 / 3: ship math (tile_compute1<4> of k3_gc_21, W = 0, no causal limit); ER = early release
template <bool ER>
__device__ __forceinline__ void tile_ship(const uint8_t* __restrict__ Kbuf, const uint8_t* __restrict__ Vbuf, const int kw,
                                          const int h, const int g, const int t, const int lane, const uint32_t (&qf)[4][4],
                                          float (&o)[8][4], float& mA, float& lA, const float cl2, const bool need,
                                          const int tile_abs, const int key_lim, const uint32_t ea, const uint32_t zero,
                                          long long& tarr, const int stamp) {
  constexpr int NT = 4;
  float sc[NT][4];
  uint4 kv[NT];
  uint32_t w01[2][4], w23[2][4];
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) {
    const int key = kw + 8 * jn + g;
    kv[jn] = *reinterpret_cast<const uint4*>(Kbuf + key * ROWB + 16 * ((4 * h + t) ^ (4 * (key & 1))));
  }
  if (ER) {
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01[kk], w23[kk]);
    const uint32_t dep = kv[0].x ^ kv[1].x ^ kv[2].x ^ kv[3].x ^ w01[0][0] ^ w23[0][0] ^ w01[1][0] ^ w23[1][0];
    __syncwarp();
    if (lane == 0) marrive(ea + (dep & zero));
    if (stamp) tarr = clock64();
  }
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) {
    const uint32_t wv[4] = {kv[jn].x, kv[jn].y, kv[jn].z, kv[jn].w};
    sc[jn][0] = 0.f; sc[jn][1] = 0.f; sc[jn][2] = 0.f; sc[jn][3] = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s)
      mma16816(sc[jn], qf[s][0], 0u, qf[s][2], 0u, e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
  }
  if (need) {
#pragma unroll
    for (int jn = 0; jn < NT; ++jn)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        if (!(tile_abs + kw + 8 * jn + 2 * t + e < key_lim)) sc[jn][e] = neg_inf();
  }
  float xA = neg_inf();
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) xA = fmaxf(xA, fmaxf(sc[jn][0], sc[jn][1]));
  xA = fmaxf(xA, __shfl_xor_sync(FULL, xA, 1));
  xA = fmaxf(xA, __shfl_xor_sync(FULL, xA, 2));
  const float nA = fmaxf(mA, xA * cl2);
  const float uA = (nA == neg_inf()) ? 0.f : nA;
  const float aA = ex2(mA - uA);
  const float bA = uA - P_OFFSET;
  mA = nA;
  uint32_t pa[NT / 2][2];
  float psA = 0.f;
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk)
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int jn = 2 * kk + hf;
      const float p0 = ex2(fmaf(sc[jn][0], cl2, -bA));
      const float p1 = ex2(fmaf(sc[jn][1], cl2, -bA));
      psA += p0 + p1;
      pa[kk][hf] = pack_h2(p0, p1);
    }
  lA = lA * aA + psA;
  if (__any_sync(FULL, aA != 1.f)) {
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] *= aA; o[jn][1] *= aA; }
  }
  uint32_t vdep = 0u;
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk) {
    if (!ER) {
      v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01[kk], w23[kk]);
      if (kk == NT / 2 - 1) vdep = w01[kk][0] ^ w23[kk][0];
    }
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      const uint32_t s01 = w01[kk][jn >> 1], s23 = w23[kk][jn >> 1];
      const uint32_t b0 = e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01);
      const uint32_t b1 = e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23);
      mma16816(o[jn], pa[kk][0], 0u, pa[kk][1], 0u, b0, b1);
    }
  }
  if (!ER) {
    __syncwarp();
    if (lane == 0) marrive(ea + (vdep & zero));
    if (stamp) tarr = clock64();
  }
}

// ---- CV 1 / 2: transposed MMAs (S^T = K Q^T, movmatrix P^T, O^T = V^T P^T); ER = early release
template <bool ER>
__device__ __forceinline__ void tile_tp(const uint8_t* __restrict__ Kbuf, const uint8_t* __restrict__ Vbuf, const int kw,
                                        const int h, const int g, const int t, const int lane, const uint32_t (&qb)[4][2],
                                        float (&oT)[4][4], float (&m)[2], float (&l)[2], const float cl2, const bool need,
                                        const int tile_abs, const int key_lim, const uint32_t ea, const uint32_t zero,
                                        long long& tarr, const int stamp) {
  uint4 kA[2], kB[2];
  uint32_t w01[2][4], w23[2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const int ka = kw + 16 * mt + g, kb = ka + 8;
    kA[mt] = *reinterpret_cast<const uint4*>(Kbuf + ka * ROWB + 16 * ((4 * h + t) ^ (4 * (ka & 1))));
    kB[mt] = *reinterpret_cast<const uint4*>(Kbuf + kb * ROWB + 16 * ((4 * h + t) ^ (4 * (kb & 1))));
  }
  if (ER) {
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01[kk], w23[kk]);
    const uint32_t dep = kA[0].x ^ kA[1].x ^ kB[0].x ^ kB[1].x ^ w01[0][0] ^ w23[0][0] ^ w01[1][0] ^ w23[1][0];
    __syncwarp();
    if (lane == 0) marrive(ea + (dep & zero));
    if (stamp) tarr = clock64();
  }
  float sT[2][4];   // [mt]: {head 2t key g, head 2t+1 key g, head 2t key g+8, head 2t+1 key g+8}, keys kw + 16 mt + .
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    sT[mt][0] = 0.f; sT[mt][1] = 0.f; sT[mt][2] = 0.f; sT[mt][3] = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t wa = kword(kA[mt], s), wb = kword(kB[mt], s);
      mma16816(sT[mt], e4m3x2_to_h2(wa), e4m3x2_to_h2(wb), e4m3x2_to_h2(wa >> 16), e4m3x2_to_h2(wb >> 16), qb[s][0], qb[s][1]);
    }
  }
  if (need) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int ak = tile_abs + kw + 16 * mt + g;
      if (!(ak < key_lim)) { sT[mt][0] = neg_inf(); sT[mt][1] = neg_inf(); }
      if (!(ak + 8 < key_lim)) { sT[mt][2] = neg_inf(); sT[mt][3] = neg_inf(); }
    }
  }
  float x0 = fmaxf(fmaxf(sT[0][0], sT[0][2]), fmaxf(sT[1][0], sT[1][2]));
  float x1 = fmaxf(fmaxf(sT[0][1], sT[0][3]), fmaxf(sT[1][1], sT[1][3]));
#pragma unroll
  for (int d = 4; d < 32; d <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, d));
    x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, d));
  }
  const float n0 = fmaxf(m[0], x0 * cl2), n1 = fmaxf(m[1], x1 * cl2);
  const float u0 = (n0 == neg_inf()) ? 0.f : n0, u1 = (n1 == neg_inf()) ? 0.f : n1;
  const float a0 = ex2(m[0] - u0), a1 = ex2(m[1] - u1);
  const float b0 = u0 - P_OFFSET, b1 = u1 - P_OFFSET;
  m[0] = n0; m[1] = n1;
  uint32_t pT[2][2];
  float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const float p0 = ex2(fmaf(sT[mt][0], cl2, -b0)), p1 = ex2(fmaf(sT[mt][1], cl2, -b1));
    const float p2 = ex2(fmaf(sT[mt][2], cl2, -b0)), p3 = ex2(fmaf(sT[mt][3], cl2, -b1));
    ps0 += p0 + p2; ps1 += p1 + p3;
    pT[mt][0] = movt(pack_h2(p0, p1));   // -> {P[head g][key 2t], P[head g][key 2t+1]} of k-step mt
    pT[mt][1] = movt(pack_h2(p2, p3));   // -> keys 2t + 8, 2t + 9
  }
  l[0] = l[0] * a0 + ps0;
  l[1] = l[1] * a1 + ps1;
  if (__any_sync(FULL, (a0 != 1.f) || (a1 != 1.f))) {
#pragma unroll
    for (int pm = 0; pm < 4; ++pm) { oT[pm][0] *= a0; oT[pm][1] *= a1; oT[pm][2] *= a0; oT[pm][3] *= a1; }
  }
  uint32_t vdep = 0u;
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    if (!ER) {
      v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01[kk], w23[kk]);
      if (kk == 1) vdep = w01[kk][0] ^ w23[kk][0];
    }
#pragma unroll
    for (int pm = 0; pm < 4; ++pm)   // rows g / g + 8 of m-tile pm = dims 8 g + 2 pm / + 1
      mma16816(oT[pm], e4m3x2_to_h2(w01[kk][pm]), e4m3x2_to_h2(w01[kk][pm] >> 16), e4m3x2_to_h2(w23[kk][pm]),
               e4m3x2_to_h2(w23[kk][pm] >> 16), pT[kk][0], pT[kk][1]);
  }
  if (!ER) {
    __syncwarp();
    if (lane == 0) marrive(ea + (vdep & zero));
    if (stamp) tarr = clock64();
  }
}

template <int CV, bool NOCOPY, int ST>
__global__ void __launch_bounds__(256, 2) fac_main(const Prm p) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * STAGE);
  uint64_t* empt = full + ST;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < ST) {
    asm volatile("mbarrier.init.shared.b64 [%0], 128;\n" ::"r"(sa(full + tid)) : "memory");
    asm volatile("mbarrier.init.shared.b64 [%0], 4;\n" ::"r"(sa(empt + tid)) : "memory");
  }
  __syncthreads();
  const int v0 = (int)(((long long)p.N * blockIdx.x) / gridDim.x), v1 = (int)(((long long)p.N * (blockIdx.x + 1)) / gridDim.x);
  const int nt = (v1 - v0 + TILE - 1) / TILE;
  if (warp >= 4) {   // VA1 producer (page size 1, Hkv 2, hp 0)
    const int q = tid - 128, ch = q & 7, r0 = q >> 3;
    const uint8_t* kc = p.kc + 16 * ch;
    const uint8_t* vc = p.vc + 16 * ch;
    int pg[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) pg[k] = (v0 + r0 + 16 * k < v1) ? __ldg(p.pt + v0 + r0 + 16 * k) : -1;
    for (int x = 0; x < nt; ++x) {
      const int st = x % ST;
      if (NOCOPY && x >= ST) {
        mwait(empt + st, ((x / ST) - 1) & 1);
        marrive(sa(full + st));
        continue;
      }
      int pgn[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int pos = v0 + (x + 1) * TILE + r0 + 16 * k;
        pgn[k] = (pos < v1) ? __ldg(p.pt + pos) : -1;
      }
      if (x >= ST) mwait(empt + st, ((x / ST) - 1) & 1);
      uint8_t* stg = smem + st * STAGE;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int r = r0 + 16 * k;
        const size_t go = pg[k] >= 0 ? (size_t)pg[k] * ROWB : (size_t)0;
        cp_async16(stg + r * ROWB + 16 * kswz(r, ch), kc + go, pg[k] >= 0 ? 16 : 0);
        cp_async16(stg + TILE * ROWB + r * ROWB + 16 * vswz(r, ch), vc + go, pg[k] >= 0 ? 16 : 0);
        pg[k] = pgn[k];
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(sa(full + st)) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    return;
  }
  const int h = warp & 1, kw = 32 * (warp >> 1), g = lane >> 2, t = lane & 3;
  const uint32_t* qa = p.q + (h * 8 + g) * 32;
  long long hold = 0, wsum = 0, tarr = 0;
  float* ob = p.o + ((size_t)blockIdx.x * 4 + warp) * 8 * 64;
  float* mb = p.ml + ((size_t)blockIdx.x * 4 + warp) * 8 * 2;
  if (CV == 4) {   // skeleton consumer: wait full, release at once (no LDS, no math)
    for (int i = 0; i < nt; ++i) {
      const int st = i % ST;
      mwait(full + st, (i / ST) & 1);
      __syncwarp();
      if (lane == 0) marrive(sa(empt + st));
    }
#pragma unroll 1
    for (int e = lane; e < 8 * 64; e += 32) ob[e] = 0.f;
    if (lane < 16) mb[lane] = (lane & 1) ? 1.f : 0.f;
  } else if (CV == 0 || CV == 3) {
    uint32_t qf[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) { qf[s][0] = qa[8 * t + 2 * s]; qf[s][2] = qa[8 * t + 2 * s + 1]; qf[s][1] = 0u; qf[s][3] = 0u; }
    float o[8][4];
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] = 0.f; o[jn][1] = 0.f; o[jn][2] = 0.f; o[jn][3] = 0.f; }
    float mA = neg_inf(), lA = 0.f;
    for (int i = 0; i < nt; ++i) {
      const int st = i % ST, ts = v0 + i * TILE;
      const long long tw = p.stamp ? clock64() : 0;
      mwait(full + st, (i / ST) & 1);
      const long long tr = p.stamp ? clock64() : 0;
      const uint8_t* Kb = smem + st * STAGE;
      tile_ship<CV == 3>(Kb, Kb + TILE * ROWB, kw, h, g, t, lane, qf, o, mA, lA, p.cl2, ts + TILE > v1, ts, v1, sa(empt + st), p.zero, tarr, p.stamp);
      if (p.stamp) { hold += tarr - tr; wsum += tr - tw; }
    }
    lA += __shfl_xor_sync(FULL, lA, 1);
    lA += __shfl_xor_sync(FULL, lA, 2);
#pragma unroll
    for (int jn = 0; jn < 8; ++jn)
#pragma unroll
      for (int e = 0; e < 2; ++e) ob[g * 64 + 16 * t + 8 * e + jn] = o[jn][e];
    if (t == 0) { mb[g * 2] = mA; mb[g * 2 + 1] = lA; }
  } else {
    uint32_t qb[4][2];
#pragma unroll
    for (int s = 0; s < 4; ++s) { qb[s][0] = qa[8 * t + 2 * s]; qb[s][1] = qa[8 * t + 2 * s + 1]; }
    float oT[4][4];
#pragma unroll
    for (int pm = 0; pm < 4; ++pm) { oT[pm][0] = 0.f; oT[pm][1] = 0.f; oT[pm][2] = 0.f; oT[pm][3] = 0.f; }
    float m[2] = {neg_inf(), neg_inf()}, l[2] = {0.f, 0.f};
    for (int i = 0; i < nt; ++i) {
      const int st = i % ST, ts = v0 + i * TILE;
      const long long tw = p.stamp ? clock64() : 0;
      mwait(full + st, (i / ST) & 1);
      const long long tr = p.stamp ? clock64() : 0;
      const uint8_t* Kb = smem + st * STAGE;
      tile_tp<CV == 2>(Kb, Kb + TILE * ROWB, kw, h, g, t, lane, qb, oT, m, l, p.cl2, ts + TILE > v1, ts, v1, sa(empt + st), p.zero, tarr, p.stamp);
      if (p.stamp) { hold += tarr - tr; wsum += tr - tw; }
    }
#pragma unroll
    for (int d = 4; d < 32; d <<= 1) { l[0] += __shfl_xor_sync(FULL, l[0], d); l[1] += __shfl_xor_sync(FULL, l[1], d); }
#pragma unroll
    for (int pm = 0; pm < 4; ++pm) {
      ob[(2 * t) * 64 + 8 * g + 2 * pm] = oT[pm][0];
      ob[(2 * t + 1) * 64 + 8 * g + 2 * pm] = oT[pm][1];
      ob[(2 * t) * 64 + 8 * g + 2 * pm + 1] = oT[pm][2];
      ob[(2 * t + 1) * 64 + 8 * g + 2 * pm + 1] = oT[pm][3];
    }
    if (g == 0) { mb[4 * t] = m[0]; mb[4 * t + 1] = l[0]; mb[4 * t + 2] = m[1]; mb[4 * t + 3] = l[1]; }
  }
  if (p.stamp && lane == 0) { p.st[(blockIdx.x * 4 + warp) * 2] = hold; p.st[(blockIdx.x * 4 + warp) * 2 + 1] = wsum; }
}

template <int CV, bool NC, int ST>
void go(const Prm& p, int grid, cudaStream_t s) {
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(fac_main<CV, NC, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_of<ST>())); attr = true; }
  fac_main<CV, NC, ST><<<grid, 256, smem_of<ST>(), s>>>(p);
  C10_CUDA_CHECK(cudaGetLastError());
}
template <int CV, int ST>
int occ() {
  int nb = 0;
  C10_CUDA_CHECK(cudaFuncSetAttribute(fac_main<CV, false, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_of<ST>()));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, fac_main<CV, false, ST>, 256, smem_of<ST>()));
  return nb;
}
}  // namespace fac

void fac_run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor q, torch::Tensor o, torch::Tensor ml,
             torch::Tensor st, int64_t N, double cl2, int64_t stamp, int64_t cv, int64_t nocopy, int64_t grid, int64_t stn) {
  fac::Prm p;
  p.kc = kc.data_ptr<uint8_t>(); p.vc = vc.data_ptr<uint8_t>(); p.pt = pt.data_ptr<int>();
  p.q = reinterpret_cast<const uint32_t*>(q.data_ptr()); p.o = o.data_ptr<float>(); p.ml = ml.data_ptr<float>();
  p.st = reinterpret_cast<long long*>(st.data_ptr<int64_t>()); p.N = (int)N; p.cl2 = (float)cl2; p.stamp = (int)stamp; p.zero = 0u;
  auto s = at::cuda::getCurrentCUDAStream();
  const int g = (int)grid;
  const int code = (int)(stn * 100 + cv * 2 + (nocopy ? 1 : 0));
#define FAC_C(ST_, CV_, NC_) case ST_ * 100 + CV_ * 2 + NC_: fac::go<CV_, NC_ == 1, ST_>(p, g, s); break;
  switch (code) {
    FAC_C(4, 0, 0) FAC_C(4, 0, 1) FAC_C(4, 1, 0) FAC_C(4, 1, 1) FAC_C(4, 2, 0) FAC_C(4, 2, 1) FAC_C(4, 3, 0) FAC_C(4, 3, 1)
    FAC_C(4, 4, 0) FAC_C(4, 4, 1) FAC_C(6, 0, 0) FAC_C(6, 2, 0) FAC_C(6, 4, 0)
    default: TORCH_CHECK(false, "fac: arm not instantiated");
  }
}
int64_t fac_occ(int64_t cv, int64_t stn) {
  if (stn == 6) return cv == 0 ? fac::occ<0, 6>() : cv == 2 ? fac::occ<2, 6>() : fac::occ<4, 6>();
  switch (cv) { case 0: return fac::occ<0, 4>(); case 1: return fac::occ<1, 4>(); case 2: return fac::occ<2, 4>(); case 3: return fac::occ<3, 4>(); default: return fac::occ<4, 4>(); }
}
"""

NAMES = {0: "ship", 1: "tp", 2: "tp_er", 3: "er", 4: "skel"}


def build():
    cap = torch.cuda.get_device_capability()
    flags = ["-O3", "-Xptxas=-v", "-lineinfo"]
    if cap >= (9, 0):
        flags.append("-gencode=arch=compute_90a,code=sm_90a")
    t0 = time.time()
    mod = load_inline(name=f"fac_bench_{cap[0]}{cap[1]}", cpp_sources=CPP, cuda_sources=CUDA, functions=["fac_run", "fac_occ"],
                      extra_cuda_cflags=flags, verbose=False)
    print(f"[fac] built in {time.time() - t0:.0f}s", flush=True)
    return mod


def make(N, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    kf = torch.randn((N, 128), generator=g, device=dev) * 0.8
    kf += torch.randn((1, 128), generator=g, device=dev) * 0.6          # a shared direction (QK-norm-like bias)
    kc = kf.to(torch.float8_e4m3fn).view(torch.uint8)
    vc = torch.randn((N, 128), generator=g, device=dev).to(torch.float8_e4m3fn).view(torch.uint8)
    del kf
    pt = torch.randperm(N, generator=g, device=dev).to(torch.int32)    # page size 1: a random row per key
    qf = torch.randn((16, 64), generator=g, device=dev)
    qf = qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + 1e-5)
    q = qf.to(torch.float16).contiguous()
    return kc, vc, pt, q


def merge(o, ml, grid):
    """per-warp (o, m, l) -> per CTA per head [grid, 16, 64] (warps h, h + 2 share kv head h)."""
    o = o.view(grid, 4, 8, 64).double()
    ml = ml.view(grid, 4, 8, 2).double()
    out = torch.empty((grid, 16, 64), dtype=torch.float64, device=o.device)
    for h in range(2):
        m = torch.stack([ml[:, h, :, 0], ml[:, h + 2, :, 0]])           # [2, grid, 8]
        l = torch.stack([ml[:, h, :, 1], ml[:, h + 2, :, 1]])
        oo = torch.stack([o[:, h], o[:, h + 2]])                       # [2, grid, 8, 64]
        mx = m.max(0).values
        w = torch.exp2(m - mx)
        out[:, 8 * h:8 * h + 8] = (w[..., None] * oo).sum(0) / (w * l).sum(0)[..., None]
    return out


def reference(kc, vc, pt, q, cl2, N, grid, ctas):
    res = {}
    for c in ctas:
        v0, v1 = N * c // grid, N * (c + 1) // grid
        rows = pt[v0:v1].long()
        K = kc[rows].view(torch.float8_e4m3fn).float().view(-1, 2, 64)
        V = vc[rows].view(torch.float8_e4m3fn).float().view(-1, 2, 64)
        Q = q.float().view(2, 8, 64)
        out = torch.empty((16, 64), dtype=torch.float64, device=kc.device)
        for h in range(2):
            s = (Q[h] @ K[:, h].T).double() * cl2                         # log2 units
            p = torch.exp2(s - s.max(-1, keepdim=True).values)
            out[8 * h:8 * h + 8] = (p @ V[:, h].double()) / p.sum(-1, keepdim=True)
        res[c] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--keys", default="1270000,2550000")   # d16f / d32f key counts (judge 325.6 / 653.8 MB at 256 B per key)
    ap.add_argument("--cvs", default="0,1,2,3")
    ap.add_argument("--arms6", default="", help="ST 6 arms (gather only), e.g. 0,2,4")
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--out", default="box_out/fac/bench.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    mod = build()
    cvs = [int(x) for x in args.cvs.split(",")]
    arms6 = [int(x) for x in args.arms6.split(",") if x]
    occ = {**{cv: int(mod.fac_occ(cv, 4)) for cv in cvs}, **{f"{cv}@6": int(mod.fac_occ(cv, 6)) for cv in arms6}}
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = 2 * sms
    print(f"[fac] {torch.cuda.get_device_name()} sms {sms} occupancy {occ}", flush=True)
    cl2 = (1.0 / 8.0) * 1.4426950408889634
    keys = [264 * 700 + 37] if args.local else [int(x) for x in args.keys.split(",")]
    flush = torch.empty((256 << 20) // 4, dtype=torch.float32, device=dev)
    out = {"occ": occ, "grid": grid, "runs": {}}
    for N in keys:
        kc, vc, pt, q = make(N, 1234, dev)
        o = torch.zeros((grid * 4 * 8 * 64,), dtype=torch.float32, device=dev)
        ml = torch.zeros((grid * 4 * 8 * 2,), dtype=torch.float32, device=dev)
        st = torch.zeros((grid * 4 * 2,), dtype=torch.int64, device=dev)
        ctas = [0, grid // 2, grid - 1]
        ref = reference(kc, vc, pt, q, cl2, N, grid, ctas)
        base = None
        tiles = math.ceil(N / grid / 64)
        for cv, stn in [(c, 4) for c in cvs] + [(c, 6) for c in arms6]:
            # correctness (gather mode)
            mod.fac_run(kc, vc, pt, q, o, ml, st, N, cl2, 0, cv, 0, grid, stn)
            torch.cuda.synchronize()
            res = merge(o, ml, grid)
            err = max(float((res[c] - ref[c]).abs().max() / ref[c].abs().max()) for c in ctas)
            if base is None and cv != 4:
                base = res
            dship = float((res - base).abs().max() / base.abs().max()) if base is not None else 0.0
            rec = {"err_ref": err, "d_ship": dship, "tiles_per_cta": tiles}
            for nc in ((0, 1) if stn == 4 else (0,)):
                mode = "nocopy" if nc else "gather"
                mod.fac_run(kc, vc, pt, q, o, ml, st, N, cl2, 1, cv, nc, grid, stn)   # stamp run
                torch.cuda.synchronize()
                s2 = st.view(grid, 4, 2).double()
                hold = float(s2[:, :, 0].mean()) / tiles
                wait = float(s2[:, :, 1].mean()) / tiles
                ts = []
                for _ in range(1 if args.local else args.reps):
                    flush.zero_()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    mod.fac_run(kc, vc, pt, q, o, ml, st, N, cl2, 0, cv, nc, grid, stn)
                    e1.record()
                    torch.cuda.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1e3)
                us = statistics.median(ts)
                rec[mode] = {"us": us, "us_all": ts, "hold_cyc_per_tile": hold, "wait_cyc_per_tile": wait,
                             "ns_per_tile": us * 1e3 / tiles, "TBs": N * 256 / us / 1e6}
            name = NAMES[cv] + ("" if stn == 4 else "@st6")
            out["runs"][f"{N}|{name}"] = rec
            g, c = rec["gather"], rec.get("nocopy", rec["gather"])
            print(f"[fac] N {N:8d} {name:9s} err {err:.2e} dship {dship:.1e} | gather {g['us']:7.1f} us {g['TBs']:4.2f} TB/s "
                  f"hold {g['hold_cyc_per_tile']:6.0f} wait {g['wait_cyc_per_tile']:6.0f} cyc/tile | nocopy {c['us']:7.1f} us "
                  f"{c['ns_per_tile']:6.0f} ns/tile hold {c['hold_cyc_per_tile']:6.0f}", flush=True)
        del kc, vc, pt
        torch.cuda.empty_cache()
    if not args.local:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

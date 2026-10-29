"""wgmma_bench.py: standalone H200 microbench of the MoE consumer's MMA feed (R-wgb-0, notes/study/wgmma_bench.md).

Challenges the R-xc-0 "RS-wgmma operand-feed floor" (64 wgmma x (~20 + 0.5 N) clk per 36 KB stage). 1 CTA per SM on every
SM (200 KB dynamic smem), per-CTA clock64 + globaltimer around the steady-state loop, a ~300 us gate (torch.cuda._sleep,
as the judge's GpuTimer) before every launch. All MMAs live (asm volatile, results to a sink behind a runtime-false flag).

  python3 -u kernels/lab/wgmma_bench.py gen OUT.cu                 write the CUDA source
  python3 -u kernels/lab/wgmma_bench.py sass CUBIN PTXAS_LOG        SASS opcode counts vs expected, spills
  python3 -u kernels/lab/wgmma_bench.py all [--out=box_out/wgb/run.json]   gen + nvcc + sass + run (GPU)

Kernels (bf16 in, fp32 accumulate, no-swizzle K-major core-matrix smem layout: LBO (K) 128 B, SBO (M/N) 256 B):
  chk_N       1 WG: RS, SS and SS-from-fragment-STS (the dequant producer's store map) vs torch, N = 8 / 64 / 256
  tput_{rs,ss}_N_CH_W   (a)/(b): NWG warpgroups each loop {fence; CH x m64nNk16 into one accumulator; commit; wait_group W}
              RS: A = 4 registers; SS: A = 2 KB smem tile (4 tiles cycled); B = N x 16 smem tile (4 cycled)
  ms_N_BM     (c): 16 warps, m16n8k16 mma.sync, one loop iteration = one 256-row x 256-K stage (32 N mma per SM);
              BM 0: B fragments in registers, BM 1: B via ldmatrix (x tile in core layout, conflict-free) per k16
  dq_M_N      (d): PW dequant WGs (cp.async staging of the NVFP4 code words + e4m3 scales, UNP-L2 deq4, bf16x2 scale
              multiply, 4 x st.shared.b32 per word into an 8-slot ring of 8 KB slabs = 256 rows x k16) -> mbarrier ->
              consumer WGs: M0 = SS (A = slab tiles, B = x; 2 WGs x 2 m64 tiles, N 8..64), M1 = transposed SS (A = x
              64 x k16 from smem, B = 128 weight rows per WG, n128, 2 WGs), M2 = transposed SS n256 (1 WG),
              M3 = transposed RS n256 (1 WG, x A fragments held in 64 registers). flags: 1 = producer skips dequant +
              stores (stream + handshake only), 2 = consumer skips the wgmma.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
TN = (8, 16, 32, 64, 128, 256)
CHS = (4, 8, 16)
MSN = (8, 16, 32, 64)
DQN = (8, 16, 32, 64)
CHKN = (8, 64, 256)
NSL = 8
DEP = 8
SMEM = 200 * 1024

HEAD = r"""
#include <stdint.h>
#define NSL %(NSL)d
#define DEP %(DEP)d
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t gdesc(uint32_t a) {   // no swizzle, K-major: LBO (K core stride) 128 B, SBO (8-row stride) 256 B
  return (uint64_t)((a & 0x3FFFFu) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(256u >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int W> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %%0;\n" :: "n"(W) : "memory"); }
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %%0, %%%%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t smid() { uint32_t r; asm volatile("mov.u32 %%0, %%%%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t c) { asm volatile("mbarrier.init.shared::cta.b64 [%%0], %%1;" :: "r"(a), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ uint32_t mbar_try(uint32_t a, uint32_t ph) {
  uint32_t r;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%%1], %%2;\nselp.u32 %%0, 1, 0, p;\n}\n" : "=r"(r) : "r"(a), "r"(ph) : "memory");
  return r;
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t ph, int* err, int& dead) {
  int n = 0;
  if (dead) return;   // after one timeout this thread's later waits fall through (the run is flagged, ends fast)
  while (!mbar_try(a, ph)) { if (++n > (1 << 22)) { atomicOr(err, 1); dead = 1; break; } }
}
__device__ __forceinline__ uint32_t ok32(uint32_t c) { uint32_t r; asm volatile("mov.b32 %%0, %%1;" : "=r"(r) : "r"(c)); return r; }
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // UNP L2 layout (tools/unp_patches.py L2_BODY)
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 10) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 13) & 0x81C081C0u;
  o[3] = ((w & 0x001B001Bu) * ok32(0x1084u)) & 0x81C081C0u;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.bf16x2 %%0, %%1, %%2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.b32 [%%0], %%1;" :: "r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void sts16(uint32_t a, uint16_t v) { asm volatile("st.shared.b16 [%%0], %%1;" :: "r"(a), "h"(v) : "memory"); }
__device__ __forceinline__ uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %%0, [%%1];" : "=r"(v) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint4 lds128(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.b32 {%%0, %%1, %%2, %%3}, [%%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.b32 {%%0, %%1}, [%%2];" : "=r"(v.x), "=r"(v.y) : "r"(a) : "memory"); return v; }
__device__ __forceinline__ void cpa16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%%0], [%%1], 16;" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ void cpa8(uint32_t s, const void* g) { asm volatile("cp.async.ca.shared.global [%%0], [%%1], 8;" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int W> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %%0;" :: "n"(W) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%%0, %%1, %%2, %%3}, [%%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm2(uint32_t a, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%%0, %%1}, [%%2];" : "=r"(r0), "=r"(r1) : "r"(a) : "memory");
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%%0, %%1, %%2, %%3}, {%%4, %%5, %%6, %%7}, {%%8, %%9}, {%%0, %%1, %%2, %%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
"""


def wg_fns(n):
    k = n // 2
    acc = ", ".join(f"%{i}" for i in range(k))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(k))
    rs = ("__device__ __forceinline__ void rs%d(float* d, const uint32_t* a, uint64_t db, uint32_t sc) {\n" % n
          + '  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %' + str(k + 5) + ', 0;\\n"\n'
          + '    "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%%%d, %%%d, %%%d, %%%d}, %%%d, p, 1, 1, 0;\\n}\\n"\n'
          % (n, acc, k, k + 1, k + 2, k + 3, k + 4)
          + '    : %s : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sc) : "memory");\n}\n' % outs)
    ss = ("__device__ __forceinline__ void ss%d(float* d, uint64_t da, uint64_t db, uint32_t sc) {\n" % n
          + '  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %' + str(k + 2) + ', 0;\\n"\n'
          + '    "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n"\n'
          % (n, acc, k, k + 1)
          + '    : %s : "l"(da), "l"(db), "r"(sc) : "memory");\n}\n' % outs)
    return rs + ss


def dispatch():
    s = "template <int N> __device__ __forceinline__ void rsN(float* d, const uint32_t* a, uint64_t db, uint32_t sc) {\n"
    s += "".join(f"  {'if' if i == 0 else 'else if'} constexpr (N == {n}) rs{n}(d, a, db, sc);\n" for i, n in enumerate(TN)) + "}\n"
    s += "template <int N> __device__ __forceinline__ void ssN(float* d, uint64_t da, uint64_t db, uint32_t sc) {\n"
    s += "".join(f"  {'if' if i == 0 else 'else if'} constexpr (N == {n}) ss{n}(d, da, db, sc);\n" for i, n in enumerate(TN)) + "}\n"
    return s


BODY = r"""
template <int N>
__device__ __forceinline__ void chk_body(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ D) {
  __shared__ __align__(1024) uint8_t sm[4096 + 256 * 32];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const uint32_t base = sa(sm);
  for (int i = tid; i < 64 * 16; i += 128) { const int r = i >> 4, k = i & 15; sts16(base + ((r >> 3) * 2 + (k >> 3)) * 128 + (r & 7) * 16 + (k & 7) * 2, A[i]); }
  for (int i = tid; i < N * 16; i += 128) { const int r = i >> 4, k = i & 15; sts16(base + 4096 + ((r >> 3) * 2 + (k >> 3)) * 128 + (r & 7) * 16 + (k & 7) * 2, B[i]); }
  uint32_t a[4];
  const int r0 = 16 * w + g;
  a[0] = (uint32_t)A[r0 * 16 + 2 * t] | ((uint32_t)A[r0 * 16 + 2 * t + 1] << 16);
  a[1] = (uint32_t)A[(r0 + 8) * 16 + 2 * t] | ((uint32_t)A[(r0 + 8) * 16 + 2 * t + 1] << 16);
  a[2] = (uint32_t)A[r0 * 16 + 2 * t + 8] | ((uint32_t)A[r0 * 16 + 2 * t + 9] << 16);
  a[3] = (uint32_t)A[(r0 + 8) * 16 + 2 * t + 8] | ((uint32_t)A[(r0 + 8) * 16 + 2 * t + 9] << 16);
  const uint32_t so = base + 2048 + 512 * w + 16 * g + 4 * t;   // the dq producer's fragment -> core-matrix store map
  sts32(so, a[0]); sts32(so + 256, a[1]); sts32(so + 128, a[2]); sts32(so + 384, a[3]);
  fence_async();
  __syncthreads();
  float d[N / 2];
#pragma unroll
  for (int v = 0; v < 3; ++v) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i) d[i] = 0.f;
    wg_fence();
    if (v == 0) rsN<N>(d, a, gdesc(base + 4096), 0u);
    else ssN<N>(d, gdesc(base + (v == 1 ? 0 : 2048)), gdesc(base + 4096), 0u);
    wg_commit();
    wg_wait<0>();
    float* o = D + v * 64 * N;
#pragma unroll
    for (int j = 0; j < N / 8; ++j) {
      o[r0 * N + 8 * j + 2 * t] = d[4 * j]; o[r0 * N + 8 * j + 2 * t + 1] = d[4 * j + 1];
      o[(r0 + 8) * N + 8 * j + 2 * t] = d[4 * j + 2]; o[(r0 + 8) * N + 8 * j + 2 * t + 1] = d[4 * j + 3];
    }
  }
}

template <int MODE, int N, int CH, int WAIT>
__device__ __forceinline__ void tput_body(const uint32_t* __restrict__ src, float* __restrict__ sink, long long* __restrict__ out, int iters, int flag) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const int tid = threadIdx.x;
  const int nfill = (8192 + 4 * N * 32) / 4;
  for (int i = tid; i < nfill; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = __ldg(src + (i & 4095));
  fence_async();
  uint32_t a[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) a[j] = __ldg(src + ((tid * 4 + j) & 4095));
  float d[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) d[i] = 0.f;
  const uint32_t ab = sa(sm), bb = ab + 8192u;
  const uint64_t da0 = gdesc(ab), db0 = gdesc(bb);
  __syncthreads();
  const long long c0 = clock64(), g0 = gtime();
  int it = 0;
#pragma unroll 1
  do {   // do-while: no zero-trip phi (register moves of in-flight accumulators serialize wait_group 1 chains, C7514)
    wg_fence();
#pragma unroll
    for (int c = 0; c < CH; ++c) {
      const uint64_t db = db0 + (uint64_t)(((c & 3) * N * 32) >> 4);
      if constexpr (MODE == 0) rsN<N>(d, a, db, 1u);
      else ssN<N>(d, da0 + (uint64_t)(((c & 3) * 2048) >> 4), db, 1u);
    }
    wg_commit();
    wg_wait<WAIT>();
  } while (++it < iters);
  wg_wait<0>();
  __syncthreads();
  if (tid == 0) { out[blockIdx.x * 4] = clock64() - c0; out[blockIdx.x * 4 + 1] = gtime() - g0; out[blockIdx.x * 4 + 2] = smid(); }
  if (flag == 12345) { float s = 0.f;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) s += d[i];
    sink[blockIdx.x * blockDim.x + tid] = s; }
}

template <int N, int BM>
__device__ __forceinline__ void ms_body(const uint32_t* __restrict__ src, float* __restrict__ sink, long long* __restrict__ out, int iters, int flag) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31;
  for (int i = tid; i < 16 * N * 8; i += blockDim.x) reinterpret_cast<uint32_t*>(sm)[i] = __ldg(src + (i & 4095));
  uint32_t a[4][4], b[N / 8][2];
  float acc[N / 8][4];
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int j = 0; j < 4; ++j) a[q][j] = __ldg(src + ((tid * 16 + q * 4 + j) & 4095));
#pragma unroll
  for (int j = 0; j < N / 8; ++j) { b[j][0] = __ldg(src + ((tid * 3 + j * 7) & 4095)); b[j][1] = __ldg(src + ((tid * 5 + j * 11) & 4095));
    acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
  const uint32_t zo = (uint32_t)flag >> 30;   // 0, opaque: ldmatrix addresses vary per iteration in the compiler's view
  const uint32_t xb = sa(sm) + (uint32_t)lane * 16u;
  __syncthreads();
  const long long c0 = clock64(), g0 = gtime();
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    const uint32_t xo = xb + (uint32_t)it * zo * 16u;
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
      if constexpr (BM == 1) {
        if constexpr (N == 8) ldsm2(xo + kk * N * 32, b[0][0], b[0][1]);
        else {
#pragma unroll
          for (int jj = 0; jj < N / 16; ++jj) ldsm4(xo + kk * N * 32 + jj * 512, b[2 * jj][0], b[2 * jj][1], b[2 * jj + 1][0], b[2 * jj + 1][1]);
        }
      }
#pragma unroll
      for (int j = 0; j < N / 8; ++j) mma16816(acc[j], a[kk & 3], b[j]);
    }
  }
  __syncthreads();
  if (tid == 0) { out[blockIdx.x * 4] = clock64() - c0; out[blockIdx.x * 4 + 1] = gtime() - g0; out[blockIdx.x * 4 + 2] = smid(); }
  if (flag == 12345) { float s = 0.f;
#pragma unroll
    for (int j = 0; j < N / 8; ++j) s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    sink[blockIdx.x * blockDim.x + tid] = s; }
}

template <int MODE, int N>
__device__ __forceinline__ void dq_body(const uint4* __restrict__ codes, const uint2* __restrict__ scl, float* __restrict__ sink,
                                        long long* __restrict__ out, int* __restrict__ err, int nstage, int pw, int flags) {
  constexpr int CW = (MODE >= 2) ? 1 : 2;
  constexpr int DN = (MODE == 0) ? N : ((MODE == 1) ? 128 : 256);
  extern __shared__ __align__(1024) uint8_t sm[];
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, w4 = (tid >> 5) & 3;
  const uint32_t base = sa(sm);
  const uint32_t ring = base, xr = base + NSL * 8192, stg = xr + 32768, bars = stg + 2 * 128 * DEP * 24;
  for (int i = tid; i < 8192; i += blockDim.x) sts32(xr + 4 * i, (reinterpret_cast<const uint32_t*>(codes)[i] & 0x803F803Fu) | 0x3C003C00u);
  if (tid == 0) {
    for (int i = 0; i < NSL; ++i) { mbar_init(bars + 8 * i, 4); mbar_init(bars + 8 * (NSL + i), 4 * CW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  fence_async();
  __syncthreads();
  const long long c0 = clock64(), g0 = gtime();
  const int nslab = nstage * 16;
  uint32_t live = 0;
  int dead = 0;
  if (wg < CW) {
    float d0[DN / 2], d1[(MODE == 0) ? DN / 2 : 1];
#pragma unroll
    for (int i = 0; i < DN / 2; ++i) d0[i] = 0.f;
#pragma unroll
    for (int i = 0; i < ((MODE == 0) ? DN / 2 : 1); ++i) d1[i] = 0.f;
    uint32_t xa[(MODE == 3) ? 16 : 1][4];
    if constexpr (MODE == 3) {
#pragma unroll
      for (int k = 0; k < 16; ++k)
#pragma unroll
        for (int j = 0; j < 4; ++j) xa[k][j] = lds32(xr + k * 2048 + (tid & 127) * 16 + 4 * j);
    }
    const bool on = !(flags & 2);
    const uint32_t zo = (uint32_t)flags >> 30;   // 0, opaque: keeps the 16 x 2-3 descriptors from being hoisted (spills)
#pragma unroll 1
    for (int s0 = 0; s0 < nslab; s0 += 16) {
      const uint32_t vo = (uint32_t)s0 * zo, rv = ring + vo, xv = xr + vo;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const int s = s0 + k;
        const uint32_t slot = (uint32_t)(k & (NSL - 1));
        mbar_wait(bars + 8 * slot, (uint32_t)(s >> 3) & 1u, err, dead);
        if (on) {
          wg_fence();
          const uint32_t sb = rv + slot * 8192u;
          if constexpr (MODE == 0) {
            const uint64_t db = gdesc(xv + k * N * 32);
            ssN<N>(d0, gdesc(sb + (2 * wg) * 2048), db, 1u);
            ssN<N>(d1, gdesc(sb + (2 * wg + 1) * 2048), db, 1u);
          } else if constexpr (MODE == 1) {
            ssN<128>(d0, gdesc(xv + k * 2048), gdesc(sb + wg * 4096), 1u);
          } else if constexpr (MODE == 2) {
            ssN<256>(d0, gdesc(xv + k * 2048), gdesc(sb), 1u);
          } else {
            rsN<256>(d0, xa[k], gdesc(sb), 1u);
          }
          wg_commit();
          wg_wait<1>();
          if (s > 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NSL + ((k + NSL - 1) & (NSL - 1)))); }
        } else {
          __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NSL + slot));
        }
      }
    }
    wg_wait<0>();
    if (flags & 0x100) { float s = 0.f;
#pragma unroll
      for (int i = 0; i < DN / 2; ++i) s += d0[i];
#pragma unroll
      for (int i = 0; i < ((MODE == 0) ? DN / 2 : 1); ++i) s += d1[i];
      sink[blockIdx.x * blockDim.x + tid] = s; }
  } else {
    const int pt = tid - CW * 128, pwi = pt >> 7, t128 = pt & 127, g = lane >> 2, t = lane & 3;
    const uint32_t my = stg + (uint32_t)pt * (DEP * 24);
    const uint32_t sto = 512u * w4 + 16u * g + 4u * t;
    const int nj = (nslab - pwi + pw - 1) / pw;
#pragma unroll
    for (int d = 0; d < DEP; ++d) {
      if (d < nj) { const int s = (pwi + pw * d) & 63; cpa16(my + d * 16, codes + s * 128 + t128); cpa8(my + DEP * 16 + d * 8, scl + s * 128 + t128); }
      cpa_commit();
    }
    const bool on = !(flags & 1);
#pragma unroll 1
    for (int j = 0; j < nj; ++j) {
      const int s = pwi + pw * j, d = j & (DEP - 1);
      cpa_wait<DEP - 1>();
      const uint4 cw = lds128(my + d * 16);
      const uint2 sw = lds64(my + DEP * 16 + d * 8);
      const uint32_t slot = (uint32_t)(s & (NSL - 1));
      mbar_wait(bars + 8 * (NSL + slot), ((uint32_t)(s >> 3) & 1u) ^ 1u, err, dead);
      if (on) {
        const uint32_t sb = ring + slot * 8192u + sto;
        const uint32_t ws[4] = {cw.x, cw.y, cw.z, cw.w};
        const uint32_t sc2[2] = {sw.x, sw.y};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          uint32_t o[4];
          deq4(ws[i], o);
          const uint32_t sv = __byte_perm(sc2[i >> 1], 0u, (i & 1) ? 0x4342u : 0x4140u);
          const uint32_t s2 = ((sv & 0x007F007Fu) << 4) + 0x7B007B00u;   // 2 x e4m3 (normal) -> bf16 x 2^126
          const uint32_t slo = __byte_perm(s2, 0u, 0x1010u), shi = __byte_perm(s2, 0u, 0x3232u);
          sts32(sb + i * 2048, bmul2(o[0], slo));
          sts32(sb + i * 2048 + 256, bmul2(o[1], shi));
          sts32(sb + i * 2048 + 128, bmul2(o[2], slo));
          sts32(sb + i * 2048 + 384, bmul2(o[3], shi));
        }
        fence_async();
      } else {
        live ^= cw.x ^ cw.y ^ cw.z ^ cw.w ^ sw.x ^ sw.y;
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bars + 8 * slot);
      const int j2 = j + DEP;
      if (j2 < nj) { const int s2 = (pwi + pw * j2) & 63; cpa16(my + d * 16, codes + s2 * 128 + t128); cpa8(my + DEP * 16 + d * 8, scl + s2 * 128 + t128); }
      cpa_commit();
    }
    if (flags & 0x100) sink[blockIdx.x * blockDim.x + tid] = (float)live;
  }
  __syncthreads();
  if (tid == 0) { out[blockIdx.x * 4] = clock64() - c0; out[blockIdx.x * 4 + 1] = gtime() - g0; out[blockIdx.x * 4 + 2] = smid(); }
}
"""

TARGS = "(const uint32_t* __restrict__ src, float* __restrict__ sink, long long* __restrict__ out, int iters, int flag)"
DARGS = ("(const uint4* __restrict__ codes, const uint2* __restrict__ scl, float* __restrict__ sink, long long* __restrict__ out, "
         "int* __restrict__ err, int nstage, int pw, int flags)")


def dq_list():
    return [(0, n) for n in DQN] + [(1, 0), (2, 0), (3, 0)]


def gen():
    s = HEAD % {"NSL": NSL, "DEP": DEP} + "".join(wg_fns(n) for n in TN) + dispatch() + BODY
    for n in CHKN:
        s += (f'extern "C" __global__ void __launch_bounds__(128, 1) chk_{n}(const uint16_t* A, const uint16_t* B, float* D) '
              f'{{ chk_body<{n}>(A, B, D); }}\n')
    for m, mn in ((0, "rs"), (1, "ss")):
        for n in TN:
            lb = 256 if n == 256 else 512
            for ch in CHS:
                for w in (0, 1):
                    s += (f'extern "C" __global__ void __launch_bounds__({lb}, 1) tput_{mn}_{n}_{ch}_{w}{TARGS} '
                          f'{{ tput_body<{m}, {n}, {ch}, {w}>(src, sink, out, iters, flag); }}\n')
    for n in MSN:
        for bm in (0, 1):
            s += (f'extern "C" __global__ void __launch_bounds__(512, 1) ms_{n}_{bm}{TARGS} '
                  f'{{ ms_body<{n}, {bm}>(src, sink, out, iters, flag); }}\n')
    for m, n in dq_list():
        lb = {0: 512, 1: 512, 2: 384, 3: 256}[m]
        s += (f'extern "C" __global__ void __launch_bounds__({lb}, 1) dq_{m}_{n}{DARGS} '
              f'{{ dq_body<{m}, {n if m == 0 else 8}>(codes, scl, sink, out, err, nstage, pw, flags); }}\n')
    return s


def expected():
    e = {}
    for n in CHKN:
        e[f"chk_{n}"] = {"HGMMA": 3}
    for mn in ("rs", "ss"):
        for n in TN:
            for ch in CHS:
                for w in (0, 1):
                    e[f"tput_{mn}_{n}_{ch}_{w}"] = {"HGMMA": ch}
    for n in MSN:
        for bm in (0, 1):
            e[f"ms_{n}_{bm}"] = {"HMMA": 16 * n // 8, "LDSM": (16 * max(1, n // 16)) if bm else 0}
    for m, n in dq_list():
        e[f"dq_{m}_{n}"] = {"HGMMA": 32 if m == 0 else 16}
    return e


def sass(cubin, ptxlog=None):
    import shutil
    obj = shutil.which("cuobjdump")
    if not obj:
        try:
            from torch.utils.cpp_extension import CUDA_HOME
            obj = str(pathlib.Path(CUDA_HOME) / "bin/cuobjdump")
        except Exception:
            obj = ""
        if not pathlib.Path(obj).exists():
            obj = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
    txt = subprocess.run([obj, "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    for ln in txt.splitlines():
        m = re.search(r"Function : (\S+)", ln)
        if m:
            cur = m.group(1)
            funcs[cur] = {}
            continue
        m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)", ln)
        if cur and m:
            op = m.group(1)
            funcs[cur][op] = funcs[cur].get(op, 0) + 1
    exp = expected()
    bad = []
    for f, want in exp.items():
        got = funcs.get(f)
        if got is None:
            bad.append(f"{f}: missing")
            continue
        for op, v in want.items():
            if got.get(op, 0) != v:
                bad.append(f"{f}: {op} {got.get(op, 0)} != {v}")
        if got.get("STL", 0) or got.get("LDL", 0):
            bad.append(f"{f}: STL {got.get('STL', 0)} LDL {got.get('LDL', 0)}")
    spills = []
    if ptxlog:
        fn = None
        for ln in pathlib.Path(ptxlog).read_text().splitlines():
            m = re.search(r"Compiling entry function '(\w+)'", ln)
            if m:
                fn = m.group(1)
            m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
            if m and fn and (int(m.group(1)) or int(m.group(2))):
                spills.append(f"{fn} spill {m.group(1)}/{m.group(2)}")
    hg = sum(f.get("HGMMA", 0) for f in funcs.values())
    print(f"[sass] {len(funcs)} kernels, {len(exp)} expected, HGMMA total {hg}, mismatches {len(bad)}, spills {len(spills)}")
    for b in (bad + spills)[:12]:
        print("[sass]  ", b)
    return {"funcs": funcs, "bad": bad, "spills": spills}


class Drv:
    def __init__(self, cubin):
        import ctypes
        import torch
        self.ct, self.torch = ctypes, torch
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.mod = ctypes.c_void_p()
        self.chk(self.cu.cuModuleLoad(ctypes.byref(self.mod), str(cubin).encode()))
        self.fns = {}

    def chk(self, r):
        if r != 0:
            raise RuntimeError(f"CUDA driver error {r}")

    def fn(self, name, smem):
        ct = self.ct
        if name not in self.fns:
            f = ct.c_void_p()
            self.chk(self.cu.cuModuleGetFunction(ct.byref(f), self.mod, name.encode()))
            if smem:
                self.chk(self.cu.cuFuncSetAttribute(f, 8, smem))   # MAX_DYNAMIC_SHARED_SIZE_BYTES
            self.fns[name] = f
        return self.fns[name]

    def launch(self, name, grid, block, smem, args):
        ct = self.ct
        vals = []
        for a in args:
            if hasattr(a, "data_ptr"):
                vals.append(ct.c_void_p(a.data_ptr()))
            else:
                vals.append(ct.c_int(int(a)))
        arr = (ct.c_void_p * len(vals))(*[ct.cast(ct.byref(v), ct.c_void_p) for v in vals])
        st = ct.c_void_p(self.torch.cuda.current_stream().cuda_stream)
        self.chk(self.cu.cuLaunchKernel(self.fn(name, smem), grid, 1, 1, block, 1, 1, smem, st, arr, None))


class Gate:
    def __init__(self, torch, us=300.0):
        self.torch, self.us, self.cyc = torch, us, int(us * 1980)
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            torch.cuda._sleep(self.cyc)
            b.record()
            b.synchronize()
            self.cyc = int(self.cyc * us / max(a.elapsed_time(b) * 1e3, 1.0))

    def __call__(self):
        self.torch.cuda._sleep(self.cyc)


def run(cubin, out_path):
    import statistics as st
    import torch
    D = Drv(cubin)
    G = torch.cuda.get_device_properties(0).multi_processor_count
    gate = Gate(torch)
    torch.manual_seed(0)
    res = {"G": G, "gate_cyc": gate.cyc, "chk": [], "tput": [], "ms": [], "dq": []}
    # correctness
    for n in CHKN:
        A = (torch.randn(64, 16, device="cuda") * 0.5).to(torch.bfloat16)
        B = (torch.randn(n, 16, device="cuda") * 0.5).to(torch.bfloat16)
        Dm = torch.zeros(3, 64, n, device="cuda")
        D.launch(f"chk_{n}", 1, 128, 0, [A.view(torch.int16), B.view(torch.int16), Dm])
        torch.cuda.synchronize()
        ref = A.float() @ B.float().t()
        e = [float((Dm[v] - ref).abs().max() / ref.abs().max()) for v in range(3)]
        res["chk"].append({"N": n, "rel_rs": e[0], "rel_ss": e[1], "rel_ssf": e[2],
                           "ss_eq_rs": bool(torch.equal(Dm[0], Dm[1])), "ssf_eq_ss": bool(torch.equal(Dm[1], Dm[2]))})
    ok = all(max(c["rel_rs"], c["rel_ss"], c["rel_ssf"]) < 1e-2 for c in res["chk"])
    print("[chk]", "PASS" if ok else "FAIL", " ".join(f"N{c['N']} {c['rel_rs']:.1e}/{c['rel_ss']:.1e}/{c['rel_ssf']:.1e}"
                                                      f"{'' if c['ss_eq_rs'] and c['ssf_eq_ss'] else ' (neq)'}" for c in res["chk"]))
    src = (torch.randn(8192, device="cuda") * 0.25).to(torch.bfloat16).view(torch.int32).contiguous()
    sink = torch.zeros(G * 512, device="cuda")
    outb = torch.zeros(G * 4, dtype=torch.int64, device="cuda")
    err = torch.zeros(1, dtype=torch.int32, device="cuda")

    def timed(name, block, args, reps=3):
        rows = []
        for _ in range(reps):
            gate()
            D.launch(name, G, block, SMEM, args(outb))
            torch.cuda.synchronize()
            o = outb.view(G, 4).cpu()
            rows.append((st.median(o[:, 0].tolist()), st.median(o[:, 1].tolist()), max(o[:, 1].tolist())))
        clk = st.median(r[0] for r in rows)
        ns = st.median(r[1] for r in rows)
        return clk, ns, st.median(r[2] for r in rows)

    t0 = time.time()
    for mn in ("rs", "ss"):
        for n in TN:
            for nwg in ((1, 2) if n == 256 else (1, 2, 4)):
                for ch in CHS:
                    for w in (0, 1):
                        tot = 8192
                        it = tot // ch
                        clk, ns, nsmax = timed(f"tput_{mn}_{n}_{ch}_{w}", 128 * nwg, lambda ob, it=it: [src, sink, ob, it, 0])
                        res["tput"].append({"mode": mn, "N": n, "nwg": nwg, "ch": ch, "wait": w, "clk_per_instr": clk / (tot * nwg),
                                            "mhz": clk / ns * 1e3, "ns_max": nsmax})
    it = 256
    for n in MSN:
        for bm in (0, 1):
            clk, ns, nsmax = timed(f"ms_{n}_{bm}", 512, lambda ob: [src, sink, ob, it, 0])
            res["ms"].append({"N": n, "bm": bm, "clk_per_stage": clk / it, "us_per_stage": ns / it / 1e3, "mhz": clk / ns * 1e3})
    codes = torch.randint(-2**31, 2**31 - 1, (64 * 128 * 4,), dtype=torch.int32, device="cuda")
    sb = (torch.randint(1, 9, (64 * 128 * 8,), device="cuda") << 3) | torch.randint(0, 8, (64 * 128 * 8,), device="cuda")
    scl = sb.to(torch.uint8).view(torch.int32).contiguous()
    nst = 256
    for m, n in dq_list():
        pws = (1,) if m == 3 else (1, 2)
        cw = 1 if m >= 2 else 2
        for pw in pws:
            for fl in (0, 1, 2):
                err.zero_()
                clk, ns, nsmax = timed(f"dq_{m}_{n}", 128 * (cw + pw), lambda ob, pw=pw, fl=fl: [codes, scl, sink, ob, err, nst, pw, fl])
                res["dq"].append({"mode": m, "N": n, "pw": pw, "flags": fl, "clk_per_stage": clk / nst, "us_per_stage": ns / nst / 1e3,
                                  "us_per_stage_max": nsmax / nst / 1e3, "mhz": clk / ns * 1e3, "err": int(err.item())})
    res["secs"] = time.time() - t0
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text(json.dumps(res, indent=1))
    summarize(res)
    return res


def summarize(res):
    tp = res["tput"]

    def best(mode, n, nwg):
        r = [x for x in tp if x["mode"] == mode and x["N"] == n and x["nwg"] == nwg]
        return min(r, key=lambda x: x["clk_per_instr"]) if r else None

    print(f"[tput] clk per m64nNk16 per SM (best ch/wait; ch16 w1 in brackets), MHz ~{st_med([x['mhz'] for x in tp]):.0f}")
    for n in TN:
        cells = []
        for mode in ("rs", "ss"):
            for nwg in (1, 2, 4):
                b = best(mode, n, nwg)
                if b:
                    c16 = [x for x in tp if x["mode"] == mode and x["N"] == n and x["nwg"] == nwg and x["ch"] == 16 and x["wait"] == 1][0]
                    cells.append(f"{mode}{nwg} {b['clk_per_instr']:.1f}[{c16['clk_per_instr']:.1f}]")
        print(f"[tput] N{n:<3} " + "  ".join(cells))
    for r in res["ms"]:
        if r["bm"] == 0:
            r1 = [x for x in res["ms"] if x["N"] == r["N"] and x["bm"] == 1][0]
            print(f"[ms]   N{r['N']:<3} stage clk B-reg {r['clk_per_stage']:.0f} B-ldsm {r1['clk_per_stage']:.0f} "
                  f"({r1['us_per_stage']:.3f} us, {r1['mhz']:.0f} MHz)")
    for m, n in dq_list():
        rows = [x for x in res["dq"] if x["mode"] == m and x["N"] == n]
        cells = [f"pw{x['pw']} f{x['flags']} {x['clk_per_stage']:.0f}/{x['us_per_stage']:.3f}{'E' if x['err'] else ''}" for x in rows]
        print(f"[dq]   M{m} N{n:<3} clk/us per stage: " + "  ".join(cells))


def st_med(v):
    import statistics
    return statistics.median(v) if v else 0.0


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "gen":
        pathlib.Path(argv[2]).write_text(gen())
        return
    if cmd == "sass":
        sass(argv[2], argv[3] if len(argv) > 3 else None)
        return
    if cmd == "summary":
        summarize(json.loads(pathlib.Path(argv[2]).read_text()))
        return
    out = "box_out/wgb/run.json"
    for a in argv[2:]:
        if a.startswith("--out="):
            out = a[6:]
    wd = ROOT / "box_out/wgb"
    wd.mkdir(parents=True, exist_ok=True)
    cu, cub, log = wd / "wgb.cu", wd / "wgb.cubin", wd / "ptxas.log"
    cu.write_text(gen())
    t0 = time.time()
    import shutil
    nvcc = shutil.which("nvcc")
    if not nvcc:
        from torch.utils.cpp_extension import CUDA_HOME
        nvcc = str(pathlib.Path(CUDA_HOME) / "bin/nvcc")
    r = subprocess.run([nvcc, "-cubin", "-arch=sm_90a", "-O3", "-std=c++17", "-Xptxas=-v", "-o", str(cub), str(cu)],
                       capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    print(f"[nvcc] rc {r.returncode} {time.time() - t0:.0f} s")
    if r.returncode:
        print((r.stdout + r.stderr)[-1500:])
        sys.exit(1)
    sass(cub, log)
    run(cub, ROOT / out)


if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3
""":"
# This same permitted file is also the LF-only Linux compile/SASS/CPU gate script.
set -eu
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
exec python -B "$0" --compile-only --out="${0%.py}_gates.json" "$@"
":"""
"""M16XD: the XR4 falsifier ladder (box_out/reset/pwr_opus.md s2 row 1 and s3 "Top-1 XR4 ladder").

CPU:         python -B kernels/lab/m16xd_micro.py --dry-run
Linux gate:  bash tools/local/cudaenv.sh bash kernels/lab/m16xd_micro.py   (CPU + compile + SASS gates; never launches CUDA)
Pod:         python3 kernels/lab/m16xd_micro.py --out=box_out/m16xd/run.json   (<= 350 s worker timeout)

Standalone copy of the GPU-validated kernels/lab/m16xc_micro.py harness: 132 CTAs x 48 units of 36,864-B NVFP4 stages,
NS = 3 TMA ring, gated CUDA-graph timer (256-MiB L2 flush + 300 us gate), 3 rotating banks, per-arm FP32 correctness with
input reset and NaN-poison replays, per-CTA clock64 / globaltimer, zero-initialised GRID-sized err buffer. The rs / rs2 /
gd arms and N = 32 are dropped. Arms per N in {8, 16}; every arm is its own template instance (no run-time arm branch in
the unit loop) and requests the SAME dynamic smem and carveout as ms:
  null, ms, ms_twin  controls: ms = m16xc mode 3 verbatim; ms_twin = the same source block, a second instance
  xreg               bound, DIAGNOSTIC (output invalid): one B quad per n8 loaded per unit, every per-k x LDS deleted
  acc4               ms + 4 accumulator sets rotated by q, (d0 + d1) + (d2 + d3) before the epilogue
  xw128              ms + x in per-lane fragment order: one LDS.128 = {b0, b1, b2, b3} of (k32 step q, n8 tile)
  xr4                warp wp owns K quarter [256 wp, +256) of all 4 row tiles; one fragment-order LDS.128 feeds 4 row
                     tiles; 4 x N/8 accumulator chains; per-warp 64 x N FP32 partial (in the x prefix behind the
                     stripe, so SMEM is unchanged); helper sums (p0 + p1) + (p2 + p3) into the ms output layout
The consumer programs are generated from plan(); the CPU emulator (emulate_unit) executes the SAME plan ops with the
same address formulas and mma.m16n8k16 fragment semantics, and must reproduce the FP64 reference for every exact arm.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
M16XC = ROOT / "kernels/lab/m16xc_micro.py"
UNIT = 36864
GRID = 132
NAMES = ("null", "ms", "ms_twin", "xreg", "acc4", "xw128", "xr4")
ARM_ID = {name: mode for mode, name in enumerate(NAMES)}
ARMS = tuple((arm, n) for n in (8, 16) for arm in NAMES)
CONTROLS = ("null", "ms", "ms_twin")
NEW_ARMS = ("xreg", "acc4", "xw128", "xr4")
DIAGNOSTIC = ("xreg",)          # output invalid by construction: excluded from correctness, used only by Gate 0
FRAG_X = ("xw128", "xr4")       # x stored in per-lane fragment order
ACC4 = ("d", "e1", "e2", "e3")  # accumulator set of k32 step q is ACC4[q % 4]
# UNP L2, exactly K2A31_BITPOS3, base lines 1648-1649 (not K2A_BITPOS).
BITPOS = ((6, 7, 8, 15), (22, 23, 24, 31), (28, 29, 30, 5), (12, 13, 14, 21),
          (25, 26, 27, 2), (9, 10, 11, 18), (4, 0, 1, 3), (20, 16, 17, 19))
# Pre-registered decisions, box_out/reset/pwr_opus.md section 3 (us per unit, in-kernel MHz).
PRE = {"source": "box_out/reset/pwr_opus.md section 3 (Top-1 XR4 ladder)",
       "validity": {"twin_us_le": 0.03, "twin_mhz_le": 15},
       "gate0_kill_if_xreg_ge_ms_minus": {"8": 0.04, "16": 0.06},
       "xr4_kill_abs_gt_both": {"8": 1.20, "16": 1.30},
       "xr4_kill_rel_n16_gt_min_acc4_xw128_minus": 0.02,
       "xr4_go_le_both": {"8": 1.15, "16": 1.24},
       "clock_est_dmhz_xr4_minus_ms": {"8": [15, 30], "16": [30, 50]},
       "clock_n16_low_lt": 10, "clock_high_ge": 60}

# Verbatim VML functions from k3_diet_rq128.py:7739-7749,8575-8601.
# These source blocks have no interior comments; none have been stripped.
DEQUANT = r"""
__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {
uint32_t r;
asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));
return r;
}
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {
o[0] = w & 0x81C081C0u;
o[1] = __funnelshift_l(w, w, 10) & 0x81C081C0u;
o[2] = __funnelshift_l(w, w, 13) & 0x81C081C0u;
o[3] = ((w & 0x001B001Bu) * unp_ok32(0x1084u)) & 0x81C081C0u;
}
#ifndef VM_TRICKS_DQ
#define VM_TRICKS_DQ 1
#endif
__device__ __forceinline__ void deq4_dq(uint32_t w, uint32_t* o, uint32_t k64, uint32_t k4) {
(void)k64; (void)k4;
deq4(w, o);
}
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
__nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
__nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void dec_half2(uint32_t w0, uint32_t w8, uint32_t sw2, uint32_t par, uint32_t* hh, uint32_t k16,
uint32_t k64, uint32_t k4) {
#if VM_TRICKS_DQ
deq4_dq(w0, hh, k64, k4); deq4_dq(w8, hh + 4, k64, k4);
#else
deq4(w0, hh); deq4(w8, hh + 4);
#endif
const uint32_t sb = 2u * par;
const uint32_t sgp = __byte_perm(sw2, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16 + 0x75807580u;
#pragma unroll
for (int j = 0; j < 4; ++j) { hh[j] = bmul2l(hh[j], sgp); hh[4 + j] = bmul2h(hh[4 + j], sgp); }
}
"""

CUDA_HEAD = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
constexpr int UB = 36864, XS = 65536, RSZ = 3 * UB;
template<int N> constexpr int EP = XS + RSZ;
template<int N> constexpr int BO = EP<N> + 2 * 64 * N * 4;
template<int N> constexpr int SMEM = BO<N> + 4096;
// XR4 per-warp FP32 partials live in the x prefix right behind the resident stripe: every arm keeps SMEM<N>.
template<int N> constexpr int XP = N * 2048;
static_assert(XP<8> + 8 * 64 * 8 * 4 <= XS && XP<16> + 8 * 64 * 16 * 4 <= XS, "XR4 partials exceed the x prefix");
constexpr int M_XR4 = @M_XR4@;
template<int N, int MODE> constexpr int DN = MODE == M_XR4 ? 2 * N : N / 2;
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t gt() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool ready(uint32_t b, int p) {
  uint32_t r;
  asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 q, [%1], %2; selp.u32 %0, 1, 0, q; }"
               : "=r"(r) : "r"(b), "r"(p) : "memory");
  return r;
}
__device__ __forceinline__ void wait(uint32_t b, int p, int* err) {
  if (ready(b, p)) return;
  const uint64_t start = gt();
  unsigned polls = 0;
  while (!ready(b, p)) {
    if ((++polls & 63u) == 0u) {
      // Failure is diagnostic only: keep every arrival and the terminal CTA barrier.
      // No thread exits early; the host rejects ALL data/timings if any err is set.
      if (atomicAdd(err, 0) || gt() - start > 100000000ull) {
        atomicExch(err, 1); break;
      }
    }
  }
}
__device__ __forceinline__ void bulk(uint32_t dst, const uint8_t* src, uint32_t b, uint64_t policy) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               :: "r"(b), "r"(UB) : "memory");
  // One contiguous, losslessly repacked 64x1024 unit; no tensor-map descriptor.
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(UB), "r"(b), "l"(policy) : "memory");
}
__device__ __forceinline__ void ms(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template<int N, int Q> __device__ __forceinline__ void ms_pair(float* d, const uint32_t* h, const uint16_t* x, int lane) {
  // One load per B fragment per (warp row-tile, k16, n8); never assume full-stripe register residency.
  const int g = lane >> 2, t = lane & 3;
  #pragma unroll
  for (int j = 0; j < N / 8; ++j) {
    const uint32_t* b = reinterpret_cast<const uint32_t*>(x + (2 * Q) * N * 16 + j * 128 + g * 8 + t * 2);
    const uint32_t b0 = b[0], b1 = b[32];
    const uint32_t b2 = b[N * 8], b3 = b[N * 8 + 32];
    ms(d + 4 * j, h[0], h[4], h[1], h[5], b0, b1);
    ms(d + 4 * j, h[2], h[6], h[3], h[7], b2, b3);
  }
}
__device__ __forceinline__ void mma2(float* d, const uint32_t* h, uint4 b) {
  // {b.x, b.y} = B fragment of k16 step 2Q, {b.z, b.w} = step 2Q + 1: exactly ms_pair's b0..b3.
  ms(d, h[0], h[4], h[1], h[5], b.x, b.y);
  ms(d, h[2], h[6], h[3], h[7], b.z, b.w);
}
template<int N, int Q> __device__ __forceinline__ void msw_pair(float* d, const uint32_t* h, const uint16_t* x, int lane) {
  // Fragment-order x: one LDS.128 per (warp row-tile, k32 step, n8) instead of four LDS.32; same bytes.
  #pragma unroll
  for (int j = 0; j < N / 8; ++j) mma2(d + 4 * j, h, reinterpret_cast<const uint4*>(x)[(Q * (N / 8) + j) * 32 + lane]);
}
template<int N> __device__ __forceinline__ uint4 xquad(const uint16_t* x, int j, int lane) {
  // xreg bound only: ms_pair's Q = 0 B quad of n8 tile j, loaded once per unit and reused by every k32 step.
  const int g = lane >> 2, t = lane & 3;
  const uint32_t* b = reinterpret_cast<const uint32_t*>(x + j * 128 + g * 8 + t * 2);
  return make_uint4(b[0], b[32], b[N * 8], b[N * 8 + 32]);
}
template<int N, int MODE> __device__ __forceinline__ void consume(float* d, const uint8_t* st, uint16_t* x, int wp, int lane) {
  // Generated straight-line specializations, no run-time arm/fragment branch.
  @CONSUME@
}
// m16xc kept the start stamps in registers live through every role region; at the 128-register cap ptxas spilled
// them (outside the loop). Here they live at a constant smem address behind the 10 barriers: no live registers.
__device__ __forceinline__ void put_stamp(uint32_t a) {
  const uint64_t t0=gt(), c0=clock64();
  asm volatile("st.shared.v2.u64 [%0], {%1, %2};" :: "r"(a), "l"(c0), "l"(t0) : "memory");
}
__device__ __forceinline__ void save_clock(int* ids, uint32_t a) {
  uint64_t c1=clock64(), t1=gt(), c0, t0;
  asm volatile("ld.shared.v2.u64 {%0, %1}, [%2];" : "=l"(c0), "=l"(t0) : "r"(a) : "memory");
  uint64_t* times=reinterpret_cast<uint64_t*>(ids + gridDim.x);
  times[2*blockIdx.x]=c1-c0; times[2*blockIdx.x+1]=t1-t0;
}

template<int N, int MODE> __global__ __launch_bounds__(384, 1)
void m16xd(const uint8_t* W, const uint16_t* X, float* O, int* smids, int* errors, int units) {
  extern __shared__ __align__(128) uint8_t s[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = sa(s + BO<N>);
  int* err = errors + blockIdx.x;
  // All threads initialize the resident stripe before register ownership changes.
  for (int i = tid * 8; i < N * 1024; i += 384 * 8)
    *reinterpret_cast<uint4*>(s + i * 2) = *reinterpret_cast<const uint4*>(X + i);
  if (tid == 0) {
    *err = 0;
    for (int j = 0; j < 3; ++j) { init(bars + j * 8, 1); init(bars + (3 + j) * 8, 4); }
    for (int j = 0; j < 2; ++j) { init(bars + (6 + j) * 8, 4); init(bars + (8 + j) * 8, 1); }
  }
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (tid == 0) put_stamp(sa(s + BO<N> + 128));
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
    if (tid == 256) {
      uint64_t policy;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
      for (int u = 0; u < units; ++u) {
        const int slot = u % 3;
        if (u >= 3) wait(bars + (3 + slot) * 8, ((u / 3) - 1) & 1, err);
        bulk(sa(s + XS + slot * UB), W + (size_t(blockIdx.x) * units + u) * UB, bars + slot * 8, policy);
      }
      // Drain EVERY outstanding slot, including the final three (no abandoned TMA).
      for (int slot = 0; slot < 3; ++slot) {
        const int last = units - 1 - ((units - 1 - slot) % 3);
        if (last >= 0) wait(bars + (3 + slot) * 8, (last / 3) & 1, err);
      }
    }
    if constexpr (MODE != 0) {
      if (warp == 9 || warp == 10) {
        const int wg = warp - 9;
        if constexpr (MODE == M_XR4) {
          // Four K-quarter partials in fragment order (float2 per (tile, jp, lane)); two adjacent lanes form one
          // contiguous float4 of a row, so the sum (p0 + p1) + (p2 + p3) lands in the ms output layout.
          const float4* pp = reinterpret_cast<const float4*>(s + XP<N> + wg * 4 * 64 * N * 4);
          for (int u = wg; u < units; u += 2) {
            wait(bars + (6 + wg) * 8, (u / 2) & 1, err);
            float4* dst = reinterpret_cast<float4*>(O + (size_t(blockIdx.x) * units + u) * 64 * N);
            // Not unrolled: an unrolled helper loop spills 4-28 B of loop-invariant addresses at 32 (or 40) regs,
            // depending on unrelated consumer code. Helpers are off the critical path (consumers wait on u - 2).
            #pragma unroll 1
            for (int j = lane; j < 64 * N / 4; j += 32) {
              const float4 a = pp[j], b = pp[j + 16 * N], c = pp[j + 32 * N], e = pp[j + 48 * N];
              const int l = (2 * j) & 31, r = (2 * j) >> 5, tile = r / (N / 4), jp = r % (N / 4);
              const int row = tile * 16 + (l >> 2) + (jp & 1) * 8, col = (jp >> 1) * 8 + (l & 3) * 2;
              float4 v;
              v.x = ((a.x + b.x) + (c.x + e.x)) * 256.f; v.y = ((a.y + b.y) + (c.y + e.y)) * 256.f;
              v.z = ((a.z + b.z) + (c.z + e.z)) * 256.f; v.w = ((a.w + b.w) + (c.w + e.w)) * 256.f;
              dst[(row * N + col) >> 2] = v;
            }
            __syncwarp();
            if (lane == 0) arrive(bars + (8 + wg) * 8);
          }
        } else {
          const float4* ep = reinterpret_cast<const float4*>(s + EP<N> + wg * 64 * N * 4);
          for (int u = wg; u < units; u += 2) {
            wait(bars + (6 + wg) * 8, (u / 2) & 1, err);
            float4* dst = reinterpret_cast<float4*>(O + (size_t(blockIdx.x) * units + u) * 64 * N);
            // Not unrolled: an unrolled helper loop spills 4-28 B of loop-invariant addresses at 32 (or 40) regs,
            // depending on unrelated consumer code. Helpers are off the critical path (consumers wait on u - 2).
            #pragma unroll 1
            for (int j = lane; j < 64 * N / 4; j += 32) {
              float4 v = ep[j];
              v.x *= 256.f; v.y *= 256.f; v.z *= 256.f; v.w *= 256.f;
              dst[j] = v;
            }
            __syncwarp();
            if (lane == 0) arrive(bars + (8 + wg) * 8);
          }
        }
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 128;" ::: "memory");
    const int wg = warp / 4, wp = warp & 3;
    for (int u = wg; u < units; u += 2) {
      const int slot = u % 3;
      // Odd ring depth alternates the owner of each slot. Do not test a future
      // full phase while the previous owner's TMA can still be in flight: its
      // old parity would look ready. The previous empty phase certifies retirement.
      if (u >= 3) wait(bars + (3 + slot) * 8, ((u / 3) - 1) & 1, err);
      wait(bars + slot * 8, (u / 3) & 1, err);
      if constexpr (MODE != 0) {
        float d[DN<N, MODE>] = {};
        consume<N, MODE>(d, s + XS + slot * UB, reinterpret_cast<uint16_t*>(s), wp, lane);
        __syncwarp();
        if (lane == 0) arrive(bars + (3 + slot) * 8);
        if (u >= 2) wait(bars + (8 + wg) * 8, ((u / 2) - 1) & 1, err);
        if constexpr (MODE == M_XR4) {
          // Fragment-order float2 (tile, jp, lane): one STS.64 per pair, 256 contiguous bytes per warp store.
          float2* pp = reinterpret_cast<float2*>(s + XP<N> + (wg * 4 + wp) * 64 * N * 4) + lane;
          #pragma unroll
          for (int i = 0; i < N; ++i) pp[i * 32] = make_float2(d[2 * i], d[2 * i + 1]);
        } else {
          float* ep = reinterpret_cast<float*>(s + EP<N> + wg * 64 * N * 4);
          #pragma unroll
          for (int j = 0; j < N / 2; ++j) {
            const int r = wp * 16 + (lane >> 2) + ((j & 3) >> 1) * 8;
            const int c = (j / 4) * 8 + (lane & 3) * 2 + (j & 1);
            ep[r * N + c] = d[j];
          }
        }
        __syncwarp();
        if (lane == 0) arrive(bars + (6 + wg) * 8);
      } else {
        // Null is ONLY ring + waits, plus identical startup/allocation and terminal bookkeeping.
        __syncwarp();
        if (lane == 0) arrive(bars + (3 + slot) * 8);
      }
    }
  }
  // Includes all 384 threads, even producer warp's unused lanes and helper warp 11.
  __syncthreads();
  if (tid == 0) {
    uint32_t sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); smids[blockIdx.x] = sm; save_clock(smids,sa(s + BO<N> + 128));
    if (!*err) for (int j = 0; j < 10; ++j)
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bars + j * 8) : "memory");
  }
}
__global__ void empty_kernel() {}
inline void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<int N, int MODE> std::vector<int> config(bool query) {
  ck(cudaFuncSetAttribute(m16xd<N, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM<N>));
  ck(cudaFuncSetAttribute(m16xd<N, MODE>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  if (!query) return {};
  cudaFuncAttributes a; ck(cudaFuncGetAttributes(&a, m16xd<N, MODE>));
  int occ; ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, m16xd<N, MODE>, 384, SMEM<N>));
  return {a.numRegs, int(a.localSizeBytes), int(a.sharedSizeBytes), SMEM<N>, occ};
}
std::vector<int> info(int mode, int n) { @INFO@ throw std::runtime_error("unsupported arm/N"); }
void launch(int mode, int n, int units, int grid, uint64_t w, uint64_t x, uint64_t o, uint64_t sm, uint64_t err, uint64_t stream) {
  if (units < 4 || (units & 1) || grid < 2 || (grid & 1)) throw std::runtime_error("invalid grid/units");
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  @DISPATCH@
  throw std::runtime_error("unsupported arm/N");
}
void launch_empty(uint64_t stream) { empty_kernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(); ck(cudaGetLastError()); }
"""

CPP = r"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
std::vector<int> info(int, int);
void launch(int, int, int, int, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
void launch_empty(uint64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("info", &info); m.def("launch", &launch); m.def("empty", &launch_empty);
}
"""


def plan(arm, n):
    """Straight-line per-warp consumer program for one unit (one tuple per CUDA statement).

    render() turns it into the CUDA specialization; run_plan() executes the very same ops on the CPU.
    """
    ops = []
    if arm == "null":
        return ops
    if arm == "xr4":
        # Warp wp: k32 steps q = 8 wp + qq of ALL four 16-row tiles (code kb = 4 wp + kk, scale sg = 2 wp + ss).
        ops += [("xr4_bases",), ("decl_h",)]
        for ss in range(2):
            ops.append(("open",))
            ops += [("ld_scale_r", f"s{tile}", tile * 256 + ss * 32) for tile in range(4)]
            for sub in range(2):
                kk = 2 * ss + sub
                ops.append(("open",))
                ops += [("ld_code_r", f"c{tile}", tile * 512 + kk * 32) for tile in range(4)]
                for hf in range(2):
                    qq = 2 * kk + hf
                    ops.append(("open",))
                    ops += [("ld_xr", f"b{j}", (qq * (n // 8) + j) * 32) for j in range(n // 8)]
                    for tile in range(4):
                        ops.append(("dec", f"c{tile}", hf, f"s{tile}", sub))
                        ops += [("mma2", "d", tile * (n // 2) + 4 * j, f"b{j}") for j in range(n // 8)]
                    ops.append(("close",))
                ops.append(("close",))
            ops.append(("close",))
        return ops
    # ms geometry: warp wp = row tile wp over all K; exactly 16 LDS.128 code + 8 LDS.32 scale loads per lane/unit.
    # The same scale word serves four k32 half-groups; no extra scale bytes.
    ops.append(("decl_h",))
    if arm == "acc4":
        ops.append(("decl_acc4",))
    if arm == "xreg":
        ops += [("ld_xquad", f"bq{j}", j) for j in range(n // 8)]
    for sg in range(8):
        ops += [("open",), ("ld_scale_ms", "scale", sg)]
        for sub in range(2):
            kb = 2 * sg + sub
            ops += [("open",), ("ld_code_ms", "c", kb)]
            for hf in range(2):
                q = 2 * kb + hf
                ops.append(("dec", "c", hf, "scale", sub))
                if arm in ("ms", "ms_twin"):
                    ops.append(("ms_pair", "d", q))
                elif arm == "acc4":
                    ops.append(("ms_pair", ACC4[q % 4], q))
                elif arm == "xw128":
                    ops.append(("msw_pair", "d", q))
                elif arm == "xreg":
                    ops += [("mma2", "d", 4 * j, f"bq{j}") for j in range(n // 8)]
                else:
                    raise ValueError(arm)
            ops.append(("close",))
        ops.append(("close",))
    if arm == "acc4":
        ops.append(("sum_acc4",))
    return ops


def render(op):
    kind, *a = op
    if kind == "open": return "{"
    if kind == "close": return "}"
    if kind == "decl_h": return "uint32_t h[8];"
    if kind == "decl_acc4": return "float e1[N / 2] = {}, e2[N / 2] = {}, e3[N / 2] = {};"
    if kind == "sum_acc4": return "#pragma unroll\nfor (int i = 0; i < N / 2; ++i) d[i] = (d[i] + e1[i]) + (e2[i] + e3[i]);"
    if kind == "xr4_bases":
        return ("const uint4* cw = reinterpret_cast<const uint4*>(st + wp * 2048) + lane;\n"
                "const uint32_t* sw = reinterpret_cast<const uint32_t*>(st + 32768 + wp * 256) + lane;\n"
                "const uint4* xb = reinterpret_cast<const uint4*>(x) + wp * (32 * N) + lane;")
    if kind == "ld_scale_ms":
        return f"const uint32_t {a[0]} = reinterpret_cast<const uint32_t*>(st + 32768 + wp * 1024)[{a[1]} * 32 + lane];"
    if kind == "ld_code_ms":
        return f"const uint4 {a[0]} = reinterpret_cast<const uint4*>(st + wp * 8192)[{a[1]} * 32 + lane];"
    if kind == "ld_scale_r": return f"const uint32_t {a[0]} = sw[{a[1]}];"
    if kind == "ld_code_r": return f"const uint4 {a[0]} = cw[{a[1]}];"
    if kind == "ld_xr": return f"const uint4 {a[0]} = xb[{a[1]}];"
    if kind == "ld_xquad": return f"const uint4 {a[0]} = xquad<N>(x, {a[1]}, lane);"
    if kind == "dec":
        c, hf, s, sub = a
        return f"dec_half2({c}.{('x', 'y')[hf]}, {c}.{('z', 'w')[hf]}, {s}, {sub}u, h, 16u, 64u, 4u);"
    if kind == "ms_pair": return f"ms_pair<N, {a[1]}>({a[0]}, h, x, lane);"
    if kind == "msw_pair": return f"msw_pair<N, {a[1]}>({a[0]}, h, x, lane);"
    if kind == "mma2": return f"mma2({a[0]} + {a[1]}, h, {a[2]});"
    raise ValueError(op)


def plan_text(arm, n):
    return "\n".join(render(op) for op in plan(arm, n))


def generate_cuda():
    """Single binary; MODE and N resolve at instantiation, never in the stage loop.

    Identical generated bodies share one block (ms and ms_twin: same source, two instances).
    """
    blocks = {}
    for arm, n in ARMS:
        if arm != "null":
            blocks.setdefault(plan_text(arm, n), []).append((ARM_ID[arm], n))
    pieces = []
    for i, (text, keys) in enumerate(blocks.items()):
        cond = " || ".join(f"(MODE == {m} && N == {k})" for m, k in keys)
        pieces.append(f"{'if' if i == 0 else 'else if'} constexpr ({cond}) {{\n{text}\n}}")
    pieces.append('else { static_assert(MODE == 0, "arm without a consumer specialization"); }')
    info, dispatch = [], []
    for arm, n in ARMS:
        mode = ARM_ID[arm]
        cond = f"if (mode == {mode} && n == {n})"
        info.append(f"{cond} return config<{n}, {mode}>(true);")
        dispatch.append(f"{cond} {{ m16xd<{n}, {mode}><<<grid,384,SMEM<{n}>,st>>>((const uint8_t*)w,(const uint16_t*)x,(float*)o,(int*)sm,(int*)err,units); ck(cudaGetLastError()); return; }}")
    src = CUDA_HEAD.replace("@CONSUME@", "\n".join(pieces)).replace("@M_XR4@", str(ARM_ID["xr4"]))
    src = src.replace("@INFO@", "\n".join(info)).replace("@DISPATCH@", "\n".join(dispatch))
    anchor = "template<int N, int Q> __device__"
    return src.replace(anchor, DEQUANT + "\n" + anchor, 1)


def expected_static(arm, n):
    """Static SASS counts the straight-line consumer must produce, derived from plan()."""
    hmma = lds32 = lds128 = 0
    for op in plan(arm, n):
        kind = op[0]
        if kind in ("ld_scale_ms", "ld_scale_r"): lds32 += 1
        elif kind in ("ld_code_ms", "ld_code_r", "ld_xr"): lds128 += 1
        elif kind == "ms_pair": lds32 += 4 * (n // 8); hmma += 2 * (n // 8)
        elif kind == "msw_pair": lds128 += n // 8; hmma += 2 * (n // 8)
        elif kind == "ld_xquad": lds32 += 4
        elif kind == "mma2": hmma += 2
    return {"HMMA": hmma, "LDS32": lds32, "LDS128_consumer_min": lds128}


def reencode(sf):
    b = sf.astype(np.int32)
    sign, ex, m = b >> 7, (b >> 3) & 15, b & 7
    if np.any((ex == 15) & (m == 7)):
        raise ValueError("NaN block scale in NVFP4 weights")
    zero = (ex == 0) & (m == 0)
    sub = (ex == 0) & (m != 0)
    le = np.array([0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int32)
    lf = np.array([0, 0, 0, 4, 0, 2, 4, 6], dtype=np.int32)
    bp = np.where(sub, (le[m] << 3) | lf[m], ((ex + 3) << 3) | m)
    return np.where(zero, 24, bp).astype(np.uint8), (sign == 1) & ~zero, zero


def pack_tiles(raw, sf):
    """[U,64,512] original code bytes + [U,64,64] E4M3 -> [U,36864]."""
    u = raw.shape[0]
    assert raw.shape == (u, 64, 512) and sf.shape == (u, 64, 64)
    bp, neg, zero = reencode(sf)
    c = np.stack((raw & 15, raw >> 4), axis=-1).reshape(u, 64, 64, 16)
    c = np.where(neg[..., None], c ^ 8, c)
    c = np.where(zero[..., None], np.uint8(0), c).astype(np.uint8)
    # Exact base KPERM3, independently for each contiguous 64-row unit.
    c = c.reshape(u, 4, 2, 8, 16, 4, 2, 8).transpose(0, 1, 4, 3, 5, 2, 6, 7)
    c = c.reshape(u, 4, 16, 32, 4, 8).astype(np.uint32)
    w = np.zeros(c.shape[:-1], np.uint32)
    for i, pos in enumerate(BITPOS):
        for bit, dst in enumerate(pos):
            w |= ((c[..., i] >> bit) & 1) << dst
    s = bp.reshape(u, 4, 2, 8, 8, 2, 4).transpose(0, 1, 4, 3, 6, 5, 2).copy()
    return np.concatenate((w.reshape(u, -1).view(np.uint8), s.reshape(u, -1)), axis=1)


def unit_coords(units, grid):
    """Cohorts of four CTAs: a CTA keeps its physical K stripe for its entire stream."""
    ids = np.arange(units * grid, dtype=np.int64)
    cta, local = ids // units, ids % units
    row_tile = (cta // 4) * units + local
    return row_tile // 128, row_tile % 128, cta % 4


def helper_dst(arm, n):
    """Helper float4 j (lanes stride 32) -> destination float4 inside one 64 x N output tile (row-major)."""
    j = np.arange(64 * n // 4, dtype=np.int64)
    if arm != "xr4":
        return j
    # CUDA: l = (2j) & 31, r = (2j) >> 5, tile = r / (N/4), jp = r % (N/4); row/col of the lane pair (l, l + 1).
    l, r = (2 * j) & 31, (2 * j) >> 5
    tile, jp = r // (n // 4), r % (n // 4)
    row = tile * 16 + (l >> 2) + (jp & 1) * 8
    col = (jp >> 1) * 8 + (l & 3) * 2
    assert np.all(col % 4 == 0) and np.all(row < 64) and np.all(col + 4 <= n)
    return (row * n + col) // 4


def address_bounds(arm, n, units, grid, capacities=None):
    """Replay every global access site in bytes; no tensors/GPU needed.

    Intervals enumerate every bulk transfer, X uint4 and helper float4. Repeated
    error polls share the same per-CTA int32 address. Capacities may come from the
    actual tensors, so a short allocation cannot pass by checking only a formula.
    """
    assert (arm, n) in ARMS and units >= 4 and units % 2 == 0 and grid >= 2 and grid % 2 == 0
    assert max(units + 2, grid) < 2**31, "C++ int loop/launch overflow"
    sizes = {"W": grid * units * UNIT, "X": n * 1024 * 2,
             "O": grid * units * 64 * 16 * 4, "sm": grid * 20, "err": grid * 4}
    if capacities is not None:
        assert set(capacities) == set(sizes)
        sizes = capacities
    assert max(sizes.values()) < 2**63, "size_t/int64 address overflow"
    accesses = {}

    def check(name, starts, width, alignment):
        starts = np.asarray(starts, dtype=np.int64)
        assert starts.size and int(starts.min()) >= 0
        end = int(starts.max()) + width
        assert end <= sizes[name], (arm, n, name, end, sizes[name])
        assert np.all(starts % alignment == 0), (name, "misaligned offset")
        accesses[name] = {"accesses": int(starts.size), "first_byte": int(starts.min()),
                          "end_byte_exclusive": end, "tensor_bytes": int(sizes[name])}

    cta = np.arange(grid, dtype=np.int64)
    u = np.arange(units, dtype=np.int64)
    tiles = cta[:, None] * units + u[None, :]  # CUDA casts blockIdx.x to size_t FIRST.
    check("W", tiles * UNIT, UNIT, 16)
    xi = np.concatenate([np.arange(tid * 8, n * 1024, 384 * 8, dtype=np.int64)
                         for tid in range(384)])
    assert np.array_equal(np.sort(xi), np.arange(0, n * 1024, 8))
    check("X", np.broadcast_to(xi * 2, (grid, xi.size)), 16, 16)
    if arm != "null":
        # Helpers 288..319 (even units) and 320..351 (odd units).
        js = np.concatenate([np.arange(lane, 64 * n // 4, 32, dtype=np.int64)
                             for lane in range(32)])
        assert np.array_equal(np.sort(js), np.arange(64 * n // 4))
        dst = helper_dst(arm, n)
        assert np.array_equal(np.sort(dst), np.arange(64 * n // 4)), "helper destination is not a permutation"
        seen = []
        for wg in (0, 1):
            owned = tiles[:, wg::2].reshape(-1)
            starts = owned[:, None] * (64 * n * 4) + dst[js][None, :] * 16
            seen.append(starts.reshape(-1))
        check("O", np.concatenate(seen), 16, 16)
    else:
        accesses["O"] = {"accesses": 0, "tensor_bytes": int(sizes["O"])}
    check("sm", cta * 4, 4, 4)  # thread 0 terminal store
    assert grid * 4 + grid * 16 <= sizes["sm"], (arm, n, "sm", "clock records exceed allocation")
    accesses["sm"]["clock_records_end_byte_exclusive"] = grid * 20
    accesses["sm"]["end_byte_exclusive"] = grid * 20
    check("err", cta * 4, 4, 4)  # thread 0 reset/read + all waiters' atomic polls
    # prepare_bank's actual raw tensor slices, including cohort/expert transitions.
    es, rows, ks = unit_coords(units, grid)
    experts = ((grid + 3) // 4 * units + 127) // 128
    assert np.all((es >= 0) & (es < experts))
    assert np.all((rows * 64 >= 0) & ((rows + 1) * 64 <= 8192))
    assert np.all((ks * 512 >= 0) & ((ks + 1) * 512 <= 2048))
    assert np.all((ks * 64 >= 0) & ((ks + 1) * 64 <= 256))
    assert len(set(zip(es.tolist(), rows.tolist(), ks.tolist()))) == grid * units
    return {"arm": arm, "n": n, "grid": grid, "units": units, "status": "PASS",
            "experts": experts, "accesses": accesses}


def tensor_bounds(arm, n, units, grid, w, x, out, sm, err):
    """Run the same CPU arithmetic against the actual launch allocations."""
    tensors = {"W": w, "X": x, "O": out, "sm": sm, "err": err}
    for name, tensor in tensors.items():
        assert tensor.is_contiguous(), (name, "non-contiguous launch tensor")
        assert tensor.data_ptr() % (4 if name in ("sm", "err") else 16) == 0
    return address_bounds(arm, n, units, grid,
                          {k: t.numel() * t.element_size() for k, t in tensors.items()})


def prepare_bank(weights, units, grid):
    """Only weight keys: identical names/shapes to the real problem, no layer state."""
    w, sf = weights["w13"], weights["w13_sf"]
    assert w.dtype == np.uint8 and sf.dtype == np.uint8  # sf is the raw E4M3 view
    assert w.shape[1:] == (8192, 2048) and sf.shape == (w.shape[0], 8192, 256)
    count = units * grid
    es, rows, ks = unit_coords(units, grid)
    assert int(es[-1]) < w.shape[0]
    out = np.empty((count, UNIT), np.uint8)
    # Bound temporary CPU RAM; this is lossless preprocessing, never timed.
    for lo in range(0, count, 64):
        hi = min(lo + 64, count)
        rr = rows[lo:hi, None] * 64 + np.arange(64)
        cc = ks[lo:hi, None] * 512 + np.arange(512)
        ss = ks[lo:hi, None] * 64 + np.arange(64)
        raw = w[es[lo:hi, None, None], rr[:, :, None], cc[:, None, :]]
        scale = sf[es[lo:hi, None, None], rr[:, :, None], ss[:, None, :]]
        out[lo:hi] = pack_tiles(raw, scale)
    return out


def make_weights(experts, seed):
    r = np.random.default_rng(seed)
    w = r.integers(0, 256, (experts, 8192, 2048), dtype=np.uint8)
    # Varied finite positive scales for timing. Correctness also covers signed/subnormals.
    s = r.integers(24, 81, (experts, 8192, 256), dtype=np.uint8)
    return {"w13": w, "w13_sf": s}


def kperm(k):
    return (k // 64) * 64 + ((k % 8) // 2) * 16 + ((k % 64) // 8) * 2 + k % 2


def pack_x(x):
    """BF16 bits [N,1024] -> 64 no-swizzle K-major descriptor tiles (the ms / acc4 / xreg layout)."""
    n = x.shape[0]
    assert n in (8, 16) and x.shape == (n, 1024) and x.dtype == np.uint16
    z = x[:, kperm(np.arange(1024))]
    return z.reshape(n // 8, 8, 64, 2, 8).transpose(2, 0, 3, 1, 4).copy().reshape(-1)


def xf_perm(n):
    """Fragment-order u32 word ((q * N/8 + j) * 32 + lane) * 4 + w  <-  pack_x u32 word of ms_pair's b_w."""
    q, j, lane, w = np.meshgrid(np.arange(32), np.arange(n // 8), np.arange(32), np.arange(4), indexing="ij")
    g, t = lane >> 2, lane & 3
    base = ((2 * q) * n * 16 + j * 128 + g * 8 + t * 2) // 2
    return (base + np.array([0, 32, n * 8, n * 8 + 32])[w]).reshape(-1)


def pack_xf(x):
    """BF16 bits [N,1024] -> per-lane fragment order (xw128 / xr4): one uint4 = {b0, b1, b2, b3} of (q, n8, lane)."""
    words = pack_x(x).view(np.uint32)
    return words[xf_perm(x.shape[0])].view(np.uint16).copy()


def x_layout(arm):
    return "frag" if arm in FRAG_X else "ms"


def pack_layout(layout, x):
    return pack_xf(x) if layout == "frag" else pack_x(x)


def fp8_numpy(b):
    b = b.astype(np.int32)
    e, m = (b >> 3) & 15, b & 7
    mag = np.where(e == 0, m * 2.0 ** -9, np.ldexp(1.0 + m / 8.0, e - 7))
    return np.where(b & 128, -mag, mag).astype(np.float32)


def unpack_tiles(packed):
    """Independent bit extraction and layout inverse for the CPU dry run."""
    u = len(packed)
    w = packed[:, :32768].copy().view(np.uint32).reshape(u, 4, 16, 32, 4)
    c = np.zeros(w.shape + (8,), np.uint8)
    for i, pos in enumerate(BITPOS):
        for bit, src in enumerate(pos):
            c[..., i] |= (((w >> src) & 1) << bit).astype(np.uint8)
    c = c.reshape(u, 4, 16, 8, 4, 2, 2, 8).transpose(0, 1, 5, 3, 2, 4, 6, 7).reshape(u, 64, 1024)
    s = packed[:, 32768:].reshape(u, 4, 8, 8, 4, 2, 2).transpose(0, 1, 6, 3, 2, 5, 4).reshape(u, 64, 64)
    # Reencoded scale bits are exactly BF16((bp << 4) + 0x7580); restore the deq bias independently.
    bits = ((s.astype(np.uint32) << 4) + 0x7580) << 16
    scales = bits.view(np.float32).astype(np.float64)
    cb = ((c.astype(np.uint32) & 7) << 6) | ((c.astype(np.uint32) & 8) << 12)
    vals = (cb << 16).view(np.float32).astype(np.float64)
    return (vals.reshape(u, 64, 64, 16) * scales[..., None] * 256).reshape(u, 64, 1024)


def emulate_cuda_a(packed):
    """Actual CUDA code/scales addresses, deq4 outputs, and RS/mma.sync A lane map."""
    w=packed[:32768].copy().view(np.uint32).reshape(4,16,32,4)
    s=packed[32768:].copy().view(np.uint32).reshape(4,8,32)
    a=np.full((64,1024),np.nan,dtype=np.float64)
    lanes=np.arange(32)
    for wp in range(4):
        for q in range(32):
            kb,hf=divmod(q,2)
            sw=s[wp,kb//2]
            for rh in range(2):
                cw=w[wp,kb,:,2*rh+hf]
                rot=lambda n:(cw<<n)|(cw>>(32-n))
                dq=(cw&0x81C081C0,rot(10)&0x81C081C0,rot(13)&0x81C081C0,
                    ((cw&0x001B001B)*np.uint32(0x1084))&0x81C081C0)
                scale=((((sw>>(8*(2*(kb%2)+rh)))&255)*16+0x7580)<<16).view(np.float32)
                for j,pair in enumerate(dq):
                    step=2*q+j//2
                    for bit in (0,1):
                        val=((((pair>>(16*bit))&65535)<<16).view(np.float32).astype(np.float64)*scale*256)
                        row=16*wp+lanes//4+8*rh
                        k=16*step+2*(lanes%4)+8*(j%2)+bit
                        assert np.isnan(a[row,k]).all()
                        a[row,k]=val
    assert np.isfinite(a).all()
    return a


# ---------------------------------------------------------------- CPU emulator of the generated consumers
LANES = np.arange(32)
LG, LT = LANES >> 2, LANES & 3
DQ_MASK = np.uint32(0x81C081C0)


def bf16_pairs(words):
    """u32 per lane -> (32, 2) float64: low / high BF16 halves (k even / k odd of a fragment register)."""
    w = np.asarray(words, dtype=np.uint32)
    lo = (w << np.uint32(16)).view(np.float32)
    hi = (w & np.uint32(0xFFFF0000)).view(np.float32)
    return np.stack((lo, hi), axis=-1).astype(np.float64)


def bf16_bits(f):
    """FP32 -> BF16 bits, round to nearest even (test inputs only)."""
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def emu_dec(w0, w8, sw, par):
    """dec_half2 on 32 lanes -> h as (32, 8, 2) float64 (h[0..3]: row g, scale byte 2 par; h[4..7]: row g + 8, byte 2 par + 1).

    The BF16 HMUL2 products are exact (E2M1 x reencoded E4M3; proven bit-exactly by the exhaustive deq4 gate).
    """
    out = np.empty((32, 8, 2))
    for base, w, byte in ((0, w0, 2 * par), (4, w8, 2 * par + 1)):
        w = np.asarray(w, dtype=np.uint32)
        rot = lambda k: (w << np.uint32(k)) | (w >> np.uint32(32 - k))
        words = (w & DQ_MASK, rot(10) & DQ_MASK, rot(13) & DQ_MASK,
                 ((w & np.uint32(0x001B001B)) * np.uint32(0x1084)) & DQ_MASK)
        sbits = (((np.asarray(sw, np.uint32) >> np.uint32(8 * byte)) & np.uint32(255)) * np.uint32(16)
                 + np.uint32(0x7580)) << np.uint32(16)
        scale = sbits.astype(np.uint32).view(np.float32).astype(np.float64)
        for j, word in enumerate(words):
            out[:, base + j] = bf16_pairs(word) * scale[:, None]
    return out


def emu_mma(D, a, b):
    """mma.sync.m16n8k16.row.col on 32 lanes: D (32, 4) in place; a = 4 x (32, 2); b = 2 x (32, 2)."""
    A = np.full((16, 16), np.nan)
    B = np.full((16, 8), np.nan)
    C = np.full((16, 8), np.nan)
    for e in (0, 1):
        A[LG, 2 * LT + e] = a[0][:, e]; A[LG + 8, 2 * LT + e] = a[1][:, e]
        A[LG, 2 * LT + 8 + e] = a[2][:, e]; A[LG + 8, 2 * LT + 8 + e] = a[3][:, e]
        B[2 * LT + e, LG] = b[0][:, e]; B[2 * LT + 8 + e, LG] = b[1][:, e]
        C[LG, 2 * LT + e] = D[:, e]; C[LG + 8, 2 * LT + e] = D[:, 2 + e]
    assert not (np.isnan(A).any() or np.isnan(B).any() or np.isnan(C).any()), "fragment hole"
    C = C + A @ B
    for e in (0, 1):
        D[:, e] = C[LG, 2 * LT + e]; D[:, 2 + e] = C[LG + 8, 2 * LT + e]


def emu_mma2(D, h, words):
    """CUDA mma2 / one j of ms_pair: words = (b0, b1, b2, b3) per lane."""
    b = [bf16_pairs(w) for w in words]
    emu_mma(D, (h[:, 0], h[:, 4], h[:, 1], h[:, 5]), (b[0], b[1]))
    emu_mma(D, (h[:, 2], h[:, 6], h[:, 3], h[:, 7]), (b[2], b[3]))


def run_plan(arm, n, wp, w32, xw):
    """Execute plan(arm, n) for warp wp of one unit; returns the accumulators (32, DN) and the per-lane load log."""
    w4, x4 = w32.reshape(-1, 4), xw.reshape(-1, 4)
    regs = {"d": np.zeros((32, 2 * n if arm == "xr4" else n // 2))}
    log = {"code": [], "scale": [], "x32": [], "x128": []}
    for op in plan(arm, n):
        kind, *a = op
        if kind in ("open", "close", "decl_h", "xr4_bases"):
            continue
        if kind == "decl_acc4":
            regs.update({e: np.zeros((32, n // 2)) for e in ACC4[1:]})
        elif kind == "sum_acc4":
            regs["d"] = (regs["d"] + regs["e1"]) + (regs["e2"] + regs["e3"])
        elif kind in ("ld_scale_ms", "ld_scale_r"):
            # (st + 32768 + wp * 1024)[sg * 32 + lane]  |  sw = (st + 32768 + wp * 256) + lane; sw[off]
            idx = ((32768 + wp * 1024) // 4 + a[1] * 32 + LANES if kind == "ld_scale_ms"
                   else (32768 + wp * 256) // 4 + LANES + a[1])
            regs[a[0]] = w32[idx]; log["scale"].append(idx)
        elif kind in ("ld_code_ms", "ld_code_r"):
            # (uint4*)(st + wp * 8192)[kb * 32 + lane]  |  cw = (uint4*)(st + wp * 2048) + lane; cw[off]
            idx = ((wp * 8192) // 16 + a[1] * 32 + LANES if kind == "ld_code_ms"
                   else (wp * 2048) // 16 + LANES + a[1])
            regs[a[0]] = w4[idx]; log["code"].append(idx)
        elif kind == "dec":
            c, hf, s, sub = a
            regs["h"] = emu_dec(regs[c][:, hf], regs[c][:, 2 + hf], regs[s], sub)
        elif kind == "ms_pair":
            acc, q = a
            for j in range(n // 8):
                b = ((2 * q) * n * 16 + j * 128 + LG * 8 + LT * 2) // 2
                idx = [b, b + 32, b + n * 8, b + n * 8 + 32]
                log["x32"] += idx
                emu_mma2(regs[acc][:, 4 * j:4 * j + 4], regs["h"], [xw[i] for i in idx])
        elif kind == "msw_pair":
            acc, q = a
            for j in range(n // 8):
                idx = (q * (n // 8) + j) * 32 + LANES
                log["x128"].append(idx)
                emu_mma2(regs[acc][:, 4 * j:4 * j + 4], regs["h"], list(x4[idx].T))
        elif kind == "ld_xquad":
            b = (a[1] * 128 + LG * 8 + LT * 2) // 2
            idx = [b, b + 32, b + n * 8, b + n * 8 + 32]
            log["x32"] += idx
            regs[a[0]] = np.stack([xw[i] for i in idx], axis=1)
        elif kind == "ld_xr":
            # xb = (uint4*)x + wp * (32 * N) + lane; xb[off]
            idx = wp * (32 * n) + LANES + a[1]
            log["x128"].append(idx); regs[a[0]] = x4[idx]
        elif kind == "mma2":
            acc, off, b = a
            emu_mma2(regs[acc][:, off:off + 4], regs["h"], list(regs[b].T))
        else:
            raise AssertionError(op)
    return regs["d"], log


def emulate_unit(arm, n, unit, xbits, layout=None):
    """Four consumer warps + epilogue STS + helper of one unit, from the generated plan; returns 64 x N and a load log."""
    xw = pack_layout(layout or x_layout(arm), xbits).view(np.uint32)
    w32 = unit.view(np.uint32)
    weight_reads = np.zeros(UNIT, np.int64)
    x_reads = np.zeros(xw.size, np.int64)
    per_warp = []
    ep = np.full((4, 64 * n), np.nan)  # xr4: one partial per warp; other arms: ep[0] is the WG buffer
    for wp in range(4):
        d, log = run_plan(arm, n, wp, w32, xw)
        for idx in log["code"]:
            np.add.at(weight_reads, (16 * idx[:, None] + np.arange(16)).reshape(-1), 1)
        for idx in log["scale"]:
            np.add.at(weight_reads, (4 * idx[:, None] + np.arange(4)).reshape(-1), 1)
        for idx in log["x32"]:
            np.add.at(x_reads, idx, 1)
        for idx in log["x128"]:
            np.add.at(x_reads, (4 * idx[:, None] + np.arange(4)).reshape(-1), 1)
        per_warp.append({k: len(v) for k, v in log.items()})
        if arm == "xr4":
            part = ep[wp].reshape(-1, 2)  # float2 (tile, jp, lane): pp[i * 32] = (d[2i], d[2i + 1])
            for i in range(n):
                k = i * 32 + LANES
                assert np.isnan(part[k]).all(), "partial overwrite"
                part[k, 0], part[k, 1] = d[:, 2 * i], d[:, 2 * i + 1]
        else:
            for j in range(n // 2):
                r = wp * 16 + LG + ((j & 3) >> 1) * 8
                c = (j // 4) * 8 + LT * 2 + (j & 1)
                assert np.isnan(ep[0, r * n + c]).all(), "epilogue overwrite"
                ep[0, r * n + c] = d[:, j]
    assert not np.isnan(ep if arm == "xr4" else ep[0]).any(), "epilogue coverage hole"
    if arm == "xr4":
        p = ep.reshape(4, -1, 4)
        vals = ((p[0] + p[1]) + (p[2] + p[3])) * 256
    else:
        vals = ep[0].reshape(-1, 4) * 256
    out = np.full((64 * n // 4, 4), np.nan)
    out[helper_dst(arm, n)] = vals
    assert not np.isnan(out).any(), "output coverage hole"
    return out.reshape(64, n), {"weight_reads": weight_reads, "x_reads": x_reads, "per_warp": per_warp}


def emulate_gates():
    """Execute every generated consumer on the CPU against the FP64 reference (exact arms) and the load-count design."""
    weights = make_weights(1, 31)
    valid = np.array([b for b in range(256) if b not in (127, 255)], np.uint8)
    weights["w13_sf"][0, :256, :] = np.resize(valid, (256, 256))  # zero / negative / subnormal / max-finite scales
    bank = prepare_bank(weights, 4, 4)
    rng = np.random.default_rng(5150)
    rows = []
    for n in (8, 16):
        jn = n // 8
        xbits = bf16_bits(rng.standard_normal((n, 1024)))
        xval = (xbits.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
        # Design counts per lane per unit (pwr_opus.md s2): x loads 128/256 -> 32/64 LDS.128 (xw128) -> 8/16 (xr4).
        design = {"ms": {"x32": 128 * jn}, "ms_twin": {"x32": 128 * jn}, "acc4": {"x32": 128 * jn},
                  "xreg": {"x32": 4 * jn}, "xw128": {"x128": 32 * jn}, "xr4": {"x128": 8 * jn}}
        for u in (0, 5):  # row tile 0 / stripe 0 and row tile 1 / stripe 1
            unit = bank[u]
            ref = unpack_tiles(unit[None])[0] @ xval.T
            tol = 1e-9 * max(1.0, float(np.abs(ref).max()))
            for arm in NAMES[1:]:
                out, st = emulate_unit(arm, n, unit, xbits)
                err = float(np.abs(out - ref).max())
                if arm in DIAGNOSTIC:
                    assert err > 1e3 * tol, (arm, n, "bound arm unexpectedly exact: the check is insensitive")
                else:
                    assert err <= tol, (arm, n, u, err, tol)
                assert np.all(st["weight_reads"] == 1), (arm, n, "weight bytes not read exactly once per unit")
                for w in st["per_warp"]:
                    assert w["code"] == 16 and w["scale"] == 8, (arm, n, "weight LDS per lane changed", w)
                    for key in ("x32", "x128"):
                        assert w[key] == design[arm].get(key, 0), (arm, n, key, w)
                xr = st["x_reads"]
                if arm == "xr4":
                    assert np.all(xr == 1), "xr4 must read every fragment-order x word exactly once per unit"
                elif arm != "xreg":
                    assert np.all(xr == 4), (arm, "ms geometry reads the stripe once per row-tile warp")
                rows.append({"arm": arm, "n": n, "unit": u, "max_abs_err": err, "tol": tol,
                             "x_loads_per_lane": st["per_warp"][0], "status": "PASS"})
            # Negative controls: the layout and the reduction must matter.
            bad, _ = emulate_unit("xr4", n, unit, xbits, layout="ms")
            assert np.abs(bad - ref).max() > 1e3 * tol, "xr4 insensitive to the x layout"
            bad, _ = emulate_unit("xw128", n, unit, xbits, layout="ms")
            assert np.abs(bad - ref).max() > 1e3 * tol, "xw128 insensitive to the x layout"
    return {"status": "PASS", "units_per_n": 2, "rows": rows,
            "negative_controls": ["xreg != reference", "xr4 / xw128 with ms-layout x != reference"]}


def layout_gate(n):
    """Fragment permutation, smem capacity of the XR4 partials, and partial/helper coverage."""
    perm = xf_perm(n)
    assert perm.size == n * 512 and np.array_equal(np.sort(perm), np.arange(n * 512)), "fragment layout aliases or omits x"
    x_end = n * 1024 * 2
    partial_end = x_end + 2 * 4 * 64 * n * 4
    assert partial_end <= 65536, (n, partial_end)
    smem = 65536 + 3 * UNIT + 2 * 64 * n * 4 + 4096
    assert smem <= 232448
    # Each warp's partial: N float2 per lane, (tile, jp, lane) -> distinct (row, col) pairs covering 64 x N once.
    cells = set()
    for i in range(n):
        tile, jp = divmod(i, n // 4)
        for lane in range(32):
            row = tile * 16 + (lane >> 2) + (jp & 1) * 8
            col = (jp >> 1) * 8 + (lane & 3) * 2
            cells.update({(row, col), (row, col + 1)})
    assert len(cells) == 64 * n
    return {"n": n, "x_bytes": x_end, "partial_bytes": partial_end - x_end, "partial_end": partial_end,
            "x_prefix": 65536, "dynamic_smem_all_arms": smem, "fragment_words": int(perm.size), "status": "PASS"}


def selected_python(path, names):
    """Grep-style extraction: retain only named top-level definitions, never import the layer."""
    blocks, active = {}, None
    with path.open(encoding="utf-8") as f:
        for line in f:
            m = re.match(r"def (\w+)\(", line)
            if m:
                active = m.group(1) if m.group(1) in names else None
                if active:
                    blocks[active] = line
                continue
            if active:
                if line.strip() and not line[0].isspace() and not blocks[active].rstrip().endswith(":"):
                    active = None
                else:
                    blocks[active] += line
    assert set(blocks) == set(names), (path, set(names) - set(blocks))
    return blocks


def base_pack_cpu(raw, sf):
    """Execute the REAL base's four prepare functions on NumPy-backed torch-shaped tensors.

    The sandbox has no torch installation. This deliberately small CPU adapter
    executes the actual extracted Python, rather than a mock state/accessor.
    The Linux gate additionally compares these same functions using real torch.
    """
    class T(np.ndarray):
        @property
        def device(self): return "cpu"
        def to(self, dtype): return np.asarray(self).astype(dtype).view(T)
        def long(self): return self.to(np.int64)
        def view(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], type):
                if issubclass(shape[0], np.ndarray):
                    return np.ndarray.view(self, type=shape[0])
                return np.asarray(self).view(shape[0]).view(T)
            return self.reshape(*shape)
        def permute(self, *dims): return self.transpose(*dims)
        def contiguous(self): return np.ascontiguousarray(self).view(T)
        def unsqueeze(self, axis): return np.expand_dims(self, axis).view(T)
    class TorchCPU:
        Tensor = T
        int32, int64, uint8 = np.int32, np.int64, np.uint8
        @staticmethod
        def tensor(v, dtype=None, device=None): return np.array(v, dtype=dtype).view(T)
        @staticmethod
        def zeros(shape, dtype=None, device=None): return np.zeros(shape, dtype).view(T)
        @staticmethod
        def arange(n, device=None): return np.arange(n).view(T)
        @staticmethod
        def stack(v, dim=0): return np.stack(v, axis=dim).view(T)
        @staticmethod
        def full_like(a, val): return np.full_like(a, val).view(T)
        @staticmethod
        def zeros_like(a): return np.zeros_like(a).view(T)
        @staticmethod
        def where(c, a, b): return np.where(c, a, b).view(T)
    names = ("_k2a_reencode_scales", "_k2a_tile_row_perm", "_k2a31_pack_words", "_k2a31_prepack_expert")
    blocks = selected_python(BASE, names)
    ns = {"torch": TorchCPU, "K2A_BITPOS": BITPOS}
    for name in names:
        exec(compile(blocks[name], str(BASE) + "::" + name, "exec"), ns)
    w, s = ns[names[-1]](raw.view(T), sf.view(T), False, BITPOS)
    joined = np.concatenate((np.asarray(w).view(np.uint8).reshape(-1), np.asarray(s).reshape(-1)))
    return joined, blocks


def normalized(text):
    return re.sub(r"\s+", " ", text)


def ms_matches_m16xc():
    """ms must be the GPU-validated m16xc mode-3 consumer, epilogue and helper (source-level identity)."""
    if not M16XC.exists():
        return "SKIPPED: kernels/lab/m16xc_micro.py absent"
    spec = importlib.util.spec_from_file_location("m16xc_reference_only", M16XC)
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)  # its __main__ guard keeps it inert
    theirs = mc.generate_cuda()
    assert "else if constexpr (MODE == 3) {\n" + plan_text("ms", 8) + "\n}" in theirs, "ms body differs from m16xc mode 3"
    ours = normalized(generate_cuda())
    for statement in ("const float4* ep = reinterpret_cast<const float4*>(s + EP<N> + wg * 64 * N * 4);",
                      "float4 v = ep[j]; v.x *= 256.f; v.y *= 256.f; v.z *= 256.f; v.w *= 256.f; dst[j] = v;",
                      "float* ep = reinterpret_cast<float*>(s + EP<N> + wg * 64 * N * 4);",
                      "const int r = wp * 16 + (lane >> 2) + ((j & 3) >> 1) * 8;",
                      "const int c = (j / 4) * 8 + (lane & 3) * 2 + (j & 1);", "ep[r * N + c] = d[j];",
                      "if (u >= 2) wait(bars + (8 + wg) * 8, ((u / 2) - 1) & 1, err);",
                      "asm volatile(\"setmaxnreg.inc.sync.aligned.u32 128;\" ::: \"memory\");"):
        assert normalized(statement) in normalized(theirs) and normalized(statement) in ours, statement
    for name in ("ms", "ms_pair", "dec_half2", "wait", "bulk", "init", "arrive", "ready"):
        m = re.search(r"__device__ __forceinline__ \w+ " + name + r"\(.*?\n}\n", theirs, re.S)
        assert m and m.group(0) in generate_cuda(), ("device helper differs from m16xc", name)
    return "PASS"


def cpu_gates():
    src = generate_cuda()
    assert "@" not in src and "__trap" not in src
    assert len(re.findall(r"m16xd<\d+, \d+><<<", src)) == len(ARMS)
    assert len(set(ARMS)) == len(ARMS) == 14
    # Tie the address/protocol replay and the CPU emulator to the generated CUDA access sites.
    for expression in (
        "constexpr int UB = 36864, XS = 65536, RSZ = 3 * UB;",
        "W + (size_t(blockIdx.x) * units + u) * UB",
        "O + (size_t(blockIdx.x) * units + u) * 64 * N",
        "for (int i = tid * 8; i < N * 1024; i += 384 * 8)",
        "reinterpret_cast<const uint4*>(X + i)",
        "int* err = errors + blockIdx.x;", "smids[blockIdx.x] = sm;",
        "reinterpret_cast<const uint32_t*>(x + (2 * Q) * N * 16 + j * 128 + g * 8 + t * 2);",
        "const uint32_t b0 = b[0], b1 = b[32];", "const uint32_t b2 = b[N * 8], b3 = b[N * 8 + 32];",
        "ms(d, h[0], h[4], h[1], h[5], b.x, b.y);", "ms(d, h[2], h[6], h[3], h[7], b.z, b.w);",
        "mma2(d + 4 * j, h, reinterpret_cast<const uint4*>(x)[(Q * (N / 8) + j) * 32 + lane]);",
        "reinterpret_cast<const uint32_t*>(x + j * 128 + g * 8 + t * 2);",
        "return make_uint4(b[0], b[32], b[N * 8], b[N * 8 + 32]);",
        "const uint4* cw = reinterpret_cast<const uint4*>(st + wp * 2048) + lane;",
        "const uint32_t* sw = reinterpret_cast<const uint32_t*>(st + 32768 + wp * 256) + lane;",
        "const uint4* xb = reinterpret_cast<const uint4*>(x) + wp * (32 * N) + lane;",
        "float2* pp = reinterpret_cast<float2*>(s + XP<N> + (wg * 4 + wp) * 64 * N * 4) + lane;",
        "for (int i = 0; i < N; ++i) pp[i * 32] = make_float2(d[2 * i], d[2 * i + 1]);",
        "const float4* pp = reinterpret_cast<const float4*>(s + XP<N> + wg * 4 * 64 * N * 4);",
        "const float4 a = pp[j], b = pp[j + 16 * N], c = pp[j + 32 * N], e = pp[j + 48 * N];",
        "const int l = (2 * j) & 31, r = (2 * j) >> 5, tile = r / (N / 4), jp = r % (N / 4);",
        "const int row = tile * 16 + (l >> 2) + (jp & 1) * 8, col = (jp >> 1) * 8 + (l & 3) * 2;",
        "v.x = ((a.x + b.x) + (c.x + e.x)) * 256.f;", "dst[(row * N + col) >> 2] = v;",
        "template<int N> constexpr int XP = N * 2048;",
    ):
        assert expression in src, ("update address replay / emulator", expression)
    assert src.count("for (int j = lane; j < 64 * N / 4; j += 32)") == 2  # ms helper and xr4 helper
    assert src.count("float d[DN<N, MODE>] = {};") == 1
    handoff = "if (u >= 3) wait(bars + (3 + slot) * 8, ((u / 3) - 1) & 1, err);"
    assert src.count(handoff) == 2  # Producer and consumer, not producer alone.
    assert handoff + "\n      wait(bars + slot * 8, (u / 3) & 1, err);" in src
    # Every arm has its own instance and identical-text arms share one block (ms / ms_twin).
    texts = {(a, n): plan_text(a, n) for a, n in ARMS if a != "null"}
    assert texts["ms", 8] == texts["ms_twin", 8] == texts["ms", 16] == texts["ms_twin", 16]
    assert len(set(texts.values())) == 7  # ms*, acc4, xw128 (N-generic) + xreg8/16 + xr4 8/16
    # Check exact copied CUDA functions against only the matching base snippets.
    wanted = ("unp_ok32", "deq4", "deq4_dq", "bmul2l", "bmul2h", "dec_half2")
    hits = {name: [] for name in wanted}
    active, lines, depth = None, [], 0
    with BASE.open(encoding="utf-8") as f:
        for line in f:
            if active is None:
                for name in wanted:
                    if re.match(r"__device__.*\b" + name + r"\(", line):
                        active, lines, depth = name, [], 0
                        break
            if active is not None:
                lines.append(line)
                depth += line.count("{") - line.count("}")
                if depth == 0 and "}" in line:
                    hits[active].append("".join(lines).rstrip())
                    active = None
    for name in wanted:
        assert any(block in DEQUANT for block in hits[name]), "verbatim dequant mismatch: " + name
    ms_verbatim = ms_matches_m16xc()
    rng = np.random.default_rng(933)
    raw = rng.integers(0, 256, (1, 64, 512), dtype=np.uint8)
    scale = rng.integers(0, 254, (1, 64, 64), dtype=np.uint8)
    scale[scale == 127] = 0
    p = pack_tiles(raw, scale)
    bp, blocks = base_pack_cpu(raw[0], scale[0])
    assert np.array_equal(bp, p[0]), "pack differs from real base prepare functions"
    c = np.stack((raw & 15, raw >> 4), axis=-1).reshape(1, 64, 64, 16)
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
    ref = lut[c & 7] * np.where(c & 8, -1, 1) * fp8_numpy(scale)[..., None]
    assert np.array_equal(unpack_tiles(p), ref.reshape(1, 64, 1024))
    assert np.array_equal(emulate_cuda_a(p[0]),ref.reshape(64,1024)[:,kperm(np.arange(1024))])
    # Exhaustive deq4 bit extraction + scale product, all 254 finite E4M3 codes.
    cases = 0
    for sb in range(256):
        if sb in (127, 255): continue
        code = np.tile(np.arange(16, dtype=np.uint32)[:, None], (1, 8))
        bs, neg, z = reencode(np.array([sb], np.uint8))
        code = np.where(z[0], 0, code ^ (8 if neg[0] else 0)).astype(np.uint32)
        w = np.zeros(16, np.uint32)
        for i, pos in enumerate(BITPOS):
            for bit, dst in enumerate(pos): w |= ((code[:, i] >> bit) & 1) << dst
        ro = lambda n: (w << n) | (w >> (32 - n))
        out = (w & 0x81C081C0, ro(10) & 0x81C081C0, ro(13) & 0x81C081C0,
               ((w & 0x001B001B) * np.uint32(0x1084)) & 0x81C081C0)
        sbits = np.array([(int(bs[0]) << 4) + 0x7580], np.uint32) << 16
        for word in out:
            for shift in (0, 16):
                val = (((word >> shift) & 65535) << 16).view(np.float32).astype(np.float64)
                expected = lut[np.arange(16) & 7] * np.where(np.arange(16) & 8, -1, 1) * fp8_numpy(np.array([sb], np.uint8))[0]
                assert np.array_equal(val * float(sbits.view(np.float32)[0]) * 256, expected)
                cases += 16
    # Every prepare-side branch: raw weight keys, batching, all stripes, E transition, each arm/N.
    weights = make_weights(2, 17)
    prepared = prepare_bank(weights, 130, 4)
    assert prepared.shape == (520, UNIT)
    eco,rco,kco = unit_coords(130,4)
    assert len(set(zip(eco,rco,kco))) == 520
    assert all(len(set(kco[i*130:(i+1)*130]))==1 for i in range(4))
    for u in (0, 1, 64, 127, 128, 129, 130, 259, 519):
        e, r, k = (int(v[u]) for v in (eco,rco,kco))
        sl = weights["w13"][e, r*64:(r+1)*64, k*512:(k+1)*512]
        sf = weights["w13_sf"][e, r*64:(r+1)*64, k*64:(k+1)*64]
        actual, _ = base_pack_cpu(sl, sf)
        assert np.array_equal(actual, prepared[u]), ("bank map", u)
    # Execute the real full-grid prepare path, not just a small representative bank.
    full_weights = make_weights(13, 1801)
    full_bank = prepare_bank(full_weights, 48, GRID)
    assert full_bank.shape == (6336, UNIT) and full_bank.nbytes == 233570304
    ec, rc, kc = unit_coords(48, GRID)
    for u in (0, 47, 48, 511, 512, 6335):
        e, r, k = int(ec[u]), int(rc[u]), int(kc[u])
        expected, _ = base_pack_cpu(full_weights["w13"][e,r*64:(r+1)*64,k*512:(k+1)*512],
                                    full_weights["w13_sf"][e,r*64:(r+1)*64,k*64:(k+1)*64])
        assert np.array_equal(full_bank[u], expected), ("full-grid bank map", u)
    full_bank_bytes = full_bank.nbytes
    del full_bank, full_weights
    bound_rows = address_gates()
    rejected = []
    for key in ("w13", "w13_sf"):
        bad = dict(weights); bad.pop(key)
        try: prepare_bank(bad, 4, 1)
        except KeyError: rejected.append(key)
    assert len(rejected) == 2
    for n in (8, 16):
        x = np.arange(n * 1024, dtype=np.uint16).reshape(n, 1024)
        xp = pack_x(x)
        for step in range(64):
            for col in range(n):
                for k in range(16):
                    idx = step*n*16 + (col//8)*128 + (k//8)*64 + (col%8)*8 + k%8
                    assert xp[idx] == x[col, kperm(step*16+k)]
        # Fragment order holds exactly the ms_pair words of (q, j, lane), in b0..b3 order.
        xf = pack_xf(x).view(np.uint32).reshape(32, n // 8, 32, 4)
        words = xp.view(np.uint32)
        for q in (0, 7, 31):
            for lane in (0, 5, 31):
                g, t = lane >> 2, lane & 3
                for j in range(n // 8):
                    b = ((2 * q) * n * 16 + j * 128 + g * 8 + t * 2) // 2
                    assert list(xf[q, j, lane]) == [words[b], words[b + 32], words[b + n * 8], words[b + n * 8 + 32]]
        coords = [(wp*16+(lane>>2)+((j&3)>>1)*8, (j//4)*8+(lane&3)*2+(j&1))
                  for wp in range(4) for lane in range(32) for j in range(n//2)]
        assert len(set(coords)) == 64*n
    layouts = [layout_gate(n) for n in (8, 16)]
    statics = {f"{a}{n}": expected_static(a, n) for a, n in ARMS if a != "null"}
    for n in (8, 16):
        # Design arithmetic (pwr_opus.md s2): per-lane x issue 128 / 256 -> 32 / 64 (xw128) -> 8 / 16 (xr4); same MACs.
        assert statics[f"ms{n}"]["LDS32"] - 8 == 128 * (n // 8)
        assert statics[f"xw128{n}"]["LDS128_consumer_min"] - 16 == 32 * (n // 8)
        assert statics[f"xr4{n}"]["LDS128_consumer_min"] - 16 == 8 * (n // 8)
        assert statics[f"xreg{n}"]["LDS32"] - 8 == 4 * (n // 8)
        assert {statics[f"{a}{n}"]["HMMA"] for a in NAMES[1:]} == {64 * (n // 8)}, "every arm issues the same MACs"
    emulation = emulate_gates()
    # Event-level interleaving of ring/epilogue protocols, including all final drains.
    protocol_gates()
    phase_report = ring_phase_gates()
    reset_report = input_reset_gates()
    # Compiler-log parser and decision boundary checks use synthetic fixtures,
    # never presented as hardware measurements and never written to a result file.
    fixtures=[]
    for arm,n in ARMS:
        fixtures.append(f"ptxas info : Compiling entry function '_Z5m16xdILi{n}ELi{ARM_ID[arm]}EEvPKhPKtPfPiS5_i' for 'sm_90a'\n"
                        "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
                        "ptxas info : Used 96 registers, 0 bytes smem\n")
    sass_fixture = ("\t\tFunction : _Z5m16xdILi8ELi6EEvPKhPKtPfPiS5_i\n"
                    "        /*0010*/                   LDS.128 R4, [R2+0x400] ;\n"
                    "        /*0020*/                   LDS R8, [R2] ;\n"
                    "        /*0030*/              @!P0 LDS.64 R8, [R2] ;\n"
                    "        /*0040*/                   HMMA.16816.F32.BF16 R4, R8, R12, R4 ;\n")
    parsed=compiler_report("".join(fixtures)+"ptxas info : (C7514) synthetic parser fixture\n",sass_fixture)
    assert {(r["arm"],r["n"]) for r in parsed["ptxas"]}==set(ARMS)
    assert len(parsed["c75xx_lines"])==1
    fx = parsed["sass_opcodes_static"]["_Z5m16xdILi8ELi6EEvPKhPKtPfPiS5_i"]
    rr = region_registers("        /*0010*/                   USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0x80 ;\n"
                          "        /*0020*/                   HMMA.16816.F32.BF16 R4, R8, R70, R4 ;\n"
                          "        /*0030*/                   USETMAXREG.DEALLOC.CTAPOOL 0x20 ;\n"
                          "        /*0040*/                   LDS.128 R12, [R2] ;  /* R99 in a comment */\n")
    assert rr == {"consumer_region_regs": 71, "helper_region_regs": 13}, rr  # named registers only (comment ignored)
    assert (lds_count(fx, 128), lds_count(fx, 64), lds_count(fx, 32), op_total(fx, "HMMA")) == (1, 1, 1, 1)
    verdict_fixture_gates()
    schedule=list(replay_schedule(12))
    assert all(a[2]!=b[2] for a,b in zip(schedule,schedule[1:]))
    assert all([sum(k==key and b==bank for _,k,b in schedule) for bank in range(3)]==[4,4,4] for key in ARMS)
    # Exercise NaN rejection rather than silently accepting malformed E4M3 input.
    for nan in (127, 255):
        try: reencode(np.array([nan], np.uint8))
        except ValueError: pass
        else: raise AssertionError("NaN scale accepted")
    return {"status": "PASS", "verbatim_cuda_functions": list(wanted), "ms_matches_m16xc": ms_verbatim,
            "finite_dequant_lanes": cases, "base_prepare_functions_executed": list(blocks), "bank_units": 520,
            "weight_layout": {"w13": [2,8192,2048], "w13_sf_raw_e4m3": [2,8192,256]},
            "arm_instances": [f"{a}_n{n}" for a,n in ARMS], "no_layer_state_access": True,
            "protocol_interleavings":120, "verdict_boundaries":"PASS", "compiler_parser":"PASS",
            "full_grid_prepare":{"units":6336,"bank_bytes":full_bank_bytes,"status":"PASS"},
            "global_address_bounds":bound_rows, "ring_phase_replay":phase_report,
            "correctness_input_reset":reset_report, "xr4_layout": layouts, "expected_static_sass": statics,
            "cpu_emulation": emulation, "source_sha256": hashlib.sha256(src.encode()).hexdigest()}


def address_gates():
    class Tensor:
        def __init__(self, count, width): self.count, self.width = count, width
        def numel(self): return self.count
        def element_size(self): return self.width
        def is_contiguous(self): return True
        def data_ptr(self): return 0x100000000  # Exercise an address above int32.
    rows=[]
    for arm,n in ARMS:
        rows.append(tensor_bounds(arm,n,48,GRID,Tensor(GRID*48*UNIT,1),Tensor(n*1024,2),
                                 Tensor(GRID*48*64*16,4),Tensor(GRID*5,4),Tensor(GRID,4)))
    # Every global buffer gets a one-byte-short negative control at its last access.
    last=next(r for r in rows if r["arm"]=="xr4" and r["n"]==16)
    sizes={k:v["tensor_bytes"] for k,v in last["accesses"].items()}
    for key in sizes:
        bad={**sizes,key:last["accesses"][key]["end_byte_exclusive"]-1}
        try: address_bounds("xr4",16,48,GRID,bad)
        except AssertionError as ex:
            assert ex.args[0][:3] == ("xr4",16,key), (key, ex)
        else: raise AssertionError(("short tensor accepted",key))
    return rows


def input_reset_gates():
    class Buffer:
        def __init__(self, x): self.value=x.copy()
        def copy_(self, x): self.value[:]=x
    cases=0
    for n in (8,16):
        bits=np.arange(n*1024,dtype=np.uint16).reshape(n,1024)
        originals={lay:pack_layout(lay,bits) for lay in ("ms","frag")}
        assert not np.array_equal(originals["ms"],originals["frag"]), "layouts must differ"
        buffers={lay:Buffer(v) for lay,v in originals.items()}
        for arm,an in ARMS:
            if an!=n: continue
            lay=x_layout(arm)
            px=buffers[lay]
            # Replay -X from a previous arm, including its early-failure path.
            px.value ^= np.uint16(0x8000)
            assert not np.array_equal(px.value,originals[lay]), "negative control did not mutate X"
            reset_correctness_input(px,originals[lay])
            assert np.array_equal(px.value,originals[lay]), (arm,n,"inherited -X")
            cases+=1
    return {"status":"PASS","arm_resets":cases,"legacy_negated_input_rejected":True,"layouts":["ms","frag"]}


def ring_phase_gates():
    """Replay actual parity tests with out-of-order TMA and eight independent warps.

    The older slot==unit simulator assumes away future-phase waits. In an odd
    ring a slot changes WG owner on reuse, so that assumption needs a handshake.
    """
    def run(units,seed,fixed,prefix=()):
        rng=np.random.default_rng(seed)
        full,empty=[0]*3,[0]*3  # Number of completed generations, parity = count & 1.
        inflight=[None]*3
        arrivals=[set() for _ in range(3)]
        next_u=[w//4 for w in range(8)]
        stage=[1]*8  # 0 previous-empty wait, 1 full wait, 2 consume/release.
        producer=0
        for tick in range(units*50):
            ops=[]
            if producer<units:
                slot=producer%3
                if producer<3 or (empty[slot]&1)!=((producer//3-1)&1): ops.append(("p",0))
            ops += [("t",s) for s in range(3) if inflight[s] is not None]
            for w,u in enumerate(next_u):
                if u>=units: continue
                s,g=u%3,u//3
                if stage[w]==0 and (empty[s]&1)!=((g-1)&1): ops.append(("e",w))
                if stage[w]==1 and (full[s]&1)!=(g&1): ops.append(("f",w))
                if stage[w]==2: ops.append(("r",w))
            if not ops: break
            op,k=prefix[tick] if tick<len(prefix) else ops[int(rng.integers(len(ops)))]
            assert (op,k) in ops, ("invalid replay prefix",op,k)
            if op=="p":
                s,g=producer%3,producer//3
                assert inflight[s] is None and full[s]==g and empty[s]==g, "overlapping refill"
                inflight[s]=producer; producer+=1
            elif op=="t":
                assert full[k]==inflight[k]//3
                full[k]+=1; inflight[k]=None
            else:
                u=next_u[k]; s,g=u%3,u//3
                if op=="e":
                    assert empty[s]==g, "future empty phase"
                    stage[k]=1
                elif op=="f":
                    assert full[s]==g+1, "future full phase accepted before TMA completion"
                    stage[k]=2
                else:
                    assert empty[s]==g and k not in arrivals[s]
                    arrivals[s].add(k)
                    if len(arrivals[s])==4:
                        assert arrivals[s]==set(range((u&1)*4,(u&1)*4+4))
                        empty[s]+=1; arrivals[s].clear()
                    next_u[k]+=2
                    stage[k]=0 if fixed and next_u[k]>=3 else 1
        assert producer==units and all(u>=units for u in next_u), "deadlock"
        assert inflight==[None]*3 and not any(arrivals)
        for s in range(3):
            uses=len(range(s,units,3))
            assert full[s]==empty[s]==uses, "terminal drain incomplete"
    # Issue 0/1/2, finish only 1, let one WG1 warp consume 1 then test future 3.
    witness=(("p",0),("p",0),("p",0),("t",1),("f",4),("r",4),("f",4))
    try: run(48,0,False,witness)
    except AssertionError as ex: assert "future full phase" in str(ex)
    else: raise AssertionError("legacy parity race was not reproduced")
    # Fixed code blocks that last operation on empty(0); then all schedules drain.
    run(48,0,True,witness[:-1])
    for units in (4,6,8,48,50):
        for seed in range(24): run(units,seed,True)
    return {"status":"PASS","warp_interleavings":121,"legacy_future_phase_rejected":True,
            "witness":"issue 0,1,2; complete 1; WG1 consumes 1; future full(3) looked ready"}


def protocol_gates():
    for units in (4, 6, 8, 48, 50):
        for null in (False, True):
            for seed in range(12):
                rng = np.random.default_rng(seed)
                slots, epi = [None]*3, [None]*2
                next_c, next_h, prod = [0,1], [0,1], 0
                consumed, drained = set(), set()
                for _ in range(units * 20):
                    ops = []
                    if prod < units and slots[prod % 3] is None: ops.append((0,0))
                    for wg in range(2):
                        u = next_c[wg]
                        if u < units and slots[u % 3] == u and (null or epi[wg] is None): ops.append((1,wg))
                        if not null and epi[wg] == next_h[wg]: ops.append((2,wg))
                    if not ops: break
                    op,wg = ops[int(rng.integers(len(ops)))]
                    if op == 0: slots[prod % 3] = prod; prod += 1
                    elif op == 1:
                        u=next_c[wg]; consumed.add(u); slots[u%3]=None; next_c[wg]+=2
                        if not null: epi[wg]=u
                    else: drained.add(epi[wg]); epi[wg]=None; next_h[wg]+=2
                assert consumed == set(range(units)) and slots == [None]*3
                assert null or (drained == consumed and epi == [None]*2)
                for slot in range(3):
                    uses = list(range(slot,units,3))
                    assert ((uses[-1] // 3) & 1) == ((len(uses)-1)&1)


@contextlib.contextmanager
def capture_fds(path):
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with path.open("wb") as log:
        try:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            for fd, original in zip((1,2), saved): os.dup2(original, fd); os.close(original)


def lds_width(op):
    parts = op.split(".")
    if parts[0] != "LDS":
        return None
    return 128 if "128" in parts[1:] else 64 if "64" in parts[1:] else 32


def lds_count(counts, width):
    return sum(v for op, v in counts.items() if lds_width(op) == width)


def op_total(counts, prefix):
    return sum(v for op, v in counts.items() if op.split(".")[0] == prefix)


def compiler_report(log, sass):
    blocks = []
    for chunk in re.split(r"(?=ptxas info\s*: Compiling entry function)", log):
        m = re.search(r"Compiling entry function '([^']+)'", chunk)
        if not m: continue
        regs = re.search(r"Used (\d+) registers", chunk)
        spill = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", chunk)
        smem = re.search(r"(\d+) bytes smem", chunk)
        blocks.append({"symbol": m[1], "registers": int(regs[1]) if regs else None,
                       "stack_bytes": int(spill[1]) if spill else None,
                       "spill_store_bytes": int(spill[2]) if spill else None,
                       "spill_load_bytes": int(spill[3]) if spill else None,
                       "static_smem_bytes": int(smem[1]) if smem else 0,
                       "ptxas_lines": chunk.strip().splitlines()})
        instance = re.search(r"m16xdILi(\d+)ELi(\d+)E",m[1])
        if instance:
            blocks[-1].update({"n":int(instance[1]),"arm":NAMES[int(instance[2])]})
    funcs, regions = {}, {}
    for chunk in re.split(r"(?=Function\s*:\s*)", sass):
        m = re.match(r"Function\s*:\s*(\S+)", chunk)
        if not m: continue
        counts = {}
        for op in re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P[0-9T]+\s+)?([A-Z][A-Z0-9_.]*)", chunk):
            counts[op] = counts.get(op,0)+1
        funcs[m[1]] = counts
        regions[m[1]] = region_registers(chunk)
    return {"ptxas": blocks, "c75xx_lines": [l for l in log.splitlines() if re.search(r"C75\d\d",l)],
            "sass_opcodes_static": funcs, "sass_region_registers_static": regions}


def region_registers(chunk):
    """ptxas reports the setmaxnreg launch allocation (128) for every instance. The real footprint, statically:
    1 + the highest NAMED R index between USETMAXREG.TRY_ALLOC and .DEALLOC (consumer warps) and after DEALLOC
    (helpers + terminal code). A lower bound: a vector / MMA register tuple can extend it by up to 3."""
    lines = [l.split(";")[0] for l in chunk.splitlines() if re.search(r"/\*[0-9a-f]{4,}\*/\s+\S", l)]
    a = next((i for i, l in enumerate(lines) if "USETMAXREG.TRY_ALLOC" in l), None)
    d = next((i for i, l in enumerate(lines) if "USETMAXREG.DEALLOC" in l), None)
    if a is None or d is None or d < a:
        return {"consumer_region_regs": None, "helper_region_regs": None}
    top = lambda ls: 1 + max((int(r) for l in ls for r in re.findall(r"\bR(\d+)\b", l)), default=-1)
    return {"consumer_region_regs": top(lines[a:d]), "helper_region_regs": top(lines[d:])}


def build(out, compile_only):
    from torch.utils.cpp_extension import load_inline
    src = generate_cuda()
    tag = hashlib.sha256((CPP+src).encode()).hexdigest()[:12]
    # Fresh per invocation to ensure ptxas diagnostics are captured even with a prior cache.
    # The compile-only gate builds in a private temp dir (fast local disk, nothing beside the source).
    root = Path(tempfile.mkdtemp(prefix="m16xd_gate_")) if compile_only else out.parent
    builddir = root / ("build_"+tag+"_"+str(os.getpid()))
    builddir.mkdir(parents=True, exist_ok=False)
    logfile = out.with_suffix(".compile.log")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ.setdefault("MAX_JOBS", "2")
    with capture_fds(logfile):
        mod = load_inline(name="m16xd_"+tag, cpp_sources=CPP, cuda_sources=src, functions=None,
                          extra_cflags=["-O3","-std=c++20","-DNDEBUG"],
                          extra_cuda_cflags=["-O3","-std=c++20","-DNDEBUG","-lineinfo",
                                             "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v,--warn-on-spills",
                                             "--keep", "--keep-dir=."],
                          build_directory=str(builddir), with_cuda=True, verbose=True, no_implicit_headers=True)
    log = logfile.read_text(errors="replace")
    sass = subprocess.run(["cuobjdump","--dump-sass",mod.__file__], capture_output=True, text=True, timeout=60, check=True).stdout
    resources = subprocess.run(["cuobjdump","--dump-resource-usage",mod.__file__], capture_output=True, text=True, timeout=15, check=True).stdout
    out.with_suffix(".sass").write_text(sass, encoding="utf-8")
    out.with_suffix(".resources.txt").write_text(resources, encoding="utf-8")
    result = compiler_report(log,sass)
    result.update({"extension":mod.__file__,"compile_log":str(logfile),"build_dir":str(builddir),"resource_dump":resources})
    kernels = [r for r in result["ptxas"] if "m16xd" in r["symbol"]]
    per = {(r["arm"], r["n"]): r for r in kernels if "arm" in r}
    sass_by, region_by = {}, {}
    for name, counts in result["sass_opcodes_static"].items():
        m = re.search(r"m16xdILi(\d+)ELi(\d+)E", name)
        if m:
            sass_by[NAMES[int(m[2])], int(m[1])] = counts
            region_by[NAMES[int(m[2])], int(m[1])] = result["sass_region_registers_static"][name]
    compute = [k for k in ARMS if k[0] != "null"]
    table = {}
    for arm, n in ARMS:
        c = sass_by.get((arm, n), {})
        r = per.get((arm, n), {})
        table[f"{arm}{n}"] = {"registers": r.get("registers"), "spill_store_bytes": r.get("spill_store_bytes"),
                              "spill_load_bytes": r.get("spill_load_bytes"), "stack_bytes": r.get("stack_bytes"),
                              "HMMA": op_total(c, "HMMA"), "LDS32": lds_count(c, 32), "LDS64": lds_count(c, 64),
                              "LDS128": lds_count(c, 128), "STS": op_total(c, "STS"), "UBLKCP": op_total(c, "UBLKCP"),
                              **region_by.get((arm, n), {"consumer_region_regs": None, "helper_region_regs": None}),
                              "expected": expected_static(arm, n) if arm != "null" else None}
    result["per_arm"] = table
    ok = len(per) == len(ARMS) and len(sass_by) == len(ARMS)
    result["gates"] = {
        "all_instances": ok,
        "registers_le_128": bool(kernels) and all(r["registers"] is not None and r["registers"] <= 128 for r in kernels)
                            and ok and all(table[f"{a}{n}"]["consumer_region_regs"] is not None
                                           and table[f"{a}{n}"]["consumer_region_regs"] <= 128
                                           and table[f"{a}{n}"]["helper_region_regs"] <= 32 for a, n in ARMS),
        # New arms: zero spill / stack. Inherited controls keep m16xc's <= 32 B tolerance (they measured 0).
        "new_arms_no_spill_no_stack": ok and all(per[k]["spill_store_bytes"] == 0 and per[k]["spill_load_bytes"] == 0
                                                 and per[k]["stack_bytes"] == 0 for k in ARMS if k[0] in NEW_ARMS),
        "controls_spill_le_32": ok and all(per[k]["spill_store_bytes"] <= 32 and per[k]["spill_load_bytes"] <= 32
                                           and per[k]["stack_bytes"] <= 16 for k in ARMS if k[0] in CONTROLS),
        "hmma_static_exact": ok and all(table[f"{a}{n}"]["HMMA"] == expected_static(a, n)["HMMA"] for a, n in compute)
                             and all(table[f"null{n}"]["HMMA"] == 0 for n in (8, 16)),
        # 32-bit LDS appear only in the consumer: scales + per-k x loads. Exact for the ms-geometry arms; xreg / xw128 /
        # xr4 must have deleted the per-k x loads (ptxas may rematerialize <= 2 fragment words at the register cap).
        "lds32_static": ok and all(table[f"{a}{n}"]["LDS32"] == expected_static(a, n)["LDS32"] for a, n in compute
                                   if a in ("ms", "ms_twin", "acc4"))
                        and all(expected_static(a, n)["LDS32"] <= table[f"{a}{n}"]["LDS32"] <= expected_static(a, n)["LDS32"] + 2
                                for a, n in compute if a in ("xreg", "xw128", "xr4")),
        "lds128_consumer_present": ok and all(table[f"{a}{n}"]["LDS128"] >= expected_static(a, n)["LDS128_consumer_min"]
                                              for a, n in compute),
        "xw128_lds128_delta_exact": ok and all(table[f"xw128{n}"]["LDS128"] - table[f"ms{n}"]["LDS128"] == 32 * (n // 8)
                                               for n in (8, 16)),
        "bulk_copy_each_instance": ok and all(table[f"{a}{n}"]["UBLKCP"] > 0 for a, n in ARMS),
        "twin_identical_sass": ok and all(sass_by["ms", n] == sass_by["ms_twin", n] for n in (8, 16)),
        "no_c75xx_warnings": not result["c75xx_lines"],
    }
    result["status"] = "PASS" if all(result["gates"].values()) else "FAIL"
    return mod,result


class Timer:
    """Copy of judge.py:318-358 gate/flush/event order; measured empty kernel graph."""
    def __init__(self, torch):
        self.torch = torch
        self.stream = torch.cuda.Stream()
        self.flush_buf = torch.ones(256*1024*1024//4, dtype=torch.float32,device="cuda")
        self.sink = torch.empty((),dtype=torch.float32,device="cuda")
        self.gate_cyc = None
    def capture(self,fn):
        t=self.torch
        with t.cuda.stream(self.stream):
            fn(); t.cuda.synchronize()
            g=t.cuda.CUDAGraph()
            with t.cuda.graph(g,stream=self.stream): fn()
        t.cuda.synchronize()
        return g
    def gate(self):
        t=self.torch
        if self.gate_cyc is None:
            cyc=int(300*1980)
            for _ in range(3):
                a,b=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                a.record(); t.cuda._sleep(cyc); b.record(); b.synchronize()
                cyc=int(cyc*300/max(a.elapsed_time(b)*1e3,1.0))
            self.gate_cyc=cyc
        t.cuda._sleep(self.gate_cyc)
    def replay(self,g):
        t=self.torch
        with t.cuda.stream(self.stream):
            t.sum(self.flush_buf,dim=0,out=self.sink)
            self.gate()
            a,b=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record()
        b.synchronize()
        return a.elapsed_time(b)*1e3


def clocks():
    cmd=["nvidia-smi","--query-gpu=index,name,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu","--format=csv,noheader,nounits"]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
    if p.returncode: raise RuntimeError("nvidia-smi clock query failed: "+p.stderr)
    return {"host_time":time.time(),"query":cmd,"csv":p.stdout.strip()}


def smi_sample(seconds, replay_fn, torch):
    """kernels/lab/f5_data_run.py: back-to-back replays for `seconds` while nvidia-smi samples W / MHz every 50 ms."""
    cmd = ["nvidia-smi", "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,nounits", "-lms", "50"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    n = 0
    try:
        time.sleep(0.3)
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            for _ in range(100):
                replay_fn()
            torch.cuda.synchronize()
            n += 100
    finally:
        p.terminate()
        try:
            out, _ = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
    rows = []
    for line in out.splitlines():
        try:
            a, b = (float(s.strip()) for s in line.split(","))
            rows.append((a, b))
        except ValueError:
            continue
    k = len(rows) // 10  # drop the first and last 10 % (ramp / tail)
    core = rows[k:len(rows) - k] if len(rows) > 2 * k + 2 else rows
    if not core:
        return {"samples": 0, "launches": n}
    return {"samples": len(core), "launches": n, "seconds": seconds,
            "power_w_median": statistics.median(r[0] for r in core), "power_w_max": max(r[0] for r in core),
            "smi_sm_mhz_median": statistics.median(r[1] for r in core)}


def reset_correctness_input(px, original):
    # A preceding arm's last graph replay deliberately leaves -X in px.
    px.copy_(original)


def gpu_correctness(mod, torch, results=None):
    """Four CTAs x four units, all K stripes, every real specialization, FP32 CPU torch oracle."""
    weights=make_weights(1,31)
    # Include zero, negative, subnormal and maximum finite E4M3, without NaNs.
    valid=np.array([b for b in range(256) if b not in (127,255)],np.uint8)
    weights["w13_sf"][0,:256,:] = np.resize(valid,(256,256))
    bank=prepare_bank(weights,4,4)
    w=torch.from_numpy(bank).cuda()
    err=torch.zeros(4,device="cuda",dtype=torch.int32)
    sm=torch.empty(4*5,device="cuda",dtype=torch.int32)
    if results is None: results=[]
    for n in (8,16):
        gen=torch.Generator().manual_seed(411+n)
        x=torch.randn(n,1024,generator=gen).to(torch.bfloat16)
        xb=x.view(torch.int16).numpy().view(np.uint16)
        def packed(lay,sign=1):
            bits=(sign*x).view(torch.int16).numpy().view(np.uint16)
            return torch.from_numpy(pack_layout(lay,bits).view(np.int16)).cuda()
        originals={lay:packed(lay) for lay in ("ms","frag")}
        buffers={lay:torch.empty_like(o) for lay,o in originals.items()}
        assert np.array_equal(originals["ms"].cpu().numpy().view(np.uint16),pack_x(xb))
        refs=[]
        for expert,row,stripe in zip(*unit_coords(4,4)):
            raw=torch.from_numpy(weights["w13"][expert,row*64:(row+1)*64,stripe*512:(stripe+1)*512].copy())
            sf=torch.from_numpy(weights["w13_sf"][expert,row*64:(row+1)*64,stripe*64:(stripe+1)*64].copy()).view(torch.float8_e4m3fn).float()
            c=torch.stack((raw&15,raw>>4),dim=-1).reshape(64,64,16).long()
            lut=torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float32)
            dw=lut[c&7]*torch.where((c&8)!=0,-1.,1.)*sf[:,:,None]
            refs.append(dw.reshape(64,1024).float() @ x.float().T)
        reference=torch.stack(refs)
        scale=torch.stack([r.abs().amax().clamp_min(1) for r in refs])[:,None,None]
        for arm,an in ARMS:
            if an != n: continue
            lay=x_layout(arm); px=buffers[lay]
            reset_correctness_input(px,originals[lay])
            out=torch.full((16,64,n),float("nan"),device="cuda",dtype=torch.float32)
            tensor_bounds(arm,n,4,4,w,px,out,sm,err)
            def call(arm=arm,px=px,out=out): mod.launch(ARM_ID[arm],n,4,4,w.data_ptr(),px.data_ptr(),out.data_ptr(),sm.data_ptr(),err.data_ptr(),torch.cuda.current_stream().cuda_stream)
            call(); torch.cuda.synchronize()
            if err.any().item():
                results.append({"arm":arm,"n":n,"status":"FAIL","why":"device protocol timeout"}); err.zero_(); continue
            got=out.cpu()
            if arm=="null":
                results.append({"arm":arm,"n":n,"status":"PASS" if bool(torch.isnan(got).all()) else "FAIL",
                                "check":"null protocol; output untouched"})
                continue
            if arm in DIAGNOSTIC:
                results.append({"arm":arm,"n":n,"status":"PASS" if bool(torch.isfinite(got).all()) else "FAIL","diagnostic":True,
                                "check":"bound arm: protocol + every output written; values invalid by construction (excluded)",
                                "max_abs_error_vs_reference":float((got-reference).abs().nan_to_num(1e30).max().item())})
                continue
            # FP32 reduction order differs; bound by FP32 sum |W*x|, not a loose BF16 tolerance.
            delta=(got-reference).abs()
            if not (torch.isfinite(got).all() and torch.all(delta <= 2e-5*scale)):   # lead: record, exclude from the verdict, keep timing
                results.append({"arm":arm,"n":n,"status":"FAIL","max_abs_error":float(delta.nan_to_num(1e30).max().item())}); continue
            # Replays use fresh inputs and poisoned outputs to catch cached/stale data.
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): call()
            torch.cuda.synchronize()
            for sign in (-1,1,-1):
                px.copy_(packed(lay,sign))
                out.fill_(float("nan")); torch.cuda.synchronize(); graph.replay(); torch.cuda.synchronize()
                ok_r=(not err.any().item()) and bool(torch.all((out.cpu()-sign*reference).abs()<=2e-5*scale))
                if not ok_r: break
            if not ok_r:
                results.append({"arm":arm,"n":n,"status":"FAIL","why":"graph replay mismatch"}); err.zero_(); continue
            results.append({"arm":arm,"n":n,"max_abs_error":delta.max().item(),"status":"PASS"})
    return results


def verdict(rows):
    """Pre-registered decisions of pwr_opus.md s3 on CORRECT rows (+ the diagnostic xreg bound)."""
    by = {(r["arm"], r["n"]): r for r in rows}
    us = lambda a, n: by[a, n]["stage_us"]
    mhz = lambda a, n: by[a, n]["clock_mhz"]
    lines, res = [], {}
    valid = True
    for n in (8, 16):
        if ("ms", n) in by and ("ms_twin", n) in by:
            du, dm = abs(us("ms_twin", n) - us("ms", n)), abs(mhz("ms_twin", n) - mhz("ms", n))
            ok = du <= 0.03 and dm <= 15
            lines.append(f"VALIDITY N={n}: |ms_twin - ms| = {du:.4f} us (<= 0.03), {dm:.1f} MHz (<= 15) -> {'PASS' if ok else 'FAIL'}")
        else:
            ok = False
            lines.append(f"VALIDITY N={n}: ms or ms_twin missing / incorrect -> FAIL")
        valid &= ok
    gate0 = None
    if all(k in by for k in (("xreg", 8), ("ms", 8), ("xreg", 16), ("ms", 16))):
        d8, d16 = us("xreg", 8) - us("ms", 8), us("xreg", 16) - us("ms", 16)
        gate0 = d8 >= -0.04 and d16 >= -0.06
        res.update({"xreg_minus_ms_n8": d8, "xreg_minus_ms_n16": d16})
        lines.append(f"GATE0: xreg - ms = {d8:+.4f} us (N8, kill if >= -0.04), {d16:+.4f} us (N16, kill if >= -0.06) -> "
                     + ("KILL the x-path family (XR4, XRR, xw128)" if gate0 else "PASS (x issue is exposed)"))
    else:
        lines.append("GATE0: xreg or ms missing -> HOLD")
    xr4 = "HOLD"
    cheap = {a: us(a, 16) for a in ("acc4", "xw128") if (a, 16) in by}
    if ("xr4", 8) in by and ("xr4", 16) in by and gate0 is not None:
        x8, x16 = us("xr4", 8), us("xr4", 16)
        kill_abs = x8 > 1.20 and x16 > 1.30
        kill_rel = bool(cheap) and x16 > min(cheap.values()) - 0.02
        go = x8 <= 1.15 and x16 <= 1.24
        why = [w for w, hit in (("Gate 0", gate0), ("xr4 > 1.20 (N8) AND > 1.30 (N16)", kill_abs),
                                (f"xr4_16 > min(acc4, xw128)_16 - 0.02 = {min(cheap.values()) - 0.02:.4f}" if cheap else "", kill_rel)) if hit]
        if why:
            xr4 = "KILL"
        elif len(cheap) < 2:
            xr4 = "HOLD"  # the relative KILL rule needs both cheaper arms correct
            why = ["acc4_16 or xw128_16 missing / incorrect: relative KILL rule not evaluable"]
        else:
            xr4 = "GO" if go else "WEAK"
        res.update({"xr4_n8": x8, "xr4_n16": x16})
        lines.append(f"XR4: N8 {x8:.4f} us (GO <= 1.15, KILL > 1.20), N16 {x16:.4f} us (GO <= 1.24, KILL > 1.30); "
                     f"min(acc4, xw128)_16 = {min(cheap.values()) if cheap else float('nan'):.4f} -> {xr4}"
                     + (f" ({'; '.join(why)})" if why else ""))
        for n in (8, 16):
            dm = mhz("xr4", n) - mhz("ms", n)
            lo, hi = PRE["clock_est_dmhz_xr4_minus_ms"][str(n)]
            note = "in EST band" if lo <= dm <= hi else "outside EST band"
            if n == 16 and dm < 10:
                note += "; < +10 at N16 -> SMEM < 0.15 pJ/bit, LDGW's energy case dies (rank it on cycles only)"
            if dm >= 60:
                note += "; >= +60 -> SMEM >= 0.5 pJ/bit, LDGW moves up"
            res[f"xr4_dmhz_n{n}"] = dm
            lines.append(f"CLOCK N={n}: dMHz(xr4 - ms) = {dm:+.1f} (EST +{lo}..+{hi}) -> {note}")
    else:
        lines.append("XR4: xr4 missing / incorrect at some N, or Gate 0 not evaluable -> HOLD")
    for arm in ("acc4", "xw128"):
        if (arm, 8) in by and (arm, 16) in by:
            alone = us(arm, 8) <= 1.15 and us(arm, 16) <= 1.24
            lines.append(f"CHEAPER {arm}: N8 {us(arm, 8):.4f}, N16 {us(arm, 16):.4f} us -> "
                         + ("passes the GO line alone (port candidate if xr4 is KILLed by the relative rule)" if alone
                            else "does not pass GO alone"))
    status = xr4 if valid else f"INVALID (twin control failed; xr4 would be {xr4})"
    res.update({"status": status, "xr4": xr4, "gate0_kill": gate0, "valid": valid, "lines": lines})
    return res


def verdict_fixture_gates():
    def rows(**over):
        base = {("ms", 8): (1.227, 1762), ("ms_twin", 8): (1.230, 1760), ("ms", 16): (1.379, 1729),
                ("ms_twin", 16): (1.375, 1731), ("xreg", 8): (1.15, 1800), ("xreg", 16): (1.25, 1790),
                ("acc4", 8): (1.21, 1770), ("acc4", 16): (1.33, 1740), ("xw128", 8): (1.20, 1770),
                ("xw128", 16): (1.30, 1750), ("xr4", 8): (1.14, 1785), ("xr4", 16): (1.23, 1770)}
        for k, v in over.items():
            arm, n = k.rsplit("_", 1)
            if v is None: base.pop((arm, int(n)))
            else: base[arm, int(n)] = v
        return [{"arm": a, "n": n, "stage_us": u, "clock_mhz": m} for (a, n), (u, m) in base.items()]
    assert verdict(rows())["status"] == "GO"
    assert verdict(rows(xr4_16=(1.29, 1770)))["status"] == "KILL"           # relative rule: 1.29 > 1.30 - 0.02
    assert verdict(rows(xr4_8=(1.18, 1780), xr4_16=(1.26, 1760)))["status"] == "WEAK"
    assert verdict(rows(xr4_8=(1.21, 1780), xr4_16=(1.31, 1700), acc4_16=(1.40, 1700), xw128_16=(1.40, 1700)))["status"] == "KILL"
    assert verdict(rows(xreg_8=(1.19, 1770), xreg_16=(1.33, 1740)))["status"] == "KILL"   # Gate 0
    assert verdict(rows(xreg_8=(1.19, 1770)))["status"] == "GO"                          # Gate 0 needs BOTH N
    assert verdict(rows(ms_twin_16=(1.42, 1731)))["status"].startswith("INVALID")
    assert verdict(rows(ms_twin_8=(1.227, 1740)))["status"].startswith("INVALID")       # 22 MHz twin gap
    assert verdict(rows(xr4_16=None))["status"] == "HOLD"
    assert verdict(rows(acc4_16=None))["status"] == "HOLD"
    assert verdict([])["status"].startswith("INVALID")
    lines = verdict(rows(xr4_16=(1.23, 1735)))["lines"]
    assert any("SMEM < 0.15 pJ/bit" in l for l in lines)


def replay_schedule(rounds):
    for rnd in range(rounds):
        order=ARMS[rnd%len(ARMS):]+ARMS[:rnd%len(ARMS)]
        for key in order:
            # Same N has the same bank for paired null ratios; every consecutive
            # timed replay changes banks, including round boundaries.
            yield rnd,key,(2*rnd+ARMS.index(key))%3


def write_partial(out, report):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def benchmark(mod, args, report, out, deadline):
    import torch
    prop=torch.cuda.get_device_properties(0)
    assert "H200" in prop.name and prop.multi_processor_count==132 and prop.major==9, str(prop)
    report["device"]={"name":prop.name,"sms":prop.multi_processor_count}
    attrs={}
    for arm,n in ARMS:
        reg,local,static,dynamic,occ=mod.info(ARM_ID[arm],n)
        assert 0<reg<=128 and occ==1 and local<=(0 if arm in NEW_ARMS else 32), (arm,n,reg,local,occ)
        attrs[f"{arm}{n}"]={"regs":reg,"local_bytes":local,"static_smem":static,"dynamic_smem":dynamic,"ctas_per_sm":occ}
    for n in (8,16):
        assert len({attrs[f"{a}{n}"]["dynamic_smem"] for a in NAMES})==1, (n,"dynamic smem differs between arms")
    report["kernel_attributes"]=attrs
    report["correctness"]=[]  # Preserve completed per-arm checks if a later launch faults.
    gpu_correctness(mod,torch,report["correctness"])
    assert all(c["status"]=="PASS" for c in report["correctness"] if c["arm"]=="null"), "invalid matched null"
    timer=Timer(torch)
    stream=timer.stream.cuda_stream
    empty=timer.capture(lambda:mod.empty(stream))
    banks=[]
    for b in range(3):
        weights=make_weights(math.ceil(GRID*args.units/512),103+b)
        packed=prepare_bank(weights,args.units,GRID)
        banks.append(torch.from_numpy(packed).cuda())
    report["working_set_bytes_per_replay"]=GRID*args.units*UNIT
    report["distinct_rotating_banks"]=3
    inputs={}
    for n in (8,16):
        gen=torch.Generator().manual_seed(103+n)
        x=torch.randn(n,1024,generator=gen).to(torch.bfloat16)
        bits=x.view(torch.int16).numpy().view(np.uint16)
        for lay in ("ms","frag"):
            inputs[lay,n]=torch.from_numpy(pack_layout(lay,bits).view(np.int16)).cuda()
    output=torch.empty((GRID,args.units,64,16),device="cuda",dtype=torch.float32)
    sm=torch.empty(GRID*5,device="cuda",dtype=torch.int32)
    err=torch.zeros(GRID,device="cuda",dtype=torch.int32)   # lead: zero-initialised and GRID-sized (false protocol errors otherwise)
    graphs={}
    report["launch_bounds"]=[]
    # H2D preparation/allocation belongs to the current stream; captures use timer.stream.
    timer.stream.wait_stream(torch.cuda.current_stream())
    for arm,n in ARMS:
        graphs[arm,n]=[]
        xin=inputs[x_layout(arm),n]
        for bank,w in enumerate(banks):
            bounds=tensor_bounds(arm,n,args.units,GRID,w,xin,output,sm,err)
            report["launch_bounds"].append({"bank":bank,**bounds})
            report["capture_progress"]={"arm":arm,"n":n,"bank":bank,"status":"STARTED"}
            graphs[arm,n].append(timer.capture(lambda arm=arm,n=n,w=w,xin=xin:mod.launch(ARM_ID[arm],n,args.units,GRID,w.data_ptr(),xin.data_ptr(),output.data_ptr(),sm.data_ptr(),err.data_ptr(),stream)))
            assert not err.any().item(), (arm,n,bank,"protocol error")
            assert len(set(sm[:GRID].cpu().tolist()))==GRID, (arm,n,bank,"grid not one CTA per SM")
            report["capture_progress"]["status"]="PASS"
    report["clocks_before"]=clocks()
    before=statistics.median(timer.replay(empty) for _ in range(15))
    samples={k:[] for k in ARMS}
    clock_samples={k:[] for k in ARMS}
    # Matched N arms share each round's bank; rotate both bank and arm position.
    for rnd,key,bank in replay_schedule(args.rounds):
        samples[key].append(timer.replay(graphs[key][bank]))
        stamp=sm[GRID:].view(torch.int64).cpu().numpy().reshape(GRID,2)
        assert (stamp[:,1]>0).all()
        clock_samples[key].append(float(np.median(1000.0*stamp[:,0]/stamp[:,1])))
        assert not err.any().item(), (key,"device protocol error")
        assert len(set(sm[:GRID].cpu().tolist()))==GRID, (key,"SM coverage")
    after=statistics.median(timer.replay(empty) for _ in range(15))
    report["clocks_after"]=clocks()
    overhead=(before+after)/2
    report["empty_kernel_us"]={"before":before,"after":after,"subtracted":overhead}
    good={(c["arm"],c["n"]) for c in report["correctness"] if c["status"]=="PASS"}
    compiled={(r.get("arm"),r.get("n")):r for r in report["compiler"]["ptxas"]}
    rows=[]
    for arm,n in ARMS:
        values=[(v-overhead)/args.units for v in samples[arm,n]]
        assert min(values)>0, "invalid empty-kernel subtraction"
        stage=statistics.median(values)
        null=statistics.median((v-overhead)/args.units for v in samples["null",n])
        paired=[(v-overhead)/(z-overhead) for v,z in zip(samples[arm,n],samples["null",n])]
        cr=compiled.get((arm,n),{})
        row={"arm":arm,"n":n,"stage_us":stage,"ratio_null":stage/null,"paired_ratio_null":statistics.median(paired),
             "clock_mhz":statistics.median(clock_samples[arm,n]), "clock_samples_mhz":clock_samples[arm,n],
             "weight_TBps":GRID*UNIT/(stage*1e6),"samples_total_us":samples[arm,n],
             "stage_p10_us":float(np.percentile(values,10)),"stage_p90_us":float(np.percentile(values,90)),
             "banks_per_round":[(2*r+ARMS.index((arm,n)))%3 for r in range(args.rounds)],
             "regs":attrs[f"{arm}{n}"]["regs"],"local_bytes":attrs[f"{arm}{n}"]["local_bytes"],
             "consumer_region_regs":report["compiler"]["per_arm"][f"{arm}{n}"]["consumer_region_regs"],
             "spill_store_bytes":cr.get("spill_store_bytes"),"spill_load_bytes":cr.get("spill_load_bytes"),
             "units_per_cta":args.units,"diagnostic":arm in DIAGNOSTIC,
             "correct":(arm,n) in good or arm=="null",
             "measurement":"MEASURED CUDA events; (kernel - empty)/units"}
        rows.append(row)
        tag="PASS" if row["correct"] else "FAIL"
        if arm in DIAGNOSTIC: tag="DIAGNOSTIC-"+tag
        print(f"M16XD {arm:7s} N={n:2d} {stage:.4f} us/unit {row['weight_TBps']:.3f} TB/s null={row['ratio_null']:.3f}x "
              f"paired={row['paired_ratio_null']:.3f}x {row['clock_mhz']:.0f} MHz regs={row['regs']}/{row['consumer_region_regs']} "
              f"spill={row['spill_store_bytes']}/{row['spill_load_bytes']} local={row['local_bytes']} correct={tag}",flush=True)
    report["results"]=rows
    report["verdict"]=verdict([r for r in rows if r["correct"]])
    for line in report["verdict"]["lines"]:
        print("M16XD "+line,flush=True)
    print("M16XD VERDICT "+report["verdict"]["status"],flush=True)
    print("M16XD correctness "+" ".join(f'{c["arm"]}{c["n"]}:{c["status"]}' for c in report["correctness"]),flush=True)
    write_partial(out, report)  # timing + verdict survive a timeout in the optional power phase
    # Separate continuous replays (outside the timed protocol): >= 3 s smi W / MHz per arm, within the time budget.
    report["power"]={}
    with torch.cuda.stream(timer.stream):
        for arm,n in ARMS:
            if args.power_seconds<=0 or time.monotonic()+args.power_seconds+3>deadline:
                report["power"][f"{arm}{n}"]={"skipped":"disabled" if args.power_seconds<=0 else "time budget"}; continue
            pw=smi_sample(args.power_seconds,graphs[arm,n][0].replay,torch)
            assert not err.any().item(), (arm,n,"protocol error in the power replay")
            report["power"][f"{arm}{n}"]=pw
            print(f"M16XD power {arm:7s} N={n:2d} {pw.get('power_w_median',float('nan')):.0f} W median, "
                  f"{pw.get('power_w_max',float('nan')):.0f} W max, smi {pw.get('smi_sm_mhz_median',float('nan')):.0f} MHz",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--compile-only",action="store_true")
    p.add_argument("--out",type=Path,default=Path("box_out/m16xd/run.json"))
    p.add_argument("--units",type=int,default=48)
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--power-seconds",type=float,default=3.0)
    p.add_argument("--timeout",type=float,default=350)
    p.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    assert args.units>=48 and args.units%2==0 and args.rounds>=6 and 0<args.timeout<=355 and 0<=args.power_seconds<=5
    # A parent process owns the timeout; it can kill a stuck CUDA call or compiler
    # even if the worker holds Python's GIL. No assumptions about shell process groups.
    if not args.dry_run and not args._worker:
        started=time.monotonic()
        cmd=[sys.executable,"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--_worker"]
        child=subprocess.Popen(cmd,start_new_session=(os.name=="posix"))
        try:
            code=child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            if os.name=="posix": os.killpg(child.pid,signal.SIGKILL)
            else: child.kill()
            child.wait(timeout=2)
            args.out.parent.mkdir(parents=True,exist_ok=True)
            try: result=json.loads(args.out.read_text(encoding="utf-8"))  # keep any results written before the kill
            except (OSError,ValueError): result={"task":"M16XD"}
            result.update({"status":"TIMEOUT","elapsed_s":time.monotonic()-started})
            args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
            raise SystemExit(124)
        raise SystemExit(code)
    started=time.monotonic()
    deadline=started+args.timeout-30
    out=args.out.resolve()
    report={"task":"M16XD","status":"RUNNING","pre_registered":PRE,"arms":[f"{a}{n}" for a,n in ARMS]}
    try:
        report["cpu"]=cpu_gates()
        if args.dry_run:
            print(json.dumps(report["cpu"],indent=2)); return
        out.parent.mkdir(parents=True,exist_ok=True)
        # Linux gate re-executes the base prepare functions with real CPU torch too.
        import torch
        torch.set_num_threads(2)
        test=make_weights(1,900)
        raw=test["w13"][0,:64,:512].copy(); sf=test["w13_sf"][0,:64,:64].copy()
        _,blocks=base_pack_cpu(raw,sf)
        ns={"torch":torch,"K2A_BITPOS":BITPOS}
        for name,block in blocks.items(): exec(compile(block,str(BASE)+"::"+name,"exec"),ns)
        a,b=ns["_k2a31_prepack_expert"](torch.from_numpy(raw),torch.from_numpy(sf),False,BITPOS)
        actual=np.concatenate((a.numpy().view(np.uint8).reshape(-1),b.numpy().reshape(-1)))
        assert np.array_equal(actual,pack_tiles(raw[None],sf[None])[0])
        report["cpu"]["real_torch_base_prepare"]="PASS"
        mod,report["compiler"]=build(out,args.compile_only)
        for arm,n in ARMS:
            r=report["compiler"]["per_arm"][f"{arm}{n}"]
            print(f"M16XD build {arm:7s} N={n:2d} regs={r['registers']} consumer_regs={r['consumer_region_regs']} "
                  f"helper_regs={r['helper_region_regs']} spill={r['spill_store_bytes']}/{r['spill_load_bytes']} "
                  f"stack={r['stack_bytes']} HMMA={r['HMMA']} LDS32={r['LDS32']} LDS128={r['LDS128']} STS={r['STS']}",flush=True)
        assert report["compiler"]["status"]=="PASS", report["compiler"]["gates"]
        if args.compile_only: print("M16XD CPU/compile/SASS gates PASS; JSON="+str(out),flush=True)
        else: benchmark(mod,args,report,out,deadline)
        report["status"]="COMPILE_GATES_PASS" if args.compile_only else "COMPLETE"
    except Exception as ex:
        report["status"]="FAIL"; report["error"]=repr(ex)
        raise
    finally:
        report["elapsed_s"]=time.monotonic()-started
        if not args.dry_run:
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()

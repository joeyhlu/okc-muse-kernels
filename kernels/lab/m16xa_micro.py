#!/usr/bin/env python3
""":"
# This same permitted file is also the LF-only Linux compile/SASS/CPU gate script.
set -eu
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
exec python "$0" --compile-only --out="${0%.py}_gates.json" "$@"
":"""
"""M16XA STEP-0: standalone, no layer import, state mutation, or GPU provisioning.

CPU: python kernels/lab/m16xa_micro.py --dry-run
Pod: python3 kernels/lab/m16xa_micro.py --out=box_out/m16xa/run.json
Linux gates: bash tools/local/cudaenv.sh bash kernels/lab/m16xa_micro.py

The bash entry compiles but NEVER launches CUDA work. Python's default entry runs
the explicitly requested benchmark. Generated build/log/JSON artifacts are made
only when the lead invokes that entry. --dry-run writes nothing.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
UNIT = 36864
GRID = 132
ARMS = (("null", 8), ("null", 16), ("null", 32),
        ("rs", 8), ("rs", 16), ("rs", 32),
        ("rs2", 8), ("rs2", 16), ("rs2", 32), ("ms", 8), ("ms", 16))
ARM_ID = {"null": 0, "rs": 1, "rs2": 2, "ms": 3}
# UNP L2, exactly K2A31_BITPOS3, base lines 1648-1649 (not K2A_BITPOS).
BITPOS = ((6, 7, 8, 15), (22, 23, 24, 31), (28, 29, 30, 5), (12, 13, 14, 21),
          (25, 26, 27, 2), (9, 10, 11, 18), (4, 0, 1, 3), (20, 16, 17, 19))

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
__device__ __forceinline__ uint64_t desc(uint32_t a) {
  // wgmma_bench.py:48-49: no-swizzle K-major 8x8 core matrices.
  return uint64_t((a & 0x3ffffu) >> 4) | (uint64_t(8) << 16) | (uint64_t(16) << 32);
}
__device__ __forceinline__ void wf() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wc() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template<int P> __device__ __forceinline__ void ww() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(P) : "memory");
}
template<int N> __device__ __forceinline__ void wr(float*, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t);
__device__ __forceinline__ void ms(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template<int N, int Q> __device__ __forceinline__ void rs_pair(float* d, const uint32_t* h, uint32_t x) {
  // Two k16 steps use disjoint halves of the exact dec_half2 result.
  wf();
  wr<N>(d, h[0], h[4], h[1], h[5], desc(x + (2 * Q) * N * 32));
  wr<N>(d, h[2], h[6], h[3], h[7], desc(x + (2 * Q + 1) * N * 32));
  wc(); ww<1>();
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
template<int N, int MODE> __device__ __forceinline__ void consume(float* d, const uint8_t* st, uint16_t* x, int wp, int lane) {
  // Generated straight-line specializations, no run-time arm/fragment branch.
  @CONSUME@
}
template<int N, int MODE> __global__ __launch_bounds__(384, 1)
void m16xa(const uint8_t* W, const uint16_t* X, float* O, int* smids, int* errors, int units) {
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
        const float4* ep = reinterpret_cast<const float4*>(s + EP<N> + wg * 64 * N * 4);
        for (int u = wg; u < units; u += 2) {
          wait(bars + (6 + wg) * 8, (u / 2) & 1, err);
          float4* dst = reinterpret_cast<float4*>(O + (size_t(blockIdx.x) * units + u) * 64 * N);
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
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 128;" ::: "memory");
    const int wg = warp / 4, wp = warp & 3;
    for (int u = wg; u < units; u += 2) {
      const int slot = u % 3;
      wait(bars + slot * 8, (u / 3) & 1, err);
      if constexpr (MODE != 0) {
        float d[N / 2] = {};
        consume<N, MODE>(d, s + XS + slot * UB, reinterpret_cast<uint16_t*>(s), wp, lane);
        // consume drains all WGMMA before D is touched or its operand bank reused.
        __syncwarp();
        if (lane == 0) arrive(bars + (3 + slot) * 8);
        if (u >= 2) wait(bars + (8 + wg) * 8, ((u / 2) - 1) & 1, err);
        float* ep = reinterpret_cast<float*>(s + EP<N> + wg * 64 * N * 4);
        #pragma unroll
        for (int j = 0; j < N / 2; ++j) {
          const int r = wp * 16 + (lane >> 2) + ((j & 3) >> 1) * 8;
          const int c = (j / 4) * 8 + (lane & 3) * 2 + (j & 1);
          ep[r * N + c] = d[j];
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
    uint32_t sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); smids[blockIdx.x] = sm;
    if (!*err) for (int j = 0; j < 10; ++j)
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bars + j * 8) : "memory");
  }
}
__global__ void empty_kernel() {}
inline void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<int N, int MODE> std::vector<int> config(bool query) {
  ck(cudaFuncSetAttribute(m16xa<N, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM<N>));
  ck(cudaFuncSetAttribute(m16xa<N, MODE>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  if (!query) return {};
  cudaFuncAttributes a; ck(cudaFuncGetAttributes(&a, m16xa<N, MODE>));
  int occ; ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, m16xa<N, MODE>, 384, SMEM<N>));
  return {a.numRegs, int(a.localSizeBytes), int(a.sharedSizeBytes), SMEM<N>, occ};
}
std::vector<int> info(int mode, int n) { @INFO@ throw std::runtime_error("unsupported arm/N"); }
void launch(int mode, int n, int units, int grid, uint64_t w, uint64_t x, uint64_t o, uint64_t sm, uint64_t err, uint64_t stream) {
  if (units < 4 || (units & 1) || grid < 1) throw std::runtime_error("invalid grid/units");
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


def generate_cuda():
    """Single binary; MODE and N resolve at instantiation, never in the stage loop."""
    wg = []
    for n in (8, 16, 32):
        k = n // 2
        outs = ",".join(f'"+f"(d[{i}])' for i in range(k))
        ds = ",".join(f"%{i}" for i in range(k))
        aa = ",".join(f"%{k+i}" for i in range(4))
        wg.append(f'''template<> __device__ __forceinline__ void wr<{n}>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t b) {{
  asm volatile("{{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{ds}}}, {{{aa}}}, %{k+4}, p, 1, 1, 0; }}"
    : {outs} : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b) : "memory");
}}''')
    # Each dec_half2 produces TWO k16 fragments (8 packed BF16 registers).
    # rs: fresh, immutable fragment locals; let ptxas allocate/serialize them.
    # Never overwrite an in-flight A operand merely because its C++ scope ended.
    # The kept-PTX gate below proves that the compiler preserved this constraint.
    # rs2: explicit two-bank reuse; wait<1> retires q-2 before its bank is decoded.
    pieces = []
    for mode in (1, 2, 3):
        pieces += [f"{'if' if mode == 1 else 'else if'} constexpr (MODE == {mode}) {{"]
        if mode != 1: pieces += ["uint32_t h[2][8];" if mode == 2 else "uint32_t h[8];"]
        # Exactly 16 LDS.128 code loads + 8 LDS.32 scale loads per lane/unit.
        # The same scale word serves four k32 half-groups; no extra scale bytes.
        for sg in range(8):
            pieces += ["{", f"const uint32_t scale = reinterpret_cast<const uint32_t*>(st + 32768 + wp * 1024)[{sg} * 32 + lane];"]
            for sub in range(2):
                kb = 2 * sg + sub
                pieces += ["{", f"const uint4 c = reinterpret_cast<const uint4*>(st + wp * 8192)[{kb} * 32 + lane];"]
                for hf in range(2):
                    q = 2 * kb + hf
                    h = f"h{q}" if mode == 1 else (f"h[{q % 2}]" if mode == 2 else "h")
                    if mode == 1: pieces.append(f"uint32_t {h}[8]; // single assignment; no pending-A overwrite")
                    pieces.append(f"dec_half2(c.{('x','y')[hf]}, c.{('z','w')[hf]}, scale, {sub}u, {h}, 16u, 64u, 4u);")
                    if mode == 3:
                        pieces.append(f"ms_pair<N, {q}>(d, {h}, x, lane);")
                    else:
                        pieces.append(f"rs_pair<N, {q}>(d, {h}, sa(x));")
                pieces += ["}"]
            pieces += ["}"]
        if mode != 3: pieces += ["ww<0>();"]
        pieces += ["}"]
    info, dispatch = [], []
    for arm, n in ARMS:
        mode = ARM_ID[arm]
        cond = f"if (mode == {mode} && n == {n})"
        info.append(f"{cond} return config<{n}, {mode}>(true);")
        dispatch.append(f"{cond} {{ m16xa<{n}, {mode}><<<grid,384,SMEM<{n}>,st>>>((const uint8_t*)w,(const uint16_t*)x,(float*)o,(int*)sm,(int*)err,units); ck(cudaGetLastError()); return; }}")
    src = CUDA_HEAD.replace("@CONSUME@", "\n".join(pieces))
    src = src.replace("@INFO@", "\n".join(info)).replace("@DISPATCH@", "\n".join(dispatch))
    anchor = "template<int N, int Q> __device__"
    return src.replace(anchor, DEQUANT + "\n" + "\n".join(wg) + "\n" + anchor, 1)


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
    """BF16 bits [N,1024] -> 64 no-swizzle K-major descriptor tiles."""
    n = x.shape[0]
    assert n in (8, 16, 32) and x.shape == (n, 1024) and x.dtype == np.uint16
    z = x[:, kperm(np.arange(1024))]
    return z.reshape(n // 8, 8, 64, 2, 8).transpose(2, 0, 3, 1, 4).copy().reshape(-1)


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


def cpu_gates():
    src = generate_cuda()
    assert "@" not in src and "__trap" not in src
    assert len(re.findall(r"m16xa<\d+, \d+><<<", src)) == len(ARMS)
    assert len(set(ARMS)) == len(ARMS)
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
    rejected = []
    for key in ("w13", "w13_sf"):
        bad = dict(weights); bad.pop(key)
        try: prepare_bank(bad, 4, 1)
        except KeyError: rejected.append(key)
    assert len(rejected) == 2
    for n in (8, 16, 32):
        x = np.arange(n * 1024, dtype=np.uint16).reshape(n, 1024)
        xp = pack_x(x)
        for step in range(64):
            for col in range(n):
                for k in range(16):
                    idx = step*n*16 + (col//8)*128 + (k//8)*64 + (col%8)*8 + k%8
                    assert xp[idx] == x[col, kperm(step*16+k)]
        coords = [(wp*16+(lane>>2)+((j&3)>>1)*8, (j//4)*8+(lane&3)*2+(j&1))
                  for wp in range(4) for lane in range(32) for j in range(n//2)]
        assert len(set(coords)) == 64*n
    # Event-level interleaving of ring/epilogue protocols, including all final drains.
    protocol_gates()
    # Compiler-log parser and decision boundary checks use synthetic fixtures,
    # never presented as hardware measurements and never written to a result file.
    fixtures=[]
    for arm,n in ARMS:
        fixtures.append(f"ptxas info : Compiling entry function '_Z5m16xaILi{n}ELi{ARM_ID[arm]}EEv' for 'sm_90a'\n"
                        "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
                        "ptxas info : Used 96 registers, 0 bytes smem\n")
    parsed=compiler_report("".join(fixtures)+"ptxas info : (C7514) synthetic parser fixture\n","")
    assert {(r["arm"],r["n"]) for r in parsed["ptxas"]}==set(ARMS)
    assert len(parsed["c75xx_lines"])==1
    safe="""mov.b32 %r1, 0;
    wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%f1}, {%r1,%r2,%r3,%r4}, %rd1, p, 1,1,0;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
    mov.b32 %r1, 1;"""
    assert ptx_a_lifetime(safe)["status"]=="PASS"
    assert ptx_a_lifetime(safe.replace("aligned 0;","aligned 1;"))["status"]=="FAIL"
    def fake_rows(t16,ratio,t8=1.,t32=1.15):
        return [{"arm":"rs2","n":n,"stage_us":t,"ratio_null":ratio}
                for n,t in ((8,t8),(16,t16),(32,t32))]
    assert verdict(fake_rows(1.,1.10))["status"]=="GO-to-stage-B"
    assert verdict(fake_rows(1.15,1.10))["status"]=="WEAK"
    assert verdict(fake_rows(1.150001,1.))["status"]=="KILL"
    assert verdict(fake_rows(1.,1.100001))["status"]=="KILL"
    assert verdict(fake_rows(1.,1.,t32=1.150001))["status"].startswith("HOLD")
    schedule=list(replay_schedule(12))
    assert all(a[2]!=b[2] for a,b in zip(schedule,schedule[1:]))
    assert all([sum(k==key and b==bank for _,k,b in schedule) for bank in range(3)]==[4,4,4] for key in ARMS)
    # Exercise NaN rejection rather than silently accepting malformed E4M3 input.
    for nan in (127, 255):
        try: reencode(np.array([nan], np.uint8))
        except ValueError: pass
        else: raise AssertionError("NaN scale accepted")
    return {"status": "PASS", "verbatim_cuda_functions": list(wanted), "finite_dequant_lanes": cases,
            "base_prepare_functions_executed": list(blocks), "bank_units": 520,
            "weight_layout": {"w13": [2,8192,2048], "w13_sf_raw_e4m3": [2,8192,256]},
            "arm_instances": [f"{a}_n{n}" for a,n in ARMS], "no_layer_state_access": True,
            "protocol_interleavings":120,"verdict_boundaries":"PASS","compiler_parser":"PASS",
            "source_sha256": hashlib.sha256(src.encode()).hexdigest()}


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
        instance = re.search(r"m16xaILi(\d+)ELi(\d+)E",m[1])
        if instance:
            blocks[-1].update({"n":int(instance[1]),"arm":tuple(ARM_ID)[int(instance[2])]})
    funcs = {}
    for chunk in re.split(r"(?=Function\s*:\s*)", sass):
        m = re.match(r"Function\s*:\s*(\S+)", chunk)
        if not m: continue
        counts = {}
        for op in re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)", chunk):
            op = op.split(".")[0]; counts[op] = counts.get(op,0)+1
        funcs[m[1]] = counts
    return {"ptxas": blocks, "c75xx_lines": [l for l in log.splitlines() if re.search(r"C75\d\d",l)],
            "sass_instruction_counts_static_not_dynamic": funcs}


def ptx_a_lifetime(ptx):
    """Reject generic accesses to A registers until their committed group retires.

    The generated k32 body is straight-line. Every unit ends in wait<0>, so its
    back edge has no live asynchronous inputs. PTX is checked before ANY GPU work;
    ptxas's subsequent physical-register serialization is retained in C75xx/SASS.
    """
    pending,current,violations=[],set(),[]
    issued=0
    text=re.sub(r"//[^\n]*","",ptx)
    reg=r"%[A-Za-z_][A-Za-z_0-9]*"
    for inst in text.split(";"):
        if "wgmma.mma_async" in inst:
            groups=re.findall(r"\{([^{}]*)\}",inst[inst.index("wgmma.mma_async"):])
            assert len(groups)>=2,"expected register/shared WGMMA"
            current.update(re.findall(reg,groups[1])); issued+=1
        elif "wgmma.commit_group" in inst:
            pending.append(current); current=set()
        elif "wgmma.wait_group" in inst:
            n=int(re.search(r"wgmma.wait_group[^\s]*\s+(\d+)",inst)[1])
            pending=pending[-n:] if n else []
        else:
            live=set().union(current,*pending)
            overlap=live.intersection(re.findall(reg,inst))
            if overlap: violations.append({"registers":sorted(overlap),"instruction":inst.strip()[-300:]})
    return {"status":"PASS" if not violations and not current and not any(pending) else "FAIL",
            "wgmma_instructions":issued,"violations":violations[:16]}


def build(out):
    from torch.utils.cpp_extension import load_inline
    src = generate_cuda()
    tag = hashlib.sha256((CPP+src).encode()).hexdigest()[:12]
    # Fresh per invocation to ensure ptxas diagnostics are captured even with a prior cache.
    builddir = out.parent / ("build_"+tag+"_"+str(os.getpid()))
    builddir.mkdir(parents=True, exist_ok=False)
    logfile = out.with_suffix(".compile.log")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ.setdefault("MAX_JOBS", "2")
    with capture_fds(logfile):
        mod = load_inline(name="m16xa_"+tag, cpp_sources=CPP, cuda_sources=src, functions=None,
                          extra_cflags=["-O3","-std=c++20","-DNDEBUG"],
                          extra_cuda_cflags=["-O3","-std=c++20","-DNDEBUG","-lineinfo",
                                             "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v,--warn-on-spills",
                                             "--keep", "--keep-dir=."],
                          build_directory=str(builddir), with_cuda=True, verbose=True, no_implicit_headers=True)
    log = logfile.read_text(errors="replace")
    sass = subprocess.run(["cuobjdump","--dump-sass",mod.__file__], capture_output=True, text=True, timeout=35, check=True).stdout
    resources = subprocess.run(["cuobjdump","--dump-resource-usage",mod.__file__], capture_output=True, text=True, timeout=15, check=True).stdout
    out.with_suffix(".sass").write_text(sass, encoding="utf-8")
    out.with_suffix(".resources.txt").write_text(resources, encoding="utf-8")
    result = compiler_report(log,sass)
    ptxfiles=list(builddir.glob("*.ptx"))
    result["ptx_a_lifetime"]={str(p):ptx_a_lifetime(p.read_text()) for p in ptxfiles}
    result.update({"extension":mod.__file__,"compile_log":str(logfile),"resource_dump":resources})
    kernels = [r for r in result["ptxas"] if "m16xa" in r["symbol"]]
    result["gates"] = {
        "eleven_instances":len(kernels)==len(ARMS),
        # lead 2026-11-01: a step-0 timing probe tolerates <= 16 B spill / stack (recorded per arm in "ptxas"); outputs are checked anyway
        "no_spills":bool(kernels) and all(r["spill_store_bytes"]<=16 and r["spill_load_bytes"]<=16 for r in kernels),
        "no_stack":bool(kernels) and all(r["stack_bytes"]<=16 for r in kernels),
        "register_pool_96_to_128":bool(kernels) and all(r["registers"] is not None and 96<=r["registers"]<=128 for r in kernels),
        "real_rs_ms_bulk":("HGMMA" in sass and "HMMA" in sass and ("UTMALDG" in sass or "LDGSTS" in sass or "UBLKCP" in sass)),
        "ptx_no_pending_a_writes":bool(ptxfiles) and all(r["status"]=="PASS" for r in result["ptx_a_lifetime"].values()),
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


def gpu_correctness(mod, torch):
    """Four CTAs x four units, all K stripes, every real specialization, FP32 CPU torch oracle."""
    weights=make_weights(1,31)
    # Include zero, negative, subnormal and maximum finite E4M3, without NaNs.
    valid=np.array([b for b in range(256) if b not in (127,255)],np.uint8)
    weights["w13_sf"][0,:256,:] = np.resize(valid,(256,256))
    bank=prepare_bank(weights,4,4)
    w=torch.from_numpy(bank).cuda()
    err=torch.empty(4,device="cuda",dtype=torch.int32)
    sm=torch.empty_like(err)
    results=[]
    for n in (8,16,32):
        gen=torch.Generator().manual_seed(411+n)
        x=torch.randn(n,1024,generator=gen).to(torch.bfloat16)
        xb=x.view(torch.int16).numpy().view(np.uint16)
        px=torch.from_numpy(pack_x(xb).view(np.int16)).cuda()
        refs=[]
        for expert,row,stripe in zip(*unit_coords(4,4)):
            raw=torch.from_numpy(weights["w13"][expert,row*64:(row+1)*64,stripe*512:(stripe+1)*512].copy())
            sf=torch.from_numpy(weights["w13_sf"][expert,row*64:(row+1)*64,stripe*64:(stripe+1)*64].copy()).view(torch.float8_e4m3fn).float()
            c=torch.stack((raw&15,raw>>4),dim=-1).reshape(64,64,16).long()
            lut=torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float32)
            dw=lut[c&7]*torch.where((c&8)!=0,-1.,1.)*sf[:,:,None]
            refs.append(dw.reshape(64,1024).float() @ x.float().T)
        reference=torch.stack(refs)
        for arm,an in ARMS:
            if an != n or arm=="null": continue
            out=torch.full((16,64,n),float("nan"),device="cuda",dtype=torch.float32)
            def call(): mod.launch(ARM_ID[arm],n,4,4,w.data_ptr(),px.data_ptr(),out.data_ptr(),sm.data_ptr(),err.data_ptr(),torch.cuda.current_stream().cuda_stream)
            call(); torch.cuda.synchronize()
            assert not err.any().item(), "device protocol timeout in correctness"
            got=out.cpu()
            # FP32 reduction order differs; bound by FP32 sum |W*x|, not a loose BF16 tolerance.
            scale=torch.stack([r.abs().amax().clamp_min(1) for r in refs])[:,None,None]
            delta=(got-reference).abs()
            assert torch.isfinite(got).all() and torch.all(delta <= 2e-5*scale), (arm,n,delta.max().item())
            # Replays use fresh inputs and poisoned outputs to catch cached/stale data.
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): call()
            torch.cuda.synchronize()
            for sign in (-1,1,-1):
                px.copy_(torch.from_numpy(pack_x((sign*x).view(torch.int16).numpy().view(np.uint16)).view(np.int16)).cuda())
                out.fill_(float("nan")); torch.cuda.synchronize(); graph.replay(); torch.cuda.synchronize()
                assert not err.any().item()
                assert torch.all((out.cpu()-sign*reference).abs()<=2e-5*scale)
            results.append({"arm":arm,"n":n,"max_abs_error":delta.max().item(),"status":"PASS"})
    return results


def verdict(rows):
    best={n:min((r for r in rows if r["n"]==n and r["arm"]!="null"),key=lambda r:r["stage_us"]) for n in (8,16,32)}
    n16=best[16]
    if n16["stage_us"]>1.15 or n16["ratio_null"]>1.10: status="KILL"
    elif best[8]["stage_us"]<=1.00 and n16["stage_us"]<=1.00 and best[32]["stage_us"]<=1.15: status="GO-to-stage-B"
    elif 1.00<n16["stage_us"]<=1.15: status="WEAK"
    else: status="HOLD: cross-N GO gate failed"
    return {"status":status,"best":{str(n):{"arm":r["arm"],"stage_us":r["stage_us"],"ratio_null":r["ratio_null"]} for n,r in best.items()}}


def replay_schedule(rounds):
    for rnd in range(rounds):
        order=ARMS[rnd%len(ARMS):]+ARMS[:rnd%len(ARMS)]
        for key in order:
            # Same N has the same bank for paired null ratios; every consecutive
            # timed replay changes banks, including round boundaries.
            yield rnd,key,(2*rnd+ARMS.index(key))%3


def benchmark(mod,args,report):
    import torch
    prop=torch.cuda.get_device_properties(0)
    assert "H200" in prop.name and prop.multi_processor_count==132 and prop.major==9, str(prop)
    report["device"]={"name":prop.name,"sms":prop.multi_processor_count}
    attrs={}
    for arm,n in ARMS:
        reg,local,static,dynamic,occ=mod.info(ARM_ID[arm],n)
        assert 96<=reg<=128 and local<=16 and occ==1, (arm,n,reg,local,occ)  # lead: <= 16 B local tolerated in the probe
        attrs[f"{arm}{n}"]={"regs":reg,"local_bytes":local,"static_smem":static,"dynamic_smem":dynamic,"ctas_per_sm":occ}
    report["kernel_attributes"]=attrs
    report["correctness"]=gpu_correctness(mod,torch)
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
    for n in (8,16,32):
        gen=torch.Generator().manual_seed(103+n)
        x=torch.randn(n,1024,generator=gen).to(torch.bfloat16)
        inputs[n]=torch.from_numpy(pack_x(x.view(torch.int16).numpy().view(np.uint16)).view(np.int16)).cuda()
    output=torch.empty((GRID,args.units,64,32),device="cuda",dtype=torch.float32)
    sm=torch.empty(GRID,device="cuda",dtype=torch.int32)
    err=torch.empty_like(sm)
    graphs={}
    for arm,n in ARMS:
        graphs[arm,n]=[timer.capture(lambda arm=arm,n=n,w=w:mod.launch(ARM_ID[arm],n,args.units,GRID,w.data_ptr(),inputs[n].data_ptr(),output.data_ptr(),sm.data_ptr(),err.data_ptr(),stream)) for w in banks]
        assert not err.any().item(), (arm,n,"protocol error")
        assert len(set(sm.cpu().tolist()))==GRID, (arm,n,"grid not one CTA per SM")
    report["clocks_before"]=clocks()
    before=statistics.median(timer.replay(empty) for _ in range(15))
    samples={k:[] for k in ARMS}
    # Matched N arms share each round's bank; rotate both bank and arm position.
    for rnd,key,bank in replay_schedule(args.rounds):
        samples[key].append(timer.replay(graphs[key][bank]))
        assert not err.any().item(), (key,"device protocol error")
        assert len(set(sm.cpu().tolist()))==GRID, (key,"SM coverage")
    after=statistics.median(timer.replay(empty) for _ in range(15))
    report["clocks_after"]=clocks()
    overhead=(before+after)/2
    report["empty_kernel_us"]={"before":before,"after":after,"subtracted":overhead}
    rows=[]
    for arm,n in ARMS:
        values=[(v-overhead)/args.units for v in samples[arm,n]]
        assert min(values)>0, "invalid empty-kernel subtraction"
        stage=statistics.median(values)
        null=statistics.median((v-overhead)/args.units for v in samples["null",n])
        paired=[(v-overhead)/(z-overhead) for v,z in zip(samples[arm,n],samples["null",n])]
        row={"arm":arm,"n":n,"stage_us":stage,"ratio_null":stage/null,"paired_ratio_null":statistics.median(paired),
             "weight_TBps":GRID*UNIT/(stage*1e6),"samples_total_us":samples[arm,n],
             "stage_p10_us":float(np.percentile(values,10)),"stage_p90_us":float(np.percentile(values,90)),
             "banks_per_round":[(2*r+ARMS.index((arm,n)))%3 for r in range(args.rounds)],
             "units_per_cta":args.units,"measurement":"MEASURED CUDA events; (kernel - empty)/units"}
        rows.append(row)
        print(f"M16XA {arm:4s} N={n:2d} {stage:.4f} us/stage {row['weight_TBps']:.3f} TB/s null={row['ratio_null']:.3f}x",flush=True)
    report["results"]=rows
    report["verdict"]=verdict(rows)
    print("M16XA "+report["verdict"]["status"],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--compile-only",action="store_true")
    p.add_argument("--out",type=Path,default=Path("box_out/m16xa/run.json"))
    p.add_argument("--units",type=int,default=48)
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--timeout",type=float,default=350)
    p.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    assert args.units>=48 and args.units%2==0 and args.rounds>=6 and 0<args.timeout<=355
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
            result={"task":"M16XA","status":"TIMEOUT","elapsed_s":time.monotonic()-started}
            args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
            raise SystemExit(124)
        raise SystemExit(code)
    started=time.monotonic()
    out=args.out.resolve()
    report={"task":"M16XA","status":"RUNNING","pre_registered":{"kill_n16_us_gt":1.15,"kill_n16_null_ratio_gt":1.10,"weak_n16_us":[1.00,1.15],"go_n8_n16_us_le":1.00,"go_n32_us_le":1.15}}
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
        mod,report["compiler"]=build(out)
        assert report["compiler"]["status"]=="PASS", report["compiler"]["gates"]
        if args.compile_only: print("M16XA CPU/compile/PTX/SASS gates PASS; JSON="+str(out),flush=True)
        else: benchmark(mod,args,report)
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

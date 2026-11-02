#!/usr/bin/env python3
""":"
# This same permitted file is also the LF-only Linux compile/SASS/CPU gate script.
set -eu
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
exec python "$0" --compile-only --out="${0%.py}_gates.json" "$@"
":"""
"""F3 attached-ring diagnostic: standalone, no layer import, state mutation, or GPU provisioning.

CPU: python -B kernels/lab/m16xb_micro.py --dry-run
Pod: python3 kernels/lab/m16xb_micro.py --out=box_out/f3/results.json
Linux gates: bash tools/local/cudaenv.sh bash tools/local/f3_gates.sh

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
# Retain the original 4096-byte barrier reservation so NS=3 is unchanged.
SMEM_CAP = 227 * 1024

def smem_bytes(n, ns):
    return 65536 + ns * UNIT + 2 * 64 * n * 4 + 4096

REQUESTED_ARMS = tuple((a, n, ns) for a, n in
                       (("null", 8), ("null", 16), ("null", 32),
                        ("ms", 8), ("ms", 16), ("rs", 32)) for ns in (2, 3, 4))
DROPPED_ARMS = tuple(k for k in REQUESTED_ARMS if smem_bytes(k[1], k[2]) > SMEM_CAP)
ARMS = tuple(k for k in REQUESTED_ARMS if k not in DROPPED_ARMS) + (("ms_twin", 16, 3),)

def arm_name(key):
    a, n, ns = key
    return f"{a}_n{n}_ns{ns}"

ARM_ID = {"null": 0, "rs": 1, "rs2": 2, "ms": 3, "ms_twin": 4}
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
constexpr int UB = 36864, XS = 65536;
template<int N, int NS> constexpr int EP = XS + NS * UB;
template<int N, int NS> constexpr int BO = EP<N, NS> + 2 * 64 * N * 4;
template<int N, int NS> constexpr int SMEM = BO<N, NS> + 4096;
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
template<int N, int MODE, int NS> __global__ __launch_bounds__(384, 1)
void m16xb(const uint8_t* W, const uint16_t* X, float* O, int* smids, int* errors, uint64_t* marks, int units) {
  extern __shared__ __align__(128) uint8_t s[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = sa(s + BO<N, NS>);
  int* err = errors + blockIdx.x;
  // All threads initialize the resident stripe before register ownership changes.
  for (int i = tid * 8; i < N * 1024; i += 384 * 8)
    *reinterpret_cast<uint4*>(s + i * 2) = *reinterpret_cast<const uint4*>(X + i);
  if (tid == 0) {
    *err = 0;
    for (int j = 0; j < NS; ++j) { init(bars + j * 8, 1); init(bars + (NS + j) * 8, 4); }
    for (int j = 0; j < 2; ++j) { init(bars + (2 * NS + j) * 8, 4); init(bars + (2 * NS + 2 + j) * 8, 1); }
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
        const int slot = u % NS;
        if (u >= NS) wait(bars + (NS + slot) * 8, ((u / NS) - 1) & 1, err);
        bulk(sa(s + XS + slot * UB), W + (size_t(blockIdx.x) * units + u) * UB, bars + slot * 8, policy);
      }
      // Drain EVERY outstanding slot, including the final NS slots (no abandoned TMA).
      for (int slot = 0; slot < NS; ++slot) {
        const int last = units - 1 - ((units - 1 - slot) % NS);
        if (last >= 0) wait(bars + (NS + slot) * 8, (last / NS) & 1, err);
      }
    }
    if constexpr (MODE != 0) {
      if (warp == 9 || warp == 10) {
        const int wg = warp - 9;
        const float4* ep = reinterpret_cast<const float4*>(s + EP<N, NS> + wg * 64 * N * 4);
        for (int u = wg; u < units; u += 2) {
          wait(bars + (2 * NS + wg) * 8, (u / 2) & 1, err);
          float4* dst = reinterpret_cast<float4*>(O + (size_t(blockIdx.x) * units + u) * 64 * N);
          for (int j = lane; j < 64 * N / 4; j += 32) {
            float4 v = ep[j];
            v.x *= 256.f; v.y *= 256.f; v.z *= 256.f; v.w *= 256.f;
            dst[j] = v;
          }
          __syncwarp();
          if (lane == 0) arrive(bars + (2 * NS + 2 + wg) * 8);
        }
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 128;" ::: "memory");
    const int wg = warp / 4, wp = warp & 3;
    // Diagnostic only, outside the ring protocol; no extra arrivals or barriers.
    // CTA0/thread0 measures its entire even-unit loop, including exposed waits.
    uint64_t t0 = 0, c0 = 0;
    if (blockIdx.x == 0 && tid == 0) { t0 = gt(); c0 = clock64(); }
    for (int u = wg; u < units; u += 2) {
      const int slot = u % NS;
      // NS=3 alternates WG owner; previous-empty certifies retirement before
      // a future-full parity can look ready while the previous TMA is in flight.
      // NS=2/4 always reuse a slot in the SAME WG (u, u+NS have equal parity).
      // Keep the same guard: all four warps must retire before refill, and a
      // faster warp must not run ahead of its three peers. Phases count SLOT
      // generations u/NS, not WG iterations u/2. This covers odd/even NS alike.
      if (u >= NS) wait(bars + (NS + slot) * 8, ((u / NS) - 1) & 1, err);
      wait(bars + slot * 8, (u / NS) & 1, err);
      if constexpr (MODE != 0) {
        float d[N / 2] = {};
        consume<N, (MODE == 4 ? 3 : MODE)>(d, s + XS + slot * UB, reinterpret_cast<uint16_t*>(s), wp, lane);
        // consume drains all WGMMA before D is touched or its operand bank reused.
        __syncwarp();
        if (lane == 0) arrive(bars + (NS + slot) * 8);
        if (u >= 2) wait(bars + (2 * NS + 2 + wg) * 8, ((u / 2) - 1) & 1, err);
        float* ep = reinterpret_cast<float*>(s + EP<N, NS> + wg * 64 * N * 4);
        #pragma unroll
        for (int j = 0; j < N / 2; ++j) {
          const int r = wp * 16 + (lane >> 2) + ((j & 3) >> 1) * 8;
          const int c = (j / 4) * 8 + (lane & 3) * 2 + (j & 1);
          ep[r * N + c] = d[j];
        }
        __syncwarp();
        if (lane == 0) arrive(bars + (2 * NS + wg) * 8);
      } else {
        // Null is ONLY ring + waits, plus identical startup/allocation and terminal bookkeeping.
        __syncwarp();
        if (lane == 0) arrive(bars + (NS + slot) * 8);
      }
    }
    if (blockIdx.x == 0 && tid == 0) {
      const uint64_t c1 = clock64(), t1 = gt();
      marks[0] = c1 - c0; marks[1] = t1 - t0;
    }
  }
  // Includes all 384 threads, even producer warp's unused lanes and helper warp 11.
  __syncthreads();
  if (tid == 0) {
    uint32_t sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); smids[blockIdx.x] = sm;
    if (!*err) for (int j = 0; j < 2 * NS + 4; ++j)
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bars + j * 8) : "memory");
  }
}
__global__ void empty_kernel() {}
inline void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<int N, int MODE, int NS> std::vector<int> config(bool query) {
  static_assert(NS >= 2 && NS <= 4 && SMEM<N, NS> <= 227 * 1024);
  ck(cudaFuncSetAttribute(m16xb<N, MODE, NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM<N, NS>));
  ck(cudaFuncSetAttribute(m16xb<N, MODE, NS>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  if (!query) return {};
  cudaFuncAttributes a; ck(cudaFuncGetAttributes(&a, m16xb<N, MODE, NS>));
  int occ; ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, m16xb<N, MODE, NS>, 384, SMEM<N, NS>));
  return {a.numRegs, int(a.localSizeBytes), int(a.sharedSizeBytes), SMEM<N, NS>, occ};
}
std::vector<int> info(int mode, int n, int ns) { @INFO@ throw std::runtime_error("unsupported arm/N"); }
void launch(int mode, int n, int ns, int units, int grid, uint64_t w, uint64_t x, uint64_t o, uint64_t sm, uint64_t err, uint64_t marks, uint64_t stream) {
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
std::vector<int> info(int, int, int);
void launch(int, int, int, int, int, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
void launch_empty(uint64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("info", &info); m.def("launch", &launch); m.def("empty", &launch_empty);
}
"""


def generate_cuda():
    """Single binary; MODE, N and NS resolve at instantiation, never in the stage loop."""
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
    for arm, n, ns in ARMS:
        mode = ARM_ID[arm]
        cond = f"if (mode == {mode} && n == {n} && ns == {ns})"
        info.append(f"{cond} return config<{n}, {mode}, {ns}>(true);")
        dispatch.append(f"{cond} {{ m16xb<{n}, {mode}, {ns}><<<grid,384,SMEM<{n}, {ns}>,st>>>((const uint8_t*)w,(const uint16_t*)x,(float*)o,(int*)sm,(int*)err,(uint64_t*)marks,units); ck(cudaGetLastError()); return; }}")
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


def address_bounds(arm, n, units, grid, capacities=None, ns=3):
    """Replay every global access site in bytes; no tensors/GPU needed.

    Intervals enumerate every bulk transfer, X uint4 and helper float4. Repeated
    error polls share the same per-CTA int32 address. Capacities may come from the
    actual tensors, so a short allocation cannot pass by checking only a formula.
    """
    assert (arm, n, ns) in ARMS and units >= 4 and units % 2 == 0 and grid >= 1
    assert max(units + 2, grid) < 2**31, "C++ int loop/launch overflow"
    sizes = {"W": grid * units * UNIT, "X": n * 1024 * 2,
             "O": grid * units * 64 * 32 * 4, "sm": grid * 4, "err": grid * 4, "marks": 16}
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
        seen = []
        for wg in (0, 1):
            owned = tiles[:, wg::2].reshape(-1)
            starts = owned[:, None] * (64 * n * 4) + js[None, :] * 16
            seen.append(starts.reshape(-1))
        check("O", np.concatenate(seen), 16, 16)
    else:
        accesses["O"] = {"accesses": 0, "tensor_bytes": int(sizes["O"])}
    check("marks", np.array([0, 8]), 8, 8)
    check("sm", cta * 4, 4, 4)  # thread 0 terminal store
    check("err", cta * 4, 4, 4)  # thread 0 reset/read + all waiters' atomic polls
    # prepare_bank's actual raw tensor slices, including cohort/expert transitions.
    es, rows, ks = unit_coords(units, grid)
    experts = ((grid + 3) // 4 * units + 127) // 128
    assert np.all((es >= 0) & (es < experts))
    assert np.all((rows * 64 >= 0) & ((rows + 1) * 64 <= 8192))
    assert np.all((ks * 512 >= 0) & ((ks + 1) * 512 <= 2048))
    assert np.all((ks * 64 >= 0) & ((ks + 1) * 64 <= 256))
    assert len(set(zip(es.tolist(), rows.tolist(), ks.tolist()))) == grid * units
    return {"arm": arm, "n": n, "ns": ns, "grid": grid, "units": units, "status": "PASS",
            "experts": experts, "accesses": accesses}


def tensor_bounds(arm, n, units, grid, w, x, out, sm, err, marks, ns=3):
    """Run the same CPU arithmetic against the actual launch allocations."""
    tensors = {"W": w, "X": x, "O": out, "sm": sm, "err": err, "marks": marks}
    for name, tensor in tensors.items():
        assert tensor.is_contiguous(), (name, "non-contiguous launch tensor")
        assert tensor.data_ptr() % (4 if name in ("sm", "err") else 16) == 0
    return address_bounds(arm, n, units, grid,
                          {k: t.numel() * t.element_size() for k, t in tensors.items()}, ns=ns)


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
    assert len(re.findall(r"m16xb<\d+, \d+, \d+><<<", src)) == len(ARMS)
    assert len(set(ARMS)) == len(ARMS)
    # Tie the address/protocol replay to the generated CUDA access sites.
    for expression in (
        "constexpr int UB = 36864, XS = 65536;",
        "W + (size_t(blockIdx.x) * units + u) * UB",
        "O + (size_t(blockIdx.x) * units + u) * 64 * N",
        "for (int i = tid * 8; i < N * 1024; i += 384 * 8)",
        "reinterpret_cast<const uint4*>(X + i)",
        "for (int j = lane; j < 64 * N / 4; j += 32)",
        "int* err = errors + blockIdx.x;", "smids[blockIdx.x] = sm;",
    ):
        assert expression in src, ("update address replay", expression)
    handoff = "if (u >= NS) wait(bars + (NS + slot) * 8, ((u / NS) - 1) & 1, err);"
    assert src.count(handoff) == 2  # Producer and consumer, not producer alone.
    assert handoff + "\n      wait(bars + slot * 8, (u / NS) & 1, err);" in src
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
    # Preserve original pure packing/gate helpers verbatim (except required arm/NS axes).
    original=ROOT/"kernels/lab/m16xa_micro.py"
    names=("pack_tiles","prepare_bank","unit_coords","pack_x","base_pack_cpu","ptx_a_lifetime")
    old_blocks=selected_python(original,names)
    new_blocks=selected_python(Path(__file__),names)
    assert old_blocks==new_blocks, "unexpected change to original packing/reference/PTX gate"
    # Consumer straight-line code and bulk primitive are copied, not redesigned.
    def source_block(path,start,end):
        lines=[]; active=False
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.startswith(start): active=True
                if active and line.startswith(end): break
                if active: lines.append(line)
        assert lines
        return "".join(lines)
    for start,end in (("__device__ __forceinline__ void bulk(","__device__ __forceinline__ uint64_t desc("),
                      ("template<int N, int Q> __device__", "template<int N, int MODE> __device__")):
        assert source_block(original,start,end)==source_block(Path(__file__),start,end)
    # Event-level interleaving of ring/epilogue protocols, including all final drains.
    protocol_reports = {str(ns): protocol_gates(ns) for ns in (2,3,4)}
    phase_report = {str(ns): ring_phase_gates(ns) for ns in (2,3,4)}
    reset_report = input_reset_gates()
    # Compiler-log parser and decision boundary checks use synthetic fixtures,
    # never presented as hardware measurements and never written to a result file.
    fixtures=[]
    for arm,n,ns in ARMS:
        fixtures.append(f"ptxas info : Compiling entry function '_Z6m16xbILi{n}ELi{ARM_ID[arm]}ELi{ns}EEv' for 'sm_90a'\n"
                        "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
                        "ptxas info : Used 96 registers, 0 bytes smem\n")
    parsed=compiler_report("".join(fixtures)+"ptxas info : (C7514) synthetic parser fixture\n","")
    assert {(r["arm"],r["n"],r["ns"]) for r in parsed["ptxas"]}==set(ARMS)
    assert len(parsed["c75xx_lines"])==1
    safe="""mov.b32 %r1, 0;
    wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%f1}, {%r1,%r2,%r3,%r4}, %rd1, p, 1,1,0;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
    mov.b32 %r1, 1;"""
    assert ptx_a_lifetime(safe)["status"]=="PASS"
    assert ptx_a_lifetime(safe.replace("aligned 0;","aligned 1;"))["status"]=="FAIL"
    decision_gates()
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
            "arm_instances": [arm_name(k) for k in ARMS], "no_layer_state_access": True,
            "preserved_original_helpers":list(names), "protocol_interleavings":protocol_reports,"verdict_boundaries":"PASS","compiler_parser":"PASS",
             "full_grid_prepare":{"units":6336,"bank_bytes":full_bank_bytes,"status":"PASS"},
             "global_address_bounds":bound_rows,"ring_phase_replay":phase_report,
             "correctness_input_reset":reset_report,
            "ring_layout":layout_gates(), "hot_loop_accounting":hot_loop_accounting(),
            "source_sha256": hashlib.sha256(src.encode()).hexdigest()}


def address_gates():
    class Tensor:
        def __init__(self, count, width): self.count, self.width = count, width
        def numel(self): return self.count
        def element_size(self): return self.width
        def is_contiguous(self): return True
        def data_ptr(self): return 0x100000000  # Exercise an address above int32.
    rows=[]
    for arm,n,ns in ARMS:
        rows.append(tensor_bounds(arm,n,48,GRID,Tensor(GRID*48*UNIT,1),Tensor(n*1024,2),
                                 Tensor(GRID*48*64*32,4),Tensor(GRID,4),Tensor(GRID,4),Tensor(2,8),ns=ns))
    # Every global buffer gets a one-byte-short negative control at its last access.
    last=next(r for r in rows if r["arm"]=="rs" and r["n"]==32)
    sizes={k:v["tensor_bytes"] for k,v in last["accesses"].items()}
    for key in sizes:
        bad={**sizes,key:last["accesses"][key]["end_byte_exclusive"]-1}
        try: address_bounds("rs",32,48,GRID,bad)
        except AssertionError as ex:
            assert ex.args[0][:3] == ("rs",32,key), (key, ex)
        else: raise AssertionError(("short tensor accepted",key))
    return rows


def input_reset_gates():
    class Buffer:
        def __init__(self, x): self.value=x.copy()
        def copy_(self, x): self.value[:]=x
    cases=0
    for n in (8,16,32):
        original=pack_x(np.arange(n*1024,dtype=np.uint16).reshape(n,1024))
        px=Buffer(original)
        for arm,an,ns in ARMS:
            if an!=n: continue
            # Replay -X from a previous arm, including its early-failure path.
            px.value ^= np.uint16(0x8000)
            assert not np.array_equal(px.value,original), "negative control did not mutate X"
            reset_correctness_input(px,original)
            assert np.array_equal(px.value,original), (arm,n,"inherited -X")
            cases+=1
    return {"status":"PASS","arm_resets":cases,"legacy_negated_input_rejected":True}


def ring_phase_gates(ns):
    """Replay actual parity tests with out-of-order TMA and eight independent warps.

    The older slot==unit simulator assumes away future-phase waits. In an odd
    ring a slot changes WG owner on reuse, so that assumption needs a handshake.
    """
    def run(units,seed,fixed,prefix=()):
        rng=np.random.default_rng(seed)
        full,empty=[0]*ns,[0]*ns  # Number of completed generations, parity = count & 1.
        inflight=[None]*ns
        arrivals=[set() for _ in range(ns)]
        next_u=[w//4 for w in range(8)]
        stage=[1]*8  # 0 previous-empty wait, 1 full wait, 2 consume/release.
        producer=0
        for tick in range(units*50):
            ops=[]
            if producer<units:
                slot=producer%ns
                if producer<ns or (empty[slot]&1)!=((producer//ns-1)&1): ops.append(("p",0))
            ops += [("t",s) for s in range(ns) if inflight[s] is not None]
            for w,u in enumerate(next_u):
                if u>=units: continue
                s,g=u%ns,u//ns
                if stage[w]==0 and (empty[s]&1)!=((g-1)&1): ops.append(("e",w))
                if stage[w]==1 and (full[s]&1)!=(g&1): ops.append(("f",w))
                if stage[w]==2: ops.append(("r",w))
            if not ops: break
            op,k=prefix[tick] if tick<len(prefix) else ops[int(rng.integers(len(ops)))]
            assert (op,k) in ops, ("invalid replay prefix",op,k)
            if op=="p":
                s,g=producer%ns,producer//ns
                assert inflight[s] is None and full[s]==g and empty[s]==g, "overlapping refill"
                inflight[s]=producer; producer+=1
            elif op=="t":
                assert full[k]==inflight[k]//ns
                full[k]+=1; inflight[k]=None
            else:
                u=next_u[k]; s,g=u%ns,u//ns
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
                    stage[k]=0 if fixed and next_u[k]>=ns else 1
        assert producer==units and all(u>=units for u in next_u), "deadlock"
        assert inflight==[None]*ns and not any(arrivals)
        for s in range(ns):
            uses=len(range(s,units,ns))
            assert full[s]==empty[s]==uses, "terminal drain incomplete"
    if ns == 3:
        # Issue 0/1/2, finish only 1, let one WG1 warp consume 1 then test future 3.
        witness=(("p",0),("p",0),("p",0),("t",1),("f",4),("r",4),("f",4))
        try: run(48,0,False,witness)
        except AssertionError as ex: assert "future full phase" in str(ex)
        else: raise AssertionError("legacy parity race was not reproduced")
        # Fixed code blocks that last operation on empty(0); then all schedules drain.
        run(48,0,True,witness[:-1])
    for units in (4,6,8,48,50):
        for seed in range(24): run(units,seed,True)
    return {"status":"PASS","warp_interleavings":120 + (ns == 3),"legacy_future_phase_rejected":ns == 3,
            "witness":"issue 0,1,2; complete 1; WG1 consumes 1; future full(3) looked ready"}


def protocol_gates(ns):
    for units in (4, 6, 8, 48, 50):
        for null in (False, True):
            for seed in range(12):
                rng = np.random.default_rng(seed)
                slots, epi = [None]*ns, [None]*2
                next_c, next_h, prod = [0,1], [0,1], 0
                consumed, drained = set(), set()
                for _ in range(units * 20):
                    ops = []
                    if prod < units and slots[prod % ns] is None: ops.append((0,0))
                    for wg in range(2):
                        u = next_c[wg]
                        if u < units and slots[u % ns] == u and (null or epi[wg] is None): ops.append((1,wg))
                        if not null and epi[wg] == next_h[wg]: ops.append((2,wg))
                    if not ops: break
                    op,wg = ops[int(rng.integers(len(ops)))]
                    if op == 0: slots[prod % ns] = prod; prod += 1
                    elif op == 1:
                        u=next_c[wg]; consumed.add(u); slots[u%ns]=None; next_c[wg]+=2
                        if not null: epi[wg]=u
                    else: drained.add(epi[wg]); epi[wg]=None; next_h[wg]+=2
                assert consumed == set(range(units)) and slots == [None]*ns
                assert null or (drained == consumed and epi == [None]*2)
                for slot in range(ns):
                    uses = list(range(slot,units,ns))
                    assert ((uses[-1] // ns) & 1) == ((len(uses)-1)&1)

    return {"status":"PASS", "interleavings":120, "ns":ns}


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
        instance = re.search(r"m16xbILi(\d+)ELi(\d+)ELi(\d+)E",m[1])
        if instance:
            blocks[-1].update({"n":int(instance[1]),"arm":tuple(ARM_ID)[int(instance[2])],"ns":int(instance[3])})
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
    os.environ["MAX_JOBS"] = "2"
    with capture_fds(logfile):
        mod = load_inline(name="m16xb_"+tag, cpp_sources=CPP, cuda_sources=src, functions=None,
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
    kernels = [r for r in result["ptxas"] if "m16xb" in r["symbol"]]
    counts=result["sass_instruction_counts_static_not_dynamic"]
    for r in kernels:
        r["dynamic_smem_bytes"]=smem_bytes(r["n"],r["ns"])
        r["sass_static_instruction_counts"]=counts.get(r["symbol"],{})
    keyed={(r["arm"],r["n"],r["ns"]):r for r in kernels}
    twin_keys=(("ms",16,3),("ms_twin",16,3))
    twin_ok=all(k in keyed for k in twin_keys)
    if twin_ok:
        a,b=(keyed[k] for k in twin_keys)
        twin_ok=all(a[f]==b[f] for f in ("registers","stack_bytes","spill_store_bytes",
                       "spill_load_bytes","static_smem_bytes","sass_static_instruction_counts"))
    def sass_arm_ok(r):
        ops=r["sass_static_instruction_counts"]
        bulk=any(k in ops for k in ("UTMALDG","LDGSTS","UBLKCP"))
        math_ok=("HMMA" in ops) if r["arm"].startswith("ms") else (("HGMMA" in ops) if r["arm"]=="rs" else True)
        return bool(ops) and bulk and math_ok
    result["gates"] = {
        "all_instances":{(r["arm"],r["n"],r["ns"]) for r in kernels}==set(ARMS),
        "per_arm_sass":bool(kernels) and all(sass_arm_ok(r) for r in kernels),
        "smem_cap":bool(kernels) and all(r["static_smem_bytes"]+r["dynamic_smem_bytes"]<=SMEM_CAP for r in kernels),
        "aa_identical_resources_and_static_instruction_counts":twin_ok,
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


def reset_correctness_input(px, original):
    # A preceding arm's last graph replay deliberately leaves -X in px.
    px.copy_(original)


def launch_arm(mod, key, units, grid, w, x, out, sm, err, marks, stream):
    arm, n, ns = key
    assert key in ARMS
    mod.launch(ARM_ID[arm], n, ns, units, grid, w.data_ptr(), x.data_ptr(),
               out.data_ptr(), sm.data_ptr(), err.data_ptr(), marks.data_ptr(), stream)


def gpu_correctness(mod, torch, results=None):
    """All compiled arms: original FP32 oracle, changed-X poisoned graph replays."""
    weights=make_weights(1,31)
    valid=np.array([b for b in range(256) if b not in (127,255)],np.uint8)
    weights["w13_sf"][0,:256,:] = np.resize(valid,(256,256))
    bank=prepare_bank(weights,4,4)
    w=torch.from_numpy(bank).cuda()
    err=torch.empty(4,device="cuda",dtype=torch.int32)
    sm=torch.empty_like(err)
    marks=torch.empty(2,device="cuda",dtype=torch.int64)
    if results is None: results=[]
    for n in (8,16,32):
        gen=torch.Generator().manual_seed(411+n)
        x=torch.randn(n,1024,generator=gen).to(torch.bfloat16)
        xb=x.view(torch.int16).numpy().view(np.uint16)
        original=torch.from_numpy(pack_x(xb).view(np.int16)).cuda()
        px=torch.empty_like(original)
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
        for key in ARMS:
            arm,an,ns=key
            if an != n: continue
            row={"arm":arm,"n":n,"ns":ns,"status":"FAIL"}
            results.append(row)
            reset_correctness_input(px,original)
            out=torch.full((16,64,n),float("nan"),device="cuda",dtype=torch.float32)
            tensor_bounds(arm,n,4,4,w,px,out,sm,err,marks,ns=ns)
            def call():
                launch_arm(mod,key,4,4,w,px,out,sm,err,marks,torch.cuda.current_stream().cuda_stream)
            call(); torch.cuda.synchronize()
            if err.any().item():
                row["why"]="device protocol timeout"
                raise RuntimeError((key,row["why"]))
            got=out.cpu()
            def agrees(value, sign):
                if arm=="null": return bool(torch.isnan(value).all())
                return bool(torch.isfinite(value).all() and torch.all((value-sign*reference).abs()<=2e-5*scale))
            if not agrees(got,1):
                row["why"]="initial FP32 reference / untouched null mismatch"
                continue
            if arm!="null": row["max_abs_error"]=float((got-reference).abs().max().item())
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): call()
            torch.cuda.synchronize()
            # Include NULL in changed-input replay checks: it must preserve poison.
            for sign in (-1,1,-1):
                px.copy_(torch.from_numpy(pack_x((sign*x).view(torch.int16).numpy().view(np.uint16)).view(np.int16)).cuda())
                out.fill_(float("nan")); torch.cuda.synchronize(); graph.replay(); torch.cuda.synchronize()
                if err.any().item(): raise RuntimeError((key,"graph protocol timeout"))
                ok_r=agrees(out.cpu(),sign)
                if not ok_r: break
            row.update({"status":"PASS" if ok_r else "FAIL", "changed_input_replays":3,
                        "check":"null untouched" if arm=="null" else "torch FP32 reference"})
    return results


def verdict(rows):
    by={(r["arm"],r["n"],r["ns"]):r for r in rows}
    if set(by)!=set(ARMS) or any(not r.get("correct",True) for r in rows):
        return {"status":"VOID: missing/incorrect arm"}
    best={n:min((by["ms",n,ns] for ns in (2,3,4)), key=lambda r:r["stage_us"]) for n in (8,16)}
    ship,twin=by["ms",16,3],by["ms_twin",16,3]
    aa=statistics.median((b-a)/48 for a,b in zip(ship["samples_total_us"],twin["samples_total_us"]))
    # Also bound difference of medians: either A/A failure voids a performance decision.
    aa_medians=twin["stage_us"]-ship["stage_us"]
    aa_ok=max(abs(aa),abs(aa_medians))<=0.03+1e-12
    pass_pace=best[16]["stage_us"]<=1.22 and best[8]["stage_us"]<=1.16
    kill_depth=by["ms",16,4]["stage_us"]>1.30
    if not aa_ok: status="VOID: A/A exceeds 0.03 us/unit"
    elif kill_depth: status="KILL: depth is not the lever"
    elif pass_pace: status="PASS"
    else: status="HOLD: neither PASS nor KILL threshold met"
    vms=by["ms",16,2]["stage_us"]>=1.45
    return {"status":status,"best_ms":{str(n):{"ns":r["ns"],"stage_us":r["stage_us"]} for n,r in best.items()},
            "aa_paired_us_per_unit":aa,"aa_median_difference_us_per_unit":aa_medians,"aa_valid":aa_ok,
            "pass_pace":pass_pace,"kill_depth":kill_depth,"ns2_n16_ge_1_45":vms,
            "vms_depth_interpretation":"consistent with depth hypothesis; no layer claim" if vms else "depth is not what limits VMS"}


def decision_gates():
    def fixture(n8=1.16,n16=1.22,n4=1.22,aa=0.,correct=True):
        rows=[]
        for arm,n,ns in ARMS:
            pace=n8 if n==8 else n16
            if arm=="ms" and n==16 and ns==4: pace=n4
            if arm=="ms_twin": pace+=aa
            rows.append({"arm":arm,"n":n,"ns":ns,"stage_us":pace,"correct":correct,
                         "samples_total_us":[pace*48+2.]*12})
        return rows
    assert verdict(fixture())["status"]=="PASS"
    assert verdict(fixture(n8=1.160001))["status"].startswith("HOLD")
    assert verdict(fixture(n16=1.220001,n4=1.220001))["status"].startswith("HOLD")
    assert not verdict(fixture(n4=1.30))["kill_depth"]
    assert verdict(fixture(n4=1.300001))["status"].startswith("KILL")
    assert verdict(fixture(aa=.03))["aa_valid"]
    assert verdict(fixture(aa=.030001))["status"].startswith("VOID")
    assert verdict(fixture(correct=False))["status"].startswith("VOID")
    assert verdict(fixture(n16=1.45))["ns2_n16_ge_1_45"]
    assert not verdict(fixture(n16=1.449999))["ns2_n16_ge_1_45"]


def replay_schedule(rounds):
    for rnd in range(rounds):
        order=ARMS[rnd%len(ARMS):]+ARMS[:rnd%len(ARMS)]
        for key in order:
            yield rnd,key,(2*rnd+ARMS.index(key))%3


def layout_gates():
    rows=[]
    for n in (8,16,32):
        for ns in (2,3,4):
            ep=65536+ns*UNIT
            bo=ep+2*64*n*4
            assert bo%128==0 and (2*ns+4)*8<=4096
            assert set(range(ns)).isdisjoint(range(ns,2*ns))
            rows.append({"n":n,"ns":ns,"dynamic_smem_bytes":smem_bytes(n,ns),
                         "barriers":2*ns+4,"fits":smem_bytes(n,ns)<=SMEM_CAP})
    assert set(DROPPED_ARMS)=={("null",32,4),("rs",32,4)}
    # Exercise the actual common launch adapter with high addresses, every selector.
    class Tensor:
        def __init__(self,k): self.k=k
        def data_ptr(self): return 2**40+self.k*256
    class Module:
        def launch(self,*args): self.args=args
    mod=Module(); tensors=[Tensor(k) for k in range(6)]
    for key in ARMS:
        launch_arm(mod,key,48,GRID,*tensors,123)
        assert mod.args==(ARM_ID[key[0]],key[1],key[2],48,GRID,
                          *(t.data_ptr() for t in tensors),123)
        assert smem_bytes(key[1],key[2])<=SMEM_CAP
    return {"status":"PASS","layouts":rows,"dropped":[arm_name(k) for k in DROPPED_ARMS],
            "launch_adapter_paths":len(ARMS),"no_layer_state_access":True}


def hot_loop_accounting():
    return {
        "basis":"EST source-level exact counts; SASS lowering reported separately, no speed inferred",
        "weight_bytes_per_cta_unit_before_after":[36864,36864],
        "bulk_instructions_per_producer_unit_before_after":[1,1],
        "note":"m16xa actually uses one merged 36864-byte bulk copy; unchanged",
        "consumer_per_thread_per_unit_before_after":{
            "code_LDS128":[16,16],"scale_LDS32":[8,8],"bytes_read_weights":[288,288],
            "dec_half2_calls":[32,32],"ms_N8_MMA":[64,64],"ms_N16_MMA":[128,128],
            "ms_N8_X_LDS32":[128,128],"ms_N16_X_LDS32":[256,256],
            "rs_N32_WGMMA":[64,64],"rs_fence_commit_wait1":[32,32],"rs_wait0":[1,1]},
        "protocol_per_cta_48_units":{
            str(ns):{"producer_previous_empty_waits":48-ns,"consumer_previous_empty_waits_per_warp_total":4*(48-ns),
                     "consumer_full_waits_per_warp_total":4*48,"empty_arrives":4*48,
                     "producer_drain_waits":ns,"barrier_init_inval_each":2*ns+4} for ns in (2,3,4)},
        "protocol_dynamic_poll_instructions":"data dependent; no invented count; modulo/division are compile-time NS; inspect SASS",
        "epilogue_bytes_per_unit_before_after":{str(n):[64*n*4,64*n*4] for n in (8,16,32)},
        "x_bytes_global_startup_per_cta_before_after":{str(n):[n*1024*2,n*1024*2] for n in (8,16,32)},
        "clock_delta":"CTA0/thread0 only: two clock64 + two globaltimer reads, two uint64 stores (16 B) per launch; zero per-stage additions",
        "extra_sync_launches":0,
        "half_slot":"deferred: pack_tiles concatenates all 32768 code bytes then all 4096 scale bytes; K512 halves are not contiguous 18432-byte units"}


def benchmark(mod,args,report):
    import torch
    prop=torch.cuda.get_device_properties(0)
    assert "H200" in prop.name and prop.multi_processor_count==132 and prop.major==9, str(prop)
    report["device"]={"name":prop.name,"sms":prop.multi_processor_count}
    attrs={}
    for key in ARMS:
        arm,n,ns=key
        reg,local,static,dynamic,occ=mod.info(ARM_ID[arm],n,ns)
        assert 96<=reg<=128 and local<=16 and occ==1, (key,reg,local,occ)
        assert dynamic==smem_bytes(n,ns) and static+dynamic<=SMEM_CAP, (key,static,dynamic)
        attrs[arm_name(key)]={"regs":reg,"local_bytes":local,"static_smem":static,"dynamic_smem":dynamic,"ctas_per_sm":occ}
    report["kernel_attributes"]=attrs
    report["correctness"]=[]
    gpu_correctness(mod,torch,report["correctness"])
    assert len(report["correctness"])==len(ARMS)
    assert all(c["status"]=="PASS" for c in report["correctness"] if c["arm"]=="null"), "invalid matched null"
    timer=Timer(torch)
    stream=timer.stream.cuda_stream
    empty=timer.capture(lambda:mod.empty(stream))
    banks=[]
    for b in range(3):
        weights=make_weights(math.ceil(GRID*args.units/512),103+b)
        packed=prepare_bank(weights,args.units,GRID)
        banks.append(torch.from_numpy(packed).cuda())
    assert len({w.data_ptr() for w in banks})==3
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
    marks=torch.empty(2,device="cuda",dtype=torch.int64)
    graphs={}
    report["launch_bounds"]=[]
    timer.stream.wait_stream(torch.cuda.current_stream())
    for key in ARMS:
        arm,n,ns=key
        graphs[key]=[]
        for bank,w in enumerate(banks):
            bounds=tensor_bounds(arm,n,args.units,GRID,w,inputs[n],output,sm,err,marks,ns=ns)
            report["launch_bounds"].append({"bank":bank,**bounds})
            report["capture_progress"]={"arm":arm,"n":n,"ns":ns,"bank":bank,"status":"STARTED"}
            graphs[key].append(timer.capture(lambda key=key,n=n,w=w:launch_arm(mod,key,args.units,GRID,w,inputs[n],output,sm,err,marks,stream)))
            assert not err.any().item(), (key,bank,"protocol error")
            assert len(set(sm.cpu().tolist()))==GRID, (key,bank,"grid not one CTA per SM")
            report["capture_progress"]["status"]="PASS"
    report["clocks_before"]=clocks()
    before=statistics.median(timer.replay(empty) for _ in range(15))
    samples={k:[] for k in ARMS}; clock_samples={k:[] for k in ARMS}
    bank_samples={k:[] for k in ARMS}
    for rnd,key,bank in replay_schedule(args.rounds):
        samples[key].append(timer.replay(graphs[key][bank]))
        assert not err.any().item(), (key,"device protocol error")
        assert len(set(sm.cpu().tolist()))==GRID, (key,"SM coverage")
        cycles,ns_delta=marks.cpu().tolist()
        assert cycles>0 and ns_delta>0,(key,"invalid loop clock")
        clock_samples[key].append({"cycles":cycles,"globaltimer_ns":ns_delta,"MHz":1000*cycles/ns_delta})
        bank_samples[key].append(bank)
    after=statistics.median(timer.replay(empty) for _ in range(15))
    report["clocks_after"]=clocks()
    overhead=(before+after)/2
    report["empty_kernel_us"]={"before":before,"after":after,"subtracted":overhead}
    rows=[]
    good={(c["arm"],c["n"],c["ns"]) for c in report["correctness"] if c["status"]=="PASS"}
    for key in ARMS:
        arm,n,ns=key
        values=[(v-overhead)/args.units for v in samples[key]]
        assert min(values)>0, "invalid empty-kernel subtraction"
        stage=statistics.median(values)
        nullkey=("null",n,ns)
        null=statistics.median((v-overhead)/args.units for v in samples[nullkey])
        paired=[(v-overhead)/(z-overhead) for v,z in zip(samples[key],samples[nullkey])]
        row={"arm":arm,"n":n,"ns":ns,"stage_us":stage,"ratio_null":stage/null,"paired_ratio_null":statistics.median(paired),
             "weight_TBps":GRID*UNIT/(stage*1e6),"samples_total_us":samples[key],
             "stage_p10_us":float(np.percentile(values,10)),"stage_p90_us":float(np.percentile(values,90)),
             "banks_per_round":bank_samples[key],"units_per_cta":args.units,"correct":key in good,
             "clock_samples":clock_samples[key],"median_MHz":statistics.median(c["MHz"] for c in clock_samples[key]),
             "measurement":"MEASURED CUDA events; (kernel - empty)/units; MHz=1000*clock64_delta/globaltimer_ns over CTA0/thread0 unit loop"}
        rows.append(row)
        print(f"F3 {arm_name(key):20s} {stage:.4f} us/unit {row['median_MHz']:.1f} MHz null={row['ratio_null']:.3f}x",flush=True)
    report["results"]=rows
    report["verdict"]=verdict(rows)
    print("F3 "+report["verdict"]["status"],flush=True)
    print("F3 correctness "+" ".join(f'{arm_name((c["arm"],c["n"],c["ns"]))}:{c["status"]}' for c in report["correctness"]),flush=True)


def write_summary(report, out):
    # Never turn pending gates or compile failures into synthetic hardware results.
    lines=["| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |",
           "|---|---|---|---|---|---|---|",
           "| attached ring | ms N8/N16; rs N32 | prior ms 1.23/1.37 per unit (BT3/MP1); this run below | target 48*(1.37-1.22)=7.20 at N16; 48*(1.23-1.16)=3.36 at N8, conditional | retiring held slots may expose refill; no PDL in micro; layer carry unproven | R-b4m-0; R-knob-1; R-dsm-1 | "+report.get("verdict",{}).get("status",report["status"])+" |",
           "", "Kill risks first: consumer issue is the pacer; deeper copies are already hidden (R-knob-1 ring4 lost at T<=8). This diagnostic tests a new attached-consumer contrast, not an in-layer recommendation.",
           "EST stream accounting: 132*36864=4,866,048 B/unit; 4,866,048/4.64e6=1.049 us/unit + 1.9/48=0.040 us ramp. Weight bytes/instructions unchanged; extra capacity only. Actual pace decides whether refill is exposed.",
           "Decision rules: PASS = best ms N16 <=1.22 AND best ms N8 <=1.16 us/unit; KILL = ms N16 NS4 >1.30. NS2 N16 >=1.45 is separately reported; if false, depth is not what limits VMS. |A/A| <=0.03 us/unit is mandatory (paired and medians).",
           "Status: "+report["status"]+". "+report.get("error","")+" Unrun: "+"; ".join(report.get("not_run",[])),
           "N32/NS4 null+rs dropped: EST 65536+4*36864+2*64*32*4+4096=233472 B > 227*1024=232448 B. Original 4096-byte reservation retained.",
           "Optional half-slots deferred: pack_tiles stores code then scales, not contiguous K512 halves; original micro issues one merged bulk copy.",
           "", "| arm | MEASURED us/unit | MEASURED MHz | registers | spill store/load B | EST dynamic smem B | correctness |",
           "|---|---:|---:|---:|---|---:|---|"]
    timing={(r["arm"],r["n"],r["ns"]):r for r in report.get("results",[])}
    compiler={(r["arm"],r["n"],r["ns"]):r for r in report.get("compiler",{}).get("ptxas",[]) if "arm" in r}
    for key in ARMS:
        r=timing.get(key,{}); c=compiler.get(key,{})
        lines.append(f"| {arm_name(key)} | {r.get('stage_us','NOT RUN')} | {r.get('median_MHz','NOT RUN')} | {c.get('registers','NOT RUN')} | {c.get('spill_store_bytes','?')}/{c.get('spill_load_bytes','?')} | {smem_bytes(key[1],key[2])} | {r.get('correct','NOT RUN')} |")
    lines += ["", "Protocol coverage (all CTAs 0..131): t0 initializes/invals 2*NS+4 barriers; t256 refills and drains all NS slots; t0..127/t128..255 consume even/odd units, elected t0,32,64,96 / t128,160,192,224 release slots and epilogues; t288..319/t320..351 drain even/odd epilogues (j=lane; j<64*N/4; j+=32), t288/t320 return credits. All t0..383 join terminal syncthreads, including inactive t257..287 and t352..383. No CTA/thread early exits added; diagnostic wait timeout keeps arrivals and invalidates timing.",
              "Changed hot loops: identical 36864 B weight stream per CTA/unit; per consumer thread 16 LDS128+8 LDS32=288 B, 32 dec_half2, ms N8/N16 64/128 MMA +128/256 X LDS32, rs N32 64 WGMMA +32 fence/commit/wait1 +1 wait0, before=after. NS controls constant modulo/division and wait/drain counts; exact lowering/poll counts need compiled SASS. Clock adds 2 clock64+2 globaltimer reads and 16 B stores once per launch, zero new per-stage operations.",
              "", "Verdict detail: `"+json.dumps(report.get("verdict",{"status":"NOT RUN"}),sort_keys=True)+"`",
              "Exact inclusive changed ranges by bit (new m16xb lines; old-to-new mapping in results.json):",
              "", "| bit | changed lines |", "|---|---|"]
    groups={}
    for change in changed_lines():
        lo,hi=change["new"]
        value=str(lo) if lo==hi else f"{lo}-{hi}"
        if hi<lo: value=f"deletion at {lo}"
        for bit in change["bits"]: groups.setdefault(bit,[]).append(value)
    lines += [f"| {bit} | {', '.join(ranges)} |" for bit,ranges in groups.items()]
    lines += ["| local gate script | tools/local/f3_gates.sh:1-12 (new file) |",
              "| pod batch | tools/box/f3_batch.sh:1-21 (new file) |"]
    out.with_name("summary.md").write_bytes(("\n".join(lines)+"\n").encode())


def changed_lines():
    """Exact inclusive diff ranges, with implementation bits for reviewer navigation."""
    import difflib
    before=(ROOT/"kernels/lab/m16xa_micro.py").read_text().splitlines()
    after=Path(__file__).read_text().splitlines()
    boundaries=[(1,"entry/metadata")]
    anchors=(("# Retain the original", "NS arm selection"),
             ("CUDA_HEAD =", "NS ring, barriers and template dispatch"),
             ("    // Diagnostic only, outside", "clock64/globaltimer start"),
             ("    for (int u = wg;", "NS consumer protocol"),
             ("    if (blockIdx.x == 0 && tid == 0) {", "clock64/globaltimer end"),
             ("  // Includes all 384", "NS terminal drain/config/dispatch"),
             ("def address_bounds(", "CPU address/launch bounds"),
             ("def cpu_gates(", "CPU prepare/parity/protocol gates"),
             ("def compiler_report(", "per-arm compiler/SASS gates"),
             ("def launch_arm(", "dispatch and all-arm correctness"),
             ("def verdict(", "F3 decision and A/A gates"),
             ("def layout_gates(", "CPU layout/byte/instruction accounting"),
             ("def benchmark(", "rotated timing and MHz reporting"),
             ("def write_summary(", "summary/JSON/entry reporting"))
    for anchor,bit in anchors:
        # Anchors for clocks/consumer must be within the compute branch, not helpers.
        matches=[i+1 for i,line in enumerate(after) if line.startswith(anchor)]
        if anchor=="    for (int u = wg;": matches=[i for i in matches if i>240]
        if bit=="clock64/globaltimer end": matches=[i for i in matches if after[i-1]==anchor]
        if matches: boundaries.append((matches[0],bit))
    boundaries.sort()
    def bits(lo,hi):
        selected=set()
        for line in range(lo,max(lo,hi)+1):
            selected.add(next(bit for start,bit in reversed(boundaries) if start<=line))
        return sorted(selected)
    return [{"bits":bits(k+1,l),"op":tag,"old":[i+1,j],"new":[k+1,l]} for tag,i,j,k,l in
            difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes() if tag!="equal"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--compile-only",action="store_true")
    p.add_argument("--out",type=Path,default=Path("box_out/f3/results.json"))
    p.add_argument("--units",type=int,default=48)
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--timeout",type=float,default=350)
    p.add_argument("--_worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    assert args.units==48 and args.rounds==12 and 0<args.timeout<=355
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
            result={"task":"F3","status":"TIMEOUT","elapsed_s":time.monotonic()-started}
            args.out.write_bytes(json.dumps(result,indent=2).encode())
            if not args.compile_only: write_summary(result,args.out)
            raise SystemExit(124)
        raise SystemExit(code)
    started=time.monotonic()
    out=args.out.resolve()
    report={"task":"F3","status":"RUNNING","pre_registered":{"pass_ms_n16_le":1.22,"pass_ms_n8_le":1.16,"kill_ms_n16_ns4_gt":1.30,"ns2_vms_test_ge":1.45,"aa_abs_le":0.03},
            "changed_source_lines":changed_lines()}
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
        if args.compile_only: print("F3 CPU/compile/PTX/SASS gates PASS; JSON="+str(out),flush=True)
        else: benchmark(mod,args,report)
        report["status"]="COMPILE_GATES_PASS" if args.compile_only else "COMPLETE"
    except Exception as ex:
        report["status"]="FAIL"; report["error"]=repr(ex)
        raise
    finally:
        report["elapsed_s"]=time.monotonic()-started
        if not args.dry_run:
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(json.dumps(report,indent=2).encode())
            if not args.compile_only: write_summary(report,out)



# F5 work in progress.  The original F3 functions and gates above are retained.
# No kernel variant or GPU runner is exposed until the attribution arms and SASS
# count gates are implemented.  In particular, do not run F3 as if it were F5.
F5_ARMS = ("base", "twin", "hmma_sink", "dq_mov", "lds_w1", "lds_x1",
           "w_zero", "x_zero", "x_pad2", "wx_zero", "null")


def f5_prepare_inputs(arm, weights, x, units, grid):
    """Pure CPU preparation for the requested data contrasts.

    x is an unpacked array of BF16 bit patterns [N,1024].  Preserve the original
    buffer and create a packed tile per call; packed weight bytes are zeroed only
    after the base prepack, so their geometry and read bounds remain unchanged.
    """
    if arm not in F5_ARMS:
        raise ValueError("unknown F5 arm: " + arm)
    if x.dtype != np.uint16 or x.ndim != 2 or x.shape[1] != 1024 or x.shape[0] not in (8, 16):
        raise ValueError("x must contain uint16 BF16 bits, shape [8 or 16, 1024]")
    bank = prepare_bank(weights, units, grid)
    if arm in ("w_zero", "wx_zero"):
        bank = np.zeros_like(bank)
    tile = x.copy()
    if arm in ("x_zero", "wx_zero"):
        tile.fill(0)
    elif arm == "x_pad2":
        tile[2:] = 0
    return bank, pack_x(tile)


def f5_dry_run():
    """Run all inherited CPU gates and all F5 input-selection paths."""
    report = cpu_gates()
    weights = make_weights(1, 177)
    x = np.arange(16 * 1024, dtype=np.uint16).reshape(16, 1024)
    x_before = x.copy()
    # Check the shipped dequant lane mapping, not merely the packed bytes.
    assert not np.any(emulate_cuda_a(np.zeros(UNIT, dtype=np.uint8)))
    original = prepare_bank(weights, 4, 4)
    controls = {}
    for n in (8, 16):
        for arm in F5_ARMS:
            w, px = f5_prepare_inputs(arm, weights, x[:n], 4, 4)
            expected_x = x[:n].copy()
            if arm in ("x_zero", "wx_zero"):
                expected_x.fill(0)
            elif arm == "x_pad2":
                expected_x[2:] = 0
            assert np.array_equal(px, pack_x(expected_x)), (arm, n, "x")
            assert w.shape == original.shape and w.nbytes == original.nbytes
            if arm in ("w_zero", "wx_zero"):
                assert not np.any(w), (arm, n, "w")
            else:
                assert np.array_equal(w, original), (arm, n, "w")
            controls[f"{arm}_n{n}"] = {"weight_bytes": w.nbytes,
                                        "x_packed_bytes": px.nbytes, "status": "PASS"}
    assert np.array_equal(x, x_before)
    for name in ("w13", "w13_sf"):
        missing = dict(weights)
        missing.pop(name)
        try:
            f5_prepare_inputs("base", missing, x, 4, 4)
        except KeyError:
            pass
        else:
            raise AssertionError("missing real base key accepted: " + name)
    try:
        f5_prepare_inputs("not_an_arm", weights, x, 4, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown arm accepted")
    return {"status": "PASS", "inherited_f3_cpu_gates": report,
            "f5_input_paths": controls,
            "limitation": "No F5 CUDA variants, compile gates, correctness, timing or power runner implemented."}


def f5_main():
    p = argparse.ArgumentParser(description="F5 CPU input preflight; GPU implementation incomplete")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.dry_run:
        print(json.dumps(f5_dry_run(), indent=2))
        return
    raise SystemExit("F5 INCOMPLETE: attribution kernels, F5 SASS gates and GPU runner are not implemented; refusing F3 fallback")


if __name__ == "__main__":
    f5_main()

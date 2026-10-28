"""vds_lab.py: VDS2 = dense BF16 projections at T = 9..32 (decode16 / decode32) on RS-wgmma with lossless P12 weights;
step-0 microbench vs the shipped GD GEMVs / DG1 cat (extracted from the layer file) and cuBLAS (notes/study/vds_dense.md).

Design (VDS2; VDS1's cluster split-K ran in two waves on the H200: max 15 clusters of 8 / 30 of 4 at 1 CTA/SM, R-vds-1):
  * swap-AB: y^T[rows, T] = W[rows, K] . x^T; wgmma m64nNk16 RS (A = P12-decoded weight fragments in registers, GD's decode),
    B = the CTA's x slice [N tokens x 64 nst k] in shared memory (no-swizzle core-matrix layout), FP32 accumulate, N = 16 / 32.
  * one producer lane streams one stage = one 64-k unit for every consumer warp (NW x 1536 B, contiguous) per 1-D
    cp.async.bulk into an NST-deep ring; FRONT-major pack (stage s of every CTA adjacent); slot released once the warp's
    48 B per lane are in registers; A double-buffered across units (wgmma.wait_group 1).
  * x first: the producer issues pf stages, then waits until the consumers have x in smem, then fills the ring (x no longer
    queues behind NST weight stages). In-layer the weight stages go out before griddepcontrol.wait (PDL); only consumers wait.
  * 1 CTA / SM, NWG consumer warpgroups (one 64-row tile each) + 1 producer warp, grid (row groups, S), NO cluster:
    K split S ways (uneven: units [ks NU / S, (ks + 1) NU / S)), FP32 partials red.add.v4 into a zeroed workspace in fragment
    order, per-tile arrival counter, the last-arriving warpgroup reads the sum, stores (bf16 / f32) and re-zeroes ws + counter.
  * per-row-group P12 window base (tb / tb2 / tb3 split at rgb1 / rgb2): the cat's pooler / router-hi / router-lo rows.
  * Bounded waits everywhere (R9): a hang sets err[0], never traps.
  Lab flags: 1 exit at entry, 2 no wgmma (decode kept alive), 4 W-only (no LDS / decode / wgmma), 8 per-CTA stamps + per-warp
  full-barrier wait clocks.

    python3 kernels/lab/vds_lab.py [--projs=pooler,cat,unpooler,qkv,o] [--ts=32,16] [--reps=15] [--rounds=6] [--build-only=1]
    python3 kernels/lab/vds_lab.py --selfcheck=1      (CPU: pack -> emulated decode -> fragments, bit-exact; no GPU kernel)
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ESC_CAP = 64   # per warp chain (two entries per lane); the cat's router-lo rows reach ~38 at CL 3

# (N, NWG, NST, NSTGM, OUT): N tokens (wgmma N), NWG consumer warpgroups = 64 NWG rows per CTA, NST ring stages, NSTGM =
# most 64-k units per CTA (x slice size), OUT 0 bf16 / 1 f32
# (N, NWG, NST, NSTGM, OUT, CL): N tokens (wgmma N), NWG consumer warpgroups = 64 NWG rows per CTA, NST ring stages (one 64-k
# unit of weights for every consumer warp + the matching N x 64 x slice), NSTGM = most units per CTA, OUT 0 bf16 / 1 f32,
# CL = cluster size along K (DSMEM reduce-scatter by 8-token blocks); VD / DG1's one-wave geometries (max clusters at 1 CTA / SM:
# 15 of 8, 30 of 4 on the box H200, R-vds-1)
VDS_INSTS = [
    (32, 2, 10, 43, 1, 3),   # 0  cat T32: 36 rg x CL 3 = 108 CTAs (42 / 43 units), f32 out
    (16, 2, 12, 43, 1, 3),   # 1  cat T16
    (32, 1, 12, 43, 0, 3),   # 2  qkv T32: 36 rg x CL 3 = 108
    (16, 1, 14, 43, 0, 3),   # 3  qkv T16
    (32, 2, 10, 32, 0, 2),   # 4  unpooler T32: 64 rg x CL 2 = 128
    (16, 2, 12, 32, 0, 2),   # 5  unpooler T16
    (32, 1, 8, 16, 0, 1),    # 6  o T32: 128 rg = 128
    (16, 1, 8, 16, 0, 1),    # 7  o T16
    (32, 2, 10, 43, 0, 3),   # 8  pooler T32: 32 rg x CL 3 = 96 (reference vs GD)
    (32, 2, 8, 43, 1, 3),    # 9  cat T32, NST 8
]

PROJ = {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "cat": (4608, 8192)}
WNAME = {"qkv": "qkv_gate", "o": "o", "pooler": "pooler", "unpooler": "unpooler"}
# (proj, T) -> [(kid, S, label)], first = primary (gets the diagnostic arms)
CONFIGS = {
    ("cat", 32): [(0, 3, "bm128_cl3"), (9, 3, "bm128_cl3_nst8")],
    ("cat", 16): [(1, 3, "bm128_cl3")],
    ("pooler", 32): [(8, 3, "bm128_cl3")],
    ("unpooler", 32): [(4, 2, "bm128_cl2")],
    ("unpooler", 16): [(5, 2, "bm128_cl2")],
    ("qkv", 32): [(2, 3, "bm64_cl3")],
    ("qkv", 16): [(3, 3, "bm64_cl3")],
    ("o", 32): [(6, 1, "bm64")],
    ("o", 16): [(7, 1, "bm64")],
}
# shipped GD routes at buckets 16 / 32 (k3_slim_3 GD_ROUTE) + pooler stand-ins (the layer runs P1M / DG1 cat there)
GD_LAB_ROUTE = {("qkv", 16): (2, 1, 4, 8, 1, 4, 0), ("qkv", 32): (4, 1, 4, 8, 1, 4, 0),
                ("o", 16): (2, 2, 4, 4, 2, 1, 0), ("o", 32): (4, 2, 4, 4, 1, 1, 0),
                ("unpooler", 16): (2, 2, 4, 16, 2, 1, 0), ("unpooler", 32): (4, 2, 4, 16, 1, 1, 0),
                ("pooler", 16): (2, 1, 8, 16, 2, 1, 1), ("pooler", 32): (4, 2, 4, 16, 1, 2, 0)}

VDS_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t vds_run(int64_t kid, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor err,
                torch::Tensor stamps, int64_t ldo, int64_t nrg, std::vector<int64_t> tbs, int64_t pf, int64_t trig, int64_t flags,
                int64_t pdl);
std::vector<int64_t> vds_info(int64_t kid);
"""

VDS_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace vds {

constexpr long long WAIT_BUDGET = 2000000000LL;   // ~1 s of clock64: a hang becomes err[0] != 0 (R9)
enum : int { F_EXIT = 1, F_NOMMA = 2, F_WONLY = 4, F_STAMP = 8 };

struct Args {
  const uint8_t* w;            // P12 stream: stage s of CTA c at (s * ncta + c) * SBW bytes; [warp][plane][lane][16 B]
  const uint32_t* esc;         // [ncta * NW][64] escape entries (step << 20 | lane << 15 | reg << 11 | half << 10 | xor7)
  const uint16_t* x;           // [T, K] bf16 bits (the predecessor's output: read after griddepcontrol.wait)
  void* out;                   // [T, ldo] bf16 (OUT 0) / f32 (OUT 1)
  int* err;
  unsigned long long* stamps;  // lab: [ncta][5] globaltimer + [ncta * NW][2] (full-wait clocks, loop clocks)
  int T, K, ldo, ncta, S, NU, pf, trig, flags, rgb1, rgb2;
  uint32_t tb, tb2, tb3;       // P12 window base x 0x01010101 per row-group range
};

template <int N, int NWG, int NST, int NSTGM, int OUT, int CL>
struct Cfg {
  static constexpr int NW = NWG * 4;               // consumer warps (one 16-row tile each)
  static constexpr int NTH = NW * 32;
  static constexpr int THREADS = NTH + 32;         // + one producer warp
  static constexpr int UB = 1536;                  // 16 rows x 64 k x 12 bit per warp-unit
  static constexpr int SBW = NW * UB;              // stage weights: one unit for every consumer warp
  static constexpr int SBX = N * 128;              // stage x: N tokens x 64 k bf16, core-matrix layout (SBO 1024)
  static constexpr int SB = SBW + SBX;
  static constexpr int NACC = N / 2;               // fp32 accumulators per consumer thread
  static constexpr int NB = N / 8;                 // 8-token blocks (DSMEM reduce-scatter unit)
  static constexpr int JMAX = (NB + CL - 1) / CL;
  static constexpr int RB_OFF = NST * SB;
  static constexpr int RB_BYTES = CL > 1 ? (CL - 1) * JMAX * NTH * 16 : 0;
  static constexpr int BAR_OFF = RB_OFF + RB_BYTES;
  static constexpr int SMEM = BAR_OFF + 8 * (2 * NST + 1) + 16;
  static constexpr int XPL = N / 4;                // 16-B x chunks per producer lane per stage
  static_assert(N % 16 == 0 && NACC % 4 == 0 && CL >= 1, "N / CL");   // a rank may own no 8-token block (N 16, CL 3)
  static_assert(SB % 16 == 0 && SMEM <= 232448, "smem");
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// @lab{
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// @lab}
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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, bool& dead) {
  if (dead || mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); dead = true; return; }
  }
}
__device__ __forceinline__ bool mbar_try_cl(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait_cl(uint32_t b, uint32_t par, int* err, bool& dead) {
  if (dead || mbar_try_cl(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try_cl(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 2); dead = true; return; }
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t lde(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 r;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a) : "memory");
  return r;
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_async4(uint32_t a, float v0, float v1, float v2, float v3, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n"
               :: "r"(a), "r"(__float_as_uint(v0)), "r"(__float_as_uint(v1)), "r"(__float_as_uint(v2)),
                  "r"(__float_as_uint(v3)), "r"(mb) : "memory");
}
// SM90 smem matrix descriptor, no swizzle (core matrices 8 rows x 16 B contiguous): LBO = K-adjacent core matrices,
// SBO = 8-row-group stride
__device__ __forceinline__ uint64_t desc_ns(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NP> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NP) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                          uint32_t lo, uint32_t hi);
@WGSPECS@

// GD's P12 decode: one 64-k unit (4 k16 steps) -> 16 A-fragment registers a[4q + i] (AND, IADD, SHF, AND, IADD, 4 PRMT / 8 w)
__device__ __forceinline__ void decode(const uint4 (&p)[3], uint32_t tb, uint32_t (&a)[16]) {
  const uint32_t L[8] = {p[0].x, p[0].y, p[0].z, p[0].w, p[1].x, p[1].y, p[1].z, p[1].w};
  const uint32_t C[4] = {p[2].x, p[2].y, p[2].z, p[2].w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t h0 = (C[q] & 0x87878787u) + tb;
    const uint32_t h1 = (__funnelshift_r(C[q], C[q], 4) & 0x87878787u) + tb;
    a[4 * q + 0] = __byte_perm(L[2 * q], h0, 0x5140);
    a[4 * q + 1] = __byte_perm(L[2 * q], h0, 0x7362);
    a[4 * q + 2] = __byte_perm(L[2 * q + 1], h1, 0x5140);
    a[4 * q + 3] = __byte_perm(L[2 * q + 1], h1, 0x7362);
  }
}
__device__ __forceinline__ void patch(uint32_t (&a)[16], uint32_t e, int lane) {
  if (lane == (int)((e >> 15) & 31u)) {
    const int r = (int)((e >> 11) & 15u);
    const uint32_t v = (e & 127u) << (8 + (((e >> 10) & 1u) << 4));
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (i == r) a[i] ^= v;
  }
}

template <int N, int NWG, int NST, int NSTGM, int OUT, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1) vds_kernel(const __grid_constant__ Args A) {
  using C = Cfg<N, NWG, NST, NSTGM, OUT, CL>;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = (int)threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // @lab{
  if (A.flags & F_EXIT) return;
  const bool stamp = (A.flags & F_STAMP) != 0;
  unsigned long long t_entry = 0, t_x = 0, t_first = 0, t_loop = 0;
  long long w_wait = 0, w_t0 = 0;
  if (stamp && tid == 0) t_entry = gtime();
  // @lab}
  if (A.trig) pdl_trigger();
  const uint32_t s0 = sa(smem);
  const uint32_t full0 = s0 + C::BAR_OFF, empty0 = full0 + 8 * NST, rbar = full0 + 16 * NST;
  const int rg = (int)blockIdx.x, ks = CL > 1 ? (int)blockIdx.y : 0;
  const int cta = ks * (int)gridDim.x + rg;
  const int u0 = (ks * A.NU) / CL, nst = ((ks + 1) * A.NU) / CL - u0;   // this CTA's 64-k units [u0, u0 + nst)
  const int NB = C::NB;
  const int jlo = (ks * NB) / CL, jhi = ((ks + 1) * NB) / CL;              // 8-token blocks this rank finishes
  bool dead = false;
  if (tid == 0) {
#pragma unroll 1
    for (int i = 0; i < NST; ++i) { mbar_init(full0 + 8 * i, 33); mbar_init(empty0 + 8 * i, C::NW); }
    if constexpr (CL > 1) { mbar_init(rbar, 1); mbar_expect_tx(rbar, (uint32_t)((CL - 1) * (jhi - jlo) * C::NTH * 16)); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if constexpr (CL > 1) cluster_arrive_relaxed();   // "my barriers are initialized" (waited before the first push)

  if (warp == C::NW) {
    // ------------------------------------------------ producer warp: lane 0 bulk-copies the weights, every lane cp.asyncs
    // its 16-B x chunks of the stage (zero fill past T) and arrives (noinc) on the stage's full barrier (count 1 + 32)
    const uint8_t* wb = A.w + (size_t)cta * C::SBW;
    const size_t wstep = (size_t)A.ncta * C::SBW;
    const uint16_t* xg = A.x;
    auto issue_w = [&](int i, int slot) {
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, (uint32_t)C::SBW);
        bulk_g2s(s0 + (uint32_t)(slot * C::SB), wb + (size_t)i * wstep, (uint32_t)C::SBW, full0 + 8 * slot);
      }
    };
    auto issue_x = [&](int i, int slot) {
      const uint32_t xs = s0 + (uint32_t)(slot * C::SB + C::SBW);
#pragma unroll
      for (int j = 0; j < C::XPL; ++j) {
        const int idx = lane + 32 * j, n = idx >> 3, c = idx & 7;
        const bool live = n < A.T;
        const uint16_t* src = xg + (live ? (size_t)n * A.K + (size_t)(u0 + i) * 64 + c * 8 : 0);
        cp_async16(xs + (uint32_t)((n >> 3) * 1024 + c * 128 + (n & 7) * 16), src, live ? 16u : 0u);
      }
      cp_async_arrive_noinc(full0 + 8 * slot);
    };
    const int npro = nst < NST ? nst : NST;
    const int pf = A.pf < npro ? A.pf : npro;
#pragma unroll 1
    for (int i = 0; i < pf; ++i) issue_w(i, i);     // weights do not depend on the predecessor (PDL)
    pdl_wait();
#pragma unroll 1
    for (int i = 0; i < pf; ++i) issue_x(i, i);
#pragma unroll 1
    for (int i = pf; i < npro; ++i) { issue_x(i, i); issue_w(i, i); }
#pragma unroll 1
    for (int i = NST; i < nst; ++i) {
      const int slot = i % NST;
      mbar_wait(empty0 + 8 * slot, (uint32_t)(((i / NST) - 1) & 1), A.err, dead);
      issue_x(i, slot);
      issue_w(i, slot);
    }
    return;
  }

  // -------------------------------------------------------------------- consumers
  const uint32_t em = lde(A.esc + (size_t)(cta * C::NW + warp) * 64 + lane);
  const uint32_t em1 = lde(A.esc + (size_t)(cta * C::NW + warp) * 64 + 32 + lane);
  const uint32_t tb = rg < A.rgb1 ? A.tb : (rg < A.rgb2 ? A.tb2 : A.tb3);
  pdl_wait();
  // @lab{
  if (stamp && tid == 0) t_x = gtime();
  if (stamp) w_t0 = clock64();
  const bool nomma = (A.flags & F_NOMMA) != 0, wonly = (A.flags & F_WONLY) != 0;
  // @lab}
  float acc[C::NACC];
#pragma unroll
  for (int i = 0; i < C::NACC; ++i) acc[i] = 0.f;
  uint32_t a0[16], a1[16];
  int ecur = 0;
  uint32_t enx = __shfl_sync(0xffffffffu, em, 0);
  const uint64_t dx = desc_ns(s0 + C::SBW, 128u, 1024u);   // x slot of ring slot 0
  const uint32_t dlo = (uint32_t)dx, dhi = (uint32_t)(dx >> 32);
  const uint32_t rp = s0 + (uint32_t)((warp * 96 + lane) * 16);
  // unit s: wait for slot s % NST, decode, 4 wgmma on that slot's x, then release slot (s - 1) % NST: its group (weights
  // in registers, x read by the async proxy) is complete after this unit's wait_group 1. An odd count runs a zero-A tail
  // unit (a conditional wgmma after the loop made ptxas serialize every wgmma: C7515).
  auto unit = [&](int s, uint32_t (&a)[16], bool live) {
    const int slot = s % NST;
    if (live) {
      // @lab{
      long long tw0 = 0;
      if (stamp) tw0 = clock64();
      // @lab}
      mbar_wait(full0 + 8 * slot, (uint32_t)((s / NST) & 1), A.err, dead);
      // @lab{
      if (stamp) w_wait += clock64() - tw0;
      if (stamp && s == 0 && tid == 0) t_first = gtime();
      // @lab}
      uint4 p[3];
      const uint32_t ra = rp + (uint32_t)(slot * C::SB);
      p[0] = lds128(ra);
      p[1] = lds128(ra + 512u);
      p[2] = lds128(ra + 1024u);
      decode(p, tb, a);
      while ((int)(enx >> 20) == s) {
        patch(a, enx, lane);
        ++ecur;
        enx = ecur < 32 ? __shfl_sync(0xffffffffu, em, ecur & 31)
                        : ecur < 64 ? __shfl_sync(0xffffffffu, em1, ecur & 31) : 0xFFFFFFFFu;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) a[i] = 0u;
    }
    // @lab{
    if (wonly || nomma) {
      if (nomma && live)
        asm volatile("" :: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
                     "r"(a[8]), "r"(a[9]), "r"(a[10]), "r"(a[11]), "r"(a[12]), "r"(a[13]), "r"(a[14]), "r"(a[15]));
      __syncwarp();
      if (live && lane == 0) mbar_arrive(empty0 + 8 * slot);
      return;
    }
    // @lab}
    const uint32_t xd = dlo + (uint32_t)((live ? slot : (s - 1) % NST) * (C::SB >> 4));   // tail: the last live x
    wg_fence();
#pragma unroll
    for (int q = 0; q < 4; ++q)
      wgmma_rs<N>(acc, a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], xd + (uint32_t)(16 * q), dhi);
    wg_commit();
    wg_wait<1>();
    if (s >= 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * ((s - 1) % NST));
    }
  };
#pragma unroll 1
  for (int s = 0; s < nst; s += 2) {
    unit(s, a0, true);
    unit(s + 1, a1, s + 1 < nst);
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < C::NACC; ++i) asm volatile("" : "+f"(acc[i]) :: "memory");
  // @lab{
  if (stamp && tid == 0) t_loop = gtime();
  if (stamp && lane == 0) {
    unsigned long long* wp = A.stamps + 5 * (size_t)A.ncta + 2 * ((size_t)cta * C::NW + warp);
    wp[0] = (unsigned long long)w_wait; wp[1] = (unsigned long long)(clock64() - w_t0);
  }
  // @lab}

  // -------------------------------------------------------------------- epilogue: DSMEM reduce-scatter by 8-token blocks
  const int g = lane >> 2, t4 = lane & 3;
  const int row0 = (rg * C::NW + warp) * 16 + g;   // this thread's rows: row0, row0 + 8
  float v[C::NACC];
#pragma unroll
  for (int i = 0; i < C::NACC; ++i) v[i] = acc[i];
  if constexpr (CL > 1) {
    cluster_wait();                                  // every rank's receive mbarrier is initialized
    const uint32_t rb0 = s0 + C::RB_OFF;
#pragma unroll
    for (int p = 0; p < CL; ++p) {
      if (p == ks) continue;
      const int plo = (p * NB) / CL, phi = ((p + 1) * NB) / CL;
      const int slot = ks < p ? ks : ks - 1;
      const uint32_t rbr = mapa(rbar, (uint32_t)p);
#pragma unroll
      for (int j = 0; j < C::NB; ++j) {
        if (j < plo || j >= phi) continue;
        st_async4(mapa(rb0 + (uint32_t)(((slot * C::JMAX + (j - plo)) * C::NTH + tid) * 16), (uint32_t)p),
                  acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3], rbr);
      }
    }
    mbar_wait_cl(rbar, 0, A.err, dead);              // the other ranks' copies of my blocks have landed
    const float4* RB = reinterpret_cast<const float4*>(smem + C::RB_OFF);
#pragma unroll
    for (int j = 0; j < C::NB; ++j) {
      if (j < jlo || j >= jhi) continue;
#pragma unroll
      for (int sl = 0; sl < CL - 1; ++sl) {
        const float4 q = RB[(sl * C::JMAX + (j - jlo)) * C::NTH + tid];
        v[4 * j] += q.x; v[4 * j + 1] += q.y; v[4 * j + 2] += q.z; v[4 * j + 3] += q.w;
      }
    }
  }
#pragma unroll
  for (int i = 0; i < C::NACC; ++i) {
    const int j = i >> 2, c = i & 3;
    if (j < jlo || j >= jhi) continue;
    const int row = row0 + 8 * (c >> 1), tok = 8 * j + 2 * t4 + (c & 1);
    if (tok < A.T) {
      if constexpr (OUT == 1) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + row] = v[i];
      else reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v[i]);
    }
  }
  // @lab{
  if (stamp && tid == 0) {
    unsigned long long* sp = A.stamps + 5 * (size_t)cta;
    sp[0] = t_entry; sp[1] = t_x; sp[2] = t_first; sp[3] = t_loop; sp[4] = gtime();
  }
  // @lab}
}

// @lab{
__global__ void null_kernel(int* e) { if (threadIdx.x == 1023) e[3] = 0; }
// @lab}

}  // namespace vds

template <int N, int NWG, int NST, int NSTGM, int OUT, int CL>
static int64_t vds_launch_t(const torch::Tensor& w, const torch::Tensor& esc, const torch::Tensor& x, const torch::Tensor& out,
                            const torch::Tensor& err, const torch::Tensor& stamps, int64_t ldo, int64_t nrg,
                            const std::vector<int64_t>& tbs, int64_t pf, int64_t trig, int64_t flags, int64_t pdl) {
  using C = vds::Cfg<N, NWG, NST, NSTGM, OUT, CL>;
  auto kern = vds::vds_kernel<N, NWG, NST, NSTGM, OUT, CL>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    attr = true;
  }
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "vds: x bf16 [T, K]");
  const int64_t T = x.size(0), K = x.size(1), NU = K / 64;
  TORCH_CHECK(T >= 1 && T <= N && K % 64 == 0 && ((uintptr_t)x.data_ptr() % 16) == 0, "vds: x shape");
  TORCH_CHECK((NU + CL - 1) / CL <= NSTGM, "vds: K split exceeds NSTGM units per CTA");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.stride(1) == 1 && out.size(0) >= T && out.size(1) >= nrg * 64 * NWG
              && ldo == out.stride(0), "vds: out");
  TORCH_CHECK(out.scalar_type() == (OUT ? at::kFloat : at::kBFloat16), "vds: out dtype");
  const int64_t ncta = nrg * CL;
  TORCH_CHECK(w.is_cuda() && w.numel() * w.element_size() == ncta * (int64_t)C::SBW * NSTGM && ((uintptr_t)w.data_ptr() % 16) == 0,
              "vds: packed weights size");
  TORCH_CHECK(esc.is_cuda() && esc.numel() == ncta * C::NW * 64, "vds: escapes");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt, "vds: err");
  TORCH_CHECK(tbs.size() == 5, "vds: tbs = (tb, tb2, tb3, rgb1, rgb2)");
  vds::Args A;
  A.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  A.esc = reinterpret_cast<const uint32_t*>(esc.data_ptr());
  A.x = reinterpret_cast<const uint16_t*>(x.data_ptr());
  A.out = out.data_ptr();
  A.err = err.data_ptr<int>();
  A.stamps = reinterpret_cast<unsigned long long*>(stamps.data_ptr());   // lab stamps (unused in the layer copy)
  A.T = (int)T; A.K = (int)K; A.ldo = (int)ldo; A.ncta = (int)ncta; A.S = CL; A.NU = (int)NU; A.pf = (int)pf;
  A.trig = (int)trig; A.flags = (int)flags;
  A.tb = (uint32_t)tbs[0]; A.tb2 = (uint32_t)tbs[1]; A.tb3 = (uint32_t)tbs[2]; A.rgb1 = (int)tbs[3]; A.rgb2 = (int)tbs[4];
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)nrg, (unsigned)CL, 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute at_[2];
  int na = 0;
  if (CL > 1) {
    at_[na].id = cudaLaunchAttributeClusterDimension;
    at_[na].val.clusterDim.x = 1; at_[na].val.clusterDim.y = CL; at_[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = na ? at_ : nullptr;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, A));
  return 0;
}

template <int N, int NWG, int NST, int NSTGM, int OUT, int CL>
static std::vector<int64_t> vds_info_t() {
  using C = vds::Cfg<N, NWG, NST, NSTGM, OUT, CL>;
  auto kern = vds::vds_kernel<N, NWG, NST, NSTGM, OUT, CL>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, C::THREADS, C::SMEM));
  int ncl = -1;
  if (CL > 1) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1, CL, 1);
    cfg.blockDim = dim3(C::THREADS, 1, 1);
    cfg.dynamicSmemBytes = C::SMEM;
    cudaLaunchAttribute a;
    a.id = cudaLaunchAttributeClusterDimension;
    a.val.clusterDim.x = 1; a.val.clusterDim.y = CL; a.val.clusterDim.z = 1;
    cfg.attrs = &a; cfg.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&ncl, kern, &cfg) != cudaSuccess) { cudaGetLastError(); ncl = -1; }
  }
  return {(int64_t)C::SMEM, (int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ, (int64_t)C::SBW, (int64_t)NSTGM,
          (int64_t)C::NW, (int64_t)ncl};
}

int64_t vds_run(int64_t kid, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor err,
                torch::Tensor stamps, int64_t ldo, int64_t nrg, std::vector<int64_t> tbs, int64_t pf, int64_t trig, int64_t flags,
                int64_t pdl) {
  // @lab{
  if (kid < 0) {
    vds::null_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream().stream()>>>(err.data_ptr<int>());
    return 0;
  }
  // @lab}
  switch (kid) {
@RUNCASES@
  }
  TORCH_CHECK(false, "vds: unknown kid ", kid);
  return -2;
}

std::vector<int64_t> vds_info(int64_t kid) {
  switch (kid) {
@INFOCASES@
  }
  TORCH_CHECK(false, "vds: unknown kid ", kid);
  return {};
}
"""


def vds_wg_specs(ns):
    """wgmma.mma_async m64nNk16 RS: f32 += bf16 (A in 4 registers) x bf16 (B: K-major smem descriptor), scale-d = 1."""
    out = []
    for n in ns:
        na = n // 2
        regs = ",".join("%%%d" % i for i in range(na))
        outs = ", ".join('"+f"(d[%d])' % i for i in range(na))
        out.append(
            "template <> __device__ __forceinline__ void wgmma_rs<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,"
            " uint32_t lo, uint32_t hi) {\n"
            '  asm volatile("{\\n.reg .b64 db;\\n.reg .pred p;\\nmov.b64 db, {%%%d, %%%d};\\nsetp.ne.b32 p, 1, 0;\\n'
            'wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%%%d, %%%d, %%%d, %%%d}, db, p, 1, 1, 0;\\n}\\n"\n'
            '               : %s\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));\n}'
            % (n, na + 4, na + 5, n, regs, na, na + 1, na + 2, na + 3, outs))
    return "\n".join(out)


def vds_cuda_src(insts=VDS_INSTS):
    run = "\n".join(f"    case {i}: return vds_launch_t<{', '.join(map(str, t))}>(w, esc, x, out, err, stamps, ldo, nrg, tbs, pf, trig, "
                    f"flags, pdl);" for i, t in enumerate(insts))
    info = "\n".join(f"    case {i}: return vds_info_t<{', '.join(map(str, t))}>();" for i, t in enumerate(insts))
    ns = sorted({ins[0] for ins in insts})
    return VDS_CUDA_SRC.replace("@RUNCASES@", run).replace("@INFOCASES@", info).replace("@WGSPECS@", vds_wg_specs(ns))


def vds_build(insts=VDS_INSTS, name="vdslab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda = vds_cuda_src(insts)
    tag = hashlib.sha1((VDS_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=VDS_CPP_SRC, cuda_sources=cuda, functions=["vds_run", "vds_info"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
                      verbose=verbose)
    print(f"[vds] built {len(insts)} kernels in {time.time() - t0:.1f}s", flush=True)
    return mod


# ---------------------------------------------------------------- P12 pack (lossless; GD's code words, natural-k fragments)
def _i32(t64):
    return torch.where(t64 >= 2 ** 31, t64 - 2 ** 32, t64).to(torch.int32)


def vds_frag(wv):
    """[n, k] bf16 -> [n / 16, k / 64, 32 lanes, 4 q, 8 j] 16-bit codes: weight j of lane (g, t) in k16 step q is
    W[16 rt + g + 8 ((j >> 1) & 1)][64 u + 16 q + 2 t + (j & 1) + 8 (j >> 2)] (wgmma A fragment, natural k order)."""
    n, k = wv.shape
    dev = wv.device
    w16 = (wv.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).reshape(-1)
    lane = torch.arange(32, device=dev)
    g = (lane >> 2).view(32, 1, 1)
    t = (lane & 3).view(32, 1, 1)
    q = torch.arange(4, device=dev).view(1, 4, 1)
    j = torch.arange(8, device=dev).view(1, 1, 8)
    row16 = (g + 8 * ((j >> 1) & 1)).expand(32, 4, 8)
    k64 = (16 * q + 2 * t + (j & 1) + 8 * (j >> 2)).expand(32, 4, 8)
    nrt, nu = n // 16, k // 64
    rows = torch.arange(nrt, device=dev).view(nrt, 1, 1, 1, 1) * 16 + row16.view(1, 1, 32, 4, 8)
    cols = torch.arange(nu, device=dev).view(1, nu, 1, 1, 1) * 64 + k64.view(1, 1, 32, 4, 8)
    return w16[rows * k + cols]


def vds_window_base(wv):
    h7 = (wv.contiguous().view(torch.int16).to(torch.int32) >> 8) & 0x7F
    hist = torch.bincount(h7.reshape(-1), minlength=128).double()
    cs = torch.cat([torch.zeros(1, dtype=hist.dtype, device=hist.device), hist.cumsum(0)])
    return int((cs[8:129] - cs[0:121]).argmax())


def vds_splits(nu, s):
    return [((ks * nu) // s, ((ks + 1) * nu) // s) for ks in range(s)]


def vds_bases(wv, nwg, seg=None):
    """per-row-group window bases; seg = row boundaries of differently scaled blocks (e.g. the cat's [4096, 4352])."""
    n = wv.shape[0]
    rows = 64 * nwg
    cuts = [0] + list(seg or []) + [n]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        assert a % rows == 0 and b % rows == 0, "segments must align with row groups"
        out += [vds_window_base(wv[a:b])] * ((b - a) // rows)
    return out


def vds_tbs(bases):
    """(tb, tb2, tb3, rgb1, rgb2) for at most 3 runs of equal bases over the row groups."""
    runs = []
    for i, b in enumerate(bases):
        if not runs or runs[-1][0] != b:
            runs.append([b, i])
    assert len(runs) <= 3, "at most 3 base runs"
    while len(runs) < 3:
        runs.append([runs[-1][0], len(bases)])
    return [runs[0][0] * 0x01010101, runs[1][0] * 0x01010101, runs[2][0] * 0x01010101, runs[1][1], runs[2][1]]


def vds_pack(wv, nwg, s, nstgm, bases=None):
    """GD's P12 code words in VDS2's stream order: stage st of CTA c = (ks, rg) at ((st * ncta + c) * NW + w) * 1536 B, inside a
    warp-unit [plane 3][lane 32][16 B]; warp w of row group rg owns 16-row tile rg * NW + w; ks owns units [u0(ks), u1(ks));
    stages past a CTA's count are zero padding (never read)."""
    n, k = wv.shape
    nw = 4 * nwg
    assert n % (64 * nwg) == 0 and k % 64 == 0
    nrg, nu = n // (64 * nwg), k // 64
    ncta, nrt = nrg * s, n // 16
    spl = vds_splits(nu, s)
    assert max(b - a for a, b in spl) <= nstgm, "split exceeds nstgm"
    if bases is None:
        bases = [vds_window_base(wv)] * nrg
    dev = wv.device
    base_rt = torch.tensor([bases[rt // nw] for rt in range(nrt)], dtype=torch.int32, device=dev).view(nrt, 1, 1, 1, 1)
    fr = vds_frag(wv)                                     # [nrt, nu, 32, 4, 8]
    lo = fr & 0xFF
    h7 = (fr >> 8) & 0x7F
    sg = (fr >> 15) & 1
    d = h7 - base_rt
    isesc = (d < 0) | (d > 7)
    d = torch.where(isesc, torch.zeros_like(d), d)
    dA, dB, sA, sB = d[..., :4], d[..., 4:], sg[..., :4], sg[..., 4:]
    sBr = sB[..., [3, 0, 1, 2]]
    byte = (dA | (sBr << 3) | (dB << 4) | (sA << 7)).to(torch.int64)
    cw = _i32(byte[..., 0] | (byte[..., 1] << 8) | (byte[..., 2] << 16) | (byte[..., 3] << 24))
    lo64 = lo.to(torch.int64)
    l0 = _i32(lo64[..., 0] | (lo64[..., 1] << 8) | (lo64[..., 2] << 16) | (lo64[..., 3] << 24))
    l1 = _i32(lo64[..., 4] | (lo64[..., 5] << 8) | (lo64[..., 6] << 16) | (lo64[..., 7] << 24))
    p0 = torch.stack([l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]], dim=-1)
    p1 = torch.stack([l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]], dim=-1)
    units = torch.stack([p0, p1, cw], dim=2).view(nrg, nw, nu, 3, 32, 4)   # [rg, w, u, 3, 32, 4]
    del fr, lo, d, dA, dB, sA, sB, sBr, byte, cw, lo64, l0, l1, p0, p1
    stream = torch.zeros((nstgm, s, nrg, nw, 3, 32, 4), dtype=torch.int32, device=dev)
    for ks, (a, b) in enumerate(spl):
        stream[: b - a, ks] = units[:, :, a:b].permute(2, 0, 1, 3, 4, 5)
    v = stream.view(-1)
    pos = isesc.nonzero()
    nesc = int(pos.shape[0])
    G = ncta * nw
    esc = torch.full((G * ESC_CAP,), -1, dtype=torch.int32, device=dev)
    if nesc:
        rt, u, ln, qq, jv = (pos[:, i] for i in range(5))
        rg, wi = rt // nw, rt % nw
        u0s = torch.tensor([a for a, _ in spl], device=dev)
        ksi = torch.bucketize(u, torch.tensor([b for _, b in spl], device=dev), right=True)
        st = u - u0s[ksi]
        gw = (ksi * nrg + rg) * nw + wi
        x7 = (h7[rt, u, ln, qq, jv] ^ base_rt.view(-1)[rt]).to(torch.int64)
        ent = (st << 20) | (ln << 15) | ((4 * qq + (jv >> 1)) << 11) | ((jv & 1) << 10) | x7
        order = torch.argsort((gw * 4096 + st) * 16 + 4 * qq + (jv >> 1))
        gws, ents = gw[order], ent[order]
        cnt = torch.bincount(gws, minlength=G)
        if int(cnt.max()) > ESC_CAP:
            raise ValueError(f"escape cap: a warp has {int(cnt.max())} > {ESC_CAP} escapes")
        rank = torch.arange(nesc, device=dev) - (torch.cumsum(cnt, 0) - cnt)[gws]
        esc[gws * ESC_CAP + rank] = _i32(ents)
    return {"w": v, "esc": esc, "tbs": vds_tbs(bases), "bases": bases, "nrg": nrg, "ncta": ncta, "nesc": nesc, "nw": nw,
            "nstgm": nstgm, "s": s, "spl": spl}


def vds_decode_check(wv, pk):
    """emulate the kernel's decode + escape patches on the packed stream (int64 torch) and compare with the A fragments
    the RS-wgmma expects (bit-exact), for every real stage of every CTA."""
    nw, s, nstgm, nrg = pk["nw"], pk["s"], pk["nstgm"], pk["nrg"]
    P = (pk["w"].view(nstgm, s, nrg, nw, 3, 32, 4).to(torch.int64) & 0xFFFFFFFF)
    L = torch.cat([P[..., 0, :, :], P[..., 1, :, :]], dim=-1)          # [.., 32, 8]
    Cw = P[..., 2, :, :]                                                  # [.., 32, 4]
    tb = torch.tensor([b * 0x01010101 for b in pk["bases"]], dtype=torch.int64, device=P.device).view(1, 1, nrg, 1, 1)

    def bp(x, y, sel):
        src = [(x >> (8 * i)) & 0xFF for i in range(4)] + [(y >> (8 * i)) & 0xFF for i in range(4)]
        o = torch.zeros_like(x)
        for i in range(4):
            o = o | (src[(sel >> (4 * i)) & 7] << (8 * i))
        return o
    regs = []
    for q in range(4):
        c = Cw[..., q]
        h0 = ((c & 0x87878787) + tb) & 0xFFFFFFFF
        rot = ((c >> 4) | (c << 28)) & 0xFFFFFFFF
        h1 = ((rot & 0x87878787) + tb) & 0xFFFFFFFF
        regs += [bp(L[..., 2 * q], h0, 0x5140), bp(L[..., 2 * q], h0, 0x7362), bp(L[..., 2 * q + 1], h1, 0x5140),
                 bp(L[..., 2 * q + 1], h1, 0x7362)]
    A = torch.stack(regs, dim=-1)                                         # [st, ks, rg, w, lane, 16]
    e = pk["esc"].to(torch.int64) & 0xFFFFFFFF
    ok = e != 0xFFFFFFFF
    if bool(ok.any()):
        gw = torch.arange(e.numel(), device=e.device)[ok] // ESC_CAP
        ev = e[ok]
        st, ln, r, hf, x7 = ev >> 20, (ev >> 15) & 31, (ev >> 11) & 15, (ev >> 10) & 1, ev & 127
        c, wi = gw // nw, gw % nw
        ksi, rg = c // nrg, c % nrg
        idx = ((((st * s + ksi) * nrg + rg) * nw + wi) * 32 + ln) * 16 + r
        mask = torch.zeros(A.numel(), dtype=torch.int64, device=A.device)
        mask.index_add_(0, idx, x7 << (8 + 16 * hf))   # the two halves of a register patch disjoint bits
        A = (A.reshape(-1) ^ mask).view(A.shape)
    fr = vds_frag(wv).to(torch.int64)                                   # [nrt, nu, 32, 4, 8]
    exp = (fr[..., 0::2] | (fr[..., 1::2] << 16)).reshape(fr.shape[0], fr.shape[1], 32, 16)
    exp = exp.view(nrg, nw, -1, 32, 16)
    good = True
    for ks, (a, b) in enumerate(pk["spl"]):
        good &= bool(torch.equal(A[: b - a, ks], exp[:, :, a:b].permute(2, 0, 1, 3, 4)))
    return good


def lab_weights(dev, gen=None):
    """the reference's dense recipe (randn / sqrt(k), bf16) + the cat = [pooler; router hi; router lo] (router FP32 rows at the
    reference's ROUTER_STD, split hi / lo as the layer does)."""
    def rn(shape):
        return torch.randn(shape, generator=gen, device=dev) if gen is not None else torch.randn(shape, device=dev)
    W = {WNAME[p]: (rn(PROJ[p]) / math.sqrt(PROJ[p][1])).to(torch.bfloat16) for p in WNAME}
    router = 0.045 * rn((256, 8192))
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    W["cat"] = torch.cat((W["pooler"], hi, lo), dim=0).contiguous()
    return W


def proj_weight(W, proj):
    return W["cat"] if proj == "cat" else W[WNAME[proj]]


def proj_bases(w, proj, nwg):
    return vds_bases(w, nwg, [4096, 4352] if proj == "cat" else None)


def selfcheck():
    torch.manual_seed(7)
    W = lab_weights("cpu")
    W["pooler"].view(torch.int16)[0, :5] = torch.tensor([0, 1, -32768, 0x0080, 0x7F7F], dtype=torch.int16)
    allok = True
    for (proj, t_), cfgs in CONFIGS.items():
        w = proj_weight(W, proj)
        for kid, s, label in cfgs:
            nn, nwg, nst, nstgm, out, cl_ = VDS_INSTS[kid]
            t0 = time.time()
            pk = vds_pack(w, nwg, s, nstgm, proj_bases(w, proj, nwg))
            ok = vds_decode_check(w, pk)
            allok &= ok
            print(f"[vds] selfcheck {proj} T{t_} kid {kid} S {s}: {pk['w'].numel() * 4 / 1e6:.1f} MB (padded) bases "
                  f"{sorted(set(pk['bases']))} tbs-runs {pk['tbs'][3:]} {pk['nesc']} escapes, decode bit-exact {ok} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    print("[vds] selfcheck", "OK" if allok else "FAILED", flush=True)
    return allok


# ---------------------------------------------------------------- shipped references (the layer's own functions, AST-extracted)
def layer_extract(path, prefixes, extra=()):
    """top-level definitions of our own layer file whose names start with prefixes (no sglang import)."""
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name and (name.startswith(prefixes) or name in extra):
            nodes.append(node)
    ns = {"torch": torch, "time": time, "hashlib": hashlib, "math": math, "HIDDEN": 8192, "D_EXPERT": 4096,
          "UNPOOL_SLICE": 1024, "SLICED_UNPOOL_MAX_TOKENS": 8, "V12_ZBUF": True, "R1_UNPOOL_INTO_ZEROS": True,
          "N_EXPERTS": 256, "TOP_K": 8, "K2A_M_MAX": 4096, "RMS_EPS": 1e-5, "_pt1_wait": lambda name: None, "_KS1": {}}
    ns["_k3_disable"] = lambda state, key, value, piece, ex: state.__setitem__(key, value)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)   # lab tool, our own layer source
    return ns


def gd_reference(layer_path, W, routes):
    ns = layer_extract(layer_path, ("GD_", "_GD", "_gd"), ("K2A_BUCKETS", "_k2a_bucket_of", "SU1_K3_TRIG_AT"))
    ns["GD_ROUTE"].clear()
    ns["GD_ROUTE"].update(routes)
    ns["GD_PDL"] = {k: False for k in ("qkv", "o", "pooler", "unpooler")}
    ns["GD_PREPARE_BY_S"] = 1e9
    state = {"dense": {k: W[k] for k in WNAME.values()},
             "v12_zbuf": torch.zeros((32, 8192), dtype=torch.bfloat16, device="cuda")}
    ns["_gd_prepare"](state, time.time())
    return ns, state


def dg1_reference(layer_path, W):
    ns = layer_extract(layer_path, ("DG1_", "_DG1", "_dg1", "dg1_"))
    ns["DG1_ROUTE"] = {("cat", 32): True}
    ns["DG1_PDL"] = False
    state = {"dense": {k: W[k] for k in WNAME.values()}, "v8": {"wcat": W["cat"]}}
    ns["_dg1_prepare"](state, time.time())
    return ns, state


# ---------------------------------------------------------------- timing
def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_many(graphs, flush, sink, reps, warm=None):
    """judge-style: read-sweep L2 flush before every replay (then x re-read into L2 when warm is given: the layer's x is the
    predecessor's fresh output), rotating order each rep, medians (us) per graph."""
    n = len(graphs)
    ts = [[] for _ in range(n)]
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(reps + 2):
        order = [(i + r) % n for i in range(n)]
        if r % 2:
            order = order[::-1]
        for i in order:
            torch.sum(flush, dim=0, out=sink)
            if warm is not None:
                warm()
            e0.record()
            graphs[i].replay()
            e1.record()
            torch.cuda.synchronize()
            if r >= 2:
                ts[i].append(e0.elapsed_time(e1) * 1e3)
    return [statistics.median(t) for t in ts]


def check(y, ref, T, n):
    """GD's judge recipe: rel L2 and worst row vs the FP64 reference rounded to BF16 (FP32 outputs: vs FP64 directly)."""
    got = y[:T, :n].double()
    rf = ref[:T].to(torch.bfloat16).double() if y.dtype == torch.bfloat16 else ref[:T]
    dd = got - rf
    rn = rf.norm(dim=-1)
    tot = float(dd.norm() / rf.norm().clamp(min=1e-30))
    row = float((dd.norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
    return tot, row, bool(torch.isfinite(got).all())


def main():
    if int(arg("selfcheck", "0")):
        sys.exit(0 if selfcheck() else 1)
    projs = arg("projs", "pooler,cat,unpooler,qkv,o").split(",")
    ts_ = [int(v) for v in arg("ts", "32,16").split(",")]
    reps = int(arg("reps", "15"))
    rounds = int(arg("rounds", "6"))
    xhot = int(arg("xhot", "1"))
    layer = arg("layer", str(ROOT / "kernels" / "lab" / "k3_slim_3_ab.py"))
    out_path = Path(arg("out", str(ROOT / "box_out" / "vds3" / "lab.json")))
    mod = vds_build(VDS_INSTS, verbose=bool(int(arg("verbose", "0"))))
    infos = {i: list(mod.vds_info(i)) for i in range(len(VDS_INSTS))}
    for i, v in infos.items():
        print(f"[vds] inst {i} {VDS_INSTS[i]}: smem {v[0]} regs {v[1]} local {v[2]} B occupancy {v[3]} SBW {v[4]} "
              f"nstgm {v[5]} max-clusters {v[7]}", flush=True)
    if int(arg("build-only", "0")):
        return
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1)
    W = lab_weights(dev, gen)
    flush = torch.empty((64 << 20,), dtype=torch.float32, device=dev).uniform_().view(4096, -1)   # 256 MB read sweep
    sink = torch.empty((flush.shape[1],), dtype=torch.float32, device=dev)
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    stamps = torch.zeros((5 * 160 + 2 * 160 * 16,), dtype=torch.int64, device=dev)
    ws = torch.zeros((4608 * 32 * 2,), dtype=torch.float32, device=dev)
    cnt = torch.zeros((256,), dtype=torch.int32, device=dev)
    gd_state, dg1_state = None, None
    if int(arg("gd", "1")) and any(p in projs for p in GD_LAB_ROUTE_PROJS()):
        try:
            routes = {key: rt for key, rt in GD_LAB_ROUTE.items() if key[0] in projs and key[1] in ts_}
            if routes:
                _, gd_state = gd_reference(layer, W, routes)
        except Exception as ex:
            print(f"[vds] GD reference unavailable: {ex!r}", flush=True)
    if int(arg("dg1", "1")) and "cat" in projs and 32 in ts_:
        try:
            _, dg1_state = dg1_reference(layer, W)
        except Exception as ex:
            print(f"[vds] DG1 reference unavailable: {ex!r}", flush=True)
    results = {"insts": {str(i): infos[i] for i in infos}, "rows": []}
    gnull = graph_of(lambda: mod.vds_run(-1, err, err, err, err, err, stamps, 0, 0, [0] * 5, 0, 0, 0, 0))
    for proj in projs:
        n, k = PROJ[proj]
        w = proj_weight(W, proj)
        f32 = proj == "cat"
        odt = torch.float32 if f32 else torch.bfloat16
        packs = {}
        for t_ in ts_:
            for kid, s, _ in CONFIGS.get((proj, t_), []):
                nn, nwg, nst, nstgm, o_, cl_ = VDS_INSTS[kid]
                key = (nwg, s, nstgm)
                if key not in packs:
                    t0 = time.time()
                    pk = vds_pack(w, nwg, s, nstgm, proj_bases(w, proj, nwg))
                    ok = vds_decode_check(w, pk)
                    print(f"[vds] pack {proj} nwg {nwg} s {s} nstgm {nstgm}: {pk['nesc']} escapes, bases "
                          f"{sorted(set(pk['bases']))}, decode bit-exact {ok} ({time.time() - t0:.1f}s)", flush=True)
                    packs[key] = pk if ok else None
                    torch.cuda.empty_cache()
        for m in ts_:
            cfgs = CONFIGS.get((proj, m))
            if not cfgs:
                continue
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ref = x.double() @ w.double().t()
            warm = (lambda x=x: torch.sum(x, dim=0)) if xhot else None
            if f32:
                y_cub = torch.empty((m, n), dtype=torch.float32, device=dev)
                arms = {"cublas": (lambda x=x, y=y_cub: y.copy_(torch.mm(x, w.t(), out_dtype=torch.float32)))}
            else:
                y_cub = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                arms = {"cublas": (lambda x=x, y=y_cub: torch.mm(x, w.t(), out=y))}
            ys, meta = {"cublas": (y_cub, 0)}, {}
            if gd_state is not None and gd_state.get("gd") is not None and (proj, m) in gd_state["gd"]["route"]:
                y_gd = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                r = gd_state["gd"]["route"][(proj, m)]

                def run_gd(x=x, y=y_gd, r=r):
                    r["mod"].gd_run(r["inst"], r["w"], r["esc"], x, y, x.shape[0], r["K"], y.stride(0), r["tb"], r["nbx"], 1, 2)
                arms["gd_" + r["desc"]] = run_gd
                ys["gd_" + r["desc"]] = (y_gd, 0)
            if f32 and m == 32 and dg1_state is not None and dg1_state.get("dg1") is not None:
                st = dg1_state["dg1"]
                r = st["route"][("cat", 32)]
                y_dg = torch.empty((m, n), dtype=torch.float32, device=dev)

                def run_dg(x=x, y=y_dg, st=st, r=r):
                    st["mod"].dg1_run(r["kid"], x, st["w"]["cat"], y, st["err"], r["S"], 0, 0, r["N"], 0)
                arms["dg1_cat"] = run_dg
                ys["dg1_cat"] = (y_dg, 0)
            for ci, (kid, s, label) in enumerate(cfgs):
                nn, nwg, nst, nstgm, o_, cl_ = VDS_INSTS[kid]
                pk = packs[(nwg, s, nstgm)]
                if pk is None:
                    continue
                arms_ = (("_pf0", 0, 0), ("_pf2", 0, 2), ("_pfall", 0, 99), ("_pf0_nomma", 2, 0), ("_pf0_wonly", 4, 0)) \
                    if ci == 0 else (("_pf0", 0, 0), ("_pfall", 0, 99))
                for tag, fl, pf in arms_:
                    y = torch.zeros((m, n), dtype=odt, device=dev)
                    name = f"vds{kid}_{label}{tag}"

                    def run(kid=kid, x=x, y=y, pk=pk, fl=fl, pf=pf, s=s):
                        mod.vds_run(kid, pk["w"], pk["esc"], x, y, err, stamps, y.stride(0), pk["nrg"], pk["tbs"], pf, 1, fl, 0)
                    arms[name] = run
                    ys[name] = (y, fl)
                    meta[name] = (kid, pk, fl, pf, s)
            for name, fn in arms.items():
                fn()
            torch.cuda.synchronize()
            e = int(err[0].item())
            chk = {name: check(y, ref, m, n) for name, (y, fl) in ys.items() if not (fl & 6)}
            print(f"[vds] {proj} T{m}: err {e} ws-clean {int(ws.ne(0).sum())} cnt-clean {int(cnt.ne(0).sum())}; "
                  "(rel, worst row, finite) vs fp64(->bf16): " + " ".join(f"{k2}=({v[0]:.1e},{v[1]:.1e},{v[2]})"
                                                                         for k2, v in chk.items()), flush=True)
            if e:
                print(f"[vds] {proj} T{m}: bounded wait timed out (err {e}) -> no timing", flush=True)
                results["rows"].append({"proj": proj, "m": m, "err": e, "chk": chk})
                err.zero_()
                ws.zero_()
                cnt.zero_()
                continue
            for name, (kid, pk, fl, pf, s) in meta.items():   # partial T (zero-filled x rows, masked stores)
                if fl or not name.endswith("_pf0"):
                    continue
                for tp in ((17, 31) if m == 32 else (9, 13)):
                    yp = torch.full((tp, n), 7.0, dtype=odt, device=dev)
                    mod.vds_run(kid, pk["w"], pk["esc"], x[:tp].contiguous(), yp, err, stamps, yp.stride(0), pk["nrg"], pk["tbs"],
                                0, 0, 0, 0)
                    torch.cuda.synchronize()
                    c2 = check(yp, ref, tp, n)
                    print(f"[vds] {proj} T{m} {name}: partial T={tp}: rel {c2[0]:.1e} row {c2[1]:.1e} err {int(err[0].item())}",
                          flush=True)
            names = list(arms.keys())
            graphs = [graph_of(arms[nm]) for nm in names] + [gnull]
            meds = {nm: [] for nm in names + ["null"]}
            for rd in range(rounds):
                tt = time_many(graphs, flush, sink, reps, warm)
                for nm, v in zip(names + ["null"], tt):
                    meds[nm].append(v)
            fl0 = statistics.median(meds["null"])
            row = {"proj": proj, "m": m, "err": e, "chk": chk, "floor_us": fl0, "xhot": xhot, "arms": {}, "stamps_us": {}, "wait": {}}
            cub = statistics.median(meds["cublas"]) - fl0
            mb12 = n * k * 1.5 / 1e6
            for nm in names:
                med = statistics.median(meds[nm]) - fl0
                row["arms"][nm] = {"net_us": med, "rounds": meds[nm], "x_vs_cublas": cub / med, "tbs_p12": mb12 / med}
            for nm, (kid, pk, fl, pf, s) in meta.items():
                stamps.zero_()
                torch.sum(flush, dim=0, out=sink)
                if warm is not None:
                    warm()
                mod.vds_run(kid, pk["w"], pk["esc"], x, ys[nm][0], err, stamps, ys[nm][0].stride(0), pk["nrg"], pk["tbs"], pf, 0,
                            fl | 8, 0)
                torch.cuda.synchronize()
                nc, nw = pk["ncta"], pk["nw"]
                sp = stamps[:5 * nc].view(-1, 5).double()
                t0 = float(sp[:, 0].min())
                q = lambda v: (round(float(v.min()), 2), round(float(v.median()), 2), round(float(v.max()), 2))  # noqa: E731
                row["stamps_us"][nm] = {nm2: q((sp[:, i] - t0) / 1e3) for i, nm2 in enumerate(("entry", "x", "first", "loop", "end"))}
                wt = stamps[5 * 160: 5 * 160 + 2 * nc * nw].view(-1, 2).double()
                row["wait"][nm] = {"wait_frac_med": round(float((wt[:, 0] / wt[:, 1].clamp(min=1)).median()), 3),
                                   "loop_clk_med": float(wt[:, 1].median())}
            for name, fn in arms.items():
                fn()
            torch.cuda.synchronize()
            row["err_after"] = int(err[0].item())
            row["chk_after"] = {name: check(y, ref, m, n) for name, (y, fl) in ys.items() if not (fl & 6)}
            row["ws_clean_after"] = int(ws.ne(0).sum())
            results["rows"].append(row)
            print(f"[vds] {proj} T{m}: floor {fl0:.2f} us (x {'hot' if xhot else 'cold'}); net us (x vs cuBLAS, TB/s on P12 "
                  "bytes) | stamps x / first / loop / end (min-med-max) | full-wait fraction:", flush=True)
            for nm in names:
                a = row["arms"][nm]
                sx = row["stamps_us"].get(nm)
                st = "" if sx is None else (f" | x {sx['x'][1]:.2f} first {sx['first'][1]:.2f} loop {sx['loop'][0]:.2f}-"
                                            f"{sx['loop'][1]:.2f}-{sx['loop'][2]:.2f} end {sx['end'][0]:.2f}-{sx['end'][1]:.2f}-"
                                            f"{sx['end'][2]:.2f} | wait {row['wait'][nm]['wait_frac_med']:.2f}")
                print(f"    {nm:30s} {a['net_us']:7.2f}  x{a['x_vs_cublas']:.3f}  {a['tbs_p12']:.2f} TB/s{st}", flush=True)
            print(f"[vds] {proj} T{m}: err after {row['err_after']}, ws dirty {row['ws_clean_after']}; recheck "
                  + " ".join(f"{k2}={v[0]:.1e}" for k2, v in row["chk_after"].items()), flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=1))
    print("[vds] done", flush=True)


def GD_LAB_ROUTE_PROJS():
    return {p for p, _ in GD_LAB_ROUTE}


if __name__ == "__main__":
    main()

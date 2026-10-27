"""vd_lab.py: VD1 = dense BF16 projections at m = 128 / 256 (verify16 / verify32) on SS-wgmma, step-0 microbench vs cuBLAS.

Design (notes/study/vd_dense.md):
  * swap-AB: y^T[N, T] = W[N, K] . x^T; wgmma m64nBNk16 SS, A = weight tile [64 rows x 64 k] per consumer warpgroup,
    B = x tile [BN tokens x 64 k] shared by every warpgroup of the CTA, FP32 accumulate, BN = 128 / 256 = ALL tokens, so each
    weight byte is read once and x is re-read once per BM = 64 * NWG weight rows.
  * Weights never go through registers: a producer lane streams one BM x 128 B tile per stage into an NST-deep ring.
    Default source = a lossless repack of W into pre-swizzled (SW128) tiles in FRONT-major order (tile (s, rb) at
    (s * nrb + rb) * BM * 128 B), so at any moment the grid reads one contiguous front (the stream lab's interleaved
    order: t0 2.2 vs 3.05 us); one 1-D cp.async.bulk per tile. Arms: slab order, and W straight from row-major via a
    per-call 2-D tensor map (cuBLAS's access pattern). x through a per-call 2-D tensor map (SW128, zero fill past T).
  * Split-K only inside a CL-CTA cluster (CL <= 3, one wave at 1 CTA / SM), reduced over DSMEM: every rank parks its
    fragments in its own (drained) ring, cluster barrier, rank r sums token blocks [r NB / CL, (r + 1) NB / CL) from all
    ranks, stages them and writes 16-B rows. No global workspace, no second kernel.
  * PDL-ready: the first NST weight tiles are issued before griddepcontrol.wait, x after (in-layer the weights stream
    under the previous kernel's tail). Bounded waits everywhere (R9): a hang sets err[0], never traps.
  Lab flags: 1 exit at entry, 2 no wgmma, 4 no x (W-only stream with 2|4), 8 L2 evict_first on W, 16 evict_last on x,
    32 per-CTA globaltimer stamps, 64 W via row-major 2-D tensor map, 128 launch_dependents after the main loop.

    python3 kernels/lab/vd_lab.py [--ts=128,256] [--projs=unpooler,pooler,qkv,o,cat] [--reps=21] [--rounds=8] [--build-only=1]
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

SMEM_MAX = 232448   # 227 KB opt-in dynamic smem per block on sm_90

# (NWG, BN, NST, CL, F32, RBD): BM = 64 NWG weight rows x BN tokens per CTA, NST ring stages, CL-CTA cluster split-K,
# RBD 1 = dedicated DSMEM receive buffer (no barrier after the main loop), 0 = receive buffer aliases the drained ring
VD_INSTS = [
    (1, 128, 9, 1, 0, 0),   # 0  BM  64 full K            unpooler / o: 128 CTAs (x 2 x W)
    (1, 128, 8, 2, 0, 1),   # 1  BM  64 CL 2 dedicated    pooler 128
    (1, 128, 9, 2, 0, 0),   # 2  BM  64 CL 2 aliased      pooler 128
    (1, 128, 8, 3, 0, 1),   # 3  BM  64 CL 3 dedicated    qkv 108
    (1, 128, 9, 3, 0, 0),   # 4  BM  64 CL 3 aliased      qkv 108
    (2, 128, 6, 2, 0, 1),   # 5  BM 128 CL 2 dedicated    unpooler / o 128 (x 1 x W)
    (2, 128, 7, 2, 0, 0),   # 6  BM 128 CL 2 aliased      unpooler / o 128
    (2, 128, 5, 3, 1, 1),   # 7  BM 128 CL 3 FP32 ded.    cat 108
    (2, 128, 7, 3, 1, 0),   # 8  BM 128 CL 3 FP32 alias   cat 108
    (2, 256, 4, 2, 0, 0),   # 9  T=256 BM 128 CL 2        unpooler / o 128
    (2, 256, 4, 3, 1, 0),   # 10 T=256 BM 128 CL 3 FP32   cat 108
    (2, 256, 4, 3, 0, 0),   # 11 T=256 BM 128 CL 3        pooler 96
]

VD_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t vd_run(int64_t kid, torch::Tensor x, torch::Tensor wpk, torch::Tensor wraw, torch::Tensor y, torch::Tensor err,
               torch::Tensor stamps, int64_t nrb, int64_t w_rb, int64_t w_s, int64_t flags, int64_t pdl, int64_t pf);
std::vector<int64_t> vd_info(int64_t kid);
"""

VD_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstring>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace vd {

constexpr long long WAIT_BUDGET = 2000000000LL;   // ~1 s of clock64: a hang becomes err[0] = 1 (R9)
enum : int { F_EXIT = 1, F_NOMMA = 2, F_NOX = 4, F_EFW = 8, F_ELX = 16, F_STAMP = 32, F_WMAP = 64, F_TRIG = 128, F_WAIT0 = 256,
             F_DYN = 512 };

struct Params {
  const uint8_t* w;          // packed tiles (bulk mode): tile (rb, s) at w + rb * w_rb + s * w_s (bytes)
  void* y;                   // [T, ldy] bf16 or f32
  int* err;
  unsigned long long* stamps;   // [grid][4] (flag 32)
  long long w_rb, w_s;
  int nks, T, ldy, flags, pf;   // pf: weight stages issued before griddepcontrol.wait (the rest go x-first)
  int rot;                      // K-start rotation per row block (host arg pf >> 8)
};

// RBD 1: the DSMEM receive buffer is dedicated smem after the ring (no barrier after the main loop);
// RBD 0: it aliases the drained ring, one cluster barrier after the main loop
template <int NWG, int BN, int NST, int CL, int F32, int RBD>
struct Cfg {
  static constexpr int BM = 64 * NWG;
  static constexpr int WB = BM * 128;              // weight tile bytes (BM rows x 64 bf16)
  static constexpr int XB = BN * 128;              // x tile bytes (BN tokens x 64 bf16)
  static constexpr int SB = WB + XB;               // stage bytes (multiple of 1024)
  static constexpr int NTH = NWG * 128;            // consumer threads
  static constexpr int THREADS = NTH + 32;         // + one producer warp
  static constexpr int NACC = BN / 2;              // fp32 accumulators per consumer thread
  static constexpr int NB = BN / 8;                // 8-token blocks
  static constexpr int JMAX = (NB + CL - 1) / CL;  // most token blocks a rank owns
  static constexpr int RB_BYTES = CL > 1 ? (CL - 1) * JMAX * NTH * 16 : 0;   // [src slot][block][thread] float4
  static constexpr int RB_OFF = RBD ? NST * SB : 0;
  static constexpr int BAR_OFF = NST * SB + (RBD ? RB_BYTES : 0);
  static constexpr int SMEM = 1024 + BAR_OFF + 16 * NST + 64;
  static constexpr int CTR_OFF = BAR_OFF + 16 * NST + 8;    // dynamic split: stage counter (rank 0's is used)
  static constexpr int ENDF_OFF = CTR_OFF + 4;               // dynamic split: ring index of this rank's end marker
  static constexpr int OB = F32 ? 4 : 2;           // output element bytes
  static constexpr int SROW = F32 ? (BM + 4) : (BM + 8);   // staging row pitch (elements), 16-B multiple
  static constexpr int S_OFF = RBD ? 0 : RB_BYTES;  // staging tile [owned tokens][BM] in the drained ring
  static constexpr int S_BYTES = JMAX * 8 * SROW * OB;
  static_assert(SB % 1024 == 0, "stage alignment");
  static_assert(S_OFF + S_BYTES <= NST * SB, "epilogue buffers alias the ring");
  static_assert(SMEM <= 232448, "smem");
  static_assert(CL >= 1 && CL <= NB, "cluster split");
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
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
__device__ __forceinline__ uint64_t pol_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t pol_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_h(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma2d_h(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t mapa(uint32_t local_addr, uint32_t rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(local_addr), "r"(rank));
  return ra;
}
__device__ __forceinline__ void st_async_f4(uint32_t raddr, float a, float b, float c, float d, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n"
               :: "r"(raddr), "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)), "r"(__float_as_uint(d)),
                  "r"(rbar) : "memory");
}
__device__ __forceinline__ int atom_add_cl(uint32_t raddr, int v) {
  int old;
  asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0, [%1], %2;\n" : "=r"(old) : "r"(raddr), "r"(v) : "memory");
  return old;
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
// SM90 smem matrix descriptor: K-major, 128-B swizzle, 8-row groups 1024 B apart (SBO), LBO unused (1)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db);
@WGSPECS@

template <int NWG, int BN, int NST, int CL, int F32, int RBD>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
vd_kernel(const __grid_constant__ CUtensorMap tmx, const __grid_constant__ CUtensorMap tmw, const Params P) {
  using C = Cfg<NWG, BN, NST, CL, F32, RBD>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t s_raw = sa(smem_raw);
  const uint32_t sbase = (s_raw + 1023u) & ~1023u;
  uint8_t* gbase = smem_raw + (sbase - s_raw);
  const uint32_t full0 = sbase + C::BAR_OFF, empty0 = full0 + 8 * NST, rbar = full0 + 16 * NST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int flags = P.flags;
  if (flags & F_EXIT) return;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int s0 = ((int)crank * P.nks) / CL, s1 = (((int)crank + 1) * P.nks) / CL;
  const int nk = s1 - s0;
  const int jlo = CL > 1 ? ((int)crank * C::NB) / CL : 0;          // token blocks this rank finishes
  const int jhi = CL > 1 ? (((int)crank + 1) * C::NB) / CL : C::NB;
  const bool stamp = (flags & F_STAMP) != 0;
  unsigned long long t_entry = 0;
  if (stamp && tid == 0) t_entry = gtime();
  bool dead = false;

  if (tid == 0) {
#pragma unroll 1
    for (int i = 0; i < NST; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    if constexpr (CL > 1) {
      mbar_init(rbar, 1);
      mbar_expect_tx(rbar, (uint32_t)((CL - 1) * (jhi - jlo) * C::NTH * 16));   // the peers' pushes of my blocks
    }
    if (CL > 1 && (flags & F_DYN) != 0) {   // dynamic split only: a cluster-scope fence costs ~0.5 us at entry (R-vd-3)
      *reinterpret_cast<volatile int*>(gbase + C::CTR_OFF) = 0;
      *reinterpret_cast<volatile int*>(gbase + C::ENDF_OFF) = 0x7fffffff;
      asm volatile("fence.acq_rel.cluster;\n" ::: "memory");   // the counter init is visible before the cluster arrive
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if constexpr (CL > 1) cluster_arrive_relaxed();   // phase 0: "my barriers are initialized" (waited before any push)

  const bool dyn = CL > 1 && (flags & F_DYN) != 0;
  if (warp == NWG * 4) {
    // ------------------------------------------------------------------ producer (one lane)
    bool pw0 = false;   // lane 0 already did its phase-0 cluster wait (dynamic split)
    if (lane == 0) {
      const bool nox = (flags & F_NOX) != 0, wmap = (flags & F_WMAP) != 0;
      const bool efw = (flags & F_EFW) != 0, elx = (flags & F_ELX) != 0;
      const uint64_t pw = efw ? pol_first() : 0ull, px = elx ? pol_last() : 0ull;
      const uint32_t txb = (uint32_t)C::WB + (nox ? 0u : (uint32_t)C::XB);
      const uint8_t* wbase = P.w + (long long)rb * P.w_rb;
      asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmx)) : "memory");
      if (wmap) asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmw)) : "memory");
      // K rotation: row block rb walks its K range starting at stage (rb * rot) mod nk, so CTAs that run in lockstep
      // read different x tiles (128 SMs hitting the same L2 lines at once is a hotspot); 0 = no rotation
      const int off = P.rot > 0 ? (int)(((long long)rb * P.rot) % nk) : 0;
      auto kst = [&](int i) { int j = i + off; if (j >= nk) j -= nk; return s0 + j; };
      auto issue_w = [&](int i, uint32_t dst, uint32_t bar) {
        const int s = kst(i);
        if (wmap) { if (efw) tma2d_h(dst, &tmw, s * 64, rb * C::BM, bar, pw); else tma2d(dst, &tmw, s * 64, rb * C::BM, bar); }
        else { const void* src = wbase + (long long)s * P.w_s;
               if (efw) bulk_g2s_h(dst, src, C::WB, bar, pw); else bulk_g2s(dst, src, C::WB, bar); }
      };
      auto issue_x = [&](int i, uint32_t dst, uint32_t bar) {
        if (nox) return;
        const int s = kst(i);
        if (elx) tma2d_h(dst + C::WB, &tmx, s * 64, 0, bar, px); else tma2d(dst + C::WB, &tmx, s * 64, 0, bar);
      };
      if (!dyn) {
      const int npro = nk < NST ? nk : NST;
      const int pf = P.pf < npro ? P.pf : npro;
#pragma unroll 1
      for (int i = 0; i < pf; ++i) {   // a few weight stages before the PDL wait (they do not depend on the previous kernel)
        mbar_expect_tx(full0 + 8 * i, txb);
        issue_w(i, sbase + i * C::SB, full0 + 8 * i);
      }
      pdl_wait();
#pragma unroll 1
      for (int i = 0; i < pf; ++i) issue_x(i, sbase + i * C::SB, full0 + 8 * i);
#pragma unroll 1
      for (int i = pf; i < npro; ++i) {   // x first: stage 0 must not queue behind NST weight tiles
        mbar_expect_tx(full0 + 8 * i, txb);
        issue_x(i, sbase + i * C::SB, full0 + 8 * i);
        issue_w(i, sbase + i * C::SB, full0 + 8 * i);
      }
#pragma unroll 1
      for (int i = NST; i < nk; ++i) {
        const int slot = i % NST;
        const uint32_t dst = sbase + slot * C::SB, bar = full0 + 8 * slot;
        mbar_wait(empty0 + 8 * slot, (uint32_t)(((i / NST) - 1) & 1), P.err, dead);
        mbar_expect_tx(bar, txb);
        issue_x(i, dst, bar);
        issue_w(i, dst, bar);
      }
      } else if constexpr (CL > 1) {
        // dynamic split inside the cluster: rank r first takes stages r, r + CL, .. (NST of them, static), then claims the
        // rest from a counter in rank 0's smem (DSMEM atomic, one claim ahead), so the ranks finish together; an end
        // marker (ring index in smem + a plain arrive on that slot's full barrier) tells the consumers where the list ends
        const int nkt = P.nks;
        const int offd = P.rot > 0 ? (int)(((long long)rb * P.rot) % nkt) : 0;
        auto ks = [&](int c) { int j = c + offd; if (j >= nkt) j -= nkt; return j; };
        auto put_w = [&](int s, uint32_t dst, uint32_t bar) {
          if (wmap) { if (efw) tma2d_h(dst, &tmw, s * 64, rb * C::BM, bar, pw); else tma2d(dst, &tmw, s * 64, rb * C::BM, bar); }
          else { const void* src = wbase + (long long)s * P.w_s;
                 if (efw) bulk_g2s_h(dst, src, C::WB, bar, pw); else bulk_g2s(dst, src, C::WB, bar); }
        };
        auto put_x = [&](int s, uint32_t dst, uint32_t bar) {
          if (nox) return;
          if (elx) tma2d_h(dst + C::WB, &tmx, s * 64, 0, bar, px); else tma2d(dst + C::WB, &tmx, s * 64, 0, bar);
        };
        int npre = 0;
        while (npre < NST && (int)crank + CL * npre < nkt) ++npre;
        const int pf = P.pf < npre ? P.pf : npre;
#pragma unroll 1
        for (int i = 0; i < pf; ++i) { mbar_expect_tx(full0 + 8 * i, txb); put_w(ks((int)crank + CL * i), sbase + i * C::SB, full0 + 8 * i); }
        pdl_wait();
#pragma unroll 1
        for (int i = 0; i < pf; ++i) put_x(ks((int)crank + CL * i), sbase + i * C::SB, full0 + 8 * i);
#pragma unroll 1
        for (int i = pf; i < npre; ++i) {
          mbar_expect_tx(full0 + 8 * i, txb);
          put_x(ks((int)crank + CL * i), sbase + i * C::SB, full0 + 8 * i);
          put_w(ks((int)crank + CL * i), sbase + i * C::SB, full0 + 8 * i);
        }
        int iend = npre;
        if (npre == NST) {
          asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");   // phase 0: rank 0's counter is initialized
          pw0 = true;
          const uint32_t ctr = mapa(sbase + C::CTR_OFF, 0u);
          int c = atom_add_cl(ctr, 1) + CL * NST;
#pragma unroll 1
          for (int i = NST; ; ++i) {
            const int slot = i % NST;
            mbar_wait(empty0 + 8 * slot, (uint32_t)(((i / NST) - 1) & 1), P.err, dead);
            if (c >= nkt || dead) { iend = i; break; }
            const int cn = atom_add_cl(ctr, 1) + CL * NST;   // claim ahead: its latency hides behind this stage's issue
            const uint32_t dst = sbase + slot * C::SB, bar = full0 + 8 * slot;
            mbar_expect_tx(bar, txb);
            put_x(ks(c), dst, bar);
            put_w(ks(c), dst, bar);
            c = cn;
          }
        }
        *reinterpret_cast<volatile int*>(gbase + C::ENDF_OFF) = iend;
        mbar_arrive(full0 + 8 * (iend % NST));                // release: the consumers read the marker after their wait
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {
      if (!(lane == 0 && pw0)) asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");   // phase 0
      if constexpr (!RBD) {   // phase 1: rings drained
        asm volatile("barrier.cluster.arrive.release;\n" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");
      }
    }
    return;
  }

  // -------------------------------------------------------------------- consumers
  const int wg = warp >> 2, t = tid & 127;
  float acc[C::NACC];
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) acc[q] = 0.f;
  const bool nomma = (flags & F_NOMMA) != 0;
  const bool w0 = (flags & F_WAIT0) != 0;
  unsigned long long t_first = 0;
#pragma unroll 1
  for (int i = 0; dyn || i < nk; ++i) {
    const int slot = i % NST;
    mbar_wait(full0 + 8 * slot, (uint32_t)((i / NST) & 1), P.err, dead);
    if (dyn && (i == *reinterpret_cast<volatile int*>(gbase + C::ENDF_OFF) || dead)) break;   // this rank's end marker
    if (stamp && i == 0 && tid == 0) t_first = gtime();
    if (!nomma) {
      const uint32_t st = sbase + slot * C::SB;
      const uint64_t da = desc_sw128(st + wg * 8192), db = desc_sw128(st + C::WB);
#pragma unroll
      for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wgmma_ss<BN>(acc, da + 2 * kk, db + 2 * kk);
      wg_commit();
      if (w0) wg_wait<0>(); else wg_wait<1>();
#pragma unroll
      for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    }
    if (w0) { if (t == 0) mbar_arrive(empty0 + 8 * slot); }
    else if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NST));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
  unsigned long long t_loop = 0;
  if (stamp && tid == 0) t_loop = gtime();
  if (flags & F_TRIG) pdl_trigger();

  // -------------------------------------------------------------------- epilogue
  named_sync(1, C::NTH);   // every warpgroup is done with the ring
  const int g = lane >> 2, t4 = lane & 3;
  const int r0 = wg * 64 + (warp & 3) * 16 + g;     // this thread's first row within the tile (second: r0 + 8)
  uint8_t* stg = gbase + C::S_OFF;
  if constexpr (CL > 1) {
    // push every token block another rank finishes straight from registers into that rank's receive buffer
    // ([sender slot][block - its jlo][thread] float4), fire-and-forget st.async completing the receiver's rbar
    cluster_wait();                                     // phase 0: every rank's rbar is initialized
    if constexpr (!RBD) { cluster_arrive(); cluster_wait(); }   // phase 1: every rank's ring is drained (buffer aliases it)
    const uint32_t rbl = sbase + C::RB_OFF;
#pragma unroll
    for (int p = 0; p < CL; ++p) {
      if (p == (int)crank) continue;
      const int plo = (p * C::NB) / CL, phi = ((p + 1) * C::NB) / CL;
      const int slot = (int)crank < p ? (int)crank : (int)crank - 1;
      const uint32_t rbuf = mapa(rbl, (uint32_t)p), rbr = mapa(rbar, (uint32_t)p);
#pragma unroll
      for (int j = 0; j < C::NB; ++j) {
        if (j < plo || j >= phi) continue;
        st_async_f4(rbuf + (uint32_t)(((slot * C::JMAX + (j - plo)) * C::NTH + tid) * 16),
                    acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3], rbr);
      }
    }
    mbar_wait_cl(rbar, 0, P.err, dead);                 // the other ranks' copies of my blocks have landed
    const float4* RB = reinterpret_cast<const float4*>(gbase + C::RB_OFF);
#pragma unroll
    for (int j = 0; j < C::NB; ++j) {
      if (j < jlo || j >= jhi) continue;
      float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
      for (int s = 0; s < CL - 1; ++s) {
        const float4 q = RB[(s * C::JMAX + (j - jlo)) * C::NTH + tid];
        v0 += q.x; v1 += q.y; v2 += q.z; v3 += q.w;
      }
      acc[4 * j] = v0; acc[4 * j + 1] = v1; acc[4 * j + 2] = v2; acc[4 * j + 3] = v3;
    }
  }
  // stage owned token blocks [jlo, jhi) as [token][BM rows] (pitch SROW)
#pragma unroll
  for (int j = 0; j < C::NB; ++j) {
    if (j < jlo || j >= jhi) continue;
    const int tt = 8 * (j - jlo) + 2 * t4;
    if constexpr (F32) {
      float* s = reinterpret_cast<float*>(stg);
      s[tt * C::SROW + r0] = acc[4 * j];
      s[(tt + 1) * C::SROW + r0] = acc[4 * j + 1];
      s[tt * C::SROW + r0 + 8] = acc[4 * j + 2];
      s[(tt + 1) * C::SROW + r0 + 8] = acc[4 * j + 3];
    } else {
      __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(stg);
      s[tt * C::SROW + r0] = __float2bfloat16_rn(acc[4 * j]);
      s[(tt + 1) * C::SROW + r0] = __float2bfloat16_rn(acc[4 * j + 1]);
      s[tt * C::SROW + r0 + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
      s[(tt + 1) * C::SROW + r0 + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
  named_sync(1, C::NTH);
  {
    constexpr int CH = C::BM * C::OB / 16;            // 16-B chunks per token row
    const int tlo = 8 * jlo;
    int thi = 8 * jhi;
    if (thi > P.T) thi = P.T;
    const int ntok = thi > tlo ? thi - tlo : 0;
    uint8_t* yb = reinterpret_cast<uint8_t*>(P.y) + (size_t)rb * C::BM * C::OB;
#pragma unroll 1
    for (int idx = tid; idx < ntok * CH; idx += C::NTH) {
      const int tt = idx / CH, c = idx - tt * CH;
      const uint4 v = *reinterpret_cast<const uint4*>(stg + (size_t)tt * C::SROW * C::OB + c * 16);
      *reinterpret_cast<uint4*>(yb + ((size_t)(tlo + tt) * P.ldy) * C::OB + c * 16) = v;
    }
  }
  if (stamp && tid == 0) {
    unsigned long long* sp = P.stamps + 4 * (size_t)blockIdx.x;
    sp[0] = t_entry; sp[1] = t_first; sp[2] = t_loop; sp[3] = gtime();
  }
}

__global__ void null_kernel(int* e) { if (threadIdx.x == 1023) e[3] = 0; }

}  // namespace vd

typedef CUresult (*vd_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static vd_encode_fn vd_get_encode() {
  static vd_encode_fn fn = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) cudaGetLastError();
    else fn = reinterpret_cast<vd_encode_fn>(p);
  }
  return fn;
}

static void vd_map(CUtensorMap* tm, const void* ptr, int64_t rows, int64_t cols, int box_rows) {
  vd_encode_fn enc = vd_get_encode();
  TORCH_CHECK(enc != nullptr, "vd: cuTensorMapEncodeTiled unavailable");
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)(cols * 2)};
  cuuint32_t box[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1u, 1u};
  const CUresult r = enc(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "vd: tensor map encode failed ", (int)r);
}

template <int NWG, int BN, int NST, int CL, int F32, int RBD>
static int64_t vd_launch_t(const torch::Tensor& x, const torch::Tensor& wpk, const torch::Tensor& wraw, const torch::Tensor& y,
                           const torch::Tensor& err, const torch::Tensor& stamps, int64_t nrb, int64_t w_rb, int64_t w_s,
                           int64_t flags, int64_t pdl, int64_t pf) {
  using C = vd::Cfg<NWG, BN, NST, CL, F32, RBD>;
  auto kern = vd::vd_kernel<NWG, BN, NST, CL, F32, RBD>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    attr = true;
  }
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "vd: x bf16 [T, K]");
  const int64_t T = x.size(0), K = x.size(1);
  TORCH_CHECK(T >= 1 && T <= BN && K % 64 == 0 && x.data_ptr() != nullptr && ((uintptr_t)x.data_ptr() % 16) == 0, "vd: x shape");
  TORCH_CHECK(y.is_cuda() && y.dim() == 2 && y.stride(1) == 1 && y.size(0) >= T && y.size(1) >= nrb * C::BM, "vd: y");
  TORCH_CHECK(y.scalar_type() == (F32 ? at::kFloat : at::kBFloat16), "vd: y dtype");
  TORCH_CHECK((y.stride(0) * C::OB) % 16 == 0 && ((uintptr_t)y.data_ptr() % 16) == 0, "vd: y alignment");
  const bool wmap = (flags & vd::F_WMAP) != 0;
  CUtensorMap tmx, tmw;
  memset(&tmw, 0, sizeof(tmw));
  vd_map(&tmx, x.data_ptr(), T, K, BN);
  if (wmap) {
    TORCH_CHECK(wraw.dim() == 2 && wraw.size(1) == K && wraw.size(0) >= nrb * C::BM && wraw.is_contiguous(), "vd: wraw");
    vd_map(&tmw, wraw.data_ptr(), wraw.size(0), K, C::BM);
  } else {
    TORCH_CHECK(wpk.is_cuda() && ((uintptr_t)wpk.data_ptr() % 16) == 0, "vd: wpk");
  }
  vd::Params P;
  P.w = reinterpret_cast<const uint8_t*>(wpk.data_ptr());
  P.y = y.data_ptr();
  P.err = err.data_ptr<int>();
  P.stamps = reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>());
  P.w_rb = w_rb; P.w_s = w_s;
  P.nks = (int)(K / 64); P.T = (int)T; P.ldy = (int)y.stride(0); P.flags = (int)flags; P.pf = (int)(pf & 255); P.rot = (int)(pf >> 8);
  if (flags & vd::F_STAMP) TORCH_CHECK(stamps.numel() >= 4 * nrb * CL, "vd: stamps");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(nrb * CL), 1, 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute at_[2];
  int na = 0;
  if (CL > 1) {
    at_[na].id = cudaLaunchAttributeClusterDimension;
    at_[na].val.clusterDim.x = CL; at_[na].val.clusterDim.y = 1; at_[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = na ? at_ : nullptr;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tmx, tmw, P));
  return 0;
}

template <int NWG, int BN, int NST, int CL, int F32, int RBD>
static std::vector<int64_t> vd_info_t() {
  using C = vd::Cfg<NWG, BN, NST, CL, F32, RBD>;
  auto kern = vd::vd_kernel<NWG, BN, NST, CL, F32, RBD>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  int ncl = -1;
  {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(132 * CL, 1, 1);
    cfg.blockDim = dim3(C::THREADS, 1, 1);
    cfg.dynamicSmemBytes = C::SMEM;
    cudaLaunchAttribute a;
    a.id = cudaLaunchAttributeClusterDimension;
    a.val.clusterDim.x = CL; a.val.clusterDim.y = 1; a.val.clusterDim.z = 1;
    cfg.attrs = &a; cfg.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&ncl, kern, &cfg) != cudaSuccess) { cudaGetLastError(); ncl = -1; }
  }
  return {(int64_t)C::SMEM, (int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)ncl, (int64_t)C::BM, (int64_t)BN,
          (int64_t)NST, (int64_t)CL, (int64_t)F32};
}

int64_t vd_run(int64_t kid, torch::Tensor x, torch::Tensor wpk, torch::Tensor wraw, torch::Tensor y, torch::Tensor err,
               torch::Tensor stamps, int64_t nrb, int64_t w_rb, int64_t w_s, int64_t flags, int64_t pdl, int64_t pf) {
  if (kid < 0) {
    vd::null_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream().stream()>>>(err.data_ptr<int>());
    return 0;
  }
  switch (kid) {
@RUNCASES@
  }
  TORCH_CHECK(false, "vd: unknown kid ", kid);
  return -2;
}

std::vector<int64_t> vd_info(int64_t kid) {
  switch (kid) {
@INFOCASES@
  }
  TORCH_CHECK(false, "vd: unknown kid ", kid);
  return {};
}
"""


def vd_wg_specs(ns):
    """wgmma.mma_async m64nNk16 SS (f32 += bf16 x bf16), A and B K-major (no transpose), scale-d = 1."""
    out = []
    for n in ns:
        na = n // 2
        regs = ",".join("%%%d" % i for i in range(na))
        outs = ", ".join('"+f"(d[%d])' % i for i in range(na))
        out.append(
            "template <> __device__ __forceinline__ void wgmma_ss<%d>(float* d, uint64_t da, uint64_t db) {\n"
            '  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 "\n'
            '               "{%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n"\n'
            '               : %s\n               : "l"(da), "l"(db));\n}'
            % (n, n, regs, na, na + 1, outs))
    return "\n".join(out)


def vd_cuda_src(insts=VD_INSTS):
    run = "\n".join(f"    case {i}: return vd_launch_t<{a}, {b}, {c}, {d}, {e}, {f}>(x, wpk, wraw, y, err, stamps, nrb, w_rb, w_s, flags, pdl, pf);"
                    for i, (a, b, c, d, e, f) in enumerate(insts))
    info = "\n".join(f"    case {i}: return vd_info_t<{a}, {b}, {c}, {d}, {e}, {f}>();" for i, (a, b, c, d, e, f) in enumerate(insts))
    ns = sorted({ins[1] for ins in insts})
    return VD_CUDA_SRC.replace("@RUNCASES@", run).replace("@INFOCASES@", info).replace("@WGSPECS@", vd_wg_specs(ns))


def vd_build(insts=VD_INSTS, name="vdlab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda = vd_cuda_src(insts)
    tag = hashlib.sha1((VD_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=VD_CPP_SRC, cuda_sources=cuda, functions=["vd_run", "vd_info"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
                      verbose=verbose)
    print(f"[vd] built {len(insts)} kernels in {time.time() - t0:.1f}s", flush=True)
    return mod


# ---------------------------------------------------------------- packing (lossless: the same bf16 bits, re-ordered)
def vd_pack(w, bm, order="front"):
    """w [N, K] bf16 -> pre-swizzled SW128 tiles of [bm rows x 64 k]; returns (flat uint8 tensor, w_rb, w_s) in bytes.
    Tile (rb, s) holds W[rb bm + r][64 s + 8 c + e] at byte r * 128 + ((c ^ (r & 7)) * 16) + 2 e (TMA SWIZZLE_128B image)."""
    n, k = w.shape
    assert n % bm == 0 and k % 64 == 0
    nrb, nks = n // bm, k // 64
    t = w.contiguous().view(nrb, bm, nks, 8, 8).permute(0, 2, 1, 3, 4)          # [rb][s][r][c][e]
    r = torch.arange(bm, device=w.device)
    c = torch.arange(8, device=w.device)
    src_c = (c[None, :] ^ (r[:, None] & 7))                                      # [r][c'] -> source chunk
    idx = src_c.view(1, 1, bm, 8, 1).expand(nrb, nks, bm, 8, 8)
    t = torch.gather(t, 3, idx)                                                  # [rb][s][r][c'][e]
    tile = bm * 128
    if order == "front":
        t = t.permute(1, 0, 2, 3, 4).contiguous()                                # [s][rb]
        return t.view(torch.uint8).view(-1), tile, nrb * tile
    t = t.contiguous()                                                           # [rb][s]
    return t.view(torch.uint8).view(-1), nks * tile, tile


def vd_unpack_check(w, bm):
    """CPU-side self-check of vd_pack (inverse permutation reproduces W bit-exactly)."""
    flat, w_rb, w_s = vd_pack(w, bm, "front")
    n, k = w.shape
    nrb, nks = n // bm, k // 64
    t = flat.view(torch.bfloat16).view(nks, nrb, bm, 8, 8)
    r = torch.arange(bm, device=w.device)
    c = torch.arange(8, device=w.device)
    inv = (c[None, :] ^ (r[:, None] & 7))
    t = torch.gather(t, 3, inv.view(1, 1, bm, 8, 1).expand(nks, nrb, bm, 8, 8))  # back to [s][rb][r][c][e]
    back = t.permute(1, 2, 0, 3, 4).reshape(n, k)
    return bool(torch.equal(back.view(torch.int16), w.view(torch.int16)))


# ---------------------------------------------------------------- lab
PROJ = {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "cat": (4608, 8192)}
# per (proj, T): candidate (kid, label) configs; the first one is the primary (gets the diagnostic arms)
# R-vd-2 configs (kept for --set=2); R-vd-3 (default): m = 128 only, K-rotation arms (pf arg = pf | rot << 8), efw default
CONFIGS2 = {
    ("unpooler", 128): [(5, "bm128_cl2_ded"), (6, "bm128_cl2_ali"), (0, "bm64_cl1")],
    ("pooler", 128): [(1, "bm64_cl2_ded"), (2, "bm64_cl2_ali")],
    ("qkv", 128): [(3, "bm64_cl3_ded"), (4, "bm64_cl3_ali")],
    ("o", 128): [(0, "bm64_cl1"), (5, "bm128_cl2_ded"), (6, "bm128_cl2_ali")],
    ("cat", 128): [(7, "bm128_cl3_ded"), (8, "bm128_cl3_ali")],
    ("unpooler", 256): [(9, "bm128_cl2")],
    ("pooler", 256): [(11, "bm128_cl3_g96")],
    ("o", 256): [(9, "bm128_cl2")],
    ("cat", 256): [(10, "bm128_cl3")],
}
ARMS_PRIMARY2 = (("", 0, 0), ("_w0", 256, 0), ("_pf2", 0, 2), ("_pfall", 0, 99), ("_wmap", 64, 0), ("_efw", 8, 0),
                 ("_wonly", 2 | 4, 0), ("_nomma", 2, 0))
ARMS_OTHER2 = (("", 0, 0), ("_w0", 256, 0))
CONFIGS = {
    ("unpooler", 128): [(5, "bm128_cl2_ded"), (0, "bm64_cl1")],
    ("pooler", 128): [(1, "bm64_cl2_ded")],
    ("qkv", 128): [(3, "bm64_cl3_ded")],
    ("o", 128): [(0, "bm64_cl1"), (5, "bm128_cl2_ded")],
    ("cat", 128): [(7, "bm128_cl3_ded"), (8, "bm128_cl3_ali")],
}
# (tag, flags, pf | rot << 8): the primary config gets all of them, the others the first three
ARMS_PRIMARY = (("_efw", 8, 0), ("_efw_rot1", 8, 1 << 8), ("_efw_rot5", 8, 5 << 8), ("_efw_pf2_rot1", 8, 2 | 1 << 8),
                ("_efw_w0", 8 | 256, 0), ("_efw_wonly", 8 | 6, 0), ("_efw_rot1_nomma", 8 | 2, 1 << 8))
ARMS_OTHER = (("_efw", 8, 0), ("_efw_rot1", 8, 1 << 8), ("_efw_rot5", 8, 5 << 8))
# --set=4: dynamic split inside the cluster (flag 512) on the CL > 1 configs
CONFIGS4 = {
    ("pooler", 128): [(1, "bm64_cl2_ded")],
    ("qkv", 128): [(3, "bm64_cl3_ded")],
    ("unpooler", 128): [(5, "bm128_cl2_ded")],
    ("cat", 128): [(7, "bm128_cl3_ded")],
}
ARMS_PRIMARY4 = (("_efw", 8, 0), ("_efw_dyn", 8 | 512, 0), ("_efw_dyn_rot1", 8 | 512, 1 << 8), ("_efw_dyn_pf2", 8 | 512, 2))


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


def time_many(graphs, flush, sink, reps):
    """judge-style: read-sweep L2 flush before every replay, rotating order each rep, medians (us) per graph."""
    n = len(graphs)
    ts = [[] for _ in range(n)]
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(reps + 2):
        order = [(i + r) % n for i in range(n)]
        if r % 2:
            order = order[::-1]
        for i in order:
            torch.sum(flush, dim=0, out=sink)
            e0.record()
            graphs[i].replay()
            e1.record()
            torch.cuda.synchronize()
            if r >= 2:
                ts[i].append(e0.elapsed_time(e1) * 1e3)
    return [statistics.median(t) for t in ts]


def stamp_summary(mod, kid, x, flat, w, y, err, stamps, nrb, w_rb, w_s, flush, sink, flags=0, pf=0):
    """one untimed cold-L2 call with per-CTA globaltimer stamps: (entry, first stage landed, main loop done, end) in us"""
    cl = VD_INSTS[kid][3]
    stamps.zero_()
    torch.sum(flush, dim=0, out=sink)
    mod.vd_run(kid, x, flat, w, y, err, stamps, nrb, w_rb, w_s, flags | 32, 0, pf)
    torch.cuda.synchronize()
    sp = stamps[:4 * nrb * cl].view(-1, 4).double()
    t0 = float(sp[:, 0].min())
    q = lambda v: (round(float(v.min()), 2), round(float(v.median()), 2), round(float(v.max()), 2))  # noqa: E731
    return {nm: q((sp[:, i] - t0) / 1e3) for i, nm in enumerate(("entry", "first", "loop", "end"))}


def main():
    ts_ = [int(v) for v in arg("ts", "128,256").split(",")]
    projs = arg("projs", "unpooler,pooler,qkv,o,cat").split(",")
    reps = int(arg("reps", "21"))
    rounds = int(arg("rounds", "8"))
    out_path = Path(arg("out", str(ROOT / "box_out" / "vd3" / "lab.json")))
    global CONFIGS, ARMS_PRIMARY, ARMS_OTHER
    if arg("set", "3") == "2":
        CONFIGS, ARMS_PRIMARY, ARMS_OTHER = CONFIGS2, ARMS_PRIMARY2, ARMS_OTHER2
    elif arg("set", "3") == "4":
        CONFIGS, ARMS_PRIMARY, ARMS_OTHER = CONFIGS4, ARMS_PRIMARY4, ARMS_PRIMARY4
    mod = vd_build(VD_INSTS, verbose=bool(int(arg("verbose", "0"))))
    infos = {i: list(mod.vd_info(i)) for i in range(len(VD_INSTS))}
    for i, v in infos.items():
        print(f"[vd] inst {i} {VD_INSTS[i]}: smem {v[0]} regs {v[1]} local {v[2]} B max-clusters {v[3]}", flush=True)
    if int(arg("build-only", "0")):
        return
    dev = torch.device("cuda")
    torch.manual_seed(1234)
    flush = torch.empty((64 << 20,), dtype=torch.float32, device=dev).uniform_().view(4096, -1)   # 256 MB read sweep
    sink = torch.empty((flush.shape[1],), dtype=torch.float32, device=dev)
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    stamps = torch.zeros((4 * 512,), dtype=torch.int64, device=dev)
    results = {"insts": {str(i): infos[i] for i in infos}, "rows": []}
    gnull = graph_of(lambda: mod.vd_run(-1, err, err, err, err, err, err, 1, 0, 0, 0, 0, 0))
    for proj in projs:
        n, k = PROJ[proj]
        w = (torch.randn((n, k), device=dev) * 0.02).to(torch.bfloat16)
        f32 = proj == "cat"
        packs = {}
        for t_ in ts_:
            for kid, _ in CONFIGS.get((proj, t_), []):
                bm = 64 * VD_INSTS[kid][0]
                if bm not in packs:
                    packs[bm] = vd_pack(w, bm, "front")
                    if not vd_unpack_check(w, bm):
                        print(f"[vd] {proj}: PACK SELF-CHECK FAILED bm {bm}", flush=True)
        for m in ts_:
            cfgs = CONFIGS.get((proj, m))
            if not cfgs:
                continue
            x = torch.randn((m, k), device=dev).to(torch.bfloat16)
            y_ref64 = x.double() @ w.double().t()
            odt = torch.float32 if f32 else torch.bfloat16
            if f32:   # the layer's cat call (pooler + router hi / lo rows, FP32 out)
                y_cub = torch.mm(x, w.t(), out_dtype=torch.float32)
                arms = {"cublas": (lambda x=x: torch.mm(x, w.t(), out_dtype=torch.float32))}
            else:
                y_cub = torch.empty((m, n), dtype=odt, device=dev)
                arms = {"cublas": (lambda x=x, y=y_cub: torch.mm(x, w.t(), out=y))}
            ys, meta = {}, {}
            for ci, (kid, label) in enumerate(cfgs):
                bm = 64 * VD_INSTS[kid][0]
                nrb = n // bm
                flat, w_rb, w_s = packs[bm]
                for tag, fl, pf in (ARMS_PRIMARY if ci == 0 else ARMS_OTHER):
                    y = torch.zeros((m, n), dtype=odt, device=dev)
                    name = f"vd{kid}_{label}{tag}"

                    def run(kid=kid, x=x, flat=flat, y=y, nrb=nrb, w_rb=w_rb, w_s=w_s, fl=fl, pf=pf):
                        mod.vd_run(kid, x, flat, w, y, err, stamps, nrb, w_rb, w_s, fl, 0, pf)
                    arms[name] = run
                    ys[name] = (y, fl)
                    meta[name] = (kid, nrb, fl, pf)
            for name, fn in arms.items():
                fn()
            torch.cuda.synchronize()
            e = int(err[0].item())
            scale = float(y_ref64.abs().max())
            rel_cub = float((y_cub.double() - y_ref64).abs().max()) / scale
            chk = {name: float((y.double() - y_ref64).abs().max()) / scale for name, (y, fl) in ys.items() if not (fl & 6)}
            first = next(iter(ys))
            bit = {name: bool(torch.equal(ys[name][0], ys[first][0])) for name in ys if not (ys[name][1] & 6)}
            print(f"[vd] {proj} m{m}: err {e}; rel vs fp64: cublas {rel_cub:.2e}; "
                  + " ".join(f"{k2}={v:.2e}" for k2, v in chk.items()), flush=True)
            print(f"[vd] {proj} m{m}: bitwise equal to {first}: " + " ".join(f"{k2}={v}" for k2, v in bit.items()), flush=True)
            if e:
                print(f"[vd] {proj} m{m}: bounded wait timed out (err {e}) -> no timing for this shape", flush=True)
                results["rows"].append({"proj": proj, "m": m, "err": e, "rel": chk})
                err.zero_()
                continue
            for kid, label in cfgs:   # partial-T correctness (zero-filled x rows, masked stores)
                bm = 64 * VD_INSTS[kid][0]
                flat, w_rb, w_s = packs[bm]
                for tp in ((100, 33) if m == 128 else (200, 129)):
                    xp = x[:tp].contiguous()
                    yp = torch.full((tp, n), 7.0, dtype=odt, device=dev)
                    mod.vd_run(kid, xp, flat, w, yp, err, stamps, n // bm, w_rb, w_s, 0, 0, 0)
                    torch.cuda.synchronize()
                    rp = float((yp.double() - y_ref64[:tp]).abs().max()) / scale
                    print(f"[vd] {proj} m{m} vd{kid}: partial T={tp}: rel {rp:.2e} err {int(err[0].item())}", flush=True)
            names = list(arms.keys())
            graphs = [graph_of(arms[nm]) for nm in names] + [gnull]
            meds = {nm: [] for nm in names + ["null"]}
            for rd in range(rounds):
                tt = time_many(graphs, flush, sink, reps)
                for nm, v in zip(names + ["null"], tt):
                    meds[nm].append(v)
            fl0 = statistics.median(meds["null"])
            row = {"proj": proj, "m": m, "err": e, "rel_cublas": rel_cub, "rel": chk, "floor_us": fl0, "arms": {}, "stamps_us": {}}
            cub = statistics.median(meds["cublas"])
            mb = n * k * 2 / 1e6
            for nm in names:
                med = statistics.median(meds[nm])
                row["arms"][nm] = {"raw_us": med, "net_us": med - fl0, "rounds": meds[nm],
                                   "x_vs_cublas_net": (cub - fl0) / (med - fl0), "tbs": mb / (med - fl0)}
            for nm, (kid, nrb, fl, pf) in meta.items():   # stamps for every real arm
                if fl & 6:
                    continue
                bm = 64 * VD_INSTS[kid][0]
                flat, w_rb, w_s = packs[bm]
                row["stamps_us"][nm] = stamp_summary(mod, kid, x, flat, w, ys[nm][0], err, stamps, nrb, w_rb, w_s, flush, sink,
                                                     fl & ~32, pf)
            for name, fn in arms.items():   # repeat check after timing
                fn()
            torch.cuda.synchronize()
            row["err_after"] = int(err[0].item())
            row["rel_after"] = {name: float((y.double() - y_ref64).abs().max()) / scale for name, (y, fl) in ys.items()
                                if not (fl & 6)}
            results["rows"].append(row)
            print(f"[vd] {proj} m{m}: floor {fl0:.2f} us; net us (x vs cuBLAS, TB/s bf16-bytes) | stamps first / loop / end (min-med-max):",
                  flush=True)
            for nm in names:
                a = row["arms"][nm]
                s = row["stamps_us"].get(nm)
                st = "" if s is None else (f" | first {s['first'][1]:.2f}  loop {s['loop'][0]:.2f}-{s['loop'][1]:.2f}-{s['loop'][2]:.2f}"
                                           f"  end {s['end'][0]:.2f}-{s['end'][1]:.2f}-{s['end'][2]:.2f}")
                print(f"    {nm:30s} {a['net_us']:7.2f}  x{a['x_vs_cublas_net']:.3f}  {a['tbs']:.2f} TB/s{st}", flush=True)
            print(f"[vd] {proj} m{m}: err after {row['err_after']}", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=1))
    print("[vd] done", flush=True)


if __name__ == "__main__":
    main()

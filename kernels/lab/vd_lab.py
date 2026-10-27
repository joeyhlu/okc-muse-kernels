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

# (NWG, BN, NST, CL, F32): BM = 64 NWG weight rows x BN tokens per CTA, NST ring stages, CL-CTA cluster split-K
VD_INSTS = [
    (1, 128, 9, 1, 0),   # 0  BM  64 full K           unpooler / o: 128 CTAs (x 2 x W)
    (1, 128, 9, 2, 0),   # 1  BM  64 CL 2             pooler 128, qkv 72
    (1, 128, 9, 3, 0),   # 2  BM  64 CL 3             qkv 108
    (2, 128, 7, 1, 0),   # 3  BM 128 full K           o 64 (diag)
    (2, 128, 7, 2, 0),   # 4  BM 128 CL 2             unpooler / o 128 (x 1 x W), pooler 64 (diag)
    (2, 128, 7, 3, 1),   # 5  BM 128 CL 3, FP32 out   cat 108
    (1, 128, 9, 2, 1),   # 6  BM  64 CL 2, FP32 out   pooler-part of cat (diag)
    (2, 256, 4, 2, 0),   # 7  T=256 BM 128 CL 2       unpooler / o 128
    (2, 256, 4, 3, 1),   # 8  T=256 BM 128 CL 3 FP32  cat 108
    (2, 256, 4, 3, 0),   # 9  T=256 BM 128 CL 3       pooler 96, qkv 54 (diag)
]

VD_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t vd_run(int64_t kid, torch::Tensor x, torch::Tensor wpk, torch::Tensor wraw, torch::Tensor y, torch::Tensor err,
               torch::Tensor stamps, int64_t nrb, int64_t w_rb, int64_t w_s, int64_t flags, int64_t pdl);
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
enum : int { F_EXIT = 1, F_NOMMA = 2, F_NOX = 4, F_EFW = 8, F_ELX = 16, F_STAMP = 32, F_WMAP = 64, F_TRIG = 128 };

struct Params {
  const uint8_t* w;          // packed tiles (bulk mode): tile (rb, s) at w + rb * w_rb + s * w_s (bytes)
  void* y;                   // [T, ldy] bf16 or f32
  int* err;
  unsigned long long* stamps;   // [grid][4] (flag 32)
  long long w_rb, w_s;
  int nks, T, ldy, flags;
};

template <int NWG, int BN, int NST, int CL, int F32>
struct Cfg {
  static constexpr int BM = 64 * NWG;
  static constexpr int WB = BM * 128;              // weight tile bytes (BM rows x 64 bf16)
  static constexpr int XB = BN * 128;              // x tile bytes (BN tokens x 64 bf16)
  static constexpr int SB = WB + XB;               // stage bytes (multiple of 1024)
  static constexpr int NTH = NWG * 128;            // consumer threads
  static constexpr int THREADS = NTH + 32;         // + one producer warp
  static constexpr int NACC = BN / 2;              // fp32 accumulators per consumer thread
  static constexpr int NB = BN / 8;                // 8-token blocks
  static constexpr int BAR_OFF = NST * SB;
  static constexpr int SMEM = 1024 + BAR_OFF + 16 * NST + 64;
  static constexpr int OB = F32 ? 4 : 2;           // output element bytes
  static constexpr int SROW = F32 ? (BM + 4) : (BM + 8);   // staging row pitch (elements), 16-B multiple
  static constexpr int F_BYTES = CL > 1 ? NB * NTH * 16 : 0;
  static constexpr int JMAX = (NB + CL - 1) / CL;
  static constexpr int S_BYTES = JMAX * 8 * SROW * OB;
  static_assert(SB % 1024 == 0, "stage alignment");
  static_assert(F_BYTES + S_BYTES <= NST * SB, "epilogue buffers alias the ring");
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
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ float4 ld_dsmem_f4(uint32_t local_addr, uint32_t rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(local_addr), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(ra) : "memory");
  return v;
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

template <int NWG, int BN, int NST, int CL, int F32>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
vd_kernel(const __grid_constant__ CUtensorMap tmx, const __grid_constant__ CUtensorMap tmw, const Params P) {
  using C = Cfg<NWG, BN, NST, CL, F32>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t s_raw = sa(smem_raw);
  const uint32_t sbase = (s_raw + 1023u) & ~1023u;
  uint8_t* gbase = smem_raw + (sbase - s_raw);
  const uint32_t full0 = sbase + C::BAR_OFF, empty0 = full0 + 8 * NST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int flags = P.flags;
  if (flags & F_EXIT) return;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int s0 = ((int)crank * P.nks) / CL, s1 = (((int)crank + 1) * P.nks) / CL;
  const int nk = s1 - s0;
  const bool stamp = (flags & F_STAMP) != 0;
  unsigned long long t_entry = 0;
  if (stamp && tid == 0) t_entry = gtime();
  bool dead = false;

  if (tid == 0) {
#pragma unroll 1
    for (int i = 0; i < NST; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (warp == NWG * 4) {
    // ------------------------------------------------------------------ producer (one lane)
    if (lane == 0) {
      const bool nox = (flags & F_NOX) != 0, wmap = (flags & F_WMAP) != 0;
      const bool efw = (flags & F_EFW) != 0, elx = (flags & F_ELX) != 0;
      const uint64_t pw = efw ? pol_first() : 0ull, px = elx ? pol_last() : 0ull;
      const uint32_t txb = (uint32_t)C::WB + (nox ? 0u : (uint32_t)C::XB);
      const uint8_t* wbase = P.w + (long long)rb * P.w_rb;
      asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmx)) : "memory");
      if (wmap) asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmw)) : "memory");
      const int npro = nk < NST ? nk : NST;
#pragma unroll 1
      for (int i = 0; i < npro; ++i) {   // weights do not depend on the previous kernel: issue before the PDL wait
        const uint32_t dst = sbase + i * C::SB, bar = full0 + 8 * i;
        mbar_expect_tx(bar, txb);
        if (wmap) { if (efw) tma2d_h(dst, &tmw, (s0 + i) * 64, rb * C::BM, bar, pw); else tma2d(dst, &tmw, (s0 + i) * 64, rb * C::BM, bar); }
        else { const void* src = wbase + (long long)(s0 + i) * P.w_s;
               if (efw) bulk_g2s_h(dst, src, C::WB, bar, pw); else bulk_g2s(dst, src, C::WB, bar); }
      }
      pdl_wait();
      if (!nox) {
#pragma unroll 1
        for (int i = 0; i < npro; ++i) {
          const uint32_t dst = sbase + i * C::SB + C::WB, bar = full0 + 8 * i;
          if (elx) tma2d_h(dst, &tmx, (s0 + i) * 64, 0, bar, px); else tma2d(dst, &tmx, (s0 + i) * 64, 0, bar);
        }
      }
#pragma unroll 1
      for (int i = NST; i < nk; ++i) {
        const int slot = i % NST;
        const uint32_t dst = sbase + slot * C::SB, bar = full0 + 8 * slot;
        mbar_wait(empty0 + 8 * slot, (uint32_t)(((i / NST) - 1) & 1), P.err, dead);
        mbar_expect_tx(bar, txb);
        if (wmap) { if (efw) tma2d_h(dst, &tmw, (s0 + i) * 64, rb * C::BM, bar, pw); else tma2d(dst, &tmw, (s0 + i) * 64, rb * C::BM, bar); }
        else { const void* src = wbase + (long long)(s0 + i) * P.w_s;
               if (efw) bulk_g2s_h(dst, src, C::WB, bar, pw); else bulk_g2s(dst, src, C::WB, bar); }
        if (!nox) { if (elx) tma2d_h(dst + C::WB, &tmx, (s0 + i) * 64, 0, bar, px); else tma2d(dst + C::WB, &tmx, (s0 + i) * 64, 0, bar); }
      }
    }
    __syncwarp();
    if constexpr (CL > 1) { cluster_sync(); cluster_sync(); }
    return;
  }

  // -------------------------------------------------------------------- consumers
  const int wg = warp >> 2, t = tid & 127;
  float acc[C::NACC];
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) acc[q] = 0.f;
  const bool nomma = (flags & F_NOMMA) != 0;
  unsigned long long t_first = 0;
#pragma unroll 1
  for (int i = 0; i < nk; ++i) {
    const int slot = i % NST;
    mbar_wait(full0 + 8 * slot, (uint32_t)((i / NST) & 1), P.err, dead);
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
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    }
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NST));
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
  int jlo = 0, jhi = C::NB;
  uint8_t* stg = gbase + C::F_BYTES;
  if constexpr (CL > 1) {
    float4* F = reinterpret_cast<float4*>(gbase);
#pragma unroll
    for (int j = 0; j < C::NB; ++j) F[j * C::NTH + tid] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    cluster_sync();
    jlo = ((int)crank * C::NB) / CL;
    jhi = (((int)crank + 1) * C::NB) / CL;
    const uint32_t fl = sa(F);
#pragma unroll
    for (int j = 0; j < C::NB; ++j) {
      if (j < jlo || j >= jhi) continue;
      float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
      for (int r = 0; r < CL; ++r) {
        if (r == (int)crank) continue;
        const float4 p = ld_dsmem_f4(fl + (uint32_t)((j * C::NTH + tid) * 16), (uint32_t)r);
        v0 += p.x; v1 += p.y; v2 += p.z; v3 += p.w;
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
  if constexpr (CL > 1) cluster_sync();   // nobody leaves while a peer may still read its fragments
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

template <int NWG, int BN, int NST, int CL, int F32>
static int64_t vd_launch_t(const torch::Tensor& x, const torch::Tensor& wpk, const torch::Tensor& wraw, const torch::Tensor& y,
                           const torch::Tensor& err, const torch::Tensor& stamps, int64_t nrb, int64_t w_rb, int64_t w_s,
                           int64_t flags, int64_t pdl) {
  using C = vd::Cfg<NWG, BN, NST, CL, F32>;
  auto kern = vd::vd_kernel<NWG, BN, NST, CL, F32>;
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
  P.nks = (int)(K / 64); P.T = (int)T; P.ldy = (int)y.stride(0); P.flags = (int)flags;
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

template <int NWG, int BN, int NST, int CL, int F32>
static std::vector<int64_t> vd_info_t() {
  using C = vd::Cfg<NWG, BN, NST, CL, F32>;
  auto kern = vd::vd_kernel<NWG, BN, NST, CL, F32>;
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
               torch::Tensor stamps, int64_t nrb, int64_t w_rb, int64_t w_s, int64_t flags, int64_t pdl) {
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
    run = "\n".join(f"    case {i}: return vd_launch_t<{a}, {b}, {c}, {d}, {e}>(x, wpk, wraw, y, err, stamps, nrb, w_rb, w_s, flags, pdl);"
                    for i, (a, b, c, d, e) in enumerate(insts))
    info = "\n".join(f"    case {i}: return vd_info_t<{a}, {b}, {c}, {d}, {e}>();" for i, (a, b, c, d, e) in enumerate(insts))
    ns = sorted({b for _, b, _, _, _ in insts})
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
CONFIGS = {
    ("unpooler", 128): [(4, "bm128_cl2"), (0, "bm64_cl1")],
    ("pooler", 128): [(1, "bm64_cl2"), (4, "bm128_cl2_g64")],
    ("qkv", 128): [(2, "bm64_cl3"), (1, "bm64_cl2_g72")],
    ("o", 128): [(4, "bm128_cl2"), (0, "bm64_cl1"), (3, "bm128_cl1_g64")],
    ("cat", 128): [(5, "bm128_cl3"), ],
    ("unpooler", 256): [(7, "bm128_cl2")],
    ("pooler", 256): [(9, "bm128_cl3_g96")],
    ("qkv", 256): [(9, "bm128_cl3_g54")],
    ("o", 256): [(7, "bm128_cl2")],
    ("cat", 256): [(8, "bm128_cl3")],
}
DIAG = ((2 | 4, "_wonly"), (2, "_nomma"), (64, "_wmap"), (8, "_efw"), (16, "_elx"), (8 | 16, "_efw_elx"))


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


def main():
    ts_ = [int(v) for v in arg("ts", "128,256").split(",")]
    projs = arg("projs", "unpooler,pooler,qkv,o,cat").split(",")
    reps = int(arg("reps", "21"))
    rounds = int(arg("rounds", "8"))
    out_path = Path(arg("out", str(ROOT / "box_out" / "vd1" / "lab.json")))
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
    gnull = graph_of(lambda: mod.vd_run(-1, err, err, err, err, err, err, 1, 0, 0, 0, 0))
    for proj in projs:
        n, k = PROJ[proj]
        w = (torch.randn((n, k), device=dev) * 0.02).to(torch.bfloat16)
        f32 = proj == "cat"
        packs = {}
        for t_ in ts_:
            for kid, _ in CONFIGS.get((proj, t_), []):
                bm = 64 * VD_INSTS[kid][0]
                for order in ("front", "slab"):
                    if (bm, order) not in packs:
                        packs[(bm, order)] = vd_pack(w, bm, order)
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
            ys = {}
            for ci, (kid, label) in enumerate(cfgs):
                nwg, bn, nst, cl, _ = VD_INSTS[kid]
                bm = 64 * nwg
                nrb = n // bm
                variants = [(0, "", "front")] + ([(0, "_slab", "slab")] + [(fl, tg, "front") for fl, tg in DIAG] if ci == 0 else [])
                for fl, tag, order in variants:
                    flat, w_rb, w_s = packs[(bm, order)]
                    y = torch.zeros((m, n), dtype=odt, device=dev)
                    name = f"vd{kid}_{label}{tag}"

                    def run(kid=kid, x=x, flat=flat, y=y, nrb=nrb, w_rb=w_rb, w_s=w_s, fl=fl):
                        mod.vd_run(kid, x, flat, w, y, err, stamps, nrb, w_rb, w_s, fl, 0)
                    arms[name] = run
                    ys[name] = (y, fl)
            # correctness before timing (diagnostic arms 2 / 4 are not checked)
            for name, fn in arms.items():
                fn()
            torch.cuda.synchronize()
            e = int(err[0].item())
            scale = float(y_ref64.abs().max())
            rel_cub = float((y_cub.double() - y_ref64).abs().max()) / scale
            chk = {}
            for name, (y, fl) in ys.items():
                if not (fl & (2 | 4)):
                    chk[name] = float((y.double() - y_ref64).abs().max()) / scale
            bit = {name: bool(torch.equal(ys[name][0], ys[next(iter(ys))][0])) for name in ys if not (ys[name][1] & 6)}
            print(f"[vd] {proj} m{m}: err {e}; rel vs fp64: cublas {rel_cub:.2e}; "
                  + " ".join(f"{k2}={v:.2e}" for k2, v in chk.items()), flush=True)
            print(f"[vd] {proj} m{m}: bitwise equal to the first vd arm: " + " ".join(f"{k2}={v}" for k2, v in bit.items()), flush=True)
            if e:
                print(f"[vd] {proj} m{m}: bounded wait timed out (err {e}) -> no timing for this shape", flush=True)
                results["rows"].append({"proj": proj, "m": m, "err": e, "rel": chk})
                err.zero_()
                continue
            # partial-T correctness (zero-filled x rows, masked stores) on the primary config
            kid0 = cfgs[0][0]
            bm0 = 64 * VD_INSTS[kid0][0]
            for tp in ((100, 33) if m == 128 else (200, 129)):
                xp = x[:tp].contiguous()
                yp = torch.full((tp, n), 7.0, dtype=odt, device=dev)
                flat, w_rb, w_s = packs[(bm0, "front")]
                mod.vd_run(kid0, xp, flat, w, yp, err, stamps, n // bm0, w_rb, w_s, 0, 0)
                torch.cuda.synchronize()
                rp = float((yp.double() - y_ref64[:tp]).abs().max()) / scale
                print(f"[vd] {proj} m{m}: partial T={tp}: rel {rp:.2e} err {int(err[0].item())}", flush=True)
            names = list(arms.keys())
            graphs = [graph_of(arms[nm]) for nm in names] + [gnull]
            meds = {nm: [] for nm in names + ["null"]}
            for rd in range(rounds):
                tt = time_many(graphs, flush, sink, reps)
                for nm, v in zip(names + ["null"], tt):
                    meds[nm].append(v)
            fl0 = statistics.median(meds["null"])
            row = {"proj": proj, "m": m, "err": e, "rel_cublas": rel_cub, "rel": chk, "floor_us": fl0, "arms": {}}
            cub = statistics.median(meds["cublas"])
            mb = n * k * 2 / 1e6
            for nm in names:
                med = statistics.median(meds[nm])
                row["arms"][nm] = {"raw_us": med, "net_us": med - fl0, "rounds": meds[nm],
                                   "x_vs_cublas_net": (cub - fl0) / (med - fl0), "tbs": mb / (med - fl0)}
            # per-CTA stamps for the primary config (one untimed call)
            nwg, bn, nst, cl, _ = VD_INSTS[kid0]
            nrb0 = n // bm0
            flat, w_rb, w_s = packs[(bm0, "front")]
            stamps.zero_()
            torch.sum(flush, dim=0, out=sink)
            mod.vd_run(kid0, x, flat, w, ys[next(iter(ys))][0], err, stamps, nrb0, w_rb, w_s, 32, 0)
            torch.cuda.synchronize()
            sp = stamps[:4 * nrb0 * cl].view(-1, 4).double()
            t0 = float(sp[:, 0].min())
            ent, fst, lop, end = [(sp[:, i] - t0) / 1e3 for i in range(4)]
            q = lambda v: (float(v.min()), float(v.median()), float(v.max()))
            row["stamps_us"] = {"entry": q(ent), "first": q(fst), "loop": q(lop), "end": q(end)}
            # repeat check after timing
            for name, fn in arms.items():
                fn()
            torch.cuda.synchronize()
            row["err_after"] = int(err[0].item())
            row["rel_after"] = {name: float((y.double() - y_ref64).abs().max()) / scale for name, (y, fl) in ys.items()
                                if not (fl & 6)}
            results["rows"].append(row)
            print(f"[vd] {proj} m{m}: floor {fl0:.2f} us; net us (x vs cuBLAS, TB/s bf16-bytes):", flush=True)
            for nm in names:
                a = row["arms"][nm]
                print(f"    {nm:34s} {a['net_us']:7.2f}  x{a['x_vs_cublas_net']:.3f}  {a['tbs']:.2f} TB/s   raw {a['raw_us']:.2f}",
                      flush=True)
            s = row["stamps_us"]
            print(f"[vd] {proj} m{m}: stamps (us from first entry, min/med/max): entry {s['entry'][0]:.2f}/{s['entry'][1]:.2f}/"
                  f"{s['entry'][2]:.2f} first {s['first'][0]:.2f}/{s['first'][1]:.2f}/{s['first'][2]:.2f} loop "
                  f"{s['loop'][0]:.2f}/{s['loop'][1]:.2f}/{s['loop'][2]:.2f} end {s['end'][0]:.2f}/{s['end'][1]:.2f}/"
                  f"{s['end'][2]:.2f}; err after {row['err_after']}", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=1))
    print("[vd] done", flush=True)


if __name__ == "__main__":
    main()

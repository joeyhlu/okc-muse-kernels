"""dgp_lab.py: DGP = DG1's swap-AB SS-wgmma dense GEMM (one-wave cluster split-K, TMA x ring, PDL) fed with lossless 12-bit
P12 weight stage blocks instead of BF16 tiles, T = 32 (cat = [pooler; router hi; lo] FP32 out, o). Step-0 microbench vs the
shipped DG1 (extracted from kernels/lab/k3_slim_4_ab.py) and cuBLAS: judge-style one-call CUDA graphs, 256 MB read-sweep L2
flush before every replay, paired and alternating, medians.

Design (what changes vs DG1, kernels/muse_decode_layer/k3_slim_4.py dg1::dg1_kernel):
  * Stage block (one 64-wide k chunk of a BM = 64 NWG row tile) = per warpgroup 3 planes x 128 threads x 16 B of GD's P12
    (two low-byte planes + one 4-bit code plane: sign + 3-bit exponent offset to a PER-ROW window start; decode of 8 weights =
    AND, IADD, SHF, AND, IADD, 4 PRMT, exact) + a 16 B escape header {n0, n1} + n0 + n1 entries (A-tile byte offset << 16 |
    true bf16 bits) for values outside the row's 16-exponent window, padded to 16 B. ONE cp.async.bulk per stage as in DG1;
    75.5 MB -> 56.6 MB for the cat.
  * Consumer warpgroup g: thread t decodes row 64 g + (t & 63), k half t >> 6 (32 weights: 3 LDS.128) into 4 x 16 B of the
    no-swizzle K-major core-matrix A tile DG1's descriptors already read (4 STS.128, conflict-free), named barrier, escape
    patches (st.shared.u16), fence.proxy.async, barrier, then the same 4 wgmma m64nTk16 SS as DG1. A tile double-buffered
    (wait_group 1 keeps the tile written at step i free of the wgmma issued at step i - 2). x ring, split-K DSMEM reduce,
    PDL, bounded waits: DG1's.
  * RS variant (rs = 1, R-dgp-2): the same stage blocks in wgmma A-FRAGMENT order (thread = warp w, lane l: rows 16 w + l / 4 and
    + 8, k pairs 2 (l % 4) and + 8 of each k16 step), decoded straight into the 16 A registers of the step (double-buffered across
    stages), escapes XOR-patched in registers (GD style), then wgmma m64nTk16 RS: no smem A tile, no fence.proxy.async, no named
    barriers; ring up to 12 stages.
  * Flags (lab): bit0 exit, bit1 no wgmma, bit2 no x, bit3 no L2 hint, bit4 no decode (stale A), bit5 no escape pass, bit6 no
    proxy fence (SS timing probe only: results wrong).

    python3 kernels/lab/dgp_lab.py [--projs=cat,o] [--reps=15] [--layer=kernels/lab/k3_slim_4_ab.py] [--out=box_out/dgp1/lab.json]
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

# (T, NWG) instantiations; kid = index
DGP_INSTS = [(32, 1), (32, 2)]
DGP_EMAX = 320          # escape entries per stage block (smem slot capacity; the pack refuses above)

DGP_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t dgp_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor base, torch::Tensor y, torch::Tensor err, int64_t S,
                int64_t pdl, int64_t trig, int64_t ldy, int64_t flags, int64_t wb, int64_t rs);
std::vector<int64_t> dgp_info(int64_t kid, int64_t S, int64_t rs);
"""

DGP_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace dgp {

constexpr long long WAIT_BUDGET = 2000000000LL;   // ~1 s at 1.98 GHz: a hang becomes err[0] = 1, never a trap
constexpr int EMAX = @EMAX@;                      // escape entries per stage block the smem slot can hold

struct Params {
  const uint8_t* w;      // P12 stage blocks [K/64][N/BM][wb B]: per warpgroup 3 planes x 128 thr x 16 B, then {n0, n1, 0, 0} + entries
  const uint8_t* base;   // [N] per-row exponent window start (h7 units)
  void* y;               // [T, ldy] bf16 (f32 = 0) or fp32 (f32 = 1)
  int* err;              // err[0] = 1 on a bounded-wait timeout
  int N, nk, T, ldy, f32, trig;
  int flags;             // bit0 exit at entry, bit1 no wgmma, bit2 no x loads, bit3 no L2 hint, bit4 no decode, bit5 no escape pass
  int wb;                // bytes per stage block (multiple of 16, <= Cfg::WBS)
};

template <int T, int NWG, bool RSW>
struct Cfg {
  static constexpr int BM = 64 * NWG;                      // weight rows per CTA
  static constexpr int XB = T * 128;                        // x tile bytes (T rows x 64 bf16), SW128
  static constexpr int PB = 6144;                           // packed plane bytes per warpgroup per stage (128 thr x 3 x 16 B)
  static constexpr int WBS = NWG * PB + 16 + 4 * EMAX;      // smem W slot: planes + escape header + entries
  static constexpr int AB = RSW ? 0 : BM * 128;              // one decoded A tile (BM rows x 64 bf16, core matrices); none for RS
  static constexpr int STAGE = XB + WBS;
  static constexpr int ST0 = (220 * 1024 - 2 * AB) / STAGE;
  static constexpr int STMAX = RSW ? 12 : 10;
  static constexpr int ST = ST0 > STMAX ? STMAX : ST0;      // ring depth
  static constexpr int RS = BM + 4;                         // partial row stride (floats): conflict-free fragment stores
  static constexpr int SMEM = 1024 + 2 * AB + ST * STAGE + 16 * ST;   // align slack + A tiles + rings + full/empty barriers
  static constexpr int THREADS = 128 * NWG + 32;
  static_assert(T % 8 == 0 && T >= 8 && T <= 256, "T");
  static_assert(T * RS * 4 <= 2 * AB + ST * XB, "partials fit the A tiles + x ring");
  static_assert(ST >= 2, "ring");
  static_assert(WBS % 16 == 0 && XB % 1024 == 0 && AB % 1024 == 0, "alignment");
};

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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, bool& dead) {
  if (dead || mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); dead = true; return; }
  }
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_x(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t cl_n() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_nctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t r) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(o) : "r"(a), "r"(r));
  return o;
}
__device__ __forceinline__ float4 ld_cl4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
__device__ __forceinline__ void sts16(uint32_t a, uint32_t v) {
  asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(a), "h"((unsigned short)v) : "memory");
}
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void bar_wg(int id) { asm volatile("bar.sync %0, 128;\n" :: "r"(id) : "memory"); }
__device__ __forceinline__ uint64_t desc_il(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(1024u >> 4) << 32);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024u >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db);
template <int N> __device__ __forceinline__ void wgmma_rs(float* d, const uint32_t* a, uint64_t db);
@WGSPECS@
template <int NR> __device__ __forceinline__ void fence_regs(float* d) {
#pragma unroll
  for (int i = 0; i < NR; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

// P12 decode of 8 weights (GD's gv1 FMT 0): L0 / L1 = low bytes of weights 0-3 / 4-7, C = 8 nibble codes, tb = window start per byte
__device__ __forceinline__ void dec8(uint32_t L0, uint32_t L1, uint32_t C, uint32_t tb, uint32_t* o) {
  const uint32_t h0 = (C & 0x87878787u) + tb;
  const uint32_t h1 = (__funnelshift_r(C, C, 4) & 0x87878787u) + tb;
  o[0] = __byte_perm(L0, h0, 0x5140);
  o[1] = __byte_perm(L0, h0, 0x7362);
  o[2] = __byte_perm(L1, h1, 0x5140);
  o[3] = __byte_perm(L1, h1, 0x7362);
}

template <int T, int NWG, bool RSW>
__global__ void __launch_bounds__(Cfg<T, NWG, RSW>::THREADS, 1)
dgp_kernel(const __grid_constant__ CUtensorMap tmx, const Params P) {
  using C = Cfg<T, NWG, RSW>;
  constexpr int ST = C::ST, XB = C::XB, WBS = C::WBS, AB = C::AB, BM = C::BM, RS = C::RS, PB = C::PB;
  extern __shared__ uint8_t dsm[];
  const uint32_t raw = sa(dsm);
  const uint32_t base = (raw + 1023u) & ~1023u;
  float* red = reinterpret_cast<float*>(dsm + (base - raw));
  const uint32_t atiles = base, xring = base + 2 * AB, wring = xring + ST * XB;
  const uint32_t bfull = wring + ST * WBS, bempty = bfull + 8 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cl_rank(), nrank = (int)cl_n();
  const int tile = blockIdx.y, ntile = P.N / BM;
  const int kc0 = (P.nk * rank) / nrank, kc1 = (P.nk * (rank + 1)) / nrank, nloc = kc1 - kc0;
  if (P.trig) pdl_launch();
  if (P.flags & 1) return;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, 4 * NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  bool dead = false;
  float acc[T / 2];
  if (warp == 4 * NWG) {
    if (lane == 0) {
      const uint64_t pol = policy_evict_first();
      const bool nox = (P.flags >> 2) & 1, noh = (P.flags >> 3) & 1;
      const uint32_t wbytes = (uint32_t)P.wb, sbytes = nox ? wbytes : wbytes + (uint32_t)XB;
      const size_t slab = (size_t)ntile * (size_t)P.wb;
      const uint8_t* wsrc = P.w + ((size_t)kc0 * ntile + tile) * (size_t)P.wb;
      const int pre = nloc < ST ? nloc : ST;
      for (int i = 0; i < pre; ++i) {   // weights first: never written by a predecessor
        mbar_expect_tx(bfull + 8 * i, sbytes);
        if (noh) bulk_g2s(wring + i * WBS, wsrc + (size_t)i * slab, wbytes, bfull + 8 * i);
        else bulk_g2s_ef(wring + i * WBS, wsrc + (size_t)i * slab, wbytes, bfull + 8 * i, pol);
      }
      pdl_wait();
      if (!nox)
        for (int i = 0; i < pre; ++i) tma_x(xring + i * XB, &tmx, (kc0 + i) * 64, 0, bfull + 8 * i);
      for (int i = ST; i < nloc; ++i) {
        const int s = i % ST;
        mbar_wait(bempty + 8 * s, (uint32_t)(((i / ST) - 1) & 1), P.err, dead);
        mbar_expect_tx(bfull + 8 * s, sbytes);
        if (noh) bulk_g2s(wring + s * WBS, wsrc + (size_t)i * slab, wbytes, bfull + 8 * s);
        else bulk_g2s_ef(wring + s * WBS, wsrc + (size_t)i * slab, wbytes, bfull + 8 * s, pol);
        if (!nox) tma_x(xring + s * XB, &tmx, (kc0 + i) * 64, 0, bfull + 8 * s);
      }
    }
  } else if constexpr (RSW) {
    const int g = warp >> 2, tg = tid - 128 * g, wq = tg >> 5;
    const bool nomma = (P.flags >> 1) & 1, nodec = (P.flags >> 4) & 1, noesc = (P.flags >> 5) & 1;
    const size_t row0 = (size_t)tile * BM + g * 64 + wq * 16 + (lane >> 2);
    const uint32_t b0 = P.base[row0], b1 = P.base[row0 + 8];
    const uint32_t tb = b0 | (b0 << 8) | (b1 << 16) | (b1 << 24);   // bytes 0 / 1 -> row r0 (regs a0, a2), 2 / 3 -> r0 + 8 (a1, a3)
    const uint32_t poff = (uint32_t)g * PB + (uint32_t)tg * 16u;
#pragma unroll
    for (int q = 0; q < T / 2; ++q) acc[q] = 0.f;
    uint32_t a0[16], a1[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) { a0[q] = 0u; a1[q] = 0u; }
    auto stage = [&](int i, uint32_t* a) {
      const int s = i % ST;
      mbar_wait(bfull + 8 * s, (uint32_t)((i / ST) & 1), P.err, dead);
      const uint32_t ws = wring + s * WBS, ehdr = ws + NWG * PB;
      const uint4 hdr = lds128(ehdr);
      const int ng = (g == 0) ? (int)hdr.x : (int)hdr.y;
      const uint32_t ebase = ehdr + 16u + ((g == 0) ? 0u : 4u * hdr.x);
      if (!nodec) {
        const uint4 p0 = lds128(ws + poff), p1 = lds128(ws + poff + 2048u), p2 = lds128(ws + poff + 4096u);
        dec8(p0.x, p0.y, p2.x, tb, a);
        dec8(p0.z, p0.w, p2.y, tb, a + 4);
        dec8(p1.x, p1.y, p2.z, tb, a + 8);
        dec8(p1.z, p1.w, p2.w, tb, a + 12);
      }
      if (!noesc) {
        for (int e = 0; e < ng; ++e) {   // entry: thread << 12 | reg << 8 | half << 7 | (h7 ^ base)
          const uint32_t ent = lds32(ebase + 4u * (uint32_t)e);
          if ((int)(ent >> 12) == tg) {
            const int rg = (int)((ent >> 8) & 15u);
            const uint32_t v = (ent & 127u) << (8 + (((ent >> 7) & 1u) << 4));
#pragma unroll
            for (int q = 0; q < 16; ++q)
              if (q == rg) a[q] ^= v;
          }
        }
      }
      wg_fence();
      if (!nomma) {
        const uint32_t xs = xring + s * XB;
#pragma unroll
        for (int j = 0; j < 4; ++j) wgmma_rs<T>(acc, a + 4 * j, desc_sw128(xs + j * 32));
      }
      wg_commit();
      wg_wait<1>();
      if (i > 0 && lane == 0) mbar_arrive(bempty + 8 * ((i - 1) % ST));
    };
    for (int i = 0; i < nloc; i += 2) {   // A registers double-buffered: a0 (even stages) / a1 (odd); wait_group 1 frees them in time
      stage(i, a0);
      if (i + 1 < nloc) stage(i + 1, a1);
    }
    wg_wait<0>();
    fence_regs<T / 2>(acc);
  } else {
    const int g = warp >> 2, tg = tid - 128 * g, r = tg & 63, h = tg >> 6;
    const bool nomma = (P.flags >> 1) & 1, nodec = (P.flags >> 4) & 1, noesc = (P.flags >> 5) & 1, nofence = (P.flags >> 6) & 1;
    const uint32_t tb = (uint32_t)P.base[(size_t)tile * BM + g * 64 + r] * 0x01010101u;
    const uint32_t aoff = (uint32_t)((r >> 3) * 1024 + h * 512 + (r & 7) * 16) + (uint32_t)g * 8192u;
    const uint32_t poff = (uint32_t)g * PB + (uint32_t)tg * 16u;
#pragma unroll
    for (int q = 0; q < T / 2; ++q) acc[q] = 0.f;
    for (int i = 0; i < nloc; ++i) {
      const int s = i % ST;
      mbar_wait(bfull + 8 * s, (uint32_t)((i / ST) & 1), P.err, dead);
      const uint32_t ws = wring + s * WBS, at = atiles + (i & 1) * AB;
      const uint32_t ehdr = ws + NWG * PB;
      const uint4 hdr = lds128(ehdr);
      const int ng = (g == 0) ? (int)hdr.x : (int)hdr.y;
      const uint32_t ebase = ehdr + 16u + ((g == 0) ? 0u : 4u * hdr.x);
      if (!nodec) {
        const uint4 p0 = lds128(ws + poff), p1 = lds128(ws + poff + 2048u), p2 = lds128(ws + poff + 4096u);
        uint32_t o[4];
        dec8(p0.x, p0.y, p2.x, tb, o);
        sts128(at + aoff, o[0], o[1], o[2], o[3]);
        dec8(p0.z, p0.w, p2.y, tb, o);
        sts128(at + aoff + 128u, o[0], o[1], o[2], o[3]);
        dec8(p1.x, p1.y, p2.z, tb, o);
        sts128(at + aoff + 256u, o[0], o[1], o[2], o[3]);
        dec8(p1.z, p1.w, p2.w, tb, o);
        sts128(at + aoff + 384u, o[0], o[1], o[2], o[3]);
      }
      if (noesc) {
        if (!nofence) fence_async_smem();
        bar_wg(1 + g);
      } else {
        bar_wg(1 + g);   // every regular store of this warpgroup's tile landed; the pack slot is fully read
        for (int e = tg; e < ng; e += 128) {
          const uint32_t ent = lds32(ebase + 4u * (uint32_t)e);
          sts16(at + (uint32_t)g * 8192u + (ent >> 16), ent & 0xFFFFu);
        }
        if (!nofence) fence_async_smem();
        bar_wg(1 + g);
      }
      wg_fence();
      if (!nomma) {
        const uint32_t xs = xring + s * XB, am = at + (uint32_t)g * 8192u;
#pragma unroll
        for (int j = 0; j < 4; ++j) wgmma_ss<T>(acc, desc_il(am + j * 256), desc_sw128(xs + j * 32));
      }
      wg_commit();
      wg_wait<1>();
      if (i > 0 && lane == 0) mbar_arrive(bempty + 8 * ((i - 1) % ST));
    }
    wg_wait<0>();
    fence_regs<T / 2>(acc);
  }
  pdl_wait();
  __syncthreads();   // every warpgroup is done with the A tiles and the rings before they are overwritten
  if (warp < 4 * NWG) {
    const int g = warp >> 2, w4 = warp & 3;
    const int r0 = g * 64 + w4 * 16 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < T / 8; ++j) {
      const int t0 = 8 * j + 2 * (lane & 3);
      red[t0 * RS + r0] = acc[4 * j];
      red[(t0 + 1) * RS + r0] = acc[4 * j + 1];
      red[t0 * RS + r0 + 8] = acc[4 * j + 2];
      red[(t0 + 1) * RS + r0 + 8] = acc[4 * j + 3];
    }
  }
  if (nrank > 1) cl_sync(); else __syncthreads();
  constexpr int R4 = BM / 4, Q4 = T * R4;
  const int q0 = (Q4 * rank) / nrank, q1 = (Q4 * (rank + 1)) / nrank;
  const uint32_t redA = sa(red);
  for (int q = q0 + tid; q < q1; q += C::THREADS) {
    const int t = q / R4, r4 = q - t * R4;
    float4 sum;
    if (nrank == 1) {
      sum = *reinterpret_cast<const float4*>(red + t * RS + 4 * r4);
    } else {
      const uint32_t off = redA + (uint32_t)(t * RS + 4 * r4) * 4u;
      sum = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int c = 0; c < nrank; ++c) {   // rank order: deterministic
        const float4 v = ld_cl4(mapa(off, (uint32_t)c));
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
      }
    }
    if (t < P.T) {
      const size_t o = (size_t)t * P.ldy + (size_t)tile * BM + 4 * r4;
      if (P.f32) {
        *reinterpret_cast<float4*>(reinterpret_cast<float*>(P.y) + o) = sum;
      } else {
        __nv_bfloat162 a = __floats2bfloat162_rn(sum.x, sum.y), b = __floats2bfloat162_rn(sum.z, sum.w);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(P.y) + o) = u;
      }
    }
  }
  if (nrank > 1) cl_sync();   // keep this CTA's partials alive until every rank has read them
}

}  // namespace dgp

typedef CUresult (*dgp_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static dgp_encode_fn dgp_get_encode() {
  static dgp_encode_fn fn = nullptr;
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
    else fn = reinterpret_cast<dgp_encode_fn>(p);
  }
  return fn;
}

template <int T, int NWG, bool RSW>
static int64_t dgp_launch_t(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& base, const torch::Tensor& y,
                            const torch::Tensor& err, int64_t S, int64_t pdl, int64_t trig, int64_t ldy, int64_t flags, int64_t wb) {
  using C = dgp::Cfg<T, NWG, RSW>;
  auto kern = dgp::dgp_kernel<T, NWG, RSW>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr = true;
  }
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "dgp: x bf16 [T, K] contiguous");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kByte && w.dim() == 1 && w.is_contiguous(), "dgp: packed w uint8 1-D");
  TORCH_CHECK(base.is_cuda() && base.scalar_type() == at::kByte && base.dim() == 1 && base.is_contiguous(), "dgp: base uint8 [N]");
  const int64_t Tact = x.size(0), K = x.size(1), N = base.numel();
  TORCH_CHECK(Tact >= 1 && Tact <= T && K % 64 == 0 && N % C::BM == 0 && N > 0, "dgp: shape not supported by this instantiation");
  TORCH_CHECK(wb % 16 == 0 && wb >= NWG * C::PB + 16 && wb <= C::WBS, "dgp: stage block bytes");
  TORCH_CHECK(w.numel() == (K / 64) * (N / C::BM) * wb, "dgp: packed w size");
  TORCH_CHECK(S >= 1 && S <= 16 && S <= K / 64, "dgp: split");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "dgp: align");
  const bool f32 = y.scalar_type() == at::kFloat;
  TORCH_CHECK(y.is_cuda() && (f32 || y.scalar_type() == at::kBFloat16) && y.dim() == 2 && y.stride(1) == 1 && y.stride(0) == ldy
              && y.size(0) >= Tact && y.size(1) >= N && ldy % 4 == 0 && reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0, "dgp: y");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, "dgp: err");
  dgp_encode_fn enc = dgp_get_encode();
  TORCH_CHECK(enc != nullptr, "dgp: cuTensorMapEncodeTiled unavailable");
  CUtensorMap tm;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)Tact};
  cuuint64_t strides[1] = {(cuuint64_t)(K * 2)};
  cuuint32_t box[2] = {64u, (cuuint32_t)T};
  cuuint32_t es[2] = {1u, 1u};
  const CUresult r = enc(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dgp: tensor map encode failed ", (int)r);
  dgp::Params P;
  P.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  P.base = reinterpret_cast<const uint8_t*>(base.data_ptr());
  P.y = y.data_ptr();
  P.err = err.data_ptr<int>();
  P.N = (int)N; P.nk = (int)(K / 64); P.T = (int)Tact; P.ldy = (int)ldy; P.f32 = f32 ? 1 : 0; P.trig = (int)trig;
  P.flags = (int)flags; P.wb = (int)wb;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)S, (unsigned)(N / C::BM), 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute la[2];
  int na = 0;
  if (S > 1) {
    la[na].id = cudaLaunchAttributeClusterDimension;
    la[na].val.clusterDim.x = (unsigned)S; la[na].val.clusterDim.y = 1; la[na].val.clusterDim.z = 1;
    ++na;
  }
  la[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  la[na].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  ++na;
  cfg.attrs = la;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tm, P));
  return 0;
}

template <int T, int NWG, bool RSW>
static std::vector<int64_t> dgp_info_t(int64_t S) {
  using C = dgp::Cfg<T, NWG, RSW>;
  auto kern = dgp::dgp_kernel<T, NWG, RSW>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)S, 1, 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeClusterDimension;
  la[0].val.clusterDim.x = (unsigned)S; la[0].val.clusterDim.y = 1; la[0].val.clusterDim.z = 1;
  cfg.attrs = la;
  cfg.numAttrs = 1;
  int ncl = -1;
  if (cudaOccupancyMaxActiveClusters(&ncl, kern, &cfg) != cudaSuccess) { cudaGetLastError(); ncl = -1; }
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  return {(int64_t)ncl, (int64_t)C::SMEM, (int64_t)C::THREADS, (int64_t)C::BM, (int64_t)C::ST, (int64_t)fa.numRegs,
          (int64_t)fa.localSizeBytes, (int64_t)C::WBS};
}

int64_t dgp_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor base, torch::Tensor y, torch::Tensor err, int64_t S,
                int64_t pdl, int64_t trig, int64_t ldy, int64_t flags, int64_t wb, int64_t rs) {
  switch (kid) {
@RUNCASES@
  }
  TORCH_CHECK(false, "dgp: unknown kid ", kid);
  return -2;
}

std::vector<int64_t> dgp_info(int64_t kid, int64_t S, int64_t rs) {
  switch (kid) {
@INFOCASES@
  }
  TORCH_CHECK(false, "dgp: unknown kid ", kid);
  return {};
}
"""


def dgp_wg_spec(n):
    nr = n // 2
    regs = ",".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    ss = (f"template <> __device__ __forceinline__ void wgmma_ss<{n}>(float* d, uint64_t da, uint64_t db) {{\n"
          f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
          f'               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
          f"               : {outs}\n               : \"l\"(da), \"l\"(db));\n}}\n")
    rs = (f"template <> __device__ __forceinline__ void wgmma_rs<{n}>(float* d, const uint32_t* a, uint64_t db) {{\n"
          f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
          f'               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{%{nr}, %{nr + 1}, %{nr + 2}, %{nr + 3}}},'
          f' %{nr + 4}, p, 1, 1, 0;\\n}}\\n"\n'
          f"               : {outs}\n               : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(db));\n}}\n")
    return ss + rs


def dgp_cuda_src(insts=DGP_INSTS):
    ns = sorted({t for t, _ in insts})
    run = "\n".join(f"    case {k}: return rs ? dgp_launch_t<{t}, {g}, true>(x, w, base, y, err, S, pdl, trig, ldy, flags, wb)"
                    f" : dgp_launch_t<{t}, {g}, false>(x, w, base, y, err, S, pdl, trig, ldy, flags, wb);" for k, (t, g) in enumerate(insts))
    info = "\n".join(f"    case {k}: return rs ? dgp_info_t<{t}, {g}, true>(S) : dgp_info_t<{t}, {g}, false>(S);"
                     for k, (t, g) in enumerate(insts))
    return (DGP_CUDA_SRC.replace("@EMAX@", str(DGP_EMAX)).replace("@WGSPECS@", "".join(dgp_wg_spec(n) for n in ns))
            .replace("@RUNCASES@", run).replace("@INFOCASES@", info))


def _dgp_row_base(h7, width=8, chunk=1024):
    """[N, K] int32 h7 -> [N] int32: start of the `width` consecutive h7 values holding most of the row's weights."""
    n = h7.shape[0]
    out = torch.empty((n,), dtype=torch.int32, device=h7.device)
    for r0 in range(0, n, chunk):
        hh = h7[r0:r0 + chunk].to(torch.int64)
        hist = torch.zeros((hh.shape[0], 128), dtype=torch.int32, device=h7.device)
        hist.scatter_add_(1, hh, torch.ones_like(hh, dtype=torch.int32))
        cs = torch.cat([torch.zeros((hh.shape[0], 1), dtype=torch.int32, device=h7.device), hist.cumsum(1, dtype=torch.int32)], 1)
        cover = cs[:, width:128 + 1] - cs[:, 0:128 + 1 - width]      # [rows, 121] windows starting at 0..120
        out[r0:r0 + chunk] = cover.argmax(1).to(torch.int32)
    return out


def _dgp_words_u8(words):
    """[..., W] int64 little-endian 32-bit words -> [..., 4 W] uint8."""
    sh = torch.arange(4, device=words.device, dtype=torch.int64) * 8
    return ((words.unsqueeze(-1) >> sh) & 0xFF).to(torch.uint8).reshape(*words.shape[:-1], words.shape[-1] * 4)


def _dgp_rs_index(dev):
    """RS (wgmma A-fragment) order: thread t = warp t >> 5, lane l = t & 31; rows (within the warpgroup's 64) r0 = 16 (t >> 5) + l / 4
    and r0 + 8; group q = k16 step: weights i = 0, 1 -> (r0, 16 q + 2 c + i), 2, 3 -> (r0 + 8, same k), 4, 5 -> (r0, + 8), 6, 7 ->
    (r0 + 8, + 8), c = l & 3. Returns rows [128, 4, 8], kk [128, 4, 8] (k within the 64-wide stage)."""
    t = torch.arange(128, device=dev)
    i = torch.arange(8, device=dev)
    q = torch.arange(4, device=dev)
    r0 = 16 * (t >> 5) + ((t & 31) >> 2)
    rows = (r0.view(128, 1, 1) + 8 * ((i >> 1) & 1).view(1, 1, 8)).expand(128, 4, 8)
    kk = (16 * q.view(1, 4, 1) + 8 * (i >> 2).view(1, 1, 8) + 2 * (t & 3).view(128, 1, 1) + (i & 1).view(1, 1, 8)).expand(128, 4, 8)
    return rows, kk


def dgp_pack(w, nwg, emax=DGP_EMAX, rs=False):
    """W [N, K] bf16 -> (blocks uint8 [K/64 * N/BM * wb], base uint8 [N], wb, stats). Stage block (kc, tile): per warpgroup g
    3 planes x 128 threads x 16 B (SS: thread t = row 64 g + (t & 63), k half t >> 6, k groups q of 8; RS: wgmma A-fragment order,
    _dgp_rs_index; plane 0 = low bytes of groups 0, 1, plane 1 = groups 2, 3, plane 2 = code words), then {n0, n1, 0, 0} and the
    escape entries (SS: A-tile byte offset << 16 | bf16 bits; RS: thread << 12 | reg << 8 | half << 7 | (h7 ^ base)), 16 B padded.
    Raises ValueError when a block needs more than emax entries (prepare falls back loudly)."""
    n, k = w.shape
    bm = 64 * nwg
    if n % bm or k % 64:
        raise ValueError(f"dgp_pack: shape {tuple(w.shape)} not a multiple of ({bm}, 64)")
    ntile, nk = n // bm, k // 64
    dev = w.device
    v = (w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF)                       # [N, K]
    base = _dgp_row_base((v >> 8) & 0x7F)                                                  # [N]
    if rs:
        rows, kk = _dgp_rs_index(dev)
        v6 = v.view(ntile, nwg, 64, nk, 64)[:, :, rows, :, kk].permute(5, 3, 4, 0, 1, 2).reshape(nk, ntile, nwg, 128, 4, 8).to(torch.int64)
        bt = base.view(ntile, nwg, 64)[:, :, rows].unsqueeze(0).to(torch.int64)             # [1, ntile, g, 128, 4, 8]
    else:
        v6 = v.view(ntile, nwg, 64, nk, 2, 4, 8).permute(3, 0, 1, 4, 2, 5, 6).reshape(nk, ntile, nwg, 128, 4, 8).to(torch.int64)
        bt = base.view(1, ntile, nwg, 1, 64).expand(nk, ntile, nwg, 2, 64).reshape(nk, ntile, nwg, 128, 1, 1).to(torch.int64)
    lo = v6 & 0xFF
    sg = (v6 >> 15) & 1
    d = ((v6 >> 8) & 0x7F) - bt
    isesc = (d < 0) | (d > 7)
    d = torch.where(isesc, torch.zeros_like(d), d)
    dA, dB, sA, sB = d[..., :4], d[..., 4:], sg[..., :4], sg[..., 4:]
    sBr = sB[..., [3, 0, 1, 2]]
    cb = dA | (sBr << 3) | (dB << 4) | (sA << 7)                                           # [nk, ntile, g, 128, 4 q, 4 bytes]
    cw = cb[..., 0] | (cb[..., 1] << 8) | (cb[..., 2] << 16) | (cb[..., 3] << 24)          # [.., 4 q]
    la = lo[..., 0] | (lo[..., 1] << 8) | (lo[..., 2] << 16) | (lo[..., 3] << 24)
    lb = lo[..., 4] | (lo[..., 5] << 8) | (lo[..., 6] << 16) | (lo[..., 7] << 24)
    p0 = torch.stack([la[..., 0], lb[..., 0], la[..., 1], lb[..., 1]], dim=-1)
    p1 = torch.stack([la[..., 2], lb[..., 2], la[..., 3], lb[..., 3]], dim=-1)
    planes = torch.stack([p0, p1, cw], dim=3)                                              # [nk, ntile, g, 3, 128, 4 words]
    pbytes = _dgp_words_u8(planes.reshape(nk, ntile, nwg * 3 * 128, 4)).reshape(nk, ntile, nwg * 6144)
    pos = isesc.nonzero()                                                                  # [E, 6]: kc, tile, g, t, q, i
    nesc = int(pos.shape[0])
    blk = pos[:, 0] * ntile + pos[:, 1]
    cnt_g = torch.zeros((nk * ntile * nwg,), dtype=torch.int64, device=dev)
    if nesc:
        cnt_g.scatter_add_(0, blk * nwg + pos[:, 2], torch.ones_like(blk))
    cnt_g = cnt_g.view(nk * ntile, nwg)
    cnt = cnt_g.sum(1)
    nmax = int(cnt.max()) if nesc else 0
    if nmax > emax:
        raise ValueError(f"dgp_pack: a stage block needs {nmax} escape entries > {emax}")
    esz = 16 + 16 * ((nmax * 4 + 15) // 16)
    wb = nwg * 6144 + esz
    out = torch.zeros((nk * ntile, wb), dtype=torch.uint8, device=dev)
    out[:, :nwg * 6144] = pbytes.reshape(nk * ntile, nwg * 6144)
    hdr = torch.zeros((nk * ntile, 4), dtype=torch.int64, device=dev)
    hdr[:, :nwg] = cnt_g
    out[:, nwg * 6144:nwg * 6144 + 16] = _dgp_words_u8(hdr)
    if nesc:
        t_ = pos[:, 3]
        q = pos[:, 4]
        i = pos[:, 5]
        val = v6[pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3], pos[:, 4], pos[:, 5]]
        if rs:
            bw = bt[0][pos[:, 1], pos[:, 2], pos[:, 3], pos[:, 4], pos[:, 5]]
            ent = (t_ << 12) | ((4 * q + (i >> 1)) << 8) | ((i & 1) << 7) | (((val >> 8) & 0x7F) ^ bw)
        else:
            r = t_ & 63
            h = t_ >> 6
            off = (r >> 3) * 1024 + (h * 4 + q) * 128 + (r & 7) * 16 + 2 * i
            ent = (off << 16) | val
        key = blk * nwg + pos[:, 2]
        order = torch.argsort(key, stable=True)
        blk_s, ent_s = blk[order], ent[order]
        start = torch.cumsum(cnt, 0) - cnt
        rank = torch.arange(nesc, device=dev) - start[blk_s]
        col = nwg * 6144 + 16 + 4 * rank
        for b in range(4):
            out[blk_s, col + b] = ((ent_s >> (8 * b)) & 0xFF).to(torch.uint8)
    stats = {"nesc": nesc, "nmax": nmax, "esz": esz, "wb": wb, "bytes": int(out.numel()), "mb": out.numel() / 1e6,
             "bf16_mb": n * k * 2 / 1e6, "rs": int(bool(rs))}
    return out.reshape(-1), base.to(torch.uint8), wb, stats


def dgp_unpack(blocks, base, wb, n, k, nwg, rs=False):
    """Inverse of dgp_pack (same decode as the kernel, escapes applied) -> bf16 [N, K]; used for the bit-exact self-check."""
    bm = 64 * nwg
    ntile, nk = n // bm, k // 64
    dev = blocks.device
    blk = blocks.view(nk * ntile, wb).to(torch.int64)
    pb = blk[:, :nwg * 6144].reshape(nk * ntile, nwg, 3, 128, 16)
    sh = torch.arange(4, device=dev, dtype=torch.int64) * 8
    words = (pb.reshape(nk * ntile, nwg, 3, 128, 4, 4) << sh).sum(-1)                     # [B, g, plane, t, 4 words]
    L = torch.stack([words[..., 0, :, 0], words[..., 0, :, 1], words[..., 0, :, 2], words[..., 0, :, 3],
                     words[..., 1, :, 0], words[..., 1, :, 1], words[..., 1, :, 2], words[..., 1, :, 3]], dim=-1)   # [B, g, t, 8]
    Cw = words[..., 2, :, :]                                                               # [B, g, t, 4]
    b64 = base.to(torch.int64).view(ntile, nwg, 64)
    if rs:
        rows, kk = _dgp_rs_index(dev)
        bb0 = b64[:, :, rows[:, 0, 0]]
        bb1 = b64[:, :, rows[:, 0, 2]]
        tb = (bb0 | (bb0 << 8) | (bb1 << 16) | (bb1 << 24)).unsqueeze(0).expand(nk, ntile, nwg, 128).reshape(nk * ntile, nwg, 128, 1)
    else:
        bt = b64.view(1, ntile, nwg, 1, 64).expand(nk, ntile, nwg, 2, 64).reshape(nk * ntile, nwg, 128, 1)
        tb = bt * 0x01010101
    rot = ((Cw >> 4) | (Cw << 28)) & 0xFFFFFFFF
    h0 = ((Cw & 0x87878787) + tb) & 0xFFFFFFFF
    h1 = ((rot & 0x87878787) + tb) & 0xFFFFFFFF
    vals = torch.empty((nk * ntile, nwg, 128, 4, 8), dtype=torch.int64, device=dev)
    for q in range(4):
        for j in range(4):
            vals[..., q, j] = ((L[..., 2 * q] >> (8 * j)) & 0xFF) | (((h0[..., q] >> (8 * j)) & 0xFF) << 8)
            vals[..., q, 4 + j] = ((L[..., 2 * q + 1] >> (8 * j)) & 0xFF) | (((h1[..., q] >> (8 * j)) & 0xFF) << 8)
    hdr = (blk[:, nwg * 6144:nwg * 6144 + 16].reshape(nk * ntile, 4, 4) << sh).sum(-1)      # [B, 4]
    nent = hdr[:, :nwg]
    tot = nent.sum(1)
    nmax = int(tot.max())
    if nmax:
        ents = (blk[:, nwg * 6144 + 16:nwg * 6144 + 16 + 4 * nmax].reshape(nk * ntile, nmax, 4) << sh).sum(-1)   # [B, nmax]
        idx = torch.arange(nmax, device=dev).view(1, -1)
        valid = idx < tot.view(-1, 1)
        g_of = (idx >= nent[:, 0:1]).to(torch.int64) if nwg > 1 else torch.zeros_like(idx).expand(nk * ntile, nmax)
        b_idx, e_idx = valid.nonzero(as_tuple=True)
        e = ents[b_idx, e_idx]
        g = g_of[b_idx, e_idx]
        if rs:
            t = e >> 12
            rg = (e >> 8) & 15
            q = rg >> 2
            i = 2 * (rg & 3) + ((e >> 7) & 1)
            vals[b_idx, g, t, q, i] = vals[b_idx, g, t, q, i] ^ ((e & 127) << 8)
        else:
            off = e >> 16
            rg, rem = off // 1024, off % 1024
            c, rem2 = rem // 128, rem % 128
            r8, i2 = rem2 // 16, rem2 % 16
            r = rg * 8 + r8
            h, q = c // 4, c % 4
            t = h * 64 + r
            vals[b_idx, g, t, q, i2 // 2] = e & 0xFFFF
    if rs:
        kc_i = torch.arange(nk, device=dev).view(nk, 1, 1, 1, 1, 1)
        tile_i = torch.arange(ntile, device=dev).view(1, ntile, 1, 1, 1, 1)
        g_i = torch.arange(nwg, device=dev).view(1, 1, nwg, 1, 1, 1)
        idx = (((tile_i * nwg + g_i) * 64 + rows.view(1, 1, 1, 128, 4, 8)) * nk + kc_i) * 64 + kk.view(1, 1, 1, 128, 4, 8)
        flat = torch.empty((n * k,), dtype=torch.int64, device=dev)
        flat[idx.reshape(-1)] = vals.reshape(-1)
        v6 = flat.view(n, k)
    else:
        v6 = vals.view(nk, ntile, nwg, 2, 64, 4, 8).permute(1, 2, 4, 0, 3, 5, 6).reshape(n, k)
    v6 = torch.where(v6 >= 32768, v6 - 65536, v6)   # two's complement int16 bits
    return v6.to(torch.int16).view(torch.bfloat16)


def dgp_build(insts=DGP_INSTS, name="dgplab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda = dgp_cuda_src(insts)
    tag = hashlib.sha1((DGP_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=DGP_CPP_SRC, cuda_sources=cuda, functions=["dgp_run", "dgp_info"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
                      verbose=verbose)
    return mod, time.time() - t0


# ---------------------------------------------------------------- lab
PROJ = {"qkv": (2304, 8192), "o": (8192, 1024), "cat": (4608, 8192), "unpooler": (8192, 4096), "pooler": (4096, 8192)}
WNAME = {"qkv": "qkv_gate", "o": "o", "pooler": "pooler", "unpooler": "unpooler"}
GEOS = {("cat", 1): [(2, 3), (2, 2)], ("o", 1): [(1, 1), (2, 2)], ("cat", 0): [(2, 3)], ("o", 0): []}   # (proj, rs) -> (NWG, S)


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def lab_weights(dev, gen=None):
    """the reference's dense recipe (randn / sqrt(k), bf16) + the cat = [pooler; router hi; router lo]."""
    def rn(shape):
        return torch.randn(shape, generator=gen, device=dev) if gen is not None else torch.randn(shape, device=dev)
    W = {WNAME[p]: (rn(PROJ[p]) / math.sqrt(PROJ[p][1])).to(torch.bfloat16) for p in WNAME}
    router = 0.045 * rn((256, 8192))
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    W["cat"] = torch.cat((W["pooler"], hi, lo), dim=0).contiguous()
    return W


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
    ns = {"torch": torch, "time": time, "hashlib": hashlib, "math": math, "_pt1_wait": lambda name: None, "_KS1": {}}
    ns["_k3_disable"] = lambda state, key, value, piece, ex: state.__setitem__(key, value)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)   # lab tool, our own layer source
    return ns


def dg1_reference(layer_path, W):
    ns = layer_extract(layer_path, ("DG1_", "_DG1", "_dg1", "dg1_"))
    ns["DG1_ROUTE"] = {("cat", 32): True, ("o", 32): True}
    ns["DG1_PDL"] = False
    state = {"dense": {k: W[k] for k in WNAME.values()}, "v8": {"wcat": W["cat"]}}
    ns["_dg1_prepare"](state, time.time())
    return ns, state


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


def time_pair(ga, gb, flush, sink, reps):
    """paired, alternating order, read-sweep flush before each replay; medians (us)."""
    ta, tb = [], []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(reps + 2):
        t = [0.0, 0.0]
        for which in ((0, 1) if r % 2 == 0 else (1, 0)):
            torch.sum(flush, dim=0, out=sink)
            e0.record()
            (ga if which == 0 else gb).replay()
            e1.record()
            torch.cuda.synchronize()
            t[which] = e0.elapsed_time(e1) * 1e3
        if r >= 2:
            ta.append(t[0])
            tb.append(t[1])
    return statistics.median(ta), statistics.median(tb)


def onehot_check(run, w, T, k, n, f32, dev, err):
    """every weight through the real kernel: x = one-hot rows -> y[t, :] == w[:, k_t] bit for bit. Returns (bad count, err flag)."""
    bad, errf = 0, 0
    y = torch.empty((T, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
    x = torch.zeros((T, k), dtype=torch.bfloat16, device=dev)
    wt = w.t()
    for k0 in range(0, k, T):
        x.zero_()
        x[torch.arange(T, device=dev), k0 + torch.arange(T, device=dev)] = 1.0
        run(x, y)
        ref = wt[k0:k0 + T]
        bad += int((y.float() != ref.float()).sum())   # float compare: +0 == -0 (an fp32 sum of one product cannot keep -0)
    torch.cuda.synchronize()
    errf = int(err[0])
    err.zero_()
    return bad, errf


def main():
    dev = torch.device("cuda")
    projs = arg("projs", "cat,o").split(",")
    reps = int(arg("reps", "15"))
    T = 32
    layer = arg("layer", str(ROOT / "kernels" / "lab" / "k3_slim_4_ab.py"))
    out_path = arg("out", "")
    modes = [int(v) for v in arg("rs", "1,0").split(",")]   # 1 = RS consumer, 0 = SS consumer
    gen = torch.Generator(device=dev).manual_seed(1)
    W = lab_weights(dev, gen)
    mod, bs = dgp_build(verbose=arg("verbose", "0") == "1")
    print(f"[dgp] build {bs:.1f}s", flush=True)
    for rs in (1, 0):
        for kid, (t, g) in enumerate(DGP_INSTS):
            line = f"[dgp] {'RS' if rs else 'SS'} kid {kid} T={t} NWG={g}:"
            for s in (1, 2, 3, 4, 8):
                i = [int(v) for v in mod.dgp_info(kid, s, rs)]
                if s == 1:
                    line += f" BM {i[3]} ST {i[4]} smem {i[1]} thr {i[2]} regs {i[5]} lmem {i[6]} WBS {i[7]} | max clusters:"
                line += f" S{s}={i[0]}"
            print(line, flush=True)
    dg1_ns, dg1_state = (None, None)
    if int(arg("dg1", "1")):
        try:
            dg1_ns, dg1_state = dg1_reference(layer, W)
        except Exception as ex:  # noqa: BLE001
            print(f"[dgp] DG1 reference FAILED: {ex!r}", flush=True)
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    flush.normal_()
    sink = torch.empty((), dtype=torch.float32, device=dev)
    err = torch.zeros(4, dtype=torch.int32, device=dev)
    tiny = torch.zeros(16, device=dev)
    g_tiny = graph_of(lambda: tiny.add_(1.0))
    rows = []
    for proj in projs:
        w = W["cat"] if proj == "cat" else W[WNAME[proj]]
        n, k = w.shape
        f32 = proj == "cat"
        x = torch.randn((T, k), generator=gen, device=dev).to(torch.bfloat16)
        x3 = torch.randn((T - 3, k), generator=gen, device=dev).to(torch.bfloat16)

        def mm(x=x, w=w, f32=f32):
            return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        ref = mm()
        g_ref = graph_of(mm)
        tr0, tt = time_pair(g_ref, g_tiny, flush, sink, reps)
        print(f"[dgp] {proj} T={T} ({n}x{k}, {n * k * 2 / 1e6:.1f} MB bf16): cublas {tr0:.2f} us, tiny graph {tt:.2f} us", flush=True)
        packs = {}
        for rs in modes:
            for nwg, _ in GEOS[(proj, rs)]:
                if (nwg, rs) in packs:
                    continue
                t0 = time.time()
                try:
                    blocks, base, wb, stats = dgp_pack(w, nwg, rs=bool(rs))
                except ValueError as ex:
                    print(f"[dgp]   pack NWG={nwg} rs={rs} REFUSED: {ex}", flush=True)
                    continue
                torch.cuda.synchronize()
                tp = time.time() - t0
                back = dgp_unpack(blocks, base, wb, n, k, nwg, rs=bool(rs))
                exact = bool(torch.equal(back.view(torch.int16), w.view(torch.int16)))
                nb = int((back.view(torch.int16) != w.view(torch.int16)).sum())
                print(f"[dgp]   pack NWG={nwg} rs={rs}: {stats['mb']:.1f} MB (x{stats['mb'] / stats['bf16_mb']:.3f} of bf16), wb {wb} esz {stats['esz']}"
                      f" escapes {stats['nesc']} (max/block {stats['nmax']}), {tp:.2f}s; torch unpack bit-exact {exact} (bad {nb})", flush=True)
                packs[(nwg, rs)] = (blocks, base, wb, stats, exact)
        for rs in modes:
            for nwg, s in GEOS[(proj, rs)]:
                if (nwg, rs) not in packs:
                    continue
                blocks, base, wb, stats, exact = packs[(nwg, rs)]
                kid = DGP_INSTS.index((T, nwg))
                info = [int(v) for v in mod.dgp_info(kid, s, rs)]
                bm = info[3]
                ntile = n // bm
                if n % bm:
                    continue
                one_wave = info[0] >= ntile
                tagg = f"dgp-{'RS' if rs else 'SS'} NWG{nwg} S{s} grid {ntile * s}{'' if one_wave else ' (2 waves)'}"

                def run(x_, y_, fl=0, kid=kid, blocks=blocks, base=base, s=s, wb=wb, rs=rs):
                    mod.dgp_run(kid, x_, blocks, base, y_, err, s, 1, 1, n, fl, wb, rs)
                ok = True
                msgs = []
                for xx in (x, x3):
                    outs = []
                    for _ in range(2):
                        yy = torch.full((xx.shape[0], n), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
                        run(xx, yy)
                        outs.append(yy)
                    torch.cuda.synchronize()
                    r64 = xx.double() @ w.double().t()
                    today = torch.mm(xx, w.t(), out_dtype=torch.float32) if f32 else torch.mm(xx, w.t())
                    fin = bool(torch.isfinite(outs[0]).all())
                    e64 = float((outs[0].double() - r64).norm() / r64.norm()) if fin else float("inf")
                    et = float((outs[0].float() - today.float()).norm() / today.float().norm()) if fin else float("inf")
                    same = bool(torch.equal(outs[0], outs[1]))
                    errf = int(err[0])
                    err.zero_()
                    good = fin and same and errf == 0 and e64 < (1e-4 if f32 else 4e-3) and et < 1e-2
                    ok &= good
                    msgs.append(f"T={xx.shape[0]} rel64 {e64:.1e} vs cublas {et:.1e} finite {fin} same {same} err {errf}")
                bad, errf = onehot_check(run, w, T, k, n, f32, dev, err)
                ok &= bad == 0 and errf == 0
                print(f"[dgp]   {tagg}: BM {bm} ST {info[4]} regs {info[5]} lmem {info[6]} clusters {info[0]}: {'; '.join(msgs)}; one-hot bad {bad}"
                      f" err {errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                y = torch.empty((T, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
                if rs:
                    arms = ((0, "full"), (2, "no-wgmma"), (4, "no-x"), (16, "no-decode"), (32, "no-escape-pass"),
                            (22, "W-only (no x/wgmma/decode)"), (1, "exit at entry"))
                else:
                    arms = ((0, "full"), (64, "no-fence (wrong results)"), (96, "no-fence no-escape"), (16, "no-decode"),
                            (80, "no-decode no-fence"), (22, "W-only (no x/wgmma/decode)"), (1, "exit at entry"))
                for fl, name in arms:
                    try:
                        g = graph_of(lambda fl=fl: run(x, y, fl))
                        tr, td = time_pair(g_ref, g, flush, sink, reps)
                        net = td - tt
                        mb = stats["mb"]
                        print(f"[dgp]     {tagg} {name:28s}: {td:7.2f} us (cublas {tr:6.2f}, x{tr / td:.3f}); net {net:6.2f} us ="
                              f" {mb / max(net, 1e-3):5.2f} TB/s on P12 bytes", flush=True)
                        rows.append({"proj": proj, "arm": "dgp", "rs": rs, "nwg": nwg, "S": s, "flags": fl, "us": td, "net": net,
                                     "cublas": tr, "ok": ok, "mb": mb})
                        del g
                    except Exception as ex:  # noqa: BLE001
                        print(f"[dgp]     {tagg} {name}: raised {ex!r}", flush=True)
                        torch.cuda.synchronize()
        # shipped DG1 (k3_slim_4 geometry) and its W-only ceiling
        if dg1_state is not None and dg1_state.get("dg1") is not None:
            st = dg1_state["dg1"]
            r = st["route"].get((proj, 32))
            if r is None:
                print(f"[dgp]   DG1 route ({proj}, 32) not live in the layer reference", flush=True)
            else:
                y = torch.empty((T, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
                for fl, name in ((0, "full"), (2, "no-wgmma"), (6, "W-only"), (1, "exit at entry")):
                    def run_dg(fl=fl, r=r, y=y):
                        st["mod"].dg1_run(r["kid"], x, st["w"][proj], y, st["err"], r["S"], 1, 1, r["N"], fl)
                    g = graph_of(run_dg)
                    tr, td = time_pair(g_ref, g, flush, sink, reps)
                    net = td - tt
                    mb = n * k * 2 / 1e6
                    print(f"[dgp]     dg1 ship S{r['S']} {name:28s}: {td:7.2f} us (cublas {tr:6.2f}, x{tr / td:.3f}); net {net:6.2f} us ="
                          f" {mb / max(net, 1e-3):5.2f} TB/s on bf16 bytes", flush=True)
                    rows.append({"proj": proj, "arm": "dg1", "S": r["S"], "flags": fl, "us": td, "net": net, "cublas": tr, "mb": mb})
                    del g
        del g_ref
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"rows": rows}, indent=1))
    print("[dgp] done", flush=True)


if __name__ == "__main__":
    main()

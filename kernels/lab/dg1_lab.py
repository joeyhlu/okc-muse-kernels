"""dg1_lab.py: DG1 = dense BF16 projection GEMM for the decode layer at T = 32 / 128 / 256 (qkv, o, cat FP32-out, unpooler),
standalone vs torch.mm (cuBLAS), judge-style (one-call CUDA graph, 256 MB read-sweep L2 flush before every replay, paired and
alternating, median of 21).

Design (swap-AB, one wave, in-kernel split-K):
  * y^T[N, T] = W[N, K] . x^T: wgmma m64nTk16 with A = weight rows (SS, no-swizzle K-major core matrices) and B = tokens
    (TMA SW128 tile [T x 64]); FP32 accumulate in registers (T / 2 floats per m64 tile per thread).
  * CTA = NWG consumer warpgroups (MT m64 tiles each, BM = 64 MT NWG weight rows) + 1 producer warp. Ring of ST stages, each
    = one 64-wide k chunk: W tile (BM x 64, prepacked at prepare time so that it is ONE contiguous cp.async.bulk of BM x 128 B,
    L2 evict_first) + x tile (TMA 2D, box 64 x T, 128B swizzle; tensor map encoded per call on the host, never in prepare).
  * Split-K = thread-block cluster (S, 1, 1): CTA rank r streams k chunks [r nk / S, (r + 1) nk / S). Partials go to the
    CTA's own smem ([T][BM + 4] fp32, reusing the ring), cluster barrier, then rank r sums slice r of every rank's partial in
    rank order 0..S-1 through DSMEM (ld.shared::cluster) and stores it: deterministic, no workspace, no atomics, no 2nd kernel.
  * PDL: launched with programmatic stream serialization; the first ST weight tiles are requested BEFORE griddepcontrol.wait
    (weights are never written by a predecessor); x loads and every global store after it. launch_dependents at entry (trig,
    one-wave grids only). Every mbarrier wait is bounded (R9): timeout -> err[0] = 1, no trap.

    python3 kernels/lab/dg1_lab.py [--ts=32,128,256] [--projs=qkv,cat,o,unpooler] [--reps=21] [--build-only=1]
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

# (T, NWG, MT) instantiations; kid = index
DG1_INSTS = [(32, 1, 1), (32, 2, 1), (32, 2, 2), (128, 1, 1), (128, 2, 1), (128, 2, 2), (256, 1, 1), (256, 2, 1)]

DG1_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t dg1_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor err, int64_t S, int64_t pdl,
                int64_t trig, int64_t ldy, int64_t flags);
std::vector<int64_t> dg1_info(int64_t kid, int64_t S);
"""

DG1_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace dg1 {

constexpr long long WAIT_BUDGET = 2000000000LL;   // ~1 s at 1.98 GHz: a hang becomes err[0] = 1, never a trap

struct Params {
  const uint8_t* w;   // packed weights [K/64][N/8][8 kc][8 rows][16 B]
  void* y;            // [T, ldy] bf16 (f32 = 0) or fp32 (f32 = 1)
  int* err;           // err[0] = 1 on a bounded-wait timeout
  int N, nk, T, ldy, f32, trig;
  int flags;          // lab: bit0 exit at entry, bit1 skip wgmma, bit2 skip x loads, bit3 no L2 hint; bit8 tile-major W layout
};

template <int T, int NWG, int MT>
struct Cfg {
  static constexpr int BM = 64 * MT * NWG;                 // weight rows per CTA
  static constexpr int XB = T * 128;                        // x tile bytes (T rows x 64 bf16), SW128
  static constexpr int WB = BM * 128;                       // W tile bytes (BM rows x 64 bf16), core matrices
  static constexpr int STAGE = XB + WB;
  static constexpr int ST0 = (220 * 1024) / STAGE;
  static constexpr int ST = ST0 > 10 ? 10 : ST0;            // ring depth
  static constexpr int RS = BM + 4;                         // partial row stride (floats): conflict-free fragment stores
  static constexpr int SMEM = 1024 + ST * STAGE + 16 * ST;  // align slack + ring + full/empty barriers
  static constexpr int THREADS = 128 * NWG + 32;
  static_assert(T % 8 == 0 && T >= 8 && T <= 256, "T");
  static_assert(T * RS * 4 <= ST * STAGE, "partials fit the ring");
  static_assert(ST >= 2, "ring");
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
// wgmma smem descriptors. A (weights): no-swizzle K-major core matrices (8 rows x 16 B contiguous), K-adjacent 128 B apart
// (LBO), M-adjacent 8-row groups 1024 B apart (SBO). B (x): 128B-swizzle K-major (TMA SWIZZLE_128B), SBO 1024 B, LBO unused.
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
@WGSPECS@
template <int NR> __device__ __forceinline__ void fence_regs(float* d) {
#pragma unroll
  for (int i = 0; i < NR; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

template <int T, int NWG, int MT>
__global__ void __launch_bounds__(Cfg<T, NWG, MT>::THREADS, 1)
dg1_kernel(const __grid_constant__ CUtensorMap tmx, const Params P) {
  using C = Cfg<T, NWG, MT>;
  constexpr int ST = C::ST, STAGE = C::STAGE, XB = C::XB, BM = C::BM, RS = C::RS;
  extern __shared__ uint8_t dsm[];
  const uint32_t raw = sa(dsm);
  const uint32_t base = (raw + 1023u) & ~1023u;
  float* red = reinterpret_cast<float*>(dsm + (base - raw));
  const uint32_t bfull = base + ST * STAGE, bempty = bfull + 8 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cl_rank(), nrank = (int)cl_n();
  const int tile = blockIdx.y;
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
  float acc[MT][T / 2];
  if (warp == 4 * NWG) {
    // ---------------- producer (one lane)
    if (lane == 0) {
      const uint64_t pol = policy_evict_first();
      const bool tl = (P.flags >> 8) & 1, nox = (P.flags >> 2) & 1, noh = (P.flags >> 3) & 1;
      const uint32_t sbytes = nox ? (uint32_t)C::WB : (uint32_t)STAGE;
      // slab-major [K/64][N/8][..]: k-chunk stride N x 128 B; tile-major [N/BM][K/64][BM x 64]: one contiguous run per CTA
      const size_t slab = tl ? (size_t)C::WB : (size_t)P.N * 128;
      const uint8_t* wsrc = tl ? P.w + ((size_t)tile * P.nk + kc0) * C::WB : P.w + (size_t)kc0 * slab + (size_t)tile * (BM * 128);
      const int pre = nloc < ST ? nloc : ST;
      for (int i = 0; i < pre; ++i) {   // weights first: never written by a predecessor
        mbar_expect_tx(bfull + 8 * i, sbytes);
        if (noh) bulk_g2s(base + i * STAGE + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * i);
        else bulk_g2s_ef(base + i * STAGE + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * i, pol);
      }
      pdl_wait();
      if (!nox)
        for (int i = 0; i < pre; ++i) tma_x(base + i * STAGE, &tmx, (kc0 + i) * 64, 0, bfull + 8 * i);
      for (int i = ST; i < nloc; ++i) {
        const int s = i % ST;
        mbar_wait(bempty + 8 * s, (uint32_t)(((i / ST) - 1) & 1), P.err, dead);
        const uint32_t st = base + s * STAGE;
        mbar_expect_tx(bfull + 8 * s, sbytes);
        if (noh) bulk_g2s(st + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * s);
        else bulk_g2s_ef(st + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * s, pol);
        if (!nox) tma_x(st, &tmx, (kc0 + i) * 64, 0, bfull + 8 * s);
      }
    }
  } else {
    // ---------------- consumers: warpgroup g owns weight rows [g MT 64, (g + 1) MT 64) of the tile
    const int g = warp >> 2;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int q = 0; q < T / 2; ++q) acc[mt][q] = 0.f;
    for (int i = 0; i < nloc; ++i) {
      const int s = i % ST;
      mbar_wait(bfull + 8 * s, (uint32_t)((i / ST) & 1), P.err, dead);
      const uint32_t xs = base + s * STAGE, ws = xs + XB + g * (MT * 8192);
      wg_fence();
      if (!(P.flags & 2)) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int j = 0; j < 4; ++j) wgmma_ss<T>(acc[mt], desc_il(ws + mt * 8192 + j * 256), desc_sw128(xs + j * 32));
      }
      wg_commit();
      wg_wait<1>();
      if (i > 0 && lane == 0) mbar_arrive(bempty + 8 * ((i - 1) % ST));
    }
    wg_wait<0>();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) fence_regs<T / 2>(acc[mt]);
  }
  // ---------------- epilogue: partials -> own smem, cluster barrier, ordered DSMEM sum of this rank's slice, store
  pdl_wait();
  __syncthreads();   // every warpgroup is done reading the ring before it is overwritten
  if (warp < 4 * NWG) {
    const int g = warp >> 2, w4 = warp & 3;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int r0 = g * MT * 64 + mt * 64 + w4 * 16 + (lane >> 2);
#pragma unroll
      for (int j = 0; j < T / 8; ++j) {
        const int t0 = 8 * j + 2 * (lane & 3);
        red[t0 * RS + r0] = acc[mt][4 * j];
        red[(t0 + 1) * RS + r0] = acc[mt][4 * j + 1];
        red[t0 * RS + r0 + 8] = acc[mt][4 * j + 2];
        red[(t0 + 1) * RS + r0 + 8] = acc[mt][4 * j + 3];
      }
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

}  // namespace dg1

typedef CUresult (*dg1_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
// Driver entry point without linking libcuda (kernels/lab/stream_lab.py's sl_get_encode, as CUTLASS does).
static dg1_encode_fn dg1_get_encode() {
  static dg1_encode_fn fn = nullptr;
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
    else fn = reinterpret_cast<dg1_encode_fn>(p);
  }
  return fn;
}

template <int T, int NWG, int MT>
static int64_t dg1_launch_t(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& y, const torch::Tensor& err,
                            int64_t S, int64_t pdl, int64_t trig, int64_t ldy, int64_t flags) {
  using C = dg1::Cfg<T, NWG, MT>;
  auto kern = dg1::dg1_kernel<T, NWG, MT>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr = true;
  }
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "dg1: x bf16 [T, K] contiguous");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.dim() == 5 && w.is_contiguous(), "dg1: packed w [K/64, N/8, 8, 8, 8]");
  const int64_t Tact = x.size(0), K = x.size(1), N = w.size(1) * 8;
  TORCH_CHECK(w.size(0) * 64 == K && w.size(2) == 8 && w.size(3) == 8 && w.size(4) == 8, "dg1: packed w shape");
  TORCH_CHECK(Tact >= 1 && Tact <= T && K % 64 == 0 && N % C::BM == 0, "dg1: shape not supported by this instantiation");
  TORCH_CHECK(S >= 1 && S <= 16 && S <= K / 64, "dg1: split");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "dg1: align");
  const bool f32 = y.scalar_type() == at::kFloat;
  TORCH_CHECK(y.is_cuda() && (f32 || y.scalar_type() == at::kBFloat16) && y.dim() == 2 && y.stride(1) == 1 && y.stride(0) == ldy
              && y.size(0) >= Tact && y.size(1) >= N && ldy % 4 == 0 && reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0, "dg1: y");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, "dg1: err");
  dg1_encode_fn enc = dg1_get_encode();
  TORCH_CHECK(enc != nullptr, "dg1: cuTensorMapEncodeTiled unavailable");
  CUtensorMap tm;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)Tact};
  cuuint64_t strides[1] = {(cuuint64_t)(K * 2)};
  cuuint32_t box[2] = {64u, (cuuint32_t)T};
  cuuint32_t es[2] = {1u, 1u};
  const CUresult r = enc(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dg1: tensor map encode failed ", (int)r);
  dg1::Params P;
  P.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  P.y = y.data_ptr();
  P.err = err.data_ptr<int>();
  P.N = (int)N; P.nk = (int)(K / 64); P.T = (int)Tact; P.ldy = (int)ldy; P.f32 = f32 ? 1 : 0; P.trig = (int)trig;
  P.flags = (int)flags;
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

template <int T, int NWG, int MT>
static std::vector<int64_t> dg1_info_t(int64_t S) {
  using C = dg1::Cfg<T, NWG, MT>;
  auto kern = dg1::dg1_kernel<T, NWG, MT>;
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
          (int64_t)fa.localSizeBytes};
}

int64_t dg1_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor err, int64_t S, int64_t pdl,
                int64_t trig, int64_t ldy, int64_t flags) {
  switch (kid) {
@RUNCASES@
  }
  TORCH_CHECK(false, "dg1: unknown kid ", kid);
  return -2;
}

std::vector<int64_t> dg1_info(int64_t kid, int64_t S) {
  switch (kid) {
@INFOCASES@
  }
  TORCH_CHECK(false, "dg1: unknown kid ", kid);
  return {};
}
"""


def dg1_wg_spec(n):
    """wgmma m64n{n}k16 bf16 SS (A and B from smem descriptors, both K-major), FP32 accumulate in place."""
    nr = n // 2
    regs = ",".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f"template <> __device__ __forceinline__ void wgmma_ss<{n}>(float* d, uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f"               : {outs}\n               : \"l\"(da), \"l\"(db));\n}}\n")


def dg1_cuda_src(insts=DG1_INSTS):
    ns = sorted({t for t, _, _ in insts})
    run = "\n".join(f"    case {k}: return dg1_launch_t<{t}, {g}, {m}>(x, w, y, err, S, pdl, trig, ldy, flags);"
                    for k, (t, g, m) in enumerate(insts))
    info = "\n".join(f"    case {k}: return dg1_info_t<{t}, {g}, {m}>(S);" for k, (t, g, m) in enumerate(insts))
    return (DG1_CUDA_SRC.replace("@WGSPECS@", "".join(dg1_wg_spec(n) for n in ns))
            .replace("@RUNCASES@", run).replace("@INFOCASES@", info))


def dg1_build(insts=DG1_INSTS, name="dg1lab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda = dg1_cuda_src(insts)
    tag = hashlib.sha1((DG1_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=DG1_CPP_SRC, cuda_sources=cuda, functions=["dg1_run", "dg1_info"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
                      verbose=verbose)
    return mod, time.time() - t0


def dg1_pack_tiles(w, bm):
    """Tile-major variant: [N/BM tiles][K/64 slabs][BM/8][8 kc][8 rows][8 elems]; a CTA's k-range is ONE contiguous run. Returned
    with the slab-major tensor's 5-D shape [K/64, N/8, 8, 8, 8] (only N, K are read from it)."""
    n, k = w.shape
    assert n % bm == 0 and k % 64 == 0
    t = w.contiguous().view(n // bm, bm // 8, 8, k // 64, 8, 8).permute(0, 3, 1, 4, 2, 5).contiguous()
    return t.view(k // 64, n // 8, 8, 8, 8)


def dg1_pack(w):
    """W [N, K] bf16 -> [K/64 slabs][N/8 groups][8 k-chunks][8 rows][8 elems] (lossless permutation): a BM-row tile of one
    64-wide slab is ONE contiguous BM x 128 B block in wgmma's no-swizzle K-major core-matrix layout."""
    n, k = w.shape
    assert n % 8 == 0 and k % 64 == 0
    return w.contiguous().view(n // 8, 8, k // 64, 8, 8).permute(2, 0, 3, 1, 4).contiguous()


# ---------------------------------------------------------------- lab
PROJ = {"qkv": (2304, 8192), "o": (8192, 1024), "cat": (4608, 8192), "unpooler": (8192, 4096)}


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


def time_pair(ga, gb, flush, sink, reps):
    """tools/box/tunable_lab.py's time_pair: paired, alternating order, read-sweep flush before each replay; medians (us)."""
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


def cands(proj, T, insts):
    n, k = PROJ[proj]
    nk = k // 64
    out = []
    for kid, (t, g, m) in enumerate(insts):
        if t != T:
            continue
        bm = 64 * g * m
        if n % bm:
            continue
        tiles = n // bm
        for s in range(1, 9):
            if s <= nk and 56 <= tiles * s <= 132:
                out.append((kid, g, m, s))
    return out


def main():
    dev = torch.device("cuda")
    ts = [int(v) for v in arg("ts", "32,128,256").split(",")]
    projs = arg("projs", "qkv,cat,o,unpooler").split(",")
    reps = int(arg("reps", "21"))
    only = arg("only", "")   # "kid:S,kid:S" restricts the sweep
    mod, bs = dg1_build(verbose=arg("verbose", "0") == "1")
    print(f"[dg1] build {bs:.1f}s", flush=True)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    for kid, (t, g, m) in enumerate(DG1_INSTS):
        for s in (1, 2, 3, 4, 6, 8):
            i = mod.dg1_info(kid, s)
            if s == 1:
                print(f"[dg1] kid {kid} T={t} NWG={g} MT={m}: BM {i[3]} ST {i[4]} smem {i[1]} thr {i[2]} regs {i[5]} lmem {i[6]}"
                      f" | max clusters:", end="", flush=True)
            print(f" S{s}={i[0]}", end="", flush=True)
        print("", flush=True)
    if arg("build-only", "0") == "1":
        return
    if arg("exp", "0") == "1":   # iteration 2: what caps the stream? diag flags + tile-major layout on the gate shapes
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
        flush.normal_()
        sink = torch.empty((), dtype=torch.float32, device=dev)
        err = torch.zeros(4, dtype=torch.int32, device=dev)
        tiny = torch.zeros(16, device=dev)
        g_tiny = graph_of(lambda: tiny.add_(1.0))
        exps = [("qkv", 32, 1, 4), ("qkv", 32, 0, 3), ("cat", 32, 1, 3), ("cat", 32, 1, 2), ("qkv", 128, 4, 5), ("qkv", 128, 3, 3),
                ("cat", 128, 4, 3), ("cat", 128, 4, 2), ("o", 32, 0, 1)]
        flagsets = [(0, "base"), (256, "tile-major"), (8, "no-L2-hint"), (2, "no-wgmma"), (6, "W-only stream"),
                    (262, "W-only tile-major"), (1, "exit at entry")]
        ws = {}
        for proj, T, kid, s in exps:
            n, k = PROJ[proj]
            if proj not in ws:
                ws[proj] = (torch.randn((n, k), device=dev) * 0.02).to(torch.bfloat16)
            w = ws[proj]
            bm = 64 * DG1_INSTS[kid][1] * DG1_INSTS[kid][2]
            packs = {0: dg1_pack(w), 1: dg1_pack_tiles(w, bm)}
            f32 = proj == "cat"
            x = torch.randn((T, k), device=dev).to(torch.bfloat16)

            def mm(x=x, w=w, f32=f32):
                return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
            ref = mm()
            g_ref = graph_of(mm)
            tr0, tt = time_pair(g_ref, g_tiny, flush, sink, reps)
            wsum = graph_of(lambda w=w: torch.sum(w.view(-1), dim=0, dtype=torch.float32, out=sink))
            _, tsum = time_pair(g_tiny, wsum, flush, sink, reps)
            mb = n * k * 2 / 1e6
            print(f"[exp] {proj} T={T} kid {kid} S {s} ({mb:.1f} MB): cublas {tr0:.2f} us, tiny-graph {tt:.2f} us, torch.sum(W) {tsum:.2f} us",
                  flush=True)
            for fl, name in flagsets:
                y = torch.full((T, n), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
                wp = packs[(fl >> 8) & 1]

                def run(kid=kid, x=x, wp=wp, y=y, s=s, n=n, fl=fl):
                    mod.dg1_run(kid, x, wp, y, err, s, 1, 1, n, fl)
                err.zero_()
                run()
                torch.cuda.synchronize()
                chk = ""
                if fl in (0, 256, 8):
                    rel = float((y.float() - ref.float()).norm() / ref.float().norm())
                    chk = f" rel vs cublas {rel:.1e} err {int(err[0])}"
                g = graph_of(run)
                tr, td = time_pair(g_ref, g, flush, sink, reps)
                net = td - tt
                print(f"[exp]   {name:18s}: {td:7.2f} us (cublas {tr:6.2f}, x{tr / td:.3f}); minus tiny graph {net:6.2f} us ="
                      f" {mb / 1e3 / max(net, 1e-3) * 1e3:5.2f} TB/s{chk}", flush=True)
                del g
            del g_ref, wsum
        return
    if arg("diag", ""):   # --diag=proj:kid:S : one-hot probes, where does each expected value land?
        proj, kid, s = arg("diag", "").split(":")
        kid, s = int(kid), int(s)
        n, k = PROJ[proj]
        T = DG1_INSTS[kid][0]
        w = torch.randn((n, k), device=dev).to(torch.bfloat16)
        wp = dg1_pack(w)
        err = torch.zeros(4, dtype=torch.int32, device=dev)
        for case in ("onehot", "random"):
            x = torch.zeros((T, k), dtype=torch.bfloat16, device=dev)
            cols = torch.arange(T, device=dev) * 37 % k
            if case == "onehot":
                x[torch.arange(T, device=dev), cols] = 1.0
            else:
                x.normal_()
            y = torch.full((T, n), float("nan"), dtype=torch.float32 if proj == "cat" else torch.bfloat16, device=dev)
            mod.dg1_run(kid, x, wp, y, err, s, 0, 0, n, 0)
            torch.cuda.synchronize()
            ref = (x.float() @ w.float().t())
            bad = ~torch.isclose(y.float(), ref, rtol=2e-2, atol=1e-2)
            print(f"[diag] {case}: err {int(err[0])} nan {int(torch.isnan(y.float()).sum())} bad {int(bad.sum())} / {bad.numel()}"
                  f" rel {float((y.float() - ref).norm() / ref.norm()):.3e}", flush=True)
            if case == "onehot" and int(bad.sum()):
                tb, nb = torch.nonzero(bad)[:12].t().tolist()
                for t_, n_ in zip(tb, nb):
                    v = y[t_, n_].float()
                    hit = torch.nonzero(ref == v)[:3].tolist() if torch.isfinite(v) else []
                    print(f"[diag]   y[{t_}, {n_}] = {float(v):.4f} want {float(ref[t_, n_]):.4f}; value found at ref {hit}", flush=True)
                print(f"[diag]   bad rows (tokens) {sorted(set(torch.nonzero(bad)[:, 0].tolist()))[:40]}", flush=True)
                print(f"[diag]   bad cols mod 64 {sorted(set((torch.nonzero(bad)[:, 1] % 64).tolist()))[:64]}", flush=True)
        return
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)
    flush.normal_()
    sink = torch.empty((), dtype=torch.float32, device=dev)
    err = torch.zeros(4, dtype=torch.int32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(1)
    rows = []
    best = {}
    for proj in projs:
        n, k = PROJ[proj]
        w = (torch.randn((n, k), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        wp = dg1_pack(w)
        f32 = proj == "cat"
        for T in ts:
            if proj == "unpooler" and T < 128:
                continue   # T <= 32 unpooler is P1M's
            x = torch.randn((T, k), generator=gen, device=dev).to(torch.bfloat16)

            def mm(x=x, w=w, f32=f32):
                return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
            ref = mm()
            ref64 = x.double() @ w.double().t()
            g_ref = graph_of(mm)
            sel = cands(proj, T, DG1_INSTS)
            if only:
                keep = {tuple(int(v) for v in p.split(":")) for p in only.split(",")}
                sel = [c for c in sel if (c[0], c[3]) in keep]
            for kid, g, m, s in sel:
                info = mod.dg1_info(kid, s)
                bm = info[3]
                clusters = n // bm
                one_wave = info[0] >= clusters
                y = torch.empty((T, n), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
                trig = 1 if one_wave else 0

                def run(kid=kid, x=x, wp=wp, y=y, s=s, trig=trig, n=n):
                    mod.dg1_run(kid, x, wp, y, err, s, 1, trig, n, 0)
                tag = f"{proj:8s} T={T:3d} BM={bm:3d} S={s} grid={clusters * s:3d} wave1={int(one_wave)}"
                try:
                    err.zero_()
                    y.fill_(float("nan"))
                    run()
                    torch.cuda.synchronize()
                    got = y.clone()
                    run()
                    torch.cuda.synchronize()
                    same = bool(torch.equal(got, y))
                    e64 = float((got.double() - ref64).norm() / ref64.norm())
                    eref = float((got.float() - ref.float()).norm() / ref.float().norm())
                    errf = int(err[0])
                    ok = errf == 0 and same and e64 < (1e-4 if f32 else 4e-3) and eref < 1e-2 and bool(torch.isfinite(got).all())
                    if not ok:
                        print(f"[dg1] {tag}: FAIL rel64 {e64:.2e} relcublas {eref:.2e} same {same} err {errf}", flush=True)
                        rows.append({"proj": proj, "T": T, "bm": bm, "S": s, "ok": False, "e64": e64, "eref": eref})
                        continue
                    g_dg = graph_of(run)
                    tr, td = time_pair(g_ref, g_dg, flush, sink, reps)
                    print(f"[dg1] {tag}: cublas {tr:7.2f} us  dg1 {td:7.2f} us  x{tr / td:.3f}   (rel64 {e64:.1e} vs cublas {eref:.1e})",
                          flush=True)
                    rows.append({"proj": proj, "T": T, "bm": bm, "S": s, "kid": kid, "ok": True, "cublas_us": tr, "dg1_us": td,
                                 "ratio": tr / td, "e64": e64, "eref": eref, "wave1": one_wave})
                    if (proj, T) not in best or tr / td > best[(proj, T)]["ratio"]:
                        best[(proj, T)] = rows[-1]
                    del g_dg
                except Exception as ex:  # noqa: BLE001
                    print(f"[dg1] {tag}: raised {ex!r}", flush=True)
                    torch.cuda.synchronize()
            del g_ref
    print("[dg1] best per (proj, T):", flush=True)
    for (proj, T), r in sorted(best.items()):
        print(f"[dg1]   {proj:8s} T={T:3d}: kid {r['kid']} BM {r['bm']} S {r['S']}: cublas {r['cublas_us']:.2f} dg1 {r['dg1_us']:.2f}"
              f" x{r['ratio']:.3f}", flush=True)
    gate = [(p, t) for p in ("qkv", "cat") for t in (32, 128)]
    passed = all((p, t) in best and best[(p, t)]["ratio"] >= 1.10 for p, t in gate)
    print(f"[dg1] KILL GATE (>= x1.10 on qkv & cat at T = 32 and 128): {'PASS' if passed else 'FAIL'}", flush=True)
    od = ROOT / "box_out" / "dg1"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"lab_{arg('tag', 'r1')}.json").write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()

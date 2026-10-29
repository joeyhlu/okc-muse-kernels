"""dn_lab.py: DN1 = dense BF16 GEMM for T = 128 / 256 (verify16 / verify32) with TOKENS AS M (wgmma m64, A = x from smem)
and WEIGHTS AS B (n256 or n128 tiles), TMA SW128 ring (NS x (16 KB x + BN*128 B W)), 1 CTA/SM, 2 consumer warpgroups +
1 producer warp, split-K S over a one-wave grid with a deterministic global-workspace reduce (each of the S CTAs of a tile
reduces 1/S of it after a per-tile arrival counter; fixed summation order p = 0..S-1). FP32 accumulation, output rounded
once (bf16 RN, or FP32 for the cat), like cuBLAS. Plus step 0(a): cuBLASLt heuristic top-32 (TN and NN = torchT) + forced
split-K / tile variants, all timed with the judge's gated cold-L2 GpuTimer (tools/judge/judge.py).
    python3 -u kernels/lab/dn_lab.py dn  [--shapes=qkv:128,...] [--out=box_out/dn1/dn.json]
    python3 -u kernels/lab/dn_lab.py lt  [--out=box_out/dn1/lt.json]
"""
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "judge"))

SHAPES = {"qkv": (2304, 8192, False), "o": (8192, 1024, False), "cat": (4608, 8192, True), "unpooler": (8192, 4096, False)}


def _wgmma(n):
    na = n // 2
    regs = ",".join(f"%{i}" for i in range(na))
    chunks = [regs[i:i + 120] for i in range(0, len(regs), 120)]
    nl, bs = "\n", "\\n"
    tmpl = ('"' + nl + '               "').join(chunks)
    outs = [f'"+f"(d[{i}])' for i in range(na)]
    outl = ("," + nl + "                 ").join(", ".join(outs[i:i + 8]) for i in range(0, na, 8))
    return (f"template <> __device__ __forceinline__ void wgmma_ss<{n}>(float* d, uint64_t da, uint64_t db) {{" + nl +
            f'  asm volatile("{{{bs}.reg .pred p;{bs}setp.ne.b32 p, 1, 0;{bs}"' + nl +
            f'               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{"' + nl +
            f'               "{tmpl}"' + nl +
            f'               "}}, %{na}, %{na + 1}, p, 1, 1, 0, 0;{bs}}}{bs}"' + nl +
            f"               : {outl}" + nl +
            f'               : "l"(da), "l"(db));' + nl + "}" + nl)


DN_CPP_SRC = r"""
#include <torch/extension.h>
int64_t dn_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor ws, torch::Tensor ctr,
               torch::Tensor err, torch::Tensor wpk, torch::Tensor stamps, int64_t S, int64_t flags);
"""

DN_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace dn {
constexpr long long WAIT_BUDGET = 40000000LL;   // ~20 ms at 1.98 GHz; sets err, never traps
constexpr int F_PDL = 1, F_TRIG = 2, F_NOMMA = 4, F_NORED = 8, F_NOX = 16, F_PACK = 32;

template <int BN, int NS>
struct Cfg {
  static constexpr int XB = 128 * 128;             // 128 tokens x 64 bf16 (SW128)
  static constexpr int WB = BN * 128;              // BN weight rows x 64 bf16
  static constexpr int SB = XB + WB;
  static constexpr int NACC = BN / 2;              // fp32 accumulators per consumer thread (m64 x BN per warpgroup)
  static constexpr int NQ = BN / 8;                // float4 groups per thread
  static constexpr int SMEM = 1024 + NS * SB + 16 * NS + 64;
  static_assert(SB % 1024 == 0, "stage alignment");
  static_assert(SMEM <= 232448, "smem");
};

struct Args {
  void* y; float4* ws; int* ctr; int* err; const uint8_t* wpk; long long* stamps;
  int T, N, nkb, S, ntiles, flags;
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
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db);
@@WGMMA@@

template <int F32>
__device__ __forceinline__ void store_frag(const Args& a, int row, int col, float4 v) {
  if (F32) {
    float* y = reinterpret_cast<float*>(a.y);
    if (row < a.T) *reinterpret_cast<float2*>(y + (size_t)row * a.N + col) = make_float2(v.x, v.y);
    if (row + 8 < a.T) *reinterpret_cast<float2*>(y + (size_t)(row + 8) * a.N + col) = make_float2(v.z, v.w);
  } else {
    __nv_bfloat16* y = reinterpret_cast<__nv_bfloat16*>(a.y);
    if (row < a.T) *reinterpret_cast<__nv_bfloat162*>(y + (size_t)row * a.N + col) = __floats2bfloat162_rn(v.x, v.y);
    if (row + 8 < a.T) *reinterpret_cast<__nv_bfloat162*>(y + (size_t)(row + 8) * a.N + col) = __floats2bfloat162_rn(v.z, v.w);
  }
}

// grid = ntiles * S * TB; blockIdx = (nt * S + s) * TB + tb (the TB token blocks of one weight chunk are adjacent)
template <int BN, int NS, int F32>
__global__ void __launch_bounds__(288, 1)
dn_kernel(const __grid_constant__ CUtensorMap tmx, const __grid_constant__ CUtensorMap tmw, const Args a) {
  using C = Cfg<BN, NS>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sbase = (sa(smem_raw) + 1023u) & ~1023u;
  const uint32_t full0 = sbase + NS * C::SB, empty0 = full0 + 8 * NS;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int TB = (int)gridDim.x / (a.ntiles * a.S);
  const int bid = (int)blockIdx.x;
  const int tb = bid % TB, s = (bid / TB) % a.S, nt = bid / (TB * a.S);
  const int kb0 = (s * a.nkb) / a.S, nk = ((s + 1) * a.nkb) / a.S - kb0;
  bool dead = false;
  if (a.stamps != nullptr && tid == 0) a.stamps[4 * bid] = gtime();
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, 2); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  }
  __syncthreads();

  if (warp == 8) {   // producer
    if ((tid & 31) == 0) {
      asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmx)) : "memory");
      asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmw)) : "memory");
      const int pre = nk < NS ? nk : NS;
      const bool pack = (a.flags & F_PACK) != 0, nox = (a.flags & F_NOX) != 0;
      const uint32_t txb = nox ? (uint32_t)C::WB : (uint32_t)C::SB;
      for (int i = 0; i < pre; ++i) {   // weights do not depend on the previous kernel: issue them before the PDL wait
        mbar_expect_tx(full0 + 8 * i, txb);
        if (pack) bulk_g2s(sbase + i * C::SB + C::XB, a.wpk + ((size_t)nt * a.nkb + kb0 + i) * C::WB, C::WB, full0 + 8 * i);
        else tma2d(sbase + i * C::SB + C::XB, &tmw, (kb0 + i) * 64, nt * BN, full0 + 8 * i);
      }
      if (a.flags & F_PDL) pdl_wait();
      if (!nox) for (int i = 0; i < pre; ++i) tma2d(sbase + i * C::SB, &tmx, (kb0 + i) * 64, tb * 128, full0 + 8 * i);
      for (int i = NS; i < nk; ++i) {
        const int slot = i % NS;
        mbar_wait(empty0 + 8 * slot, (uint32_t)(((i / NS) - 1) & 1), a.err, dead);
        const uint32_t st = sbase + slot * C::SB, bar = full0 + 8 * slot;
        mbar_expect_tx(bar, txb);
        if (!nox) tma2d(st, &tmx, (kb0 + i) * 64, tb * 128, bar);
        if (pack) bulk_g2s(st + C::XB, a.wpk + ((size_t)nt * a.nkb + kb0 + i) * C::WB, C::WB, bar);
        else tma2d(st + C::XB, &tmw, (kb0 + i) * 64, nt * BN, bar);
      }
    }
    return;
  }

  if (a.flags & F_PDL) pdl_wait();
  const int wg = warp >> 2, t = tid & 127;
  float acc[C::NACC];
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) acc[q] = 0.f;
#pragma unroll 1
  for (int i = 0; i < nk; ++i) {
    const int slot = i % NS;
    mbar_wait(full0 + 8 * slot, (uint32_t)((i / NS) & 1), a.err, dead);
    const uint32_t st = sbase + slot * C::SB;
    const uint64_t da = desc_sw128(st + wg * 8192), db = desc_sw128(st + C::XB);
#pragma unroll
    for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    if (!(a.flags & F_NOMMA))
      for (int kk = 0; kk < 4; ++kk) wgmma_ss<BN>(acc, da + 2 * kk, db + 2 * kk);
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NS));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
  if (a.flags & F_TRIG) pdl_trigger();
  if (a.stamps != nullptr && tid == 0) a.stamps[4 * bid + 1] = gtime();

  const int lane = tid & 31, w4 = (tid & 127) >> 5;
  const int row0 = tb * 128 + 64 * wg + 16 * w4 + (lane >> 2);
  const int col0 = nt * BN + 2 * (lane & 3);
  if (a.S == 1 || (a.flags & F_NORED)) {
#pragma unroll
    for (int j = 0; j < C::NQ; ++j)
      store_frag<F32>(a, row0, col0 + 8 * j, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
    return;
  }
  const int tile = tb * a.ntiles + nt;
  float4* base = a.ws + (size_t)tile * a.S * C::NQ * 256;
  float4* mine = base + (size_t)s * C::NQ * 256;
#pragma unroll
  for (int j = 0; j < C::NQ; ++j)
    __stcg(mine + j * 256 + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  named_sync(1, 256);
  if (tid == 0) {
    __threadfence();
    int* c = a.ctr + tile;
    const int arr = atomicAdd(c, 1);
    const int target = (arr / a.S + 1) * a.S;
    if (ld_acquire(c) < target) {
      const long long t0 = clock64();
      while (ld_acquire(c) < target) {
        if (clock64() - t0 > WAIT_BUDGET) { atomicExch(a.err, 2); break; }
      }
    }
    __threadfence();
    if (a.stamps != nullptr) a.stamps[4 * bid + 2] = gtime();
  }
  named_sync(1, 256);
  const int j0 = (s * C::NQ) / a.S, j1 = ((s + 1) * C::NQ) / a.S;
#pragma unroll 1
  for (int j = j0; j < j1; ++j) {
    float4 v[16];
#pragma unroll
    for (int p = 0; p < 16; ++p)
      if (p < a.S) v[p] = __ldcg(base + ((size_t)p * C::NQ + j) * 256 + tid);
    float4 r = v[0];
#pragma unroll
    for (int p = 1; p < 16; ++p)
      if (p < a.S) { r.x += v[p].x; r.y += v[p].y; r.z += v[p].z; r.w += v[p].w; }
    store_frag<F32>(a, row0, col0 + 8 * j, r);
  }
  if (a.stamps != nullptr && tid == 0) a.stamps[4 * bid + 3] = gtime();
}
}  // namespace dn

typedef CUresult (*dn_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static dn_encode_fn dn_get_encode() {
  static dn_encode_fn fn = nullptr;
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
    else fn = reinterpret_cast<dn_encode_fn>(p);
  }
  return fn;
}
static void dn_map(CUtensorMap* tm, const void* ptr, int64_t rows, int64_t cols, int box_rows) {
  dn_encode_fn enc = dn_get_encode();
  TORCH_CHECK(enc != nullptr, "dn: cuTensorMapEncodeTiled unavailable");
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)(cols * 2)};
  cuuint32_t box[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1u, 1u};
  const CUresult r = enc(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dn: tensor map encode failed ", (int)r);
}

template <int BN, int NS, int F32>
static int64_t dn_launch_t(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& y, const torch::Tensor& ws,
                           const torch::Tensor& ctr, const torch::Tensor& err, const torch::Tensor& wpk,
                           const torch::Tensor& stamps, int64_t S, int64_t flags) {
  using C = dn::Cfg<BN, NS>;
  auto kern = dn::dn_kernel<BN, NS, F32>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    attr = true;
  }
  const int64_t T = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && x.is_contiguous() && w.is_contiguous(),
              "dn: x / w bf16 contiguous");
  TORCH_CHECK(w.size(1) == K && K % 64 == 0 && N % BN == 0 && T >= 1 && T <= 256, "dn: shape");
  TORCH_CHECK(y.size(0) == T && y.size(1) == N && y.is_contiguous() &&
              y.scalar_type() == (F32 ? at::kFloat : at::kBFloat16), "dn: y");
  const int TB = (int)((T + 127) / 128), ntiles = (int)(N / BN);
  const int64_t grid = (int64_t)ntiles * S * TB;
  int dev = 0, nsm = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  TORCH_CHECK(S >= 1 && S <= 16 && S <= K / 64 && grid <= nsm, "dn: grid ", grid, " > SMs ", nsm);
  TORCH_CHECK(S == 1 || (ws.numel() * 4 >= grid * C::NQ * 256 * 16 && ctr.numel() >= (int64_t)TB * ntiles), "dn: workspace");
  CUtensorMap tmx, tmw;
  dn_map(&tmx, x.data_ptr(), T, K, 128);
  dn_map(&tmw, w.data_ptr(), N, K, BN);
  dn::Args a;
  a.y = y.data_ptr(); a.ws = reinterpret_cast<float4*>(ws.data_ptr()); a.ctr = ctr.data_ptr<int>(); a.err = err.data_ptr<int>();
  a.wpk = reinterpret_cast<const uint8_t*>(wpk.data_ptr());
  a.stamps = stamps.numel() >= grid * 4 ? reinterpret_cast<long long*>(stamps.data_ptr()) : nullptr;
  TORCH_CHECK(!(flags & dn::F_PACK) || wpk.numel() * wpk.element_size() >= N * K * 2, "dn: wpk");
  a.T = (int)T; a.N = (int)N; a.nkb = (int)(K / 64); a.S = (int)S; a.ntiles = ntiles; a.flags = (int)flags;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3(288);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  if (flags & dn::F_PDL) {
    at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at_;
    cfg.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tmx, tmw, a));
  return grid;
}

int64_t dn_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor ws, torch::Tensor ctr,
               torch::Tensor err, torch::Tensor wpk, torch::Tensor stamps, int64_t S, int64_t flags) {
  switch (kid) {
    case 0: return dn_launch_t<256, 4, 0>(x, w, y, ws, ctr, err, wpk, stamps, S, flags);
    case 1: return dn_launch_t<256, 4, 1>(x, w, y, ws, ctr, err, wpk, stamps, S, flags);
    case 2: return dn_launch_t<128, 6, 0>(x, w, y, ws, ctr, err, wpk, stamps, S, flags);
    case 3: return dn_launch_t<128, 6, 1>(x, w, y, ws, ctr, err, wpk, stamps, S, flags);
  }
  TORCH_CHECK(false, "dn: kid");
  return 0;
}
""".replace("@@WGMMA@@", _wgmma(256) + _wgmma(128))

KID = {(256, False): 0, (256, True): 1, (128, False): 2, (128, True): 3}


def build_dn(arch_flags=("-gencode=arch=compute_90a,code=sm_90a",), verbose=False):
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name="dn_lab_v1", cpp_sources=DN_CPP_SRC, cuda_sources=DN_CUDA_SRC, functions=["dn_run"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", *arch_flags] +
                      (["-Xptxas=-v"] if verbose else []), verbose=verbose)
    print(f"[dn] built in {time.time() - t0:.1f}s", flush=True)
    return mod


def floor_us(N, K, T):
    return max(N * K * 2 / 4.49e6 + 1.5, 2.0 * T * N * K / 989e6)


def dn_cands(N, K, T):
    TB = (T + 127) // 128
    out = []
    for bn in (256, 128):
        nt = N // bn
        if N % bn:
            continue
        smax = min(16, K // 64, 132 // (nt * TB))
        ss = sorted({s for s in (smax, smax - 1, (smax + 1) // 2) if s >= 1})
        for s in ss:
            out.append((bn, s))
    return out


class Bench:
    def __init__(self):
        import judge as J
        self.timer = J.GpuTimer()

    def graph(self, fn):
        s = self.timer.stream
        with torch.cuda.stream(s):
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.synchronize()
        return g

    def race(self, graphs, rounds=8, reps=10):
        ts = {k: [] for k in graphs}
        keys = list(graphs)
        for _ in range(rounds):
            random.shuffle(keys)
            for k in keys:
                for _ in range(reps):
                    ts[k].append(self.timer.time_replay(graphs[k]))
        return {k: statistics.median(v) for k, v in ts.items()}


def run_dn(args):
    global NOST
    mod = build_dn()
    dev = torch.device("cuda")
    NOST = torch.empty((0,), dtype=torch.int64, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    bench = Bench()
    ws = torch.zeros((132 * 32 * 256 * 4,), dtype=torch.float32, device=dev)
    ctr = torch.zeros((64 * 1024,), dtype=torch.int32, device=dev)
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    shapes = [s.split(":") for s in args.get("shapes", "qkv:128,cat:128,unpooler:128,o:128,qkv:256,cat:256,unpooler:256,o:256").split(",")]
    res = []
    coff = 0
    for proj, Ts in shapes:
        T = int(Ts)
        N, K, f32 = SHAPES[proj]
        w = (torch.randn((N, K), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        wt = w.t().contiguous()
        x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
        od = torch.float32 if f32 else torch.bfloat16
        ref64 = x.double() @ w.double().t()

        def today(x=x, w=w):
            return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())

        def tt(x=x, wt=wt):
            return torch.mm(x, wt, out_dtype=torch.float32) if f32 else torch.mm(x, wt)

        yt = today()
        e_today = float((yt.double() - ref64).norm() / ref64.norm())
        graphs = {"today": bench.graph(today), "torchT": bench.graph(tt)}
        info = {}
        for bn, S in dn_cands(N, K, T):
            name = f"dn{bn}s{S}"
            for pdl in (0,):
                c = ctr[coff:coff + 256]
                coff += 256
                outs = []
                ok = True
                for tsub in sorted({T, T - 5, max(1, T // 2 + 3)}, reverse=True):
                    xs = x[:tsub].contiguous()
                    for _ in range(2):
                        y = torch.full((tsub, N), float("nan"), dtype=od, device=dev)
                        err.zero_()
                        try:
                            mod.dn_run(KID[(bn, f32)], xs, w, y, ws, c, err, w, NOST, S, pdl)
                            torch.cuda.synchronize()
                        except Exception as ex:  # noqa: BLE001
                            print(f"[dn] {proj} T={T} {name}: launch failed {ex}", flush=True)
                            ok = False
                            break
                        outs.append(y)
                    if not ok:
                        break
                    r64 = ref64[:tsub]
                    fin = bool(torch.isfinite(outs[-1]).all())
                    e64 = float((outs[-1].double() - r64).norm() / r64.norm()) if fin else float("inf")
                    rep = bool(torch.equal(outs[-1], outs[-2]))
                    ok = ok and fin and rep and int(err.sum()) == 0 and e64 < max(4 * e_today, 1e-3)
                    if tsub == T:
                        info[name] = {"e64": e64, "rep": rep, "err": int(err.sum())}
                    outs = []
                if ok:
                    yy = torch.empty((T, N), dtype=od, device=dev)

                    def f(xx=x, yy=yy, bn=bn, S=S, c=c):
                        mod.dn_run(KID[(bn, f32)], xx, w, yy, ws, c, err, w, NOST, S, 0)
                    graphs[name] = bench.graph(f)
                    info[name]["ok"] = True
                else:
                    print(f"[dn] {proj} T={T} {name}: FAILED {info.get(name)}", flush=True)
        med = bench.race(graphs)
        fl = floor_us(N, K, T)
        best = min(med, key=med.get)
        line = " ".join(f"{k}:{v:.2f}" for k, v in sorted(med.items(), key=lambda kv: kv[1]))
        print(f"[dn] {proj} T={T} floor {fl:.2f} | today {med['today']:.2f} (x{med['today'] / fl:.2f} floor) | best {best} "
              f"{med[best]:.2f} (x{med['today'] / med[best]:.3f} vs today) | {line} | e_today {e_today:.1e} "
              f"{ {k: round(v['e64'], 5) for k, v in info.items()} }", flush=True)
        res.append({"proj": proj, "T": T, "floor": fl, "med": med, "info": info, "e_today": e_today})
        del graphs
        torch.cuda.synchronize()
    out = Path(args.get("out", "box_out/dn1/dn.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def pack_w(w, bn):
    """Lossless SW128 tile repack: [nt][kb][bn rows x 128 B], 16-B chunk c of row r holds logical chunk c ^ (r & 7)."""
    N, K = w.shape
    nt, nkb = N // bn, K // 64
    t = w.contiguous().view(nt, bn, nkb, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bn, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7)).view(1, 1, bn, 8, 1).expand(nt, nkb, bn, 8, 8)
    return torch.gather(t, 3, idx).contiguous().view(-1)


F = {"pdl": 1, "trig": 2, "nomma": 4, "nored": 8, "nox": 16, "pack": 32}


def run_diag(args):
    global NOST
    mod = build_dn()
    dev = torch.device("cuda")
    NOST = torch.empty((0,), dtype=torch.int64, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    bench = Bench()
    ws = torch.zeros((132 * 32 * 256 * 4,), dtype=torch.float32, device=dev)
    ctr = torch.zeros((64 * 1024,), dtype=torch.int32, device=dev)
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    stamps = torch.zeros((132 * 4,), dtype=torch.int64, device=dev)
    base = {"qkv:128": [(256, 14), (128, 7)], "unpooler:128": [(256, 4), (128, 1)], "o:128": [(256, 4), (128, 1)],
            "cat:128": [(256, 7)]}
    arms = [a.split("+") if a else [] for a in args.get("arms", ",pack,pack+nomma,pack+nored,pack+nox,pack+nox+nomma+nored,nomma,nored").split(",")]
    coff = 0
    res = []
    for key in args.get("shapes", "qkv:128,unpooler:128,o:128,cat:128").split(","):
        proj, T = key.split(":")[0], int(key.split(":")[1])
        N, K, f32 = SHAPES[proj]
        od = torch.float32 if f32 else torch.bfloat16
        w = (torch.randn((N, K), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
        yt = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        graphs = {"today": bench.graph(lambda: torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t()))}
        runs = {}
        for bn, S in base[key]:
            wpk = pack_w(w, bn)
            for arm in arms:
                fl = sum(F[a] for a in arm)
                name = f"dn{bn}s{S}" + ("+" + "+".join(arm) if arm else "")
                c = ctr[coff:coff + 256]
                coff += 256
                y = torch.full((T, N), float("nan"), dtype=od, device=dev)
                err.zero_()
                mod.dn_run(KID[(bn, f32)], x, w, y, ws, c, err, wpk, NOST, S, fl)
                torch.cuda.synchronize()
                e = float((y.float() - yt.float()).norm() / yt.float().norm())
                if fl & (F["nomma"] | F["nored"] | F["nox"]) == 0 and not (e < 1e-2 and int(err.sum()) == 0):
                    print(f"[diag] {key} {name}: WRONG rel {e:.2e} err {int(err.sum())}", flush=True)
                    continue
                yy = torch.empty((T, N), dtype=od, device=dev)
                graphs[name] = bench.graph(lambda bn=bn, S=S, fl=fl, c=c, wpk=wpk, yy=yy:
                                           mod.dn_run(KID[(bn, f32)], x, w, yy, ws, c, err, wpk, NOST, S, fl))
                runs[name] = (bn, S, fl, c, wpk)
        med = bench.race(graphs, rounds=6, reps=10)
        print(f"[diag] {key}: today {med['today']:.2f} floor {floor_us(N, K, T):.2f}", flush=True)
        for name, (bn, S, fl, c, wpk) in runs.items():
            st = []
            for _ in range(5):
                yy = torch.empty((T, N), dtype=od, device=dev)
                stamps.zero_()
                bench.timer.flush()
                bench.timer.gate()
                mod.dn_run(KID[(bn, f32)], x, w, yy, ws, c, err, wpk, stamps, S, fl)
                torch.cuda.synchronize()
                g = (N // bn) * S * ((T + 127) // 128)
                v = stamps[:4 * g].view(g, 4).double()
                t0 = float(v[:, 0].min())
                q = lambda col: v[:, col] - t0  # noqa: E731
                st.append([float(q(0).max()), float(q(1).median()), float(q(1).max()),
                           float(q(2).max()) if S > 1 and not fl & F["nored"] else 0.0, float(q(3).max())])
            m = [statistics.median(z[i] for z in st) / 1e3 for i in range(5)]
            print(f"[diag]   {name:34s} {med[name]:6.2f} us | entry spread {m[0]:.2f} loop med {m[1]:.2f} max {m[2]:.2f} "
                  f"barrier {m[3]:.2f} end {m[4]:.2f}", flush=True)
        res.append({"key": key, "med": med})
        del graphs
    out = Path(args.get("out", "box_out/dn2/diag.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


LT_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>
static cublasLtHandle_t g_h = nullptr;
struct LtP { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; std::vector<cublasLtMatmulAlgo_t> al; size_t ws; };
static std::vector<LtP> g_p;
int64_t lt_plan(int64_t T, int64_t N, int64_t K, int64_t nn, int64_t f32, int64_t nheur, int64_t ws) {
  if (!g_h) TORCH_CHECK(cublasLtCreate(&g_h) == CUBLAS_STATUS_SUCCESS, "lt create");
  LtP p; p.ws = (size_t)ws;
  const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = nn ? CUBLAS_OP_N : CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  if (nn) cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, N, K, N); else cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, K, N, K);
  cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, K, T, K);
  cublasLtMatrixLayoutCreate(&p.lc, ct, N, T, N);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  uint64_t w64 = (uint64_t)ws;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &w64, sizeof(w64));
  std::vector<cublasLtMatmulHeuristicResult_t> res((size_t)nheur);
  int n = 0;
  cublasLtMatmulAlgoGetHeuristic(g_h, p.op, p.la, p.lb, p.lc, p.lc, pref, (int)nheur, res.data(), &n);
  cublasLtMatmulPreferenceDestroy(pref);
  for (int i = 0; i < n; ++i) if (res[i].state == CUBLAS_STATUS_SUCCESS) p.al.push_back(res[i].algo);
  g_p.push_back(p);
  return (int64_t)g_p.size() - 1;
}
int64_t lt_count(int64_t pid) { return (int64_t)g_p[(size_t)pid].al.size(); }
static uint64_t cfg_get(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t t) {
  uint64_t v = 0; size_t sz = 0;
  if (cublasLtMatmulAlgoConfigGetAttribute(&a, t, nullptr, 0, &sz) != CUBLAS_STATUS_SUCCESS || sz == 0 || sz > 8) return 0xFFFF;
  if (cublasLtMatmulAlgoConfigGetAttribute(&a, t, &v, sz, &sz) != CUBLAS_STATUS_SUCCESS) return 0xFFFF;
  return v;
}
std::vector<int64_t> lt_desc(int64_t pid, int64_t ai) {
  const cublasLtMatmulAlgo_t& a = g_p[(size_t)pid].al[(size_t)ai];
  return {(int64_t)(int32_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_ID), (int64_t)(uint32_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_TILE_ID),
          (int64_t)(uint32_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_STAGES_ID), (int64_t)(uint32_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
          (int64_t)(uint32_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME), (int64_t)(uint32_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
          (int64_t)(uint32_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION), (int64_t)(uint16_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID),
          (int64_t)(uint16_t)cfg_get(a, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)};
}
static bool lt_check(LtP& p, cublasLtMatmulAlgo_t& a) {
  cublasLtMatmulHeuristicResult_t r;
  if (cublasLtMatmulAlgoCheck(g_h, p.op, p.la, p.lb, p.lc, p.lc, &a, &r) != CUBLAS_STATUS_SUCCESS) return false;
  return r.state == CUBLAS_STATUS_SUCCESS && r.workspaceSize <= p.ws;
}
// variants of heuristic algo ai: mode 0 = split-K in {1,2,3,4,6,8} (reduction COMPUTE_TYPE when > 1);
// mode 1 = every tile id of its algo id x {its stages, 0}, split-K 1.
int64_t lt_variants(int64_t pid, int64_t ai, int64_t mode) {
  LtP& p = g_p[(size_t)pid];
  const cublasLtMatmulAlgo_t base = p.al[(size_t)ai];
  int64_t added = 0;
  if (mode == 0) {
    const uint32_t ks[6] = {1, 2, 3, 4, 6, 8};
    for (uint32_t k : ks) {
      cublasLtMatmulAlgo_t a = base;
      uint32_t red = k > 1 ? (uint32_t)CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE : (uint32_t)CUBLASLT_REDUCTION_SCHEME_NONE;
      if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &k, sizeof(k)) != CUBLAS_STATUS_SUCCESS) continue;
      if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red)) != CUBLAS_STATUS_SUCCESS) continue;
      if (lt_check(p, a)) { p.al.push_back(a); ++added; }
    }
    return added;
  }
  const int32_t id = (int32_t)cfg_get(base, CUBLASLT_ALGO_CONFIG_ID);
  const uint32_t st0 = (uint32_t)cfg_get(base, CUBLASLT_ALGO_CONFIG_STAGES_ID);
  cublasLtMatmulAlgo_t a0;
  const cudaDataType_t ct = CUDA_R_16BF;
  (void)ct;
  size_t sz = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz) != CUBLAS_STATUS_SUCCESS || sz < 4) return 0;
  std::vector<uint32_t> tiles(sz / 4);
  cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), tiles.size() * 4, &sz);
  (void)id; (void)a0;
  for (uint32_t tile : tiles) {
    for (int si = 0; si < 2; ++si) {
      cublasLtMatmulAlgo_t a = base;
      uint32_t st = si ? 0u : st0, k = 1, red = (uint32_t)CUBLASLT_REDUCTION_SCHEME_NONE;
      if (cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile)) != CUBLAS_STATUS_SUCCESS) continue;
      cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &st, sizeof(st));
      cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &k, sizeof(k));
      cublasLtMatmulAlgoConfigSetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red));
      if (lt_check(p, a)) { p.al.push_back(a); ++added; }
    }
  }
  return added;
}
void lt_run(int64_t pid, int64_t ai, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor ws) {
  LtP& p = g_p[(size_t)pid];
  const float alpha = 1.f, beta = 0.f;
  TORCH_CHECK(cublasLtMatmul(g_h, p.op, &alpha, w.data_ptr(), p.la, x.data_ptr(), p.lb, &beta, y.data_ptr(), p.lc, y.data_ptr(),
                             p.lc, &p.al[(size_t)ai], ws.data_ptr(), p.ws, at::cuda::getCurrentCUDAStream()) == CUBLAS_STATUS_SUCCESS,
              "lt run");
}
"""


def run_lt(args):
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    lt = load_inline(name="dn_lt_v1", cpp_sources=LT_CPP, functions=["lt_plan", "lt_count", "lt_desc", "lt_variants", "lt_run"],
                     extra_ldflags=["-L/usr/local/cuda/lib64", "-lcublasLt"], extra_cflags=["-O2"], with_cuda=True)
    print(f"[lt] built in {time.time() - t0:.1f}s", flush=True)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev)
    gen.manual_seed(99)
    bench = Bench()
    WS = 64 << 20
    ws = torch.empty((WS,), dtype=torch.uint8, device=dev)
    res, pins = [], {}
    for proj, T in (("qkv", 128), ("cat", 128), ("unpooler", 128), ("o", 128), ("qkv", 256), ("cat", 256), ("unpooler", 256), ("o", 256)):
        N, K, f32 = SHAPES[proj]
        od = torch.float32 if f32 else torch.bfloat16
        w = (torch.randn((N, K), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        wt = w.t().contiguous()
        x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
        yt = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        cands = {}
        for nn in (0, 1):
            pid = lt.lt_plan(T, N, K, nn, 1 if f32 else 0, 32, WS)
            nh = lt.lt_count(pid)
            for ai in range(min(nh, 4)):
                lt.lt_variants(pid, ai, 0)
            for ai in range(min(nh, 2)):
                lt.lt_variants(pid, ai, 1)
            for ai in range(lt.lt_count(pid)):
                d = tuple(lt.lt_desc(pid, ai))
                key = ("NN" if nn else "TN",) + d
                if key not in cands:
                    cands[key] = (pid, ai, nn, ai < nh)
        stage1 = {}
        for key, (pid, ai, nn, heur) in cands.items():
            y = torch.full((T, N), float("nan"), dtype=od, device=dev)
            try:
                lt.lt_run(pid, ai, x, wt if nn else w, y, ws)
                torch.cuda.synchronize()
            except Exception as ex:  # noqa: BLE001
                print(f"[lt] {proj} T={T} {key}: run failed {str(ex)[:80]}", flush=True)
                continue
            if not bool(torch.isfinite(y).all()) or float((y.float() - yt.float()).norm() / yt.float().norm()) > 1e-2:
                continue
            ts = []
            for _ in range(3):
                bench.timer.flush()
                bench.timer.gate()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                lt.lt_run(pid, ai, x, wt if nn else w, y, ws)
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b) * 1e3)
            stage1[key] = min(ts)
        top = sorted(stage1, key=stage1.get)[:8]
        graphs = {"today": bench.graph(lambda: torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t()))}
        for key in top:
            pid, ai, nn, heur = cands[key]
            y = torch.empty((T, N), dtype=od, device=dev)
            graphs[key] = bench.graph(lambda pid=pid, ai=ai, nn=nn, y=y: lt.lt_run(pid, ai, x, wt if nn else w, y, ws))
        med = bench.race(graphs, rounds=int(args.get('rounds', 6)), reps=10)
        fl = floor_us(N, K, T)
        best = min(med, key=med.get)
        print(f"[lt] {proj} T={T}: {len(cands)} cfgs, {len(stage1)} valid; floor {fl:.2f} today {med['today']:.2f} best "
              f"{best} {med[best]:.2f} (x{med['today'] / med[best]:.3f})", flush=True)
        for key in sorted(med, key=med.get)[:5]:
            print(f"[lt]    {med[key]:.2f}  {key}", flush=True)
        res.append({"proj": proj, "T": T, "floor": fl, "med": {str(k): v for k, v in med.items()}, "n": len(cands)})
        if best != "today" and med["today"] / med[best] >= 1.03 and best[0] == "TN":
            pins[f"{proj},{T}"] = list(best[1:])
        del graphs
    out = Path(args.get("out", "box_out/dn1/lt.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    (out.parent / "pins.json").write_text(json.dumps(pins, indent=1))
    print(f"[lt] pins {pins}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dn"
    kv = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    if mode == "dn":
        run_dn(kv)
    elif mode == "diag":
        run_diag(kv)
    elif mode == "lt":
        run_lt(kv)
    elif mode == "compile":
        build_dn(verbose=True)

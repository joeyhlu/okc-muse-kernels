"""dm_lab.py: DM1 = dense BF16 projections at m = 128 / 256 (verify16 / verify32), standalone vs torch.mm (cuBLAS).

Diagnosis (notes/study/dense_m128_fresh.md): at m = 128 the four GEMMs are near-balanced (compute 8.6 GF vs 67 MB per big
matrix), so a kernel must stream at the HBM line AND run the tensor pipe at ~60-70 % at the same time. cuBLAS's m = 128 picks
(nvjet 128x64 / 96x64, n64 token tiles, SS wgmma) push A + B through shared memory twice (TMA write + wgmma read, ~3.5 MB per
SM on the unpooler = ~14 us of smem port at 100 %), and DG1's bulk-ring weight stream alone reached only 0.24 us/MB (4.17 TB/s,
static 1-CTA/SM partition) -> both sit at ~3.3 TB/s. The unpooler's BF16 stream line (2.2 us + 67 MB / 4.54 TB/s = 17.0 us) is
already within 1.2x of cuBLAS (20.4), so x1.15 on all three big matrices needs fewer bytes: P12 (lossless 12-bit, gv1's pack).

DM1 (this file):
  * swap-AB: y^T[N, T] = W[N, K] . x^T, wgmma m64nTk16 RS: A = P12 weights decoded straight into registers (gv1 / PT decode,
    plain mma order), B = x tile [T x 64] in smem (TMA SW128, tensor map encoded per call), FP32 accumulate.
  * CTA = 2 consumer warpgroups (128 weight rows, ALL T tokens -> x is staged once per 128 rows, not per 64 as in PT); thread 0
    also issues the x copies (TMA ring of STX stages, refilled one unit after release; a 9th warp caps registers at 168).
    Weights never touch smem: each consumer warp streams its own 16-row x 64-k
    units (1536 B) with ST units in flight in registers (gv1's engine: grid-interleaved stream order, ld.global.nc.no_allocate),
    so weight depth costs no smem and no producer issue slots: 8 warps x ST x 1.5 KB in flight per SM.
  * Two wgmma groups in flight (wait<1> at the top of unit u frees unit u-2's A registers and x stage).
  * Split-K over S CTAs per 128-row tile, S chosen for one wave (pooler 4, unpooler 2, qkv 7, o 2); partials as float4 to a
    global FP32 workspace (L2), per-tile arrival counter (bounded spin, R9), every CTA of the tile sums 1/S of the tile in split
    order (deterministic) and stores BF16; the last finisher resets the counters (graph-replay safe).
  Lab flags: 1 = exit at entry (floor), 2 = no wgmma, 4 = no x (no copies, consumers do not wait), 8 = no reduce (partials
    stored, no spin / sum), 16 = no decode (wgmma on raw words).

    python3 kernels/lab/dm_lab.py [--ms=128,256] [--projs=qkv,o,pooler,unpooler] [--reps=21] [--rounds=8] [--build-only=1]
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
sys.path.insert(0, str(ROOT / "kernels" / "lab"))

# (T, ST, STX) instantiations; kid = index
DM_INSTS = [(128, 4, 8), (128, 6, 8), (128, 2, 8), (256, 2, 6)]

DM_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t dm_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor esc, torch::Tensor y, torch::Tensor ws,
               torch::Tensor cnt, torch::Tensor err, int64_t S, int64_t ntile, int64_t kumax, int64_t tb, int64_t flags);
std::vector<int64_t> dm_info(int64_t kid);
"""

DM_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace dm {

constexpr long long WAIT_BUDGET = 2000000000LL;   // ~1 s: a hang becomes err[0] = 1, never a trap (R9)

struct Params {
  const uint8_t* w;      // P12 stream: unit (step j, global warp gw) at (j * G + gw) * 1536 B, G = S * ntile * 8
  const uint32_t* esc;   // [G * 32] escape XOR entries (step << 20 | ...), sentinel 0xFFFFFFFF
  void* y;               // [T, ldy] bf16
  float* ws;             // [S][ntile][8 warps][NA / 4][32 lanes] float4 partials
  int* cnt;              // [2 * ntile]: arrivals, finishes
  int* err;
  int ntile, S, T, ldy, nk, flags;
  uint32_t tb;
};

template <int T, int ST, int STX>
struct Cfg {
  static constexpr int XB = T * 128;         // x stage: T rows x 64 bf16, SW128
  static constexpr int NA = T / 2;           // accumulator floats per thread (m64 x T per warpgroup)
  static constexpr int THREADS = 256;        // 8 consumer warps; thread 0 also issues the x copies (9 warps cap regs at 168)
  static constexpr int SMEM = 1024 + STX * XB + 16 * STX;
  static_assert(ST % 2 == 0, "ST even: unit parity = slot parity");
  static_assert(SMEM <= 227 * 1024, "smem");
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
__device__ __forceinline__ void tma_x(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ uint4 ldw(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t lde(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ int ld_acq(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// gv1 P12 unit: 3 planes x 512 B (two low-byte planes + code plane), lane owns 16 B of each
__device__ __forceinline__ void ldunit(const uint8_t* up, int lane, uint4 (&r)[3]) {
  r[0] = ldw(up + lane * 16); r[1] = ldw(up + 512 + lane * 16); r[2] = ldw(up + 1024 + lane * 16);
}
// gv1 decode<0>: one 64-k unit (4 k-steps) -> 16 A-fragment registers a[4q + i] (exact, no rounding)
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
// gv1 escape entry: step << 20 | lane << 15 | reg << 11 | half << 10 | xor7
__device__ __forceinline__ void patch(uint32_t (&a)[16], uint32_t e, int lane) {
  if (lane == (int)((e >> 15) & 31u)) {
    const int r = (int)((e >> 11) & 15u);
    const uint32_t v = (e & 127u) << (8 + (((e >> 10) & 1u) << 4));
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (i == r) a[i] ^= v;
  }
}
// B (x): 128B-swizzle K-major (TMA SWIZZLE_128B), SBO 1024 B, LBO unused (DG1's desc_sw128)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024u >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db);
@WGSPECS@
template <int NR> __device__ __forceinline__ void fence_regs(float* d) {
#pragma unroll
  for (int i = 0; i < NR; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

__global__ void null_kernel(int* err) { if (threadIdx.x == 1023) err[1] = 0; }

template <int T, int ST, int STX>
__global__ void __launch_bounds__(256, 1) dm_kernel(const __grid_constant__ CUtensorMap tmx, const Params P) {
  using C = Cfg<T, ST, STX>;
  constexpr int XB = C::XB, NA = C::NA;
  extern __shared__ uint8_t dsm[];
  const uint32_t raw = sa(dsm);
  const uint32_t base = (raw + 1023u) & ~1023u;
  const uint32_t bfull = base + STX * XB, bempty = bfull + 8 * STX;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int split = blockIdx.x, tile = blockIdx.y, S = P.S;
  const int kb = (P.nk * split) / S, nloc = (P.nk * (split + 1)) / S - kb;
  const int flags = P.flags;
  if (flags & 1) return;
  if (tid == 0) {
    for (int s = 0; s < STX; ++s) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  bool dead = false;
  float acc[NA];
  const bool issuer = tid == 0 && !(flags & 4);
  if (issuer) {   // x prologue: the first STX stages
    pdl_wait();
    const int pre = nloc < STX ? nloc : STX;
    for (int i = 0; i < pre; ++i) {
      mbar_expect_tx(bfull + 8 * i, (uint32_t)XB);
      tma_x(base + i * XB, &tmx, (kb + i) * 64, 0, bfull + 8 * i);
    }
  }
  {
    // ---------------- consumers: warp w owns weight rows [16 w, 16 w + 16) of the tile; warpgroup g = w >> 2
    const int G = S * P.ntile * 8;
    const int gw = (split * P.ntile + tile) * 8 + warp;
    const size_t wstep = (size_t)G * 1536;
    const uint8_t* wp = P.w + (size_t)gw * 1536;
    uint4 R[ST][3];
#pragma unroll
    for (int i = 0; i < ST; ++i)
      if (i < nloc) ldunit(wp + (size_t)i * wstep, lane, R[i]);
    const uint32_t em = lde(P.esc + (size_t)gw * 32 + lane);
    int ecur = 0;
    uint32_t enx = __shfl_sync(0xffffffffu, em, 0);
#pragma unroll
    for (int i = 0; i < NA; ++i) acc[i] = 0.f;
    uint32_t a[2][16];
    const bool nox = (flags & 4) != 0, nomma = (flags & 2) != 0, nodec = (flags & 16) != 0;
#pragma unroll 1
    for (int ub = 0; ub < nloc; ub += ST) {
#pragma unroll
      for (int d = 0; d < ST; ++d) {
        const int u = ub + d;
        if (u < nloc) {
          const int ab = d & 1;
          wg_wait<1>();   // unit u - 2's wgmma group is done: a[ab] and x stage (u - 2) are free
          if (u >= 2 && lane == 0 && !nox) mbar_arrive(bempty + 8 * ((u - 2) % STX));
          if (issuer && u >= 3 && u - 3 + STX < nloc) {   // refill the stage every warp released one unit ago
            const int i = u - 3 + STX, sr = (u - 3) % STX;
            mbar_wait(bempty + 8 * sr, (uint32_t)(((u - 3) / STX) & 1), P.err, dead);
            mbar_expect_tx(bfull + 8 * sr, (uint32_t)XB);
            tma_x(base + sr * XB, &tmx, (kb + i) * 64, 0, bfull + 8 * sr);
          }
          if (nodec) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              a[ab][4 * q] = R[d][0].x ^ q; a[ab][4 * q + 1] = R[d][1].y; a[ab][4 * q + 2] = R[d][2].z; a[ab][4 * q + 3] = R[d][0].w;
            }
          } else {
            decode(R[d], P.tb, a[ab]);
          }
          if (u + ST < nloc) ldunit(wp + (size_t)(u + ST) * wstep, lane, R[d]);
          while ((int)(enx >> 20) == u) {
            patch(a[ab], enx, lane);
            ++ecur;
            enx = ecur < 32 ? __shfl_sync(0xffffffffu, em, ecur & 31) : 0xFFFFFFFFu;
          }
          const int s = u % STX;
          if (!nox) mbar_wait(bfull + 8 * s, (uint32_t)((u / STX) & 1), P.err, dead);
          const uint32_t xs = base + s * XB;
          wg_fence();
          if (!nomma) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
              wgmma_rs<T>(acc, a[ab][4 * q], a[ab][4 * q + 1], a[ab][4 * q + 2], a[ab][4 * q + 3], desc_sw128(xs + q * 32));
          }
          wg_commit();
        }
      }
    }
    wg_wait<0>();
    fence_regs<NA>(acc);
  }
  // ---------------- epilogue
  const int ntile = P.ntile;
  constexpr int Q = 8 * (NA / 4) * 32;   // float4 slots per tile partial
  if (S == 1) {
    {
      const int g = lane >> 2, t = lane & 3;
      const int row = tile * 128 + 16 * warp + g;
      __nv_bfloat16* y = reinterpret_cast<__nv_bfloat16*>(P.y);
#pragma unroll
      for (int j = 0; j < NA / 4; ++j) {
        const int tok = 8 * j + 2 * t;
        if (tok < P.T) { y[(size_t)tok * P.ldy + row] = __float2bfloat16_rn(acc[4 * j]); y[(size_t)tok * P.ldy + row + 8] = __float2bfloat16_rn(acc[4 * j + 2]); }
        if (tok + 1 < P.T) { y[(size_t)(tok + 1) * P.ldy + row] = __float2bfloat16_rn(acc[4 * j + 1]); y[(size_t)(tok + 1) * P.ldy + row + 8] = __float2bfloat16_rn(acc[4 * j + 3]); }
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(P.ws) + (size_t)(split * ntile + tile) * Q + (size_t)warp * (NA / 4) * 32 + lane;
#pragma unroll
    for (int j = 0; j < NA / 4; ++j) wsp[j * 32] = make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
  }
  if (flags & 8) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    atomicAdd(P.cnt + tile, 1);
    const long long t0 = clock64();
    while (ld_acq(P.cnt + tile) < S) {
      if (clock64() - t0 > WAIT_BUDGET) { atomicExch(P.err, 2); break; }
      __nanosleep(32);
    }
    __threadfence();
  }
  __syncthreads();
  const int q0 = (Q * split) / S, q1 = (Q * (split + 1)) / S;
  const float4* wsr = reinterpret_cast<const float4*>(P.ws) + (size_t)tile * Q;
  __nv_bfloat16* y = reinterpret_cast<__nv_bfloat16*>(P.y);
  for (int q = q0 + tid; q < q1; q += 256) {
    float4 v = __ldcg(wsr + q);
    for (int sp = 1; sp < S; ++sp) {   // split order: deterministic
      const float4 p = __ldcg(wsr + (size_t)sp * ntile * Q + q);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
    }
    const int ln = q & 31, j = (q >> 5) % (NA / 4), w = q / (32 * (NA / 4));
    const int row = tile * 128 + 16 * w + (ln >> 2), tok = 8 * j + 2 * (ln & 3);
    if (tok < P.T) { y[(size_t)tok * P.ldy + row] = __float2bfloat16_rn(v.x); y[(size_t)tok * P.ldy + row + 8] = __float2bfloat16_rn(v.z); }
    if (tok + 1 < P.T) { y[(size_t)(tok + 1) * P.ldy + row] = __float2bfloat16_rn(v.y); y[(size_t)(tok + 1) * P.ldy + row + 8] = __float2bfloat16_rn(v.w); }
  }
  __syncthreads();
  if (tid == 0) {
    const int old = atomicAdd(P.cnt + ntile + tile, 1);
    if (old == S - 1) { atomicExch(P.cnt + tile, 0); atomicExch(P.cnt + ntile + tile, 0); }
  }
}

}  // namespace dm

typedef CUresult (*dm_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static dm_encode_fn dm_get_encode() {
  static dm_encode_fn fn = nullptr;
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
    else fn = reinterpret_cast<dm_encode_fn>(p);
  }
  return fn;
}

template <int T, int ST, int STX>
static int64_t dm_launch_t(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& esc, const torch::Tensor& y,
                           const torch::Tensor& ws, const torch::Tensor& cnt, const torch::Tensor& err, int64_t S, int64_t ntile,
                           int64_t tb, int64_t flags) {
  using C = dm::Cfg<T, ST, STX>;
  auto kern = dm::dm_kernel<T, ST, STX>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    attr = true;
  }
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "dm: x bf16 [T, K]");
  const int64_t Tact = x.size(0), K = x.size(1), N = ntile * 128;
  TORCH_CHECK(Tact >= 1 && Tact <= T && K % 64 == 0, "dm: shape");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.stride(1) == 1 && y.size(0) >= Tact
              && y.size(1) >= N, "dm: y");
  TORCH_CHECK(S >= 1 && S <= K / 64, "dm: split");
  TORCH_CHECK(ws.numel() >= S * ntile * 8 * (T / 8) * 32 * 4 || S == 1, "dm: workspace");
  TORCH_CHECK(cnt.numel() >= 2 * ntile, "dm: counters");
  dm_encode_fn enc = dm_get_encode();
  TORCH_CHECK(enc != nullptr, "dm: cuTensorMapEncodeTiled unavailable");
  CUtensorMap tm;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)Tact};
  cuuint64_t strides[1] = {(cuuint64_t)(K * 2)};
  cuuint32_t box[2] = {64u, (cuuint32_t)T};
  cuuint32_t es[2] = {1u, 1u};
  const CUresult r = enc(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dm: tensor map encode failed ", (int)r);
  dm::Params P;
  P.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  P.esc = reinterpret_cast<const uint32_t*>(esc.data_ptr());
  P.y = y.data_ptr();
  P.ws = ws.data_ptr<float>();
  P.cnt = cnt.data_ptr<int>();
  P.err = err.data_ptr<int>();
  P.ntile = (int)ntile; P.S = (int)S; P.T = (int)Tact; P.ldy = (int)y.stride(0); P.nk = (int)(K / 64); P.flags = (int)flags;
  P.tb = (uint32_t)tb;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)S, (unsigned)ntile, 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cfg.attrs = nullptr;
  cfg.numAttrs = 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tm, P));
  return 0;
}

template <int T, int ST, int STX>
static std::vector<int64_t> dm_info_t() {
  using C = dm::Cfg<T, ST, STX>;
  auto kern = dm::dm_kernel<T, ST, STX>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  int nb = -1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, C::THREADS, C::SMEM));
  return {(int64_t)C::SMEM, (int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)nb, (int64_t)T, (int64_t)ST, (int64_t)STX};
}

int64_t dm_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor esc, torch::Tensor y, torch::Tensor ws,
               torch::Tensor cnt, torch::Tensor err, int64_t S, int64_t ntile, int64_t kumax, int64_t tb, int64_t flags) {
  (void)kumax;
  if (kid < 0) {
    dm::null_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream().stream()>>>(err.data_ptr<int>());
    return 0;
  }
  switch (kid) {
@RUNCASES@
  }
  TORCH_CHECK(false, "dm: unknown kid ", kid);
  return -2;
}

std::vector<int64_t> dm_info(int64_t kid) {
  switch (kid) {
@INFOCASES@
  }
  TORCH_CHECK(false, "dm: unknown kid ", kid);
  return {};
}
"""


def dm_wg_specs(ns):
    """wgmma.mma_async m64nNk16 RS (f32 += bf16 x bf16): gv1_d / K2A operand order and immediates."""
    out = []
    for n in ns:
        na = n // 2
        regs = ",".join("%%%d" % i for i in range(na))
        outs = ", ".join('"+f"(d[%d])' % i for i in range(na))
        out.append(
            "template <> __device__ __forceinline__ void wgmma_rs<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
            "uint32_t a3, uint64_t db) {\n"
            '  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 "\n'
            '               "{%s}, {%%%d,%%%d,%%%d,%%%d}, %%%d, p, 1, 1, 0;\\n}\\n"\n'
            '               : %s\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));\n}'
            % (n, n, regs, na, na + 1, na + 2, na + 3, na + 4, outs))
    return "\n".join(out)


def dm_cuda_src(insts=DM_INSTS):
    run = "\n".join(f"    case {i}: return dm_launch_t<{t}, {s}, {x}>(x, w, esc, y, ws, cnt, err, S, ntile, tb, flags);"
                    for i, (t, s, x) in enumerate(insts))
    info = "\n".join(f"    case {i}: return dm_info_t<{t}, {s}, {x}>();" for i, (t, s, x) in enumerate(insts))
    ns = sorted({t for t, _, _ in insts})
    return DM_CUDA_SRC.replace("@RUNCASES@", run).replace("@INFOCASES@", info).replace("@WGSPECS@", dm_wg_specs(ns))


def dm_build(insts=DM_INSTS, name="dmlab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda = dm_cuda_src(insts)
    tag = hashlib.sha1((DM_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=DM_CPP_SRC, cuda_sources=cuda, functions=["dm_run", "dm_info"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
                      verbose=verbose)
    print(f"[dm] built {len(insts)} kernels in {time.time() - t0:.1f}s", flush=True)
    return mod


# ---------------------------------------------------------------- packing (gv1_d's P12 pack, per K split, plain mma order)
def dm_pack(w, S):
    """w [N, K] bf16 -> dict(w = stream [KUmax, S, ntile, 8, 1536 B] int32 view, esc [S*ntile*8*32], tb, ntile, kumax, nesc).
    Split s owns k64 units [nk s / S, nk (s + 1) / S) (the kernel's split); stream slots beyond a split's units are zero."""
    import gv1_d as gv
    n, k = w.shape
    assert n % 128 == 0 and k % 64 == 0
    nk, ntile = k // 64, n // 128
    h7 = ((w.contiguous().view(torch.int16).to(torch.int32) >> 8) & 0x7F)
    b = gv._window_base(h7)
    bases = (b, b)
    ranges = [((nk * s) // S, (nk * (s + 1)) // S) for s in range(S)]
    kumax = max(e - a for a, e in ranges)
    stream = torch.zeros((kumax, S, ntile * 8, 384), dtype=torch.int32, device=w.device)
    escs, nesc, tb = [], 0, None
    for s, (a, e) in enumerate(ranges):
        p = gv.pack(w[:, a * 64:e * 64].contiguous(), 0, 8, 1, e - a, 1, bases, perm=False)
        stream[:e - a, s] = p["w"].view(e - a, ntile * 8, 384)
        escs.append(p["esc"])
        nesc += p["nesc"]
        tb = p["tb"]
    return {"w": stream.view(-1), "esc": torch.cat(escs), "tb": int(tb), "ntile": ntile, "kumax": kumax, "nesc": nesc}


# ---------------------------------------------------------------- lab
PROJ = {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
SPLITS = {"qkv": (7, 6), "o": (2, 1), "pooler": (4, 3), "unpooler": (2, 1)}   # one wave at 1 CTA / SM (<= 132 CTAs)


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
    ms = [int(v) for v in arg("ms", "128,256").split(",")]
    projs = arg("projs", "qkv,o,pooler,unpooler").split(",")
    reps = int(arg("reps", "21"))
    rounds = int(arg("rounds", "10"))
    out_path = Path(arg("out", str(ROOT / "box_out" / "dm1" / "lab.json")))
    mod = dm_build(verbose=bool(int(arg("verbose", "0"))))
    if int(arg("build-only", "0")):
        return
    infos = {i: list(mod.dm_info(i)) for i in range(len(DM_INSTS))}
    for i, v in infos.items():
        print(f"[dm] inst {i} {DM_INSTS[i]}: smem {v[0]} regs {v[1]} local {v[2]} B occ {v[3]}", flush=True)
    dev = torch.device("cuda")
    torch.manual_seed(1234)
    flush = torch.empty((64 << 20,), dtype=torch.float32, device=dev).uniform_().view(4096, -1)   # 256 MB read sweep
    sink = torch.empty((flush.shape[1],), dtype=torch.float32, device=dev)
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    results = {"insts": {str(i): infos[i] for i in infos}, "rows": []}
    # empty-graph floor (one null kernel), the judge subtracts its own fixed cost
    gnull = graph_of(lambda: mod.dm_run(-1, err, err, err, err, err, err, err, 1, 1, 1, 0, 0))
    for proj in projs:
        n, k = PROJ[proj]
        w = (torch.randn((n, k), device=dev) * 0.02).to(torch.bfloat16)
        packs = {}
        for S in SPLITS[proj]:
            t0 = time.time()
            packs[S] = dm_pack(w, S)
            print(f"[dm] {proj} S{S}: packed {packs[S]['w'].numel() * 4 / 1e6:.1f} MB, {packs[S]['nesc']} escapes, "
                  f"{time.time() - t0:.1f}s", flush=True)
        ntile = n // 128
        cnt = torch.zeros((2 * ntile,), dtype=torch.int32, device=dev)
        for m in ms:
            x = (torch.randn((m, k), device=dev)).to(torch.bfloat16)
            y_ref64 = (x.double() @ w.double().t())
            y_cub = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            arms = {"cublas": (lambda x=x, y=y_cub: torch.mm(x, w.t(), out=y))}
            ys = {}
            for kid, (t, st, stx) in enumerate(DM_INSTS):
                if t != m:
                    continue
                for S in SPLITS[proj]:
                    p = packs[S]
                    ws = torch.empty((S * ntile * 8 * (t // 8) * 32 * 4,), dtype=torch.float32, device=dev)
                    for fl, tag in ((0, ""), (2, "_nomma"), (4, "_nox"), (6, "_wonly"), (8, "_noreduce"), (16, "_nodec")):
                        if fl and not (S == SPLITS[proj][0] and kid in (0, 3)):
                            continue
                        y = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                        name = f"dm{kid}_st{st}_x{stx}_S{S}{tag}"

                        def run(kid=kid, x=x, p=p, y=y, ws=ws, S=S, fl=fl):
                            mod.dm_run(kid, x, p["w"], p["esc"], y, ws, cnt, err, S, ntile, p["kumax"], p["tb"], fl)
                        arms[name] = run
                        ys[name] = (y, fl)
            # correctness (flag 0 arms) before timing
            arms["cublas"]()
            for name, fn in arms.items():
                if name != "cublas":
                    fn()
            torch.cuda.synchronize()
            e = int(err[0].item())
            scale = float(y_ref64.abs().max())
            rel_cub = float((y_cub.double() - y_ref64).abs().max()) / scale
            chk = {}
            for name, (y, fl) in ys.items():
                if fl == 0:
                    chk[name] = float((y.double() - y_ref64).abs().max()) / scale
            print(f"[dm] {proj} m{m}: err {e}; rel vs fp64: cublas {rel_cub:.2e}; "
                  + " ".join(f"{k2}={v:.2e}" for k2, v in chk.items()), flush=True)
            if e:
                print(f"[dm] {proj} m{m}: bounded wait timed out (err {e}) -> no timing for this shape", flush=True)
                results["rows"].append({"proj": proj, "m": m, "err": e, "rel": chk})
                err.zero_()
                cnt.zero_()
                continue
            names = list(arms.keys())
            graphs = [graph_of(arms[nm]) for nm in names] + [gnull]
            meds = {nm: [] for nm in names + ["null"]}
            for rd in range(rounds):
                t = time_many(graphs, flush, sink, reps)
                for nm, v in zip(names + ["null"], t):
                    meds[nm].append(v)
            fl0 = statistics.median(meds["null"])
            row = {"proj": proj, "m": m, "err": e, "rel_cublas": rel_cub, "rel": chk, "floor_us": fl0, "arms": {}}
            cub = statistics.median(meds["cublas"])
            for nm in names:
                med = statistics.median(meds[nm])
                row["arms"][nm] = {"raw_us": med, "net_us": med - fl0, "rounds": meds[nm],
                                   "x_vs_cublas_net": (cub - fl0) / (med - fl0)}
            # repeat check after timing
            for name, fn in arms.items():
                fn()
            torch.cuda.synchronize()
            row["err_after"] = int(err[0].item())
            row["rel_after"] = {name: float((y.double() - y_ref64).abs().max()) / scale for name, (y, fl) in ys.items() if fl == 0}
            results["rows"].append(row)
            print(f"[dm] {proj} m{m}: floor {fl0:.2f} us; net us (x vs cuBLAS):", flush=True)
            for nm in names:
                a = row["arms"][nm]
                print(f"    {nm:28s} {a['net_us']:7.2f}  x{a['x_vs_cublas_net']:.3f}   raw {a['raw_us']:.2f}", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=1))
    print("[dm] done", flush=True)


if __name__ == "__main__":
    main()

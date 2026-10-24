# okc: problem=muse_bf16_proj
# nonce r1b-v2-1 (measured route table, job f5e82357e66f4494: GEMV only at m=1 for qkv/pooler/unpooler, router GEMV m<=16, stock router 17..63, kcat >=64)
"""R1b: router hi||lo as one cuBLAS GEMM (L1) + a bandwidth-bound split-K GEMV for small m (L2).

L1  R1B_ROUTER_SPLIT: the FP32 router weight is split once in prepare() into BF16 hi = bf16(w) and
    lo = bf16(w - hi) (SGLang's MuseSpark1xRouter._build_split2), stored K-concatenated as [256, 2*8192].
    Per call: x2 = cat(x, x) [m, 16384] and ONE torch.mm(x2, W_cat^T, out_dtype=fp32) (FP32 accumulation).
    Replaces SGLang's FP32 F.linear (m <= 8) and its two GEMMs + add (m > 8). Modes: "kcat" (default),
    "ncat" ([hi; lo] along N, one GEMM then a half-sum), "two" (SGLang's exact path).
L2  R1B_GEMV: one CUDA kernel (load_inline) for m <= 8 on the BF16 layers and m <= 32 on the router, which
    it runs on the FP32 weight directly (same 8.4 MB as hi+lo, and exact). CTA = 8 warps; warp (wr, wk) owns
    RPW consecutive rows and every WK-th 32*VEC*U-wide K chunk of the CTA's K slice; 16-B evict-first weight
    loads, U=4 vectors per row in flight per lane plus the next chunk prefetched; x[MT tokens][K slice] staged
    once per CTA in shared memory (zeros past m); FP32 accumulation; shuffle + smem reduce over lanes and WK.
    Split-K over CTAs (grid.y = S) is reduced in the same launch by the last-arriving CTA (self-resetting
    counter, R1B_FIXUP) or by a second tiny launch. grid.z tiles tokens by MT (router m = 16, 32).
    Per-(layer, m-bucket) route table GEMV_ROUTES; anything absent goes to cuBLAS (SGLang's dispatch).
Every lever is a constant; prepare() self-tests each route against an FP64 product (probe rows included)
and drops only the failing route. CUDA-graph safe: no syncs, branching on shapes only, fixed grids.
"""

import hashlib
import time
from typing import TypedDict

import torch
import torch.nn.functional as F

R1B_ROUTER_SPLIT = True        # L1: router hi||lo, FP32 accumulation, at every m not taken by the GEMV
R1B_ROUTER_SPLIT_MODE = "kcat"  # "kcat" | "ncat" | "two"
R1B_GEMV = True                # L2: custom GEMV for small m (route table below)
R1B_FIXUP = True               # L2: split-K reduced in-kernel by the last CTA (False: second reduce launch)
R1B_SELF_TEST = True           # prepare(): check every live route vs FP64, drop the ones that fail
ROUTER_GEMV_MAX_M = 16         # measured: router GEMV 1.57x (m1), 3.82x (m8), 1.35x (m16), 0.88x (m32)
ROUTER_STOCK_M = (17, 63)      # router m in this range uses SGLang's own path (kcat/GEMV unproven there)

# (layer, bucket) -> (MT tokens per CTA, RPW rows per warp, WK warps along K, S CTAs along K).
# bucket: 1 for m == 1, 8 for 2 <= m <= 8, 32 for 9 <= m <= 32 (router only). The unpooler at m <= 8 is
# the 1024-row slice (config n). Shapes are [N, K]: qkv_gate 2304x8192, o 8192x1024, pooler 4096x8192,
# unpooler 1024x4096 (sliced), router 256x8192 FP32. Rows per CTA = (8 / WK) * RPW.
GEMV_ROUTES = {
    ("qkv_gate", 1): (1, 1, 1, 1),   # 288 CTAs, 8 rows x 16 KiB, one launch
    ("pooler", 1): (1, 1, 1, 1),     # 512 CTAs, 8 rows x 16 KiB
    ("unpooler", 1): (1, 1, 4, 1),   # 512 CTAs, 2 rows x 8 KiB, 4 warps per row
    ("router", 8): (8, 1, 1, 8),     # 256 CTAs, 8 rows x 4 KiB (FP32), x stage 16 KiB
    ("router", 32): (8, 1, 1, 8),    # grid.z = ceil(m / 8): weights re-read from L2
}

NWARP = 8
UNROLL = 4
SMEM_MAX = 40 * 1024
COUNTERS = 16384


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]; up to 8 tokens, only rows 0-1023 (this GPU's slice)
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [M, K]
    config: Config


print(f"[r1b] levers: ROUTER_SPLIT={R1B_ROUTER_SPLIT}/{R1B_ROUTER_SPLIT_MODE} GEMV={R1B_GEMV} FIXUP={R1B_FIXUP} "
      f"SELF_TEST={R1B_SELF_TEST} ROUTER_GEMV_MAX_M={ROUTER_GEMV_MAX_M} routes={len(GEMV_ROUTES)}", flush=True)


# ----------------------------------------------------------------------------------------------
# Host-side geometry (shared with r1b_emulate.py, which checks it on CPU)
# ----------------------------------------------------------------------------------------------

def bucket(layer: str, m: int):
    if m == 1 and layer != "router":
        return 1
    if m <= 8:
        return 8
    if layer == "router" and m <= ROUTER_GEMV_MAX_M:
        return 32
    return None


def vec_of(w_dtype) -> int:
    """Weight elements per 16-B vector."""
    return 4 if w_dtype == torch.float32 else 8


def geometry(m: int, n: int, k: int, vec: int, route):
    """Launch geometry of one GEMV call, or None when the route does not fit this shape.
    Returns dict(grid=(gx, gy, gz), kslice, chunk, nchunk, rows_cta, smem)."""
    mt, rpw, wk, split = route
    if NWARP % wk or k % split or m < 1:
        return None
    kslice = k // split
    chunk = 32 * vec * UNROLL
    rows_cta = (NWARP // wk) * rpw
    smem = mt * kslice * 2
    if kslice % chunk or n % rows_cta or n % 4 or smem > SMEM_MAX:
        return None
    gz = (m + mt - 1) // mt
    if gz > 65535 or (n // rows_cta) * gz > COUNTERS:
        return None
    return {"grid": (n // rows_cta, split, gz), "kslice": kslice, "chunk": chunk, "nchunk": kslice // chunk,
            "rows_cta": rows_cta, "smem": smem}


def split_router(w32: torch.Tensor):
    """FP32 [N, K] -> (hi, lo) BF16 with hi + lo ~= w (SGLang's split): hi = bf16(w), lo = bf16(w - hi)."""
    hi = w32.to(torch.bfloat16)
    lo = (w32 - hi.float()).to(torch.bfloat16)
    return hi, lo


# ----------------------------------------------------------------------------------------------
# CUDA
# ----------------------------------------------------------------------------------------------

CPP_SRC = r"""
#include <torch/extension.h>
void r1b_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor part, torch::Tensor counters,
          int64_t M, int64_t N, int64_t K, int64_t mt, int64_t rpw, int64_t wk, int64_t split, int64_t fixup);
void r1b_reduce(torch::Tensor part, torch::Tensor out, int64_t split);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace r1b {

constexpr int NT = 256;       // threads per CTA
constexpr int NW = 8;         // warps per CTA
constexpr int U = 4;          // 16-B weight vectors per lane per row per chunk

template <typename WT> struct Vec;
template <> struct Vec<__nv_bfloat16> { static constexpr int N = 8; };
template <> struct Vec<float> { static constexpr int N = 4; };

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// 16 B of weights -> VEC floats
__device__ __forceinline__ void unpack_w(const float4 v, float (&f)[8], __nv_bfloat16*) {
  const uint32_t a = __float_as_uint(v.x), b = __float_as_uint(v.y), c = __float_as_uint(v.z), d = __float_as_uint(v.w);
  f[0] = bf_lo(a); f[1] = bf_hi(a); f[2] = bf_lo(b); f[3] = bf_hi(b);
  f[4] = bf_lo(c); f[5] = bf_hi(c); f[6] = bf_lo(d); f[7] = bf_hi(d);
}
__device__ __forceinline__ void unpack_w(const float4 v, float (&f)[4], float*) {
  f[0] = v.x; f[1] = v.y; f[2] = v.z; f[3] = v.w;
}
// VEC bf16 of x from shared memory -> floats
__device__ __forceinline__ void load_x(const __nv_bfloat16* p, float (&f)[8]) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}
__device__ __forceinline__ void load_x(const __nv_bfloat16* p, float (&f)[4]) {
  const uint2 v = *reinterpret_cast<const uint2*>(p);
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
}
__device__ __forceinline__ void store_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }

// out[M, N] = x[M, K] . w[N, K]^T (rows of w at stride K), FP32 accumulation; OT = WT (bf16 or fp32).
// grid = (N / rows_cta, split, ceil(M / MT)); block = 256; dynamic smem = MT * kslice * 2 bytes.
template <typename WT, int MT, int RPW>
__global__ void __launch_bounds__(NT, 2) gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const WT* __restrict__ w, WT* __restrict__ out,
    float* __restrict__ part, int* __restrict__ counters,
    int M, int N, int K, int kslice, int WK, int fixup) {
  constexpr int VEC = Vec<WT>::N;
  constexpr int CH = 32 * VEC * U;                 // K elements per warp chunk (per row)
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // [MT][kslice]
  __shared__ float red[NW * RPW * MT];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wk = warp % WK, wr = warp / WK;
  const int rows_cta = (NW / WK) * RPW;
  const int row0 = blockIdx.x * rows_cta + wr * RPW;
  const int kbase = blockIdx.y * kslice;
  const int t0 = blockIdx.z * MT;
  const int mloc = min(MT, M - t0);
  const int nch = kslice / CH;

  // first chunk's weights go out before the x stage, so the stage hides under their latency
  float4 wv[RPW][U];
  int c = wk;
  if (c < nch) {
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const WT* p = w + (size_t)(row0 + r) * K + kbase + c * CH + lane * VEC;
#pragma unroll
      for (int u = 0; u < U; ++u) wv[r][u] = __ldcs(reinterpret_cast<const float4*>(p + u * 32 * VEC));
    }
  }

  // stage x[t0 : t0 + MT, kbase : kbase + kslice] (zeros for tokens >= mloc)
  const int vpr = kslice >> 3;                     // 16-B vectors per staged token row
  for (int i = tid; i < MT * vpr; i += NT) {
    const int t = i / vpr, v = i - t * vpr;
    uint4 val = make_uint4(0u, 0u, 0u, 0u);
    if (t < mloc) val = *reinterpret_cast<const uint4*>(x + (size_t)(t0 + t) * K + kbase + v * 8);
    reinterpret_cast<uint4*>(xs)[i] = val;
  }
  __syncthreads();

  float acc[RPW][MT];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;

  while (c < nch) {
    float4 cur[RPW][U];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int u = 0; u < U; ++u) cur[r][u] = wv[r][u];
    const int cn = c + WK;
    if (cn < nch) {                                // prefetch the next chunk
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const WT* p = w + (size_t)(row0 + r) * K + kbase + cn * CH + lane * VEC;
#pragma unroll
        for (int u = 0; u < U; ++u) wv[r][u] = __ldcs(reinterpret_cast<const float4*>(p + u * 32 * VEC));
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int kk = c * CH + u * 32 * VEC + lane * VEC;   // offset inside the K slice
      float wf[RPW][VEC];
#pragma unroll
      for (int r = 0; r < RPW; ++r) unpack_w(cur[r][u], wf[r], (WT*)nullptr);
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        float xf[VEC];
        load_x(xs + t * kslice + kk, xf);
#pragma unroll
        for (int r = 0; r < RPW; ++r)
#pragma unroll
          for (int j = 0; j < VEC; ++j) acc[r][t] = fmaf(wf[r][j], xf[j], acc[r][t]);
      }
    }
    c = cn;
  }

  // lanes -> warp total, warps -> smem
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      acc[r][t] = v;
    }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int t = 0; t < MT; ++t) red[(warp * RPW + r) * MT + t] = acc[r][t];
  }
  __syncthreads();

  // value i = (t, rr): token t0 + t, row blockIdx.x * rows_cta + rr; rr = wr' * RPW + r'
  const int split = gridDim.y;
  for (int i = tid; i < rows_cta * MT; i += NT) {
    const int t = i / rows_cta, rr = i - t * rows_cta;
    if (t >= mloc) continue;
    const int wr2 = rr / RPW, r2 = rr - wr2 * RPW;
    float s = 0.f;
    for (int q = 0; q < WK; ++q) s += red[((wr2 * WK + q) * RPW + r2) * MT + t];
    const size_t o = (size_t)(t0 + t) * N + (size_t)blockIdx.x * rows_cta + rr;
    if (split == 1) store_out(out + o, s);
    else part[(size_t)blockIdx.y * M * N + o] = s;
  }
  if (split == 1 || !fixup) return;

  // split-K fixup: the last CTA of this (row tile, token tile) sums the split partials in order q = 0..S-1
  __threadfence();
  __syncthreads();
  const int tile = blockIdx.z * gridDim.x + blockIdx.x;
  if (tid == 0) s_last = (atomicAdd(counters + tile, 1) == split - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int i = tid; i < rows_cta * MT; i += NT) {
    const int t = i / rows_cta, rr = i - t * rows_cta;
    if (t >= mloc) continue;
    const size_t o = (size_t)(t0 + t) * N + (size_t)blockIdx.x * rows_cta + rr;
    float s = 0.f;
    for (int q = 0; q < split; ++q) s += __ldcg(part + (size_t)q * M * N + o);
    store_out(out + o, s);
  }
  if (tid == 0) counters[tile] = 0;                // self-reset for the next launch / graph replay
}

// out[i] = sum_q part[q][i], i over M*N (multiple of 4)
template <typename OT>
__global__ void __launch_bounds__(256) reduce_kernel(const float* __restrict__ part, OT* __restrict__ out,
                                                     int MN, int split) {
  const int i = (blockIdx.x * 256 + threadIdx.x) * 4;
  if (i >= MN) return;
  float4 s = *reinterpret_cast<const float4*>(part + i);
  for (int q = 1; q < split; ++q) {
    const float4 v = *reinterpret_cast<const float4*>(part + (size_t)q * MN + i);
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  store_out(out + i, s.x); store_out(out + i + 1, s.y); store_out(out + i + 2, s.z); store_out(out + i + 3, s.w);
}

template <typename WT, int MT, int RPW>
static void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, torch::Tensor& part,
                   torch::Tensor& counters, int M, int N, int K, int wk, int split, int fixup) {
  const int kslice = K / split;
  const int rows_cta = (NW / wk) * RPW;
  const dim3 grid(N / rows_cta, split, (M + MT - 1) / MT);
  const size_t smem = (size_t)MT * kslice * 2;
  auto st = at::cuda::getCurrentCUDAStream();
  gemv_kernel<WT, MT, RPW><<<grid, NT, smem, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const WT*>(w.data_ptr()),
      reinterpret_cast<WT*>(out.data_ptr()), split > 1 ? part.data_ptr<float>() : nullptr,
      counters.data_ptr<int>(), M, N, K, kslice, wk, fixup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace r1b

void r1b_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor part, torch::Tensor counters,
          int64_t M, int64_t N, int64_t K, int64_t mt, int64_t rpw, int64_t wk, int64_t split, int64_t fixup) {
  const bool f32 = w.scalar_type() == at::kFloat;
  const int vec = f32 ? 4 : 8;
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(0) == M && x.size(1) == K, "x");
  TORCH_CHECK(w.is_contiguous() && w.size(1) == K && w.size(0) >= N, "w");
  TORCH_CHECK(f32 || w.scalar_type() == at::kBFloat16, "w dtype");
  TORCH_CHECK(out.scalar_type() == w.scalar_type() && out.is_contiguous() && out.size(0) == M && out.size(1) == N, "out");
  TORCH_CHECK(r1b::NW % wk == 0 && split >= 1 && K % split == 0, "wk/split");
  const int64_t kslice = K / split, rows_cta = (r1b::NW / wk) * rpw;
  TORCH_CHECK(kslice % (32 * vec * r1b::U) == 0 && N % rows_cta == 0 && N % 4 == 0, "geometry");
  TORCH_CHECK(mt * kslice * 2 <= 40 * 1024, "smem");
  TORCH_CHECK(split == 1 || (part.scalar_type() == at::kFloat && part.numel() >= split * M * N), "part");
  TORCH_CHECK(counters.numel() >= (N / rows_cta) * ((M + mt - 1) / mt), "counters");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "align");
  const int m = (int)M, n = (int)N, k = (int)K, wk_ = (int)wk, sp = (int)split, fx = (int)fixup;
  if (f32) {
    TORCH_CHECK(mt == 8 && (rpw == 1 || rpw == 2), "fp32 variant");
    if (rpw == 1) r1b::launch<float, 8, 1>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
    else r1b::launch<float, 8, 2>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
  } else {
    TORCH_CHECK((mt == 1 || mt == 8) && (rpw == 1 || rpw == 2), "bf16 variant");
    if (mt == 1 && rpw == 1) r1b::launch<__nv_bfloat16, 1, 1>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
    else if (mt == 1) r1b::launch<__nv_bfloat16, 1, 2>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
    else if (rpw == 1) r1b::launch<__nv_bfloat16, 8, 1>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
    else r1b::launch<__nv_bfloat16, 8, 2>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
  }
}

void r1b_reduce(torch::Tensor part, torch::Tensor out, int64_t split) {
  const int MN = (int)out.numel();
  TORCH_CHECK(MN % 4 == 0 && part.numel() >= split * MN && out.is_contiguous(), "reduce");
  auto st = at::cuda::getCurrentCUDAStream();
  const int blocks = (MN / 4 + 255) / 256;
  if (out.scalar_type() == at::kFloat)
    r1b::reduce_kernel<float><<<blocks, 256, 0, st>>>(part.data_ptr<float>(), out.data_ptr<float>(), MN, (int)split);
  else
    r1b::reduce_kernel<__nv_bfloat16><<<blocks, 256, 0, st>>>(
        part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), MN, (int)split);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CPP_SRC + CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"r1b_gemv_{tag}",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["r1b_gemv", "r1b_reduce"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        verbose=True,
    )
    print(f"[r1b] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


# ----------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------

_DISPATCH = {"fn": None}


def _bf16_gemm(x, w):
    fn = _DISPATCH["fn"]
    return fn(x, w, None) if fn is not None else F.linear(x, w)


def _router_split(x, state, mode):
    """L1: router logits from the BF16 hi/lo pair with FP32 accumulation."""
    if mode == "kcat":
        return torch.mm(torch.cat((x, x), dim=1), state["router_cat"].t(), out_dtype=torch.float32)
    if mode == "ncat":
        n = state["router_ncat"].shape[0] // 2
        y = torch.mm(x, state["router_ncat"].t(), out_dtype=torch.float32)
        return y[:, :n] + y[:, n:]
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _router_stock(x, state):
    """SGLang's own router path (baseline.py)."""
    if x.shape[0] <= 8:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _gemv(state, x, w, n, route):
    """L2 launch(es). Shapes were checked by geometry() when the route was admitted."""
    mod = state["mod"]
    m, k = x.shape
    mt, rpw, wk, split = route
    out = torch.empty((m, n), dtype=w.dtype, device=x.device)
    part = state["dummy"] if split == 1 else torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    fix = 1 if R1B_FIXUP else 0
    mod.r1b_gemv(x, w, out, part, state["counters"], m, n, k, mt, rpw, wk, split, fix)
    if split > 1 and not fix:
        mod.r1b_reduce(part, out, split)
    return out


def _gemv_route(state, layer, m, n, k):
    if state.get("mod") is None:
        return None
    b = bucket(layer, m)
    if b is None:
        return None
    route = state["routes"].get((layer, b))
    if route is None:
        return None
    w = state[layer]
    if geometry(m, n, k, vec_of(w.dtype), route) is None:
        return None
    return route


def _aligned(x):
    return x.is_contiguous() and x.data_ptr() % 16 == 0


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    x, layer, n = data["x"], cfg["layer"], cfg["n"]
    m, k = x.shape
    route = _gemv_route(state, layer, m, n, k)
    if route is not None and _aligned(x):
        return _gemv(state, x, state[layer], n, route)
    if layer != "router":
        # the unpooler's slice is its first n rows (all of them above 8 tokens)
        return _bf16_gemm(x, state[layer][:n])
    if ROUTER_STOCK_M[0] <= m <= ROUTER_STOCK_M[1]:
        return _router_stock(x, state)
    if state.get("router_mode"):
        return _router_split(x, state, state["router_mode"])
    return _router_stock(x, state)


# ----------------------------------------------------------------------------------------------
# prepare + self-test
# ----------------------------------------------------------------------------------------------

def _test_input(m, k, seed, device, outliers=False, probe="force"):
    """The judge's recipe (normal x, optional 30x channels, ~1/8 probe rows with one power-of-two entry).
    probe: "force" = at least one probe row at any m; "judge" = forced only at m >= 8; "none" = all rows dense."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device=device)
    if outliers:
        idx = torch.randperm(k, generator=gen, device=device)[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device=device) < 1 / 8
    if probe == "force" or (probe == "judge" and m >= 8):
        probes[torch.randint(m, (1,), generator=gen, device=device)] = True
    if probe == "none":
        probes[:] = False
    cols = torch.randint(k, (m,), generator=gen, device=device)
    signs = torch.randint(2, (m,), generator=gen, device=device) * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device=device)
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16), probes


def _errors(out, x, w, probes):
    """(total rel-L2, worst row with the judge's 0.1-of-mean floor, worst probe row) vs FP64."""
    ref = x.double() @ w.double().T
    ref = ref.to(out.dtype).double()
    d = out.double() - ref
    tot = float(d.norm() / ref.norm().clamp(min=1e-30))
    rn = ref.norm(dim=-1)
    row = float((d.norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
    pr = 0.0
    if bool(probes.any()):
        pr = float((d[probes].norm(dim=-1) / rn[probes].clamp(min=1e-30)).max())
    return tot, row, pr


def _self_test(state, weights):
    dev = weights["router"].device
    ok_routes = {}
    for (layer, b), route in state["routes"].items():
        n_full, k = weights[layer].shape
        ms = {1: (1,), 8: (8, 5, 1), 32: (9, 16, 32)}[b]
        good = True
        for m in ms:
            n = 1024 if (layer == "unpooler" and m <= 8) else n_full
            if geometry(m, n, k, vec_of(weights[layer].dtype), route) is None:
                print(f"[r1b] route {layer}/{b} {route}: geometry rejects m={m} n={n} k={k}", flush=True)
                good = False
                break
            for outl, probe in ((False, "none"), (False, "force"), (True, "none"), (True, "judge")):
                x, probes = _test_input(m, k, 1000 + 7 * m + (1 if outl else 0), dev, outl, probe)
                try:
                    tmp = dict(state)
                    tmp["routes"] = {(layer, b): route}
                    got = _gemv(tmp, x, state[layer], n, route)
                    got2 = _gemv(tmp, x, state[layer], n, route)   # counters must have self-reset
                    torch.cuda.synchronize()
                    tot, row, pr = _errors(got, x, weights[layer][:n], probes)
                    same = bool(torch.equal(got, got2))
                except Exception as ex:  # noqa: BLE001
                    print(f"[r1b] route {layer}/{b} m={m}: EXC {ex!r}", flush=True)
                    good = False
                    break
                tol = 1e-5 if layer == "router" else 4e-3
                ptol = 1e-6 if layer == "router" else 1e-6
                fin = bool(torch.isfinite(got.float()).all())
                print(f"[r1b] self-test {layer}/{b} {route} m={m} outl={outl} probe={probe}: rel {tot:.2e} row {row:.2e} "
                      f"probe {pr:.1e} repeat_eq={same} finite={fin}", flush=True)
                if not (fin and same and tot < tol and row < 5 * tol and pr <= ptol):
                    good = False
            if not good:
                break
        if good:
            ok_routes[(layer, b)] = route
        else:
            print(f"[r1b] route {layer}/{b} DROPPED -> cuBLAS", flush=True)
    return ok_routes


def _self_test_router_split(state, weights):
    w = weights["router"]
    dev = w.device
    for m in (1, 8, 9, 33, 128):
        for probe in ("none", "force"):
            x, probes = _test_input(m, w.shape[1], 2000 + m, dev, outliers=(m == 9), probe=probe)
            got = _router_split(x, state, state["router_mode"])
            torch.cuda.synchronize()
            tot, row, pr = _errors(got, x, w, probes)
            print(f"[r1b] self-test router_split/{state['router_mode']} m={m} probe={probe}: rel {tot:.2e} "
                  f"row {row:.2e} probe {pr:.1e}", flush=True)
            if not (bool(torch.isfinite(got).all()) and tot < 2e-5 and row < 1e-4 and pr < 1e-4):
                return False
    return True


def prepare(weights: Weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    dev = state["router"].device
    try:
        from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
        _DISPATCH["fn"] = _bf16_gemm_dispatch_impl
    except Exception as ex:  # noqa: BLE001
        print(f"[r1b] SGLang dispatch unavailable, using F.linear: {ex!r}", flush=True)
    hi, lo = split_router(state["router"])
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_cat"] = torch.cat((hi, lo), dim=1).contiguous()      # [256, 16384]
    state["router_ncat"] = torch.cat((hi, lo), dim=0).contiguous()     # [512, 8192]
    state["router_mode"] = R1B_ROUTER_SPLIT_MODE if R1B_ROUTER_SPLIT else None
    if state["router_mode"]:
        try:
            if not _self_test_router_split(state, weights):
                raise RuntimeError("tolerance")
        except Exception as ex:  # noqa: BLE001
            print(f"[r1b] router split {state['router_mode']} FAILED -> SGLang router path: {ex!r}", flush=True)
            state["router_mode"] = None
    state["mod"] = None
    state["routes"] = {}
    state["counters"] = torch.zeros((COUNTERS,), dtype=torch.int32, device=dev)
    state["dummy"] = torch.zeros((4,), dtype=torch.float32, device=dev)
    if R1B_GEMV:
        try:
            state["mod"] = _build()
            state["routes"] = dict(GEMV_ROUTES)
        except Exception as ex:  # noqa: BLE001
            msg = str(ex)
            print("[r1b] CUDA build FAILED -> cuBLAS everywhere. Compiler output tail:\n" + msg[-12000:], flush=True)
            state["mod"] = None
        if state["mod"] is not None and R1B_SELF_TEST:
            try:
                state["routes"] = _self_test(state, weights)
            except Exception as ex:  # noqa: BLE001
                print(f"[r1b] self-test crashed -> cuBLAS everywhere: {ex!r}", flush=True)
                state["mod"] = None
                state["routes"] = {}
            state["counters"].zero_()
    live = sorted(f"{a}/{b}" for a, b in state["routes"])
    print(f"[r1b] prepare done in {time.time() - t0:.1f}s; router_mode={state['router_mode']}; "
          f"gemv routes live: {live}", flush=True)
    return state

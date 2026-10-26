"""P1M: P1's lossless 12-bit packed projections for T = 9..32 tokens (decode 16 / 32), the same pack (p1_module.p1_pack).

P1's p1s kernel holds one n8 token group (T <= 8). P1M generalizes it (notes/study/scout/raw/r1_c.md, Nathen v18 "MT2 / MT4"):
  * MT n8 token groups per warp (T <= 8 MT): every decoded A fragment feeds MT mma.m16n8k16 (no re-decode per token group);
  * RT row tiles per warp (16 RT rows): every x B fragment feeds RT row tiles, so the x traffic from L2 drops by RT
    (x bytes per weight byte = 0.67 MT / RT; at T = 32, RT = 1 that is 2.7x the weight stream);
  * D-deep register ring of weight SBs and x fragments (as p1s), x tokens >= T clamped to T - 1 (no zero fill);
  * per-row-tile format branch (packed / RAW, CTA-uniform since RAW groups are 16 tiles), BF16 or FP32 output (the pooler
    "cat" GEMM [pooler; router hi; router lo] is FP32 out), split-K with the same deterministic last-CTA fixup.
Bit-exactness of the weights is P1's (the pack / unpack check); the arithmetic is FP32 accumulation of exact BF16 products.

    mod = p1m_build(); P = p1_pack(w); p1m_call(mod, P, x, out, geo=(warps, split, depth, mt, rt))
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p1_module", os.path.join(HERE, "p1_module.py"))
P1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P1)

# (depth, mt, rt) instantiated in the lab build; the layer builds only its routed winners
P1M_INST = ((1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (1, 2, 1), (2, 2, 1), (1, 2, 2), (2, 2, 2), (1, 4, 1), (2, 4, 1), (1, 4, 2))

P1M_CPP_ADD = r"""
void p1m_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mt, int64_t rt,
              int64_t pdl, torch::Tensor ilbuf, int64_t il_step);
std::vector<int64_t> p1m_info(int64_t depth, int64_t mt, int64_t rt, int64_t warps);
void p1k_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t ctas, int64_t depth, int64_t mt, int64_t rt,
              int64_t kmax, int64_t pdl);
std::vector<int64_t> p1k_info(int64_t depth, int64_t mt, int64_t rt, int64_t warps);
"""


def _inst_lines(inst, fmt):
    return "\n".join(fmt.format(d=d, m=m, r=r) for d, m, r in inst)


def p1m_cuda_src(inst=P1M_INST, kinst=None):
    kinst = inst if kinst is None else kinst
    disp = _inst_lines(inst, "  P1M_DISPATCH({d}, {m}, {r}, (p1m::launch<{d}, {m}, {r}>(A, w, S, pd)); return)")
    info = _inst_lines(inst, "  P1M_DISPATCH({d}, {m}, {r}, return (p1m::info_t<{d}, {m}, {r}>(w)))")
    kdisp = _inst_lines(kinst, "  P1M_DISPATCH({d}, {m}, {r}, (p1m::launch_k<{d}, {m}, {r}>(A, Kp, w, C, pd)); return)")
    kinfo = _inst_lines(kinst, "  P1M_DISPATCH({d}, {m}, {r}, return (p1m::info_k<{d}, {m}, {r}>(w)))")
    return (P1M_CUDA_ADD.replace("@DISPATCH@", disp).replace("@INFO@", info).replace("@KDISPATCH@", kdisp)
            .replace("@KINFO@", kinfo))


P1M_CUDA_ADD = r"""
namespace p1m {
using p1::SBP; using p1::SBR; using p1::BIG; using p1::dec_kt; using p1::mma16816; using p1::ldw;
using p1::esc_key; using p1::esc_patch4;

// x B fragments from the k-permuted pack (p1m_pack): lane t's 16 k of a 64-k SB are contiguous -> two 16-B loads per token
__device__ __forceinline__ uint4 ldx4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void ldb(uint32_t (&b)[8], const uint4* p) {
  const uint4 q0 = ldx4(p), q1 = ldx4(p + 1);
  b[0] = q0.x; b[1] = q0.y; b[2] = q0.z; b[3] = q0.w; b[4] = q1.x; b[5] = q1.y; b[6] = q1.z; b[7] = q1.w;
}

// warps per CTA: up to 16 (512 threads) at RT = 1, 8 at RT = 2 (the [warp][row][token] partials stay < 48 KB static smem)
template <int MT, int RT> struct MaxW { static constexpr int v = RT == 1 ? 16 : 8; };

struct MArgs {
  const __nv_bfloat16* x; int M; int ldx;             // x [M, K] bf16 row-major, 1 <= M <= 8 MT
  const uint8_t* pk; const int64_t* toff; const float* scale; const int* eoff; const int2* esc; const uint8_t* fmt;
  void* out; int ldo; int N; int nsb_total; int f32;  // out [M, ldo] bf16 (f32 = 0) or fp32 (f32 = 1), columns [0, N)
  float* ws; int* cnt;                                 // split > 1: FP32 slots [S][M][N] + counters [N / (16 RT)]
  int probe;                                           // 1: stream probe (loads consumed by XOR, no decode / mma; timing only)
  long long il_step;                                   // 0: P1 layout (toff + sb * SB bytes); > 0: grid-interleaved stream
                                                       //    order (p1m_interleave): warp gw's step-s block of row tile r at
                                                       //    (gw * RT + r) * 1536 + s * il_step (every step one coherent front)
};

// One warp streams SBs [sb_lo, sb_lo + nsb) of row tiles rt0 .. rt0 + RT - 1 (all packed, or all RAW: KM 1) for MT token
// groups. acc[r][m]: rows 16 r + g (+8), tokens 8 m + 2 t (+1).
template <int D, int MT, int RT, int KM>
__device__ __forceinline__ void mstream(const MArgs& A, int rt0, int gw, int lane, int sb_lo, int nsb, float (&acc)[RT][MT][4]) {
  constexpr bool RAW = KM == 1;
  constexpr int NCH = RAW ? 4 : 3;
  constexpr int SBB = RAW ? SBR : SBP;
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* wp[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r)
    wp[r] = A.il_step ? A.pk + ((long long)gw * RT + r) * SBP + lane * 16
                      : A.pk + A.toff[rt0 + r] + (long long)sb_lo * SBB + lane * 16;
  const long long step = A.il_step ? A.il_step : SBB;
  // k-permuted pack: logical (kt, 2t + 8a + h) of an SB = physical k 16t + 4kt + 2a + h, so b[j] = word j of x[16t .. 16t + 15]
  const uint4* xq[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int tok = min(g + 8 * m, A.M - 1);
    xq[m] = reinterpret_cast<const uint4*>(A.x + (size_t)tok * A.ldx + (size_t)sb_lo * 64 + 16 * t);
  }
  uint4 w[D][RT][NCH];
  uint32_t b[D][MT][8];
  // 1. weights of the first D SBs: prepare()-time data, issued before the PDL wait
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < nsb) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int c = 0; c < NCH; ++c) w[d][r][c] = ldw(wp[r] + d * step + c * 512);
    }
  // 2. escape entries of each row tile (one per lane; in-range ones contiguous: the list is key-sorted)
  uint32_t em[RT], ev[RT];
  int cur[RT], end[RT], nkey[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    em[r] = 0u; ev[r] = 0u; cur[r] = 32; end[r] = 0; nkey[r] = BIG;
    if constexpr (!RAW) {
      const int o0 = A.eoff[rt0 + r], o1 = A.eoff[rt0 + r + 1];
      const int idx = o0 + lane;
      int2 e = make_int2(0, 0);
      if (idx < o1) e = A.esc[idx];
      em[r] = (uint32_t)e.x;
      ev[r] = (uint32_t)e.y;
      const int esb = (int)(em[r] & 0xFFFFu);
      const bool in = idx < o1 && esb >= sb_lo && esb < sb_lo + nsb;
      const uint32_t msk = __ballot_sync(0xffffffffu, in);
      cur[r] = msk ? __ffs(msk) - 1 : 32;
      end[r] = msk ? 32 - __clz(msk) : 0;
      const uint32_t mf = __shfl_sync(0xffffffffu, em[r], cur[r] & 31);
      nkey[r] = msk ? esc_key(mf, sb_lo) : BIG;
    }
  }
  // 3. PDL: x (and the output) belong to the predecessor kernel. A no-op without programmatic serialization.
  asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < nsb) {
#pragma unroll
      for (int m = 0; m < MT; ++m) ldb(b[d][m], xq[m] + d * 8);
    }
  // 4. rolling stream
  for (int s0 = 0; s0 < nsb; s0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int s = s0 + d;
      if (s < nsb) {
        if constexpr (KM == 3) {
          uint32_t px = 0u;
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int c = 0; c < NCH; ++c) px ^= w[d][r][c].x ^ w[d][r][c].y ^ w[d][r][c].z ^ w[d][r][c].w;
#pragma unroll
          for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int j = 0; j < 8; ++j) px ^= b[d][m][j];
          acc[0][0][0] += __uint_as_float(px & 0x007FFFFFu);   // finite, keeps the loads live
        } else {
#pragma unroll
          for (int kt = 0; kt < 4; ++kt) {
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              uint32_t a[4];
              if constexpr (RAW) {
                a[0] = w[d][r][kt].x; a[1] = w[d][r][kt].y; a[2] = w[d][r][kt].z; a[3] = w[d][r][kt].w;
              } else {
                dec_kt(w[d][r], kt, a);
                const int key = s * 4 + kt;
                if (key == nkey[r]) esc_patch4(a, key, sb_lo, lane, em[r], ev[r], cur[r], end[r], nkey[r]);
              }
#pragma unroll
              for (int m = 0; m < MT; ++m) mma16816(acc[r][m], a, b[d][m][2 * kt], b[d][m][2 * kt + 1]);
            }
          }
        }
        if (s + D < nsb) {
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int c = 0; c < NCH; ++c) w[d][r][c] = ldw(wp[r] + (s + D) * step + c * 512);
#pragma unroll
          for (int m = 0; m < MT; ++m) ldb(b[d][m], xq[m] + (s + D) * 8);
        }
      }
    }
  }
}

__device__ __forceinline__ void mstore(const MArgs& A, int tok, int row, float v) {
  if (A.f32) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + row] = v;
  else reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
}

// out[t, row] = scale[row / 16] * sum_k x[t, k] W''[row, k], t < M <= 8 MT. grid = (N / (16 RT), S), block = 32 W.
template <int D, int MT, int RT>
__global__ void __launch_bounds__(512, 1) p1m_kernel(const __grid_constant__ MArgs A) {
  constexpr int TT = 8 * MT, RS = TT + 1, ROWS = 16 * RT;
  __shared__ float red[MaxW<MT, RT>::v][ROWS * RS];   // [warp][row * RS + token]
  __shared__ int s_last;
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = (int)(blockDim.x >> 5);
  const int rt0 = (int)blockIdx.x * RT, S = (int)gridDim.y, sidx = (int)blockIdx.y;
  const int c0 = (int)((long long)sidx * A.nsb_total / S), c1 = (int)((long long)(sidx + 1) * A.nsb_total / S);
  const int sb_lo = c0 + (int)((long long)warp * (c1 - c0) / nw);
  const int sb_hi = c0 + (int)((long long)(warp + 1) * (c1 - c0) / nw);
  float acc[RT][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) { acc[r][m][0] = 0.f; acc[r][m][1] = 0.f; acc[r][m][2] = 0.f; acc[r][m][3] = 0.f; }
  const int gw = ((int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x) * nw + warp;   // global warp (interleaved layout)
  if (A.probe) mstream<D, MT, RT, 3>(A, rt0, gw, lane, sb_lo, sb_hi - sb_lo, acc);
  else if (A.fmt[rt0]) mstream<D, MT, RT, 1>(A, rt0, gw, lane, sb_lo, sb_hi - sb_lo, acc);   // RAW group (CTA-uniform)
  else mstream<D, MT, RT, 0>(A, rt0, gw, lane, sb_lo, sb_hi - sb_lo, acc);
  {
    const int g = lane >> 2, t = lane & 3;
    float* rw = red[warp];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        rw[(16 * r + g) * RS + 8 * m + 2 * t] = acc[r][m][0];
        rw[(16 * r + g) * RS + 8 * m + 2 * t + 1] = acc[r][m][1];
        rw[(16 * r + g + 8) * RS + 8 * m + 2 * t] = acc[r][m][2];
        rw[(16 * r + g + 8) * RS + 8 * m + 2 * t + 1] = acc[r][m][3];
      }
  }
  __syncthreads();
  const int nout = ROWS * A.M;
  for (int idx = tid; idx < nout; idx += (int)blockDim.x) {
    const int rl = idx % ROWS, tok = idx / ROWS;
    float v = 0.f;
    for (int q = 0; q < nw; ++q) v += red[q][rl * RS + tok];
    v *= A.scale[rt0 + (rl >> 4)];
    const int row = rt0 * 16 + rl;
    if (S == 1) mstore(A, tok, row, v);
    else __stcg(A.ws + ((size_t)sidx * A.M + tok) * A.N + row, v);   // this slice's slot
  }
  if (S == 1) return;
  // split-K fixup: the last CTA of this row-tile group sums the S slots in rank order (bit-reproducible) and resets
  // the counter. No spin: the other CTAs just exit.
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(A.cnt + blockIdx.x, 1) == S - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int idx = tid; idx < nout; idx += (int)blockDim.x) {
    const int rl = idx % ROWS, tok = idx / ROWS, row = rt0 * 16 + rl;
    float sum = 0.f;
    for (int q = 0; q < S; ++q) sum += __ldcg(A.ws + ((size_t)q * A.M + tok) * A.N + row);
    mstore(A, tok, row, sum);
  }
  if (tid == 0) A.cnt[blockIdx.x] = 0;
}

template <int D, int MT, int RT>
static void launch(const MArgs& A, int warps, int S, int pdl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(A.N / (16 * RT)), (unsigned)S, 1);
  cfg.blockDim = dim3((unsigned)(warps * 32), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, p1m_kernel<D, MT, RT>, A));
}

template <int D, int MT, int RT>
static std::vector<int64_t> info_t(int warps) {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, p1m_kernel<D, MT, RT>));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p1m_kernel<D, MT, RT>, warps * 32, 0));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ};
}

// ---- Stream-K (persistent): G = gridDim.x * nw warps; warp gw owns the units [gw U / G, (gw + 1) U / G) of the
// (N / 16 RT) x nsb unit grid (unit = one SB of RT row tiles), crossing row-tile-group boundaries, so every warp streams the
// same weight bytes (+- 1 SB) and there is no tail wave. A group split over n > 1 warps: each writes its FP32 partial to slot
// k = gw - (first owner); the last to arrive (ticket) sums the n slots in k order (bit-reproducible) and stores the output.
struct KArgs { float* ws; int* cnt; int kmax; };

__device__ __forceinline__ int k_owner(long long X, long long U, int G) { return (int)(((X + 1) * G + U - 1) / U) - 1; }

template <int D, int MT, int RT>
__global__ void __launch_bounds__(512, 1) p1k_kernel(const __grid_constant__ MArgs A, const __grid_constant__ KArgs Kp) {
  constexpr int ROWS = 16 * RT;
  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5, nw = (int)(blockDim.x >> 5);
  const int G = (int)gridDim.x * nw, gw = (int)blockIdx.x * nw + warp;
  const int nsb = A.nsb_total, ng = A.N / ROWS, M = A.M;
  const long long U = (long long)ng * nsb;
  long long u = (long long)gw * U / G;
  const long long u1 = (long long)(gw + 1) * U / G;
  const int g8 = lane >> 2, t = lane & 3;
  while (u < u1) {
    const int grp = (int)(u / nsb), sb_lo = (int)(u - (long long)grp * nsb);
    const int len = (int)min((long long)(nsb - sb_lo), u1 - u);
    const int rt0 = grp * RT;
    float acc[RT][MT][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int m = 0; m < MT; ++m) { acc[r][m][0] = 0.f; acc[r][m][1] = 0.f; acc[r][m][2] = 0.f; acc[r][m][3] = 0.f; }
    if (A.probe) mstream<D, MT, RT, 3>(A, rt0, 0, lane, sb_lo, len, acc);
    else if (A.fmt[rt0]) mstream<D, MT, RT, 1>(A, rt0, 0, lane, sb_lo, len, acc);
    else mstream<D, MT, RT, 0>(A, rt0, 0, lane, sb_lo, len, acc);
    const int fw = k_owner((long long)grp * nsb, U, G), lw = k_owner((long long)(grp + 1) * nsb - 1, U, G);
    const int n = lw - fw + 1;
    if (n == 1) {
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const float sc = A.scale[rt0 + r];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int row = 16 * r + g8 + 8 * (e >> 1), tok = 8 * m + 2 * t + (e & 1);
            if (tok < M) mstore(A, tok, rt0 * 16 + row, acc[r][m][e] * sc);
          }
      }
    } else {
      float* slot = Kp.ws + ((size_t)grp * Kp.kmax + (gw - fw)) * (size_t)(ROWS * M);   // [row][tok], unscaled
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int row = 16 * r + g8 + 8 * (e >> 1), tok = 8 * m + 2 * t + (e & 1);
            if (tok < M) __stcg(slot + row * M + tok, acc[r][m][e]);
          }
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) last = atomicAdd(Kp.cnt + grp, 1) == n - 1;
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        __threadfence();
        const float* base = Kp.ws + (size_t)grp * Kp.kmax * (size_t)(ROWS * M);
        for (int idx = lane; idx < ROWS * M; idx += 32) {
          float s = 0.f;
          for (int k = 0; k < n; ++k) s += __ldcg(base + (size_t)k * (ROWS * M) + idx);
          const int row = idx / M, tok = idx - row * M;
          mstore(A, tok, rt0 * 16 + row, s * A.scale[rt0 + (row >> 4)]);   // scales are powers of two: exact
        }
        if (lane == 0) Kp.cnt[grp] = 0;
      }
    }
    u += len;
  }
}

template <int D, int MT, int RT>
static void launch_k(const MArgs& A, const KArgs& Kp, int warps, int ctas, int pdl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ctas, 1, 1);
  cfg.blockDim = dim3((unsigned)(warps * 32), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, p1k_kernel<D, MT, RT>, A, Kp));
}

template <int D, int MT, int RT>
static std::vector<int64_t> info_k(int warps) {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, p1k_kernel<D, MT, RT>));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p1k_kernel<D, MT, RT>, warps * 32, 0));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ};
}

}  // namespace p1m

#define P1M_DISPATCH(D_, M_, R_, CALL) if (depth == D_ && mt == M_ && rt == R_) { CALL; }

void p1m_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mt, int64_t rt,
              int64_t pdl, torch::Tensor ilbuf, int64_t il_step) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "p1m: x bf16 contiguous");
  TORCH_CHECK(mt >= 1 && M >= 1 && M <= 8 * mt && x.size(0) == M && x.size(1) == K, "p1m: 1 <= M <= 8 mt and x [M, K]");
  TORCH_CHECK(rt >= 1 && K % 64 == 0 && N % (16 * rt) == 0 && N >= 16 * rt, "p1m: K % 64, N % (16 rt)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "p1m: x align");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt && fmt.scalar_type() == at::kByte,
              "p1m: packed dtypes");
  const int nrt = (int)(N / 16), nsb = (int)(K / 64);
  TORCH_CHECK(toff.numel() >= nrt && scale.numel() >= nrt && eoff.numel() >= nrt + 1 && fmt.numel() >= nrt, "p1m: meta");
  const bool f32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK((f32 || out.scalar_type() == at::kBFloat16) && out.stride(1) == 1 && out.stride(0) == ldo && ldo >= N &&
              out.size(0) >= M && out.size(1) >= N, "p1m: out bf16 / fp32 [M, >= N] with row stride ldo");
  TORCH_CHECK((warps == 4 || warps == 8 || (warps == 16 && rt == 1)) && split >= 1 && split <= 8 && split * warps <= nsb,
              "p1m: warps in {4, 8} (16 at rt 1), 1 <= split <= 8, split * warps <= K / 64");
  if (split > 1)
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= split * M * N && cnt.scalar_type() == at::kInt &&
                cnt.numel() >= N / (16 * rt), "p1m: split-K workspace [S * M * N] fp32 + counters (zeroed)");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1m::MArgs A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = out.data_ptr(); A.ldo = (int)ldo; A.N = (int)N; A.nsb_total = nsb; A.f32 = f32 ? 1 : 0;
  A.ws = split > 1 ? ws.data_ptr<float>() : nullptr; A.cnt = split > 1 ? cnt.data_ptr<int>() : nullptr;
  A.il_step = il_step;
  A.probe = (int)((pdl >> 1) & 1);   // pdl bit 1: stream probe
  pdl &= 1;
  if (il_step > 0) {
    TORCH_CHECK(ilbuf.scalar_type() == at::kByte && nsb % (split * warps) == 0 && il_step == (int64_t)(N / 16) * split * warps * 1536 &&
                ilbuf.numel() >= (int64_t)(N / 16) * nsb * 1536, "p1m: interleaved layout built for this (N, split, warps)");
    A.pk = ilbuf.data_ptr<uint8_t>();
  }
  const int w = (int)warps, S = (int)split, pd = (int)pdl;
@DISPATCH@
  TORCH_CHECK(false, "p1m: no instantiation for this (depth, mt, rt)");
}

std::vector<int64_t> p1m_info(int64_t depth, int64_t mt, int64_t rt, int64_t warps) {
  const int w = (int)warps;
@INFO@
  return {0, 0, 0};
}

void p1k_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t ctas, int64_t depth, int64_t mt, int64_t rt,
              int64_t kmax, int64_t pdl) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "p1k: x bf16 contiguous");
  TORCH_CHECK(mt >= 1 && M >= 1 && M <= 8 * mt && x.size(0) == M && x.size(1) == K, "p1k: 1 <= M <= 8 mt and x [M, K]");
  TORCH_CHECK(rt >= 1 && K % 64 == 0 && N % (16 * rt) == 0 && N >= 16 * rt, "p1k: K % 64, N % (16 rt)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "p1k: x align");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt && fmt.scalar_type() == at::kByte,
              "p1k: packed dtypes");
  const int nrt = (int)(N / 16), nsb = (int)(K / 64), ng = (int)(N / (16 * rt));
  TORCH_CHECK(toff.numel() >= nrt && scale.numel() >= nrt && eoff.numel() >= nrt + 1 && fmt.numel() >= nrt, "p1k: meta");
  const bool f32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK((f32 || out.scalar_type() == at::kBFloat16) && out.stride(1) == 1 && out.stride(0) == ldo && ldo >= N &&
              out.size(0) >= M && out.size(1) >= N, "p1k: out bf16 / fp32 [M, >= N] with row stride ldo");
  TORCH_CHECK((warps == 4 || warps == 8 || warps == 16) && ctas >= 1 && ctas * warps <= (int64_t)ng * nsb,
              "p1k: warps in {4, 8, 16}, every warp needs >= 1 unit");
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= (int64_t)ng * kmax * 16 * rt * M && cnt.scalar_type() == at::kInt &&
              cnt.numel() >= ng && kmax >= 1, "p1k: workspace [groups * kmax * 16 rt * M] fp32 + counters [groups] (zeroed)");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1m::MArgs A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = out.data_ptr(); A.ldo = (int)ldo; A.N = (int)N; A.nsb_total = nsb; A.f32 = f32 ? 1 : 0;
  A.ws = nullptr; A.cnt = nullptr; A.il_step = 0;
  A.probe = (int)((pdl >> 1) & 1);   // pdl bit 1: stream probe
  pdl &= 1;
  p1m::KArgs Kp;
  Kp.ws = ws.data_ptr<float>(); Kp.cnt = cnt.data_ptr<int>(); Kp.kmax = (int)kmax;
  const int w = (int)warps, C = (int)ctas, pd = (int)pdl;
@KDISPATCH@
  TORCH_CHECK(false, "p1k: no instantiation for this (depth, mt, rt)");
}

std::vector<int64_t> p1k_info(int64_t depth, int64_t mt, int64_t rt, int64_t warps) {
  const int w = (int)warps;
@KINFO@
  return {0, 0, 0};
}
"""

_P1M_MOD = {}


def p1m_build(inst=P1M_INST, verbose=False):
    """One extension with P1's p1s (lab instantiations) + P1M's (depth, mt, rt) instantiations. Prepare-time only."""
    key = tuple(inst)
    if key in _P1M_MOD:
        return _P1M_MOD[key]
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a", "-DP1_LAYER_ONLY=1"]
    cpp = P1.P1_CPP_SRC + P1M_CPP_ADD
    cuda = P1.P1_CUDA_SRC + p1m_cuda_src(inst)
    tag = hashlib.sha1((cpp + cuda + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"p1m_mod_{tag}", cpp_sources=cpp, cuda_sources=cuda,
                      functions=["p1s_gemv", "p1s_info", "p1_unpack", "p1m_gemm", "p1m_info", "p1k_gemm", "p1k_info"],
                      extra_cuda_cflags=flags, verbose=verbose)
    print(f"[p1m] compiled {len(inst)} instantiations in {time.time() - t0:.1f}s", flush=True)
    _P1M_MOD[key] = mod
    return mod


def p1m_ws(P, split, M):
    """Split-K workspace / counters sized for (split, M) (prepare-time allocation)."""
    need = split * M * P["N"]
    if P.get("m_ws") is None or P["m_ws"].numel() < need:
        P["m_ws"] = torch.zeros(need, dtype=torch.float32, device=P["buf"].device)
    if P.get("m_cnt") is None:
        P["m_cnt"] = torch.zeros(P["nrt"], dtype=torch.int32, device=P["buf"].device)


def p1m_lane_maps(device):
    """P1's (row, k) per (lane, register o, half h) with k permuted inside the 64-k SB: logical 16 kt + 2 t + 8 a + h
    -> physical 16 t + 4 kt + 2 a + h (a = (o % 4) // 2). Lane t's B fragments are then x[16 t .. 16 t + 15]."""
    L = torch.arange(32, device=device).view(32, 1, 1)
    o = torch.arange(16, device=device).view(1, 16, 1)
    h = torch.arange(2, device=device).view(1, 1, 2)
    g, t, kt, i = L // 4, L % 4, o // 4, o % 4
    row = (g + 8 * (i % 2) + 0 * h).expand(32, 16, 2)
    kk = (16 * t + 4 * kt + 2 * (i // 2) + h).expand(32, 16, 2)
    return row, kk


P1M_KPERM = [0] * 64     # classic position (16 kt + 2 t + 8 a + h) -> physical k
for _kt in range(4):
    for _t in range(4):
        for _a in range(2):
            for _h in range(2):
                P1M_KPERM[16 * _kt + 2 * _t + 8 * _a + _h] = 16 * _t + 4 * _kt + 2 * _a + _h


def p1m_pack(w, name="w"):
    """P1's lossless pack with the k-permuted lane map (same bytes, same escapes, same formats)."""
    keep = P1.p1_lane_maps
    P1.p1_lane_maps = p1m_lane_maps
    try:
        P = P1.p1_pack(w, name=name)
    finally:
        P1.p1_lane_maps = keep
    P["kperm"] = True
    return P


def p1m_verify(mod, P, w):
    """Bit mismatches of P1's unpack kernel (classic positions) against W with the SB k-permutation applied. Must be 0."""
    u = P1.p1_unpack(mod, P)
    K = w.shape[1]
    perm = torch.tensor(P1M_KPERM, device=w.device)
    cols = (torch.arange(K // 64, device=w.device).view(-1, 1) * 64 + perm.view(1, -1)).reshape(-1)
    return int((u.view(torch.int16) != w[:, cols].view(torch.int16)).sum())


def p1m_interleave(P, warps, split, rt):
    """Grid-interleaved stream-order copy of an all-packed P1 pack for geometry (warps, split, rt) (notes/study/scout/raw/r1_c.md
    item 1): block (gw, r, s) = SB sb_lo(gw) + s of row tile rt0 + r goes to ((s * G + gw) * RT + r) * 1536, G = total warps, so
    at every step the whole grid reads one contiguous region. Prepare-time; None if the pack has RAW tiles or the SB split is
    uneven. Same SB bytes (the escape lists stay keyed by the logical SB)."""
    nrt, nsb = P["nrt"], P["nsb"]
    if any(P["fmt_cpu"]) or nsb % (split * warps) or nrt % rt:
        return None
    per = nsb // (split * warps)
    G = (nrt // rt) * split * warps
    src = P["buf"][: nrt * nsb * 1536].view(nrt, nsb, 1536)          # all packed: tile rt at rt * nsb * 1536 (toff)
    bx = torch.arange(nrt // rt, device=src.device)
    sx = torch.arange(split, device=src.device)
    w = torch.arange(warps, device=src.device)
    r = torch.arange(rt, device=src.device)
    st = torch.arange(per, device=src.device)
    # [step, sidx, bx, warp, r] -> (row tile, SB); gw = (sidx * nbx + bx) * warps + w (the kernel's order)
    rtile = (bx.view(1, 1, -1, 1, 1) * rt + r.view(1, 1, 1, 1, -1)).expand(per, split, nrt // rt, warps, rt)
    sb = ((sx.view(1, -1, 1, 1, 1) * warps + w.view(1, 1, 1, -1, 1)) * per + st.view(-1, 1, 1, 1, 1)).expand(per, split, nrt // rt, warps, rt)
    il = src[rtile.reshape(-1), sb.reshape(-1)].reshape(-1).contiguous()
    assert il.numel() == nrt * nsb * 1536 and G * rt * per * 1536 == il.numel()
    return {"buf": il, "step": G * rt * 1536, "key": (warps, split, rt)}


def p1m_call(mod, P, x, out, geo, n_rows=None, ldo=None, pdl=0, il=None):
    """out[:, :n_rows] = x @ W[:n_rows].T. geo = (warps, split, depth, mt, rt). out bf16 or fp32. il = p1m_interleave(...) for
    the same (warps, split, rt) -> stream-order weights."""
    warps, split, depth, mt, rt = geo
    n_rows = P["N"] if n_rows is None else n_rows
    ldo = out.stride(0) if ldo is None else ldo
    if split > 1:
        p1m_ws(P, split, x.shape[0])
    ws = P.get("m_ws") if split > 1 else P["ws"]
    cnt = P.get("m_cnt") if split > 1 else P["cnt"]
    if il is not None:
        assert il["key"] == (warps, split, rt) and n_rows == P["N"]
    mod.p1m_gemm(x, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], out, ws, cnt, x.shape[0], n_rows,
                 P["K"], ldo, warps, split, depth, mt, rt, pdl, il["buf"] if il is not None else P["buf"],
                 il["step"] if il is not None else 0)
    return out


def p1k_plan(P, warps, ctas, rt, M):
    """Stream-K plan for (warps, ctas, rt): kmax = the most warps any row-tile group is split over (sizes the workspace).
    Mirrors the kernel's owner(X) = ceil((X + 1) G / U) - 1 with units u = group * nsb + sb."""
    nsb, ng = P["nsb"], P["N"] // (16 * rt)
    G, U = ctas * warps, ng * nsb
    if G > U:
        return None

    def owner(X):
        return -(-((X + 1) * G) // U) - 1
    kmax = max(owner((g + 1) * nsb - 1) - owner(g * nsb) + 1 for g in range(ng))
    return {"warps": warps, "ctas": ctas, "rt": rt, "kmax": kmax, "G": G, "U": U}


def p1k_call(mod, P, x, out, geo, plan, pdl=0):
    """Stream-K launch: geo = (warps, ctas, depth, mt, rt); plan = p1k_plan(P, warps, ctas, rt, M). Workspace / counters
    live in P (allocated on first use for the largest plan; counters are re-zeroed by the kernel)."""
    warps, ctas, depth, mt, rt = geo
    M, N = x.shape[0], P["N"]
    ng = N // (16 * rt)
    need = ng * plan["kmax"] * 16 * rt * M
    if P.get("k_ws") is None or P["k_ws"].numel() < need:
        P["k_ws"] = torch.zeros(need, dtype=torch.float32, device=P["buf"].device)
    if P.get("k_cnt") is None:
        P["k_cnt"] = torch.zeros(P["nrt"], dtype=torch.int32, device=P["buf"].device)
    mod.p1k_gemm(x, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], out, P["k_ws"], P["k_cnt"], M, N,
                 P["K"], out.stride(0), warps, ctas, depth, mt, rt, plan["kmax"], pdl)
    return out


def p1m_esc_columns(esc):
    """Physical k (within the matrix) of each escape entry of a k-permuted pack (int32 [E, 2] meta / vals), for one-hot
    probes: (sb, lane L, register o, half h) -> 64 sb + 16 (L & 3) + 4 (o >> 2) + 2 ((o & 3) >> 1) + h."""
    e = esc[:, 0].long()
    sb, L, o, h = e & 0xFFFF, (e >> 16) & 31, (e >> 21) & 15, (e >> 25) & 1
    return 64 * sb + 16 * (L & 3) + 4 * (o >> 2) + 2 * ((o & 3) >> 1) + h

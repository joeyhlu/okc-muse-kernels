# p1 lab (round 2): P12 lossless BF16 weights + pure streaming decode/GEMV kernel for T <= 8 (LAB ONLY, never submit)
"""P1 lab, round 2: pack the layer's BF16 projection weights losslessly on the GPU, verify bit-exact unpack for every
projection, check the packed GEMV, and time {today's k3_v12 path, the streaming kernel on BF16, the streaming kernel on
P12} per (projection, T in {1, 8}) in ONE process (CUDA graph per call, 256 MB L2 flush before every replay, paired
alternating order). Prints a table (achieved TB/s, bytes in flight per SM) and a gate line.

    tools/box/box.sh run   'python3 tools/box/sass_gate.py kernels/muse_bf16_proj/p1_lab.py'   # compile gate
    tools/box/box.sh timed 'python3 kernels/muse_bf16_proj/p1_lab.py 2>&1 | tee box_out/p1_lab2.log'

Design: kernels/muse_bf16_proj/p1_design.md section R (round 2). Round 1's kernel (smem x stage + cluster split-K)
measured 0.7-2.2 TB/s (box_out/p1_lab.log) and is gone; the P12 format, packer and unpack are unchanged except that
escape entries are now sorted by (row tile, SB, k-tile) for the per-k-tile patch.

Format P12 (per 16-row tile, per 16x64 super-block SB = 1536 B = 3 x 512 B, A-fragment order of mma.m16n8k16):
  lane L, register o (0..15), half h:  g = L>>2, t = L&3, kt = o>>2, i = o&3,
      row = g + 8*(i&1),  k = 64*sb + 16*kt + 2*t + 8*(i>>1) + h
  chunk 0/1: byte plane (lane's 16 B = bytes 2*(o&7)+h of o = 0..7 / 8..15): byte = (n & 1) << 7 | mantissa7
  chunk 2  : code plane (lane's 4 words cw[kt]): bits n1, n2, n3, sign of (o, h) at ((8,9,10,15) + 16h + 4i) mod 32
  n = e - base[rt] (window of 16 exponents per row tile); decode to the SCALED bf16 (exponent 0x70 | n):
      a[o] = prmt(bw[o>>1], 0x38383838, o odd ? 0x4342 : 0x4140) | (rotr(cw[kt], 4*i) & 0x87008700)
  the GEMV result of a row tile is multiplied by scale[rt] = 2^(base - 112) (exact power-of-two scaling).
  escapes are coded 0 and patched in registers from a per-row-tile list (<= 32, int2 {sb | lane<<16 | o<<21 | h<<25,
  scaled16 | raw16<<16}, sorted by (sb, o>>2)); a 256-row group needing more is stored RAW (plain bf16, fragment order,
  2048 B per SB, scale 1). Unpack = decode, exponent back to base + n, escapes from raw16: bit-exact.

Kernel p1s_kernel<D, MODE> (T <= 8): CTA = one 16-row tile x a K slice (grid nrt x S); its W warps split the slice
into contiguous SB runs; per SB a lane issues 3 x ld.global.nc.L1::no_allocate.v4 (weights) + 8 x 4-B ld.global.nc
(its x B-fragment words, L1-cached, token g < T); D stages in flight per warp, each refilled right after use; per
k-tile: decode (4 PRMT + 3 SHF + 4 LOP3) -> mma.m16n8k16 (FP32 accumulate) -> per-k-tile escape key compare.
Epilogue: the W warp partials meet in 8 KB of smem (one barrier) -> x scale -> BF16 store (S = 1), or FP32
red.global.add into a self-cleaning workspace + last-CTA fixup (S > 1; counter, no spin). No smem x stage, no cluster,
no wgmma/TMA. MODE 0 = all tiles packed, 1 = all RAW (the uncompressed control), 2 = mixed (branch per CTA).
"""

import argparse
import hashlib
import importlib.util
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

P1_C = 0x70             # scaled-domain exponent base (multiple of 16; the hi-byte constant is 0x38)
P1_ESC_MAX = 32         # escapes per row tile in the in-register patch list (one per lane)
P1_RAW_GROUP = 16       # row tiles that share a format (256 rows)
P1_TS = (1, 8)          # timed: decode 1 (T = 1) and verify 1 (T = 8); T >= 16 stays on cuBLAS this round
P1_EXTRA_TS = (2, 5)    # correctness only; T = 9 must be refused by the host check
P1_MAXT = 8
REPS_SWEEP = 9
REPS_FINAL = 41
GATE = 1.15             # packed must be >= 15% faster (net of the graph replay floor) than today's path
N_SMS = 132

P1_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void p1s_gemv(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mode, int64_t pdl);
std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps);
void p1_unpack(torch::Tensor pk, torch::Tensor toff, torch::Tensor base, torch::Tensor fmt, torch::Tensor eoff,
               torch::Tensor esc, torch::Tensor out, int64_t N, int64_t K, int64_t use_esc);
"""

P1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

namespace p1 {

constexpr uint32_t KC4 = 0x38383838u;     // hi byte of every scaled bf16: exponent bits e4..e7 of 0x70
constexpr uint32_t CMASK = 0x87008700u;   // code bits per half: e1 e2 e3 (bits 8-10) and the sign (bit 15)
constexpr int SBP = 1536;                 // packed super-block bytes (16 rows x 64 k)
constexpr int SBR = 2048;                 // raw super-block bytes
constexpr int BIG = 0x7fffffff;
constexpr int MAXW = 16;                  // warps per CTA (<= 512 threads)

struct Args {
  const __nv_bfloat16* x; int M; int ldx;             // x [M, K] bf16 row-major, 1 <= M <= 8
  const uint8_t* pk; const int64_t* toff; const float* scale; const int* eoff; const int2* esc; const uint8_t* fmt;
  __nv_bfloat16* out; int ldo; int N; int nsb_total;  // out [M, ldo] bf16, columns [0, N)
  float* ws; int* cnt;                                 // split > 1: FP32 workspace [8][N] + counters [N/16], self-cleaning
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int r) { return __funnelshift_r(x, x, r); }

// k-tile kt of a packed SB (the lane's three 16-B chunks) -> the 4 A-fragment registers (scaled bf16x2)
__device__ __forceinline__ void dec_kt(const uint4 (&p)[3], const int kt, uint32_t (&a)[4]) {
  const uint4 bc = p[kt >> 1];
  const uint32_t w0 = (kt & 1) ? bc.z : bc.x;
  const uint32_t w1 = (kt & 1) ? bc.w : bc.y;
  const uint32_t cw = kt == 0 ? p[2].x : kt == 1 ? p[2].y : kt == 2 ? p[2].z : p[2].w;
  a[0] = __byte_perm(w0, KC4, 0x4140) | (cw & CMASK);
  a[1] = __byte_perm(w0, KC4, 0x4342) | (rotr(cw, 4) & CMASK);
  a[2] = __byte_perm(w1, KC4, 0x4140) | (rotr(cw, 8) & CMASK);
  a[3] = __byte_perm(w1, KC4, 0x4342) | (rotr(cw, 12) & CMASK);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// weights: streamed once, keep them out of L1
__device__ __forceinline__ uint4 ldw(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
// x B-fragment words: re-read by every row tile on the SM, so L1-cached; volatile keeps them issued with the weights
__device__ __forceinline__ uint32_t ldx(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}

// escape patch of k-tile `key` (= local SB * 4 + kt) into a[4]; entries sorted by key; warp-uniform loop
__device__ __forceinline__ int esc_key(uint32_t m, int sb_lo) { return ((int)(m & 0xFFFFu) - sb_lo) * 4 + (int)((m >> 23) & 3u); }
__device__ __forceinline__ void esc_patch4(uint32_t (&a)[4], int key, int sb_lo, int lane, uint32_t em, uint32_t ev,
                                           int& cur, int end, int& nkey) {
  while (cur < end) {
    const uint32_t m = __shfl_sync(0xffffffffu, em, cur);
    if (esc_key(m, sb_lo) != key) break;
    const uint32_t v = __shfl_sync(0xffffffffu, ev, cur);
    if (lane == (int)((m >> 16) & 31u)) {
      const int i = (int)((m >> 21) & 3u), h = (int)((m >> 25) & 1u);
      const uint32_t sc = v & 0xFFFFu;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (q == i) a[q] = h ? ((a[q] & 0x0000FFFFu) | (sc << 16)) : ((a[q] & 0xFFFF0000u) | sc);
    }
    ++cur;
  }
  const uint32_t mn = __shfl_sync(0xffffffffu, em, cur & 31);
  nkey = cur < end ? esc_key(mn, sb_lo) : BIG;
}

// One warp streams SBs [sb_lo, sb_lo + nsb) of row tile rt with D stages in flight, accumulating into acc (rows g, g+8;
// tokens 2t, 2t+1). RAW: plain bf16 fragments (4 x 16 B per lane per SB), no decode, no escapes.
template <int D, bool RAW>
__device__ __forceinline__ void stream(const Args& A, int rt, int lane, int sb_lo, int nsb, float (&acc)[4]) {
  constexpr int NCH = RAW ? 4 : 3;
  constexpr int SBB = RAW ? SBR : SBP;
  const int g = lane >> 2, t = lane & 3;
  const bool xon = g < A.M;
  const uint8_t* wp = A.pk + A.toff[rt] + (long long)sb_lo * SBB + lane * 16;
  // word t + 4j of the SB's 32 x-words of token g = b[j]: b[2kt] = x[g][16kt + 2t ..], b[2kt+1] = x[g][16kt + 2t + 8 ..]
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(A.x + (size_t)(xon ? g : 0) * A.ldx + (size_t)sb_lo * 64 + 2 * t);
  uint4 w[D][NCH];
  uint32_t b[D][8];
  // 1. weights of the first D SBs: prepare()-time data, issued before the PDL wait
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < nsb) {
#pragma unroll
      for (int c = 0; c < NCH; ++c) w[d][c] = ldw(wp + d * SBB + c * 512);
    }
  // 2. this warp's escape entries (one per lane; the in-range ones are contiguous because the list is key-sorted)
  uint32_t em = 0u, ev = 0u;
  int cur = 32, end = 0, nkey = BIG;
  if constexpr (!RAW) {
    const int o0 = A.eoff[rt], o1 = A.eoff[rt + 1];
    const int idx = o0 + lane;
    int2 e = make_int2(0, 0);
    if (idx < o1) e = A.esc[idx];
    em = (uint32_t)e.x;
    ev = (uint32_t)e.y;
    const int esb = (int)(em & 0xFFFFu);
    const bool in = idx < o1 && esb >= sb_lo && esb < sb_lo + nsb;
    const uint32_t msk = __ballot_sync(0xffffffffu, in);
    cur = msk ? __ffs(msk) - 1 : 32;
    end = msk ? 32 - __clz(msk) : 0;
    const uint32_t mf = __shfl_sync(0xffffffffu, em, cur & 31);
    nkey = msk ? esc_key(mf, sb_lo) : BIG;
  }
  // 3. PDL: x (and the output) belong to the predecessor kernel. A no-op without programmatic serialization.
  asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < nsb) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t v = ldx(xw + d * 32 + 4 * j);
        b[d][j] = xon ? v : 0u;
      }
    }
  // 4. rolling stream
  for (int s0 = 0; s0 < nsb; s0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int s = s0 + d;
      if (s < nsb) {
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
          uint32_t a[4];
          if constexpr (RAW) {
            a[0] = w[d][kt].x; a[1] = w[d][kt].y; a[2] = w[d][kt].z; a[3] = w[d][kt].w;
          } else {
            dec_kt(w[d], kt, a);
            const int key = s * 4 + kt;
            if (key == nkey) esc_patch4(a, key, sb_lo, lane, em, ev, cur, end, nkey);
          }
          mma16816(acc, a, b[d][2 * kt], b[d][2 * kt + 1]);
        }
        if (s + D < nsb) {
#pragma unroll
          for (int c = 0; c < NCH; ++c) w[d][c] = ldw(wp + (s + D) * SBB + c * 512);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t v = ldx(xw + (s + D) * 32 + 4 * j);
            b[d][j] = xon ? v : 0u;
          }
        }
      }
    }
  }
}

// D = 1: <= 64 registers (2 x 512 or 4 x 256 threads per SM); D = 2: <= 128 (no spills; the sweep picks W / S so the
// grid stays resident: e.g. qkv W = 8, S = 2 -> 288 CTAs at 3 x 256 threads per SM)
template <int D, int MODE> struct Bounds { static constexpr int minb = D == 1 ? 2 : 1; };

// out[t, rt*16 + r] = scale[rt] * sum_k x[t, k] W''[rt*16 + r, k], t < M <= 8. grid = (N / 16, S), block = 32 W.
// MODE 0: every row tile packed; 1: every row tile RAW; 2: mixed (CTA-uniform branch on fmt[rt]).
template <int D, int MODE>
__global__ void __launch_bounds__(512, Bounds<D, MODE>::minb) p1s_kernel(const __grid_constant__ Args A) {
  __shared__ float red[MAXW][128];      // [warp][row * 8 + token]
  __shared__ int s_last;
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = (int)(blockDim.x >> 5);
  const int rt = (int)blockIdx.x, S = (int)gridDim.y, sidx = (int)blockIdx.y;
  const int c0 = (int)((long long)sidx * A.nsb_total / S), c1 = (int)((long long)(sidx + 1) * A.nsb_total / S);
  const int sb_lo = c0 + (int)((long long)warp * (c1 - c0) / nw);
  const int sb_hi = c0 + (int)((long long)(warp + 1) * (c1 - c0) / nw);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (MODE == 0) {
    stream<D, false>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  } else if constexpr (MODE == 1) {
    stream<D, true>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  } else {
    if (A.fmt[rt]) stream<D, true>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
    else stream<D, false>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  }
  // warp partials -> smem -> one thread per (row, token)
  {
    const int g = lane >> 2, t = lane & 3;
    float* rw = red[warp];
    rw[g * 8 + 2 * t] = acc[0];
    rw[g * 8 + 2 * t + 1] = acc[1];
    rw[(g + 8) * 8 + 2 * t] = acc[2];
    rw[(g + 8) * 8 + 2 * t + 1] = acc[3];
  }
  __syncthreads();
  const int row = rt * 16 + (tid & 15), tok = tid >> 4;
  const bool mine = tid < 16 * A.M;
  if (mine) {
    float v = 0.f;
    for (int q = 0; q < nw; ++q) v += red[q][(tid & 15) * 8 + tok];
    v *= A.scale[rt];
    if (S == 1) A.out[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
    else atomicAdd(A.ws + (size_t)tok * A.N + row, v);       // RED.E.ADD.F32 (result unused)
  }
  if (S == 1) return;
  // split-K fixup: the last CTA of this row tile converts the FP32 sums, re-zeroes them and resets the counter
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(A.cnt + rt, 1) == S - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (mine) {
    float* p = A.ws + (size_t)tok * A.N + row;
    A.out[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(__ldcg(p));
    *p = 0.f;
  }
  if (tid == 0) A.cnt[rt] = 0;
}

// Bit-exact unpack: one warp per (SB, row tile) -> original bf16 W[N, K]. use_esc = 0 is the negative control.
__global__ void p1_unpack_kernel(const uint8_t* __restrict__ pk, const int64_t* __restrict__ toff,
                                 const int* __restrict__ base, const uint8_t* __restrict__ fmt,
                                 const int* __restrict__ eoff, const int2* __restrict__ esc,
                                 uint16_t* __restrict__ out, int K, int use_esc) {
  const int sb = (int)blockIdx.x, rt = (int)blockIdx.y, lane = (int)threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  uint32_t v[16];
  if (fmt[rt]) {
    const uint8_t* p = pk + toff[rt] + (long long)sb * SBR + lane * 16;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 q = ldw(p + c * 512);
      v[4 * c] = q.x; v[4 * c + 1] = q.y; v[4 * c + 2] = q.z; v[4 * c + 3] = q.w;
    }
  } else {
    const uint8_t* p = pk + toff[rt] + (long long)sb * SBP + lane * 16;
    uint4 q[3] = {ldw(p), ldw(p + 512), ldw(p + 1024)};
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      uint32_t t4[4];
      dec_kt(q, kt, t4);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        // scaled exponent 0x70 | n -> original base + n
        uint32_t wv = t4[i], o = 0u;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t x = (wv >> (16 * h)) & 0xFFFFu;
          const uint32_t n = (x >> 7) & 15u;
          const uint32_t y = (x & 0x807Fu) | (((uint32_t)(base[rt] + (int)n) & 0xFFu) << 7);
          o |= y << (16 * h);
        }
        v[4 * kt + i] = o;
      }
    }
    if (use_esc) {
      for (int j = eoff[rt]; j < eoff[rt + 1]; ++j) {
        const int2 e = esc[j];
        const uint32_t m = (uint32_t)e.x;
        if ((int)(m & 0xFFFFu) != sb || (int)((m >> 16) & 31u) != lane) continue;
        const int o = (int)((m >> 21) & 15u), h = (int)((m >> 25) & 1u);
        const uint32_t rw = ((uint32_t)e.y >> 16) & 0xFFFFu;
#pragma unroll
        for (int q2 = 0; q2 < 16; ++q2)
          if (q2 == o) v[q2] = h ? ((v[q2] & 0xFFFFu) | (rw << 16)) : ((v[q2] & 0xFFFF0000u) | rw);
      }
    }
  }
#pragma unroll
  for (int o = 0; o < 16; ++o) {
    const int kt = o >> 2, i = o & 3;
    const int row = rt * 16 + g + 8 * (i & 1);
    const int k = sb * 64 + 16 * kt + 2 * t + 8 * (i >> 1);
    *reinterpret_cast<uint32_t*>(out + (size_t)row * K + k) = v[o];   // (k, k+1) = (lo, hi)
  }
}

template <int D, int MODE>
static void launch(const Args& A, int warps, int S, int nrt, int pdl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)nrt, (unsigned)S, 1);
  cfg.blockDim = dim3((unsigned)(warps * 32), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, p1s_kernel<D, MODE>, A));
}

template <int D, int MODE>
static std::vector<int64_t> info_t(int warps) {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, p1s_kernel<D, MODE>));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p1s_kernel<D, MODE>, warps * 32, 0));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ};
}

}  // namespace p1

#define P1S_DISPATCH(D_, M_, CALL) if (depth == D_ && mode == M_) { CALL; }

void p1s_gemv(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mode, int64_t pdl) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "p1s: x bf16 contiguous");
  TORCH_CHECK(M >= 1 && M <= 8 && x.size(0) == M && x.size(1) == K, "p1s: 1 <= M <= 8 and x [M, K]");
  TORCH_CHECK(K % 64 == 0 && N % 16 == 0 && N >= 16, "p1s: K % 64, N % 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "p1s: x align");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt && fmt.scalar_type() == at::kByte,
              "p1s: packed dtypes");
  const int nrt = (int)(N / 16), nsb = (int)(K / 64);
  TORCH_CHECK(toff.numel() >= nrt && scale.numel() >= nrt && eoff.numel() >= nrt + 1 && fmt.numel() >= nrt, "p1s: meta");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 && out.stride(0) == ldo && ldo >= N &&
              out.size(0) >= M && out.size(1) >= N, "p1s: out bf16 [M, >= N] with row stride ldo");
  TORCH_CHECK((warps == 4 || warps == 8 || warps == 16) && (depth == 1 || depth == 2) && mode >= 0 && mode <= 2,
              "p1s: warps in {4, 8, 16}, depth in {1, 2}, mode in {0, 1, 2}");
  TORCH_CHECK(split >= 1 && split * warps <= nsb, "p1s: every warp needs >= 1 super-block (split * warps <= K / 64)");
  if (split > 1)
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= 8 * N && cnt.scalar_type() == at::kInt &&
                cnt.numel() >= nrt, "p1s: split-K workspace [8 * N] fp32 + counters [N / 16] int32 (zeroed)");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1::Args A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); A.ldo = (int)ldo; A.N = (int)N; A.nsb_total = nsb;
  A.ws = split > 1 ? ws.data_ptr<float>() : nullptr; A.cnt = split > 1 ? cnt.data_ptr<int>() : nullptr;
  const int w = (int)warps, S = (int)split, pd = (int)pdl;
  P1S_DISPATCH(1, 0, (p1::launch<1, 0>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(2, 0, (p1::launch<2, 0>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(1, 1, (p1::launch<1, 1>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(2, 1, (p1::launch<2, 1>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(1, 2, (p1::launch<1, 2>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(2, 2, (p1::launch<2, 2>(A, w, S, nrt, pd)); return)
  TORCH_CHECK(false, "p1s: no instantiation");
}

std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps) {
  const int w = (int)warps;
  P1S_DISPATCH(1, 0, return (p1::info_t<1, 0>(w)))
  P1S_DISPATCH(2, 0, return (p1::info_t<2, 0>(w)))
  P1S_DISPATCH(1, 1, return (p1::info_t<1, 1>(w)))
  P1S_DISPATCH(2, 1, return (p1::info_t<2, 1>(w)))
  P1S_DISPATCH(1, 2, return (p1::info_t<1, 2>(w)))
  P1S_DISPATCH(2, 2, return (p1::info_t<2, 2>(w)))
  return {0, 0, 0};
}

void p1_unpack(torch::Tensor pk, torch::Tensor toff, torch::Tensor base, torch::Tensor fmt, torch::Tensor eoff,
               torch::Tensor esc, torch::Tensor out, int64_t N, int64_t K, int64_t use_esc) {
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == N && out.size(1) == K, "unpack: out");
  TORCH_CHECK(K % 64 == 0 && N % 16 == 0, "unpack: shape");
  const c10::cuda::OptionalCUDAGuard guard(out.device());
  const dim3 grid((unsigned)(K / 64), (unsigned)(N / 16));
  p1::p1_unpack_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      pk.data_ptr<uint8_t>(), toff.data_ptr<int64_t>(), base.data_ptr<int>(), fmt.data_ptr<uint8_t>(),
      eoff.data_ptr<int>(), reinterpret_cast<const int2*>(esc.data_ptr<int>()),
      reinterpret_cast<uint16_t*>(out.data_ptr()), (int)K, (int)use_esc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


# ================================================================================================ build
def p1_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((P1_CPP_SRC + P1_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"p1s_lab_{tag}", cpp_sources=P1_CPP_SRC, cuda_sources=P1_CUDA_SRC,
        functions=["p1s_gemv", "p1s_info", "p1_unpack"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++17"],
        verbose=True)
    print(f"[p1] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


# ================================================================================================ packer (GPU)
def _lane_maps(device):
    L = torch.arange(32, device=device).view(32, 1, 1)
    o = torch.arange(16, device=device).view(1, 16, 1)
    h = torch.arange(2, device=device).view(1, 1, 2)
    g, t, kt, i = L // 4, L % 4, o // 4, o % 4
    row = (g + 8 * (i % 2) + 0 * h).expand(32, 16, 2)
    kk = (16 * kt + 2 * t + 8 * (i // 2) + h).expand(32, 16, 2)
    return row, kk


def _wrap32(v64):
    return torch.where(v64 >= 2 ** 31, v64 - 2 ** 32, v64).to(torch.int32)


def pack_p12(w, esc_max=P1_ESC_MAX, raw_group=P1_RAW_GROUP, name="w"):
    """Lossless P12 packing of a BF16 [N, K] CUDA tensor (prepare-time only). Returns the packed state dict."""
    t0 = time.time()
    assert w.dtype == torch.bfloat16 and w.is_cuda and w.dim() == 2
    w = w.contiguous()
    dev = w.device
    N, K = w.shape
    assert N % 16 == 0 and K % 64 == 0
    nrt, nsb = N // 16, K // 64
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    # per-row-tile base: the 16-exponent window [b, b+15] inside [1, 254] holding the most weights
    hist = torch.zeros((nrt, 256), dtype=torch.int32, device=dev)
    hist.scatter_add_(1, e.view(nrt, 16 * K).long(), torch.ones((nrt, 16 * K), dtype=torch.int32, device=dev))
    cs = torch.cat([torch.zeros((nrt, 1), dtype=torch.int64, device=dev), hist.long().cumsum(1)], 1)
    win = cs[:, 17:256] - cs[:, 1:240]                       # b = 1 .. 239
    base = (win.argmax(1) + 1).to(torch.int32)
    # fragment order [nrt, nsb, 32 lanes, 16 regs, 2 halves]
    row, kk = _lane_maps(dev)
    fr = bits.view(nrt, 16, nsb, 64)[:, row, :, kk].permute(3, 4, 0, 1, 2).contiguous()
    del bits, e
    s = (fr >> 15) & 1
    ee = (fr >> 7) & 0xFF
    m = fr & 0x7F
    n = ee - base.view(nrt, 1, 1, 1, 1)
    inw = (n >= 0) & (n <= 15)
    nn = torch.where(inw, n, torch.zeros_like(n))
    sv = torch.where(inw, s, torch.zeros_like(s))
    byte = torch.where(inw, ((nn & 1) << 7) | m, torch.zeros_like(m))
    bp = byte.view(nrt, nsb, 32, 2, 8, 2).permute(0, 1, 3, 2, 4, 5).reshape(nrt, nsb, 2, 32, 16).to(torch.uint8)
    ii = (torch.arange(4, device=dev).view(4, 1))
    hh = (torch.arange(2, device=dev).view(1, 2))
    cw = torch.zeros((nrt, nsb, 32, 4), dtype=torch.int64, device=dev)
    for f, tgt in (((nn >> 1) & 1, 8), ((nn >> 2) & 1, 9), ((nn >> 3) & 1, 10), (sv, 15)):
        pos = (tgt + 16 * hh + 4 * ii) % 32                    # [4 (i), 2 (h)]
        cw += (f.view(nrt, nsb, 32, 4, 4, 2).to(torch.int64) << pos).sum(dim=(-1, -2))
    cw8 = _wrap32(cw).contiguous().view(torch.uint8).view(nrt, nsb, 1, 32, 16)
    packed_t = torch.cat([bp, cw8], dim=2).reshape(nrt, nsb * 1536)
    del byte, bp, cw, cw8, nn, sv, s, m
    # raw form (fragment order, original bits)
    fr16 = torch.where(fr >= 32768, fr - 65536, fr).to(torch.int16)
    raw_t = fr16.view(nrt, nsb, 32, 4, 4, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(torch.uint8).reshape(nrt, nsb * 2048)
    # escapes in fragment order (rt, sb, L, o, h): row-major nonzero = sorted by (rt, sb)
    idx = (~inw).nonzero()
    # round 2: the kernel patches per k-tile, so each row tile's entries are sorted by (sb, kt = o >> 2)
    if idx.shape[0]:
        key = (idx[:, 0] * nsb + idx[:, 1]) * 4 + (idx[:, 3] >> 2)
        idx = idx[torch.argsort(key, stable=True)]
    rt_i, sb_i, L_i, o_i, h_i = idx.unbind(1)
    raw16 = fr[rt_i, sb_i, L_i, o_i, h_i]
    rv = torch.where(raw16 >= 32768, raw16 - 65536, raw16).to(torch.int16).view(torch.bfloat16).double()
    j = (P1_C - base[rt_i]).double()
    sd = rv * torch.pow(torch.full_like(j, 2.0), j)
    sb16 = sd.to(torch.bfloat16)
    nan = torch.isnan(rv)
    exact = nan | (sb16.double() == sd)
    sc_bits = sb16.view(torch.int16).to(torch.int32) & 0xFFFF
    sc_bits = torch.where(nan, torch.full_like(sc_bits, 0x7FC0), sc_bits)
    cnt = torch.bincount(rt_i, minlength=nrt)
    bad = torch.bincount(rt_i[~exact], minlength=nrt) > 0 if bool((~exact).any()) else torch.zeros(nrt, dtype=torch.bool, device=dev)
    need_raw = (cnt > esc_max) | bad
    ng = (nrt + raw_group - 1) // raw_group
    pad = torch.zeros(ng * raw_group, dtype=torch.bool, device=dev)
    pad[:nrt] = need_raw
    raw_tile = pad.view(ng, raw_group).any(1, keepdim=True).expand(ng, raw_group).reshape(-1)[:nrt]
    keep = ~raw_tile[rt_i]
    meta = (sb_i | (L_i << 16) | (o_i << 21) | (h_i << 25))[keep]
    vals = (sc_bits.long() | (raw16.long() << 16))[keep]
    esc = torch.stack([_wrap32(meta.long()), _wrap32(vals)], 1).contiguous()
    if esc.numel() == 0:
        esc = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    cnt_k = torch.bincount(rt_i[keep], minlength=nrt)
    eoff = torch.zeros(nrt + 1, dtype=torch.int32, device=dev)
    eoff[1:] = cnt_k.cumsum(0).to(torch.int32)
    base_f = torch.where(raw_tile, torch.full_like(base, P1_C), base)
    scale = torch.pow(2.0, (base_f - P1_C).double()).float()
    fmt = raw_tile.to(torch.uint8)
    sizes = torch.where(raw_tile, torch.full_like(base, nsb * 2048), torch.full_like(base, nsb * 1536)).long()
    toff = torch.zeros(nrt, dtype=torch.int64, device=dev)
    toff[1:] = sizes.cumsum(0)[:-1]
    total = int(sizes.sum())
    buf = torch.empty(total + 64, dtype=torch.uint8, device=dev)
    toff_c, raw_c = toff.cpu().tolist(), raw_tile.cpu().tolist()
    for r in range(nrt):
        src = raw_t[r] if raw_c[r] else packed_t[r]
        buf[toff_c[r]:toff_c[r] + src.numel()] = src
    n_esc_all = int(idx.shape[0])
    st = {"name": name, "N": N, "K": K, "nrt": nrt, "nsb": nsb, "buf": buf, "toff": toff, "scale": scale, "fmt": fmt,
          "fmt_cpu": raw_c, "ws": torch.zeros(8 * N, dtype=torch.float32, device=dev),
          "cnt": torch.zeros(nrt, dtype=torch.int32, device=dev),
          "eoff": eoff, "esc": esc, "base": base_f.contiguous(), "bytes": total + nrt * 17 + 8 * int(keep.sum()),
          "n_esc": int(keep.sum()), "n_esc_all": n_esc_all, "n_raw_tiles": int(raw_tile.sum()),
          "max_esc_tile": int(cnt.max()) if nrt else 0, "unscalable": int((~exact).sum()),
          "pack_s": time.time() - t0}
    return st


def describe(P):
    nw = P["N"] * P["K"]
    return (f"{P['name']} [{P['N']}x{P['K']}]: {P['bytes'] / 1e6:.2f} MB ({8 * P['bytes'] / nw:.3f} bits/w, "
            f"{P['bytes'] / (2 * nw):.4f} of BF16); escapes {P['n_esc_all']} ({P['n_esc_all'] / nw:.2e}/w, max/tile "
            f"{P['max_esc_tile']}, unscalable {P['unscalable']}); RAW tiles {P['n_raw_tiles']}/{P['nrt']}; "
            f"packed in {P['pack_s']:.2f}s")




def unpack(mod, P, use_esc=True):
    out = torch.empty((P["N"], P["K"]), dtype=torch.bfloat16, device=P["buf"].device)
    mod.p1_unpack(P["buf"], P["toff"], P["base"], P["fmt"], P["eoff"], P["esc"], out, P["N"], P["K"], 1 if use_esc else 0)
    return out



# ================================================================================================ weights
def make_weights(dev, seed):
    """The judge's generator (problems/muse_decode_layer/reference.py) when importable, else the same recipe here."""
    try:
        sys.path.insert(0, os.path.join(REPO, "problems", "muse_decode_layer"))
        import reference as R   # lab only (never in a submission)
        w = R.generate_weights(seed, device=str(dev))
        print(f"[p1] weights: reference.generate_weights(seed={seed})", flush=True)
        return {k: w[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
    except Exception as ex:  # noqa: BLE001
        print(f"[p1] reference.py unavailable ({ex!r}); using the same recipe locally", flush=True)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def dense(n, k):
        return (torch.randn((n, k), generator=gen, device=dev) / math.sqrt(k)).to(torch.bfloat16)
    mean = torch.randn((8192,), generator=torch.Generator().manual_seed(5354)).to(dev)
    rows = 0.045 * torch.randn((256, 8192), generator=gen, device=dev) + \
        0.07 * torch.randn((256, 1), generator=gen, device=dev) * mean[None, :] / math.sqrt(8192)
    return {"qkv_gate": dense(2304, 8192), "o": dense(8192, 1024), "pooler": dense(4096, 8192),
            "unpooler": dense(8192, 4096), "router": rows.contiguous()}


def stress_matrix(dev, n=512, k=1024, seed=99):
    """Held-out style: per-row log-normal scales, Student-t(2) entries, 1% zeros, subnormals, inf, NaN, outliers."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    sc = torch.exp2(-6.0 + 1.5 * torch.randn((n, 1), generator=gen, device=dev))
    z = torch.randn((n, k), generator=gen, device=dev)
    chi = (torch.randn((n, k), generator=gen, device=dev) ** 2 + torch.randn((n, k), generator=gen, device=dev) ** 2) / 2
    w = (sc * z / chi.clamp_min(1e-12).sqrt()).to(torch.bfloat16)
    w[torch.rand((n, k), generator=gen, device=dev) < 0.01] = 0
    bits = w.view(torch.int16)
    specials = torch.tensor([0x0001, -32767, 0x007F, 0x7F80, -128, 0x7FC0, -63, 0x7F7F, 0x0080, -32768], dtype=torch.int16,
                            device=dev)   # -128 = 0xFF80 (-inf), -63 = 0xFFC1 (NaN), -32767 = 0x8001, -32768 = -0
    pos = torch.randint(0, n * k, (200,), generator=gen, device=dev)
    bits.view(-1)[pos] = specials[torch.arange(200, device=dev) % specials.numel()]
    # one fully subnormal 16-row tile and one huge-outlier tile
    bits[32:48] = (bits[32:48] & -32768) | (bits[32:48] & 0x7F)
    w[64:80, :8] = 3.0e38
    return w


def stress_mild(dev, n=512, k=8192, seed=98):
    """Generator-like rows plus sparse specials (zeros, +-subnormals, +-inf, NaN) that stay within the patch budget,
    and one unscalable value (3e38) that forces its 256-row group to RAW: exercises the in-register patch path."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn((n, k), generator=gen, device=dev) / math.sqrt(k)).to(torch.bfloat16)
    bits = w.view(torch.int16)
    specials = torch.tensor([0x0000, -32768, 0x0001, -32767, 0x0003, 0x007F, 0x7F80, -128, 0x7FC0], dtype=torch.int16,
                            device=dev)
    pos = torch.randint(0, (n // 2) * k, (150,), generator=gen, device=dev)          # first 256-row group: patched
    bits.view(-1)[pos] = specials[torch.arange(150, device=dev) % specials.numel()]
    w[300, 5] = 3.0e38                                                                  # second group: RAW
    return w


# ================================================================================================ timing
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


def time_pair(ga, gb, flush, reps):
    """Paired alternating order, cold L2 (256 MB flush) before every replay. -> (median a, median b, median a/b)."""
    ta, tb, rr = [], [], []
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for r in range(reps):
        order = (0, 1) if r % 2 == 0 else (1, 0)
        t = [0.0, 0.0]
        for which in order:
            flush.add_(1.0)
            ev[0].record()
            (ga if which == 0 else gb).replay()
            ev[1].record()
            torch.cuda.synchronize()
            t[which] = ev[0].elapsed_time(ev[1]) * 1e3
        ta.append(t[0]); tb.append(t[1]); rr.append(t[0] / t[1])
    return statistics.median(ta), statistics.median(tb), statistics.median(rr)


def time_one(g, flush, reps):
    ts = []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        flush.add_(1.0)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)


# ================================================================================================ today's path
def load_r1b():
    try:
        spec = importlib.util.spec_from_file_location("r1b_v2_lab", os.path.join(HERE, "r1b_v2.py"))
        r1b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r1b)
        mod = r1b._build()
        dev = torch.device("cuda")
        st = {"mod": mod, "dummy": torch.zeros((4,), dtype=torch.float32, device=dev),
              "counters": torch.zeros((r1b.COUNTERS,), dtype=torch.int32, device=dev)}
        return r1b, st
    except Exception as ex:  # noqa: BLE001
        print(f"[p1] R1B unavailable -> T=1 'today' rows use cuBLAS (k3_v12 uses R1B there): {ex!r}", flush=True)
        return None, None


def bf16_gemm_fn():
    try:
        from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
        return lambda x, w: _bf16_gemm_dispatch_impl(x, w, None), "sglang _bf16_gemm_dispatch_impl"
    except Exception as ex:  # noqa: BLE001
        print(f"[p1] SGLang dispatch unavailable ({ex!r}); F.linear", flush=True)
        return (lambda x, w: F.linear(x, w)), "F.linear"




# ================================================================================================ calls (round 2)
def mode_of(P, n_rows):
    """0 = every row tile in [0, n_rows) packed, 1 = every one RAW, 2 = mixed (per-CTA branch)."""
    f = P["fmt_cpu"][: n_rows // 16]
    return 1 if all(f) else (0 if not any(f) else 2)


def p1s_call(mod, P, x, out, n_rows, ldo, geo, pdl=0):
    warps, split, depth = geo
    mod.p1s_gemv(x, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], out, P["ws"], P["cnt"],
                 x.shape[0], n_rows, P["K"], ldo, warps, split, depth, mode_of(P, n_rows), pdl)
    return out


def geometries(P, n_rows, min_ctas=1):
    """(warps per CTA, split-K CTAs per row tile, stages in flight per warp) accepted by the host checks."""
    nrt = n_rows // 16
    out = []
    for warps in (4, 8, 16):
        for split in (1, 2, 4, 8):
            if split * warps > P["nsb"] or nrt * split < min_ctas:
                continue
            for depth in (1, 2):
                out.append((warps, split, depth))
    return out


def residency(mod, P, n_rows, geo):
    """(regs/thread, local bytes, CTAs/SM by occupancy, average warps/SM, weight bytes in flight per SM, waves)."""
    warps, split, depth = geo
    mode = mode_of(P, n_rows)
    regs, local, occ = mod.p1s_info(depth, mode, warps)
    ctas = (n_rows // 16) * split
    waves = ctas / max(1, occ * N_SMS)
    avg_warps = min(occ * warps, ctas * warps / N_SMS)
    sbb = 2048 if mode == 1 else 1536
    sb_per_warp = P["nsb"] / (split * warps)
    return regs, local, occ, avg_warps, avg_warps * min(depth, sb_per_warp) * sbb, waves


# ================================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5353)
    ap.add_argument("--quick", action="store_true", help="fewer geometries / reps")
    args = ap.parse_args()
    reps_sweep, reps_final = (5, 15) if args.quick else (REPS_SWEEP, REPS_FINAL)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"[p1] torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)}", flush=True)
    t_all = time.time()
    mod = p1_build()
    for depth in (1, 2):
        for mode in (0, 1, 2):
            for warps in (4, 8, 16):
                regs, local, occ = mod.p1s_info(depth, mode, warps)
                print(f"[p1] p1s_kernel<D={depth}, MODE={mode}> warps {warps}: {regs} regs/thread, {local} B local, "
                      f"{occ} CTAs/SM ({occ * warps} warps/SM)", flush=True)
    W = make_weights(dev, args.seed)
    hi = W["router"].to(torch.bfloat16)
    lo = (W["router"] - hi.float()).to(torch.bfloat16)
    wcat = torch.cat((W["pooler"], hi, lo), 0).contiguous()             # k3_v12 "cat": [pooler; hi; lo]
    mats = {"qkv": W["qkv_gate"].contiguous(), "o": W["o"].contiguous(), "cat": wcat,
            "unpooler": W["unpooler"].contiguous()}

    # ---------------------------------------------------------------- 1. pack + bit-exact verification
    print("\n===== P1 PACK + VERIFY =====", flush=True)
    P, ok_all = {}, True
    for name, w in list(mats.items()) + [("stress", stress_matrix(dev)), ("stress_mild", stress_mild(dev))]:
        pk = pack_p12(w, name=name)
        u = unpack(mod, pk)
        torch.cuda.synchronize()
        mism = int((u.view(torch.int16) != w.view(torch.int16)).sum())
        neg = int((unpack(mod, pk, use_esc=False).view(torch.int16) != w.view(torch.int16)).sum())
        neg_ok = (neg > 0) if pk["n_esc"] > 0 else True
        # the per-k-tile patch needs every row tile's entries sorted by (sb, kt)
        e = pk["esc"][:, 0].long()
        keyv = (e & 0xFFFF) * 4 + ((e >> 23) & 3)
        eo = pk["eoff"].long().cpu().tolist()
        kc = keyv.cpu().tolist()
        sorted_ok = all(kc[j] <= kc[j + 1] for r in range(pk["nrt"]) for j in range(eo[r], eo[r + 1] - 1))
        ok = mism == 0 and neg_ok and sorted_ok
        ok_all &= ok
        print(f"[p1] {describe(pk)}", flush=True)
        print(f"[p1]   unpack mismatches {mism} (must be 0) | negative control without patches: {neg} mismatches "
              f"(must be > 0 when escapes exist) | entries key-sorted {sorted_ok} -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        P[name] = pk
        if name.startswith("stress"):
            mats[name] = w
    PR = {}   # the same kernel on uncompressed BF16 (every tile RAW): separates structure from compression
    for name in ("qkv", "o", "cat", "unpooler"):
        PR[name] = pack_p12(mats[name], esc_max=-1, name=name + "_raw")
        mism = int((unpack(mod, PR[name]).view(torch.int16) != mats[name].view(torch.int16)).sum())
        ok_all &= mism == 0 and PR[name]["n_raw_tiles"] == PR[name]["nrt"]
        print(f"[p1] {describe(PR[name])} | unpack mismatches {mism}", flush=True)
    if not ok_all:
        print("[p1] GATE FAIL: unpack is not bit-exact; stopping before timing", flush=True)
        return 1

    # ---------------------------------------------------------------- 2. GEMV correctness
    print("\n===== P1 GEMV CHECKS =====", flush=True)
    gen = torch.Generator(device=dev).manual_seed(7)

    def check_geos(pk, n_rows):
        geos = geometries(pk, n_rows)
        pick = [geos[0]]
        pick += [g for g in geos if g[1] > 1 and g[2] == 2][:1]       # split-K atomics + fixup path
        pick += [g for g in geos if g[1] == 1 and g[2] == 2][-1:]     # deepest S = 1 path
        return list(dict.fromkeys(pick))

    def one_hot_cols(pk, w, n_rows, cols, mb):
        """x = e_k rows: y[t, :] must equal W[:n_rows, k_t] exactly (one exact product per output). Rows with an
        inf/NaN weight are skipped (0 * inf = NaN in any GEMM); subnormals read back as 0 are counted apart
        (tensor-core input flush: the same on raw BF16 through cuBLAS)."""
        bad = flushed = 0
        fin = torch.isfinite(w[:n_rows].float()).all(1)
        for geo in check_geos(pk, n_rows):
            for i in range(0, len(cols), mb):
                cc = cols[i:i + mb]
                x = torch.zeros((len(cc), pk["K"]), dtype=torch.bfloat16, device=dev)
                x[torch.arange(len(cc), device=dev), torch.tensor(cc, device=dev)] = 1.0
                out = torch.empty((len(cc), n_rows), dtype=torch.bfloat16, device=dev)
                p1s_call(mod, pk, x, out, n_rows, n_rows, geo)
                ref = w[:n_rows, torch.tensor(cc, device=dev)].t().float()
                got = out.float()
                mis = (got != ref) & ~(torch.isnan(got) & torch.isnan(ref)) & fin[None, :]
                sub = mis & (got == 0) & (ref.abs() < 2.0 ** -126)
                flushed += int(sub.sum())
                bad += int((mis & ~sub).sum())
        return bad, flushed

    gemm_ok = True
    for name in ("qkv", "o", "cat", "unpooler", "stress", "stress_mild"):
        pk, w = P[name], mats[name]
        cols_esc = []
        if pk["n_esc"]:
            e = pk["esc"][:, 0].long()
            sb, L, o, h = e & 0xFFFF, (e >> 16) & 31, (e >> 21) & 15, (e >> 25) & 1
            kk = 64 * sb + 16 * (o >> 2) + 2 * (L & 3) + 8 * ((o & 3) >> 1) + h
            cols_esc = sorted(set(kk.tolist()))[:384]
        cols = cols_esc + torch.randint(0, pk["K"], (32,), generator=gen, device=dev).tolist()
        n_rows = pk["N"]
        bad8, fl8 = one_hot_cols(pk, w, n_rows, cols, 8)
        bad1, fl1 = one_hot_cols(pk, w, n_rows, cols[:6], 1)
        print(f"[p1] one-hot probe {name} (mode {mode_of(pk, n_rows)}): {len(cols)} columns ({len(cols_esc)} escape "
              f"columns) x {len(check_geos(pk, n_rows))} geometries: {bad8 + bad1} mismatches (must be 0); subnormals "
              f"flushed by the tensor core {fl8 + fl1}", flush=True)
        gemm_ok &= (bad8 + bad1) == 0
        if name.startswith("stress"):
            continue
        for var, pv in (("packed", pk), ("raw", PR[name])):
            for m in P1_TS + P1_EXTRA_TS:
                x = torch.randn((m, pv["K"]), generator=gen, device=dev).to(torch.bfloat16)
                ref = x.double() @ w[:n_rows].double().t()
                errs = []
                geos = geometries(pv, n_rows)
                for geo in geos[:: (3 if args.quick else 1)]:
                    out = torch.empty((m, n_rows), dtype=torch.bfloat16, device=dev)
                    p1s_call(mod, pv, x, out, n_rows, n_rows, geo)
                    out2 = out.clone()
                    p1s_call(mod, pv, x, out, n_rows, n_rows, geo)
                    rel = float((out.double() - ref).norm() / ref.norm())
                    errs.append((rel, bool(torch.equal(out, out2)), geo))
                worst = max(errs)
                good = worst[0] < 4e-3 and all(er[1] for er in errs)
                gemm_ok &= good
                if m in P1_TS or not good:
                    print(f"[p1] gemv {name} {var} M={m}: {len(errs)} geometries, worst rel-L2 vs fp64 {worst[0]:.2e} "
                          f"(tol 4e-3) at {worst[2]}, repeat bit-identical {all(er[1] for er in errs)} -> "
                          f"{'ok' if good else 'FAIL'}", flush=True)
    try:
        x9 = torch.randn((9, 8192), device=dev).to(torch.bfloat16)
        p1s_call(mod, P["qkv"], x9, torch.empty((9, 2304), dtype=torch.bfloat16, device=dev), 2304, 2304, (8, 1, 2))
        print("[p1] M=9 was accepted (expected a refusal -> current path)", flush=True)
        gemm_ok = False
    except Exception:  # noqa: BLE001
        print("[p1] M=9 refused by the host check (T > 8 stays on the current path): ok", flush=True)
    if not gemm_ok:
        print("[p1] GATE FAIL: GEMV checks failed; stopping before timing", flush=True)
        return 1

    # ---------------------------------------------------------------- 3. timing: today vs p1s on BF16 vs p1s packed
    print("\n===== P1 TIMING (CUDA graph, 256 MB flush per replay, paired alternating order) =====", flush=True)
    gemm, gemm_name = bf16_gemm_fn()
    r1b, r1b_st = load_r1b()
    flush = torch.empty((64 * 1024 * 1024,), dtype=torch.float32, device=dev)
    zbuf = torch.zeros((8, 8192), dtype=torch.bfloat16, device=dev)
    tiny = torch.zeros((1,), device=dev)
    floor_us = time_one(graph_of(lambda: tiny.add_(1.0)), flush, 21)
    print(f"[p1] graph replay floor (one tiny kernel): {floor_us:.2f} us; cuBLAS path: {gemm_name}", flush=True)

    def today(name, m, x):
        """(callable, label) = k3_v12's projection path at this T (T <= 8)."""
        if r1b is not None and m == 1 and name in ("qkv", "cat", "unpooler"):
            layer, n, w = {"qkv": ("qkv_gate", 2304, mats["qkv"]), "cat": ("pooler", 4096, W["pooler"]),
                           "unpooler": ("unpooler", 1024, mats["unpooler"])}[name]
            route = r1b.GEMV_ROUTES[(layer, 1)]
            return (lambda: r1b._gemv(r1b_st, x, w, n, route)), f"R1B GEMV {route}"
        if name == "qkv":
            return (lambda: gemm(x, mats["qkv"])), "cuBLAS"
        if name == "o":
            return (lambda: gemm(x, mats["o"])), "cuBLAS"
        if name == "cat":
            return (lambda: gemm(x, W["pooler"])), "cuBLAS pooler"
        ws_ = mats["unpooler"][:1024]
        return (lambda: torch.mm(x, ws_.t(), out=zbuf[:m, :1024])), "cuBLAS slice->zbuf"

    rows = []
    shapes = [("qkv", 8192, 2304), ("o", 1024, 8192), ("cat", 8192, 4096), ("unpooler", 4096, 1024)]
    for name, k, n_rows in shapes:
        for m in P1_TS:
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ldo = 8192 if name == "unpooler" else n_rows
            out = zbuf[:m] if name == "unpooler" else torch.empty((m, n_rows), dtype=torch.bfloat16, device=dev)
            fa, label = today(name, m, x)
            ga = graph_of(fa)
            bests = {}
            for var, pv in (("packed", P[name]), ("raw", PR[name])):
                geos = geometries(pv, n_rows, min_ctas=32)
                if args.quick:
                    geos = geos[::2]
                sweep, best = [], None
                for geo in geos:
                    try:
                        p1s_call(mod, pv, x, out, n_rows, ldo, geo)
                        gb = graph_of(lambda geo=geo, pv=pv: p1s_call(mod, pv, x, out, n_rows, ldo, geo))
                    except Exception as ex:  # noqa: BLE001
                        print(f"[p1]   {name} M={m} {var} geo {geo}: launch failed {ex!r}", flush=True)
                        continue
                    _, tb, _ = time_pair(ga, gb, flush, reps_sweep)
                    sweep.append((tb, geo))
                    if best is None or tb < best[0]:
                        best = (tb, geo, gb)
                sweep.sort()
                print(f"[p1] {name} M={m} {var} sweep (us, (warps,split,depth)): " +
                      ", ".join(f"{t:.1f} {g}" for t, g in sweep[:6]), flush=True)
                if best is not None:
                    bests[var] = best
            if "packed" not in bests:
                print(f"[p1] {name} M={m}: no packed geometry ran", flush=True)
                continue
            best = bests["packed"]
            t_a, t_b, ratio = time_pair(ga, best[2], flush, reps_final)
            net = (t_a - floor_us) / max(t_b - floor_us, 1e-3)
            t_r = c_ratio = float("nan")
            if "raw" in bests:
                t_r, _, c_ratio = time_pair(bests["raw"][2], best[2], flush, reps_final)
            mb = P[name]["bytes"] * n_rows / P[name]["N"] / 1e6
            regs, local, occ, wsm, infl, waves = residency(mod, P[name], n_rows, best[1])
            rows.append((name, m, label, t_a, t_r, t_b, ratio, net, c_ratio, best[1], regs, wsm, infl, waves, mb))
            print(f"[p1] {name} M={m}: today {label} {t_a:.2f} us | p1s BF16 {t_r:.2f} us | packed {t_b:.2f} us "
                  f"{best[1]} | paired x{ratio:.3f} | net x{net:.3f} | compression alone x{c_ratio:.3f} | "
                  f"{mb / max(t_b - floor_us, 1e-3):.2f} TB/s net | {regs} regs, {wsm:.1f} warps/SM, "
                  f"{infl / 1024:.0f} KB/SM in flight, {waves:.2f} waves", flush=True)

    # ---------------------------------------------------------------- 3b. PDL lever (needs an early trigger to matter)
    print("\n===== P1 PDL (producer x.copy_ -> p1s, paired: pdl off / pdl on) =====", flush=True)
    for (name, m, label, t_a, t_r, t_b, ratio, net, c_ratio, geo, *_rest) in rows:
        if m != 1:
            continue
        k, n_rows = {"qkv": (8192, 2304), "o": (1024, 8192), "cat": (8192, 4096), "unpooler": (4096, 1024)}[name]
        ldo = 8192 if name == "unpooler" else n_rows
        out = zbuf[:m] if name == "unpooler" else torch.empty((m, n_rows), dtype=torch.bfloat16, device=dev)
        xsrc = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        x = torch.empty_like(xsrc)
        try:
            g0 = graph_of(lambda: (x.copy_(xsrc), p1s_call(mod, P[name], x, out, n_rows, ldo, geo, 0)))
            g1 = graph_of(lambda: (x.copy_(xsrc), p1s_call(mod, P[name], x, out, n_rows, ldo, geo, 1)))
            t0_, t1_, r_ = time_pair(g0, g1, flush, reps_final)
            print(f"[p1] PDL {name} M={m}: off {t0_:.2f} us | on {t1_:.2f} us | x{r_:.3f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[p1] PDL {name} M={m}: failed {ex!r}", flush=True)

    # ---------------------------------------------------------------- 4. table + gate
    print("\n===== P1 SUMMARY (event us incl. graph replay; net = minus the one-kernel floor "
          f"{floor_us:.2f} us; TB/s = packed MB / net us) =====", flush=True)
    print(f"| proj | T | today's path | today us | p1s BF16 us | p1s packed us | paired x | net x | compression x | "
          f"geometry (W,S,D) | regs | warps/SM | KB/SM in flight | waves | packed MB | achieved TB/s | gate x{GATE} |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    gate_ok = True
    for (name, m, label, t_a, t_r, t_b, ratio, net, c_ratio, geo, regs, wsm, infl, waves, mb) in rows:
        g = net >= GATE
        gate_ok &= g
        print(f"| {name if name != 'cat' else 'pooler'} | {m} | {label} | {t_a:.2f} | {t_r:.2f} | {t_b:.2f} | "
              f"{ratio:.3f} | {net:.3f} | {c_ratio:.3f} | {geo} | {regs} | {wsm:.1f} | {infl / 1024:.0f} | "
              f"{waves:.2f} | {mb:.1f} | {mb / max(t_b - floor_us, 1e-3):.2f} | {'PASS' if g else 'fail'} |")
    n_shapes = len(rows)
    print(f"\n[p1] GATE (bit-exact unpack on all weights; packed >= x{GATE} net vs today on qkv, o, pooler, unpooler at "
          f"T in {P1_TS}): {'PASS' if (gate_ok and ok_all and n_shapes == 2 * len(shapes)) else 'FAIL'} "
          f"({sum(r[7] >= GATE for r in rows)}/{n_shapes} shapes pass); total {time.time() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

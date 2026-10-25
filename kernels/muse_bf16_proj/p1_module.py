# p1_module: P12 lossless 12-bit BF16 weights + p1s streaming GEMV for T <= 8 (integration module for decode_layer)
"""P1 integration module (round 3b = the measured round-2b kernel). Self-contained: torch + one load_inline build;
no os / sys / subprocess. Round 3's ring/2-accumulator/k-permuted kernel was slower (111 regs, spills) and is gone.

Prepare time (untimed, deterministic, on the GPU):
    st = p1_prepare({"qkv": qkv_gate, "o": o, "pooler": pooler, "unpooler": unpooler})
    -> packs each BF16 [N, K] weight into P12 (12 bits/weight, lossless), checks the unpack is BIT-EXACT against the
       original (any mismatch drops that projection to the caller's path), allocates the split-K workspace/counters
       (self-resetting), and fixes the route table. Raw BF16 weights stay with the caller for T > 8 / fallbacks.
Call time (CUDA-graph safe: no allocation when `out` is given, no host sync, branches on shapes only):
    geo = p1_route(st, "qkv", T)                     # None -> keep today's path (route table / shape not covered)
    p1_gemv(st, "qkv", x, out)                       # x [T, K] bf16 contiguous (T <= 8), out [T, N] bf16
    p1_gemv(st, "unpooler", x, zbuf[:T], n_rows=1024, ldo=8192)   # T <= 8 slice -> columns 0..1023 of the zero buffer

Format P12 (round-2b lane map, the mma.m16n8k16 A-fragment order): 16-row tile x 64-k super-block (SB) = 1536 B
(3 x 512 B), lane L = (g, t) = (L >> 2, L & 3), A-fragment register o = 4 kt + i, half h:
    row = g + 8 (i & 1),   k = 64 sb + 16 kt + 2 t + 8 (i >> 1) + h
    byte plane (chunks 0/1): byte 2 (o & 7) + h = (n & 1) << 7 | mantissa7;  code plane (chunk 2): word kt holds
    n1, n2, n3, sign of (o, h) at ((8, 9, 10, 15) + 16 h + 4 i) mod 32;  n = e - base[rt] in [0, 15]
    decode: a[o] = prmt(bw[o >> 1], 0x38383838, o odd ? 0x4342 : 0x4140) | (rotr(cw[kt], 4 i) & 0x87008700)
    (scaled bf16, exponent 0x70 | n); the tile's result x scale[rt] = 2^(base - 112), exact.
    escapes: per-row-tile list (<= 32, sorted by (sb, kt)) patched in registers; 256-row groups needing more -> RAW.

Kernel p1s_kernel<1, MODE> (round 2b, 56 regs, 0 spills): CTA = one 16-row tile x a K slice (grid nrt x S); W warps
split the slice into contiguous SB runs; per SB a lane issues 3 x 16-B weight loads (no L1 allocate) + 8 x 4-B x
loads (L1-cached, token g < T); per k-tile decode -> mma -> escape key compare; the stage is refilled after use; W
partials meet in 8 KB smem; S > 1: each slice stores its FP32 partial in its own slot, the last CTA (counter, no spin)
sums the slots in rank order (bit-reproducible). MODE 0 packed (the only one the layer builds: -DP1_LAYER_ONLY=1),
1 RAW (BF16 control), 2 mixed (per-CTA branch), 3 stream probe (MODE 0's loads, XOR consume: the ceiling).
"""

import hashlib
import time

import torch

P1_C = 0x70             # scaled-domain exponent base (multiple of 16; the hi-byte constant is 0x38)
P1_ESC_MAX = 32         # escapes per row tile in the in-register patch list (one per lane)
P1_RAW_GROUP = 16       # row tiles that share a format (256 rows)
P1_MAXT = 8             # T <= 8 (one n8 token group)

# Hard-coded geometry (warps per CTA, split-K CTAs per row tile, ring depth) per (projection, T bucket), bucket 1 = T 1,
# bucket 8 = T 2..8: the round-2b sweep winners (box_out/p1_lab2b.log). No runtime autotune.
P1_GEOMETRY = {
    ("qkv", 1): (16, 1, 1), ("qkv", 8): (4, 8, 1),
    ("o", 1): (8, 1, 1), ("o", 8): (8, 1, 1),
    ("pooler", 1): (16, 1, 1), ("pooler", 8): (16, 1, 1),
    ("unpooler", 1): (16, 1, 1), ("unpooler", 8): (16, 1, 1),
}
# Route: True = P1 (measured >= 1.03x net vs today's path, round 2b, same process), False = keep today's path:
# qkv 1.066 / 1.237, o 1.108 / 1.000, pooler 1.143 / 1.481, unpooler slice 0.856 / 1.086 (T = 1 / T = 8).
P1_ROUTE = {
    ("qkv", 1): True, ("qkv", 8): True,
    ("o", 1): True, ("o", 8): False,
    ("pooler", 1): True, ("pooler", 8): True,
    ("unpooler", 1): False, ("unpooler", 8): True,
}
P1_ROUTE_MIN = 1.03
# rows used at T <= 8 (the unpooler runs this GPU's 1/8 slice up to 8 tokens)
P1_ROWS_T8 = {"unpooler": 1024}

P1_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void p1s_gemv(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mode, int64_t pdl,
              int64_t red_mode);
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

#ifndef P1_LAYER_ONLY
#define P1_LAYER_ONLY 0
#endif

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
  float* ws; int* cnt;                                 // split > 1: FP32 workspace [9][8][N] + counters [N/16]
  int red_mode;                                        // 0: deterministic slots [s][tok][N], summed q = 0..S-1 by the
                                                       //    last CTA; 1: FP32 atomics into slot 8 (round 2's
                                                       //    order-dependent reduction, kept as a negative control)
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
// tokens 2t, 2t+1). KM 0: packed P12; 1: RAW plain bf16 fragments (4 x 16 B per lane per SB, no decode, no escapes);
// 3: stream probe (the packed loads + x loads of KM 0, consumed by XOR: no decode, no mma -> the ceiling at this grid).
template <int D, int KM>
__device__ __forceinline__ void stream(const Args& A, int rt, int lane, int sb_lo, int nsb, float (&acc)[4]) {
  constexpr bool RAW = KM == 1;
  constexpr bool PROBE = KM == 3;
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
  if constexpr (KM == 0) {
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
  uint32_t px = 0u;
  for (int s0 = 0; s0 < nsb; s0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int s = s0 + d;
      if (s < nsb) {
        if constexpr (PROBE) {
#pragma unroll
          for (int c = 0; c < NCH; ++c) px ^= w[d][c].x ^ w[d][c].y ^ w[d][c].z ^ w[d][c].w;
#pragma unroll
          for (int j = 0; j < 8; ++j) px ^= b[d][j];
        } else {
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
  if constexpr (PROBE) acc[0] = __uint_as_float(px & 0x007FFFFFu);   // finite, deterministic, keeps the loads live
}

// D = 1: <= 64 registers (2 x 512 or 4 x 256 threads per SM); D = 2: <= 128 (no spills; the sweep picks W / S so the
// grid stays resident: e.g. qkv W = 8, S = 2 -> 288 CTAs at 3 x 256 threads per SM)
template <int D, int MODE> struct Bounds { static constexpr int minb = D == 1 ? 2 : 1; };

// out[t, rt*16 + r] = scale[rt] * sum_k x[t, k] W''[rt*16 + r, k], t < M <= 8. grid = (N / 16, S), block = 32 W.
// MODE 0: every row tile packed; 1: every row tile RAW; 2: mixed (CTA-uniform branch on fmt[rt]); 3: stream probe.
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
    stream<D, 0>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  } else if constexpr (MODE == 1) {
    stream<D, 1>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  } else if constexpr (MODE == 3) {
    stream<D, 3>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  } else {
    if (A.fmt[rt]) stream<D, 1>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
    else stream<D, 0>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
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
    else if (A.red_mode == 0) __stcg(A.ws + ((size_t)sidx * 8 + tok) * A.N + row, v);   // this slice's slot
    else atomicAdd(A.ws + ((size_t)64 + tok) * A.N + row, v);   // control: order-dependent FP32 sum (slot 8)
  }
  if (S == 1) return;
  // split-K fixup: the last CTA of this row tile sums the S slots in rank order (bit-reproducible; nothing to
  // re-zero) and resets the counter. No spin: the other CTAs just exit.
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(A.cnt + rt, 1) == S - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (mine) {
    float sum = 0.f;
    if (A.red_mode == 0) {
      for (int q = 0; q < S; ++q) sum += __ldcg(A.ws + ((size_t)q * 8 + tok) * A.N + row);
    } else {
      float* p = A.ws + ((size_t)64 + tok) * A.N + row;
      sum = __ldcg(p);
      *p = 0.f;                                                 // the atomic slot must start at zero next call
    }
    A.out[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(sum);
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
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mode, int64_t pdl,
              int64_t red_mode) {
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
  TORCH_CHECK((warps == 4 || warps == 8 || warps == 16) && depth == 1 && mode >= 0 && mode <= 3,
              "p1s: warps in {4, 8, 16}, depth 1 (the round-2b winners), mode in 0..3");
  TORCH_CHECK(split >= 1 && split <= 8 && split * warps <= nsb,
              "p1s: 1 <= split <= 8 and every warp needs >= 1 super-block (split * warps <= K / 64)");
  TORCH_CHECK(red_mode == 0 || red_mode == 1, "p1s: red_mode 0 (deterministic) or 1 (atomic control)");
  if (split > 1)
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= 72 * N && cnt.scalar_type() == at::kInt &&
                cnt.numel() >= nrt, "p1s: split-K workspace [9 * 8 * N] fp32 (slot 8 zeroed) + counters [N / 16] int32 (zeroed)");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1::Args A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); A.ldo = (int)ldo; A.N = (int)N; A.nsb_total = nsb;
  A.ws = split > 1 ? ws.data_ptr<float>() : nullptr; A.cnt = split > 1 ? cnt.data_ptr<int>() : nullptr;
  A.red_mode = (int)red_mode;
  const int w = (int)warps, S = (int)split, pd = (int)pdl;
  P1S_DISPATCH(1, 0, (p1::launch<1, 0>(A, w, S, nrt, pd)); return)
#if !P1_LAYER_ONLY
  P1S_DISPATCH(1, 1, (p1::launch<1, 1>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(1, 2, (p1::launch<1, 2>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(1, 3, (p1::launch<1, 3>(A, w, S, nrt, pd)); return)
#endif
  TORCH_CHECK(false, "p1s: no instantiation");
}

std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps) {
  const int w = (int)warps;
  P1S_DISPATCH(1, 0, return (p1::info_t<1, 0>(w)))
#if !P1_LAYER_ONLY
  P1S_DISPATCH(1, 1, return (p1::info_t<1, 1>(w)))
  P1S_DISPATCH(1, 2, return (p1::info_t<1, 2>(w)))
  P1S_DISPATCH(1, 3, return (p1::info_t<1, 3>(w)))
#endif
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

_P1_MOD = None


def p1_build(verbose=False, layer_only=False):
    """Compile once per process (load_inline caches by source hash). Never called inside kernel_fn.
    layer_only: instantiate only p1s_kernel<1, 0> (the routed path) - what the decode_layer file builds."""
    global _P1_MOD
    if _P1_MOD is not None:
        return _P1_MOD
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a",
             f"-DP1_LAYER_ONLY={1 if layer_only else 0}"]
    tag = hashlib.sha1((P1_CPP_SRC + P1_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    _P1_MOD = load_inline(
        name=f"p1s_mod_{tag}", cpp_sources=P1_CPP_SRC, cuda_sources=P1_CUDA_SRC,
        functions=["p1s_gemv", "p1s_info", "p1_unpack"], extra_cuda_cflags=flags, verbose=verbose)
    print(f"[p1] compiled in {time.time() - t0:.1f}s", flush=True)
    return _P1_MOD


# ================================================================================================ packer (GPU)
def p1_lane_maps(device):
    """(row_local, k_local) of (lane, register o, half h) inside one 16 x 64 SB (round-2b A-fragment map)."""
    L = torch.arange(32, device=device).view(32, 1, 1)
    o = torch.arange(16, device=device).view(1, 16, 1)
    h = torch.arange(2, device=device).view(1, 1, 2)
    g, t, kt, i = L // 4, L % 4, o // 4, o % 4
    row = (g + 8 * (i % 2) + 0 * h).expand(32, 16, 2)
    kk = (16 * kt + 2 * t + 8 * (i // 2) + h).expand(32, 16, 2)
    return row, kk


def p1_esc_columns(esc):
    """k (within the matrix) of each escape entry (int32 [E, 2] meta/vals), for the one-hot probes."""
    e = esc[:, 0].long()
    sb, L, o, h = e & 0xFFFF, (e >> 16) & 31, (e >> 21) & 15, (e >> 25) & 1
    return 64 * sb + 16 * (o >> 2) + 2 * (L & 3) + 8 * ((o & 3) >> 1) + h


def _wrap32(v64):
    return torch.where(v64 >= 2 ** 31, v64 - 2 ** 32, v64).to(torch.int32)


def p1_pack(w, esc_max=P1_ESC_MAX, raw_group=P1_RAW_GROUP, name="w"):
    """Lossless P12 packing of a BF16 [N, K] CUDA tensor (prepare-time only, deterministic)."""
    t0 = time.time()
    assert w.dtype == torch.bfloat16 and w.is_cuda and w.dim() == 2
    w = w.contiguous()
    dev = w.device
    N, K = w.shape
    assert N % 16 == 0 and K % 64 == 0
    nrt, nsb = N // 16, K // 64
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    hist = torch.zeros((nrt, 256), dtype=torch.int32, device=dev)
    hist.scatter_add_(1, e.view(nrt, 16 * K).long(), torch.ones((nrt, 16 * K), dtype=torch.int32, device=dev))
    cs = torch.cat([torch.zeros((nrt, 1), dtype=torch.int64, device=dev), hist.long().cumsum(1)], 1)
    win = cs[:, 17:256] - cs[:, 1:240]                       # windows [b, b + 15], b = 1 .. 239
    base = (win.argmax(1) + 1).to(torch.int32)
    row, kk = p1_lane_maps(dev)
    fr = bits.view(nrt, 16, nsb, 64)[:, row, :, kk].permute(3, 4, 0, 1, 2).contiguous()   # [nrt, nsb, 32, 16, 2]
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
    ii = torch.arange(4, device=dev).view(4, 1)
    hh = torch.arange(2, device=dev).view(1, 2)
    cw = torch.zeros((nrt, nsb, 32, 4), dtype=torch.int64, device=dev)
    for f, tgt in (((nn >> 1) & 1, 8), ((nn >> 2) & 1, 9), ((nn >> 3) & 1, 10), (sv, 15)):
        pos = (tgt + 16 * hh + 4 * ii) % 32
        cw += (f.view(nrt, nsb, 32, 4, 4, 2).to(torch.int64) << pos).sum(dim=(-1, -2))
    cw8 = _wrap32(cw).contiguous().view(torch.uint8).view(nrt, nsb, 1, 32, 16)
    packed_t = torch.cat([bp, cw8], dim=2).reshape(nrt, nsb * 1536)
    del byte, bp, cw, cw8, nn, sv, s, m
    fr16 = torch.where(fr >= 32768, fr - 65536, fr).to(torch.int16)
    raw_t = fr16.view(nrt, nsb, 32, 4, 4, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(torch.uint8).reshape(nrt, nsb * 2048)
    idx = (~inw).nonzero()
    if idx.shape[0]:                                          # per-k-tile patch order: (rt, sb, kt = o >> 2)
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
    bad = (torch.bincount(rt_i[~exact], minlength=nrt) > 0 if bool((~exact).any())
           else torch.zeros(nrt, dtype=torch.bool, device=dev))
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
    return {"name": name, "N": N, "K": K, "nrt": nrt, "nsb": nsb, "buf": buf, "toff": toff, "scale": scale, "fmt": fmt,
            "fmt_cpu": raw_c, "ws": torch.zeros(72 * N, dtype=torch.float32, device=dev),
            "cnt": torch.zeros(nrt, dtype=torch.int32, device=dev),
            "eoff": eoff, "esc": esc, "base": base_f.contiguous(), "bytes": total + nrt * 17 + 8 * int(keep.sum()),
            "n_esc": int(keep.sum()), "n_esc_all": int(idx.shape[0]), "n_raw_tiles": int(raw_tile.sum()),
            "max_esc_tile": int(cnt.max()) if nrt else 0, "unscalable": int((~exact).sum()),
            "pack_s": time.time() - t0}


def p1_describe(P):
    nw = P["N"] * P["K"]
    return (f"{P['name']} [{P['N']}x{P['K']}]: {P['bytes'] / 1e6:.2f} MB ({8 * P['bytes'] / nw:.3f} bits/w, "
            f"{P['bytes'] / (2 * nw):.4f} of BF16); escapes {P['n_esc_all']} ({P['n_esc_all'] / nw:.2e}/w, max/tile "
            f"{P['max_esc_tile']}, unscalable {P['unscalable']}); RAW tiles {P['n_raw_tiles']}/{P['nrt']}; "
            f"packed in {P['pack_s']:.2f}s")


def p1_unpack(mod, P, use_esc=True):
    out = torch.empty((P["N"], P["K"]), dtype=torch.bfloat16, device=P["buf"].device)
    mod.p1_unpack(P["buf"], P["toff"], P["base"], P["fmt"], P["eoff"], P["esc"], out, P["N"], P["K"], 1 if use_esc else 0)
    return out


def p1_mode_of(P, n_rows):
    """0 = every row tile in [0, n_rows) packed, 1 = every one RAW, 2 = mixed (per-CTA branch)."""
    f = P["fmt_cpu"][: n_rows // 16]
    return 1 if all(f) else (0 if not any(f) else 2)


def p1_call(mod, P, x, out, n_rows, ldo, geo, pdl=0, red_mode=0, mode=None):
    """Low-level launch. geo = (warps, split, depth). mode None = from the packed formats; 3 = stream probe."""
    warps, split, depth = geo
    if mode is None:
        mode = p1_mode_of(P, n_rows)
    mod.p1s_gemv(x, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], out, P["ws"], P["cnt"],
                 x.shape[0], n_rows, P["K"], ldo, warps, split, depth, mode, pdl, red_mode)
    return out


def p1_verify(mod, P, w):
    """(bit mismatches of the unpack, mismatches with patches disabled). The first must be 0."""
    u = p1_unpack(mod, P)
    mism = int((u.view(torch.int16) != w.view(torch.int16)).sum())
    neg = int((p1_unpack(mod, P, use_esc=False).view(torch.int16) != w.view(torch.int16)).sum())
    return mism, neg


# ================================================================================================ layer API
def p1_bucket(T):
    return 1 if T == 1 else 8


def p1_prepare(weights, geometry=None, route=None, build=True, layer_only=True):
    """weights: {"qkv": [2304, 8192], "o": [8192, 1024], "pooler": [4096, 8192], "unpooler": [8192, 4096]} BF16 CUDA.
    Returns the P1 state (or None if the build fails). A projection whose pack is not bit-exact is left out (its
    calls route to the caller's path). Prepare-time only: allocates, compiles, synchronizes."""
    try:
        mod = p1_build(layer_only=layer_only) if build else _P1_MOD
    except Exception as ex:  # noqa: BLE001
        print(f"[p1] build FAILED -> today's path everywhere: {ex!r}", flush=True)
        return None
    st = {"mod": mod, "geo": dict(geometry or P1_GEOMETRY), "route": dict(route or P1_ROUTE), "proj": {}}
    for name, w in weights.items():
        try:
            P = p1_pack(w, name=name)
            mism, neg = p1_verify(mod, P, w)
            torch.cuda.synchronize()
            ok = mism == 0 and (neg > 0 or P["n_esc"] == 0)
            print(f"[p1] {p1_describe(P)} | unpack mismatches {mism}, without patches {neg} -> "
                  f"{'ok' if ok else 'DROPPED'}", flush=True)
            if ok:
                for b in (1, 8):          # every routed geometry must pass the host checks for its row count
                    geo = st["geo"].get((name, b))
                    n_rows = P1_ROWS_T8.get(name, P["N"])
                    if geo is not None and not (geo[1] * geo[0] <= P["nsb"] and 1 <= geo[1] <= 8):
                        st["route"][(name, b)] = False
                P["rows_t8"] = P1_ROWS_T8.get(name, P["N"])
                P["mode_t8"] = p1_mode_of(P, P["rows_t8"])
                st["proj"][name] = P
        except Exception as ex:  # noqa: BLE001
            print(f"[p1] {name}: pack/verify FAILED -> today's path: {ex!r}", flush=True)
    return st


def p1_route(st, proj, T):
    """Geometry for this (projection, T) or None -> keep today's path. Shape-only: graph safe."""
    if st is None or T < 1 or T > P1_MAXT or proj not in st["proj"]:
        return None
    b = p1_bucket(T)
    if not st["route"].get((proj, b)):
        return None
    return st["geo"].get((proj, b))


def p1_gemv(st, proj, x, out=None, n_rows=None, ldo=None, geo=None):
    """y = x @ W[:n_rows].T (BF16 out, FP32 accumulate) into out [T, >= n_rows] (row stride ldo). x [T, K] bf16
    contiguous, T <= 8. Allocates only when out is None (shape-only allocation). Raises on unsupported shapes."""
    P = st["proj"][proj]
    T = x.shape[0]
    if n_rows is None:
        n_rows = P["rows_t8"]
    if out is None:
        out = torch.empty((T, n_rows), dtype=torch.bfloat16, device=x.device)
    if ldo is None:
        ldo = out.stride(0)
    if geo is None:
        geo = st["geo"][(proj, p1_bucket(T))]
    mode = P["mode_t8"] if n_rows == P["rows_t8"] else p1_mode_of(P, n_rows)
    p1_call(st["mod"], P, x, out, n_rows, ldo, geo, mode=mode)
    return out

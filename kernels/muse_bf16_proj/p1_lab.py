# p1 lab: P12 lossless BF16 weight code (12 bits/weight) + fused decode/skinny-GEMM for T <= 32 (LAB ONLY, never submit)
"""P1 lab: pack the layer's BF16 projection weights losslessly on the GPU, verify bit-exact unpack for every
projection, check the packed GEMM, and time packed vs today's k3_v12 path per (projection, T) in ONE process
(CUDA graph per call, 256 MB L2 flush before every replay, paired alternating order). Prints a table and a gate line.

    tools/box/box.sh timed 'python3 kernels/muse_bf16_proj/p1_lab.py 2>&1 | tee box_out/p1_lab.log'
    python3 tools/box/sass_gate.py kernels/muse_bf16_proj/p1_lab.py          # compile gate (spills / regs)

Design: kernels/muse_bf16_proj/p1_design.md. CPU emulation of the format: scratchpad p1/emulate.py.

Format P12 (per 16-row tile, per 16x64 super-block SB = 1536 B = 3 x 512 B, A-fragment order of mma.m16n8k16):
  lane L, register o (0..15), half h:  g = L>>2, t = L&3, kt = o>>2, i = o&3,
      row = g + 8*(i&1),  k = 64*sb + 16*kt + 2*t + 8*(i>>1) + h
  chunk 0/1: byte plane (lane's 16 B = bytes 2*(o&7)+h of o = 0..7 / 8..15): byte = (n & 1) << 7 | mantissa7
  chunk 2  : code plane (lane's 4 words cw[kt]): bits n1, n2, n3, sign of (o, h) at ((8,9,10,15) + 16h + 4i) mod 32
  n = e - base[rt] (window of 16 exponents per row tile); decode to the SCALED bf16 (exponent 0x70 | n):
      a[o] = prmt(bw[o>>1], 0x38383838, o odd ? 0x4342 : 0x4140) | (rotr(cw[kt], 4*i) & 0x87008700)
  the GEMM result of a row tile is multiplied by scale[rt] = 2^(base - 112) (exact power-of-two scaling).
  escapes (exponent outside the window: zeros, subnormals, inf/NaN, outliers) are coded 0 and patched in registers
  from a per-row-tile list (<= 32 entries, int2 {sb | lane<<16 | o<<21 | h<<25, scaled16 | raw16<<16}); a group of
  16 row tiles with more escapes (or an escape whose scaled value is not exact) is stored RAW (plain bf16 in fragment
  order, 2048 B per SB, scale 1). Unpack = decode, exponent back to base + n, escapes from raw16: bit-exact.
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
P1_RAW_GROUP = 16       # row tiles that share a format (256 rows): CTA-uniform format for every geometry
P1_TS = (1, 8, 16, 32)  # scored T <= 32 (decode 1/16/32, verify 1); T = 128/256 stay on cuBLAS
P1_EXTRA_TS = (2, 5, 9, 24, 33)   # correctness only (nearby shapes; 33 must be refused -> current path)
REPS_SWEEP = 9
REPS_FINAL = 41
GATE = 1.15             # packed must be >= 15% faster (net of the graph replay floor) than today's path

P1_CPP_SRC = r"""
#include <torch/extension.h>
void p1_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor fmt,
             torch::Tensor eoff, torch::Tensor esc, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t ldo,
             int64_t warps, int64_t rowt, int64_t split, int64_t depth, int64_t pdl);
void p1_unpack(torch::Tensor pk, torch::Tensor toff, torch::Tensor base, torch::Tensor fmt, torch::Tensor eoff,
               torch::Tensor esc, torch::Tensor out, int64_t N, int64_t K, int64_t use_esc);
"""

P1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace p1 {

constexpr uint32_t KC4 = 0x38383838u;     // hi byte of every scaled bf16: exponent bits e4..e7 of 0x70
constexpr uint32_t CMASK = 0x87008700u;   // code bits per half: e1 e2 e3 (bits 8-10) and the sign (bit 15)
constexpr int SBP = 1536;                 // packed super-block bytes (16 rows x 64 k)
constexpr int SBR = 2048;                 // raw super-block bytes
constexpr int BIG = 0x7fffffff;

struct Args {
  const __nv_bfloat16* x; int M; int K; int ldx;
  const uint8_t* pk; const int64_t* toff; const float* scale; const uint8_t* fmt;
  const int* eoff; const int2* esc;
  void* out; int ldo; int N; int nrt; int nsb_total; int ksm;
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int r) { return __funnelshift_r(x, x, r); }

// k-tile kt of a packed SB (the lane's three 16-B chunks) -> the 4 A-fragment registers (scaled bf16x2):
// 4 PRMT + 3 SHF + 4 LOP3 per 8 weights
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

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ uint4 ldw(const uint8_t* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// B fragments (x = tokens x k in smem, token-major) of k-tile kt of the current SB. NT == 1: one x4 = two k-tiles.
template <int NT>
__device__ __forceinline__ void load_b(uint32_t xk, int kt, uint32_t pp_stride, uint32_t (&bc)[4], uint32_t (&b)[NT][2]) {
  if constexpr (NT == 1) {
    if ((kt & 1) == 0) ldsm4(bc, xk + kt * 32);
    b[0][0] = bc[(kt & 1) * 2];
    b[0][1] = bc[(kt & 1) * 2 + 1];
  } else {
#pragma unroll
    for (int pp = 0; pp < NT / 2; ++pp) {
      uint32_t t4[4];
      ldsm4(t4, xk + kt * 32 + pp * pp_stride);
      b[2 * pp][0] = t4[0]; b[2 * pp][1] = t4[1]; b[2 * pp + 1][0] = t4[2]; b[2 * pp + 1][1] = t4[3];
    }
  }
}

// In-register escape patch of row tile r's registers a[16] at local SB s (entries sorted by SB; warp-uniform loop).
__device__ __forceinline__ void esc_patch(uint32_t (&a)[16], int s, int sb0, int lane, uint32_t em, uint32_t ev,
                                          int& cur, int end, int& next) {
  while (cur < end) {
    const uint32_t m = __shfl_sync(0xffffffffu, em, cur);
    if ((int)(m & 0xFFFFu) - sb0 != s) break;
    const uint32_t v = __shfl_sync(0xffffffffu, ev, cur);
    if (lane == (int)((m >> 16) & 31u)) {
      const int o = (int)((m >> 21) & 15u), h = (int)((m >> 25) & 1u);
      const uint32_t sc = v & 0xFFFFu;
#pragma unroll
      for (int q = 0; q < 16; ++q)
        if (q == o) a[q] = h ? ((a[q] & 0x0000FFFFu) | (sc << 16)) : ((a[q] & 0xFFFF0000u) | sc);
    }
    ++cur;
  }
  const uint32_t mn = __shfl_sync(0xffffffffu, em, cur & 31);
  next = cur < end ? (int)(mn & 0xFFFFu) - sb0 : BIG;
}

// Prologue loads, x stage (+ the CTA barrier), main loop. RAW is CTA-uniform (format groups of 256 rows).
template <int NT, int ROWT, int D, bool RAW>
__device__ __forceinline__ void run(const Args& A, unsigned char* smem, int tid, int lane, int nthr, int rt0,
                                    bool active, int sb0, int nsb, float (&acc)[ROWT][NT][4]) {
  constexpr int SBB = RAW ? SBR : SBP;
  constexpr int NCH = RAW ? 4 : 3;
  const int LD = A.ksm + 8;   // x row stride in elements (16 B pad: conflict-free ldmatrix)

  const uint8_t* tp[ROWT];
#pragma unroll
  for (int r = 0; r < ROWT; ++r) {
    const int rt = min(rt0 + r, A.nrt - 1);
    tp[r] = A.pk + A.toff[rt] + (long long)sb0 * SBB + lane * 16;
  }
  // 1. the first D super-blocks go out before anything else
  uint4 rb[D][ROWT][NCH];
  if (active) {
#pragma unroll
    for (int d = 0; d < D; ++d)
      if (d < nsb)
#pragma unroll
        for (int r = 0; r < ROWT; ++r)
#pragma unroll
          for (int c = 0; c < NCH; ++c) rb[d][r][c] = ldw(tp[r] + d * SBB + c * 512);
  }
  // 2. this warp's escape entries inside the slice [sb0, sb0 + nsb) (one per lane, <= 32 per row tile)
  uint32_t em[ROWT], ev[ROWT];
  int ecur[ROWT], eend[ROWT], enext[ROWT];
#pragma unroll
  for (int r = 0; r < ROWT; ++r) { em[r] = 0u; ev[r] = 0u; ecur[r] = 32; eend[r] = 0; enext[r] = BIG; }
  if constexpr (!RAW) {
    if (active) {
#pragma unroll
      for (int r = 0; r < ROWT; ++r) {
        const int rt = min(rt0 + r, A.nrt - 1);
        const int o0 = A.eoff[rt], o1 = A.eoff[rt + 1];
        const int idx = o0 + lane;
        int2 e = make_int2(0, 0);
        if (idx < o1) e = A.esc[idx];
        em[r] = (uint32_t)e.x; ev[r] = (uint32_t)e.y;
        const int esb = (int)(em[r] & 0xFFFFu);
        const bool in = idx < o1 && esb >= sb0 && esb < sb0 + nsb;
        const uint32_t msk = __ballot_sync(0xffffffffu, in);
        ecur[r] = msk ? __ffs(msk) - 1 : 32;
        eend[r] = msk ? 32 - __clz(msk) : 0;
        const int first = __shfl_sync(0xffffffffu, esb, ecur[r] & 31);
        enext[r] = msk ? first - sb0 : BIG;
      }
    }
  }
  // PDL: everything above reads prepare()-time data only; x (and the output) belong to the predecessor kernel.
  // A no-op when launched without programmatic stream serialization.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  // 3. stage x[0:NT*8, k0:k0+ks] (zero rows >= M)
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
  {
    const int k0 = sb0 * 64, vpr = nsb * 8;
    for (int i = tid; i < NT * 8 * vpr; i += nthr) {
      const int t = i / vpr, v = i - t * vpr;
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (t < A.M) val = *reinterpret_cast<const uint4*>(A.x + (size_t)t * A.ldx + k0 + v * 8);
      *reinterpret_cast<uint4*>(xs + (size_t)t * LD + v * 8) = val;
    }
  }
  __syncthreads();
  if (!active) return;

  // 4. stream: decode + mma per k-tile; refill each stage after use (D stages in flight)
  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  uint32_t lane_off;
  {
    const int q = lane >> 3;
    if constexpr (NT == 1) lane_off = (uint32_t)(((lane & 7) * LD + 16 * (q >> 1) + 8 * (q & 1)) * 2);
    else lane_off = (uint32_t)(((8 * (q >> 1) + (lane & 7)) * LD + 8 * (q & 1)) * 2);
  }
  const uint32_t xl = xs_base + lane_off;
  const uint32_t pp_stride = (uint32_t)(16 * LD * 2);
  for (int s0 = 0; s0 < nsb; s0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int s = s0 + d;
      if (s < nsb) {
        const uint32_t xk = xl + (uint32_t)(s * 128);
        bool hit = false;
        if constexpr (!RAW) {
#pragma unroll
          for (int r = 0; r < ROWT; ++r) hit |= (s == enext[r]);
        }
        if (!hit) {
          uint32_t bc[4];
#pragma unroll
          for (int kt = 0; kt < 4; ++kt) {
            uint32_t b[NT][2];
            load_b<NT>(xk, kt, pp_stride, bc, b);
#pragma unroll
            for (int r = 0; r < ROWT; ++r) {
              uint32_t a[4];
              if constexpr (RAW) {
                const uint4 c = rb[d][r][kt];
                a[0] = c.x; a[1] = c.y; a[2] = c.z; a[3] = c.w;
              } else {
                uint4 pc[3] = {rb[d][r][0], rb[d][r][1], rb[d][r][2]};
                dec_kt(pc, kt, a);
              }
#pragma unroll
              for (int ng = 0; ng < NT; ++ng) mma16816(acc[r][ng], a, b[ng][0], b[ng][1]);
            }
          }
        } else if constexpr (!RAW) {
          // rare (~7% of SBs per row tile on N(0,1/K) weights): decode the whole SB, patch, then mma
#pragma unroll
          for (int r = 0; r < ROWT; ++r) {
            uint32_t a[16];
            uint4 pc[3] = {rb[d][r][0], rb[d][r][1], rb[d][r][2]};
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
              uint32_t t4[4];
              dec_kt(pc, kt, t4);
              a[4 * kt] = t4[0]; a[4 * kt + 1] = t4[1]; a[4 * kt + 2] = t4[2]; a[4 * kt + 3] = t4[3];
            }
            if (s == enext[r]) esc_patch(a, s, sb0, lane, em[r], ev[r], ecur[r], eend[r], enext[r]);
            uint32_t bc[4];
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
              uint32_t b[NT][2];
              load_b<NT>(xk, kt, pp_stride, bc, b);
              const uint32_t ak[4] = {a[4 * kt], a[4 * kt + 1], a[4 * kt + 2], a[4 * kt + 3]};
#pragma unroll
              for (int ng = 0; ng < NT; ++ng) mma16816(acc[r][ng], ak, b[ng][0], b[ng][1]);
            }
          }
        }
        if (s + D < nsb) {
#pragma unroll
          for (int r = 0; r < ROWT; ++r)
#pragma unroll
            for (int c = 0; c < NCH; ++c) rb[d][r][c] = ldw(tp[r] + (s + D) * SBB + c * 512);
        }
      }
    }
  }
}

// out[t, n] = sum_k x[t, k] W[n, k]  (t < M <= 8*NT), W packed; grid = (row groups, S) with cluster (1, S, 1):
// the S split-K CTAs of a row group reduce through DSMEM in rank order (deterministic), no atomics, no workspace.
// register estimate per thread: D stages x ROWT x 12 raw + accumulators + B fragments + ~35 (escapes, addresses);
// above ~112 the 2-CTA/SM bound (128 regs) would spill, so those instantiations ask for 1 CTA/SM of 8 warps
template <int NT, int ROWT, int D>
struct Occ { static constexpr int regs = 12 * ROWT * D + 4 * NT * ROWT + 2 * NT + 35; static constexpr int minb = regs > 112 ? 1 : 2; };

template <int NT, int ROWT, int D, bool F32>
__global__ void __launch_bounds__(256, Occ<NT, ROWT, D>::minb) p1_gemm_kernel(const __grid_constant__ Args A) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5, nthr = (int)blockDim.x, nw = nthr >> 5;
  const int S = (int)gridDim.y, sidx = (int)blockIdx.y;
  const int sb0 = (int)((long long)sidx * A.nsb_total / S), sb1 = (int)((long long)(sidx + 1) * A.nsb_total / S);
  const int nsb = sb1 - sb0;
  const int bx = (int)blockIdx.x;
  const int rt0 = (bx * nw + warp) * ROWT;
  const bool active = rt0 < A.nrt;
  const int rtg = min(bx * nw * ROWT, A.nrt - 1);   // CTA's first row tile: its format group's flag
  const bool raw = A.fmt[rtg] != 0;

  float acc[ROWT][NT][4];
#pragma unroll
  for (int r = 0; r < ROWT; ++r)
#pragma unroll
    for (int ng = 0; ng < NT; ++ng)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[r][ng][j] = 0.f;

  if (raw) run<NT, ROWT, D, true>(A, smem, tid, lane, nthr, rt0, active, sb0, nsb, acc);
  else run<NT, ROWT, D, false>(A, smem, tid, lane, nthr, rt0, active, sb0, nsb, acc);

  // epilogue: scale 2^(base - 112) per row tile, fragments -> smem [NT*8][RC] fp32, cluster reduce, store
  __syncthreads();
  const int RC = nw * ROWT * 16;
  float* red = reinterpret_cast<float*>(smem);
  if (active) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int r = 0; r < ROWT; ++r) {
      const float sc = A.scale[min(rt0 + r, A.nrt - 1)];
      const int rl = (warp * ROWT + r) * 16 + g;
#pragma unroll
      for (int ng = 0; ng < NT; ++ng) {
        const int tok = 8 * ng + 2 * t;
        red[tok * RC + rl] = acc[r][ng][0] * sc;
        red[(tok + 1) * RC + rl] = acc[r][ng][1] * sc;
        red[tok * RC + rl + 8] = acc[r][ng][2] * sc;
        red[(tok + 1) * RC + rl + 8] = acc[r][ng][3] * sc;
      }
    }
  }
  cg::cluster_group cluster = cg::this_cluster();
  if (S > 1) cluster.sync(); else __syncthreads();
  const int rank = S > 1 ? (int)cluster.block_rank() : 0;
  const int rpr = RC / S;
  const int row_base = bx * RC + rank * rpr;
  for (int i = tid; i < A.M * rpr; i += nthr) {
    const int tok = i / rpr, rr = i - tok * rpr;
    const int row = row_base + rr;
    if (row >= A.N) continue;
    const int li = tok * RC + rank * rpr + rr;
    float v = 0.f;
    if (S > 1) {
      for (int p = 0; p < S; ++p) v += cluster.map_shared_rank(red, p)[li];
    } else {
      v = red[li];
    }
    if constexpr (F32) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + row] = v;
    else reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
  }
  if (S > 1) cluster.sync();   // keep this CTA's smem alive until every rank has read it
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
        uint32_t w = t4[i], o = 0u;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t x = (w >> (16 * h)) & 0xFFFFu;
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

template <int NT, int ROWT, int D, bool F32>
static void launch(const Args& A, int warps, int S, int ngroups, size_t smem, int pdl) {
  auto kern = p1_gemm_kernel<NT, ROWT, D, F32>;
  static size_t smem_set = 0;   // per instantiation; set eagerly (warm-up) before any graph capture
  if (smem > smem_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    smem_set = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ngroups, (unsigned)S, 1);
  cfg.blockDim = dim3((unsigned)(warps * 32), 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (S > 1) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = 1;
    attr[na].val.clusterDim.y = (unsigned)S;
    attr[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attr;
  cfg.numAttrs = (unsigned)na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, A));
}

}  // namespace p1

void p1_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor fmt,
             torch::Tensor eoff, torch::Tensor esc, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t ldo,
             int64_t warps, int64_t rowt, int64_t split, int64_t depth, int64_t pdl) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "p1: x bf16 contiguous");
  TORCH_CHECK(x.size(0) == M && x.size(1) == K && M >= 1 && M <= 32, "p1: x shape / 1 <= M <= 32");
  TORCH_CHECK(K % 64 == 0 && N % 16 == 0, "p1: K % 64, N % 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "p1: x align");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              fmt.scalar_type() == at::kByte && eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt,
              "p1: packed dtypes");
  const int nrt = (int)(N / 16);
  TORCH_CHECK(toff.numel() >= nrt && scale.numel() >= nrt && fmt.numel() >= nrt && eoff.numel() >= nrt + 1, "p1: meta");
  const bool f32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK(f32 || out.scalar_type() == at::kBFloat16, "p1: out dtype");
  TORCH_CHECK(out.stride(1) == 1 && out.stride(0) == ldo && ldo >= N && out.size(0) >= M && out.size(1) >= N, "p1: out");
  TORCH_CHECK((warps == 4 || warps == 8) && (rowt == 1 || rowt == 2) && (depth == 2 || depth == 3), "p1: geometry");
  const int nsb_total = (int)(K / 64);
  TORCH_CHECK(split >= 1 && split <= 8 && split <= nsb_total, "p1: split");
  const int rc = (int)(warps * rowt * 16);
  TORCH_CHECK(rc % split == 0, "p1: rows per CTA % split");
  const int nt = M <= 8 ? 1 : (M <= 16 ? 2 : 4);
  const int ksm = 64 * (int)((nsb_total + split - 1) / split);
  const size_t xs_bytes = (size_t)nt * 8 * (ksm + 8) * 2, red_bytes = (size_t)nt * 8 * rc * 4;
  const size_t smem = xs_bytes > red_bytes ? xs_bytes : red_bytes;
  TORCH_CHECK(smem <= 227 * 1024, "p1: smem");
  const int ngroups = (nrt + (int)(warps * rowt) - 1) / (int)(warps * rowt);
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1::Args A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.K = (int)K; A.ldx = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.fmt = fmt.data_ptr<uint8_t>(); A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>());
  A.out = out.data_ptr(); A.ldo = (int)ldo; A.N = (int)N; A.nrt = nrt; A.nsb_total = nsb_total; A.ksm = ksm;
  const int w = (int)warps, S = (int)split;
#define P1_CASE(NT_, R_, D_, F_) \
  if (nt == NT_ && rowt == R_ && depth == D_ && f32 == F_) { p1::launch<NT_, R_, D_, F_>(A, w, S, ngroups, smem, (int)pdl); return; }
  P1_CASE(1, 1, 2, false) P1_CASE(1, 1, 3, false) P1_CASE(1, 2, 2, false) P1_CASE(1, 2, 3, false)
  P1_CASE(2, 1, 2, false) P1_CASE(2, 1, 3, false) P1_CASE(2, 2, 2, false) P1_CASE(2, 2, 3, false)
  P1_CASE(4, 1, 2, false) P1_CASE(4, 1, 3, false) P1_CASE(4, 2, 2, false) P1_CASE(4, 2, 3, false)
  P1_CASE(2, 1, 2, true) P1_CASE(2, 1, 3, true) P1_CASE(2, 2, 2, true) P1_CASE(2, 2, 3, true)
  P1_CASE(4, 1, 2, true) P1_CASE(4, 1, 3, true) P1_CASE(4, 2, 2, true) P1_CASE(4, 2, 3, true)
#undef P1_CASE
  TORCH_CHECK(false, "p1: no instantiation for this (NT, rowt, depth, out dtype)");
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
        name=f"p1_lab_{tag}", cpp_sources=P1_CPP_SRC, cuda_sources=P1_CUDA_SRC,
        functions=["p1_gemm", "p1_unpack"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a"],
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


# ================================================================================================ calls
def p1_call(mod, P, x, out, n_rows, ldo, geo, pdl=0):
    warps, rowt, split, depth = geo
    mod.p1_gemm(x, P["buf"], P["toff"], P["scale"], P["fmt"], P["eoff"], P["esc"], out, x.shape[0], n_rows,
                P["K"], ldo, warps, rowt, split, depth, pdl)
    return out


def unpack(mod, P, use_esc=True):
    out = torch.empty((P["N"], P["K"]), dtype=torch.bfloat16, device=P["buf"].device)
    mod.p1_unpack(P["buf"], P["toff"], P["base"], P["fmt"], P["eoff"], P["esc"], out, P["N"], P["K"], 1 if use_esc else 0)
    return out


def geometries(P, n_rows, m, f32, min_ctas=48):
    """Valid (warps, rowt, split, depth) for this call (mirrors the host checks)."""
    nt = 1 if m <= 8 else (2 if m <= 16 else 4)
    nrt = n_rows // 16
    out = []
    for warps in (4, 8):
        for rowt in (1, 2):
            rc = warps * rowt * 16
            for split in (1, 2, 4, 8):
                if rc % split or split > P["nsb"]:
                    continue
                ksm = 64 * (-(-P["nsb"] // split))
                smem = max(nt * 8 * (ksm + 8) * 2, nt * 8 * rc * 4)
                if smem > 227 * 1024:
                    continue
                ctas = (-(-nrt // (warps * rowt))) * split
                if ctas < min_ctas or ctas > 4096:
                    continue
                for depth in (2, 3):
                    if f32 and nt == 1:
                        continue
                    out.append((warps, rowt, split, depth))
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
        ok = mism == 0 and neg_ok
        ok_all &= ok
        print(f"[p1] {describe(pk)}", flush=True)
        print(f"[p1]   unpack mismatches {mism} (must be 0) | negative control without patches: {neg} mismatches "
              f"(must be > 0 when escapes exist) -> {'PASS' if ok else 'FAIL'}", flush=True)
        P[name] = pk
        if name.startswith("stress"):
            mats[name] = w
    # the same kernel on uncompressed BF16 (every tile RAW): separates the kernel-structure gain from compression
    PR = {}
    for name in ("qkv", "o", "cat", "unpooler"):
        PR[name] = pack_p12(mats[name], esc_max=-1, name=name + "_raw")
        mism = int((unpack(mod, PR[name]).view(torch.int16) != mats[name].view(torch.int16)).sum())
        ok_all &= mism == 0 and PR[name]["n_raw_tiles"] == PR[name]["nrt"]
        print(f"[p1] {describe(PR[name])} | unpack mismatches {mism}", flush=True)
    if not ok_all:
        print("[p1] GATE FAIL: unpack is not bit-exact; stopping before timing", flush=True)
        return 1

    # ---------------------------------------------------------------- 2. packed GEMM correctness
    print("\n===== P1 GEMM CHECKS =====", flush=True)
    gen = torch.Generator(device=dev).manual_seed(7)

    def one_hot_cols(pk, w, n_rows, cols, f32):
        """x = e_k rows: y[t, :] must equal W[:n_rows, k_t] exactly (one exact product per output). Rows holding
        an inf/NaN weight are skipped (0 * inf = NaN in any GEMM); a subnormal weight read back as 0 is counted
        apart (tensor-core input flush: the same on raw BF16 through cuBLAS, not a packing error)."""
        bad = flushed = 0
        fin = torch.isfinite(w[:n_rows].float()).all(1)
        for i in range(0, len(cols), 32):
            cc = cols[i:i + 32]
            x = torch.zeros((len(cc), pk["K"]), dtype=torch.bfloat16, device=dev)
            x[torch.arange(len(cc), device=dev), torch.tensor(cc, device=dev)] = 1.0
            out = torch.empty((len(cc), n_rows), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
            geo = geometries(pk, n_rows, len(cc), f32, min_ctas=1)[0]
            p1_call(mod, pk, x, out, n_rows, n_rows, geo)
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
        # every escape column (patched path) + 64 random columns, M = 32 and M = 1 (two kernels)
        cols_esc = []
        ex_rows = pk["esc"].shape[0] if pk["n_esc"] else 0
        if ex_rows:
            e = pk["esc"][:, 0].long()
            sb, L, o, h = e & 0xFFFF, (e >> 16) & 31, (e >> 21) & 15, (e >> 25) & 1
            kk = 64 * sb + 16 * (o >> 2) + 2 * (L & 3) + 8 * ((o & 3) >> 1) + h
            cols_esc = sorted(set(kk.tolist()))[:512]
        cols = cols_esc + torch.randint(0, pk["K"], (64,), generator=gen, device=dev).tolist()
        n_rows = pk["N"]
        bad, fl = one_hot_cols(pk, w, n_rows, cols, f32=False)
        bad1, fl1 = one_hot_cols(pk, w, n_rows, cols[:8], f32=False)     # NT = 1 kernel too
        print(f"[p1] one-hot probe {name}: {len(cols)} columns ({len(cols_esc)} escape columns): {bad + bad1} mismatches "
              f"(must be 0); subnormals flushed by the tensor core {fl + fl1}", flush=True)
        gemm_ok &= (bad + bad1) == 0
        if name.startswith("stress"):
            continue
        for m in P1_TS + P1_EXTRA_TS:
            if m > 32:
                continue
            x = torch.randn((m, pk["K"]), generator=gen, device=dev).to(torch.bfloat16)
            n_rows = pk["N"] if name != "cat" or m > 8 else 4096
            f32 = name == "cat" and m > 8
            ref = (x.double() @ w[:n_rows].double().t())
            errs = []
            for geo in geometries(pk, n_rows, m, f32)[:: (4 if args.quick else 1)]:
                out = torch.empty((m, n_rows), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
                p1_call(mod, pk, x, out, n_rows, n_rows, geo)
                out2 = out.clone()
                p1_call(mod, pk, x, out, n_rows, n_rows, geo)
                rel = float((out.double() - ref).norm() / ref.norm())
                errs.append((rel, bool(torch.equal(out, out2)), geo))
            worst = max(errs)
            tol = 2e-5 if f32 else 4e-3
            good = worst[0] < tol and all(e[1] for e in errs)
            gemm_ok &= good
            print(f"[p1] gemm {name} M={m}: {len(errs)} geometries, worst rel-L2 vs fp64 {worst[0]:.2e} (tol {tol:.0e}), "
                  f"repeat bit-identical {all(e[1] for e in errs)} -> {'ok' if good else 'FAIL'}", flush=True)
    try:
        x33 = torch.randn((33, 8192), device=dev).to(torch.bfloat16)
        mod.p1_gemm(x33, P["qkv"]["buf"], P["qkv"]["toff"], P["qkv"]["scale"], P["qkv"]["fmt"], P["qkv"]["eoff"],
                    P["qkv"]["esc"], torch.empty((33, 2304), dtype=torch.bfloat16, device=dev), 33, 2304, 8192, 2304,
                    8, 1, 4, 2, 0)
        print("[p1] M=33 was accepted (expected a refusal -> current path)", flush=True)
        gemm_ok = False
    except Exception:  # noqa: BLE001
        print("[p1] M=33 refused by the host check (T > 32 stays on the current path): ok", flush=True)
    if not gemm_ok:
        print("[p1] GATE FAIL: GEMM checks failed; stopping before timing", flush=True)
        return 1

    # ---------------------------------------------------------------- 3. timing, packed vs today's k3_v12 path
    print("\n===== P1 TIMING (CUDA graph, 256 MB flush per replay, paired alternating order) =====", flush=True)
    gemm, gemm_name = bf16_gemm_fn()
    r1b, r1b_st = load_r1b()
    flush = torch.empty((64 * 1024 * 1024,), dtype=torch.float32, device=dev)
    zbuf = torch.zeros((8, 8192), dtype=torch.bfloat16, device=dev)
    tiny = torch.zeros((1,), device=dev)
    floor_us = time_one(graph_of(lambda: tiny.add_(1.0)), flush, 21)
    print(f"[p1] graph replay floor (one tiny kernel): {floor_us:.2f} us; cuBLAS path: {gemm_name}", flush=True)

    def today(name, m, x):
        """(callable, label) = k3_v12's projection path at this T."""
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
            if m <= 8:
                return (lambda: gemm(x, W["pooler"])), "cuBLAS pooler"
            return (lambda: torch.mm(x, wcat.t(), out_dtype=torch.float32)), "cuBLAS cat fp32"
        if m <= 8:
            ws = mats["unpooler"][:1024]
            return (lambda: torch.mm(x, ws.t(), out=zbuf[:m, :1024])), "cuBLAS slice->zbuf"
        return (lambda: gemm(x, mats["unpooler"])), "cuBLAS"

    rows = []
    shapes = [("qkv", 8192), ("o", 1024), ("cat", 8192), ("unpooler", 4096)]
    for name, k in shapes:
        pk = P[name]
        for m in P1_TS:
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            n_rows = {"cat": 4096 if m <= 8 else 4608, "unpooler": 1024 if m <= 8 else 8192}.get(name, pk["N"])
            f32 = name == "cat" and m > 8
            ldo = 8192 if (name == "unpooler" and m <= 8) else n_rows
            out = (zbuf[:m] if (name == "unpooler" and m <= 8) else
                   torch.empty((m, n_rows), dtype=torch.float32 if f32 else torch.bfloat16, device=dev))
            fa, label = today(name, m, x)
            ga = graph_of(fa)
            bests = {}
            for var, pv in (("packed", pk), ("raw", PR[name])):
                geos = geometries(pv, n_rows, m, f32)
                if args.quick:
                    geos = geos[::2]
                sweep, best = [], None
                for geo in geos:
                    try:
                        p1_call(mod, pv, x, out, n_rows, ldo, geo)
                        gb = graph_of(lambda geo=geo, pv=pv: p1_call(mod, pv, x, out, n_rows, ldo, geo))
                    except Exception as ex:  # noqa: BLE001
                        print(f"[p1]   {name} M={m} {var} geo {geo}: launch failed {ex!r}", flush=True)
                        continue
                    _, tb, _ = time_pair(ga, gb, flush, reps_sweep)
                    sweep.append((tb, geo))
                    if best is None or tb < best[0]:
                        best = (tb, geo, gb)
                sweep.sort()
                print(f"[p1] {name} M={m} {var} sweep (us, (warps,rowt,split,depth)): " +
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
                t_r, t_b2, c_ratio = time_pair(bests["raw"][2], best[2], flush, reps_final)
            mb = (pk["bytes"] if n_rows == pk["N"] else pk["bytes"] * n_rows / pk["N"]) / 1e6
            bf16_mb = n_rows * k * 2 / 1e6
            rows.append((name, m, label, t_a, t_b, ratio, net, best[1], mb, bf16_mb, t_r, c_ratio))
            print(f"[p1] {name} M={m}: today {label} {t_a:.2f} us | packed {t_b:.2f} us {best[1]} | paired x{ratio:.3f} "
                  f"| net of floor x{net:.3f} | same kernel on BF16 {t_r:.2f} us (compression alone x{c_ratio:.3f})",
                  flush=True)

    # ---------------------------------------------------------------- 3b. PDL lever (round-2 candidate)
    # graph = [producer kernel writing x; p1]: p1 with programmatic stream serialization issues its weight prefetch
    # before griddepcontrol.wait, so its ramp overlaps the producer's tail. Measured on the best packed geometry.
    print("\n===== P1 PDL (producer x.copy_ -> p1, paired: pdl off / pdl on) =====", flush=True)
    pdl_rows = []
    for (name, m, label, t_a, t_b, ratio, net, geo, mb, bmb, t_r, c_ratio) in rows:
        if m not in (1, 32):
            continue
        k = {"qkv": 8192, "o": 1024, "cat": 8192, "unpooler": 4096}[name]
        pk = P[name]
        n_rows = {"cat": 4096 if m <= 8 else 4608, "unpooler": 1024 if m <= 8 else 8192}.get(name, pk["N"])
        f32 = name == "cat" and m > 8
        ldo = 8192 if (name == "unpooler" and m <= 8) else n_rows
        out = (zbuf[:m] if (name == "unpooler" and m <= 8) else
               torch.empty((m, n_rows), dtype=torch.float32 if f32 else torch.bfloat16, device=dev))
        xsrc = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        x = torch.empty_like(xsrc)
        try:
            g0 = graph_of(lambda: (x.copy_(xsrc), p1_call(mod, pk, x, out, n_rows, ldo, geo, 0)))
            g1 = graph_of(lambda: (x.copy_(xsrc), p1_call(mod, pk, x, out, n_rows, ldo, geo, 1)))
            ref = out.clone()
            g1.replay()
            torch.cuda.synchronize()
            same = bool(torch.equal(out, ref))
            t0_, t1_, r_ = time_pair(g0, g1, flush, reps_final)
            pdl_rows.append((name, m, t0_, t1_, r_, same))
            print(f"[p1] PDL {name} M={m}: off {t0_:.2f} us | on {t1_:.2f} us | x{r_:.3f} | output identical {same}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[p1] PDL {name} M={m}: failed {ex!r}", flush=True)

    # ---------------------------------------------------------------- 4. table + gate
    print("\n===== P1 SUMMARY (event us incl. graph replay; net = minus the one-kernel floor "
          f"{floor_us:.2f} us) =====", flush=True)
    print(f"| proj | T | today's path | today us | p1 on BF16 us | packed us | paired x | net x | compression x | "
          f"geometry (w,r,S,D) | packed MB | eff TB/s packed | BF16-equiv TB/s | gate x{GATE} |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    gate_ok = True
    for (name, m, label, t_a, t_b, ratio, net, geo, mb, bmb, t_r, c_ratio) in rows:
        nt = max(t_b - floor_us, 1e-3)
        g = net >= GATE
        gate_ok &= g
        print(f"| {name} | {m} | {label} | {t_a:.2f} | {t_r:.2f} | {t_b:.2f} | {ratio:.3f} | {net:.3f} | {c_ratio:.3f} | "
              f"{geo} | {mb:.1f} | {mb / nt:.2f} | {bmb / nt:.2f} | {'PASS' if g else 'fail'} |")
    n_shapes = len(rows)
    print(f"\n[p1] GATE (bit-exact unpack on all weights; packed >= x{GATE} net on qkv, o, pooler, unpooler at T <= 32): "
          f"{'PASS' if (gate_ok and ok_all and n_shapes == 16) else 'FAIL'} ({sum(r[6] >= GATE for r in rows)}/{n_shapes} "
          f"shapes pass); total {time.time() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

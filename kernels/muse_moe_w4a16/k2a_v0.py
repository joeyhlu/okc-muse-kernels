# okc: problem=muse_moe_w4a16
# nonce k2a-v0-1
"""K2a v0: custom NVFP4 W4A16 MoE experts for decode/verify (sm_90, mma.sync m16n8k16, swap-AB).

Pipeline per call (4 launches, all fixed grids, CUDA-graph capturable):
  1. route   (1 CTA x 1024): per-local-expert row lists, active-expert list, compact rows, pair->row map,
              split-K factor for W2 -- all on device.
  2. gemm<1> (persistent, 1 CTA/SM x 16 warps): H = X . W13^T for every active expert; each 32-row tile
              (16 gate rows + the matching 16 up rows) x full K is one unit, its K split over the 16 warps
              and reduced in smem; epilogue x(gs*2^8) in FP32, SiLU(gate)*up, BF16 intermediate.
  3. gemm<2> (same kernel, MODE 2): Y = inter . W2^T, unit = 32 rows x K/KS, FP32 partials per k-split.
  4. final   (1 CTA per token): sum k-split partials, y->BF16, *(1+w) (BF16, as SGLang), RMSNorm (FP32),
              * topk weight, sum over the token's local routes, BF16 out (zeros if no local route).
Weights are read from HBM once per touched expert per call (multi-chunk experts re-read from L1/L2).

Layouts (see k2a_design.md and k2a_emulate.py, which checks them bit-exactly on CPU):
  WQ[e][rt][kb][lane][4 x u32]: row tile rt (32 prepacked rows), k32 block kb, lane = 4g+t; word wi = 2*rho+h
     holds the 8 E2M1 codes of tile row 16*rho+8*h+g at k = 32*kb+8*t+i (i=0..7), bits per BITPOS.
  SQ[e][rt][kb][64 B]: byte gi*32 + g*4 + 2*rho+h = scale of row 16*rho+8*h+g, group 2*kb+gi, re-encoded
     losslessly so that bf16_bits = b*16 + 0x7580 = sf * 2^118.
  W13 prepacked row m = 32*rt + 16*rho + i  <->  original row rho*4096 + 16*rt + i (gate/up pairs per tile).
Dequantized weight = e2m1 * 2^-126 (bit placement) * sf * 2^118 = exact BF16 of w_true * 2^-8; the FP32
global scale (x 2^8) is applied to the accumulator.
"""

import hashlib
import time
from typing import TypedDict

import torch

FALLBACK = False   # True: pure PyTorch path only (diagnostics)
SELF_TEST = True   # prepare() compares the CUDA path against an FP32 reference and falls back on mismatch

E_LOC = 32
D_HIDDEN = 4096
D_EXPERT = 4096
TOP_K = 8
POST_NORM_EPS = 1e-8
NWARP = 16
META_HDR = 128
KS2_MAX = 4

# Bit positions (m0, m1, m2, sign) of element i (k offset i within the lane's 8) in a 32-bit word.
# Device decode: o0 = w & MASK; o1 = rotr(w,3) & MASK; o2 = rotr(w,6) & MASK;
#                o3 = ((w<<6)&0x00C000C0) | ((w<<5)&0x01000100) | ((w<<11)&0x80008000);
# o_j = bf16x2 {element 2j (lo), element 2j+1 (hi)} = e2m1 * 2^-126 each.
BITPOS = ((6, 7, 8, 15), (22, 23, 24, 31), (9, 10, 11, 18), (25, 26, 27, 2),
          (12, 13, 14, 21), (28, 29, 30, 5), (0, 1, 3, 4), (16, 17, 19, 20))
SCALE_BF16_BIAS = 0x7580   # bf16 bits of (sf * 2^118) = b' * 16 + 0x7580
W_SHIFT = 256.0            # dequantized weights are w_true * 2^-8


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


# ----------------------------------------------------------------------------------------------
# Lossless repack (device-agnostic torch; also imported by k2a_emulate.py)
# ----------------------------------------------------------------------------------------------

def row_perm(n_rows: int, gate_up: bool, device=None) -> torch.Tensor:
    """prepacked row m -> original row."""
    m = torch.arange(n_rows, device=device)
    if not gate_up:
        return m
    rt, r = m // 32, m % 32
    return (r // 16) * (n_rows // 2) + 16 * rt + (r % 16)


def reencode_scales(sf_u8: torch.Tensor):
    """E4M3 bytes -> (b' uint8, negative mask, zero mask, has_nan). b' = (e+10)<<3 | f for |sf| = 2^e (1+f/8);
    every finite E4M3 magnitude (normal and subnormal) is exact. Zero scales get b' = 24 (their codes are
    zeroed), negative scales have their codes' signs flipped by the caller."""
    b = sf_u8.to(torch.int32)
    sign, ex, m = b >> 7, (b >> 3) & 15, b & 7
    nan = (ex == 15) & (m == 7)
    zero = (ex == 0) & (m == 0)
    sub = (ex == 0) & (m != 0)
    lut_e5 = torch.tensor([0, 1, 2, 2, 3, 3, 3, 3], dtype=torch.int32, device=b.device)
    lut_f = torch.tensor([0, 0, 0, 4, 0, 2, 4, 6], dtype=torch.int32, device=b.device)
    bp = ((ex + 3) << 3) | m
    bp = torch.where(sub, (lut_e5[m.long()] << 3) | lut_f[m.long()], bp)
    bp = torch.where(zero, torch.full_like(bp, 24), bp)
    return bp.to(torch.uint8), (sign == 1) & ~zero, zero, bool(nan.any())


def pack_words(codes: torch.Tensor) -> torch.Tensor:
    """codes [..., 8] (0..15) -> int32 [...] per BITPOS."""
    c = codes.to(torch.int64)
    w = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for i, (p0, p1, p2, ps) in enumerate(BITPOS):
        ci = c[..., i]
        w |= ((ci & 1) << p0) | (((ci >> 1) & 1) << p1) | (((ci >> 2) & 1) << p2) | (((ci >> 3) & 1) << ps)
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w)
    return w.to(torch.int32)


def prepack_expert(w: torch.Tensor, sf: torch.Tensor, gate_up: bool):
    """w uint8 [N, K/2] (E2M1, low nibble first), sf e4m3 [N, K/16] ->
    (wq int32 [N/32, K/32, 32, 4], sq uint8 [N/32, K/32, 64])."""
    n, k2 = w.shape
    k = 2 * k2
    nrt, kbn = n // 32, k // 32
    sfb = sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf
    bp, neg, zero, bad = reencode_scales(sfb)
    if bad:
        raise ValueError("NaN block scale in NVFP4 weights")
    codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(n, k)
    if bool(neg.any()) or bool(zero.any()):
        cb = codes.view(n, k // 16, 16)
        cb = torch.where(neg.unsqueeze(-1), cb ^ 8, cb)
        cb = torch.where(zero.unsqueeze(-1), torch.zeros_like(cb), cb)
        codes = cb.reshape(n, k)
    perm = row_perm(n, gate_up, device=w.device)
    codes = codes[perm]
    bp = bp[perm]
    c = codes.view(nrt, 2, 2, 8, kbn, 4, 8).permute(0, 4, 3, 5, 1, 2, 6).reshape(nrt, kbn, 32, 4, 8)
    wq = pack_words(c)
    sq = bp.view(nrt, 2, 2, 8, kbn, 2).permute(0, 4, 5, 3, 1, 2).reshape(nrt, kbn, 64)
    return wq.contiguous(), sq.contiguous()


def ng_for_bs(bs: int) -> int:
    """Max 8-token groups per chunk (tokens per chunk = 8*NG); static per shape."""
    return 1 if bs <= 16 else (2 if bs <= 64 else 4)


# ----------------------------------------------------------------------------------------------
# CUDA
# ----------------------------------------------------------------------------------------------

CPP_SRC = r"""
#include <torch/extension.h>
void route(torch::Tensor ids, torch::Tensor meta, int64_t M, int64_t lo, int64_t G);
void gemm(int64_t mode, int64_t ng, torch::Tensor wq, torch::Tensor sq, torch::Tensor gsc, torch::Tensor x,
          torch::Tensor meta, torch::Tensor out, int64_t M, int64_t G);
void finalize(torch::Tensor y, torch::Tensor meta, torch::Tensor ids, torch::Tensor wts, torch::Tensor nw,
              torch::Tensor out, int64_t M, int64_t lo);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace k2a {

constexpr int KDIM = 4096;      // K of both GEMMs
constexpr int KBLK = 128;       // k32 blocks per row
constexpr int NWARP = 16;
constexpr int NTHR = NWARP * 32;
constexpr int META_HDR = 128;
constexpr int RING = 4;         // items in flight per warp (each item: 16 B weights + 4 B scales per lane)

__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }

// One packed word (8 E2M1 codes) -> 4 bf16x2 holding e2m1 * 2^-126 (sign-magnitude bit placement).
__device__ __forceinline__ void deq4(uint32_t w, uint32_t o[4]) {
  o[0] = w & 0x81C081C0u;
  o[1] = rotr(w, 3) & 0x81C081C0u;
  o[2] = rotr(w, 6) & 0x81C081C0u;
  o[3] = ((w << 6) & 0x00C000C0u) | ((w << 5) & 0x01000100u) | ((w << 11) & 0x80008000u);
}

// bf16x2 multiply (exact here: e2m1*sf has <= 6 significant bits); a*b + (-0) == a*b.
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------------------------
// Routing. meta: [0]=n_act [1]=KS2 [2]=routes, [8+a]=local expert of active a, [40+a]=rows, [72+a]=first
// compact row; [128 + r] = token of compact row r (r < 8M); [128 + 8M + p] = compact row of pair p or -1.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ ids, int* __restrict__ meta,
                                                     int M, int lo, int G) {
  __shared__ int s_cnt[32];
  __shared__ int s_off[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;   // == local expert handled in passes 1 and 2
  int* act_rows = meta + META_HDR;
  int* pair2row = meta + META_HDR + 8 * M;

  {
    int cnt = 0;
    for (int base = 0; base < M; base += 32) {
      const int t = base + lane;
      bool hit = false;
      if (t < M) {
#pragma unroll
        for (int s = 0; s < 8; ++s) hit |= (ids[t * 8 + s] - lo == warp);
      }
      cnt += __popc(__ballot_sync(0xffffffffu, hit));
    }
    if (lane == 0) s_cnt[warp] = cnt;
  }
  for (int p = threadIdx.x; p < 8 * M; p += blockDim.x) {
    const int e = ids[p] - lo;
    if (e < 0 || e >= 32) pair2row[p] = -1;
  }
  __syncthreads();
  if (warp == 0) {
    const int c = s_cnt[lane];
    const unsigned act = __ballot_sync(0xffffffffu, c > 0);
    const int a = __popc(act & ((1u << lane) - 1u));
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int nb = __shfl_up_sync(0xffffffffu, v, d);
      if (lane >= d) v += nb;
    }
    const int off = v - c;
    s_off[lane] = off;
    if (c > 0) {
      meta[8 + a] = lane;
      meta[40 + a] = c;
      meta[72 + a] = off;
    }
    if (lane == 31) {
      const int n_act = __popc(act);
      const int u1 = n_act * 128;   // W2 row tiles
      const int ks = (u1 >= 4 * G) ? 1 : ((2 * u1 >= 4 * G) ? 2 : 4);
      meta[0] = n_act;
      meta[1] = ks;
      meta[2] = v;
    }
  }
  __syncthreads();
  if (s_cnt[warp] > 0) {
    int pos = s_off[warp];
    for (int base = 0; base < M; base += 32) {
      const int t = base + lane;
      int slot = -1;
      if (t < M) {
#pragma unroll
        for (int s = 0; s < 8; ++s)
          if (ids[t * 8 + s] - lo == warp) slot = s;
      }
      const unsigned b = __ballot_sync(0xffffffffu, slot >= 0);
      if (slot >= 0) {
        const int r = pos + __popc(b & ((1u << lane) - 1u));
        act_rows[r] = t;
        pair2row[t * 8 + slot] = r;
      }
      pos += __popc(b);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Grouped swap-AB GEMM. Unit u = (active expert a, row tile rt, k-split ks), a-major, ks fastest.
// CTA b takes units b, b+G, ...; each unit loops over its expert's token chunks (8*NG tokens).
// Warp w covers k32 blocks [ks*(128/KS) + w*NKB, +NKB), NKB = 8/KS. An "item" is one k32 block.
// ---------------------------------------------------------------------------------------------
struct Cur {
  int u, c, kb, a, rt, ks, nch;
};

__device__ __forceinline__ void cur_unit(Cur& q, int u, int U, int NRT, int KS, const int* s_cnt, int TCH) {
  q.u = u;
  q.c = 0;
  q.kb = 0;
  if (u < U) {
    const int per = NRT * KS;
    q.a = u / per;
    const int rem = u - q.a * per;
    q.rt = rem / KS;
    q.ks = rem - q.rt * KS;
    q.nch = (s_cnt[q.a] + TCH - 1) / TCH;
  }
}

__device__ __forceinline__ void cur_next(Cur& q, int NKB, int G, int U, int NRT, int KS, const int* s_cnt,
                                         int TCH) {
  if (++q.kb == NKB) {
    q.kb = 0;
    if (++q.c == q.nch) cur_unit(q, q.u + G, U, NRT, KS, s_cnt, TCH);
  }
}

template <int MODE, int NG>
__global__ void __launch_bounds__(NTHR, 1)
gemm_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sq, const float* __restrict__ gsc,
            const __nv_bfloat16* __restrict__ X, const int* __restrict__ meta,
            __nv_bfloat16* __restrict__ out_bf, float* __restrict__ out_f, int M) {
  constexpr int NRT = (MODE == 1) ? 256 : 128;
  constexpr int TCH = 8 * NG;
  constexpr int RED_W = 8 * NG;     // floats per lane per warp slab: (rho, group, i)
  __shared__ float red[8 * RED_W * 32];
  __shared__ int s_e[32], s_cnt[32], s_off[32];
  __shared__ int s_hdr[2];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;

  if (tid < 32) {
    s_e[tid] = meta[8 + tid];
    s_cnt[tid] = meta[40 + tid];
    s_off[tid] = meta[72 + tid];
  }
  if (tid == 0) {
    s_hdr[0] = meta[0];
    s_hdr[1] = (MODE == 1) ? 1 : meta[1];
  }
  __syncthreads();
  const int n_act = s_hdr[0];
  const int KS = s_hdr[1];
  const int NKB = 8 / KS;
  const int KPS = KBLK / KS;        // k32 blocks per k-split
  const int U = n_act * NRT * KS;
  const int G = gridDim.x;
  const int R_cap = 8 * M;
  const int* act_rows = meta + META_HDR;

  Cur pc, cc;
  cur_unit(pc, blockIdx.x, U, NRT, KS, s_cnt, TCH);
  cc = pc;

  uint4 wr[RING];
  uint32_t sr[RING];
#pragma unroll
  for (int d = 0; d < RING; ++d) {
    if (pc.u < U) {
      const int kbg = pc.ks * KPS + warp * NKB + pc.kb;
      const size_t blk = ((size_t)s_e[pc.a] * NRT + pc.rt) * KBLK + kbg;
      wr[d] = __ldg(wq + blk * 32 + lane);
      sr[d] = __ldg(sq + blk * 16 + (t >> 1) * 8 + g);
      cur_next(pc, NKB, G, U, NRT, KS, s_cnt, TCH);
    } else {
      wr[d] = make_uint4(0u, 0u, 0u, 0u);
      sr[d] = 0u;
    }
  }

  float acc[2][NG][4];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int q = 0; q < NG; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][q][i] = 0.f;
  int xrow[NG];
#pragma unroll
  for (int q = 0; q < NG; ++q) xrow[q] = 0;
  int ng = 0;

  while (cc.u < U) {
#pragma unroll
    for (int d = 0; d < RING; ++d) {
      if (cc.u < U) {
        const uint4 w4 = wr[d];
        const uint32_t sw = sr[d];
        if (pc.u < U) {   // refill this ring slot with the item RING ahead
          const int kbg = pc.ks * KPS + warp * NKB + pc.kb;
          const size_t blk = ((size_t)s_e[pc.a] * NRT + pc.rt) * KBLK + kbg;
          wr[d] = __ldg(wq + blk * 32 + lane);
          sr[d] = __ldg(sq + blk * 16 + (t >> 1) * 8 + g);
          cur_next(pc, NKB, G, U, NRT, KS, s_cnt, TCH);
        }
        const int T = s_cnt[cc.a];
        const int n0 = cc.c * TCH;
        if (cc.kb == 0) {   // new (unit, chunk): token rows of this lane's B column
          ng = min(NG, (T - n0 + 7) >> 3);
#pragma unroll
          for (int q = 0; q < NG; ++q) {
            int n = n0 + q * 8 + g;
            if (n >= T) n = T - 1;          // padded column: any valid row, result discarded
            const int crow = s_off[cc.a] + n;
            xrow[q] = (MODE == 1) ? act_rows[crow] : crow;
          }
        }
        const int kel = (cc.ks * KPS + warp * NKB + cc.kb) * 32 + t * 8;
        uint4 xv[NG];
#pragma unroll
        for (int q = 0; q < NG; ++q) {
          if (q < ng) xv[q] = __ldg(reinterpret_cast<const uint4*>(X + (size_t)xrow[q] * KDIM + kel));
          else xv[q] = make_uint4(0u, 0u, 0u, 0u);
        }
        uint32_t S[4];
#pragma unroll
        for (int bi = 0; bi < 4; ++bi) S[bi] = __byte_perm(sw, 0u, 0x4040u + 0x0101u * bi) * 16u + 0x75807580u;
        const uint32_t wv[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
        for (int rho = 0; rho < 2; ++rho) {
          uint32_t o0[4], o1[4];
          deq4(wv[2 * rho], o0);
          deq4(wv[2 * rho + 1], o1);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            o0[j] = bmul2(o0[j], S[2 * rho]);
            o1[j] = bmul2(o1[j], S[2 * rho + 1]);
          }
#pragma unroll
          for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int q = 0; q < NG; ++q) {
              if (q < ng)
                mma16816(acc[rho][q], o0[2 * s], o1[2 * s], o0[2 * s + 1], o1[2 * s + 1],
                         s ? xv[q].z : xv[q].x, s ? xv[q].w : xv[q].y);
            }
          }
        }
        if (cc.kb == NKB - 1) {   // (unit, chunk) done: reduce the 16 warps' K-slices, epilogue
          if (warp >= 8) {
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
              for (int q = 0; q < NG; ++q)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                  red[((warp - 8) * RED_W + (r * NG + q) * 4 + i) * 32 + lane] = acc[r][q][i];
          }
          __syncthreads();
          if (warp < 8) {
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
              for (int q = 0; q < NG; ++q)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                  red[(warp * RED_W + (r * NG + q) * 4 + i) * 32 + lane] += acc[r][q][i];
          }
          __syncthreads();
          const int crow0 = s_off[cc.a] + n0;
          const float gs = gsc[s_e[cc.a]];
          if (MODE == 1) {
            const int n = tid >> 4, rr = tid & 15;
            if (n < 8 * ng && n0 + n < T) {
              const int gg = rr & 7, hi = rr >> 3, q = n >> 3, tt = (n & 7) >> 1, i = 2 * hi + (n & 1);
              const int ln = gg * 4 + tt;
              float gate = 0.f, up = 0.f;
#pragma unroll
              for (int w8 = 0; w8 < 8; ++w8) {
                gate += red[(w8 * RED_W + (0 * NG + q) * 4 + i) * 32 + ln];
                up += red[(w8 * RED_W + (1 * NG + q) * 4 + i) * 32 + ln];
              }
              const float hg = gate * gs, hu = up * gs;
              const float v = hg / (1.f + __expf(-hg)) * hu;
              out_bf[(size_t)(crow0 + n) * KDIM + cc.rt * 16 + rr] = __float2bfloat16_rn(v);
            }
          } else {
            for (int idx = tid; idx < 32 * 8 * ng; idx += NTHR) {
              const int n = idx >> 5, r = idx & 31;
              if (n0 + n < T) {
                const int rho = r >> 4, rr = r & 15, gg = rr & 7, hi = rr >> 3, q = n >> 3, tt = (n & 7) >> 1;
                const int i = 2 * hi + (n & 1), ln = gg * 4 + tt;
                float sum = 0.f;
#pragma unroll
                for (int w8 = 0; w8 < 8; ++w8) sum += red[(w8 * RED_W + (rho * NG + q) * 4 + i) * 32 + ln];
                out_f[((size_t)cc.ks * R_cap + crow0 + n) * KDIM + cc.rt * 32 + r] = sum * gs;
              }
            }
          }
          __syncthreads();
#pragma unroll
          for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int q = 0; q < NG; ++q)
#pragma unroll
              for (int i = 0; i < 4; ++i) acc[r][q][i] = 0.f;
        }
        cur_next(cc, NKB, G, U, NRT, KS, s_cnt, TCH);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Post-expert norm + weighted sum. One CTA (256 threads) per token, 16 columns per thread.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) final_kernel(const float* __restrict__ Y, const int* __restrict__ meta,
                                                    const int* __restrict__ ids, const float* __restrict__ wts,
                                                    const float* __restrict__ nw, __nv_bfloat16* __restrict__ out,
                                                    int M, int lo) {
  __shared__ float s_red[2][8];
  const int tok = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int KS = meta[1];
  const int R_cap = 8 * M;
  const int* pair2row = meta + META_HDR + 8 * M;
  float acc[16], nwv[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float4 v = reinterpret_cast<const float4*>(nw)[j * 256 + tid];
    nwv[4 * j + 0] = v.x; nwv[4 * j + 1] = v.y; nwv[4 * j + 2] = v.z; nwv[4 * j + 3] = v.w;
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.f;
  int par = 0;
  for (int s = 0; s < 8; ++s) {
    const int e = ids[tok * 8 + s] - lo;
    if (e < 0 || e >= 32) continue;   // uniform over the block
    const int crow = pair2row[tok * 8 + s];
    float f[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) f[i] = 0.f;
    for (int ks = 0; ks < KS; ++ks) {
      const float4* yp = reinterpret_cast<const float4*>(Y + ((size_t)ks * R_cap + crow) * KDIM);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float4 v = yp[j * 256 + tid];
        f[4 * j + 0] += v.x; f[4 * j + 1] += v.y; f[4 * j + 2] += v.z; f[4 * j + 3] += v.w;
      }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float yb = __bfloat162float(__float2bfloat16_rn(f[i]));   // SGLang's y is BF16
      f[i] = yb * nwv[i];
      ss += f[i] * f[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) s_red[par][warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) tot += s_red[par][w];
    par ^= 1;
    const float r = rsqrtf(tot * (1.f / 4096.f) + 1e-8f) * wts[tok * 8 + s];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] += f[i] * r;
  }
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(out + (size_t)tok * KDIM);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c2 = (j * 256 + tid) * 2;   // bf16x2 index of column 4*(j*256+tid)
    op[c2] = __floats2bfloat162_rn(acc[4 * j + 0], acc[4 * j + 1]);
    op[c2 + 1] = __floats2bfloat162_rn(acc[4 * j + 2], acc[4 * j + 3]);
  }
}

}  // namespace k2a

void route(torch::Tensor ids, torch::Tensor meta, int64_t M, int64_t lo, int64_t G) {
  auto st = at::cuda::getCurrentCUDAStream();
  k2a::route_kernel<<<1, 1024, 0, st>>>(ids.data_ptr<int>(), meta.data_ptr<int>(), (int)M, (int)lo, (int)G);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MODE, int NG>
static void launch_gemm(torch::Tensor& wq, torch::Tensor& sq, torch::Tensor& gsc, torch::Tensor& x,
                        torch::Tensor& meta, torch::Tensor& out, int M, int G) {
  auto st = at::cuda::getCurrentCUDAStream();
  const uint4* wqp = reinterpret_cast<const uint4*>(wq.data_ptr());
  const uint32_t* sqp = reinterpret_cast<const uint32_t*>(sq.data_ptr());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* ob = (MODE == 1) ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr;
  float* of = (MODE == 2) ? reinterpret_cast<float*>(out.data_ptr()) : nullptr;
  k2a::gemm_kernel<MODE, NG><<<G, k2a::NTHR, 0, st>>>(wqp, sqp, gsc.data_ptr<float>(), xp, meta.data_ptr<int>(),
                                                     ob, of, M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm(int64_t mode, int64_t ng, torch::Tensor wq, torch::Tensor sq, torch::Tensor gsc, torch::Tensor x,
          torch::Tensor meta, torch::Tensor out, int64_t M, int64_t G) {
  TORCH_CHECK(mode == 1 || mode == 2, "mode");
  TORCH_CHECK(ng == 1 || ng == 2 || ng == 4, "ng");
  if (mode == 1) {
    if (ng == 1) launch_gemm<1, 1>(wq, sq, gsc, x, meta, out, (int)M, (int)G);
    else if (ng == 2) launch_gemm<1, 2>(wq, sq, gsc, x, meta, out, (int)M, (int)G);
    else launch_gemm<1, 4>(wq, sq, gsc, x, meta, out, (int)M, (int)G);
  } else {
    if (ng == 1) launch_gemm<2, 1>(wq, sq, gsc, x, meta, out, (int)M, (int)G);
    else if (ng == 2) launch_gemm<2, 2>(wq, sq, gsc, x, meta, out, (int)M, (int)G);
    else launch_gemm<2, 4>(wq, sq, gsc, x, meta, out, (int)M, (int)G);
  }
}

void finalize(torch::Tensor y, torch::Tensor meta, torch::Tensor ids, torch::Tensor wts, torch::Tensor nw,
              torch::Tensor out, int64_t M, int64_t lo) {
  auto st = at::cuda::getCurrentCUDAStream();
  k2a::final_kernel<<<(unsigned)M, 256, 0, st>>>(
      y.data_ptr<float>(), meta.data_ptr<int>(), ids.data_ptr<int>(), wts.data_ptr<float>(), nw.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)M, (int)lo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CPP_SRC + CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"k2a_v0_{tag}",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["route", "gemm", "finalize"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-Xptxas=-v"],
        verbose=True,
    )
    print(f"[k2a] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


# ----------------------------------------------------------------------------------------------
# PyTorch paths: fallback (graph-safe, slow) and an FP32 reference for prepare()'s self-test
# ----------------------------------------------------------------------------------------------

_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_GRIDS = {}


def _grid(device):
    # cached per device (filled eagerly in prepare), so the fallback stays graph-capturable
    device = torch.device(device)
    if device not in _GRIDS:
        _GRIDS[device] = torch.tensor(_GRID, dtype=torch.float32, device=device)
    return _GRIDS[device]


def _dequant(packed, sf, gs):
    grid = _grid(packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)
    return vals.view(*vals.shape[:-2], k)


def _torch_path(data, w):
    """Every local expert on every token, weighted by its routing weight (graph-capturable)."""
    cfg = data["config"]
    x = data["hidden_states"]
    ids = data["topk_ids"].long()
    lo, cnt = cfg["local_start"], cfg["local_count"]
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    nw = w["post_norm_w"].float() + 1.0
    for e in range(lo, lo + cnt):
        g = (data["topk_weights"].float() * (ids == e)).sum(-1)
        w13 = _dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).to(torch.bfloat16)
        h = (x @ w13.T).float()
        inter = (torch.nn.functional.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]).to(torch.bfloat16)
        w2 = _dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).to(torch.bfloat16)
        y = (inter @ w2.T).float() * nw
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        out += g[:, None] * y
    return out.to(torch.bfloat16)


def _ref_fp32(data, w):
    cfg = data["config"]
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    lo, hi = cfg["local_start"], cfg["local_start"] + cfg["local_count"]
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not lo <= e < hi:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]
        y = (inter @ _dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].float().unsqueeze(1))
    return out.to(torch.bfloat16)


def _rel(a, b):
    a, b = a.float(), b.float()
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row


def _self_test_inputs(bs, seed, device, skew=False):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=device)
    if skew:
        scores[:, :32] += torch.rand((32,), generator=gen, device=device).pow(6) * 4
    else:
        scores[:, :32] += 0.15   # make local experts busier, like the judge's "busiest" rotation
    top = scores.topk(TOP_K, dim=-1)
    return {
        "hidden_states": x,
        "topk_ids": top.indices.to(torch.int32),
        "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": D_HIDDEN, "d_expert": D_EXPERT, "n_experts": 256,
                   "local_start": 0, "local_count": E_LOC, "top_k": TOP_K, "sharding": "ep8"},
    }


# ----------------------------------------------------------------------------------------------
# prepare / kernel_fn
# ----------------------------------------------------------------------------------------------

def prepare(weights: Weights):
    dev = weights["w13"].device
    t0 = time.time()
    state = {"weights": weights, "mod": None}
    _grid(dev)
    try:
        w13q = torch.empty((E_LOC, 256, 128, 32, 4), dtype=torch.int32, device=dev)
        s13q = torch.empty((E_LOC, 256, 128, 64), dtype=torch.uint8, device=dev)
        w2q = torch.empty((E_LOC, 128, 128, 32, 4), dtype=torch.int32, device=dev)
        s2q = torch.empty((E_LOC, 128, 128, 64), dtype=torch.uint8, device=dev)
        for e in range(E_LOC):
            w13q[e], s13q[e] = prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True)
            w2q[e], s2q[e] = prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False)
        state.update(
            w13q=w13q, s13q=s13q, w2q=w2q, s2q=s2q,
            gsc13=(weights["w13_gs"][:E_LOC].float() * W_SHIFT).contiguous(),
            gsc2=(weights["w2_gs"][:E_LOC].float() * W_SHIFT).contiguous(),
            nw=(weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous(),
            sms=torch.cuda.get_device_properties(dev).multi_processor_count,
        )
        print(f"[k2a] repacked in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[k2a] repack failed, using PyTorch fallback: {ex!r}", flush=True)
        return state
    if FALLBACK:
        print("[k2a] FALLBACK=True: PyTorch path", flush=True)
        return state
    try:
        state["mod"] = _build()
    except Exception as ex:  # noqa: BLE001
        msg = str(ex)
        print("[k2a] CUDA build FAILED, using PyTorch fallback. Compiler output tail:\n" + msg[-12000:], flush=True)
        state["mod"] = None
        return state
    if SELF_TEST:
        try:
            cases = ((1, 11, False), (5, 12, False), (40, 13, False), (100, 14, False), (128, 15, True), (257, 16, False))
            for bs, seed, skew in cases:
                d = _self_test_inputs(bs, seed, dev, skew)
                got = kernel_fn(d, state)
                torch.cuda.synchronize()
                tot, row = _rel(got, _ref_fp32(d, weights))
                print(f"[k2a] self-test bs={bs} skew={skew}: rel {tot:.3e} worst row {row:.3e}", flush=True)
                if not (tot < 1.0e-2 and row < 5.0e-2) or not bool(torch.isfinite(got.float()).all()):
                    raise RuntimeError(f"self-test mismatch at bs={bs}")
        except Exception as ex:  # noqa: BLE001
            print(f"[k2a] self-test FAILED, using PyTorch fallback: {ex!r}", flush=True)
            state["mod"] = None
    print(f"[k2a] prepare done in {time.time() - t0:.1f}s", flush=True)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    mod = state["mod"]
    if mod is None or cfg["local_count"] != E_LOC or cfg["local_start"] != 0 \
            or cfg["d_hidden"] != D_HIDDEN or cfg["d_expert"] != D_EXPERT or cfg["top_k"] != TOP_K:
        return _torch_path(data, state["weights"])
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    dev = x.device
    r_cap = 8 * m
    sms = state["sms"]
    ng = ng_for_bs(m)
    meta = torch.empty((META_HDR + 16 * m,), dtype=torch.int32, device=dev)
    inter = torch.empty((r_cap, D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((KS2_MAX, r_cap, D_HIDDEN), dtype=torch.float32, device=dev)
    out = torch.empty((m, D_HIDDEN), dtype=torch.bfloat16, device=dev)
    lo = cfg["local_start"]
    mod.route(ids, meta, m, lo, sms)
    mod.gemm(1, ng, state["w13q"], state["s13q"], state["gsc13"], x, meta, inter, m, sms)
    mod.gemm(2, ng, state["w2q"], state["s2q"], state["gsc2"], inter, meta, y, m, sms)
    mod.finalize(y, meta, ids, wts, state["nw"], out, m, lo)
    return out

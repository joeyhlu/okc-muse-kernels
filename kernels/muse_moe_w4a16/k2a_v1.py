# okc: problem=muse_moe_w4a16
# nonce k2a-v1-1
"""K2a v1: custom NVFP4 W4A16 MoE experts for decode/verify (sm_90, mma.sync m16n8k16, swap-AB).

Same repacked layout, dequant and numerics as v0 (k2a_v0.py, check ff4ae9cda6f34ffc: correct, ~1.9 TB/s).
v1 changes, aimed at v0's measured streaming ceiling (see k2a_design.md "v1"):
  * X (the B operand) is prefetched XR items ahead in registers instead of being loaded at use, and weights
    are loaded with ld.global.nc.L1::no_allocate so the per-warp X slices stay L1-resident.
  * Deeper weight ring (8/6/4 items per warp for NG = 1/2/4).
  * Contiguous, cost-balanced unit ranges per CTA (consecutive units share the expert, so X stays hot).
  * NG=4 (verify): 4 row tiles per CTA unit share X (KW=4 K-split instead of 16), 4x fewer X bytes per
    weight byte through L1/L2.
  * W2 at NG=1: 2 CTAs/SM x 8 warps; KS2 in {1, 2} so each warp does >= 8 items between reductions.
  * route stages ids in smem (v0: 9-16 us at bs 128/256); final issues all slot loads before one block
    reduction (v0: 6.3 us at bs 1).
Pipeline per call (4 launches, fixed grids, CUDA-graph capturable): route -> gemm W13 (SiLU*up, BF16 inter)
-> gemm W2 (FP32 partials per k-split) -> final (RMSNorm(y*(1+w)) * topk weight, summed per token).

Layouts (checked bit-exactly by k2a_emulate.py):
  WQ[e][rt][kb][lane][4 x u32]: row tile rt (32 prepacked rows), k32 block kb, lane = 4g+t; word wi = 2*rho+h
     holds the 8 E2M1 codes of tile row 16*rho+8*h+g at k = 32*kb+8*t+i (i=0..7), bits per BITPOS.
  SQ[e][rt][kb][64 B]: byte gi*32 + g*4 + 2*rho+h = scale of row 16*rho+8*h+g, group 2*kb+gi, re-encoded
     losslessly so that bf16_bits = b*16 + 0x7580 = sf * 2^118.
  W13 prepacked row m = 32*rt + 16*rho + i  <->  original row rho*4096 + 16*rt + i (gate/up pairs per tile).
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
META_HDR = 128
KS2_MAX = 2

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
# CUDA. KERNEL_SRC is shared verbatim with kernels/lab/../k2a_lab_v1.py; the host dispatch is generated
# from a variant list (gen_host), so the lab and the submission instantiate the same template.
# ----------------------------------------------------------------------------------------------

# Variant knobs (template parameters of k2a::Cfg):
#   ng    8-token groups per pass (tokens per chunk = 8*ng)
#   ring  weight/scale register ring depth (items in flight per warp; item = 16 B weights + 4 B scales per lane)
#   xr    X (B operand) prefetch depth in items (0 = load at use, as v0)
#   na    1 = weights via ld.global.nc.L1::no_allocate (keeps X resident in L1), 0 = __ldg
#   asg   0 = round-robin units (v0), 1 = contiguous cost-balanced unit ranges per CTA
#   nw    warps per CTA; kw = warps that split one row tile's K (rwg = nw/kw row tiles per CTA unit)
#   probe 0 = real kernel, 1 = stream only (no X, no dequant/MMA/epilogue), 2 = no X loads (B = 1.0)
#   minb  __launch_bounds__ min blocks per SM (grid = SMs * minb)
VKEYS = ("mode", "ng", "ring", "xr", "na", "asg", "nw", "kw", "probe", "minb")

KERNEL_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

namespace k2a {

constexpr int KDIM = 4096;      // K of both GEMMs
constexpr int KBLK = 128;       // k32 blocks per row
constexpr int META_HDR = 128;

template <int NG_, int RING_, int XR_, int NA_, int ASG_, int NW_, int KW_, int PROBE_, int MINB_>
struct Cfg {
  static constexpr int NG = NG_;
  static constexpr int RING = RING_;
  static constexpr int XR = XR_;
  static constexpr int NA = NA_;
  static constexpr int ASG = ASG_;
  static constexpr int NW = NW_;
  static constexpr int KW = KW_;
  static constexpr int PROBE = PROBE_;
  static constexpr int MINB = MINB_;
  static constexpr int NTHR = NW_ * 32;
  static constexpr int RWG = NW_ / KW_;
};

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

template <int NA>
__device__ __forceinline__ uint4 ldw4(const uint4* p) {
  if constexpr (NA != 0) {
    int4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return make_uint4((uint32_t)r.x, (uint32_t)r.y, (uint32_t)r.z, (uint32_t)r.w);
  } else {
    return __ldg(p);
  }
}

template <int NA>
__device__ __forceinline__ uint32_t ldw1(const uint32_t* p) {
  if constexpr (NA != 0) {
    int r;
    asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];\n" : "=r"(r) : "l"(p));
    return (uint32_t)r;
  } else {
    return __ldg(p);
  }
}

// ---------------------------------------------------------------------------------------------
// Routing. meta: [0]=n_act [1]=KS2 [2]=routes, [8+a]=local expert of active a, [40+a]=rows, [72+a]=first
// compact row; [128 + r] = token of compact row r (r < 8M); [128 + 8M + p] = compact row of pair p or -1.
// v1: ids staged in smem (M <= 512), KS2 in {1, 2} chosen from the W2 variant's units per expert.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ ids, int* __restrict__ meta,
                                                     int M, int lo, int SMS, int W2UPE) {
  __shared__ int s_ids[4096];
  __shared__ int s_cnt[32];
  __shared__ int s_off[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;   // == local expert handled in passes 1 and 2
  int* act_rows = meta + META_HDR;
  int* pair2row = meta + META_HDR + 8 * M;
  const bool staged = (M <= 512);
  if (staged) {
    for (int p = threadIdx.x; p < 8 * M; p += blockDim.x) s_ids[p] = ids[p];
  }
  __syncthreads();
  const int* src = staged ? s_ids : ids;

  {
    int cnt = 0;
    for (int base = 0; base < M; base += 32) {
      const int t = base + lane;
      bool hit = false;
      if (t < M) {
#pragma unroll
        for (int s = 0; s < 8; ++s) hit |= (src[t * 8 + s] - lo == warp);
      }
      cnt += __popc(__ballot_sync(0xffffffffu, hit));
    }
    if (lane == 0) s_cnt[warp] = cnt;
  }
  for (int p = threadIdx.x; p < 8 * M; p += blockDim.x) {
    const int e = src[p] - lo;
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
      meta[0] = n_act;
      meta[1] = (n_act * W2UPE >= 3 * SMS) ? 1 : 2;   // W2 split-K across CTAs (W2UPE: W2 units/expert)
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
          if (src[t * 8 + s] - lo == warp) slot = s;
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
// Grouped swap-AB GEMM. CTA unit u = (active expert a, row-tile group rg of RWG tiles, k-split ks), a-major,
// ks fastest. Warp w -> (rw = w / KW, kw = w % KW): row tile rt = rg*RWG + rw, k32 blocks
// [ks*KPS + kw*NKB, +NKB). Each unit loops over its expert's token chunks (8*NG tokens) inside.
// ---------------------------------------------------------------------------------------------
struct Sched {
  int ue, ustep, NRG, KS, NKB, TCH;
  const int* cnt;
};

struct Cur {
  int u, c, kb, a, rg, ks, nch;
};

__device__ __forceinline__ void cur_unit(Cur& q, int u, const Sched& S) {
  q.u = u;
  q.c = 0;
  q.kb = 0;
  if (u < S.ue) {
    const int per = S.NRG * S.KS;
    q.a = u / per;
    const int rem = u - q.a * per;
    q.rg = rem / S.KS;
    q.ks = rem - q.rg * S.KS;
    q.nch = (S.cnt[q.a] + S.TCH - 1) / S.TCH;
  }
}

__device__ __forceinline__ void cur_next(Cur& q, const Sched& S) {
  if (++q.kb == S.NKB) {
    q.kb = 0;
    if (++q.c == q.nch) cur_unit(q, q.u + S.ustep, S);
  }
}

// Unit cost for contiguous assignment: chunk re-passes re-dequantize (weights come from L1/L2).
__device__ __forceinline__ long long unit_cost(int cnt, int TCH) { return 4 + 3 * ((cnt + TCH - 1) / TCH - 1); }

// First unit whose cost offset is >= x (units of expert a start at base_a + j*cost_a).
__device__ int first_unit(long long x, int n_act, int UPE, const int* cnt, int TCH) {
  long long base = 0;
  for (int a = 0; a < n_act; ++a) {
    const long long cost = unit_cost(cnt[a], TCH);
    const long long span = cost * UPE;
    if (base + span > x) {
      const long long j = (x > base) ? (x - base + cost - 1) / cost : 0;
      if (j < UPE) return a * UPE + (int)j;
    }
    base += span;
  }
  return n_act * UPE;
}

template <int MODE, int NG>
__device__ __forceinline__ void x_fetch(const Cur& q, int KPS, int NKB, int kw, int g, int t, const int* s_cnt,
                                        const int* s_off, const int* act_rows, const __nv_bfloat16* X,
                                        int (&xrow)[NG], int& ngv, uint4 (&dst)[NG]) {
  constexpr int TCH = 8 * NG;
  if (q.kb == 0) {
    const int T = s_cnt[q.a];
    const int n0 = q.c * TCH;
    ngv = min(NG, (T - n0 + 7) >> 3);
#pragma unroll
    for (int j = 0; j < NG; ++j) {
      int n = n0 + j * 8 + g;
      if (n >= T) n = T - 1;             // padded column: any valid row, result discarded
      const int crow = s_off[q.a] + n;
      xrow[j] = (MODE == 1) ? act_rows[crow] : crow;
    }
  }
  const int kel = (q.ks * KPS + kw * NKB + q.kb) * 32 + t * 8;
#pragma unroll
  for (int j = 0; j < NG; ++j) {
    if (j < ngv) dst[j] = __ldg(reinterpret_cast<const uint4*>(X + (size_t)xrow[j] * KDIM + kel));
  }
}

template <int MODE, class C>
__global__ void __launch_bounds__(C::NTHR, C::MINB)
gemm_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sq, const float* __restrict__ gsc,
            const __nv_bfloat16* __restrict__ X, const int* __restrict__ meta,
            __nv_bfloat16* __restrict__ out_bf, float* __restrict__ out_f, int M) {
  constexpr int NG = C::NG, RING = C::RING, XR = C::XR, NW = C::NW, KW = C::KW, RWG = C::RWG;
  constexpr int NTHR = C::NTHR, PROBE = C::PROBE;
  constexpr int NRT = (MODE == 1) ? 256 : 128;
  constexpr int NRG = NRT / RWG;
  constexpr int TCH = 8 * NG;
  constexpr int RED_W = 8 * NG;              // floats per lane per warp slab: (rho, group, i)
  constexpr int H2 = (KW >= 2) ? KW / 2 : 1;
  constexpr bool USE_RED = (KW >= 2) && (PROBE != 1);
  constexpr int RED_N = USE_RED ? (NW / 2) * RED_W * 32 : 1;
  constexpr int XRS = (XR > 0) ? XR : 1;
  static_assert(NRT % RWG == 0, "row tiles per CTA unit");
  static_assert(XR == 0 || RING % XR == 0, "x ring must divide the weight ring");
  static_assert(KW == 1 || KW % 2 == 0, "kw");
  __shared__ float red[RED_N];
  __shared__ int s_e[32], s_cnt[32], s_off[32];
  __shared__ int s_hdr[4];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int rw = warp / KW;
  const int kw = warp - rw * KW;

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
  const int KPS = KBLK / KS;        // k32 blocks per k-split
  const int NKB = KPS / KW;         // k32 blocks per warp per unit
  const int UPE = NRG * KS;         // units per expert
  const int U = n_act * UPE;
  const int G = gridDim.x;
  const int R_cap = 8 * M;
  const int* act_rows = meta + META_HDR;

  Sched S;
  S.NRG = NRG;
  S.KS = KS;
  S.NKB = NKB;
  S.TCH = TCH;
  S.cnt = s_cnt;
  int ub;
  if constexpr (C::ASG == 1) {
    if (tid == 0) {
      long long tot = 0;
      for (int a = 0; a < n_act; ++a) tot += unit_cost(s_cnt[a], TCH) * UPE;
      s_hdr[2] = first_unit(tot * (long long)blockIdx.x / G, n_act, UPE, s_cnt, TCH);
      s_hdr[3] = first_unit(tot * (long long)(blockIdx.x + 1) / G, n_act, UPE, s_cnt, TCH);
    }
    __syncthreads();
    ub = s_hdr[2];
    S.ue = s_hdr[3];
    S.ustep = 1;
  } else {
    ub = blockIdx.x;
    S.ue = U;
    S.ustep = G;
  }

  Cur pc, cc, xc;
  cur_unit(pc, ub, S);
  cc = pc;
  xc = pc;

  uint4 wr[RING];
  uint32_t sr[RING];
#pragma unroll
  for (int d = 0; d < RING; ++d) {
    wr[d] = make_uint4(0u, 0u, 0u, 0u);
    sr[d] = 0u;
    if (pc.u < S.ue) {
      const size_t blk = ((size_t)s_e[pc.a] * NRT + (pc.rg * RWG + rw)) * KBLK + (pc.ks * KPS + kw * NKB + pc.kb);
      wr[d] = ldw4<C::NA>(wq + blk * 32 + lane);
      sr[d] = ldw1<C::NA>(sq + blk * 16 + (t >> 1) * 8 + g);
      cur_next(pc, S);
    }
  }

  uint4 xr[XRS][NG];
  int xrow_x[NG];
  int ngx = 0;
#pragma unroll
  for (int d = 0; d < XRS; ++d)
#pragma unroll
    for (int q = 0; q < NG; ++q) xr[d][q] = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int q = 0; q < NG; ++q) xrow_x[q] = 0;
  if constexpr (XR > 0 && PROBE == 0) {
#pragma unroll
    for (int d = 0; d < XR; ++d) {
      if (xc.u < S.ue) {
        x_fetch<MODE, NG>(xc, KPS, NKB, kw, g, t, s_cnt, s_off, act_rows, X, xrow_x, ngx, xr[d]);
        cur_next(xc, S);
      }
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
  uint32_t sink = 0u;

  while (cc.u < S.ue) {
#pragma unroll
    for (int d = 0; d < RING; ++d) {
      if (cc.u < S.ue) {
        const uint4 w4 = wr[d];
        const uint32_t sw = sr[d];
        if (pc.u < S.ue) {   // refill this ring slot with the item RING ahead
          const size_t blk =
              ((size_t)s_e[pc.a] * NRT + (pc.rg * RWG + rw)) * KBLK + (pc.ks * KPS + kw * NKB + pc.kb);
          wr[d] = ldw4<C::NA>(wq + blk * 32 + lane);
          sr[d] = ldw1<C::NA>(sq + blk * 16 + (t >> 1) * 8 + g);
          cur_next(pc, S);
        }
        if constexpr (PROBE == 1) {
          sink ^= w4.x ^ w4.y ^ w4.z ^ w4.w ^ sw;
        } else {
          const int T = s_cnt[cc.a];
          const int n0 = cc.c * TCH;
          if (cc.kb == 0) ng = min(NG, (T - n0 + 7) >> 3);
          uint4 xv[NG];
#pragma unroll
          for (int q = 0; q < NG; ++q) xv[q] = make_uint4(0u, 0u, 0u, 0u);
          if constexpr (PROBE == 2) {
#pragma unroll
            for (int q = 0; q < NG; ++q) xv[q] = make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u);
          } else if constexpr (XR > 0) {
            const int sl = d % XRS;
#pragma unroll
            for (int q = 0; q < NG; ++q) xv[q] = xr[sl][q];
            if (xc.u < S.ue) {   // refill with the X item XR ahead
              x_fetch<MODE, NG>(xc, KPS, NKB, kw, g, t, s_cnt, s_off, act_rows, X, xrow_x, ngx, xr[sl]);
              cur_next(xc, S);
            }
          } else {
            x_fetch<MODE, NG>(cc, KPS, NKB, kw, g, t, s_cnt, s_off, act_rows, X, xrow, ng, xv);
          }
          uint32_t Sc[4];
#pragma unroll
          for (int bi = 0; bi < 4; ++bi) Sc[bi] = __byte_perm(sw, 0u, 0x4040u + 0x0101u * bi) * 16u + 0x75807580u;
          const uint32_t wv[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
          for (int rho = 0; rho < 2; ++rho) {
            uint32_t o0[4], o1[4];
            deq4(wv[2 * rho], o0);
            deq4(wv[2 * rho + 1], o1);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              o0[j] = bmul2(o0[j], Sc[2 * rho]);
              o1[j] = bmul2(o1[j], Sc[2 * rho + 1]);
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
          if (cc.kb == NKB - 1) {   // (unit, chunk) done
            const int crow0 = s_off[cc.a] + n0;
            const float gs = gsc[s_e[cc.a]];
            if constexpr (KW == 1) {   // each warp owns full K of its row tile: epilogue from registers
              const int rt = cc.rg * RWG + rw;
#pragma unroll
              for (int q = 0; q < NG; ++q) {
                if (q < ng) {
#pragma unroll
                  for (int i = 0; i < 4; ++i) {
                    const int n = q * 8 + 2 * t + (i & 1);
                    const int rr = g + 8 * (i >> 1);
                    if (n0 + n < T) {
                      if constexpr (MODE == 1) {
                        const float hg = acc[0][q][i] * gs, hu = acc[1][q][i] * gs;
                        out_bf[(size_t)(crow0 + n) * KDIM + rt * 16 + rr] =
                            __float2bfloat16_rn(hg / (1.f + __expf(-hg)) * hu);
                      } else {
#pragma unroll
                        for (int r = 0; r < 2; ++r)
                          out_f[((size_t)cc.ks * R_cap + crow0 + n) * KDIM + rt * 32 + r * 16 + rr] = acc[r][q][i] * gs;
                      }
                    }
                  }
                }
              }
            } else {   // reduce the KW warps of each row tile in smem: upper half stores, lower half adds
              if (kw >= H2) {
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                  for (int q = 0; q < NG; ++q)
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                      red[((rw * H2 + (kw - H2)) * RED_W + (r * NG + q) * 4 + i) * 32 + lane] = acc[r][q][i];
              }
              __syncthreads();
              if (kw < H2) {
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                  for (int q = 0; q < NG; ++q)
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                      red[((rw * H2 + kw) * RED_W + (r * NG + q) * 4 + i) * 32 + lane] += acc[r][q][i];
              }
              __syncthreads();
              if constexpr (MODE == 1) {
                const int per = 16 * 8 * ng;
                for (int o = tid; o < RWG * per; o += NTHR) {
                  const int ro = o / per;
                  const int rem = o - ro * per;
                  const int n = rem >> 4, rr = rem & 15;
                  if (n0 + n < T) {
                    const int gg = rr & 7, hi = rr >> 3, q = n >> 3, tt = (n & 7) >> 1, i = 2 * hi + (n & 1);
                    const int ln = gg * 4 + tt;
                    float gate = 0.f, up = 0.f;
#pragma unroll
                    for (int k2 = 0; k2 < H2; ++k2) {
                      gate += red[((ro * H2 + k2) * RED_W + (0 * NG + q) * 4 + i) * 32 + ln];
                      up += red[((ro * H2 + k2) * RED_W + (1 * NG + q) * 4 + i) * 32 + ln];
                    }
                    const float hg = gate * gs, hu = up * gs;
                    out_bf[(size_t)(crow0 + n) * KDIM + (cc.rg * RWG + ro) * 16 + rr] =
                        __float2bfloat16_rn(hg / (1.f + __expf(-hg)) * hu);
                  }
                }
              } else {
                const int per = 32 * 8 * ng;
                for (int o = tid; o < RWG * per; o += NTHR) {
                  const int ro = o / per;
                  const int rem = o - ro * per;
                  const int n = rem >> 5, r = rem & 31;
                  if (n0 + n < T) {
                    const int rho = r >> 4, rr = r & 15, gg = rr & 7, hi = rr >> 3, q = n >> 3, tt = (n & 7) >> 1;
                    const int i = 2 * hi + (n & 1), ln = gg * 4 + tt;
                    float sum = 0.f;
#pragma unroll
                    for (int k2 = 0; k2 < H2; ++k2) sum += red[((ro * H2 + k2) * RED_W + (rho * NG + q) * 4 + i) * 32 + ln];
                    out_f[((size_t)cc.ks * R_cap + crow0 + n) * KDIM + (cc.rg * RWG + ro) * 32 + r] = sum * gs;
                  }
                }
              }
              __syncthreads();
            }
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
              for (int q = 0; q < NG; ++q)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[r][q][i] = 0.f;
          }
        }
        cur_next(cc, S);
      }
    }
  }
  if constexpr (PROBE == 1) {   // keep the stream loads alive; never true in practice
    if (sink == 0x9E3779B9u) {
      if constexpr (MODE == 1) out_bf[tid] = __float2bfloat16_rn(1.f);
      else out_f[tid] = 1.f;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Post-expert norm + weighted sum. One CTA (512 threads) per token, 8 columns per thread
// (cols 4*tid..+3 and 2048+4*tid..+3). All local slots' loads are issued before one block reduction.
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(512) final_kernel(const float* __restrict__ Y, const int* __restrict__ meta,
                                                    const int* __restrict__ ids, const float* __restrict__ wts,
                                                    const float* __restrict__ nw, __nv_bfloat16* __restrict__ out,
                                                    int M, int lo) {
  __shared__ float s_red[16][8];
  const int tok = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int KS = meta[1];
  const int R_cap = 8 * M;
  const int* pair2row = meta + META_HDR + 8 * M;
  float nwv[8];
  {
    const float4 a = reinterpret_cast<const float4*>(nw)[tid];
    const float4 b = reinterpret_cast<const float4*>(nw)[512 + tid];
    nwv[0] = a.x; nwv[1] = a.y; nwv[2] = a.z; nwv[3] = a.w;
    nwv[4] = b.x; nwv[5] = b.y; nwv[6] = b.z; nwv[7] = b.w;
  }
  unsigned loc = 0u;
  int crow[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int e = ids[tok * 8 + s] - lo;
    crow[s] = 0;
    if (e >= 0 && e < 32) {
      loc |= 1u << s;
      crow[s] = pair2row[tok * 8 + s];
    }
  }
  float f[8][8];
  float ss[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
#pragma unroll
    for (int i = 0; i < 8; ++i) f[s][i] = 0.f;
    if (loc & (1u << s)) {
      for (int ks = 0; ks < KS; ++ks) {
        const float4* yp = reinterpret_cast<const float4*>(Y + ((size_t)ks * R_cap + crow[s]) * KDIM);
        const float4 a = yp[tid];
        const float4 b = yp[512 + tid];
        f[s][0] += a.x; f[s][1] += a.y; f[s][2] += a.z; f[s][3] += a.w;
        f[s][4] += b.x; f[s][5] += b.y; f[s][6] += b.z; f[s][7] += b.w;
      }
    }
    ss[s] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float yb = __bfloat162float(__float2bfloat16_rn(f[s][i]));   // SGLang's y is BF16
      f[s][i] = yb * nwv[i];
      ss[s] += f[s][i] * f[s][i];
    }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[s] += __shfl_xor_sync(0xffffffffu, ss[s], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < 8; ++s) s_red[warp][s] = ss[s];
  }
  __syncthreads();
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    if (loc & (1u << s)) {
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < 16; ++w) tot += s_red[w][s];
      const float r = rsqrtf(tot * (1.f / 4096.f) + 1e-8f) * wts[tok * 8 + s];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] += f[s][i] * r;
    }
  }
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(out + (size_t)tok * KDIM);
  op[2 * tid] = __floats2bfloat162_rn(acc[0], acc[1]);
  op[2 * tid + 1] = __floats2bfloat162_rn(acc[2], acc[3]);
  op[1024 + 2 * tid] = __floats2bfloat162_rn(acc[4], acc[5]);
  op[1024 + 2 * tid + 1] = __floats2bfloat162_rn(acc[6], acc[7]);
}

}  // namespace k2a

void route(torch::Tensor ids, torch::Tensor meta, int64_t M, int64_t lo, int64_t SMS, int64_t W2UPE) {
  auto st = at::cuda::getCurrentCUDAStream();
  k2a::route_kernel<<<1, 1024, 0, st>>>(ids.data_ptr<int>(), meta.data_ptr<int>(), (int)M, (int)lo, (int)SMS,
                                        (int)W2UPE);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MODE, class C>
static void launch_gemm(torch::Tensor& wq, torch::Tensor& sq, torch::Tensor& gsc, torch::Tensor& x,
                        torch::Tensor& meta, torch::Tensor& out, int M, int G) {
  auto st = at::cuda::getCurrentCUDAStream();
  const uint4* wqp = reinterpret_cast<const uint4*>(wq.data_ptr());
  const uint32_t* sqp = reinterpret_cast<const uint32_t*>(sq.data_ptr());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* ob = (MODE == 1) ? reinterpret_cast<__nv_bfloat16*>(out.data_ptr()) : nullptr;
  float* of = (MODE == 2) ? reinterpret_cast<float*>(out.data_ptr()) : nullptr;
  k2a::gemm_kernel<MODE, C><<<G, C::NTHR, 0, st>>>(wqp, sqp, gsc.data_ptr<float>(), xp, meta.data_ptr<int>(),
                                                  ob, of, M);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MODE, class C>
static std::vector<int64_t> attrs_gemm() {
  cudaFuncAttributes fa;
  cudaError_t err = cudaFuncGetAttributes(&fa, k2a::gemm_kernel<MODE, C>);
  if (err != cudaSuccess) return std::vector<int64_t>{-1, -1, -1, -1};
  return std::vector<int64_t>{(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)fa.sharedSizeBytes,
                              (int64_t)fa.maxThreadsPerBlock};
}

void finalize(torch::Tensor y, torch::Tensor meta, torch::Tensor ids, torch::Tensor wts, torch::Tensor nw,
              torch::Tensor out, int64_t M, int64_t lo) {
  auto st = at::cuda::getCurrentCUDAStream();
  k2a::final_kernel<<<(unsigned)M, 512, 0, st>>>(
      y.data_ptr<float>(), meta.data_ptr<int>(), ids.data_ptr<int>(), wts.data_ptr<float>(), nw.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)M, (int)lo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

HOST_TEMPLATE = r"""
void gemm(int64_t vid, torch::Tensor wq, torch::Tensor sq, torch::Tensor gsc, torch::Tensor x, torch::Tensor meta,
          torch::Tensor out, int64_t M, int64_t G) {
  switch (vid) {
@CASES@
    default: TORCH_CHECK(false, "k2a: unknown gemm variant");
  }
}

std::vector<int64_t> gemm_attrs(int64_t vid) {
  switch (vid) {
@ATTRS@
    default: return std::vector<int64_t>{-1, -1, -1, -1};
  }
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void route(torch::Tensor ids, torch::Tensor meta, int64_t M, int64_t lo, int64_t SMS, int64_t W2UPE);
void gemm(int64_t vid, torch::Tensor wq, torch::Tensor sq, torch::Tensor gsc, torch::Tensor x, torch::Tensor meta,
          torch::Tensor out, int64_t M, int64_t G);
std::vector<int64_t> gemm_attrs(int64_t vid);
void finalize(torch::Tensor y, torch::Tensor meta, torch::Tensor ids, torch::Tensor wts, torch::Tensor nw,
              torch::Tensor out, int64_t M, int64_t lo);
"""


def gen_host(variants):
    """C++ dispatch for a list of variant dicts (keys VKEYS); case index = list index."""
    cases, attrs = [], []
    for i, v in enumerate(variants):
        cfg = "k2a::Cfg<%d, %d, %d, %d, %d, %d, %d, %d, %d>" % tuple(v[k] for k in VKEYS[1:])
        cases.append("    case %d: launch_gemm<%d, %s >(wq, sq, gsc, x, meta, out, (int)M, (int)G); break;"
                     % (i, v["mode"], cfg))
        attrs.append("    case %d: return attrs_gemm<%d, %s >();" % (i, v["mode"], cfg))
    return HOST_TEMPLATE.replace("@CASES@", "\n".join(cases)).replace("@ATTRS@", "\n".join(attrs))


def w2_upe(v):
    """W2 units per expert at KS2 = 1 for a W2 variant (128 row tiles / row tiles per CTA unit)."""
    return 128 // (v["nw"] // v["kw"])


def variant(mode, ng, ring, xr, na, asg, nw, kw, probe=0, minb=None):
    if minb is None:
        minb = 1 if nw == 16 else 2
    return {"mode": mode, "ng": ng, "ring": ring, "xr": xr, "na": na, "asg": asg, "nw": nw, "kw": kw,
            "probe": probe, "minb": minb}


def build_ext(name, variants, verbose=False):
    from torch.utils.cpp_extension import load_inline
    src = KERNEL_SRC + gen_host(variants)
    tag = hashlib.sha1((CPP_SRC + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"{name}_{tag}",
        cpp_sources=CPP_SRC,
        cuda_sources=src,
        functions=["route", "gemm", "gemm_attrs", "finalize"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=verbose,
    )
    return mod, time.time() - t0


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
# v1 kernel configuration per (mode, NG). Lab: kernels/muse_moe_w4a16/k2a_lab_v1.py sweeps the alternatives.
# ----------------------------------------------------------------------------------------------

V1_CFG = {
    (1, 1): variant(1, 1, 8, 2, 1, 1, 16, 16),
    (2, 1): variant(2, 1, 8, 2, 1, 1, 8, 8),
    (1, 2): variant(1, 2, 6, 2, 1, 1, 16, 16),
    (2, 2): variant(2, 2, 6, 2, 1, 1, 16, 16),
    (1, 4): variant(1, 4, 4, 1, 1, 1, 16, 4),
    (2, 4): variant(2, 4, 4, 1, 1, 1, 16, 4),
}
V1_KEYS = sorted(V1_CFG)
V1_VARIANTS = [V1_CFG[k] for k in V1_KEYS]
V1_VID = {k: i for i, k in enumerate(V1_KEYS)}


def _build():
    mod, dt = build_ext("k2a_v1", V1_VARIANTS)
    print(f"[k2a] compiled in {dt:.1f}s", flush=True)
    for k in V1_KEYS:
        a = mod.gemm_attrs(V1_VID[k])
        print(f"[k2a] gemm mode={k[0]} ng={k[1]}: regs {a[0]} spill(local) {a[1]} B smem {a[2]} B", flush=True)
    return mod


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
    v1, v2 = V1_CFG[(1, ng)], V1_CFG[(2, ng)]
    meta = torch.empty((META_HDR + 16 * m,), dtype=torch.int32, device=dev)
    inter = torch.empty((r_cap, D_EXPERT), dtype=torch.bfloat16, device=dev)
    y = torch.empty((KS2_MAX, r_cap, D_HIDDEN), dtype=torch.float32, device=dev)
    out = torch.empty((m, D_HIDDEN), dtype=torch.bfloat16, device=dev)
    lo = cfg["local_start"]
    mod.route(ids, meta, m, lo, sms, w2_upe(v2))
    mod.gemm(V1_VID[(1, ng)], state["w13q"], state["s13q"], state["gsc13"], x, meta, inter, m, sms * v1["minb"])
    mod.gemm(V1_VID[(2, ng)], state["w2q"], state["s2q"], state["gsc2"], inter, meta, y, m, sms * v2["minb"])
    mod.finalize(y, meta, ids, wts, state["nw"], out, m, lo)
    return out

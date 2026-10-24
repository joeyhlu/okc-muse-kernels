# okc: problem=muse_moe_w4a16
# nonce k2a-lab-v2-1
"""K2a v2 LAB (check runs only; never submit for score). Generated from k2a_v2.py by gen_lab.py: everything up to
the submission's prepare/kernel_fn is copied verbatim (same CUDA source, same repack, same incumbent path), then the
lab section replaces prepare/kernel_fn. kernel_fn is the incumbent (r1_glue_v3 path), so the check passes; prepare()
times the v2 template variants (CTAs/SM, prefetch distance, token chunking NC / token-group warps TW) and runtime
knobs (fuse 0/1/2 launches, KS target = split-K, round-robin vs cost-balanced units, forced atomics, stream-only /
dequant-only probes) at the six scored shapes with CUDA-graph replay and a cold L2, against an in-file incumbent row,
plus an L2->SM rate probe and a hot-expert dedup probe (skewed bs128). Summary block printed last (<16 KB tail).
"""
import math


import hashlib
import time
from typing import TypedDict

import torch

try:  # incumbent (r1_glue_v3) imports; absent on a CPU-only box (the emulator imports the repack only)
    import types
    import triton
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    _HAVE_SGLANG = True
except Exception:  # noqa: BLE001
    _HAVE_SGLANG = False

try:
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        _post_expert_norm_weighted_sum_kernel as _POSTNORM_KERNEL,
    )
except Exception:  # noqa: BLE001
    _POSTNORM_KERNEL = None

# ---------------------------------------------------------------------------------------------- switches
FALLBACK = False        # True: never build v2 (incumbent everywhere)
SELF_TEST = True        # prepare() checks v2 against an FP32 reference; any failure disables v2
AUTOTUNE = True         # prepare() times v2 vs the incumbent per bucket and may switch a bucket to v2
AUTOTUNE_MARGIN = 1.03  # v2 must be this much faster (median of paired replays) to take a bucket
AUTOTUNE_REPS = 15

E_LOC = 32
D_HIDDEN = 4096
D_EXPERT = 4096
TOP_K = 8
POST_NORM_EPS = 1e-8
M_MAX = 320             # in-kernel routing lists are sized for 8 * M_MAX pairs (tests go to 257)
META_HDR = 128

BITPOS = ((6, 7, 8, 15), (22, 23, 24, 31), (9, 10, 11, 18), (25, 26, 27, 2),
          (12, 13, 14, 21), (28, 29, 30, 5), (0, 1, 3, 4), (16, 17, 19, 20))
W_SHIFT = 256.0         # dequantized weights are w_true * 2^-8; the FP32 global scale carries the 2^8

# Variant table (template parameters of k2a2::Cfg<NC, U, PF, TW, XE, MINB>):
#   NC   8-token chunks per warp per pass (accumulators 4*NC regs)
#   U    16 B weight loads per thread per group (U items of 512 B per warp)
#   PF   0 = issue a group's loads, then consume (stream_lab D winner); 1 = prefetch the next group first
#   TW   token-group warps per row tile (they share one weight fetch through L1); RW = 8 / TW row tiles per unit
#   XE   bf16 elements of the staged X slice (k per slice = XE / (TW * 8 * NC))
#   MINB __launch_bounds__ min CTAs per SM (4 -> <= 64 regs, 3 -> <= 80)
VKEYS = ("nc", "u", "pf", "tw", "xe", "minb")


def variant(nc, u=4, pf=0, tw=1, xe=8192, minb=4):
    return {"nc": nc, "u": u, "pf": pf, "tw": tw, "xe": xe, "minb": minb}


V2_VARIANTS = [
    variant(1, 4, 0, 1, 8192, 4),     # 0: small T (bs <= 24): 8 tokens / pass
    variant(2, 4, 0, 1, 16384, 4),    # 1: bs 25-48: 16 tokens / pass
    variant(4, 4, 0, 1, 16384, 3),    # 2: verify (bs >= 49): 32 tokens / pass, <= 80 regs
]
# Per bucket: (lo, hi, representative scored bs). Decisions are taken per bucket (nearby shapes route alike).
BUCKETS = ((1, 4, 1), (5, 12, 8), (13, 24, 16), (25, 48, 32), (49, 192, 128), (193, M_MAX, 256))
# v2 plan per bucket: (variant index, fuse, ks_target, asg). Lab (k2a_lab_v2.py) decides; these are the design
# defaults: 2 launches, KS chosen so units >= 2 x grid, round-robin units (concurrent passes -> L2 dedup).
V2_PLAN = {1: (0, 1, 2, 0), 8: (0, 1, 2, 0), 16: (0, 1, 2, 0), 32: (1, 1, 2, 0), 128: (2, 1, 2, 0), 256: (2, 1, 2, 0)}
# Static route table: "inc" (r1_glue_v3 path) or "v2". Default incumbent everywhere until measured (R2).
ROUTE = {1: "inc", 8: "inc", 16: "inc", 32: "inc", 128: "inc", 256: "inc"}


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


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
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


# ----------------------------------------------------------------------------------------------
# Lossless repack (device-agnostic torch; imported by k2a_v2_emulate.py, so the emulated bytes are these)
# ----------------------------------------------------------------------------------------------

def reencode_scales(sf_u8: torch.Tensor):
    """E4M3 bytes -> (b' uint8, negative mask, zero mask, has_nan). b' = (e+10)<<3 | f for |sf| = 2^e (1+f/8);
    every finite E4M3 magnitude (normal and subnormal) is exact. Zero scales get b' = 24 (their codes are
    zeroed), negative scales have their codes' signs flipped by the caller. (Unchanged from v0/v1.)"""
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


def tile_row_perm(n_rows: int, gate_up: bool, device=None) -> torch.Tensor:
    """prepacked row m (= 16*tile + rho) -> original row. W13: rho < 8 -> gate 8*tile + rho, rho >= 8 -> up."""
    m = torch.arange(n_rows, device=device)
    if not gate_up:
        return m
    tile, rho = m // 16, m % 16
    return (rho // 8) * (n_rows // 2) + 8 * tile + (rho % 8)


def prepack_expert(w: torch.Tensor, sf: torch.Tensor, gate_up: bool):
    """w uint8 [N, K/2] (E2M1, low nibble first), sf e4m3 [N, K/16] ->
    (wq int32 [N/16, K/64, 32, 4], sq uint8 [N/16, K/128, 32, 4])."""
    n, k2 = w.shape
    k = 2 * k2
    rt, nkb = n // 16, k // 64
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
    perm = tile_row_perm(n, gate_up, device=w.device)
    codes = codes[perm]
    bp = bp[perm]
    # codes[16*r + 8*hi + g][64*kb + 16*t + 8*half + i] -> (r, kb, g, t, hi, half, i)
    c = codes.view(rt, 2, 8, nkb, 4, 2, 8).permute(0, 3, 2, 4, 1, 5, 6).reshape(rt, nkb, 32, 4, 8)
    wq = pack_words(c)
    # bp[16*r + 8*hi + g][group 8*kp + 4*par + t] -> (r, kp, g, t, par, hi): byte 2*par + hi
    sq = bp.view(rt, 2, 8, nkb // 2, 2, 4).permute(0, 3, 2, 5, 4, 1).reshape(rt, nkb // 2, 32, 4)
    return wq.contiguous(), sq.contiguous()


def bucket_of(m: int):
    for lo, hi, rep in BUCKETS:
        if lo <= m <= hi:
            return rep
    return None


# ----------------------------------------------------------------------------------------------
# CUDA (shared verbatim with k2a_lab_v2.py; the host dispatch is generated from a variant list)
# ----------------------------------------------------------------------------------------------

KERNEL_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

namespace k2a2 {

constexpr int KDIM = 4096;          // K of both GEMMs, N of W2
constexpr int NKB = 64;             // k64 items per tile row
constexpr int M_MAX = 320;
constexpr int P_MAX = 8 * M_MAX;    // (token, slot) pairs
constexpr int MH = 128;             // meta header ints; meta[MH + p] = compact row of pair p (or -1)
constexpr int PH_ROUTE = 1, PH_G1 = 2, PH_G2 = 4, PH_FIN = 8;
constexpr unsigned FULL = 0xffffffffu;
constexpr long long SPIN_BUDGET = 1ll << 26;   // clock64 cycles (~34 ms at 1.98 GHz) per grid barrier

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint4* w13q; const uint32_t* s13q; const uint4* w2q; const uint32_t* s2q;
  const float* gsc13; const float* gsc2; const float* nw;
  float* h; __nv_bfloat16* inter; float* y; int* meta; unsigned* bar; int* err;
  __nv_bfloat16* out;
  int M, lo, mask, ks_target, asg, force_atom, probe;
};

template <int NC_, int U_, int PF_, int TW_, int XE_, int MINB_>
struct Cfg {
  static constexpr int NC = NC_, U = U_, PF = PF_, TW = TW_, XE = XE_, MINB = MINB_;
  static constexpr int TPW = 8 * NC;           // tokens per warp per pass
  static constexpr int TPP = TW * TPW;         // tokens per pass
  static constexpr int RW = 8 / TW;            // row tiles per CTA unit
  static constexpr int KSW = XE / TPP;         // k per staged X slice
  static constexpr int XSTR = KSW + 8;         // padded smem row stride (bf16): 16 B pad -> conflict-free LDS.128
  static constexpr int XS = TPP * XSTR;
  static_assert(TW == 1 || TW == 2 || TW == 4 || TW == 8, "TW");
  static_assert(U == 2 || U == 4, "U");
  static_assert(KSW >= 64 * U && KSW % (64 * U) == 0, "a slice holds whole load groups");
};

struct Route {
  int cnt[32], off[32], np[32];
  int ub[33];
  long long cb[33];
  int wc[8][32];
  int hdr[8];        // 0 R, 1 NV, 2 KS1, 3 KS2, 4 n_act, 5 atom1, 6 atom2
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }

// One packed word (8 E2M1 codes) -> 4 bf16x2 {element 2j (lo), 2j+1 (hi)} = e2m1 * 2^-126 (v0, GPU-validated).
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {
  o[0] = w & 0x81C081C0u;
  o[1] = rotr(w, 3) & 0x81C081C0u;
  o[2] = rotr(w, 6) & 0x81C081C0u;
  o[3] = ((w << 6) & 0x00C000C0u) | ((w << 5) & 0x01000100u) | ((w << 11) & 0x80008000u);
}

// bf16x2 multiply, exact here (e2m1 * sf has <= 6 significant bits): a*b + (-0) == a*b.
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

// Weights: never written by any kernel -> .nc. TW == 1: L1::no_allocate (stream); TW > 1: allocate in L1 so
// the TW warps that read the same tile share one fetch.
template <int TW>
__device__ __forceinline__ uint4 ldw4(const uint4* p) {
  if constexpr (TW == 1) {
    int4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return make_uint4((uint32_t)r.x, (uint32_t)r.y, (uint32_t)r.z, (uint32_t)r.w);
  } else {
    return __ldg(p);
  }
}
template <int TW>
__device__ __forceinline__ uint32_t ldw1(const uint32_t* p) {
  if constexpr (TW == 1) {
    int r;
    asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];\n" : "=r"(r) : "l"(p));
    return (uint32_t)r;
  } else {
    return __ldg(p);
  }
}

__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }

__device__ __forceinline__ int choose_ks(int units1, int target) {
  int ks = 1;
  while (ks < 16 && units1 * ks < target) ks <<= 1;
  return ks;
}

// ------------------------------------------------------------------------------------------------
// Grid barrier (all CTAs co-resident: grid = SMs x occupancy). Sense via a generation counter; bounded spin.
// ------------------------------------------------------------------------------------------------
__device__ void grid_barrier(const Params& P) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* vgen = P.bar + 1;
    const unsigned gen = *vgen;
    __threadfence();
    const unsigned arrived = atomicAdd(P.bar, 1u);
    if (arrived == gridDim.x - 1) {
      atomicExch(P.bar, 0u);
      __threadfence();
      atomicAdd(P.bar + 1, 1u);
    } else {
      const long long t0 = clock64();
      while (*vgen == gen) {
        if (clock64() - t0 > SPIN_BUDGET) { atomicExch(P.err, 1); break; }
        __nanosleep(64);
      }
    }
    __threadfence();
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------
// Routing, every CTA, deterministic. Compact row of pair p = off[e] + rank of p among e's pairs (p order).
// ------------------------------------------------------------------------------------------------
template <class C>
__device__ void route_prologue(const Params& P, Route& S, int8_t* s_ide, int16_t* s_list) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int NP = 8 * P.M;
  for (int i = tid; i < 8 * 32; i += 256) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  const int L = ((NP + 255) / 256) * 32;   // pairs per warp, multiple of 32
  const int p0 = warp * L;
  for (int base = 0; base < L; base += 32) {
    const int p = p0 + base + lane;
    int e = -1;
    if (p < NP) {
      const int v = P.ids[p] - P.lo;
      e = (v >= 0 && v < 32) ? v : -1;
      s_ide[p] = (int8_t)e;
    }
    if (e >= 0) atomicAdd(&S.wc[warp][e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    int c = 0;
    for (int w = 0; w < 8; ++w) { const int v = S.wc[w][lane]; S.wc[w][lane] = c; c += v; }
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    const int np = (c + C::TPP - 1) / C::TPP;
    int nv = np;
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) nv += __shfl_xor_sync(FULL, nv, d);
    S.cnt[lane] = c; S.off[lane] = off; S.np[lane] = np;
    for (int w = 0; w < 8; ++w) S.wc[w][lane] += off;
    const unsigned act = __ballot_sync(FULL, c > 0);
    if (lane == 31) {
      S.hdr[0] = v;          // R = total local routes
      S.hdr[1] = nv;         // NV = passes
      S.hdr[4] = __popc(act);
    }
  }
  __syncthreads();
  for (int base = 0; base < L; base += 32) {
    const int p = p0 + base + lane;
    const int e = (p < NP) ? (int)s_ide[p] : -1;
    const unsigned m = __match_any_sync(FULL, e);
    const int rank = __popc(m & ((1u << lane) - 1u));
    const int b = (e >= 0) ? S.wc[warp][e] : 0;
    __syncwarp();
    if (e >= 0) {
      const int c = b + rank;
      s_list[c] = (int16_t)p;
      if (blockIdx.x == 0) P.meta[MH + p] = c;
      if (rank == 0) S.wc[warp][e] = b + __popc(m);
    } else if (blockIdx.x == 0 && p < NP) {
      P.meta[MH + p] = -1;
    }
    __syncwarp();
  }
  if (tid == 0) {
    const int nv = S.hdr[1];
    const int G = (int)gridDim.x;
    const int ks1 = choose_ks(nv * (512 / C::RW), P.ks_target * G);
    const int ks2 = choose_ks(nv * (256 / C::RW), P.ks_target * G);
    S.hdr[2] = ks1; S.hdr[3] = ks2;
    S.hdr[5] = (ks1 > 1 || P.force_atom) ? 1 : 0;
    S.hdr[6] = (ks2 > 1 || P.force_atom) ? 1 : 0;
    if (blockIdx.x == 0) for (int i = 0; i < 8; ++i) P.meta[i] = S.hdr[i];
  }
  if (blockIdx.x == 0 && tid < 32) {
    P.meta[8 + tid] = S.cnt[tid]; P.meta[40 + tid] = S.off[tid]; P.meta[72 + tid] = S.np[tid];
  }
  __syncthreads();
}

__device__ void load_meta(const Params& P, Route& S) {
  const int tid = threadIdx.x;
  if (tid < 32) { S.cnt[tid] = __ldcg(P.meta + 8 + tid); S.off[tid] = __ldcg(P.meta + 40 + tid); S.np[tid] = __ldcg(P.meta + 72 + tid); }
  if (tid < 8) S.hdr[tid] = __ldcg(P.meta + tid);
  __syncthreads();
}

// Units of a GEMM phase: u -> (expert e, pass j, row group rg, k-split ks); within an expert's block the pass
// index is fastest, so passes over the same weights are adjacent (round-robin -> concurrent CTAs -> L2 hits).
template <class C, int MODE>
__device__ void phase_units(Route& S, int KS) {
  constexpr int NRG = ((MODE == 1) ? 512 : 256) / C::RW;
  if (threadIdx.x == 0) {
    int u = 0;
    long long cst = 0;
    for (int e = 0; e < 32; ++e) {
      S.ub[e] = u; S.cb[e] = cst;
      const int n = S.np[e] * NRG * KS;
      u += n;
      const int tl = min(S.cnt[e], C::TPW);
      cst += (long long)n * (4 + (tl + 7) / 8);
    }
    S.ub[32] = u; S.cb[32] = cst;
  }
  __syncthreads();
}

__device__ int first_unit(const Route& S, long long x) {
  for (int e = 0; e < 32; ++e) {
    const int n = S.ub[e + 1] - S.ub[e];
    if (n == 0) continue;
    if (S.cb[e + 1] > x) {
      const long long ce = (S.cb[e + 1] - S.cb[e]) / n;
      const long long j = (x > S.cb[e]) ? (x - S.cb[e] + ce - 1) / ce : 0;
      if (j < n) return S.ub[e] + (int)j;
    }
  }
  return S.ub[32];
}

// Stage the B operand rows of a pass for k in [k0, k0 + kw): MODE 1 x[token]; MODE 2 inter[c] (bf16) or
// SwiGLU(h[c]) (fp32 gate/up partials, KS1 > 1). 16 B pieces; rows >= ntok are not touched (their columns
// only feed discarded outputs).
template <class C, int MODE>
__device__ void stage_x(const Params& P, const int16_t* s_list, __nv_bfloat16* xs, int c0, int ntok, int k0, int kw,
                        bool from_h) {
  const int ppr = kw >> 3;
  const int sh = __ffs(ppr) - 1;     // ppr is a power of two
  const int tot = ntok * ppr;
  for (int idx = threadIdx.x; idx < tot; idx += 256) {
    const int n = idx >> sh, pc = idx & (ppr - 1);
    const int k = k0 + pc * 8;
    uint4 v;
    if constexpr (MODE == 1) {
      const int tok = ((int)s_list[c0 + n]) >> 3;
      v = __ldg(reinterpret_cast<const uint4*>(P.x + (size_t)tok * KDIM + k));
    } else {
      if (!from_h) {
        v = __ldcg(reinterpret_cast<const uint4*>(P.inter + (size_t)(c0 + n) * KDIM + k));
      } else {
        const float* hr = P.h + (size_t)(c0 + n) * (2 * KDIM);
        const float4 g0 = __ldcg(reinterpret_cast<const float4*>(hr + k));
        const float4 g1 = __ldcg(reinterpret_cast<const float4*>(hr + k + 4));
        const float4 u0 = __ldcg(reinterpret_cast<const float4*>(hr + KDIM + k));
        const float4 u1 = __ldcg(reinterpret_cast<const float4*>(hr + KDIM + k + 4));
        v.x = pack_bf2(swiglu(g0.x, u0.x), swiglu(g0.y, u0.y));
        v.y = pack_bf2(swiglu(g0.z, u0.z), swiglu(g0.w, u0.w));
        v.z = pack_bf2(swiglu(g1.x, u1.x), swiglu(g1.y, u1.y));
        v.w = pack_bf2(swiglu(g1.z, u1.z), swiglu(g1.w, u1.w));
      }
    }
    *reinterpret_cast<uint4*>(xs + n * C::XSTR + pc * 8) = v;
  }
}

template <class C>
__device__ __forceinline__ void load_group(const uint4* wt, const uint32_t* st, int kb, uint4* w, uint32_t* s) {
#pragma unroll
  for (int i = 0; i < C::U; ++i) w[i] = ldw4<C::TW>(wt + (size_t)(kb + i) * 32);
#pragma unroll
  for (int i = 0; i < C::U / 2; ++i) s[i] = ldw1<C::TW>(st + (size_t)((kb >> 1) + i) * 32);
}

// One item: 16 B = rows g, g+8 x k64 (4 k16 steps) of this warp's tile; B from smem for each valid chunk.
template <class C>
__device__ __forceinline__ void consume_item(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow,
                                             int kloc, int nch, int t, float (*acc)[4], int probe, uint32_t& sink) {
  if (probe == 1) { sink ^= w.x ^ w.y ^ w.z ^ w.w ^ sw2; return; }
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;   // row g
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;   // row g + 8
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  if (probe == 2) {
    uint32_t z = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) z ^= ag[j] ^ a8[j];
    sink ^= z;
    return;
  }
#pragma unroll
  for (int q = 0; q < C::NC; ++q) {
    if (q < nch) {
      const uint4* xp = reinterpret_cast<const uint4*>(xrow + q * 8 * C::XSTR + kloc + 16 * t);
      const uint4 x0 = xp[0], x1 = xp[1];
      mma16816(acc[q], ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
      mma16816(acc[q], ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
      mma16816(acc[q], ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
      mma16816(acc[q], ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
    }
  }
}

template <class C, int MODE>
__device__ __forceinline__ void flush(const Params& P, float (*acc)[4], int nch, int ntok, int t0w, int c0, int tile,
                                      int g, int t, float gs, bool atom) {
#pragma unroll
  for (int q = 0; q < C::NC; ++q) {
    if (q < nch) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = t0w + q * 8 + 2 * t + hh;
        if (n < ntok) {
          const size_t c = (size_t)(c0 + n);
          const float v0 = acc[q][hh] * gs;       // tile row g
          const float v1 = acc[q][2 + hh] * gs;   // tile row g + 8
          if constexpr (MODE == 1) {
            if (!atom) {
              P.inter[c * KDIM + 8 * tile + g] = __float2bfloat16_rn(swiglu(v0, v1));
            } else {
              atomicAdd(P.h + c * (2 * KDIM) + 8 * tile + g, v0);
              atomicAdd(P.h + c * (2 * KDIM) + KDIM + 8 * tile + g, v1);
            }
          } else {
            float* yp = P.y + c * KDIM + 16 * tile + g;
            if (!atom) { yp[0] = v0; yp[8] = v1; }
            else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
          }
        }
      }
    }
  }
}

template <class C, int MODE>
__device__ void gemm_phase(const Params& P, Route& S, const int16_t* s_list, __nv_bfloat16* xs) {
  constexpr int RT = (MODE == 1) ? 512 : 256;
  constexpr int U = C::U;
  const int KS = (MODE == 1) ? S.hdr[2] : S.hdr[3];
  const bool atom = ((MODE == 1) ? S.hdr[5] : S.hdr[6]) != 0;
  const bool from_h = (MODE == 2) && (S.hdr[5] != 0);
  phase_units<C, MODE>(S, KS);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rw = warp / C::TW, tw = warp - rw * C::TW;
  const int G = (int)gridDim.x;
  const int UT = S.ub[32];
  int u_beg, u_end, u_step;
  if (P.asg) {
    u_beg = first_unit(S, S.cb[32] * (long long)blockIdx.x / G);
    u_end = first_unit(S, S.cb[32] * (long long)(blockIdx.x + 1) / G);
    u_step = 1;
  } else {
    u_beg = (int)blockIdx.x; u_end = UT; u_step = G;
  }
  const int KR = KDIM / KS;
  const int KSWe = min(C::KSW, KR);
  const int NSL = KR / KSWe;
  const int NGRP = KSWe / (64 * U);
  const uint4* wq = (MODE == 1) ? P.w13q : P.w2q;
  const uint32_t* sq = (MODE == 1) ? P.s13q : P.s2q;
  const float* gsc = (MODE == 1) ? P.gsc13 : P.gsc2;
  const int t0w = tw * C::TPW;
  const __nv_bfloat16* xrow = xs + (t0w + g) * C::XSTR;
  uint32_t sink = 0u;

  for (int u = u_beg; u < u_end; u += u_step) {
    int e = 0;
    while (S.ub[e + 1] <= u) ++e;
    const int l = u - S.ub[e];
    const int npe = S.np[e];
    const int j = l % npe;
    const int r = l / npe;
    const int ks = r % KS;
    const int rg = r / KS;
    const int ntok = min(C::TPP, S.cnt[e] - j * C::TPP);
    const int c0 = S.off[e] + j * C::TPP;
    const int tile = rg * C::RW + rw;
    const int nch = max(0, min(C::NC, (ntok - t0w + 7) >> 3));
    const size_t tb = (size_t)e * RT + tile;
    const uint4* wt = wq + tb * (NKB * 32) + lane;
    const uint32_t* st = sq + tb * ((NKB / 2) * 32) + lane;
    const float gs = __ldg(gsc + e);
    float acc[C::NC][4];
#pragma unroll
    for (int q = 0; q < C::NC; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[q][i] = 0.f;

    for (int sl = 0; sl < NSL; ++sl) {
      const int k0 = ks * KR + sl * KSWe;
      const int kb0 = k0 >> 6;
      uint4 wa[U];
      uint32_t sa[U / 2];
      if (nch > 0) load_group<C>(wt, st, kb0, wa, sa);     // in flight across the staging below
      __syncthreads();                                     // previous slice / unit done with xs
      stage_x<C, MODE>(P, s_list, xs, c0, ntok, k0, KSWe, from_h);
      __syncthreads();
      if (nch > 0) {
        for (int gi = 0; gi < NGRP; ++gi) {
          if constexpr (C::PF) {
            uint4 wb[U];
            uint32_t sb[U / 2];
            if (gi + 1 < NGRP) load_group<C>(wt, st, kb0 + (gi + 1) * U, wb, sb);
#pragma unroll
            for (int i = 0; i < U; ++i)
              consume_item<C>(wa[i], sa[i >> 1], i & 1, xrow, (gi * U + i) * 64, nch, t, acc, P.probe, sink);
#pragma unroll
            for (int i = 0; i < U; ++i) wa[i] = wb[i];
#pragma unroll
            for (int i = 0; i < U / 2; ++i) sa[i] = sb[i];
          } else {
#pragma unroll
            for (int i = 0; i < U; ++i)
              consume_item<C>(wa[i], sa[i >> 1], i & 1, xrow, (gi * U + i) * 64, nch, t, acc, P.probe, sink);
            if (gi + 1 < NGRP) load_group<C>(wt, st, kb0 + (gi + 1) * U, wa, sa);
          }
        }
      }
    }
    if (nch > 0 && P.probe == 0) flush<C, MODE>(P, acc, nch, ntok, t0w, c0, tile, g, t, gs, atom);
  }
  if (sink == 0x9E3779B9u && P.probe != 0) atomicAdd(P.err + 1, 1);   // keeps probe loads alive; never true
}

// Post-expert norm + weighted sum; tokens strided over CTAs; 16 columns per thread. Zeroes h rows when the
// W13 phase used atomics (h is all-zero between calls; prepare() zeroes it once).
__device__ void fin_phase(const Params& P, const Route& S, float (*s_red)[8]) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int col = 16 * tid;
  float nwv[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(P.nw + col) + j);
    nwv[4 * j] = a.x; nwv[4 * j + 1] = a.y; nwv[4 * j + 2] = a.z; nwv[4 * j + 3] = a.w;
  }
  for (int tok = blockIdx.x; tok < P.M; tok += gridDim.x) {
    int cs[8];
    unsigned loc = 0u;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      cs[s] = __ldcg(P.meta + MH + tok * 8 + s);
      if (cs[s] >= 0) loc |= 1u << s;
    }
    float ss[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ss[s] = 0.f;
      if (loc & (1u << s)) {
        const float4* yp = reinterpret_cast<const float4*>(P.y + (size_t)cs[s] * KDIM + col);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float4 v = __ldcg(yp + j);
          const float f0 = bf16r(v.x) * nwv[4 * j], f1 = bf16r(v.y) * nwv[4 * j + 1];
          const float f2 = bf16r(v.z) * nwv[4 * j + 2], f3 = bf16r(v.w) * nwv[4 * j + 3];
          ss[s] += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
        }
      }
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss[s] += __shfl_xor_sync(FULL, ss[s], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int s = 0; s < 8; ++s) s_red[warp][s] = ss[s];
    }
    __syncthreads();
    float rs[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) tot += s_red[w][s];
      rs[s] = (loc & (1u << s)) ? rsqrtf(tot * (1.f / 4096.f) + 1e-8f) * __ldg(P.wts + tok * 8 + s) : 0.f;
    }
    __syncthreads();
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (loc & (1u << s)) {
        const float4* yp = reinterpret_cast<const float4*>(P.y + (size_t)cs[s] * KDIM + col);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float4 v = __ldcg(yp + j);
          acc[4 * j] += bf16r(v.x) * nwv[4 * j] * rs[s];
          acc[4 * j + 1] += bf16r(v.y) * nwv[4 * j + 1] * rs[s];
          acc[4 * j + 2] += bf16r(v.z) * nwv[4 * j + 2] * rs[s];
          acc[4 * j + 3] += bf16r(v.w) * nwv[4 * j + 3] * rs[s];
        }
      }
    }
    uint4 o0, o1;
    o0.x = pack_bf2(acc[0], acc[1]); o0.y = pack_bf2(acc[2], acc[3]);
    o0.z = pack_bf2(acc[4], acc[5]); o0.w = pack_bf2(acc[6], acc[7]);
    o1.x = pack_bf2(acc[8], acc[9]); o1.y = pack_bf2(acc[10], acc[11]);
    o1.z = pack_bf2(acc[12], acc[13]); o1.w = pack_bf2(acc[14], acc[15]);
    uint4* op = reinterpret_cast<uint4*>(P.out + (size_t)tok * KDIM + col);
    op[0] = o0; op[1] = o1;
  }
  if (S.hdr[5] != 0) {
    const size_t n4 = (size_t)S.hdr[0] * (2 * KDIM) / 4;
    float4* hp = reinterpret_cast<float4*>(P.h);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)blockIdx.x * 256 + tid; i < n4; i += (size_t)gridDim.x * 256) hp[i] = z;
  }
}

template <class C>
__global__ void __launch_bounds__(256, C::MINB) k2a2_kernel(const Params P) {
  __shared__ __align__(16) __nv_bfloat16 xs[C::XS];
  __shared__ int8_t s_ide[P_MAX];
  __shared__ int16_t s_list[P_MAX];
  __shared__ Route S;
  __shared__ float s_red[8][8];
  if (P.mask & PH_ROUTE) route_prologue<C>(P, S, s_ide, s_list);
  else load_meta(P, S);
  if (P.mask & PH_G1) {
    if (S.hdr[6] != 0) {   // W2 will accumulate with atomics: zero its rows now (W2 runs after this phase)
      const size_t n4 = (size_t)S.hdr[0] * KDIM / 4;
      float4* yp = reinterpret_cast<float4*>(P.y);
      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += (size_t)gridDim.x * 256) yp[i] = z;
    }
    gemm_phase<C, 1>(P, S, s_list, xs);
  }
  if ((P.mask & PH_G1) && (P.mask & (PH_G2 | PH_FIN))) grid_barrier(P);
  if (P.mask & PH_G2) gemm_phase<C, 2>(P, S, s_list, xs);
  if ((P.mask & PH_G2) && (P.mask & PH_FIN)) grid_barrier(P);
  if (P.mask & PH_FIN) fin_phase(P, S, s_red);
}

}  // namespace k2a2

static k2a2::Params make_params(torch::Tensor& x, torch::Tensor& ids, torch::Tensor& wts, torch::Tensor& w13q,
                                torch::Tensor& s13q, torch::Tensor& w2q, torch::Tensor& s2q, torch::Tensor& gsc13,
                                torch::Tensor& gsc2, torch::Tensor& nw, torch::Tensor& h, torch::Tensor& inter,
                                torch::Tensor& y, torch::Tensor& meta, torch::Tensor& bar, torch::Tensor& err,
                                torch::Tensor& out, int64_t M, int64_t lo, int64_t mask, int64_t ks_target,
                                int64_t asg, int64_t force_atom, int64_t probe) {
  k2a2::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ids = ids.data_ptr<int>();
  p.wts = wts.data_ptr<float>();
  p.w13q = reinterpret_cast<const uint4*>(w13q.data_ptr());
  p.s13q = reinterpret_cast<const uint32_t*>(s13q.data_ptr());
  p.w2q = reinterpret_cast<const uint4*>(w2q.data_ptr());
  p.s2q = reinterpret_cast<const uint32_t*>(s2q.data_ptr());
  p.gsc13 = gsc13.data_ptr<float>();
  p.gsc2 = gsc2.data_ptr<float>();
  p.nw = nw.data_ptr<float>();
  p.h = h.data_ptr<float>();
  p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  p.y = y.data_ptr<float>();
  p.meta = meta.data_ptr<int>();
  p.bar = reinterpret_cast<unsigned*>(bar.data_ptr());
  p.err = err.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.M = (int)M; p.lo = (int)lo; p.mask = (int)mask; p.ks_target = (int)ks_target; p.asg = (int)asg;
  p.force_atom = (int)force_atom; p.probe = (int)probe;
  return p;
}

template <class C>
static void launch_k(const k2a2::Params& p, int G) {
  k2a2::k2a2_kernel<C><<<G, 256, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Called in prepare (before any capture): max-shared carveout, then attributes and occupancy.
template <class C>
static std::vector<int64_t> attrs_k() {
  cudaFuncSetAttribute(k2a2::k2a2_kernel<C>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncAttributes fa;
  if (cudaFuncGetAttributes(&fa, k2a2::k2a2_kernel<C>) != cudaSuccess) return std::vector<int64_t>{-1, -1, -1, -1};
  int occ = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k2a2::k2a2_kernel<C>, 256, 0) != cudaSuccess) occ = -1;
  return std::vector<int64_t>{(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)fa.sharedSizeBytes,
                              (int64_t)occ};
}
"""

HOST_TEMPLATE = r"""
void run(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor w13q,
         torch::Tensor s13q, torch::Tensor w2q, torch::Tensor s2q, torch::Tensor gsc13, torch::Tensor gsc2,
         torch::Tensor nw, torch::Tensor h, torch::Tensor inter, torch::Tensor y, torch::Tensor meta,
         torch::Tensor bar, torch::Tensor err, torch::Tensor out, int64_t M, int64_t lo, int64_t G,
         int64_t ks_target, int64_t asg, int64_t force_atom, int64_t probe) {
  const k2a2::Params p = make_params(x, ids, wts, w13q, s13q, w2q, s2q, gsc13, gsc2, nw, h, inter, y, meta, bar, err,
                                     out, M, lo, mask, ks_target, asg, force_atom, probe);
  switch (vid) {
@CASES@
    default: TORCH_CHECK(false, "k2a2: unknown variant");
  }
}

std::vector<int64_t> attrs(int64_t vid) {
  switch (vid) {
@ATTRS@
    default: return std::vector<int64_t>{-1, -1, -1, -1};
  }
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void run(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor w13q,
         torch::Tensor s13q, torch::Tensor w2q, torch::Tensor s2q, torch::Tensor gsc13, torch::Tensor gsc2,
         torch::Tensor nw, torch::Tensor h, torch::Tensor inter, torch::Tensor y, torch::Tensor meta,
         torch::Tensor bar, torch::Tensor err, torch::Tensor out, int64_t M, int64_t lo, int64_t G,
         int64_t ks_target, int64_t asg, int64_t force_atom, int64_t probe);
std::vector<int64_t> attrs(int64_t vid);
"""


def cfg_str(v):
    return "k2a2::Cfg<%d, %d, %d, %d, %d, %d>" % tuple(v[k] for k in VKEYS)


def gen_host(variants):
    cases, attrs = [], []
    for i, v in enumerate(variants):
        cases.append("    case %d: launch_k<%s >(p, (int)G); break;" % (i, cfg_str(v)))
        attrs.append("    case %d: return attrs_k<%s >();" % (i, cfg_str(v)))
    return HOST_TEMPLATE.replace("@CASES@", "\n".join(cases)).replace("@ATTRS@", "\n".join(attrs))


def build_ext(name, variants, verbose=False):
    from torch.utils.cpp_extension import load_inline
    src = KERNEL_SRC + gen_host(variants)
    tag = hashlib.sha1((CPP_SRC + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"{name}_{tag}",
        cpp_sources=CPP_SRC,
        cuda_sources=src,
        functions=["run", "attrs"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills"],
        verbose=verbose,
    )
    return mod, time.time() - t0


PH_ROUTE, PH_G1, PH_G2, PH_FIN = 1, 2, 4, 8
FUSE_MASKS = {0: (PH_ROUTE | PH_G1, PH_G2, PH_FIN), 1: (PH_ROUTE | PH_G1, PH_G2 | PH_FIN),
              2: (PH_ROUTE | PH_G1 | PH_G2 | PH_FIN,)}


def v2_call(st, data, vid, fuse=1, ks_target=2, asg=0, force_atom=0, probe=0):
    """One v2 call (graph-capturable: fixed buffers, no host sync, no value-dependent host branching)."""
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    g = st["grid"][vid]
    lo = data["config"]["local_start"]
    for mask in FUSE_MASKS[fuse]:
        st["mod"].run(vid, mask, x, ids, wts, st["w13q"], st["s13q"], st["w2q"], st["s2q"], st["gsc13"],
                      st["gsc2"], st["nw"], st["h"], st["inter"], st["y"], st["meta"], st["bar"], st["err"], out,
                      m, lo, g, ks_target, asg, force_atom, probe)
    return out


def v2_prepare(weights, variants, name="k2a_v2", log=print):
    """Repack + build + attributes. Returns the v2 state dict (raises on failure)."""
    dev = weights["w13"].device
    t0 = time.time()
    w13q = torch.empty((E_LOC, 512, 64, 32, 4), dtype=torch.int32, device=dev)
    s13q = torch.empty((E_LOC, 512, 32, 32, 4), dtype=torch.uint8, device=dev)
    w2q = torch.empty((E_LOC, 256, 64, 32, 4), dtype=torch.int32, device=dev)
    s2q = torch.empty((E_LOC, 256, 32, 32, 4), dtype=torch.uint8, device=dev)
    for e in range(E_LOC):
        w13q[e], s13q[e] = prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True)
        w2q[e], s2q[e] = prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False)
    st = dict(
        w13q=w13q, s13q=s13q, w2q=w2q, s2q=s2q,
        gsc13=(weights["w13_gs"][:E_LOC].float() * W_SHIFT).contiguous(),
        gsc2=(weights["w2_gs"][:E_LOC].float() * W_SHIFT).contiguous(),
        nw=(weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous(),
        h=torch.zeros((8 * M_MAX, 2 * D_EXPERT), dtype=torch.float32, device=dev),
        inter=torch.zeros((8 * M_MAX, D_EXPERT), dtype=torch.bfloat16, device=dev),
        y=torch.zeros((8 * M_MAX, D_HIDDEN), dtype=torch.float32, device=dev),
        meta=torch.zeros((META_HDR + 8 * M_MAX,), dtype=torch.int32, device=dev),
        bar=torch.zeros((2,), dtype=torch.int32, device=dev),
        err=torch.zeros((4,), dtype=torch.int32, device=dev),
    )
    torch.cuda.synchronize()
    log(f"[k2a2] repacked in {time.time() - t0:.1f}s")
    mod, dt = build_ext(name, variants)
    log(f"[k2a2] compiled {len(variants)} variants in {dt:.1f}s")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = []
    for i, v in enumerate(variants):
        a = mod.attrs(i)
        occ = a[3] if a[3] > 0 else 1
        grid.append(sms * min(occ, v["minb"]) if occ >= 1 else sms)
        log(f"[k2a2] v{i} {cfg_str(v)}: regs {a[0]} spill(local) {a[1]} B smem {a[2]} B occ {a[3]} CTAs/SM "
            f"-> grid {grid[-1]}")
    st.update(mod=mod, grid=grid, sms=sms, variants=variants)
    return st


def v2_reset(st):
    """Restore the zero invariants (h all-zero, barrier counters 0) after anything that may have aborted."""
    st["h"].zero_()
    st["bar"].zero_()
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------------
# Incumbent: r1_glue_v3's path (Marlin + Triton post-norm + block-size ranges), copied verbatim
# ----------------------------------------------------------------------------------------------

N_EXPERTS = 256
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
R1_POSTNORM_TRITON_ALL_T = True
R1_EXPERT_MAP_INDEX_SELECT = True
R1_ALIGN_LOCAL_BUCKETS = True
R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS = 9
BLOCK_SIZE_M_OVERRIDE = {}
BLOCK_SIZE_M_RANGES = ((49, 1 << 30, 16),)


def _bm_override(num_tokens, default):
    if num_tokens in BLOCK_SIZE_M_OVERRIDE:
        return BLOCK_SIZE_M_OVERRIDE[num_tokens]
    for lo, hi, bm in BLOCK_SIZE_M_RANGES:
        if lo <= num_tokens <= hi:
            return bm
    return default


def _marlin_layer(weights, start, count):
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def inc_prepare(weights):
    device = weights["w13"].device
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def _post_expert_norm_weighted_sum(y, topk_weights, norm_w, slot_ids):
    if not (R1_POSTNORM_TRITON_ALL_T and _POSTNORM_KERNEL is not None):
        return fused_post_expert_norm_weighted_sum(
            y, topk_weights, norm_weight=norm_w, eps=POST_NORM_EPS, weight_first=True,
            slot_expert_ids=slot_ids)
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = triton.next_power_of_2(hidden)
    _POSTNORM_KERNEL[(tokens,)](
        y, norm_w, topk_weights, out, slot_ids,
        TOPK=topk, H=hidden, eps=POST_NORM_EPS, APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True,
        HAS_IDS=True, BLOCK_H=block_h, num_warps=8 if block_h >= 4096 else 4, num_stages=2,
    )
    return out


def inc_kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]
    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        if R1_EXPERT_MAP_INDEX_SELECT:
            local_ids = torch.index_select(expert_map, 0, local_ids.reshape(-1)).view(num_tokens, TOP_K)
        else:
            local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    block_size_m = _bm_override(num_tokens, block_size_m)
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        align_experts = N_EXPERTS if expert_map is not None else bank["count"]
        if (R1_ALIGN_LOCAL_BUCKETS and expert_map is not None
                and num_tokens >= R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS):
            align_experts = bank["count"]
        sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, align_experts)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _post_expert_norm_weighted_sum(y, topk_weights, state["norm_w"], local_ids)


# ----------------------------------------------------------------------------------------------
# Synthetic inputs (own generator, same distribution family as the task: Student-t popularity, busiest rank)
# and an FP32 reference for the self-test
# ----------------------------------------------------------------------------------------------

_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant(packed, sf, gs):
    grid = torch.tensor(_GRID, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)
    return vals.view(*vals.shape[:-2], k)


def ref_fp32(data, w):
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


def rel_err(a, b):
    a, b = a.float(), b.float()
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row


def synth_inputs(bs, seed, device, dist="normal"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=device)
    if dist == "normal":
        z = torch.randn((256,), generator=gen, device=device)
        u = torch.rand((256,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    else:
        scores = scores + torch.rand((256,), generator=gen, device=device).pow(6) * 4
    top = scores.topk(TOP_K, dim=-1)
    ids = top.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": D_HIDDEN, "d_expert": D_EXPERT, "n_experts": 256,
                   "local_start": 0, "local_count": E_LOC, "top_k": TOP_K, "sharding": "ep8"},
    }


def time_graph(fn, data, flush, reps):
    """(median replay time in us, output of the last replay) of a CUDA graph of fn(data), cold L2 before every
    replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(data)
        fn(data)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = fn(data)
    ts = []
    for _ in range(reps):
        flush.add_(1)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], o



# ==============================================================================================
# LAB (check runs only, never submit). kernel_fn = the incumbent (r1_glue_v3 path) so the run passes;
# prepare() runs the incumbent prepare, then times K2a v2 template variants and runtime knobs at the six scored
# shapes (CUDA-graph replay, cold L2 before every replay), against an in-file incumbent reference row.
# Read the "===== K2A LAB V2 SUMMARY" block (printed last).
# ==============================================================================================

LAB_DEADLINE_S = 240.0
LAB_REPS = 20
LAB_VARIANTS = [
    variant(1, 4, 0, 1, 8192, 4),     # L0 = submission v0: small T, 4 CTAs/SM, <= 64 regs, no prefetch
    variant(2, 4, 0, 1, 16384, 4),    # L1 = submission v1: 16 tok/pass
    variant(4, 4, 0, 1, 16384, 3),    # L2 = submission v2: 32 tok/pass, 3 CTAs/SM
    variant(1, 2, 1, 1, 8192, 4),     # L3: prefetch distance 1 group (register double buffer), U=2
    variant(1, 4, 0, 1, 8192, 3),     # L4: CTAs/SM 3 instead of 4 (more regs, fewer warps)
    variant(2, 4, 0, 2, 16384, 4),    # L5: token-group warps TW=2 (32 tok/pass sharing one weight fetch)
    variant(4, 2, 1, 1, 16384, 3),    # L6: verify with prefetch
    variant(4, 4, 0, 2, 16384, 3),    # L7: 64 tok/pass (TW=2 x NC=4): hot experts in fewer passes
]
LAB_SHAPES = ((1, 401), (16, 402), (32, 403), (8, 404), (128, 405), (256, 406))
# Incumbent per-shape us from the judge (r1_glue_v2, job 96a32200948f4dc6), for orientation only.
JUDGE_INC_US = {1: 54.0, 16: 156.3, 32: 213.9, 8: 111.6, 128: 344.0, 256: 475.6}

LAB_CUDA = r"""
__global__ void l2probe_kernel(const uint4* __restrict__ p, long long n16, int iters, unsigned* sink) {
  uint32_t acc = 0u;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (int it = 0; it < iters; ++it) {
#pragma unroll 4
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
      const uint4 v = __ldcg(p + i);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x12345678u) atomicAdd(sink, 1u);
}
void l2probe(torch::Tensor buf, int64_t iters, torch::Tensor sink, int64_t G) {
  l2probe_kernel<<<(unsigned)G, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(buf.data_ptr()), (long long)(buf.numel() * buf.element_size() / 16), (int)iters,
      reinterpret_cast<unsigned*>(sink.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
LAB_CPP = "void l2probe(torch::Tensor buf, int64_t iters, torch::Tensor sink, int64_t G);\n"


def lab_build(variants):
    from torch.utils.cpp_extension import load_inline
    src = KERNEL_SRC + LAB_CUDA + gen_host(variants)
    tag = hashlib.sha1((CPP_SRC + LAB_CPP + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"k2a_lab_v2_{tag}", cpp_sources=CPP_SRC + LAB_CPP, cuda_sources=src,
                      functions=["run", "attrs", "l2probe"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills"])
    return mod, time.time() - t0


def _touched_bytes(d):
    ids = d["topk_ids"].reshape(-1)
    loc = ids[(ids >= 0) & (ids < E_LOC)]
    n = int(torch.unique(loc).numel())
    cnt = torch.bincount(loc.long(), minlength=E_LOC)
    top = sorted([int(c) for c in cnt if c > 0], reverse=True)
    return n, n * (3 * D_EXPERT * D_HIDDEN) * (0.5 + 1.0 / 16), top


def lab(weights, inc_state, t_start):
    out = []

    def P(s):   # progress lines (kept short); SUM lines are printed once, in the summary block
        out.append(s)
        print(s, flush=True)

    dev = weights["w13"].device
    summ = []
    try:
        st = None
        t0 = time.time()
        w13q = torch.empty((E_LOC, 512, 64, 32, 4), dtype=torch.int32, device=dev)
        s13q = torch.empty((E_LOC, 512, 32, 32, 4), dtype=torch.uint8, device=dev)
        w2q = torch.empty((E_LOC, 256, 64, 32, 4), dtype=torch.int32, device=dev)
        s2q = torch.empty((E_LOC, 256, 32, 32, 4), dtype=torch.uint8, device=dev)
        for e in range(E_LOC):
            w13q[e], s13q[e] = prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True)
            w2q[e], s2q[e] = prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False)
        st = dict(
            w13q=w13q, s13q=s13q, w2q=w2q, s2q=s2q,
            gsc13=(weights["w13_gs"][:E_LOC].float() * W_SHIFT).contiguous(),
            gsc2=(weights["w2_gs"][:E_LOC].float() * W_SHIFT).contiguous(),
            nw=(weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous(),
            h=torch.zeros((8 * M_MAX, 2 * D_EXPERT), dtype=torch.float32, device=dev),
            inter=torch.zeros((8 * M_MAX, D_EXPERT), dtype=torch.bfloat16, device=dev),
            y=torch.zeros((8 * M_MAX, D_HIDDEN), dtype=torch.float32, device=dev),
            meta=torch.zeros((META_HDR + 8 * M_MAX,), dtype=torch.int32, device=dev),
            bar=torch.zeros((2,), dtype=torch.int32, device=dev),
            err=torch.zeros((4,), dtype=torch.int32, device=dev),
        )
        P(f"LAB repack {time.time() - t0:.1f}s")
        try:
            mod, dt = lab_build(LAB_VARIANTS)
        except Exception as ex:  # noqa: BLE001
            P("LAB BUILD FAILED. nvcc tail:\n" + str(ex)[-9000:])
            summ.append("SUM BUILD FAILED (see tail above)")
            return summ
        P(f"LAB build {len(LAB_VARIANTS)} variants OK in {dt:.1f}s")
        summ.append(f"SUM build {len(LAB_VARIANTS)} variants {dt:.0f}s")
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        grid, vdesc = [], []
        for i, v in enumerate(LAB_VARIANTS):
            a = mod.attrs(i)
            occ = a[3]
            grid.append(sms * max(1, min(occ, v["minb"])) if occ >= 1 else sms)
            vdesc.append(f"L{i} nc{v['nc']}u{v['u']}p{v['pf']}t{v['tw']}x{v['xe'] // 1024}k m{v['minb']} "
                         f"r{a[0]} l{a[1]} s{a[2] // 1024}K o{occ} g{grid[-1]}")
            summ.append("SUM " + vdesc[-1])
        st.update(mod=mod, grid=grid, sms=sms, variants=LAB_VARIANTS)
        flush = torch.empty((64 * 1024 * 1024,), dtype=torch.float32, device=dev)
        tiny = torch.zeros((16,), dtype=torch.float32, device=dev)
        floor, _ = time_graph(lambda d: tiny.add_(1.0), None, flush, LAB_REPS)
        P(f"LAB floor (1 tiny kernel graph replay) {floor:.2f} us")
        summ.append(f"SUM floor {floor:.2f}us (raw us below include it; judge subtracts ~4us from both sides)")
        # L2 -> SM rate (32 MiB resident buffer, no flush)
        try:
            buf = torch.zeros((8 * 1024 * 1024,), dtype=torch.float32, device=dev)
            sink = torch.zeros((1,), dtype=torch.int32, device=dev)
            res = []
            for gmul in (4, 8):
                mod.l2probe(buf, 2, sink, sms * gmul)
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                mod.l2probe(buf, 20, sink, sms * gmul)
                e1.record()
                torch.cuda.synchronize()
                res.append(f"g{gmul}x {20 * buf.numel() * 4 / (e0.elapsed_time(e1) * 1e-3) / 1e12:.2f}")
            summ.append("SUM L2->SM read TB/s (32MiB resident, ld.cg 16B): " + " ".join(res))
            del buf
        except Exception as ex:  # noqa: BLE001
            summ.append(f"SUM L2 probe error {ex!r}"[:200])
        # shapes
        data, refs, incs, tb = {}, {}, {}, {}
        for bs, seed in LAB_SHAPES:
            d = synth_inputs(bs, seed, dev)
            data[bs] = d
            refs[bs] = ref_fp32(d, weights)
            tb[bs] = _touched_bytes(d)
        dsk = synth_inputs(128, 7, dev, "skewed")
        data["sk"], refs["sk"], tb["sk"] = dsk, ref_fp32(dsk, weights), _touched_bytes(dsk)
        for k in list(data):
            o = inc_kernel_fn(data[k], inc_state)
            torch.cuda.synchronize()
            r, _ = rel_err(o, refs[k])
            incs[k], _ = time_graph(lambda dd: inc_kernel_fn(dd, inc_state), data[k], flush, LAB_REPS)
            P(f"LAB inc bs={k} {incs[k]:.1f}us rel {r:.1e} touched {tb[k][0]} top {tb[k][2][:4]}")
        summ.append("SUM REF inc us   | " + " ".join(f"bs{bs} {incs[bs]:.1f}" for bs, _ in LAB_SHAPES)
                    + f" | sk128 {incs['sk']:.1f}")

        def check_and_time(d, ref, vid, fuse=1, kst=2, asg=0, fa=0, probe=0):
            st["err"].zero_()
            o = v2_call(st, d, vid, fuse, kst, asg, fa, probe)
            torch.cuda.synchronize()
            r = rel_err(o, ref)[0] if probe == 0 else 0.0
            errf = int(st["err"][0])
            t, o2 = time_graph(lambda dd: v2_call(st, dd, vid, fuse, kst, asg, fa, probe), d, flush, LAB_REPS)
            torch.cuda.synchronize()
            errf |= int(st["err"][0])                       # barrier timeout during the replays
            if probe == 0:
                r = max(r, rel_err(o2, ref)[0])             # output of the last replay
            v2_reset(st)
            return t, r, errf

        # correctness first (eager), every variant at bs 1 and 33: a fault here stops the lab before timing
        for vid in range(len(LAB_VARIANTS)):
            for bs in (1, 33):
                d = synth_inputs(bs, 900 + bs, dev)
                o = v2_call(st, d, vid, 1, 2, 0)
                torch.cuda.synchronize()
                r, row = rel_err(o, ref_fp32(d, weights))
                if r > 1e-2 or row > 5e-2 or int(st["err"][0]):
                    P(f"LAB check L{vid} bs={bs}: BAD rel {r:.2e} row {row:.2e} err {int(st['err'][0])}")
                    summ.append(f"SUM check L{vid} bs{bs} BAD rel {r:.2e} row {row:.2e}")
                v2_reset(st)
        best = {}
        vt = {}
        for vid in range(len(LAB_VARIANTS)):
            cells = []
            for k in [bs for bs, _ in LAB_SHAPES] + ["sk"]:
                if time.time() - t_start > LAB_DEADLINE_S:
                    cells.append(f"{k}:deadline")
                    continue
                t, r, errf = check_and_time(data[k], refs[k], vid)
                vt[(vid, k)] = t
                bad = "" if (r < 1e-2 and errf == 0) else f"!BAD r{r:.1e} e{errf}"
                if not bad and (k not in best or t < best[k][0]):
                    best[k] = (t, vid)
                cells.append(f"{k}:{t:.1f}{bad}")
            summ.append(f"SUM L{vid} us | " + " ".join(cells))
        # knobs on the best variant per shape
        for k in [bs for bs, _ in LAB_SHAPES] + ["sk"]:
            if k not in best:
                continue
            tb_, vid = best[k]
            cells = [f"base {tb_:.1f}"]
            for name, kw in (("f0", dict(fuse=0)), ("f2", dict(fuse=2)), ("ks1", dict(kst=1)), ("ks4", dict(kst=4)),
                             ("asg1", dict(asg=1)), ("atom", dict(fa=1)), ("pr1", dict(probe=1)),
                             ("pr2", dict(probe=2))):
                if time.time() - t_start > LAB_DEADLINE_S:
                    cells.append(f"{name}:deadline")
                    continue
                t, r, errf = check_and_time(data[k], refs[k], vid, **kw)
                bad = "" if (r < 1e-2 and errf == 0) else "!BAD"
                cells.append(f"{name} {t:.1f}{bad}")
                if not bad and name not in ("pr1", "pr2") and t < best[k][0]:
                    best[k] = (t, vid, name)
            summ.append(f"SUM knobs bs{k} L{vid}: " + " ".join(cells))
        # best per shape vs incumbent, with effective TB/s on the touched experts' bytes
        cells = []
        geo_inc, geo_v2 = 0.0, 0.0
        for bs, _ in LAB_SHAPES:
            if bs not in best:
                continue
            b = best[bs]
            t = b[0]
            eff = tb[bs][1] / max(t - floor, 1e-3) / 1e6
            cells.append(f"bs{bs} L{b[1]}{'/' + b[2] if len(b) > 2 else ''} {t:.1f} ({incs[bs] / t:.2f}x inc, "
                         f"{eff:.2f}TB/s)")
            geo_inc += math.log(max(incs[bs] - floor, 1e-3))
            geo_v2 += math.log(max(min(t, incs[bs]) - floor, 1e-3))
        summ.append("SUM BEST " + " | ".join(cells))
        n = len([1 for bs, _ in LAB_SHAPES if bs in best])
        if n:
            summ.append(f"SUM routed geomean gain vs incumbent (min per shape) {math.exp((geo_inc - geo_v2) / n):.3f}x "
                        f"over {n} shapes")
        if "sk" in best:
            b = best["sk"]
            summ.append(f"SUM dedup sk128 (top {tb['sk'][2][:3]}, {tb['sk'][0]} experts {tb['sk'][1] / 1e6:.0f}MB): "
                        f"best L{b[1]} {b[0]:.1f}us = {tb['sk'][1] / max(b[0] - floor, 1e-3) / 1e6:.2f}TB/s eff; "
                        f"L2(32/pass) {vt.get((2, 'sk'), 0):.1f} vs L7(64/pass) {vt.get((7, 'sk'), 0):.1f}")
    except Exception as ex:  # noqa: BLE001
        summ.append(f"SUM LAB ERROR {ex!r}"[:600])
    return summ


def prepare(weights: Weights):
    t_start = time.time()
    state = inc_prepare(weights)
    print(f"LAB k2a-lab-v2 (check only); incumbent prepared in {time.time() - t_start:.1f}s", flush=True)
    summ = lab(weights, state, t_start)
    torch.cuda.synchronize()
    print("===== K2A LAB V2 SUMMARY (raw graph-replay us incl. floor; cold L2; median of %d) =====" % LAB_REPS)
    for s in summ:
        print(s)
    print("SUM judge incumbent us (r1_glue_v2 96a32200948f4dc6): "
          + " ".join(f"bs{k} {v}" for k, v in JUDGE_INC_US.items()))
    print(f"SUM lab total {time.time() - t_start:.0f}s")
    print("===== END K2A LAB V2 SUMMARY =====", flush=True)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    return inc_kernel_fn(data, state)

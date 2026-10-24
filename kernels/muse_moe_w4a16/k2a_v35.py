# okc: problem=muse_moe_w4a16
# nonce k2a-v35-1
"""K2a v3: warp-specialized NVFP4 x BF16 MoE experts (design: k2a_v3_design.md). Every bucket runs v3 when its
prepare() self-test passes; a failing bucket falls back to v2.2 (unchanged below), then to the incumbent.
v3: 1 producer warp (one 18 KB cp.async.bulk per stage: codes + scales, re-tiled losslessly from v2.2's repack, plus
X slices) + 8 consumer warps (v2.2's decode + mma.sync m16n8k16, operands from smem) + 1 epilogue warp (W13 split
items), mbarrier ring, 2 CTAs/SM, stream-K over the global stage sequence, 3 launches chained with early-release PDL
(K1 route+W13+SwiGLU | K2 W2 | K3 post-norm + weighted sum). Numerics identical to v2.2.

---- v2.2 notes (the fallback path) ----
K2a v2.2: custom NVFP4 x BF16 MoE experts for decode/verify (results: k2a_v22.md; design: k2a_v2_design.md).
Judge clone on the rented H200 (judge image): 1.471x [1.449, 1.493] vs SGLang (v2.1: 1.397x).

v2.2 over v2.1 (same repacked layout and numerics; every change measured on the box):
  * post-expert norm latency-batched: item = (token, column quarter) when 4M <= grid, slot pairs per load batch
    (v2.1 chained up to 16 dependent L2 round trips per token: 9-11 us at bs <= 8, now ~6 us at bs 1);
  * 1-launch plan for bs 1/32/128/256 with the first W2 weight group issued before the W13->W2 grid barrier;
  * X staging via cp.async (all pieces in flight, one wait) for the >= 16-token variants (bs 128/256 -3..4%);
    the 8-token variant keeps v2.1's loop (cp.async there raised its spills 24 -> 96 B);
  * warp-parallel phase setup; per-CTA globaltimer trace (probe bit 4) for box profiling;
  * per-bucket plan from the box sweep (tools/box/v22_bench.py); AUTOTUNE off; ROUTE v2 on every bucket.
Tried and rejected on the box (kept compiled out / in kernels/dev/): double-buffered step pipeline (slower,
spills), quantization-aware split-K, U=8 / PF=1, and a separate hot-expert launch at 64 tokens/pass (K2A2_HOT;
slower: 425 vs 373 us on a 143-token-expert bs256 routing).

v2 description (unchanged below):

What changed vs v0/v1 (which ran 512 threads x 1 CTA/SM at 93-128 regs and streamed ~1.9 / ~1.1 TB/s):
  * occupancy = stream_lab part D's winner: 256-thread CTAs, 3-4 CTAs/SM, __launch_bounds__(256, MINB)
    (<= 64 regs at MINB 4, <= 80 at MINB 3), plain 16 B weight loads, U loads per thread per group
    (64 B/thread at U = 4, 64 KiB/SM nominal at 4 CTAs/SM), PF = register double-buffer knob.
  * B operand (activations) staged per CTA unit in shared memory (16 B-padded rows, conflict-free LDS.128);
    no X loads on the weight stream's critical path, no X register ring.
  * one warp = one 16-row tile; one 16 B weight load per lane = 1 m16 tile x k64 (4 mma k16 steps), so the
    accumulator footprint is only 4 regs per 8-token chunk (NC chunks per pass).
  * hot experts: tokens beyond 8*NC*TW per pass become extra passes over the same weights; passes of the same
    (expert, row group, k-split) are adjacent units, assigned round-robin to concurrently running CTAs, so the
    2nd..n-th read is an L2 hit (HBM reads each touched expert once). TW (token-group warps sharing one weight
    fetch inside a CTA) is a template knob for the lab.
  * routing is folded into the first launch (every CTA computes it deterministically from topk_ids; CTA 0
    writes the meta for later launches); all experts of a call share one launch; split-K partials go through
    FP32 atomics (only when the device-chosen KS > 1), never a reduce kernel.
  * launches per call: FUSE 0 = 3 (route+W13 | W2 | finalize), 1 = 2 (route+W13 | W2 + grid barrier +
    finalize), 2 = 1 (everything, two grid barriers). Grid barriers are bounded (clock64 budget -> error flag).
  * per-shape route table vs the incumbent (r1_glue_v3: Marlin + Triton post-norm + block-size ranges, copied
    verbatim below), default = incumbent until the lab reports; optional prepare-time autotune (CUDA-graph
    replay, cold L2) can switch a bucket to v2 only if v2 wins by AUTOTUNE_MARGIN.

Numerics: E2M1 x E4M3 dequant is exact in BF16 (v0's GPU-validated bit layout, scale re-encoding lossless),
FP32 global scale (x 2^8) on the FP32 accumulator, SwiGLU in FP32 rounded once to BF16 (the W2 B operand),
y rounded to BF16 before (1 + w) and the RMSNorm (as SGLang), norm/weight/sum in FP32, BF16 out.

Layouts (prepare() repacks losslessly; checked bit-exactly by k2a_v2_emulate.py):
  WQ[e][tile][kb][lane][4 x u32]: tile = 16 prepacked rows, kb = k64 block (0..63), lane = 4g + t; words
     (x, y) = row g at k = 64kb + 16t + (0..7, 8..15), words (z, w) = row g+8 likewise; 8 E2M1 codes per word
     at the bit positions BITPOS (device: deq4).
  SQ[e][tile][kb/2][lane] u32: bytes (row g, kb even), (row g+8, kb even), (row g, kb odd), (row g+8, kb odd)
     of scale group 4kb + t, re-encoded so that bf16 bits = b*16 + 0x7580 = sf * 2^118.
  W13 tile r = gate rows 8r..8r+7 then up rows 8r..8r+7 (a lane holds gate and up of one intermediate index).
"""

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
AUTOTUNE = False        # prepare() times v2 vs the incumbent per bucket and may switch a bucket to v2
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
# v3.1 code layout: deq4 = {w & M, rotl(w, 6) & M, rotl(w, 12) & M, ((w >> 6) & 0x01000100) | ((w << 2) & 0x80C080C0)},
# M = 0x81C081C0: 7-8 integer ops per word instead of ~10 (identical bf16 outputs; checked bit-exactly vs BITPOS's deq4).
BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (0, 1, 2, 9), (16, 17, 18, 25),
           (26, 27, 28, 3), (10, 11, 12, 19), (4, 5, 14, 13), (20, 21, 30, 29))
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
    variant(1, 4, 0, 1, 8192, 4),     # 0: 8 tokens / pass, 4 CTAs/SM (<= 64 regs), synchronous X staging
    variant(2, 4, 0, 1, 16384, 4),    # 1: 16 tokens / pass, cp.async X staging
    variant(4, 4, 0, 1, 16384, 3),    # 2: 32 tokens / pass, 3 CTAs/SM (<= 80 regs)
]
# Per bucket: (lo, hi, representative scored bs). Decisions are taken per bucket (nearby shapes route alike).
BUCKETS = ((1, 4, 1), (5, 12, 8), (13, 24, 16), (25, 48, 32), (49, 192, 128), (193, M_MAX, 256))
# v2 plan per bucket: (variant index, fuse, ks_target, asg). Lab (k2a_lab_v2.py) decides; these are the design
# defaults: 2 launches, KS chosen so units >= 2 x grid, round-robin units (concurrent passes -> L2 dedup).
V2_PLAN = {1: (0, 2, 1, 0), 8: (0, 1, 2, 0), 16: (0, 1, 1, 0), 32: (0, 2, 2, 0), 128: (1, 2, 2, 0), 256: (1, 2, 4, 0)}  # box sweep 2026-10-24 (v22_bench)
# Static route table: "inc" (r1_glue_v3 path) or "v2". Default incumbent everywhere until measured (R2).
ROUTE = {1: "v2", 8: "v2", 16: "v2", 32: "v2", 128: "v2", 256: "v2"}  # judge clone: v2 beats the incumbent on every bucket


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


def pack_words(codes: torch.Tensor, bitpos=BITPOS) -> torch.Tensor:
    """codes [..., 8] (0..15) -> int32 [...] per bitpos (BITPOS: v2.2, BITPOS3: v3.1)."""
    c = codes.to(torch.int64)
    w = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for i, (p0, p1, p2, ps) in enumerate(bitpos):
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


def prepack_expert(w: torch.Tensor, sf: torch.Tensor, gate_up: bool, bitpos=BITPOS):
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
    wq = pack_words(c, bitpos)
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

// K2A2_HOT: separate hot-expert launch (v2.2 experiment, measured slower on the box; compiled out by default --
// its extra code raised the 8-token variant's spills from 24 B to 96 B).
#ifndef K2A2_HOT
#define K2A2_HOT 0
#endif

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
  long long* trace;     // per-CTA globaltimer marks (probe & 4), else nullptr
  int M, lo, mask, ks_target, asg, force_atom, probe;
  int hot_t, hot_mode;  // hot_t > 0: experts with >= hot_t tokens are "hot"; hot_mode 0 = cold launch, 1 = hot launch
};

// Expert inclusion for the current launch (hot-expert split; hot_t <= 0: every expert).
__device__ __forceinline__ bool incl_expert(const Params& P, int cnt) {
#if K2A2_HOT
  if (P.hot_t <= 0) return true;
  return (P.hot_mode == 1) ? (cnt >= P.hot_t) : (cnt < P.hot_t);
#else
  return true;
#endif
}

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
  static_assert(U == 2 || U == 4 || U == 8, "U");
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

__device__ __forceinline__ void trace_mark(const Params& P, int i) {
  if (P.trace != nullptr && threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    P.trace[blockIdx.x * 8 + i] = (long long)t;
  }
}

// Split-K choice. ks_target < 8: smallest KS with units >= ks_target * grid (v2.1). ks_target >= 8: quantization-aware:
// minimise makespan(KS) = ceil(units1 * KS / grid) * (1 / KS + alpha), alpha = ks_target / 256 = per-unit overhead
// relative to a full-K unit (uniform units: the slowest CTA does ceil(U/G) of them).
__device__ __forceinline__ int choose_ks(int units1, int ks_target, int G) {
  if (ks_target < 8) {
    int ks = 1;
    while (ks < 16 && units1 * ks < ks_target * G) ks <<= 1;
    return ks;
  }
  const float alpha = (float)ks_target * (1.f / 256.f);
  int best = 1;
  float bc = 3.0e38f;
  for (int ks = 1; ks <= 16; ks <<= 1) {
    const int waves = (units1 * ks + G - 1) / G;
    const float c = (float)waves * (1.f / (float)ks + alpha);
    if (c < bc * 0.999f) { bc = c; best = ks; }
  }
  return best;
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
    int nv = incl_expert(P, c) ? np : 0;
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
      if (blockIdx.x == 0 && P.hot_mode != 1) P.meta[MH + p] = c;
      if (rank == 0) S.wc[warp][e] = b + __popc(m);
    } else if (blockIdx.x == 0 && p < NP && P.hot_mode != 1) {
      P.meta[MH + p] = -1;
    }
    __syncwarp();
  }
  if (tid == 0) {
    const int nv = S.hdr[1];
    const int G = (int)gridDim.x;
    const int ks1 = choose_ks(nv * (512 / C::RW), P.ks_target, G);
    const int ks2 = choose_ks(nv * (256 / C::RW), P.ks_target, G);
    S.hdr[2] = ks1; S.hdr[3] = ks2;
    S.hdr[5] = (ks1 > 1 || P.force_atom) ? 1 : 0;
    S.hdr[6] = (ks2 > 1 || P.force_atom) ? 1 : 0;
    if (blockIdx.x == 0 && P.hot_mode != 1) for (int i = 0; i < 8; ++i) P.meta[i] = S.hdr[i];
  }
  if (blockIdx.x == 0 && tid < 32 && P.hot_mode != 1) {
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
__device__ void phase_units(const Params& P, Route& S, int KS) {
  constexpr int NRG = ((MODE == 1) ? 512 : 256) / C::RW;
  if (threadIdx.x < 32) {   // warp scan over the 32 experts (was a serial thread-0 loop)
    const int lane = threadIdx.x;
    const int n = incl_expert(P, S.cnt[lane]) ? S.np[lane] * NRG * KS : 0;
    const int tl = min(S.cnt[lane], C::TPW);
    const int c = n * (4 + (tl + 7) / 8);
    int iu = n, ic = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(FULL, iu, d), b = __shfl_up_sync(FULL, ic, d);
      if (lane >= d) { iu += a; ic += b; }
    }
    S.ub[lane] = iu - n; S.cb[lane] = ic - c;
    if (lane == 31) { S.ub[32] = iu; S.cb[32] = ic; }
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
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
// (strings as stream_lab S2 / k1_v0, GPU-validated)
__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
  const uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// Stage the B operand rows of a pass for k in [k0, k0 + kw). x[token] (MODE 1) / inter[c] (MODE 2) via cp.async:
// every piece is in flight at once (v2.1's load->store loop serialized ~ntok*kw/2048 L2 round trips per slice,
// e.g. 8 at 64 tokens). SwiGLU(h[c]) (KS1 > 1, small T) through registers. Rows >= ntok are not
// touched (they only feed discarded output columns). The caller's __syncthreads publishes the tile.
template <class C, int MODE>
__device__ void stage_x(const Params& P, const int16_t* s_list, __nv_bfloat16* xs, int c0, int ntok, int k0, int kw,
                        bool from_h) {
  const int ppr = kw >> 3;
  const int sh = __ffs(ppr) - 1;     // ppr is a power of two
  const int tot = ntok * ppr;
  if constexpr (C::NC == 1) {   // 8-token variant: v2.1's loop verbatim (lowest spills)
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
    return;
  }
  if (MODE == 1 || !from_h) {
    for (int idx = threadIdx.x; idx < tot; idx += 256) {
      const int n = idx >> sh, pc = idx & (ppr - 1);
      const int k = k0 + pc * 8;
      const __nv_bfloat16* src;
      if constexpr (MODE == 1) src = P.x + (size_t)(((int)s_list[c0 + n]) >> 3) * KDIM + k;
      else src = P.inter + (size_t)(c0 + n) * KDIM + k;
      cp_async16(xs + n * C::XSTR + pc * 8, src);
    }
    cp_async_commit();
    cp_async_wait_all();
  } else {
    for (int idx = threadIdx.x; idx < tot; idx += 256) {
      const int n = idx >> sh, pc = idx & (ppr - 1);
      const float* hr = P.h + (size_t)(c0 + n) * (2 * KDIM) + k0 + pc * 8;
      const float4 g0 = __ldcg(reinterpret_cast<const float4*>(hr));
      const float4 g1 = __ldcg(reinterpret_cast<const float4*>(hr + 4));
      const float4 u0 = __ldcg(reinterpret_cast<const float4*>(hr + KDIM));
      const float4 u1 = __ldcg(reinterpret_cast<const float4*>(hr + KDIM + 4));
      uint4 v;
      v.x = pack_bf2(swiglu(g0.x, u0.x), swiglu(g0.y, u0.y));
      v.y = pack_bf2(swiglu(g0.z, u0.z), swiglu(g0.w, u0.w));
      v.z = pack_bf2(swiglu(g1.x, u1.x), swiglu(g1.y, u1.y));
      v.w = pack_bf2(swiglu(g1.z, u1.z), swiglu(g1.w, u1.w));
      *reinterpret_cast<uint4*>(xs + n * C::XSTR + pc * 8) = v;
    }
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
__device__ void gemm_phase(const Params& P, Route& S, const int16_t* s_list, __nv_bfloat16* xs, bool bar_before) {
  constexpr int RT = (MODE == 1) ? 512 : 256;
  constexpr int U = C::U;
  const int KS = (MODE == 1) ? S.hdr[2] : S.hdr[3];
  const bool atom = ((MODE == 1) ? S.hdr[5] : S.hdr[6]) != 0;
  const bool from_h = (MODE == 2) && (S.hdr[5] != 0);
  phase_units<C, MODE>(P, S, KS);
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
  // bar_before (G2 in the same launch as G1): the first unit's first weight group is issued before the grid barrier,
  // so the W2 ramp overlaps the barrier wait.
  uint4 wa[U];
  uint32_t sa[U / 2];
  bool pre = false;
  if (bar_before) {
    if (u_beg < u_end) {
      int e = 0;
      while (S.ub[e + 1] <= u_beg) ++e;
      const int l = u_beg - S.ub[e];
      const int r = l / S.np[e];
      const int j = l % S.np[e];
      const int ntok = min(C::TPP, S.cnt[e] - j * C::TPP);
      if (max(0, min(C::NC, (ntok - t0w + 7) >> 3)) > 0) {
        const size_t tb = (size_t)e * RT + (r / KS) * C::RW + rw;
        load_group<C>(wq + tb * (NKB * 32) + lane, sq + tb * ((NKB / 2) * 32) + lane, ((r % KS) * KR) >> 6, wa, sa);
        pre = true;
      }
    }
    grid_barrier(P);
  }

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
      if (nch > 0 && !pre) load_group<C>(wt, st, kb0, wa, sa);     // in flight across the staging below
      pre = false;
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

// Post-expert norm + weighted sum, latency-batched (v2.1 chained up to 16 dependent L2 round trips per token: 9-11 us
// at bs <= 8). Item = (token, column quarter) when 4M <= grid (small M: every quarter on its own CTA), else a token.
// Phase 1: sum of squares of bf16(y)*(1+w) over the full row for all local slots, two slots per batch of
// independent loads; phase 2: the item's columns for all 8 slots in one batch (quarter) or in slot pairs (token).
// Zeroes h rows when the W13 phase used atomics (h is all-zero between calls; prepare() zeroes it once).
__device__ void fin_phase(const Params& P, const Route& S, float (*s_red)[8], int* s_fc, float* s_fr) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const bool quarter = (4 * P.M <= (int)gridDim.x);
  const int items = quarter ? 4 * P.M : P.M;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int tok = quarter ? (it >> 2) : it;
    if (tid < 8) {
      s_fc[tid] = __ldcg(P.meta + MH + tok * 8 + tid);
      s_fr[tid] = __ldg(P.wts + tok * 8 + tid);
    }
    __syncthreads();
    float ss[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) ss[s] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; s += 2) {
      const int c0 = s_fc[s], c1 = s_fc[s + 1];
      if (c0 >= 0 || c1 >= 0) {
        const float4* p0 = reinterpret_cast<const float4*>(P.y + (size_t)max(c0, 0) * KDIM + 16 * tid);
        const float4* p1 = reinterpret_cast<const float4*>(P.y + (size_t)max(c1, 0) * KDIM + 16 * tid);
        float4 a[4], b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[j] = (c0 >= 0) ? __ldcg(p0 + j) : z4; b[j] = (c1 >= 0) ? __ldcg(p1 + j) : z4; }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + 16 * tid) + j);
          float f;
          f = bf16r(a[j].x) * w.x; ss[s] += f * f;  f = bf16r(a[j].y) * w.y; ss[s] += f * f;
          f = bf16r(a[j].z) * w.z; ss[s] += f * f;  f = bf16r(a[j].w) * w.w; ss[s] += f * f;
          f = bf16r(b[j].x) * w.x; ss[s + 1] += f * f;  f = bf16r(b[j].y) * w.y; ss[s + 1] += f * f;
          f = bf16r(b[j].z) * w.z; ss[s + 1] += f * f;  f = bf16r(b[j].w) * w.w; ss[s + 1] += f * f;
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
    if (tid < 8) {
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) tot += s_red[w][tid];
      s_fr[tid] = (s_fc[tid] >= 0) ? rsqrtf(tot * (1.f / 4096.f) + 1e-8f) * s_fr[tid] : 0.f;
    }
    __syncthreads();
    if (quarter) {
      const int col = (it & 3) * 1024 + 4 * tid;
      float4 v[8];
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int c = s_fc[s];
        v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
      }
      const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const float r = s_fr[s];
        a0 += bf16r(v[s].x) * w.x * r; a1 += bf16r(v[s].y) * w.y * r;
        a2 += bf16r(v[s].z) * w.z * r; a3 += bf16r(v[s].w) * w.w * r;
      }
      uint2 o;
      o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
      *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;
    } else {
      float acc[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
      for (int s = 0; s < 8; s += 2) {
        const int c0 = s_fc[s], c1 = s_fc[s + 1];
        if (c0 >= 0 || c1 >= 0) {
          const float r0 = s_fr[s], r1 = s_fr[s + 1];
          const float4* p0 = reinterpret_cast<const float4*>(P.y + (size_t)max(c0, 0) * KDIM + 16 * tid);
          const float4* p1 = reinterpret_cast<const float4*>(P.y + (size_t)max(c1, 0) * KDIM + 16 * tid);
          float4 a[4], b[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) { a[j] = (c0 >= 0) ? __ldcg(p0 + j) : z4; b[j] = (c1 >= 0) ? __ldcg(p1 + j) : z4; }
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + 16 * tid) + j);
            acc[4 * j] += (bf16r(a[j].x) * r0 + bf16r(b[j].x) * r1) * w.x;
            acc[4 * j + 1] += (bf16r(a[j].y) * r0 + bf16r(b[j].y) * r1) * w.y;
            acc[4 * j + 2] += (bf16r(a[j].z) * r0 + bf16r(b[j].z) * r1) * w.z;
            acc[4 * j + 3] += (bf16r(a[j].w) * r0 + bf16r(b[j].w) * r1) * w.w;
          }
        }
      }
      uint4 o0, o1;
      o0.x = pack_bf2(acc[0], acc[1]); o0.y = pack_bf2(acc[2], acc[3]);
      o0.z = pack_bf2(acc[4], acc[5]); o0.w = pack_bf2(acc[6], acc[7]);
      o1.x = pack_bf2(acc[8], acc[9]); o1.y = pack_bf2(acc[10], acc[11]);
      o1.z = pack_bf2(acc[12], acc[13]); o1.w = pack_bf2(acc[14], acc[15]);
      uint4* op = reinterpret_cast<uint4*>(P.out + (size_t)tok * KDIM + 16 * tid);
      op[0] = o0; op[1] = o1;
    }
    __syncthreads();   // s_fc / s_fr reuse by the next item
  }
  if (S.hdr[5] != 0) {
    const size_t n4 = (size_t)S.hdr[0] * (2 * KDIM) / 4;
    float4* hp = reinterpret_cast<float4*>(P.h);
    for (size_t i = (size_t)blockIdx.x * 256 + tid; i < n4; i += (size_t)gridDim.x * 256) hp[i] = z4;
  }
}

#if K2A2_HOT
// Hot launch, W13 used atomics: inter = bf16(SwiGLU(h)) for the hot experts' rows, and those h rows back to zero.
__device__ void convert_hot(const Params& P, const Route& S) {
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int e = 0; e < 32; ++e) {
    const int cnt = S.cnt[e];
    if (cnt == 0 || !incl_expert(P, cnt)) continue;
    const int tot = cnt * (KDIM / 8);
    for (int idx = blockIdx.x * 256 + threadIdx.x; idx < tot; idx += gridDim.x * 256) {
      const size_t c = (size_t)(S.off[e] + (idx >> 9));
      const int k = (idx & 511) * 8;
      float* hr = P.h + c * (2 * KDIM);
      const float4 g0 = __ldcg(reinterpret_cast<const float4*>(hr + k));
      const float4 g1 = __ldcg(reinterpret_cast<const float4*>(hr + k + 4));
      const float4 u0 = __ldcg(reinterpret_cast<const float4*>(hr + KDIM + k));
      const float4 u1 = __ldcg(reinterpret_cast<const float4*>(hr + KDIM + k + 4));
      uint4 v;
      v.x = pack_bf2(swiglu(g0.x, u0.x), swiglu(g0.y, u0.y));
      v.y = pack_bf2(swiglu(g0.z, u0.z), swiglu(g0.w, u0.w));
      v.z = pack_bf2(swiglu(g1.x, u1.x), swiglu(g1.y, u1.y));
      v.w = pack_bf2(swiglu(g1.z, u1.z), swiglu(g1.w, u1.w));
      *reinterpret_cast<uint4*>(P.inter + c * KDIM + k) = v;
      *reinterpret_cast<float4*>(hr + k) = z4; *reinterpret_cast<float4*>(hr + k + 4) = z4;
      *reinterpret_cast<float4*>(hr + KDIM + k) = z4; *reinterpret_cast<float4*>(hr + KDIM + k + 4) = z4;
    }
  }
}
#endif

template <class C>
__global__ void __launch_bounds__(256, C::MINB) k2a2_kernel(const Params P) {
  __shared__ __align__(16) __nv_bfloat16 xs[C::XS];
  __shared__ int8_t s_ide[P_MAX];
  __shared__ int16_t s_list[P_MAX];
  __shared__ Route S;
  __shared__ float s_red[8][8];
  __shared__ int s_fc[8];
  __shared__ float s_fr[8];
  trace_mark(P, 0);
  if (P.mask & PH_ROUTE) route_prologue<C>(P, S, s_ide, s_list);
  else load_meta(P, S);
  trace_mark(P, 1);
  if (P.mask & PH_G1) {
    if (S.hdr[6] != 0) {   // W2 will accumulate with atomics: zero its rows now (W2 runs after this phase)
      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      float4* yp = reinterpret_cast<float4*>(P.y);
#if K2A2_HOT
      if (P.hot_mode == 1) {   // hot launch: only the hot experts' rows (the cold rows belong to the last launch)
        for (int e = 0; e < 32; ++e) {
          if (S.cnt[e] == 0 || !incl_expert(P, S.cnt[e])) continue;
          const size_t b4 = (size_t)S.off[e] * KDIM / 4, n4 = (size_t)S.cnt[e] * KDIM / 4;
          for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += (size_t)gridDim.x * 256) yp[b4 + i] = z;
        }
      } else
#endif
      {
        const size_t n4 = (size_t)S.hdr[0] * KDIM / 4;
        for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += (size_t)gridDim.x * 256) yp[i] = z;
      }
    }
    gemm_phase<C, 1>(P, S, s_list, xs, false);
    trace_mark(P, 2);
#if K2A2_HOT
    if (P.hot_mode == 1 && (P.mask & PH_G2) && S.hdr[5] != 0) {   // hot launch: SwiGLU(h) -> inter, h -> 0
      grid_barrier(P);
      convert_hot(P, S);
      __syncthreads();
      if (threadIdx.x == 0) S.hdr[5] = 0;                         // W2 then stages from inter
      __syncthreads();
    }
#endif
  }
  if ((P.mask & PH_G1) && !(P.mask & PH_G2) && (P.mask & PH_FIN)) grid_barrier(P);
  if (P.mask & PH_G2) { gemm_phase<C, 2>(P, S, s_list, xs, (P.mask & PH_G1) != 0); trace_mark(P, 3); }
  if ((P.mask & PH_G2) && (P.mask & PH_FIN)) grid_barrier(P);
  trace_mark(P, 4);
  if (P.mask & PH_FIN) fin_phase(P, S, s_red, s_fc, s_fr);
  trace_mark(P, 5);
}

}  // namespace k2a2

static k2a2::Params make_params(torch::Tensor& x, torch::Tensor& ids, torch::Tensor& wts, torch::Tensor& w13q,
                                torch::Tensor& s13q, torch::Tensor& w2q, torch::Tensor& s2q, torch::Tensor& gsc13,
                                torch::Tensor& gsc2, torch::Tensor& nw, torch::Tensor& h, torch::Tensor& inter,
                                torch::Tensor& y, torch::Tensor& meta, torch::Tensor& bar, torch::Tensor& err,
                                torch::Tensor& out, torch::Tensor& trace, int64_t M, int64_t lo, int64_t mask,
                                int64_t ks_target, int64_t asg, int64_t force_atom, int64_t probe, int64_t hot_t,
                                int64_t hot_mode) {
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
  p.trace = (probe & 4) ? reinterpret_cast<long long*>(trace.data_ptr()) : nullptr;
  p.M = (int)M; p.lo = (int)lo; p.mask = (int)mask; p.ks_target = (int)ks_target; p.asg = (int)asg;
  p.force_atom = (int)force_atom; p.probe = (int)probe; p.hot_t = (int)hot_t; p.hot_mode = (int)hot_mode;
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
         torch::Tensor bar, torch::Tensor err, torch::Tensor out, torch::Tensor trace, int64_t M, int64_t lo,
         int64_t G, int64_t ks_target, int64_t asg, int64_t force_atom, int64_t probe, int64_t hot_t, int64_t hot_mode) {
  const k2a2::Params p = make_params(x, ids, wts, w13q, s13q, w2q, s2q, gsc13, gsc2, nw, h, inter, y, meta, bar, err,
                                     out, trace, M, lo, mask, ks_target, asg, force_atom, probe, hot_t, hot_mode);
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
         torch::Tensor bar, torch::Tensor err, torch::Tensor out, torch::Tensor trace, int64_t M, int64_t lo,
         int64_t G, int64_t ks_target, int64_t asg, int64_t force_atom, int64_t probe, int64_t hot_t, int64_t hot_mode);
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
HOT_LAUNCH_COMPILED = False   # K2A2_HOT=0 in KERNEL_SRC: the hot-expert launch split is compiled out
FUSE_MASKS = {0: (PH_ROUTE | PH_G1, PH_G2, PH_FIN), 1: (PH_ROUTE | PH_G1, PH_G2 | PH_FIN),
              2: (PH_ROUTE | PH_G1 | PH_G2 | PH_FIN,)}


def v2_call(st, data, vid, fuse=1, ks_target=2, asg=0, force_atom=0, probe=0, hot_vid=-1, hot_t=0):
    """One v2 call (graph-capturable: fixed buffers, no host sync, no value-dependent host branching).
    hot_t > 0 (with hot_vid): experts with >= hot_t tokens go to a separate launch of variant hot_vid (more tokens per
    pass, one weight sweep), between (route + W13 cold) and (W2 cold + post-norm)."""
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    lo = data["config"]["local_start"]
    if hot_t > 0 and hot_vid >= 0 and HOT_LAUNCH_COMPILED:
        launches = ((vid, PH_ROUTE | PH_G1, 0), (hot_vid, PH_ROUTE | PH_G1 | PH_G2, 1), (vid, PH_G2 | PH_FIN, 0))
    else:
        launches = tuple((vid, mk, 0) for mk in FUSE_MASKS[fuse])
    for v, mask, hm in launches:
        st["mod"].run(v, mask, x, ids, wts, st["w13q"], st["s13q"], st["w2q"], st["s2q"], st["gsc13"],
                      st["gsc2"], st["nw"], st["h"], st["inter"], st["y"], st["meta"], st["bar"], st["err"], out,
                      st["trace"], m, lo, st["grid"][v], ks_target, asg, force_atom, probe, hot_t, hm)
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
        trace=torch.zeros((8 * 2048,), dtype=torch.int64, device=dev),
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
# K2A v3 (design: k2a_v3_design.md): small-T path (M <= M3). 1 producer warp + 8 consumer warps per CTA,
# 2 CTAs/SM, 18 KB bulk-copy stages (weights + scales, one cp.async.bulk) + X slices, mbarrier ring, stream-K
# over the global stage sequence, 3 launches with early-release PDL (K1 W13+SwiGLU | K2 W2 | K3 post-norm).
# ----------------------------------------------------------------------------------------------

M3 = 320                # v3 serves M <= 320 (all buckets)
V3_ENABLE = True
V3_SCALE5 = False       # v3.5: 5-bit block scales (lossless on 59/64 matrices) measured -3..-5%: off
HOT_T = 16              # v3.3: in HOTCAP variants, experts with more than HOT_T tokens run 64-token wgmma passes
HOT_W4 = 8              # v3.3: warm-work scale (x8) in the byte model of the CTA role split (box sweeps k33_sweep*)
SW3 = 18432             # bytes per stage: 16 KB codes + 2 KB scales
# (NC = 8-token chunks per pass, NST = ring stages, CTAs/SM). Box labs (k31_lab2/10): NST 2-4 and 1-3 CTAs/SM are all
# within 3% (NST 3 best at small T); NC 8 is 25-40% slower everywhere; NC 4 wins only at bs 256.
V3_VARIANTS = [(1, 3, 2, 0), (2, 3, 2, 0), (2, 2, 2, 1), (1, 3, 2, 2), (1, 2, 2, 1)]   # (NC, NST, CTAs/SM, 0 / 1 HOTCAP / 2 fused K12)
V3_PLAN = {1: 0, 8: 0, 16: 0, 32: 0, 128: 2, 256: 2}   # v3.3: bs 128/256 with warm experts on wgmma
V3_HOT_T = {128: 16, 256: 16}   # per-bucket warm threshold (judge clone: bs128 hot_t 16 1.023x, 32 1.008x vs v3.1)


def stage_repack5(wq: torch.Tensor, sq: torch.Tensor, base: int) -> torch.Tensor:
    """v3.5: like stage_repack, but the 2 KB of scale bytes b' become 5-bit codes c = b' - base (caller checks
    0 <= c < 32): per tile, [lane 32] u32 of nibbles (code q = 4 kp + byte, kp = the stage's 2 kb pairs) then
    [lane 32] u8 of high bits. Stage = 16384 + 1024 + 256 = 17664 B, stored in an 18432 B slot."""
    rt = wq.shape[0]
    nrb = rt // 8
    wb = wq.contiguous().view(torch.uint8).view(nrb, 8, 16, 4, 512).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 16384)
    c = (sq.to(torch.int64) - base).view(nrb, 8, 16, 2, 32, 4).permute(0, 2, 1, 4, 3, 5).reshape(nrb, 16, 8, 32, 8)
    sh = torch.arange(8, device=sq.device, dtype=torch.int64)
    nib = ((c & 15) << (4 * sh)).sum(-1)
    nib = torch.where(nib >= 2 ** 31, nib - 2 ** 32, nib).to(torch.int32)          # [nrb, 16, 8, 32]
    hb = (((c >> 4) & 1) << sh).sum(-1).to(torch.uint8)                             # [nrb, 16, 8, 32]
    pad = torch.zeros((nrb, 16, SW3 - 17664), dtype=torch.uint8, device=sq.device)
    return torch.cat((wb, nib.contiguous().view(torch.uint8).reshape(nrb, 16, 1024), hb.reshape(nrb, 16, 256), pad),
                     dim=-1).contiguous()


def stage_repack(wq: torch.Tensor, sq: torch.Tensor) -> torch.Tensor:
    """v2.2 (wq int32 [rt, 64, 32, 4], sq uint8 [rt, 32, 32, 4]) -> uint8 [rt/8, 16, 18432]: stage (rb, s) =
    [tile w 8][kb i 4][lane][16 B] = wq[8rb+w][4s+i] then [w 8][kp 2][lane][4 B] = sq[8rb+w][2s+kp]. Lossless."""
    rt = wq.shape[0]
    nrb = rt // 8
    wb = wq.contiguous().view(torch.uint8).view(nrb, 8, 16, 4, 512).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 16384)
    sb = sq.contiguous().view(nrb, 8, 16, 2, 128).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 2048)
    return torch.cat((wb, sb), dim=-1).contiguous()


KERNEL_SRC_V3 = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace k3 {

constexpr int KDIM = 4096;
constexpr int M3 = 320;
constexpr int P3 = 8 * M3;
constexpr int SW = 18432;
constexpr int SWC = 17664;           // v3.5: stage with 5-bit scales (16384 codes + 1024 nibbles + 256 high bits)
constexpr int XROW = 528;            // bytes per staged X row (512 + 16 pad: conflict-free LDS.128)
constexpr int XST = XROW / 2;        // bf16 elements per staged X row
constexpr int MH = 128;
constexpr unsigned FULL = 0xffffffffu;
#ifndef K3_NW_DISPATCH
#define K3_NW_DISPATCH 1
#endif
#ifndef V3_EARLY
#define V3_EARLY 0
#endif
constexpr long long WAIT_BUDGET = 1ll << 31;   // clock64 cycles (~1 s) per mbarrier wait -> error flag

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint8_t* ws13; const uint8_t* ws2;
  const float* gsc13; const float* gsc2; const float* nw;
  float* h; __nv_bfloat16* inter; float* y; int* meta; int* cnt; int* err;
  __nv_bfloat16* out; unsigned* sink;
  float* ss; int* cnt2;   // v3.1: per-route sum of squares of bf16(y)*(1+w) (K2), W2 split-item counters
  uint8_t* xw; uint8_t* xw2;   // v3.2: hot passes' W13 / W2 B operands in wgmma's core-matrix layout
  int hot_t;                   // v3.3: expert is warm (wgmma, 64-token passes) if cnt > hot_t (> 0); 0 = off
  int hot_w4;                  // v3.3: warm-work scale in the CTA role split's byte model, x8
  int* cnt13;                  // v3.4 fused K12: W13 items completed per expert (reset by K3)
  const uint8_t* ws13c; const uint8_t* ws2c;   // v3.5: stages with 5-bit block scales (per-expert format)
  const int* sfmt;             // v3.5: [64] per (matrix, expert): 0 = plain stage, else bf16x2 scale constant K
  int M, lo, probe;
};

struct Route {
  int cnt[32], off[32], np[32], hp[32];
  int sbh[33], sbc[33];   // v3.3: stage prefixes over warm experts only / cold experts only
  int sb2[33];            // v3.4 fused K12: W2 stage prefix (= sb / 2)
  int sb[33];
  int wc[8][32];
  int R;
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
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
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }

// ---- mbarrier / bulk copy / PDL (PTX ISA 8.x, sm_90) ----
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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void cp_async16_ef(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

// ---- routing: every CTA, deterministic (v2.2's algorithm; warps 0-7) ----
template <int NC>
__device__ void route(const Params& P, Route& S, int8_t* s_ide, int16_t* s_list, int nrb, bool write_meta) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int NP = 8 * P.M;
  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  const int L = ((NP + 255) / 256) * 32;
  const int p0 = warp * L;
  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) {
        const int v = P.ids[p] - P.lo;
        e = (v >= 0 && v < 32) ? v : -1;
      }
      if (e >= 0) atomicAdd(&S.wc[warp][e], 1);
    }
  }
  __syncthreads();
  if (warp == 0) {
    int c = 0;
    for (int w = 0; w < 8; ++w) { const int v = S.wc[w][lane]; S.wc[w][lane] = c; c += v; }
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    const bool hot = P.hot_t > 0 && c > P.hot_t;   // v3.3: warm expert -> 64-token wgmma passes
    const int np = hot ? (c + 63) / 64 : (c + 8 * NC - 1) / (8 * NC);
    int hv = hot ? np : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, hv, d); if (lane >= d) hv += nb; }
    S.hp[lane] = hv - (hot ? np : 0);
    {
      int cv = hot ? 0 : np;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, cv, d); if (lane >= d) cv += nb; }
      S.sbh[lane] = 16 * nrb * (hv - (hot ? np : 0));
      S.sbc[lane] = 16 * nrb * (cv - (hot ? 0 : np));
      if (lane == 31) { S.sbh[32] = 16 * nrb * hv; S.sbc[32] = 16 * nrb * cv; }
    }
    int sv = np;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, sv, d); if (lane >= d) sv += nb; }
    S.cnt[lane] = c; S.off[lane] = off; S.np[lane] = np;
    S.sb[lane] = 16 * nrb * (sv - np);
    if (lane == 31) { S.sb[32] = 16 * nrb * sv; S.R = v; }
    for (int w = 0; w < 8; ++w) S.wc[w][lane] += off;
  }
  __syncthreads();
  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) { const int v = __ldg(P.ids + p) - P.lo; e = (v >= 0 && v < 32) ? v : -1; }
      const unsigned m = __match_any_sync(FULL, e);
      const int rank = __popc(m & ((1u << lane) - 1u));
      const int b = (e >= 0) ? S.wc[warp][e] : 0;
      __syncwarp();
      if (e >= 0) {
        const int c = b + rank;
        s_list[c] = (int16_t)p;
        if (write_meta) P.meta[MH + p] = c;
        if (rank == 0) S.wc[warp][e] = b + __popc(m);
      } else if (write_meta && p < NP) {
        P.meta[MH + p] = -1;
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

// Stream-K range of this CTA over the stage sequence [0, S).
__device__ __forceinline__ void my_range2(long long S, int cta, int ncta, int& beg, int& end) {
  beg = (int)(S * (long long)cta / ncta);
  end = (int)(S * (long long)(cta + 1) / ncta);
}
__device__ __forceinline__ void my_range(long long S, int& beg, int& end) {
  const long long G = gridDim.x;
  beg = (int)(S * (long long)blockIdx.x / G);
  end = (int)(S * (long long)(blockIdx.x + 1) / G);
}

// Start of the segment processed first: the item holding the range's last stage (shared with the next CTA) goes
// first, so both contributions to every split item land early and its finalize overlaps the stream, not the tail.
__device__ __forceinline__ int tail_first(const Route& S, int beg, int end) {
  if (end <= beg) return beg;
  int e = 0;
  while (S.sb[e + 1] <= end - 1) ++e;
  const int m = S.sb[e] + ((end - 1 - S.sb[e]) & ~15);
  return max(beg, m);
}

// One kb (16 B of codes) of this warp's tile, B from the staged X slice.
template <int NC>
__device__ __forceinline__ void consume_kb_s(const uint4 w, const uint32_t Sg, const uint32_t S8,
                                             const __nv_bfloat16* xrow, int kloc, int nch, int t, float (*acc)[4]);
template <int NC>
__device__ __forceinline__ void consume_kb(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow,
                                           int kloc, int nch, int t, float (*acc)[4]) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  consume_kb_s<NC>(w, Sg, S8, xrow, kloc, nch, t, acc);
}
// v3.5 compressed scales: per lane per stage, 8 five-bit codes q = 4 kp + 2 par + (row g + 8); nibbles in nw, high bits
// in hb; scale bf16 bits = (base + code) * 16 + 0x7580, i.e. code * 0x00100010 + K (K per expert, both halves).
__device__ __forceinline__ uint32_t sc5(uint32_t nw, uint32_t hb, int q, uint32_t K) {
  const uint32_t c = ((nw >> (4 * q)) & 15u) | (((hb >> q) & 1u) << 4);
  return c * 0x00100010u + K;
}
template <int NC>
__device__ __forceinline__ void consume_kb_s(const uint4 w, const uint32_t Sg, const uint32_t S8,
                                             const __nv_bfloat16* xrow, int kloc, int nch, int t, float (*acc)[4]) {
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
#pragma unroll
  for (int q = 0; q < NC; ++q) {
    if (q < nch) {
      const uint4* xp = reinterpret_cast<const uint4*>(xrow + q * 8 * XST + kloc + 16 * t);
      const uint4 x0 = xp[0], x1 = xp[1];
      mma16816(acc[q], ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
      mma16816(acc[q], ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
      mma16816(acc[q], ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
      mma16816(acc[q], ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
    }
  }
}

// Lab probes: B operand from registers (no X smem reads) / decode only.
template <int NC>
__device__ __forceinline__ void consume_kb_breg(const uint4 w, const uint32_t sw2, int par, uint4 x0, uint4 x1,
                                                int nch, float (*acc)[4], uint32_t& sink, bool mma) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  if (!mma) {
#pragma unroll
    for (int j = 0; j < 8; ++j) sink ^= ag[j] ^ a8[j];
    return;
  }
#pragma unroll
  for (int q = 0; q < NC; ++q) {
    if (q < nch) {
      mma16816(acc[q], ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
      mma16816(acc[q], ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
      mma16816(acc[q], ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
      mma16816(acc[q], ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
    }
  }
}

// MODE 1: W13 (x -> inter / h), MODE 2: W2 (interS -> y).
// v3.2 hot passes: bf16 element index of (pass pid, token n of the pass, k = col) in the wgmma B layout
// (v2.3's no-swizzle K-major core matrices; per stage of 256 k: [n8][kc 32][row 8][16 B], u32 q of the 32 B piece
// k = 64 i + 16 t + 2q is core matrix kc = 8 i + q, u32 t of row n & 7).
__device__ __forceinline__ size_t xw_u32(int pid, int NW, int s2, int n, int i, int t, int q) {
  return (size_t)(pid * 16 + s2) * (NW * 128) + (n >> 3) * 1024 + 256 * i + 32 * q + 4 * (n & 7) + t;
}
__device__ __forceinline__ size_t xw_elem(int pid, int NW, int n, int col) {
  const int kk = col & 255;
  return 2 * xw_u32(pid, NW, col >> 8, n, kk >> 6, (kk >> 4) & 3, (kk >> 1) & 7) + (kk & 1);
}
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
#define K3_WG_HEAD "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned."
template <int N> __device__ __forceinline__ void wgmma_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                          uint64_t db);
template <> __device__ __forceinline__ void wgmma_rs<64>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                         uint64_t db) {
  asm volatile(K3_WG_HEAD "m64n64k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
@WGSPECS@
#undef K3_WG_HEAD
__device__ __forceinline__ void wgmma_nw(int nw, float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
  if (nw <= 16) wgmma_rs<16>(d, a0, a1, a2, a3, db);
  else if (nw <= 32) wgmma_rs<32>(d, a0, a1, a2, a3, db);
  else if (nw <= 48) wgmma_rs<48>(d, a0, a1, a2, a3, db);
  else wgmma_rs<64>(d, a0, a1, a2, a3, db);
}

// v3.3 warm-expert item (all of its stages s0..s1): wgmma m64nNk16, A = this warp's dequantized tile (RS form, the
// mma.m16n8k16 wiring), B = the stage's X slice in core-matrix layout. Half-kb groups (2 wgmmas) alternate between
// two 8-register A buffers (wait_group 1 before reuse); a stage's slot is released after the next stage's second
// wait, the last one after wait_group 0. Columns >= the pass's tokens read stale rows and are discarded.
template <int N, int NST, int STG, int MODE>
__device__ __forceinline__ void hot_item(const Params& P, const Route& S, const uint8_t* ring, uint32_t full_a,
                                      uint32_t empty_a, int& it_ref, int s0, int s1, int e, int j, int rb, int c0,
                                      int ntok, int npart) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  float hacc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) hacc[i] = 0.f;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    const uint8_t* st = ring + slot * STG;
    const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
    const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);
    // per kb: decode into one of two 16-register A buffers, 4 wgmmas, commit; wait_group 1 before a buffer is
    // rewritten, 0 before the slot is released (this form is the one ptxas does not serialize)
    uint32_t hb[2][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 w = wp[i * 32];
      const uint32_t sw2 = sp[(i >> 1) * 32];
      const int par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
      if (i >= 2) wg_wait<1>();
      uint32_t* h = hb[i & 1];
      deq4(w.x, h); deq4(w.y, h + 4); deq4(w.z, h + 8); deq4(w.w, h + 12);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) { h[jj] = bmul2(h[jj], Sg); h[8 + jj] = bmul2(h[8 + jj], S8); }
      wg_fence();
      wgmma_rs<N>(hacc, h[0], h[8], h[1], h[9], d0 + 64 * i);
      wgmma_rs<N>(hacc, h[2], h[10], h[3], h[11], d0 + 64 * i + 16);
      wgmma_rs<N>(hacc, h[4], h[12], h[5], h[13], d0 + 64 * i + 32);
      wgmma_rs<N>(hacc, h[6], h[14], h[7], h[15], d0 + 64 * i + 48);
      wg_commit();
    }
    wg_wait<0>();
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  it_ref = it;
  // ---- flush (same as the cold path's, 8-token chunks q < ceil(ntok / 8)) ----
  const int nch = (ntok + 7) >> 3;
  const int tile = rb * 8 + warp;
  const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
  const bool whole = (s0 == 0 && s1 == 16);
  const int pid = S.hp[e] + j;
#pragma unroll
  for (int q = 0; q < N / 8; ++q) {
    if (q < nch) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = q * 8 + 2 * t + hh;
        if (n < ntok) {
          const int c = c0 + n;
          const float v0 = hacc[4 * q + hh] * gs, v1 = hacc[4 * q + 2 + hh] * gs;
          if constexpr (MODE == 1) {
            const int col = 8 * tile + g;
            if (whole) {
              reinterpret_cast<__nv_bfloat16*>(P.xw2)[xw_elem(pid, 64, n, col)] = __float2bfloat16_rn(swiglu(v0, v1));
            } else {
              atomicAdd(P.h + (size_t)c * (2 * KDIM) + col, v0);
              atomicAdd(P.h + (size_t)c * (2 * KDIM) + KDIM + col, v1);
            }
          } else {
            float* yp = P.y + (size_t)c * KDIM + 16 * tile + g;
            if (whole) { yp[0] = v0; yp[8] = v1; }
            else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
          }
        }
      }
    }
  }
  if (MODE == 2 && whole) {
    const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
    for (int q = 0; q < N / 8; ++q) {
      if (q < nch) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const float z0 = bf16r(hacc[4 * q + hh] * gs) * w0, z1 = bf16r(hacc[4 * q + 2 + hh] * gs) * w1;
          float pz = z0 * z0 + z1 * z1;
          pz += __shfl_xor_sync(FULL, pz, 4);
          pz += __shfl_xor_sync(FULL, pz, 8);
          pz += __shfl_xor_sync(FULL, pz, 16);
          const int n = q * 8 + 2 * t + hh;
          if (g == 0 && n < ntok) atomicAdd(P.ss + c0 + n, pz);
        }
      }
    }
  }
  if (!whole) asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
}

// HM 0: cold items (mma.sync), HM 1 (v3.3): warm-expert items (64-token wgmma passes). S.sb holds the role's stage
// prefix; this CTA is number cta of ncta in its role.
__device__ __forceinline__ int tail_first2(const int* SB, int beg, int end) {
  if (end <= beg) return beg;
  int e = 0;
  while (SB[e + 1] <= end - 1) ++e;
  return max(beg, SB[e] + ((end - 1 - SB[e]) & ~15));
}
__device__ __forceinline__ int count_items(const int* SB, int beg, int end) {
  int n = 0, e = 0;
  for (int x = beg; x < end;) {
    while (SB[e + 1] <= x) ++e;
    const int s0 = (x - SB[e]) & 15;
    x += min(16, s0 + (end - x)) - s0;
    ++n;
  }
  return n;
}
// v3.4: bounded wait for a W13 expert-completion count (acquire); error flag on timeout, never trap
__device__ __forceinline__ void wait_count(const int* p, int need, int* err) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  if (v >= need) return;
  const long long t0 = clock64();
  while (true) {
    __nanosleep(100);
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    if (v >= need) return;
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }
  }
}

// FUSED (v3.4 K12): MODE 1 hands every item to the epilogue warp (expert-completion counts), zeroes its y / ss slice;
// MODE 2 runs in the same CTA right after (stage counter it0, handoff index hbase), its producer waits per item for
// the expert's W13 count before the X copy. Handoffs use named barriers 2 + (h % 12).
template <int NC, int NST, int MODE, int HM, int STG, bool FUSED = false>
__device__ void gemm(const Params& P, Route& S, const int16_t* s_list, uint8_t* ring, uint64_t* bars, bool pdl,
                     int cta, int ncta, int it0 = 0, int hbase = 0) {
  constexpr bool ALLH = FUSED && MODE == 1;   // every item handed off
  const int* SB = (FUSED && MODE == 2) ? S.sb2 : S.sb;
  constexpr bool HOTCAP = (HM == 1);
  constexpr int QMAX = HOTCAP ? 8 : NC;   // accumulator chunks (8 tokens each)
  constexpr int NRB = (MODE == 1) ? 64 : 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long long Stot = SB[32];
  int beg, end;
  my_range2(Stot, cta, ncta, beg, end);
  const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;
  const uint32_t ring_a = sa(ring);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);

  if (warp == 8) {
    // ---------------- producer ----------------
    // X copies that depend on the PDL predecessor (K2: interS / xw2 from K1; K1 hot items: xw from P0) are deferred
    // until griddepcontrol.wait, which is taken lazily: only when a deferred slot would otherwise be reused, or at the
    // end. Weight copies (and K1's cold X copies from the input) never wait.
    int it = it0, e = 0;
    bool waited = !pdl;
    int nd = 0;
    const int pmode = P.probe & 15;
    const bool hint = (P.probe & 16) != 0;
    const uint64_t pol = policy_evict_first();
    uint32_t dDst[NST], dBar[NST];
    const __nv_bfloat16* dSrc[NST];
    uint32_t dB[NST];
    const int mid = tail_first2(SB, beg, end);
    for (int seg = 0; seg < 2; ++seg) {
    const int xb = seg ? beg : mid, xe = seg ? mid : end;
    e = 0;
    for (int x = xb; x < xe;) {
      while (SB[e + 1] <= x) ++e;
      const int np = S.np[e];
      constexpr bool hot = HOTCAP;
      const int TP = hot ? 64 : 8 * NC;
      const int l = x - SB[e];
      const int item = l >> 4;
      const int s0 = l & 15;
      const int s1 = min(16, s0 + (xe - x));
      const int rb = item / np, j = item - rb * np;
      const int ntok = min(TP, S.cnt[e] - j * TP);
      const int c0 = S.off[e] + j * TP;
      const int kfmt = (!HOTCAP && P.sfmt != nullptr) ? __ldg(P.sfmt + (MODE == 1 ? 0 : 32) + e) : 0;
      const uint32_t swb = kfmt ? SWC : SW;   // v3.5: compressed stages are 17664 B (5-bit scales)
      const uint8_t* wb = ((!HOTCAP && P.sfmt != nullptr) ? ((MODE == 1) ? P.ws13c : P.ws2c) : wsrc) +
                          ((size_t)(e * NRB + rb) * 16) * SW;
      const int ntx = (pmode == 5) ? 0 : ntok;
      int tok = 0;
      if (MODE == 1 && NC <= 2 && !hot && lane >= 1 && lane <= ntx) tok = ((int)s_list[c0 + lane - 1]) >> 3;
      const uint32_t xbytes = hot ? ((ntok + 7) >> 3) * 4096 : (MODE == 1) ? ntx * 512 : ntok * XROW;
      for (int s = s0; s < s1; ++s, ++it) {
        const int slot = it % NST;
        const uint32_t par = ((it / NST) & 1) ^ 1;
        if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now
          pdl_wait();
          asm volatile("fence.proxy.async;\n" ::: "memory");
          waited = true;
          if (lane == 0) {
#pragma unroll
            for (int d = 0; d < NST; ++d)
              if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
          }
          nd = 0;
        }
        mbar_wait(empty_a + 8 * slot, par, P.err);
        const uint32_t dst = ring_a + slot * STG;
        const uint32_t fb = full_a + 8 * slot;
        if (lane == 0) {
          mbar_expect_tx(fb, swb + xbytes);
          const uint8_t* src = wb + (size_t)s * SW;
          if (hint) bulk_g2s_ef(dst, src, swb, fb, pol);
          else bulk_g2s(dst, src, swb, fb);
        }
        mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals
        __syncwarp();
        if (MODE == 1 && !hot) {
          if constexpr (NC <= 2) {
            if (lane >= 1 && lane <= ntx)
              bulk_g2s(dst + SW + (lane - 1) * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);
          } else {   // up to 8 NC rows: lane i copies rows i, i + 32
            for (int i = lane; i < ntx; i += 32)
              bulk_g2s(dst + SW + i * XROW, P.x + (size_t)(((int)s_list[c0 + i]) >> 3) * KDIM + s * 256, 512, fb);
          }
        } else {
          if (FUSED && MODE == 2 && s == s0) {   // the expert's W13 items must be complete (interS written)
            if (lane == 0) {
              wait_count(P.cnt13 + e, 64 * np, P.err);
              asm volatile("fence.proxy.async.global;\n" ::: "memory");
            }
            __syncwarp();
          }
          const int pid = S.hp[e] + j;
          const __nv_bfloat16* src = hot
              ? reinterpret_cast<const __nv_bfloat16*>(((MODE == 1) ? P.xw : P.xw2) + (size_t)(pid * 16 + s) * (64 * 512))
              : P.inter + ((size_t)s * (8 * M3) + c0) * XST;
          if (waited) {
            if (lane == 0) bulk_g2s(dst + SW, src, xbytes, fb);
          } else {
#pragma unroll
            for (int d = 0; d < NST; ++d)
              if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = src; dB[d] = xbytes; }
            ++nd;
          }
        }
      }
      x += s1 - s0;
    }
    }
    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain
    if (!waited) {
      pdl_wait();
      asm volatile("fence.proxy.async;\n" ::: "memory");
      if (lane == 0) {
#pragma unroll
        for (int d = 0; d < NST; ++d)
          if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
      }
    }
    return;
  }

  if (warp == 9) {
    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item ----------------
    // K1: SwiGLU(h) -> interS, h -> 0. K2: per-route sum of squares of z = bf16(y)*(1+w) over the item's 128 columns.
    {
      if ((P.probe & 15) != 0) return;   // probes: consumers never hand over
      int npart = hbase, e = 0;
      const int mid = tail_first2(SB, beg, end);
      for (int seg = 0; seg < 2; ++seg) {
        const int xb = seg ? beg : mid, xe = seg ? mid : end;
        e = 0;
        for (int x = xb; x < xe;) {
          while (SB[e + 1] <= x) ++e;
          const int np = S.np[e];
          const int l = x - SB[e];
          const int item = l >> 4;
          const int s0 = l & 15;
          const int s1 = min(16, s0 + (xe - x));
          x += s1 - s0;
          const bool whole_item = (s0 == 0 && s1 == 16);
          if (whole_item && !ALLH) continue;
          constexpr bool hot = HOTCAP;
          const int TP = hot ? 64 : 8 * NC;
          const int rb = item / np, j = item - rb * np;
          const int ntok = min(TP, S.cnt[e] - j * TP);
          const int c0 = S.off[e] + j * TP;
          asm volatile("bar.sync %0, 288;\n" :: "r"(2 + (npart % 12)) : "memory");
          ++npart;
          __threadfence();   // cumulative over the consumers' h atomics (ordered before us by the barrier)
          if (ALLH && whole_item) {   // fused K12: one more completed W13 item of expert e
            if (lane == 0) atomicAdd(P.cnt13 + e, 1);
            continue;
          }
          int last = 0;
          if (lane == 0) {
            int* cp = ((MODE == 1) ? P.cnt : P.cnt2) + (e * NRB + rb) * 64 + j;
            const int old = atomicAdd(cp, s1 - s0);
            last = (old + (s1 - s0) == 16);
            if (last) atomicExch(cp, 0);
          }
          last = __shfl_sync(FULL, last, 0);
          if (MODE == 2 && last) {
            __threadfence();
            const int col = rb * 128 + 4 * lane;
            const float4 wv = __ldg(reinterpret_cast<const float4*>(P.nw + col));
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float4 yv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u)
                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                float pz = 0.f;
                if (n0 + u < ntok) {
                  float z;
                  z = bf16r(yv[u].x) * wv.x; pz += z * z;  z = bf16r(yv[u].y) * wv.y; pz += z * z;
                  z = bf16r(yv[u].z) * wv.z; pz += z * z;  z = bf16r(yv[u].w) * wv.w; pz += z * z;
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pz += __shfl_xor_sync(FULL, pz, o);
                if (lane == 0 && n0 + u < ntok) atomicAdd(P.ss + c0 + n0 + u, pz);
              }
            }
          }
          if (MODE == 1 && last) {
            __threadfence();
            // ntok * 64 columns; lane handles column pair (2*(lane&31)) of 64 for tokens n = 0..ntok-1, batched
            const int cl = 2 * lane;
            const int col = rb * 64 + cl;
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float2 gv[4], uv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const float* hp = P.h + (size_t)(c0 + n0 + u) * (2 * KDIM) + col;
                  gv[u] = __ldcg(reinterpret_cast<const float2*>(hp));
                  uv[u] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
                }
              }
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const int c = c0 + n0 + u;
                  float* hp = P.h + (size_t)c * (2 * KDIM) + col;
                  const uint32_t pk = pack_bf2(swiglu(gv[u].x, uv[u].x), swiglu(gv[u].y, uv[u].y));
                  if (hot) {
                    *reinterpret_cast<uint32_t*>(P.xw2 + 2 * xw_elem(S.hp[e] + j, 64, n0 + u, col)) = pk;
                  } else {
                    *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
                  }
                  *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
                  *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
                }
              }
            }
            if (ALLH) {
              asm volatile("fence.proxy.async.global;\n" ::: "memory");
              __threadfence();
              __syncwarp();
              if (lane == 0) atomicAdd(P.cnt13 + e, 1);
            }
          }
        }
      }
    }
    return;
  }

  // ---------------- consumers (warps 0-7: warp w = tile w of the row block) ----------------
  if (MODE == 1 && !FUSED && ((P.probe & 15) == 0 || (P.probe & 15) >= 6)) {
    // y rows [0, R) and ss[0, R) -> 0 while the first stages are in flight (K2 accumulates into both after its wait)
    const size_t n4 = (size_t)S.R * KDIM / 4;
    float4* yp = reinterpret_cast<float4*>(P.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)blockIdx.x * 256 + tid; i < n4; i += (size_t)gridDim.x * 256) yp[i] = z;
    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;
  }
  const int g = lane >> 2, t = lane & 3;
  int it = it0, e = 0;
  uint32_t sink = 0u;
  int npart = hbase;
  const int mid = tail_first2(SB, beg, end);
  for (int seg = 0; seg < 2; ++seg) {
  const int xb = seg ? beg : mid, xe = seg ? mid : end;
  e = 0;
  for (int x = xb; x < xe;) {
    while (SB[e + 1] <= x) ++e;
    const int np = S.np[e];
    const int l = x - SB[e];
    const int item = l >> 4;
    const int s0 = l & 15;
    const int s1 = min(16, s0 + (xe - x));
    constexpr bool hot = HOTCAP;
    const int TP = hot ? 64 : 8 * NC;
    const int rb = item / np, j = item - rb * np;
    const int ntok = min(TP, S.cnt[e] - j * TP);
    const int nch = (ntok + 7) >> 3;
    const int c0 = S.off[e] + j * TP;
    const int nw = ((ntok + 15) >> 4) << 4;   // hot: wgmma N in {16, 32, 48, 64}
    const int kfmt = (!HOTCAP && P.sfmt != nullptr) ? __ldg(P.sfmt + (MODE == 1 ? 0 : 32) + e) : 0;
    float acc[QMAX][4], acc2[NC][4];   // acc2: second mma chain (NC == 1 only)
    if constexpr (HOTCAP) {
      // warm-expert item on wgmma (this CTA's role has only warm items)
      if ((P.probe & 15) == 0) {
        if (nw <= 16) hot_item<16, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else if (nw <= 32) hot_item<32, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else if (nw <= 48) hot_item<48, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else hot_item<64, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
      }
      if (!(s0 == 0 && s1 == 16)) ++npart;
      x += s1 - s0;
      continue;
    } else {
#pragma unroll
    for (int q = 0; q < QMAX; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[q][i] = 0.f;
#pragma unroll
    for (int q = 0; q < NC; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc2[q][i] = 0.f;
    for (int s = s0; s < s1; ++s, ++it) {
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
      const uint8_t* st = ring + slot * STG;
      const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
      const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;
      if ((P.probe & 15) == 1 || (P.probe & 15) == 5) {
        const uint4 a = wp[0], b = wp[96];
        sink ^= a.x ^ b.w ^ sp[0];
      } else if ((P.probe & 15) == 11 || (P.probe & 15) == 12) {
        // lab: 11 = decode half of the words (x, y) + scale, no mma; 12 = mma on raw words (no decode), B from smem
        uint4 wv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
        const uint32_t s0v = sp[0], s1v = sp[32];
        if ((P.probe & 15) == 11) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t sw2 = (i < 2) ? s0v : s1v;
            const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * (i & 1)) * 16u + 0x75807580u;
            uint32_t a[8];
            deq4(wv[i].x, a); deq4(wv[i].y, a + 4);
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) sink ^= bmul2(a[jj], Sg);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint4* xp = reinterpret_cast<const uint4*>(xrow + i * 64 + 16 * t);
            const uint4 x0 = xp[0], x1 = xp[1];
            const uint4 w = wv[i];
            mma16816(acc[0], w.x, w.z, w.y ^ s0v, w.w, x0.x, x0.y);
            mma16816(acc[0], w.y, w.w, w.x, w.z ^ s1v, x0.z, x0.w);
            mma16816(acc[0], w.z, w.x, w.w, w.y, x1.x, x1.y);
            mma16816(acc[0], w.w, w.y, w.z, w.x, x1.z, x1.w);
          }
        }
      } else if ((P.probe & 15) == 9 || (P.probe & 15) == 10) {
        uint4 wv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
        const uint32_t s0v = sp[0], s1v = sp[32];
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_a + 8 * slot);   // operands in registers: release the slot first
        const uint4 xb0 = make_uint4(s0v, s1v, s0v ^ 1u, s1v ^ 1u), xb1 = make_uint4(s1v, s0v, s1v ^ 2u, s0v ^ 2u);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          consume_kb_breg<NC>(wv[i], (i < 2) ? s0v : s1v, i & 1, xb0, xb1, nch, (i < 2 || NC > 1) ? acc : acc2, sink,
                              (P.probe & 15) == 9);
        continue;
      } else if (NC == 1 && V3_EARLY) {
        // v3.1: all of this warp's operands of the stage (codes, scales, X fragments) into registers, release the
        // slot, then decode + mma, so the ring cycle no longer includes the consumer's math.
        uint4 wv[4], x0[4], x1[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          wv[i] = wp[i * 32];
          const uint4* xp = reinterpret_cast<const uint4*>(xrow + i * 64 + 16 * t);
          x0[i] = xp[0]; x1[i] = xp[1];
        }
        const uint32_t s0v = sp[0], s1v = sp[32];
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_a + 8 * slot);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          consume_kb_breg<NC>(wv[i], (i < 2) ? s0v : s1v, i & 1, x0[i], x1[i], nch, (i < 2 || NC > 1) ? acc : acc2, sink, true);
        continue;
      } else {
        uint4 wv[4];
        uint32_t sv[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
        if (kfmt != 0) {   // v3.5: 5-bit scales (nibble word + high-bit byte per lane)
          const uint32_t nw = *reinterpret_cast<const uint32_t*>(st + 16384 + warp * 128 + lane * 4);
          const uint32_t hb = st[16384 + 1024 + warp * 32 + lane];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int q = (i >> 1) * 4 + 2 * (i & 1);
            consume_kb_s<NC>(wv[i], sc5(nw, hb, q, (uint32_t)kfmt), sc5(nw, hb, q + 1, (uint32_t)kfmt), xrow, i * 64,
                             nch, t, (i < 2 || NC > 1) ? acc : acc2);
          }
        } else {
          sv[0] = sp[0]; sv[1] = sp[32];
#pragma unroll
          for (int i = 0; i < 4; ++i) consume_kb<NC>(wv[i], sv[i >> 1], i & 1, xrow, i * 64, nch, t, (i < 2 || NC > 1) ? acc : acc2);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
    }
    // ---- flush ----
#pragma unroll
    for (int q = 0; q < NC; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) if (NC == 1) acc[q][i] += acc2[q][i];
    const int fmode = P.probe & 15;   // 6: partial items stored like whole ones (timing only); 7: no finalize
    if (fmode != 0 && fmode < 6) {
#pragma unroll
      for (int q = 0; q < QMAX; ++q) sink ^= __float_as_uint(acc[q][0] + acc[q][1] + acc[q][2] + acc[q][3]);
    } else {
      const int tile = rb * 8 + warp;
      const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
      const bool whole = (s0 == 0 && s1 == 16) || fmode == 6;
      if constexpr (MODE == 1) {
#pragma unroll
        for (int q = 0; q < QMAX; ++q) {
          if (q < nch) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int n = q * 8 + 2 * t + hh;
              if (n < ntok) {
                const int c = c0 + n;
                const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
                const int col = 8 * tile + g;
                if (whole && hot) {
                  reinterpret_cast<__nv_bfloat16*>(P.xw2)[xw_elem(S.hp[e] + j, 64, n, col)] = __float2bfloat16_rn(swiglu(v0, v1));
                } else if (whole) {
                  P.inter[((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));
                } else {
                  atomicAdd(P.h + (size_t)c * (2 * KDIM) + col, v0);
                  atomicAdd(P.h + (size_t)c * (2 * KDIM) + KDIM + col, v1);
                }
              }
            }
          }
        }
        if (FUSED) {   // zero this item's y / ss slice (W2 of the same expert runs only after the expert's W13 count)
          for (int idx = tid; idx < ntok * 16; idx += 256)
            *reinterpret_cast<float4*>(P.y + (size_t)(c0 + (idx >> 4)) * KDIM + 64 * rb + 4 * (idx & 15)) =
                make_float4(0.f, 0.f, 0.f, 0.f);
          if (rb == 0 && tid < ntok) P.ss[c0 + tid] = 0.f;
          asm volatile("fence.proxy.async.global;\n" ::: "memory");   // interS: generic writes, TMA reads
        }
        if ((!whole || ALLH) && fmode != 7) {
          // hand the item to the epilogue warp (warp 9): non-blocking for the consumers
          asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + (npart % 12)) : "memory");
        }
      } else {
#pragma unroll
        for (int q = 0; q < QMAX; ++q) {
          if (q < nch) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int n = q * 8 + 2 * t + hh;
              if (n < ntok) {
                float* yp = P.y + (size_t)(c0 + n) * KDIM + 16 * tile + g;
                const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
                if (whole) { yp[0] = v0; yp[8] = v1; }
                else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
              }
            }
          }
        }
        if (whole) {
          // sum of squares of z = bf16(y) * (1 + w) over this item's 128 columns, per route (one red per warp)
          const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
          for (int q = 0; q < QMAX; ++q) {
            if (q < nch) {
#pragma unroll
              for (int hh = 0; hh < 2; ++hh) {
                const float z0 = bf16r(acc[q][hh] * gs) * w0, z1 = bf16r(acc[q][2 + hh] * gs) * w1;
                float pz = z0 * z0 + z1 * z1;
                pz += __shfl_xor_sync(FULL, pz, 4);
                pz += __shfl_xor_sync(FULL, pz, 8);
                pz += __shfl_xor_sync(FULL, pz, 16);
                const int n = q * 8 + 2 * t + hh;
                if (g == 0 && n < ntok) atomicAdd(P.ss + c0 + n, pz);
              }
            }
          }
        } else if (fmode != 7) {
          asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + (npart % 12)) : "memory");
        }
      }
    }
    if (!(s0 == 0 && s1 == 16) || ALLH) ++npart;
    x += s1 - s0;
  }
  }
  if (sink == 0x9E3779B9u) atomicAdd(P.sink, 1u);   // keeps probe work alive; never true in practice
}

template <int NC, int NST>
__device__ __forceinline__ void setup_bars(uint64_t* bars) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + NST + i), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
}

// v3.3 role split (HOTCAP): the first Gh CTAs take the warm experts' stages (wgmma), the rest the cold stages; Gh is
// proportional to the work (a warm stage weighs ~2 cold ones). Each CTA then runs one code path only (an item-level
// branch between the two made ptxas serialize every wgmma: C7520 / C7513).
template <int NC>
__device__ __forceinline__ bool pick_role(Route& S, int P_hot_w4, int hot_t, int& cta, int& ncta) {
  // work model: a stage costs its bulk bytes, 36 half-KB of codes + scales plus 1 half-KB per staged token row
  // (warm rows rounded to 8), warm work scaled by hot_w4 / 8
  const long long Sh = S.sbh[32], Sc = S.sbc[32];
  const int G = (int)gridDim.x;
  int Gh = 0;
  if (Sh > 0) {
    long long Wh = 0, Wc = 0;
    for (int e = 0; e < 32; ++e) {
      const int c = S.cnt[e];
      if (c == 0) continue;
      if (c > hot_t) {
        const int full = c >> 6, rem = c & 63;
        Wh += (long long)full * (36 + 64) + (rem ? 36 + ((rem + 7) & ~7) : 0);
      } else {
        const int tp = 8 * NC, full = c / tp, rem = c - full * tp;
        Wc += (long long)full * (36 + tp) + (rem ? 36 + rem : 0);
      }
    }
    Wh *= P_hot_w4;
    Wc *= 8;
    Gh = (int)((Wh * G + (Wh + Wc) / 2) / (Wh + Wc));
    Gh = max(1, min(Gh, Sc > 0 ? G - 1 : G));
  }
  const bool hr = (int)blockIdx.x < Gh;
  cta = hr ? (int)blockIdx.x : (int)blockIdx.x - Gh;
  ncta = hr ? Gh : G - Gh;
  __syncthreads();
  if (threadIdx.x < 33) S.sb[threadIdx.x] = hr ? S.sbh[threadIdx.x] : S.sbc[threadIdx.x];
  __syncthreads();
  return hr;
}

// K1: route + W13 (+ SwiGLU). HOTCAP (v3.3): warm experts on wgmma CTAs, their X from P0 (PDL chain, lazy wait).
template <int NC, int NST, int MINB, bool HOTCAP>
__global__ void __launch_bounds__(320, MINB) k1_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = HOTCAP ? SW + 32768 : SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  if constexpr (!HOTCAP) setup_bars<NC, NST>(bars);
  long long clk0 = 0, ns0 = 0;
  if ((P.probe & 128) && threadIdx.x == 0) {   // lab: SM clock over the kernel (sum of clocks / sum of ns, all CTAs)
    clk0 = clock64();
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns0));
  }
  route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  if constexpr (HOTCAP) {
    // cold-role CTAs re-cut the same ring into more, smaller stages (no 32 KB warm X slot)
    constexpr int STGC = SW + 8 * NC * XROW, NSTC = (NST * STG) / STGC;
    int cta, ncta;
    const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);
    if (threadIdx.x == 0) {
      const int n = hr ? NST : NSTC;
      for (int i = 0; i < n; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + n + i), 8); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (hr) gemm<NC, NST, 1, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);
    else gemm<NC, NSTC, 1, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);
  } else {
    gemm<NC, NST, 1, 0, STG>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);
  }
  if ((P.probe & 128) && threadIdx.x == 0) {
    long long ns1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns1));
    atomicAdd(reinterpret_cast<unsigned long long*>(P.sink), (unsigned long long)(clock64() - clk0));
    atomicAdd(reinterpret_cast<unsigned long long*>(P.sink) + 1, (unsigned long long)(ns1 - ns0));
  }
}

// K2: route + W2. PDL: weights of the first NST stages are requested before griddepcontrol.wait.
template <int NC, int NST, int MINB, bool HOTCAP>
__global__ void __launch_bounds__(320, MINB) k2_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = HOTCAP ? SW + 32768 : SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  if constexpr (!HOTCAP) setup_bars<NC, NST>(bars);
  route<NC>(P, S, nullptr, s_list, 32, false);
  if constexpr (HOTCAP) {
    constexpr int STGC = SW + 8 * NC * XROW, NSTC = (NST * STG) / STGC;
    int cta, ncta;
    const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);
    if (threadIdx.x == 0) {
      const int n = hr ? NST : NSTC;
      for (int i = 0; i < n; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + n + i), 8); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (hr) gemm<NC, NST, 2, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);
    else gemm<NC, NSTC, 2, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);
  } else {
    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);
  }
}

// K12 (v3.4): K1 and K2 fused into one launch (cold path only). Each CTA streams its W13 range, then its W2 range in
// the same ring; a W2 item's X copy waits for its expert's W13 completion count (all CTAs are co-resident: grid =
// SMs x occupancy, bounded waits). Removes the K1 -> K2 boundary (ramp + tail + second routing).
template <int NC, int NST, int MINB>
__global__ void __launch_bounds__(320, MINB) k12_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<NC, NST>(bars);
  route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  if (threadIdx.x < 33) S.sb2[threadIdx.x] = S.sb[threadIdx.x] >> 1;
  __syncthreads();
  int b1, e1;
  my_range2(S.sb[32], blockIdx.x, gridDim.x, b1, e1);
  const int h1 = count_items(S.sb, b1, e1);
  gemm<NC, NST, 1, 0, STG, true>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x, 0, 0);
  gemm<NC, NST, 2, 0, STG, true>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x, e1 - b1, h1);
}

// P0 (v3.3): warm experts' x rows packed into wgmma's core-matrix layout (64-token passes), one 32 B piece per thread.
template <int NC>
__global__ void __launch_bounds__(256) p0_kernel(const Params P) {
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  route<NC>(P, S, nullptr, s_list, 64, false);
  int tot = 0;
  for (int e = 0; e < 32; ++e) if (S.cnt[e] > P.hot_t) tot += S.cnt[e];
  uint32_t* xw = reinterpret_cast<uint32_t*>(P.xw);
  for (int idx = blockIdx.x * 256 + threadIdx.x; idx < tot * 256; idx += gridDim.x * 256) {
    int r = idx >> 8;
    const int piece = idx & 255, s2 = piece >> 4, i = (piece >> 2) & 3, t = piece & 3;
    int e = 0;
    for (; e < 32; ++e) {
      if (S.cnt[e] <= P.hot_t) continue;
      if (r < S.cnt[e]) break;
      r -= S.cnt[e];
    }
    const int j = r >> 6, n = r & 63;
    const int tok = ((int)s_list[S.off[e] + r]) >> 3;
    const uint4* src = reinterpret_cast<const uint4*>(P.x + (size_t)tok * KDIM + s2 * 256 + 64 * i + 16 * t);
    const uint4 a = __ldg(src), b = __ldg(src + 1);
    const uint32_t v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    const size_t u0 = xw_u32(S.hp[e] + j, 64, s2, n, i, t, 0);
#pragma unroll
    for (int q = 0; q < 8; ++q) xw[u0 + 32 * q] = v[q];
  }
}

// K3: post-expert norm + weighted sum, one pass (v3.1): the per-route sums of squares come from K2 (ss), so an item
// (token, column quarter) needs one batch of meta / weights / ss loads and one batch of y loads. PDL wait first.
__global__ void __launch_bounds__(256) k3_kernel(const Params P) {
  __shared__ int s_fc[8];
  __shared__ float s_fr[8];
  pdl_wait();
  const int tid = threadIdx.x;
  if (blockIdx.x == 0 && tid < 32) P.cnt13[tid] = 0;   // v3.4 K12 expert counts (K12 has completed)
  const int items = 4 * P.M;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int tok = it >> 2;
    if (tid < 8) {
      const int c = __ldcg(P.meta + MH + tok * 8 + tid);
      const float wt = __ldg(P.wts + tok * 8 + tid);
      s_fc[tid] = c;
      s_fr[tid] = (c >= 0) ? rsqrtf(__ldcg(P.ss + c) * (1.f / 4096.f) + 1e-8f) * wt : 0.f;
    }
    __syncthreads();
    const int col = (it & 3) * 1024 + 4 * tid;
    float4 v[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int c = s_fc[s];
      v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
    }
    const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const float r = s_fr[s];
      a0 += bf16r(v[s].x) * w.x * r; a1 += bf16r(v[s].y) * w.y * r;
      a2 += bf16r(v[s].z) * w.z * r; a3 += bf16r(v[s].w) * w.w * r;
    }
    uint2 o;
    o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
    *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;
    __syncthreads();
  }
}

}  // namespace k3

template <int NC, int NST, bool HOT = false>
static constexpr int smem_bytes() { return NST * (k3::SW + 8 * NC * (HOT ? 512 : k3::XROW)) + 2 * NST * 8; }

// mask: 1 K1, 2 K2, 4 K3 (7 = full call). HOTCAP variants (v3.3) launch P0 first and K1 as a PDL secondary.
template <int NC, int NST, int MINB, bool HOTCAP>
static constexpr int smem_bytes3() {
  return HOTCAP ? NST * (k3::SW + 32768) + 2 * 8 * (NST * (k3::SW + 32768) / (k3::SW + 8 * NC * k3::XROW))
                : NST * (k3::SW + 8 * NC * k3::XROW) + 2 * NST * 8;
}

template <int NC, int NST, int MINB, int HOTCAP>
static void launch3(const k3::Params& p, int G, int G3, int mask) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes3<NC, NST, MINB, HOTCAP == 1>();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if (HOTCAP == 2) {   // v3.4 fused K12 (+ K3)
    if (mask & 1) {
      k3::k12_kernel<NC, NST, MINB><<<G, 320, SM, st>>>(p);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (mask & 4) {
      cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));
    }
    return;
  }
  if (mask & 1) {
    if (HOTCAP == 1) {
      k3::p0_kernel<NC><<<G3, 256, 0, st>>>(p);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel<NC, NST, MINB, HOTCAP == 1>, p));
    } else {
      k3::k1_kernel<NC, NST, MINB, HOTCAP == 1><<<G, 320, SM, st>>>(p);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  if (mask & 2) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel<NC, NST, MINB, HOTCAP == 1>, p));
  }
  if (mask & 4) {
    cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));
  }
}

template <int NC, int NST, int MINB, int HOTCAP>
static std::vector<int64_t> attrs3() {
  constexpr int SM = smem_bytes3<NC, NST, MINB, HOTCAP == 1>();
  std::vector<int64_t> r;
  auto one = [&](const void* f, int nt) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, SM) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM), (int64_t)occ});
  };
  if (HOTCAP == 2) {
    one((const void*)k3::k12_kernel<NC, NST, MINB>, 320);
    one((const void*)k3::k12_kernel<NC, NST, MINB>, 320);
  } else {
    one((const void*)k3::k1_kernel<NC, NST, MINB, HOTCAP == 1>, 320);
    one((const void*)k3::k2_kernel<NC, NST, MINB, HOTCAP == 1>, 320);
  }
  return r;
}

void run3(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
          torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
          torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
          torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw,
          torch::Tensor xw2, torch::Tensor cnt13, torch::Tensor ws13c, torch::Tensor ws2c, torch::Tensor sfmt,
          int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe) {
  k3::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ids = ids.data_ptr<int>();
  p.wts = wts.data_ptr<float>();
  p.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  p.ws2 = reinterpret_cast<const uint8_t*>(ws2.data_ptr());
  p.gsc13 = gsc13.data_ptr<float>();
  p.gsc2 = gsc2.data_ptr<float>();
  p.nw = nw.data_ptr<float>();
  p.h = h.data_ptr<float>();
  p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  p.y = y.data_ptr<float>();
  p.meta = meta.data_ptr<int>();
  p.cnt = cnt.data_ptr<int>();
  p.err = err.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.sink = reinterpret_cast<unsigned*>(sink.data_ptr());
  p.ss = ss.data_ptr<float>();
  p.cnt2 = cnt2.data_ptr<int>();
  p.xw = reinterpret_cast<uint8_t*>(xw.data_ptr());
  p.xw2 = reinterpret_cast<uint8_t*>(xw2.data_ptr());
  p.cnt13 = cnt13.data_ptr<int>();
  p.ws13c = reinterpret_cast<const uint8_t*>(ws13c.data_ptr());
  p.ws2c = reinterpret_cast<const uint8_t*>(ws2c.data_ptr());
  p.sfmt = (sfmt.numel() >= 64) ? sfmt.data_ptr<int>() : nullptr;
  p.hot_t = (int)(hot_t % 1000);
  p.hot_w4 = (hot_t >= 1000) ? (int)(hot_t / 1000) : 8;   // x8
  p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;
  switch (vid) {
@CASES3@
    default: TORCH_CHECK(false, "k3: unknown variant");
  }
}

std::vector<int64_t> attrs3v(int64_t vid) {
  switch (vid) {
@ATTRS3@
    default: return std::vector<int64_t>{-1};
  }
}
"""

CPP_SRC_V3 = r"""
#include <torch/extension.h>
#include <vector>
void run3(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
          torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
          torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
          torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw,
          torch::Tensor xw2, torch::Tensor cnt13, torch::Tensor ws13c, torch::Tensor ws2c, torch::Tensor sfmt,
          int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe);
std::vector<int64_t> attrs3v(int64_t vid);
"""


def _wg_specs():
    """wgmma.mma_async m64nNk16 (RS form) specializations for N = 16, 32, 48 (64 is hand-written above)."""
    out = []
    for n in (16, 32, 48):
        k = n // 2
        regs = ",".join("%%%d" % i for i in range(k))
        dl = ", ".join('"+f"(d[%d])' % i for i in range(k))
        out.append(
            "template <> __device__ __forceinline__ void wgmma_rs<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
            "uint32_t a3, uint64_t db) {\n  asm volatile(K3_WG_HEAD \"m64n%dk16.f32.bf16.bf16 {%s}, {%%%d,%%%d,%%%d,%%%d}, "
            "%%%d, p, 1, 1, 0;\\n}\\n\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(db));\n}"
            % (n, n, regs, k, k + 1, k + 2, k + 3, k + 4, dl))
    return "\n".join(out)


def build_ext_v3(name, variants, verbose=False):
    from torch.utils.cpp_extension import load_inline
    vv = [tuple(v) + (0,) * (4 - len(v)) for v in variants]
    cases = "\n".join("    case %d: launch3<%d, %d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, *v)
                      for i, v in enumerate(vv))
    attrs = "\n".join("    case %d: return attrs3<%d, %d, %d, %d>();" % (i, *v) for i, v in enumerate(vv))
    src = KERNEL_SRC_V3.replace("@CASES3@", cases).replace("@ATTRS3@", attrs).replace("@WGSPECS@", _wg_specs())
    tag = hashlib.sha1((CPP_SRC_V3 + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC_V3, cuda_sources=src, functions=["run3", "attrs3v"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"], verbose=verbose)   # wgmma: sm_90a
    return mod, time.time() - t0


def v3_prepare(weights, st2=None, variants=V3_VARIANTS, name="k2a_v35", log=print):
    """Stage repack (from v2.2's repacked tensors when given, else repacked here) + build + occupancy."""
    dev = weights["w13"].device
    t0 = time.time()
    ws13 = torch.empty((E_LOC, 64, 16, SW3), dtype=torch.uint8, device=dev)
    ws2 = torch.empty((E_LOC, 32, 16, SW3), dtype=torch.uint8, device=dev)
    ws13c = torch.empty((E_LOC, 64, 16, SW3) if V3_SCALE5 else (1,), dtype=torch.uint8, device=dev)
    ws2c = torch.empty((E_LOC, 32, 16, SW3) if V3_SCALE5 else (1,), dtype=torch.uint8, device=dev)
    sfmt_h = [0] * 64
    sfmt = sfmt_h
    nc5 = 0
    for e in range(E_LOC):
        w13q, s13q = prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True, bitpos=BITPOS3)
        w2q, s2q = prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False, bitpos=BITPOS3)
        ws13[e] = stage_repack(w13q, s13q)
        ws2[e] = stage_repack(w2q, s2q)
        if V3_SCALE5:   # per (matrix, expert): 5-bit codes when every b' fits one 32-value window, else plain
            for mi, (wq_, sq_, dst) in enumerate(((w13q, s13q, ws13c), (w2q, s2q, ws2c))):
                lo, hi = int(sq_.min()), int(sq_.max())
                base = max(0, min(lo, hi - 31))
                if hi - base <= 31 and lo >= base:
                    dst[e] = stage_repack5(wq_, sq_, base)
                    sfmt[32 * mi + e] = (base * 16 + 0x7580) * 0x10001
                    nc5 += 1
                else:
                    dst[e] = stage_repack(wq_, sq_)
    st = dict(
        ws13=ws13, ws2=ws2, ws13c=ws13c, ws2c=ws2c,
        sfmt=torch.tensor(sfmt_h, dtype=torch.int32, device=dev) if V3_SCALE5 else None,
        nofmt=torch.zeros((1,), dtype=torch.int32, device=dev),
        gsc13=(weights["w13_gs"][:E_LOC].float() * W_SHIFT).contiguous(),
        gsc2=(weights["w2_gs"][:E_LOC].float() * W_SHIFT).contiguous(),
        nw=(weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous(),
        h=torch.zeros((8 * M3, 2 * D_EXPERT), dtype=torch.float32, device=dev),
        inter=torch.zeros((16 * 8 * M3 * 264,), dtype=torch.bfloat16, device=dev),
        y=torch.zeros((8 * M3, D_HIDDEN), dtype=torch.float32, device=dev),
        meta=torch.zeros((META_HDR + 8 * M3,), dtype=torch.int32, device=dev),
        cnt=torch.zeros((E_LOC * 64 * 64,), dtype=torch.int32, device=dev),
        err=torch.zeros((4,), dtype=torch.int32, device=dev),
        sink=torch.zeros((4,), dtype=torch.int32, device=dev),
        ss=torch.zeros((8 * M3,), dtype=torch.float32, device=dev),
        cnt2=torch.zeros((E_LOC * 32 * 64,), dtype=torch.int32, device=dev),
        cnt13=torch.zeros((32,), dtype=torch.int32, device=dev),
        # warm passes (> HOT_T tokens, 64 per pass): <= 8 * M3 / (HOT_T + 1) + 32 passes x 16 stages x 64 x 512 B
        xw=torch.zeros(((8 * M3 // (HOT_T + 1) + 32) * 16 * 64 * 512,), dtype=torch.uint8, device=dev),
        xw2=torch.zeros(((8 * M3 // (HOT_T + 1) + 32) * 16 * 64 * 512,), dtype=torch.uint8, device=dev),
    )
    torch.cuda.synchronize()
    log(f"[k2a3] stage repack in {time.time() - t0:.1f}s; 5-bit scales on {nc5} of 64 (matrix, expert)")
    mod, dt = build_ext_v3(name, variants)
    log(f"[k2a3] compiled {len(variants)} variants in {dt:.1f}s")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = []
    for i, v in enumerate(variants):
        a = mod.attrs3v(i)
        occ = max(1, min(a[3], a[7], v[2]))
        grid.append(sms * occ)
        log(f"[k2a3] v{i} {tuple(v)}: K1 regs {a[0]} spill {a[1]} smem {a[2]} occ {a[3]} | K2 regs {a[4]} "
            f"spill {a[5]} smem {a[6]} occ {a[7]} -> grid {grid[-1]}")
    st.update(mod=mod, grid=grid, sms=sms, variants=variants)
    return st


V3_HINT = 16   # probe bit 4: weight copies with L2::evict_first (box lab: -2..3 us per call)


def v3_call(st, data, vid, probe=V3_HINT, mask=7, hot_t=None, hot_w4=None, use_c=None):
    v = st["variants"][vid]
    if use_c is None:
        use_c = V3_SCALE5
    hot_t = (HOT_T if hot_t is None else hot_t) if (len(v) >= 4 and v[3] == 1) else 0
    hot_t += 1000 * (HOT_W4 if hot_w4 is None else hot_w4)
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"],
                   st["inter"], st["y"], st["meta"], st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"],
                   st["xw"], st["xw2"], st["cnt13"], st["ws13c"], st["ws2c"],
                   st["sfmt"] if (use_c and st["sfmt"] is not None) else st["nofmt"], hot_t, m,
                   data["config"]["local_start"], st["grid"][vid], st["sms"] * 2, probe)
    return out


def v3_reset(st):
    st["cnt13"].zero_()
    st["h"].zero_()
    st["cnt"].zero_()
    st["cnt2"].zero_()
    torch.cuda.synchronize()



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


# ----------------------------------------------------------------------------------------------
# prepare / kernel_fn
# ----------------------------------------------------------------------------------------------

def _v2_ok_for(cfg, m):
    return (cfg["local_count"] == E_LOC and cfg["local_start"] == 0 and cfg["d_hidden"] == D_HIDDEN
            and cfg["d_expert"] == D_EXPERT and cfg["top_k"] == TOP_K and 1 <= m <= M_MAX)


V3_SELF_TEST_CASES = ((1, 11, "normal"), (5, 12, "normal"), (16, 17, "normal"), (33, 13, "normal"),
                      (100, 14, "normal"), (128, 15, "skewed"), (257, 16, "normal"), (320, 18, "normal"))


def _v3_try(weights, dev):
    """Build v3 and self-test it per bucket against the FP32 reference (two calls per case: the second checks that
    the h / counter invariants were restored). Returns (state or None, {bucket: ok})."""
    ok = {rep: False for _, _, rep in BUCKETS}
    if not V3_ENABLE:
        return None, ok
    try:
        st = v3_prepare(weights, log=lambda s: print(s, flush=True))
    except Exception as ex:  # noqa: BLE001
        print("[k2a3] v3 build/repack FAILED. Tail:\n" + str(ex)[-8000:], flush=True)
        return None, ok
    for rep in ok:
        ok[rep] = True
    for bs, seed, dist in V3_SELF_TEST_CASES:
        rep = bucket_of(bs)
        try:
            d = synth_inputs(bs, seed, dev, dist)
            ref = ref_fp32(d, weights)
            for k in range(2):
                st["err"].zero_()
                got = v3_call(st, d, V3_PLAN[rep], hot_t=V3_HOT_T.get(rep))
                torch.cuda.synchronize()
                tot, row = rel_err(got, ref)
                errf = int(st["err"][0])
                good = tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(got.float()).all())
                print(f"[k2a3] self-test bs={bs} {dist} v{V3_PLAN[rep]} call {k}: rel {tot:.3e} worst row {row:.3e} "
                      f"err={errf} -> {'ok' if good else 'FAIL'}", flush=True)
                if not good:
                    ok[rep] = False
        except Exception as ex:  # noqa: BLE001
            print(f"[k2a3] self-test bs={bs} raised {ex!r}", flush=True)
            ok[rep] = False
        v3_reset(st)
    return st, ok


def prepare(weights: Weights):
    t0 = time.time()
    dev = weights["w13"].device
    state = {"inc": inc_prepare(weights), "v2": None, "route": dict(ROUTE), "v3": None, "route3": {}}
    print(f"[k2a2] incumbent (r1_glue_v3) prepared in {time.time() - t0:.1f}s", flush=True)
    if FALLBACK:
        print("[k2a2] FALLBACK=True: incumbent everywhere", flush=True)
        return state
    st3, ok3 = _v3_try(weights, dev)
    if st3 is not None and any(ok3.values()):
        state["v3"], state["route3"] = st3, ok3
    print(f"[k2a3] v3 buckets {ok3} ({time.time() - t0:.1f}s)", flush=True)
    if all(ok3.values()):
        return state   # v2.2 is only needed as a fallback for buckets v3 lost
    try:
        st = v2_prepare(weights, V2_VARIANTS, log=lambda s: print(s, flush=True))
    except Exception as ex:  # noqa: BLE001
        print("[k2a2] v2 build/repack FAILED, incumbent everywhere. Tail:\n" + str(ex)[-8000:], flush=True)
        return state
    ok = True
    if SELF_TEST:
        try:
            cases = ((1, 11, "normal"), (5, 12, "normal"), (33, 13, "normal"), (100, 14, "normal"),
                     (128, 15, "skewed"), (257, 16, "normal"))
            for bs, seed, dist in cases:
                d = synth_inputs(bs, seed, dev, dist)
                ref = ref_fp32(d, weights)
                rep = bucket_of(bs)
                vid, fuse, kst, asg, hv, ht = (tuple(V2_PLAN[rep]) + (-1, 0))[:6]
                for fz in sorted({fuse, 0, 2}):
                    got = v2_call(st, d, vid, fz, kst, asg, 0, 0, hv if fz == fuse else -1, ht if fz == fuse else 0)
                    torch.cuda.synchronize()
                    tot, row = rel_err(got, ref)
                    errf = int(st["err"][0])
                    print(f"[k2a2] self-test bs={bs} {dist} v{vid} fuse={fz}: rel {tot:.3e} worst row {row:.3e} "
                          f"err={errf}", flush=True)
                    if not (tot < 1.0e-2 and row < 5.0e-2) or errf or not bool(torch.isfinite(got.float()).all()):
                        raise RuntimeError(f"self-test mismatch at bs={bs} fuse={fz}")
        except Exception as ex:  # noqa: BLE001
            print(f"[k2a2] self-test FAILED, incumbent everywhere: {ex!r}", flush=True)
            ok = False
    v2_reset(st)
    if not ok:
        return state
    state["v2"] = st
    if AUTOTUNE:
        try:
            flush = torch.empty((64 * 1024 * 1024,), dtype=torch.float32, device=dev)   # 256 MiB > L2
            for lo, hi, rep in BUCKETS:
                d = synth_inputs(rep, 400 + rep, dev)
                vid, fuse, kst, asg, hv, ht = (tuple(V2_PLAN[rep]) + (-1, 0))[:6]
                ref = ref_fp32(d, weights)
                st["err"].zero_()
                t_inc, _ = time_graph(lambda dd: inc_kernel_fn(dd, state["inc"]), d, flush, AUTOTUNE_REPS)
                t_v2, o_v2 = time_graph(lambda dd: v2_call(st, dd, vid, fuse, kst, asg, 0, 0, hv, ht), d, flush,
                                        AUTOTUNE_REPS)
                torch.cuda.synchronize()
                errf = int(st["err"][0])            # grid-barrier timeout during the replays (co-residency)
                r_rep, _ = rel_err(o_v2, ref)       # output of the last replay, not just the eager call
                ok_rep = errf == 0 and r_rep < 1.0e-2
                pick = "v2" if (ok_rep and t_inc > AUTOTUNE_MARGIN * t_v2) else state["route"][rep]
                state["route"][rep] = pick
                print(f"[k2a2] autotune bs={rep}: incumbent {t_inc:.1f} us, v2 {t_v2:.1f} us (replayed rel "
                      f"{r_rep:.2e}, err {errf}) -> {pick}", flush=True)
                v2_reset(st)
            del flush
            v2_reset(st)
        except Exception as ex:  # noqa: BLE001
            print(f"[k2a2] autotune failed, static route table kept: {ex!r}", flush=True)
            state["route"] = dict(ROUTE)
            v2_reset(st)
    print(f"[k2a2] route table {state['route']} | prepare done in {time.time() - t0:.1f}s", flush=True)
    return state


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    m = data["hidden_states"].shape[0]
    rep = bucket_of(m)
    st3 = state.get("v3")
    if st3 is not None and rep is not None and state["route3"].get(rep) and _v2_ok_for(cfg, m) and m <= M3:
        x = data["hidden_states"]
        if (not x.is_contiguous()) or x.data_ptr() % 16 != 0:   # bulk copies need 16 B-aligned rows
            data = dict(data)
            data["hidden_states"] = x.contiguous().clone()
        return v3_call(st3, data, V3_PLAN[rep], hot_t=V3_HOT_T.get(rep))
    st = state["v2"]
    if st is not None and rep is not None and state["route"].get(rep) == "v2" and _v2_ok_for(cfg, m):
        vid, fuse, kst, asg, hv, ht = (tuple(V2_PLAN[rep]) + (-1, 0))[:6]
        return v2_call(st, data, vid, fuse, kst, asg, 0, 0, hv, ht)
    return inc_kernel_fn(data, state["inc"])

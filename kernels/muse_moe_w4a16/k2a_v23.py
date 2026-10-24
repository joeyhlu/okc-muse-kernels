# okc: problem=muse_moe_w4a16
# nonce k2a-v22-1
"""K2a v2.2: custom NVFP4 x BF16 MoE experts for decode/verify (results: k2a_v22.md; design: k2a_v2_design.md).
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
W_SHIFT = 256.0         # dequantized weights are w_true * 2^-8; the FP32 global scale carries the 2^8

# Variant table (template parameters of k2a2::Cfg<NC, U, PF, TW, XE, MINB>):
#   NC   8-token chunks per warp per pass (accumulators 4*NC regs)
#   U    16 B weight loads per thread per group (U items of 512 B per warp)
#   PF   0 = issue a group's loads, then consume (stream_lab D winner); 1 = prefetch the next group first
#   TW   token-group warps per row tile (they share one weight fetch through L1); RW = 8 / TW row tiles per unit
#   XE   bf16 elements of the staged X slice (k per slice = XE / (TW * 8 * NC))
#   MINB __launch_bounds__ min CTAs per SM (4 -> <= 64 regs, 3 -> <= 80)
VKEYS = ("nc", "u", "pf", "tw", "xe", "minb", "wg")


def variant(nc, u=4, pf=0, tw=1, xe=8192, minb=4, wg=0):
    return {"nc": nc, "u": u, "pf": pf, "tw": tw, "xe": xe, "minb": minb, "wg": wg}


V2_VARIANTS = [
    variant(1, 4, 0, 1, 8192, 4),         # 0: 8 tokens / pass, 4 CTAs/SM (<= 64 regs), synchronous X staging
    variant(2, 4, 0, 1, 16384, 4),        # 1: 16 tokens / pass, cp.async X staging
    variant(4, 4, 0, 1, 16384, 3, 1),     # 2: WG 32 tokens / pass, 3 CTAs/SM
    variant(8, 4, 0, 1, 16384, 2, 1),     # 3: WG 64 tokens / pass, 2 CTAs/SM
    variant(8, 2, 0, 1, 16384, 3, 1),     # 4: WG 64 tokens / pass, U 2, 3 CTAs/SM
    variant(4, 2, 0, 1, 16384, 4, 1),     # 5: WG 32 tokens / pass, U 2, 4 CTAs/SM
    variant(8, 4, 0, 1, 16384, 3, 1),     # 6: WG 64 tokens / pass, U 4, 3 CTAs/SM
    variant(2, 4, 0, 1, 16384, 4, 1),     # 7: WG 16 tokens / pass, 4 CTAs/SM
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

template <int NC_, int U_, int PF_, int TW_, int XE_, int MINB_, int WG_ = 0>
struct Cfg {
  static constexpr int NC = NC_, U = U_, PF = PF_, TW = TW_, XE = XE_, MINB = MINB_, WG = WG_;
  static constexpr int TPW = 8 * NC;           // tokens per warp per pass
  static constexpr int TPP = TW * TPW;         // tokens per pass
  static constexpr int RW = 8 / TW;            // row tiles per CTA unit
  static constexpr int KSW = XE / TPP;         // k per staged X slice
  static constexpr int XSTR = KSW + 8;         // padded smem row stride (bf16): 16 B pad -> conflict-free LDS.128
  // WG: X slice in wgmma's no-swizzle K-major core-matrix layout (8 tokens x 16 B per 128 B core matrix), no pad.
  static constexpr int XS = WG ? XE : TPP * XSTR;
  static_assert(TW == 1 || TW == 2 || TW == 4 || TW == 8, "TW");
  static_assert(U == 2 || U == 4 || U == 8, "U");
  static_assert(KSW >= 64 * U && KSW % (64 * U) == 0, "a slice holds whole load groups");
  static_assert(!WG || (TW == 1 && PF == 0 && (NC == 2 || NC == 4 || NC == 8)), "WG: one token group, PF 0, N <= 64");
};

// ------------------------------------------------------------------------------------------------
// v2.3 WG path: wgmma.mma_async m64nNk16, A = the dequantized weight fragments in registers (RS form; per warp the
// m16k16 layout of mma.m16n8k16, i.e. exactly consume_item's (ag, a8) wiring), B = the staged X slice in shared
// memory read through a no-swizzle K-major matrix descriptor (LBO = 128 B between the two k8 core matrices of a
// k16 step, SBO = 16 * kw B between 8-token row groups). A warpgroup (4 warps = 4 row tiles = 64 rows) shares one B
// read per step, so tokens per pass go to 32/64 without mma.sync's per-warp B re-reads from shared memory.
// Box unit test (tools/box/wgmma_unit.py): exact for N 8/16/32/64 in natural and v2 item order (2026-10-24).
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint64_t wg_desc(const void* smem, uint32_t sbo) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

#define K2A_WG_HEAD "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned."
template <int N> __device__ __forceinline__ void wgmma_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                          uint64_t db);
template <> __device__ __forceinline__ void wgmma_rs<8>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                        uint64_t db) {
  asm volatile(K2A_WG_HEAD "m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
template <> __device__ __forceinline__ void wgmma_rs<16>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                         uint64_t db) {
  asm volatile(K2A_WG_HEAD "m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
template <> __device__ __forceinline__ void wgmma_rs<32>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                         uint64_t db) {
  asm volatile(K2A_WG_HEAD "m64n32k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
template <> __device__ __forceinline__ void wgmma_rs<64>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                         uint64_t db) {
  asm volatile(K2A_WG_HEAD "m64n64k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
#undef K2A_WG_HEAD

// N = 8 * ncase columns (ncase in {1, 2, 4, 8}, warp-uniform; columns >= ntok are stale smem rows -> discarded outputs).
template <int NC>
__device__ __forceinline__ void wgmma_n(int ncase, float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint64_t db) {
  if constexpr (NC >= 8) { if (ncase > 4) { wgmma_rs<64>(d, a0, a1, a2, a3, db); return; } }
  if constexpr (NC >= 4) { if (ncase > 2) { wgmma_rs<32>(d, a0, a1, a2, a3, db); return; } }
  if (ncase > 1) { wgmma_rs<16>(d, a0, a1, a2, a3, db); return; }
  wgmma_rs<8>(d, a0, a1, a2, a3, db);
}

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
  if constexpr (C::WG) {
    // WG layout. Item idx -> t = idx & 3, nr = (idx >> 2) & 7, (n8, i) = idx >> 5 (i fastest, NI = kw / 64 items):
    // lane (nr, t) owns row n = 8 n8 + nr, k = k0 + 64 i + 16 t + 0..15 (32 B). Its u32 q (bf16 pair 16t + 2q) is
    // logical k-slot 2t (+8 if q odd) of k16 step 4i + q/2 (consume_item's item order), i.e. core matrix
    // kc = 8 i + q of row group n8: u32 n8 * 4 kw + 32 (8 i + q) + 4 nr + t. Bank = 4 nr + t: conflict-free.
    const int lni = __ffs(kw >> 6) - 1;          // log2(NI)
    const int tot = ((ntok + 7) >> 3) << (lni + 5);
    uint32_t* xs32 = reinterpret_cast<uint32_t*>(xs);
    for (int idx = threadIdx.x; idx < tot; idx += 256) {
      const int t = idx & 3, nr = (idx >> 2) & 7, rest = idx >> 5;
      const int n8 = rest >> lni, i = rest & ((1 << lni) - 1);
      const int n = 8 * n8 + nr;
      if (n < ntok) {
        const int k = k0 + 64 * i + 16 * t;
        uint32_t v[8];
        if (MODE == 1 || !from_h) {
          uint4 a, b;
          if constexpr (MODE == 1) {
            const uint4* src = reinterpret_cast<const uint4*>(P.x + (size_t)(((int)s_list[c0 + n]) >> 3) * KDIM + k);
            a = __ldg(src); b = __ldg(src + 1);
          } else {
            const uint4* src = reinterpret_cast<const uint4*>(P.inter + (size_t)(c0 + n) * KDIM + k);
            a = __ldcg(src); b = __ldcg(src + 1);
          }
          v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
          const float* hr = P.h + (size_t)(c0 + n) * (2 * KDIM) + k;
#pragma unroll
          for (int hq = 0; hq < 4; ++hq) {
            const float4 g4 = __ldcg(reinterpret_cast<const float4*>(hr) + hq);
            const float4 u4 = __ldcg(reinterpret_cast<const float4*>(hr + KDIM) + hq);
            v[2 * hq] = pack_bf2(swiglu(g4.x, u4.x), swiglu(g4.y, u4.y));
            v[2 * hq + 1] = pack_bf2(swiglu(g4.z, u4.z), swiglu(g4.w, u4.w));
          }
        }
        uint32_t* dst = xs32 + n8 * (4 * kw) + 256 * i + 4 * nr + t;
#pragma unroll
        for (int q = 0; q < 8; ++q) dst[32 * q] = v[q];
      }
    }
    fence_async_smem();   // generic-proxy smem writes -> visible to wgmma (async proxy) after the caller's barrier
    return;
  }
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

// WG item: the same 16 B / scale word as consume_item, dequantized in two halves (k16 steps 0-1 from words x, z;
// steps 2-3 from y, w) so one half's wgmmas run while the other half is dequantized. Before a half's registers are
// rewritten, wait_group 1 retires the group that read them two commits ago. Step s reads core matrices
// kc = 2 (4 item + s) of the slice: descriptor start + (8 item + 2 s) * 128 B (desc >> 4 units: + 8 (8 item + 2 s)).
template <class C>
__device__ __forceinline__ void consume_item_wg(const uint4 w, const uint32_t sw2, int par, uint64_t desc0, int item,
                                                int ncase, float* acc, int probe, uint32_t& sink) {
  if (probe == 1) { sink ^= w.x ^ w.y ^ w.z ^ w.w ^ sw2; return; }
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;   // row g
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;   // row g + 8
  const uint64_t d = desc0 + (uint64_t)(64 * item);
  uint32_t h0[8], h1[8];
  wg_wait<1>();
  deq4(w.x, h0); deq4(w.z, h0 + 4);
#pragma unroll
  for (int j = 0; j < 4; ++j) { h0[j] = bmul2(h0[j], Sg); h0[4 + j] = bmul2(h0[4 + j], S8); }
  if (probe == 2) {
    uint32_t z = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) z ^= h0[j];
    sink ^= z;
    return;
  }
  wg_fence();
  wgmma_n<C::NC>(ncase, acc, h0[0], h0[4], h0[1], h0[5], d);         // step 0: (ag0, a8_0, ag1, a8_1)
  wgmma_n<C::NC>(ncase, acc, h0[2], h0[6], h0[3], h0[7], d + 16);    // step 1: (ag2, a8_2, ag3, a8_3)
  wg_commit();
  wg_wait<1>();
  deq4(w.y, h1); deq4(w.w, h1 + 4);
#pragma unroll
  for (int j = 0; j < 4; ++j) { h1[j] = bmul2(h1[j], Sg); h1[4 + j] = bmul2(h1[4 + j], S8); }
  wg_fence();
  wgmma_n<C::NC>(ncase, acc, h1[0], h1[4], h1[1], h1[5], d + 32);    // step 2: (ag4, a8_4, ag5, a8_5)
  wgmma_n<C::NC>(ncase, acc, h1[2], h1[6], h1[3], h1[7], d + 48);    // step 3: (ag6, a8_6, ag7, a8_7)
  wg_commit();
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
  uint64_t desc0 = 0;
  if constexpr (C::WG) desc0 = wg_desc(xs, 16u * (uint32_t)KSWe);
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
    const int ncase = (nch <= 1) ? 1 : (nch <= 2) ? 2 : (nch <= 4) ? 4 : 8;   // WG: N = 8 * ncase
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
          } else if constexpr (C::WG) {
#pragma unroll
            for (int i = 0; i < U; ++i)
              consume_item_wg<C>(wa[i], sa[i >> 1], i & 1, desc0, gi * U + i, ncase, &acc[0][0], P.probe, sink);
            if (gi + 1 < NGRP) load_group<C>(wt, st, kb0 + (gi + 1) * U, wa, sa);
          } else {
#pragma unroll
            for (int i = 0; i < U; ++i)
              consume_item<C>(wa[i], sa[i >> 1], i & 1, xrow, (gi * U + i) * 64, nch, t, acc, P.probe, sink);
            if (gi + 1 < NGRP) load_group<C>(wt, st, kb0 + (gi + 1) * U, wa, sa);
          }
        }
        if constexpr (C::WG) wg_wait<0>();   // xs is restaged / acc flushed next: retire every wgmma of the slice
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
  __shared__ __align__(128) __nv_bfloat16 xs[C::XS];
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
    return "k2a2::Cfg<%d, %d, %d, %d, %d, %d, %d>" % tuple(v.get(k, 0) for k in VKEYS)


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
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                           "-gencode=arch=compute_90a,code=sm_90a"],   # wgmma is sm_90a-only
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


def prepare(weights: Weights):
    t0 = time.time()
    dev = weights["w13"].device
    state = {"inc": inc_prepare(weights), "v2": None, "route": dict(ROUTE)}
    print(f"[k2a2] incumbent (r1_glue_v3) prepared in {time.time() - t0:.1f}s", flush=True)
    if FALLBACK:
        print("[k2a2] FALLBACK=True: incumbent everywhere", flush=True)
        return state
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
    st = state["v2"]
    rep = bucket_of(m)
    if st is not None and rep is not None and state["route"].get(rep) == "v2" and _v2_ok_for(cfg, m):
        vid, fuse, kst, asg, hv, ht = (tuple(V2_PLAN[rep]) + (-1, 0))[:6]
        return v2_call(st, data, vid, fuse, kst, asg, 0, 0, hv, ht)
    return inc_kernel_fn(data, state["inc"])

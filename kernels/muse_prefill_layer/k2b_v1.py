# okc: problem=muse_prefill_layer
# nonce k2b-v1-1
"""K2b v1: SGLang's prefill decoder layer (baseline.py, verbatim) with the routed experts' two Marlin W4A16
GEMMs replaced by a warp-specialised CUDA grouped GEMM on sm_90a wgmma (design: k2b_v1_design.md).

Experts path (per call; static shapes, no host syncs, CUDA-graph capturable):
  1. moe_align_block_size(local_ids, BM, 32, ignore_invalid_expert=True) (SGLang's CUDA kernel, contract checked
     in prepare(); else the baseline's verbatim call, whose -1 blocks the kernel skips).
  2. k2b1 MODE 1: inter[sorted row, 4096] = SiLU(gs13 * x W1^T) * (gs13 * x W3^T)   (BF16, sorted order)
  3. k2b1 MODE 2: y[pair, 4096] = BF16(gs2 * inter W2^T)                            (BF16, pair order)
  4. fused_post_expert_norm_weighted_sum (SGLang's Triton kernel, as the baseline).

Kernel (one source, MODE/BM/BN/S/NB/LAYOUT/REG template parameters): persistent CTAs (grid = #SMs) walk the
tile list (m-block of BM sorted rows, n-tile of BN weight rows), n-tile fastest. 384 threads:
  * producer warpgroup (WG0): cp.async 16 B streams of the gathered activation rows, the packed NVFP4 codes and
    the E4M3 scales into an S-stage smem ring (completion on mbarriers via cp.async.mbarrier.arrive.noinc), then
    dequantises each stage ONCE into a BF16 B tile (BN weight rows x 64 k, 128B-swizzled K-major, or the
    no-swizzle core-matrix layout) in an NB-deep buffer ring; fence.proxy.async; mbarrier arrive.
  * two consumer warpgroups: SS wgmma m64nNk16 (A = activations, B = dequantised weights, both from smem),
    FP32 accumulators in registers, 4 wgmma per 64-k stage, one group in flight while the next stage waits;
    epilogue applies the FP32 global scale, SwiGLU in registers (gate/up rows of K2A's W13 tiles land in the
    same thread), BF16 stores.
Every mbarrier wait is bounded (clock64 budget -> err flag, later waits bail out): no hang (rule R9).

WEIGHT LAYOUT = K2A v2.2's (kernels/muse_moe_w4a16/k2a_v22.py), copied verbatim below and checked bit-exactly
by k2b_v1_emulate.py: one layout per layer serves decode (K2A) and prefill (K2b).
  WQ[e][tile16][kb][lane][4 x u32]: word x/y = row g at k = 64kb+16t+0..7 / 8..15, z/w = row g+8 (lane = 4g+t),
     8 E2M1 codes per word at BITPOS (deq4 -> 4 bf16x2 = 8 consecutive k = one 16 B chunk of the B tile).
  SQ[e][tile16][kb/2][lane] u32: bytes (g, kb even), (g+8, kb even), (g, kb odd), (g+8, kb odd) of group 4kb+t,
     re-encoded so that bf16 bits = b*16 + 0x7580 = sf * 2^118 (zero/negative scales folded into the codes).
  W13 tile r = gate rows 8r..8r+7 then up rows 8r..8r+7.
Numerics: E2M1 x E4M3 exact in BF16 (as K2A/Marlin), FP32 accumulate, gs*2^8 in FP32 on the accumulator,
h never rounded, SiLU*up rounded once to BF16 (inter), y BF16, finalize as SGLang.
prepare(): repack, build (load_inline, sm_90a), self-test every config against an FP32 reference, time the passing
ones against the stock Marlin path under CUDA graphs with a cold L2, keep the faster per token-count class
(the stock path stays wherever nothing beats it by K2B_MIN_GAIN, or anything fails).
"""

from typing import TypedDict

import hashlib
import time
import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


# ===== K2B PURE BEGIN =====  (torch + strings only; k2b_v1_emulate.py execs this section on CPU)
K2B_ENABLE = True              # False: the stock Marlin experts everywhere
K2B_LAB = False                # True (k2b_v1_lab.py): verbose ptxas, more configs, probe timings; check runs only
K2B_DEADLINE_S = 200.0         # since prepare() started: no new candidate is built/tested after this
K2B_TIMING_DEADLINE_S = 265.0  # since prepare() started: no further candidate is timed after this (limit 300)
K2B_MIN_GAIN = 0.97            # a candidate replaces Marlin for a class only if t_cand < K2B_MIN_GAIN * t_marlin
K2B_TIMING_REPS = 5
K2B_RUNTIME_FLAGS = 8          # OR'ed into every launch: 8 = no consumer-side fence.proxy.async (as CUTLASS; passed
                               # every box self-test), 16 = test_wait polling (slower)
K2B_PROBES = (1, 2, 3, 7) if K2B_LAB else ()   # in-situ stage probes at T = 16384 (lab only)

K2B_E = 32                     # local experts
K2B_K = 4096                   # K of both GEMMs
K2B_N13 = 8192                 # W13 rows (gate 4096 + up 4096, K2A tile-interleaved)
K2B_N2 = 4096                  # W2 rows
K2B_TOPK = 8

# name -> (BM, BN, S, NB, LAYOUT, REG)
#   BM     sorted rows (tokens) per tile = moe_align block; 128: the 2 consumer WGs split M (64 rows each, wgmma
#          N = BN); 64: they split N (wgmma N = BN/2)
#   BN     weight rows per tile (wgmma N side); S smem ring stages (64 k each); NB dequantised B buffers
#   LAYOUT 0: K-major 128B swizzle (CUTLASS Layout_K_SW128), 1: K-major no-swizzle core matrices (the layout
#          tools/box/wgmma_unit.py validated on the H200; fallback)
#   REG    0: no setmaxnreg; 1: producer 88 / consumers 208 registers
K2B_CONFIGS = {                # scored file: the box lab's winners (2026-10-24, k2b_v1_bench5: T=753/3012/16384
                               # 627/1584/7006 us vs Marlin 740/2275/11399 us, same process and routing)
    "sw128x256s5b3r": (128, 256, 5, 3, 0, 1),
    "sw128x256s4b3r": (128, 256, 4, 3, 0, 1),
}
K2B_LAB_CONFIGS = {            # k2b_v1_lab.py adds these (tile/stage/register/layout sweep)
    "sw128x256s4": (128, 256, 4, 2, 0, 0),
    "il128x256s4": (128, 256, 4, 2, 1, 0),
    "sw64x256s5": (64, 256, 5, 2, 0, 0),
    "sw128x128s6": (128, 128, 6, 3, 0, 0),
    "sw128x256s4r": (128, 256, 4, 2, 0, 1),
    "sw128x256s4b3": (128, 256, 4, 3, 0, 0),
}
# token-count classes: (name, max tokens, representative tokens for timing)
K2B_CLASSES = (("small", 1024, 753), ("mid", 6144, 3012), ("large", 1 << 30, 16384))

_K2B_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def k2b_class(tokens):
    for name, hi, _ in K2B_CLASSES:
        if tokens <= hi:
            return name
    return K2B_CLASSES[-1][0]


def k2b_all_configs(lab):
    cfgs = dict(K2B_CONFIGS)
    if lab:
        cfgs.update(K2B_LAB_CONFIGS)
    return cfgs


def k2b_smem_bytes(cfg):
    """Dynamic smem of one config (must equal Cfg::SMEM in the CUDA source; the emulator checks both)."""
    bm, bn, s, nb, _, _ = cfg
    return s * bm * 128 + nb * bn * 128 + s * bn * 32 + s * bn * 8 + (2 * s + 2 * nb) * 8


# ---- K2A v2.2 repack, VERBATIM from kernels/muse_moe_w4a16/k2a_v22.py (one layout per layer; the emulator
#      asserts bit-identical output against that file) ----
BITPOS = ((6, 7, 8, 15), (22, 23, 24, 31), (9, 10, 11, 18), (25, 26, 27, 2),
          (12, 13, 14, 21), (28, 29, 30, 5), (0, 1, 3, 4), (16, 17, 19, 20))
W_SHIFT = 256.0         # dequantized weights are w_true * 2^-8; the FP32 global scale carries the 2^8


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
# ---- end of the K2A v2.2 copy ----


def k2b_dequant(w, sf, gs):
    """NVFP4 -> float32 (reference semantics, as nvfp4.dequantize). w uint8 [..., K/2], sf E4M3 [..., K/16]."""
    grid = torch.tensor(_K2B_E2M1, dtype=torch.float32, device=w.device)
    p = w.view(torch.uint8)
    codes = torch.stack((p & 15, p >> 4), dim=-1).flatten(-2).long()
    mag = grid[codes & 7]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)
    return vals.view(*vals.shape[:-2], k)


def k2b_synth_routing(tokens, seed, device, hot=16, local_share=0.25, zipf=1.0):
    """Busiest-GPU-like routing (as k2b_v0: 1.82-1.84 local routes/token on 16 touched experts, top expert
    ~0.47 routes/token; the judge's busiest GPU: 1.75-1.86 routes/token, ~15-17 experts). ids int32 [T, 8]
    (local 0..31, others 32..255, distinct per token), weights float32 [T, 8]."""
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.ones(256, dtype=torch.float32, device=device)
    z = 1.0 / torch.arange(1, hot + 1, dtype=torch.float32, device=device) ** zipf
    w[:hot] = z / z.sum() * (local_share / (1.0 - local_share)) * 224.0
    w[hot:32] = 0.0
    u = torch.rand((tokens, 256), generator=g, device=device).clamp(1e-9, 1.0 - 1e-7)
    logits = w.log()[None, :] - torch.log(-torch.log(u))
    ids = logits.topk(K2B_TOPK, dim=-1).indices.to(torch.int32)
    s = torch.rand((tokens, K2B_TOPK), generator=g, device=device) + 0.05
    return ids, (s / s.sum(-1, keepdim=True)).float()


def k2b_ref_experts(x, ids, wts, weights, post_w, n_local=K2B_E):
    """FP32 semantics of reference._experts: x [T, K], ids int [T, 8] (local experts 0..n_local-1, others
    >= n_local), wts [T, 8]; returns float32 [T, N2]."""
    xf = x.float()
    n2 = weights["w2"].shape[1]
    kh = weights["w13"].shape[1] // 2
    out = torch.zeros((xf.shape[0], n2), dtype=torch.float32, device=x.device)
    nw = 1.0 + post_w.float()
    ids = ids.long()
    for e in torch.unique(ids).tolist():
        if not 0 <= e < n_local:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = xf[tok] @ k2b_dequant(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :kh]) * h[:, kh:]
        y = inter @ k2b_dequant(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        y = y * nw
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * wts[tok, slot, None].float())
    return out


def k2b_rel(a, b):
    a, b = a.float(), b.float()
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row


def k2b_wgmma_src(n):
    """CUDA wrapper for wgmma.mma_async m64n{n}k16 f32 += bf16 x bf16, A and B from smem descriptors (both
    K-major: tnspA = tnspB = 0), scale-d predicate from an int (0 on the first k16 of a tile: D = A*B)."""
    nr = n // 2
    regs = ", ".join("%%%d" % i for i in range(nr))
    outs = ", ".join('"+f"(d[%d])' % i for i in range(nr))
    return ("template <> __device__ __forceinline__ void wgmma_ss<%d>(float* d, uint64_t da, uint64_t db, int sd) {\n"
            "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
            "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n\"\n"
            "    : %s\n    : \"l\"(da), \"l\"(db), \"r\"(sd));\n}\n"
            % (n, nr + 2, n, regs, nr, nr + 1, outs))


K2B_KERNEL_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ---- index maps shared with k2b_v1_emulate.py: each body is also a valid Python expression (non-negative ints;
//      only + * << >> & ^ | and parentheses), and the emulator evaluates exactly these strings ----
// smem byte offset of (row r, 16 B chunk c) inside an operand tile of R rows x 64 k (BF16, K-major, 8-row atoms
// of 1024 B). SW: 128B swizzle (chunk ^ row%8). IL: no swizzle, core matrix = 8 rows x 16 B, chunk stride 128 B.
#define K2B_ADDR_SW(r, c) ((((r) >> 3) * 1024) + (((r) & 7) * 128) + (((c) ^ ((r) & 7)) << 4))
#define K2B_ADDR_IL(r, c) ((((r) >> 3) * 1024) + ((c) * 128) + (((r) & 7) * 16))
// descriptor fields per layout: k16 advance (bytes), LBO, SBO (bytes), layout type (bits 62-63)
#define K2B_KADV_SW 32
#define K2B_KADV_IL 256
#define K2B_LBO_SW 16
#define K2B_LBO_IL 128
#define K2B_SBO 1024
#define K2B_LT_SW 1
#define K2B_LT_IL 0
// dequant item idx (one 16 B K2A lane item) -> K2A tile in the n-tile, lane; word wd (0..3 = x y z w) -> B row, chunk
#define K2B_ITEM_TILE(idx) ((idx) >> 5)
#define K2B_ITEM_LANE(idx) ((idx) & 31)
#define K2B_WROW(i, lane, wd) (((i) * 16) + ((lane) >> 2) + (((wd) >> 1) * 8))
#define K2B_WCHUNK(lane, wd) ((((lane) & 3) * 2) + ((wd) & 1))
// __byte_perm selector of the scale byte (kb parity par, row half hi) duplicated into both bf16 lanes
#define K2B_SSEL(par, hi) (0x4040 + (0x0101 * (hi)) + (0x0202 * (par)))
// global K2A indices: code item (uint4 units) and scale word (u32 units); rt = 16-row tiles per expert,
// nkb = 64-k blocks per row (64 for K = 4096)
#define K2B_WQ_IDX(e, rt, tile, kb, lane, nkb) ((((((e) * (rt)) + (tile)) * (nkb) + (kb)) * 32) + (lane))
#define K2B_SQ_IDX(e, rt, tile, kb, lane, nkb) ((((((e) * (rt)) + (tile)) * ((nkb) >> 1) + ((kb) >> 1)) * 32) + (lane))
// producer: thread p copies 16 B chunk K2B_A_CHUNK(p) of activation rows K2B_A_ROW(p, i), i < BM/16: 8 consecutive
// threads cover one 128 B row (coalesced, full sectors), a warp instruction covers 4 rows
#define K2B_A_ROW(p, i) (((p) >> 3) + ((i) * 16))
#define K2B_A_CHUNK(p) ((p) & 7)
// tile t -> (m-block, n-tile), n-tile fastest (ntl = log2 n-tiles)
#define K2B_TILE_MB(t, ntl) ((t) >> (ntl))
#define K2B_TILE_NT(t, ntl) ((t) & ((1 << (ntl)) - 1))
// wgmma m64nNk16 f32 accumulator: register v of thread (warp w of the WG, lane) -> (row, col) of the 64 x N tile
#define K2B_ACC_ROW(w, lane, v) (((w) * 16) + ((lane) >> 2) + ((((v) >> 1) & 1) * 8))
#define K2B_ACC_COL(v, lane) (((((v) >> 2)) * 8) + (((lane) & 3) * 2) + ((v) & 1))
// SwiGLU pairing (K2A W13 tile i of the WG's N range: cols 16i..16i+7 gate, 16i+8..16i+15 up): register of the
// gate / up value for row half h (0: row g, 1: row g+8), element e2; intermediate column (tb = first K2A tile)
#define K2B_GATE_REG(i, h, e2) (((i) * 8) + ((h) * 2) + (e2))
#define K2B_UP_REG(i, h, e2) (((i) * 8) + 4 + ((h) * 2) + (e2))
#define K2B_INTER_COL(tb, i, lane, e2) (((((tb) + (i)) * 8)) + (((lane) & 3) * 2) + (e2))

namespace k2b1 {
constexpr int KDIM = 4096;
constexpr int KS = KDIM / 64;            // 64-k stages per tile
constexpr int TOPK = 8;
constexpr int NE = 32;
constexpr long long SPIN_BUDGET = 1ll << 30;   // clock64 cycles (~0.5 s) for the first timed-out wait

template <int MODE_, int BM_, int BN_, int S_, int NB_, int LAYOUT_, int REG_>
struct Cfg {
  static constexpr int MODE = MODE_, BM = BM_, BN = BN_, S = S_, NB = NB_, LAYOUT = LAYOUT_, REG = REG_;
  static constexpr int N = MODE == 1 ? 8192 : 4096;            // weight rows (W13 / W2)
  static constexpr int RT = N / 16;                              // K2A 16-row tiles per expert
  static constexpr int NT = N / BN;                              // n-tiles
  static constexpr int NTL = NT == 16 ? 4 : NT == 32 ? 5 : NT == 64 ? 6 : -1;
  static constexpr int WM = BM / 64, WN = 2 / WM, NW = BN / WN;  // consumer WG grid; wgmma N per WG
  static constexpr int A_BYTES = BM * 128, B_BYTES = BN * 128, C_BYTES = BN * 32, SC_BYTES = BN * 8;
  static constexpr int OFF_A = 0;
  static constexpr int OFF_B = OFF_A + S * A_BYTES;
  static constexpr int OFF_C = OFF_B + NB * B_BYTES;
  static constexpr int OFF_SC = OFF_C + S * C_BYTES;
  static constexpr int OFF_BAR = OFF_SC + S * SC_BYTES;
  static constexpr int SMEM = OFF_BAR + (2 * S + 2 * NB) * 8;   // dynamic smem base is 1024-aligned (no static smem;
                                                                // checked at run time -> err 7)
  static constexpr int D = S - NB;   // load lookahead: the slot load(j + D) refills was freed by wgmma(j - NB), the same
                                     // event that frees the B buffer dequant(j) writes -> producer runs NB-1 stages ahead
  static constexpr int RPT = BM / 16;                            // producer: activation rows per thread
  static constexpr int ITEMS = BN * 2 / 128;                     // dequant items (and code chunks) per thread
  static constexpr int SCH = BN / 2;                             // 16 B scale chunks per stage
  static constexpr int KADV = LAYOUT == 0 ? K2B_KADV_SW : K2B_KADV_IL;
  static constexpr int LBO = LAYOUT == 0 ? K2B_LBO_SW : K2B_LBO_IL;
  static constexpr int LT = LAYOUT == 0 ? K2B_LT_SW : K2B_LT_IL;
  static_assert(BM == 64 || BM == 128, "BM");
  static_assert(BN == 128 || BN == 256, "BN");
  static_assert(NTL > 0, "NT must be 16, 32 or 64");
  static_assert(S >= 3 && NB >= 2 && S - NB >= 1, "pipeline depth");
  static_assert(SMEM <= 232448, "smem");
  static_assert(SCH <= 128, "scale chunks");
};

struct Params {
  const __nv_bfloat16* x;    // MODE 1: activations [T, 4096] (gathered by pair >> 3); MODE 2: inter [n_sorted, 4096]
  const uint4* wq;           // K2A codes
  const uint32_t* sq;        // K2A scale words
  const float* gs;           // [32] global scale * 2^8
  __nv_bfloat16* out;        // MODE 1: inter [n_sorted, 4096]; MODE 2: y [n_pairs, 4096]
  const int* sorted;         // moe_align sorted pair ids [n_sorted] (padding = n_pairs)
  const int* bexp;           // moe_align block experts [n_blocks]
  const int* ntp;            // [1] tokens post padding
  int* err;                  // [4]: err[0] = first timed-out wait's code
  int n_pairs, n_sorted, n_blocks, probe;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void cpasync_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.b32 %0, 1, 0, P1;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
// Bounded wait (R9): after SPIN_BUDGET cycles the first waiter records `code` in err[0] and returns; every later
// wait returns as soon as it sees err[0] != 0, so a broken pipeline ends the kernel quickly with garbage + flag.
// The common path is a tight try_wait loop (try_wait suspends in hardware until the phase completes or a short
// time limit); clock64 is read every 4096 polls and err (a global load) only then, so the slow path never delays
// noticing a completed phase.
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n.reg .pred P1;\nmbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\nselp.b32 %0, 1, 0, P1;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
// poll: 0 = try_wait (may suspend), 1 = test_wait (pure polling; runtime flag 16)
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity, int* err, int code, int poll = 0) {
  if (poll ? mbar_test(bar, parity) : mbar_try(bar, parity)) return;
  const long long t0 = clock64();
#pragma unroll 1
  for (uint32_t n = 1;; ++n) {
    if (poll ? mbar_test(bar, parity) : mbar_try(bar, parity)) return;
    if ((n & 4095u) == 0u) {
      if (*(volatile int*)err != 0) return;
      if (clock64() - t0 > SPIN_BUDGET) { atomicCAS(err, 0, code); return; }
    }
  }
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void producer_bar() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }   // WG0 only
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }
// K2A v2.2 deq4 (verbatim): 8 codes at BITPOS -> 4 bf16x2 = e2m1 * 2^-126 (k pairs 0-1, 2-3, 4-5, 6-7)
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
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr, int lbo, int sbo, int lt) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) |
         ((uint64_t)lt << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int NR>
__device__ __forceinline__ void fence_acc(float* d) {
#pragma unroll
  for (int i = 0; i < NR; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db, int sd);
@WGMMA@

template <class C>
__device__ __forceinline__ int next_valid(int t, int n_tiles, const int* bexp) {
  while (t < n_tiles) {
    const int e = __ldg(bexp + K2B_TILE_MB(t, C::NTL));
    if (e >= 0 && e < NE) break;
    t += gridDim.x;
  }
  return t;
}
__device__ __forceinline__ bool pair_ok(const Params& P, int srow) {
  const int sid = srow < P.n_sorted ? __ldg(P.sorted + srow) : -1;
  return sid >= 0 && sid < P.n_pairs;
}

// ---------------------------------------------------------------- producer warpgroup (WG0, thread p = 0..127)
template <class C>
struct LoadCur {
  int t, kb, e, nt;
  const __nv_bfloat16* src[C::RPT];   // this thread's activation rows (k = 0, its chunk)
  uint32_t okmask;                    // bit i: row i is a real pair (else zero-fill, src-size 0)
};

template <class C>
__device__ __forceinline__ void load_tile_setup(const Params& P, LoadCur<C>& L, int p) {
  const int mb = K2B_TILE_MB(L.t, C::NTL);
  L.nt = K2B_TILE_NT(L.t, C::NTL);
  L.e = __ldg(P.bexp + mb);
  L.kb = 0;
  L.okmask = 0;
#pragma unroll
  for (int i = 0; i < C::RPT; ++i) {
    const int srow = mb * C::BM + K2B_A_ROW(p, i);
    const int sid = srow < P.n_sorted ? __ldg(P.sorted + srow) : -1;
    const bool ok = sid >= 0 && sid < P.n_pairs;
    long long row = C::MODE == 1 ? (long long)(sid >> 3) : (long long)srow;
    if (!ok) row = 0;
    L.src[i] = P.x + row * KDIM + K2B_A_CHUNK(p) * 8;
    L.okmask |= (ok ? 1u : 0u) << i;
  }
}

template <class C>
__device__ __forceinline__ void producer_load(const Params& P, uint32_t sbase, uint32_t j, const LoadCur<C>& L, int p) {
  const int slot = j % C::S;
  const uint32_t bar_empty = sbase + C::OFF_BAR + 8 * (C::S + slot);
  mbar_wait(bar_empty, ((j / C::S) & 1) ^ 1, P.err, 1, (P.probe >> 4) & 1);
  if (!(P.probe & 4)) {
    const uint32_t a_dst = sbase + C::OFF_A + slot * C::A_BYTES;
    const int c = K2B_A_CHUNK(p);
#pragma unroll
    for (int i = 0; i < C::RPT; ++i) {
      const int r = K2B_A_ROW(p, i);
      const uint32_t off = C::LAYOUT == 0 ? K2B_ADDR_SW(r, c) : K2B_ADDR_IL(r, c);
      cp_async16(a_dst + off, L.src[i] + L.kb * 64, ((L.okmask >> i) & 1u) ? 16 : 0);
    }
    const int tile0 = L.nt * (C::BN / 16);
    const uint32_t c_dst = sbase + C::OFF_C + slot * C::C_BYTES;
#pragma unroll
    for (int q = 0; q < C::ITEMS; ++q) {
      const int idx = q * 128 + p;
      const size_t gi = K2B_WQ_IDX((size_t)L.e, C::RT, tile0 + K2B_ITEM_TILE(idx), L.kb, K2B_ITEM_LANE(idx), KS);
      cp_async16(c_dst + idx * 16, P.wq + gi, 16);
    }
    if (p < C::SCH) {
      const size_t gi = K2B_SQ_IDX((size_t)L.e, C::RT, tile0 + (p >> 3), L.kb, 0, KS) + (p & 7) * 4;
      cp_async16(sbase + C::OFF_SC + slot * C::SC_BYTES + p * 16, P.sq + gi, 16);
    }
  }
  cp_async_commit();   // one group per stage; the dequant waits with cp.async.wait_group (no per-thread mbarrier arrivals)
}

template <class C>
__device__ __forceinline__ void producer_dequant(const Params& P, uint32_t sbase, uint32_t j, int kb, int p, bool more) {
  const int slot = j % C::S, b = j % C::NB;
  if (more) cp_async_wait<C::D>();       // this thread's group for stage j is complete (D younger groups may fly)
  else cp_async_wait<0>();               // tail: fewer younger groups
  producer_bar();                        // every producer thread's stage-j data (A, codes, scales) has landed
  mbar_wait(sbase + C::OFF_BAR + 8 * (2 * C::S + C::NB + b), ((j / C::NB) & 1) ^ 1, P.err, 3, (P.probe >> 4) & 1);  // B buffer free
  if (!(P.probe & 1)) {
    const uint32_t c_src = sbase + C::OFF_C + slot * C::C_BYTES;
    const uint32_t s_src = sbase + C::OFF_SC + slot * C::SC_BYTES;
    const uint32_t b_dst = sbase + C::OFF_B + b * C::B_BYTES;
    const int par = kb & 1;
#pragma unroll
    for (int q = 0; q < C::ITEMS; ++q) {
      const int idx = q * 128 + p;
      const int i = K2B_ITEM_TILE(idx), lane = K2B_ITEM_LANE(idx);
      const uint4 w = lds128(c_src + idx * 16);
      const uint32_t sw = lds32(s_src + idx * 4);
      const uint32_t Sg = __byte_perm(sw, 0u, K2B_SSEL(par, 0)) * 16u + 0x75807580u;   // row g
      const uint32_t S8 = __byte_perm(sw, 0u, K2B_SSEL(par, 1)) * 16u + 0x75807580u;   // row g + 8
      const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        uint32_t o[4];
        deq4(wv[wd], o);
        const uint32_t S = wd < 2 ? Sg : S8;
        const int r = K2B_WROW(i, lane, wd), c = K2B_WCHUNK(lane, wd);
        const uint32_t off = C::LAYOUT == 0 ? K2B_ADDR_SW(r, c) : K2B_ADDR_IL(r, c);
        sts128(b_dst + off, bmul2(o[0], S), bmul2(o[1], S), bmul2(o[2], S), bmul2(o[3], S));
      }
    }
  }
  fence_proxy_async();     // generic-proxy STS and cp.async data -> visible to wgmma (async proxy) after the consumers' wait
  __syncwarp();
  if ((p & 31) == 0) mbar_arrive(sbase + C::OFF_BAR + 8 * (2 * C::S + b));   // one arrival per producer warp (count 4)
}

template <class C>
__device__ void producer(const Params& P, uint32_t sbase, int n_tiles, int p) {
  constexpr int D = C::D;
  LoadCur<C> L;
  L.t = next_valid<C>(blockIdx.x, n_tiles, P.bexp);
  if (L.t < n_tiles) load_tile_setup<C>(P, L, p);
  uint32_t jl = 0, jd = 0;
  auto advance = [&]() {
    if (++L.kb == KS) {
      L.t = next_valid<C>(L.t + gridDim.x, n_tiles, P.bexp);
      if (L.t < n_tiles) load_tile_setup<C>(P, L, p);
    }
  };
  for (int i = 0; i < D && L.t < n_tiles; ++i) { producer_load<C>(P, sbase, jl++, L, p); advance(); }
  for (int t = next_valid<C>(blockIdx.x, n_tiles, P.bexp); t < n_tiles;
       t = next_valid<C>(t + gridDim.x, n_tiles, P.bexp)) {
#pragma unroll 1
    for (int kb = 0; kb < KS; ++kb) {
      const bool more = L.t < n_tiles;
      if (more) { producer_load<C>(P, sbase, jl++, L, p); advance(); }
      producer_dequant<C>(P, sbase, jd++, kb, p, more);
    }
  }
}

// ---------------------------------------------------------------- consumer warpgroups (WG1, WG2)
template <class C>
__device__ void consumer(const Params& P, uint32_t sbase, int n_tiles, int c, int tw) {
  constexpr int NR = C::NW / 2;
  const int cm = C::WM == 2 ? c : 0, cn = C::WM == 2 ? 0 : c;
  const int w = tw >> 5, lane = tw & 31;
  float acc[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) acc[i] = 0.f;
  uint32_t j = 0;
  for (int t = next_valid<C>(blockIdx.x, n_tiles, P.bexp); t < n_tiles;
       t = next_valid<C>(t + gridDim.x, n_tiles, P.bexp)) {
    const int mb = K2B_TILE_MB(t, C::NTL), nt = K2B_TILE_NT(t, C::NTL);
    const int e = __ldg(P.bexp + mb);
    const int base = mb * C::BM + cm * 64;
    const bool any = __any_sync(0xffffffffu, pair_ok(P, base + lane) || pair_ok(P, base + lane + 32));
    const bool active = any && !(P.probe & 2);
#pragma unroll 1
    for (int kb = 0; kb < KS; ++kb, ++j) {
      const int slot = j % C::S, b = j % C::NB;
      // B tile dequantised; also implies A stage landed (the producers passed wait_group + bar.sync before arriving)
      mbar_wait(sbase + C::OFF_BAR + 8 * (2 * C::S + b), (j / C::NB) & 1, P.err, 4, (P.probe >> 4) & 1);
      if (active) {
        if (!(P.probe & 8)) fence_proxy_async();   // flag 8 drops this consumer-side insurance fence (CUTLASS omits it)
        const uint64_t da = make_desc(sbase + C::OFF_A + slot * C::A_BYTES + cm * 8192, C::LBO, K2B_SBO, C::LT);
        const uint64_t db = make_desc(sbase + C::OFF_B + b * C::B_BYTES + cn * C::NW * 128, C::LBO, K2B_SBO, C::LT);
        fence_acc<NR>(acc);
        wg_fence();
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16)
          wgmma_ss<C::NW>(acc, da + (uint64_t)(k16 * (C::KADV >> 4)), db + (uint64_t)(k16 * (C::KADV >> 4)),
                          (kb | k16) != 0);
        wg_commit();
        fence_acc<NR>(acc);
      }
      wg_wait<1>();                       // wgmma(j - 1) retired: its A slot and B buffer are free
      if (kb > 0 && lane == 0) {
        mbar_arrive(sbase + C::OFF_BAR + 8 * (C::S + (j - 1) % C::S));
        mbar_arrive(sbase + C::OFF_BAR + 8 * (2 * C::S + C::NB + (j - 1) % C::NB));
      }
    }
    wg_wait<0>();
    fence_acc<NR>(acc);
    if (lane == 0) {
      mbar_arrive(sbase + C::OFF_BAR + 8 * (C::S + (j - 1) % C::S));
      mbar_arrive(sbase + C::OFF_BAR + 8 * (2 * C::S + C::NB + (j - 1) % C::NB));
    }
    if (!active) continue;
    // epilogue: FP32 global scale, SwiGLU (MODE 1) in registers, BF16 stores of valid rows only
    const float gs = __ldg(P.gs + e);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int srow = base + K2B_ACC_ROW(w, lane, 2 * h);
      if (!pair_ok(P, srow)) continue;
      if constexpr (C::MODE == 1) {
        const int tb = (nt * C::BN + cn * C::NW) / 16;
        __nv_bfloat16* orow = P.out + (size_t)srow * (C::N / 2);
#pragma unroll
        for (int i = 0; i < C::NW / 16; ++i) {
          float v[2];
#pragma unroll
          for (int e2 = 0; e2 < 2; ++e2) {
            const float g = acc[K2B_GATE_REG(i, h, e2)] * gs, u = acc[K2B_UP_REG(i, h, e2)] * gs;
            v[e2] = g / (1.f + __expf(-g)) * u;
          }
          *reinterpret_cast<uint32_t*>(orow + K2B_INTER_COL(tb, i, lane, 0)) = pack_bf2(v[0], v[1]);
        }
      } else {
        const int pair = __ldg(P.sorted + srow);
        __nv_bfloat16* orow = P.out + (size_t)pair * C::N + nt * C::BN + cn * C::NW;
#pragma unroll
        for (int jj = 0; jj < C::NW / 8; ++jj) {
          const int v0 = 4 * jj + 2 * h;
          *reinterpret_cast<uint32_t*>(orow + K2B_ACC_COL(v0, lane)) = pack_bf2(acc[v0] * gs, acc[v0 + 1] * gs);
        }
      }
    }
  }
}

template <class C>
__global__ void __launch_bounds__(384, 1) k2b1_gemm(const Params P) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  const int tid = threadIdx.x, wg = tid >> 7, tw = tid & 127;
  if (sbase & 1023u) {   // the SW128 atoms need 1024 B alignment; never seen, but fail loudly instead of silently
    if (tid == 0) atomicCAS(P.err, 0, 7);
    return;
  }
  if (tid == 0) {
    for (int s = 0; s < C::S; ++s) {
      mbar_init(sbase + C::OFF_BAR + 8 * s, 1);                // (unused since v1.1: cp.async.wait_group + bar.sync)
      mbar_init(sbase + C::OFF_BAR + 8 * (C::S + s), 8);       // empty: one lane per consumer warp
    }
    for (int b = 0; b < C::NB; ++b) {
      mbar_init(sbase + C::OFF_BAR + 8 * (2 * C::S + b), 4);            // B full: one lane per producer warp
      mbar_init(sbase + C::OFF_BAR + 8 * (2 * C::S + C::NB + b), 8);    // B empty: one lane per consumer warp
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int ntp = __ldg(P.ntp);
  int n_mb = ntp > 0 ? ntp / C::BM : 0;
  n_mb = min(n_mb, min(P.n_blocks, P.n_sorted / C::BM));
  const int n_tiles = n_mb * C::NT;
  if (wg == 0) {
    if constexpr (C::REG) asm volatile("setmaxnreg.dec.sync.aligned.u32 88;\n" ::: "memory");
    producer<C>(P, sbase, n_tiles, tw);
  } else {
    if constexpr (C::REG) asm volatile("setmaxnreg.inc.sync.aligned.u32 208;\n" ::: "memory");
    consumer<C>(P, sbase, n_tiles, wg - 1, tw);
  }
}

template <class C>
void launch(const Params& p, int grid) {
  static bool attr = false;
  if (!attr) {
    TORCH_CHECK(cudaFuncSetAttribute(k2b1_gemm<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess,
                "k2b1: cudaFuncSetAttribute failed");
    attr = true;
  }
  k2b1_gemm<C><<<grid, 384, C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "k2b1 launch: ", cudaGetErrorString(e));
}
}  // namespace k2b1

void run(int64_t vid, torch::Tensor x, torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor out,
         torch::Tensor sorted, torch::Tensor bexp, torch::Tensor ntp, torch::Tensor err, int64_t n_pairs,
         int64_t grid, int64_t probe) {
  k2b1::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.wq = reinterpret_cast<const uint4*>(wq.data_ptr());
  p.sq = reinterpret_cast<const uint32_t*>(sq.data_ptr());
  p.gs = gs.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.sorted = sorted.data_ptr<int>();
  p.bexp = bexp.data_ptr<int>();
  p.ntp = ntp.data_ptr<int>();
  p.err = err.data_ptr<int>();
  p.n_pairs = (int)n_pairs;
  p.n_sorted = (int)sorted.numel();
  p.n_blocks = (int)bexp.numel();
  p.probe = (int)probe;
  switch (vid) {
@CASES@
    default: TORCH_CHECK(false, "k2b1: unknown variant");
  }
}

int64_t smem_bytes(int64_t vid) {
  switch (vid) {
@SMEM@
    default: return -1;
  }
}
"""
K2B_CPP_SRC = """#include <torch/extension.h>
void run(int64_t vid, torch::Tensor x, torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor out,
         torch::Tensor sorted, torch::Tensor bexp, torch::Tensor ntp, torch::Tensor err, int64_t n_pairs,
         int64_t grid, int64_t probe);
int64_t smem_bytes(int64_t vid);
"""


def k2b_variants(cfgs):
    """[(mode, bm, bn, s, nb, layout, reg)]: variant id 2*i + (mode - 1) for the i-th config."""
    out = []
    for cfg in cfgs.values():
        for mode in (1, 2):
            out.append((mode,) + tuple(cfg))
    return out


def k2b_gen_source(variants):
    cases, smem = [], []
    for i, v in enumerate(variants):
        cfg = "k2b1::Cfg<%d, %d, %d, %d, %d, %d, %d>" % v
        cases.append("    case %d: k2b1::launch<%s >(p, (int)grid); break;" % (i, cfg))
        smem.append("    case %d: return %s::SMEM;" % (i, cfg))
    nws = sorted({v[2] // (2 // (v[1] // 64)) for v in variants})
    wg = "".join(k2b_wgmma_src(n) for n in nws)
    return (K2B_KERNEL_SRC.replace("@WGMMA@", wg).replace("@CASES@", "\n".join(cases))
            .replace("@SMEM@", "\n".join(smem)))
# ===== K2B PURE END =====


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    t_start = time.time()
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    state["k2b"] = _k2b1_prepare(weights, state, t_start)
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)




# ----------------------------------------------------------------------------------------------
# K2b v1 experts (host side)
# ----------------------------------------------------------------------------------------------

def _k2b1_build(cfgs, verbose):
    """load_inline one extension holding MODE 1 + MODE 2 of every config in cfgs. Returns (module, variants)."""
    from torch.utils.cpp_extension import load_inline
    variants = k2b_variants(cfgs)
    src = k2b_gen_source(variants)
    tag = hashlib.sha1((K2B_CPP_SRC + src).encode()).hexdigest()[:10]
    flags = ["-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-warn-spills"]
    if verbose:
        flags.append("-Xptxas=-v")
    mod = load_inline(name=f"k2b_v1_{tag}", cpp_sources=K2B_CPP_SRC, cuda_sources=src,
                      functions=["run", "smem_bytes"], extra_cuda_cflags=flags, verbose=verbose)
    return mod, variants


def _k2b1_launch(k, name, mode, x, wq, sq, gs, out, sorted_ids, block_experts, ntp, n_pairs, probe=0):
    bm, bn = k["cfgs"][name][0], k["cfgs"][name][1]
    n = K2B_N13 if mode == 1 else K2B_N2
    max_tiles = max(1, (sorted_ids.numel() // bm) * (n // bn))
    grid = min(k["sms"], max_tiles)
    k["mod"].run(k["vid"][(name, mode)], x, wq, sq, gs, out, sorted_ids, block_experts, ntp, k["err"],
                 n_pairs, grid, probe | K2B_RUNTIME_FLAGS)


def _k2b1_align(local_ids, bm, ignore):
    if ignore:
        return moe_align_block_size(local_ids, bm, LOCAL_EXPERTS, ignore_invalid_expert=True)
    return moe_align_block_size(local_ids, bm, N_EXPERTS)


def _k2b1_experts(x, topk_weights, topk_ids, state, name, probe=0):
    """Same contract as _experts: x bf16 [T, 4096], topk_ids [T, 8] with this GPU's experts as 0..31."""
    k = state["k2b"]
    bank = state["bank"]
    num_tokens = x.shape[0]
    x = x.contiguous()
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    bm = k["cfgs"][name][0]
    sorted_ids, block_experts, ntp = _k2b1_align(local_ids, bm, k["align_ignore"])
    n_pairs = num_tokens * TOP_K
    inter = torch.empty((sorted_ids.numel(), D_EXPERT), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((n_pairs, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    _k2b1_launch(k, name, 1, x, k["wq13"], k["sq13"], k["gs13"], inter, sorted_ids, block_experts, ntp, n_pairs,
                 probe)
    _k2b1_launch(k, name, 2, inter, k["wq2"], k["sq2"], k["gs2"], y, sorted_ids, block_experts, ntp, n_pairs,
                 probe)
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, TOP_K, D_EXPERT), topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=local_ids)


def _experts_dispatch(x, topk_weights, topk_ids, state):
    k = state.get("k2b")
    if k is not None and k["plan"]:
        name = k["plan"].get(k2b_class(x.shape[0]))
        if name is not None:
            return _k2b1_experts(x, topk_weights, topk_ids, state, name)
    return _experts(x, topk_weights, topk_ids, state)


def _k2b1_check_align(device, bms):
    """moe_align_block_size(..., ignore_invalid_expert=True): every local pair exactly once, in a block labelled
    with its expert; no -1 block inside ntp (as k2b_v0, which passed on the judge)."""
    try:
        ids, _ = k2b_synth_routing(997, 7, device)
        emap = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
        emap[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
        local = emap[ids.long()].contiguous()
        for bm in sorted(bms):
            sid, bexp, ntp = moe_align_block_size(local, bm, LOCAL_EXPERTS, ignore_invalid_expert=True)
            torch.cuda.synchronize()
            n = local.numel()
            ntp = int(ntp.reshape(-1)[0])
            if ntp % bm or ntp > sid.numel() or ntp // bm > bexp.numel():
                return False, f"bm={bm}: ntp {ntp} vs {sid.numel()} / {bexp.numel()}"
            s = sid[:ntp].long()
            blk_e = bexp[: ntp // bm].long().repeat_interleave(bm)
            real = s < n
            flat = local.reshape(-1).long()
            want = torch.nonzero(flat >= 0).reshape(-1)
            got = torch.sort(s[real]).values
            if got.numel() != want.numel() or not bool(torch.equal(got, want)):
                return False, f"bm={bm}: pair set mismatch ({got.numel()} vs {want.numel()})"
            if not bool((flat[s[real]] == blk_e[real]).all()):
                return False, f"bm={bm}: pair in a block of another expert"
            if bool((bexp[: ntp // bm] < 0).any()):
                return False, f"bm={bm}: -1 block inside ntp"
        return True, "ok"
    except Exception as ex:  # noqa: BLE001
        return False, repr(ex)


def _k2b1_graph_time(fn, flush, reps=K2B_TIMING_REPS):
    """Median ms of a CUDA-graph replay of fn() after an L2-flushing write (as the judge times)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    del graph
    times.sort()
    return times[len(times) // 2]


def _k2b1_err(k):
    e = int(k["err"][0])
    if e:
        k["err"].zero_()
    return e


def _k2b1_prepare(weights, state, t_start):
    """Repack (K2A v2.2 layout), build, self-test and time the candidates; returns the K2b state (plan {} =
    Marlin everywhere). Deadlines are absolute from prepare()'s start (the 300 s limit covers the baseline's)."""
    t0 = time.time()
    print(f"[k2b] baseline prepare took {t0 - t_start:.1f}s", flush=True)
    dev = weights["w13"].device
    k = {"plan": {}, "align_ignore": False}
    if not K2B_ENABLE:
        print("[k2b] K2B_ENABLE=False: stock Marlin experts", flush=True)
        return k
    try:
        wq13 = torch.empty((K2B_E, K2B_N13 // 16, K2B_K // 64, 32, 4), dtype=torch.int32, device=dev)
        sq13 = torch.empty((K2B_E, K2B_N13 // 16, K2B_K // 128, 32, 4), dtype=torch.uint8, device=dev)
        wq2 = torch.empty((K2B_E, K2B_N2 // 16, K2B_K // 64, 32, 4), dtype=torch.int32, device=dev)
        sq2 = torch.empty((K2B_E, K2B_N2 // 16, K2B_K // 128, 32, 4), dtype=torch.uint8, device=dev)
        for e in range(K2B_E):
            wq13[e], sq13[e] = prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True)
            wq2[e], sq2[e] = prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False)
        k.update(wq13=wq13, sq13=sq13, wq2=wq2, sq2=sq2,
                 gs13=(weights["w13_gs"][:K2B_E].float() * W_SHIFT).contiguous(),
                 gs2=(weights["w2_gs"][:K2B_E].float() * W_SHIFT).contiguous(),
                 err=torch.zeros((4,), dtype=torch.int32, device=dev),
                 sms=torch.cuda.get_device_properties(dev).multi_processor_count)
        torch.cuda.synchronize()
        print(f"[k2b] repacked (K2A v2.2 layout) in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[k2b] repack failed, stock Marlin experts: {ex!r}", flush=True)
        return k

    cfgs = k2b_all_configs(K2B_LAB)
    t1 = time.time()
    try:
        mod, variants = _k2b1_build(cfgs, verbose=K2B_LAB)
    except Exception as ex:  # noqa: BLE001
        print(f"[k2b] build of all configs failed ({time.time() - t1:.1f}s): {str(ex)[-2500:]}", flush=True)
        first = next(iter(cfgs))
        cfgs = {first: cfgs[first]}
        try:
            mod, variants = _k2b1_build(cfgs, verbose=True)
        except Exception as ex2:  # noqa: BLE001
            print(f"[k2b] build of {first} failed too: {str(ex2)[-2500:]}; stock Marlin experts", flush=True)
            return k
    k["mod"], k["cfgs"] = mod, cfgs
    k["vid"] = {}
    for i, name in enumerate(cfgs):
        k["vid"][(name, 1)], k["vid"][(name, 2)] = 2 * i, 2 * i + 1
    print(f"[k2b] built {len(variants)} variants in {time.time() - t1:.1f}s; smem "
          + ", ".join(f"{n} {mod.smem_bytes(k['vid'][(n, 1)])}" for n in cfgs), flush=True)

    ok, msg = _k2b1_check_align(dev, {c[0] for c in cfgs.values()})
    k["align_ignore"] = ok
    print(f"[k2b] align ignore_invalid_expert: {'ok' if ok else 'NOT ok (' + msg + '), verbatim -1 blocks'}",
          flush=True)

    sub = {"k2b": k, "bank": state["bank"], "post_expert_w": state["post_expert_w"]}
    tests = []
    try:
        for tokens, seed, zipf in ((300, 11, 1.0), (2048, 12, 2.0)):
            g = torch.Generator(device=dev).manual_seed(seed + 100)
            x = torch.randn((tokens, D_EXPERT), generator=g, device=dev).to(torch.bfloat16)
            ids, wts = k2b_synth_routing(tokens, seed, dev, zipf=zipf)
            ref = k2b_ref_experts(x, ids, wts, weights, weights["post_expert_norm"])
            tests.append((tokens, x, ids, wts, ref))
        for tokens, x, ids, wts, ref in tests:
            got = _experts(x, wts, ids, state)
            torch.cuda.synchronize()
            tot, row = k2b_rel(got, ref)
            print(f"[k2b] marlin vs fp32 ref T={tokens}: rel {tot:.3e} worst row {row:.3e}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[k2b] self-test setup failed, stock Marlin experts: {ex!r}", flush=True)
        return k

    passed = []
    for name in cfgs:
        if time.time() - t_start > K2B_DEADLINE_S:
            print(f"[k2b] deadline reached ({time.time() - t_start:.0f}s), skipping {name}", flush=True)
            continue
        t2 = time.time()
        try:
            worst = 0.0
            for tokens, x, ids, wts, ref in tests:
                got = _k2b1_experts(x, wts, ids, sub, name)
                torch.cuda.synchronize()
                err = _k2b1_err(k)
                tot, row = k2b_rel(got, ref)
                fin = bool(torch.isfinite(got.float()).all())
                print(f"[k2b] {name} T={tokens}: rel {tot:.3e} worst row {row:.3e} finite={fin} err={err}",
                      flush=True)
                if err or not (fin and tot < 1.0e-2 and row < 5.0e-2):
                    raise RuntimeError(f"self-test mismatch at T={tokens}")
                worst = max(worst, tot)
            passed.append(name)
            print(f"[k2b] {name} PASS (worst rel {worst:.3e}, {time.time() - t2:.1f}s)", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[k2b] {name} FAILED ({time.time() - t2:.1f}s): {str(ex)[-1500:]}", flush=True)
            _k2b1_err(k)
    del tests
    torch.cuda.empty_cache()
    if not passed:
        print("[k2b] no candidate passed: stock Marlin experts", flush=True)
        return k

    # Timing under CUDA graphs (cold L2) on busiest-GPU-like synthetic routing; choose per token-count class.
    try:
        flush = torch.empty((96 << 20,), dtype=torch.uint8, device=dev)
        for cls, _, tokens in K2B_CLASSES:
            g = torch.Generator(device=dev).manual_seed(tokens)
            x = torch.randn((tokens, D_EXPERT), generator=g, device=dev).to(torch.bfloat16)
            ids, wts = k2b_synth_routing(tokens, tokens + 1, dev)
            routes = int(((ids >= 0) & (ids < LOCAL_EXPERTS)).sum())
            flop = 2.0 * routes * 3 * D_EXPERT * D_EXPERT
            t_m = _k2b1_graph_time(lambda: _experts(x, wts, ids, state), flush)
            line = [f"marlin {t_m * 1e3:.0f}us ({flop / t_m / 1e9:.0f} TF/s)"]
            best, t_best = None, t_m * K2B_MIN_GAIN
            for name in passed:
                if time.time() - t_start > K2B_TIMING_DEADLINE_S:
                    line.append(f"(deadline: {name} and later not timed)")
                    break
                try:
                    t_c = _k2b1_graph_time(lambda: _k2b1_experts(x, wts, ids, sub, name), flush)
                    if _k2b1_err(k):
                        line.append(f"{name} ERR(timeout flag)")
                        continue
                except Exception as ex:  # noqa: BLE001
                    line.append(f"{name} ERR {str(ex)[-200:]}")
                    continue
                line.append(f"{name} {t_c * 1e3:.0f}us ({flop / t_c / 1e9:.0f} TF/s)")
                if t_c < t_best:
                    best, t_best = name, t_c
            k["plan"][cls] = best
            print(f"[k2b] time {cls} T={tokens} routes={routes}: " + " | ".join(line) + f" -> {best or 'marlin'}",
                  flush=True)
            if K2B_PROBES and cls == "large":
                # In-situ stage probes (results are garbage, timing only): 1 = no dequant, 2 = no wgmma,
                # 3 = loads only, 7 = pipeline handshakes only. Each probe removes one stage's work.
                for name in passed[:4]:
                    if time.time() - t_start > K2B_TIMING_DEADLINE_S:
                        break
                    pl = []
                    for probe in K2B_PROBES:
                        try:
                            t_p = _k2b1_graph_time(lambda: _k2b1_experts(x, wts, ids, sub, name, probe), flush)
                            _k2b1_err(k)
                            pl.append(f"p{probe} {t_p * 1e3:.0f}us")
                        except Exception as ex:  # noqa: BLE001
                            pl.append(f"p{probe} ERR {str(ex)[-120:]}")
                    print(f"[k2b] probe {name} T={tokens}: " + " | ".join(pl), flush=True)
        x = ids = wts = flush = None
        torch.cuda.empty_cache()
    except Exception as ex:  # noqa: BLE001
        print(f"[k2b] timing failed ({ex!r}); stock Marlin experts", flush=True)
        k["plan"] = {}
    torch.cuda.empty_cache()
    print(f"[k2b] plan {k['plan']} (k2b prepare {time.time() - t0:.1f}s, total {time.time() - t_start:.1f}s)",
          flush=True)
    return k


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts_dispatch(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids

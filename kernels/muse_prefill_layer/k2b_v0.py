# okc: problem=muse_prefill_layer
# nonce k2b-v0-1
"""K2b v0: SGLang's prefill decoder layer (baseline.py, verbatim) with the routed experts' two Marlin
W4A16 GEMMs replaced by a Triton grouped NVFP4 x BF16 GEMM (wgmma via tl.dot on sm_90).

Experts path (per call, all shapes static, no host syncs, CUDA-graph capturable):
  1. moe_align_block_size (SGLang's CUDA kernel): (token, slot) pairs sorted by local expert, padded to
     BM rows per block. Non-local slots are dropped (ignore_invalid_expert=True) when prepare() verified
     that contract on this image, else the baseline's verbatim call (-1 blocks, skipped by the GEMM).
  2. _k2b_gemm MODE 1:  inter[sorted row] = SiLU(gs13*(x W1^T)) * gs13*(x W3^T)   (bf16, sorted order)
     tile = BM sorted rows x BN interleaved gate/up weight rows, full K = 4096, fp32 accumulator.
  3. _k2b_gemm MODE 2:  y[pair] = bf16(gs2 * (inter W2^T))                         (bf16, pair order)
  4. fused_post_expert_norm_weighted_sum (SGLang's Triton kernel, as the baseline).
NVFP4 weights stay NVFP4 (lossless repack: 0.5 B codes + 1 B E4M3 scale per 16, same 0.5625 B/elem):
  wq[e][ks][h][n][j] (uint8, K-step ks of 64): low nibble = W[n, 64ks+32h+j]  (scale group 4ks+2h),
                                                high nibble = W[n, 64ks+32h+16+j] (scale group 4ks+2h+1)
  sq[e][ks][n]       (int32 = 4 raw E4M3 bytes, byte g) = scale of W[n, 64ks+16g .. +15]  (one 4 B load per
                     row and K-step, so the scale loads are cp.async-eligible and pipeline with the rest)
  W13 rows are interleaved: prepacked row 2i = gate row i, 2i+1 = up row i (SwiGLU inside one tile).
Dequant (exact in BF16, see notes/study/03 section C): DECODE 0 = inline-asm bit placement
e2m1 * 2^-126 (BF16; code 1 is a subnormal, as Marlin/k2a) times BF16(sf * 2^118) with fma.rn.bf16x2,
global scale x 2^8 on the fp32 accumulator; DECODE 1 = integer ops building normal BF16 e2m1 values, times
BF16(sf), global scale on the accumulator. Each 16-wide K group is one tl.dot(K=16) (4 per K-step).
prepare() repacks, compiles a few candidate configs, self-tests each against an FP32 reference, times the
passing ones against the stock Marlin path under CUDA graphs, and keeps the faster per token-count class
(the stock Marlin path stays whenever nothing beats it, or anything fails).
"""

from typing import TypedDict

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

import triton
import triton.language as tl


# ===== K2B PURE BEGIN =====  (torch only; k2b_emulate.py execs this section on CPU)
K2B_ENABLE = True              # False: the stock Marlin experts everywhere
K2B_SELF_TEST = True           # a candidate is used only after it passes the FP32-reference self-test
K2B_DEADLINE_S = 190.0         # since prepare() started: no new candidate is compiled/tested after this
K2B_TIMING_DEADLINE_S = 250.0  # since prepare() started: no further candidate is timed after this (limit 300)
K2B_MIN_GAIN = 0.97            # a candidate replaces Marlin for a class only if t_cand < K2B_MIN_GAIN * t_marlin

K2B_E = 32                     # local experts
K2B_K = 4096                   # K of both GEMMs (D_EXPERT)
K2B_N13 = 8192                 # W13 rows (gate 4096 + up 4096)
K2B_N2 = 4096                  # W2 rows
K2B_KSTEP = 64                 # elements per K-step = 4 scale groups of 16
K2B_TOPK = 8

# name -> (orient, decode, BM, BN, num_warps, num_stages, group_m)
#   orient 0: acc[BM tokens, BN weights] = x . w^T  (decoded weights are tl.dot's B operand)
#   orient 1: acc[BN weights, BM tokens] = w . x^T  (swap-AB: decoded weights are tl.dot's A operand)
#   decode 0: inline asm (bit placement + fma.rn.bf16x2), decode 1: plain integer ops (normal BF16 only)
#   BM is also the moe_align block size (rows of one expert per tile).
K2B_CONFIGS = {                # tried in this order (prepare stops adding candidates at K2B_DEADLINE_S)
    "ss_asm_128x256": (0, 0, 128, 256, 8, 3, 8),
    "ss_asm_64x256": (0, 0, 64, 256, 4, 4, 8),
    "sw_asm_256x128": (1, 0, 128, 256, 8, 3, 8),
    "ss_pure_128x256": (0, 1, 128, 256, 8, 3, 8),
    "ss_asm_256x128": (0, 0, 256, 128, 8, 3, 8),
    "sw_asm_128x64": (1, 0, 64, 128, 4, 4, 8),
}
# token-count classes: (name, max tokens, representative tokens for timing)
K2B_CLASSES = (("small", 1024, 753), ("mid", 6144, 3012), ("large", 1 << 30, 16384))

_K2B_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def k2b_class(tokens):
    for name, hi, _ in K2B_CLASSES:
        if tokens <= hi:
            return name
    return K2B_CLASSES[-1][0]


def k2b_row_perm(n_rows, gate_up, device=None):
    """prepacked row -> original row. W13: 2i -> gate row i, 2i+1 -> up row i (n_rows/2 + i)."""
    m = torch.arange(n_rows, device=device)
    if not gate_up:
        return m
    return (m % 2) * (n_rows // 2) + m // 2


def k2b_repack_expert(w, sf, gate_up):
    """One expert. w uint8 [N, K/2] (low nibble = even k), sf E4M3 [N, K/16] ->
    wq uint8 [K/64, 2, N, 16], sq int32 [K/64, N] (the 4 raw E4M3 bytes of a row's K-step, group g in byte g).
    Lossless (a byte permutation of the codes and of the scales)."""
    n, k2 = w.shape
    k = 2 * k2
    ks = k // K2B_KSTEP
    perm = k2b_row_perm(n, gate_up, device=w.device)
    w = w.view(torch.uint8)[perm]
    sfb = sf.view(torch.uint8)[perm]
    codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(n, k)           # codes[n, k]
    c = codes.view(n, ks, 2, 2, 16)                                         # k = 64ks + 32h + 16lohi + j
    byte = c[:, :, :, 0, :] | (c[:, :, :, 1, :] << 4)                       # [n, ks, h, j]
    wq = byte.permute(1, 2, 0, 3).contiguous()                              # [ks, h, n, j]
    sq = sfb.view(n, ks, 4).permute(1, 0, 2).contiguous().view(torch.int32).view(ks, n)   # [ks, n] x 4 B
    return wq, sq


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
    """Busiest-GPU-like routing for self-tests and timing (defaults give 1.82-1.84 local routes per token on
    16 touched experts, top expert ~0.47 routes/token; the judge's busiest GPU: 1.75-1.86 routes/token,
    ~15-17 experts): local ids 0..31 (after the local_start rotation), `hot` of them with Zipf skew, the
    rest of the routes on ids 32..255. Distinct ids per token (Gumbel top-k). Returns ids int32 [T, 8],
    weights float32 [T, 8]."""
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


def k2b_ref_experts(x, ids, wts, weights, post_w):
    """FP32 semantics of reference._experts: x [T, 4096], ids int [T, 8] (local experts 0..31, others >= 32),
    wts [T, 8]; returns float32 [T, 4096]."""
    xf = x.float()
    out = torch.zeros(xf.shape, dtype=torch.float32, device=x.device)
    nw = 1.0 + post_w.float()
    ids = ids.long()
    for e in torch.unique(ids).tolist():
        if not 0 <= e < K2B_E:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = xf[tok] @ k2b_dequant(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :K2B_K]) * h[:, K2B_K:]
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
# ===== K2B PURE END =====


# ----------------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------------

@triton.jit
def _k2b_e2m1_bf16(c):
    """c int32 E2M1 codes (0..15) -> BF16 holding the exact E2M1 value (normal numbers only)."""
    mag = c & 7
    bits = tl.where(mag >= 2, 0x3F00 + (mag << 6), mag * 0x3F00)
    bits = bits | ((c & 8) << 12)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _k2b_scale(sw, g: tl.constexpr):
    """Byte g of the packed int32 scale word -> the E4M3 block scale as float32 (exact)."""
    return ((sw >> (8 * g)) & 0xFF).to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)


@triton.jit
def _k2b_decode(wb, s0, s1, BN: tl.constexpr, DECODE: tl.constexpr):
    """wb uint8 [BN, 16] packed codes, s0/s1 float32 [BN] block scales of the low/high nibbles' groups ->
    (lo, hi) BF16 [BN, 16]: DECODE 0: e2m1*sf*2^-8; DECODE 1: e2m1*sf (both exact)."""
    if DECODE == 0:
        S0 = tl.broadcast_to((s0 * 3.3230699894622794e+35).to(tl.bfloat16).to(tl.int16, bitcast=True)[:, None],
                             (BN, 16))
        S1 = tl.broadcast_to((s1 * 3.3230699894622794e+35).to(tl.bfloat16).to(tl.int16, bitcast=True)[:, None],
                             (BN, 16))
        # Per byte i of the packed u32 ($4): lane i of the outputs. Byte = (hi code << 4) | lo code, code =
        # sign:e1:e0:m. BF16 bits of e2m1 * 2^-126: sign -> bit 15, e1 e0 m -> bits 8..6. Then x scale.
        lo_i, hi_i = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 t01, t23, ta, tb, r0, r1, r2, r3, zz, nz;
            mov.b32 zz, 0;
            mov.b32 nz, 0x80008000;
            prmt.b32 t01, $4, zz, 0x4140;
            prmt.b32 t23, $4, zz, 0x4342;
            shl.b32 ta, t01, 12;
            and.b32 ta, ta, 0x80008000;
            shl.b32 tb, t01, 6;
            and.b32 tb, tb, 0x01C001C0;
            or.b32 ta, ta, tb;
            fma.rn.bf16x2 r0, ta, $5, nz;
            shl.b32 ta, t23, 12;
            and.b32 ta, ta, 0x80008000;
            shl.b32 tb, t23, 6;
            and.b32 tb, tb, 0x01C001C0;
            or.b32 ta, ta, tb;
            fma.rn.bf16x2 r1, ta, $6, nz;
            shl.b32 ta, t01, 8;
            and.b32 ta, ta, 0x80008000;
            shl.b32 tb, t01, 2;
            and.b32 tb, tb, 0x01C001C0;
            or.b32 ta, ta, tb;
            fma.rn.bf16x2 r2, ta, $7, nz;
            shl.b32 ta, t23, 8;
            and.b32 ta, ta, 0x80008000;
            shl.b32 tb, t23, 2;
            and.b32 tb, tb, 0x01C001C0;
            or.b32 ta, ta, tb;
            fma.rn.bf16x2 r3, ta, $8, nz;
            mov.b32 $0, r0;
            mov.b32 $1, r1;
            mov.b32 $2, r2;
            mov.b32 $3, r3;
            }
            """,
            constraints="=r,=r,=r,=r,r,r,r,r,r",
            args=[wb, S0, S1],
            dtype=(tl.int16, tl.int16),
            is_pure=True,
            pack=4,
        )
        lo = lo_i.to(tl.bfloat16, bitcast=True)
        hi = hi_i.to(tl.bfloat16, bitcast=True)
    else:
        v = wb.to(tl.int32)
        lo = _k2b_e2m1_bf16(v & 15) * s0.to(tl.bfloat16)[:, None]
        hi = _k2b_e2m1_bf16(v >> 4) * s1.to(tl.bfloat16)[:, None]
    return lo, hi


@triton.jit(do_not_specialize=["n_pairs", "n_mb", "n_sorted"])
def _k2b_gemm(x_ptr, wq_ptr, sq_ptr, gs_ptr, out_ptr, sorted_ptr, expert_ptr, ntp_ptr,
              n_pairs, n_mb, n_sorted,
              N: tl.constexpr, K: tl.constexpr, TOP_K: tl.constexpr, MODE: tl.constexpr,
              ORIENT: tl.constexpr, DECODE: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
    """One (BM sorted rows) x (BN weight rows) tile over the full K.
    MODE 1: x = activations [T, K] gathered by pair // TOP_K; out = inter [n_mb*BM, N/2] in sorted order.
    MODE 2: x = inter [n_mb*BM, K] in sorted order; out = y [n_pairs, N] scattered by pair id."""
    pid = tl.program_id(0)
    NUM_N: tl.constexpr = N // BN
    per_group = GROUP_M * NUM_N
    group = pid // per_group
    first_m = group * GROUP_M
    gsize = tl.minimum(n_mb - first_m, GROUP_M)
    pid_m = first_m + (pid % per_group) % gsize
    pid_n = (pid % per_group) // gsize
    ntp = tl.load(ntp_ptr)
    if pid_m * BM >= ntp:
        return
    e = tl.load(expert_ptr + pid_m)
    if (e < 0) | (e >= 32):          # -1: another GPU's experts (verbatim align); >= 32 never happens
        return
    e64 = e.to(tl.int64)
    offs_m = pid_m * BM + tl.arange(0, BM)
    pair = tl.load(sorted_ptr + offs_m, mask=offs_m < n_sorted, other=n_pairs)
    valid = (pair >= 0) & (pair < n_pairs)
    if MODE == 1:
        xrow = tl.where(valid, pair // TOP_K, 0).to(tl.int64)
    else:
        xrow = offs_m.to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_j = tl.arange(0, 16)
    KS: tl.constexpr = K // 64
    x_ptrs = x_ptr + xrow[:, None] * K + offs_j[None, :]
    w_ptrs = wq_ptr + e64 * (KS * 2 * N * 16) + offs_n[:, None] * 16 + offs_j[None, :]
    s_ptrs = sq_ptr + e64 * (KS * N) + offs_n
    if ORIENT == 0:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    else:
        acc = tl.zeros((BN, BM), dtype=tl.float32)
    for ks in range(KS):
        sw = tl.load(s_ptrs + ks * N)                       # [BN] int32: this K-step's 4 scale bytes per row
        for h in tl.static_range(2):
            wb = tl.load(w_ptrs + (ks * 2 + h) * (N * 16))
            s0 = _k2b_scale(sw, 2 * h)
            s1 = _k2b_scale(sw, 2 * h + 1)
            xa = tl.load(x_ptrs + (ks * 64 + h * 32), mask=valid[:, None], other=0.0)
            xb = tl.load(x_ptrs + (ks * 64 + h * 32 + 16), mask=valid[:, None], other=0.0)
            wl, wh = _k2b_decode(wb, s0, s1, BN, DECODE)
            if ORIENT == 0:
                acc = tl.dot(xa, tl.trans(wl), acc)
                acc = tl.dot(xb, tl.trans(wh), acc)
            else:
                acc = tl.dot(wl, tl.trans(xa), acc)
                acc = tl.dot(wh, tl.trans(xb), acc)
    if ORIENT == 1:
        acc = tl.trans(acc)
    gs = tl.load(gs_ptr + e64)
    if DECODE == 0:
        gs = gs * 256.0
    acc = acc * gs
    if MODE == 1:
        gate, up = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
        v = gate * tl.sigmoid(gate) * up
        offs_c = pid_n * (BN // 2) + tl.arange(0, BN // 2)
        o_ptrs = out_ptr + offs_m.to(tl.int64)[:, None] * (N // 2) + offs_c[None, :]
        tl.store(o_ptrs, v.to(tl.bfloat16))
    else:
        o_ptrs = out_ptr + pair.to(tl.int64)[:, None] * N + offs_n[None, :]
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=valid[:, None])


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
    state["k2b"] = _k2b_prepare(weights, state, t_start)
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
# K2b experts (host side)
# ----------------------------------------------------------------------------------------------

def _k2b_align(local_ids, bm, ignore):
    if ignore:
        return moe_align_block_size(local_ids, bm, LOCAL_EXPERTS, ignore_invalid_expert=True)
    return moe_align_block_size(local_ids, bm, N_EXPERTS)


def _k2b_launch(mode, cfg, x, wq, sq, gs, out, sorted_ids, block_experts, ntp, n_pairs):
    orient, decode, bm, bn, warps, stages, group_m = K2B_CONFIGS[cfg]
    n = K2B_N13 if mode == 1 else K2B_N2
    n_mb = block_experts.numel()
    grid = (n_mb * (n // bn),)
    _k2b_gemm[grid](x, wq, sq, gs, out, sorted_ids, block_experts, ntp, n_pairs, n_mb, sorted_ids.numel(),
                    N=n, K=K2B_K, TOP_K=K2B_TOPK, MODE=mode, ORIENT=orient, DECODE=decode,
                    BM=bm, BN=bn, GROUP_M=group_m, num_warps=warps, num_stages=stages)


def _k2b_experts(x, topk_weights, topk_ids, state, cfg):
    """Same contract as _experts: x bf16 [T, 4096], topk_ids [T, 8] with this GPU's experts as 0..31."""
    k = state["k2b"]
    bank = state["bank"]
    num_tokens = x.shape[0]
    x = x.contiguous()
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    bm = K2B_CONFIGS[cfg][2]
    sorted_ids, block_experts, ntp = _k2b_align(local_ids, bm, k["align_ignore"])
    n_mb = block_experts.numel()
    n_pairs = num_tokens * TOP_K
    inter = torch.empty((n_mb * bm, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((n_pairs, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    _k2b_launch(1, cfg, x, k["wq13"], k["sq13"], k["gs13"], inter, sorted_ids, block_experts, ntp, n_pairs)
    _k2b_launch(2, cfg, inter, k["wq2"], k["sq2"], k["gs2"], y, sorted_ids, block_experts, ntp, n_pairs)
    return fused_post_expert_norm_weighted_sum(
        y.view(num_tokens, TOP_K, D_EXPERT), topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS,
        weight_first=True, slot_expert_ids=local_ids)


def _experts_dispatch(x, topk_weights, topk_ids, state):
    k = state.get("k2b")
    if k is not None and k["plan"]:
        cfg = k["plan"].get(k2b_class(x.shape[0]))
        if cfg is not None:
            return _k2b_experts(x, topk_weights, topk_ids, state, cfg)
    return _experts(x, topk_weights, topk_ids, state)


def _k2b_check_align(device):
    """Verify moe_align_block_size(..., ignore_invalid_expert=True) drops non-local slots and keeps the
    block contract (every local pair exactly once, in a block labelled with its expert)."""
    try:
        ids, _ = k2b_synth_routing(997, 7, device)
        emap = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
        emap[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
        local = emap[ids.long()].contiguous()
        for bm in sorted({c[2] for c in K2B_CONFIGS.values()}):
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


def _k2b_graph_time(fn, flush, reps=5):
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


def _k2b_prepare(weights, state, t_start):
    """Repack, self-test and time the candidates; returns the K2b state (plan {} = Marlin everywhere).
    Deadlines are absolute from prepare()'s start (the 300 s limit covers the baseline's prepare too)."""
    t0 = time.time()
    print(f"[k2b] baseline prepare took {t0 - t_start:.1f}s", flush=True)
    dev = weights["w13"].device
    k = {"plan": {}, "align_ignore": False}
    if not K2B_ENABLE:
        print("[k2b] K2B_ENABLE=False: stock Marlin experts", flush=True)
        return k
    try:
        wq13 = torch.empty((K2B_E, K2B_K // 64, 2, K2B_N13, 16), dtype=torch.uint8, device=dev)
        sq13 = torch.empty((K2B_E, K2B_K // 64, K2B_N13), dtype=torch.int32, device=dev)
        wq2 = torch.empty((K2B_E, K2B_K // 64, 2, K2B_N2, 16), dtype=torch.uint8, device=dev)
        sq2 = torch.empty((K2B_E, K2B_K // 64, K2B_N2), dtype=torch.int32, device=dev)
        for e in range(K2B_E):
            wq13[e], sq13[e] = k2b_repack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True)
            wq2[e], sq2[e] = k2b_repack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False)
        k.update(wq13=wq13, sq13=sq13, wq2=wq2, sq2=sq2,
                 gs13=weights["w13_gs"][:K2B_E].float().contiguous(),
                 gs2=weights["w2_gs"][:K2B_E].float().contiguous())
        torch.cuda.synchronize()
        print(f"[k2b] repacked in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[k2b] repack failed, stock Marlin experts: {ex!r}", flush=True)
        return k
    ok, msg = _k2b_check_align(dev)
    k["align_ignore"] = ok
    print(f"[k2b] align ignore_invalid_expert: {'ok' if ok else 'NOT ok (' + msg + '), verbatim -1 blocks'}",
          flush=True)

    # Self-tests: FP32 reference at two sizes (normal and skewed routing), plus a Marlin cross-check.
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
    for name in K2B_CONFIGS:
        if time.time() - t_start > K2B_DEADLINE_S:
            print(f"[k2b] deadline reached ({time.time() - t_start:.0f}s), skipping {name}", flush=True)
            continue
        t1 = time.time()
        try:
            worst = 0.0
            for tokens, x, ids, wts, ref in tests:
                got = _k2b_experts(x, wts, ids, {"k2b": k, "bank": state["bank"],
                                                  "post_expert_w": state["post_expert_w"]}, name)
                torch.cuda.synchronize()
                tot, row = k2b_rel(got, ref)
                fin = bool(torch.isfinite(got.float()).all())
                print(f"[k2b] {name} T={tokens}: rel {tot:.3e} worst row {row:.3e} finite={fin}", flush=True)
                if not (fin and tot < 1.0e-2 and row < 5.0e-2):
                    raise RuntimeError(f"self-test mismatch at T={tokens}")
                worst = max(worst, tot)
            passed.append(name)
            print(f"[k2b] {name} PASS (worst rel {worst:.3e}, {time.time() - t1:.1f}s incl. compile)", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[k2b] {name} FAILED ({time.time() - t1:.1f}s): {str(ex)[-3000:]}", flush=True)
    del tests
    torch.cuda.empty_cache()
    if not passed:
        print("[k2b] no candidate passed: stock Marlin experts", flush=True)
        return k

    # Timing under CUDA graphs on busiest-GPU-like synthetic routing; choose per token-count class.
    try:
        flush = torch.empty((96 << 20,), dtype=torch.uint8, device=dev)
        bank_state = {"k2b": k, "bank": state["bank"], "post_expert_w": state["post_expert_w"]}
        for cls, _, tokens in K2B_CLASSES:
            g = torch.Generator(device=dev).manual_seed(tokens)
            x = torch.randn((tokens, D_EXPERT), generator=g, device=dev).to(torch.bfloat16)
            ids, wts = k2b_synth_routing(tokens, tokens + 1, dev)
            routes = int(((ids >= 0) & (ids < LOCAL_EXPERTS)).sum())
            flop = 2.0 * routes * 3 * D_EXPERT * D_EXPERT
            t_m = _k2b_graph_time(lambda: _experts(x, wts, ids, state), flush)
            line = [f"marlin {t_m * 1e3:.0f}us ({flop / t_m / 1e9:.0f} TF/s)"]
            best, t_best = None, t_m * K2B_MIN_GAIN
            for name in passed:
                if time.time() - t_start > K2B_TIMING_DEADLINE_S:
                    line.append(f"(deadline: {name} and later not timed)")
                    break
                try:
                    t_c = _k2b_graph_time(lambda: _k2b_experts(x, wts, ids, bank_state, name), flush)
                except Exception as ex:  # noqa: BLE001
                    line.append(f"{name} ERR {str(ex)[-200:]}")
                    continue
                line.append(f"{name} {t_c * 1e3:.0f}us ({flop / t_c / 1e9:.0f} TF/s)")
                if t_c < t_best:
                    best, t_best = name, t_c
            k["plan"][cls] = best
            print(f"[k2b] time {cls} T={tokens} routes={routes}: " + " | ".join(line) + f" -> {best or 'marlin'}",
                  flush=True)
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

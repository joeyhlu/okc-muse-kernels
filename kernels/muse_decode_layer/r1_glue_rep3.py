# okc: problem=muse_decode_layer
# nonce r1-1-rep3 (repeat of r1_glue.py for the 3-check protocol)
"""Rung 1: baseline.py (SGLang's decoder layer) with Python-level glue changes only. Every kernel
launched is one SGLang ships (R1_QK_FUSED_FP8_STAGE compiles a new variant of SGLang's own Triton
QK-norm kernel, with its FP8 KV-store epilogue enabled). Each change sits
behind a module-level R1_* constant so a failing check can be bisected by flipping one line
(see kernels/rung1_levers.md for evidence, expected savings and bisect order):

  R1_QK_FUSED_FP8_STAGE      FusedQKNorm's own Triton kernel with its FP8 KV-store epilogue writes
                             FP8 q, k, v in one launch (into staging rows 1..T, because the kernel
                             skips cache row 0, which is live in this problem); store_cache then
                             scatters k, v to the real rows. 5 launches -> 2. kv_scale == 1 only.
  R1_LOCAL_ID_TABLE          (topk_ids - local_start) % 256 -> expert_map as one add + one
                             index_select on a [256 x 256] table (3+ launches -> 2).
  R1_ALIGN_LOCAL_BUCKETS     moe_align_block_size over the 32 local experts (33 buckets): one launch
                             instead of align + sort for 8 < T < 128.
  R1_POSTNORM_TRITON_ALL_T   SGLang's Triton post-expert-norm weighted sum at every T (the stock
                             wrapper runs ~11 torch kernels below 32 tokens).
  R1_UNPOOL_INTO_ZEROS       T <= 8: the sliced unpooler GEMM writes straight into columns
                             [0, 1024) of a zeroed [T, 8192] (stock: GEMM, then F.pad = fill+copy).
  BLOCK_SIZE_M_OVERRIDE      per-T Marlin moe_block_size; empty = stock heuristic.
  NUM_SPLITS_OVERRIDE        per-shape FA3 num_splits; empty = stock heuristic.
  R1_ROUTER_SPLIT_ALL_T      OFF: BF16 hi+lo router at T <= 8 too (allowed; no expected gain, the
                             router is hidden behind the pooler on its side stream).
  R1_HOPPER_GEMV_T1          OFF: SGLang's opt-in Hopper BF16 GEMV for the T == 1 projections. It is
                             a JIT module the baseline never builds (compile time), so rung 1b/2.

Original baseline docstring follows.

SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
import triton
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

try:  # the @triton.jit kernel behind fused_post_expert_norm_weighted_sum (imported, never modified)
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        _post_expert_norm_weighted_sum_kernel as _POSTNORM_KERNEL,
    )
except ImportError:  # pragma: no cover - only if the image differs from a55ab7d5
    _POSTNORM_KERNEL = None

# ---------------------------------------------------------------- rung-1 levers (bisect here)
R1_QK_FUSED_FP8_STAGE = True
R1_LOCAL_ID_TABLE = True
R1_ALIGN_LOCAL_BUCKETS = True
# Up to 8 tokens the stock align is already one launch (Triton small-numel path); keep it there.
R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS = 9
R1_POSTNORM_TRITON_ALL_T = True
R1_UNPOOL_INTO_ZEROS = True
R1_ROUTER_SPLIT_ALL_T = False
R1_HOPPER_GEMV_T1 = False
# {T: moe_block_size}; rung-0b measurements fill this. Empty = MuseSpark1xFp4Experts' heuristic.
BLOCK_SIZE_M_OVERRIDE = {}
# {("sliding" | "full", batch, q_len): num_splits}; empty = FA3's own heuristic (stock). An entry
# also drops the judge's scheduler_metadata, which was built with num_splits=0.
NUM_SPLITS_OVERRIDE = {}
# Staging rows for the fused QK/KV write: cache_loc = 1..T (constant, built once in prepare).
STAGE_LOC_MAX = 1 << 16

print(f"[r1] layer levers: qk_fused={R1_QK_FUSED_FP8_STAGE} local_table={R1_LOCAL_ID_TABLE} "
      f"align_local={R1_ALIGN_LOCAL_BUCKETS} postnorm_triton={R1_POSTNORM_TRITON_ALL_T and _POSTNORM_KERNEL is not None} "
      f"unpool_zeros={R1_UNPOOL_INTO_ZEROS} router_split_all={R1_ROUTER_SPLIT_ALL_T} gemv_t1={R1_HOPPER_GEMV_T1} "
      f"bm_override={BLOCK_SIZE_M_OVERRIDE} splits_override={NUM_SPLITS_OVERRIDE}", flush=True)


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


_GEMV = None  # (use_hopper_bf16_gemv, hopper_bf16_gemv) once prepare() imports them (R1_HOPPER_GEMV_T1)


def _gemm(x, w):
    if _GEMV is not None and x.shape[0] == 1 and _GEMV[0](1, w.shape[0], w.shape[1]):
        # SGLang's --bf16-gemm-backend gemv path (unquant.py:434-441): bf16 in/out, fp32 accumulate.
        return _GEMV[1](x.view(1, -1), w)
    return _bf16_gemm_dispatch_impl(x, w, None)


def _local_id_table(device):
    """[256 * 256] int32: entry g + 256 * s = expert_map[(g - s) % 256] for a GPU whose experts
    start at s, i.e. (g - s) % 256 when that is < 32, else -1. Covers every s in [0, 256)."""
    s = torch.arange(N_EXPERTS, dtype=torch.int64, device=device).view(-1, 1)
    g = torch.arange(N_EXPERTS, dtype=torch.int64, device=device).view(1, -1)
    rel = (g - s) % N_EXPERTS
    return torch.where(rel < LOCAL_EXPERTS, rel, torch.full_like(rel, -1)).to(torch.int32).reshape(-1).contiguous()


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
        # Rung 1: constants only (nothing here depends on any input).
        "local_id_table": _local_id_table(device),
        "stage_loc": torch.arange(1, STAGE_LOC_MAX + 1, dtype=torch.int64, device=device),
    }
    if R1_HOPPER_GEMV_T1:
        global _GEMV
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv, use_hopper_bf16_gemv

        _GEMV = (use_hopper_bf16_gemv, hopper_bf16_gemv)
    del freqs
    torch.cuda.synchronize()
    return state


def _post_expert_norm_weighted_sum(y, topk_weights, norm_w, slot_ids):
    """fused_post_expert_norm_weighted_sum(y, w, norm_weight=norm_w, eps=1e-8, weight_first=True,
    slot_expert_ids=slot_ids), always on its Triton kernel: the same launch as the wrapper's own
    >= 32-token branch (fused_post_expert_norm_weighted_sum_triton.py:101-118)."""
    if not (R1_POSTNORM_TRITON_ALL_T and _POSTNORM_KERNEL is not None):
        return fused_post_expert_norm_weighted_sum(
            y, topk_weights, norm_weight=norm_w, eps=POST_NORM_EPS, weight_first=True,
            slot_expert_ids=slot_ids)
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = triton.next_power_of_2(hidden)
    _POSTNORM_KERNEL[(tokens,)](
        y, norm_w, topk_weights, out, slot_ids,
        TOPK=topk, H=hidden, eps=POST_NORM_EPS,
        APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True,
        BLOCK_H=block_h, num_warps=8 if block_h >= 4096 else 4, num_stages=2,
    )
    return out


def _experts(x, topk_weights, topk_ids, state, ids_are_local=False):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize.
    ids_are_local: topk_ids already went through expert_map (R1_LOCAL_ID_TABLE)."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = (topk_ids if ids_are_local else bank["expert_map"][topk_ids]).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    block_size_m = BLOCK_SIZE_M_OVERRIDE.get(num_tokens, block_size_m)
    align_experts = N_EXPERTS
    if R1_ALIGN_LOCAL_BUCKETS and num_tokens >= R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS:
        # local_ids are in [-1, 32): buckets id+1 in [0, 32] hold them all and the remote bucket
        # still maps to expert -1 (moe_align_kernel.cu: expert_ids = bucket - 1).
        align_experts = LOCAL_EXPERTS
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, align_experts)
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
    return _post_expert_norm_weighted_sum(y, topk_weights, state["post_expert_w"], local_ids)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    rope = (data["positions"], state["cos_sin"]) if cfg["rope_theta"] else (None, None)
    if R1_QK_FUSED_FP8_STAGE and not scaled:
        # FusedQKNorm's Triton kernel with its FP8 KV-store epilogue (fused_qk_norm.py:105-121):
        # q -> FP8 output, k (normed, roped, bf16-rounded) and raw v -> FP8 rows of the "cache"
        # given to it. The kernel skips cache row 0 (SGLang's padding slot), which is live here,
        # so it writes staging rows 1..T and store_cache moves them to the real rows exactly as
        # the baseline does (reserved_skip_index=-1). Values equal the baseline's bf16 -> fp8
        # casts (the kernel's +-448 clamp cannot fire: |k| <= 8 after the norm).
        width = hkv * d
        k_stage = torch.empty((tokens + 1, width), dtype=FP8, device=qkv.device)
        v_stage = torch.empty((tokens + 1, width), dtype=FP8, device=qkv.device)
        stage_loc = state["stage_loc"]
        loc = (stage_loc[:tokens] if tokens <= stage_loc.shape[0]
               else torch.arange(1, tokens + 1, dtype=torch.int64, device=qkv.device))
        q8, _ = state["qk_norm"].forward_triton(qkv, rope[0], rope[1], (k_stage, v_stage, loc))
        store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8),
                    k_cache.view(-1, width).view(torch.uint8), v_cache.view(-1, width).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q_fp8 = q8.view(tokens, hq, d)
    else:
        if cfg["rope_theta"]:
            q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
        else:
            q, k = state["qk_norm"](qkv)
        v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
        k = k.reshape(tokens, hkv * d)
        if scaled:
            k, v = k / data["k_scale"], v / data["v_scale"]
        # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
        store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                    k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q_fp8 = q.view(tokens, hq, d).to(FP8)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    sched = data["scheduler_metadata"]
    splits = NUM_SPLITS_OVERRIDE.get(("sliding" if window else "full", batch, length), 0)
    if splits:
        descale["num_splits"] = splits  # (descale doubles as the extra-kwargs dict)
        sched = None  # the judge's metadata was built for num_splits=0; let FA3 rebuild it
    out = flash_attn_with_kvcache(
        q=q_fp8,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=sched,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS or R1_ROUTER_SPLIT_ALL_T
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
    if R1_LOCAL_ID_TABLE:
        # table[g + 256 * s] = expert_map[(g - s) % 256]: one add, one int32 index_select.
        idx = torch.add(topk_ids, data["local_start"], alpha=N_EXPERTS)
        local_ids = torch.index_select(state["local_id_table"], 0, idx.reshape(-1)).view(tokens, TOP_K)
    else:
        local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state,
                             ids_are_local=R1_LOCAL_ID_TABLE)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        if R1_UNPOOL_INTO_ZEROS:
            # Same cuBLAS GEMM as F.linear (x @ w.T), written with ldc = 8192 into the zeroed
            # output; the other seven GPUs' columns stay zero (all-gather: identity).
            ffn_output = torch.zeros((tokens, HIDDEN), dtype=expert_output.dtype, device=expert_output.device)
            torch.mm(expert_output, dense["unpooler"][:UNPOOL_SLICE].t(), out=ffn_output[:, :UNPOOL_SLICE])
        else:
            unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
            ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids

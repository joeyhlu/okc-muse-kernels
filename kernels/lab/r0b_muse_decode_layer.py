# okc: problem=muse_decode_layer
# LAB FILE — check runs only, never submit for score
# rung 0b lab: SGLang's baseline.py with kernel_fn unchanged, plus printed H200 measurements (kernels/lab/README.md). nonce r0b-1
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
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


def _baseline_prepare(weights):
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
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids

# ===== LAB helpers (shared by the r0b_* lab files; copied into each, since a submission is one file) =====
import time as _lab_time


def _lab_print(*a):
    print(*a, flush=True)


def _lab_err(e, n=600):
    s = "%s: %s" % (type(e).__name__, str(e).replace("\n", " | "))
    return s if len(s) <= n else s[: n - 120] + " ... " + s[-115:]


def _lab_flush_buf(device):
    """256 MiB (> 50 MB L2). Read before every timed replay, as the judge does."""
    return torch.ones(64 << 20, dtype=torch.int32, device=device)


def _lab_graph_us(fn, flush, reps):
    """(median, min) microseconds of CUDA-graph replays of fn(), each replay preceded by a full read of
    `flush` (cold L2). fn is warmed twice on a side stream before capture, like torch's graph recipe."""
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
    g.replay()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in evs:
        flush.sum()
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) * 1e3 for a, b in evs)
    del g
    return ts[len(ts) // 2], ts[0]


def _lab_replay_floor(flush, reps, device):
    x = torch.zeros(1, device=device)
    return _lab_graph_us(lambda: x.add_(1.0), flush, reps)


_LAB_SUM = []


def _lab_sum(line):
    """Collect a compact line for the SUMMARY block printed last (the front of our output is what falls out of
    the 16,000-char tail first)."""
    _LAB_SUM.append(line)


def _lab_print_summary(tag):
    _lab_print("===== %s LAB SUMMARY (raw us; subtract the 1-kernel replay floor to compare with the judge) =====" % tag)
    for line in _LAB_SUM:
        _lab_print(line)
    _lab_print("===== END LAB SUMMARY =====")


def _lab_rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))

# ===== BEGIN microbench_cuda.py (inlined verbatim into kernels/lab/r0b_*.py) =====
# H200 microbenchmarks M1-M5 (notes/study/03-hopper-gemm-attn.md §A.8-A.9) in ONE small
# torch.utils.cpp_extension.load_inline extension. Public entry point: run_all(print_fn, parts, deadline).
#   dev : device properties (torch + cudaDeviceGetAttribute by raw enum value)
#   M1  : idle pointer-chase latency (1 thread, ld.global.cg, random cyclic permutation of 128 B slots)
#   M2  : the same chase while the other blocks stream-read a different 1 GiB region
#   M3  : streaming read bandwidth vs bytes in flight per SM (vector width x unroll x blocks/SM)
#   M4  : random-row gather bandwidth, 64 B vs 128 B rows (page_size-1 KV), index loads included
#   M5  : E2M1 -> BF16 conversion throughput (Marlin bit placement, PRMT LUT, smem LUT x2) + INT peaks
# Every kernel has fixed trip counts (no data-dependent loops except the bounded M2 flag), every
# index is in range by construction or checked. Nothing here touches the problem's harness.
import time as _mb_time

_MB_CUDA = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define MB_MAGIC 0x9E3779B9u

static inline cudaStream_t mb_cur_stream() { return at::cuda::getCurrentCUDAStream().stream(); }

__device__ __forceinline__ unsigned long long mb_gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---------------- M1: idle pointer chase (one thread) ----------------
__global__ void mb_chase_kernel(const unsigned* __restrict__ buf, unsigned start, int warm, int hops,
                                long long* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  unsigned p = start;
  for (int i = 0; i < warm; ++i) p = __ldcg(buf + p);
  const long long c0 = clock64();
  const unsigned long long t0 = mb_gtimer();
  for (int i = 0; i < hops; ++i) p = __ldcg(buf + p);
  const long long c1 = clock64();
  const unsigned long long t1 = mb_gtimer();
  out[0] = c1 - c0;
  out[1] = (long long)(t1 - t0);
  out[2] = (long long)p;
}

// ---------------- M2: chase in block 0 while blocks 1.. stream ----------------
template <int U>
__global__ void __launch_bounds__(256) mb_chase_load_kernel(
    const unsigned* __restrict__ cbuf, unsigned start, int hops,
    const uint4* __restrict__ sbuf, long long n16, int max_passes,
    int* __restrict__ flag, long long* __restrict__ out,
    unsigned long long* __restrict__ bytes, uint4* __restrict__ sink) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      unsigned p = start;
      for (int i = 0; i < 64; ++i) p = __ldcg(cbuf + p);
      const long long c0 = clock64();
      const unsigned long long t0 = mb_gtimer();
      for (int i = 0; i < hops; ++i) p = __ldcg(cbuf + p);
      const long long c1 = clock64();
      const unsigned long long t1 = mb_gtimer();
      out[0] = c1 - c0;
      out[1] = (long long)(t1 - t0);
      out[2] = (long long)p;
      __threadfence();
      atomicExch(flag, 1);
    }
    return;
  }
  const long long nthreads = (long long)(gridDim.x - 1) * blockDim.x;
  const long long tid = (long long)(blockIdx.x - 1) * blockDim.x + threadIdx.x;
  uint4 acc = make_uint4(0u, 0u, 0u, 0u);
  unsigned long long cnt = 0ull;
  for (int pass = 0; pass < max_passes; ++pass) {
    if (*((volatile int*)flag) != 0) break;
    for (long long i = tid; i + (long long)(U - 1) * nthreads < n16; i += (long long)U * nthreads) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) v[u] = __ldcs(sbuf + i + (long long)u * nthreads);
#pragma unroll
      for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
      cnt += (unsigned long long)U;
    }
  }
  atomicAdd(bytes, cnt * 16ull);
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == MB_MAGIC) sink[0] = acc;
}

// ---------------- M3: streaming read, V-wide loads, U in flight per thread ----------------
__device__ __forceinline__ void mb_zero(unsigned& a) { a = 0u; }
__device__ __forceinline__ void mb_zero(uint2& a) { a = make_uint2(0u, 0u); }
__device__ __forceinline__ void mb_zero(uint4& a) { a = make_uint4(0u, 0u, 0u, 0u); }
__device__ __forceinline__ void mb_xor(unsigned& a, unsigned b) { a ^= b; }
__device__ __forceinline__ void mb_xor(uint2& a, uint2 b) { a.x ^= b.x; a.y ^= b.y; }
__device__ __forceinline__ void mb_xor(uint4& a, uint4 b) { a.x ^= b.x; a.y ^= b.y; a.z ^= b.z; a.w ^= b.w; }
__device__ __forceinline__ unsigned mb_fold(unsigned a) { return a; }
__device__ __forceinline__ unsigned mb_fold(uint2 a) { return a.x ^ a.y; }
__device__ __forceinline__ unsigned mb_fold(uint4 a) { return a.x ^ a.y ^ a.z ^ a.w; }

template <typename V, int U>
__global__ void __launch_bounds__(256) mb_stream_kernel(const V* __restrict__ src, long long n, V* __restrict__ sink) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  V acc;
  mb_zero(acc);
  for (; i + (long long)(U - 1) * stride < n; i += (long long)U * stride) {
    V v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = __ldg(src + i + (long long)u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) mb_xor(acc, v[u]);
  }
  for (; i < n; i += stride) mb_xor(acc, __ldg(src + i));
  if (mb_fold(acc) == MB_MAGIC) sink[0] = acc;
}

// ---------------- M4: random row gather (16 B per thread, RB/16 threads per row) ----------------
template <int RB, int U>
__global__ void __launch_bounds__(256) mb_gather_kernel(const uint4* __restrict__ table, long long nrows,
                                                        const int* __restrict__ idx, long long nidx,
                                                        uint4* __restrict__ sink) {
  constexpr int TPR = RB / 16;
  const long long gtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int sub = (int)(gtid % TPR);
  const long long grp = gtid / TPR;
  const long long ngrp = ((long long)gridDim.x * blockDim.x) / TPR;
  uint4 acc = make_uint4(0u, 0u, 0u, 0u);
  for (long long r = grp; r < nidx; r += ngrp * U) {
    int rows[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long rr = r + (long long)u * ngrp;
      rows[u] = (rr < nidx) ? __ldg(idx + rr) : -1;
    }
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int row = rows[u];
      v[u] = (row >= 0 && (long long)row < nrows) ? __ldg(table + (long long)row * TPR + sub)
                                                 : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == MB_MAGIC) sink[0] = acc;
}

// ---------------- M5: E2M1 -> BF16 conversion variants ----------------
__device__ __forceinline__ unsigned mb_prmt(unsigned a, unsigned b, unsigned c) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// VAR 1: exactly sglang marlin/dequant.h:393-407 (dequant<nv_bfloat162, kFE2M1f, true>) applied the way
// marlin_moe/marlin_template.h:1262-1276 does (b_quant_1 = q, b_quant_0 = q << 8). Output BF16 = e2m1 * 2^-126.
// Slots: o0 = (n0, n4), o1 = (n1, n5), o2 = (n2, n6), o3 = (n3, n7); n_i = nibble i of q (bits 4i..4i+3).
__device__ __forceinline__ void mb_cvt_marlin(unsigned q, unsigned& o0, unsigned& o1, unsigned& o2, unsigned& o3) {
  unsigned x = q << 8;
  const unsigned a1 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x <<= 4;
  const unsigned a2 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x = q;
  const unsigned b1 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x <<= 4;
  const unsigned b2 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  o0 = a2; o1 = a1; o2 = b2; o3 = b1;
}

// VAR 2: PRMT byte-LUT. Layout (a lossless offline repack): nibble i bits 0..2 = |v_i| code, nibble i bit 3 =
// sign of v_{s(i)} with s = {0,2,4,6,1,3,5,7}. Output o_j = (v_2j, v_2j+1), BF16 = e2m1 exactly (unscaled).
__device__ __forceinline__ void mb_cvt_prmt(unsigned q, unsigned& o0, unsigned& o1, unsigned& o2, unsigned& o3) {
  const unsigned HA = 0x3F3F3F00u, HB = 0x40404040u;  // high bytes of bf16(0, .5, 1, 1.5) | (2, 3, 4, 6)
  const unsigned LA = 0xC0800000u, LB = 0xC0804000u;  // low bytes
  const unsigned m = q & 0x77777777u;
  const unsigned mh = m >> 16;
  const unsigned hl = mb_prmt(HA, HB, m), ll = mb_prmt(LA, LB, m);
  const unsigned hh = mb_prmt(HA, HB, mh), lh = mb_prmt(LA, LB, mh);
  const unsigned p0 = mb_prmt(ll, hl, 0x5140u);
  const unsigned p1 = mb_prmt(ll, hl, 0x7362u);
  const unsigned p2 = mb_prmt(lh, hh, 0x5140u);
  const unsigned p3 = mb_prmt(lh, hh, 0x7362u);
  o0 = p0 | ((q << 12) & 0x80008000u);
  o1 = p1 | ((q << 8) & 0x80008000u);
  o2 = p2 | ((q << 4) & 0x80008000u);
  o3 = p3 | (q & 0x80008000u);
}

__device__ __forceinline__ unsigned mb_e2m1_bits(unsigned nib) {
  const unsigned mag = nib & 7u;
  const unsigned b = (mag == 0u) ? 0u : ((mag == 1u) ? 0x3F00u : (0x3F80u + (mag - 2u) * 0x40u));
  return b | ((nib & 8u) << 12);
}
__device__ __forceinline__ unsigned mb_lut_entry(unsigned byte) {
  return mb_e2m1_bits(byte & 15u) | (mb_e2m1_bits((byte >> 4) & 15u) << 16);
}

template <int VAR>
__device__ __forceinline__ void mb_fill_lut(unsigned* lut) {
  if constexpr (VAR == 3) {
    for (int e = threadIdx.x; e < 256; e += blockDim.x) lut[e] = mb_lut_entry((unsigned)e);
  }
  if constexpr (VAR == 4) {
    for (int e = threadIdx.x; e < 256 * 32; e += blockDim.x) lut[e] = mb_lut_entry((unsigned)(e >> 5));
  }
  __syncthreads();
}

// VAR 3: 256-entry smem LUT (one bf16x2 per byte, natural order o_k = (n_2k, n_2k+1)); random banks.
// VAR 4: the same LUT replicated per lane (entry*32 + lane): conflict-free, 32 KiB per block.
template <int VAR>
__device__ __forceinline__ void mb_cvt_any(unsigned q, const unsigned* lut, unsigned lane,
                                           unsigned& o0, unsigned& o1, unsigned& o2, unsigned& o3) {
  if constexpr (VAR == 1) {
    mb_cvt_marlin(q, o0, o1, o2, o3);
  } else if constexpr (VAR == 2) {
    mb_cvt_prmt(q, o0, o1, o2, o3);
  } else if constexpr (VAR == 3) {
    o0 = lut[q & 0xFFu];
    o1 = lut[(q >> 8) & 0xFFu];
    o2 = lut[(q >> 16) & 0xFFu];
    o3 = lut[q >> 24];
  } else if constexpr (VAR == 4) {
    o0 = lut[((q & 0xFFu) << 5) | lane];
    o1 = lut[(((q >> 8) & 0xFFu) << 5) | lane];
    o2 = lut[(((q >> 16) & 0xFFu) << 5) | lane];
    o3 = lut[((q >> 24) << 5) | lane];
  } else {
    const unsigned t = q & 0x3FFF3FFFu;  // VAR 0: harness overhead only
    o0 = t; o1 = t; o2 = t; o3 = t;
  }
}

__device__ __forceinline__ __nv_bfloat162 mb_u2bf(unsigned u) { return *reinterpret_cast<const __nv_bfloat162*>(&u); }
__device__ __forceinline__ unsigned mb_bf2u(__nv_bfloat162 b) { return *reinterpret_cast<const unsigned*>(&b); }

template <int VAR>
__global__ void __launch_bounds__(256) mb_cvt_bench_kernel(int iters, unsigned seed, uint4* __restrict__ sink,
                                                           long long* __restrict__ tinfo) {
  extern __shared__ unsigned mb_lut[];
  mb_fill_lut<VAR>(mb_lut);
  const unsigned lane = threadIdx.x & 31u;
  unsigned q = seed ^ ((blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u);
  unsigned r = q * 2246822519u + 3266489917u;
  __nv_bfloat162 a0 = __float2bfloat162_rn(0.0f);
  __nv_bfloat162 a1 = a0, a2 = a0, a3 = a0;
  const __nv_bfloat162 kk = __float2bfloat162_rn(1.0f);
  long long c0 = 0;
  unsigned long long t0 = 0ull;
  if (threadIdx.x == 0) { c0 = clock64(); t0 = mb_gtimer(); }
#pragma unroll 2
  for (int it = 0; it < iters; ++it) {
    q = q * 1664525u + 1013904223u;
    r = r * 22695477u + 1u;
    unsigned o0, o1, o2, o3, p0, p1, p2, p3;
    mb_cvt_any<VAR>(q, mb_lut, lane, o0, o1, o2, o3);
    mb_cvt_any<VAR>(r, mb_lut, lane, p0, p1, p2, p3);
    a0 = __hfma2(mb_u2bf(o0), kk, a0);
    a1 = __hfma2(mb_u2bf(o1), kk, a1);
    a2 = __hfma2(mb_u2bf(o2), kk, a2);
    a3 = __hfma2(mb_u2bf(o3), kk, a3);
    a0 = __hfma2(mb_u2bf(p0), kk, a0);
    a1 = __hfma2(mb_u2bf(p1), kk, a1);
    a2 = __hfma2(mb_u2bf(p2), kk, a2);
    a3 = __hfma2(mb_u2bf(p3), kk, a3);
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    tinfo[0] = clock64() - c0;
    tinfo[1] = (long long)(mb_gtimer() - t0);
  }
  const unsigned f = mb_bf2u(a0) ^ mb_bf2u(a1) ^ mb_bf2u(a2) ^ mb_bf2u(a3);
  if (f == MB_MAGIC) sink[0] = make_uint4(f, 0u, 0u, 0u);
}

template <int VAR>
__global__ void __launch_bounds__(256) mb_cvt_check_kernel(const unsigned* __restrict__ words, int n,
                                                           uint4* __restrict__ out) {
  extern __shared__ unsigned mb_lut[];
  mb_fill_lut<VAR>(mb_lut);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned o0, o1, o2, o3;
  mb_cvt_any<VAR>(words[i], mb_lut, threadIdx.x & 31u, o0, o1, o2, o3);
  out[i] = make_uint4(o0, o1, o2, o3);
}

// INT pipe peaks: 8 independent chains of LOP3 (KIND 0) or PRMT (KIND 1), 32 ops per thread per iteration.
template <int KIND>
__global__ void __launch_bounds__(256) mb_intpeak_kernel(int iters, unsigned seed, unsigned* __restrict__ sink) {
  unsigned a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u;
  unsigned a4 = a0 * 11u, a5 = a0 * 13u, a6 = a0 * 17u, a7 = a0 * 19u;
  const unsigned b = seed * 2654435761u, c = (seed >> 3) | 0x5140u;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if constexpr (KIND == 0) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a0) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a1) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a2) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a3) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a4) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a5) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a6) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a7) : "r"(b), "r"(c));
      } else {
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a0) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a1) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a2) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a3) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a4) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a5) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a6) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a7) : "r"(b), "r"(c));
      }
    }
  }
  const unsigned f = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
  if (f == MB_MAGIC) sink[0] = f;
}

// ---------------- host wrappers (return 0 on success, cudaError code otherwise, -2 bad config) ----------------
std::vector<int64_t> mb_dev_attrs() {
  int dev = 0;
  cudaGetDevice(&dev);
  // raw cudaDeviceAttr values (stable ABI, avoids enum names removed in newer toolkits):
  // 13 clockRate kHz, 16 SMs, 36 memClockRate kHz, 37 bus width bits, 38 L2 bytes, 39 max threads/SM,
  // 81 max smem/SM, 82 max regs/SM, 8 smem/block, 97 smem/block opt-in
  const int ids[10] = {13, 16, 36, 37, 38, 39, 81, 82, 8, 97};
  std::vector<int64_t> r;
  for (int k = 0; k < 10; ++k) {
    int v = -1;
    if (cudaDeviceGetAttribute(&v, (cudaDeviceAttr)ids[k], dev) != cudaSuccess) {
      v = -1;
      cudaGetLastError();
    }
    r.push_back((int64_t)v);
  }
  return r;
}

int64_t mb_chase(torch::Tensor buf, int64_t start, int64_t warm, int64_t hops, torch::Tensor out) {
  mb_chase_kernel<<<1, 32, 0, mb_cur_stream()>>>((const unsigned*)buf.data_ptr(), (unsigned)start, (int)warm,
                                                 (int)hops, (long long*)out.data_ptr());
  return (int64_t)cudaGetLastError();
}

template <int U>
static int64_t mb_chase_load_launch(const unsigned* cb, unsigned start, int hops, const uint4* sb, long long n16,
                                    int max_passes, int sblocks, int* fl, long long* o, unsigned long long* by,
                                    uint4* sk) {
  mb_chase_load_kernel<U><<<1 + sblocks, 256, 0, mb_cur_stream()>>>(cb, start, hops, sb, n16, max_passes, fl, o, by, sk);
  return (int64_t)cudaGetLastError();
}

int64_t mb_chase_load(torch::Tensor cbuf, int64_t start, int64_t hops, torch::Tensor sbuf, int64_t sblocks,
                      int64_t unroll, int64_t max_passes, torch::Tensor flag, torch::Tensor out,
                      torch::Tensor bytes, torch::Tensor sink) {
  const unsigned* cb = (const unsigned*)cbuf.data_ptr();
  const uint4* sb = (const uint4*)sbuf.data_ptr();
  const long long n16 = (long long)(sbuf.numel() * sbuf.element_size()) / 16;
  int* fl = (int*)flag.data_ptr();
  long long* o = (long long*)out.data_ptr();
  unsigned long long* by = (unsigned long long*)bytes.data_ptr();
  uint4* sk = (uint4*)sink.data_ptr();
  const unsigned st = (unsigned)start;
  const int h = (int)hops, mp = (int)max_passes, nb = (int)sblocks;
  switch (unroll) {
    case 1: return mb_chase_load_launch<1>(cb, st, h, sb, n16, mp, nb, fl, o, by, sk);
    case 2: return mb_chase_load_launch<2>(cb, st, h, sb, n16, mp, nb, fl, o, by, sk);
    case 4: return mb_chase_load_launch<4>(cb, st, h, sb, n16, mp, nb, fl, o, by, sk);
    case 8: return mb_chase_load_launch<8>(cb, st, h, sb, n16, mp, nb, fl, o, by, sk);
    default: return -2;
  }
}

template <typename V, int U>
static int64_t mb_stream_launch(const void* src, long long nbytes, int blocks, void* sink, bool occ_only) {
  if (occ_only) {
    int nb = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mb_stream_kernel<V, U>, 256, 0) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    return (int64_t)nb;
  }
  const long long n = nbytes / (long long)sizeof(V);
  mb_stream_kernel<V, U><<<blocks, 256, 0, mb_cur_stream()>>>((const V*)src, n, (V*)sink);
  return (int64_t)cudaGetLastError();
}

int64_t mb_stream_bw(torch::Tensor buf, int64_t vec_bytes, int64_t unroll, int64_t blocks, torch::Tensor sink,
                     int64_t occ_only) {
  const void* src = buf.data_ptr();
  const long long nbytes = (long long)(buf.numel() * buf.element_size());
  void* sk = sink.data_ptr();
  const int b = (int)blocks;
  const bool oo = occ_only != 0;
#define MB_SCASE(VB, VT, UU) \
  if (vec_bytes == VB && unroll == UU) return mb_stream_launch<VT, UU>(src, nbytes, b, sk, oo);
  MB_SCASE(4, unsigned, 1) MB_SCASE(4, unsigned, 2) MB_SCASE(4, unsigned, 4) MB_SCASE(4, unsigned, 8)
  MB_SCASE(8, uint2, 1) MB_SCASE(8, uint2, 2) MB_SCASE(8, uint2, 4) MB_SCASE(8, uint2, 8)
  MB_SCASE(16, uint4, 1) MB_SCASE(16, uint4, 2) MB_SCASE(16, uint4, 4) MB_SCASE(16, uint4, 8)
#undef MB_SCASE
  return -2;
}

template <int RB, int U>
static int64_t mb_gather_launch(const uint4* t, long long nrows, const int* ix, long long nidx, int blocks, uint4* sk) {
  mb_gather_kernel<RB, U><<<blocks, 256, 0, mb_cur_stream()>>>(t, nrows, ix, nidx, sk);
  return (int64_t)cudaGetLastError();
}

int64_t mb_gather(torch::Tensor table, torch::Tensor idx, int64_t row_bytes, int64_t unroll, int64_t blocks,
                  torch::Tensor sink) {
  if (row_bytes != 64 && row_bytes != 128) return -2;
  const uint4* t = (const uint4*)table.data_ptr();
  const long long nrows = (long long)(table.numel() * table.element_size()) / (long long)row_bytes;
  const int* ix = (const int*)idx.data_ptr();
  const long long nidx = (long long)idx.numel();
  uint4* sk = (uint4*)sink.data_ptr();
  const int b = (int)blocks;
#define MB_GCASE(RB, UU) \
  if (row_bytes == RB && unroll == UU) return mb_gather_launch<RB, UU>(t, nrows, ix, nidx, b, sk);
  MB_GCASE(64, 1) MB_GCASE(64, 2) MB_GCASE(64, 4) MB_GCASE(64, 8)
  MB_GCASE(128, 1) MB_GCASE(128, 2) MB_GCASE(128, 4) MB_GCASE(128, 8)
#undef MB_GCASE
  return -2;
}

static size_t mb_cvt_smem(int var) { return var == 3 ? (size_t)(256 * 4) : (var == 4 ? (size_t)(256 * 32 * 4) : (size_t)0); }

template <int VAR>
static int64_t mb_cvt_launch(int blocks, int iters, unsigned seed, uint4* sk, long long* ti, bool occ_only) {
  const size_t sm = mb_cvt_smem(VAR);
  if (occ_only) {
    int nb = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mb_cvt_bench_kernel<VAR>, 256, sm) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    return (int64_t)nb;
  }
  mb_cvt_bench_kernel<VAR><<<blocks, 256, sm, mb_cur_stream()>>>(iters, seed, sk, ti);
  return (int64_t)cudaGetLastError();
}

int64_t mb_cvt(int64_t var, int64_t blocks, int64_t iters, int64_t seed, torch::Tensor sink, torch::Tensor tinfo,
               int64_t occ_only) {
  uint4* sk = (uint4*)sink.data_ptr();
  long long* ti = (long long*)tinfo.data_ptr();
  const int b = (int)blocks, it = (int)iters;
  const unsigned sd = (unsigned)seed;
  const bool oo = occ_only != 0;
  switch (var) {
    case 0: return mb_cvt_launch<0>(b, it, sd, sk, ti, oo);
    case 1: return mb_cvt_launch<1>(b, it, sd, sk, ti, oo);
    case 2: return mb_cvt_launch<2>(b, it, sd, sk, ti, oo);
    case 3: return mb_cvt_launch<3>(b, it, sd, sk, ti, oo);
    case 4: return mb_cvt_launch<4>(b, it, sd, sk, ti, oo);
    default: return -2;
  }
}

template <int VAR>
static int64_t mb_cvt_check_launch(const unsigned* w, int n, uint4* o) {
  const int blocks = (n + 255) / 256;
  mb_cvt_check_kernel<VAR><<<blocks, 256, mb_cvt_smem(VAR), mb_cur_stream()>>>(w, n, o);
  return (int64_t)cudaGetLastError();
}

int64_t mb_cvt_check(int64_t var, torch::Tensor words, torch::Tensor out) {
  const unsigned* w = (const unsigned*)words.data_ptr();
  const int n = (int)words.numel();
  uint4* o = (uint4*)out.data_ptr();
  if (n <= 0) return -2;
  switch (var) {
    case 0: return mb_cvt_check_launch<0>(w, n, o);
    case 1: return mb_cvt_check_launch<1>(w, n, o);
    case 2: return mb_cvt_check_launch<2>(w, n, o);
    case 3: return mb_cvt_check_launch<3>(w, n, o);
    case 4: return mb_cvt_check_launch<4>(w, n, o);
    default: return -2;
  }
}

template <int KIND>
static int64_t mb_intpeak_launch(int blocks, int iters, unsigned seed, unsigned* sk, bool occ_only) {
  if (occ_only) {
    int nb = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mb_intpeak_kernel<KIND>, 256, 0) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    return (int64_t)nb;
  }
  mb_intpeak_kernel<KIND><<<blocks, 256, 0, mb_cur_stream()>>>(iters, seed, sk);
  return (int64_t)cudaGetLastError();
}

int64_t mb_intpeak(int64_t kind, int64_t blocks, int64_t iters, int64_t seed, torch::Tensor sink, int64_t occ_only) {
  unsigned* sk = (unsigned*)sink.data_ptr();
  const bool oo = occ_only != 0;
  if (kind == 0) return mb_intpeak_launch<0>((int)blocks, (int)iters, (unsigned)seed, sk, oo);
  if (kind == 1) return mb_intpeak_launch<1>((int)blocks, (int)iters, (unsigned)seed, sk, oo);
  return -2;
}
"""

_MB_CPP = r"""
#include <vector>
std::vector<int64_t> mb_dev_attrs();
int64_t mb_chase(torch::Tensor buf, int64_t start, int64_t warm, int64_t hops, torch::Tensor out);
int64_t mb_chase_load(torch::Tensor cbuf, int64_t start, int64_t hops, torch::Tensor sbuf, int64_t sblocks,
                      int64_t unroll, int64_t max_passes, torch::Tensor flag, torch::Tensor out,
                      torch::Tensor bytes, torch::Tensor sink);
int64_t mb_stream_bw(torch::Tensor buf, int64_t vec_bytes, int64_t unroll, int64_t blocks, torch::Tensor sink,
                     int64_t occ_only);
int64_t mb_gather(torch::Tensor table, torch::Tensor idx, int64_t row_bytes, int64_t unroll, int64_t blocks,
                  torch::Tensor sink);
int64_t mb_cvt(int64_t var, int64_t blocks, int64_t iters, int64_t seed, torch::Tensor sink, torch::Tensor tinfo,
               int64_t occ_only);
int64_t mb_cvt_check(int64_t var, torch::Tensor words, torch::Tensor out);
int64_t mb_intpeak(int64_t kind, int64_t blocks, int64_t iters, int64_t seed, torch::Tensor sink, int64_t occ_only);
"""

_MB_FUNCS = ["mb_dev_attrs", "mb_chase", "mb_chase_load", "mb_stream_bw", "mb_gather", "mb_cvt", "mb_cvt_check",
             "mb_intpeak"]
_MB_STATE = {"mod": None}
_MB_GIB = 1 << 30
_MB_MIB = 1 << 20


def _mb_build(P):
    """Compile the extension once. No arch flags of our own: torch adds -gencode for the visible device
    (sm_90) when TORCH_CUDA_ARCH_LIST is unset; the arch flags it will use are printed first."""
    if _MB_STATE["mod"] is not None:
        return _MB_STATE["mod"]
    from torch.utils import cpp_extension as _ce
    try:
        P("MB arch flags: %s" % " ".join(_ce._get_cuda_arch_flags()))
    except Exception as e:
        P("MB arch flags n/a (%s)" % type(e).__name__)
    t = _mb_time.time()
    mod = _ce.load_inline(name="okc_lab_mb_v1", cpp_sources=[_MB_CPP], cuda_sources=[_MB_CUDA],
                          functions=_MB_FUNCS, extra_cflags=["-O2"], extra_cuda_cflags=["-O3"], verbose=False)
    P("MB compile %.1f s" % (_mb_time.time() - t))
    _MB_STATE["mod"] = mod
    return mod


def _mb_ok(rc, what):
    if rc != 0:
        raise RuntimeError("%s returned %d" % (what, rc))


def _mb_err(e, n=1500):
    s = "%s: %s" % (type(e).__name__, e)
    return s if len(s) <= n else s[:300] + " ... " + s[-(n - 300):]


def _mb_dev(P, torch, info):
    dev = torch.cuda.current_device()
    pr = torch.cuda.get_device_properties(dev)
    sms = int(pr.multi_processor_count)
    info["sms"] = sms
    extra = []
    for k in ("L2_cache_size", "clock_rate", "memory_clock_rate", "memory_bus_width", "regs_per_multiprocessor",
              "max_threads_per_multi_processor", "shared_memory_per_multiprocessor"):
        v = getattr(pr, k, None)
        if v is not None:
            extra.append("%s=%s" % (k, v))
    P("DEV %s cc%d.%d SMs=%d mem=%.1fGiB torch=%s cuda=%s %s" % (pr.name, pr.major, pr.minor, sms,
                                                                pr.total_memory / _MB_GIB, torch.__version__,
                                                                torch.version.cuda, " ".join(extra)))
    try:
        P("DEV nvml clock_rate MHz=%s" % torch.cuda.clock_rate(dev))
    except Exception as e:  # pynvml may be absent
        P("DEV nvml n/a (%s)" % type(e).__name__)
    return dev


def _mb_dev_attrs(P, mod):
    a = mod.mb_dev_attrs()
    names = ["clk_kHz", "SMs", "memclk_kHz", "bus_bits", "L2_B", "thr/SM", "smem/SM", "regs/SM", "smem/blk",
             "smem/blk_optin"]
    P("DEV attrs " + " ".join("%s=%d" % (n, v) for n, v in zip(names, a)))
    if a[2] > 0 and a[3] > 0:
        P("DEV nominal HBM BW = %.0f GB/s (2 x memclk x bus/8)" % (2.0 * a[2] * 1e3 * a[3] / 8 / 1e9))
    return a


def _mb_make_chase(torch, buf, ws_bytes, slot_bytes=128):
    """Random single-cycle permutation over the 128 B slots of buf[:ws_bytes]; every element of the
    region holds a valid element index (zeros elsewhere), so the chase can never leave the region."""
    se = slot_bytes // 4
    S = ws_bytes // slot_bytes
    region = buf[: S * se]
    region.zero_()
    perm = torch.randperm(S, device=buf.device)
    nxt = torch.empty_like(perm)
    nxt[perm] = torch.roll(perm, -1)
    region.view(S, se)[:, 0].copy_((nxt * se).to(torch.int32))
    start = int(perm[0].item()) * se
    del perm, nxt
    return start, S


def _mb_m1(P, torch, mod, buf, info):
    out = torch.zeros(4, dtype=torch.int64, device=buf.device)
    total = buf.numel() * 4
    rows = []
    for ws in (4 * _MB_MIB, 24 * _MB_MIB, 256 * _MB_MIB, _MB_GIB, 2 * _MB_GIB):
        if ws > total:
            continue
        start, S = _mb_make_chase(torch, buf, ws)
        hops = 20000
        warm = min(S, 300000) if ws <= 64 * _MB_MIB else 2000
        _mb_ok(mod.mb_chase(buf, start, warm, hops, out), "mb_chase")
        torch.cuda.synchronize()
        cyc, ns = int(out[0].item()), int(out[1].item())
        lat_ns, lat_cyc = ns / hops, cyc / hops
        ghz = cyc / max(ns, 1)
        rows.append((ws, lat_ns, lat_cyc, ghz))
        P("M1 ws=%6.0fMiB lat=%7.1f ns %7.1f cyc  (SM clk %.3f GHz, %d hops)" % (ws / _MB_MIB, lat_ns, lat_cyc, ghz,
                                                                               hops))
    info["m1"] = rows
    dram = [r for r in rows if r[0] >= 256 * _MB_MIB]
    if dram:
        info["lat_ns"] = min(r[1] for r in dram)
        info["clk_mem_ghz"] = dram[0][3]
    return rows


def _mb_m2(P, torch, mod, buf, info):
    sms = info["sms"]
    half = buf.numel() // 2
    cbuf, sbuf = buf[:half], buf[half:]
    start, _ = _mb_make_chase(torch, cbuf, half * 4)
    out = torch.zeros(4, dtype=torch.int64, device=buf.device)
    flag = torch.zeros(1, dtype=torch.int32, device=buf.device)
    nbytes = torch.zeros(1, dtype=torch.int64, device=buf.device)
    sink = torch.zeros(64, dtype=torch.int32, device=buf.device)
    hops = 8192
    res = []
    for bps, u in ((0, 1), (1, 1), (1, 2), (1, 4), (2, 4), (4, 4)):
        flag.zero_()
        nbytes.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        _mb_ok(mod.mb_chase_load(cbuf, start, hops, sbuf, bps * sms, u, 400, flag, out, nbytes, sink), "mb_chase_load")
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        cyc, ns = int(out[0].item()), int(out[1].item())
        gbs = int(nbytes.item()) / (ms * 1e-3) / 1e9 if bps else 0.0
        infl = bps * 256 * 16 * u
        lat = ns / hops
        res.append((infl, gbs, lat))
        need = gbs * 1e9 * lat * 1e-9 / sms / 1024 if bps else 0.0
        P("M2 stream %2dKiB/SM in flight: %6.0f GB/s  chase lat=%7.1f ns %6.1f cyc  -> BWxlat/SM=%5.1f KiB" % (
            infl // 1024, gbs, lat, cyc / hops, need))
    info["m2"] = res
    return res


def _mb_m3(P, torch, mod, buf, info):
    sms = info["sms"]
    sbuf = buf[: _MB_GIB // 4]
    nbytes = sbuf.numel() * 4
    sink = torch.zeros(64, dtype=torch.int32, device=buf.device)
    res = []
    for vb in (4, 8, 16):
        for u in (1, 2, 4, 8):
            occ = mod.mb_stream_bw(sbuf, vb, u, 0, sink, 1)
            cells = []
            for bps in (1, 2, 4, 8):
                eff = min(bps, occ) if occ > 0 else bps
                blocks = eff * sms
                _mb_ok(mod.mb_stream_bw(sbuf, vb, u, blocks, sink, 0), "mb_stream_bw warm")
                best = 1e30
                for _ in range(3):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    _mb_ok(mod.mb_stream_bw(sbuf, vb, u, blocks, sink, 0), "mb_stream_bw")
                    e1.record()
                    torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1))
                gbs = nbytes / (best * 1e-3) / 1e9
                infl = eff * 256 * vb * u
                res.append((vb, u, eff, infl, gbs))
                cells.append("b%d%s %5.0f(%3dK)" % (bps, "" if eff == bps else ">%d" % eff, gbs, infl // 1024))
            P("M3 v%-2d u%d occ%d | " % (vb, u, occ) + " | ".join(cells))
    mx = max(r[4] for r in res)
    info["bw_max"] = mx
    P("M3 max %.0f GB/s (%.0f GB/s/SM)" % (mx, mx / sms))
    info["m3_knee"] = {}
    for frac in (0.8, 0.9, 0.95):
        ok = [r for r in res if r[4] >= frac * mx]
        r = min(ok, key=lambda x: x[3])
        info["m3_knee"][frac] = (r[3] / 1024, r[3] * sms / r[4])
        P("M3 knee: >=%2.0f%% of max with %5.1f KiB/SM in flight (v%d u%d b%d) -> implied lat %.0f ns" % (
            frac * 100, r[3] / 1024, r[0], r[1], r[2], r[3] * sms / (r[4] * 1e9) * 1e9))
    lat = info.get("lat_ns")
    if lat:
        for tgt in (4.2e12, 0.9 * mx * 1e9):
            P("M3 Little: idle lat %.0f ns x %.2f TB/s / %d SMs = %.1f KiB/SM needed" % (
                lat, tgt / 1e12, sms, tgt * lat * 1e-9 / sms / 1024))
    return res


def _mb_m4(P, torch, mod, buf, info):
    sms = info["sms"]
    table = buf[: _MB_GIB // 4]
    sink = torch.zeros(64, dtype=torch.int32, device=buf.device)
    blocks = 4 * sms
    for rb in (64, 128):
        nrows = _MB_GIB // rb
        nidx = (256 * _MB_MIB) // rb
        idx = torch.randperm(nrows, device=buf.device)[:nidx].to(torch.int32)
        seq = torch.arange(nidx, device=buf.device, dtype=torch.int32)
        cells = []
        for name, ix, us in (("rand", idx, (1, 2, 4, 8)), ("seq", seq, (4,))):
            for u in us:
                _mb_ok(mod.mb_gather(table, ix, rb, u, blocks, sink), "mb_gather warm")
                best = 1e30
                for _ in range(3):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    _mb_ok(mod.mb_gather(table, ix, rb, u, blocks, sink), "mb_gather")
                    e1.record()
                    torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1))
                gbs = nidx * rb / (best * 1e-3) / 1e9
                m4 = info.setdefault("m4", {})
                m4[(rb, name)] = max(gbs, m4.get((rb, name), 0.0))
                cells.append("%s u%d %5.0f GB/s %4.0f Mrow/s" % (name, u, gbs, nidx / (best * 1e-3) / 1e6))
        P("M4 row%3dB (%d thr/row, 1024 thr/SM, +4 B idx/row) | " % (rb, rb // 16) + " | ".join(cells))
        del idx, seq
    if "bw_max" in info:
        P("M4 compare: M3 streaming max %.0f GB/s" % info["bw_max"])


def _mb_expected(torch, var, w64):
    """Expected BF16 bit patterns (int64 [n, 4]) for variant var, from words w64 (int64 in [0, 2^32))."""
    dev = w64.device
    t8 = torch.tensor([0, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0], dtype=torch.int64, device=dev)
    nib = [(w64 >> (4 * i)) & 15 for i in range(8)]

    def bits(n):
        return t8[n & 7] | ((n & 8) << 12)

    if var == 1:
        def mb(n):
            return ((n & 8) << 12) | ((n & 7) << 6)
        cols = [mb(nib[a]) | (mb(nib[b]) << 16) for a, b in ((0, 4), (1, 5), (2, 6), (3, 7))]
    elif var == 2:
        slot = (0, 4, 1, 5, 2, 6, 3, 7)  # sign of v_i lives in nibble slot[i]

        def val(i):
            return bits((nib[i] & 7) | (nib[slot[i]] & 8))
        cols = [val(2 * j) | (val(2 * j + 1) << 16) for j in range(4)]
    else:
        cols = [bits(nib[2 * j]) | (bits(nib[2 * j + 1]) << 16) for j in range(4)]
    return torch.stack(cols, 1)


def _mb_m5(P, torch, mod, info):
    sms = info["sms"]
    dev = torch.cuda.current_device()
    sink = torch.zeros(64, dtype=torch.int32, device=dev)
    tinfo = torch.zeros(4, dtype=torch.int64, device=dev)
    # correctness of each conversion on random words + every byte pattern
    n = 1 << 14
    w64 = torch.randint(0, 1 << 32, (n,), dtype=torch.int64, device=dev)
    w64[:256] = torch.arange(256, device=dev) * 0x01010101
    w32 = torch.where(w64 >= (1 << 31), w64 - (1 << 32), w64).to(torch.int32)
    names = {0: "overhead", 1: "marlin-lop3", 2: "prmt-lut", 3: "smem-lut256", 4: "smem-lut-lane"}
    status = {}
    for var in (1, 2, 3, 4):
        out = torch.zeros((n, 4), dtype=torch.int32, device=dev)
        _mb_ok(mod.mb_cvt_check(var, w32, out), "mb_cvt_check")
        torch.cuda.synchronize()
        got = out.to(torch.int64) & 0xFFFFFFFF
        bad = int((got != _mb_expected(torch, var, w64)).any(1).sum().item())
        status[var] = "ok" if bad == 0 else "MISMATCH %d/%d" % (bad, n)
    iters = 4096
    for var in (0, 1, 2, 3, 4):
        occ = mod.mb_cvt(var, 0, 0, 0, sink, tinfo, 1)
        blocks = max(occ, 1) * sms
        _mb_ok(mod.mb_cvt(var, blocks, 64, 1234, sink, tinfo, 0), "mb_cvt warm")
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        _mb_ok(mod.mb_cvt(var, blocks, iters, 99991, sink, tinfo, 0), "mb_cvt")
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        ghz = int(tinfo[0].item()) / max(int(tinfo[1].item()), 1)
        info["clk_alu_ghz"] = ghz
        vals = blocks * 256.0 * iters * 16
        vpc = vals / (ms * 1e6 * ghz) / sms
        info.setdefault("m5", {})[var] = vpc
        info.setdefault("m5_check", {})[var] = status.get(var, "-")
        P("M5 %-13s %6.1f values/clk/SM  (occ %d blk/SM, %.2f ms, clk %.3f GHz) check=%s" % (
            names[var], vpc, occ, ms, ghz, status.get(var, "-")))
    for kind, nm in ((0, "LOP3"), (1, "PRMT")):
        occ = mod.mb_intpeak(kind, 0, 0, 0, sink, 1)
        blocks = max(occ, 1) * sms
        _mb_ok(mod.mb_intpeak(kind, blocks, 64, 7, sink, 0), "mb_intpeak warm")
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        _mb_ok(mod.mb_intpeak(kind, blocks, iters, 7, sink, 0), "mb_intpeak")
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        ghz = info.get("clk_alu_ghz", 1.755)
        ops = blocks * 256.0 * iters * 32
        info.setdefault("m5_peak", {})[nm] = ops / (ms * 1e6 * ghz) / sms
        P("M5 peak %s %6.1f ops/clk/SM (at %.3f GHz from the cvt kernels; %.2f ms)" % (nm, ops / (ms * 1e6 * ghz) / sms,
                                                                                       ghz, ms))
    ghz = info.get("clk_alu_ghz", 1.755)
    for bw in (4.2e12, 4.8e12):
        need = bw / 0.5625 / sms / (ghz * 1e9)
        P("M5 need at %.1f TB/s of NVFP4 (0.5625 B/value): %.1f values/clk/SM at %.3f GHz" % (bw / 1e12, need, ghz))


def mb_summary_lines(info):
    """A few compact lines with the decision-relevant microbench numbers (printed last by the lab files)."""
    out = []
    if info.get("m1"):
        out.append("SUM M1 lat ns/cyc: " + " ".join("%dMiB=%.0f/%.0f" % (r[0] // _MB_MIB, r[1], r[2])
                                                   for r in info["m1"]))
    if info.get("m2"):
        out.append("SUM M2 (KiB/SM in flight: GB/s, lat ns): " + " ".join("%d:%.0f,%.0f" % (r[0] // 1024, r[1], r[2])
                                                                          for r in info["m2"]))
    if info.get("bw_max"):
        k = info.get("m3_knee", {})
        out.append("SUM M3 max %.0f GB/s; KiB/SM for 80/90/95%%: %s; idle lat %s ns" % (
            info["bw_max"], "/".join("%.1f" % k[f][0] for f in (0.8, 0.9, 0.95) if f in k),
            "%.0f" % info["lat_ns"] if info.get("lat_ns") else "n/a"))
    if info.get("m4"):
        out.append("SUM M4 best GB/s: " + " ".join("%s%dB=%.0f" % (n, rb, v) for (rb, n), v in sorted(info["m4"].items())))
    if info.get("m5"):
        nm = {0: "ovh", 1: "marlin", 2: "prmt", 3: "lut256", 4: "lutlane"}
        out.append("SUM M5 values/clk/SM: " + " ".join("%s=%.1f(%s)" % (nm[v], x, info.get("m5_check", {}).get(v, "-"))
                                                      for v, x in sorted(info["m5"].items()))
                   + " | peaks " + " ".join("%s=%.1f" % kv for kv in sorted(info.get("m5_peak", {}).items()))
                   + " | clk %.3f GHz" % info.get("clk_alu_ghz", 0.0))
    return out


def run_all(print_fn=print, parts=("dev", "M1", "M2", "M3", "M4", "M5"), deadline=None):
    """Run the selected microbenchmarks; each part is guarded so one failure never stops the rest.
    deadline: absolute time.time() after which remaining parts are skipped."""
    import torch
    P = print_fn
    t0 = _mb_time.time()
    info = {}
    try:
        _mb_dev(P, torch, info)
    except Exception as e:
        P("DEV ERR " + _mb_err(e))
        info.setdefault("sms", 132)
    try:
        mod = _mb_build(P)
    except Exception as e:
        P("MB BUILD FAILED " + _mb_err(e, 6000))
        return info
    try:
        _mb_dev_attrs(P, mod)
    except Exception as e:
        P("DEV attrs ERR " + _mb_err(e))
    need_big = any(p in parts for p in ("M1", "M2", "M4"))
    buf = None
    try:
        buf = torch.empty((2 if need_big else 1) * _MB_GIB // 4, dtype=torch.int32, device="cuda")
        buf.zero_()
    except Exception as e:
        P("MB alloc ERR " + _mb_err(e))
    steps = (("M1", _mb_m1, True), ("M2", _mb_m2, True), ("M3", _mb_m3, True), ("M4", _mb_m4, True),
             ("M5", None, False))
    for name, fn, needs_buf in steps:
        if name not in parts:
            continue
        if deadline is not None and _mb_time.time() > deadline:
            P("%s skipped (deadline)" % name)
            continue
        if needs_buf and buf is None:
            P("%s skipped (no buffer)" % name)
            continue
        t = _mb_time.time()
        try:
            if name == "M5":
                _mb_m5(P, torch, mod, info)
            else:
                fn(P, torch, mod, buf, info)
        except Exception as e:
            P("%s ERR %s" % (name, _mb_err(e)))
        try:
            torch.cuda.synchronize()
        except Exception as e:
            P("%s sync ERR %s" % (name, _mb_err(e)))
        P("%s done in %.1f s" % (name, _mb_time.time() - t))
    del buf
    torch.cuda.empty_cache()
    P("MB total %.1f s" % (_mb_time.time() - t0))
    return info
# ===== END microbench_cuda.py =====

# ----- FA3 num_splits sweep on the scored attention shapes (page_size 1, ctx 80K) -----
import math as _lab_math

LAB_CTX = 80000
LAB_SPLITS = (1, 2, 4, 8, 16)
LAB_SM_SCALE = 64 ** -0.5 * 5.4730064863875  # sm_scale * qk_scale_factor, as the baseline passes it
LAB_AGENTIC = 0.075


def _lab_longest(ctx, spread):
    if spread == "fixed":
        return ctx
    if spread == "agentic":
        return _lab_math.ceil(ctx * (1 + LAB_AGENTIC))
    return 4 * ctx


def _lab_lengths(gen, batch, ctx, spread, device):
    if spread == "fixed":
        return torch.full((batch,), ctx, dtype=torch.int64, device=device)
    if spread == "agentic":
        u = torch.rand((batch,), generator=gen, device=device)
        return (ctx * (1 + LAB_AGENTIC * (2 * u - 1))).long().clamp(1, _lab_longest(ctx, spread))
    z = torch.randn((batch,), generator=gen, device=device)
    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()


def _lab_shared_cache(device):
    """One FP8 K and V pool big enough for every scored shape (batch 32 x (86000 + 8) rows of [2, 64]).
    Values are N(0,1) cast to E4M3: timing does not depend on them."""
    rows = 32 * (_lab_longest(LAB_CTX, "agentic") + 8)
    kc = torch.empty((rows, 1, 2, 64), dtype=torch.float8_e4m3fn, device=device)
    vc = torch.empty((rows, 1, 2, 64), dtype=torch.float8_e4m3fn, device=device)
    for dst in (kc, vc):
        flat = dst.view(-1, 2, 64)
        for at in range(0, rows, 1 << 20):
            n = min(1 << 20, rows - at)
            flat[at:at + n] = torch.randn((n, 2, 64), device=device).to(torch.float8_e4m3fn)
    return kc, vc


def _lab_attn_case(batch, ctx, layer, seed, q_len, spread, device, page=1):
    """The step metadata of the task's generate_input (copied, not imported): lengths, scattered page table,
    positions, FA3 lengths and, on full-attention decode, the scheduler metadata. q is a unit-RMS random
    query cast to E4M3 (what the baseline hands FA3); the KV pool comes from _lab_shared_cache."""
    from sgl_kernel.flash_attn import get_scheduler_metadata
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = batch * q_len
    seq_lens = _lab_lengths(gen, batch, ctx, spread, device)
    pages_per = (seq_lens + q_len + page - 1) // page
    longest = _lab_longest(ctx, spread)
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    meta = None
    if layer == "full" and q_len == 1:
        meta = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=16, num_heads_k=2,
            headdim=64, cache_seqlens=cache_seqlens, qkv_dtype=torch.float8_e4m3fn, cu_seqlens_q=cu_seqlens_q,
            page_size=page, causal=True, num_splits=0)
    qf = torch.randn((tokens, 16, 64), device=device)
    q8 = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + 1e-5)).to(torch.float8_e4m3fn)
    return {"q8": q8, "pt": page_table, "cs": cache_seqlens, "cuq": cu_seqlens_q, "meta": meta, "L": q_len,
            "win": (2047, 0) if layer == "sliding" else (-1, -1), "batch": batch, "total": total,
            "seq_lens": seq_lens.to(torch.int32), "positions": positions, "cache_loc": cache_loc}


def _lab_fa3(c, kc, vc, num_splits, meta):
    out = flash_attn_with_kvcache(
        q=c["q8"], k_cache=kc, v_cache=vc, page_table=c["pt"], cache_seqlens=c["cs"], cu_seqlens_q=c["cuq"],
        max_seqlen_q=c["L"], softmax_scale=LAB_SM_SCALE, causal=True, window_size=c["win"],
        scheduler_metadata=meta, num_splits=num_splits)
    return out[0] if isinstance(out, tuple) else out


def _lab_fa3_sweep(shapes, reps, t_begin, deadline_s, tag):
    """shapes: (label, batch, q_len, layer, spread, seed). Per shape: 'base' = exactly the baseline's call
    (num_splits=0 + the given scheduler_metadata), 's0nm' = num_splits=0 without metadata (full decode only),
    'sN' = num_splits=N without metadata (FA3 builds a consistent schedule itself; forcing N together with
    metadata built for N=0 is NOT tried: its dynamic splits could exceed the N-sized accumulators)."""
    device = torch.device("cuda", torch.cuda.current_device())
    flush = _lab_flush_buf(device)
    floor = _lab_replay_floor(flush, reps, device)
    _lab_print("%s 1-kernel graph replay: median %.1f us, min %.1f us" % (tag, floor[0], floor[1]))
    _lab_sum("SUM %s replay floor %.1f us" % (tag, floor[0]))
    t = _lab_time.time()
    kc, vc = _lab_shared_cache(device)
    torch.cuda.synchronize()
    _lab_print("%s KV pool %d rows x 2 heads x 64 B (K and V), built in %.1f s" % (tag, kc.shape[0], _lab_time.time() - t))
    _lab_print("%s FA3 sweep: median us of %d graph replays, cold L2, raw; ! = rel-L2 vs base > 1e-2" % (tag, reps))
    for label, batch, q_len, layer, spread, seed in shapes:
        if _lab_time.time() - t_begin > deadline_s:
            _lab_print("%s %s skipped (deadline)" % (tag, label))
            continue
        try:
            c = _lab_attn_case(batch, LAB_CTX, layer, seed, q_len, spread, device)
            ref = _lab_fa3(c, kc, vc, 0, c["meta"]).float()
            torch.cuda.synchronize()
            configs = [("base", 0, c["meta"])]
            if c["meta"] is not None:
                configs.append(("s0nm", 0, None))
            configs += [("s%d" % n, n, None) for n in LAB_SPLITS]
            cells, best, base_us = [], None, None
            for name, n, meta in configs:
                try:
                    o = _lab_fa3(c, kc, vc, n, meta).float()
                    torch.cuda.synchronize()
                    rel = _lab_rel(o, ref)
                    us = _lab_graph_us(lambda n=n, meta=meta: _lab_fa3(c, kc, vc, n, meta), flush, reps)[0]
                    cells.append("%s %6.1f%s" % (name, us, "!" if rel > 1e-2 else ""))
                    if rel <= 1e-2 and (best is None or us < best[0]):
                        best = (us, name)
                    if name == "base":
                        base_us = us
                except Exception as e:
                    cells.append("%s ERR" % name)
                    _lab_print("%s %s %s ERR %s" % (tag, label, name, _lab_err(e)))
            extra = ""
            if c["meta"] is not None:
                extra = " | meta[:4]=%s" % c["meta"][:4].tolist()
            gain = (" best %s %.2fx" % (best[1], base_us / best[0])) if (best and base_us) else ""
            _lab_print("%s %-11s | %s |%s%s" % (tag, label, " ".join(cells), gain, extra))
            if best and base_us:
                _lab_sum("SUM %s %-11s base %6.1f | best %-4s %6.1f | %.2fx%s" % (
                    tag, label, base_us, best[1], best[0], base_us / best[0], extra))
            else:
                _lab_sum("SUM %s %-11s no valid result" % (tag, label))
        except Exception as e:
            _lab_print("%s %s ERR %s" % (tag, label, _lab_err(e, 1200)))
    del kc, vc, flush
    torch.cuda.empty_cache()

# ===== LAB (muse_decode_layer): prepare() = the baseline's prepare + printed measurements =====
# kernel_fn above is the baseline's, unchanged; prepare() returns exactly the baseline's state.
LAB_ENABLED = True
LAB_PARTS = ("fa3", "pen", "router", "micro")
LAB_DEADLINE_S = 230.0  # start no new lab work after this many seconds of prepare() (the limit is 300 s)
LAB_MICRO_START_BY_S = 160.0  # M3 + M5 (~45-90 s incl. compile) start only before this
LAB_MICRO_DEADLINE_S = 265.0  # ...and their parts (~5 s) may still start until this, after the compile
LAB_REPS = 15
LAB_ATTN_SHAPES = (  # (label, batch, q_len, layer, spread, seed) as in meta.json; ctx 80000, page 1
    ("dec1 slide", 1, 1, "sliding", "fixed", 801), ("dec1 full", 1, 1, "full", "fixed", 802),
    ("dec16 slide", 16, 1, "sliding", "agentic", 803), ("dec16 full", 16, 1, "full", "agentic", 804),
    ("dec32 slide", 32, 1, "sliding", "agentic", 805), ("dec32 full", 32, 1, "full", "agentic", 806),
    ("ver1 slide", 1, 8, "sliding", "fixed", 807), ("ver1 full", 1, 8, "full", "fixed", 808),
    ("ver16 slide", 16, 8, "sliding", "agentic", 809), ("ver16 full", 16, 8, "full", "agentic", 810),
    ("ver32 slide", 32, 8, "sliding", "agentic", 811), ("ver32 full", 32, 8, "full", "agentic", 812),
)


def _lab_pen(state):
    """Post-expert norm + weighted sum at T < 32: the stock torch chain (what fused_post_expert_norm_weighted_sum
    runs below 32 tokens) vs calling its Triton kernel directly (what it runs at >= 32), against an FP64 chain."""
    from sglang.kernels.ops.muse_spark_v12 import fused_post_expert_norm_weighted_sum_triton as pen_mod
    device = state["post_expert_w"].device
    w = state["post_expert_w"]
    flush = _lab_flush_buf(device)
    gen = torch.Generator(device=device).manual_seed(77)
    _lab_print("PEN post-expert norm, T<32: torch chain (stock) vs Triton kernel; err vs FP64 = rel-L2 / max-abs")
    for T in (1, 8, 16):
        scale = torch.exp(2 * torch.randn((T, 8, 1), generator=gen, device=device))
        y = (torch.randn((T, 8, D_EXPERT), generator=gen, device=device) * scale).to(torch.bfloat16)
        s = torch.rand((T, 8), generator=gen, device=device)
        s = (s / s.sum(-1, keepdim=True)).float().contiguous()
        local = torch.rand((T, 8), generator=gen, device=device) < 0.4
        ids = torch.where(local, torch.randint(0, 32, (T, 8), generator=gen, device=device),
                          torch.full((T, 8), -1, device=device, dtype=torch.int64)).to(torch.int32).contiguous()

        def chain():
            return pen_mod.fused_post_expert_norm_weighted_sum(
                y, s, norm_weight=w, eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=ids)

        def triton_path():
            out = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=device)
            pen_mod._post_expert_norm_weighted_sum_kernel[(T,)](
                y, w, s, out, ids, TOPK=8, H=D_EXPERT, eps=POST_NORM_EPS, APPLY_NORM=True, HAS_WEIGHT=True,
                WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=triton_next_pow2(D_EXPERT), num_warps=8, num_stages=2)
            return out

        f = y.double() * w.double()
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        f = torch.where((ids >= 0).unsqueeze(-1), f, torch.zeros((), dtype=f.dtype, device=device))
        ref = (f * s.double().unsqueeze(-1)).sum(1)
        row = []
        for name, fn in (("chain", chain), ("triton", triton_path)):
            try:
                o = fn()
                torch.cuda.synchronize()
                rel = float((o.double() - ref).norm() / ref.norm().clamp_min(1e-30))
                mx = float((o.double() - ref).abs().max())
                us = _lab_graph_us(fn, flush, LAB_REPS)[0]
                row.append("%s %6.1f us err %.1e/%.1e" % (name, us, rel, mx))
            except Exception as e:
                row.append("%s ERR %s" % (name, _lab_err(e, 300)))
        try:
            d = _lab_rel(triton_path(), chain())
            row.append("triton-vs-chain %.1e" % d)
        except Exception as e:
            row.append("cmp ERR %s" % _lab_err(e, 200))
        _lab_print("PEN T=%-2d | %s" % (T, " | ".join(row)))
        _lab_sum("SUM PEN T=%-2d %s" % (T, " | ".join(r[:60] for r in row)))
    del flush


def triton_next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _lab_router(state):
    """Router logits at T <= 8: stock FP32 F.linear vs the BF16 hi+lo pair with FP32 accumulation (allowed at
    any T). Errors vs an FP64 product of the FP32 input; 'hl/bf16in' = vs FP64 of the BF16-rounded input."""
    device = state["router"].device
    flush = _lab_flush_buf(device)
    gen = torch.Generator(device=device).manual_seed(78)
    _lab_print("ROUTER allow_tf32=%s" % torch.backends.cuda.matmul.allow_tf32)
    for T in (1, 8):
        x32 = torch.randn((T, HIDDEN), generator=gen, device=device)
        xb = x32.to(torch.bfloat16)

        def fp32():
            return F.linear(x32, state["router"])

        def hilo():
            lg = torch.mm(xb, state["router_hi_t"], out_dtype=torch.float32)
            lg += torch.mm(xb, state["router_lo_t"], out_dtype=torch.float32)
            return lg

        ref = x32.double() @ state["router"].double().t()
        refb = xb.double() @ state["router"].double().t()
        row = []
        for name, fn in (("fp32", fp32), ("hilo", hilo)):
            try:
                o = fn()
                torch.cuda.synchronize()
                rel = float((o.double() - ref).norm() / ref.norm())
                relb = float((o.double() - refb).norm() / refb.norm())
                us = _lab_graph_us(fn, flush, LAB_REPS)[0]
                row.append("%s %6.1f us err %.1e (bf16in %.1e)" % (name, us, rel, relb))
            except Exception as e:
                row.append("%s ERR %s" % (name, _lab_err(e, 300)))
        _lab_print("ROUTER T=%d | %s" % (T, " | ".join(row)))
        _lab_sum("SUM ROUTER T=%d %s" % (T, " | ".join(r[:60] for r in row)))
    del flush


def prepare(weights):
    t_begin = _lab_time.time()
    state = _baseline_prepare(weights)
    if not LAB_ENABLED:
        return state
    try:
        _lab_print("LAB muse_decode_layer r0b: baseline prepare took %.1f s" % (_lab_time.time() - t_begin))
        for part in LAB_PARTS:
            elapsed = _lab_time.time() - t_begin
            if elapsed > LAB_DEADLINE_S or (part == "micro" and elapsed > LAB_MICRO_START_BY_S):
                _lab_print("LAB %s skipped at %.0f s (deadline)" % (part, elapsed))
                continue
            try:
                if part == "fa3":
                    _lab_fa3_sweep(LAB_ATTN_SHAPES, LAB_REPS, t_begin, LAB_DEADLINE_S, "FA3L")
                elif part == "pen":
                    _lab_pen(state)
                elif part == "router":
                    _lab_router(state)
                elif part == "micro":
                    info = run_all(_lab_print, parts=("dev", "M3", "M5"), deadline=t_begin + LAB_MICRO_DEADLINE_S)
                    for line in mb_summary_lines(info):
                        _lab_sum(line)
            except Exception as e:
                _lab_print("LAB %s ERR %s" % (part, _lab_err(e, 2000)))
            _lab_print("LAB %s finished at %.1f s" % (part, _lab_time.time() - t_begin))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as e:
        _lab_print("LAB ERR " + _lab_err(e, 2000))
    _lab_sum("SUM LAB prepare total %.1f s" % (_lab_time.time() - t_begin))
    _lab_print_summary("muse_decode_layer")
    return state

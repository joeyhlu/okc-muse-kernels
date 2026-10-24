# okc: problem=muse_moe_w4a16
# nonce r1-v3-rep2 (r1_glue + Marlin block_size_m 16 at >=49 tokens; review kernels/review_scored_delta_1.md)
"""Rung 1: SGLang's own NVFP4 expert path (baseline.py, MuseSpark1xFp4Experts._forward_marlin at
a55ab7d5) with Python-level glue changes only. Every kernel launched here is one SGLang ships;
two are not launched by the baseline (moe_align_block_size's small-batch path, index_select).
No new compiled code. Each change is behind a module-level constant
(R1_*) so a failing check can be bisected by flipping one line. See kernels/rung1_levers.md.

  R1_POSTNORM_TRITON_ALL_T   launch SGLang's Triton post-expert-norm weighted-sum kernel at every
                             token count. The stock wrapper only uses it at >= 32 tokens and runs a
                             ~11-kernel torch chain below that (bs 1, 8, 16 here).
  R1_EXPERT_MAP_INDEX_SELECT expert_map lookup as one index_select (int32 index, no int64 cast).
  R1_ALIGN_LOCAL_BUCKETS     moe_align_block_size over the 32 local experts instead of 256 global
                             ids: 33 buckets <= 64 lets the CUDA align take its single-launch
                             small-batch path for 8 < M < 128 (one launch instead of align + sort).
  BLOCK_SIZE_M_OVERRIDE      per-M Marlin moe_block_size table; empty = stock heuristic.

Numerics are those of the baseline: same Marlin GEMMs, same silu_and_mul, the post-expert norm is
the kernel SGLang itself runs at >= 32 tokens (FP32 math, BF16 out, dead slots skipped).

Differs from SGLang: the expert all-reduce is not run (as in baseline.py)."""

from typing import TypedDict

import types

import torch
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

try:  # the @triton.jit kernel behind fused_post_expert_norm_weighted_sum (imported, never modified)
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        _post_expert_norm_weighted_sum_kernel as _POSTNORM_KERNEL,
    )
except ImportError:  # pragma: no cover - only if the image differs from a55ab7d5
    _POSTNORM_KERNEL = None

# ---------------------------------------------------------------- rung-1 levers (bisect here)
R1_POSTNORM_TRITON_ALL_T = True
R1_EXPERT_MAP_INDEX_SELECT = True
R1_ALIGN_LOCAL_BUCKETS = True
# Below this many tokens the stock call already takes a single launch (Triton small-numel path for
# numel <= 64); keep it there and only switch to 32 buckets where it removes a launch.
R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS = 9
# {M: moe_block_size}; rung-0b measurements fill this. Empty = MuseSpark1xFp4Experts' heuristic.
BLOCK_SIZE_M_OVERRIDE = {}
# Marlin moe_block_size: 16 from 49 tokens up (stock picks 48/64 there). Judge-measured vs the same glue with
# stock block sizes (jobs 4305bd8494ea450d, 96a32200948f4dc6): bs128 1.003x -> 1.366x, bs256 1.002x -> 1.292x.
# Below 49 tokens the stock pick is kept: the lab's small-T preferences (8 vs 16) did not reproduce on the judge.
# A range, not exact shapes, so nearby token counts get the same choice; outputs differ from stock by <5e-4 rel-L2.
BLOCK_SIZE_M_RANGES = ((49, 1 << 30, 16),)


def _bm_override(num_tokens, default):
    if num_tokens in BLOCK_SIZE_M_OVERRIDE:
        return BLOCK_SIZE_M_OVERRIDE[num_tokens]
    for lo, hi, bm in BLOCK_SIZE_M_RANGES:
        if lo <= num_tokens <= hi:
            return bm
    return default

print(f"[r1] moe levers: postnorm_triton={R1_POSTNORM_TRITON_ALL_T and _POSTNORM_KERNEL is not None} "
      f"index_select={R1_EXPERT_MAP_INDEX_SELECT} align_local={R1_ALIGN_LOCAL_BUCKETS} "
      f"bm_ranges={BLOCK_SIZE_M_RANGES}", flush=True)


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
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
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
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
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def _post_expert_norm_weighted_sum(y, topk_weights, norm_w, slot_ids):
    """fused_post_expert_norm_weighted_sum(y, w, norm_weight=norm_w, eps=1e-8, weight_first=True,
    slot_expert_ids=slot_ids), always on its Triton kernel. Same launch as the wrapper's own
    >= 32-token branch (fused_post_expert_norm_weighted_sum_triton.py:101-118)."""
    if not (R1_POSTNORM_TRITON_ALL_T and _POSTNORM_KERNEL is not None):
        return fused_post_expert_norm_weighted_sum(
            y, topk_weights, norm_weight=norm_w, eps=POST_NORM_EPS, weight_first=True,
            slot_expert_ids=slot_ids)
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = triton.next_power_of_2(hidden)
    _POSTNORM_KERNEL[(tokens,)](
        y,                       # contiguous [M, 8, H] bf16 view of Marlin's output
        norm_w,                  # bf16 [H], 1 + w
        topk_weights,            # fp32 [M, 8], contiguous
        out,
        slot_ids,                # int32 [M, 8], -1 = remote slot: skipped, never read
        TOPK=topk,
        H=hidden,
        eps=POST_NORM_EPS,
        APPLY_NORM=True,
        HAS_WEIGHT=True,
        WEIGHT_FIRST=True,
        HAS_IDS=True,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


def kernel_fn(data, state):
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
            # One gather kernel; index_select takes the int32 ids as they are.
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
            # local_ids are in [-1, count): buckets id+1 in [0, count] cover them all; the remote
            # bucket still comes out as expert -1 (moe_align_kernel.cu, expert_ids = bucket - 1).
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

"""Lab-only NVFP4/BF16 Humming adapter; no benchmark harness.

prepare(weights, tokens, mode="default"), run(data, state) -> BF16 [T,D].
Modes: default = indexed with the installed library's unchanged heuristic;
fullk = the same indexed heuristic with only use_stream_k=False;
grouped = grouped_masked with the installed library's unchanged heuristic.
Four explicit v0.1.12 probes: wg8, wg16, wg16_grouped, wg8_k128 (see PRESETS).
All force WGMMA before weight repacking. All keep BF16 activations,
FP32 accumulation, group-16 E4M3 scales, and FP32 expert global scales.

Targets installed humming-kernels==0.1.12.
No harness, model imports, source-path injection, or activation quantization.
Humming is imported normally from the installed environment, lazily in prepare.
prepare repacks the raw E2M1 bytes and E4M3 group-16 scales, compiles one
configuration per projection, and allocates buffers. Warm run once before graph
capture to compile the Triton helpers. One state is for one T and one stream.

0.1.12 scheduling audit: tune/sm90.py:get_config1 enables Stream-K unless
batch-invariant execution is requested. Non-indexed GEMMs also enable warp
specialization, TMA, and mbarriers. LayerConfig defaults to WGMMA on SM90 for
both layouts; the generic profiler symbol MmaOpClass proves neither MMA type.
Each projection exposes actual selected-kernel metadata in kernel_metadata,
including mma_type, generated PTX instruction, tiles, and thread counts.
scheduler.cuh:86-114 spreads the last wave over K iterations with
Stream-K; fullk assigns complete M/N tiles. Disabling it reduces tail work
distribution, not the configured number of resident CTAs. The old fullk result
alone therefore does not measure the default indexed or grouped paths.

The old heuristic simulates T*8 routes uniformly among LOCAL experts; it has
no remote-route mass. calc_num_block_list minimizes M-block count, not padded
row work. The K64 probes fix M=8/16 with one K partition, removing the
heuristic's 2/4-way K reduction and reducing its 8/16 math warps to four.
wg8_k128 keeps two K partitions/eight math warps to test pipeline granularity.
Smaller M may reload hot-expert weights more often; these are unmeasured
configurations, not claimed speedups. No input counts are copied to the host.
Current upstream's eight-row BF16 MMA lowering is absent from 0.1.12: the old
generator emits unsupported mma.sync.m8n8k8.bf16. Do not copy that preset back.

Grouped contract: GROUPED_MASKED passes device counts [E] as expert_layout.
Its scheduler sums ceil(count[e]/BM) and addresses expert e at e*capacity, so
unused capacity adds no GEMM M tiles. Fixed buffers are [E,T,...], flattened
only at the GEMM API. This uses the model's distinct top-8 expert IDs per token
(so every expert count is <= T). Glue scatters original assignment slots into
expert*T+rank and touches only local routes, not the capacity tail. No tail is
zero-filled. Empty-local replay schedules zero GEMM tiles.

The earlier GROUPED_CONTIGUOUS implementation assigned remote capacity to the
last expert: pinned scheduler.cuh:145 uses input.shape[0] rather than the final
offset. Merely passing local_total as the final device offset cannot fix it.
That dummy-tail implementation is not definitive grouped-performance evidence
and is no longer selected by this adapter.

Global-scale audit (v0.1.12): WeightScale2Type.TENSOR means one FP32 multiplier
per expert, NOT a reciprocal. epilogue/pipeline.cuh:83-88 loads bs2[expert_id];
arith/epilogue_arith.cuh:51-75 compensates internal power-of-two exponent offsets
and multiplies FP32 accumulators before BF16 output conversion. Do not fold gs
into E4M3 scales, convert it to BF16, invert it, or apply it a second time.
This arithmetic can differ from a prototype that rounds (code*sf*gs) to BF16
before MMA; numerical agreement must be measured by the caller's harness.

Primary sources:
https://github.com/vllm-project/humming/blob/v0.1.12/humming/ops/__init__.py
https://github.com/vllm-project/humming/blob/v0.1.12/humming/transform.py
https://github.com/vllm-project/humming/blob/v0.1.12/humming/schema/humming.py
https://github.com/vllm-project/humming/blob/v0.1.12/humming/include/humming/arith/epilogue_arith.cuh
https://github.com/vllm-project/humming/blob/v0.1.12/humming/tune/sm90.py
https://github.com/vllm-project/humming/blob/v0.1.12/humming/include/humming/scheduler.cuh

This is not a submission: audit the installed launcher's allocations, compile
time, and source-checker compatibility before promoting it. No TMA descriptors
or raw device pointers are constructed or cached by this adapter in prepare.
"""

import dataclasses
import json
import re
from importlib.metadata import PackageNotFoundError, version

import torch
import triton as tr
import triton.language as tl


# Exactly four source-justified probes, shared by W13 and W2. These are complete
# tuning choices, not overrides layered over potentially incompatible defaults.
# SM90 WGMMA uses four warps along N: 128/32. K=warpK avoids intra-CTA K reduction.
# wg8_k128 follows the tile/residency in current test_nvfp4_a16_uses_narrow_first_wave_only.
PRESETS = {
    "wg8": {"mma_type": "wgmma", "gemm_type": "indexed", "m": 8, "warp_n": 32,
            "k": 64, "warp_k": 64, "num_ctas_per_sm": 2, "use_warp_spec": False},
    "wg16": {"mma_type": "wgmma", "gemm_type": "indexed", "m": 16, "warp_n": 32,
             "k": 64, "warp_k": 64, "num_ctas_per_sm": 2, "use_warp_spec": False},
    "wg16_grouped": {"mma_type": "wgmma", "gemm_type": "grouped_masked", "m": 16,
                     "k": 64, "warp_n": 32, "warp_k": 64, "num_ctas_per_sm": 2,
                     "use_warp_spec": True},
    "wg8_k128": {"mma_type": "wgmma", "gemm_type": "indexed", "m": 8, "warp_n": 32,
                 "k": 128, "warp_k": 64, "num_ctas_per_sm": 3, "use_warp_spec": False},
}


def _preset_tuning(preset):
    specialized = preset["use_warp_spec"]
    return {
        "block_shape": (preset["m"], 128, preset["k"]),
        "warp_shape": (preset["m"], preset["warp_n"], preset["warp_k"]),
        "num_stages": 4, "num_ctas_per_sm": preset["num_ctas_per_sm"],
        "use_stream_k": True, "use_warp_spec": specialized,
        "use_tma": specialized, "use_mbarrier": specialized,
    }


def _kernel_metadata(kernel, layer_config):
    # Inspect the actual instantiated object, not its generic CUDA wrapper name.
    ptx = kernel.mma_op_class.generate_ptx()
    instruction = re.search(r"(?:wgmma\.mma_async|mma\.sync)[\w.:]*", ptx)
    result = {
        "layer_mma_type": json.loads(layer_config.to_str()).get("mma_type"),
        "mma_type": kernel.mma_type.value,
        "mma_op_impl": type(kernel.mma_op_class).__name__,
        "mma_shape": tuple(kernel.mma_op_class.shape),
        "ptx_instruction": instruction.group(0) if instruction else None,
        "block_shape": tuple(kernel.block_shape), "warp_shape": tuple(kernel.warp_shape),
        "estimated_smem_bytes": int(kernel.estimated_smem_size),
    }
    for name in ("num_math_threads", "num_load_threads", "num_threads", "num_stages",
                 "num_ctas_per_sm", "use_stream_k", "use_warp_spec", "use_tma",
                 "use_tma_a", "use_tma_b", "use_tma_c", "use_mbarrier"):
        result[name] = getattr(kernel, name)
    return result


@tr.jit
def _count_routes(Ids, Counts, SLOTS: tl.constexpr, R: tl.constexpr):
    expert = tl.program_id(0)
    slot = tl.arange(0, R)
    ids = tl.load(Ids + slot, slot < SLOTS, other=-1)
    count = tl.sum(((slot < SLOTS) & (ids == expert)).to(tl.int32), 0)
    tl.store(Counts + expert, count)


@tr.jit
def _pack_routes(Ids, Counts, Sorted, Experts, Padded,
                 SLOTS: tl.constexpr, E: tl.constexpr, BM: tl.constexpr,
                 R: tl.constexpr, ER: tl.constexpr):
    expert = tl.program_id(0)
    es = tl.arange(0, ER)
    counts = tl.load(Counts + es, es < E, other=0)
    lengths = tl.cdiv(counts, BM) * BM
    start = tl.sum(tl.where(es < expert, lengths, 0), 0)
    count = tl.sum(tl.where(es == expert, counts, 0), 0)
    padded = tl.cdiv(count, BM) * BM
    if expert == 0:
        tl.store(Padded, tl.sum(lengths, 0))

    slot = tl.arange(0, R)
    ids = tl.load(Ids + slot, slot < SLOTS, other=-1)
    hit = (slot < SLOTS) & (ids == expert)
    rank = tl.cumsum(hit.to(tl.int32), 0) - 1
    tl.store(Sorted + start + rank, slot, hit)
    # Disjoint from real destinations [start, start+count): no scatter/fill race.
    tl.store(Sorted + start + slot, SLOTS, (slot >= count) & (slot < padded))
    tl.store(Experts + start // BM + slot, expert, slot < padded // BM)


@tr.jit
def _pack_grouped_routes(Ids, Reverse, SLOTS: tl.constexpr,
                         CAPACITY: tl.constexpr, R: tl.constexpr):
    expert = tl.program_id(0)
    slot = tl.arange(0, R)
    ids = tl.load(Ids + slot, slot < SLOTS, other=-1)
    hit = (slot < SLOTS) & (ids == expert)
    rank = tl.cumsum(hit.to(tl.int32), 0) - 1
    tl.store(Reverse + slot, expert * CAPACITY + rank, hit)


@tr.jit
def _gather_grouped_input(X, Ids, Reverse, Grouped,
                          D: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0)
    n = tl.arange(0, B)
    expert = tl.load(Ids + slot)
    valid = (expert >= 0) & (expert < E)
    row = tl.load(Reverse + slot, valid, other=0)
    value = tl.load(X + (slot // 8) * D + n, valid & (n < D), other=0)
    tl.store(Grouped + row * D + n, value, valid & (n < D))


@tr.jit
def _swiglu(H, Ids, Inter, D: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0)
    n = tl.arange(0, B)
    expert = tl.load(Ids + slot)
    valid = (expert >= 0) & (expert < E)
    gate = tl.load(H + slot * (2 * D) + n, valid & (n < D), other=0).to(tl.float32)
    up = tl.load(H + slot * (2 * D) + D + n, valid & (n < D), other=0).to(tl.float32)
    inter = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(Inter + slot * D + n, inter.to(tl.bfloat16), n < D)


@tr.jit
def _swiglu_grouped(H, Ids, Reverse, Inter,
                     D: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0)
    n = tl.arange(0, B)
    expert = tl.load(Ids + slot)
    valid = (expert >= 0) & (expert < E)
    row = tl.load(Reverse + slot, valid, other=0)
    gate = tl.load(H + row * (2 * D) + n, valid & (n < D), other=0).to(tl.float32)
    up = tl.load(H + row * (2 * D) + D + n, valid & (n < D), other=0).to(tl.float32)
    inter = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(Inter + row * D + n, inter.to(tl.bfloat16), valid & (n < D))


@tr.jit
def _norm_slots(Y, NW, Ids, Z, D: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0)
    n = tl.arange(0, B)
    expert = tl.load(Ids + slot)
    valid = (expert >= 0) & (expert < E)
    y = tl.load(Y + slot * D + n, valid & (n < D), other=0).to(tl.float32)
    nw = tl.load(NW + n, n < D, other=0).to(tl.float32)
    y = y * (1.0 + nw)
    scale = tl.rsqrt(tl.sum(y * y, 0) / D + 1.0e-8)
    tl.store(Z + slot * D + n, y * scale, n < D)


@tr.jit
def _norm_grouped_slots(Y, NW, Ids, Reverse, Z,
                         D: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0)
    n = tl.arange(0, B)
    expert = tl.load(Ids + slot)
    valid = (expert >= 0) & (expert < E)
    row = tl.load(Reverse + slot, valid, other=0)
    y = tl.load(Y + row * D + n, valid & (n < D), other=0).to(tl.float32)
    nw = tl.load(NW + n, n < D, other=0).to(tl.float32)
    y = y * (1.0 + nw)
    scale = tl.rsqrt(tl.sum(y * y, 0) / D + 1.0e-8)
    tl.store(Z + slot * D + n, y * scale, n < D)


@tr.jit
def _combine_slots(Z, Weights, Out, D: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0.0, tl.float32)
    for slot in tl.static_range(8):
        weight = tl.load(Weights + token * 8 + slot).to(tl.float32)
        z = tl.load(Z + (token * 8 + slot) * D + n, n < D, other=0)
        acc += weight * z
    tl.store(Out + token * D + n, acc.to(tl.bfloat16), n < D)


def _package_version():
    try:
        return version("humming-kernels")
    except PackageNotFoundError:
        return "unversioned installed/source package"


def _prepare_projection(packed, scales, globals_, tokens, mode):
    from humming import dtypes, ops
    from humming.config import MmaType, WeightScale2Type
    from humming.kernel.humming import HummingKernel
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.transform import prepare_layer_config, transform_humming_tensors
    from humming.tune import get_heuristics_config

    experts, rows, k_bytes = packed.shape
    k = k_bytes * 2
    schema = HummingWeightSchema(
        b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3,
        weight_scale_group_size=16, weight_scale_type="group",
        weight_scale_2_type="tensor",
    )
    config = prepare_layer_config(
        shape_n=rows, shape_k=k, weight_schema=schema,
        input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),
        num_experts=experts, torch_dtype=torch.bfloat16,
    )
    preset = PRESETS.get(mode)
    if preset is not None:
        if torch.cuda.get_device_capability(packed.device)[0] != 9:
            raise ValueError("explicit probes target SM90")
        if "mma_type" not in {field.name for field in dataclasses.fields(config)}:
            raise RuntimeError("explicit probes require the v0.1.12 LayerConfig MMA API")
        # In 0.1.12 MMA type belongs to LayerConfig, and changes the weight layout.
        # Set it BEFORE transforming; changing only the compiled kernel is wrong.
        config = dataclasses.replace(config, mma_type=MmaType(preset["mma_type"]))
    assert config.a_dtype == dtypes.bfloat16 and config.bs_dtype == dtypes.float8e4m3
    assert config.weight_scale_group_size == 16
    assert config.weight_scale_2_type == WeightScale2Type.TENSOR
    assert not config.use_int_weight_scale and not config.use_fused_e8m0_scale
    # Eight low-nibble-first FP4 codes per int32, with no numeric conversion.
    tensors = transform_humming_tensors(config, {
        "weight": packed.contiguous().view(torch.int32),
        "weight_scale": scales.contiguous(),
        "weight_scale_2": globals_.contiguous(),
    })
    gs = tensors["weight_scale_2"]
    assert gs.dtype == torch.float32 and tuple(gs.shape) == (experts,)
    assert tensors["weight_scale"].dtype == torch.float8_e4m3fn

    gemm_type = preset["gemm_type"] if preset else ("grouped_masked" if mode == "grouped" else "indexed")
    compute = {"gemm_type": gemm_type, "use_f16_accum": False}
    tuning = dict(get_heuristics_config(config, shape_m=tokens * 8, **compute))
    heuristic_tuning = tuning.copy()
    if preset is not None:
        tuning = _preset_tuning(preset)
    elif mode == "fullk":
        tuning["use_stream_k"] = False
    configs = HummingKernel.prepare_kernels(
        config.to_str(), json.dumps(compute), json.dumps(tuning)
    )
    # v0.1.12 prepare_kernels returns CPU launch metadata, never input counts.
    if configs.device.type != "cpu" or configs.numel() != 4:
        raise RuntimeError("expected one v0.1.12 CPU kernel configuration")
    kernel = HummingKernel._id2kernel[int(configs[2].item())]
    metadata = _kernel_metadata(kernel, config)
    if preset is not None:
        assert metadata["mma_type"] == preset["mma_type"]
        assert metadata["ptx_instruction"] is not None
        assert metadata["ptx_instruction"].startswith(
            "wgmma.mma_async" if preset["mma_type"] == "wgmma" else "mma.sync"
        )
    block_m = int(tuning["block_shape"][0])
    slots = tokens * 8
    capacity = tr.cdiv(slots + experts * (block_m - 1), block_m) * block_m
    num_sms = torch.cuda.get_device_properties(packed.device).multi_processor_count
    lock_count = max(2048, num_sms * int(tuning.get("num_ctas_per_sm", 1)) * 16)
    indexed = gemm_type == "indexed"
    return {
        "weight": tensors["weight"], "sf": tensors["weight_scale"], "gs": gs,
        "configs": configs, "launch": ops.launch_kernel,
        "block_m": block_m, "tuning": tuning, "heuristic_tuning": heuristic_tuning,
        "layer_config": config.to_str(), "compute_config": compute,
        "kernel_metadata": metadata, "preset": mode if preset else None,
        "gemm_type": gemm_type, "use_stream_k": bool(tuning.get("use_stream_k", True)),
        "mma_type": metadata["mma_type"],
        "sorted_ids": torch.empty(capacity, dtype=torch.int32, device=packed.device) if indexed else None,
        "expert_ids": torch.empty(capacity // block_m, dtype=torch.int32, device=packed.device) if indexed else None,
        "padded": torch.empty((), dtype=torch.int32, device=packed.device) if indexed else None,
        "expert_layout": None,
        "locks": torch.zeros(lock_count, dtype=torch.int32, device=packed.device),
    }


@torch.no_grad()
def prepare(weights, tokens, mode="default"):
    """Losslessly repack local experts; compile Humming; allocate fixed-T state."""
    if mode not in ("default", "fullk", "grouped") and mode not in PRESETS:
        raise ValueError(f"unknown mode {mode!r}; expected default/fullk/grouped or {tuple(PRESETS)}")
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    w13, w2 = weights["w13"], weights["w2"]
    if w13.ndim != 3 or w2.ndim != 3 or not w13.is_cuda:
        raise ValueError("expected CUDA packed weights [E,N,K/2]")
    experts, _, k_bytes = w13.shape
    d = k_bytes * 2
    if tuple(w13.shape) != (experts, 2 * d, d // 2):
        raise ValueError("W13 must have gate rows then up rows, each D x D")
    if tuple(w2.shape) != (experts, d, d // 2) or d % 128:
        raise ValueError("expected W2 [E,D,D/2] and D divisible by 128")
    for name, rows in (("w13", 2 * d), ("w2", d)):
        if weights[name].dtype != torch.uint8:
            raise TypeError(f"{name} must be raw packed uint8 E2M1")
        sf, gs = weights[name + "_sf"], weights[name + "_gs"]
        if sf.dtype != torch.float8_e4m3fn or tuple(sf.shape) != (experts, rows, d // 16):
            raise TypeError(f"{name}_sf must be E4M3 [E,N,D/16]")
        if gs.dtype != torch.float32 or tuple(gs.shape) != (experts,):
            raise TypeError(f"{name}_gs must be FP32 [E], in multiply convention")
    if tuple(weights["post_norm_w"].shape) != (d,):
        raise ValueError("post_norm_w must have shape [D]")
    if any(value.device != w13.device for value in weights.values() if torch.is_tensor(value)):
        raise ValueError("all weights must share a CUDA device")

    with torch.cuda.device(w13.device):
        p13 = _prepare_projection(w13, weights["w13_sf"], weights["w13_gs"], tokens, mode)
        p2 = _prepare_projection(w2, weights["w2_sf"], weights["w2_gs"], tokens, mode)
        if p2["block_m"] == p13["block_m"]:
            for key in ("sorted_ids", "expert_ids", "padded"):
                p2[key] = p13[key]
        # Masked kernels flatten [E,capacity,N]; their device counts limit work.
        grouped_capacity = tokens
        grouped = p13["gemm_type"] == "grouped_masked"
        assert p2["gemm_type"] == p13["gemm_type"]
        gemm_rows = experts * grouped_capacity if grouped else tokens * 8
        state = {
            "tokens": tokens, "d": d, "experts": experts, "mode": mode,
            "grouped": grouped,
            "w13": p13, "w2": p2, "version": _package_version(),
            "post_norm_w": weights["post_norm_w"].contiguous(),
            "counts": torch.empty(experts, dtype=torch.int32, device=w13.device),
            "h13": torch.empty((gemm_rows, 2 * d), dtype=torch.bfloat16, device=w13.device),
            "inter": torch.empty((gemm_rows, d), dtype=torch.bfloat16, device=w13.device),
            "y": torch.empty((gemm_rows, d), dtype=torch.bfloat16, device=w13.device),
            "z": torch.empty((tokens * 8, d), dtype=torch.float32, device=w13.device),
            "out": torch.empty((tokens, d), dtype=torch.bfloat16, device=w13.device),
        }
        if grouped:
            state.update({
                "reverse": torch.empty(tokens * 8, dtype=torch.int32, device=w13.device),
                "grouped_capacity": grouped_capacity,
                "grouped_x": torch.empty((gemm_rows, d), dtype=torch.bfloat16, device=w13.device),
                "expert_layout": state["counts"],
                "grouped_buffer_bytes": 5 * gemm_rows * d * 2,
            })
            p13["expert_layout"] = p2["expert_layout"] = state["expert_layout"]
        return state


def _routes(ids, state, projection):
    slots = state["tokens"] * 8
    # R also covers the final BM padding when slots < BM or BM is non-power-of-two.
    r = tr.next_power_of_2(tr.cdiv(slots, projection["block_m"]) * projection["block_m"])
    _pack_routes[(state["experts"],)](
        ids, state["counts"], projection["sorted_ids"], projection["expert_ids"],
        projection["padded"], slots, state["experts"], projection["block_m"],
        r, tr.next_power_of_2(state["experts"]), num_warps=4,
    )


def _gemm(x, out, projection, tokens, top_k):
    # Reset the independent lock buffer before every call, including graph replay.
    if projection["use_stream_k"]:
        projection["locks"].zero_()
    projection["launch"](
        configs=projection["configs"], inputs=x,
        weight=projection["weight"], weight_scale=projection["sf"],
        weight_scale_2=projection["gs"], outputs=out,
        sorted_ids=projection["sorted_ids"], expert_ids=projection["expert_ids"],
        num_tokens_padded=projection["padded"], expert_layout=projection["expert_layout"],
        locks=projection["locks"],
        top_k=top_k, valid_shape_m=tokens * 8,
    )


@torch.no_grad()
def run(data, state):
    """Return BF16 [T,D]; remote experts (negative or >= E) contribute zero."""
    x, ids, weights = data["hidden_states"], data["topk_ids"], data["topk_weights"]
    t, d, e = state["tokens"], state["d"], state["experts"]
    if tuple(x.shape) != (t, d) or x.dtype != torch.bfloat16:
        raise ValueError("state requires BF16 hidden_states of its fixed [T,D]")
    if tuple(ids.shape) != (t, 8) or tuple(weights.shape) != (t, 8):
        raise ValueError("topk_ids and topk_weights must have shape [T,8]")
    if not x.is_contiguous() or not ids.is_contiguous() or not weights.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be int32 or int64")
    if any(value.device != state["out"].device for value in (x, ids, weights)):
        raise ValueError("inputs and prepared state must share a device")

    p13, p2 = state["w13"], state["w2"]
    _count_routes[(e,)](ids, state["counts"], t * 8, tr.next_power_of_2(t * 8))
    if state["grouped"]:
        _pack_grouped_routes[(e,)](
            ids, state["reverse"], t * 8, state["grouped_capacity"],
            tr.next_power_of_2(t * 8), num_warps=4,
        )
        _gather_grouped_input[(t * 8,)](
            x, ids, state["reverse"], state["grouped_x"], d, e, tr.next_power_of_2(d)
        )
        _gemm(state["grouped_x"], state["h13"], p13, t, 1)
        _swiglu_grouped[(t * 8,)](
            state["h13"], ids, state["reverse"], state["inter"], d, e, tr.next_power_of_2(d)
        )
        _gemm(state["inter"], state["y"], p2, t, 1)
        _norm_grouped_slots[(t * 8,)](
            state["y"], state["post_norm_w"], ids, state["reverse"],
            state["z"], d, e, tr.next_power_of_2(d)
        )
    else:
        _routes(ids, state, p13)
        if p2["block_m"] != p13["block_m"]:
            _routes(ids, state, p2)
        _gemm(x, state["h13"], p13, t, 8)
        _swiglu[(t * 8,)](state["h13"], ids, state["inter"], d, e, tr.next_power_of_2(d))
        _gemm(state["inter"], state["y"], p2, t, 1)
        _norm_slots[(t * 8,)](
            state["y"], state["post_norm_w"], ids, state["z"], d, e, tr.next_power_of_2(d)
        )
    _combine_slots[(t, tr.cdiv(d, 256))](state["z"], weights, state["out"], d, 256)
    return state["out"]

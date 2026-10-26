"""One-shot lab hot/cold composition; no benchmark or GPU provisioning.

prepare(weights, tokens, incumbent) -> state; run(data, state) -> BF16 [T,D].
Scope: T128/256, E32, local expert IDs 0..31, remote IDs >=32 or negative.
incumbent(data) must run the existing v34 MoE with the SAME prepared weights
and return contiguous BF16 [T,D] on the current stream. It remains external.

Every timed call counts the ORIGINAL routes on device, sends whole experts
with count >=32 to existing Humming GROUPED_MASKED, and sends the remainder
to the callback. Filtered IDs/counts/output are preallocated; weights and
input activations are unchanged. Both backends run even when their subset
is empty. No host count read, input-dependent host branch, or cached hot set.
Classification, filtering, both complete backend calls, and final merge are
all inside run. Humming keeps its existing T/E-based library tile heuristic;
runtime counts select which experts use that tile family versus v34.

PRECISION: each backend accumulates internally in FP32, but its public output
is BF16. The final merge adds those TWO ROUNDED PARTIALS in FP32 and stores
BF16. This is NOT the original eight-expert FP32 sum and is not a claim of
production precision compliance. The parent must gate against its FP32
reference, including same-graph fresh hot/cold switches and empty-local replay.
No partial-subset timings may be added to claim a full-call speedup.

Warm run before graph capture. One state/callback workspace is for one stream;
do not execute overlapping calls through a shared incumbent workspace.
"""

import torch
import triton as tr
import triton.language as tl

import fresh_humming_adapter as H


HOT_THRESHOLD = 32


@tr.jit
def _split_routes(Ids, Counts, Hot, Cold, SLOTS: tl.constexpr,
                  E: tl.constexpr, THRESHOLD: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0) * B + tl.arange(0, B)
    ids = tl.load(Ids + slot, slot < SLOTS, other=-1)
    local = (slot < SLOTS) & (ids >= 0) & (ids < E)
    safe_expert = tl.minimum(tl.maximum(ids, 0), E - 1)
    count = tl.load(Counts + safe_expert, local, other=0)
    hot = local & (count >= THRESHOLD)
    cold = local & ~hot
    tl.store(Hot + slot, tl.where(hot, ids, 255), slot < SLOTS)
    tl.store(Cold + slot, tl.where(cold, ids, 255), slot < SLOTS)


@tr.jit
def _merge_partials(Hot, Cold, Out, N: tl.constexpr, B: tl.constexpr):
    n = tl.program_id(0) * B + tl.arange(0, B)
    hot = tl.load(Hot + n, n < N, other=0).to(tl.float32)
    cold = tl.load(Cold + n, n < N, other=0).to(tl.float32)
    tl.store(Out + n, (hot + cold).to(tl.bfloat16), n < N)


@torch.no_grad()
def prepare(weights, tokens, incumbent, parallel=False):
    """Prepare Humming hot workspace and fixed buffers; callback owns v34 state."""
    if tokens not in (128, 256):
        raise ValueError("this single diagnostic targets T128/256")
    if not callable(incumbent):
        raise TypeError("incumbent must be a prepared callable: data -> BF16 [T,D]")
    if weights["w13"].ndim != 3 or weights["w13"].shape[0] != 32:
        raise ValueError("hybrid requires E32 packed weights and local IDs 0..31")
    hot = H.prepare(weights, tokens, mode="grouped")
    d, device = hot["d"], hot["out"].device
    return {
        "tokens": tokens, "d": d, "experts": 32, "threshold": HOT_THRESHOLD,
        "hot": hot, "incumbent": incumbent,
        "side_stream": torch.cuda.Stream(device=device) if parallel else None,
        "counts": torch.empty(32, dtype=torch.int32, device=device),
        "hot_ids": torch.empty((tokens, 8), dtype=torch.int32, device=device),
        "cold_ids": torch.empty((tokens, 8), dtype=torch.int32, device=device),
        "out": torch.empty((tokens, d), dtype=torch.bfloat16, device=device),
        "metadata": {
            "hot_backend": "Humming grouped_masked", "cold_backend": "external v34 callback",
            "threshold": HOT_THRESHOLD, "classification": "runtime device counts >=32",
            "hot_tuning": {key: hot[key]["tuning"] for key in ("w13", "w2")},
            "partial_dtype": "BF16", "merge_dtype": "FP32", "output_dtype": "BF16",
            "preserves_full_fp32_expert_combination": False,
            "filtering_and_merge_timed": True, "gpu_numerics_validated": False,
            "execution": "fork_join" if parallel else "sequential",
        },
    }


@torch.no_grad()
def run(data, state):
    """Classify, filter, run both backends and merge within the measured call."""
    x, ids, weights = data["hidden_states"], data["topk_ids"], data["topk_weights"]
    t, d = state["tokens"], state["d"]
    if tuple(x.shape) != (t, d) or x.dtype != torch.bfloat16:
        raise ValueError("state requires BF16 hidden_states of its fixed [T,D]")
    if tuple(ids.shape) != (t, 8) or ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be int32/int64 [T,8] with local IDs 0..31")
    if tuple(weights.shape) != (t, 8) or weights.dtype != torch.float32:
        raise TypeError("topk_weights must be FP32 [T,8]")
    if any(not value.is_contiguous() for value in (x, ids, weights)):
        raise ValueError("inputs must be contiguous")
    if any(value.device != state["out"].device for value in (x, ids, weights)):
        raise ValueError("inputs and prepared state must share a device")

    H._count_routes[(32,)](ids, state["counts"], t * 8, tr.next_power_of_2(t * 8))
    _split_routes[(tr.cdiv(t * 8, 256),)](
        ids, state["counts"], state["hot_ids"], state["cold_ids"],
        t * 8, 32, state["threshold"], 256,
    )
    # Own dictionaries, shared live activations/weights; original data is untouched.
    hot_data = {**data, "topk_ids": state["hot_ids"]}
    cold_data = {**data, "topk_ids": state["cold_ids"]}
    side = state["side_stream"]
    if side is None:
        hot_out = H.run(hot_data, state["hot"])
        cold_out = state["incumbent"](cold_data)
    else:
        # Both branches depend on the timed split and rejoin before the merge.
        # Humming and the callback have separate workspaces. The caller must
        # serialize whole calls, as for the sequential implementation.
        origin = torch.cuda.current_stream(x.device)
        side.wait_stream(origin)
        with torch.cuda.stream(side):
            hot_out = H.run(hot_data, state["hot"])
        cold_out = state["incumbent"](cold_data)
        origin.wait_stream(side)
    if (not torch.is_tensor(cold_out) or tuple(cold_out.shape) != (t, d)
            or cold_out.dtype != torch.bfloat16 or cold_out.device != x.device
            or not cold_out.is_contiguous()):
        raise TypeError("incumbent callback must return contiguous BF16 [T,D] on the input device")
    _merge_partials[(tr.cdiv(t * d, 256),)](hot_out, cold_out, state["out"], t * d, 256)
    return state["out"]

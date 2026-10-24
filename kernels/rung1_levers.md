# Rung 1: Python-level levers over SGLang's own kernels (written 2026-10-23, no GPU)

Files (each is a full submission: first line `# okc: problem=<p>`, second `# nonce r1-1`):
- `kernels/muse_decode_layer/r1_glue.py`: the vehicle. Levers L1–L5 are ON. L6/L7 are empty tables. L8/L9 are OFF.
- `kernels/muse_moe_w4a16/r1_glue.py`: L1, L3', L4 ON; L6 table empty.
- `kernels/muse_attention_decode/r1_glue.py`: L2 ON; L7 table empty.
- No `muse_bf16_proj` file. The only proj lever found is L9 (the Hopper GEMV), which builds a new JIT module, so it belongs to rung 1b/2.

`S/` = `repos/sglang/python/sglang` (a55ab7d5). `B/<p>` = `problems/<p>/baseline.py`. Every lever is a module constant
(`R1_*`, or a table) near the top of the file. Each file prints its lever states once at import (`[r1] ...`), so the run log
shows which variant ran.

**Cost model for "expected saving".** One tiny kernel removed from a CUDA graph saves about 2–4 µs: the launch gap plus
a minimum runtime of about 1–2 µs. This is UNVERIFIED and will be checked against the rung-0 profile. Byte costs are
added only where they matter; every intermediate below is L2-resident within a replay. The per-load speedup is
`t_stock / (t_stock − Δ)`, with `t_stock` taken from the rung-0 check. For example, a 150 µs decode-1 layer with Δ = 40 µs
gives 1.36×.

## Launch accounting (CPU stub model, `scratchpad/r1_smoke.py`)
The stub model counts stub kernels plus the non-view aten ops issued from the problem file. FA3 helper kernels are excluded. moe_align counts as 1 entry, although it is 2 real launches at 8 < T < 128 on the stock path; see L4.

| problem / T | stock | r1 | removed (+ align) |
|---|---|---|---|
| decode_layer T=1, 8 | 36 | 21 | 15 |
| decode_layer T=16 | 37 | 22 | 15 (+1) |
| decode_layer T=32/33 | 26 | 22 | 4 (+1) |
| decode_layer T=128, 256 | – | – | 4 (QK 3 + remap 1) |
| moe bs 1, 8 | 17 | 6 | 11 |
| moe bs 16 | 17 | 6 | 11 (+1) |
| moe bs 32 | 6 | 6 | 0 (+1) |
| attention, every shape with kv_scale = 1 | 7 | 4 | 3 |

**Expected Δ per shape (2–4 µs per launch):**
- decode_layer:
  - T=1, 8, 16: ~16 launches, Δ ≈ 32–64 µs. Add ~3–5 µs at T=16 for the ~20–30 MiB of FP32 L2 traffic in the torch post-norm chain.
  - T=32: 5 launches, Δ ≈ 10–20 µs.
  - T=128, 256: 4 launches, Δ ≈ 8–16 µs.
- MoE:
  - bs 1, 8: Δ ≈ 22–44 µs.
  - bs 16: Δ ≈ 24–48 µs.
  - bs 32: Δ ≈ 2–4 µs.
  - bs 128, 256: ≈ 0.
- attention: Δ ≈ 6–12 µs on every shape. The sliding shapes (weight 46, floors 0.1–3.7 µs) are launch-bound, so this is a large fraction of them.

---

## L1 `R1_POSTNORM_TRITON_ALL_T`: Triton post-expert norm + weighted sum at every T (layer, MoE)

**Evidence**
- `S/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py`:
  - `:7` sets `_MIN_TOKENS_FOR_FUSED = 32`.
  - Below 32 tokens, `:77-99` runs where(dead slots → 0), float, ×w, pow, mean, +eps, rsqrt, ×, ×score, sum, to(bf16). That is 12 launches counting the `torch.zeros(())` fill.
  - At ≥ 32 tokens, `:101-118` launches the `@triton.jit` kernel `:10-60`.
- Both baselines call the wrapper (`B/muse_decode_layer:264-266`, `B/muse_moe_w4a16:149-151`), so the scored shapes that run the chain are T = 1, 8, 16. We import `_post_expert_norm_weighted_sum_kernel` and launch it with exactly the stock arguments: grid `(T,)`, `BLOCK_H=4096`, 8 warps, 2 stages.
- We never touch `_MIN_TOKENS_FOR_FUSED`, because editing a loaded module would get the submission rejected.

**Why it is exact**
- Both paths compute the same thing, `Σ_k score_k · rmsnorm(x_k · w)`, in FP32 with one BF16 rounding at the end.
- Dead slots (local id −1): the torch path zeroes the slot first. 0·w → var 0 → 0·rsqrt(1e-8) = 0, so it adds 0. The kernel skips the slot (`:38-40`). Neither path reads Marlin's uninitialised remote rows. CPU check: remote rows filled with NaN, both paths finite and equal.
- Neither path rounds `x·w` to BF16 before the norm. Neither does `reference.py:88-90`; that rounding is a quirk of `template.py` only.
- The only difference is the FP32 reduction order (and `rsqrt.approx`), about 1e-7 relative. The judge already accepts this kernel at the T ≥ 32 shapes.

**Saving:** 11 launches at T < 32 (T = 1, 8, 16 → decode-1, verify-1, decode-16 in both problems), plus the FP32 passes.

**Risk:** very low. It is the same compiled kernel; Triton does not specialise on the grid, so T < 32 reuses the T ≥ 32 binary. The import is guarded: if the symbol is missing, the stock wrapper runs and the import line prints `postnorm_triton=False`.

**Bisect:** set the constant to False. The file is then stock for this step.

## L2 `R1_QK_FUSED_FP8_STAGE`: QK-norm(+RoPE) with the FP8 Q/K/V write fused in (layer, attention)

**Evidence**
- `S/kernels/ops/muse_spark_v12/fused_qk_norm.py`, the kernel the baseline already runs (`FusedQKNorm` → TRITON first, `:147`):
  - It has an epilogue enabled by `HAS_KV_STORE`/`FP8_KV_STORE` (`:105-127`): it rounds q and k to BF16, clamps to ±448, stores k and raw v as FP8 rows of the given cache, and writes FP8 q (`:324-329`).
- Stock path is 5 launches: qk_norm, `k.to(fp8)`, `v.to(fp8)`, store_cache, `q.to(fp8)` (`B/muse_decode_layer:277-298`, `B/muse_attention_decode:90-112`).

**Trap: the kernel skips cache row 0**
- `:112` says "Slot zero is reserved for CUDA-graph padding" and masks `loc != 0`.
- In this problem row 0 is a live row. The baseline passes `reserved_skip_index=-1` (`B/muse_decode_layer:287-290`).
- The judge draws pages as a random permutation that includes page 0 (`problems/muse_decode_layer/reference.py:261-269`). In the 11-page test, about 9% of calls hit row 0.

**How we use it**
- So we never hand the kernel the real cache. It writes to staging tensors `k_stage`, `v_stage` of shape [T+1, hkv·64] FP8, with `cache_loc = arange(1, T+1)`. This is a constant built in `prepare()`; the attention file has no `prepare()`, so it caches it at module level outside capture, like the baseline's `_cos_sin`.
- Then `store_cache(k_stage[1:], v_stage[1:], ..., data["cache_loc"], reserved_skip_index=-1)` runs exactly as the baseline's store does.
- We call the public method `FusedQKNorm.forward_triton(qkv, positions|None, cos_sin|None, kv_store)`. That is the same method the baseline's dynamic dispatch reaches (`S/kernels/fused_op.py:575-580`), so the same PDL flag applies.
- Gated on `cfg["kv_scale"] == 1.0`. The kernel requires unit scales (`:186`), and the two kv_scale = 1/64 tests take the stock branch unchanged.

**Why it is exact**
- q and k: stock rounds FP32 → BF16 (the kernel's bf16 output), then PyTorch rounds BF16 → FP8 (RNE). The fused path rounds `x.to(bf16).to(f32)` (`:105-106`), then stores to FP8 (Triton RNE satfinite). That is the same double rounding.
- v is raw BF16 → FP8 in both paths.
- The ±448 clamp cannot fire:
  - |k| ≤ √64 = 8 after the per-head RMSNorm, and RoPE preserves pair norms (CPU check: max 4.2).
  - v is a projection of order 1–10. For |v| ≥ 464, stock gives NaN while the fused path saturates, so they differ only where stock itself would fail.
- CPU checks:
  - the staging-row addressing for hq/hkv = 16/2 and 128/16;
  - bf16→fp8 equals clamp→fp8 on 1e5 values;
  - the stub smoke test shows identical outputs and caches, with row 0 a new token's row and both capture modes.

**PDL ordering (checked because L2 puts these kernels next to each other, which the baseline never does)**
- The QK kernel is launched with `launch_pdl` and calls `gdc_launch_dependents()` before its stores (`fused_qk_norm.py:57-61, 339`).
- `store_cache` follows it directly. It is PDL-launched too, but it runs `PDLWaitPrimary` (griddepcontrol.wait) before it loads any k/v byte (`S/kernels/jit/csrc/elementwise/kvcache.cuh:65` vs loads at `:66-72`). So it cannot read staging rows that have not been written yet.
- After `store_cache` comes FA3 (in the baseline a q cast sat between them). Upstream FA3 launches its main kernel with PDL only behind its own `prepare_varlen` kernel (`repos/flash-attention/hopper/flash_fwd_launch_template.h:197`), and that kernel is launched normally. With scheduler metadata given (full decode) there is no PDL at all. Either way FA3 runs only after store_cache has completed.
- The sgl-attn fork is assumed to match upstream here (UNVERIFIED).

**Saving:** 3 launches on every shape of decode_layer and attention, plus one [T, 1024] BF16 q round-trip.

**Risk:** low–moderate.
1. It compiles a new Triton specialisation of an existing kernel (HAS_KV_STORE, FP8 store), which the baseline never compiles on sm_90. FP32 → e4m3 stores are supported on sm_89+, but a compile error would show at the first call.
2. `_forward_triton` asserts that the caches are contiguous, which the staging tensors are.

**Bisect:** set the constant to False to return to the stock 5-launch path. A failure on "new cache entries", or a Triton compile error in the log, points here.

## L3 `R1_LOCAL_ID_TABLE` (layer) / L3' `R1_EXPERT_MAP_INDEX_SELECT` (MoE): global → local expert ids

**Evidence**
- Layer: stock runs `(topk_ids - local_start) % 256` (sub + remainder, `B/muse_decode_layer:335`), then `expert_map[ids]` (an index kernel with an int32 index, `:241`): 3+ launches.
- r1 builds `table[g + 256·s] = expert_map[(g − s) % 256]` over all s, g in [0, 256) (256 KiB int32, a constant built in `prepare()`). It then runs `torch.add(topk_ids, local_start, alpha=256)` and one `index_select` with the int32 index: 2 launches.
- `local_start` stays a device tensor, so there is no sync.
- MoE: `index_select(expert_map, 0, ids)` replaces `expert_map[ids]`. Same count, and it avoids any int64 index conversion inside `index`.

**Why it is exact:** integer arithmetic. CPU check: equal to the stock formula for all 256 starts (not only multiples of 32) on random ids. Requires `local_start ∈ [0, 256)`; TASK.md says it is a multiple of 32 (`reference.py:486`).

**Saving:** 1 launch on every layer shape; 0–1 on MoE.

**Risk:** very low. **Bisect:** set the constant to False.

## L4 `R1_ALIGN_LOCAL_BUCKETS`: moe_align over the 32 local experts (layer, MoE)

**Evidence**
- Stock calls `moe_align_block_size(local_ids, bm, 256)` (`B/muse_decode_layer:246`), so the kernel has 257 buckets.
- Wrapper `S/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`:
  - `:114-128`: a Triton single launch for numel ≤ 64.
  - `:153-163`: otherwise the AOT kernel. With more than 64 buckets that is `align` + `count_and_sort` (`S/kernels/aot/csrc/moe/moe_align_kernel.cu:405-446`), so 2 launches.
- With num_experts = 32 (33 buckets ≤ 64) and numel < 1024, it takes the single-launch `small_batch_expert` kernel (`.cu:387-404`, kernel `:284-365`).
- Every path uses bucket = id + 1 and expert = bucket − 1 (`.cu:39, 147, 329, 280, 356`). Local ids are in [−1, 32), so the remote (−1) blocks come out as expert −1 exactly as before, and Marlin skips them (`is_ep`).
- Marlin's grid is `sms · blocks_per_sm` (`S/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh:752`), and its block loop is bounded by the device `num_tokens_past_padded` (`:68-69`, `marlin_template.h:384-385`). The smaller `sorted_ids` buffer only lowers `c_tmp` and workspace minimums, which the wrapper sizes from the buffer it gets (`moe_wna16_marlin.py:119-128`, `.cuh:1077-1079`).
- Applied only for T ≥ 9 (`R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS`). For T ≤ 8, stock is already a single (Triton) launch.

**Why it is exact:** the padded total, the per-block expert ids and the per-block token sets are identical; only the order inside a block can change. Marlin rows are independent, and the work partition depends only on the device padded total. CPU check: the bucket emulation for T ∈ {1..257}, 20 trials each, is identical.

**Saving:** 1 launch at 8 < T < 128 (decode-16 and decode-32 in both problems). T ≥ 128 keeps 2 launches, with a 33-entry scan instead of 257.

**Risk:** low. The CUDA small-batch kernel with −1 ids is a path the baseline never takes; its semantics were read from source but have not run.

**Bisect:** set the constant to False (or raise MIN_TOKENS). A MoE-output or residual failure at T = 16/32/33 only points here.

## L5 `R1_UNPOOL_INTO_ZEROS`: sliced unpooler at T ≤ 8 (layer)

**Evidence**
- Stock runs `F.linear(y, unpooler[:1024])`, then `F.pad` (`B/muse_decode_layer:338-342`). `constant_pad_nd` is a fill plus a copy, so that is 3 launches.
- r1 runs `zeros([T, 8192])`, then `torch.mm(y, w[:1024].t(), out=z[:, :1024])`: 2 launches.
- `F.linear` without bias is `mm` (`S/srt/layers/quantization/unquant.py:451`, falling through to `F.linear` on H200), so this is the same cuBLAS GEMM.
- The out view has strides (8192, 1), which ATen hands to cuBLAS as ldc = 8192 with no proxy.

**Why it is exact:** same GEMM; the other columns are zero in both. CPU check: bit-equal.

**Saving:** 1 launch at T = 1 and 8 (decode-1, verify-1).

**Risk:** low. If ATen inserted a proxy and copy, the result would still be correct but the gain would be zero.

**Bisect:** first in the bisect order.

## L6 `BLOCK_SIZE_M_OVERRIDE` (layer, MoE) and L7 `NUM_SPLITS_OVERRIDE` (layer, attention): tables, empty = stock
- L6: a {T: bm} table applied after the stock heuristic (`B/muse_decode_layer:243-245`). Rung 0b fills it from measured {8, 16, 32, 48, 64} sweeps. Correctness does not depend on bm.
- L7: keyed by {("sliding" | "full", batch, q_len): num_splits}, passed to `flash_attn_with_kvcache(num_splits=)` (`S/kernels/aot/python/sgl_kernel/flash_attn.py:63`).
  - The judge builds `scheduler_metadata` with `num_splits=0` (`problems/muse_decode_layer/reference.py:238-241`), so an override also passes `scheduler_metadata=None`, letting FA3 build its own (one extra prepare launch on full decode).
  - Only fill this table from rung-0b measurements.

## L8 `R1_ROUTER_SPLIT_ALL_T` (OFF)
- BF16 hi + lo with FP32 accumulation is allowed at any T (TASK.md Rules; `S/srt/models/muse_spark.py:284-289`).
- At T ≤ 8 it turns 1 FP32 GEMM (8 MiB) into 2 BF16 GEMMs (2 × 4 MiB) plus an add. That is the same bytes and more launches, all on the side stream hidden under the 64 MiB pooler (`B/muse_decode_layer:320-330`).
- Expected gain is ≤ 0, so it stays OFF.

## L9 `R1_HOPPER_GEMV_T1` (OFF; rung 1b/2)
- `S/kernels/ops/gemm/hopper_bf16_gemv.py` is SGLang's opt-in `--bf16-gemm-backend gemv` path (`unquant.py:318-330`, `:434-441`). It claims 3.9–4.3 vs 3.2–3.9 TB/s against cuBLAS at M = 1.
- At T = 1, qkv, o and pooler qualify (`use_hopper_bf16_gemv`, `:43-55`). That is ~116 MiB, so roughly 2–4 µs on the decode-1 load only.
- Numerics: BF16 in and out, FP32 accumulation, so it is allowed.
- It is a `load_jit` CUDA module the baseline never builds, which adds compile time inside the 15-minute limit. Try it in its own check run.

---

## Bisect order if an `okc check` fails
Flip each lever OFF one at a time; every change is one line in one file:
1. `R1_UNPOOL_INTO_ZEROS`
2. `R1_QK_FUSED_FP8_STAGE`
3. `R1_ALIGN_LOCAL_BUCKETS`
4. `R1_LOCAL_ID_TABLE` / `R1_EXPERT_MAP_INDEX_SELECT`
5. `R1_POSTNORM_TRITON_ALL_T`

Symptoms that point straight at one lever:
- "new cache entries" fails, or the log shows a Triton/fp8 compile error: L2.
- A routing-rule failure: L3.
- MoE output fails only at T = 16, 32 or 33: L4.
- Everything fails at T ≤ 8 only: L5 or L1.

Batch the bisect variants (nonce r1-2 …) in one submission wave, not one after another.

## Verified on CPU (torch 2.5.1)
- `python3 -m py_compile` passes on all three files.
- `scratchpad/r1_units.py` (ALL PASS) covers: post-norm algorithm vs chain with NaN-filled dead slots; the local-id table for all starts; bucket layout 257 vs 33; staging-row addressing including row-0 untouched; FP8 clamp equivalence; mm(out=slice).
- `scratchpad/r1_smoke.py` (ALL PASS) runs `baseline.py` and `r1_glue.py` side by side on stub kernels that mirror the real signatures and the semantics that matter (row-0 skip, bucket rule, NaN garbage rows, FP8 casts). It compares outputs, routing and caches bit-for-bit, and counts launches:
  - attention: 4 configs incl. hq = 128 / page 64 and kv_scale = 1/64;
  - MoE: bs 1–33;
  - layer: T = 1, 8, 16, 16, 33 with local_start 0–224.
- Scratchpad = `<tmp>`.

**Not verified:**
- anything that needs the GPU: Triton compile of the FP8 store variant, ATen's handling of the strided `out=`, real launch costs;
- the judge's harness doing an eager warm-up before capture (the baseline relies on that too).

## Rung 2 candidates found here (each needs new kernel code)
- QK-norm + FP8 + a direct cache write without the slot-0 mask: an in-file copy of the Triton kernel, which saves store_cache (2 → 1).
- Hopper GEMV (L9) and a router hi‖lo single GEMM (K = 16384 concat).
- Pre-expert norm: torch 2.14.1 in the image (`repos/sglang/python/pyproject.toml:6`) should have a fused `F.rms_norm`. If the rung-0 profile shows ~8 elementwise kernels there, swap in sgl_kernel's rmsnorm (one launch).

# K1-FA4: L=8 full-layer attention on FA4's CuTe-DSL Hopper forward (2026-10-25)

**Files:** `k1_fa4.py` (kernel), `k1_fa4_route.py` (drop-in route + self-test + timing), `k1_fa4_lab.py`
(brackets and variants). Logs: `box_out/fa4_job1.log` … `fa4_job6_route.log` (serverless H200, judge image).

## 1. Why this route, after k1_v5–v8f

The hand-written PTX wgmma kernels (v5–v8f) were stuck between ptxas serialization (C7511/C7513) and latency.
FA4's sm90 forward (`sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py` in the image, identical
to `repos/sglang/python/sglang/kernels/ops/attention/flash_attn/cute/`) has expert-written wgmma pipelines
with correct operand fences, pack-GQA, paged KV through cp.async (`paged_kv.py`), and split-KV with a combine
kernel. Its FP8 path is SM100-only, so the question was whether an FP8-load / 16-bit-compute derivative is fast.

## 2. The image (job 1)

| item | value |
|---|---|
| CuTe DSL | nvidia-cutlass-dsl 4.8.0 (cu13), quack-kernels 0.6.5, apache-tvm-ffi 0.1.11 |
| FA4 | flash-attn-4 4.0.0b19 (pip) and the sglang-vendored copy (the one used here) |
| JIT compile | 1.6–2.6 s per kernel config (process-local cache; `SGLANG_CUTE_AOT_CACHE_DIR=""`) |

## 3. Measurements (attention core only, same process, CUDA graph replay, cold L2, µs)

FA3 = `sgl_kernel` `flash_attn_with_kvcache`, E4M3 Q/K/V, as SGLang calls it. FA4 F16 = stock FA4 with the FP8
cache converted to F16 once outside the timing (2x bytes; a bracket, not a legal kernel). `l2` = page table
folded into a 16K-row pool (L2-resident: the compute/latency-bound time); `half` = half the context.

| variant | ver16 full | ver32 full | notes |
|---|---|---|---|
| FA3 (stock) | 190.7 | 378.0 | l2 170.1 / 335.7 |
| FA4 F16 stock, IWO, tile 64x128 | 222.1 | 449.0 | l2 152.9 / 302.4; 1 CTA/SM |
| FA4 F16 stock, no IWO | 194.2 | 389.8 | l2 163.7 / 325.4 |
| FA4 F16, 2 CTAs/SM (p2), no IWO | 166.2 | 327.5 | l2 131.5 / 256.8 |
| K1-FA4 FP8, barrier mapping | 169.6 | 338.5 | upcast costs ~60 µs at v32 |
| K1-FA4 FP8, own mapping (shipped) | **151.1** | **300.4** (299.4 at 2x splits) | l2 143.0 / 277.8, rel L2 1.7e-3 |
| same, upcast skipped (timing only) | 123.2 | 244.2 | floor of this producer structure |

Gate (≤ 165 / ≤ 300): **met** at v16, **at the line** at v32 (300.4 at the default split count, 299.4 at 2x).
Versus FA3: 1.26x at both shapes. Versus k1_v7 (187 / 337 in its own bench): 1.24x / 1.12x.

## 4. What the kernel is

1. **Occupancy.** For M = 64 rows (8 tokens x 8 q heads per KV head) FA4 picks one MMA warpgroup per CTA, and
   `min_blocks_per_mp=1` with `setmaxnreg (256, 56)` gives one CTA, so one consumer chain, per SM. Patch:
   `min_blocks_per_mp=2`, `setmaxnreg (200, 56)` (fits 256 threads x 128 registers). L2-resident v32 303 → 257.
   Splits = the FA4 heuristic x 2 (CTAs per SM).
2. **FP8 producer** (`K1FA4Sm90.load`, replaces FA4's producer warpgroup loop; the consumer is untouched):
   - FA4's `PagedKVManager` gathers FP8 rows (page size 1, 64 B per row and head) with cp.async into a 2-slot raw
     FP8 ring (2 x 16 KB). The next tile is in flight while the current one is upcast.
   - Each thread upcasts exactly the 16-byte chunks its own cp.async wrote (E4M3 → F16 is exact), with 128-bit
     stores into the swizzled F16 sK/sV stage. Visibility comes from the thread's own `cp.async.wait_group 1`;
     raw-slot reuse is ordered by program order. There is no barrier (the 8x8-mapping version needed two
     named barriers per tile and was 40 µs slower at v32).
   - `fence.proxy.async.shared::cta`, then a plain `mbarrier.arrive` per thread on FA4's full barrier (count
     128, the same count its cp.async `arrive.noinc` uses).
3. **Numerics.** QK and PV are F16 wgmma with FP32 accumulation; P is F16 (stock FA3: E4M3 Q and P, E4M3 PV);
   the output is BF16 from FA4's split combine. The cache bytes stay FP8 and every key is attended. Rel L2 vs
   an exact FP32 softmax is 1.7e-3, which is the BF16 output rounding (FA3: 4.4e-2 / 6.8e-2).
4. **Smem:** Q 8 KB + K/V F16 stages 2 x 32 KB + raw 32 KB = 104 KB per CTA, 2 CTAs/SM.
5. **Packaging.** The image's FA4 files are patched as text at import (6 anchors in `flash_fwd_sm90.py`, 2 in
   `interface.py`, each asserted unique) and loaded from /tmp, so no FA4 source is copied into the repo. The
   CuTe compile cache is process-local. The kernel is graph-capturable: the lab and route time it under CUDA graphs.

## 5. Route (`k1_fa4_route.py`)

- `kernel_fn` uses K1-FA4 only for q_len 8, full layer, page 1, kv_scale 1, head dim 64, Hq = 8 Hkv, and B ≥ 16.
  Everything else goes to `r1_glue.kernel_fn`.
- The prologue and epilogue are r1's (FusedQKNorm Triton with the FP8 KV-store epilogue, `store_cache`,
  `rms_norm_gate`). The E4M3 query is cast to F16.
- `max_seqlen_k` comes from the page table's shape, so there is no host sync.
- `prepare()` and `self_test()` run `reference.check_implementation` on forced-route shapes: ctx 100 / 5000
  mixed / 6000 edges / 80K coherent mixed / 100K. The route goes live only if every shape passes.
- Results of the self-test and the full-kernel_fn timing: §6.

## 6. Route results (job 6, `box_out/fa4_job6_route.log`)

**Self-test: 5/5 PASS** (forced route, `reference.check_implementation`; prepare including JIT took 4.9 s):

| shape | rel L2 | bound |
|---|---|---|
| ctx 100K, B2 | 9.20e-2 | 0.150 |
| ctx 6000 edges, B4 | 8.36e-2 | 0.150 |
| ctx 100, B3 | 6.72e-2 | 0.150 |
| ctx 5000 mixed, B5 | 7.99e-2 | 0.150 |
| ctx 80K coherent mixed, B4 | 3.69e-2 | 0.107 |

The error equals the reference's own error with E4M3 queries (0.093, 0.082, …), so it comes from the stock E4M3
query. The attention core itself is at 1.7e-3. Using a BF16 query instead (r1's non-staged prologue) would
lower it, at the cost of more prologue launches.

Full `kernel_fn` (QK-norm + store + attention + gate), graph replay, cold L2:

| shape | r1 (stock FA3) | K1-FA4 route | speedup |
|---|---|---|---|
| ver16 full | 195.5 µs | **159.5 µs** | 1.23x |
| ver32 full | 382.9 µs | **313.3 µs** | 1.22x |

For comparison, the brief's in-layer k1_v7 numbers are 185.8 / 338.2 µs.

## 7. Open levers (not done)

- The upcast costs 56 µs at v32 (300 vs 244 with the upcast skipped). An FP8 QK (E4M3 Q, the K tile straight
  from the raw ring, as stock does) would remove half of it; V still needs F16 for an MN-major PV. Estimate
  v32 ≈ 270–280.
- IWO (FA3-order overlap) is slower than no-IWO at 1 CTA/SM. It was not re-tested with the FP8 producer.
- The layer integration (decode_layer k3 file) needs the module inlined or imported. The submission size is
  fine (this file is about 12 KB).

## 8. v2: FP8 QK + sliding-window layers (`k1_fa4v2.py`, `k1_fa4v2_route.py`; 2026-10-25)

**Kernel changes vs v1.**
- QK is an E4M3 x E4M3 wgmma. Q is r1's E4M3 query, the precision stock FA3 uses (A16), loaded by FA4's
  pack-GQA TMA into an FP8 sQ.
- K is gathered by FA4's `PagedKVManager` straight into the FP8 swizzled sK stages and signalled with FA4's own
  cp.async mbarrier arrive. There is no K upcast.
- PV is unchanged from v1: F16 with F16 P. V goes through the raw FP8 ring and the barrier-free own-bytes upcast.
- Producer order per block: K(i) issued, then V(i-1) upcast. This serves both the no-IWO and IWO consumers.
- Sliding layers use FA4's local mask: window_left = 2047, right = 0, causal, the same semantics as r1's FA3
  call with window (2047, 0).
- K/V pipeline depth is 3 on full layers and 2 on sliding layers. The compile key separates local from full, so
  both compiled kernels coexist.

**A19 packaging.**
- The image's files are read, patched as text and loaded from a `tempfile.mkdtemp` dir under the new names
  `k1fa4v2_*`.
- No environment writes. The JIT cache is process-local through `cache_dir=None` in the patched interface's
  own cache factory.
- No existing module or sys.modules entry is touched.

**Attention core** (`box_out/fa4v2_job1_core.log`; same harness as §3):

| build | ver16 full | ver32 full | rel L2 vs exact FP32 |
|---|---|---|---|
| FA3 (stock) | 191.0 | 378.0 | 4.4e-2 / 6.8e-2 |
| v1 (F16 QK/PV) | 153.0 | 300.6 | 1.7e-3 |
| **v2 (FP8 QK, F16 PV), stages 2** | **132.0** | **261.8** | 2.3e-2 / 1.9e-2 (the E4M3 query) |
| v2, stages 3 | 131.6 | 258.1 | |
| v2 L2-resident / half context | 125.5 / 69.9 | 247.1 / 138.1 | |

The targets were ≤ 140 and ≤ 280: both met. That is 1.45x FA3 and 1.15x v1 at verify32.

**Full kernel_fn** (QK-norm + store + attention + gate; graph replay, cold L2; jobs 1–3, repeats agree within
about 1 µs):

| shape | r1 (stock FA3) | v1 route | v2 route | v2 vs r1 |
|---|---|---|---|---|
| ver1 sliding | 18.1 | — | **13.5** (auto splits) | −25% |
| ver16 sliding | 23.9 | — | **18.0** (4 splits; auto 18.7) | −25% |
| ver32 sliding | 29.5 | — | **22.4** (auto = 4 splits) | −24% |
| ver1 full | 31.2 | — | 30.3 | −3% (not routed) |
| ver16 full | 195.2 | 159.6 | **136.5–137.6** | −30% |
| ver32 full | 383.1 | 311.1 | **263.5** (stages 3; 266.3 with 2) | −31% |

- Sliding gate (≥ 15% faster than r1 at ver16 and ver32): **met**, at 21–25%.
- IWO (FA3-order overlap) is within ±1 µs of no-IWO everywhere, so no-IWO is kept.
- Sliding split count: 2 is too few at B16 (21.5). 8 and 16 are too many at B32 (27.6 / 33.7): combine and
  per-CTA overhead dominate.

**Self-test** (forced route, attention problem's `check_implementation`): 9/9 PASS.
- Full: ctx 100 / 6000 edges / 80K coherent mixed / 100K.
- Sliding: ctx 1000 (shorter than the window) / 5000 mixed / 6000 edges (keys at both window edges) / 6000 soft / 100K.
- Errors are at the reference's own E4M3-query level, for example 0.092 against a bound of 0.150.
- decode_layer uses the same bound constants (REL_L2_TOL 0.15, FP8_QUERY_FACTOR 2.5, floor 0.03). The layer
  judge clone of the integration is the real check there.

**What the integration should route** (`route_ok`):
- L=8 full layers for B ≥ 16.
- L=8 sliding layers for all B (1 / 16 / 32).
- Page size 1 and kv_scale 1 only, with the 8:1 GQA head layout and head dim 64.
- Splits: `num_splits_for` (FA4 heuristic x 2), except sliding B16 → 4.
- Everything else stays on r1.

**Open levers.**
- An FP8 PV with E4M3 P (stock numerics) would remove the V upcast, at most about 30 µs at verify32. It needs a
  Vᵀ byte transpose in the producer and a key permutation of the P fragment (the v5 trick). Not done.
- B=1 full shows no gain with this split heuristic.

**Judge clone, attention problem** (`box_out/fa4v2_judge.log`, `fa4v2_judge.json`; route file as the submission,
lazy prepare on the first eager call):
- **16/16 tests pass. Speedup x1.3226 [1.303, 1.342].**
- Verify loads (routed): x1.588 / 1.523 / 1.515. Decode loads (r1 rung-1 levers only): x1.246 / 1.107 / 1.059.
- Per shape:

  | shape | stock | ours | speedup |
  |---|---|---|---|
  | ver1 sliding | 24.3 | 12.9 | x1.88 |
  | ver16 sliding | 29.7 | 17.6 | x1.69 |
  | ver32 sliding | 36.0 | 21.5 | x1.67 |
  | ver16 full | 205.0 | 140.1 | x1.46 |
  | ver32 full | 390.1 | 263.8 | x1.48 |

- For the attention-problem submission, the route should be combined with k1_v4's decode kernels.

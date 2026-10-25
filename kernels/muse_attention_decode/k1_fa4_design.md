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

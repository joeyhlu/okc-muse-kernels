

Baseline (t05 harness, `box_out/mv1_t05.log`, same process, raw µs, fuse 0):

| T | v2.4 | stock Marlin |
|---|---|---|
| 8 | 110.0 | 113.5 |
| 128 | 289.3 | 327.1 |
| 256 | 402.3 | 452.2 |
| 320 | 479.9 | 624.4 |

## 1. Where v2.4's time goes at these shapes (from measured numbers only)

Inputs: the routing stats (`consumer_probe.log` ROUTING; t05 uses the same Student-t busiest-rank generator), the
consumer probe (§2 of k2a_gap_analysis.md) and the box timings.

| T | touched | passes @16 | processed MB (passes × 28.31) | v2.4 µs − 5 floor | processed rate | consumer ceiling for 16-token passes |
|---|---|---|---|---|---|---|
| 8 | 8.4 | 8.4 (v0, 8-token) | 238 | 105 | 2.26 TB/s | 3.92 (4 mma / 16 B) |
| 128 | 22.3 | 26.8 | 758 | 284 | **2.67 TB/s** | 3.34 (8 mma / 16 B) |
| 256 | 24.5 | 37.7 | 1,067 | 397 | **2.69 TB/s** | 3.34 |
| 320 | ~25 | ~45 | ~1,274 | 475 | **~2.68 TB/s** | 3.34 |

1. **At T ≥ 128 the time is processed bytes at a constant ~2.68 TB/s, 80% of the 16-token consumer ceiling.**
   - Two terms explain all of it:
     - (a) hot-expert re-passes: +20% / +54% / +80% bytes over read-once at T = 128 / 256 / 320;
     - (b) a ~20% structure tax spread over many items.
   - Even a tax-free kernel at 3.34 TB/s would save only −55 µs at T=128.
2. **At T = 8 the rate is 2.26 TB/s.** The tax is larger there: ramp, three launches, small units, post-norm.

## 2. Candidate items (expected µs in t05's harness = my best estimate, then ×0.5 haircut per R7)

| # | item | how (fuse-0 compatible, no grid barrier) | best estimate raw µs at T=8/128/256/320 | **haircut** | effort | µs/h at T=128 (haircut) | evidence / risk |
|---|---|---|---|---|---|---|---|
| 1 | **Launch boundaries**: PDL across the 3 fuse-0 launches | `cudaLaunchKernelEx` + `ProgrammaticStreamSerialization`. Each grid triggers `griddepcontrol.launch_dependents` early. The next grid prefetches its first W2 / norm-weight lines into L2, then `griddepcontrol.wait`s (completion semantics: no co-residency, HiCache-safe). Captured into the graph as programmatic edges | −8 / −10 / −12 / −12 | **−4 / −5 / −6 / −6** | 2 h | 2.5 | Hides about 2 × (launch + ramp) plus part of each launch's tail; no measurement yet |
| 2 | **Unit tails**: dynamic work queue (atomic counter, reset by v2.4's exit-counter pattern), heaviest units first, plus remainder splitting of the last partial round where the flush is already atomic (W2 at KS2 > 1, W13 at KS1 > 1) | as listed | −6 / −10 / −10 / −12 | **−3 / −5 / −5 / −6** | 2.5 h | 2.0 | The naive count model promises −19% at T=128 (3.25 units per CTA → makespan 4). But the last-round CTAs run with ~1/4 of the SM's contention, so the real tail is about 1/3 of a unit. The v2.2 lab's quantization-aware split-K gave **no gain**; the traces' 15–35% median→max spread is mostly this self-healing round |
| 3 | **Activation staging + per-slice syncs**: double-buffered X slice (cp.async the next slice while consuming) and/or one slice per unit (larger XE in dynamic smem) | needs 64 KB smem/CTA, so 3 CTAs/SM | −5 / −8 / −10 / −10 | −2.5 / −4 / −5 / −5 | 3 h | 1.3 | **v2.2 measured a double-buffered step loop slower everywhere**; 3 CTAs/SM cost more than the staging bubble in lab L4 |
| 4 | **Post-norm launch removed**: the W2 launch does the norm per token with a last-arriver counter (each W2 unit bumps its pass's token counters; the CTA completing a token's last route normalizes it). Fences only, no waiting | as listed | −5 / −5 / −5 / −5 | −2.5 each | 2.5 h | 1.0 | Overlaps with item 1, which already hides most of that boundary |
| 5 | **Lever 2: hot experts read once**, as a 4th launch in fuse 0 for experts with > 32 tokens, at 32 tokens/pass (NC4, 3 CTAs/SM) or WG64 (2 CTAs/SM); cold experts stay on the v1 launch | as listed | 0 / −3 / −25 / −40 | **0 / −1.5 / −12 / −20** | 4 h | 0.4 (5 at T=320) | Probe model: a hot pass at NC4 costs 6.21 vs 2 × 4.79 per 16 B (−35% on hot work). Minus a launch + ramp (~8 µs). K2A2_HOT (64 tok/pass, mma.sync) and the WG path were both measured slower; at T=128 the hot share is too small |
| 6 | Scale loads, split-K atomics | — | < 1 | ≈ 0 | — | — | 0.5 u32 per 16 B; W13 is store-path at KS1 = 1 for T ≥ 128 |
| 7 | W13 → W2 hand-off | already bf16 `inter` at KS1 = 1 (cp.async staging); fuse 0 has no barrier to remove | — | (in 1) | — | — | mv1 showed what the FP32-h alternative costs |

A mixed in-kernel variant (hot NC4 units and cold NC2 units in one kernel) is not listed. Its register allocation
forces 3 CTAs/SM on the cold majority. Lab L2 (NC4 everywhere) measured 458 vs 391 µs at bs256.

## 3. Ranking and the Stage A kill

By µs per hour at T=128: 1 (2.5) > 2 (2.0) > 3 (1.3) > 4 (1.0) > 5 (0.4).

**Top two together, haircut: −5 + −5 = −10 µs at T=128, against the −25 µs threshold. KILL at Stage A.**
- Even unhaircut they reach about −20.
- The full structure tax at T=128 is only about −55 µs (2.67 → 3.34 TB/s), spread over five or more items.
- At T=128 the hot-expert lever is small (+20% bytes) and its per-pass cost penalty eats most of it.
- Items 1 + 2 would give about −9 to −12 µs per step-layer at T = 8–320 after the haircut, about 0.5–0.7 ms per
  60-layer step. That is real but small, and below the bar set for this stage.

If this is reopened, the best µs per hour is at T = 256 / 320 (c = 32): item 5 (−12 / −20 haircut) plus items 1 + 2.
That only makes sense if c=32 verify becomes a priority over prefill. 07's model says prefill dominates c = 16 / 32.

No Stage B was run, and there is no box time, no k2a_v25.py, and no commit.

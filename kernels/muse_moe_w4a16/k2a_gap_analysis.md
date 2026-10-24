# K2A gap analysis: how far 2× MoE is from physics (2026-10-24)

Inputs:
- The judge's exact routing: `box_out/consumer_probe.log` `ROUTING` lines. They come from generate_input on the GPU
  with the judge clone's seeds (seed + 42·(rep + 1), 100 reps per shape).
- The judge's SGLang µs: the mean of k2a_v21's 3 real judge checks.
- A consumer ceiling probe: `tools/box/consumer_probe.py`, log `box_out/consumer_probe.log`.
- "v2.2 judge-est" = box µs × (k2a_v21 judge µs / k2a_v21 box µs) per shape. Its geomean is 1.374×, which matches
  the ≈ 1.38 judge estimate. The coordinator's uniform box µs × 1.067 is shown after the slash.

## 1. Per-shape floors (bytes = touched × 28.31 MB: W13 + W2 + E4M3 scales, each touched expert read once)

| shape | touched | routes | MB | floor 4.3 TB/s + 3 µs | floor 3.02 + 3 | floor 3.92 + 3 (probe, §2) | v2.2 judge-est µs | SGLang µs | 2.0× target | 2.35× (#1) target | TB/s for 2× | TB/s for 2.35× |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bs1 | 2.58 | 2.6 | 73 | 20.0 | 27.2 | 21.6 | 50.0 / 50.8 | 78.2 | 39.1 | 33.3 | 2.02 | 2.41 |
| bs8 | 8.36 | 13.8 | 237 | 58.0 | 81.4 | 63.4 | 110.2 / 105.1 | 150.7 | 75.4 | 64.1 | 3.27 | 3.87 |
| bs16 | 11.73 | 25.7 | 332 | 80.2 | 113.0 | 87.7 | 140.3 / 143.4 | 187.0 | 93.5 | 79.6 | **3.67** | 4.34 |
| bs32 | 15.65 | 49.0 | 443 | 106.0 | 149.7 | 116.0 | 184.9 / 182.4 | 209.1 | 104.5 | 89.0 | **4.36** | 5.15 |
| bs128 | 22.33 | 192.6 | 632 | 150.0 | 212.3 | 164.3 | 313.8 / 309.3 | 465.4 | 232.7 | 198.0 | 2.75 | 3.24 |
| bs256 | 24.46 | 377.9 | 693 | 164.0 | 232.3 | 179.7 | 442.5 / 440.0 | 622.9 | 311.5 | 265.1 | 2.24 | 2.64 |

"TB/s for N×" = bytes / (target − 3 µs), the rate the whole call must sustain through the consumer.

Geomean speedup if every shape ran exactly at the floor:

| rate | geomean |
|---|---|
| 4.3 TB/s | 2.86× |
| 3.02 TB/s | 2.04× |
| 3.92 TB/s (probe) | 2.62× |
| v2.2 judge-est | 1.37× |

What each shape needs for 2×:

| shape | verdict for 2× |
|---|---|
| bs1 | Inside physics at 3.0 TB/s (needs 2.0). The cost is fixed overhead, not bytes: v2.2 is 50 µs against a 22–27 µs floor. |
| bs8 | Needs 3.3 TB/s: above 3.0, below the probe's 3.9. |
| bs16 | **Needs 3.7 TB/s**: above 3.5, below the probe's 3.9. |
| bs32 | **Needs 4.4 TB/s**: above the plain-load streaming ceiling (4.33 TB/s measured below). Impossible. |
| bs128 | Inside physics at 3.0 (needs 2.75). Needs hot experts read once: passes@16 = 26.8 vs 22.3 touched. |
| bs256 | Inside physics at 3.0 (needs 2.24). Needs hot experts read once: passes@16 = 37.7 vs 24.5 touched. |

**#1's implied time is below the 4.3 TB/s floor at bs32 (89 vs 106 µs).** At bs16 it is 80 µs against an 80 µs
floor, which is at the floor.
- I checked the byte assumption. The judge's routing touches exactly what is assumed: 15.65 experts at bs32, matching
  the CPU estimate of 15.5.
- No touched expert can be skipped: every output needs every row of W13 and W2.
- Codes are uniform random bytes (`nvfp4.random_weights`), so they are incompressible.
- Only the E4M3 scales compress, losslessly: log2 ~ N(7, 0.4) gives about 4 bits of entropy in 8. Scales are 11% of
  the bytes, so this saves about 5%.
- So #1's 2.349 is not uniform across shapes. The floors allow up to 3.9× at bs1, 3.1× at bs128 and 3.8× at bs256,
  but under 2× at bs32.
- A 2.35 geomean needs about 3.4–3.5 TB/s in situ on average.

## 2. Ceiling probe (one timed box call)

Setup: stream_lab part D's S1 consumer: plain LDG.128, 4 loads in flight per thread, 256 threads × 4 CTAs/SM,
persistent grid. Two measurements:
- 1 GiB: the asymptotic rate, median of 7.
- One 56.6 MB launch (3 × W13), cold L2, graph replay, raw µs including the ~4.8 µs replay floor.

| consumer per 16 B | regs | 1 GiB TB/s | 56.6 MB one launch (raw µs) |
|---|---|---|---|
| w0: stream only | 32 | **4.33** | 18.1 |
| w1m: Marlin LOP3 dequant + scale FMA (lab D's w1) | 50 | 3.69 | 22.0 |
| w2m: w1m + 4 mma.sync (lab D's w2, the "3.02" constant) | 59 | **3.62** | 22.5 |
| w1v: v2 deq4 + scale decode + bmul2 | 48 | 3.88 | 21.2 |
| **w2v: v2 decode + 4 mma.sync (one 8-token chunk, v0)** | 53 | **3.92** | 21.0 |
| w3v: v2 decode + 8 mma.sync (two chunks = v2.2's 16-token pass, v1) | 55 | 3.34 | 24.1 |

- **The "3.02 TB/s" constant was pessimistic.** The same Marlin consumer gives 3.62 TB/s here: the box, with a
  simpler chunk loop.
- v2's decode is about 8% better than Marlin's (3.92 vs 3.62).
- Doubling the MMAs per 16 B, as in v2.2's 16-token variant, costs 15% (3.34).
- The consumer is not what holds v2.2 back. v2.2 runs at these in-situ rates (bytes / judge-est µs):
  - bs1: 1.46 TB/s
  - bs8: 2.15 TB/s
  - bs16: 2.37 TB/s
  - bs32: 2.40 TB/s
  - bs128: 2.01 TB/s
  - bs256: 1.57 TB/s
- So 35–45% of the time at bs8–32 is structure tax. The probe has none of these: B fragments staged in shared memory
  and per-slice `__syncthreads`, unit decode and tails, scale loads, split-K atomics, the route prologue, the
  W13→W2 barrier, and the post-norm.
- At bs128/256 the hot-expert re-reads come on top of that tax.

## 3. Verdict: **KILL, per the rule set up front**

The rule: "if 2× at bs16/32 needs more than 3.5 TB/s through the consumer even with the better decode, K2A board
work stops".
- bs16 needs 3.67 TB/s and bs32 needs 4.36 TB/s. Both are above 3.5, and bs32 is above the 4.33 TB/s streaming
  ceiling.

One nuance for the decision record: the rule is stricter than a 2× MoE *board* score needs.
- A 2.0× geomean needs about 3.0 TB/s in situ at every shape, not 2× at every shape. The floor geomean at 3.02 TB/s
  is 2.04×.
- The probe says the consumer can sustain 3.9 TB/s.
- So 2× geomean is inside physics, but only by removing most of the structure tax at every shape *and* the
  hot-expert re-reads. That is a full rewrite, and the WG attempt (k2a_v23.md) shows how expensive a structural
  change is.

Levers if the work were continued, by expected judge µs saved (estimates, haircut ×0.5 per R7):

| lever | bs1 | bs8 | bs16 | bs32 | bs128 | bs256 | basis |
|---|---|---|---|---|---|---|---|
| 1. Structure tax down to 3.3 TB/s in situ (B fragments from registers/broadcast, fewer syncs per slice, no split-K atomics at small T) | −8 | −18 | −18 | −24 | −30 | −40 | gap between in-situ and the probe; split unknown without ncu |
| 2. Hot experts read and dequantized once (mixed kernel, cold units on mma.sync) | 0 | 0 | 0 | −2 | −15 | −40 | passes@16 vs touched; WG16 penalty measured |
| 3. Fixed overhead at bs1 (route + post-norm + barrier, ~20 µs) | −7 | −3 | −2 | −2 | −2 | −2 | v2.2 trace: norm ~6 µs, route ~1.5, barriers |
| 4. Lossless scale compression (4-bit codes + escape) | −1 | −3 | −4 | −5 | −7 | −8 | 5% of bytes at the bandwidth limit |

All four together: bs1 34, bs8 86, bs16 116, bs32 152, bs128 260, bs256 352 µs, a geomean of about **1.75×**. That is
short of 2× and #50's 1.92.

## 4. k2a_v24.py (M1 hardening, nonce `k2a-v24-1`)

v2.4 is v2.2 plus M1 hardening (review_scored_v2.md M1). Plans, numerics and the timed kernel paths are unchanged
apart from four small additions:
- **Exit-counter reset.** The last CTA out of each launch resets the grid-barrier counter. Each CTA increments an
  exit counter exactly once, with no waiting, so this works whatever happened at the barriers.
- **Error flags.** A barrier timeout sets a per-call error flag, plus a sticky one that prepare's self-test reads.
- **NaN output.** The post-norm writes NaN to the whole output when the per-call flag is set. The last CTA of the
  post-norm launch then clears that flag.
- **h cleanup.** After a timed-out call, the last CTA re-zeroes the W13 split-K accumulator rows `h[0:R)`.

Control ints, one tensor: `ctl[0:2]` barrier (count, generation), `ctl[2:6]` = per-call flag, probe sink, sticky
flag, exit counter.

Judge clone on the box, three versions measured:

| version | log | speedup | bs1 | bs16 | bs32 | bs8 | bs128 | bs256 |
|---|---|---|---|---|---|---|---|---|
| v2.2 (`v22b.log`) | | 1.4707× [1.449, 1.493] | 47.63 | 134.37 | 170.94 | 98.51 | 289.84 | 412.41 |
| v2.2 code paths (`v23_judge2.log`) | | 1.4687× | 47.93 | 135.38 | 172.36 | 98.63 | 291.15 | 413.83 |
| v2.4a: per-call memset node | `k2a_v24_judge.log` | 1.4462× [1.425, 1.468] | 49.19 | 136.39 | 173.55 | 101.78 | 291.97 | 413.84 |
| v2.4b: exit-counter reset, no h cleanup | `k2a_v24a_judge.log` | 1.4563× [1.435, 1.478] | 48.45 | 136.18 | 172.90 | 99.91 | 293.02 | 415.22 |
| **v2.4 (shipped): v2.4b + h cleanup** | `k2a_v24_judge3.log` | **1.4480× [1.426, 1.470]** | 49.25 | 137.49 | 174.38 | 101.16 | 291.86 | 414.98 |

- The memset node costs about 1–3 µs per call, which is −1.7%. That is why v2.4 resets in-kernel instead.
- v2.4b is −0.8% against the v2.2-code run, inside the ±1.5% interval.
- The shipped v2.4 is −1.4% (1.4480 vs 1.4687). It is slower on all six shapes, by 1–2.5 µs per call, both times it
  ran. That is just at the edge of the interval.
  - The likely cause: 528 same-address atomics on the exit counter at every launch tail. They serialize at L2, as the
    grid barrier's arrivals do.
  - **So M1 hardening is not free. It costs about 1–2 µs per call (−1 to −1.5%) whichever reset is used.**
- Recommendation for scoring: keep k2a_v22.
  - If hardening is wanted on the board, the zero-cost option is review M1(b) alone: NaN on a sticky error flag, no
    reset. After a timeout every later call is NaN, which fails loudly.
  - That option is not measured. It changes only the post-norm's stores plus one `__ldcg` per CTA.

Fault injection (`tools/box/k2a_m1_test.py`): a call launched with twice the co-resident grid, so that every barrier
times out, followed by two normal calls.
Logs: `box_out/k2a_m1_test_v24a.log` for the v2.2 and v2.4b rows, `box_out/k2a_m1_test.log` for the shipped v2.4.

| file | shape | timed-out call | the next call |
|---|---|---|---|
| v2.2 | bs16 (2 launches) | finite and wrong (rel 0.68) | correct |
| v2.2 | bs1 (1 launch) | finite and wrong (rel 1.1) | **wrong (rel 0.56)** |
| v2.4b | bs16 | 100% NaN | correct |
| v2.4b | bs1 | 100% NaN | **wrong (rel 0.58)** |
| **v2.4 (shipped)** | bs16 / bs1 | 100% NaN, `err=[0,0,1,0]` | **correct at both** (rel 3.75e-3 / 3.82e-3) |

- v2.2: the barrier counter happened to recover in this injection. The broken next call at bs1 comes from late W13
  atomics into `h` that land after the post-norm zeroed it.
- v2.4b: the flags are correct (`err=[0,0,1,0]`). The bs1 next-call failure is that same leftover `h`, which v2.4's
  h cleanup fixes.



1. **HiCache coexistence.**
   - The risk: HiCache's H2D/D2H transfer kernels run on another stream. A persistent-grid launch with in-kernel grid
     barriers then is not guaranteed co-residency. A barrier waits about 34 ms, then (v2.2) silently produces stale
     output, and the counter stays broken for every later call.
   - The design:
     - Keep `SGLANG_MUSE_K2A_FUSE=0` as the default: 3 launches, no grid barrier. It never waits on another CTA, so
       any co-scheduling is safe.
     - Its cost is small: lab_v2 knobs f0 vs f2 = +0.6 to +1.8 µs per call (bs8 +0.6, bs1 +1.4, bs128 +1.6, bs32
       +1.8). That is about +40–110 µs per 60-layer forward.
     - The measured per-bucket `plan` (1–2 launches with barriers) stays opt-in.
2. **3-launch no-grid-barrier default.**
   - Already the default in 05. Keep it, and document `plan` as safe only with M1 (item 3) and only when nothing else
     can occupy SMs: no HiCache, no overlap scheduler streams.
   - Do not regenerate with a per-call memset. On the box it cost about 2 µs per call (§4), about 120 µs per forward.
   - Point `gen_05_k2a.py`'s `SRC` at `kernels/muse_moe_w4a16/k2a_v24.py`. Its KERNEL_SRC differs from v2.2 only by
     the exit-counter reset, the per-call/sticky error flags and the NaN post-norm, so the byte-identity assertions
     carry over.
   - Replace the shared `bar`/`err` tensors with one 6-int `ctl`: `bar = ctl[0:2]`, `err = ctl[2:6]`.
   - The self-test reads the sticky flag, `err[2]`, instead of `err[0]`.
   - Log a one-line warning if `err[2]` is ever found set when a layer's self-test or the periodic health check reads
     it.

## Files
- `tools/box/consumer_probe.py`, log `box_out/consumer_probe.log`
- `kernels/muse_moe_w4a16/k2a_v24.py`; `tools/box/k2a_m1_test.py`; logs `box_out/k2a_v24_judge*.log`/`.json`,
  `box_out/k2a_m1_test.log`

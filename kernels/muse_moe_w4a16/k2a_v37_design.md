# K2A v3.7 design: dynamic tail queue + mailbox epilogue + warp-parallel prologue (2026-10-25)

Status: **design first, then build** (`k2a_v37.py`, bench `tools/box/k37_bench.py`). Base = v3.6 as shipped in
`k3_v15.py` (bs1/8: vid 0 = v3.3 NC1; bs16: vid 3 role split r8; bs32: vid 3 role split r12; bs128/256: vid 2 v3.3
HOTCAP). v3.7 touches only new code (HK 3 variants); vid 0–3 keep v3.6's kernels.

## 1. The measured losses (A2 lab, `box_out/a2_lab.log` traced rows `baseT`, layer routing)

| bs | K1 done first / p50 / last (per-SM spread) | K2 entry .. ; griddep done; 1st land p50 / max | K2 done first / p50 / last | hand-off | tail | ramp |
|---|---|---|---|---|---|---|
| 1 | 11.1 / 14.1 / 18.9 (6.0) | 12.0..19.1; 19.9; 22.0 / 24.9 | 26.2 / 27.6 / 30.5 | 7.3 | 3.6 | 4.0 |
| 8 | 33.1 / 40.3 / 44.1 (3.7) | 33.7..44.4; 44.9; 45.6 / 52.0 | 61.9 / 67.1 / 71.9 | 5.8 | 5.9 | 4.5 |
| 32 | 55.3 / 67.8 / 74.7 (6.8) | 56.1..75.6; 75.8; 77.2 / 83.7 | 104.8 / 111.5 / 117.4 | 6.8 | 6.5 | 5.5 |

Reading:
- **K1 finish spread is per-SM rate + first-land skew**, not work: every CTA has the same stage count (31–55), entry skew
  ≤ 0.9 µs, but first land spreads 2.5–3 µs and per-SM ingest is ±6% (31–35 GB/s). K2 waits (griddepcontrol.wait) for
  the LAST K1 CTA, so the K1 spread is paid in full at the hand-off.
- **K2's tail is the late entrants**: the K2 CTAs that replace the last K1 CTAs start ~7 µs after the others' first land
  (route up to 3.6–4.3 µs, 1st issue +2.2–2.5, first copy 3.7–6.4 µs under full load) and still own a full static
  share (16–28 stages). The slowest K2 CTAs are exactly those (entry 73–75.5 µs at bs32).
- **Ramp**: K1 route ~2.0–2.3 µs + "1st issue" +1.1–2.2 µs before the first copy. The latter is serial thread-0 /
  per-warp code: `segs36` (role-split cost loop over 32 experts with 64-bit math, two `tail_first2` walks of up to 32
  dependent smem loads) and each role's `while (sb[e + 1] <= x) ++e` from e = 0 at every segment start
  (~35–40 clk per step at ~1.75 GHz → 0.5–0.8 µs per walk). K2 pays the same plus a routing scatter it never uses.

## 2. Changes (all in new device code; HK 3 variants)

**(H5a) Dynamic tail queue, K1 and K2.** Per kind sequence (hot `sbh`, cold `sbc`), the last
`nd = D·floor(f·S/D)` stages are a pool of D-stage chunks (D ∈ {2, 4}; D | 16 and pool start D-aligned, so a chunk
never straddles an item). v3.6's static plan (role split or dual range, tail-first order) covers `[0, S − nd)` of each
kind. After its static part, a CTA's producer takes tickets (`atomicAdd`, next ticket prefetched one chunk ahead):
hot chunks first, then cold; any CTA takes any kind (cross-role stealing also absorbs the role split's cost-model
error). `f` per kernel (`f1`, `f2` in 64ths) and `D` are runtime (packed in `probe` bits 8+), so the bench sweeps them
without rebuilds; `f = 0` = v3.6's static plan on the new code (isolates H1 + mailbox).
- Producer → consumers: per-slot tag in smem, written before the full-barrier arrive; END = tag −1 delivered through the
  ring (33 arrivals, no tx). Consumers read the tag after the full wait of the chunk's first stage.
- Every chunk is a split item: FP32 reds into h (K1) / y (K2) and v3.3's item counter (Σ stages = 16 → the last
  contributor finalizes: SwiGLU → interS, or Σz²).
- Ticket counters `meta[120..123]` (K1 ticket, K1 done, K2 ticket, K2 done): the producer that takes the grid's last END
  (done counter reaches G) resets both of its kernel's counters. Self-contained per kernel: any launch mask, any
  predecessor; the next launch sees zeros at the kernel boundary.

**(H5b) Mailbox epilogue.** All consumer → epilogue-warp hand-offs (static split items, K1 NOSWI hot items, chunks)
go through a 4-entry smem mailbox with mbarriers (`efull` 256 arrivals = every consumer thread after its reds;
`eempty` 1). Same release/acquire chain as v3.3's `bar.arrive`/`bar.sync` + `__threadfence`, but no named-barrier id
budget (ids ≤ 15 capped v3.6 at M ≤ 48 and 4 split items per CTA). HK 3 therefore serves every M ≤ 320.

**(H1) Warp-parallel prologue.** `find_e(sb, x) = popc(ballot(sb[lane + 1] ≤ x))` replaces every serial expert walk at
segment / chunk starts; `segs37` runs on warp 0 (lane = expert: Wh/Wc of the role split by warp reduction).
K2's `route` skips the scatter pass (K2 never reads `s_list`: its X rows are contiguous route positions).

**Kept**: v3.6 numerics (cold = v3.3 NC1 consume + flush, hot = hc_item; same epilogue formulas; stock-match
post-norm in K3), bounded waits (flag + return; a timed-out wait in a dynamic loop also exits the loop), PDL chain (K1
normal launch, K2/K3 PDL, launch_dependents at entry), evict_first hint, the route table with fallback, self-test.
**Determinism**: v3.6 is not bitwise deterministic either (Σz² by 32 FP32 reds per route; ≥ 3-way splits at bs1).
v3.7 adds ≥ 3-contributor items in the pool (order of FP32 reds varies with ticket order): same ulp class, within the
self-test bounds.

Not done (reasons): K2 prefetch beyond its 3-slot ring (the HK2 ring is at the 2-CTA smem limit; the queue absorbs
the late entrants instead); expert-major K1→K2 overlap with per-expert counters (a W2 stage needs 4 W13 row blocks
that end at neighbouring CTAs' range ends, i.e. ≈ the same time as the grid wait; K1/K2 cannot co-reside by
registers); upstream routing compaction (needs the layer's router kernel: integration item for k3_v16).

## 3. Variants, plan, prediction

| vid | (NC, NST, CTAs/SM, HK, NCH) | role |
|---|---|---|
| 0–3 | v3.6's | unchanged (fallback) |
| 4 | (1, 3, 2, 3, 1) | v3.7 cold-only (bs 1/8; NCH 1 = vid 0's 22.6 KB slots) |
| 5 | (1, 3, 2, 3, 4) | v3.7 hot path (bs 16/32, role split; bench rows at 128/256) |

Plan (bench decides f/D): bs1 → vid 4 (f1 16, f2 24, D 2), bs8 → vid 4 (f 12/20, D 4), bs16 → vid 5 r8, bs32 →
vid 5 r12 (f 12/20, D 4), bs128/256 → v3.6's vid 2 unless the bench row wins.

Prediction (raw, layer routing, per call): K1 spread to ~1 chunk + finalize: −1.5…−3; K2 tail: −2…−4; prologue:
−0.5…−1.5 (K1, on the critical path) and part of K2's late-entrant ramp; chunk costs (+reds, end-of-kernel finalize
~1–2 µs chain): +0.5…+1.5. **Net −3…−7 µs at bs 8/16/32, −2…−4 at bs1**; ×0.5 (R7) → −1.5…−3.5.

## 4. Gates

1. sass_gate on the HK 3 kernels: ≤ 96 regs, 0 spill, no C75xx (C7519 allowed as before).
2. Self-test (`k37_bench.py check`, `prepare()`): every bucket plan and every bench row on k2a_v36's cases (now incl.
   bs 100/128/257/320 for HK 3), two calls in a row (counter reset), err flags 0.
3. Timing (`k37_bench.py run`, paired, one process): Δ(v3.7 plan − v3.6 shipped plan) ≤ **−3 µs at bs1, 8, 16 and
   32** (layer routing), no bucket slower (128/256 included; else that bucket keeps v3.6). **Kill: mean Δ > −1.5 µs.**

## 5. Results (4 serverless jobs, judge image, H200; `box_out/k37_gate*.log`, `k37_check*.log`, `k37_run1..4.log`)

**Compile gate (job 4, final code)**: `k1_kernel37<4,3>` 89 regs, `k2_kernel37<4,3>` 95, `<1,3>` 78 / 75; 0 spills, no
C75xx. (The only FAILs are v3.3 HOTCAP's spills, already shipped.) Job 1–3 had an 8 B (K1) and then a 4 B (K2) spill.
The fix: K1 recomputes the mailbox arguments after the item, K2 keeps them. **Check**: PASS in all 4 jobs (every row, every
k2a_v36 self-test case up to M = 320, one layer and one moe routing per bs, two calls, err 0).

**Timing, layer routing, median of 3 routings, d = µs per call vs v3.6 as shipped (job 4; jobs 2–3 agree ±1–2 µs):**

| bs | v3.6 ship | f0 (HK3 static) | best queue row | v3.7 plan | Δ plan |
|---|---|---|---|---|---|
| 1 | 29.8 | +4.0 | k1q4 +2.6 | = v3.6 (vid 0) | −0.5 (noise; s37 +0.5) |
| 8 | 68.0 | +0.8 | k1q4 +1.3 | = v3.6 (vid 0) | −0.3 |
| 16 | 71.3 | +1.0 | k1q8 +4.8 | = v3.6 (vid 3 r8) | −0.5 |
| 32 | 103.0 | +0.7 | k1q4 +0.1 | = v3.6 (vid 3 r12) | −0.5 |
| 128 | 224.2 | −9.8 | q4_8 −14.7, k1q8d8 −14.5 | vid 5 + K1 queue 8/64 D8 | **−13.9** (moe −5.4) |
| 256 | 314.8 | +44.9 | q4_8 +24.4 | = v3.6 (vid 2) | +0.7 (noise) |

- **Gate (≤ −3 µs at bs1/8/16/32): MISS. Kill (mean > −1.5 µs): fires for the bs ≤ 32 target.** The queue loses at bs ≤ 32 in
  every setting tried: f 4–24/64, D 2/4/8, K1-only, K2-only, cold-only pool, and the pool taken after the tail-first
  item (+10…+24 µs).
- Why (inference from the sweep, not traced): every round trip issued from a streaming SM waits behind the SM's
  in-flight bulk data (≈ one ring, ~3 µs, = the measured copy latency). A pool chunk adds the ticket atomic and, in the
  epilogue warp, fence + counter atomic + h/y readback for each chunk, serially. That is 2–4 round trips per chunk
  against a balancing gain of ≈ the per-SM spread (4–7 µs per kernel). It pays only where a CTA has hundreds of stages
  (bs128). The HK3 static path itself costs +0.7…+4 µs at bs ≤ 16 against v3.6 (mailbox / END protocol / code size, not
  isolated).
- **By-product that pays**: the mailbox epilogue removes HK2's M ≤ 48 limit. At bs128 (verify 16), HK3 NCH4 with the
  role split beats v3.3's HOTCAP wgmma (vid 2, which spills): −9.8 µs static. With a K1 tail queue the gain is −13.9 µs
  on the layer routing and −5.4 µs on moe, reproduced in 3 jobs (−12.6…−15.5 / −1.6…−5.7). bs256 stays on vid 2.
- H1 (warp-parallel prologue, K2 without scatter) is not separable in these rows: f0 ≈ v3.6 at bs32 (+0.7…−1.3 across
  jobs).

# K2A v3.8 design: the K1→K2 boundary and the K2 tail (2026-10-25, laptop only)

Status: **design + probe plan**. No GPU was used. Every number below is measured unless it is marked as a model.
Model gains are haircut ×0.5 (R7). New measurements in §1 come from re-analysing the A2 lab traces
(`box_out/a2_lab/traces.json.gz`, v3.3 code, `baseT` / `mathoffT` / `nullT` rows), using `tools/box/a2_lab.py convert`.

**Verdict.**
- The largest remaining MoE structure at bs 1/8/16 is the **boundary chain**: the K1 end, the K2 start and the K2 end.
  With math on it costs 8–9 µs per call more than the null consumer (A2 WASTE: hand-off + tail).
- At bs32 the largest item is still the hot-expert pass. It costs about 17 µs above the touched-bytes model, but its
  root cause is unknown (η ≈ 0.75 per extra chunk, measured), so it is a lab item (L3), not a 1-day build.
- **v3.8 = five small, switchable boundary fixes (B1–B5).** They add no new cross-CTA protocol. B1/B3/B4/B5 keep
  numerics bit-identical; B2 keeps the same ulp class.
- Expected: **−1.5…−2.5 µs per call at bs8–32 and −1.5 at bs1 after the haircut, i.e. +0.6…+0.9% decode_layer**
  (1.704 → ~1.715). This does not change the top-50 picture: #50 at 1.9065 needs +11.9%.

## 0. Gain model with today's per-load numbers (k3_v18 scored, 08588c0fe95444b3)

Weighted µs per load = 46 × sliding + 16 × full (judge µs). Saving 1 µs per MoE call (both layer types, weight 62)
changes the 6-load geomean by:

| MoE call T | load | weighted µs | layer gain per µs |
|---|---|---|---|
| 1 | dec1 | 5,854 | **+0.178%** |
| 8 | ver1 | 9,036 | **+0.115%** |
| 16 | dec16 | 11,558 | **+0.090%** |
| 32 | dec32 | 14,906 | **+0.070%** |
| 128 | ver16 | 21,768 | +0.048% |
| 256 | ver32 | 30,923 | +0.033% |

Per-call baselines (box µs, layer routing, plans shipped in k3_v18):

| bucket | baseline | plan |
|---|---|---|
| bs1 | 29.2 | vid 0 = v3.3 NC1 |
| bs8 | 68.0 | vid 0 |
| bs16 | 71.3 | v3.6 vid 3, r8 |
| bs32 | 103.0 | v3.6 vid 3, r12 |
| bs128 | 210 | v3.7 vid 5 + K1 queue |

## 1. What the A2 traces say about the boundaries (new analysis, v3.3 code, layer routing)

| quantity (math on, `baseT`) | bs1 | bs8 | bs16 | bs32 | math off / null |
|---|---|---|---|---|---|
| WASTE hand-off / tail (µs) | 7.3 / 3.6 | 5.8 / 5.9 | 6.3 / 6.9 | 6.8 / 6.5 | null 2.0–4.1 / 0.5–2.2 |
| last consume after last land (µs, p50) | 1.05 | 1.06 | 1.08–1.13 | 0.98–1.26 | 0.27–0.34 |
| **epilogue end lag**, max exit − max consumer end, K1 + K2 (µs) | **2.2 + 2.2** | 0 + 1.7 | 0.2 + 3.4 | 0 + 2.3 | no epilogue |
| epilogue busy per split part, p50 (µs) | 9.9 / 10.2 | 12.8 / 12.6 | 14.1 / 12.5 | 13.9 / 15.2 | — |
| epilogue busy / CTA life, K1 / K2 | 0.93 / 0.89 | 0.64 / **0.95** | 0.69 / **0.92** | 0.48 / 0.68 | — |
| K2 entry order vs blockIdx (rank corr.) | 0.998 | 0.999 | 0.999 | 0.998 | same |
| K2 exit vs K2 entry (rank corr.) | 0.28 | 0.74 | 0.83 | 0.83 | — |
| K1 exit vs blockIdx (rank corr.) | 0.53 | 0.78 | 0.86 | 0.83 | 0.20–0.48 |
| K2 1st land p50 → max (µs) | 22.0 → 24.8 | 45.6 → 51.9 | 48.7 → 56.7 | 76.8 → 83.4 | — |
| **bound: late K2 entrants land at p50 + 0.5 µs** (Δ max exit, µs) | 0.0 | **−2.2** | **−2.6** | **−1.5** | moe routing −1.0…−4.0 |
| K2 1st→2nd land gap after the wait vs steady (µs) | — | 2.60 vs 0.97 | 2.07 vs 0.91 | 1.90 vs 0.97 | — |
| consumer flush per unit, 1st / later, p50 (p90) (µs) | 0.30 / 0.25 (0.85) | 0.57 / 0.38 (1.40) | 1.04 / 0.50 (2.45) | 0.83 / 0.50 (1.64) | 0.10 (probe skips it) |
| K2 route prologue p50 / max (µs) | 1.38 / 2.66 | 1.94 / 3.88 | 2.03 / 3.46 | 2.04 / 3.59 | — |

Reading:
1. **Split items are the expensive structure.**
   - Each split part costs the epilogue warp 10–15 µs: bar → `__threadfence` → counter `atomicAdd` → (if last:
     fence → h/y readback → finalize). Each of these round trips queues behind the SM's in-flight bulk data, which is
     ~2–3 µs at the measured copy latency of 2.0–2.4 µs p50.
   - In K2, and in K1 at bs1, the CTA life is short, so the single epilogue warp saturates (0.9–0.95 of the life).
     The CTA then exits 1.7–3.4 µs after its consumers have finished.
   - The v3-era probe `k3_lab5` (K1 alone) shows the same thing: storing split parts like whole ones (probe 22) saved
     **4.5 µs at bs1 and 5.2 µs at bs8**. Skipping only the epilogue (probe 23) saved 3.6 / 4.2 µs.
   - This is also why v3.6's dual range and v3.7's queue lost: both add split items.
2. **The K2 tail is the late entrants, and they are known statically.**
   - K2 CTAs enter in blockIdx order, so the high-blockIdx CTAs enter last.
   - Their first copies queue behind the HBM burst: 1st land up to 6–8 µs after p50.
   - Their exits are 2.3–3.6 µs (p50) later than everyone else's.
   - K1 exits also follow blockIdx with math on (0.78–0.86), because first-land order follows blockIdx and nothing
     rebalances it.
3. After the wait, every K2 CTA loses about 1–1.6 µs before its 2nd stage. That is the first unit's flush, whose
   per-expert `__ldg(gsc2 + e)` and post-norm weight loads are loaded round trips, on top of the X landing.
4. During the hand-off, HBM has about hand-off × 4.4 TB/s ≈ 25–30 MB of unused capacity. On the bs8 layer trace the
   ingest bins fall from 4.0 → 3.4 → 3.3 → 2.1 → 0.1 → 1.5 → 0.6 TB/s, which is 31 MB unused against 4.4 TB/s.
   The early K2 CTAs sit idle in exactly this window, after only 3 smem stages of prefetch.

## 2. Ranked levers (µs per call: raw model → after ×0.5; layer gain uses §0)

| # | lever | evidence (measured) | bs1 | bs8 | bs16 | bs32 | bs128 | layer, haircut |
|---|---|---|---|---|---|---|---|---|
| **1** | **v3.8 boundary package B1–B5 (§3)** | §1 rows 1–3, 7–12 | −3.1 → **−1.5** | −5.0 → **−2.5** | −4.5 → **−2.2** | −3.4 → **−1.7** | −3.0 → −1.5 | **+0.6…+0.9%** |
| 2 | Cluster split-K (DSMEM pairs/quads, at most 1 global split per CTA; v3.9) | probes 22/23 (−4.5/−5.2 µs, K1 alone, v3); end lags §1; overshoot at C=4: K1 3.1%, K2 5–18% (28 A2 routings) | −3…−5 | −1.5…−4 | needs HK2 smem | needs HK2 smem | — | +0.1…+0.4% beyond B5 |
| 3 | Hot-pass η lab (bs16/32): why a 4-chunk hot stage costs 3.25 cold stages (r12) when issue, HMMA and smem all bound it at ≤ 1.4 | k36 bench: r12/r8 best, η_eff 1.2–2.5 in dual | — | — | −3.5 (η .25) | −11.7 (η .25) | small | EV +0.2% (p ≈ 0.35) |
| 4 | Upstream routing compaction + K1 as PDL secondary of `v8::tail_kernel` | K1 route p50 1.97–2.04 µs + 1st issue 1.1–1.4 µs; ramp 4.0–5.5 µs | −1.5…−2.5 at every bs | | | | | +0.4% (k3-side, G2 risk: in-kernel tails cost what launches did) |
| — | X-ingest: stage X once per expert, DSMEM multicast, pre-gathered X (candidate a) | mathoffNX vs mathoff: −0.6 (bs32 layer), +0.2 (bs16), −0.6 (bs8), −1.6 (moe bs32). The WASTE `xing` column (5.7 / 10.8) is bytes at the steady rate, an attribution only | ≤ 0.6 | | | | | **closed** for cold experts; hot-stage X goes into lever 3's lab |
| — | Fused K1/K2 with per-slice flags (candidate b) | v3.4: 0.976–1.002. In stream-K order roughly one item per CTA is that CTA's last one, so every 4-row-block slice completes at the K1 end. A slice-major dual range doubles the split items (10–15 µs of epilogue each) | | | | | | **closed** until splits are cheap (after lever 2) |
| — | Math tax / clock (candidate c): fewer HMMA per byte, or SM role split to hold 1980 MHz | v3.5: any mma.sync stream sets 1790 MHz (mma on raw words: 1794 at bs16 with mostly empty columns). Decode alone at full clock costs as much. v3.1: INT-op cut, X smem, ring cycle and warp count all had no effect. The clock cap is chip-wide, and an FFMA path is 2.3× the ALU work (issue-bound) | | | | | | **closed** |
| — | Deeper ring / early slot release | v3.1: NST 2–4 within ±3% (deeper worse); early release slightly worse | | | | | | closed |
| — | Dynamic tail queue | v3.7: +0.1…+4.8 µs at bs ≤ 32 | | | | | | closed (bs128 shipped) |

Public code (candidate d):
- Marlin-MoE uses lock-serialized global split-K reduction, the same cost class as our counters.
- CUTLASS sm90 stream-K uses a global workspace plus a flag fixup.
- DeepGEMM's masked grouped GEMM has no split-K and uses cluster TMA multicast of operands.
- FlashInfer xqa and CUTLASS `SM90_BULK_PREFETCH` use `cp.async.bulk.prefetch.L2.global`, which is the tool for B1.
- Megakernels (Hazy, MPK) use per-task counters, i.e. lever b, which pays only when there are no split items.
- Nothing public restructures small-batch W4A16 MoE decode beyond this. The borrowable pieces are L2 bulk prefetch
  (B1) and cluster/DSMEM reduction (lever 2).

## 3. Chosen design: K2A v3.8 = B1–B5 on the shipped kernels (vid 0 bs1/8, vid 3 bs16/32, vid 5 bs128)

All five are runtime-switchable through one new `Params` field, `v38`, packed as below. With every field 0 the
kernels behave like v3.7 bit for bit. Base file: `k2a_v37.py`, the module shipped in k3_v18.

| bits | field | meaning |
|---|---|---|
| 0–3 | P | B1 own-prefetch depth, 0–15 stages |
| 4–10 | H | B1 assist CTAs, 0–127 |
| 11–16 | α2 | K2 tilt, in 1/64 |
| 17–22 | α1 | K1 tilt, in 1/64 |
| 23 | B3 | flush hygiene on |
| 24 | B4 | K2 skips the scatter |
| 25 | B5 | dual epilogue (vid 0 builds only) |

**B1. K2 pre-wait L2 staging.**
- Where: the K2 producer, in `gemm` and `gemm36` (warp 8, lane 0). It runs after the NST pre-wait smem copies and
  before `pdl_wait()`.
- Lane 0 walks its own stage sequence on to stages [NST, NST + P) and issues one
  `cp.async.bulk.prefetch.L2.global.L2::cache_hint [src], 18432, pol_last` per stage. `pol_last` comes from
  `createpolicy.fractional.L2::evict_last`; the K1 weight stream stays `evict_first`, so these lines are evicted last.
- **Assist:** CTA b < H also computes the range of CTA b' = b + G − H, using the same range function (`segs36` for the
  role split), and prefetches b''s stages [0, NST + P). Those are the late entrants identified in §1.
- Budget: prefetched bytes ≤ min(24 MB, the K2 range) per call. At P = 4 and H = 66 that is
  264 × 4 × 18 KB + 66 × 7 × 18 KB = 28 MB, so cap P at 3 when H > 0.
- No smem, no mbarrier, no waits: prefetches are fire-and-forget hints.
- Numerics: unchanged.
- Model: the late-entrant bound (§1) is −2.2 / −2.6 / −1.5 µs at bs8/16/32. Own staging gives
  P × (τ_cold − τ_warm) per CTA, with τ_warm measured by J1; conservatively −1 µs at P = 4.

**B2. blockIdx-tilted static ranges, K1 (α1) and K2 (α2).**
- Stream-K boundaries become B(c) = ⌊S · (c/G + α·c·(G − c)/G²)⌋, monotone for |α| < 1, with B(0) = 0 and B(G) = S.
- The role split applies it inside each role's sequence.
- Low blockIdx CTAs, which enter or land first, take more stages.
- Model: a first-order replay on the traces gives K2 −1.2…−3.7 µs in-sample, but it is mixed out of sample. B2 overlaps
  with the B1 assist, so count max(B1-assist, B2), not the sum. The bench sweeps α1 ∈ {0, 3, 6}/64 and
  α2 ∈ {0, 4, 8}/64.
- Numerics: split points move, so the FP32 order inside split items changes. This is the same class as today's
  per-bucket plans; the self-test bounds are unchanged.
- Emulator: extend `k2a_v36_emulate.py`. Every stage must be covered exactly once at α ∈ {−8, 0, 8, 16}/64, and the
  barrier-id budget must hold. The negative control is a non-monotone α = 70/64, which must be caught.

**B3. Flush hygiene.**
- Stage `gsc13[0..31]` / `gsc2[0..31]` into static smem (+256 B) in the route prologue. All flushes then read
  `gs = s_gsc[e]`: an LDS instead of a loaded `__ldg`.
- vid 0 only (74 registers): load the K2 whole-item post-norm weights `w0`/`w1` (2 floats per lane) at item start,
  before the stage loop. HK2 K2 already sits at 96 registers, so it keeps the original load.
- Numerics: identical values.
- Model: §1 row 11, the K2 first-gap excess (0.9–1.6 µs) of which ~0.4–0.7 is flush, plus K1 first-flush p90 of
  1.4–2.45 µs. Total −0.6…−1.2 µs per call.

**B4. K2 skips the routing scatter.**
- Port v3.7's `route<…, true>` (HK3 already does this) to the vid 0 and vid 3 K2 kernels. K2's X rows are contiguous
  route positions, so `s_list` is never read.
- Model: it removes part of the 1.4–2.0 µs p50 (max 2.7–3.9) K2 route time on the late entrants' path. That is
  counted inside the B1-assist bound, not added to it.

**B5. Dual epilogue warps, vid 0 only (bs1, bs8).**
- Add warp 10 (352 threads, `__launch_bounds__(352, 2)`, register cap 93; vid 0 is at 73/74). Split part n is handled
  by epilogue warp 9 + (n & 1).
- Named barrier ids are unchanged (2 + n), and the `bar.sync` count stays 288 (256 consumer threads plus one epilogue
  warp).
- Two chains then run in parallel, so the epilogue no longer saturates.
- Model: the end lags (§1 row 3) are bs1 K1 2.2 + K2 2.2 and bs8 K2 1.7; half of that is −2.2 / −0.85 µs.
  HK2 cannot take the extra warp (352 × 96 × 2 > 64K registers).

Per-bucket model (raw, before the haircut). B1-assist and B2 overlap, so the larger of the two is used:

| bs | B1 assist / own | B3 | B5 | total raw | after ×0.5 |
|---|---|---|---|---|---|
| 1 | 0 / −0.3 | −0.6 | −2.2 | −3.1 | −1.5 |
| 8 | −2.2 / −1.0 | −0.9 | −0.85 | −5.0 | −2.5 |
| 16 | −2.6 / −1.0 | −0.9 | — | −4.5 | −2.2 |
| 32 | −1.5 / −1.0 | −0.9 | — | −3.4 | −1.7 |
| 128 (extrapolated: no A2 trace at T = 128) | −2.0 / −0.5 | −0.5 | — | −3.0 | −1.5 |

Layer gain: 1.5 × 0.178 + 2.5 × 0.115 + 2.2 × 0.090 + 1.7 × 0.070 + 1.5 × 0.048 = **+0.94%**. The terms overlap,
so plan on **+0.6…+0.9%**.

Bounded waits (R9): B1–B4 add no waits. B5 reuses the existing per-part `bar.sync` and the flag + return path.

Registers and smem:
- B1 is producer-only code. It shares the per-kernel register maximum with the consumers, so `sass_gate` must show
  HK2 K2 ≤ 96, 0 spills, and no C75xx (C7519 allowed).
- B3 adds 256 B of static smem. HK2 has 5 KB spare at 2 CTAs/SM.
- If HK2 K2 crosses 96 registers, drop B1-assist in HK2 to the own-prefetch only (no second range walk).

## 4. Probes before building (≤ 2 serverless jobs)

**J1: zero kernel edits.** It adds rows to `tools/box/a2_lab.py` `ROWS`. The tuple is
(kind, lab, vid, launch mask, probe, mode, mul, unit, desc); probe 22/23 is v3.3's existing `fmode` 6/7.

```
"base_k1": ("lab", 0, 0, 1, 16, 0, 7, 16, "K1 only"),
"f6":      ("lab", 0, 0, 3, 22, 0, 7, 16, "split parts stored as whole: split-cost ceiling (timing only)"),
"f6T":     ("lab", 1, 0, 3, 22, 0, 7, 16, "traced f6"),
"f6_k1":   ("lab", 0, 0, 1, 22, 0, 7, 16, "f6, K1 only"),
"f7":      ("lab", 0, 0, 3, 23, 0, 7, 16, "split reds kept, no epilogue (timing only; re-zero h/cnt/cnt2 after)"),
"k2c":     ("lab", 0, 0, 2, 16, 0, 7, 16, "K2 only, cold L2"),
"k2w":     ("lab2x", 0, 0, 2, 16, 0, 7, 16, "K2 twice in one graph; report t - t(k2c) = L2-warm K2"),
```

- `lab2x` is a 6-line `make_call` kind that calls `a2run` twice.
- After every `f7` replay set: `st["h"].zero_(); st["cnt"].zero_(); st["cnt2"].zero_()`. The CHECK rows run before
  the probes.

Run:

```
python tools/serverless/sls.py run 'python3 tools/box/a2_lab.py run --bs=1,8,16,32 --src=layer,moe --routings=3 --rows=ship,base,baseT,base_k1,f6,f6T,f6_k1,f7,k2c,k2w 2>&1 | tee box_out/a2b_lab.log' --timeout 2400
```

That is ~8 min of GPU (4 bs × 6 routings × 10 rows × 30 rounds). Readings:
- **τ_warm / τ_cold** = (k2w − k2c) / k2c, per stage. At bs1 W2 (18.9 MB) is fully L2-resident; at bs8 partly.
  - B1-own is kept if τ_warm ≤ 0.85 τ_cold at bs1 and ≤ 0.9 at bs8.
  - Otherwise P = 0, and only the assist is kept: its gain is first-land latency, not rate.
- **f6 − base, f7 − base, f6_k1 − base_k1** per bs: the split-cost ceiling in today's code. f6T's WASTE row gives the
  hand-off and tail with no splits, i.e. how much of §1's 8–9 µs excess is split-caused.
  - Lever 2 (v3.9) **go** if f6 − base ≤ −3 µs at bs1 and ≤ −2 at bs8. **Stop** if > −1.5 at both.
  - B5's ceiling = f7 − base at bs1/8. If f7 ≈ base, the reds and not the epilogue are the cost, and B5 is dropped.

**J2: B1 prototype only.**
- About 40 lines in `k2a_v37.py`'s K2 producers: the `v38` field, P and H only.
- A bench `tools/box/k38_bench.py` cloned from `k37_bench.py`, with rows ship (= v3.7 plan) and pf(P, H) for
  P ∈ {0, 2, 4} and H ∈ {0, 33, 66}.

```
python tools/serverless/sls.py run 'python3 tools/box/sass_gate.py kernels/muse_moe_w4a16/k2a_v38.py --src KERNEL_SRC_V3 --subst "@CASES3@=call:_gate_cases" "@ATTRS3@=call:_gate_attrs" "@WGSPECS@=call:_wg_specs" --allow C7519 --json box_out/k38_gate/gate.json && python3 tools/box/k38_bench.py check && python3 tools/box/k38_bench.py run --bs=1,8,16,32,128 --rows=pf 2>&1 | tee box_out/k38_pf.log' --timeout 3000
```

- The `--subst` set is the one used for k36/k37 (the k37 gate log `box_out/k37_gate4.log` lists the 15 functions).
- **pf(0, 0) must equal ship within ±0.3 µs** (H0). The full call includes K1, so a K1 slowed by prefetch traffic
  shows up here.

## 5. Gates and kill

| gate | condition |
|---|---|
| G-J2 (B1 alive) | best pf row Δ ≤ −1.0 µs at ≥ 2 of bs 8/16/32 (layer, median of 3 routings), and moe routing not slower by > 0.5 µs |
| Kill B1 | every pf row > −0.5 µs at bs8–32 |
| Kill v3.8 | B1 killed and J1 f7/f6 show < 1 µs of split/epilogue cost; then B2–B5's model is < −1 µs per call. Stop MoE at bs ≤ 32 and run lever 3's lab |
| Build gate | `sass_gate` PASS (HK2 K2 ≤ 96 regs, 0 spills, no C75xx except C7519) |
| Build self-test | every bucket plan × {B-bits on}; rel-L2 < 1e-2, worst row < 5e-2, err 0, two calls (counters restored) |
| Emulator | B2 ranges at α ∈ {−8, 0, 8, 16}/64, with negative controls |
| Ship gate (k38_bench, paired, same process; layer routing, median of 3) | Δ(v3.8 plan − v3.7 ship) ≤ −1.5 µs at bs8, 16 and 32, ≤ −1.0 at bs1 and bs128; no bucket slower by > 0.3 µs (else that bucket keeps v3.7 through the route table) |
| Kill after build | mean Δ over bs 1–32 > −1.0 µs |
| In-layer | `layer_ab2` k38_all ≥ ×1.004 → k3_v19 |

## 6. One-day build plan (after J1/J2)

1. `k2a_v38.py` = `k2a_v37.py` + the `v38` field. B1 (from J2), B2 (range functions in `route`/`segs36`/`ranges36`),
   B3, B4 (route template flag), B5 (vid 0 builds: 352 threads, epilogue parity).
2. `k2a_v36_emulate.py`: add B2 coverage and the B5 part-parity map, with negative controls.
3. `k38_bench.py`:
   - sweep P {0, 2, 3}, H {0, 66}, α1 {0, 3, 6}, α2 {0, 4, 8}, B3, B4, B5;
   - the plan comes from the layer routing; moe must not lose > 0.5 µs.
4. One serverless job: gate + check + bench. One more: `layer_ab2` + judge clone. Then hand to k3_v19.

## 7. Next after v3.8

- **v3.9 (lever 2, if J1 says go).** K1/K2 launched as 4-CTA clusters with PDL.
  - Each item split inside a cluster is reduced into its owner by `st.async.shared::cluster` + a local mbarrier
    (bounded wait): no fence, counter or readback.
  - Splits across clusters (≤ 1 per CTA, processed first by tail-first order) keep the global protocol.
  - Cluster boundaries are item-aligned when the overshoot ≤ 0.5 stage (bs1: 0.1–0.2 stage for 2 touched experts).
  - Before building, J3 measures the DSMEM round trip under full streaming (gate p50 ≤ 1.0 µs vs ~2.5 µs for a global
    round trip) and `cudaOccupancyMaxActiveClusters(4)` = 66 at the K1 footprint.
  - Smem: the receive buffer is 4 KB per partner at NC1. It fits vid 0 only; HK2 has 5 KB spare.
- **Lever 3 lab.** Add `fmode`-style probes to `hc_item` (math off / no mma / no decode / B from registers / K1 hot X
  copies off) and run v36_all and r12 at bs16/32. That finds what makes a 4-chunk hot stage cost 3.25×. Hot-stage X
  (16.5 KB per stage from L2, i.e. candidate a) is one of the probes.

## 8. Build status (2026-10-25 evening): built, NOT yet compiled or measured

The two serverless attempts on account A (jobs `7f026d42…-u2` and `49cd53a5…-u1`) were both cancelled after 20 min
`IN_QUEUE` because no H200 capacity was free. The endpoint was paused after each attempt (workersMax 0, 0 workers).
Nothing ran, so there is no measured number for v3.8 yet.

Built on the laptop and ready for one job:

| file | what it is | checked locally |
|---|---|---|
| `tools/gen_k2a_v38.py` → `kernels/muse_moe_w4a16/k2a_v38.py` | k2a_v37 plus B1 (own and assist L2 prefetch before `griddepcontrol.wait`, vid 0 and vid 3 K2), B2 (`my_range2t` tilt in `gemm` and `ranges36`/`segs36c`), B3 (`S.gs` for vid 0's cold flush), B4 (K2 `route` returns before the scatter). 20 anchored patches; switches in probe bits 8..28 through `v38_call` / `v38_code`; all 0 = v3.7's arithmetic | Python parses; `sass_gate --list-src` extracts `KERNEL_SRC_V3`; tilt coverage emulation: 68,768 configs (S ≤ 40,000, n ∈ {1 … 264}, a 0..31), 0 failures, a = 0 equals `my_range2`, non-monotone control caught. **Not compiled: there is no nvcc on this PC** |
| `tools/box/k38_bench.py` | check (k2a_v38 self-test cases ≤ 48 plus 1 layer and 1 moe routing per bs, two calls, `==ship` marks bit-identical rows) and run (12 rows; per-bs gate and kill as in §5) | parses |
| `tools/box/a2_lab.py` | J1 rows `base_k1`, `f6`, `f6T`, `f6_k1`, `f7`. h / cnt / cnt2 are re-zeroed after the timing-only probes | selftest PASS |

B5 (dual epilogue warps) is not built yet: it changes the thread count and launch config of the vid 0 kernels.

Rerun, the whole batch as one job, ~15–20 min of GPU (order: v3.8 gate, check, bench, then J1):

```
python tools/serverless/endpoint_ctl.py up a
timeout 1800 python tools/serverless/sls.py run 'mkdir -p box_out/k38_gate; python3 tools/box/sass_gate.py kernels/muse_moe_w4a16/k2a_v38.py --src KERNEL_SRC_V3 --subst "@CASES3@=call:_gate_cases" "@ATTRS3@=call:_gate_attrs" "@WGSPECS@=call:_wg_specs" --allow C7519 --json box_out/k38_gate/gate.json > box_out/k38_gate.log 2>&1; grep -E "^\[|ptxas|RESULT|error" box_out/k38_gate.log | head -80; python3 tools/box/k38_bench.py check --bs=1,8,16,32 > box_out/k38_check.log 2>&1; grep -E "regs|WARNING|CHECK|Error|error" box_out/k38_check.log | tail -40; python3 tools/box/k38_bench.py run --bs=1,8,16,32 --routings=2 --rounds=20 > box_out/k38_run.log 2>&1; grep -E "^SUMMARY|^  bs=|^GATE|^BEST|TIMEOUT|correctness|Error" box_out/k38_run.log; python3 tools/box/a2_lab.py run --bs=1,8,16,32 --src=layer --routings=2 --rounds=20 --traces=1 --soak=0 --save-traces=none --rows=ship,base,baseT,base_k1,f6,f6T,f6_k1,f7 > box_out/a2b_lab.log 2>&1; grep -A20 "^WASTE" box_out/a2b_lab.log' --timeout 1500
python tools/serverless/endpoint_ctl.py down a
```

Readings:
- **Gate.** Register counts must equal v3.7's (`box_out/k37_gate4.log`), or stay ≤ 96 for HK2 K2, with 0 spills and no C75xx.
- **Check.** Rows with only B1, B3 or B4 must print `==ship`. Tilt rows must stay within tolerance.
- **Bench.** The `GATE <row>` lines carry the §5 gate.
- **J1.** f6 / f7 against base give the v3.9 decision.

**k3_v20 integration (only after the bench passes).** Follow the `tools/gen_k3_v17.py` pattern:
- `K2A33_KERNEL_SRC` becomes k2a_v38's `KERNEL_SRC_V3` byte for byte. Vids are unchanged, so `K2A33_VARIANTS` and
  the gate cases are too.
- Add a `K2A38_PLAN = {rep: (P, H, a2, a1, b3, b4)}` from the bench's best layer row per bucket. Buckets 1/8 go to
  vid 0 and 16/32 to vid 3; bucket 128 stays on v3.7's vid 5.
- `_v11_k2a33_call` and `_k2a36_call` OR `_k2a38_code(K2A38_PLAN[rep])` into their `probe` argument, read at call
  time behind `K2A38_ENABLE`.
- A prepare self-test per bucket: the bucket's k2a_v38 cases, two calls each, with the plan's bits on. A failure
  clears that bucket's entry, which falls back to the v3.7 code path.
- Then `layer_ab2` k38_all ≥ ×1.004 and the judge clone.

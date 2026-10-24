# K2b v3: hybrid of v1m's tile and load path with register-fed weights (stage A: design + emulation)

Status (2026-10-24, laptop only; the pod stays stopped):
- This doc.
- `k2b_v3_emulate.py`: ALL PASS, 10/10 negative controls fail as required.
- No CUDA has been written.
- **Kill check at stage A: PASS** (central haircut prediction +20.5% over v1m at judge 16K, against the +15% bar),
  but with a stated pessimistic case at +13%. Stage B's first measurement decides between them (§7).

## 1. Why: what today's v1m probes say (k2b_v1_design.md §12)

Judge T = 16,384, same process, v1m s5b3r (µs). Per stage = (t − 0.25 ms) / 5,632 stages per SM.

| probe | µs | per stage |
|---|---|---|
| full | 7,258 | 1.244 |
| p1: no dequant (coupled) | 4,476 | 0.750 |
| p37: decoupled MMA only | 3,895 | 0.647 |
| p33: decoupled loads + MMA | 4,347 | 0.727 |
| p34 / p35: decoupled loads + dequant / loads only | 4,199 / 1,965 | dequant work 0.397 |
| p32: decoupled full | 6,477 | 1.106 |

- **Loads and MMA overlap; handshakes are cheap.** Coupling costs 0.13 ms without the dequant.
- **The producer dequant does not overlap the MMA, even fully decoupled.** It is hardware contention: the
  dequant's 32 KB STS + 10 KB LDS per stage against the 80 KB of SS-wgmma smem reads.
- Every MMA-involving case runs at an effective 64–75 B/clk of smem traffic; the dequant round trip is what
  v3 removes.

**v3 in one line:** keep v1m's tile grid, producer and load path (exactly v1's 26 KB/stage: 16 KB activations, 8 KB
codes, 2 KB scales), drop the producer dequant and the BF16 weight tile, and let the consumers dequantise the K2A
lane items straight into wgmma **A registers** (v2's wiring: proven on the box by k2b_v2_unit.py and all of v2's
self-tests).

## 2. Kernel structure

**Tile.** RS wgmma needs the register operand on M, so the weights move to M (swap-AB). The CTA tile is v1's grid
transposed: **256 weight rows (v1's n-tile) × ≤ 128 tokens (v1's moe_align block of 128) × 64 k per stage**.
- Consumer WG c owns weight rows 128c..128c+127 as **two m64 blocks** (MW = 2): warp w of block mb owns K2A tile
  `K2B2_WTILE(nt, 2, c, mb, w) = 16nt + 8c + 4mb + w`.
- Each warp dequantises two lane items per stage.
- The wgmma is m64nNk16 RS with N = `K2B2_NSEL(valid)` ∈ {16, 32, 48, 64, 96, 128}, one templated mainloop per N,
  dispatched per tile. This replaces v1's 64-row empty-half skip.

**Scheduler and producer.** Exactly v1m's: persistent CTAs, n-tile fastest, coalesced 8-threads-per-row activation
cp.async, codes and scales, `cp.async.wait_group` + `bar.sync` + per-warp `full` arrival. The producer has no
dequant (its registers drop to ~40). The activation source is `x_perm` in MODE 1 (v2's pre-pass, ≈ 70 µs at 16K)
and the permuted inter in MODE 2, because the K2A item order permutes k inside each 64-block (`K2B2_PERM64`).

**Consumer per stage j:**
1. Wait `full[j]`.
2. For block b in (0, 1):
   - `wgmma.wait_group 1`, which retires the group that last read `A[b]`;
   - LDS.128 the item + LDS.32 the scale word;
   - 4 × deq4 + 16 × bmul2 into `A[b]` (`K2B2_AWORD/AOUT/AHI`);
   - `wgmma.fence`, 4 × wgmma RS into `acc[b]`, commit.
3. After block 1's wait, all of stage j−1 has retired: release slot j−1.

There is **one A buffer per block** (no double buffer). While a block is being dequantised, the other block's group
and the other WG's groups keep the tensor pipe fed (§4).

**Epilogue.**
- **MODE 1:** SwiGLU in registers (gate row g / up row g+8 of each K2A tile, per block). The CTA's 128 intermediates
  are exactly inter's logical 64-blocks 2nt and 2nt+1.
- **MODE 2:** the CTA's 256 output features of y.
- Both go through a 32 KB smem staging tile and coalesced 16 B stores; MODE 2 is staged in two halves.
- **v1m numerics are kept:** g, u = bf16(bf16(acc)·bf16(gs)); stock silu_and_mul; y = bf16(bf16(acc)·bf16(gs)).
- Stores skip padded tokens.

**R9 and capture** are as v1: bounded mbarrier waits, static shapes, branching on T only.

## 3. Shared memory per stage (N = 128) and footprint

| per stage (256 weights × 128 tokens × 64 k) | v1m | v3 |
|---|---|---|
| cp.async writes (activations 16 + codes 8 + scales 2) | 26 KB | 26 KB |
| dequant STS (BF16 weight tile) | 32 KB | **0** |
| dequant LDS (codes + scales) | 10 KB (producer) | 10 KB (consumers) |
| wgmma smem reads | A 16 + B 64 = 80 KB (SS; weight tile read by both WGs) | B only: 2 WG × 2 blocks × 4 × 4 KB = **64 KB** (RS) |
| **total** | **148 KB** | **100 KB (−32%)** |

At v1m's measured effective rate under wgmma (64–75 B/clk), 100 KB is ≈ 0.66–0.78 µs/stage of smem time. That is
the same as v1m's no-dequant pipeline (p1: 106 KB, 0.75 µs) and below v1m's full stage (1.24 µs).

**Footprint.**
- Ring S × 26,624 B + 32 KB staging + barriers; S = 6 → 192.5 KB, S = 7 → 219 KB.
- There are no B buffers (v1m spends 96 KB on three), so the load lookahead D = S − 2 is deeper than v1m's
  S − NB = 2.

## 4. Registers and where the consumer dequant lands

**Registers.**

| item | registers |
|---|---|
| accumulators: 2 blocks × N/2 (N ≤ 128) | 128 |
| A: 2 blocks × 16 (one buffer per block) | 32 |
| dequant temporaries (codes uint4, scale word, Sg/S8, transient deq4 outputs) | ~20 |
| descriptors / loop / addressing | ~15–20 |
| **consumer total** | **≈ 195–200** |

- setmaxnreg **56 / 216**: 128·56 + 256·216 = 62,464 ≤ 65,536. That stays far above the C7512 serialization line
  (≤ 80; lessons §7).
- The RS register-reuse trap is handled in straight-line code: `A[b]` is rewritten only after `wait_group 1` retires
  the group that read it. v2's equivalent two-buffer pattern compiled with no "wgmma serialized" line.

**Dequant against MMA issue** (measured rates, 1,980 MHz):
- The consumer dequant does the same 16,384 values per stage as v1m's producer: 512 items × ~65 instructions ≈ 1,040
  warp instructions per SM per stage, ≈ 260 issue cycles per SMSP.
- Decoupled MMA alone takes 0.647 µs = 1,281 cycles per stage, so the dequant ALU uses **~20% of issue slots**,
  spread over 8 consumer warps (2 per SMSP) instead of v1m's 4 producer warps.
- It generates **no smem stores** (only the 10 KB of LDS), so it no longer feeds the smem contention p32 exposed.
- **Bubble check for the single A buffer.** One item's dequant is ≈ 150–300 cycles of dependent latency. During it,
  the tensor pipe holds this WG's other-block group (64 × 128 × 64 × 2 = 1.05 MFLOP) plus the other WG's group:
  ≈ 630 cycles at the measured 81% wgmma efficiency. It hides.
- **Measured precedent:** v2's consumer-side dequant (one item per thread per stage) leaked only 307 µs at 16K, i.e.
  0.053 µs/stage (full 9,761 against no-dequant 9,454).

## 5. Stage model and predictions (calibrated on today's probes; R7 haircut = half the gain)

**Model:** v3 stage = v1m's coupled no-dequant stage (p1: 0.750 µs, which already includes loads, MMA and
handshakes) + the consumer-dequant leak.
- Central leak: 2 × v2's measured 0.053 µs (two items per thread), so 0.857 µs/stage, a GEMM-time ratio of 0.689
  against v1m's 1.244 µs.
- The model is applied to every case's GEMM time (measured v1m − non-GEMM), then x_perm is added (≈ 5 / 15 / 70 µs).
- Not credited, all upside:
  - the smaller wgmma B reads (64 vs 80 KB);
  - the deeper load lookahead.


The optimistic case (leak = v2's 0.053 µs) gives haircut +24.9% at 16K.


## 6. Emulation (`k2b_v3_emulate.py`, 14 s)

The checks reuse `k2b_v2_maps.cuh` (MW = 2), v1's staging macros and the K2A repack, plus v1m's epilogue numerics.


Negative controls; each FAILS as required:

| mutation | caught by |
|---|---|
| wtile_mb (block/warp swap in WTILE) | [1] |
| outcol_mb | [1] |
| gateup | [4] rel 0.80 |
| aout | [1] |
| intercol (natural-order inter) | [4] rel 1.35 |
| nsel (rounds down) | [2] |
| numerics (v1 FP32 epilogue instead of v1m's) | gives different BF16 values |
| **abuf_early** (dequant before wait_group 1) | **[6] A-buffer hazard** |
| **release_early** (release after block 0's wait) | **[6] slot hazard** |
| **lookahead** (D = S−1) | **[6] deadlock** |

**What the CPU cannot prove:**
- that the consumer dequant overlaps RS wgmma on the H200 (the leak);
- ptxas register allocation and serialization;
- why v2's own activation loads were 2.7× slower. v3 avoids v2's 256-token blocks and keeps v1's exact load geometry
  (16 KB activations + 10 KB weights per stage).

## 7. Stage B plan (when the pod is started)

1. **Write `k2b_v3.py`** from `k2b_v1m.py`:
   - the producer drops the dequant (ring with no B buffers, D = S − 2);
   - the consumer takes v2's per-item dequant and `wgmma_rs`, with per-block groups and MW = 2;
   - NSEL mainloops for N ≤ 128;
   - x_perm and the permuted inter;
   - v1m epilogue numerics with smem staging.
   About 300 lines changed. Paste `k2b_v2_maps.cuh` unchanged.
2. **Build gate:** no spills, no "wgmma serialized". **Unit test:** extend `k2b_v2_unit.py` to MW = 2 with v1's
   128-row staging.
3. **First measurement, which decides the kill:** v3 full against v3 no-dequant (probe 1) at judge 16K, i.e. the
   consumer leak.
   - Leak ≤ 0.106 µs/stage (central) → continue.
   - Leak ≥ 0.21 µs/stage (pessimistic) → stop.
4. Same-process bench against v1m and Marlin on both routings, then judge clone ×1, the t05b harness and memcheck.
5. Lock estimate: ≈ 45 min.

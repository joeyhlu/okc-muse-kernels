# K2b v2: RS wgmma, swap-AB, K2A lane items as the A operand (stage A: design + emulation, no CUDA yet)

- Laptop stage A is complete:
  - this doc;
  - `k2b_v2_maps.cuh` (every v2 index map, Python/C dual syntax);
  - `k2b_v2_emulate.py`: ALL PASS, 13/13 negative controls fail as required.
- No CUDA has been written. The GPU stage B is gated on one 2-minute probe of v1 (§7), because v2's predicted gain
  depends on it.
- **Layout: K2A v2.2's own**, by the coordinator's decision. The Marlin-layout evaluation was dropped: K2A reading
  Marlin's layout measured 1.8× slower; see kernels/muse_moe_w4a16/k2a_mv1.md §5.

## 0. Baseline to beat: K2b v1 as measured (honest, like-for-like)

Same process, same routing; Marlin timed beside K2b in one harness (box_out/k2b_v1_judge3.log, k2b_v1_bench_t05.log).



Judge clone, problem score: ×1.180 (3 repeats, ±0.018).

## 1. Why v2 (and what the v1 probes actually say)

v1 probe table at T = 16,384 (sw128x256s5b3r, per 128×256×64 stage per SM; GEMM part ≈ total − 0.25 ms):

| full | p1: no dequant | p2: no MMA | p3: loads only | p7: handshakes only |
|---|---|---|---|---|
| 1.24 µs | 0.92 µs | 0.79 µs | 0.42 µs | 0.27 µs |

**The contradiction in v1's §11, resolved.** v1's §11 named shared-memory bandwidth as the binder, but p1 moved the
same 106 KB/stage that v2 will move (80 KB SS-wgmma reads + 26 KB cp.async writes), and it ran at 0.92 µs, not the
0.42 µs that 128 B/clk would allow. So **smem bandwidth is not the binder**, and v2 is not justified by the smem
table.

What v2 is justified by:
- Removing the producer dequant: full 1.24 → p1 0.92 µs/stage, **≈ +30% raw**.
- Removing its 32 KB B-tile STS round trip.
- Fixing small-T padding (§5).

The open question is whether the v1 mainloop itself (MMA + handshakes, 0.92 µs against an ideal ≈ 0.52 µs) is
slow because of the issue chain or because loads interfere with MMA. v2 inherits the former and worsens the
latter. Probe 5 (§7) decides it before any CUDA is written.

## 2. Kernel structure

**Tile.** One CTA = 2 consumer warpgroups × 64 weight rows (MW = 1), each issuing wgmma **m64nNk16 RS**:
- A = the dequantised K2A item in registers;
- B = activations in smem, K-major SW128, staged exactly like v1's A tile.

That gives a 128 weight rows × N tokens tile, N ≤ 256. The weight m-tile is 8 K2A 16-row tiles:
- warp w of WG c owns tile `K2B2_WTILE(mt, 1, c, 0, w) = 8mt + 4c + w`;
- W13: 8 tiles = 64 gate + 64 up rows, i.e. **64 intermediates = exactly one 64-block of inter**.

**Token blocks.**
- moe_align block = 256 (ignore_invalid_expert).
- The valid prefix v of the block gives N = `K2B2_NSEL(v)`: multiples of 8 up to 64, then multiples of 32 up to
  256. That is 14 wgmma instantiations, dispatched per tile as in k2a's `wgmma_n`.
- Rows v..N−1 are zero-filled with cp.async src-size 0 (as v1), and rows ≥ N are never touched. Stale data there
  would be harmless anyway, since output column n depends only on B row n and padded columns are discarded.

**Scheduler.** As v1: persistent, tile t = (block, m-tile), m-tile fastest, so ~2 token blocks are live and each
expert's weights are re-read from L2.

**k order (the one subtlety).**
- A K2A lane item holds rows g and g+8 at physical k = 64kb + 16t + 0..15.
- In K2A v2.3's item order (tools/box/wgmma_unit.py mode 1, H200-validated), wgmma step q takes
  `a_s = deq4(word K2B2_AWORD(q,s))[K2B2_AOUT(q,s)] × S(K2B2_AHI(s))`. So logical column l = 16q + κ of a k64 stage
  is physical k `K2B2_PERM64(l)`, a pair-granular 4×8 transpose.
- 16 B cp.async cannot apply a 4-byte-granular permutation. So:
  - **MODE 2 inter:** the MODE 1 epilogue writes inter already permuted (`K2B2_INTER_COL`), for free.
- The alternative (natural-order B, then a 4×4 quad transpose of the A registers with 16 shfl per stage per thread)
  was rejected: +~50 consumer instructions per stage against a 1% pre-pass.

**Producer warpgroup (WG0).** cp.async only, no dequant:
- per stage: N activation rows × 128 B (v1's coalesced 8-threads-per-row map), 4 KB codes, 1 KB scales into an
  S-stage ring;
- `cp.async.wait_group D` + `bar.sync 1,128` + one arrival per warp on `full[s]` (count 4);
- D = S − 2 (no B buffers, so the consumers' release of j−2 is the only gate).

**Consumer warpgroups.** Per stage j:
1. Wait `full[s]`.
2. LDS.128 the lane's item and LDS.32 its scale word.
3. 4 × deq4 + 16 × bmul2 into **A buffer j&1** (two named register buffers; the k-loop is unrolled by 2).
4. `wgmma.fence`, 4 × wgmma RS, commit.
5. `wgmma.wait_group 1` (retires j−1, freeing A buffer (j−1)&1 for stage j+1).
6. Lane 0 of each warp releases slot j−1 (count 8).

The dequant of stage j overlaps wgmma(j−1).

**Epilogue (now transposed).** Weights are on M, so a thread holds one weight row × many tokens, while inter and y
are [token][feature].
- **SwiGLU in registers:** K2A's gate row g and up row g+8 of a tile are registers 4j+e and 4j+2+e of the same
  thread (`K2B2_GATE_REG` / `K2B2_UP_REG`).
- **Stores go through an smem staging tile [token][64·MW columns per WG]** (`K2B2_STAGE_OFF`). Then each token row
  is written with coalesced 16 B stores:
  - MODE 1: 128 B per row (one inter 64-block);
  - MODE 2: 2 × 128 B per row of y.
- Direct stores would scatter 2 B values at 8 KB stride. With the permuted inter columns a warp store covers only
  4 B pieces, ~12% sector efficiency, about 0.5 ms of L2 write traffic at 16K. Rejected.
- **Smem budget.** The ring is S × 37,888 B (activations 32 KB + codes 4 KB + scales 1 KB), plus staging, plus
  barriers.
  - With 64 KB of staging (MODE 2 y at N = 256 in one piece), S = 4 fits (217 KB) and S = 5 does not (255 KB).
  - **Plan: stage MODE 2's y in two 32 KB halves (WG 0's 64 columns, then WG 1's), so S = 5 and D = 3 fit
    (222 KB).** That keeps v1-s5b3r-like load lookahead.
  - Fallback: S = 4, D = 2.

**MW = 2 for small N** (N ≤ 128; selected per tile like N).
- Each consumer WG issues two m64 blocks (accumulators = N ≤ 128 registers), so the CTA tile is 256 weight rows
  and the per-stage fixed cost is spread over twice the MMA.
- `K2B2_WTILE` / `K2B2_OUT_COL` already carry MW; the emulator checks the MW = 2 tile coverage.

## 3. Shared memory and L2 traffic per stage (N = 256; v1 = sw128x256s5b3r)

| per stage (128 weights × 256 tokens × 64 k) | v1 | v2 |
|---|---|---|
| SS / RS wgmma smem reads | A 16 + B 64 = 80 KB | B only: 2 WG × 32 = 64 KB |
| dequant STS (B tile) | 32 KB | 0 |
| dequant / item LDS | 10 KB (producer) | 5 KB (consumers: 16 + 4 B per lane) |
| cp.async writes | 26 KB | 37 KB (activations 32 + codes 4 + scales 1) |
| **total smem** | **148 KB** | **106 KB (−28%)** |
| **L2→SM** | **26.6 KB** | **37.9 KB (+42%; avg N 245 → 36.4 KB)** |

**L2 feed at 16K.**
- v2 runs 761,856 stages × 36.4 KB = 27.7 GB per call.
- At the kill bar (v1 + 10% = 471 TF/s → 6.44 ms total, ≈ 6.1 ms GEMM) that needs **4.5 TB/s**.
- **Measured bound:** v1's p3 streamed 19.8 GB in ≤ 2.09 ms, so ≥ 9.5 TB/s L2→SM. That is an upper bound on time
  and includes the 0.27 µs/stage handshake floor.
- **Sensitivity:** if the real sustained rate were only 5.5 TB/s (typical published H100 figures), the feed would
  co-bind near 3,031 GFLOP / (5.0 + 0.33 ms) ≈ **570 TF/s**. That is still above the 471 bar.
- **Verdict: the L2 math does not kill v2.**

## 4. Registers (lessons §7: wgmma at ≤ 80 regs is serialized by ptxas, C7512)

| thread | contents | regs |
|---|---|---|
| consumer, MW 1, N ≤ 256 | accumulators 128 + A buffers 2 × 16 + item/scale 5 + addressing/loop ~20 | ~185 |
| consumer, MW 2, N ≤ 128 | accumulators ≤ 128 + A buffers 2 × 32 + ~25 | ~215 |
| producer | 16 activation row pointers (32) + loop ~20 | ~56 |

- setmaxnreg **56 / 224**: 128·56 + 256·224 = 64,512 ≤ 65,536. Consumers stay far above 128, so no C7512 with 2
  consumer warpgroups.
- The kill check passes.
- **RS-specific ptxas trap:** the A registers of an in-flight wgmma must be provably untouched. This is why there
  are two named A buffers and the loop is unrolled by 2. Stage B's first gate is a build with `-Xptxas=-v` showing no
  "wgmma ... serialized" line.
- The producer can instead re-derive its row pointers from a per-tile smem row table (1 KB), if 56 registers spill.

## 5. Small-T padding (emulation [7]; block 256 + NSEL vs v1's BM 128 with the empty-half skip)




What remains at small N is the **fixed per-stage cost**: at N = 40 the MMA is ≈ 0.08 µs, against a ≈ 0.27–0.30 µs
floor. Two things address it:
- MW = 2 halves the stage count;
- only N activation rows are loaded.

## 6. Prediction (conditional on probe 5; R7 haircut = half the predicted gain)

**Stage model**, calibrated on v1's measurements: t_stage(N) = t_fixed + (t_p1 − t_fixed)·N/256, with t_fixed =
0.30 µs (the v1 p7 floor plus the consumer dequant) and t_p1 = 0.92 µs (v1's mainloop without dequant at N = 256).
- GEMM = Σ tiles × 64 stages × t_stage / 132 / wave efficiency.
- Plus non-GEMM: align + x_perm + finalize + launches, ≈ 25 µs at 753, 60 µs at 3,012, 330 µs at 16K.


(Computed with the stage model above: tiles = 96 m-tiles × blocks / MW; the MW 2 fixed cost is taken as 1.3× the MW 1
one; wave efficiency from tiles / 132.)

- If the RS mainloop also beats v1's SS p1 rate (a CUTLASS-class 75% would give t_p1 ≈ 0.69 µs), 16K goes to
  ≈ 4.3 ms (≈ 700 TF/s). That is upside only; it is not planned on.
- **Kill check against the 16K bar** (≥ 471 TF/s): the haircut prediction, 488, passes, but thinly. It holds only
  if probe 5 shows the mainloop is not load-interfered (§7).

**Problem-score sketch.** The judge's layer-time arithmetic with the haircut v2 per class, against v1's measured
×1.180, gives ≈ ×1.22–1.25.

## 7. Stage B plan (GPU), in order. The first step decides go/no-go.

1. **Probe 5 on v1** (runtime flags already compiled in: probe 5 = no dequant + no loads, i.e. MMA + handshakes),
   at T = 16,384, next to p1 and p7. Also poll `nvidia-smi --query-gpu=clocks.sm` during the bench (no counter
   permission needed); the 0.52 µs ideal assumes 1.98 GHz.

   **Rule.** The interference p1 − p5 is taken to scale with v2's +42% cp.async bytes, so v2's 16K stage is ≈ p5 +
   1.42·(p1 − p5). The R7-haircut v2 clears the 471 TF/s bar only if that is ≤ 0.986 µs, i.e. **p5 ≥ 0.76 µs/stage**
   (v1 p1 = 0.92).

   | p5 (µs/stage) | v2 stage | v2 raw | v2 haircut | verdict |
   |---|---|---|---|---|
   | 0.92 (= p1: the issue chain is the cost) | 0.920 | 548 TF/s | 488 TF/s | **go** |
   | 0.85 | 0.949 | 532 | 480 | go |
   | 0.78 | 0.979 | 517 | 473 | go, marginal |
   | 0.70 | 1.012 | 501 | 465 | **stop** (below the bar after the haircut) |
   | 0.55 (loads interfere with MMA) | 1.075 | 474 | 451 | **stop**, redesign (2-CTA sharing / multicast of the activation tile) |

2. **Write `k2b_v2.py` from k2b_v1.py.**
   - The producer keeps v1's activation path and drops the dequant.
   - New consumer: RS wgmma with 12 N variants and MW ∈ {1, 2}, the x_perm pre-pass, the permuted-inter epilogue,
     and smem staging.
   - Paste `k2b_v2_maps.cuh` unchanged; the emulator already checks it.
3. **Unit test:** RS m64nNk16 with K2A items + permuted SW128 B (extends k2b_v1_unit.py; wgmma_unit.py mode 1 is
   the H200 precedent for the item order).
4. **Build gate:** no spills, no "wgmma serialized" line. **Self-tests** against FP32.
5. **Same-process bench** against v1 and Marlin on both routings, then the judge clone ×3, then memcheck.

## 8. Emulation (`k2b_v2_emulate.py`, 11 s full, ≤ 300 MB)

**Taken from the design's own sources:**
- every `K2B2_*` macro of `k2b_v2_maps.cuh`;
- v1's activation-staging macros (`K2B_ADDR_SW`, the coalesced producer map, descriptor fields) from k2b_v1.py's
  CUDA source;
- the K2A repack (bit-exact against k2a_v22.py).

**Written independently:**
- CuTe's SM90 `ALayout_64x16` and SM80 16x8x16 `ALayout` (literal shapes and strides), and `CLayout_64xN`;
- the wgmma B fetch from CUTLASS's canonical K-major layouts;
- deq4, `__byte_perm` and `fma.rn.bf16x2` bit semantics;
- moe_align, the post-norm and a float64 reference;
- tools/box/wgmma_unit.py mode 1's scatter and A-side formulas (transcribed), as the box-validated check of the
  permutation.

```
[1] SM90 ALayout_64x16 restricted to warp w == SM80 m16n8k16 ALayout shifted by 16w rows (128 threads x 8 values)
[2] PERM64/IPERM64 inverse, pair-granular, == wgmma_unit.py mode 1 (box-validated) scatter and A-side formulas
[3] A registers from K2A items == W[rows, PERM64(k)] exactly (2 experts incl. zero/neg/subnormal scales); WTILE covers MW 1/2
[4] RS wgmma (A regs x staged permuted B) == W x^T: valid 5 -> N 8, 40 -> 40, 64 -> 64, 77 -> 96, 160 -> 160, 256 -> 256
[5] ACC maps == CLayout_64xN; gate/up (K2A rho g / g+8) in one thread; INTER_PHYS/INTER_COL consistent with PERM64; OUT_COL
[6] mini MoE (x_perm pre-pass, block-256 align + NSEL, MODE 1 -> permuted inter -> MODE 2, finalize): rel 2.34e-3
    (T 40) / 2.24e-3 (T 300, an expert spanning 2 blocks), both moe_align contracts; every local pair's y row written once
[7] padding table (section 5)
[8] 600 random interleavings of the v2 protocol (S 3-6, D = S-2, 2 A buffers per WG): no hazard, no deadlock
ALL PASS (11s)
```

Negative controls (`--mutate`); each FAILS as required:

| mutation | caught by |
|---|---|
| perm_th | [2] IPERM inverse |
| **perm_both** (self-consistent wrong permutation) | **[2] wgmma_unit scatter cross-check** |
| aword | [3] |
| aout | [3] |
| ahi (scale row swap) | [3] |
| wtile | out-of-range / [5] |
| gateup | [5] |
| intercol (natural-order inter) | [5] |
| acctok | [5] |
| nsel (rounds down) | [4] |
| abuf_early (dequant before wait_group) | **[8] A-buffer hazard** |
| release_early | **[8] slot hazard** |
| lookahead (D = S−1) | **[8] deadlock** |

**What the CPU cannot prove:**
- RS wgmma's hardware A-register semantics beyond CuTe's documented layout. wgmma_unit.py mode 1 already passed on
  the H200 with this exact item order, at N ≤ 64.
- That ptxas will not serialize the RS loop.
- Any performance number.

## 9. Kill-criteria verdict (stage A)

1. **Registers unserialized with ≥ 2 consumer warpgroups: PASS.** ~185–215 per consumer under setmaxnreg 56/224,
   ≥ 128 everywhere.
2. **L2 feed math allows ≥ +10% over 428 TF/s at 16K: PASS.** It needs 4.5 TB/s against a measured ≥ 9.5. Even at
   5.5 TB/s it co-binds only near 570 TF/s.
3. **But the gain itself is thin after the haircut:** +14% (488 TF/s) and conditional on probe 5. Recommendation:
   run probe 5 (2 minutes) before writing any v2 CUDA.

## 10. Stage B result (H200 box, 2026-10-24): **killed by its own rule**

**Gate: GO.** Probe 5 on v1 at T = 16,384 was 4,860 µs, i.e. 0.82 µs/stage, above the 0.76 threshold (p1 5,475;
p3 2,354; p7 1,550; full 7,350; Marlin 11,419). The SM clock under sustained load was a median of 1,980 MHz (min
1,875; 33 samples above 450 W), so the 1.98 GHz assumption holds.

**Built and correct.**
- k2b_v2.py (nonce k2b-v2-1) compiled first try: 41.5 s cold for 4 variants, 0 spills, no "wgmma serialized" line
  from ptxas (setmaxnreg 56/224).
- k2b_v2_unit.py drives the real `consumer_stage` and passes (UNIT PASS: rel ≤ 2.3e-7 at N = 16/48/128/256, err 0).
- Every config passes the FP32 self-test at rel 3.38e-3. The stock-match numerics config (NUM 1) is at 4.22e-3.

**Same-process bench** (k2b_v2_bench.py; box_out/k2b_v2_bench1.log, bench2.log), µs:



**v2 probes, T = 16,384, v2s4** (µs): full 9,761; p1 no dequant 9,454; p3 loads only 6,294; p5 MMA + handshakes
4,782; p7 handshakes 1,689.

What was learned:
1. **The RS mainloop issues no faster than v1's SS mainloop.** p5 is 4.78 ms against v1's 4.86. The SS → RS switch
   bought nothing at the MMA-issue level, so v2's whole premise (gain = the removed producer dequant) did not hold.
2. **v2's load path is 2.7× slower in situ than v1's** (loads only: 6.29 vs 2.35 ms), against a byte ratio of only
   1.42×. Stage depth (S = 3/4/5) changes nothing, so this is not lookahead depth. The fence is ruled out.
   Candidates, in order:
   - the burst size per stage (2,432 cp.async per CTA, against 1,664);
   - per-SM outstanding-request limits;
   - poorer L2 locality of the 256-row activation blocks re-read by 64 m-tiles.
   ncu counters (not permitted in the pod) would separate these.
4. **In both kernels, loads and MMA add instead of overlap** (p1 ≈ p3 + p5 − p7). That is the common bottleneck,
   and it is not the dequant.

**Next options** (none started):
- **(a) A 2-CTA cluster sharing the activation tile** (DSMEM, or multicast after a gather pre-pass). This halves the
  dominant activation L2 stream per CTA and applies to v1 directly. It is the lever named at the probe-5 gate.
- **(b) A 20-minute micro-lab on v1's mainloop** before any redesign:
  - acc-operand fence placement (v1/v2 fence after commit; CUTLASS fences after `warpgroup_wait`);
  - wait_group depth 1 vs 2;
  - a single-CTA loads + MMA overlap test with resident data.
  This should explain why loads and MMA do not overlap.

# K2b v1: warp-specialised wgmma grouped NVFP4 × BF16 GEMM for the prefill experts (sm_90a, H200)

Status (2026-10-24, after the first GPU burst): compiled, unit-tested and self-tested on the H200, and faster than
Marlin in every class in the same-process harness. §10 has the box results; §3 describes v1 as it now stands (the
GPU phase changed the producer's synchronisation, its activation-load mapping and the load lookahead).
- `k2b_v1.py` (`# nonce k2b-v1-1`): the scored candidate.
- `k2b_v1_lab.py` (`# nonce k2b-v1-lab-1`): the check-only lab. It is generated from k2b_v1.py by `sed`; do not edit it by hand.
- `k2b_v1_emulate.py`: the CPU emulation. ALL PASS, and all 12 negative controls fail as they must (§8).
- `k2b_v1_unit.py`: the box unit test for the SS-wgmma operand layouts. Run it first in the GPU burst.

## 0. Hard requirement: one weight layout per layer, K2A v2.2's


**K2b v1 consumes K2A v2.2's layout directly.** The functions are copied verbatim from
`kernels/muse_moe_w4a16/k2a_v22.py`: `prepack_expert`, `reencode_scales`, `pack_words`, `tile_row_perm`, `BITPOS`
and `W_SHIFT`. The emulator asserts that the outputs are bit-identical to that file and that the source text is
identical.

**Cost to K2A: zero.** No change, no second copy, no new layout.

Why K2A's layout is actually a good fit for a dequant-to-smem SS-wgmma kernel (checked in emulation [2] and [3]):

1. **One K2A 32-bit word decodes to exactly one 16 B chunk of the B tile.**
   - `WQ[e][tile16][kb][lane][4×u32]`: word x/y holds row g at k = 64kb+16t+0..7 / 8..15, and z/w holds row g+8.
   - Each word holds 8 consecutive k of one row in natural k order.
   - `deq4` + 4 × `fma.rn.bf16x2` turns a word into 4 bf16x2 registers, which is one `st.shared.v4`.
   - So the K order needs no permutation, and the activation tile is a plain row copy.
2. **The loads are contiguous and the scales are self-contained.**
   - Per 16-row tile and per k64 step, the codes are 512 B contiguous (one cp.async per producer thread per 16 B).
   - Each lane's 16 B item uses exactly one scale byte per row (group 4kb+t), taken from its own `SQ` word.
3. **The bank-conflict pattern is free with the 128B swizzle.** A quarter-warp (g ∈ {0,1}, t ∈ 0..3) writes row g,
   chunk 2t+(wd&1). With the swizzle, the physical chunks (2t ⊕ g) are all distinct, so the dequant STS is
   conflict-free (emulation [3]: 1-way).
4. **SwiGLU needs no shuffle.**
   - K2A's W13 tile r is gate rows 8r..8r+7 followed by up rows 8r..8r+7.
   - With weights on the wgmma N side, gate and up of one intermediate index land in the **same thread**:
     columns 16i+2t+e and 16i+8+2t+e, which are accumulator registers 8i+e and 8i+4+e.
   - The intermediate columns of an n-tile are 128 contiguous columns.
5. **Numerics are identical to K2A.**
   - The scale re-encoding gives bf16 = sf·2^118, and the codes are e2m1·2^-126, so the B tile holds e2m1·sf·2^-8
     exactly.
   - The FP32 accumulator gets gs·2^8 (the same `W_SHIFT`).
   - Zero and negative scales are already folded into the code signs by K2A's prepack.

The prefill-board file keeps its own `prepare()`, which repacks the native NVFP4 weights the judge supplies.
In SGLang, K2A's prepack runs once per layer and both kernels receive the same `wq`/`sq` tensors.

## 1. What has to get faster (measured)

Prefill MoE work, from the judge profile of b0fab9da53fc4c53 and v0's in-prepare timing in 790cb81e9bb641ca:

| class (shape) | local routes | MoE GFLOP | Marlin µs (TF/s) | v0 Triton best µs (TF/s) |
|---|---|---|---|---|
| small (turn 1, T = 753) | 1,374–1,384 | 138 | 742 (188) | 693 (201) |
| mid (turn 4, T = 3,012) | 5,462–5,527 | 550 | 2,298 (242) | 2,045 (272) |
| large (16K chunk) | ~30,000 | 3,031 | 12,290 (245) | 9,567 (315) |

v0 plateaued at 31% of the 989 TF/s peak. Triton's register dequant feeding `tl.dot` cannot keep wgmma busy (see the
k2b_design.md §2 risk, confirmed by measurement). Target: **≥ 55% of peak (≥ 540 TF/s) on the large class.**

## 2. Tile shape and orientation: why tokens on M and weights on N

Both operands come from smem (SS), so either orientation is possible.

- **Chosen: CTA tile = 128 tokens (M, two consumer warpgroups of 64) × 256 weight rows (N = 256) × 64 k per stage.**
  This is CUTLASS's classic cooperative 128×256.
- **L2→SM traffic is the binding unknown** (the advisor's point; we have no H200 measurement, which is an R1 gap).
  - Each tile re-reads its activation block 32× (W13) / 16× (W2) across n-tiles.
  - Intensity is 2·BM·BN / (2·BM + 0.5625·BN) FLOP per byte of K (activations 2 B, packed weights 0.5625 B).
    128×256 gives **164 FLOP/B**, which means
    3.3 TB/s of L2→SM at 540 TF/s. The swapped 256 tok × 128 w gives 112 FLOP/B (4.8 TB/s).
  - v0 only demonstrated about 2.8 TB/s at 112 FLOP/B. Per stage the CTA reads 16 KB of activations + 8 KB of codes
    + 2 KB of scales.
- **The weight-wide tile also halves smem reads by wgmma** compared with weights-as-A.
  - Per 64-k stage, 2 WG × 4 × (A 2 KB + B 8 KB) = 80 KB, against 96 KB for 2 × m64n128 per WG.
- **Dequant cost is 16,384 values per stage per CTA**, one producer warpgroup (§4), against 1,024 MMA clk per stage at
  100%.
- **Small class (T = 753): padding is the issue.** At BM = 128, moe_align pads to 2,560 rows (+86%).
  - v1 adds an **empty-half skip**: a consumer WG whose 64 rows are all padding issues no wgmma and no epilogue (a
    warp `__any_sync` over the block's sorted ids).
  - This brings MMA rows down to 1,920 (+40%), the same as BM = 64, while keeping BM = 128's lower weight re-read.
  - The BM = 64 config (WGs split N, wgmma N = 128) is also in the sweep.
  - Per-tile N selection (k2a's `wgmma_n`-style, N ∈ {64, 128, 256} by valid-row count) is the v1.1 lever if the
    small class lags.

Padding, tiles and waves come from prepare()'s synthetic busiest-GPU routing (1.82 routes/token, 16 experts):

| T | routes | BM 128: padded / MMA rows (skip) | BM 64: rows | BM128×BN256 tiles W13 / W2 (waves, tail eff.) |
|---|---|---|---|---|
| 753 | 1,374 | 2,560 / 1,920 (+40%) | 1,920 (+40%) | 640 / 320 (4.8 / 2.4 waves; 0.97 / 0.81) |
| 3,012 | 5,462 | 6,528 / 5,952 (+9%) | 5,952 (+9%) | 1,632 / 816 (12.4 / 6.2; 0.95 / 0.88) |
| 16,384 | 30,108 | 30,976 / 30,656 (+2%) | 30,656 (+2%) | 7,744 / 3,872 (58.7 / 29.3; 0.99 / 0.98) |

## 3. Kernel structure (`k2b1_gemm<Cfg<MODE, BM, BN, S, NB, LAYOUT, REG>>`, 384 threads, 1 CTA/SM)

**Persistent tile scheduler.**
- The grid is min(#SMs, max tiles), with max tiles taken from the static padded size.
- n_mb = min(ntp/BM, blocks, n_sorted/BM) is read from the device `ntp`, so there are no host syncs and the kernel is
  graph-capturable.
- Tile t maps to (m-block t >> log2 NT, n-tile t & (NT−1)), **n-tile fastest**. About 132 concurrent tiles cover
  ~4 m-blocks × all 32 n-tiles, so one expert's 18.9 MB of W13 stays L2-resident across its m-blocks and the
  activations (4 MB) are read once.
- CTA b walks t = b, b+G, …. Blocks whose expert id is outside [0, 32) (−1 in the verbatim moe_align contract) are
  skipped identically by all roles.

**WG0, producer (128 threads).**
1. **Load stage j + D, with D = S − NB.**
   - Wait `empty[slot]`.
   - cp.async 16 B into ring slot (j+D) % S:
     - activation rows: MODE 1 gathers `x[pair >> 3]`, MODE 2 reads `inter[sorted row]`. Padding rows are
       zero-filled with src-size 0. **8 consecutive threads cover one 128 B row** (thread p: chunk p&7 of rows
       (p>>3)+16i, i < BM/16), so a warp instruction reads 4 full lines. The first mapping (one row per thread)
       used half of every L2 sector and cost 2×: loads-only went from 5.0 to 2.5 ms at 16K.
     - codes: 2·BN items of 16 B, 4 per thread.
     - scales: 16 B chunks of the K2A SQ words; both kb parities ride along, which is +1 KB/stage.
   - `cp.async.commit_group` (one group per stage).
   - D = S − NB means the refilled slot was freed by wgmma(j−NB), the same event that frees the B buffer the
     dequant writes. So the producer can run NB−1 stages ahead of the consumers. The first version used D = S − 2,
     which tied the producer to stage j−2 whatever NB was (lockstep). D = S − NB + 1 deadlocks: the protocol
     emulation proves it as negative control "lookahead".
2. **Dequant stage j.**
   - `cp.async.wait_group D` (this thread's stage-j group has landed), then `bar.sync 1, 128` across the producer
     warps (everyone's stage-j data has landed; replaces the `full` mbarrier and its 128 per-thread noinc
     arrivals). Then wait `bempty[b]`.
   - For each of the 4 items per thread: LDS.128 codes + LDS.32 scale word. `__byte_perm` + IMAD build Sg/S8. Then 4 ×
     (`deq4`, 4 × `fma.rn.bf16x2`, STS.128) into the BF16 B tile [BN rows × 64 k] at `K2B_ADDR_*(row, chunk)`.
   - `fence.proxy.async.shared::cta`, `__syncwarp`, then lane 0 of each producer warp arrives on `bfull[b]` (count
     4). This also publishes the stage's A data: the arrivals come after the named barrier.

**WG1 and WG2, consumers.**
- BM 128: WG c owns tokens 64c..64c+63 against all BN weights (m64n256k16, 128 FP32 accumulator registers).
- BM 64: both WGs own the same 64 tokens and split N (m64n128k16).
- Per stage:
  1. Wait `bfull[b]`. The consumer-side `fence.proxy.async` is dropped (runtime flag 8, the default: CUTLASS omits
     it; it passed every box self-test).
  2. `wgmma.fence`, then 4 × wgmma (descriptor + k16·KADV, scale-d = 0 on the first k16 of a tile), then commit.
  3. `wgmma.wait_group 1`, which retires stage j−1.
  4. Lane 0 of each warp arrives on `empty[(j−1)%S]` and `bempty[(j−1)%NB]` (count 8).
- Tile end: `wait_group 0`, release, then the epilogue:
  - `acc·gs` (FP32, gs = gs·2^8);
  - MODE 1: SiLU(gate)·up in FP32 in registers, then BF16 `inter[sorted row, 128·nt + 8i + 2t + e]`;
  - MODE 2: BF16 `y[pair, col]`;
  - valid rows only, 4 B stores.
- The finalize (`fused_post_expert_norm_weighted_sum`) is unchanged.

**Operand layouts (template LAYOUT).** Both have 1024 B-aligned buffers and 8-row atoms with SBO = 1024.
- **0, SW128.** The CUTLASS `Layout_K_SW128`: chunk c of row r sits at (r>>3)·1024 + (r&7)·128 + ((c ⊕ r&7) << 4).
  Descriptor layout type 1, LBO 16 (ignored), and the k16 advance is +32 B on the start address.
- **1, IL.** No swizzle: core matrix = 8 rows × 16 B, chunk stride 128 B, LBO 128, layout type 0, k16 advance +256 B.
  This is the descriptor recipe `tools/box/wgmma_unit.py` validated on the H200 (RS, B operand).
  - It is a fallback config in the scored set: 4-way bank conflicts on the dequant STS, which is perf only.

**R9 (no hangs).**
- Every mbarrier wait spins on `try_wait.parity` in a tight loop, reading clock64 every 4,096 polls. After 2^30 clk
  (about 0.5 s) the first waiter records its code in `err[0]` and returns. (`test_wait` polling measured slower.)
- Every later wait returns as soon as it sees `err[0] ≠ 0`, so a broken pipeline ends in bounded time with garbage
  output and the flag set.
- prepare() reads the flag after each self-test and timing, and rejects the candidate if it is set.
- The only named barrier is producer-internal (`bar.sync 1, 128`, all 4 producer warps always reach it). There are
  no grid barriers.

**Probes (runtime int, lab only).**
- 1: no dequant.
- 2: no wgmma/epilogue.
- 3: loads only.
- 7: handshakes only.

Each probe removes one stage's work in situ, so the lab measures which of L2 feed, dequant, MMA and handshakes binds
(R1/R5).

## 4. Budgets

**Shared memory (bytes; ≤ 232,448).**

The 1 KB alignment slack is gone: there is no static smem, so the dynamic base is 1024-aligned, and the kernel
checks this at run time (err 7).

| config | ring S × (A + codes + scales) | B buffers | total incl. barriers |
|---|---|---|---|
| **sw128x256s5b3r** (scored) | 5 × (16,384 + 8,192 + 2,048) | 3 × 32,768 | 231,552 |
| **sw128x256s4b3r** (scored) | 4 × 26,624 | 3 × 32,768 | 204,912 |
| sw128x256s4(r) / il128x256s4 | 4 × 26,624 | 2 × 32,768 | 172,128 |
| sw64x256s5 | 5 × (8,192 + 8,192 + 2,048) | 2 × 32,768 | 157,808 |
| sw128x128s6 | 6 × (16,384 + 4,096 + 1,024) | 3 × 16,384 | 178,320 |

**Registers.**
- `__launch_bounds__(384, 1)` gives 168 per thread.
- Consumer: 128 accumulators + about 25 (descriptors, counters, row bases), so it fits without `setmaxnreg`.
- Producer: about 60–90 (4 items × 4 words unrolled).
- REG = 1 is a lab lever: producer 88 / consumers 208, with exact pool accounting 128·88 + 256·208 = 64,512.

**Per stage (128×256×64; MMA = 4.19 MFLOP).** All at 1.83 GHz, the clock implied by 989 TF/s.

| resource | per stage | at 100% MMA (1,024 clk) | caps η at |
|---|---|---|---|
| tensor core | 1,024 clk | — | 1.00 |
| smem bandwidth (128 B/clk): wgmma 80 KB + dequant STS 32 + LDS 10 + cp.async writes 26 | 148 KB = 1,156 clk | 113% | **0.89** |
| producer dequant issue: 16,384 values; deq4 path ≈ 1.9 ALU op/value + 7 mem op/item, 4 warps (1 per SMSP) | ≈ 300–550 clk (IPC 0.5–0.9) | 30–55% | 1.9–3.4 |
| L2→SM feed: 26 KB/stage/SM | 6.1 TB/s × η | — | **0.57–0.74 if L2 sustains 3.5–4.5 TB/s (unmeasured)** |
| epilogue bubble (consumers drain per tile, ~1.5–2 µs per 36 µs/η tile) | — | — | ×0.97 |

Measured constants used:
- dequant 38.4 values/clk/SM for Marlin's path; ours uses fewer ops per value (lessons §5);
- streaming 4.3–4.5 TB/s from DRAM;
- launch +0.97 µs per kernel (lessons §6).

The L2→SM rate is the one design constant not yet measured. Probe 3 of the lab measures it in situ (R1), and §9
names the multicast fallback.

## 5. Predictions (experts GEMMs only; align and finalize as stock/v0)

Model: t = MMA-row FLOP / (η · 989 TF/s · wave efficiency) + ~15 µs (2 launches + ramp). TF/s below is on *real*
route FLOP, as v0's log reports it.

| class | real GFLOP | MMA-row GFLOP | raw model η 0.60 → µs (TF/s) | R7-haircut η 0.40 → µs (TF/s) | Marlin µs |
|---|---|---|---|---|---|
| small T=753 | 138 | 193 | 370 (373) | 545 (253) | 742 |
| mid T=3,012 | 550 | 599 | 1,100 (500) | 1,640 (335) | 2,298 |
| large T=16,384 | 3,031 | 3,086 | 5,300 (572) | 7,800 (389) | 12,290 |

- **Raw η 0.60** is min(0.80 for CUTLASS-class SS mainloop, 0.89 smem, L2 ≈ 0.65) × 0.97 bubble × wave tail.
- **R7 haircut:** blind custom-kernel predictions have been ~2× optimistic, so plan on η ≈ 0.40.

Layer: SGLang − Marlin (profile) − act_and_mul + K2b.

| load | raw η 0.60 | haircut η 0.40 |
|---|---|---|
| turn 1 (sliding/full) | 722 / 1,144 µs → **×1.43** | ×1.18 |
| turn 4 | 2,078 / 3,815 µs → **×1.48** | ×1.22 |
| first turn (4 shapes) | 11,008–18,158 µs → **×1.51** | ×1.25 |
| **problem score (geomean)** | **≈ ×1.47** | **≈ ×1.21** |

For comparison: v0 scored 1.075 and board #1 is 1.745.


## 6. prepare() and the lab

1. Baseline prepare.
2. Repack all 32 experts into the K2A layout (+0.9 GB).
3. `load_inline` one extension: MODE 1 + 2 of every config (scored: 4 configs = 8 instantiations; lab: 8 = 16),
   `-gencode compute_90a,sm_90a`. If the full build fails, it retries the first config alone (verbose).
4. moe_align `ignore_invalid_expert` contract check for BM ∈ {64, 128}, as v0, which passed on the judge.
5. Self-tests at T = 300 and T = 2,048 (Zipf 2) against the FP32 reference of `reference._experts`.
   - A candidate passes with rel < 1e-2, worst row < 5e-2, all finite, and `err` = 0. Marlin's own error is printed
     for scale.
6. CUDA-graph timing with an L2-flushing write before each replay, at T = 753 / 3,012 / 16,384. It prints µs and
   TF/s for Marlin and each config.
   - The plan per class is the fastest config that beats Marlin by 3% (`K2B_MIN_GAIN`), else Marlin. So the file is
     never slower than the incumbent per class.
7. Lab only: probes 1/2/3/7 for the first 4 passing configs at T = 16,384, and `-Xptxas=-v` for registers, spills
   and "wgmma serialization" warnings.
8. Deadlines are absolute from prepare()'s start: candidates until 200 s, timing until 265 s.
9. `kernel_fn` branches on T only. There are no `os`, `sys`, `environ` or `subprocess` calls in the file
   (`load_inline` is the path k2a already passed the judge with).

## 7. GPU-phase plan (box, in order)

**Gate A: the build is outside the deadline guard.**
- `K2B_DEADLINE_S` bounds only self-tests and timing. `load_inline` itself is unbounded, and a cold-ccache build
  past the 300 s prepare limit is a failed run, not a Marlin fallback.
- So: **no `okc check` of k2b_v1.py until the box reports the lab's build time.**
- If 16 lab variants take more than ~90 s, trim the scored `K2B_CONFIGS` to sw128x256s4 + il128x256s4 (or the lab's
  winners) before the first judge check.

**Gate B: run the lab through the box judge clone, not `okc check`.**
- `-Xptxas=-v` (registers, spills, "wgmma serialization") prints during the build, before every `[k2b]` line.
- The judge returns only the last 16,000 characters, so `okc check` of the lab would lose exactly the data it exists
  to collect.

Steps:
1. `box.sh sync`, then `box.sh run 'python3 kernels/muse_prefill_layer/k2b_v1_unit.py'` (~1 min). Expected output:
   - 4 plain-staged cases exact;
   - the negative control (SW data, IL descriptor) mismatching;
   - 2 cases staged through the kernel's own cp.async + `arrive.noinc` + bounded-wait path (with a zero-filled row)
     exact, with err = 0;
   - then `UNIT PASS`.
   Reading a failure:
   - plain-staged SW fails, IL passes: the descriptor or swizzle is wrong. Put the IL configs first and fix SW offline.
   - only the cp.async cases fail: the pipeline (noinc count, start parity, zero-fill) is wrong, not the layout.
2. `box.sh timed 'python3 tools/judge/judge.py problems/muse_prefill_layer kernels/muse_prefill_layer/k2b_v1_lab.py'`.
   Read off:
   - build time;
   - ptxas registers, spills and serialisation warnings per variant;
   - self-test rel and `err` per config;
   - µs and TF/s per class against Marlin;
   - the probes.
3. `box.sh run 'compute-sanitizer --tool memcheck python3 <the T=300 self-test only>'`: the only runtime check of the
   bounds the emulator asserts (sorted, bexp, ntp guards; inter and y row ranges).
4. Decide the next lever from the probes (T = 16,384, per config; "full" = the normal timing):

| observation | binding resource | lever |
|---|---|---|
| p3 (loads only) ≫ p7 (handshakes), and p3 ≈ full | L2→SM activation feed | 2-CTA cluster on two n-tiles of one m-block sharing the activation tile (13 instead of 26 KB/stage) |
| p2 (no wgmma) ≈ full | producer dequant / issue | second dequant warpgroup (setmaxnreg 2×80 + 2×168), or consumers dequant half each |
| p1 (no dequant) ≈ full, p2 ≪ full | MMA mainloop / smem bandwidth | S / NB / REG sweep; check ptxas for serialisation; ping-pong consumers |
| full ≫ max(p1, p2) | handoff latency (pipeline too shallow) | deeper S/NB (lab s5, nb3) |

5. Hard-code the winners in k2b_v1.py (≤ 2 configs), run `okc check`, then `okc submit` if it wins per class.

## 8. Emulation (`k2b_v1_emulate.py`, CPU torch 2.5.1, 13 s full / 3 s `--quick`, ≤ 300 MB RAM)

**Taken from k2b_v1.py itself, not re-typed.**
- The PURE section (repack, configs, smem formula).
- **Every index map of the CUDA source.** All `#define K2B_*` bodies are parsed out of `K2B_KERNEL_SRC` and evaluated
  as Python. They are written in the common subset: non-negative ints, `+ * << >> & ^ |`. They cover:
  - smem addresses for both layouts;
  - descriptor fields and k16 advance;
  - dequant item → (row, chunk);
  - scale selectors;
  - K2A global indices;
  - producer row/chunk split;
  - tile → (m-block, n-tile);
  - accumulator (warp, lane, reg) → (row, col);
  - SwiGLU register pairing and intermediate columns.
- The emulator also asserts that every macro is used by the device code.

**Written independently (the "hardware").**
- wgmma operand fetch from CUTLASS's canonical K-major layouts:
  - INTERLEAVE ((8,m),(T,2)):((1T,SBO),(1,LBO));
  - B128 ((8,m),(T,2)):((8T,SBO),(1,T)) ∘ Swizzle<3,4,3> (byte bits [4,7) ^= [7,10));
  - `GmmaDescriptor` bitfields, and the k-advance as +offset on the low word (CUTLASS `DescriptorIterator`).
- CuTe `CLayout_64xN` for the accumulator.
- deq4, `__byte_perm` and `fma.rn.bf16x2` bit semantics, with an exactness assertion.
- moe_align (both contracts, random in-expert order, garbage past ntp), the post-expert norm, and a float64 reference.

Results:

```
[1] repack bit-identical to k2a_v22.prepack_expert (3 shapes incl. zero/negative/subnormal scales); source text identical
[2] B tile = e2m1*sf*2^-8 exactly (deq4 + byte_perm scale + fma.rn.bf16x2), all 64 k of every row, 2 cases
[3] sw128x256s4    layout SW128: smem 173152 B, operands exact for both WGs x 4 k16 x 2 stages; bank ways: dequant STS 1, A cp.async 1
[3] sw128x256s4    layout IL   : smem 173152 B, operands exact for both WGs x 4 k16 x 2 stages; bank ways: dequant STS 4, A cp.async 1
[3] sw64x256s5     layout SW128: smem 158832 B, operands exact ...; bank ways: dequant STS 1, A cp.async 1
[3] sw128x128s6    layout SW128: smem 179344 B, operands exact ...; bank ways: dequant STS 1, A cp.async 1
[4] accumulator maps == CLayout_64xN; SwiGLU gate/up pairing with K2A W13 tiles; inter cols contiguous; MODE 2 cols
[5] normal/skewed x 4 configs x align ignore/verbatim (16 runs): rel 2.29e-03 .. 2.30e-03, every tile visited once,
    exactly the local pairs' y rows written (E=3, K 256/128, N 256, T=40, 3 persistent CTAs)
[6] 1200 random interleavings (S 3-6, NB 2-3, 1-4 tiles, random inactive WGs): no hazard, no deadlock
ALL PASS (13s)
```

Negative controls (`--mutate NAME`; each must FAIL, and each does):

| mutation | result |
|---|---|
| no_swizzle_write | A operand mismatch |
| kadv (k16 advance 64 B) | A operand mismatch |
| sbo (512) | A operand mismatch |
| lt (layout type 2) | not modelled |
| gateup | SwiGLU pairing |
| wchunk | B tile unwritten (NaN read) |
| ssel (row half) | dequant mismatch |
| accrow | ACC_ROW vs CLayout |
| intercol | SwiGLU pairing |
| sqidx (kb parity) | dequant mismatch |
| lookahead (D = S−1) | **deadlock** |
| early_release (release before wait_group) | **hazard: dequant into a B buffer wgmma is reading** |

**What the CPU cannot prove.**
- That the H200 implements the descriptor and swizzle exactly as CUTLASS documents them. `k2b_v1_unit.py` checks
  that first, and prepare() self-tests every config, dropping failures.
- That `cp.async.mbarrier.arrive.noinc` completion and proxy visibility behave as modelled. A failure there shows up
  as a self-test mismatch or the timeout flag, never a hang.
- Performance: all of §4–§5 is a model until the lab reports.

Full-size emulation (K = 4096, 32 experts) is not run on the laptop (16 GB RAM, swapping). If wanted, run it on the
pod CPU with `box.sh run`. The index maps are size-generic, and the only size-specific constants (KS = 64, NT ∈ {16,
32, 64}) are static_asserted in the CUDA source.

## 9. Risks and next levers

1. **Compile.** This is the first contact with nvcc.
   - Risk spots:
     - 131-operand inline asm (CUTLASS does the same);
     - device lambdas in the producer;
     - template smem attributes;
     - build time of 8/16 instantiations with fully unrolled m64n256. The lab reports it; the scored file will
       carry ≤ 2 configs.
   - Fallback: if the full build fails, prepare() retries the first config alone and prints the error.
2. **ptxas serialising wgmma** (accumulator touched between issue and wait) or consumer spills at 168 registers.
   - Read the lab's `-Xptxas=-v`. Levers: REG = 1 (consumers 208), or move the release arrives after the epilogue.
3. **L2→SM feed below 3.5 TB/s** (probe 3 ≈ probe 7 + real time).
   - Lever: a 2-CTA cluster on two n-tiles of the same m-block, sharing the activation tile via DSMEM or multicast.
     This halves the activation L2 reads, giving 13 KB/stage instead of 26.
4. **Producer dequant binding** (probe 1 ≪ full).
   - Lever: a second dequant warpgroup. 4 WGs fit as 2 × 128 × 80 + 2 × 128 × 168 = 63,488 of 65,536 registers via
     setmaxnreg. Alternatively, consumers dequant half each (the v1 alternative design: 3 B buffers, one named
     barrier per stage).
5. **Small class.** Per-tile wgmma N by valid count (64/128/256), and a split-K-free 2-tile-per-CTA ordering for the
   W2 tail (2.4 waves, 0.81).
6. **Epilogue bubble.** Ping-pong consumers (one WG's epilogue under the other's MMA) or an stmatrix-staged epilogue.

## 10. GPU phase results (H200 box, 2026-10-24)

**Unit test** (`k2b_v1_unit.py`, box_out/k2b_v1_unit.log): **UNIT PASS on the first run.**
- SW128 and IL descriptors are exact at N = 128 and 256 (2 WGs, 2 slots, 3 stages).
- The negative control (SW data, IL descriptor) mismatches 16,370/16,384, as it should.
- The kernel's own cp.async + `arrive.noinc` + bounded-wait path is exact, with the zero-fill row and err = 0.

**Build.**
- Cold builds (ccache off): 36.4 s for the 4-config scored set at the time (8 variants), 38.8 s for the final
  2-config set (4 variants), and 41 s for the 16-variant lab. Host headers dominate.
- ptxas: 0 spills in every variant and no wgmma-serialisation warnings. Consumer MODE 2 sits at the 168-register
  cap (154 for MODE 1); BN 128 variants use 96–118.

**Correctness.** Every config passes the self-test.
- rel 3.38e-3 (worst row 3.6e-3) against FP32. Marlin is 5.17e-3 (worst row 6.1e-3).
- err = 0.
- The judge clone's 5 tests pass with the lab file.

**Lab, v1 as written (judge clone, box_out/k2b_v1_lab.log).** µs (TF/s on real route FLOP); Marlin from the same
process and the same routing.

| config | T = 753 | T = 3,012 | T = 16,384 |
|---|---|---|---|
| Marlin | 745 (187) | 2,292 (243) | 11,689 (258) |
| sw128x256s4 | 858 (162) | 2,390 (233) | 10,814 (279) |
| il128x256s4 | 1,103 (126) | 3,010 (185) | 13,151 (229) |
| sw64x256s5 | 1,057 (132) | 3,101 (179) | 14,995 (201) |
| sw128x128s6 | 1,172 (119) | 3,550 (157) | 16,517 (183) |
| sw128x256s3 / s5 | 860 / 854 | 2,432 / 2,414 | 10,636 / 10,576 |
| sw128x256s4r | 810 (172) | 2,294 (243) | 9,974 (302) |
| sw128x256s4b3 | 874 | 2,475 | 10,825 |

Probes at T = 16,384 (sw128x256s4, full 10,814 µs): no dequant 8,928; no MMA 7,757; loads only 5,383; handshakes
only 1,973. These are nearly additive, so the stages were not overlapping.

**Fixes, each measured (bench = `k2b_v1_bench.py`: prepare() in one process, same routing as Marlin).**

| step | change | sw128x256s4r at 16K | best at 753 / 3,012 / 16,384 |
|---|---|---|---|
| 0 | as written | 10,227 | — |
| 1 | tight try_wait loop (the err global load had been read every 32 polls) | no change | — |
| 2 | test_wait polling instead of try_wait | slower (+0.4 ms) | rejected |
| 3 | producer sync: cp.async.wait_group + `bar.sync` + per-warp `bfull` arrival (no 128-thread arrivals); flag 8 (no consumer fence) | 9,113 | — |
| 4 | **coalesced activation loads** (8 threads per 128 B row) | **7,375** | 657 / 1,681 / 7,375 |
| 5 | **load lookahead D = S − NB** (producer NB−1 stages ahead), no smem slack, NB = 3 | 7,081 (s4b3r) | **626 / 1,584 / 7,006** |

**Current best (sw128x256s5b3r / sw128x256s4b3r, the scored set), from the same process and routing as Marlin:**


**Probe table for the current best** (sw128x256s5b3r, T = 16,384, µs; includes ~0.2–0.3 ms of align + finalize):

| full | p1: no dequant | p2: no MMA | p3: loads only | p7: handshakes only |
|---|---|---|---|---|
| 7,006 | 5,452 | 4,448 | 2,340 | 1,542 |

What the probes say:
- **The consumer chain binds.** With no dequant at all (p1), a stage takes 0.97 µs, against ≈ 0.52 µs of ideal tensor
  time (wgmma at about 55–60% of peak even alone). Dequant adds 0.27 µs/stage on top.
- **Loads and handshakes are no longer the limit.** p3 is 0.42 µs/stage, and p7 is about 0.27 µs/stage of serial
  producer overhead, which is hidden by the producer's slack.
- **The leading suspect is shared-memory bandwidth.**
  - Per stage: the 2 WGs' SS wgmma reads (80 KB, B read twice) + dequant STS (32 KB) + LDS (10 KB) + cp.async writes
    (26 KB) = 148 KB, which at 128 B/clk is ≥ 0.58 µs.
  - Without the dequant traffic it is 106 KB (0.42 µs).
  - This is what ncu should confirm (§11).

**Judge clone on the scored k2b_v1.py, 3 repeats** (box_out/k2b_v1_judge{1,2,3}.log; all 5 tests pass every time):

| run | speedup [95%] | turn 1 | turn 4 | first turn |
|---|---|---|---|---|
| 1 | ×1.1807 [1.163, 1.198] | ×1.074 | ×1.188 | ×1.290 |
| 2 | ×1.1799 [1.162, 1.198] | ×1.072 | ×1.186 | ×1.291 |
| 3 | ×1.1802 [1.163, 1.198] | ×1.074 | ×1.185 | ×1.292 |

v0 was ×1.075 and stock is ×1.0. In run 3, the per-shape speedup at T = 16,384 is ×1.18–1.32, and the experts are
7.08 ms against Marlin's 11.71 ms in the same process.

**compute-sanitizer memcheck** on the T = 300 self-test (sw128x256s5b3r): 0 errors, and the self-test passes under
memcheck (err = 0). See box_out/k2b_v1_san_mem.log.



At T = 753 on this flatter routing (25–30 experts, ~35–45 routes each), the 64-row M granularity pads ~1.9× and K2b
trails Marlin by 4%. The absolute targets are met, but only against the other harness's Marlin. K2b v2 (tokens on
the wgmma N side, padding +9%; k2b_v2_design.md) is the fix.

**ncu**: not available. The container lacks GPU performance-counter permission (ERR_NVGPUCTRPERM), and enabling it
is a host setting we do not change. The binder below rests on the probes plus arithmetic.

## 11. Next lever (from the probe table; ncu is unavailable on the pod)

> Superseded in part (2026-10-24): k2b_v2_design.md §1 shows, from v1's own p1 probe (106 KB/stage at 0.92 µs),
> that shared-memory bandwidth is **not** the binder. v2's case rests on removing the producer dequant and on padding,
> and its gain is gated on probe 5.

At 430 TF/s (43%), the consumer chain binds even with the dequant removed (p1: 0.97 µs per 128×256×64 stage, ideal
≈ 0.52). Smem traffic per stage is 148 KB:

| traffic | KB per stage |
|---|---|
| SS wgmma reads (B read by both WGs) | 80 |
| dequant STS | 32 |
| dequant LDS | 10 |
| cp.async writes | 26 |

At 128 B/clk that is 1,156 clk, which caps the tile at ≈ 0.58 µs/stage *if perfectly spread*. The tensor pipe
waits on it.

1. **K2b v2: RS wgmma, swap-AB, with K2A's lane items as the A fragments.**
   - This is the one lever that removes the binder instead of tuning around it.
   - K2A's per-lane 16 B item (rows g, g+8 × 4 k16 steps of one k64 block) is exactly an m64 RS A fragment for 4
     wgmma k16 steps. K2A v2.3 already runs RS wgmma on this layout (tools/box/wgmma_unit.py mode 1 validated).
   - Consumers dequant in registers: one LDS.128 per lane item, `deq4` + `bmul2`.
   - Tokens become the wgmma N (≤ 256), and activations are the only smem operand.
   - Per stage: activations 2 WG × 32 KB reads + 37 KB cp.async writes + 4.5 KB code LDS ≈ 105 KB, against 148 KB
     today. The 32 KB dequant STS and the 64 KB B reads disappear, and so does the producer dequant (p2 − p3 ≈
     2.1 ms at 16K).
   - The cost is more L2→SM activation traffic (256-token tiles re-read per 128-row weight tile: 37 vs 26 KB/stage).
     p3 says the feed is not binding today (0.42 µs/stage), but it would be the next limit.
   - Same layout, same scheduler, same harness: a contained change to the consumer and the producer.
2. **Inside v1** (smaller): ping-pong consumers (hide the per-tile epilogue and drain), and REG = 1 everywhere.
   - REG = 1 is consistently 8–12% faster.
   - Per-tile wgmma N by valid row count for T ≈ 753, where the tail tiles are mostly padding.

## 12. Overlap micro-lab on v1m (2026-10-24; box_out/k2b_v1n_lab_*.log, k2b_v1n_probe2.log; vehicle: k2b_v1m_lab.py)

Knobs, compiled per config; defaults reproduce v1m:
- FENCE: accumulator operand-fence placement;
- WAITN: wgmma groups in flight;
- runtime bit 32 ("decoupled"): no cross-role mbarrier waits, timing only.

Same-process µs (Marlin and every config timed in one process on the same routing):



Probes at judge T = 16,384, v1m (µs; ~0.25 ms is align + finalize):

| full | p1 no dequant | p2 no MMA | p3 loads only | p32 decoupled full | p33 decoupled, no dequant | p34 decoupled loads + dequant | p35 decoupled loads only | p37 decoupled MMA only |
|---|---|---|---|---|---|---|---|---|
| 7,258 | 4,476 | 4,577 | 2,443 | 6,477 | 4,347 | 4,199 | 1,965 | 3,895 |

What this says (it corrects §10 finding 4):
1. **Loads and MMA largely overlap.** Decoupled loads + MMA (4,347) is close to MMA alone (3,895); only ~0.45 ms of
   the ~1.7 ms load stream leaks.
2. **The non-overlap is dequant against MMA, and it is hardware contention, not handshake coupling.**
   - Decoupled full (6,477) is ≈ MMA (3,895) + the dequant's own cost (p34 − p35 ≈ 2,234), not
     max(3,895, 4,199).
   - The producer's dequant (32 KB STS + 10 KB LDS per stage) and the two consumer warpgroups' SS wgmma (80 KB of
     smem reads per stage) fight over shared memory.
   - The handshakes cost ~0.8 ms (coupled 7,258 against decoupled 6,477).
3. **Wgmma alone runs at about 81% of peak** (0.64 µs per 128×256×64 stage against 0.52 ideal at 1,980 MHz).

Next lever: take the dequant's smem round trip off the SS wgmma's path.
- Register-sourced weights (v2's direction) remove it, but v2's own activation loads were 2.7× slower in situ.
- A hybrid that keeps v1's 128-token × 256-weight tile and load path, with the weights as RS A fragments, would need
  weights on M (swap-AB), i.e. 256 weight rows = 2 m64 per consumer warpgroup with ≤ 128 tokens: accumulators 2 × 64
  = 128 registers.
- Alternatively, shrink the B-tile traffic: one consumer warpgroup per 256-column B read (it is read twice today).

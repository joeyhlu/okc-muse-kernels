# K2b — grouped NVFP4 × BF16 GEMM for the prefill experts (sm_90, H200)

Status: v0 written blind (no GPU). The repack layout, the inline-asm dequant (the PTX text itself, run by
an interpreter), the tile program (pid mapping, early exits, every flat address, masked loads, the 4 K16
dots per K-step, the SwiGLU epilogue, the stores) and the finalize are **emulated on CPU and match the
reference semantics** (§8). Not yet compiled. Submission: `k2b_v0.py` (`# nonce k2b-v0-1`), a copy of
`baseline.py` with only the experts call replaced (`_experts_dispatch`). Not submitted.

## 1. What has to get faster (measured, job b0fab9da53fc4c53)

Per 16K-token chunk on the busiest GPU the routed experts are ≈30,000 routes × 2·3·4096² = **3.0 TFLOP**,
weights ≤ 16–17 touched experts × 28.3 MB ≈ 0.48 GB (0.1 ms at 4.8 TB/s): **compute-bound everywhere**,
including turn 1 (1,359 routes = 137 GFLOP vs 0.4 GB). SGLang's two Marlin calls run at 19–29 % of the
989 TFLOP/s peak:

| shape | SGLang µs | Marlin µs (share) | routes (back-solved) | MoE GFLOP | Marlin TF/s (% peak) |
|---|---|---|---|---|---|
| turn 1 sliding (w 46) | 1,100 | 733 (67 %) | 1,359 | 137 | 187 (19 %) |
| turn 1 full (w 16) | 1,431 | 642 (44 %) | 1,314 | 132 | 206 (21 %) |
| turn 4 sliding (w 46) | 3,337 | 2,306 (65 %) | 5,543 | 558 | 242 (24 %) |
| turn 4 full (w 16) | 4,869 | 2,101 (42 %) | 5,330 | 537 | 255 (26 %) |
| 16K ctx 0 full (w 16) | 16,994 | 10,540 (61 %) | 30,420 | 3,062 | 291 (29 %) |
| 16K ctx 32K sliding (w 138) | 17,187 | 11,189 (65 %) | 30,499 | 3,070 | 274 (28 %) |
| 16K ctx 32K full (w 16) | 20,278 | 10,440 (51 %) | 30,002 | 3,020 | 289 (29 %) |
| 16K ctx 64K full (w 16) | 24,727 | 11,579 (45 %) | 28,878 | 2,907 | 251 (25 %) |

Also on the experts' account: `act_and_mul` 15 / 53 / 282–293 µs (turn 1 / turn 4 / 16K) — fused away in
K2b — and `count_and_sort` (moe_align) 5 / 16 / 81 µs, which K2b keeps (smaller with
`ignore_invalid_expert`, which drops the ~100K non-local slots at 16K).

Why Marlin is slow here (notes/study/03 §B.1): mma.sync (not wgmma), small moe-blocks (≤ 64 rows) that
re-stream each expert's weights per block, and the h = x·W13ᵀ round trip in BF16 through HBM.

## 2. Route decision: Triton (Option T) for v0

**What ≥ 50 % of peak needs** (per SM: 2,048 BF16 FMA/clk; 4 warp-instr issue/clk; ~128 B/clk LSU smem):
1. wgmma. mma.sync tops out well below 60 % on H100/H200 in practice; this alone rules out a Marlin-style
   kernel.
2. Dequant issue ≲ 50 % of the MMA time **and** overlapped with it (async wgmma).
3. L2→SM traffic ≤ ~7 TB/s: tile intensity 2·BM·BN / (2·BM + 0.5625·BN) FLOP per byte of K. Activations
   cost 3.6× more bytes than packed weights, so a wide **weight** dimension pays: BM 128 × BN 256 gives
   164 FLOP/B (6.0 TB/s at peak, 3.3 TB/s at 55 %); 256×128 gives 112; 64×256 gives 120.

**Option T (chosen).** Triton supplies wgmma (`tl.dot` → MMAv3), cp.async multi-stage pipelining, swizzled
smem and register allocation for free, and the image's Triton is the one SGLang's own Muse/MoE kernels use
(the API surface used here — `inline_asm_elementwise` with tuple dtypes and pack, `tl.reshape`+`tl.split`,
`tl.dot(a, b, acc)`, `float8e4nv` bitcast — all appears in `sglang/kernels/ops/moe/fused_moe_triton_kernels.py`
or the Muse Helion helpers at this commit). The only hand-written low-level piece is a 30-line PTX dequant
block whose arithmetic (E2M1 bit placement into a BF16 subnormal, then `fma.rn.bf16x2` by the block scale)
is the same trick k2a uses — and **k2a's self-test passed on the judge's H200** (job db513c266a324a79:
rel 3.7e-3 at bs 1…257), so the hardware side of the trick is validated. Failure modes are benign: a
compile error or a wrong result is caught in `prepare()` and that candidate is dropped; no mbarriers, so
no hang risk. Predicted 45–55 % of peak (§5). What stays uncertain is Triton's codegen: whether the
register-produced B operand (`local_alloc` per K16 dot) keeps the wgmma asynchronous, and how it schedules
4 small dots per iteration. That is exactly what the in-`prepare()` variant timing measures (§6): one
`okc check` evaluates 6 configs (SS vs swap-AB, tile shapes, asm vs pure decode) against Marlin.

**Option C (deferred to v2 if T plateaus < 50 %).** A warp-specialised CUDA kernel (TMA/bulk-copy
producer, dequant warpgroup writing BF16 into a 128B-swizzled smem ring, two consumer warpgroups on
SS wgmma m64n256k16, 2-CTA multicast of the packed weight tile) has a 65–75 % ceiling. But blind it needs
correct wgmma smem descriptors (LBO/SBO/swizzle bits, 1024 B-aligned atoms), `fence.proxy.async` between
generic smem writes and the async proxy, mbarrier phase tracking (a bug is a hang = a charged timeout) and
the accumulator fragment layout for the epilogue. None of that can be validated on this Mac beyond our own
model of it. Expected 3–6 iterations to correctness alone, so it does not fit "≥ 50 % within ~6 checks".
The K2b repack, decode bit trick, emulator and prepare() harness carry over unchanged to Option C.

## 3. Kernel (`_k2b_gemm`, one source for both GEMMs)

**Pipeline per call** (`_k2b_experts`): `expert_map[topk_ids]` (local 0..31 / −1) →
`moe_align_block_size(ids, BM, 32, ignore_invalid_expert=True)` (SGLang's CUDA kernel; prepare() verifies
the contract, else the baseline's verbatim `(ids, BM, 256)` call whose −1 blocks the GEMM skips) →
`_k2b_gemm` MODE 1 (W13 + SwiGLU) → `_k2b_gemm` MODE 2 (W2) → SGLang's
`fused_post_expert_norm_weighted_sum` (unchanged, `slot_expert_ids` skips non-local slots).
5 launches (Marlin path: 5 + `silu_and_mul`); all grids static (`n_mb × N/BN` from the static
`max_num_tokens_padded`), early exit on `pid_m·BM ≥ ntp` (device scalar) and on non-local blocks;
no host syncs → CUDA-graph safe.

**Repack (lossless, 0.5625 B/element as stored)**, per expert, K-step ks of 64 elements:
- `wq[e][ks][h][n][j]` uint8, h∈{0,1}, j∈0..15: low nibble = code of `W[n, 64ks+32h+j]` (scale group 4ks+2h),
  high nibble = code of `W[n, 64ks+32h+16+j]` (group 4ks+2h+1). A tile's `[BN, 16]` byte load is
  BN·16 contiguous bytes (16 B per row → one cp.async per thread).
- `sq[e][ks][n]` int32 = the 4 raw E4M3 bytes `sf[n, 4ks+g]` (byte g = group g): one `[BN]` int32 load
  per K-step (4 B per thread, so cp.async-eligible and pipelined like the other loads; a `[BN]` uint8
  vector would be 0.5–2 B per thread and stay a synchronous L2-latency load every iteration). Bytes are
  extracted with `(w >> 8g) & 0xFF` → uint8 → `float8e4nv` bitcast → fp32 (exact). Still 1 byte per scale.
- W13 rows interleaved: prepacked row 2i = gate row i, 2i+1 = up row i, so a BN tile holds BN/2 complete
  (gate, up) pairs and SwiGLU is `tl.split(tl.reshape(acc, (BM, BN/2, 2)))` (SGLang's FUSE_SWIGLU idiom).

**Tile** = BM sorted rows (one expert) × BN weight rows × full K = 4096. Per K-step: 2 byte loads
`[BN,16]`, 1 scale-word load `[BN]` int32, 4 activation loads `[BM,16]` (MODE 1 gathers row `pair // 8`, MODE 2
reads the sorted `inter` rows), decode → 4 BF16 `[BN,16]` operands, **4 × `tl.dot(K=16)`** into one fp32
accumulator (orient 0: `acc[BM,BN] += x · wᵀ`; orient 1 = swap-AB: `acc[BN,BM] += w · xᵀ`, transposed
once in the epilogue). Scale groups align with dots, so a scale is a `[BN,1]` broadcast — no reshapes,
joins or splits in the main loop.

**Dequant** (both exact in BF16; proof notes/study/03 §C):
- DECODE 0 (asm, `pack=4`, elementwise per byte so Triton may regroup elements freely): per 4 packed
  bytes, `prmt` spreads bytes into 16-bit lanes, `(t<<12)&0x80008000 | (t<<6)&0x01C001C0` places the low
  nibble as BF16 `e2m1·2^-126` (high nibble: shifts 8 and 2), then `fma.rn.bf16x2` by `BF16(sf·2^118)` →
  `e2m1·sf·2^-8`, exact; the FP32 accumulator gets `gs·2^8`. ≈ 26 ALU + 4 fma per 8 elements.
- DECODE 1 (pure Triton): `bits = mag≥2 ? 0x3F00+(mag<<6) : mag·0x3F00`, sign → bit 15, bitcast → exact
  normal BF16 e2m1; × `BF16(sf)` (exact); gs on the accumulator. ≈ 9–10 ops/element, no subnormals at all.

**Epilogue**: `acc·gs` in FP32. MODE 1: `silu(g)·u` in FP32 → BF16 `inter[sorted row]` (all BM rows of a
live block, padding rows become 0). MODE 2: BF16 `y[pair]` for real rows only.

**Numerics vs SGLang**: weights exact in BF16 (same as Marlin); gs in FP32 on the accumulator (Marlin:
gs rounded to BF16, applied after acc→BF16); h never rounded (Marlin stores h in BF16); SiLU·up rounded
once to BF16; y BF16 (as Marlin); finalize identical. Emulated error vs the FP32 reference: 3.4e-3.

**Bounds**: every address is in range whatever moe_align returns: the expert id is range-checked
(`0 ≤ e < 32`), the sorted-id load is masked by its length (`other = n_pairs`), pair ids outside
`[0, n_pairs)` are padding, `inter` has `n_mb·BM` rows. The emulator asserts all of this per tile.

## 4. Tile configurations (candidates, tried in this order)

| name | orient | decode | BM (tokens) | BN (weights) | warps / stages | MMA clk per K-step | decode clk per K-step (asm ≈3.6 instr/elem) | L2 FLOP/B |
|---|---|---|---|---|---|---|---|---|
| ss_asm_128x256 | SS | asm | 128 | 256 | 8 / 3 | 1,024 | ≈ 460 (45 %) | 164 |
| ss_asm_64x256 | SS | asm | 64 | 256 | 4 / 4 | 512 | ≈ 460 (90 %) | 120 |
| sw_asm_256x128 | swap-AB | asm | 128 | 256 | 8 / 3 | 1,024 | ≈ 460 (45 %) | 164 |
| ss_pure_128x256 | SS | pure | 128 | 256 | 8 / 3 | 1,024 | ≈ 1,300 (125 %) | 164 |
| ss_asm_256x128 | SS | asm | 256 | 128 | 8 / 3 | 1,024 | ≈ 230 (22 %) | 112 |
| sw_asm_128x64 | swap-AB | asm | 64 | 128 | 4 / 4 | 256 | ≈ 230 (90 %) | 85 |

Scale extraction (shift, mask, cvt per group) costs ~4 int ops × 4 groups per row per K-step, i.e. BN·16
ops against BN·64 decoded elements (< 0.3 op/element, included in the 3.6). Smem per stage ≈ BM·128 + BN·36 B
(25 KB for 128×256) plus the BF16 B-operand buffers; ≤ ~160 KB. All
loads in the K loop are ≥ 4 B per thread (cp.async-eligible): codes 16 B, activations 16–32 B, scale word 4 B.
BM is also the alignment block: padding ≈ touched·BM/2 rows (turn 1: 14 experts → +33 % rows at BM 64,
+66 % at BM 128; 16K: +3.5 % at BM 128). The small class is where BM 64 / swap-AB should win.

## 5. Predictions

Layer time = SGLang µs − Marlin µs − act_and_mul µs + K2b GEMMs, with K2b = padded FLOPs / (η·989 TF/s)
+ 15 µs (launch / early-exit overhead), BM 64 for turn 1 and 128 otherwise (`/scratchpad/pred.py`):

| shape | SGLang µs | K2b GEMMs µs @ η 35 / 45 / 55 % | layer µs @ 35 / 45 / 55 % | speedup @ 45 % |
|---|---|---|---|---|
| turn 1 sliding | 1,100 | 545 / 427 / 352 | 896 / 779 / 704 | 1.41 |
| turn 1 full | 1,431 | 527 / 414 / 341 | 1,303 / 1,189 / 1,116 | 1.20 |
| turn 4 sliding | 3,337 | 1,938 / 1,511 / 1,239 | 2,916 / 2,489 / 2,217 | 1.34 |
| turn 4 full | 4,869 | 1,866 / 1,455 / 1,193 | 4,583 / 4,172 / 3,910 | 1.17 |
| 16K ctx 0 full | 16,994 | 9,148 / 7,118 / 5,827 | 15,319 / 13,290 / 11,998 | 1.28 |
| 16K ctx 32K sliding | 17,187 | 9,199 / 7,158 / 5,859 | 14,909 / 12,868 / 11,569 | 1.34 |
| 16K ctx 32K full | 20,278 | 9,064 / 7,053 / 5,773 | 18,620 / 16,609 / 15,329 | 1.22 |
| 16K ctx 64K full | 24,727 | 8,739 / 6,800 / 5,566 | 21,594 / 19,656 / 18,422 | 1.26 |

Problem score (geomean over the 3 loads of Σw·SGLang / Σw·ours): **×1.15 @ 35 %, ×1.31 @ 45 %, ×1.44 @ 55 %**
(per load @ 45 %: turn 1 ×1.34, turn 4 ×1.28, first turn ×1.31). Expected v0 outcome: the 128×256 configs
at 40–55 % on 16K; the small class 30–45 % (7 waves, BM-64 padding). The run cannot regress the experts:
a class keeps Marlin unless a candidate beats it by ≥ 3 % in prepare()'s own graph timing.


## 6. What prepare() does (and what the log shows)

1. Baseline prepare (unchanged; its duration is printed), then `_k2b_prepare`: repack 32 experts
   (~0.9 GB extra).
2. `_k2b_check_align`: runs `moe_align_block_size(..., 32, ignore_invalid_expert=True)` for BM 64/128/256 on
   synthetic routing and checks every local pair appears once in a block labelled with its expert, no −1
   block inside ntp → `[k2b] align ignore_invalid_expert: ok` (else the verbatim call is used).
3. Self-tests at T = 300 and T = 2048 (Zipf-2 skew, multi-block hot experts) against an FP32 dequantized
   reference of `reference._experts`; also prints Marlin's own error for scale. A candidate passes with
   rel < 1e-2, worst row < 5e-2, finite. Each first call compiles MODE 1 + MODE 2 (`[k2b] <name> PASS (…s
   incl. compile)` gives the compile time). No new candidate after 190 s from prepare()'s start
   (absolute deadline: the 300 s limit includes the baseline's own prepare).
4. Timing under CUDA graphs with an L2-flushing write before each replay (as the judge), T = 753 / 3012 /
   16384 on busiest-GPU-like synthetic routing; prints µs and TF/s for Marlin and every passing candidate,
   and the chosen plan per class (small ≤ 1024 < mid ≤ 6144 < large tokens); no candidate is timed after
   250 s from prepare()'s start. The synthetic routing gives 1.82–1.84 local routes/token on 16 touched
   experts (top expert ≈ 0.47 routes/token), matching the judge's busiest GPU (1.75–1.86, ~15–17 experts).
5. `kernel_fn` branches on the token count only (allowed): `_experts_dispatch` → K2b config or Marlin.
Triton specialisation: the three runtime ints are `do_not_specialize`, so the test/scored shapes reuse the
kernels compiled in prepare() and nothing compiles inside the judge's capture.

## 7. Compile-risk spots (first contact with the GPU)

1. `inline_asm_elementwise` operand packing: 2 int16 outputs × pack 4 → `$0..$3`, then uint8 (`$4`, 4 bytes
   in one reg), 2 × int16 scale tensors (`$5,$6`, `$7,$8`). If Triton packs differently the asm candidates
   fail to compile or fail the self-test; `ss_pure_128x256` is the fallback candidate.
2. PTX syntax: `prmt.b32` with an immediate selector, `fma.rn.bf16x2` (sm_80+), immediates in `and.b32`.
3. `tl.dot` with K = 16 on a `tl.trans` of a register tensor (SS) or with a register LHS (swap-AB) — legal,
   but Triton may keep each wgmma synchronous (perf risk, not correctness): compare SS vs swap-AB timings.
4. `tl.reshape(acc,(BM,BN/2,2))` + `tl.split` on an MMA-layout fp32 accumulator (SGLang does it on BF16);
   `tl.trans(acc)` for swap-AB.
5. Smem: 3 stages × ~25–37 KB + B buffers + epilogue scratch; an OutOfResources is caught per candidate.
6. `float8e4nv` bitcast of a uint8 load; fp32 → bf16 of `sf·2^118` (exact, < 2^127).
7. moe_align `ignore_invalid_expert` contract (verified in prepare, fallback verbatim). With 32 (+1)
   buckets and numel = 8T ≥ 1600 the CUDA host code takes the general 2-kernel path (align +
   `count_and_sort`; the single-block small-batch path needs numel < 1024), and the −1 slots — ~100K of
   131K at 16K, all contending on one bucket counter in the baseline — are skipped.
8. prepare() wall clock: baseline prepare + ≤ 12 Triton compiles + tests + timing must stay < 300 s
   (absolute deadlines: candidates until 190 s, timing until 250 s from prepare()'s start).
9. Scale word: `(sw >> 8g) & 0xFF` on a signed int32 (arithmetic shift, masked) → `.to(tl.uint8)` →
   `float8e4nv` bitcast.

## 8. Emulation (`k2b_emulate.py`, CPU torch 2.5.1)

Execs the torch-only section of k2b_v0.py (repack/configs), extracts the PTX string and runs it through a
small interpreter with Triton's pack=4 packing, and transliterates the kernel's index math. Kernel-written
buffers start NaN; stores assert in-bounds and write-once; the grid is checked to visit each
(m-block, n-tile) exactly once; moe_align is emulated for both contracts with a random order inside each
expert and garbage expert ids past ntp. Unit tests: all 256 bytes × 254 finite E4M3 scales (incl. 0, −0,
negatives, subnormals) exact for both decoders; asm result independent of how Triton groups the 4 packed
elements; repack round trip through the kernel's flat addressing. One emulator bug found on the way: its
own float32 path flushed the BF16 subnormal on torch worker threads (FTZ); bit conversions now go through
integer field decoding — a reminder that the asm variant must keep the `e2m1·2^-126` subnormal inside the
`fma.rn.bf16x2` (it does: the subnormal never leaves the asm block).

Results: see `EMULATION RESULTS` appended below.


- **Call site**: `MuseSpark1xFp4Experts._marlin_expert_outputs`
  (`python/sglang/srt/layers/muse_spark_fp4_moe.py:451`, used by `_forward_marlin` :576). Its contract —
  return `(expert_outputs [T, topk, 4096] BF16 in pair order, local_topk_ids)` — is exactly what
  `_k2b_experts` produces before the finalize, so `fused_post_expert_norm_weighted_sum` and everything after
  stay unchanged. Replace the two `moe_wna16_marlin_gemm` + `silu_and_mul` for `num_tokens ≥ ~256`
  (extend/prefill batches; SGLang runs extend eagerly, so the ~5 launches cost µs, not graph nodes).
- **Weights**: `MuseSparkNvFp4FusedMoEMethod._process_weights_for_marlin`
  (`srt/layers/quantization/muse_spark_nvfp4_moe.py:128`) sees the native `gate_up_proj` [E, 8192, 2048]
  uint8, `gate_up_proj_weight_scale` [E, 8192, 256] E4M3, `*_weight_scale_2`; K2b's repack must run there
  **before** Marlin's repack frees the natives. Memory: one layout per layer — either K2b replaces Marlin
  for all token counts (decode then needs a K2b-layout decode kernel; k2a's layout differs) or both are kept
  (+~0.9 GB of experts per layer per GPU, likely too much). Real checkpoints carry separate gate/up global
  scales; Marlin collapses them to the gate one (the post-expert RMSNorm cancels the constant gain, comment
  at :576); K2b can apply both exactly (two scalars per expert in the epilogue: even columns gate, odd up).
- **Shapes per 16K chunk, one EP8 rank**: x [16384, 4096] BF16 (after pooler + pre-expert norm);
  topk_ids [16384, 8] global → expert_map → 0..31 / −1; local routes ≈ 16.4K on an average rank, ≈ 30K on the
  busiest; ~15–17 of 32 experts touched; inter [≈ routes + 33·(BM−1), 4096] BF16 (≈ 1.1 GB allocated for the
  static worst case 8T), y [131072, 4096] BF16 (1.07 GB, same as Marlin's cache3; only local rows written).
  W13 grid ≈ 240 live m-blocks × 32 n-tiles at BM 128 / BN 256; W2 × 16 n-tiles.
- **Config choice** without prepare()-time timing: a static table by token count taken from the challenge
  runs' `[k2b] time …` lines.

## 10. v1 roadmap (after the first check)

1. Read `[k2b] time` lines: TF/s per config and class; compile times; which variants failed and why.
2. If SS and swap-AB are both < 45 % at 16K: try BK = 128 per iteration (8 dots), num_stages 4, GROUP_M 4/16,
   a persistent grid (removes the empty-CTA tail and pipeline refills), `lop3` in the asm (−15 % decode ops).
3. Small class: swap-AB with BM 32 (tokens as wgmma N = 32) to cut padding; split the W2 grid finer.
4. Fuse the finalize into MODE 2 (per-row sum of squares needs the full 4096 columns: second tiny pass or
   BN = 4096/…); fold the expert_map/align into one Triton launch.
5. If Triton plateaus below 50 %: Option C (§2) on the same repack and emulator.

## EMULATION RESULTS (2026-10-23, `python3 -I k2b_emulate.py`, CPU torch 2.5.1, 651 s)

Final code (int32 scale words, expert-range and sorted-length guards). 8 emulated local experts with
full-size W13/W2 (judge's random_weights; the wide-scales case uses log-uniform 2^-6..448 block scales),
6 configs x 2 moe_align contracts x 3 routings; tolerance 1.5e-2 (checks.compare).

```
primitive unit tests: OK (asm + pure decode exact for 256 bytes x 254 E4M3 scales incl. 0/neg, pack-grouping independence, prmt, repack round trip)
normal      T=160 routes= 230 ss_asm_128x256   align=ignore  PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=1024 n_mb=43 tiles=384 (91.9s)
normal      T=160 routes= 230 ss_asm_128x256   align=verbatim PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=2176 n_mb=265 tiles=384 (5.5s)
normal      T=160 routes= 230 ss_asm_64x256    align=ignore  PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=576 n_mb=53 tiles=432 (4.7s)
normal      T=160 routes= 230 ss_asm_64x256    align=verbatim PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=1664 n_mb=273 tiles=432 (4.1s)
normal      T=160 routes= 230 sw_asm_256x128   align=ignore  PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=1024 n_mb=43 tiles=384 (4.4s)
normal      T=160 routes= 230 sw_asm_256x128   align=verbatim PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=2176 n_mb=265 tiles=384 (5.1s)
normal      T=160 routes= 230 ss_pure_128x256  align=ignore  PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=1024 n_mb=43 tiles=384 (55.5s)
normal      T=160 routes= 230 ss_pure_128x256  align=verbatim PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=2176 n_mb=265 tiles=384 (7.3s)
normal      T=160 routes= 230 ss_asm_256x128   align=ignore  PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=2048 n_mb=38 tiles=768 (15.9s)
normal      T=160 routes= 230 ss_asm_256x128   align=verbatim PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=3328 n_mb=261 tiles=768 (11.6s)
normal      T=160 routes= 230 sw_asm_128x64    align=ignore  PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=576 n_mb=53 tiles=864 (6.1s)
normal      T=160 routes= 230 sw_asm_128x64    align=verbatim PASS rel 3.379e-03 (worst row 3.591e-03) | ntp=1664 n_mb=273 tiles=864 (6.7s)
skewed      T=200 routes= 254 ss_asm_128x256   align=ignore  PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=896 n_mb=46 tiles=336 (73.0s)
skewed      T=200 routes= 254 ss_asm_128x256   align=verbatim PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=2304 n_mb=268 tiles=336 (8.4s)
skewed      T=200 routes= 254 ss_asm_64x256    align=ignore  PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=512 n_mb=58 tiles=384 (6.2s)
skewed      T=200 routes= 254 ss_asm_64x256    align=verbatim PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=1920 n_mb=278 tiles=384 (5.6s)
skewed      T=200 routes= 254 sw_asm_256x128   align=ignore  PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=896 n_mb=46 tiles=336 (9.0s)
skewed      T=200 routes= 254 sw_asm_256x128   align=verbatim PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=2304 n_mb=268 tiles=336 (7.5s)
skewed      T=200 routes= 254 ss_pure_128x256  align=ignore  PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=896 n_mb=46 tiles=336 (59.0s)
skewed      T=200 routes= 254 ss_pure_128x256  align=verbatim PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=2304 n_mb=268 tiles=336 (6.7s)
skewed      T=200 routes= 254 ss_asm_256x128   align=ignore  PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=1536 n_mb=40 tiles=576 (14.6s)
skewed      T=200 routes= 254 ss_asm_256x128   align=verbatim PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=3072 n_mb=263 tiles=576 (13.8s)
skewed      T=200 routes= 254 sw_asm_128x64    align=ignore  PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=512 n_mb=58 tiles=768 (14.1s)
skewed      T=200 routes= 254 sw_asm_128x64    align=verbatim PASS rel 3.387e-03 (worst row 3.526e-03) | ntp=1920 n_mb=278 tiles=768 (11.9s)
wide-scales T= 96 routes= 138 ss_asm_128x256   align=ignore  PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=1024 n_mb=39 tiles=384 (49.1s)
wide-scales T= 96 routes= 138 ss_asm_128x256   align=verbatim PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=1664 n_mb=261 tiles=384 (5.3s)
wide-scales T= 96 routes= 138 ss_asm_64x256    align=ignore  PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=512 n_mb=45 tiles=384 (3.4s)
wide-scales T= 96 routes= 138 ss_asm_64x256    align=verbatim PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=1152 n_mb=265 tiles=384 (4.4s)
wide-scales T= 96 routes= 138 sw_asm_256x128   align=ignore  PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=1024 n_mb=39 tiles=384 (9.9s)
wide-scales T= 96 routes= 138 sw_asm_256x128   align=verbatim PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=1664 n_mb=261 tiles=384 (4.1s)
wide-scales T= 96 routes= 138 ss_pure_128x256  align=ignore  PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=1024 n_mb=39 tiles=384 (36.9s)
wide-scales T= 96 routes= 138 ss_pure_128x256  align=verbatim PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=1664 n_mb=261 tiles=384 (6.6s)
wide-scales T= 96 routes= 138 ss_asm_256x128   align=ignore  PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=2048 n_mb=36 tiles=768 (33.0s)
wide-scales T= 96 routes= 138 ss_asm_256x128   align=verbatim PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=2816 n_mb=259 tiles=768 (18.6s)
wide-scales T= 96 routes= 138 sw_asm_128x64    align=ignore  PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=512 n_mb=45 tiles=768 (11.6s)
wide-scales T= 96 routes= 138 sw_asm_128x64    align=verbatim PASS rel 3.376e-03 (worst row 3.541e-03) | ntp=1152 n_mb=265 tiles=768 (10.0s)
ALL PASS (651s)
```

Per-expert routes: normal T=160: [83, 45, 21, 20, 15, 20, 17, 9]; skewed T=200: [160, 50, 25, 10, 4, 5, 0, 0]
(multi-block hot expert at every BM); wide-scales T=96: [40, 32, 14, 9, 11, 15, 9, 8]. The asm and the pure
decode give bitwise-identical outputs in every case (as they must: both dequantize exactly and
power-of-two scale factors commute with fp32 rounding).

History: an earlier full run (pre int32-scale layout) showed a pure-vs-asm difference in the skewed case
and an exactness assert in the wide case. **Its root cause was not identified**; the emulator was rerun from
scratch with the final code (not patched around it) and neither reproduces: all 8 experts × both decodes
exact, asm == pure bitwise in all 36 runs. The one emulator bug that was understood: float32 subnormal flush
on torch worker threads (fixed by integer BF16 field decoding).

Negative controls (`--mutate NAME`, normal case; each must FAIL, and does):

| mutation | what is broken | result |
|---|---|---|
| asm_shift | low-nibble sign shift 12 → 11 in the PTX text | unit test: 14,096 / 65,024 decodes wrong; GEMM rel 1.08 |
| asm_prmt | byte selector 0x4342 → 0x4243 in the PTX text | unit test: result depends on Triton's pack grouping; rel 1.32 |
| lohi | repack swaps which nibble holds which K half | rel 1.42 (asm and pure) |
| gateup | repack without the gate/up row interleave | rel 1.41 |
| sgroup | scale byte h instead of 2h (wrong 16-group) | rel 0.47 |
| xcol | activation columns h·16 instead of h·32 | rel 1.32 |

# K2a — custom NVFP4 W4A16 MoE experts for decode/verify (sm_90a, H200)

Status: v0 written blind (no GPU). Layout, bit tricks and work decomposition are **bit-exactly emulated on
CPU** (`k2a_emulate.py`, all cases pass `checks.compare`, see §8). Not yet compiled by nvcc; the CUDA text
was syntax/type-checked with clang against mock CUDA/torch headers. Submission file: `k2a_v0.py`
(`# nonce k2a-v0-1`).

## 1. What the call has to do, in bytes

Per expert: W13 = 8192 × 4096 E2M1 (16 MiB) + 8192 × 256 E4M3 scales (2 MiB) = 18.87 MB; W2 = 9.44 MB;
**28.31 MB per touched expert = 5.90 µs at 4.8 TB/s, 6.74 µs at 4.2 TB/s.** Activations are ≤ 2 MB. The
judge flushes L2 before every replay, so every weight byte comes from HBM.

The floor formula counts only 0.75 × the mean distinct experts (`ROUTING_SHARE`), so the realistic target is
the full touched count (back-solved from the floors, `notes/study/01-challenge.md`):

| load | bs | touched experts (mean) | routes (mean) | rows / touched expert | published floor | full bytes @4.8 TB/s | @4.2 TB/s | **v0 predicted** |
|---|---|---|---|---|---|---|---|---|
| decode 1 | 1 | 2.55 | 2.55 | 1.0 | 11.4 µs | 15.0 µs | 17.2 µs | **25–28 µs** |
| verify 1 | 8 | 8.42 | 13.7 | 1.6 | 37.3 µs | 49.7 µs | 56.8 µs | **66–75 µs** |
| decode 16 | 16 | 11.85 | 25.7 | 2.2 | 52.5 µs | 69.9 µs | 79.9 µs | **91–103 µs** |
| decode 32 | 32 | 15.98 | 49.8 | 3.1 | 70.8 µs | 94.3 µs | 107.7 µs | **120–136 µs** |
| verify 16 | 128 | 22.8 | 193.6 | 8.5 | 101.3 µs | 134.5 µs | 153.7 µs | **170–195 µs** |
| verify 32 | 256 | 24.4 | 385.3 | 15.8 (skewed) | 108.9 µs | 143.9 µs | 164.5 µs | **180–215 µs** |

v0 prediction = touched × 28.31 MB / (3.5–4.0 TB/s effective streaming, incl. SM imbalance and per-unit
epilogue bubbles) + ~7 µs fixed (route ≈ 2 µs, final ≈ 2–3 µs, 3 kernel boundaries ≈ 1 µs each). At bs 256 the
upper end adds MMA time for hot experts (2·385·50.3 M = 39 GFLOP ≈ 65–80 µs of mma.sync, mostly hidden).

**Speedup arithmetic.** `sglang_us = 13448.5` per step = 60 × geomean of SGLang's per-shape times, so SGLang's
geomean is ≈ 224 µs. Since every load has weight 60, our speedup = 224 / geomean(ours). v0 predicted geomean is
≈ 92 µs (4.0 TB/s) to ≈ 102 µs (3.5 TB/s) → **≈ 2.2–2.45×** (board leader 2.33×). The full-byte bound at
4.2 TB/s is a geomean of 77 µs → 2.9×; the published floors give 4.4× (unreachable, they count 0.75× bytes).

FLOPs never bind below bs 256 (bs 128: 19.4 GFLOP = 20 µs at the 989 TFLOP/s dense peak).

## 2. Instruction budget for in-register dequant (per SM, at 4.3 TB/s)

4.3e12 / 132 / 1.755e9 ≈ 18.6 B/clk/SM → **33 weight elements/clk/SM**. Issue: 4 warp-instr/clk/SM; INT ALU
(LOP3/SHF/PRMT): 64 lanes/clk/SM = 2 warp-instr/clk; IMAD and HFMA2 go to the FMA pipe.

v0 dequant per 32-bit word (8 codes → 4 bf16x2), chosen so that no word needs Marlin's 3-op form for all outputs:

```
o0 = w & 0x81C081C0                       // 1 LOP3
o1 = rotr(w,3) & 0x81C081C0               // SHF + LOP3
o2 = rotr(w,6) & 0x81C081C0               // SHF + LOP3
o3 = ((w<<6)&0x00C000C0) | ((w<<5)&0x01000100) | ((w<<11)&0x80008000)   // 3 shifts (IMAD.SHL) + 3 LOP3
```
= 8 ALU + 3 FMA-pipe ops per 8 elements (**1.0 ALU op/element**, vs ≈ 1.9 for SGLang's Marlin path,
`notes/study/03` §A.9). A full 4-rotation tiling of the target bit set {6,7,8,15,22,23,24,31} does not exist
(exhaustive search, reconfirmed this session), so one output needs a 3-part extraction; this layout is a
minimal-ALU member of the "3 rotations + 1 compound" family. Scales: 1 PRMT + 1 IMAD per {S,S} register,
amortized over 16 elements (0.125 ALU/elem, thanks to the k-permutation in §4). Scale multiply: 1 HFMA2 per
2 elements (FMA pipe).

Per k32 item per warp (1024 elements): ≈ 44 (dequant) + 8 (scales) + 16 (HFMA2) + 4·ng (MMA) + 2+ng (LDG) + ≈20
(addressing/cursor) ≈ 91 + 5·ng warp-instructions → issue ceiling ≈ 4·1024/96 ≈ **42 elem/clk/SM** (ng = 1),
37 (ng = 4); ALU ceiling ≈ 2·1024/46 ≈ 44 elem/clk/SM. Both above the 33 needed, but only by 1.1–1.3×:
issue rate is the first thing to measure (profile `sm__inst_issued` vs `dram__throughput`), and the reason v1
moves to wgmma (one instruction per warpgroup per m64nNk16, async) and trims the cursor arithmetic.

## 3. Work decomposition (all grids fixed; everything value-dependent lives in device memory)

1. **route** — 1 CTA × 1024 threads; warp e owns local expert e. Pass 1: ballot over tokens (8 slots per lane)
   → row counts. Warp 0: active-expert compaction (ballot + popc), exclusive scan of counts → compact row
   offsets; picks the W2 split factor `KS2 = 1 / 2 / 4` from `n_act·128` vs `4·#SMs`. Pass 2: ballot ranks give
   each (token, slot) its compact row in ascending token order (deterministic). Outputs a small int32 `meta`.
2. **gemm<1> (W13)** — persistent, grid = #SMs (132), 1 CTA/SM × 16 warps, `__launch_bounds__(512,1)`.
   Unit = (active expert a, 32-row tile rt) with rt ∈ [0,256); the tile holds gate rows 16rt..16rt+15 and up rows
   4096+16rt..+15. CTA b takes units b, b+132, … (a-major, so concurrent CTAs share an expert's X in L2). The
   16 warps split K = 4096 (8 k32 blocks each); partials are reduced in 32 KB of smem (warps 8–15 store,
   0–7 add, then 8 sums in fixed order: deterministic). Epilogue: × gs·2^8 in FP32, SiLU(gate)·up, BF16 store
   to `inter[compact row][16rt + r]`. No K-split across CTAs → no partial buffers, no counters.
   Each touched expert's W13 is streamed from HBM exactly once; an expert with more than 8·NG rows loops over
   token chunks inside the unit, re-reading its 4 KB/warp slice from L1/L2.
3. **gemm<2> (W2)** — same kernel, MODE 2. Unit = (a, rt ∈ [0,128), ks ∈ [0,KS2)); warps split K/KS2. Output
   FP32 partials `Y[ks][compact row][4096]` (plain stores, deterministic).
4. **final** — 1 CTA (256 threads) per token: Σ_ks Y, y → BF16, f = y·bf16(1+w) in FP32, RMSNorm over 4096
   (shuffle + 8-entry smem reduce), × topk weight, Σ over the token's local slots, BF16 out; zero rows for
   tokens without local routes.

**Balance** (units per CTA at the mean touched count, 132 CTAs): W13 = n_act·256 units → bs 1: 4.9 (max 5 →
97 %), bs 8: 16.3, bs 16: 23, bs 32: 31, bs 128/256: 44–47. W2 = n_act·128·KS2: n_act = 1 → KS2 = 4 (3.9/CTA),
2 → 4 (7.8), 3–4 → 2 (5.8–7.8), ≥ 5 → 1 (≥ 4.8). Units are equal in bytes; hot experts' units cost more MMA,
but a-major round-robin spreads every expert's 256 (128) units over all CTAs, so skew does not pile onto SMs.

**Bytes in flight.** Each warp keeps a ring of RING = 4 items (16 B weights + 4 B scales per lane) → 512 threads
× 80 B = 40 KB per SM, above the 16–31 KiB Little's-law target at 0.5–1 µs loaded latency (`03` §A.8). The ring
runs across unit/chunk boundaries, so loads for the next unit are in flight during an epilogue's syncs.

## 4. Layouts (prepare() repacks losslessly; NVFP4 stays NVFP4, 0.5625 B/element)

* **Swap-AB mma.sync m16n8k16** (bf16, FP32 accumulate): A = 16 weight rows × 16 k (dequantized in registers),
  B = 16 k × 8 tokens, C = 16 rows × 8 tokens. A warp tile is 2 m16 tiles (ρ = 0,1) × k32 per item.
* **k-permutation.** Within a k32 block, lane t's B registers come from one 16 B load `X[token][32kb+8t .. +7]`:
  step s uses elements {4s,4s+1} as b0 and {4s+2,4s+3} as b1. The A side uses the same physical k, so the MMA
  sums the right products. Consequence: each lane touches a single 16-wide scale group per k32 block
  (group 2kb + (t>>1)), halving scale decode work.
* **WQ** int32 `[32][NRT][128][32 lanes][4]`; flat uint4 index `((e·NRT + rt)·128 + kb)·32 + lane`
  (512 B contiguous per warp load). Word wi = 2ρ + h holds row 16ρ + 8h + g (g = lane>>2) of the tile,
  k = 32kb + 8t + i, i = 0..7, with bit positions (m0,m1,m2,sign) per element:
  `e0 (6,7,8,15) e1 (22,23,24,31) e2 (9,10,11,18) e3 (25,26,27,2) e4 (12,13,14,21) e5 (28,29,30,5)
  e6 (0,1,3,4) e7 (16,17,19,20)`. o_j = {e_2j, e_2j+1}; registers for (ρ, step s): a0 = o_2s(word 2ρ),
  a1 = o_2s(word 2ρ+1), a2 = o_2s+1(word 2ρ), a3 = o_2s+1(word 2ρ+1).
* **W13 row order**: prepacked row 32rt + 16ρ + i ↔ original row ρ·4096 + 16rt + i, so a lane holds gate and up of
  the same intermediate index in tiles 0 and 1 (SwiGLU without cross-thread traffic after the reduction).
* **SQ** uint8 `[32][NRT][128][64]`: byte gi·32 + g·4 + (2ρ+h) = scale of row 16ρ+8h+g, group 2kb+gi. Lane reads
  the u32 at `blk·16 + (t>>1)·8 + g` (64 B per warp, each address read by 2 lanes; no byte inflation).
* **Scale re-encoding (lossless)**: b' = ((e+10)<<3) | f for |sf| = 2^e (1+f/8) — normals are b' = b + 24,
  E4M3 subnormals are renormalized, zero scales zero their codes, negative scales flip their codes' signs, NaN →
  fallback. Device: `{S,S} = __byte_perm(sw, 0, 0x4040 + 0x0101·bi) · 16 + 0x75807580` = BF16(|sf|·2^118).
* **Exact dequant**: o_j = e2m1·2^-126 (code 1 is a BF16 subnormal), × BF16(sf·2^118) via `fma.rn.bf16x2 a,b,-0`
  = e2m1·sf·2^-8, exact (≤ 6 significant bits, magnitude in [2^-18, 10.5]; proof `03` §C). The FP32 global
  scale × 2^8 is applied once to the FP32 accumulator.
* **Reduction smem** `red[(w8·8NG + (ρ·NG+q)·4 + i)·32 + lane]`, 8·NG KB. Epilogue maps (row r, token n) →
  lane = 4(r&7) + ((n&7)>>1), i = 2((r>>3)&1) + (n&1) (checked against CuTe's CLayout in the emulator).

## 5. Numerics (rounding points)

Mirrors SGLang (baseline.py + Marlin + fused_post_expert_norm_weighted_sum) where it keeps BF16, and is
more accurate elsewhere:

| point | SGLang | K2a v0 |
|---|---|---|
| weight e2m1·sf | BF16 exact (scales re-encoded) | BF16 exact |
| global scale | rounded to BF16, applied in BF16 after acc→BF16 | FP32 on the FP32 accumulator |
| h = x·W13ᵀ | BF16 | FP32 (never stored) |
| SiLU(gate)·up | BF16 (from BF16 h) | FP32 math, one BF16 rounding (it is the MMA B operand) |
| y = inter·W2ᵀ | BF16 | FP32 partials, rounded to BF16 before the norm (as SGLang) |
| (1+w) | BF16 | BF16 |
| y·(1+w), RMSNorm, ×g, Σ | FP32 | FP32 |
| output | BF16 | BF16 |

The template's extra BF16 rounding of y·(1+w) is deliberately not reproduced (SGLang does not do it). The
emulated error vs the FP32 reference is 3.8e-3 (tolerance 1.5e-2).

## 6. CUDA-graph safety

No host syncs or value-dependent host control flow in `kernel_fn`; `torch.empty` buffers sized by M only;
all launches on `at::cuda::getCurrentCUDAStream()`; n_act, unit counts, KS2, chunk counts all read from
device memory; padded token columns clamp to a valid row (results discarded). Compilation (`load_inline`,
`TORCH_CUDA_ARCH_LIST=9.0`) and a self-test on 6 shapes (incl. skewed bs 128 and bs 257) run in prepare(), so
every template instance is loaded before capture. A failed build prints the nvcc output tail and a failed
self-test prints the error; both fall back to a graph-safe PyTorch path (`FALLBACK = True` forces it).

## 7. Risks

1. **Issue/INT throughput** may cap streaming below 4 TB/s (§2 headroom only 1.1–1.3×). Mitigation: v1 wgmma
   (RS, A = dequantized weights in registers, `m64nNk16`), fewer cursor ops per item (precomputed per-unit
   base pointers), PRMT-free scales via `.H0_H0` operand swizzle.
2. **Per-unit epilogue bubble** (3 `__syncthreads` + smem reduce per (unit, chunk)); at bs 1 a CTA does ~5
   units of 73.7 KB. The 40 KB ring hides part. Mitigation: double-buffered `red`, fewer syncs, or 8-warp K-split
   with 2 tiles in flight.
3. **Kernel boundaries** (4 launches) cost ~1 µs each under graphs; at bs 1 that is ~15 % of the time.
   Mitigation v1: fuse route into gemm<1>'s prologue (every CTA recomputes routing from 8·M ids), PDL
   (`griddepcontrol`) so gemm<2>/final prologues overlap tails; or a single persistent kernel with a grid barrier.
4. **BF16 subnormal handling**: code 1 (0.5) decodes to a BF16 subnormal before the scale multiply; correctness
   needs `fma.rn.bf16x2` not to flush subnormal inputs. SGLang's Marlin BF16 FP4 path relies on the same
   (`dequant_skip_flop`), and PTX has no `.ftz` for bf16x2 — but the emulator cannot test hardware behaviour.
   The prepare() self-test would catch it (error ≫ tolerance) and fall back.
5. **Register pressure** at NG = 4 (acc 32 + B 16 + ring 20 + …) under the 128-reg cap of 512 threads/SM;
   spills cost speed, not correctness. `-Xptxas=-v` prints usage in the log.
6. **Skew at bs 128/256**: a hot expert with T ≫ 32 rows loops over ceil(T/32) chunks per unit, re-dequantizing
   (INT cost × chunks) and MMA-bound; balanced across SMs by the a-major round robin, but it lengthens every
   CTA. v1: NG = 8 accumulators via wgmma (n up to 256 per instruction) removes the chunk loop.
7. **Allocation size**: `Y` is `[4, 8M, 4096]` FP32 (135 MB at M = 257) from the caching allocator; only the
   touched rows are written/read.
8. **L1 pollution**: weights use `ld.global.nc` (L1-allocating) — could evict X lines; v1 uses
   `L1::no_allocate` / TMA for weights.

**Compile-risk spots** (first nvcc contact): inline PTX `fma.rn.bf16x2` with an `"r"` constant operand;
`mma.sync ... {%0..%3} reused as C` with `"+f"`; `__launch_bounds__(NTHR,1)` on a template kernel with 32 KB
static smem; `min(int,int)` in device code; `__ldg(const uint4*)` on reinterpret-cast bf16 pointers;
`C10_CUDA_KERNEL_LAUNCH_CHECK` include path; load_inline binding of `int64_t` arguments.

## 8. Emulation (`k2a_emulate.py`, CPU, torch 2.5.1)

Imports prepare()'s repack from `k2a_v0.py`, re-implements every device formula from the .cu text, takes the
mma fragment maps from CuTe's SM80_16x8x16 layouts, starts kernel-written buffers as NaN, simulates the
persistent cursor/ring loop of every CTA (asserting each (active expert, row tile, k-split) is visited exactly
once and every ring slot holds the item being consumed), and checks each output against a copy of
`ref_kernel` with `checks.compare`. Unit tests: 16 codes × 8 positions through pack → decode; all 256 scale
bytes (252 finite non-zero exact, zero/sign/NaN flagged); all 4032 code × scale products exact in BF16;
fragment-map bijections; epilogue index map vs CuTe CLayout; MMA wiring vs A@B.

Results: see the table appended at the end of this file by the run (`EMULATION RESULTS`).

## 9. v1 roadmap (ordered by expected gain)

1. Measure (first `okc check`): per-kernel times from the profile, achieved DRAM BW, issue utilization.
2. Fuse route into gemm<1> + PDL between gemm<1>→gemm<2>→final (−3–5 µs at every shape; ~15 % at bs 1).
3. wgmma RS (`m64n8..32k16`, weights in A registers via the same prepack idea, B = X tile in smem via TMA),
   setmaxnreg producer/consumer split; cp.async.bulk / TMA weight ring with 32–64 KB in flight.
4. Prefetch W2 tiles into smem during gemm<1>'s tail (weights do not depend on activations).
5. Final kernel fused into gemm<2> via a per-chunk completion counter (last CTA normalizes).

## EMULATION RESULTS (2026-10-23, `python3 -I k2a_emulate.py`, CPU torch 2.5.1, 214 s)

```
primitive unit tests: OK (16 codes x 8 positions, 252 finite nonzero E4M3 scales, 4032 products exact,
                          fragment maps, epilogue index map, MMA wiring)
bs=   1 normal  PASS  rel-L2 3.839e-03 (worst row 3.839e-03) | n_act=2  routes=2   KS2=4 NG=1 units1=512  units2=1024
bs=   4 normal  PASS  rel-L2 3.821e-03 (worst row 3.918e-03) | n_act=7  routes=8   KS2=1 NG=1 units1=1792 units2=896
bs=   8 normal  PASS  rel-L2 3.789e-03 (worst row 3.839e-03) | n_act=11 routes=14  KS2=1 NG=1 units1=2816 units2=1408
bs=  16 normal  PASS  rel-L2 3.749e-03 (worst row 3.867e-03) | n_act=12 routes=36  KS2=1 NG=1 units1=3072 units2=1536
bs=  33 normal  PASS  rel-L2 3.736e-03 (worst row 3.959e-03) | n_act=16 routes=44  KS2=1 NG=2 units1=4096 units2=2048
bs= 128 normal  PASS  rel-L2 3.753e-03 (worst row 3.907e-03) | n_act=26 routes=168 KS2=1 NG=4 units1=6656 units2=3328
bs= 128 skewed  PASS  rel-L2 3.740e-03 (worst row 3.913e-03) | n_act=4  routes=305 KS2=2 NG=4 units1=1024 units2=1024
bs= 257 normal  PASS  rel-L2 3.758e-03 (worst row 3.940e-03) | n_act=28 routes=366 KS2=1 NG=4 units1=7168 units2=3584
bs=   4 wide-scales PASS rel-L2 3.693e-03 (worst row 3.749e-03) | n_act=6   (block scales log-uniform 2^-6..448)
ALL PASS
```
Coverage: KS2 ∈ {1,2,4}, NG ∈ {1,2,4}, multi-chunk hot experts (skewed bs 128: 305 routes on 4 experts →
up to 4 chunks per unit), padded token columns, unit coverage + ring consistency for all 132 CTAs.

Negative controls (`K2A_MUTATE=… K2A_BS=4`), each must FAIL and does: wrong scale lane offset
(`(t&1)` for `(t>>1)`) → 4.7e-1; swapped a1/a2 register wiring → 1.3; wrong B k-permutation → 1.3; epilogue
column off by one row (`rr^1`) → 1.4.

The submission's own PyTorch fallback passes `compare` on CPU (5.5e-3 at bs 2 and 9) and its self-test
reference `_ref_fp32` equals the copied `ref_kernel` exactly.

---

# v1 (2026-10-23 evening): diagnosis of v0's ~1.9 TB/s, the fixes, and the lab sweep

## Measured v0 (check ff4ae9cda6f34ffc; decode_layer 6811699d6da448ef)

W13 streams ≈ 1.9 TB/s at NG = 1/2 at every shape, ≈ 1.1 TB/s at NG = 4; W2 is slower (bs 1: 23.6 MB in 19.7 µs ≈ 1.2 TB/s).
route takes 9.0 / 16.4 µs at bs 128 / 256; final takes 6.3 µs at bs 1. In the layer, v0 loses to Marlin with the rung-1
Triton post-norm at every T. **New bar per shape:** beat "Marlin + Triton post-norm" (≈ stock − 15–20 µs at bs ≤ 16),
which needs ≈ 3 TB/s effective. Judge by per-shape wall µs only; the profile's kernel times overlap.

## Why v0 stalls at ~14.4 GB/s per SM (from the design, consistent with every number above)

1. **The B operand (X) sits on the critical path of every item.** v0 loads X at use (`x_fetch` at consume time). Each of
   the 16 K-split warps reads a *different* K-slice of X, so the CTA's X footprint is the full T × 4096 × 2 B per expert.
   Weights stream through L1 (`ld.global.nc` allocates), evicting X, so each item waits an L2 round trip (≈ 0.6–1 µs
   loaded), and the 4-deep weight ring cannot hide that. Little's law: 40 KB in flight at 1.9 TB/s means 2.8 µs of
   effective latency per byte, implausible for HBM, but consistent with ~1 X round trip serialized per item per warp.
2. **NG = 4: L2 traffic, not HBM.** Per item a warp moves T/8 × 512 B of X against 576 B of weights. With no reuse across
   the 16 K-split warps and L1 thrashed, X re-reads from L2 are 2–3.5× the HBM stream at T = 16–32, which fits 1.1 TB/s.
3. **W2 at bs 1:** KS2 = 4 gave NKB = 2 items per warp between epilogues. That is 3 `__syncthreads` plus an smem reduction
   for every 2 × 576 B per warp.
4. **route:** each warp's ballot loop does dependent global loads of ids, ~1 µs per iteration × 2 passes × M/32.
5. **final:** per slot, a dependent load → reduce → sync chain, with up to 4 KS partial rows read serially.
6. Lesser: 1 CTA/SM × 512 threads caps registers at 128, so RING 4 was the most that fit next to NG = 4 accumulators.
   Round-robin units mean a CTA hops experts every unit (no X locality).

Unit granularity at bs 1 (≈ 5 units per CTA) costs ~3 % in balance; it is not the cause.

## v1 changes (k2a_v1.py, `# nonce k2a-v1-1`)

| change | targets | knob |
|---|---|---|
| X prefetched XR items ahead in a register ring (own cursor) | 1 | `xr` = 2/2/1 for NG = 1/2/4 |
| weights via `ld.global.nc.L1::no_allocate` (keeps X slices L1-resident) | 1, 2 | `na` = 1 |
| deeper weight ring | 1 | `ring` = 8/6/4 |
| contiguous cost-balanced unit ranges per CTA (consecutive units share the expert, so X stays hot) | 1, 6 | `asg` = 1 |
| NG = 4: 4 row tiles per CTA unit share X (K split over 4 warps), 4× less X per weight byte through L1/L2 | 2 | `kw` = 4 |
| W2 at NG = 1: 8 warps × 2 CTAs/SM; KS2 ∈ {1, 2} chosen from the W2 variant's units per expert | 3 | `nw/kw/minb` = 8/8/2 |
| route stages ids in smem (M ≤ 512) | 4 | — |
| final: 512 threads, all slots' loads issued first, one block reduction for all 8 slots | 5 | — |

The layout, dequant and numerics are unchanged (all v0 emulation checks still apply). The kernel is now one template
`gemm_kernel<MODE, Cfg<ng, ring, xr, na, asg, nw, kw, probe, minb>>` with a generated dispatch, shared verbatim with the
lab. **Prediction (conditional on cause 1 being dominant):** 3.0–3.6 TB/s on W13 at NG = 1/2, which gives
bs1 ≈ 30–36 µs, bs8 ≈ 80–95, bs16 ≈ 110–130, bs32 ≈ 150–175 (vs Marlin+tpn ≈ 56–61 / 130–135 / 167–172 / ≈ 200).
For verify (NG = 4, KW = 4): ≈ 2.5–3 TB/s, so bs128 ≈ 230–280 and bs256 ≈ 260–320 (vs ≈ 471 / 614). If the lab shows
`noX0 ≈ v0` (X not the bottleneck), cause 1 is wrong. Then the stream probes say whether the memory pattern itself
(S.r4/S.r8) or issue rate is the limit, and v2 moves to cp.async/TMA staging into smem.

## Lab: k2a_lab_v1.py (check only, never submit)

`kernel_fn` is baseline.py verbatim. `prepare()` runs the baseline prepare, repacks for K2a, compiles **one**
`load_inline` extension with 32 gemm variants (+ route/final), and times each at the six scored shapes (meta.json seeds
401–406). Timing: CUDA-graph replay, 256 MiB flush before every replay, median of 30. It prints raw µs and TB/s on the
touched experts' weight bytes after subtracting the measured 1-kernel replay floor. Every non-probe variant's output is
compared with the v0 variant's (`maxrel`, flag > 1e-2). Per shape it also times the full v1 pipeline, **stock**, and
**"marlin+tpn" = rung-1's kernel_fn** (r1_glue.py, copied verbatim: Marlin + Triton post-norm at all T + its other
levers), each checked against an FP32 reference. Read the `===== muse_moe_w4a16 k2a-lab-v1 LAB SUMMARY` block printed
last:

- `SUM REF marlin+tpn` / `SUM REF stock` / `SUM PIPE k2a v1`: per-shape wall µs. These are the decision rows.
- `SUM W13 <fam>` / `SUM W2 <fam>`: per shape `us/TB-s`, then `ng:regs/spillB`, then max rel vs v0.
- `SUM BEST` per shape and mode. `SUM EST` = route + best W13 + best W2 + final, compared with marlin+tpn: the best
  composable pipeline from this run.

| family | isolates |
|---|---|
| v0 | v0's kernel (ring 4, X at use, `__ldg`, round robin, 16-warp K-split); should reproduce ≈ 1.9 TB/s |
| na | + L1 no_allocate for weights |
| xpf | + X prefetch |
| cont | + contiguous cost-balanced assignment |
| v1 | the shipped config (= cont + ring 8 at NG 1; ring 6 at NG 2; KW 4 at NG 4) |
| w8 | 8 warps × 2 CTAs/SM, K-split 8, ring 8 |
| kw4 | 4 row tiles share X, K split 4 |
| kw1 | 16 row tiles per CTA unit, no K-split: no smem reduction, no `__syncthreads` |
| kw1n8 | 8 warps × 1 CTA/SM (255 regs), no K-split, 64 tokens per pass: hot experts read once up to 64 rows |
| S.r4 / S.r8 | stream-only probes (no X, no dequant, no MMA): memory ceiling of v0's / v1's access pattern |
| noX0 / noX1 | v0 / v1 with a constant B operand: what X loads cost before / after the prefetch |

Compile estimate: v0's 6 instances took 38.5 s total. 32 instances are ≈ 60–110 s, inside the 240 s guard. A failure
prints and skips; a CUDA fault would end the run, but every result line is flushed first.

## Emulation status (k2a_emulate.py now emulates v1)

Generalized to the (row-group × K-split) decomposition, cost-balanced assignment, the X prefetch ring (slot identity
asserted), the KW = 1 register epilogue, and a literal flat-smem reduction (writer, adder and reader slab formulas).
Results: see `EMULATION RESULTS v1` below. Every lab family's cursor, ring and coverage was checked at 7 routings
(84 pairs). Math was checked for v1 at all 8 cases + wide scales, the alternate set (v0-like round robin, KW = 1 at
NG 2/4, KW = 4 and 8-warp W2), and NG = 8 KW = 1 on a skewed bs 100 input.

One emulator caveat: three CPU runs of the KW = 4 W13 + 8-warp W2 pair at bs 33 / seed 403 gave 6e-3–1.2e-2 instead of
3.76e-3. Identical reruns gave 3.756e-3. Per-expert runs were deterministic and ≤ 3e-3 against a dense reference, and
KW = 4 W13 matched KW = 16 to 1.2e-5. Some runs, but not others, overlapped other heavy jobs on this Mac. A formula bug
would reproduce, so I treat this as CPU-side flakiness. The lab's on-GPU `maxrel` vs the v0 variant is the backstop.

## v1 risks

- Register pressure: NG 1 with ring 8 + X ring needs ≈ 120–125 of the 128 regs at 512 threads. Spills cost speed only;
  `_build()` and the lab print regs/spill per instance.
- Contiguous assignment balances by a guessed cost (4 + 3 per extra chunk); a wrong guess shows as `cont` < `xpf` at bs 16.
- Hot experts at verify still re-dequantize per 32-token chunk (kw1n8 measures the 64-token alternative).
- If cause 1 is not dominant, v1 will land near v0. The lab then picks the direction: smem staging via cp.async/TMA,
  or a wgmma rewrite.

## EMULATION RESULTS v1 (2026-10-23, `python3 -I k2a_emulate.py`, 367 s)

```
primitive unit tests: OK
[v1] bs=   1 normal  PASS 3.839e-03 (worst row 3.839e-03) | n_act=2  routes=2   KS2=2 NG=1 W13 512 units (max/CTA 4) W2 512 (max 2)
[v1] bs=   4 normal  PASS 3.821e-03 (3.918e-03)            | n_act=7  routes=8   KS2=1 NG=1
[v1] bs=   8 normal  PASS 3.789e-03 (3.839e-03)            | n_act=11 routes=14  KS2=1 NG=1
[v1] bs=  16 normal  PASS 3.749e-03 (3.867e-03)            | n_act=12 routes=36  KS2=1 NG=1 W13 max/CTA 27 vs mean 23.3 units
[v1] bs=  33 normal  PASS 3.736e-03 (3.959e-03)            | n_act=16 routes=44  KS2=1 NG=2
[v1] bs= 128 normal  PASS 3.754e-03 (3.926e-03)            | n_act=26 routes=168 KS2=1 NG=4 (KW=4)
[v1] bs= 128 skewed  PASS 3.740e-03 (3.913e-03)            | n_act=4  routes=305 KS2=2 NG=4
[v1] bs= 257 normal  PASS 3.758e-03 (3.940e-03)            | n_act=28 routes=366 KS2=1 NG=4
[alt] bs=  1 / 33 / 128-skewed PASS 3.753e-03 / 3.738e-03 / 3.753e-03 (v0-like RR, KW=1 NG 2/4, KW=4 W2, 8-warp W2)
[v1] bs=   4 wide-scales PASS 3.693e-03
ALL PASS
negative controls (bs 4): scale_lane 4.7e-1, frag 1.3, kperm 1.3, rowperm 1.4 -> all FAIL as required
```
Also passing: NG = 8 KW = 1 (kw1n8) on skewed bs 100 (3.75e-3); w8 W13 NG 1 with kw4 W2 NG 1 at bs 16 (3.75e-3).
Cursor/ring/coverage checked for all 18 lab families at 7 routings.

# K2a v2: NVFP4 x BF16 MoE decode/verify kernel, designed to measured H200 constants

Status (2026-10-23): written blind (no GPU). The CUDA source passes a clang CUDA syntax/type check (sm_90, device and
host passes, template instantiation confirmed by mutation). The layouts, routing, unit decomposition, staging,
dequant, MMA fragment wiring, flush and finalize are **bit-exactly emulated on CPU** and pass every case (section 9).
Nothing has been compiled by nvcc or timed.

Files:
- `k2a_v2.py` is the submission (`# nonce k2a-v2-1`). The route table defaults to the incumbent, and an optional
  prepare-time autotune can switch individual buckets to v2.
- `k2a_lab_v2.py` is check-only (`# nonce k2a-lab-v2-1`). It is generated from k2a_v2.py by
  `scratchpad/k2a_v2/gen_lab.py`, so the CUDA source is byte-identical.
- `k2a_v2_emulate.py` is the CPU emulator.

## 0. The measured constants this is designed to (R1)

| constant | value | source |
|---|---|---|
| compute-attached stream (dequant + BF16 scale + 4 mma per 16 B) | **3.02 TB/s** (1 GiB), 2.47 eff on 3×W13 | stream_lab D, S1 u4 t256 c4 |
| config that achieves it | **256 thr × 4 CTAs/SM, 64 regs, 4 × LDG.128 per thread, issue → consume** | same |
| same at 512 thr × 1 CTA/SM / 2 CTAs/SM (v0/v1 style) | 2.76 / 2.93 (106 regs) | stream_lab D |
| dequant only | 3.36 TB/s | stream_lab D w1 |
| per-launch ramp/tail of that kernel | t0 ≈ **4.2 µs** (56.6 MB at 2.47 eff vs 3.02 asymptotic) | derived from D |
| extra kernel launch inside a graph | +0.97 µs | stream_lab floor |
| split-K reduction | fused FP32 atomics +0.35 µs; a second kernel +5.4 µs | stream_lab B2 |
| plain LDG streaming ceiling | 4.4 TB/s with ≥ 32 KiB/SM in flight | lab M3 / stream_lab A |
| routing per busiest GPU (40 seeds each, `scratchpad/k2a_v2/routing_stats.py`) | touched 2.6 / 11.9 / 15.5 / 8.2 / 22.4 / 24.7; routes 2.6 / 26 / 48 / 14 / 192 / 399; **max tokens on one expert: mean 1 / 7.6 / 14.7 / 4.4 / 64 / 120, worst 1 / 16 / 32 / 8 / 128 / 256** (bs 1/16/32/8/128/256) | CPU generate_input clone |

What v0/v1 got wrong:
- They used 512 threads × 1 CTA/SM at 93–128 regs, which is too few warps to keep loads in flight once compute is
  attached (D measures exactly this).
- X (the B operand) sat on the weight stream's critical path: loaded at use in v0, and through a register ring in v1.
- Hot experts were re-read once per 32-token chunk.

## 1. Decomposition

**One kernel template, three phases, 1–3 launches.** `k2a2_kernel<Cfg<NC,U,PF,TW,XE,MINB>>` has 256 threads. The grid
is persistent: SMs × min(occupancy, MINB), with the occupancy queried in prepare after setting the max-shared
carveout. A runtime phase mask selects:

1. **ROUTE (in the first launch, every CTA).**
   - Pairs are staged as int8 local-expert ids in smem.
   - Per-warp histograms come from smem atomics. Warp 0 computes the expert prefix (compact offsets), passes per
     expert, and per-(warp, expert) bases.
   - Ranks are deterministic: each warp walks its contiguous pair range in 32-pair chunks; `__match_any_sync` gives the
     rank within a chunk, and the group leader advances the base.
   - The result is compact row = off[e] + (rank of the pair among e's pairs in pair order), identical in every CTA
     (the emulator checks it against a stable sort).
   - The device chooses `KS1`/`KS2` (split-K): the smallest power of two ≤ 16 with units ≥ ks_target × grid.
   - CTA 0 writes the meta (header, counts, offsets, passes, pair→compact row) for later launches.
2. **G1 = W13 + SwiGLU.**
   - Units are u → (expert e, pass j, row group rg, k-split ks). The pass index is fastest, so passes over the same
     weights are adjacent. They are assigned round-robin (`asg=0`) to concurrently running CTAs, or as contiguous
     cost-balanced ranges (`asg=1`).
   - A unit is RW = 8/TW row tiles of 16 rows over KR = 4096/KS of K, for up to TPP = TW·8·NC tokens.
   - Each warp owns one 16-row tile (and, if TW > 1, one token group).
   - The B operand (x rows of the pass's tokens) is staged per k-slice into a padded smem tile. The warp's first U
     weight loads are issued *before* the staging and its `__syncthreads`, so they stay in flight across it.
   - Flush:
     - KS1 = 1: SwiGLU in-register (the W13 tile packs gate rows 8r..8r+7 and up rows 8r..8r+7, so a lane holds both
       halves of one intermediate index), with a BF16 store to `inter`.
     - KS1 > 1: FP32 `atomicAdd` of gate/up partials into `h`.
   - If W2 will use atomics, G1 also zeroes `y` rows [0, R).
3. **G2 = W2.**
   - Same unit machinery over W2's 256 tiles.
   - The B operand is staged from `inter` (BF16) or, when KS1 > 1, from SwiGLU(h) computed in FP32 while staging and
     rounded once to BF16.
   - Flush: stores to `y` (KS2 = 1) or FP32 atomics.
4. **FIN.**
   - Tokens are strided over CTAs, 16 columns per thread.
   - Two passes over y (sum of squares for all 8 slots, then one block reduction, then the weighted sum):
     bf16(y)·bf16(1+w) → RMSNorm → × topk weight → Σ slots → BF16. Tokens with no local route get zeros.
   - If W13 used atomics, FIN zeroes `h` rows [0, R), which keeps the invariant "h is all-zero between calls"
     (prepare zeroes it once and again after the self-test/autotune).

**Launch plans** (`fuse`):

| fuse | launches | what crosses a launch boundary |
|---|---|---|
| 0 | 3: (ROUTE+G1), G2, FIN | each phase is its own launch |
| 1 (default) | 2: (ROUTE+G1), then G2 + bounded grid barrier + FIN | one launch boundary, one barrier |
| 2 | 1: ROUTE + G1 + barrier + G2 + barrier + FIN | no launch boundary; W2's first loads are issued right after the barrier |

- The grid barrier is the cooperative-groups pattern: sense via a generation counter, `__threadfence`, and a
  clock64-bounded spin (2^26 cycles) that sets an error flag (R9).
- Inside a fused launch, data written in-kernel (inter/h/y/meta) is read with `__ldcg` (never `.nc`). Weights use
  `ld.global.nc.L1::no_allocate` (TW = 1) or `__ldg` (TW > 1, so the token-group warps share one fetch in L1).
- There are no memset nodes and no reduce kernels. All experts of a call share each launch.

## 2. Register budget (target ≤ 64 regs at 4 CTAs/SM, ≤ 80 at 3) and the hot-expert decision

Per-warp work item (one 16 B load per lane, 512 B per warp) is **one m16 tile × k64**:
- Words x/y hold row g at k = 64kb+16t+0..15, and words z/w hold row g+8.
- In MMA terms that is 4 k16 steps of one m16 tile. Logical k-slot 2t+{0,1} ↔ physical 16t+4s+{0,1}, and 2t+8+{0,1}
  ↔ 16t+4s+{2,3}. A and B use the same map, so each lane's B for an item is one contiguous 32 B smem read.
- Each lane covers exactly one NVFP4 scale group per row per item. Scales are 2 bytes per lane per item, so one u32
  load serves 2 items.

| live registers per thread | NC=1 (8 tok/pass) | NC=2 | NC=4 |
|---|---|---|---|
| weights in flight (U=4 × uint4) | 16 | 16 | 16 |
| scales in flight (U/2 u32) | 2 | 2 | 2 |
| accumulators (4 per 8-token chunk) | 4 | 8 | 16 |
| dequantized A (16 bf16x2, transient) | 16 | 16 | 16 |
| B fragments (2 × LDS.128, transient) | 8 | 8 | 8 |
| pointers / loop / unit state (est.) | ~16 | ~16 | ~16 |
| **total (est.)** | **~62 → MINB 4** | **~66 → MINB 4 (may spill a little)** | **~74 → MINB 3** |

PF = 1 (register double buffer) doubles the weights-in-flight row, which only fits at U = 2.

**Why not "all tokens of a hot expert in one weight pass per warp".** The accumulator footprint is inherent: 16 rows ×
T tokens / 32 lanes = **T/2 FP32 regs per thread**.
- T = 128 needs 64 regs and T = 256 needs 128, before any operand.
- wgmma m64nNk16 has the same footprint (N/2 per thread for m64 per warpgroup). It removes the B-operand register loads
  but does not make the accumulators smaller.
- So at 64–80 regs a warp holds at most ~32–48 tokens.

v2 therefore splits an expert with T > TPP tokens into ceil(T/TPP) passes. Weights still come from HBM once, for
three reasons:
- Passes of the same (expert, rg, ks) are adjacent units.
- Round-robin puts them on different CTAs that start together and do equal work per item.
- The 2nd..n-th read lands well inside the ~20 µs L2 residency (60 MB / 3 TB/s).

TW (token-group warps sharing one weight fetch through L1 inside a CTA) is the alternative mechanism. The lab compares
it as L5/L7: 32/64 tokens per pass at the same register cost.
- At bs 256 the passes are 29.7 vs 24.7 touched experts, so the SMs process +20% bytes.
- The extra ~140 MB is L2 traffic, not HBM.

**X restaging** (the price of a 64–80 reg budget):
- Each CTA unit covers 128 rows (RW = 8). The B operand is re-staged once per unit, which is X bytes per weight byte
  ≈ 2·T/(9·RW): 0.03 at T = 1, 0.9 at T = 32.
- Per call that is routes × 786 KB from L2: 2 / 11 / 20 / 38 / 151 / **314 MB** at bs 1 / 8 / 16 / 32 / 128 / 256.
- L2→SM bandwidth is **not** a measured constant yet, so the lab measures it (`SUM L2->SM`).
- If it is the binding limit at bs 256, the fix is RW = 16 (2 tiles per warp at NC ≤ 2, accumulators 8·NC).

## 3. Exactness and rounding points

- E2M1 × E4M3 is exact in BF16. This uses v0's GPU-validated bit layout (deq4: 8 LOP3/SHF + 3 shift ops per 8 values,
  vs ~14 for Marlin's dequant that lab D measured). Scales are re-encoded losslessly so that bf16 bits = b·16 + 0x7580
  = sf·2^118.
- `fma.rn.bf16x2(code·2^-126, sf·2^118, -0)` = e2m1·sf·2^-8 exactly. The FP32 global scale × 2^8 is applied to the
  FP32 accumulator.
- SwiGLU runs in FP32 and is rounded once to BF16, since it is the W2 MMA operand.
- y is rounded to BF16 before (1+w) and the norm, as SGLang's Marlin output → Triton post-norm does. The template's
  extra BF16 rounding of y·(1+w) is deliberately not reproduced.
- The norm, the weighting and the sum are in FP32, with BF16 output.
- Emulated error vs the FP32 reference is 3.73–3.77e-3 (tolerance 1.5e-2), the same as v0/v1 on the GPU (3.7e-3).

## 4. Predicted µs per scored shape (from the constants above; `scratchpad/k2a_v2/predict.py`)

The model is `t = t0 (4.2) + [route 1.5] + processed_bytes / BW(chunks) + [0.97 + t0 for a 2nd launch] + barrier 1.5 + fin 2`.
- BW(chunks) = 3.36 / (1 + 0.113·chunks_per_pass), i.e. lab D's dequant-only rate with D's per-chunk mma cost added.
  It gives 3.02 TB/s at 1 chunk, 2.87 at bs 128 and 2.72 at bs 256.
- Processed bytes = passes × 28.31 MB per expert. HBM bytes = touched × 28.31 MB.
- Incumbent = r1_glue_v2 judge job 96a32200948f4dc6.

| shape | NC | touched / passes | HBM MB | pred fuse=1 µs | pred fuse=2 µs | ×0.5 haircut (fuse=1) | incumbent µs | SGLang µs | pred vs incumbent |
|---|---|---|---|---|---|---|---|---|---|
| decode 1 (bs 1) | 1 | 2.6 / 2.6 | 74 | **38.7** | 35.1 | 77.5 | 54.0 | 77.5 | 1.39× |
| verify 1 (bs 8) | 1 | 8.2 / 8.2 | 231 | **90.8** | 87.1 | 181.5 | 111.6 | 143.1 | 1.23× |
| decode 16 | 1 | 11.9 / 12.3 | 336 | **129.7** | 126.0 | 259.4 | 156.3 | 188.1 | 1.21× |
| decode 32 | 2 | 15.5 / 15.8 | 440 | **163.4** | 159.8 | 326.9 | 213.9 | 210.6 | 1.31× |
| verify 16 (bs 128) | 4 | 22.4 / 24.1 | 635 | **251.8** | 248.1 | 503.5 | 344.0 | 469.9 | 1.37× |
| verify 32 (bs 256) | 4 | 24.7 / 29.7 | 700 | **323.7** | 320.0 | 647.3 | 475.6 | 614.3 | 1.47× |

Geomean vs SGLang:
- **1.66× (fuse=1)** and 1.72× (fuse=2), against 1.252× for the incumbent today. Routed min(v2, incumbent) is 1.66×.
- With the ×0.5 haircut, v2 loses every shape and the routed file stays at the incumbent's **1.252×**. The route table
  and autotune exist for exactly this case: the file cannot lose a shape.

**Honest gap to #1 (2.326×).** At the measured compute-attached rate (~3 TB/s), the decomposition is not the limit;
bandwidth per SM with dequant attached is.
- The same model gives 2.01× at 3.5 TB/s, **2.26× at 4.0** and 2.45× at 4.4 TB/s per processed byte.
- #1 implies the dequant + MMA consumer streams at ≈4 TB/s.
- The lab probes that say where the remaining 25% is:
  - `pr1` (this kernel's structure, stream only) and `pr2` (+ dequant, no MMA) give the per-shape w0/w1/w2 curve of
    *this* kernel.
  - The cheaper deq4 (vs Marlin's dequant in lab D) should show up as `pr2` above D's 3.36.
  - L3/L6 test whether prefetch (PF) closes the issue→consume gap.
  - L4 is the occupancy point: 3 CTAs/SM instead of 4.

## 5. Per-shape route table (never lose a shape)

- `BUCKETS` route by M range: [1,4], [5,12], [13,24], [25,48], [49,192], [193,320], each with its scored
  representative. Nearby shapes route like their representative.
- `ROUTE` (static) = incumbent everywhere until the lab reports (R2). `V2_PLAN` holds the v2 (variant, fuse, ks_target,
  asg) per bucket.
- `AUTOTUNE` (prepare, untimed):
  - It times the incumbent vs v2's plan per bucket representative on synthetic inputs from the task's distribution
    family (own generator; reference.py is not imported).
  - Each timing is the median of 15 CUDA-graph replays, each after a 256 MiB L2 flush.
  - A bucket switches to v2 only if v2 wins by ≥3%.
  - This is a measurement on synthetic data, not on the judge's inputs, so nothing derived from `data` is cached.
- The incumbent path is r1_glue_v3's prepare/kernel_fn copied verbatim. Its sglang imports are guarded so the CPU
  emulator can import the repack.
- After the lab: hard-code `ROUTE`/`V2_PLAN` from `SUM BEST` and keep AUTOTUNE as a guard.

## 6. CUDA-graph safety and hygiene

- No host syncs, allocations or value-dependent host branches in kernel_fn. The only per-call allocation is `out`
  (torch.empty, shape-only).
- All scratch is persistent state sized for M ≤ 320:
  - h: 84 MB
  - inter: 21 MB
  - y: 42 MB
  - meta: 11 KB
  - bar: 8 B
- KS, passes, units and atomics-vs-stores are decided on the device from meta.
- Bounded waits only: the grid barrier sets an error flag on timeout, and the self-test fails on a set flag.
  - Autotune and the lab also read the flag **after the graph replays**, and compare the last replay's output with
    the FP32 reference, because a co-residency miss would otherwise look like a slow but plausible µs.
  - Autotune never moves a bucket to v2 if either check fails.
  - `fuse=0` (3 launches, no grid barrier) is the fallback plan if the flag ever appears.
- The self-test covers 6 shapes (bs 1/5/33/100/128-skewed/257) × fuse {plan, 0, 2} against an FP32 reference. Any
  failure disables v2 (incumbent everywhere).
- No os/sys/environ/subprocess. Imports: torch, hashlib, time, typing, plus the incumbent's sglang/triton imports.
- Static smem is ≤ 43.6 KB per CTA (no >48 KB opt-in), 4 CTAs/SM = 172 KB.
- `cudaFuncSetAttribute(PreferredSharedMemoryCarveout, 100)` and the occupancy query run in prepare, before any
  capture.

## 7. Compile-risk spots (first nvcc contact)

1. Register caps. `__launch_bounds__(256, 4)` forces ≤ 64 regs. NC=2 at MINB 4 is the likeliest to spill (~66 est.).
   `-Xptxas=-warn-spills` warns, and `attrs` prints regs/local bytes/occupancy per variant. Spills cost speed only.
2. `__match_any_sync` (sm_70+), `__nanosleep`, `__ldcg` on `const int*`/`uint4*`/`float4*`, and `int8_t`/`int16_t`
   shared arrays (all standard, but first use in this repo).
3. Inline PTX is limited to strings already GPU-validated in k2a v0/v1 and stream_lab:
   - `ld.global.nc.L1::no_allocate.v4.s32` and `.s32`
   - `fma.rn.bf16x2`
   - `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
4. The `Route` struct with `long long cb[33]` in shared memory. The `float (*acc)[4]` parameter to force-inlined
   helpers must stay in registers after unrolling (check `l` = 0 in attrs).
5. load_inline binding of a 26-argument `run(...)` with int64 scalars (same pattern as v1's 9-argument gemm).
6. One extension holds all variants: the submission has 3, the lab 8 plus the L2 probe. Each instance contains all
   four phases. The estimate is ~10–15 s per instance (v1: 6 single-phase instances in 33 s), so 30–45 s for the
   submission and 80–120 s for the lab. The lab has a 240 s deadline guard, and a build failure prints the nvcc tail.

## 8. What the lab sweep decides (`k2a_lab_v2.py`, read `===== K2A LAB V2 SUMMARY`)

| line | decides |
|---|---|
| `SUM L0..L7 us` (8 template variants × 6 shapes + skewed bs128) | CTAs/SM (L0 vs L4), prefetch distance (L0 vs L3, L2 vs L6), token chunking NC (L0/L1/L2), token-group warps TW (L1 vs L5, L2 vs L7) |
| `SUM knobs bs<k>` on the best variant | fuse 0/1/2 (launch count vs grid barriers), ks_target 1/2/4 (split-K), asg round-robin vs cost-balanced, `atom` (forced atomics at KS = 1: cost of atomic flush at high route counts), `pr1`/`pr2` (stream-only / no-MMA ceilings of this kernel) |
| `SUM BEST` | best per shape, × vs the in-file incumbent row, effective TB/s on the touched bytes; `V2_PLAN`/`ROUTE` are filled from this |
| `SUM routed geomean gain` | what the routed file would gain over the incumbent |
| `SUM L2->SM` | L2 read bandwidth (32 MiB resident), the missing constant behind the X restaging cost |
| `SUM dedup sk128` | experts [128, 118, 28]: effective TB/s with multi-pass (L2, 32/pass) vs TW (L7, 64/pass); multi-pass dedup works if TB/s ≈ the normal-shape rate |
| `SUM floor` | graph-replay floor (raw µs include it; the judge subtracts ~4 µs) |

Decision rules:
- If `SUM BEST` beats `SUM REF inc` by ≥ 3% on a shape, route it to v2 with that plan.
- If `pr2` ≫ base (MMA/issue-bound), try RW = 16 or wgmma for verify.
- If `pr1` < 4 TB/s, the load pattern itself is short (more U, or cp.async staging of weights).
- If `atom` ≈ base, KS can be raised freely.

## 9. Emulation (`k2a_v2_emulate.py`, CPU torch 2.5.1, 133 s)

Shared vs independent:
- The repack is imported from k2a_v2.py.
- Every device formula is re-written from the CUDA text.
- Fragment maps come from CuTe's SM80_16x8x16 layouts, not from the kernel's (g, t) formulas.
- inter and y start as NaN, and stale smem rows (≥ ntok) are NaN.
- h must be zero at entry and zero again after FIN.
- Unit coverage is checked for asg 0 and 1 at every case: every (expert, pass, rg, ks) exactly once, all tiles, every
  token of a pass exactly once across warps/chunks.
- `check_index_chain` walks gemm_phase's per-warp loop literally (no numerics):
  - kb0 = (ks·KR + sl·KSWe) >> 6, groups of U, wa[i] / sa[i>>1], parity i&1, smem column (gi·U+i)·64.
  - PF 0 and the PF 1 register swap.
  - U ∈ {2,4}, KSW 128–1024, KS 1–16: 4480 item visits.
  - It asserts that weight item, scale word, parity and B column agree, and that every k64 item is consumed exactly
    once.
  - The numeric path indexes items by kb directly, so this walk is what covers the loop arithmetic.

```
primitive unit tests: OK (codes x positions, 252 scales x 4 selectors, 4032 exact products, fragment maps,
                          kernel (g,t) assumptions, repack -> A operand round trip)
index chain walk: OK (4480 item visits over U in {2,4}, KSW 128..1024, KS 1..16, PF 0/1)
bs=  1 normal  NC1 TW1 G528 kst2 asg0      PASS 3.762e-03 (row 3.762e-03) R=2   NV=2  KS1=16 KS2=16 atom1/2=1/1
bs=  4 normal  NC1 TW1 G528 kst2 asg1      PASS 3.770e-03 (row 3.944e-03) R=6   NV=4  KS1=8  KS2=16
bs=  8 normal  NC1 TW1 G528 kst2 asg0      PASS 3.769e-03 (row 3.860e-03) R=13  NV=5  KS1=4  KS2=8
bs= 16 normal  NC1 TW1 G528 kst1 asg0      PASS 3.750e-03 (row 3.972e-03) R=25  NV=12 KS1=1  KS2=2  (SwiGLU-store path)
bs= 33 normal  NC2 TW1 G528 kst2 asg0      PASS 3.734e-03 (row 3.852e-03) R=48  NV=15 KS1=2  KS2=4
bs= 33 normal  NC1 TW1 G528 kst1 forced-atomics PASS 3.734e-03          R=48  NV=15 KS1=1  KS2=2  atom1/2=1/1
bs=128 skewed  NC4 TW1 G396 kst2 asg0      PASS 3.744e-03 (row 3.960e-03) R=274 NV=9  [128,118,28] (4+4+1 passes)
bs=128 skewed  NC2 TW2 G528 kst2 asg1      PASS 3.745e-03 (row 3.961e-03) R=274 NV=9  KS1=1 KS2=2
bs=257 normal  NC4 TW1 G396 kst2 asg0      PASS 3.749e-03 (row 3.948e-03) R=469 NV=33 KS1=1  KS2=1  (all-store path)
ALL PASS
```

Negative controls (`K2A2_MUTATE=…`) must each FAIL, and do:

| mutation | what it breaks | result |
|---|---|---|
| `par` | wrong scale byte selector | 6.5e-1 |
| `bword` | B-operand word order | 1.4 |
| `flushrow` | epilogue row off by one | 1.4 |
| `rank` | reversed in-chunk rank | assertion "rank != stable sort" |
| `leader` | missing leader base update | assertion "two pairs on one compact row" |

Not covered by the emulator (hardware-only, caught by the prepare self-test):
- BF16 subnormal handling in `fma.rn.bf16x2`, the same as v0/v1, which passed on the GPU.
- Memory-ordering across the grid barrier (cooperative-groups pattern; `__ldcg` for in-kernel data).
- The PF=1 path's numerics: its index/swap logic is walked, but the numeric runs use PF=0 configs.
- Register spills and timing.

# P1: lossless 12-bit BF16 weights + fused decode/skinny-GEMM for the layer projections at T <= 32

Status:
- Round 1 ran on the H200 (`box_out/p1_lab.log`). **The format passes and the kernel fails.** Unpack is bit-exact on
  every projection and on both stress matrices; the one-hot probes have 0 mismatches. But the gate failed 0/16: the
  kernel streamed at 0.7–2.2 TB/s.
- **Round 2 (§R below) replaces the kernel.** The format, packer and unpack are unchanged. It is laptop-only again:
  never compiled, never submitted.
- Lab: `kernels/muse_bf16_proj/p1_lab.py`. CPU emulation of the format: `scratchpad/p1/emulate.py`, standard library only.

## R. Round 2: a pure streaming kernel for T ≤ 8 (plan, written before the code)

### R.1 What round 1 measured, and why it was slow

| shape | today net µs | round-1 packed net µs | packed TB/s | best geometry |
|---|---|---|---|---|
| qkv T=1 | 12.6 (R1B, 3.0 TB/s BF16) | 18.3 | 1.55 | (8 warps, 1 tile, **S=8**, D=3): 144 CTAs |
| pooler T=1 | 18.9 (R1B) | 23.7 | 2.16 | (4, 1, **8**, 2) |
| o T=1 | 7.1 (cuBLAS) | 8.6 | 1.46 | (4, 1, 4, 2) |
| unpooler slice T=1 | 5.0 (R1B) | 9.1 | 0.69 | (4, 1, **8**, 3) |

"Net" means event time minus the measured 5.25 µs graph floor. "Compression alone" (the same kernel on packed vs on BF16)
was only 1.02–1.10×, and the BF16 version of the kernel was itself slower than today. So the kernel was not
decode-bound. It was latency-bound, for four reasons:

1. **Too few bytes in flight.** Every sweep chose the largest split (more CTAs). qkv ran 144 CTAs × 8 warps, about 9
   warps per SM. Each warp held at most D = 3 super-blocks (4.5 KB) of a 16-SB K slice, so it made about 5
   dependent DRAM round trips. In flight per SM: about 36 KB peak, much lower on average. The consumer probe
   (`box_out/consumer_probe.log`) shows what works: decode + 4 mma per 16 B reaches **3.92 TB/s** with 32 warps per SM,
   4 independent 16-B loads per lane and 53 registers.
2. **Registers.** Round 1 used 86–174 registers. That capped occupancy at 8–16 warps per SM.
3. **Per-CTA fixed work.** Each CTA staged x in shared memory (8 token rows even at T = 1) and synced, and ran two
   cluster barriers plus a DSMEM reduce.
4. **Per-CTA work too small for that fixed cost.** 16 SBs per warp.

PDL changed nothing (0.99–1.02×), because a torch copy kernel as producer gives no early trigger. That lever is parked.

### R.2 New kernel `p1s_kernel<D, RAW>` (T ≤ 8; one launch; no smem staging of x, no cluster, no wgmma/TMA)

**Work unit.**
- CTA = one 16-row tile × a K slice (grid = nrt × S). Its W warps (W ∈ {4, 8, 16}) split the slice. Each warp streams
  its own contiguous run of super-blocks.
- **S = 1 wherever there are at least 144 row tiles** (qkv, o, pooler). Then nothing crosses CTAs: the CTA's W partials
  meet in 4·W·128 B of shared memory, behind one `__syncthreads`, and are stored as BF16 straight away.
- S ∈ {2, 4, 8} only for the 64-tile unpooler slice. There each CTA plain-stores its 16×T FP32 partial into its own
  slot `ws[s][tok][N]`. The last CTA of the row tile (one counter `atomicAdd`, no spinning) sums the slots in rank
  order q = 0..S−1, stores BF16 and resets the counter. It is bit-reproducible and there is nothing to re-zero. This is
  the R1B fixup pattern; R9 holds, since nothing waits.
- **Round-2b fix (pod run 2).** The first version added the partials with FP32 `atomicAdd`. The lab's repeat check failed
  at M ≥ 2 for split 4/8, on packed and on raw.
  - S = 2 is order-free (0 + a + b = 0 + b + a), but at S ≥ 3 the arrival order changes the FP32 sum. emulate.py §H
    measures 68% of outputs at S = 4 and 94% at S = 8.
  - That crosses a BF16 rounding boundary about 2e-4 of the time at S = 8: about 3.7 flips per qkv M=8 call pair, and
    0.5 at M=1. That is why only M ≥ 2 failed.
  - The S = 1 path never had shared-memory reuse or a race.
  - The atomic version survives as `red_mode = 1`, the negative control of the lab's determinism stress: 8 repeats at
    the largest split, M = 8. Rank-ordered must give 1 distinct result; atomics are expected to give more than 1.

**Streaming.**
- Per super-block, a lane issues 3 × `ld.global.nc.L1::no_allocate.v4` (48 B of packed weights). It also issues the
  super-block's 8 B-fragment words of x: 8 × 4-B `ld.global.nc`, L1-cached, predicated on token g < T. The x loads
  replace shared-memory staging: there is no CTA barrier before the stream, and the CTAs on an SM share x through L1
  (16 KB at T = 1, 128 KB at T = 8).
- **D stages rolling.** Stage d is refilled with super-block s + D right after s is consumed, so each warp always has D
  super-blocks requested.
- The first D stages and the escape list are issued before `griddepcontrol.wait` (no-op without PDL).
- Consume: per k-tile, `dec_kt` (4 PRMT + 3 SHF + 4 LOP3 → 4 bf16x2 A registers), then 1 × mma.m16n8k16 against x as B
  (T ≤ 8 tokens = one n8 group), with FP32 accumulation in 4 registers.

**Escapes.**
- The format is the same; the patch moves to **per-k-tile granularity**. The warp's in-range entries (≤ 32, one per
  lane) are keyed by (local SB · 4 + k-tile). There is one warp-uniform compare per k-tile.
- On a hit, the owning lane rewrites one bf16 half of one of the k-tile's 4 A registers. That removes round 1's
  16-register slow path.
- Rows whose 256-row group is RAW stream 4 × v4 per super-block with no decode: the `RAW = true` instantiation,
  selected per CTA.

**Registers (target ≤ 64, so that 2 × 512-thread CTAs or 4 × 256-thread CTAs fit per SM, i.e. 32 warps).**
- D = 1: 12 weights + 8 x + 4 acc + 4 A + 5 escape + about 10 addressing ≈ 45. It gets `__launch_bounds__(512, 2)`,
  a hard cap of 64.
- D = 2: about 65–70 registers, so `__launch_bounds__(512, 1)` (cap 128, no spills). The sweep keeps the grid resident:
  for qkv, W = 8 and S = 2 give 288 CTAs at 3 × 256 threads per SM.
- The lab prints registers, local bytes and CTAs/SM per instantiation via `p1s_info`.
- **6 instantiations**: D ∈ {1, 2} × MODE ∈ {packed, RAW (the BF16 control), mixed (CTA-uniform branch for matrices
  with RAW groups)}. Compile time is a fraction of round 1's 51 s, which keeps the layer's prepare budget safe; the layer
  needs only 1–2 of them.

**Bytes in flight per SM.**

| shape | grid | warps total | warps/SM | in flight/SM (D=2) | serial rounds per warp |
|---|---|---|---|---|---|
| qkv | 144 × 16 warps | 2304 | about 17.5 (16 or 32) | about 54 KB | 4 (8 SBs, D=2) |
| pooler | 256 × 16 | 4096 | 31 | about 95 KB | 4 |
| o (16 SBs per tile) | 512 × 8, 2 SBs per warp | 4096 | 31 | about 95 KB | 1: all of o is requested at once |
| unpooler slice | 64 × S=4 × 8 warps, 2 SBs per warp | 2048 | 15.5 | about 48 KB | 1 |

All of these exceed lessons §5's 32 KiB/SM. The lab sweeps W, S and D per shape and prints achieved TB/s and bytes in
flight per SM, from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and the registers per thread.

**ALU per super-block per warp.** 44 INT + 4 HMMA + 3 + 8 loads + 4 escape compares + about 6 loop ≈ 69 warp-instructions.
At 4.4 TB/s of packed bytes that is 0.86 of the 4 warp-instructions/clk/SM (21%), and the INT pipe runs at 27%.
Unchanged from round 1: the redesign targets latency, not ALU.

**T = 16/32 stay on cuBLAS in round 2.** Per-warp B fragments from global would re-read x once per 16-row tile: 1.3×
(T = 16) or 2.7× (T = 32) the weight bytes in L2 traffic. A skinny GEMM that shares x across row tiles needs smem
chunks plus barriers, or split-K atomics of T × N × S. That is a separate round-3 design (§R.5).

### R.3 Predicted (net µs = event − 5.25 µs floor; today = round-1 log, same process)

Model: net = 1.0 µs ramp + MB / BW + epilogue. BW = 3.9 TB/s packed (consumer probe w2v) or 4.2 TB/s for BF16 (probe w0:
4.33). Epilogue = 0.2 µs (S = 1) or 0.8 µs (atomics + fixup). Add 0.2 µs at T = 8 for x. Haircut = today − 0.5 × gain
(R7).

| proj | T | today | p1s on BF16 | **p1s packed** | × vs today | haircut | haircut × | packed TB/s |
|---|---|---|---|---|---|---|---|---|
| qkv | 1 | 12.6 | 10.2 | **8.5** | 1.49 | 10.5 | 1.20 | 3.35 |
| qkv | 8 | 16.3 | 10.4 | **8.7** | 1.88 | 12.5 | 1.31 | 3.27 |
| o | 1 | 7.1 | 5.2 | **4.4** | 1.61 | 5.8 | 1.23 | 2.84 |
| o | 8 | 7.2 | 5.4 | **4.6** | 1.55 | 5.9 | 1.22 | 2.72 |
| pooler (alone) | 1 | 18.9 | 17.2 | **14.1** | 1.34 | 16.5 | 1.15 | 3.57 |
| pooler (alone) | 8 | 25.3 | 17.4 | **14.3** | 1.77 | 19.8 | 1.28 | 3.52 |
| unpooler slice | 1 | 5.0 | 3.8 | **3.4** | 1.47 | 4.2 | 1.19 | 1.84 |
| unpooler slice | 8 | 7.1 | 4.0 | **3.6** | 1.97 | 5.4 | 1.33 | 1.74 |

- Coordinator target: qkv ≤ 7.5 µs net (≥ 3.8 TB/s including ramp). The model gives 8.5 central, so reaching 7.5 needs
  a ramp of about 0.3 µs or 4.3 TB/s steady state.
- **Layer** (lb10 calibrated; only decode 1 and verify 1 change; the pooler's in-layer gain is reduced by the
  side-stream FP32 router's 8.4 MB): **+4.6% central, +2.2% haircut**.
- About two thirds of the gain is structure. The control (p1s on BF16) already beats R1B and cuBLAS. Compression adds
  about 1.15–1.2× on top.

### R.4 Lab changes, gate and kill
- Timing is now T ∈ {1, 8} × {qkv, o, pooler (= cat rows 0–4095), unpooler slice into zbuf}: 8 shapes. Correctness also
  covers T ∈ {2, 5}. The host refuses T > 8.
- Variants per shape: today, p1s on BF16 (all tiles RAW) and p1s packed, each swept over (W, S, D) then re-timed with 41
  paired reps.
- The summary adds the columns **achieved TB/s** (packed MB / net µs) and **bytes in flight per SM** (occupancy × warps ×
  D × bytes per SB, with the registers per thread printed).
- Bit-exact unpack, the negative controls, the one-hot probes (every escape column, through p1s at T = 8 and T = 1) and
  the random-x checks against FP64 (repeat calls bit-identical) are kept.
- **Gate:** bit-exact unpack, plus packed ≥ 1.15× net vs today on all 8 T ≤ 8 shapes, in the same process.
- **Kill:** packed below 3.0 TB/s net on qkv and pooler at T = 1 after this round, **or** "compression alone" below
  1.10× there.
  - In the second case, keep p1s-BF16 as an uncompressed streaming GEMV if it beats today.
  - In the first, the streaming premise itself failed; check the probe numbers before any round 3.

**Pod commands (round 2).** Compile time is small and the timing is about 2–3 min.

```
tools/box/box.sh sync
tools/box/box.sh run   'python3 tools/box/sass_gate.py kernels/muse_bf16_proj/p1_lab.py --json box_out/p1_gate2.json 2>&1 | tee box_out/p1_gate2.log'
tools/box/box.sh timed 'python3 kernels/muse_bf16_proj/p1_lab.py 2>&1 | tee box_out/p1_lab2.log'
tools/box/box.sh pull
```

**What to read in the log.**
- The `p1s_kernel<D, MODE>` lines: registers, local bytes, CTAs/SM. Expect D = 1 at ≤ 64 registers with 0 local bytes,
  and D = 2 at ≤ 80.
- The PACK + VERIFY block: everything PASS, with "entries key-sorted True".
- The GEMV CHECKS block: one-hot probes with 0 mismatches. They cover mode 0 (qkv, o, unpooler), mode 1 (stress) and
  mode 2 (the cat matrix's RAW lo rows, stress_mild), each through split 1 and split > 1.
- The SUMMARY: the achieved TB/s and KB/SM-in-flight columns explain any miss.
  - High in-flight with low TB/s means the decode, mma or issue path, so compare against p1s on BF16.
  - Low in-flight means a different geometry (W, S) or D = 2.

### R.6 Round 3 (kernel in `p1_module.py`; the lab imports it)

**Round 2b measured** (net × vs today):

| shape | T=1 | T=8 |
|---|---|---|
| qkv | 1.066 | 1.237 |
| o | 1.108 | 1.000 |
| pooler | 1.143 | 1.481 |
| unpooler slice | 0.856 | 1.086 |

Achieved only 1.0–3.1 TB/s, with waves 0.24–0.97 and 12–52 KB/SM in flight; every sweep picked S=1 / 16 warps. The cause
is register-limited residency: at about 70 registers, any candidate with more CTAs than 132 × (CTAs/SM) ran a second wave.

**Changes:**
1. **Single-wave sweep.** W ∈ {2, 4, 6, 8, 12, 16}, S ∈ 1..8, D ∈ 1..3. A candidate is kept only if every CTA is resident
   at once (from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) with 0 local bytes. Candidates are ranked by weight
   bytes in flight per SM; the round-2b incumbent is always kept.
2. **Ring refill before the HMMA batch.** Decode the SB → copy its x words → reload the stage with SB s + D → escape
   patch → 4 HMMA. D SBs stay in flight continuously instead of D − 1 while the HMMA chain runs.
3. **Two accumulators** (k-tiles 0, 2 / 1, 3), so a chain is 2 dependent HMMAs, not 4.
4. **k-permuted lane map**, the only layout change. Lane t owns the 16 contiguous k `[64 sb + 16 t, +16)`, with
   `k = 64 sb + 16 t + 4 kt + 2 (i >> 1) + h`. Its x B-fragment is then 2 × `ld.global.nc.v4`, replacing 8 × 4-B loads.
   The mma contraction is unchanged because the same slot→k map is used on A and B (emulate.py check A).
5. **Stream probe** (MODE 3): the same loads at the same grid with an XOR consume, timed next to every packed winner.
   It gives the ceiling per shape. The scout note asks for ≥ 4.2 TB/s on the big shapes.
6. **Cheaper fixup.** Barrier, then one thread fences and increments the counter (CUTLASS-semaphore order). The
   rank-ordered slot sum is unchanged.
7. **10 instantiations:** MODE 0/1 at D 1–3, MODE 2 at D 1–2, MODE 3 at D 2–3.

**Predicted net µs** (central; the floor is round 2b's measured number). The table header gives the coordinator's target.

| shape | 2b measured | round 3 central | × vs today | probe ceiling |
|---|---|---|---|---|
| qkv T1 | 11.8 | 9.0 | 1.40 | 7.5 |
| qkv T8 | 13.2 | 9.4 | 1.74 | – |
| o T1 | 6.4 | 5.0 | 1.43 | 4.3 |
| o T8 | 7.2 | 5.3 | 1.36 | – |
| pooler T1 | 16.5 | 14.0 | 1.35 | 12.5 |
| pooler T8 | 17.1 | 14.5 | 1.75 | – |
| slice T1 (needs S 4–8) | 5.9 | 3.8 | 1.32 | 2.8 |
| slice T8 | 6.6 | 4.0 | 1.78 | – |

qkv T1 at 9.0 µs is 3.15 TB/s including a 1 µs ramp, so it reaches the 3.5 TB/s target only if the ramp falls to about
0.5 µs. Pooler T1 at 14.0 µs is 3.6 TB/s.

**Layer integration.** Use `p1_prepare` / `p1_route` / `p1_gemv` from `p1_module.py`. Geometry is hard-coded per
(projection, T bucket) from the lab's paste-ready `P1_GEOMETRY` / `P1_ROUTE`. Routes need ≥ 1.03× net, otherwise
today's path is kept.

### R.7 Round 3b: back to the 2b kernel, now inside `p1_module.py`

**Round 3 on the box was worse:**
- 111 registers, and `<1,2>` / `<1,0>` spilled 64 / 44 B.
- About 15 warps per SM; the stream probe reached only 2.0–3.1 TB/s at that grid.
- Net × fell on every shape (qkv 0.914 / 1.185 …).

**Changes:**
- The module now carries the **exact round-2b kernel**: per-k-tile decode → mma, refill after use, 8 × 4-B x loads, the
  2b lane map, and 56 registers in 2b.
- It keeps 2b's deterministic per-slot split-K and adds only an XOR stream probe (MODE 3).
- The k-permuted v4 x loads are **not** kept. Their register effect was never measured apart from round 3's bundle,
  and they would change the measured format.
- The sweep is gone. The geometry is hard-coded from `box_out/p1_lab2b.log`:

| projection | T=1 (W, S, D) | T=8 (W, S, D) |
|---|---|---|
| qkv | (16, 1, 1) | (4, 8, 1) |
| o | (8, 1, 1) | (8, 1, 1) |
| pooler | (16, 1, 1) | (16, 1, 1) |
| unpooler | (16, 1, 1) | (16, 1, 1) |

- `P1_ROUTE` = qkv T1/T8, o T1, pooler T1/T8 and unpooler T8 on P1; o T8 and unpooler T1 keep today's path.
- **Instantiations.** The layer builds only `p1s_kernel<1,0>` (`p1_build(layer_only=True)`, i.e.
  `-DP1_LAYER_ONLY=1`). The lab also builds `<1,1>` (BF16 control), `<1,2>` (mixed) and `<1,3>` (probe).

### R.5 Round 3 sketch (T = 16/32, not started)
The CTA would cover 4–8 row tiles over a K chunk, with x chunks of 512 k double-buffered in smem (one barrier per chunk),
B fragments from ldmatrix shared by all the CTA's warps, and split-K S = 2–4 via `red.global.add.v2.f32` plus last-CTA
fixup. Atomic volume T × N × S (≤ 0.6 M at T = 32), and an x reuse ratio of 1.33·T / rows per CTA.

## 0. Decision summary (round 1; format sections 2–3 still apply, the kernel sections 4–7 are superseded by §R)

| item | choice / number |
|---|---|
| code | **P12**, fixed 12 bits per weight: an 8-bit byte plane and a 4-bit code plane, laid out in mma.m16n8k16 A-fragment order. Measured size: 0.7503–0.7509 of BF16, metadata included. |
| window | 16 exponents per 16-row tile. Each tile picks the base that holds the most of its weights. |
| escape rate, judge generator (`randn/sqrt(K)`) | **6.5e-5** (K = 8192), **9.8e-5** (K = 1024 and 4096) per weight. Monte Carlo and analytic agree. That is about 9 escapes per 16×8192 tile. Expected RAW tiles: about 1e-7. |
| 11-bit variant (8-exponent window) | escapes **1.25–1.35%** of weights, about 1770 per tile. Rejected. |
| escape mechanism | Escaped weights are coded 0. Each 16-row tile has a sorted patch list of up to 32 entries (8 B each), applied in registers before the mma: the tensor core sees the exact weight. A 256-row group whose tile exceeds 32 escapes, or has an escape that cannot be scaled exactly, is stored RAW (plain BF16). |
| exactness | Bit-exact unpack for any input: zeros, ±0, subnormals, ±inf, NaN, outliers, heavy tails. Emulation: 0 failures; all 3 negative controls fail as they must. |
| kernel | One launch per projection: register-direct 16-B loads, decode to bf16x2 A fragments, mma.sync m16n8k16 with x as B from shared memory (ldmatrix), FP32 accumulation. Split-K reduces through a thread-block cluster's DSMEM (no atomics, no second kernel, deterministic). Prefetch of D = 2–3 super-blocks; PDL-ready. |
| ALU at 4.4 TB/s packed | INT pipe 27%, issue 18–24%, HMMA (T = 32) 30–40%, ldmatrix smem 40% (20% with 2 row tiles per warp). Not decode-bound on paper. |
| T = 128 / 256 | **Stay on cuBLAS.** The skinny design is mma-issue and x-reuse limited there; a packed wgmma GEMM would be its own project. |
| predicted layer gain (lb10 model, calibrated to the judge) | **+5.1% central, +2.5% after the R7 ×0.5 haircut** (1.612 → 1.694 / 1.652). |
| gate risk | **o_proj is predicted to miss the 15% gate** (1.02–1.07×). It is ramp-bound: 3.05 µs of its 6.6 µs is fixed cost. See §6. |

## 1. Why

lb10 (`box_out/lb10_analysis.md`, column "ours" = k3_v09) gives the per-op exclusive µs below. k3_v12 runs the same projection
path, except that it writes the T ≤ 8 unpooler into a zero buffer.

| shape | qkv | o | pooler (+router) | unpooler | projections / layer |
|---|---|---|---|---|---|
| decode 1 (T=1) | 12.1 (R1B GEMV) | 6.7 (cuBLAS) | 18.2 (R1B + FP32 router on the side stream) | 5.8 (R1B, 1/8 slice) | 43% sliding / 40% full |
| verify 1 (T=8) | 14.0 (cuBLAS) | 6.8 | 22.9 (cuBLAS + side router) | 8.1 (cuBLAS slice) | 34% / 31% |
| decode 16 | 13.6 | 6.6 | 24.4 (cuBLAS "cat" [pooler; hi; lo], FP32 out) | 19.9 (cuBLAS, full) | 37% / 26% |
| decode 32 | 13.8 | 6.8 | 24.0 ("cat") | 20.1 | 31% / 17% |

These GEMVs and skinny GEMMs are weight-bandwidth bound. Their bytes are BF16 `randn/sqrt(K)`
(`problems/muse_decode_layer/reference.py` `dense`), so the exponent carries only 2.54 bits of entropy, measured. TASK.md
"Weights" allows "BF16 weights stay BF16, or smaller, losslessly", and "Dequantizing inside your kernels is fine".

**Why the competitor's attempt failed** (research_competitors.md §b.5, bra1ndump): "Lossless BF16 compression: 41 µs vs
19–20 µs uncompressed, abandoned." Our reading is that the decode cost more than the bytes saved. That suggests a
variable-length or table decode, or one in a consumer that was already issue-bound (their M1 path is FMA-GEMV style).
P12 is built around that failure:
- fixed width, so no bit-stream state and no tables;
- 2.75 integer ops per two weights;
- the decoded registers are the mma A fragments themselves, so there is no BF16→FP32 conversion and no FMA per weight;
- the tensor cores do the products at every T, including T = 1, where they sit 7/8 idle at no cost.

## 2. Code format P12

**Geometry.**
- Row tile: 16 rows.
- Super-block (SB): 16 rows × 64 k = 1024 weights = **1536 B = 3 × 512 B**. One 16-B load per lane per chunk, fully
  coalesced (a warp instruction reads 512 contiguous bytes).
- Lane L owns 32 weights per SB: the 16 bf16x2 A-fragment registers o = 0..15 of the four 16×16 k-tiles.
- With g = L>>2, t = L&3, kt = o>>2, i = o&3, half h:
  `row = g + 8(i&1)`, `k = 64·sb + 16·kt + 2t + 8(i>>1) + h`. This is the m16n8k16 A layout.

**Per tile.**
- base b = argmax over b ∈ [1, 239] of #weights with exponent e ∈ [b, b+15]. So e = 0 (zero, subnormal) and e = 255
  (inf, NaN) always escape.
- n = e − b ∈ [0, 15].

**Planes.**
- chunk 0/1 (byte plane): byte = `(n & 1) << 7 | mantissa7`, at byte `2(o&7)+h` of the lane's 16 B.
- chunk 2 (code plane): 4 u32 words cw[kt]. Bits n1, n2, n3, sign of (o, h) sit at
  `((8, 9, 10, 15) + 16h + 4i) mod 32`. The four rotations tile the 32 bits exactly; emulate.py check A.

**Decode to the scaled domain** (exponent 0x70 | n, C = 112):

```
a[o] = prmt(bw[o>>1], 0x38383838, o odd ? 0x4342 : 0x4140) | (rotr(cw[kt], 4i) & 0x87008700)
```

- PRMT puts the two stored bytes in bits 0–7 and 16–23 and the constant exponent bits e4..e7 = 0111 (0x38) in each
  high byte.
- One rotate (none for i = 0) plus one LOP3 merge the code bits e1 e2 e3 and the sign.
- Per 8 weights: **4 PRMT + 3 SHF + 4 LOP3** = 1.375 INT ops per weight, 0.92 per packed byte.

**Exact rescale.**
- The tile's GEMM result is multiplied by `scale[rt] = 2^(b − 112)`, one FMUL per output.
- Multiplying every weight by the same power of two commutes with FP32 rounding, and with the tensor core's aligned
  accumulation, unless an intermediate overflows or goes subnormal. So the result equals the GEMM on the original
  weights.
- Escapes are patched with W·2^(112−b). That value is checked in prepare() to be exactly representable; otherwise the
  group goes RAW.
- Choosing C as a multiple of 16 is what makes the exponent an OR rather than an add. It saves one op per two weights.

**Unpack.** `orig = (v & 0x807F) | ((b + (e'' & 15)) << 7)`. Escapes take the raw16 field of their entry. The result is
bit-exact by construction and verified on the GPU for every projection.

**Metadata.** Per tile: `toff` (int64), `scale` (f32), `base` (i32, unpack only), `fmt` (u8), `eoff` (i32). Per escape:
int2 `{sb | lane<<16 | o<<21 | h<<25, scaled16 | raw16<<16}`. Total size: **12.005–12.015 bits/weight**.

**Why not fewer bits.**
- 3-bit code: escapes 1.3%, so a patch path on 99.99% of SBs or RAW everywhere.
- Entropy coding (H(e) = 2.54 → about 10.5 bits/weight) brings back the competitor's failure mode.
- Sign and mantissa are incompressible (uniform).

## 3. Escape scheme

| case | handling | cost on the hot path |
|---|---|---|
| e outside [b, b+15] (tiny values, outliers), ±0, subnormal, ±inf, NaN (NaN stored as 0x7FC0 scaled, raw bits kept) | entry in the tile's patch list, ≤ 32 per tile (one per lane) | one compare per SB per row tile (warp-uniform). On a hit (~7% of SBs), the slow path decodes all 16 registers, the warp walks its entries with shfl, and the owning lane rewrites one bf16 half. |
| > 32 escapes in a tile, or an escape that is not exactly scalable (e.g. 3e38 in a small-base tile) | the whole **256-row group** is stored RAW: plain BF16 in fragment order, 2048 B/SB, scale 1. The format is CTA-uniform for every geometry with ≤ 256 rows per CTA, so the branch sits outside the loop. | none (a separate instantiation of the loop) |

Measured on the generator distribution (emulate.py §F, 2M samples per K; analytic in brackets):

| matrix | N × K | escapes/weight | per 16-row tile | E[RAW tiles] | 11-bit escapes |
|---|---|---|---|---|---|
| qkv_gate | 2304×8192 | 6.5e-5 [6.9e-5] | 9.0 | 9e-8 | 1.35% |
| o | 8192×1024 | 9.8e-5 [9.7e-5] | 1.6 | 0 | 1.23% |
| pooler | 4096×8192 | 6.5e-5 [6.9e-5] | 9.0 | 2e-7 | 1.35% |
| unpooler | 8192×4096 | 9.9e-5 [9.7e-5] | 6.4 | 5e-11 | 1.25% |
| router hi (bf16(w)) | 256×8192 | 8.0e-5 | 10.5 | 4e-7 | 1.40% |
| router lo (bf16(w − hi)) | 256×8192 | 5.5e-4 | 72 | **all 16 (RAW)** | 3.96% |

The router lo plane only rides in the T ≥ 16 "cat" GEMM. It stays BF16 (4.2 MB), which costs 1 MB of savings (~0.2 µs).

**Exactness evidence** (emulate.py, 0 failures, 27 s):
- B. Full-K row tiles of every shape. Bit-exact unpack, and the GEMM view (scaled × 2^(b−112)) is value-exact.
- C. Stress inputs:
  - heavy tails, 1% zeros and inf/NaN/subnormal specials: all RAW, exact;
  - the same data with an unlimited patch list: 413 patches plus 2 unscalable-RAW tiles, exact;
  - an all-specials tile: exact.
- D. A split-K GEMM view with S = 1/3/8 against float64: max relative error 0, and every escape applied exactly once.
- E. Negative controls, all detected:
  - skipping the patches: 9 mismatches;
  - a wrong rotation for o = 6: 6572 mismatches;
  - one flipped code bit: 1 mismatch;
  - the 11-bit window escapes 1.34%.

## 4. Kernel `p1_gemm_kernel<NT, ROWT, D, F32>` (round 1; replaced by `p1s_kernel`, §R.2)

**Mapping.**
- `out[t, n] = Σ_k x[t,k] W[n,k]` for t < M ≤ 32. NT = ⌈M/8⌉ ∈ {1, 2, 4} n8 token groups. M ≤ 8 uses NT = 1 and
  zero-pads the tokens.
- Warp = ROWT ∈ {1, 2} row tiles × the CTA's K slice. CTA = 4 or 8 warps = 64–256 rows.
- grid = (row groups, S), with **cluster (1, S, 1)**, S ∈ {1, 2, 4, 8}. Uneven K slices are allowed.

**Phases.**
1. **Prologue.** Issue the first D super-blocks: 3 × LDG.128 per row tile, or 4 for RAW. Load the warp's in-slice
   escape entries: one per lane, then ballot, ffs and clz.
2. `griddepcontrol.wait`. Everything before it reads prepare()-time data only, so under PDL the weight stream starts
   inside the predecessor's tail. Without PDL it is a no-op.
3. Stage x[0:8·NT, k-slice] in shared memory, zero rows ≥ M. Use a 16-B row pad for conflict-free ldmatrix.
   `__syncthreads`.
4. **Loop over SBs.** Unrolled by D; each stage is refilled right after its use. Fast path per k-tile:
   - B fragments: NT = 1 takes one ldmatrix.x4 per two k-tiles; NT ≥ 2 takes one ldmatrix.x4 per pair of token groups;
   - per row tile: `dec_kt` (4 PRMT, 3 SHF, 4 LOP3), then NT × mma.m16n8k16.bf16 with FP32 accumulation.
   If any row tile of the warp has its next escape in this SB, the slow path runs: decode 16, patch, mma.
5. **Epilogue.**
   - × scale[rt], fragments → smem [8NT][rows] FP32, `cluster.sync()`;
   - rank r sums its rows/S share over the S ranks in fixed order (DSMEM `map_shared_rank`) and stores BF16 (or FP32
     for "cat");
   - `cluster.sync()`. There are no global atomics, no workspace, no counters and no second kernel (lessons §6:
     a 2nd kernel costs +5.4 µs).
   - R9: the only waits are hardware barriers (`__syncthreads`, `barrier.cluster`, `griddepcontrol`). No spin loops,
     so no flag and no `__trap`.

**Bytes in flight.**
- 8 warps × 2 CTAs × ROWT=1 × ~2.5 SB × 1.5 KB ≈ **60 KB/SM** (NT ≤ 2). With 1 CTA/SM × 8 warps × ROWT=2, also ~60 KB.
- Both exceed the 32 KiB/SM that plain LDG.128 needs for 4.3–4.5 TB/s (lessons §5).
- Weights never touch shared memory.

**Registers (estimate).**
- NT=1/ROWT=1/D=3 ≈ 80, giving 2–3 CTAs/SM.
- NT=4/ROWT=2/D=3 ≈ 150, so `__launch_bounds__(256, 1)` is chosen by `Occ<>`. The compile gate reports the real counts
  and spills.

**Geometry is not hard-coded.** The lab sweeps (warps ∈ {4, 8}, ROWT ∈ {1, 2}, S ∈ {1, 2, 4, 8}, D ∈ {2, 3}) per
(projection, T), subject to smem ≤ 227 KB and 48–4096 CTAs, and prints the top 6.

**Call sites** (k3_v12 `kernel_fn`):

| projection | today | packed path |
|---|---|---|
| qkv | T=1 R1B, else cuBLAS | p1 for T ≤ 32 |
| o | cuBLAS | p1 for T ≤ 32 (only if it passes) |
| pooler, T ≤ 8 | R1B / cuBLAS, with the FP32 router (`f32`) kept on the side stream | p1 on the first 4096 rows of the packed **cat** matrix |
| pooler, T = 16/32 | cuBLAS cat | p1 on all 4608 cat rows, FP32 out into v8_tail as today |
| unpooler, T ≤ 8 | R1B / cuBLAS slice | p1 on the first 1024 rows, written into `v12_zbuf` with ldo = 8192 |
| unpooler, T = 16/32 | cuBLAS | p1 on the full matrix |

At T > 32 everything stays as today. The raw BF16 weights stay resident for those paths, which is allowed: the packed copy
is an extra lossless representation.

## 5. ALU budget (per byte, at the power-cap clock of 1.755 GHz, 132 SMs)

- 4.4 TB/s of packed bytes = **19.0 B/clk/SM** = 12.7 weights/clk/SM = 0.0124 SB/clk/SM.
- INT (PRMT/SHF/LOP3, 64 lanes/clk/SM, measured 63.2): 44 warp-ops per SB → **17.4 lane-ops/clk/SM = 27%**. The budget
  is 3.4 INT ops per packed byte; P12 uses 0.92.
- Issue: 59 warp-instructions per SB at NT=1 (44 INT + 3 LDG + 4 HMMA + 2 LDSM + ~6 loop/branch), 77 at NT=4 →
  0.73–0.95 of 4 warp-instructions/clk/SM = **18–24%**.
- HMMA at T = 32: 16 per SB → 0.20/clk/SM, against the measured mma.sync rate of 0.5–0.67 (lessons §7) → 30–40%.
- ldmatrix at NT=4 with ROWT=1: 4 KB per SB → 51 of 128 B/clk/SM = 40%. ROWT=2 halves it.

For comparison, the NVFP4 stream (lessons §6) hit 3.0–3.4 TB/s at 1 INT op per weight once it had too few warps. That
failure was occupancy, not ALU. Here the decode is 1.4 INT ops per weight at less than half that weight rate. **The risk
is latency hiding, not decode.** That is why the lab separates "the same kernel on BF16" from "packed".

## 6. Per-shape model (µs, nsys-exclusive kernel time as in lb10)

**Model.**
- BF16 formula = 3.05 + MB/4.74 (lessons §6). It includes the concurrent 8.4 MB FP32 router for the pooler at T ≤ 8.
- packed central = 3.05 + 0.30 (cluster epilogue, x stage) + MB_packed/(4.74 × 0.92 decode interference)
  + 0.1·T/8 (x stage).
- haircut = today − 0.5 × (today − central) (R7).
- "p1 on BF16" is the same kernel on RAW tiles: it isolates the structural gain.

| proj | T | today | BF16 formula | p1 on BF16 | packed ideal | **packed central** | **haircut** | central × | haircut × | compression alone × |
|---|---|---|---|---|---|---|---|---|---|---|
| qkv | 1 | 12.1 | 11.0 | 11.3 | 9.0 | **9.9** | 11.0 | 1.22 | 1.10 | 1.15 |
| qkv | 8 | 14.1 | 11.0 | 11.4 | 9.0 | **9.9** | 12.0 | 1.42 | 1.17 | 1.15 |
| qkv | 16 | 13.6 | 11.0 | 11.5 | 9.0 | **10.0** | 11.8 | 1.35 | 1.15 | 1.15 |
| qkv | 32 | 13.9 | 11.0 | 11.7 | 9.0 | **10.2** | 12.1 | 1.36 | 1.15 | 1.14 |
| o | 1 | 6.7 | 6.6 | 6.9 | 5.7 | **6.2** | 6.5 | 1.07 | 1.03 | 1.10 |
| o | 8 | 6.8 | 6.6 | 7.0 | 5.7 | **6.3** | 6.6 | 1.07 | 1.04 | 1.10 |
| o | 16 | 6.6 | 6.6 | 7.1 | 5.7 | **6.4** | 6.5 | 1.03 | 1.01 | 1.10 |
| o | 32 | 6.8 | 6.6 | 7.3 | 5.7 | **6.6** | 6.7 | 1.02 | 1.01 | 1.10 |
| pooler (+router) | 1 | 18.4 | 19.0 | 19.3 | 15.4 | **16.7** | 17.5 | 1.10 | 1.05 | 1.16 |
| pooler (+router) | 8 | 22.8 | 19.0 | 19.4 | 15.4 | **16.8** | 19.8 | 1.36 | 1.15 | 1.16 |
| pooler cat | 16 | 24.2 | 19.0 | 19.5 | 15.2 | **16.8** | 20.5 | 1.45 | 1.18 | 1.16 |
| pooler cat | 32 | 24.1 | 19.0 | 19.7 | 15.2 | **17.0** | 20.5 | 1.42 | 1.17 | 1.16 |
| unpooler slice | 1 | 5.8 | 4.8 | 5.1 | 4.4 | **4.8** | 5.3 | 1.21 | 1.09 | 1.07 |
| unpooler slice | 8 | 8.1 | 4.8 | 5.2 | 4.4 | **4.9** | 6.5 | 1.64 | 1.24 | 1.07 |
| unpooler | 16 | 19.9 | 17.2 | 17.7 | 13.7 | **15.1** | 17.5 | 1.32 | 1.14 | 1.17 |
| unpooler | 32 | 20.3 | 17.2 | 17.9 | 13.7 | **15.3** | 17.8 | 1.33 | 1.14 | 1.17 |

**Layer** (lb10 judge-calibrated groups, gains × cal, verify 16/32 unchanged): 1.6123 → **1.694 central (+5.1%)** /
**1.652 haircut (+2.5%)**. By projection (central): qkv +1.4%, pooler +2.1%, unpooler +1.3%, o +0.1%.

**What the table says.**
- About half the gain at T = 8–32 comes from the kernel structure beating cuBLAS: the "p1 on BF16" column, 1.12–1.54×.
- Compression adds a uniform **1.14–1.17×** on the large matrices.
- At T = 1, R1B is already at the formula, so compression must carry the whole gain: qkv 1.22, pooler 1.10 in-layer.
- o_proj is ramp-bound. −25% of 16.8 MB is at most −0.9 µs of 6.6, so it cannot reach 1.15 without PDL overlap of its
  3 µs fixed cost (§7 lever 1).
- The lab measures the pooler alone, without the side-stream router. Standalone T = 1 central is about 1.15–1.20.

## 7. Pod lab plan (one burst, ~10 min of GPU)

```
tools/box/box.sh sync
tools/box/box.sh run   'python3 tools/box/sass_gate.py kernels/muse_bf16_proj/p1_lab.py --json box_out/p1_gate.json 2>&1 | tee box_out/p1_gate.log'
tools/box/box.sh timed 'python3 kernels/muse_bf16_proj/p1_lab.py 2>&1 | tee box_out/p1_lab.log'
tools/box/box.sh pull
```

1. **Compile gate** (no GPU time). Expect PASS: 20 instantiations, no wgmma, 0 spills. Read the registers per
   instantiation. If the `minb = 2` variants spill, lower `Occ<>::regs`' threshold. If the source fails to compile, fix it
   from nvcc.log; the source has never seen nvcc.
2. **Lab, in one process.**
   - Build: about 2–4 min, plus about 1 min for R1B.
   - Weights from `reference.generate_weights(5353)`, falling back to the same recipe.
   - **PACK + VERIFY** on qkv, o, cat [pooler; hi; lo], unpooler, plus two stress matrices. Unpack mismatches must be 0;
     the negative control without patches must show > 0. The RAW-BF16 packs are checked too.
   - **GEMM CHECKS**:
     - a one-hot probe through every escape column and 64 random ones, at M = 32 and 8. Outputs must equal the weights
       bit for bit; tensor-core subnormal flushes are counted apart;
     - random x at M ∈ {1, 2, 5, 8, 9, 16, 24, 32} over every geometry, against FP64 (BF16 out < 4e-3, FP32 out < 2e-5),
       with repeat calls bit-identical;
     - M = 33 must be refused.
   - **TIMING** per (projection, T ∈ {1, 8, 16, 32}):
     - today's k3_v12 path (R1B GEMV at T = 1, cuBLAS via SGLang's dispatch, the cat FP32 GEMM at T ≥ 16, the slice into
       zbuf at T ≤ 8);
     - a sweep of packed geometries and of "p1 on BF16" geometries;
     - the best of each re-timed with 41 paired reps;
     - CUDA graph per call, 256 MB flush before every replay, alternating order, median of paired ratios;
     - "net" ratios subtract the measured one-kernel replay floor.
   - **PDL**: producer `x.copy_` → p1, with and without programmatic serialization, at T = 1 and 32.
   - **SUMMARY** table and the GATE line.
3. **Gate for the pod phase** (as briefed): bit-exact unpack on all weights, **and** packed ≥ 1.15× net vs today on
   pooler, unpooler, qkv and o_proj at T ≤ 32, in the same process.
   - Expected outcome: qkv, pooler, unpooler PASS on central numbers (borderline on the haircut); **o_proj FAILS**.
   - Recommended reading if o fails alone: keep o on cuBLAS. That costs 0.1% of layer and does not block integration.
     Or retry o with PDL from K1, its predecessor, which is our own kernel and can trigger early.
4. **Kill.** Decode-bound below 10% gain at T = 1 after one optimization round, i.e. "compression alone" (packed vs p1 on
   BF16, same process) < 1.10 on qkv and pooler at T = 1 after round 2 → drop P12.
   - If "p1 on BF16" itself beats today at T ≥ 8, keep that kernel as the uncompressed skinny GEMM: it is worth about
     half the predicted gain by itself.
   - Round-2 levers, in order:
     1. PDL with an early trigger in the producers we own (K1 → o, K2A → unpooler);
     2. D = 4 / ROWT = 2 at T = 1;
     3. x staged in halves at T = 32;
     4. `ld.global.nc.L1::no_allocate` / L2 256B prefetch hints;
     5. 2 row tiles per warp sharing B at NT = 4.

## 8. After the gate: integration (k3_v13, not started)

- Generator `tools/gen_k3_v13.py` (the gen_k3_v1x pattern): add P1_CUDA_SRC to the combined build, with **only the
  winning instantiations** (about 6). The prepare budget is 300 s, and K2A/K1 already compile there; record the measured
  p1 compile time first.
- prepare():
  - `pack_p12` on qkv, o, the cat matrix and the unpooler (GPU, under 1 s each);
  - bit-exact unpack check;
  - one-hot probe of the escape columns;
  - a random-x check per routed (projection, T bucket).
  Any failure → the incumbent path for that projection (print one `[p1]` line).
- kernel_fn: route by `_k2a_bucket_of(tokens)` ≤ 32 (shape-only branches; no syncs, no allocation by value).
- The judge's held-out weights are repacked by the same data-adaptive packer. Correctness never depends on the escape
  count: a heavy-tailed or sparse checkpoint degrades to RAW groups, at BF16 speed with the same kernel.
- Layer A/B with `tools/box/layer_ab2.py`, k3_v13 vs k3_v12, judge clone, paired. Ship bar as usual (+3% box).

## 9. Risks and open questions

1. **Never compiled.** No nvcc on the laptop and no stub compile (no clang either). Main suspects:
   - `cudaLaunchKernelEx` with cluster + PDL attributes inside graph capture;
   - `__grid_constant__` together with `cg::this_cluster()` in a non-cluster launch (S = 1);
   - template-dependent `__launch_bounds__`.
   The compile gate exists for this.
2. **Cluster scheduling.** 8-CTA clusters at 1 CTA/SM need 8 free SMs in one GPC. If occupancy drops, the sweep shows it.
   S ∈ {1, 2, 4} avoids the problem; the atomics fallback is not implemented.
3. **x staging at T = 32**: 32 × K-slice × 2 B per CTA from L2 before the first mma. Modelled at +0.4 µs; could be +1 µs.
   Lever: smaller slices or a two-phase stage.
4. **Pooler at T ≤ 8 in the layer**: the FP32 router (8.4 MB) shares bandwidth on the side stream. The lab measures the
   pooler alone, so only the k3_v13 A/B shows the joint effect. The router could be packed too (FP32 with a 4-bit
   exponent offset = 28 bits, −12.5% of 8.4 MB ≈ 0.2 µs). Not worth it now.
5. **Tensor-core subnormal handling.** If mma flushes bf16 subnormal inputs, a packed or RAW subnormal weight
   contributes 0, exactly as it would through cuBLAS on raw BF16. Exactness of the storage is unaffected. The lab counts
   these flushes.
6. **Rules reading** (conservative, A-list candidate): the packed copy is lossless and verified bit-exact in prepare();
   escape weights reach the tensor core as their exact value × 2^j; the output scale is an exact power of two. No
   precision change. Packing in prepare() is allowed ("Repack them in prepare()").
7. **PDL in the real layer** depends on the predecessor. SGLang's JIT boundaries do not trigger early, so the gain there
   is only launch-gap overlap. K1 and K2A are ours and can call `griddepcontrol.launch_dependents`.
8. **Prepare compile time** with 20 instantiations is probably 2–4 min. That is fine for the lab; the layer must instantiate
   only the winners.

# MoE v4 study: what structure takes in-layer MoE from ~1.8× to ~2.3×? (design study, 2026-10-25)

Scope: design only. No GPU time, no submissions, no code. Base file: k3_v18 (scored 1.7043; per-op budget
`box_out/lb18_analysis.md`). Every rate below is a measured constant (lessons §5–§8, A2 lab, k2a_v3x labs, K2b
v1–v3 stage B). New mechanisms get the R7 ×0.5 haircut on their *gain*. "Central" = the model with measured
constants and no haircut; "planning" = the haircut number. Box µs are raw graph-replay µs on the judge-clone box,
which reads ~4–5% faster than the judge.

## 0. Answer

1. **2.3× in-layer MoE is not reachable with measured constants. About 2.0× (planning) to 2.15–2.2× (central) is.**
   - At T = 8 / 16 / 32 stock is already within 2.3× of the 4.3 TB/s byte floor: stock/floor is 2.1 / 2.25 / 1.8.
     So those shapes can't reach 2.3× even at the floor.
   - A 2.3× weighted mean would need T = 128 / 256 within ~10% of their floor. That is above the measured
     decode + mma consumer asymptote (3.92 TB/s, already 1.10× the floor before ramps).
   - Even if 2.3× landed, the layer would be **~1.88**. MoE exactly at its byte floor on every shape gives **1.99**.
2. **78% of the remaining MoE gap is at T = 128 / 256, and it comes from one mechanism: how hot experts
   (> 8 tokens) are processed.**
   - Shipped v3.3 HOTCAP reads them in 64-token passes over 128-row tiles. That costs about 1.8× the weight bytes
     again in activation (X) traffic, plus re-reads for experts above 64 tokens.
   - The v3.6/3.7 mma.sync hot passes cost 1 + η(n−1) stages, with η ≈ 0.55–0.7 per extra 8-token chunk.
   - A byte-and-η model built only on measured constants reproduces T = 128 / 256 within 2–4% (§3).
3. **Recommended structure: K2A v4-H, a weight-stationary "hot-expert" kernel.**
   - One CTA per SM: a producer warpgroup with setmaxnreg 40, and 2 consumer warpgroups with setmaxnreg 232.
   - Tiles are 256 rows. wgmma m64nNk16 runs in RS form with N = 8·⌈n/8⌉ ≤ 128, so every expert of ≤ 128 tokens
     is read **once**.
   - Hot W13 → SwiGLU → hot W2 runs in one launch, with per-(expert, k-stage) flags.
   - The producer gathers X itself, so the P0 launch goes away.
   - Cold experts stay on v3.6/3.7 unchanged.
   - Gain: T = 256 −64 / −32 µs (central / planning), T = 128 −35 / −18 µs per call. That is **+5.3% / +2.6%
     layer**. With the K3 fold and a small-T hand-off fix: **+7.0% / +3.3% → layer 1.82 / 1.76**.
4. **Buildable in ~6 days with ~1.5–2 GPU-h only for Phase 1** (hot kernel at T = 128/256, plus the K3 fold).
   It is gated by two ≤15-min probes (§6) before any kernel code.
   - The main risk is measured history: the team's RS-wgmma mainloops (K2b v2/v3) ran slower than modeled, and
     "loads and MMA add instead of overlap".
   - Phase 2 (one 1-CTA/SM kernel for all roles, which brings T = 32 in) and the small-T fused dataflow do not fit
     in the same 6 days.
5. **Top 50 by Nov 4 is not plausible** (my estimate: < 10%; §8).

## 1. Where the in-layer MoE time is (lb18, judge-calibrated; experts + unpooler, because the labels shift)

| shape (w) | stock µs | ours µs | floor µs | ours/floor | stock/ours | stock/floor (max ×) | weighted gap/6 |
|---|---|---|---|---|---|---|---|
| dec1 S (46) / F (16) | 78.0 / 81.1 | 28.4 / 37.7 | 24.8 / 26.4 | 1.15 / 1.43 | 2.75 / 2.15 | 3.15 / 3.07 | 58 |
| ver1 S / F (T=8) | 125.3 / 142.9 | 62.3 / 72.3 | 59.0 / 57.7 | 1.06 / 1.25 | 2.01 / 1.98 | **2.12** / 2.48 | 64 |
| dec16 S / F | 157.7 / 166.6 | 83.9 / 88.7 | 70.4 / 73.1 | 1.19 / 1.21 | 1.88 / 1.88 | **2.24 / 2.28** | 145 |
| dec32 S / F | 154.2 / 141.1 | 114.8 / 113.1 | 83.4 / 79.8 | 1.38 / 1.42 | 1.34 / 1.25 | **1.85 / 1.77** | 330 |
| ver16 S / F (T=128) | 394.3 / 397.4 | 215.3 / 219.9 | 148.7 / 148.4 | 1.45 / 1.48 | 1.83 / 1.81 | 2.65 / 2.68 | **701** |
| ver32 S / F (T=256) | 561.2 / 571.9 | 306.7 / 312.4 | 168.3 / 166.0 | **1.82 / 1.88** | 1.83 / 1.83 | 3.33 / 3.45 | **1451** |
| **weighted total** | 15,278 | 8,471 (**1.80×**) | 5,721 (max **2.67×**) | | | | **2,749** |

The floor is touched experts × 28.31 MB at 4.3 TB/s, plus a 3 µs ramp, plus the unpooler's standalone floor.

**Layer leverage per µs saved per call** (S and F of a bucket together): T=1 0.19%, T=8 0.12%, T=16 0.096%,
T=32 0.071%, T=128 0.052%, T=256 0.036%.
- A µs at T=1 is worth five at T=256. But T ≤ 16 has almost no µs left: its whole gap is 4–15 µs per call.
- Closing each bucket's gap completely would give: T=1 +1.1%, T=8 +0.8%, T=16 +1.4%, T=32 +2.4%, T=128 +4.0%,
  T=256 +5.9%. Sum ≈ +16.8%, i.e. **1.99 judge = the MoE route's hard ceiling**.

**Per-call anatomy at the large shapes** (lb18 kernel trace, µs; PDL overlaps the launches):

| T | P0 | K1 (W13) | K2 after K1 | K3 after K2 | span | rate on touched bytes |
|---|---|---|---|---|---|---|
| 32 | — | 60.8 | 32.7 | 3.2 | 96.7 | 2.69 TB/s |
| 128 | — | 134.7 | 77.3 | 5.7 | 217.7 | 2.60 TB/s |
| 256 | 9.3 (7.6 exposed) | 189.0 | 93.6 | 10.5 | 300.7 | **2.07 TB/s** |

## 2. Measured constants this study uses (nothing else)

- **Streaming.** Plain 16 B loads or 18 KB bulk stages reach 4.3–4.5 TB/s with ≥ 32 KiB/SM in flight. In the K2A
  steady state, ingest is 4.1–4.6 TB/s (A2 `rate1`). One launch costs 3.05 µs + bytes / 4.74 TB/s; each extra
  launch +0.97 µs.
- **Consumer.** Decode + 4 mma.sync per 16 B: 3.92 TB/s (1 GiB). With 8 mma per 16 B: 3.34. A hot pass costs
  1 + η(nch − 1) stages, with η = 0.55–0.7 (v3.6).
- **mma.sync** runs at 0.5–0.67 m16n8k16 per clock per SM, about 50–67% of wgmma peak (2048 FMA/clk/SM).
- **Clock.** The in-kernel SM clock drops to 1786–1806 MHz with any mma.sync stream (k35, clock64/globaltimer);
  decode alone stays at 1971–1978 MHz.
  - nvml reads 1980 MHz in every A2/K2b run. nvml is not the effective SM clock, so the K2b "1980 MHz under wgmma"
    reading does not show that wgmma escapes the drop (P5).
- **wgmma constraints.**
  - ptxas serializes at ≤ 80 registers (C7512), and on any per-item branch between the wgmma and mma.sync paths
    (C7520/C7513/C7510).
  - The working form is v3.3's: a role-uniform path, a per-kb double-buffered A, wait_group 1.
- **v3.3 warm path** (lab `k33_lab2`, all-hot cases), µs: 8 experts × 64 tokens 200.7 (176 at 1 CTA/SM);
  8 × 128 tokens 314.9 (277).
- **K2b v2/v3** (RS wgmma, swap-AB, K2A items as A):
  - The RS mainloop issued no faster than the SS one.
  - "Loads and MMA add instead of overlap" (p1 ≈ p3 + p5 − p7).
  - The MW-2 register-fed loop was 0.28 µs/stage slower, and consumer dequant leaked 0.245 µs/stage. Both killed.
  - K2b v1 (SS, producer dequant): 1.24 µs per 2.1 M-FMA stage, i.e. ~46% of peak.
- **Cold-path X copies are free:** A2 `mathoffNX` = `mathoff` within ±1 µs at bs16 (6 routings). Removing K1's X
  copies saves nothing, so at ≤ 14 tokens X ingest is not a cost.
- **Small-T structure (A2, traced, layer routing):** hand-off + tail is 10.9 / 11.7 / 13.2 / 13.3 µs at
  bs1 / 8 / 16 / 32.
  - The cause is the K1 finish spread (per-SM rate ±6% + first-land skew) and K2's late entrants.
  - Every atomic or flag round trip from a streaming SM waits behind ~3 µs of in-flight copies (v3.7).
- **Clusters:** 2-CTA +8…+16 µs, 4-CTA +47…+111 µs (v3.9). L2 prefetch and tilts: ≤ ±1.4 µs (v3.8).

## 3. Diagnosis: the hot-expert byte account explains T = 128 / 256

Layer routing, token counts per touched expert (a2/k37 ROUTING lines):
- T = 256: [122, 83, 45, 39, 30, 28, 15, 13, 9, 6, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1];
- T = 128: [49, 27, 27, 21, 13, 13, 8, 6, 4, 3, 2 ×5, 1 ×5];
- T = 32: [26, 11, 7, 6, 3, 3, 2, 1, 1, 1].

**X traffic per weight byte.** For a pass with N padded tokens over R-row tiles, X bytes / W bytes =
(2N·K)/(0.5625·R·K) = **3.56·N/R**.
- HOTCAP (R = 128, N up to 64): 1.78 at N = 64. Every 64-token pass streams 2.78× the expert's weight bytes into
  the SMs. An expert above 64 tokens pays that twice.
- RF limit: FP32 accumulators in registers cap R × N at ~40–45 K per SM. So the ratio is ≥ 3.56·N²/45 K, i.e.
  1.3 at N = 128 and 0.32 at N = 64. Even the ideal tile has X ≈ W for the very hottest experts.

**Model** (chip-wide µs; W-eq = one expert's 28.31 MB = 6.43 µs at 4.4 TB/s):

| T | part | today (v3.6/3.7 as shipped) | W-eq | µs |
|---|---|---|---|---|
| 256 | hot (6 experts > 16 tok) | HOTCAP: passes N = 64+64, 64+32, 48, 48, 32, 32 → W 226 MB + X 295 MB | 18.4 | 118 × κ 1.1 = 130 |
| 256 | cold (17 experts) | NC1, 15/13/9 re-passed → 20 passes + X | 21.7 | 140 |
| 256 | structure | P0 7.6 + K3 10.5 + ramp/hand-off/tail ~12 | | 30 |
| 256 | **total** | model 300 vs **measured span 300.7** | | |
| 128 | hot (6 > 8 tok) | HK3 NCH4 on mma.sync: 49 → 2 passes; 27, 27, 21, 13, 13 one each; 1 + 0.6(nch−1) | 16.8 | 108 |
| 128 | cold (14) + structure | | 14.9 | 96 + 18 |
| 128 | **total** | model 222 vs **measured 217.7** | | |

κ is the warm path's measured inefficiency over its ingest bound:
- κ ≈ 1.0–1.1 on 8 × 128 tokens (277–315 µs vs 286 modeled);
- κ ≈ 1.2–1.4 on 8 × 64 tokens (176–201 µs vs 143).

**Alternative reading (the P1 decision).** If X from L2 is free even at 32 KB/stage, as it is for cold stages,
then the same 8 × 64 data says the warm path runs MMA-limited at ~30% of tensor peak:
25.8 G FMA in 176 µs = 1.47e14 FMA/s.
- Either way, the new design attacks the bound: fewer bytes (R = 256, no re-pass) and a non-spilling, deeper
  1-CTA wgmma consumer.
- But the risk differs. Byte accounting has predicted well here (lessons §1). The MMA-efficiency fixes are what the
  team missed twice (K2b v2, v3).

## 4. Candidates

Per-call µs model, layer routing, Δ vs v3.6/3.7 as shipped (box µs: bs1 29.8, bs8 68.0, bs16 71.3, bs32 103.0,
T128 210.3, T256 314.8; lb18 in-layer spans 96.7 / 217.7 / 300.7). Cells are central / planning.

| # | candidate | bs1 | bs8 | bs16 | bs32 | T128 | T256 | layer (central / planning) | legal | risk | effort |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | **Weight-stationary hot kernel (Phase 1)** | 0 | 0 | 0 | 0 (route off; +5 if on) | **−35 / −18** | **−64 / −32** | **+5.3 / +2.6%** | yes | high | 4 d |
| K | K3 fold at T ≥ 128 (last-arriver post-norm in K2) | 0 | 0 | 0 | 0 | −2.7 / −1.3 | −7.5 / −3.7 | +0.5 / +0.25% | yes | low | 0.5–1 d |
| F | Small-T dataflow fusion (W13→W2 one launch, per-(e,s) deps) | −4 / −2 | −4 / −2 | −4 / −2 | −4 / −2 | (in H) | (in H) | +1.4 / +0.7% | yes | high (v3.4 ±0) | 3 d |
| U | Phase 2: one 1-CTA/SM kernel, all roles (wgmma hot at T = 16/32) | 0? | 0? | −2.4 / −1.2 | −16.7 / −8 | — | — | +1.4 / +0.7% | yes | high (cold at 8 warps/SM) | +2–3 d |
| FP8 | FP4→FP8 weights + FP8 wgmma | — | — | — | — | — | — | — | **no** | — | — |
| CS | Split the clocks (stream-only SMs + MMA SMs) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | yes | dead | — |
| OV | Overlap MoE with the router / shared expert / pooler / unpooler | ≤ 1 | ≤ 1 | ≤ 1 | ≤ 1 | ≤ 2 | ≤ 2 | ≤ 0.3% | yes | — | — |
| XC | Cold X amortization (2 row tiles per warp) | 0 | 0 | 0 | 0 | 0 | 0 | 0 (killed by `mathoffNX`) | yes | — | — |

**Combined.**
- H + K + F: central +7.0% (1.82), planning +3.3% (1.76). In-layer MoE 2.13× / 1.95×.
- Adding U: central 1.85, planning 1.77. In-layer MoE 2.19× / 1.98×.

### 4.1 H: weight-stationary hot-expert kernel ("expert-parallel, weights once, tokens broadcast")

**Roles.**
- Hot = experts with cnt > H (H = 8; sweep 8/12/16): 9 experts at T = 256, 6 at T = 128.
- Cold experts stay on the shipped v3.6/3.7 cold kernels; their route lists are filtered with cnt ≤ H.

**Launch order.**
- v8_tail → **HOT** → cold K1 → cold K2 → K3, as a PDL chain.
- HOT needs only X and routing. Cold K1 prefetches its first NST weight stages before griddepcontrol.wait, which
  hides the HOT tail.

**Work.**
- A stage is (expert e, 256-row block, k-stage of 256): 2 adjacent v3-layout 18 KB stages (36 KB weights + scales)
  plus the X slice (N × 512 B).
- Stream-K over all hot stages, cost-weighted by the stage model max(ingest(N), MMA(N)). Balancing on stage count
  is the v3.3 role-split weakness: ±25% weight → 5–10%.
- Each CTA does its W13 stages first, then its W2 stages. W2 stage (e, rb2, s) waits on a flag counting the 2 W13
  256-row items that produce intermediate columns [256s, 256s + 256).
- No deadlock: no CTA waits before finishing its W13 share. Spins are bounded with an error flag (R9).

**CTA.**
- 384 threads, `__launch_bounds__(384, 1)`, setmaxnreg 40 for the producer WG and 232 for the two consumer WGs.
- Producer WG:
  - warp 0, lane 0 issues the two 18 KB `cp.async.bulk` per stage;
  - warps 1–3 gather X rows by route index (LDG.128 → STS into the 128B-swizzled K-major B tile), then
    `fence.proxy.async` and arrive on the full barrier.
  - This replaces P0 (−7.6 µs exposed at T = 256).
  - W2's B operand is the W13 epilogue's interH, written directly in B layout (v3.3 already does this): one bulk
    copy.
- Consumer WG w:
  - owns rows [128w, 128w + 128) as 2 × m64 tiles;
  - per kb: v3.3's decode → RS A fragments for both tiles, then 4 × k16 wgmma m64nNk16 per tile;
  - A is double-buffered with wait_group 1, in v3.3's C75xx-clean form;
  - N is per expert (16/32/…/128), switched per item exactly as v3.3 already switches N = 16/32/48/64.
- Registers at N = 128: 128 accumulator + 16 A + ~24 decode + ~16 addressing ≈ 184 ≤ 232.
- Smem: 2 × (36 + 64) KB at N = 128, or 3 × 68 KB at N ≤ 64. ≤ 205 KB, 1 CTA/SM.

**Epilogues** (all v3.3 code and numerics):
- W13: FP32 × gs → SwiGLU in FP32 → BF16 interH. Split items use `red.add` + last contributor.
- W2: y (FP32 red or store) for K3.

**Legality.**
- NVFP4 stays in memory; dequant to exact BF16 in registers; FP32 accumulation; BF16 activations and intermediate.
- Only 1-D bulk copies, no tensor-map descriptors (rule: none baked in prepare()). TMA gives no bandwidth
  advantage anyway (lessons §6).

**µs model, new design.**
- T = 256:
  - hot W 255 MB + X (R = 256, Σ N_pad = 416) 160 MB = 415 MB → 94 µs ingest.
  - MMA 20.9 G FMA at η 0.45 → 96 µs.
  - Hot total: max × κ 1.1 = **106** (range 94–144 for κ 1.0–1.4, η 0.3–0.6).
  - Cold (14 experts, 1 pass each, ~416 MB-eq) 95; structure ~24 (cold ramp/hand-off/tail 12, HOT ramp + W13→W2
    wait + tail 9, K3 3).
  - **≈ 225–230 vs 300.7.** H alone (P0 removal included, K3 excluded) is **−64** (−33…−80).
- T = 128:
  - hot W 170 + X 68 = 238 MB → 54 µs; MMA 8.85 G FMA → 41 µs; hot 60 (54–77).
  - Cold 96, structure 24.
  - **≈ 180 vs 217.7 → −35** (−21…−41).
- T = 32 as a separate launch: hot 19 + cold 55 + structure 28 (+9 for the extra launch) → +5 µs. **Route off.**
  It only pays inside U.
- Planning (×0.5 on the gains): T = 256 −32, T = 128 −18.

**Evidence that this is physically real, not just modeled.**
- MoE-board #1 (2.523) cannot be uniform: at bs32 it would sit below the floor.
- Plausible splits put bs128/256 at ~2.9×, i.e. ~1.05–1.3× the floor with hot experts. Someone streams hot experts
  near the floor.
- emeraldant's hybrid gave only +3% at bs256, but it kept 64-token × 128-row tiles: the same 1.8× X tax.

**Risks** (in order):
1. The RS mainloop base. K2b v2/v3 missed by 1.3–2.7× on an unmeasured base → P2 gates it.
2. ptxas C75xx with N switching at 232 registers → sass_gate before GPU.
3. 1-CTA/SM tails and in-kernel W13→W2 waits; ~9 µs is modeled.
4. Replay soak (t/42): `fence.proxy.async` before every empty-barrier arrive after generic reads, and split-K reds
   into a self-zeroing workspace. Run 2,000 queued replays per scored shape.

**Effort.** HOTCAP is the base: decode, wgmma wrappers, the interH layout, route prologue and warm epilogues exist.
- ~1,200 CUDA lines new or changed: 2.5–3 d.
- Self-test and bench rows in k37_bench style: 0.5 d.
- gen_k3 integration and route table: 0.5 d.

### 4.2 F: "single persistent W13→SwiGLU→W2 with an on-chip intermediate", and why v3.4 lost

- **On-chip as stated is impossible.**
  - One expert is 28.3 MB; one SM ingests ~33 GB/s, so one SM per expert would take ~860 µs. An expert must span
    ≥ 64 SMs.
  - W2 needs all 4096 intermediate columns per output row, so the intermediate (8 KB/token) must cross SMs through
    L2. Its bytes are trivial; the cost is the synchronization.
- **Why v3.4 lost (0.95–1.00).**
  - Its dependency was per expert, and stream-K spreads an expert's W13 over whole CTA ranges. Every expert
    therefore completed at the end of the W13 phase, which is a grid barrier, plus per-item fences.
- **What would fix it.**
  - Make W2 stage (e, rb2, s) depend only on the 4 × 128-row W13 items that produce intermediate stage s.
  - Give W2 k-stage s to the CTAs that produced (or neighbour) those W13 rows.
  - Keep the producer streaming W2 weights across the phase boundary.
  - Fold K3 as a per-expert last arriver.
- **Ceiling** (A2): hand-off + tail is 10.9–13.3 µs at bs1–32. Fusion can remove the global finish spread and the
  late entrants, but not the neighbour spread.
  - Each dependency poll is a ~3 µs round trip behind in-flight copies (v3.7).
  - W2 becomes 4–16-way split-K, and J1 measured split costs of 0.9–5.5 µs.
- **Model:** central −4 µs, planning −2 µs per call at bs1–32. Layer +1.4% central, +0.7% planning.
- **Probe P4 decides it:** if the nullH ceiling is < 6 µs at bs1–16, kill.

### 4.3 wgmma "with the 128-register / 1-CTA trade done right"

- This is H's consumer: setmaxnreg moves the registers to the consumers, and the bulk-copy producer decouples bytes
  in flight from occupancy. A 205 KB ring at 1 CTA/SM gives more in flight than today's 2 × 68–100 KB.
- **Why only the hot role.**
  - Cold stages (≤ 8 tokens) already stream at 4.1–4.6 TB/s steady with 16 consumer warps/SM.
  - At 1 CTA/SM a cold consumer would have 8 warps, and its rate there is unmeasured (P3).
  - wgmma m64n8 has no per-FMA advantage over mma.sync n8 for 1–8 tokens.
- **SS vs RS.**
  - SS with a dequant WG would move ~120 KB smem per 256 × 128 × 64 stage (A write + A read + B read ×2 + landing):
    ~210 B/clk at peak MMA, above the 128 B/clk smem port. So **RS**.
  - The K2b lesson is the risk: the MW-2 RS loop waited twice per stage. Measure the base first (P2).

### 4.4 Dequant-free / FP8 paths: illegal (and mechanically broken)

- FP8 wgmma needs **both** operands in FP8, so BF16 activations would have to be quantized to E4M3.
- TASK.md: "Activations. BF16 wherever SGLang's kernel keeps BF16". Marlin W4A16 keeps them BF16. Only attention
  *queries* may be E4M3.
- A hi/lo E4M3 split of BF16 is not exact (4 + 4 significant bits within E4M3's range, subnormals lost across a
  row). A 3-term split costs 1.5× BF16 tensor work. Conservative reading: **illegal**, already listed as a dead end
  in niche_approaches.
- Mechanically:
  - E2M1 ⊂ E4M3 is exact, but the per-16-k E4M3 scale differs per row and group, while FP8 wgmma has k = 32.
    The scale can't be applied inside the instruction.
  - INT8 IMMA (E2M1 × 2 is an integer) needs integer activations: illegal too.
- Dequant stays. The only choice is where (RS registers).

### 4.5 Splitting the two clocks: dead

- The drop is triggered by tensor activity, not power: decode-only draws the same power as decode + mma at
  1971 vs 1794 MHz (k35). It applies to the chip, as far as measured.
- Stream-only SMs can't multiply. Handing them dequantized BF16 through L2 is 4× the bytes.
- An FFMA path for 1–2-token experts is issue-bound at T ≥ 8 (~2.3 ops/value, k35 §1). At T = 1 the in-layer MoE
  is already 1.15× floor.
- **One cheap unknown worth 10 min (P5):** the effective (clock64/globaltimer) MHz of a *wgmma* stream. nvml's
  "1980 under wgmma" (K2b) is not that measurement. If wgmma ran at ≥ 1900 MHz, a wgmma cold path would be a
  ~+5% layer lever. Prior estimate: ~15%.

### 4.6 Overlap inside the layer: nothing material for MoE

- **No shared expert:** the TASK.md layer has only routed local experts.
- **The router is already folded** into the pooler/v8 tail; ≤ 5.7 µs is exposed, only at T = 8.
- **MoE's neighbours (pooler, unpooler) are bandwidth-bound,** so overlap only hides ramps. PDL already does that
  (G2-B).
- **The unpooler L2-warm during K3** is the same family as v3.8's prefetch, which lost +0.3…+4.7 µs.
- **The one idle-HBM window is sliding attention at T ≤ 8** (~13–15 µs). Routing is unknown then, because it
  depends on the attention output. So experts can't be prefetched; only the pooler could be L2-warmed. That is a
  proj/glue item, not MoE; probe it separately if wanted.

### 4.7 K: glue at the MoE edges

- K3 at T = 256 is 10.5 µs exposed after K2: 410 routes, latency-bound.
- As a per-expert last arriver in K2 (and in HOT), the experts finish at staggered times and the tail is ~1–3 µs.
- At T ≤ 32 K3 is 2–3 µs, and G2 showed serial tails cost what the launch saved. **Keep K3 at T ≤ 32.**

## 5. Combined projection (per call, box µs, layer routing)

| | bs1 | bs8 | bs16 | bs32 | T128 | T256 |
|---|---|---|---|---|---|---|
| shipped (v3.6/3.7) | 29.8 | 68.0 | 71.3 | 103.0 | 210.3 | 314.8 |
| byte floor at 4.3 TB/s (+2 ramps) | 21.9 | 55.5 | 55.5 | 65.4 | 127.9 | 147.0 |
| H + K + F, central | 25.8 | 64.0 | 67.3 | 99.0 | 172.6 | 243.3 |
| H + K + F, planning | 27.8 | 66.0 | 69.3 | 101.0 | 191.0 | 279.1 |
| + U, central | 25.8 | 64.0 | 64.9 | 82.3 | 172.6 | 243.3 |

Layer (exact per-load ln model, k3_v18 calibrated loads):

| | central | planning | in-layer MoE × |
|---|---|---|---|
| H + K + F | 1.704 → **1.82** | → **1.76** | 2.13 / 1.95 |
| + U | 1.85 | 1.77 | 2.19 / 1.98 |
| hypothetical 2.3× MoE (gap closed 66% everywhere) | 1.88 | — | 2.30 |
| MoE at its byte floor on every shape | 1.99 | — | 2.67 |

## 6. Cheapest probes (each ≤ 15 min of GPU, with a gate decided before the run)

- **P1, HOTCAP anatomy (base of H; existing code + masks, ~2 h laptop).**
  - k2a_v37 vid 2 K1/K2 get runtime mask bits:
    - m1 skip X copies (stale smem, expect_tx adjusted);
    - m2 skip wgmma (A sunk via asm);
    - m3 raw words instead of dequant;
    - m4 skip re-pass weight copies;
    - m5 all off.
  - Cases: 8 × 64 tok, 8 × 128 tok, layer T = 256 ×3, layer T = 128.
  - Read: X-bound if m1 ≥ −25% on 8 × 64; MMA-bound if m2 ≥ −25%.
  - **Kill H if null (m5) is within 15% of base**: then the cost is structure, and the byte model is wrong.
- **P4, nullH (ceiling of F; same build, one flag).** K2 skips griddepcontrol.wait and reads stale X (timing only)
  at bs1/8/16/32. Gate for F: ≥ 6 µs at bs1–16.
- **P5, tensor clock (same burst).** In-kernel MHz (k35 probe) for the HOTCAP warm role on 8 × 64 vs the k35
  mma-only stream. ≥ 1900 MHz opens a wgmma-cold study; otherwise close the clock question for good.
- **P2, hot-mainloop base (new mini-kernel, ~1 d laptop, 10 min GPU). This is the gate for writing H.**
  - Standalone: 132 × 384 threads, setmaxnreg 40/232, the §4.1 ring and consumers.
  - Load: 8 synthetic experts (W13 + W2 sizes), X in L2, N ∈ {16, 32, 64, 128}.
  - Rows: full / X-off / MMA-off / loads-only, plus in-kernel MHz.
  - **Gate:** full ≤ 1.3 × max(ingest / 4.4 TB/s, FMA / (0.5 × 4.84e14)) at N = 64 and 128; and full ≤ 0.75 ×
    P1's HOTCAP µs on 8 × 64 and 8 × 128; and sass_gate clean (no C75xx, 0 spill).
  - Fail → stop MoE structure work for the kernel track.
- **P3, only for U.** Cold consumer at 1 CTA/SM: 8 warps, each owning 2 tiles (2× ILP). Pass at ≥ 95% of v3.6's
  bs8/16 rate.

P1 + P4 + P5 fit one pod burst (one build, ~10 min of runs).

## 7. Six-day plan and GPU budget (Phase 1 only)

| day | work | GPU |
|---|---|---|
| 1 | P1 / P4 / P5 masks into k2a_v37 (laptop) → one burst. Start P2 mini-kernel. K: K3 last-arriver at T ≥ 128 (laptop) | 15 min |
| 2 | P2 finished + emulated → burst. **Gate H** | 15 min |
| 3–4 | H kernel: HOT W13 → flags → HOT W2, producer X gather, route filter in the cold kernels, self-tests (k36 cases up to M = 320), sass_gate | 2 × 15 min |
| 5 | k37_bench A/B (H sweep 8/12/16; T = 128 / 256 rows), gen_k3 → k3_v20 (route T ≥ 65), judge clone, memcheck, 2,000-replay soak | 2 × 15 min |
| 6 | okc check → scored (judge, no box). Slack for one fix round | 0–15 min |

- Total ≈ 1.5–2 GPU-h at $3.6–5.3/h. Balances are $3.73 + $15: enough, with almost no slack; only the user tops up.
- F and U don't fit in the same 6 days for one Opus agent. F only goes ahead if P4 ≥ 6 µs and a second agent is free.

## 8. Ranked recommendation and an honest top-50 statement

1. **Probe burst P1 + P4 + P5 (day 1).** It is cheap, it decides H's mechanism, and it closes the clock and fusion
   questions with numbers.
2. **H (weight-stationary hot kernel) for T = 128 / 256, only if P2 passes.**
   - Largest remaining MoE lever with a byte-level mechanism: +2.6% planning, +5.3% central.
3. **K (K3 fold at T ≥ 128):** +0.25–0.5%, low risk. Ship with H.
4. **F (small-T dataflow fusion):** only if nullH ≥ 6 µs. +0.7–1.4%, with a ±0 precedent (v3.4).
5. **U (all-roles 1-CTA kernel):** only after H ships and P3 passes. +0.7–1.4%.
6. **Closed:** FP8 / INT8 compute (illegal), clock split, in-layer overlap, cold X amortization, clusters, L2
   prefetch, tail queue at small T.

**Top 50 by Nov 4: not plausible (my estimate: < 10%).**
- The bar is 1.907 today, up +0.1/day over Oct 24–25 and slowing. A guess, not data: 2.0–2.2 by Nov 4.
- The best MoE outcome is 1.76 (planning) to 1.82–1.85 (central, everything landing).
- MoE at its exact byte floor on every shape is 1.99.
- Adding the other measured levers (8-token attention at the bare stream bound +2.8% max, glue ~1%) still leaves a
  central-case best of ~1.90, i.e. at today's bar.
- Per R7, team predictions have run 2–3× optimistic.
- The remaining gap is spread out (MoE ~60%, attention ~25%, projections/glue ~15%; top teams also pack BF16
  projections losslessly), and no single route closes it.

## 9. P1 gate: built and laptop-checked (2026-10-25)

**Files.** Both are new; the shipped path (k2a_v37.py, k3_*) is untouched:
- `kernels/lab/hot_p1_lab.py` (generator);
- `tools/box/hot_p1.py` (driver).

**The lab copy.** The lab is v3.7's CUDA source read via ast and patched at 30 region-checked anchors.
- Each patch must match exactly once, or the generator raises.
- It is emitted in 4 namespace copies, one per compile-time consumer mask: 0, hot-MMA-off, hot-dequant-off, null.
- The masks sit behind `if constexpr`, so the wgmma path never gets a runtime branch (C75xx).
- Producer and launch bits are runtime:
  - hot-X off, cold-X off;
  - re-pass weights off;
  - K2 skips griddepcontrol.wait (the fusion ceiling);
  - P0 skipped.
- Each K1/K2 CTA records clock64/%globaltimer marks: chip work = Σ busy / G, per-role spans, and MHz per CTA and
  per SM.
- No new wait loop (R9): hp_null uses the bounded mbar_wait.

**Laptop checks only — nothing has been compiled.**
- Generation works, every anchor is unique, braces balance, and all placeholders are filled.
- Every patched function was reviewed in the generated source.
- `decide()` was exercised on synthetic summaries for both readings.
- No `-std` flag is added (the CFLAGS are the shipped ones).

**Rows.**
- T=256 (vid 2) and T=128 (vid 5), against `base` = copy 0, which must match `ship` (the k2a_v37 module, k3_v18 plan):
  `nop0` (T=256), `xhot`, `xall`, `rpw`, `mmaoff`, `deqoff`, `null`, `nullx`, `nullxr`, `nowait`.
- bs 1/8/16/32: `base` / `nowait` on vid 0.

**Correction to §3.** vid 2's cold role is NC2 (16 tokens per pass). At T=256 the cold experts (≤ 16 tokens) take one
pass each. The re-reads are hot-only: 57 MB (the 122- and 83-token experts), not the 3 cold re-passes §3 counted. The
script's model column uses the corrected count.

**Command** (one burst; ~3–5 min cold build + ~6–8 min run; use `--routings=2` if the 1200 s budget is tight):

    bash tools/box/burst.sh 'python3 tools/box/hot_p1.py run 2>&1 | tee box_out/hot_p1.log' 1200

Output: `box_out/hot_p1.log` and `box_out/hot_p1/summary.json`. `python3 tools/box/hot_p1.py decide
box_out/hot_p1/summary.json` re-prints the split and the decision without a GPU.

**Decision rule** (printed as DECISION; medians over 3 layer routings). d = total µs/call vs base; dW = chip-work µs vs
base.

0. **Validity.** Every one of these must hold, or the run is LAB INVALID and gives no decision:
   - base within 3% of ship (except bs16/32, where ship runs vid 3);
   - base output within 2e-3 of ship and 1e-2 of the FP32 reference;
   - no timeout flag on ship or base;
   - lab grids equal the shipped grids.
1. **Measured value of H** (only its byte levers are credited; the consumer rewrite gets nothing):
   - H_est256 = −(0.5·dW_X + dW_R) − d_P0 − 9, with dW_X = dW(xhot) − dW(nop0), dW_R = dW(rpw), d_P0 = d(nop0);
   - H_est128 = −(0.5·dW_X + 0.5·dW_M + dW_R) − 9, where dW_M = dW(mmaoff): HK3's mma.sync hot math, half of it
     credited to wgmma.
   - The 0.5 is R = 256's halving of X; the 9 µs is H's own structure (§4.1).
2. **GREEN → write P2:** H_est256 ≥ 32 µs/call (the study's planning number). T=128 joins H only if H_est128 ≥ 15.
3. **AMBER → P2 with only its MMA-efficiency gate:** H_est256 < 32 but dW(mmaoff) at T=256 ≤ −30 µs. The bytes fall
   short, but the hot consumer itself holds the time; H is then a consumer rewrite with the K2b v2/v3 risk.
4. **KILL:** anything else. The MoE structure route closes for the kernel track.

**Readings printed beside the decision.**

| reading | rule |
|---|---|
| mechanism | ingest-bound if dW_X ≤ −40 and dW_M > −20; MMA-bound if dW_X > −20 and dW_M ≤ −40; else mixed |
| small-T fusion (F) | design it only if the mean d(nowait) at bs 1/8/16 ≤ −6 µs; else F is closed |
| clock (P5) | hot (wgmma) vs cold (mma.sync) CTA MHz, plus cold CTAs on SMs with vs without a hot CTA; a difference ≥ ~80 MHz means a per-SM clock (reopens wgmma for cold work); else chip-wide is confirmed |

Model expectations at seed-like routing:

| T | hot X | dW_X if ingest-bound (0 if X is free) | re-pass bytes | dW_R |
|---|---|---|---|---|
| 256 | 289 MB | −72 | 57 MB | −13 |
| 128 | 119 MB | −30 | 28 MB | −6 |

## 10. P1 reread and P2 (2026-10-25, after the burst; lead: the H256 gate is dropped)

### What `null` −86.5 is made of, and why `mmaoff` −11.6 understates the MMA cost

The reading uses the per-role chip work from the P1 marks (`box_out/hot_p1/summary.json`), T=256, median of 3
routings, in µs of chip work. Base is K1 hot 71.0 / K1 cold 73.1 / K2 hot 56.7 / K2 cold 50.7.

| row | K1 hot | K1 cold | K2 hot | K2 cold |
|---|---|---|---|---|
| null | −26.7 | −22.3 | −20.2 | −19.2 |
| mmaoff | −21.5 | — | +9.9 | — |
| deqoff | −10.8 | — | +5.1 | — |
| nowait | — | — | −24.6 | −18.8 |

1. **The null delta splits about evenly between the roles:** hot −46.9, cold −41.5.
2. **K2 chip work is mostly waiting.** About 43 µs of it is waiting for the last K1 CTA, as nowait shows.
   - Any row that speeds the hot K1 role makes the hot K2 CTAs start earlier and wait longer. That is the +9.9 that
     hides mmaoff's real −21.5 in the total.
   - The same artifact made most of P1's xhot/xall K2 savings. Their deferred X copies took away the wait; in K1 the
     X copies cost +1.2 / −0.2, i.e. they are free.
3. **The clean K1 numbers.** The hot consumer (−26.7) is wgmma (−21.5) plus dequant (−10.8), and the two overlap.
   - So the hot role's own feeding (weight LDS, flush, hand-off) is ≤ ~5 µs.
   - The other ~22 µs of K1 consumer cost is the cold role (NC2 mma.sync: X-fragment LDS, decode, 8 HMMA per 16 B,
     epilogue). It is unmasked, so its parts are not separated.
4. **T=128 (mixed CTAs):** null K1 −34.0, K2 −7.4; mma.sync hot −11.9; dequant −7.2 / −6.3; nowait K2 −11.0.
5. **A 5-minute extra probe would split the cold consumer.** It needs one rebuild of hot_p1_lab with 3 more
   compile-time copies:
   - cold mma off;
   - cold B fragments from registers, i.e. no X LDS (v3.7's `consume_kb_breg` path);
   - split-item epilogue and hand-off off (fmode-6-like).
   - That is about 1 min of build plus 2 min of rows.
6. **H256 under this reading** (×0.5 on the consumer savings):
   - **Hot experts only:** ≈ 0.5 × (26.7 + ~13 for K2) + 10 (P0) − 9 ≈ **21 µs**. That is below 32 on its own.
   - **Every expert at T=256 on the new consumer:** add 0.5 × the cold bound (22.3 + ~11) and 0.5 × the 21.5 µs
     K1→K2 hand-off. That gives ≈ **48 µs**.
   - So the structure that pays is the all-experts weight-stationary kernel, not a hot-only add-on.
   - Per the lead, P2 is built regardless of the number. The old kernel's probes no longer gate the new structure.

### P2 as built

Files: `kernels/lab/hot_p2_lab.py` and `tools/box/hot_p2.py`. Laptop-checked only:
- generation works, braces balance, the sass_gate substitution equals the build source;
- the planner and `decide()` were run on synthetic routings;
- no nvcc, torch or GPU here.

**Kernel `p2_gemm<MODE>`:**
- One CTA per SM, 384 threads, setmaxnreg 56/224 (K2b v2's validated numbers).
- Producer: one lane issues two `cp.async.bulk` per 64-k stage, 9216 B of NVFP4 and N×128 B of tokens, into a
  4-slot ring with one expect_tx barrier.
- 3 dequant warps: K2A v3's exact `deq4`/`bmul2` → BF16 A tile in smem (K2b v1's IL layout), double-buffered, with
  `fence.proxy.async` before the arrive.
- 2 consumer WGs: each owns 128 rows, as 2 × wgmma m64nNk16 **SS**, N ∈ {8,16,32,48,64,96,128}.
  - Every expert of ≤ 128 tokens is read once.
  - Accumulators are the only register operands.
  - K2b v1's fence_acc / wait_group 1 / deferred-release discipline: no RS operands, so no C7513 pattern.
- Bounded waits only (R9). No PDL in the lab, so no stale-read hazard.

**Repack:** lossless, the same codes and re-encoded scales as K2A, with 8 consecutive k per BITPOS3 word.

**Host side:**
- Items are (pass, 256-row block), assigned by host-side LPT. This is lab-only; a real kernel plans on device.
- The token gather and SwiGLU pack run as separate prep kernels. The `p2gemm` row isolates the GEMMs.
- Projected real-kernel time = p2all + 12 µs (routing compaction, post-norm + weighted sum, launches).

**Correctness, every routing:**
- W13 and W2 outputs vs the FP32 dequantised reference (≤ 1e-4);
- SwiGLU mismatch ≤ 1e-3;
- the MoE output (P2 outputs + K3's post-norm / weighted sum in torch) vs the shipped K2A output (≤ 3e-3) and vs the
  FP32 reference (≤ 1e-2).

sass_gate runs on the lab at the start of every run (`--allow C7519`).

**Command** (≈ 8–12 min; the timeout guard skips P2 rows if a bounded wait fires):

    bash tools/box/burst.sh 'python3 tools/box/hot_p2.py run 2>&1 | tee box_out/hot_p2.log' 1800

**Report.** Per T it prints:
- ship µs and ship ÷ floor;
- P2 µs → projected µs, projected ÷ floor (floor = 2 × 3.05 + touched × 28.31 MB / 4.74 TB/s: 143.5 µs at T=256);
- gain vs ship, the correctness figures, the sass_gate exit code;
- a `next:` list read off the diagnostic rows (p2gemm, p2nodeq, p2nomma).

Verdicts:

| verdict | condition | what follows |
|---|---|---|
| ON TARGET | projected ≤ 1.2 × floor at both T | build the real fused kernel |
| AHEAD OF SHIP | beats ship, not yet on target | iterate on the `next:` items |
| BEHIND SHIP | slower than ship at some T | iterate before building anything |
| INCORRECT | correctness checks fail | read no timing |

**What I'd iterate on next, in order:**
1. **W2 load balance.** 368 items over 132 CTAs, LPT imbalance 1.19 vs 1.08 for W13. Split W2 items along K, or use
   128-row W2 items.
2. **Fuse the prep.** Gather tokens in the producer (cp.async 16 B, as K2b v1), SwiGLU in the W13 epilogue, and
   W13 → W2 in one launch with per-(expert, k-stage) flags. That removes 3 launches and the global K1→K2 wait, which
   P1 measured at 21.5 µs at T=256.
3. **If p2nodeq shows the dequant WG binding:** 2 dequant WGs (512 threads), or RS consumers that dequantise for
   N ≥ 64.
4. **Small-N passes (N = 8/16) are ingest-bound by construction.** Check them against the weights-only model and
   batch two cold experts per CTA item if the ramp shows.

# K2A v3.6 design: per-expert pass width (hot experts read once), cold experts unchanged (2026-10-25)

Status: **plan + laptop build**. The compile gate runs on the serverless H200 (compile only). Timing is the lead's
GPU run (`tools/box/k36_bench.py`). Files:
- `k2a_v36.py`: the moe-problem file. It is v3.3 plus HK2 variants, the self-test and the route table.
- `k2a_v36_emulate.py`: CPU emulation of the routing, range and flush index math, with negative controls.
- `tools/box/k36_bench.py`: v3.6 vs v3.3 (shipped plans) vs NC2, paired, in one process.

## 1. The measured problem (A2 lab, `box_out/a2_lab.log`, `a2_lab/summary.json`)

- At bs32 on the layer routing, the hot experts carry 24–26 tokens, so NC1 (8 tokens per pass) streams them 3–4 times.
  - base 108.8 µs, null 89.0, nullD 65.8: the re-passes cost **~23 µs** with a null consumer.
  - bs16 layer: **~3.4 µs**.
- Fit of v3.3 over all 28 lab routings (bs 1/8/16/32, layer + moe):
  - **t = 15.4 µs + 1.133 µs × (stages per CTA, K1 + K2)**, rms 1.3 µs.
  - A re-pass costs a full stage in real math (ρ = 1.0; it is 0.9 in null, where re-pass bytes partly hit L2).
  - So with real math, a re-pass stage costs ~1.13 µs per CTA-stage. Decode, X and mma are all paid again.
- NC2 / NC4 for all experts (lab rows `nc2`, `nc4`), fitted the same way (mean model, every stage counted):
  - **every extra 8-token chunk in a stage costs η ≈ 0.55–0.7 of a stage**;
  - plus a flat build penalty σ = 1.063 (NC2) and 1.29–1.36 (NC4 at 2 stages);
  - at bs8, where nch = 1 everywhere, `nc2` is still 6% slower.
- A hot pass at nch chunks therefore costs 1 + η(nch − 1) stages instead of nch. The gain hinges on η.
  - **η is the crux, and it is not measured for a dedicated wide-pass consumer.**
  - The NC2/NC4 builds use a runtime-bounded chunk loop (`if (q < nch)` over QMAX). Their SASS has rolled HMMA loops:
    8 HMMA static in K1 NC4, against 20 in NC1.
- Physics bound on η. One extra chunk per CTA-stage is:
  - 128 HMMA → ~210 clk at ~0.6 HMMA/clk/SM;
  - 32 KB of X smem reads (8 warps × 4 KB) → 256 clk at 128 B/clk;
  - ~56 issue clk/SMSP.

  That is **0.10–0.15 of the 2000-clk stage period**. So η ≈ 0.6 is codegen / latency, not throughput (or it is
  something the lab did not resolve). At nch = 4 the hot stage carries 16.5 KB of X from L2 on top of 18 KB of
  weights. If per-SM ingest (~45 GB/s seen in mathoff) binds, that alone gives h(4) ≈ 1.6, i.e. η ≈ 0.2.

## 2. Options (bytes in flight, registers)

All three options are mma.sync m16n8k16, 320 threads (producer + 8 consumers + epilogue), 2 CTAs/SM, ≤ 96 registers.
This avoids the wgmma serialization rules (C7510/C7513/C7520) and v3.4's measured warm-path loss at bs16/32
(0.93–0.96: P0 launch, 96 registers with spills, 2-stage ring).

| option | hot stage | weight bytes in flight / SM | regs (hot path) | verdict |
|---|---|---|---|---|
| **(a) two work-unit kinds in one launch**: cold items = v3.3's NC1 code; hot items (> 8 tokens) = one pass of ≤ 32 tokens, compile-time chunk count `hc_item<2/3/4>` | 18 KB W + ≤ 16.5 KB X, ring 3 × 35.3 KB = 106 KB (113 KB/CTA with static: **fits 2 CTAs/SM, 5 KB spare**) | 2 stages × 18 KB × 2 CTAs = **72 KB** (≥ 64 KiB target, lessons §5) | acc 16 + A 16 + X 8–32 + w 4: est. 80–92 | **chosen** |
| (b) hot units loop over token chunks, weight tile kept dequantized (regs/smem) | decoded stage = 64 KB bf16 smem, or 64 regs/thread for 4 kb of A | ring shrinks to 1–2 stages: ≤ 18–36 KB/SM, ~80% BW | > 120, or smem ring cut | for ≤ 32 tokens it *is* (a): the A fragment of a kb is reused across the nch chunks in registers. Beyond 32 it does not fit |
| (c) split a hot expert's rows across more CTAs (64-row half-stages, all tokens) | 9 KB W + 16.5 KB X per stage, 2× the stages and mbarrier round trips | 36 KB/SM of weights | ~75 | X/W ingest 1.8× per byte, decode duplicated or still NC4 accumulators: loses |

**Load balance** (the v3.3 role split was "sharp": ±25% weight costs 5–10% at bs256).

- The model says a 25% error in the hot/cold weight turns −14.5 µs into −8.4 µs at bs32.
- v3.6 therefore gives every CTA both kinds by default (**dual range**). CTA c takes stages [c·Sh/G, (c+1)·Sh/G) of
  the hot sequence and [c·Sc/G, (c+1)·Sc/G) of the cold sequence.
  - Every CTA does 1/G of each kind, whatever η really is. There is no weight to tune.
  - Order: CTAs in the first wave (blockIdx < G/2) run hot then cold, the second wave cold then hot. K1's co-resident
    pairs are blockIdx (i, i+132 | i+124 | i+84) (A2 trace smids), so an SM rarely runs two hot parts at once.
- Cost: hot items (16 stages) get split ~2-way along K, because a CTA's hot share is ~6–12 stages.
  - That adds ~1M extra FP32 reds into h/y per kernel at bs32 (v3.3 does ~0.5M), finished by the existing epilogue
    warp.
  - At most 4 split items per CTA (named barrier ids 2..5 ≤ 15; the emulator checks this).
- The v3.3-style role split (first Gh CTAs hot, Gh from a cost model with η16) is kept as runtime mode 0, for the bench.

**Cold path = v3.3 bit-for-bit numerics**:
- the same consume code (two accumulator chains);
- the same flush (SwiGLU into interS once, FP32 atomics for split items, y + per-route Σz² in K2);
- the same epilogue warp, evict_first weight copies, PDL chain and bounded mbarrier waits (flag + return, no trap).

Hot items use the same flush formulas. Only the FP32 summation order inside a pass changes (one chain per chunk), as
in the shipped NC2 variant. NVFP4 weights and scales stay as they are (same 18 KB stages); X and inter stay BF16.

## 3. The model per bucket (layer routing; µs; median of the 3 lab routings)

t = A + τ·Σ_stages cost / G, with A = 15.4, τ = 1.133, cold stage = 1, hot stage = 1 + η(nch − 1) (dual range: max = mean).

WASTE terms of v3.3 → v3.6. Each cell is v3.3's lab value → v3.6's predicted value at η = 0.45.

| bs | floor | X-ingest | re-pass W | hot chunks (new) | hand-off | tail | ramp | v3.3 | **v3.6 (η .25 / .45 / .65)** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 12.0 | 0.4 | 0 | 0 | 7.3 | 3.6 | 4.0 | 28.7 | **= v3.3** (no expert > 8 tokens; route table keeps vid 0) |
| 8 | 47.8 | 2.2 | 0 | 0 | 5.8 | 5.9 | 4.5 | 66.0 | **= v3.3** (max 8 tokens/expert; vid 0) |
| 16 | 41.8 | 5.7 | 6.4 → 0 | +3.0 | 6.3 (+0.2 split) | 6.9 | 4.8 | 70.2 | **−4.6 / −3.2 / −2.0** |
| 32 | 53.8 | 10.8 (same bytes, packed into hot stages) | 25.2 → 0 | +10.7 | 6.8 (+0.5) | 6.5 (+0.5) | 5.5 | 108.8 | **−18.8 / −13.5 / −8.2** |

- moe routing, same model (`k36_bench.py model`):
  - bs16 ±0 (3 of 4 routings have no expert > 8; seed 101 −3.6 at η .45);
  - bs32 median −4.9 / −3.6 / −2.3 (one hot expert of 11–21 tokens).
- The extra-split cost is included in the bold column: −0.3 µs at bs16, −1.0 at bs32.
- Hand-off and tail are unchanged by construction: both ranges are balanced, and the last part of half the CTAs is cold.
- Ramp grows ~+0.1 µs (two range computations). Floor is unchanged.
- NCH 2 instead of 4 (16 tokens per hot pass) at η .45: bs32 −10.9, bs16 −3.5 → kept as bench variant vid 5.

R7 haircut ×0.5 on the η = .45 column: bs32 ≈ −7, bs16 ≈ −1.6 (the blind case). Layer gain if η = .45 holds:

| bs | Δ µs | ×%/µs | layer gain |
|---|---|---|---|
| 32 | −13.5 | 0.067 | +0.9% |
| 16 | −3.2 | 0.089 | +0.28% |

Total ≈ **+1.2% layer** (≈ +0.6% after the haircut).

## 4. Gates and kill (lead's GPU run, `k36_bench.py`, same process, paired)

1. **Compile gate** (serverless, done here): `sass_gate.py` on the HK2 kernels. Need no C75xx except C7519, 0 spills,
   ≤ 96 registers.
2. **Self-test** (in the bench `--check` and in `prepare()`).
   - Every bucket's plan and both balance modes: rel-L2 < 1e-2, worst row < 5e-2, err flag 0, two calls in a row.
   - Cases: normal 1/5/16/33/100/257/320, skewed 12/16/24/32/128, and `hot8`. `hot8` sends every token to the same
     8 experts, so nch = 4 with ntok 20/32.
   - The vid 0 output must match v3.3's shipped module within the same tolerance, and v3.6 must match v3.3.
3. **Timing gate** (layer routing, median of 3):
   - Δ(v3.6 − v3.3 shipped plan) ≤ **−12 µs at bs32** and ≤ **−3 µs at bs16**;
   - no bucket slower than its fallback: bs1/8 stay on vid 0, so they are equal by construction. moe bs16/32 must not
     be slower than v3.3 by > 1 µs, else that bucket falls back.
4. **Kill**: Δ > −6 µs at bs32 (layer). Then η ≥ ~0.8: the wide pass is not cheaper than re-reading, and MoE stops at
   v3.3 (lessons: rate-bound with mma.sync).
5. **Diagnostics** that decide the next step whatever the outcome:
   - `v36_all` (every expert hot, hot_t 0) vs `v33` at bs8: the hot path's own build penalty σ_h (nch = 1 vs NC1).
   - At bs32, `v36_all` against `nc2`/`nc4` gives η for the compile-time chunk path vs v3.3's runtime-bounded NC2.
   - `role η4/8/12` vs `dual`: whether the role split's SM pairing beats exact balance.
   - `v36_n2` (NCH 2) vs NCH 4.

## 5. As built (k2a_v36.py, nonce k2a-v36-1)

- `V3_VARIANTS` = v3.3's three, plus vid 3 `(1, 3, 2, 2, 4)` = HK2: cold NC1, hot NCH 4, 3 ring stages of 35.3 KB.
- `V36_PLAN`:

  | bucket | plan |
  |---|---|
  | bs1, bs8 | vid 0, v3.3 |
  | bs16, bs32 | vid 3, hot = (hot_t 8, mode 3 dual + second wave cold-first, hot_w4 8) |
  | bs128, bs256 | vid 2, v3.3 wgmma |

- `prepare()` self-tests every bucket's plan. On failure it falls back to `V33_PLAN`, then v2.2, then the incumbent.
  A variant below 2 CTAs/SM is never used. `v36_call` refuses HK2 above M = 48.
- New device code (namespace k3):
  - `route<NC, TPH, HGE>`: v3.3's call sites are unchanged;
  - `flush_item<NQ, MODE, NOSWI>`, `cold_item`, and `hc_item<NQ>` with a half-kb decode;
  - `gemm36`, `ranges36`, `segs36`, `k1_kernel36` / `k2_kernel36`, and the host `launch36` / `attrs36`.

  vid 0–2 compile to v3.3's code (gate SASS: same registers 73/74, 71/80, 96 + spill, and the same instruction
  counts 2992/2912; only the constant-bank offsets move, by the one new Params field).
- Register work to reach ≤ 96 at NCH 4 (the compile gate's history is in `box_out/k36_gate/gate*.json`):
  1. decode a kb in two halves, so 8 A registers and 4 X registers per chunk are live;
  2. hold the part ranges and segments in shared memory (`segs36`);
  3. offset the output bases by an opaque zero (`asm volatile "+r"`, `K36_LAUNDER`), so the per-chunk 64-bit
     addresses are formed inside the flush instead of being hoisted into the prologue. Laundering the pointer itself
     (gate 5) had turned every atomic and store generic (`LD/FADD/ST` with a shared-window check); the integer form
     keeps them `ATOMG`/`STG`;
  4. K1 hot items with ≥ 3 chunks hand their SwiGLU to the epilogue warp (`NOSWI`): FP32 reds onto zero h, then the
     same SwiGLU → bf16 → interS. The IEEE-division slow-path call no longer sits in the 16-accumulator flush.
     Whole items take a barrier id only while npart < 10, which keeps ids ≤ 15 (the emulator checks it); past the
     budget the error flag is set.
- Emulator (`k2a_v36_emulate.py`): 81 routings, all 28 A2 lab routings included. Grids 264/263/132, K1 + K2,
  NCH 2/4, modes 3/1/0, hot_t 0/8/16. Checks:
  - every stage once, the pass partition and split counters;
  - barrier ids ≤ 11 (≤ 15 always), hand-off budget never exhausted at M ≤ 48, dual balance ≤ 1 stage;
  - the flush and X-lane maps;
  - smem: 2 × 114,224 of 233,472 B.

  6 negative controls must all be caught (pass width mismatch, range off-by-one, missing role clamp, flush map,
  X-lane map, no hand-off budget).

**Compile gate** (`sass_gate.py`, judge nvcc 13.0, sm_90a; 6 serverless jobs; final run `box_out/k36_gate/gate6a.json`):

| function | regs | spill | C75xx | verdict |
|---|---|---|---|---|
| `k1_kernel36<4,3>` (plan) | 93 | 0 | none | **PASS** |
| `k2_kernel36<4,3>` (plan) | 96 | 0 | none | **PASS** |
| `k1/k2_kernel36<2,3>`, `<2,4>` (bench) | 72 / 73 | 0 | none | **PASS** |
| v3.3 vid 0/1, p0, k3 | unchanged | 0 | none | PASS |
| v3.3 HOTCAP `k1/k2_kernel<2,2,2,true>` | 96 | 228 / 156 B | none | FAIL: **pre-existing, shipped in v3.3** (bs128/256 plan, unchanged) |

- Final SASS (full range): 0 STL/LDL and 0 generic LD/ST/ATOM in the HK2 kernels.
- HMMA = 160 static per kernel: cold 16 + hc<2> 32 + hc<3> 48 + hc<4> 64, fully unrolled with compile-time chunk
  counts.
- Alternatives gated in the same jobs and rejected: `K36_KB_ROLL=1` (8 B spill), `K36_LAUNDER=0` (28 B spill), and
  `K36_HALF_FENCE` (no effect).

## 6. Lead commands (serverless; timing is the lead's)

```
# compile gate (CPU-only, ~30 s; already PASS for the HK2 kernels, re-run after any kernel edit)
python tools/serverless/sls.py run 'python3 tools/box/sass_gate.py kernels/muse_moe_w4a16/k2a_v36.py --src KERNEL_SRC_V3 --subst "@CASES3@=call:_gate_cases" "@ATTRS3@=call:_gate_attrs" "@WGSPECS@=call:_wg_specs" --inst "k3::k1_kernel36<2,3>" "k3::k2_kernel36<2,3>" "k3::k1_kernel36<2,4>" "k3::k2_kernel36<2,4>" --allow C7519 --json box_out/k36_gate/gate.json' --timeout 900
python tools/serverless/sls.py run 'python3 tools/box/k36_bench.py check 2>&1 | tee box_out/k36_check.log' --timeout 3000
python tools/serverless/sls.py run 'python3 tools/box/k36_bench.py run --bs=1,8,16,32 2>&1 | tee box_out/k36_bench.log' --timeout 3000
```

- `check` builds both modules and runs every bench row on k2a_v36's self-test cases (bs ≤ 48) plus one layer and one
  moe routing per bs: rel-L2, worst row, ~ship diff, err flags.
- `run` prints per routing: counts, the model's prediction, per row µs / x / d, and η_eff for the v36 row. The SUMMARY
  ends with the gate verdict and the best row per bucket (input for `V36_PLAN`).
- Short on time: `run --rows=core --routings=2`.
- `python tools/box/k36_bench.py model` prints the design model on the laptop.

## 7. Risks

- **η stays ~0.6** (the NC2 builds' value). Then bs32 is −8 to −9 µs: gate miss, no kill. Ship only if the in-layer
  A/B pays (≈ +0.5% layer), otherwise bundle with the next layer change.
- **Registers at the cap**: K2 NCH4 sits at exactly 96. Any later change to the hot path or flush can spill; re-run
  the compile gate (§6). NCH 2 (73 regs) is the fallback.
- **K1 hot SwiGLU on the epilogue warp**: every K1 hot item with ≥ 3 chunks now pays FP32 reds + a readback.
  - In dual mode these items are nearly all split anyway, so the cost is ~0 there.
  - In role mode the whole ones add epilogue work; `v36_r*` rows measure it.
- **smem at 113 KB/CTA** leaves 5 KB of the SM's 228 KB. Any extra static smem drops occupancy to 1 CTA/SM; the
  bench prints occupancy and the self-test refuses grid ≠ 264.
- **i-cache**: the kernel gains hc<2/3/4> bodies (~+40% K1/K2 SASS). Not measurable before the GPU run.
- **Split reds**: +1M FP32 reds per kernel at bs32 are modeled at ≤ 1 µs. The role-split row in the bench has no
  extra splits and isolates this cost.

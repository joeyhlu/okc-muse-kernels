# K1-PP: warp-specialized ping-pong wgmma kernel for L=8 (stage A design, 2026-10-24)

**What it builds on.** v5–v7 established the parts that carry over unchanged:
- wgmma FP8 QK/PV with the layouts and E4M3 Q/P precision of `k1_wg_design.md` §1–3, verified on the box;
- v7's n72 PV, which gets row sums from the tensor core;
- v7's pairwise exact max, vote-gated rescale and fused packs.

**What changes is the schedule.** It moves to FA3's structure.

## 1. Why: calibration from the v6/v7 measurements (ver32 full, B=32, ~19.3 K keys per SM)

v6/v7 run 2 CTAs/SM with 2 consumer warpgroups (WGs) each, one per KV head, and every thread loads data.

| build (ver32 full) | µs | clk per 64-key tile per SM | clk per key |
|---|---|---|---|
| bare cp.async stream (v3 measurement) | 193 | 1257 (memory) | 19.6 |
| v7 | 351.7 | 2290 | 35.8 |
| v6 | 361–366 | 2370 | 37 |
| v6, no softmax math | 257 | 1674 | 26 |
| v6, no wgmma | 259 | 1687 | 26 |
| v6, no Vᵀ transpose | 330 | 2150 | 34 |

At 2 CTAs/SM the SM alternates two CTA chains, so a CTA's per-tile chain is C ≈ 2 × the per-SM tile time. For v7:

| part of the chain | clk per tile per WG | basis |
|---|---|---|
| total chain C (v7) | ≈ 4580 | 2 × 2290 |
| softmax, S_m | ≈ 1300 | measured by removal |
| wgmma latency incl. waits, W_g | ≈ 1300 | measured by removal |
| transpose | ≈ 400 | measured by removal |
| the rest (barrier, memory wait, issue) | ≈ 1600 | remainder |

**Throughput floors** per SM per tile (both heads, 64 keys). These are counted, not measured:

| resource | count | clk |
|---|---|---|
| MUFU exp2 | 8192 at 16/clk | 512 |
| softmax ALU issue | ≈ 1120 warp-instr / 4 issue slots | ≈ 280 |
| FP8 tensor work | 1.1 M MAC at ~4 K/clk | ≈ 270 |
| memory | | 1257 |

**So v7 is latency-bound, not throughput-bound.** S_m and W_g run back to back on every WG's chain, plus a CTA-wide
barrier per tile.

## 2. The schedule

**CTA layout.** One CTA per SM, 3 warpgroups, `__launch_bounds__(384, 1)`.
- **WG0 and WG1 are consumers**, one per KV head. They use threads 0–255, so the epilogue and merge code
  (`threadIdx.x < 256`) carry over.
- **WG2 is the producer**, threads 256–383.
- `setmaxnreg`: the producer decreases to 40 registers and the consumers increase to 232.
  128 × 40 + 256 × 232 = 64 512 ≤ 65 536.

**Producer**, per tile i, with slot s = i % ST:
1. Wait on `empty[s]` (parity (i/ST − 1) & 1) when i ≥ ST.
2. Issue 8 cp.async of 16 B: K and V of 4 keys {16q+2t′, +1, +8, +9} × chunk c. K goes straight into the wgmma
   K-major layout; V goes into its own raw slot.
3. Commit.
4. For tile j = i − D (lookahead D): `cp.async.wait_group<D>`, then the 4-key byte transpose of its own V (24 PRMT
   + 16 STS.32, conflict-free) into slot s's Vᵀ, then `fence.proxy.async`, then `mbarrier.arrive(full[j % ST])`.

`full[s]` counts 128 producer arrivals. Consumers never touch cp.async or the transpose.

**Consumer c**, per tile i. There are two score accumulators S_A / S_B and two P register sets; the loop is unrolled
×2 so all register indexing is static.

```
wait QK(i)                         wgmma.wait_group<1>   (in flight: QK(i), PV(i-1); QK(i) is older)
wait full[(i+1)%ST]; issue QK(i+1) -> S_other                         <- overlaps softmax(i)
softmax(i) on S_this: mask, pairwise max, exp2, fused pack -> P_this
wgmma.wait_group<1>                PV(i-1) done (QK(i+1) may still run)
arrive empty[(i-1)%ST]             one thread per WG (empty counts 2)
vote-gated rescale of O (72 columns incl. row sums); issue PV(i) (P_this, Vᵀ slot i%ST)   <- overlaps QK wait / next softmax
```

**Ping-pong** (FA3 `NamedBarriers`, compile option `K1PP_PINGPONG`). Each consumer does `bar.sync(id_c, 256)`
before issuing its GEMMs and `bar.arrive(id_{1−c}, 256)` after. The two WGs alternate GEMM issue, so one WG's
softmax overlaps the other's tensor work.

**No `__syncthreads` per tile.** Shared state moves only through mbarriers. Every wait is bounded: a try_wait loop
with a spin cap that sets an error flag instead of hanging. No `__trap`.

**Prologue and epilogue:**
- The plan and the mbarrier init run before the single CTA barrier.
- Consumers do Q prep, new tokens, the new-tile Vᵀ and the ones rows, then a consumer-only named barrier
  `bar.sync 1, 256` plus a proxy fence.
- Epilogue: the new tile, O/l → smem, then the v4 fold merge or partials. All of it uses `bar.sync 1, 256`; the
  producer has already returned after its last arrive.

## 3. Budget

| resource | value | limit | ok |
|---|---|---|---|
| ring slot: K 8 KB (wgmma layout) + raw V 8 KB + Vᵀ 9 792 B (n72, permuted keys) | 26 176 B | | |
| ST = 7 slots | 183.2 KB | | |
| Q (e4m3, wgmma layout) | 8 KB | | |
| new-token K + Vᵀ | 17.6 KB | | |
| cos/sin + mbarriers (2 × ST × 8 B) + plan | 2.2 KB | | |
| **dynamic smem, ST=7** | **206.7 KB** | 227 KB | ✓ (ST=6: 181 KB) |
| bytes in flight (ST−1 = 6 tiles × 16 KB) | 96 KB/SM | ≥ 64 KB | ✓ |
| warps/SM | 12 (8 consumer + 4 producer) | ≥ 8 | ✓ |
| consumer registers: S×2 64 + O 36 + P×2 16 + m 2 + descriptors/addresses ~30 | ≈ 150 | 232 (setmaxnreg) | ✓ (no C7512 risk; ≥ 128) |
| producer registers: 4 page entries + addresses + 16-B staging | ≈ 36 | 40 | ✓ (else use 56/224) |
| epilogue O staging 32 KB + lse | aliases the drained ring | | ✓ |

## 4. Clock model for ver32 full (kill criterion: must be able to reach ≤ 300 µs)

**Per-SM tile time.** One CTA/SM, and both consumers work on the same tile:

T_tile = max( T_mem = 1257, C′ ) + tail

C′ is the per-WG chain with QK(i+1) running during softmax(i) and PV(i) asynchronous:

C′ ≈ max(S_m′, W_g′) + ε

| case | S_m′ | W_g′ | ε | C′ | ver32 full (302 tiles/SM) |
|---|---|---|---|---|---|
| conservative: same latencies as measured under 4 WGs/SM | 1300 | 1300 | 300 (waits, vote, pack) | 1600 | 302 × 1600 / 1.98 GHz = 244 µs + tail 25–45 µs ≈ **270–290 µs** |
| optimistic: 2 consumer WGs/SM contend less, S_m′ ≈ 900 | 900 | | | ≈ 1257 (memory-bound) | 192 + tail ≈ **220–240 µs** |
| pessimistic: no intra-WG overlap (QK(i+1) waits behind softmax) | | | | S + W + ε ≈ 2900 | 442 + tail ≈ 470 µs |

The tail (25–45 µs) is the measured v4/v7 spread between the median and the slowest CTA, plus the merge.

**Gate A: the model reaches ≤ 300 µs** in the conservative and optimistic cases, so the design is not killed. Stage
B tests whether the overlap holds.

At ver16 full (151 tiles/SM) the conservative case gives 122 µs + tail ≈ 145–160 µs, against the ≤ 165 gate.

## 5. Risks and checks

1. **The producer's own completion:** cp.async completion is tracked per thread with `wait_group`, the same
   mechanism v5–v7 validated, before the transpose and `fence.proxy.async`.
2. **`setmaxnreg`** needs all 128 threads of a WG to execute it, with `__launch_bounds__(384, 1)`. ptxas
   `-Xptxas -v` must show no C7512.
3. **The x2 unroll** keeps S_A / S_B and P_A / P_B static.
4. **Deadlock safety:**
   - every consumer `full` wait has exactly one producer arrival round;
   - `empty` waits start only at i ≥ ST;
   - CTAs with ntl = 0 skip the loop on both sides consistently, because the decision is plan-uniform.
5. **The self-test** reuses the v5–v7 L=8 cases; errors must stay within 1.5× of v7.

## 6. Stage B result (2026-10-24): gate missed, stopped

**Build.** k1_v8.py with `k1_pp<6/7>`, sm_90a, cold build on the fresh pod.
- `__launch_bounds__(384, 1)` capped ptxas at **168 registers**, 0 B of spills.
- **Correct on the first run.** The self-test passes in both merge modes with errors identical to v7 (4.174e-3,
  5.129e-3, 6.805e-3, 9.535e-3, …). The ping-pong build also passes.
- **ptxas warned, for both `k1_pp` instantiations:**
  - `(C7513) wgmma.mma_async instructions are serialized due to non wgmma instructions defining input registers of a
    wgmma between start and end of the pipeline stage`;
  - plus two injected `warpgroup.arrive` (C7519).

**Same-process bench** (box_out/k1v8_stageB.log), µs:

| shape | pp ST=7 | pp ST=6 | ping-pong ST=7 / 6 | v7 (k1_main, ST=3) | rung-1 | gate |
|---|---|---|---|---|---|---|
| ver16 full | 282.1 | 278.7 | 378.6 / 376.5 | 195.7 | 195.7 | ≤ 165 ✗ |
| ver32 full | 567.9 | 565.0 | 775.4 / 771.5 | 354.0 | 384.5 | ≤ 300 ✗ |
| ver16 sliding | 39.6 | 39.6 | — | 40.8 | 24.7 | — |
| ver32 sliding | 44.5 | 44.6 | — | 42.9 | 30.6 | — |

**Diagnosis.** The kernel lands in the model's pessimistic case or worse: about 565 µs, against the predicted ~470
for "no overlap". ptxas reports that it serialized the wgmmas.

**The intended overlap is exactly what triggers C7513.** QK(i+1) is issued, then softmax(i) defines the PV(i)
inputs (the P registers and the rescaled O accumulator) while QK(i+1)'s pipeline stage is still open. ptxas then
inserts waits, so 1 CTA/SM with 2 consumer chains is strictly worse than v7's 2 CTAs × 2 chains.

**A fix tried within the lab:** a write-only ("=f") first QK k-step, so the old S is not a wgmma input. C7513
remains; the RS-form P/O definitions are the cause.

**What would be needed** (not attempted):
1. An SS-form PV, with P stored to smem so it is no longer a register input.
2. O moved out of the in-flight window: FA3's `RescaleOBeforeGemm` / delayed-rescale ordering together with
   `warpgroup_fence_operand` discipline.
3. Or building on CUTLASS/CuTe's GMMA atoms, which manage these fences.

Each is a further multi-hour rewrite with an uncertain ptxas outcome.

**Status.** The L=8 route stays on rung-1 (the layer keeps it). v7's k1_main remains the best K1 L=8 kernel: 1.09×
at ver32 full, a tie at ver16 full. It is unrouted.

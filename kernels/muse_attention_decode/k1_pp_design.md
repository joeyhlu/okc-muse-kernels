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

## 7. k1_v8f: FA3's step order in the consumer loop (B1 of `notes/study/niche_approaches.md`, 2026-10-25)

Plan written before the code; §7.6 holds the laptop results.

### 7.1 What changes versus k1_v8

Only the consumer loop. The producer warpgroup, the mbarrier ring, `setmaxnreg` 40/232, the 3-WG layout, the
epilogue/merge, the numerics (E4M3 Q/K/V/P per A16, fp32 accumulation, n72 PV row sums) and every route switch
stay as in k1_v8.

The step order is FA3's IntraWGOverlap `fwd_step` (`mainloop_fwd_sm90_tma_gmma_ws.hpp:1165–1207`):

```
first tile (ring tile 0, or the new tile when ntl = 0):
  wait full[0]; QK(0) -> S; wait_group 0; softmax(S); P = pack(S); O *= a
step i = 0 .. ntl-2   (rolled: #pragma unroll 1)
  wait full[i+1]
  fence(S,O,P); QK(i+1) -> S; commit; fence(S); PV(i): O += P.Vt(i); commit; fence(O,P)   <- no wait between
  wait_group 1          QK(i+1) retired. PV(i) is in flight: P and O are not touched until wait_group 0
  fence(S); softmax on S in place: mask, max, a = (max moved ? exp2(m_old - m_new) : 1.0), m, exp2
  wait_group 0          PV(i) retired; fence(O,P,S)
  release slot i (empty[i % ST], one arrive per WG)
  P = pack(S); O *= a   (all 36 columns incl. the row sums; no branch, no vote)
last split: one more step with QK(new tile), PV(ntl-1)
tail: PV(last tile); wait_group 0; release (non-last splits)
```

- **Nothing is in flight across a step boundary or the loop back edge**, and no non-wgmma instruction writes a register
  that an in-flight wgmma reads. Both are true of FA3 and false of v8 (PV(i−1) crossed the back edge; P and O were
  written while QK(i+1) was in flight).
- **One S and one P register set in the rolled variant**, as in FA3 (its `tSrS` is per step). QK(i+1) may overwrite S
  because P(i) was already packed from it. v8 needed SA/SB only because it issued QK(i+1) before softmax(i).
- **Variant (b)**: the body unrolled ×2, SA/SB and PA/PB alternate statically (NVBUG 5431330 workaround direction).
  Both variants are template instantiations in one compile: stage codes **6/7 = rolled (a), 16/17 = unrolled (b)**.
  `K1_STAGES_T` keeps 7 for L=8, and L=8 stays routed to rung-1.
- **Operand fences** (`cute::warpgroup_fence_operand`): empty `asm volatile("" : "+f"/"+r" ... :: "memory")` on S,
  O and P before each issue, after each commit and after each wait. Descriptors are pinned the same way before the
  issue, so no address arithmetic can be scheduled between QK(i+1) and PV(i).
- **Each GEMM is one asm block** `{wgmma.fence; wgmma; wgmma; wgmma.commit_group}` with every descriptor and the
  scale-d predicates as inputs, so nvcc cannot place anything between the fence and the commit. The first QK k-step
  has scale-d = 0 (constant, so ptxas folds it to the zero-accumulator form).
- **Masking is chosen outside the window.** v8 branched on `need` inside the in-flight window. v8f instantiates the
  softmax masked and unmasked and branches before the issue, as FA3 splits its masked and unmasked iterations into
  separate loops.
- **Release moves one tile earlier**: slot i is released at the end of step i, after wait_group 0 (v8: in the middle
  of step i+1). The producer never waits on consumers to publish, so there is no new cycle.
- **R9:** the bounded wait (2^22 try_waits) now short-circuits once the CTA's flag is set, so each thread spins
  through at most one timeout. The producer returns on a timeout. Consumers poison O with NaN in the epilogue when
  the flag is set, so the self-test's `finite` check fails and the file falls back. (In v8 the flag was written and
  never read.) There is no `__trap`, and every named barrier keeps its count on the error path.

### 7.2 What moves in the model (versus §4)

FA3's order hides **PV(i)** behind softmax(i+1). It does not hide **QK(i+1)** behind softmax(i), which was v8's
intent: the QK latency is exposed on each WG's chain.

Per WG and tile, C″ ≈ T_QK (issue to retire of 2 × m64n64k32, ~150–250 clk) + S_m′ (900–1300) + pack and rescale
(~100; the rescale is now 36 FMUL on every tile) + ε (~150).

The two consumer WGs work on different heads of the same tile, so they run concurrently, and
T_tile ≈ max(T_mem 1257, C″) ≈ 1300–1800 clk.

| case | C″ (clk) | ver32 full (302 tiles/SM, at 1.78–1.98 GHz) + tail 25–45 µs |
|---|---|---|
| optimistic | ~1300 | 198–220 + tail ≈ **225–265 µs** |
| conservative | ~1800 | 275–305 + tail ≈ **300–350 µs** |

The ≤ 300 µs gate sits inside this range. If the build is unserialized and lands at 300–330 µs, the one remaining
lever is `K1PP_PINGPONG=1` (FA3's warp-scheduler barrier: one WG's GEMMs during the other's softmax), which needs a
compile only.

### 7.3 Gates for the pod phase

1. **Compile gate** (no GPU time):
   - Run `python3 tools/box/sass_gate.py kernels/muse_attention_decode/k1_v8f.py`, once with
     `--define K1F_PP_ONLY=1` (only the k1_pp kernels, so the exit code is their verdict) and once as shipped.
   - It must show, **per function**, for all four `k1::k1_pp<ST, UNR>` (ST 6/7, UNR 0/1):
     - zero C75xx advisories (this includes C7519, injected warpgroup.arrive);
     - 0 B spill stores and loads;
     - in SASS, no `WARPGROUP.DEPBAR` between the QGMMAs of one commit group: GMMA runs of 2 and 4, not 1, and DEPBAR
       count ≤ commit groups.
   - In the shipped build the module verdict can FAIL on `k1_main<8,3/4>` (v7's known 8–12 B spills). That is not
     this gate: read the k1_pp lines. The `K1F_PP_ONLY=1` build holds only the four k1_pp kernels plus `k1_combine`
     (no wgmma).
   - Variant (b) passing while (a) fails is a pass for (b).
2. **Unit test:** `_self_test(l8_stages=(7, 17, 6, 16, 3))` passes. L=8 errors must be within 1.5× of v7 (4.174e-3,
   5.129e-3, 6.805e-3, 9.535e-3, …), the counters must reset, and the output must have no NaN (a NaN = an R9 timeout).
3. **Timing gate** (same process as rung-1 and k1_v7 = stage code 3): best k1_pp code at **ver32 full ≤ 300 µs AND
   ver16 full ≤ 165 µs**.
4. **Memcheck:** compute-sanitizer memcheck clean for codes 7 and 17 (fold and separate merge).

**Kill criteria:**
- Both variants still serialized (C7513/C7514/C7516, or DEPBAR after every QGMMA) → stop B1. Recommend B3
  (ThunderKittens-style independent consumer WGs, `wait_group 0` after every GEMM) or B4 (SS-form PV, P through
  smem).
- Unserialized, but ver32 full > 330 µs for every code, `K1PP_PINGPONG=1` included → the wgmma chain is not the limit.
  Stop L=8 full work.
- 300–330 µs → one `K1PP_PINGPONG=1` build in the same burst, then decide.

### 7.4 Pod commands (one burst, in this order; everything writes to box_out/)

```
tools/box/box.sh sync
# 1. compile gate (compile-only). K1F_PP_ONLY=1 builds only the four k1_pp kernels (identical device code; host
#    dispatch only), so the exit code is the k1_pp verdict; the second line is the shipping build, read its k1_pp lines.
tools/box/box.sh run 'python3 tools/box/sass_gate.py kernels/muse_attention_decode/k1_v8f.py --define K1F_PP_ONLY=1 --json box_out/k1v8f_sass.json --keep box_out/k1v8f_sass > box_out/k1v8f_sass.log 2>&1; cat box_out/k1v8f_sass.log'
tools/box/box.sh run 'python3 tools/box/sass_gate.py kernels/muse_attention_decode/k1_v8f.py --json box_out/k1v8f_sass_full.json > box_out/k1v8f_sass_full.log 2>&1; grep -A4 k1_pp box_out/k1v8f_sass_full.log'
#    optional, compile-only knobs if (a) and (b) both fail (append to the first line, new --json name):
#      --define K1F_QK_WO=1        (QK accumulators write-only "=f")
#      --define K1PP_PINGPONG=1    (needed anyway for the 300-330 us branch)
# 2. unit test: self-test of every L=8 kernel (codes 7, 17, 6, 16 and v7's 3)
tools/box/box.sh run 'python3 kernels/muse_attention_decode/k1_v8f_lab.py --unit > box_out/k1v8f_unit.log 2>&1; tail -60 box_out/k1v8f_unit.log'
# 3. same-process bench (GPU lock): v8f rolled/unrolled x ST 7/6 vs v7 (k1_main ST=3) and rung-1
tools/box/box.sh timed 'python3 kernels/muse_attention_decode/k1_v3_bench.py --file kernels/muse_attention_decode/k1_v8f.py --shapes 9,11,8,10 --variants "f7=stages:7;f6=stages:6;u7=stages:17;u6=stages:16;v7=stages:3;r1=r1" > box_out/k1v8f_bench.log 2>&1; cat box_out/k1v8f_bench.log'
# 4. memcheck (codes 7 and 17, fold + separate merge, plus the routed kernel_fn)
tools/box/box.sh run 'compute-sanitizer --tool memcheck --print-limit 10 python3 kernels/muse_attention_decode/k1_v8f_lab.py --san > box_out/k1v8f_san_mem.log 2>&1; tail -30 box_out/k1v8f_san_mem.log'
tools/box/box.sh pull box_out/
# only for the 300-330 us branch: ping-pong build as its own file (CUDA_SRC hash -> its own extension), same bench
tools/box/box.sh run 'sed "s/^#define K1PP_PINGPONG 0$/#define K1PP_PINGPONG 1/" kernels/muse_attention_decode/k1_v8f.py > kernels/muse_attention_decode/k1_v8f_pp.py'
tools/box/box.sh timed 'python3 kernels/muse_attention_decode/k1_v3_bench.py --file kernels/muse_attention_decode/k1_v8f_pp.py --shapes 9,11 --variants "p7=stages:7;q7=stages:17;v7=stages:3;r1=r1" > box_out/k1v8f_pp_bench.log 2>&1; cat box_out/k1v8f_pp_bench.log'
```

Fallback for step 1 if `sass_gate.py` cannot run: `--keep box_out/k1v8f_sass` leaves `k1_v8f.cu`, `command.txt`,
`nvcc.log` and `sass.txt`. By hand:
- write `#include <torch/types.h>` followed by CUDA_SRC into `k1_v8f.cu`;
- compile with `nvcc -cubin -O3 -lineinfo -Xptxas=-v -gencode=arch=compute_90a,code=sm_90a -std=c++20
  -DTORCH_API_INCLUDE_EXTENSION_H --expt-relaxed-constexpr -isystem <torch include_paths()> -isystem <python include>
  -o k1_v8f.cubin k1_v8f.cu`;
- `grep -E 'C75[0-9][0-9]|spill' nvcc.log`;
- `cuobjdump -sass k1_v8f.cubin`, then read the `k1_ppILi7ELi0E` / `ILi7ELi1E` functions for `QGMMA` /
  `WARPGROUP.DEPBAR` runs.

Optional calibration, compile-only and 1 min: dump the stock sgl FA3 hdim-64 FP8 kernel
(`cuobjdump -sass -fun <mangled name in box_out/k1wg_stock_fa3_fp8_evidence.txt> <sgl_kernel flash_ops .so>`).
Its mainloop should show QGMMA runs without a DEPBAR in between, which is the pattern this gate requires.

### 7.5 Emulation (laptop, `k1_v8f_emulate.py`, numpy only)

What the emulator checks:
- the consumer step order at register level (S/O/P fragment maps, masks, branchless rescale, first and last tile,
  ntl = 0 … many, both variants, ST 6/7);
- an async wgmma model, with these failure modes:
  - an operand modified in flight is a hazard;
  - a read of an in-flight accumulator is NaN;
  - a release while an op still reads the slot is a fault;
- a ptxas model: a non-wgmma definition of a wgmma input inside an open pipeline stage, or a stage open at a loop
  back edge, means "would serialize";
- the producer/consumer ring parity under random interleavings.

The comparisons are against two references:
- a same-rounding reference (E4M3 P at the per-tile running max: tight);
- the exact softmax (judge-level).

Negative controls must fail.

### 7.6 Laptop results (2026-10-25)

**Files:**
- `k1_v8f.py`: kernel. Only the k1_pp consumer loop, `mbar_wait` and the host stage dispatch differ from k1_v8.
  CPP_SRC, the routes and `K1_STAGES_T` are AST-identical to k1_v8.
- `k1_v8f_emulate.py`: emulator, numpy only. The laptop has no torch, CUDA or C++ compiler, so nothing was compiled
  here. The first compile is the pod's step 1.
- `k1_v8f_lab.py`: pod driver.

**Emulation** (`python -I k1_v8f_emulate.py`, 21 s): **PASS**.
- Layout check: f_pack bytes match the PV A fragment (row, permuted key), and the S, O (n72) and A maps cover each
  element once.
- The v8f rolled step's register-access trace is identical to FA3 `fwd_step`'s (12 events: issue QK, commit, issue
  PV, commit, wait 1, r/w S, wait 0, r S, w P, r/w O).
- **68/68 runs**: 17 cases × ST 6/7 × rolled/unrolled.
  - Inputs: seq 0, 1, 20, 63, 64, 65, 130, 200, 300, 700, 1000, 1500 full and 2047, 2100, 2500, 3000, 5000 sliding;
    1–5 splits, so ntl = 0, 1, 2, 4, 5, 6, …, 32; distributions normal / rising (the max moves on every tile) /
    falling (a = 1 on every later tile) / edges / soft; kv_scale 1 and 1/64; ping-pong accounting on two of the four
    combinations.
  - Zero hazards and zero ptxas-model flags.
  - Same-rounding reference: worst 2.3e-7, against a tolerance of 2e-5.
  - Exact softmax after the split merge: worst row 3.9e-2 and rel L2 0.4–2.5e-2 (the E4M3-P level).
  - Key coverage is exact.
- **Ring:** 360 control-flow shapes (ntl 0–20, 37, 60 × ST × last × variant × ping-pong) × 12 random interleavings.
  There was no deadlock, every pass was at the exact phase, every slot was refilled only after both releases, and
  every named-barrier instance was one sync + one arrive.
- **Negative controls: 7/7 fail as required.**

  | control | result |
  |---|---|
  | NC1: v8's order | 80 C7513-model flags with correct numerics: reproduces the box |
  | NC2: pack/rescale before wait_group 0 | 96 hazards |
  | NC3: stale a | tight error 4.8 |
  | NC4: release before PV retires | 64 hazards |
  | NC5: P-pack byte swap | tight error 1.2 |
  | NC6: missing release | ring deadlock |
  | NC7: flipped full parity | ABA caught |

**What the emulator cannot show:**
- Whether ptxas (CUDA 13.0) accepts the order. A stricter reading than the one modelled (any non-wgmma access to a
  register of the open stage) would flag FA3's own fwd_step in the same way, because softmax writes S after
  wait_group 1 while PV is in flight. The two traces are equal, so under ptxas v8f is exactly as safe as FA3, no
  more.
- The timing: QK latency is now exposed (§7.2).

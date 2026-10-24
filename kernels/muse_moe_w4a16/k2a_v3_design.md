# K2A v3 design: pacing-first small-T MoE (bs 1–32), warp-specialized bulk-copy ring + stream-K + PDL

Scope: the four small-T buckets (bs 1 / 8 / 16 / 32, M ≤ 48). bs 128 / 256 stay on k2a_v22 (hot-expert re-reads are
their problem, not pacing). Per-bucket self-test + fallback to v2.2 (then the incumbent), as in v2.2.

## 1. Why (evidence)
- v2.2 streams 1.5–2.4 TB/s in situ; its consumer alone sustains 3.9 TB/s (consumer probe), so 35–45% is structure:
  per-warp plain loads at 64 regs, per-slice `__syncthreads` + X staging, unit tails, 3 launches / grid barrier.
- Other teams: decode+mma with smem-resident tiles = 5.3–6 TB/s-eq; 18 KB bulk stages 4.42 TB/s vs 9 KB 1.64;
  structure 1 producer + 8 consumer warps, 2 CTAs/SM, 16 KB W + 2 KB S stages, one launch for all experts.

## 2. Layout (lossless re-tiling of v2.2's repack; same words, same scale bytes, same numerics)
v2.2: `WQ[e][tile16][kb64][lane][16 B]`, `SQ[e][tile16][kb/2][lane][4 B]`.
v3 stage = (expert e, row block rb = 8 tiles = 128 prepacked rows, k-stage s = 256 k = 4 kb), 18,432 B contiguous:
- bytes [0, 16384): `[w 8][i 4][lane 32][16 B]` = `WQ[e][8rb+w][4s+i]`
- bytes [16384, 18432): `[w 8][kp 2][lane 32][4 B]` = `SQ[e][8rb+w][2s+kp]`
`WS13[e][64 rb][16 s][18432 B]`, `WS2[e][32 rb][16 s][18432 B]`. One `cp.async.bulk` (18 KB) per stage.
W13 tiles keep v2.2's gate/up interleave, so a lane holds gate and up of one intermediate column.

## 3. Kernels (3 launches, PDL edges, no grid barrier, no co-residency assumption)
CTA = 288 threads: warp 8 = producer, warps 0–7 = consumers. `__launch_bounds__(288, 2)` (≤ 112 regs, no spills).
Ring: NST = 4 stages of [18 KB weights+scales | X slice 8·NC rows × 528 B (512 + 16 pad, conflict-free LDS.128)];
NC = 1: 22.7 KB/stage, 90 KB ring; NC = 2: 26.9 KB/stage, 107.5 KB ring. full/empty mbarriers per stage
(full: expect_tx; empty: 8 consumer-warp arrivals). Bounded waits set an error flag, never trap.

Routing: every CTA, every kernel, deterministic (v2.2's route_prologue, P_MAX = 8·48): compact row c per pair,
per-expert cnt/off, passes np_e = ceil(cnt / 8NC). CTA 0 of K1 writes `meta` for K3.

Work = global stage sequence: for e (cnt > 0), rb, pass j: 16 stages. **Stream-K**: CTA c takes stages
[⌊cS/G⌋, ⌊(c+1)S/G⌋) — balance to one 18 KB stage, independent of how many experts are touched (bs1 has 166 W13
items for 264 CTAs; item-granular scheduling would idle 37% of CTAs).

- **K1 (route + W13 + SwiGLU)**, normal launch. `griddepcontrol.launch_dependents` at entry. Zeroes y rows [0,R).
  Producer: per stage, lane 0 `arrive.expect_tx(18432 + ntok·512)` + bulk copy of weights; lanes 1..ntok one 512 B
  bulk copy each of x[token][256s .. +256) into the X slot (rows ≥ ntok untouched: they only feed discarded columns).
  Consumers: warp w = tile w of the block; per stage 4 × (LDS.128 weights, LDS.32 scales per kb pair, deq4 ×4,
  bmul2 ×16, per chunk 2 × LDS.128 X + 4 mma.m16n8k16) — v2.2's consume_item, operands from smem.
  Flush: item fully owned → inter = bf16(SwiGLU(acc·gs)) written straight into `interS[s][c][264]` (stage-sliced,
  padded, so K2's X slice is ONE contiguous copy of ntok × 528 B). Partial → `red.add.f32` into h[c][8192] (zero
  invariant), `__threadfence`, `atomicAdd(cnt[e][rb][j], stages_done)`; the contributor that brings it to 16 does
  the SwiGLU for the block's 64 columns × ntok, writes interS, re-zeroes h and the counter.
- **K2 (route + W2)**, PDL launch. `launch_dependents` at entry. Routing, then the producer issues the first NST
  weight copies (weights don't depend on K1), then `griddepcontrol.wait` + `fence.proxy.async`, then the X copies
  from interS. Flush: owned → store y; partial → `red.add.f32` y (zeroed by K1).
- **K3 (post-norm + weighted sum)**, PDL launch, `griddepcontrol.wait` first: v2.2's fin_phase verbatim (reads meta,
  y), without the h zeroing (h invariant is K1's).

Numerics: identical to v2.2 (same decode, exact bf16 dequant, FP32 accumulate, gs on FP32, SwiGLU in FP32 rounded
once to bf16, y rounded to bf16 before (1+w) / RMSNorm). Split-K changes FP32 summation order only (as v2.2 KS>1).

NC per bucket (ROUTING: max-expert tokens mean 1.0 / 4.3 / 7.5 / 14.6 at bs 1/8/16/32): NC1 for bs 1/8, NC2 for
16/32 (passes@16 at bs32 = 16.0 vs 15.65 touched). Experts above 8·NC tokens get extra passes (adjacent in the
stage order → L2 hits).

## 4. Predicted judge µs (t = MB / R + 5 µs; MB from k2a_gap_analysis §1; SGLang = judge)

| bs | MB | v2.2 judge-est | R = 3.0 | R = 3.5 | floor 4.8 | SGLang | × at 3.5 |
|---|---|---|---|---|---|---|---|
| 1 | 73 | 50.0 | 29 | 26 | 11.4 | 78.2 | 3.0 |
| 8 | 237 | 110.2 | 84 | 73 | 37.3 | 150.7 | 2.06 |
| 16 | 332 | 140.3 | 116 | 100 | 52.5 | 187.0 | 1.87 |
| 32 | 443 | 184.9 | 153 | 132 | 70.8 | 209.1 | 1.58 |
| 128 | v2.2 | 313.8 | 314 | 314 | 101.3 | 465.4 | 1.48 |
| 256 | v2.2 | 442.5 | 443 | 443 | 108.9 | 622.9 | 1.41 |

Geomean: R 3.0 → ~1.68×, R 3.5 → ~1.83× (v2.2 judge-est 1.37×). +22% / +34%. MoE 2.0 additionally needs the
bs128/256 hot-expert work (later).

## 5. Gates
- Lab (box, ≤ 20 min): K1 alone, real weights, judge-like routing at bs8 / bs32, probe = math off / decode only /
  full. W13 bytes / (graph µs − replay floor). Need ≥ 3.5 TB/s math-off and ≥ 3.0 math-on, else stop.
- Build: judge clone v3 vs v2.2 (`--baseline k2a_v22.py`) same session; kill if < +15%.

## 6. As built (deviations from the plan above; results in k2a_v3.md)
- Scope widened to all six buckets: M3 = 320, NC2 at 3 stages for bs 128/256 (beats v2.2 by ~15% there).
  bs 1–32 use NC1 (NC2 was no better at bs16/32).
- K1 has a 10th warp (320 threads): the epilogue warp does split-item counters + SwiGLU; consumers only `bar.arrive`.
- Weight copies carry an L2::evict_first policy.
- Each CTA processes its split tail item first (stream-K partner contributions land early).
- Judge clone: v3 / v2.2 = 1.245×, v3 / SGLang = 1.823× (box).

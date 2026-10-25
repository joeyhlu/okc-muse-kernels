# G2 — glue fusion at T <= 8 on top of k3_v17 (2026-10-25)

Source of truth for the numbers: `box_out/lb17_analysis.md` (fresh nsys trace of k3_v17, job a2c9601a, shapes 0/1/6/7/2).

## Measured critical path (k3_v17, first replay, µs from the first kernel)

| shape | P1 o / cuBLAS o | attn bnd | pooler ‖ router (side) | join + tail | K2A | unpool | ffn bnd |
|---|---|---|---|---|---|---|---|
| decode 1 sliding (T=1) | 21.7–28.5 (gap 0.6) | 28.6–31.3 | 31.7–48.0 ‖ 31.9–48.6 | gap 1.2, tail 49.2–54.0 | 54.5–86.0 | r1b 86.4–91.3 | 91.6–94.4 |
| verify 1 sliding (T=8) | cuBLAS 23.3–29.7 | 29.9–32.9 | 33.3–50.0 ‖ **46.3–55.7** | gap **6.4**, tail 56.4–61.1 | 61.6–125.5 | P1 126.0–133.1 | 133.2–136.1 |

Where the glue µs are (T=1 / T=8): router + pooler + join + tail = 22.3 / 27.8 µs for ~58 MB of weights (~17.7 µs at
the rate the pooler alone reaches). At T=8 the FP32 router (`router32<8>`, 256 CTAs) cannot get an SM until the P1 pooler's
256 x 512-thread CTAs (2/SM, all registers) retire: it starts 13 µs late and the tail waits 6.4 µs. The tail (routing +
pre-expert norm + local ids) is a 4.8 µs one-CTA-per-token kernel after a stream join. Boundaries are 2.7–3.0 µs each
(SGLang's cluster kernel, PDL-launched, already the algebraic one-reduction form); P1 launches are not PDL (gaps 0.4–0.6).

## Fusions (each behind its own call-time switch, self-tested against the unfused path at prepare, fallback = k3_v17)

**G2-A `G2_PRT` — pooler + FP32 router + tail in ONE kernel (T <= 8; replaces router32 on the side stream, the stream
fork/join, the P1 pooler launch and the tail kernel).** Grid = the P1 pooler's 256 row-tile CTAs x 16 warps. Each CTA
also owns router32's CTA task (bx = c % 32, K slice sl = c / 32): its 8 router rows x 1024-wide K slice of FP32 weights
are cp.async'd into smem *before* the PDL wait (prepare-time data; the 8.4 MB stream overlaps the boundary kernel), the
router input slice after the wait; the pooler part is p1s_kernel<1, 0>'s code (same split, same epilogue); then warps
0–7 compute router32's dot products (same lane / fma order) into the same [8, T, 256] partials. A ticket counter picks
the last CTA, which runs tail_kernel<false, 8>'s math for every token (thread tx owns elements 8tx..8tx+7, warp t routes
token t) and resets the counter (no spin, no co-residency assumption). Every value is computed by the same expression
in the same order as today -> expected bit-identical xn / topk weights / ids / local ids. Launched with PDL.
Expected (haircut 0.5 on the model): T=1 −1 µs, T=8 −3.5 µs.

**G2-B `G2_P1_PDL` — P1 launches with programmatic stream serialization (per projection).** p1s_kernel already issues
its first weight loads before griddepcontrol.wait and reads x / writes out only after it (audited); the layer launched
it with pdl = 0. On for o (after the FA4 v5 combine) and unpooler (after K2A K3); pooler only when G2-A is off and it is
not racing the side-stream router (default off). Expected −0.5..1 µs per launch.

**G2-C `G2_FFN` (+ `G2_FFN_T1`) — P1 unpooler + FFN boundary in one kernel (T <= 8).** The unpooler at T <= 8 is only this
GPU's 1024 rows = 64 P1 row-tile CTAs (16 warps), so a grid barrier is cheap and co-residency is trivial (checked at
prepare). Each CTA owns 128 of the 8192 boundary columns (its 16 branch columns + 112 stream-only columns, where the branch
is zero), computes the cluster kernel's four moment sums (branch², residual², cross, scaled²) for them, stores the CTA
partial, fences, arrives on a monotonic counter (+64 per call, bounded spin with a timeout flag, no trap), then every CTA
sums the 64 partials in CTA order (deterministic, identical in every CTA) and writes new stream + next attention input for
its columns with the cluster kernel's expressions (post rstd, algebraic pre rstd, bf16 normed, fma residual). Removes the
2.8–2.9 µs boundary kernel; costs one barrier (~1 µs). At T = 1 it also moves the unpooler from R1B to P1 (0.86x there),
so `G2_FFN_T1` is its own arm. Expected T=8 −1 µs, T=1 ~−0.4 µs.

Not built (measured reasoning): the attention-side boundary in the o-proj epilogue needs a 512-CTA row reduction at T=1
(o = 8192 rows): atomic arrive + flag + re-read ≈ 2–3 µs, the same as the 2.7 µs cluster kernel it would replace.
Pre-expert norm into K2A's X staging: a change inside K2A's bulk-copy pipeline, out of the time box.

## Numerics
All FP32 (router dot, norms, sigmoid) as today; the pooler output stays BF16 and the norm input is that BF16 tensor; the
FP32 router input at T <= 8 is unchanged. Self-test: T in {1, 2, 5, 8} x local_start in {0, 64, 224}, fused vs
(router32 + P1 pooler + tail): xn / tw max rel diff, ids / local ids equal, ticket counter back at 0, repeat identical.

## A/B (tools/box/ab_g2.json): ref = all G2 off (= k3_v17), prt, pdl, all on. Gate all-on >= +1.0% load-geomean, kill < +0.4%.

## Results (2026-10-25, serverless H200; box_out/g2_ab1..5.log, k3g2*_judge.log, lbg2_analysis.md)

Correctness: G2-A bit-identical to the three-kernel path (xn, weights, ids, local ids; 6 cases x 9 launch combos); G2-C stream
bit-identical, next input <= 3.8e-6 rel; G2-B pdl 1 == pdl 0 bitwise; layer on vs off (eager + graph) <= 7.6e-6 rel, ids
equal, judge checks pass at shapes 0/1/6/7. Cold prepare 229 s on a fresh worker (G2 build 44 s); warm 15 s.

| arm (paired A/B vs k3_v17 path, shapes 0/1/6/7) | dec1 sliding | dec1 full | ver1 sliding | ver1 full |
|---|---|---|---|---|
| G2-B P1 PDL (o, unpooler) | +0.7..1.4% | +0.2..0.7% | +0.0..0.4% | -0.1..+0.6% |
| G2-A fused, T=1 only | +1.1..1.5% | +0.4..1.6% | — | — |
| G2-A fused at T=8 (smem weights / register weights) | — | — | -0.4% / -0.2% | -0.3..-0.8% / +0.2% |
| G2-A router-first + fused pooler/tail ('main') | -1.4% | -1.3% | -1.4% | -1.5% |
| G2-C FFN boundary in unpooler (T=8 / with T=1) | -1.1% at T=1 | -1.1% at T=1 | -0.8% | -1.0% |
| router side_first at T=8 (existing lever) | — | — | +0.5..1.0% | +0.2..0.6% |
| final = G2-A@T1 + G2-B + side_first@T8 (12 seeds) | +1.6% | +1.6% | +0.2% | +1.0% |

Final file judge clone: 13/13, x1.7779 (k3_v17 x1.7757; the G2-A@T1 + G2-B clone without side_first x1.7807) -> about
+0.3-0.4% load-geomean, below the +1.0% gate and at the +0.4% kill line.

Why the model missed: the fused kernels removed launches but their in-kernel serial parts cost what the launches did. The
last-CTA tail (one SM, all tokens, after the slowest pooler CTA) is ~1 µs slower than the separate one-CTA-per-token tail
even with a redux.sync top-8; the 64-CTA grid barrier + partial exchange in G2-C costs ~3-4 µs vs the 2.9 µs cluster kernel;
at T=8 the fused router+pooler kernel (31.7 µs in the trace) is slower than pooler || delayed router (side_first recovers
part of that delay for free). Launch gaps at T <= 8 were already 0.2-0.6 µs (most kernels PDL-chained).

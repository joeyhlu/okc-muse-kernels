# K2A v3 results (H200 box, judge image, 2026-10-24)

File: `k2a_v3.py` (`# nonce k2a-v3-1`, 108 KB). Design: `k2a_v3_design.md`. Every bucket runs v3 when its prepare()
self-test passes (2 calls per case, 8 cases incl. bs 320 and bs 128 skewed); a failing bucket builds v2.2 (unchanged
code, same file) and then falls back to the incumbent. All six buckets passed on the box, so v2.2 is not built.

## Judge clone (`tools/judge/judge.py`, same session)

| shape | v2.2 µs (baseline run) | v3 µs | v3 / v2.2 (paired) | SGLang µs | v3 vs SGLang |
|---|---|---|---|---|---|
| decode 1 (bs 1) | 46.69 | 35.08 | 1.331 | 82.78 | 2.274 |
| decode 16 | 133.09 | 103.79 | 1.282 | 189.76 | 1.820 |
| decode 32 | 171.28 | 139.04 | 1.232 | 203.97 | 1.466 |
| verify 1 (bs 8) | 101.10 | 77.31 | 1.308 | 146.34 | 1.931 |
| verify 16 (bs 128) | 285.97 | 246.98 | 1.158 | 451.26 | 1.794 |
| verify 32 (bs 256) | 403.70 | 344.54 | 1.172 | 611.23 | 1.746 |

- **v3 vs v2.2: 1.245× [1.227, 1.264]** (`box_out/k3_judge_vs_v22.log`, `--baseline k2a_v22.py`). Gate was +15%: passed.
- **v3 vs SGLang: 1.823× [1.796, 1.850]** (`box_out/k3_judge.log`). v2.2 was 1.471× on the box. The box reads
  ~+7–9% high for MoE (lessons §7), so the real judge estimate is ~1.67–1.70.
- All 5 judge tests pass (rel-L2 3.76–3.79e-3, same as v2.2).
- memcheck: 0 errors on the full judge flow (`box_out/k3_memcheck.log`, caching allocator; the no-caching allocator
  breaks the harness's graph capture) and 0 errors eager with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` at bs 1/4/8/16/33/
  128 skewed/257/320, both variants (`box_out/k3_memcheck_eager.log`).

## Pacing lab (stage B; `tools/box/k3_lab.py`, `k3_lab2.py`; logs `box_out/k3_lab*.log`)

K1 alone (route + W13), real weights, reference routing, cold L2, TB/s = touched W13 bytes / (graph µs − 5.1 µs floor):

| bs | variant | math off, no X (p21) | math off (p1/p17) | decode+mma, sink (p19) | real flush, final (p16) |
|---|---|---|---|---|---|
| 8 | NC1 | 3.54 | 3.30–3.49 | 3.08 | 3.07 |
| 32 | NC1 | 3.51 | 3.45–3.51 | 3.11 | 3.05 |
| 32 | NC2 | 3.60 | 3.58 | 3.14 | 2.99 |

- Gate (≥ 3.5 math-off, ≥ 3.0 math-on): bs8 math-off was 3.30–3.40 in the first run (3.49–3.55 with the
  L2 evict_first hint on the weight copies, which is now the default); math-on 3.03–3.15. Marginal at bs8; I continued
  because the full path was already built and correct, and the judge clone is the decisive gate.
- 1 CTA/SM gives the same K1 time at bs8 as 2 CTAs/SM, so the consumers are not throughput-bound at bs8; at bs32
  (NC2) 1 CTA/SM is much slower.
- The K1 rate includes ~4–5 µs ramp (launch, route, first-copy latency): bs1 K1 math-off is only 3.1 TB/s.

What moved the numbers (box, same process):

| change | effect |
|---|---|
| L2::evict_first cache hint on the 18 KB weight copies | −2 to −3 µs per call |
| split-item finalize (fence + counter + SwiGLU of h) moved from the 8 consumer warps to a 9th "epilogue" warp (consumers `bar.arrive`, epilogue `bar.sync` + fence + counter) | K1 −5 µs at bs1/8/32 (flush cost 5.9 → ~0 µs); full call bs8 87.6 → 82.2 |
| tail-first item order (each CTA processes its split tail item first) | ~0 (kept; harmless) |
| two independent mma accumulator chains | ~0 (kept) |
| v3 at bs 128/256 with 16 tokens/pass (NC2, 3 stages) instead of v2.2 | bs128 −15%, bs256 −15% vs v2.2; NC1 there is 10–25% slower |

Per-launch split at bs 8 (raw µs incl. 5 µs floor): K1 54.0, +K2 77.9, +K3 83.2. bs 1: 31.2 / 43.1 / 48.0.

## Where the remaining time is (next levers, not done)
- Steady-state stream is ~3.5 TB/s with math off vs 4.3 plain-load ceiling; decode+mma costs another ~12%.
- K3 (post-norm) costs 5–7 µs per call at every shape; K1 ramp ~4–5 µs.
- bs32 is the weakest shape (1.47× vs SGLang).
- bs128/256: NC2 at 3 stages (NST 4 does not fit 2 CTAs/SM with the M ≤ 320 routing lists in smem).

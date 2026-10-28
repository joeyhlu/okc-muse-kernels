# Isolated MoE results

Final status: stopped at the user's request to report back. Every pod created for this task was confirmed EXITED. Estimated combined spend: $0.93 of the $5 cap (uptime x hourly price; billing settlement may differ slightly). All 34 original source inputs still match their snapshot hashes. No original code/configuration/notes were edited, and no judge submission was made.

Best lead: 72-token jobs with the 640-thread / 112-register consumer at T=256. It saved a median 3.73 us versus the current MoE control (about 1.9% of its 201.08 us median time), won all 10 T=256 routing draws, and passed all checks. T=128 saved 1.22 us; T=32 lost 0.70 us, so a candidate should retain the existing T=32 path. This is a provisional size-specific solution, not a universal replacement or a validated layer-level improvement.

The five-seed full-layer follow-up was stopped after one router seed produced 12 shape records. Its raw results are preserved in layer_results.tgz. Do not infer an overall layer gain from this incomplete follow-up. The pre-registered broad standalone adoption gate was not met.

Mechanism: the 72-token candidate saved about 3–4 us in T=256 draws with no removed weight pass, and 8–16 us when it removed one or two repeated expert weight passes. This supports reread avoidance as part, but not all, of the gain. The extra-buffer 80-token variant did not produce a robust win and should not be selected from this batch.

All times below are standalone MoE on layer-derived routings, not full-layer speedups.

| Tokens | Arm | Base us | Median delta us | Worst delta us | Faster draws |
|---:|---|---:|---:|---:|---:|
| 32 | b2 | 78.10 | -0.08 | +0.34 | 6/10 |
| 32 | w2 | 78.10 | +1.81 | +3.07 | 0/10 |
| 32 | w2j72 | 78.10 | +0.70 | +2.99 | 3/10 |
| 32 | w2j80 | 78.10 | +1.53 | +3.41 | 1/10 |
| 32 | w80x3 | 78.10 | +0.31 | +1.86 | 3/10 |
| 128 | b2 | 153.36 | +0.12 | +1.39 | 4/10 |
| 128 | w2 | 153.36 | +2.01 | +2.99 | 0/10 |
| 128 | w2j72 | 153.36 | -1.22 | +1.01 | 9/10 |
| 128 | w2j80 | 153.36 | +1.24 | +2.75 | 0/10 |
| 128 | w80x3 | 153.36 | -0.59 | +2.06 | 9/10 |
| 256 | b2 | 201.08 | +0.17 | +0.80 | 4/10 |
| 256 | w2 | 201.08 | +0.02 | +0.77 | 4/10 |
| 256 | w2j72 | 201.08 | -3.73 | -3.01 | 10/10 |
| 256 | w2j80 | 201.08 | -1.90 | +2.83 | 6/10 |
| 256 | w80x3 | 201.08 | -2.85 | +2.43 | 7/10 |

Correctness: 30/30 routing cases passed.
Identical-arm control invalid sizes: [].
Pre-registered broad winner: None.

Mechanism isolation: w2 changes register/producer allocation only; w2j72/80 also widen jobs. w80x3 exchanges one weight-buffer slot for one activation-buffer slot.

| Tokens | Max expert count, median | 64-to-80 jobs removed, median | 64-to-80 jobs removed, max |
|---:|---:|---:|---:|
| 32 | 21 | 0.0 | 0 |
| 128 | 48 | 0.0 | 0 |
| 256 | 100 | 0.5 | 2 |

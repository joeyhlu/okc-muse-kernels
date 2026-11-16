# Muse Spark Open decode kernels for H200 (Open Kernel Challenge 2026, kernel track)

Team joeyhlu's kernel-track work for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev): CUDA kernels for one
decode step of a Muse Spark Open layer (816B MoE with NVFP4 experts) on an NVIDIA H200. This is a snapshot of the code as it stood
when the kernel track closed (2026-11-05), published as one commit.

## Result

| board | file | score | judge job |
|---|---|---|---|
| decode layer (the overall board) | `kernels/muse_decode_layer/k3_tinys7_s1.py` | **2.2772** (CI 2.243-2.311, 13 / 13 checks) | `9ebfc5068bd34ed6` |

The score is the judge's speedup over the reference implementation (geometric mean over the problem's shapes). The row placed 166th
on the board at the close.

## What a decode-layer file contains

Every submitted file is self-contained Python: CUDA sources are embedded as strings and compiled in `prepare()`, and the timed call
only launches kernels. The final file covers:

- **Dense BF16 projections** (QKV, O, pooler / unpooler, router): `mma` / `wgmma` small-GEMM kernels with cp.async and bulk-copy
  weight streaming and per-shape plans.
- **MoE with NVFP4 (W4A16) experts**: a route step that builds per-expert job tables, then grouped kernels that dequantize the NVFP4
  weights in registers and run BF16 MMA with FP32 accumulation; small token counts take a one-launch path.
- **Attention decode** over the FP8 KV cache, sliding-window and full layers.
- **Glue**: fused norms and residuals, programmatic dependent launch between kernels, and startup self-tests that compare each
  custom path with the reference before it is used.

## Layout

- `kernels/<problem>/` - files submitted to each board (`*_s1.py` etc. are scored copies that differ from the checked file only in
  a nonce line), earlier generations and design notes.
- `kernels/lab/` - A/B and probe variants used to measure single changes.
- `kernels/results/` - judge result JSON by job id.
- `tools/gen_k3_*.py`, `tools/*_patches.py` - the decode-layer files are generated from a base file plus patch scripts, applied with
  verbatim slices and AST checks.
- `tools/box/layer_ab*.py`, `layer_budget.py`, `kernel_excl.py`, `route_stats.py` - paired in-layer A/B timing, per-kernel time
  budgets from Nsight Systems exports, and routing statistics.
- `side_moe_20261028/` - a small performance model of the MoE stage.

Comments in the code sometimes cite `notes/...` or `box_out/...` paths: process notes and logs that are not part of this snapshot.

## Licences

Files that include Apache-2.0 code keep its licence notices in `THIRD_PARTY_NOTICES`; the licence text is in `licenses/Apache-2.0.txt`.

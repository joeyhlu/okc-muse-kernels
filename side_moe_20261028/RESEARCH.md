# MoE research follow-up — 2026-10-28

Scope: read-only review of current work, primary-source research and a CPU accounting model. No GPU spend, no submissions, no original-code/configuration edits. Only this side directory receives new files. Predictions below are hypotheses, not measured speedups.

## What changed since our batch

The shared notes now contain later experiments. `notes/study/moe_t32plus.md` section 11 and `box_out/m32e16/summary.log` report w72 at T128+256 giving approximately **+0.145% six-load, corrected for identical controls**, including roughly +0.9% on verify32. This narrowly misses their +0.15% gate. Our standalone improvement therefore does not imply a large overall win.

The later 104-consumer / 64-producer register split improved standalone times by about 1.5/1.1/0.4 us at T256/128/32 for 64-token jobs. The 72-token version had one unexplained call-to-call error of about 7.1e-4. It is blocked pending diagnosis. Do not relax the repeatability check simply to admit it: first compare the incumbent's variability, check FP32 error and changed-input replay, and inspect counter/output resets and reduction ordering.

The shared state also records gc5's exact unpack improvement (+0.63% in-layer), with gc6 intended to combine it with w72. Avoid duplicating that integration. A multi-variant cold prepare took 417 s and disabled other paths through their time guards: **compile only the selected MoE variants in the eventual ship file**, and cold-test the complete preparation against the 300 s limit.

## Ranked actions

1. **Finish the existing integration and correctness work.** Validate w72 plus the exact unpack improvement against gc5, without extra experimental variants in prepare. Separately diagnose the c104 repeatability result. These have measured evidence and limited implementation scope; do not advertise additive gains before a paired test.
2. **Best new bounded experiment: T32, two output-row tiles sharing one activation stage.** Keep four consumer warpgroups and cap token width at 32. Each consumer computes two 64-row tiles instead of one: a 512-row CTA tile with two accumulator sets. Stage the matching activation slice once for both. This attacks activation movement and segment overhead without changing precision or creating wider token jobs.
3. **Park the three-consumer wide-job design unless a probe overcomes its traffic penalty.** It solves the register budget on paper but introduces costs large enough to erase the benefit. Details below.

## T32 double-row design

At T32, any expert receives at most 32 distinct tokens. Restrict the new kernel to that bucket; retain existing kernels elsewhere. Keep the current stream-K plan as the comparison, while defining logical output units as pairs of adjacent current units.

Candidate geometry: four compute warpgroups; 512 output rows per weight stage; K-stage 256; token capacity 32; two weight-ring slots and three activation slots. Packed weights occupy 72 KiB/stage including scales; each activation slot occupies 16 KiB. The rings total **192 KiB**, leaving some room below Hopper's 227 KiB per-block limit for metadata and synchronization. This is arithmetic feasibility only; compiler and occupancy checks remain mandatory. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

On all ten observed T32 routings, `research_model.py` calculates unchanged packed weight bytes and padded tensor work, **50% fewer activation-stage bytes** and **50% fewer logical output units**. These are traffic/work counts, not a 50% time prediction. Splitting CTA ranges can add fixup segments, so the model does not claim that every actual split/epilogue is halved.

Implementation issues to settle before spending:

- Two accumulator sets and fragment liveness must compile without spills or serialized WGMMA. The N<=32 restriction is important; do not generalize to N64 without another budget.
- A contiguous 512-row stage needs a lossless weight repack. Alternatively use the old layout with additional bulk-copy instructions; count those instructions explicitly in the probe.
- Preserve the mapping of gate/up row pairs, FP32 accumulation, BF16 SwiGLU output, phase-B readiness credits and reset logic. A larger tile is not permission to skip keys/experts or alter precision.
- Existing W2 was **two** consumer warpgroups doing 128 rows each (256 rows total); that lost. This proposal retains **four** consumers and moves 512 rows per CTA. The earlier failure is relevant caution, not a measurement of this geometry.

Cheap gate: compile-only first. Then a stage probe at N8/N16/N24/N32 with an identical control, real activation copies, exact unpack, reference checks, and warm/cold cache cases. Compare a 512-row stage with **two** incumbent 256-row stages, not one. Stop if double-row stages lose materially; do not build the full pipeline just because the register arithmetic fits. If competitive, a full T32 MoE must save at least 2 us across five router seeds with clean replay before a layer trial.

## Why three consumers are not the first build

Three consumer WGs plus one producer WG give 512 threads. A 144/64 register split uses 63,488 registers; 152/56 uses 65,536. Both fit the SM total on paper if the launch actually reserves 128 registers/thread and the compiler obeys the role layout. `setmaxnreg` draws from the **CTA's** pool, and an increase blocks until registers are available; an SM-total inequality alone is insufficient. [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)

The topology is established elsewhere: ThunderKittens' H100 attention uses three consumer WGs and one producer WG, with 160/32 register limits. That demonstrates topology feasibility, not MoE performance or transferability of its barriers/live ranges. [ThunderKittens source](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/attention/mha_h100/mha_h100.cu)

For a straightforward 192-row MoE tile, the observed T256 routings give these median changes versus 256-row / 72-token jobs:

| Token cap | Packed weight bytes | Activation-stage bytes | Logical output units |
|---:|---:|---:|---:|
| 96 | −3.16% | +35.42% | +29.11% |
| 112 | −5.21% | +33.38% | +26.39% |
| 128 | −7.29% | +33.38% | +23.62% |

The model includes padding to whole 192-row tiles (1.56% extra rows overall). At T32 and T128 it removes no jobs in this sample and simply adds traffic. The larger activation and segment counts make this a poor first bet despite attractive register availability. It also requires reworking hard-coded four-WG indexing and phase-B readiness mapping. A ragged/unpadded layout could change these costs but needs its own accounting.

## Approaches to stop repeating

- L2 pairing of split experts: later R-rev-4 already halved weight HBM traffic in the probe but saved only 2–3.5% time; unchanged on-chip ingestion limits the payoff. It is not a strong standalone MoE lever.
- More unpack/descriptor instruction reductions without a measured in-layer or energy effect. The exact L2 unpack win is already being integrated; earlier descriptor-uniformization did not improve runtime.
- Generic dynamic tail pools and greedy job splitting: measured failures or sub-microsecond modeled gains are already recorded.
- Drop-in CUTLASS mixed-input GEMM as a presumed solution: NVIDIA's example primarily targets compute-bound cases, converts the narrow operand through registers, and imposes scale/group-size constraints. Our NVFP4 block-16 scaling and small routed jobs require adaptation; INT4 support is not automatically NVFP4 support. [CUTLASS mixed-dtype documentation](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/README.md)

Recommendation: let the existing gc5/w72 and c104 work finish. If one fresh kernel idea receives the next small budget, test **T32 double-row activation reuse**, with the stated compile/probe gates. No speedup is claimed for it yet.

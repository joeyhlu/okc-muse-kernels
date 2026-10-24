# r1b: muse_bf16_proj router hi‖lo (L1) + small-m split-K GEMV (L2)

File: `kernels/muse_bf16_proj/r1b_router_gemv.py` (`# okc: problem=muse_bf16_proj`, `# nonce r1b-1`). CPU checks: `kernels/muse_bf16_proj/r1b_emulate.py`.
Nothing here has run on a GPU. It has not been submitted.

## Levers (module constants)

| constant | default | what it does |
|---|---|---|
| `R1B_ROUTER_SPLIT` | True | **L1.** `prepare()` splits the FP32 router once: `hi = bf16(w)`, `lo = bf16(w − hi)` (SGLang's `_build_split2`). Every router call the GEMV doesn't take runs as BF16×BF16 with FP32 accumulation. |
| `R1B_ROUTER_SPLIT_MODE` | `"kcat"` | `kcat` = `torch.mm(cat(x, x), [hi‖lo]ᵀ, out_dtype=fp32)` over K = 16384: one GEMM, plus one copy for x. `ncat` = one GEMM against `[hi; lo]` (N = 512), then a half-sum. `two` = SGLang's two GEMMs and the `+=`. Note that one launch can't be had inside cuBLAS: either x gets duplicated or the output gets summed. |
| `R1B_GEMV` | True | **L2.** One `load_inline` CUDA kernel, used on the shapes in `GEMV_ROUTES` (BF16 layers at m ≤ 8; router at m ≤ 32, which it runs on the **FP32 weight directly**). Every other shape goes to SGLang's `_bf16_gemm_dispatch_impl`, i.e. cuBLAS. |
| `R1B_FIXUP` | True | Split-K is reduced inside the same launch: the last CTA to arrive sums the S partials in order q = 0..S−1, then resets its counter to 0. False uses a second tiny reduce launch instead. Both orders are deterministic. |
| `R1B_SELF_TEST` | True | `prepare()` runs each route at its bucket's m values, normal and outlier x, probe rows included, and checks it against an FP64 product. It also checks that two calls give bit-identical output, which catches a counter that didn't reset. A route that fails is dropped by itself and goes to cuBLAS. Failures to build or self-test are printed, and everything falls back to cuBLAS + L1. |
| `ROUTER_GEMV_MAX_M` | 32 | Router m above this uses L1. |

**Kernel.** A CTA is 8 warps. Warp (wr, wk) owns RPW consecutive rows and every WK-th chunk of the CTA's K slice. A chunk is 32·VEC·U wide, with U = 4 16-B vectors per row per lane. Weights are loaded with `__ldcs`, which is evict-first. The first chunk is issued before x is staged, and the next chunk is prefetched while the current one is computed. x[MT][K slice] is staged in shared memory, zero-filled past m. Accumulation is FP32. The reduction is a shuffle-xor tree, then smem across WK. grid = (N / rows_cta, S, ⌈m / MT⌉). There are 6 instantiations: bf16 × MT∈{1, 8} × RPW∈{1, 2}, and fp32 × MT=8 × RPW∈{1, 2}.

**Why m ≤ 8 for BF16.** CUDA-core FP32 FMA on H200 is ~67 TFLOP/s. At m = 8, pooler needs ~8 µs of FMA issue against 14 µs of bandwidth, so it is co-limited. At m = 16 the FMAs alone exceed the bandwidth time. m ≥ 16 on the BF16 layers needs `mma.sync` (a skinny tensor-core GEMM: rung 2). The router is small (N = 256), so m = 32 is still ~2 µs of FMA. Its z-tiles re-read the 8.4 MB weight from L2.

## Route table (defaults)

| layer, bucket | (MT, RPW, WK, S) | grid | per CTA | launches |
|---|---|---|---|---|
| qkv_gate m=1 | (1,1,1,1) | 288×1 | 8 rows × 16 KiB, x 16 KiB smem | 1 |
| qkv_gate m≤8 | (8,2,1,4) | 144×4 | 16 rows × 4 KiB, x 32 KiB | 1 (fixup) |
| o m=1 | (1,2,1,1) | 512×1 | 16 rows × 2 KiB | 1 |
| o m≤8 | (8,2,1,1) | 512×1 | x 16 KiB | 1 |
| pooler m=1 | (1,1,1,1) | 512×1 | 8 rows × 16 KiB | 1 |
| pooler m≤8 | (8,2,1,4) | 256×4 | 16 rows × 4 KiB | 1 (fixup) |
| unpooler m=1 (slice 1024×4096) | (1,1,4,1) | 512×1 | 2 rows × 8 KiB, 4 warps/row | 1 |
| unpooler m≤8 (slice) | (8,2,1,2) | 64×2 | 16 rows × 4 KiB | 1 (fixup) |
| router m≤8 (FP32) | (8,1,1,8) | 32×8 | 8 rows × 4 KiB, x 16 KiB | 1 (fixup) |
| router 9..32 (FP32) | (8,1,1,8) | 32×8×⌈m/8⌉ | z-tiles re-read W from L2 | 1 (fixup) |

Dispatch for the scored and test shapes is printed by `r1b_emulate.py` §C. Every scored m=1 and m=8 shape, plus router m=16 and m=32, takes the GEMV. m = 16 and up on the BF16 layers goes to cuBLAS, and router m = 128 and 256 go to L1 kcat. Of the tests, m 7 pooler, m 5 unpooler, m 8 router and m 9 router (2 z-tiles) take the GEMV. The rest go to cuBLAS.

## Predicted µs per shape (judge-measured units)

Model: GEMV time = bytes / 3.9 TB/s + ~1.0 µs ramp (x stage, first DRAM round trip, tail), plus ~1.5 µs graph-node overhead. That overhead is the measured-minus-profile gap seen in rung 0 for single-kernel shapes, where it ranged 1.0–2.6 µs. L1 kcat = copy (~1 µs) + one cuBLAS GEMM over K = 16384 (assumed split-K + reduce, ≈ SGLang's pair minus one GEMM's launch and the add). The BF16 m=8 GEMVs are co-limited by compute and could land on top of cuBLAS. The router and unpooler-slice numbers carry the most confidence. o carries the least: cuBLAS already runs one kernel at ~3.3–4.1 TB/s there.

| load | qkv_gate | o | pooler | unpooler | router | load sum | speedup |
|---|---|---|---|---|---|---|---|
| decode 1, SGLang | 14.1 | 6.7 | 21.5 | 7.0 | 10.4 | 59.8 | |
| decode 1, r1b | 12.2 | 6.5 | 19.7 | 4.7 | 5.0 | 48.1 | 1.24× |
| decode 16, SGLang | 14.1 | 7.1 | 21.7 | 18.4 | 14.7 | 76.1 | |
| decode 16, r1b | (cuBLAS) | | | | 5.5 | 66.9 | 1.14× |
| decode 32, SGLang | 14.5 | 7.1 | 21.9 | 18.8 | 15.3 | 77.6 | |
| decode 32, r1b | (cuBLAS) | | | | 6.5 | 68.8 | 1.13× |
| verify 1 (m=8), SGLang | 14.1 | 6.8 | 21.3 | 7.3 | 26.1 | 75.6 | |
| verify 1, r1b | 13.0 | 6.8 | 20.5 | 4.9 | 4.8 | 50.0 | 1.51× |
| verify 16 (m=128), SGLang | 16.8 | 7.2 | 25.3 | 19.7 | 16.3 | 85.2 | |
| verify 16, r1b | (cuBLAS) | | | | 13.5 (L1) | 82.5 | 1.03× |
| verify 32 (m=256), SGLang | 19.5 | 9.4 | 30.9 | 24.0 | 17.5 | 101.4 | |
| verify 32, r1b | (cuBLAS) | | | | 14.5 (L1) | 98.3 | 1.03× |

**Problem speedup (geomean of the 6 loads):**
- central: **≈ 1.17×**
- L1 only, with GEMV off (router kcat ≈ 9 / 9.5 / 10.5 / 11 / 13.5 / 14.5 µs at m = 1 / 8 / 16 / 32 / 128 / 256): **≈ 1.08×**
- optimistic, with the GEMV at 4.2 TB/s and 1.3 µs overhead: **≈ 1.21×**

The board leader is at 1.45×. Getting there needs the BF16 layers at m ≥ 16 (rung 2: an `mma.sync` skinny GEMM, or a tuned cuBLASLt algo without split-K reduce).

## Validation (CPU, `python3 -I r1b_emulate.py`: 114 checks, 0 failures, 89 s)

- **L1 numerics.** The judge's own `generate_input` / `check_implementation` were run on CPU, with BF16×BF16 products accumulated in FP32. That is exact per product, so only the order of the sums differs from cuBLAS. Tested at m ∈ {1, 2, 5, 8, 9, 16, 32, 33, 128, 256, 512}, normal and outliers:
  - split bound: max |hi + lo − w| / |w| = 7.61e-6 ≤ 2⁻¹⁷.
  - kcat total rel-L2 2.3–3.7e-6 (tol 1e-4, ≥ 27× margin). ncat and two: 2.3–2.9e-6.
  - probe rows ≤ 3.0e-6 (tol 1e-3, ≥ 330× margin).
  - Router in the GEMV, FP32-direct: 1.5e-6 total, probe rows exact (0).
  - As a control, BF16-hi-only fails the check at every m, which shows the check has teeth.
- **L2 emulation.** A literal NumPy port of `gemv_kernel`'s index math covers CTA/warp/lane/chunk/u/j → (n, k), the x stage, the xor tree, the `red[]` layout, the partials and the fixup. It ran on every route at its real shapes (m = 1; 2, 5, 7, 8; router 1, 9, 16, 32):
  - every (n, k) multiplied exactly once per token tile;
  - every output written exactly once;
  - small-integer data bit-exact to x·Wᵀ;
  - real data (probes, outliers) passes the judge's checker with BF16 output rel-L2 0 to 1.7e-4 vs the BF16-rounded FP64 result, router 1.3–1.6e-7, and every probe row exact.
  - The reduce-kernel path (`R1B_FIXUP=False`) is bit-exact too.
  - `geometry()` rejects N tails, short K slices and smem > 40 KiB, so those shapes fall back to cuBLAS and are never launched.
- **Compile.** `py_compile` is OK. The CUDA source was also stub-compiled as C++17 with clang (`-Wall -Wextra`, CUDA and torch types stubbed, `<<<>>>` stripped): all 6 kernel instantiations and both reduce kernels are clean. No nvcc run.
- **Rules.** No os / sys / environ / subprocess. reference.py is never imported. There are no syncs in `kernel_fn`, which branches on shapes, the route table and `x.data_ptr() % 16` only. Buffers are fixed: a persistent zeroed counter array and per-call `torch.empty`. Weights stay at their precision: BF16 as given, the router FP32 directly in the GEMV or as the TASK-sanctioned hi + lo in L1.

## Risks

1. **No load_inline file has run on the judge yet.** Compile time (~40–90 s), nvcc and ninja in the sandbox, or the static checker could bite. A build exception falls back to cuBLAS + L1, which keeps the L1 gain. A hang past the 300 s prepare limit would not be caught.
2. **`torch.mm(..., out_dtype=fp32)`** is SGLang's own call (torch 2.14.1), so it is low risk. If kcat fails its self-test, the router uses SGLang's path.
3. **Fixup memory ordering.** This is the standard threadfence-reduction pattern: store, `__threadfence`, sync, atomicAdd; then the last CTA does sync, `__threadfence`, `__ldcg`. The self-test runs every route twice and checks bit-equality, and a bad counter would show up as wrong or stale output. `R1B_FIXUP=False` is the one-line escape. The counters are shared module state, so two graphs replaying concurrently on different streams would race. The judge runs serially.
4. **Performance unknowns.**
   - The BF16 m=8 routes (qkv, pooler, o) may not beat cuBLAS: they are FMA co-limited, and the x stage adds 25–50 % extra L2→SM traffic.
   - o at m=1 is predicted at parity.
   - Router z-tiles rely on the evict-first weight lines surviving in L2 between tiles.
   - Register use is capped at 128 by `__launch_bounds__(256, 2)`. The build log prints the `-Xptxas=-v` counts. Check them for spills on the MT=8, RPW=2 variants.
   - **Waves.** 2 CTAs/SM guaranteed = 264 slots (more if registers allow). qkv m≤8 is 576 CTAs, which is 2.18 waves with a near-empty third. If it lands near cuBLAS, try (8,1,1,8): 1152 CTAs, 1 chunk each. pooler m≤8 is 1024 CTAs, ≈3.9 waves, which is fine. The unpooler-slice m≤8 route is only 128 CTAs (under one wave), so its bytes in flight rest entirely on the 2-chunk prefetch. Alternatives there are (8,1,1,4) at 512 CTAs or (8,2,2,2) at 256 CTAs.
   - **Self-test inputs.** Each route is self-tested on dense rows (`probe="none"`, which matches the judge's m=1 test 7 times in 8), on forced-probe rows, and on outliers with the judge's probe rule. So the accumulation path is exercised at m=1, not only a single weight column.
5. **Nearby shapes** (the final review reruns them). m = 2..7 run the MT=8 kernel with wasted FMAs, but no slower than m = 8. m = 9..32 BF16 and m > 32 router are unchanged from SGLang or L1.

## First check plan

1. `okc check kernels/muse_bf16_proj/r1b_router_gemv.py`. In the log tail, read `[r1b] compiled in`, the ptxas register lines, each `self-test` line and `gemv routes live`.
2. Per shape: delete from `GEMV_ROUTES` any route that loses to SGLang, most likely o and the m=8 qkv/pooler. Try `R1B_ROUTER_SPLIT_MODE = "two"` / `"ncat"` for router m = 128 and 256 if kcat doesn't win there. Try S / WK variants for the unpooler slice and the router. Bump the nonce each time.
3. Next levers:
   - an in-`prepare()` graph-timed autotune of each route vs cuBLAS;
   - the router GEMV with `__ldg` instead of `__ldcs` when ⌈m/8⌉ > 1;
   - rung 2: an `mma.sync` skinny GEMM for the BF16 layers at m = 16–256.

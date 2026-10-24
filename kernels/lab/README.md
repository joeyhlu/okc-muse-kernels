# LAB FILE — check runs only, never submit for score

Rung-0b "lab" check files (runbook_token_day.md §2). Each `r0b_<problem>.py` is SGLang's `baseline.py` with
`kernel_fn` byte-identical to `problems/<p>/baseline.py`, so the tests pass and the timed part stays the stock kernel.
The measurements run in `prepare()` (moe, decode_layer) or at module import (attention_decode, which has no
prepare) and are printed. They come back in the last 16,000 characters of stdout (`log_tail`). Never `okc submit`
these files. They do untimed work outside the timed calls and are instrumentation only (gray-area call G1).

```
okc check muse_moe_w4a16        kernels/lab/r0b_muse_moe_w4a16.py        --no-wait --json
okc check muse_decode_layer     kernels/lab/r0b_muse_decode_layer.py     --no-wait --json
okc check muse_attention_decode kernels/lab/r0b_muse_attention_decode.py --no-wait --json
```
Each costs one check slot. A repeat needs a changed `# nonce r0b-N` on line 3, because identical code is deduplicated.

Nothing here has run on a GPU. It was checked on the Mac (see "What was verified").

## Files

| file | what it is |
|---|---|
| `microbench_cuda.py` | canonical source of the microbenchmark snippet: one `load_inline` extension (CUDA string `_MB_CUDA`, decls `_MB_CPP`) + `run_all(print_fn, parts, deadline)` |
| `r0b_muse_moe_w4a16.py` | baseline + `prepare()` = baseline prepare → Marlin `block_size_m` sweep → full microbench (dev, M1–M5) |
| `r0b_muse_decode_layer.py` | baseline + `prepare()` = baseline prepare → FA3 `num_splits` sweep → post-expert-norm Triton vs torch chain → router FP32 vs hi+lo → microbench subset (dev, M3, M5) |
| `r0b_muse_attention_decode.py` | baseline + import-time probes: (c) FusedQKNorm fused KV store → (b) FA3 `num_splits` sweep → (a) FA3 append + RoPE into the FP8 paged cache |

The block between `# ===== BEGIN microbench_cuda.py` and `# ===== END microbench_cuda.py =====` in the moe and
decode_layer files is a verbatim copy of `microbench_cuda.py`, which is verified identical with `cmp`. Edit
`microbench_cuda.py`, then paste the block again (a submission must be one file). The generic helpers
(`_lab_graph_us` …) and the attention helpers (`_lab_attn_case`, `_lab_fa3_sweep` …) are likewise copied into each
file that needs them.

The first lines of each lab file are, in order: `# okc: problem=<p>` (must stay line 1), the LAB marker, and the nonce.

Switches at the top of each lab section:
- `LAB_ENABLED` / `LAB_AT_IMPORT`: set False to get the plain baseline back.
- `LAB_PARTS` / `LAB_PROBES`: which parts run, in that order.
- `LAB_DEADLINE_S`: no new lab work starts after this many seconds of prepare/import. Prepare's own limit is 300 s.
- `LAB_MICRO_START_BY_S`: the microbench, which compiles first, starts only before this. It is 140 s for moe and
  160 s for decode_layer.
- `LAB_MICRO_DEADLINE_S`: 265 s. Microbench parts may still start until this, after the compile. Worst case is the
  start-by time + ~90 s compile + ~10 s of parts, which stays ≤ ~260 s, under the 300 s prepare limit.
- `LAB_REPS`: graph replays per point.

**Read the `===== <problem> LAB SUMMARY =====` block first.** It is printed last, at the end of prepare (or of
the import probes). It repeats the decision-relevant lines in compact form, in case the front of our output falls
out of the 16,000-char tail:
- MoE: per shape, `auto` vs the best `block_size_m`;
- FA3: per shape, `base` vs the best `num_splits`, plus `meta[:4]`;
- PEN and ROUTER rows;
- ATTN(a)/(c) results;
- `SUM M1…M5`: latencies, M2 points, M3 max and knee, M4 best GB/s per row size, M5 values/clk/SM with check status
  and the INT peaks.

All lab work is in `try/except` and prints `... ERR <type>: <msg>` instead of raising. The one thing a `try` cannot
contain is a CUDA fault (an illegal address poisons the context, and then the run fails). That is why the risky
FA3 append probe runs last, and why every result line is printed with `flush=True` before the next step.

## How every timing is taken

`_lab_graph_us(fn)` works like this:
1. Warm `fn` twice on a side stream.
2. Capture it in a `torch.cuda.CUDAGraph`.
3. Replay it `LAB_REPS` times. Before each replay, a 256 MiB `flush.sum()` read evicts the 50 MB L2.
4. Time each replay alone with CUDA events and report the median.

The numbers are raw: the ~4 µs replay cost is not subtracted. Each file first prints `1-kernel graph replay: median
X us`. Subtract X to compare with the judge's numbers.

All inputs are built by copying the task's `generate_input` logic into the file (it is never imported). The copies
are the MoE `normal`/`busiest` branch, and the attention/layer step metadata with page size 1, the same seeds as
`meta.json`, and an 80K context. One fixed input is used per shape, while the judge draws a fresh input every
repetition. So routing-dependent MoE times are for one routing.

## What each file prints and how to read it

### moe (`r0b_muse_moe_w4a16.py`) — expected ~90–150 s of prepare, plus any Marlin JIT compile
- `LAB ... baseline prepare took X s`.
- **`MOE` table**: one row per scored shape (bs 1, 16, 32, 8, 128, 256). The columns are:
  - `rts`/`dst`: local routes and distinct local experts for this routing.
  - `first_s`: wall time of the first eager `kernel_fn` call. On the first row this includes the Marlin MoE JIT
    compile if it is not cached in the image. That number matters for the 15-min budget.
  - `auto`: the stock `kernel_fn`.
  - `bsm8 … bsm64`: `kernel_fn` with `block_size_m` forced. `*` marks the stock pick and `!` marks an output that
    differs from stock by rel-L2 > 1e-2.
  - `best gain`: stock time divided by the best config.

  A gain outside ~±3% (the judge's interval) on any shape is a rung-1 lever: a per-shape `block_size_m` table.
  The `max rel-L2 diff` line confirms the forced configs compute the same thing. Built-in consistency check: the
  `auto` column and the `*`-marked `bsm` cell run the same configuration, so they must agree within noise (a few %).
  If they don't, distrust the table: it means replay noise or a stale-L2 effect.
- **Microbench** (sections below), all parts: dev, M1, M2, M3, M4, M5.

### decode_layer (`r0b_muse_decode_layer.py`) — expected ~80–130 s of prepare
- **`FA3L` table**: one row per scored attention shape (decode/verify × batch 1/16/32 × sliding/full). The cells
  are:
  - `base`: exactly the baseline's FA3 call (`num_splits=0`, plus the given `scheduler_metadata` on full decode).
  - `s0nm`: `num_splits=0` without metadata (full decode only). `s0nm − base` is the cost of FA3's own prepare
    kernel, which the shared metadata saves.
  - `s1 … s16`: forced `num_splits`, with FA3 building its own consistent schedule.
  - `best … x`: the gain over base.
  - `meta[:4]`: the first ints of the metadata buffer, which should be the per-batch dynamic splits FA3 actually
    uses. This settles study §D.1's UNVERIFIED split counts (expected 57 at full decode B=1 and 2 at B=32).

  The variant "forced N + metadata built for N=0" is deliberately not run. Its dynamic splits can exceed the
  N-sized accumulators, which is an out-of-bounds risk. A real tuned-splits submission must call
  `get_scheduler_metadata(num_splits=N)` itself.
- **`PEN` rows**, T = 1, 8, 16. The stock `fused_post_expert_norm_weighted_sum` (a ~10-kernel torch chain below 32
  tokens) is compared with calling its `_post_expert_norm_weighted_sum_kernel` Triton kernel directly. Each row
  shows µs and the error vs an FP64 chain (rel-L2 / max-abs), plus `triton-vs-chain`. If Triton is faster with equal
  error, that is a rung-1 lever.
- **`ROUTER` rows**, T = 1, 8. The stock FP32 `F.linear` is compared with the BF16 hi+lo pair, which the rules allow
  at any T. The rows show µs and errors vs FP64. `bf16in` is the error relative to the BF16-rounded input, which is
  what hi+lo can reach at best. The line also prints `allow_tf32`.
- **Microbench subset**: dev, M3, M5. There is no M1 here, so M3's `Little` lines (which need M1's idle latency)
  do not print in this file. Read them from the moe run.

### attention_decode (`r0b_muse_attention_decode.py`) — expected ~30–90 s at import (most of it baseline JIT warm-up)
- **`ATTN(c)`**, sliding and full, B=2, L=8, ctx 3000. FusedQKNorm is called with `kv_store=` (norm + RoPE + FP8 KV
  store + FP8 q in one Triton kernel) and compared with the baseline's qk_norm + `store_cache` + q cast. The line
  gives:
  - the fraction of identical q bytes and K/V cache-row bytes, plus rel-L2;
  - whether any `cache_loc == 0` occurred. The fused kernel skips slot 0, but the judge treats row 0 as live: a
    correctness trap if we adopt it.

  All-equal bytes mean the fusion is a free rung-1 win: two fewer kernels per sliding/full layer.
- **`FA3A` table**: the same `num_splits` sweep as `FA3L`, on the attention problem's seeds.
- **`ATTN(a)`**: FA3 with `k=`/`v=`, which appends into the paged FP8 cache (page 1) inside the attention kernel.
  Cases:
  - full layer: append only;
  - sliding layer: (i) RoPE applied by qk_norm, then append; (ii) FA3 in-kernel RoPE with bf16 cos/sin, expected to
    be refused because FA3 requires cos/sin to have q's dtype; (iii) FA3 RoPE with FP8 cos/sin, which upstream FA3's
    own tests use.

  Each line prints `OK` with output rel-L2 vs the baseline and new-K/V-row rel-L2, or `REFUSED/ERR <msg>`.
  `This flash attention build does not support appending KV` means the sgl-attn build has APPENDKV disabled.
  Bounds to compare against: output 3–15% (2.5× the FP8-query error), cache 8%.

## Microbench sections (from `microbench_cuda.py`)
- `DEV`:
  - torch device properties: name, SMs, L2, plus whatever clock fields this torch exposes;
  - the NVML clock, if pynvml is present;
  - `DEV attrs`, from `cudaDeviceGetAttribute` by raw enum value: SM clock kHz, SMs, memory clock kHz, bus width,
    L2 bytes, threads/SM, smem/SM, regs/SM, smem/block, opt-in smem/block;
  - the nominal HBM bandwidth implied by memory clock × bus width.
- `MB arch flags`: the `-gencode` flags torch will use. `MB compile X s` is the nvcc time for the one extension;
  the target is < 60 s.
- **M1**, idle latency. One thread chases `p = buf[p]` with `ld.global.cg` through a random single-cycle permutation
  of 128 B slots, over working sets of 4 MiB, 24 MiB (L2), 256 MiB, 1 GiB and 2 GiB (DRAM; the larger sets add TLB
  misses). Each line prints ns/hop from `%globaltimer`, cycles/hop from `clock64`, and the SM clock (cycles/ns)
  during a memory-bound kernel. `lat_ns` (the minimum over the ≥ 256 MiB sets) fills the study §A.8 "H200 measured
  idle" row.
- **M2**, latency under load. The same chase runs in block 0 while `bps × SMs` blocks stream a different 1 GiB
  region (16 B `ld.global.cs`, U loads in flight per thread). Configurations: 0, 4, 8, 16, 32 and 64 KiB in flight
  per SM. Each line prints the achieved GB/s, the loaded chase latency, and BW×latency/SM, which is Little's law at
  that operating point. The knee fills the "measured at 90% of peak" row. The chaser shares an SM with streamers,
  so its latency includes that SM's own queueing, which is what a real kernel sees.

  M2's streamers use `ld.global.cs` (evict-first) and their GB/s is averaged over the whole kernel. It is the
  x-axis of the latency-vs-load curve, not M3's bandwidth curve (M3 uses `ld.global.nc`).
- **M3**, bandwidth vs bytes in flight. A grid-stride read of 1 GiB (`ld.global.nc` via `__ldg`) with vector width
  {4, 8, 16} B × unroll {1, 2, 4, 8} × blocks/SM {1, 2, 4, 8}, 256 threads per block.
  - Each cell is `b<bps> GB/s(KiB in flight per SM)`. `b8>4` means occupancy capped the requested blocks/SM at 4,
    and the in-flight figure uses the real value.
  - The summary gives max GB/s, the smallest in-flight/SM reaching 80/90/95% of max, and the implied loaded latency
    (in-flight × SMs / BW).
  - The `Little` lines give the in-flight/SM needed for 4.2 TB/s and for 90% of max at M1's idle latency.
  - This is the design constant for the MoE and attention kernels (study §A.8).
- **M4**, random row gather. 256 MiB of rows at random row indices (a no-duplicate permutation) from a 1 GiB table:
  - 64 B rows use 4 threads per row (8 rows per warp request); 128 B rows use 8 threads (4 rows per warp request);
  - 1024 threads per SM, with U independent rows in flight per thread group;
  - the index load (4 B/row) is included, as with FA3's page-table-then-gather;
  - a `seq` control uses sorted indices.

  Each cell is `GB/s of row bytes, Mrow/s`. If 64 B rows reach much lower GB/s than 128 B rows, the full-attention
  kernel should load both KV heads' 128 B row per key (study §D.4 option a). Compare against M3's max.
- **M5**, E2M1 → BF16 conversion throughput, in values per clock per SM. Each thread converts 2 LCG-generated words
  (16 values) per iteration and feeds each bf16x2 into an HFMA2 accumulator, so nothing is dead-code-eliminated
  (that runs on the FMA pipe, not the INT pipe). Occupancy is the maximum the API reports. Variants:
  - `overhead`: LCG + HFMA2 + loop only, the ceiling of this harness;
  - `marlin-lop3`: exactly `marlin/dequant.h:393-407`, applied as `marlin_template.h:1262-1276` does;
  - `prmt-lut`: two 8-entry byte LUTs (hi/lo byte of the bf16) via PRMT on masked magnitudes, PRMT interleave,
    then a sign merge from a repacked sign-slot layout (lossless offline repack; 17 ops per 8 values counted by
    hand);
  - `smem-lut256`: one 256-entry bf16x2 table indexed by byte, so random banks give conflicts;
  - `smem-lut-lane`: the same table replicated per lane (entry×32 + lane), conflict-free, 32 KiB/block.

  `check=ok` means the GPU output matched the expected BF16 bit patterns on 16,384 random words plus all byte
  patterns. `peak LOP3/PRMT` lines are 8 independent chains each: the INT-pipe ops/clk/SM, which tests the
  "~64/clk/SM" recalled in study §A.9. The `need at 4.2/4.8 TB/s` lines give the values/clk/SM a kernel must
  convert to keep up with HBM at the measured clock (~31 at 4.2 TB/s). A variant below that number means the MoE
  kernel is ALU-bound with that conversion (study §0 item 2 / open question 1).

## Expected runtime and output size
| file | lab time | of which compile | printed |
|---|---|---|---|
| moe | ~90–150 s (+ Marlin JIT if uncached; that compile would happen in the tests anyway) | microbench nvcc ~30–60 s | ~7 KB |
| decode_layer | ~80–130 s, plus baseline prepare | microbench nvcc ~30–60 s | ~6 KB |
| attention_decode | ~30–90 s at import | none of ours; baseline JIT kernels warm up here instead of in the tests | ~4 KB |

Each part's time is printed (`M3 done in X s`, `LAB fa3 finished at X s`). The whole run stays well inside 15
min: the lab adds at most ~4 min, and the deadlines enforce it.

**Tail risk.** All our output is printed before the harness runs tests and timing. If the harness prints more than
~9 KB after prepare, the start of our output falls out of the 16,000-char tail. In that case, re-run with fewer
`LAB_PARTS` (and a new nonce). The r0 run's `log_tail` shows how chatty the harness is.

## Compile-risk spots and other places I was unsure (nothing here has run on a GPU)
1. **The whole extension is one compile.** If it fails, the log shows `MB BUILD FAILED` and the nvcc error tail
   (up to 6000 chars). The sweeps still run, because they do not use it.
   - Checked on the Mac: the CUDA string was syntax- and type-checked as host C++ with clang++ (stub headers, `<<<>>>`
     stripped, asm stubbed), with 0 errors. The `_MB_CPP` declarations link against the definitions.
   - The four conversion functions were run on the host, with PRMT emulated per the PTX spec, against independent
     float-derived expectations over 2M words: 0 mismatches. `_mb_expected` (the Python check) agrees with them on
     3,000 words.
   - Not checkable without nvcc: whether the intrinsic overloads exist as I remember them. These are
     `__ldcg(const unsigned*)`, `__ldcs(const uint4*)`, `__ldg` for `unsigned/uint2/uint4/int`,
     `__hfma2(__nv_bfloat162 x3)` and `__float2bfloat162_rn`.
   - Also unverified: whether torch's `-D__CUDA_NO_BFLOAT16_CONVERSIONS__` interferes. I use no bf16 operators or
     conversions, only intrinsics and `reinterpret_cast`, so it should not.
2. **No `-arch=sm_90a`.** No probe needs sm_90a features, and passing `-arch` alongside torch's own `-gencode` flags
   is itself a risk. torch adds `-gencode` for the visible device when `TORCH_CUDA_ARCH_LIST` is unset. The
   muse-spark Dockerfile sets it only as a build ARG. If the image does export a multi-arch list (for example
   `9.0;10.0a;10.3a`), the compile takes ~3× longer but should still succeed. `MB arch flags` shows which happened.
   I avoided touching `os.environ` so the static checker sees nothing that looks like env/secret access.
3. **Build directory.** `load_inline` uses the default `~/.cache/torch_extensions`. If HOME is read-only in the
   judge container, the build fails with a permission error in the log.
4. **Raw `cudaDeviceAttr` integers** (13, 16, 36, 37, 38, 39, 81, 82, 8, 97) are used instead of names, in case
   CUDA 13 removed some enum names. If a number is wrong, that label is wrong but nothing breaks. Cross-check SMs and
   L2 against the torch line.
5. **`clock64` vs `%globaltimer`** give the SM clock during each kernel. The last chase hop may finish after the
   closing clock read, which is a 1/20,000 error.
6. **M5 op counts are not from SASS** (cuobjdump would be a subprocess). The values/clk number is what matters.
   `overhead` shows how much the harness itself costs.
7. **FA3 sweep.** `meta[:4]` assumes the upstream metadata layout, with `num_splits_dynamic` first. If the fork
   reordered it, the numbers will look implausible: ignore them.
8. **FA3 append probe** may hit a TORCH_CHECK, which is printed and harmless. An untested kernel path could fault,
   which kills the run after all other output has printed. It is last for that reason.
9. **Post-expert-norm probe** calls the private `_post_expert_norm_weighted_sum_kernel` directly with the same
   arguments the stock wrapper uses at ≥ 32 tokens. Its name could change with a different image.
10. **Repeated imports (attention_decode).** The probes run at module import. If the harness imports the
    submission once per test case or process, they run again each time, along with the baseline JIT warm-ups they
    trigger. That costs ~30–90 s × number of imports.
    - Detect it: count the `LAB muse_attention_decode r0b:` headers in `log_tail`.
    - Remedy: set `LAB_AT_IMPORT = False`, or a smaller `LAB_PROBES` subset, with a new nonce.
    - Not fixed in code on purpose: any cross-process "already ran" marker would mean file I/O.
11. **Static checks.** No import of the task's checker module and no `subprocess`/`socket`/`inspect`/`os`. Nothing
    patches or edits loaded modules: the private Triton kernel and `_qk_norm`/`_cos_sin` are only called.
    `load_inline` runs nvcc internally, which is the sanctioned route (llms.txt "compile it inside the file with
    PyTorch's load_inline").

## Not done (and why)
- M3 variants with `cp.async` 16 B and TMA 1-D bulk loads (study §A.8). Each needs its own pipeline code, over the
  ~150-line budget per probe. `ld.global.nc` gives the baseline curve first.
- M6 (PDL edges under `torch.cuda.graph`). It needs a two-kernel `cudaLaunchKernelEx` path; queued for a later lab
  run.
- SASS inspection of the conversion variants: not possible without a subprocess.

# stream_lab — the streaming-primitive lab, and the reusable `okc_stream` block

**LAB FILE: check runs only, never submit for score.** Nothing here has run on a GPU yet. See "Validation" for what was
checked on the Mac.

| file | what it is |
|---|---|
| `stream_lab.py` | check-only file for `muse_bf16_proj`. `kernel_fn` and the TypedDicts are byte-identical to `problems/muse_bf16_proj/baseline.py`, so the run passes. `prepare()` runs the baseline prepare, then the lab (untimed, deadline 190 s) |
| `stream_block.py` | `STREAM_BLOCK_CUH`: the reusable device-side bulk-copy + mbarrier ring with bounded waits. The lab embeds it verbatim between `// ===== BEGIN/END okc stream_block v1 =====` and measures it as S3/S4 |

```
okc check kernels/lab/stream_lab.py --no-wait --json      # line 1 names the problem (muse_bf16_proj)
```
A repeat needs a new `# nonce stream-lab-N` on line 3 (identical code is deduplicated). Read
`===== STREAM SUMMARY =====` first. It is printed last. The harness adds ~5 KB of `USDT ... profiler_start/stop`
lines after prepare. Our output is ~8.5–9.5 KB (summary ≤ 2.7 KB), which fits the 16,000-char tail. If the front falls off, the summary
still has everything decision-relevant.

## What it measures (and what it no longer re-measures)

Lab M1–M5 (job 7516684e37bb45d1) already answered whole-GPU plain-load streaming:
- LDG.128 reaches 4.29–4.48 TB/s on 1 GiB with ≥ 32 KiB/SM in flight.
- Random 128 B rows reach 3.67 TB/s and 64 B rows 2.24 TB/s.
- Marlin's dequant runs at 38.4 values/clk/SM, against 28.6–32.7 needed for 4.2–4.8 TB/s.

So S1 appears only as reference points. The lab targets the open questions:

| part | question it answers |
|---|---|
| **A** contig 1 GiB | Do the smem pipelines (S2 cp.async, S3 bulk ring) reach the same ~4.4 TB/s as LDG, and with how many KiB/SM in flight and how many CTAs/SM? Does a 1-CTA/SM ring get there? Consumer-warp count (n1/n2/n4/n8). N-split vs chunk-interleaved split. |
| **B** whole matrices | What does one launch of a 9–64 MiB matrix actually stream under graph replay with a cold L2 (the judge's conditions)? Matrices: expert W2 9.4 MB, W13 ×1/×2/×3 = 18.9/37.7/56.6 MB, qkv 36 MiB, pooler 64 MiB, plus a 256 MiB control. The fit `t = t0 + bytes/BW` splits the cost into a fixed overhead and a rate. |
| **B2** MoE bs1 | With 1 or 3 experts over all 132 SMs, does K-split (2, 4) beat N-split? What does the split-K reduction cost: fp32 atomics fused into the stream kernel, or a second reduce kernel in the same graph? |
| **C** gathers | Can random 128 B rows (paged KV, page size 1) pass 3.67 TB/s with more bytes in flight (S1 up to 256 KiB/SM, S2/S3 smem rings up to 192 KiB/SM)? 32-lane vs 1-lane bulk issue. 64 B and 256 B rows (256 B = both KV heads of a token). Table size 1.34 GB vs 0.32 GB. |
| **D** with compute | Does the stream hold up while the consumer does what an NVFP4 GEMM does per 16 B: Marlin LOP3 dequant (exactly M5's marlin-lop3) + BF16 scale FMA + 4× `mma.sync m16n8k16`? Measured for S1 (registers), S1 PIPE (register double buffer), S2, and S3 with 4/8/12 consumer warps (and 2 CTAs/SM), on 1 GiB and on W13x3, with regs/local per kernel. This is K2A v0's failure mode. |
| **E** S4 TMA 2-D | The same with tensor-map boxes `[h rows × 2048 B]`, and also on whole matrices, K-split boxes and with the D consumer. Runs last because a bad tensor map is the one thing that can fault (see risks). |

### Schemes and the in-flight formula used in every table (KiB/SM = bytes in flight per SM, nominal)

| scheme | how | bytes in flight per SM |
|---|---|---|
| S1 `u t c` | `ld.global.nc.L1::no_allocate.v4`, U loads per thread before use; `p` = register double buffer | CTAs/SM × threads × 16 × U |
| S2 `s t v c` | `cp.async.cg` 16 B, `s`-stage smem ring, `wait_group s-2` + `__syncthreads` per stage, v copies/thread/stage | CTAs/SM × (s−1) × threads × 16 × v |
| S3 `s chunk n c` | okc_stream ring: 1 producer warp + n consumer warps; contig = one bulk copy per chunk; gather/K-split = one bulk copy per row segment (32 lanes, or 1 lane for `l1`) | CTAs/SM × (s−1) × chunk |
| S4 `s h n c` | as S3, one `cp.async.bulk.tensor.2d` box of h rows × 2048 B (1152 B for 2304 B rows, rb/k for K-split) | CTAs/SM × (s−1) × box bytes |

Suffixes: `il` = chunk-interleaved split (chunk g → CTA g % G) instead of N-split slices; `kN` = K-split into N row
segments; `l1` = one issuing lane. CTAs/SM is capped by occupancy (`o` column). The in-flight figure uses the
capped value.

## How to read the output

Every measurement is checked first:
- `ok`: the kernel's checksum (sum of every consumed 32-bit word, mod 2^32) equals torch's sum over exactly the bytes
  the pattern defines. The checksum is warp-reduced and spread over 1024 sink slots, ~4 atomics per address, so the
  epilogue does not distort the short whole-matrix timings.
- `BAD`: the split or pipeline read wrong or stale data.
- `TIMEOUT`: an mbarrier wait hit its bounded budget. The abort flag is set, the scheme is disabled for the rest of
  the lab, and the run continues.

TB/s = 1e12 B/s. The percentage is of 4.8 TB/s.

- `SL build … OK in X s (S4 on|off)`. If the full build fails, it is retried once with `-DSL_NO_S4`. Then
  `SL BUILD FAILED …` shows the nvcc tail and `SUM E S4: skipped: S4 not compiled`.
- `SL floor: graph replay of 1 empty launch X us; +Y us per extra launch in the same graph`. X is what the judge
  subtracts (~4 µs). Y is the price of a kernel boundary inside a graph, i.e. the cost of splitting work into two
  kernels.
- **A/C/E pattern tables.** One line per scheme:
  - the best config, with TB/s, %, KiB/SM, regs (`cudaFuncGetAttributes`), occupancy and check status;
  - then the cumulative curve `X:T`: the best TB/s of any config with ≤ X KiB/SM in flight, printed where it
    improves. Read it as "how much in flight do I need for T".
  - The smallest in-flight config reaching 90/95% of each pattern's best is in the `SUM` lines (`90% @NK …`).
  - `A slice->interleave` shows N-split vs interleaved for the best S1/S3.
  - `C128 table 1.34GB->0.32GB` shows the effect of table size (TLB/DRAM-page locality).
  - S1 gathers are register/occupancy-capped. The U=8 kernels hold row/part/index arrays plus 32 regs of loads, so
    their real depth tops out near M4's 64 KiB/SM. `SUM C … S1 deepest NK` prints the deepest S1 point reached.
    "S1 cannot go deeper, smem rings can" is a finding, not a sweep gap.
- **B table.**
  - Rows are configs; columns are matrices `name/K`.
  - Cell `e1|eK`:
    - e1 = bytes / (median replay time − floor), one launch per replay. This is what the judge would see for a
      kernel that only streams this matrix.
    - eK = the same with K launches on K cold copies in one replay. This removes the replay floor and shows
      back-to-back rate including the in-graph launch gap.
  - `t0 … BW …`: the fit over W2..pool64M. t0 is the fixed cost per launch beyond the floor (launch, ramp, tail);
    BW is the asymptotic rate. A large t0 with BW ≈ A's rate means small matrices lose to latency and ramp, not to
    the load scheme.
- **B2.**
  - `W13x1/W13x3 S1(…) k1/k2/k4 a/b/c | S3(…) …` gives eff TB/s for N-split and K-split 2/4, using the best S1 and
    S3 configs of A.
  - `split-K reduce on W13x3`:
    - `fused atomics +x us`: the same kernel with 3 × 8192 × k fp32 atomicAdds at the CTAs' ends, i.e. one per output
      partial at bs1.
    - `2nd reduce kernel +y us`: a `torch.sum` over the [k, 24576] partials appended to the same graph.
- **D.**
  - `D cfg  w0/w1/w2 | W13x3 w0 … w2 … | rR/lL/oO`.
  - TB/s on 1 GiB at work 0 (checksum only), 1 (dequant + scale) and 2 (dequant + scale + 4 mma.sync per 16 B).
  - Then eff TB/s on 3 experts at w0/w2, then regs/local-memory bytes/occupancy of the WORK kernel.
  - The question: which scheme keeps the most of its w0 rate at w2. Whether S3 needs n8/n12 consumer warps
    (issue slots) answers question (4), whether bulk copies free enough registers and issue slots.
- **Summary** `SUM …` lines repeat all of this compactly, then `SUM VERDICT` lines:
  - ≥ 4.0 TB/s on 1 GiB per scheme, with the minimum KiB/SM;
  - ≥ 4.0 on whole matrices (W13x1, W13x3, qkv, pooler);
  - ≥ 4.0 on gathers;
  - the best w2 (dequant + mma) streaming rate.
  - `SUM health` gives measurements, BAD checksums, timeouts, errors and which schemes were disabled.

Decision rules this is meant to feed (R1: numbers, not assumptions):
- **MoE (K2A v2).** Take the D winner at w2 and its config.
  - If S3 (n8/n12) holds ≥ ~4 TB/s at w2 while S1 drops, build v2 on `okc_stream` with that stages × chunk.
  - Use B2 to pick N- vs K-split and atomics vs a 2nd kernel at bs1, and the B fit t0 to budget the fixed cost per
    expert kernel.
- **Attention (K1 full layers).** If C shows S2/S3 above S1's ~3.67 at 128 B with deeper rings, move the K/V row
  fetch to that scheme. If 256 B rows beat 128 B, fetch both heads' rows per token.
- **proj GEMV.** Use B's qkv36M/pool64M e1 and t0. If they are close to A's rate, the stream is not the gap and the
  epilogue/launch is.

## `okc_stream` block API (`stream_block.py`, v1)

Include `STREAM_BLOCK_CUH` verbatim after `<cuda_runtime.h>` and `<stdint.h>`. It needs sm_90 (PTX ≥ 8.0) and
nothing sm_90a. Namespace `okc_stream`:

```cpp
constexpr int kMaxStages = 8;
struct Guard { int* flag; long long budget; };            // global abort flag; clock64 cycles per wait (1<<25 ≈ 17 ms)
struct Ring  { uint8_t* data; uint32_t data_s, full_s, empty_s, slot_bytes; int stages; };

Ring ring_make(uint8_t* data, uint64_t* bars, int stages, uint32_t slot_bytes);  // bars: __shared__ uint64_t[2*kMaxStages]
void ring_init(const Ring&, uint32_t consumer_warps);      // thread 0 only, then __syncthreads() by the CTA

// producer warp (warp-collective = all 32 lanes, converged)
bool producer_acquire(const Ring&, int k, const Guard&);   // slot of chunk k free? false = timeout/abort -> leave loop
void producer_put(const Ring&, int k, const void* src, uint32_t bytes);      // ONE lane: expect + one bulk copy
void producer_expect(const Ring&, int k, uint32_t total_bytes);              // ONE lane: announce chunk k's bytes
void producer_copy(const Ring&, int k, uint32_t dst_off, const void* src, uint32_t bytes);  // any lanes, after expect
void producer_put_tile(const Ring&, int k, const void* tmap, int c0, int c1, uint32_t box_bytes); // ONE lane: TMA box

// consumer warps (warp-collective)
const uint8_t* consumer_wait(const Ring&, int k, const Guard&);   // slot pointer, or nullptr on timeout/abort
void consumer_release(const Ring&, int k);                        // __syncwarp + lane 0 arrives on empty

// raw wrappers, if you build your own pipeline: mbar_init, fence_mbar_init, mbar_arrive, mbar_arrive_expect_tx,
// mbar_try_wait, wait_warp, bulk_g2s, tma_g2s_2d, smem_addr
```

Skeleton (the lab's `s3_kernel` is the complete, measured version):
```cpp
extern __shared__ __align__(128) uint8_t smem[];
__shared__ __align__(8) uint64_t bars[2 * okc_stream::kMaxStages];
uint8_t* data = smem + ((128u - (okc_stream::smem_addr(smem) & 127u)) & 127u);    // request stages*chunk + 128 B
const okc_stream::Ring R = okc_stream::ring_make(data, bars, STAGES, CHUNK);
const okc_stream::Guard g{err_flag, 1ll << 25};
if (threadIdx.x == 0) okc_stream::ring_init(R, NC);
__syncthreads();
if (warp == 0) {                                   // producer
  for (int k = 0; k < n; ++k) {
    if (!okc_stream::producer_acquire(R, k, g)) break;
    if (lane == 0) okc_stream::producer_put(R, k, src_of(k), bytes_of(k));
  }
} else {                                           // NC consumer warps
  for (int k = 0; k < n; ++k) {
    const uint8_t* s = okc_stream::consumer_wait(R, k, g);
    if (!s) break;
    /* dequant + mma from s; every lane reads its part */
    okc_stream::consumer_release(R, k);
  }
}
// no __syncthreads after the loops (a timed-out role must be able to exit alone)
```

Rules (each one is a hang or a race if broken):
1. **`expect_tx` == the sum of the bytes of the copies for that chunk.** A mismatch never completes the phase. The wait
   then times out and sets the flag; it does not hang. Gather: lane 0 calls `producer_expect(nr*rowbytes)` before
   `__syncwarp()`, then lanes copy. Copies may also land before the expect, since the tx-count may go negative.
2. `full` count is 1 (one expect per fill) and `empty` count is the number of consumer warps (one release per warp per
   chunk). Every consumer warp must wait and release **every** chunk k, in order.
3. Warp-collective functions need all 32 lanes converged (they use `__all_sync`/`__shfl_sync`). Never call them under
   lane-divergent control flow.
4. Bytes and offsets are multiples of 16, src/dst 16 B aligned. Tensor boxes need 128 B-aligned slots, so keep
   `slot_bytes % 128 == 0` and the data base 128 B aligned. stages ≤ 8. Pending tx per phase < 2^20 B.
5. After the loops, no `__syncthreads()`. On timeout, all roles leave their loops independently and the CTA exits.
   The host checks the flag after the kernel. The lab treats a set flag as "measurement aborted"; production code
   should treat it as a bug.
6. `consumer_wait` returns a generic pointer into smem. Reads after it are ordered: the wait is acquire, and its asm
   has a `"memory"` clobber. `consumer_release` orders the warp's reads before the arrive (`__syncwarp`).
7. Cost: `k % stages` / `k / stages` is a runtime division per chunk (~20 instructions). That is negligible for
   chunks ≥ 4 KiB. Inline a (slot, phase) counter if you use tiny chunks.

Why this block: plain LDG already streams 4.48 TB/s, but only with ≥ 32 KiB/SM in flight in registers. A
dequant + mma consumer cannot spare the 16–32 regs/thread that costs, and its issue slots stall the loads. The ring
holds 32–192 KiB/SM in smem with one issuing lane, so the consumer warps only compute. Part D measures whether that
argument holds. If D shows S1/S1p ≥ S3 at w2, use the S1 PIPE pattern instead; it is in the lab's `s1_kernel`.

## Validation done on the Mac (scripts in the agent's scratch dir `stream_lab/`)

- `py_compile` passes for both files. The `sync_block.py` check confirms the block inside `stream_lab.py` is
  byte-identical to `STREAM_BLOCK_CUH`.
- **Static scan** (`static_scan.py`):
  - lines 1–3 are correct;
  - imports are only `torch`, `typing`, `time`, `hashlib` and the baseline's sglang import, with no
    os/sys/subprocess/environ/inspect/ctypes and no reference import;
  - `kernel_fn` and the TypedDicts are AST-identical to baseline.py, and `prepare` is the baseline body + the lab hook;
  - all 13 inline-asm statements have consistent operand counts and only `r/l/f/n` constraints.
- **clang** (Apple clang 17, CUDA mode with stub headers):
  - Device pass (sm_90) and host pass, with and without `-DSL_NO_S4`, give 0 errors and 0 warnings with
    `-Wall -Wextra`.
  - A mutation test confirmed the device pass instantiates and checks the kernel templates.
  - It does **not** check the asm text or asm constraint letters. Those were checked by hand against CUTLASS and
    GPU-validated in-repo code; the citations are next to each asm.
- **CPU emulation** (`emulate.py`, ~80 s):
  - Python ports of `c_init/c_chunk/g_init/k_init/r_chunk` and every per-thread mapping (S1 contig and rows, S2
    contig and gather, S3 producer copies and consumer reads, S4 boxes) over 2,143 configurations. These include real
    W13x3 on grids 132/264, K-split 2/4, interleave, W2 on 528 CTAs and random gathers.
  - Results: every 16 B piece is consumed exactly once, the copies cover exactly what the consumers read, expect_tx
    equals the sum of copies, everything is aligned, gather indices are in bounds, and mma call sites are
    warp-uniform.
- **mbarrier ring model check:** 600 random schedules (stages 1–8, 1–12 consumer warps, 0–40 chunks, out-of-order
  copy completion, expect before or after the copies) with PTX phase/arrival/tx-count/`try_wait.parity` semantics.
  The run terminates, every consumer reads its own chunk, and no slot is overwritten before release.
- **Injected bugs:**
  - producer parity, consumer parity, expect_tx off by 16 B, wrong empty count, release before read, double release.
  - Each is caught by an assertion or becomes a bounded abort. 0 hangs.
- **S2 `cp.async.wait_group` accounting:** stages 2–8 × 0–39 stages.
- **Dry run of the whole Python orchestration** (`dryrun.py`):
  - CPU torch 2.14.1 (the judge's version) with mocked CUDA events and graphs and a fake extension. The fake reads
    the real buffers through the raw pointers and writes the checksum the kernel would.
  - Normal run: 417 measurements, 0 BAD, 0 errors, 8.7 KB output.
  - Injected faults: a timeout disables S3 only; a poisoned context stops after 6 errors and prints the summary; a
    build failure prints the summary. Each returns the baseline state.

## Compile and hang-risk spots (nothing compiled by nvcc yet)

1. **One extension, ~33 kernel instantiations**, including 11 WORK kernels with the mma consumer. Estimate 40–70 s of
   nvcc. The microbench, with 30 light kernels, compiled in 22 s on the judge, and k1/k2a in 32–42 s. The deadline
   logic leaves ≥ 90 s of measuring even at 100 s of compile. On a build failure the retry without S4 costs a second
   compile. It is skipped after 100 s.
2. **S4 host code is the main compile risk:**
   - `CUtensorMap`, the `CU_TENSOR_MAP_*` enums, and `cudaGetDriverEntryPointByVersion(…, 12000, cudaEnableDefault,
     &q)` under `CUDART_VERSION >= 12050`. This mirrors CUTLASS `cuda_host_adapter.hpp`'s CUDA 13 branch.
   - `<cuda.h>` is header-only here (load_inline already includes it), and nothing links libcuda.
   - If any of it mismatches the CUDA 13.0 headers, the retry without S4 keeps S1–S3.
3. **S4 runtime fault risk.** An encode error is returned and printed, not launched. A malformed but accepted map
   could make the TMA unit raise an illegal-instruction or address error. That poisons the context, so S4 runs
   **last**: everything else has already printed, and the summary is pure Python and still prints. The run would
   then fail in the tests; the log still comes back. Set `LAB_S4 = False` (and a new nonce) to drop it.
4. **Hang protection.** Every mbarrier wait (S3/S4) is bounded: 2^25 clock64 cycles ≈ 17 ms per wait, against kernels
   of ≤ 0.4 ms. On timeout the flag is set, both roles leave their loops, and the CTA exits. Worst case, a broken
   launch costs ~17 ms plus the time for the other CTAs to see the flag. S1/S2 have no waits that can block (cp.async
   always completes). Residual risk: a CTA that exits on timeout while its bulk copies are still in flight. That
   happens only after a protocol bug, which the CPU model checker found none of.
5. **PTX strings:**
   - From CUTLASS (`barrier.h`, `copy_sm90_tma.hpp`, `mma_sm80.hpp`, `array.h`): mbarrier init/arrive/expect_tx/
     try_wait.parity, fence.mbarrier_init, cp.async.bulk, cp.async.bulk.tensor.2d (without the optional
     `.L2::cache_hint`), mma.sync m16n8k16 bf16, fma.rn.bf16x2.
   - From GPU-validated in-repo code: cp.async.cg/commit/wait_group (k1_v0), and
     `ld.global.nc.L1::no_allocate.v*.s32` (sglang `utils.cuh`, k2a).
   - The mbarrier section anchor in the PTX ISA URL is from memory; the cp.async.bulk(.tensor) and membar anchors
     are cited elsewhere in repos/.
6. **Graph capture.** `cudaFuncSetAttribute` and occupancy queries happen in `sl_attrs` before any capture. The
   tensor-map entry point is resolved on the first eager S4 launch, before any S4 capture. Captured launches call
   only `cudaGetLastError`.
7. **Registers.** WORK kernels carry ~16 bf16x2 + 16 fp32 extra registers. The `r`/`l` columns show regs and local
   (spill) bytes from `cudaFuncGetAttributes`. Nonzero `l` in D means the consumer spilled, so read its w2 numbers
   with that in mind.
8. **Not modelled.** Real NVFP4 scale loads: the scale is a data-dependent register constant, while a real kernel
   also streams 256 B of scales per 2048 B row, but those bytes are in the stream here. The B operand (activations)
   comes from registers. The atomics epilogue models only the bs1 partial count.

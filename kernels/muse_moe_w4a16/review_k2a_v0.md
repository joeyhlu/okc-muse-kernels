# Review of K2a v0 (`k2a_v0.py`, nonce `k2a-v0-1`): an independent pre-`okc check` CUDA review

Reviewer: an independent agent. I read the code only; nothing was compiled or run (no nvcc or GPU here). Files reviewed: `k2a_v0.py` (809 lines),
`k2a_design.md`, `k2a_emulate.py`. I checked them against the spec (`problems/muse_moe_w4a16/*`), the rules (`sources/okc_site/llms.txt`,
`notes/study/01-challenge.md` §c), the PTX ISA, CuTe and SGLang's own Marlin/JIT sources.

## Verdict: **SHIP to `okc check`, after deleting line 625** (a 1-line should-fix). **Blockers: none.**

I found no compile error, no out-of-bounds access, no race, no deadlock, no graph-capture violation and no rules
violation in the CUDA or Python. Every inline-asm, intrinsic, launch-bounds and shared-memory spot that the design doc
flagged as a compile risk checks out against a working precedent or the PTX ISA (§1). Every global and shared index is
bounded for M = 1, 4, 33, 257, including the case where one expert gets every token (§2). The one change I would
make before spending a check run is S1 (`os.environ` write). The run can still die in three ways that the fallback
cannot catch (§5): a prepare() timeout, a sticky device fault, or a hang. None of these is predicted by the code as
written.

| id | severity | item |
|---|---|---|
| S1 | **should-fix (before the check)** | `os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"` (line 625): risk from the static checker, and the value leaks into the process |
| S2 | should-fix (cheap insurance) | Host-side `TORCH_CHECK`s for dtype, contiguity, 16-B alignment and sizes, so that misuse raises a catchable exception instead of a sticky fault |
| S3 | should-fix (process) | Read `[k2a] compiled in Xs` and the `-Xptxas=-v` spill lines from the first check log; act if compile is over ~150 s or there are large spills |
| N1–N12 | note | See §2–§6 |

---

## 1. nvcc compile

| spot (k2a_v0.py) | finding | evidence |
|---|---|---|
| `bmul2`, l.195–199: `fma.rn.bf16x2 %0,%1,%2,%3` with `"=r"`, `"r"`, `"r"(0x80008000u)` | **OK.** The bf16x2 operands are `.b32` registers. An `"r"` constraint with a constant just materializes a `mov`. The operand count is 4. PTX: "`fma{.relu}.{bf16, bf16x2}` require `sm_80` or higher", PTX ISA 7.0. There is **no `.ftz` form for bf16/bf16x2**. | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-fma |
| `mma16816`, l.201–208: `"+f"` C/D tied, with `{%0..%3}` reused as the C operand | **OK.** This is the identical asm string and constraint pattern that SGLang ships: `repos/sglang/.../jit/csrc/attention/kda_prefill.cu:135-138` (`{%0,%1,%2,%3},{%4..%7},{%8,%9},{%0,%1,%2,%3}` with `"+f"` ×4 and `"r"` ×6). The CuTe form, with separate C inputs, is the same instruction: `repos/cutlass/include/cute/arch/mma_sm80.hpp:224-247` (`SM80_16x8x16_F32BF16BF16F32_TN`). There are 10 operands, %0–%9. | files cited |
| `template<int MODE,int NG> __global__ void __launch_bounds__(NTHR,1) gemm_kernel` (l.320–322) | **OK.** This is the standard placement. NTHR = 512 gives 65536/512 = **128 regs/thread cap**, so ptxas spills rather than producing a "too many resources requested for launch" error. Spills are a performance risk only (S3). | — |
| static smem, l.328–330 | **OK.** `red` = 8·(8NG)·32·4 B = 8 KB·NG. That is **32,768 B at NG = 4**, + 392 B of s_e/s_cnt/s_off/s_hdr, under the 48 KB static limit. No `cudaFuncSetAttribute` is needed, and there is no dynamic smem. route: 256 B; final: 64 B. | — |
| `route_kernel` `__launch_bounds__(1024)` (l.214) | **OK.** The cap is 64 regs; the kernel is trivially under it. | — |
| `min(NG, (T-n0+7)>>3)` (l.405) | **OK.** Both operands are `int`, so this is CUDA's `__host__ __device__ int min(int,int)` (crt/math_functions). There is no `using namespace std` in the TU. | — |
| `__ldg(const uint4*)` (l.368, 398, 418), `__ldg(const uint32_t*)` (l.369, 399) | **OK.** Both overloads exist (sm_32_intrinsics: uint4 and unsigned int). The reinterpret-cast of the bf16 pointer is fine. | — |
| `__byte_perm(sw, 0u, 0x4040u + 0x0101u*bi)` (l.423) | **OK.** The selector nibbles are 0–4, so bit 3 (sign-replicate in raw `prmt`) is never set, and the upper 16 selector bits are 0. The IMAD `*16 + 0x75807580` cannot carry between halves: the max half is 150·16 + 0x7580 = 0x7EE0. | — |
| `__funnelshift_r(x, x, s)` (l.184) | **OK.** This is a rotate right. | — |
| `C10_CUDA_KERNEL_LAUNCH_CHECK()` | **OK.** `<c10/cuda/CUDAException.h>` is included at l.171. | — |
| `#include <cuda_bf16.h>` (l.172) | **OK.** torch's `-D__CUDA_NO_BFLOAT16_CONVERSIONS__` / `__CUDA_NO_BFLOAT16_OPERATORS__` remove only the implicit `__nv_bfloat16` conversions and operators. The code uses only the intrinsics `__float2bfloat16_rn`, `__bfloat162float` and `__floats2bfloat162_rn`, plus bf16 copy-assignment. This closes the open question in lab README §1. | — |
| `int64_t` bindings via `functions=[...]` (l.161–166 vs l.577, 597, 612) | **OK.** The declarations in `CPP_SRC` match the definitions token for token (`route(Tensor,Tensor,int64_t,int64_t,int64_t)`, `gemm(int64_t,int64_t,Tensor×6,int64_t,int64_t)`, `finalize(Tensor×6,int64_t,int64_t)`). load_inline generates `m.def` for each. Python ints bind to `int64_t`. | — |
| `launch_gemm<MODE,NG>(torch::Tensor&...)` called with by-value params (l.602–608) | **OK.** They are lvalues. `CUDAStream` converts implicitly to `cudaStream_t`. | — |
| `extra_cuda_cflags` (l.633) | **OK.** `-Xptxas=-v` in that exact spelling is what SGLang's own JIT passes (`repos/sglang/.../jit/utils/compile/loader.py:94`). `-O3`, `-lineinfo` and `--expt-relaxed-constexpr` are fine; any duplicates of torch's own flags are harmless. | — |
| arch | **OK.** `TORCH_CUDA_ARCH_LIST=9.0` gives `compute_90/sm_90`. Nothing uses sm_90a-only features (no wgmma, setmaxnreg or TMA). mma.sync m16n8k16 bf16 and fma.rn.bf16x2 are sm_80+. See S1 for why the env write should go anyway. | — |
| toolchain presence | **Likely fine.** The image's SGLang itself runs `ninja` (`jit/utils/compile/ninja.py:156`) and calls `torch.utils.cpp_extension.load_inline` (`srt/utils/cuda_vmm_utils.py:453`, `srt/distributed/.../pynccl_allocator.py:211`). If `~/.cache/torch_extensions` is unwritable, the exception is caught at l.763 and the run falls back. | files cited |

**Compile time (N1).** There are 6 `gemm_kernel` instances. Each has the RING = 4 loop `#pragma unroll`ed, and the loop
body contains the full epilogue (reduction, two `__syncthreads`, MODE-1/2 stores), so each instance inlines the
epilogue 4×. There are also 2 TUs that both include `<torch/extension.h>`. I expect roughly 1–3 min, which fits
inside the 300 s prepare budget. l.636 prints it. The lab README estimates ~30–60 s of nvcc for a smaller extension.
If the first check shows more than ~150 s, the cheapest cut is to stop unrolling the RING loop (`#pragma unroll 1`
over d, with the ring held in a small register array indexed by `d & 3`, or a 2-deep manual unroll), or to drop the
NG = 2 instantiation.

---

## 2. Runtime: bounds, races, deadlocks, uninitialized reads

**Capacity worst case.** `topk` ids are distinct per token, so one expert holds at most M rows. "257×8 routes on one
expert" cannot happen. Total routes ≤ 8M, and `act_rows`/`pair2row`/`inter`/`Y` are all sized 8M (`r_cap`,
l.797–802). With M = 257 and one expert taking all tokens, nch = ceil(257/32) = 9 chunks per unit. The cursor handles
this because nch ≥ 1 for every active expert, so `++q.c == q.nch` always fires (l.316).

| access | index range | bound |
|---|---|---|
| route `ids[t*8+s]` (l.230, 276) | t < M | 8M ✓ |
| route `pair2row[p] = -1` (l.236–239), `pair2row[t*8+slot] = r` (l.282) | < 8M | slice [128+8M, 128+16M) of meta (numel 128+16M, l.800) ✓ |
| route `act_rows[r]` (l.281) | r < total ≤ 8M | ✓ |
| route `meta[8+a]`, `[40+a]`, `[72+a]` (l.254–256) | a < 32 → indices up to 103 | < META_HDR = 128 ✓ |
| gemm `wq[blk*32+lane]` (l.368, 398) | blk = (e·NRT+rt)·128+kbg, with e = meta[8+a] ∈ [0,32), rt < NRT, kbg = ks·KPS+w·NKB+kb ≤ 127 for KS ∈ {1,2,4} | numel/4 of w13q/w2q ✓ |
| gemm `sq[blk*16+(t>>1)*8+g]` (l.369, 399) | ≤ blk·16+15 | ✓ |
| gemm `X[xrow*4096 + kel .. +7]` MODE 1 (l.418) | xrow = act_rows[crow] < M (token id); kel+8 ≤ 4096 | ✓ |
| gemm `X` MODE 2 (= inter) | xrow = crow < total ≤ 8M | ✓ |
| gemm `red[...]` (l.453, 463, 476–477, 491) | max ((7·8NG + (2NG−1)·4+3)·32+31) = 2048·NG − 1 | size 2048·NG ✓ |
| gemm `out_bf[(crow0+n)*4096 + 16rt+rr]` (l.481) | crow0+n < off+T ≤ total; col < 4096 | ✓ |
| gemm `out_f[((ks·8M)+crow0+n)*4096 + 32rt+r]` (l.492) | ks < KS ≤ 4 = KS2_MAX | y is [4, 8M, 4096] ✓ |
| gemm `gsc[s_e[a]]` (l.467) | e < 32 | gsc13/gsc2 are [32] ✓ |
| final `nw` float4 `[j*256+tid]` (l.526) | < 1024 | 4096 floats ✓ |
| final `Y` float4 (l.540–543) | (ks·8M+crow)·4096 + 4·(j·256+tid)+3 | ✓ |
| final `out` bf16x2 (l.570–571) | 2·(j·256+tid)+1 < 2048 pairs per row | ✓ |

Every size is a multiple of 16 B, and the bases come from the caching allocator, so all uint4/float4/bf16x2 accesses
are aligned. The one exception is caller-provided `hidden_states` (see S2).

**Uninitialized reads (N2).** Only `meta[8+a]`, `meta[40+a]` and `meta[72+a]` for a ≥ n_act are read without being
written (l.338–342, into smem). They are never used, because every use is guarded by `u < U`, which implies
`a < n_act`. That is harmless, but compute-sanitizer `initcheck` would report it. Everything else that is read is
written earlier in the same call:
* `inter` rows [0,total) × all 4096 columns: every (a, rt ∈ [0,256)) unit writes 16 columns for every row of that expert.
* `Y[ks][row]` for every ks < KS.
* `pair2row` for every pair.

Stale contents across graph replays are therefore never read. This matches the emulator's NaN-poisoned buffers.

**Races.**
* route: lane 0 writes `s_cnt`, then a barrier (l.240); warp 0 writes `s_off`/meta, then a barrier (l.267). Pass 2
  writes disjoint `act_rows`/`pair2row` slots. The `-1` fill (l.236) touches only non-local pairs and pass 2 only
  local ones, so they never overlap.
* gemm: warps 8–15 store, barrier, warps 0–7 add into their own slab, barrier, epilogue reads, barrier (l.455, 465,
  496). The next writes to `red` happen only after the l.496 barrier.
* final: `s_red[par]` is double-buffered. A slot is rewritten two iterations later, and every thread has crossed the
  intervening barrier after finishing its reads.

**`__syncthreads` in conditional code.** The gemm barriers sit in `if (cc.kb == NKB-1)` inside `while (cc.u < U)`.
That flow is **block-uniform**: the cursor (`cur_unit`/`cur_next`) depends only on blockIdx, gridDim, smem `s_cnt` and
`KS`, never on the warp or lane. Only `kbg` uses `warp`. A CTA with `blockIdx.x ≥ U` skips the loop as a whole block.
In final, `continue` (l.534) depends only on (tok, s), so it is uniform too. `mma.sync.aligned` runs under `q < ng`
and `cc.u < U`, and both are warp-uniform.

**Deadlock and co-residency.** There is no inter-CTA communication: no counters, spin-waits or grid barriers. The
persistent grid of `sms` CTAs is therefore correct even if not all CTAs are co-resident.

**Route kernel (1024 threads).** It uses no atomics; all ranks come from ballot plus popc, so it is deterministic,
with rows in ascending token order. The KS2 rule (l.261, G = 132) gives n_act ∈ {1,2} → 4, {3,4} → 2, ≥ 5 → 1.
`gemm<2>` and `final` both read KS from `meta[1]`, so they agree. With n_act = 0, U = 0 and every row is zero
(KS = 4 is never used). That never happens with rank="busiest".

**Duplicate ids within a token's top-8 (N3, judge cannot produce them).** Pass 1 counts tokens, and pass 2 keeps only
the last matching slot (l.276). The earlier slot's `pair2row` entry would then be left unwritten (`torch.empty`), and
`final` would read `Y` at a garbage row, causing an illegal address. `scores.topk` never yields duplicates, so this is
a note only. If the kernel is ever reused outside the judge, count pairs and write every matching slot.

---

## 3. Numerics: where the GPU can differ from the CPU model

* **BF16 subnormal inputs to `fma.rn.bf16x2`.** Code ±0.5 decodes to the BF16 subnormal 2^-127 (l.188–191). The PTX
  ISA says of half-precision `fma`/`mul`: "By default, subnormal numbers are supported". `.ftz` exists only for
  `.f16`/`.f16x2`; the bf16 forms have no `.ftz` (URL above, sections 9.7.4.3 and 9.7.4.4).

  There is also empirical precedent on the judge's own hardware: SGLang's Marlin FP4 to BF16 path builds the identical
  encoding.
  * `jit/csrc/gemm/marlin/dequant.h:393-407` builds `(q & 0x80008000) | ((q & 0x70007000) >> 6)`, i.e. e2m1·2^-126
    with code 1 subnormal.
  * It is enabled for FE2M1f + FE4M3fn scales at `marlin_moe/marlin_template.h:367` (`dequant_skip_flop`).
  * It is scaled with a bf16x2 multiply at `marlin_moe/marlin_template.h:188-189/1303`.

  The baseline passes correctness on the same H200, so bf16x2 multiply does not flush subnormal inputs there. All
  products are e2m1·sf·2^-8 ∈ [2^-18, 10.5], which are normal and exact, so the MMA sees no subnormals from the weight
  side.

  If the hardware did flush, code ±0.5 carries 0.25/68.5 of E2M1 energy, so the relative weight error would be about
  √0.0037 ≈ 6 % per GEMM. The prepare() self-test (threshold 1e-2) would catch that and fall back.
* **MMA accumulation.** The emulator uses an FP32 einsum. Hardware `mma.sync` uses FP32 accumulate, but its internal
  alignment and rounding of the 16 products are not IEEE-sequential. The difference is around 1e-6 relative, against
  an emulated error budget of 3.8e-3 out of 1.5e-2. Not a concern.
* **Reduction order.** The cross-warp smem reduction order is mirrored exactly in the emulator (`acc[8:]+acc[:8]`,
  then w8 = 0..7 in sequence; `k2a_emulate.py:477-480`). The per-warp K loop and the final-kernel shuffle tree are not
  mirrored. Both are FP32-rounding-level effects.
* **Transcendentals.** `__expf` (l.480) and `rsqrtf` (l.562) are approximate, a few ulp. They are irrelevant at this
  tolerance. SiLU at hg ≪ 0 gives `hg/inf = -0`, not NaN.
* **BF16 rounding points.**
  * `inter`: one RN-even rounding of FP32 SiLU·up (l.481). Stock does three (h, then SiLU output); fewer roundings is
    more accurate.
  * `y`: FP32 partials summed, then RN to BF16 before the norm (l.550), matching Marlin's `use_fp32_reduce=True`
    followed by a BF16 output.
  * `(1+w)`: rounded to BF16 (l.751), as baseline.py:102 does.
  * `torch.to(bfloat16)` and `__float2bfloat16_rn` are both RN-even, so emulation and GPU agree bit-for-bit at these
    points, given equal FP32 inputs.

**What the emulator independently verifies vs mirrors (N4).**
* **Mirrored, so a shared misreading would pass.** It imports the repack (`prepack_expert`, `reencode_scales`,
  `pack_words`, `row_perm`) from k2a_v0.py. It re-implements every device formula from the same author's reading of
  the .cu: deq4, scale decode, addressing, epilogue math.
* **Independent anchors.**
  * Fragment and epilogue maps from CuTe's `SM80_16x8x16` A/B/C layouts, not from the kernel's (g, t) formulas.
  * Exhaustive E2M1 × all-252-finite-E4M3 product exactness.
  * Literal per-CTA ring/cursor simulation with full unit coverage.
  * NaN-poisoned outputs.
  * A copied `ref_kernel` + `checks.compare`.
  * Negative controls that do fail.

I also hand-checked the mirrored parts against the PTX m16n8k16 bf16 fragment definitions:
* deq4 masks and shifts against BITPOS for all 8 positions.
* The k-permutation consistency of `a0/a2` vs `b0/b1` (l.440–441).
* The scale byte order (`2ρ+h`) and group `(t>>1)`.
* The SQ/WQ permutes at l.144–146.
* The red-slab indexing.

All are consistent.

---

## 4. Rules

* **NVFP4 stays NVFP4: OK.**
  * **Same size.** Per expert, W13 is WQ 256·128·32·4·4 B = 16 MiB (= 8192·4096/2) plus SQ 256·128·64 B = 2 MiB
    (= 8192·4096/16). That is exactly 0.5 B/value plus 1 B per 16 values, the same as the source.
  * **Lossless.** The scale re-encoding (l.97–111) is a bijection on the 252 finite non-zero E4M3 magnitudes,
    E4M3 subnormals included (emulator test (2)). Signs move into the 16 codes' sign bits (l.136–138). Zero-scale
    blocks get zero codes (l.139). Dequantized values are bit-identical.
  * **Global scale.** The 2^-8 moves into the FP32 global scale as an exact power of two (l.749–750).
  * **Precedent: stock SGLang does the same kind of thing.** `srt/layers/quantization/marlin_utils_fp4.py:29-44`
    re-encodes NVFP4 scales (×2^7, bit shift), lossily for negative scales. `:47-59` rescales the global scale.
    v0's encoding is stricter than stock's.
* **Activations BF16 where stock keeps BF16: OK.**
  * `inter` is stored BF16 (l.481).
  * `h` and SiLU stay FP32 in registers, which is *higher* precision than stock. The rules forbid only lower
    ("compute at lower precision than the rules allow", llms.txt:137).
  * `y` is FP32 then BF16 (l.550). The output is BF16.
* **Nothing cached from `data`: OK.** kernel_fn allocates `meta`, `inter`, `y` and `out` per call, with sizes from M
  only (l.800–803). `state` holds only weight-derived tensors, the module and `sms`.
* **prepare() does no input-dependent work: OK.** The self-test uses synthetic inputs from its own generators
  (`_self_test_inputs`, l.712–727, seeds 11–16), never `data`.
* **load_inline spawning nvcc/ninja.** This is the sanctioned route (llms.txt:96, "compile it inside the file with
  PyTorch's `load_inline` (which uses nvcc)"). Note only.
* **Printing: OK.** Prints happen only in prepare/_build. There are no tensor prints in kernel_fn, so nothing syncs.
* **S1, should-fix: `os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"` (l.625).** "A submission is rejected if it …
  touches harness I/O or secrets" (TASK.md:78), and `OKC_TOKEN` lives in the environment. An `os.environ` write is the
  most plausible thing in this file to trip a grep- or AST-based checker. The lab author avoided `os` entirely for
  this reason (lab README §2, §11). The write also leaks into the rest of the process. That is harmless for the
  baseline, because SGLang's JIT pins its own arch (`jit/utils/compile/toolchain.py:125-136`, `target_flags()`), but
  it is unhygienic.

  **Fix:**
  1. Delete l.625.
  2. Delete `import os` (l.27); `os` has no other use.
  3. Delete `import sys` (l.28), which is unused.

  With the variable unset, torch's `_get_cuda_arch_flags` auto-detects the visible H200 and emits
  `compute_90/sm_90 (+PTX)`. Do **not** substitute a manual `-gencode` in `extra_cuda_cflags`, because torch adds its
  own gencode flags and the duplicate target is an avoidable risk.

  Residual risk: if the image *exports* a multi-arch `TORCH_CUDA_ARCH_LIST`, the compile gets about 3× slower. The
  lab README says the Dockerfile sets it only as a build ARG. `verbose=True` prints the nvcc command line, so the
  first check log shows which `-gencode` flags were used. With the variable unset, torch also prints a harmless
  `UserWarning: TORCH_CUDA_ARCH_LIST is not set ...` to stderr. Expect it; it is not an error.

---

## 5. Fallback path and failure modes

* **Graph-capturable: yes.** `_grid(dev)` is pre-cached at l.738. `_torch_path` (l.666–683) allocates only by shape
  and branches only on `config`, with no `.item()`. cuBLAS matmuls capture fine.
* **Correct: yes.** It was emulated at 5.5e-3 on CPU (design doc §8): BF16 matmuls, FP32 norm, all 32 experts
  weighted by routing.
* **Cost (N5).** One W13 dequant moves about 1.5 GB through ~12 elementwise kernels (int64 index tensor, fp32
  temporaries), roughly 0.5 ms. W2 is about half that. ×32 experts gives **about 20–30 ms per call**. With about 600–1200
  test, timed and warm-up calls, that is 0.5–1 min. **So a build failure yields a passing but slow run (speedup ≪ 1)
  well inside 15 min.** Build failure (l.761–767), repack failure (l.755–757) and self-test numeric mismatch or
  TORCH_CHECK (l.779–781) are all caught.
* **The fallback does NOT protect against these (N6):**
  1. **A sticky device fault** (illegal or misaligned address, illegal instruction) during the self-test.
     `torch.cuda.synchronize()` (l.774) raises, the `except` sets `mod = None`, but the CUDA context is dead, and the
     judge's first test fails on the fallback's first kernel. The log will at least show
     `[k2a] self-test FAILED … illegal memory access`. Only wrong results and host-side exceptions are recoverable.
     This is why S2 matters.
  2. **A prepare() timeout.** The 300 s budget covers compile, repack and the 6-case self-test. If nvcc/ptxas is slow
     on the judge box, the exception path never runs. See N1 and S3.
  3. **A kernel hang.** No loop in the code can hang for valid meta: nch ≥ 1 and U is bounded. But a hang would be
     charged as a 15-min timeout, not a fallback.
  4. **The baseline's own Marlin JIT compile** shares the 15-min run budget and is outside our control (lab README
     §moe).

**S2, should-fix: convert would-be faults into catchable exceptions.** This is cheap, host-only, and graph-safe,
because it checks pointers, dtypes and shapes, never values. Add to the C++ wrappers:

```cpp
// route():
TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && ids.numel() == 8 * M);
TORCH_CHECK(meta.scalar_type() == at::kInt && meta.numel() >= 128 + 16 * M);
// gemm():
TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == 4096);
TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "x not 16-B aligned");
TORCH_CHECK(wq.is_contiguous() && sq.is_contiguous() && gsc.numel() >= 32);
TORCH_CHECK(mode == 1 ? out.scalar_type() == at::kBFloat16 : out.scalar_type() == at::kFloat);
// finalize():
TORCH_CHECK(y.scalar_type() == at::kFloat && wts.scalar_type() == at::kFloat && nw.numel() == 4096);
```

In `kernel_fn`, also guard against the one caller-controlled alignment by adding
`if x.data_ptr() % 16: x = x.clone()` after l.792. A pointer is not an input *value*, so this is allowed under
capture. Today `generate_input` and graph buffers are allocator-aligned, so this is insurance, not a known bug.

**S3, process.** From the first check log, record:
* `[k2a] compiled in Xs` (l.636), `repacked in` (l.754) and `prepare done in` (l.782).
* The `-Xptxas=-v` lines for `gemm_kernel<1,4>` / `<2,4>`: registers (cap 128) and spill stores/loads.
* The six `[k2a] self-test` lines.

If compile time is over ~150 s, cut the unrolled RING body (N1). If there are large spills at NG = 4, that explains
any shortfall against the predicted 2.2–2.45× before any other tuning.

---

## 6. CUDA graphs and streams

* There are no host syncs in kernel_fn: no `.item()`, `.cpu()`, `synchronize` or prints. n_act, KS2, unit counts and
  chunk counts all live in device `meta` (l.258–265, l.344–352).
* Grids are fixed by shape. route is 1×1024. gemm is `sms` (132) × 512 for both modes, with `sms` read once in
  prepare (l.752). final is M×256.
* Allocations are `torch.empty` sized by M only (l.800–803), so they come from the graph's private pool under capture.
  `y` is 135 MB at M = 257 (N7, memory only).
* All launches use `at::cuda::getCurrentCUDAStream()` (l.578, 586, 614), which is the capture stream during capture.
  `C10_CUDA_KERNEL_LAUNCH_CHECK` (`cudaGetLastError`) is legal during capture.
* The self-test (l.770) runs all 8 kernel instances before any capture: route, gemm<1|2, 1|2|4>, final; bs 1/5 give
  NG 1, bs 40 gives NG 2, bs 100/128/257 give NG 4. Lazy module loading never happens inside a capture.
* N8: there is no `CUDAGuard` in the wrappers. That is fine on the single-GPU judge.

---

## 7. Other notes

* **N9 (perf).** In the MODE-1 epilogue, each thread stores one 2-byte bf16 to a scattered `(row, 16rt+rr)` (l.481).
  In MODE 2, stores are 4-byte FP32 runs of 32. Both are uncoalesced-ish, but small next to the weight stream. The
  RING-unrolled epilogue (N1) also inflates SASS and I-cache footprint. Measure before changing either.
* **N10 (perf).** `__ldg` (`ld.global.nc`, L1-allocating) on streamed weights can evict X/`inter` lines. The design
  doc already lists this (§7.8).
* **N11.** `hit |= (...)` on `bool` (l.230) is legal C++. nvcc may warn about the conversion; that is harmless.
* **N12.** Line 1 `# okc: problem=muse_moe_w4a16` and line 2 nonce `k2a-v0-1` are present, so the run will not be
  deduplicated against an earlier result.

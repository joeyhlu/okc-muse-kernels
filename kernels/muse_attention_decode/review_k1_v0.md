# Review of K1 v0 (`k1_v0.py`, nonce `k1-v0-1`): independent pre-`okc check` CUDA review

Reviewer: an independent agent, 2026-10-23. I read the code only. Nothing was compiled or run: there is no nvcc or GPU
here, and no local torch or CUDA headers.

Files reviewed:
- `k1_v0.py` (1190 lines)
- `k1_design.md`
- `k1_emulate.py` (plan, process and epilogue paths)

Checked against:
- the spec: `problems/muse_attention_decode/{TASK.md,DESCRIPTION.md,reference.py,checks.py,baseline.py,template.py}`
- the rules: `sources/okc_site/llms.txt:104-110,133-139` and `notes/study/01-challenge.md` §b–c
- the PTX ISA
- CUTLASS precedents in `repos/cutlass`, and vLLM/SGLang precedents in `repos/`

Line numbers below refer to `k1_v0.py`.

## Verdict: **SHIP to `okc check`. Blockers: none.**

I found none of the following:
- a compile error
- an out-of-bounds global or shared access, for any of the 16 tests or the 12 scored shapes
- a race, barrier divergence or cp.async misuse
- an uninitialized read
- a graph-capture violation
- a rules violation

The split plan terminates, and both kernels compute it identically. Every new token is attended exactly once, from
smem, and is never read back from the cache in the same call. FP16 cannot overflow anywhere on the data path (§3).

Before the check, I recommend four cheap Python-only edits: S1–S3 and S5. None touches CUDA_SRC, and each is at most a
few lines. They prevent a silent fall-back to FA3, or a test that dies on a TORCH_CHECK, so the first check run tells
you more. Ship as-is if you'd rather not touch it.

| id | severity | item |
|---|---|---|
| — | **blocker** | none |
| S1 | should-fix (cheap) | The self-test has no `heads dp` + `q_len 1` case. Test 5 (`batch 8, ctx 8192, heads dp, full, spread mixed`) runs that path unguarded. |
| S2 | should-fix (cheap) | The self-test's new-K/V threshold of 1e-2 can fail spuriously, on one e4m3 rounding flip in the 128-element case. The run would then silently stay on FA3. |
| S3 | should-fix (cheap) | If the first call is captured, K1 is silently never used. Log it, and consider running the self-test at import. |
| S4 | should-fix (process) | With `verbose=True`, a build failure's exception text has no nvcc output, so l.823 prints only "Error building extension". |
| S5 | should-fix (cheap) | `_supported` does not check contiguity or dtypes that `k1_forward` TORCH_CHECKs. A mismatch raises inside the test instead of falling back. |
| S6 | should-fix (perf hedge) | Add a per-(layer, q_len) route switch, so check 2 can send verify-full to FA3 without code surgery if K1 loses there. |
| N1–N12 | note | See §1–§6 |

---

## 1. Compile (nvcc 12.x/13, sm_90, torch `load_inline`)

| spot | finding | evidence |
|---|---|---|
| `mma16816` l.185-193: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, `"+f"`×4 tied C/D, `"r"`×6 | **OK.** Same opcode string as CuTe `SM80_16x8x16_F32F16F16F32_TN`. The tied form (C operand `{%0..%3}`) is the same pattern SGLang ships (`kda_prefill.cu`, cited in review_k2a_v0 §1). It is not `volatile`, which is fine because its outputs are consumed. | `repos/cutlass/include/cute/arch/mma_sm80.hpp:158-180`; https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma |
| `cp_async16` l.195-200: `cp.async.cg.shared.global [%0], [%1], 16, %2` with `"r"(src_bytes)`, `"memory"` | **OK.** `.cg` requires cp-size 16; src-size is a u32 ≤ cp-size, and 0 zero-fills all 16 B with no global read. CuTe ZFILL is identical apart from the `.L2::128B` hint (`"n"(16)`, `"r"(src_size)`). The invalid case uses `goff = 0`, i.e. `kc + 16*ch` inside the cache, so the address is valid anyway. | `repos/cutlass/include/cute/arch/copy_sm80.hpp:130-155`; https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async |
| `commit_group` / `wait_group %0` with `"n"(N)` l.202-213 | **OK.** See §2.4 for the counting. | `copy_sm80.hpp:183`; https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group |
| `ex2.approx.ftz.f32` l.156-160 | **OK.** The PTX table gives `-Inf → +0.0`. The masked-key path (l.337/340 → l.377-383) relies on this. | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2 |
| `__nv_cvt_fp8x2_to_halfraw2(v & 0xFFFF, __NV_E4M3)` l.175-178, byte order | **OK, now confirmed** (design §6 had this "as I recall it"). On sm_89+ the header lowers it to `cvt.rn.f16x2.e4m3x2 d, a`. PTX puts the upper byte of `a` in the upper half of `d` and the lower byte in the lower half. CUTLASS uses the instruction with exactly this reading: `out[0] = .x` from `source[0]`, the low byte. So low byte → `r.x` → bits 0-15. That matches the fragment derivation (dims 16t+4s+{0,1} → k rows 2t, 2t+1). | `repos/cutlass/include/cutlass/numeric_conversion.h:1440-1462`, `float8.h:463-470`; https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt |
| `__nv_cvt_float2_to_fp8x2(make_float2(x.x/sc, x.y/sc), __NV_SATFINITE, __NV_E4M3)` l.535-536 | **OK.** Lowers to `cvt.rn.satfinite.e4m3x2.f32 d, a, b`, where `a` → upper byte and `b` → lower byte. The header passes `x.y` as `a` and `x.x` as `b`, so `x.x` → low byte → `dst[2*lane]`, as intended. CUTLASS does the same: `"%0, %2, %1"` with `source[0]` as `%1`, and `from_float` puts `flt` in `b`. | `numeric_conversion.h:1495-1510`; `float8.h:433-444`; PTX cvt URL above |
| `cudaFuncSetAttribute(k1::k1_main<L>, …)` l.725, `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::k1_main<1>, …)` l.743/746 | **OK.** A template-id names a single specialization, so the runtime's `template<class T>` overloads deduce `T`. vLLM ships both forms through nvcc. Commas are nested in the call's parentheses, so `C10_CUDA_CHECK(...)` does not split them. | `repos/vllm/csrc/dsv3_fused_a_gemm.cu:700-703`; `repos/vllm/csrc/topk.cu:111-121` |
| dynamic smem | **OK.** `smem_bytes<1>` = 71,744 B and `smem_bytes<8>` = 86,080 B, both above 48 KB, so they need the opt-in. The opt-in runs before both the occupancy query and the launch (l.733, 742, 745). Two CTAs/SM need 145,536 B (L=1) and 174,208 B (L=8) including 1 KB reserved each, which is ≤ 233,472 B per SM. | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-9-0 (cc 9.0: 227 KB per block opt-in) |
| `__launch_bounds__(NTHR, (L == 1) ? 2 : 1)` l.424 | **OK.** Template-dependent arguments are allowed. L=1 is capped at 128 regs; L=8 at 255 regs, and 255 × 256 ≤ 64K, so the launch cannot fail on "too many resources". | — |
| lambdas `pe_load`/`issue` l.468-486 inside a `__global__` template | **OK.** A lambda defined inside device code is implicitly `__device__`. `--extended-lambda` is only for `__device__`/`__host__ __device__` lambdas written in host code and passed to kernels. Neither lambda captures the kernel parameter `p`; they capture only locals. | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda |
| `__ldg(p.cache_loc + tok)` l.537 on `const int64_t*` (= `const long*` on Linux) | **Very likely OK, unverified here.** The Programming Guide lists `long`/`unsigned long` among `__ldg`'s `T`. I found no in-repo precedent for `__ldg` on int64. If nvcc complains, the fix is `const int64_t row = p.cache_loc[tok];` (N1). | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#read-only-data-cache-load-function |
| constexpr scalars used in device code (`LOG2E`, `P_OFFSET`, `FULL`, `TILE`, …) | **OK.** Namespace-scope constexpr scalars may be used by value in device code. Nothing binds them to a reference: `max`/`min`/`fmaxf` are CUDA's by-value overloads, and there is no `using namespace std`. | Programming Guide, "constexpr variables" |
| torch's `-D__CUDA_NO_HALF*/BFLOAT16_CONVERSIONS__` | **OK.** Only intrinsics, raw structs and `reinterpret_cast` are used: `__floats2half2_rn`, `__bfloat1622float2`, `__floats2bfloat162_rn`, and copy-assignment of `__nv_bfloat162`. | review_k2a_v0 §1 (same conclusion for bf16) |
| `k1_main<L><<<grid, NTHR, smem_bytes<L>(), st>>>` l.734; shadowed `rowB` l.613 vs l.459 | **OK.** Shadowing is legal. | — |
| `CPP_SRC` declarations vs `CUDA_SRC` definitions | **OK.** The two match token for token (`k1_forward`: 12 tensors, 9 `int64_t`, `double`; `k1_occupancy(int64_t)`). `functions=[…]` generates the bindings. Python passes ints, a float and tensors. | l.96-101 vs l.739, 751-755 |
| `-gencode=arch=compute_90,code=sm_90` l.811 | **OK either way (N2).** If torch's JIT path honours user arch flags, there is one sm_90 compile. If it still appends its own (device or `TORCH_CUDA_ARCH_LIST`), you get extra compiles; the code is valid on sm_80+, and `#if K1_ARCH_OK` covers anything older. The only cost is compile time. Nothing uses sm_90a-only features. | — |

What the clang-plus-stubs check did not cover, and I covered by hand:
- **asm text.** Every asm string above matches CUTLASS precedent or the PTX grammar.
- **Real header signatures.** `__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t, __nv_fp8_interpretation_t) → __half2_raw` and
  `__nv_cvt_float2_to_fp8x2(float2, __nv_saturation_t, __nv_fp8_interpretation_t) → __nv_fp8x2_storage_t` match as
  used. The `(uint32_t)r.x` and `(uint16_t)q8` casts are on `unsigned short`.

**Compile time.** Two `k1_main` instantiations, three `tile_compute` instantiations, and two TUs that include
`<torch/extension.h>`. I expect about 1–2 min, far inside 15 min.

---

## 2. Runtime: bounds, plan, races, barriers, cp.async

### 2.1 Index audit (all 16 tests and the scored shapes)

Heads:
- tp8: Hq/Hkv = 16/2, HP = 1, rowB = 128 B.
- dp: 128/16, HP = 8, rowB = 1024 B.

Gx (l.879):
- tp8: 264 (L=1 at 2 CTAs/SM) or 132.
- dp: max(33 or 16, B+1) = 33 or 16.

| access | worst case | bound |
|---|---|---|
| page-table read `pt[absk >> pshift]` l.471 | only taken when `absk < seq`, so logical page < pages_per[b]. 1M test: page index ≤ 16383 < pt_stride 16385. | ✓ (−1 entries are never read) |
| cp.async src `kc + ((pe<<pshift)\|(absk&pmask))*rowB + 16ch` l.478-484 | 1M test: pe < 16385, so `pe<<6` ≤ 1,048,576 (int OK), row < 1,048,640, byte offset < 134,225,920 (`size_t`). B=4 at 320K (mixed ×4, page 1): rows < 1,280,032. | ✓ |
| cp.async dst `stg + lk*128 + 16*(ch^…)` and `+TILE*ROWB` | < 16,384 per stage; `ch ^ {0,4}` and `ch ^ {0,2,4,6}` stay in 0..7 | ✓; 16-B aligned |
| ring read l.323 `Kbuf + key*128 + 16*chunk`; `vld` l.308 | key < 64 (ring) or < 16 (new tile, NT=2, kw=0) | ✓ |
| `qs` reads l.556-564 | max word (64+48+7)·32+256+31 = 4095 < 4096 (L=8); 511 < 512 (L=1) | ✓ |
| new-token store l.538-539 | `cache_loc[tok]*rowB + head*64 + 2·lane`, head = 2hp+hh < Hkv | ✓; 2-B aligned |
| new tile l.543 | j < L ≤ 8 < NEWK = 16 | ✓ |
| epilogue scratch | L=8: ost is 32 KB plus 512 B of lse. L=1: 16 KB + 256 + 256 + 4 KB + 64 B. Both fit in the 64 KB ring, and `ost` is 16-B aligned (offset 16,896) for the float4 copy-out. | ✓ |
| partial write l.662-666 | pbase = hp·Gx + c < HP·Gx; `torch.empty` is sized exactly (l.880-881), with numel TORCH_CHECKed (l.770-771) | ✓ |
| combine reads l.687-698 | (hp·Gx + off + s)·ROWS + row with off + s < Σ S_b ≤ Gx | ✓ |
| gate / out l.710-715 | (Hq + 2Hkv)·64 + qh·64 + 2·lane; tok·Hq·64 + … | ✓ |

### 2.2 Split plan (l.241-296, l.449-455, l.683-686)

- **Upper bound always fits.** With `C = ceil(T/(Gx−B))`: Σ max(1, ⌈nt_b/C⌉) < T/C + B ≤ Gx. Every nt_b ≥ 1,
  because seq ≥ 1: `_lengths` clamps to ≥ 1 or 16.
- **Search.** If `hi = lo` because lo > ⌈T/ge⌉, monotonicity still gives cnt(lo) ≤ Gx. The loop always shrinks
  [lo, hi), so it terminates. It is warp-uniform: `cnt` is reduced over the warp before the branch.
- **Both kernels compute the same plan.** They run identical integer code on the same `seq_lens`, `B`, `W`, `Gx` and
  `min_chunk`, and nothing writes `seq_lens` in between. The combine finds `(S_b, off_b)` with `by_cta=false`, which
  uses the same scan.
- **Gx.** The launched grid (l.791) and the plan's Gx are the same variable.
- **Single wave.** Gx is sized for one wave, but correctness does not depend on co-residency: there is no inter-CTA
  communication.
- **Balanced boundaries.** `floor((s+1)nt/S) − floor(s·nt/S) ≥ floor(nt/S) ≥ 1` because S ≤ nt. No split is empty,
  and the emulator asserts this over 300 random plans.
- **Long contexts.** 1M test: C = 125, S = 132 at Gx 132, or C = 63, S = 261 at Gx 264. B = 33 (the layer problem
  later) runs two scan iterations; the carry is correct and the emulator covers B up to 69.

### 2.3 New-token path (l.515-545, 597-599): exactly once, never read back

- **One writer per request.** Only the last split (`s_idx == S_b − 1`, exactly one per request and HP) quantizes
  and stores the new tokens. The writes are byte-disjoint across HP CTAs (head·64) and across warps (kind, hh, j).
- **No read-back.** Every split, the last one included, reads old rows only, for positions `< seq` (l.471). The new
  rows are positions seq..seq+L−1. With page 64 they can share a physical page with old rows, but never a row: the
  partial last tile zero-fills slots ≥ seq without reading.
- **No ordering hazard.** No CTA reads a row another CTA writes in this call. The attention uses the exact bytes
  that `collect()` reads back.
- **Shared pages.** Pages are disjoint per request in `generate_input` (randperm). With prefix sharing, only old,
  read-only rows would be shared, so there is still no race.
- **Attended once.**
  - L=1: only warps 0 and 1 (kw = 0, one per head) process the 16-key new tile.
  - L=8: each warp owns disjoint rows and processes all 16 keys for them.
  - The mask `ak < seq+L && ak ≤ seq+j` keeps exactly keys seq..seq+j.
  - The zeroed slots (l.439, before barrier l.448; written after, then barrier l.546) are masked to −inf, and their
    V is 0.

### 2.4 cp.async ring and barriers

- **Group accounting.** The prologue commits 3 groups, some possibly empty. Before iteration i there are 3+i groups,
  and `wait_group 2` completes groups 0..i, so tile i is resident.
- **Slot reuse.** `issue(i+3)` targets slot (i−1)%4. It runs after the l.580 barrier, which every warp reaches only
  after finishing `tile_compute` on tile i−1, so there is no WAR hazard.
- **Short splits.** `ntl` < 3 works because empty groups complete trivially.
- **Drain.** `wait_group 0` plus the barrier (l.594-595) drain before the epilogue reuses the ring. No cp.async is
  outstanding then, because `issue` runs only when `nx < ntl`.
- **Barrier uniformity.**
  - Every `__syncthreads` (l.448, 546, 580, 595, 641, 660) is CTA-uniform: `ntl`, `last` and `b` come from smem
    after l.448. The `return` at l.450 is whole-CTA.
  - `tile_compute` has only warp shuffles, and it is called under warp-uniform conditions (l.597).
  - `plan_C`/`plan_find` run with all 32 lanes, in warp 0 or in the 32-thread combine block.

### 2.5 Uninitialized reads

- **Ring.** Every slot of an issued tile is written by a cp.async, as data or zero-fill:
  - `lk` = tid>>2 covers 64 keys;
  - `ch` = 2·part+u covers 8 chunks;
  - this holds for both K and V.
  - So there is no stale NaN for `p = 0 × V` to multiply. The −inf mask is applied to FP32 scores after the QK MMA,
    so no ±inf ever enters an MMA.
- **L=1 merge.** All 8 warps write `ow`, `mw` and `lw` before l.641. That includes warps whose key quarter was fully
  masked, which leave m = −inf, l = 0, o = 0. `Mu` handles the all-−inf case.
- **Partials.** The combine reads only `[off_b, off_b+S_b)`, and each of those CTAs writes all ROWS×64 values plus
  ROWS lse values (L=8: rows h·64 + r0 + g {+8} cover 0..127; L=1: 16 rows via the merge loop). Idle CTAs' slots are
  never read. This matches the emulator's NaN-poisoned partials.

### 2.6 Window and mask edges

- **Sliding, L=1.** `kstart = seq − 2047` is exactly the window start, so there is no masking except on the partial
  last tile.
- **Sliding, L=8.** Only keys `[kstart, seq+L−W)`, at most 7, need the window mask, and they all sit in the first
  tile. `need` (l.590) flags it, and the mask `ak > seq + j − W` is right.
- **`edges` tests.** The planted keys at p−2047 (in) and p−2048 (out) land correctly.

### 2.7 Sticky faults (N3)

The fallback cannot rescue an illegal address, a misaligned access or a hang: the CUDA context is dead and the run
fails at that test. I found no path to any of these. If the first check dies with
`CUDA error: an illegal memory access`, set `FALLBACK = True` with a new nonce to get a passing run, then bisect with
compute-sanitizer offline.

---

## 3. Numerics on GPU vs the CPU emulation

**Overflow audit: no FP16 overflow anywhere.**

| quantity | stored as | max magnitude | limit |
|---|---|---|---|
| Q (normalized, RoPE'd; ×5.473 is folded into `cl2`) | FP16 | √64 = 8 | 65504 |
| K, V from cache or new | e4m3 → FP16, exact | 448 (satfinite). e4m3 subnormals ≥ 2⁻⁹ are FP16 normals. | 65504 |
| QK scores | FP32 (mma accumulate) | 8·448·8 ≈ 2.9e4 | — |
| P = 2^(s·cl2 − m + 8) | FP16 | 256 · (1 + ~1 ulp) | 65504 |
| P·V products | inside the tensor core at full precision, FP32 accumulate | 256·448 per product, ≤ ~1e9 per split | FP32 |
| l (row sum) | FP32, summed from the FP32 p before FP16 rounding, as FA2 does | ≤ 256 · keys | — |

`V up to raw projection magnitudes` cannot reach the MMA unbounded: new V passes through
`x / v_scale → satfinite e4m3` first.

**P underflow and coherent tails (N4).**
- **Where P underflows.** FP16 P becomes subnormal below 2⁻²² of the running max (≈15 nats) and flushes below 2⁻³²
  (≈22 nats).
- **Scored full-layer distribution.** c2 ∈ (0.5, 0.62), so the per-key score σ ≈ 0.684·8·√(1−c2) ≈ 3.6 nats, and
  the max over 80K keys sits ≈ 15 nats above the mean.
- **Where the mass is.** The coherent tail (keys outside the top 2,048) carries ≈ 5 % of the mass, at ≈ max − 8 nats,
  where P is a normal FP16 number with full 11-bit precision. The mass below the flush point is ~1e-8.
- **Rules.** This is FP16 rounding, not skipping by weight: there is no branch on p. It is far finer than FA3's
  E4M3 P with offset 8, which goes subnormal below 2⁻¹⁴ relative and flushes below 2⁻¹⁷.

**GPU-only differences from the emulation.** Each one is at or below about 1e-6 relative:
- `rsqrtf` (≤ 2 ulp)
- `ex2.approx` (≤ 2⁻²² relative)
- `sincosf` vs torch `cos`/`sin`
- tensor-core accumulation order and truncation
- warp-tree summation

All of these are far below the 3–15 % bound. The emulator rounds Q and P with `.half()`, which is RNE like
`__floats2half2_rn`.

**RoPE (N5).**
- **The judge's reference.** It computes cos/sin on the fly (`reference.py:271-278`): `inv_freq` from
  `1/(theta ** (arange(0,64,2).float()/64))` on the data's device, then `ang = pos.float() * inv_freq`, then
  `ang.cos()`/`ang.sin()`. It does not use a table; only the SGLang *baseline* precomputes a 2²¹-row table
  (`baseline.py:72-79`).
- **The kernel matches it.**
  - `_inv_freq` (l.857) is the same torch expression on the same device, so its bits are identical.
  - `__fmul_rn((float)pos, invf)` is the same single-rounded FP32 angle.
  - Non-fast-math `sincosf` and torch's `cosf`/`sinf` are each within ~1–2 ulp of the true value.
- **Correction to design §6.** RoPE runs only on sliding layers. The 1,048,576-context test is a full layer, so it
  has no RoPE. The largest sliding position is ⌈80000·1.075⌉ + 7 = 86,007 (scored) or 20,007 (the "mixed" test).
  Both are under sincosf's fast-path range (|x| ≤ ~105,615), so there is no Payne-Hanek slow path or stack frame
  from it.

**Cache quantization.** The kernel and the reference both compute an IEEE `x / scale` (torch divides elementwise by a
0-dim CUDA tensor; nvcc defaults to `-prec-div=true`), then RNE to e4m3.
- The only divergence: for |x/scale| ≥ 480, torch produces NaN and `satfinite` produces 448.
- That is unreachable here (N6). It would need a unit-RMS key element ≥ 7.5 at kv_scale 1/64, or |randn| ≥ 7.5 for V.

**kv_scale 1/64.**
- k_scale is folded into `cl2` (l.571).
- v_scale is applied in the combine *before* the RMSNorm (l.703). This is needed, because eps = 1e-5 matters at
  1/64.
- New entries are divided by the scale (l.536).

---

## 4. Graph capture

- **No host syncs.** There is no `.item()`, `.cpu()` or synchronize on the K1 path. All lengths are read on device,
  and the plan is recomputed every replay from that replay's `seq_lens`.
- **Shape-only sizing.** Grids and workspaces depend only on (B, L, Hq, Hkv, SMs, occupancy). `gx` and `part_*` are
  sized by shape. `torch.empty` inside capture comes from the graph's private pool.
- **Stream.** `getCurrentCUDAStream()` is the capture stream. `C10_CUDA_KERNEL_LAUNCH_CHECK` (`cudaGetLastError`)
  is capture-safe.
- **Caches filled before capture.** `_SMS`, `_OCC[L]` for both L, and `_INV_FREQ` for θ ∈ {500000.0, 0.0} are all
  filled by the self-test's eight cases, which cover both L and both layer types. `cudaFuncSetAttribute` (a static
  flag per L) and the modules' lazy loading also happen eagerly in the self-test. A captured call therefore only does
  dictionary lookups and two launches.
- **The first call.** It needs an eager call before capture, which the baseline needs too: its `_cos_sin` builds a
  2²¹×64 table on the first call, and `FusedQKNorm` JIT-compiles Triton. So the judge almost certainly warms up
  eagerly. If it ever does not, see S3.

---

## 5. Rules

| rule | status |
|---|---|
| attend every key | **OK.** The window is `[p−2047, p]` exactly, or `[0, p]` on full layers. The emulator checks per-row key multisets, and mutation tests show the check fires. No key is skipped by weight. |
| full softmax | **OK.** Online softmax, combined exactly through lse. |
| FP8 cache stays FP8 | **OK.** Reads are e4m3 and dequantized in registers. Writes are e4m3 (satfinite RNE), at `cache_loc`. |
| queries no coarser than E4M3 | **OK.** FP16 is a strict superset of E4M3 in precision (10 vs 3 mantissa bits) and in range (65504 vs 448; minimum subnormal 2⁻²⁴ vs 2⁻⁹). |
| BF16 where SGLang keeps BF16 | **OK.** The attention output stays FP32 into the RMSNorm, where SGLang rounds it to BF16; FP32 is finer. The output is BF16. P is E4M3 in FA3's FP8 path; K1 keeps it in FP16, which is finer. |
| nothing cached from data | **OK.** The module state is `_STATE` (from the synthetic self-test), `_SMS`/`_OCC` (hardware), `_INV_FREQ`/`_ROPE` (per θ from config), and `_NORMS` (shape). |
| static checks | **OK.** The imports are `hashlib`, `time`, `typing` and `torch`, plus the SGLang symbols the baseline uses. There is no os/sys/environ/subprocess/inspect, no reference import and no monkeypatching. `load_inline` running nvcc is the sanctioned route (`llms.txt:96`). |
| nearby shapes | **OK.** The plan, grid and roles are generic; nothing is keyed to 80K or B ∈ {1, 16, 32}. |

---

## 6. Fallback

**When it works.**
- `_fa3_path` (l.928-982) with `R1_QK_FUSED_FP8_STAGE = False` is `baseline.py` op for op, so it is correct and
  exactly as graph-safe as the baseline (it needs the same eager warm-up).
- It covers a failed build (l.820-825), a failed or raising self-test (l.1147-1173), an unsupported shape, and a
  captured first call.
- `_torch_path` is used only if the SGLang imports fail. It is capture-safe but slow; it never triggers in the image.

**What it does not cover.**
- sticky faults and hangs (§2.7);
- a TORCH_CHECK in `k1_forward` on a judge input that `_supported` accepted (S5);
- a GPU-only K1 bug on a path the self-test does not exercise (S1).

The last two make the check *fail*, which is informative. With S1 and S5 applied, they fall back and log instead.

---

## Findings and exact fixes

**S1 (should-fix, 1 line): add a dp/L=1 self-test case.**
- The cases at l.1137-1146 cover dp only at L=8.
- Test 5 (`batch 8, ctx 8192, heads dp, full, spread mixed`, page 64, L=1) is the only path that combines HP = 8,
  Gx = 33, the L=1 role table and its 4-warp merge, and the self-test does not exercise it.
- Add, e.g., `(3, 1500, 1, False, 64, 1.0, 109, (128, 16)),` to `cases`.

**S2 (should-fix, 1 line): loosen the new-entry threshold at l.1167** from `kt < 1e-2 and vt < 1e-2` to
`kt < 3e-2 and vt < 3e-2`. The judge's bound is 8e-2.
- **Why it can fail.** Kernel and torch differ by about 1e-7 relative before rounding: `rsqrtf` and warp-tree sum
  vs torch's mean and rsqrt, and `sincosf` vs torch's cos/sin. So about 1e-5 of elements can land on the other side
  of an e4m3 rounding midpoint.
- **Why one flip is enough.** In the 128-element case `(1, 9, 1, …)`, one flip on an element of magnitude 1–2 costs
  0.125–0.25 / √128 = 1.1–2.2 %, which is over 1e-2.
- **Consequence.** Roughly a 1e-3 chance per run of silently running FA3 for the whole check.

**S3 (should-fix, 2 lines + optional): make a capture-first harness visible, and consider an import-time self-test.**
- **Current behaviour.** l.1184-1185 returns the fallback with no log, and keeps doing so for every captured call
  until an eager one arrives. The same silent outcome happens if the harness enables
  `torch.cuda.set_sync_debug_mode("error")`: the self-test's `synchronize` raises, is caught, and K1 is disabled.
- **Minimum fix.** Print once in that branch, e.g. `print("[k1] first call is under graph capture: FA3 until an eager call", flush=True)`.
- **Optional.** Run `_self_test(torch.device("cuda"))` at import, right after `_build()`, when
  `torch.cuda.is_available()`. There is no `prepare()`, and the lab's r0b file already does GPU work at import for
  this problem. Do this only after r0b-attn confirms that import-time CUDA init is fine in the judge, since a
  fork-after-init harness would break.

**S4 (should-fix, process): build-failure text.**
- With `verbose=True` (l.812), torch's `_run_ninja_build` sends ninja/nvcc output to raw fd 1 and raises
  `RuntimeError("Error building extension 'k1_v0_…'")` with no compiler text.
- So `str(_ex)[-12000:]` (l.823) shows one line, and the nvcc error is in the log only if the harness captures at
  the fd level, which is likely but unconfirmed.
- Option A: keep it as is, and look for nvcc errors above the `[k1]` line.
- Option B: `verbose=False`, so `error.output` carries the full compiler output into the exception. Then print
  registers and spills robustly through a 5-line host helper
  (`cudaFuncGetAttributes(&a, k1::k1_main<L>)` → `a.numRegs`, `a.localSizeBytes`) instead of relying on
  `-Xptxas=-v` lines.

**S5 (should-fix, 1 line): widen `_supported` (l.864-869)** so that inputs failing a TORCH_CHECK fall back instead
of raising inside a test. Add:

```
data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
data["k_scale"].dtype == torch.float32 and data["v_scale"].dtype == torch.float32
data["page_table"].dtype == torch.int32 and data["seq_lens"].dtype == torch.int32
data["cache_loc"].dtype == torch.int64
```

The judge's inputs satisfy all of these today (`reference.py:229-238`). This is insurance.

**S6 (perf hedge): add a route switch.** For example, `K1_ROUTE = {("sliding", 1), ("sliding", 8), ("full", 1), ("full", 8)}`,
tested in `kernel_fn` on `(("full" if cfg["window"] == 0 else "sliding"), cfg["q_len"])`.
- Design §6 already rates verify-full as at risk: it could be cvt-bound at about 2× memory time.
- Branching on config is allowed, and this lets check 2 isolate the loss without editing CUDA.

**Notes**

- **N1.** `__ldg(const int64_t*)` at l.537: see §1. The fallback edit is a plain load.
- **N2.** Arch flags: see §1. A doubled compile only costs time.
- **N3.** Sticky faults: see §2.7.
- **N4.** FP16 P underflow and coherent tails: see §3.
- **N5.** RoPE: see §3. Design §6's "1.05 M rad" premise is wrong but harmless.
- **N6.** Satfinite vs torch's NaN at ≥ 480 is unreachable here. K1's behaviour is the more robust one.
- **N7. Self-test cost.** Eight cases, CPU generation and an O(B·L) reference loop: a few seconds. It runs inside
  the first (eager, test) call, before `_k1(data)`.
- **N8. Combine plan recomputation.** B·L·Hq warps each rerun `plan_C`, a binary search of ≤ 5–10 steps over ≤ 32
  L1-resident `seq_lens`, for about 1 µs. It runs in parallel. It is a latency item, not a correctness one.
- **N9. Occupancy and carveout.** No carveout preference is set. The occupancy API and the launch use the same
  driver policy, and `_ctas_per_sm` caps at `K1_CPS = 2`, so Gx matches actual residency. If ptxas lands L=1 above
  128 regs it spills, and the kernel stays correct.
- **N10. Docs.** Design §5 says "that call goes to the fallback path". In fact *every* captured call does, until an
  eager call runs the self-test (S3).
- **N11. Combine one-wave.** At 32-thread blocks and 4,096 blocks max (B32·L8·Hq16), there are ≤ 32 blocks per SM,
  which is one wave on 132 SMs.
- **N12. `positions`, `cache_seqlens`, `cu_seqlens_q`, `scheduler_metadata` are unused.** RoPE uses
  `seq_lens + j`, which equals `positions` by construction (`reference.py:222`), and the reference itself uses
  `seq_lens` (l.302). Fine.

---

## Only a GPU run can settle these (first check)

1. The nvcc compile, especially the `__ldg(long)` overload (N1) and the real `cuda_fp8.hpp` signatures.
   - **Log:** `[k1] compiled in …`.
2. Registers and spills, read from the ptxas lines or the S4 helper:
   - `k1_main<1>` at ≤ 128 regs gives occupancy 2;
   - `k1_main<8>` should show occupancy 1 or 2.
   - **Log:** `[k1] k1_main<L> occupancy …`.
3. Eight (nine with S1) `[k1] self-test …` lines, with output rel ≲ 5e-3 and new K/V ≲ 1e-3.
4. Performance: the `cvt.rn.f16x2.e4m3x2` issue rate, which is the verify-full risk (S6), and random-row
   efficiency η.

## Submission advice

- Bump the nonce for every variant (identical code returns the earlier result).
- S1–S3 and S5 are Python-only, so the CUDA hash, and with it the build tag, is unchanged. S4 option B adds a host
  helper to CUDA_SRC and so triggers a rebuild.
- Consider a sibling check in the same wave with `FALLBACK = True` (nonce `k1-v0-fb`). It costs one check, and it
  separates "K1 broke the run" from "the harness or environment broke the run" if the first check fails.

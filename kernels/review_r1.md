# Review of the rung-1 files (nonce r1-1): independent reviewer, 2026-10-23, no GPU

Scope: `kernels/{muse_decode_layer,muse_moe_w4a16,muse_attention_decode}/r1_glue.py` and `kernels/rung1_levers.md`.
I compared each file to `problems/<p>/baseline.py`, checked it against the spec (TASK.md, reference.py, checks.py) and the
rules (llms.txt:104-110, 133-139; TASK.md "Rules"), and read every SGLang symbol it calls at a55ab7d5.
`S/` = `repos/sglang/python/sglang`.

## Verdicts

| file | verdict | blockers |
|---|---|---|
| `muse_decode_layer/r1_glue.py` | **SHIP to okc check** | none |
| `muse_moe_w4a16/r1_glue.py` | **SHIP to okc check** | none |
| `muse_attention_decode/r1_glue.py` | **SHIP to okc check** | none |

I found no blocker. `py_compile` passes on all three files. Every imported symbol exists with the signature used, and no
lever changes any rounding point relative to the baseline. Every listed test case either goes through a lever correctly
or falls back to the stock path. Below are the evidence and the non-blocking findings. Things that only a GPU can settle
are listed last.

---

## Per-lever verification (what I checked, and the result)

**L1: Triton post-expert norm at every T** (layer r1:317-334; MoE r1:140-168)
- The kernel `_post_expert_norm_weighted_sum_kernel` exists at `S/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py:10-60`.
- The r1 launch matches the wrapper's own ≥32-token launch (`:101-118`) argument for argument: the same pointers, int32
  ids, fp32 scores, bf16 `1+w`, constexprs, `num_warps=8` and `num_stages=2`. The Triton cache key is therefore
  identical, so T<32 reuses the binary that stock already compiles for the T≥32 scored shapes.
- `y` is contiguous, because it is a view of the head of `cache13`. The wrapper's `.contiguous()` calls are no-ops.
- Both paths compute `Σ s_k · rmsnorm(x_k·w)` in fp32 with a single bf16 rounding at the end. Dead slots: the torch path
  zeroes them (`:80-84`), and 0 normed with eps 1e-8 is 0. The kernel skips them (`:38-40`). Neither path reads Marlin's
  uninitialised remote rows.
- A token whose 8 slots are all remote gives a zero row on both paths, as the MoE reference requires.
- Saving: about 12 launches at T = 1, 8, 16. Real.

**L2: FusedQKNorm FP8 KV-store epilogue into staging rows** (layer r1:387-404; attention r1:126-144)
- `forward_triton(qkv, positions, cos_sin_cache, kv_store)` exists (`fused_qk_norm.py:304-311`). It is the method the
  baseline's dynamic dispatch reaches (`S/kernels/fused_op.py:556-580`, priority TRITON first, `:147`).
- Staging addressing:
  - `loc = 1..T` means the kernel's `loc != 0` mask (`:111-113`) never fires.
  - The kernel writes `kv_dst = loc·(hkv·64) + head·64` into a `[T+1, hkv·64]` buffer, so `k_stage[1:]` holds token t
    at row t.
  - The v heads are read from columns `[(hq+hkv)·64, (hq+2hkv)·64)`, which is the baseline's v slice. This holds for
    16/2 and for "dp" 128/16 (`reference.py:18`).
  - `_forward_triton` asserts contiguous caches (`:344`), which the staging tensors satisfy.
- The real-cache write is still `store_cache(..., data["cache_loc"], reserved_skip_index=-1)`. That keyword exists
  (`S/kernels/ops/kvcache/kvcache.py:81`), and -1 disables the skip (`kvcache.cuh:76`). Row 0 is therefore written when
  it is live.
- `store_cache` still gets matching shapes. Its matcher (`kvcache.cuh:115-143`) needs the same dtype everywhere, row
  stride 1, `B` rows and int32/int64 indices. `k_stage[1:]` is `[T,128]` uint8 at a 128-byte offset, which is aligned.
- **Rounding is unchanged; the double-rounding claim is correct.**
  - Stock: the kernel stores fp32→bf16 (RNE), then torch casts bf16→e4m3 (RNE).
  - Fused: the kernel rounds `x.to(bf16).to(f32)` (`:105-106`), clamps to ±448 (`:110`) and stores fp32→e4m3.
  - The Triton default float downcast being RTNE (lowered to `cvt.rn.satfinite` on sm_89 and later) comes from my memory
    of Triton's `semantic.cast`. It is **not verified in this image**.
  - In-repo corroboration: SGLang's own reference for this epilogue, `forward_native` → `_store_kv`
    (`fused_qk_norm.py:203-224`), defines the store as a clamp followed by torch's RNE `.to(fp8)`. The Triton path is
    meant to match it.
  - If Triton used RTZ instead, bytes could differ by 1 ulp (12.5%), which is near the 8e-2 cache tolerance. The r1-1
    check settles it: "new cache entries" fails if so.
  - No in-tree test exercises the FP8 epilogue: `bench_fused_qknorm_rope.py` has no `kv_store` or float8 cases.
  - A bf16 value is exact in fp32, so the two paths give identical bytes for |x| < 448.
  - v is the raw bf16 value on both paths.
  - CPU check over 1e6 values: identical in range. torch gives NaN only for bf16 |v| > 464 (464 itself rounds to 448;
    the doc's "≥ 464" is off by one, which does not matter).
  - |q|,|k| ≤ 8 after the per-head RMSNorm, so the clamp cannot fire. Queries stay at E4M3, as in the baseline.
- `kv_scale = 1/64` tests take the stock branch (`scaled` gate), as the kernel requires unit scales.
- PDL ordering (new adjacency QK → store_cache → FA3):
  - `store_cache` is PDL-launched (`kvcache.cuh:178-179`) and calls `PDLWaitPrimary` before its loads (`:65`), which
    waits for the QK grid to complete and its memory to be flushed.
  - FA3 follows `store_cache` directly. Upstream, the prepare kernel is a plain `<<<>>>` launch
    (`repos/flash-attention/hopper/flash_prepare_scheduler.cu:232`), the main kernel calls `wait_on_dependent_grids()`
    before loading Q/K/V (`flash_fwd_kernel_sm90.h:325`), and with `scheduler_metadata` set there is no PDL at all
    (`flash_fwd_launch_template.h:197`). This is safe; see N1 for the fork.
- Saving: 3 launches on every kv_scale = 1 shape. Real: the k cast, the v cast (non-contiguous slice) and the q cast are
  each one eager kernel.

**L3 / L3': local-id table / index_select** (layer r1:468-473 + 213-219; MoE r1:181-187)
- The `table[g+256s]` formula is exact for all 256 starts. CPU check, int32 in and int32 out:
  `/private/tmp/claude-501/-scratchscratch-2026-10-23-2c70b6/53b3e8c6-937d-4eba-ace1-232be3e3cd26/scratchpad/rev_l3_l5.py`.
- The table does not depend on `local_start`, which is an input. It is built in `prepare()`, so nothing is
  input-specialised there.
- `topk_ids` are int32 (`fused_muse_spark_routing.py:213-220`), and `index_select` accepts an int32 index.

**L4: align over 33 buckets** (layer r1:350-355; MoE r1:199-205)
- Wrapper (`S/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`):
  - At T ≥ 9, numel ≥ 72 > `SMALL_NUMEL_LIMIT` = 64 (`moe_align_small_numel.py:19`), and 33 ≤ 64 buckets, so the Triton
    path is skipped (`:114-128`).
  - The AOT `small_batch_expert_mode` runs when numel < 1024 (`moe_align_kernel.cu:387`); otherwise the generic
    align+sort runs. Both are bucket-agnostic.
- The -1 ids go to bucket 0, so their blocks come out with expert -1 (`.cu:329,356`). The padded total and the per-block
  experts are identical to the 257-bucket layout.
- The buffer bound `numel + 33·(bm−1)` is a correct upper bound.
- Marlin's per-call temporaries are unchanged for every tested and scored T:
  - `c_tmp = min(size_n·len(sorted), sms·4·bm·256)` (`moe_wna16_marlin.py:122-127`). The cap binds in both stock and r1:
    T=9 gives 4096·304 = 1.25M vs 1.08M, and T=32/33 also hit the cap.
  - `min_workspace` hits `sms·4` in both (`.cuh:1077-1079`).
  - Marlin bounds its block loop by the device padded total (`.cuh:68-69`).

**L5: unpooler mm into a zeroed output** (layer r1:477-485)
- `_bf16_gemm_dispatch_impl` is `F.linear` on H200 (`unquant.py:451-452`; the split-K, GEMV and CuTe paths are not
  initialised), so the GEMM is the same `mm`.
- The out view `(T,1024)` with strides `(8192,1)` passes ATen's cuBLAS result-layout test (row stride ≥ cols, or
  size(0) == 1), so there is no proxy copy.
- CPU: bit-equal to `F.pad(F.linear(...))` at T = 1 and 8.
- Saving: 1 launch at T ≤ 8. `constant_pad_nd` is a fill plus a copy. Real.

**Rule compliance (all files)**
- No `.item()`, `.cpu()` or sync in `kernel_fn`. Every branch is on config or shape, plus `is_current_stream_capturing`,
  which the baseline also uses.
- Allocations depend only on shape.
- `prepare()` builds only constants: the arange and the 256×256 table.
- No reference import, no harness access, no editing of loaded modules. The private kernel is imported and launched,
  never patched.
- `print` at import is explicitly allowed (TASK.md:120).
- Nothing is cached from `data`. The attention file's `_STAGE_LOC` is a constant arange keyed by device
  (attention r1:89-95), the same pattern as the baseline's `_cos_sin`.
- Every test case is covered:
  - ctx 10; batch 33 → T=33 (L4 small-batch, bm=16);
  - q_len 8 → T = 16, 24, 32, 40;
  - page 64 (`cache_loc` is generic);
  - heads dp (L2 widths parametric in hq/hkv);
  - kv_scale 1/64 → stock;
  - ctx 1048576 → positions < 2²¹;
  - outliers / edges / soft / coherent change only values, not lever paths.

---

## Findings (none blocking)

**N1 (note): FA3 now follows `store_cache` with no op in between.**
- Evidence: layer r1:401-442 and attention r1:141-184. `store_cache` triggers its dependents before its stores
  (`kvcache.cuh:74` vs `:79-80`).
- Upstream FA3 is safe, as shown under L2. The sgl-kernel build fetches the fork `sgl-attn@f89bc230`
  (`S/kernels/aot/CMakeLists.txt:86`), which is not in `repos/`, so this is UNVERIFIED.
- Mitigating evidence (an inference, not proof): in SGLang's FA3 decode, the sequence is
  `set_kv_buffer` (`S/srt/layers/attention/flashattention_backend.py:1990`), then `prepare_paged_mha_query`, which
  casts q only for a non-"auto" (FP8) cache (`:1375-1385`), then `flash_attn_with_kvcache(q=q.contiguous()...)`
  (`:2096` ff).
- Symptom if it is wrong: the output check fails intermittently while the new cache entries pass.
- Fix: none needed now. If that symptom appears, set `R1_QK_FUSED_FP8_STAGE = False`.

**N2 (should-fix before a *scored* submit, not before check): the docstrings overclaim.**
- Organizers read the top files, so claims should be accurate.
- decode_layer r1:3-4 says "no new compiled code with the default constants". In fact L2 compiles a new Triton
  specialisation (`HAS_KV_STORE`/`FP8_KV_STORE`) that the baseline never builds on sm_90.
- MoE r1:4-5 says every kernel is one "the baseline already runs". L4's CUDA small-batch align kernel and
  `index_select` are not.

**N3 (note): L3 has no modulo guard.**
- Evidence: layer r1:470-471. An id outside [0,256) (e.g. -1) would make `index_select` go out of bounds, causing a
  device assert that kills the run. Stock's `% 256` tolerates such an id.
- Not reachable: the routing only emits -1 under `HAS_NUM_VALID` (`fused_muse_spark_routing.py:118-128`), which this
  file never passes.
- No change needed.

**N4 (note): the savings are probably under-claimed for L3, and the cost model is optimistic.**
- ATen's `index` promotes an int32 index to int64 with a separate copy kernel. So the stock `expert_map[ids]` is
  probably 2 launches:
  - MoE L3' then saves 1 launch at every bs (the doc says 0).
  - The layer's L3 saves 2.
- Inside a graph, a removed tiny node is closer to 2–3 µs than 4. Plan the Δ table on the low end until the rung-1
  profile is in.

**N5 (note): dead L9 code in decode_layer** (r1:203-210, 307-311; `global _GEMV` in `prepare`)
- It is OFF and its signatures match (`S/kernels/ops/gemm/hopper_bf16_gemv.py:43,58`).
- It is harmless. If you want the smallest possible static-check surface, move it to the 1b file.

**N6 (note): the L6/L7 tables are keyed on exact shapes** (layer r1:112-115, 349, 425-428; attention r1:81, 166-169)
- Empty now. Once filled, nearby shapes fall back to stock, so they cannot be "much slower nearby", but the tables will
  read as per-shape tuning.
- When you fill them, prefer range buckets (e.g. `T >= 16`, `batch >= 16`).
- An L7 entry drops the judge's `scheduler_metadata` and costs FA3's prepare launch. That is allowed.

**N7 (note): the attention file needs one eager call before capture** (attention r1:89-95)
- If the first call is captured, `_stage_loc` adds one arange per replay. That costs speed only, not correctness.
- The baseline's `_cos_sin` relies on the same eager warm-up.

**N8 (note): compile time.**
- L2 adds a few Triton specialisations: rope and no-rope, times the `n_rows = 20T` and `kv_capacity = T+1`
  divisibility classes.
- That is seconds of compile, far inside 15 minutes.

---

## Only a GPU run can settle these (the r1-1 check will)
   - On failure, the log shows a Triton compile error at the first test: L2.
2. The fork FA3 PDL behaviour (N1).
3. Real per-launch savings and the `mm(out=strided)` cuBLAS path. Correctness is CPU-checked; speed is not.

## Submission advice
- Send all three r1-1 checks in one wave.
- For decode_layer, send a pre-staged **r1-2 = r1-1 with `R1_QK_FUSED_FP8_STAGE = False`** in the same wave. L2 is the
  only lever with both a new compile and a new PDL adjacency, so this isolates it without a second round-trip.
- Check runs are 20 per day per problem, so the extra variant is cheap.
- Bump the nonce on every variant (identical code returns the earlier result).

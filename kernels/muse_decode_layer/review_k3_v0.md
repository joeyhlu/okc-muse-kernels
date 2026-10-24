# Review of K3 v0 (`k3_v0.py` + two bisect variants): independent pre-`okc check` review of the composition

Reviewer: an independent agent. Nothing was compiled or run on a GPU (none here). I re-derived the byte-identity
claim myself, read every line of new glue against the layer reference and the three reviewed pieces, AST-scanned
the file, and re-ran the CPU emulator. Files: `k3_v0.py` (2,307 lines), `k3_v0_k2a_only.py`, `k3_v0_k1_only.py`,
`k3_v0.md`, `k3_emulate.py`. Spec: `problems/muse_decode_layer/{TASK.md,reference.py,checks.py,baseline.py}`. Rules:
`sources/okc_site/llms.txt`, `notes/study/01-challenge.md` §b–c.

## Verdict

| file | verdict | why |
|---|---|---|
| `k3_v0.py` (nonce `k3-v0-1`) | **SHIP to `okc check`** | No blocker. The plumbing matches the reference. Every one of the 13 tests and 12 scored shapes routes to K1 + K2a, and each piece falls back to r1 on its own. No rule violation found. Emulator ALL PASS (re-run here). |
| `k3_v0_k2a_only.py` (`USE_K1 = False`) | **SHIP to `okc check`** | It differs from `k3_v0.py` only in the nonce line and `USE_K1` (verified with `diff`). It has no import-time nvcc. |
| `k3_v0_k1_only.py` (`USE_K2A = False`) | **SHIP to `okc check`** | It differs only in the nonce line and `USE_K2A` (verified). It has no prepare-time nvcc. |
| `k3_v0.md` (doc) | accurate, two corrections | N4: the T8 row is stale. N5: it overstates what is unvalidated for `R1_LOCAL_ID_TABLE = False`. The §8 bisect paragraph is right; §4 here states it more precisely (`k2a_only` still compiles K2a in prepare). |
| `k3_emulate.py` (harness) | sound for testing the glue | The stand-ins re-apply K1's TORCH_CHECKs verbatim, assert the reviewed source text, `lo == 0` and `ng`, and NaN-poison kernel-written buffers. Re-run here: ALL PASS, and T8 bit-identical 3×. |

**Blockers: none.**

Should-fix before the first check (each cheap, Python only, no CUDA change):

| # | item | why |
|---|---|---|
| S1 | Put the compile times (and K1 occupancy) into the final `[k3]` summary line | It is the one line that is sure to sit in the 16,000-char log tail |
| S2 *(optional)* | One log line per config naming the paths taken | `_k1_supported`, `_k2a_supported` and `_k1_ready` fall back silently |

S3 (inherited K1 review S4) and S4 (time budget) are process notes. N1–N9 are notes.

---

## 1. Byte identity of the CUDA/C++ sources (check 1)

I AST-parsed every module-level string constant and compared the three k3 files with `k1_v0.py` and `k2a_v0.py`.
Separately, I compared the raw literal text with a regex on `^NAME = r"""…"""` (script:
`scratchpad/bytecheck.py`).

| source | sha1 (12) | k3_v0 | k2a_only | k1_only |
|---|---|---|---|---|
| K1 `CPP_SRC` | `abaea080a55b` | identical | identical | identical |
| K1 `CUDA_SRC` | `c3c056d9fc56` | identical | identical | identical |
| K2a `CPP_SRC` | `ae129455999b` | identical | identical | identical |
| K2a `CUDA_SRC` | `dce48c0dafab` | identical | identical | identical |

Both the AST values and the raw literals match. The `load_inline` names (`k1_v0_<sha1(cpp+cuda)[:10]>`,
`k2a_v0_<…>`) therefore equal the standalone ones.

**r1 parts are verbatim.**
- `diff` of `r1_glue.py` against k3 is clean for:
  - `_gemm` … `_marlin_bank` (k3 l.1962–2026);
  - `_post_expert_norm_weighted_sum` and `_experts` (l.2097–2155);
  - the levers block (l.142–162);
  - the constants and TypedDicts (l.166–243).
- `prepare` → `_r1_prepare` (l.2027–2074) differs only in the name and docstring.
- `_r1_attention` (l.2156–2223) is r1 `kernel_fn`'s attention block. Its only change is `o = …` → `return …`.
- `kernel_fn` (l.2235–2307) differs from r1 only in the two dispatch blocks (l.2242–2252 and l.2283–2292).

**K1 and K2a glue.**
- `_k1_*` are k1_v0.py's `_sms`/`_ctas_per_sm`/`_inv_freq`/`_k1`/`_mk_case`/`_ref_*`/`_self_test`. The intended
  changes:
  - `qkv` is passed in;
  - the cases come from `K1_SELF_TEST_CASES`;
  - `_k1_supported` adds review S5's dtype/contiguity checks plus `K1_ROUTE`.
- `_k2a_*` are k2a_v0.py's repack, `_build`, `_ref_fp32`, `_rel` and kernel launches. The changes:
  - `lo = 0`;
  - the weight names (`post_expert_norm`);
  - the `-1` id convention in the self-test inputs;
  - an alignment clone guard.

---

## 2. Plumbing vs the layer reference (check 2)

| item | reference (`problems/muse_decode_layer/reference.py`) | k3 | verdict |
|---|---|---|---|
| qkv split | `[Q 1024 \| K 128 \| V 128 \| gate 1024]` of `x @ qkv_gate.T` (l.362–363) | `qkv = _gemm(attn_input, qkv_gate)` [T, 2304] (l.2241), passed raw to K1 (l.2247). K1 reads q at `qh·64` (l.681), k at `(Hq+head)·64` and v at `(Hq+Hkv+head)·64` (l.697–699), and gate at `(Hq+2Hkv)·64 + qh·64` (l.885). Same columns as r1's `gate = qkv[:, hq*d + 2*hkv*d:]` (l.2161). | OK |
| positions | `seq_lens + arange(L)` (l.364) | K1 uses `seq + j` on device. `positions` is unused, and equal by construction (`generate_input` l.264). | OK |
| seq_lens / cache_loc / page_table / page_size | int32 [B] / int64 [T] / int32 [B, max_pages] / int | Passed as is (l.1069–1072). `page_shift = page_size.bit_length() - 1`, a power of 2 checked in `_k1_supported` (l.1041). `pt_stride = page_table.size(1)` in C++. | OK |
| kv scales | fp32 0-dim device tensors; `_quant(x/scale)`, `_dequant(x·scale)` | Passed as device tensors, read on device. kv_scale 1/64 goes through K1, the same as the attention problem's tested path. | OK |
| sliding vs full | `window` 2048/0, `rope_theta` 5e5/0 from config (l.304–305) | `kind = "sliding" if cfg["window"] else "full"` (l.1039). `use_rope = 1 if theta`. `inv_freq` comes from the same torch expression as the reference's `_rope`, prefilled for both θ (l.1227–1228). | OK |
| q_len 1 vs 8 | causal inside the block | `L` comes from `cfg["q_len"]`, checked to be in (1, 8). `k1_main<1>` or `<8>`. Both occupancy entries are prefilled (l.1225–1226). | OK |
| K1 replaces exactly QK-norm+RoPE+FP8 write+FA3+norm·gate | — | If K1 returns, `_r1_attention` is not called (l.2251), so none of `qk_norm`, `forward_triton`, `store_cache`, the q cast, FA3 or `o_norm_gate` runs. K1 does all of them. One writer per call: in the emulator, the only cache rows written are the new slots. | OK, nothing doubled or skipped |
| K1 exception path | — | The TORCH_CHECKs in `k1_forward` all run before any launch. The only work done before raising is `torch.empty` of part_o/part_lse/out, so a raise under capture records no kernel. r1 then runs. | OK (launch failure not covered, as the doc says) |
| o_proj input | `o [T, 1024]` (l.388) → `@ o.T` | K1 `out [B, L, 1024]` is token-major (`tok = b·L + j`). `.view(T, 1024)` (l.2247), then `_gemm(o, o_w)` (l.2253). | OK |
| experts input | `rms(h @ pooler.T)·(1+w)` [T, 4096] | `expert_x = state["pre_expert_norm"](expert_input)` [T, 4096] BF16 (l.2282). It is a fresh allocation, so the 16-B clone guard at l.1873 never fires. | OK (same rounding points as the baseline) |
| topk ids, `lo = 0` | `local = (ids - local_start) % 256`, local iff `< 32` (l.454, l.452) | `R1_LOCAL_ID_TABLE = True`: `table[g + 256 s]` gives local experts 0..31 and remote −1 (l.2277–2278). K2a route: `ids - lo == warp` with warp ∈ [0, 32) (CUDA l.1396, 1442), so −1 never matches. `if (e < 0 \|\| e >= 32)` marks the pair remote (l.1404) and finalize skips it (l.1700). | OK |
| `R1_LOCAL_ID_TABLE = False` | — | `(topk_ids - local_start) % 256` is int32 in [0, 256). With lo = 0, local ids are 0..31 and remote ones ≥ 32 → `e >= 32` → remote. This is **exactly the standalone K2a convention** (global ids, lo = 0), which the K2a self-test and `k2a_emulate` already cover, and `_k2a_self_test_inputs` switches to it automatically. The Marlin fallback `bank["expert_map"][local_ids]` maps 0..31 to themselves and the rest to −1 (r1 behaviour). Not run in the k3 harness. | OK |
| returned ids | the judge checks *your* global ids | `return …, topk_ids` (l.2307): the global FusedMuseSparkRouting ids, not `local_ids` | OK |
| weight names | MoE `post_norm_w` ↔ layer `post_expert_norm` | `nw = bf16(1 + post_expert_norm)` (l.1913), which is baseline's `post_expert_w`. Self-test map at l.1934–1936. | OK |
| W13 row order | both problems: rows `[:4096]` gate, `[4096:]` up; low nibble first. `nvfp4.py` is byte-identical between the two problems (md5 `ccb2e4ca…`). | `_k2a_row_perm(gate_up=True)` interleaves the same way in both | OK |
| per-expert global scales | layer: `gs·exp(N(0,1))` per expert (l.104–105). MoE problem: one `expand`ed constant. | `gsc13/gsc2 = gs[:32]·256` indexed by local e on device. Evidence: the self-test runs on the layer's real weights and passes in the emulator. | OK |
| where topk weights apply | `post_expert_norm(y)·w_k`, summed (l.455) | K2a finalize: `bf16(y_fp32)·nw → rsqrt(mean + 1e-8) · wts[tok, s]`, then a FP32 sum and a BF16 store. SGLang: Marlin with `mul_topk_weights=False`, weights applied in the post-norm weighted sum. Same point. | OK |
| token with no local expert | contributes a zero row | `acc = 0` → BF16 zeros → zero unpooler row | OK |
| T ≤ 8 vs T > 8 unpooler | slice of 1024 + zeros / full (l.474–478) | K2a output [T, 4096] BF16 contiguous is the same contract as r1's post-norm output. Unpooler branch unchanged (l.2293–2303). | OK |

---

## 3. Test-case routing (check 3)

`K1_ROUTE` holds all four `(kind, L)` classes. `K2A_MAX_TOKENS = 512`. Every case below satisfies `_k1_supported`:
- 16/2 heads, head_dim 64;
- page 1 or 64;
- FP8 contiguous caches;
- fp32 scales, int32 page table and seq_lens, int64 cache_loc.

Every case also satisfies `_k2a_supported` (32 local experts / top-8 / 256 experts).

| test | T | K1 class | K2a NG | note |
|---|---|---|---|---|
| b1 ctx10 sliding (page 64) | 1 | sliding,1 | 1 | |
| b1 ctx3000 full | 1 | full,1 | 1 | |
| b8 ctx2047 sliding | 8 | sliding,1 | 1 | T = 8: FP32 router, sliced unpooler |
| b9 ctx5000 sliding mixed p1 | 9 | sliding,1 | 1 | T = 9: split router, full unpooler |
| b3 ctx6000 full q8 outliers | 24 | full,8 | 2 | |
| b5 ctx5000 sliding q8 mixed p1 | 40 | sliding,8 | 2 | |
| b33 ctx20000 full mixed p1 | 33 | full,1 | 2 | Gx = max(132·cps, 34). Emulated as T33. |
| b2 ctx300000 sliding q8 (page 64) | 16 | sliding,8 | 1 | RoPE positions ≈ 300,000 (N1) |
| b1 ctx1048576 full q8 (page 64) | 8 | full,8 | 1 | no RoPE. Same shape as the attention problem's test. |
| b4 ctx7000 sliding q8 p1 kv_scale 1/64 | 32 | sliding,8 | 2 | K1's scaled path, in the K1 self-test |
| b1 ctx80000 full soft p1 | 1 | full,1 | 1 | `scheduler_metadata` ignored by K1 |
| b4 ctx6000 sliding q8 soft (page 64) | 32 | sliding,8 | 2 | |
| b16 ctx20000 full mixed soft p1 | 16 | full,1 | 1 | |
| scored: T = 1/16/32/8/128/256 | | all four classes | 1/1/2/1/4/4 | NG 4 is covered by self-test bs 100/128/257 |

None of these takes a silent fallback by design. If one does anyway (a dtype surprise in the judge's own copy of
`generate_input`), it happens with no log line. See S2.

---

## 4. Time budget (check 4)

**Where the compiles run.**
- K1's nvcc runs at **import** (l.992–999).
- K2a's nvcc runs **inside `prepare()`** (l.1922).
- The SGLang JIT kernels (`FusedPostNormResidualPreNorm`, `FusedRMSNormGate`, `FusedMuseSparkRouting`, …) load
  lazily at their first forward, i.e. at the first test call, not in `prepare()`. I checked
  `sglang/kernels/ops/muse_spark_v12/fused_norm_gate.py:233` and `fused_muse_spark_routing.py:907`. The baseline pays
  for them in every submission.

**`prepare()` arithmetic (cold H200).** Measured on CPU in the emulator: repack 21 s, everything else about 4 s
excluding the emulated kernels.

| step | estimate |
|---|---|
| `_r1_prepare`: 512 MB cos/sin table, Marlin repack of 32 experts, sync | 5–15 s |
| `_k1_prepare`: 9 synthetic cases, CPU case generation up to ~7 M randn, FP32 ref loops ≤ 24 einsums each | 3–10 s |
| K2a repack: 32 experts, ≤ 300 MB int64 temporaries per expert | 2–5 s |
| K2a nvcc: 6 `gemm_kernel` instances + route + final, plus the pybind host TU in parallel. Review estimate. | **60–180 s** |
| K2a self-test: 6 cases, FP32 dequant reference over ≤ 32 experts | 3–10 s |
| **total, cold** | **~75–220 s** (cached build dir: ~20–40 s) |

**The guard.**
- `K2A_SELF_TEST_START_BY_S = 250` is measured from `t0` at `prepare()` entry (l.2079 → l.1930). That is the right
  start *if* the harness's 300 s covers only the `prepare()` call. Then it is the correct budget.
- Its consequence by design: if nvcc ends after 250 s, K2a is **disabled** (Marlin) rather than run untested.
- Nothing bounds nvcc itself. Prepare exceeds 300 s only if nvcc alone takes about 270 s or more, more than the
  review's upper estimate.
- If the harness also charges import time to the 300 s, the guard measures from the wrong start. The worst case
  becomes K1 nvcc (60–120 s) + the above, which can pass 300 s.

**Cached vs cold.**
- `load_inline` names include the source sha1, so a surviving `~/.cache/torch_extensions` gives
  "compiled in ~1 s" for both. A fresh container is cold. Expect cold.
- A stale `lock` file left in a persistent build directory by a killed earlier build makes torch's `FileBaton` wait
  forever. Only possible if the cache persists across runs; low risk.

**15-minute run.**
- The run is: the judge's `weights_fields` router search, both prepares, baseline JIT compiles, 13 tests each with
  2× `ref_front` + 2× `ref_back`, and 12 timed shapes, each repetition checked against the reference.
- k3 adds two cold nvcc builds (about 2–5 min in total) to whatever r0/r1 take.
- Nobody has measured the base duration yet. k3 is the most compile-heavy file in the wave. **Send r0/r1 in the same
  wave**, as the doc says, so the base duration is known.

**Variant bisect, made precise.**
- `k3_v0_k2a_only` removes only the **import-time** K1 compile, which helps the 15-min budget. It still compiles K2a
  in `prepare()`, so it does **not** bisect a 300-s prepare timeout, unless import is charged to prepare.
- `k3_v0_k1_only` removes the **prepare-time** compile.
- k3_v0.md §8 says this correctly. The point to remember: a prepare timeout in both `k3_v0` and `k2a_only` means K2a's
  nvcc itself is too slow.

**Second-iteration option, only if the first log shows prepare near 300 s.** Move `_k2a_build()` to import, next to
K1's, and pass the same `-gencode=arch=compute_90,code=sm_90` that K1 passes.
- torch's `_get_cuda_arch_flags` returns `[]` as soon as any user flag contains `"arch"`, so there is no duplicate
  target. I checked this in this Mac's CPU torch 2.5.1 (`cpp_extension.py`, `_get_cuda_arch_flags`), **not** in
  the image's torch. The behaviour has been stable across torch 2.x, and K1's own build already relies on it (comment
  at l.983–984). Confirm it on the first log: K1's nvcc command line should show only the one `-gencode`.
- That pin also keeps import from calling `torch.cuda.get_device_capability()`, so no CUDA context is created at
  import. This corrects the K2a review's §1 S1 caution against a manual `-gencode`.
- Not needed for the first check.

**Fork safety (retires the K1 review S3 caveat for this file).** Import runs no `torch.cuda.*` call. `_k1_build`
calls only `load_inline`, and its `-gencode` flag stops torch from querying the device for arch flags. No context
exists before `prepare()`.

---

## 5. Log tail (check 5)

**Where the build output goes.** Verified in torch's `cpp_extension.py` (`_run_ninja_build`, l.2068–2110):
- With `verbose=True`, ninja's `-v` command lines and all nvcc/ptxas output go to **raw fd 1** (stdout).
- torch's own "Using … / Emitting … / Building … / Loading …" lines go to **stderr**.

**Rough volumes.**
- Each build prints 3 nvcc/g++ command lines (≈ 1–1.5 KB each) plus `-Xptxas=-v` blocks (≈ 4 lines per entry: 3
  entries for K1, 8 for K2a). That is about 5–8 KB per build, before any header warnings.
- The prepare tail after K2a's build is about 1.5 KB: 6 self-test lines, `prepare done` and the summary.

**What the 16,000-char tail keeps, in order.**
- The ptxas text all comes **before** the `[k3] attention=… experts=…; prepare took …s` line (l.2092–2093). It can
  only push out things printed earlier than it:
  - `[k1] compiled in`;
  - K1's ptxas lines;
  - the 9 `[k1] self-test` lines;
  - `[k2a] repacked in`.
- The summary line is threatened only by what the harness prints *after* prepare. That is the test and timing report,
  plus any `[k3] … raised at call time` line. Its size is unknown, likely a few KB. baseline.py prints nothing.
- Expect the summary line, the K2a self-test lines and most of K2a's ptxas block to survive.
- Expect K1's compile time and register/spill lines to be lost.

**S1 (should-fix, minimal): make the summary line self-sufficient.** Do not switch to `verbose=False` for this. That
would drop the ptxas register/spill lines the reviews asked you to read (K2a S3, K1 N2). Exact patch:

```python
# l.992-999: record the K1 build time
_K1_MOD = None
_K3_BUILD_S = {}                      # piece -> nvcc seconds (log only; not derived from data)
if USE_K1:
    _t_build = time.time()
    try:
        _K1_MOD = _k1_build()
        _K3_BUILD_S["k1"] = time.time() - _t_build
    except Exception as _ex:  # noqa: BLE001
        ...unchanged...

# l.1921-1922 in _k2a_prepare: record the K2a build time
    t_b = time.time()
    try:
        st["mod"] = _k2a_build()
        _K3_BUILD_S["k2a"] = time.time() - t_b
    except ...unchanged...

# l.2092-2093 in prepare(): one line with everything
    occ = "/".join(str(_K1_OCC.get((L, str(device)), "-")) for L in (1, 8))
    builds = " ".join(f"{k}={v:.0f}s" for k, v in _K3_BUILD_S.items()) or "none"
    print(f"[k3] attention={'K1' if state['k1_ok'] else 'r1-FA3'} experts={'K2A' if state['k2a'] is not None else 'r1-Marlin'}"
          f" (USE_K1={USE_K1} USE_K2A={USE_K2A}); nvcc {builds}; k1 CTAs/SM L1/L8 {occ};"
          f" prepare took {time.time() - t0:.1f}s", flush=True)
```

Optionally, also carry the worst self-test `rel/row` of each piece into the line: track a max in `_k1_self_test`
and in the K2a loop. Either way, the line stays under 300 chars.

**S2 (optional, diagnostic): log silent fallbacks once per config.** `_k1_supported`, `_k1_ready` and
`_k2a_supported` returning False fall back with no line. The proposed line names the paths for each test and timed
config, and it lands in the tail between the harness's own lines.
- Keying a logging set on `config` is a branch on config, which is allowed. It caches nothing used in the
  computation.
- Sketch:
  - before l.2242: `k1_used = k2a_used = False`;
  - set `k1_used = True` after l.2247;
  - set `k2a_used = True` after l.2287;
  - before l.2307:

```python
    key = (cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"])
    if key not in _K3_SEEN:            # module-level set(); prints a string only, no tensor values
        _K3_SEEN.add(key)
        print(f"[k3] B={key[0]} L={key[1]} {'sliding' if key[2] else 'full'} page={key[3]}:"
              f" attention={'K1' if k1_used else 'r1'} experts={'K2A' if k2a_used else 'r1'}", flush=True)
```

This is about 25 lines per run in total. Printing a Python string under capture is not a sync.

**S3 (inherited, K1 review S4; note): build-failure text.**
- With `verbose=True`, a failed build raises `RuntimeError("Error building extension '…'")` and carries **no
  compiler text**. So the `[k3] … build FAILED … Compiler output tail:` lines (l.997–998, l.1924–1925) print one line.
- The nvcc errors are in fd 1 above that line, and can be long enough to push out the head of the log.
- Acceptable for check 1. If a build fails, re-check with `verbose=False` for that piece. The exception then carries
  the full output, and the existing `str(ex)[-12000:]` prints it.

---

## 6. Rules (check 6)

From an AST scan of `k3_v0.py` (names, attributes, calls), plus reading:
- **Banned names or imports: none.** No os/sys/subprocess/threading/environ/eval/exec/open/importlib/inspect, and no
  `reference`/`checks` import.
- **Imports.** hashlib, time, types, torch, triton, sgl_kernel, sglang, and `torch.utils.cpp_extension.load_inline`
  (the sanctioned nvcc route).
- **No arch env write.** K2a's review-S1 fix is in the verbatim source (no `os.environ`). The image sets
  `TORCH_CUDA_ARCH_LIST` only as a Docker build `ARG` (`repos/sglang/docker/muse_spark/muse-spark.Dockerfile:11`; no
  `ENV` in either Dockerfile), so K2a's auto-detected arch is sm_90.
- **Host syncs.** `.tolist()`, `torch.cuda.synchronize`, and `bool()`/`float()`/`int()` on tensors appear only in
  prepare-time code: the self-tests, the repack's NaN/neg/zero checks, and r1's final sync. The `int()`/`float()` hits
  in `kernel_fn`'s call tree are on Python config values:
  - `float(cfg[...])` l.1052, l.1073;
  - `int(_K1_MOD.k1_occupancy(L))` l.1018, which is prefilled and so never reached in `kernel_fn`.
- **Module-level state.** `_K1_SMS` (device), `_K1_OCC` (L, device), `_K1_INV_FREQ` (θ, device) and `_K3_WARNED`
  (piece name) are constants keyed by config or device. Nothing is keyed by, or derived from, input values.
- **`_k3_disable`** mutates `state` only after an exception that a shape or dtype check raised, never a value.
- **Capturability.** Branches are on config, dtype, contiguity, `is_current_stream_capturing()`, and one pointer
  property (`data_ptr() % 16`, fixed at capture). Allocations depend on shape only. Nothing is read from a device
  tensor. K1's `cudaFuncSetAttribute`/occupancy run only in prepare: the static `attr_set` guard, with the self-test
  covering L = 1 and L = 8.
- **No input-dependent work in prepare.** Prepare touches weights and synthetic seeded inputs only, through explicit
  generators, so the global RNG is untouched.
- **Precision.**
  - K2a keeps the weights as NVFP4: a lossless repack and scale re-encode (K2a review).
  - K1 uses FP16 Q/P, finer than the E4M3 the rules allow.
  - New K/V go fp32 → e4m3 in one rounding.
  - Activations at SGLang's BF16 points or finer.
- **Not checked:** submission size limits (109 KB, nothing published for kernel files) and the judge's static-check
  heuristics beyond the published rule text.

---

## 7. CPU emulation (check 7)

`python3 -I k3_emulate.py main` ran unmodified, in 257 s: **ALL PASS**, exit 0.
- All 9 K1 self-test cases pass: out rel 1.69–1.77e-3, new K/V bit-exact.
- All 6 K2a self-test cases pass: rel 3.73–3.76e-3, worst row ≤ 3.98e-3.
- Prepare: `[k3] attention=K1 experts=K2A`.
- Every layer case passes with paths K1/K2A, no writes outside the new slots, and correct shapes and dtypes. Calls 2–6
  ran under simulated capture.

| case | upd err [bound] | hid err [bound] | k3_v0.md table |
|---|---|---|---|
| T1 sliding ctx10 page64 | 1.345e-2 [0.103] | 8.69e-3 [0.064] | same |
| T1 full ctx3000 page1 soft | 7.04e-3 [0.150] | 5.19e-3 [0.103] | same |
| T8 sliding L8 ctx2500 page1 soft | **1.051e-2** [0.130] | **7.52e-3** [0.090] | doc says 1.11e-2 / 7.93e-3 (N4) |
| T16 sliding L8 page1 kv1/64 ls96 | 7.81e-3 [0.150] | 5.47e-3 [0.105] | same |
| T24 full L8 page64 outliers | 7.55e-3 [0.150] | 4.36e-3 [0.080] | same |
| T33 full L1 page1 mixed | 7.83e-3 [0.150] | 5.62e-3 [0.106] | same |

**Determinism re-run.** I also ran the T8 case 3× in one process (eager, then twice under capture), with the shortened
self-test lists. The result was bit-identical each time: 1.051e-2 / 7.519e-3. No non-determinism reproduced. This
also bears on the doc's "unexplained bs=40 deviation" footnote: no repro here either.

I read the stand-ins that matter for this review:
- `FakeK1.k1_forward` re-applies every TORCH_CHECK verbatim. It rebuilds the call only from the arguments and checks
  key coverage.
- `FakeK2A` asserts `lo == 0` and `ng == ng_for_bs(M)`, and NaN-poisons the kernel-written buffers.
- `fake_load_inline` asserts the reviewed sources.

So the emulation tests the glue, not a copy of the data.

---

## 8. Notes

- **N1. ctx 300,000 sliding test: a K1 input range no review covered.**
  - `review_k1_v0.md` l.234–236 says the largest sliding position is 86,007, so `sincosf` stays in its fast path
    (|x| ≲ 105,615). That holds for the attention problem's tests. The **layer** test `batch 2, ctx 300000, sliding,
    q_len 8` has RoPE angles up to ~3.0e5 rad, so `sincosf` (l.393) takes its Payne–Hanek path.
  - Correctness is unaffected. Both sides form the same single-rounded fp32 angle (l.391 `__fmul_rn`; the reference's
    `pos.float() * inv_freq`), and both reduce it accurately. Speed does not matter: this is a test, not a timed shape.
  - It is the first time K1's RoPE sees a large argument on hardware. The K1 self-test's largest sliding position is
    about 4,200. Read that test's line in the first log.
- **N2. K2a has no host TORCH_CHECKs on route/finalize.** K2a review S2 was not applied; only `gemm`'s mode/ng
  checks exist (l.1765–1766).
  - So the "host-side exception → `_k3_disable`" fallback (k3_v0.md §2) is effectively a K1 feature.
  - For K2a, misuse would fault rather than raise. The layer builds every K2a input itself with fixed dtypes (int32
    ids, fp32 weights, fresh BF16 x), so the risk is low.
  - Adding S2's checks would change `K2A_CUDA_SRC` and therefore the reviewed hash. Not for this round.
- **N3. The K1 self-test includes two `heads (128, 16)` cases the layer never runs** (l.264–265).
  - That is conservative: a dp-heads-only GPU bug would turn K1 off for the layer.
  - If the first log shows `[k3] K1 self-test FAILED` on one of those two cases only, drop them from
    `K1_SELF_TEST_CASES` and re-check.
- **N4. k3_v0.md §6 table, T8 row, is stale.** It says 1.11e-2 / 7.93e-3. The current file and emulator give
  1.051e-2 / 7.519e-3, reproducibly. Doc-only; the conclusion is unchanged.
- **N5. k3_v0.md §6 "not validated: `R1_LOCAL_ID_TABLE = False`".** That convention (ids in [0, 256), lo = 0, local
  0..31) is the standalone K2a's own tested convention. Only the composition harness did not run it. See §2.
- **N6. Silent paths, for the record.** Each of these sends a piece to r1 with **no** log line:
  - `_k1_ready` (only possible if a constant were missing; all are prefilled);
  - `_k1_supported` returning False;
  - `_k2a_supported` returning False.

  None is expected on the judge's inputs. S2 makes any of them visible.
- **N7. Faults are still fatal** (unchanged from the piece reviews): a sticky device fault in a self-test or a timed
  call, a hang, or a launch error under capture after r1 work was recorded. The fallback design cannot catch these.
  The variants bisect them.
- **N8. Memory** is resident and fine on H200: the 512 MB cos/sin table, Marlin bank, 0.86 GB K2a repack, and
  per-call K2a `y` ≤ 256 MB at T = 512.
- **N9. Dedupe.** The three files have distinct nonce lines and an identical line 1 `# okc: problem=muse_decode_layer`.

---

## 9. What to read in the first check log (adds to k3_v0.md §8)

1. `[k3] attention=K1 experts=K2A …; nvcc k1=…s k2a=…s; …; prepare took …s`, if S1 is applied.
   - Prepare above ~200 s: K2a's nvcc is the cost. See §4 for the second-iteration option.
2. Any `[k3] … FAILED` or `[k3] … raised at call time` line.
3. The `batch 2, ctx 300000, sliding, q_len 8` test result (N1).
4. If S2 is applied: one path line per config. Every one should read `attention=K1 experts=K2A`.
5. If `k3_v0` times out and `k3_v0_k1_only` passes: the K2a compile is the problem (prepare or 15 min). If
   `k3_v0_k2a_only` also passes, it was the combined 15-min budget.

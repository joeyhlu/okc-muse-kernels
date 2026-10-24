# K1: fused attention decode/verify kernel, v0 design (`k1_v0.py`)

Status: the CUDA is written, py_compiled, and clang-syntax-checked (sm_90 device and host passes, against stub
headers). The decomposition is CPU-emulated and **passes**: 21/21 cases against a copy of the judge's reference,
plus exact per-row key coverage and fragment/bank tests (`k1_emulate.py`). Nothing has run on a GPU yet. Every
µs figure below is a model, not a measurement.

## 1. What runs per call

Two launches. Both have fixed grids derived from shapes and config only. Every per-request length is read on
device. There are no host syncs and no allocation that depends on the inputs.

| launch | grid × block | work |
|---|---|---|
| `k1_main<L>` | (Gx, Hkv/2) × 256 | QK-norm (+RoPE) of Q. The new K/V go through norm and RoPE, become E4M3, are written to the cache, and are attended from smem. Then split-KV attention over the paged FP8 cache produces FP32 partials |
| `k1_combine` | B·L·Hq × 32 | merges the splits, then ×v_scale, RMSNorm, ×sigmoid(gate), and writes BF16 out |

For comparison, the stock path is 7–9 launches: FusedQKNorm, k cast, v cast, store_cache, q cast, FA3 prepare
(when no metadata is given), FA3 main, FA3 combine and rms_norm_gate. The sliding shapes have weight 46 and are
latency-bound, so going from 7–9 launches to 2 is where most of the gain comes from.

### 1.1 Split plan (on device; identical integer math in both kernels)
- **Old keys per request.** Request b has old keys `[kstart, seq)`, with `kstart = max(0, seq − 2047)` on
  sliding layers and 0 on full ones. They are cut into `nt_b = ceil(n_old/64)` tiles of 64 keys.
- **Per-split cap C.** C is the smallest cap such that `Σ_b max(1, ceil(nt_b/C)) ≤ Gx`.
  - It is found by binary search inside `[max(MIN_CHUNK=2, ⌈T/Gx⌉), ⌈T/(Gx−B)⌉]`.
  - The upper end always fits, because Gx > B.
  - The search takes 0 iterations at B=1, 1 for sliding B=32, and ≤5 for full B=32.
- **Splits.** `S_b = max(1, ceil(nt_b/C))`. Split s covers tiles `[s·nt_b/S_b, (s+1)·nt_b/S_b)`, so the splits
  are balanced and each has ≤ C tiles.
- **Grid size.** `Gx = max(SMs × min(2, occupancy(k1_main<L>)) / (Hkv/2), B+1)`. The occupancy is queried once,
  eagerly, through `k1_occupancy`, so the grid is always a single wave. CTAs with `c ≥ Σ S_b` exit immediately.
- **Partial indexing.** Partials are indexed by `(hp, c)`. The combine recomputes `off_b` and `S_b` with the
  same function.
- **Plans for the scored shapes** (`k1_emulate.scored_plan_stats`; lengths uniform in 80K ± 7.5 %):

| shape | Gx (occ 2 / 1) | CTAs | tiles per CTA, max / mean / ideal |
|---|---|---|---|
| sliding B=1 | 264 | 16 | 2 / 2 / – (MIN_CHUNK) |
| sliding B=16 | 264 / 132 | 256 / 128 | 2 / 2.0 / 1.9 ; 4 / 4.0 / 3.9 |
| sliding B=32 | 264 / 132 | 256 / 128 | 4 / 4.0 / 3.9 ; 8 / 8.0 / 7.8 |
| full B=1 | 264 / 132 | 250 / 125 | 5 / 5.0 / 4.7 ; 10 / 10 / 9.5 |
| full B=16 | 264 / 132 | 262 / 132 | 78 / 76.0 / 75.4 ; 162 / 150.8 / 150.8 |
| full B=32 | 264 / 132 | 263 / 130 | 162 / 151.9 / 151.3 ; 332 / 307 / 303 |

The first plan version used fixed-size chunks and left full B=32 at 173 max against 151 ideal, 14 % imbalance.
The balanced boundaries bring this to 7 %.

### 1.2 One CTA = (split of one request, one KV-head pair)
- **Full cache rows.** Both KV heads of a row are adjacent, `[Hkv][64]` e4m3. So each page-size-1 key costs one
  full 128 B K row slice plus one 128 B V slice. FA3 reads two separate 64 B halves from different CTAs.
- **Pipeline.** 4-stage `cp.async.cg` ring of 64 keys per stage (8 KB K + 8 KB V).
  - Thread `tid` handles key `tid>>2` and chunks `2(tid&3)`, `2(tid&3)+1`.
  - Slots past the range get `src_size = 0`, i.e. zero-fill, so no stale NaN can reach the MMA.
  - Page-table entries are prefetched two tiles ahead into registers.
  - The page index is `page_table[b][abs >> shift]`, with `shift = log2(page_size)`, so page 64 is handled too.
- **Q prep.** All 16·L query rows of the head pair get RMSNorm in FP32. Sliding layers then get interleaved RoPE:
  the angle is `__fmul_rn(float(pos), inv_freq[i])` and `sincosf` is the accurate version. `inv_freq` is the
  torch-computed [32] table. The rows are stored as FP16 in smem. The ×5.473/8 factor and k_scale are folded into
  the softmax constant `cl2 = qk_scale_factor·sm_scale·k_scale·log2 e`. Every split recomputes Q, which costs
  ≤ 16 KB of L2 reads and is negligible.
- **New tokens.** This is the ordering the brief asked for.
  - Only the **last split** of each request handles them. It normalizes, ropes and quantizes the L new keys and
    quantizes the L new values. Quantization is fp32 IEEE `x / scale`, then `cvt.rn.satfinite.e4m3x2`.
  - It stores them at `cache_loc` (16-bit stores, coalesced per warp).
  - It writes the same bytes into a zeroed 16-key smem tile, using the same swizzle as the ring.
  - After the old tiles it attends that tile, with the causal mask `key ≤ seq + j`.
  - No CTA ever reads a cache row being written in this call: other splits only read rows `< seq`. So there is
    no inter-CTA ordering hazard, and the attention uses exactly the bytes the judge reads back.
- **Warp roles.**
  - **L=1:** warp w = (KV head `w&1`, key quarter `w>>1` of every 64-key tile). Each mma covers 8 real rows
    (8 q heads) and 8 padding rows. The 4 warps of a head are merged in smem at the end.
  - **L=8:** warp w = (KV head `w>>2`, m-tile `w&3` = 16 rows = tokens 2m and 2m+1 × 8 q heads). Each row is
    owned by exactly one warp.
- **Softmax.**
  - Online softmax in log2 units. The row max is kept in raw-dot units and reduced over the quad.
  - `p = exp2(s·cl2 − m + 8) ∈ (0, 256]`. The offset keeps FP16 P normal down to 2.4e-7 of the row max, which
    covers the coherent long tail.
  - Masked scores are −inf. The guard `m == −inf → use 0` avoids −inf − (−inf).
  - Masking is computed only on flagged tiles: a partial last tile, or the first sliding tile at L=8. The
    emulator asserts that unflagged tiles have nothing masked.
- **Epilogue.** The partial is `O/l` (FP32) plus `lse = m + log2 l − 8`, staged through the drained ring and
  written as float4.

### 1.3 Combine
- One warp per (b, token, q head).
- Max of the lse values, then `Σ 2^(lse−max)·O`, then ×v_scale. v_scale has to be applied before the RMSNorm
  because eps = 1e-5 is not negligible at a scale of 1/64.
- Then RMSNorm over 64 dims, ×sigmoid(gate) in FP32, and BF16 out.
- If all lse values are −inf the output row is zeros. This cannot happen: the new tile always has the token's
  own key.

### 1.4 Shared memory, registers, occupancy

| | L=1 | L=8 |
|---|---|---|
| ring (4 × 16 KB) | 64 KB | 64 KB |
| new tile (K + V, 16 keys) | 4 KB | 4 KB |
| Q (FP16) | 2 KB | 16 KB |
| total dyn smem | 71.7 KB | 86.1 KB |
| `__launch_bounds__` | (256, 2) → ≤128 regs, 2 CTAs/SM | (256, 1) → ≤255 regs; occupancy queried |
| epilogue scratch (aliases ring) | 16 KB per-warp O + 4 KB staging | 32 KB staging |

## 2. Numerics: the precision boundaries, against `02-model-layer.md` §3 and the reference

| quantity | baseline (SGLang) | reference | K1 |
|---|---|---|---|
| QK-norm math | FP32, BF16 out | FP32 | FP32 |
| RoPE | FP32 on BF16-rounded normed values | FP32 | FP32 (accurate `sincosf`, fp32 angle like torch) |
| new K/V into the cache | bf16 → e4m3 (double rounding) | fp32 → e4m3 | fp32 → e4m3, RNE, satfinite (single rounding, as the reference) |
| Q into the MMA | E4M3 | BF16 (bound ref.: E4M3) | **FP16**: finer than E4M3, which the rules allow ("no coarser") |
| K/V into the MMA | E4M3 (FP8 wgmma) | FP32 of e4m3 | FP16, exact: e4m3 ⊂ fp16 via `cvt.rn.f16x2.e4m3x2` |
| P | E4M3 (Max_offset 8) | FP32 | FP16 with 2^8 offset |
| accumulate | FP32 | FP32 | FP32 (mma.sync) |
| attention out → norm | BF16 round, then rms_norm_gate | FP32 | FP32 straight into the RMSNorm (no BF16 round) |
| output | BF16 | BF16 cast of FP32 | BF16 |

- Emulated error against the BF16-query reference is 0.3–0.6 % relative L2, with the worst row ≤ 0.85 %. The
  case bounds are 10.6–15 %. The reference's own E4M3-query error on the same cases is 3.6–9.3 %.
- New cache entries match the reference's quantization, because the rounding chain is identical.
- No activation the baseline keeps in BF16 is made coarser. The KV cache stays FP8.

## 3. Why these instruction choices (bytes/issue arithmetic, H200 numbers from 03 §A)

- **Budget.** At B=32 on a full layer the memory budget is 256 B per key. That is 14.8 cycles per key per SM at
  4 TB/s effective, i.e. 132 SMs at 1.75 GHz.
- **CUDA-core FMA is ruled out**, even at L=1. One key costs 2 heads × 8 q heads × 64 × 2 (QK and PV) = 2048 FMA.
  The SM does 128 FMA/clk, so that is 16 cycles per key, which is above the budget. Tensor cores are mandatory.
- **mma.sync m16n8k16 FP16** is the simplest correct tensor path without a GPU to debug on. It needs no wgmma
  descriptors, no V transpose in smem, and no warp specialization.
  - **L=1:** 2 mma per key (half the rows are padding), about 3.5 cycles of issue per key. This is
    **memory-bound with a large margin**.
  - **L=8:** 8 mma per key plus 128 exp2 per key. About 12.5 cycles of issue per key if
    `cvt.rn.f16x2.e4m3x2` runs at full ALU rate, and about 32 if it runs at quarter rate. Each of the 4 warps of
    a head converts the same K/V bytes, so the cost is 512 cvt per key. **This is the main performance risk of
    v0** (see §6).
- **FP16, not BF16.** E4M3 converts to FP16 exactly in one instruction. FP16 has 3 more mantissa bits than BF16
  for Q and P. And the 2^8 P offset keeps tail probabilities normal.
- **Rows packed as (token, q head) within one KV head.** The B operand (K or V) differs per KV head, so two KV
  heads cannot share an m16 tile.
  - At L=1 that leaves 8 of 16 rows as padding. A swap-AB form, with keys as M, would fill all rows, but it needs
    P transposed between the two MMAs. It is not worth it, because L=1 is memory-bound anyway.
- **Separate combine instead of a last-CTA-done counter.** A single "last CTA" merging all splits is serial:
  - at full B=1, L=8, that is 250 splits × 32 KB = 8 MB through one SM, about 80 µs;
  - a graph-safe counter would also need a self-resetting persistent buffer.
  - The separate combine is about 1.5–2.5 µs, parallel over B·L·Hq warps, and deterministic.
- **No PDL in v0.** Whether `torch.cuda.graph` capture keeps the programmatic edge for extension launches is
  unverified (03 §A.5). It is listed as lever 2.

## 4. Bytes, launches, predicted µs (model) against the TASK.md floors

Model:
- `t = 3.0 µs (launch ramp + seq_lens → page table → first KV round trips + epilogue)`
  `+ max(bytes / (η·4.8 TB/s), issue) × split imbalance + 2.2 µs (combine incl. dependent-launch gap)`.
- η is the efficiency of random 128 B rows. The middle value is 0.8, with a range of 0.7–0.9.
- Bytes = keys × (256 B KV + 4 B page table). The new keys are not read from the cache.
- The L=8 rows assume 1 CTA/SM. The range covers η 0.9 / fast cvt through η 0.7 / quarter-rate cvt.

| load / shape | keys | bytes | CTAs × max tiles | partials (L2) | predicted µs (range) | floor µs |
|---|---|---|---|---|---|---|
| decode 1 sliding | 2.0 K | 0.5 MB | 16 × 2 | 0.1 MB | **5.3** (5.3–5.4) | 0.1 |
| decode 1 full | 80 K | 20.8 MB | 250 × 5 | 2.1 MB | **10.6** (10.0–11.4) | 4.3 |
| decode 16 sliding | 33 K | 8.5 MB | 256 × 2 | 2.1 MB | **7.4** (7.2–7.7) | 1.6 |
| decode 16 full | 1.30 M | 337 MB | 263 × 79 | 2.2 MB | **95** (85–108) | 63.2 |
| decode 32 sliding | 66 K | 17.0 MB | 256 × 4 | 2.1 MB | **9.6** (9.1–10.3) | 3.3 |
| decode 32 full | 2.55 M | 663 MB | 264 × 161 | 2.2 MB | **~170–190** (169–216) | 126.4 |
| verify 1 sliding | 2.0 K | 0.5 MB | 16 × 2 | 1.1 MB | **5.3** (5.3–5.5) | 0.1 |
| verify 1 full | 80 K | 20.8 MB | 125 × 10 | 8.3 MB | **10.6** (10.0–16.3) | 4.3 |
| verify 16 sliding | 33 K | 8.5 MB | 128 × 4 | 8.5 MB | **7.4** (7.2–9.8) | 1.9 |
| verify 16 full | 1.29 M | 337 MB | 131 × 166 | 8.7 MB | **~100** (89–198) | 63.4 |
| verify 32 sliding | 66 K | 17.1 MB | 128 × 8 | 8.5 MB | **9.7** (9.2–14.3) | 3.7 |
| verify 32 full | 2.57 M | 669 MB | 132 × 330 | 8.8 MB | **~175–195** (173–391) | 126.8 |

How to read these against the floors:
- **The full-layer floors are unreachable by construction.** They assume 0.925 × ctx keys
  (`speed_of_light_ns` uses `_shortest_share("agentic") = 0.925`), but the mean length is about 80 K. At B=32 the
  true KV volume is 655 MB, which is **136.5 µs at 4.8 TB/s** and about 156 µs at 4.2 TB/s. So "~170 µs" means
  ~80 % of HBM peak. Only random 128 B row efficiency, measured as M4, can move this.
- **Sliding shapes are latency-bound by design.** The floors are 0.1–3.7 µs. The model's floor for K1 is two
  dependent kernels, about 5 µs. The baseline issues 7–9 dependent launches for the same work, so even at
  ~2–2.5 µs per tiny graph node it costs well above 15 µs (unmeasured; the r1-1 profile will tell). The expected
  speedup on the weight-46 shapes is therefore **~2–3×**, more than on any full-layer shape.
- **Full layers, B=16/32.** The gain over FA3 has to come from full 128 B rows, versus FA3's two separate 64 B
  half rows per key from different CTAs, and from no prepare/combine/glue launches. A modest 1.1–1.3× is
  plausible at L=1. At L=8 v0 may tie or lose to FA3's FP8 wgmma (see §6).

## 5. Graph capture and rules

- **Graph capture.**
  - Both grids depend only on (B, L, Hq, Hkv, SM count, occupancy).
  - Partials and output use `torch.empty` per call, which is safe in the capture's private pool.
  - `inv_freq` (per theta), SM count and occupancy are cached eagerly on the first call.
  - The first eager call runs the self-test, which does use `.item()` and `synchronize`, only outside capture. If
    the first call ever comes while capturing, that call goes to the fallback path.
- **Every key is attended.** Full softmax and window `[p−2047, p]`, exactly. The emulator checks per (request,
  head pair, KV head, row) that the attended key multiset equals `range(max(0, p−W+1), p+1)`, each key exactly
  once.
- **Precision.** Queries are FP16, finer than the allowed E4M3. The cache stays E4M3. The output is BF16.
- **Static-check surface.** No os/sys/environ/subprocess; `load_inline` is the house pattern. Nothing derived from
  `data` is cached.

## 6. Risks

### Performance
1. **L=8 conversion and smem redundancy.** Each of the 4 warps of a KV head loads and converts the whole K/V tile
   of its head: 512 `cvt.f16x2.e4m3x2` and 1 KB of LDS per key.
   - If that cvt runs at quarter rate (16/clk/SM), verify full B=32 goes compute-bound at about 2× memory time;
     that is the 391 µs upper end of the range.
   - Levers, in order:
     - (a) A warp layout of (head, row half, key half): 2× less redundancy and registers ~170, which fits under
       (256, 1). The emulator needs a third role table.
     - (b) QK on `mma.sync.m16n8k32.e4m3`: Q in E4M3 (allowed), K bytes used as the B operand with no conversion,
       and half the QK mma count. PV stays FP16.
     - (c) wgmma (FA3-style: QK FP8 SS, V transposed in smem, P in registers). This is the real fix for 64-row
       tiles, since one smem B read serves a whole warpgroup.
2. **Random 128 B row efficiency (η)**, unmeasured (M4 in 03 §A.8). It sets every full-layer number, ±15 %.
3. **Latency floor on the sliding shapes.**
   - The current costs are: seq_lens, then page table, then KV, three dependent DRAM trips per CTA under a cold
     L2; plus the combine's launch gap.
   - Levers:
     - PDL between main and combine;
     - start the page-table prefetch before the plan (the plan needs only seq_lens);
     - fold the combine into the main kernel for B ≥ 16, where S_b ≤ 16 makes a last-CTA merge cheap.
4. **Occupancy at L=1.** The kernel needs ≤ 128 registers for 2 CTAs/SM, which is enforced by launch bounds. If
   ptxas spills (it prints `-Xptxas=-v`), drop to (256, 1); Gx adapts automatically through `k1_occupancy`.

### Correctness: covered by emulation, GPU-only items listed
- The fp8 helpers' byte order (`__nv_cvt_fp8x2_to_halfraw2` low byte → low half;
  `__nv_cvt_float2_to_fp8x2` x.x → low byte) is taken from `cuda_fp8.hpp` as I recall it. The self-test would
  catch a swap; the new-entry check is at 1 %.
- `ex2.approx.ftz(−inf) = +0` is assumed (PTX spec). Masked keys depend on it.
- Partials are never read uninitialized: the emulator fills them with NaN and asserts. Zero-fill guarantees
  finite smem.

### Compile-risk spots (for the first `okc check`)
- The build pins `-gencode=arch=compute_90,code=sm_90`. Because a user flag contains `arch`, `cpp_extension` then
  ignores the image's `TORCH_CUDA_ARCH_LIST`, so there is exactly one compile with one set of ptxas lines. The
  `#if __CUDA_ARCH__ >= 800` body guards remain as a second line of defence: empty stubs would fail the self-test
  and route to FA3.
- `cudaFuncSetAttribute(k1::k1_main<L>, …)` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` take a template
  kernel pointer. This is the standard templated runtime API.
- Lambdas inside the kernel (`pe_load`, `issue`) capture by reference. This is plain C++ device code, and no
  `--extended-lambda` is needed because they are not `__device__` lambdas passed to the host.
- `sincosf` with large arguments (pos·1 up to 1.05 M rad) takes the slow-path range reduction. It may show local
  memory ("stack frame") in the ptxas output. It runs once per row, so it does not matter.
- `asm` blocks: `mma.sync…f32.f16.f16.f32`, `cp.async.cg…16, src_size`, `ex2.approx.ftz.f32`. All are valid on
  sm_80 and later.
- Clang syntax-checks clean on the device pass (sm_90) and the host pass against stub headers. That checks
  syntax, name lookup and template instantiation, but not the asm text or real header signatures.

## 7. Emulation results (`k1_emulate.py`, CPU, ~11 s)
- **Fragment test.** The QK and PV operands, rebuilt from per-lane gathers through the swizzled smem images,
  match direct math to 1.5e-7 relative. This covers NT = 2 and 8, both heads, all warp key offsets and all row
  offsets.
- **Banks.** 224 LDS phases (K .128 in 8-lane phases, V .64 in 16-lane phases) are conflict-free. The Q-fragment
  LDS has an 8-way conflict, but it runs once per CTA.
- **New-tile writer.** The per-lane 2-byte placement equals the swizzled image.
- **Plan.** 300 random plans (B 1–69, both layers, 1–139 SMs, HP 1 or 8). Every tile is covered exactly once,
  each request has exactly one last split, there are no empty splits, `Σ S_b ≤ Gx`, and every split has ≤ C
  tiles.
- **End-to-end.** All 21 cases pass the judge rule (own new entries fed back, E4M3-query bound, row rule) and a
  tight 1 % self-consistency check. Exact key coverage holds for every row.
  - Cases: ctx 10 (page 64); ctx 2047 at the window edge (L=1 and L=8); log-normal "mixed" lengths; full B4;
    `edges` on sliding and full, at L=1 and L=8; kv_scale 1/64 on sliding L=8 and full; soft; coherent at page 1,
    both agentic and B=1 L=8 at 20 K; `dp` heads (128/16) at L=1 and L=8; tiny contexts at L=8 (causal
    dominates); ctx 2100 edges at L=8; and 1- to 3-SM grids that force long splits.
- **Mutation tests.** These were run to show the checks can fail:
  - Caught: RoPE position +1, V-swizzle read slip, causal-mask removal (short contexts), window `>` → `>=` (L=8,
    via coverage), and `need_mask` without the window term (via the assertion).
  - Benign, correctly: kstart off by one (the per-element mask still holds), and the constant lse offset (it
    cancels).

## 8. First GPU run: what to read in the log
1. `[k1] compiled in …` and the ptxas lines for `k1_main<1>`, `k1_main<8>` and `k1_combine`. Look at registers
   and spills.
2. `[k1] k1_main<L> occupancy …`. If L=1 shows < 2, check registers.
3. Eight `[k1] self-test …` lines. Output relative error should be ≲ 5e-3, and new K/V ≲ 1e-3.
4. If anything fails, the run continues on FA3 (r1 path). Set `FALLBACK = True` to bisect.
5. Microbenchmarks worth one check run each: M4 (random 128 B rows), cvt.f16x2.e4m3x2 throughput, and PDL in
   graphs.

---

# v1 (`k1_v1.py`, `k1_lab_v1.py`): response to the judge run of v0 (job ded828a1027f427a, 1.081×)

## What v0 measured
- **L=1 wins everywhere except B=1 full.** Speedups are 1.42–1.73×. At B=1 full the 16-warp combine looped over
  about 250 splits and took 19.5 µs.
- **Full layers run at about 2.9 TB/s.** For example k1_main at B=32 L=1 took 229 µs.
- **L=8 is issue-bound.**
  - On full layers L=8 takes 2.1× as long as L=1, while FA3's L=8 and L=1 times are equal.
  - It runs about 43–55 cycles per key per SM, against a memory budget of about 15.
  - The causes are as predicted: each K/V byte is converted by 4 warps, and only 8 warps per SM hide latency.
- **The combine kernel costs about 4 µs** on every shape, i.e. 20–30 % of the sliding (weight-46) shapes.

## v1 changes (all emulated: `k1_emulate.py` passes 21/21; coverage, fp8 fragments, both merge paths)
1. **L=8 warp layout:** (KV head, row half = 2 m-tiles, key half). Each K/V byte is loaded and converted by 2 warps
   instead of 4, and each V fragment feeds 2 m-tiles. The key halves are merged in smem at the end.
2. **L=8 QK on `mma.sync.m16n8k32.e4m3`.**
   - Q is rounded to E4M3 at the judge's FP8-query point: normalized, roped, before ×5.473.
   - The K cache bytes are used directly as the B operand.
   - Dim map: logical k `4t+i` ↔ dim `16t+8s+i`, and `16+4t+i` ↔ `16t+8s+4+i`. Each lane does one LDS.128 per K
     row, and k-step s uses words 2s and 2s+1.
   - PV stays FP16.
   - Emulated error equals the E4M3-query reference error (ratio about 1.0, against the judge's 2.5× + 1.5 %).
     Against the E4M3-query reference itself the error is ≤ 4e-4.
3. **O rescale skipped** unless some row max moved (warp vote). **RoPE (cos, sin) table** per (token, pair) in smem,
   instead of a `sincosf` per row element.
4. **Fold.** The last-arriving CTA of each (b, head pair) merges the splits and writes the final BF16, so there is
   no combine launch.
   - Protocol: every thread does `__threadfence` after its partial stores, then `__syncthreads`, then one
     `atomicAdd` on `counters[b·HP+hp]`. The last CTA fences again, reads the partials with `__ldcg`, and resets
     the counter to 0 at the very end.
   - The merge order is fixed (s = 0..S−1), so the result is deterministic.
   - When `S_b == 1` the CTA finalizes straight from smem and never touches the counter.
   - The counters are a persistent 64K-int buffer allocated eagerly. If it is first needed under capture, the call
     uses the separate path and logs it.
5. **k1_combine (separate path)** is one CTA per (b, j, q head): nsg split groups × 16 float4 lanes, with an
   in-order smem reduce. nsg is 32 at B=1, 8 at B≤4, otherwise 4. The fold and separate outputs agree to fp32
   rounding (emulator asserts; the lab measures it on GPU as `maxdiff`).
6. **Host tables** keyed by (layer kind, q_len): `K1_ROUTE_MIN_BATCH` (all 1 → K1 everywhere),
   `K1_FOLD_MIN_BATCH` (sliding 1, full 8), `K1_MIN_CHUNK_T` (2, but 4 for sliding L=8), `K1_CPS_T` (2, capped by
   occupancy).

## Predicted judge µs for v1 (rough, from v0's measured kernels; the lab replaces these)

| shape | SGLang | v0 | v1 predicted | basis |
|---|---|---|---|---|
| dec1 sliding | 21.3 | 15.0 | ~11.5 | −4.0 combine, +0.7 last-CTA merge |
| dec1 full | 30.8 | 36.6 | ~20 | combine 19.5 → ~3.5 (32 groups × ~8 splits each) |
| dec16 / dec32 sliding | 26.6 / 31.6 | 18.3 / 21.7 | ~14.5 / ~18 | −4 combine |
| dec16 / dec32 full | 199.9 / 382.8 | 115.4 / 238.1 | ~111 / ~233 | −5 combine |
| ver1 sliding | 23.2 | 27.8 | ~16–19 | compute ~½, no combine |
| ver1 full | 35.7 | 45.6 | ~23–28 | main 28 → ~15, combine 11.9 → ~3.5 |
| ver16 / ver32 sliding | 29.3 / 34.6 | 33.5 / 40.8 | ~21 / ~26 | compute ~½, no combine |
| ver16 / ver32 full | 204.6 / 388.1 | 256.3 / 491.0 | ~120–135 / ~235–265 | ~36 vs ~81 warp-instr per key; floor is L=1's memory-bound 229 µs |

The geometric mean over loads comes to about 1.4–1.6× if the L=8 estimate holds. If L=8 falls short at some
shape, set that `K1_ROUTE_MIN_BATCH` entry to `1 << 30` (FA3).

## Reading `k1_lab_v1.py` (one check slot)
- The self-test lines come first: 11 cases × (fold, sep, fold), each with `counters 0`.
- Then one table row per scored shape. Columns: FA3, the default tables (`def`), then F1/F2/F4 (fold,
  min_chunk 1/2/4), F2c1 (1 CTA/SM), S1/S2/S4/S2c1 (separate combine), and S2n8/S2n32 (nsg).
- `best` gives the speedup over FA3, raw and judge-adjusted (replay floor subtracted).
- `maxdiff` is the largest rel-L2 difference of any variant from `def`. It should be ≲ 1e-3; anything larger is a
  bug in that path.
- Fill `K1_FOLD_MIN_BATCH`, `K1_MIN_CHUNK_T` and `K1_CPS_T` from the best columns, and the route table from
  `best` vs FA3.

## Next levers (not done)
- **FP8 PV:** P as e4m3 with the same 2^8 offset, and a V byte transpose (4 keys × 4 dims per 8 prmt). This saves
  32 cvt + 16 mma per warp-tile. It is a new precision boundary and needs its own emulator path.
- **PDL** between k1_main and k1_combine on the separate path.
- **Lower B=1 full latency** with a two-level fold (group counters), which would remove the combine there too.

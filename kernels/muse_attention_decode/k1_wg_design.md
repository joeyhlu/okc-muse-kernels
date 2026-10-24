# K1-WG: wgmma L=8 (verify) attention kernel — design (stage A, 2026-10-24)


**Stage gates:**
- **A:** ≥ 8 warps/SM, 2 CTAs/SM, registers ≤ 128.
- **B:** QK-only prototype: ver32 full ≤ 300 µs, ver16 full ≤ 160 µs.
- **C:** the full kernel at the same targets.

**What stays from K1 v4:**
- the split plan, fold/separate merge (v4 batched merge), and route tables;
- the new-token handling and the 128 B KV-row gather from the FP8 page-1 cache.

The only change is the L=8 tile loop: it moves from mma.sync to wgmma.

## 1. Precision boundary: E4M3 P matches stock

**Stock converts P to E4M3 for an FP8 PV wgmma.**

- **Source.** In `flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` (Dao-AILab clone at 10b74d98, in
  `repos/`):
  - `:44` defines `Is_FP8` when `Element` is `float_e4m3_t`.
  - `:83` reads `static_assert(!(!MmaPV_is_RS && Is_FP8), "MmaPV must be RS if FP8")`.
  - `:1156–1159`: `permute_Cregs_fp8(tSrS); tOrP = make_tensor_like<Element>(tOrP_acc); convert_type_out(tOrP_acc,
    tOrP)` converts the fp32 P to `Element` = e4m3.
  - `:1182` passes that P as the register (RS) A operand of the PV `flash::gemm`.
- **The binary actually used.** The a55ab7d5 tree has no FA3 source. Stock imports `sgl_kernel.flash_ops` from the
  wheel `sglang_kernel 0.4.9`, which is a sgl fork with an extra template parameter compared with upstream. I
  dumped the SASS of its hdim-64 FP8 kernel, mangled name
  `...CollectiveMainloopFwdSm90ILi2E...tupleI64,128,64...Li64ENS_12float_e4m3_tEf...`
  (box_out/k1wg_stock_fa3_fp8_evidence.txt). It contains:

  | instruction | count | role |
  |---|---|---|
  | `QGMMA.64x128x32.F32.E4M3.E4M3` | 4 | QK |
  | `QGMMA.64x64x32.F32.E4M3.E4M3` | 8 | PV, with E4M3 A = P |
  | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | 64 | fp32 P → e4m3 |
  | HGMMA (FP16 mma) | none | |

  It uses kBlockN = 128 keys.

So the precision chain is: Q E4M3 (the allowed floor), K/V E4M3 (the cache), P E4M3 with a 2^8 offset (stock FA3
uses `Max_offset` 8 for FP8), fp32 accumulation, fp32 row sums. This matches stock at every point. The emulator (§6)
checks the error against the judge bound.

## 2. Work decomposition

The CTA is unchanged: one split of one request × one KV-head pair, 256 threads, 2 CTAs/SM.
- **Warpgroup h** (warps 4h..4h+3) owns KV head h: 64 rows = 8 tokens × 8 q heads, i.e. M = 64, exactly one wgmma M
  tile.
- **Warp w of the warpgroup** holds rows 16w..16w+15 = tokens 2w, 2w+1. This is the v3/v4 m-tile mapping, so the
  masking and the epilogue code are reused.

Per 64-key tile, each warpgroup issues:

| GEMM | instruction | A | B (smem descriptor) | D |
|---|---|---|---|---|
| S = Q·Kᵀ | 2 × `wgmma.m64n64k32.f32.e4m3.e4m3` (head dims 0–31, 32–63) | Q, smem desc (staged once per CTA) | K tile from the ring (K-major = the cache row layout) | 32 f32 regs/thread |
| O += P·V | 2 × `wgmma.m64n64k32.f32.e4m3.e4m3` (tile keys in 2 k32 steps) | P, registers (e4m3, packed from S) | Vᵀ tile (K-major over keys, keys permuted, §3.3) | 32 f32 regs/thread (O) |

**Accumulator layout** (PTX m64nN f32 D): thread (warp w, g = lane>>2, t = lane&3) holds rows 16w+g and 16w+g+8, at
columns 8j+2t+{0,1} for j = 0..7. This is identical to the mma.sync C layout used since v1, so the masking, row
max/sum (quad shuffles), exp2 and O epilogue code carry over unchanged.

**Per tile and warpgroup:**
1. `wgmma.fence`; QK(2); `commit`; `wait 0`.
2. Mask, row max, exp2 → fp32 p.
3. Rescale O if the max moved (warp vote); l += Σp.
4. Pack P to e4m3 registers.
5. `wgmma.fence`; PV(2); `commit`; `wait 0`.

There is **one `__syncthreads` per tile** (§4).

## 3. Shared-memory layouts (wgmma K-major INTERLEAVE = no swizzle)

The PTX/CUTLASS canonical layouts are in `cute/arch/mma_sm90_desc.hpp`:
- `:110–125` give the bitfields: start>>4 at [0,14), LBO>>4 at [16,30), SBO>>4 at [32,46), layout type at
  [62,64), with 0 = INTERLEAVE.
- For K-major ("N"), LBO is the stride between the two 16-B columns of the 8×2 core brick, and SBO is the stride
  between 8-row groups.
- The byte address of element (row r, k) of an 8-bit K-major operand is therefore
  `start + (r%8)*16 + (r/8)*SBO + (k%16) + (k/16)*LBO`.

### 3.1 K tile (B of QK), written directly by cp.async

Per head, N = 64 keys × K = 64 dims:
- LBO = 128 B, SBO = 512 B, giving 4 KB per head and 8 KB per tile (both heads).
- The k32 step s descriptor starts at `+ 256·s`.
- A 16-B cache chunk (key x, chunk c = head c>>2, dims 16(c&3)..+15) lands at
  `h·4096 + (x>>3)·512 + (c&3)·128 + (x&7)·16`.

### 3.2 Q (A of QK), staged once per CTA

The layout is the same as K, with rows m = token·8 + q-head-in-group: 4 KB per head, 8 KB total. Q is RMS-normed,
roped, and rounded to e4m3 before the ×5.473, exactly as v1–v3 L=8. The 5.473 · sm_scale · k_scale factor stays in
`cl2`.

### 3.3 Vᵀ tile (B of PV), keys permuted so P never leaves its registers

**The constraint.** The PV A operand (8-bit) in registers follows the m16n8k32 A layout per warp:
- reg a0 = row g, logical k 4t..4t+3;
- reg a1 = row g+8, same k;
- reg a2/a3 = the same rows, k 16+4t..16+4t+3.

The thread's p values sit at keys 8j+2t+{0,1}.

**The fix: choose the key order of the PV reduction so that a thread's own p's form its A registers.**
- For k-step s and logical k, write k = 16·(q&1) + 4t + i, with q = 2s + (k ≥ 16) and i ∈ 0..3. The mapping is
  **key(s, k) = 16q + 8·(i>>1) + 2t + (i&1)**.
- Byte i of the thread's A register for (row, set q) is p at key 16q + 8(i>>1) + 2t + (i&1).
- These are exactly the keys this thread holds for j = 2q, 2q+1: two `cvt.rn.satfinite.e4m3x2.f32` per register,
  and no shuffles.
- (FA3 instead permutes C registers with byte shuffles; see `permute_Cregs_fp8` above.)

**Layout.**
- Vᵀ for head h, step s: N = 64 dims × K = 32 logical keys, with LBO = 128 and SBO = 256.
- That is 2 KB per (head, step), 8 KB per tile, double-buffered (16 KB).

**Writers (no cross-thread data).**
1. Thread tid loads K and V of the key pair k0 = 16q + 8b + 2t′ and k0+1, plus chunk c. Here q = tid>>6,
   b = (tid>>5)&1, t′ = (tid>>3)&3, c = tid&7. That is 4 cp.async of 16 B: K → the §3.1 layout; V → its own 32-B
   slot in a raw V ring.
2. After its own `cp.async.wait_group`, it builds 16 halfwords `V[k0][d] | V[k0+1][d]<<8` (8 PRMT).
3. It stores each halfword (16 × STS.U16) at Vᵀ(n = 16(c&3)+d, logical k = 16(q&1) + 4t′ + 2b).

Each thread uses its own pair of page-table entries for both K and V, so there are 2 page-table loads per thread
per tile.

**Proxy fence.** cp.async and STS are generic-proxy writes and wgmma reads through the async proxy. Every thread
executes `fence.proxy.async.shared::cta` after its writes and before the barrier.

### 3.4 New tokens (last split)

The prologue writes the L new keys into a zeroed 64-key K tile and the new values into a zeroed Vᵀ tile, using the
same layouts and permutation. These are dedicated 16 KB, ready before the loop. They are attended after the loop
with mask `key < seq + L`, causal.

## 4. Pipeline and barrier (one `__syncthreads` per tile)

```
prologue: issue tiles 0..ST-2 (cp.async), page entries prefetched 2 tiles ahead
for i in tiles:
  cp.async.wait_group<ST-2>                 // this thread's chunks of tile i landed
  own V pairs of tile i -> Vt[i&1]          // Vt[i&1] last read by PV(i-2), which completed before barrier(i-1)
  fence.proxy.async.shared::cta
  __syncthreads                             // K(i), Vt(i) visible to every warpgroup; PV(i-1) of all WGs done
  issue cp.async tile i+ST-1 into slot (i-1)%ST   // K slot: QK(i-1) waited; raw V slot: read only by its own thread
  QK(i) -> S; wait;  softmax;  pack P;  rescale O;  PV(i); wait
after: new-token tiles (QK/PV with mask), quad-reduce l, O/l + lse -> smem -> fold merge (v4) or partials
```

## 5. Budget (the stage-A gate)

| resource | per CTA | per SM at 2 CTAs | limit | ok |
|---|---|---|---|---|
| K ring, ST=4 × 8 KB (wgmma layout) | 32 KB | | | |
| raw V ring, ST=4 × 8 KB | 32 KB | | | |
| Vᵀ, 2 × 8 KB | 16 KB | | | |
| Q (e4m3, wgmma layout) | 8 KB | | | |
| new-token K + Vᵀ tiles | 16 KB | | | |
| cos/sin 2 KB + plan/aux 1 KB | 3 KB | | | |
| **dynamic smem** | **107 KB** (+1 KB reserved) | **216 KB** | 228 KB | ✓ |
| epilogue O staging 32 KB + lse | aliases the drained rings (64 KB) | | | ✓ |
| bytes in flight (ST−1 = 3 tiles × 16 KB) | 48 KB | 96 KB | ≥ 64 KB/SM for 128 B gathers (lessons §6) | ✓ |
| threads / warps | 256 / 8 | 512 / **16 warps** | ≥ 8 warps | ✓ |
| registers/thread (estimate) | S 32 + O 32 + P 8 + m,l 4 + page entries 4 + Q/K/Vᵀ descriptors 6 + addressing ~25 ≈ **111** | 512 × 128 = 64 K | ≤ 128 at `__launch_bounds__(256, 2)`; wgmma needs room (lessons §7: ≤ 80 → serialized, C7512) | ✓ (check ptxas; fallback ST=3 and/or (256,1)) |
| wgmma per tile / WG | 4 instructions (2 QK + 2 PV) instead of 64 HMMA per warp | | | |

**Gate A: passes on paper.** 2 CTAs/SM, 16 warps/SM, about 111 registers. The risk is ptxas register allocation
around the wgmma accumulators. If ptxas reports > 128 or C7512, the fallbacks are:
- QK and PV share one accumulator set via split tiles (N = 32);
- 1 CTA/SM with 3 warpgroups.

**Throughput estimate.** These are rates, not a measurement.

| item | clk/key |
|---|---|
| per-key MACs: QK 128 rows × 64 = 8 K, PV 8 K | |
| FP8 wgmma peak ≈ 4 K MAC/clk/SM | 4 at peak (16 at 25 % efficiency for N = 64 tiles) |
| exp2 (128/key at 16/clk) | 8 |
| P pack (64 cvt/key at ~26/clk) | ~2.5 |
| FMNMX, FFMA | ~2 |
| Vᵀ transpose (16 STS.U16 + 8 PRMT per thread-tile) | ~1 |
| memory floor (3.4 TB/s bare pipeline) | ~20 |

The tensor work leaves the per-warp issue stream (4 async instructions), which is what killed mma.sync v3/v4
(~45 clk/key, latency-bound). The softmax chain remains.

**Stage-B expectation:** ver32 full 230–300 µs if softmax + wgmma waits stay under the memory time. Stage B
measures it.

## 6. Emulator (`k1_wg_emulate.py`)

**Layout tests:**
1. **K and Q tiles:** cp.async destinations → operands rebuilt with the descriptor formula → equal to Kᵀ / Q for
   both heads and both k32 steps.
2. **Vᵀ:** halfword stores → rebuilt B → equal to V[key(s, k)][n].
3. **P packing:** S-accumulator positions → A registers → rebuilt logical P, compared to P[m][key(s, k)]; and
   P·Vᵀ_perm equals P·V.
4. **Bank conflicts:** Vᵀ STS.U16 phases and own-V LDS phases.

**Numerics:** Q e4m3, P e4m3 (2^8 offset, RN satfinite), fp32 row sums. All 21 k1_emulate cases are checked with
the judge rule. Two tight checks are reported:
- against the reference with E4M3 queries;
- against an FP32 emulation of E4M3 Q + E4M3 P, which is the self-consistency.

The A-register layout (§3.3) and the descriptor bit positions are PTX facts the CPU cannot confirm. The first box
step is a 1-CTA wgmma unit test, `k1_wg_lab_unit.py`, compared against torch:
- SS FP8 m64n64k32 with these LBO/SBO values;
- RS with the packed P registers;
- the accumulator layout.

It runs before the kernel.

## 7. Results so far (2026-10-24)

### Stage A (laptop, then a unit test on the box)

- **The budget table above.** 2 CTAs/SM, 16 warps/SM, 108 KB dynamic smem at ST=4.
- **Emulator `k1_wg_emulate.py --quick`: 12/12 L=8 cases pass the judge rule.**
  - **Layout tests:** K and Q INTERLEAVE tiles; Vᵀ with permuted keys; P register packing; P_perm·Vᵀ_perm = P·V;
    bank conflicts. The first Vᵀ strides (SBO 256) gave an 8-way STS conflict. They became SBO 272 with a
    2192-byte region stride, which is conflict-free.
  - **Numerics:** error relative to the reference is ≈ the E4M3-query reference's own error (0.036–0.085), against
    bounds of 0.106–0.150, so at most 0.55 of the bound. Against the E4M3-query reference itself, E4M3 P adds only
    0.16–0.9 %.
- **Unit test `k1_wg_lab_unit.py` on the box: the layouts are correct.**
  - It covers SS FP8 wgmma with the §3.1/3.2 descriptors, and RS with packed P plus the §3.3 Vᵀ layout.
  - Both match torch to a max relative error of 2.3e-4 / 1.4e-4. That residual is the FP8 wgmma accumulation
    precision; a wrong layout would give errors of order 1.

### Kernel build (k1_v5.py, sm_90a)

- **Correct first time.** The self-test passes on every L=8 case, at 0.5–0.95 % against the E4M3-query reference.
- **ptxas:** 128 registers, 24–32 B of spill stores, no C7512 serialization warning.

### Stage B (box, same process as rung-1; box_out/k1v5_stageB.log), µs

| shape | QK-only, ST=4 | QK-only, ST=3 | full wgmma, ST=4 | full, ST=3 | rung-1 | gate |
|---|---|---|---|---|---|---|
| ver16 full | 186.9 | 176.8 | 202.5 | 197.8 | 196.4 | QK-only ≤ 160 ✗ |
| ver32 full | 326.2 | 317.0 | 363.5 | 364.4 | 385.6 | QK-only ≤ 300 ✗ |
| ver1 full | | | 33.9 | 34.8 | 32.7 | |
| ver1 / ver16 / ver32 sliding | | | 43.0 / 41.8 / 44.0 | 42.7 / 41.0 / 43.7 | 18.3 / 24.8 / 30.5 | |

**Stage B kill gate: missed.**
- QK-only measured 317 µs against ≤ 300, and 177 µs against ≤ 160.
- The full kernel is the first K1 L=8 to beat rung-1 on a full layer: ver32 at 363 µs is 1.06× over rung-1. Ver16
  is a tie (0.97–0.99×). The sliding shapes are still dominated by the L=8 last-CTA merge (~20 µs) and the prologue.
- Per SM: 32 clk/key QK-only and 37 clk/key full, against v4 mma.sync at ~45 and the bare stream at ~20 (193 µs).

**What a pass would need, as candidates (not measured):**
1. Overlap QK(i+1) with softmax(i) (FA3's intra-warpgroup overlap). This needs a second S accumulator (+32
   registers), so 1 CTA/SM with 3 warpgroups, or 128-key tiles.
2. Barrier-free producer/consumer staging (FA3-style mbarriers instead of the per-tile `__syncthreads`).
3. The L=8 merge fix, for the sliding shapes.

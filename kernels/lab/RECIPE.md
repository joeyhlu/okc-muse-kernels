# RECIPE: kernels/lab/k3_hh_3.py (final decode_layer candidate)

`k3_hh_3.py` (935,816 B, sha1 `88e2c90f936012b115aa6d18f5c32d57f605c7ae`, commit ed8df84f) is a generated file. This is the
single ordered recipe (review 8.1). A clean rebuild from the scored ship is **byte-identical, all lines including the nonce**
(verified 2026-10-30 on Windows, Python 3.11.9, 3 runs). Header: line 1 `# okc: problem=muse_decode_layer`, line 2 `# nonce ...`.

## Rebuild (about 11 s, no GPU, no network)

```
python tools/build_final.py                 # base = scored ship k3_gc_21cg_s1; output build_final/k3_hh_3.py
python tools/build_final.py --from-gc21     # one link further back: k3_gc_21 -> cg_patches -> the ship
python tools/build_final.py --scored-copy=kernels/muse_decode_layer/k3_gc_NN_s1.py --note="what this copy is for"
```

It prints each step's input / output sha1 and size, compares with the committed intermediate and with the sha1 below, warns if a
patcher / helper blob differs from the recorded one, asserts `rebuilt == kernels/lab/k3_hh_3.py` (`--strict`: also lines 1-2;
otherwise a lines 1-2 difference is printed and tolerated, anything from line 3 on is a failure with a line diff and the first
step whose output left the recorded sha1), then runs `tools/lead_gate.py file` on the rebuilt file (result: **PASS**; also PASS on
`kernels/lab/k3_hh_3.py`). Exit code 0 = identical and gate PASS. Work dir `build_final/` is regenerable; never commit it.
Each step runs as a fresh `python -I -X utf8 tools/<patcher> ...` from the repo root. `--start-at=N` resumes from `build_final/`.

## The 12 steps (base hash -> patcher -> output hash)

Output file STEMS matter: robust / fnc / rider3 / rob2 derive the line-2 nonce from `--out`'s stem and the base's stem.
All "committed" columns were compared byte-for-byte against the HEAD blob. Paths below are in `build_final/` unless noted.

| # | command (run from the repo root) | output | sha1 | bytes | committed twin |
|---|---|---|---|---|---|
| base | `kernels/muse_decode_layer/k3_gc_21cg_s1.py` (scored ship, 2.2091, commit 415810b2) | | `9752060f1c503aff1557ade0599b96d89f9280d5` | 983,524 | itself |
| 1 | `python -I tools/robust_patches.py --r1 --r2 --base=k3_gc_21cg_s1 --out=build_final/k3_rob_1.py` | k3_rob_1 | `4f7b213c09dc9715a16177cd219131e919b367ec` | 987,732 | kernels/lab/k3_rob_1.py |
| 2 | `python -I tools/fnc_patches.py --base=build_final/k3_rob_1.py --out=build_final/k3_b23_1.py --clean=1` | k3_b23_1 | `57a34c50e9d89e87c092005805d36ae379030803` | 989,388 | kernels/lab/k3_b23_1.py |
| 3 | `python -I tools/rider3_patches.py --base=build_final/k3_b23_1.py --out=build_final/k3_rider3_1.py --r1 --r2 --r3` | k3_rider3_1 | `c93b69240cea26609d54187fa2f4a4c290d61ac3` | 991,535 | kernels/lab/k3_rider3_1.py |
| 4 | `python -I tools/vse_patches.py build_final/k3_rider3_1.py --ps=1 --out=build_final/k3_b24_v.py` | k3_b24_v | `c0e6bb41e0407641beaa2556fc4e56bb6e17eee6` | 997,861 | kernels/lab/k3_b24_v.py |
| 5 | `python -I tools/dpin_patches.py --base=build_final/k3_b24_v.py --out=build_final/k3_dpin_2.py --pins=q128,q256,o128,u256` | k3_dpin_2 | `2aa4a8fe0c425588897c3bdb4feb207902af22f7` | 999,922 | kernels/lab/k3_dpin_2.py |
| 6 | BND attn-256 route entry: one anchored replace, **no patcher exists** (see Gaps) | k3_bnd256 | `5a10ad4bfadb8817333c02d0a1ad7fdb3a20ac88` | 999,956 | none |
| 7 | `python -I tools/ax2_patches.py --base=build_final/k3_bnd256.py --out=build_final/k3_ax2_b29.py --vx=3 --st=3` | k3_ax2_b29 | `4501ba472a275d38c6bdc3b9a821f65c6e9c9bc7` | 1,006,851 | none |
| 8 | `python -I tools/ax1_patches.py --base=build_final/k3_ax2_b29.py --out=build_final/k3_b29_raw.py --route=1 --slim=0`, then line 2 set to the hand-typed b29_raw nonce | k3_b29_raw | `6933008fa2ad9388265e82cdb7e69d448e431e59` | 1,054,699 | kernels/lab/k3_b29_raw.py (e2452ef6) |
| 9 | `python -I tools/hya_patches.py --base=build_final/k3_b29_raw.py --out=build_final/k3_hya_3r.py --mode=ship --nonce="k3-hya-3 (b29_raw + tools/hya_patches.py --mode=ship: k3_hya_1 + prepare-only HYA workspace; slim_patches last)"` | k3_hya_3r | `21e81c4fe9611c8e885f686b11638bb7b4ab73a9` | 957,234 | kernels/lab/k3_hya_3r.py (e5344201) |
| 10 | `python -I tools/hyb_patches.py --base=build_final/k3_hya_3r.py --out=build_final/k3_hh_3r.py --pdl=1` | k3_hh_3r | `f38403522e88afc7caaef7c1f7362ebd6e9d98ef` | 1,018,977 | kernels/lab/k3_hh_3r.py |
| 11 | `python -I tools/rob2_patches.py --base=build_final/k3_hh_3r.py --out=build_final/k3_hh_3rr.py --no-p2` | k3_hh_3rr | `f2167cbcb618a6ecd000940cfe114cf471288870` | 1,018,927 | kernels/lab/k3_hh_3rr.py |
| 12 | `python -I tools/slim_patches.py --base=build_final/k3_hh_3rr.py --out=build_final/k3_hh_3.py` (LAST; keeps license blocks, R-slim-1) | k3_hh_3 | `88e2c90f936012b115aa6d18f5c32d57f605c7ae` | 935,816 | kernels/lab/k3_hh_3.py (ed8df84f) |
| 0 (opt.) | `python -I tools/cg_patches.py --base=kernels/muse_decode_layer/k3_gc_21.py --out=build_final/k3_gc_21cg_s1.py --trig=1 --ph=0`, then line 2 set to the ship's nonce | = base | `9752060f1c503aff1557ade0599b96d89f9280d5` | 983,524 | the ship; input k3_gc_21.py `310ad0acffb9b92a0ea0eeef5edf784190f495fa` (979,117 B, scored 2.1887) |

Order checked against the history: the chain in the task brief was right as given. Evidence: b24_v = rider3 + VSE_PS (54bd7266; vse is
a positional-input tool, `--ps=1` sets VSE_PS on bf2 and w72); dpin_2 == `dpin_patches --base=k3_b24_v` (lab.md 2026-10-30 ~23:30 UTC);
b29 stack = dpin -> BND256 -> ax2 (vx 3 st 3) -> ax1 `--route=1 --slim=0` (lab.md same entry); bundle order b29_raw -> hya -> hyb
-> rob2 `--no-p2` -> slim LAST (lab.md 2026-10-31 ~00:10 UTC). The rebuilt files equal the committed ones, which settles it.

**Step 13 (2026-10-31, lead): kvs** `tools/kvs_patches.py --base=k3_hh_3.py --out=k3_hh_3k.py` (R-kvs-2, 34ca93c7: kv_scale != 1
at q_len 1 -> HYA, non-strict 3-case self-test, FA3 fallback). It anchors on slimmed text, so it runs AFTER slim; output ==
committed kernels/lab/k3_hh_3k.py (sha1 9d979dcd, 937,781 B), FINAL = k3_hh_3k. rob2 P2 does not apply on the hya chain (AX1 site
is `if o is None and _ax1_wanted(...)` after HYA) and stays off until ported.

## Gaps and hand edits (all automated in build_final.py; none changes the result)

1. **BND attn-256 (step 6).** `V9_BND_ROUTE` gets `('attn', 256): (1, (1024, True))` (R-bndx-2). It was a hand sed in the b25 bundle
   (00a3b624, lab.md "BND256 sed, line 3884"); no tool contains it. Automated as a single-occurrence replace of the dict tail
   `('ffn', 32): (16, None)}` in `_bnd256()`. To make it a patcher later: move `_bnd256` into `tools/` unchanged.
2. **b29_raw nonce (step 8).** ax1 leaves ax2's nonce on line 2; the committed k3_b29_raw carries a typed nonce
   (`# nonce k3-b29-raw (UNSLIMMED patch base: ...)`). Stored verbatim as `NONCE_B29_RAW`. Cosmetic: step 9's `--nonce` replaces line 2 anyway.
3. **hya nonce text (step 9).** Passed as `--nonce=` (string stored as `NONCE_HYA_3`); hyb then keeps it (k3_hh_3r still says `k3-hya-3`),
   rob2 writes `k3-hh-3rr (k3_hh_3r + P1 six call-time disables shape-scoped; tools/rob2_patches.py)`, slim keeps lines 1-2.
   So the nonce of k3_hh_3 comes from rob2 and is derived, not typed.
4. **Gates not re-run.** `--check` of hya / hyb / ax1 (WSL nvcc / ptxas / SASS identity, pyflakes) validate but do not change the
   output, so the rebuild omits them. dpin's own gate (ast, ascii, <1,000,000 B) and slim's gates (ast, tokens, license) always run.
5. **No committed twin for steps 6 and 7** (k3_bnd256, k3_ax2_b29): pinned by sha1 only. Their product (step 8) matches the committed k3_b29_raw.
6. **Reach.** Back to `k3_gc_21.py` (`--from-gc21`). Before that: gc_21 = b21_2 = gc_20 + RPO-2 g20_7 PF routes
   (`rpo_patches`, per-route flags not recorded in one place) + FAST-TAIL (`fasttail_patches`); not attempted.
7. **Platform.** Verified on Windows / Python 3.11.9 only. Every patcher writes with `newline="\n"`; the script warns if a CR byte appears.

## Scored / check copy (nonce convention)

A scored or check copy differs from `kernels/lab/k3_hh_3.py` ONLY in line 2 (`# nonce <name> (= kernels/lab/k3_hh_3.py: <note>)`);
lines 3.. stay byte-identical, name = file stem with `_` -> `-`. Existing examples: `kernels/muse_decode_layer/k3_gc_32_s1.py`
(scored 2.2068) and `k3_hh3_c1.py` / `k3_hh3_c2.py` (checks, listed in `tools/wave_hh3c.txt`).

```
python tools/build_final.py --no-gate --scored-copy=kernels/muse_decode_layer/k3_gc_33_s1.py --note="hya fixed ws + hyb pdl + rob2 P1, fixed slim; scored"
```

The flag refuses to overwrite and asserts lines 3.. identical, ASCII, LF only, < 1,000,000 B. Submit from WSL as usual
(`python3 -I tools/okc_wave.py submit <wavefile> check|submit`). A nonce-only re-run of an identical body is a policy call for
main / the user (notes/lab.md 2026-10-31 ~01:30 UTC), not something this recipe decides.

## Keeping the recipe true

- New rider: insert its step BEFORE slim (slim stays LAST), add it to `STEPS` in `tools/build_final.py` (argv, deps, recorded sha1 / size,
  committed twin), rebuild, update this table. Raw bases go over 1 MB by design; only the slim output is capped.
- A patcher or helper edit changes its blob: `build_final.py` prints `WARN <path>: blob ... != recorded ...`. If the output still
  matches, update `DEP_BLOBS`; if not, the first step whose sha1 left the recorded value is where to look.
- Recorded patcher blobs (git hash-object): see `DEP_BLOBS` in `tools/build_final.py` (23 files: 17 patcher / helper modules,
  2 vendored vjeux files, 3 alygg77 files, lead_gate).

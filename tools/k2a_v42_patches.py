"""k2a_v42_patches.py: K2A v4.2 = v4.0 (tools/k2a_v40_patches.py, already in k3_v23) + WSK, as exact text patches against
kernels/muse_decode_layer/k3_v23.py (the K2A source embedded there, K2A33_KERNEL_SRC, and the bucket-256 call site).

  1 WSK     The v3.3 gemm's Stream-K ranges (both roles, K1 and K2) are cut at equal byte-model weight instead of equal stage
            counts, on HOTCAP calls only (P.hot_t > 0, i.e. vid 2; vid 0 / 1 keep my_range2 exactly). The weight of a stage of
            a pass with n token rows is pick_role's own model: 36 + n (hot passes: n rounded up to 8). A 64-token hot stage
            costs far more than a 32-token one, so equal-stage ranges left a long per-CTA tail: K1 hot CTA gemm ends p10 118 /
            p50 145 / max 188 us at seed 853 (lab lf_f28), with WSK 125 / 142 / 165 (box_out/v40/run5.log SPREAD lines).
            Only the cut points move: same items, same math; split-item FP32 atomics land elsewhere (vs shipped ~3e-4, as
            any role-split change; vs the FP32 reference 3.75e-3, unchanged).
  2 CALL256 bucket 256 weight 78 -> 76 (26 / 32): the best weight with WSK on the shipped-module path (box_out/v40/run6.log:
            shipP_f26 -24.3 us median, f28 -21.7, f30 -12.8, f32 -7.4).

Measured, layer routing, 3 routings x 20 paired rounds, cold L2, PAIRED VS SHIPPED-WITH-v4.0 (tools/box/k2a_v40.py --ref=shipA:
shipA = k2a_v37 + k2a_v40 KERNEL_PATCHES = k3_v23's K2A at weight 78; shipP = shipA + this file at weight 76; box_out/v40/run7-9.log):
  T=256  v4.2 vs v4.0: -11.4 us (run7: -20.9 / -9.1 / -11.4, x1.038) and -15.1 us (run9: -22.0 / -9.1 / -15.1, x1.050)
         (vs the v3.7 module: -24.3 / -25.0 us, x1.085).
  T=128 (vid 5) -0.6 / +0.4 / ..., T=32 / 16 (vid 3) -0.6 / +0.2, T=8 / 1 (vid 0) +0.0 / -0.3 (x0.998 / x1.014): unchanged
         (none of them runs WSK). An earlier runtime-only gate (`if (P.hot_t > 0)` without `if constexpr (NC == 2)`) cost vid 0
         +0.6 / +1.4 us at T=8 / 1 from code generation alone (run8): keep the constexpr gate. Checks PASS everywhere.

Not included (lab only, kernels/lab/k2a_v40_lab.py V40_HANDOFF): the per-(expert, pass) K1 -> K2 hand-off. On top of WSK it is
worth ~ -1.4 us (hw_f28 -27.5 vs lw_f28 -26.1, run5) because K1 finishes every expert near its role's end (no expert is ready
early), and it needs a new run3 tensor + host plumbing. Its no-wait ceiling on top of the cold fixes was -0.7 us (hoenw_f28).

    python -I tools/k2a_v42_patches.py [k3_v23.py] [out.py]   # asserts every `old` occurs exactly once, applies all to a copy
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v23.py"

WSK_HELPERS = r"""// v4.2 WSK: Stream-K cut at equal byte-model weight (pick_role's model: a stage of a pass with n token rows weighs 36 + n,
// hot n rounded up to 8) instead of equal stage counts; HOTCAP calls only (P.hot_t > 0). Only the cut points move.
template <bool HOT, int NC>
__device__ __forceinline__ long long wsk_wpass(int c, int jj) {
  constexpr int TP = HOT ? 64 : 8 * NC;
  const int nt = min(TP, c - jj * TP);
  return 36 + (HOT ? ((nt + 7) & ~7) : nt);
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ int wsk_pos(const Route& S, long long T) {
  for (int e = 0; e < 32; ++e) {
    if (S.sb[e + 1] == S.sb[e]) continue;
    const int np = S.np[e], c = S.cnt[e];
    long long wrb = 0;
    for (int jj = 0; jj < np; ++jj) wrb += 16 * wsk_wpass<HOT, NC>(c, jj);
    if (T >= wrb * NRB) { T -= wrb * NRB; continue; }
    const int rb = (int)(T / wrb);
    T -= rb * wrb;
    for (int jj = 0; jj < np; ++jj) {
      const long long w = wsk_wpass<HOT, NC>(c, jj);
      if (T >= 16 * w) { T -= 16 * w; continue; }
      return S.sb[e] + (rb * np + jj) * 16 + (int)(T / w);
    }
    return S.sb[e] + (rb + 1) * np * 16;
  }
  return S.sb[32];
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ void wsk_range(const Route& S, int cta, int ncta, int& beg, int& end) {
  long long W = 0;
  for (int e = 0; e < 32; ++e) {
    if (S.sb[e + 1] == S.sb[e]) continue;
    for (int jj = 0; jj < S.np[e]; ++jj) W += 16LL * NRB * wsk_wpass<HOT, NC>(S.cnt[e], jj);
  }
  beg = wsk_pos<HOT, NC, NRB>(S, W * cta / ncta);
  end = (cta + 1 == ncta) ? S.sb[32] : wsk_pos<HOT, NC, NRB>(S, W * (cta + 1) / ncta);
}

"""

# (what, old, new): every `old` occurs exactly once in kernels/muse_decode_layer/k3_v23.py
PATCHES = [
    ("WSK: byte-weighted Stream-K range helpers (before hot_item)",
     "// v3.3 warm-expert item (all of its stages s0..s1)",
     WSK_HELPERS + "// v3.3 warm-expert item (all of its stages s0..s1)"),
    ("WSK: v3.3 gemm ranges by weight on HOTCAP calls (P.hot_t > 0); vid 0 / 1 keep equal-stage ranges",
     "  int beg, end;\n  my_range2(Stot, cta, ncta, beg, end);\n",
     "  int beg, end;\n"
     "  if constexpr (NC == 2) {   // v4.2 WSK: NC 2 builds only (HOTCAP roles; vid 0's NC 1 code stays byte-identical)\n"
     "    if (P.hot_t > 0) wsk_range<HOTCAP, NC, NRB>(S, cta, ncta, beg, end);\n"
     "    else my_range2(Stot, cta, ncta, beg, end);\n"
     "  } else {\n"
     "    my_range2(Stot, cta, ncta, beg, end);\n"
     "  }\n"),
    ("CALL256: bucket 256 role-split weight 78 -> 76 (26 / 32) with WSK",
     "                       hot_w4=(78 if rep == 256 else None))   # v4.0: LEAN + FLUSH2 hot role -> weight 28 / 32\n",
     "                       hot_w4=(76 if rep == 256 else None))   # v4.2: LEAN + FLUSH2 + WSK -> weight 26 / 32\n"),
]
KERNEL_PATCHES = PATCHES[:2]   # CUDA-source ones; they also apply to k2a_v37.KERNEL_SRC_V3 after k2a_v40's KERNEL_PATCHES (lab check)


def apply(text, patches=PATCHES):
    for what, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"k2a_v42 patch {what!r}: `old` occurs {n} times (want 1)")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v23_k2a42_check.py")
    text = src.read_text(encoding="utf-8")
    for what, old, _ in PATCHES:
        print(f"  {text.count(old)}x  {what}")
    patched = apply(text)
    out.write_text(patched, encoding="utf-8")
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

"""k2a_v43_patches.py: K2A v4.3 = v4.2 (in k3_v25) + two tail levers, as exact text patches against
kernels/muse_decode_layer/k3_v25.py.

  1 WSK72   v4.2's WSK stage weight for HOT passes 36 + rows -> 72 + rows (cold stays 36 + rows; hot and cold ranges are cut
            separately, so only the hot role's cut points move). Lab sweep (box_out/v40/run10.log, run11.log; lab copy lp =
            v4.0 + v4.2 with runtime weights): heavier row weights lose (36 + 2 rows +10 us, 36 + 4 rows +20 us at T=256),
            lighter wins: 72 + rows vs 36 + rows -5.2 us median of 5 routings (-0.5 / -2.3 / -7.0 / -5.2 / -9.8), vs the
            shipped v4.2 module -1.8 us (+0.8 / -0.2 / -6.4 / -1.8 / -7.3; the lab copy reads ~1 us slow). 108 + rows and
            72 + rows on the cold role too are in between; weight 76 stays best (74 / 78 lose). Only the cut points move.
  2 DYN128  bucket 128 (vid 5, HK3): K2 tail-pool fraction 0 -> 8/64 (dyn (8, 0, 3) -> (8, 8, 3)), route table only.
            Shipped-module rows (k3_v25's K2A: k2a_v37 + v4.0 + v4.2), 5 layer routings, paired: -7.1 us median
            (-8.2 / -5.8 / -4.5 / -7.1 / -7.8 vs the same module at (8, 0, 3); x1.031). K2 pools 4/64 -4.6, 12/64 -1.8,
            K1 12/64 + K2 8/64 -6.6; K1 alone 12 / 16 / 4 of 64 and chunk 4 do not help (run10).

Measured with tools/box/k2a_v40.py (layer routing, 20 rounds, cold L2). Numerics: no math changes; cut points / queue order move,
so split-item FP32 atomics differ (vs shipped <= 3.7e-4, vs the FP32 reference 3.75e-3 as shipped). Every other bucket is
untouched: patch 1 runs only in NC 2 HOTCAP calls (vid 2), patch 2 is bucket 128's plan entry. The per-CTA end spread at T=256
(K1 hot p10 ~130 -> max ~172 us) is not a stage-cost error (no weight model removes it); the next lever is a tail-only
dynamic queue in the v3.3 gemm (not built here).

    python -I tools/k2a_v43_patches.py [k3_v25.py] [out.py]   # asserts every `old` occurs exactly once, applies all to a copy
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v25.py"

# (what, old, new): every `old` occurs exactly once in kernels/muse_decode_layer/k3_v25.py
PATCHES = [
    ("WSK72: hot-pass stage weight 72 + rows (cold 36 + rows)",
     "  return 36 + (HOT ? ((nt + 7) & ~7) : nt);\n",
     "  return HOT ? 72 + ((nt + 7) & ~7) : 36 + nt;   // v4.3: hot rows weigh half as much (lab sweep)\n"),
    ("DYN128: bucket 128 K2 tail pool 8/64",
     "K2A37_PLAN = {128: (5, (8, 0, 12), (8, 0, 3))}",
     "K2A37_PLAN = {128: (5, (8, 0, 12), (8, 8, 3))}"),
]
KERNEL_PATCHES = PATCHES[:1]   # CUDA-source one (also applies to k2a_v37 after k2a_v40 + k2a_v42 KERNEL_PATCHES)


def apply(text, patches=PATCHES):
    for what, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"k2a_v43 patch {what!r}: `old` occurs {n} times (want 1)")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v25_k2a43_check.py")
    text = src.read_text(encoding="utf-8")
    for what, old, _ in PATCHES:
        print(f"  {text.count(old)}x  {what}")
    patched = apply(text)
    out.write_text(patched, encoding="utf-8")
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

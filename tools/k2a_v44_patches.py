"""k2a_v44_patches.py: K2A v4.4 = vid 3 (HK2, bs 16 / 32) work-split retune -- RESULT: NO CHANGE (PATCHES is empty).

Validated per lessons R15 on 5 router seeds (default, 12345, 777, 4242, 9001) x 2 layer inputs = 10 routings per bs, paired vs
the shipped plan (k3_v28 K2A36_PLAN 16: (8, 0, 8), 32: (8, 0, 12)); tools/box/k2a_v44.py, box_out/v44/sweepA-C.log.
Rule: keep only if median d < 0 AND no routing worse by > 1 us. Per-routing noise of a ship-identical row: +-1 us.
  role-split weight hot_w4 4..24: bs16 best = shipped 8 (others median +0.6 .. +3.2); bs32 best = shipped 12 (8: +0.05,
      worst +1.4; 16..24: +1.8 .. +4.3)
  dual range (hot_mode 1 / 3): bs16 +2.7 / +8.2, bs32 +2.9 / +9.1 median
  hot_t 4 / 6 / 12 / 16: bs16 +0.4 .. +2.4, bs32 best h6:0:12 median -0.02 but worst +6.1 -> fails
  WSK36 (REJECTED below: hot-role Stream-K cut at equal weight 16 + hot_w4 (nq - 1) per stage, kernels/lab/k2a_v44_lab.py)
      bs16 median +1.0 .. +2.2 (w4 8..16); bs32 w4 12 median -0.21 but worst +13.4 (4/10 lose > 1 us); lab control build
      (same source, WSK off) +0.26 / +0.16 -> the loss is the cut itself.
The shipped (8, 0, 8) / (8, 0, 12) stays.

    python -I tools/k2a_v44_patches.py [k3_v28.py] [out.py]   # self-check: PATCHES (none) and the REJECTED anchors once each
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v28.py"
sys.path.insert(0, str(ROOT / "kernels/lab"))
import k2a_v44_lab as LB  # noqa: E402

PATCHES = []                      # (what, old, new): nothing passed the R15 rule
REJECTED = LB.patches(wsk=1)      # WSK36, kept for reference (anchors checked below)


def apply(text, patches=PATCHES):
    for what, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"k2a_v44 patch {what!r}: `old` occurs {n} times (want 1)")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    text = src.read_text(encoding="utf-8")
    for what, old, _ in REJECTED:
        print(f"  {text.count(old)}x  (rejected) {what}")
        assert text.count(old) == 1, what
    patched = apply(text)
    assert patched == text
    if out is not None:
        out.write_text(patched, encoding="utf-8")
        py_compile.compile(str(out), doraise=True)
    print(f"{len(PATCHES)} patches (no change recommended); rejected WSK36 anchors present once each in {src.name}")

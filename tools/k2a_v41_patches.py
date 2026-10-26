"""K2A v4.1 small-T patches for the layer file (kernels/muse_decode_layer/k3_v21.py, embedded K2A33 source = k2a_v37's
KERNEL_SRC_V3). Measured in kernels/lab/k2a_v41_small_lab.py (copy `base`) with tools/box/k2a_v41.py, box_out/v41/run1-3.log.

What it does: vid 0 (v3.3 NC1, bs 1 / 8) runs each cold item through v41_cold_item = cold_item's code (v3.6's NC1 consumer
+ flush_item, the same mma order and flush formulas as v3.3's inline NC1 path, so outputs are bitwise those of the
shipped path), instead of v3.3's inline loop with its per-stage runtime probe tests. K1 / K2 regs 73 / 74 -> 71 / 71.
Measured (paired vs ship, 3 routings each, 3 runs): bs 1 -0.9 / -0.5 / -0.7 us (x1.024 / 1.019 / 1.030), bs 8 -0.7 / -1.2 /
-1.1 us (x1.009 / 1.021 / 1.018); bitwise-equal outputs (eq 1.0000; ship vs ship 0.9999 from atomics order).
vid 3 (bs 16 / 32) is NOT patched (its v41 copies were within +-0.4 us; the call-site patch is not exported).
Only these anchors are touched; none of them overlaps the v40 patches (hot_item, HOTCAP flush, NC 2 cold loop, pick_role).

  python tools/k2a_v41_patches.py            # self-check: anchors once in k3_v21.py and k2a_v37.py, patched texts parse
  python tools/k2a_v41_patches.py OUT.py     # also writes the patched k3_v21.py text to OUT.py
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab"))
import k2a_v41_small_lab as L  # noqa: E402

LAYER = ROOT / "kernels/muse_decode_layer/k3_v21.py"

# (what, old, new): the lab's export patches with the winning knobs fixed (V41_VAR 0, V41_MASK 0), vid 3 left alone
PATCHES = [p for p in L.patches(var=0, mask=0) if p[0] != "vid 3 items"]


def apply(text, patches=PATCHES):
    for what, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"k2a_v41_patches: {what!r} anchor occurs {n} times (expected 1)")
        text = text.replace(old, new)
    return text


def main():
    layer = LAYER.read_text(encoding="utf-8")
    src, _, _ = L._shipped()
    for what, old, new in PATCHES:
        assert layer.count(old) == 1 and src.count(old) == 1, what
        assert "\\" not in new and "@V41" not in new, what   # safe inside raw or plain string literals; knobs fixed
    out = apply(layer)
    ast.parse(out)
    psrc = apply(src)
    lab = L.apply(src, [p for p in L.patches(var=0, mask=0) if p[0] != "vid 3 items"])
    assert psrc == lab
    assert out.count("v41_cold_item<NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart);") == 1
    print(f"[k2a_v41] {len(PATCHES)} patches: anchors unique in {LAYER.name} and k2a_v37.py; patched layer parses "
          f"({len(layer)} -> {len(out)} chars)")
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(out, encoding="utf-8")
        print(f"[k2a_v41] wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()

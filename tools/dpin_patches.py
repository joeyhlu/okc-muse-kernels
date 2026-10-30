"""dpin_patches.py: DPIN = dense cuBLASLt pins with batched / NN layouts at T >= 128 (notes/study/t128_idle_dense.md), on top of
tools/ulp_patches.py's lt_plan_b (batch count + NN layout per pin, ULP_LAYOUT table, prepare self-test with fallback).

  python tools/dpin_patches.py --base=kernels/lab/k3_b24_v.py --out=kernels/lab/k3_dpin_1.py [--pins=q128,q256,o128] [--check]

Pins (decoded from the public alygg77 layer's algo words: w0 = tile << 32 | id, w1 = split-K << 32 | stages, w3 = cluster << 48 |
custom; tuple order = LT_CPP kAttrs: id, tile, stages, split-K, reduction, swizzle, custom, inner shape, cluster):
  q128  ('qkv', 128): (66, 92, 35, 1, 0, 0, 2, 0, 5), TN batch 16   (today: tile 80, custom 2, cluster 7, batch 1)
  q256  ('qkv', 256): (66, 124, 35, 1, 0, 0, 0, 0, 2), TN batch 1   (today: tile 134, custom 2, cluster 9)
  o128  ('o', 128): (66, 18, 35, 1, 0, 0, 3, 0, 5), NN batch 8      (today: no pin, torch default)
  o256  ('o', 256): (66, 20, 35, 1, 0, 0, 6, 0, 5), NN batch 2      (optional; today: torch default)
  u256  ('unpooler', 256): (66, 20, 35, 1, 0, 0, 7, 0, 5), NN batch 2  (main's R-ulp-0 pin, for the combined rider)
Batch slices the OUTPUT features (no split-K): every output still accumulates over the full K in the kernel's own order.
Local gates: every ulp anchor once, ast, ASCII, size < 1,000,000 B.
"""
import argparse
import ast
import pathlib
import re
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ulp_patches as U  # noqa: E402

PINS = {"q128": (("qkv", 128), (66, 92, 35, 1, 0, 0, 2, 0, 5), (0, 16)),
        "q256": (("qkv", 256), (66, 124, 35, 1, 0, 0, 0, 0, 2), (0, 1)),
        "o128": (("o", 128), (66, 18, 35, 1, 0, 0, 3, 0, 5), (1, 8)),
        "o256": (("o", 256), (66, 20, 35, 1, 0, 0, 6, 0, 5), (1, 2)),
        "u256": (("unpooler", 256), (66, 20, 35, 1, 0, 0, 7, 0, 5), (1, 2))}


def apply(base, pins):
    t = U.apply(base, types.SimpleNamespace(t256="0", t128="0", batch="2", custom256="7", custom128="3", cluster="5"))
    m = re.search(r"^LT_PIN = (\{[^\n]*\})\n", t, re.M)
    if not m or len(re.findall(r"^LT_PIN = ", t, re.M)) != 1:
        raise SystemExit("[dpin_patches] LT_PIN line: expected exactly one")
    lt_pin = ast.literal_eval(m.group(1))
    layout = {}
    for p in pins:
        key, cfg, lay = PINS[p]
        lt_pin[key] = cfg
        if lay != (0, 1):
            layout[key] = lay
    t = t[:m.start()] + f"LT_PIN = {lt_pin!r}\n" + t[m.end():]
    old = "ULP_LAYOUT = {}   # (proj, T) -> (nn, batch); tools/ulp_patches.py\n"
    if t.count(old) != 1:
        raise SystemExit("[dpin_patches] ULP_LAYOUT line: expected exactly one")
    return t.replace(old, f"ULP_LAYOUT = {layout!r}   # (proj, T) -> (nn, batch); tools/dpin_patches.py\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="kernels/lab/k3_b24_v.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pins", default="q128,q256,o128")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    pins = [p for p in a.pins.split(",") if p]
    bad = [p for p in pins if p not in PINS]
    if bad:
        raise SystemExit(f"[dpin_patches] unknown pins {bad}")
    base = (ROOT / a.base).read_text(encoding="utf-8")
    t = apply(base, pins)
    ok = U.gate(t)
    out = ROOT / a.out
    out.write_text(t, encoding="utf-8", newline="\n")
    print(f"[dpin_patches] {a.base} -> {a.out} pins {pins}: {len(t.encode())} B (+{len(t.encode()) - len(base.encode())} B)")
    if a.check:
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

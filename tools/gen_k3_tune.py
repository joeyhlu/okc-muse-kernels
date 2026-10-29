"""R-tune: per-shape autotune of existing route tables in k3_gc_16 (table changes only, no kernel-body edits).

  py tools/gen_k3_tune.py lab            -> kernels/lab/k3_tune_lab1.py (gc_16 + superset build tables) + tools/box/tune_ab1.json
  py tools/gen_k3_tune.py ship <n> k=v.. -> kernels/lab/k3_tune_<n>.py (gc_16 with whole-line table replacements)
The lab file only widens what prepare builds / self-tests (VML w72 self-test at bucket 32, VSL ks3 (8, 1), KS3 NS 16,
VF_SBB2 bucket 32 self-test); every ship-arm route is unchanged. Never overwrites a file.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/muse_decode_layer/k3_gc_16.py"


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def write_new(p, text):
    assert not p.exists(), f"{p} exists"
    p.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {p} ({len(text.encode())} B)")


LAB_SUBS = [
    ("VML_ABVARS = ()\n", "VML_ABVARS = ('w72',)\n"),
    ("VML_BVAR = {256: 'w72', 128: 'w72'}\n", "VML_BVAR = {256: 'w72', 128: 'w72', 32: 'bf2'}\n"),
    ("KS3_INSTS = (8,)\n", "KS3_INSTS = (8, 16)\n"),
    ("VSL_KS3_INSTS = ((16, 1),)\n", "VSL_KS3_INSTS = ((16, 1), (8, 1))\n"),
    ("VF_SBB2 = {1: True, 8: True, 16: False}\n", "VF_SBB2 = {1: True, 8: True, 16: False, 32: False}\n"),
]

GD = {("qkv", 1): (1, 1, 4, 8, 2, 4, 1), ("qkv", 8): (1, 1, 4, 8, 2, 4, 1), ("qkv", 16): (2, 1, 4, 8, 1, 4, 0),
      ("qkv", 32): (4, 1, 4, 8, 1, 4, 0),
      ("o", 1): (1, 2, 4, 4, 2, 1, 0), ("o", 8): (1, 1, 8, 2, 2, 1, 0), ("o", 16): (2, 2, 4, 4, 2, 1, 0),
      ("pooler", 1): (1, 1, 8, 16, 2, 1, 1), ("pooler", 8): (1, 1, 8, 16, 2, 1, 1),
      ("unpooler", 1): (1, 1, 16, 4, 2, 1, 0), ("unpooler", 8): (1, 1, 16, 4, 2, 1, 0),
      ("unpooler", 16): (2, 2, 4, 16, 2, 1, 0), ("unpooler", 32): (4, 2, 4, 16, 1, 1, 0)}
VSL = {(1, 1): ("ks1", 1, 16, 1, 1), (1, 8): ("ks1", 8, 16, 4, 1), (16, 1): ("ks3", 16, 1)}
RT = {1: "f32", 8: "f32", 16: "gemv", 32: "stock", 128: "kcat", 256: "kcat"}
PL = {1: "side", 8: "side_first", 16: "side", 32: "side", 128: "side", 256: "main"}


def js(d):
    if isinstance(d, dict):
        return {(",".join(str(p) for p in k) if isinstance(k, tuple) else str(k)): js(v) for k, v in d.items()}
    if isinstance(d, tuple):
        return [js(x) for x in d]
    return d


def without(d, *keys):
    return {k: (None if k in keys else v) for k, v in d.items()}


def arms():
    vsl_a = dict(VSL)
    del vsl_a[(16, 1)]
    return [
        ("ship", {}),
        ("mid", {}),
        # MoE: VML variant per bucket (32 -> w72; 128 / 256 -> bf2); VMS at T <= 8 only (T 16 -> K2A v3.6)
        ("vmlx", {"VML_BVAR": {256: "bf2", 128: "bf2", 32: "w72"}}),
        ("vms8", {"VMS_MAXT": 8}),
        # attention: d16s VSL off (KS3 NS 8); d32s VSL ks3 n16; B=1 full FA4 splits 64 (auto 128); VA1 segk 560
        ("attA", {"VSL_ROUTE": {**vsl_a, (32, 1): ("ks3", 16, 1)}, "K1FA4_SPLITS": {("sliding", 1): 32, ("full", 1): 64},
                  "VA1_SEGK": 560}),
        # d16s KS3 NS 16; d32s VSL ks3 n8; FA4 full B=1 96 splits; VFA segk 420
        ("attB", {"VSL_ROUTE": {**vsl_a, (32, 1): ("ks3", 8, 1)}, "KS3_NS": {(32, 1): 8, (16, 1): 16},
                  "K1FA4_SPLITS": {("sliding", 1): 32, ("full", 1): 96}, "VFA_SEGK": 420}),
        # d32s KS3 NS 16; ATT1 PW off (B=1 full); VA1 segk 480; VFA segk 360
        ("attC", {"KS3_NS": {(32, 1): 16, (16, 1): 8}, "ATT1_PW": {}, "VA1_SEGK": 480, "VFA_SEGK": 360}),
        # router GEMM route / placement per bucket
        ("rtA", {"V8_ROUTER_ROUTE": {**RT, 16: "kcat", 32: "kcat", 128: "stock", 256: "stock"}}),
        ("rtB", {"V8_ROUTER_ROUTE": {**RT, 16: "stock"}, "V8_ROUTER_PLACE": {**PL, 128: "main", 256: "side"}}),
        ("rtC", {"V8_ROUTER_ROUTE": {**RT, 32: "kcat"},
                 "V8_ROUTER_PLACE": {**PL, 16: "main", 32: "main", 128: "side_first", 256: "side_first"}}),
        ("mid2", {}),
        # coordinator 2026-10-29: boundary (norm / sbb) and GD unpooler / o arms dropped (owned by a builder).
        # vm_main as PDL secondary at T = 32 only (R-vml-11 "pdl32", VML flags bit 8 = 256; bit 8 fails at T >= 128)
        ("pdl32", {"VML_FLAGS": {128: 193, 256: 193, 32: 193 | 256}}),
        # dense: GD off per (proj, T 16 / 32) -> the prepared fallback (P1M at 16, DG1 at 32)
        ("gdA", {"GD_ROUTE": without(GD, ("qkv", 16), ("qkv", 32))}),
        ("gdB", {"GD_CAT": {16: False, 32: False}}),
        ("mid3", {}),
    ]


def main():
    text = SRC.read_text(encoding="utf-8")
    if sys.argv[1] == "lab":
        for o, n in LAB_SUBS:
            text = sub(text, o, n)
        if not (ROOT / "kernels/lab/k3_tune_lab1.py").exists():
            write_new(ROOT / "kernels/lab/k3_tune_lab1.py", text)
        a = [{"name": n, "set": js(s)} for n, s in arms()]
        p = ROOT / "tools/box/tune_ab2.json"
        assert not p.exists()
        p.write_text(json.dumps(a, indent=1) + "\n")
        print(f"wrote {p}: {len(a)} arms")
    elif sys.argv[1] == "ship":
        n = sys.argv[2]
        for kv in sys.argv[3:]:   # NAME=python-literal: replaces the whole 'NAME = ...' line (single-line tables only)
            k, v = kv.split("=", 1)
            lines = [ln for ln in text.split("\n") if ln.startswith(f"{k} = ")]
            assert len(lines) == 1, k
            text = sub(text, lines[0] + "\n", f"{k} = {v}\n")
        write_new(ROOT / f"kernels/lab/k3_tune_{n}.py", text)


if __name__ == "__main__":
    main()

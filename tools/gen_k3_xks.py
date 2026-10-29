"""gen_k3_xks.py: XKS patch (tools/xks_patches.py; all XKS_* default 0 = shipped code) on a layer's VML block (default base
k3_gc_18).

  python -I tools/gen_k3_xks.py --ab [--base=k3_gc_18] [--out=k3_xks_ab]   # kernels/lab/<out>.py: variants bf2xks / w72xks
        (VML_ABVARS; VML_BVAR[32] in the file so prepare self-tests bucket 32) + tools/box/ab_xks.json (ship, xks, mid, mid2).
  python -I tools/gen_k3_xks.py --ship --out=k3_xks_1                      # bf2 and w72 themselves carry -DXKS=1.
Refuses to overwrite (--dry: checks only).
"""
import ast
import hashlib
import json
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/box")]
import m32e_patches as P  # noqa: E402
import xks_patches as X  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_18')}.py"
AB = "--ab" in sys.argv
OUTN = ARGS.get("out", "k3_xks_ab" if AB else "k3_xks_1")
OUT = ROOT / f"kernels/lab/{OUTN}.py"
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_k3_xks] FAIL {msg}")


def main():
    check("--dry" in sys.argv or not OUT.exists(), f"{OUT.name} exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    check(bn.startswith("# nonce "), "line 2 is not a nonce line")
    a0, b0 = P.cuda_span(b, "_vml_CUDA_T")
    t = b[:a0] + X.patch(b[a0:b0]) + b[b0:]
    defs = bvar = None
    for n in ast.parse(t).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id == "_VML_DEFS":
                defs = ast.literal_eval(n.value)
            elif n.targets[0].id == "VML_BVAR":
                bvar = ast.literal_eval(n.value)
    check(bvar == {256: "w72", 128: "w72"}, f"VML_BVAR {bvar}")
    olds = {v: f'"{v}": (' + ", ".join(f'"{f}"' for f in defs[v]) + ")" for v in ("bf2", "w72")}
    check(all(t.count(o) == 1 for o in olds.values()), "_VML_DEFS bf2 / w72 entries not verbatim")
    check(t.count("\nVML_VAR = 'bf2'\n") == 1, "VML_VAR")
    if AB:
        names = ["bf2xks", "w72xks"]
        add = "".join("," + NL + "             " + f'"{v}xks": (' + ", ".join(f'"{f}"' for f in list(defs[v]) + ["-DXKS=1"]) + ")"
                      for v in ("bf2", "w72"))
        t = t.replace(olds["w72"], olds["w72"] + add, 1)
        sx = '"bf2", "w72") else _VML["sc"]'
        check(t.count(sx) == 1, "scx tuple")
        t = t.replace(sx, '"bf2", "w72", ' + ", ".join(f'"{n}"' for n in names) + ') else _VML["sc"]')
        o_ab = NL + "VML_ABVARS = ()" + NL
        check(t.count(o_ab) == 1, "VML_ABVARS")
        t = t.replace(o_ab, NL + "VML_ABVARS = " + repr(tuple(names)) + NL)
        o_bv = NL + "VML_BVAR = {256: 'w72', 128: 'w72'}" + NL
        check(t.count(o_bv) == 1, "VML_BVAR line")
        t = t.replace(o_bv, NL + "VML_BVAR = {256: 'w72', 128: 'w72', 32: 'bf2xks'}" + NL)   # bucket 32 enters the prepare self-test
        nonce = f"# nonce {OUTN.replace('_', '-')} (A/B copy: {BASE.stem} + XKS patch, variants {', '.join(names)}; tools/gen_k3_xks.py)"
        rep = f"A/B variants {names}"
        sh = {"256": "w72", "128": "w72", "32": "bf2"}
        xk = {"256": "w72xks", "128": "w72xks", "32": "bf2xks"}
        js = [{"name": "ship", "set": {"VML_BVAR": sh}}, {"name": "xks", "set": {"VML_BVAR": xk}},
              {"name": "mid", "set": {"VML_BVAR": sh}}, {"name": "xks2", "set": {"VML_BVAR": xk}}]
    else:
        for v in ("bf2", "w72"):
            t = t.replace(olds[v], olds[v][:-1] + ', "-DXKS=1")', 1)
        nonce = f"# nonce {OUTN.replace('_', '-')} ({BASE.stem} + XKS K-split stage_m in VML bf2 + w72: -DXKS=1; tools/gen_k3_xks.py)"
        rep = "ship XKS"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(max(len(x) for x in t.split(NL)) <= max(len(x) for x in b.split(NL)), "line longer than the base's longest")
    check(re.search(r"importlib|sys\.modules", t[a0:a0 + 200000]) is None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_xks] OK {rep}; {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    OUT.write_text(t, encoding="utf-8", newline=NL)
    if AB:
        jp = ROOT / "tools/box/ab_xks.json"
        check(not jp.exists(), "tools/box/ab_xks.json exists")
        jp.write_text("[" + ("," + NL + " ").join(json.dumps(x) for x in js) + "]" + NL)
    print(f"[gen_k3_xks] wrote {OUT.relative_to(ROOT)} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""gen_k3_vcp.py: VCP consumer patch (tools/box/vcp_run.py patch(), all VCP_* default = shipped code) on a layer's VML block
(default base k3_gc_10) + w72 variants carrying vcp_run ARMS defines.

  python -I tools/gen_k3_vcp.py --ab=c1,c2 [--base=k3_gc_10] [--out=k3_vcp_ab]   # kernels/lab/<out>.py: variants w72<arm>
        (VML_ABVARS) + tools/box/ab_vcp.json (ship, arms, mid, mid2; arms switch VML_BVAR at 128 / 256).
  python -I tools/gen_k3_vcp.py --ship=c1 [--base=k3_gc_10] --out=k3_vcp_1           # w72 itself carries the arm's defines.
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
import vcp_run as V  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_10')}.py"
AB = "ab" in ARGS
OUTN = ARGS.get("out", "k3_vcp_ab" if AB else "k3_vcp_1")
OUT = ROOT / f"kernels/lab/{OUTN}.py"
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_k3_vcp] FAIL {msg}")


def main():
    check("--dry" in sys.argv or not OUT.exists(), f"{OUT.name} exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    check(bn.startswith("# nonce "), "line 2 is not a nonce line")
    a0, b0 = P.cuda_span(b, "_vml_CUDA_T")
    t = b[:a0] + V.patch(b[a0:b0]) + b[b0:]
    defs = bvar = None
    for n in ast.parse(t).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id == "_VML_DEFS":
                defs = ast.literal_eval(n.value)
            elif n.targets[0].id == "VML_BVAR":
                bvar = ast.literal_eval(n.value)
    check(bvar == {256: "w72", 128: "w72"}, f"VML_BVAR {bvar}")
    old = '"w72": (' + ", ".join(f'"{f}"' for f in defs["w72"]) + ")"
    check(t.count(old) == 1, "_VML_DEFS w72 entry not verbatim")
    if AB:
        arms = ARGS["ab"].split(",")
        check(all(V.ARMS.get(a) for a in arms), f"arms {arms}")
        names = [f"w72{a}" for a in arms]
        add = "".join("," + NL + "             " + f'"w72{a}": (' + ", ".join(f'"{f}"' for f in list(defs["w72"]) + list(V.ARMS[a])) + ")"
                      for a in arms)
        t = t.replace(old, old + add, 1)
        sx = '"bf2", "w72") else _VML["sc"]'
        check(t.count(sx) == 1, "scx tuple")
        t = t.replace(sx, '"bf2", "w72", ' + ", ".join(f'"{n}"' for n in names) + ') else _VML["sc"]')
        o_ab = NL + "VML_ABVARS = ()" + NL
        check(t.count(o_ab) == 1, "VML_ABVARS")
        t = t.replace(o_ab, NL + "VML_ABVARS = " + repr(tuple(names)) + NL)
        nonce = f"# nonce {OUTN.replace('_', '-')} (A/B copy: {BASE.stem} + VCP patch, variants {', '.join(names)}; tools/gen_k3_vcp.py)"
        rep = f"A/B variants {names}"
        sh = {"256": "w72", "128": "w72", "32": "bf2"}
        js = [{"name": "ship", "set": {"VML_BVAR": sh}}]
        for i, a in enumerate(arms):
            js.append({"name": a, "set": {"VML_BVAR": dict(sh, **{"256": f"w72{a}", "128": f"w72{a}"})}})
            if i == 0:
                js.append({"name": "mid", "set": {"VML_BVAR": sh}})
        js.append({"name": "mid2", "set": {"VML_BVAR": sh}})
    else:
        a = ARGS["ship"]
        check(bool(V.ARMS.get(a)), f"arm {a}")
        t = t.replace(old, old[:-1] + ", " + ", ".join(f'"{f}"' for f in V.ARMS[a]) + ")", 1)
        nonce = f"# nonce {OUTN.replace('_', '-')} ({BASE.stem} + VCP arm {a}: {' '.join(V.ARMS[a])} in VML w72; tools/gen_k3_vcp.py)"
        rep = f"ship arm {a}"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t[a0:a0 + 200000]) is None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_vcp] OK {rep}; {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    OUT.write_text(t, encoding="utf-8", newline=NL)
    if AB:
        jp = ROOT / "tools/box/ab_vcp.json"
        check(not jp.exists(), "tools/box/ab_vcp.json exists")
        jp.write_text("[" + ("," + NL + " ").join(json.dumps(x) for x in js) + "]" + NL)
    print(f"[gen_k3_vcp] wrote {OUT.relative_to(ROOT)} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

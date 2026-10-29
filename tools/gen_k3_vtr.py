"""gen_k3_vtr.py: VTR riders (tools/vtr_patches.py: vm_fin entry trigger, vm_main trigger after its wait, vm_fin trims,
tile-aligned route() job split) on a layer's VML block (default base k3_gc_9). Compile-time flags, default off.

  python -I tools/gen_k3_vtr.py --ab [--base=k3_gc_9] [--out=k3_vtr_ab]      # A/B copy kernels/lab/<out>.py: the base's live
        variants (bf2, w72) + <v>A (all riders), <v>F (FT), <v>R (FTR), <v>L (LD 2), <v>T (TA); arms switch VML_BVAR.
  python -I tools/gen_k3_vtr.py --ship=FT,FTR,LD,TA [--base=k3_gc_9] --out=k3_vtr_1   # kernels/lab/<out>.py with those riders on
        in every live variant (bf2, w72). Refuses to overwrite (--dry: checks only).
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import m32e_patches as P  # noqa: E402
import vtr_patches as V  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_9')}.py"
AB = "--ab" in sys.argv
OUTN = ARGS.get("out", "k3_vtr_ab" if AB else "k3_vtr_1")
OUT = ROOT / f"kernels/lab/{OUTN}.py"
NL = chr(10)
RID = {"FT": ("-DVM_FT=1",), "FTR": ("-DVM_FTR=1",), "LD": ("-DVM_LD=2",), "TA": ("-DVM_TA=1",)}
ARMS = {"A": ("FT", "FTR", "LD", "TA"), "B": ("FT", "FTR", "TA"), "C": ("FTR", "LD", "TA"), "F": ("FT",), "R": ("FTR",), "L": ("LD",), "T": ("TA",)}


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_k3_vtr] FAIL {msg}")


def fl(names):
    return tuple(f for n in names for f in RID[n])


def main():
    check("--dry" in sys.argv or not OUT.exists(), f"{OUT.name} exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    check(bn.startswith("# nonce "), "line 2 is not a nonce line")
    a0, b0 = P.cuda_span(b, "_vml_CUDA_T")
    t = b[:a0] + V.apply(b[a0:b0]) + b[b0:]
    var = bvar = defs = None
    for n in ast.parse(t).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
            if nm == "VML_VAR":
                var = ast.literal_eval(n.value)
            elif nm == "VML_BVAR":
                bvar = ast.literal_eval(n.value)
            elif nm == "_VML_DEFS":
                defs = ast.literal_eval(n.value)
    live = [var] + sorted(set(bvar.values()) - {var})
    check(live == ["bf2", "w72"], f"live variants {live}")
    for v in live:
        check(any(f == "-DVM_RT=1" for f in defs[v]) and not any(f.startswith(("-DVM_LD=", "-DVM_FT", "-DVM_TA")) for f in defs[v]),
              f"variant {v} flags {defs[v]}")
    old = {v: f'"{v}": (' + ", ".join(f'"{f}"' for f in defs[v]) + ")" for v in live}
    for v in live:
        check(t.count(old[v]) == 1, f"_VML_DEFS entry {v} not found verbatim")
    if AB:
        names = []
        for v in live:
            add = "".join("," + NL + "             " + f'"{v}{k}": (' + ", ".join(f'"{f}"' for f in defs[v] + fl(r)) + ")"
                          for k, r in ARMS.items())
            t = t.replace(old[v], old[v] + add, 1)
            names += [f"{v}{k}" for k in ARMS]
        sx = '"bf2", "w72") else _VML["sc"]'
        check(t.count(sx) == 1, "scx tuple")
        t = t.replace(sx, '"bf2", "w72", ' + ", ".join(f'"{n}"' for n in names) + ') else _VML["sc"]')
        o_ab = NL + "VML_ABVARS = ()" + NL
        check(t.count(o_ab) == 1, "VML_ABVARS")
        t = t.replace(o_ab, NL + "VML_ABVARS = " + repr(tuple(names)) + NL)
        bv = dict(bvar)
        bv.setdefault(32, var)
        t = t.replace(NL + "VML_BVAR = " + repr(bvar) + NL, NL + "VML_BVAR = " + repr(bv) + NL, 1)
        check(("VML_BVAR = " + repr(bv)) in t, "VML_BVAR")
        nonce = f"# nonce {OUTN.replace('_', '-')} (A/B copy: {BASE.stem} + VTR riders, variants {', '.join(names)}; tools/gen_k3_vtr.py)"
        rep = f"A/B variants {names}"
    else:
        # --ship=FT,FTR (every live variant) or --ship=bf2:FT,FTR,TA/w72:FT,FTR,LD,TA (per variant)
        spec = ARGS.get("ship", "")
        per = {v: [s for s in (dict(x.split(":", 1) for x in spec.split("/")).get(v, "") if ":" in spec else spec).split(",") if s]
               for v in live}
        ship = sorted({s for v in live for s in per[v]})
        check(ship and all(s in RID for s in ship), f"--ship={spec}")
        for v in live:
            if per[v]:
                t = t.replace(old[v], old[v][:-1] + ", " + ", ".join(f'"{f}"' for f in fl(per[v])) + ")", 1)
        ship = [f"{v}:{'+'.join(per[v])}" for v in live]
        nonce = f"# nonce {OUTN.replace('_', '-')} ({BASE.stem} + VTR riders {'+'.join(ship)} in VML {', '.join(live)}; tools/gen_k3_vtr.py)"
        rep = f"ship riders {ship}"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t[a0:a0 + 200000]) is None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_vtr] OK {rep}; {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    OUT.write_text(t, encoding="utf-8", newline=NL)
    print(f"[gen_k3_vtr] wrote {OUT.relative_to(ROOT)} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""gen_k3_rt.py: k3_gc_<n> = a layer (default k3_gc_6) + VM_RT (tools/rt_patches.py: route() in two CTA barriers) in its VML
block, switched on (-DVM_RT=1) for the compiled variants (VML_VAR and every VML_BVAR variant). Compile-time only; at
VM_RT 0 the source is the base's. Writes kernels/muse_decode_layer/<out>.py and kernels/lab/<out>_ab.py (A/B copy: the base's variants
+ "<v>r" = <v> + VM_RT, switched per arm through VML_BVAR); refuses to overwrite.

    python -I tools/gen_k3_rt.py --base=k3_gc_6 --out=k3_gc_7 [--dry]
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
import rt_patches as R  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_6')}.py"
OUTN = ARGS.get("out", "k3_gc_7")
OUT, LAB = ROOT / f"kernels/muse_decode_layer/{OUTN}.py", ROOT / f"kernels/lab/{OUTN}_ab.py"
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_k3_rt] FAIL {msg}")


def main():
    check("--dry" in sys.argv or not (OUT.exists() or LAB.exists()), f"{OUT.name} or its lab copy exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    check(bn.startswith("# nonce "), "line 2 is not a nonce line")
    a0, b0 = P.cuda_span(b, "_vml_CUDA_T")
    t = b[:a0] + R.apply(b[a0:b0]) + b[b0:]
    tree = ast.parse(t)
    var = bvar = defs = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
            if nm == "VML_VAR":
                var = ast.literal_eval(n.value)
            elif nm == "VML_BVAR":
                bvar = ast.literal_eval(n.value)
            elif nm == "_VML_DEFS":
                defs = ast.literal_eval(n.value)
    live = [var] + sorted(set(bvar.values()) - {var})
    rep = []
    for v in live:
        check(not any(f.startswith(("-DVM_DP=", "-DVM_SC=")) for f in defs[v]), f"variant {v} has a dynamic / SC plan")
        old = f'"{v}": (' + ", ".join(f'"{f}"' for f in defs[v]) + ")"
        check(t.count(old) == 1, f"_VML_DEFS entry {v} not found verbatim")
        t = t.replace(old, old[:-1] + ', "-DVM_RT=1")')
        rep.append(f"{v}: -DVM_RT=1")
    # A/B copy (kernels/lab/<out>_ab.py): the BASE's variants unchanged + "<v>r" = <v> + VM_RT for each live variant, all built
    # in one prepare (ABVARS), bucket 32 explicit in VML_BVAR so its (32, <v>r) pair is self-tested too; arms switch VML_BVAR.
    ab = b[:a0] + R.apply(b[a0:b0]) + b[b0:]
    for v in live:
        old = f'"{v}": (' + ", ".join(f'"{f}"' for f in defs[v]) + ")"
        ab = ab.replace(old, old + "," + NL + "             " + f'"{v}r": (' + ", ".join(f'"{f}"' for f in defs[v]) + ', "-DVM_RT=1")', 1)
    sx = '"bf2", "w72") else _VML["sc"]'
    check(ab.count(sx) == 1, "scx tuple")
    ab = ab.replace(sx, '"bf2", "w72", ' + ", ".join(f'"{v}r"' for v in live) + ') else _VML["sc"]')
    o_ab = NL + "VML_ABVARS = ()" + NL
    check(ab.count(o_ab) == 1, "VML_ABVARS")
    ab = ab.replace(o_ab, NL + "VML_ABVARS = " + repr(tuple(f"{v}r" for v in live)) + NL)
    bv = dict(bvar)
    bv.setdefault(32, var)
    ab = ab.replace(NL + "VML_BVAR = " + repr(bvar) + NL, NL + "VML_BVAR = " + repr(bv) + NL, 1)
    check(("VML_BVAR = " + repr(bv)) in ab, "VML_BVAR")
    ab = ab.replace(bn, f"# nonce {OUTN.replace('_', '-')}-ab (A/B copy: {BASE.stem} variants + " + ", ".join(f"{v}r" for v in live) + " = VM_RT)", 1)
    tag = OUTN.replace("_", "-")
    nonce = f"# nonce {tag} ({BASE.stem} + VM_RT route() in two barriers for VML {', '.join(live)}; tools/gen_k3_rt.py)"
    t = t.replace(bn, nonce, 1)
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    check(t.split(NL)[1] == nonce and re.search(r"importlib|sys\.modules", R.RT_SRC) is None, "nonce / checker")
    print(f"[gen_k3_rt] OK  {'; '.join(rep)}; {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    OUT.write_text(t, encoding="utf-8", newline=NL)
    LAB.write_text(ab, encoding="utf-8", newline=NL)
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(ab, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_rt] wrote {OUT.relative_to(ROOT)} + {LAB.relative_to(ROOT)} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

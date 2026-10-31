"""vmlsm_patches.py (R-vmlsm-0, lab file only): VML bf2 dynamic-smem variants for the d32 smem-class question (R-hybsm-2).

  python -I tools/vmlsm_patches.py [--base=kernels/lab/k3_hh_3tkv.py] [--out=kernels/lab/k3_vmlsm_ab.py] [--check]
  python -I tools/lead_gate.py file <out>

R-hybsm-2: the same HYB code with 47.6 KB of unused dynamic smem (196 -> 228 KB carveout class) costs +7.85 us at d32s. VML bf2 asks for
about 213 KB (3 x 36,864 weight ring + 3 x 32,768 x ring + 4.3 KB table) = the 228 KB class. Two extra VML builds, FLAGS ONLY (no CUDA
source edit, so every existing build keeps its cache tag and the pod compiles just these two):
  bx2 = bf2 + -DVM_NX=2    x ring 3 -> 2 slots, about 180 KB: the 196 KB class (limit 199,680 B = 200,704 - 1,024 reserve)
  bxp = bf2 + -DVM_XSN=72  x slot 64 -> 72 rows, +12,288 B of rows the MMA never reads: still the 228 KB class (a within-class dose)
Host-side edits only: _VML_DEFS entries, VML_ABVARS = (bx2, bxp), VML_BVAR gains 32: 'bf2' (production routing unchanged; it only makes
_m32e_vartest self-test the extra variants at bucket 32 vs bf2 and FP32), VML32_VAR (None in production) routes bucket 32 to a variant
for an A/B arm, and each variant gets its own scratch ("scv", the vfx lesson: no scratch sharing between VML builds). A variant that
fails its self-test falls back to bf2 silently inside _vml_route, so the runner must read _VML['okv'] before trusting an arm.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[vmlsm_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def apply(s, base_name):
    lines = s.split(NL)
    lines[1] = f"# nonce k3-vmlsm-ab ({base_name} + R-vmlsm-0 lab: VML bf2 variants bx2 (x ring 2) / bxp (+12 KB rows); tools/vmlsm_patches.py)"
    s = NL.join(lines)
    s = sub(s, "VML_ABVARS = ()\n", 'VML_ABVARS = ("bx2", "bxp")\nVML32_VAR = None\n', "VML_ABVARS")
    s = sub(s, "VML_BVAR = {256: 'w72', 128: 'w72'}\n", "VML_BVAR = {256: 'w72', 128: 'w72', 32: 'bf2'}\n", "VML_BVAR")
    old = '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, "scw": None, "ok": {}, "err": None, "okv": {}, "force": None}\n'
    new = ('_VML_DEFS["bx2"] = _VML_DEFS["bf2"] + ("-DVM_NX=2",)\n'
           '_VML_DEFS["bxp"] = _VML_DEFS["bf2"] + ("-DVM_XSN=72",)\n'
           '_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, "scw": None, "scv": {}, "ok": {}, "err": None, "okv": {}, "force": None}\n')
    s = sub(s, old, new, "_VML dict")
    old = '    var = _VML["force"] if _VML["force"] is not None else VML_BVAR.get(rep, VML_VAR)\n'
    new = old + '    if rep == 32 and VML32_VAR is not None and _VML["force"] is None:\n        var = VML32_VAR\n'
    s = sub(s, old, new, "route override")
    old = '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n'
    new = '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72", "bx2", "bxp") else _VML["sc"]\n'
    s = sub(s, old, new, "scratch tuple")
    old = '    if var == "w72" and _VML.get("scw") is not None:\n        scx = _VML["scw"]\n'
    new = old + '    if var in _VML["scv"]:\n        scx = _VML["scv"][var]\n'
    s = sub(s, old, new, "scratch override")
    old = '        _VML["scw"] = dict(_vml_alloc(dev, sms), err=_VML["sc"]["err"])\n'
    new = old + '        _VML["scv"] = {v: dict(_vml_alloc(dev, sms), err=_VML["sc"]["err"]) for v in ("bx2", "bxp")}\n'
    s = sub(s, old, new, "scratch alloc")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3tkv.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_vmlsm_ab.py")
    s = apply(base.read_bytes().decode("ascii"), base.stem)
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check(len(s.encode()) < LIMIT, f"size {len(s.encode()):,} >= {LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s, f"{out.name} != apply({base.name})")
        print(f"[vmlsm_patches] --check PASS: {out.name} == apply({base.name}) ({len(s):,} B)")
        return
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[vmlsm_patches] {base.name} -> {out.name} ({len(s):,} B, {len(s) - base.stat().st_size:+,} B)")


if __name__ == "__main__":
    main()

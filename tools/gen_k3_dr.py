"""gen_k3_dr.py: k3_dr_<n> = k3_gc_6 + the VM_DR switches (tools/dr_patches.py) with VML_BVAR[32] = --var (default drc =
VM_DR + setmaxnreg 64 / 104, the spill-free build), buckets 128 / 256 unchanged (w72). Prepare builds the selected VML
variants in parallel (gc_6's _vml_build_job), the VM_DRP weight copy only if a drp variant is selected, and
_m32e_vartest self-tests (32, var) at bs 25 / 32 / 48 against bf2 (< 2e-3) and FP32 (< 1e-2, row < 5e-2), two calls,
error flag; a failing pair runs bf2. The output goes through gen_k3_m32.finish (strip_layer / strip_cxx / judge-checker
scan) when that chain imports; otherwise through the same strips and the standalone checks here (compiles, ASCII, line 1,
nonce, no duplicate top-level names, < 1 MB). Refuses to overwrite.

    python -I tools/gen_k3_dr.py --n=1 [--var=drc] [--ship]   # kernels/lab/k3_dr_1_ab.py (--ship: + muse_decode_layer/k3_dr_1.py)
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
import dr_patches as P  # noqa: E402

N = ARGS.get("n", "1")
VAR = ARGS.get("var", "drc")
OUTN = f"k3_dr_{N}"
OUT = ROOT / f"kernels/muse_decode_layer/{OUTN}.py"
LAB = ROOT / f"kernels/lab/{OUTN}_ab.py"
TAG = OUTN.replace("_", "-")
NL = chr(10)


def check(cond, msg):
    if not cond:
        raise SystemExit(f"[gen_k3_dr] FAIL: {msg}")


def _protected(text):
    """short string constants with a `//` line the Python code searches for at run time (kept by strip_cxx; gen_k3_m32)."""
    out = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 400 and "//" in n.value:
            for line in n.value.splitlines():
                if line.strip().startswith("//"):
                    out.add(line.strip())
    return sorted(out)


def finish_local(text, nonce, tag, rep):
    import strip_cxx
    import strip_layer
    s1 = strip_layer.strip(text)
    ok, msg = strip_layer.equivalent(text, s1)
    check(ok, f"strip_layer: {msg}")
    prot = _protected(s1)
    lic0 = strip_cxx.LICENSE
    strip_cxx.LICENSE = re.compile("|".join([lic0.pattern] + [re.escape(p) for p in prot]), lic0.flags)
    try:
        s2 = strip_cxx.strip(s1)
        ok2, msg2 = strip_cxx.equivalent(s1, s2)
    finally:
        strip_cxx.LICENSE = lic0
    check(ok2, f"strip_cxx: {msg2}")
    for p in prot:
        check(s2.count(p) == s1.count(p), f"{tag}: strip_cxx dropped a searched-for line {p!r}")
    tmp = ROOT / f"kernels/lab/{tag}.gen_tmp.py"
    tmp.write_text(s2, encoding="utf-8", newline=NL)
    try:
        py_compile.compile(str(tmp), doraise=True)
    finally:
        tmp.unlink()
    check(s2.isascii(), f"{tag}: non-ASCII character")
    check(s2.split(NL, 1)[0] == "# okc: problem=muse_decode_layer", f"{tag}: line 1")
    check(s2.split(NL)[1] == nonce and s2.count(tag) == 1 and "k3-gc-6" not in s2, f"{tag}: nonce")
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s2).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    check(not dup, f"{tag}: duplicate top-level names {dup}")
    check(len(s2.encode()) < 1_000_000, f"{tag}: {len(s2.encode())} B >= 1 MB")
    for bad in ("importlib", "sys.modules"):
        check(bad not in s2, f"{tag}: judge checker rejects {bad!r}")
    rep.append(f"{tag}: {len(text.encode())} -> strip_layer {len(s1.encode())} -> strip_cxx {len(s2.encode())} B "
               "(local finish: gen_k3_m32 chain not importable here; run on a full checkout for the judge-checker scan)")
    return s2


def main():
    check(VAR in P.DEFS and VAR not in ("bf2",), f"--var must be one of {sorted(set(P.DEFS) - {'bf2'})}")
    check("--dry" in sys.argv or (not LAB.exists() and ("--ship" not in sys.argv or not OUT.exists())),
          f"{LAB.name} or {OUT.name} exists; pick another --n")
    b = P.BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    t = P.patch_layer(b, {256: "w72", 128: "w72", 32: VAR})
    rep = [f"VML CUDA: {len(P.CUDA_PATCHES)} anchored VM_DR patches (VM_DR 0 preprocesses to the base: dr_patches check)",
           f"glue: VML_BVAR {{256: 'w72', 128: 'w72', 32: {VAR!r}}}, variants {sorted(set(P.DEFS) - {'bf2'})}, "
           "VM_DRP repack in prepare only if selected, _m32e_vartest self-test with bf2 fallback"]
    nonce = (f"# nonce {TAG} (k3_gc_6 + VML VM_DR T32 double-row at bucket 32 = {VAR!r} (512-row stages, 2 x 64-row tiles per "
             "WG, jobs <= 32 tokens), w72 at 128 / 256; tools/gen_k3_dr.py)")
    t = t.replace(bn, nonce)
    try:
        _argv, sys.argv = sys.argv, [sys.argv[0]]
        try:
            import gen_k3_m32 as GM
        finally:
            sys.argv = _argv
        _, chk_base, _ = GM.G37.generate()
        s = GM.finish(t, nonce, TAG, chk_base, rep)
    except ImportError as ex:
        rep.append(f"gen_k3_m32 chain unavailable ({ex}); local finish")
        s = finish_local(t, nonce, TAG, rep)
    for r in rep:
        print(f"[gen_k3_dr] OK  {r}")
    if "--dry" in sys.argv:
        print(f"[gen_k3_dr] --dry: nothing written ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B)")
        return 0
    LAB.write_text(s, encoding="utf-8", newline=NL)
    if "--ship" in sys.argv:
        OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_dr] wrote {LAB.relative_to(ROOT)}{' and ' + str(OUT.relative_to(ROOT)) if '--ship' in sys.argv else ''} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

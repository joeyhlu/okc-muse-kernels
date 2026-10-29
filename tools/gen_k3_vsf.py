"""gen_k3_vsf.py: kernels/lab/<out>.py = a layer (default k3_gc_7) + VMS_FX (tools/vsf_patches.py: VMS fixed-latency chain
cuts, compile-time bitmask, built at -DVMS_FX=<fx>), and kernels/lab/<out>_ab.py (A/B copy: the base VMS + the FX build in one
prepare, switched per arm by the module global VMS_VAR "base" | "fx"; prepare self-tests that FX is bitwise equal to base).
At VMS_FX 0 the patched source preprocesses to the base's. Refuses to overwrite.

    python -I tools/gen_k3_vsf.py --base=k3_gc_7 --out=k3_vsf_1 [--fx=7] [--dry]
"""
import hashlib
import pathlib
import py_compile
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vsf_patches as F  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_7')}.py"
OUTN = ARGS.get("out", "k3_vsf_1")
FX = int(ARGS.get("fx", 7))
OUT, LAB = ROOT / f"kernels/lab/{OUTN}.py", ROOT / f"kernels/lab/{OUTN}_ab.py"
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_k3_vsf] FAIL {msg}")


def rep(t, old, new, what):
    check(t.count(old) == 1, f"{what}: {t.count(old)} matches")
    return t.replace(old, new)


def pycheck(t):
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)


def main():
    check("--dry" in sys.argv or not (OUT.exists() or LAB.exists()), f"{OUT.name} or its A/B copy exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    check(bn.startswith("# nonce "), "line 2 is not a nonce line")
    k0 = 'VMS_CUDA_SRC = r"""'
    a0 = b.index(k0) + len(k0)
    b0 = b.index('"""', a0)
    src0 = b[a0:b0]
    src1 = F.apply_fx(src0)
    check(F.check_fx0(src0, src1), "VMS_FX 0 view differs from the base source")
    t = b[:a0] + src1 + b[b0:]
    fl = '"-DVMS_NS=3", "-Xptxas=-warn-spills"'
    tag = OUTN.replace("_", "-")
    # ship candidate: FX build only
    s = rep(t, fl, f'"-DVMS_NS=3", "-DVMS_FX={FX}", "-Xptxas=-warn-spills"', "ship flags")
    s = s.replace(bn, f"# nonce {tag} ({BASE.stem} + VMS_FX={FX}: VMS acquire poll / acq_rel fences / early first copies; "
                      f"tools/gen_k3_vsf.py)", 1)
    # A/B copy: base + fx in one prepare
    ab = rep(t, '_VMS = {"mod": None, "G": None}', '_VMS = {"mod": None, "G": None, "mods": {}}', "_VMS")
    old = ('    _VMS["G"] = G' + NL + '    _VMS["mod"] = mod' + NL)
    new = ('    flx = [f for f in flags if f != "-DVMS_NS=3"] + ["-DVMS_NS=3", "-DVMS_FX=' + str(FX) + '"]' + NL +
           '    tgx = hashlib.sha1((VMS_CUDA_SRC + " ".join(flx)).encode()).hexdigest()[:10]' + NL +
           '    modx = load_inline(name=f"k3_vms_{tgx}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC, functions=["run", "attrs"],' + NL +
           '                       extra_cuda_cflags=flx)' + NL +
           '    _VMS["mods"] = {"base": mod, "fx": modx}' + NL + old)
    ab = rep(ab, old, new, "_vms_build")
    ab = rep(ab, NL + "VMS_ENABLE = True" + NL, NL + "VMS_ENABLE = True" + NL + 'VMS_VAR = "base"   # A/B arm: "base" | "fx"' + NL,
             "VMS_VAR")
    ab = rep(ab, '    vs["mod"].run(x, ids, tw,', '    vs["mods"][VMS_VAR].run(x, ids, tw,', "_vms_call")
    ab = rep(ab, '        st["vms"] = {"mod": mod, "wA": wA,', '        st["vms"] = {"mod": mod, "mods": _VMS["mods"], "wA": wA,',
             "st vms")
    old = ("                          f\"err={errf} -> {'ok' if ok else 'FAIL'}\", flush=True)" + NL +
           "                good &= ok" + NL + "            _k2a33_reset(st)" + NL)
    new = (old.split(NL)[0] + NL + "                good &= ok" + NL +
           '            _vv = globals()["VMS_VAR"]' + NL +
           '            for vv in ("fx", "base", "fx"):' + NL +
           '                globals()["VMS_VAR"] = vv' + NL +
           '                o2 = _vms_call(st, x, d["topk_ids"], d["topk_weights"])' + NL +
           '                torch.cuda.synchronize()' + NL +
           '                cg = st["vms"]["ctr"]' + NL +
           '                same = bool(torch.equal(o2, o)) and not int(cg[2]) and int(cg[1]) == int(cg[0]) * _VMS["G"]' + NL +
           '                if not same:' + NL +
           '                    print(f"[vsf] self-test bs={bs} {dist} {vv}: NOT bitwise equal to base (err {int(cg[2])}) -> FAIL", flush=True)' + NL +
           '                good &= same' + NL +
           '            globals()["VMS_VAR"] = _vv' + NL +
           "            _k2a33_reset(st)" + NL)
    ab = rep(ab, old, new, "self-test")
    ab = rep(ab, '        print(f"[vms] VMS {\'LIVE at T <= \'', '        print(f"[vsf] fx self-tests bitwise vs base: {\'ok\' if good else \'FAIL\'}", flush=True)' + NL +
             '        print(f"[vms] VMS {\'LIVE at T <= \'', "live print")
    ab = ab.replace(bn, f"# nonce {tag}-ab (A/B copy: {BASE.stem} VMS base + VMS_FX={FX}, switched by VMS_VAR; tools/gen_k3_vsf.py)", 1)
    for nm, x in (("ship", s), ("ab", ab)):
        check(x.isascii(), f"{nm} non-ASCII")
        check(len(x.encode()) < 1_000_000, f"{nm} {len(x.encode())} B >= 1 MB")
        check("importlib" not in x[a0:a0 + len(src1)], f"{nm} checker")
        pycheck(x)
    print(f"[gen_k3_vsf] OK fx={FX}; ship {len(s.encode())} B, ab {len(ab.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(ab, encoding="utf-8", newline=NL)
    print(f"[gen_k3_vsf] wrote {OUT.relative_to(ROOT)} + {LAB.relative_to(ROOT)} (sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

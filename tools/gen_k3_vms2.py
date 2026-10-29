"""gen_k3_vms2.py: VMS2 builds (tools/vms2_patches.py: VMS_FX / VMS_RO / VMS_NRB / VMS_MK2 compile-time levers) of a layer.
  lab:  kernels/lab/<out>.py = layer + every --vars build in one prepare (parallel threads), switched by the module global
        VMS_VAR (rev_ab3 "set"); VMS_MARKS (int64 tensor, G*32 for MK2 builds) is passed as the marks buffer when set.
        Prepare self-tests every variant (FP32 / K2A tolerance, as the base).
  ship: kernels/lab/<out>.py = layer + one build with --ship flags (no switches).
At the defaults the patched source preprocesses to the base's. Refuses to overwrite.

    python -I tools/gen_k3_vms2.py --base=k3_gc_14 --out=k3_vms2_lab1 --vars="fx:-DVMS_FX=7;ro:-DVMS_RO=1;mk:-DVMS_MK2=1"
    python -I tools/gen_k3_vms2.py --base=k3_gc_14 --out=k3_vms2_1 --ship="-DVMS_FX=7 -DVMS_RO=1"
"""
import hashlib
import pathlib
import py_compile
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vms2_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_14')}.py"
OUTN = ARGS["out"]
OUT = ROOT / f"kernels/lab/{OUTN}.py"
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_k3_vms2] FAIL {msg}")


def rep(t, old, new, what):
    check(t.count(old) == 1, f"{what}: {t.count(old)} matches")
    return t.replace(old, new)


def main():
    check("--dry" in sys.argv or not OUT.exists(), f"{OUT.name} exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    check(bn.startswith("# nonce "), "line 2 is not a nonce line")
    k0 = 'VMS_CUDA_SRC = r"""'
    a0 = b.index(k0) + len(k0)
    b0 = b.index('"""', a0)
    src0 = b[a0:b0]
    src1 = P.apply(src0)
    check(P.check0(src0, src1), "default view differs from the base source")
    t = b[:a0] + src1 + b[b0:]
    fl = '"-DVMS_NS=3", "-Xptxas=-warn-spills"'
    tag = OUTN.replace("_", "-")
    if "ship" in ARGS:
        defs = ARGS["ship"].split()
        check(all(d.startswith("-DVMS_") and "MK2" not in d for d in defs), "ship flags")
        x = rep(t, fl, '"-DVMS_NS=3", ' + "".join(f'"{d}", ' for d in defs) + '"-Xptxas=-warn-spills"', "ship flags")
        x = x.replace(bn, f"# nonce {tag} ({BASE.stem} + VMS {' '.join(defs)}; tools/gen_k3_vms2.py)", 1)
    else:
        var = [v.split(":", 1) for v in ARGS["vars"].split(";")]
        check(all(n and n != "base" for n, _ in var), "variant names")
        vd = "{" + ", ".join(f'"{n}": {d.split()!r}' for n, d in var) + "}"
        x = rep(t, '_VMS = {"mod": None, "G": None}', '_VMS = {"mod": None, "G": None, "mods": {}}' + NL +
                f"VMS2_VARS = {vd}   # A/B arms (VMS_VAR); base = the layer's build", "_VMS")
        old = '    _VMS["G"] = G' + NL + '    _VMS["mod"] = mod' + NL
        new = ('    mods, ths = {"base": mod}, []' + NL +
               '    def _one(nm, dd):' + NL +
               '        rd = {d.split("=")[0] for d in dd}' + NL +
               '        flx = [f for f in flags[:-1] if f.split("=")[0] not in rd] + list(dd) + flags[-1:]' + NL +
               '        tgx = hashlib.sha1((VMS_CUDA_SRC + " ".join(flx)).encode()).hexdigest()[:10]' + NL +
               '        mods[nm] = load_inline(name=f"k3_vms_{tgx}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC,' + NL +
               '                               functions=["run", "attrs"], extra_cuda_cflags=flx)' + NL +
               '    for nm, dd in VMS2_VARS.items():' + NL +
               '        ths.append(threading.Thread(target=_one, args=(nm, dd), daemon=True))' + NL +
               '        ths[-1].start()' + NL +
               '    for th in ths:' + NL +
               '        th.join()' + NL +
               '    _VMS["mods"] = mods' + NL +
               '    print(f"[vms2] builds {[(k, list(v.attrs())) for k, v in mods.items()]}", flush=True)' + NL + old)
        x = rep(x, old, new, "_vms_build")
        x = rep(x, NL + "VMS_ENABLE = True" + NL, NL + "VMS_ENABLE = True" + NL + 'VMS_VAR = "base"   # A/B arm (VMS2_VARS)' + NL +
                "VMS_MARKS = None   # lab: int64 marks tensor (G * 32 for MK2 builds)" + NL, "VMS_VAR")
        x = rep(x, '    vs["mod"].run(x, ids, tw,', '    _VMS["mods"].get(VMS_VAR, vs["mod"]).run(x, ids, tw,', "_vms_call")
        x = rep(x, '                  out, vs["nomarks"], m,', '                  out, vs["nomarks"] if VMS_MARKS is None else VMS_MARKS, m,',
                "marks arg")
        old = ("            for k in range(2):" + NL + "                o = _vms_call(st, x, d[\"topk_ids\"], d[\"topk_weights\"])" + NL)
        new = ("            for k in range(2 * len(_VMS[\"mods\"])):" + NL +
               "                globals()[\"VMS_VAR\"] = list(_VMS[\"mods\"])[k // 2]" + NL +
               "                o = _vms_call(st, x, d[\"topk_ids\"], d[\"topk_weights\"])" + NL)
        x = rep(x, old, new, "self-test loop")
        x = rep(x, "                if not ok or k == 0:" + NL + "                    print(f\"[vms] self-test bs={bs} {dist} call {k}:",
                "                if not ok or k % 2 == 0:" + NL + "                    print(f\"[vms] self-test bs={bs} {dist} {VMS_VAR} call {k}:",
                "self-test print")
        x = rep(x, "            _k2a33_reset(st)" + NL + "        st[\"vms_ok\"] = good" + NL,
                "            _k2a33_reset(st)" + NL + "        globals()[\"VMS_VAR\"] = \"base\"" + NL + "        st[\"vms_ok\"] = good" + NL,
                "var reset")
        x = x.replace(bn, f"# nonce {tag} (lab: {BASE.stem} VMS base + {ARGS['vars']}; VMS_VAR / VMS_MARKS; tools/gen_k3_vms2.py)", 1)
    check(x.isascii(), "non-ASCII")
    check(len(x.encode()) < 1_000_000, f"{len(x.encode())} B >= 1 MB")
    check("importlib" not in x, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(x, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_vms2] OK {OUTN}: {len(x.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    OUT.write_text(x, encoding="utf-8", newline=NL)
    print(f"[gen_k3_vms2] wrote {OUT.relative_to(ROOT)} (sha1 {hashlib.sha1(x.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

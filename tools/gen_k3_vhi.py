"""gen_k3_vhi.py: in-layer A/B file for VMS_HI (moe redesign, R-vhi-1) = kernels/lab/k3_b29_raw.py + tools/vpre_patches.py +
tools/vhi_patches.py on VMS_CUDA_SRC (both at their defaults = the base's SASS) + a second VMS module built with -DVMS_HI=1
(_VMS["mod_hi"], same scratch / generation protocol) used by _vms_call when the module global VMS_HI_ON is True (layer_ab2
variants flip it during capture), then tools/slim_patches.py LAST (parent: b29_raw -> patchers -> slim reproduces b29). The VMS
self-test runs on both modules (VMS_HI_ON False then True); HI is dropped if its self-test fails.
    python -I tools/gen_k3_vhi.py [--base=kernels/lab/k3_b29_raw.py] [--out=kernels/lab/k3_vhi_ab.py] [--hi=1]
  --hi=1 (default): A/B file with both modules; --hi=ship: a single-module file built with -DVMS_HI=1 (the candidate itself).
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vhi_patches as PH  # noqa: E402
import vpre_patches as PR  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / ARGS.get("base", "kernels/lab/k3_b29_raw.py")
OUT = ROOT / ARGS.get("out", "kernels/lab/k3_vhi_ab.py")
MODE = ARGS.get("hi", "1")
NL = chr(10)


def _rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)


def main():
    s = BASE.read_text(encoding="utf-8")
    a = s.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    b = s.index('"""', a)
    s = s[:a] + PH.apply(PR.apply(s[a:b])) + s[b:]
    fl = '    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]\n'
    if MODE == "ship":
        s = _rep(s, fl, fl.replace('"-DVMS_RO=1", ', '"-DVMS_RO=1", "-DVMS_HI=1", '))
    else:
        s = _rep(s, '_VMS = {"mod": None, "G": None}\n',
                 '_VMS = {"mod": None, "G": None, "mod_hi": None}\nVMS_HI_ON = False   # R-vhi-1 A/B: layer_ab2 variants flip it during capture\n')
        s = _rep(s, '    _VMS["G"] = G\n    _VMS["mod"] = mod\n',
                 '    _VMS["G"] = G\n    _VMS["mod"] = mod\n'
                 '    fl_hi = flags[:4] + ["-DVMS_HI=1"] + flags[4:]   # R-vhi-1: helper refills the ring, no per-unit CTA sync\n'
                 '    tag_hi = hashlib.sha1((VMS_CUDA_SRC + " ".join(fl_hi)).encode()).hexdigest()[:10]\n'
                 '    _VMS["mod_hi"] = load_inline(name=f"k3_vmshi_{tag_hi}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC,\n'
                 '                                 functions=["run", "attrs"], extra_cuda_cflags=fl_hi)\n')
        s = _rep(s, '        st["vms"] = {"mod": mod, ', '        st["vms"] = {"mod": mod, "mod_hi": _VMS.get("mod_hi"), ')
        s = _rep(s, '    vs["mod"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],\n',
                 '    md = vs["mod_hi"] if (VMS_HI_ON and vs.get("mod_hi") is not None) else vs["mod"]   # R-vhi-1\n'
                 '    md.run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],\n')
    tmp = OUT.with_name(OUT.stem + "_unslim.py")
    tmp.write_text(s, encoding="utf-8", newline=NL)
    r = subprocess.run([sys.executable, "-I", str(ROOT / "tools/slim_patches.py"), f"--base={tmp}", f"--out={OUT}"],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode:
        sys.exit(f"slim_patches failed rc {r.returncode}")
    t = OUT.read_text(encoding="utf-8")
    assert t.isascii() and "importlib" not in t and "sys.modules" not in t and "__trap" not in t
    assert not [x for x in t.split(NL) if len(x) >= 1000]
    print(f"[gen_k3_vhi] wrote {OUT.relative_to(ROOT)} ({len(t.encode())} B, mode {MODE}); unslimmed {tmp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""gen_k3_vt16.py: in-layer A/B file for VMS_T16 (moe redesign, R-vt16-1) = <base> + tools/vt16_patches.py on VMS_CUDA_SRC (at its
defaults = the base's SASS) + a second VMS module built with -DVMS_T16=1 [-DVMS_WT2=w] (_VMS["mod_hi"], same scratch / generation
protocol) used by _vms_call for calls with > 8 tokens when the module global VMS_HI_ON is True (layer_ab2 variants flip it), then
tools/slim_patches.py LAST. --hi=ship: one VMS module for T <= 8 and the T16 one for T 9-16 (the candidate itself).
    python -I tools/gen_k3_vt16.py [--base=kernels/lab/k3_b29_raw.py] [--out=kernels/lab/k3_vt16_ab.py] [--hi=1|ship] [--wt2=26]
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vt16_patches as PT  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
BASE = ROOT / ARGS.get("base", "kernels/lab/k3_b29_raw.py")
OUT = ROOT / ARGS.get("out", "kernels/lab/k3_vt16_ab.py")
WT2 = ARGS.get("wt2", "26")
MODE = ARGS.get("hi", "1")
PAD = ARGS.get("pad", "")   # R-vt16-3: e.g. 81920 -> a third VMS module = the base + VMS_LPAD bytes of unused smem (VMS_HI_ON 2)
M2 = [f for f in ARGS.get("m2", "").split(",") if f]   # R-vt16-4: the third module's defines instead of the pad (e.g. -DVMS_T16=1,-DVMS_T16R=1)
HIX = [f for f in ARGS.get("hix", "").split(",") if f]   # extra defines for the T16 (VMS_HI_ON 1) module
NL = chr(10)


def _rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)


def main():
    s = BASE.read_text(encoding="utf-8")
    a = s.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    b = s.index('"""', a)
    s = s[:a] + PT.apply(s[a:b]) + s[b:]
    fl = '    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]\n'
    if MODE == "ship":
        pass   # ship = the A/B file with VMS_HI_ON = True (T16 module for > 8 tokens)
    if True:
        s = _rep(s, '_VMS = {"mod": None, "G": None}\n',
                 '_VMS = {"mod": None, "G": None, "mod_hi": None}\nVMS_HI_ON = {ON}   # R-vt16-1: True = the T16 VMS module for calls with > 8 tokens\n')
        s = _rep(s, '    _VMS["G"] = G\n    _VMS["mod"] = mod\n',
                 '    _VMS["G"] = G\n    _VMS["mod"] = mod\n'
                 '    fl_hi = flags[:4] + ["-DVMS_T16=1", "-DVMS_WT2=@WT2@"] + flags[4:]   # R-vt16-1: 16-token jobs, weighted split\n'
                 '    tag_hi = hashlib.sha1((VMS_CUDA_SRC + " ".join(fl_hi)).encode()).hexdigest()[:10]\n'
                 '    _VMS["mod_hi"] = load_inline(name=f"k3_vmshi_{tag_hi}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC,\n'
                 '                                 functions=["run", "attrs"], extra_cuda_cflags=fl_hi)\n')
        s = _rep(s, '        st["vms"] = {"mod": mod, ', '        st["vms"] = {"mod": mod, "mod_hi": _VMS.get("mod_hi"), ')
        s = _rep(s, '    vs["mod"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],\n',
                 '    md = vs["mod_hi"] if (VMS_HI_ON and m > 8 and vs.get("mod_hi") is not None) else vs["mod"]   # R-vt16-1\n'
                 '    md.run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],\n')
    if HIX:   # extra T16-module defines; a define already in the ship flags (e.g. -DVMS_NS) is replaced, not duplicated
        s = _rep(s, '"-DVMS_WT2=@WT2@"]', '"-DVMS_WT2=@WT2@"' + "".join(f', "{f}"' for f in HIX) + "]")
        keys = sorted({f.split("=")[0] for f in HIX})
        s = _rep(s, "    fl_hi = flags[:4] + [", f"    fl_hi = [f for f in flags[:4] if f.split('=')[0] not in {keys!r}] + [")
    if PAD or M2:   # R-vt16-3: VMS_HI_ON 2 = a third module for > 8 tokens (pad = smem-size control, or --m2 defines)
        s = _rep(s, '    _VMS["mod_hi"] = load_inline(',
                 '    fl_pd = [f for f in flags[:4] if f.split("=")[0] not in @M2K@] + [@M2L@] + flags[4:]' + NL
                 + '    tag_pd = hashlib.sha1((VMS_CUDA_SRC + " ".join(fl_pd)).encode()).hexdigest()[:10]' + NL
                 + '    _VMS["mod_pd"] = load_inline(name=f"k3_vmspd_{tag_pd}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC,' + NL
                 + '                                 functions=["run", "attrs"], extra_cuda_cflags=fl_pd)' + NL
                 + '    _VMS["mod_hi"] = load_inline(')
        s = _rep(s, '        st["vms"] = {"mod": mod, "mod_hi": _VMS.get("mod_hi"), ', '        st["vms"] = {"mod": mod, "mod_hi": _VMS.get("mod_hi"), "mod_pd": _VMS.get("mod_pd"), ')
        s = _rep(s, '    md = vs["mod_hi"] if (VMS_HI_ON and m > 8 and vs.get("mod_hi") is not None) else vs["mod"]   # R-vt16-1' + NL,
                 '    md = vs["mod_hi"] if (VMS_HI_ON == 1 and m > 8 and vs.get("mod_hi") is not None) else vs["mod"]   # R-vt16-1' + NL
                 + '    md = vs["mod_pd"] if (VMS_HI_ON == 2 and m > 8 and vs.get("mod_pd") is not None) else md   # R-vt16-3' + NL)
        s = s.replace("@M2L@", ", ".join(f'"{f}"' for f in (M2 or [f"-DVMS_LPAD={PAD}"])))
        s = s.replace("@M2K@", repr(sorted({f.split("=")[0] for f in (M2 or [f"-DVMS_LPAD={PAD}"])})))
    s = s.replace("{ON}", "True" if MODE == "ship" else "False").replace("@WT2@", WT2)
    tmp = OUT.with_name(OUT.stem + "_unslim.py")
    tmp.write_text(s, encoding="utf-8", newline=NL)
    r = subprocess.run([sys.executable, "-I", str(ROOT / "tools/slim_patches.py"), f"--base={tmp}", f"--out={OUT}"],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode:
        sys.exit(f"slim_patches failed rc {r.returncode}")
    t = OUT.read_text(encoding="utf-8")
    b0 = BASE.read_text(encoding="utf-8")   # the hh chain carries a "no __trap() in a bounded wait" comment: no NEW ones
    assert t.isascii() and "importlib" not in t and "sys.modules" not in t and t.count("__trap") <= b0.count("__trap")
    assert not [x for x in t.split(NL) if len(x) >= 1000]
    print(f"[gen_k3_vt16] wrote {OUT.relative_to(ROOT)} ({len(t.encode())} B, mode {MODE}); unslimmed {tmp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

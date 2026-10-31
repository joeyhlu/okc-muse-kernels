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


# ship mode: the T16 module (calls with 9-16 tokens) is self-tested on its own before VMS's cases; a failure drops only the T16
# module (the ship VMS module then serves every T <= 16), instead of turning VMS off through the shared self-test below it
T16_SELF_TEST = r'''        if VMS_HI_ON and st["vms"].get("mod_hi") is not None:
            for bs, seed, dist in ((12, 61, "skewed"), (16, 62, "normal"), (16, 63, "hot8"), (16, 64, "skewed"), (9, 65, "skewed")):
                d = _k2a36_synth_inputs(bs, seed, dev, "hot8") if dist == "hot8" else _k2a_self_test_inputs(bs, seed, dev, dist)
                if dist == "hot8":
                    d["topk_ids"] = torch.where(d["topk_ids"] < K2A_E_LOC, d["topk_ids"], torch.full_like(d["topk_ids"], -1))
                ref = _k2a_ref_fp32(d, w)
                o = _vms_call(st, d["hidden_states"].contiguous(), d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(o, ref)
                errf = int(st["vms"]["ctr"][2])
                if not (tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o.float()).all())):
                    print(f"[vms] T16 module self-test bs={bs} {dist}: rel {tot:.3e} row {row:.3e} err={errf} -> T16 OFF", flush=True)
                    st["vms"]["mod_hi"] = None
                    break
            else:
                print("[vms] T16 module (9-16 tokens) self-test ok", flush=True)
'''


def _rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)


def main():
    s = BASE.read_text(encoding="utf-8")
    a = s.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    b = s.index('"""', a)
    s = s[:a] + PT.apply(s[a:b]) + s[b:]
    fl = '    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]\n'
    if MODE == "ship":   # ship = VMS_HI_ON True + a T16-only prepare self-test that drops the T16 module (not VMS) on a failure
        anchor = "        good = True" + NL + "        for bs, seed, dist in VMS_SELF_TEST_CASES:" + NL
        s = _rep(s, anchor, T16_SELF_TEST + anchor)
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
    if MODE == "ship":   # the T16 module compiles in a thread next to the ship module (prepare: ~max, not the sum, of two nvcc runs)
        i0 = s.index('    mod = load_inline(name=f"k3_vms_{tag}"')
        i1 = s.index('    _VMS["mod_hi"] = load_inline(', i0)
        i2 = s.index("extra_cuda_cflags=fl_hi)" + NL, i1) + len("extra_cuda_cflags=fl_hi)" + NL)
        blk = s[i0:i2]
        base_ld = blk[:blk.index('    _VMS["G"] = G' + NL)]
        hi_def = blk[blk.index("    fl_hi = "):blk.index('    _VMS["mod_hi"] = load_inline(')]
        hi_ld = blk[blk.index('    _VMS["mod_hi"] = load_inline('):]
        new = (hi_def + "    import threading" + NL + NL
               + "    def _hi_build():   # R-vt16: a failed T16 build leaves the ship module at every T" + NL
               + "        try:" + NL
               + "".join("        " + ln + NL for ln in hi_ld.rstrip(NL).split(NL))
               + "        except Exception as ex:  # noqa: BLE001" + NL
               + '            _VMS["mod_hi"] = None' + NL
               + '            print(f"[vms] T16 module build failed {ex!r}: ship VMS module at every T", flush=True)' + NL
               + "    th_hi = threading.Thread(target=_hi_build, daemon=True)" + NL
               + "    th_hi.start()" + NL
               + base_ld + "    th_hi.join()" + NL
               + '    _VMS["G"] = G' + NL + '    _VMS["mod"] = mod' + NL)
        s = s[:i0] + new + s[i2:]
    if ARGS.get("nonce"):   # line 2 of the output
        L = s.split(NL); L[1] = "# nonce " + ARGS["nonce"]; s = NL.join(L)
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

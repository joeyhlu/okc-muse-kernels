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
LO = [f for f in ARGS.get("lo", "").split(",") if f]   # R-vil: defines of a module for M == 1 calls (VMS_LO_ON), e.g. -DVMS_IL=1
LOMAX = int(ARGS.get("lomax", "1"))   # R-vcb: the lo module serves calls with M <= LOMAX tokens
NL = chr(10)
import vil_patches as PI  # noqa: E402

# ship mode with --lo: the M == 1 module is self-tested on its own (a failure drops only it, like the T16 module)
LO_SELF_TEST = r'''        if VMS_LO_ON and st["vms"].get("mod_lo") is not None:
            for bs, seed, dist in ((1, 66, "normal"), (1, 67, "skewed"), (1, 68, "hot8"), (1, 69, "skewed")):
                d = _k2a36_synth_inputs(bs, seed, dev, "hot8") if dist == "hot8" else _k2a_self_test_inputs(bs, seed, dev, dist)
                if dist == "hot8":
                    d["topk_ids"] = torch.where(d["topk_ids"] < K2A_E_LOC, d["topk_ids"], torch.full_like(d["topk_ids"], -1))
                ref = _k2a_ref_fp32(d, w)
                o = _vms_call(st, d["hidden_states"].contiguous(), d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(o, ref)
                errf = int(st["vms"]["ctr"][2])
                if not (tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o.float()).all())):
                    print(f"[vms] M == 1 module self-test {dist}: rel {tot:.3e} row {row:.3e} err={errf} -> its module OFF", flush=True)
                    st["vms"]["mod_lo"] = None
                    break
            else:
                print("[vms] M == 1 module self-test ok", flush=True)
'''


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
    src = PT.apply(s[a:b])
    if LO or ARGS.get("vil") == "1":
        src = PI.apply(src)
    if ARGS.get("vcb") == "1" or any("VMS_CB" in f for f in LO + HIX):   # R-vcb: batched combine loads (flag-gated)
        import vcb_patches as PC
        src = PC.apply(src)
    if ARGS.get("ver") == "1" or any("VMS_ER" in f for f in LO + HIX):   # R-ver: early slot release (flag-gated)
        import ver_patches as PV
        src = PV.apply(src)
    s = s[:a] + src + s[b:]
    fl ='    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]\n'
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
    if LO:   # R-vil: a module for M == 1 calls (VMS_LO_ON), compiled next to the others
        keys = sorted({f.split("=")[0] for f in LO})
        s = _rep(s, '    _VMS["mod_hi"] = load_inline(',
                 f'    fl_lo = [f for f in flags[:4] if f.split("=")[0] not in {keys!r}] + [' + ", ".join(f'"{f}"' for f in LO) + '] + flags[4:]' + NL
                 + '    tag_lo = hashlib.sha1((VMS_CUDA_SRC + " ".join(fl_lo)).encode()).hexdigest()[:10]' + NL
                 + '    _VMS["mod_lo"] = load_inline(name=f"k3_vmslo_{tag_lo}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC,' + NL
                 + '                                 functions=["run", "attrs"], extra_cuda_cflags=fl_lo)' + NL
                 + '    _VMS["mod_hi"] = load_inline(')
        s = _rep(s, '        st["vms"] = {"mod": mod, "mod_hi": _VMS.get("mod_hi"), ', '        st["vms"] = {"mod": mod, "mod_hi": _VMS.get("mod_hi"), "mod_lo": _VMS.get("mod_lo"), ')
        s = _rep(s, '    md.run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],' + NL,
                 '    md = vs["mod_lo"] if (VMS_LO_ON and m <= @LOMAX@ and vs.get("mod_lo") is not None) else md   # R-vil / R-vcb' + NL
                 + '    md.run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],' + NL)
        s = _rep(s, "VMS_HI_ON = {ON}", "VMS_LO_ON = {ON}   # R-vil: True = the M == 1 VMS module for calls with 1 token" + NL + "VMS_HI_ON = {ON}")
        if MODE == "ship":
            anchor = "        good = True" + NL + "        for bs, seed, dist in VMS_SELF_TEST_CASES:" + NL
            s = _rep(s, anchor, LO_SELF_TEST + anchor)
    s = s.replace("@LOMAX@", str(LOMAX))
    s = s.replace("{ON}", "True" if MODE == "ship" else "False").replace("@WT2@", WT2)
    if MODE == "ship":   # the T16 module compiles in a thread next to the ship module (prepare: ~max, not the sum, of two nvcc runs)
        i0 = s.index('    mod = load_inline(name=f"k3_vms_{tag}"')
        i1 = s.index('    _VMS["mod_hi"] = load_inline(', i0)
        i2 = s.index("extra_cuda_cflags=fl_hi)" + NL, i1) + len("extra_cuda_cflags=fl_hi)" + NL)
        blk = s[i0:i2]
        base_ld = blk[:blk.index('    _VMS["G"] = G' + NL)]
        hi_def = blk[blk.index("    fl_hi = "):blk.index('    _VMS["mod_hi"] = load_inline(')]
        hi_ld = blk[blk.index('    _VMS["mod_hi"] = load_inline('):]
        side = [("hi", "T16", hi_def, hi_ld)]
        if LO:   # the M == 1 module: its own thread too
            lo_def = hi_def[:hi_def.index('    _VMS["mod_lo"] = load_inline(')]
            lo_ld = hi_def[hi_def.index('    _VMS["mod_lo"] = load_inline('):]
            hi_def = hi_def[:hi_def.index("    fl_lo = ")] + hi_def[hi_def.index('    _VMS["mod_lo"] = load_inline('):]
            hi_def = hi_def[:hi_def.index('    _VMS["mod_lo"] = load_inline(')]
            lo_def = lo_def[lo_def.index("    fl_lo = "):]
            side = [("hi", "T16", hi_def, hi_ld), ("lo", "M == 1", lo_def, lo_ld)]
        new = "    import threading" + NL + "    ths = []" + NL
        for key, label, d_, ld in side:
            new += (d_ + NL + f"    def _{key}_build():   # a failed {label} build leaves the ship module for those calls" + NL
                    + "        try:" + NL
                    + "".join("        " + ln + NL for ln in ld.rstrip(NL).split(NL))
                    + "        except Exception as ex:  # noqa: BLE001" + NL
                    + f'            _VMS["mod_{key}"] = None' + NL
                    + f'            print(f"[vms] {label} module build failed {{ex!r}}: ship VMS module instead", flush=True)' + NL
                    + f"    ths.append(threading.Thread(target=_{key}_build, daemon=True))" + NL
                    + "    ths[-1].start()" + NL)
        new += (base_ld + "    for th in ths:" + NL + "        th.join()" + NL
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

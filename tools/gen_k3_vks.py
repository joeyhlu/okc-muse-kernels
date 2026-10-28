"""gen_k3_vks.py: VMS-KS J1 (tools/vks_patches.py) + the R-ver-1 barrier early return on a k3 layer file (default k3_gc_5).

Kernel: VMS_CUDA_SRC + compile-time VMS_KS (J1 jobs for 9..16-token experts) and VMS_ER (R-ver-1); both = 0 preprocess back to
the base source (checked here). Layer: builds keyed by cfg = (ks, er); VMS_KS / VMS_ER pick the build at call time; VMS_AB_CFGS
adds builds for A/B copies (PT1 jobs "vms<ks><er>"). Scratch: ybuf gets VMS_FX_ROWS rows behind the 128 pair rows for the J1
fixup partials, ctr gets 16 x 384 fixup counters behind the doneA counters. Self-tests run every build (2 calls each) with
barrier-counter coherence, all fixup counters back at 0, and every build vs the base cfg rel < 1e-3.

    python -I tools/gen_k3_vks.py --ab [--base=k3_gc_5] [--out=kernels/lab/k3_vks_1_ab.py]    # cfg (1,0) + (0,0), (1,1)
    python -I tools/gen_k3_vks.py --n=<n> [--ks=1] [--er=0] [--base=k3_gc_5] [--check]        # ship file k3_gc_<n>
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vks_patches as KP  # noqa: E402

Q3 = '"' * 3

ER_OLD = "  const int gen = S.gen;\n  if (tid == 0) {\n    __threadfence();\n    atomicAdd(p.ctr + 1, 1);\n"
ER_NEW = ("  const int gen = S.gen;\n#if VMS_ER\n"
          "  if (c >= max(1, p.m * 4)) {   // no final-pass items: arrive and leave (c == 0 always waits for all G arrivals)\n"
          "    if (tid == 0) { __threadfence(); atomicAdd(p.ctr + 1, 1); }\n    return;\n  }\n#endif\n"
          "  if (tid == 0) {\n    __threadfence();\n    atomicAdd(p.ctr + 1, 1);\n")


def sub1(t, old, new):
    assert t.count(old) == 1, (old[:90], t.count(old))
    return t.replace(old, new)


def kernel(src):
    t = sub1(src, "#ifndef VMS_G\n#define VMS_G 132\n#endif\n",
             "#ifndef VMS_G\n#define VMS_G 132\n#endif\n#ifndef VMS_ER\n#define VMS_ER 0\n#endif\n")
    t = sub1(t, ER_OLD, ER_NEW)
    t = KP.apply_ks(t)
    assert KP.pp_flag(KP.pp_flag(t, "VMS_KS", 0), "VMS_ER", 0) == src, "VMS_KS = VMS_ER = 0 must preprocess to the base"
    return t


def apply_vks(t, ks, er, ab_cfgs):
    a = t.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    b = t.index(Q3, a)
    t = t[:a] + kernel(t[a:b]) + t[b:]
    t = sub1(t, "VMS_ENABLE = True\nVMS_MAXT = 16\n",
             "VMS_ENABLE = True\nVMS_MAXT = 16\n"
             f"VMS_KS = {bool(ks)}   # VMS-KS J1: a 9..16-token expert is one 16-token job over two K-halves (R-vks-2)\n"
             f"VMS_ER = {bool(er)}   # grid barrier: CTAs without final-pass items arrive and return (R-ver-1)\n"
             f"VMS_AB_CFGS = {tuple(ab_cfgs)!r}   # A/B copies: extra (ks, er) builds, picked at call time by (VMS_KS, VMS_ER)\n"
             "VMS_FX_ROWS = 1536   # ybuf rows behind the 128 pair rows: J1 fixup partials (16 jobs x 384 pairs x 2 halves x 2 KB)\n"
             "VMS_FX_CTR = 16 * 384   # fixup arrival counters behind the doneA counters\n"
             "VMS_KS_MINT = 9   # T <= 8 cannot hold a J1 job (an expert gets <= T tokens): those calls use the ks 0 build\n")
    t = sub1(t, '_VMS = {"mod": None, "G": None}\n', '_VMS = {"mod": None, "G": None, "mods": {}}\n')
    t = sub1(t, "def _vms_build():\n",
             "def _vms_cfg(m=None):\n    return (int(bool(VMS_KS) and (m is None or m >= VMS_KS_MINT)), int(bool(VMS_ER)))\n\n\n"
             "def _vms_extra_cfgs():\n    d = _vms_cfg()\n    cs = [(int(c[0]), int(c[1])) for c in VMS_AB_CFGS] + ([(0, d[1])] if d[0] else [])\n"
             "    return [c for i, c in enumerate(cs) if c != d and c not in cs[:i]]\n\n\n"
             "def _vms_build(cfg=None):\n    cfg = _vms_cfg() if cfg is None else (int(cfg[0]), int(cfg[1]))\n")
    t = sub1(t, '"-DVMS_NS=3", "-Xptxas=-warn-spills"',
             '"-DVMS_NS=3", f"-DVMS_KS={cfg[0]}", f"-DVMS_ER={cfg[1]}", "-Xptxas=-warn-spills"')
    t = sub1(t, '    _VMS["G"] = G\n    _VMS["mod"] = mod\n    _K3_BUILD_S["vms"] = time.time() - t1\n',
             '    _VMS["G"] = G\n    _VMS["mods"][cfg] = mod\n    if cfg == _vms_cfg():\n        _VMS["mod"] = mod\n'
             '    _K3_BUILD_S["vms" if cfg == _vms_cfg() else f"vms{cfg[0]}{cfg[1]}"] = time.time() - t1\n')
    t = sub1(t, '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n',
             '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n'
             '        for cf in _vms_extra_cfgs():\n'
             '            jobs.append((f"vms{cf[0]}{cf[1]}", lambda cf=cf: _vms_build(cf)))\n')
    t = sub1(t, "def _vms_call(st, x, local_ids, topk_weights):\n    vs = st[\"vms\"]\n",
             "def _vms_call(st, x, local_ids, topk_weights, mod=None):\n    vs = st[\"vms\"]\n"
             "    if mod is None:\n        mod = vs[\"mods\"].get(_vms_cfg(x.shape[0]), vs[\"mod\"])\n")
    t = sub1(t, '    vs["mod"].run(x, ids, tw, vs["wA"]', '    mod.run(x, ids, tw, vs["wA"]')
    t = sub1(t, '        mod = _VMS["mod"] if _VMS["mod"] is not None else _vms_build()\n',
             '        mod = _VMS["mod"] if _VMS["mod"] is not None else _vms_build()\n        mods = {_vms_cfg(): mod}\n'
             '        for cf in _vms_extra_cfgs():\n'
             '            _pt1_wait(f"vms{cf[0]}{cf[1]}")\n'
             '            alt = _VMS["mods"].get(cf)\n'
             '            mods[cf] = alt if alt is not None else _vms_build(cf)\n')
    t = sub1(t, '        st["vms"] = {"mod": mod, "wA": wA,', '        st["vms"] = {"mod": mod, "mods": mods, "wA": wA,')
    t = sub1(t, '"ybuf": torch.zeros((128, K2A_D_HIDDEN), dtype=torch.float32, device=dev),',
             '"ybuf": torch.zeros((128 + VMS_FX_ROWS, K2A_D_HIDDEN), dtype=torch.float32, device=dev),')
    t = sub1(t, '"ctr": torch.zeros((72,), dtype=torch.int32, device=dev),',
             '"ctr": torch.zeros((72 + VMS_FX_CTR,), dtype=torch.int32, device=dev),')
    old = '''            for k in range(2):
                o = _vms_call(st, x, d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                e2, _ = _k2a_rel_err(o, ok_k)
                tot, row = _k2a_rel_err(o, ref)
                errf = int(st["vms"]["ctr"][2])
                ok = e2 < 8e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o.float()).all())
                if not ok or k == 0:
                    print(f"[vms] self-test bs={bs} {dist} call {k}: rel vs K2A {e2:.2e} vs FP32 {tot:.3e} worst row {row:.3e} "
                          f"err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
'''
    new = '''            outs = {}
            for cf, md in mods.items():
                for k in range(2):
                    o = _vms_call(st, x, d["topk_ids"], d["topk_weights"], md)
                    torch.cuda.synchronize()
                    e2, _ = _k2a_rel_err(o, ok_k)
                    tot, row = _k2a_rel_err(o, ref)
                    ctr = st["vms"]["ctr"]
                    errf = int(ctr[2])
                    coh = int(ctr[1]) == int(ctr[0]) * _VMS["G"] and int(ctr[72:].abs().sum()) == 0
                    ok = (e2 < 8e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and coh
                          and bool(torch.isfinite(o.float()).all()))
                    if not ok or k == 0:
                        print(f"[vms] self-test bs={bs} {dist} cfg={cf} call {k}: rel vs K2A {e2:.2e} vs FP32 {tot:.3e} "
                              f"worst row {row:.3e} err={errf} counters {'ok' if coh else 'INCOHERENT'} -> {'ok' if ok else 'FAIL'}",
                              flush=True)
                    good &= ok
                    outs[cf] = o.clone()
            o0 = outs.get((0, 0), outs[_vms_cfg()])
            for cf, o in outs.items():
                ex, _ = _k2a_rel_err(o, o0)
                if not ex < 1e-3:   # VMS's own run-to-run spread is ~1.5e-4 (FP32 ss atomics)
                    print(f"[vms] self-test bs={bs} {dist}: cfg {cf} vs base rel {ex:.2e} -> FAIL", flush=True)
                    good = False
'''
    t = sub1(t, old, new)
    t = sub1(t, '''        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "''',
             '''        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "
              f"(builds (ks, er) {sorted(mods)}, call-time cfg {_vms_cfg()}) "''')
    return t


def main(argv):
    opt = {a.split("=", 1)[0].lstrip("-"): (a.split("=", 1)[1] if "=" in a else True) for a in argv}
    base = opt.get("base", "k3_gc_5")
    src = (ROOT / "kernels/muse_decode_layer" / f"{base}.py").read_text(encoding="utf-8")
    first, rest = src.split("\n", 1)
    nonce, rest = rest.split("\n", 1)
    assert first.startswith("# okc: problem=muse_decode_layer") and nonce.startswith("# nonce ")
    if opt.get("ab"):
        out = ROOT / opt.get("out", "kernels/lab/k3_vks_1_ab.py")
        ks, er, cfgs = 1, 0, ((0, 0), (1, 1), (0, 1))
        tag = (f"# nonce k3-vks-1-ab ({base} + VMS-KS J1 + barrier early return, builds ks/er 10, 00, 11, 01, ks only at "
               f"T >= 9; tools/gen_k3_vks.py)")
    else:
        n = int(opt["n"])
        ks, er, cfgs = int(opt.get("ks", 1)), int(opt.get("er", 0)), ()
        out = ROOT / "kernels/muse_decode_layer" / f"k3_gc_{n}.py"
        tag = (f"# nonce k3-gc-{n} ({base} + VMS-KS J1 ks {ks} er {er}: experts with 9..16 tokens as one 16-token job over two "
               f"K-halves with a pair fixup; tools/gen_k3_vks.py)")
    text = first + "\n" + tag + "\n" + apply_vks(rest, ks, er, cfgs)
    compile(text, str(out), "exec")
    assert len(text.encode()) < 1_000_000, len(text.encode())
    if opt.get("check"):
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print(f"[vks] {out.name}: {'reproduced' if same else 'DIFFERS'}")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"[vks] wrote {out.relative_to(ROOT)} ({len(text.encode()):,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

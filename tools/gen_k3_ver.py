"""gen_k3_ver.py: R-ver-1 = VMS grid-barrier early return on a k3 layer file (default base k3_gc_4).

Kernel: after the B phase every CTA still arrives at the grid barrier (fence + atomicAdd ctr[1]); a CTA with no
final-pass items (c >= max(1, 4 m)) then returns instead of spinning. c == 0 always has an item, so the resetter still
waits for all G arrivals before it zeroes the doneA counters / the next ss buffer and bumps gen. Compile-time
VMS_ER (default 0 preprocesses back to the base source, i.e. the base codegen).
Layer: VMS_ER (build used at T <= 16) and VMS_ER_AB (A/B copies: also build the other variant in a PT1 job "vmser";
_vms_call picks by VMS_ER at call time). Self-tests run every built variant (2 calls each) and add a barrier-counter
coherence check (ctr[1] == ctr[0] * G after each call) and, with both builds, er vs base rel < 1e-3.

    python -I tools/gen_k3_ver.py --ab   [--base=k3_gc_4] [--out=kernels/lab/k3_ver_1_ab.py]   # A/B copy
    python -I tools/gen_k3_ver.py --n=5  [--base=k3_gc_4] [--check]                            # ship file k3_gc_<n>
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
Q3 = '"' * 3


def sub1(t, old, new, lo=0, hi=None):
    hi = len(t) if hi is None else hi
    seg = t[lo:hi]
    assert seg.count(old) == 1, (old[:80], seg.count(old))
    return t[:lo] + seg.replace(old, new) + t[hi:]


def apply_ver(t, ab):
    a = t.index('VMS_CUDA_SRC = r"""')
    b = t.index(Q3, a + 20)
    # kernel (inside VMS_CUDA_SRC only)
    t = sub1(t, "#ifndef VMS_G\n#define VMS_G 132\n#endif\n",
             "#ifndef VMS_G\n#define VMS_G 132\n#endif\n#ifndef VMS_ER\n#define VMS_ER 0\n#endif\n", a, b)
    b = t.index(Q3, a + 20)
    t = sub1(t, "  const int gen = S.gen;\n  if (tid == 0) {\n    __threadfence();\n    atomicAdd(p.ctr + 1, 1);\n",
             "  const int gen = S.gen;\n#if VMS_ER\n"
             "  if (c >= max(1, p.m * 4)) {   // no final-pass items: arrive and leave (c == 0 always waits for all G arrivals)\n"
             "    if (tid == 0) { __threadfence(); atomicAdd(p.ctr + 1, 1); }\n    return;\n  }\n#endif\n"
             "  if (tid == 0) {\n    __threadfence();\n    atomicAdd(p.ctr + 1, 1);\n", a, b)
    # globals
    t = sub1(t, "VMS_ENABLE = True\nVMS_MAXT = 16\n",
             "VMS_ENABLE = True\nVMS_MAXT = 16\n"
             "VMS_ER = True   # grid barrier: CTAs without final-pass items arrive and return (R-ver-1)\n"
             f"VMS_ER_AB = {bool(ab)}   # A/B copies: also build the other VMS_ER variant, picked at call time\n")
    t = sub1(t, '_VMS = {"mod": None, "G": None}\n', '_VMS = {"mod": None, "G": None, "mods": {}}\n')
    # build
    t = sub1(t, "def _vms_build():\n", "def _vms_build(er=None):\n    er = VMS_ER if er is None else bool(er)\n")
    t = sub1(t, '"-DVMS_NS=3", "-Xptxas=-warn-spills"', '"-DVMS_NS=3", f"-DVMS_ER={int(er)}", "-Xptxas=-warn-spills"')
    t = sub1(t, '    _VMS["G"] = G\n    _VMS["mod"] = mod\n    _K3_BUILD_S["vms"] = time.time() - t1\n',
             '    _VMS["G"] = G\n    _VMS["mods"][er] = mod\n    if er == VMS_ER:\n        _VMS["mod"] = mod\n'
             '    _K3_BUILD_S["vms" if er == VMS_ER else "vmser"] = time.time() - t1\n')
    t = sub1(t, '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n',
             '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n        if VMS_ER_AB:\n'
             '            jobs.append(("vmser", lambda: _vms_build(not VMS_ER)))\n')
    # call
    t = sub1(t, "def _vms_call(st, x, local_ids, topk_weights):\n    vs = st[\"vms\"]\n",
             "def _vms_call(st, x, local_ids, topk_weights, mod=None):\n    vs = st[\"vms\"]\n"
             "    if mod is None:\n        mod = vs[\"mods\"].get(bool(VMS_ER), vs[\"mod\"])\n")
    t = sub1(t, '    vs["mod"].run(x, ids, tw, vs["wA"]', '    mod.run(x, ids, tw, vs["wA"]')
    # prepare
    t = sub1(t, '        mod = _VMS["mod"] if _VMS["mod"] is not None else _vms_build()\n',
             '        mod = _VMS["mod"] if _VMS["mod"] is not None else _vms_build()\n        mods = {bool(VMS_ER): mod}\n'
             '        if VMS_ER_AB:\n            _pt1_wait("vmser")\n'
             '            alt = _VMS["mods"].get(not VMS_ER)\n'
             '            mods[not VMS_ER] = alt if alt is not None else _vms_build(not VMS_ER)\n')
    t = sub1(t, '        st["vms"] = {"mod": mod, "wA": wA,', '        st["vms"] = {"mod": mod, "mods": mods, "wA": wA,')
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
            for ver, md in mods.items():
                for k in range(2):
                    o = _vms_call(st, x, d["topk_ids"], d["topk_weights"], md)
                    torch.cuda.synchronize()
                    e2, _ = _k2a_rel_err(o, ok_k)
                    tot, row = _k2a_rel_err(o, ref)
                    errf = int(st["vms"]["ctr"][2])
                    coh = int(st["vms"]["ctr"][1]) == int(st["vms"]["ctr"][0]) * _VMS["G"]
                    ok = (e2 < 8e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and coh
                          and bool(torch.isfinite(o.float()).all()))
                    if not ok or k == 0:
                        print(f"[vms] self-test bs={bs} {dist} er={int(ver)} call {k}: rel vs K2A {e2:.2e} vs FP32 {tot:.3e} "
                              f"worst row {row:.3e} err={errf} bar {'ok' if coh else 'INCOHERENT'} -> {'ok' if ok else 'FAIL'}",
                              flush=True)
                    good &= ok
                    outs[ver] = o.clone()
            if len(outs) == 2:
                ex, _ = _k2a_rel_err(outs[True], outs[False])
                if not ex < 1e-3:   # run-to-run spread of VMS itself reaches ~1.5e-4 (FP32 ss atomics, R-ver-1 seed 777)
                    print(f"[vms] self-test bs={bs} {dist}: er vs base rel {ex:.2e} -> FAIL", flush=True)
                    good = False
'''
    t = sub1(t, old, new)
    t = sub1(t, '''        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "''',
             '''        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "
              f"(ER builds {sorted(int(v) for v in mods)}, call-time VMS_ER) "''')
    return t


def main(argv):
    opt = {a.split("=", 1)[0].lstrip("-"): (a.split("=", 1)[1] if "=" in a else True) for a in argv}
    base = opt.get("base", "k3_gc_4")
    src = (ROOT / "kernels/muse_decode_layer" / f"{base}.py").read_text(encoding="utf-8")
    first, rest = src.split("\n", 1)
    nonce, rest = rest.split("\n", 1)
    assert first.startswith("# okc: problem=muse_decode_layer") and nonce.startswith("# nonce ")
    ab = bool(opt.get("ab"))
    if ab:
        out = ROOT / opt.get("out", "kernels/lab/k3_ver_1_ab.py")
        tag = f"# nonce k3-ver-1-ab ({base} + R-ver-1 barrier early return, both builds; tools/gen_k3_ver.py)"
    else:
        n = int(opt["n"])
        out = ROOT / "kernels/muse_decode_layer" / f"k3_gc_{n}.py"
        tag = f"# nonce k3-gc-{n} ({base} + R-ver-1 VMS grid-barrier early return for CTAs without final-pass items; tools/gen_k3_ver.py)"
    text = first + "\n" + tag + "\n" + apply_ver(rest, ab)
    compile(text, str(out), "exec")
    assert len(text.encode()) < 1_000_000, len(text.encode())
    if opt.get("check"):
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print(f"[ver] {out.name}: {'reproduced' if same else 'DIFFERS'}")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"[ver] wrote {out.relative_to(ROOT)} ({len(text.encode()):,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

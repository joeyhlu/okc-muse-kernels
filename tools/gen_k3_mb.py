"""gen_k3_mb.py: k3_mb_<n> = a layer file (default k3_vfa_7) + MoE-board-loop riders measured on the judge GPU
(notes/study/moe_board_loop.md). Rider 1 (k3_mb_1): VML route()'s balanced-job-size countdown loop
`if (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }` (warp 0, up to 63 dependent iterations per lane while every
CTA waits; ~1-1.6 us per call at T >= 32, mb_4 / mb_5 bit 10 / 13 on layer-like routings r 0.984 / 0.994 / 0.993) ->
its closed form `if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));`, which this tool asserts equal to the
loop for every (base0, nj, tot) in range before writing (identical jobs, bitwise-identical outputs).
Output goes through gen_k3_m32.finish (strip, compile, ASCII, nonce, checker scan, < 1 MB). Refuses to overwrite.

    python -I tools/gen_k3_mb.py [--base=k3_vfa_7] [--n=1]
"""
import hashlib
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv

N = ARGS.get("n", "1")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vfa_7')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_mb_{N}.py"
TAG = f"k3-mb-{N}"
NL = chr(10)
LOOP = "if (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }"
CLOSED = ("if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));   "
          "// == the countdown loop for every (base, nj, tot) (tools/gen_k3_mb.py asserts it); mb bit 13")


def assert_closed_form(bmax=72, njmax=96, totmax=2600):
    """brute force: the loop's result vs the closed form over base0 in [-2, bmax], nj in [0, njmax], tot in [0, totmax]
    (C int semantics; the code has base0 = 64, nj = ceil(tot / 64) <= 40, tot <= 320 x 8)"""
    nj = np.arange(0, njmax + 1, dtype=np.int64)[:, None]
    tot = np.arange(0, totmax + 1, dtype=np.int64)[None, :]
    bad = 0
    for b0 in range(-2, bmax + 1):
        b = np.full((nj.shape[0], tot.shape[1]), b0, dtype=np.int64)
        on = np.broadcast_to(nj != 0, b.shape)
        for _ in range(max(0, b0) + 1):   # the loop runs at most b0 - 1 times
            dec = on & (b > 1) & ((b - 1) * nj >= tot)
            b = np.where(dec, b - 1, b)
        njs = np.where(nj == 0, 1, nj)
        cf = np.where((nj != 0) & (b0 > 1), np.maximum(1, np.minimum(b0, (tot + njs - 1) // njs)), b0)
        bad += int((cf != b).sum())
    GM.check(bad == 0, f"closed form differs from the loop at {bad} points")
    return (bmax + 3) * (njmax + 1) * (totmax + 1)


VMS2_BLOCK = r'''

# ======== VMS2: the T = VMS2_MINT..VMS_MAXT MoE (decode16) as 16-token jobs with the second x tile in shared memory (X2S) and
# cost-weighted CTA unit ranges (VMS_WT 6: a 2-tile unit weighs 1.5 one-tile units), so a hot expert of 9-16 tokens is
# streamed once instead of twice (tools/gen_k3_mb.py; MoE board loop mb_6..8, notes/study/moe_board_loop.md s6: layer-like
# T=16 r 0.950-0.953 vs the shipped VMS). Shares the shipped VMS's weights; own scratch / counters; self-tested against the
# shipped VMS and FP32 in prepare; any build / self-test / call failure falls back to the shipped VMS (loud). T <= 8: shipped.
VMS2_ENABLE = True
VMS2_MINT = 9
VMS2_FLAGS = ("-DVMS_NS=3", "-DVMS_MAXT=16", "-DVMS_FL=1152", "-DVMS_WT=6")
VMS2_SELF_TEST_CASES = ((9, 61, "normal"), (12, 62, "skewed"), (16, 63, "normal"), (16, 64, "hot8"), (16, 65, "skewed"),
                        (13, 66, "hot8"))
VMS2_CUDA_SRC = r"""@VMS2CUDA@"""
VMS2_CPP_SRC = r"""@VMS2CPP@"""
_VMS2 = {"mod": None, "G": None}


def _vms2_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}"] + list(VMS2_FLAGS) + ["-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS2_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vms2_{tag}", cpp_sources=VMS2_CPP_SRC, cuda_sources=VMS2_CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags)
    _VMS2["G"] = G
    _VMS2["mod"] = mod
    _K3_BUILD_S["vms2"] = time.time() - t1
    return mod


def _vms2_call(st, x, local_ids, topk_weights):
    vs, v2 = st["vms"], st["vms2"]
    ids = local_ids.to(torch.int32).contiguous()
    tw = topk_weights.to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    v2["mod"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], v2["inter"], v2["ybuf"], v2["ss"], v2["ctr"],
                  out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0, 0)
    return out


def _vms2_prepare(st, w, dev):
    st["vms2_ok"] = False
    if not (VMS2_ENABLE and st.get("vms_ok")):
        print(f"[vms2] OFF (VMS2_ENABLE {VMS2_ENABLE}, shipped VMS ok {st.get('vms_ok')}): shipped VMS at T <= 16", flush=True)
        return
    t1 = time.time()
    try:
        _pt1_wait("vms2")
        mod = _VMS2["mod"] if _VMS2["mod"] is not None else _vms2_build()
        if _VMS2["G"] != torch.cuda.get_device_properties(0).multi_processor_count:
            print("[vms2] built for a different SM count: OFF, shipped VMS at T <= 16", flush=True)
            return
        st["vms2"] = {"mod": mod, "inter": torch.zeros((128, K2A_D_EXPERT), dtype=torch.bfloat16, device=dev),
                      "ybuf": torch.zeros((128, K2A_D_HIDDEN), dtype=torch.float32, device=dev),
                      "ss": torch.zeros((256,), dtype=torch.float32, device=dev),
                      "ctr": torch.zeros((72,), dtype=torch.int32, device=dev)}
        good = True
        for bs, seed, dist in VMS2_SELF_TEST_CASES:
            if dist == "hot8":
                d = _k2a36_synth_inputs(bs, seed, dev, "hot8")
                ids = d["topk_ids"]
                d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
            else:
                d = _k2a_self_test_inputs(bs, seed, dev, dist)
            x = d["hidden_states"].contiguous()
            ref = _k2a_ref_fp32(d, w)
            o1 = _vms_call(st, x, d["topk_ids"], d["topk_weights"]).clone()   # shipped VMS (vms2_ok is still False)
            torch.cuda.synchronize()
            for k in range(2):
                o = _vms2_call(st, x, d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                e1, _ = _k2a_rel_err(o, o1)
                tot, row = _k2a_rel_err(o, ref)
                errf = int(st["vms2"]["ctr"][2])
                ok = e1 < 8e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o.float()).all())
                if not ok:
                    print(f"[vms2] self-test bs={bs} {dist} call {k}: rel vs VMS {e1:.2e} vs FP32 {tot:.3e} worst row "
                          f"{row:.3e} err={errf} -> FAIL", flush=True)
                good &= ok
        st["vms2_ok"] = good
        print(f"[vms2] VMS2 {'LIVE at ' + str(VMS2_MINT) + ' <= T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, shipped VMS at T <= 16'}"
              f" (regs/local/smem/G/NS {list(mod.attrs())}, {time.time() - t1:.1f}s)", flush=True)
    except Exception as ex:  # noqa: BLE001
        st["vms2_ok"] = False
        print(f"[vms2] prepare raised {ex!r}: OFF, shipped VMS at T <= 16", flush=True)
'''


def _vms2_sources():
    import mb_gen
    lab = mb_gen._lab_strings(mb_gen.VMS_LAB, ("CUDA_SRC", "CPP_SRC"))
    cuda = lab["CUDA_SRC"]
    for a, b2 in mb_gen.VMS_PATCHES + mb_gen.X2S_PATCHES:
        GM.check(cuda.count(a) == 1, f"VMS2 patch anchor: {a[:60]!r}")
        cuda = cuda.replace(a, b2)
    GM.check('"""' not in cuda and '"""' not in lab["CPP_SRC"], "triple quotes in the VMS2 source")
    return cuda, lab["CPP_SRC"]


def main_vms2():
    """k3_mb_<n> = <base> + VMS2 (T = 9..16) with the shipped VMS as the fallback"""
    GM.check(not OUT.exists(), f"{OUT.name} exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    cuda, cpp = _vms2_sources()
    block = VMS2_BLOCK.replace("@VMS2CUDA@", cuda).replace("@VMS2CPP@", cpp)
    nonce = (f"# nonce {TAG} ({BASE.stem} + VMS2 at T = 9..16: 16-token jobs with the second x tile in smem and cost-weighted "
             "CTA ranges, shipped VMS as fallback; MoE board loop mb_6..8)")
    t = b.replace(bn, nonce)
    for a, b2, what in (
        ("\nVMS_SELF_TEST_CASES = (", block + "\n\nVMS_SELF_TEST_CASES = (", "VMS2 block before VMS_SELF_TEST_CASES"),
        ('def _vms_call(st, x, local_ids, topk_weights):\n    vs = st["vms"]\n',
         'def _vms_call(st, x, local_ids, topk_weights):\n'
         '    if VMS2_ENABLE and st.get("vms2_ok") and VMS2_MINT <= x.shape[0] <= VMS_MAXT:\n'
         '        try:\n            return _vms2_call(st, x, local_ids, topk_weights)\n'
         '        except Exception as ex:  # noqa: BLE001\n            st["vms2_ok"] = False\n'
         '            print(f"[vms2] call raised ({ex!r}): shipped VMS from here on", flush=True)\n'
         '    vs = st["vms"]\n', "VMS2 route in _vms_call"),
        ("    _vms_prepare(st, w, dev)\n", "    _vms_prepare(st, w, dev)\n    _vms2_prepare(st, w, dev)\n", "VMS2 prepare hook"),
        ('    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n',
         '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n    if VMS2_ENABLE:\n        jobs.append(("vms2", _vms2_build))\n',
         "VMS2 PT1 build job"),
    ):
        GM.check(t.count(a) == 1, f"anchor for {what}: found {t.count(a)}")
        t = t.replace(a, b2)
        rep.append(what)
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_mb] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_mb] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


def main():
    if "--vms2" in sys.argv:
        return main_vms2()
    GM.check(not OUT.exists(), f"{OUT.name} exists; pick another --n")
    npts = assert_closed_form()
    print(f"[gen_k3_mb] closed form == countdown loop at all {npts:,} (base0, nj, tot) points")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    GM.check(b.count(LOOP) == 1, f"VML countdown loop found {b.count(LOOP)} times in {BASE.name}")
    nonce = (f"# nonce {TAG} ({BASE.stem} + VML route(): balanced job size in closed form instead of the countdown loop, "
             "identical jobs; MoE board loop mb_4 / mb_5)")
    t = b.replace(bn, nonce).replace(LOOP, CLOSED)
    rep.append(f"VML route(): countdown loop -> closed form ({LOOP!r})")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_mb] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_mb] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    # A/B copy (box only): the loop and the closed form as compile-time variants of one VML block (VM_CF), both built in
    # prepare; arms switch VML_VAR 's' (= the base) / 'sc' (= k3_mb_1's VML) at call time
    ab = b.replace(bn, nonce.replace(TAG, TAG + "-ab"))
    ab = ab.replace(LOOP, "#ifndef VM_CF" + NL + "#define VM_CF 0" + NL + "#endif" + NL + "#if VM_CF" + NL + CLOSED + NL
                    + "#else" + NL + LOOP + NL + "#endif")
    old_defs = '_VML_DEFS = {"": (), "o": ("-DVM_TRICKS=0",), "s": ("-DVM_TRICKS_DQ=0",)}'
    GM.check(ab.count(old_defs) == 1 and ab.count(NL + "VML_ABVARS = ()" + NL) == 1 and ab.count(NL + "VML_VAR = 's'" + NL) == 1,
             "VML variant glue of the base differs")
    ab = ab.replace(old_defs, old_defs[:-1] + ', "sc": ("-DVM_TRICKS_DQ=0", "-DVM_CF=1")}')
    ab = ab.replace(NL + "VML_ABVARS = ()" + NL, NL + "VML_ABVARS = ('sc',)" + NL)
    labp = ROOT / f"kernels/lab/k3_mb_{N}_ab.py"
    GM.check(not labp.exists(), f"{labp.name} exists")
    labp.write_text(ab, encoding="utf-8", newline=NL)
    import py_compile
    py_compile.compile(str(labp), doraise=True)
    print(f"[gen_k3_mb] wrote {labp.relative_to(ROOT)} ({len(ab.encode())} B; VML_VAR 's' = base, 'sc' = closed form)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

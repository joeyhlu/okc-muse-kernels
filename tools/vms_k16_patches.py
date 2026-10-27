"""vms_k16_patches.py: VMS-K16 = the VMS kernel with jobs of <= 16 tokens (2 n-tiles, one dequant feeding 2 HMMA; lab flag
VMS_FL bit 128, kernels/lab/vms_moe_lab.py) for 9 <= T <= VMS_MAXT, so an expert with 9..16 tokens streams its weights once
instead of twice (R-vms-6, notes/study/vm_small.md). T <= 8 keeps the v1 VMS module untouched.

Base: the stripped layer file on disk (--base, default k3_vml_3 = k3_vms_1k + VA1 + VML1), asserted a finish() fixed point.
  * VMS16_CUDA_SRC / VMS16_CPP_SRC = the live lab sources (plain top-level strings); PT1 job "vms16" builds them with
    -DVMS_FL=128 (VMS16_FL) as their own extension; it shares the v1 module's weight copy and scratch (same protocol).
  * _vms_prepare: after the v1 self-test, _vms16_prepare self-tests VMS-K16 vs K2A and FP32 at bs 9 / 12 / 16 (normal,
    skewed, hot8; two calls each, err flag) -> st["vms16_ok"]; loud print either way.
  * _vms_call: m > 8 and VMS16_ENABLE and st["vms16_ok"] -> the K16 module (VMS16_ENABLE read at call time: A/B arms).

    python -I tools/vms_k16_patches.py [--n=2] [--base=k3_vml_3] [--check]   # -> kernels/muse_decode_layer/k3_vms_<n>.py + lab copy
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vms_patches  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 2))
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vml_3')}.py"
NL = chr(10)
OUT = ROOT / f"kernels/muse_decode_layer/k3_vms_{N}.py"
LAB = ROOT / f"kernels/lab/k3_vms_{N}_ab.py"
TAG = f"k3-vms-{N}"
NONCE = (f"# nonce {TAG} (k3_va_1 + VMS-K16: 16-token VMS jobs at 9 <= T <= 16 (one weight stream per expert), "
         "self-tested vs K2A with fallback)")

PY16 = r'''
VMS16_ENABLE = True
VMS16_FL = 128


def _vms16_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", f"-DVMS_FL={VMS16_FL}", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS16_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vms16_{tag}", cpp_sources=VMS16_CPP_SRC, cuda_sources=VMS16_CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags)
    _VMS["G16"] = G
    _VMS["mod16"] = mod
    _K3_BUILD_S["vms16"] = time.time() - t1
    return mod


def _vms16_prepare(st, w, dev):
    st["vms16_ok"] = False
    if not VMS16_ENABLE or not st.get("vms_ok") or VMS_MAXT <= 8:
        print(f"[vms16] not prepared (VMS16_ENABLE={VMS16_ENABLE}, vms_ok={st.get('vms_ok')}, VMS_MAXT={VMS_MAXT})", flush=True)
        return
    t1 = time.time()
    try:
        _pt1_wait("vms16")
        mod = _VMS.get("mod16") if _VMS.get("mod16") is not None else _vms16_build()
        if _VMS.get("G16") != torch.cuda.get_device_properties(0).multi_processor_count:
            print("[vms16] built for a different SM count: VMS-K16 OFF", flush=True)
            return
        st["vms"]["mod16"] = mod
        good = True
        for bs, seed, dist in ((9, 61, "normal"), (12, 62, "skewed"), (16, 63, "normal"), (16, 64, "hot8"), (16, 65, "skewed")):
            if bs > VMS_MAXT:
                continue
            if dist == "hot8":
                d = _k2a36_synth_inputs(bs, seed, dev, "hot8")
                ids = d["topk_ids"]
                d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
            else:
                d = _k2a_self_test_inputs(bs, seed, dev, dist)
            x = d["hidden_states"].contiguous()
            ref = _k2a_ref_fp32(d, w)
            st["err"].zero_()
            st["vms16_ok"] = False
            ok_k = _v11_k2a33_call(x, d["topk_ids"], d["topk_weights"], dict(st, vms_ok=False)).clone()
            torch.cuda.synchronize()
            for k in range(2):
                o = _vms16_call(st, x, d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                e2, _ = _k2a_rel_err(o, ok_k)
                tot, row = _k2a_rel_err(o, ref)
                errf = int(st["vms"]["ctr"][2])
                ok = e2 < 8e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o.float()).all())
                if not ok or k == 0:
                    print(f"[vms16] self-test bs={bs} {dist} call {k}: rel vs K2A {e2:.2e} vs FP32 {tot:.3e} worst row {row:.3e} "
                          f"err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            _k2a33_reset(st)
        st["vms16_ok"] = good
        print(f"[vms16] VMS-K16 {'LIVE at 9 <= T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, v1 VMS at T <= 16'} "
              f"(regs/local/smem/G/NS {list(mod.attrs())}, {time.time() - t1:.1f}s)", flush=True)
    except Exception as ex:
        st["vms16_ok"] = False
        print(f"[vms16] prepare raised {ex!r}: VMS-K16 OFF (v1 VMS at T <= 16)", flush=True)
        try:
            _k2a33_reset(st)
        except Exception:
            pass


def _vms16_call(st, x, local_ids, topk_weights):
    vs = st["vms"]
    ids = local_ids.to(torch.int32).contiguous()
    tw = topk_weights.to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    vs["mod16"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"],
                    vs["ctr"], out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0, VMS16_FL)
    return out

'''

CALL_OLD = """def _vms_call(st, x, local_ids, topk_weights):
    vs = st["vms"]
"""
CALL_NEW = """def _vms_call(st, x, local_ids, topk_weights):
    if x.shape[0] > 8 and VMS16_ENABLE and st.get("vms16_ok"):
        return _vms16_call(st, x, local_ids, topk_weights)
    vs = st["vms"]
"""
PREP_OLD = "    _vms_prepare(st, w, dev)\n"
PREP_NEW = "    _vms_prepare(st, w, dev)\n    _vms16_prepare(st, w, dev)\n"
PT1_OLD = """    if VMS_ENABLE:
        jobs.append(("vms", _vms_build))
"""
PT1_NEW = PT1_OLD + """    if VMS_ENABLE and VMS16_ENABLE:
        jobs.append(("vms16", _vms16_build))
"""
ANCHOR = "VMS_SELF_TEST_CASES = ("


def apply(text):
    cuda, cpp = vms_patches._lab_sources()
    assert "#define VMS_SKT" in cuda, "live lab without the K16 path"
    for nm in ("VMS16_CUDA_SRC", "_vms16_call", "VMS16_ENABLE"):
        assert nm not in text, nm
    block = 'VMS16_CUDA_SRC = r"""' + cuda + '"""\nVMS16_CPP_SRC = r"""' + cpp + '"""\n' + PY16
    for what, old, new in (("block", ANCHOR, block + ANCHOR), ("call", CALL_OLD, CALL_NEW), ("prepare", PREP_OLD, PREP_NEW),
                           ("pt1", PT1_OLD, PT1_NEW)):
        n = text.count(old)
        assert n == 1, f"vms_k16 anchor {what!r} found {n} times"
        text = text.replace(old, new, 1)
    return text


def main():
    """gen_k3_vml.py's pattern: the stripped base file on disk (a finish() fixed point) + this patch, then GM.finish."""
    GM = vms_patches_gm()
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{BASE.name} is not a fixed point of finish()")
    rep = [f"base {BASE.name} {len(b.encode())} B is a finish() fixed point"]
    nonce = NONCE.replace("k3_va_1 +", f"{BASE.stem} +")
    s = GM.finish(apply(b.replace(bn, nonce)), nonce, TAG, chk_base, rep)
    rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B (+{len(s.encode()) - len(b.encode())} B, headroom "
               f"{1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[vms_k16] OK  {r}")
    if "--check" in sys.argv:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs")
        print(f"[vms_k16] --check: {OUT.name} up to date")
        return 0
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vms_k16] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


def vms_patches_gm():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    return GM


if __name__ == "__main__":
    sys.exit(main())

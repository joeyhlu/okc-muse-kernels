"""vms_patches.py: the VMS one-launch T <= VMS_MAXT MoE route (kernels/lab/vms_moe_lab.py, notes/study/vm_small.md) as a
patch on the k3_m32_3 pre-strip layer text.

  * VMS_CUDA_SRC / VMS_CPP_SRC = the lab kernel's sources verbatim (plain top-level strings, no module creation);
    _vms_* helpers = the lab's prepack (bit / byte permutation of the NVFP4 weights: 36,864 B units) and call.
  * PT1 prefetch job "vms" (VMS_ENABLE): load_inline build with VMS_G = this GPU's SM count, NS = 3, sm_90a.
  * _v11_k2a33_prepare: _vms_prepare(st, w, dev) after _sk1_prepare: scale-byte check (0x08..0x7E, else off), prepack,
    self-test vs the layer's own K2A call and the FP32 reference (bs 1 / 3 / 5 / 8 / 12 / 16, normal / skewed / hot,
    two calls each, err flag); on any failure st["vms_ok"] = False with a loud print (K2A stays).
  * _v11_k2a33_call: tokens <= VMS_MAXT and st["vms_ok"] -> _vms_call (PDL launch, as K2A's K1); an exception disables it.

    text = apply_vms(text, maxt=16)
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAB = ROOT / "kernels/lab/vms_moe_lab.py"


def _lab_sources():
    t = LAB.read_text(encoding="utf-8")
    cuda = t.split("CUDA_SRC = r'''", 1)[1].split("'''", 1)[0]
    cpp = t.split("CPP_SRC = r'''", 1)[1].split("'''", 1)[0]
    assert '"""' not in cuda and '"""' not in cpp
    return cuda, cpp


PY_BLOCK = r'''
VMS_ENABLE = True
VMS_MAXT = @MAXT@
VMS_UB = 36864
VMS_POS = (((6, 7, 8, 15), (22, 23, 24, 31)), ((2, 3, 4, 11), (18, 19, 20, 27)),
           ((28, 29, 30, 5), (12, 13, 14, 21)), ((0, 1, 26, 9), (16, 17, 10, 25)))
_VMS = {"mod": None, "G": None}


def _vms_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vms_{tag}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags)
    _VMS["G"] = G
    _VMS["mod"] = mod
    _K3_BUILD_S["vms"] = time.time() - t1
    return mod


def _vms_pack_words(cu):
    nu = cu.shape[0]
    x = cu.view(nu, 2, 8, 8, 8, 4, 4, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8).contiguous()
    x = x.view(nu, 8, 8, 8, 4, 4, 4, 2).to(torch.int64)
    word = torch.zeros(x.shape[:6], dtype=torch.int64, device=cu.device)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << VMS_POS[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return word.view(nu, 8192)


def _vms_prepack_expert(w, sf, gate_up):
    n = w.shape[0]
    rows = torch.arange(n, device=w.device)
    if gate_up:
        rb, i = rows // 16, rows % 16
        rows = torch.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    wr = w[rows]
    cu = torch.stack((wr & 15, wr >> 4), dim=-1).reshape(n // 16, 16, 4096)
    sb = sf.view(torch.uint8)[rows].view(n // 16, 2, 8, 8, 8, 4).permute(0, 3, 2, 5, 4, 1).contiguous().view(n // 16, 4096)
    words = _vms_pack_words(cu).view(torch.uint8).view(n // 16, 32768)
    return torch.cat((words, sb), dim=1).contiguous()


def _vms_call(st, x, local_ids, topk_weights):
    vs = st["vms"]
    ids = local_ids.to(torch.int32).contiguous()
    tw = topk_weights.to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    vs["mod"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],
                  out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0)
    return out


def _vms_prepare(st, w, dev):
    st["vms_ok"] = False
    if not VMS_ENABLE:
        return
    t1 = time.time()
    try:
        _pt1_wait("vms")
        mod = _VMS["mod"] if _VMS["mod"] is not None else _vms_build()
        if _VMS["G"] != torch.cuda.get_device_properties(0).multi_processor_count:
            print("[vms] built for a different SM count: VMS OFF (K2A at every T)", flush=True)
            return
        b13, b2 = w["w13_sf"].view(torch.uint8), w["w2_sf"].view(torch.uint8)
        if not (bool(((b13 >= 8) & (b13 <= 0x7E)).all()) and bool(((b2 >= 8) & (b2 <= 0x7E)).all())):
            print("[vms] block-scale bytes outside 0x08..0x7E (PRMT scale decode inexact): VMS OFF (K2A at every T)", flush=True)
            return
        E = w["w13"].shape[0]
        if E != K2A_E_LOC or tuple(w["w13"].shape[1:]) != (2 * K2A_D_EXPERT, K2A_D_HIDDEN // 2):
            print("[vms] expert shapes differ: VMS OFF", flush=True)
            return
        wA = torch.empty((E, 512, VMS_UB), dtype=torch.uint8, device=dev)
        wB = torch.empty((E, 256, VMS_UB), dtype=torch.uint8, device=dev)
        for e in range(E):
            wA[e] = _vms_prepack_expert(w["w13"][e], w["w13_sf"][e], True)
            wB[e] = _vms_prepack_expert(w["w2"][e], w["w2_sf"][e], False)
        two62 = float(2 ** 62)
        st["vms"] = {"mod": mod, "wA": wA, "wB": wB, "gsA": (w["w13_gs"].float().reshape(E) * two62).contiguous(),
                     "gsB": (w["w2_gs"].float().reshape(E) * two62).contiguous(),
                     "nw": (1.0 + w["post_norm_w"].float()).contiguous(),
                     "inter": torch.zeros((128, K2A_D_EXPERT), dtype=torch.bfloat16, device=dev),
                     "ybuf": torch.zeros((128, K2A_D_HIDDEN), dtype=torch.float32, device=dev),
                     "ss": torch.zeros((256,), dtype=torch.float32, device=dev),
                     "ctr": torch.zeros((72,), dtype=torch.int32, device=dev),
                     "nomarks": torch.empty((0,), dtype=torch.int64, device=dev)}
        good = True
        for bs, seed, dist in VMS_SELF_TEST_CASES:
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
            ok_k = _v11_k2a33_call(x, d["topk_ids"], d["topk_weights"], st).clone()
            torch.cuda.synchronize()
            for k in range(2):
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
            _k2a33_reset(st)
        st["vms_ok"] = good
        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "
              f"(regs/local/smem/G/NS {list(mod.attrs())}, {time.time() - t1:.1f}s)", flush=True)
    except Exception as ex:
        st["vms_ok"] = False
        print(f"[vms] prepare raised {ex!r}: VMS OFF (K2A at every T)", flush=True)
        try:
            _k2a33_reset(st)
        except Exception:
            pass


VMS_SELF_TEST_CASES = ((1, 41, "normal"), (1, 42, "skewed"), (3, 43, "normal"), (5, 44, "normal"), (8, 45, "normal"),
                       (8, 46, "skewed"), (12, 47, "skewed"), (16, 48, "normal"), (16, 49, "hot8"), (16, 50, "skewed"))

'''

CALL_OLD = """    rep = _k2a_bucket_of(x.shape[0])
    if M32_PX and rep == 32"""
CALL_NEW = """    if VMS_ENABLE and st.get("vms_ok") and x.shape[0] <= VMS_MAXT:
        try:
            return _vms_call(st, x, local_ids, topk_weights)
        except Exception as ex:
            print(f"[k3] VMS route raised ({ex!r}): K2A from here on", flush=True)
            st["vms_ok"] = False
    rep = _k2a_bucket_of(x.shape[0])
    if M32_PX and rep == 32"""
PREP_OLD = "    _sk1_prepare(st, w, dev)\n"
PREP_NEW = "    _sk1_prepare(st, w, dev)\n    _vms_prepare(st, w, dev)\n"
PT1_OLD = """    if GD_BUILD:
        jobs.append(("gd", _gd_build))
"""
PT1_NEW = PT1_OLD + """    if VMS_ENABLE:
        jobs.append(("vms", _vms_build))
"""
DEF_ANCHOR = "def _v11_k2a33_call(x, local_ids, topk_weights, st):\n"


def _once(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"vms_patches: anchor {what} found {n} times"
    return text.replace(old, new)


def apply_vms(text, maxt=16):
    cuda, cpp = _lab_sources()
    for nm in ("VMS_CUDA_SRC", "VMS_CPP_SRC", "_vms_call", "_vms_prepare", "VMS_ENABLE"):
        assert nm not in text, f"vms_patches: {nm} already in the text"
    assert re.search(r"^import hashlib$", text, re.M) and re.search(r"^import time$", text, re.M), "imports"
    block = ('VMS_CUDA_SRC = r"""' + cuda + '"""\nVMS_CPP_SRC = r"""' + cpp + '"""\n' + PY_BLOCK.replace("@MAXT@", str(int(maxt)))
             + "\n")
    text = _once(text, DEF_ANCHOR, block + DEF_ANCHOR, "def _v11_k2a33_call")
    text = _once(text, CALL_OLD, CALL_NEW, "_v11_k2a33_call rep line")
    text = _once(text, PREP_OLD, PREP_NEW, "_sk1_prepare call")
    text = _once(text, PT1_OLD, PT1_NEW, "PT1 gd job")
    return text

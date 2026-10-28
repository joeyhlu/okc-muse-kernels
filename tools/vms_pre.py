"""vms_pre.py: VMS-PRE = tail_kernel2 kept; VMS does its own top-8 routing from the side-stream router32 partials and issues
its first NS weight copies BEFORE its griddepcontrol.wait (parent lever S1, notes/study/system_gap_oct28.md). The norm stays
in tail_kernel2: after the wait VMS reads tail2's xn as before. Kernel = tools/vms_abs.py's bitwise routing edits with
VMS_ABS_NORM = 0 (the norm block reduces to the PDL wait; no grid-wide counter). Legal: router32 completed before tail2
started (full graph edge), and VMS cannot start before tail2 starts.

Layer patch on a stripped layer file (default k3_slim_5; finish() fixed point): VMS_PRE_CUDA_SRC / _CPP_SRC (own copy; the v1
VMS source stays byte-identical), PT1 build "vmspre", _vms_pre_prepare (self-test vs tail_kernel2 + VMS v1: global ids
bitwise, output rel < 1e-4 = VMS's own run-to-run spread, near-tie top-8, BF16 / FP32 pooler outputs, two calls each;
loud fallback), forward: T <= VMS_PRE_MAXT with 3-D side-stream logits -> tail_kernel2 then VMS-PRE (instead of
_v8_experts). VMS_PRE_ENABLE is read at call time (A/B arms).

    python -I tools/vms_pre.py [--n=6] [--base=k3_slim_5] [--check]   # -> kernels/muse_decode_layer/k3_vms_<n>.py
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vms_abs  # noqa: E402

NL = chr(10)
Q3 = vms_abs.Q3
BS = chr(92)

PRE_EDITS = (
    ("norm define", "#ifndef VMS_ABS\n#define VMS_ABS 0\n#endif\n",
     "#ifndef VMS_ABS\n#define VMS_ABS 0\n#endif\n#ifndef VMS_ABS_NORM\n#define VMS_ABS_NORM 1\n#endif\n"),
    ("norm off", "#if VMS_ABS\n    {   // xn = tail_kernel2's pre-expert norm",
     "#if VMS_ABS && !VMS_ABS_NORM\n#if __CUDA_ARCH__ >= 900\n    asm volatile(\"griddepcontrol.wait;" + BS + "n\" ::: \"memory\");\n#endif\n"
     "#elif VMS_ABS\n    {   // xn = tail_kernel2's pre-expert norm"),
    ("dbg on", "  if (c == 0 && tid < p.m * 8) {   // self-test view",
     "#if VMS_ABS_NORM\n  if (c == 0 && tid < p.m * 8) {   // self-test view"),
    ("dbg off", "    reinterpret_cast<float*>(p.xnb + 8 * 4096)[128 + tid] = S.twv[tid];\n  }\n",
     "    reinterpret_cast<float*>(p.xnb + 8 * 4096)[128 + tid] = S.twv[tid];\n  }\n#endif\n"),
)


def pre_sources(cuda, cpp):
    cuda, cpp = vms_abs.edit_sources(cuda, cpp)
    for what, old, new in PRE_EDITS:
        n = cuda.count(old)
        assert n == 1, f"vms_pre anchor {what!r} found {n} times"
        cuda = cuda.replace(old, new, 1)
    return cuda, cpp


PY_BLOCK = r'''
VMS_PRE_ENABLE = True
VMS_PRE_MAXT = 8


def _vms_pre_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_ABS=1", "-DVMS_ABS_NORM=0", "-Xptxas=-warn-spills",
             "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_PRE_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vmspre_{tag}", cpp_sources=VMS_PRE_CPP_SRC, cuda_sources=VMS_PRE_CUDA_SRC,
                      functions=["run_abs", "attrs"], extra_cuda_cflags=flags)
    _VMS["Gpre"] = G
    _VMS["mod_pre"] = mod
    _K3_BUILD_S["vmspre"] = time.time() - t1
    return mod


def _vms_pre_call(state, xn, logits, data):
    vs = state["k2a33"]["st"]["vms"]
    v8 = state["v8"]
    m = xn.shape[0]
    xn = xn.contiguous()
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=xn.device)
    vs["mod_pre"].run_abs(xn, xn, logits, v8["bias"], data["local_start"], v8["nw"], vs["wA"], vs["wB"], vs["gsA"], vs["gsB"],
                          vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"], out, vs["nomarks"], m, K2A_POST_NORM_EPS,
                          RMS_EPS, 0.1767766952966369, 1.0, 1e-15, 1, vs["pretid"])
    return out


def _vms_pre_live(state, tokens, place, logits):
    if not (VMS_PRE_ENABLE and state.get("vms_pre_ok") and tokens <= VMS_PRE_MAXT):
        return False
    ok = place != "main" and logits is not None and logits.dim() == 3 and logits.shape[0] <= 8 and logits.stride(2) == 1
    key = ("vmspre_seen", tokens, ok)
    if key not in _VMS:
        _VMS[key] = True
        print(f"[vmspre] T={tokens} place={place} logits={None if logits is None else tuple(logits.shape)} -> "
              f"{'tail_kernel2 + VMS-PRE' if ok else 'tail_kernel2 + VMS'}", flush=True)
    return ok


def _vms_pre_prepare(state):
    state["vms_pre_ok"] = False
    if not VMS_PRE_ENABLE:
        return
    t1 = time.time()
    try:
        k33 = state.get("k2a33")
        st = k33["st"] if k33 is not None else None
        v8 = state.get("v8")
        if st is None or not st.get("vms_ok") or v8 is None or not TL1_TAIL2:
            print(f"[vmspre] OFF (vms_ok={st.get('vms_ok') if st else None}, tail={'yes' if v8 else 'no'}, TL1={TL1_TAIL2})",
                  flush=True)
            return
        _pt1_wait("vmspre")
        mod = _VMS.get("mod_pre") if _VMS.get("mod_pre") is not None else _vms_pre_build()
        if _VMS.get("Gpre") != torch.cuda.get_device_properties(0).multi_processor_count:
            print("[vmspre] built for a different SM count: OFF", flush=True)
            return
        dev = st["vms"]["wA"].device
        st["vms"]["mod_pre"] = mod
        st["vms"]["pretid"] = torch.zeros((16, TOP_K), dtype=torch.int32, device=dev)
        good = True
        for T, seed, kind, ydt in ((1, 81, "normal", torch.bfloat16), (1, 82, "normal", torch.float32), (5, 83, "tie", torch.bfloat16),
                                   (8, 84, "normal", torch.bfloat16), (8, 85, "tie", torch.float32), (2, 86, "normal", torch.float32)):
            g = torch.Generator(device=dev).manual_seed(seed)
            y = torch.randn((T, K2A_D_HIDDEN), generator=g, device=dev).to(ydt)
            lg = torch.randn((8, T, N_EXPERTS), generator=g, device=dev) * 0.5
            ls = int(torch.randint(0, 8, (1,), generator=g, device=dev).item()) * 32
            v8b = v8["bias"].clone()
            if kind == "tie":
                lg[:, :, ls:ls + 40] = 0.7
                v8["bias"][ls:ls + 40] = v8b[ls]
            data = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
            try:
                for k in range(2):
                    xn, tw, tid, lid = _v8_tail(y, lg, state, data)
                    ref = _vms_call(st, xn, lid, tw).clone()
                    st["vms"]["ctr"][2] = 0
                    o = _vms_pre_call(state, xn, lg, data)
                    torch.cuda.synchronize()
                    errf = int(st["vms"]["ctr"][2])
                    ids_ok = bool(torch.equal(st["vms"]["pretid"][:T], tid))
                    rel = float((o.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12))
                    same = ids_ok and rel < 1e-4 and errf == 0 and bool(torch.isfinite(o.float()).all())
                    if not same or k == 0:
                        print(f"[vmspre] self-test T={T} {kind} y={str(ydt)[6:]} call {k}: ids {ids_ok} out rel {rel:.1e} err={errf} "
                              f"-> {'ok' if same else 'FAIL'}", flush=True)
                    good &= same
            finally:
                v8["bias"].copy_(v8b)
            _k2a33_reset(st)
        state["vms_pre_ok"] = good
        print(f"[vmspre] VMS-PRE {'LIVE at T <= ' + str(VMS_PRE_MAXT) if good else 'OFF: self-test failed, tail_kernel2 + VMS'} "
              f"(regs/local/smem/G/NS {list(mod.attrs())}, {time.time() - t1:.1f}s)", flush=True)
    except Exception as ex:
        state["vms_pre_ok"] = False
        print(f"[vmspre] prepare raised {ex!r}: OFF (tail_kernel2 + VMS)", flush=True)


'''
FWD_TAIL_OLD = "        if tail is not None:\n            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(expert_input, logits, state, data)\n"
FWD_TAIL_NEW = FWD_TAIL_OLD + "            if _vms_pre_live(state, tokens, place, logits):\n                pre_lg = logits\n"
FWD_EXP_OLD = "        expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)\n"
FWD_EXP_NEW = ("        expert_output = (_vms_pre_call(state, expert_x, pre_lg, data) if pre_lg is not None\n"
               "                         else _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens))\n")
FWD_INIT_OLD = "    g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n"
FWD_INIT_NEW = "    pre_lg = None\n" + FWD_INIT_OLD
PT1_OLD = '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n'
PT1_NEW = PT1_OLD + '    if VMS_ENABLE and VMS_PRE_ENABLE:\n        jobs.append(("vmspre", _vms_pre_build))\n'
PREP_OLD = "    _p206_prepare(state, t0)\n"
PREP_NEW = "    _p206_prepare(state, t0)\n    _vms_pre_prepare(state)\n"
ANCHOR = "def _v8_tail(y, logits, state, data):\n"


def apply_layer(text):
    assert "VMS_PRE_ENABLE" not in text and "run_abs" not in text
    cuda, cpp = pre_sources(vms_abs._get(text, "VMS_CUDA_SRC"), vms_abs._get(text, "VMS_CPP_SRC"))
    block = "VMS_PRE_CUDA_SRC = r" + Q3 + cuda + Q3 + NL + "VMS_PRE_CPP_SRC = r" + Q3 + cpp + Q3 + NL
    text = text.replace(ANCHOR, block + ANCHOR, 1)
    for what, old, new in (("fwd tail", FWD_TAIL_OLD, FWD_TAIL_NEW), ("fwd experts", FWD_EXP_OLD, FWD_EXP_NEW),
                           ("fwd init", FWD_INIT_OLD, FWD_INIT_NEW), ("pt1", PT1_OLD, PT1_NEW), ("prepare", PREP_OLD, PREP_NEW),
                           ("py block", ANCHOR, PY_BLOCK + ANCHOR)):
        n = text.count(old)
        assert n == 1, f"vms_pre layer anchor {what!r} found {n} times"
        text = text.replace(old, new, 1)
    return text


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    n = int(args.get("n", 6))
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_slim_5')}.py"
    out = ROOT / f"kernels/muse_decode_layer/k3_vms_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vms_{n}_ab.py"
    tag = f"k3-vms-{n}"
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{base.name} is not a fixed point of finish()")
    rep = [f"base {base.name} {len(b.encode())} B is a finish() fixed point"]
    nonce = (f"# nonce {tag} ({base.stem} + VMS-PRE: tail_kernel2 kept, VMS routes from the router32 partials and issues its "
             "first weight copies before its PDL wait at T <= 8, self-tested vs tail_kernel2 + VMS with fallback)")
    s = GM.finish(apply_layer(b.replace(bn, nonce)), nonce, tag, chk_base, rep)
    rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B ({len(s.encode()) - len(b.encode()):+d} B, headroom "
               f"{1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[vms_pre] OK  {r}")
    if "--check" in sys.argv:
        GM.check(out.exists() and out.read_text(encoding="utf-8") == s, f"{out.name} on disk differs")
        return 0
    out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vms_pre] wrote {out.relative_to(ROOT)} and {lab.relative_to(ROOT)} ({len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""smc_patches.py (R-smc, smem-class builder): three arms on k3_hh_3tkv for the 228 KB carveout-class question (R-hybsm-2).

  python -I tools/smc_patches.py [--base=kernels/lab/k3_hh_3tkv.py] [--mode=lab|ship] [--arms=P,H2,V4] [--out=<file>] [--check]
  python -I tools/lead_gate.py file <out>

  --mode=lab (default; out kernels/lab/k3_smc_ab.py): every arm switchable at run time, all OFF by default (= the ship file's
    behaviour); one prepare builds everything:
      SMC_P  bit 1: tail_kernel3 / tail_kernel2 launch gets cudaLaunchAttributePreferredSharedMemoryCarveout = 100 (max shared)
             bit 2: vm_fin (VML finalize) launch gets the same attribute. A per-LAUNCH attribute (it is recorded in the CUDA-graph
             kernel node and takes precedence over the function attribute), so arms captured into different graphs stay apart.
      SMC_H2 "h2": a second HYB module with the weight ring WGS_NS 3 -> 2 (231,168 -> 194,304 B dynamic smem, the 196 KB class;
             x ring stays 3); "h2p": the same with 36,864 B of unused dynamic smem added back (231,168 B, the 228 class: ring
             depth alone). Own scratch per module; shared weights. Self-tested in prepare (HYB_SELF_TEST + a concentrated
             input, 3 local experts / 69 routes) vs FP32 and vs the ship HYB; a variant that fails stays off.
      SMC_V4 True: the VD cat route at bucket 128 runs vd<2,128,4,3,1,1> (4 stages, 181,376 B, the 196 class) instead of
             vd<2,128,5,3,1,1> (214,160 B); self-tested (VD selftest + bitwise vs the 5-stage route).
  --mode=ship --arms=P|H2|V4 (out kernels/lab/k3_smc_<arms>.py): the arm hard-wired, no lab-only code:
      P  = the two launch-attribute edits with SMC_P = 3;  H2 = WGS_NS 2 in the HYB source;  V4 = VD_INSTS / vd_run 5 -> 4 stages.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hybst_patches as HST  # noqa: E402  (EPF = its OKC_HYBST bit 1 CUDA edits)
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[smc_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


# ---------------- P: carveout launch attribute on the 8 KB-class neighbours (tail before VML / HYB, vm_fin after VML) ----------
TAIL_A = ("cudaLaunchAttribute at[1];\nat[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
          "at[0].val.programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;\ncudaLaunchConfig_t lc = {};\n")
TAIL_B = "lc.attrs = at;\nlc.numAttrs = 1;\nif (y.scalar_type() == at::kFloat)\n"
TAIL_B_NEW = ("lc.attrs = at;\nlc.numAttrs = 1;\n"
              "if (pdl & 4096) { at[1].id = cudaLaunchAttributePreferredSharedMemoryCarveout; at[1].val.sharedMemCarveout = 100; "
              "lc.numAttrs = 2; }   /* R-smc P */\n"
              "if (y.scalar_type() == at::kFloat)\n")
VFIN = ("cudaLaunchAttribute at[1];\nat[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
        "at[0].val.programmaticStreamSerializationAllowed = 1;\ncfg.attrs = at; cfg.numAttrs = 1;\n"
        "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));\n")
VFIN_NEW = ("cudaLaunchAttribute at[2];\nat[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
            "at[0].val.programmaticStreamSerializationAllowed = 1;\ncfg.attrs = at; cfg.numAttrs = 1;\n"
            "if ((V[5] >> 24) & 1) { at[1].id = cudaLaunchAttributePreferredSharedMemoryCarveout; at[1].val.sharedMemCarveout = 100; "
            "cfg.numAttrs = 2; }   /* R-smc P */\n"
            "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));\n")
VFLAGS = "a.magG = (unsigned long long)V[3]; a.flags = (int)V[5];\n"
VFLAGS_NEW = "a.magG = (unsigned long long)V[3]; a.flags = (int)(V[5] & ~(int64_t)(1 << 24));\n"
PY_TAIL = "                      | (_ft_flag(tokens) if ftf is None else ftf))\n"
PY_TAIL_NEW = "                      | (_ft_flag(tokens) if ftf is None else ftf) | (4096 if SMC_P & 1 else 0))\n"
PY_VML = "    fl = (_vml_FLAGS_DEFAULT if fl is None else fl) | (VML_DPFLAGS if dp else 0) | (VML_SCFLAGS if sc else 0)\n"
PY_VML_NEW = PY_VML.rstrip(NL) + " | ((1 << 24) if SMC_P & 2 else 0)\n"


def arm_p(s, on):
    a = s.index('V8_TAIL_CUDA_SRC = r"""')
    b = s.index('"""', a + 30)
    blk = s[a:b]
    check(blk.count(TAIL_A) == 1 and blk.count(TAIL_B) == 1, "tail launch block anchors")
    blk = blk.replace(TAIL_A, TAIL_A.replace("at[1];", "at[2];")).replace(TAIL_B, TAIL_B_NEW)
    s = s[:a] + blk + s[b:]
    s = sub(s, VFIN, VFIN_NEW, "vm_fin launch")
    s = sub(s, VFLAGS, VFLAGS_NEW, "VML a.flags")
    s = sub(s, PY_TAIL, PY_TAIL_NEW, "_v8_tail flags")
    s = sub(s, PY_VML, PY_VML_NEW, "_vml_route flags")
    s = sub(s, "VML_ENABLE = True\n", f"SMC_P = {3 if on else 0}\nVML_ENABLE = True\n", "SMC_P global")
    return s


# ---------------- H2: HYB weight ring 3 -> 2 slots -----------------------------------------------------------------------------
NS3 = "constexpr int WGS_NS = 3;\n"
HYB_CALL = '''def _hyb_call(x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    if _HYB["pdl"] != pdl:
        _HYB["mod"].okc_set_pdl(pdl)
        _HYB["pdl"] = pdl
    _HYB["mod"].okc_run(_HYB["w"], _HYB["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
'''
HYB_CALL_NEW = '''def _hyb_call(x, ids, wts):
    H = _HYBX.get(SMC_H2) if SMC_H2 else None
    return _hyb_run(H if (H is not None and H["ok"]) else _HYB, x, ids, wts)
def _hyb_run(H, x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    if H["pdl"] != pdl:
        H["mod"].okc_set_pdl(pdl)
        H["pdl"] = pdl
    H["mod"].okc_run(_HYB["w"], H["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
'''
HYB_LAB = '''SMC_H2 = None
SMC_H2P_PAD = 36864
_HYBX_KEYS = ("h2", "h2e", "cg")
SMC_H2E_EPI = False
_HYBX = {k: {"mod": None, "ok": False, "sc": None, "full": None, "pdl": -1, "build_s": 0.0, "err": None} for k in _HYBX_KEYS}
def _hybx_src(k):
    s = _HYB_MOE_CUDA
    assert s.count("constexpr int WGS_NS = 3;\\n") == 1 and s.count("size_t sm = okc_smem_bytes();\\n") == 1
    if k in ("h2", "h2p", "h2e"):
        s = s.replace("constexpr int WGS_NS = 3;\\n", "constexpr int WGS_NS = 2;\\n")
    if k == "h2p":
        s = s.replace("size_t sm = okc_smem_bytes();\\n", "size_t sm = okc_smem_bytes() + (size_t)%d;\\n" % SMC_H2P_PAD)
    if k == "h2e":
        for a, n in (_SMC_EPF if SMC_H2E_EPI else _SMC_EPF[:3]):
            assert s.count(a) == 1, a[:60]
            s = s.replace(a, n)
    fl = list(HYB_FLAGS) + (["-DOKC_HYBST=1"] if k == "h2e" else ["-Xptxas=-dlcm=cg"] if k == "cg" else [])
    return s, fl
def _hybx_build():
    import threading
    from torch.utils.cpp_extension import load_inline
    def one(k):
        H = _HYBX[k]
        try:
            src, fl = _hybx_src(k)
            tag = hashlib.sha1((_HYB_MOE_CPP + src + " ".join(fl)).encode()).hexdigest()[:12]
            t0 = time.time()
            H["mod"] = load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=src,
                                   functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
            H["build_s"] = time.time() - t0
        except Exception as ex:
            H["err"] = repr(ex)[-600:]
    ths = [threading.Thread(target=one, args=(k,)) for k in _HYBX_KEYS]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
def _hybx_conc(T, dev, seed=777):
    ids = torch.full((T, 8), -1, dtype=torch.int32)
    for t in range(69):
        ids[t, 0] = (0, 7, 19)[t % 3]
    g = torch.Generator(device="cpu").manual_seed(4241 * seed + T)
    tw = torch.rand((T, 8), generator=g) + 0.05
    return (torch.randn((T, 4096), generator=g).to(torch.bfloat16).to(dev).contiguous(), ids.to(dev).contiguous(),
            (tw / tw.sum(-1, keepdim=True)).float().to(dev).contiguous())
def _hybx_prepare(st, w, dev):
    _pt1_wait("hybx")
    cases = []
    for bs, seed, dist in HYB_SELF_TEST:
        d = _k2a_self_test_inputs(bs, seed, dev, dist)
        cases.append((f"{bs}{dist[0]}", d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(),
                      d["topk_weights"].float().contiguous()))
    cases.append(("conc69",) + _hybx_conc(128, dev))
    refs = []
    for name, x, ids, tw in cases:
        d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": {"local_start": 0, "local_count": K2A_E_LOC}}
        refs.append((_k2a_ref_fp32(d, w), _hyb_run(_HYB, x, ids, tw).clone()))
    for k, H in _HYBX.items():
        H["ok"] = False
        if H["mod"] is None:
            print(f"[smc] HYB {k}: build failed {H['err']}", flush=True)
            continue
        try:
            H["sc"], H["full"] = _hyb_scratch(dev)
            torch.cuda.synchronize()
            good, worst = True, [0.0, 0.0, 0.0, 0.0]
            for (name, x, ids, tw), (ref, o3) in zip(cases, refs):
                oa = _hyb_run(H, x, ids, tw).clone()
                ob = _hyb_run(H, x, ids, tw).clone()
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(oa, ref)
                ev = _k2a_rel_err(oa, o3)[0]
                rp = _k2a_rel_err(ob, oa)[0]
                fin = bool(torch.isfinite(oa.float()).all() and torch.isfinite(ob.float()).all())
                ok = fin and tot < 1e-2 and row < 5e-2 and ev < 1e-2 and rp < 1e-3
                worst = [max(worst[0], tot), max(worst[1], row), max(worst[2], ev), max(worst[3], rp)]
                if not ok:
                    print(f"[smc] HYB {k} self-test {name}: vs FP32 {tot:.2e} row {row:.2e} vs ship {ev:.2e} repeat {rp:.1e} finite {fin} -> FAIL", flush=True)
                good &= ok
            gok = all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in H["full"])
            H["ok"] = bool(good and gok)
            print(f"[smc] HYB {k} {'LIVE' if H['ok'] else 'OFF'}: {len(cases)} cases "
                  f"worst vs FP32 {worst[0]:.2e} row {worst[1]:.2e} vs ship {worst[2]:.2e} repeat {worst[3]:.1e}, guards {gok}, nvcc {H['build_s']:.0f}s", flush=True)
        except Exception as ex:
            H["ok"] = False
            print(f"[smc] HYB {k} prepare raised {ex!r} -> OFF", flush=True)
'''
HYB_PREP_END = ('              f"{len(THIRD_PARTY_NOTICES)} chars of third-party notices kept)", flush=True)\n'
                '    except Exception as ex:\n        _HYB["ok"] = False\n')
HYB_PREP_END_NEW = ('              f"{len(THIRD_PARTY_NOTICES)} chars of third-party notices kept)", flush=True)\n'
                    '        if _HYB["ok"]:\n            _hybx_prepare(st, w, dev)\n'
                    '    except Exception as ex:\n        _HYB["ok"] = False\n')
PT1_HYB = '    if HYB_ENABLE:\n        jobs.append(("hyb", _hyb_build))\n'
PT1_HYB_NEW = PT1_HYB + '        jobs.append(("hybx", _hybx_build))\n'


EPF = ((HST.A_DEF, HST.N_DEF), (HST.A_SEGT, HST.N_SEGT), (HST.A_FIRST, HST.N_FIRST), (HST.A_EPI, HST.N_EPI))
NOTE_EPF = ("helpers.\nModified by team joeyhlu (2026-10-31, tools/smc_patches.py): weight ring 3 -> 2 slots (WGS_NS) and, with\n"
            "-DOKC_HYBST bit 1 (tools/hybst_patches.py), the unit-end epilogue constants (gs, normw lines)\n"
            "prefetched to L1 at each segment start; HYB's arithmetic is unchanged.\n\"\"\"\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n")


def hyb_cuda(s):
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    return i0, i1, s[i0:i1]


def arm_h2_lab(s):
    _, _, cu = hyb_cuda(s)
    for a, n in EPF:
        check(cu.count(a) == 1 and s.count(a) == 1, f"EPF anchor not unique in _HYB_MOE_CUDA: {a[:60]!r}")
    s = sub(s, HYB_CALL, HYB_CALL_NEW, "_hyb_call")
    s = sub(s, "def _hyb_on(m):\n", HYB_LAB + f"_SMC_EPF = {list(EPF)!r}\n" + "def _hyb_on(m):\n", "HYB lab block")
    s = sub(s, HYB_PREP_END, HYB_PREP_END_NEW, "_hyb_prepare end")
    s = sub(s, PT1_HYB, PT1_HYB_NEW, "PT1 hyb job")
    return s


def arm_h2_ship(s, epf):
    i0, i1, cu = hyb_cuda(s)
    check(cu.count(NS3) == 1, "WGS_NS anchor")
    cu = cu.replace(NS3, "constexpr int WGS_NS = 2;   /* R-smc H2: weight ring 3 -> 2 (194,304 B, 196 KB class) */\n")
    if epf:
        for a, n in EPF[:3]:   # prefetch only (the hoisted epilogue, EPF[3], spills 4 / 32 B with WGS_NS 2)
            check(cu.count(a) == 1 and s.count(a) == 1, f"EPF anchor {a[:60]!r}")
            cu = cu.replace(a, n.replace("#define OKC_HYBST 0\n", "#define OKC_HYBST 1\n"))
    s = s[:i0] + cu + s[i1:]
    return sub(s, HST.A_NOTE, NOTE_EPF, "HYB notice end")


# ---------------- V4: VD cat route at bucket 128, 5 -> 4 pipeline stages ------------------------------------------------------
VD_INSTS = "VD_INSTS = [(2, 128, 5, 3, 1, 1)]\n"
VD_CASE = "    case 0: return vd_launch_t<2, 128, 5, 3, 1, 1>(x, wpk, wraw, y, err, stamps, nrb, w_rb, w_s, flags, pdl, pf);\n"
VD_RP_END = ('        print(f"[vd] {proj}/{b} inst {VD_INSTS[kid]}: {msg} -> {\'LIVE\' if good else \'FAILED: today path\'}", flush=True)\n'
             '        if good:\n            st["route"][(proj, b)] = r\n')
VD_RP_NEW = VD_RP_END + '''    st["route4"] = {}
    for key, r in sorted(st["route"].items()):
        if VD_INSTS[r["kid"]] != (2, 128, 5, 3, 1, 1):
            continue
        try:
            r4 = dict(r, kid=SMC_V4_KID)
            w = _dg1_weight(state, r["proj"])
            good, msg = _vd_selftest(st, r4, w, min(r4["T"], key[1]), gen)
            same = True
            for t in sorted({min(r4["T"], key[1]), 77, 1}):
                x = torch.randn((t, r["K"]), generator=gen, device=w.device).to(torch.bfloat16)
                o5 = torch.full((t, r["N"]), float("nan"), dtype=torch.float32 if r["f32"] else torch.bfloat16, device=w.device)
                o4 = o5.clone()
                _vd_call(st, r, x, o5)
                _vd_call(st, r4, x, o4)
                torch.cuda.synchronize()
                same &= bool(torch.equal(o4, o5))
            good = good and same
        except Exception as ex:
            good, msg, same = False, repr(ex)[:300], False
        print(f"[smc] VD {key} inst {VD_INSTS[SMC_V4_KID]}: {msg}; bitwise vs 5 stages {same} -> {'LIVE' if good else 'OFF'}", flush=True)
        if good:
            st["route4"][key] = r4
'''
VD_SEL = ('    r = st["route"].get((proj, b))\n'
          '    if r is None or T > r["T"] or x.shape[1] != r["K"] or not x.is_contiguous() or x.data_ptr() % 16 != 0:\n')
VD_SEL_NEW = ('    r = (st.get("route4") or {}).get((proj, b)) if SMC_V4 else None\n'
              '    r = r if r is not None else st["route"].get((proj, b))\n'
              '    if r is None or T > r["T"] or x.shape[1] != r["K"] or not x.is_contiguous() or x.data_ptr() % 16 != 0:\n')


def arm_v4_lab(s):
    s = sub(s, VD_INSTS, "VD_INSTS = [(2, 128, 5, 3, 1, 1), (2, 128, 4, 3, 1, 1)]\nSMC_V4 = False\nSMC_V4_KID = 1\n", "VD_INSTS")
    s = sub(s, VD_CASE, VD_CASE + VD_CASE.replace("case 0", "case 1").replace("128, 5, 3", "128, 4, 3"), "vd_run case")
    s = sub(s, VD_RP_END, VD_RP_NEW, "_vd_routes_prepare end")
    s = sub(s, VD_SEL, VD_SEL_NEW, "_vd route select")
    return s


def arm_v4_ship(s):
    s = sub(s, VD_INSTS, "VD_INSTS = [(2, 128, 4, 3, 1, 1)]\n", "VD_INSTS")
    return sub(s, VD_CASE, VD_CASE.replace("128, 5, 3", "128, 4, 3"), "vd_run case")


def apply(s, base_name, mode, arms):
    lines = s.split(NL)
    tag = "ab" if mode == "lab" else "-".join(arms)
    lines[1] = (f"# nonce k3-smc-{tag} ({base_name} + R-smc {mode}: " + ("SMC_H2 h2 / h2p / h2e / cg + SMC_V4 switchable, all off by default"
                if mode == "lab" else "+".join(arms)) + "; tools/smc_patches.py)")
    s = NL.join(lines)
    if mode == "lab":
        s = arm_h2_lab(s)
        s = arm_v4_lab(s)
    else:
        if "P" in arms:
            s = arm_p(s, True)
        if "H2" in arms or "H2E" in arms:
            s = arm_h2_ship(s, "H2E" in arms)
        if "V4" in arms:
            s = arm_v4_ship(s)
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3tkv.py")
    mode = ARGS.get("mode", "lab")
    arms = [a for a in ARGS.get("arms", "").split(",") if a]
    check(mode in ("lab", "ship"), f"mode {mode}")
    check(mode == "lab" or (arms and set(arms) <= {"P", "H2", "H2E", "V4"} and not {"H2", "H2E"} <= set(arms)), f"ship needs --arms from P,H2|H2E,V4: {arms}")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_smc_ab.py" if mode == "lab" else f"kernels/lab/k3_smc_{'_'.join(arms)}.py")
    s = apply(base.read_bytes().decode("ascii"), base.stem, mode, arms)
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check(len(s.encode()) < LIMIT, f"size {len(s.encode()):,} >= {LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s, f"{out.name} != apply({base.name})")
        print(f"[smc_patches] --check PASS: {out.name} == apply({base.name}, {mode}, {arms}) ({len(s):,} B)")
        return
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[smc_patches] {base.name} -> {out.name} ({mode} {arms}; {len(s):,} B, {len(s) - base.stat().st_size:+,} B)")


if __name__ == "__main__":
    main()

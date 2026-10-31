"""padz_patches.py (R-padz, pad hygiene builder; Fable outside view C1 + D1, Muse notes/study/ext_muse_pad_hygiene.md): zero the
pad rows of the wgmma B operand tiles of HYB (alygg77's WGS MoE engine, T 49..320) and VML (T 17..48).

Both engines stream npad = ((cnt + 7) >> 3) << 3 token rows per job and stage, but write only the r < cnt rows of xA (gather) and the
tk < cnt rows of xB (phase-A epilogue). xA / xB are prepare-time scratch, so the pad rows hold stale activations of earlier calls with
other routings, and every hot wgmma multiplies them (outputs discarded). Zero B columns switch less -> less tensor power -> maybe a
higher SM clock on the tensor-bound hot stages. Outputs must stay bitwise == ship (pad outputs are never stored).

  python -I tools/padz_patches.py [--base=kernels/lab/k3_hh_5_sk2.py] [--out=kernels/lab/k3_padz_ab.py]          lab file
  python -I tools/padz_patches.py --ship --base=<final> --out=<new> [--hyb=1] [--vml=1]                           ship form
  python -I tools/lead_gate.py file <out>

HYB (-DOKC_PADZ, default 0 = the ship text): 1 = wgs_gather stores uint4 zeros for cnt <= r < npad (same slot math as the real rows,
so one writer per 16 B chunk, before the existing CTA_SYNC + red.release on xready); both phase-A epilogue variants (wgs_epi_tile<false>
used by wgs_flush and the direct path without HYBST & 1, wgs_epi_tile_c<false> used by the T >= 193 build HYBST = 7) store a BF16 zero
for cnt <= tk < npad (the lane that owns (col, tk); every n-tile nt < ntt is visited, so the guard fires for every pad element, before
the bar.sync / fence / red.release on doneA(s) that phase B waits on). 2 = the same, enabled at run time by Params.pzr (constant bank, no register)
(lab marks build only: p.pzr = 1 iff a 5th scratch tensor is passed; one module, on / off arms with separate scratch).
VML (-DVM_PADZ, default 0): gather / gather_warp / gather_w5 store zeros for cn <= r < npad (gather_warp = the bf2 T = 32 path), the
consumer self-fill writes zero pad rows into its smem x slots (before its fence.proxy.async + bar.sync 5 + mbarrier arrive), and the
phase-A epilogue (both the VCP_PK packed and the plain store loop) stores zero for cn <= n < npad.
Lab keys: HYBP_ST (0 = ship): 1 = PADZ (5 = PADZ + HYB_F7, used at T >= HYB_T7), 17 = PADZ run-time toggle + per-stage marks
(tools/hybsync_patches.py EDITS with -DOKC_HSY=0 -DOKC_HSY_MK=1 = ship protocol + marks). VMLZ_ON (0 = ship): 1 = the VML_VAR (bf2)
build with -DVM_PADZ=1 and its own scratch. The ship CUDA text stays in the lab file; the edits are applied at run time.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import hybpro_patches as HP  # noqa: E402  (the lab-vehicle Python anchors)
import hybsync_patches as HS  # noqa: E402  (per-stage marks edits, used with OKC_HSY = 0)
sys.argv = _argv
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[padz_patches] FAIL: {msg}")


def sub(text, old, new, what, n_expected=1):
    n = text.count(old)
    check(n == n_expected, f"anchor {what}: found {n}x, expected {n_expected}: {old[:90]!r}")
    return text.replace(old, new)


# ---------------- HYB CUDA edits ---------------------------------------------------------------------------------------------
H_DEF_A = "#ifndef OKC_HYBFB\n#define OKC_HYBFB 4\n#endif\n"
H_DEF_N = H_DEF_A + ("#ifndef OKC_PADZ\n#define OKC_PADZ 0\n#endif\n#if OKC_PADZ == 2\n"
                     "#define HYB_PZ (p.pzr != 0)\n#else\n#define HYB_PZ (OKC_PADZ != 0)\n#endif\n")
H_DA = "d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));\n"
H_GA = "if (r < cnt) {\nconst int ntt = (cnt + 7) >> 3;\n" + H_DA + "{\n"
H_GN = ("#if OKC_PADZ\nif (r < cnt || HYB_PZ) {\nconst int ntt = (cnt + 7) >> 3;\n" + H_DA + "v[k] = make_uint4(0u, 0u, 0u, 0u);\n"
        "if (r < cnt) {\n#else\n" + H_GA + "#endif\n")
H_XB = "xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8)"
H_EA = ("if (tk < cnt) {\nconst float gt = d4[h] * gsc, up = d4[2 + h] * gsc;\nconst float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n"
        "st_keep_bf16(" + H_XB + ",\n__float2bfloat16(probe_q(v)));\n}\n")
H_EN = ("#if OKC_PADZ\nif (tk < cnt || HYB_PZ) {\nconst float gt = d4[h] * gsc, up = d4[2 + h] * gsc;\n"
        "const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\nst_keep_bf16(" + H_XB + ",\n__float2bfloat16(probe_q(tk < cnt ? v : 0.f)));\n}\n"
        "#else\n" + H_EA + "#endif\n")
H_PRM_A = "unsigned long long* dbg;\n"
H_PRM_N = H_PRM_A + "#if OKC_PADZ == 2\nint pzr;\n#endif\n"
H_LAU_A = "p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n"
H_LAU_N = H_LAU_A + "#if OKC_PADZ == 2\np.pzr = scratch.size() > 4 ? 1 : 0;\n#endif\n"
HYB_EDITS = [(H_DEF_A, H_DEF_N, 1), (H_PRM_A, H_PRM_N, 1), (H_GA, H_GN, 1), (H_EA, H_EN, 2), (H_LAU_A, H_LAU_N, 1)]


def apply_hyb(cu):
    for a, n, k in HYB_EDITS:
        cu = sub(cu, a, n, "HYB " + a[:50], k)
    check(cu.count("|| HYB_PZ)") == 3 and cu.count("#if OKC_PADZ\n") == 3 and cu.count("#if OKC_PADZ == 2\n") == 3, "HYB PADZ inserted")
    return cu


# ---------------- VML CUDA edits ---------------------------------------------------------------------------------------------
V_DEF_A = "__device__ void gather(const Args& A, const Tab& T, const uint16_t* ent) {\n"
V_DEF_N = "#ifndef VM_PADZ\n#define VM_PADZ 0\n#endif\n" + V_DEF_A
V_G_A = ("off[k] = T.jrow[j] * 2048 + s * npad * 128 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);\n"
         "v[k] = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + tid);\n}\n")
V_G_N = V_G_A + ("#if VM_PADZ\nelse {\nconst int npad = (cn + 7) & ~7;\n"
                 "off[k] = T.jrow[j] * 2048 + s * npad * 128 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);\n"
                 "v[k] = make_uint4(0u, 0u, 0u, 0u);\n}\n#endif\n")
V_SRC = "const uint4* src = reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + lane;\n"
V_SRCZ = "const uint4* src = reinterpret_cast<const uint4*>(A.x + (size_t)(zr_ ? 0 : (ent[T.jS[j] + r] >> 3)) * KDIM) + lane;\n"
V_W_A = ("if (r >= cn) continue;\nconst int npad = (cn + 7) & ~7;\n" + V_SRC + "uint4 v[16];\n#pragma unroll\n"
         "for (int m = 0; m < 16; ++m) v[m] = __ldg(src + 32 * m);\n")
V_W_N = ("#if VM_PADZ\nconst bool zr_ = r >= cn;\nconst int npad = (cn + 7) & ~7;\n" + V_SRCZ + "uint4 v[16];\n#pragma unroll\n"
         "for (int m = 0; m < 16; ++m) v[m] = zr_ ? make_uint4(0u, 0u, 0u, 0u) : __ldg(src + 32 * m);\n#else\n" + V_W_A + "#endif\n")
V_5_A = ("if (r >= cn) continue;\nconst int npad = (cn + 7) & ~7;\n" + V_SRC
         + "uint32_t* d = xa + T.jrow[j] * 2048 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);\n#pragma unroll\n"
         "for (int h = 0; h < 16 / VM_GV; ++h) {\nuint4 v[VM_GV];\n#pragma unroll\n"
         "for (int m = 0; m < VM_GV; ++m) v[m] = __ldg(src + 32 * (VM_GV * h + m));\n")
V_5_N = ("#if VM_PADZ\nconst bool zr_ = r >= cn;\nconst int npad = (cn + 7) & ~7;\n" + V_SRCZ
         + "uint32_t* d = xa + T.jrow[j] * 2048 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);\n#pragma unroll\n"
         "for (int h = 0; h < 16 / VM_GV; ++h) {\nuint4 v[VM_GV];\n#pragma unroll\n"
         "for (int m = 0; m < VM_GV; ++m) v[m] = zr_ ? make_uint4(0u, 0u, 0u, 0u) : __ldg(src + 32 * (VM_GV * h + m));\n#else\n" + V_5_A + "#endif\n")
V_F_A = ("for (int w = tid; w < cn * 32; w += 32 * NCW) {\nconst int r = w >> 5, ch = w & 31, i = ch >> 3, t = (ch >> 1) & 3, qb = (ch & 1) * 4;\n"
         "const uint4 v = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM + 256 * st) + ch);\n")
V_F_N = ("#if VM_PADZ\nfor (int w = tid; w < ((cn + 7) & ~7) * 32; w += 32 * NCW) {\n"
         "const int r = w >> 5, ch = w & 31, i = ch >> 3, t = (ch >> 1) & 3, qb = (ch & 1) * 4;\n"
         "const uint4 v = (r < cn) ? __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM + 256 * st) + ch)\n"
         ": make_uint4(0u, 0u, 0u, 0u);\n#else\n" + V_F_A + "#endif\n")
V_XP = "xb + ((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + b)) * 4 + tt) * 2"
V_EP_A = "if (n < cn) *reinterpret_cast<uint32_t*>(" + V_XP + ") = w2;\n"
V_EP_N = V_EP_A + "#if VM_PADZ\nelse if (n < npad) *reinterpret_cast<uint32_t*>(" + V_XP + ") = 0u;\n#endif\n"
V_XS = "xb[((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + hh)) * 4 + tt) * 2 + b]"
V_ES_A = "if (n < cn)\n" + V_XS + " =\n__float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));\n"
V_ES_N = V_ES_A + "#if VM_PADZ\nelse if (n < npad)\n" + V_XS + " = __float2bfloat16_rn(0.f);\n#endif\n"
VML_EDITS = [(V_DEF_A, V_DEF_N, 1), (V_G_A, V_G_N, 1), (V_W_A, V_W_N, 1), (V_5_A, V_5_N, 1), (V_F_A, V_F_N, 1), (V_EP_A, V_EP_N, 1),
             (V_ES_A, V_ES_N, 1)]


def apply_vml(cu):
    for a, n, k in VML_EDITS:
        cu = sub(cu, a, n, "VML " + a[:50], k)
    check(cu.count("#if VM_PADZ\n") == 6, "VML PADZ inserted")
    return cu


# ---------------- Python glue (lab vehicle) -----------------------------------------------------------------------------------
CALL_A = '''def _hyb_call(x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
'''
CALL_N = '''def _hyb_call(x, ids, wts):
    m = x.shape[0]
    if HYBP_ST:
        H = _HYBP.get(HYBP_ST | 4) if m >= HYB_T7 else None
        if H is None or not H["ok"]:
            H = _HYBP.get(HYBP_ST)
        if H is not None and H["ok"]:
            return _hybp_run(H, x, ids, wts)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
'''
HYB_LAB = '''HYBP_ST = 0
_HYBP_KEYS = (1, 5, 17)
_HYBP = {k: {"mod": None, "ok": False, "sc": None, "full": None, "pdl": -1, "build_s": 0.0, "err": None, "dbg": None} for k in _HYBP_KEYS}
_HYBP_CPP = _HYB_MOE_CPP
def _hybp_src(k):
    s = _HYB_MOE_CUDA
    for a, n, c in _HYBP_EDITS + (_HYBP_MKEDITS if k & 16 else []):
        assert s.count(a) == c, a[:60]
        s = s.replace(a, n)
    pz = 2 if k & 16 else (1 if k & 1 else 0)
    fl = list(HYB_FLAGS) + (list(HYB_F7) if k & 4 else []) + [f"-DOKC_PADZ={pz}"] + (["-DOKC_HSY=0", "-DOKC_HSY_MK=1"] if k & 16 else [])
    return s, fl
def _hybp_build():
    from torch.utils.cpp_extension import load_inline
    def one(k):
        H = _HYBP[k]
        try:
            src, fl = _hybp_src(k)
            tag = hashlib.sha1((_HYBP_CPP + src + " ".join(fl)).encode()).hexdigest()[:12]
            t0 = time.time()
            H["mod"] = load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYBP_CPP, cuda_sources=src,
                                   functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
            H["build_s"] = time.time() - t0
        except Exception as ex:
            H["err"] = repr(ex)[-600:]
    ths = [threading.Thread(target=one, args=(k,)) for k in _HYBP_KEYS]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
def _hybp_run(H, x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    if H["pdl"] != pdl:
        H["mod"].okc_set_pdl(pdl)
        H["pdl"] = pdl
    sc = H["sc"] + [H["dbg"]] + ([H["dbg"]] if H.get("pzon") else []) if H.get("dbg") is not None else H["sc"]
    H["mod"].okc_run(_HYB["w"], sc, x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
def _hybp_conc(T, dev, seed=777):
    ids = torch.full((T, 8), -1, dtype=torch.int32)
    for t in range(69):
        ids[t, 0] = (0, 7, 19)[t % 3]
    g = torch.Generator(device="cpu").manual_seed(4241 * seed + T)
    tw = torch.rand((T, 8), generator=g) + 0.05
    return (torch.randn((T, 4096), generator=g).to(torch.bfloat16).to(dev).contiguous(), ids.to(dev).contiguous(),
            (tw / tw.sum(-1, keepdim=True)).float().to(dev).contiguous())
def _hybp_ship(m, x, ids, tw, k):
    md = _HYB["mod7"] if ((k & 4) and _HYB["mod7"] is not None) else _HYB["mod"]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    md.okc_run(_HYB["w"], _HYB["sc"], x, ids, tw, out, 0, 32, _HYB["nsm"])
    return out
def _hybp_prepare(st, w, dev):
    _pt1_wait("hybp")
    cases = []
    for bs, seed, dist in HYB_SELF_TEST:
        d = _k2a_self_test_inputs(bs, seed, dev, dist)
        cases.append((f"{bs}{dist[0]}", d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(),
                      d["topk_weights"].float().contiguous()))
    cases.append(("conc69",) + _hybp_conc(128, dev))
    refs = {}
    for name, x, ids, tw in cases:
        d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": {"local_start": 0, "local_count": K2A_E_LOC}}
        refs[name] = _k2a_ref_fp32(d, w)
    for k, H in _HYBP.items():
        H["ok"] = False
        if H["mod"] is None:
            print(f"[hybp] HYB pz{k}: build failed {H['err']}", flush=True)
            continue
        try:
            H["sc"], H["full"] = _hyb_scratch(dev)
            H["dbg"] = torch.zeros(_HYB_SLOTS * 4 * 64, dtype=torch.int64, device=dev) if k & 16 else None
            torch.cuda.synchronize()
            good, worst, nbit, ncase = True, [0.0, 0.0, 0.0, 0.0], 0, 0
            for name, x, ids, tw in cases:
                m = x.shape[0]
                o3 = _hybp_ship(m, x, ids, tw, k).clone()
                o4 = _hybp_ship(m, x, ids, tw, k).clone()
                oa = _hybp_run(H, x, ids, tw).clone()
                ob = _hybp_run(H, x, ids, tw).clone()
                torch.cuda.synchronize()
                rep3 = float((o4.float() - o3.float()).abs().max())
                tot, row = _k2a_rel_err(oa, refs[name])
                ev = _k2a_rel_err(oa, o3)[0]
                dd = float((oa.float() - o3.float()).abs().max())
                rp = _k2a_rel_err(ob, oa)[0]
                fin = bool(torch.isfinite(oa.float()).all() and torch.isfinite(ob.float()).all())
                ok = fin and tot < 1e-2 and row < 5e-2 and ev < 2e-3 and rp < 1e-3
                nbit += dd == 0.0
                ncase += 1
                worst = [max(worst[0], tot), max(worst[1], row), max(worst[2], ev), max(worst[3], rp)]
                if not ok or dd != 0.0:
                    print(f"[hybp] HYB pz{k} self-test {name}: vs FP32 {tot:.2e} row {row:.2e} vs ship {ev:.2e} (max abs {dd:.2e}, "
                          f"ship repeat {rep3:.1e}) repeat {rp:.1e} finite {fin} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            gok = all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in H["full"])
            H["ok"] = bool(good and gok)
            print(f"[hybp] HYB pz{k} {'LIVE' if H['ok'] else 'OFF'}: {ncase} cases ({nbit} bitwise == ship) worst vs FP32 "
                  f"{worst[0]:.2e} row {worst[1]:.2e} vs ship {worst[2]:.2e} repeat {worst[3]:.1e}, guards {gok}, nvcc {H['build_s']:.0f}s",
                  flush=True)
        except Exception as ex:
            H["ok"] = False
            print(f"[hybp] HYB pz{k} prepare raised {ex!r} -> OFF", flush=True)
'''
VML_LAB = '''VMLZ_ON = 0
_VMLZ = {"mod": None, "ok": False, "sc": None, "build_s": 0.0, "err": None}
def _vmlz_build():
    from torch.utils.cpp_extension import load_inline
    try:
        s = _vml_CUDA
        for a, n, c in _VMLZ_EDITS:
            assert s.count(a) == c, a[:60]
            s = s.replace(a, n)
        fl = _vml_FLAGS + list(_VML_DEFS[VML_VAR]) + ["-DVM_PADZ=1"]
        tag = hashlib.sha1((_vml_CPP + s + " ".join(fl)).encode()).hexdigest()[:10]
        t0 = time.time()
        _VMLZ["mod"] = load_inline(name=f"vml_moe_{tag}", cpp_sources=_vml_CPP, cuda_sources=s, functions=["vml_run", "vml_cfg"],
                                   extra_cuda_cflags=fl, verbose=False)
        _VMLZ["build_s"] = time.time() - t0
    except Exception as ex:
        _VMLZ["err"] = repr(ex)[-600:]
def _vmlz_prepare(st, w, dev):
    global VMLZ_ON
    _VMLZ["ok"] = False
    _pt1_wait("vmlz")
    if _VMLZ["mod"] is None or not _VML["ok"].get(32):
        print(f"[vmlz] OFF: build {_VMLZ['err']} / VML bucket 32 {_VML['ok'].get(32)}", flush=True)
        return
    try:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        _VMLZ["sc"] = dict(_vml_alloc(dev, sms), err=_VML["sc"]["err"])
        torch.cuda.synchronize()
        good, nbit, ncase = True, 0, 0
        for bs, seed, dist in ((25, 48, "normal"), (32, 49, "skewed"), (32, 54, "normal"), (48, 50, "skewed")):
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            d = {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                 "topk_weights": d["topk_weights"].float().contiguous(), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
            ref = _k2a_ref_fp32(d, w)
            _VML["sc"]["err"].zero_()
            VMLZ_ON = 0
            o0 = _vml_route(st, d).clone()
            o1 = _vml_route(st, d).clone()
            VMLZ_ON, _VMLZ["ok"] = 1, True
            oa = _vml_route(st, d).clone()
            ob = _vml_route(st, d).clone()
            VMLZ_ON, _VMLZ["ok"] = 0, False
            torch.cuda.synchronize()
            errf = int(_VML["sc"]["err"][0])
            rep0 = float((o1.float() - o0.float()).abs().max())
            tot, row = _k2a_rel_err(oa, ref)
            ev = _k2a_rel_err(oa, o0)[0]
            dd = float((oa.float() - o0.float()).abs().max())
            rp = _k2a_rel_err(ob, oa)[0]
            fin = bool(torch.isfinite(oa.float()).all() and torch.isfinite(ob.float()).all())
            ok = fin and not errf and tot < 1e-2 and row < 5e-2 and ev < 2e-3 and rp < 1e-3
            nbit += dd == 0.0
            ncase += 1
            print(f"[vmlz] self-test bs={bs} {dist}: vs FP32 {tot:.2e} row {row:.2e} vs ship {ev:.2e} (max abs {dd:.1e}, ship repeat "
                  f"{rep0:.1e}) repeat {rp:.1e} err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
            good &= ok
        _VMLZ["ok"] = bool(good)
        print(f"[vmlz] VML PADZ ({VML_VAR}) {'LIVE' if good else 'OFF'}: {ncase} cases ({nbit} bitwise == ship), nvcc {_VMLZ['build_s']:.0f}s",
              flush=True)
    except Exception as ex:
        VMLZ_ON, _VMLZ["ok"] = 0, False
        print(f"[vmlz] prepare raised {ex!r} -> OFF", flush=True)
'''
VROUTE_A = '''    return _vml_call(_VML["mods"].get(var, _VML["mod"]), st, scx, x, data["topk_ids"].to(torch.int32).contiguous(),
'''
VROUTE_N = '''    mz = _VML["mods"].get(var, _VML["mod"])
    if VMLZ_ON and var == VML_VAR and _VMLZ["ok"]:
        mz, scx = _VMLZ["mod"], _VMLZ["sc"]
    return _vml_call(mz, st, scx, x, data["topk_ids"].to(torch.int32).contiguous(),
'''
VPREP_A = '''        _VML["force"] = None
        _m32e_vartest(st, w, dev)
'''
VPREP_N = VPREP_A + '''        _vmlz_prepare(st, w, dev)
'''
PT1_VML = '''    if VML_ENABLE:
        jobs.append(("vml", _vml_build_job))
'''
PT1_VML_N = PT1_VML + '''        jobs.append(("vmlz", _vmlz_build))
'''
NOTE_A = "HYB's arithmetic is unchanged.\n\"\"\"\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n"
NOTE_LAB = ("HYB's arithmetic is unchanged.\nModified by team joeyhlu (2026-10-31, tools/padz_patches.py, lab only): optional builds that "
            "store zeros into the pad rows\n(cnt <= r < npad) of the x / inter B-operand tiles (-DOKC_PADZ) and per-stage timing marks; "
            "HYB's arithmetic and outputs are unchanged.\n\"\"\"\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n")
NOTE_SHIP = ("HYB's arithmetic is unchanged.\nModified by team joeyhlu (2026-10-31, tools/padz_patches.py, -DOKC_PADZ=1): the gather and "
             "both phase-A epilogue variants store zeros\ninto the pad rows (cnt <= r < npad) of the x / inter B-operand tiles instead of "
             "leaving stale activations there;\nHYB's arithmetic, real-row bytes and wgmma order are unchanged.\n\"\"\"\n"
             "THIRD_PARTY_NOTICES += _HYB_NOTICE\n")


def block(s, a, b, what):
    check(s.count(a) == 1 and s.count(b) == 1, what)
    i0, i1 = s.index(a), s.index(b)
    check(i0 < i1, what)
    return i0, i1


def hyb_block(s):
    return block(s, '_HYB_MOE_CUDA = r"""', '_HYB_MOE_CPP = r"""', "HYB CUDA block")


def vml_block(s):
    return block(s, '_vml_CUDA_T = r"""', '_vml_CPP = r"""', "VML CUDA block")


def apply_lab(s, base_name):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = (f"# nonce k3-padz-ab ({base_name} + R-padz lab arms: HYBP_ST 1 = HYB pad zeroing (5 = + F7 at T >= 193), 17 = run-time "
                "toggle + marks; VMLZ_ON 1 = VML pad zeroing; tools/padz_patches.py)")
    s = NL.join(lines)
    i0, i1 = hyb_block(s)
    h = apply_hyb(s[i0:i1])                       # dry runs: the file keeps the ship CUDA text
    for a, n in HS.EDITS:
        h = sub(h, a, n, "HYB marks " + a[:40])
    j0, j1 = vml_block(s)
    apply_vml(s[j0:j1])
    s = sub(s, CALL_A, CALL_N, "_hyb_call")
    mk = [(a, n, 1) for a, n in HS.EDITS]
    s = sub(s, "def _hyb_on(m):\n", HYB_LAB + f"_HYBP_EDITS = {HYB_EDITS!r}\n_HYBP_MKEDITS = {mk!r}\n" + "def _hyb_on(m):\n", "HYBP lab block")
    s = sub(s, HP.HYB_PREP_END, HP.HYB_PREP_END_NEW, "_hyb_prepare end")
    s = sub(s, HP.PT1_HYB, HP.PT1_HYB_NEW, "PT1 hyb job")
    s = sub(s, NOTE_A, NOTE_LAB, "HYB notice end")
    s = sub(s, "def _vml_route(st, data):\n", VML_LAB + f"_VMLZ_EDITS = {VML_EDITS!r}\n" + "def _vml_route(st, data):\n", "VMLZ lab block")
    s = sub(s, VROUTE_A, VROUTE_N, "_vml_route call")
    s = sub(s, VPREP_A, VPREP_N, "_vml_prepare hook")
    s = sub(s, PT1_VML, PT1_VML_N, "PT1 vml job")
    return s


def apply_ship(s, hyb, vml):
    check(hyb or vml, "--hyb=1 and / or --vml=1")
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] += f" + R-padz pad zeroing{' HYB' if hyb else ''}{' VML' if vml else ''} [tools/padz_patches.py]"
    s = NL.join(lines)
    if hyb:
        i0, i1 = hyb_block(s)
        s = s[:i0] + apply_hyb(s[i0:i1]) + s[i1:]
        a_flags = next(ln for ln in s.split(NL) if ln.startswith("HYB_FLAGS = ("))
        check(a_flags.endswith(")") and "-DOKC_PADZ" not in a_flags, "HYB_FLAGS line")
        s = sub(s, a_flags + NL, a_flags[:-1] + ", '-DOKC_PADZ=1')" + NL, "HYB_FLAGS")
        s = sub(s, NOTE_A, NOTE_SHIP, "HYB notice end")
    if vml:
        j0, j1 = vml_block(s)
        s = s[:j0] + apply_vml(s[j0:j1]) + s[j1:]
        a_bf2 = next(ln for ln in s.split(NL) if ln.lstrip().startswith('"bf2": ("-DVM_TRICKS_DQ=0"'))
        check(a_bf2.endswith("),") and "VM_PADZ" not in a_bf2, "bf2 defs line")
        s = sub(s, a_bf2 + NL, a_bf2[:-2] + ', "-DVM_PADZ=1"),' + NL, "bf2 defs")
    return s


def gates(path, s):
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check("\r" not in s, "CR in output")
    n = len(s.encode("ascii"))
    check(n < LIMIT or "_ab" in path.name or "lab" in path.name, f"size {n} B >= {LIMIT} (lab files may exceed the cap; a ship form may not)")
    path.write_bytes(s.encode("ascii"))
    py_compile.compile(str(path), doraise=True)
    return n


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_5_sk2.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_padz_ab.py")
    src = base.read_bytes().decode("ascii")
    if "ship" in FLAGS:
        new = apply_ship(src, ARGS.get("hyb", "1") == "1", ARGS.get("vml", "0") == "1")
    else:
        new = apply_lab(src, base.stem)
    n = gates(out, new)
    print(f"padz_patches: {base.name} -> {out.name} ({n:,} B, {n - len(src):+,} B, {LIMIT - n:,} B under the cap; "
          f"{'ship form' if 'ship' in FLAGS else 'lab vehicle HYBP_ST 1 / 5 / 17, VMLZ_ON 1'})")


if __name__ == "__main__":
    main()

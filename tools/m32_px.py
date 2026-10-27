"""m32_px.py: PX (kernels/lab/px_moe_lab.py, the code R-m32-2 measured as px4 / px8) as the decode-layer route at bucket 32
(17-32 tokens), on top of tools/m32_patches.py's WSK edits. Called by m32_patches.apply(text, px=True).

R-m32-2 (box_out/m32b, 10 decode_layer routings per bs): px4 bs32 -6.50 us median (10/10 better, worst -0.30), px8 -6.62
(worst +3.06) -> D = 4 here (M32_PX_D). bs16 stays on vid 3 (PX -0.8..+0.2 there).

Device (inside K2A33_KERNEL_SRC only; every anchor asserted unique in its region):
  * Params + rdy / pxmk / pxd and the PX probe bits (PXB 28 = PX mode, set only by pxrun; PXW 29 / PXV 30 never set: v1
    producer, as measured); run3 sets rdy = pxmk = nullptr (shipped calls never see PX).
  * gemm36's K1 y zeroing is skipped in PX mode (PX's y is self-reset by k3_px); K1's end-of-CTA readiness signal
    (px_k1_signal) after gemm36 in PX mode only. Both read bit 28, which no shipped call sets.
  * PX's helpers / k2_px / k3_px verbatim, plus the layer's SU1 early trigger (probe bit 25) at k3_px's entry, as k3_kernel.
  * launch_px: K1 with the layer's v4.7 PDL attribute (probe bit 24, K1 after the routing tail) like launch36; K2 / K3 PDL.
Host / Python:
  * K2A33_CPP_SRC + pxrun / pxattrs; both K2A33 builds (PT1's combined module and the separate fallback) export them.
  * M32_PX / M32_PX_D (read at call time: A/B arms); _m32px_prepare (after the K2A v3.7 prepare): PX buffers of its own,
    occupancy (k2_px 2 CTAs/SM), self-test at bs 20 / 24 / 32 (K2A36_SELF_TEST_CASES) against the live vid 3 route: rel <
    2e-3 on three repeat calls each (self-reset), vs the FP32 reference < 1e-2 / worst row < 5e-2, err flags 0, finite;
    any failure prints FALLBACK and leaves vid 3. _v11_k2a33_call routes bucket 32 to PX when the self-test passed; an
    exception at call time disables PX loudly and falls back to vid 3.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kernels/lab"))
import px_moe_lab as PXL  # noqa: E402

K33 = "k33"


def _px_dev():
    head, k2part = PXL.DEV.split("// K2: dynamic finalize-free W2 queue.", 1)
    old = "  long long* mk = px_rec(P, 2);\n  pdl_wait();\n"
    if k2part.count(old) != 1:
        raise RuntimeError("m32 px: k3_px entry anchor")
    k2part = k2part.replace(old, "  long long* mk = px_rec(P, 2);\n"
                                 "  if ((P.probe >> 25) & 1) pdl_launch();   // SU1: the staged unpooler may launch now (as k3_kernel)\n"
                                 "  pdl_wait();\n")
    return head, "// K2: dynamic finalize-free W2 queue." + k2part


def _px_host():
    h = PXL.HOST.replace("PXB_HOST", str(PXL.PXB))
    old = ("  if (mask & 1) {\n    k3::k1_kernel36<NCH, NST><<<G, 320, SM, st>>>(p);\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n  }\n"
           "  cudaLaunchAttribute at[1];\n  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
           "  at[0].val.programmaticStreamSerializationAllowed = 1;\n  cudaLaunchConfig_t cfg = {};\n  cfg.stream = st;\n"
           "  cfg.attrs = at;\n  cfg.numAttrs = 1;\n")
    new = ("  cudaLaunchAttribute at[1];\n  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
           "  cudaLaunchConfig_t cfg = {};\n  cfg.stream = st;\n  cfg.attrs = at;\n  cfg.numAttrs = 1;\n"
           "  if (mask & 1) {   // M32: K1 as launch36 does (v4.7: PDL secondary of the routing tail when probe bit 24)\n"
           "    at[0].val.programmaticStreamSerializationAllowed = (int)((p.probe >> 24) & 1);\n"
           "    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;\n"
           "    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel36<NCH, NST>, p));\n  }\n"
           "  at[0].val.programmaticStreamSerializationAllowed = 1;\n")
    if h.count(old) != 1:
        raise RuntimeError("m32 px: launch_px K1 anchor")
    h = h.replace(old, new)
    for nm in ("void pxrun(", "std::vector<int64_t> pxattrs()", "static void launch_px("):
        if h.count(nm) != 1:
            raise RuntimeError(f"m32 px: host {nm!r}")
    return h


PX_CPP_DECL = """std::vector<int64_t> pxattrs();
void pxrun(int64_t D, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
           torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
           torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
           torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor rdy,
           torch::Tensor marks, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t probe);
"""

PY_FUNCS = r'''
def _m32px_call(st, data):
    ps = st["px"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    vid, hot = st["px_route"]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    probe = (K2A33_HINT | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0)
             | ((1 << 7) if RS1_ENABLE else 0))
    st["mod"].pxrun(M32_PX_D, 7, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], ps["h"],
                    ps["inter"], ps["y"], ps["meta"], ps["cnt"], ps["err"], out, ps["sink"], ps["ss"], ps["cnt2"],
                    ps["rdy"], ps["nomk"], _k2a36_hot_code(st["variants"][vid], hot), m, data["config"]["local_start"],
                    st["grid"][vid], probe)
    return out


def _m32px_prepare(st, w, dev):
    st["px_ok"] = False
    print(f"[m32] levers: M32_PX={M32_PX} D={M32_PX_D} M32_WSK={M32_WSK} (w4 {M32_WSK_W4})", flush=True)
    if not M32_PX:
        return
    t2 = time.time()
    try:
        r36 = st.get("r36") or {}
        if 32 not in r36 or not hasattr(st["mod"], "pxrun"):
            print(f"[m32] PX FALLBACK: no live vid 3 route at bucket 32 or no pxrun ({sorted(r36)})", flush=True)
            return
        vid = r36[32][0]
        st["px_route"] = (vid, tuple(K2A36_PLAN[32][1]))   # PX as measured (R-m32-2): shipped hot (8, 0, 12), no WSK cut
        a = list(st["mod"].pxattrs())
        if a[7] < 2 or a[11] < 1 or a[1] or a[5]:
            print(f"[m32] PX FALLBACK: attrs {a} (k2_px needs 2 CTAs/SM, no spill)", flush=True)
            return
        z = lambda t: torch.zeros_like(t)  # noqa: E731
        st["px"] = dict(h=z(st["h"]), inter=z(st["inter"]), y=z(st["y"]), meta=z(st["meta"]), cnt=z(st["cnt"]),
                        err=torch.zeros((4,), dtype=torch.int32, device=dev), sink=z(st["sink"]), ss=z(st["ss"]),
                        cnt2=z(st["cnt2"]), rdy=torch.zeros((256,), dtype=torch.int32, device=dev),
                        nomk=torch.zeros((1,), dtype=torch.int64, device=dev))
        good_all = True
        for bs, seed, dist in K2A36_SELF_TEST_CASES:
            if _k2a_bucket_of(bs) != 32:
                continue
            d = _v14_self_test_inputs(bs, seed, dev, dist)
            data = {"hidden_states": d["hidden_states"], "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
                    "config": {"local_start": 0}}
            ref = _k2a_ref_fp32(d, w)
            st["err"].zero_()
            base = _k2a36_call(st, data, vid, r36[32][1]).clone()
            torch.cuda.synchronize()
            _k2a33_reset(st)
            for k in range(3):
                got = _m32px_call(st, data)
                torch.cuda.synchronize()
                e_v, _ = _k2a_rel_err(got, base)
                tot, row = _k2a_rel_err(got, ref)
                errf = int(st["px"]["err"].abs().sum()) + int(st["err"][0])
                good = (e_v < 2.0e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf
                        and bool(torch.isfinite(got.float()).all()))
                print(f"[m32] PX self-test bs={bs} {dist} call {k}: rel vs vid {vid} {e_v:.2e}, vs FP32 {tot:.3e} worst row "
                      f"{row:.3e} err={errf} -> {'ok' if good else 'FAIL'}", flush=True)
                good_all &= good
            st["err"].zero_()
            if not good_all:
                break
        st["px_ok"] = bool(good_all)
    except Exception as ex:
        print(f"[m32] PX FALLBACK: prepare raised {ex!r}", flush=True)
        st["px_ok"] = False
    print(f"[m32] PX route at bucket 32: {'LIVE' if st['px_ok'] else 'FALLBACK to vid 3'} ({time.time() - t2:.1f}s)",
          flush=True)


'''

EDITS = [
    ("px params", "  int M, lo, probe;\n};",
     "  int M, lo, probe;\n  int* rdy; long long* pxmk; int pxd;   // M32 PX\n};\n"
     f"constexpr int PXB = {PXL.PXB};   // M32 PX probe bits (PXW / PXV never set: the v1 producer, as measured)\n"
     f"constexpr int PXW = {PXL.PXW};\nconstexpr int PXV = {PXL.PXV};\n", K33),
    ("px run3 fields", "p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;",
     "p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;\n  p.rdy = nullptr; p.pxmk = nullptr; p.pxd = 4;", K33),
    ("px K1 y zeroing", "    const size_t n4 = (size_t)S.R * KDIM / 4;\n",
     "    const size_t n4 = ((P.probe >> PXB) & 1) ? (size_t)0 : (size_t)S.R * KDIM / 4;   // M32 PX: y is self-reset by k3_px\n",
     ("__device__ void gemm36(", "k1_kernel36(const Params P) {")),
    ("px K1 hook", "  gemm36<NCH, NST, 1>(P, S, s_list, smem, bars, false, s_pr);\n}\n", PXL.K1_HOOK, K33),
    ("px device helpers",
     "// K1 / K2 of the HK2 variants (cold NC1, hot up to 8*NCH tokens per pass, NST ring stages); 2 CTAs/SM.\n",
     "@PXHEAD@// K1 / K2 of the HK2 variants (cold NC1, hot up to 8*NCH tokens per pass, NST ring stages); 2 CTAs/SM.\n", K33),
    ("px kernels", "}  // namespace k3\n", "@PXK2@}  // namespace k3\n", K33),
    ("px host", "void run3(int64_t vid", "@PXHOST@\nvoid run3(int64_t vid", K33),
    ("px cpp", 'std::vector<int64_t> attrs3v(int64_t vid);\n"""\n\ndef _k2a33_wg_specs():',
     'std::vector<int64_t> attrs3v(int64_t vid);\n' + PX_CPP_DECL + '"""\n\ndef _k2a33_wg_specs():', None),
    ("px combined build exports", '"r1b_gemv", "r1b_reduce",\n                       "v8_tail", "v8_router32"],',
     '"r1b_gemv", "r1b_reduce",\n                       "v8_tail", "v8_router32", "pxrun", "pxattrs"],', None),
    ("px separate build exports", 'cpp_sources=K2A33_CPP_SRC, cuda_sources=src, functions=["run3", "attrs3v"],',
     'cpp_sources=K2A33_CPP_SRC, cuda_sources=src, functions=["run3", "attrs3v", "pxrun", "pxattrs"],', None),
    ("px globals", "M32_WSK_W4 = 8\n",
     "M32_WSK_W4 = 8\nM32_PX = True   # M32 (R-m32-2): bucket 32 = PX (finalize-free W2 queue under K1's tail), self-tested\n"
     "M32_PX_D = 4\n", None),
    ("px python", "def _v17_k2a37_prepare(st, w, dev):\n", PY_FUNCS.lstrip("\n") + "def _v17_k2a37_prepare(st, w, dev):\n", None),
    ("px prepare hook", "    _v17_k2a37_prepare(st, w, dev)\n", "    _v17_k2a37_prepare(st, w, dev)\n    _m32px_prepare(st, w, dev)\n", None),
    ("px route",
     '    rep = _k2a_bucket_of(x.shape[0])\n    r37 = st["r37"].get(rep) if (K2A37_ENABLE and st.get("r37")) else None\n',
     '    rep = _k2a_bucket_of(x.shape[0])\n'
     '    if M32_PX and rep == 32 and st.get("px_ok") and x.shape[0] <= 48:\n'
     '        try:\n'
     '            return _m32px_call(st, data)\n'
     '        except Exception as ex:\n'
     '            print(f"[k3] M32 PX route raised ({ex!r}): vid 3 from here on", flush=True)\n'
     '            st["px_ok"] = False\n'
     '    r37 = st["r37"].get(rep) if (K2A37_ENABLE and st.get("r37")) else None\n', None),
]


def _region(text, region):
    if region is None:
        return 0, len(text)
    a = text.index('K2A33_KERNEL_SRC = r"""')
    b = text.index('"""', a + 30)
    if region == K33:
        return a, b
    i = text.index(region[0], a)
    j = text.index(region[1], i)
    assert j < b, region
    return i, j


def apply(text):
    head, k2 = _px_dev()
    host = _px_host()
    for name, old, new, region in EDITS:
        a, b = _region(text, region)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"m32 px patch {name!r}: anchor found {n}x in {region or 'text'} (want 1)")
        new = new.replace("@PXHEAD@", head).replace("@PXK2@", k2).replace("@PXHOST@", host)
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text

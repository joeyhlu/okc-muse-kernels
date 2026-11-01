"""ZW: exact-anchor, in-file WST attention bank. CPU-only generation; never launches jobs.

python -B tools/zw_patches.py
python -B tools/local/zw_compile.py --cpu-only --stdout
The output is a LAB vehicle; its size is reported, never silently treated as submit-ready.
"""
import argparse
import ast
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz_rt.py"
DONOR = ROOT / "vendor_ext/keell0renz/muse_attention_decode.py"
OUT = ROOT / "kernels/lab/k3_zw_ab.py"


def sub(text, old, new, name):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"ZW anchor {name}: expected 1, got {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def literal(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


# No hot-loop edits. PDL waits precede every global read; each thread signals once
# on its exit path (idle CTA, producer completion, or consumer pre-fold).
CUDA_EDITS = [
    ("float rcp_k = 1.f, float rcp_v = 1.f, float eps = 1e-5f)\n{",
     "float rcp_k = 1.f, float rcp_v = 1.f, float eps = 1e-5f, int zw_pdl = 0)\n{\n"
     "    // ZW: acquire the layer qkv projection before any input read.\n"
     '    if (zw_pdl) asm volatile("griddepcontrol.wait;" ::: "memory");', "kernel PDL acquire"),
    ("    if (c >= Ge) return;", "    if (c >= Ge) { zw_signal(zw_pdl); return; }", "idle CTA signal"),
    ("    if (ntl == 0) return;", "    if (ntl == 0) { zw_signal(zw_pdl); return; }", "setup-only CTA signal"),
    ("        return;\n    }\n\n    // ---------- consumer warpgroups",
     "        zw_signal(zw_pdl);\n        return;\n    }\n\n    // ---------- consumer warpgroups", "producer signal"),
    ("    const float vsc = v_scale[0];\n#pragma unroll 1\n    for (int fb = b0;",
     "    zw_signal(zw_pdl); // ZW: successor may prefetch; its wait still joins all folds.\n"
     "    const float vsc = v_scale[0];\n#pragma unroll 1\n    for (int fb = b0;", "consumer signal"),
    ("__nv_bfloat16* out, int* tick, const float* vs, const unsigned char* q8 = nullptr)",
     "__nv_bfloat16* out, int* tick, const float* vs, int zw_pdl, const unsigned char* q8 = nullptr)", "launcher argument"),
    ("    cudaLaunchAttribute attr[1];", "    cudaLaunchAttribute attr[2];", "launch attribute capacity"),
    ("    cfg.numAttrs = 1;\n    cudaError_t e = cudaLaunchKernelEx",
     "    cfg.numAttrs = 1;\n    if (zw_pdl) {\n"
     "        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
     "        attr[1].val.programmaticStreamSerializationAllowed = 1;\n"
     "        cfg.numAttrs = 2;\n    }\n    cudaError_t e = cudaLaunchKernelEx", "PDL launch attribute"),
    ("                                       rcp_k, rcp_v, eps);",
     "                                       rcp_k, rcp_v, eps, zw_pdl);", "kernel argument"),
    ("torch::Tensor out, torch::Tensor tickets, torch::Tensor v_scale) {",
     "torch::Tensor out, torch::Tensor tickets, torch::Tensor v_scale, int64_t zw_pdl) {", "binding argument"),
    ("tickets.data_ptr<int>(), v_scale.data_ptr<float>());",
     "tickets.data_ptr<int>(), v_scale.data_ptr<float>(), (int)zw_pdl);", "launcher call"),
    ("#define DINL __device__ __forceinline__\n",
     '#define DINL __device__ __forceinline__\nDINL void zw_signal(int pdl) {\n'
     '    if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n}\n', "PDL helper"),
    ("        cudaFuncSetAttribute(f0, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);",
     "        cudaError_t ae = cudaFuncSetAttribute(f0, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n"
     "        TORCH_CHECK(ae == cudaSuccess, cudaGetErrorString(ae));", "attribute error gate"),
]

GLUE = '''
import math
# ZW: ZhimaoL/sglang okc/x-vfgpdl okc_patch/_cuda_src_wst.py (Apache-2.0).
# Via keell0renz/muse_attention_decode.py (Apache-2.0); donor header retained in ZW_CUDA_SRC.
_ZW_NOTICE = """
[ZW] ZhimaoL/sglang okc/x-vfgpdl okc_patch/_cuda_src_wst.py (Apache-2.0),
via keell0renz/muse_attention_decode.py (Apache-2.0), ZH_WST_CUDA / ZH_WST_CPP.
Donor attribution: # + ZhimaoL/sglang okc/x-vfgpdl okc_patch/_cuda_src_wst.py, _cuda_src_vfg.py (Apache-2.0)
Donor attribution: # Based on github.com/vjeux/okc-2026 muse_attention_decode.py (Apache-2.0). Copied for the OKC starting point.
Modified by team joeyhlu: layer qkv adapter, B=32 scratch, PDL entry/exit signaling,
checked launch attribute, prepare validation and graph-replay gates. The tile loops,
folds and rolling ticket protocol are unchanged. Apache-2.0 text is reproduced above.
"""
THIRD_PARTY_NOTICES += _ZW_NOTICE
ZW_SEL = frozenset()  # live A/B switch: {}, {16}, {32}, {16,32}; select before graph capture
ZW_FLAGS = ("-O3", "--use_fast_math", "-gencode=arch=compute_90a,code=sm_90a", "-DOKC_WST_ACC32=1", "-DZH_SCH=1")
# Compare each implementation to the SAME independent reference, as _hya_self_test
# does. HYA is itself approximate (FP8 Q/P); distance to HYA is diagnostic only.
ZW_REF_FACTOR = 1.25
ZW_REF_SLACK = 5e-3
ZW_KV_TOL = 2e-3
_ZW = {"mod": None, "attempted": False, "err": None, "banks": {}, "validated": set(), "checks": []}

def _zw_build():
    _pt1_wait("zw")
    if _ZW["attempted"]:
        return _ZW["mod"]
    _ZW["attempted"] = True
    try:
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1((ZW_CPP_SRC + ZW_CUDA_SRC + repr(ZW_FLAGS)).encode()).hexdigest()[:12]
        _ZW["mod"] = load_inline(name="zw_wst_" + tag, cpp_sources=[ZW_CPP_SRC], cuda_sources=[ZW_CUDA_SRC],
                                 functions=["wst_attn_bal"], extra_cuda_cflags=list(ZW_FLAGS), verbose=False)
    except Exception as ex:
        _ZW["err"] = repr(ex)[:1000]
    return _ZW["mod"]

def _zw_bank(dev):
    key = str(dev)
    if key not in _ZW["banks"]:
        sms = _hya_sms(dev)
        if not 1 <= sms <= 255:
            raise ValueError("ZW rolling ticket requires fewer than 256 pieces/request")
        n = (sms + 32) * 2 * 64
        _ZW["banks"][key] = {
            "sms": sms,
            "ws": (torch.empty((n, 64), dtype=torch.float32, device=dev),
                   torch.empty((n,), dtype=torch.float32, device=dev), torch.empty((n,), dtype=torch.float32, device=dev)),
            "tickets": {B: torch.zeros((2 * B,), dtype=torch.int32, device=dev) for B in (16, 32)},
            "out": {B: torch.empty((B, 8, 1024), dtype=torch.bfloat16, device=dev) for B in (16, 32)}}
    return _ZW["banks"][key]

def _zw_wanted(qkv, data, cfg, state, first):
    # Use exactly the base HYA layout predicate, then restrict the donor's geometry.
    B = cfg["batch"]
    return (B in ZW_SEL and B in (16, 32) and state.get("zw", False) and _ZW["err"] is None
            and _ZW["mod"] is not None and str(qkv.device) in _ZW["validated"]
            and cfg["q_len"] == 8 and cfg["page_size"] == 1 and not cfg["window"] and not cfg["rope_theta"]
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2 and cfg["head_dim"] == 64
            and cfg["kv_scale"] == 1.0 and _hya_wanted(qkv, data, cfg, state, first)
            and data["k_scale"].numel() == 1 and data["v_scale"].numel() == 1
            and all(data[k].device == qkv.device for k in ("k_cache", "v_cache", "page_table", "seq_lens", "cache_loc", "k_scale", "v_scale")))

def _zw_launch(qkv, data, cfg, pdl):
    # NO allocations/builds/resets here: the bank is constructed only in prepare.
    B = cfg["batch"]
    bank = _ZW["banks"][str(qkv.device)]
    acc, m_, l_ = bank["ws"]
    out, pt = bank["out"][B], data["page_table"]
    _ZW["mod"].wst_attn_bal(qkv, data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), pt,
                            data["seq_lens"], data["cache_loc"], acc, m_, l_, data["k_scale"],
                            float(cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634),
                            B, 8, pt.stride(0), bank["sms"], 0, 3, 1.0, 1.0, 1e-5,
                            out.view(B * 8, -1), bank["tickets"][B], data["v_scale"], int(bool(pdl)))
    return out

def _zw_check(d, cfg, qkv, ref, fp32, candidate, before, label):
    eo = _hya_rel(candidate, ref)
    eh, ef = _hya_rel(ref, fp32), _hya_rel(candidate, fp32)
    limit = ZW_REF_FACTOR * eh + ZW_REF_SLACK
    ek = [_hya_rel(d[name].view(torch.uint8).view(-1, 128)[d["cache_loc"]].view(FP8),
                   d["zw_ref_" + name]) for name in ("k_cache", "v_cache")]
    ekv = max(ek)
    tick = _ZW["banks"][str(qkv.device)]["tickets"][cfg["batch"]]
    # Evaluate independently: the old `or eo > .02 or ...` never inspected tickets
    # on the failing pod case, despite printing '/ ticket residue' unconditionally.
    residue = bool(((tick & 255) != 0).any())
    expected = (before.to(torch.int64) + 256) & 0xffffffff
    step_bad = bool(((tick.to(torch.int64) & 0xffffffff) != expected).any())
    finite = (all(bool(torch.isfinite(x).all()) for x in (candidate, ref, fp32))
              and all(math.isfinite(e) for e in (eo, eh, ef, *ek)))
    reasons = []
    if not finite:
        reasons.append("nonfinite")
    if not math.isfinite(ef) or not math.isfinite(limit) or ef > limit:
        reasons.append("output_vs_fp32")
    if not all(math.isfinite(e) for e in ek) or ekv > ZW_KV_TOL:
        reasons.append("new_kv")
    if residue:
        reasons.append("ticket_residue")
    if step_bad:
        reasons.append("ticket_step")
    row = {"B": cfg["batch"], "case": label, "out_relL2": eo, "kv_relL2": ekv,
           "hya_fp32_relL2": eh, "out_fp32_relL2": ef, "fp32_limit": limit,
           "ticket_residue": residue, "ticket_step_bad": step_bad, "reasons": reasons}
    _ZW["last_check"] = row  # preserve failure diagnostics even if self_test raises
    if reasons:
        raise RuntimeError(f"ZW B{cfg['batch']} {label} failed={','.join(reasons)} "
                           f"relL2 vsHYA={eo:.5g} vsFP32={ef:.5g} HYA/FP32={eh:.5g} limit={limit:.5g} "
                           f"KV={ekv:.5g} ticket_residue={residue} ticket_step_bad={step_bad}")
    return row

def _zw_self_test(dev, t0):
    checks = []
    for B in (16, 32):
        for ctx in (129, 4097, 80000):
            if time.time() - t0 > 290.0:
                raise RuntimeError("ZW validation exceeded prepare guard")
            d = _k1fa4_case(B, ctx, 8, 0, 7300 + B + ctx, True, dev)
            qkv = d["qkv"].view(B * 8, -1).contiguous()
            cfg = dict(d["config"], tokens=B * 8)
            # The real base reference does FP32 attention with its BF16 scaled Q
            # and FP8 new K/V. It neither reads our output nor writes the cache.
            fp32 = _k1fa4_reference(dict(d, qkv=qkv))
            ref = _hya_launch(qkv, d, cfg, True).clone()
            for name in ("k_cache", "v_cache"):
                d["zw_ref_" + name] = d[name].view(torch.uint8).view(-1, 128)[d["cache_loc"]].clone().view(FP8)
                d[name].view(torch.uint8).view(-1, 128)[d["cache_loc"]] = 0  # catch a missing cache write
            bank = _ZW["banks"][str(dev)]
            bank["tickets"][B].fill_(-256)  # exercise unsigned 32-bit rollover on the first launch
            for pdl in (False, True):
                before = bank["tickets"][B].clone()
                bank["out"][B].fill_(float("nan"))
                out = _zw_launch(qkv, d, cfg, pdl)
                torch.cuda.synchronize(dev)
                checks.append(_zw_check(d, cfg, qkv, ref, fp32, out, before, f"ctx{ctx}/pdl{int(pdl)}"))
            # Capture the actual no-allocation adapter, then change qkv between replays.
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                out = _zw_launch(qkv, d, cfg, True)
            for replay in range(3):
                if time.time() - t0 > 290.0:
                    raise RuntimeError("ZW validation exceeded prepare guard")
                qkv.mul_(-1.0)
                fp32 = _k1fa4_reference(dict(d, qkv=qkv))
                ref = _hya_launch(qkv, d, cfg, True).clone()
                for name in ("k_cache", "v_cache"):
                    d["zw_ref_" + name] = d[name].view(torch.uint8).view(-1, 128)[d["cache_loc"]].clone().view(FP8)
                    d[name].view(torch.uint8).view(-1, 128)[d["cache_loc"]] = 0
                before = bank["tickets"][B].clone()
                bank["out"][B].fill_(float("nan"))
                gr.replay()
                torch.cuda.synchronize(dev)
                checks.append(_zw_check(d, cfg, qkv, ref, fp32, out, before, f"ctx{ctx}/replay{replay}"))
            del gr, d, qkv, ref, fp32, out, before
    return checks

def _zw_prepare(state, t0):
    state["zw"] = False
    try:
        if not state.get("hya"):
            raise RuntimeError("HYA validation baseline unavailable")
        dev = state["router"].device  # exact accessor from _hya_prepare
        if _zw_build() is None:
            raise RuntimeError(_ZW["err"] or "WST build unavailable")
        if _ZW["err"] is not None:
            raise RuntimeError(_ZW["err"])
        _zw_bank(dev)
        if str(dev) not in _ZW["validated"]:
            checks = _zw_self_test(dev, t0)
            _ZW["checks"].extend(checks)
            _ZW["validated"].add(str(dev))
        state["zw"] = True
        eo = max(x["out_relL2"] for x in _ZW["checks"])
        ef = max(x["out_fp32_relL2"] for x in _ZW["checks"])
        eh = max(x["hya_fp32_relL2"] for x in _ZW["checks"])
        ek = max(x["kv_relL2"] for x in _ZW["checks"])
        print(f"[zw] READY B16/B32 relL2 vsHYA={eo:.4g} vsFP32={ef:.4g} HYA/FP32={eh:.4g} KV={ek:.4g}; "
              f"per-case FP32/PDL/replay/ticket+256 PASS; selected={sorted(ZW_SEL)}", flush=True)
    except Exception as ex:
        _ZW["err"] = repr(ex)[:1000]
        _ZW["validated"].clear()
        print(f"[zw] OFF: {_ZW['err']}", flush=True)
'''

BUILD_ANCHOR = '    if HYA_ENABLE:\n        jobs.append(("hya", _hya_build))\n'
PREP_ANCHOR = '    _gft_prepare(state, t0)\n    builds = '
TRY_ANCHOR = 'def _hya_try(qkv, data, cfg, state, first):\n'
INSERT_ANCHOR = 'def _r2_page1(data, cfg, state):\n'
TRY_BODY = '''    if _zw_wanted(qkv, data, cfg, state, first):
        try:
            return _zw_launch(qkv, data, cfg, HYA_PDL_T.get(cfg["tokens"], HYA_PDL)).view(cfg["tokens"], -1)
        except Exception as ex:
            state["zw"] = False
            _ZW["err"] = repr(ex)[:1000]
            _ZW["validated"].clear()
            print(f"[zw] OFF at call: {_ZW['err']}", flush=True)
'''


def generate(base=BASE):
    source = pathlib.Path(base).read_text(encoding="utf-8")
    donor = DONOR.read_text(encoding="utf-8")
    cuda, cpp = literal(donor, "ZH_WST_CUDA"), literal(donor, "ZH_WST_CPP")
    for old, new, label in CUDA_EDITS:
        cuda = sub(cuda, old, new, label)
    cpp = sub(cpp, "torch::Tensor tickets, torch::Tensor v_scale);",
              "torch::Tensor tickets, torch::Tensor v_scale, int64_t zw_pdl);", "C++ declaration")
    block = ('# ZW BEGIN; donor SHA256 ' + hashlib.sha256(donor.encode()).hexdigest() + '\n'
             + 'ZW_CUDA_SRC = r"""' + cuda + '"""\nZW_CPP_SRC = r"""' + cpp + '"""\n' + GLUE + '\n# ZW END\n')
    result = sub(source, BUILD_ANCHOR, BUILD_ANCHOR + '    jobs.append(("zw", _zw_build))\n', "build bank")
    result = sub(result, PREP_ANCHOR, '    _gft_prepare(state, t0)\n    _zw_prepare(state, t0)\n    builds = ', "prepare bank")
    result = sub(result, TRY_ANCHOR, TRY_ANCHOR + TRY_BODY, "attention arm")
    result = sub(result, INSERT_ANCHOR, block + INSERT_ANCHOR, "vendor block")
    compile(result, str(OUT), "exec")
    # Strip exactly our additions: baseline text must round-trip byte-for-byte after newline normalization.
    undo = result.replace(block, "", 1).replace(TRY_BODY, "", 1)
    undo = undo.replace('    jobs.append(("zw", _zw_build))\n', "", 1).replace('    _zw_prepare(state, t0)\n', "", 1)
    assert undo == source, "unscoped baseline changes"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=pathlib.Path, default=BASE)
    parser.add_argument("--check", action="store_true", help="verify existing generated output, without writing")
    args = parser.parse_args()
    result = generate(args.base)
    if args.check:
        assert OUT.read_text(encoding="utf-8") == result, "generated output is stale"
    else:
        OUT.write_bytes(result.encode("utf-8"))
    print(f"ZW generator PASS: {len(result.encode('utf-8'))} bytes; default ZW_SEL empty; only four exact-anchor base additions")


if __name__ == "__main__":
    main()

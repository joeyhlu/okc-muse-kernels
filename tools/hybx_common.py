"""hybx_common.py: shared helper text for extra HYB (alygg77 WGS engine, A32) modules, inserted ONCE by tools/hybtab_patches.py
(lab table module "l") and tools/wgs32_patches.py (FT_T = 32 module "32"). Our glue, same style as _hyb_call / _hyb_prepare.

An extra module = HYB's CUDA source (_HYB_MOE_CUDA) after a text transform, built with HYB_FLAGS by load_inline (own PT1 job
"hyb<name>"), its OWN scratch (_hyb_scratch: counters / fixup / xA-xB never shared with the ship HYB module, R-v32-7 lesson) and
the ship HYB weight copies (read-only; FT_T does not change the copy layout). Prepare self-test per case: rel L2 vs the FP32
reference < 1e-2, worst row < 5e-2, vs the ship HYB output < 1e-2, repeat < 1e-3, all finite; guards intact. Any failure ->
the module stays OFF and its route falls through to the existing engines.
"""

ANCHOR = "def _hyb_prepare(st, w, dev):\n"

HELPER = r'''_HYBX = {}
def _hybx_inputs(bs, seed, dev, dist):
    if not dist.startswith("conc"):
        return _k2a_self_test_inputs(bs, seed, dev, dist)
    ne = int(dist[4:])
    g = torch.Generator(device="cpu").manual_seed(seed)
    loc = torch.randperm(K2A_E_LOC, generator=g)[:ne]
    ids = torch.stack([torch.randperm(256 - K2A_E_LOC, generator=g)[:8] + K2A_E_LOC for _ in range(bs)])
    for k in range(ne):
        ids[torch.rand(bs, generator=g) < (1.0 if k == 0 else 0.6), k] = loc[k]
    tw = torch.rand((bs, 8), generator=g) + 0.05
    return {"hidden_states": torch.randn((bs, 4096), generator=g).to(torch.bfloat16).to(dev),
            "topk_ids": torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1)).to(torch.int32).to(dev),
            "topk_weights": (tw / tw.sum(-1, keepdim=True)).float().to(dev), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
def _hybx_build(name):
    from torch.utils.cpp_extension import load_inline
    X = _HYBX[name]
    tag = hashlib.sha1((_HYB_MOE_CPP + X["src"] + " ".join(HYB_FLAGS)).encode()).hexdigest()[:12]
    t0 = time.time()
    X["mod"] = load_inline(name=f"hyb{name}_moe_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=X["src"],
                           functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=list(HYB_FLAGS), verbose=False)
    X["build_s"] = time.time() - t0
    return X["mod"]
def _hybx_call(name, x, ids, wts, w=None, dbg=None):
    X = _HYBX[name]
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    if X["pdl"] != pdl:
        X["mod"].okc_set_pdl(pdl)
        X["pdl"] = pdl
    X["mod"].okc_run(X["w"] if w is None else w, X["sc"] if dbg is None else X["sc"] + [dbg], x, ids, wts, out, 0, 32,
                     _HYB["nsm"])
    return out
def _hybx_prepare(name, w, dev, tests, wmod=None):
    X = _HYBX[name]
    X["ok"], X["pdl"] = False, -1
    if not (HYB_ENABLE and _HYB["ok"]):
        print(f"[hyb{name}] OFF: the ship HYB module is not live (no weight copies)", flush=True)
        return
    t1 = time.time()
    try:
        _pt1_wait("hyb" + name)
        if X.get("mod") is None:
            _hybx_build(name)
        X["w"] = list(_HYB["w"])
        if wmod is not None:
            wmod(X["w"])
        X["sc"], X["full"] = _hyb_scratch(dev)
        torch.cuda.synchronize()
        good, worst = True, 0.0
        for bs, seed, dist in tests:
            d = _hybx_inputs(bs, seed, dev, dist)
            d = {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                 "topk_weights": d["topk_weights"].float().contiguous(), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
            ref = _k2a_ref_fp32(d, w)
            o_h = _hyb_call(d["hidden_states"], d["topk_ids"], d["topk_weights"]).clone()
            prev = None
            for k in range(2):
                o = _hybx_call(name, d["hidden_states"], d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(o, ref)
                eh = _k2a_rel_err(o, o_h)[0]
                rep_d = 0.0 if prev is None else _k2a_rel_err(o, prev)[0]
                fin = bool(torch.isfinite(o.float()).all())
                ok = tot < 1e-2 and row < 5e-2 and eh < 1e-2 and rep_d < 1e-3 and fin
                worst = max(worst, tot)
                if not ok or k == 1:
                    print(f"[hyb{name}] self-test bs={bs} {dist} call {k}: vs FP32 {tot:.3e} worst row {row:.3e} vs HYB {eh:.2e} "
                          f"repeat {rep_d:.1e} finite {int(fin)} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
                prev = o.clone()
        gok = all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in X["full"])
        X["ok"] = bool(good and gok)
        print(f"[hyb{name}] {'LIVE' if X['ok'] else 'OFF'} ({len(tests)} cases, worst vs FP32 {worst:.2e}, guards {gok}, "
              f"nvcc {X.get('build_s', 0.0):.0f}s, prepare {time.time() - t1:.1f}s)", flush=True)
    except Exception as ex:
        X["ok"] = False
        print(f"[hyb{name}] prepare raised {ex!r}: route stays on the existing engines", flush=True)
'''


def ensure(s, sub):
    """insert HELPER once (before _hyb_prepare); idempotent so the two patchers stack in either order"""
    if "\ndef _hybx_build(name):\n" in s:
        return s
    return sub(s, ANCHOR, HELPER + ANCHOR, "hybx helper")

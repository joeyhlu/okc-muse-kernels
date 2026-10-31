"""hya_src.py: the Python host block that tools/hya_patches.py inserts into a decode-layer file (R-hya, team joeyhlu).

Text between the BEGIN / END marker lines is pasted verbatim (before `def _r2_page1(`). It runs in the layer file's
namespace: torch, time, hashlib, FP8, ROPE_THETA, _pt1_wait, _PT1, _k3_disable, _k1fa4_case, _k1fa4_reference,
_k1fa4_r1_path, THIRD_PARTY_NOTICES, HYA_CPP_SRC / HYA_CUDA_SRC and the HYA_* route constants come from there. This module is
not imported by anything; it exists so the block can be read, linted and diffed as plain Python.

Provenance: _hya_plan and _hya_launch are derived from _plan / _launch in vjeux's muse_attention_decode.py
(https://github.com/vjeux/okc-2026, commit f35a41ee1c, Apache License 2.0; copy and LICENSE in
tools/vendor/vjeux_okc2026_f35a41ee/). Modifications are marked "HYA M<n>" and listed in _HYA_NOTICE. The rest of the
block (route, workspace handling, self-test, prepare) is team joeyhlu's own code.
"""
# ---- HYA BLOCK BEGIN ----
# HYA (R-hya, team joeyhlu): vjeux's public attention decode kernel (HYA_CUDA_SRC, Apache-2.0, see THIRD_PARTY_NOTICES) for
# the shapes in HYA_ROUTE (first, before AX1 / KS1) and as the general fallback after AX1 / KS1 (HYA_DEFAULT).
# kv_scale != 1 and unsupported layouts take the r1 path (SGLang FA3). _hya_plan / _hya_launch: derived from vjeux's
# _plan / _launch (github.com/vjeux/okc-2026 f35a41ee, Apache-2.0), modified as HYA M2 / M3 in _HYA_NOTICE.
_HYA = {"mod": None, "ws": {}, "inv": {}, "sms": {}, "build_s": 0.0, "seen": set(), "mk": False}


def _hya_build():
    _pt1_wait("hya")
    if _HYA["mod"] is not None:
        return _HYA["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((HYA_CPP_SRC + HYA_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    # HYA M3: own extension name (vjeux: okc_attn_v298_90a); flags as vjeux's _build
    _HYA["mod"] = load_inline(name=f"hya_attn_v298_{tag}", cpp_sources=HYA_CPP_SRC, cuda_sources=HYA_CUDA_SRC,
                              functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a",
                                                                    "--expt-relaxed-constexpr"], verbose=False)
    _HYA["build_s"] = time.time() - t0
    return _HYA["mod"]


def _hya_ws(n_rows, dev):
    # vjeux's persistent FP32 accumulators (the kernel's last CTA zeroes them again). HYA M3: made ONCE, in prepare, at the
    # largest size any routed shape can ask for (HYA_WS_ROWS: B 512 x q_len 8 x 16 rows at one copy; B 1 plans use <= 8
    # copies x 128 rows), and never reallocated or freed afterwards: a CUDA graph captured earlier keeps the raw
    # pointers, so a grown-and-freed buffer would be a use-after-free on its replays. A call that needs more rows (or
    # comes before prepare made the buffers) declines: the r1 path takes it.
    ws = _HYA["ws"].get(str(dev))
    if ws is None and _HYA["mk"]:
        n = max(n_rows, HYA_WS_ROWS)
        ws = _HYA["ws"][str(dev)] = (torch.zeros(n * 64, dtype=torch.float32, device=dev),
                                     torch.zeros(n, dtype=torch.float32, device=dev),
                                     torch.zeros(n, dtype=torch.int32, device=dev))
    if ws is None or ws[1].numel() < n_rows:
        return None
    return ws


def _hya_inv_freq(theta, dev):
    # RoPE tables: made in prepare only (thetas 0 and ROPE_THETA); any other theta at call time declines (r1 path)
    key = (float(theta), str(dev))
    t = _HYA["inv"].get(key)
    if t is None and _HYA["mk"]:
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=dev).float() / 64))
        else:
            t = torch.zeros(32, device=dev)
        t = _HYA["inv"][key] = t.float().contiguous()
    return t


def _hya_sms(dev):
    key = str(dev)
    if key not in _HYA["sms"]:
        _HYA["sms"][key] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _HYA["sms"][key]


def _hya_plan(B, pairs, L, window, max_pages, P, sms):
    # derived from vjeux's _plan (f35a41ee, Apache-2.0); HYA M2: SM count passed in, _splits / _copies / _headroom
    # inlined; the plan itself is unchanged
    target = 2 * sms
    if not window and 1 < B <= 512:
        return dict(n_splits=max(1, target // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B * pairs
    if not window:
        if L == 8:
            n = max(1, min(tiles, ((target * 21) // 22) // reqs))
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 8)))
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)), coop=0)
    if L == 1 and reqs < 24:
        return dict(n_splits=max(1, min(tiles, target // reqs)), flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16:
        return dict(n_splits=max(1, min(tiles, target // reqs)), flat=0, cs=0, ncl=1, nc=1, coop=1)
    if L == 8 and reqs >= 16:
        return dict(n_splits=max(1, min(tiles, (target if reqs >= 24 else sms) // reqs)), flat=0, cs=0, ncl=1, nc=1, coop=1)
    if reqs >= 24:
        return dict(n_splits=max(1, min(tiles, target // reqs)), flat=0, cs=0, ncl=1, nc=1)
    cs = 16
    limit = 2 * sms if L == 1 else sms
    if L == 8 and target == sms:
        limit = (sms * 15) // 16
    while cs > 4 and reqs * cs > limit:
        cs //= 2
    ncl = 1
    while cs * ncl > tiles and cs > 4:
        cs //= 2
    if cs * ncl > tiles or cs < 4:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)))
    return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=max(1, min(ncl, 4)))


def _hya_launch(qkv, data, cfg, pdl, override=None):
    # derived from vjeux's _launch (f35a41ee, Apache-2.0); HYA M3: qkv = our qkv projection output (passed separately),
    # PDL from the caller (HYA_PDL), per-shape overrides from HYA_TUNED (= vjeux's _TUNED), workspace / RoPE table from
    # prepare (None inside a capture -> the caller falls back); plan, coop and mode bits unchanged
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"]
    dev = qkv.device
    pairs = hkv // 2
    window = cfg["window"]
    pt = data["page_table"]
    sms = _hya_sms(dev)
    plan = _hya_plan(B, pairs, L, window, pt.shape[1], P, sms)
    if override is None:
        override = HYA_TUNED.get((B, L, bool(window), pairs))
    if override:
        plan = dict(plan, **override)
        if "cs" in override and override["cs"] > 1:
            plan["flat"] = 0
            if "ncl" not in override:
                plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
            plan["n_splits"] = plan["cs"] * plan["ncl"]
            if "nc" not in override:
                plan["nc"] = max(1, min(plan["ncl"], 4))
        elif "n_splits" in override and "nc" not in override:
            plan["nc"] = 1 if plan["flat"] else max(1, min(plan["n_splits"], 4))
    n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
    if flat:
        coop = int(plan.get("coop", int(L == 8 and n_splits * pairs <= 2 * sms)))
    else:
        coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else int(n_splits * B * pairs <= 2 * sms)))
    tma = int(plan.get("tma", 0)) | (coop << 1) | (int(bool(pdl)) << 16)
    if flat or cs <= 1:
        cs, ncl = 0, 1
    ws = _hya_ws(B * pairs * nc * L * 16, dev)
    inv = _hya_inv_freq(cfg["rope_theta"], dev)
    if ws is None or inv is None:
        return None
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    _HYA["mod"].run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], pt, data["k_scale"],
                    data["v_scale"], inv, out, ws[0], ws[1], ws[2], B, L, hq, hkv, P, window,
                    1 if cfg["rope_theta"] else 0, n_splits, flat, nc,
                    float(cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634), None, cs, ncl, tma)
    return out


def _hya_key(cfg):
    return ("sliding" if cfg["window"] else "full", cfg["batch"], cfg["q_len"])


def _hya_wanted(qkv, data, cfg, state, first):
    # first = the explicit HYA_ROUTE pass before AX1 / KS1; else the fallback pass after them (HYA_DEFAULT)
    if not (HYA_ENABLE and state.get("hya") and _HYA["mod"] is not None):
        return False
    r = HYA_ROUTE.get(_hya_key(cfg))
    if not (r is True if first else (r if r is not None else HYA_DEFAULT)):
        return False
    B, L, hq, hkv, P = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"]
    kc, vc, pt, sl, loc = data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], data["cache_loc"]
    return (cfg["head_dim"] == 64 and hq == 8 * hkv and hkv >= 2 and not (hkv & (hkv - 1)) and L in (1, 8)
            and 1 <= B <= 512 and B * (hkv // 2) <= 4096 and P > 0 and not (P & (P - 1)) and cfg["kv_scale"] == 1.0
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.is_contiguous() and qkv.shape[0] == B * L
            and qkv.shape[1] == (2 * hq + 2 * hkv) * 64 and qkv.numel() < (1 << 31) and B * L * hq * 64 < (1 << 31)
            and kc.dtype == FP8 and vc.dtype == FP8 and kc.is_contiguous() and vc.is_contiguous()
            and kc.numel() == vc.numel() and kc.numel() // (hkv * 64) < (1 << 31)
            and pt.dtype == torch.int32 and pt.dim() == 2 and pt.is_contiguous() and pt.shape[0] >= B
            and pt.numel() < (1 << 31) and sl.dtype == torch.int32 and sl.is_contiguous() and sl.numel() == B
            and loc.dtype == torch.int64 and loc.is_contiguous() and loc.numel() == B * L
            and data["k_scale"].dtype == torch.float32 and data["v_scale"].dtype == torch.float32)


def _hya_try(qkv, data, cfg, state, first):
    if not _hya_wanted(qkv, data, cfg, state, first):
        return None
    try:
        o = _hya_launch(qkv, data, cfg, HYA_PDL_T.get(cfg["tokens"], HYA_PDL))
    except Exception as ex:
        _k3_disable(state, "hya", False, "HYA", ex)
        return None
    if o is None:
        return None
    k = _hya_key(cfg)
    if k not in _HYA["seen"]:
        _HYA["seen"].add(k)
        print(f"[hya] first call {k} page {cfg['page_size']} pass {'route' if first else 'fallback'}", flush=True)
    return o.view(cfg["tokens"], -1)


def _hya_paged(d, P, seed):
    # a page-1 _k1fa4_case re-laid into page-size-P pages (same keys, permuted pages, new rows at the mapped slots)
    cfg = d["config"]
    B, L, hkv = cfg["batch"], cfg["q_len"], cfg["num_kv_heads"]
    dev = d["qkv"].device
    pt1 = d["page_table"]
    n = pt1.shape[1]
    npg = (n + P - 1) // P
    g = torch.Generator(device=dev).manual_seed(seed)
    ptp = torch.randperm(B * npg, generator=g, device=dev).to(torch.int32).view(B, npg)
    t = torch.arange(n, device=dev)
    rows = ptp.long()[:, t // P] * P + t % P
    ok = pt1 >= 0
    out = dict(d, page_table=ptp, config=dict(cfg, page_size=P))
    for name in ("k_cache", "v_cache"):
        src = d[name].view(torch.uint8).view(-1, hkv * 64)
        dst = torch.zeros((B * npg * P, hkv * 64), dtype=torch.uint8, device=dev)
        dst[rows[ok]] = src[pt1[ok].long()]
        out[name] = dst.view(FP8).view(B * npg, P, hkv, 64)
    pos = d["seq_lens"].long()[:, None] + torch.arange(L, device=dev)
    out["cache_loc"] = torch.gather(rows, 1, pos).reshape(-1).contiguous()
    return out


def _hya_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def _hya_self_test(dev):
    # per case: FP32 reference + the stock r1 path (SGLang FA3, its cache bytes) on fresh copies; HYA must be finite,
    # within 1.25 x the stock error + 5e-3 (the KS1 rule), write the same new cache bytes, agree with itself across PDL
    # on / off (the split merge uses FP32 atomics: not bitwise), and leave the workspace zero
    worst = 0.0
    for B, ctx, L, win, seed, row0, P in HYA_SELF_TEST:
        ref = _k1fa4_reference(_k1fa4_case(B, ctx, L, win, seed, row0, dev)).view(B * L, -1)
        d1 = _k1fa4_case(B, ctx, L, win, seed, row0, dev)
        o1 = _k1fa4_r1_path(d1).float().view(B * L, -1)
        b1 = [d1[c].view(torch.uint8).view(-1, 128)[d1["cache_loc"]] for c in ("k_cache", "v_cache")]
        e1 = _hya_rel(o1, ref)
        dk = _k1fa4_case(B, ctx, L, win, seed, row0, dev)
        if P > 1:
            dk = _hya_paged(dk, P, seed + 7)
        qkv = dk["qkv"].view(B * L, -1).contiguous()
        cfg = dict(dk["config"], tokens=B * L)
        oa = _hya_launch(qkv, dk, cfg, True).float().view(B * L, -1)
        ob = _hya_launch(qkv, dk, cfg, False).float().view(B * L, -1)
        torch.cuda.synchronize(dev)
        ws = _HYA["ws"][str(dev)]
        clean = not (float(ws[0].abs().sum()) or float(ws[1].abs().sum()))
        bk = [dk[c].view(torch.uint8).view(-1, 128)[dk["cache_loc"]] for c in ("k_cache", "v_cache")]
        nbad = sum(int((x != y).sum()) for x, y in zip(bk, b1))
        # new cache rows: vjeux's in-kernel QK-norm / RoPE rounds a few K bytes 1 FP8 step off the stock Triton path
        # (R-hya-1b on H200: 1.2e-2 .. 2.0e-2 rel); the judge compares dequantized rows at 8e-2 rel (CACHE_TOL): require 4e-2
        ekv = max(_hya_rel(x.view(FP8), y.view(FP8)) for x, y in zip(bk, b1))
        ea, eab = _hya_rel(oa, ref), _hya_rel(ob, oa)
        ok = (bool(torch.isfinite(oa).all()) and bool(torch.isfinite(ob).all()) and ea <= 1.25 * e1 + 5e-3
              and eab <= 2e-3 and ekv <= 4e-2 and clean)
        print(f"[hya] self-test B{B} ctx{ctx} L{L} w{win} P{P}{' row0' if row0 else ''}: {'PASS' if ok else 'FAIL'} err {ea:.3e} "
              f"(r1 {e1:.3e}) pdl on/off {eab:.1e} new K/V vs r1 {ekv:.1e} ({nbad} bytes differ) ws {'clean' if clean else 'DIRTY'}",
              flush=True)
        if not ok and HYA_SELF_TEST_STRICT:
            raise RuntimeError(f"self-test B{B} ctx{ctx} L{L} w{win} P{P}")
        worst = max(worst, ea)
        del ref, d1, o1, dk, oa, ob
    for B, ctx, L, win, seed in HYA_GRAPH_CASES:
        d = _k1fa4_case(B, ctx, L, win, seed, False, dev)
        qkv = d["qkv"].view(B * L, -1).contiguous()
        cfg = dict(d["config"], tokens=B * L)
        o1 = _hya_launch(qkv, d, cfg, True).clone()
        torch.cuda.synchronize(dev)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            og = _hya_launch(qkv, d, cfg, True)
        if og is None:
            raise RuntimeError("graph check: the launch declined inside the capture")
        gr.replay()
        gr.replay()
        torch.cuda.synchronize(dev)
        e = _hya_rel(og, o1)
        print(f"[hya] graph check B{B} ctx{ctx} L{L} w{win}: replay vs eager {e:.1e}", flush=True)
        if not e <= 2e-3 and HYA_SELF_TEST_STRICT:
            raise RuntimeError(f"graph check B{B} L{L}")
        del gr, og, o1, d
    return worst


def _hya_prepare(state, t0):
    state["hya"] = False
    print(f"[hya] levers: HYA_ENABLE={HYA_ENABLE} route {HYA_ROUTE} default {HYA_DEFAULT} pdl {HYA_PDL} {HYA_PDL_T} "
          f"tuned {HYA_TUNED}; notices: {len(THIRD_PARTY_NOTICES)} chars, HYA section {'present' if _HYA_NOTICE in THIRD_PARTY_NOTICES else 'MISSING'}", flush=True)
    if not HYA_ENABLE:
        return
    if time.time() - t0 > HYA_PREPARE_BY_S:
        print(f"[hya] skipped: prepare already took {time.time() - t0:.1f}s (> {HYA_PREPARE_BY_S}s): r1 path", flush=True)
        return
    dev = state["router"].device
    t1 = time.time()
    try:
        if _PT1["errors"].get("hya"):
            raise RuntimeError(f"build failed: {_PT1['errors']['hya']}")
        _hya_build()
        _HYA["mk"] = True
        _hya_ws(HYA_WS_ROWS, dev)
        for theta in (0.0, ROPE_THETA):
            _hya_inv_freq(theta, dev)
        _HYA["mk"] = False
        worst = _hya_self_test(dev)
        state["hya"] = True
        print(f"[hya] LIVE: route {sorted(k for k, v in HYA_ROUTE.items() if v)} + fallback {HYA_DEFAULT}, worst err {worst:.3e}, "
              f"build {_HYA['build_s']:.0f}s, self-test {time.time() - t1:.1f}s, prepare so far {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:
        _HYA["mk"] = False
        print(f"[hya] FAILED, r1 path for its shapes: {ex!r}"[:1500], flush=True)
        for t in _HYA["ws"].values():
            for x in t:
                x.zero_()
# ---- HYA BLOCK END ----

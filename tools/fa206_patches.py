"""fa206_patches.py: PR206 = the SM90 native-FP8 forward of flash-attention PR #206 (pinned 303f735573162381) as the attention
core of the FULL-attention layers at q_len 8 and batch >= 16 (verify 16 / 32 full). Exact text patches against
kernels/muse_decode_layer/k3_v34.py (tools/gen_k3_v20.py); the wrapper tools/gen_k3_fa.py writes k3_fa_<n>.py.

  core      kernels/muse_attention_decode/fa206_vendor.py (tools/gen_fa206_vendor.py): the PR's FlashAttentionForwardSm90 and the
            flash_attn.cute classes it needs that differ from the image's installed copy, as plain top-level source (p206_ names),
            method bodies unedited. Compiled ONCE in prepare for the one configuration below; P206_CONFIG is checked on the
            compiled object.
  config    E4M3 Q/K/V (FA4 v8's prologue output and the FP8 cache), tile 64 x 128 (the lab's tile_mn), one MMA warpgroup +
            a producer warpgroup that loads K / V by cp.async (page size 1) and transposes V with LDSM.T / STSM, two-level FP32
            PV accumulation, two CTAs / SM, causal full attention, SplitKV with the PR's own split count (p206_plan: its
            interface._get_fwd_config + ctas_per_sm rule with num_splits=0), FP32 partials.
  layer     FA4 v8's Triton prologue (QK-norm, E4M3 q, FP8 K/V rows into the cache) -> PR206 core -> FA4 v8's Triton combine +
            RMS-norm x sigmoid(gate) (PDL), i.e. k1fa4_run with only the core swapped (the lab arm used the PR's combine + a
            separate norm-gate kernel). Routed when the PR's plan gives >= 2 splits and 2 CTAs / SM (every full verify shape at
            batch >= 16 with a long context); anything else keeps FA4 v8.
  measured  box, in-layer paired A/B (box_out/fresh_fa4_layer/full.log, PR core + PR combine, 4 seeds): verify16 full 419.6 ->
            412.7 us (x1.016), verify32 full 626.1 -> 616.7 us (x1.013); sliding unchanged (x1.001). Operator level
            (box_out/fresh_fa4/batch.json): B16 / B32 ctx 80K full L8 115.9 -> 112.1 / 226.8 -> 212.7 us.
  prepare   _p206_prepare after DG1: JIT + self-test (B 16 / 32, ctx 1.5K..12K, a new key in cache row 0) against the FP32
            reference with FA4 v8's rule (error <= 1.25 x r1's + 5e-3 and <= 1.25 x FA4 v8's + 5e-3, finite, new cache
            bytes == r1's), then a CUDA-graph check
            (replay == eager bit for bit, and again after new inputs are copied into the captured tensors). Skipped when
            prepare is past P206_PREPARE_BY_S. Any failure (or a host exception at call time) keeps FA4 v8 everywhere.

    python tools/fa206_patches.py [k3_v34.py] [out.py]     # apply + py_compile check only
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v34.py"
VENDOR = ROOT / "kernels/muse_attention_decode/fa206_vendor.py"

GLUE = r'''

# ===================================================================================================================
# PR206 route (tools/fa206_patches.py): FULL-attention layers at q_len 8 and batch >= P206_MIN_BATCH run the vendored PR206
# native-FP8 core above between FA4 v8's prologue and its combine + RMS-norm x sigmoid(gate). Box A/B: verify16 full x1.016,
# verify32 full x1.013 (box_out/fresh_fa4_layer/full.log). Live only after _p206_prepare's self-test and graph check.
# ===================================================================================================================
P206_ENABLE = True              # master switch, read at call time
P206_MIN_BATCH = 16             # full verify at batch >= 16 (measured at 16 and 32); smaller batches keep FA4 v8
P206_TILE = (64, 128)           # (tile_m, tile_n) = the lab's tile_mn
P206_PREPARE_BY_S = 235.0       # no PR206 JIT / self-test starts after this many seconds of prepare (limit 300 s)
P206_SELF_TEST = ((16, 3000, 8, 0, 501, False), (32, 1500, 8, 0, 502, True), (16, 12000, 8, 0, 503, False))
P206_GRAPH_CASES = ((16, 2500, 8, 0, 511, 512),)   # (batch, ctx, q_len, window, seed, seed of the copied-in inputs)
# the measured configuration, checked on the compiled object (attributes set by __init__ / __call__ / kernel at trace time)
P206_CONFIG = {"native_fp8": True, "fp8_kv_dequant": False, "use_tma_KV": False, "tma_box_n": None, "intra_wg_overlap": False,
               "is_split_kv": True, "pack_gqa": True, "is_causal": True, "is_local": False, "mma_pv_is_rs": True,
               "transpose_v_in_producer": True, "dedicated_kv_load_warp": False, "use_tma_Q": True, "use_tma_O": False,
               "use_block_sparsity": False, "num_wg_mma": 1, "ctas_per_sm": 2, "tile_m": 64, "tile_n": 128, "num_stages": 2,
               "v_transpose_warps": 4, "num_threads": 256, "direct_3d_grid": False, "Q_in_regs": False, "score_mod": None,
               "mask_mod": None, "_mDynamicCausal": None, "_seqlen_first": None}
P206_STATE = {"ready": False}
_p206_compiled = {}
_p206_sms = {}


def p206_fits_two_ctas(tile_m, tile_n, d, dv):
    """PR206 interface._fits_two_ctas_sm90_native_fp8."""
    smem = max(tile_m * d, 2 * tile_m * dv) + 1024 + 2 * tile_n * (d + 2 * dv)
    return tile_m == 64 and dv <= 128 and smem <= 228 * 1024 // 2 - 1024 - 1024


def p206_plan(batch, hkv, qhead, q_len, max_k, d, num_sms):
    """(num_splits, ctas_per_sm) as PR206's interface picks them for this configuration with num_splits=0 (auto): the
    native-FP8 part of _get_fwd_config (pack-GQA, causal, not local) and _flash_attn_fwd's ctas_per_sm rule."""
    tile_m, tile_n = P206_TILE
    num_m = (q_len * qhead + tile_m - 1) // tile_m
    total = batch * hkv * num_m
    nnb = (max_k + tile_n - 1) // tile_n
    ns = 1 if (nnb <= 4 or total == 0) else min(num_sms // total, 128, nnb)   # num_splits_heuristic
    mb = max(1, total)
    if ns > 1:
        ns = min(ns, max(1, nnb), max(1, num_sms // mb))
    fits = p206_fits_two_ctas(tile_m, tile_n, d, d)
    if fits:
        ns = max(ns, min(2 * num_sms // mb, nnb // 4, 128))
    return ns, (2 if total * max(1, ns) > num_sms and fits else 1)


def _p206_num_sms(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _p206_sms.get(idx)
    if n is None:
        n = _p206_sms[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return n


def _p206_route(data):
    """PR206 for this call? Full attention, q_len 8, batch >= P206_MIN_BATCH and the PR's plan = the compiled configuration
    (>= 2 splits, 2 CTAs / SM). Shapes / config only (graph-safe); _v18_fa4_wanted has already checked dtypes and layouts."""
    cfg = data["config"]
    if cfg["window"] or cfg["q_len"] != 8 or cfg["batch"] < P206_MIN_BATCH or cfg["head_dim"] != 64:
        return False
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    if hq != 8 * hkv or data["k_cache"].shape[1] != 1:
        return False
    ns, cps = p206_plan(cfg["batch"], hkv, hq // hkv, 8, data["page_table"].shape[1], 64, _p206_num_sms(data["k_cache"].device))
    return ns >= 2 and cps == 2


def p206_fwd_partials(q8, k8, v8, page_table, seqused_k, softmax_scale, num_splits):
    """PR206 core, host path of its interface._flash_attn_fwd for this one configuration (compile once, then the TVM-FFI call
    with the same arguments): q8 [B, L, Hq, 64], k8 / v8 [P, 1, Hkv, 64] E4M3, page_table [B, W] int32, seqused_k [B] int32.
    Returns out_partial f32 [S, B, L, Hq, 64] (per-split normalised) and lse_partial f32 [S, B, Hq, L] (natural log), FA4 v8's
    k1fa4_fwd_partials layout."""
    assert q8.dtype == k8.dtype == v8.dtype == torch.float8_e4m3fn and k8.shape[1] == 1 and num_splits >= 2
    B, L, hq, d = q8.shape
    hkv = k8.shape[2]
    assert d == 64 and hq == 8 * hkv and L * (hq // hkv) <= P206_TILE[0]
    o_part = torch.empty((num_splits, B, L, hq, d), dtype=torch.float32, device=q8.device)
    lse_part = torch.empty((num_splits, B, hq, L), dtype=torch.float32, device=q8.device)
    key = (hq // hkv, d)
    fn = _p206_compiled.get(key)
    if fn is None:
        fa = p206_FlashAttentionForwardSm90(
            cutlass.Float8E4M3FN, d, d, hq // hkv, is_causal=True, is_local=False, is_split_kv=True, pack_gqa=True,
            tile_m=P206_TILE[0], tile_n=P206_TILE[1], num_stages=2, num_threads=384, Q_in_regs=False, intra_wg_overlap=True,
            mma_pv_is_rs=True, mask_mod=None, score_mod=None, has_aux_tensors=False, q_subtile_factor=1, paged_kv_non_tma=True,
            output_quant_key=None, kv_dtype=cutlass.Float8E4M3FN, fp8_kv_dequant=False, native_fp8=True,
            native_fp8_o_dtype=cutlass.BFloat16, native_fp8_tma_box_n=None, native_fp8_direct_3d_grid=False,
            native_fp8_ctas_per_sm=2)
        tc = k1fa4_to_cute_tensor
        fn = cute.compile(
            fa, tc(q8), tc(k8), tc(v8), tc(o_part), tc(lse_part, assumed_align=4), softmax_scale, None, None, None,
            tc(seqused_k, assumed_align=4, leading_dim=0), None, tc(page_table, assumed_align=4, leading_dim=1), None, None,
            None, None, p206_AuxData(None, None), None, None, None, None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        bad = {k: getattr(fa, k, "<missing>") for k, v in P206_CONFIG.items() if getattr(fa, k, "<missing>") != v}
        if bad:
            raise RuntimeError(f"PR206 compiled a different configuration: {bad}")
        _p206_compiled[key] = fn
    fn(q8.view(torch.uint8), k8.view(torch.uint8), v8.view(torch.uint8), o_part, lse_part, softmax_scale, None, None, None,
       seqused_k, None, page_table, None, None, None, None, p206_AuxData(None, None), None, None, None, None)
    return o_part, lse_part


def p206_run(data, pdl=True):
    """k1fa4_run (FA4 v8's prologue -> core -> combine + RMS-norm x sigmoid(gate), PDL-chained) with the PR206 core; the same
    contract: data["qkv"] [T, 2304], returns bf16 [B, L, Hq * 64]. The core is a plain launch (no PDL): it starts after the
    prologue completes; the combine is its PDL secondary (gate loads before griddepcontrol.wait, as in k1fa4_run)."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    assert not cfg["window"] and length == 8
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    q8 = torch.empty((tokens, hq * d), dtype=torch.float8_e4m3fn, device=device)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = K1FA4_PRO_BM or (4 if n_rows < 1024 else (8 if n_rows < 8192 else 16))
    theta = cfg["rope_theta"]
    cos_sin = k1fa4_cos_sin(theta, d, device) if theta else qkv
    pdl_kw = {"launch_pdl": True} if pdl else {}
    _k1fa4_prologue[(triton.cdiv(n_rows, block_m),)](
        qkv, q8, data["positions"] if theta else qkv, cos_sin, k_cache.view(-1, hkv * d), v_cache.view(-1, hkv * d),
        data["cache_loc"], K1FA4_RMS_EPS, qkv.stride(0), n_rows, hq, hkv, d, triton.next_power_of_2(d), block_m,
        HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=K1FA4_PRO_WARPS, **pdl_kw)
    ns, _ = p206_plan(batch, hkv, hq // hkv, length, data["page_table"].shape[1], d, _p206_num_sms(device))
    o_part, lse_part = p206_fwd_partials(q8.view(batch, length, hq, d), k_cache, v_cache, data["page_table"],
                                         data["cache_seqlens"], cfg["sm_scale"] * cfg["qk_scale_factor"], ns)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    rows = tokens * hq
    hb = k1fa4_comb_hb(hq, triton.next_power_of_2(ns), d, length)
    if hb > 1:
        _k1fa4_combine_gate_hb[(tokens * (hq // hb),)](
            o_part, lse_part, qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, K1FA4_RMS_EPS,
            H=hq, D=d, NS_P2=triton.next_power_of_2(ns), HB=hb, USE_PDL=pdl, num_warps=K1FA4_COMB_WARPS, **pdl_kw)
    else:
        _k1fa4_combine_gate[(rows,)](
            o_part, lse_part, qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, K1FA4_RMS_EPS,
            H=hq, D=d, NS_P2=triton.next_power_of_2(ns), USE_PDL=pdl, num_warps=2 if ns <= 8 else 4, **pdl_kw)
    return out


def _p206_self_test(device):
    """Per case (all on the PR206 route): rel L2 vs the FP32 reference <= 1.25 x the r1 path's + 5e-3 (k1fa4_self_test's rule)
    and <= 1.25 x the current path's (FA4 v8, k1fa4_run) + 5e-3, finite, new cache bytes == r1's."""
    ok_all = True
    for batch, ctx, length, window, seed, row0 in P206_SELF_TEST:
        ref = _k1fa4_reference(_k1fa4_case(batch, ctx, length, window, seed, row0, device))
        d1 = _k1fa4_case(batch, ctx, length, window, seed, row0, device)
        o1 = _k1fa4_r1_path(d1)
        dp = _k1fa4_case(batch, ctx, length, window, seed, row0, device)
        if not _p206_route(dp):
            print(f"[p206] self-test B{batch} ctx{ctx}: case is not on the PR206 route", flush=True)
            return False
        op = p206_run(dp)
        dk = _k1fa4_case(batch, ctx, length, window, seed, row0, device)
        ok8 = k1fa4_run(dk)
        torch.cuda.synchronize()
        e1 = float((o1.float() - ref).norm() / ref.norm())
        ep = float((op.float() - ref).norm() / ref.norm())
        e8 = float((ok8.float() - ref).norm() / ref.norm())
        rows = dp["cache_loc"]
        same = (torch.equal(dp["k_cache"].view(-1, 128).view(torch.uint8)[rows], d1["k_cache"].view(-1, 128).view(torch.uint8)[rows])
                and torch.equal(dp["v_cache"].view(-1, 128).view(torch.uint8)[rows], d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]))
        ok = bool(torch.isfinite(op).all()) and ep <= 1.25 * e1 + 5e-3 and ep <= 1.25 * e8 + 5e-3 and same
        ok_all &= ok
        ns, cps = p206_plan(batch, 2, 8, length, dp["page_table"].shape[1], 64, _p206_num_sms(device))
        print(f"[p206] self-test B{batch} ctx{ctx} L{length}{' row0' if row0 else ''} splits {ns} ctas/SM {cps}: "
              f"{'PASS' if ok else 'FAIL'} err {ep:.4e} (r1 {e1:.4e}, FA4 v8 {e8:.4e}) cache==r1 {same}", flush=True)
        del ref, d1, o1, dp, op, dk, ok8
    return ok_all


def _p206_graph_check(device):
    """CUDA-graph capture + replay of p206_run: replay == eager bit for bit; then new inputs (another seed) copied into the
    captured tensors: replay == an eager run on those inputs, bit for bit (nothing input-derived is kept). Prepare only."""
    for batch, ctx, length, window, seed, seed2 in P206_GRAPH_CASES:
        d = _k1fa4_case(batch, ctx, length, window, seed, False, device)
        o1 = p206_run(d).clone()
        d = _k1fa4_case(batch, ctx, length, window, seed, False, device)
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            og = p206_run(d)
        g.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1))
        fresh = _k1fa4_case(batch, ctx, length, window, seed2, False, device)
        o2 = p206_run(_k1fa4_case(batch, ctx, length, window, seed2, False, device)).clone()
        for k, v in fresh.items():
            if isinstance(v, torch.Tensor):
                assert d[k].shape == v.shape and d[k].dtype == v.dtype, k
                d[k].copy_(v)
        og.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize(device)
        same2 = bool(torch.equal(og, o2)) and bool(torch.isfinite(og).all())
        print(f"[p206] graph check B{batch} ctx{ctx} L{length}: replay identical {same}, after new inputs {same2}", flush=True)
        del g, og, o1, o2, d, fresh
        if not (same and same2):
            return False
    return True


def _p206_prepare(state, t0):
    """JIT + self-test + graph check of the PR206 route; state["p206"] / P206_STATE["ready"] = True only if all passed. Needs
    the FA4 v8 route (prologue, combine, cache, r1 reference path) live; skipped past P206_PREPARE_BY_S."""
    state["p206"] = False
    P206_STATE["ready"] = False
    print(f"[p206] levers: P206_ENABLE={P206_ENABLE} min batch {P206_MIN_BATCH} tile {P206_TILE} prepare-by {P206_PREPARE_BY_S}s", flush=True)
    if not P206_ENABLE:
        return
    if not state.get("fa4v6") or not K1FA4_STATE["ready"]:
        print("[p206] skipped: the FA4 v8 route is not live (FA4 v8 / k3_v14 routes stay)", flush=True)
        return
    if time.time() - t0 > P206_PREPARE_BY_S:
        print(f"[p206] skipped: prepare already took {time.time() - t0:.1f}s (> {P206_PREPARE_BY_S}s): FA4 v8 stays", flush=True)
        return
    device = state["router"].device
    t1 = time.time()
    t2 = t1
    try:
        ok = _p206_self_test(device)
        t2 = time.time()
        if ok:
            ok = _p206_graph_check(device)
    except Exception as ex:  # noqa: BLE001
        print(f"[p206] self-test raised {repr(ex)[:600]}", flush=True)
        ok = False
    torch.cuda.synchronize()
    P206_STATE["ready"] = bool(ok)
    state["p206"] = bool(ok)
    print(f"[p206] route {'live' if ok else 'FAILED: FA4 v8 stays on every shape'} (JIT + self-test {t2 - t1:.1f}s, graph check "
          f"{time.time() - t2:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)

'''

# (what, old, new): each `old` must occur exactly once in the layer text
PATCHES = [
    ("nonce", "# nonce k3-v34-1 (k3_v19 + the v20 levers listed in tools/gen_k3_v20.py)\n",
     "# nonce k3-fa-1 (k3_v34 + PR206 full verify attention, tools/fa206_patches.py)\n"),
    ("generated-by", "# GENERATED by tools/gen_k3_v20.py from kernels/muse_decode_layer/k3_v19.py",
     "# GENERATED by tools/gen_k3_fa.py (k3_v34 + tools/fa206_patches.py); k3_v34 by tools/gen_k3_v20.py from kernels/muse_decode_layer/k3_v19.py"),
    ("docstring", '"""K3 v20 = k3_v19 + the v20 levers (tools/gen_k3_v20.py):',
     '"""K3 FA = k3_v34 + PR206 (tools/fa206_patches.py): the full-attention layers at q_len 8, batch >= 16 (verify 16 / 32 full) run the\n'
     'SM90 native-FP8 forward of flash-attention PR #206 (vendored as plain top-level source, kernels/muse_attention_decode/fa206_vendor.py)\n'
     'between FA4 v8\'s prologue and combine; box in-layer A/B x1.016 / x1.013 on those shapes. Live after its prepare self-test, else FA4 v8.\n\n'
     'K3 v20 = k3_v19 + the v20 levers (tools/gen_k3_v20.py):'),
    ("route", '''    return k1fa4_run(dict(data, qkv=qkv), pdl=pdl)


def _v18_graph_check(device):''',
     '''    if P206_ENABLE and P206_STATE["ready"] and _p206_route(data):
        try:
            # PR206: full attention, q_len 8, batch >= 16 (tools/fa206_patches.py); a host exception keeps FA4 v8 from now on
            return p206_run(dict(data, qkv=qkv), pdl=pdl)
        except Exception as ex:  # noqa: BLE001
            P206_STATE["ready"] = False
            print(f"[p206] raised at call time, FA4 v8 from now on: {ex!r}", flush=True)
    return k1fa4_run(dict(data, qkv=qkv), pdl=pdl)


def _v18_graph_check(device):'''),
    ("prepare", "    _dg1_prepare(state, t0)   # DG1: swap-AB wgmma projections at T = 32 / 128 / 256 (today's cuBLAS stays on failure)\n",
     "    _dg1_prepare(state, t0)   # DG1: swap-AB wgmma projections at T = 32 / 128 / 256 (today's cuBLAS stays on failure)\n"
     "    _p206_prepare(state, t0)   # PR206: full verify attention at batch >= 16 (FA4 v8 stays on failure)\n"),
]
ANCHOR = "def _r1_attention(qkv, data, state):\n"


def section():
    vendor = VENDOR.read_text(encoding="utf-8")
    return "\n" + vendor.rstrip("\n") + "\n" + GLUE.rstrip("\n") + "\n\n\n"


def apply(text, patches=PATCHES):
    for what, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"fa206 patch {what!r}: `old` occurs {n} times (want 1)")
        text = text.replace(old, new)
    if text.count(ANCHOR) != 1:
        raise RuntimeError("fa206 section anchor must occur once")
    return text.replace(ANCHOR, section() + ANCHOR)


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_fa_check.py")
    text = src.read_text(encoding="utf-8")
    patched = apply(text)
    out.write_bytes(patched.encode("utf-8"))
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches + the section hit once; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

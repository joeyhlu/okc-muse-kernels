# V20 layer glue for P1M (kernels/muse_bf16_proj/p1m_module.py): P1's lossless 12-bit packed projections at T = 9..32.
# Pasted verbatim into k3_v20.py by tools/gen_k3_v20.py, after P1M's own nodes (P1M_CPP_ADD, P1M_CUDA_ADD, p1m_cuda_src,
# p1m_lane_maps, P1M_KPERM, p1m_interleave, p1m_ws, p1m_call, p1m_esc_columns) and _v20_p1_pack (the layer's p1_pack with
# the k-permuted lane map, generated). Names it uses from the layer: p1_unpack, p1_describe, _k3_disable, _k2a_bucket_of.
import torch

V20_P1M_ENABLE = True   # master switch (read at call time; False also keeps P1M out of the P1 build)
# (projection, bucket) -> (warps, split, depth code, mt, rt, interleaved): box sweep winners vs today's cuBLAS path
V20_P1M_ROUTE = {}
V20_P1M_TOL = (4e-3, 1e-2)   # self-test: rel-L2 vs FP64, vs today's path (P1's tolerances)
V20_P1M_PDL = True    # P1M launched with programmatic stream serialization (weights + escapes before its wait; read at call time)
V20_P1M_TRIG = True   # P1M executes griddepcontrol.launch_dependents at entry (one-wave grids; read at call time)
V20_P1_TRIG = {'qkv': True, 'o': True, 'pooler': True, 'unpooler': True}   # same for P1 (T <= 8), per projection
# (box A/B ab_v21a: P1 triggers x1.0074, + P1M PDL + trigger x1.0116 over the T = 1 / 8 / 16 / 32 loads)


def _v20_inst():
    """(depth code, mt, rt) instantiations the layer's P1 build needs for the routed geometries."""
    return tuple(sorted({(g[2], g[3], g[4]) for g in V20_P1M_ROUTE.values()}))


def _v20_ok(x, k):
    """p1m_gemm's host checks for x, shape / stride / pointer only (graph safe)."""
    return x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == k and x.is_contiguous() and x.data_ptr() % 16 == 0


def _v20_p1m(state, proj, x):
    """P1M for (projection, T = x.shape[0]) when routed and self-tested, else None -> the caller keeps today's path.
    proj "cat" = [pooler; router hi; router lo] with FP32 out (the T >= 16 pooler + router GEMM). Shape-only branches; a
    host exception disables P1M for the rest of the run (once, with a line), as P1 / K1 / K2A do."""
    st = state.get("p1m")
    if not V20_P1M_ENABLE or st is None:
        return None
    T = x.shape[0]
    if T <= 8 or T > 32:
        return None
    key = (proj, _k2a_bucket_of(T))
    if key not in V20_P1M_ROUTE:   # call-time filter: A/B variants switch routes off without a new prepare
        return None
    geo = st["route"].get(key)
    P = st["proj"].get(proj)
    if geo is None or P is None or T > 8 * geo[3] or not _v20_ok(x, P["K"]):
        return None
    try:
        out = torch.empty((T, P["N"]), dtype=torch.float32 if proj == "cat" else torch.bfloat16, device=x.device)
        pdl = (1 if V20_P1M_PDL else 0) | (4 if V20_P1M_TRIG and st["wave1"].get(key) else 0)
        p1m_call(st["mod"], P, x, out, geo[:5], il=st["il"].get(key), pdl=pdl)
        return out
    except Exception as ex:  # noqa: BLE001
        _k3_disable(state, "p1m", None, "P1M", ex)
        return None


def _v20_weight(state, proj):
    return state["v8"]["wcat"] if proj == "cat" else state["dense"]["qkv_gate" if proj == "qkv" else proj]


def _v20_selftest(st, state, proj, key, gen):
    """(ok, message) for one routed (projection, bucket) at T = bucket: one-hot probes (x = e_k gives y = W[:, k] exactly;
    escape columns first, the tensor core's subnormal flush counted apart, rows with inf / NaN weights skipped), random x
    against FP64 and against today's path, and a repeat that must be bit-identical. Prepare-time only."""
    P = st["proj"][proj]
    geo = st["route_try"][key]
    w = _v20_weight(state, proj)
    N, K, dev, T = P["N"], P["K"], w.device, key[1]
    f32 = proj == "cat"

    def run(x):
        out = torch.empty((T, N), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
        p1m_call(st["mod"], P, x, out, geo[:5], il=st["il"].get(key))
        return out
    fin = torch.isfinite(w.float()).all(1)
    esc = sorted(set(p1m_esc_columns(P["esc"]).tolist())) if P["n_esc"] else []
    esc = esc[::max(1, len(esc) // 48)][:48]
    rnd = torch.randint(0, K, (64,), generator=gen, device=dev).tolist()
    cols = esc + rnd[:16]
    cols += rnd[16:16 + (-len(cols)) % T]
    bad = flushed = 0
    for i in range(0, len(cols), T):
        cc = cols[i:i + T]
        x = torch.zeros((T, K), dtype=torch.bfloat16, device=dev)
        ci = torch.tensor(cc, device=dev)
        x[torch.arange(T, device=dev), ci] = 1.0
        got = run(x).float()
        ref = w[:, ci].t().float()
        mis = (got != ref) & ~(torch.isnan(got) & torch.isnan(ref)) & fin[None, :]
        sub = mis & (got == 0) & (ref.abs() < 2.0 ** -126)
        flushed += int(sub.sum())
        bad += int((mis & ~sub).sum())
    x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
    got = run(x)
    again = run(x)
    ref64 = x.double() @ w.double().t()
    today = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
    torch.cuda.synchronize()
    e64 = float((got.double() - ref64).norm() / ref64.norm())
    etoday = float((got.float() - today.float()).norm() / today.float().norm())
    same = bool(torch.equal(got, again))
    ok = bad == 0 and e64 < V20_P1M_TOL[0] and etoday < V20_P1M_TOL[1] and same
    return ok, (f"{len(cols)} one-hot cols ({len(esc)} escape) bad {bad} (subnormal flush {flushed}); random x rel vs fp64 "
                f"{e64:.2e} vs today {etoday:.2e}; repeat identical {same}")


def _v20_prepare(state):
    """P1M packs (k-permuted, bit-exact unpack check), interleaved layouts and per-route self-tests, after P1 (whose build
    holds p1m_gemm) and the v8 tail ([pooler; hi; lo]). state["p1m"] stays None (today's path) when off or nothing passes."""
    state["p1m"] = None
    if not V20_P1M_ENABLE or not V20_P1M_ROUTE:
        return None
    st1 = state.get("p1")
    mod = st1["mod"] if st1 is not None else None
    if mod is None or not hasattr(mod, "p1m_gemm"):
        print("[v20] P1M: no P1 build with p1m_gemm -> today's path at T = 9..32", flush=True)
        return None
    t0 = time.time()
    st = {"mod": mod, "proj": {}, "route": {}, "route_try": dict(V20_P1M_ROUTE), "il": {}, "wave1": {}}
    try:
        gen = torch.Generator(device=state["dense"]["qkv_gate"].device).manual_seed(2020)
        for proj in sorted({k[0] for k in V20_P1M_ROUTE}):
            if proj == "cat" and state.get("v8") is None:
                continue
            w = _v20_weight(state, proj)
            P = _v20_p1_pack(w, name=proj)
            mism = p1m_verify(mod, P, w)
            torch.cuda.synchronize()
            print(f"[v20] P1M {p1_describe(P)} | k-permuted unpack mismatches {mism} -> {'ok' if mism == 0 else 'DROPPED'}",
                  flush=True)
            if mism:
                continue
            st["proj"][proj] = P
            for key in sorted(k for k in V20_P1M_ROUTE if k[0] == proj):
                geo = V20_P1M_ROUTE[key]
                if geo[5]:
                    il = p1m_interleave(P, geo[0], geo[1], geo[4])
                    if il is None:
                        print(f"[v20] P1M {key}: no interleaved layout for {geo} -> route off", flush=True)
                        continue
                    st["il"][key] = il
                ok, msg = _v20_selftest(st, state, proj, key, gen)
                print(f"[v20] P1M self-test {proj} T={key[1]} geo {geo}: {msg} -> {'ok' if ok else 'FAIL: route off'}", flush=True)
                if ok:
                    st["route"][key] = geo
                    occ = int(mod.p1m_info(geo[2], geo[3], geo[4], geo[0])[2])
                    sms = torch.cuda.get_device_properties(w.device).multi_processor_count
                    st["wave1"][key] = (P["N"] // (16 * geo[4])) * geo[1] <= occ * sms   # early trigger only for one-wave grids
                else:
                    st["il"].pop(key, None)
    except Exception as ex:  # noqa: BLE001
        print(f"[v20] P1M prepare / self-test FAILED -> today's path at T = 9..32: {ex!r}", flush=True)
        return None
    print(f"[v20] P1M routes live {sorted(st['route'])} (prepare {time.time() - t0:.1f}s)", flush=True)
    state["p1m"] = st if st["route"] else None
    return state["p1m"]


def p1m_verify(mod, P, w):
    """Bit mismatches of P1's unpack kernel (classic positions) against W with the SB k-permutation applied. Must be 0."""
    u = p1_unpack(mod, P)
    K = w.shape[1]
    perm = torch.tensor(P1M_KPERM, device=w.device)
    cols = (torch.arange(K // 64, device=w.device).view(-1, 1) * 64 + perm.view(1, -1)).reshape(-1)
    return int((u.view(torch.int16) != w[:, cols].view(torch.int16)).sum())


_V20_WAVE1 = {}


def _v20_p1_trig(st, proj, geo, mode):
    """2 (P1's launch_dependents bit) when V20_P1_TRIG is on for proj and the P1 grid (rows / 16 x split CTAs) fits in one wave
    at its occupancy (cached per geometry); else 0. Shape-only; graph safe after the first (prepare-time) call per geometry."""
    warps, split, depth = geo
    P = st["proj"][proj]
    key = (proj, warps, split, depth, mode, P["rows_t8"])
    if key not in _V20_WAVE1:   # filled on the first call (P1's prepare self-tests), whatever the switch says
        occ = int(st["mod"].p1s_info(depth, mode, warps)[2])
        sms = torch.cuda.get_device_properties(P["buf"].device).multi_processor_count
        _V20_WAVE1[key] = (P["rows_t8"] // 16) * split <= occ * sms
    return 2 if (V20_P1_TRIG.get(proj) and _V20_WAVE1[key]) else 0

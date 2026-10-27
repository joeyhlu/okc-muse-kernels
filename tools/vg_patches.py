"""vg_patches.py: VG glue riders at T >= 32 (VG glue builder, notes/study/vg_glue.md) on a stripped layer base.

VG-B: the layer boundary (post-norm + gated residual + pre-norm) as our own kernel (kernels/lab/vg_bnd_lab.py is the one
source of the CUDA text) or as SGLang's fused_norm_residual with another launch shape, routed per (boundary, K2A bucket)
by VG_BND_ROUTE; empty = the base's behaviour (SBB1 / SGLang cluster at T <= 32, the SGLang module at T >= 128).
prepare() self-tests every route it may use (VG_BND_ROUTE + VG_BND_CAND) against the SGLang module op at the bucket's
low / representative / high token counts and drops a failing route loudly (the base path stays). VG_BND_ROUTE is read
at call time (one lab file serves every A/B arm).

    python -I tools/vg_patches.py [--base=k3_va_3] [--n=1] [--route=<json>] [--cand=<json>] [--lab-only]
      --route: {"attn,128": ["vgb", <geo>], "ffn,256": ["fnr", 2, null], ...}   (default: {} = base behaviour)
      --cand:  {"attn,128": [[...], ...], ...}   extra routes self-tested at prepare (A/B lab copies only)
Writes kernels/muse_decode_layer/k3_vg_<n>.py (unless --lab-only) and the byte-identical box copy
kernels/lab/k3_vg_<n>_ab.py.
"""
import hashlib
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
NL = chr(10)


def _lab():
    spec = importlib.util.spec_from_file_location("vg_bnd_lab", str(ROOT / "kernels/lab/vg_bnd_lab.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _route_lit(d):
    """{"attn,128": [...]} (JSON) -> a Python literal {('attn', 128): (...)}."""
    out = {}
    for k, v in d.items():
        w, b = k.split(",")
        out[(w, int(b))] = v
    return out


def _tup(v):
    return tuple(_tup(x) for x in v) if isinstance(v, list) else v


VG_PY = r'''
VG_ENABLE = True
VG_BND_ROUTE = @ROUTE@
VG_BND_CAND = @CAND@
VG_PREPARE_BY_S = 255.0
VGF_ENABLE = True
VGF_PDL = True
_VG = {"mod": None, "w": {}, "ok": set(), "build_s": 0.0, "vgf_ok": False}


def _vgf_on():
    return VGF_ENABLE and _VG["vgf_ok"] and _VG["mod"] is not None


def _vml_fin(st, sc, ids, wts, out, m):
    _VG["mod"].vgf_run(sc["y"], sc["rmap"], wts, st["nw"], out, sc["xready"], sc["doneA"], sc["cnt"], m, 1 if VGF_PDL else 0)
    return out


def _vgf_fc(rep):
    """VML1's per-bucket flags / cost table where the base has them (k3_vml_5+), else the VML defaults (None)."""
    return globals().get("VML_FLAGS", {}).get(rep), globals().get("VML_COST", {}).get(rep)


def _vgf_prepare(state):
    _VG["vgf_ok"] = False
    if not VGF_ENABLE:
        return
    k33 = state.get("k2a33")
    if k33 is None or _VML.get("sc") is None or _VML.get("mod") is None or not any(_VML["ok"].values()):
        print("[vg] VG-F: VML1 not live -> vm_fin stays", flush=True)
        return
    try:
        _vg_build()
        st, sc = k33["st"], _VML["sc"]
        dev = st["nw"].device
        good, n = True, 0
        for bs, seed, dist in ((100, 41, "normal"), (128, 42, "skewed"), (160, 43, "skewed"), (200, 44, "skewed"),
                               (256, 45, "normal"), (320, 47, "normal")):
            if not _VML["ok"].get(_k2a_bucket_of(bs)):
                continue
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            x = d["hidden_states"].contiguous()
            ids = d["topk_ids"].to(torch.int32).contiguous()
            wts = d["topk_weights"].to(torch.float32).contiguous()
            fl, co = _vgf_fc(_k2a_bucket_of(bs))
            if (_vml_FLAGS_DEFAULT if fl is None else fl) & 8:
                continue
            prev = None
            for call in range(3):
                # one vm_main, then VG-F and vm_fin on the SAME y (vm_main's split-unit fixups are not bitwise
                # reproducible across calls: R-vg-2); VG-F runs first so its counter reset is what the check sees
                sc["err"].zero_()
                out = torch.full((bs, _vml_KDIM), 3.0, dtype=torch.bfloat16, device=dev)
                ref = torch.full((bs, _vml_KDIM), -3.0, dtype=torch.bfloat16, device=dev)
                _vml_call(_VML["mod"], st, sc, x, ids, wts, 0, mask=1, out=out, flags=fl, cost=co)
                _vml_fin(st, sc, ids, wts, out, bs)
                torch.cuda.synchronize()
                clean = int(sc["xready"].abs().sum()) + int(sc["doneA"].abs().sum()) + int(sc["cnt"].abs().sum())
                _vml_call(_VML["mod"], st, sc, x, ids, wts, 0, mask=2, out=ref, flags=fl, cost=co)
                torch.cuda.synchronize()
                eq = bool(torch.equal(out, ref))
                ok = eq and not int(sc["err"][0]) and clean == 0 and bool(torch.isfinite(out.float()).all())
                rep_eq = "-" if prev is None else str(bool(torch.equal(prev, ref)))
                prev = ref
                good &= ok
                n += 1
                if not ok or call in (0, 2):
                    print(f"[vg] VG-F self-test bs={bs} {dist} call {call}: bitwise vs vm_fin {eq} err {int(sc['err'][0])} "
                          f"counters {clean} (vm_main reproducible across calls: {rep_eq}) -> {'ok' if ok else 'FAIL'}",
                          flush=True)
        _VG["vgf_ok"] = bool(good and n)
    except Exception as ex:
        _VG["vgf_ok"] = False
        print(f"[vg] VG-F self-test raised {ex!r} -> vm_fin stays", flush=True)
    try:
        _k2a33_reset(state["k2a33"]["st"])
    except Exception:
        pass
    print(f"[vg] VG-F (VML1 finalize) {'LIVE' if _VG['vgf_ok'] else 'FALLBACK to vm_fin'}", flush=True)


def _vg_build():
    _pt1_wait("vg")
    if _VG["mod"] is not None:
        return _VG["mod"]
    from torch.utils.cpp_extension import load_inline
    t1 = time.time()
    _VG["mod"] = load_inline("vg_k3", cpp_sources=VG_DECL + chr(10) + VGF_DECL, cuda_sources=VG_CU + VG_HOST + VGF_CU + VGF_HOST,
                             functions=(["vgb_run"] if VG_DECL else []) + ["vgf_run"],
                             extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _VG["build_s"] = time.time() - t1
    print(f"[vg] VG-B / VG-F compiled in {_VG['build_s']:.1f}s", flush=True)
    return _VG["mod"]


def _vg_key(route):
    return tuple(_vg_key(x) if isinstance(x, (list, tuple)) else x for x in route)


def _vg_run(which, route, stream, branch):
    pw, qw, al, bt, isc, pe, qe = _VG["w"][which]
    T = branch.shape[0]
    if route[0] == "vgb":
        out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=branch.device)
        ns = torch.empty((T, HIDDEN), dtype=torch.float32, device=branch.device)
        if not _VG["mod"].vgb_run(branch.contiguous(), stream.contiguous(), out, ns, pw, qw, al, bt, pe, qe, int(route[1])):
            raise RuntimeError(f"vgb geometry {int(route[1]):#x} not built")
        return out, ns, None
    if route[0] == "fnr":
        lc = tuple(route[2]) if route[2] is not None else None
        o = _V9_FNR(branch.contiguous(), stream.contiguous(), pe, qe, post_norm_is_input_scale=isc,
                    post_weight=pw if pw.numel() else None, pre_weight=qw if qw.numel() else None, alpha=al, beta=bt,
                    want_router=False, launch_config=lc, cluster_size=int(route[1]))
        return o[0], o[1], None
    raise ValueError(f"vg route {route!r}")


def _vg_bnd(which, stream, branch, router_dtype, state, tokens):
    if not VG_ENABLE or router_dtype is not None or not VG_BND_ROUTE:
        return None
    rep = _k2a_bucket_of(tokens)
    route = VG_BND_ROUTE.get((which, rep))
    if route is None or (which, rep, _vg_key(route)) not in _VG["ok"]:
        return None
    if branch.dtype != torch.bfloat16 or stream.dtype != torch.float32:
        return None
    return _vg_run(which, route, stream, branch)


def _vg_prepare(state, t0):
    _VG["ok"] = set()
    if not VG_ENABLE:
        return
    if time.time() - t0 <= VG_PREPARE_BY_S:
        _vgf_prepare(state)
    else:
        print(f"[vg] VG-F SKIPPED (prepare already at {time.time() - t0:.0f}s): vm_fin stays", flush=True)
    routes = {}
    for k, r in list(VG_BND_ROUTE.items()) + [(k, r) for k, rs in VG_BND_CAND.items() for r in rs]:
        lst = routes.setdefault(k, [])
        if _vg_key(r) not in [_vg_key(x) for x in lst]:
            lst.append(r)
    if not routes:
        return
    if time.time() - t0 > VG_PREPARE_BY_S:
        print(f"[vg] SKIPPED (prepare already at {time.time() - t0:.0f}s): base boundaries stay", flush=True)
        return
    t1 = time.time()
    try:
        if any(r[0] == "vgb" for rs in routes.values() for r in rs):
            _vg_build()
        for which in ("attn", "ffn"):
            op = state["attn_boundary" if which == "attn" else "ffn_boundary"]
            op._ensure_cached_weights(torch.bfloat16)
            dev = state["router"].device
            e = torch.empty(0, device=dev, dtype=torch.bfloat16)
            pw = op._cc_post_weight.to(torch.bfloat16).contiguous() if op._post_norm.has_weight else e
            qw = op._cc_pre_weight.to(torch.bfloat16).contiguous() if op._pre_norm.has_weight else e
            if not op._use_residual_gate or op._cc_alpha.numel() != HIDDEN:
                raise RuntimeError(f"{which}: no channel-wise residual gate")
            isc = bool(op._post_norm_is_input_scale)
            if pw.numel() and not isc:
                raise RuntimeError(f"{which}: post weight after the norm is not supported")
            _VG["w"][which] = (pw, qw, op._cc_alpha.float().contiguous(), op._cc_beta.float().contiguous(), isc,
                               float(op._post_norm.eps), float(op._pre_norm.eps))
        gen = torch.Generator(device=state["router"].device).manual_seed(6363)
        for (which, rep), rs in sorted(routes.items(), key=str):
            lo, hi = [(a, b) for a, b, r in K2A_BUCKETS if r == rep][0]
            op = state["attn_boundary" if which == "attn" else "ffn_boundary"]
            for r in rs:
                if r[0] == "fnr" and _V9_FNR is None:
                    print(f"[vg] {which} bucket {rep} {r}: SGLang fused_norm_residual unavailable", flush=True)
                    continue
                if r[0] == "vgb" and which == "ffn" and (int(r[1]) >> 21) & 1:
                    print(f"[vg] {which} bucket {rep} {r}: stream before the PDL wait is illegal here -> dropped", flush=True)
                    continue
                good, worst = True, (0.0, 0.0)
                for T in sorted({lo, rep, min(hi, K2A_M_MAX)}):
                    stream = torch.randn((T, HIDDEN), generator=gen, device=state["router"].device) * 2.0
                    branch = torch.randn((T, HIDDEN), generator=gen, device=state["router"].device).to(torch.bfloat16)
                    ref = op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=None)
                    for _ in range(2):
                        got = _vg_run(which, r, stream, branch)
                        torch.cuda.synchronize()
                        e0 = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                        e1 = float((got[1] - ref[1]).norm() / ref[1].norm())
                        worst = (max(worst[0], e0), max(worst[1], e1))
                        good &= e0 < 1e-2 and e1 < 1e-4 and bool(torch.isfinite(got[1]).all()) and bool(
                            torch.isfinite(got[0].float()).all())
                tag = "ok" if good else "FAIL"
                print(f"[vg] self-test {which} bucket {rep} {r}: out {worst[0]:.1e} stream {worst[1]:.1e} -> {tag}", flush=True)
                if good:
                    _VG["ok"].add((which, rep, _vg_key(r)))
        live = sorted(f"{w}/{b}" for (w, b), r in VG_BND_ROUTE.items() if (w, b, _vg_key(r)) in _VG["ok"])
        dead = sorted(f"{w}/{b}" for (w, b), r in VG_BND_ROUTE.items() if (w, b, _vg_key(r)) not in _VG["ok"])
        print(f"[vg] VG-B boundaries LIVE {live}{' FALLBACK (base boundary) ' + str(dead) if dead else ''} "
              f"({len(_VG['ok'])} routes ok, {time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
    except Exception as ex:
        _VG["ok"] = set()
        print(f"[vg] FAILED -> base boundaries everywhere: {ex!r}", flush=True)

'''


def apply(text, route=None, cand=None, prepare_by=255.0):
    lab = _lab()
    geos = lab.VGB_GEOS_SM90
    cu = lab.VGB_CU
    route = {k: _tup(v) for k, v in _route_lit(route or {}).items()}
    cand = {k: [_tup(x) for x in v] for k, v in _route_lit(cand or {}).items()}
    used = []
    for k, r in list(route.items()) + [(k, r) for k, rs in cand.items() for r in rs]:
        if k[0] not in ("attn", "ffn") or k[1] not in (32, 128, 256) or r[0] not in ("vgb", "fnr"):
            raise ValueError(f"route {k} -> {r}")
        if r[0] == "vgb":
            g = ((r[1] & 0xfff), (r[1] >> 12) & 0xff)
            if g not in geos:
                raise ValueError(f"vgb geometry {r[1]:#x} not built")
            if g not in used:
                used.append(g)
    # only the VG-B geometries a route can reach are instantiated (none: VG-F alone, no VG-B compile cost)
    host = lab.VGB_HOST.replace("VGB_CASES", " ".join(f"VG_CASE({n}, {c})" for n, c in used)).replace("#if VGB_PDL", "#if 1")
    py = (VG_PY.replace("@ROUTE@", repr(route)).replace("@CAND@", repr(cand))
          .replace("VG_PREPARE_BY_S = 255.0", f"VG_PREPARE_BY_S = {float(prepare_by)!r}"))
    if used:
        vgb = "VG_CU = r'''" + cu + "'''" + NL + "VG_HOST = r'''" + host + "'''" + NL + "VG_DECL = " + repr(lab.VGB_DECL) + NL
    else:   # VG-F alone: no VG-B text in the file (size); VG_DECL empty -> the module exports vgf_run only
        vgb = "VG_CU = ''" + NL + "VG_HOST = ''" + NL + "VG_DECL = ''" + NL
    block = (vgb + "VGF_CU = r'''" + lab.VGF_CU + "'''" + NL + "VGF_HOST = r'''" + lab.VGF_HOST + "'''" + NL
             + "VGF_DECL = " + repr(lab.VGF_DECL) + NL + py)
    for s in (cu, host, lab.VGF_CU, lab.VGF_HOST):
        if "'''" in s:
            raise ValueError("triple quote in CUDA text")
    reps = [
        (NL + NL + "GD_ENABLE = True" + NL, NL + NL + block + NL + "GD_ENABLE = True" + NL, "insert before GD"),
        ("def _v9_bnd(which, stream, branch, router_dtype, state, tokens):" + NL,
         "def _v9_bnd(which, stream, branch, router_dtype, state, tokens):" + NL
         + "    vgr = _vg_bnd(which, stream, branch, router_dtype, state, tokens)" + NL
         + "    if vgr is not None:" + NL + "        return vgr" + NL, "_v9_bnd hook"),
        ('    if SBB1_ENABLE:' + NL + '        jobs.append(("sbb1", _sbb1_build))' + NL,
         '    if SBB1_ENABLE:' + NL + '        jobs.append(("sbb1", _sbb1_build))' + NL
         + '    if VG_ENABLE and (VGF_ENABLE or any(r[0] == "vgb" for r in list(VG_BND_ROUTE.values()) + [x for v in '
           'VG_BND_CAND.values() for x in v])):' + NL + '        jobs.append(("vg", _vg_build))' + NL, "pt1 job"),
        ('def _vml_route(st, data):' + NL + '    x = data["hidden_states"].contiguous()' + NL
         + '    return _vml_call(_VML["mod"], st, _VML["sc"], x, data["topk_ids"].to(torch.int32).contiguous(),' + NL
         + '                         data["topk_weights"].to(torch.float32).contiguous(), int(data["config"]["local_start"]))' + NL,
         'def _vml_route(st, data):' + NL + '    x = data["hidden_states"].contiguous()' + NL
         + '    ids = data["topk_ids"].to(torch.int32).contiguous()' + NL
         + '    wts = data["topk_weights"].to(torch.float32).contiguous()' + NL
         + '    if _vgf_on():' + NL
         + '        out = torch.empty((x.shape[0], _vml_KDIM), dtype=torch.bfloat16, device=x.device)' + NL
         + '        _vml_call(_VML["mod"], st, _VML["sc"], x, ids, wts, int(data["config"]["local_start"]), mask=1, out=out)' + NL
         + '        return _vml_fin(st, _VML["sc"], ids, wts, out, x.shape[0])' + NL
         + '    return _vml_call(_VML["mod"], st, _VML["sc"], x, ids, wts, int(data["config"]["local_start"]))' + NL,
         "_vml_route hook"),
        ("    _sbb1_prepare(state, t0)" + NL, "    _sbb1_prepare(state, t0)" + NL + "    _vg_prepare(state, t0)" + NL,
         "prepare hook"),
    ]
    # k3_vml_5+ VML block: _vml_route passes the per-bucket VML_FLAGS / VML_COST (kept as is; VG-F only where the flags
    # leave the finalize to a separate launch, i.e. bit 3 = in-launch finalize is off)
    vml5_old = ('def _vml_route(st, data):' + NL + '    x = data["hidden_states"].contiguous()' + NL
                + '    rep = _k2a_bucket_of(x.shape[0])' + NL
                + '    return _vml_call(_VML["mod"], st, _VML["sc"], x, data["topk_ids"].to(torch.int32).contiguous(),' + NL
                + '                     data["topk_weights"].to(torch.float32).contiguous(), int(data["config"]["local_start"]),' + NL
                + '                     flags=VML_FLAGS.get(rep), cost=VML_COST.get(rep))' + NL)
    vml5_new = ('def _vml_route(st, data):' + NL + '    x = data["hidden_states"].contiguous()' + NL
                + '    rep = _k2a_bucket_of(x.shape[0])' + NL
                + '    ids = data["topk_ids"].to(torch.int32).contiguous()' + NL
                + '    wts = data["topk_weights"].to(torch.float32).contiguous()' + NL
                + '    fl, co, ls = VML_FLAGS.get(rep), VML_COST.get(rep), int(data["config"]["local_start"])' + NL
                + '    if _vgf_on() and not ((_vml_FLAGS_DEFAULT if fl is None else fl) & 8):' + NL
                + '        out = torch.empty((x.shape[0], _vml_KDIM), dtype=torch.bfloat16, device=x.device)' + NL
                + '        _vml_call(_VML["mod"], st, _VML["sc"], x, ids, wts, ls, mask=1, out=out, flags=fl, cost=co)' + NL
                + '        return _vml_fin(st, _VML["sc"], ids, wts, out, x.shape[0])' + NL
                + '    return _vml_call(_VML["mod"], st, _VML["sc"], x, ids, wts, ls, flags=fl, cost=co)' + NL)
    if text.count(vml5_old) == 1:
        reps = [r for r in reps if r[2] != "_vml_route hook"] + [(vml5_old, vml5_new, "_vml_route hook (vml5)")]
    for old, new, what in reps:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"vg anchor {what!r} found {n} times")
        text = text.replace(old, new, 1)
    return text


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    lab_only = "--lab-only" in sys.argv
    n = args.get("n", "1")
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_vml_5')}.py"
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{base.name} is not a finish() fixed point")
    route = json.loads(args.get("route", "{}"))
    cand = json.loads(args.get("cand", "{}"))
    tag = f"k3-vg-{n}"
    desc = ", ".join(f"{k}={v[0]}" + (f":{v[1]:#x}" if v[0] == "vgb" else f":c{v[1]}") for k, v in route.items()) or "none"
    nonce = (f"# nonce {tag} ({base.stem} + VG-F VML1 finalize (y loads in flight, bitwise vm_fin, self-tested with "
             f"fallback) + VG-B boundary routes [{desc}]; VGF_ENABLE / VG_BND_ROUTE)")
    t = apply(b.replace(bn, nonce), route, cand, float(args.get("prepare-by", 255.0)))
    rep = [f"base {base.name} {len(b.encode())} B is a finish() fixed point"]
    if lab_only:
        chk0 = GM.check
        GM.check = lambda ok, msg: chk0(ok or "file >= 1 MB" in msg, msg)
        try:
            s = GM.finish(t, nonce, tag, chk_base, rep)
        finally:
            GM.check = chk0
    else:
        s = GM.finish(t, nonce, tag, chk_base, rep)
    rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B (headroom {1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[vg_patches] OK  {r}")
    out, lab = ROOT / f"kernels/muse_decode_layer/k3_vg_{n}.py", ROOT / f"kernels/lab/k3_vg_{n}_ab.py"
    if not lab_only:
        out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vg_patches] wrote {'' if lab_only else str(out.relative_to(ROOT)) + ' and '}{lab.relative_to(ROOT)} "
          f"({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

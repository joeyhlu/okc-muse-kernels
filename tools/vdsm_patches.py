"""vdsm_patches.py: lever P6a (notes/study/gap_oct28.md s6) in the layer = the attention boundary's per-token moments from GD o's
epilogue (OUT 2) and SBBM (SBB1 without its 16-CTA cluster exchange) at the GD o buckets (VDS builder, 2026-10-28).
kernels/lab/vds_mom_lab.py is the one source of the GD patch (gdm_patch) and of SBBM's CUDA text.

Adds, on a stripped base (finish() fixed point):
  * GD: OUT 2 instances (twins of the o routes) + gd_run_mom; SBB1 module: sbbm_run.
  * _gdm_prepare (after _gd_prepare): per bucket, GD o OUT 2 vs OUT 0 (o bitwise), the partial-record sums vs FP64, and
    GD o OUT 2 -> SBBM vs the boundary op (SBB1's own tolerances); a failing bucket prints FALLBACK and stays on GD o + SBB1.
  * call time: GDM_ENABLE (A/B arms); _gdm_o replaces GD o and leaves (o, moments) pending; _v9_bnd("attn") takes SBBM when
    the branch is that o.

    python -I tools/vdsm_patches.py --base=k3_slim_5 --n=1 [--lab-only] [--dry --dryout=<path>]
Writes kernels/muse_decode_layer/k3_vdsm_<n>.py (unless --lab-only) and kernels/lab/k3_vdsm_<n>_ab.py; never overwrites.
"""
import hashlib
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
NL = chr(10)


def _lab():
    sys.path.insert(0, str(ROOT / "kernels/lab"))
    spec = importlib.util.spec_from_file_location("vds_mom_lab", str(ROOT / "kernels/lab/vds_mom_lab.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GDM_PY = r'''
GDM_ENABLE = True
GDM_BUILD = True
GDM_BUCKETS = (1, 8, 16)
_GDM = {"pend": None, "ok": set(), "inst2": {}, "flags": 0, "al": None, "be": None, "pw": None}


def _gdm_inputs(state):
    op = state["attn_boundary"]
    op._ensure_cached_weights(torch.bfloat16)
    dev = state["router"].device
    e = torch.empty(0, device=dev)
    gated = bool(op._use_residual_gate)
    al = op._cc_alpha.float().contiguous() if gated else e
    be = op._cc_beta.float().contiguous() if gated else e
    pw = op._cc_post_weight.to(torch.bfloat16).contiguous() if op._post_norm.has_weight else e
    flags = (1 if pw.numel() else 0) | (2 if op._post_norm_is_input_scale else 0) | (4 if gated else 0) | (
        8 if al.numel() == HIDDEN else 0)
    return al, be, pw, flags


def _gdm_run(r, inst2, x, out, stream, mom, pdl):
    r["mod"].gd_run_mom(inst2, r["w"], r["esc"], x, out, x.shape[0], r["K"], r["N"], r["tb"], r["nbx"], 1, pdl, stream,
                        _GDM["al"], _GDM["be"], _GDM["pw"], mom, _GDM["flags"])


def _gdm_prepare(state, t0):
    _GDM["ok"] = set()
    st, sb = state.get("gd"), _SBB1.get("st")
    if not (GDM_BUILD and st is not None and sb is not None and hasattr(sb["mod"], "sbbm_run")):
        print("[gdm] not built (GD / SBB1 missing) -> GD o + SBB1", flush=True)
        return None
    t1 = time.time()
    try:
        _GDM["al"], _GDM["be"], _GDM["pw"], _GDM["flags"] = _gdm_inputs(state)
        op = state["attn_boundary"]
        gen = torch.Generator(device=state["router"].device).manual_seed(6262)
        w = state["dense"]["o"]
        for b in GDM_BUCKETS:
            r = st["route"].get(("o", b))
            if r is None or not sb["ok"].get(b):
                print(f"[gdm] FALLBACK o/{b}: no GD o route or SBB1 bucket -> GD o + SBB1", flush=True)
                continue
            key = _gd_key(*_gd_nk("o", b), GD_ROUTE[("o", b)])
            k2 = (2,) + tuple(key[1:])
            if k2 not in _GD["keys"]:
                print(f"[gdm] FALLBACK o/{b}: OUT 2 instance not built", flush=True)
                continue
            inst2 = _GD["keys"].index(k2)
            good, msg = True, []
            for T in GD_SELF_TEST_T[b]:
                x = torch.randn((T, w.shape[1]), generator=gen, device=w.device).to(torch.bfloat16)
                stream = torch.randn((T, HIDDEN), generator=gen, device=w.device)
                o0 = torch.full((T, HIDDEN), float("nan"), dtype=torch.bfloat16, device=w.device)
                o2 = torch.full((T, HIDDEN), float("nan"), dtype=torch.bfloat16, device=w.device)
                P = r["nbx"]
                mom = torch.full((T * P * 4,), float("nan"), dtype=torch.float32, device=w.device)
                r["mod"].gd_run(r["inst"], r["w"], r["esc"], x, o0, T, r["K"], r["N"], r["tb"], r["nbx"], 1, 0)
                _gdm_run(r, inst2, x, o2, stream, mom, 0)
                rd = torch.float32 if T <= ROUTER_FP32_MAX_TOKENS else None
                ref = op(stream, o0, branch_dtype=torch.bfloat16, router_dtype=rd)
                got = _gdm_sbbm(sb, stream, o2, rd, T, mom, P)
                torch.cuda.synchronize()
                v = o0.double()
                al = _GDM["al"].double() if _GDM["al"].numel() else torch.ones((), dtype=torch.float64, device=w.device)
                be = _GDM["be"].double() if _GDM["be"].numel() else torch.ones((), dtype=torch.float64, device=w.device)
                rr_, sbv = al * stream.double(), be * v
                ref_m = torch.stack([(v * v).sum(1), (rr_ * rr_).sum(1), (rr_ * sbv).sum(1), (sbv * sbv).sum(1)], 1)
                ref_a = torch.stack([(v * v).sum(1), (rr_ * rr_).sum(1), (rr_ * sbv).abs().sum(1), (sbv * sbv).sum(1)], 1)
                gm = mom.view(T, P, 4).double().sum(1)
                mrel = float(((gm - ref_m).abs() / ref_a.clamp(min=1e-30)).max()) if bool(torch.isfinite(gm).all()) else float("inf")
                same = bool(torch.equal(o0, o2))
                e0 = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                e1 = float((got[1] - ref[1]).norm() / ref[1].norm())
                e2 = float((got[2] - ref[2]).norm() / ref[2].norm()) if rd is not None else 0.0
                ok = same and mrel < 1e-4 and e0 < 1e-2 and e1 < 1e-4 and e2 < 1e-2 and bool(torch.isfinite(got[1]).all())
                good &= ok
                msg.append(f"T={T} o-bitwise {same} mom {mrel:.1e} out {e0:.1e} stream {e1:.1e} router {e2:.1e}")
            print(f"[gdm] self-test o/{b} inst2 {inst2}: {'; '.join(msg)} -> {'ok' if good else 'FALLBACK (GD o + SBB1)'}",
                  flush=True)
            if good:
                _GDM["ok"].add(b)
                _GDM["inst2"][b] = inst2
    except Exception as ex:
        _GDM["ok"] = set()
        print(f"[gdm] FAILED -> GD o + SBB1 everywhere: {ex!r}", flush=True)
    print(f"[gdm] live buckets {sorted(_GDM['ok'])} ({time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
    return None


def _gdm_sbbm(sb, stream, branch, router_dtype, tokens, mom, P):
    dev = branch.device
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    ns = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    rt = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev) if router_dtype is not None else sb["e"]
    pw, qw, al, bt, isc, pe, qe = sb["w"]
    sb["mod"].sbbm_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe, isc, mom, P)
    return out, ns, (rt if router_dtype is not None else None)


def _gdm_o(state, x, stream):
    _GDM["pend"] = None
    if not GDM_ENABLE:
        return None
    T = x.shape[0]
    b = _gd_bucket(T)
    if b not in _GDM["ok"] or stream is None or stream.dtype != torch.float32 or not stream.is_contiguous():
        return None
    r = _gd_live(state, "o", T)
    if r is None or not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == r["K"] and x.is_contiguous()
                         and x.data_ptr() % 16 == 0):
        return None
    try:
        out = torch.empty((T, r["N"]), dtype=torch.bfloat16, device=x.device)
        P = r["nbx"]
        mom = torch.empty((T * P * 4,), dtype=torch.float32, device=x.device)
        pdl = (1 if GD_PDL.get("o") else 0) | (2 if GD_TRIG.get("o") else 0)
        _gdm_run(r, _GDM["inst2"][b], x, out, stream, mom, pdl)
        _GDM["pend"] = (out, mom, P)
        return out
    except Exception as ex:
        _GDM["ok"] = set()
        _GDM["pend"] = None
        print(f"[gdm] call failed -> GD o + SBB1 from now on: {ex!r}", flush=True)
        return None


def _gdm_bnd(stream, branch, router_dtype, tokens):
    pend = _GDM["pend"]
    _GDM["pend"] = None
    sb = _SBB1.get("st")
    if not GDM_ENABLE or pend is None or pend[0] is not branch or sb is None:
        return None
    if branch.dtype != torch.bfloat16 or stream.dtype != torch.float32:
        return None
    return _gdm_sbbm(sb, stream, branch, router_dtype, tokens, pend[1], pend[2])

'''


def _replace_literal(text, name, fn):
    head = f"{name} = r'''"
    i = text.index(head)
    if text.count(head) != 1:
        raise ValueError(f"{name}: {text.count(head)} definitions")
    j = text.index("'''", i + len(head))
    body = text[i + len(head):j]
    new = fn(body)
    if "'''" in new:
        raise ValueError(f"{name}: triple quote")
    return text[:i + len(head)] + new + text[j:]


def apply(text):
    lab = _lab()
    cpp_holder = {}

    def patch_cuda(body):
        cuda, cpp = lab.gdm_patch(body, cpp_holder["cpp"])
        cpp_holder["cpp2"] = cpp
        return cuda
    i = text.index("GD_CPP_SRC = r'''") + len("GD_CPP_SRC = r'''")
    cpp_holder["cpp"] = text[i:text.index("'''", i)]
    text = _replace_literal(text, "GD_CUDA_SRC", patch_cuda)
    text = _replace_literal(text, "GD_CPP_SRC", lambda body: cpp_holder["cpp2"])
    text = _replace_literal(text, "SBB1_HOST", lambda body: body + lab.SBBM_CU)
    reps = [
        ("SBB1_DECL = 'void sbb1_run(", "SBB1_DECL = " + repr(lab.SBBM_DECL + " ")[:-1] + "void sbb1_run(", "SBB1_DECL"),
        ('functions=["sbb1_run", "vf_router32"]', 'functions=["sbb1_run", "vf_router32", "sbbm_run"]', "sbb1 functions"),
        ('functions=["gd_run", "gd_info", "gd_unpack"]', 'functions=["gd_run", "gd_info", "gd_unpack", "gd_run_mom"]',
         "gd functions"),
        ("        if key is not None and key not in keys:" + NL + "            keys.append(key)" + NL + "    return keys" + NL,
         "        if key is not None and key not in keys:" + NL + "            keys.append(key)" + NL
         + '        if key is not None and proj == "o" and GDM_BUILD and b in GDM_BUCKETS and (2,) + tuple(key[1:]) not in keys:'
         + NL + "            keys.append((2,) + tuple(key[1:]))" + NL + "    return keys" + NL, "_gd_keys"),
        (NL + NL + "GD_ENABLE = True" + NL, NL + NL + GDM_PY + NL + "GD_ENABLE = True" + NL, "insert before GD"),
        ("    _gd_prepare(state, t0)" + NL, "    _gd_prepare(state, t0)" + NL + "    _gdm_prepare(state, t0)" + NL, "prepare hook"),
        ('    attn_output = _gd(state, "o", o)' + NL,
         '    attn_output = _gdm_o(state, o, data["stream"])' + NL + "    if attn_output is None:" + NL
         + '        attn_output = _gd(state, "o", o)' + NL, "o hook"),
        ("def _v9_bnd(which, stream, branch, router_dtype, state, tokens):" + NL,
         "def _v9_bnd(which, stream, branch, router_dtype, state, tokens):" + NL + '    if which == "attn":' + NL
         + "        gb = _gdm_bnd(stream, branch, router_dtype, tokens)" + NL + "        if gb is not None:" + NL
         + "            return gb" + NL, "_v9_bnd hook"),
    ]
    for old, new, what in reps:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"vdsm anchor {what!r} found {n} times")
        text = text.replace(old, new, 1)
    return text


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    lab_only, dry = "--lab-only" in sys.argv, "--dry" in sys.argv
    n = args["n"]
    base = ROOT / f"kernels/muse_decode_layer/{args['base']}.py"
    out, labf = ROOT / f"kernels/muse_decode_layer/k3_vdsm_{n}.py", ROOT / f"kernels/lab/k3_vdsm_{n}_ab.py"
    for f in ([] if dry else [labf] if lab_only else [out, labf]):
        if f.exists():
            raise SystemExit(f"[vdsm_patches] refusing to overwrite {f.relative_to(ROOT)} (use a new --n)")
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{base.name} is not a finish() fixed point")
    tag = f"k3-vdsm-{n}"
    nonce = (f"# nonce {tag} ({base.stem} + GDM: attention-boundary moments from GD o's epilogue + SBBM at T 1 / 8 / 16, "
             f"self-tested with fallback; GDM_ENABLE)")
    t = apply(b.replace(bn, nonce))
    rep = [f"base {base.name} {len(b.encode())} B is a finish() fixed point"]
    s = GM.finish(t, nonce, tag, chk_base, rep)
    rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B (headroom {1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[vdsm_patches] OK  {r}")
    if dry:
        if args.get("dryout"):
            pathlib.Path(args["dryout"]).write_text(s, encoding="utf-8", newline=NL)
        print(f"[vdsm_patches] --dry: no repo output written ({len(s.encode())} B)")
        return 0
    if not lab_only:
        out.write_text(s, encoding="utf-8", newline=NL)
    labf.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vdsm_patches] wrote {'' if lab_only else str(out.relative_to(ROOT)) + ' and '}{labf.relative_to(ROOT)} "
          f"({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

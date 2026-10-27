"""vd_patches.py: VD1 (notes/study/vd_dense.md) as the layer's dense route at buckets 128 / 256 (verify16 / verify32): SS-wgmma
swap-AB with all tokens per CTA, pre-swizzled front-major weight tiles streamed by 1-D bulk copies, DSMEM cluster split-K, PDL.

Applied to a finish()-fixed-point base (default k3_va_3): the kernel source comes from kernels/lab/vd_lab.py (vd_cuda_src of
the route instances, expanded here), the Python glue is ours. Hooks (each anchor asserted unique): VD build joins PT1's
threaded prefetch; _vd_prepare (pack + self-test every route vs FP64 and today's cuBLAS, twice, err 0; fallback loudly)
runs after _dg1_prepare; _dg1 tries _vd first (so qkv / o / cat / unpooler at T >= 49 get VD where a route is live).
VD_ENABLE and VD_ROUTE are read at call time (one file serves every A/B arm).

    python -I tools/vd_patches.py [--base=k3_va_3] [--n=1] [--routes=qkv:128:2,...] [--lab-only]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "kernels" / "lab"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)

# (proj, bucket) -> (NWG, BN, NST, CL, F32); filled from R-vd-1's per-shape winners
DEFAULT_ROUTES = {
    ("qkv", 128): (1, 128, 9, 3, 0),
    ("o", 128): (2, 128, 7, 2, 0),
    ("cat", 128): (2, 128, 7, 3, 1),
    ("unpooler", 128): (2, 128, 7, 2, 0),
}


def parse_routes(spec):
    """'qkv:128:1-128-9-3-0,o:128:2-128-7-2-0' -> {(proj, bucket): inst}"""
    out = {}
    for item in spec.split(","):
        if not item:
            continue
        p, b, inst = item.split(":")
        out[(p, int(b))] = tuple(int(v) for v in inst.split("-"))
    return out


PY_TMPL = r'''

VD_ENABLE = True
VD_PDL = True
VD_TRIG = True
VD_TOL = (1.0e-2, 1.0e-4, 1.0e-2)
VD_INSTS = @INSTS@
VD_ROUTE = @ROUTE@
VD_CPP_SRC = @CPP@
VD_CUDA_SRC = @CUDA@
_VD = {"mod": None, "build_s": 0.0}


def _vd_build():
    if _VD["mod"] is not None:
        return _VD["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((VD_CPP_SRC + VD_CUDA_SRC).encode()).hexdigest()[:10]
    t1 = time.time()
    _VD["mod"] = load_inline(name=f"vd_{tag}", cpp_sources=VD_CPP_SRC, cuda_sources=VD_CUDA_SRC, functions=["vd_run", "vd_info"],
                             extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _VD["build_s"] = time.time() - t1
    return _VD["mod"]


def _vd_pack(w, bm):
    n, k = w.shape
    nrb, nks = n // bm, k // 64
    t = w.contiguous().view(nrb, bm, nks, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bm, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7)).view(1, 1, bm, 8, 1).expand(nrb, nks, bm, 8, 8)
    t = torch.gather(t, 3, idx).permute(1, 0, 2, 3, 4).contiguous()
    back = torch.gather(t, 3, idx.permute(1, 0, 2, 3, 4)).permute(1, 2, 0, 3, 4).reshape(n, k)
    if not torch.equal(back.view(torch.int16), w.contiguous().view(torch.int16)):
        raise RuntimeError("VD pack is not bit-exact")
    return t.view(torch.uint8).view(-1), bm * 128, nrb * bm * 128


def _vd_call(st, r, x, out):
    flat, w_rb, w_s = st["w"][(r["proj"], r["bm"])]
    st["mod"].vd_run(r["kid"], x, flat, flat, out, st["err"], st["stamps"], r["nrb"], w_rb, w_s,
                     128 if VD_TRIG else 0, 1 if VD_PDL else 0)


def _vd_selftest(st, r, w, T, gen):
    f32 = r["f32"]
    dev = w.device
    ok, parts = True, []
    for t in sorted({T, T - 3, max(1, T // 2 + 1)}, reverse=True):
        x = torch.randn((t, r["K"]), generator=gen, device=dev).to(torch.bfloat16)
        outs = []
        for _ in range(2):
            out = torch.full((t, r["N"]), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
            _vd_call(st, r, x, out)
            outs.append(out)
        ref64 = x.double() @ w.double().t()
        today = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        torch.cuda.synchronize()
        got = outs[0]
        fin = bool(torch.isfinite(got).all())
        e64 = float((got.double() - ref64).norm() / ref64.norm()) if fin else float("inf")
        etoday = float((got.float() - today.float()).norm() / today.float().norm()) if fin else float("inf")
        same = bool(torch.equal(outs[0], outs[1]))
        errf = int(st["err"][0])
        good = fin and same and errf == 0 and e64 < VD_TOL[1 if f32 else 0] and etoday < VD_TOL[2]
        ok &= good
        parts.append(f"T={t}: rel vs fp64 {e64:.2e} vs today {etoday:.2e} finite {fin} repeat {same} err {errf}")
        st["err"].zero_()
    return ok, "; ".join(parts)


def _vd_prepare(state, t0):
    state["vd"] = None
    if not VD_ENABLE or not VD_ROUTE:
        return None
    dev = state["dense"]["qkv_gate"].device
    if torch.cuda.get_device_capability(dev)[0] < 9:
        print("[vd] not sm_90: VD off", flush=True)
        return None
    t1 = time.time()
    _pt1_wait("vd")
    try:
        mod = _vd_build()
    except Exception as ex:
        print(f"[vd] build FAILED -> dense stays on today's path: {ex!r}", flush=True)
        return None
    st = {"mod": mod, "err": torch.zeros((4,), dtype=torch.int32, device=dev),
          "stamps": torch.zeros((4,), dtype=torch.int64, device=dev), "w": {}, "route": {}}
    gen = torch.Generator(device=dev).manual_seed(4242)
    for (proj, b), kid in sorted(VD_ROUTE.items()):
        nwg, bn, nst, cl, f32 = VD_INSTS[kid]
        w = _dg1_weight(state, proj)
        if w is None or w.dtype != torch.bfloat16 or w.shape[0] % (64 * nwg) or w.shape[1] % 64:
            print(f"[vd] {proj}/{b}: no weight -> today's path", flush=True)
            continue
        bm = 64 * nwg
        try:
            if (proj, bm) not in st["w"]:
                st["w"][(proj, bm)] = _vd_pack(w, bm)
            r = {"kid": kid, "proj": proj, "bm": bm, "nrb": w.shape[0] // bm, "N": w.shape[0], "K": w.shape[1], "T": bn,
                 "f32": bool(f32)}
            good, msg = _vd_selftest(st, r, w, min(bn, b), gen)
        except Exception as ex:
            good, msg = False, repr(ex)[:300]
        print(f"[vd] {proj}/{b} inst {VD_INSTS[kid]}: {msg} -> {'LIVE' if good else 'FAILED: today path'}", flush=True)
        if good:
            st["route"][(proj, b)] = r
    state["vd"] = st if st["route"] else None
    print(f"[vd] routes live {sorted(st['route'])} (build {_VD['build_s']:.0f}s, prepare {time.time() - t1:.1f}s)", flush=True)
    return state["vd"]


def _vd(state, proj, x):
    st = state.get("vd")
    if not VD_ENABLE or st is None:
        return None
    T = x.shape[0]
    b = _k2a_bucket_of(T)
    if VD_ROUTE.get((proj, b)) is None:
        return None
    r = st["route"].get((proj, b))
    if (r is None or T > r["T"] or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != r["K"] or not x.is_contiguous()
            or x.data_ptr() % 16 != 0):
        return None
    try:
        out = torch.empty((T, r["N"]), dtype=torch.float32 if r["f32"] else torch.bfloat16, device=x.device)
        _vd_call(st, r, x, out)
        return out
    except Exception as ex:
        _k3_disable(state, "vd", None, "VD", ex)
        return None

'''


def _lab():
    """vd_lab's source strings and generator functions, read with ast (no torch import needed here)."""
    import ast
    import types
    tree = ast.parse((ROOT / "kernels/lab/vd_lab.py").read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("VD_CPP_SRC", "VD_CUDA_SRC"):
            ns[node.targets[0].id] = ast.literal_eval(node.value)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("vd_wg_specs", "vd_cuda_src"):
            if node.name == "vd_cuda_src":
                node.args.defaults = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), "vd_lab", "exec"), ns)
    return types.SimpleNamespace(**ns)


def layer_code(routes):
    L = _lab()
    insts = sorted(set(routes.values()))
    route = {k: insts.index(v) for k, v in sorted(routes.items())}
    cuda = L.vd_cuda_src(insts)
    for bad in ("'''", '"""'):
        assert bad not in cuda and bad not in L.VD_CPP_SRC
    code = (PY_TMPL.replace("@INSTS@", repr(insts)).replace("@ROUTE@", repr(route))
            .replace("@CPP@", 'r"""' + L.VD_CPP_SRC + '"""').replace("@CUDA@", 'r"""' + cuda + '"""'))
    return code, insts, route


ANCHOR_DEF = "def _dg1(state, proj, x):\n"
ANCHOR_PT1 = '        jobs.append(("vms", _vms_build))\n'
ANCHOR_PREP = "    _dg1_prepare(state, t0)\n"


def apply(text, routes):
    code, insts, route = layer_code(routes)
    for a in (ANCHOR_DEF, ANCHOR_PT1, ANCHOR_PREP):
        assert text.count(a) == 1, f"anchor not unique: {a!r}"
    hook = ("def _dg1(state, proj, x):\n    vd_out = _vd(state, proj, x)\n    if vd_out is not None:\n        return vd_out\n")
    text = text.replace(ANCHOR_DEF, code.lstrip(NL) + NL + NL + hook)
    text = text.replace(ANCHOR_PT1, ANCHOR_PT1 + '    if VD_ENABLE and VD_ROUTE:\n        jobs.append(("vd", _vd_build))\n')
    text = text.replace(ANCHOR_PREP, ANCHOR_PREP + "    _vd_prepare(state, t0)\n")
    return text


def main():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_va_3')}.py"
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{base.name} is not a finish() fixed point")
    n = ARGS.get("n", "1")
    routes = parse_routes(ARGS["routes"]) if "routes" in ARGS else DEFAULT_ROUTES
    tag = f"k3-vd-{n}"
    desc = ", ".join(f"{p}@{bk}" for p, bk in sorted(routes))
    nonce = f"# nonce {tag} ({base.stem} + VD1 dense SS-wgmma routes {desc}, self-tested vs FP64 / cuBLAS with fallback; VD_ENABLE)"
    t = apply(b.replace(bn, nonce), routes)
    rep = [f"base {base.name} {len(b.encode())} B is a finish() fixed point"]
    lab_only = "--lab-only" in sys.argv
    if lab_only:
        chk0 = GM.check
        GM.check = lambda ok, msg: chk0(ok or "file >= 1 MB" in msg, msg)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B (headroom {1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[vd_patches] OK  {r}")
    out, lab = ROOT / f"kernels/muse_decode_layer/k3_vd_{n}.py", ROOT / f"kernels/lab/k3_vd_{n}_ab.py"
    if not lab_only:
        out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vd_patches] wrote {'' if lab_only else str(out.relative_to(ROOT)) + ' and '}{lab.relative_to(ROOT)} "
          f"({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""vds_patches.py: VDS2 dense routes at T = 9..32 (RS-wgmma on lossless P12 weights; kernels/lab/vds_lab.py is the one
source of the CUDA text and the pack) on a stripped layer base (finish() fixed point), VDS builder (notes/study/vds_dense.md).

What it adds (call time, first in each chain; any failure falls through to the base's route):
  * qkv / o / unpooler (BF16 out) and the cat (pooler + router hi / lo rows, FP32 out into _v8_tail's cat mode) at the
    buckets in VDS_ROUTE = {(proj, 16 | 32): (inst, S, pf)}; inst indexes VDS_INSTS (N, NWG, NST, NSTGM, OUT).
  * prepare: one PT1 build job (only the used instances), per route a pack + bit-exact emulated-decode check, a persistent
    zeroed split-K workspace + counters, and GD's self-test recipe (FP64 -> BF16 reference, worst row, exact probe rows,
    outlier channels, NaN-filled outputs, plain and PDL launches, bitwise repeat, workspace clean after each call);
    a failing route prints FALLBACK and the base route stays.
  * VDS_ENABLE is read at call time (A/B arms); False = the base exactly.

    python -I tools/vds_patches.py --base=k3_slim_3 --n=1 --routes='{"cat,32": [2, 7, 0], ...}' [--lab-only]
Writes kernels/muse_decode_layer/k3_vds_<n>.py (unless --lab-only) and the byte-identical box copy
kernels/lab/k3_vds_<n>_ab.py; refuses to overwrite an existing output.
"""
import hashlib
import importlib.util
import inspect
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
NL = chr(10)


def _lab():
    spec = importlib.util.spec_from_file_location("vds_lab", str(ROOT / "kernels/lab/vds_lab.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def slim(src):
    """layer copy of the CUDA text: drop the `// @lab{` .. `// @lab}` regions (stamps, diagnostics, null kernel)."""
    out, skip = [], 0
    for line in src.split(NL):
        tag = line.strip()
        if tag == "// @lab{":
            skip += 1
            continue
        if tag == "// @lab}":
            skip -= 1
            continue
        if not skip:
            out.append(line)
    assert skip == 0, "unbalanced @lab markers"
    s = NL.join(out)
    assert "@lab" not in s
    return s


PACK_FNS = ["_i32", "vds_frag", "vds_window_base", "vds_splits", "vds_bases", "vds_tbs", "vds_pack", "vds_decode_check"]


def pack_source(lab):
    """the lab's pack functions verbatim, renamed into the layer's _vds_ namespace."""
    src = NL.join(inspect.getsource(getattr(lab, f)) for f in PACK_FNS)
    for f in PACK_FNS:
        new = "_vds_i32" if f == "_i32" else "_" + f
        src = re.sub(r"\b" + re.escape(f) + r"\b", new, src)
    src = src.replace("ESC_CAP", "VDS_ESC_CAP")
    return src


VDS_PY = r'''
VDS_ENABLE = True
VDS_BUILD = True
VDS_PREPARE_BY_S = 240.0
VDS_ESC_CAP = 64
VDS_INSTS = @INSTS@
VDS_ROUTE = @ROUTE@
VDS_PDL = {"qkv": False, "o": True, "cat": True, "unpooler": True}
VDS_TRIG = True
VDS_SELF_TEST_T = {16: (16, 9), 32: (32, 17)}
VDS_TOL = (2e-3, 1e-2, 1e-5)   # bf16 outputs: rel L2 / worst row vs FP64 -> BF16; f32 (cat): rel L2 vs FP64
_VDS = {"mod": None, "build_s": 0.0}
_VDS_SEEN = set()
VDS_CPP_SRC = @CPP@
VDS_CUDA_SRC = @CUDA@


def _vds_bucket(T):
    return 16 if 9 <= T <= 16 else 32 if 17 <= T <= 32 else None


def _vds_build():
    _pt1_wait("vds")
    if _VDS["mod"] is not None:
        return _VDS["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((VDS_CPP_SRC + VDS_CUDA_SRC).encode()).hexdigest()[:10]
    t1 = time.time()
    _VDS["mod"] = load_inline(name=f"vds_{tag}", cpp_sources=VDS_CPP_SRC, cuda_sources=VDS_CUDA_SRC,
                              functions=["vds_run", "vds_info"],
                              extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _VDS["build_s"] = time.time() - t1
    return _VDS["mod"]


@PACKSRC@


def _vds_weight(state, proj):
    if proj == "cat":
        return state["v8"]["wcat"] if state.get("v8") is not None else None
    return state["dense"]["qkv_gate" if proj == "qkv" else proj]


def _vds_launch(r, x, out, pdl, pf):
    r["mod"].vds_run(r["kid"], r["pk"]["w"], r["pk"]["esc"], x, out, r["err"], r["err"], out.stride(0), r["pk"]["nrg"],
                     r["pk"]["tbs"], pf, 1 if VDS_TRIG else 0, 0, 1 if pdl else 0)


def _vds_selftest(r, w, b, proj):
    n, k = w.shape
    dev = w.device
    f32 = proj == "cat"
    parts = []
    for T in VDS_SELF_TEST_T[b]:
        for outl in (False, True):
            x, probes = _gd_test_input(T, k, 1311 + 37 * T + int(outl), dev, outl)
            outs = []
            for pdl in (0, 1, 1):
                o = torch.full((T, n), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
                if pdl:
                    o.fill_(float("nan"))   # a predecessor kernel for the PDL launch
                _vds_launch(r, x, o, pdl, r["pf"])
                outs.append(o)
            ref = x.double() @ w.double().T
            torch.cuda.synchronize()
            got = outs[0].double()
            if not bool(torch.isfinite(got).all()):
                return False, f"T={T} outl={outl}: non-finite"
            if int(r["err"][0]) != 0:
                return False, f"T={T} outl={outl}: bounded wait expired (err {int(r['err'][0])})"
            rf = ref if f32 else ref.to(torch.bfloat16).double()
            dd = got - rf
            rn = rf.norm(dim=-1)
            tot = float(dd.norm() / rf.norm().clamp(min=1e-30))
            row = float((dd.norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
            pr = float((dd[probes].norm(dim=-1) / rn[probes].clamp(min=1e-30)).max()) if (bool(probes.any()) and not f32) else 0.0
            rep = all(bool(torch.equal(outs[0], o)) for o in outs[1:])
            ok = (tot <= VDS_TOL[2] and rep) if f32 else (tot <= VDS_TOL[0] and row <= VDS_TOL[1] and pr == 0.0 and rep)
            if not ok:
                return False, f"T={T} outl={outl}: rel {tot:.1e} row {row:.1e} probe {pr:.1e} repeat {rep}"
            parts.append(f"T={T}{'o' if outl else ''} {tot:.1e}")
    return True, " ".join(parts)


def _vds_prepare(state, t0):
    state["vds"] = None
    if not (VDS_BUILD and VDS_ROUTE):
        return None
    if time.time() - t0 > VDS_PREPARE_BY_S:
        print(f"[vds] skipped: prepare already took {time.time() - t0:.1f}s (> {VDS_PREPARE_BY_S}s): base routes", flush=True)
        return None
    t1 = time.time()
    try:
        mod = _vds_build()
    except Exception as ex:
        print(f"[vds] build FAILED -> base routes everywhere: {str(ex)[-2000:]}", flush=True)
        return None
    routes, live, packs = {}, [], {}
    for (proj, b), (kid, S, pf) in sorted(VDS_ROUTE.items(), key=str):
        why = None
        w = _vds_weight(state, proj)
        nn, nwg, nst, nstgm, outf, cl = VDS_INSTS[kid]
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
            why = "no bf16 weight"
        elif nn < b or (proj == "cat") != (outf == 1) or S != cl or w.shape[0] % (64 * nwg) or w.shape[1] % 64:
            why = f"instance {VDS_INSTS[kid]} does not fit {proj}/{b} {tuple(w.shape)}"
        else:
            info = [int(v) for v in mod.vds_info(kid)]
            nrg = w.shape[0] // (64 * nwg)
            sms = torch.cuda.get_device_properties(w.device).multi_processor_count
            if info[2] > 0 or info[3] < 1 or nrg * S > sms or (S > 1 and info[7] < nrg):
                why = f"regs {info[1]} local {info[2]} occupancy {info[3]} clusters {info[7]} for {nrg} x {S} CTAs (> one wave)"
        if why is None:
            key = (proj, nwg, S, nstgm)
            if key not in packs:
                try:
                    seg = [4096, 4352] if proj == "cat" else None
                    pk = _vds_pack(w, nwg, S, nstgm, _vds_bases(w, nwg, seg))
                    exact = _vds_decode_check(w, pk)
                    print(f"[vds] pack {proj} {tuple(w.shape)} nwg {nwg} S {S}: {pk['w'].numel() * 4 / 1e6:.1f} MB, bases "
                          f"{sorted(set(pk['bases']))}, {pk['nesc']} escapes, bit-exact decode {exact}", flush=True)
                    packs[key] = pk if exact else None
                except Exception as ex:
                    print(f"[vds] pack {proj} nwg {nwg} S {S} failed: {ex!r}", flush=True)
                    packs[key] = None
                torch.cuda.empty_cache()
            if packs[key] is None:
                why = "pack / bit-exact decode failed"
        if why is None:
            pk = packs[key]
            dev = w.device
            r = {"mod": mod, "kid": kid, "S": S, "pf": pf, "pk": pk, "N": w.shape[0], "K": w.shape[1], "T": nn,
                 "f32": outf == 1, "err": torch.zeros((4,), dtype=torch.int32, device=dev),
                 "desc": f"i{kid}{VDS_INSTS[kid]}s{S}pf{pf}"}
            try:
                ok, msg = _vds_selftest(r, w, b, proj)
            except Exception as ex:
                ok, msg = False, f"EXC {ex!r}"
            if not ok:
                why = f"self-test: {msg}"
            else:
                routes[(proj, b)] = r
                live.append(f"{proj}/{b} {r['desc']}")
                print(f"[vds] self-test {proj}/{b} {r['desc']}: ok ({msg})", flush=True)
        if why is not None:
            print(f"[vds] FALLBACK {proj}/{b}: {why} -> base route", flush=True)
    state["vds"] = {"route": routes} if routes else None
    print(f"[vds] live ({time.time() - t1:.1f}s, build {_VDS['build_s']:.0f}s; prepare so far {time.time() - t0:.1f}s): "
          f"{'; '.join(live) or 'none'}", flush=True)
    return None


def _vds(state, proj, x):
    st = state.get("vds")
    if not VDS_ENABLE or st is None:
        return None
    T = x.shape[0]
    b = _vds_bucket(T)
    ent = VDS_ROUTE.get((proj, b)) if b is not None else None   # read at call time: A/B arms may drop routes / change pf
    r = st["route"].get((proj, b)) if ent else None
    if r is None or T > r["T"] or not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == r["K"] and x.is_contiguous()
                                       and x.data_ptr() % 16 == 0):
        return None
    try:
        out = torch.empty((T, r["N"]), dtype=torch.float32 if r["f32"] else torch.bfloat16, device=x.device)
        _vds_launch(r, x, out, VDS_PDL.get(proj, False), int(ent[2]))
        if (proj, T) not in _VDS_SEEN:
            _VDS_SEEN.add((proj, T))
            print(f"[vds] first call {proj} T={T}: {r['desc']}", flush=True)
        return out
    except Exception as ex:
        _k3_disable(state, "vds", None, "VDS", ex)
        return None

'''


def apply(text, routes, insts):
    lab = _lab()
    cu = slim(lab.vds_cuda_src(insts))
    cpp = lab.VDS_CPP_SRC
    for s_ in (cu, cpp):
        if "'''" in s_:
            raise ValueError("triple quote in CUDA text")
    py = (VDS_PY.replace("@INSTS@", repr([tuple(i) for i in insts])).replace("@ROUTE@", repr(routes))
          .replace("@CPP@", "r'''" + cpp + "'''").replace("@CUDA@", "r'''" + cu + "'''")
          .replace("@PACKSRC@", pack_source(lab)))
    reps = [
        (NL + NL + "GD_ENABLE = True" + NL, NL + NL + py + NL + "GD_ENABLE = True" + NL, "insert before GD"),
        ("    if GD_BUILD:" + NL + '        jobs.append(("gd", _gd_build))' + NL,
         "    if GD_BUILD:" + NL + '        jobs.append(("gd", _gd_build))' + NL
         + "    if VDS_BUILD and VDS_ROUTE:" + NL + '        jobs.append(("vds", _vds_build))' + NL, "pt1 job"),
        ("    _gd_prepare(state, t0)" + NL, "    _gd_prepare(state, t0)" + NL + "    _vds_prepare(state, t0)" + NL, "prepare hook"),
        ('    qkv = _gd(state, "qkv", data["attn_input"])' + NL,
         '    qkv = _vds(state, "qkv", data["attn_input"])' + NL + "    if qkv is None:" + NL
         + '        qkv = _gd(state, "qkv", data["attn_input"])' + NL, "qkv hook"),
        ('    attn_output = _gd(state, "o", o)' + NL,
         '    attn_output = _vds(state, "o", o)' + NL + "    if attn_output is None:" + NL
         + '        attn_output = _gd(state, "o", o)' + NL, "o hook"),
        ('        y32 = _v20_p1m(state, "cat", ffn_input)' + NL,
         '        y32 = _vds(state, "cat", ffn_input)' + NL + "        if y32 is None:" + NL
         + '            y32 = _v20_p1m(state, "cat", ffn_input)' + NL, "cat hook"),
        ("    ffn_output = _gd_unpool(state, expert_output, tokens)" + NL,
         '    ffn_output = _vds(state, "unpooler", expert_output) if tokens > SLICED_UNPOOL_MAX_TOKENS else None' + NL
         + "    if ffn_output is None:" + NL + "        ffn_output = _gd_unpool(state, expert_output, tokens)" + NL,
         "unpooler hook"),
    ]
    for old, new, what in reps:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"vds anchor {what!r} found {n} times")
        text = text.replace(old, new, 1)
    return text


def _routes(js):
    out = {}
    for k, v in json.loads(js).items():
        p, b = k.split(",")
        out[(p, int(b))] = tuple(int(t) for t in v)
    return out


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    lab_only = "--lab-only" in sys.argv
    dry = "--dry" in sys.argv
    n = args["n"]
    base = ROOT / f"kernels/muse_decode_layer/{args['base']}.py"
    routes = _routes(args.get("routes", "{}"))
    lab = _lab()
    used = sorted({kid for kid, _, _ in routes.values()})
    remap = {kid: i for i, kid in enumerate(used)}
    insts = [lab.VDS_INSTS[k] for k in used]
    routes = {k: (remap[v[0]], v[1], v[2]) for k, v in routes.items()}
    out, labf = ROOT / f"kernels/muse_decode_layer/k3_vds_{n}.py", ROOT / f"kernels/lab/k3_vds_{n}_ab.py"
    for f in ([] if dry else [labf] if lab_only else [out, labf]):
        if f.exists():
            raise SystemExit(f"[vds_patches] refusing to overwrite {f.relative_to(ROOT)} (use a new --n)")
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{base.name} is not a finish() fixed point")
    tag = f"k3-vds-{n}"
    desc = ", ".join(f"{p}/{bb}:i{v[0]}s{v[1]}pf{v[2]}" for (p, bb), v in sorted(routes.items())) or "none"
    nonce = f"# nonce {tag} ({base.stem} + VDS RS-wgmma P12 dense routes [{desc}], self-tested with fallback; VDS_ENABLE)"
    t = apply(b.replace(bn, nonce), routes, insts)
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
        print(f"[vds_patches] OK  {r}")
    if dry:
        if args.get("dryout"):
            pathlib.Path(args["dryout"]).write_text(s, encoding="utf-8", newline=NL)
        print(f"[vds_patches] --dry: no repo output written ({len(s.encode())} B){' -> ' + args['dryout'] if args.get('dryout') else ''}")
        return 0
    if not lab_only:
        out.write_text(s, encoding="utf-8", newline=NL)
    labf.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vds_patches] wrote {'' if lab_only else str(out.relative_to(ROOT)) + ' and '}{labf.relative_to(ROOT)} "
          f"({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

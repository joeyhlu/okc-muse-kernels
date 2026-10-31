"""sk2lo_patches.py (R-sk2-2, sk2 builder): alygg77's sk2 for qkv / unpooler at T = 16 / 8 / 1 as in-layer arms on the R-sk2-1 ship
form (k3_hh_5_sk2.py: sk2 qkv + unpooler at T = 32 stays on in every arm).

  python -I tools/sk2lo_patches.py [--base=kernels/lab/k3_hh_5_sk2.py] [--out=kernels/lab/k3_sk2lo_ab.py] [--check]

Their picks (bfp_SK2_PICK) at tokr 16: qkv16_r176, unpooler16h_r100; at tokr 8 (m <= 8, incl. m = 1; packed x): qkv8_r176,
slice8_r192 (unpooler rows 0..1023 = our UNPOOL_SLICE at T <= 8). Second load_inline module (verbatim lab text of
tools/sk2_patches.py + their bfp__sk2_src dispatch for these four picks, sk2f dropped) with one edit: the output row stride may
be passed in the unused `dg` argument for TOKR == 8 instances (`(TOKR == 8 && dg > 0 ? dg : N)`; folds to N for every other
instance), so slice8 writes straight into our zero buffer zb[:T] (ldo HIDDEN) like GD's sliced unpooler.
Arms: SK2L_ON = {(proj, T): 1} (proj qkv / unpooler, T 1 / 8 / 16), SK2L_PDL = {"qkv": 0, "unpooler": 2}. Default {} = base.
Their layer's qkv at m 2..8 is a narrow-escape variant (_de_source); not tested here (the plain pick is).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import sk2_patches as P  # noqa: E402
sys.argv = _argv
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
PICK_LO = {("qkv_gate", 16): "qkv16_r176", ("unpooler", 16): "unpooler16h_r100", ("qkv_gate", 8): "qkv8_r176",
           ("unpooler", 8): "slice8_r192"}
STRIDE = ("OutT* o = out + (size_t)tt * N + cta * R + row;\n",
          "OutT* o = out + (size_t)tt * (TOKR == 8 && dg > 0 ? dg : N) + cta * R + row;\n")


def cuda_lo(consts, fns):
    ns = {"bfp_SK2_CONFIGS": consts["bfp_SK2_CONFIGS"], "bfp_SK2_PICK": dict(PICK_LO), "bfp_SK2_K": consts["bfp_SK2_K"],
          "bfp_SK2_SRC": consts["bfp_SK2_SRC"]}
    for name in ("bfp__sk2_geom", "bfp__sk2_src"):
        exec(compile(ast.Module(body=[fns[name]], type_ignores=[]), f"<{name}>", "exec"), ns)
    sk2 = ns["bfp__sk2_src"](True)
    lines = sk2.split("\n")
    fl = [ln for ln in lines if "launch_sk2<" in ln and ", true>(Ws, x, out, ptab" in ln]
    P.check(len(fl) == 3, "sk2f lines")
    sk2 = "\n".join(ln for ln in lines if ln not in fl)
    sd = [ln for ln in sk2.split("\n") if ln.lstrip().startswith("if (cfg == ")]
    P.check(len(sd) == 4, f"lo dispatch {sd}")
    common = consts["bfp_CUDA_SRC"].replace("@@MBAR@@", consts["bfp_MBAR_HELPERS"]).replace("@@BODY@@", "")
    common = P.cut(common, "// cross-CTA spins are bounded", "__device__ __forceinline__ void okc_spin_ge", "spin comment + trap")
    a = common.index("__device__ __forceinline__ void okc_spin_ge")
    b = common.index("\n}\n", a) + 3
    common = common[:a] + common[b:]
    mp = consts["bfp_MAP_SRC"]
    mp = P.cut(mp, "__device__ __forceinline__ void tma_load_2d_pol(", "__device__ __forceinline__ void tma_load_3d(", "2d loads")
    a = mp.index("static CUtensorMap make_map_2d(")
    b = mp.index("\n}\n", a) + 3
    mp = mp[:a] + mp[b:]
    t = P.slim_cuda(common + mp + sk2)
    P.check(t.count(STRIDE[0]) == 1, "stride anchor")
    t = t.replace(STRIDE[0], STRIDE[1])
    P.check("trap" not in t and '"""' not in t and t.isascii(), "lo CUDA text")
    cpp = consts["bfp_CPP_SRC"]
    cpp = "#include <torch/extension.h>\n" + cpp[cpp.index("void sk2("):cpp.index("void sk2f(")] + "void sk2_set_pdl(int64_t n);\n"
    return t, cpp


LAB = r'''"""[SK2L] R-sk2-2 lab (tools/sk2lo_patches.py): alygg77 sk2 qkv / unpooler at T = 16 / 8 / 1 (their tokr 16 / 8 picks; second
module, Apache-2.0 via _SK2_NOTICE); SK2L_ON = {(proj, T): 1}. Defaults: base dispatch."""
_SK2L_CUDA = r"""@CUDA@"""
_SK2L_CPP = r"""@CPP@"""
SK2L_ON = {}
SK2L_PDL = {"qkv": 0, "unpooler": 2}
_SK2L_CFGS = @CFGS@
_SK2L_PICK = {("qkv", 16): "qkv16_r176", ("unpooler", 16): "unpooler16h_r100", ("qkv", 8): "qkv8_r176", ("unpooler", 8): "slice8_r192"}
_SK2L = {"mod": None, "err": None, "secs": 0.0, "ok": {}, "w": {}, "calls": {}}
def _sk2l_build():
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    try:
        tag = hashlib.sha1((_SK2L_CPP + _SK2L_CUDA + " ".join(_SK2_FLAGS)).encode()).hexdigest()[:12]
        kw = dict(cpp_sources=_SK2L_CPP, cuda_sources=_SK2L_CUDA, functions=["sk2", "sk2_set_pdl"], extra_cuda_cflags=list(_SK2_FLAGS),
                  verbose=False)
        try:
            _SK2L["mod"] = load_inline(name=f"sk2l_{tag}", no_implicit_headers=True, **kw)
        except TypeError:
            _SK2L["mod"] = load_inline(name=f"sk2l_{tag}i", **kw)
    except Exception as ex:
        _SK2L["err"] = repr(ex)[-600:]
    _SK2L["secs"] = time.time() - t0
def _sk2l_launch(name, x, out, ldo, pdl):
    ci, stream, tab, pmax, b4, n_rows, k = _SK2L["w"][name]
    m = x.shape[0]
    mod = _SK2L["mod"]
    mod.sk2_set_pdl(pdl)
    mod.sk2(stream, x, out, tab, pmax, b4, m, n_rows, k, ci, 8 if m <= 8 else 16 if m <= 16 else 32, ldo, _SK2["dummy"])
    return out
def _sk2l(state, proj, x):
    m = x.shape[0]
    if not SK2L_ON.get((proj, m)):
        return None
    name = _SK2L_PICK.get((proj, 8 if m <= 8 else 16 if m <= 16 else 32))
    if name is None or not _SK2L["ok"].get(name):
        return None
    k = _SK2L["w"][name][6]
    if not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == k and x.is_contiguous() and x.data_ptr() % 16 == 0):
        return None
    try:
        if name == "slice8_r192":
            zb = state.get("v12_zbuf")
            if not (V12_ZBUF and R1_UNPOOL_INTO_ZEROS and zb is not None and zb.dtype == x.dtype and m <= zb.shape[0]):
                return None
            o = _sk2l_launch(name, x, zb[:m], HIDDEN, SK2L_PDL.get(proj, 0))
        else:
            o = _sk2l_launch(name, x, torch.empty((m, _SK2L["w"][name][5]), dtype=torch.bfloat16, device=x.device), 0, SK2L_PDL.get(proj, 0))
        _SK2L["calls"][(proj, m)] = _SK2L["calls"].get((proj, m), 0) + 1
        return o
    except Exception as ex:
        _SK2L["ok"][name] = False
        print(f"[sk2l] {proj} T={m} raised {ex!r}: off", flush=True)
        return None
def _sk2l_prepare(state, t0):
    _SK2L["ok"], _SK2L["w"] = {}, {}
    t1 = time.time()
    try:
        _pt1_wait("sk2l")
        if _SK2L["mod"] is None and _SK2L["err"] is None:
            _sk2l_build()
        if _SK2L["mod"] is None or _SK2["dummy"] is None:
            print(f"[sk2l] OFF: build {_SK2L['err']} / sk2 dummy {_SK2['dummy'] is not None}", flush=True)
            return
        dense = state["dense"]
        dev = dense["qkv_gate"].device
        gen = torch.Generator(device=dev)
        gen.manual_seed(20261101)
        comps, res = {}, {}
        for name, (ci, layer, n_rows, R, NT, NWR, WSK, UPW) in _SK2L_CFGS.items():
            w = dense[layer].contiguous()
            if layer not in comps:
                comps[layer] = _sk2_compress(w)
            comp = comps[layer]
            if comp is None:
                continue
            tab, pmax = _sk2_table(comp, n_rows, R)
            k = w.shape[1]
            _SK2L["w"][name] = (ci, _sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax, comp["base4"], n_rows, k)
            ok = True
            for m in ((16, 9) if name in ("qkv16_r176", "unpooler16h_r100") else (8, 5, 1)):
                x = (torch.randn((m, k), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
                ref = x.float() @ w[:n_rows].float().t()
                ldo = HIDDEN if name == "slice8_r192" else 0
                outs = [_sk2l_launch(name, x, torch.zeros((m, HIDDEN if ldo else n_rows), dtype=torch.bfloat16, device=dev), ldo, p)
                        for p in (0, 2, 2)]
                torch.cuda.synchronize()
                e = max(float((o[:, :n_rows].float() - ref).norm() / ref.norm().clamp_min(1e-30)) for o in outs)
                rep = all(torch.equal(o, outs[0]) for o in outs)
                tail0 = all(not bool(o[:, n_rows:].any()) for o in outs)
                good = e < 4e-3 and rep and tail0 and all(bool(torch.isfinite(o.float()).all()) for o in outs)
                ok = ok and good
                print(f"[sk2l] self-test {name} M={m} (N {n_rows} K {k}, pmax {pmax}, ldo {ldo}): rel {e:.2e} repeat {rep} tail0 {tail0} -> {'ok' if good else 'FAIL'}", flush=True)
            res[name] = ok
        del comps
        _SK2L["ok"] = res
        print(f"[sk2l] ok {res} nvcc {_SK2L['secs']:.0f}s prepare {time.time() - t1:.1f}s", flush=True)
    except Exception as ex:
        _SK2L["ok"] = {}
        print(f"[sk2l] prepare raised {ex!r}: lab arms off", flush=True)
'''
HOOKS = (
    ('    qkv = _sk2(state, "qkv", data["attn_input"]) if SK2_ON else None\n',
     '    qkv = _sk2l(state, "qkv", data["attn_input"]) if SK2L_ON else None\n    if qkv is None:\n'
     '        qkv = _sk2(state, "qkv", data["attn_input"]) if SK2_ON else None\n', "qkv hook"),
    ('    ffn_output = _sk2(state, "unpooler", expert_output) if SK2_ON else None\n',
     '    ffn_output = _sk2l(state, "unpooler", expert_output) if SK2L_ON else None\n    if ffn_output is None:\n'
     '        ffn_output = _sk2(state, "unpooler", expert_output) if SK2_ON else None\n', "unpooler hook"),
    ('    _sk2_prepare(state, t0)\n', '    _sk2_prepare(state, t0)\n    _sk2l_prepare(state, t0)\n', "prepare call"),
    ('    if SK2_ENABLE:\n        jobs.append(("sk2", _sk2_build))\n',
     '    if SK2_ENABLE:\n        jobs.append(("sk2", _sk2_build))\n        jobs.append(("sk2l", _sk2l_build))\n', "PT1 job"),
)


def apply(s, base_name):
    consts, fns = P.upstream()
    t, cpp = cuda_lo(consts, fns)
    cfg = consts["bfp_SK2_CONFIGS"]
    names = [c[0] for c in cfg]
    cfgs = {nm: (names.index(nm),) + tuple(cfg[names.index(nm)][1:8]) for nm in PICK_LO.values()}
    lines = s.split(NL)
    P.check(lines[1].startswith("# nonce "), "nonce line")
    lines[1] = f"# nonce k3-sk2lo-ab ({base_name} + R-sk2-2 lab arms: alygg77 sk2 qkv / unpooler at T 16 / 8 / 1; tools/sk2lo_patches.py)"
    s = NL.join(lines)
    blk = LAB.replace("@CUDA@", t).replace("@CPP@", cpp).replace("@CFGS@", repr(cfgs))
    P.check(s.count(P.CALL_A) == 1, "kernel_fn def")
    s = s.replace(P.CALL_A, blk + P.CALL_A)
    for old, new, what in HOOKS:
        s = P.sub(s, old, new, what)
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_5_sk2.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_sk2lo_ab.py")
    new = apply(base.read_bytes().decode("ascii"), base.stem)
    if "check" in FLAGS:
        P.check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        print(f"sk2lo_patches: --check OK ({out.name} == apply({base.name}), {len(new):,} B)")
        return
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    ast.parse(new)
    print(f"sk2lo_patches: {base.name} -> {out.name} ({len(new):,} B, {len(new) - base.stat().st_size:+,} B; lab-only)")


if __name__ == "__main__":
    main()

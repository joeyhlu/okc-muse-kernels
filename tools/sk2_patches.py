"""sk2_patches.py (R-sk2-1, sk2 builder): alygg77's skinny BF16 GEMM `sk2` (their bf16_proj entry inlined in
kernels/muse_decode_layer.py, Apache-2.0) as in-layer arms at T = 32 only (decode 32, sliding + full), on the current final.

  python -I tools/sk2_patches.py [--base=kernels/lab/k3_hh_5.py] [--out=kernels/lab/k3_sk2_ab.py] [--check]
  python -I tools/lead_gate.py file kernels/lab/k3_sk2_ab.py
  ship form (R-sk2-1 GO, qkv + unpooler): python -I tools/sk2_patches.py --ship [--base=kernels/lab/k3_hh_5.py] [--out=<new>]
      specialized CUDA (two instances, SASS == lab: tools/local/sk2_compile.py --pair), P1M (V20) block cut (dead on every
      scored shape: see P1M_DROP), hooks at qkv / unpooler only.

Why: dense board at m = 32 (notes/study/dense_board_probe.md s3): their sk2 holds qkv 10.78 / o 6.09 / pooler 14.87 /
unpooler 14.46 / router 5.24 us against our GD MT4 11.64 / 7.06 / 16.07 / 16.53 / 5.94. Never run in our layer.

What is vendored (verbatim text, only comments / indentation stripped): bfp_CUDA_SRC (MBAR helpers in, wgmma BODY out, the
unused bounded spin + trap removed), bfp_MAP_SRC (the 2-D map / loads removed: unused), bfp_SK2_SRC with their own dispatch
generator (bfp__sk2_src, run on their text) restricted to the T = 32 picks (qkv32_r208, o32h_r100, pooler32_r144,
unpooler32_r144, router; sk2f = their fused pooler + router with the 'pooler' config at tokr 32), the sk2 / sk2f /
sk2_set_pdl declarations of bfp_CPP_SRC, and their Python packers bfp__sk2_geom / bfp_sk2_tileunits / bfp_sk2_compress /
bfp_sk2_stream / bfp_sk2_table (renamed _sk2_*, docstrings dropped). Compile flags = theirs (-O3, sm_90a,
no_implicit_headers).

Arms (module globals, set per rev_ab3 variant): SK2_ON = {"qkv": 1, "o": 1, "unpooler": 1, "cat": "f" (sk2f fused) | "s"
(sk2 pooler32 + sk2 router, their layer's T = 32 pick)}; SK2_PDL = weight stages issued before griddepcontrol.wait per GEMM
(their BFP_PDL_N = 2 at T <= 32; qkv 0 = no PDL edge, as our GD qkv). Defaults SK2_ON = {} -> ship dispatch, file identical
outside these hunks: nonce line, lab block before kernel_fn, 4 route hooks in kernel_fn, one prepare call after
_pt1_join_all (before _gft_prepare: no guard-timed prepare step waits for the sk2 nvcc), one PT1 job.
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000
TP = ROOT / "third_party" / "alygg77-okc-muse-spark-kernels-6cae49a"
PIN = "8f17613bafb2369152f2739638bf8f6dfb193f6a79dbad3e6d25795cf1ebaf4a"   # = tools/hyblo_patches.py PIN
CONSTS = ("bfp_CUDA_SRC", "bfp_MBAR_HELPERS", "bfp_MAP_SRC", "bfp_SK2_SRC", "bfp_CPP_SRC", "bfp_SK2_CONFIGS", "bfp_SK2_K")
FUNCS = ("bfp__sk2_geom", "bfp__sk2_src", "bfp_sk2_tileunits", "bfp_sk2_compress", "bfp_sk2_stream", "bfp_sk2_table")
PY_FUNCS = ("bfp__sk2_geom", "bfp_sk2_tileunits", "bfp_sk2_compress", "bfp_sk2_stream", "bfp_sk2_table")
RENAME = (("bfp__sk2_geom", "_sk2_geom"), ("bfp_sk2_tileunits", "_sk2_tileunits"), ("bfp_sk2_compress", "_sk2_compress"),
          ("bfp_sk2_stream", "_sk2_stream"), ("bfp_sk2_table", "_sk2_table"), ("bfp_SK2_K", "_SK2_K"))
# our proj -> their config name at tokr 32 (their bfp_SK2_PICK[(layer, 32)]); sk2f uses the 'pooler' config
PICK32 = {("qkv_gate", 32): "qkv32_r208", ("o", 32): "o32h_r100", ("pooler", 32): "pooler32_r144",
          ("unpooler", 32): "unpooler32_r144", ("router", 32): "router"}
FLAGS_NVCC = ("-O3", "-gencode=arch=compute_90a,code=sm_90a")


def check(c, msg):
    if not c:
        raise SystemExit(f"[sk2_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def slim_cuda(src):   # = tools/hyblo_patches.py slim_cuda
    out = []
    for ln in src.split("\n"):
        i = ln.find("//")
        while i >= 0 and (ln[:i].replace('\\"', "").count('"') % 2 or "'" in ln[:i]):
            i = ln.find("//", i + 2)
        if i >= 0 and not ln.rstrip().endswith("\\"):
            ln = ln[:i]
        ln = ln.strip()
        if ln:
            out.append(ln)
    return "\n".join(out) + "\n"


def cut(text, start, end, what):
    """remove text[start .. end) where start / end are unique markers (end kept)"""
    check(text.count(start) == 1 and text.count(end) >= 1, f"cut markers {what}")
    a = text.index(start)
    b = text.index(end, a)
    return text[:a] + text[b:]


def upstream():
    p = TP / "kernels" / "muse_decode_layer.py"
    raw = p.read_bytes()
    check(hashlib.sha256(raw).hexdigest() == PIN, "alygg77 file sha256 differs from the pin")
    src = raw.decode("utf-8")
    tree = ast.parse(src)
    consts, fns = {}, {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in CONSTS:
            check(n.targets[0].id not in consts, f"{n.targets[0].id} assigned twice")
            consts[n.targets[0].id] = ast.literal_eval(n.value)
        if isinstance(n, ast.FunctionDef) and n.name in FUNCS:
            check(n.name not in fns, f"{n.name} defined twice")
            fns[n.name] = n
    check(set(consts) == set(CONSTS) and set(fns) == set(FUNCS), "upstream names")
    return consts, fns


def cuda_text(consts, fns):
    """their sk2 translation unit (as their _de_source: common + MBAR + MAP + SK2 with a dispatch), T = 32 picks only"""
    ns = {"bfp_SK2_CONFIGS": consts["bfp_SK2_CONFIGS"], "bfp_SK2_PICK": dict(PICK32), "bfp_SK2_K": consts["bfp_SK2_K"],
          "bfp_SK2_SRC": consts["bfp_SK2_SRC"]}
    for name in ("bfp__sk2_geom", "bfp__sk2_src"):
        exec(compile(ast.Module(body=[fns[name]], type_ignores=[]), f"<{name}>", "exec"), ns)
    sk2 = ns["bfp__sk2_src"](True)
    # sk2f: their generator emits the 'pooler' config for tokr 8 / 16 / 32; keep tokr 32
    lines = sk2.split("\n")
    fl = [ln for ln in lines if "launch_sk2<" in ln and ", true>(Ws, x, out, ptab" in ln]
    check(len(fl) == 3, f"sk2f dispatch lines {len(fl)}")
    keep = [ln for ln in fl if ln.lstrip().startswith("if (tokr == 32)")]
    check(len(keep) == 1, "sk2f tokr 32 line")
    sk2 = "\n".join(ln for ln in lines if ln not in fl or ln in keep)
    sd = [ln for ln in sk2.split("\n") if ln.lstrip().startswith("if (cfg == ")]
    check(len(sd) == 5 and all("tokr == 32" in ln for ln in sd), f"sk2 dispatch lines {sd}")
    common = consts["bfp_CUDA_SRC"].replace("@@MBAR@@", consts["bfp_MBAR_HELPERS"]).replace("@@BODY@@", "")
    common = cut(common, "// cross-CTA spins are bounded", "__device__ __forceinline__ void okc_spin_ge", "spin comment + trap")
    a = common.index("__device__ __forceinline__ void okc_spin_ge")
    b = common.index("\n}\n", a) + 3
    common = common[:a] + common[b:]
    common = common.replace('__device__ __noinline__ void okc_spin_trap() { asm volatile("trap;"); }\n', "")
    check("trap" not in common and "okc_spin" not in common, "bounded spin + trap removed")
    mp = consts["bfp_MAP_SRC"]
    mp = cut(mp, "__device__ __forceinline__ void tma_load_2d_pol(", "__device__ __forceinline__ void tma_load_3d(", "2d loads")
    a = mp.index("static CUtensorMap make_map_2d(")
    b = mp.index("\n}\n", a) + 3
    mp = mp[:a] + mp[b:]
    check("make_map_2d" not in mp and "tma_load_2d" not in mp, "2-D map removed")
    t = slim_cuda(common + mp + sk2)
    check('"""' not in t and t.isascii() and "@@" not in t, "CUDA text embeddable as r-string")
    for need in ("sk2_kernel", "void sk2(", "void sk2f(", "void sk2_set_pdl(", "make_map_x3d", "decode8"):
        check(need in t, f"CUDA text lacks {need}")
    cpp = consts["bfp_CPP_SRC"]
    cpp = "#include <torch/extension.h>\n" + cpp[cpp.index("void sk2("):].strip() + "\n"
    check(cpp.count("void ") == 3, "sk2 / sk2f / sk2_set_pdl declarations")
    return t, cpp


def helpers(fns):
    out = []
    for name in PY_FUNCS:
        n = fns[name]
        if isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            n.body = n.body[1:]   # docstring dropped
        s = ast.unparse(n)
        for a, b in RENAME:
            s = s.replace(a, b)
        check("bfp" not in s, f"{name}: bfp name left")
        out.append(s)
    return "\n".join(out) + "\n"


def cfg_table(consts):
    cfgs = consts["bfp_SK2_CONFIGS"]
    names = [c[0] for c in cfgs]
    want = sorted(set(PICK32.values()) | {"pooler"}, key=names.index)
    return {nm: (names.index(nm),) + tuple(cfgs[names.index(nm)][1:]) for nm in want}


LAB = r'''"""[SK2] R-sk2-1 lab (tools/sk2_patches.py): alygg77's skinny BF16 GEMM sk2 (their bf16_proj entry; Apache-2.0,
_SK2_NOTICE) at T = 32 only: SK2_ON = {"qkv": 1, "o": 1, "unpooler": 1, "cat": "f" (fused sk2f) | "s" (sk2 pooler + router)}.
Defaults: ship dispatch."""
_SK2_NOTICE = r"""
==== alygg77/okc-muse-spark-kernels (Apache-2.0) -> the SK2 lab block (search: _SK2_, _sk2_) ====
Same source, license and notices as the HYB block above (commit 6cae49a, kernels/muse_decode_layer.py): its bf16_proj
section (bfp_CUDA_SRC with bfp_MBAR_HELPERS, bfp_MAP_SRC, bfp_SK2_SRC with the dispatch of their bfp__sk2_src, the sk2
declarations of bfp_CPP_SRC, bfp_SK2_CONFIGS / bfp_SK2_K entries, bfp__sk2_geom, bfp_sk2_tileunits, bfp_sk2_compress,
bfp_sk2_stream, bfp_sk2_table). Their NOTICE item 2 (NathenFernandes/muse-bf16-proj, MIT) names small-M routes of
kernels/muse_bf16_proj.py; the sk2f launch here uses their 'pooler' config, so that notice is kept too:
  NathenFernandes/muse-bf16-proj - https://github.com/NathenFernandes/muse-bf16-proj
  Copyright (c) 2026 Nathen Fernandes. Licensed under the MIT License:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
  persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
  this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS
  PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Modified by team joeyhlu (2026-10-31, lab only): the dispatch is restricted to their T = 32 picks, comments and indentation
are dropped, the unused bounded spin helper and the 2-D tensor-map helpers are removed, the Python helpers are renamed
_sk2_* without docstrings, and the prepare / self-test / routing glue is ours.
"""
THIRD_PARTY_NOTICES += _SK2_NOTICE
_SK2_CUDA = r"""@CUDA@"""
_SK2_CPP = r"""@CPP@"""
_SK2_FLAGS = @FLAGS@
SK2_ENABLE = True
SK2_ON = {}
SK2_PDL = {"qkv": 0, "o": 2, "unpooler": 2, "cat": 2}
SK2_T = 32
_SK2_CFGS = @CFGS@
_SK2_K = @KMAP@
_SK2_PROJ = {"qkv": ("qkv32_r208", 2304, HIDDEN), "o": ("o32h_r100", HIDDEN, 1024), "unpooler": ("unpooler32_r144", HIDDEN, D_EXPERT)}
_SK2 = {"mod": None, "err": None, "secs": 0.0, "ok": {}, "w": {}, "dummy": None, "part": None, "cnt": None, "calls": {}}
@HELPERS@def _sk2_build():
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    try:
        tag = hashlib.sha1((_SK2_CPP + _SK2_CUDA + " ".join(_SK2_FLAGS)).encode()).hexdigest()[:12]
        kw = dict(cpp_sources=_SK2_CPP, cuda_sources=_SK2_CUDA, functions=["sk2", "sk2f", "sk2_set_pdl"],
                  extra_cuda_cflags=list(_SK2_FLAGS), verbose=False)
        try:
            _SK2["mod"] = load_inline(name=f"sk2_{tag}", no_implicit_headers=True, **kw)
        except TypeError:   # torch without no_implicit_headers: the implicit torch prelude compiles the same text
            _SK2["mod"] = load_inline(name=f"sk2_{tag}i", **kw)
    except Exception as ex:
        _SK2["err"] = repr(ex)[-600:]
    _SK2["secs"] = time.time() - t0
def _sk2_run(name, x, n, k, pdl):
    ci, stream, tab, pmax, b4 = _SK2["w"][name]
    out = torch.empty((x.shape[0], n), dtype=torch.float32 if name == "router" else torch.bfloat16, device=x.device)
    mod = _SK2["mod"]
    mod.sk2_set_pdl(pdl)
    mod.sk2(stream, x, out, tab, pmax, b4, x.shape[0], n, k, ci, 32, 0, _SK2["dummy"])
    return out
def _sk2_runf(x, pdl):
    ci, stream, tab, pmax, b4 = _SK2["w"]["pooler"]
    m = x.shape[0]
    out = torch.empty((m, D_EXPERT), dtype=torch.bfloat16, device=x.device)
    outr = torch.empty((m, N_EXPERTS), dtype=torch.float32, device=x.device)
    mod = _SK2["mod"]
    mod.sk2_set_pdl(pdl)
    mod.sk2f(stream, _SK2["w"]["router"][1], x, out, outr, tab, pmax, b4, _SK2["part"], _SK2["cnt"], m, 32)
    return out, outr
def _sk2_xok(x, k):
    return x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[0] == SK2_T and x.shape[1] == k and x.is_contiguous() and x.data_ptr() % 16 == 0
def _sk2(state, proj, x):
    p = _SK2_PROJ.get(proj)
    if not SK2_ON.get(proj) or p is None or not _SK2["ok"].get(p[0]) or not _sk2_xok(x, p[2]):
        return None
    try:
        o = _sk2_run(p[0], x, p[1], p[2], SK2_PDL.get(proj, 0))
        _SK2["calls"][proj] = _SK2["calls"].get(proj, 0) + 1
        return o
    except Exception as ex:
        _SK2["ok"][p[0]] = False
        print(f"[sk2] {proj} raised {ex!r}: off", flush=True)
        return None
def _sk2_cat(state, x, data, tokens):
    on = SK2_ON.get("cat")
    if not on or tokens != SK2_T or state.get("v8") is None or not _sk2_xok(x, HIDDEN):
        return None
    pdl = SK2_PDL.get("cat", 0)
    try:
        if on == "f" and _SK2["ok"].get("fused"):
            y, lg = _sk2_runf(x, pdl)
        elif on == "s" and _SK2["ok"].get("pooler32_r144") and _SK2["ok"].get("router"):
            y = _sk2_run("pooler32_r144", x, D_EXPERT, HIDDEN, pdl)
            lg = _sk2_run("router", x, N_EXPERTS, HIDDEN, pdl)
        else:
            return None
        _SK2["calls"]["cat" + on] = _SK2["calls"].get("cat" + on, 0) + 1
        return _v8_tail(y, lg, state, data)
    except Exception as ex:
        _SK2["ok"]["fused" if on == "f" else "router"] = False
        print(f"[sk2] cat {on} raised {ex!r}: off", flush=True)
        return None
def _sk2_rel(o, ref):
    return float((o.float() - ref).norm() / ref.norm().clamp_min(1e-30))
def _sk2_prepare(state, t0):
    _SK2["ok"], _SK2["w"] = {}, {}
    if not SK2_ENABLE:
        return
    t1 = time.time()
    try:
        _pt1_wait("sk2")
        if _SK2["mod"] is None and _SK2["err"] is None:
            _sk2_build()
        if _SK2["mod"] is None:
            print(f"[sk2] OFF: build failed {_SK2['err']}", flush=True)
            return
        dense = state["dense"]
        dev = dense["o"].device
        W = {"qkv_gate": dense["qkv_gate"], "o": dense["o"], "pooler": dense["pooler"], "unpooler": dense["unpooler"],
             "router": state["router"]}
        _SK2["dummy"] = torch.zeros(8, dtype=torch.int64, device=dev)
        _SK2["part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
        _SK2["cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
        comps, shared = {}, {}
        for name, (ci, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL) in _SK2_CFGS.items():
            w = W[layer]
            if layer == "router":
                w = w.float().contiguous()
                hi = w.to(torch.bfloat16)
                lo = (w - hi.float()).to(torch.bfloat16)
                T_, U = w.shape[0] // 8, w.shape[1] // 64
                tub = torch.cat([_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T_, U, 1024),
                                 _sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T_, U, 1024)], dim=-1)
                _SK2["w"][name] = (ci, _sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW), _SK2["dummy"], 0, 0)
                continue
            if layer not in comps:
                comps[layer] = _sk2_compress(w.contiguous())
            comp = comps[layer]
            if comp is None:
                print(f"[sk2] {name}: weights not 12-bit packable -> off", flush=True)
                continue
            tab, pmax = _sk2_table(comp, n_rows, R)
            _SK2["w"][name] = (ci, _sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax, comp["base4"])
        del comps
        torch.cuda.synchronize()
        gen = torch.Generator(device=dev)
        gen.manual_seed(20261031)
        res = {}
        for name, (ci, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL) in _SK2_CFGS.items():
            if name not in _SK2["w"]:
                continue
            k = _SK2_K[layer]
            x = (torch.randn((32, k), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            w = W[layer].float()
            if name == "pooler":   # sk2f: pooler + router in one launch
                if "router" not in _SK2["w"]:
                    continue
                ref, refr = x.float() @ w.t(), x.float() @ W["router"].float().t()
                outs = [_sk2_runf(x, p) for p in (0, 2, 2)]
                torch.cuda.synchronize()
                e = max(_sk2_rel(o[0], ref) for o in outs)
                er = max(_sk2_rel(o[1], refr) for o in outs)
                rep = all(torch.equal(o[0], outs[0][0]) and torch.equal(o[1], outs[0][1]) for o in outs)
                ok = e < 4e-3 and er < 1e-4 and rep and all(bool(torch.isfinite(o[1]).all()) for o in outs)
                res["fused"] = ok
                print(f"[sk2] self-test fused (sk2f 'pooler'): pooler rel {e:.2e} router rel {er:.2e} repeat {rep} -> {'ok' if ok else 'FAIL'}", flush=True)
                continue
            ref = x.float() @ w.t()
            outs = [_sk2_run(name, x, n_rows, k, p) for p in (0, 2, 2)]
            torch.cuda.synchronize()
            e = max(_sk2_rel(o, ref) for o in outs)
            rep = all(torch.equal(o, outs[0]) for o in outs)
            ok = e < (1e-4 if layer == "router" else 4e-3) and rep and all(bool(torch.isfinite(o.float()).all()) for o in outs)
            res[name] = ok
            print(f"[sk2] self-test {name} ({layer} N {n_rows} K {k}, pmax {_SK2['w'][name][3]}): rel {e:.2e} repeat {rep} -> {'ok' if ok else 'FAIL'}", flush=True)
        _SK2["ok"] = res
        print(f"[sk2] ok {res} nvcc {_SK2['secs']:.0f}s prepare {time.time() - t1:.1f}s", flush=True)
    except Exception as ex:
        _SK2["ok"] = {}
        print(f"[sk2] prepare raised {ex!r}: lab arms off", flush=True)
'''
HOOKS = (
    ('    qkv = _gd(state, "qkv", data["attn_input"])\n',
     '    qkv = _sk2(state, "qkv", data["attn_input"]) if SK2_ON else None\n    if qkv is None:\n'
     '        qkv = _gd(state, "qkv", data["attn_input"])\n', "qkv hook"),
    ('    attn_output = _gd(state, "o", o)\n',
     '    attn_output = _sk2(state, "o", o) if SK2_ON else None\n    if attn_output is None:\n'
     '        attn_output = _gd(state, "o", o)\n', "o hook"),
    ('    elif tail == "cat" and (gc := _gd_cat(state, ffn_input, data, tokens)) is not None:\n',
     '    elif tail == "cat" and SK2_ON and (gc := _sk2_cat(state, ffn_input, data, tokens)) is not None:\n'
     '        expert_x, topk_weights, topk_ids, local_ids = gc\n'
     '    elif tail == "cat" and (gc := _gd_cat(state, ffn_input, data, tokens)) is not None:\n', "cat hook"),
    ('    ffn_output = _gd_unpool(state, expert_output, tokens)\n',
     '    ffn_output = _sk2(state, "unpooler", expert_output) if SK2_ON else None\n    if ffn_output is None:\n'
     '        ffn_output = _gd_unpool(state, expert_output, tokens)\n', "unpooler hook"),
    ('    _gft_prepare(state, t0)\n', '    _sk2_prepare(state, t0)\n    _gft_prepare(state, t0)\n', "prepare call (after _pt1_join_all)"),
    ('    if GD_BUILD:\n        jobs.append(("gd", _gd_build))\n',
     '    if GD_BUILD:\n        jobs.append(("gd", _gd_build))\n    if SK2_ENABLE:\n        jobs.append(("sk2", _sk2_build))\n', "PT1 job"),
)
CALL_A = "def kernel_fn(data, state):\n"


def build_lab(consts, fns):
    t, cpp = cuda_text(consts, fns)
    kmap = {k: v for k, v in consts["bfp_SK2_K"].items()}
    return (LAB.replace("@CUDA@", t).replace("@CPP@", cpp).replace("@FLAGS@", repr(FLAGS_NVCC))
            .replace("@CFGS@", repr(cfg_table(consts))).replace("@KMAP@", repr(kmap)).replace("@HELPERS@", helpers(fns)))


def apply(s, base_name="k3_hh_5"):
    consts, fns = upstream()
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = f"# nonce k3-sk2-ab ({base_name} + R-sk2 lab arms: alygg77 sk2 skinny BF16 GEMM at T = 32; tools/sk2_patches.py)"
    s = NL.join(lines)
    check(s.count(CALL_A) == 1, "kernel_fn def")
    s = s.replace(CALL_A, build_lab(consts, fns) + CALL_A)
    for old, new, what in HOOKS:
        s = sub(s, old, new, what)
    return s


# ------------------------------------------------------------------ ship form (R-sk2-1 GO: qkv + unpooler at T = 32)
SHIP_CFGS = ("qkv32_r208", "unpooler32_r144")
# dead-branch removal for the shipped instances (CL 1, COMP, !HL, !FUSE, !PK, SK2_DIAG 0): each edit is exact on the slimmed
# lab text; tools/local/sk2_compile.py --pair proves the two kernels' SASS identical to the lab text's
D = "\n"
SHIP_CUDA_EDITS = (
    ("template <typename T> __device__ __forceinline__ T cvt_out(float v);\ntemplate <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }\n"
     "template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }\n", "", 1),
    ("__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {\nuint32_t ok = 0;\ndo {\n"
     'asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\\n"\n'
     '" selp.u32 %0, 1, 0, p;\\n}\\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");\n} while (!ok);\n}\n', "", 1),
    ("#define SK2_DIAG 0\n", "", 1),
    ("const int dgv = SK2_DIAG ? dg : 0;\n"
     'auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };\n'
     "const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;\n"
     "const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);\n"
     "unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;\n"
     "const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;\n"
     "if (trace && threadIdx.x == 0) tr[0] = t_start;\n", "", 1),
    ("float* peer = corr + NESC * CSZ;\nint2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));\n",
     "int2* ptab_s = reinterpret_cast<int2*>(corr + NESC * CSZ);\n", 1),
    ("constexpr int NSTR = FUSE ? 4 : 0;\nconstexpr int WBYR = NW * 2048;\n"
     'static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");\n'
     "const int rt = cta >> 2, rq = cta & 3;\nconst uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;\nconst int nstt = nst + NSTR;\n",
     "const int nstt = nst;\n", 1),
    ("if constexpr (CL > 1) {\nmbar_init(empty0, NW + (COMP ? NESC : 0));\n"
     "for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }\n"
     "mbar_init(pbar, 1);\nmbar_init(cbar, 32 * NESC);\nmbar_init(qbar, 1);\nif (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);\n}\n", "", 1),
    ("mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);\n", "mbar_expect_tx(full0 + 8 * st, XBYr + WBY);\n", 1),
    ("if (!(dgv & 4))\nfor (int st = 0; st < P; ++st) tma_load_3d(", "for (int st = 0; st < P; ++st) tma_load_3d(", 1),
    ('if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");\n', "", 1),
    ("if (trace) tr[16 + 3 * st] = gtime();\n", "", 1),
    ("if (!FUSE || st < nst) {\nmbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
     "bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);\n"
     "if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n} else {\n"
     "const int gsr = rq * NSTR + (st - nst);\nmbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);\n"
     "bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);\n"
     "tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);\n}\n",
     "mbar_expect_tx(full0 + 8 * slot, XBYr + WBY);\nbulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);\n"
     "tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n", 1),
    ('__syncwarp();\nif constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");\nreturn;\n', "__syncwarp();\nreturn;\n", 2),
    ("if (FUSE && st >= nst) {\nSK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n__syncwarp();\nif (lane == 0) mbar_arrive(empty0 + 8 * slot);\ncontinue;\n}\n", "", 1),
    ("if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();\n", "", 1),
    ("if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }\n", "", 1),
    ("uint32_t bl[HL ? NT : 1][8];\n", "", 1),
    ("if constexpr (COMP) {\nasm volatile(\"ld.shared.v4.u32", "asm volatile(\"ld.shared.v4.u32", 1),
    ('"r"(wb + nt * TU + 512 + lane * 8));\n} else {\nconst uint32_t a0 = wb + nt * TU + lane * 32;\n', '"r"(wb + nt * TU + 512 + lane * 8));\n', 1),
    ('asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));\n'
     'asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));\n'
     'if constexpr (HL) {\nasm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));\n'
     'asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));\n}\n}\n}\n',
     "}\n", 1),
    ("uint32_t a[4][MT][4];\nif (dgv & 16) {\n#pragma unroll\nfor (int j = 0; j < 4; ++j)\n#pragma unroll\n"
     "for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;\n} else {\n",
     "uint32_t a[4][MT][4];\n", 1),
    ("a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);\n}\nif (UPW == 1 && uu == 0) {", "a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);\nif (UPW == 1 && uu == 0) {", 1),
    ("if (dgv & 8) {\nconst int nh = 0;\n#pragma unroll\nfor (int j = 0; j < 4; ++j)\n#pragma unroll\nfor (int nt = 0; nt < NT; ++nt) {\n"
     "if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);\n"
     "else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);\n}\ncontinue;\n}\n", "", 1),
    ("constexpr int HM = HL ? 2 : 1;\nmma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt],", "mma_bf16(acc[(uu * 4 + j) % NACC][mt][nt],", 1),
    ("if constexpr (HL)\nmma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);\n", "", 1),
    ("if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();\n", "", 1),
    ("if (trace && lane == 0) tr[1 + warp] = gtime();\nif (dgv & 1) {\nif (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);\nreturn;\n}\n", "", 1),
    ("if (trace && tid == 0) tr[9] = gtime();\n", "", 1),
    ('if (trace && mt == 0 && nt == 0) {\nasm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));\nif (tid == 0) tr[12] = gtime();\n}\n', "", 1),
    ("if (trace && tid == 0) tr[11] = gtime();\n", "", 1),
    ("if (trace && tid == 0) tr[10] = gtime();\nif (trace && tid == 0) tr[13] = gtime();\n"
     'if constexpr (CL > 1) {\nasm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");\nmbar_wait_cluster(qbar, 0);\n}\n'
     "if (trace && tid == 0) tr[14] = gtime();\n", "", 1),
    ("#pragma unroll\nfor (int c = 0; c < CL - 1; ++c) {\nconst float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);\n"
     "v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;\n}\n", "", 1),
    ("if constexpr (sizeof(OutT) == 4) {\n*reinterpret_cast<float4*>(o) = v;\n} else {\n__nv_bfloat162 lo", "__nv_bfloat162 lo", 1),
    ("*reinterpret_cast<uint2*>(o) = pk;\n}\n}\nif (trace && tid == 0) tr[15] = gtime();\n"
     'if ((dgv & 32) && dbg != nullptr && tid == 0) {\nunsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));\n'
     "dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();\n}\n}\n",
     "*reinterpret_cast<uint2*>(o) = pk;\n}\n}\n", 1),
    ("(CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;\n", "(size_t)pmax * 8 + 128;\n", 1),
    ("auto go = [&](auto pkc) {\nconstexpr bool PK = decltype(pkc)::value;\nauto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE>;\n",
     'TORCH_CHECK(M == TOKR, "sk2: M must equal the tile rows");\nauto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, false, FUSE>;\n', 1),
    ("PK ? (uint32_t)M : (uint32_t)TOKR, UPS);\n", "(uint32_t)TOKR, UPS);\n", 1),
    ("if (CL > 1 || pdl > 0) {\n", "if (pdl > 0) {\n", 1),
    ("lc.gridDim = dim3(N / R * CL);\n", "lc.gridDim = dim3(N / R);\n", 1),
    ("cudaLaunchAttribute at_[2];\nint na = 0;\nif (CL > 1) {\nat_[na].id = cudaLaunchAttributeClusterDimension;\nat_[na].val.clusterDim.x = CL;\n"
     "at_[na].val.clusterDim.y = 1;\nat_[na].val.clusterDim.z = 1;\n++na;\n}\nif (pdl > 0) {\nat_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
     "at_[na].val.programmaticStreamSerializationAllowed = 1;\n++na;\n}\nlc.attrs = at_;\nlc.numAttrs = na;\n",
     "cudaLaunchAttribute at_[1];\nat_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
     "at_[0].val.programmaticStreamSerializationAllowed = 1;\nlc.attrs = at_;\nlc.numAttrs = 1;\n", 1),
    ("C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n};\nif (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});\n}\n",
     "C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n}\n", 1),
)


def ship_cuda(t, cpp):
    t = cut(t, "if constexpr (FUSE) {\nfloat accr[2][MT][4];", "if (trace && lane == 0) tr[1 + warp] = gtime();", "FUSE router piece")
    t = cut(t, "if (CL > 1 && crank > 0) {\n", "named_bar(1, NC);\nif (trace && tid == 0) tr[9] = gtime();", "cluster epilogue")
    for old, new, n in SHIP_CUDA_EDITS:
        c = t.count(old)
        check(c == n, f"ship CUDA edit: found {c}x, expected {n}: {old[:80]!r}")
        t = t.replace(old, new)
    a = t.index("void sk2f(")
    b = t.index("void sk2(", a)
    t = t[:a] + t[b:]
    keep = [ln for ln in t.split("\n") if ln.startswith("if (cfg == ")]
    names = {6: "qkv32_r208", 11: "unpooler32_r144"}
    for ln in keep:
        ci = int(ln.split("cfg == ", 1)[1].split(" ", 1)[0])
        if ci not in names:
            t = t.replace(ln + "\n", "")
    for w in ("dgv", "trace", "gtime", "FUSE &&", "peer", "mbar_wait_cluster", "cvt_out", "crank > 0", "accr", "sk2f", "cfg == 0 ", "cfg == 8 "):
        check(w not in t, f"ship CUDA: {w!r} left")
    check(t.count("if (cfg == ") == 2, "ship dispatch = qkv32_r208 + unpooler32_r144")
    cpp = cpp[:cpp.index("void sk2f(")] + cpp[cpp.index("void sk2_set_pdl("):]
    return t, cpp


SHIP = r'''"""[SK2] R-sk2-1 (tools/sk2_patches.py --ship): alygg77's skinny BF16 GEMM sk2 (their bf16_proj entry; Apache-2.0, _SK2_NOTICE)
for qkv (qkv32_r208) and the unpooler (unpooler32_r144) at T = 32, ahead of GD (in-layer paired, 24 rounds: d32s -3.78 / d32f
-3.40 us; R-sk2-1 run 2). Any failure -> GD as before."""
_SK2_NOTICE = r"""
==== alygg77/okc-muse-spark-kernels (Apache-2.0) -> the SK2 block (search: _SK2_, _sk2_) ====
Same source, license and notices as the HYB block above (commit 6cae49a, kernels/muse_decode_layer.py): its bf16_proj
section (bfp_CUDA_SRC with bfp_MBAR_HELPERS, bfp_MAP_SRC, bfp_SK2_SRC with the dispatch of their bfp__sk2_src, the sk2
declarations of bfp_CPP_SRC, the bfp_SK2_CONFIGS entries qkv32_r208 / unpooler32_r144, bfp_sk2_tileunits,
bfp_sk2_compress, bfp_sk2_stream, bfp_sk2_table). Their NOTICE item 2 (NathenFernandes/muse-bf16-proj, MIT) names small-M
routes of kernels/muse_bf16_proj.py that share this kernel family, so that notice is kept too:
  NathenFernandes/muse-bf16-proj - https://github.com/NathenFernandes/muse-bf16-proj
  Copyright (c) 2026 Nathen Fernandes. Licensed under the MIT License:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
  persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
  this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS
  PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Modified by team joeyhlu (2026-10-31): the dispatch is restricted to the two configs above at 32 tokens, the branches those
two instances never take (clusters, the fused router piece, hi/lo weights, packed x, diagnostics) and the unused helpers are
removed (same SASS for both kernels, tools/local/sk2_compile.py), comments and indentation are dropped, the Python helpers
are renamed _sk2_* without docstrings, and the prepare / self-test / routing glue is ours.
"""
THIRD_PARTY_NOTICES += _SK2_NOTICE
_SK2_CUDA = r"""@CUDA@"""
_SK2_CPP = r"""@CPP@"""
_SK2_FLAGS = @FLAGS@
SK2_ENABLE = True
SK2_ON = {"qkv": 1, "unpooler": 1}
SK2_PDL = {"qkv": 0, "unpooler": 2}
SK2_T = 32
_SK2_CFGS = @CFGS@
_SK2_K = @KMAP@
_SK2_PROJ = {"qkv": ("qkv32_r208", 2304, HIDDEN), "unpooler": ("unpooler32_r144", HIDDEN, D_EXPERT)}
_SK2 = {"mod": None, "err": None, "secs": 0.0, "ok": {}, "w": {}, "dummy": None}
@HELPERS@def _sk2_build():
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    try:
        tag = hashlib.sha1((_SK2_CPP + _SK2_CUDA + " ".join(_SK2_FLAGS)).encode()).hexdigest()[:12]
        kw = dict(cpp_sources=_SK2_CPP, cuda_sources=_SK2_CUDA, functions=["sk2", "sk2_set_pdl"], extra_cuda_cflags=list(_SK2_FLAGS),
                  verbose=False)
        try:
            _SK2["mod"] = load_inline(name=f"sk2_{tag}", no_implicit_headers=True, **kw)
        except TypeError:
            _SK2["mod"] = load_inline(name=f"sk2_{tag}i", **kw)
    except Exception as ex:
        _SK2["err"] = repr(ex)[-600:]
    _SK2["secs"] = time.time() - t0
def _sk2_run(name, x, n, k, pdl):
    ci, stream, tab, pmax, b4 = _SK2["w"][name]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    mod = _SK2["mod"]
    mod.sk2_set_pdl(pdl)
    mod.sk2(stream, x, out, tab, pmax, b4, x.shape[0], n, k, ci, 32, 0, _SK2["dummy"])
    return out
def _sk2(state, proj, x):
    p = _SK2_PROJ.get(proj)
    if not (SK2_ON.get(proj) and p is not None and _SK2["ok"].get(p[0]) and x.dtype == torch.bfloat16 and x.dim() == 2
            and x.shape[0] == SK2_T and x.shape[1] == p[2] and x.is_contiguous() and x.data_ptr() % 16 == 0):
        return None
    try:
        return _sk2_run(p[0], x, p[1], p[2], SK2_PDL.get(proj, 0))
    except Exception as ex:
        _SK2["ok"][p[0]] = False
        print(f"[sk2] {proj} raised {ex!r}: GD", flush=True)
        return None
def _sk2_prepare(state, t0):
    _SK2["ok"], _SK2["w"] = {}, {}
    if not SK2_ENABLE:
        return
    t1 = time.time()
    try:
        _pt1_wait("sk2")
        if _SK2["mod"] is None and _SK2["err"] is None:
            _sk2_build()
        if _SK2["mod"] is None:
            print(f"[sk2] OFF: build failed {_SK2['err']}", flush=True)
            return
        dense = state["dense"]
        dev = dense["qkv_gate"].device
        _SK2["dummy"] = torch.zeros(8, dtype=torch.int64, device=dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(20261031)
        res = {}
        for name, (ci, layer, n_rows, R, NT, NWR, WSK, UPW) in _SK2_CFGS.items():
            w = dense[layer].contiguous()
            comp = _sk2_compress(w)
            if comp is None:
                print(f"[sk2] {name}: weights not 12-bit packable -> GD", flush=True)
                continue
            tab, pmax = _sk2_table(comp, n_rows, R)
            _SK2["w"][name] = (ci, _sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW), tab, pmax, comp["base4"])
            del comp
            k = _SK2_K[layer]
            x = (torch.randn((32, k), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            ref = x.float() @ w.float().t()
            outs = [_sk2_run(name, x, n_rows, k, p) for p in (0, 2, 2)]
            torch.cuda.synchronize()
            e = max(float((o.float() - ref).norm() / ref.norm().clamp_min(1e-30)) for o in outs)
            rep = all(torch.equal(o, outs[0]) for o in outs)
            res[name] = ok = e < 4e-3 and rep and all(bool(torch.isfinite(o.float()).all()) for o in outs)
            print(f"[sk2] self-test {name} (N {n_rows} K {k}, pmax {pmax}): rel {e:.2e} repeat {rep} -> {'ok' if ok else 'FAIL'}", flush=True)
        _SK2["ok"] = res
        print(f"[sk2] ok {res} nvcc {_SK2['secs']:.0f}s prepare {time.time() - t1:.1f}s", flush=True)
    except Exception as ex:
        _SK2["ok"] = {}
        print(f"[sk2] prepare raised {ex!r}: GD", flush=True)
'''
SHIP_HOOKS = (HOOKS[0], HOOKS[3], HOOKS[4], HOOKS[5])
SHIP_PY = ("bfp_sk2_tileunits", "bfp_sk2_compress", "bfp_sk2_stream", "bfp_sk2_table")
# P1M (V20) cut: every route of V20_P1M_ROUTE ((qkv | o | cat | unpooler, 16), (unpooler, 32)) sits behind a GD route for the same
# (proj, T) in kernel_fn (GD_ROUTE qkv / o / unpooler 16 + unpooler 32, GD_CAT 16), and behind sk2 at (unpooler, 32); engine traces of
# all 12 scored shapes (box_out/rat/trace_*.json, box_out/vh/trace*.json, box_out/sk2/trace*.json) show no P1M call that ran.
# Behind it: P1 / DG1 / cuBLAS (_v12_gemm, torch.mm), all correct. Kept: V20_P1M_ENABLE / _ROUTE / _PDL (now False / {}), the
# P1 trigger helpers (_V20_WAVE1, V20_P1_TRIG, _v20_p1_trig: used by P1).
P1M_DROP = ("P1M_INST", "P1M_CPP_ADD", "_inst_lines", "p1m_cuda_src", "P1M_CUDA_ADD", "p1m_ws", "p1m_lane_maps", "P1M_KPERM",
            "p1m_interleave", "p1m_call", "p1m_esc_columns", "_v20_p1_pack", "V20_P1M_TOL", "V20_P1M_TRIG", "_v20_inst", "_v20_ok",
            "_v20_p1m", "_v20_weight", "_v20_selftest", "_v20_prepare", "p1m_verify")
P1M_EDITS = (
    ("    if V20_P1M_ENABLE and V20_P1M_ROUTE:\n        cpp, cuda, fns = cpp + P1M_CPP_ADD, cuda + p1m_cuda_src(_v20_inst(), ()), fns + [\"p1m_gemm\", \"p1m_info\"]\n", ""),
    ("    _v20_prepare(state)\n", ""),
    ('    if qkv is None:\n        qkv = _v20_p1m(state, "qkv", data["attn_input"])\n', ""),
    ('    if attn_output is None:\n        attn_output = _v20_p1m(state, "o", o)\n', ""),
    ('        y32 = _v20_p1m(state, "cat", ffn_input)\n        if y32 is None:\n            y32 = _dg1(state, "cat", ffn_input)\n',
     '        y32 = _dg1(state, "cat", ffn_input)\n'),
    ('        ffn_output = _v20_p1m(state, "unpooler", expert_output)\n        if ffn_output is None:\n'
     '            ffn_output = _dg1(state, "unpooler", expert_output)\n', '        ffn_output = _dg1(state, "unpooler", expert_output)\n'),
    ("V20_P1M_ENABLE = True\n", "V20_P1M_ENABLE = False\n"),
)


def cut_p1m(s):
    import re
    tree = ast.parse(s)
    lines = s.split("\n")
    spans, first, last = [], None, None
    for n in tree.body:
        nm = n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else (
            n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
        if nm in P1M_DROP or (isinstance(n, ast.For) and "P1M_KPERM[" in ast.unparse(n)):
            spans.append((n.lineno - 1, n.end_lineno))
        if nm == "V20_P1M_ROUTE":
            spans.append((n.lineno - 1, n.end_lineno))
    check(len(spans) == len(P1M_DROP) + 2, f"P1M spans {len(spans)} vs {len(P1M_DROP) + 2}")
    keep = [True] * len(lines)
    for a, b in spans:
        for i in range(a, b):
            keep[i] = False
    vr = next(i for i in range(len(lines)) if lines[i].startswith("V20_P1M_ROUTE = "))
    lines[vr] = "V20_P1M_ROUTE = {}"
    keep[vr] = True
    s = "\n".join(ln for ln, k in zip(lines, keep) if k)
    for old, new in P1M_EDITS:
        s = sub(s, old, new, f"P1M edit {old[:40]!r}")
    for nm in P1M_DROP:
        check(not re.search(r"\b%s\b" % re.escape(nm), s), f"P1M cut: {nm} still referenced")
    return s


def apply_ship(s, base_name):
    consts, fns = upstream()
    t, cpp = cuda_text(consts, fns)
    t, cpp = ship_cuda(t, cpp)
    cfg = cfg_table(consts)
    cfgs = {k: cfg[k][:8] for k in SHIP_CFGS}
    kmap = {cfg[k][1]: consts["bfp_SK2_K"][cfg[k][1]] for k in SHIP_CFGS}
    out = []
    for name in SHIP_PY:
        n = fns[name]
        if isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            n.body = n.body[1:]
        u = ast.unparse(n)
        for a, b in RENAME:
            u = u.replace(a, b)
        out.append(u)
    block = (SHIP.replace("@CUDA@", t).replace("@CPP@", cpp).replace("@FLAGS@", repr(FLAGS_NVCC)).replace("@CFGS@", repr(cfgs))
             .replace("@KMAP@", repr(kmap)).replace("@HELPERS@", "\n".join(out) + "\n"))
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = lines[1] + " + R-sk2 alygg77 sk2 qkv / unpooler at T=32, P1M cut [tools/sk2_patches.py --ship]"
    s = NL.join(lines)
    s = cut_p1m(s)
    check(s.count(CALL_A) == 1, "kernel_fn def")
    s = s.replace(CALL_A, block + CALL_A)
    for old, new, what in SHIP_HOOKS:
        s = sub(s, old, new, what)
    return s, t


def gates(path, s, limit=True):
    sys.path.insert(0, str(ROOT / "tools"))
    import pyslim
    n = len(s.encode())
    if limit:
        check(n < LIMIT, f"size {n} B >= {LIMIT}")
    check(s.isascii(), "non-ASCII byte")
    ast.parse(s)
    d = pyslim.decorated_sources(s)
    bad = [x for x in d if x.startswith("UNPARSABLE")]
    check(not bad, f"decorated defs that do not re-parse: {bad[:2]}")
    for a in ("def _sk2_prepare(", 'jobs.append(("sk2", _sk2_build))', "THIRD_PARTY_NOTICES += _SK2_NOTICE\n"):
        check(s.count(a) == 1, f"missing {a!r}")
    check(s.count("_sk2_prepare(") == 2, "one _sk2_prepare def + one call")
    print(f"sk2_patches: gates OK on {path.name}: {n:,} B ({LIMIT - n:+,} B vs the cap), {len(d)} decorated defs re-parse")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_5.py")
    ship = "ship" in FLAGS
    out = ROOT / ARGS.get("out", f"kernels/lab/{base.stem}_sk2.py" if ship else "kernels/lab/k3_sk2_ab.py")
    new = apply_ship(base.read_bytes().decode("ascii"), base.stem)[0] if ship else apply(base.read_bytes().decode("ascii"), base.stem)
    if "check" in FLAGS:
        check(out.exists(), f"{out} missing")
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        gates(out, new, limit=ship)
        print(f"sk2_patches: --check OK ({out.name} == apply({base.name}))")
        return
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"sk2_patches: {base.name} -> {out.name} ({len(new):,} B, {len(new) - base.stat().st_size:+,} B)")
    gates(out, new, limit=ship)


if __name__ == "__main__":
    main()

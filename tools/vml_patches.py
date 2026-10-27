"""vml_patches.py: VML1 (kernels/lab/vml_moe.py, notes/study/vm_large.md) as the T >= 17 MoE route of a generated layer.

Inserted as plain top-level source (no importlib, A19): the lab module's code from its constants to its end, with every
top-level Python name prefixed `_vml_` (tokenize: NAME tokens only, string literals - the CUDA text - untouched), plus:
  * VML_ENABLE / VML_BUCKETS (read at call time) and _VML (module, scratch, per-bucket self-test result);
  * a PT1 prefetch build job "vml" (the nvcc build runs in parallel with the other layer builds);
  * _vml_prepare(st, w, dev), called at the end of _v11_k2a33_prepare: waits for the build, allocates the scratch, and
    self-tests every enabled bucket against the shipped K2A route (_v11_k2a33_call with VML off) and the FP32 reference
    (synthetic normal / skewed inputs, two calls each, error flag, finite); a failing bucket stays on K2A (loud);
  * the route: _v11_k2a33_call returns _vml_call(st, data) for an enabled, self-tested bucket, before the PX / v3.7 /
    v3.6 / v3.3 routes; any exception disables VML for the rest of the process (loud) and falls through to K2A.
Every anchor is asserted unique.

    text = apply(text)
"""
import ast
import io
import pathlib
import tokenize

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB = ROOT / "kernels/lab/vml_moe.py"
START = "E_LOC, TOPK, KDIM = 32, 8, 4096\n"


def lab_block():
    src = LAB.read_text(encoding="utf-8")
    assert src.count(START) == 1
    body = src[src.index(START):]
    names = set()
    for n in ast.parse(body).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                for x in (t.elts if isinstance(t, ast.Tuple) else [t]):
                    if isinstance(x, ast.Name):
                        names.add(x.id)
    toks = []
    for tok in tokenize.generate_tokens(io.StringIO(body).readline):
        if tok.type == tokenize.NAME and tok.string in names:
            tok = tok._replace(string="_vml_" + tok.string.lstrip("_"))
        toks.append(tok)
    out = tokenize.untokenize(toks)
    a = out.index('_vml_CUDA_T = r"""')
    b = out.index('"""', a + 18)
    out = out[:a] + chr(10).join(ln.lstrip(" ") for ln in out[a:b].split(chr(10))) + out[b:]   # layer copy: CUDA dedented
    left = {n.id for n in ast.walk(ast.parse(out)) if isinstance(n, ast.Name) and n.id in names}
    assert not left, f"unrenamed top-level names (f-string?): {left}"
    return out, sorted(names)


GLUE = r'''

# ======== VML1: T >= 17 MoE route (notes/study/vm_large.md; tools/vml_patches.py) ========
VML_ENABLE = True
VML_BUCKETS = {128: True, 256: True, 32: @VML32@}
VML_FLAGS = @VMLFLAGS@
VML_COST = @VMLCOST@
_VML = {"mod": None, "sc": None, "ok": {}, "err": None}


def _vml_build_job():
    _VML["mod"], _ = _vml_build()


def _vml_route(st, data):
    x = data["hidden_states"].contiguous()
    rep = _k2a_bucket_of(x.shape[0])
    return _vml_call(_VML["mod"], st, _VML["sc"], x, data["topk_ids"].to(torch.int32).contiguous(),
                     data["topk_weights"].to(torch.float32).contiguous(), int(data["config"]["local_start"]),
                     flags=VML_FLAGS.get(rep), cost=VML_COST.get(rep))


def _vml_prepare(st, w, dev):
    _VML["ok"] = {}
    if not VML_ENABLE:
        return
    try:
        _pt1_wait("vml")
        if _VML["mod"] is None:
            _vml_build_job()
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        if sms > _vml_SLOTS:
            print(f"[vml] OFF: {sms} SMs unsupported", flush=True)
            return
        _VML["sc"] = _vml_alloc(dev, sms)
        cases = {128: ((100, 41, "normal"), (128, 42, "skewed"), (160, 43, "skewed")),
                 256: ((200, 44, "skewed"), (256, 45, "normal"), (257, 46, "skewed"), (320, 47, "normal")),
                 32: ((25, 48, "normal"), (32, 49, "skewed"), (48, 50, "skewed"))}
        for rep, on in VML_BUCKETS.items():
            if not on:
                continue
            good = True
            for bs, seed, dist in cases[rep]:
                d = _k2a_self_test_inputs(bs, seed, dev, dist)
                d = {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                     "topk_weights": d["topk_weights"].float().contiguous(), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
                ref = _k2a_ref_fp32(d, w)
                ok_k = dict(_VML["ok"])
                _VML["ok"] = {}
                o_k = _v11_k2a33_call(d["hidden_states"], d["topk_ids"], d["topk_weights"], st).clone()
                _VML["ok"] = ok_k
                for k in range(2):
                    _VML["sc"]["err"].zero_()
                    o = _vml_route(st, d)
                    torch.cuda.synchronize()
                    ek, _ = _k2a_rel_err(o, o_k)
                    tot, row = _k2a_rel_err(o, ref)
                    errf = int(_VML["sc"]["err"][0])
                    ok = ek < 2e-3 and tot < 1e-2 and row < 5e-2 and not errf and bool(torch.isfinite(o.float()).all())
                    print(f"[vml] self-test bucket {rep} bs={bs} {dist} call {k}: rel vs K2A {ek:.2e} vs FP32 {tot:.3e} "
                          f"worst row {row:.3e} err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                    good &= ok
                _k2a33_reset(st)
            _VML["ok"][rep] = good
        print(f"[vml] VML1 live buckets {[r for r, v in _VML['ok'].items() if v]} (self-test {_VML['ok']})", flush=True)
    except Exception as ex:
        _VML["ok"] = {}
        _VML["err"] = repr(ex)[:500]
        print(f"[vml] prepare raised {ex!r}: every bucket stays on K2A", flush=True)
        try:
            _k2a33_reset(st)
        except Exception:
            pass
'''

PT1_OLD = """    if GD_BUILD:
        jobs.append(("gd", _gd_build))
"""
PT1_NEW = """    if GD_BUILD:
        jobs.append(("gd", _gd_build))
    if VML_ENABLE:
        jobs.append(("vml", _vml_build_job))
"""
PREP_OLD = """    print(f"[v8] K2A v3.3 buckets {ok} in {time.time() - t1:.1f}s; route {V11_K2A33_ROUTE}", flush=True)
"""
PREP_NEW = """    _vml_prepare(st, w, dev)
    print(f"[v8] K2A v3.3 buckets {ok} in {time.time() - t1:.1f}s; route {V11_K2A33_ROUTE}", flush=True)
"""
CALL_OLD = """    rep = _k2a_bucket_of(x.shape[0])
    if M32_PX and rep == 32 and st.get("px_ok") and x.shape[0] <= 48:
"""
CALL_NEW = """    rep = _k2a_bucket_of(x.shape[0])
    if VML_ENABLE and VML_BUCKETS.get(rep) and _VML["ok"].get(rep) and x.shape[0] <= _vml_MMAX:
        try:
            return _vml_route(st, data)
        except Exception as ex:
            print(f"[k3] VML1 route raised ({ex!r}): K2A from here on", flush=True)
            _VML["ok"] = {}
    if M32_PX and rep == 32 and st.get("px_ok") and x.shape[0] <= 48:
"""
ANCHOR = "\nK2A36_ENABLE = True\n"


def apply(text, t32=False, flags=None, cost=None):
    """flags / cost: per-bucket dicts {rep: int} / {rep: [16 ints]} baked in as VML_FLAGS / VML_COST (None = lab defaults)."""
    for old in (PT1_OLD, PREP_OLD, CALL_OLD, ANCHOR):
        assert text.count(old) == 1, old[:80]
    block, names = lab_block()
    for nm in names:
        assert ("_vml_" + nm.lstrip("_")) not in text, nm
    text = text.replace(PT1_OLD, PT1_NEW).replace(PREP_OLD, PREP_NEW).replace(CALL_OLD, CALL_NEW)
    glue = (GLUE.replace("@VML32@", "True" if t32 else "False").replace("@VMLFLAGS@", repr(dict(flags or {})))
            .replace("@VMLCOST@", repr({k: list(v) for k, v in (cost or {}).items()})))
    text = text.replace(ANCHOR, "\n" + block + glue + ANCHOR)
    ast.parse(text)
    return text

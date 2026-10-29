"""gen_k3_fxs.py: k3_gc_18 + FX-S (one-launch verify-sliding attention at B = 16 / 32, kernels/lab/fxs_src.py) as a
compile-time-flagged route (FXS_ENABLE) -> kernels/lab/k3_fxs_<n>.py.

    python tools/gen_k3_fxs.py [--n 1] [--route 16:8,32:4] [--off]

The route sits before KS1 / VFA / FA4 in kernel_fn for (B in FXS_ROUTE, q_len 8, window 2048); prepare builds the
extension in the PT1 prefetch pool, self-tests every routed NS against _k1fa4_reference (+ graph replay identity) and
routes only what passed. --off emits FXS_ENABLE = False (the bitwise control layer)."""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels" / "muse_decode_layer" / "k3_gc_18.py"

PY_BLOCK = r'''

# ---- FX-S (R-fxs, 2026-10-29): one-launch verify-sliding attention at B = 16 / 32 (q_len 8), see FXS_CUDA_SRC ----
FXS_ENABLE = @ENABLE@
FXS_ROUTE = @ROUTE@
FXS_TRIG = 1
FXS_PREPARE_BY_S = 245.0
FXS_SELF_TEST_CASES = ((16, 3000, 1311), (32, 2600, 1312), (16, 2047, 1313), (32, 80000, 1314), (3, 100, 1315))
_FXS = {"mod": None, "cnt": {}, "build_s": 0.0}


def _fxs_build():
    _pt1_wait("fxs")
    if _FXS["mod"] is not None:
        return _FXS["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((FXS_CPP_SRC + FXS_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    _FXS["mod"] = load_inline(name=f"fxs_{tag}", cpp_sources=FXS_CPP_SRC, cuda_sources=FXS_CUDA_SRC,
                              functions=["fxs_forward", "fxs_info"],
                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _FXS["build_s"] = time.time() - t0
    return _FXS["mod"]


def _fxs_counters(dev):
    key = str(dev)
    if key not in _FXS["cnt"]:
        _FXS["cnt"][key] = torch.zeros(1024, dtype=torch.int32, device=dev)
    return _FXS["cnt"][key]


def _fxs_call(qkv, data, cos_sin, ns=None):
    cfg = data["config"]
    B = cfg["batch"]
    ns = FXS_ROUTE[B] if ns is None else ns
    dev = qkv.device
    po = torch.empty((B * ns * 128 * 64,), dtype=torch.float32, device=dev)
    pml = torch.empty((B * ns * 128 * 2,), dtype=torch.float32, device=dev)
    out = torch.empty((B * 8, 16 * 64), dtype=torch.bfloat16, device=dev)
    prof = torch.empty((0,), dtype=torch.int64, device=dev)
    _FXS["mod"].fxs_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                            _fxs_counters(dev), out, prof, int(ns), 1, float(cfg["sm_scale"] * cfg["qk_scale_factor"]),
                            KS1_EPS, int(FXS_TRIG), 1 if KS1_PDL else 0, 1 if cfg["rope_theta"] else 0)
    return out


def _fxs_wanted(qkv, data, cfg, state):
    if not (FXS_ENABLE and state.get("fxs") and _FXS["mod"] is not None):
        return False
    B = cfg["batch"]
    if not (cfg["q_len"] == 8 and B in FXS_ROUTE and FXS_ROUTE[B] in state["fxs"] and cfg["window"] == 2048
            and cfg["rope_theta"] == ROPE_THETA and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0
            and cfg["head_dim"] == 64 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2):
        return False
    cs = state.get("cos_sin")
    return (cs is not None and cs.dtype == torch.float32 and cs.is_contiguous() and cs.dim() == 2 and cs.shape[1] == 64
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.shape[0] == B * 8
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2 and data["page_table"].is_contiguous()
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].numel() == B
            and data["cache_loc"].dtype == torch.int64 and data["cache_loc"].is_contiguous() and data["cache_loc"].numel() == B * 8
            and data["positions"].dtype == torch.int64 and data["positions"].is_contiguous())


def _fxs_self_test(cos_sin, dev):
    ok = set()
    for ns in sorted(set(FXS_ROUTE.values())):
        good = True
        for B, ctx, seed in FXS_SELF_TEST_CASES:
            d = _k1fa4_case(B, ctx, 8, 2048, seed, False, dev)
            ref = _k1fa4_reference(d).view(B * 8, -1).float()
            fa4_ok = bool(K1FA4_STATE.get("ready"))
            fa = None
            if fa4_ok:
                d2 = _k1fa4_case(B, ctx, 8, 2048, seed, False, dev)
                fa = _v18_fa4_core(d2["qkv"].view(B * 8, -1), d2).view(B * 8, -1).float()
                del d2
            o = _fxs_call(d["qkv"].view(B * 8, -1), d, cos_sin, ns).float()
            torch.cuda.synchronize(dev)
            err = float((o - ref).norm() / ref.norm())
            ferr = float((fa - ref).norm() / ref.norm()) if fa is not None else 0.03
            clean = int(_fxs_counters(dev).abs().sum()) == 0
            fine = bool(torch.isfinite(o).all()) and clean and err <= max(1.25 * ferr, ferr + 2e-3)
            print(f"[fxs] self-test NS{ns} B{B} ctx{ctx}: rel {err:.2e} (FA4 {ferr:.2e}{'' if fa4_ok else ' assumed'})"
                  f" clean {clean} -> {'ok' if fine else 'FAIL'}", flush=True)
            good = good and fine
            del d, ref, fa, o
        if good:   # graph replay identity
            d = _k1fa4_case(32, 3000, 8, 2048, 1319, False, dev)
            q = d["qkv"].view(32 * 8, -1)
            o1 = _fxs_call(q, d, cos_sin, ns).clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                og = _fxs_call(q, d, cos_sin, ns)
            g.replay()
            torch.cuda.synchronize(dev)
            good = bool(torch.equal(og, o1))
            print(f"[fxs] graph check NS{ns}: replay identical {good}", flush=True)
            del g, og, o1, d
        if good:
            ok.add(ns)
    return ok


def _fxs_prepare(state, t0):
    state["fxs"] = set()
    if not FXS_ENABLE:
        print("[fxs] FXS_ENABLE = False", flush=True)
        return
    if time.time() - t0 > FXS_PREPARE_BY_S:
        print(f"[fxs] skipped: prepare already took {time.time() - t0:.1f}s", flush=True)
        return
    try:
        mod = _fxs_build()
        dev = state["router"].device
        cs = state.get("cos_sin")
        if cs is None:
            print("[fxs] no cos_sin in state: not routed", flush=True)
            return
        info = {ns: mod.fxs_info(ns, 1) for ns in sorted(set(FXS_ROUTE.values()))}
        state["fxs"] = _fxs_self_test(cs, dev)
        print(f"[fxs] route {FXS_ROUTE} ok {sorted(state['fxs'])}; regs/local/smem/occ {info}; build {_FXS['build_s']:.0f}s", flush=True)
    except Exception as ex:
        state["fxs"] = set()
        print(f"[fxs] prepare FAILED, not routed: {ex!r}", flush=True)
'''

ROUTE_SNIP = '''    if FXS_ENABLE and _fxs_wanted(qkv, data, cfg, state):
        try:
            o = _fxs_call(qkv, data, state["cos_sin"]).view(tokens, hq * d)
        except Exception as ex:
            _k3_disable(state, "fxs", set(), "FXS", ex)
            o = None
    if o is None and _ks1_wanted(qkv, data, cfg, state):
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--route", default="16:8,32:4")
    ap.add_argument("--off", action="store_true")
    a = ap.parse_args()
    route = {int(k): int(v) for k, v in (x.split(":") for x in a.route.split(","))}
    src = BASE.read_text()
    fx = {}
    for node in ast.parse((ROOT / "kernels" / "lab" / "fxs_src.py").read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("FXS_CPP_SRC", "FXS_CUDA_SRC"):
            fx[node.targets[0].id] = ast.literal_eval(node.value)
    assert '"""' not in fx["FXS_CUDA_SRC"] and '"""' not in fx["FXS_CPP_SRC"]
    mini = []   # 1-MB budget: drop // comments, indentation and blank lines of the CUDA source (no // inside its strings)
    for ln in fx["FXS_CUDA_SRC"].splitlines():
        if "//" in ln and '"' not in ln.split("//", 1)[0][-1:]:
            ln = ln.split("//", 1)[0]
        ln = ln.strip()
        if ln:
            mini.append(ln)
    fx["FXS_CUDA_SRC"] = "\n" + "\n".join(mini) + "\n"
    block = (PY_BLOCK.replace("@ENABLE@", "False" if a.off else "True").replace("@ROUTE@", repr(route))
             + '\nFXS_CPP_SRC = r"""' + fx["FXS_CPP_SRC"] + '"""\n\nFXS_CUDA_SRC = r"""' + fx["FXS_CUDA_SRC"] + '"""\n\n')
    edits = [
        ("\ndef kernel_fn(data, state):\n", block + "\ndef kernel_fn(data, state):\n"),
        ("    if _ks1_wanted(qkv, data, cfg, state):\n", ROUTE_SNIP),
        ('    if LT_ENABLE and LT_PIN:\n        jobs.append(("lt", _lt_build))\n',
         '    if LT_ENABLE and LT_PIN:\n        jobs.append(("lt", _lt_build))\n    if FXS_ENABLE:\n        jobs.append(("fxs", _fxs_build))\n'),
        ("    _vfa_prepare(state, t0)\n    torch.cuda.synchronize()\n",
         "    _vfa_prepare(state, t0)\n    _fxs_prepare(state, t0)\n    torch.cuda.synchronize()\n"),
    ]
    for old, new in edits:
        assert src.count(old) == 1, old[:60]
        src = src.replace(old, new)
    ast.parse(src)
    out = ROOT / "kernels" / "lab" / f"k3_fxs_{a.n}{'_off' if a.off else ''}.py"
    out.write_text(src, newline="\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(src.encode())} B) route {route} enable {not a.off}")


if __name__ == "__main__":
    main()

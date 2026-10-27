"""gen_vd_board.py: kernels/muse_bf16_proj/vd_<n>.py = vp_4 + VD routes at m = 128 / 256 (notes/study/vd_dense.md).

The VD kernel source comes from kernels/lab/vd_lab.py (vd_cuda_src of the board's instances, expanded here; read with ast, no
torch import). vp_4's autotune is kept as is: VD routes are extra candidates of its exact-m (128 / 256) search next to the
pinned cuBLASLt algorithms, addmm and torchT, verified with the same judge-metric check (+ bitwise repeat, err 0) and kept
only if they win the cold-L2 timing against SGLang's call. The VD module builds in a thread started at prepare entry.

    python -I tools/gen_vd_board.py [--n=1] [--routes=pooler:128:1-128-8-2-0-1,...]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)

DEFAULT_ROUTES = {}   # (layer, m) -> [inst, ...] (filled from R-vd-2)


def parse_routes(spec):
    out = {}
    for item in spec.split(","):
        if not item:
            continue
        lay, m, inst = item.split(":")
        out.setdefault((lay, int(m)), []).append(tuple(int(v) for v in inst.split("-")))
    return out


PY_TMPL = r'''

VD_ENABLE = True
VD_FLAGS = 8   # L2 evict_first on the weight tiles (R-vd-2: >= neutral on every shape)
VD_INSTS = @INSTS@
VD_BOARD = @ROUTE@
VD_CPP_SRC = @CPP@
VD_CUDA_SRC = @CUDA@
_VD = {"mod": None, "err": None, "w": {}, "errt": None}


def _vd_build_job():
    try:
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1((VD_CPP_SRC + VD_CUDA_SRC).encode()).hexdigest()[:10]
        t1 = time.time()
        _VD["mod"] = load_inline(name=f"vd_{tag}", cpp_sources=VD_CPP_SRC, cuda_sources=VD_CUDA_SRC,
                                 functions=["vd_run", "vd_info"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
        print(f"[vd] built {len(VD_INSTS)} kernels in {time.time() - t1:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        _VD["err"] = repr(ex)[-1500:]


def _vd_pack(w, bm):
    """lossless repack: pre-swizzled (TMA SW128 image) [bm x 64] tiles in front-major order (tile (s, rb) at s nrb + rb)"""
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


def _vd_call(r, x, out):
    r["mod"].vd_run(r["kid"], x, r["w"], r["w"], out, r["err"], r["stamps"], r["nrb"], r["w_rb"], r["w_s"], VD_FLAGS, 0, 0)


def _vd_cands(state, layer, m, w):
    """VD candidate routes for (layer, m), or [] (logged) when the module or the shape does not fit."""
    if not VD_ENABLE or (layer, m) not in VD_BOARD:
        return []
    th = _VD.get("thread")
    if th is not None:
        th.join()
    if _VD["mod"] is None:
        print(f"[vd] module unavailable -> no VD candidates: {_VD['err']}", flush=True)
        return []
    if _VD["errt"] is None:
        _VD["errt"] = torch.zeros((4,), dtype=torch.int32, device=w.device)
        _VD["stt"] = torch.zeros((4,), dtype=torch.int64, device=w.device)
    out = []
    for kid in VD_BOARD[(layer, m)]:
        nwg, bn, nst, cl, f32, rbd = VD_INSTS[kid]
        bm = 64 * nwg
        if f32 or m > bn or w.shape[0] % bm or w.shape[1] % 64:
            continue
        try:
            if (layer, bm) not in _VD["w"]:
                _VD["w"][(layer, bm)] = _vd_pack(w, bm)
        except Exception as ex:  # noqa: BLE001
            print(f"[vd] {layer} pack bm {bm} failed: {ex!r}", flush=True)
            continue
        flat, w_rb, w_s = _VD["w"][(layer, bm)]
        out.append({"kind": "vd", "mod": _VD["mod"], "kid": kid, "w": flat, "w_rb": w_rb, "w_s": w_s, "nrb": w.shape[0] // bm,
                    "err": _VD["errt"], "stamps": _VD["stt"], "desc": f"vd{kid}(bm{bm},cl{cl},st{nst},rb{rbd})"})
    return out

'''

ANCHOR_LT = "def _lt_search(state, mod, timer, floor, dev, budget_end):\n"
ANCHOR_CAND = ('                r = {"kind": "mmT", "wt": kmaj[layer], "desc": "torchT"}\n'
               '                got = _route_call(r, x, n, layer)\n'
               '                ok, msg = _check(got, x, w, probes, False, ref, 5e-3)\n'
               '                if ok:\n'
               '                    cands.append(r)\n')
VD_CAND = ('                for r in _vd_cands(state, layer, m, w):\n'
           '                    try:\n'
           '                        got = _route_call(r, x, n, layer)\n'
           '                        got2 = _route_call(r, x, n, layer)\n'
           '                        torch.cuda.synchronize()\n'
           '                        ok, msg = _check(got, x, w, probes, False, ref, 5e-3)\n'
           '                        ok = ok and bool(torch.equal(got, got2)) and int(r["err"][0]) == 0\n'
           '                        for tp in (m - 3, m // 2 + 1):\n'
           '                            xp = x[:tp].contiguous()\n'
           '                            gp = _route_call(r, xp, n, layer)\n'
           '                            torch.cuda.synchronize()\n'
           '                            okp, msgp = _check(gp, xp, w, probes[:tp], False, None, 5e-3)\n'
           '                            ok, msg = ok and okp, msg + f"; m{tp} " + msgp\n'
           '                    except Exception as ex:  # noqa: BLE001\n'
           '                        ok, msg = False, repr(ex)[:160]\n'
           '                    print(f"[vd] {layer}/m{m} {r[\'desc\']}: {msg} -> {\'candidate\' if ok else \'REJECTED\'}", flush=True)\n'
           '                    if ok:\n'
           '                        cands.append(r)\n')
ANCHOR_CALL = '    if r["kind"] == "mmT":\n'
VD_CALL = ('    if r["kind"] == "vd":\n'
           '        out = torch.empty((x.shape[0], n_out), dtype=torch.bfloat16, device=x.device)\n'
           '        _vd_call(r, x, out)\n'
           '        return out\n')
ANCHOR_THREAD = "        lt_thread.start()\n"
VD_THREAD = ('    if VD_ENABLE and VD_BOARD and _VD["mod"] is None and torch.cuda.get_device_capability(dev)[0] >= 9:\n'
             '        _VD["thread"] = threading.Thread(target=_vd_build_job)\n'
             '        _VD["thread"].start()\n')


def _lab():
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


def main():
    n = ARGS.get("n", "1")
    routes = parse_routes(ARGS["routes"]) if "routes" in ARGS else DEFAULT_ROUTES
    assert routes, "no routes"
    L = _lab()
    insts = sorted({i for v in routes.values() for i in v})
    board = {k: [insts.index(i) for i in v] for k, v in sorted(routes.items())}
    cuda = L.vd_cuda_src(insts)
    code = (PY_TMPL.replace("@INSTS@", repr(insts)).replace("@ROUTE@", repr(board))
            .replace("@CPP@", 'r"""' + L.VD_CPP_SRC + '"""').replace("@CUDA@", 'r"""' + cuda + '"""'))
    base = (ROOT / "kernels/muse_bf16_proj/vp_4.py").read_text(encoding="utf-8")
    lines = base.split(NL)
    assert lines[1].startswith("# nonce vp-4")
    desc = ", ".join(f"{la}@{m}" for la, m in sorted(routes))
    lines[1] = f"# nonce vd-{n} (vp_4 + VD1 SS-wgmma candidates in the m = 128 / 256 autotune: {desc})"
    t = NL.join(lines)
    for a in (ANCHOR_LT, ANCHOR_CAND, ANCHOR_CALL, ANCHOR_THREAD):
        assert t.count(a) == 1, f"anchor not unique: {a[:60]!r}"
    t = t.replace(ANCHOR_LT, code.lstrip(NL) + NL + NL + ANCHOR_LT)
    t = t.replace(ANCHOR_CAND, ANCHOR_CAND + VD_CAND)
    t = t.replace(ANCHOR_CALL, VD_CALL + ANCHOR_CALL)
    t = t.replace(ANCHOR_THREAD, ANCHOR_THREAD + VD_THREAD)
    compile(t, f"vd_{n}.py", "exec")
    t.encode("ascii")
    out = ROOT / f"kernels/muse_bf16_proj/vd_{n}.py"
    assert len(t.encode()) < 1_000_000
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[gen_vd_board] wrote {out.relative_to(ROOT)} ({len(t.encode())} B, sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]}),"
          f" routes {board}, insts {insts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

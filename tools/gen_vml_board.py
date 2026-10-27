"""gen_vml_board.py: kernels/muse_moe_w4a16/vml_<n>.py = the VMS board file (kernels/muse_moe_w4a16/vms_1.py: our layer MoE
K2A routes + VMS at bs <= 16) + VML1 (kernels/lab/vml_moe.py, tools/vml_patches.py block) at bs >= 17, with an in-prepare
variant table: per bucket (32 / 128 / 256) correctness vs K2A and FP32 (bs 25 / 32 / 33 / 48, 100 / 128 skewed, 200 / 256 /
257, two calls each), then cold-L2 graph timings of K2A vs VML flags 1 / 0 on synthetic inputs, one printed line per
bucket; a bucket routes to the fastest VML arm only if it passed and beats K2A by > 1 us. Check only, never scored.

    python -I tools/gen_vml_board.py [--n=1]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vml_patches  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 1))
SRC = ROOT / "kernels/muse_moe_w4a16/vms_1.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/vml_{N}.py"

TAIL = r'''

VML_ENABLE = True
VML_BUCKETS = {32: True, 128: True, 256: True}
VML_ARMS = (1, 0)
_VML = {"mod": None, "sc": None, "route": {}}
_VML_CASES = {32: ((25, 61, "normal"), (32, 62, "skewed"), (33, 63, "normal"), (48, 64, "normal")),
              128: ((100, 65, "normal"), (128, 66, "skewed"), (160, 67, "normal")),
              256: ((200, 68, "skewed"), (256, 69, "normal"), (257, 70, "normal"))}


def _vml_board_prepare(state):
    st, w = state["st"], state["w"]
    dev = w["w13"].device
    t0 = time.time()
    dt = 0.0
    if _VML["mod"] is None:
        _VML["mod"], dt = _vml_build()
    _VML["sc"] = _vml_alloc(dev, torch.cuda.get_device_properties(dev).multi_processor_count)
    flush = torch.ones((256 << 20) // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream()

    def vml(d, f):
        return _vml_call(_VML["mod"], st, _VML["sc"], d["hidden_states"], d["topk_ids"], d["topk_weights"],
                         int(d["config"]["local_start"]), flags=f)

    def norm(d):
        return {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                "topk_weights": d["topk_weights"].float().contiguous(),
                "config": {"local_start": int(d["config"]["local_start"]), "local_count": K2A_E_LOC}}
    for rep, on in VML_BUCKETS.items():
        if not on:
            continue
        good, worst = True, 0.0
        for bs, seed, dist in _VML_CASES[rep]:
            d = norm(_k2a_self_test_inputs(bs, seed, dev, dist))
            ref = _k2a_ref_fp32(d, w)
            o_k = _vmsb_kernel_fn(d, state).clone()
            for f in VML_ARMS:
                for k in range(2):
                    _VML["sc"]["err"].zero_()
                    o = vml(d, f)
                    torch.cuda.synchronize()
                    ek, _ = _k2a_rel_err(o, o_k)
                    tot, row = _k2a_rel_err(o, ref)
                    ok = (ek < 2e-3 and tot < 1e-2 and row < 5e-2 and int(_VML["sc"]["err"][0]) == 0
                          and bool(torch.isfinite(o.float()).all()))
                    worst = max(worst, tot)
                    if not ok:
                        print(f"[vmlb] FAIL bs={bs} {dist} flags {f}: vs K2A {ek:.2e} FP32 {tot:.2e} row {row:.2e} "
                              f"err {_VML['sc']['err'].tolist()}", flush=True)
                    good &= ok
            _k2a33_reset(st)
        d = norm(_k2a_self_test_inputs(rep, 70 + rep, dev, "normal"))
        fns = {"k2a": lambda d=d: _vmsb_kernel_fn(d, state)}
        for f in VML_ARMS:
            fns[f"vml{f}"] = (lambda d=d, f=f: vml(d, f))
        graphs = {}
        for nm, fn in fns.items():
            with torch.cuda.stream(stream):
                fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    fn()
            torch.cuda.synchronize()
            graphs[nm] = g
        ts = {nm: [] for nm in graphs}
        names = list(graphs)
        for rd in range(12):
            for nm in names[rd % len(names):] + names[:rd % len(names)]:
                with torch.cuda.stream(stream):
                    torch.sum(flush, dim=0, out=sink)
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    graphs[nm].replay()
                    b.record()
                b.synchronize()
                ts[nm].append(a.elapsed_time(b) * 1e3)
        med = {nm: sorted(v)[len(v) // 2] for nm, v in ts.items()}
        best = min((f for f in VML_ARMS), key=lambda f: med[f"vml{f}"])
        live = good and med[f"vml{best}"] < med["k2a"] - 1.0
        if live:
            _VML["route"][rep] = best
        print(f"[vmlb] bs={rep}: k2a {med['k2a']:.1f} " + " ".join(f"vml{f} {med[f'vml{f}']:.1f}" for f in VML_ARMS)
              + f" us | checks {'ok' if good else 'FAIL'} (FP32 worst {worst:.2e}) -> {'vml%d' % best if live else 'K2A'}",
              flush=True)
        del graphs
        _k2a33_reset(st)
    del flush
    print(f"[vmlb] VML1 routes {_VML['route']} (build {dt:.0f} s, prepare {time.time() - t0:.0f} s)", flush=True)


def kernel_fn(data, state):
    x = data["hidden_states"]
    m = x.shape[0]
    f = _VML["route"].get(_k2a_bucket_of(m))
    if f is not None and m <= _vml_MMAX:
        return _vml_call(_VML["mod"], state["st"], _VML["sc"], x.contiguous(), data["topk_ids"].to(torch.int32).contiguous(),
                         data["topk_weights"].to(torch.float32).contiguous(), int(data["config"]["local_start"]), flags=f)
    return _vmsb_kernel_fn(data, state)


def prepare(weights):
    th = None
    if VML_ENABLE:   # VML1's nvcc build overlaps K2A's / VMS's
        import threading
        th = threading.Thread(target=lambda: _VML.__setitem__("mod", _vml_build()[0]), daemon=True)
        th.start()
    state = _vmsb_prepare(weights)
    if th is not None:
        th.join()
    if VML_ENABLE and state.get("st") is not None:
        try:
            _vml_board_prepare(state)
        except Exception as ex:  # noqa: BLE001
            _VML["route"] = {}
            print(f"[vmlb] prepare raised {ex!r}: K2A / VMS everywhere", flush=True)
        torch.cuda.synchronize()
    return state
'''


def main():
    t = SRC.read_text(encoding="utf-8")
    for a, b in (("\ndef kernel_fn(data, state):\n", "\ndef _vmsb_kernel_fn(data, state):\n"),
                 ("\ndef prepare(weights):\n", "\ndef _vmsb_prepare(weights):\n")):
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    lines = t.split("\n")
    assert lines[0] == "# okc: problem=muse_moe_w4a16"
    lines[1] = (f"# vml-board-{N}: vms_1 (K2A + VMS at bs <= 16) + VML1 one-launch RS-wgmma stream-K MoE at bs >= 17 "
                "(notes/study/vm_large.md), in-prepare variant table; check only, never scored.")
    t = "\n".join(lines)
    block, _ = vml_patches.lab_block()
    tail = TAIL
    if "--research" in sys.argv:   # in-prepare refit + variant table (tools/vml_board_research.py)
        import vml_board_research
        a, b = TAIL.index("\nVML_ENABLE = True\n"), TAIL.index("\ndef prepare(weights):\n")
        tail = TAIL[:a] + vml_board_research.RESEARCH + "\n" + TAIL[b:]
    s = t.rstrip("\n") + "\n\n\n" + block + tail
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    assert not dup, dup
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline="\n")
    py_compile.compile(str(OUT), doraise=True)
    print(f"[gen_vml_board] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

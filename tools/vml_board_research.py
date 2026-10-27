"""vml_board_research.py: the research tail of tools/gen_vml_board.py --research (kernels/muse_moe_w4a16/vml_<n>.py):
per bucket an in-prepare cost-table refit from per-CTA marks (least squares of phase ends on per-CTA stage counts by
n-tiles), correctness of three VML arms (v1 = flags 1, v3 = flags 3 (+ x L2 prefetch), v65 = flags 65 (self-fill of the first x slots + producer-warp gather), v67 = 65 + x prefetch, v65f = v65 + refit table) vs
K2A and FP32, cold-L2 graph timings of K2A and the arms, and marks quantiles; one printed line per item (judge log_tail
keeps 16 KB). The fastest passing arm serves the bucket only if it beats K2A by > 1 us.
"""
RESEARCH = r'''

VML_ENABLE = True
VML_BUCKETS = {32: True, 128: True, 256: True}
_VML = {"mod": None, "sc": None, "route": {}}
_VML_CASES = {32: ((25, 61, "normal"), (33, 63, "normal"), (48, 64, "skewed")),
              128: ((100, 65, "normal"), (128, 66, "skewed")),
              256: ((200, 68, "skewed"), (256, 69, "normal"), (257, 70, "normal"))}


def _vml_fit(run, ds, G):
    out = []
    for pb in (0, 1):
        X, Y = [], []
        for d in ds:
            mv = run(d).double()
            t0 = mv[:, 0].min()
            aend = mv[:, 6] - t0
            end = mv[:, 1:5].max(1).values - t0
            X.append(torch.cat([torch.ones(G, 1, dtype=torch.float64), mv[:, 8 + 8 * pb:16 + 8 * pb]], 1))
            Y.append(aend if pb == 0 else end - aend)
        X, Y = torch.cat(X), torch.cat(Y) / 1e3
        seen = (X[:, 1:] > 0).any(0)
        ks = [k for k in range(8) if bool(seen[k])]
        coef = torch.linalg.lstsq(X[:, [0] + [1 + k for k in ks]], Y.unsqueeze(1)).solution.squeeze(1)
        c = {k: float(coef[i + 1]) for i, k in enumerate(ks)}
        ref = max(c, key=lambda k: float(X[:, 1 + k].sum())) if c else 0
        tab = list(_vml_COST)
        if c and c[ref] > 0:
            for k, v in c.items():
                if v > 0:
                    tab[k] = int(min(255, max(16, round(v / c[ref] * _vml_COST[ref]))))
        out += tab
        print(f"[vmlb] fit {'AB'[pb]}: us/stage by n-tile { {k + 1: round(v, 3) for k, v in c.items()} } "
              f"icpt {float(coef[0]):.1f} -> {tab}", flush=True)
    return out


def _vml_board_prepare(state):
    st, w = state["st"], state["w"]
    dev = w["w13"].device
    t0 = time.time()
    dt = 0.0
    if _VML["mod"] is None:
        _VML["mod"], dt = _vml_build()
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    _VML["sc"] = _vml_alloc(dev, G)
    flush = torch.ones((256 << 20) // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream()
    marks = torch.zeros((G * 32,), dtype=torch.int64, device=dev)

    def vml(d, f, cost=None, mk=None):
        return _vml_call(_VML["mod"], st, _VML["sc"], d["hidden_states"], d["topk_ids"], d["topk_weights"],
                         int(d["config"]["local_start"]), flags=f, cost=cost, marks=mk)

    def norm(d):
        return {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                "topk_weights": d["topk_weights"].float().contiguous(),
                "config": {"local_start": int(d["config"]["local_start"]), "local_count": K2A_E_LOC}}

    def marked(d, f=3, cost=None):
        marks.zero_()
        with torch.cuda.stream(stream):
            torch.sum(flush, dim=0, out=sink)
            vml(d, f, cost, marks)
        torch.cuda.synchronize()
        return marks.view(G, 32).cpu()

    def q3(v):
        v = sorted(v)
        return f"{v[0]:.1f}/{v[len(v) // 2]:.1f}/{v[-1]:.1f}"
    for rep, on in VML_BUCKETS.items():
        if not on or time.time() - t0 > 150:
            continue
        ds = [norm(_k2a_self_test_inputs(rep, 900 + 7 * i + rep, dev, "normal" if i % 2 == 0 else "skewed"))
              for i in range(3)]
        cost = _vml_fit(lambda d: marked(d, 193), ds, G)
        cost65 = _vml_fit(lambda d: marked(d, 65), ds, G)
        arms = {"v1": (1, None), "v65": (65, None), "v193": (193, None), "v65f": (65, cost65), "v193f": (193, cost)}
        good = {nm: True for nm in arms}
        for bs, seed, dist in _VML_CASES[rep]:
            d = norm(_k2a_self_test_inputs(bs, seed, dev, dist))
            ref = _k2a_ref_fp32(d, w)
            o_k = _vmsb_kernel_fn(d, state).clone()
            for nm, (f, cst) in arms.items():
                for k in range(2):
                    _VML["sc"]["err"].zero_()
                    o = vml(d, f, cst)
                    torch.cuda.synchronize()
                    ek, _ = _k2a_rel_err(o, o_k)
                    tot, row = _k2a_rel_err(o, ref)
                    ok = (ek < 2e-3 and tot < 1e-2 and row < 5e-2 and int(_VML["sc"]["err"][0]) == 0
                          and bool(torch.isfinite(o.float()).all()))
                    if not ok:
                        print(f"[vmlb] FAIL {nm} bs={bs} {dist}: vs K2A {ek:.2e} FP32 {tot:.2e} row {row:.2e} "
                              f"err {_VML['sc']['err'].tolist()}", flush=True)
                    good[nm] &= ok
            _k2a33_reset(st)
        d = ds[0]
        fns = {"k2a": lambda d=d: _vmsb_kernel_fn(d, state)}
        for nm, (f, cst) in arms.items():
            fns[nm] = (lambda d=d, f=f, cst=cst: vml(d, f, cst))
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
        for rd in range(16):
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
        del graphs
        ok_arms = [nm for nm in arms if good[nm]]
        best = min(ok_arms, key=lambda nm: med[nm]) if ok_arms else None
        live = best is not None and med[best] < med["k2a"] - 1.0
        if live:
            _VML["route"][rep] = arms[best]
        print(f"[vmlb] bs={rep}: " + " ".join(f"{nm} {v:.1f}" for nm, v in med.items())
              + f" us | ok {ok_arms} -> {best if live else 'K2A'}", flush=True)
        for nm in ("v1", "v193f"):
            mv = marked(ds[1], *arms[nm]).double()
            t00 = float(mv[:, 0].min())
            r = lambda col: [(float(x) - t00) / 1e3 for x in mv[:, col]]  # noqa: E731
            print(f"[vmlb] bs={rep} {nm} marks us min/p50/max: route {q3(r(7))} gather {q3(r(5))} A end {q3(r(6))} "
                  f"end {q3([(float(x) - t00) / 1e3 for x in mv[:, 1:5].max(1).values])}", flush=True)
        _k2a33_reset(st)
    del flush
    print(f"[vmlb] VML1 routes { {k: v[0] for k, v in _VML['route'].items()} } (build {dt:.0f} s, "
          f"prepare {time.time() - t0:.0f} s)", flush=True)


def kernel_fn(data, state):
    x = data["hidden_states"]
    m = x.shape[0]
    r = _VML["route"].get(_k2a_bucket_of(m))
    if r is not None and m <= _vml_MMAX:
        return _vml_call(_VML["mod"], state["st"], _VML["sc"], x.contiguous(), data["topk_ids"].to(torch.int32).contiguous(),
                         data["topk_weights"].to(torch.float32).contiguous(), int(data["config"]["local_start"]),
                         flags=r[0], cost=r[1])
    return _vmsb_kernel_fn(data, state)
'''

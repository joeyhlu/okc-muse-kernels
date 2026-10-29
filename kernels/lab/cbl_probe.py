"""R-cbl-0: why the layer's T >= 128 dense cuBLAS GEMMs run 3-6 us longer in-layer than standalone (CBL session).

Each arm = one CUDA graph [prefix op, GEMM]; every replay goes through the judge's GpuTimer protocol (256 MB read flush +
300 us gate) unless the arm name ends in "_ug" (ungated, back-to-back like the layer_budget nsys loop). nsys gives the GEMM
kernel's own duration (the last kernel inside each NVTX range "C|arm|r"). Prefixes separate the candidate causes:
  none      judge-like standalone (three copies: none_a first, none_b middle, none_c last = position controls)
  spin      150 us torch.cuda._sleep (same delay, ~idle power)
  read      torch.sum over 512 MB (HBM-heavy ~130 us, leaves L2 clean)     -> HBM / power transient after a streamer
  mmh       bf16 GEMM 2048 x 2048 x 16384 (~150 us tensor-core heavy)       -> SM power / clock transient
  wr8/wr40  fill_ 8 MB / 40 MB (dirty L2 lines)                             -> write-back traffic during the GEMM
  wr40s     fill_ 40 MB then 150 us spin                                    -> does an idle gap drain the write-back
    python3 -u kernels/lab/cbl_probe.py run --reps=24     (under nsys --capture-range=cudaProfilerApi)
    python3 -u kernels/lab/cbl_probe.py analyze trace.sqlite
"""
import json
import random
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "judge"))

# (proj, T): N, K, fp32 out, torchT (x @ W^T with W^T stored (K, N) contiguous, = gc_16's VD_TT unpooler/128 route)
GEMMS = {("unpooler", 128): (8192, 4096, False, True), ("unpooler", 256): (8192, 4096, False, False),
         ("cat", 256): (4608, 8192, True, False), ("qkv", 128): (2304, 8192, False, False),
         ("o", 128): (8192, 1024, False, False), ("o", 256): (8192, 1024, False, False)}
PREFIXES = ["none_a", "spin", "read", "mmh", "none_b", "wr8", "wr40", "wr40s", "none_ug", "read_ug", "none_c"]


def run(args):
    import torch
    import judge as J
    dev = torch.device("cuda")
    timer = J.GpuTimer()
    timer.gate()   # calibrate
    gen = torch.Generator(device=dev).manual_seed(7)
    big = torch.ones((512 << 20) // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    w8 = torch.empty((8 << 20) // 4, dtype=torch.float32, device=dev)
    w40 = torch.empty((40 << 20) // 4, dtype=torch.float32, device=dev)
    ha = (torch.randn((2048, 16384), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
    hb = (torch.randn((2048, 16384), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
    hc = torch.empty((2048, 2048), dtype=torch.bfloat16, device=dev)
    cyc = timer.gate_cyc * 150 // 300

    def prefix(p):
        p = p.replace("_ug", "")
        if p == "spin":
            torch.cuda._sleep(cyc)
        elif p == "read":
            torch.sum(big, dim=0, out=sink)
        elif p == "mmh":
            torch.mm(ha, hb.t(), out=hc)
        elif p == "wr8":
            w8.fill_(1.0)
        elif p in ("wr40", "wr40s"):
            w40.fill_(1.0)
            if p == "wr40s":
                torch.cuda._sleep(cyc)

    graphs, keep = {}, []   # keep: every tensor / closure a graph reads stays alive (R-cbl-0 attempt 1 faulted without it)
    for (proj, T), (N, K, f32, tt) in GEMMS.items():
        w = (torch.randn((N, K), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        wt = w.t().contiguous() if tt else None
        x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
        out = torch.empty((T, N), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)

        def gemm(x=x, w=w, wt=wt, out=out, f32=f32):
            if wt is not None:
                torch.mm(x, wt, out=out)
            elif f32:
                torch.mm(x, w.t(), out_dtype=torch.float32)   # gc_16 today path for cat (graph-pool output)
            else:
                torch.mm(x, w.t(), out=out)
        for p in PREFIXES:
            def fn(p=p, gemm=gemm):
                prefix(p)
                gemm()
            s = timer.stream
            with torch.cuda.stream(s):
                fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    fn()
            torch.cuda.synchronize()
            graphs[(proj, T, p)] = g
            keep.append((fn, gemm, x, w, wt, out))
    print(f"[cbl] {len(graphs)} graphs captured; gate {timer.gate_cyc} cyc", flush=True)
    for k, g in graphs.items():   # one checked replay each before the timed loop
        with torch.cuda.stream(timer.stream):
            g.replay()
        torch.cuda.synchronize()
    print("[cbl] every graph replays", flush=True)
    reps = int(args.get("reps", 24))
    torch.cuda.profiler.start()
    for r in range(reps):
        keys = list(GEMMS)
        random.Random(r).shuffle(keys)
        for (proj, T) in keys:
            for p in PREFIXES:           # fixed prefix order inside a GEMM block: none_a / none_b / none_c = position controls
                g = graphs[(proj, T, p)]
                with torch.cuda.stream(timer.stream):
                    timer.flush()
                    if not p.endswith("_ug"):
                        timer.gate()
                    torch.cuda.synchronize()     # flush / gate kernels end before the range opens
                    torch.cuda.nvtx.range_push(f"C|{proj}{T}|{p}|{r}")
                    g.replay()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
    print("[cbl] done", flush=True)


def analyze(path):
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    rng = c.execute("select start, end, text from NVTX_EVENTS where text like 'C|%' order by start").fetchall()
    ks = c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    rows = {}
    j = 0
    for s, e, t in rng:
        _, g, p, _ = t.split("|")
        while j < len(ks) and ks[j][0] < s:
            j += 1
        inside, k = [], j
        while k < len(ks) and ks[k][0] < e:
            inside.append(ks[k])
            k += 1
        rows.setdefault(g, {}).setdefault(p, []).append(inside)
    res, kname = {}, {}
    for g, by in rows.items():
        m = min(len(x) for x in by["none_a"])        # the GEMM's own kernels = the last m launched in each range
        for p, lst in by.items():
            res.setdefault(g, {})[p] = [sum(q[1] - q[0] for q in x[-m:]) / 1e3 for x in lst if len(x) >= m]
        kname[g] = "+".join(names.get(q[2], "?")[:30] for q in by["none_a"][0][-m:])[:44]
    out = {}
    for g in res:
        med = {p: statistics.median(v) for p, v in res[g].items()}
        out[g] = med
        base = statistics.median([med[p] for p in ("none_a", "none_b", "none_c") if p in med])
        print(f"{g:12s} {kname[g]:44s} none {base:5.1f} | " +
              " ".join(f"{p} {med[p] - base:+.1f}" for p in PREFIXES if p in med and not p.startswith("none_")) +
              f" | ctl a/b/c {med.get('none_a', 0):.1f}/{med.get('none_b', 0):.1f}/{med.get('none_c', 0):.1f}")
    Path(path).with_suffix(".json").write_text(json.dumps(out, indent=1))


GC_PIN = {("cat", 256): (66, 125, 35, 1, 0, 0, 2, 0, 5)}       # k3_gc_16 LT_PIN entries on the probed shapes
GC_ROUTE = {("unpooler", 128): "torchT", ("unpooler", 256): "torch", ("cat", 256): "pin"}


def run_ltd(args):
    """R-cbl-1: cuBLASLt (TN + NN) enumeration of cat/256, unpooler/128, unpooler/256 with a DIRTY-L2 prefix (fill_ of
    `dirty` MB right before the GEMM, after the judge's flush + gate) vs clean; ranks configs by the time under dirty L2."""
    import torch
    sys.path.insert(0, str(ROOT / "kernels" / "lab"))
    import dn_lab as D
    from torch.utils.cpp_extension import load_inline
    lt = load_inline(name="dn_lt_v1", cpp_sources=D.LT_CPP, functions=["lt_plan", "lt_count", "lt_desc", "lt_variants", "lt_run"],
                     extra_ldflags=["-L/usr/local/cuda/lib64", "-lcublasLt"], extra_cflags=["-O2"], with_cuda=True)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(99)
    bench = D.Bench()
    WS = 32 << 20
    ws = torch.empty((WS,), dtype=torch.uint8, device=dev)
    dmb = int(args.get("dirty", 16))
    dbuf = torch.empty(((dmb << 20) // 4,), dtype=torch.float32, device=dev)
    keep, out = [], {}
    for proj, T in GC_ROUTE:
        N, K, f32 = D.SHAPES[proj]
        od = torch.float32 if f32 else torch.bfloat16
        w = (torch.randn((N, K), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        wt = w.t().contiguous()
        x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
        keep += [w, wt, x]
        yt = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        cands = {}
        for nn in (0, 1):
            pid = lt.lt_plan(T, N, K, nn, 1 if f32 else 0, 32, WS)
            nh = lt.lt_count(pid)
            for ai in range(min(nh, 4)):
                lt.lt_variants(pid, ai, 0)
            for ai in range(min(nh, 2)):
                lt.lt_variants(pid, ai, 1)
            for ai in range(lt.lt_count(pid)):
                key = ("NN" if nn else "TN",) + tuple(lt.lt_desc(pid, ai))
                cands.setdefault(key, (pid, ai, nn))
        stage1 = {}
        for key, (pid, ai, nn) in cands.items():
            y = torch.full((T, N), float("nan"), dtype=od, device=dev)
            try:
                lt.lt_run(pid, ai, x, wt if nn else w, y, ws)
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                continue
            if not bool(torch.isfinite(y).all()) or float((y.float() - yt.float()).norm() / yt.float().norm()) > 1e-2:
                continue
            ts = []
            for _ in range(3):
                bench.timer.flush()
                bench.timer.gate()
                dbuf.fill_(1.0)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                lt.lt_run(pid, ai, x, wt if nn else w, y, ws)
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b) * 1e3)
            stage1[key] = min(ts)
        top = sorted(stage1, key=stage1.get)[:8]
        pin = GC_PIN.get((proj, T))
        pkey = ("TN",) + pin if pin else None
        if pkey and pkey in stage1 and pkey not in top:
            top.append(pkey)
        route = GC_ROUTE[(proj, T)]

        def today(x=x, w=w, wt=wt, f32=f32, route=route):
            if route == "torchT":
                return torch.mm(x, wt)
            return torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        fns = {"today_torch": today}
        for key in top:
            pid, ai, nn = cands[key]
            y = torch.empty((T, N), dtype=od, device=dev)
            keep.append(y)
            fns[key] = lambda pid=pid, ai=ai, nn=nn, y=y: lt.lt_run(pid, ai, x, wt if nn else w, y, ws)
        res = {}
        for mode in ("clean", "dirty"):
            graphs = {}
            for k, f in fns.items():
                graphs[k] = bench.graph((lambda f=f: (dbuf.fill_(1.0), f())) if mode == "dirty" else f)
            graphs["fill_only"] = bench.graph(lambda: dbuf.fill_(1.0)) if mode == "dirty" else bench.graph(lambda: dbuf[:1].fill_(1.0))
            keep.append(graphs)
            res[mode] = bench.race(graphs, rounds=int(args.get("rounds", 8)), reps=10)
        base = res["dirty"]["fill_only"]
        gc = "today_torch" if route != "pin" else pkey
        ranked = sorted((k for k in fns), key=lambda k: res["dirty"][k])
        print(f"[ltd] {proj}/{T} ({len(cands)} cfgs, {len(stage1)} valid; dirty {dmb} MB, fill_only {base:.2f}): gc route {route} "
              f"clean {res['clean'].get(gc, float('nan')):.2f} dirty {res['dirty'].get(gc, float('nan')):.2f}", flush=True)
        for k in ranked[:5]:
            print(f"[ltd]    clean {res['clean'][k]:6.2f}  dirty {res['dirty'][k]:6.2f}  {k}", flush=True)
        out[f"{proj}/{T}"] = {"route": route, "gc": str(gc), "res": {m: {str(k): v for k, v in r.items()} for m, r in res.items()}}
    Path(args.get("out", "box_out/cbl1/ltd.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    a = dict(x.lstrip("-").split("=", 1) for x in sys.argv[2:] if "=" in x)
    if sys.argv[1] == "run":
        run(a)
    elif sys.argv[1] == "ltd":
        run_ltd(a)
    else:
        analyze(sys.argv[2])

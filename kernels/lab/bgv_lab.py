"""bgv_lab.py: step 0 of BGV (R-bgv-0): the 1-CTA/SM bulk-ring GEMV (kernels/lab/bgv_src.py) vs GD's shipped T = 1 / 8 routes.
    python3 -u kernels/lab/bgv_lab.py [--layer=kernels/muse_decode_layer/k3_gc_18.py] [--out=box_out/bgv0/lab.json] [--reps=40]
    local (RTX 3070): --synthetic=1 --arch=sm_86   (C = 1 only, small ring, no clusters / PDL / bulk copies)
GD block (GD_* / _gd* names) of the layer file by AST, judge weight recipe, judge-like timing: 256 MB read flush (torch.sum), a
300 us sleep gate, one CUDA graph replay, median over reps (arms interleaved per rep), net of an empty graph. Arms per (matrix, T):
GD ship (twice: first and last = controls), BGV configs (full), BGV stream-only (consumers release slots, no math), and PDL pairs
after a 4 us latency-bound predecessor (one CTA / SM spinning, launch_dependents at entry): pred alone, pred + GD, pred + BGV.
"""
import ast
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
LAYER = ROOT / ARGS.get("layer", "kernels/muse_decode_layer/k3_gc_18.py")
OUT = ROOT / ARGS.get("out", "box_out/bgv0/lab.json")
REPS = int(ARGS.get("reps", "40"))
RSEED = int(ARGS.get("rseed", "8677534258059892491"))
ARCH = ARGS.get("arch", "sm_90a")
LOCAL = not ARCH.startswith("sm_90")
SPIN_NS = int(ARGS.get("spin_ns", "4000"))
sys.path.insert(0, str(ROOT / "tools/judge"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
from bgv_src import BGV_CPP_SRC, BGV_CUDA_SRC, bgv_pack  # noqa: E402


def extract(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name and (name.startswith(("GD_", "_GD", "_gd")) or name in ("K2A_BUCKETS", "_k2a_bucket_of")):
            nodes.append(node)
    ns = {"torch": torch, "time": time, "hashlib": hashlib, "math": math, "HIDDEN": 8192, "D_EXPERT": 4096, "UNPOOL_SLICE": 1024,
          "SLICED_UNPOOL_MAX_TOKENS": 8, "V12_ZBUF": True, "R1_UNPOOL_INTO_ZEROS": True, "K2A_M_MAX": 4096, "N_EXPERTS": 256,
          "TOP_K": 8, "RMS_EPS": 1e-5, "_pt1_wait": lambda name: None, "DISABLED": [], "_k3_disable": lambda *a: None,
          "_v8_tail": None, "SU1_K3_TRIG_AT": set(), "_nb64": lambda T: False, "NB64_GD_B": {}, "NB64_USE": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)   # lab tool only
    return ns


# (C, SU, NS, NCW, MODE, ER)
INSTS_H200 = [(1, 16, 8, 8, 0, 2), (1, 16, 8, 8, 1, 2), (2, 16, 8, 8, 0, 2), (2, 16, 8, 8, 1, 2), (1, 16, 4, 8, 0, 2),
              (1, 16, 3, 8, 0, 2), (2, 16, 6, 8, 0, 2), (2, 16, 4, 8, 0, 2)]
INSTS_LOCAL = [(1, 16, 2, 8, 0, 2), (1, 16, 2, 8, 1, 2), (1, 16, 3, 8, 0, 2)]
# per matrix: BGV arms (C, SU, NS, MODE); the first full arm is the candidate used for the PDL pair
ARMS_H200 = {"qkv": [(2, 16, 8, 0), (2, 16, 6, 0), (2, 16, 8, 1)],
             "pooler": [(1, 16, 8, 0), (1, 16, 3, 0), (2, 16, 6, 0), (1, 16, 8, 1)],
             "o": [(1, 16, 8, 0), (1, 16, 4, 0), (1, 16, 8, 1)],
             "unpooler": [(2, 16, 8, 0), (2, 16, 4, 0), (2, 16, 8, 1)]}
ARMS_LOCAL = {p: [(1, 16, 2, 0), (1, 16, 2, 1), (1, 16, 3, 0)] for p in ("qkv", "pooler", "o", "unpooler")}
PROJS = [p for p in ARGS.get("projs", "qkv,pooler,o,unpooler").split(",") if p]
TS = [int(t) for t in ARGS.get("ts", "1,8").split(",")]


def main():
    ns = extract(LAYER)
    dev = "cuda"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    dims = {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (1024, 4096)}   # unpooler: T <= 8 slice
    if ARGS.get("synthetic", "0") == "1":
        g = torch.Generator(device=dev).manual_seed(3)
        W = {p: (torch.randn(dims[p], generator=g, device=dev) / math.sqrt(dims[p][1])).to(torch.bfloat16) for p in PROJS}
    else:
        import judge as J  # noqa: E402  (tools/judge on the box)
        prob = J.load_problem(ROOT / "problems/muse_decode_layer")
        ref = prob.ref
        weights = ref.generate_weights(ref.WEIGHTS_SEED, device=dev, router_seed=RSEED)
        W = {p: weights[ns["GD_WEIGHT"][p]].contiguous() for p in PROJS}
        W = {p: (w[:1024] if p == "unpooler" else w).contiguous() for p, w in W.items()}
        del weights
        torch.cuda.empty_cache()
    gencode = [f"-gencode=arch=compute_{ARCH[3:].rstrip('a')}{'a' if ARCH.endswith('a') else ''},code={ARCH}"]
    from torch.utils.cpp_extension import load_inline
    # GD baseline module: the shipped T = 1 / 8 routes
    gkeys, gmap = [], {}
    for proj in PROJS:
        for T in TS:
            rt = ns["GD_ROUTE"].get((proj, T))
            n, k = dims[proj]
            key = ns["_gd_key"](n, k, rt) if rt else None
            if key is None or (LOCAL and rt[5] > 1):
                continue
            gmap[(proj, T)] = (rt, key)
            if key not in gkeys:
                gkeys.append(key)
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in kk)) for i, kk in enumerate(gkeys))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in kk)) for i, kk in enumerate(gkeys))
    gsrc = ns["GD_CUDA_SRC"].replace("@INST@", inst).replace("@INFO@", info)
    insts = INSTS_LOCAL if LOCAL else INSTS_H200
    bsrc = BGV_CUDA_SRC.replace("@INSTS@", "\n".join("BGV_X(%d, %s)" % (i, ", ".join(map(str, v))) for i, v in enumerate(insts)))
    t0 = time.time()
    gmod = load_inline(name="bgvgd_" + hashlib.sha1((ns["GD_CPP_SRC"] + gsrc + ARCH).encode()).hexdigest()[:10],
                       cpp_sources=ns["GD_CPP_SRC"], cuda_sources=gsrc, functions=["gd_run", "gd_info"],
                       extra_cuda_cflags=["-O3"] + gencode, verbose=False)
    bmod = load_inline(name="bgv_" + hashlib.sha1((BGV_CPP_SRC + bsrc + ARCH).encode()).hexdigest()[:10],
                       cpp_sources=BGV_CPP_SRC, cuda_sources=bsrc, functions=["bgv_run", "bgv_info", "bgv_pred"],
                       extra_cuda_cflags=["-O3", "-Xptxas=-v"] + (["-DBGV_LOCAL=1"] if LOCAL else []) + gencode, verbose=False)
    print(f"[bgv] built GD ({len(gkeys)} inst) + BGV ({len(insts)} inst) in {time.time() - t0:.0f}s", flush=True)
    binfo = {}
    for i, v in enumerate(insts):
        r = bmod.bgv_info(i)
        binfo[v] = r
        print(f"[bgv] inst {i} C{v[0]} SU{v[1]} NS{v[2]} NCW{v[3]} M{v[4]}: regs {r[0]} lmem {r[1]} occ {r[2]} smem {r[3]} "
              f"clusters {r[4]}", flush=True)

    stream = torch.cuda.Stream()
    flush_buf = torch.ones((256 << 20) // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((1,), dtype=torch.float32, device=dev)
    psink = torch.zeros((sms,), dtype=torch.int32, device=dev)
    err = torch.zeros((1,), dtype=torch.int32, device=dev)
    gate_cyc = [int(300 * 1980)]
    with torch.cuda.stream(stream):
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            torch.cuda._sleep(gate_cyc[0])
            b.record()
            b.synchronize()
            gate_cyc[0] = int(gate_cyc[0] * 300 / max(a.elapsed_time(b) * 1e3, 1.0))

    graphs = []   # (name, graph)

    def capture(name, fn):
        with torch.cuda.stream(stream):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=stream):
                fn()
        torch.cuda.synchronize()
        graphs.append((name, gr))

    capture("empty", lambda: None)
    capture("pred", lambda: bmod.bgv_pred(SPIN_NS, sms, psink))
    rows, meta = {}, {}
    gen = torch.Generator(device=dev).manual_seed(5)
    keep = []
    for proj in PROJS:
        n, k = dims[proj]
        ldo = 8192 if proj == "unpooler" else n
        packs = {}
        for T in TS:
            x = torch.randn((T, k), generator=gen, device=dev).to(torch.bfloat16)
            refo = (x.double() @ W[proj].double().T)
            keep.append(x)
            outs = {}
            if (proj, T) in gmap:
                rt, key = gmap[(proj, T)]
                gi = gkeys.index(key)
                pk = ns["_gd_pack"](W[proj], rt[1], rt[2], rt[3], rt[5], ns["_gd_window_base"](W[proj]))
                y = torch.full((T, ldo), float("nan"), dtype=torch.bfloat16, device=dev)
                keep += [pk, y]

                def gd_fn(pdl, pk=pk, gi=gi, x=x, y=y, k=k, ldo=ldo, T=T):
                    return lambda: gmod.gd_run(gi, pk["w"], pk["esc"], x, y, T, k, ldo, pk["tb"], pk["nbx"], 1, pdl)
                gd_fn(2)()
                torch.cuda.synchronize()
                outs["gd"] = y[:, :n].clone()
                capture(f"{proj}/{T}/gd_a", gd_fn(2))
                if not LOCAL:
                    capture(f"{proj}/{T}/pred+gd", lambda f=gd_fn(3): (bmod.bgv_pred(SPIN_NS, sms, psink), f()))
            arms = (ARMS_LOCAL if LOCAL else ARMS_H200)[proj]
            for (C, SU, NS, MODE) in arms:
                v = (C, SU, NS, 8, MODE, 2)
                if v not in insts:
                    continue
                bi = insts.index(v)
                ncl = sms // C if C == 1 else min(sms // C, binfo[v][4])
                if ncl <= 0:
                    print(f"[bgv] {proj} C{C}: no cluster fits", flush=True)
                    continue
                pkey = (C, ncl, SU)
                if pkey not in packs:
                    try:
                        packs[pkey] = bgv_pack(ns, W[proj], C, ncl, 8, SU, 2)
                    except (AssertionError, ValueError) as ex:
                        print(f"[bgv] {proj} C{C} SU{SU}: pack skipped {ex!r}"[:200], flush=True)
                        packs[pkey] = None
                pk = packs[pkey]
                if pk is None:
                    continue
                y = torch.full((T, ldo), float("nan"), dtype=torch.bfloat16, device=dev)
                keep += [y, pk]   # graphs hold raw pointers: keep every pack alive
                name = f"{proj}/{T}/bgv_c{C}s{SU}n{NS}{'_stream' if MODE else ''}"

                def b_fn(pdl, bi=bi, pk=pk, x=x, y=y, k=k, ldo=ldo, T=T):
                    return lambda: bmod.bgv_run(bi, pk["w"], pk["esc"], x, y, T, k, ldo, pk["tb"], pk["ntiles"], pk["upi"],
                                                pk["grid"], pdl, err)
                err.zero_()
                try:
                    b_fn(2)()
                    torch.cuda.synchronize()
                except Exception as ex:
                    print(f"[bgv] {name}: LAUNCH FAILED {ex!r}"[:300], flush=True)
                    continue
                e1 = int(err.item())
                if MODE == 0:
                    got = y[:, :n].clone()
                    b_fn(2)()
                    torch.cuda.synchronize()
                    rep = bool(torch.equal(got, y[:, :n]))
                    rel = float((got.double() - refo).norm() / refo.norm())
                    gdd = None
                    if "gd" in outs:
                        dd = (got.float() - outs["gd"].float()).abs()
                        gdd = (int((dd > 0).sum()), float(dd.max()))
                    rel_gd = float((outs["gd"].double() - refo).norm() / refo.norm()) if "gd" in outs else None
                    meta[name] = {"rel": rel, "rel_gd": rel_gd, "repeat": rep, "err": e1, "vs_gd": gdd, "nesc": pk["nesc"],
                                  "maxesc": pk["maxesc"], "items": (pk["nit_min"], pk["nit_max"]), "grid": pk["grid"]}
                    print(f"[bgv] check {name}: rel {rel:.2e} (gd {rel_gd if rel_gd is None else f'{rel_gd:.2e}'}) repeat {rep} "
                          f"err {e1} vs_gd {gdd} esc {pk['nesc']}/{pk['maxesc']} items {pk['nit_min']}-{pk['nit_max']} "
                          f"grid {pk['grid']}", flush=True)
                    ok = rel < 2e-3 and rep and e1 == 0 and bool(torch.isfinite(got).all())
                    if not ok:
                        print(f"[bgv] {name}: CHECK FAILED -> not timed", flush=True)
                        continue
                    if not LOCAL:
                        capture(f"{proj}/{T}/pred+{name.split('/')[-1]}", lambda f=b_fn(3): (bmod.bgv_pred(SPIN_NS, sms, psink), f()))
                elif e1:
                    print(f"[bgv] {name}: err flag {e1}", flush=True)
                    continue
                elif not LOCAL:
                    capture(f"{proj}/{T}/pred+{name.split('/')[-1]}", lambda f=b_fn(3): (bmod.bgv_pred(SPIN_NS, sms, psink), f()))
                capture(name, b_fn(2))
            if (proj, T) in gmap:
                rt, key = gmap[(proj, T)]
                gi = gkeys.index(key)
                capture(f"{proj}/{T}/gd_b", gd_fn(2))
    # timing: arms interleaved per rep
    ts = {name: [] for name, _ in graphs}
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in graphs]
    for rep in range(REPS):
        for (name, gr), (a, b) in zip(graphs, ev):
            with torch.cuda.stream(stream):
                torch.sum(flush_buf, dim=0, out=sink)
                torch.cuda._sleep(gate_cyc[0])
                a.record()
                gr.replay()
                b.record()
            b.synchronize()
            ts[name].append(a.elapsed_time(b) * 1e3)
    empty = statistics.median(ts["empty"])
    pred = statistics.median(ts["pred"]) - empty
    res = {}
    for name, v in ts.items():
        med = statistics.median(v) - empty
        res[name] = {"net_us": round(med, 3), "q25": round(sorted(v)[len(v) // 4] - empty, 3),
                     "q75": round(sorted(v)[(3 * len(v)) // 4] - empty, 3)}
    print(f"[bgv] empty {empty:.2f} us, pred {pred:.2f} us (spin {SPIN_NS} ns), reps {REPS}", flush=True)
    for proj in PROJS:
        n, k = dims[proj]
        mb = n * k * 1.5 / 1e6
        for T in TS:
            pre = f"{proj}/{T}/"
            arms = {nm[len(pre):]: r["net_us"] for nm, r in res.items() if nm.startswith(pre)}
            if not arms:
                continue
            gd = statistics.mean([arms[a] for a in ("gd_a", "gd_b") if a in arms]) if "gd_a" in arms else None
            parts = []
            for a, us in arms.items():
                if a.startswith("pred+"):
                    parts.append(f"{a} {us - pred:.2f}")
                else:
                    parts.append(f"{a} {us:.2f}" + (f" ({mb / us:.2f}TB/s)" if a.startswith("bgv") or a.startswith("gd") else ""))
            best = min([us for a, us in arms.items() if a.startswith("bgv") and "stream" not in a], default=None)
            sp = f" best bgv vs gd {100 * (gd - best) / gd:+.1f}%" if gd and best else ""
            print(f"[bgv] {proj:8s} T={T} {mb:5.1f}MB: " + ", ".join(parts) + sp, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"empty_us": empty, "pred_us": pred, "spin_ns": SPIN_NS, "reps": REPS, "res": res, "meta": meta,
                               "info": {str(k): v for k, v in binfo.items()}, "sms": sms}, indent=1))
    print(f"[bgv] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

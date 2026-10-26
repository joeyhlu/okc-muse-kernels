"""Sliding overhead levers lab 3 (box only; final v7 vs v8 A/B + prologue sweep): K1-FA4 v8 (tools/gen_k1_fa4v8.py: producer pre-wait K1FA4_PREWAIT, fat
combine K1FA4_COMB_HB) vs v7 on the judge shapes; full kernel_fn, graph replay, cold L2 (1 GB read sweep), floor
subtracted, all variants of a shape replayed interleaved. Correctness: v8 self-test, check_implementation per shape, and a
PDL race probe (graph = poison the new tokens' cache rows with E4M3 448 -> kernel_fn, replayed 30x; every replay's
output must equal the first, which passed the check).
  python3 kernels/muse_attention_decode/k1_sl_lab2.py [--shapes 0,2,4,6,8,10,1,7,9,11]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROB = HERE.parents[1] / "problems" / "muse_attention_decode"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROB))
import reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="0,2,4,6,8,10,1,7,9,11")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--pro-bm", default="2,4,8,16,32")
ap.add_argument("--pro-warps", default="4")
ap.add_argument("--race-reps", type=int, default=10)
ap.add_argument("--selftest", type=int, default=1)
args = ap.parse_args()

import k1_fa4v7 as V7  # noqa: E402
import k1_fa4v8 as V8  # noqa: E402

dev = torch.device("cuda")
benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
flush_buf = torch.ones(256 * 1024 * 1024, dtype=torch.float32, device=dev)
sink = torch.empty((), dtype=torch.float32, device=dev)


def say(*a):
    print(*a, flush=True)


def time_interleaved(graphs, reps):
    ts = [[] for _ in graphs]
    for _ in range(reps):
        for j, g in enumerate(graphs):
            torch.sum(flush_buf, dim=0, out=sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts[j].append(a.elapsed_time(b) * 1000.0)
    return [statistics.median(t) for t in ts]


def capture(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg):
    pass
FLOOR = time_interleaved([eg], 50)[0]


def setv8(pre=False, hb=-1, bm=0, warps=4):
    V8.K1FA4_PREWAIT = pre
    V8.K1FA4_COMB_HB = hb
    V8.K1FA4_PRO_BM = bm
    V8.K1FA4_PRO_WARPS = warps


if args.selftest:
    setv8()
    t0 = time.time()
    ok = V8.k1fa4_self_test(dev)
    say(f"[sl3] v8 self-test (default: pre-wait off, comb auto) {'PASS' if ok else 'FAIL'} {time.time() - t0:.1f}s")


def label(i):
    bm = benches[i]
    return f"[{i:2d}] {bm['load']} {bm['layer']} B{bm['batch']}"


for i in [int(x) for x in args.shapes.split(",")]:
    bm = benches[i]
    kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
    t0 = time.time()
    # correctness of the default v8 config + race probe
    setv8()
    data = reference.generate_input(**kw)
    out = V8.k1fa4_run(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    hkv, d = data["config"]["num_kv_heads"], data["config"]["head_dim"]
    kc = data["k_cache"].view(-1, hkv * d).view(torch.uint8)
    vc = data["v_cache"].view(-1, hkv * d).view(torch.uint8)
    cl = data["cache_loc"]
    ref_out = out.clone()
    res = {}

    def poisoned():
        kc.index_fill_(0, cl, 0x7E)
        vc.index_fill_(0, cl, 0x7E)
        res["o"] = V8.k1fa4_run(data)
    gr = capture(poisoned)
    same = 0
    for _ in range(args.race_reps):
        torch.sum(flush_buf, dim=0, out=sink)
        gr.replay()
        torch.cuda.synchronize()
        same += int(torch.equal(res["o"], ref_out))
    del gr
    say(f"CHK {label(i)} v8 {'PASS' if ok else 'FAIL'} {msg[:70]} | race probe: {same}/{args.race_reps} replays == first")
    # timing: v7 vs v8 default (pre-wait off, auto fat combine), then prologue rows-per-program sweep on v8
    variants = [("v7", lambda: V7.k1fa4_run(data), None), ("v8", None, (False, -1, 0, 4))]
    for bm_ in [int(x) for x in args.pro_bm.split(",")]:
        for w_ in [int(x) for x in args.pro_warps.split(",")]:
            variants.append((f"v8 proBM{bm_}w{w_}", None, (False, -1, bm_, w_)))
    variants.append(("v7b", lambda: V7.k1fa4_run(data), None))
    variants.append(("v8b", None, (False, -1, 0, 4)))
    gs, names = [], []
    for nm, fn, knobs in variants:
        try:
            if knobs is not None:
                setv8(*knobs)
                fn = (lambda: V8.k1fa4_run(data))
            gs.append(capture(fn))
            names.append(nm)
        except Exception as ex:  # noqa: BLE001
            say(f"   {nm}: ERR {type(ex).__name__} {str(ex)[:160]}")
    setv8()
    ts = [x - FLOOR for x in time_interleaved(gs, args.reps)]
    base = ts[0]
    say(f"SL3 {label(i)} | " + " | ".join(f"{n} {t:6.2f} ({t - base:+.2f})" for n, t in zip(names, ts)))
    del gs, data, out, ref_out, res
    torch.cuda.empty_cache()
    say(f"   ({time.time() - t0:.1f}s)")
say("[sl3] done")

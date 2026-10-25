"""K1-FA4 v6 lab (box only): v6 self-test, then v5 (runtime-patched loader) vs v6 (plain vendored source) in one
process on the attention problem's 12 scored shapes (full kernel_fn, graph replay, cold L2), with v6 checked by the
problem's check_implementation.

  python3 kernels/muse_attention_decode/k1_fa4v6_lab.py [--shapes 0,...,11] [--reps 40]
"""
import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROB = HERE.parents[1] / "problems" / "muse_attention_decode"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROB))
import reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="0,1,2,3,4,5,6,7,8,9,10,11")
ap.add_argument("--reps", type=int, default=40)
args = ap.parse_args()

t0 = time.time()
import k1_fa4v6 as V6  # noqa: E402

print(f"[lab6] import k1_fa4v6 {time.time() - t0:.1f}s", flush=True)
t0 = time.time()
ok = V6.k1fa4_self_test(torch.device("cuda"))
print(f"[lab6] v6 self-test {'PASS' if ok else 'FAIL'} in {time.time() - t0:.1f}s (incl. JIT)", flush=True)
import k1_fa4v5 as V5  # noqa: E402

dev = torch.device("cuda")
benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
flush_buf = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
sink = torch.empty((), dtype=torch.float32, device=dev)


def time_graph(g, reps):
    ts = []
    for _ in range(reps):
        torch.sum(flush_buf, dim=0, out=sink)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts)


eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg):
    pass
floor = time_graph(eg, 50)
for i in [int(x) for x in args.shapes.split(",")]:
    bm = benches[i]
    kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
    row = []
    for name, fn in (("v5", lambda d: V5.run(d)), ("v6", lambda d: V6.k1fa4_run(d)), ("v5b", lambda d: V5.run(d)),
                     ("v6b", lambda d: V6.k1fa4_run(d))):
        try:
            data = reference.generate_input(**kw)
            out = fn(data)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, reference.collect(data, out))
            for _ in range(2):
                fn(data)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(data)
            t = time_graph(g, args.reps) - floor
            row.append(f"{name} {t:7.2f}us{'' if ok else ' FAIL(' + str(msg)[:60] + ')'}")
            del g, data
        except Exception as ex:  # noqa: BLE001
            row.append(f"{name} ERR {type(ex).__name__}: {str(ex)[:160]}")
            print("\n".join("TB| " + ln for ln in traceback.format_exc()[-2000:].splitlines()), flush=True)
        torch.cuda.empty_cache()
    cfg = {"q_len": bm.get("q_len", 1), "page_size": 1, "kv_scale": 1.0, "head_dim": 64, "num_q_heads": 16,
           "num_kv_heads": 2, "window": 2048 if bm["layer"] == "sliding" else 0, "batch": bm["batch"]}
    print(f"BENCH [{i:2d}] {bm['load']} {bm['layer']} (v6 route {V6.k1fa4_route_ok(cfg)}) | " + " | ".join(row),
          flush=True)

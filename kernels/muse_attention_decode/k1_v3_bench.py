"""K1 v3 variant bench on the box (R5: one run = many experiments). Judge-style timing: one CUDA graph per
(shape, variant), cold L2 (256 MB read) before every timed replay, median over reps, empty-graph replay subtracted.
Every variant is also checked with the problem's reference.check_implementation on the same inputs.

  python3 kernels/muse_attention_decode/k1_v3_bench.py [--file k1_v3.py] [--shapes 6,7,...] [--variants spec] [--reps 40]
variants spec: ';'-separated name=key:val,key:val (keys: fold, min_chunk, cps, nsg, stages) or name=r1 (rung-1 FA3 path)
"""
import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PROB = ROOT / "problems" / "muse_attention_decode"
sys.path.insert(0, str(PROB))
import reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--file", default=str(Path(__file__).with_name("k1_v3.py")))
ap.add_argument("--shapes", default="")
ap.add_argument("--variants", default="def=;r1=r1")
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--no-check", action="store_true")
ap.add_argument("--prof", action="store_true", help="K1_PROF build: print per-phase CTA timestamps (one cold call)")
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("k1sub", args.file)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
dev = torch.device("cuda")
_st_ok = mod._self_test(dev)
assert _st_ok or args.no_check, "self-test failed"

benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
idx = [int(x) for x in args.shapes.split(",")] if args.shapes else list(range(len(benches)))
variants = []
for part in args.variants.split(";"):
    name, _, kv = part.partition("=")
    if kv == "r1":
        variants.append((name, None))
        continue
    kw = {}
    for item in filter(None, kv.split(",")):
        k, _, v = item.partition(":")
        kw[k] = (v == "1") if k == "fold" else int(v)
    variants.append((name, kw))

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
print(f"BENCH empty-graph replay {floor:.2f} us (subtracted)", flush=True)

for i in idx:
    bm = benches[i]
    kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
    data = reference.generate_input(**kw)
    label = f"[{i:2d}] {bm['load']:9s} {bm['layer']:7s} B{bm['batch']} L{bm.get('q_len', 1)}"
    row = []
    for name, vkw in variants:
        fn = (lambda d: mod._fa3_path(d)) if vkw is None else (lambda d, vkw=vkw: mod._k1(d, **vkw))
        try:
            out = fn(data)
            torch.cuda.synchronize()
            ok_s = ""
            if not args.no_check:
                ok, msg = reference.check_implementation(data, reference.collect(data, out))
                ok_s = "" if ok else " FAIL(" + msg[:80] + ")"
            if args.prof and vkw is not None:
                for _ in range(3):
                    fn(data)
                mod._MOD.k1_prof_read()
                torch.sum(flush_buf, dim=0, out=sink)
                fn(data)
                pr = mod._MOD.k1_prof_read().view(4096, 16).double()
                v = pr[pr[:, 0] > 0]
                t0 = v[:, 0].min()
                parts = []
                for sl in range(11):
                    col = v[:, sl]
                    col = col[col > 0]
                    if len(col):
                        d = (col - t0) / 1000.0
                        parts.append(f"s{sl} med {d.median():.2f} max {d.max():.2f} (n{len(col)})")
                print(f"PROF {label} {name} ctas={len(v)}: " + "; ".join(parts), flush=True)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn(data)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(data)
            t = time_graph(g, args.reps) - floor
            row.append(f"{name} {t:7.2f}{ok_s}")
            del g
        except Exception as ex:  # noqa: BLE001
            row.append(f"{name} ERR {str(ex)[:120]}")
    print(f"BENCH {label} | " + " | ".join(row), flush=True)
    del data
    torch.cuda.empty_cache()

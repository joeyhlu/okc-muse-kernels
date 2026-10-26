"""Sliding-window attention overhead lab (box only): where do the per-call microseconds of the K1-FA4 v7 path go?
Per shape (judge inputs, graph replay, cold L2 = 1 GB read sweep, empty-graph floor subtracted, variants interleaved):
  fn      prologue -> core -> combine, PDL (k1fa4_run)          fn_nopdl  the same without PDL
  pro     prologue alone          core   FA4 core alone (k1fa4_fwd_partials)          comb   combine+norm-gate alone
  pc      prologue -> core (PDL)  cm     core -> combine (PDL)
  e1 / e3 one / three empty Triton kernels (three PDL-chained)
and a split sweep (core alone and fn) per shape.
  python3 kernels/muse_attention_decode/k1_sl_lab.py [--shapes 0,2,4,6,8,10,1,7] [--splits 2,4,8,16,32]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
PROB = HERE.parents[1] / "problems" / "muse_attention_decode"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROB))
import reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="0,2,4,6,8,10,1,7")
ap.add_argument("--splits", default="2,4,8,16,32")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--mod", default="k1_fa4v7")
args = ap.parse_args()

M = __import__(args.mod)
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


@triton.jit
def _empty(X, USE_PDL: tl.constexpr):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg):
    pass
FLOOR = time_interleaved([eg], 50)[0]
dummy = torch.zeros(1, device=dev)
ge1 = capture(lambda: _empty[(1,)](dummy, USE_PDL=False))
ge3 = capture(lambda: [_empty[(1,)](dummy, USE_PDL=True, launch_pdl=True) for _ in range(3)])
e1, e3 = [x - FLOOR for x in time_interleaved([ge1, ge3], 50)]
say(f"[sl] floor {FLOOR:.2f} us; empty kernel {e1:.2f} us; 3 empty PDL-chained {e3:.2f} us")


def pieces(data, ns, pdl=True):
    """Closures for the three launches of M.k1fa4_run (same arguments), sharing q8 / partial / out buffers."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    window = cfg["window"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    q8 = torch.empty((tokens, hq * d), dtype=torch.float8_e4m3fn, device=dev)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = 4 if n_rows < 1024 else (8 if n_rows < 8192 else 16)
    theta = cfg["rope_theta"]
    cos_sin = M.k1fa4_cos_sin(theta, d, dev) if theta else qkv
    kw = {"launch_pdl": True} if pdl else {}
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    rows = tokens * hq
    st = {}

    def pro():
        M._k1fa4_prologue[(triton.cdiv(n_rows, block_m),)](
            qkv, q8, data["positions"] if theta else qkv, cos_sin, data["k_cache"].view(-1, hkv * d),
            data["v_cache"].view(-1, hkv * d), data["cache_loc"], M.K1FA4_RMS_EPS, qkv.stride(0), n_rows, hq, hkv, d,
            triton.next_power_of_2(d), block_m, HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=4, **kw)

    def core():
        M.k1fa4_K["pdl"] = True
        st["o"], st["l"] = M.k1fa4_fwd_partials(q8.view(batch, length, hq, d), data["k_cache"], data["v_cache"],
                                                data["page_table"], data["cache_seqlens"],
                                                cfg["sm_scale"] * cfg["qk_scale_factor"], ns, window=window)

    def comb():
        M._k1fa4_combine_gate[(rows,)](
            st["o"], st["l"], qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, M.K1FA4_RMS_EPS,
            H=hq, D=d, NS_P2=triton.next_power_of_2(ns), USE_PDL=pdl, num_warps=2 if ns <= 8 else 4, **kw)
    pro()
    core()
    comb()
    torch.cuda.synchronize()
    return pro, core, comb


def default_splits(data):
    cfg = data["config"]
    window, length, batch = cfg["window"], cfg["q_len"], cfg["batch"]
    max_k = data["page_table"].shape[1]
    keys = min(max_k, window + length) if window else max_k
    return (M.K1FA4_SPLITS.get(("sliding" if window else "full", batch))
            or M.k1fa4_num_splits_for(batch, cfg["num_kv_heads"], keys))


for i in [int(x) for x in args.shapes.split(",")]:
    bm = benches[i]
    kwg = {k: v for k, v in bm.items() if k not in ("load", "weight")}
    data = reference.generate_input(**kwg)
    ns0 = default_splits(data)
    t0 = time.time()
    pro, core, comb = pieces(data, ns0)
    pro_n, core_n, comb_n = pieces(data, ns0, pdl=False)
    gs = {
        "fn": capture(lambda: (pro(), core(), comb())),
        "fn_nopdl": capture(lambda: (pro_n(), core_n(), comb_n())),
        "pro": capture(pro), "core": capture(core), "comb": capture(comb),
        "pc": capture(lambda: (pro(), core())), "cm": capture(lambda: (core(), comb())),
    }
    names = list(gs)
    t = dict(zip(names, (x - FLOOR for x in time_interleaved([gs[n] for n in names], args.reps))))
    keys = int(data["cache_seqlens"].float().mean())
    say(f"SL [{i:2d}] {bm['load']} {bm['layer']} B{bm['batch']} splits {ns0} keys~{keys} | "
        + " | ".join(f"{n} {t[n]:6.2f}" for n in names)
        + f" | fn-core {t['fn'] - t['core']:5.2f}")
    del gs
    sw = []
    for s in [int(x) for x in args.splits.split(",")]:
        try:
            p2, c2, m2 = pieces(data, s)
            sw.append((s, capture(c2), capture(lambda p2=p2, c2=c2, m2=m2: (p2(), c2(), m2()))))
        except Exception as ex:  # noqa: BLE001
            say(f"   split {s}: ERR {type(ex).__name__} {str(ex)[:120]}")
    if sw:
        ts = [x - FLOOR for x in time_interleaved([g for _, a, b in sw for g in (a, b)], args.reps)]
        say(f"   splits sweep (core / fn): " + " | ".join(
            f"S{s} {ts[2 * j]:6.2f} / {ts[2 * j + 1]:6.2f}" for j, (s, _, _) in enumerate(sw)))
    del sw, data
    torch.cuda.empty_cache()
    say(f"   ({time.time() - t0:.1f}s)")
say("[sl] done")

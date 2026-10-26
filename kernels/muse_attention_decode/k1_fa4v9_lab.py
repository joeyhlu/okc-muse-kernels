"""K1-FA4 v9 lab (box only): bound the verify-full FP8-PV core (k1_fa4v9.py = v8 + K1FA4_MASK / K1FA4_IWO knobs).
Core = k1fa4_fwd_partials alone (prologue run once outside the graph), judge inputs, graph replay, cold L2 (1 GB read
sweep), all configs of a shape replayed interleaved. Configs: name=mask:stages:iwo (stages / iwo for full layers).
  python3 kernels/muse_attention_decode/k1_fa4v9_lab.py --shapes 11,9 --cfgs base=0:3:0,pvoff=1:3:0,...
  [--selftest-iwo 1] [--check 11,9,10,8]
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
ap.add_argument("--shapes", default="11,9")
ap.add_argument("--cfgs", default="base=0:3:0")
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--selftest-iwo", type=int, default=0)
ap.add_argument("--check", default="")
ap.add_argument("--fn", type=int, default=0)
ap.add_argument("--selftest-cfg", default="0:3:1")   # 1: time the full kernel_fn instead of the core
args = ap.parse_args()

import k1_fa4v8 as V8  # noqa: E402
import k1_fa4v9 as V9  # noqa: E402

dev = torch.device("cuda")
benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
flush = torch.ones(256 * 1024 * 1024, dtype=torch.float32, device=dev)
sink = torch.empty((), dtype=torch.float32, device=dev)


def say(*a):
    print(*a, flush=True)


DEF = dict(pv8=V9.K1FA4_PV8[("full", 8)], tile=V9.K1FA4_TILE, minb=V9.k1fa4_K["minb"], regs=dict(V9.K1FA4_REGS1))


def setcfg(mask, stages, iwo, pv8=1, tilen=128, minb=2, rc=0, rp=0, splits=0):
    """lab knobs; every change drops the in-process compile dict (its key does not hold the tile)"""
    V9._k1fa4_compiled.clear()
    V9.K1FA4_MASK = mask
    V9.K1FA4_STAGES["full"] = stages
    V9.K1FA4_IWO["full"] = bool(iwo)
    V9.K1FA4_PV8[("full", 8)] = bool(pv8)
    V9.K1FA4_TILE = (64, tilen)
    V9.k1fa4_K["minb"] = minb
    V9.K1FA4_REGS1.update(DEF["regs"])
    if rc:
        V9.K1FA4_REGS1[bool(pv8)] = (rc, rp)
    V9.K1FA4_SPLITS.pop(("full", 32), None)
    V9.K1FA4_SPLITS.pop(("full", 16), None)
    if splits:
        V9.K1FA4_SPLITS[("full", 32)] = splits
        V9.K1FA4_SPLITS[("full", 16)] = 2 * splits


def timeit(graphs, reps):
    ts = [[] for _ in graphs]
    for _ in range(reps):
        for j, g in enumerate(graphs):
            torch.sum(flush, dim=0, out=sink)
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


def core_closure(M, data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    q8 = torch.empty((tokens, hq * d), dtype=torch.float8_e4m3fn, device=dev)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = 4 if n_rows < 1024 else (8 if n_rows < 8192 else 16)
    theta = cfg["rope_theta"]
    cos_sin = M.k1fa4_cos_sin(theta, d, dev) if theta else qkv
    M._k1fa4_prologue[(M.triton.cdiv(n_rows, block_m),)](
        qkv, q8, data["positions"] if theta else qkv, cos_sin, data["k_cache"].view(-1, hkv * d),
        data["v_cache"].view(-1, hkv * d), data["cache_loc"], M.K1FA4_RMS_EPS, qkv.stride(0), n_rows, hq, hkv, d,
        M.triton.next_power_of_2(d), block_m, HAS_ROPE=bool(theta), USE_PDL=False, num_warps=4)
    window = cfg["window"]
    max_k = data["page_table"].shape[1]
    keys = min(max_k, window + length) if window else max_k
    ns = M.K1FA4_SPLITS.get(("sliding" if window else "full", batch)) or M.k1fa4_num_splits_for(batch, hkv, keys)
    M.k1fa4_K["pdl"] = True
    scale = cfg["sm_scale"] * cfg["qk_scale_factor"]
    q8v = q8.view(batch, length, hq, d)

    def fn():
        return M.k1fa4_fwd_partials(q8v, data["k_cache"], data["v_cache"], data["page_table"], data["cache_seqlens"],
                                    scale, ns, window=window)
    return fn


cfgs = []
for c in args.cfgs.split(","):
    nm, spec = c.split("=")
    cfgs.append((nm, [int(x) for x in spec.split(":")]))

if args.selftest_iwo:
    setcfg(*[int(x) for x in args.selftest_cfg.split(":")])
    t0 = time.time()
    ok = V9.k1fa4_self_test(dev)
    say(f"[v9] self-test cfg {args.selftest_cfg}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
for i in [int(x) for x in args.check.split(",") if x]:
    kw = {k: v for k, v in benches[i].items() if k not in ("load", "weight")}
    msgs = []
    for nm, cf in cfgs:
        if cf[0]:
            continue
        setcfg(*cf)
        d = reference.generate_input(**kw)
        out = V9.k1fa4_run(d)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(d, reference.collect(d, out))
        msgs.append(f"{nm} {'PASS' if ok else 'FAIL'} {msg[:46]}")
        del d
    say(f"CHK [{i}] {benches[i]['load']} {benches[i]['layer']}: " + " | ".join(msgs))
setcfg(0, 3, 0)

for i in [int(x) for x in args.shapes.split(",") if x]:
    bm = benches[i]
    kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
    data = reference.generate_input(**kw)
    kvb = int(data["cache_seqlens"].sum()) * 256
    graphs, names = [], []
    f8 = core_closure(V8, data) if not args.fn else (lambda: V8.k1fa4_run(data))
    graphs.append(capture(f8))
    names.append("v8")
    for nm, cf in cfgs:
        try:
            setcfg(*cf)
            f = core_closure(V9, data) if not args.fn else (lambda: V9.k1fa4_run(data))
            graphs.append(capture(f))
            names.append(nm)
        except Exception as ex:  # noqa: BLE001
            say(f"   {nm}: ERR {type(ex).__name__}: {str(ex)[:300]}")
    setcfg(0, 3, 0)
    t = timeit(graphs, args.reps)
    base = t[0]
    say(f"CORE [{i:2d}] {bm['load']} {bm['layer']} B{bm['batch']} ({kvb / 1e6:.0f} MB) | "
        + " | ".join(f"{n} {x:7.2f} ({x - base:+.2f}, {kvb / x / 1e6:.2f} TB/s)" for n, x in zip(names, t)))
    del graphs, data
    torch.cuda.empty_cache()
say("[v9] done")

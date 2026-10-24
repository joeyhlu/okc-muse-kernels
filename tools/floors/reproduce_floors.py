"""Reproduce every published floor (TASK.md 'floor' column) from each problem's own
reference.speed_of_light_ns, and recover the hidden task.yml routing fields.

Usage (CPU is fine, no GPU needed):
  python3 -I tools/floors/reproduce_floors.py                 # floors + back-solve + MoE simulation
  python3 -I tools/floors/reproduce_floors.py --sim-layer 300 # also simulate decode_layer routing (slow, ~15 min)
  python3 -I tools/floors/reproduce_floors.py --json out.json

What it does
1. For each problem p, copies problems/p/{reference,checks,nvfp4}.py into a temp dir and imports
   reference.py from there (fresh module names per problem, so `checks`/`nvfp4` siblings resolve).
   speed_of_light_ns only reads hw["hbm_bytes_per_s"] and hw["bf16_dense_flops_per_s"]
   (verified by reading every reference.py), so no CUDA stub is needed to call it.
2. Evaluates speed_of_light_ns(**benchmark, hw=HW) for every scored shape in meta.json, with the
   router fields left at their defaults (0), and splits it into its byte and flop sides by calling
   it again with an infinite FLOP rate / infinite bandwidth.
3. For MoE / decode_layer / prefill_layer, back-solves router_distinct (if bytes bind) or
   router_load (if flops bind) from the published floor, as an interval because floors are
   printed to 0.1 us.
4. Independently estimates the fields by simulating generate_input on CPU over many seeds
   (MoE: the routing generate_input draws; decode_layer: the judge's weights_fields inner loop,
   _mean_busiest_load over many router seeds).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"]
HW = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
HW_BYTES_ONLY = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 1e300}
HW_FLOPS_ONLY = {"hbm_bytes_per_s": 1e300, "bf16_dense_flops_per_s": 989e12}
ROUTED = {"muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"}
_TMP = tempfile.TemporaryDirectory(prefix="okc_floors_")


def load_reference(problem: str):
    """Import problems/<problem>/reference.py from a temp copy with its checks/nvfp4 siblings."""
    src = ROOT / "problems" / problem
    dst = Path(_TMP.name) / problem
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("reference.py", "checks.py", "nvfp4.py"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    for mod in ("reference", "checks", "nvfp4"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(dst))
    try:
        ref = importlib.import_module("reference")
    finally:
        sys.path.remove(str(dst))
    for mod in ("reference", "checks", "nvfp4"):  # keep the module objects, free the names
        sys.modules.pop(mod, None)
    return ref


def cpu_fp8_shim(ref):
    """CPU torch 2.5 has no index kernels for float8_e4m3fn. Make the reference keep its FP8
    cache as float32 tensors holding exactly the E4M3-rounded values (fp8 -> fp32 is exact), so
    generate_input/ref_front run on CPU with identical values. Used only for the CPU simulations."""
    ref.FP8 = torch.float32
    ref._quant = lambda x, scale: (x.float() / scale).to(torch.float8_e4m3fn).float()
    return ref


def parse_task_table(problem: str):
    """[(load, shape, weight, floor_us)] from TASK.md's '| load | shape | weight | floor |' table."""
    rows = []
    for line in (ROOT / "problems" / problem / "TASK.md").read_text().splitlines():
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([\d,\.]+)\s*µs\s*\|$", line)
        if m:
            rows.append((m[1], m[2], int(m[3]), float(m[4].replace(",", ""))))
    return rows


def sol_parts(ref, bench: dict, **fields):
    """(floor_us, bytes_us, flops_us, bytes, flops) for one benchmark dict."""
    args = {**bench, **fields}
    t = ref.speed_of_light_ns(**args, hw=HW) / 1e3
    tb = ref.speed_of_light_ns(**args, hw=HW_BYTES_ONLY) / 1e3
    tf = ref.speed_of_light_ns(**args, hw=HW_FLOPS_ONLY) / 1e3
    return t, tb, tf, tb * 1e-6 * HW["hbm_bytes_per_s"], tf * 1e-6 * HW["bf16_dense_flops_per_s"]


def _solve(fn, target, lo=0.0, hi=1e7, iters=200):
    """x >= lo with fn(x) == target for nondecreasing fn, or None if fn(lo) > target or fn(hi) < target."""
    if fn(lo) > target or fn(hi) < target:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def back_solve(ref, bench: dict, published_us: float):
    """Interval of router_distinct if the byte side binds, and of router_load if the flop side
    binds, consistent with the published floor printed to 0.1 us (+-0.05)."""
    lo_t, hi_t = published_us - 0.05, published_us + 0.05
    tb = lambda rd: ref.speed_of_light_ns(**bench, router_distinct=rd, hw=HW_BYTES_ONLY) / 1e3  # noqa: E731
    tf = lambda rl: ref.speed_of_light_ns(**bench, router_load=rl, hw=HW_FLOPS_ONLY) / 1e3  # noqa: E731
    rd = (_solve(tb, lo_t), _solve(tb, hi_t))
    rl = (_solve(tf, lo_t), _solve(tf, hi_t))
    return {"rd_if_bytes_bind": rd, "rl_if_flops_bind": rl, "tb0": tb(0.0), "tf0": tf(0.0)}


# ----------------------------------------------------------------------------------------------
# Independent estimates by simulation
# ----------------------------------------------------------------------------------------------

def sim_moe(ref, bench: dict, n: int, seed0: int = 10_000_000):
    """Mean (routes, distinct) to the local experts 0..31 of generate_input's routing (rank busiest)."""
    routes, distinct = [], []
    lo, count = ref.LOCAL[bench["sharding"]]
    for i in range(n):
        d = ref.generate_input(bench["bs"], bench["sharding"], seed0 + i, device="cpu")
        ids = d["topk_ids"].long()
        local = ids[(ids >= lo) & (ids < lo + count)]
        routes.append(local.numel())
        distinct.append(torch.unique(local).numel())
    r, dd = torch.tensor(routes, dtype=torch.float64), torch.tensor(distinct, dtype=torch.float64)
    return {"router_load": float(r.mean()), "router_load_se": float(r.std() / math.sqrt(n)),
            "router_distinct": float(dd.mean()), "router_distinct_se": float(dd.std() / math.sqrt(n)), "n": n}


def sim_layer(ref, benches: list[dict], n_routers: int, seed0: int = 20_000_000, ctx_cap: int | None = None):
    """The judge's weights_fields inner loop, run over n_routers router seeds instead of searching:
    for each case, h = ref_front(...)['h'] over ROUTE_SAMPLES inputs (fresh seeds), then
    _mean_busiest_load(h, router, n) per router. Returns per-case mean/sd over routers, plus the
    value under the default router (WEIGHTS_SEED). ctx_cap shrinks contexts (approximation: routing
    depends on attention only through rmsnorm(o-proj output)); None = exact contexts."""
    cpu_fp8_shim(ref)
    torch.manual_seed(0)
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cpu")
    hs = []
    for c in benches:
        case = {k: v for k, v in c.items() if k not in ("load", "weight", "repeats")}
        if ctx_cap is not None:
            case["ctx"] = min(case["ctx"], ctx_cap)
        tokens = case.get("batch", 1) * case.get("q_len", 1)
        n = ref.ROUTE_SAMPLES if tokens <= ref.SAMPLE_TOKENS else 1
        rows = []
        with torch.no_grad():
            for j in range(n):
                data = ref.generate_input(**{**case, "seed": seed0 + 1000 * len(hs) + j, "local_start": 0}, device="cpu")
                rows.append(ref.ref_front(data, weights)["h"])
                del data
        hs.append((torch.cat(rows), n))
        print(f"  h for {c['load']:>10s} {case['layer']:>7s}: {tokens} tokens x {n} inputs", file=sys.stderr, flush=True)
    per_router = []
    default = torch.stack([ref._mean_busiest_load(h, {"router": weights["router"], "router_bias": weights["router_bias"]}, n)
                           for h, n in hs])
    for i in range(1, n_routers + 1):
        rs = (seed0 + i * 0x9E3779B97F4A7C15) % (2**63 - 1)
        router = ref._router(torch.Generator(device="cpu").manual_seed(rs), "cpu", weights["pre_ffn_norm"])
        per_router.append(torch.stack([ref._mean_busiest_load(h, router, n) for h, n in hs]))
    pr = torch.stack(per_router).double()  # [routers, cases, 2]
    return {"mean": pr.mean(0), "sd": pr.std(0), "median": pr.median(0).values, "default_router": default.double(),
            "n_routers": n_routers}


# ----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-seeds", type=int, default=2000)
    ap.add_argument("--sim-layer", type=int, default=0, help="router seeds for the decode_layer simulation (0 = skip)")
    ap.add_argument("--layer-ctx-cap", type=int, default=None)
    ap.add_argument("--sim-prefill", type=int, default=0,
                    help="router seeds for an approximate prefill_layer simulation (contexts capped, see --prefill-ctx-cap)")
    ap.add_argument("--prefill-ctx-cap", type=int, default=4096)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = {}
    print(f"hw = {HW}\n")
    for p in PROBLEMS:
        ref = load_reference(p)
        meta = json.loads((ROOT / "problems" / p / "meta.json").read_text())
        table = parse_task_table(p)
        benches = meta["benchmarks"]
        assert len(table) == len(benches), (p, len(table), len(benches))
        print(f"## {p}  ({len(benches)} scored shapes)\n")
        print("| load | shape | published µs | ours µs | delta % | rounds to published | binds | bytes (MB) | flops (GFLOP) | bytes µs | flops µs |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for (load, shape, weight, pub), b in zip(table, benches):
            assert load == b["load"] and weight == b["weight"], (p, load, b)
            t, tb, tf, nbytes, nflops = sol_parts(ref, b)
            delta = 100 * (t - pub) / pub
            binds = "bytes" if tb >= tf else "flops"
            rnd = abs(round(t, 1) - pub) < 1e-9
            rows.append({"load": load, "shape": shape, "published": pub, "ours": t, "delta_pct": delta, "binds": binds,
                         "bytes": nbytes, "flops": nflops, "bytes_us": tb, "flops_us": tf, "rounds": rnd})
            print(f"| {load} | {shape} | {pub:,.1f} | {t:,.2f} | {delta:+.2f} | {'yes' if rnd else 'NO'} | {binds} |"
                  f" {nbytes / 1e6:,.2f} | {nflops / 1e9:,.2f} | {tb:,.2f} | {tf:,.2f} |")
        print()
        out[p] = {"rows": rows}
        if p in ROUTED:
            print(f"### {p}: back-solved hidden fields (floors printed to 0.1 µs => intervals)\n")
            print("| load | shape | published µs | lower-bound floor (fields=0) | router_distinct if bytes bind | router_load if flops bind |")
            print("|---|---|---|---|---|---|")
            bs_rows = []
            for (load, shape, weight, pub), b in zip(table, benches):
                r = back_solve(ref, b, pub)
                fmt = lambda iv: "n/a" if iv[0] is None and iv[1] is None else (  # noqa: E731
                    f"[{iv[0] if iv[0] is not None else 0:.2f}, {iv[1] if iv[1] is not None else float('inf'):.2f}]")
                t0 = max(r["tb0"], r["tf0"])
                print(f"| {load} | {shape} | {pub:,.1f} | {t0:,.2f} | {fmt(r['rd_if_bytes_bind'])} | {fmt(r['rl_if_flops_bind'])} |")
                bs_rows.append(r)
            print()
            out[p]["back_solve"] = bs_rows
        if p == "muse_moe_w4a16" and args.moe_seeds:
            print(f"### {p}: simulated routing over {args.moe_seeds} CPU seeds (mean ± SE), and the floor it implies\n")
            print("| load | bs | router_load sim | router_distinct sim | floor with sim fields µs | published µs | delta % | binds |")
            print("|---|---|---|---|---|---|---|---|")
            sims = []
            for (load, shape, weight, pub), b in zip(table, benches):
                n = args.moe_seeds if b["bs"] <= 32 else max(200, args.moe_seeds // 4)
                s = sim_moe(ref, b, n)
                t, tb, tf, *_ = sol_parts(ref, b, router_load=s["router_load"], router_distinct=s["router_distinct"])
                print(f"| {load} | {b['bs']} | {s['router_load']:.2f} ± {s['router_load_se']:.2f} |"
                      f" {s['router_distinct']:.2f} ± {s['router_distinct_se']:.2f} | {t:,.2f} | {pub:,.1f} |"
                      f" {100 * (t - pub) / pub:+.2f} | {'bytes' if tb >= tf else 'flops'} |")
                sims.append({**s, "floor": t})
            print()
            out[p]["sim"] = sims
        n_sim = args.sim_layer if p == "muse_decode_layer" else args.sim_prefill if p == "muse_prefill_layer" else 0
        if n_sim:
            cap = args.layer_ctx_cap if p == "muse_decode_layer" else args.prefill_ctx_cap
            print(f"### {p}: simulated busiest-GPU load over {n_sim} router seeds (judge's weights_fields loop)"
                  f"{'' if cap is None else f', contexts capped at {cap} (approximation)'}\n")
            s = sim_layer(ref, benches, n_sim, ctx_cap=cap)
            print("| load | shape | routes mean (sd) | distinct mean (sd) | default router routes / distinct | floor with sim means µs | published µs | delta % | binds |")
            print("|---|---|---|---|---|---|---|---|---|")
            sims = []
            for i, ((load, shape, weight, pub), b) in enumerate(zip(table, benches)):
                rl, rd = float(s["mean"][i, 0]), float(s["mean"][i, 1])
                t, tb, tf, *_ = sol_parts(ref, b, router_load=rl, router_distinct=rd)
                dr = s["default_router"][i]
                print(f"| {load} | {shape} | {rl:.2f} ({float(s['sd'][i, 0]):.2f}) | {rd:.2f} ({float(s['sd'][i, 1]):.2f}) |"
                      f" {float(dr[0]):.2f} / {float(dr[1]):.2f} | {t:,.2f} | {pub:,.1f} | {100 * (t - pub) / pub:+.2f} |"
                      f" {'bytes' if tb >= tf else 'flops'} |")
                sims.append({"router_load": rl, "router_distinct": rd, "floor": t,
                             "sd": [float(s["sd"][i, 0]), float(s["sd"][i, 1])]})
            print()
            out[p]["sim"] = sims
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()

"""layer_ab3 = layer_ab2 + a bitwise output comparison of every variant vs variant 0 after each seed.
In-layer A/B of lever variants of ONE layer file, judge-style (run from /root/okc):
    python3 tools/box/layer_ab2.py kernels/muse_decode_layer/k3_v08.py tools/box/ab_variants.json [--shapes=0,1] \
        [--seeds=4] [--pairs=4] [--router-seed=8677534258059892491] [--check=1]
variants JSON: [{"name": "v07", "set": {"V8_DENSE_GEMV_T1": false, ...}}, ...]; variant 0 is the reference.
One process, one prepare. Per shape: one CUDA graph per variant (module globals set during capture, restored after),
each on its own static inputs; per seed fresh judge inputs (generate_input + case_fields, router re-drawn to the
task.yml targets) copied into all; 256 MB L2 flush before every replay; the replay order rotates every round.
Prints per variant the median µs and the median of per-round ratios t(variant 0) / t(variant) (> 1: faster than 0).
"""
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as J  # noqa: E402
from layer_budget import FIELDS  # noqa: E402


def arg(name, default):
    for a in sys.argv[3:]:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def keyfix(v):
    if isinstance(v, dict) and any(isinstance(k, str) and "," in k for k in v):   # "sliding,16,8" -> ("sliding", 16, 8)
        return {tuple(int(p) if p.isdigit() else p for p in k.split(",")): keyfix(x) for k, x in v.items()}
    if isinstance(v, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): keyfix(x) for k, x in v.items()}
    return v


def main():
    prob = J.load_problem(Path("problems/muse_decode_layer"))
    ref = prob.ref
    for b in prob.benches:
        b.update(FIELDS[b["load"]])
    rs = arg("router-seed", 8677534258059892491)
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda", router_seed=rs)
    mod = J.load_kernel(Path(sys.argv[1]), "okc_submission", prob)
    variants = [{"name": v["name"], "set": keyfix(v.get("set", {}))} for v in json.loads(Path(sys.argv[2]).read_text())]
    for kv in arg("preset", "").split(","):
        if kv:
            k, v = kv.split("=")
            setattr(mod, k, json.loads(v))
    state = mod.prepare(J.clone_data(weights))
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    shapes = [int(s) for s in arg("shapes", ",".join(str(i) for i in range(len(prob.benches)))).split(",")]
    n_seeds, n_pairs, do_check = arg("seeds", 4), arg("pairs", 4), arg("check", 1)
    call = lambda d: mod.kernel_fn(d, state)  # noqa: E731
    print(f"[ab2] overhead {ov:.2f} us; {len(variants)} variants: {[v['name'] for v in variants]}", flush=True)
    summary = {}
    for idx in shapes:
        case = prob.benches[idx]
        t0 = time.time()
        d0 = J.make_input(ref, case, case["seed"], weights, "cuda")
        T = d0["config"]["tokens"]
        statics, graphs = [], []
        for v in variants:
            saved = {k: copy.deepcopy(getattr(mod, k)) for k in v["set"]}
            for k, x in v["set"].items():
                setattr(mod, k, copy.deepcopy(x))
            st = J.clone_data(d0)
            try:
                graphs.append(timer.capture(call, st))
            finally:
                for k, x in saved.items():
                    setattr(mod, k, x)
            statics.append(st)
        times = [[] for _ in variants]
        ratios = [[] for _ in variants]
        bad = {}
        beq = {}
        rnd = 0
        for si in range(n_seeds):
            data = J.make_input(ref, case, case["seed"] + J.SEED_STEP * (si + 1), weights, "cuda")
            chk = J.clone_data(data) if do_check and si == n_seeds - 1 else None
            for st in statics:
                J.copy_into(st, data)
            torch.cuda.synchronize()
            for _ in range(n_pairs):
                order = list(range(len(variants)))
                order = order[rnd % len(order):] + order[:rnd % len(order)]
                rnd += 1
                t = {}
                for i in order:
                    t[i] = max(timer.time_replay(graphs[i][0]) - ov, 1e-3)
                for i in range(len(variants)):
                    times[i].append(t[i])
                    ratios[i].append(t[0] / t[i])
            for i in range(1, len(variants)):   # ab3: bitwise vs variant 0 (same inputs, last replay of each)
                a, b = graphs[i][1], graphs[0][1]
                a = a[0] if isinstance(a, (tuple, list)) else a
                b = b[0] if isinstance(b, (tuple, list)) else b
                beq.setdefault(i, []).append(bool(torch.equal(a, b)))
            if chk is not None:
                for i, v in enumerate(variants):
                    ok, msg = J.check_output(ref, chk, statics[i], graphs[i][1], weights)
                    if not ok:
                        bad[v["name"]] = msg
        line = [f"[ab2] [{idx:2d}] {case['load']:>10s} {case['layer']:>7s} T={T:3d}:"]
        for i, v in enumerate(variants):
            med = statistics.median(times[i])
            r = statistics.median(ratios[i])
            summary.setdefault(v["name"], {})[idx] = (med, r)
            line.append(f"{v['name']} {med:7.1f}us x{r:.3f}")
        print("  ".join(line) + f"  ({time.time() - t0:.0f}s)", flush=True)
        print("[ab3]      bitwise vs " + variants[0]["name"] + ": " + ", ".join(
            f"{variants[i]['name']} {sum(v)}/{len(v)}" for i, v in sorted(beq.items())), flush=True)
        for k, msg in bad.items():
            print(f"[ab2]      CHECK FAILED {k}: {msg}", flush=True)
        del graphs, statics
        torch.cuda.synchronize()
    # load-level geomean of each variant vs variant 0, judge weights, using variant 0's µs as the SGLang-free base
    import math
    loads = {}
    for idx in shapes:
        c = prob.benches[idx]
        loads.setdefault(c["load"], []).append(idx)
    for v in variants:
        acc = []
        for ld, ids in loads.items():
            base = sum(prob.benches[i]["weight"] * summary[variants[0]["name"]][i][0] for i in ids)
            mine = sum(prob.benches[i]["weight"] * summary[variants[0]["name"]][i][0] / summary[v["name"]][i][1]
                       for i in ids)
            acc.append(math.log(base / mine))
        print(f"[ab2] {v['name']:>16s}: load-geomean vs {variants[0]['name']} x{math.exp(sum(acc) / len(acc)):.4f}"
              f" over {len(loads)} loads", flush=True)


if __name__ == "__main__":
    main()

"""layer_abf.py: in-layer A/B of DIFFERENT layer files, judge-style, in one process (run from /root/okc):

    python3 tools/box/layer_abf.py tools/box/<arms>.json [--shapes=0,1,..] [--seeds=4] [--pairs=4]
        [--router-seed=8677534258059892491] [--check=1] [--out=box_out/x/ab.json]

arms JSON: [{"name": "vml8", "file": "kernels/lab/k3_vml_8_ab.py", "set": {...}}, ...]; arm 0 is the reference. Each
distinct file is imported once under its own module name and prepared once on the same judge weights (identical CUDA
sources share their built extension). Per shape: one CUDA graph per arm (its module's globals from "set" applied during the
capture, restored after), each on its own static inputs; per seed fresh judge inputs (generate_input + case_fields)
copied into every arm; 256 MB L2 flush before every replay; the replay order rotates every round. Graphs are kept until
the shape is done (never destroyed between arms). Prints per arm the median us and the median of per-round ratios
t(arm 0) / t(arm); the JSON has layer_ab2's layout (tools/box/vd_ab_summary.py / vms_ab_summary.py read it).
"""
import copy
import json
import math
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
    for a in sys.argv[2:]:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def keyfix(v):
    if isinstance(v, dict) and any(isinstance(k, str) and "," in k for k in v):
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
    raw_arms = json.loads(Path(sys.argv[1]).read_text())   # written back as given: keyfix makes tuple keys JSON cannot hold
    arms = [{"name": a["name"], "file": a["file"], "set": keyfix(a.get("set", {}))} for a in raw_arms]
    mods, states = {}, {}
    for i, f in enumerate(dict.fromkeys(a["file"] for a in arms)):
        t0 = time.time()
        try:
            mods[f] = J.load_kernel(Path(f), f"okc_submission_{i}", prob)
            states[f] = mods[f].prepare(J.clone_data(weights))
            torch.cuda.synchronize()
            print(f"[abf] prepared {f} in {time.time() - t0:.1f}s", flush=True)
        except Exception as ex:   # one broken candidate must not void the whole session (arm 0 must prepare)
            if i == 0:
                raise
            print(f"[abf] PREPARE FAILED {f}: {ex!r:.300} -> arm dropped", flush=True)
            mods.pop(f, None); states.pop(f, None)
    arms = [a for a in arms if a["file"] in states]
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    shapes = [int(s) for s in arg("shapes", ",".join(str(i) for i in range(len(prob.benches)))).split(",")]
    n_seeds, n_pairs, do_check = arg("seeds", 4), arg("pairs", 4), arg("check", 1)
    print(f"[abf] overhead {ov:.2f} us; {len(arms)} arms: {[(a['name'], Path(a['file']).name) for a in arms]}", flush=True)
    summary, measurements = {}, []
    for idx in shapes:
        case = prob.benches[idx]
        t0 = time.time()
        d0 = J.make_input(ref, case, case["seed"], weights, "cuda")
        T = d0["config"]["tokens"]
        statics, graphs = [], []
        for a in arms:
            mod, st_ = mods[a["file"]], states[a["file"]]
            saved = {k: copy.deepcopy(getattr(mod, k)) for k in a["set"]}
            for k, x in a["set"].items():
                setattr(mod, k, copy.deepcopy(x))
            st = J.clone_data(d0)
            try:
                graphs.append(timer.capture(lambda d, mod=mod, st_=st_: mod.kernel_fn(d, st_), st))
            finally:
                for k, x in saved.items():
                    setattr(mod, k, x)
            statics.append(st)
        times = [[] for _ in arms]
        ratios = [[] for _ in arms]
        bad = {}
        rnd = 0
        for si in range(n_seeds):
            data = J.make_input(ref, case, case["seed"] + J.SEED_STEP * (si + 1), weights, "cuda")
            chk = J.clone_data(data) if do_check and (do_check > 1 or si == n_seeds - 1) else None
            for st in statics:
                J.copy_into(st, data)
            torch.cuda.synchronize()
            for _ in range(n_pairs):
                order = list(range(len(arms)))
                order = order[rnd % len(order):] + order[:rnd % len(order)]
                rnd += 1
                t = {}
                for i in order:
                    t[i] = max(timer.time_replay(graphs[i][0]) - ov, 1e-3)
                for i in range(len(arms)):
                    times[i].append(t[i])
                    ratios[i].append(t[0] / t[i])
            if chk is not None:
                for i, a in enumerate(arms):
                    ok, msg = J.check_output(ref, chk, statics[i], graphs[i][1], weights)
                    if not ok:
                        bad[a["name"]] = msg
        line = [f"[abf] [{idx:2d}] {case['load']:>10s} {case['layer']:>7s} T={T:3d}:"]
        for i, a in enumerate(arms):
            med, r = statistics.median(times[i]), statistics.median(ratios[i])
            summary.setdefault(a["name"], {})[idx] = (med, r)
            line.append(f"{a['name']} {med:7.1f}us x{r:.3f}")
        print("  ".join(line) + f"  ({time.time() - t0:.0f}s)", flush=True)
        for k, msg in bad.items():
            print(f"[abf]      CHECK FAILED {k}: {msg}", flush=True)
        measurements.append(dict(index=idx, case=case, variants=[dict(name=a["name"], times_us=times[i], ratios=ratios[i])
                                                                  for i, a in enumerate(arms)], failures=bad))
        if arg("out", ""):
            dst = Path(arg("out", ""))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(dict(module="files", arms=raw_arms, router_seed=rs, shapes=measurements), indent=2, default=str))
        del graphs, statics
        torch.cuda.synchronize()
    loads = {}
    for idx in shapes:
        loads.setdefault(prob.benches[idx]["load"], []).append(idx)
    for a in arms:
        acc = []
        for ld, ids in loads.items():
            base = sum(prob.benches[i]["weight"] * summary[arms[0]["name"]][i][0] for i in ids)
            mine = sum(prob.benches[i]["weight"] * summary[arms[0]["name"]][i][0] / summary[a["name"]][i][1] for i in ids)
            acc.append(math.log(base / mine))
        print(f"[abf] {a['name']:>10s}: load-geomean vs {arms[0]['name']} x{math.exp(sum(acc) / len(acc)):.4f} over {len(loads)} loads",
              flush=True)


if __name__ == "__main__":
    main()

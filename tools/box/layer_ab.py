"""In-layer A/B of the experts route per scored shape (run from /root/okc):
    python3 tools/box/layer_ab.py kernels/muse_decode_layer/k3_v07.py [--shapes 0,1,4,5] [--seeds 6] [--pairs 6]
One process, one prepare. Per shape: two CUDA graphs of the same k3_v07 kernel_fn, captured with the shape's bucket
routed to K2A v2.2 (A) and to r1 Marlin (B), on separate static inputs (as the judge does); fresh judge-style inputs
per seed (copied into both); L2 flush before every replay; order alternates per pair. Prints raw medians and the
median of paired B/A ratios (> 1: K2A faster)."""
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/root/okc/tools/judge")
import judge as J  # noqa: E402


def arg(name, default):
    for a in sys.argv[2:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def main():
    prob = J.load_problem(Path("problems/muse_decode_layer"))
    ref = prob.ref
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")
    wf = ref.weights_fields(weights, prob.benches, 0) if hasattr(ref, "weights_fields") else {}
    if wf.get("router_seed") is not None:
        weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda", router_seed=wf["router_seed"])
        print(f"[ab] router_seed {wf['router_seed']}", flush=True)
    mod = J.load_kernel(Path(sys.argv[1]), "okc_submission", prob)
    state = mod.prepare(J.clone_data(weights))
    assert state["k2a"] is not None and state["k1_ok"], "K2A/K1 not live"
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    shapes = [int(s) for s in arg("shapes", ",".join(str(i) for i in range(len(prob.benches)))).split(",")]
    n_seeds, n_pairs = int(arg("seeds", "6")), int(arg("pairs", "6"))
    call = lambda d: mod.kernel_fn(d, state)  # noqa: E731
    print(f"[ab] overhead {ov:.2f} us; seeds {n_seeds} x pairs {n_pairs}", flush=True)
    for idx in shapes:
        case = prob.benches[idx]
        t0 = time.time()
        d0 = J.make_input(ref, case, case["seed"], weights, "cuda")
        T = d0["config"]["tokens"]
        rep = mod._k2a_bucket_of(T)
        saved = mod.K2A_ROUTE[rep]
        static = {s: J.clone_data(d0) for s in "AB"}
        graphs = {}
        for side, route in (("A", "v2"), ("B", "marlin")):
            mod.K2A_ROUTE[rep] = route
            graphs[side] = timer.capture(call, static[side])
        mod.K2A_ROUTE[rep] = saved
        ta, tb, ratios = [], [], []
        for si in range(n_seeds):
            data = J.make_input(ref, case, case["seed"] + J.SEED_STEP * (si + 1), weights, "cuda")
            for side in "AB":
                J.copy_into(static[side], data)
            for p in range(n_pairs):
                order = "AB" if (si * n_pairs + p) % 2 == 0 else "BA"
                t = {}
                for side in order:
                    t[side] = max(timer.time_replay(graphs[side][0]) - ov, 1e-3)
                ta.append(t["A"])
                tb.append(t["B"])
                ratios.append(t["B"] / t["A"])
        print(f"[ab] [{idx:2d}] {case['load']:>10s} {case['layer']:>7s} T={T:3d} bucket {rep}: K2A {statistics.median(ta):7.1f} us"
              f"  Marlin {statistics.median(tb):7.1f} us  Marlin/K2A x{statistics.median(ratios):.3f}"
              f"  (n={len(ratios)}, {time.time() - t0:.0f}s)", flush=True)
        del graphs, static
        torch.cuda.synchronize()
    print(f"[ab] err flag {int(state['k2a']['err'][0])}", flush=True)


if __name__ == "__main__":
    main()

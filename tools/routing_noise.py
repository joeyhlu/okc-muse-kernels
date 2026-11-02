"""FP8-query vs BF16-query router-score noise on the quantities check_routing uses (2026-11-02, spare session, $0, local 3070).
"""
import sys, json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "judge"))
import judge as J
prob = J.load_problem(ROOT / "problems" / "muse_decode_layer")
ref = prob.ref
RS = int(sys.argv[1]) if len(sys.argv) > 1 else 8677534258059892491
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = "cuda"
weights = ref.generate_weights(ref.WEIGHTS_SEED, device=dev, router_seed=RS)
FP8 = torch.float8_e4m3fn
res = {}
for idx, case in enumerate(prob.benches):
    d1s, d2s, dabs = [], [], []
    for rep in range(reps):
        seed = case["seed"] + J.SEED_STEP * (rep + 1)
        args = {**J.case_args(case), "seed": seed, "device": dev, "local_start": 0}
        with torch.no_grad():
            data = ref.generate_input(**args)
            truth = ref.ref_front(data, weights)["biased"].float().cpu()
            ours = ref.ref_front(data, weights, FP8)["biased"].float().cpu()
        dabs.append((ours - truth).abs().flatten())
        for T0, (a, b) in enumerate([(ours, truth)]):
            ta, tb = a.topk(9, -1), b.topk(9, -1)
            ninth_a, ninth_b = ta.values[:, 8:9], tb.values[:, 8:9]
            eight_a, eight_b = ta.values[:, 7:8], tb.values[:, 7:8]
            # keep-side error on our top-8: |(v - nine)_ours - (v - nine)_truth| with v = the same expert's score
            va = ta.values[:, :8]; vb = b.gather(1, ta.indices[:, :8])
            d1s.append(((va - ninth_a) - (vb - ninth_b)).abs().flatten())
            # route-side error for the 40 best-scoring candidates: |(eight - s)_ours - (eight - s)_truth|
            top40 = a.topk(40, -1).indices
            sa = a.gather(1, top40); sb = b.gather(1, top40)
            d2s.append(((eight_a - sa) - (eight_b - sb)).abs().flatten())
        del data
        torch.cuda.empty_cache()
    d1 = torch.cat(d1s); d2 = torch.cat(d2s); da = torch.cat(dabs)
    q = lambda x, p: float(torch.quantile(x[: min(len(x), 4_000_000)], p))
    res[idx] = dict(load=case["load"], layer=case["layer"], n=len(d1), keep_max=float(d1.max()), keep_q999=q(d1, .999),
                    route_max=float(d2.max()), route_q999=q(d2, .999), abs_max=float(da.max()), abs_q999=q(da, .999))
    print(idx, json.dumps(res[idx]), flush=True)
print("overall keep_max", max(r["keep_max"] for r in res.values()), "route_max", max(r["route_max"] for r in res.values()))

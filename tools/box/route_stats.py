"""Routes / distinct local experts / per-expert token counts per layer benchmark for one router seed (box, GPU, ~1 min).
Tells how far the layer's MoE is from its byte floor (distinct x 28.31 MB).
    python3 tools/box/route_stats.py --router-seed=<seed>
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as J  # noqa: E402
from layer_budget import FIELDS, arg  # noqa: E402


def main():
    prob = J.load_problem(Path("problems/muse_decode_layer"))
    ref = prob.ref
    rs = arg(sys.argv[1:], "router-seed", -1)
    for b in prob.benches:
        b.update(FIELDS[b["load"]])
    w = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda", router_seed=rs)
    with torch.no_grad():
        for i, c in enumerate(prob.benches):
            case = {k: v for k, v in c.items() if k not in ("load", "weight", "router_load", "router_distinct")}
            ls = ref.case_fields(case, w)["local_start"]
            d = ref.generate_input(**{**case, "local_start": ls})
            ids = ref.ref_front(d, w)["biased"].topk(ref.TOP_K, dim=-1).indices
            loc = ids[(ids - ls) % ref.N_EXPERTS < ref.LOCAL_EXPERTS]
            cnt = torch.bincount(((loc - ls) % ref.N_EXPERTS).flatten(), minlength=32)
            nz = sorted(cnt[cnt > 0].tolist(), reverse=True)
            print(f"[{i:2d}] {c['load']:9s} {c['layer']:7s} T={ids.shape[0]:3d} routes {int(loc.numel()):4d} distinct {len(nz):2d} "
                  f"({len(nz) * 28.31:6.1f} MB) per-expert {nz[:12]}", flush=True)


if __name__ == "__main__":
    main()

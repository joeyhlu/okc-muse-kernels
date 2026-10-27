"""CPU model of the SWEEP MoE schedule (notes/study/moe_redesign_independent.md) over simulated moe_w4a16 board routings.

Re-implements generate_input's routing math (normal dist, ep8, busiest rank) on a CPU generator; does not import the
reference. For each bs it reports: touched bytes, X/inter traffic per weight byte for the lockstep filler and for the
tiles, tile count and byte share, the per-SM byte imbalance bound, the worst per-SM tile tensor time, SMs whose tile
bytes exceed their byte target (filler < 0), and the weight re-read share (token groups for T_e > 192).

usage: python tools/sweep_model.py [n_routings=200] [theta=8] [rmax_bs256=256]
  (run in WSL: tools/local/cudaenv.sh python tools/sweep_model.py ...; needs torch)
"""
import math
import statistics as st
import sys

import torch

N_EXPERTS, TOPK, COUNT, POP, NSM = 256, 8, 32, 0.03, 132
GRAN = 16 * 4096 // 2 + 16 * 256      # 36,864 B: one 16-row full-K granule (E2M1 codes + E4M3 scales)
EXP = 768 * GRAN                      # 28.31 MB per expert (512 W13 + 256 W2 granules)
SM_BW = 4.45e12 / NSM                 # per-SM share of the measured stream rate
TILE_MACS = 2.6e12                    # per-SM RS-wgmma MAC/s used for the tile tensor-time column


def routing(bs, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.randn((bs, 4096), generator=g)                 # consume the RNG like generate_input (x first)
    s = torch.rand((bs, N_EXPERTS), generator=g)
    z = torch.randn((N_EXPERTS,), generator=g)
    u = torch.rand((N_EXPERTS,), generator=g).clamp(min=1e-12)
    s = s + POP * z / (-u.log()).sqrt()
    ids = s.topk(TOPK, dim=-1).indices
    p = torch.bincount((ids // COUNT).flatten(), minlength=N_EXPERTS // COUNT)
    ids = (ids - COUNT * p.argmax()) % N_EXPERTS
    return [v for v in torch.bincount(ids[ids < COUNT], minlength=COUNT).tolist() if v > 0]


def tile_shape(t, rmax):
    """(N, R, token groups) for an expert with t tokens; R x N <= 24.6K accumulators per CTA (2 tile WGs at 144 regs)."""
    n = ((t + 7) // 8) * 8
    if n <= 48:
        return n, rmax, 1
    if n <= 96:
        return n, 256, 1
    if n <= 192:
        return n, 128, 1
    groups = math.ceil(n / 192)
    return ((math.ceil(t / groups) + 7) // 8) * 8, 128, groups


def model(bs, theta, rmax, seeds):
    rows = {k: [] for k in ("mb", "xf", "xt", "ntile", "frac", "imb", "tens", "neg", "reread")}
    for sd in seeds:
        c = routing(bs, sd)
        total = len(c) * EXP
        tiles, reread = [], 0
        for t in c:
            if t <= theta:
                continue
            n, r, groups = tile_shape(t, rmax)
            for rows_kind in (8192, 4096):
                for _ in range(rows_kind // r):
                    for gi in range(groups):
                        tiles.append((r * 4096 * n, (r * 2048 + r * 256) if gi == 0 else 0, (t / groups) * 8192))
            reread += (groups - 1) * EXP
        tiles.sort(key=lambda x: -x[0])
        tb, tc = [0.0] * NSM, [0.0] * NSM
        for i, x in enumerate(tiles):                     # serpentine deal by decreasing compute
            j = i % NSM
            sm = j if (i // NSM) % 2 == 0 else NSM - 1 - j
            tc[sm] += x[0]
            tb[sm] += x[1]
        target = total / NSM
        rows["mb"].append(total / 1e6)
        rows["xf"].append(NSM * sum(t for t in c if t <= theta) * 2 * 8192 / total)
        rows["xt"].append(sum(x[2] for x in tiles) / total)
        rows["ntile"].append(len(tiles))
        rows["frac"].append(sum(x[1] for x in tiles) / total)
        rows["imb"].append((max(0.0, max(tb) - target) + 2 * GRAN) / SM_BW * 1e6)
        rows["tens"].append(max(tc) / TILE_MACS * 1e6 if tiles else 0.0)
        rows["neg"].append(sum(1 for b in tb if target - b < -GRAN))
        rows["reread"].append(reread / total)
    f = lambda k, p=3: f"{st.mean(rows[k]):.{p}g}/{max(rows[k]):.{p}g}"
    print(f"bs {bs:3d} th {theta:3d} rmax {rmax}: MB {f('mb')} | X/W filler {f('xf')} tiles {f('xt')} | tiles {f('ntile')} "
          f"byte share {f('frac')} | imbalance bound us {f('imb')} | worst-SM tile tensor us {f('tens')} | "
          f"SMs filler<0 {f('neg')} | re-read {f('reread')}   (mean/max)")


if __name__ == "__main__":
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    th = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rmax256 = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    seeds = range(5000, 5000 + nr)
    for bs in (1, 8, 16, 32, 128, 256):
        model(bs, 99 if bs <= 8 else th, rmax256 if bs == 256 else 512, seeds)

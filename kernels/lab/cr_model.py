"""CR (clean-room MoE) paper model: per-shape time bound from measured constants (notes/study/moe_cleanroom.md s2).

Structure modelled (CR): 1 CTA/SM x 16 mma.sync consumer warps (R_w = 32 rows per warp, all tokens of the expert in one
weight pass up to 64 tokens), cost-balanced contiguous (item, k-stage) ranges per CTA, W13 launch -> PDL -> W2 launch
-> PDL -> finalize. Per stage the SM is either byte-bound (BW) or tensor-bound (mma.sync at TC); the plan equalizes
the sum of stage costs per CTA, so the phase time = sum of costs / 132 + start + tail.

  python kernels/lab/cr_model.py [n_routings=200]
"""
import math
import statistics
import sys

import torch

W13_MB, W2_MB = 18.874, 9.437          # per expert incl. scales (8192 x 4096 / 2 * 1.125; 4096 x 4096 / 2 * 1.125)
W13_MAC, W2_MAC = 8192 * 4096, 4096 * 4096
SGL = {1: 80.7, 8: 143.6, 16: 188.6, 32: 214.3, 128: 470.4, 256: 621.9}   # judge us (competitor_moe.md s1)
K2A = {1: 36.6, 8: 73.1, 16: 102.1, 32: 140.7, 128: 246.2, 256: 314.6}
TPASS = 64


def routing(bs, seed):
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand((bs, 256), generator=g)
    z = torch.randn((256,), generator=g)
    u = torch.rand((256,), generator=g).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    ids = scores.topk(8, dim=-1).indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    loc = ids[ids < 32]
    return [int(c) for c in torch.bincount(loc, minlength=32) if c > 0]


def phase_us(cnts, mb, mac, bw, tc):
    tot = 0.0
    for t in cnts:
        npass = math.ceil(t / TPASS)
        for p in range(npass):
            tp = min(TPASS, t - p * TPASS)
            tpad = 8 * math.ceil(tp / 8)
            tot += max(mb * 1e6 / (bw * 1e12) * 1e6, mac * tpad / (tc * 1e12) * 1e6)
    return tot


def call_us(cnts, bw, tc, fixed):
    return fixed + phase_us(cnts, W13_MB, W13_MAC, bw, tc) + phase_us(cnts, W2_MB, W2_MAC, bw, tc)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    scen = {"central": (3.8, 225e0, 7.0), "low": (3.5, 200e0, 9.0), "high": (4.0, 245e0, 6.0)}
    print("bs | experts | hottest | CR central / low / high us | K2A us | leader model us | CR ratio vs SGLang")
    lnsum = {k: 0.0 for k in scen}
    lnk = 0.0
    for bs in (1, 8, 16, 32, 128, 256):
        rs = [routing(bs, 990001 + 1000 * bs + r) for r in range(n)]
        ne = statistics.mean(len(c) for c in rs)
        hot = statistics.median(max(c) for c in rs)
        res = {}
        for k, (bw, tc, fx) in scen.items():
            res[k] = statistics.mean(call_us(c, bw, tc, fx) for c in rs)
            lnsum[k] += math.log(SGL[bs] / res[k])
        lnk += math.log(SGL[bs] / K2A[bs])
        lead = ne * 28.31 / 3.6 + 5
        print(f"{bs} | {ne:.2f} | {hot:.0f} | {res['central']:.1f} / {res['low']:.1f} / {res['high']:.1f} | {K2A[bs]} | "
              f"{lead:.1f} | {SGL[bs] / res['central']:.2f}")
    print("board geomean: " + ", ".join(f"{k} {math.exp(v / 6):.3f}" for k, v in lnsum.items()) +
          f"; K2A {math.exp(lnk / 6):.3f}")


if __name__ == "__main__":
    main()

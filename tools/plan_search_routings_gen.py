# H200 emulation: torch's CUDA RNG offset increment of the big x randn depends on the grid (min(SMs x 8, numel / 1024));
# on a 132-SM H200 it is 4 for every bs <= 1056 (the x tensor bs x 4096 needs <= 1056 blocks); on the 46-SM 3070 it is
# 8 / 12 at bs 128 / 256. A dummy randn of 1024 elements advances the generator by 4 = the H200 path.
"""Regenerates tools/plan_search_routings.json: _mb_board_input(bs, seed 100..139) routings (local expert counts in id order)
as an H200 would draw them, on any CUDA GPU (run in WSL: /root/okcenv/bin/python tools/plan_search_routings_gen.py)."""
import json, os, torch
def board_input(bs, seed, dev):
    gen = torch.Generator(device=dev).manual_seed(7919 * seed + 13 * bs + 1)
    torch.randn((1024,), generator=gen, device=dev)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    ids = scores.topk(8, dim=-1).indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return ids
out = {}
for bs in (16, 32, 128, 256):
    out[bs] = []
    for s in range(100, 140):
        ids = board_input(bs, s, "cuda").flatten().cpu()
        out[bs].append(torch.bincount(ids[ids < 32], minlength=32).tolist())
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_search_routings.json"), "w"))
for bs in (128, 256):
    for i in range(6): print(bs, i, sum(out[bs][i]), sorted([x for x in out[bs][i] if x], reverse=True)[:8], len([x for x in out[bs][i] if x]))

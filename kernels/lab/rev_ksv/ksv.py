"""R-rev-18 (notes/study/independent_review_oct28.md s25): KSV prototype - one-launch sliding attention for the 8-token verify
block at B > 1 (kernels/lab/rev_ksv/ksv.cu). Loader, attention-problem adapter, and a correctness test against the problem's own
reference / check_implementation.

  python ksv.py test        # local (RTX 3070 sm_86) or H200: correctness on short / long / edge cases, B 1 / 3 / 16 / 32
"""
import hashlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CPP = """void ksv_forward(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor acc_o, torch::Tensor acc_l, torch::Tensor counters, torch::Tensor out, int64_t NS,
                 double qk_sm, double eps, int64_t rope);
std::vector<int64_t> ksv_info(int64_t NS);"""


def build(verbose=False):
    from torch.utils.cpp_extension import load_inline
    cc = torch.cuda.get_device_capability()
    sfx = "a" if cc[0] == 9 else ""
    gen = f"-gencode=arch=compute_{cc[0]}{cc[1]}{sfx},code=sm_{cc[0]}{cc[1]}{sfx}"
    src = (HERE / "ksv.cu").read_text(encoding="utf-8")
    tag = hashlib.sha1((src + gen).encode()).hexdigest()[:10]
    return load_inline(name=f"rev_ksv_{tag}", cpp_sources=CPP, cuda_sources=src, functions=["ksv_forward", "ksv_info"],
                       extra_cuda_cflags=["-O3", gen, "-Xptxas=-warn-spills"], verbose=verbose)


def make_state(device, max_b, maxpos, theta=500000.0):
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
    ang = torch.arange(maxpos, device=device).float()[:, None] * inv[None, :]
    return {"cos_sin": torch.cat([ang.cos(), ang.sin()], 1).contiguous(),
            "acc_o": torch.zeros(max_b, 128, 64, device=device), "acc_l": torch.zeros(max_b, 128, device=device),
            "counters": torch.zeros(max_b, dtype=torch.int32, device=device)}


def forward(ext, data, state, NS):
    cfg = data["config"]
    B, Lq = cfg["batch"], cfg["q_len"]
    assert Lq == 8 and cfg["page_size"] == 1 and cfg["window"] == 2048 and float(cfg["kv_scale"]) == 1.0
    qkv = data["qkv"].reshape(B * Lq, -1)
    out = torch.empty(B * Lq, 1024, dtype=torch.bfloat16, device=qkv.device)
    ext.ksv_forward(qkv, data["seq_lens"], data["cache_loc"], data["positions"], data["page_table"],
                    data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), state["cos_sin"],
                    state["acc_o"], state["acc_l"], state["counters"], out, NS,
                    cfg["sm_scale"] * cfg["qk_scale_factor"], 1e-5, 1 if cfg["rope_theta"] else 0)
    return out.view(B, Lq, 1024)


def _ref():
    p = str(ROOT / "problems/muse_attention_decode")
    if p not in sys.path:
        sys.path.insert(0, p)
    import reference as R
    return R


def check(ext, state, B, ctx, seed, NS, spread="agentic", dist="soft", reps=2):
    R = _ref()
    data = R.generate_input(batch=B, ctx=ctx, heads="tp8", layer="sliding", seed=seed, q_len=8, spread=spread, dist=dist, page=1)
    res = []
    for _ in range(reps):   # repeat: the accumulators / counters must reset themselves
        d2 = dict(data)
        d2["k_cache"], d2["v_cache"] = data["k_cache"].clone(), data["v_cache"].clone()
        out = forward(ext, d2, state, NS)
        torch.cuda.synchronize()
        ok, msg = R.check_implementation(data, R.collect(d2, out))
        res.append((ok, msg, out.float().clone()))
    same = torch.equal(res[0][2], res[-1][2])
    clean = int(state["counters"].abs().sum()) == 0 and float(state["acc_o"].abs().sum()) == 0.0
    return res[0][0] and same and clean, f"B {B} ctx {ctx} seed {seed} NS {NS} {spread}/{dist}: {res[0][1]} | repeat equal {same} state clean {clean}"


def main(argv):
    torch.manual_seed(0)
    ext = build()
    big = torch.cuda.get_device_properties(0).shared_memory_per_block_optin >= 160 * 1024
    nss = (4, 8) if big else (8,)   # NS 4 needs 152 KB of smem (H200 only)
    for ns in nss:
        print(f"[ksv] NS {ns} regs / local / smem / occupancy: {list(ext.ksv_info(ns))}", flush=True)
    state = make_state("cuda", 32, 140000)
    cases = [(1, 3000, 11), (3, 2100, 12), (3, 1500, 13), (3, 2045, 14), (16, 2600, 15), (16, 80000, 16), (32, 80000, 17)]
    if len(argv) > 1 and argv[1] == "quick":
        cases = cases[:5]
    allok = True
    for B, ctx, seed in cases:
        for ns in nss:
            ok, msg = check(ext, state, B, ctx, seed, ns)
            allok &= ok
            print(f"[ksv] {'PASS' if ok else 'FAIL'} {msg}", flush=True)
    print(f"[ksv] ALL {'PASS' if allok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else ["test"])

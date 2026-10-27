"""Numerics check for a max-free ("unified max") softmax on muse_attention_decode inputs.

p_j = exp2(s'_j - m_row), s'_j = log2(e) * (5.473/8) * (q8 . k_j), m_row = log2(e) * 5.473 * ||q8|| (Cauchy-Schwarz bound
for unit-RMS keys, ||k|| = 8), P rounded to BF16 (FTZ below 2^-126), O and l accumulated in FP32, no running max, no
rescale. Compared with the problem's own reference and tolerance (check_implementation's bound).
Usage: python maxfree_sim.py <path to problems/muse_attention_decode>
"""
import math
import sys

import torch

sys.path.insert(0, sys.argv[1])
import reference as R  # noqa: E402
from checks import query_precision_bound, relative_l2  # noqa: E402

LOG2E = 1.4426950408889634


def maxfree(data, k_new, v_new):
    cfg = data["config"]
    B, L, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    g = hq // hkv
    q_in, _, _, gate = R.split_qkv(data)
    pos = data["seq_lens"].long()[:, None] + torch.arange(L, device=q_in.device)
    qn = R._rms(q_in)
    if cfg["rope_theta"]:
        qn = R._rope(qn, pos, cfg["rope_theta"])
    q8 = qn.to(torch.float8_e4m3fn).float()                     # E4M3 of the unit-RMS query (FA3's rounding)
    c = LOG2E * cfg["qk_scale_factor"] * cfg["sm_scale"]         # 0.987: exponent scale per q8.k
    m_row = LOG2E * cfg["qk_scale_factor"] * q8.norm(dim=-1)     # = c * ||q8|| * 8  [B, L, hq]
    kf = data["k_cache"].view(-1, hkv, 64)
    vf = data["v_cache"].view(-1, hkv, 64)
    out = torch.empty(B, L, hq, 64, device=q_in.device)
    stats = {"emax": -1e9, "rowmax_min": 1e9, "under": 0, "n": 0}
    for b in range(B):
        seq = int(data["seq_lens"][b])
        size = cfg["page_size"]
        past = torch.arange(seq, device=q_in.device)
        rows = data["page_table"][b].long()[past // size] * size + past % size
        keys = torch.cat([R._dequant(kf[rows], data["k_scale"]), R._dequant(k_new[b], data["k_scale"])])
        vals = torch.cat([R._dequant(vf[rows], data["v_scale"]), R._dequant(v_new[b], data["v_scale"])])
        for j in range(L):
            p_ = seq + j
            start = max(0, p_ - cfg["window"] + 1) if cfg["window"] else 0
            kb, vb = keys[start:p_ + 1], vals[start:p_ + 1]
            s = torch.einsum("kgd,tkd->kgt", q8[b, j].view(hkv, g, 64), kb) * c        # FP32 logits, log2 units
            e = s - m_row[b, j].view(hkv, g, 1)
            stats["emax"] = max(stats["emax"], float(e.max()))
            stats["rowmax_min"] = min(stats["rowmax_min"], float(e.amax(-1).min()))
            p = torch.exp2(e)
            p = torch.where(p < 2.0 ** -126, torch.zeros_like(p), p)                     # FTZ
            stats["under"] += int((p == 0).sum())
            stats["n"] += p.numel()
            pb = p.to(torch.bfloat16).float()                                           # BF16 P (PV operand)
            l = pb.sum(-1, keepdim=True)
            o = torch.einsum("kgt,tkd->kgd", pb, vb) / l
            out[b, j] = o.reshape(hq, 64)
    out = R._rms(out) * torch.sigmoid(gate.float()).view(B, L, hq, 64)
    return out.reshape(B, L, hq * 64).to(torch.bfloat16), stats


def main():
    cases = [
        dict(batch=2, ctx=80000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent", spread="agentic"),
        dict(batch=2, ctx=80000, heads="tp8", layer="full", q_len=1, page=1, dist="coherent", spread="agentic"),
        dict(batch=2, ctx=80000, heads="tp8", layer="full", q_len=8, page=1, dist="soft", spread="agentic"),
        dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1),
        dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges"),
        dict(batch=2, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625),
        dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="soft"),
        dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1),
    ]
    for i, cs in enumerate(cases):
        data = R.generate_input(seed=1234 + i, **cs)
        _, k_new, v_new = R.ref_kernel(data)
        ref = R.ref_kernel(data, new_kv=(k_new, v_new))[0].to(torch.bfloat16)
        e4 = relative_l2(R.ref_kernel(data, torch.float8_e4m3fn, new_kv=(k_new, v_new))[0].to(torch.bfloat16), ref)
        tol = query_precision_bound(e4, R.FP8_QUERY_FACTOR, R.FP8_QUERY_FLOOR, R.REL_L2_TOL, R.FP8_QUERY_SLACK)
        got, st = maxfree(data, k_new, v_new)
        err = relative_l2(got, ref)
        rows = (got.float() - ref.float()).norm(dim=-1) / ref.float().norm(dim=-1).clamp_min(1e-30)
        finite = bool(torch.isfinite(got.float()).all())
        print(f"case {i} {cs}: maxfree err {err:.4f} (worst row {float(rows.max()):.4f}) | fp8-q ref err {e4:.4f} "
              f"| bound {tol:.3f} | finite {finite} | max exponent {st['emax']:.1f} | lowest row max {st['rowmax_min']:.1f} "
              f"| underflow {st['under'] / st['n']:.2e}", flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()

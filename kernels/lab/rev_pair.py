"""REV pair probe (independent review, notes/study/independent_review_oct28.md s11; request R-rev-4): when one hot expert is
split into two jobs (VMS at T=16, VML at T=256), each job streams the expert's weights from HBM. If the two jobs' CTAs
stream the same stages at the same time, does the second read hit L2 (HBM bytes and stream energy once)?

Kernel: kernels/lab/rev_probe.py (vml_probe2 + shipped v13 consumer switches) plus one switch:
  PG_EVN  1: weight stages copied with a plain cp.async.bulk (no L2::evict_first hint), so a partner CTA can still hit
             the lines in L2 (the shipped producer marks every weight line evict_first)
Plans (host side, same kernel):
  std   vml_probe.plan: 924 units, 7 per CTA, every CTA streams distinct units (= today's single-job experts)
  pair  66 CTA pairs (c, c + G/2): both stream the same 7 units of an expert, CTA c for job A (tokens set A), CTA c + G/2
        for job B (tokens set B) -> 462 distinct units, each streamed twice, in lockstep (= a split hot expert, paired)
  lag   as pair, but the partner's sequence is rotated by 3 units (same units, ~3 units apart in time: no L2 reuse)
"""
import torch

import rev_probe as R

P2 = R.P2
V1 = P2.V1

_OLD = """      bulk_g2s_ef(dst, wb + (size_t)s * SW, SW, fb, pol);
      bulk_g2s_ef(dst + SW, wb + (size_t)(16 + s) * SW, SW, fb, pol);
#if PG_L2D"""
_NEW = """#if PG_EVN
      bulk_g2s(dst, wb + (size_t)s * SW, SW, fb);
      bulk_g2s(dst + SW, wb + (size_t)(16 + s) * SW, SW, fb);
#else
      bulk_g2s_ef(dst, wb + (size_t)s * SW, SW, fb, pol);
      bulk_g2s_ef(dst + SW, wb + (size_t)(16 + s) * SW, SW, fb, pol);
#endif
#if PG_L2D"""
assert R.HEAD.count(_OLD) == 1
HEAD = R.HEAD.replace(_OLD, _NEW).replace("#ifndef PG_L2D", "#ifndef PG_EVN\n#define PG_EVN 0\n#endif\n#ifndef PG_L2D", 1)
CUDA = HEAD + R.BODY
CPP = R.CPP
_S = {"PG_SC2": 1, "PG_DS": 1, "PG_CL": 1}
BUILDS = {"S": dict(_S), "SN": dict(_S, PG_EVN=1)}


def flags_of(b):
    return P2.V1.FLAGS + [f"-D{k}={v}" for k, v in BUILDS[b].items()]


def build(b, verbose=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    fl = flags_of(b)
    tag = hashlib.sha1((CPP + CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"rev_pair_{b}_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["vp2_run", "vp2_attrs"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


def plan_pair(ntok, G, dev, mode="pair", per=7, seed=7):
    """Two jobs per expert (token sets A / B); CTAs [0, G/2) run job A, [G/2, G) job B on the same unit sequences
    (mode 'lag': the B sequence rotated by 3 units). Returns the vml_probe.plan dict layout plus 'jobtoks'."""
    half = G // 2
    nd = half * per
    ne = -(-nd // 32)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((256, 4096), generator=g).to(torch.bfloat16).to(dev)
    npad = -(-ntok // 8) * 8
    jobs, jbase, xparts, jt, xo, row = [], [], [], [], 0, 0
    for e in range(ne):
        for _ in range(2):
            tk = torch.randperm(256, generator=g)[:ntok].tolist()
            xparts.append(V1.pack_x(x, tk, npad))
            jobs.append((e, ntok, npad, xo))
            jbase.append(row)
            jt.append(tk)
            xo += 16 * npad * 512
            row += ntok
    dist = [(u // 32, u % 32) for u in range(nd)]
    units = []
    for c in range(half):
        units += [(2 * e, k) for e, k in dist[c * per:(c + 1) * per]]
    for c in range(half):
        seq = dist[c * per:(c + 1) * per]
        if mode == "lag":
            seq = seq[3:] + seq[:3]
        units += [(2 * e + 1, k) for e, k in seq]
    cta = [(c * per, (c + 1) * per) for c in range(G)]
    t = lambda v: torch.tensor(v, dtype=torch.int32).to(dev).contiguous()  # noqa: E731
    return {"x": x, "jobtoks": jt, "job": t(jobs), "jbase": t(jbase), "unit": t(units), "cta": t(cta),
            "xp": torch.cat(xparts).contiguous(), "rows": row, "ne": 2 * ne, "units": G * per, "full_experts": nd // 32}


def ref_job(P, w, j, dequant):
    """FP32 reference of job j's SwiGLU rows (expert job[j][0], token set jobtoks[j])."""
    e = int(P["job"][j, 0])
    x = P["x"].float()[torch.as_tensor(P["jobtoks"][j], dtype=torch.long, device=P["x"].device)]
    W = dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e])
    h = x @ W.T
    return torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]

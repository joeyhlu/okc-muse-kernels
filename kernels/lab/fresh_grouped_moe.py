"""Independent NVFP4/BF16 grouped-GEMM experiment; no cross-CTA reductions.

Every CTA owns a complete output tile. A device route map and expert-major tile
order expose weight reuse to L2. This is a lab, not an accepted submission.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def route_map(Ids, Map, Reverse, Counts, T: tl.constexpr, E: tl.constexpr,
              R: tl.constexpr):
    e = tl.program_id(0)
    r = tl.arange(0, R)
    ids = tl.load(Ids + r, r < T * 8, other=-1)
    hit = (r < T * 8) & (ids == e)
    rank = tl.cumsum(hit.to(tl.int32)) - 1
    tl.store(Map + e * T + rank, r, hit)
    tl.store(Reverse + r, rank, hit)
    tl.store(Counts + e, tl.sum(hit.to(tl.int32)))


@tr.jit
def decode_fp4(code):
    a = code & 7
    # Exact E2M1 conversion, including signed zero.
    v = tl.where(a < 2, a.to(tl.float32) * 0.5,
                 tl.where(a < 4, (a - 2).to(tl.float32) * 0.5 + 1.,
                          tl.where(a < 6, (a - 4).to(tl.float32) + 2.,
                                   (a - 6).to(tl.float32) * 2. + 4.)))
    return tl.where((code & 8) != 0, -v, v)


@tr.jit
def decode_fp4_bits(code):
    code = code.to(tl.int32)
    a = code & 7
    bits = tl.where(a == 0, 0, ((a + 252) << 6) - tl.where(a == 1, 64, 0))
    bits |= (code & 8) << 12
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@tr.jit
def grouped(X, W, S, GS, Map, Counts, Out,
            T: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            GRID: tl.constexpr, GATE: tl.constexpr, BITDECODE: tl.constexpr = False):
    es = tl.arange(0, E)
    cs = tl.load(Counts + es)
    ms = tl.cdiv(cs, BM)
    ends = tl.cumsum(ms * (D // BN))
    total = tl.sum(ms) * (D // BN)
    tile = tl.program_id(0)
    while tile < total:
        e = tl.sum((tile >= ends).to(tl.int32))
        start = tl.sum(tl.where(es < e, ms * (D // BN), 0))
        nm = tl.sum(tl.where(es == e, ms, 0))
        count = tl.sum(tl.where(es == e, cs, 0))
        # M varies fastest: repeated weight tiles remain adjacent in the schedule.
        mi = (tile - start) % nm
        ni = (tile - start) // nm
        m = mi * BM + tl.arange(0, BM)
        if GATE:
            rr = tl.arange(0, 2 * BN)
            n = ni * BN + rr // 2 + (rr % 2) * D
            acc = tl.full((2 * BN, BM), 0., tl.float32)
        else:
            n = ni * BN + tl.arange(0, BN)
            acc = tl.full((BN, BM), 0., tl.float32)
        pair = tl.load(Map + e * T + m, m < count, other=0)
        gs = tl.load(GS + e)
        for ki in range(tl.cdiv(D, BK)):
            k = ki * BK + tl.arange(0, BK)
            rows: tl.constexpr = 2 * D if GATE else D
            p = tl.load(W + (e * rows + n[:, None]) * (D // 2) + k[None, :] // 2)
            code = (p >> ((k[None, :] % 2) * 4)) & 15
            sf = tl.load(S + (e * rows + n[:, None]) * (D // 16) + k[None, :] // 16).to(tl.float32)
            if BITDECODE:
                # E2M1 * E4M3 is exactly representable in BF16. Keep the FP32
                # tensor-global multiplier out of the weight-load inner loop.
                w = (decode_fp4_bits(code).to(tl.float32) * sf).to(tl.bfloat16)
            else:
                w = (decode_fp4(code) * sf * gs).to(tl.bfloat16)
            if GATE:
                x = tl.load(X + (pair[None, :] // 8) * D + k[:, None], m[None, :] < count, other=0)
            else:
                x = tl.load(X + (e * T + m[None, :]) * D + k[:, None], m[None, :] < count, other=0)
            acc += tl.dot(w, x)
        if BITDECODE:
            acc *= gs
        if GATE:
            # GEMM outputs round to BF16 before SwiGLU, matching stock precision.
            a = acc.to(tl.bfloat16).to(tl.float32)
            gate, up = tl.split(a.reshape((BN, 2, BM)).permute((0, 2, 1)))
            value = (gate / (1. + tl.exp(-gate)) * up).to(tl.bfloat16)
            outn = ni * BN + tl.arange(0, BN)
        else:
            value = acc.to(tl.bfloat16)
            outn = n
        tl.store(Out + (e * T + m[None, :]) * D + outn[:, None], value, m[None, :] < count)
        tile += GRID


@tr.jit
def normalize(Y, NW, Counts, Z, T: tl.constexpr, D: tl.constexpr):
    e = tl.program_id(0)
    m = tl.program_id(1)
    count = tl.load(Counts + e)
    if m < count:
        n = tl.arange(0, D)
        y = tl.load(Y + (e * T + m) * D + n).to(tl.float32)
        nw = tl.load(NW + n).to(tl.float32)
        y = y * (1. + nw)
        y = y * tl.rsqrt(tl.sum(y * y) / D + 1.e-8)
        tl.store(Z + (e * T + m) * D + n, y)


@tr.jit
def combine(Z, Ids, Weights, Reverse, Out, T: tl.constexpr, D: tl.constexpr,
            E: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v = tl.full((BLOCK,), 0., tl.float32)
    for s in tl.static_range(8):
        e = tl.load(Ids + t * 8 + s)
        valid = (e >= 0) & (e < E)
        rank = tl.load(Reverse + t * 8 + s, valid, other=0)
        wt = tl.load(Weights + t * 8 + s)
        y = tl.load(Z + (e * T + rank) * D + n, valid & (n < D), other=0)
        v += wt * y
    tl.store(Out + t * D + n, v.to(tl.bfloat16), n < D)


def workspace(weights, tokens):
    dev = weights['w13'].device
    e, _, k2 = weights['w13'].shape
    d = k2 * 2
    return dict(map=torch.empty((e, tokens), device=dev, dtype=torch.int32),
                rev=torch.empty((tokens, 8), device=dev, dtype=torch.int32),
                counts=torch.empty(e, device=dev, dtype=torch.int32),
                inter=torch.empty((e, tokens, d), device=dev, dtype=torch.bfloat16),
                y=torch.empty((e, tokens, d), device=dev, dtype=torch.bfloat16),
                z=torch.empty((e, tokens, d), device=dev, dtype=torch.float32),
                out=torch.empty((tokens, d), device=dev, dtype=torch.bfloat16))


def run(data, weights, st, config=(16, 32, 64, 4, 2), bitdecode=False):
    bm, bn, bk, warps, stages = config
    x, ids, wt = data['hidden_states'], data['topk_ids'], data['topk_weights']
    t, d = x.shape
    e = weights['w13'].shape[0]
    grid = 264
    route_map[(e,)](ids, st['map'], st['rev'], st['counts'], t, e, tr.next_power_of_2(t * 8))
    grouped[(grid,)](x, weights['w13'], weights['w13_sf'], weights['w13_gs'],
                      st['map'], st['counts'], st['inter'], t, d, e, bm, bn, bk, grid, True, bitdecode,
                      num_warps=warps, num_stages=stages)
    grouped[(grid,)](st['inter'], weights['w2'], weights['w2_sf'], weights['w2_gs'],
                      st['map'], st['counts'], st['y'], t, d, e, bm, bn, bk, grid, False, bitdecode,
                      num_warps=warps, num_stages=stages)
    normalize[(e, t)](st['y'], weights['post_norm_w'], st['counts'], st['z'], t, d)
    combine[(t, tr.cdiv(d, 256))](st['z'], ids, wt, st['rev'], st['out'], t, d, e, 256)
    return st['out']

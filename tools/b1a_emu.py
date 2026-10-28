"""b1a_emu.py: CPU (numpy) lane-level emulation of the B1A kernel (tools/b1a_patches.py), for checks without a GPU.

The kernel's per-lane code is transliterated (key ranges, swizzled cp.async rows, Q / new K / V fragments, the unit loop, the
absolute-frame merge, the finalize) on top of primitives written from the PTX ISA definitions, independently of the kernel:
ldmatrix .x4 (and .trans) on b16, mma.sync m16n8k32 e4m3 fragment layouts (A rows g / g + 8, k 4t..4t+3 / +16; B k 4t..4t+3 /
+16, n g; C rows g / g + 8, cols 2t, 2t + 1), prmt byte selection, e4m3 RN satfinite. Registers holding 4 e4m3 bytes are
modelled as 4 floats (byte e = element e), so --exact can keep P unrounded: then the emulation must equal a float64
reference to ~1e-6, which proves the index maths (swizzles, the P key permutation, the V prmt, the output dim order, the
causal mask, ranges and copies). Without --exact P is e4m3 as on the GPU (expected error ~1e-2, as VFA / FA4).

    python -I tools/b1a_emu.py [--exact] [--quick]
"""
import sys

import numpy as np

HQ, HKV, D, ROWB, TK, NU = 16, 2, 64, 128, 64, 32
LOG2E = 1.4426950408889634
POFF = 8.0


def _e4m3_values():
    v = np.zeros(256)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 15, c & 7
        v[c] = np.nan if (e == 15 and m == 7) else (s * m / 8 * 2.0 ** -6 if e == 0 else s * (1 + m / 8) * 2.0 ** (e - 7))
    return v


E4 = _e4m3_values()
POS = E4[:127]


def e4m3_code(x):
    """RN-even, satfinite (cvt.rn.satfinite.e4m3x2.f32)."""
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), 448.0)
    i = np.clip(np.searchsorted(POS, a), 1, 126)
    lo, hi = POS[i - 1], POS[i]
    up = (a - lo > hi - a) | ((a - lo == hi - a) & (i % 2 == 0))
    return (np.where(up, i, i - 1) | (np.signbit(x).astype(np.int64) << 7)).astype(np.uint8)


def e4m3(x):
    return E4[e4m3_code(x)]


def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


EXACT = False   # --exact: no rounding of Q, P or the output path on either side (index maths only)


def rnd_bf16(x):
    return x if EXACT else bf16(x)


def rnd_q(x):
    return x if EXACT else e4m3(x)


LANE = np.arange(32)
G_, T_ = LANE >> 2, LANE & 3


def ldsm_x4(smem, addr):
    """addr[32]: lane 8m + i gives row i of matrix m. Returns r[4][32, 4] (decoded bytes 4t .. 4t + 3 of row g)."""
    out = []
    for mtx in range(4):
        rows = addr[8 * mtx + G_]
        out.append(E4[smem[rows[:, None] + 4 * T_[:, None] + np.arange(4)[None, :]]])
    return out


def ldsm_x4_t(smem, addr):
    """.trans: lane (g, t) gets b16 elements (row 2t, col g), (row 2t + 1, col g): bytes 2g, 2g + 1 of rows 2t, 2t + 1."""
    out = []
    for mtx in range(4):
        r0, r1 = addr[8 * mtx + 2 * T_], addr[8 * mtx + 2 * T_ + 1]
        b = np.stack([smem[r0 + 2 * G_], smem[r0 + 2 * G_ + 1], smem[r1 + 2 * G_], smem[r1 + 2 * G_ + 1]], 1)
        out.append(E4[b])
    return out


def prmt(a, b, sel):
    src = np.concatenate([a, b], 1)
    return src[:, [(sel >> (4 * i)) & 7 for i in range(4)]]


def mma8(d, a, b0, b1):
    A = np.zeros((16, 32))
    B = np.zeros((32, 8))
    for ln in range(32):
        g, t = ln >> 2, ln & 3
        A[g, 4 * t:4 * t + 4] = a[0][ln]
        A[g + 8, 4 * t:4 * t + 4] = a[1][ln]
        A[g, 16 + 4 * t:20 + 4 * t] = a[2][ln]
        A[g + 8, 16 + 4 * t:20 + 4 * t] = a[3][ln]
        B[4 * t:4 * t + 4, g] = b0[ln]
        B[16 + 4 * t:20 + 4 * t, g] = b1[ln]
    C = A @ B
    d[:, 0] += C[G_, 2 * T_]
    d[:, 1] += C[G_, 2 * T_ + 1]
    d[:, 2] += C[G_ + 8, 2 * T_]
    d[:, 3] += C[G_ + 8, 2 * T_ + 1]


def quad_max(v):
    return np.repeat(v.reshape(8, 4).max(1), 4)


class Emu:
    def __init__(self, exact):
        self.exact = exact

    def pk4(self, a, b, c, d):
        x = np.stack([a, b, c, d], 1)
        return x if self.exact else e4m3(x)

    def unit(self, smem, kb, vb, nb_, new, r2, cnt, h, tok0, tok1, qa, o, la, m, cl2):
        s = np.zeros((nb_, 32, 4))
        sw = LANE & 7
        for nb in range(nb_):
            r = ldsm_x4(smem, kb + (8 * nb + (LANE & 7)) * ROWB + (((4 * h + (LANE >> 3)) ^ sw) << 4))
            mma8(s[nb], qa[0], r[0], r[1])
            mma8(s[nb], qa[1], r[2], r[3])
        if new or cnt < 8 * nb_:
            for nb in range(nb_):
                for e in range(2):
                    key = 8 * nb + 2 * T_ + e
                    s[nb][~((key < cnt) & ((not new) | (key <= tok0))), e] = -np.inf
                    s[nb][~((key < cnt) & ((not new) | (key <= tok1))), 2 + e] = -np.inf
        x0 = np.full(32, -np.inf)
        x1 = np.full(32, -np.inf)
        for nb in range(nb_):
            x0 = np.maximum(x0, np.maximum(s[nb][:, 0], s[nb][:, 1]))
            if r2:
                x1 = np.maximum(x1, np.maximum(s[nb][:, 2], s[nb][:, 3]))
        x0 = quad_max(x0) * cl2
        if r2:
            x1 = quad_max(x1) * cl2
        n0 = np.maximum(m[0], x0)
        n1 = np.maximum(m[1], x1) if r2 else m[1].copy()
        if np.any((n0 > m[0]) | (n1 > m[1])):
            with np.errstate(invalid="ignore"):
                a0 = np.where(n0 > m[0], np.exp2(m[0] - n0), 1.0)
                a1 = np.where(n1 > m[1], np.exp2(m[1] - n1), 1.0)
            o[:, :, 0] *= a0
            o[:, :, 1] *= a0
            o[:, :, 2] *= a1
            o[:, :, 3] *= a1
            la[:, 0] *= a0
            la[:, 1] *= a0
            la[:, 2] *= a1
            la[:, 3] *= a1
            m[0], m[1] = n0, n1
        b0 = np.where(m[0] == -np.inf, 0.0, m[0] - POFF)
        b1 = np.where(m[1] == -np.inf, 0.0, m[1] - POFF)

        def ex(v, b):
            return np.exp2(v * cl2 - b)
        for kq in range(nb_ // 4):
            q = 4 * kq
            pa = [self.pk4(ex(s[q][:, 0], b0), ex(s[q][:, 1], b0), ex(s[q + 1][:, 0], b0), ex(s[q + 1][:, 1], b0)), None,
                  self.pk4(ex(s[q + 2][:, 0], b0), ex(s[q + 2][:, 1], b0), ex(s[q + 3][:, 0], b0), ex(s[q + 3][:, 1], b0)), None]
            if r2:
                pa[1] = self.pk4(ex(s[q][:, 2], b1), ex(s[q][:, 3], b1), ex(s[q + 1][:, 2], b1), ex(s[q + 1][:, 3], b1))
                pa[3] = self.pk4(ex(s[q + 2][:, 2], b1), ex(s[q + 2][:, 3], b1), ex(s[q + 3][:, 2], b1), ex(s[q + 3][:, 3], b1))
            else:
                pa[1] = np.zeros((32, 4))
                pa[3] = np.zeros((32, 4))
            for cc in range(4):
                r = ldsm_x4_t(smem, vb + (32 * kq + LANE) * ROWB + (((4 * h + cc) ^ sw) << 4))
                mma8(o[2 * cc], pa, prmt(r[0], r[1], 0x6420), prmt(r[2], r[3], 0x6420))
                mma8(o[2 * cc + 1], pa, prmt(r[0], r[1], 0x7531), prmt(r[2], r[3], 0x7531))
            mma8(la, pa, np.ones((32, 4)), np.ones((32, 4)))


def key_range(n, newv, c, G):
    T = n + newv
    v0, v1 = (T * c) // G, (T * (c + 1)) // G
    a = min(v0, n)
    return a, max(min(v1, n) - a, 0)


def run(case, G, NT, W, exact, spec=True):
    """case: dict(qkv [L, 2304] bf16-valued, kc8 / vc8 [rows, 128] uint8 codes, pt [width], seqlen, cache_loc [L], page,
    ks, vs, qk_sm, eps). Returns (out [L, 1024], kc8, vc8 after the writes)."""
    em = Emu(exact)
    L, page = case["L"], case["page"]
    lp = page.bit_length() - 1
    qkv, pt = case["qkv"], case["pt"]
    kc8, vc8 = case["kc8"].copy(), case["vc8"].copy()
    nrows = kc8.shape[0]
    ROWS = 16 * L
    MB = 4 if L == 8 else 1
    KS = 4 // MB
    R2 = L == 8
    NFIN = ROWS // 8
    ws_o = np.zeros((W, ROWS, D))
    ws_l = np.zeros((W, ROWS))
    ks, vs, eps = case["ks"], case["vs"], case["eps"]
    cl2 = case["qk_sm"] * ks * LOG2E
    nmax = max(len(pt) * page - L, 0)
    n_true = min(max(case["seqlen"] - L, 0), nmax)
    n = nmax if spec else n_true
    misses = 0
    smem_bytes = 2 * NT * TK * ROWB + 2 * NU * ROWB
    OFF_V, OFF_NK = NT * TK * ROWB, 2 * NT * TK * ROWB
    OFF_NV = OFF_NK + NU * ROWB
    # new rows (written by the last CTA after the wait; no CTA reads cache rows >= n, so order vs the copies is free here)
    newk = np.zeros((NU, ROWB), np.uint8)
    newv = np.zeros((NU, ROWB), np.uint8)
    for job in range(4 * L):
        kind, hh, jj = job & 1, (job >> 1) & 1, job >> 2
        x = qkv[jj, (HQ + kind * HKV + hh) * D:(HQ + kind * HKV + hh + 1) * D].astype(np.float64)
        if kind == 0:
            rs = 1.0 / np.sqrt(np.sum(x * x) / 64 + eps)
            x = bf16(x * rs)
        e8 = e4m3_code(x / (vs if kind else ks))
        loc = case["cache_loc"][jj]
        (vc8 if kind else kc8)[loc, hh * D:(hh + 1) * D] = e8
        (newv if kind else newk)[jj, hh * D:(hh + 1) * D] = e8
    for c in range(G):
        tail = c == G - 1
        a, kc = key_range(n, 32, c, G)
        if n_true != n:
            misses += 1
            a, kc = key_range(n_true, 32, c, G)
        ntl = (kc + TK - 1) // TK
        assert ntl <= NT, "keys per CTA beyond NT x 64"
        smem = np.full(smem_bytes, 0xFF, np.uint8)   # garbage (NaN bytes) where nothing is written
        for tid in range(256):
            ch, r0 = tid & 7, tid >> 3
            for i in range(ntl):
                for k in range(2):
                    jj = r0 + 32 * k
                    j = TK * i + jj
                    pg = pt[(a + j) >> lp] if j < kc else -1
                    row = (pg << lp) + ((a + TK * i + jj) & (page - 1))
                    ok = pg >= 0 and row < nrows
                    dst = i * TK * ROWB + jj * ROWB + ((ch ^ (jj & 7)) << 4)
                    smem[dst:dst + 16] = kc8[row, ch * 16:ch * 16 + 16] if ok else 0
                    smem[OFF_V + dst:OFF_V + dst + 16] = vc8[row, ch * 16:ch * 16 + 16] if ok else 0
        if tail:   # zero-filled unit, then the new rows with the 128B swizzle
            smem[OFF_NK:OFF_NV + NU * ROWB] = 0
            for jj in range(L):
                for bb in range(0, ROWB, 2):
                    d = jj * ROWB + (((bb >> 4) ^ (jj & 7)) << 4) + (bb & 15)
                    smem[OFF_NK + d:OFF_NK + d + 2] = newk[jj, bb:bb + 2]
                    smem[OFF_NV + d:OFF_NV + d + 2] = newv[jj, bb:bb + 2]
        os_ = np.zeros((4, 16, D))
        ls_ = np.zeros((4, 16))
        for warp in range(8):
            h, wq = warp >> 2, warp & 3
            mb, ksw = wq % MB, wq // MB
            tok0, tok1 = (2 * mb, 2 * mb + 1) if L == 8 else (0, 0)
            xr = []
            for rr in range(2 if R2 else 1):
                tok = tok1 if rr else tok0
                x = np.zeros((32, 16))
                for cc in range(4):
                    for e in range(4):
                        x[:, 4 * cc + e] = qkv[tok, (h * 8 + G_) * D + 16 * cc + 4 * T_ + e]
                ss = np.repeat((x * x).sum(1).reshape(8, 4).sum(1), 4)
                rs = 1.0 / np.sqrt(ss / 64 + eps)
                xr.append(rnd_bf16(x * rs[:, None]))
            qa = []
            for s2 in range(2):
                r0_ = em_q(xr[0][:, 8 * s2:8 * s2 + 4])
                r2_ = em_q(xr[0][:, 8 * s2 + 4:8 * s2 + 8])
                r1_ = em_q(xr[1][:, 8 * s2:8 * s2 + 4]) if R2 else np.zeros((32, 4))
                r3_ = em_q(xr[1][:, 8 * s2 + 4:8 * s2 + 8]) if R2 else np.zeros((32, 4))
                qa.append([r0_, r1_, r2_, r3_])
            o = np.zeros((8, 32, 4))
            la = np.zeros((32, 4))
            m = [np.full(32, -np.inf), np.full(32, -np.inf)]
            if L == 8:
                for i in range(ntl):
                    em.unit(smem, i * TK * ROWB, OFF_V + i * TK * ROWB, 4, False, R2, min(32, kc - TK * i), h, tok0, tok1, qa, o, la, m, cl2)
                    if kc - TK * i > 32:
                        em.unit(smem, i * TK * ROWB + 32 * ROWB, OFF_V + i * TK * ROWB + 32 * ROWB, 4, False, R2, min(32, kc - TK * i - 32),
                                h, tok0, tok1, qa, o, la, m, cl2)
                if tail:
                    em.unit(smem, OFF_NK, OFF_NV, 4, True, R2, L, h, tok0, tok1, qa, o, la, m, cl2)
            else:
                nu = (kc + 31) >> 5
                for u in range(ksw, nu, KS):
                    i, off = u >> 1, (u & 1) * 32
                    em.unit(smem, i * TK * ROWB + off * ROWB, OFF_V + i * TK * ROWB + off * ROWB, 4, False, R2,
                            min(32, kc - TK * i - off), h, tok0, tok1, qa, o, la, m, cl2)
                if tail and ksw == nu % KS:
                    em.unit(smem, OFF_NK, OFF_NV, 4, True, R2, L, h, tok0, tok1, qa, o, la, m, cl2)
            f0 = np.where(m[0] == -np.inf, 0.0, np.exp2(np.minimum(m[0] - POFF, 126.0)))
            f1 = np.where(m[1] == -np.inf, 0.0, np.exp2(np.minimum(m[1] - POFF, 126.0)))
            cp = c % W
            anyk = kc > 0 or tail
            for ln in range(32):
                g, t = ln >> 2, ln & 3
                for cc in range(4):
                    v0 = np.array([o[2 * cc][ln, 0], o[2 * cc + 1][ln, 0], o[2 * cc][ln, 1], o[2 * cc + 1][ln, 1]]) * f0[ln]
                    v1 = np.array([o[2 * cc][ln, 2], o[2 * cc + 1][ln, 2], o[2 * cc][ln, 3], o[2 * cc + 1][ln, 3]]) * f1[ln]
                    if L == 8:
                        if anyk:
                            ws_o[cp, tok0 * 16 + 8 * h + g, 16 * cc + 4 * t:16 * cc + 4 * t + 4] += v0
                            ws_o[cp, tok1 * 16 + 8 * h + g, 16 * cc + 4 * t:16 * cc + 4 * t + 4] += v1
                    else:
                        os_[ksw, 8 * h + g, 16 * cc + 4 * t:16 * cc + 4 * t + 4] = v0
                if t == 0:
                    if L == 8:
                        if anyk:
                            ws_l[cp, tok0 * 16 + 8 * h + g] += la[ln, 0] * f0[ln]
                            ws_l[cp, tok1 * 16 + 8 * h + g] += la[ln, 2] * f1[ln]
                    else:
                        ls_[ksw, 8 * h + g] = la[ln, 0] * f0[ln]
        if L == 1 and (kc > 0 or tail):
            cp = c % W
            ws_o[cp, :16] += os_.sum(0)
            ws_l[cp, :16] += ls_.sum(0)
    out = np.zeros((L, HQ * D))
    for r in range(NFIN * 8):
        acc = ws_o[:, r].sum(0)
        ls = ws_l[:, r].sum()
        inv = vs / ls if ls > 0 else 0.0
        y = rnd_bf16(acc * inv)
        rs = 1.0 / np.sqrt(np.sum(y * y) / 64 + eps)
        tok, qh = r >> 4, r & 15
        gate = qkv[tok, (HQ + 2 * HKV + qh) * D:(HQ + 2 * HKV + qh + 1) * D]
        out[tok, qh * D:(qh + 1) * D] = rnd_bf16(y * rs / (1 + np.exp(-gate)))
    return out, kc8, vc8, misses


def em_q(x4):
    return rnd_q(x4)


def reference(case, kc8, vc8):
    """float64: e4m3 queries (bf16-rounded unit-RMS q), the cache after the writes, exact softmax, bf16 attention output."""
    L, page = case["L"], case["page"]
    qkv, pt, eps = case["qkv"], case["pt"], case["eps"]
    n = case["seqlen"] - L
    pos = np.arange(n + L)
    rows = pt[pos // page] * page + pos % page
    K = E4[kc8[rows]].reshape(-1, HKV, D) * case["ks"]
    V = E4[vc8[rows]].reshape(-1, HKV, D) * case["vs"]
    out = np.zeros((L, HQ * D))
    for j in range(L):
        for qh in range(HQ):
            q = qkv[j, qh * D:(qh + 1) * D].astype(np.float64)
            q = rnd_q(rnd_bf16(q / np.sqrt(np.sum(q * q) / 64 + eps)))
            hh = qh // 8
            s = K[:n + j + 1, hh] @ q * case["qk_sm"]
            p = np.exp(s - s.max())
            y = rnd_bf16((p @ V[:n + j + 1, hh]) / p.sum())
            rs = 1.0 / np.sqrt(np.sum(y * y) / 64 + eps)
            gate = qkv[j, (HQ + 2 * HKV + qh) * D:(HQ + 2 * HKV + qh + 1) * D]
            out[j, qh * D:(qh + 1) * D] = y * rs / (1 + np.exp(-gate))
    return out


def make_case(ctx, L, page, kvs, extra, seed):
    rng = np.random.default_rng(seed)
    need = (ctx + L + page - 1) // page
    width = need + extra
    total = width + 5
    perm = rng.permutation(total)
    pt = np.full(width, -1, np.int64)
    pt[:need] = perm[:need]
    u = rng.standard_normal((HKV, D))
    kf = 0.7 * u + 0.7 * rng.standard_normal((total * page, HKV, D))
    vf = rng.standard_normal((total * page, HKV, D))
    kc8 = e4m3_code(kf / kvs).reshape(total * page, ROWB)
    vc8 = e4m3_code(vf / kvs).reshape(total * page, ROWB)
    qkv = bf16(np.concatenate([2 * rng.standard_normal((L, HQ * D)), 2 * rng.standard_normal((L, HKV * D)),
                               rng.standard_normal((L, HKV * D)), rng.standard_normal((L, HQ * D))], 1))
    pos = ctx + np.arange(L)
    loc = pt[pos // page] * page + pos % page
    return dict(L=L, page=page, qkv=qkv, pt=pt, kc8=kc8, vc8=vc8, seqlen=ctx + L, cache_loc=loc, ks=kvs, vs=kvs,
                qk_sm=0.125 * 5.4730064863875, eps=1e-5)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main():
    global EXACT
    exact = EXACT = "--exact" in sys.argv
    quick = "--quick" in sys.argv
    cases = [(1500, 1, 1, 1.0, 0, 1, 6), (1500, 8, 1, 1.0, 0, 2, 16), (70, 8, 1, 1.0, 0, 3, 16), (3, 1, 1, 1.0, 0, 4, 6),
             (0, 8, 1, 1.0, 0, 5, 16), (900, 8, 1, 1.0, 9, 6, 16), (900, 1, 1, 1.0, 9, 7, 6), (1200, 8, 64, 1.0, 0, 8, 16),
             (1200, 1, 64, 1.0, 0, 9, 6), (700, 8, 1, 1.0 / 64, 0, 10, 16), (700, 1, 1, 1.0 / 64, 0, 11, 6),
             (2500, 8, 1, 1.0, 0, 12, 16), (2500, 1, 1, 1.0, 0, 13, 8)]
    if quick:
        cases = cases[:3]
    worst = 0.0
    for ctx, L, page, kvs, extra, seed, G in cases:
        cs = make_case(ctx, L, page, kvs, extra, seed)
        out, kc8, vc8, miss = run(cs, G, 5, 8 if G >= 8 else 3, exact)
        ref = reference(cs, kc8, vc8)
        e = rel(out, ref)
        worst = max(worst, e)
        rk = rel(E4[kc8[cs["cache_loc"]]].ravel(), E4[e4m3_code(_ref_k(cs))].ravel())
        print(f"ctx {ctx:5d} L {L} page {page:2d} kvs {kvs:g} width+{extra} G {G}: rel {e:.2e} (spec misses {miss}) new k {rk:.1e}",
              flush=True)
    lim = 2e-6 if exact else 3e-2
    print(f"worst rel {worst:.2e} -> {'PASS' if worst < lim else 'FAIL'} (limit {lim:g}, {'exact P' if exact else 'e4m3 P'})")
    return 0 if worst < lim else 1


def _ref_k(cs):
    L = cs["L"]
    k = cs["qkv"][:, HQ * D:(HQ + HKV) * D].reshape(L, HKV, D)
    k = bf16(k / np.sqrt((k * k).sum(-1, keepdims=True) / 64 + cs["eps"]))
    return (k / cs["ks"]).reshape(L, -1)


if __name__ == "__main__":
    sys.exit(main())

"""Literal CPU port of ps_kernel's index math (partition, segments, ldmatrix addressing, C fragments, flush, cluster
reduction) on decoded float weights (the P12 decode itself is P1's verified dec_kt). Checks every output vs x @ W.T."""
import numpy as np

NW, CL = 8, 8


def ldsm_frag(xs, xs_w, M_rows, lane_addr_fn, nmat):
    """Emulate ldmatrix (non-trans): lane L' = 8 mi + r supplies the row address of row r of matrix mi.
    Returns frag[lane][mi] = (elem0, elem1) for row lane//4, cols 2(lane%4), +1 of each matrix."""
    out = np.zeros((32, nmat, 2))
    for mi in range(nmat):
        rows = []
        for r in range(8):
            addr = lane_addr_fn(8 * mi + r)          # byte address within xs
            tok, colb = divmod(addr, xs_w * 4)
            rows.append((tok, colb // 2))            # (token row, first bf16 column)
        for L in range(32):
            tok, c0 = rows[L // 4]
            out[L, mi, 0] = xs[tok, c0 + 2 * (L % 4)]
            out[L, mi, 1] = xs[tok, c0 + 2 * (L % 4) + 1]
    return out


def run(W, x, R, Q, NG, pair, nclu=16):
    M, K = x.shape
    TP = NG * 8
    out_tiles = nclu * (R // 2 if pair else R)
    nhalf = out_tiles if pair else 0
    out = np.full((M, 16 * out_tiles), np.nan)
    xs_w = Q * 32 + 4

    def tile_of(cid, tl):
        if not pair:
            return cid * R + tl
        h = R >> 1
        return cid * h + tl if tl < h else nhalf + cid * h + (tl - h)

    for cid in range(nclu):
        parts = []
        for crank in range(CL):
            # x smem: TP rows x (xs_w*2) bf16 columns; rows >= M zero; columns [0, 64Q) = x[t, crank*Q*64 + c]
            xs = np.zeros((TP, xs_w * 2))
            xs[:M, :Q * 64] = x[:, crank * Q * 64:(crank + 1) * Q * 64]
            part = np.zeros((R * 16, TP))
            RQ = R * Q
            for warp in range(NW):
                s_lo, s_hi = warp * RQ // NW, (warp + 1) * RQ // NW
                acc = np.zeros((32, NG, 4))
                seg_tl = -1
                for s in range(s_lo, s_hi):
                    tl, sbq = divmod(s, Q)
                    if tl != seg_tl:
                        if seg_tl >= 0:
                            flush(part, seg_tl, acc, TP, NG)
                        seg_tl, seg_tile = tl, tile_of(cid, tl)
                    sb = crank * Q + sbq
                    for kt in range(4):
                        # A fragments (P12 lane map): a[L][i][h] = W[row, k]
                        A = np.zeros((32, 4, 2))
                        for L in range(32):
                            g, t = L >> 2, L & 3
                            for i in range(4):
                                for h in range(2):
                                    row = seg_tile * 16 + g + 8 * (i & 1)
                                    k = 64 * sb + 16 * kt + 2 * t + 8 * (i >> 1) + h
                                    A[L, i, h] = W[row, k]
                        kofs = (sbq * 64 + 16 * kt) * 2

                        def lrow(Lp):
                            return (((Lp & 7) + 8 * ((Lp >> 4) & 1)) * xs_w) * 4 + ((Lp >> 3) & 1) * 16
                        B = np.zeros((32, NG, 2, 2))   # [lane][group][b0/b1][elem]
                        if NG == 1:
                            fr = ldsm_frag(xs, xs_w, TP, lambda Lp: lrow(Lp) + kofs, 2)
                            B[:, 0, 0], B[:, 0, 1] = fr[:, 0], fr[:, 1]
                        else:
                            for gp in range(NG // 2):
                                fr = ldsm_frag(xs, xs_w, TP, lambda Lp: lrow(Lp) + gp * 16 * xs_w * 4 + kofs, 4)
                                B[:, 2 * gp, 0], B[:, 2 * gp, 1] = fr[:, 0], fr[:, 1]
                                B[:, 2 * gp + 1, 0], B[:, 2 * gp + 1, 1] = fr[:, 2], fr[:, 3]
                        mma(acc, A, B, NG)
                if seg_tl >= 0:
                    flush(part, seg_tl, acc, TP, NG)
            parts.append(part)
        orows = 8 * R if pair else 16 * R
        rows_per = orows // CL
        for crank in range(CL):
            for i in range(rows_per * M):
                rr = crank * rows_per + i % rows_per
                tok = i // rows_per
                v = 0.0
                for q in range(CL):
                    rp = parts[q]
                    if pair:
                        tl0, r = rr >> 4, rr & 15
                        v += rp[tl0 * 16 + r, tok] + rp[(tl0 + (R >> 1)) * 16 + r, tok]
                    else:
                        v += rp[rr, tok]
                assert np.isnan(out[tok, cid * orows + rr]), "double write"
                out[tok, cid * orows + rr] = v
    return out


def mma(acc, A, B, NG):
    """m16n8k16 per lane: gather the warp's A (16x16) and B (16x8) from fragments, d += A @ B, scatter to C frags."""
    Am = np.zeros((16, 16))
    for L in range(32):
        g, t = L >> 2, L & 3
        for i in range(4):
            for h in range(2):
                Am[g + 8 * (i & 1), 2 * t + 8 * (i >> 1) + h] = A[L, i, h]
    for gi in range(NG):
        Bm = np.zeros((16, 8))
        for L in range(32):
            g, t = L >> 2, L & 3
            Bm[2 * t, g], Bm[2 * t + 1, g] = B[L, gi, 0]
            Bm[2 * t + 8, g], Bm[2 * t + 9, g] = B[L, gi, 1]
        Cm = Am @ Bm
        for L in range(32):
            g, t = L >> 2, L & 3
            acc[L, gi] += [Cm[g, 2 * t], Cm[g, 2 * t + 1], Cm[g + 8, 2 * t], Cm[g + 8, 2 * t + 1]]


def flush(part, tl, acc, TP, NG):
    for L in range(32):
        g, t = L >> 2, L & 3
        for gi in range(NG):
            part[tl * 16 + g, gi * 8 + 2 * t] += acc[L, gi, 0]
            part[tl * 16 + g, gi * 8 + 2 * t + 1] += acc[L, gi, 1]
            part[tl * 16 + g + 8, gi * 8 + 2 * t] += acc[L, gi, 2]
            part[tl * 16 + g + 8, gi * 8 + 2 * t + 1] += acc[L, gi, 3]
    acc[:] = 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for (R, Q, M, NG, pair, nclu) in [(1, 2, 5, 1, False, 4), (2, 1, 20, 4, False, 2), (2, 2, 12, 2, True, 2),
                                      (3, 2, 8, 1, False, 2), (1, 3, 32, 4, False, 2)]:
        K = CL * Q * 64
        n_rows = 16 * nclu * R
        W = rng.standard_normal((n_rows, K))
        x = rng.standard_normal((M, K))
        out = run(W, x, R, Q, NG, pair, nclu)
        if pair:
            h = n_rows // 2
            ref = x @ (W[:h] + W[h:]).T
        else:
            ref = x @ W.T
        err = np.nanmax(np.abs(out - ref)) if not np.isnan(out).any() else float("nan")
        print(f"R{R} Q{Q} M{M} NG{NG} pair{int(pair)} nclu{nclu}: max err {err:.2e}  {'OK' if err < 1e-9 else 'BAD'}")

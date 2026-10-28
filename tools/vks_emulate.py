"""CPU emulation of kernels/lab/vks_probe.py's index math (numpy only; no GPU, no torch).

Mirrors the kernel literally: fill<K> slot byte layout from VMS W13 units, consume<K> code / scale / x addressing, the
m16n8k16 fragment semantics (A rows g / g+8 and k-cols from the word bit positions POS, B from the xr registers), the
per-chain accumulators, reduce_out's K-slice sum and the fragment-native output tile mapping. Compares every output
element with a dense W x^T reference built from the same random codes / scales (decoded values exact; fp64 sums).

  python3 -I tools/vks_emulate.py            -> max rel error per kind (expect ~1e-15)
"""
from __future__ import annotations

import numpy as np

UB, UC = 36864, 32768
POS = (((6, 7, 8, 15), (22, 23, 24, 31)),
       ((2, 3, 4, 11), (18, 19, 20, 27)),
       ((28, 29, 30, 5), (12, 13, 14, 21)),
       ((0, 1, 26, 9), (16, 17, 10, 25)))
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e4m3(b):
    b = np.asarray(b, dtype=np.int64)
    e, m = (b >> 3) & 15, b & 7
    return np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7)) * np.where(b & 128, -1.0, 1.0)


def fp4(c):
    c = np.asarray(c, dtype=np.int64)
    return GRID[c & 7] * np.where(c & 8, -1.0, 1.0)


# ---- VMS prepack (numpy port of vms_moe_lab._pack_unit_words / _pack_unit_scales / prepack_expert, gate_up=True)
def pack_words(cu):
    nu = cu.shape[0]
    x = cu.reshape(nu, 2, 8, 8, 8, 4, 4, 2, 2).transpose(0, 3, 4, 2, 5, 6, 7, 1, 8).reshape(nu, 8, 8, 8, 4, 4, 4, 2)
    x = x.astype(np.int64)
    word = np.zeros(x.shape[:6], dtype=np.int64)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << POS[r][h][b]
    return word.astype(np.uint32).reshape(nu, 8192)


def pack_scales(su):
    nu = su.shape[0]
    return su.reshape(nu, 2, 8, 8, 8, 4).transpose(0, 3, 2, 5, 4, 1).reshape(nu, 4096)


def prepack_expert(codes, sfb):
    n = codes.shape[0]
    rows = np.arange(n)
    rb, i = rows // 16, rows % 16
    rows = np.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    cu = codes[rows].reshape(n // 16, 16, 4096)
    sb = sfb[rows].reshape(n // 16, 16, 256)
    words = pack_words(cu).view(np.uint8).reshape(n // 16, 32768)
    return np.concatenate((words, pack_scales(sb)), axis=1)


# ---- kernel mirror
def fill(K, wA, c, s, spc):
    slot = np.zeros(UB, dtype=np.uint8)
    if K == 0:
        return wA[c * spc + s].copy()
    if K == 1:
        h, P = c & 1, (c >> 1) * spc + s
        for mt in range(2):
            v = wA[2 * P + mt]
            slot[mt * 16384:(mt + 1) * 16384] = v[16384 * h:16384 * h + 16384]
            slot[UC + mt * 2048:UC + mt * 2048 + 2048] = v[UC + 2048 * h:UC + 2048 * h + 2048]
        return slot
    q = c & 3
    for ub in range(2):
        U = (c >> 2) * 2 * spc + 2 * s + ub
        for mt in range(2):
            v = wA[2 * U + mt]
            o = ub * 18432
            slot[o + mt * 8192:o + mt * 8192 + 8192] = v[8192 * q:8192 * q + 8192]
            slot[o + 16384 + mt * 1024:o + 16384 + mt * 1024 + 1024] = v[UC + 1024 * q:UC + 1024 * q + 1024]
    return slot


def u32(buf, off):
    return int(np.frombuffer(buf[off:off + 4].tobytes(), dtype=np.uint32)[0])


def frag_from_word(qw, sbyte_g, sbyte_g8):
    """A fragment (16 x 16 as dict (row_local, kcol) -> value) of one lane from one code word, literal POS decode."""
    out = {}
    for r in range(4):
        for h in range(2):
            code = sum(((qw >> POS[r][h][b]) & 1) << b for b in range(4))
            row_hi = r & 1                         # a1 / a3: row g + 8
            kc = h + (8 if r >= 2 else 0)          # + 2t added by the caller
            sc = sbyte_g8 if row_hi else sbyte_g
            out[(row_hi, kc)] = fp4(code) * e4m3(sc)
    return out


def load_x(K, x, w, lane, part):
    g, t = lane >> 2, lane & 3
    NTL = 1 if K == 0 else 2 if K == 1 else 4
    NCH = 8 // NTL
    xr = [None] * 16
    for nt in range(NTL):
        k0 = 512 * w if K == 0 else (2048 * part + 256 * w if K == 1 else 1024 * part + 128 * w)
        base = k0 + 16 * t
        for cc in range(NCH):
            xr[nt * 2 * NCH + 2 * cc] = x[8 * nt + g, base + 64 * cc:base + 64 * cc + 8]
            xr[nt * 2 * NCH + 2 * cc + 1] = x[8 * nt + g, base + 64 * cc + 8:base + 64 * cc + 16]
    return xr


def xb(X0, X1, s, which):
    """bf16 pair of xb0 / xb1 for step s: (X0.x, X0.y) / (X0.z, X0.w) / (X1.x, X1.y) / (X1.z, X1.w); .x = elems 0,1 ..."""
    v = X0 if s < 2 else X1
    word = 2 * (s & 1) + which                    # 0 .x, 1 .y, 2 .z, 3 .w
    return v[2 * word], v[2 * word + 1]


def mma_lane_contrib(acc, A, B):
    """Accumulate D += A B for one warp: A[lane] dict, B[lane] dict; acc[lane][4]."""
    Am = np.zeros((16, 16))
    Bm = np.zeros((16, 8))
    for ln in range(32):
        g, t = ln >> 2, ln & 3
        for (rh, kc), v in A[ln].items():
            Am[g + 8 * rh, 2 * t + kc] = v
        for (kc, n), v in B[ln].items():
            Bm[2 * t + kc, g] = v
    D = Am @ Bm
    for ln in range(32):
        g, t = ln >> 2, ln & 3
        acc[ln][0] += D[g, 2 * t]; acc[ln][1] += D[g, 2 * t + 1]
        acc[ln][2] += D[g + 8, 2 * t]; acc[ln][3] += D[g + 8, 2 * t + 1]


def consume(K, slot, xrs, w):
    """-> acc[f][lane][4] (f = mt * ntl + nt) for warp w, literal mirror of consume0 / 1 / 2 (one J2 unit = slot bytes)."""
    if K == 0:
        F, MT, NTL, CH = 1, 1, 1, 8
    elif K == 1:
        F, MT, NTL, CH = 4, 2, 2, 4
    else:
        F, MT, NTL, CH = 8, 2, 4, 2
    acc = np.zeros((F, 32, 4))
    for mt in range(MT):
        for cc in range(CH):
            A_s = [[None] * 32 for _ in range(4)]
            for ln in range(32):
                if K == 0:
                    qoff = (w * 8 + cc) * 512 + ln * 16
                    sbase = UC + w * 512 + ln * 16
                    sw = u32(slot, sbase + 4 * (cc >> 1))
                elif K == 1:
                    qoff = mt * 16384 + ((w >> 1) * 8 + 4 * (w & 1) + cc) * 512 + ln * 16
                    sw = u32(slot, UC + mt * 2048 + (w >> 1) * 512 + ln * 16 + 8 * (w & 1) + 4 * (cc >> 1))
                else:
                    qoff = mt * 8192 + ((w >> 2) * 8 + 2 * (w & 3) + cc) * 512 + ln * 16
                    sw = u32(slot, 16384 + mt * 1024 + (w >> 2) * 512 + ln * 16 + 4 * (w & 3))
                byte = 2 * (cc & 1)
                sg, sg8 = (sw >> (8 * byte)) & 255, (sw >> (8 * (byte + 1))) & 255
                for s in range(4):
                    A_s[s][ln] = frag_from_word(u32(slot, qoff + 4 * s), sg, sg8)
            for s in range(4):
                for nt in range(NTL):
                    B = []
                    for ln in range(32):
                        xr = xrs[ln]
                        base = nt * 2 * CH + 2 * cc
                        b0 = xb(xr[base], xr[base + 1], s, 0)
                        b1 = xb(xr[base], xr[base + 1], s, 1)
                        B.append({(0, 0): b0[0], (1, 0): b0[1], (8, 0): b1[0], (9, 0): b1[1]})
                    mma_lane_contrib(acc[mt * NTL + nt], A_s[s], B)
    return acc


def run(K, wA, x, G, spc):
    ntile = G * spc
    out = np.zeros((ntile, 4, 32, 4))
    for c in range(G):
        part = (c & 1) if K == 1 else (c & 3)
        xrs = {w: [load_x(K, x, w, ln, part) for ln in range(32)] for w in range(8)}
        for s in range(spc):
            slot = fill(K, wA, c, s, spc)
            subs = [(slot, None)] if K < 2 else [(slot[ub * 18432:(ub + 1) * 18432], ub) for ub in range(2)]
            for sub, ub in subs:
                tot = sum(consume(K, sub, xrs[w], w) for w in range(8))         # K-slice sum over the 8 warps
                if K == 0:
                    tile0, ntl = c * spc + s, 1
                elif K == 1:
                    tile0, ntl = 2 * ((c >> 1) * spc + s), 2
                else:
                    tile0, ntl = 2 * ((c >> 2) * 2 * spc + 2 * s + ub), 4
                for f in range(tot.shape[0]):
                    out[tile0 + f // ntl, f % ntl] += tot[f]
    return out


def reference(codes, sfb, x, ntile, ntok):
    W = (fp4(codes).reshape(codes.shape[0], -1, 16) * e4m3(sfb)[..., None]).reshape(codes.shape[0], -1)
    h = W @ x[:ntok].T                                 # [8192, ntok]
    ref = np.zeros((ntile, 4, 32, 4))
    for R in range(ntile):
        rb = R % 512
        for ln in range(32):
            g, t = ln >> 2, ln & 3
            for cidx in range(4):
                i = g + 8 * (cidx >> 1)
                row = 8 * rb + i if i < 8 else 4096 + 8 * rb + (i - 8)
                for nt in range(4):
                    tok = 8 * nt + 2 * t + (cidx & 1)
                    if tok < ntok:
                        ref[R, nt, ln, cidx] = h[row, tok]
    return ref


def main():
    rng = np.random.default_rng(5)
    codes = rng.integers(0, 16, size=(8192, 4096), dtype=np.uint8)
    sfb = rng.integers(0x30, 0x48, size=(8192, 256), dtype=np.uint8)
    x = rng.standard_normal((32, 4096))
    # codes as unpacked nibbles (the shipped prepack unpacks the uint8 pairs first: vms_moe_lab._codes)
    wA = prepack_expert(codes, sfb)
    G, spc = 8, 2
    for K, ntok in ((0, 8), (1, 16), (2, 32)):
        out = run(K, wA, x, G, spc)
        ref = reference(codes, sfb, x, G * spc, ntok)
        err = np.abs(out - ref).max() / np.abs(ref).max()
        print(f"[vks-emu] J{K}: tiles {G * spc}, tokens {ntok}: max rel err {err:.2e} "
              f"({'OK' if err < 1e-12 else 'MISMATCH'})")


if __name__ == "__main__":
    main()

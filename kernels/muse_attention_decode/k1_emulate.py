"""K1 CPU emulation: the exact work decomposition of k1_v0.py's CUDA kernels, checked against a copy of
the judge's reference semantics (problems/muse_attention_decode/reference.py, copied below with FP8 caches
held as uint8, since CPU torch 2.5.1 cannot index float8 tensors).

Run:  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -I k1_emulate.py

Three layers, each mirroring a piece of the CUDA one-for-one (same names, same formulas):
  1. fragment_test(): per-lane gathers from the swizzled smem byte images (K: one LDS.128 per key row and
     head, V: LDS.64 + PRMT pairing), e4m3 -> f16 decode, reconstruction of the m16n8k16 A/B operands from
     the PTX fragment layouts, S = Q.K^T and O = P.V checked against direct math (dims permuted as the
     kernel permutes them); plus a bank-conflict assertion for every LDS phase.
  2. plan(): the on-device split plan (identical integer math in k1_main and k1_combine), with the
     sum(S_b) <= Gx bound asserted for random and scored-shape length sets.
  3. emulate_k1(): CTA/warp-granular pipeline (tiles, per-warp key ownership, masks incl. need_mask
     flag validation, per-warp online softmax with the -inf guard and the 2^8 P offset, fp16 Q and P
     roundings, cross-warp merge for L=1, partial format, combine with v_scale, RMSNorm, sigmoid(gate)).
     Checked with a copy of reference.check_implementation (own new entries fed back, E4M3-query bound).
"""

import math
import time

import torch

FP8 = torch.float8_e4m3fn

# ------------------------------------------------------------------------------------------------
# Kernel constants (mirrored in k1_v0.py: K1_TILE, K1_STAGES, K1_NEWK, ...)
# ------------------------------------------------------------------------------------------------
TILE = 64          # keys per pipeline stage
STAGES = 4         # cp.async ring depth
NEWK = 16          # keys in the new-token tile buffer (L <= 8 valid, rest zero and masked)
ROWB = 128         # bytes per key per (K or V) per head pair in smem: 2 heads x 64 dims x 1 B
NWARP = 8
CPS = 2            # target CTAs per SM (grid sizing)
MIN_CHUNK = 2      # minimum tiles per split
LOG2E = 1.4426950408889634
P_OFFSET = 8.0     # p = exp2(s2 - m + 8) in (0, 256]
RMS_EPS = 1e-5


def ceil_div(a, b):
    return -(-a // b)


# ------------------------------------------------------------------------------------------------
# 1. Fragment-level layout test
# ------------------------------------------------------------------------------------------------

def k_phys_chunk(key, c):
    """16-byte chunk c (0..7; head = c >> 2) of key row `key` in a K stage buffer."""
    return c ^ (4 * (key & 1))


def v_phys_chunk(key, c):
    return c ^ (2 * ((key >> 1) & 3))


def build_image(rows_u8, phys_chunk):
    """rows_u8 uint8 [n, 128] logical rows -> swizzled smem image [n, 128]."""
    n = rows_u8.shape[0]
    img = torch.zeros_like(rows_u8)
    for key in range(n):
        for c in range(8):
            pc = phys_chunk(key, c)
            img[key, 16 * pc:16 * pc + 16] = rows_u8[key, 16 * c:16 * c + 16]
    return img


def dec(u8):
    """e4m3 bytes -> float (exact; the f16 the kernel's cvt.rn.f16x2.e4m3x2 produces)."""
    return u8.contiguous().view(FP8).float()


def byte_perm(x4, y4, sel):
    """__byte_perm(x, y, sel) on 4-byte little-endian arrays: result byte i = {x0..x3,y0..y3}[nibble i]."""
    src = list(x4) + list(y4)
    return [src[(sel >> (4 * i)) & 0xF] for i in range(4)]


def fragment_test(seed=0):
    g_ = torch.Generator().manual_seed(seed)
    nkeys = 64
    Kl = (torch.randn((nkeys, 2, 64), generator=g_) * 2).to(FP8).view(torch.uint8).reshape(nkeys, 128)
    Vl = torch.randn((nkeys, 2, 64), generator=g_).to(FP8).view(torch.uint8).reshape(nkeys, 128)
    imgK = build_image(Kl, k_phys_chunk)
    imgV = build_image(Vl, v_phys_chunk)
    Qs = (torch.randn((2, 64, 64), generator=g_)).half()   # [head][row][dim]
    Kd = dec(Kl).view(nkeys, 2, 64)
    Vd = dec(Vl).view(nkeys, 2, 64)
    worst = 0.0
    for NT, kw_list, r0_list in ((2, (0, 16, 32, 48), (0,)), (8, (0,), (0, 16, 32, 48))):
        for h in range(2):
            for kw in kw_list:
                for r0 in r0_list:
                    # ---- QK: per n-tile jn, 4 k-steps
                    S_all = torch.zeros((16, 8 * NT))
                    for jn in range(NT):
                        S = torch.zeros((16, 8))
                        for s in range(4):
                            A = torch.zeros((16, 16))
                            Bm = torch.zeros((16, 8))
                            for lane in range(32):
                                g, t = lane >> 2, lane & 3
                                # A fragment (Q) from smem fp16 [row][64]: dims 16t+4s .. +3
                                d0 = 16 * t + 4 * s
                                A[g, 2 * t] = Qs[h, r0 + g, d0].float()
                                A[g, 2 * t + 1] = Qs[h, r0 + g, d0 + 1].float()
                                A[g + 8, 2 * t] = Qs[h, r0 + g + 8, d0].float()
                                A[g + 8, 2 * t + 1] = Qs[h, r0 + g + 8, d0 + 1].float()
                                A[g, 2 * t + 8] = Qs[h, r0 + g, d0 + 2].float()
                                A[g, 2 * t + 9] = Qs[h, r0 + g, d0 + 3].float()
                                A[g + 8, 2 * t + 8] = Qs[h, r0 + g + 8, d0 + 2].float()
                                A[g + 8, 2 * t + 9] = Qs[h, r0 + g + 8, d0 + 3].float()
                                # B fragment (K): one 16-byte load of key kw+8jn+g, chunk 4h+t (swizzled)
                                key = kw + 8 * jn + g
                                pc = k_phys_chunk(key, 4 * h + t)
                                c16 = imgK[key, 16 * pc:16 * pc + 16]
                                w = c16[4 * s:4 * s + 4]           # word s
                                lo = dec(w[0:2])                     # cvt(lo16): (byte0, byte1)
                                hi = dec(w[2:4])                     # cvt(hi16): (byte2, byte3)
                                Bm[2 * t, g], Bm[2 * t + 1, g] = lo[0], lo[1]
                                Bm[2 * t + 8, g], Bm[2 * t + 9, g] = hi[0], hi[1]
                            S += A @ Bm
                        S_all[:, 8 * jn:8 * jn + 8] = S
                    ref = Qs[h, r0:r0 + 16].float() @ Kd[kw:kw + 8 * NT, h].T
                    err = float((S_all - ref).abs().max() / ref.abs().max())
                    worst = max(worst, err)
                    assert err < 1e-5, f"QK fragment mismatch NT={NT} h={h} kw={kw} r0={r0}: {err}"
                    # ---- PV: P = S (any matrix) as A from C fragments, V via LDS.64 + PRMT
                    P = torch.randn((16, 8 * NT), generator=g_).half().float()
                    O = torch.zeros((16, 64))
                    for kk in range(NT // 2):
                        kb = kw + 16 * kk
                        A = torch.zeros((16, 16))
                        for lane in range(32):
                            g, t = lane >> 2, lane & 3
                            # C fragments of S n-tiles 2kk and 2kk+1 -> A fragment of P
                            for half, jn in ((0, 2 * kk), (1, 2 * kk + 1)):
                                c0 = P[g, 8 * jn + 2 * t]
                                c1 = P[g, 8 * jn + 2 * t + 1]
                                c2 = P[g + 8, 8 * jn + 2 * t]
                                c3 = P[g + 8, 8 * jn + 2 * t + 1]
                                A[g, 2 * t + 8 * half] = c0
                                A[g, 2 * t + 1 + 8 * half] = c1
                                A[g + 8, 2 * t + 8 * half] = c2
                                A[g + 8, 2 * t + 1 + 8 * half] = c3
                        for jn in range(8):
                            Bm = torch.zeros((16, 8))
                            for lane in range(32):
                                g, t = lane >> 2, lane & 3
                                words = []
                                for x in (kb + 2 * t, kb + 2 * t + 1, kb + 2 * t + 8, kb + 2 * t + 9):
                                    c = 4 * h + (g >> 1)
                                    pc = v_phys_chunk(x, c)
                                    b8 = imgV[x, 16 * pc + 8 * (g & 1):16 * pc + 8 * (g & 1) + 8].tolist()
                                    words.append((b8[0:4], b8[4:8]))
                                (A0, A1), (B0, B1), (C0, C1), (D0, D1) = words
                                sel = 0x5140 if (jn & 1) == 0 and (jn & 2) == 0 else None
                                # jn -> (word half, selector, which cvt half)
                                src01 = (A0, B0) if jn < 4 else (A1, B1)
                                src23 = (C0, D0) if jn < 4 else (C1, D1)
                                sel = 0x5140 if (jn & 2) == 0 else 0x7362
                                r01 = byte_perm(src01[0], src01[1], sel)
                                r23 = byte_perm(src23[0], src23[1], sel)
                                pick = (jn & 1) * 2   # lo16 for even jn, hi16 for odd
                                v01 = dec(torch.tensor(r01[pick:pick + 2], dtype=torch.uint8))
                                v23 = dec(torch.tensor(r23[pick:pick + 2], dtype=torch.uint8))
                                Bm[2 * t, g], Bm[2 * t + 1, g] = v01[0], v01[1]
                                Bm[2 * t + 8, g], Bm[2 * t + 9, g] = v23[0], v23[1]
                            Oj = A @ Bm
                            for n in range(8):
                                O[:, 8 * n + jn] += Oj[:, n]     # col n of n-tile jn <-> dim 8n + jn
                    refo = P @ Vd[kw:kw + 8 * NT, h]
                    err = float((O - refo).abs().max() / refo.abs().max())
                    worst = max(worst, err)
                    assert err < 1e-5, f"PV fragment mismatch NT={NT} h={h} kw={kw} r0={r0}: {err}"
    return worst


def fragment_test_fp8(seed=1):
    """K1 v1 L=8 QK on mma.sync m16n8k32 .e4m3 (PTX ISA m16n8k32 8-bit layouts):
       A (16x32, row): a0 = (row g, k 4t+i), a1 = (row g+8, k 4t+i), a2 = (row g, k 16+4t+i), a3 = (row g+8, ...),
       B (32x8, col):  b0 = (k 4t+i, n g), b1 = (k 16+4t+i, n g); byte i of each u32 = element i (lowest k lowest).
       Dim map (both operands): k-step s, logical k 4t+i <-> dim 16t+8s+i, k 16+4t+i <-> dim 16t+8s+4+i, so lane
       (g,t) reads the 16 bytes [16t, 16t+16) of a K row (one LDS.128, words w0..w3; step s uses w[2s], w[2s+1])
       and of a Q row (e4m3 bytes in smem). Matrices are rebuilt from per-byte extraction."""
    g_ = torch.Generator().manual_seed(seed)
    nkeys = 64
    Kl = (torch.randn((nkeys, 2, 64), generator=g_) * 2).to(FP8).view(torch.uint8).reshape(nkeys, 128)
    imgK = build_image(Kl, k_phys_chunk)
    Kd = dec(Kl).view(nkeys, 2, 64)
    Q8 = (torch.randn((2, 64, 64), generator=g_) * 2).clamp(-448, 448).to(FP8).view(torch.uint8)   # [h][row][dim]
    Qd = dec(Q8)
    worst = 0.0
    for h in range(2):
        for kw in (0, 32):
            for r0 in (0, 16, 32, 48):
                S_all = torch.zeros((16, 32))
                for jn in range(4):
                    S = torch.zeros((16, 8))
                    for s in range(2):
                        A = torch.zeros((16, 32))
                        Bm = torch.zeros((32, 8))
                        for lane in range(32):
                            g, t = lane >> 2, lane & 3
                            qa = Q8[h, r0 + g, 16 * t:16 * t + 16]          # lane's 16 Q bytes, row g
                            qb = Q8[h, r0 + g + 8, 16 * t:16 * t + 16]      # row g+8
                            a0, a2 = qa[8 * s:8 * s + 4], qa[8 * s + 4:8 * s + 8]
                            a1, a3 = qb[8 * s:8 * s + 4], qb[8 * s + 4:8 * s + 8]
                            key = kw + 8 * jn + g
                            pc = k_phys_chunk(key, 4 * h + t)
                            c16 = imgK[key, 16 * pc:16 * pc + 16]
                            b0, b1 = c16[8 * s:8 * s + 4], c16[8 * s + 4:8 * s + 8]   # words 2s, 2s+1
                            for i in range(4):
                                A[g, 4 * t + i] = dec(a0[i:i + 1])[0]
                                A[g + 8, 4 * t + i] = dec(a1[i:i + 1])[0]
                                A[g, 16 + 4 * t + i] = dec(a2[i:i + 1])[0]
                                A[g + 8, 16 + 4 * t + i] = dec(a3[i:i + 1])[0]
                                Bm[4 * t + i, g] = dec(b0[i:i + 1])[0]
                                Bm[16 + 4 * t + i, g] = dec(b1[i:i + 1])[0]
                        S += A @ Bm
                    S_all[:, 8 * jn:8 * jn + 8] = S
                ref = Qd[h, r0:r0 + 16] @ Kd[kw:kw + 32, h].T
                err = float((S_all - ref).abs().max() / ref.abs().max())
                worst = max(worst, err)
                assert err < 1e-6, f"fp8 QK fragment mismatch h={h} kw={kw} r0={r0}: {err}"
    return worst


def bank_test():
    """Every LDS phase of the K (.128, 8-lane phases) and V (.64, 16-lane phases) loads is conflict-free."""
    n_checked = 0
    for kw in (0, 16, 32, 48):
        for h in range(2):
            # K: n-tiles of the warp's keys (8 n-tiles cover 64 keys from kw=0; 2 from kw=16q4)
            for jn in range(8 if kw == 0 else 2):
                for phase in range(4):
                    banks = []
                    for lane in range(8 * phase, 8 * phase + 8):
                        g, t = lane >> 2, lane & 3
                        key = kw + 8 * jn + g
                        addr = key * ROWB + 16 * k_phys_chunk(key, 4 * h + t)
                        banks += [(addr // 4 + i) % 32 for i in range(4)]
                    assert len(set(banks)) == 32, f"K bank conflict kw={kw} h={h} jn={jn} phase={phase}"
                    n_checked += 1
            for kk in range(4 if kw == 0 else 1):
                kb = kw + 16 * kk
                for e in (0, 1, 8, 9):
                    for phase in range(2):
                        banks = []
                        for lane in range(16 * phase, 16 * phase + 16):
                            g, t = lane >> 2, lane & 3
                            x = kb + 2 * t + e
                            addr = x * ROWB + 16 * v_phys_chunk(x, 4 * h + (g >> 1)) + 8 * (g & 1)
                            banks += [(addr // 4 + i) % 32 for i in range(2)]
                        assert len(set(banks)) == 32, f"V bank conflict kb={kb} h={h} e={e} phase={phase}"
                        n_checked += 1
    return n_checked


def newtile_write_test():
    """The new-token writer's per-lane (2 bytes at logical byte 64h + 2*lane of key j) placement equals
    build_image's swizzle."""
    g_ = torch.Generator().manual_seed(5)
    rows = torch.randint(0, 120, (NEWK, 128), generator=g_, dtype=torch.uint8)
    for phys, name in ((k_phys_chunk, "K"), (v_phys_chunk, "V")):
        img = torch.zeros((NEWK, 128), dtype=torch.uint8)
        for j in range(NEWK):
            for h in range(2):
                for lane in range(32):
                    b = 64 * h + 2 * lane
                    c, off = b >> 4, b & 15
                    dst = 16 * phys(j, c) + off
                    img[j, dst:dst + 2] = rows[j, b:b + 2]
        assert torch.equal(img, build_image(rows, phys)), f"new-tile {name} writer mismatch"


# ------------------------------------------------------------------------------------------------
# 2. Split plan (k1_plan in CUDA)
# ------------------------------------------------------------------------------------------------

def n_old_keys(seq, W):
    return min(seq, W - 1) if W else seq


def split_count(nt, C):
    return max(1, ceil_div(nt, C))


def plan(seq_lens, W, Gx, min_chunk=MIN_CHUNK):
    """Smallest per-split tile cap C with sum_b max(1, ceil(nt_b / C)) <= Gx (binary search inside
    [max(min_chunk, ceil(T/Gx)), ceil(T/(Gx-B))], where the upper end always fits); split s of request b
    covers tiles [s*nt_b // S_b, (s+1)*nt_b // S_b) (balanced, each <= C tiles)."""
    B = len(seq_lens)
    nt = [ceil_div(n_old_keys(s, W), TILE) for s in seq_lens]
    T = sum(nt)
    lo = max(min_chunk, ceil_div(T, Gx))
    hi = max(lo, ceil_div(T, max(1, Gx - B)))
    n_iter = 0
    while lo < hi:
        mid = (lo + hi) >> 1
        if sum(split_count(n, mid) for n in nt) <= Gx:
            hi = mid
        else:
            lo = mid + 1
        n_iter += 1
    C = lo
    S = [split_count(n, C) for n in nt]
    off, acc = [], 0
    for x in S:
        off.append(acc)
        acc += x
    assert acc <= Gx, (acc, Gx)
    return nt, C, S, off


def grid_x(n_sm, B, HP, cps=CPS):
    return max(max(1, (n_sm * cps) // HP), B + 1)


def cta_work(c, pl):
    nt, C, S, off = pl
    for b in range(len(S)):
        if off[b] <= c < off[b] + S[b]:
            s = c - off[b]
            t0 = (s * nt[b]) // S[b]
            t1 = ((s + 1) * nt[b]) // S[b]
            return b, s, t0, t1, s == S[b] - 1
    return None


def plan_test():
    g_ = torch.Generator().manual_seed(1)
    for _ in range(300):
        B = int(torch.randint(1, 70, (1,), generator=g_))
        W = 2048 if bool(torch.rand(1, generator=g_) < 0.5) else 0
        n_sm = int(torch.randint(1, 140, (1,), generator=g_))
        HP = int((1, 8)[int(torch.randint(0, 2, (1,), generator=g_))])
        Gx = grid_x(n_sm, B, HP)
        seq = torch.randint(0, 300000, (B,), generator=g_).tolist()
        if bool(torch.rand(1, generator=g_) < 0.3):
            seq = [int(x) % 200 for x in seq]
        mc = int((1, 2, 4)[int(torch.randint(0, 3, (1,), generator=g_))])
        pl = plan(seq, W, Gx, mc)
        # every tile of every request covered exactly once, new tile in the last split
        cover = [[0] * pl[0][b] for b in range(B)]
        lasts = [0] * B
        for c in range(Gx):
            wk = cta_work(c, pl)
            if wk is None:
                continue
            b, s, t0, t1, last = wk
            assert t0 <= t1 and (t1 - t0) <= pl[1]
            assert (t1 > t0) or pl[0][b] == 0, "empty split for a request with tiles"
            for i in range(t0, t1):
                cover[b][i] += 1
            lasts[b] += int(last)
        assert all(all(x == 1 for x in cv) for cv in cover)
        assert all(x == 1 for x in lasts)


# ------------------------------------------------------------------------------------------------
# Reference (copied from problems/muse_attention_decode/reference.py + checks.py; caches as uint8)
# ------------------------------------------------------------------------------------------------
HEAD_DIM = 64
HEADS = {"tp8": (16, 2), "dp": (128, 16)}
SLIDING_WINDOW = 2048
ROPE_THETA = 500_000.0
QK_SCALE_FACTOR = 5.4730064863875
SM_SCALE = HEAD_DIM ** -0.5
PAGE = 64
K_SCALE = 1.0
REL_L2_TOL = 0.15
FP8_QUERY_FACTOR = 2.5
FP8_QUERY_FLOOR = 0.03
FP8_QUERY_SLACK = 0.015
QUERY_DTYPE = torch.bfloat16
CACHE_TOL = 8e-2
TIGHT_TOL = 1.0e-2   # emulator-only: overall rel L2 1%, worst row 5%
SOFT_C2 = {"sliding": (0.3, 0.75), "full": (0.5, 0.62)}
COHERENT_HEAD = 2048
COHERENT_ALPHA = 1.0
AGENTIC_SPREAD = 0.075
CACHE_CHUNK = 1 << 20
ROW_FACTOR = 5.0
ROW_NORM_FLOOR = 0.1


def relative_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def query_precision_bound(e4m3_err, factor, floor, cap, slack=0.0):
    return min(cap, max(floor, factor * e4m3_err + slack))


def compare(output, ref, tol):
    if output.shape != ref.shape or output.dtype != ref.dtype:
        return False, f"expected {tuple(ref.shape)} {ref.dtype}, got {tuple(output.shape)} {output.dtype}"
    out = output.float()
    if not torch.isfinite(out).all():
        return False, "output contains NaN or Inf"
    r = ref.float()
    diff = out - r
    err = float(diff.norm() / r.norm().clamp(min=1e-12))
    if err > tol:
        return False, f"relative L2 error {err:.3e} > {tol:g}"
    rows_diff = diff.reshape(-1, r.shape[-1]).norm(dim=-1)
    rows_ref = r.reshape(-1, r.shape[-1]).norm(dim=-1)
    floor = (ROW_NORM_FLOOR * rows_ref.mean()).clamp(min=1e-12)
    row_err = rows_diff / torch.maximum(rows_ref, floor)
    worst_err = float(row_err.max())
    if worst_err > ROW_FACTOR * tol:
        return False, f"relative L2 error {err:.3e}, but worst row {worst_err:.3e} > {ROW_FACTOR:g} x {tol:g}"
    return True, f"relative L2 error {err:.3e} (worst row {worst_err:.3e})"


def _rms(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope(x, pos, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=x.device).float() / HEAD_DIM))
    ang = pos.float()[..., None] * inv_freq
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def _quant(x, scale):
    return (x.float() / scale).to(FP8).view(torch.uint8)


def _dequant(x_u8, scale):
    return x_u8.contiguous().view(FP8).float() * scale


def _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, scale):
    hkv, hq = v_cache.shape[-2], q.shape[-2]
    group = hq // hkv
    tail_gen = torch.Generator(device=v_cache.device).manual_seed(seed + 0xC0DE)
    u = torch.randn((hkv, HEAD_DIM), generator=tail_gen, device=v_cache.device)
    u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
    qn = _rms(q) * QK_SCALE_FACTOR
    if layer == "sliding":
        pos = seq_lens.long()[:, None] + torch.arange(q.shape[1], device=q.device)
        qn = _rope(qn, pos, ROPE_THETA)
    k_flat = k_cache.view(-1, hkv, HEAD_DIM)
    v_flat = v_cache.view(-1, hkv, HEAD_DIM)
    for b in range(len(seq_lens)):
        seq = int(seq_lens[b])
        first = max(0, seq - SLIDING_WINDOW + 1) if layer == "sliding" else 0
        past = torch.arange(first, seq, device=q.device)
        rows = page_table[b].long()[past // page] * page + past % page
        keys = _dequant(k_flat[rows], scale)
        logits = torch.einsum("lkgd,skd->kgls", qn[b].view(-1, hkv, group, HEAD_DIM), keys)
        best = logits.flatten(1, 2).amax(1)
        if best.shape[1] <= COHERENT_HEAD:
            continue
        tail = best < best.topk(COHERENT_HEAD, -1).values[:, -1:]
        vals = _dequant(v_flat[rows], scale) + COHERENT_ALPHA * tail.T[..., None] * u
        v_flat[rows] = _quant(vals, scale)


def _longest(ctx, spread):
    if spread == "fixed":
        return ctx
    if spread == "agentic":
        return math.ceil(ctx * (1 + AGENTIC_SPREAD))
    return 4 * ctx


def _lengths(gen, batch, ctx, spread, device):
    if spread == "fixed":
        return torch.full((batch,), ctx, dtype=torch.int64, device=device)
    if spread == "agentic":
        u = torch.rand((batch,), generator=gen, device=device)
        return (ctx * (1 + AGENTIC_SPREAD * (2 * u - 1))).long().clamp(1, _longest(ctx, spread))
    z = torch.randn((batch,), generator=gen, device=device)
    return (ctx * torch.exp(z - 0.5)).clamp(16, 4 * ctx).long()


def _fill_cache(gen, shape, device, scale=K_SCALE, seed=0, dist="normal", layer="full"):
    hkv = shape[-2]
    k_cache = torch.empty(shape, dtype=torch.uint8, device=device)
    v_cache = torch.empty(shape, dtype=torch.uint8, device=device)
    mix = None
    if dist in ("soft", "coherent"):
        soft = torch.Generator(device=device).manual_seed(seed + 0x50F7)
        u = torch.randn((hkv, HEAD_DIM), generator=soft, device=device)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
        lo, hi = SOFT_C2[layer]
        c2 = lo + (hi - lo) * torch.rand((hkv, 1), generator=soft, device=device)
        mix = (c2.sqrt() * u, (1 - c2).sqrt())
    for dst, keys in ((k_cache, True), (v_cache, False)):
        flat = dst.view(-1, hkv, HEAD_DIM)
        for at in range(0, flat.shape[0], CACHE_CHUNK):
            n = min(CACHE_CHUNK, flat.shape[0] - at)
            x = torch.randn((n, hkv, HEAD_DIM), generator=gen, device=device)
            if keys and mix is not None:
                x = mix[0] + mix[1] * x
            flat[at:at + n] = _quant(x, scale)
    return k_cache, v_cache


def generate_input(batch, ctx, heads, layer, seed, q_len=1, spread="fixed", dist="normal", page=PAGE,
                   kv_scale=K_SCALE, device="cpu", **_):
    gen = torch.Generator(device=device).manual_seed(seed)
    hq, hkv = HEADS[heads]
    seq_lens = _lengths(gen, batch, ctx, spread, device)
    pages_per = (seq_lens + q_len + page - 1) // page
    longest = _longest(ctx, spread)
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    shape = (total, page, hkv, HEAD_DIM)
    k_cache, v_cache = _fill_cache(gen, shape, device, kv_scale, seed, dist, layer)
    q = (2 * torch.randn((batch, q_len, hq, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)
    if dist == "coherent":
        _coherent_tail(v_cache, k_cache, q, seq_lens, page_table, layer, seed, page, kv_scale)
    if dist == "edges":
        qn = _rms(q[:, 0])
        if layer == "sliding":
            qn = _rope(qn, seq_lens, ROPE_THETA)
        planted = _quant(qn[:, :: hq // hkv], kv_scale)
        k_flat = k_cache.view(-1, hkv, HEAD_DIM)
        for b in range(batch):
            p = int(seq_lens[b])
            spots = [p - SLIDING_WINDOW + 1, p - SLIDING_WINDOW] if layer == "sliding" else [0]
            for t in spots:
                if 0 <= t < p:
                    k_flat[int(page_table[b, t // page]) * page + t % page] = planted[b]
    k_new = (2 * torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device)).to(torch.bfloat16)
    v_new = torch.randn((batch, q_len, hkv, HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)
    gate = torch.randn((batch, q_len, hq * HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)
    qkv = torch.cat([q.flatten(2), k_new.flatten(2), v_new.flatten(2), gate], -1)
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=device)
    return {
        "qkv": qkv, "positions": positions, "cache_loc": cache_loc, "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q, "scheduler_metadata": None,
        "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "v_scale": torch.tensor(kv_scale, dtype=torch.float32, device=device),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "config": {
            "batch": batch, "q_len": q_len, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": HEAD_DIM,
            "page_size": page, "window": SLIDING_WINDOW if layer == "sliding" else 0,
            "rope_theta": ROPE_THETA if layer == "sliding" else 0.0, "qk_scale_factor": QK_SCALE_FACTOR,
            "sm_scale": SM_SCALE, "kv_scale": kv_scale,
        },
    }


def split_qkv(data):
    cfg = data["config"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    batch, length = cfg["batch"], cfg["q_len"]
    q, k, v, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)
    return (q.reshape(batch, length, hq, d), k.reshape(batch, length, hkv, d), v.reshape(batch, length, hkv, d),
            gate)


def _rows(data, positions):
    size = data["config"]["page_size"]
    page = torch.gather(data["page_table"].long(), 1, positions // size)
    return page * size + positions % size


def new_slots(data):
    pos = data["seq_lens"].long()[:, None] + torch.arange(data["config"]["q_len"])
    return _rows(data, pos)


def ref_kernel(data, query_dtype=None, new_kv=None):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    q_in, k_in, v_in, gate = split_qkv(data)
    pos = data["seq_lens"].long()[:, None] + torch.arange(length, device=q_in.device)
    q = _rms(q_in) * cfg["qk_scale_factor"]
    k = _rms(k_in)
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    query_dtype = query_dtype or QUERY_DTYPE
    if query_dtype == torch.bfloat16:
        q = q.to(torch.bfloat16).float()
    else:
        q = (q / cfg["qk_scale_factor"]).to(query_dtype).float() * cfg["qk_scale_factor"]
    k_new, v_new = _quant(k, data["k_scale"]), _quant(v_in, data["v_scale"])
    if new_kv is not None:
        k_new, v_new = new_kv
    k_flat = data["k_cache"].view(-1, hkv, HEAD_DIM)
    v_flat = data["v_cache"].view(-1, hkv, HEAD_DIM)
    group = hq // hkv
    out = torch.empty((batch, length, hq, HEAD_DIM))
    for b in range(batch):
        seq = int(data["seq_lens"][b])
        past = torch.arange(seq)
        size = cfg["page_size"]
        past_rows = data["page_table"][b].long()[past // size] * size + past % size
        keys_past = _dequant(k_flat[past_rows], data["k_scale"])
        vals_past = _dequant(v_flat[past_rows], data["v_scale"])
        keys = torch.cat([keys_past, _dequant(k_new[b], data["k_scale"])])
        vals = torch.cat([vals_past, _dequant(v_new[b], data["v_scale"])])
        for j in range(length):
            p = seq + j
            start = max(0, p - cfg["window"] + 1) if cfg["window"] else 0
            kb, vb = keys[start:p + 1], vals[start:p + 1]
            att = torch.einsum("kgd,tkd->kgt", q[b, j].view(hkv, group, HEAD_DIM), kb) * cfg["sm_scale"]
            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vb).reshape(hq, HEAD_DIM)
    out = _rms(out) * torch.sigmoid(gate.float()).view(batch, length, hq, HEAD_DIM)
    return out.reshape(batch, length, hq * HEAD_DIM), k_new, v_new


def check_implementation(data, out, k_cache_after, v_cache_after):
    """reference.check_implementation with collect/unpack inlined: new entries read at new_slots."""
    cfg = data["config"]
    hkv = cfg["num_kv_heads"]
    slots = new_slots(data)
    k = k_cache_after.view(-1, hkv, HEAD_DIM)[slots]
    v = v_cache_after.view(-1, hkv, HEAD_DIM)[slots]
    _, ref_k, ref_v = ref_kernel(data)
    for name, got, want, scale in (("key", k, ref_k, data["k_scale"]), ("value", v, ref_v, data["v_scale"])):
        ok, kv_msg = compare(_dequant(got, scale), _dequant(want, scale), CACHE_TOL)
        if not ok:
            return False, f"new {name} cache entries: {kv_msg}"
    ref = ref_kernel(data, new_kv=(k, v))[0].to(torch.bfloat16)
    e4m3 = relative_l2(ref_kernel(data, torch.float8_e4m3fn, new_kv=(k, v))[0].to(torch.bfloat16), ref)
    tol = query_precision_bound(e4m3, FP8_QUERY_FACTOR, FP8_QUERY_FLOOR, REL_L2_TOL, FP8_QUERY_SLACK)
    ok, msg = compare(out, ref, tol)
    # Tight self-consistency check (not the judge's). L=1 (FP16 Q/P, exact K/V) sits at ~0.5% vs the BF16-query
    # reference; L=8 (v1: E4M3 Q, FP16 P) is checked against the reference with E4M3 queries (the same rounding
    # point: normalized, roped, before x5.473). Masking/indexing slips the loose judge bound tolerates still fail.
    tref = ref if cfg["q_len"] == 1 else ref_kernel(data, torch.float8_e4m3fn, new_kv=(k, v))[0].to(torch.bfloat16)
    tight = compare(out, tref, TIGHT_TOL)
    ok = ok and tight[0]
    return ok, (f"{msg} [bound {tol:.3f}; e4m3-query ref err {e4m3:.4f}; tight vs {'bf16' if cfg['q_len'] == 1 else 'e4m3'}"
                f"-query ref {tight[1].split(' (')[0].replace('relative L2 error ', '')}: {'ok' if tight[0] else 'FAIL'}]")


# ------------------------------------------------------------------------------------------------
# 3. Pipeline emulation of k1_main + k1_combine
# ------------------------------------------------------------------------------------------------

def quant_dev(x, scale):
    """Device fp32 -> e4m3: IEEE divide, RNE, SATFINITE (clamp to +-448 before torch's cast)."""
    return (x.float() / scale).clamp(-448.0, 448.0).to(FP8).view(torch.uint8)


def warp_roles(L):
    """k1_main warp roles. L=1: warp w = (KV head w&1, key quarter w>>1 of each 64-key stage), rows = 8 q heads.
    L=8 (v1): warp w = (KV head w>>2, row half rh=(w>>1)&1 -> rows 32rh..32rh+31 = 2 m-tiles, key half kh=w&1 ->
    keys 32kh..32kh+31). Warps of one merge group own the same rows and different keys; only the first key slice
    (q4 == 0 / kh == 0) attends the 16-key new tile."""
    roles = []
    for w in range(NWARP):
        if L == 1:
            h, q = w & 1, w >> 1
            roles.append(dict(h=h, ko=16 * q, kn=16, rows=torch.arange(0, 8), new=(q == 0), grp=(h, 0), slot=q))
        else:
            h, rh, kh = w >> 2, (w >> 1) & 1, w & 1
            roles.append(dict(h=h, ko=32 * kh, kn=32, rows=torch.arange(32 * rh, 32 * rh + 32), new=(kh == 0),
                              grp=(h, rh), slot=kh))
    return roles


def finalize_rows(o4, vs, qkv, out, b, hp, rows_r, L, Hq, Hkv, wsum):
    """Device finalize at (row, quad) granularity: o4 [nrows, 16, 4] = sum_s w_s O_s (quads u = dims 4u..4u+3),
    wsum [nrows]. Row r -> KV head h = r // 8L, token j = (r % 8L) >> 3, q head (2hp+h)*8 + (r & 7)."""
    R8 = 8 * L
    o4 = o4 * (vs / wsum)[:, None, None]
    ss = o4.pow(2).sum(-1).sum(-1)                       # half-warp (16 quads) reduce of squares
    rs = torch.rsqrt(ss * (1.0 / 64.0) + RMS_EPS)
    for i, r in enumerate(rows_r.tolist()):
        h, rr = r // R8, r % R8
        j, gi = rr >> 3, rr & 7
        qh = (2 * hp + h) * 8 + gi
        tok = b * L + j
        for u in range(16):
            gcol = (Hq + 2 * Hkv) * 64 + qh * 64 + 4 * u
            gate = qkv[tok, gcol:gcol + 4].float()
            out[tok, qh * 64 + 4 * u:qh * 64 + 4 * u + 4] = (o4[i, u] * rs[i] * torch.sigmoid(gate)).to(torch.bfloat16)


def merge_fold(part_O_bh, part_lse_bh, S):
    """Last-CTA merge (k1_main fold path): items (row, quad); s = 0..S-1 in order. part_O_bh [S, ROWS, 64]."""
    lse = part_lse_bh[:S]                                # [S, ROWS]
    M = lse.max(0).values
    Mu = torch.where(M == float("-inf"), torch.zeros_like(M), M)
    o4 = torch.zeros((lse.shape[1], 16, 4))
    wsum = torch.zeros(lse.shape[1])
    for s_ in range(S):
        w = torch.exp2(lse[s_] - Mu)
        wsum = wsum + w
        o4 = o4 + w[:, None, None] * part_O_bh[s_].view(-1, 16, 4)
    return o4, wsum


def merge_sep(part_O_bh, part_lse_bh, S, row, NSG):
    """k1_combine (separate path): CTA per (b, j, qh) row; thread = (split group sg = tid/16, quad u = tid%16);
    group sg sums s = sg, sg+NSG, ...; groups reduced in order sg = 0..NSG-1."""
    lse = part_lse_bh[:S, row]
    M = lse.max()
    Mu = 0.0 if M == float("-inf") else M
    acc = torch.zeros((NSG, 16, 4))
    ws = torch.zeros(NSG)
    for sg in range(NSG):
        for s_ in range(sg, S, NSG):
            w = torch.exp2(lse[s_] - Mu)
            ws[sg] += w
            acc[sg] += w * part_O_bh[s_, row].view(16, 4)
    o4 = torch.zeros((16, 4))
    wt = torch.zeros(())
    for sg in range(NSG):
        o4 = o4 + acc[sg]
        wt = wt + ws[sg]
    return o4[None], wt[None]


def emulate_k1(data, n_sm=132, cps=CPS, min_chunk=MIN_CHUNK, stats=None, visits=None, fold=True, nsg=8):
    """Runs the k1_main(+k1_combine) decomposition; writes new K/V into data's caches (uint8) in place; returns
    out. fold=True: the last CTA of each (request, head pair) merges the splits (S_b == 1: straight from smem);
    fold=False: separate k1_combine with nsg split groups per (b, j, q head)."""
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    Hq, Hkv, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    W, P = cfg["window"], cfg["page_size"]
    assert D == 64 and Hq == 8 * Hkv and Hkv % 2 == 0 and L in (1, 8) and (P & (P - 1)) == 0
    shift = P.bit_length() - 1
    HP = Hkv // 2
    R8 = 8 * L                      # rows per KV head (8 q heads x L tokens)
    ROWS = 16 * L                   # rows per CTA (2 KV heads)
    Gx = grid_x(n_sm, B, HP, cps)
    seq_l = [int(x) for x in data["seq_lens"].tolist()]
    pl = plan(seq_l, W, Gx, min_chunk)
    nt_l, C, S_list, off = pl
    stride = (2 * Hq + 2 * Hkv) * 64
    qkv = data["qkv"].reshape(B * L, stride)
    kc = data["k_cache"].view(-1, Hkv * 64)
    vc = data["v_cache"].view(-1, Hkv * 64)
    pt = data["page_table"].long()
    cache_loc = data["cache_loc"].long()
    ks = data["k_scale"].float()
    vs = data["v_scale"].float()
    qk_sm = torch.tensor(cfg["qk_scale_factor"] * cfg["sm_scale"], dtype=torch.float32)
    c_log2 = (qk_sm * ks * torch.tensor(LOG2E, dtype=torch.float32)).float()
    theta = cfg["rope_theta"]
    inv_freq = (1.0 / (theta ** (torch.arange(0, 64, 2).float() / 64))) if theta else None
    part_O = torch.full((HP, Gx, ROWS, 64), float("nan"))
    part_lse = torch.full((HP, Gx, ROWS), float("nan"))
    out = torch.full((B * L, Hq * 64), float("nan")).to(torch.bfloat16)
    NEG = float("-inf")
    n_need = n_tiles = 0
    roles = warp_roles(L)
    nr = len(roles[0]["rows"])

    def rope_rows(x, pos):     # x [n, 64] fp32, pos [n] int; device: cos/sin table per (token, pair)
        ang = pos.float()[:, None] * inv_freq[None, :]
        cs, sn = ang.cos(), ang.sin()
        ev, od = x[:, 0::2], x[:, 1::2]
        return torch.stack([ev * cs - od * sn, ev * sn + od * cs], -1).flatten(-2)

    for hp in range(HP):
        for c in range(Gx):
            wk = cta_work(c, pl)
            if wk is None:
                continue
            b, s_idx, t0, t1, last = wk
            seq = seq_l[b]
            kstart = max(0, seq - W + 1) if W else 0
            # ---- Q prep: rows (h, r = j*8 + gi), q head (2hp+h)*8 + gi; FP16 for L=1, E4M3 for L=8
            h_idx = torch.arange(2 * R8) // R8
            r_idx = torch.arange(2 * R8) % R8
            j_idx, gi_idx = r_idx // 8, r_idx % 8
            qh = (2 * hp + h_idx) * 8 + gi_idx
            tok = b * L + j_idx
            col = qh[:, None] * 64 + torch.arange(64)[None, :]
            x = qkv[tok[:, None], col].float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
            if theta:
                x = rope_rows(x, seq + j_idx)
            if L == 1:
                Qs = x.half().float().view(2, R8, 64)
            else:
                Qs = dec(x.clamp(-448.0, 448.0).to(FP8).view(torch.uint8)).view(2, R8, 64)
            # ---- new K/V (last split only)
            newK = torch.zeros((NEWK, 2, 64), dtype=torch.uint8)
            newV = torch.zeros((NEWK, 2, 64), dtype=torch.uint8)
            if last:
                for h in range(2):
                    for j in range(L):
                        t_ = b * L + j
                        kx = qkv[t_, (Hq + 2 * hp + h) * 64:(Hq + 2 * hp + h + 1) * 64].float()[None]
                        kx = kx * torch.rsqrt(kx.pow(2).mean(-1, keepdim=True) + RMS_EPS)
                        if theta:
                            kx = rope_rows(kx, torch.tensor([seq + j]))
                        kb8 = quant_dev(kx[0], ks)
                        vx = qkv[t_, (Hq + Hkv + 2 * hp + h) * 64:(Hq + Hkv + 2 * hp + h + 1) * 64].float()
                        vb8 = quant_dev(vx, vs)
                        row = int(cache_loc[t_])
                        kc[row, (2 * hp + h) * 64:(2 * hp + h + 1) * 64] = kb8
                        vc[row, (2 * hp + h) * 64:(2 * hp + h + 1) * 64] = vb8
                        newK[j, h] = kb8
                        newV[j, h] = vb8
            m = torch.full((NWARP, nr), NEG)
            lsum = torch.zeros((NWARP, nr))
            O = torch.zeros((NWARP, nr, 64))

            def process(Kt, Vt, abs_keys, key_lim, need, warps, keys_of):
                for w in warps:
                    h = roles[w]["h"]
                    ko, kn = keys_of(w)
                    rows = roles[w]["rows"]
                    tokj = rows // 8
                    S = Qs[h, rows] @ Kt[ko:ko + kn, h].T          # [nr, kn] raw dot (exact products, fp32 acc)
                    ak = abs_keys[ko:ko + kn]
                    full_mask = (ak[None, :] < key_lim) & (ak[None, :] <= seq + tokj[:, None])
                    if W:
                        full_mask &= ak[None, :] > seq + tokj[:, None] - W
                    if need:
                        S = torch.where(full_mask, S, torch.tensor(NEG))
                    else:
                        assert bool(full_mask.all()), "need_mask=False but some key is masked"
                    if visits is not None:
                        eff = full_mask if need else torch.ones_like(full_mask)
                        for ri in range(len(rows)):
                            visits.setdefault((b, hp, h, int(rows[ri])), []).extend(ak[eff[ri]].tolist())
                    mx = S.max(dim=1).values
                    m_new = torch.maximum(m[w], c_log2 * mx)
                    m_use = torch.where(m_new == NEG, torch.zeros_like(m_new), m_new)
                    alpha = torch.exp2(m[w] - m_use)
                    p = torch.exp2(S * c_log2 - (m_use - P_OFFSET)[:, None])
                    lsum[w] = lsum[w] * alpha + p.sum(1)
                    if bool((alpha != 1.0).any()):           # device: warp vote, skip the rescale otherwise
                        O[w] = O[w] * alpha[:, None]
                    O[w] = O[w] + p.half().float() @ Vt[ko:ko + kn, h]
                    m[w] = m_new

            for i in range(t0, t1):
                ts = kstart + TILE * i
                abs_k = ts + torch.arange(TILE)
                valid = abs_k < seq
                pidx = torch.where(valid, abs_k >> shift, torch.zeros_like(abs_k))
                prow = pt[b, pidx] * P + (abs_k & (P - 1))
                prow = torch.where(valid, prow, torch.zeros_like(prow))
                cols = slice(hp * 128, hp * 128 + 128)
                Kb = torch.where(valid[:, None], kc[prow, cols], torch.zeros((), dtype=torch.uint8))
                Vb = torch.where(valid[:, None], vc[prow, cols], torch.zeros((), dtype=torch.uint8))
                need = (ts + TILE > seq) or (W > 0 and ts < seq + L - W)
                n_need += int(need)
                n_tiles += 1
                process(dec(Kb).view(TILE, 2, 64), dec(Vb).view(TILE, 2, 64), abs_k, seq, need,
                        range(NWARP), lambda w: (roles[w]["ko"], roles[w]["kn"]))
            if last:
                abs_n = seq + torch.arange(NEWK)
                process(dec(newK).view(NEWK, 2, 64), dec(newV).view(NEWK, 2, 64), abs_n, seq + L, True,
                        [w for w in range(NWARP) if roles[w]["new"]], lambda w: (0, 16))
            # ---- epilogue: merge the warps that share rows (in slot order) -> partial (O/l, lse in log2 units)
            Ost = torch.zeros((ROWS, 64))
            lse = torch.full((ROWS,), NEG)
            groups = {}
            for w in range(NWARP):
                groups.setdefault(roles[w]["grp"], []).append(w)
            for grp, ws in groups.items():
                ws = sorted(ws, key=lambda w: roles[w]["slot"])
                h = roles[ws[0]]["h"]
                rows = roles[ws[0]]["rows"]
                M = torch.stack([m[w] for w in ws]).max(0).values
                M_use = torch.where(M == NEG, torch.zeros_like(M), M)
                lt = torch.zeros(nr)
                Ot = torch.zeros((nr, 64))
                for w in ws:
                    f = torch.exp2(m[w] - M_use)
                    lt = lt + f * lsum[w]
                    Ot = Ot + f[:, None] * O[w]
                ok = lt > 0
                rr = h * R8 + rows
                Ost[rr] = torch.where(ok[:, None], Ot / lt.clamp(min=1e-30)[:, None], 0.0)
                lse[rr] = torch.where(ok, M + torch.log2(lt.clamp(min=1e-30)) - P_OFFSET, torch.tensor(NEG))
            part_O[hp, c] = Ost
            part_lse[hp, c] = lse
            if fold and last:
                # device: the last-arriving CTA of (b, hp); all S_b partials are written by now (emulation order:
                # the last split is also the highest c of the request, so every partial below it exists)
                S = S_list[b]
                pO = part_O[hp, off[b]:off[b] + S]
                pL = part_lse[hp, off[b]:off[b] + S]
                assert not torch.isnan(pO).any() and not torch.isnan(pL).any(), "fold merge read an unwritten partial"
                o4, wsum = merge_fold(pO, pL, S)
                finalize_rows(o4, vs, qkv, out, b, hp, torch.arange(ROWS), L, Hq, Hkv, wsum)

    if not fold:
        for b in range(B):
            for j in range(L):
                for qh in range(Hq):
                    hp, h, gi = qh >> 4, (qh >> 3) & 1, qh & 7
                    row = h * R8 + j * 8 + gi
                    S = S_list[b]
                    pO = part_O[hp, off[b]:off[b] + S]
                    pL = part_lse[hp, off[b]:off[b] + S]
                    assert not torch.isnan(pL[:, row]).any() and not torch.isnan(pO[:, row]).any(), \
                        "combine read an unwritten partial"
                    o4, wsum = merge_sep(pO, pL, S, row, nsg)
                    finalize_rows(o4, vs, qkv, out, b, hp, torch.tensor([row]), L, Hq, Hkv, wsum)
    assert not torch.isnan(out.float()).any(), "some output row was never written"
    if stats is not None:
        stats.update(Gx=Gx, chunk=C, S=S_list, ctas=sum(S_list) * HP, tiles=n_tiles, need=n_need)
    return out.view(B, L, Hq * 64)


def check_coverage(data, visits):
    """Every (request, head pair, KV head, row) attends exactly the reference key set, each key once."""
    cfg = data["config"]
    L, W, Hkv = cfg["q_len"], cfg["window"], cfg["num_kv_heads"]
    n_rows = 0
    for b, seq in enumerate(data["seq_lens"].tolist()):
        for hp in range(Hkv // 2):
            for h in range(2):
                for r in range(8 * L):
                    j = r // 8
                    p = seq + j
                    want = list(range(max(0, p - W + 1) if W else 0, p + 1))
                    got = sorted(visits.get((b, hp, h, r), []))
                    assert got == want, f"coverage b={b} hp={hp} h={h} row={r}: got {len(got)} keys, want {len(want)}"
                    n_rows += 1
    return n_rows


# ------------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------------

CASES = [
    # (name, generate_input kwargs, n_sm for the emulated grid)
    ("ctx10 sliding page64 L1", dict(batch=1, ctx=10, heads="tp8", layer="sliding", seed=11), 132),
    ("ctx2047 sliding edge B3", dict(batch=3, ctx=2047, heads="tp8", layer="sliding", seed=12), 132),
    ("ctx2047 sliding edge B3 L8", dict(batch=3, ctx=2047, heads="tp8", layer="sliding", seed=13, q_len=8, page=1), 132),
    ("sliding L8 mixed page1", dict(batch=3, ctx=3000, heads="tp8", layer="sliding", seed=14, q_len=8, spread="mixed", page=1), 132),
    ("full B4 ctx3000", dict(batch=4, ctx=3000, heads="tp8", layer="full", seed=15), 132),
    ("sliding edges page1", dict(batch=3, ctx=6000, heads="tp8", layer="sliding", seed=16, dist="edges", page=1), 132),
    ("sliding edges page1 L8", dict(batch=2, ctx=5000, heads="tp8", layer="sliding", seed=17, dist="edges", page=1, q_len=8), 132),
    ("full edges L8", dict(batch=3, ctx=6000, heads="tp8", layer="full", seed=18, dist="edges", q_len=8), 132),
    ("sliding L8 kv1/64 page1", dict(batch=3, ctx=4000, heads="tp8", layer="sliding", seed=19, q_len=8, page=1, kv_scale=0.015625), 132),
    ("full kv1/64", dict(batch=2, ctx=9000, heads="tp8", layer="full", seed=20, kv_scale=0.015625), 132),
    ("sliding soft L8 page1", dict(batch=2, ctx=6000, heads="tp8", layer="sliding", seed=21, dist="soft", q_len=8, page=1), 132),
    ("dp full mixed", dict(batch=3, ctx=2500, heads="dp", layer="full", seed=22, spread="mixed"), 132),
    ("dp full L8 soft", dict(batch=2, ctx=3000, heads="dp", layer="full", seed=23, q_len=8, dist="soft"), 132),
    ("full coherent page1 agentic", dict(batch=3, ctx=8000, heads="tp8", layer="full", seed=24, dist="coherent", page=1, spread="agentic"), 132),
    ("full coherent L8 page1 B1", dict(batch=1, ctx=20000, heads="tp8", layer="full", seed=25, dist="coherent", page=1, q_len=8), 132),
    ("sliding soft page1 agentic", dict(batch=4, ctx=6000, heads="tp8", layer="sliding", seed=26, dist="soft", page=1, spread="agentic"), 132),
    # small grids force many-tiles-per-split and S_b = 1 paths
    ("full B4 small grid", dict(batch=4, ctx=3000, heads="tp8", layer="full", seed=27, q_len=8, page=1), 3),
    ("sliding B5 tiny grid", dict(batch=5, ctx=2500, heads="tp8", layer="sliding", seed=28, page=1), 1),
    ("full L8 seq tiny", dict(batch=2, ctx=20, heads="tp8", layer="full", seed=29, page=1, q_len=8), 132),
    ("sliding L8 seq tiny page64", dict(batch=2, ctx=40, heads="tp8", layer="sliding", seed=30, q_len=8), 132),
    ("sliding L8 ctx2100 edges", dict(batch=2, ctx=2100, heads="tp8", layer="sliding", seed=31, q_len=8, dist="edges", page=1), 132),
]


def scored_plan_stats():
    rows = []
    g_ = torch.Generator().manual_seed(7)
    for layer in ("sliding", "full"):
        for B in (1, 16, 32):
            for L in (1, 8):
                if B == 1:
                    seq = [80000]
                else:
                    u = torch.rand((B,), generator=g_)
                    seq = (80000 * (1 + 0.075 * (2 * u - 1))).long().tolist()
                W = 2048 if layer == "sliding" else 0
                for occ in (2, 1):
                    Gx = grid_x(132, B, 1, occ)
                    pl = plan(seq, W, Gx)
                    nt, C, S, off = pl
                    tl = [w[3] - w[2] for w in (cta_work(c, pl) for c in range(Gx)) if w is not None]
                    rows.append((layer, B, L, occ, Gx, C, sum(S), min(S), max(S), max(tl), sum(tl) / len(tl),
                                 sum(nt) / Gx))
    print("scored-shape plans (n_sm=132): layer B | occ Gx C CTAs S_b(min..max) tiles/CTA max, mean, ideal(T/Gx)")
    for r in rows:
        if r[2] != 1:
            continue   # the plan does not depend on L
        print(f"  {r[0]:7s} B={r[1]:2d} | occ={r[3]} Gx={r[4]} C={r[5]} CTAs={r[6]} S_b={r[7]}..{r[8]}"
              f" tiles/CTA max={r[9]} mean={r[10]:.1f} ideal={r[11]:.1f}")


def main():
    t0 = time.time()
    w = fragment_test()
    print(f"[frag] QK/PV fragment reconstruction OK (max rel err {w:.2e})")
    w8 = fragment_test_fp8()
    print(f"[frag] L=8 v1 QK m16n8k32.e4m3 fragment reconstruction OK (max rel err {w8:.2e})")
    nb = bank_test()
    print(f"[frag] {nb} LDS phases conflict-free (K .128 and V .64 with swizzles)")
    newtile_write_test()
    print("[frag] new-tile writer placement == swizzled image")
    plan_test()
    print("[plan] 300 random plans (min_chunk 1/2/4): full tile coverage, one last split per request, sum(S_b) <= Gx")
    scored_plan_stats()
    n_fail = 0
    for name, kw, n_sm in CASES:
        t1 = time.time()
        data = generate_input(**kw)
        kc0, vc0 = data["k_cache"].clone(), data["v_cache"].clone()
        st = {}
        vis = {}
        out = emulate_k1(data, n_sm=n_sm, stats=st, visits=vis, fold=True)
        n_cov = check_coverage(data, vis)
        kc1, vc1 = data["k_cache"], data["v_cache"]
        # the separate-combine path (k1_combine, 8 split groups; and 1 group) must agree with the fold path
        for nsg in (8, 1):
            d2 = dict(data)
            d2["k_cache"], d2["v_cache"] = kc0.clone(), vc0.clone()
            out2 = emulate_k1(d2, n_sm=n_sm, fold=False, nsg=nsg)
            dmax = float((out2.float() - out.float()).abs().max())
            assert dmax <= 2.0 ** -6 * float(out.float().abs().max()), f"fold vs separate(nsg={nsg}) differ by {dmax}"
            assert torch.equal(d2["k_cache"], kc1) and torch.equal(d2["v_cache"], vc1)
        # the reference must see the original cache for past positions; ours only wrote the new rows
        slots = new_slots(data).reshape(-1)
        hkv = data["config"]["num_kv_heads"]
        changed = (kc1.view(-1, hkv * 64) != kc0.view(-1, hkv * 64)).any(-1).nonzero().reshape(-1)
        assert set(changed.tolist()) <= set(slots.tolist()), "emulated kernel wrote outside new slots"
        ok, msg = check_implementation(data, out, kc1, vc1)
        n_fail += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:32s} {msg}  | Gx={st['Gx']} chunk={st['chunk']}"
              f" ctas={st['ctas']} tiles={st['tiles']} masked={st['need']} cov_rows={n_cov} ({time.time() - t1:.1f}s)")
    print(f"done in {time.time() - t0:.1f}s; {len(CASES) - n_fail}/{len(CASES)} cases pass")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

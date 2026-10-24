"""CPU emulation of K2a v2 (k2a_v2.py): repacked layout, routing, unit decomposition, staging, dequant, MMA
fragment wiring, flush, finalize.

Shared with the submission vs independent:
  * prepare()'s repack (prepack_expert, reencode_scales, pack_words, tile_row_perm, BITPOS) is IMPORTED from
    k2a_v2.py, so the bytes emulated here are the bytes the GPU reads.
  * Every device formula is RE-WRITTEN from the .cu text: routing (per-warp ranges, __match_any_sync ranks,
    leader updates), choose_ks, phase_units / first_unit / round-robin unit decode, stage_x piece math into a
    padded smem tile (rows >= ntok left NaN = stale), the per-warp loop index chain (kb0, group, item, scale
    word, parity, smem column; PF 0 and 1) walked literally by check_index_chain (the numeric path indexes
    items by kb directly and the flat uint4 / u32 tile offsets through the repacked tensor shapes),
    deq4 bit ops, __byte_perm + IMAD scale decode, fma.rn.bf16x2 (exactness asserted), the register -> A
    operand wiring, the smem -> B operand addressing, flush (store / atomic, SwiGLU, h / inter / y index math),
    finalize (bf16(y) * nw, RMSNorm, topk weight, sum) and the zero invariants of h and y.
  * m16n8k16 fragment maps come from CuTe's SM80_16x8x16 ALayout/BLayout/CLayout, not from the kernel's (g, t)
    formulas, so a wrong operand / epilogue assumption shows up as an error.
  * inter and y start as NaN (any read of an unwritten element poisons the output); h starts at zero and must be
    zero again after finalize.
Reference: generate_input / ref_kernel copied from problems/muse_moe_w4a16/reference.py (CPU), checks.compare.

Run: /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -I k2a_v2_emulate.py [quick]
Negative controls: K2A2_MUTATE=par|bword|flushrow|rank|leader (each must FAIL).
"""

import importlib.util
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROB = os.path.join(ROOT, "problems", "muse_moe_w4a16")
sys.path.insert(0, PROB)
from checks import compare  # noqa: E402
from nvfp4 import dequantize, random_weights  # noqa: E402

spec = importlib.util.spec_from_file_location("k2a_v2", os.path.join(HERE, "k2a_v2.py"))
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)

torch.set_flush_denormal(False)
torch.set_num_threads(max(1, os.cpu_count() or 1))

D = 4096
N_EXPERTS = 256
POST_NORM_EPS = 1e-8
POPULARITY = 0.03
M32 = 0xFFFFFFFF
MH = 128
MUTATE = os.environ.get("K2A2_MUTATE", "")
NAN = float("nan")


# ---------------------------------------------------------------------------------------------
# Reference (copied from reference.py; weights only for the 32 local experts)
# ---------------------------------------------------------------------------------------------

def generate_input(bs, seed, dist="normal"):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((bs, D), generator=gen).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen)
    if dist == "normal":
        z = torch.randn((N_EXPERTS,), generator=gen)
        u = torch.rand((N_EXPERTS,), generator=gen).clamp(min=1e-12)
        scores = scores + POPULARITY * z / (-u.log()).sqrt()
    if dist == "skewed":
        heat = torch.rand((N_EXPERTS,), generator=gen).pow(6) * 4
        scores = scores + heat
        from nvfp4 import outlier_channels
        x = outlier_channels(x, generator=gen)
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % N_EXPERTS
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": D, "d_expert": D, "n_experts": N_EXPERTS,
                   "local_start": 0, "local_count": 32, "top_k": 8, "sharding": "ep8"},
    }


def post_expert_norm(y, w):
    y = y.float() * (1.0 + w.float())
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)


def ref_kernel(data, weights):
    x = data["hidden_states"].float()
    out = torch.zeros(x.shape, dtype=torch.float32)
    ids = data["topk_ids"].long()
    for e in torch.unique(ids).tolist():
        if not 0 <= e < 32:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :D]) * h[:, D:]
        y = inter @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        out.index_add_(0, tok, post_expert_norm(y, weights["post_norm_w"]) * data["topk_weights"][tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------
# Device primitives (from the CUDA text)
# ---------------------------------------------------------------------------------------------

def rotr(x, s):
    return ((x >> s) | (x << (32 - s))) & M32


def deq4(w):
    return [w & 0x81C081C0,
            rotr(w, 3) & 0x81C081C0,
            rotr(w, 6) & 0x81C081C0,
            (((w << 6) & 0x00C000C0) | ((w << 5) & 0x01000100) | ((w << 11) & 0x80008000)) & M32]


def byte_perm(x, y, sel):
    out = torch.zeros_like(x)
    for n in range(4):
        idx = (sel >> (4 * n)) & 0xF
        assert idx < 8
        src = x if idx < 4 else y
        out |= ((src >> (8 * (idx % 4))) & 0xFF) << (8 * n)
    return out


def scale_reg_sel(sw, sel):
    """__byte_perm(sw, 0, sel) * 16 + 0x75807580 (IMAD, mod 2^32)."""
    return (byte_perm(sw, torch.zeros_like(sw), sel) * 16 + 0x75807580) & M32


def bf16_bits_to_f32(h):
    v = (h & 0xFFFF) << 16
    v = torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
    return v.view(torch.float32)


def f32_to_bf16_bits_exact(f):
    b = f.contiguous().view(torch.int32).to(torch.int64) & M32
    assert bool(((b & 0xFFFF) == 0).all()), "bf16x2 product not exact"
    return b >> 16


def bmul2(a, b):
    lo = f32_to_bf16_bits_exact(bf16_bits_to_f32(a) * bf16_bits_to_f32(b))
    hi = f32_to_bf16_bits_exact(bf16_bits_to_f32(a >> 16) * bf16_bits_to_f32(b >> 16))
    return lo | (hi << 16)


def bf16r(f):
    return f.to(torch.bfloat16).float()


def swiglu(g, u):
    return g / (1.0 + torch.exp(-g)) * u


# CuTe SM80_16x8x16 layouts (independent of the kernel's (g, t) formulas)
def cute_A(lane, v):   # ((4,8),(2,2,2)) : ((32,1),(16,8,128)) -> (m, k) in 16x16 column-major
    t0, t1 = lane % 4, lane // 4
    off = 32 * t0 + t1 + 16 * (v % 2) + 8 * ((v // 2) % 2) + 128 * (v // 4)
    return off % 16, off // 16


def cute_B(lane, v):   # ((4,8),(2,2)) : ((16,1),(8,64)) -> (n, k) in N8 x K16 column-major
    t0, t1 = lane % 4, lane // 4
    off = 16 * t0 + t1 + 8 * (v % 2) + 64 * (v // 2)
    return off % 8, off // 8


def cute_C(lane, v):   # ((4,8),(2,2)) : ((32,1),(16,8)) -> (m, n) in 16x8 column-major
    t0, t1 = lane % 4, lane // 4
    off = 32 * t0 + t1 + 16 * (v % 2) + 8 * (v // 2)
    return off % 16, off // 16


A_MK = torch.tensor([[cute_A(l, v) for v in range(8)] for l in range(32)])    # [32, 8, 2] (m, k)
B_NK = torch.tensor([[cute_B(l, v) for v in range(4)] for l in range(32)])    # [32, 4, 2] (n, k)
C_MN = torch.tensor([[cute_C(l, v) for v in range(4)] for l in range(32)])    # [32, 4, 2] (m, n)
LANE = torch.arange(32)
LG, LT = LANE >> 2, LANE & 3       # kernel's g = lane >> 2, t = lane & 3


# ---------------------------------------------------------------------------------------------
# Config (mirror of k2a2::Cfg)
# ---------------------------------------------------------------------------------------------

class Cfg:
    def __init__(self, nc, u=4, pf=0, tw=1, xe=8192, minb=4):
        self.NC, self.U, self.PF, self.TW, self.XE, self.MINB = nc, u, pf, tw, xe, minb
        self.TPW = 8 * nc
        self.TPP = tw * self.TPW
        self.RW = 8 // tw
        self.KSW = xe // self.TPP
        self.XSTR = self.KSW + 8
        assert self.KSW >= 64 * u and self.KSW % (64 * u) == 0

    def __repr__(self):
        return f"NC{self.NC} U{self.U} PF{self.PF} TW{self.TW} XE{self.XE} MINB{self.MINB}"


# ---------------------------------------------------------------------------------------------
# Routing (route_prologue), literal per warp
# ---------------------------------------------------------------------------------------------

def choose_ks(units1, target):
    ks = 1
    while ks < 16 and units1 * ks < target:
        ks <<= 1
    return ks


def emu_route(ids_flat, lo, M, C, G, ks_target, force_atom):
    NP = 8 * M
    L = ((NP + 255) // 256) * 32
    wc = [[0] * 32 for _ in range(8)]
    ide = [-1] * NP
    for w in range(8):
        for base in range(0, L, 32):
            for ln in range(32):
                p = w * L + base + ln
                if p < NP:
                    v = int(ids_flat[p]) - lo
                    e = v if 0 <= v < 32 else -1
                    ide[p] = e
                    if e >= 0:
                        wc[w][e] += 1
    cnt = [0] * 32
    for e in range(32):
        c = 0
        for w in range(8):
            v = wc[w][e]
            wc[w][e] = c
            c += v
        cnt[e] = c
    off = [0] * 32
    run = 0
    for e in range(32):
        off[e] = run
        run += cnt[e]
    R = run
    np_ = [(c + C.TPP - 1) // C.TPP for c in cnt]
    NV = sum(np_)
    for w in range(8):
        for e in range(32):
            wc[w][e] += off[e]
    s_list = [None] * max(R, 1)
    p2c = [-1] * NP
    for w in range(8):
        for base in range(0, L, 32):
            es = []
            for ln in range(32):
                p = w * L + base + ln
                es.append(ide[p] if p < NP else -1)
            # __match_any_sync groups; rank = lanes below in the same group; leader (rank 0) advances the base
            bases = [wc[w][e] if e >= 0 else 0 for e in es]
            for ln in range(32):
                p = w * L + base + ln
                e = es[ln]
                if e < 0:
                    continue
                m = [l2 for l2 in range(32) if es[l2] == e]
                rank = sum(1 for l2 in m if l2 < ln)
                if MUTATE == "rank":
                    rank = len(m) - 1 - rank
                c = bases[ln] + rank
                assert s_list[c] is None, "two pairs on one compact row"
                s_list[c] = p
                p2c[p] = c
            if MUTATE != "leader":
                for e in set(x for x in es if x >= 0):
                    wc[w][e] += sum(1 for x in es if x == e)
    ks1 = choose_ks(NV * (512 // C.RW), ks_target * G)
    ks2 = choose_ks(NV * (256 // C.RW), ks_target * G)
    hdr = dict(R=R, NV=NV, KS1=ks1, KS2=ks2, atom1=int(ks1 > 1 or force_atom), atom2=int(ks2 > 1 or force_atom))
    return dict(cnt=cnt, off=off, np=np_, s_list=s_list, p2c=p2c, hdr=hdr)


def check_route_vs_sort(rt, ids_flat, lo, M):
    """Compact row order must be: experts ascending, pairs ascending within an expert (a stable sort)."""
    NP = 8 * M
    pairs = [(int(ids_flat[p]) - lo, p) for p in range(NP) if 0 <= int(ids_flat[p]) - lo < 32]
    want = [p for _, p in sorted(pairs)]
    assert rt["s_list"][:len(want)] == want, "deterministic rank != stable sort"


# ---------------------------------------------------------------------------------------------
# Units (phase_units / first_unit / gemm_phase's decode), coverage over all CTAs
# ---------------------------------------------------------------------------------------------

def phase_units(rt, C, mode, KS):
    NRG = (512 if mode == 1 else 256) // C.RW
    ub, cb = [], []
    u, cst = 0, 0
    for e in range(32):
        ub.append(u)
        cb.append(cst)
        n = rt["np"][e] * NRG * KS
        u += n
        tl = min(rt["cnt"][e], C.TPW)
        cst += n * (4 + (tl + 7) // 8)
    ub.append(u)
    cb.append(cst)
    return ub, cb, NRG


def first_unit(ub, cb, x):
    for e in range(32):
        n = ub[e + 1] - ub[e]
        if n == 0:
            continue
        if cb[e + 1] > x:
            ce = (cb[e + 1] - cb[e]) // n
            j = (x - cb[e] + ce - 1) // ce if x > cb[e] else 0
            if j < n:
                return ub[e] + j
    return ub[32]


def decode(rt, ub, KS, u):
    e = 0
    while ub[e + 1] <= u:
        e += 1
    l = u - ub[e]
    npe = rt["np"][e]
    j, r = l % npe, l // npe
    return e, j, r // KS, r % KS


def cta_units(rt, C, mode, KS, G, asg, b):
    ub, cb, _ = phase_units(rt, C, mode, KS)
    if asg:
        beg = first_unit(ub, cb, cb[32] * b // G)
        end = first_unit(ub, cb, cb[32] * (b + 1) // G)
        return list(range(beg, end))
    return list(range(b, ub[32], G))


def check_coverage(rt, C, mode, KS, G, asg):
    ub, cb, NRG = phase_units(rt, C, mode, KS)
    seen = set()
    per_cta = []
    for b in range(G):
        us = cta_units(rt, C, mode, KS, G, asg, b)
        per_cta.append(len(us))
        for u in us:
            assert u not in seen, "unit visited twice"
            seen.add(u)
    assert seen == set(range(ub[32])), "units not covered"
    # every (expert, pass, row group, k-split) exactly once; row tiles / tokens / k covered
    got = set(decode(rt, ub, KS, u) for u in range(ub[32]))
    want = set((e, j, rg, ks) for e in range(32) for j in range(rt["np"][e]) for rg in range(NRG) for ks in range(KS))
    assert got == want and len(got) == ub[32]
    KR = D // KS
    KSWe = min(C.KSW, KR)
    assert KR % KSWe == 0 and KSWe % (64 * C.U) == 0 and (KR // 64) % 2 == 0
    tiles = set()
    for rg in range(NRG):
        for rw in range(C.RW):
            tiles.add(rg * C.RW + rw)
    assert tiles == set(range(512 if mode == 1 else 256))
    for e in range(32):
        for j in range(rt["np"][e]):
            ntok = min(C.TPP, rt["cnt"][e] - j * C.TPP)
            assert ntok >= 1
            cov = []
            for tw in range(C.TW):
                t0w = tw * C.TPW
                nch = max(0, min(C.NC, (ntok - t0w + 7) >> 3))
                cov += [n for q in range(nch) for n in range(t0w + 8 * q, t0w + 8 * q + 8) if n < ntok]
            assert sorted(cov) == list(range(ntok)), "tokens of a pass not covered exactly once"
    return max(per_cta) if per_cta else 0, (sum(per_cta) / G)


# ---------------------------------------------------------------------------------------------
# Operand construction from the repacked bytes / staged smem (through CuTe maps)
# ---------------------------------------------------------------------------------------------

def a_logical(wq_e, sq_e):
    """wq_e int32 [RT, 64, 32, 4], sq_e uint8 [RT, 32, 32, 4] -> A [RT, 16 (m), 4096 (k_log = kb*64+s*16+kk)]:
    what the kernel's registers feed the MMA (consume_item), placed by CuTe's A layout."""
    RT, NKB = wq_e.shape[0], wq_e.shape[1]
    w = wq_e.to(torch.int64) & M32                                         # [RT, 64, 32, 4]
    s8 = sq_e.to(torch.int64)                                              # [RT, 32, 32, 4]
    sw = s8[..., 0] | (s8[..., 1] << 8) | (s8[..., 2] << 16) | (s8[..., 3] << 24)   # [RT, 32 kp, 32 lane]
    kb = torch.arange(NKB)
    sw_it = sw[:, kb >> 1, :]                                              # load_group: st + ((kb>>1)+i)*32
    par = (kb & 1).view(1, NKB, 1)
    if MUTATE == "par":
        par = 1 - par
    Sg = torch.where(par == 0, scale_reg_sel(sw_it, 0x4040), scale_reg_sel(sw_it, 0x4242))
    S8 = torch.where(par == 0, scale_reg_sel(sw_it, 0x4141), scale_reg_sel(sw_it, 0x4343))
    ag = [bmul2(o, Sg) for o in deq4(w[..., 0]) + deq4(w[..., 1])]         # 8 x [RT, 64, 32]
    a8 = [bmul2(o, S8) for o in deq4(w[..., 2]) + deq4(w[..., 3])]
    vals = torch.empty((RT, NKB, 4, 32, 8), dtype=torch.float32)
    for s in range(4):
        regs = [ag[2 * s], a8[2 * s], ag[2 * s + 1], a8[2 * s + 1]]       # a0, a1, a2, a3
        for v in range(8):
            r = regs[v // 2]
            vals[:, :, s, :, v] = bf16_bits_to_f32(r if v % 2 == 0 else r >> 16)
    A = torch.full((RT, NKB, 4, 256), NAN, dtype=torch.float32)
    flat = (A_MK[..., 0] * 16 + A_MK[..., 1]).reshape(-1)                # (lane, v) -> m*16 + kk
    A[:, :, :, flat] = vals.reshape(RT, NKB, 4, 256)
    assert not bool(torch.isnan(A).any())
    return A.view(RT, NKB, 4, 16, 16).permute(0, 3, 1, 2, 4).reshape(RT, 16, NKB * 64)


def b_logical(xs, C, kb0, nkb, ntok):
    """Staged smem tile xs [TPP, XSTR] -> B [nkb*64 (k_log), TPP (column)] as consume_item reads it:
    lane (g, t), chunk q of warp-group tw reads row t0w + 8q + g at kloc + 16t + 4s + v (b0 lo/hi, b1 lo/hi);
    CuTe's B layout says which (n, kk) each register half is."""
    TPP = C.TPP
    B = torch.full((nkb * 64, TPP), NAN, dtype=torch.float32)
    vword = torch.tensor([0, 1, 2, 3])
    if MUTATE == "bword":
        vword = torch.tensor([2, 3, 0, 1])
    for i in range(nkb):
        kloc = i * 64
        for s in range(4):
            for cb in range(0, TPP, 8):                 # cb = t0w + 8q
                rows = cb + LG                           # [32]
                for v in range(4):
                    cols = kloc + 16 * LT + 4 * s + vword[v]
                    vals = xs[rows, cols]
                    n_c, kk_c = B_NK[:, v, 0], B_NK[:, v, 1]
                    B[(i * 64 + s * 16 + kk_c), cb + n_c] = vals
    return B


# ---------------------------------------------------------------------------------------------
# GEMM phase (math per (pass, k-split), vectorized over row tiles; decomposition checked separately)
# ---------------------------------------------------------------------------------------------

def stage_x(C, mode, src_rows, ntok, k0, kw):
    """stage_x: pieces idx -> (n = idx >> sh, pc = idx & (ppr-1)) -> xs[n*XSTR + pc*8 .. +8] = src[k0+pc*8 ..]."""
    xs = torch.full((C.TPP, C.XSTR), NAN, dtype=torch.float32)       # stale smem
    ppr = kw >> 3
    sh = (ppr & -ppr).bit_length() - 1
    assert (1 << sh) == ppr
    tot = ntok * ppr
    idx = torch.arange(tot)
    n, pc = idx >> sh, idx & (ppr - 1)
    assert int(pc.max()) * 8 + 8 <= C.KSW <= C.XSTR - 8
    for jj in range(8):
        xs[n, pc * 8 + jj] = src_rows[n, k0 + pc * 8 + jj]
    return xs


def emu_gemm(mode, C, rt, wq, sq, gsc, X_rows_for, bufs, e_list):
    """Runs every (pass, k-split) of every touched expert. X_rows_for(e, c0, ntok) -> [ntok, 4096] source rows
    (bf16-valued floats), as stage_x reads them."""
    RT = 512 if mode == 1 else 256
    KS = rt["hdr"]["KS1" if mode == 1 else "KS2"]
    atom = rt["hdr"]["atom1" if mode == 1 else "atom2"]
    KR = D // KS
    KSWe = min(C.KSW, KR)
    NSL = KR // KSWe
    for e in e_list:
        A = a_logical(wq[e], sq[e])                                    # [RT, 16, 4096]
        gs = float(gsc[e])
        tile = torch.arange(RT)
        for j in range(rt["np"][e]):
            ntok = min(C.TPP, rt["cnt"][e] - j * C.TPP)
            c0 = rt["off"][e] + j * C.TPP
            src = torch.full((C.TPP, D), NAN)
            src[:ntok] = X_rows_for(e, c0, ntok)
            for ks in range(KS):
                Cacc = torch.zeros((RT, 16, C.TPP), dtype=torch.float32)
                for sl in range(NSL):
                    k0 = ks * KR + sl * KSWe
                    kb0 = k0 >> 6
                    xs = stage_x(C, mode, src, ntok, k0, KSWe)
                    B = b_logical(xs, C, kb0, KSWe // 64, ntok)
                    Cacc += torch.matmul(A[:, :, kb0 * 64:(kb0 + KSWe // 64) * 64], B)
                # flush: per lane (g, t), chunk q of warp-group tw, accumulator i placed by CuTe's C layout
                for tw in range(C.TW):
                    t0w = tw * C.TPW
                    nch = max(0, min(C.NC, (ntok - t0w + 7) >> 3))
                    for q in range(nch):
                        cb = t0w + 8 * q
                        acc = torch.empty((RT, 32, 4))
                        for i in range(4):
                            m_c, n_c = C_MN[:, i, 0], C_MN[:, i, 1]
                            acc[:, :, i] = Cacc[:, m_c, cb + n_c]
                        for hh in range(2):
                            n = cb + 2 * LT + hh                        # [32]
                            ok = n < ntok
                            if not bool(ok.any()):
                                continue
                            v0 = acc[:, :, hh] * gs                     # tile row g
                            v1 = acc[:, :, 2 + hh] * gs                 # tile row g + 8
                            g = LG
                            if MUTATE == "flushrow":
                                g = (LG + 1) % 8
                            cc = (c0 + n)[ok]
                            gg = g[ok]
                            v0, v1 = v0[:, ok], v1[:, ok]
                            if mode == 1:
                                col = 8 * tile.view(-1, 1) + gg.view(1, -1)          # [RT, nl]
                                rows = cc.view(1, -1).expand_as(col)
                                if not atom:
                                    bufs["inter"][rows, col] = bf16r(swiglu(v0, v1))
                                else:
                                    bufs["h"].index_put_((rows, col), v0, accumulate=True)
                                    bufs["h"].index_put_((rows, D + col), v1, accumulate=True)
                            else:
                                col = 16 * tile.view(-1, 1) + gg.view(1, -1)
                                rows = cc.view(1, -1).expand_as(col)
                                if not atom:
                                    bufs["y"][rows, col] = v0
                                    bufs["y"][rows, col + 8] = v1
                                else:
                                    bufs["y"].index_put_((rows, col), v0, accumulate=True)
                                    bufs["y"].index_put_((rows, col + 8), v1, accumulate=True)


def emu_fin(rt, bufs, wts, nw, M):
    out = torch.zeros((M, D), dtype=torch.float32)
    for tok in range(M):
        cs = [rt["p2c"][tok * 8 + s] for s in range(8)]
        acc = torch.zeros(D)
        for s in range(8):
            if cs[s] < 0:
                continue
            f = bf16r(bufs["y"][cs[s]]) * nw
            r = 1.0 / math.sqrt(float((f * f).sum()) / 4096.0 + 1e-8) * float(wts[tok * 8 + s])
            acc += f * r
        out[tok] = acc
    if rt["hdr"]["atom1"]:
        bufs["h"][:rt["hdr"]["R"]] = 0.0
    return out.to(torch.bfloat16)


def emulate(data, st, C, G, ks_target=2, asg=0, force_atom=0, bufs=None):
    x = data["hidden_states"]
    ids = data["topk_ids"].reshape(-1)
    wts = data["topk_weights"].reshape(-1)
    M = x.shape[0]
    rt = emu_route(ids, 0, M, C, G, ks_target, force_atom)
    check_route_vs_sort(rt, ids, 0, M)
    hdr = rt["hdr"]
    info = {}
    for mode, KS in ((1, hdr["KS1"]), (2, hdr["KS2"])):
        for a in (0, 1):
            mx, mean = check_coverage(rt, C, mode, KS, G, a)
            if a == asg:
                info[mode] = (mx, mean)
    assert bool((bufs["h"] == 0).all()), "h zero invariant broken at call start"
    bufs["inter"].fill_(NAN)
    bufs["y"].fill_(NAN)
    if hdr["atom2"]:                                   # G1 phase zeroes y rows [0, R)
        bufs["y"][:hdr["R"]] = 0.0
    e_list = [e for e in range(32) if rt["cnt"][e] > 0]
    xf = x.float()

    def x_rows_g1(e, c0, ntok):
        toks = [rt["s_list"][c0 + n] >> 3 for n in range(ntok)]
        return xf[toks]

    emu_gemm(1, C, rt, st["w13q"], st["s13q"], st["gsc13"], x_rows_g1, bufs, e_list)

    def x_rows_g2(e, c0, ntok):
        if not hdr["atom1"]:
            return bufs["inter"][c0:c0 + ntok]
        hr = bufs["h"][c0:c0 + ntok]
        return bf16r(swiglu(hr[:, :D], hr[:, D:]))

    emu_gemm(2, C, rt, st["w2q"], st["s2q"], st["gsc2"], x_rows_g2, bufs, e_list)
    out = emu_fin(rt, bufs, wts, st["nw"], M)
    assert bool((bufs["h"] == 0).all()), "h not zero after finalize"
    return out, rt, info


# ---------------------------------------------------------------------------------------------
# Primitive unit tests
# ---------------------------------------------------------------------------------------------

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e4m3_value(b):
    s, e, m = (b >> 7) & 1, (b >> 3) & 15, b & 7
    if e == 15 and m == 7:
        return float("nan")
    v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


def test_primitives():
    bits = sorted(b for tup in K.BITPOS for b in tup)
    assert bits == list(range(32))
    for pos in range(8):
        for code in range(16):
            codes = torch.full((1, 8), 5, dtype=torch.uint8)
            codes[0, pos] = code
            o = deq4(K.pack_words(codes).to(torch.int64) & M32)
            vals = []
            for j in range(4):
                vals += [float(bf16_bits_to_f32(o[j])[0]), float(bf16_bits_to_f32(o[j] >> 16)[0])]
            assert vals[pos] * 2.0 ** 126 == float(E2M1[code])
            assert math.copysign(1, vals[pos]) == math.copysign(1, float(E2M1[code]))
    allb = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    bp, neg, zero, _ = K.reencode_scales(allb)
    n_exact = 0
    for b in range(256):
        v = e4m3_value(b)
        if math.isnan(v) or v == 0.0:
            continue
        for bi, sel in enumerate((0x4040, 0x4141, 0x4242, 0x4343)):
            sw = torch.tensor([int(bp[b]) << (8 * bi)], dtype=torch.int64)
            reg = scale_reg_sel(sw, sel)
            assert float(bf16_bits_to_f32(reg)[0]) == float(bf16_bits_to_f32(reg >> 16)[0]) == abs(v) * 2.0 ** 118
        n_exact += 1
    assert n_exact == 252
    for b in range(256):
        v = e4m3_value(b)
        if math.isnan(v) or v == 0.0:
            continue
        reg = scale_reg_sel(torch.tensor([int(bp[b])], dtype=torch.int64), 0x4040)
        for code in range(16):
            o = deq4(K.pack_words(torch.full((1, 8), code, dtype=torch.uint8)).to(torch.int64) & M32)
            p = bmul2(o[0], reg)
            assert float(bf16_bits_to_f32(p)[0]) == float(E2M1[code]) * abs(v) * 2.0 ** -8
    assert sorted((A_MK[..., 0] * 16 + A_MK[..., 1]).reshape(-1).tolist()) == list(range(256))
    assert sorted((B_NK[..., 1] * 8 + B_NK[..., 0]).reshape(-1).tolist()) == list(range(128))
    # the kernel's own (g, t) assumptions, stated: B column = g; C (m, n) = (g + 8*(i>>1), 2t + (i&1))
    for ln in range(32):
        for v in range(4):
            assert cute_B(ln, v)[0] == ln >> 2
            assert cute_C(ln, v) == ((ln >> 2) + 8 * (v >> 1), 2 * (ln & 3) + (v & 1))
    # repack round trip: tile row / k of every 16 B lane load (one small random expert)
    gen = torch.Generator().manual_seed(1)
    w = torch.randint(0, 256, (32, 128), dtype=torch.uint8, generator=gen)
    sf = torch.full((32, 16), 0x38, dtype=torch.uint8)   # 1.0
    wq, sq = K.prepack_expert(w, sf.view(torch.float8_e4m3fn), gate_up=True)
    A = a_logical(wq, sq)                                 # [2, 16, 256] (k_log over 4 kb)
    deq = dequantize(w, sf.view(torch.float8_e4m3fn), torch.tensor(1.0))   # [32, 256]
    perm = K.tile_row_perm(32, True)
    # physical k of logical (kb, s, kk): lane t holds 16t + 4s + {0,1 | 2,3}; kk -> (t, slot) per CuTe
    phys = torch.empty(256, dtype=torch.long)
    for kb in range(4):
        for s in range(4):
            for ln in range(32):
                for v in range(8):
                    m, kk = cute_A(ln, v)
                    slot = (v % 2) + 2 * (v // 4)          # a0/a1 -> +0/+1, a2/a3 -> +2/+3
                    phys[kb * 64 + s * 16 + kk] = kb * 64 + 16 * (ln & 3) + 4 * s + slot
    for tile in range(2):
        for m in range(16):
            want = deq[perm[16 * tile + m]][phys] * 2.0 ** -8
            assert torch.equal(A[tile, m], want), (tile, m)
    print("primitive unit tests: OK (codes x positions, 252 scales x 4 selectors, 4032 exact products, "
          "fragment maps, kernel (g,t) assumptions, repack -> A operand round trip)", flush=True)


def check_index_chain():
    """gemm_phase's per-warp loop, walked literally (no numerics): kb0 = (ks*KR + sl*KSWe) >> 6, load_group(kb0 +
    gi*U) -> wa[i] = item kb + i, sa[j] = scale u32 (kb >> 1) + j; consume item i with sa[i >> 1], parity i & 1, smem
    column kloc = (gi*U + i)*64. PF = 0 (consume, then load next) and PF = 1 (load next, consume, swap). Asserts the
    consumed weight item, scale word, parity and B column agree, and every k64 item of the k-range is consumed once."""
    n = 0
    for U, KSW in ((4, 1024), (4, 512), (4, 256), (2, 1024), (2, 512), (2, 256), (2, 128)):
        for KS in (1, 2, 4, 8, 16):
            KR = 4096 // KS
            KSWe = min(KSW, KR)
            NSL = KR // KSWe
            NGRP = KSWe // (64 * U)
            assert NSL * KSWe == KR and NGRP >= 1 and NGRP * 64 * U == KSWe, (U, KSW, KS)
            for PF in (0, 1):
                tile_items = []
                for ks in range(KS):
                    got = []
                    for sl in range(NSL):
                        k0 = ks * KR + sl * KSWe
                        kb0 = k0 >> 6

                        def load_group(kb):
                            return [kb + i for i in range(U)], [(kb >> 1) + j for j in range(U // 2)]
                        wa, sa = load_group(kb0)
                        for gi in range(NGRP):
                            if PF:
                                wb, sb = load_group(kb0 + (gi + 1) * U) if gi + 1 < NGRP else (None, None)
                            for i in range(U):
                                kb_item = wa[i]
                                kloc = (gi * U + i) * 64
                                assert kb_item == kb0 + kloc // 64, "weight item != B column"
                                assert sa[i >> 1] == kb_item >> 1 and (i & 1) == (kb_item & 1), "scale word / parity"
                                assert kloc + 16 * 3 + 4 * 3 + 3 < KSWe, "smem column out of the slice"
                                got.append(kb_item)
                                n += 1
                            if PF:
                                wa, sa = wb, sb
                            elif gi + 1 < NGRP:
                                wa, sa = load_group(kb0 + (gi + 1) * U)
                    assert got == list(range(ks * KR // 64, (ks + 1) * KR // 64)), (U, KSW, KS, PF, ks)
                    tile_items += got
                assert sorted(tile_items) == list(range(64))
    print(f"index chain walk: OK ({n} item visits over U in {{2,4}}, KSW 128..1024, KS 1..16, PF 0/1)", flush=True)


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def make_weights(seed=5252):
    gen = torch.Generator().manual_seed(seed)
    w13, w13_sf, gs13 = random_weights(2 * D, D, generator=gen, device="cpu", leading=(32,))
    w2, w2_sf, gs2 = random_weights(D, D, generator=gen, device="cpu", leading=(32,))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(),
            "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
            "post_norm_w": (0.1 * torch.randn((D,), generator=gen)).to(torch.bfloat16)}


def prepack_all(weights):
    t0 = time.time()
    st = {"w13q": [], "s13q": [], "w2q": [], "s2q": []}
    for e in range(32):
        a, b = K.prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True)
        st["w13q"].append(a)
        st["s13q"].append(b)
        a, b = K.prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False)
        st["w2q"].append(a)
        st["s2q"].append(b)
    st["gsc13"] = (weights["w13_gs"].float() * K.W_SHIFT)
    st["gsc2"] = (weights["w2_gs"].float() * K.W_SHIFT)
    st["nw"] = (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float()
    print(f"repacked 32 experts in {time.time() - t0:.0f}s", flush=True)
    return st


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    t_all = time.time()
    test_primitives()
    check_index_chain()
    weights = make_weights()
    st = prepack_all(weights)
    bufs = {"h": torch.zeros((8 * K.M_MAX, 2 * D)), "inter": torch.empty((8 * K.M_MAX, D)),
            "y": torch.empty((8 * K.M_MAX, D))}
    c1, c2, c4 = Cfg(1, 4, 0, 1, 8192, 4), Cfg(2, 4, 0, 1, 16384, 4), Cfg(4, 4, 0, 1, 16384, 3)
    c2t2 = Cfg(2, 4, 0, 2, 16384, 4)
    cases = [  # (bs, seed, dist, cfg, G, ks_target, asg, force_atom)
        (1, 401, "normal", c1, 528, 2, 0, 0),
        (4, 3, "normal", c1, 528, 2, 1, 0),
        (8, 404, "normal", c1, 528, 2, 0, 0),
        (16, 402, "normal", c1, 528, 1, 0, 0),
        (33, 5, "normal", c2, 528, 2, 0, 0),
        (33, 5, "normal", c1, 528, 1, 0, 1),
        (128, 7, "skewed", c4, 396, 2, 0, 0),
        (128, 7, "skewed", c2t2, 528, 2, 1, 0),
        (257, 9, "normal", c4, 396, 2, 0, 0),
    ]
    if quick:
        cases = [cases[0], cases[4]]
    if MUTATE:
        cases = [(4, 3, "normal", c1, 528, 2, 0, 0)] if MUTATE not in ("flushrow", "leader") else [(33, 5, "normal", c2, 528, 2, 0, 0)]
    ok_all = True
    for bs, seed, dist, C, G, kst, asg, fa in cases:
        t0 = time.time()
        data = generate_input(bs, seed, dist)
        try:
            out, rt, info = emulate(data, st, C, G, kst, asg, fa, bufs)
            ref = ref_kernel(data, weights)
            ok, msg = compare(out, ref, 1.5e-2)
        except AssertionError as ex:
            ok, msg = False, f"assertion: {ex}"
            rt, info = {"hdr": {}, "cnt": [0]}, {}
        ok_all &= ok
        h = rt["hdr"]
        cnts = sorted([c for c in rt["cnt"] if c > 0], reverse=True)[:6]
        print(f"bs={bs:4d} {dist:6s} [{C!r}] G={G} kst={kst} asg={asg} fa={fa}: {'PASS' if ok else 'FAIL'} {msg} | "
              f"R={h.get('R')} NV={h.get('NV')} KS1={h.get('KS1')} KS2={h.get('KS2')} atom1={h.get('atom1')} "
              f"atom2={h.get('atom2')} top counts {cnts} units/CTA max/mean W13 {info.get(1)} W2 {info.get(2)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(("ALL PASS" if ok_all else "SOME FAILED") + f" in {time.time() - t_all:.0f}s", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

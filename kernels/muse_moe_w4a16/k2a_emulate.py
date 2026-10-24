"""CPU emulation of K2a v1 (k2a_v1.py): exact repacked layout, per-CTA work decomposition, dequant bit ops.

What is shared with the submission and what is independent:
  * prepare()'s repack (prepack_expert, reencode_scales, pack_words, row_perm) is IMPORTED from k2a_v1.py, so
    the bytes emulated here are the bytes the GPU will read.
  * Everything the CUDA kernels do is RE-WRITTEN here from the .cu text: flat uint4/u32 address formulas,
    the deq4 rotate/shift/mask ops, __byte_perm + IMAD scale decode, fma.rn.bf16x2 (exactness asserted per
    element), register-to-fragment wiring, the smem reduction index math and the epilogue index math.
  * The m16n8k16 fragment maps come from CuTe's SM80_16x8x16 ALayout/BLayout/CLayout (cutlass
    include/cute/atom/mma_traits_sm80.hpp), NOT from the (g, t) formulas the kernel uses, so a wrong
    epilogue/fragment assumption in the kernel shows up as an error here.
  * Buffers the kernels write (inter, Y) start as NaN, so any read of an unwritten element poisons the output.
  * The persistent-CTA cursor + RING prefetch loop is simulated literally per CTA for coverage: every
    (active expert, row tile, k-split) exactly once, inactive experts never touched, ring slots consistent.
Reference: ref_kernel / generate_input copied from problems/muse_moe_w4a16/reference.py (CPU), checks.compare.

Run: /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -I k2a_emulate.py
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
from nvfp4 import BLOCK, dequantize, random_weights  # noqa: E402

spec = importlib.util.spec_from_file_location("k2a_v1", os.path.join(HERE, "k2a_v1.py"))
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)

torch.set_flush_denormal(False)
torch.set_num_threads(max(1, os.cpu_count() or 1))

D_HIDDEN = 4096
D_EXPERT = 4096
N_EXPERTS = 256
ROUTED_PER_TOKEN = 8
LOCAL = {"ep8": (0, 32)}
POST_NORM_EPS = 1e-8
REL_L2_TOL = 1.5e-2
POPULARITY = 0.03
M32 = 0xFFFFFFFF
G_SMS = 132          # H200 SM count = persistent grid size
MUTATE = os.environ.get("K2A_MUTATE", "")   # negative controls: scale_lane | frag | kperm | rowperm
KBLK = 128
META_HDR = 128


# ---------------------------------------------------------------------------------------------
# Reference (copied from reference.py; weights only for the 32 local experts)
# ---------------------------------------------------------------------------------------------

def generate_input(bs, sharding, seed, dist="normal", rank="busiest", device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=device)
    if dist == "normal":
        z = torch.randn((N_EXPERTS,), generator=gen, device=device)
        u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + POPULARITY * z / (-u.log()).sqrt()
    if dist == "skewed":
        heat = torch.rand((N_EXPERTS,), generator=gen, device=device).pow(6) * 4
        scores = scores + heat
        from nvfp4 import outlier_channels
        x = outlier_channels(x, generator=gen)
    routed = scores.topk(ROUTED_PER_TOKEN, dim=-1)
    start, count = LOCAL[sharding]
    ids = routed.indices
    if rank == "busiest" and count < N_EXPERTS:
        pairs = torch.bincount((ids // count).flatten(), minlength=N_EXPERTS // count)
        ids = (ids - count * pairs.argmax()) % N_EXPERTS
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": D_HIDDEN, "d_expert": D_EXPERT, "n_experts": N_EXPERTS,
                   "local_start": start, "local_count": count, "top_k": ROUTED_PER_TOKEN, "sharding": sharding},
    }


def post_expert_norm(y, w):
    y = y.float() * (1.0 + w.float())
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)


def ref_kernel(data, weights):
    cfg = data["config"]
    x = data["hidden_states"].float()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    ids = data["topk_ids"].long()
    lo, hi = cfg["local_start"], cfg["local_start"] + cfg["local_count"]
    for e in torch.unique(ids).tolist():
        if not lo <= e < hi:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]
        y = inter @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        out.index_add_(0, tok, post_expert_norm(y, weights["post_norm_w"]) * data["topk_weights"][tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------
# Device primitives, written from the CUDA text
# ---------------------------------------------------------------------------------------------

def rotr(x, s):
    return ((x >> s) | (x << (32 - s))) & M32


def deq4(w):
    """k2a::deq4 on int64 tensors holding u32 values."""
    return [w & 0x81C081C0,
            rotr(w, 3) & 0x81C081C0,
            rotr(w, 6) & 0x81C081C0,
            (((w << 6) & 0x00C000C0) | ((w << 5) & 0x01000100) | ((w << 11) & 0x80008000)) & M32]


def byte_perm(x, y, sel):
    out = torch.zeros_like(x)
    for n in range(4):
        idx = (sel >> (4 * n)) & 0xF
        assert idx < 8, "sign-replicate mode not used"
        src = x if idx < 4 else y
        b = (src >> (8 * (idx % 4))) & 0xFF
        out |= b << (8 * n)
    return out


def scale_reg(sw, bi):
    """__byte_perm(sw, 0, 0x4040 + 0x0101*bi) * 16 + 0x75807580 (IMAD, mod 2^32)."""
    return (byte_perm(sw, torch.zeros_like(sw), 0x4040 + 0x0101 * bi) * 16 + 0x75807580) & M32


def bf16_bits_to_f32(h):
    v = (h & 0xFFFF) << 16
    v = torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
    return v.view(torch.float32)


def f32_to_bf16_bits_exact(f):
    b = f.view(torch.int32).to(torch.int64) & M32
    assert bool(((b & 0xFFFF) == 0).all()), "bf16x2 product not exact"
    return b >> 16


def bmul2(a, b):
    """fma.rn.bf16x2 a, b, -0 with exactness asserted (both halves)."""
    lo = f32_to_bf16_bits_exact(bf16_bits_to_f32(a) * bf16_bits_to_f32(b))
    hi = f32_to_bf16_bits_exact(bf16_bits_to_f32(a >> 16) * bf16_bits_to_f32(b >> 16))
    return lo | (hi << 16)


def bf16x2_to_f32_pair(r):
    return bf16_bits_to_f32(r), bf16_bits_to_f32(r >> 16)


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


A_MK = torch.tensor([[cute_A(l, v) for v in range(8)] for l in range(32)])    # [32, 8, 2]
B_NK = torch.tensor([[cute_B(l, v) for v in range(4)] for l in range(32)])    # [32, 4, 2]
C_MN = torch.tensor([[cute_C(l, v) for v in range(4)] for l in range(32)])    # [32, 4, 2]
A_FLAT = (A_MK[..., 0] * 16 + A_MK[..., 1]).reshape(-1)     # (lane, v) -> m*16 + k
B_FLAT = (B_NK[..., 1] * 8 + B_NK[..., 0]).reshape(-1)      # (lane, v) -> k*8 + n
C_FLAT = (C_MN[..., 0] * 8 + C_MN[..., 1]).reshape(-1)      # (lane, v) -> m*8 + n


# ---------------------------------------------------------------------------------------------
# Unit tests of the primitives
# ---------------------------------------------------------------------------------------------

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e4m3_value(b):
    s, e, m = (b >> 7) & 1, (b >> 3) & 15, b & 7
    if e == 15 and m == 7:
        return float("nan")
    v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


def test_primitives():
    # (1) every code at every element position: pack -> deq4 -> value * 2^126 == E2M1 grid
    for pos in range(8):
        for code in range(16):
            for fill in (0, 5, 15):
                codes = torch.full((1, 8), fill, dtype=torch.uint8)
                codes[0, pos] = code
                w = K.pack_words(codes).to(torch.int64) & M32
                o = deq4(w)
                vals = []
                for j in range(4):
                    lo, hi = bf16x2_to_f32_pair(o[j])
                    vals += [float(lo[0]), float(hi[0])]
                got = vals[pos] * 2.0 ** 126
                want = float(E2M1[code])
                assert got == want and math.copysign(1, got) == math.copysign(1, want), (pos, code, got, want)
                for p2 in range(8):   # other positions decode their own codes
                    assert vals[p2] * 2.0 ** 126 == float(E2M1[int(codes[0, p2])]), (pos, code, p2)
    # all 32 bits used exactly once
    bits = sorted(b for t in K.BITPOS for b in t)
    assert bits == list(range(32)), bits
    # (2) every scale byte: re-encode -> device decode == |sf| * 2^118, flags for sign / zero / nan
    allb = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    bp, neg, zero, _ = K.reencode_scales(allb)
    n_exact = 0
    for b in range(256):
        v = e4m3_value(b)
        if math.isnan(v):
            continue
        if v == 0.0:
            assert bool(zero[b]) and not bool(neg[b]), b
            continue
        assert bool(neg[b]) == (v < 0), b
        sw = torch.tensor([int(bp[b])], dtype=torch.int64)       # byte 0 of the scale word
        reg = scale_reg(sw, 0)
        lo, hi = bf16x2_to_f32_pair(reg)
        assert float(lo[0]) == float(hi[0]) == abs(v) * 2.0 ** 118, (b, float(lo[0]), v)
        n_exact += 1
    assert n_exact == 252, n_exact   # 254 finite - {+0, -0}
    # bf16 products e2m1*sf*2^-8 are exact for every (code, finite nonzero scale)
    for b in range(256):
        v = e4m3_value(b)
        if math.isnan(v) or v == 0.0:
            continue
        reg = scale_reg(torch.tensor([int(bp[b])], dtype=torch.int64), 0)
        for code in range(16):
            codes = torch.full((1, 8), code, dtype=torch.uint8)
            o = deq4(K.pack_words(codes).to(torch.int64) & M32)
            p = bmul2(o[0], reg)   # asserts exactness
            lo, _ = bf16x2_to_f32_pair(p)
            assert float(lo[0]) == float(E2M1[code]) * abs(v) * 2.0 ** -8, (b, code)
    # (3) fragment maps are bijections, and the kernel's epilogue formulas agree with CuTe's C layout
    assert sorted(A_FLAT.tolist()) == list(range(256))
    assert sorted(B_FLAT.tolist()) == list(range(128))
    assert sorted(C_FLAT.tolist()) == list(range(128))
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(4):
            assert cute_C(lane, i) == (g + 8 * (i >> 1), 2 * t + (i & 1)), (lane, i)
    # epilogue inverse: (row rr in 16-tile, token n) -> (lane, i)
    for rr in range(16):
        for n in range(8):
            gg, hi, tt = rr & 7, rr >> 3, (n & 7) >> 1
            i, ln = 2 * hi + (n & 1), gg * 4 + tt
            assert cute_C(ln, i) == (rr, n), (rr, n)
    # (4) MMA from registers == A @ B for random dense matrices
    A = torch.randn(16, 16)
    B = torch.randn(16, 8)
    a_regs = A.reshape(-1)[A_FLAT].view(32, 8)
    b_regs = B.reshape(-1)[B_FLAT].view(32, 4)
    C = emu_mma_from_regs(a_regs.unsqueeze(0), b_regs.unsqueeze(0))[0]
    assert torch.allclose(C, A @ B, atol=1e-5), (C - A @ B).abs().max()
    print("primitive unit tests: OK (16 codes x 8 positions, 252 finite nonzero E4M3 scales, 4032 products exact,"
          " fragment maps, epilogue index map, MMA wiring)")


def emu_mma_from_regs(a_vals, b_vals):
    """a_vals [..., 32, 8] (per lane, CuTe value order), b_vals [..., 32, 4] -> C [..., 16, 8]."""
    lead = a_vals.shape[:-2]
    A = torch.zeros(*lead, 256, dtype=torch.float32)
    A[..., A_FLAT] = a_vals.reshape(*lead, 256)
    Bm = torch.zeros(*lead, 128, dtype=torch.float32)
    Bm[..., B_FLAT] = b_vals.reshape(*lead, 128)
    return A.view(*lead, 16, 16) @ Bm.view(*lead, 16, 8)


# ---------------------------------------------------------------------------------------------
# Kernel emulation
# ---------------------------------------------------------------------------------------------

def emu_route(ids, lo, G, w2upe=128):
    """route_kernel (v1), same algorithm (deterministic ballot order = ascending token). G = #SMs,
    w2upe = W2 units per expert of the W2 variant in use (K.w2_upe)."""
    M = ids.shape[0]
    meta = torch.full((META_HDR + 16 * M,), -777777, dtype=torch.int64)
    loc = ids.to(torch.int64) - lo
    cnt = [int(((loc == e).any(dim=1)).sum()) for e in range(32)]
    pair2row_base = META_HDR + 8 * M
    flat = loc.reshape(-1)
    for p in range(8 * M):
        if not (0 <= int(flat[p]) < 32):
            meta[pair2row_base + p] = -1
    off, acc = [0] * 32, 0
    a = 0
    for e in range(32):
        off[e] = acc
        if cnt[e] > 0:
            meta[8 + a], meta[40 + a], meta[72 + a] = e, cnt[e], acc
            a += 1
        acc += cnt[e]
    n_act = a
    meta[0], meta[1], meta[2] = n_act, (1 if n_act * w2upe >= 3 * G else 2), acc
    for e in range(32):
        if cnt[e] == 0:
            continue
        pos = off[e]
        for t in range(M):
            slot = -1
            for s in range(8):
                if int(loc[t, s]) == e:
                    slot = s
            if slot >= 0:
                meta[META_HDR + pos] = t
                meta[pair2row_base + t * 8 + slot] = pos
                pos += 1
    return meta


def geom(mode, cfg, meta):
    """Decomposition constants exactly as gemm_kernel computes them."""
    NRT = 256 if mode == 1 else 128
    RWG = cfg["nw"] // cfg["kw"]
    NRG = NRT // RWG
    KS = 1 if mode == 1 else int(meta[1])
    KPS = KBLK // KS
    NKB = KPS // cfg["kw"]
    UPE = NRG * KS
    n_act = int(meta[0])
    return dict(NRT=NRT, RWG=RWG, NRG=NRG, KS=KS, KPS=KPS, NKB=NKB, UPE=UPE, U=n_act * UPE, n_act=n_act,
                TCH=8 * cfg["ng"], H2=max(1, cfg["kw"] // 2))


def unit_cost(cnt, TCH):
    return 4 + 3 * ((cnt + TCH - 1) // TCH - 1)


def first_unit(x, n_act, UPE, cnt, TCH):
    base = 0
    for a in range(n_act):
        cost = unit_cost(cnt[a], TCH)
        span = cost * UPE
        if base + span > x:
            j = (x - base + cost - 1) // cost if x > base else 0
            if j < UPE:
                return a * UPE + j
        base += span
    return n_act * UPE


def cta_range(cfg, gm, cnt, b, G):
    """(first unit, step, end) of CTA b, as the kernel computes it."""
    if cfg["asg"] == 1:
        tot = sum(unit_cost(cnt[a], gm["TCH"]) * gm["UPE"] for a in range(gm["n_act"]))
        lo_, hi_ = tot * b // G, tot * (b + 1) // G
        return (first_unit(lo_, gm["n_act"], gm["UPE"], cnt, gm["TCH"]), 1,
                first_unit(hi_, gm["n_act"], gm["UPE"], cnt, gm["TCH"]))
    return b, G, gm["U"]


def simulate_cursor(mode, meta, cfg, G):
    """Literal simulation of the persistent loop: cursors pc (weights, RING ahead), xc (X, XR ahead), cc;
    asserts every ring slot holds the consumed item and every (active expert, row group, k-split) unit is
    visited exactly once. Returns (#units, max units per CTA, mean)."""
    gm = geom(mode, cfg, meta)
    cnt = [int(meta[40 + a]) for a in range(gm["n_act"])]
    RINGc, XR = cfg["ring"], cfg["xr"]

    def unit(u, ue, step):
        q = {"u": u, "c": 0, "kb": 0}
        if u < ue:
            per = gm["NRG"] * gm["KS"]
            q["a"] = u // per
            rem = u - q["a"] * per
            q["rg"], q["ks"] = rem // gm["KS"], rem % gm["KS"]
            q["nch"] = (cnt[q["a"]] + gm["TCH"] - 1) // gm["TCH"]
        return q

    def nxt(q, ue, step):
        q = dict(q)
        q["kb"] += 1
        if q["kb"] == gm["NKB"]:
            q["kb"] = 0
            q["c"] += 1
            if q["c"] == q["nch"]:
                q = unit(q["u"] + step, ue, step)
        return q

    witem = lambda q: (q["a"], q["rg"], q["ks"], q["kb"])
    xitem = lambda q: (q["a"], q["c"], q["ks"], q["kb"])
    visited, per_cta = [], []
    for b in range(G):
        ub, step, ue = cta_range(cfg, gm, cnt, b, G)
        pc = unit(ub, ue, step)
        cc, xc = dict(pc), dict(pc)
        ring = [None] * RINGc
        for d in range(RINGc):
            if pc["u"] < ue:
                ring[d] = witem(pc)
                pc = nxt(pc, ue, step)
        xring = [None] * max(1, XR)
        if XR > 0 and cfg["probe"] == 0:
            for d in range(XR):
                if xc["u"] < ue:
                    xring[d] = xitem(xc)
                    xc = nxt(xc, ue, step)
        n_here = 0
        while cc["u"] < ue:
            for d in range(RINGc):
                if cc["u"] < ue:
                    got = ring[d]
                    if pc["u"] < ue:
                        ring[d] = witem(pc)
                        pc = nxt(pc, ue, step)
                    assert got == witem(cc), ("weight ring mismatch", b, got, witem(cc))
                    if XR > 0 and cfg["probe"] == 0:
                        sl = d % XR
                        gx = xring[sl]
                        if xc["u"] < ue:
                            xring[sl] = xitem(xc)
                            xc = nxt(xc, ue, step)
                        assert gx == xitem(cc), ("X ring mismatch", b, gx, xitem(cc))
                    if cc["kb"] == 0 and cc["c"] == 0:
                        visited.append((cc["a"], cc["rg"], cc["ks"]))
                        n_here += 1
                    cc = nxt(cc, ue, step)
        per_cta.append(n_here)
    want = sorted((a, rg, ks) for a in range(gm["n_act"]) for rg in range(gm["NRG"]) for ks in range(gm["KS"]))
    assert sorted(visited) == want, "unit coverage mismatch"
    return len(visited), max(per_cta), sum(per_cta) / len(per_cta)


def emu_gemm(mode, cfg, wq, sq, gsc, X, meta, M, out):
    """gemm_kernel<MODE, Cfg>, vectorized over (row tile, k-split, kw, k32 item, lane) per active expert.
    Row tile rt = rg*RWG + rw (a bijection), so the per-rt tensors cover every (unit, rw) of the kernel.
    X: bf16 [rows, 4096]; out: MODE 1 float [8M, 4096] holding bf16 values, MODE 2 f32 [KS2_MAX, 8M, 4096]."""
    NG, KW = cfg["ng"], cfg["kw"]
    gm = geom(mode, cfg, meta)
    NRT, KS, KPS, NKB, TCH, H2 = gm["NRT"], gm["KS"], gm["KPS"], gm["NKB"], gm["TCH"], gm["H2"]
    n_act = gm["n_act"]
    R_cap = 8 * M
    wq_u4 = wq.reshape(-1, 4).to(torch.int64) & M32           # flat uint4 array
    sq_u32 = sq.reshape(-1).view(torch.int32).to(torch.int64) & M32
    Xf = X.float()
    lane = torch.arange(32)
    g_l, t_l = lane >> 2, lane & 3
    rt_i = torch.arange(NRT).view(NRT, 1, 1, 1, 1)
    ks_i = torch.arange(KS).view(1, KS, 1, 1, 1)
    kw_i = torch.arange(KW).view(1, 1, KW, 1, 1)
    kb_i = torch.arange(NKB).view(1, 1, 1, NKB, 1)
    ln_i = lane.view(1, 1, 1, 1, 32)
    kbg = ks_i * KPS + kw_i * NKB + kb_i                          # [1, KS, KW, NKB, 1]
    for a in range(n_act):
        e = int(meta[8 + a])
        T = int(meta[40 + a])
        off = int(meta[72 + a])
        blk = (e * NRT + rt_i) * KBLK + kbg                      # [NRT, KS, KW, NKB, 1]
        W = wq_u4[(blk * 32 + ln_i).reshape(-1)].view(NRT, KS, KW, NKB, 32, 4)
        sl = ((t_l & 1) if MUTATE == "scale_lane" else (t_l >> 1)) * 8 + g_l
        SW = sq_u32[(blk * 16 + sl.view(1, 1, 1, 1, 32)).reshape(-1)].view(NRT, KS, KW, NKB, 32)
        S = [scale_reg(SW, bi) for bi in range(4)]
        a_vals = torch.empty(NRT, KS, KW, NKB, 2, 2, 32, 8, dtype=torch.float32)
        for rho in range(2):
            o0 = [bmul2(o, S[2 * rho]) for o in deq4(W[..., 2 * rho])]
            o1 = [bmul2(o, S[2 * rho + 1]) for o in deq4(W[..., 2 * rho + 1])]
            for s in range(2):
                regs = (o0[2 * s], o1[2 * s], o0[2 * s + 1], o1[2 * s + 1])
                if MUTATE == "frag":
                    regs = (o0[2 * s], o0[2 * s + 1], o1[2 * s], o1[2 * s + 1])
                for r, reg in enumerate(regs):
                    lo_v, hi_v = bf16x2_to_f32_pair(reg)
                    a_vals[:, :, :, :, rho, s, :, 2 * r] = lo_v
                    a_vals[:, :, :, :, rho, s, :, 2 * r + 1] = hi_v
        A = torch.zeros(NRT, KS, KW, NKB, 2, 2, 256, dtype=torch.float32)
        A[..., A_FLAT] = a_vals.reshape(NRT, KS, KW, NKB, 2, 2, 256)
        A = A.view(NRT, KS, KW, NKB, 2, 2, 16, 16)
        del a_vals
        nch = (T + TCH - 1) // TCH
        gs = float(gsc[e])
        for c in range(nch):
            n0 = c * TCH
            ng = min(NG, (T - n0 + 7) >> 3)
            xrow = torch.empty(NG, 32, dtype=torch.int64)
            for q in range(NG):
                for l in range(32):
                    n = n0 + q * 8 + (l >> 2)
                    if n >= T:
                        n = T - 1
                    crow = off + n
                    xrow[q, l] = int(meta[META_HDR + crow]) if mode == 1 else crow
            kel = (kbg.view(KS, KW, NKB, 1) * 32 + 8 * t_l.view(1, 1, 1, 32))        # [KS,KW,NKB,32]
            elems = kel.unsqueeze(-1) + torch.arange(8).view(1, 1, 1, 1, 8)          # [KS,KW,NKB,32,8]
            Bm = torch.zeros(KS, KW, NKB, NG, 2, 16, 8, dtype=torch.float32)
            for q in range(ng):
                xv = Xf[xrow[q].view(1, 1, 1, 32, 1), elems]                          # [KS,KW,NKB,32,8]
                for s in range(2):
                    # b0 = s ? xv.z : xv.x  (elements 4s, 4s+1); b1 = s ? xv.w : xv.y (4s+2, 4s+3)
                    bv = torch.stack((xv[..., 4 * s], xv[..., 4 * s + 1], xv[..., 4 * s + 2], xv[..., 4 * s + 3]), -1)
                    if MUTATE == "kperm":
                        bv = torch.stack((xv[..., 2 * s], xv[..., 2 * s + 1], xv[..., 2 * s + 4], xv[..., 2 * s + 5]), -1)
                    tmp = torch.zeros(KS, KW, NKB, 128)
                    tmp[..., B_FLAT] = bv.reshape(KS, KW, NKB, 128)
                    Bm[:, :, :, q, s] = tmp.view(KS, KW, NKB, 16, 8)
            C = torch.einsum("rkwbpsmi,kwbqsin->rkwpqmn", A, Bm)                      # [NRT,KS,KW,2,NG,16,8]
            acc = C.reshape(NRT, KS, KW, 2, NG, 128)[..., C_FLAT].view(NRT, KS, KW, 2, NG, 32, 4)
            crow0 = off + n0
            if KW == 1:
                # per-warp epilogue from registers: n = 8q + 2t + (i&1), rr = g + 8(i>>1)
                for q in range(ng):
                    for l in range(32):
                        gg, tt = l >> 2, l & 3
                        for i in range(4):
                            n = q * 8 + 2 * tt + (i & 1)
                            rr = gg + 8 * (i >> 1)
                            if not (n0 + n < T):
                                continue
                            if mode == 1:
                                hg, hu = acc[:, 0, 0, 0, q, l, i] * gs, acc[:, 0, 0, 1, q, l, i] * gs
                                cols = torch.arange(NRT) * 16 + (rr ^ 1 if MUTATE == "rowperm" else rr)
                                out[crow0 + n, cols] = (hg / (1.0 + torch.exp(-hg)) * hu).to(torch.bfloat16).float()
                            else:
                                for r in range(2):
                                    for ks in range(KS):
                                        out[ks, crow0 + n, torch.arange(NRT) * 32 + r * 16 + rr] = acc[:, ks, 0, r, q, l, i] * gs
                continue
            # smem reduction, literal: flat red[] per CTA unit (rg, ks); writers kw >= H2 store at slab
            # rw*H2 + (kw-H2), adders kw < H2 add at slab rw*H2 + kw, readers sum slabs ro*H2 + k2 in order.
            RWG, NRG = gm["RWG"], gm["NRG"]
            RED_W = 8 * NG
            accg = acc.view(NRG, RWG, KS, KW, 2, NG, 32, 4)
            red = torch.zeros(NRG, KS, (cfg["nw"] // 2) * RED_W * 32, dtype=torch.float32)
            ridx = torch.zeros(2, NG, 4, 32, dtype=torch.int64)
            for r in range(2):
                for q in range(NG):
                    for i in range(4):
                        ridx[r, q, i] = ((r * NG + q) * 4 + i) * 32 + torch.arange(32)
            for rw in range(RWG):
                for kw in range(H2, KW):
                    slab = rw * H2 + (kw - H2)
                    red[:, :, (slab * RED_W * 32 + ridx).reshape(-1)] = accg[:, rw, :, kw].permute(0, 1, 2, 3, 5, 4).reshape(NRG, KS, -1)
            for rw in range(RWG):
                for kw in range(0, H2):
                    slab = rw * H2 + kw
                    ix = (slab * RED_W * 32 + ridx).reshape(-1)
                    red[:, :, ix] = red[:, :, ix] + accg[:, rw, :, kw].permute(0, 1, 2, 3, 5, 4).reshape(NRG, KS, -1)
            rgv = torch.arange(NRG)
            if mode == 1:
                per = 16 * 8 * ng
                for o in range(RWG * per):
                    ro = o // per
                    rem = o - ro * per
                    n, rr = rem >> 4, rem & 15
                    if not (n0 + n < T):
                        continue
                    gg, hi, q, tt = rr & 7, rr >> 3, n >> 3, (n & 7) >> 1
                    i, ln = 2 * hi + (n & 1), gg * 4 + tt
                    gate = torch.zeros(NRG)
                    up = torch.zeros(NRG)
                    for k2 in range(H2):
                        gate = gate + red[:, 0, ((ro * H2 + k2) * RED_W + (0 * NG + q) * 4 + i) * 32 + ln]
                        up = up + red[:, 0, ((ro * H2 + k2) * RED_W + (1 * NG + q) * 4 + i) * 32 + ln]
                    hg, hu = gate * gs, up * gs
                    v = hg / (1.0 + torch.exp(-hg)) * hu
                    cols = (rgv * RWG + ro) * 16 + (rr ^ 1 if MUTATE == "rowperm" else rr)
                    out[crow0 + n, cols] = v.to(torch.bfloat16).float()
            else:
                per = 32 * 8 * ng
                for o in range(RWG * per):
                    ro = o // per
                    rem = o - ro * per
                    n, r = rem >> 5, rem & 31
                    if not (n0 + n < T):
                        continue
                    rho, rr = r >> 4, r & 15
                    gg, hi, q, tt = rr & 7, rr >> 3, n >> 3, (n & 7) >> 1
                    i, ln = 2 * hi + (n & 1), gg * 4 + tt
                    sm = torch.zeros(NRG, KS)
                    for k2 in range(H2):
                        sm = sm + red[:, :, ((ro * H2 + k2) * RED_W + (rho * NG + q) * 4 + i) * 32 + ln]
                    cols = (rgv * RWG + ro) * 32 + r
                    for ks in range(KS):
                        out[ks, crow0 + n, cols] = sm[:, ks] * gs
        del A


def emu_final(Y, meta, ids, wts, nw, M, lo):
    KS = int(meta[1])
    out = torch.zeros(M, 4096, dtype=torch.float32)
    for tok in range(M):
        acc = torch.zeros(4096)
        for s in range(8):
            e = int(ids[tok, s]) - lo
            if e < 0 or e >= 32:
                continue
            crow = int(meta[META_HDR + 8 * M + tok * 8 + s])
            assert crow >= 0
            f = torch.zeros(4096)
            for ks in range(KS):
                f = f + Y[ks, crow]
            yb = f.to(torch.bfloat16).float()
            f = yb * nw
            ss = (f * f).sum()
            r = torch.rsqrt(ss / 4096.0 + 1e-8) * float(wts[tok, s])
            acc = acc + f * r
        out[tok] = acc
    return out.to(torch.bfloat16)


def emulate(data, st, cfgs, sms=G_SMS):
    x = data["hidden_states"]
    ids = data["topk_ids"]
    wts = data["topk_weights"]
    M = x.shape[0]
    lo = data["config"]["local_start"]
    NG = K.ng_for_bs(M)
    c1, c2 = cfgs[(1, NG)], cfgs[(2, NG)]
    meta = emu_route(ids, lo, sms, K.w2_upe(c2))
    n1 = simulate_cursor(1, meta, c1, sms * c1["minb"])
    n2 = simulate_cursor(2, meta, c2, sms * c2["minb"])
    inter = torch.full((8 * M, 4096), float("nan"))
    emu_gemm(1, c1, st["w13q"], st["s13q"], st["gsc13"], x, meta, M, inter)
    Y = torch.full((K.KS2_MAX, 8 * M, 4096), float("nan"))
    emu_gemm(2, c2, st["w2q"], st["s2q"], st["gsc2"], inter.to(torch.bfloat16), meta, M, Y)
    out = emu_final(Y, meta, ids, wts, st["nw"], M, lo)
    info = dict(n_act=int(meta[0]), ks2=int(meta[1]), routes=int(meta[2]), u1=n1, u2=n2, ng=NG)
    return out, info


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def make_weights(seed=5252, wide=False):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w13, w13_sf, gs13 = random_weights(2 * D_EXPERT, D_HIDDEN, generator=gen, device="cpu", leading=(32,), wide_scales=wide)
    w2, w2_sf, gs2 = random_weights(D_HIDDEN, D_EXPERT, generator=gen, device="cpu", leading=(32,), wide_scales=wide)
    return {
        "w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(),
        "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
        "post_norm_w": (0.1 * torch.randn((D_HIDDEN,), generator=gen)).to(torch.bfloat16),
    }


def prepack_all(weights):
    """Exactly prepare()'s repack (functions imported from k2a_v1.py)."""
    st = {
        "w13q": torch.empty((32, 256, 128, 32, 4), dtype=torch.int32),
        "s13q": torch.empty((32, 256, 128, 64), dtype=torch.uint8),
        "w2q": torch.empty((32, 128, 128, 32, 4), dtype=torch.int32),
        "s2q": torch.empty((32, 128, 128, 64), dtype=torch.uint8),
    }
    for e in range(32):
        st["w13q"][e], st["s13q"][e] = K.prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True)
        st["w2q"][e], st["s2q"][e] = K.prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False)
    st["gsc13"] = (weights["w13_gs"].float() * K.W_SHIFT)
    st["gsc2"] = (weights["w2_gs"].float() * K.W_SHIFT)
    st["nw"] = (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float()
    return st


V = None  # set in main


def alt_cfgs():
    """Other decompositions the lab file runs on the GPU: v0-like round robin, KW=1 (no K-split), KW=4, 8 warps."""
    v = K.variant
    return {
        (1, 1): v(1, 1, 4, 0, 0, 0, 16, 16), (2, 1): v(2, 1, 4, 2, 1, 0, 16, 4),
        (1, 2): v(1, 2, 4, 2, 1, 1, 16, 1), (2, 2): v(2, 2, 8, 2, 1, 1, 8, 8),
        (1, 4): v(1, 4, 4, 1, 1, 1, 16, 1), (2, 4): v(2, 4, 4, 0, 0, 0, 16, 16),
    }


def main():
    t0 = time.time()
    test_primitives()
    only = os.environ.get("K2A_BS")
    cases = [(1, "normal", 501), (4, "normal", 502), (8, "normal", 503), (16, "normal", 504), (33, "normal", 505),
             (128, "normal", 506), (128, "skewed", 507), (257, "normal", 508)]
    alt_cases = [(1, "normal", 511), (33, "normal", 512), (128, "skewed", 513)]
    if only:
        keep = {int(b) for b in only.split(",")}
        cases = [c for c in cases if c[0] in keep]
        alt_cases = [c for c in alt_cases if c[0] in keep]
    if os.environ.get("K2A_ALT", "1") != "1":
        alt_cases = []
    weights = make_weights()
    print(f"weights generated ({time.time() - t0:.0f}s)", flush=True)
    st = prepack_all(weights)
    print(f"prepacked 32 experts with k2a_v1.prepack_expert ({time.time() - t0:.0f}s)", flush=True)
    all_ok = True
    for tag, cfgs, cl in (("v1", K.V1_CFG, cases), ("alt", alt_cfgs(), alt_cases)):
        for bs, dist, seed in cl:
            t1 = time.time()
            data = generate_input(bs, "ep8", seed, dist=dist)
            out, info = emulate(data, st, cfgs)
            ref = ref_kernel(data, weights)
            ok, msg = compare(out, ref, REL_L2_TOL)
            all_ok &= ok
            u1, u2 = info["u1"], info["u2"]
            print(f"[{tag}] bs={bs:4d} {dist:7s} {'PASS' if ok else 'FAIL'}  {msg}  | n_act={info['n_act']}"
                  f" routes={info['routes']} KS2={info['ks2']} NG={info['ng']} units W13 {u1[0]} (max/CTA {u1[1]},"
                  f" mean {u1[2]:.2f}) W2 {u2[0]} (max {u2[1]}, mean {u2[2]:.2f}) ({time.time() - t1:.0f}s)", flush=True)
    if os.environ.get("K2A_WIDE", "1") == "1" and not only:
        t1 = time.time()
        ww = make_weights(seed=77, wide=True)
        stw = prepack_all(ww)
        data = generate_input(4, "ep8", 509)
        out, info = emulate(data, stw, K.V1_CFG)
        ok, msg = compare(out, ref_kernel(data, ww), REL_L2_TOL)
        all_ok &= ok
        print(f"[v1] bs=   4 wide-scales {'PASS' if ok else 'FAIL'}  {msg}  | n_act={info['n_act']} ({time.time() - t1:.0f}s)")
    print("ALL PASS" if all_ok else "SOME FAILED", f"({time.time() - t0:.0f}s total)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""CPU emulation of K2b v0 (k2b_v0.py): repack layout, Triton tile program, dequant bit ops, epilogue.

What is shared with the submission and what is independent:
  * The torch-only section of k2b_v0.py (between the K2B PURE markers: repack, row permutation, configs,
    synthetic routing) is EXEC'd from the file, so the bytes emulated here are the bytes the GPU reads.
  * The inline-asm decode is the PTX TEXT extracted from k2b_v0.py and run by a small PTX interpreter
    (mov/prmt/shl/and/or/fma.rn.bf16x2 on uint32 lanes), with Triton's pack=4 operand packing.
  * Everything else the Triton kernel does is RE-WRITTEN here from the kernel text: grouped pid ->
    (m-block, n-tile) mapping, early exits, flat pointer arithmetic for x / wq / sq / out (every address is
    bounds-checked), masked loads, 4 x tl.dot(K=16) per K-step, fp32 accumulation, global scale, the
    reshape(BM, BN/2, 2)+split SwiGLU epilogue and the stores (each output element written at most once).
  * moe_align_block_size is emulated for both contracts (ignore_invalid_expert, and the verbatim call with
    the -1 bucket first); order within an expert is randomised; unused expert_ids are garbage.
  * Kernel-written buffers (inter, y) start as NaN, so a read of an unwritten element poisons the output.
  * Finalize = SGLang's _post_expert_norm_weighted_sum_kernel semantics (BF16 (1+w), fp32 norm, skip -1).
Reference: reference.py's _experts / post_expert_norm (FP32, nvfp4.dequantize), checks.compare.

Run: /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -I k2b_emulate.py [--mutate NAME] [--quick]
"""

import math
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROB = os.path.join(ROOT, "problems", "muse_prefill_layer")
sys.path.insert(0, PROB)
from checks import compare  # noqa: E402
from nvfp4 import dequantize, random_weights  # noqa: E402

SRC_PATH = os.path.join(HERE, "k2b_v0.py")
SRC = open(SRC_PATH).read()
_pure = SRC.split("# ===== K2B PURE BEGIN =====")[1].split("# ===== K2B PURE END =====")[0].split("\n", 1)[1]
K = {"torch": torch}
exec(compile(_pure, "k2b_v0.py:pure", "exec"), K)
_m = re.search(r'asm="""(.*?)"""', SRC, re.S)
ASM = _m.group(1)
_c = re.search(r'constraints="([^"]*)"', SRC)
CONSTRAINTS = _c.group(1)

torch.set_flush_denormal(False)
torch.set_num_threads(max(1, os.cpu_count() or 1))

ARGS = sys.argv[1:]
MUTATE = ARGS[ARGS.index("--mutate") + 1] if "--mutate" in ARGS else ""
QUICK = "--quick" in ARGS
M32 = 0xFFFFFFFF
D = 4096
TOP_K = 8
E_EMU = 8            # experts with real weights in the emulation (ids 8..31 never routed)
POST_NORM_EPS = 1e-8

if MUTATE == "asm_shift":       # sign bit of the low nibble moved to the wrong place
    ASM = ASM.replace("shl.b32 ta, t01, 12;", "shl.b32 ta, t01, 11;", 1)
if MUTATE == "asm_prmt":        # wrong byte selector
    ASM = ASM.replace("0x4342", "0x4243", 1)


# ---------------------------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------------------------

def bf16_bits_to_f64(bits):
    """BF16 bit patterns -> float64 by integer field decoding (no float32 subnormal path, so the result
    does not depend on any thread's flush-to-zero mode; BF16 subnormals such as e2m1 0.5 * 2^-126 are kept)."""
    b = bits.long() & 0xFFFF
    s, ex, m = b >> 15, (b >> 7) & 0xFF, b & 0x7F
    mag = torch.where(ex == 0, torch.ldexp(m.double(), torch.full_like(m, -133)),
                      torch.ldexp((m + 128).double(), ex - 134))
    mag = torch.where(ex == 255, torch.full_like(mag, float("nan")), mag)
    return torch.where(s == 1, -mag, mag)


def f64_to_bf16_bits(v, exact=True):
    """float64 values that are exactly representable in BF16 (asserted) -> BF16 bit patterns, by frexp."""
    assert exact
    v = v.double()
    nan = torch.isnan(v)
    sign = torch.signbit(v).long()
    a = torch.where(nan, torch.zeros_like(v), v.abs())
    mant, ex = torch.frexp(a)                               # a = mant * 2^ex, mant in [0.5, 1)
    be = ex.long() + 126                                    # biased BF16 exponent of a normal value
    mn = (mant * 256.0 - 128.0)                             # 7 mantissa bits if normal
    sub = torch.ldexp(a, torch.full_like(ex, 133))          # a / 2^-133 if subnormal
    normal = be >= 1
    mbits = torch.where(normal, mn, sub)
    assert bool((torch.where(a == 0, torch.zeros_like(mbits), mbits - mbits.round()) == 0).all()), "bf16 not exact"
    assert bool((torch.where(normal, be, torch.zeros_like(be)) < 255).all()), "bf16 overflow"
    bits = torch.where(normal, (be << 7) | mbits.round().long(), mbits.round().long())
    bits = torch.where(a == 0, torch.zeros_like(bits), bits) | (sign << 15)
    return torch.where(nan, torch.full_like(bits, 0x7FC0), bits)


E4M3_TAB = torch.arange(256, dtype=torch.int64).to(torch.uint8).view(torch.float8_e4m3fn).float().double()


def e4m3_manual(b):
    s, e, m = (b >> 7) & 1, (b >> 3) & 15, b & 7
    if e == 15 and m == 7:
        return float("nan")
    v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
    return -v if s else v


def prmt(a, b, sel):
    assert isinstance(sel, int)
    pool = [(a >> (8 * i)) & 0xFF for i in range(4)] + [(b >> (8 * i)) & 0xFF for i in range(4)]
    out = torch.zeros_like(a)
    for i in range(4):
        s = (sel >> (4 * i)) & 0xF
        byte = pool[s & 7]
        if s & 8:
            byte = torch.where((byte & 0x80) != 0, torch.full_like(byte, 0xFF), torch.zeros_like(byte))
        out = out | (byte << (8 * i))
    return out


def fma_bf16x2(a, b, c):
    out = torch.zeros_like(a)
    for lane in range(2):
        sh = 16 * lane
        va, vb, vc = (bf16_bits_to_f64((t >> sh) & 0xFFFF) for t in (a, b, c))
        r = va * vb + vc            # bf16 x bf16 is exact in f64; + (-0) keeps the sign rules of IEEE
        out = out | (f64_to_bf16_bits(r) << sh)
    return out


def run_ptx(asm, ops):
    """Interpret the inline-asm block. ops: {index: int64 tensor of uint32 values} for inputs; returns ops
    with outputs filled. Supports exactly the instructions the kernel uses."""
    regs = {}

    def val(tok):
        tok = tok.strip()
        if tok.startswith("$"):
            return ops[int(tok[1:])]
        if tok.lower().startswith("0x"):
            return int(tok, 16)
        if tok.isdigit():
            return int(tok)
        return regs[tok]

    def put(tok, v):
        if isinstance(v, int):
            v = torch.full_like(next(iter(ops.values())), v)
        v = v & M32
        tok = tok.strip()
        if tok.startswith("$"):
            ops[int(tok[1:])] = v
        else:
            regs[tok] = v

    for stmt in asm.split(";"):
        s = stmt.replace("{", " ").replace("}", " ").strip()
        if not s or s.startswith(".reg"):
            continue
        op, rest = s.split(None, 1)
        a = [t.strip() for t in rest.split(",")]
        if op == "mov.b32":
            put(a[0], val(a[1]))
        elif op == "prmt.b32":
            put(a[0], prmt(val(a[1]), val(a[2]) if not isinstance(val(a[2]), int)
                           else torch.full_like(val(a[1]), val(a[2])), val(a[3])))
        elif op == "shl.b32":
            put(a[0], val(a[1]) << val(a[2]))
        elif op == "and.b32":
            put(a[0], val(a[1]) & val(a[2]))
        elif op == "or.b32":
            put(a[0], val(a[1]) | val(a[2]))
        elif op == "fma.rn.bf16x2":
            put(a[0], fma_bf16x2(val(a[1]), val(a[2]), val(a[3])))
        else:
            raise ValueError(f"unsupported PTX op {op!r}")
    return ops


def asm_decode(wb, S0, S1, perm=None):
    """Triton inline_asm_elementwise(pack=4) on wb uint8-valued [..., 16], S0/S1 int16-bit patterns [..., 16]:
    groups 4 consecutive elements (optionally after a permutation of the last dim, to show the asm is
    elementwise), packs lanes little-endian, runs the PTX, unpacks. Returns (lo, hi) bf16 bit patterns."""
    assert CONSTRAINTS == "=r,=r,=r,=r,r,r,r,r,r"
    if perm is not None:
        wb, S0, S1 = wb[..., perm], S0[..., perm], S1[..., perm]
    shp = wb.shape
    wb4 = wb.reshape(-1, 4).long()
    s04 = S0.reshape(-1, 4).long() & 0xFFFF
    s14 = S1.reshape(-1, 4).long() & 0xFFFF
    ops = {4: wb4[:, 0] | (wb4[:, 1] << 8) | (wb4[:, 2] << 16) | (wb4[:, 3] << 24),
           5: s04[:, 0] | (s04[:, 1] << 16), 6: s04[:, 2] | (s04[:, 3] << 16),
           7: s14[:, 0] | (s14[:, 1] << 16), 8: s14[:, 2] | (s14[:, 3] << 16)}
    ops = run_ptx(ASM, ops)
    lo = torch.stack((ops[0] & 0xFFFF, ops[0] >> 16, ops[1] & 0xFFFF, ops[1] >> 16), dim=-1).reshape(shp)
    hi = torch.stack((ops[2] & 0xFFFF, ops[2] >> 16, ops[3] & 0xFFFF, ops[3] >> 16), dim=-1).reshape(shp)
    if perm is not None:
        inv = torch.argsort(perm)
        lo, hi = lo[..., inv], hi[..., inv]
    return lo, hi


def pure_e2m1_bits(c):
    mag = c & 7
    bits = torch.where(mag >= 2, 0x3F00 + (mag << 6), mag * 0x3F00)
    return (bits | ((c & 8) << 12)) & 0xFFFF


def decode_tile(wb, s0, s1, decode):
    """Mirror of _k2b_decode: wb [..., BN, 16] byte values, s0/s1 [..., BN] float (E4M3 values) ->
    (lo, hi) float64 values of the BF16 weights the kernel feeds to tl.dot."""
    if decode == 0:
        S0 = f64_to_bf16_bits((s0 * 2.0 ** 118).float().double(), exact=True)
        S1 = f64_to_bf16_bits((s1 * 2.0 ** 118).float().double(), exact=True)
        S0 = S0[..., None].expand(*S0.shape, 16)
        S1 = S1[..., None].expand(*S1.shape, 16)
        lo, hi = asm_decode(wb, S0, S1)
        return bf16_bits_to_f64(lo), bf16_bits_to_f64(hi)
    v = wb.long()
    lo = bf16_bits_to_f64(pure_e2m1_bits(v & 15))
    hi = bf16_bits_to_f64(pure_e2m1_bits(v >> 4))
    sb0 = s0.float().to(torch.bfloat16).double()[..., None]
    sb1 = s1.float().to(torch.bfloat16).double()[..., None]
    lo, hi = lo * sb0, hi * sb1
    f64_to_bf16_bits(lo)    # asserts the BF16 product is exact
    f64_to_bf16_bits(hi)
    return lo, hi


# ---------------------------------------------------------------------------------------------
# Unit tests of the primitives
# ---------------------------------------------------------------------------------------------

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_value(c):
    v = E2M1[c & 7]
    return -v if c & 8 else v


def test_primitives():
    # E4M3 table vs a manual decode
    for b in range(256):
        m, t = e4m3_manual(b), float(E4M3_TAB[b])
        assert (math.isnan(m) and math.isnan(t)) or m == t, (b, m, t)
    # every byte x every finite non-negative-or-negative scale, both decoders, exact values
    bytes_ = torch.arange(256, dtype=torch.int64)
    fin = [b for b in range(256) if not math.isnan(e4m3_manual(b))]
    sc = torch.tensor([e4m3_manual(b) for b in fin], dtype=torch.float64)       # 254 scales incl. 0, -0, negatives
    nb, ns = len(bytes_), len(sc)
    wb = bytes_[:, None].expand(nb, ns).reshape(-1, 16)                          # [nb*ns/16, 16]
    s0 = sc[None, :].expand(nb, ns).reshape(-1, 16)
    s1 = sc.flip(0)[None, :].expand(nb, ns).reshape(-1, 16)
    want_lo = torch.tensor([e2m1_value(b & 15) for b in range(256)], dtype=torch.float64)[:, None] * sc[None, :]
    want_hi = torch.tensor([e2m1_value(b >> 4) for b in range(256)], dtype=torch.float64)[:, None] * sc.flip(0)[None, :]
    want_lo, want_hi = want_lo.reshape(-1, 16), want_hi.reshape(-1, 16)
    # asm: per-element scales (rows of 16 elements share nothing here: the asm is elementwise)
    S0 = f64_to_bf16_bits((s0 * 2.0 ** 118).float().double(), exact=True)
    S1 = f64_to_bf16_bits((s1 * 2.0 ** 118).float().double(), exact=True)
    lo, hi = asm_decode(wb, S0, S1)
    glo, ghi = bf16_bits_to_f64(lo) * 2.0 ** 8, bf16_bits_to_f64(hi) * 2.0 ** 8
    bad = ((glo != want_lo) | (ghi != want_hi)).sum()
    assert int(bad) == 0, f"asm decode: {int(bad)} wrong of {glo.numel()}"
    # elementwise property: a random regrouping of the pack gives the same answer
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(3))
    lo2, hi2 = asm_decode(wb, S0, S1, perm=perm)
    assert torch.equal(lo2, lo) and torch.equal(hi2, hi), "asm decode depends on the pack grouping"
    # pure decode (per-element scales via the same broadcast-free path)
    v = wb.long()
    plo = bf16_bits_to_f64(pure_e2m1_bits(v & 15)) * s0.float().to(torch.bfloat16).double()
    phi = bf16_bits_to_f64(pure_e2m1_bits(v >> 4)) * s1.float().to(torch.bfloat16).double()
    assert bool((plo == want_lo).all() and (phi == want_hi).all()), "pure decode"
    f64_to_bf16_bits(plo)
    f64_to_bf16_bits(phi)
    # pure decode: no subnormal BF16 anywhere for nonzero codes
    raw = bf16_bits_to_f64(pure_e2m1_bits(torch.arange(16)))
    assert bool(((raw == 0) | (raw.abs() >= 2.0 ** -126)).all())
    # BF16 subnormal decode: code 1 with the asm bit placement is 2^-127
    assert float(bf16_bits_to_f64(torch.tensor([0x0040]))) == 2.0 ** -127
    assert int(f64_to_bf16_bits(torch.tensor([2.0 ** -127]))) == 0x0040
    assert int(f64_to_bf16_bits(torch.tensor([-0.0]))) == 0x8000
    # prmt sanity
    a = torch.tensor([0x44332211], dtype=torch.int64)
    assert int(prmt(a, torch.zeros_like(a), 0x4140)) == 0x00220011
    assert int(prmt(a, torch.zeros_like(a), 0x4342)) == 0x00440033
    # repack round trip for one random expert through the kernel's flat addressing
    g = torch.Generator().manual_seed(9)
    n, kk = 512, 4096
    w = torch.randint(0, 256, (n, kk // 2), dtype=torch.uint8, generator=g)
    sf = torch.randint(0, 256, (n, kk // 16), dtype=torch.uint8, generator=g)
    for gate_up in (False, True):
        wq, sq = K["k2b_repack_expert"](w, sf.view(torch.float8_e4m3fn), gate_up)
        wq, sq = wq.reshape(-1).long(), sq.reshape(-1).long()
        perm_rows = K["k2b_row_perm"](n, gate_up)
        codes = torch.stack((w & 15, w >> 4), -1).reshape(n, kk).long()
        nn_ = torch.arange(n)
        for ks in (0, 17, 63):
            for h in (0, 1):
                addr = (ks * 2 + h) * (n * 16) + nn_[:, None] * 16 + torch.arange(16)[None, :]
                byte = wq[addr]
                k_lo = ks * 64 + h * 32 + torch.arange(16)
                assert torch.equal(byte & 15, codes[perm_rows][:, k_lo])
                assert torch.equal(byte >> 4, codes[perm_rows][:, k_lo + 16])
                for gg in (2 * h, 2 * h + 1):
                    s = (sq[ks * n + nn_] >> (8 * gg)) & 0xFF
                    assert torch.equal(s, sf.long()[perm_rows][:, ks * 4 + gg])
    print(f"primitive unit tests: OK (asm + pure decode exact for 256 bytes x {ns} E4M3 scales incl. 0/neg,"
          f" pack-grouping independence, prmt, repack round trip)")


# ---------------------------------------------------------------------------------------------
# moe_align_block_size (SGLang contract) and the tile program
# ---------------------------------------------------------------------------------------------

def emu_align(local_ids, bm, ignore, seed):
    """local_ids int [T, 8]: 0..31 local, -1 otherwise. ignore=True: moe_align_block_size(ids, bm, 32,
    ignore_invalid_expert=True); else (ids, bm, 256) with the -1 bucket first."""
    flat = local_ids.reshape(-1).long()
    numel = flat.numel()
    num_experts = 32 if ignore else 256
    if numel < num_experts + 1:
        cap = numel * bm
    else:
        cap = numel + (num_experts + 1) * (bm - 1)
    cap = (cap + 3) // 4 * 4
    n_mb = (cap + bm - 1) // bm
    g = torch.Generator().manual_seed(seed)
    sorted_ids = torch.full((cap,), numel, dtype=torch.int64)
    expert_ids = torch.randint(-5, 40, (n_mb,), generator=g)        # garbage beyond ntp
    buckets = ([] if ignore else [-1]) + list(range(num_experts))
    pos = 0
    for b in buckets:
        idx = torch.nonzero(flat == b).reshape(-1)
        if idx.numel() == 0:
            continue
        idx = idx[torch.randperm(idx.numel(), generator=g)]          # order within an expert is free
        nblk = (idx.numel() + bm - 1) // bm
        sorted_ids[pos: pos + idx.numel()] = idx
        expert_ids[pos // bm: pos // bm + nblk] = b
        pos += nblk * bm
    assert pos <= cap
    return sorted_ids, expert_ids, pos


class Buf:
    """A flat kernel-written buffer starting as NaN, with write-once and bounds checking."""

    def __init__(self, n):
        self.v = torch.full((n,), float("nan"), dtype=torch.float64)
        self.w = torch.zeros((n,), dtype=torch.int8)

    def store(self, addr, val, mask=None):
        addr = addr.reshape(-1)
        val = val.reshape(-1)
        if mask is not None:
            m = mask.reshape(-1)
            addr, val = addr[m], val[m]
        assert int(addr.min()) >= 0 and int(addr.max()) < self.v.numel(), "store out of bounds"
        assert int(self.w[addr].max()) == 0, "element written twice"
        self.w[addr] = 1
        self.v[addr] = val


_DEC_CACHE = {}


def _decoded(wq, sq, e, N, decode):
    """Decode expert e's whole repacked matrix through the kernel's flat addressing (offs_n = 0..N-1):
    w_ptrs + (ks*2+h)*(N*16), s_ptrs + (ks*4+2h)*N and + N. Returns (lo, hi) float64 [KS, 2, N, 16]."""
    key = (id(wq), id(sq), e, N, decode)
    if key in _DEC_CACHE:
        return _DEC_CACHE[key]
    KS = D // 64
    offs_n = torch.arange(N)
    offs_j = torch.arange(16)
    ks_ = torch.arange(KS)
    hh = torch.arange(2)
    w_addr = (e * (KS * 2 * N * 16) + offs_n[None, None, :, None] * 16 + offs_j[None, None, None, :]
              + (ks_[:, None, None, None] * 2 + hh[None, :, None, None]) * (N * 16))          # [KS, 2, N, 16]
    # s_ptrs + ks*N (int32 word), then byte 2h / 2h+1 of it (_k2b_scale)
    s_addr = (e * (KS * N) + offs_n[None, None, :] + ks_[:, None, None] * N).expand(KS, 2, N)
    for a, size in ((w_addr, wq.numel()), (s_addr, sq.numel())):
        assert int(a.min()) >= 0 and int(a.max()) < size, "weight/scale read out of bounds"
    wb = wq[w_addr]
    sw = sq[s_addr]                                     # int32 words (sign-extended in int64)
    g0 = 2 * hh[None, :, None]
    if MUTATE == "sgroup":      # scale of the wrong 16-group (byte h instead of 2h)
        g0 = hh[None, :, None]
    s0 = E4M3_TAB[(sw >> (8 * g0)) & 0xFF]
    s1 = E4M3_TAB[(sw >> (8 * (g0 + 1))) & 0xFF]
    lo, hi = decode_tile(wb, s0, s1, decode)
    blo, bhi = lo.to(torch.bfloat16), hi.to(torch.bfloat16)           # the values are BF16 (asserted exact)
    assert bool((blo.double() == lo).all() and (bhi.double() == hi).all())
    _DEC_CACHE[key] = (blo, bhi)
    return blo, bhi


def emu_gemm(mode, cfg, x_flat, x_rows, wq, sq, gs, out, sorted_ids, expert_ids, ntp, n_pairs, n_rows_w,
             stats):
    """Literal mirror of _k2b_gemm for one launch. x_flat float64 flat (NaN where unwritten), wq/sq flat int64
    byte values, gs float64 [E]."""
    orient, decode, BM, BN, _, _, GM = cfg
    N = n_rows_w
    KS = D // 64
    n_mb = expert_ids.numel()
    NUM_N = N // BN
    grid = n_mb * NUM_N
    seen = torch.zeros((n_mb, NUM_N), dtype=torch.int32)
    offs_j = torch.arange(16)
    ks_ = torch.arange(KS)
    for pid in range(grid):
        per_group = GM * NUM_N
        group = pid // per_group
        first_m = group * GM
        gsize = min(n_mb - first_m, GM)
        pid_m = first_m + (pid % per_group) % gsize
        pid_n = (pid % per_group) // gsize
        seen[pid_m, pid_n] += 1
        if pid_m * BM >= ntp:
            continue
        e = int(expert_ids[pid_m])
        if e < 0 or e >= 32:
            continue
        assert e < E_EMU, "routed to an expert outside the emulated bank"
        stats["tiles"] += 1
        offs_m = pid_m * BM + torch.arange(BM)
        inb = offs_m < sorted_ids.numel()                     # masked load, other = n_pairs
        pair = torch.where(inb, sorted_ids[torch.where(inb, offs_m, torch.zeros_like(offs_m))],
                           torch.full_like(offs_m, n_pairs))
        valid = (pair >= 0) & (pair < n_pairs)
        xrow = torch.where(valid, pair // TOP_K, torch.zeros_like(pair)) if mode == 1 else offs_m
        offs_n = pid_n * BN + torch.arange(BN)
        assert int(offs_n.max()) < N
        # The kernel's w/s addresses are affine in offs_n, so the decode of all N rows (cached per expert)
        # sliced at offs_n is exactly what this tile reads; addresses are bounds-checked in _decoded().
        lo_all, hi_all = _decoded(wq, sq, e, N, decode)
        lo, hi = lo_all[:, :, offs_n].double(), hi_all[:, :, offs_n].double()
        # x_ptrs + (ks*64 + h*32) and + 16, masked by valid
        hh = torch.arange(2)
        xa_addr = (xrow[None, None, :, None] * D + offs_j[None, None, None, :]
                   + (ks_[:, None, None, None] * 64 + hh[None, :, None, None] * 32))          # [KS, 2, BM, 16]
        xb_addr = xa_addr + 16
        if MUTATE == "xcol":    # activation columns of the wrong K half (h*16 instead of h*32)
            xa_addr = xa_addr - hh[None, :, None, None] * 16
            xb_addr = xa_addr + 16
        vm = valid[None, None, :, None].expand_as(xa_addr)
        for a in (xa_addr, xb_addr):
            av = a[vm]
            if av.numel():
                assert int(av.min()) >= 0 and int(av.max()) < x_flat.numel(), "x read out of bounds"
        xa = torch.where(vm, x_flat[torch.where(vm, xa_addr, torch.zeros_like(xa_addr))], torch.zeros(()).double())
        xb = torch.where(vm, x_flat[torch.where(vm, xb_addr, torch.zeros_like(xb_addr))], torch.zeros(()).double())
        # tl.dot(xa, trans(lo)) + tl.dot(xb, trans(hi)) for every (ks, h), fp32 accumulation
        A = torch.cat((xa, xb), dim=-1).permute(2, 0, 1, 3).reshape(BM, -1).float()             # [BM, K]
        Bw = torch.cat((lo, hi), dim=-1).permute(2, 0, 1, 3).reshape(BN, -1).float()            # [BN, K]
        if orient == 0:
            acc = A @ Bw.T                                                                     # [BM, BN]
        else:
            acc = (Bw @ A.T).T                                                                 # trans([BN, BM])
        gsv = float(torch.tensor(float(gs[e]), dtype=torch.float32) * (256.0 if decode == 0 else 1.0))
        acc = acc * torch.tensor(gsv, dtype=torch.float32)
        if mode == 1:
            pairs3 = acc.reshape(BM, BN // 2, 2)                    # tl.reshape(acc, (BM, BN//2, 2)) + split
            gate, up = pairs3[..., 0], pairs3[..., 1]
            v = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).double()
            offs_c = pid_n * (BN // 2) + torch.arange(BN // 2)
            out.store(offs_m[:, None] * (N // 2) + offs_c[None, :], v)
        else:
            out.store(pair[:, None] * N + offs_n[None, :], acc.to(torch.bfloat16).double(),
                      mask=valid[:, None].expand(BM, BN))
    assert bool((seen == 1).all()), "grid does not cover every (m-block, n-tile) exactly once"


def emu_finalize(y_flat, wts, local_ids, post_w):
    """SGLang _post_expert_norm_weighted_sum_kernel (APPLY_NORM, HAS_WEIGHT, WEIGHT_FIRST, HAS_IDS)."""
    T = wts.shape[0]
    w = (post_w.float() + 1.0).to(torch.bfloat16).float()          # MuseSpark1xFp4Experts' effective weight
    acc = torch.zeros((T, D), dtype=torch.float32)
    y = y_flat.reshape(T * TOP_K, D)
    for kk in range(TOP_K):
        live = local_ids[:, kk] >= 0
        rows = torch.arange(T) * TOP_K + kk
        x = torch.where(live[:, None], y[rows].float(), torch.zeros(()))
        yy = x * w
        var = (yy * yy).sum(-1, keepdim=True) / D
        yy = yy * torch.rsqrt(var + POST_NORM_EPS)
        acc += torch.where(live[:, None], yy * wts[:, kk, None].float(), torch.zeros(()))
    return acc.to(torch.bfloat16)


def emulate(x, ids, wts, weights, wq13, sq13, wq2, sq2, cfg, ignore, seed):
    T = x.shape[0]
    local = torch.where((ids >= 0) & (ids < 32), ids, torch.full_like(ids, -1)).long()
    bm = cfg[2]
    sorted_ids, expert_ids, ntp = emu_align(local, bm, ignore, seed)
    n_mb = expert_ids.numel()
    n_pairs = T * TOP_K
    stats = {"tiles": 0}
    inter = Buf(n_mb * bm * D)
    x_flat = x.double().reshape(-1)
    emu_gemm(1, cfg, x_flat, T, wq13, sq13, weights["w13_gs"].double(), inter, sorted_ids, expert_ids, ntp,
             n_pairs, 8192, stats)
    y = Buf(n_pairs * D)
    emu_gemm(2, cfg, inter.v, n_mb * bm, wq2, sq2, weights["w2_gs"].double(), y, sorted_ids, expert_ids, ntp,
             n_pairs, 4096, stats)
    out = emu_finalize(y.v, wts, local, weights["post_expert_norm"])
    return out, {"ntp": ntp, "n_mb": n_mb, "tiles": stats["tiles"]}


# ---------------------------------------------------------------------------------------------
# Reference (reference.py semantics)
# ---------------------------------------------------------------------------------------------

def post_expert_norm(y, w):
    y = y.float() * (1.0 + w.float())
    return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)


def ref_experts(x, ids, topk_weights, weights):
    out = torch.zeros(x.shape, dtype=torch.float32)
    for e in torch.unique(ids).tolist():
        if not 0 <= e < 32:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok].float() @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :D]) * h[:, D:]
        y = inter @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T
        out.index_add_(0, tok, post_expert_norm(y, weights["post_expert_norm"]) * topk_weights[tok, slot, None])
    return out


def make_weights(seed, wide=False):
    gen = torch.Generator().manual_seed(seed)
    w13, w13_sf, gs13 = random_weights(2 * D, D, generator=gen, device="cpu", leading=(E_EMU,), wide_scales=wide)
    w2, w2_sf, gs2 = random_weights(D, D, generator=gen, device="cpu", leading=(E_EMU,), wide_scales=wide)
    gs13 = gs13 * torch.exp(torch.randn((E_EMU,), generator=gen))
    gs2 = gs2 * torch.exp(torch.randn((E_EMU,), generator=gen))
    pen = (0.1 * torch.randn((D,), generator=gen)).to(torch.bfloat16)
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13, "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2,
            "post_expert_norm": pen}


def repack_all(weights):
    if MUTATE == "lohi":   # swap which nibble holds which K half (repack bug)
        orig = K["k2b_repack_expert"]

        def bad(w, sf, gate_up):
            wq, sq = orig(w, sf, gate_up)
            return ((wq & 15) << 4) | (wq >> 4), sq
        rep = bad
    elif MUTATE == "gateup":  # repack without the gate/up interleave the epilogue assumes
        orig = K["k2b_repack_expert"]

        def rep(w, sf, gate_up):
            return orig(w, sf, False)
    else:
        rep = K["k2b_repack_expert"]
    out = {}
    for name, gu in (("13", True), ("2", False)):
        wqs, sqs = [], []
        for e in range(E_EMU):
            wq, sq = rep(weights["w" + name][e], weights["w" + name + "_sf"][e], gu)
            wqs.append(wq)
            sqs.append(sq)
        out["wq" + name] = torch.stack(wqs).reshape(-1).long()
        out["sq" + name] = torch.stack(sqs).reshape(-1).long()
    return out


def routing(T, seed, zipf):
    """Busiest-GPU-like routing restricted to the E_EMU emulated local experts (others never chosen)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.ones(256, dtype=torch.float64)
    z = 1.0 / torch.arange(1, E_EMU + 1, dtype=torch.float64) ** zipf
    w[:E_EMU] = z / z.sum() * 0.29 * 224.0
    w[E_EMU:32] = 0.0
    u = torch.rand((T, 256), generator=g, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
    logits = w.log()[None, :] - torch.log(-torch.log(u))
    ids = logits.topk(TOP_K, dim=-1).indices.to(torch.int32)
    s = torch.rand((T, TOP_K), generator=g) + 0.05
    return ids, (s / s.sum(-1, keepdim=True)).float()


def main():
    t0 = time.time()
    if MUTATE in ("", "asm_shift", "asm_prmt"):
        try:
            test_primitives()
        except AssertionError as ex:
            print(f"primitive unit tests: FAIL ({ex})")
            if not MUTATE:
                raise
    results = []
    cases = [("normal", 160, 21, 1.0, False), ("skewed", 200, 22, 2.5, False), ("wide-scales", 96, 23, 1.0, True)]
    if QUICK or MUTATE:
        cases = cases[:1]
    cfgs = list(K["K2B_CONFIGS"].items())
    if MUTATE:
        cfgs = [c for c in cfgs if c[0] in ("ss_asm_128x256", "ss_pure_128x256")]
        if MUTATE in ("asm_shift", "asm_prmt"):
            cfgs = cfgs[:1]
    all_ok = True
    for cname, T, seed, zipf, wide in cases:
        _DEC_CACHE.clear()
        weights = make_weights(5000 + seed, wide=wide)
        packed = repack_all(weights)
        gx = torch.Generator().manual_seed(seed)
        x = torch.randn((T, D), generator=gx).to(torch.bfloat16)
        ids, wts = routing(T, seed, zipf)
        ref = ref_experts(x, ids.long(), wts, weights)
        n_local = int(((ids >= 0) & (ids < 32)).sum())
        per_e = torch.bincount(ids[(ids >= 0) & (ids < 32)].long(), minlength=E_EMU).tolist()
        for i, (name, cfg) in enumerate(cfgs):
            for ignore in (True, False):
                if (QUICK or MUTATE) and not ignore:
                    continue
                t1 = time.time()
                out, info = emulate(x, ids, wts, weights, packed["wq13"], packed["sq13"], packed["wq2"],
                                    packed["sq2"], cfg, ignore, seed * 7 + i)
                ok, msg = compare(out.float(), ref, 1.5e-2)
                all_ok &= ok
                line = (f"{cname:11s} T={T:3d} routes={n_local:4d} per-expert={per_e} {name:16s}"
                        f" align={'ignore ' if ignore else 'verbatim'} {'PASS' if ok else 'FAIL'} {msg}"
                        f" | ntp={info['ntp']} n_mb={info['n_mb']} tiles={info['tiles']} ({time.time() - t1:.1f}s)")
                print(line, flush=True)
                results.append(line)
    print(f"{'ALL PASS' if all_ok else 'SOME FAILED'} ({time.time() - t0:.0f}s){' [mutation ' + MUTATE + ']' if MUTATE else ''}")
    return all_ok


if __name__ == "__main__":
    main()

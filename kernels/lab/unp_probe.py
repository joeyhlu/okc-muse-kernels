"""UNP probe: what the RS-wgmma consumer's NVFP4 -> BF16 unpack costs per 36 KB stage, and cheaper exact unpacks.

Built on kernels/lab/vml_probe2.py (same kernel, plan, harness), base = the shipped VML v13 consumer (VM_TRICKS with
VM_TRICKS_DQ 0 = probe switches SC2 + DS + CL). Compile-time switches added here (text patches of vml_probe2.BODY):
  PG_UQ   unpack variant inside dec_half2 (every other instruction of the stage is unchanged):
          0 shipped deq4 (BITPOS3: 5 LOP3 + 3 SHF + 1 IMAD.SHL per 8 codes, + 4 HMUL2)
          1 FM: the same bits, every shift on the FMA pipe (IMAD.SHL x2, IMAD.WIDE rotl 12 -> LOP3 (lo | hi) & M,
            IMAD.HI for >> 6): ALU 5 / FMA 4 per 8 codes (bitwise identical to 0)
          2 L2a: new in-word bit layout (one fixed 32-bit permutation of the BITPOS3 word, lossless repack, l2_perm):
            o0 = w & M, o1 = rotl(w, 10) & M, o2 = rotl(w, 13) & M, o3 = ((w & 0x001B001B) * 0x1084) & M (carry-free
            three-way bit gather by one IMAD): ALU 7 (5 LOP3 + 2 SHF) + FMA 1 per 8 codes; reads l2_repack(ws13)
          3 L2b: L2 with both rotations as hi | lo of an IMAD pair: ALU 5 LOP3 + FMA 5; reads l2_repack(ws13)
          8 NS: deq4 without the scale multiply (timing only)
          9 Z1: one LOP3 per output register, no shift, no scale (timing only: floor of any register-writing unpack)
         10 Z2: one LOP3 + the scale HMUL2 per output register (timing only: floor of any exact mask + scale unpack)
  PG_NOMMA 1: no wgmma / fence / commit / wait (unpack only; outputs sunk by an empty asm with register inputs)
Arms run in F (stream) and C (resident, PG_RES 1) modes; T = resident tensor only (PG_RES 2).
"""
import vml_probe2 as P2

L2_RM, L2_K, L2_R1, L2_R2 = 0x001B001B, 0x1084, 10, 13

_DEQ = r"""
#ifndef PG_UQ
#define PG_UQ 0
#endif
#ifndef PG_NOMMA
#define PG_NOMMA 0
#endif
__device__ __forceinline__ uint32_t ok32(uint32_t c) {   // c, opaque to NVVM (ptxas still sees a MOV)
  uint32_t r;
  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));
  return r;
}
__device__ __forceinline__ void deq_u(uint32_t w, uint32_t* o, uint32_t k64, uint32_t k4) {
  constexpr uint32_t M = 0x81C081C0u;
#if PG_UQ == 1
  const uint32_t k6 = ok32(64u), k12 = ok32(4096u), k26 = ok32(1u << 26), k2 = ok32(4u);   // independent opaque constants
  o[0] = w & M;
  o[1] = (w * k6) & M;                                 // rotl 6 == shl 6 under M
  o[2] = (__umulhi(w, k12) | (w * k12)) & M;           // rotl 12 = hi | lo of w * 2^12 (IMAD.WIDE or 2 IMAD)
  o[3] = (__umulhi(w, k26) & 0x01000100u) | ((w * k2) & 0x80C080C0u);
#elif PG_UQ == 2
  const uint32_t kk = ok32(0x1084u);
  o[0] = w & M;
  o[1] = __funnelshift_l(w, w, 10) & M;
  o[2] = __funnelshift_l(w, w, 13) & M;
  o[3] = ((w & 0x001B001Bu) * kk) & M;
#elif PG_UQ == 3
  const uint32_t kk = ok32(0x1084u), k10 = ok32(1024u), k13 = ok32(8192u);
  o[0] = w & M;
  o[1] = (__umulhi(w, k10) | (w * k10)) & M;
  o[2] = (__umulhi(w, k13) | (w * k13)) & M;
  o[3] = ((w & 0x001B001Bu) * kk) & M;
#elif PG_UQ == 9 || PG_UQ == 10
  o[0] = w & M; o[1] = w & (M >> 1); o[2] = w & (M >> 2); o[3] = w & (M >> 3);
#else
  deq4(w, o);
#endif
}
constexpr bool UQ_SCALE = !(PG_UQ == 8 || PG_UQ == 9);
"""

_OLD_DQ = """  if constexpr (PG_DQ) { deq4_dq(w0, hh, k64, k4); deq4_dq(w8, hh + 4, k64, k4); }
  else { deq4(w0, hh); deq4(w8, hh + 4); }
  if constexpr (PG_SC2) {"""
_NEW_DQ = """  deq_u(w0, hh, k64, k4); deq_u(w8, hh + 4, k64, k4);
  if constexpr (!UQ_SCALE) {
  } else if constexpr (PG_SC2) {"""

_OLD_MMA = """    wg_fence();
#pragma unroll
    for (int h2 = 0; h2 < GS / 2; ++h2) {
      const uint32_t kk = (uint32_t)(gi * GS + 2 * h2);   // k16 step (even); SW128: k64 block kk >> 2, 32 B per step inside
      const uint32_t off = PG_XSW ? (kk >> 2) * (uint32_t)N * 8u + (kk & 3u) * 2u : 16u * kk;
      const uint32_t st2 = PG_XSW ? 2u : 16u;
      if (first && gi == 0 && h2 == 0) mma2<N, true>(d, hh + 8 * h2, dlo + off, dhi, st2);
      else mma2<N, false>(d, hh + 8 * h2, dlo + off, dhi, st2);
    }
    wg_commit();
    wg_wait<NPD>();
"""
_NEW_MMA = """    if constexpr (PG_NOMMA) {
#pragma unroll
      for (int k = 0; k < 4 * GS; ++k) asm volatile("" :: "r"(hh[k]));
    } else {
""" + _OLD_MMA + """    }
"""


def _patch(src, old, new):
    assert src.count(old) == 1, old[:60]
    return src.replace(old, new)


BODY = P2.BODY
BODY = _patch(BODY, "__device__ __forceinline__ void dec_half2(", _DEQ + "__device__ __forceinline__ void dec_half2(")
BODY = _patch(BODY, _OLD_DQ, _NEW_DQ)
BODY = _patch(BODY, _OLD_MMA, _NEW_MMA)
CUDA = P2.HEAD + BODY
CPP = P2.CPP
_S = {"PG_SC2": 1, "PG_DS": 1, "PG_CL": 1}   # shipped VML v13 consumer (VM_TRICKS 1, VM_TRICKS_DQ 0)
ARMS = {"S": dict(_S), "FM": dict(_S, PG_UQ=1), "L2a": dict(_S, PG_UQ=2), "L2b": dict(_S, PG_UQ=3),
        "NS": dict(_S, PG_UQ=8), "Z1": dict(_S, PG_UQ=9), "Z2": dict(_S, PG_UQ=10), "U": dict(_S, PG_NOMMA=1),
        "Sc": dict(_S, PG_RES=1), "FMc": dict(_S, PG_UQ=1, PG_RES=1), "L2ac": dict(_S, PG_UQ=2, PG_RES=1),
        "Z1c": dict(_S, PG_UQ=9, PG_RES=1), "Z2c": dict(_S, PG_UQ=10, PG_RES=1), "Uc": dict(_S, PG_NOMMA=1, PG_RES=1),
        "T": {"PG_RES": 2}}
EXACT = ("S", "FM", "L2a", "L2b")         # streaming arms whose outputs must match S bit for bit
NOCHECK = ("PG_RES", "PG_XR", "PG_NOMMA", "TIMING")
L2_ARMS = ("L2a", "L2b", "L2ac")               # read the l2_repack weights (L2a / L2b streaming arms are checked)
for _a in ("NS", "Z1", "Z2"):
    ARMS[_a]["TIMING"] = 1               # wrong results by design (codes not repacked / no scale / no shifts)
to_sw128 = P2.to_sw128


def flags_of(arm):
    return P2.V1.FLAGS + [f"-D{k}={v}" for k, v in ARMS[arm].items() if k != "TIMING"]


def build(arm, verbose=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    fl = flags_of(arm)
    tag = hashlib.sha1((CPP + CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"unp_probe_{arm}_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["vp2_run", "vp2_attrs"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


def l2_perm():
    """new-word bit s <- BITPOS3 word bit perm[s] (the L2 layout carries every code bit to where L2 decodes it)."""
    Pm = [6, 7, 8, 15, 22, 23, 24, 31]
    old = {}   # (output j, target p) -> BITPOS3 source bit
    for p in Pm:
        old[(0, p)] = p
        old[(1, p)] = (p - 6) % 32
        old[(2, p)] = (p - 12) % 32
        old[(3, p)] = p + 6 if p in (8, 24) else p - 2
    new = {}
    for p in Pm:
        new[(0, p)] = p
        new[(1, p)] = (p - L2_R1) % 32
        new[(2, p)] = (p - L2_R2) % 32
        src = [p - k for k in (2, 7, 12) if p - k >= 0 and (L2_RM >> (p - k)) & 1]
        assert len(src) == 1
        new[(3, p)] = src[0]
    perm = [None] * 32
    for key, s in new.items():
        perm[s] = old[key]
    assert sorted(perm) == list(range(32))
    return perm


def _deq4_py(w):
    M = 0x81C081C0
    rot = lambda x, r: ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF  # noqa: E731
    return [w & M, rot(w, 6) & M, rot(w, 12) & M, ((w >> 6) & 0x01000100) | ((w << 2) & 0x80C080C0)]


def _l2_py(w):
    M = 0x81C081C0
    rot = lambda x, r: ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF  # noqa: E731
    return [w & M, rot(w, L2_R1) & M, rot(w, L2_R2) & M, (((w & L2_RM) * L2_K) & 0xFFFFFFFF) & M]


def l2_selftest(n=200000, seed=1):
    """L2 decode of the permuted word == BITPOS3 deq4 of the original word, bit for bit (random words + all one-hots)."""
    import random
    perm = l2_perm()
    rng = random.Random(seed)
    words = [1 << b for b in range(32)] + [0, 0xFFFFFFFF] + [rng.getrandbits(32) for _ in range(n)]
    for w in words:
        nw = 0
        for s in range(32):
            nw |= ((w >> perm[s]) & 1) << s
        assert _l2_py(nw) == _deq4_py(w), hex(w)
    return len(words)


def l2_repack(ws, perm=None):
    """K2A stage tensor [..., 18432] uint8 (16 KB codes = int32 words, then 2 KB scales) -> the same with every code word
    bit-permuted into the L2 layout (new bit s <- old bit perm[s]); scales untouched. Lossless, per leading slice."""
    import torch
    perm = perm or l2_perm()
    out = ws.clone()
    flat = out.view(-1, ws.shape[-1])
    for i in range(0, flat.shape[0], 4096):
        c = flat[i:i + 4096, :16384].contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        n = torch.zeros_like(c)
        for s in range(32):
            n |= ((c >> perm[s]) & 1) << s
        n = torch.where(n >= 2 ** 31, n - 2 ** 32, n).to(torch.int32)
        flat[i:i + 4096, :16384] = n.view(torch.uint8).view(-1, 16384)
    return out

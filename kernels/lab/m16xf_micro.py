#!/usr/bin/env python3
"""F6: CUDA-core (FFMA, no tensor instruction) consumer for 1-2 token experts on the F3 consumer-pace micro.

Every arm is one compile-time MODE of the F3 kernel template m16xb<8, MODE, 3> (imported verbatim from
kernels/lab/m16xb_micro.py, pinned by sha256): the same ring / mbarrier / bulk-copy protocol, the same 36,864-B
weight stream, the same helper-warp epilogue (out = 256 * d).  There is ONE kernel-text edit: a constexpr-guarded
xprep<N, MODE>(s, tid) between the two start-up __syncthreads.  It is MODE >= 11 only, so null/base/twin instantiate
the F3 text token for token (their SASS is pinned to the F5b gate's).  All arms use the N=8 template shape (16 KB
packed x, 64x8 epilogue).  The ffma arms read token 0 (tokens 0-1) of the same packed x that base uses.

  MODE 0   null            ring only (clock reference)
  MODE 3   base            shipped ms consumer (F3 'ms', byte-identical consume body), N8
  MODE 4   twin            A/A copy of base (separate instance of consume<8, 3>)
  MODE 11  ffma1           1 token: shipped dec_half2 (BF16 weight/256, exact) -> widen to FP32 -> FFMA
  MODE 12  ffma1_cvtfree   1 token: FP32 weights straight from the bit placement (16 ops / 8 weights, no HMUL2,
                           no widening); the 16-block scale is folded into the FP32 accumulate:
                           blk = sum_16 w*x', acc += s * blk
  MODE 13  ffma2           2 tokens, as ffma1
  MODE 14  ffma2_cvtfree   2 tokens, as ffma1_cvtfree

Lane map (the shipped mma.sync A fragment, dec_half2 + ms_pair).  Warp wp, lane = 4g + t:
  - Per dec_half2 call q (0..31, k16 steps 2q and 2q+1) the lane holds 16 weights:
    rows 16wp+g (code word w0) and 16wp+g+8 (w8), 8 k per row = 8 weights per k16 step.
  - Code i of a word sits at A-domain k = 16(2q + i/4) + 2t + 8((i/2)&1) + (i&1) (pair j = i/2, low/high half i&1),
    which is original column kperm(k).
  - x needed: x[tau, kperm(k)]. These are the same 8 values for both rows and for all 8 lanes with this t.
    xprep stores them once per CTA as FP32 x'[q][tau][t][8] (2 LDS.128 per q per token).
  - Scale: a lane's 16 weights of one row in one kb (calls 2kb, 2kb+1) share ONE E4M3 scale
    (byte 2*sub + row-half of scale word kb/2), because kperm(k)/16 = 4(k/64) + t.
  - Reduction: the 4 lanes t = 0..3 share rows g, g+8 -> butterfly shfl_xor 1, 2 per (row, token) at unit end.
    Every lane holds the full sums, so the F3 epilogue map (d[j] -> row g+8(j>>1), column 2t+(j&1)) writes
    out[u][r][c] = y[u][r][c % T] in all 8 columns (checked against the FP32 oracle).

CPU:    python -B kernels/lab/m16xf_micro.py --dry-run                     (numpy only, writes nothing)
Gates:  bash tools/local/f6_gates.sh   (WSL; CPU + compile + SASS claims; never launches CUDA work)
Pod:    python3 -u -B kernels/lab/m16xf_micro.py --out box_out/f6/results.json   (via tools/box/f6_batch.sh)
"""
import argparse
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m16xb_micro as mb  # noqa: E402  (F3 micro; its __main__ guard keeps it inert)

NS = 3
N = 8
UNITS = 48
ARM_MODE = {"null": 0, "base": 3, "twin": 4, "ffma1": 11, "ffma1_cvtfree": 12, "ffma2": 13, "ffma2_cvtfree": 14}
MODE_ARM = {v: k for k, v in ARM_MODE.items()}
ARMS = ("null", "base", "twin", "ffma1", "ffma1_cvtfree", "ffma2", "ffma2_cvtfree")
# FFMA arms: tokens T, formulation ('dq' = shipped dec_half2 + widen, 'cf' = cvtfree), x' exponent K (x' = x * 2^K).
SPEC = {"ffma1": (1, "dq", 0), "ffma1_cvtfree": (1, "cf", 43), "ffma2": (2, "dq", 0), "ffma2_cvtfree": (2, "cf", 43)}
FFMA_ARMS = tuple(SPEC)
CONSUME_MODES = (3, 11, 12, 13, 14)     # twin (4) maps to consume<N, 3> inside the F3 kernel; null never consumes
XB = N * 2048                           # x' byte offset in the resident x region [0, XS = 65536)
# Bank rotation: bank = (BANK_OFF[arm index] + round) % 3 (the F3/F5b (2r + idx) % 3 form repeats a bank at a round
# boundary with 7 arms); CPU gate: consecutive replays always change bank, every arm sees each bank 4x in 12 rounds.
BANK_OFF = (0, 1, 0, 1, 0, 2, 1)
# The imported F3 sources this file builds on (same pins as m16xe_micro.py / F5b).
PINS = {"CUDA_HEAD": "72d8c8df15153ea041509cfd59b362d5f0b96c958971f8495b213acca0cf8a84",
        "DEQUANT": "68d9eb97f2b8070ae4e0f2b1576ca51bf6d274051c80f62e2e1ad8747b4d7116",
        "CPP": "fe848d62c0be42ea76204dbaa03dbd6882ad969ef91269b8cbefb5aabd1cdca8"}
# Pre-registered (lead brief F6): PASS if ffma1 <= 1.15 us/unit AND MHz >= 1900 with correct outputs; KILL if
# ffma1 > 1.22; ffma2 PASS-2 if <= 1.18; contrast VOID unless |twin - base| <= 0.03 us/unit and <= 15 MHz.
# Applied to each ffma1-class arm (ffma1 = shipped dequant, ffma1_cvtfree = the cheaper formulation); the ffma1
# decision is PASS if either passes, KILL only if both are > 1.22.  PASS-2 likewise over ffma2 / ffma2_cvtfree.
PASS_US, PASS_MHZ, KILL_US, PASS2_US, AA_US, AA_MHZ = 1.15, 1900.0, 1.22, 1.18, 0.03, 15.0

# The one kernel-text edit (an insertion; the consume() signature and call are F3's).
FENCE = '  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");\n'
XPREP_LINE = "  if constexpr (MODE >= 11) xprep<N, MODE>(s, tid);  // F6: FP32 x' for the FFMA consumers (MODE 11..14)\n"

# wx8: code i of a packed word -> FP32 bits with magnitude at 22..24 and sign at 31 (the high-half BF16 positions of
# deq4), value code * 2^-126 exactly (code 1 -> the FP32 denormal 2^-127).  BITPOS3 (mag0, mag1, mag2, sign):
#   0 (6,7,8,15) shl 16 | 1 (22,23,24,31) as is | 2 (28,29,30,5) rotl 26 | 3 (12,13,14,21) shl 10
#   4 (25,26,27,2) rotl 29 | 5 (9,10,11,18) shl 13 | 6, 7: p = w & 0x1B001B (codes 6 and 7 only), p * 0x1084 is
#   carry-free (deq4's o[3]): code 7 already at 22..24/31, code 6 at 6..8/15, i.e. p * 0x10840000 for code 6.
WX8 = (("shl", 16), ("id", 0), ("rot", 26), ("shl", 10), ("rot", 29), ("shl", 13), ("mul", 0x10840000), ("mul", 0x1084))
WMASK = 0x81C00000
SC_BIAS = 0x60000000   # exponent 192: (bp << 20) | bias = E4M3 scale * 2^75 (bp >> 3 <= 18 never reaches bit 29)


def wx8_cuda():
    expr = {"shl": lambda a: f"(w << {a})", "id": lambda a: "w", "rot": lambda a: f"__funnelshift_l(w, w, {a})",
            "mul": lambda a: f"(p * unp_ok32(0x{a:x}u))"}
    body = [f"  v[{i}] = __uint_as_float({expr[k](a)} & M);" for i, (k, a) in enumerate(WX8)]
    return "\n".join(["__device__ __forceinline__ void wx8(uint32_t w, float* v) {",
                      "  // The 8 e2m1 codes of one packed code word as exact FP32 code * 2^-126 (no scale: it is applied per",
                      "  // 16-block).  16 ops / 8 weights: 8 masks, 5 shifts/rotates, 1 prefilter + 2 IMAD (codes 6, 7).",
                      f"  constexpr uint32_t M = 0x{WMASK:X}u;",
                      "  const uint32_t p = w & 0x001B001Bu;"] + body + ["}"])


F6_HELPERS = r"""
// ---- F6 FFMA consumer helpers (MODE 11..14 only; null/base/twin compile the F3 code unchanged) ----
template<int MODE> constexpr int F6T = (MODE == 13 || MODE == 14) ? 2 : 1;   // tokens per expert
template<int MODE> constexpr int F6K = (MODE == 12 || MODE == 14) ? 43 : 0;  // x' = x * 2^K (cvtfree arms)
template<int N, int MODE> __device__ __forceinline__ void xprep(uint8_t* s, int tid) {
  // Once per CTA inside the kernel, before the ring starts: FP32 x'[q][tau][t][i] = x * 2^K at s + N * 2048, the 8
  // B values a lane with t = lane & 3 needs for dec_half2 call q (code order i, pair j = i / 2: k16 step 2q + j / 2,
  // k = 2t + 8(j & 1) + (i & 1)), read from the resident packed BF16 B tiles (one u32 = the x of codes 2j, 2j + 1).
  constexpr int T = F6T<MODE>, K = F6K<MODE>;
  if (tid < 256) {
    #pragma unroll
    for (int r = 0; r < 2 * T; ++r) {
      const int e = tid + 256 * r, j = e & 3, t = (e >> 2) & 3, tau = (e >> 4) % T, q = (e >> 4) / T;
      const uint32_t v = reinterpret_cast<const uint32_t*>(s)[((2 * q + (j >> 1)) * N * 16 + (tau >> 3) * 128 + (j & 1) * 64 + (tau & 7) * 8 + 2 * t) >> 1];
      float lo = __uint_as_float(v << 16), hi = __uint_as_float(v & 0xffff0000u);
      if constexpr (K != 0) {
        lo = __fmul_rn(lo, __uint_as_float(uint32_t(127 + K) << 23));
        hi = __fmul_rn(hi, __uint_as_float(uint32_t(127 + K) << 23));
      }
      reinterpret_cast<float2*>(s + N * 2048)[((q * T + tau) * 4 + t) * 4 + j] = make_float2(lo, hi);
    }
  }
}
@WX8@
template<int B> __device__ __forceinline__ float sc32(uint32_t s) {
  // Re-encoded scale byte B as FP32 (bp << 20) + 0x60000000 = E4M3 scale * 2^75 (shipped BF16: (bp << 4) + 0x7580 =
  // scale * 2^118).  With w = code * 2^-126 and x' = x * 2^43: s * w * x' = scale * code * x / 256.  The shipped
  // PRMT + LEA idiom (one immediate each; a mask + OR form needs a 2nd constant that ptxas re-materializes per use).
  return __uint_as_float((__byte_perm(s, 0u, 0x4440u | B) << 20) + 0x60000000u);
}
template<int T, int SUB, bool FIRST> __device__ __forceinline__ void cf_kb(float (&acc)[2][T], const uint4 c, uint32_t scale, const float4* xq) {
  // cvtfree, one kb (calls q = 2kb, 2kb + 1): row g = words c.x, c.y; row g + 8 = c.z, c.w.  Each row's 16 weights
  // share one scale (byte 2 SUB + row half): blk = sum_i w_i x'_i (FP32), acc += s * blk.
  // xq[tau * 8 + h] / xq[(T + tau) * 8 + h]: this lane's x' for q = 2kb / 2kb + 1.
  float wg[16], w8[16];
  wx8(c.x, wg); wx8(c.y, wg + 8); wx8(c.z, w8); wx8(c.w, w8 + 8);
  const float sg = sc32<2 * SUB>(scale), s8 = sc32<2 * SUB + 1>(scale);
  #pragma unroll
  for (int tau = 0; tau < T; ++tau) {
    const float4 a = xq[tau * 8], b = xq[tau * 8 + 1], e = xq[(T + tau) * 8], f = xq[(T + tau) * 8 + 1];
    const float xv[16] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, e.x, e.y, e.z, e.w, f.x, f.y, f.z, f.w};
    float bg = __fmul_rn(wg[0], xv[0]), b8 = __fmul_rn(w8[0], xv[0]);
    #pragma unroll
    for (int i = 1; i < 16; ++i) { bg = __fmaf_rn(wg[i], xv[i], bg); b8 = __fmaf_rn(w8[i], xv[i], b8); }
    if constexpr (FIRST) { acc[0][tau] = __fmul_rn(sg, bg); acc[1][tau] = __fmul_rn(s8, b8); }
    else { acc[0][tau] = __fmaf_rn(sg, bg, acc[0][tau]); acc[1][tau] = __fmaf_rn(s8, b8, acc[1][tau]); }
  }
}
template<int T, bool FIRST> __device__ __forceinline__ void dq_q(float (&p)[2][T][4], const uint32_t* h, const float4* xq) {
  // ffma1 / ffma2, one dec_half2 call: h = the shipped BF16 pairs (weight / 256, exact; h[0..3] row g, h[4..7]
  // row g + 8), widened to FP32 (low half = code 2j, high half = code 2j + 1); partial accumulator j per (row, tau).
  float w[2][8];
  #pragma unroll
  for (int r = 0; r < 2; ++r)
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      w[r][2 * j] = __uint_as_float(h[4 * r + j] << 16);
      w[r][2 * j + 1] = __uint_as_float(h[4 * r + j] & 0xffff0000u);
    }
  #pragma unroll
  for (int tau = 0; tau < T; ++tau) {
    const float4 a = xq[tau * 8], b = xq[tau * 8 + 1];
    const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    #pragma unroll
    for (int r = 0; r < 2; ++r)
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        if constexpr (FIRST) p[r][tau][j] = __fmul_rn(w[r][2 * j], xv[2 * j]);
        else p[r][tau][j] = __fmaf_rn(w[r][2 * j], xv[2 * j], p[r][tau][j]);
        p[r][tau][j] = __fmaf_rn(w[r][2 * j + 1], xv[2 * j + 1], p[r][tau][j]);
      }
  }
}
__device__ __forceinline__ float rsum4(float v) {
  // Sum over the 4 lanes (t = lane & 3) that share a row; bitwise identical in all 4 lanes: (a+b)+(c+d) up to commutation.
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
template<int T> __device__ __forceinline__ void f6_out(float* d, const float (&acc)[2][T]) {
  // F3 epilogue map at N = 8: d[0] -> (row g, col 2t), d[1] -> (g, 2t + 1), d[2] -> (g + 8, 2t), d[3] -> (g + 8, 2t + 1),
  // so out[u][r][c] = y[u][r][c % T] in all 8 columns.
  d[0] = rsum4(acc[0][0]); d[2] = rsum4(acc[1][0]);
  if constexpr (T == 2) { d[1] = rsum4(acc[0][1]); d[3] = rsum4(acc[1][1]); } else { d[1] = d[0]; d[3] = d[2]; }
}
template<int T> __device__ __forceinline__ void f6_out_dq(float* d, const float (&p)[2][T][4]) {
  float acc[2][T];
  #pragma unroll
  for (int r = 0; r < 2; ++r)
    #pragma unroll
    for (int tau = 0; tau < T; ++tau)
      acc[r][tau] = __fadd_rn(__fadd_rn(p[r][tau][0], p[r][tau][1]), __fadd_rn(p[r][tau][2], p[r][tau][3]));
  f6_out<T>(d, acc);
}
""".replace("@WX8@", wx8_cuda())

XPREP_INDEX = "((2 * q + (j >> 1)) * N * 16 + (tau >> 3) * 128 + (j & 1) * 64 + (tau & 7) * 8 + 2 * t) >> 1"
XPREP_STORE = "((q * T + tau) * 4 + t) * 4 + j"


def consume_body(mode):
    """Straight-line body of one consume MODE (one 64x1024 unit per warp). MODE 3 is F3's ms body, byte for byte."""
    assert mode in CONSUME_MODES
    if mode == 3:
        lines = ["uint32_t h[8];"]
        for sg in range(8):
            lines += ["{", f"const uint32_t scale = reinterpret_cast<const uint32_t*>(st + 32768 + wp * 1024)[{sg} * 32 + lane];"]
            for sub in range(2):
                kb = 2 * sg + sub
                lines += ["{", f"const uint4 c = reinterpret_cast<const uint4*>(st + wp * 8192)[{kb} * 32 + lane];"]
                for hf in range(2):
                    q = 2 * kb + hf
                    lines.append(f"dec_half2(c.{('x', 'y')[hf]}, c.{('z', 'w')[hf]}, scale, {sub}u, h, 16u, 64u, 4u);")
                    lines.append(f"ms_pair<N, {q}>(d, h, x, lane);")
                lines.append("}")
            lines.append("}")
        return lines
    T, form, _ = SPEC[MODE_ARM[mode]]
    lines = ["const float4* xf = reinterpret_cast<const float4*>(reinterpret_cast<const uint8_t*>(x) + N * 2048) + (lane & 3) * 2;"]
    lines += ["uint32_t h[8];", f"float p[2][{T}][4];"] if form == "dq" else [f"float acc[2][{T}];"]
    for sg in range(8):
        lines += ["{", f"const uint32_t scale = reinterpret_cast<const uint32_t*>(st + 32768 + wp * 1024)[{sg} * 32 + lane];"]
        for sub in range(2):
            kb = 2 * sg + sub
            lines += ["{", f"const uint4 c = reinterpret_cast<const uint4*>(st + wp * 8192)[{kb} * 32 + lane];"]
            if form == "dq":
                for hf in range(2):
                    q = 2 * kb + hf
                    lines.append(f"dec_half2(c.{('x', 'y')[hf]}, c.{('z', 'w')[hf]}, scale, {sub}u, h, 16u, 64u, 4u);")
                    lines.append(f"dq_q<{T}, {'true' if q == 0 else 'false'}>(p, h, xf + {q * T * 8});")
            else:
                lines.append(f"cf_kb<{T}, {sub}, {'true' if kb == 0 else 'false'}>(acc, c, scale, xf + {kb * 2 * T * 8});")
            lines.append("}")
        lines.append("}")
    lines.append(f"f6_out_dq<{T}>(d, p);" if form == "dq" else f"f6_out<{T}>(d, acc);")
    return lines


def generate_cuda():
    """One binary: 7 instances m16xb<8, MODE, 3>; MODE resolves at instantiation, never in the stage loop."""
    head = mb.CUDA_HEAD
    assert head.count(FENCE) == 1, "F3 anchor moved"
    head = head.replace(FENCE, XPREP_LINE + FENCE)
    pieces = []
    for i, mode in enumerate(CONSUME_MODES):
        pieces.append(f"{'if' if i == 0 else 'else if'} constexpr (MODE == {mode}) {{")
        pieces += consume_body(mode)
        pieces.append("}")
    info, dispatch = [], []
    for arm in ARMS:
        mode = ARM_MODE[arm]
        cond = f"if (mode == {mode} && n == {N} && ns == {NS})"
        info.append(f"{cond} return config<{N}, {mode}, {NS}>(true);")
        dispatch.append(f"{cond} {{ m16xb<{N}, {mode}, {NS}><<<grid,384,SMEM<{N}, {NS}>,st>>>((const uint8_t*)w,(const uint16_t*)x,(float*)o,(int*)sm,(int*)err,(uint64_t*)marks,units); ck(cudaGetLastError()); return; }}")
    src = head.replace("@CONSUME@", "\n".join(pieces))
    src = src.replace("@INFO@", "\n".join(info)).replace("@DISPATCH@", "\n".join(dispatch))
    anchor = "template<int N, int Q> __device__"
    assert src.count(anchor) == 2  # rs_pair and ms_pair (rs_pair is never instantiated: no wgmma in this binary)
    return src.replace(anchor, mb.DEQUANT + F6_HELPERS + "\n" + anchor, 1)


def launch_arm(mod, arm, units, grid, w, x, out, sm, err, marks, stream):
    assert arm in ARMS
    mod.launch(ARM_MODE[arm], N, NS, units, grid, w.data_ptr(), x.data_ptr(), out.data_ptr(), sm.data_ptr(),
               err.data_ptr(), marks.data_ptr(), stream)


def bounds_key(arm):
    """Global access sites: every compute arm issues exactly F3 ms's (bulk W, X init, epilogue O, marks/sm/err)."""
    return ("null" if arm == "null" else "ms", N, NS)


def replay_schedule(rounds):
    for rnd in range(rounds):
        order = ARMS[rnd % len(ARMS):] + ARMS[:rnd % len(ARMS)]
        for arm in order:
            yield rnd, arm, (BANK_OFF[ARMS.index(arm)] + rnd) % 3


# --------------------------------------------------------------------------------------------- numpy replays (CPU)
LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])


def wx8_bits(w):
    """numpy replay of wx8 (built from the same WX8 table as the CUDA text): u32 words -> [..., 8] FP32 bit patterns."""
    w = np.asarray(w, np.uint64)
    m, M = np.uint64(0xFFFFFFFF), np.uint64(WMASK)
    p = w & np.uint64(0x001B001B)
    out = []
    for kind, a in WX8:
        if kind == "shl":
            v = (w << np.uint64(a)) & m
        elif kind == "id":
            v = w
        elif kind == "rot":
            v = ((w << np.uint64(a)) | (w >> np.uint64(32 - a))) & m
        else:
            v = (p * np.uint64(a)) & m
        out.append(v & M)
    return np.stack(out, -1).astype(np.uint32)


def f32(bits):
    return np.ascontiguousarray(bits, dtype=np.uint32).view(np.float32).astype(np.float64)


def sc32_bits(s, b):
    """numpy replay of sc32<b>: __byte_perm(s, 0, 0x4440 | b) = byte b of s; (byte << 20) + 0x60000000."""
    s = np.asarray(s, np.uint64)
    return ((((s >> np.uint64(8 * b)) & np.uint64(255)) << np.uint64(20)) + np.uint64(SC_BIAS)).astype(np.uint32)


def encode_word(codes):
    """[..., 8] 4-bit codes -> packed u32 words through BITPOS3 (as pack_tiles)."""
    w = np.zeros(codes.shape[:-1], np.uint64)
    for i, pos in enumerate(mb.BITPOS):
        for bit, dst in enumerate(pos):
            w |= ((codes[..., i].astype(np.uint64) >> np.uint64(bit)) & np.uint64(1)) << np.uint64(dst)
    return w


def code_value(c):
    c = np.asarray(c)
    return LUT[c & 7] * np.where(c & 8, -1.0, 1.0)


def xprep_np(xp, T, K):
    """Replay xprep's loop (tid < 256, r < 2T) on the packed BF16 x: -> x'[32][T][4][8] float64 and the write count."""
    xs = xp.view(np.uint32)
    out = np.full((32 * T * 4 * 4, 2), np.nan)
    hits = np.zeros(32 * T * 4 * 4, np.int64)
    for tid in range(256):
        for r in range(2 * T):
            e = tid + 256 * r
            j, t, tau, q = e & 3, (e >> 2) & 3, (e >> 4) % T, (e >> 4) // T
            idx = ((2 * q + (j >> 1)) * N * 16 + (tau >> 3) * 128 + (j & 1) * 64 + (tau & 7) * 8 + 2 * t) >> 1
            assert 0 <= idx * 4 and idx * 4 + 4 <= N * 2048
            v = int(xs[idx])
            st = ((q * T + tau) * 4 + t) * 4 + j
            assert XB + st * 8 + 8 <= XB + T * 4096 <= 65536
            lo, hi = f32(np.array([(v << 16) & 0xFFFFFFFF]))[0], f32(np.array([v & 0xFFFF0000]))[0]
            out[st] = (lo * 2.0 ** K, hi * 2.0 ** K)
            hits[st] += 1
    return out.reshape(32, T, 4, 8), hits


def dq8_vals(cw, sw, sub, rh):
    """Shipped deq4 + BF16 scale (dec_half2) for one code word: [lanes] -> [lanes, 8] float64 = weight / 256."""
    cw = np.asarray(cw, np.uint64)
    m = np.uint64(0xFFFFFFFF)
    rot = lambda n: ((cw << np.uint64(n)) | (cw >> np.uint64(32 - n))) & m
    dq = (cw & np.uint64(0x81C081C0), rot(10) & np.uint64(0x81C081C0), rot(13) & np.uint64(0x81C081C0),
          ((cw & np.uint64(0x001B001B)) * np.uint64(0x1084)) & np.uint64(0x81C081C0))
    sc = f32((((np.asarray(sw, np.uint64) >> np.uint64(8 * (2 * sub + rh))) & np.uint64(255)) * np.uint64(16) + np.uint64(0x7580)) << np.uint64(16))
    out = np.empty(cw.shape + (8,))
    for j, pair in enumerate(dq):
        for bit in (0, 1):
            half = f32(((pair >> np.uint64(16 * bit)) & np.uint64(65535)) << np.uint64(16))
            v = half * sc
            # HMUL2 is exact here: the product is representable in BF16 (<= 8 significant bits, normal range).
            assert np.array_equal(f32(np.ascontiguousarray(v.astype(np.float32)).view(np.uint32) & np.uint32(0xFFFF0000)), v)
            out[..., 2 * j + bit] = v
    return out


def emulate_unit(packed, xpr, T, form, swap_rows_scale=False):
    """Lane-level replay of one unit: -> y [64, T] float64 (true row sums, after the epilogue's x 256).
    swap_rows_scale: negative control (rows g / g+8 take each other's scale byte)."""
    w = packed[:32768].copy().view(np.uint32).reshape(4, 16, 32, 4)
    s = packed[32768:].copy().view(np.uint32).reshape(4, 8, 32)
    lanes = np.arange(32)
    t = lanes & 3
    y = np.zeros((64, T))
    for wp in range(4):
        part = np.zeros((32, 2, T))
        for kb in range(16):
            sw, sub = s[wp, kb // 2], kb % 2
            for rh in range(2):
                if form == "cf":
                    vals = np.concatenate([f32(wx8_bits(w[wp, kb, :, 2 * rh + hf])) for hf in range(2)], -1)
                    sc = f32(sc32_bits(sw, 2 * sub + (rh ^ int(swap_rows_scale))))
                    for tau in range(T):
                        xv = np.concatenate([xpr[2 * kb + hf, tau, t] for hf in range(2)], -1)
                        part[:, rh, tau] += sc * (vals * xv).sum(-1)
                else:
                    for hf in range(2):
                        vals = dq8_vals(w[wp, kb, :, 2 * rh + hf], sw, sub, rh)
                        for tau in range(T):
                            part[:, rh, tau] += (vals * xpr[2 * kb + hf, tau, t]).sum(-1)
        red = part.reshape(8, 4, 2, T).sum(1)              # the 4 lanes t of each g
        for rh in range(2):
            y[16 * wp + np.arange(8) + 8 * rh] = red[:, rh]
    return y * 256.0


def epilogue_layout(T):
    """F3 epilogue (N = 8) with f6_out's d[] assignment: -> {(row, col): (row, token)} for all 64 x 8 outputs."""
    cells = {}
    for wp in range(4):
        for lane in range(32):
            g = lane >> 2
            d = [(16 * wp + g, 0), (16 * wp + g, T - 1), (16 * wp + g + 8, 0), (16 * wp + g + 8, T - 1)]
            for j in range(N // 2):
                r = wp * 16 + (lane >> 2) + ((j & 3) >> 1) * 8
                c = (j // 4) * 8 + (lane & 3) * 2 + (j & 1)
                assert (r, c) not in cells
                cells[(r, c)] = d[j]
    return cells


# ----------------------------------------------------------------------------------------------- SASS classes / claims
SIGNATURES = (   # m16xe's dequant signatures (pinned base counts) + the cvtfree weight mask
    ("deq_LOP3", re.compile(r"^LOP3\.LUT R\d+, R\d+, (0x81c081c0|0x1b001b), ")),
    ("deq_SHF", re.compile(r"^SHF\.L\.W\.U32(\.HI)? R\d+, R\d+, 0x[ad], R\d+")),
    ("deq_IMAD", re.compile(r"^IMAD R\d+, R\d+, 0x1084, RZ")),
    ("scale_LEA", re.compile(r"^LEA\S* R\d+, R\d+, 0x75807580, ")),
    ("sink_IMAD", re.compile(r"^IMAD R\d+, R\d+, R\d+, (R\d+|RZ)$")),
    ("cf_LOP3", re.compile(r"^LOP3\.LUT R\d+, R\d+, (0x81c00000|0x81c0), ")),
    ("cf_LEA", re.compile(r"^LEA\S* R\d+, R\d+, 0x60000000, ")),
)
TABLE_CLASSES = ("HMMA", "FFMA", "FMUL", "FADD", "HMUL2", "HFMA2", "IMAD", "LOP3", "SHF", "PRMT", "LEA", "IADD3", "MOV",
                 "SHFL", "LDS", "LDS.128", "LDS.32", "STS", "deq_LOP3", "deq_SHF", "deq_IMAD", "scale_LEA", "cf_LOP3",
                 "cf_LEA", "FTZ", "TOTAL")
# base_n8 / twin_n8 / null_n8 static class counts of the F5b local gate (tools/local/f5b_gates.json, m16xe classify;
# equal to the pod's F5-data SASS).  The F6 kernel-text edit must leave them unchanged.
F5B_BASE = {"HMMA": 64, "sink_IMAD": 0, "IMAD": 129, "deq_LOP3": 320, "LOP3": 349, "deq_SHF": 128, "SHF": 157,
            "deq_IMAD": 64, "PRMT": 16, "HMUL2": 256, "scale_LEA": 16, "LEA": 36, "LDS": 156, "LDS.128": 20,
            "LDS.32": 136, "MOV": 24, "IADD3": 21, "TOTAL": 1736, "LDS.64": 0}
F5B_NULL = {"HMMA": 0, "sink_IMAD": 0, "IMAD": 43, "deq_LOP3": 0, "LOP3": 18, "deq_SHF": 0, "SHF": 21, "deq_IMAD": 0,
            "PRMT": 0, "HMUL2": 0, "scale_LEA": 0, "LEA": 14, "LDS": 0, "LDS.128": 0, "LDS.32": 0, "MOV": 18,
            "IADD3": 16, "TOTAL": 552, "LDS.64": 0}
F5B_PIN = {"base": F5B_BASE, "twin": F5B_BASE, "null": F5B_NULL}


def sass_functions(sass):
    """Copy of m16xe.sass_functions: static instructions per SASS function: (opcode counts incl. modifiers, texts)."""
    funcs = {}
    for chunk in re.split(r"(?=Function\s*:\s*)", sass):
        m = re.match(r"Function\s*:\s*(\S+)", chunk)
        if not m:
            continue
        ops, texts = {}, []
        for ins in re.findall(r"/\*[0-9a-f]+\*/\s+([@A-Z][^;]*);", chunk):
            body = re.sub(r"\s+", " ", re.sub(r"^@!?U?P[T\d]+\s+", "", ins).replace(".reuse", "")).strip()
            op = body.split(" ")[0]
            ops[op] = ops.get(op, 0) + 1
            texts.append(body)
        funcs[m[1]] = (ops, texts)
    return funcs


def classify(ops, texts=()):
    """m16xe.classify (IMAD excludes IMAD.MOV*, counted under MOV; LDS split by width) + FP32 / SHFL / STS / FTZ."""
    c = {k: 0 for k in TABLE_CLASSES + ("LDS.64", "sink_IMAD")}
    for op, k in ops.items():
        parts = op.split(".")
        base = parts[0]
        c["TOTAL"] += k
        if base == "MOV" or op.startswith("IMAD.MOV"):
            c["MOV"] += k
        elif base == "LDS":
            c["LDS"] += k
            c["LDS.128" if "128" in parts else ("LDS.64" if "64" in parts else "LDS.32")] += k
        elif base in c:
            c[base] += k
        if base in ("FFMA", "FMUL", "FADD") and "FTZ" in parts:
            c["FTZ"] += k
    for t in texts:
        for name, rx in SIGNATURES:
            if rx.match(t):
                c[name] += 1
    return c


def ffma_delta(arm):
    """Pre-registered per-unit consume + xprep counts of an FFMA arm (exact by construction: __fmul_rn/__fmaf_rn/
    __fadd_rn are never contracted).  dq: per (row, tau) 4 partials (first q: FMUL) -> 3 FADD; cf: per (row, tau, kb)
    blk = FMUL + 15 FFMA, acc: FMUL (kb 0) then FFMA; xprep: 2T LDS.32, 4T FMUL if K != 0; rsum4: 2 SHFL + 2 FADD,
    i.e. 4T SHFL + 4T FADD on the executed path.  STATIC counts: ptxas guards __shfl_xor_sync with BRA.DIV and emits a
    cold divergent-warp fallback copy of the butterfly (WARPSYNC.COLLECTIVE + SHFL + FADD; never taken: all 32 lanes
    are converged at the reduction), so the static SHFL / butterfly FADD are 8T."""
    T, form, K = SPEC[arm]
    if form == "dq":
        return {"FFMA": 504 * T, "FMUL": 8 * T, "FADD": 6 * T + 8 * T, "SHFL": 8 * T, "LDS.128": 64 * T}
    return {"FFMA": 510 * T, "FMUL": 34 * T + (4 * T if K else 0), "FADD": 8 * T, "SHFL": 8 * T, "LDS.128": 64 * T}


def expected(arm, base):
    """Claims of an FFMA arm: (exact, at_least).  Shell classes relative to base (same F3 shell), consume classes
    absolute.  LDS.32 is a lower bound: 8 scale + 2T xprep; ptxas may add a re-load of one x' float that its LDS.128
    also loads (seen: +1 per token; reported as 'x_reloads')."""
    T, form, _ = SPEC[arm]
    e = {k: base.get(k, 0) + v for k, v in ffma_delta(arm).items()}
    e.update({"HMMA": 0, "HMUL2": 256 if form == "dq" else 0, "FTZ": 0, "LDS.64": base.get("LDS.64", 0)})
    if form == "dq":   # dequant as shipped: the base's dec_half2 signatures, unchanged
        e.update({k: base.get(k, 0) for k in ("deq_LOP3", "deq_SHF", "deq_IMAD", "scale_LEA")})
    else:              # cvtfree weights: 8 masks / word (64 words), 64 prefilters + 64 + 64 IMAD (codes 6, 7), 32 scale LEA
        e.update({"cf_LOP3": 512, "deq_IMAD": 64, "cf_LEA": 32})
    return e, {"LDS.32": base.get("LDS.32", 0) - 128 + 2 * T}


def est_issue(arm):
    """EST source-level warp-instructions per unit per consumer warp (consume body only; SASS reported separately)."""
    if arm in ("base", "twin"):
        return {"dequant": 800, "HMMA": 64, "x_LDS": 128, "w_LDS": 24, "total": 1016}
    T, form, K = SPEC[arm]
    d = ffma_delta(arm)
    fp = d["FFMA"] + d["FMUL"] - (4 * T if K else 0) + (10 * T if form == "dq" else 4 * T)   # executed FADD path
    if form == "dq":
        r = {"dequant": 800, "widen": 512, "fp32": fp, "SHFL": 4 * T, "x_LDS": 64 * T, "w_LDS": 24}
    else:
        r = {"weights_fp32": 64 * 16, "scales": 32 * 2, "fp32": fp, "SHFL": 4 * T, "x_LDS": 64 * T, "w_LDS": 24}
    r["total"] = sum(r.values())
    r["per_weight"] = r["total"] / 512
    return r


def class_claims(table):
    """table: {arm: class counts}.  base/twin/null pinned to F5b; twin == base (all opcodes); FFMA arms exact."""
    rows, ok = [], True
    base = table.get("base")
    if base is None or table.get("twin") is None or table.get("null") is None:
        return {"status": "FAIL", "why": "missing base/twin/null", "rows": []}
    for arm in ("null", "base", "twin"):
        got, pin = table[arm], F5B_PIN[arm]
        bad = {k: {"expected": v, "actual": got.get(k, 0)} for k, v in pin.items() if got.get(k, 0) != v}
        rows.append({"arm": arm, "claim": "class counts equal the F5b gate's", "status": "PASS" if not bad else "FAIL", "mismatch": bad})
        ok &= not bad
    same = table["twin"] == base
    rows.append({"arm": "twin", "claim": "every class count identical to base", "status": "PASS" if same else "FAIL"})
    ok &= same
    for arm in FFMA_ARMS:
        got = table.get(arm)
        if got is None:
            rows.append({"arm": arm, "status": "FAIL", "why": "missing"}); ok = False; continue
        exp, least = expected(arm, base)
        bad = {k: {"expected": v, "actual": got.get(k, 0)} for k, v in exp.items() if got.get(k, 0) != v}
        bad.update({k: {"expected_at_least": v, "actual": got.get(k, 0)} for k, v in least.items() if got.get(k, 0) < v})
        rows.append({"arm": arm, "status": "PASS" if not bad else "FAIL", "mismatch": bad, "claimed_exact": exp,
                     "claimed_at_least": least, "x_reloads": got.get("LDS.32", 0) - least["LDS.32"],
                     "reported_delta_vs_base": {k: got.get(k, 0) - base.get(k, 0) for k in TABLE_CLASSES}})
        ok &= not bad
    return {"status": "PASS" if ok else "FAIL", "rows": rows}


ALU_OPS = ("LOP3", "SHF", "IADD3", "LEA", "PRMT", "ISETP", "VIADD", "SEL", "PLOP3", "FLO", "POPC", "BMSK", "VIMNMX")
FMA_OPS = ("FFMA", "FMUL", "FADD", "IMAD", "HMUL2", "HFMA2")


def consume_window(texts):
    """Report only: the straight-line consume region of one unit, from the first LDS.128 (the first code load: the
    xprep prologue has no LDS.128 and the helper warps' come after) to the first STS.64 after it (the d -> ep store).
    One unit per warp executes it once (fast path), so it is the per-unit issue count of the consumer warp."""
    ops = [re.sub(r"^@!?U?P[T\d]+\s+", "", t).split(" ")[0] for t in texts]
    try:
        a = next(i for i, o in enumerate(ops) if o == "LDS.128")
        b = next(i for i in range(a, len(ops)) if ops[i] == "STS.64")
    except StopIteration:
        return None
    w = ops[a:b + 1]
    kind = lambda o: "MOV" if (o.split(".")[0] == "MOV" or o.startswith("IMAD.MOV")) else o.split(".")[0]
    k = [kind(o) for o in w]
    return {"instructions": len(w), "first_index": a, "last_index": b,
            "alu_pipe": sum(x in ALU_OPS for x in k), "fma_pipe": sum(x in FMA_OPS for x in k),
            "imad_family": sum(x == "IMAD" for x in k), "mov": k.count("MOV"), "lds": k.count("LDS"),
            "hmma": k.count("HMMA"), "hfma2": k.count("HFMA2"), "shfl": k.count("SHFL"), "bra": k.count("BRA")}


def class_table_lines(table):
    lines = ["| arm | " + " | ".join(TABLE_CLASSES) + " |", "|---|" + "---:|" * len(TABLE_CLASSES)]
    for arm in ARMS:
        c = table.get(arm)
        if c:
            lines.append(f"| {arm} | " + " | ".join(str(c.get(k, 0)) for k in TABLE_CLASSES) + " |")
    return lines


# ------------------------------------------------------------------------------------------------------ reading
def reading(rows, correctness):
    """Pre-registered F6 decision (see PASS_US ... AA_MHZ)."""
    by = {r["arm"]: r for r in rows}
    ok = {c["arm"]: c.get("status") == "PASS" for c in correctness}
    b, t = by.get("base"), by.get("twin")
    if not b or not t:
        return {"status": "VOID: base/twin missing"}
    aa_us, aa_mhz = abs(t["us_per_unit"] - b["us_per_unit"]), abs(t["mhz_median"] - b["mhz_median"])
    controls_ok = all(ok.get(a, False) for a in ("null", "base", "twin"))
    valid = aa_us <= AA_US + 1e-12 and aa_mhz <= AA_MHZ + 1e-12 and controls_ok
    arms = {}
    for arm in FFMA_ARMS:
        r = by.get(arm)
        if not r:
            continue
        T = SPEC[arm][0]
        us, mhz = r["us_per_unit"], r["mhz_median"]
        if not ok.get(arm, False):
            st = "FAIL: incorrect outputs"
        elif T == 1:
            st = "PASS" if (us <= PASS_US + 1e-12 and mhz >= PASS_MHZ) else ("KILL" if us > KILL_US else "HOLD")
        else:
            st = "PASS-2" if us <= PASS2_US + 1e-12 else "no PASS-2"
        arms[arm] = {"status": st, "us_per_unit": us, "mhz": mhz, "d_us_vs_base": us - b["us_per_unit"],
                     "d_mhz_vs_base": mhz - b["mhz_median"], "correct": ok.get(arm, False),
                     "ratio_vs_base": us / b["us_per_unit"]}
    one = [arms[a]["status"] for a in ("ffma1", "ffma1_cvtfree") if a in arms]
    two = [arms[a]["status"] for a in ("ffma2", "ffma2_cvtfree") if a in arms]
    if not valid:
        why = [] if controls_ok else ["null/base/twin correctness"]
        why += [f"|twin-base| {aa_us:.4f} us/unit > {AA_US}"] if aa_us > AA_US + 1e-12 else []
        why += [f"|twin-base| {aa_mhz:.1f} MHz > {AA_MHZ}"] if aa_mhz > AA_MHZ + 1e-12 else []
        d1 = d2 = "VOID: " + "; ".join(why)
    else:
        d1 = "PASS" if "PASS" in one else ("KILL" if one and all(s in ("KILL", "FAIL: incorrect outputs") for s in one) else "HOLD")
        d2 = "PASS-2" if "PASS-2" in two else "no PASS-2"
    return {"status": f"ffma1: {d1}; ffma2: {d2}", "ffma1": d1, "ffma2": d2, "aa_us_per_unit": aa_us,
            "aa_mhz": aa_mhz, "aa_valid": valid, "base_us_per_unit": b["us_per_unit"], "base_mhz": b["mhz_median"],
            "arms": arms}


# ------------------------------------------------------------------------------------------------------ CPU gates
def cpu_gates():
    inherited = mb.cpu_gates()          # F3: real base prepare, verbatim helpers vs m16xa, ring/protocol replays, bounds
    for k, h in PINS.items():
        assert hashlib.sha256(getattr(mb, k).encode()).hexdigest() == h, ("m16xb_micro.py changed", k)
    src = generate_cuda()
    assert "@" not in src and "__trap" not in src and "wgmma.mma_async" not in src  # rs_pair never instantiated
    assert len(set(ARMS)) == len(ARMS) == 7 and len(set(ARM_MODE.values())) == len(ARM_MODE) and set(ARMS) == set(ARM_MODE)
    assert len(re.findall(r"m16xb<8, \d+, 3><<<", src)) == len(ARMS)
    assert len(re.findall(r"return config<8, \d+, 3>\(true\);", src)) == len(ARMS)
    for arm in ARMS:
        assert src.count(f"if (mode == {ARM_MODE[arm]} && n == {N} && ns == {NS})") == 2, (arm, "info+dispatch")
    # CUDA constexpr T / K tables == SPEC; xprep index formulas == the numpy replay's.
    for arm, (T, form, K) in SPEC.items():
        mode = ARM_MODE[arm]
        assert (mode in (13, 14)) == (T == 2) and (mode in (12, 14)) == (K == 43) and (form == "cf") == (K != 0)
    assert "template<int MODE> constexpr int F6T = (MODE == 13 || MODE == 14) ? 2 : 1;" in src
    assert "template<int MODE> constexpr int F6K = (MODE == 12 || MODE == 14) ? 43 : 0;" in src
    assert XPREP_INDEX in src and XPREP_STORE in src
    # base consume body is F3's ms body byte for byte; the kernel differs from F3 only by the inserted xprep line.
    mb_src = mb.generate_cuda()
    block = "\n".join(consume_body(3)) + "\n}"
    assert ("else if constexpr (MODE == 3) {\n" + block) in mb_src, "base consume differs from F3 ms"
    assert ("if constexpr (MODE == 3) {\n" + block) in src
    kern = lambda s: s[s.index("template<int N, int MODE, int NS> __global__"):s.index("__global__ void empty_kernel")]
    a, b = kern(mb.CUDA_HEAD).splitlines(), kern(src).splitlines()
    ops = [o for o in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if o[0] != "equal"]
    assert len(ops) == 1 and ops[0][0] == "insert" and b[ops[0][3]:ops[0][4]] == [XPREP_LINE.rstrip("\n")], ops
    pairs = mb.CUDA_HEAD[mb.CUDA_HEAD.index("template<int N, int Q> __device__ __forceinline__ void rs_pair"):mb.CUDA_HEAD.index("template<int N, int MODE> __device__")]
    assert pairs in src and mb.DEQUANT in src, "ms_pair / dequant not verbatim"
    assert "consume<N, (MODE == 4 ? 3 : MODE)>(d, s + XS + slot * UB, reinterpret_cast<uint16_t*>(s), wp, lane);" in src
    # per-mode source accounting (exact at source level; SASS claims are checked by the compile gate)
    acct = {}
    for mode in CONSUME_MODES:
        body = "\n".join(consume_body(mode))
        cnt = {k: body.count(k) for k in ("dec_half2(", "ms_pair<N,", "dq_q<", "cf_kb<", "reinterpret_cast<const uint4*>",
                                          "reinterpret_cast<const uint32_t*>", "f6_out")}
        assert cnt["reinterpret_cast<const uint4*>"] == 16 and cnt["reinterpret_cast<const uint32_t*>"] == 8, mode
        arm = MODE_ARM[mode]
        if mode == 3:
            assert cnt["dec_half2("] == 32 and cnt["ms_pair<N,"] == 32 and cnt["dq_q<"] + cnt["cf_kb<"] == 0
        elif SPEC[arm][1] == "dq":
            assert cnt["dec_half2("] == 32 and cnt["dq_q<"] == 32 and cnt["ms_pair<N,"] + cnt["cf_kb<"] == 0
            assert body.count(", true>(p, h, xf + 0);") == 1 and cnt["f6_out"] == 1
        else:
            assert cnt["cf_kb<"] == 16 and cnt["dec_half2("] + cnt["ms_pair<N,"] + cnt["dq_q<"] == 0
            assert body.count(", true>(acc, c, scale, xf + 0);") == 1 and cnt["f6_out"] == 1
        acct[arm] = cnt
    # xf offsets: q * T * 8 float4 (dq) / 2 kb * T * 8 (cf); every x' read stays inside [XB, XB + T * 4096).
    for arm, (T, form, _) in SPEC.items():
        offs = [int(m) for m in re.findall(r"xf \+ (\d+)\);", "\n".join(consume_body(ARM_MODE[arm])))]
        assert offs == ([q * T * 8 for q in range(32)] if form == "dq" else [kb * 2 * T * 8 for kb in range(16)]), arm
        reach = max(offs) + (T * 8 + (T - 1) * 8 + 1 if form == "cf" else (T - 1) * 8 + 1)
        assert XB + 16 * (3 * 2 + reach + 1) <= XB + T * 4096 <= 65536, arm   # + t * 2 float4 (t <= 3)
    # wx8: exact FP32 value code * 2^-126 for every code in every position (random words: each (position, code)
    # appears ~4096 times; plus the all-equal words of every code).
    rng = np.random.default_rng(66)
    codes = np.concatenate([rng.integers(0, 16, (65536, 8)), np.repeat(np.arange(16), 8).reshape(16, 8)])
    vals = f32(wx8_bits(encode_word(codes)))
    assert np.array_equal(vals, code_value(codes) * 2.0 ** -126), "wx8 bit placement"
    assert all(np.sum(codes[:, i] == c) >= 3500 for i in range(8) for c in range(16))
    assert f32(np.array([0x00400000]))[0] == 2.0 ** -127                # code 1: the FP32 denormal (no FTZ allowed)
    # The 0x1B001B prefilter makes p * 0x1084 carry-free (all 12 partial-product bits distinct, lows below highs).
    lows = sorted(b + s for b in (0, 1, 3, 4) for s in (2, 7, 12))
    highs = sorted(b + s for b in (16, 17, 19, 20) for s in (2, 7, 12))
    assert len(set(lows)) == 12 and len(set(highs)) == 12 and max(lows) < min(highs)
    # sc32: every finite E4M3 scale (incl. subnormals, negatives) in every byte position, random other bytes.
    sf = np.array([v for v in range(256) if v not in (127, 255)], np.uint8)
    bp, neg, zero = mb.reencode(sf)
    for bb in range(4):
        other = rng.integers(0, 256, (sf.size, 4)).astype(np.uint64)
        other[:, bb] = bp
        word = sum(other[:, k] << np.uint64(8 * k) for k in range(4))
        sv = f32(sc32_bits(word, bb))
        assert np.array_equal(sv[~zero], np.abs(mb.fp8_numpy(sf[~zero])).astype(np.float64) * 2.0 ** 75), ("sc32", bb)
        assert np.all(np.isfinite(sv)) and np.all(sv > 0)
    # full lane-level replays (numpy, float64) of all four FFMA formulations against the float64 reference, on a unit
    # with every finite scale code (signed, subnormal, zero) and random codes; x from torch-free random BF16 bits.
    raw = rng.integers(0, 256, (2, 64, 512), dtype=np.uint8)
    scale = np.resize(sf, (2, 64, 64)).copy()
    scale[1] = rng.permutation(scale[1].reshape(-1)).reshape(64, 64)
    packed = mb.pack_tiles(raw, scale)
    xr = (rng.standard_normal((N, 1024)) * np.exp2(rng.integers(-6, 7, (N, 1024)))).astype(np.float32)
    xb = (xr.view(np.uint32) >> 16).astype(np.uint16)                     # BF16 bits (truncation; exact in BF16)
    xb[:, :8] = [0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x3F80, 0xBF80, 0x7F7F >> 4]   # zeros, BF16 denormals, normals
    xfull = f32(xb.astype(np.uint32) << 16).reshape(N, 1024)
    xp = mb.pack_x(xb)
    errs = {}
    for arm, (T, form, K) in SPEC.items():
        xpr, hits = xprep_np(xp, T, K)
        assert np.all(hits == 1), (arm, "xprep writes every x' pair exactly once")
        # independent derivation of x' from the A-fragment lane map (code i -> A-domain k -> original column kperm(k))
        for q in range(32):
            for tau in range(T):
                for t in range(4):
                    for i in range(8):
                        k = 16 * (2 * q + i // 4) + 2 * t + 8 * ((i // 2) & 1) + (i & 1)
                        assert xpr[q, tau, t, i] == xfull[tau, mb.kperm(k)] * 2.0 ** K, (arm, q, tau, t, i)
        for u in range(2):
            c = np.stack((raw[u] & 15, raw[u] >> 4), axis=-1).reshape(64, 64, 16)
            dw = (code_value(c) * mb.fp8_numpy(scale[u]).astype(np.float64)[:, :, None]).reshape(64, 1024)
            ref = dw @ xfull[:T].T
            got = emulate_unit(packed[u], xpr, T, form)
            err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1.0))
            assert err <= 1e-10, (arm, u, err)      # float64 summation order only; a map error is O(1)
            errs[f"{arm}_u{u}"] = err
            # negative controls: a wrong x' code order / a swapped row scale must be caught by this replay
            bad = [emulate_unit(packed[u], xpr[..., [1, 0, 2, 3, 4, 5, 6, 7]], T, form)]
            if form == "cf":
                bad.append(emulate_unit(packed[u], xpr, T, form, swap_rows_scale=True))
            assert all(np.abs(v - ref).max() / max(np.abs(ref).max(), 1.0) > 1e-3 for v in bad), (arm, u, "vacuous replay")
        # output layout: out[u][r][c] = y[u][r][c % T] in all 64 x 8 cells
        cells = epilogue_layout(T)
        assert len(cells) == 64 * N and all(cells[(r, c)] == (r, c % T) for r in range(64) for c in range(N)), arm
    # the shipped lane map that the emulation assumes (mb.emulate_cuda_a: dec_half2 -> A fragment rows/k)
    assert np.array_equal(mb.emulate_cuda_a(packed[0]), mb.unpack_tiles(packed[:1])[0][:, mb.kperm(np.arange(1024))])
    # timing schedule: every consecutive replay changes bank; every arm sees each bank 4x in 12 rounds.
    sched = list(replay_schedule(12))
    assert all(x[2] != y[2] for x, y in zip(sched, sched[1:]))
    assert all([sum(k == arm and bk == bank for _, k, bk in sched) for bank in range(3)] == [4, 4, 4] for arm in ARMS)
    # global bounds through the F3 replay (identical access sites for every compute arm).
    class Tensor:
        def __init__(self, count, width): self.count, self.width = count, width
        def numel(self): return self.count
        def element_size(self): return self.width
        def is_contiguous(self): return True
        def data_ptr(self): return 0x100000000
    bounds = {}
    for arm in ARMS:
        arm_mb, n, ns = bounds_key(arm)
        r = mb.tensor_bounds(arm_mb, n, UNITS, mb.GRID, Tensor(mb.GRID * UNITS * mb.UNIT, 1), Tensor(n * 1024, 2),
                             Tensor(mb.GRID * UNITS * 64 * 32, 4), Tensor(mb.GRID, 4), Tensor(mb.GRID, 4), Tensor(2, 8), ns=ns)
        bounds[arm] = r["status"]
    assert mb.smem_bytes(N, NS) <= mb.SMEM_CAP and XB + 2 * 4096 <= 65536
    # SASS parser + claim checker on synthetic fixtures (never presented as measurements).
    fx = ("Function : f\n /*0000*/ HMMA.16816.F32.BF16 R0, R4.reuse, R8, R0 ;  /* 0x0000001c201c723c */\n"
          " /*0010*/ IMAD.MOV.U32 R1, RZ, RZ, R2 ;\n /*0020*/ @!P0 LDS.128 R4, [R1] ;\n /*0030*/ LDS R5, [R2+0x80] ;\n"
          " /*0040*/ IMAD R3, R3.reuse, 0x1084, RZ ;\n /*0050*/ FFMA R7, R9.reuse, R10, R7 ;\n /*0060*/ FMUL.FTZ R6, R9, R11 ;\n"
          " /*0070*/ LOP3.LUT R2, R22, 0x81c00000, RZ, 0xc0, !PT ;\n /*0080*/ SHFL.BFLY PT, R24, R23, 0x1, 0x1f ;\n"
          " /*0090*/ FADD R37, R34, R35 ;\n /*00a0*/ STS.64 [R6], R8 ;\n /*00b0*/ LOP3.LUT R2, R22, 0x1b001b, RZ, 0xc0, !PT ;\n")
    c = classify(*sass_functions(fx)["f"])
    assert (c["HMMA"], c["MOV"], c["IMAD"], c["LDS"], c["LDS.128"], c["LDS.32"], c["TOTAL"]) == (1, 1, 1, 2, 1, 1, 12)
    assert (c["FFMA"], c["FMUL"], c["FADD"], c["FTZ"], c["SHFL"], c["STS"], c["LOP3"]) == (1, 1, 1, 1, 1, 1, 2)
    assert (c["deq_IMAD"], c["cf_LOP3"], c["deq_LOP3"]) == (1, 1, 1)
    synth = {"base": dict(F5B_BASE, FFMA=0, FMUL=16, FADD=0, SHFL=4, STS=31, FTZ=0, cf_LOP3=0),
             "twin": dict(F5B_BASE, FFMA=0, FMUL=16, FADD=0, SHFL=4, STS=31, FTZ=0, cf_LOP3=0),
             "null": dict(F5B_NULL)}
    for arm in FFMA_ARMS:
        exact, least = expected(arm, synth["base"])
        synth[arm] = dict(synth["base"]); synth[arm].update(exact); synth[arm].update(least)
    assert class_claims(synth)["status"] == "PASS"
    for arm, field, step in (("ffma1", "FFMA", 1), ("ffma1_cvtfree", "FMUL", 1), ("ffma2", "deq_SHF", 1),
                             ("ffma2_cvtfree", "HMMA", 1), ("ffma1_cvtfree", "FTZ", 1), ("ffma2", "LDS.128", 1),
                             ("ffma2_cvtfree", "cf_LOP3", -1), ("ffma1", "LDS.32", -1), ("null", "TOTAL", 1),
                             ("twin", "MOV", 1)):
        bad = {k: dict(v) for k, v in synth.items()}
        bad[arm][field] += step
        assert class_claims(bad)["status"] == "FAIL", (arm, field)
    more = {k: dict(v) for k, v in synth.items()}
    more["ffma2"]["LDS.32"] += 2                                     # x' re-loads are allowed (lower bound)
    assert class_claims(more)["status"] == "PASS"
    fx_w = [t for t in ("LDS R1, [R2]", "LDS.128 R4, [R1]", "LOP3.LUT R2, R2, 0x1, RZ, 0xc0, !PT", "FFMA R1, R2, R3, R1",
                        "IMAD.MOV.U32 R1, RZ, RZ, R2", "HMMA.16816.F32.BF16 R0, R4, R8, R0", "STS.64 [R1], R2", "STS.64 [R1], R4")]
    cw = consume_window(fx_w)
    assert (cw["instructions"], cw["alu_pipe"], cw["fma_pipe"], cw["mov"], cw["hmma"]) == (6, 1, 1, 1, 1)
    # reading boundaries on synthetic rows.
    def fx_rows(f1=1.15, m1=1900.0, f2=1.18, aa=0.0, aam=0.0, wrong=()):
        rows = [{"arm": a, "us_per_unit": 1.226 + (aa if a == "twin" else 0), "mhz_median": 1738.0 + (aam if a == "twin" else 0)}
                for a in ("null", "base", "twin")]
        rows += [{"arm": "ffma1", "us_per_unit": f1, "mhz_median": m1}, {"arm": "ffma1_cvtfree", "us_per_unit": 1.3, "mhz_median": 1950.0},
                 {"arm": "ffma2", "us_per_unit": f2, "mhz_median": 1900.0}, {"arm": "ffma2_cvtfree", "us_per_unit": 1.3, "mhz_median": 1900.0}]
        corr = [{"arm": a, "status": "FAIL" if a in wrong else "PASS"} for a in ARMS]
        return reading(rows, corr)
    assert fx_rows()["status"] == "ffma1: PASS; ffma2: PASS-2"
    assert fx_rows(f1=1.1501)["ffma1"] == "HOLD" and fx_rows(m1=1899.9)["ffma1"] == "HOLD"
    assert fx_rows(f1=1.2201)["ffma1"] == "KILL" and fx_rows(f1=1.22)["ffma1"] == "HOLD"
    assert fx_rows(f2=1.1801)["ffma2"] == "no PASS-2"
    assert fx_rows(wrong=("ffma1",))["ffma1"] == "KILL"          # incorrect ffma1, ffma1_cvtfree > 1.22
    assert fx_rows(wrong=("ffma1",))["arms"]["ffma1"]["status"].startswith("FAIL")
    assert fx_rows(aa=0.03)["aa_valid"] and fx_rows(aam=15.0)["aa_valid"]
    assert fx_rows(aa=0.0301)["ffma1"].startswith("VOID") and fx_rows(aam=15.01)["ffma2"].startswith("VOID")
    assert fx_rows(wrong=("twin",))["ffma1"].startswith("VOID")
    summary_lines({"status": "SYNTHETIC", "rows": [], "reading": fx_rows()})   # writer exercised, nothing written
    return {"status": "PASS", "inherited_f3_cpu_gates": inherited["status"], "pins": PINS, "arms": list(ARMS),
            "modes": ARM_MODE, "spec": {k: {"tokens": v[0], "form": v[1], "x_exponent": v[2]} for k, v in SPEC.items()},
            "source_accounting_per_unit": acct, "est_issue_per_unit_warp": {a: est_issue(a) for a in ("base",) + FFMA_ARMS},
            "emulation_rel_err_vs_fp64": errs, "global_bounds": bounds, "kernel_text_edit_vs_f3": XPREP_LINE.strip(),
            "source_sha256": hashlib.sha256(src.encode()).hexdigest()}


# --------------------------------------------------------------------------------------------------------- build
def compiler_gates(log, sass):
    blocks = {}
    for chunk in re.split(r"(?=ptxas info\s*: Compiling entry function)", log):
        m = re.search(r"Compiling entry function '([^']+)'", chunk)
        inst = re.search(r"m16xbILi(\d+)ELi(\d+)ELi(\d+)E", m[1]) if m else None
        if not inst:
            continue
        regs = re.search(r"Used (\d+) registers", chunk)
        spill = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", chunk)
        smem = re.search(r"(\d+) bytes smem", chunk)
        n, mode, ns = int(inst[1]), int(inst[2]), int(inst[3])
        blocks[MODE_ARM.get(mode, f"mode{mode}")] = {
            "symbol": m[1], "n": n, "mode": mode, "ns": ns, "registers": int(regs[1]) if regs else None,
            "stack_bytes": int(spill[1]) if spill else None, "spill_store_bytes": int(spill[2]) if spill else None,
            "spill_load_bytes": int(spill[3]) if spill else None, "static_smem_bytes": int(smem[1]) if smem else 0}
    table, raw, window = {}, {}, {}
    for sym, (ops, texts) in sass_functions(sass).items():
        inst = re.search(r"m16xbILi(\d+)ELi(\d+)ELi(\d+)E", sym)
        if inst:
            key = MODE_ARM.get(int(inst[2]), f"mode{inst[2]}")
            table[key], raw[key], window[key] = classify(ops, texts), ops, consume_window(texts)
    claims = class_claims(table)
    k = list(blocks.values())
    c75 = [l for l in log.splitlines() if re.search(r"C75\d\d", l)]
    gates = {
        "all_instances": set(blocks) == set(ARMS) and set(table) == set(ARMS),
        "no_spills": bool(k) and all(r["spill_store_bytes"] is not None and r["spill_store_bytes"] <= 16 and r["spill_load_bytes"] <= 16 for r in k),
        "no_stack": bool(k) and all(r["stack_bytes"] is not None and r["stack_bytes"] <= 16 for r in k),
        "registers_le_128": bool(k) and all(r["registers"] is not None and 0 < r["registers"] <= 128 for r in k),
        "smem_cap": bool(k) and all(r["static_smem_bytes"] + mb.smem_bytes(r["n"], NS) <= mb.SMEM_CAP for r in k),
        "bulk_copy_every_instance": all(any(o.startswith(p) for o in raw.get(a, {}) for p in ("UBLKCP", "UTMALDG", "LDGSTS")) for a in ARMS),
        "hmma_only_in_base_twin": all((table.get(a, {}).get("HMMA", 0) > 0) == (a in ("base", "twin")) for a in ARMS),
        "no_wgmma": "HGMMA" not in sass,
        "no_c75xx": not c75,
        "no_ftz_fp32": all(table.get(a, {}).get("FTZ", 1) == 0 for a in FFMA_ARMS),
    }
    return {"ptxas": blocks, "sass_class_counts_static": table, "sass_opcodes_static": raw, "class_claims": claims,
            "consume_window_per_unit_warp": window, "gates": gates, "status": "PASS" if all(gates.values()) else "FAIL",
            "c75xx_lines": c75}, table


def build(out):
    from torch.utils.cpp_extension import load_inline
    src = generate_cuda()
    tag = hashlib.sha256((mb.CPP + src).encode()).hexdigest()[:12]
    builddir = out.parent / ("build_" + tag + "_" + str(os.getpid()))   # fresh: ptxas -v is always captured
    builddir.mkdir(parents=True, exist_ok=False)
    logfile = out.with_suffix(".compile.log")
    out.with_suffix(".cu").write_text(src, encoding="utf-8")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ["MAX_JOBS"] = "2"
    with mb.capture_fds(logfile):
        mod = load_inline(name="m16xf_" + tag, cpp_sources=mb.CPP, cuda_sources=src, functions=None,
                          extra_cflags=["-O3", "-std=c++20", "-DNDEBUG"],
                          extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-lineinfo",
                                             "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v,--warn-on-spills",
                                             "--keep", "--keep-dir=."],
                          build_directory=str(builddir), with_cuda=True, verbose=True, no_implicit_headers=True)
    log = logfile.read_text(errors="replace")
    sass = subprocess.run(["cuobjdump", "--dump-sass", mod.__file__], capture_output=True, text=True, timeout=60, check=True).stdout
    resources = subprocess.run(["cuobjdump", "--dump-resource-usage", mod.__file__], capture_output=True, text=True, timeout=30, check=True).stdout
    out.with_suffix(".sass").write_text(sass, encoding="utf-8")
    out.with_suffix(".resources.txt").write_text(resources, encoding="utf-8")
    result, table = compiler_gates(log, sass)
    result.update({"extension": mod.__file__, "compile_log": str(logfile), "generated_cuda": str(out.with_suffix(".cu")),
                   "build_dir": str(builddir)})
    return mod, result, table


# -------------------------------------------------------------------------------------------------------- GPU run
def smi_sample(seconds, replay_fn, torch):
    """Copy of m16xe.smi_sample: back-to-back replays for `seconds`, nvidia-smi samples power/clock every 50 ms."""
    cmd = ["nvidia-smi", "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,nounits", "-lms", "50"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        time.sleep(0.3)
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < seconds:
            for _ in range(100):
                replay_fn()
            torch.cuda.synchronize()
            n += 100
        elapsed = time.monotonic() - t0
    finally:
        p.terminate()
        try:
            out, _ = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
    rows = []
    for line in out.splitlines():
        try:
            a, b = (float(s.strip()) for s in line.split(","))
            rows.append((a, b))
        except ValueError:
            continue
    k = len(rows) // 10   # drop the first and last 10 % (ramp / tail)
    core = rows[k:len(rows) - k] if len(rows) > 2 * k + 2 else rows
    if not core:
        return {"samples": 0, "launches": n, "seconds": elapsed}
    return {"samples": len(core), "launches": n, "seconds": elapsed,
            "power_w_median": statistics.median(r[0] for r in core), "power_w_max": max(r[0] for r in core),
            "smi_sm_mhz_median": statistics.median(r[1] for r in core)}


def gpu_correctness(mod, torch):
    """All arms vs the F3 FP32 oracle + changed-input NaN-poisoned graph replays; null: output untouched.
    FFMA arms: out[u][r][c] must equal y[u][r][c % T] (token 0 / tokens 0-1 of the same x as base) in all 8 columns."""
    weights = mb.make_weights(1, 31)
    valid = np.array([b for b in range(256) if b not in (127, 255)], np.uint8)
    weights["w13_sf"][0, :256, :] = np.resize(valid, (256, 256))
    bank = mb.prepare_bank(weights, 4, 4)
    w = torch.from_numpy(bank).cuda()
    err = torch.zeros(4, device="cuda", dtype=torch.int32)
    sm = torch.empty_like(err)
    marks = torch.empty(2, device="cuda", dtype=torch.int64)
    gen = torch.Generator().manual_seed(411 + N)
    x = torch.randn(N, 1024, generator=gen).to(torch.bfloat16)
    original = torch.from_numpy(mb.pack_x(x.view(torch.int16).numpy().view(np.uint16)).view(np.int16)).cuda()
    px = torch.empty_like(original)
    refs = []
    for expert, row, stripe in zip(*mb.unit_coords(4, 4)):
        raw = torch.from_numpy(weights["w13"][expert, row * 64:(row + 1) * 64, stripe * 512:(stripe + 1) * 512].copy())
        sf = torch.from_numpy(weights["w13_sf"][expert, row * 64:(row + 1) * 64, stripe * 64:(stripe + 1) * 64].copy()).view(torch.float8_e4m3fn).float()
        c = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(64, 64, 16).long()
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
        dw = lut[c & 7] * torch.where((c & 8) != 0, -1., 1.) * sf[:, :, None]
        refs.append(dw.reshape(64, 1024).float() @ x.float().T)
    reference = torch.stack(refs)
    results = []
    for arm in ARMS:
        cols = list(range(N)) if arm not in SPEC else [c % SPEC[arm][0] for c in range(N)]
        expect = reference[:, :, cols]
        scale = expect.abs().amax(dim=(1, 2)).clamp_min(1)[:, None, None]
        row = {"arm": arm, "status": "FAIL", "layout": "out[u][r][c] = y[u][r][c]" if arm not in SPEC else
               f"out[u][r][c] = y[u][r][c % {SPEC[arm][0]}]"}
        results.append(row)
        mb.reset_correctness_input(px, original)
        out = torch.full((16, 64, N), float("nan"), device="cuda", dtype=torch.float32)
        mb.tensor_bounds(*bounds_key(arm)[:2], 4, 4, w, px, out, sm, err, marks, ns=NS)
        def call(arm=arm, out=out):
            launch_arm(mod, arm, 4, 4, w, px, out, sm, err, marks, torch.cuda.current_stream().cuda_stream)
        call(); torch.cuda.synchronize()
        if err.any().item():
            raise RuntimeError((arm, "device protocol timeout"))
        def agrees(value, sign):
            if arm == "null":
                return bool(torch.isnan(value).all())
            return bool(torch.isfinite(value).all() and torch.all((value - sign * expect).abs() <= 2e-5 * scale))
        got = out.cpu()
        if arm != "null":
            row["max_abs_error"] = float((got - expect).abs().nan_to_num(float("inf")).max().item())
            row["max_rel_error_vs_unit_scale"] = float(((got - expect).abs() / scale).nan_to_num(float("inf")).max().item())
        if not agrees(got, 1):
            row["why"] = "initial FP32 reference / untouched null mismatch"; continue
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call()
        torch.cuda.synchronize()
        ok_r = True
        for sign in (-1, 1, -1):
            px.copy_(torch.from_numpy(mb.pack_x((sign * x).view(torch.int16).numpy().view(np.uint16)).view(np.int16)).cuda())
            out.fill_(float("nan")); torch.cuda.synchronize(); graph.replay(); torch.cuda.synchronize()
            if err.any().item():
                raise RuntimeError((arm, "graph protocol timeout"))
            ok_r = agrees(out.cpu(), sign)
            if not ok_r:
                break
        row.update({"status": "PASS" if ok_r else "FAIL", "changed_input_replays": 3,
                    "check": "null untouched" if arm == "null" else "torch FP32 reference (2e-5 x unit max)"})
    return results


def benchmark(mod, args, report):
    import torch
    prop = torch.cuda.get_device_properties(0)
    assert "H200" in prop.name and prop.multi_processor_count == 132 and prop.major == 9, str(prop)
    report["device"] = {"name": prop.name, "sms": prop.multi_processor_count}
    attrs = {}
    for arm in ARMS:   # cudaFuncSetAttribute (dynamic smem) for EVERY instance before any launch
        reg, local, static, dynamic, occ = mod.info(ARM_MODE[arm], N, NS)
        assert 0 < reg <= 128 and local <= 16 and occ == 1, (arm, reg, local, occ)
        assert dynamic == mb.smem_bytes(N, NS) and static + dynamic <= mb.SMEM_CAP, (arm, static, dynamic)
        attrs[arm] = {"regs": reg, "local_bytes": local, "static_smem": static, "dynamic_smem": dynamic, "ctas_per_sm": occ}
    report["kernel_attributes"] = attrs
    report["correctness"] = gpu_correctness(mod, torch)
    bad = [c for c in report["correctness"] if c["arm"] in ("null", "base", "twin") and c["status"] != "PASS"]
    assert not bad, ("base/twin/null correctness", bad)
    print("[f6] correctness " + " ".join(f'{c["arm"]}:{c["status"]}' for c in report["correctness"]), flush=True)
    timer = mb.Timer(torch)
    stream = timer.stream.cuda_stream
    empty = timer.capture(lambda: mod.empty(stream))
    banks = []
    for b in range(3):
        weights = mb.make_weights(math.ceil(mb.GRID * args.units / 512), 103 + b)
        banks.append(torch.from_numpy(mb.prepare_bank(weights, args.units, mb.GRID)).cuda())
    assert len({w.data_ptr() for w in banks}) == 3
    gen = torch.Generator().manual_seed(103 + N)
    xin = torch.randn(N, 1024, generator=gen).to(torch.bfloat16)
    xin = torch.from_numpy(mb.pack_x(xin.view(torch.int16).numpy().view(np.uint16)).view(np.int16)).cuda()
    output = torch.empty((mb.GRID, args.units, 64, 32), device="cuda", dtype=torch.float32)
    sm = torch.empty(mb.GRID, device="cuda", dtype=torch.int32)
    err = torch.zeros_like(sm)
    marks = torch.empty(2, device="cuda", dtype=torch.int64)
    timer.stream.wait_stream(torch.cuda.current_stream())
    graphs = {}
    for arm in ARMS:
        graphs[arm] = []
        for bank, w in enumerate(banks):
            mb.tensor_bounds(*bounds_key(arm)[:2], args.units, mb.GRID, w, xin, output, sm, err, marks, ns=NS)
            graphs[arm].append(timer.capture(lambda arm=arm, w=w: launch_arm(mod, arm, args.units, mb.GRID, w, xin, output, sm, err, marks, stream)))
            assert not err.any().item(), (arm, bank, "protocol error")
            assert len(set(sm.cpu().tolist())) == mb.GRID, (arm, bank, "grid not one CTA per SM")
    print("[f6] graphs captured", flush=True)
    report["clocks_before"] = mb.clocks()
    before = statistics.median(timer.replay(empty) for _ in range(15))
    samples = {a: [] for a in ARMS}
    mhz = {a: [] for a in ARMS}
    banks_used = {a: [] for a in ARMS}
    for rnd, arm, bank in replay_schedule(args.rounds):
        samples[arm].append(timer.replay(graphs[arm][bank]))
        assert not err.any().item(), (arm, "device protocol error")
        cyc, ns_delta = marks.cpu().tolist()
        assert cyc > 0 and ns_delta > 0, (arm, "invalid loop clock")
        mhz[arm].append(1000.0 * cyc / ns_delta)
        banks_used[arm].append(bank)
    after = statistics.median(timer.replay(empty) for _ in range(15))
    report["clocks_after"] = mb.clocks()
    overhead = (before + after) / 2
    report["empty_kernel_us"] = {"before": before, "after": after, "subtracted": overhead}
    rows = []
    null = [(s - overhead) / args.units for s in samples["null"]]
    for arm in ARMS:
        vals = [(s - overhead) / args.units for s in samples[arm]]
        assert min(vals) > 0, "invalid empty-kernel subtraction"
        rows.append({"arm": arm, "mode": ARM_MODE[arm], "n_template": N, "us_per_unit": statistics.median(vals),
                     "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90)),
                     "ratio_null": statistics.median(vals) / statistics.median(null),
                     "mhz_median": statistics.median(mhz[arm]), "mhz_p10": float(np.percentile(mhz[arm], 10)),
                     "mhz_p90": float(np.percentile(mhz[arm], 90)), "mhz_samples": mhz[arm],
                     "samples_total_us": samples[arm], "banks_per_round": banks_used[arm],
                     "measurement": "MEASURED CUDA events, judge flush + 300 us gate; (kernel - empty)/units; "
                                    "MHz = 1000*clock64/globaltimer over CTA0 thread0's unit loop"})
    by = {r["arm"]: r for r in rows}
    b = by["base"]
    for r in rows:
        print(f"[f6] {r['arm']:14s} {r['us_per_unit']:.4f} us/unit ({r['p10']:.4f}..{r['p90']:.4f})  "
              f"{r['mhz_median']:.0f} MHz ({r['mhz_p10']:.0f}..{r['mhz_p90']:.0f})  dMHz {r['mhz_median'] - b['mhz_median']:+.0f}  "
              f"d us {r['us_per_unit'] - b['us_per_unit']:+.4f}", flush=True)
    report["rows"] = rows
    for arm in ARMS:   # separate continuous runs, outside the timed protocol
        g = graphs[arm][0]
        with torch.cuda.stream(timer.stream):
            pw = smi_sample(args.power_seconds, g.replay, torch)
        torch.cuda.synchronize()
        assert not err.any().item(), (arm, "protocol error in power run")
        cyc, ns_delta = marks.cpu().tolist()
        pw["hot_in_kernel_mhz_last_replay"] = 1000.0 * cyc / ns_delta if ns_delta > 0 else None
        by[arm]["power"] = pw
        print(f"[f6] power {arm:14s} {pw}", flush=True)
    report["reading"] = reading(rows, report["correctness"])
    print(f"[f6] reading {report['reading']['status']}", flush=True)


def summary_lines(report):
    rows = report.get("rows", [])
    rd = report.get("reading") or {}
    comp = report.get("compiler") or {}
    regs = comp.get("ptxas", {})
    corr = {c["arm"]: c for c in report.get("correctness", [])}
    L = ["# F6: CUDA-core (FFMA) consumer for 1-2 token experts vs the shipped HMMA consumer (NS=3, N8 shape, m16xf_micro.py)", "",
         f"status: {report.get('status')}  elapsed {report.get('elapsed_s', 0):.0f} s",
         f"Pre-registered: PASS if ffma1-class <= {PASS_US} us/unit AND MHz >= {PASS_MHZ:.0f} with correct outputs; KILL if > {KILL_US}; "
         f"ffma2-class PASS-2 if <= {PASS2_US}; VOID unless |twin-base| <= {AA_US} us/unit and <= {AA_MHZ:.0f} MHz.", "",
         f"**Reading: {rd.get('status', 'n/a')}**" + (f" (|twin-base| {rd['aa_us_per_unit']:.4f} us/unit, {rd['aa_mhz']:.1f} MHz)" if "aa_mhz" in rd else ""), "",
         "| arm | us/unit (p10..p90) | d us vs base | in-kernel MHz (p10..p90) | dMHz vs base | smi power W med/max | smi MHz | hot MHz | regs / spill B | correct | verdict |",
         "|---|---|---:|---|---:|---|---:|---:|---|---|---|"]
    by = {r["arm"]: r for r in rows}
    b = by.get("base")
    for r in rows:
        pw = r.get("power") or {}
        pr = regs.get(r["arm"], {})
        v = (rd.get("arms", {}).get(r["arm"], {}) or {}).get("status", "")
        L.append("| %s | %.4f (%.4f..%.4f) | %s | %.0f (%.0f..%.0f) | %s | %s | %s | %s | %s | %s | %s |" % (
            r["arm"], r["us_per_unit"], r.get("p10", r["us_per_unit"]), r.get("p90", r["us_per_unit"]),
            ("%+.4f" % (r["us_per_unit"] - b["us_per_unit"])) if b else "n/a",
            r["mhz_median"], r.get("mhz_p10", r["mhz_median"]), r.get("mhz_p90", r["mhz_median"]),
            ("%+.0f" % (r["mhz_median"] - b["mhz_median"])) if b else "n/a",
            ("%.0f/%.0f" % (pw["power_w_median"], pw.get("power_w_max", pw["power_w_median"]))) if pw.get("power_w_median") is not None else "n/a",
            ("%.0f" % pw["smi_sm_mhz_median"]) if pw.get("smi_sm_mhz_median") is not None else "n/a",
            ("%.0f" % pw["hot_in_kernel_mhz_last_replay"]) if pw.get("hot_in_kernel_mhz_last_replay") else "n/a",
            f"{pr.get('registers', '?')} / {pr.get('spill_store_bytes', '?')}", corr.get(r["arm"], {}).get("status", "n/a"), v))
    if comp.get("sass_class_counts_static"):
        L += ["", "SASS static instruction counts per class (one unit is straight-line; IMAD excludes IMAD.MOV, MOV includes it):", ""]
        L += class_table_lines(comp["sass_class_counts_static"])
        cl = comp.get("class_claims", {})
        L += ["", f"exact class claims: {cl.get('status')}"]
        for r in cl.get("rows", []):
            if r["status"] != "PASS":
                L.append(f"  FAIL {r['arm']}: {r.get('mismatch', r.get('why'))}")
    if comp.get("consume_window_per_unit_warp"):
        L += ["", "Per-unit consume window (first code LDS.128 .. d->ep STS.64; one warp, straight-line):", "",
              "| arm | instructions | ALU pipe (half rate) | FMA pipe | IMAD family | MOV | LDS | HMMA | HFMA2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            w = comp["consume_window_per_unit_warp"].get(arm)
            if w:
                L.append(f"| {arm} | {w['instructions']} | {w['alu_pipe']} | {w['fma_pipe']} | {w['imad_family']} | {w['mov']} | {w['lds']} | {w['hmma']} | {w['hfma2']} |")
    return L


def write_summary(report, path):
    path.write_bytes(("\n".join(summary_lines(report)) + "\n").encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="CPU gates only (numpy); writes nothing")
    ap.add_argument("--compile-only", action="store_true", help="CPU + compile + SASS gates; never launches CUDA work")
    ap.add_argument("--out", type=Path, default=Path("box_out/f6/results.json"))
    ap.add_argument("--units", type=int, default=UNITS)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--power-seconds", type=float, default=3.0)
    args = ap.parse_args()
    assert args.units == UNITS and args.rounds == 12 and args.power_seconds >= 3.0
    started = time.monotonic()
    out = args.out.resolve()
    report = {"task": "F6", "status": "RUNNING", "arms": list(ARMS), "modes": ARM_MODE,
              "pre_registered": {"pass_us_le": PASS_US, "pass_mhz_ge": PASS_MHZ, "kill_us_gt": KILL_US,
                                 "pass2_us_le": PASS2_US, "aa_us_le": AA_US, "aa_mhz_le": AA_MHZ}}
    claims_ok = True
    try:
        report["cpu"] = cpu_gates()
        if args.dry_run:
            print(json.dumps(report["cpu"], indent=2)); return
        out.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.set_num_threads(4)
        # Linux: the real base prepare functions with real CPU torch (inherited F3 check).
        test = mb.make_weights(1, 900)
        raw = test["w13"][0, :64, :512].copy(); sf = test["w13_sf"][0, :64, :64].copy()
        _, blocks = mb.base_pack_cpu(raw, sf)
        ns = {"torch": torch, "K2A_BITPOS": mb.BITPOS}
        for name, block in blocks.items():
            exec(compile(block, str(mb.BASE) + "::" + name, "exec"), ns)
        a, b = ns["_k2a31_prepack_expert"](torch.from_numpy(raw), torch.from_numpy(sf), False, mb.BITPOS)
        assert np.array_equal(np.concatenate((a.numpy().view(np.uint8).reshape(-1), b.numpy().reshape(-1))), mb.pack_tiles(raw[None], sf[None])[0])
        report["cpu"]["real_torch_base_prepare"] = "PASS"
        mod, report["compiler"], table = build(out)
        print("[f6] SASS static class counts per arm:\n" + "\n".join(class_table_lines(table)), flush=True)
        print("[f6] ptxas " + json.dumps({a: (r["registers"], r["spill_store_bytes"], r["stack_bytes"]) for a, r in report["compiler"]["ptxas"].items()}), flush=True)
        print("[f6] consume window " + json.dumps({a: (w or {}).get("instructions") for a, w in report["compiler"]["consume_window_per_unit_warp"].items()}), flush=True)
        print("[f6] compile gates " + json.dumps(report["compiler"]["gates"]), flush=True)
        claims = report["compiler"]["class_claims"]
        claims_ok = claims["status"] == "PASS"
        print(f"[f6] exact class claims {claims['status']}" + "".join(
            f"\n  FAIL {r['arm']}: {r.get('mismatch', r.get('why'))}" for r in claims["rows"] if r["status"] != "PASS"), flush=True)
        assert report["compiler"]["status"] == "PASS", report["compiler"]["gates"]
        if args.compile_only:
            report["status"] = "COMPILE_GATES_PASS" if claims_ok else "COMPILE_PASS_CLAIMS_FAIL"
            print(f"[f6] {report['status']}; JSON={out}", flush=True)
        else:
            benchmark(mod, args, report)
            report["status"] = "COMPLETE" if claims_ok else "COMPLETE (exact class claims FAILED: read the table)"
    except Exception as ex:  # noqa: BLE001
        report["status"] = "FAIL"
        report["error"] = repr(ex)
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(json.dumps(report, indent=2, default=str).encode("utf-8"))
            if not args.compile_only:
                write_summary(report, out.with_name("summary.md"))
    if args.compile_only and not claims_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

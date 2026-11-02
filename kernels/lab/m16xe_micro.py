#!/usr/bin/env python3
"""F5b: instruction-class power attribution arms on the F3 consumer-pace micro (shipped ms consumer, NS=3, N=8/16).

Every arm is one compile-time MODE of the unchanged F3 kernel template m16xb<N, MODE, NS> (imported verbatim from
kernels/lab/m16xb_micro.py; pinned by sha256): same ring / mbarrier / bulk-copy protocol, same weight stream, same
epilogue.  Only the straight-line consume body differs.  Outputs of every arm except base/twin are wrong by design.

  MODE 0  null          ring only (clock reference)
  MODE 3  base          shipped ms consumer (F3 'ms', byte-identical consume body)
  MODE 4  twin          A/A copy of base (separate template instance, consume<N, 3>)
  MODE 5  hmma_sink     every mma.sync -> mad.lo.u32 into the FP32 accumulator words viewed as u32; every A register
                        feeds one mad (N=16: one mad per MMA; N=8: two per MMA, see f5b_impl.md); B LDS kept volatile
  MODE 6  dq_mov        dec_half2's dequant ALU chain -> 24 asm volatile mov.b32 per call writing the 8 A registers
                        from the packed code/scale words (ptxas copy-propagates them: SASS keeps only placement MOVs);
                        code/scale LDS and MMAs kept
  MODE 7  lds_w1        weight code (LDS.128) / scale (LDS.32) loads: every lane reads lane 0's address (broadcast)
  MODE 8  lds_x1        B-fragment (x) LDS.32: every lane reads lane 0's address (ms_pair called with lane = 0)
  MODE 9  hmma_dq_sink  MODE 5 + MODE 6 (no tensor math, no dequant math)
  MODE 10 dq_eq         dequant ALU chain -> the SAME number (800 / unit) of PRMT with a run-time identity selector that
                        the compilers cannot see (A registers == dq_mov's at run time; issue count == base)

CPU:    python -B kernels/lab/m16xe_micro.py --dry-run                       (numpy only, writes nothing)
Gates:  bash tools/local/f5b_gates.sh   (WSL; CPU + compile + SASS class-count claims; never launches CUDA work)
Pod:    python3 -u -B kernels/lab/m16xe_micro.py --out box_out/f5b/results.json   (via tools/box/f5b_batch.sh)
"""
import argparse
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
NS_ARMS = (8, 16)
UNITS = 48
ARM_MODE = {"null": 0, "base": 3, "twin": 4, "hmma_sink": 5, "dq_mov": 6, "lds_w1": 7, "lds_x1": 8,
            "hmma_dq_sink": 9, "dq_eq": 10}
MODE_ARM = {v: k for k, v in ARM_MODE.items()}
ARM_ORDER = ("null", "base", "twin", "hmma_sink", "dq_mov", "dq_eq", "lds_w1", "lds_x1", "hmma_dq_sink")
ARMS = tuple((a, n) for n in NS_ARMS for a in ARM_ORDER)
CONSUME_MODES = (3, 5, 6, 7, 8, 9, 10)          # twin (4) maps to consume<N, 3> inside the F3 kernel
ATTRIB_ARMS = ("hmma_sink", "dq_mov", "dq_eq", "lds_w1", "lds_x1", "hmma_dq_sink")
CLASS_OF = {"hmma_sink": "HMMA (tensor pipe)", "dq_mov": "dequant ALU removed (LOP3/SHF/IMAD/HMUL2/PRMT/LEA)",
            "dq_eq": "dequant ALU -> equal-count identity PRMT", "lds_w1": "weight LDS bank/data (broadcast)",
            "lds_x1": "x LDS bank/data (broadcast)", "hmma_dq_sink": "no-math floor (HMMA + dequant removed)"}
# The imported F3 sources this file builds on; any edit to m16xb_micro.py must be re-reviewed before F5b runs.
PINS = {"CUDA_HEAD": "72d8c8df15153ea041509cfd59b362d5f0b96c958971f8495b213acca0cf8a84",
        "DEQUANT": "68d9eb97f2b8070ae4e0f2b1576ca51bf6d274051c80f62e2e1ad8747b4d7116",
        "CPP": "fe848d62c0be42ea76204dbaa03dbd6882ad969ef91269b8cbefb5aabd1cdca8"}
# Pre-registered diagnostic reading (lead brief): a class moves the clock if its arm raises in-kernel MHz by >= 40 vs
# base with |twin - base| <= 15 MHz; flag any arm >= 1850 MHz.
MOVE_MHZ, AA_MHZ, FLAG_MHZ = 40.0, 15.0, 1850.0

# Exactly two edits of the F3 kernel text: consume() also receives `units` (an opaque run-time value for dq_eq's
# identity selector; base/twin never read it).
REPLACEMENTS = (
    ("template<int N, int MODE> __device__ __forceinline__ void consume(float* d, const uint8_t* st, uint16_t* x, int wp, int lane) {",
     "template<int N, int MODE> __device__ __forceinline__ void consume(float* d, const uint8_t* st, uint16_t* x, int wp, int lane, int units) {"),
    ("consume<N, (MODE == 4 ? 3 : MODE)>(d, s + XS + slot * UB, reinterpret_cast<uint16_t*>(s), wp, lane);",
     "consume<N, (MODE == 4 ? 3 : MODE)>(d, s + XS + slot * UB, reinterpret_cast<uint16_t*>(s), wp, lane, units);"),
)

F5B_HELPERS = r"""
// ---- F5b attribution helpers (MODE 5..10 only; base/twin use the F3 code above/below unchanged) ----
__device__ __forceinline__ void ld32v(uint32_t a) {
  // Volatile shared load with no consumer: ptxas keeps the LDS (same count/width) without adding a consumer op.
  // (Empty asm volatile "consumers" do not survive ptxas: it deletes their producers; see f5b_impl.md.)
  uint32_t v; asm volatile("ld.volatile.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
}
__device__ __forceinline__ void hsink(float& acc, uint32_t a, uint32_t b) {
  // One integer multiply-add in place of tensor math, accumulating into the FP32 accumulator word viewed as u32.
  uint32_t v = __float_as_uint(acc);
  asm volatile("mad.lo.u32 %0, %1, %2, %0;" : "+r"(v) : "r"(a), "r"(b));
  acc = __uint_as_float(v);
}
template<int N, int Q> __device__ __forceinline__ void ms_pair_sink(float* d, const uint32_t* h, const uint16_t* x, int lane) {
  // ms_pair with every mma.sync removed: the same four B LDS.32 per (n8 tile) at the same addresses (volatile, no
  // consumer), and each of the 8 A registers feeds exactly one mad (N=16: 1 mad per MMA; N=8: 2 per MMA).
  const int g = lane >> 2, t = lane & 3;
  const uint32_t xb = sa(x + (2 * Q) * N * 16 + g * 8 + t * 2);
  #pragma unroll
  for (int j = 0; j < N / 8; ++j) {
    ld32v(xb + j * 256); ld32v(xb + j * 256 + 128); ld32v(xb + j * 256 + N * 32); ld32v(xb + j * 256 + N * 32 + 128);
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
      // A fragment of k16 step s, in ms_pair's operand order {h[2s], h[4+2s], h[2s+1], h[5+2s]}.
      const uint32_t a[4] = {h[2 * s], h[4 + 2 * s], h[2 * s + 1], h[5 + 2 * s]};
      if constexpr (N == 8) {
        hsink(d[2 * s], a[0], a[1]); hsink(d[2 * s + 1], a[2], a[3]);
      } else {
        hsink(d[4 * j + ((2 * (Q & 1) + s) & 3)], a[2 * j], a[2 * j + 1]);
      }
    }
  }
}
__device__ __forceinline__ uint32_t mv(uint32_t a) {
  uint32_t r; asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(a)); return r;
}
// A-register sources of the dq arms (packed code words w0/w8 of this call and its scale word sc):
//   h0=w0 h1=w0 h2=w8 h3=w0 h4=w8 h5=sc h6=sc h7=w0.  Every code and scale word reaches an A register (so the code
//   LDS.128 and scale LDS.32 keep their base form), and the four A pairs that ms_pair_sink multiplies,
//   (h0,h4) (h1,h5) (h2,h6) (h3,h7) = (w0,w8) (w0,sc) (w8,sc) (w0,w0), are distinct: no mad can be CSE'd.
__device__ __forceinline__ void dq_mov8(uint32_t w0, uint32_t w8, uint32_t sc, uint32_t* hh) {
  // dec_half2's 24 per-call dequant ALU ops (10 LOP3, 4 SHF, 2 IMAD, 8 HMUL2) -> 24 mov.b32, 3 per A register.
  // ptxas copy-propagates register moves: only the MOVs register placement needs survive (gate table).
  const uint32_t src[8] = {w0, w0, w8, w0, w8, sc, sc, w0};
  #pragma unroll
  for (int j = 0; j < 8; ++j) hh[j] = mv(mv(mv(src[j])));
}
__device__ __forceinline__ uint32_t pid(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r; asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r;
}
template<int SUB> __device__ __forceinline__ uint32_t dq_eq_scale(uint32_t s, uint32_t sel) {
  // The per-(sg, sub) scale PRMT + LEA that dec_half2's two calls share -> 2 identity PRMT (distinct per SUB).
  return pid(pid(s, SUB ? sel : s, sel), s, sel);
}
__device__ __forceinline__ void dq_eq8(uint32_t w0, uint32_t w8, uint32_t sp, uint32_t sel, uint32_t* hh) {
  // 24 PRMT per call, 3 per A register.  sel == 0x3210 (identity) at run time but opaque to nvcc/ptxas, so hh equals
  // dq_mov8's hh (sp == the scale word) while the issue count equals base's dequant.  Registers that share a source
  // take distinct first b-operands (k-th of its source group -> o[k]), so no PRMT is CSE'd.
  const uint32_t src[8] = {w0, w0, w8, w0, w8, sp, sp, w0};
  const int k[8] = {0, 1, 0, 2, 1, 0, 1, 3};
  const uint32_t o[4] = {w0, w8, sp, sel};
  #pragma unroll
  for (int j = 0; j < 8; ++j)
    hh[j] = pid(pid(pid(src[j], o[k[j]], sel), o[(k[j] + 1) & 3], sel), o[(k[j] + 2) & 3], sel);
}
"""


def consume_body(mode):
    """Straight-line body of one consume MODE (one 64x1024 unit per warp). MODE 3 is F3's ms body, byte for byte."""
    assert mode in CONSUME_MODES
    lines = ["uint32_t h[8];"]
    if mode == 10:
        lines += ['uint32_t z; asm volatile("mov.b32 %0, %1;" : "=r"(z) : "r"(units));',
                  "const uint32_t sel = 0x3210u + (z >> 31);  // 0x3210 (PRMT identity) at run time; opaque to nvcc/ptxas"]
    lane_w = "" if mode == 7 else " + lane"
    for sg in range(8):
        lines += ["{", f"const uint32_t scale = reinterpret_cast<const uint32_t*>(st + 32768 + wp * 1024)[{sg} * 32{lane_w}];"]
        for sub in range(2):
            kb = 2 * sg + sub
            lines += ["{", f"const uint4 c = reinterpret_cast<const uint4*>(st + wp * 8192)[{kb} * 32{lane_w}];"]
            if mode == 10:
                lines.append(f"const uint32_t sp = dq_eq_scale<{sub}>(scale, sel);")
            for hf in range(2):
                q = 2 * kb + hf
                w0, w8 = f"c.{('x', 'y')[hf]}", f"c.{('z', 'w')[hf]}"
                if mode in (6, 9):
                    lines.append(f"dq_mov8({w0}, {w8}, scale, h);")
                elif mode == 10:
                    lines.append(f"dq_eq8({w0}, {w8}, sp, sel, h);")
                else:
                    lines.append(f"dec_half2({w0}, {w8}, scale, {sub}u, h, 16u, 64u, 4u);")
                if mode in (5, 9):
                    lines.append(f"ms_pair_sink<N, {q}>(d, h, x, lane);")
                elif mode == 8:
                    lines.append(f"ms_pair<N, {q}>(d, h, x, 0);")
                else:
                    lines.append(f"ms_pair<N, {q}>(d, h, x, lane);")
            lines.append("}")
        lines.append("}")
    return lines


def generate_cuda():
    """One binary: 18 instances m16xb<N, MODE, 3>; MODE and N resolve at instantiation, never in the stage loop."""
    head = mb.CUDA_HEAD
    for old, new in REPLACEMENTS:
        assert head.count(old) == 1, ("F3 anchor moved", old)
        head = head.replace(old, new)
    pieces = []
    for i, mode in enumerate(CONSUME_MODES):
        pieces.append(f"{'if' if i == 0 else 'else if'} constexpr (MODE == {mode}) {{")
        pieces += consume_body(mode)
        pieces.append("}")
    info, dispatch = [], []
    for arm, n in ARMS:
        mode = ARM_MODE[arm]
        cond = f"if (mode == {mode} && n == {n} && ns == {NS})"
        info.append(f"{cond} return config<{n}, {mode}, {NS}>(true);")
        dispatch.append(f"{cond} {{ m16xb<{n}, {mode}, {NS}><<<grid,384,SMEM<{n}, {NS}>,st>>>((const uint8_t*)w,(const uint16_t*)x,(float*)o,(int*)sm,(int*)err,(uint64_t*)marks,units); ck(cudaGetLastError()); return; }}")
    src = head.replace("@CONSUME@", "\n".join(pieces))
    src = src.replace("@INFO@", "\n".join(info)).replace("@DISPATCH@", "\n".join(dispatch))
    anchor = "template<int N, int Q> __device__"
    assert src.count(anchor) == 2  # rs_pair and ms_pair (rs_pair is never instantiated: no wgmma in this binary)
    return src.replace(anchor, mb.DEQUANT + F5B_HELPERS + "\n" + anchor, 1)


def launch_arm(mod, key, units, grid, w, x, out, sm, err, marks, stream):
    arm, n = key
    assert key in ARMS
    mod.launch(ARM_MODE[arm], n, NS, units, grid, w.data_ptr(), x.data_ptr(), out.data_ptr(), sm.data_ptr(),
               err.data_ptr(), marks.data_ptr(), stream)


def bounds_key(key):
    """Global access sites: every compute arm issues exactly F3 ms's (bulk W, X init, epilogue O, marks/sm/err)."""
    return ("null" if key[0] == "null" else "ms", key[1], NS)


def replay_schedule(rounds):
    for rnd in range(rounds):
        order = ARMS[rnd % len(ARMS):] + ARMS[:rnd % len(ARMS)]
        for key in order:
            yield rnd, key, (2 * rnd + ARMS.index(key)) % 3


# ----------------------------------------------------------------------------------------------- SASS classes / claims
# Signature counts identify the instructions each arm removes/adds (operand patterns of the base lowering, checked
# on the base itself): dequant = per dec_half2 10 LOP3 (masks 0x81c081c0 / 0x1b001b), 4 SHF.L.W (rotate 10/13),
# 2 IMAD x 0x1084, 8 HMUL2, + per (sg, sub) 1 PRMT and 1 LEA (scale * 16 + 0x75807580); x32 / x16 per unit = 800.
SIGNATURES = (
    ("deq_LOP3", re.compile(r"^LOP3\.LUT R\d+, R\d+, (0x81c081c0|0x1b001b), ")),
    ("deq_SHF", re.compile(r"^SHF\.L\.W\.U32(\.HI)? R\d+, R\d+, 0x[ad], R\d+")),
    ("deq_IMAD", re.compile(r"^IMAD R\d+, R\d+, 0x1084, RZ")),
    ("scale_LEA", re.compile(r"^LEA\S* R\d+, R\d+, 0x75807580, ")),
    ("sink_IMAD", re.compile(r"^IMAD R\d+, R\d+, R\d+, (R\d+|RZ)$")),   # register x register (+ acc): the mad sinks
)
DQ_SIG = {"deq_LOP3": 320, "deq_SHF": 128, "deq_IMAD": 64, "HMUL2": 256, "PRMT": 16, "scale_LEA": 16}
TABLE_CLASSES = ("HMMA", "sink_IMAD", "IMAD", "deq_LOP3", "LOP3", "deq_SHF", "SHF", "deq_IMAD", "PRMT", "HMUL2",
                 "scale_LEA", "LEA", "LDS", "LDS.128", "LDS.32", "MOV", "IADD3", "TOTAL")
EXACT = ("HMMA", "sink_IMAD", "deq_LOP3", "deq_SHF", "deq_IMAD", "PRMT", "HMUL2", "scale_LEA", "LDS.32", "LDS.64", "LDS.128")


def sass_functions(sass):
    """Static (not dynamic) instructions per SASS function: (opcode counts incl. modifiers, normalized texts)."""
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
    """IMAD excludes IMAD.MOV* (a register copy, counted under MOV); LDS split by access width; signature counts."""
    c = {k: 0 for k in TABLE_CLASSES + ("LDS.64",)}
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
    for t in texts:
        for name, rx in SIGNATURES:
            if rx.match(t):
                c[name] += 1
    return c


def expected(arm, n):
    """Pre-registered static counts per unit (absolute) for the EXACT classes; LDS widths are 'equal to base'."""
    e = dict(DQ_SIG) if arm in ("base", "twin", "hmma_sink", "lds_w1", "lds_x1") else {k: 0 for k in DQ_SIG}
    if arm == "dq_eq":
        e["PRMT"] = 800               # 32 x 24 + 16 x 2 run-time-identity PRMT == base's 800 dequant ALU ops
    e["HMMA"] = 0 if arm in ("hmma_sink", "hmma_dq_sink") else 8 * n     # (n/8) n8 tiles x 2 k16 x 32 q per unit
    e["sink_IMAD"] = 128 if arm in ("hmma_sink", "hmma_dq_sink") else 0  # N=16: 1 per removed MMA; N=8: 2 per MMA
    return e


def class_claims(table):
    """table: {(arm, n): class counts}. Exact claims; everything else (address arithmetic, MOV) is reported."""
    rows, ok = [], True
    for n in NS_ARMS:
        base, twin = table.get(("base", n)), table.get(("twin", n))
        if base is None or twin is None:
            return {"status": "FAIL", "why": "missing base/twin", "rows": []}
        same = twin == base
        rows.append({"arm": "twin", "n": n, "claim": "every opcode count identical to base", "status": "PASS" if same else "FAIL"})
        ok &= same
        for arm in ("base",) + ATTRIB_ARMS:
            got = table.get((arm, n))
            if got is None:
                rows.append({"arm": arm, "n": n, "status": "FAIL", "why": "missing"}); ok = False; continue
            exp = expected(arm, n)
            exp.update({k: base.get(k, 0) for k in ("LDS.32", "LDS.64", "LDS.128")})
            bad = {k: {"expected": v, "actual": got.get(k, 0)} for k, v in exp.items() if got.get(k, 0) != v}
            sig = sum(got.get(k, 0) for k in EXACT) + got["MOV"]
            other = (got["TOTAL"] - sig) - (base["TOTAL"] - sum(base.get(k, 0) for k in EXACT) - base["MOV"])
            rows.append({"arm": arm, "n": n, "status": "PASS" if not bad else "FAIL", "mismatch": bad, "claimed": exp,
                         "reported_delta_vs_base": {k: got.get(k, 0) - base.get(k, 0) for k in TABLE_CLASSES},
                         "other_instruction_delta": other})
            ok &= not bad
    return {"status": "PASS" if ok else "FAIL", "rows": rows}


def class_table_lines(table):
    head = "| arm | " + " | ".join(TABLE_CLASSES) + " |"
    lines = [head, "|---|" + "---:|" * len(TABLE_CLASSES)]
    for arm, n in ARMS:
        c = table.get((arm, n))
        if c:
            lines.append(f"| {arm}_n{n} | " + " | ".join(str(c.get(k, 0)) for k in TABLE_CLASSES) + " |")
    return lines


# ------------------------------------------------------------------------------------------------------ CPU gates
def prmt_ref(a, b, s):
    """PTX prmt.b32 default mode, independent of the CUDA source (for the dq_eq identity claim)."""
    by = [(a >> (8 * i)) & 255 for i in range(4)] + [(b >> (8 * i)) & 255 for i in range(4)]
    out = 0
    for i in range(4):
        nib = (s >> (4 * i)) & 15
        v = by[nib & 7]
        if nib & 8:
            v = 255 if v & 128 else 0
        out |= v << (8 * i)
    return out


def b_offsets_ms_pair(n, q, lane):
    g, t = lane >> 2, lane & 3
    out = []
    for j in range(n // 8):
        base = 2 * ((2 * q) * n * 16 + j * 128 + g * 8 + t * 2)   # bytes; b points at uint32 words
        out += [base + 4 * k for k in (0, 32, n * 8, n * 8 + 32)]
    return out


def b_offsets_sink(n, q, lane):
    g, t = lane >> 2, lane & 3
    xb = 2 * ((2 * q) * n * 16 + g * 8 + t * 2)
    return [xb + j * 256 + k for j in range(n // 8) for k in (0, 128, n * 32, n * 32 + 128)]


def sink_operands(n, q):
    """Replay ms_pair_sink's mad schedule: which A registers / accumulator words each mad touches."""
    used, words = [], []
    for j in range(n // 8):
        for s in range(2):
            a = (2 * s, 4 + 2 * s, 2 * s + 1, 5 + 2 * s)
            if n == 8:
                used += [(a[0], a[1]), (a[2], a[3])]; words += [2 * s, 2 * s + 1]
            else:
                used.append((a[2 * j], a[2 * j + 1])); words.append(4 * j + ((2 * (q & 1) + s) & 3))
    return used, words


def reading(rows):
    """Pre-registered diagnostic reading (lead brief F5b)."""
    by = {(r["arm"], r["n"]): r for r in rows}
    res = {}
    for n in NS_ARMS:
        b, t = by.get(("base", n)), by.get(("twin", n))
        if not b or not t:
            res[str(n)] = {"status": "VOID: base/twin missing"}; continue
        aa = abs(t["mhz_median"] - b["mhz_median"])
        valid = aa <= AA_MHZ
        bp = (b.get("power") or {}).get("power_w_median")
        items = []
        for arm in ATTRIB_ARMS + ("null",):
            r = by.get((arm, n))
            if not r:
                continue
            p = (r.get("power") or {}).get("power_w_median")
            dm = r["mhz_median"] - b["mhz_median"]
            items.append({"arm": arm, "class": CLASS_OF.get(arm, "reference"), "d_mhz": dm,
                          "d_us_per_unit": r["us_per_unit"] - b["us_per_unit"],
                          "d_power_w": (p - bp) if (p is not None and bp is not None) else None,
                          "moves_clock": bool(valid and dm >= MOVE_MHZ and arm != "null"),
                          "ge_1850_mhz": r["mhz_median"] >= FLAG_MHZ})
        attrib = [i for i in items if i["arm"] != "null"]
        res[str(n)] = {"aa_mhz": aa, "aa_valid": valid, "base_mhz": b["mhz_median"], "items": items,
                       "rank_by_d_mhz": [i["arm"] for i in sorted(attrib, key=lambda i: -i["d_mhz"])],
                       "rank_by_d_power": [i["arm"] for i in sorted([i for i in attrib if i["d_power_w"] is not None],
                                                                    key=lambda i: i["d_power_w"])],
                       "moves_clock": [i["arm"] for i in attrib if i["moves_clock"]],
                       "flag_ge_1850": [i["arm"] for i in items if i["ge_1850_mhz"]]}
    return res


def cpu_gates():
    inherited = mb.cpu_gates()          # F3: real base prepare, verbatim helpers vs m16xa, ring/protocol replays, bounds
    for k, h in PINS.items():
        assert hashlib.sha256(getattr(mb, k).encode()).hexdigest() == h, ("m16xb_micro.py changed", k)
    src = generate_cuda()
    assert "@" not in src and "__trap" not in src and "wgmma.mma_async" not in src  # rs_pair never instantiated
    assert len(set(ARMS)) == len(ARMS) == 18 and len(set(ARM_MODE.values())) == len(ARM_MODE)
    assert len(re.findall(r"m16xb<\d+, \d+, 3><<<", src)) == len(ARMS)
    assert len(re.findall(r"return config<\d+, \d+, 3>\(true\);", src)) == len(ARMS)
    for arm, n in ARMS:
        assert src.count(f"if (mode == {ARM_MODE[arm]} && n == {n} && ns == {NS})") == 2, (arm, n, "info+dispatch")
    # base consume body is F3's ms body byte for byte; the F3 kernel differs only by the two REPLACEMENTS.
    mb_src = mb.generate_cuda()
    block = "\n".join(consume_body(3)) + "\n}"
    assert ("else if constexpr (MODE == 3) {\n" + block) in mb_src, "base consume differs from F3 ms"
    assert ("if constexpr (MODE == 3) {\n" + block) in src
    kern = lambda s: s[s.index("template<int N, int MODE, int NS> __global__"):s.index("__global__ void empty_kernel")]
    a, b = kern(mb.CUDA_HEAD).splitlines(), kern(src).splitlines()
    assert len(a) == len(b) and [i for i in range(len(a)) if a[i] != b[i]] == [next(i for i, l in enumerate(a) if "consume<N, (MODE == 4" in l)]
    pairs = mb.CUDA_HEAD[mb.CUDA_HEAD.index("template<int N, int Q> __device__ __forceinline__ void rs_pair"):mb.CUDA_HEAD.index("template<int N, int MODE> __device__")]
    assert pairs in src and mb.DEQUANT in src, "ms_pair / dequant not verbatim"
    # per-mode source accounting (EST, exact at source level; SASS claims are checked by the compile gate)
    acct = {}
    for mode in CONSUME_MODES:
        body = "\n".join(consume_body(mode))
        cnt = {k: body.count(k) for k in ("dec_half2(", "ms_pair<N,", "ms_pair_sink<N,", "dq_mov8(", "dq_eq8(",
                                          "dq_eq_scale<", "ld32v(", "reinterpret_cast<const uint4*>", "reinterpret_cast<const uint32_t*>")}
        assert cnt["reinterpret_cast<const uint4*>"] == 16, mode                       # 16 code LDS.128 / unit
        assert cnt["reinterpret_cast<const uint32_t*>"] + cnt["ld32v("] == 8, mode     # 8 scale LDS.32 / unit
        assert cnt["dec_half2("] + cnt["dq_mov8("] + cnt["dq_eq8("] == 32, mode
        assert cnt["ms_pair<N,"] + cnt["ms_pair_sink<N,"] == 32, mode
        assert (cnt["ms_pair_sink<N,"] == 32) == (mode in (5, 9)) and (cnt["dq_mov8("] == 32) == (mode in (6, 9))
        assert (cnt["dq_eq8("] == 32 and cnt["dq_eq_scale<"] == 16) == (mode == 10)
        assert (" + lane]" not in body) == (mode == 7) and ("(d, h, x, 0);" in body) == (mode == 8)
        acct[MODE_ARM[mode]] = cnt
    # ms_pair_sink: identical B addresses per lane to ms_pair; every A register exactly once; all accumulator words.
    for n in NS_ARMS:
        hmma = 0
        for q in range(32):
            for lane in range(32):
                assert sorted(b_offsets_ms_pair(n, q, lane)) == sorted(b_offsets_sink(n, q, lane))
                assert max(b_offsets_ms_pair(n, q, lane)) + 4 <= n * 2048
            used, words = sink_operands(n, q)
            assert sorted(r for p in used for r in p) == list(range(8)), (n, q)
            hmma += 2 * (n // 8)
            assert len(used) == 4
        w = set()
        for q in (0, 1):
            w |= set(sink_operands(n, q)[1])
        assert w == set(range(n // 2)), (n, "accumulator words")
        assert hmma == 8 * n
    # dq arms: both A-register maps are the same; every packed word reaches an A register; the A pairs that the
    # sink mads multiply are pairwise distinct (no CSE of products in hmma_dq_sink).
    maps = re.findall(r"const uint32_t src\[8\] = \{([^}]*)\};", F5B_HELPERS)
    assert len(maps) == 2
    m_mov, m_eq = ([s.strip() for s in m.split(",")] for m in maps)
    assert [("sc" if s == "sp" else s) for s in m_eq] == m_mov and set(m_mov) == {"w0", "w8", "sc"}
    ks = [int(v) for v in re.search(r"const int k\[8\] = \{([^}]*)\};", F5B_HELPERS)[1].split(",")]
    firsts = [(m_eq[j], ks[j]) for j in range(8)]
    assert len(set(firsts)) == 8, "dq_eq first PRMT steps must be distinct"
    for n in NS_ARMS:
        for q in range(2):
            used, _ = sink_operands(n, q)
            prods = [tuple(sorted((m_mov[a], m_mov[b]))) for a, b in used]   # multiplication commutes
            assert len(set(prods)) == len(prods) == 4, (n, q, prods)
    # lds_w1 / lds_x1 addresses are lane 0's base addresses (subset of base's, in bounds).
    for wp in range(4):
        for kb in range(16):
            assert wp * 8192 + kb * 32 * 16 in {wp * 8192 + (kb * 32 + l) * 16 for l in range(32)}
    # dq_eq: identity selector reproduces dq_mov's A registers; the run-time selector is never anything else.
    rng = np.random.default_rng(5)
    for a_, b_ in rng.integers(0, 2**32, (2000, 2), dtype=np.uint64).tolist():
        assert prmt_ref(a_, b_, 0x3210 + ((48 & 0xFFFFFFFF) >> 31)) == a_
    assert prmt_ref(0x11223344, 0x55667788, 0x7654) == 0x55667788  # reference sanity: selects b
    assert prmt_ref(0x11223344, 0x55667788, 0x0123) == 0x44332211  # byte reverse of a
    # timing schedule: every consecutive replay changes bank; every key sees each bank 4x in 12 rounds.
    sched = list(replay_schedule(12))
    assert all(x[2] != y[2] for x, y in zip(sched, sched[1:]))
    assert all([sum(k == key and bk == bank for _, k, bk in sched) for bank in range(3)] == [4, 4, 4] for key in ARMS)
    # global bounds through the F3 replay (identical access sites for every compute arm).
    class Tensor:
        def __init__(self, count, width): self.count, self.width = count, width
        def numel(self): return self.count
        def element_size(self): return self.width
        def is_contiguous(self): return True
        def data_ptr(self): return 0x100000000
    bounds = {}
    for key in ARMS:
        arm_mb, n, ns = bounds_key(key)
        r = mb.tensor_bounds(arm_mb, n, UNITS, mb.GRID, Tensor(mb.GRID * UNITS * mb.UNIT, 1), Tensor(n * 1024, 2),
                             Tensor(mb.GRID * UNITS * 64 * 32, 4), Tensor(mb.GRID, 4), Tensor(mb.GRID, 4), Tensor(2, 8), ns=ns)
        bounds[f"{key[0]}_n{key[1]}"] = r["status"]
    # SASS parser + claim checker on synthetic fixtures (never presented as measurements).
    fx = ("Function : f\n /*0000*/ HMMA.16816.F32.BF16 R0, R4.reuse, R8, R0 ;  /* 0x0000001c201c723c */\n"
          " /*0010*/ IMAD.MOV.U32 R1, RZ, RZ, R2 ;\n /*0020*/ @!P0 LDS.128 R4, [R1] ;\n /*0030*/ LDS R5, [R2+0x80] ;\n"
          " /*0040*/ IMAD R3, R3.reuse, 0x1084, RZ ;\n /*0050*/ IMAD R7, R9.reuse, R10, R7 ;\n /*0060*/ IMAD R6, R9, R11, RZ ;\n"
          " /*0070*/ LOP3.LUT R2, R22, 0x81c081c0, RZ, 0xc0, !PT ;\n /*0080*/ SHF.L.W.U32.HI R24, R23, 0xa, R23 ;\n"
          " /*0090*/ LEA R37, R34, 0x75807580, 0x4 ;\n /*00a0*/ IMAD R6, R9, UR4, RZ ;\n /*00b0*/ LOP3.LUT R2, R22, 0x3fc, RZ, 0xc0, !PT ;\n")
    c = classify(*sass_functions(fx)["f"])
    assert (c["HMMA"], c["MOV"], c["IMAD"], c["LDS"], c["LDS.128"], c["LDS.32"], c["TOTAL"]) == (1, 1, 4, 2, 1, 1, 12)
    assert (c["deq_IMAD"], c["sink_IMAD"], c["deq_LOP3"], c["LOP3"], c["deq_SHF"], c["scale_LEA"]) == (1, 2, 1, 2, 1, 1)
    synth = {}
    for n in NS_ARMS:
        filler = {k: 1000 for k in TABLE_CLASSES}
        filler.update({"LDS.64": 0, "MOV": 40})
        for arm in ("base", "twin") + ATTRIB_ARMS:
            got = dict(filler); got.update(expected(arm, n))
            synth[(arm, n)] = got
    assert class_claims(synth)["status"] == "PASS"
    for key, field in ((("dq_eq", 16), "PRMT"), (("twin", 8), "MOV"), (("hmma_sink", 8), "sink_IMAD"),
                       (("lds_w1", 16), "LDS.128"), (("base", 8), "deq_SHF")):
        bad = {k: dict(v) for k, v in synth.items()}
        bad[key][field] += 1
        if key[0] == "base":
            bad[("twin", key[1])][field] += 1
        assert class_claims(bad)["status"] == "FAIL", (key, field)
    # reading boundaries on synthetic rows.
    def fx_rows(aa, dm):
        rows = []
        for n in NS_ARMS:
            for arm in ARM_ORDER:
                m = 1730.0 + (aa if arm == "twin" else 0) + (dm if arm == "hmma_sink" else 0) + (250 if arm == "null" else 0)
                rows.append({"arm": arm, "n": n, "mhz_median": m, "us_per_unit": 1.3, "power": {"power_w_median": 690.0}})
        return rows
    assert reading(fx_rows(15, 40))["16"]["moves_clock"] == ["hmma_sink"]
    assert reading(fx_rows(15.01, 40))["16"]["moves_clock"] == []
    assert reading(fx_rows(0, 39.99))["8"]["moves_clock"] == []
    assert reading(fx_rows(0, 120))["8"]["flag_ge_1850"] == ["hmma_sink", "null"]
    summary_lines({"status": "SYNTHETIC", "rows": fx_rows(0, 50)})   # writer exercised, nothing written
    return {"status": "PASS", "inherited_f3_cpu_gates": inherited["status"], "pins": PINS,
            "arms": [f"{a}_n{n}" for a, n in ARMS], "modes": ARM_MODE, "source_accounting_per_unit": acct,
            "global_bounds": bounds, "kernel_text_edits_vs_f3": [old for old, _ in REPLACEMENTS],
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
        blocks[(MODE_ARM.get(mode, f"mode{mode}"), n)] = {
            "symbol": m[1], "n": n, "mode": mode, "ns": ns, "registers": int(regs[1]) if regs else None,
            "stack_bytes": int(spill[1]) if spill else None, "spill_store_bytes": int(spill[2]) if spill else None,
            "spill_load_bytes": int(spill[3]) if spill else None, "static_smem_bytes": int(smem[1]) if smem else 0}
    table, raw = {}, {}
    for sym, (ops, texts) in sass_functions(sass).items():
        inst = re.search(r"m16xbILi(\d+)ELi(\d+)ELi(\d+)E", sym)
        if inst:
            key = (MODE_ARM.get(int(inst[2]), f"mode{inst[2]}"), int(inst[1]))
            table[key], raw[key] = classify(ops, texts), ops
    claims = class_claims(table)
    k = list(blocks.values())
    gates = {
        "all_instances": set(blocks) == set(ARMS) and set(table) == set(ARMS),
        "no_spills": bool(k) and all(r["spill_store_bytes"] is not None and r["spill_store_bytes"] <= 16 and r["spill_load_bytes"] <= 16 for r in k),
        "no_stack": bool(k) and all(r["stack_bytes"] is not None and r["stack_bytes"] <= 16 for r in k),
        "registers_le_128": bool(k) and all(r["registers"] is not None and 0 < r["registers"] <= 128 for r in k),
        "smem_cap": bool(k) and all(r["static_smem_bytes"] + mb.smem_bytes(r["n"], NS) <= mb.SMEM_CAP for r in k),
        "bulk_copy_every_instance": all(any(o.startswith(p) for o in raw.get(key, {}) for p in ("UBLKCP", "UTMALDG", "LDGSTS")) for key in ARMS),
        "hmma_only_where_expected": all((table.get(key, {}).get("HMMA", 0) > 0) == (key[0] in ("base", "twin", "dq_mov", "dq_eq", "lds_w1", "lds_x1")) for key in ARMS),
        "no_wgmma": "HGMMA" not in sass,
    }
    return {"ptxas": {f"{a}_n{n}": v for (a, n), v in blocks.items()},
            "sass_class_counts_static": {f"{a}_n{n}": v for (a, n), v in table.items()},
            "sass_opcodes_static": {f"{a}_n{n}": v for (a, n), v in raw.items()},
            "class_claims": claims, "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL",
            "c75xx_lines": [l for l in log.splitlines() if re.search(r"C75\d\d", l)]}, table


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
        mod = load_inline(name="m16xe_" + tag, cpp_sources=mb.CPP, cuda_sources=src, functions=None,
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
    result.update({"extension": mod.__file__, "compile_log": str(logfile), "generated_cuda": str(out.with_suffix(".cu"))})
    return mod, result, table


# -------------------------------------------------------------------------------------------------------- GPU run
def smi_sample(seconds, replay_fn, torch):
    """Copy of f5_data_run.smi_sample: back-to-back replays for `seconds`, nvidia-smi samples power/clock every 50 ms."""
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
    """base/twin: F3's FP32 oracle + changed-input poisoned graph replays; null: output untouched; others: protocol only."""
    weights = mb.make_weights(1, 31)
    valid = np.array([b for b in range(256) if b not in (127, 255)], np.uint8)
    weights["w13_sf"][0, :256, :] = np.resize(valid, (256, 256))
    bank = mb.prepare_bank(weights, 4, 4)
    w = torch.from_numpy(bank).cuda()
    err = torch.zeros(4, device="cuda", dtype=torch.int32)
    sm = torch.empty_like(err)
    marks = torch.empty(2, device="cuda", dtype=torch.int64)
    results = []
    for n in NS_ARMS:
        gen = torch.Generator().manual_seed(411 + n)
        x = torch.randn(n, 1024, generator=gen).to(torch.bfloat16)
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
        scale = torch.stack([r.abs().amax().clamp_min(1) for r in refs])[:, None, None]
        for key in ARMS:
            arm, an = key
            if an != n:
                continue
            row = {"arm": arm, "n": n, "status": "FAIL"}
            results.append(row)
            mb.reset_correctness_input(px, original)
            out = torch.full((16, 64, n), float("nan"), device="cuda", dtype=torch.float32)
            mb.tensor_bounds(*bounds_key(key)[:2], 4, 4, w, px, out, sm, err, marks, ns=NS)
            def call(key=key, out=out):
                launch_arm(mod, key, 4, 4, w, px, out, sm, err, marks, torch.cuda.current_stream().cuda_stream)
            call(); torch.cuda.synchronize()
            if err.any().item():
                raise RuntimeError((key, "device protocol timeout"))
            if arm in ATTRIB_ARMS:
                row.update({"status": "RUN-ONLY", "check": "protocol only; outputs wrong by design"}); continue
            def agrees(value, sign):
                if arm == "null":
                    return bool(torch.isnan(value).all())
                return bool(torch.isfinite(value).all() and torch.all((value - sign * reference).abs() <= 2e-5 * scale))
            if not agrees(out.cpu(), 1):
                row["why"] = "initial FP32 reference / untouched null mismatch"; continue
            if arm != "null":
                row["max_abs_error"] = float((out.cpu() - reference).abs().max().item())
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
                    raise RuntimeError((key, "graph protocol timeout"))
                ok_r = agrees(out.cpu(), sign)
                if not ok_r:
                    break
            row.update({"status": "PASS" if ok_r else "FAIL", "changed_input_replays": 3,
                        "check": "null untouched" if arm == "null" else "torch FP32 reference"})
    return results


def benchmark(mod, args, report):
    import torch
    prop = torch.cuda.get_device_properties(0)
    assert "H200" in prop.name and prop.multi_processor_count == 132 and prop.major == 9, str(prop)
    report["device"] = {"name": prop.name, "sms": prop.multi_processor_count}
    attrs = {}
    for key in ARMS:   # cudaFuncSetAttribute (dynamic smem) for EVERY instance before any launch
        arm, n = key
        reg, local, static, dynamic, occ = mod.info(ARM_MODE[arm], n, NS)
        assert 0 < reg <= 128 and local <= 16 and occ == 1, (key, reg, local, occ)
        assert dynamic == mb.smem_bytes(n, NS) and static + dynamic <= mb.SMEM_CAP, (key, static, dynamic)
        attrs[f"{arm}_n{n}"] = {"regs": reg, "local_bytes": local, "static_smem": static, "dynamic_smem": dynamic, "ctas_per_sm": occ}
    report["kernel_attributes"] = attrs
    report["correctness"] = gpu_correctness(mod, torch)
    bad = [c for c in report["correctness"] if c["arm"] in ("null", "base", "twin") and c["status"] != "PASS"]
    assert not bad, ("base/twin/null correctness", bad)
    print("[f5b] correctness " + " ".join(f'{c["arm"]}_n{c["n"]}:{c["status"]}' for c in report["correctness"]), flush=True)
    timer = mb.Timer(torch)
    stream = timer.stream.cuda_stream
    empty = timer.capture(lambda: mod.empty(stream))
    banks = []
    for b in range(3):
        weights = mb.make_weights(math.ceil(mb.GRID * args.units / 512), 103 + b)
        banks.append(torch.from_numpy(mb.prepare_bank(weights, args.units, mb.GRID)).cuda())
    assert len({w.data_ptr() for w in banks}) == 3
    inputs = {}
    for n in NS_ARMS:
        gen = torch.Generator().manual_seed(103 + n)
        x = torch.randn(n, 1024, generator=gen).to(torch.bfloat16)
        inputs[n] = torch.from_numpy(mb.pack_x(x.view(torch.int16).numpy().view(np.uint16)).view(np.int16)).cuda()
    output = torch.empty((mb.GRID, args.units, 64, 32), device="cuda", dtype=torch.float32)
    sm = torch.empty(mb.GRID, device="cuda", dtype=torch.int32)
    err = torch.zeros_like(sm)
    marks = torch.empty(2, device="cuda", dtype=torch.int64)
    timer.stream.wait_stream(torch.cuda.current_stream())
    graphs = {}
    for key in ARMS:
        arm, n = key
        graphs[key] = []
        for bank, w in enumerate(banks):
            mb.tensor_bounds(*bounds_key(key)[:2], args.units, mb.GRID, w, inputs[n], output, sm, err, marks, ns=NS)
            graphs[key].append(timer.capture(lambda key=key, n=n, w=w: launch_arm(mod, key, args.units, mb.GRID, w, inputs[n], output, sm, err, marks, stream)))
            assert not err.any().item(), (key, bank, "protocol error")
            assert len(set(sm.cpu().tolist())) == mb.GRID, (key, bank, "grid not one CTA per SM")
    print("[f5b] graphs captured", flush=True)
    report["clocks_before"] = mb.clocks()
    before = statistics.median(timer.replay(empty) for _ in range(15))
    samples = {k: [] for k in ARMS}
    mhz = {k: [] for k in ARMS}
    banks_used = {k: [] for k in ARMS}
    for rnd, key, bank in replay_schedule(args.rounds):
        samples[key].append(timer.replay(graphs[key][bank]))
        assert not err.any().item(), (key, "device protocol error")
        cyc, ns_delta = marks.cpu().tolist()
        assert cyc > 0 and ns_delta > 0, (key, "invalid loop clock")
        mhz[key].append(1000.0 * cyc / ns_delta)
        banks_used[key].append(bank)
    after = statistics.median(timer.replay(empty) for _ in range(15))
    report["clocks_after"] = mb.clocks()
    overhead = (before + after) / 2
    report["empty_kernel_us"] = {"before": before, "after": after, "subtracted": overhead}
    rows = []
    for key in ARMS:
        arm, n = key
        vals = [(s - overhead) / args.units for s in samples[key]]
        assert min(vals) > 0, "invalid empty-kernel subtraction"
        null = [(s - overhead) / args.units for s in samples[("null", n)]]
        rows.append({"arm": arm, "n": n, "mode": ARM_MODE[arm], "us_per_unit": statistics.median(vals),
                     "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90)),
                     "ratio_null": statistics.median(vals) / statistics.median(null),
                     "mhz_median": statistics.median(mhz[key]), "mhz_p10": float(np.percentile(mhz[key], 10)),
                     "mhz_p90": float(np.percentile(mhz[key], 90)), "mhz_samples": mhz[key],
                     "samples_total_us": samples[key], "banks_per_round": banks_used[key],
                     "measurement": "MEASURED CUDA events, judge flush + 300 us gate; (kernel - empty)/units; "
                                    "MHz = 1000*clock64/globaltimer over CTA0 thread0's unit loop"})
    by = {(r["arm"], r["n"]): r for r in rows}
    for r in rows:
        b = by[("base", r["n"])]
        print(f"[f5b] {r['arm']:13s} n{r['n']:<2d} {r['us_per_unit']:.4f} us/unit ({r['p10']:.4f}..{r['p90']:.4f})  "
              f"{r['mhz_median']:.0f} MHz ({r['mhz_p10']:.0f}..{r['mhz_p90']:.0f})  dMHz {r['mhz_median'] - b['mhz_median']:+.0f}", flush=True)
    report["rows"] = rows
    for key in ARMS:   # separate continuous runs, outside the timed protocol
        g = graphs[key][0]
        with torch.cuda.stream(timer.stream):
            pw = smi_sample(args.power_seconds, g.replay, torch)
        torch.cuda.synchronize()
        assert not err.any().item(), (key, "protocol error in power run")
        cyc, ns_delta = marks.cpu().tolist()
        pw["hot_in_kernel_mhz_last_replay"] = 1000.0 * cyc / ns_delta if ns_delta > 0 else None
        by[key]["power"] = pw
        print(f"[f5b] power {key[0]:13s} n{key[1]:<2d} {pw}", flush=True)
    report["reading"] = reading(rows)


def summary_lines(report):
    rows = report.get("rows", [])
    L = ["# F5b: instruction-class power attribution on the shipped ms consumer (NS=3, m16xe_micro.py)", "",
         f"status: {report.get('status')}  elapsed {report.get('elapsed_s', 0):.0f} s",
         "Reading (pre-registered, diagnostic): a class 'moves the clock' if its arm raises in-kernel MHz by >= 40 vs base with "
         "|twin - base| <= 15 MHz; rank by dMHz and by d power; flag any arm >= 1850 MHz.", "",
         "| arm | us/unit (p10..p90) | in-kernel MHz (p10..p90) | dMHz vs base | smi power W med/max | smi MHz | hot MHz |",
         "|---|---|---|---:|---|---:|---:|"]
    by = {(r["arm"], r["n"]): r for r in rows}
    for r in rows:
        b = by.get(("base", r["n"]))
        pw = r.get("power") or {}
        L.append("| %s_n%d | %.4f (%.4f..%.4f) | %.0f (%.0f..%.0f) | %s | %s | %s | %s |" % (
            r["arm"], r["n"], r["us_per_unit"], r.get("p10", r["us_per_unit"]), r.get("p90", r["us_per_unit"]),
            r["mhz_median"], r.get("mhz_p10", r["mhz_median"]), r.get("mhz_p90", r["mhz_median"]),
            ("%+.0f" % (r["mhz_median"] - b["mhz_median"])) if b else "n/a",
            ("%.0f/%.0f" % (pw["power_w_median"], pw.get("power_w_max", pw["power_w_median"]))) if pw.get("power_w_median") is not None else "n/a",
            ("%.0f" % pw["smi_sm_mhz_median"]) if pw.get("smi_sm_mhz_median") is not None else "n/a",
            ("%.0f" % pw["hot_in_kernel_mhz_last_replay"]) if pw.get("hot_in_kernel_mhz_last_replay") else "n/a"))
    for n, res in (report.get("reading") or reading(rows) if rows else {}).items():
        if "aa_mhz" not in res:
            L += ["", f"N={n}: {res.get('status')}"]; continue
        L += ["", f"N={n}: |twin-base| = {res['aa_mhz']:.0f} MHz ({'valid' if res['aa_valid'] else 'VOID: A/A > 15 MHz'}); "
                  f"moves the clock: {res['moves_clock'] or 'none'}; >= 1850 MHz: {res['flag_ge_1850'] or 'none'}",
              f"  rank by dMHz: {res['rank_by_d_mhz']}", f"  rank by d power (most saved first): {res['rank_by_d_power']}"]
        for i in res["items"]:
            L.append(f"  {i['arm']:13s} {i['class']}: dMHz {i['d_mhz']:+.0f}, d us/unit {i['d_us_per_unit']:+.4f}, "
                     f"d power {('%+.0f W' % i['d_power_w']) if i['d_power_w'] is not None else 'n/a'}")
    comp = report.get("compiler") or {}
    if comp.get("sass_class_counts_static"):
        table = {tuple([k.rsplit("_n", 1)[0], int(k.rsplit("_n", 1)[1])]): v for k, v in comp["sass_class_counts_static"].items()}
        L += ["", "SASS static instruction counts per class (one unit is straight-line; IMAD excludes IMAD.MOV, MOV includes it):", ""]
        L += class_table_lines(table)
        cl = comp.get("class_claims", {})
        L += ["", f"'same count' claims: {cl.get('status')}"]
        for r in cl.get("rows", []):
            if r["status"] != "PASS":
                L.append(f"  FAIL {r['arm']}_n{r['n']}: {r.get('mismatch', r.get('why'))}")
    return L


def write_summary(report, path):
    path.write_bytes(("\n".join(summary_lines(report)) + "\n").encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="CPU gates only (numpy); writes nothing")
    ap.add_argument("--compile-only", action="store_true", help="CPU + compile + SASS gates; never launches CUDA work")
    ap.add_argument("--out", type=Path, default=Path("box_out/f5b/results.json"))
    ap.add_argument("--units", type=int, default=UNITS)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--power-seconds", type=float, default=3.0)
    args = ap.parse_args()
    assert args.units == UNITS and args.rounds == 12 and args.power_seconds >= 3.0
    started = time.monotonic()
    out = args.out.resolve()
    report = {"task": "F5b", "status": "RUNNING", "arms": [f"{a}_n{n}" for a, n in ARMS], "modes": ARM_MODE,
              "pre_registered": {"moves_clock_d_mhz_ge": MOVE_MHZ, "aa_abs_mhz_le": AA_MHZ, "flag_mhz_ge": FLAG_MHZ}}
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
        print("[f5b] SASS static class counts per arm:\n" + "\n".join(class_table_lines(table)), flush=True)
        print("[f5b] compile gates " + json.dumps(report["compiler"]["gates"]), flush=True)
        claims = report["compiler"]["class_claims"]
        claims_ok = claims["status"] == "PASS"
        print(f"[f5b] same-count claims {claims['status']}" + "".join(
            f"\n  FAIL {r['arm']}_n{r['n']}: {r.get('mismatch', r.get('why'))}" for r in claims["rows"] if r["status"] != "PASS"), flush=True)
        assert report["compiler"]["status"] == "PASS", report["compiler"]["gates"]
        if args.compile_only:
            report["status"] = "COMPILE_GATES_PASS" if claims_ok else "COMPILE_PASS_CLAIMS_FAIL"
            print(f"[f5b] {report['status']}; JSON={out}", flush=True)
        else:
            benchmark(mod, args, report)
            report["status"] = "COMPLETE" if claims_ok else "COMPLETE (same-count claims FAILED: read the table)"
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

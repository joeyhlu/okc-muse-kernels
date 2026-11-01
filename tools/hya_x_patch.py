"""hya_x_patch.py (attention lane, 2026-11-01): hya_t0 + HYA_MARK timestamps (tools/hya_marks_patch.py) + runtime-switchable merge-workspace / gate-row
tweaks -> kernels/lab/hya_x*.py (ONE CUDA source, so one JIT build; the arms differ only in Python-side `_TUNED` plan bits, plan["tma"] bits 5..8 ->
Params::pfw).

Hypothesis (R-att-3/4): the judge flushes L2 before every replay, so everything the finalize touches is cold. hya_t0 warms the GATE rows (split 0 only)
but not the merge workspace, and its finalize loads the gate rows after the ticket / barrier, in the same round trip as the accumulator reads.
  pfw & 1  prefetch.global.L2 of this CTA's ws_acc lines (L * 32 x 128 B) and ws_l lines          (all non-flat CTAs, before pdl_wait)
  pfw & 2  prefetch.global.L2 of the ticket counter ws_cnt[bp] and the cooperative counters (+8192, +12288)
  pfw & 4  the gate-row prefetch (hya_t0: split 0 only) by every CTA, so the last CTA's gate loads hit L2 whichever CTA is last
  pfw & 8  the finalize's gate-row load is issued BEFORE the ticket (last-CTA mode, L = 1) / the barrier (cooperative mode) and held in registers
  skew     (plan["tma"] bits 9..16 -> Params::skew, flat kernels only): the first half of the grid (CTA ids < G / 2: the first-dispatched CTA of each SM)
           takes (1024 + skew) / 1024 of an equal key share, the second half (1024 - skew) / 1024. R-att-4 marks: at the q_len 8 flat shapes the second-half
           CTAs end 13-25 us after the first-half ones (v16f 76.9 vs 90.1 us, v32f 146.3 vs 171.6 us median loop end), at q_len 1 only 1-2 us.
Arms: xa control (pfw 0, marks inert), xt its twin (A/A), xb 1, xc 3, xd 7, xe 4, xh 8, xm 15; xs1..xs4 = the skew arms. Numerics unchanged (prefetches, load
hoisting and the split of the key range between CTAs only; the merge is order-independent up to fp32 atomics).
    python -I tools/hya_x_patch.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hya_marks_patch as M  # noqa: E402

ROOT = M.ROOT
SRC = M.SRC

PF_BLOCK = r"""  cp_commit();
  if (!WS && (p.pfw & 3)) {
    // L2 warm-up of the merge workspace (cold after the judge's L2 flush: the first red.add / the ticket would start from DRAM)
    if (p.pfw & 1) {
      const char* wa = reinterpret_cast<const char*>(p.ws_acc + (ix_t)((bp * p.nc + copy) * L) * (16 * 64));
      if (tid < L * 32) asm volatile("prefetch.global.L2 [%0];" ::"l"(wa + (ix_t)tid * 128));
      if (tid >= 64 && tid < 64 + L) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.ws_l + (ix_t)((bp * p.nc + copy) * L + (tid - 64)) * 16));
    }
    if ((p.pfw & 2) && tid >= 96 && tid < 99)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p.ws_cnt + (tid == 96 ? 0 : (tid == 97 ? 8192 : 12288)) + bp));
  }
  if (!WS && (MODE & 8)) {"""

GPRE_OLD = "  HYA_MARK(3);\n  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----\n"
GPRE_NEW = r"""  HYA_MARK(3);
  uint4 gpre = make_uint4(0u, 0u, 0u, 0u);   // pfw & 8: this CTA's last-CTA-finalize gate row, loaded before the merge
  if (L == 1 && !WS && (p.pfw & 8) && tid < NT)
    gpre = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (tid >> 3)) * 64 + (tid & 7) * 8);
  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
"""

FIN_SIG_OLD = "                                     const int dry) {\n  const int tid = threadIdx.x;\n"
FIN_SIG_NEW = ("                                     const int dry, const uint4 gpre = make_uint4(0u, 0u, 0u, 0u), const bool use_pre = false) {\n"
               "  const int tid = threadIdx.x;\n")
FIN_LOAD_OLD = ("      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);\n"
                "    }\n#pragma unroll\n    for (int u = 0; u < GP; ++u) {\n      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);\n"
                "      const int j = rr >> 4, hh = rr & 15;\n      float* acc")
FIN_LOAD_NEW = ("      if (use_pre) graw[u] = gpre;\n"
                "      else graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);\n"
                "    }\n#pragma unroll\n    for (int u = 0; u < GP; ++u) {\n      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);\n"
                "      const int j = rr >> 4, hh = rr & 15;\n      float* acc")

COOP_OLD = '    asm volatile("" : "+r"(nact), "+r"(R));\n'
COOP_NEW = r"""    asm volatile("" : "+r"(nact), "+r"(R));
    const bool use_pre = (p.pfw & 8) && R <= NT / 8;   // one pass of the slice loop: the gate row can be loaded before the barrier
    uint4 gate_pre = make_uint4(0u, 0u, 0u, 0u);
    if (use_pre) {
      const int rr = split_idx * R + (tid >> 3);
      if (rr < min(L * 16, split_idx * R + R)) {
        const int j = rr >> 4, hh = rr & 15;
        gate_pre = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + (tid & 7) * 8);
      }
    }
"""
COOP_LOAD_OLD = ("      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)\n"
                 "                             : make_uint4(0, 0, 0, 0);\n")
COOP_LOAD_NEW = ("      const uint4 graw = use_pre ? gate_pre : (act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)\n"
                 "                             : make_uint4(0, 0, 0, 0));\n")


SKEW_FN = r"""// first half of the grid takes (1024 + dq) / 1024 of an equal share of the T keys, the second half (1024 - dq) / 1024; boundary of CTA c
__device__ __forceinline__ int skew_bnd(int c, int64_t T, int G, int dq) {
  if (c >= G) return (int)T;
  const int half = G >> 1;
  const int64_t W = c <= half ? (int64_t)c * (1024 + dq) : (int64_t)half * (1024 + dq) + (int64_t)(c - half) * (1024 - dq);
  return (int)(T * W / ((int64_t)1024 * G));
}
template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {"""

EDITS = [
    ("  cp_commit();\n  if (!WS && (MODE & 8)) {", PF_BLOCK, 1),
    ("  if (split == 0 && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last",
     "  if ((split == 0 || (p.pfw & 4)) && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last", 1),
    ("  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n};",
     "  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)\n  int pfw;              // bits 0 / 1 / 2 / 3: merge-workspace / counter / gate-row L2 warm-up, finalize gate-load hoist (R-att-3/4)\n};", 1),
    ("  p.coop = (int)((tma >> 1) & 1);\n", "  p.coop = (int)((tma >> 1) & 1);\n  p.pfw = (int)((tma >> 5) & 15);\n", 1),
    (GPRE_OLD, GPRE_NEW, 1),
    ("    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0);\n", "    fin_nc1<L, NT>(p, b, pair, bp, vscale, 0, gpre, L == 1 && (p.pfw & 8) != 0);\n", 1),
    (FIN_SIG_OLD, FIN_SIG_NEW, 1),
    (FIN_LOAD_OLD, FIN_LOAD_NEW, 1),
    (COOP_OLD, COOP_NEW, 1),
    (COOP_LOAD_OLD, COOP_LOAD_NEW, 1),
    ("  HYA_MARK(0);\n", "  HYA_MARK(0);\n  if (p.dbg && threadIdx.x == 0) { unsigned sm_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(sm_)); p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 14] = (long long)sm_; }\n", 1),
    ("  const int nseg = S.nseg;\n", "  const int nseg = S.nseg;\n  if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 15] = (long long)nseg;\n", 1),
    ("template <int L, bool FLAT, int MODE = 0>\n__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {", SKEW_FN, 1),
    ("  int pfw;              // bits 0 / 1 / 2 / 3: merge-workspace / counter / gate-row L2 warm-up, finalize gate-load hoist (R-att-3/4)\n};",
     "  int pfw;              // bits 0 / 1 / 2 / 3: merge-workspace / counter / gate-row L2 warm-up, finalize gate-load hoist (R-att-3/4)\n  int skew;             // flat kernels: first-half CTAs take (1024 + skew) / 1024 of an equal key share (R-att-5)\n};", 1),
    ("  p.pfw = (int)((tma >> 5) & 15);\n", "  p.pfw = (int)((tma >> 5) & 15);\n  p.skew = ((tma >> 17) & 1) ? (int)((tma >> 9) & 255) - 128 : 0;\n", 1),
    ("        int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);",
     "        int csg = skew_bnd((int)blockIdx.x, Tg, (int)gridDim.x, p.skew), ceg = skew_bnd((int)blockIdx.x + 1, Tg, (int)gridDim.x, p.skew);", 1),
    ("      int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);",
     "      int csf = skew_bnd(cf, Tf, Gf, p.skew), cef = skew_bnd(cf + 1, Tf, Gf, p.skew);", 1),
    ("    cs = (int)((int64_t)cpart * T / G);\n    ce = (int)((int64_t)(cpart + 1) * T / G);\n",
     "    cs = skew_bnd(cpart, T, G, p.skew);\n    ce = skew_bnd(cpart + 1, T, G, p.skew);\n", 1),
    ("  int skew;             // flat kernels: first-half CTAs take (1024 + skew) / 1024 of an equal key share (R-att-5)\n};",
     "  int skew;             // flat kernels: first-half CTAs take (1024 + skew) / 1024 of an equal key share (R-att-5)\n  int segk;             // flat kernels: virtual keys charged per request boundary (0: the compiled default 600 / 400)\n};", 1),
    ("  p.skew = ((tma >> 17) & 1) ? (int)((tma >> 9) & 255) - 128 : 0;\n", "  p.skew = ((tma >> 17) & 1) ? (int)((tma >> 9) & 255) - 128 : 0;\n  p.segk = (int)((tma >> 18) & 1023);\n", 1),
    ("        constexpr int SEGKF = (L == 8) ? 600 : 400;\n", "        const int SEGKF = p.segk ? p.segk : ((L == 8) ? 600 : 400);\n", 1),
    ("      constexpr int SEGKF = (L == 8) ? 600 : 400;\n", "      const int SEGKF = p.segk ? p.segk : ((L == 8) ? 600 : 400);\n", 1),
    ("  constexpr int SEGK = (L == 8) ? 600 : 400;\n", "  const int SEGK = p.segk ? p.segk : ((L == 8) ? 600 : 400);\n", 1),
]

# plan overrides per arm: pfw bits go in "tma" (bits 5..8). The base plan stays as in hya_t0 except for the two shapes hya_t0 already overrides.
BASE = {(32, 8, True): dict(n_splits=4, nc=1, coop=1), (1, 1, False): dict(nc=8, coop=1, n_splits=252)}
SHAPES = [(1, 1, True), (16, 1, True), (32, 1, True), (1, 8, True), (16, 8, True), (32, 8, True), (1, 1, False), (1, 8, False)]
ARMS = {"xa": 0, "xt": 0, "xb": 1, "xc": 3, "xd": 7, "xe": 4, "xh": 8, "xm": 15, "xs1": 0, "xs2": 0, "xs3": 0, "xs4": 0, "xn1": 0, "xn2": 0, "xn3": 0, "xn4": 0, "xk1": 0, "xk2": 0, "xk3": 0, "xk4": 0}
FLAT_SHAPES = [(16, 1, False), (32, 1, False), (16, 8, False), (32, 8, False)]
# virtual keys per request boundary: (q_len 1, q_len 8); the compiled defaults are 400 / 600
SEGKS = {"xk1": (200, 300), "xk2": (300, 450), "xk3": (500, 750), "xk4": (600, 900)}
# skew in 1/1024 per arm: (q_len 1, q_len 8)
SKEW = {"xs1": (4, 40), "xs2": (8, 60), "xs3": (12, 80), "xs4": (16, 100), "xn1": (-4, -40), "xn2": (-8, -80), "xn3": (-12, -120), "xn4": (-16, -127)}
OLD_TUNED = """_TUNED = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}
"""


def fmt(d):
    return "dict(" + ", ".join(f"{k}={v}" for k, v in d.items()) + ")"


def main():
    text = SRC.read_text(encoding="ascii")
    for old, new, n in M.EDITS + EDITS:
        c = text.count(old)
        if c != n:
            sys.exit(f"anchor count {c} != {n} for: {old[:90]!r}")
        text = text.replace(old, new)
    assert text.count("okc_attn_v298mk") == 1
    text = text.replace("okc_attn_v298mk", "okc_attn_v298x")   # its own extension build (never shared with hya_marks / hya_t0)
    assert text.count(OLD_TUNED) == 1
    for name, bits in ARMS.items():
        ent = ""
        for key in SHAPES:
            d = dict(BASE.get(key, {}))
            if bits:
                d["tma"] = bits << 5
            if d:
                ent += f"    ({key[0]}, {key[1]}, {key[2]}, 1): {fmt(d)},\n"
        if name in SKEW:   # flat full-layer shapes: weighted key split (first half of the grid takes more)
            for key in FLAT_SHAPES:
                dq = SKEW[name][0] if key[1] == 1 else SKEW[name][1]
                ent += f"    ({key[0]}, {key[1]}, {key[2]}, 1): dict(tma={((dq + 128) << 9) | (1 << 17)}),\n"
        if name in SEGKS:
            for key in FLAT_SHAPES:
                sk = SEGKS[name][0] if key[1] == 1 else SEGKS[name][1]
                ent += f"    ({key[0]}, {key[1]}, {key[2]}, 1): dict(tma={sk << 18}),\n"
        t = text.replace(OLD_TUNED, "_TUNED = {\n" + ent + "}\n")
        ls = t.split("\n")
        ls[1] = f"# nonce hya-x-{name} (attention lane Nov 1: hya_t0 + HYA_MARK + pfw bits {bits}; tools/hya_x_patch.py; lab only)"
        out = ROOT / "kernels" / "lab" / f"hya_{name}.py"
        out.write_text("\n".join(ls), encoding="ascii", newline="\n")
        print(out.relative_to(ROOT), len(out.read_bytes()), "B", "pfw", bits)


if __name__ == "__main__":
    main()

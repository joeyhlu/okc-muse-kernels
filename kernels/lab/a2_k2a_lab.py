"""A2 lab kernel: K2A v3.3 (kernels/muse_moe_w4a16/k2a_v33.py, nonce k2a-v33-1) + compile-time instrumentation.

LAB FILE. Box runs only, through tools/box/a2_lab.py. Never submit it, and never import it from a scored file.
Plan and hypotheses: notes/study/a2_lab_plan.md.

The CUDA source is v3.3's KERNEL_SRC_V3, read from k2a_v33.py (via ast, so this module imports without torch). It
is patched at anchors (PATCHES: each `old` must occur exactly once, or lab_source() raises), and then emitted once per
lab mask, each copy in its own namespace `a2L<mask>` with `constexpr int A2_LAB = <mask>`. Every patch is behind
`if constexpr (A2_LAB & ...)`, so the mask-0 copy compiles to v3.3's kernel code. Params gains 4 trailing fields
(the trace pointers and 4 ints), the only difference. The driver times the mask-0 copy against the shipped module in
the same process, to check that.

Mask bits (combine freely; tools/box/a2_lab.py builds LABS):
  TRACE  1  per-CTA header (smid, %globaltimer + clock64 pairs at entry / route done / role ends, pdl-wait clocks,
            stage counts, epilogue finalize busy/wait) + per-stage clock64 rows (producer empty-wait start, copy
            issue, consumer warp 0 full-wait return and slot release, unit flush end, stage tag). One lane per role,
            plain global stores, bounded by a2_smax.
  NULL   2  consumers arrive on `empty` right after `full` returns: no smem reads, no decode/mma, no flush, no epilogue
            hand-off, no y/ss zeroing. Producer, ring, X copies, PDL chain and stream-K schedule unchanged.
  CAMP   4  each CTA starts its stream-K range at offset (cta * a2_mul mod n) units of a2_unit stages (16 = whole items,
            1 = stages; a2_mul 0 = hashed) and wraps. Same items, same split-K partners, only the order changes.
  DEDUP  8  passes j >= 1 of an (expert, row block) skip the weight copy (X still copied). Only meaningful with NULL:
            the bound for "every re-pass is free".
  RAMP  16  the kernels end after the prologue. a2_mode 0: launch + barrier init; 1: + routing; 2: + the CTA's first
            NST stages (weights only) landed in smem.
Bounded waits only (v3.3's mbar_wait budget -> error flag); the lab adds no new wait loop.
"""
import ast
import hashlib
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
K2A_FILE = ROOT / "kernels/muse_moe_w4a16/k2a_v33.py"

TRACE, NULL, CAMP, DEDUP, RAMP = 1, 2, 4, 8, 16
LABS = (0, TRACE, NULL, NULL | TRACE, CAMP, CAMP | TRACE, CAMP | NULL, CAMP | NULL | TRACE, NULL | DEDUP,
        NULL | DEDUP | TRACE, RAMP, RAMP | TRACE)
# (NC, NST, CTAs/SM, HOTCAP): vid 0 = v3.3's plan for bs 1-32; 1 = its bs128/256 cold variant (16 tokens/pass);
# 2 = 32 tokens/pass at 2 stages (fewer re-passes of hot experts)
VARIANTS = [(1, 3, 2, 0), (2, 3, 2, 0), (4, 2, 2, 0)]
# variants compiled per mask (tools/box/a2_lab.py ROWS use these): NC2 / NC4 only real and null, everything else NC1
BUILD = {lab: ((0, 1, 2) if lab in (0, NULL) else (0,)) for lab in LABS}
SW = 18432
SMEM_MIN = 80 * 1024   # dynamic smem floor of the instrumented copies: 3 x (80 KiB + static) > 228 KiB/SM -> 2 CTAs/SM
HREC, ROWS = 32, 6
H = {"SMID": 0, "GT0": 1, "CK0": 2, "CKR": 3, "GTR": 4, "PW0": 5, "PW1": 6, "PCK": 7, "PGT": 8, "CCK": 9, "CGT": 10,
     "ECK": 11, "EGT": 12, "NP": 13, "NC": 14, "NFIN": 15, "EBUSY": 16, "BEG": 17, "END": 18, "MID": 19, "PC0": 20,
     "EC0": 21, "EWAIT": 22, "NUNIT": 23, "GRID": 24, "NSPLIT": 25}
R = {"ISSUE": 0, "LAND": 1, "REL": 2, "TAG": 3, "FLUSH": 4, "PWAIT": 5}
# header slots: CK0/GT0 entry (thread 0); CKR/GTR gemm entry = routing done (thread 0); PC0/EC0 producer / epilogue
# clock bases; PCK/PGT, CCK/CGT, ECK/EGT producer / consumer warp 0 / epilogue end; PW0/PW1 producer clock before /
# after griddepcontrol.wait (K2); NP/NC stages issued / consumed; NUNIT consumer units; NFIN / NSPLIT / EBUSY / EWAIT
# epilogue finalizes, hand-offs, clocks busy and in bar.sync. Rows: ISSUE / PWAIT rel. PC0, LAND / REL / FLUSH rel. CKR.


def decode_tag(t):
    """(e, j, rb, s, ntok, seg) from a stage tag (a2_tag)."""
    return (t >> 16) & 31, (t >> 10) & 63, (t >> 4) & 63, t & 15, (t >> 21) & 63, (t >> 27) & 1


def _shipped():
    """(KERNEL_SRC_V3, _wg_specs()) of k2a_v33.py, without importing it (it needs torch)."""
    tree = ast.parse(K2A_FILE.read_text())
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "KERNEL_SRC_V3" for t in node.targets):
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "_wg_specs":
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
    return ns["KERNEL_SRC_V3"], ns["_wg_specs"]()


_TR = "if constexpr ((A2_LAB & A2_TRACE) != 0)"

A2_DEVICE = r"""
// ======== A2 lab instrumentation (kernels/lab/a2_k2a_lab.py); every use is behind if constexpr (A2_LAB & ...) ========
constexpr int A2_TRACE = 1, A2_NULL = 2, A2_CAMP = 4, A2_DEDUP = 8, A2_RAMP = 16;
constexpr int A2_HREC = @HREC@, A2_ROWS = @ROWS@;
@HCONST@
__device__ __forceinline__ long long a2_gt() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ long long a2_clk() {
  if constexpr ((A2_LAB & A2_TRACE) != 0) return clock64();
  else return 0;
}
__device__ __forceinline__ long long* a2_hdr(const Params& P, int k) {
  return P.a2h + ((size_t)k * gridDim.x + blockIdx.x) * A2_HREC;
}
__device__ __forceinline__ void a2_put(const Params& P, int k, int r, int i, int v) {
  if (i < P.a2_smax) P.a2s[(((size_t)k * gridDim.x + blockIdx.x) * A2_ROWS + r) * P.a2_smax + i] = v;
}
__device__ __forceinline__ void a2_stamp(const Params& P, int k, int r, int i, long long c0) {
  a2_put(P, k, r, i, (int)(clock64() - c0));
}
__device__ __forceinline__ void a2_mark(const Params& P, int k, int slot_clk, int slot_gt) {   // clock64 + globaltimer pair
  long long* h = a2_hdr(P, k);
  const long long c = clock64();
  h[slot_gt] = a2_gt();
  h[slot_clk] = c;
}
__device__ __forceinline__ int a2_tag(int e, int j, int rb, int s, int ntok, int seg) {
  return (s & 15) | ((rb & 63) << 4) | ((j & 63) << 10) | ((e & 31) << 16) | (min(ntok, 63) << 21) | ((seg & 1) << 27);
}
__device__ __forceinline__ void a2_entry(const Params& P, int k) {
  if (threadIdx.x == 0) {
    long long* h = a2_hdr(P, k);
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const long long c = clock64();
    h[H_GT0] = a2_gt();
    h[H_CK0] = c;
    h[H_SMID] = smid;
    h[H_GRID] = gridDim.x;
  }
}
"""

A2_MID = r"""
// A2 lab (A2_CAMP): rotate the start of the CTA's range (channel-camping probe); else v3.3's tail-first split.
__device__ __forceinline__ int a2_mid(const Params& P, const Route& S, int beg, int end, int cta) {
  if constexpr ((A2_LAB & A2_CAMP) != 0) {
    if (end <= beg) return beg;
    const unsigned r = (P.a2_mul > 0) ? (unsigned)cta * (unsigned)P.a2_mul : ((unsigned)cta * 2654435761u) >> 7;
    if (P.a2_unit <= 1) return beg + (int)(r % (unsigned)(end - beg));   // stage units
    const int b0 = (beg + 15) & ~15;   // item units: items start at multiples of 16 stages
    if (b0 >= end) return beg;
    return b0 + 16 * (int)(r % (unsigned)((end - b0 + 15) >> 4));
  } else {
    return tail_first(S, beg, end);
  }
}
"""

A2_RAMP_FN = r"""
// A2 lab (A2_RAMP): the prologue alone. a2_mode 1: routing done; 2: the CTA's first NST stages (weights only) land.
template <int NC, int NST, int MODE, int STG>
__device__ void a2_ramp(const Params& P, const Route& S, uint8_t* ring, uint64_t* bars) {
  constexpr int A2K = MODE - 1;
  constexpr int NRB = (MODE == 1) ? 64 : 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long long c0 = a2_clk();
  (void)c0;
  if constexpr ((A2_LAB & A2_TRACE) != 0) {
    if (tid == 0) { long long* h = a2_hdr(P, A2K); h[H_CKR] = c0; h[H_GTR] = a2_gt(); }
  }
  int n = 0, beg = 0, end = 0;
  if (P.a2_mode >= 2) {
    my_range(S.sb[32], beg, end);
    n = min(NST, end - beg);
  }
  const uint32_t full_a = sa(bars);
  if (warp == 8) {
    const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;
    int e = 0;
    for (int i = 0; i < n; ++i) {
      const int x = beg + i;
      while (S.sb[e + 1] <= x) ++e;
      const int l = x - S.sb[e], item = l >> 4, s = l & 15, rb = item / S.np[e];
      const uint32_t fb = full_a + 8 * i;
      if (lane == 0) {
        mbar_expect_tx(fb, SW);
        bulk_g2s(sa(ring + i * STG), wsrc + ((size_t)(e * NRB + rb) * 16 + s) * SW, SW, fb);
        if constexpr ((A2_LAB & A2_TRACE) != 0) {
          a2_stamp(P, A2K, R_ISSUE, i, c0);
          a2_put(P, A2K, R_PWAIT, i, 0);
          a2_put(P, A2K, R_TAG, i, a2_tag(e, item - rb * S.np[e], rb, s, 0, 0));
        }
      }
      mbar_arrive(fb);
    }
    if constexpr ((A2_LAB & A2_TRACE) != 0) {
      if (lane == 0) { long long* h = a2_hdr(P, A2K); h[H_PC0] = c0; h[H_NP] = n; a2_mark(P, A2K, H_PCK, H_PGT); }
    }
  } else if (warp < 8) {
    for (int i = 0; i < n; ++i) {
      mbar_wait(full_a + 8 * i, 0, P.err);
      if constexpr ((A2_LAB & A2_TRACE) != 0) {
        if (tid == 0) { a2_stamp(P, A2K, R_LAND, i, c0); a2_stamp(P, A2K, R_REL, i, c0); }
      }
    }
    if constexpr ((A2_LAB & A2_TRACE) != 0) {
      if (tid == 0) {
        long long* h = a2_hdr(P, A2K);
        h[H_NC] = n; h[H_BEG] = beg; h[H_END] = end; h[H_MID] = beg;
        a2_mark(P, A2K, H_CCK, H_CGT);
      }
    }
  }
}
__device__ __forceinline__ void a2_ramp_end(const Params& P, int k) {
  if constexpr ((A2_LAB & A2_TRACE) != 0) {
    if (threadIdx.x == 0) a2_mark(P, k, H_CCK, H_CGT);
  }
}

"""


def _entry_patch(k, route_line):
    mode = k + 1
    old = ("  __shared__ Route S;\n  pdl_launch();\n  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);\n"
           "  if constexpr (!HOTCAP) setup_bars<NC, NST>(bars);\n" + route_line)
    new = ("  __shared__ Route S;\n"
           f"  {_TR} a2_entry(P, {k});\n"
           "  pdl_launch();\n  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);\n"
           "  if constexpr (!HOTCAP) setup_bars<NC, NST>(bars);\n"
           "  if constexpr ((A2_LAB & A2_RAMP) != 0) {\n"
           f"    if (P.a2_mode == 0) {{ a2_ramp_end(P, {k}); return; }}   // A2 ramp: launch + barrier init only\n"
           "  }\n" + route_line +
           "  if constexpr ((A2_LAB & A2_RAMP) != 0) {   // A2 ramp: prologue only\n"
           f"    if constexpr (!HOTCAP) a2_ramp<NC, NST, {mode}, STG>(P, S, smem, bars);\n"
           "    return;\n  }\n")
    return old, new


PATCHES = [
    ("params",
     "  int M, lo, probe;\n};",
     "  int M, lo, probe;\n"
     "  long long* a2h; int* a2s;   // A2 lab trace: header [2][grid][A2_HREC], stage clocks [2][grid][A2_ROWS][a2_smax]\n"
     "  int a2_mode, a2_mul, a2_unit, a2_smax;   // A2 lab: ramp sub-mode, camping multiplier / unit, stage slots per CTA\n"
     "};"),
    ("helpers", "  int R;\n};\n", "  int R;\n};\n" + A2_DEVICE),
    ("a2_mid",
     "  const int m = S.sb[e] + ((end - 1 - S.sb[e]) & ~15);\n  return max(beg, m);\n}\n",
     "  const int m = S.sb[e] + ((end - 1 - S.sb[e]) & ~15);\n  return max(beg, m);\n}\n" + A2_MID),
    ("gemm_entry",
     "  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);\n\n  if (warp == 8) {",
     "  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);\n"
     "  constexpr int A2K = MODE - 1;\n"
     "  const long long a2c0 = a2_clk();   // A2 lab: this role's clock base (0 unless tracing)\n"
     "  (void)a2c0;\n"
     f"  {_TR} {{\n"
     "    if (tid == 0) { long long* h = a2_hdr(P, A2K); h[H_CKR] = a2c0; h[H_GTR] = a2_gt(); h[H_BEG] = beg; h[H_END] = end; }\n"
     "  }\n\n  if (warp == 8) {"),
    ("prod_mid",
     "    uint32_t dB[NST];\n    const int mid = tail_first(S, beg, end);",
     "    uint32_t dB[NST];\n    const int mid = a2_mid(P, S, beg, end, cta);"),
    ("prod_pdl_lazy",
     "          pdl_wait();\n          asm volatile(\"fence.proxy.async;\\n\" ::: \"memory\");\n          waited = true;",
     f"          {_TR} {{ if (lane == 0) a2_hdr(P, A2K)[H_PW0] = clock64(); }}\n"
     "          pdl_wait();\n"
     f"          {_TR} {{ if (lane == 0) a2_hdr(P, A2K)[H_PW1] = clock64(); }}\n"
     "          asm volatile(\"fence.proxy.async;\\n\" ::: \"memory\");\n          waited = true;"),
    ("prod_issue",
     "        mbar_wait(empty_a + 8 * slot, par, P.err);\n"
     "        const uint32_t dst = ring_a + slot * STG;\n"
     "        const uint32_t fb = full_a + 8 * slot;\n"
     "        if (lane == 0) {\n"
     "          mbar_expect_tx(fb, SW + xbytes);\n"
     "          const uint8_t* src = wb + (size_t)s * SW;\n"
     "          if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);\n"
     "          else bulk_g2s(dst, src, SW, fb);\n"
     "        }",
     f"        {_TR} {{ if (lane == 0) a2_stamp(P, A2K, R_PWAIT, it, a2c0); }}\n"
     "        mbar_wait(empty_a + 8 * slot, par, P.err);\n"
     "        const uint32_t dst = ring_a + slot * STG;\n"
     "        const uint32_t fb = full_a + 8 * slot;\n"
     "        const bool a2skip = ((A2_LAB & A2_DEDUP) != 0) && j > 0;   // A2 dedup: re-passes skip the weights\n"
     "        if (lane == 0) {\n"
     "          mbar_expect_tx(fb, (a2skip ? 0u : (uint32_t)SW) + xbytes);\n"
     "          const uint8_t* src = wb + (size_t)s * SW;\n"
     "          if (a2skip) {\n"
     "          } else if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);\n"
     "          else bulk_g2s(dst, src, SW, fb);\n"
     f"          {_TR} {{ a2_stamp(P, A2K, R_ISSUE, it, a2c0); a2_put(P, A2K, R_TAG, it, a2_tag(e, j, rb, s, ntok, seg)); }}\n"
     "        }"),
    ("prod_pdl_end",
     "    if (!waited) {\n      pdl_wait();",
     "    if (!waited) {\n"
     f"      {_TR} {{ if (lane == 0) a2_hdr(P, A2K)[H_PW0] = clock64(); }}\n"
     "      pdl_wait();\n"
     f"      {_TR} {{ if (lane == 0) a2_hdr(P, A2K)[H_PW1] = clock64(); }}"),
    ("prod_end",
     "    return;\n  }\n\n  if (warp == 9) {",
     f"    {_TR} {{\n"
     "      if (lane == 0) { long long* h = a2_hdr(P, A2K); h[H_PC0] = a2c0; h[H_NP] = it; a2_mark(P, A2K, H_PCK, H_PGT); }\n"
     "    }\n    return;\n  }\n\n  if (warp == 9) {"),
    ("epi_ret",
     "      if ((P.probe & 15) != 0) return;   // probes: consumers never hand over",
     "      if ((A2_LAB & A2_NULL) != 0 || (P.probe & 15) != 0) {   // probes / A2 null consumer: no hand-offs\n"
     f"        {_TR} {{ if (lane == 0) {{ a2_hdr(P, A2K)[H_EC0] = a2c0; a2_mark(P, A2K, H_ECK, H_EGT); }} }}\n"
     "        return;\n      }"),
    ("epi_mid",
     "      int npart = 0, e = 0;\n      const int mid = tail_first(S, beg, end);",
     "      int npart = 0, e = 0;\n      const int mid = a2_mid(P, S, beg, end, cta);\n"
     "      long long a2e_last = 0, a2e_ws = 0, a2e_busy = 0, a2e_wait = 0;   // A2 trace: epilogue clocks\n"
     "      int a2e_n = 0, a2e_fin = 0;\n"
     "      (void)a2e_last; (void)a2e_ws; (void)a2e_busy; (void)a2e_wait; (void)a2e_n; (void)a2e_fin;\n"
     f"      {_TR} {{ if (lane == 0) a2_hdr(P, A2K)[H_EC0] = a2c0; }}"),
    ("epi_sync",
     "          asm volatile(\"bar.sync %0, 288;\\n\" :: \"r\"(2 + npart) : \"memory\");\n          ++npart;",
     f"          {_TR} {{ a2e_ws = clock64(); if (a2e_n > 0) a2e_busy += a2e_ws - a2e_last; }}\n"
     "          asm volatile(\"bar.sync %0, 288;\\n\" :: \"r\"(2 + npart) : \"memory\");\n"
     f"          {_TR} {{ a2e_last = clock64(); a2e_wait += a2e_last - a2e_ws; ++a2e_n; }}\n"
     "          ++npart;"),
    ("epi_last",
     "          last = __shfl_sync(FULL, last, 0);\n",
     "          last = __shfl_sync(FULL, last, 0);\n"
     f"          {_TR} a2e_fin += last;\n"),
    ("epi_end",
     "      }\n    }\n    return;\n  }\n\n  // ---------------- consumers",
     "      }\n"
     f"      {_TR} {{\n"
     "        if (a2e_n > 0) a2e_busy += clock64() - a2e_last;\n"
     "        if (lane == 0) {\n"
     "          long long* h = a2_hdr(P, A2K);\n"
     "          h[H_NFIN] = a2e_fin; h[H_EBUSY] = a2e_busy; h[H_EWAIT] = a2e_wait; h[H_NSPLIT] = a2e_n;\n"
     "          a2_mark(P, A2K, H_ECK, H_EGT);\n"
     "        }\n"
     "      }\n"
     "    }\n    return;\n  }\n\n  // ---------------- consumers"),
    ("yzero",
     "  if (MODE == 1 && ((P.probe & 15) == 0 || (P.probe & 15) >= 6)) {",
     "  if (MODE == 1 && (A2_LAB & A2_NULL) == 0 && ((P.probe & 15) == 0 || (P.probe & 15) >= 6)) {"),
    ("cons_mid",
     "  uint32_t sink = 0u;\n  int npart = 0;\n  const int mid = tail_first(S, beg, end);",
     "  uint32_t sink = 0u;\n  int npart = 0;\n  const int mid = a2_mid(P, S, beg, end, cta);\n"
     "  int a2_nu = 0;   // A2 trace: consumer units\n  (void)a2_nu;\n"
     f"  {_TR} {{ if (tid == 0) a2_hdr(P, A2K)[H_MID] = mid; }}"),
    ("cons_land",
     "      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);\n      const uint8_t* st = ring + slot * STG;",
     "      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);\n"
     f"      {_TR} {{ if (tid == 0) a2_stamp(P, A2K, R_LAND, it, a2c0); }}\n"
     "      const uint8_t* st = ring + slot * STG;"),
    ("cons_null",
     "      if ((P.probe & 15) == 1 || (P.probe & 15) == 5) {",
     "      if constexpr ((A2_LAB & A2_NULL) != 0) {\n"
     "        // A2 null consumer: no smem reads, no math; the slot is released right away\n"
     "      } else if ((P.probe & 15) == 1 || (P.probe & 15) == 5) {"),
    ("cons_rel",
     "      __syncwarp();\n      if (lane == 0) mbar_arrive(empty_a + 8 * slot);\n    }\n    }\n    // ---- flush ----",
     "      __syncwarp();\n      if (lane == 0) mbar_arrive(empty_a + 8 * slot);\n"
     f"      {_TR} {{ if (tid == 0) a2_stamp(P, A2K, R_REL, it, a2c0); }}\n"
     "    }\n    }\n    // ---- flush ----"),
    ("fmode",
     "    const int fmode = P.probe & 15;",
     "    const int fmode = ((A2_LAB & A2_NULL) != 0) ? 1 : (P.probe & 15);"),
    ("cons_end",
     "    if (!(s0 == 0 && s1 == 16)) ++npart;\n    x += s1 - s0;\n  }\n  }\n"
     "  if (sink == 0x9E3779B9u) atomicAdd(P.sink, 1u);   // keeps probe work alive; never true in practice\n}",
     f"    {_TR} {{ if (tid == 0 && it > 0) a2_stamp(P, A2K, R_FLUSH, it - 1, a2c0); ++a2_nu; }}\n"
     "    if (!(s0 == 0 && s1 == 16)) ++npart;\n    x += s1 - s0;\n  }\n  }\n"
     "  if (sink == 0x9E3779B9u) atomicAdd(P.sink, 1u);   // keeps probe work alive; never true in practice\n"
     f"  {_TR} {{\n"
     "    if (tid == 0) { long long* h = a2_hdr(P, A2K); h[H_NC] = it; h[H_NUNIT] = a2_nu; a2_mark(P, A2K, H_CCK, H_CGT); }\n"
     "  }\n}"),
    ("ramp_fn",
     "template <int NC, int NST>\n__device__ __forceinline__ void setup_bars(uint64_t* bars) {",
     A2_RAMP_FN + "template <int NC, int NST>\n__device__ __forceinline__ void setup_bars(uint64_t* bars) {"),
    ("k1_entry",) + _entry_patch(0, "  route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);\n"),
    ("k2_entry",) + _entry_patch(1, "  route<NC>(P, S, nullptr, s_list, 32, false);\n"),
    ("smem_pad",
     "static constexpr int smem_bytes3() {\n  return HOTCAP ?",
     "static constexpr int smem_bytes3() {\n  constexpr int v = HOTCAP ?"),
    ("smem_pad_ret",
     "                : NST * (k3::SW + 8 * NC * k3::XROW) + 2 * NST * 8;\n}",
     "                : NST * (k3::SW + 8 * NC * k3::XROW) + 2 * NST * 8;\n"
     "  // A2 lab: lighter (null / ramp) builds would fit 3 CTAs/SM by registers; padding the dynamic smem keeps v3.3's\n"
     "  // residency (2 CTAs/SM, no K2 CTA beside two K1 CTAs during the PDL hand-off)\n"
     "  return (A2_LAB != 0 && v < A2_SMEM_MIN) ? A2_SMEM_MIN : v;\n}"),
    ("run3_sig",
     "          torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe) {\n"
     "  k3::Params p;",
     "          torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe,\n"
     "          torch::Tensor a2h, torch::Tensor a2s, int64_t a2_mode, int64_t a2_mul, int64_t a2_unit, int64_t a2_smax) {\n"
     "  k3::Params p;"),
    ("run3_fill",
     "  p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;",
     "  p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;\n"
     "  p.a2h = reinterpret_cast<long long*>(a2h.data_ptr()); p.a2s = reinterpret_cast<int*>(a2s.data_ptr());\n"
     "  p.a2_mode = (int)a2_mode; p.a2_mul = (int)a2_mul; p.a2_unit = (int)a2_unit; p.a2_smax = (int)a2_smax;"),
]

RUN_ARGS = ("int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, "
            "torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, "
            "torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err, "
            "torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw, "
            "torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe, "
            "torch::Tensor a2h, torch::Tensor a2s, int64_t a2_mode, int64_t a2_mul, int64_t a2_unit, int64_t a2_smax")
RUN_NAMES = ("vid, mask, x, ids, wts, ws13, ws2, gsc13, gsc2, nw, h, inter, y, meta, cnt, err, out, sink, ss, cnt2, "
             "xw, xw2, hot_t, M, lo, G, G3, probe, a2h, a2s, a2_mode, a2_mul, a2_unit, a2_smax")

GTCAL = r"""
// A2 lab: %globaltimer granularity and clock64 rate per SM (one CTA per SM, lane 0 polls for a fixed count)
__global__ void a2_gtcal_kernel(long long* out, int iters) {
  if (threadIdx.x != 0) return;
  long long g0, g1, last;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  const long long c0 = clock64();
  last = g0;
  long long mind = 1ll << 62;
  int changes = 0;
  for (int i = 0; i < iters; ++i) {
    long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    if (g != last) { if (g - last < mind) mind = g - last; ++changes; last = g; }
  }
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  const long long c1 = clock64();
  unsigned smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  long long* o = out + blockIdx.x * 8;
  o[0] = smid; o[1] = mind; o[2] = changes; o[3] = g1 - g0; o[4] = c1 - c0;
}
"""


def lab_source(labs=LABS, variants=VARIANTS, build=None):
    """(cuda source, cpp declarations) of the lab extension: one patched v3.3 copy per lab mask, each with the
    variant ids build[mask] (default BUILD; a vid outside it fails at run time with "unknown variant")."""
    build = BUILD if build is None else build
    src, wg = _shipped()
    for name, old, new in PATCHES:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"a2 patch {name!r}: anchor found {n} times (k2a_v33.py changed?)")
        src = src.replace(old, new)
    if src.count("tail_first(S, beg, end)") != 1:   # only a2_mid's own call is left
        raise RuntimeError("a2: a tail_first call site was not routed through a2_mid")
    hconst = ("constexpr int " + ", ".join(f"H_{k} = {v}" for k, v in H.items()) + ";\n"
              "constexpr int " + ", ".join(f"R_{k} = {v}" for k, v in R.items()) + ";")
    src = src.replace("@HREC@", str(HREC)).replace("@ROWS@", str(ROWS)).replace("@HCONST@", hconst)
    vv = [tuple(v) + (0,) * (4 - len(v)) for v in variants]
    if any(v[3] for v in vv):
        raise ValueError("a2 lab: HOTCAP variants are not instrumented")
    src = src.replace("@WGSPECS@", wg)
    k = src.index("namespace k3 {")
    head, body = src[:k], src[k:]
    parts = [head, GTCAL]
    for lab in labs:
        vids = [i for i in build.get(lab, range(len(vv))) if i < len(vv)]
        cases = "\n".join("    case %d: launch3<%d, %d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, *vv[i])
                          for i in vids)
        attrs = "\n".join("    case %d: return attrs3<%d, %d, %d, %d>();" % (i, *vv[i]) for i in vids)
        copy = body.replace("@CASES3@", cases).replace("@ATTRS3@", attrs)
        parts.append(f"\nnamespace a2L{lab} {{\nconstexpr int A2_LAB = {lab};\nconstexpr int A2_SMEM_MIN = {SMEM_MIN};\n"
                     f"{copy}\n}}  // namespace a2L{lab}\n")
    run_cases = "\n".join(f"    case {lab}: a2L{lab}::run3({RUN_NAMES}); break;" for lab in labs)
    attr_cases = "\n".join(f"    case {lab}: return a2L{lab}::attrs3v(vid);" for lab in labs)
    parts.append(
        f"\nvoid a2run(int64_t lab, {RUN_ARGS}) {{\n  switch (lab) {{\n{run_cases}\n"
        "    default: TORCH_CHECK(false, \"a2: lab mask not built\");\n  }\n}\n"
        f"std::vector<int64_t> a2attrs(int64_t lab, int64_t vid) {{\n  switch (lab) {{\n{attr_cases}\n"
        "    default: return std::vector<int64_t>{-1};\n  }\n}\n"
        "void a2gtcal(torch::Tensor out, int64_t iters) {\n"
        "  const int n = (int)(out.numel() / 8);\n"
        "  a2_gtcal_kernel<<<n, 32, 0, at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<long long*>(out.data_ptr()), (int)iters);\n"
        "  C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n")
    cpp = ("#include <torch/extension.h>\n#include <vector>\n"
           f"void a2run(int64_t lab, {RUN_ARGS});\n"
           "std::vector<int64_t> a2attrs(int64_t lab, int64_t vid);\n"
           "void a2gtcal(torch::Tensor out, int64_t iters);\n")
    return "".join(parts), cpp


CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def build(labs=LABS, variants=VARIANTS, name="a2_k2a_lab", verbose=False):
    """torch extension with a2run / a2attrs / a2gtcal (needs torch + nvcc). Returns (module, seconds)."""
    from torch.utils.cpp_extension import load_inline
    cuda, cpp = lab_source(labs, variants)
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=cuda,
                      functions=["a2run", "a2attrs", "a2gtcal"], extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0


# tools/box/sass_gate.py reads module-level *_SRC literals with ast; the lab source is generated, so the gate gets it
# through this placeholder: python3 kernels/lab/a2_k2a_lab.py box_out/a2_lab_gen.cu, then
# python3 tools/box/sass_gate.py kernels/lab/a2_k2a_lab.py --subst '@A2_LAB_CUDA@=file:box_out/a2_lab_gen.cu'
A2_GATE_SRC = "@A2_LAB_CUDA@"

if __name__ == "__main__":   # write the generated CUDA source (+ .cpp declarations) for sass_gate / an offline nvcc
    import sys
    cu, cp = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "a2_k2a_lab_gen.cu")
    out.write_text(cu)
    out.with_suffix(".cpp").write_text(cp)
    print(f"wrote {out} ({len(cu)} chars) and {out.with_suffix('.cpp')}")

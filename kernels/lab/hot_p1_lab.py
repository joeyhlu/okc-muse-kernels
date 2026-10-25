"""P1 lab kernel (kernels/muse_moe_w4a16/moe_v4_study.md §6/§9): K2A v3.7 as shipped in k3_v18
(kernels/muse_moe_w4a16/k2a_v37.py, KERNEL_SRC_V3), patched at anchors, for the hot-path anatomy at T = 128 / 256.

LAB FILE. Box runs only, through tools/box/hot_p1.py. Never submit it, and never import it from a scored file.

The CUDA source is read from k2a_v37.py through ast, so this module imports without torch. It is patched at anchors:
every `old` must occur exactly once inside its region, or lab_source() raises. The result is emitted once per
consumer mask, each copy in its own namespace `hpL<m>` with `constexpr int HP_LAB = <m>`. The consumer patches sit
behind `if constexpr`, so the m = 0 copy's consumers compile to v3.7's code. The m = 0 copy is checked against the
shipped module in the same process: outputs, registers and µs.

Compile-time consumer masks (HP_LAB; one namespace copy each):
  HP_MMAOFF 1  hot items without tensor work. HOTCAP (vid 2) hot_item: no wgmma, so no X smem reads by wgmma either.
               HK3 (vid 5) hc_item: no mma.sync and no X LDS. The decoded A registers are XOR-sunk.
  HP_DEQOFF 2  hot items without dequant: A = raw code words & 0x3F3F3F3F (finite bf16, 0.25 op/register instead of
               deq4 + bmul2). Tensor work unchanged.
  HP_NULLC  4  every consumer warp (hot and cold) only waits on `full` and releases `empty`: no smem reads, math,
               flush or hand-off. HOTCAP/cold gemm: the epilogue warp returns at once. gemm37: consumers post only the
               final END, so the epilogue exits. Producer, ring, copies, PDL chain and schedule are unchanged.
Runtime producer / launch bits (Params.hp, run3's new `hp` argument; producer warp and host only, never in a wgmma path):
  HP_XHOT   1  hot items' X copies are skipped: expect_tx shrinks and consumers read stale smem. K1 hot X comes from P0,
               K2 hot X from xw2 / interS.
  HP_XCOLD  2  the same for cold items (K1 x rows, K2 interS rows).
  HP_RPW    4  passes j >= 1 of an (expert, row block) skip the weight copy (stale weights): the cost of re-reads.
  HP_NOWAIT2 8 K2's producer never takes griddepcontrol.wait; its X copies read stale interS / xw2. This is the K1 -> K2
               hand-off ceiling for any fusion. Timing only.
  HP_NOP0  16  HOTCAP: P0 is not launched; K1 hot X reads the stale xw. This is P0's exposed cost.
Clock and role marks: Params.hpclk (run3's new `hpclk` tensor; empty = off). Thread 0 of each K1/K2 CTA writes
[smid, role, clock64 at gemm start, %globaltimer at gemm start, clock64 at gemm end, %globaltimer at gemm end,
%globaltimer at kernel entry, 0], with role 1 = HOTCAP hot, 0 = cold, 2 = HK3 mixed. The record is at
((k * 1024 + cta) * 8), with k 0 = K1, 1 = K2. That is enough for per-role spans, chip work (Σ busy / G) and MHz per CTA.
Waits stay v3.7's bounded mbar_wait / mbar_wait_ok, which set an error flag (R9). The lab adds no new wait loop:
hp_null uses mbar_wait.
"""
import ast
import hashlib
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
K2A_FILE = ROOT / "kernels/muse_moe_w4a16/k2a_v37.py"

HP_MMAOFF, HP_DEQOFF, HP_NULLC = 1, 2, 4
HP_XHOT, HP_XCOLD, HP_RPW, HP_NOWAIT2, HP_NOP0 = 1, 2, 4, 8, 16
COPIES = (0, HP_MMAOFF, HP_DEQOFF, HP_NULLC)
# vid -> variant tuple (= k2a_v37.V3_VARIANTS[vid]; asserted below). vid 0: v3.3 NC1 (bs1/8 shipped; the small-T
# fusion-ceiling rows), vid 2: v3.3 HOTCAP (T = 256 shipped), vid 5: v3.7 HK3 NCH 4 (T = 128 shipped)
VIDS = {0: (1, 3, 2, 0), 2: (2, 2, 2, 1), 5: (1, 3, 2, 3, 4)}
BUILD = {0: (0, 2, 5), HP_MMAOFF: (2, 5), HP_DEQOFF: (2, 5), HP_NULLC: (2, 5)}
CLK_STRIDE = 1024   # CTAs per kernel record block (grids are 264)


def _shipped():
    """(KERNEL_SRC_V3, _wg_specs(), V3_VARIANTS) of k2a_v37.py, without importing it (it needs torch)."""
    tree = ast.parse(K2A_FILE.read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("KERNEL_SRC_V3", "V3_VARIANTS")
                                                for t in node.targets):
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "_wg_specs":
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
    return ns["KERNEL_SRC_V3"], ns["_wg_specs"](), ns["V3_VARIANTS"]


HELPERS = r"""
// ======== hot_p1 lab helpers (kernels/lab/hot_p1_lab.py) ========
__device__ __forceinline__ long long hp_gt() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// slot 0: kernel entry (smid, %globaltimer); 1: gemm start (role, clock64, %globaltimer); 2: gemm end (clock64, %globaltimer)
__device__ __forceinline__ void hp_mark(const Params& P, int k, int slot, int role) {
  if (P.hpclk == nullptr || threadIdx.x != 0 || blockIdx.x >= 1024) return;
  long long* h = P.hpclk + ((size_t)k * 1024 + blockIdx.x) * 8;
  const long long c = clock64();
  const long long g = hp_gt();
  if (slot == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    h[0] = (long long)smid;
    h[6] = g;
  } else if (slot == 1) {
    h[1] = role; h[2] = c; h[3] = g;
  } else {
    h[4] = c; h[5] = g;
  }
}
// HP_NULLC consumer: walk the item's stages, wait full, release empty (one arrival per warp, as the real consumers)
template <int NST>
__device__ __forceinline__ void hp_null(const Params& P, uint32_t full_a, uint32_t empty_a, int& it, int s0, int s1) {
  const int lane = threadIdx.x & 31;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
}
"""

# (name, region start, region end, old, new). Region = [first occurrence of start, the next occurrence of end after it);
# None = whole source. `old` must occur exactly once in the region.
PATCHES = [
    ("lab constants", None, None, "namespace k3 {\n",
     "namespace k3 {\nconstexpr int HP_LAB = @HPLAB@;\n"
     "constexpr int HP_MMAOFF = 1, HP_DEQOFF = 2, HP_NULLC = 4;\n"
     "constexpr int HP_XHOT = 1, HP_XCOLD = 2, HP_RPW = 4, HP_NOWAIT2 = 8, HP_NOP0 = 16;\n"),
    ("params", None, None, "  int M, lo, probe;\n};",
     "  int M, lo, probe;\n  int hp; long long* hpclk;   // hot_p1 lab: producer / launch bits, clock + role marks\n};"),
    ("helpers", None, None, "constexpr int DQ37 = 120;\n", HELPERS + "constexpr int DQ37 = 120;\n"),
    # ---------------- v3.3 HOTCAP hot_item (wgmma RS) ----------------
    ("hot_item sink", "__device__ __forceinline__ void hot_item(", "// HM 0: cold items (mma.sync)",
     "  float hacc[N / 2];\n", "  float hacc[N / 2];\n  uint32_t hsink = 0u;\n"),
    ("hot_item kb", "__device__ __forceinline__ void hot_item(", "// HM 0: cold items (mma.sync)",
     "      if (i >= 2) wg_wait<1>();\n"
     "      uint32_t* h = hb[i & 1];\n"
     "      deq4(w.x, h); deq4(w.y, h + 4); deq4(w.z, h + 8); deq4(w.w, h + 12);\n"
     "#pragma unroll\n"
     "      for (int jj = 0; jj < 8; ++jj) { h[jj] = bmul2(h[jj], Sg); h[8 + jj] = bmul2(h[8 + jj], S8); }\n"
     "      wg_fence();\n"
     "      wgmma_rs<N>(hacc, h[0], h[8], h[1], h[9], d0 + 64 * i);\n"
     "      wgmma_rs<N>(hacc, h[2], h[10], h[3], h[11], d0 + 64 * i + 16);\n"
     "      wgmma_rs<N>(hacc, h[4], h[12], h[5], h[13], d0 + 64 * i + 32);\n"
     "      wgmma_rs<N>(hacc, h[6], h[14], h[7], h[15], d0 + 64 * i + 48);\n"
     "      wg_commit();\n"
     "    }\n"
     "    wg_wait<0>();\n",
     "      if constexpr ((HP_LAB & HP_MMAOFF) == 0) {\n"
     "        if (i >= 2) wg_wait<1>();\n"
     "      }\n"
     "      uint32_t* h = hb[i & 1];\n"
     "      if constexpr ((HP_LAB & HP_DEQOFF) != 0) {\n"
     "        const uint32_t m0 = w.x & 0x3F3F3F3Fu, m1 = w.y & 0x3F3F3F3Fu, m2 = w.z & 0x3F3F3F3Fu, m3 = w.w & 0x3F3F3F3Fu;\n"
     "#pragma unroll\n"
     "        for (int jj = 0; jj < 4; ++jj) { h[4 * jj] = m0; h[4 * jj + 1] = m1; h[4 * jj + 2] = m2; h[4 * jj + 3] = m3; }\n"
     "      } else {\n"
     "        deq4(w.x, h); deq4(w.y, h + 4); deq4(w.z, h + 8); deq4(w.w, h + 12);\n"
     "#pragma unroll\n"
     "        for (int jj = 0; jj < 8; ++jj) { h[jj] = bmul2(h[jj], Sg); h[8 + jj] = bmul2(h[8 + jj], S8); }\n"
     "      }\n"
     "      if constexpr ((HP_LAB & HP_MMAOFF) != 0) {\n"
     "#pragma unroll\n"
     "        for (int jj = 0; jj < 16; ++jj) hsink ^= h[jj];\n"
     "      } else {\n"
     "        wg_fence();\n"
     "        wgmma_rs<N>(hacc, h[0], h[8], h[1], h[9], d0 + 64 * i);\n"
     "        wgmma_rs<N>(hacc, h[2], h[10], h[3], h[11], d0 + 64 * i + 16);\n"
     "        wgmma_rs<N>(hacc, h[4], h[12], h[5], h[13], d0 + 64 * i + 32);\n"
     "        wgmma_rs<N>(hacc, h[6], h[14], h[7], h[15], d0 + 64 * i + 48);\n"
     "        wg_commit();\n"
     "      }\n"
     "    }\n"
     "    if constexpr ((HP_LAB & HP_MMAOFF) == 0) wg_wait<0>();\n"),
    ("hot_item sink out", "__device__ __forceinline__ void hot_item(", "// HM 0: cold items (mma.sync)",
     "  it_ref = it;\n  // ---- flush",
     "  it_ref = it;\n  if constexpr ((HP_LAB & HP_MMAOFF) != 0) { if (hsink == 0x9E3779B9u) atomicAdd(P.sink, 1u); }\n"
     "  // ---- flush"),
    # ---------------- v3.3 gemm (HOTCAP roles and the cold variants): producer bits, NULL consumers ----------------
    ("gemm wait", "__device__ void gemm(const Params& P, Route& S,", "__device__ __forceinline__ void setup_bars(",
     "    bool waited = !pdl;\n", "    bool waited = !pdl || (MODE == 2 && (P.hp & HP_NOWAIT2) != 0);\n"),
    ("gemm xbytes", "__device__ void gemm(const Params& P, Route& S,", "__device__ __forceinline__ void setup_bars(",
     "      const uint32_t xbytes = hot ? ((ntok + 7) >> 3) * 4096 : (MODE == 1) ? ntx * 512 : ntok * XROW;\n",
     "      const uint32_t xbytes = (P.hp & (hot ? HP_XHOT : HP_XCOLD)) != 0\n"
     "          ? 0u : (uint32_t)(hot ? ((ntok + 7) >> 3) * 4096 : (MODE == 1) ? ntx * 512 : ntok * XROW);\n"
     "      const bool wskip = (P.hp & HP_RPW) != 0 && j > 0;\n"),
    ("gemm copies", "__device__ void gemm(const Params& P, Route& S,", "__device__ __forceinline__ void setup_bars(",
     "        if (lane == 0) {\n"
     "          mbar_expect_tx(fb, SW + xbytes);\n"
     "          const uint8_t* src = wb + (size_t)s * SW;\n"
     "          if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);\n"
     "          else bulk_g2s(dst, src, SW, fb);\n"
     "        }\n"
     "        mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals\n"
     "        __syncwarp();\n"
     "        if (MODE == 1 && !hot) {\n",
     "        if (lane == 0) {\n"
     "          const uint32_t tot = (wskip ? 0u : (uint32_t)SW) + xbytes;\n"
     "          if (tot != 0u) mbar_expect_tx(fb, tot);\n"
     "          else mbar_arrive(fb);\n"
     "          const uint8_t* src = wb + (size_t)s * SW;\n"
     "          if (!wskip) {\n"
     "            if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);\n"
     "            else bulk_g2s(dst, src, SW, fb);\n"
     "          }\n"
     "        }\n"
     "        mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals\n"
     "        __syncwarp();\n"
     "        if (xbytes == 0u) {\n"
     "          // hot_p1 HP_XHOT / HP_XCOLD: no X copy (consumers read stale smem)\n"
     "        } else if (MODE == 1 && !hot) {\n"),
    ("gemm epilogue null", "__device__ void gemm(const Params& P, Route& S,", "__device__ __forceinline__ void setup_bars(",
     "      if ((P.probe & 15) != 0) return;   // probes: consumers never hand over\n",
     "      if ((P.probe & 15) != 0 || (HP_LAB & HP_NULLC) != 0) return;   // probes: consumers never hand over\n"),
    ("gemm consumer null", "__device__ void gemm(const Params& P, Route& S,", "__device__ __forceinline__ void setup_bars(",
     "    const int nw = ((ntok + 15) >> 4) << 4;   // hot: wgmma N in {16, 32, 48, 64}\n",
     "    const int nw = ((ntok + 15) >> 4) << 4;   // hot: wgmma N in {16, 32, 48, 64}\n"
     "    if constexpr ((HP_LAB & HP_NULLC) != 0) {\n"
     "      hp_null<NST>(P, full_a, empty_a, it, s0, s1);\n"
     "      x += s1 - s0;\n"
     "      continue;\n"
     "    }\n"),
    # ---------------- v3.6 hc_item (HK3 hot items on mma.sync) ----------------
    ("hc_item sink", "__device__ __forceinline__ void hc_item(", "// v3.6 gemm over four segments",
     "  float acc[NQ][4];\n", "  float acc[NQ][4];\n  uint32_t hsink = 0u;\n"),
    ("hc_item kb", "__device__ __forceinline__ void hc_item(", "// v3.6 gemm over four segments",
     "        uint32_t a[4], b[4];\n"
     "        deq4(hf ? w.y : w.x, a);   // gate rows g: consume_kb's ag[4 hf .. 4 hf + 3]\n"
     "        deq4(hf ? w.w : w.z, b);   // rows g + 8:  consume_kb's a8[4 hf .. 4 hf + 3]\n"
     "#pragma unroll\n"
     "        for (int jj = 0; jj < 4; ++jj) { a[jj] = bmul2(a[jj], Sg); b[jj] = bmul2(b[jj], S8); }\n"
     "#pragma unroll\n"
     "        for (int q = 0; q < NQ; ++q) {\n"
     "          const uint4 xv = *reinterpret_cast<const uint4*>(xrow + q * 8 * XST + i * 64 + 8 * hf);   // x0 / x1\n"
     "          mma16816(acc[q], a[0], b[0], a[1], b[1], xv.x, xv.y);\n"
     "          mma16816(acc[q], a[2], b[2], a[3], b[3], xv.z, xv.w);\n"
     "        }\n",
     "        uint32_t a[4], b[4];\n"
     "        if constexpr ((HP_LAB & HP_DEQOFF) != 0) {\n"
     "          const uint32_t ma = (hf ? w.y : w.x) & 0x3F3F3F3Fu, mb = (hf ? w.w : w.z) & 0x3F3F3F3Fu;\n"
     "#pragma unroll\n"
     "          for (int jj = 0; jj < 4; ++jj) { a[jj] = ma; b[jj] = mb; }\n"
     "        } else {\n"
     "          deq4(hf ? w.y : w.x, a);   // gate rows g: consume_kb's ag[4 hf .. 4 hf + 3]\n"
     "          deq4(hf ? w.w : w.z, b);   // rows g + 8:  consume_kb's a8[4 hf .. 4 hf + 3]\n"
     "#pragma unroll\n"
     "          for (int jj = 0; jj < 4; ++jj) { a[jj] = bmul2(a[jj], Sg); b[jj] = bmul2(b[jj], S8); }\n"
     "        }\n"
     "        if constexpr ((HP_LAB & HP_MMAOFF) != 0) {\n"
     "#pragma unroll\n"
     "          for (int jj = 0; jj < 4; ++jj) hsink ^= a[jj] ^ b[jj];\n"
     "        } else {\n"
     "#pragma unroll\n"
     "          for (int q = 0; q < NQ; ++q) {\n"
     "            const uint4 xv = *reinterpret_cast<const uint4*>(xrow + q * 8 * XST + i * 64 + 8 * hf);   // x0 / x1\n"
     "            mma16816(acc[q], a[0], b[0], a[1], b[1], xv.x, xv.y);\n"
     "            mma16816(acc[q], a[2], b[2], a[3], b[3], xv.z, xv.w);\n"
     "          }\n"
     "        }\n"),
    ("hc_item sink out", "__device__ __forceinline__ void hc_item(", "// v3.6 gemm over four segments",
     "  it_ref = it;\n  constexpr bool NOSWI = (MODE == 1 && NQ >= 3);\n",
     "  it_ref = it;\n  if constexpr ((HP_LAB & HP_MMAOFF) != 0) { if (hsink == 0x9E3779B9u) atomicAdd(P.sink, 1u); }\n"
     "  constexpr bool NOSWI = (MODE == 1 && NQ >= 3);\n"),
    # ---------------- v3.7 gemm37 (HK3): producer bits, NULL consumers ----------------
    ("gemm37 wait", "__device__ void gemm37(", "// K1 / K2 of the HK3 variants",
     "    bool waited = !pdl;\n", "    bool waited = !pdl || (MODE == 2 && (P.hp & HP_NOWAIT2) != 0);\n"),
    ("gemm37 xbytes", "__device__ void gemm37(", "// K1 / K2 of the HK3 variants",
     "        const uint32_t xbytes = (MODE == 1) ? ntok * 512 : ntok * XROW;\n",
     "        const uint32_t xbytes = (P.hp & (hk ? HP_XHOT : HP_XCOLD)) != 0\n"
     "            ? 0u : (uint32_t)((MODE == 1) ? ntok * 512 : ntok * XROW);\n"
     "        const bool wskip = (P.hp & HP_RPW) != 0 && j > 0;\n"),
    ("gemm37 copies", "__device__ void gemm37(", "// K1 / K2 of the HK3 variants",
     "          if (lane == 0) {\n"
     "            s_tag[slot] = tag;\n"
     "            mbar_expect_tx(fb, SW + xbytes);\n"
     "            const uint8_t* src = wb + (size_t)s * SW;\n"
     "            if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);\n"
     "            else bulk_g2s(dst, src, SW, fb);\n"
     "          }\n"
     "          mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals\n"
     "          __syncwarp();\n"
     "          if (MODE == 1) {\n",
     "          if (lane == 0) {\n"
     "            s_tag[slot] = tag;\n"
     "            const uint32_t tot = (wskip ? 0u : (uint32_t)SW) + xbytes;\n"
     "            if (tot != 0u) mbar_expect_tx(fb, tot);\n"
     "            else mbar_arrive(fb);\n"
     "            const uint8_t* src = wb + (size_t)s * SW;\n"
     "            if (!wskip) {\n"
     "              if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);\n"
     "              else bulk_g2s(dst, src, SW, fb);\n"
     "            }\n"
     "          }\n"
     "          mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals\n"
     "          __syncwarp();\n"
     "          if (xbytes == 0u) {\n"
     "            // hot_p1 HP_XHOT / HP_XCOLD: no X copy (consumers read stale smem)\n"
     "          } else if (MODE == 1) {\n"),
    ("gemm37 consumer null", "__device__ void gemm37(", "// K1 / K2 of the HK3 variants",
     "      const bool ho = (MODE == 1) && hk && nch >= 3;\n",
     "      const bool ho = (MODE == 1) && hk && nch >= 3;\n"
     "      if constexpr ((HP_LAB & HP_NULLC) != 0) {\n"
     "        hp_null<NST>(P, full_a, empty_a, it, s0, s1);\n"
     "        x += s1 - s0;\n"
     "        continue;\n"
     "      }\n"),
    # ---------------- clock / role marks in the four K1 / K2 kernels ----------------
    ("k1_37 entry", "k1_kernel37(const Params P) {", "k2_kernel37(const Params P) {",
     "  pdl_launch();\n", "  pdl_launch();\n  hp_mark(P, 0, 0, 2);\n"),
    ("k1_37 gemm", "k1_kernel37(const Params P) {", "k2_kernel37(const Params P) {",
     "  gemm37<NCH, NST, 1>(P, S, s_list, smem, bars, false);\n",
     "  hp_mark(P, 0, 1, 2);\n  gemm37<NCH, NST, 1>(P, S, s_list, smem, bars, false);\n  hp_mark(P, 0, 2, 2);\n"),
    ("k2_37 entry", "k2_kernel37(const Params P) {", "// K1: route + W13 (+ SwiGLU).",
     "  pdl_launch();\n", "  pdl_launch();\n  hp_mark(P, 1, 0, 2);\n"),
    ("k2_37 gemm", "k2_kernel37(const Params P) {", "// K1: route + W13 (+ SwiGLU).",
     "  gemm37<NCH, NST, 2>(P, S, nullptr, smem, bars, true);\n",
     "  hp_mark(P, 1, 1, 2);\n  gemm37<NCH, NST, 2>(P, S, nullptr, smem, bars, true);\n  hp_mark(P, 1, 2, 2);\n"),
    ("k1 entry", "k1_kernel(const Params P) {", "// K2: route + W2.",
     "  pdl_launch();\n", "  pdl_launch();\n  hp_mark(P, 0, 0, 0);\n"),
    ("k1 gemm hot", "k1_kernel(const Params P) {", "// K2: route + W2.",
     "    if (hr) gemm<NC, NST, 1, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    else gemm<NC, NSTC, 1, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);\n",
     "    hp_mark(P, 0, 1, hr ? 1 : 0);\n"
     "    if (hr) gemm<NC, NST, 1, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    else gemm<NC, NSTC, 1, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    hp_mark(P, 0, 2, 0);\n"),
    ("k1 gemm cold", "k1_kernel(const Params P) {", "// K2: route + W2.",
     "    gemm<NC, NST, 1, 0, STG>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);\n",
     "    hp_mark(P, 0, 1, 0);\n"
     "    gemm<NC, NST, 1, 0, STG>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);\n"
     "    hp_mark(P, 0, 2, 0);\n"),
    ("k2 entry", "k2_kernel(const Params P) {", "// P0 (v3.3)",
     "  pdl_launch();\n", "  pdl_launch();\n  hp_mark(P, 1, 0, 0);\n"),
    ("k2 gemm hot", "k2_kernel(const Params P) {", "// P0 (v3.3)",
     "    if (hr) gemm<NC, NST, 2, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    else gemm<NC, NSTC, 2, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);\n",
     "    hp_mark(P, 1, 1, hr ? 1 : 0);\n"
     "    if (hr) gemm<NC, NST, 2, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    else gemm<NC, NSTC, 2, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    hp_mark(P, 1, 2, 0);\n"),
    ("k2 gemm cold", "k2_kernel(const Params P) {", "// P0 (v3.3)",
     "    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);\n",
     "    hp_mark(P, 1, 1, 0);\n"
     "    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);\n"
     "    hp_mark(P, 1, 2, 0);\n"),
    # ---------------- host: P0 skip, run3's two new arguments ----------------
    ("launch3 p0", None, None,
     "      k3::p0_kernel<NC><<<G3, 256, 0, st>>>(p);\n      C10_CUDA_KERNEL_LAUNCH_CHECK();\n",
     "      if ((p.hp & k3::HP_NOP0) == 0) {\n"
     "        k3::p0_kernel<NC><<<G3, 256, 0, st>>>(p);\n"
     "        C10_CUDA_KERNEL_LAUNCH_CHECK();\n"
     "      }\n"),
    ("run3 args", None, None,
     "int64_t G3, int64_t probe) {\n  k3::Params p;\n",
     "int64_t G3, int64_t probe, int64_t hp,\n          torch::Tensor hpclk) {\n  k3::Params p;\n"
     "  p.hp = (int)hp;\n"
     "  p.hpclk = hpclk.numel() >= 2 * 1024 * 8 ? reinterpret_cast<long long*>(hpclk.data_ptr()) : nullptr;\n"),
]

RUN_ARGS = ("int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, "
            "torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, "
            "torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err, "
            "torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw, "
            "torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe, "
            "int64_t hp, torch::Tensor hpclk")
RUN_NAMES = ("vid, mask, x, ids, wts, ws13, ws2, gsc13, gsc2, nw, h, inter, y, meta, cnt, err, out, sink, ss, cnt2, "
             "xw, xw2, hot_t, M, lo, G, G3, probe, hp, hpclk")


def _region(src, start, end):
    if start is None:
        return 0, len(src)
    a = src.find(start)
    if a < 0:
        raise RuntimeError(f"hot_p1: region start {start!r} not found (k2a_v37.py changed?)")
    b = src.find(end, a + len(start)) if end is not None else len(src)
    if b < 0:
        raise RuntimeError(f"hot_p1: region end {end!r} not found after {start!r}")
    return a, b


def patched_source():
    """v3.7's KERNEL_SRC_V3 with every PATCHES entry applied (placeholders @HPLAB@ / @CASES3@ / @ATTRS3@ left)."""
    src, wg, variants = _shipped()
    for vid, v in VIDS.items():
        if tuple(variants[vid]) != v:
            raise RuntimeError(f"hot_p1: k2a_v37 V3_VARIANTS[{vid}] = {variants[vid]}, lab expects {v}")
    for name, start, end, old, new in PATCHES:
        a, b = _region(src, start, end)
        n = src[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"hot_p1 patch {name!r}: anchor found {n} times in its region (k2a_v37.py changed?)")
        i = src.index(old, a)
        src = src[:i] + new + src[i + len(old):]
    return src.replace("@WGSPECS@", wg)


def _case(vid):
    v = VIDS[vid]
    nc, nst, minb, hk, nch = (tuple(v) + (0,) * (5 - len(v)))[:5]
    if hk == 3:
        return (f"    case {vid}: launch37<{nch}, {nst}>(p, (int)G, (int)G3, (int)mask); break;",
                f"    case {vid}: return attrs37<{nch}, {nst}>();")
    if hk in (0, 1):
        return (f"    case {vid}: launch3<{nc}, {nst}, {minb}, {hk}>(p, (int)G, (int)G3, (int)mask); break;",
                f"    case {vid}: return attrs3<{nc}, {nst}, {minb}, {hk}>();")
    raise ValueError(v)


def lab_source(copies=COPIES, build=None):
    """(cuda source, cpp declarations): one patched copy per consumer mask, each with its BUILD vids."""
    build = BUILD if build is None else build
    src = patched_source()
    k = src.index("namespace k3 {")
    head, body = src[:k], src[k:]
    parts = [head]
    for m in copies:
        cs = [_case(v) for v in build[m]]
        copy = (body.replace("@HPLAB@", str(m)).replace("@CASES3@", "\n".join(c[0] for c in cs))
                .replace("@ATTRS3@", "\n".join(c[1] for c in cs)))
        parts.append(f"\nnamespace hpL{m} {{\n{copy}\n}}  // namespace hpL{m}\n")
    run_cases = "\n".join(f"    case {m}: hpL{m}::run3({RUN_NAMES}); break;" for m in copies)
    attr_cases = "\n".join(f"    case {m}: return hpL{m}::attrs3v(vid);" for m in copies)
    parts.append(
        f"\nvoid hprun(int64_t lab, {RUN_ARGS}) {{\n  switch (lab) {{\n{run_cases}\n"
        "    default: TORCH_CHECK(false, \"hot_p1: consumer mask not built\");\n  }\n}\n"
        f"std::vector<int64_t> hpattrs(int64_t lab, int64_t vid) {{\n  switch (lab) {{\n{attr_cases}\n"
        "    default: return std::vector<int64_t>{-1};\n  }\n}\n")
    cpp = ("#include <torch/extension.h>\n#include <vector>\n"
           f"void hprun(int64_t lab, {RUN_ARGS});\n"
           "std::vector<int64_t> hpattrs(int64_t lab, int64_t vid);\n")
    return "".join(parts), cpp


CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def build(copies=COPIES, name="hot_p1_lab", verbose=False):
    """torch extension with hprun / hpattrs (needs torch + nvcc). Returns (module, seconds)."""
    from torch.utils.cpp_extension import load_inline
    cuda, cpp = lab_source(copies)
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=cuda, functions=["hprun", "hpattrs"],
                      extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0


if __name__ == "__main__":   # write the generated CUDA source (+ .cpp declarations) for review / an offline nvcc
    import sys
    cu, cp = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "hot_p1_lab_gen.cu")
    out.write_text(cu, encoding="utf-8")
    out.with_suffix(".cpp").write_text(cp, encoding="utf-8")
    print(f"wrote {out} ({len(cu)} chars) and {out.with_suffix('.cpp')}")

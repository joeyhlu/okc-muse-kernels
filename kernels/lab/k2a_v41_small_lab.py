"""K2A v4.1 small-T lab (T = 1 / 8 / 16 / 32: vid 0 = v3.3 NC1 cold, vid 3 = v3.6 HK2 NCH 4): K2A v3.7's source
(kernels/muse_moe_w4a16/k2a_v37.py, KERNEL_SRC_V3 = the K2A33 source embedded in k3_v21.py) with the small-T consumers
routed through lab copies of cold_item / hc_item (v41_cold_item / v41_hc_item), emitted once per COPY in its own
namespace `v41_<name>`.

LAB FILE. Box runs only, through tools/box/k2a_v41.py. Never submit it, never import it from a scored file. The patches
for the layer file are exported by tools/k2a_v41_patches.py (same anchors, the winning knobs fixed).

Anchors (each occurs once in k2a_v37.py's KERNEL_SRC_V3 and once in k3_v21.py; none is touched by the v40 patches of
kernels/lab/k2a_v40_lab.py, which edit hot_item / the HOTCAP flush / the NC 2 cold loop / pick_role):
  A_FUNCS  before "// One cold item (<= 8 tokens): v3.3's NC1 consumer": the v41 functions + knobs.
  A_V0     vid 0's gemm (v3.3, HM 0, NC 1): the item goes to v41_cold_item before the `if constexpr (HOTCAP)` branch
           (cold_item is v3.3's NC1 consumer + flush verbatim, so this is bitwise the inline path).
  A_V3     gemm36 (vid 3): cold_item / hc_item<2..4> calls -> v41_cold_item / v41_hc_item. gemm37 (vid 5) unchanged.
Per copy (compile-time):
  V41_VAR  bit 1 RING  slot / parity kept incrementally (one % and / per item instead of per stage); bitwise.
           bit 2 BASE  per-lane shared base pointers formed once per item (codes, scales, X), the stage offset added once
                       per stage; bitwise.
           bit 4 DUAL4 (cold item) four accumulator chains (one per kb) instead of two; NOT bitwise (FP32 order).
           (vid 0's item already runs without v3.3's per-stage runtime probe tests: cold_item has none.)
  V41_MASK lab only, results invalid: 1 no mma and no X loads (decoded A XOR-sunk), 2 no dequant (A = raw codes),
           4 null consumer (wait full, release empty; flush of zero accumulators keeps the hand-off protocol).
Waits stay v3.7's bounded mbar_wait (R9).
"""
import ast
import hashlib
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
K2A_FILE = ROOT / "kernels/muse_moe_w4a16/k2a_v37.py"
LAYER_FILE = ROOT / "kernels/muse_decode_layer/k3_v21.py"

VIDS = {0: (1, 3, 2, 0), 3: (1, 3, 2, 2, 4), 6: (1, 4, 2, 0), 7: (1, 2, 2, 0)}   # 6 / 7: lab-only vid 0 with NST 4 / 2
SHIPPED_VIDS = (0, 3)
RING, BASE, DUAL4 = 1, 2, 4
M_NOMMA, M_NODEQ, M_NULL = 1, 2, 4
# name -> (V41_VAR, V41_MASK)
COPIES = {
    "base": (0, 0),
    "null": (0, M_NULL), "nomma": (0, M_NOMMA), "nodeq": (0, M_NODEQ),
    "ring": (RING, 0), "rb": (RING | BASE, 0), "rbd": (RING | BASE | DUAL4, 0),
    "xoff": (0, 8), "k2nw": (0, 16), "xk": (0, 24), "nullxk": (0, 28),
}

A_FUNCS = "// One cold item (<= 8 tokens): v3.3's NC1 consumer"
A_V0 = "    if constexpr (HOTCAP) {\n      // warm-expert item on wgmma (this CTA's role has only warm items)\n"
A_V3 = ("        if (nch <= 1) {\n          cold_item<NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart);\n"
        "        } else if constexpr (NCH >= 4) {\n"
        "          if (nch >= 4) hc_item<4, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, ho);\n"
        "          else if (nch == 3) hc_item<3, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, ho);\n"
        "          else hc_item<2, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, false);\n"
        "        } else {\n"
        "          hc_item<2, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, false);\n"
        "        }\n")

FUNCS = r"""// ======== k2a_v41 (small-T consumers): cold / hot-chunk items with leaner per-stage bookkeeping ========
constexpr int V41_VAR = @V41VAR@;    // bit 1 RING, 2 BASE, 4 DUAL4 (not bitwise)
constexpr int V41_MASK = @V41MASK@;  // lab only (results invalid): 1 no mma / X loads, 2 no dequant, 4 null consumer
// One kb of this warp's tile, 1 chunk (consume_kb<1>'s arithmetic and mma order); xq = X row g at k = kloc + 16 t.
__device__ __forceinline__ void v41_kb1(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xq, float* acc,
                                        uint32_t& sink) {
  uint32_t ag[8], a8[8];
  if constexpr ((V41_MASK & 2) != 0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) { ag[j] = w.x & 0x3F3F3F3Fu; ag[4 + j] = w.y & 0x3F3F3F3Fu; a8[j] = w.z & 0x3F3F3F3Fu; a8[4 + j] = w.w & 0x3F3F3F3Fu; }
  } else {
    const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
    const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
    deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
    for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  }
  if constexpr ((V41_MASK & 1) != 0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) sink ^= ag[j] ^ a8[j];
    return;
  }
  const uint4* xp = reinterpret_cast<const uint4*>(xq);
  const uint4 x0 = xp[0], x1 = xp[1];
  mma16816(acc, ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
  mma16816(acc, ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
  mma16816(acc, ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
  mma16816(acc, ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
}

// cold_item (<= 8 tokens) with the V41_VAR / V41_MASK knobs; V41_VAR 0 (or RING / BASE), mask 0 = cold_item bit for bit.
template <int NST, int STG, int MODE, bool MB = false>
__device__ __forceinline__ void v41_cold_item(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                              int& it_ref, int s0, int s1, int e, int rb, int c0, int ntok, int npart) {
  constexpr bool RINGV = (V41_VAR & 1) != 0, BASEV = (V41_VAR & 2) != 0, D4 = (V41_VAR & 4) != 0;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  uint32_t sink = 0u;
  float acc[4][4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[k][i] = 0.f;
  int slot = it % NST;
  uint32_t par = (uint32_t)(it / NST) & 1u;
  const uint8_t* wb = ring + warp * 2048 + lane * 16;
  const uint8_t* sbp = ring + 16384 + warp * 256 + lane * 4;
  const uint8_t* xbp = ring + SW + 2 * (g * XST + 16 * t);
  for (int s = s0; s < s1; ++s, ++it) {
    if constexpr (!RINGV) { slot = it % NST; par = (uint32_t)(it / NST) & 1u; }
    mbar_wait(full_a + 8 * slot, par, P.err);
    if constexpr ((V41_MASK & 4) == 0) {
      const uint4* wp;
      const uint32_t* sp;
      const __nv_bfloat16* xq;
      if constexpr (BASEV) {
        const int so = slot * STG;
        wp = reinterpret_cast<const uint4*>(wb + so);
        sp = reinterpret_cast<const uint32_t*>(sbp + so);
        xq = reinterpret_cast<const __nv_bfloat16*>(xbp + so);
      } else {
        const uint8_t* st = ring + slot * STG;
        wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
        sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
        xq = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST + 16 * t;
      }
      uint4 wv[4];
      uint32_t sv[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
      sv[0] = sp[0]; sv[1] = sp[32];
#pragma unroll
      for (int i = 0; i < 4; ++i) v41_kb1(wv[i], sv[i >> 1], i & 1, xq + i * 64, acc[D4 ? i : (i >> 1)], sink);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    if constexpr (RINGV) { if (++slot == NST) { slot = 0; par ^= 1u; } }
  }
  it_ref = it;
  if constexpr (D4) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[0][i] = (acc[0][i] + acc[1][i]) + (acc[2][i] + acc[3][i]);
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[0][i] += acc[1][i];
  }
  if (sink == 0x9E3779B9u) atomicAdd(P.sink, 1u);
  const bool whole = s0 == 0 && s1 == 16;
  flush_item<1, MODE>(P, acc, whole, e, rb, c0, ntok, npart, MB ? false : !whole);
}

// hc_item<NQ> (2..4 chunks) with the V41_VAR (RING, BASE) / V41_MASK knobs; VAR 0, mask 0 = hc_item bit for bit.
template <int NQ, int NST, int STG, int MODE, bool MB = false>
__device__ __forceinline__ void v41_hc_item(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                            int& it_ref, int s0, int s1, int e, int rb, int c0, int ntok, int npart,
                                            bool ho) {
  constexpr bool RINGV = (V41_VAR & 1) != 0, BASEV = (V41_VAR & 2) != 0;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  uint32_t sink = 0u;
  float acc[NQ][4];
#pragma unroll
  for (int q = 0; q < NQ; ++q)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[q][i] = 0.f;
  int slot = it % NST;
  uint32_t par = (uint32_t)(it / NST) & 1u;
  const uint8_t* wb = ring + warp * 2048 + lane * 16;
  const uint8_t* sbp = ring + 16384 + warp * 256 + lane * 4;
  const uint8_t* xbp = ring + SW + 2 * (g * XST + 16 * t);
  for (int s = s0; s < s1; ++s, ++it) {
    if constexpr (!RINGV) { slot = it % NST; par = (uint32_t)(it / NST) & 1u; }
    mbar_wait(full_a + 8 * slot, par, P.err);
    if constexpr ((V41_MASK & 4) == 0) {
      const uint4* wp;
      const uint32_t* sp;
      const __nv_bfloat16* xrow;
      if constexpr (BASEV) {
        const int so = slot * STG;
        wp = reinterpret_cast<const uint4*>(wb + so);
        sp = reinterpret_cast<const uint32_t*>(sbp + so);
        xrow = reinterpret_cast<const __nv_bfloat16*>(xbp + so);
      } else {
        const uint8_t* st = ring + slot * STG;
        wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
        sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
        xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST + 16 * t;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 w = wp[i * 32];
        const uint32_t sw2 = sp[(i >> 1) * 32];
        const int par2 = i & 1;
        const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par2) * 16u + 0x75807580u;
        const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par2) * 16u + 0x75807580u;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
#if K36_HALF_FENCE
          if (hf) asm volatile("" ::: "memory");
#endif
          uint32_t a[4], b[4];
          if constexpr ((V41_MASK & 2) != 0) {
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) { a[jj] = (hf ? w.y : w.x) & 0x3F3F3F3Fu; b[jj] = (hf ? w.w : w.z) & 0x3F3F3F3Fu; }
          } else {
            deq4(hf ? w.y : w.x, a);
            deq4(hf ? w.w : w.z, b);
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) { a[jj] = bmul2(a[jj], Sg); b[jj] = bmul2(b[jj], S8); }
          }
          if constexpr ((V41_MASK & 1) != 0) {
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) sink ^= a[jj] ^ b[jj];
          } else {
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
              const uint4 xv = *reinterpret_cast<const uint4*>(xrow + q * 8 * XST + i * 64 + 8 * hf);
              mma16816(acc[q], a[0], b[0], a[1], b[1], xv.x, xv.y);
              mma16816(acc[q], a[2], b[2], a[3], b[3], xv.z, xv.w);
            }
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    if constexpr (RINGV) { if (++slot == NST) { slot = 0; par ^= 1u; } }
  }
  it_ref = it;
  if (sink == 0x9E3779B9u) atomicAdd(P.sink, 1u);
  constexpr bool NOSWI = (MODE == 1 && NQ >= 3);
  const bool whole = s0 == 0 && s1 == 16;
  if (!MB && NOSWI && whole && !ho) { atomicExch(P.err, 1); atomicExch(P.err + 3, 1); }   // barrier-id budget exhausted
  flush_item<NQ, MODE, NOSWI>(P, acc, whole, e, rb, c0, ntok, npart, MB ? false : (NOSWI ? (ho || !whole) : !whole));
}

"""

# vid 0 (v3.3 gemm, HM 0): the NC 1 item goes to v41_cold_item (whole item: stage loop + flush), before the HOTCAP branch
V0_NEW = ("    if constexpr (!HOTCAP && NC == 1) {\n"
          "      // k2a_v41: v3.3's NC1 cold item = cold_item's code (bitwise); v41 knobs (V41_VAR / V41_MASK)\n"
          "      v41_cold_item<NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart);\n"
          "      if (!(s0 == 0 && s1 == 16)) ++npart;\n"
          "      x += s1 - s0;\n"
          "      continue;\n"
          "    }\n" + A_V0)
V3_NEW = A_V3.replace("cold_item<", "v41_cold_item<").replace("hc_item<", "v41_hc_item<")


# Lab-only producer masks (timing only, results invalid; never exported): V41_MASK bit 8 = K1 X copies off (vid 0 and
# vid 3 producers: expect_tx without the X bytes, consumers read stale X), bit 16 = K2's producer never takes
# griddepcontrol.wait (its X copies read stale interS: the K1 -> K2 hand-off ceiling).
LAB_PATCHES = [
    ("lab pmask", "namespace k3 {\n", "namespace k3 {\nconstexpr int V41_PMASK = @V41MASK@;   // k2a_v41 lab producer masks\n"),
    ("v0 xoff", "      const int ntx = (pmode == 5) ? 0 : ntok;\n",
     "      const int ntx = (pmode == 5 || (V41_PMASK & 8) != 0) ? 0 : ntok;\n"),
    ("v0 k2 nowait", "    int it = 0, e = 0;\n    bool waited = !pdl",
     "    int it = 0, e = 0;\n    bool waited = !pdl || (V41_PMASK & 16) != 0"),
    ("v3 xoff bytes", "          const uint32_t xbytes = (MODE == 1) ? ntok * 512 : ntok * XROW;\n",
     "          const uint32_t xbytes = (MODE == 1) ? (((V41_PMASK & 8) != 0) ? 0 : ntok * 512) : ntok * XROW;\n"),
    ("v3 xoff copy", "              if (r < ntok) bulk_g2s(dst + SW + r * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);\n",
     "              if ((V41_PMASK & 8) == 0 && r < ntok) bulk_g2s(dst + SW + r * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);\n"),
    ("v3 k2 nowait", "copies X row r of the pass (ntok <= 32).\n    int it = 0;\n    bool waited = !pdl;",
     "copies X row r of the pass (ntok <= 32).\n    int it = 0;\n    bool waited = !pdl || (V41_PMASK & 16) != 0;"),
]


def patches(var="@V41VAR@", mask="@V41MASK@"):
    """[(what, old, new)]: each old occurs once in KERNEL_SRC_V3 and once in k3_v21.py."""
    funcs = FUNCS.replace("@V41VAR@", str(var)).replace("@V41MASK@", str(mask))
    return [("v41 functions", A_FUNCS, funcs + A_FUNCS), ("vid 0 cold item", A_V0, V0_NEW), ("vid 3 items", A_V3, V3_NEW)]


def _shipped():
    tree = ast.parse(K2A_FILE.read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("KERNEL_SRC_V3", "V3_VARIANTS")
                                                for t in node.targets):
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name in ("_wg_specs",):
            exec(compile(ast.Module([node], []), str(K2A_FILE), "exec"), ns)
    return ns["KERNEL_SRC_V3"], ns["_wg_specs"](), ns["V3_VARIANTS"]


def apply(src, pl):
    for what, old, new in pl:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"k2a_v41 patch {what!r}: anchor found {n} times")
        src = src.replace(old, new)
    return src


def _case(i, vid):
    nc, nst, minb, hk, nch = (tuple(VIDS[vid]) + (0,) * 5)[:5]
    if hk == 2:
        return (f"    case {i}: launch36<{nch}, {nst}>(p, (int)G, (int)G3, (int)mask); break;",
                f"    case {i}: return attrs36<{nch}, {nst}>();")
    return (f"    case {i}: launch3<{nc}, {nst}, {minb}, {hk}>(p, (int)G, (int)G3, (int)mask); break;",
            f"    case {i}: return attrs3<{nc}, {nst}, {minb}, {hk}>();")


def lab_source(copies=None):
    """(cuda, cpp, names): one namespace copy per COPY, each with vids 0 / 3 as run3 cases 0 / 1."""
    copies = COPIES if copies is None else copies
    src, wg, variants = _shipped()
    for vid in SHIPPED_VIDS:
        v = VIDS[vid]
        if tuple(variants[vid]) != v:
            raise RuntimeError(f"k2a_v41: V3_VARIANTS[{vid}] = {variants[vid]}, lab expects {v}")
    src = apply(apply(src, patches()), LAB_PATCHES).replace("@WGSPECS@", wg)
    cs = [_case(i, v) for i, v in enumerate(VIDS)]
    src = src.replace("@CASES3@", "\n".join(c[0] for c in cs)).replace("@ATTRS3@", "\n".join(c[1] for c in cs))
    k = src.index("namespace k3 {")
    head, body = src[:k], src[k:]
    parts, names = [head], list(copies)
    for name in names:
        var, mask = copies[name]
        parts.append(f"\nnamespace v41_{name} {{\n{body.replace('@V41VAR@', str(var)).replace('@V41MASK@', str(mask))}\n"
                     f"}}  // namespace v41_{name}\n")
    args = RUN_ARGS
    run_cases = "\n".join(f"    case {i}: v41_{n}::run3({RUN_NAMES}); break;" for i, n in enumerate(names))
    attr_cases = "\n".join(f"    case {i}: return v41_{n}::attrs3v(vid);" for i, n in enumerate(names))
    parts.append(f"\nvoid v41run(int64_t copy, {args}) {{\n  switch (copy) {{\n{run_cases}\n"
                 "    default: TORCH_CHECK(false, \"k2a_v41: copy not built\");\n  }\n}\n"
                 f"std::vector<int64_t> v41attrs(int64_t copy, int64_t vid) {{\n  switch (copy) {{\n{attr_cases}\n"
                 "    default: return std::vector<int64_t>{-1};\n  }\n}\n")
    cpp = ("#include <torch/extension.h>\n#include <vector>\n"
           f"void v41run(int64_t copy, {args});\nstd::vector<int64_t> v41attrs(int64_t copy, int64_t vid);\n")
    return "".join(parts), cpp, names


RUN_ARGS = ("int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, "
            "torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, "
            "torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err, "
            "torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw, "
            "torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe")
RUN_NAMES = ("vid, mask, x, ids, wts, ws13, ws2, gsc13, gsc2, nw, h, inter, y, meta, cnt, err, out, sink, ss, cnt2, "
             "xw, xw2, hot_t, M, lo, G, G3, probe")
CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def build(copies=None, name="k2a_v41_lab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda, cpp, names = lab_source(copies)
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    ext = f"{name}_{tag}"
    mod = load_inline(name=ext, cpp_sources=cpp, cuda_sources=cuda, functions=["v41run", "v41attrs"],
                      extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0, names, ext


if __name__ == "__main__":   # write the generated CUDA (for an offline nvcc -cubin + SASS count)
    cu, cp, names = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "k2a_v41_lab_gen.cu")
    out.write_text(cu, encoding="utf-8")
    out.with_suffix(".cpp").write_text(cp, encoding="utf-8")
    print(f"wrote {out} ({len(cu)} chars), copies {names}")

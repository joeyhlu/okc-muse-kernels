"""k1w_switch.py: turn K1-W (tools/at_k1w_patches.py apply_k1w, applied first) into a template flag of K1 v7 so that one
layer file serves both arms of an in-layer A/B: k1_main<L, ST, KW = false> keeps the original L = 1 loader, KW = true the
whole-row loader. Host: launch_main<L, ST, KW>, stages 104 / 106 = (4 / 6) + K1-W (L = 1 only); Python passes +100 when
K1W_ENABLE (read at call time). Every anchor is asserted unique in the whole text.

    text = k1w_switch(apply_k1w(text))
"""
import at_k1w_patches as K

K1_MAIN = "template <int L, int ST>\n__global__ void __launch_bounds__(NTHR, (L == 1) ? 2 : K1_L8_MINB) k1_main(const Params p) {"
PATCHES = (
    ("k1_main template", K1_MAIN, K1_MAIN.replace("template <int L, int ST>", "template <int L, int ST, bool KW = false>")),
    ("pe_load (K1-W)", K.K1W_PATCHES[1][1],
     "      const int absk = kstart + (t0 + i) * TILE + (KW ? wk0 : lk);   // K1-W switch\n"
     "      if (i < ntl && absk < seq) pe.x = __ldg(pt + (absk >> pshift));\n"
     "      if (KW && i < ntl && absk + 8 < seq) pe.y = __ldg(pt + ((absk + 8) >> pshift));\n    } else {\n"),
    ("issue (K1-W)", K.K1W_PATCHES[2][1],
     "      if constexpr (KW) {   // K1-W switch: whole-row map\n" + K.K1W_PATCHES[2][1] + "      } else {\n"
     + K.K1W_PATCHES[2][0] + "      }\n"),
    ("ensure_smem_attr", "template <int L, int ST>\nstatic void ensure_smem_attr() {\n  static bool attr_set = false;\n"
     "  if (!attr_set) {\n    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::k1_main<L, ST>,",
     "template <int L, int ST, bool KW = false>\nstatic void ensure_smem_attr() {\n  static bool attr_set = false;\n"
     "  if (!attr_set) {\n    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::k1_main<L, ST, KW>,"),
    ("launch_main", "template <int L, int ST>\nstatic void launch_main(const k1::Params& p, dim3 grid, cudaStream_t st) {\n"
     "  ensure_smem_attr<L, ST>();\n  k1::k1_main<L, ST><<<",
     "template <int L, int ST, bool KW = false>\nstatic void launch_main(const k1::Params& p, dim3 grid, cudaStream_t st) {\n"
     "  ensure_smem_attr<L, ST, KW>();\n  k1::k1_main<L, ST, KW><<<"),
    ("stages_ok", "static bool stages_ok(int64_t L, int64_t ST) { return (L == 1) ? (ST == 4 || ST == 6) : (ST == 3 || ST == 4); }",
     "static bool stages_ok(int64_t L, int64_t ST) { return (L == 1) ? (ST == 4 || ST == 6 || ST == 104 || ST == 106) "
     ": (ST == 3 || ST == 4); }   // 1xx: K1-W"),
    ("occupancy", "  if (L == 1) return (int64_t)(ST == 4 ? occ_of<1, 4>() : occ_of<1, 6>());",
     "  if (L == 1) return (int64_t)(ST % 100 == 4 ? occ_of<1, 4>() : occ_of<1, 6>());"),
    ("dispatch", "    if (stages == 4) launch_main<1, 4>(p, grid, st);\n    else launch_main<1, 6>(p, grid, st);\n",
     "    if (stages == 4) launch_main<1, 4>(p, grid, st);\n    else if (stages == 104) launch_main<1, 4, true>(p, grid, st);\n"
     "    else if (stages == 106) launch_main<1, 6, true>(p, grid, st);\n    else launch_main<1, 6>(p, grid, st);\n"),
    ("python call", "1 if fold else 0, int(nsg), int(stages))",
     "1 if fold else 0, int(nsg), int(stages) + (100 if (K1W_ENABLE and L == 1) else 0))"),
    ("python global", "def _k1_occupancy(L, st, dev):\n",
     "K1W_ENABLE = True   # K1-W (tools/at_k1w_patches.py): L = 1 whole-row K / V loader (k1_main<1, ST, true>); read at call time\n\n\n"
     "def _k1_occupancy(L, st, dev):\n"),
)


def k1w_switch(text):
    for what, old, new in PATCHES:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"K1-W switch anchor {what!r} found {n} times")
        text = text.replace(old, new, 1)
    if text.count("k1::k1_main<L, ST>") != 1:   # occ_of only (same smem and launch bounds for KW)
        raise ValueError(f"K1-W switch: {text.count('k1::k1_main<L, ST>')} untemplated k1_main references left (want 1)")
    return text

"""epi1_patches.py (study-a): EPI1 = a shorter post-consumer path for K2A v3.6's epilogue warp (vid 3: T=16 / T=32, gemm36).

Why (notes/study/study_a_k2a_variance.md section 5; box_out/sa3, 18 routing x arm points):
  * kernel completion = max(last cold consumer end + 3.5, last hot consumer end + 9.6) us for K1 and (4.8, 12.1) for K2
    (mean |err| 0.4 / 0.7 us): a hot CTA needs ~6-7 us more after its stage loop than a cold one.
  * The epilogue warp (warp 9) finishes every handed-over item (K1 hot items with >= 3 chunks = NOSWI, and split items of
    either kernel) with a token loop of 4 tokens per iteration, each iteration one dependent L2 round trip (__ldcg of h /
    y, then SwiGLU or the sum of squares, then stores). A 32-token hot item = 8 round trips after ~2 us of bar.sync +
    fences + counter atomic; a <= 8-token cold item = 2. That matches E_hot - E_cold.
EPI1 (runtime switch: probe bit 5, set from EPI1_ENABLE at call time; bit 5 is unused by every other K2A path):
  1. the token loop loads up to 8 (K1; 16 spills at the 96-register cap) / 16 (K2) tokens per iteration (4 / 2 round
     trips for a 32-token item instead of 8);
  2. a whole handed-over item (one contributor, the K1 NOSWI hot items) skips the arrival counter (it would go 0 -> 16 -> 0)
     and the second fence; split items keep both.
Numerics: bitwise identical (same per-element operations and stores; only the issue order of loads to distinct addresses
changes; the counter's end state is the same 0). Registers: the epilogue warp holds 8 float2 x 2 (K1) / 16 float4 (K2).

    python -I tools/epi1_patches.py [k3_v35_pt1.py] [out.py]
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ('K2A33_KERNEL_SRC = r"""', '"""')
REGION = ("epilogue warp: split-item counters; the last contributor finishes the item (v3.3)",
          "// ---------------- consumers (warps 0-7: warp w = tile w of the row block) ----------------")

# (what, old, new, expected count inside the gemm36 epilogue region)
REGION_PATCHES = [
    ("epi1 flag",
     "    int npart = 0;\n",
     "    int npart = 0;\n"
     "    const bool epi1 = (P.probe >> 5) & 1;   // EPI1 (study-a): 16-token batches, no counter for whole items\n"
     "    const int epi_nb = epi1 ? ((MODE == 1) ? EPI_NB1 : EPI_NB2) : 4;\n", 1),
    ("whole items skip the counter",
     "          int last = 0;\n"
     "          if (lane == 0) {\n"
     "            int* cp = ((MODE == 1) ? P.cnt : P.cnt2) + (e * NRB + rb) * 64 + j;\n"
     "            const int old = atomicAdd(cp, s1 - s0);\n"
     "            last = (old + (s1 - s0) == 16);\n"
     "            if (last) atomicExch(cp, 0);\n"
     "          }\n"
     "          last = __shfl_sync(FULL, last, 0);\n",
     "          const bool epi_whole = epi1 && s0 == 0 && s1 == 16;   // EPI1: one contributor (counter 0 -> 16 -> 0)\n"
     "          int last = epi_whole ? 1 : 0;\n"
     "          if (!epi_whole) {\n"
     "            if (lane == 0) {\n"
     "              int* cp = ((MODE == 1) ? P.cnt : P.cnt2) + (e * NRB + rb) * 64 + j;\n"
     "              const int old = atomicAdd(cp, s1 - s0);\n"
     "              last = (old + (s1 - s0) == 16);\n"
     "              if (last) atomicExch(cp, 0);\n"
     "            }\n"
     "            last = __shfl_sync(FULL, last, 0);\n"
     "          }\n", 1),
    ("K2 fence",
     "if (MODE == 2 && last) {\n            __threadfence();",
     "if (MODE == 2 && last) {\n            if (!epi_whole) __threadfence();", 1),
    ("K2 batched loop",
     "            for (int n0 = 0; n0 < ntok; n0 += 4) {\n"
     "              float4 yv[4];\n"
     "#pragma unroll\n"
     "              for (int u = 0; u < 4; ++u)\n"
     "                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));\n"
     "#pragma unroll\n"
     "              for (int u = 0; u < 4; ++u) {\n"
     "                float pz = 0.f;\n"
     "                if (n0 + u < ntok) {\n",
     "            for (int n0 = 0; n0 < ntok; n0 += epi_nb) {\n"
     "              float4 yv[EPI_NB2];\n"
     "#pragma unroll\n"
     "              for (int u = 0; u < EPI_NB2; ++u)\n"
     "                if (u < epi_nb && n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));\n"
     "#pragma unroll\n"
     "              for (int u = 0; u < EPI_NB2; ++u) {\n"
     "                if (u >= epi_nb) break;   // uniform\n"
     "                float pz = 0.f;\n"
     "                if (n0 + u < ntok) {\n", 1),
    ("K1 fence",
     "if (MODE == 1 && last) {\n            __threadfence();",
     "if (MODE == 1 && last) {\n            if (!epi_whole) __threadfence();", 1),
    ("K1 batched loop",
     "            for (int n0 = 0; n0 < ntok; n0 += 4) {\n"
     "              float2 gv[4], uv[4];\n"
     "#pragma unroll\n"
     "              for (int u = 0; u < 4; ++u) {\n"
     "                if (n0 + u < ntok) {\n",
     "            for (int n0 = 0; n0 < ntok; n0 += epi_nb) {\n"
     "              float2 gv[EPI_NB1], uv[EPI_NB1];\n"
     "#pragma unroll\n"
     "              for (int u = 0; u < EPI_NB1; ++u) {\n"
     "                if (u < epi_nb && n0 + u < ntok) {\n", 1),
    ("K1 batched compute",
     "#pragma unroll\n"
     "              for (int u = 0; u < 4; ++u) {\n"
     "                if (n0 + u < ntok) {\n"
     "                  const int c = c0 + n0 + u;",
     "#pragma unroll\n"
     "              for (int u = 0; u < EPI_NB1; ++u) {\n"
     "                if (u < epi_nb && n0 + u < ntok) {\n"
     "                  const int c = c0 + n0 + u;", 1),
]
SRC_TOP = ("#ifndef K36_LAUNDER\n",
           "#ifndef EPI_NB1\n#define EPI_NB1 8          // EPI1 (study-a): K1 tokens per epilogue-warp batch (probe bit 5; 16 spills K1)\n#endif\n"
           "#ifndef EPI_NB2\n#define EPI_NB2 16         // EPI1 (study-a): K2 tokens per epilogue-warp batch (probe bit 5)\n#endif\n"
           "#ifndef K36_LAUNDER\n")
PY_CALL_OLD = "probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG[\"k3trig\"] else 0))"
PY_CALL_NEW = ("probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG[\"k3trig\"] else 0)"
               " | ((1 << 5) if EPI1_ENABLE else 0))")
PY_GLOBAL_ANCHOR = "def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):\n"
PY_GLOBAL = ("EPI1_ENABLE = True   # EPI1 (tools/epi1_patches.py): vid 3 epilogue warp, 8 / 16-token batches + no counter for "
             "whole items (probe bit 5, read at call time)\n"
             "KW3_W4 = {}   # KW3 (R-study-a-3): bucket rep -> vid 3 role-split hot_w4 override, read at call time ({} = the plan)\n\n")
# KW3: the role-split weight of a v3.6 bucket, overridable at call time (st["r36"] is frozen at prepare)
PY_R36_OLD = "            return _k2a36_call(st, data, r36[0], r36[1])\n"
PY_R36_NEW = ("            hot36 = r36[1] if not KW3_W4.get(rep) else (r36[1][0], r36[1][1], int(KW3_W4[rep]))   # KW3\n"
              "            return _k2a36_call(st, data, r36[0], hot36)\n")


def apply(text):
    a = text.index(SRC[0])
    b = text.index(SRC[1], a + len(SRC[0]))
    src = text[a:b]
    s = src.index(REGION[0])
    e = src.index(REGION[1], s)
    reg = src[s:e]
    for what, old, new, n in REGION_PATCHES:
        if reg.count(old) < 1:
            raise RuntimeError(f"epi1 {what}: anchor not found")
        if what == "K2 batched loop" or what == "K1 batched loop":
            reg = reg.replace(old, new, 1)   # the region holds one loop per mode; each anchor's text is mode-specific
        else:
            if reg.count(old) != n:
                raise RuntimeError(f"epi1 {what}: anchor found {reg.count(old)} times")
            reg = reg.replace(old, new)
    src = src[:s] + reg + src[e:]
    if src.count(SRC_TOP[0]) != 1:
        raise RuntimeError("epi1 top define anchor")
    src = src.replace(SRC_TOP[0], SRC_TOP[1])
    text = text[:a] + src + text[b:]
    if text.count(PY_CALL_OLD) != 2:
        raise RuntimeError(f"epi1 call sites: {text.count(PY_CALL_OLD)}")
    text = text.replace(PY_CALL_OLD, PY_CALL_NEW)
    if text.count(PY_GLOBAL_ANCHOR) != 1:
        raise RuntimeError("epi1 global anchor")
    text = text.replace(PY_GLOBAL_ANCHOR, PY_GLOBAL + PY_GLOBAL_ANCHOR)
    if text.count(PY_R36_OLD) != 1:
        raise RuntimeError(f"kw3 r36 call: {text.count(PY_R36_OLD)}")
    text = text.replace(PY_R36_OLD, PY_R36_NEW)
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_decode_layer/k3_v35_pt1.py"
    out = apply(src.read_text(encoding="utf-8"))
    if len(sys.argv) > 2:
        pathlib.Path(sys.argv[2]).write_text(out, encoding="utf-8", newline="\n")
    print(f"epi1: patched {src.name} ({len(out)} chars)")

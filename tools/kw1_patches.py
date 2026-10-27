"""kw1_patches.py: KW1 = a K2-only hot weight for K2A v3.7's role split at T=128 (vid 5), with the hot role clamped to at
least half the grid (notes/study/k2a_consumer_audit.md lever 2). Scoped text patches against kernels/muse_decode_layer/k3_v34.py.

Why (per-CTA marks, box_out/fb4 + box_out/r1, 12 routings, 3 router seeds):
  * vid 5's K2 is a static role split. The first Gh CTAs take the hot-expert stages, the rest the cold ones. Gh comes
    from the cost model "hot pass = 16 + hot_w4 x (nch - 1) sixteenths of a stage", with hot_w4 = 12 shared with K1.
  * The fitted K2 cost of an extra 8-token chunk-stage is beta = 0.32-0.43 of a cold stage (median 0.36), while the model
    uses 12/16 = 0.75. So Gh = 149-183 is too large: hot-role CTAs finish at work p50 39-50 us, cold-role ones at 52-70 us
    (max 67-87).
  * The model reproduces the observed Gh exactly. Re-weighting K2 to w4 = 6 with Gh >= G/2 (one hot CTA on every SM, so no
    cold+cold SM pairs) predicts K2 -10.6 us median (-7.4..-14.5) over the 12 routings. K1 keeps hot_w4 and its 8/64 ticket
    pool.
Encoding: probe bits 26-30 = K2 hot weight (0 = off = shipped behaviour). The Python side sets them from KW1_K2_W4, read at
call time (A/B arms toggle it). Only vid 5's K2 (gemm37, MODE 2) reads them. Numerics: same items, different CTA ranges, so
only the FP32-red order of split items changes; it is covered by the existing v3.7 prepare self-test and the judge checks.

    python -I tools/kw1_patches.py [k3_v34.py] [out.py]
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
K2A33 = ('K2A33_KERNEL_SRC = r"""', '"""')

SRC_PATCHES = [
    ("segs37 signature",
     "__device__ __forceinline__ void segs37(const Params& P, const Route& S, int f, int* sg, int* sd) {",
     "__device__ __forceinline__ void segs37(const Params& P, const Route& S, int f, int* sg, int* sd, int w4 = -1) {\n"
     "  const bool kw1 = w4 > 0;            // KW1: K2-only hot weight, hot role >= half the grid\n"
     "  if (w4 <= 0) w4 = P.hot_w4;"),
    ("segs37 weight",
     "          wh = (long long)full * (16 + P.hot_w4 * (NCH - 1)) + (rem ? 16 + P.hot_w4 * (((rem + 7) >> 3) - 1) : 0);",
     "          wh = (long long)full * (16 + w4 * (NCH - 1)) + (rem ? 16 + w4 * (((rem + 7) >> 3) - 1) : 0);"),
    ("segs37 clamp",
     "      Gh = max(1, min(Gh, Sc > 0 ? G - 1 : G));\n    }\n    const bool hr = c < Gh;",
     "      Gh = max(1, min(Gh, Sc > 0 ? G - 1 : G));\n"
     "      if (kw1) Gh = min(max(Gh, G / 2), Sc > 0 ? G - 1 : G);   // KW1: keep a hot CTA on every SM\n"
     "    }\n    const bool hr = c < Gh;"),
    ("gemm37 call",
     "  if (warp == 0) segs37<NCH>(P, S, (MODE == 1) ? ((P.probe >> 8) & 63) : ((P.probe >> 14) & 63), pr, sd);",
     "  if (warp == 0) segs37<NCH>(P, S, (MODE == 1) ? ((P.probe >> 8) & 63) : ((P.probe >> 14) & 63), pr, sd,\n"
     "                             (MODE == 2) ? ((P.probe >> 26) & 31) : 0);   // KW1: K2 hot weight (0 = hot_w4)"),
]
PY_OLD = "    pb = probe | (_k2a37_dyn_code(dyn) if v[3] == 3 else 0)\n"
PY_NEW = ("    pb = probe | (_k2a37_dyn_code(dyn) if v[3] == 3 else 0)\n"
          "    if v[3] == 3 and 0 < KW1_K2_W4 < 32:   # KW1: K2-only hot weight (probe bits 26-30), read at call time\n"
          "        pb |= KW1_K2_W4 << 26\n")
PY_GLOBAL_ANCHOR = "def _k2a37_call(st, data, vid, hot=None, dyn=None, probe=K2A33_HINT, mask=7):\n"
PY_GLOBAL = ("KW1_K2_W4 = 6   # KW1 (tools/kw1_patches.py): K2-only hot weight for vid 5's role split (0 = shipped hot_w4)\n\n")


def apply(text):
    a = text.index(K2A33[0])
    b = text.index(K2A33[1], a + len(K2A33[0]))
    src = text[a:b]
    for what, old, new in SRC_PATCHES:
        if src.count(old) != 1:
            raise RuntimeError(f"kw1 {what}: anchor found {src.count(old)} times")
        src = src.replace(old, new)
    text = text[:a] + src + text[b:]
    for what, old, new in (("call", PY_OLD, PY_NEW), ("global", PY_GLOBAL_ANCHOR, PY_GLOBAL + PY_GLOBAL_ANCHOR)):
        if text.count(old) != 1:
            raise RuntimeError(f"kw1 {what}: anchor found {text.count(old)} times")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_decode_layer/k3_v34.py"
    out = apply(src.read_text(encoding="utf-8"))
    if len(sys.argv) > 2:
        pathlib.Path(sys.argv[2]).write_text(out, encoding="utf-8", newline="\n")
    print(f"kw1: patched {src.name} ({len(out)} chars)")

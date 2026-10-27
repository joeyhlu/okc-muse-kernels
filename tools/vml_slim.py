"""vml_slim.py: remove K2A v3.7 (HK3, vids 4 / 5: the bucket-128 K2A route that VML1 replaces) from a generated layer, to
make room for VML1 under the 1 MB limit. Bucket 128's K2A fallback becomes the v3.6 / v3.3 plan (vid 2), as when
K2A37_ENABLE was False.

  * K2A33 CUDA: find_e .. k2_kernel37 (tail_first37, chunk37, segs37, setup_bars37, gemm37, k1/k2_kernel37) deleted;
    mbar_wait_ok and NE37 stay (PX uses mbar_wait_ok);
  * host: smem_bytes37 / launch37 / attrs37 replaced by stubs (launch37 raises; attrs37 returns a fixed, valid-looking
    row so the variant table / grid code is unchanged; vids 4 / 5 are never launched: r37 stays empty);
  * Python: K2A37_ENABLE = False, _v17_k2a37_prepare keeps only its header (empty r37), _k2a37_call raises.
Every anchor is asserted unique.

    text = apply(text)
"""
CU_A = "__device__ __forceinline__ int find_e(const int* sb, int x) {\n"
CU_B = ("template <int NC, int NST, int MINB, bool HOTCAP, bool SK1 = false>\n"
        "__global__ void __launch_bounds__(320, MINB) k1_kernel(")
H_A = "template <int NCH, int NST>\nstatic constexpr int smem_bytes37() {"
H_END = "  one((const void*)k3::k2_kernel37<NCH, NST>, 320);\n  return r;\n}\n"
H_NEW = ("template <int NCH, int NST>\nstatic void launch37(const k3::Params& p, int G, int G3, int mask) {\n"
         "  TORCH_CHECK(false, \"K2A v3.7 removed (VML1 serves bucket 128)\");\n}\n\n"
         "template <int NCH, int NST>\nstatic std::vector<int64_t> attrs37() { return {0, 0, 0, 2, 0, 0, 0, 2}; }\n")
P_ENABLE = "\nK2A37_ENABLE = True\n"
P_PREP = "def _v17_k2a37_prepare(st, w, dev):\n"
P_PREP_KEEP = "    if not K2A37_ENABLE:\n        return\n"
P_CALL = "def _k2a37_call(st, data, vid, hot=None, dyn=None, probe=K2A33_HINT, mask=7):\n"


def _fn_end(text, i):
    """end of the top-level def starting at i (next line that starts with a non-blank, non-space character)."""
    j = text.index("\n", i) + 1
    while j < len(text):
        nl = text.find("\n", j)
        line = text[j:nl if nl >= 0 else len(text)]
        if line and not line[0].isspace():
            return j
        j = (nl + 1) if nl >= 0 else len(text)
    return len(text)


def apply(text):
    for s in (CU_A, CU_B, H_A, H_END, P_ENABLE, P_PREP, P_PREP_KEEP, P_CALL):
        assert text.count(s) == 1, s[:70]
    a, b = text.index(CU_A), text.index(CU_B)
    assert a < b
    text = text[:a] + text[b:]
    a = text.index(H_A)
    b = text.index(H_END, a) + len(H_END)
    text = text[:a] + H_NEW + text[b:]
    text = text.replace(P_ENABLE, "\nK2A37_ENABLE = False   # VML1 serves bucket 128 (tools/vml_slim.py)\n")
    p = text.index(P_PREP)
    e = _fn_end(text, p)
    k = text.index(P_PREP_KEEP, p)
    assert k < e
    text = text[:k + len(P_PREP_KEEP)] + "\n\n" + text[e:]
    c = text.index(P_CALL)
    e = _fn_end(text, c)
    text = text[:c] + P_CALL + "    raise RuntimeError(\"K2A v3.7 removed (VML1 serves bucket 128)\")\n\n\n" + text[e:]
    for sym in ("gemm37", "tail_first37", "setup_bars37", "k1_kernel37", "k2_kernel37"):
        assert sym not in text, sym
    return text

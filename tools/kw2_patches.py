"""kw2_patches.py: KW2 = K2-only role weights for the K2A role splits that KW1 (vid 5, T=128) did not cover
(notes/study/k2a_consumer_audit.md section 9). Scoped text patches against a layer that already carries su1b + KW1
(tools/gen_k3_v35.py output); every `old` must occur exactly once in its scope.

  * vid 2 (HOTCAP, T=256, WSK): K2's pick_role takes the fine role weight 50 + kw (kw/32 per hot byte) when probe bits 26-30
    hold kw > 0, else the shared hot_w4 (shipped 76 = 26/32). v40 run10/11 (shipped WSK, 8 routings): K2 cold-role CTAs end
    8-29 us after hot-role ones while K1 is balanced -> fewer hot CTAs in K2 only. Arms 20/32 and 17/32 (role-level estimate of
    balance ~19/32).
  * vid 3 (HK2, T=16 / 32): K2's ranges36 role split takes w4 = kw when probe bits 26-30 hold kw > 0, else hot_w4 (shipped 12
    at T=32). fb4 + R1 model: K2-only w4 8 at T=32 -> -2.2 us median (-1.9..-2.4).
  * Python: KW2_K2_ROLE_W {bucket: kw} (vid 2) and KW2_K2_W4_36 {bucket: kw} (vid 3), read at call time; 0 = shipped. The bits
    are set by _k2a33_call / _k2a36_call only (vid 5's _k2a37_call keeps KW1's own bits 26-30: one call = one vid).
Numerics: only which CTA processes which stages changes (split-item FP32-red order), as with KW1.

    python -I tools/kw2_patches.py in.py out.py
"""
import pathlib
import py_compile
import sys

K2A33 = ('K2A33_KERNEL_SRC = r"""', '"""')

SRC = [
    ("HOTCAP K2 role weight", ("k2_kernel(const Params P) {", "__global__"),
     "    const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);",
     "    const int kw2 = (P.probe >> 26) & 31;   // KW2: K2-only fine role weight (32nds), 0 = shared hot_w4\n"
     "    const bool hr = pick_role<NC>(S, kw2 ? 50 + kw2 : P.hot_w4, P.hot_t, cta, ncta);"),
    ("ranges36 signature", None,
     "__device__ __forceinline__ void ranges36(const Params& P, const Route& S, int& b0, int& e0, int& k0, int& b1, int& e1,\n"
     "                                         int& k1) {\n",
     "__device__ __forceinline__ void ranges36(const Params& P, const Route& S, int& b0, int& e0, int& k0, int& b1, int& e1,\n"
     "                                         int& k1, int w4 = -1) {\n"
     "  if (w4 <= 0) w4 = P.hot_w4;   // KW2: K2-only w4 (vid 3), else the shared hot_w4\n"),
    ("ranges36 weight", ("void ranges36(", "void segs36("),
     "        Wh += (long long)full * (16 + P.hot_w4 * (NCH - 1)) + (rem ? 16 + P.hot_w4 * (((rem + 7) >> 3) - 1) : 0);",
     "        Wh += (long long)full * (16 + w4 * (NCH - 1)) + (rem ? 16 + w4 * (((rem + 7) >> 3) - 1) : 0);"),
    ("segs36 signature", None,
     "__device__ __forceinline__ void segs36(const Params& P, const Route& S, int* sg) {\n"
     "  int b[2], e[2], k[2];\n"
     "  ranges36<NCH>(P, S, b[0], e[0], k[0], b[1], e[1], k[1]);\n",
     "__device__ __forceinline__ void segs36(const Params& P, const Route& S, int* sg, int w4 = -1) {\n"
     "  int b[2], e[2], k[2];\n"
     "  ranges36<NCH>(P, S, b[0], e[0], k[0], b[1], e[1], k[1], w4);\n"),
    ("k2_kernel36 call", ("k2_kernel36(const Params P) {", "__global__"),
     "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);",
     "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr, (P.probe >> 26) & 31);   // KW2: K2-only w4 (0 = hot_w4)"),
]
PY_HELPER_ANCHOR = "def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):\n"
PY_HELPER = '''KW2_K2_ROLE_W = {256: 0}   # KW2 (tools/kw2_patches.py): bucket -> vid 2 (HOTCAP) K2-only fine role weight in 32nds (0 = shared hot_w4; read at call time)
KW2_K2_W4_36 = {32: 0}     # KW2: bucket -> vid 3 (HK2) K2-only w4 (0 = shared hot_w4; read at call time)


def _kw2_bits(variant, m):
    """probe bits 26-30 for this call's K2 role split (vid 2 / vid 3 only; vid 5 = KW1 in _k2a37_call)."""
    hk = (tuple(variant) + (0, 0))[3]
    rep = _k2a_bucket_of(m)
    w = KW2_K2_ROLE_W.get(rep, 0) if hk == 1 else KW2_K2_W4_36.get(rep, 0) if hk == 2 else 0
    return (w << 26) if 0 < w < 32 else 0


'''
PROBE_OLD = 'probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0))'
PY = [
    ("K2A33 call bits", ("def _k2a33_call(", "\ndef "), PROBE_OLD,
     'probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0) | _kw2_bits(v, m))'),
    ("K2A36 call bits", ("def _k2a36_call(", "\ndef "), PROBE_OLD,
     'probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0) | _kw2_bits(st["variants"][vid], m))'),
]


def _region(text, scope, a0=0):
    if scope is None:
        return a0, len(text)
    a = text.index(scope[0], a0)
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def _sub(text, scope, old, new, what, a0=0, b0=None):
    sub = text[a0:(len(text) if b0 is None else b0)]
    a, b = _region(sub, scope)
    n = sub[a:b].count(old)
    if n != 1:
        raise RuntimeError(f"kw2 {what}: anchor found {n} times in its scope")
    i = a0 + sub.index(old, a)
    return text[:i] + new + text[i + len(old):]


def apply(text):
    if "KW1_K2_W4" not in text or "_su1_g3" not in text or "KW2_" in text:
        raise RuntimeError("kw2: needs a layer with su1b + KW1 and without KW2")
    a = text.index(K2A33[0])
    b = text.index(K2A33[1], a + len(K2A33[0]))
    src = text[a:b]
    for what, scope, old, new in SRC:
        src = _sub(src, scope, old, new, what)
    text = text[:a] + src + text[b:]
    if text.count(PY_HELPER_ANCHOR) != 1:
        raise RuntimeError("kw2: helper anchor")
    text = text.replace(PY_HELPER_ANCHOR, PY_HELPER + PY_HELPER_ANCHOR)
    for what, scope, old, new in PY:
        text = _sub(text, scope, old, new, what)
    return text


if __name__ == "__main__":
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = apply(src.read_text(encoding="utf-8"))
    out.write_text(t, encoding="utf-8", newline="\n")
    py_compile.compile(str(out), doraise=True)
    print(f"{out}: {len(t.encode())} B, all {len(SRC) + len(PY) + 1} patches hit once, compiles")

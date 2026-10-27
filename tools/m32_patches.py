"""m32_patches.py: the MoE bs16/32 designer's bucket-32 levers on the k3_gd_1 text (tools/gen_k3_m32.py writes the files).

M32 WSK rider (R-m32-2: wk8 -2.92 us median on 10 decode_layer routings at bs32, bs16 untouched):
  * K2A33 ranges36 (HK2, vid 3): with probe bit 27 AND RS1 (bit 7, segs36 on warp 0) the hot role's Stream-K cut is at
    equal cost-model weight (stage weight 16 + hot_w4 (nch - 1), ranges36's own Gh model, averaged over passes), computed by
    warp 0 in parallel (lane = expert, 32-bit inclusive scan + ballot). Bit off = the shipped my_range2 cut, byte-identical
    behaviour. Only the cut points move: same items, same math.
  * _k2a36_call: at bucket 32 (17-32 tokens, HK2 variant, RS1 on, M32_WSK True) hot_w4 8 instead of 12 and bit 27. M32_WSK is
    read at call time (A/B arms); the existing K2A36 prepare self-test (K2A36_SELF_TEST_CASES incl. bs 20 / 24 / 32) runs it.

M32 PX (px=True): see PX_* below.

    python -I tools/m32_patches.py in.py out.py [--px]
"""
import pathlib
import py_compile
import sys

K33_BEGIN = 'K2A33_KERNEL_SRC = r"""'

WSK_HELPER = r"""// M32 (R-m32-2): WSK36's weighted hot cut computed by a full warp (lane = expert). Stage weight of expert e: ws_e / np_e
// with ws_e = sum over its passes of 16 + hot_w4 (nch - 1); cost_e = span_e / np_e * ws_e. CTA c of G takes
// [pos(W c / G), pos(W (c + 1) / G)), pos = the stage at which the cost prefix reaches the target. Needs all 32 lanes.
__device__ __forceinline__ int m32_pos(const Route& S, int T, int W, int lane, int incl, int excl, int ws, int np) {
  if (T >= W) return S.sbh[32];
  const unsigned bal = __ballot_sync(0xffffffffu, incl > T);
  const int e = __ffs(bal) - 1;
  const int p = (lane == e) ? S.sbh[lane] + (int)(((long long)(T - excl) * np) / ws) : 0;
  return __shfl_sync(0xffffffffu, p, e);
}
__device__ __forceinline__ void m32_cut(const Params& P, const Route& S, int c, int G, int tp, int& beg, int& end) {
  const int lane = threadIdx.x & 31;
  const int span = S.sbh[lane + 1] - S.sbh[lane];
  int ws = 16, np = 1, cost = 0;
  if (span > 0) {
    np = S.np[lane];
    const int n = S.cnt[lane];
    ws = 0;
    for (int j = 0; j < np; ++j) { const int nt = min(tp, n - j * tp); ws += 16 + P.hot_w4 * (((nt + 7) >> 3) - 1); }
    cost = span / np * ws;
  }
  int incl = cost;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
  const int W = __shfl_sync(0xffffffffu, incl, 31), excl = incl - cost;
  beg = (c == 0) ? 0 : m32_pos(S, (int)((long long)W * c / G), W, lane, incl, excl, ws, np);
  end = (c + 1 == G) ? S.sbh[32] : m32_pos(S, (int)((long long)W * (c + 1) / G), W, lane, incl, excl, ws, np);
}

"""

# (name, old, new, region): region = None (whole text) or "k33" (inside K2A33_KERNEL_SRC only)
WSK_EDITS = [
    ("wsk helper before ranges36",
     "// Ranges of this CTA (all threads compute the same from S). hot_mode bit 0 = 1: dual range, CTA c takes\n",
     WSK_HELPER + "// Ranges of this CTA (all threads compute the same from S). hot_mode bit 0 = 1: dual range, CTA c takes\n",
     "k33"),
    ("wsk cut in ranges36",
     "  if (hr) my_range2(Sh, c, Gh, b0, e0);\n  else my_range2(Sc, c - Gh, G - Gh, b0, e0);\n",
     "  if (hr) {\n"
     "    if (((P.probe >> 27) & 1) && ((P.probe >> 7) & 1)) m32_cut(P, S, c, Gh, 8 * NCH, b0, e0);   // M32 weighted hot cut\n"
     "    else my_range2(Sh, c, Gh, b0, e0);\n"
     "  } else {\n"
     "    my_range2(Sc, c - Gh, G - Gh, b0, e0);\n"
     "  }\n",
     "k33"),
    ("wsk globals",
     "K2A36_PLAN = {16: (3, (8, 0, 8)), 32: (3, (8, 0, 12))}\n",
     "K2A36_PLAN = {16: (3, (8, 0, 8)), 32: (3, (8, 0, 12))}\n"
     "M32_WSK = True   # M32 rider (R-m32-2): bucket 32 = weighted hot cut (probe bit 27, warp 0) + hot_w4 M32_WSK_W4\n"
     "M32_WSK_W4 = 8\n",
     None),
    ("wsk call",
     'def _k2a36_call(st, data, vid, hot=None, probe=K2A33_HINT, mask=7):\n    x = data["hidden_states"].contiguous()\n',
     'def _k2a36_call(st, data, vid, hot=None, probe=K2A33_HINT, mask=7):\n    x = data["hidden_states"].contiguous()\n'
     '    if M32_WSK and RS1_ENABLE and _k2a_bucket_of(x.shape[0]) == 32 and (tuple(st["variants"][vid]) + (0, 0))[3] == 2:\n'
     '        h0 = K2A36_HOT_DEFAULT if hot is None else hot\n'
     '        hot, probe = (h0[0], h0[1], M32_WSK_W4), probe | (1 << 27)\n',
     None),
]


def _region(text, region):
    if region is None:
        return 0, len(text)
    a = text.index(K33_BEGIN)
    b = text.index('"""', a + len(K33_BEGIN))
    return a, b


def _repl(text, edits, tag):
    for name, old, new, region in edits:
        a, b = _region(text, region)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"m32 {tag} patch {name!r}: anchor found {n}x in {region or 'text'} (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


def apply(text, px=False):
    for name in ("M32_WSK", "m32_cut", "M32_PX"):
        if name in text:
            raise RuntimeError(f"m32 patch: {name!r} already in the base")
    if text.count(K33_BEGIN) != 1:
        raise RuntimeError("m32 patch: K2A33_KERNEL_SRC anchor")
    text = _repl(text, WSK_EDITS, "wsk")
    if px:
        import m32_px
        text = m32_px.apply(text)
    return text


if __name__ == "__main__":
    s, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = s.read_text(encoding="utf-8")
    out = apply(t, px="--px" in sys.argv)
    d.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(d), doraise=True)
    print(f"m32: applied {s.name} -> {d} ({len(out) - len(t):+d} chars, compiles)")

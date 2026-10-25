"""Generate kernels/muse_moe_w4a16/k2a_v38.py from k2a_v37.py (design: kernels/muse_moe_w4a16/k2a_v38_design.md).

v3.8 = v3.7 + boundary switches in the v3.3 (vid 0-2) and v3.6 HK2 (vid 3) kernels, packed in probe bits 8..28
(HK3 vids 4/5 own bits 8..23 for their queue; v38_call never sets v3.8 bits for them). All bits 0 = v3.7 bit for bit
(the new code only runs behind the bits). Every patch is anchored and must match exactly once inside KERNEL_SRC_V3.

  python tools/gen_k2a_v38.py          # writes the file, prints the anchor count
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/muse_moe_w4a16/k2a_v37.py"
DST = ROOT / "kernels/muse_moe_w4a16/k2a_v38.py"

HELPERS_A = r'''__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

// ---- v3.8 switches (probe bits; vids 0-3 only; all 0 = v3.7) ----
// bits 8-10 P: K2 own L2 prefetch depth (stages after the NST smem stages), bits 11-16 H/2: K2 assist CTAs (CTA c < H
// also prefetches the first NST + P stages of CTA G - H + c, a late entrant), bits 17-21 a2 / 22-26 a1: K2 / K1
// blockIdx tilt of the stream-K ranges (/64), bit 27 B3: per-expert global scale from smem, bit 28 B4: K2 routing
// without the scatter pass (K2 never reads s_list).
__device__ __forceinline__ int v38_pf(const Params& P) { return (P.probe >> 8) & 7; }
__device__ __forceinline__ int v38_h(const Params& P) { return 2 * ((P.probe >> 11) & 63); }
__device__ __forceinline__ int v38_a(const Params& P, int mode) { return (mode == 1) ? ((P.probe >> 22) & 31) : ((P.probe >> 17) & 31); }
__device__ __forceinline__ bool v38_b3(const Params& P) { return ((P.probe >> 27) & 1) != 0; }
__device__ __forceinline__ bool v38_b4(const Params& P) { return ((P.probe >> 28) & 1) != 0; }
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void l2_prefetch(const void* src, uint32_t bytes, uint64_t pol) {
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n" :: "l"(src), "r"(bytes), "l"(pol) : "memory");
}'''

HELPERS_B = r'''// v3.8 tilted stream-K boundary B(c) = floor(S c (64 n + a (n - c)) / (64 n^2)): B(0) = 0, B(n) = S, monotone for
// 0 <= a < 64 (step >= S (64 - a) / (64 n)); a = 0 is my_range2 exactly. Low blockIdx (early K2 entrants) get more.
__device__ __forceinline__ int tilt_b(long long S, int c, int n, int a) {
  return (int)((S * (long long)c * (64ll * n + (long long)a * (n - c))) / (64ll * n * n));
}
__device__ __forceinline__ void my_range2t(long long S, int cta, int ncta, int a, int& beg, int& end) {
  if (a == 0) { my_range2(S, cta, ncta, beg, end); return; }
  beg = tilt_b(S, cta, ncta, a);
  end = tilt_b(S, cta + 1, ncta, a);
}'''

HELPERS_C = r'''
// v3.8 B1: walk [xb, xe) of a stage prefix sb (as the producer does), counting stages in k; issue an L2 prefetch of
// the weight stage for skip <= k < lim. Returns k. Fire-and-forget hints: no smem, no barrier, no wait.
__device__ __forceinline__ int pf_seg(const Route& S, const int* sb, const uint8_t* wsrc, int nrb, int xb, int xe,
                                      int k, int skip, int lim, uint64_t pol) {
  int e = 0;
  for (int x = xb; x < xe && k < lim; ++x, ++k) {
    if (k < skip) continue;
    while (sb[e + 1] <= x) ++e;
    const int l = x - sb[e];
    const int rb = (l >> 4) / S.np[e];
    l2_prefetch(wsrc + ((size_t)(e * nrb + rb) * 16 + (l & 15)) * SW, SW, pol);
  }
  return k;
}
// K2 of the v3.3 kernels (vid 0): own stages [NST, NST + P) in processing order, plus the assist (CTA c < H: the first
// NST + P stages of CTA G - H + c). Lane 0 of the producer, just before griddepcontrol.wait.
template <int NST>
__device__ __forceinline__ void v38_k2pf0(const Params& P, const Route& S, const uint8_t* wsrc, int beg, int end,
                                          int cta, int ncta) {
  const int pf = v38_pf(P), H = v38_h(P);
  if (pf == 0 && H == 0) return;
  const uint64_t pol = policy_evict_last();
  if (pf > 0) {
    const int mid = tail_first(S, beg, end);
    const int k = pf_seg(S, S.sb, wsrc, 32, mid, end, 0, NST, NST + pf, pol);
    pf_seg(S, S.sb, wsrc, 32, beg, mid, k, NST, NST + pf, pol);
  }
  if (H > 0 && ncta == (int)gridDim.x && cta < H && ncta - H + cta > cta) {
    int b2, e2;
    my_range2t((long long)S.sb[32], ncta - H + cta, ncta, v38_a(P, 2), b2, e2);
    const int m2 = tail_first(S, b2, e2);
    const int k = pf_seg(S, S.sb, wsrc, 32, m2, e2, 0, 0, NST + pf, pol);
    pf_seg(S, S.sb, wsrc, 32, b2, m2, k, 0, NST + pf, pol);
  }
}
'''

HELPERS_D = r'''template <int NCH>
__device__ __forceinline__ void segs36c(const Params& P, const Route& S, int c, int a, int* sg);
// K2 of the HK2 kernels (vid 3): the same over the CTA's four segments (pr) and, for the assist, the late CTA's.
template <int NCH, int NST>
__device__ __forceinline__ void v38_k2pf36(const Params& P, const Route& S, const uint8_t* wsrc, const int* pr) {
  const int pf = v38_pf(P), H = v38_h(P);
  if (pf == 0 && H == 0) return;
  const uint64_t pol = policy_evict_last();
  if (pf > 0) {
    int k = 0;
#pragma unroll
    for (int sg = 0; sg < 4; ++sg)
      k = pf_seg(S, pr[3 * sg + 2] ? S.sbh : S.sbc, wsrc, 32, pr[3 * sg], pr[3 * sg + 1], k, NST, NST + pf, pol);
  }
  const int c = (int)blockIdx.x, G = (int)gridDim.x;
  if (H > 0 && c < H && G - H + c > c) {
    int sg2[12];
    segs36c<NCH>(P, S, G - H + c, v38_a(P, 2), sg2);
    int k = 0;
#pragma unroll
    for (int sg = 0; sg < 4; ++sg)
      k = pf_seg(S, sg2[3 * sg + 2] ? S.sbh : S.sbc, wsrc, 32, sg2[3 * sg], sg2[3 * sg + 1], k, 0, NST + pf, pol);
  }
}

'''

TRAIL = ("    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain\n"
         "    if (!waited) {\n      pdl_wait();")


def sub1(text, old, new):
    n = text.count(old)
    if n != 1:
        sys.exit(f"anchor matched {n} times: {old[:90]!r}")
    return text.replace(old, new)


def main():
    full = SRC.read_text(encoding="utf-8")
    a = full.index('KERNEL_SRC_V3 = r"""')
    b = full.index('CPP_SRC_V3 = r"""')
    head, k, tail = full[:a], full[a:b], full[b:]
    n = 0

    # helpers
    k = sub1(k, '__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\\n" ::: "memory"); }',
             HELPERS_A); n += 1
    k = sub1(k, "  int wc[8][32];\n  int R;\n};", "  int wc[8][32];\n  int R;\n  float gs[32];   // v3.8 B3: per-expert global scale (gsc13 / gsc2)\n};"); n += 1
    # route: B3 fill (load issued before the first barrier), B4 early return for K2 (nrb 32)
    k = sub1(k, "  const int NP = 8 * P.M;\n  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;",
             "  const int NP = 8 * P.M;\n  float gsv = 0.f;\n"
             "  if (warp == 0 && v38_b3(P)) gsv = __ldg(((nrb == 64) ? P.gsc13 : P.gsc2) + lane);\n"
             "  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;"); n += 1
    k = sub1(k, "    S.cnt[lane] = c; S.off[lane] = off; S.np[lane] = np;\n",
             "    S.cnt[lane] = c; S.off[lane] = off; S.np[lane] = np;\n    if (v38_b3(P)) S.gs[lane] = gsv;\n"); n += 1
    k = sub1(k, "  if (!LIST) return;", "  if (!LIST || (nrb == 32 && v38_b4(P))) return;"); n += 1
    # tilt helpers after my_range2, B1 walker after tail_first
    k = sub1(k, "__device__ __forceinline__ void my_range(long long S, int& beg, int& end) {",
             HELPERS_B + "\n__device__ __forceinline__ void my_range(long long S, int& beg, int& end) {"); n += 1
    k = sub1(k, "  const int m = S.sb[e] + ((end - 1 - S.sb[e]) & ~15);\n  return max(beg, m);\n}\n",
             "  const int m = S.sb[e] + ((end - 1 - S.sb[e]) & ~15);\n  return max(beg, m);\n}\n" + HELPERS_C); n += 1
    # gemm (v3.3 kernels): tilted range, B1 before the deferred wait, B3 in the cold flush
    k = sub1(k, "  my_range2(Stot, cta, ncta, beg, end);", "  my_range2t(Stot, cta, ncta, v38_a(P, MODE), beg, end);"); n += 1
    k = sub1(k, "        if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now\n"
                "          pdl_wait();",
             "        if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now\n"
             "          if (MODE == 2 && lane == 0) v38_k2pf0<NST>(P, S, wsrc, beg, end, cta, ncta);\n"
             "          pdl_wait();"); n += 1
    k = sub1(k, "      const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);\n"
                "      const bool whole = (s0 == 0 && s1 == 16) || fmode == 6;",
             "      const float gs = v38_b3(P) ? S.gs[e] : __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);\n"
             "      const bool whole = (s0 == 0 && s1 == 16) || fmode == 6;"); n += 1
    # trailing waits: occurrence 0 = gemm, 1 = gemm36 (2 = gemm37 unchanged)
    i0 = k.index(TRAIL)
    i1 = k.index(TRAIL, i0 + 1)
    k2 = k[:i1] + TRAIL.replace("      pdl_wait();", "      if (MODE == 2 && lane == 0) v38_k2pf36<NCH, NST>(P, S, wsrc, pr);\n      pdl_wait();") + k[i1 + len(TRAIL):]
    k = k2[:i0] + TRAIL.replace("      pdl_wait();", "      if (MODE == 2 && lane == 0) v38_k2pf0<NST>(P, S, wsrc, beg, end, cta, ncta);\n      pdl_wait();") + k2[i0 + len(TRAIL):]
    n += 2
    # gemm36: walker declaration before it; B1 before its deferred wait
    k = sub1(k, "template <int NCH, int NST, int MODE>\n__device__ void gemm36(",
             HELPERS_D + "template <int NCH, int NST, int MODE>\n__device__ void gemm36("); n += 1
    k = sub1(k, "            if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now\n"
                "              pdl_wait();",
             "            if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now\n"
             "              if (MODE == 2 && lane == 0) v38_k2pf36<NCH, NST>(P, S, wsrc, pr);\n"
             "              pdl_wait();"); n += 1
    # ranges36 -> ranges36c (CTA index and tilt as parameters), segs36 -> segs36c
    k = sub1(k, "__device__ __forceinline__ void ranges36(const Params& P, const Route& S, int& b0, int& e0, int& k0, int& b1, int& e1,\n"
                "                                         int& k1) {\n"
                "  const int G = (int)gridDim.x, c = (int)blockIdx.x;",
             "__device__ __forceinline__ void ranges36(const Params& P, const Route& S, int c, int a, int& b0, int& e0, int& k0,\n"
             "                                         int& b1, int& e1, int& k1) {\n"
             "  const int G = (int)gridDim.x;"); n += 1
    k = sub1(k, "    my_range2(Sh, c, G, hb, he);\n    my_range2(Sc, c, G, cb, ce);\n    const bool cold_first",
             "    my_range2t(Sh, c, G, a, hb, he);\n    my_range2t(Sc, c, G, a, cb, ce);\n    const bool cold_first"); n += 1
    k = sub1(k, "  if (hr) my_range2(Sh, c, Gh, b0, e0);\n  else my_range2(Sc, c - Gh, G - Gh, b0, e0);",
             "  if (hr) my_range2t(Sh, c, Gh, a, b0, e0);\n  else my_range2t(Sc, c - Gh, G - Gh, a, b0, e0);"); n += 1
    k = sub1(k, "__device__ __forceinline__ void segs36(const Params& P, const Route& S, int* sg) {\n"
                "  int b[2], e[2], k[2];\n"
                "  ranges36<NCH>(P, S, b[0], e[0], k[0], b[1], e[1], k[1]);",
             "__device__ __forceinline__ void segs36c(const Params& P, const Route& S, int c, int a, int* sg) {\n"
             "  int b[2], e[2], k[2];\n"
             "  ranges36<NCH>(P, S, c, a, b[0], e[0], k[0], b[1], e[1], k[1]);"); n += 1
    i0 = k.index("  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);")
    i1 = k.index("  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);", i0 + 1)
    old = "  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);"
    k = (k[:i0] + "  if (threadIdx.x == 0) segs36c<NCH>(P, S, (int)blockIdx.x, v38_a(P, 1), s_pr);" + k[i0 + len(old):i1]
         + "  if (threadIdx.x == 0) segs36c<NCH>(P, S, (int)blockIdx.x, v38_a(P, 2), s_pr);" + k[i1 + len(old):])
    n += 2
    if "segs36<" in k or "ranges36<NCH>(P, S, b[0]" in k:
        sys.exit("stale segs36 / ranges36 call left")

    # Python side
    head = sub1(head, "# nonce k2a-v37-1", "# nonce k2a-v38-1")
    head = sub1(head, '"""K2a v3.7 = v3.6 + HK3 variants',
                '"""K2a v3.8 = v3.7 + boundary switches (v38_call; design kernels/muse_moe_w4a16/k2a_v38_design.md;\n'
                'generated by tools/gen_k2a_v38.py; bench tools/box/k38_bench.py). Probe bits 8..28 on vids 0-3: B1 K2 L2\n'
                'prefetch before griddepcontrol.wait (own stages + late-entrant assist), B2 blockIdx tilt of the stream-K\n'
                'ranges, B3 per-expert global scale from smem, B4 K2 routing without the scatter. All 0 = v3.7 bit for bit.\n\n'
                'K2a v3.7 = v3.6 + HK3 variants')
    tail = sub1(tail, 'def v3_prepare(weights, st2=None, variants=V3_VARIANTS, name="k2a_v37", log=print, share=None):',
                'def v3_prepare(weights, st2=None, variants=V3_VARIANTS, name="k2a_v38", log=print, share=None):')
    tail = sub1(tail, "def v3_reset(st):", V38_PY + "\n\ndef v3_reset(st):")
    out = head + k + tail
    DST.write_text(out, encoding="utf-8")
    print(f"[gen_k2a_v38] {n} kernel patches + python; wrote {DST} ({len(out)} B)")


V38_PY = r'''def v38_code(v38):
    """probe bits of the v3.8 switches: v38 = (P, H, a2, a1, b3, b4) (P own prefetch stages <= 7, H assist CTAs (even,
    <= 126), a2 / a1 K2 / K1 tilt in 64ths <= 31, b3 / b4 flags). None or all 0 = v3.7."""
    if v38 is None:
        return 0
    pf, h, a2, a1, b3, b4 = (tuple(v38) + (0,) * 6)[:6]
    assert 0 <= pf <= 7 and 0 <= h <= 126 and h % 2 == 0 and 0 <= a2 <= 31 and 0 <= a1 <= 31, v38
    return (pf << 8) | ((h // 2) << 11) | (a2 << 17) | (a1 << 22) | ((1 << 27) if b3 else 0) | ((1 << 28) if b4 else 0)


def v38_call(st, data, vid, hot=None, dyn=None, v38=None, probe=V3_HINT, mask=7):
    """v37_call plus the v3.8 switches on vids 0-3 (HK3 vids 4/5 keep v3.7: their queue owns probe bits 8..23)."""
    v = tuple(st["variants"][vid]) + (0, 0)
    if v[3] == 3:
        return v37_call(st, data, vid, hot, dyn, probe=probe, mask=mask)
    return v37_call(st, data, vid, hot, None, probe=probe | v38_code(v38), mask=mask)'''


if __name__ == "__main__":
    main()

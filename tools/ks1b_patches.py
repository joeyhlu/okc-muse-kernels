"""ks1b_patches.py: KS1B = a ticket-free merge for the in-layer sliding attention KS1 (T <= 8, B = 1).

KS1 at decode1 / verify1 ends (VSL red, MG 1): red.add of every split into a zeroed workspace -> acq_rel ticket (waits for
the reds' acks, then the atomic's own round trip) -> the last CTA re-reads the summed rows -> finalize (R-lat-0: ticket 1.28 +
last-CTA merge 0.8 us of KS1's 4.48 us). New VSL merge forms (template argument MG, MG 0 / 1 / 2 unchanged):
  MG 3 "ll":  every split stores its CTA-merged partial (unnormalized fp32 o, (max, sum)) with st.relaxed.gpu as
              self-validating words (bits ^ 0xffffffff, so a stored word is never 0; the workspace rests at 0).
              CTA s = NS - r finalizes row r: 16 lanes x 16 dim chunks poll the 17 partial rows with ld.relaxed.gpu until
              every word is non-zero (bounded: 2^20 polls, then counters[1023] = 1, never a trap), merge with shuffles,
              one smem exchange for the row's RMS, gated store, and re-zero the consumed words. No fence, no atomic.
  MG | 8:     the fp8 -> f16 K / V tile conversion runs before griddepcontrol.wait (it only needs the pre-wait loads).
Variants: ("ks1", L, 16, NTG, MG) for MG 3 (ll), 11 (ll + pre-wait conversion), 9 (red + pre-wait conversion); routed by
VSL_ROUTE (read at call time, so one file serves every A/B arm), only after prepare()'s VSL self-test passes them.

    python -I tools/ks1b_patches.py --out=kernels/lab/k3_ks1b_1_ab.py [--base=kernels/muse_decode_layer/k3_gc_16.py]
        [--insts=1:3,1:11,1:9,8:3,8:11,8:9] [--route=1:11,8:11] [--ll=2|1]   (ll 1 = R-ks1b-1's v1 merge)    (route: VSL_ROUTE for q_len 1 / 8; default = base's)
Never overwrites an existing file.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEO = {1: (16, 1), 8: (16, 4)}

HELPERS = r'''__device__ __forceinline__ uint4 ks1b_ld4(const uint32_t* p) {   // KS1B: polled L2 loads / stores (gpu scope, relaxed)
  uint4 v;
  asm volatile("ld.relaxed.gpu.global.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ks1b_ld2(const uint32_t* p) {
  uint2 v;
  asm volatile("ld.relaxed.gpu.global.v2.b32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void ks1b_st4(uint32_t* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("st.relaxed.gpu.global.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void ks1b_st2(uint32_t* p, uint32_t a, uint32_t b) {
  asm volatile("st.relaxed.gpu.global.v2.b32 [%0], {%1,%2};\n" ::"l"(p), "r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ uint32_t ks1b_enc(float x) { return __float_as_uint(x) ^ 0xFFFFFFFFu; }   // never 0 (no 0xffffffff NaN)
__device__ __forceinline__ float ks1b_dec(uint32_t w) { return __uint_as_float(w ^ 0xFFFFFFFFu); }
__device__ __forceinline__ float ks1b_hmax(float x) {   // max over the 16-lane half-warp
  x = fmaxf(x, __shfl_xor_sync(FULL, x, 8)); x = fmaxf(x, __shfl_xor_sync(FULL, x, 4));
  x = fmaxf(x, __shfl_xor_sync(FULL, x, 2)); return fmaxf(x, __shfl_xor_sync(FULL, x, 1));
}
__device__ __forceinline__ float ks1b_hsum(float x) {
  x += __shfl_xor_sync(FULL, x, 8); x += __shfl_xor_sync(FULL, x, 4);
  x += __shfl_xor_sync(FULL, x, 2); return x + __shfl_xor_sync(FULL, x, 1);
}
'''

GATE3 = r'''  uint2 g3 = make_uint2(0u, 0u);                               // KS1B MG 3: gate chunk (row NS - s, dims 4 u3 ..) this CTA finalizes
  if constexpr ((MG & 3) == 3) {
    const int r3 = NS - s, u3 = warp * 2 + (lane >> 4);
    if (r3 >= 0 && r3 < ROWS)
      g3 = __ldcg(reinterpret_cast<const uint2*>(p.qkv + (size_t)(tb + tg * TPG + (r3 >> 3)) * sq + (size_t)(HQ + 2 * HKV) * 64 +
                                                 (size_t)(h * 8 + (r3 & 7)) * 64 + 4 * u3));
  }
'''

LL = r'''    if constexpr ((MG & 3) == 3) {   // KS1B LL merge: self-validating partial words, no fence / ticket; CTA NS - r finalizes row r
      static_assert(NS == 16 && ROWS <= NS + 1, "KS1B: 17 partials per group, one finalizer CTA per row");
      {
        const size_t prow = ((size_t)grp * (NS + 1) + s) * ROWS + mr;
        uint32_t* wo = reinterpret_cast<uint32_t*>(p.part_o) + prow * 64 + DPT * mu;
        if constexpr (DPT == 4) ks1b_st4(wo, ks1b_enc(acc[0]), ks1b_enc(acc[1]), ks1b_enc(acc[DPT - 2]), ks1b_enc(acc[DPT - 1]));
        else ks1b_st2(wo, ks1b_enc(acc[0]), ks1b_enc(acc[1]));
        if (mu == 0) ks1b_st2(reinterpret_cast<uint32_t*>(p.part_ml) + prow * 2, ks1b_enc(M), ks1b_enc(lt));
      }
      KS1P(7);
      const int r = NS - s;
      if (r >= ROWS) return;
      const int q = lane & 15, u = warp * 2 + (lane >> 4);         // source split q (and split NS for every lane), dims 4 u .. 4 u + 3
      const size_t r0 = (size_t)grp * (NS + 1) * ROWS + r;
      uint32_t* pa = reinterpret_cast<uint32_t*>(p.part_o) + (r0 + (size_t)q * ROWS) * 64 + 4 * u;
      uint32_t* pb = reinterpret_cast<uint32_t*>(p.part_o) + (r0 + (size_t)NS * ROWS) * 64 + 4 * u;
      uint32_t* ma = reinterpret_cast<uint32_t*>(p.part_ml) + (r0 + (size_t)q * ROWS) * 2;
      uint32_t* mb = reinterpret_cast<uint32_t*>(p.part_ml) + (r0 + (size_t)NS * ROWS) * 2;
      uint4 xa, xb;
      uint2 ya, yb;
      bool ok = false;
      for (int it = 0; it < (1 << 20) && !ok; ++it) {             // bounded wait (flag on timeout, never a trap)
        xa = ks1b_ld4(pa); xb = ks1b_ld4(pb); ya = ks1b_ld2(ma); yb = ks1b_ld2(mb);
        ok = (xa.x != 0u) & (xa.y != 0u) & (xa.z != 0u) & (xa.w != 0u) & (xb.x != 0u) & (xb.y != 0u) & (xb.z != 0u) &
             (xb.w != 0u) & (ya.x != 0u) & (ya.y != 0u) & (yb.x != 0u) & (yb.y != 0u);
      }
      if (!ok) atomicExch(p.counters + 1023, 1);
      KS1P(8);
      const float mq = ks1b_dec(ya.x), lq = ks1b_dec(ya.y), mn = ks1b_dec(yb.x), ln = ks1b_dec(yb.y);
      const float Mx = fmaxf(ks1b_hmax(mq), mn);
      const float Mu3 = (Mx == neg_inf()) ? 0.f : Mx;
      const float wq = ex2(mq - Mu3), wn = ex2(mn - Mu3);
      const float den = fmaf(wn, ln, ks1b_hsum(wq * lq));
      const float a0 = fmaf(wn, ks1b_dec(xb.x), ks1b_hsum(wq * ks1b_dec(xa.x)));
      const float a1 = fmaf(wn, ks1b_dec(xb.y), ks1b_hsum(wq * ks1b_dec(xa.y)));
      const float a2 = fmaf(wn, ks1b_dec(xb.z), ks1b_hsum(wq * ks1b_dec(xa.z)));
      const float a3 = fmaf(wn, ks1b_dec(xb.w), ks1b_hsum(wq * ks1b_dec(xa.w)));
      const float dd = (den > 0.f) ? den : 1.f;
      const float y0 = bfr(a0 / dd), y1 = bfr(a1 / dd), y2 = bfr(a2 / dd), y3 = bfr(a3 / dd);
      float* ssr = reinterpret_cast<float*>(misc);                 // [16] per-chunk sums of squares of the row
      if (q == 0) ssr[u] = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
      __syncthreads();                                             // every poll has returned: re-zero the consumed words
      ks1b_st4(pa, 0u, 0u, 0u, 0u);
      if (q == 0) ks1b_st4(pb, 0u, 0u, 0u, 0u);
      if (u == 0) ks1b_st2(ma, 0u, 0u);
      if (tid == 0) ks1b_st2(mb, 0u, 0u);
      if (q == 0) {
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += ssr[i];
        const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
        const float2 g01 = bf2f(g3.x), g23 = bf2f(g3.y);
        const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0 * rs * (1.f / (1.f + __expf(-g01.x))), y1 * rs * (1.f / (1.f + __expf(-g01.y))));
        const __nv_bfloat162 o23 = __floats2bfloat162_rn(y2 * rs * (1.f / (1.f + __expf(-g23.x))), y3 * rs * (1.f / (1.f + __expf(-g23.y))));
        *reinterpret_cast<uint2*>(p.out + (size_t)(tb + tg * TPG + (r >> 3)) * (HQ * 64) + (size_t)(h * 8 + (r & 7)) * 64 + 4 * u) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&o01), *reinterpret_cast<const uint32_t*>(&o23));
      }
      KS1P(9);
      return;
    }
'''


LL2 = r'''    if constexpr ((MG & 3) == 3) {   // KS1B LL2: self-validating partial words, no fence / ticket; CTA NS - r finalizes row r.
      // R-ks1b-1: ll v1 polled every (max, sum) line from all 8 warps of every finalizer (64 / 128 warps per line): same-line
      // strong-op serialization at L2 (+3.7 / +5.8 us). LL2: (max, sum) as [grp][row][split] (each finalizer polls its own 136 B
      // with warp 0 only), o polled once per word (lines exclusive to one finalizer), weights shared through smem.
      static_assert(NS == 16 && ROWS <= NS + 1, "KS1B: 17 partials per group, one finalizer CTA per row");
      {
        const size_t prow = ((size_t)grp * (NS + 1) + s) * ROWS + mr;
        uint32_t* wo = reinterpret_cast<uint32_t*>(p.part_o) + prow * 64 + DPT * mu;
        if constexpr (DPT == 4) ks1b_st4(wo, ks1b_enc(acc[0]), ks1b_enc(acc[1]), ks1b_enc(acc[DPT - 2]), ks1b_enc(acc[DPT - 1]));
        else ks1b_st2(wo, ks1b_enc(acc[0]), ks1b_enc(acc[1]));
        if (mu == 0) ks1b_st2(reinterpret_cast<uint32_t*>(p.part_ml) + (((size_t)grp * ROWS + mr) * (NS + 1) + s) * 2, ks1b_enc(M), ks1b_enc(lt));
      }
      KS1P(7);
      const int r = NS - s;
      if (r >= ROWS) return;
      const int q = lane & 15, u = warp * 2 + (lane >> 4);         // source split q (and split NS for every lane), dims 4 u .. 4 u + 3
      const size_t r0 = (size_t)grp * (NS + 1) * ROWS + r;
      uint32_t* pa = reinterpret_cast<uint32_t*>(p.part_o) + (r0 + (size_t)q * ROWS) * 64 + 4 * u;
      uint32_t* pb = reinterpret_cast<uint32_t*>(p.part_o) + (r0 + (size_t)NS * ROWS) * 64 + 4 * u;
      uint32_t* pm = reinterpret_cast<uint32_t*>(p.part_ml) + (((size_t)grp * ROWS + r) * (NS + 1) + min(lane, NS)) * 2;
      const bool mlw = (warp == 0) && (lane <= NS);                 // warp 0, lanes 0 .. 16: (max, sum) of split = lane
      uint4 xa, xb;
      uint2 ym = make_uint2(1u, 1u);
      bool ok = false;
      for (int it = 0; it < (1 << 20) && !ok; ++it) {             // bounded wait (flag on timeout, never a trap)
        xa = ks1b_ld4(pa);
        if (q == 0) xb = ks1b_ld4(pb);
        if (mlw) ym = ks1b_ld2(pm);
        ok = (xa.x != 0u) & (xa.y != 0u) & (xa.z != 0u) & (xa.w != 0u) & (ym.x != 0u) & (ym.y != 0u);
        if (q == 0) ok = ok & (xb.x != 0u) & (xb.y != 0u) & (xb.z != 0u) & (xb.w != 0u);
      }
      if (!ok) atomicExch(p.counters + 1023, 1);
      KS1P(8);
      float* sw = reinterpret_cast<float*>(qs);                    // (q tile, dead after attention) [0, 17): split weights, 17: denominator, [18, 34): row sums of squares
      if (warp == 0) {
        const float mq = mlw ? ks1b_dec(ym.x) : neg_inf(), lq = mlw ? ks1b_dec(ym.y) : 0.f;
        float Mx = mq;
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) Mx = fmaxf(Mx, __shfl_xor_sync(FULL, Mx, d));
        const float Mu3 = (Mx == neg_inf()) ? 0.f : Mx;
        const float w = mlw ? ex2(mq - Mu3) : 0.f;
        const float den = warp_sum(w * lq);
        if (mlw) sw[lane] = w;
        if (lane == 0) sw[NS + 1] = den;
      }
      __syncthreads();
      const float wq = sw[q], wn = sw[NS], den = sw[NS + 1];
      const float on0 = (q == 0) ? wn * ks1b_dec(xb.x) : 0.f, on1 = (q == 0) ? wn * ks1b_dec(xb.y) : 0.f;
      const float on2 = (q == 0) ? wn * ks1b_dec(xb.z) : 0.f, on3 = (q == 0) ? wn * ks1b_dec(xb.w) : 0.f;
      const float a0 = ks1b_hsum(fmaf(wq, ks1b_dec(xa.x), on0)), a1 = ks1b_hsum(fmaf(wq, ks1b_dec(xa.y), on1));
      const float a2 = ks1b_hsum(fmaf(wq, ks1b_dec(xa.z), on2)), a3 = ks1b_hsum(fmaf(wq, ks1b_dec(xa.w), on3));
      const float dd = (den > 0.f) ? den : 1.f;
      const float y0 = bfr(a0 / dd), y1 = bfr(a1 / dd), y2 = bfr(a2 / dd), y3 = bfr(a3 / dd);
      if (q == 0) sw[NS + 2 + u] = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
      __syncthreads();                                             // every poll has returned: re-zero the consumed words
      ks1b_st4(pa, 0u, 0u, 0u, 0u);
      if (q == 0) ks1b_st4(pb, 0u, 0u, 0u, 0u);
      if (mlw) ks1b_st2(pm, 0u, 0u);
      if (q == 0) {
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += sw[NS + 2 + i];
        const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
        const float2 g01 = bf2f(g3.x), g23 = bf2f(g3.y);
        const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0 * rs * (1.f / (1.f + __expf(-g01.x))), y1 * rs * (1.f / (1.f + __expf(-g01.y))));
        const __nv_bfloat162 o23 = __floats2bfloat162_rn(y2 * rs * (1.f / (1.f + __expf(-g23.x))), y3 * rs * (1.f / (1.f + __expf(-g23.y))));
        *reinterpret_cast<uint2*>(p.out + (size_t)(tb + tg * TPG + (r >> 3)) * (HQ * 64) + (size_t)(h * 8 + (r & 7)) * 64 + 4 * u) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&o01), *reinterpret_cast<const uint32_t*>(&o23));
      }
      KS1P(9);
      return;
    }
'''


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"anchor {what}: {n} matches")
    return text.replace(old, new)


def apply(text, insts, route):
    a = text.index("KS1_CUDA_SRC = r\"\"\"")
    b = text.index("\"\"\"", a + 20)
    cu = text[a:b]
    cu = once(cu, "__device__ __forceinline__ uint32_t vsl_sa(const void* x)",
              HELPERS + "__device__ __forceinline__ uint32_t vsl_sa(const void* x)", "helpers")
    l0 = cu.index("#pragma unroll\n  for (int kk = 0; kk < (oldc ? G::KPT : 0); ++kk) {")
    l1 = cu.index("#pragma unroll\n  for (int j = 0; j < 2; ++j) {\n    const int r = warp + 8 * j;\n    uint32_t hv = 0u;")
    conv = cu[l0:l1]
    assert conv.count("kh + swz(slot") == 2 and len(conv) < 2000, "conversion loop slice"
    body = "  if constexpr ((MG & 8) == 0) {   // KS1B: MG | 8 converts before the wait\n" + conv + "  }\n"
    cu = cu[:l0] + GATE3 + body + cu[l1:]
    cu = once(cu, "  KS1P(1);\n  pdl_wait();",
              "  if constexpr ((MG & 8) != 0) {   // KS1B: fp8 -> f16 K / V tiles before the wait (pre-wait loads only)\n"
              + conv + "  }\n  KS1P(1);\n  pdl_wait();", "prewait")
    cu = once(cu, "    const size_t prow = ((size_t)grp * (NS + 1) + s) * ROWS + mr;\n    if (p.phalf) {",
              (LL2 if arg("ll", "2") == "2" else LL) + "    const size_t prow = ((size_t)grp * (NS + 1) + s) * ROWS + mr;\n    if (p.phalf) {", "ll merge")
    text = text[:a] + cu + text[b:]
    old_i = "VSL_KS1_INSTS = ((1, 16, 1, 1), (8, 16, 4, 1))\n"
    ins = [(1, 16, 1, 1), (8, 16, 4, 1)] + [(l, *GEO[l], m) for l, m in insts if (l, *GEO[l], m) not in ((1, 16, 1, 1), (8, 16, 4, 1))]
    text = once(text, old_i, f"VSL_KS1_INSTS = {tuple(ins)!r}\n", "insts")
    if route:
        old_r = "VSL_ROUTE = {(1, 1): ('ks1', 1, 16, 1, 1), (1, 8): ('ks1', 8, 16, 4, 1), (16, 1): ('ks3', 16, 1)}\n"
        r = {(1, 1): ("ks1", 1, 16, 1, 1), (1, 8): ("ks1", 8, 16, 4, 1), (16, 1): ("ks3", 16, 1)}
        for l, m in route:
            r[(1, l)] = ("ks1", l, *GEO[l], m)
        text = once(text, old_r, f"VSL_ROUTE = {r!r}\n", "route")
    return text


def pairs(s):
    return [tuple(int(x) for x in p.split(":")) for p in s.split(",") if p]


def main():
    base = ROOT / arg("base", "kernels/muse_decode_layer/k3_gc_16.py")
    out = ROOT / arg("out", "kernels/lab/k3_ks1b_1_ab.py")
    if out.exists():
        raise SystemExit(f"{out} exists (never overwrite)")
    text = apply(base.read_text(encoding="utf-8"), pairs(arg("insts", "1:3,1:11,1:9,8:3,8:11,8:9")), pairs(arg("route", "")))
    ast.parse(text)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{out.relative_to(ROOT)}: {len(text.encode())} B")


if __name__ == "__main__":
    main()

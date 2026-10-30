"""vll_patches.py: VMS_LL = fence-free, counter-free VMS hand-offs with LL-style flagged words (moe redesign, R-vll).

Mechanism (NCCL LL protocol): every hand-off word is 8 bytes = 4 B data + 4 B launch generation flag, stored with one
st.relaxed.gpu.b64 (single-copy atomic) and read with ld.relaxed.gpu (L2). A reader that sees the flag sees the data, so
no writer fence, no counter, no acquire round trip. Compile-time VMS_LL bits:
  1 (LLH) h, A -> B: inter as [route][2048] words of 2 bf16 dims (helper lanes g, g^1 pair their dims through one shuffle);
          B warps poll their K slice at segment start in 2 bulk rounds of 32 words (spin on one word, load all in parallel,
          re-spin on a stale one). Replaces: helper __threadfence + doneA atomicAdd, the counter poll, ld.acquire, h load.
  2 (LLY) y, B -> combine: ybuf as [route][4096] words of one fp32 + per-B-unit sum-of-squares partials [route][256]
          (fixed-order block sum: deterministic, unlike the ss atomics). Combine CTAs load every word of an item in parallel
          and re-poll only stale ones. Replaces: the helper's trailing fence, the grid barrier, the ctr / ss resets.
  gen: flag = ctr[0] + 1 read at entry (as today). LLY: the LAST CTA to finish (ctr[3] count) bumps ctr[0], resets ctr[3].
          LLH alone keeps the shipped barrier / reset path (it bumps ctr[0]; the doneA counters stay 0).
Buffers: inter >= MAXP x 2 x ID bf16 (2 MB), ybuf >= MAXP x (2 x HD + 512) fp32 (4.25 MB); zero-initialised (flag 0 never
matches). At VMS_LL 0 the preprocessed source is the base's (SASS identical, tools/vll_local_test.py).

  python -I tools/vll_patches.py --check        # apply to the base VMS source; asserts every anchor once
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_b23_1.py"   # final base (VMS source identical to k3_gc_21cg_s1)


def _rep(s, old, new, tag):
    n = s.count(old)
    assert n == 1, f"vll anchor '{tag}' found {n} times"
    return s.replace(old, new)


HELPERS = r'''
#ifndef VMS_LL
#define VMS_LL 0   // R-vll: bit 1 = LL h hand-off (A -> B), bit 2 = LL y combine (no grid barrier)
#endif
#define VMS_LLH ((VMS_LL & 1) != 0)
#define VMS_LLY ((VMS_LL & 2) != 0)
#if VMS_LLY && !VMS_LLH
#error "VMS_LL: the LL combine (bit 2) needs the LL h hand-off (bit 1): no doneA counter reset without the barrier"
#endif
#if VMS_LL
// R-vll: LL words (4 B data | 4 B flag << 32), one relaxed 8-byte access each (single-copy atomic)
__device__ __forceinline__ void ll_st(unsigned long long* a, uint32_t data, uint32_t fl) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;\n" :: "l"(a), "l"(((unsigned long long)fl << 32) | data) : "memory");
}
__device__ __forceinline__ unsigned long long ll_ld(const unsigned long long* a) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];\n" : "=l"(v) : "l"(a) : "memory");
  return v;
}
__device__ __forceinline__ void ll_ld2(const unsigned long long* a, unsigned long long& v0, unsigned long long& v1) {
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];\n" : "=l"(v0), "=l"(v1) : "l"(a) : "memory");
}
__device__ __forceinline__ bool ll_ok(unsigned long long v, uint32_t fl) { return (uint32_t)(v >> 32) == fl; }
// one word: spin until its flag matches (bounded; err code ec)
__device__ __forceinline__ uint32_t ll_wait(const unsigned long long* a, uint32_t fl, int* err, int ec) {
  unsigned long long v = ll_ld(a);
  int n = 0;
  while (!ll_ok(v, fl)) {
    if (++n > (1 << 22)) { atomicExch(err, ec); break; }
    __nanosleep(32);
    v = ll_ld(a);
  }
  return (uint32_t)v;
}
// 4 groups of 8 consecutive words (group k at a + 32 k, 16-B aligned): load all 32 in parallel (16 x v2); while any lane of
// the warp has a stale word, ONE elected lane spins on its first stale word and the warp reloads (<= 1 spinner per warp).
// Call with the warp's participating lanes converged (mask = __activemask() at the call site).
__device__ __forceinline__ void ll_bulk32(const unsigned long long* a, uint32_t fl, int* err, uint32_t (&d)[4][8]) {
  const unsigned am = __activemask();
  const int ln = threadIdx.x & 31;
  int n = 0;
  for (;;) {
    unsigned long long v[4][8];
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int i = 0; i < 4; ++i) ll_ld2(a + 32 * k + 2 * i, v[k][2 * i], v[k][2 * i + 1]);
    int bad = -1;
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int i = 0; i < 8; ++i) { d[k][i] = (uint32_t)v[k][i]; if (bad < 0 && !ll_ok(v[k][i], fl)) bad = 8 * k + i; }
    const unsigned m = __ballot_sync(am, bad >= 0);
    if (!m) break;
    if (++n > 64) { if (bad >= 0) atomicExch(err, 5); break; }
    if (ln == __ffs(m) - 1) (void)ll_wait(a + 32 * (bad >> 3) + (bad & 7), fl, err, 4);
    __syncwarp(am);
  }
}
#endif
'''

OLD_SMEM = "  float4 red[NRB][NW][32];\n};\n"
NEW_SMEM = "  float4 red[NRB][NW][32];\n#if VMS_LLY\n  float rs[8][NW];   // R-vll: combine block sums\n#endif\n};\n"

# B segment start: no counter wait under LLH (the h words are polled at the load below)
OLD_BWAIT = "#if VMS_FX & 1\n          if (lane == 0) {\n            int n = 0;\n            while (ld_acq(p.ctr + 8 + jj) < RBA) {"
NEW_BWAIT = ("#if VMS_LLH\n          // R-vll: no doneA counter; the h words carry their own flags (polled at the load below)\n"
             "#elif VMS_FX & 1\n          if (lane == 0) {\n            int n = 0;\n            while (ld_acq(p.ctr + 8 + jj) < RBA) {")

OLD_HLOAD = ("        if (g < cnt) {\n"
             "          const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID)\n")
NEW_HLOAD = r'''#if VMS_LLH
        if (PB && g < cnt) {   // R-vll: this lane's 8 x 8 LL words of h (dims [8u, 8u + 16) per cc), flag = gen + 1
          const unsigned long long* hp = reinterpret_cast<const unsigned long long*>(p.inter) + (size_t)(st + g) * (ID / 2)
                                         + 256 * warp + 8 * t;
          const uint32_t fl = (uint32_t)S.gen + 1u;
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {   // cc 4 hh .. 4 hh + 3: 32 words, one bulk poll
            uint32_t dd[4][8];
            ll_bulk32(hp + 128 * hh, fl, p.ctr + 2, dd);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              xr[2 * (4 * hh + k)] = make_uint4(dd[k][0], dd[k][1], dd[k][2], dd[k][3]);
              xr[2 * (4 * hh + k) + 1] = make_uint4(dd[k][4], dd[k][5], dd[k][6], dd[k][7]);
            }
          }
        } else
#endif
''' + OLD_HLOAD
OLD_HLOAD_END = ("#pragma unroll\n          for (int cc = 0; cc < 16; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);\n        }\n      }\n")
NEW_HLOAD_END = ("#pragma unroll\n          for (int cc = 0; cc < 16; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);\n        }\n"
                 "#if VMS_LLH\n        __syncwarp();   // R-vll: lanes leave their polls at different times\n"
                 "        if (PB && mk && tid == 0 && mrow[4] == 0) mrow[4] = gt();   // h landed (LL: the B go)\n#endif\n      }\n")

AEPI_LOOP = ("#pragma unroll\n        for (int h = 0; h < 2; ++h) {\n          const int tk = 2 * t + h;\n          if (tk < cnt) {\n"
             "            const float ga = (h ? sm.y : sm.x) * gs, up = (h ? sm.w : sm.z) * gs;\n"
             "            const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;\n"
             "            p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);\n          }\n        }\n")
AEPI_CREDIT = "        if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units\n"
OLD_AEPI = AEPI_LOOP + AEPI_CREDIT
NEW_AEPI = r'''#if VMS_LLH
        {   // R-vll: lane (g, t) has dim rb*8+g for tokens 2t, 2t+1; lanes g, g^1 (lane ^ 4) pair dims into one LL word
          const float ga0 = sm.x * gs, up0 = sm.z * gs, ga1 = sm.y * gs, up1 = sm.w * gs;
          const __nv_bfloat16 h0 = __float2bfloat16(__fdividef(ga0, 1.f + __expf(-ga0)) * up0);
          const __nv_bfloat16 h1 = __float2bfloat16(__fdividef(ga1, 1.f + __expf(-ga1)) * up1);
          const uint32_t u0 = (uint32_t)__bfloat16_as_ushort(h0), u1 = (uint32_t)__bfloat16_as_ushort(h1);
          const uint32_t rcv = __shfl_xor_sync(0xffffffffu, (g & 1) ? u0 : u1, 4);
          const int tk = 2 * t + (g & 1);   // even g: token 2t dims (g, g+1); odd g: token 2t+1 dims (g-1, g)
          const uint32_t wd = (g & 1) ? (rcv | (u1 << 16)) : (u0 | (rcv << 16));
          if (tk < cnt) ll_st(reinterpret_cast<unsigned long long*>(p.inter) + (size_t)(st + tk) * (ID / 2) + rb * 4 + (g >> 1),
                              wd, (uint32_t)S.gen + 1u);
        }
        if (mk && lane == 0 && q == nA - 1) mrow[6] = gt();
        if (false) {   // the shipped credit (fence + doneA atomic) is not needed
#else
''' + AEPI_LOOP + AEPI_CREDIT + "#endif\n"

OLD_BEPI = ("          if (tk < cnt) {\n            p.ybuf[(size_t)(st + tk) * HD + n0] = y0;\n            p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;\n"
            "          }\n          sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n          sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n"
            "          sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n          if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);\n")
NEW_BEPI = r'''#if VMS_LLY
          {   // R-vll: y as LL fp32 words, one sum-of-squares partial per (route, B unit)
            unsigned long long* yl = reinterpret_cast<unsigned long long*>(p.ybuf);
            const uint32_t fl = (uint32_t)S.gen + 1u;
            if (tk < cnt) {
              ll_st(yl + (size_t)(st + tk) * HD + n0, __float_as_uint(y0), fl);
              ll_st(yl + (size_t)(st + tk) * HD + n0 + 8, __float_as_uint(y1), fl);
            }
            sq += __shfl_xor_sync(0xffffffffu, sq, 4);
            sq += __shfl_xor_sync(0xffffffffu, sq, 8);
            sq += __shfl_xor_sync(0xffffffffu, sq, 16);
            if (g == 0 && tk < cnt) ll_st(yl + (size_t)MAXP * HD + (size_t)(st + tk) * RBB + rb, __float_as_uint(sq), fl);
          }
#else
''' + OLD_BEPI + "#endif\n"

OLD_HFENCE = ("#if VMS_FX & 2\n    asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");\n#else\n    __threadfence();\n#endif\n"
              "    if (mk && lane == 0) mrow[7] = gt();\n")
NEW_HFENCE = ("#if VMS_LLY\n    // R-vll: no trailing fence (every y / partial word is self-flagged)\n#elif VMS_FX & 2\n"
              "    asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");\n#else\n    __threadfence();\n#endif\n"
              "    if (mk && lane == 0) mrow[7] = gt();\n")

OLD_TAIL_HEAD = "  const int gen = S.gen;\n  if (tid == 0) {\n#if VMS_FX & 2\n"
LL_TAIL = r'''#if VMS_LLY
  {   // R-vll: no grid barrier: combine items poll their routes' y words and sum-of-squares partials directly
    const uint32_t fl = (uint32_t)S.gen + 1u;
    const unsigned long long* yl = reinterpret_cast<const unsigned long long*>(p.ybuf);
    const unsigned long long* sqp = yl + (size_t)MAXP * HD;
    if (mk && tid == 0) mrow[8] = gt();
    for (int it = c; it < p.m * 4; it += G) {
      const int tk = it >> 2, col = (it & 3) * 1024 + (tid & 255) * 4;
      if (tid >= NT) break;
      uint32_t vm = 0;   // valid route slots of this token (same in every thread: S.pos)
#pragma unroll
      for (int s = 0; s < 8; ++s) vm |= (S.pos[tk * 8 + s] >= 0 ? 1u : 0u) << s;
      float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
      while (vm) {   // two routes per round: their 2 partials + 8 y words in parallel, re-spin on a stale one
        const int s0 = __ffs(vm) - 1;
        vm &= vm - 1;
        const int s1 = vm ? __ffs(vm) - 1 : -1;
        if (s1 >= 0) vm &= vm - 1;
        const int p0 = S.pos[tk * 8 + s0], p1 = s1 >= 0 ? S.pos[tk * 8 + s1] : p0;
        float pv[2], y4[2][4];
        const unsigned long long* spin = nullptr;
        int n = 0;
        for (;;) {   // all 256 threads load; one elected lane per warp spins on a stale word, then the warp reloads
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const int pr = r ? p1 : p0;
            const unsigned long long* pw = sqp + (size_t)pr * RBB + tid;
            const unsigned long long* yw = yl + (size_t)pr * HD + col;
            unsigned long long a0 = ll_ld(pw), b0, b1, b2, b3;
            ll_ld2(yw, b0, b1);
            ll_ld2(yw + 2, b2, b3);
            pv[r] = __uint_as_float((uint32_t)a0);
            y4[r][0] = __uint_as_float((uint32_t)b0); y4[r][1] = __uint_as_float((uint32_t)b1);
            y4[r][2] = __uint_as_float((uint32_t)b2); y4[r][3] = __uint_as_float((uint32_t)b3);
            if (!spin) {
              if (!ll_ok(a0, fl)) spin = pw;
              else if (!ll_ok(b0, fl)) spin = yw;
              else if (!ll_ok(b1, fl)) spin = yw + 1;
              else if (!ll_ok(b2, fl)) spin = yw + 2;
              else if (!ll_ok(b3, fl)) spin = yw + 3;
            }
          }
          const unsigned m = __ballot_sync(0xffffffffu, spin != nullptr);
          if (!m) break;
          if (++n > 256) { if (spin) atomicExch(p.ctr + 2, 7); break; }
          if (lane == __ffs(m) - 1) (void)ll_wait(spin, fl, p.ctr + 2, 6);
          __syncwarp();
          spin = nullptr;
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {   // block sum of the route's 256 partials (one per thread, fixed order)
          float v = pv[r];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
          if (lane == 0) S.rs[r][warp] = v;
        }
        cta_sync_compute();
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          if (r == 1 && s1 < 0) break;
          float ss2 = 0.f;
#pragma unroll
          for (int w = 0; w < NW; ++w) ss2 += S.rs[r][w];
          const float cf = __ldg(p.tw + tk * 8 + (r ? s1 : s0)) * rsqrtf(ss2 * (1.f / HD) + p.eps);
          a.x += cf * y4[r][0]; a.y += cf * y4[r][1]; a.z += cf * y4[r][2]; a.w += cf * y4[r][3];
        }
        cta_sync_compute();   // S.rs is rewritten by the next round / item
      }
      __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x, a.y), o1 = __floats2bfloat162_rn(a.z, a.w);
      uint2 pk; pk.x = *reinterpret_cast<uint32_t*>(&o0); pk.y = *reinterpret_cast<uint32_t*>(&o1);
      *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = pk;
    }
    if (tid == 0) {   // the last CTA to finish starts the next generation (every CTA read gen at entry)
      if (atomicAdd(p.ctr + 3, 1) == G - 1) { p.ctr[3] = 0; atomicAdd(p.ctr, 1); }
    }
  }
#else
'''
OLD_TAIL_END = "  if (mk && tid == 0) mrow[9] = gt();\n"
NEW_TAIL_END = "#endif\n  if (mk && tid == 0) mrow[9] = gt();\n"


def apply(src):
    s = src
    s = _rep(s, "namespace vms {\n", HELPERS + "namespace vms {\n", "helpers")
    s = _rep(s, OLD_SMEM, NEW_SMEM, "smem")
    s = _rep(s, OLD_BWAIT, NEW_BWAIT, "bwait")
    s = _rep(s, OLD_HLOAD, NEW_HLOAD, "hload")
    s = _rep(s, OLD_HLOAD_END, NEW_HLOAD_END, "hload end")
    s = _rep(s, OLD_AEPI, NEW_AEPI, "A epilogue")
    s = _rep(s, OLD_BEPI, NEW_BEPI, "B epilogue")
    s = _rep(s, OLD_HFENCE, NEW_HFENCE, "helper fence")
    s = _rep(s, OLD_TAIL_HEAD, LL_TAIL + OLD_TAIL_HEAD, "tail")
    s = _rep(s, OLD_TAIL_END, NEW_TAIL_END, "tail end")
    return s


def layer_src(path=BASE):
    L = pathlib.Path(path).read_text(encoding="utf-8")
    a = L.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    c = L.index('VMS_CPP_SRC = r"""') + len('VMS_CPP_SRC = r"""')
    return L[a:L.index('"""', a)], L[c:L.index('"""', c)]


if __name__ == "__main__":
    src0, _ = layer_src()
    s = apply(src0)
    print(f"[vll] base VMS source {len(src0)} B -> patched {len(s)} B (+{len(s) - len(src0)})")
    if "--check" in sys.argv:
        print("[vll] all anchors applied once")

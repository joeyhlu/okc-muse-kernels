"""KS3G: decode SLIDING attention at B = 16 / 32 (q_len 1), many warp-private gathers + pipelined consumption + red merge.

Differences from KS3 (k3_gc_18, namespace ks3): every warp owns (KV head hw, key part) and gathers ONLY its own data
(page-table entries -> shfl -> 16-B cp.async of its head's 64-B K and V halves), one commit group per 32-key chunk,
consumes chunk c after wait_group(NCH-1-c) + __syncwarp (no block barrier before compute); q-prep per warp (8 heads);
the new key sits in slot n_old of the window (no extra tile); each warp red.adds its partial in the absolute exp2
frame straight into a zeroed workspace (VSL MG 1 numerics), ticket, last CTA finalizes (o-norm + sigmoid gate) and
re-zeroes. NULLC = null consumer (gather + one smem read per lane) for the ceiling probe. Grid (NS, B), 64 NP threads.
Appended to gc_18's KS1_CUDA_SRC (helpers from namespace ks1)."""

KS3G_CPP = r"""
void ks3g_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                  torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                  torch::Tensor ws_o, torch::Tensor ws_l, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                  int64_t NS, int64_t NP, int64_t nullc, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope);
std::vector<int64_t> ks3g_info(int64_t NS, int64_t NP, int64_t nullc);
"""

KS3G_CU = r"""
namespace ks3g {
using namespace ks1;

struct PG {
  const __nv_bfloat16* qkv;
  const int* cache_seqlens;
  const int64_t* cache_loc;
  const int64_t* positions;
  const int* page_table;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const float* cos_sin;
  float* ws_o;            // [B][16][64] fp32, zero between calls (self-resetting)
  float* ws_l;            // [B][16]
  int* counters;          // [B]
  __nv_bfloat16* out;     // [B, 1024]
  long long* prof;        // [B NS][8] or nullptr
  int stride_qkv, pt_cols, B, trig, rope;
  float qk_sm, eps;
};

__device__ __forceinline__ void cpa16(void* sp, const void* gp, int nb) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(sp);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gp), "r"(nb) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void waitg() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void waitg_rt(int n) {   // n = groups allowed in flight (<= 7)
  switch (n) {
    case 0: waitg<0>(); break; case 1: waitg<1>(); break; case 2: waitg<2>(); break; case 3: waitg<3>(); break;
    case 4: waitg<4>(); break; case 5: waitg<5>(); break; case 6: waitg<6>(); break; default: waitg<7>(); break;
  }
}
// 16-B slot of (key x, chunk c) in a warp tile of 64-B rows (two keys per 128-B line): conflict-free LDS.128 for the
// QK fragments (keys x0 + 8 j + g, chunk t) and LDS.64 for the PV fragments (keys x0 + 16 kk + 2 t (+1, +8, +9), chunk g / 2).
__device__ __forceinline__ int gsl(int x, int c) { return (x >> 1) * 8 + (((x ^ (x >> 1)) & 1) << 2) + (c ^ (((x >> 2) & 1) << 1)); }
#define KGP(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[cta * 8 + (k)] = gtime(); } while (0)

__device__ __forceinline__ float2 g_norm_rope(uint32_t raw, float2 cs, int rope, float eps) {
  const float2 x = bf2f(raw);
  const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
  const float rs = rsqrtf(var + eps);
  const float e = bfr(x.x * rs), o = bfr(x.y * rs);
  if (!rope) return make_float2(e, o);
  return make_float2(bfr(__fmaf_rn(e, cs.x, -__fmul_rn(o, cs.y))), bfr(__fmaf_rn(e, cs.y, __fmul_rn(o, cs.x))));
}

template <typename F>
__device__ __forceinline__ void gchunk(const uint8_t* Kw, const uint8_t* Vw, const int x0, const uint32_t (&qf)[4][2],
                                       float (&o)[8][4], float& m, float& l, const float cl2, F valid) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  float sc[4][2];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int key = x0 + 8 * j + g;
    const uint4 kv = *reinterpret_cast<const uint4*>(Kw + 16 * gsl(key, t));
    const uint32_t wv[4] = {kv.x, kv.y, kv.z, kv.w};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t a[4] = {qf[s][0], 0u, qf[s][1], 0u};
      mma16816(c, a, e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
    }
    sc[j][0] = c[0];
    sc[j][1] = c[1];
  }
  float x = neg_inf();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      sc[j][e] = valid(x0 + 8 * j + 2 * t + e) ? sc[j][e] * cl2 : neg_inf();
      x = fmaxf(x, sc[j][e]);
    }
  }
  x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
  x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
  const float nm = fmaxf(m, x);
  const float u = (nm == neg_inf()) ? 0.f : nm;
  const float al = ex2(m - u);
  m = nm;
  l *= al;
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] *= al; o[n][1] *= al; }
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    const float p00 = ex2(sc[2 * kk][0] - u), p01 = ex2(sc[2 * kk][1] - u);
    const float p10 = ex2(sc[2 * kk + 1][0] - u), p11 = ex2(sc[2 * kk + 1][1] - u);
    l += (p00 + p01) + (p10 + p11);
    const uint32_t pa[4] = {pack_h2(p00, p01), 0u, pack_h2(p10, p11), 0u};
    const int xa = x0 + 16 * kk + 2 * t;
    const int vo = 8 * (g & 1), vc = g >> 1;
    const uint2 va = *reinterpret_cast<const uint2*>(Vw + 16 * gsl(xa, vc) + vo);
    const uint2 vb = *reinterpret_cast<const uint2*>(Vw + 16 * gsl(xa + 1, vc) + vo);
    const uint2 vcc = *reinterpret_cast<const uint2*>(Vw + 16 * gsl(xa + 8, vc) + vo);
    const uint2 vd = *reinterpret_cast<const uint2*>(Vw + 16 * gsl(xa + 9, vc) + vo);
    const uint32_t w01[4] = {__byte_perm(va.x, vb.x, 0x5140), __byte_perm(va.x, vb.x, 0x7362),
                             __byte_perm(va.y, vb.y, 0x5140), __byte_perm(va.y, vb.y, 0x7362)};
    const uint32_t w23[4] = {__byte_perm(vcc.x, vd.x, 0x5140), __byte_perm(vcc.x, vd.x, 0x7362),
                             __byte_perm(vcc.y, vd.y, 0x5140), __byte_perm(vcc.y, vd.y, 0x7362)};
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      const uint32_t s01 = w01[jn >> 1], s23 = w23[jn >> 1];
      mma16816(o[jn], pa, e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01), e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23));
    }
  }
}

template <int NS, int NP> struct GG {
  static constexpr int NW = 2 * NP, NT = 32 * NW, KMAX = WIN / NS, KPW = KMAX / NP, NCH = KPW / 32;
  static constexpr int OFF_Q = NW * KPW * 128;      // per warp: K tile KPW x 64 B, V tile KPW x 64 B
  static constexpr int OFF_MISC = OFF_Q + NW * 1024;
  static constexpr int SMEM = OFF_MISC + 64;
  static_assert(KPW % 32 == 0 && NCH >= 1 && NCH <= 8, "KS3G geometry");
};

template <int NS, int NP, bool NULLC>
__global__ void __launch_bounds__(64 * NP) ks3g_kernel(const __grid_constant__ PG p) {
#if KS1_ARCH_OK
  using G = GG<NS, NP>;
  constexpr int KMAX = G::KMAX, KPW = G::KPW, NCH = G::NCH, NT = G::NT;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, b = blockIdx.y, cta = b * NS + s;
  const int hw = warp & 1, part = warp >> 1;
  uint8_t* Kw = smem + warp * (KPW * 128);
  uint8_t* Vw = Kw + KPW * 64;
  uint8_t* qs = smem + G::OFF_Q + warp * 1024;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  if (p.trig == 2) pdl_trigger();
  KGP(0);
  const int seq = max(__ldg(p.cache_seqlens + b) - 1, 0);
  const int n_old = min(seq, WIN - 1);
  const int kstart = seq - n_old;
  const int base = s * KMAX + part * KPW;                       // first window slot of this warp
  {
    const int* pt = p.page_table + (size_t)b * p.pt_cols + kstart;
    int prow[NCH];
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const int gs = base + 32 * c + lane;
      prow[c] = (gs < n_old && kstart + gs < p.pt_cols) ? __ldg(pt + gs) : -1;
    }
    const int ch = lane & 3;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int kk = (lane >> 2) + 8 * i, x = 32 * c + kk;
        const int r = __shfl_sync(FULL, prow[c], kk);
        if (base + x != n_old) {                                 // slot n_old = the new key (written after the wait)
          const bool ok = r >= 0;
          const size_t go = (size_t)(ok ? r : 0) * 128 + hw * 64 + 16 * ch;
          cpa16(Kw + 16 * gsl(x, ch), p.k_cache + go, ok ? 16 : 0);
          cpa16(Vw + 16 * gsl(x, ch), p.v_cache + go, ok ? 16 : 0);
        }
      }
      commit();
    }
  }
  KGP(1);
  if constexpr (NULLC) {
    waitg<0>();
    __syncwarp();
    const uint4 v = *reinterpret_cast<const uint4*>(Kw + 16 * lane);
    pdl_wait();
    if ((v.x ^ v.y ^ v.z ^ v.w) == 0x9e3779b9u) p.counters[b] = 1;
    KGP(7);
    return;
  }
  float2 cs = make_float2(1.f, 0.f);
  if (p.rope) {
    const long long pos = __ldg(p.positions + b);
    cs = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
  }
  const int xn = n_old - base;                                   // tile slot of the new key, if this warp owns it
  const bool own_new = (xn >= 0 && xn < KPW);
  long long loc = 0;
  if (own_new) loc = __ldg(p.cache_loc + b);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  KGP(2);

  const __nv_bfloat16* xr = p.qkv + (size_t)b * p.stride_qkv;
  uint32_t qr[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) qr[i] = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(hw * 8 + i) * 64) + lane);
  uint32_t nk = 0u, nv = 0u;
  if (own_new) {
    nk = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + hw) * 64) + lane);
    nv = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + HKV + hw) * 64) + lane);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 a = g_norm_rope(qr[i], cs, p.rope, p.eps);
    reinterpret_cast<uint32_t*>(qs + i * 128)[lane] = pack_h2(a.x, a.y);
  }
  if (own_new) {
    float2 kf = g_norm_rope(nk, cs, p.rope, p.eps);
    float2 vf = bf2f(nv);
    kf.x = fminf(fmaxf(kf.x, -FP8_MAX), FP8_MAX); kf.y = fminf(fmaxf(kf.y, -FP8_MAX), FP8_MAX);
    vf.x = fminf(fmaxf(vf.x, -FP8_MAX), FP8_MAX); vf.y = fminf(fmaxf(vf.y, -FP8_MAX), FP8_MAX);
    const uint16_t k8 = (uint16_t)__nv_cvt_float2_to_fp8x2(kf, __NV_SATFINITE, __NV_E4M3);
    const uint16_t v8 = (uint16_t)__nv_cvt_float2_to_fp8x2(vf, __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint16_t*>(p.k_cache + (size_t)loc * 128 + hw * 64 + 2 * lane) = k8;
    *reinterpret_cast<uint16_t*>(p.v_cache + (size_t)loc * 128 + hw * 64 + 2 * lane) = v8;
    const int so = 16 * gsl(xn, lane >> 3) + ((2 * lane) & 15);
    *reinterpret_cast<uint16_t*>(Kw + so) = k8;
    *reinterpret_cast<uint16_t*>(Vw + so) = v8;
  }
  __syncwarp();
  uint32_t qf[4][2];
  {
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs + g * 128);
#pragma unroll
    for (int k = 0; k < 4; ++k) { qf[k][0] = qa[8 * t + 2 * k]; qf[k][1] = qa[8 * t + 2 * k + 1]; }
  }
  KGP(3);
  const float cl2 = p.qk_sm * LOG2E;
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m = neg_inf(), l = 0.f;
  const int lim = n_old + 1 - base;                              // valid tile slots: x < lim (old keys + the new key)
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    waitg_rt(NCH - 1 - c);
    __syncwarp();
    if (32 * c < lim) gchunk(Kw, Vw, 32 * c, qf, o, m, l, cl2, [&](int k) { return k < lim; });
  }
  l += __shfl_xor_sync(FULL, l, 1);
  l += __shfl_xor_sync(FULL, l, 2);
  KGP(4);
  {
    const float f = (m == neg_inf()) ? 0.f : ex2(fminf(m, 126.f));
    float* wo = p.ws_o + ((size_t)b * 16 + hw * 8 + g) * 64;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int d0 = 8 * (2 * t + e);
      vsl_red4(wo + d0, o[0][e] * f, o[1][e] * f, o[2][e] * f, o[3][e] * f);
      vsl_red4(wo + d0 + 4, o[4][e] * f, o[5][e] * f, o[6][e] * f, o[7][e] * f);
    }
    if (t == 0) vsl_red1(p.ws_l + (size_t)b * 16 + hw * 8 + g, l * f);
  }
  KGP(5);
  __syncthreads();
  if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
  __syncthreads();
  KGP(6);
  if (misc[0] == 0) return;
#pragma unroll
  for (int it = tid; it < 256; it += NT) {
    const int mr = it >> 4, mu = it & 15;
    float* wa = p.ws_o + ((size_t)b * 16 + mr) * 64 + 4 * mu;
    float* wl = p.ws_l + (size_t)b * 16 + mr;
    const uint2 graw = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 4 * mu));
    float den = (mu == 0) ? __ldcg(wl) : 0.f;
    const float4 v = __ldcg(reinterpret_cast<const float4*>(wa));
    __stcg(reinterpret_cast<float4*>(wa), make_float4(0.f, 0.f, 0.f, 0.f));
    if (mu == 0) __stcg(wl, 0.f);
    den = __shfl_sync(FULL, den, lane & ~15);
    vsl_fin3(v, den, graw, p.eps, p.out + (size_t)b * (HQ * 64) + (size_t)mr * 64 + 4 * mu);
  }
  if (tid == 0) p.counters[b] = 0;
  KGP(7);
#endif
}

template <int NS, int NP, bool NULLC>
static void launchg(const PG& P, bool pdl, cudaStream_t st) {
  constexpr int SM = GG<NS, NP>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks3g_kernel<NS, NP, NULLC>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)GG<NS, NP>::NT, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  unsigned na = 0;
  if (pdl) {
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    na = 1;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks3g_kernel<NS, NP, NULLC>, P));
}

template <int NS, int NP, bool NULLC>
static std::vector<int64_t> infog() {
  constexpr int SM = GG<NS, NP>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks3g_kernel<NS, NP, NULLC>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks3g_kernel<NS, NP, NULLC>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks3g_kernel<NS, NP, NULLC>, GG<NS, NP>::NT, SM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
}
}  // namespace ks3g

#define KS3G_LIST(X) X(8, 2) X(16, 2) X(32, 2) X(8, 4) X(16, 4)

void ks3g_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                  torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                  torch::Tensor ws_o, torch::Tensor ws_l, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                  int64_t NS, int64_t NP, int64_t nullc, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {
  const int B = (int)qkv.size(0);
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 &&
              qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64, "ks3g: qkv");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous() && cache_seqlens.numel() >= B, "ks3g: seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == B && positions.numel() == B, "ks3g: loc / pos");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.is_contiguous() && page_table.size(0) == B, "ks3g: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous(), "ks3g: caches");
  TORCH_CHECK(ws_o.scalar_type() == at::kFloat && ws_o.numel() >= B * 16 * 64 && ws_l.numel() >= B * 16, "ks3g: ws");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= B, "ks3g: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * ks1::HQ * 64, "ks3g: out");
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 8 * B * NS), "ks3g: prof");
  ks3g::PG P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr<at::BFloat16>());
  P.cache_seqlens = cache_seqlens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.positions = positions.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = k_cache.data_ptr<uint8_t>();
  P.v_cache = v_cache.data_ptr<uint8_t>();
  P.cos_sin = rope ? cos_sin.data_ptr<float>() : nullptr;
  P.ws_o = ws_o.data_ptr<float>();
  P.ws_l = ws_l.data_ptr<float>();
  P.counters = counters.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  P.prof = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  P.stride_qkv = (int)qkv.stride(0);
  P.pt_cols = (int)page_table.size(1);
  P.B = B;
  P.trig = (int)trig;
  P.rope = (int)rope;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define KS3G_X(NS_, NP_) \
  if (NS == NS_ && NP == NP_) { if (nullc) ks3g::launchg<NS_, NP_, true>(P, pdl != 0, st); else ks3g::launchg<NS_, NP_, false>(P, pdl != 0, st); return; }
  KS3G_LIST(KS3G_X)
#undef KS3G_X
  TORCH_CHECK(false, "ks3g: (NS, NP) not built");
}

std::vector<int64_t> ks3g_info(int64_t NS, int64_t NP, int64_t nullc) {
#define KS3G_X(NS_, NP_) \
  if (NS == NS_ && NP == NP_) return nullc ? ks3g::infog<NS_, NP_, true>() : ks3g::infog<NS_, NP_, false>();
  KS3G_LIST(KS3G_X)
#undef KS3G_X
  TORCH_CHECK(false, "ks3g: (NS, NP) not built");
  return {};
}
"""

GC18 = "kernels/muse_decode_layer/k3_gc_18.py"
CONST = {"KS1_CUDA_SRC", "KS1_CPP_SRC", "KS1_INSTS", "K1FA4_RMS_EPS", "K1FA4_ROPE_POSITIONS", "KS3_SELF_TEST_CASES"}
FUNCS = {"k1fa4_cos_sin", "_k1fa4_case", "_k1fa4_reference"}


def load_gc18(root):
    import ast
    import torch
    modp = root / GC18
    ns = {"torch": torch, "_k1fa4_rope": {}}
    for node in ast.parse(modp.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in CONST:
            ns[node.targets[0].id] = eval(compile(ast.Expression(node.value), str(modp), "eval"), {})
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            exec(compile(ast.Module([node], []), str(modp), "exec"), ns)
    return ns


def build(root, arch="90a", name="ks3g_v1", local=False):
    """KS1/KS3 source of gc_18 (KS1 X insts, KS3 NS 4/8/16 MG 0 + VSL KS3 NS 16 MG 1) + KS3G."""
    from torch.utils.cpp_extension import load_inline
    m = load_gc18(root)
    src = m["KS1_CUDA_SRC"]
    if local:
        src = src.replace("(__CUDA_ARCH__ >= 900)", "(__CUDA_ARCH__ >= 800)")
        for ins in ('asm volatile("griddepcontrol.wait;\\n" ::: "memory");',
                    'asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");'):
            src = src.replace(ins, "")
        for old, new in (('asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");',
                          "atomicAdd(a, x); atomicAdd(a + 1, y); atomicAdd(a + 2, z); atomicAdd(a + 3, w);"),
                         ('asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\\n" ::"l"(a), "f"(x), "f"(y) : "memory");',
                          "atomicAdd(a, x); atomicAdd(a + 1, y);")):
            assert src.count(old) == 1, old
            src = src.replace(old, new)
    insts = " ".join(f"KS1_X({l}, {n}, {g})" for l, n, g in m["KS1_INSTS"])
    src = src.replace("@INSTS@", insts).replace("@MINSTS@", "").replace("@M3INSTS@", "KS3_XM(16, 1)") + KS3G_CU
    mod = load_inline(name=name, cpp_sources=m["KS1_CPP_SRC"] + KS3G_CPP, cuda_sources=src,
                      functions=["ks3_forward", "ks3_info", "ks3g_forward", "ks3g_info"],
                      extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)
    return mod, m

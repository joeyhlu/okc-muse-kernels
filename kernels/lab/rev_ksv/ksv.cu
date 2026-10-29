// ksv.cu - R-rev-18 (notes/study/independent_review_oct28.md s25): KSV = ONE-launch sliding-window attention for the 8-token
// verify block at B > 1 (verify16 / verify32 sliding layers). The shipped path there is a 3-launch chain (prologue -> FA4 core,
// one CTA per request x head looping over all 2048 keys -> combine-gate): judge k3_gc_7 15.7 / 19.2 us.
// Skeleton of KS3 (tools/ks3_patches.py; helpers copied, not shared): grid (NS splits, B requests), both KV heads per CTA,
// 128-B FP8 K / V rows staged by cp.async into swizzled smem (the whole split at once), K1 register fragments. New for q_len 8:
//   - each warp owns 16 query rows = 2 tokens x the 8 q heads of one KV head (KS3 left MMA rows 8-15 zero at q_len 1) and loops
//     over ALL keys of its split (no intra-CTA merge);
//   - per-token sliding-window / causal masks; the block's 8 new keys are written to the cache and attended from a smem tile by
//     the last split;
//   - split partials are added in the absolute exp2 frame with red.add (KS3's MG 1 'red' merge); the last CTA of a request
//     normalizes, RMSNorm(64) x sigmoid(gate), writes bf16 and re-zeroes the accumulators (self-resetting state).
// Arch: mma.sync / cp.async from sm_80 (local RTX 3070 test), red.v4 and griddepcontrol on sm_90 only.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define KSV_MMA 1
#else
#define KSV_MMA 0
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define KSV_SM90 1
#else
#define KSV_SM90 0
#endif

namespace ksv {
constexpr int HQ = 16, HKV = 2, WIN = 2048, L = 8, ROWS = L * HQ;   // 128 query rows per request
constexpr int NWARP = 8, NTHR = NWARP * 32;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float FP8_MAX = 448.0f;
constexpr unsigned FULL = 0xffffffffu;

struct Params {
  const __nv_bfloat16* qkv;   // [B*L, stride]: q (16 x 64) | k (2 x 64) | v (2 x 64) | gate (16 x 64) per token
  const int* seq_lens;        // [B] cached tokens before the block
  const int64_t* cache_loc;   // [B*L] flat cache row of each new token
  const int64_t* positions;   // [B*L]
  const int* page_table;      // [B, pt_cols] (page size 1)
  uint8_t* k_cache;           // e4m3 rows of 2 x 64 B (both KV heads)
  uint8_t* v_cache;
  const float* cos_sin;       // [maxpos][64]: 32 cos | 32 sin
  float* acc_o;               // [B][128][64] zero on entry, re-zeroed by each request's last CTA
  float* acc_l;               // [B][128]
  int* counters;              // [B]
  __nv_bfloat16* out;         // [B*L, 1024]
  int stride_qkv, pt_cols, rope;
  float qk_sm, eps;
};

template <int NS> struct G {
  static constexpr int KMAX = WIN / NS;                 // old-key slots per split
  static constexpr int NCH = KMAX / 32;
  static constexpr int OFF_V = KMAX * 128;
  static constexpr int OFF_KN = 2 * KMAX * 128;         // new-key tiles: 32 rows (0..7 the block's keys, the rest zero)
  static constexpr int OFF_VN = OFF_KN + 32 * 128;
  static constexpr int OFF_Q = OFF_VN + 32 * 128;       // q fp16 [128 rows (token x 16 + head)][64]
  static constexpr int OFF_MISC = OFF_Q + ROWS * 128;
  static constexpr int SMEM = OFF_MISC + 64;
  static_assert(KMAX % 32 == 0 && (KMAX * 8) % NTHR == 0, "KSV geometry");
  static_assert(2 * KMAX * 128 >= ROWS * 64 * 4 + ROWS * 4, "o / l staging must fit in the K | V region");
};

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(FULL, v, d);
  return v;
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t v) {
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xFFFFu), __NV_E4M3);
  return (uint32_t)r.x | ((uint32_t)r.y << 16);
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 bf2f(uint32_t v) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v)); }
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if KSV_MMA
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
#if KSV_MMA
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gptr), "r"(src_bytes) : "memory");
#endif
}
__device__ __forceinline__ void cp_commit() {
#if KSV_MMA
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_wait_all() {
#if KSV_MMA
  asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}
__device__ __forceinline__ int atom_add_acqrel(int* a, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void red4(float* a, float4 v) {
#if KSV_SM90
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
#else
  atomicAdd(a, v.x); atomicAdd(a + 1, v.y); atomicAdd(a + 2, v.z); atomicAdd(a + 3, v.w);
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if KSV_SM90
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
// KS3's swizzles: K chunks XOR 4 on odd rows, V chunks XOR 2 x ((row >> 1) & 3); V fragments as 8-dim slices per (head, g)
__device__ __forceinline__ int kswz(int x, int ch) { return ch ^ (4 * (x & 1)); }
__device__ __forceinline__ int vswz(int x, int ch) { return ch ^ (2 * ((x >> 1) & 3)); }
__device__ __forceinline__ uint2 vld(const uint8_t* Vs, int x, int h, int g) {
  return *reinterpret_cast<const uint2*>(Vs + x * 128 + 16 * vswz(x, 4 * h + (g >> 1)) + 8 * (g & 1));
}
// RMSNorm over 64 dims (2 per lane) then interleaved RoPE, bf16-rounded like KS3
__device__ __forceinline__ float2 norm_rope(uint32_t raw, float2 cs, int rope, float eps) {
  const float2 x = bf2f(raw);
  const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
  const float rs = rsqrtf(var + eps);
  const float e = bfr(x.x * rs), o = bfr(x.y * rs);
  if (!rope) return make_float2(e, o);
  return make_float2(bfr(__fmaf_rn(e, cs.x, -__fmul_rn(o, cs.y))), bfr(__fmaf_rn(e, cs.y, __fmul_rn(o, cs.x))));
}

// One 32-key chunk for a warp's 16 rows: row g = token A (qa), row g + 8 = token B (qb), both heads 8 h + g.
template <typename F>
__device__ __forceinline__ void chunk32(const uint8_t* Ks, const uint8_t* Vs, const int x0, const int h, const uint32_t (&qa)[4][2],
                                        const uint32_t (&qb)[4][2], float (&o)[8][4], float (&m)[2], float (&l)[2],
                                        const float cl2, F valid) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  float sc[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int key = x0 + 8 * j + g;
    const uint4 kv = *reinterpret_cast<const uint4*>(Ks + key * 128 + 16 * kswz(key, 4 * h + t));
    const uint32_t wv[4] = {kv.x, kv.y, kv.z, kv.w};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t a[4] = {qa[s][0], qb[s][0], qa[s][1], qb[s][1]};
      mma16816(c, a, e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) sc[j][e] = c[e];
  }
  float xm[2] = {neg_inf(), neg_inf()};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int k = x0 + 8 * j + 2 * t + e;
      sc[j][e] = valid(0, k) ? sc[j][e] * cl2 : neg_inf();
      sc[j][2 + e] = valid(1, k) ? sc[j][2 + e] * cl2 : neg_inf();
      xm[0] = fmaxf(xm[0], sc[j][e]);
      xm[1] = fmaxf(xm[1], sc[j][2 + e]);
    }
  }
  float u[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    xm[r] = fmaxf(xm[r], __shfl_xor_sync(FULL, xm[r], 1));
    xm[r] = fmaxf(xm[r], __shfl_xor_sync(FULL, xm[r], 2));
    const float nm = fmaxf(m[r], xm[r]);
    u[r] = (nm == neg_inf()) ? 0.f : nm;
    const float al = ex2(m[r] - u[r]);
    m[r] = nm;
    l[r] *= al;
#pragma unroll
    for (int n = 0; n < 8; ++n) { o[n][2 * r] *= al; o[n][2 * r + 1] *= al; }
  }
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    const float p00 = ex2(sc[2 * kk][0] - u[0]), p01 = ex2(sc[2 * kk][1] - u[0]);
    const float p10 = ex2(sc[2 * kk + 1][0] - u[0]), p11 = ex2(sc[2 * kk + 1][1] - u[0]);
    const float q00 = ex2(sc[2 * kk][2] - u[1]), q01 = ex2(sc[2 * kk][3] - u[1]);
    const float q10 = ex2(sc[2 * kk + 1][2] - u[1]), q11 = ex2(sc[2 * kk + 1][3] - u[1]);
    l[0] += (p00 + p01) + (p10 + p11);
    l[1] += (q00 + q01) + (q10 + q11);
    const uint32_t pa[4] = {pack_h2(p00, p01), pack_h2(q00, q01), pack_h2(p10, p11), pack_h2(q10, q11)};
    const int xa = x0 + 16 * kk + 2 * t;
    const uint2 va = vld(Vs, xa, h, g), vb = vld(Vs, xa + 1, h, g), vc = vld(Vs, xa + 8, h, g), vd = vld(Vs, xa + 9, h, g);
    const uint32_t w01[4] = {__byte_perm(va.x, vb.x, 0x5140), __byte_perm(va.x, vb.x, 0x7362),
                             __byte_perm(va.y, vb.y, 0x5140), __byte_perm(va.y, vb.y, 0x7362)};
    const uint32_t w23[4] = {__byte_perm(vc.x, vd.x, 0x5140), __byte_perm(vc.x, vd.x, 0x7362),
                             __byte_perm(vc.y, vd.y, 0x5140), __byte_perm(vc.y, vd.y, 0x7362)};
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      const uint32_t s01 = w01[jn >> 1], s23 = w23[jn >> 1];
      mma16816(o[jn], pa, e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01), e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23));
    }
  }
}

template <int NS>
__global__ void __launch_bounds__(NTHR, 2) ksv_kernel(const __grid_constant__ Params p) {
  using GG = G<NS>;
  constexpr int KMAX = GG::KMAX;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* Ks = smem;
  uint8_t* Vs = smem + GG::OFF_V;
  uint8_t* kn = smem + GG::OFF_KN;
  uint8_t* vn = smem + GG::OFF_VN;
  uint8_t* qs = smem + GG::OFF_Q;
  int* misc = reinterpret_cast<int*>(smem + GG::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, b = blockIdx.y;
  const bool tail = (s == NS - 1);                              // also writes and attends the block's new keys
  const int hw = warp & 1, tp = warp >> 1;                      // KV head, token pair (tokens 2 tp, 2 tp + 1)

  const int seq = __ldg(p.seq_lens + b);
  const int n_old = min(seq, WIN - 1);
  const int kstart = seq - n_old;
  const int base = s * KMAX;
  const bool any_old = base < n_old;
  if (any_old) {   // old K / V rows of this split, issued before the producer wait
    constexpr int PER = KMAX * 8 / NTHR;
    const int* pt = p.page_table + (size_t)b * p.pt_cols;
    int rows[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int key = (tid + NTHR * j) >> 3;
      const int gs = base + key;
      rows[j] = (gs < n_old && kstart + gs < p.pt_cols) ? __ldg(pt + kstart + gs) : -1;
    }
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int r = tid + NTHR * j, key = r >> 3, ch = r & 7;
      const bool ok = rows[j] >= 0;
      const size_t go = (size_t)(ok ? rows[j] : 0) * 128 + 16 * ch;
      cp_async16(Ks + key * 128 + 16 * kswz(key, ch), p.k_cache + go, ok ? 16 : 0);
      cp_async16(Vs + key * 128 + 16 * vswz(key, ch), p.v_cache + go, ok ? 16 : 0);
    }
    cp_commit();
  }
  if (tail) {
    reinterpret_cast<uint4*>(kn)[tid] = make_uint4(0u, 0u, 0u, 0u);    // kn | vn: 2 x 4 KB, rows >= 8 stay zero
    reinterpret_cast<uint4*>(kn)[tid + NTHR] = make_uint4(0u, 0u, 0u, 0u);
  }
  pdl_wait();
  __syncthreads();

  // q rows (token j = warp, heads 0..15) -> RMSNorm, RoPE, bf16-round, fp16 in smem (v2: all loads issued before any use)
  {
    const size_t tok = (size_t)b * L + warp;
    uint32_t raw[HQ];
#pragma unroll
    for (int h = 0; h < HQ; ++h) raw[h] = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + tok * p.stride_qkv + (size_t)h * 64) + lane);
    float2 cs = make_float2(1.f, 0.f);
    if (p.rope) {
      const long long pos = __ldg(p.positions + tok);
      cs = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
    }
#pragma unroll
    for (int h = 0; h < HQ; ++h) {
      const float2 a = norm_rope(raw[h], cs, p.rope, p.eps);
      reinterpret_cast<uint32_t*>(qs + (warp * 16 + h) * 128)[lane] = pack_h2(a.x, a.y);
    }
  }
  if (tail) {   // the block's 8 new keys / values: items (token, kind, head) = 32, warp w handles w, w + 8, w + 16, w + 24
    uint32_t raws[4];
    long long poss[4], locs[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {   // v2: all loads first
      const int it = warp + 8 * k, j = it >> 2, kind = (it >> 1) & 1, head = it & 1;
      const size_t tok = (size_t)b * L + j;
      raws[k] = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + tok * p.stride_qkv + (size_t)(HQ + kind * HKV + head) * 64) + lane);
      poss[k] = __ldg(p.positions + tok);
      locs[k] = __ldg(p.cache_loc + tok);
    }
    float2 css[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      css[k] = p.rope ? make_float2(__ldg(p.cos_sin + poss[k] * 64 + lane), __ldg(p.cos_sin + poss[k] * 64 + 32 + lane))
                      : make_float2(1.f, 0.f);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int it = warp + 8 * k, j = it >> 2, kind = (it >> 1) & 1, head = it & 1;
      const uint32_t raw = raws[k];
      float2 v = (kind == 0) ? norm_rope(raw, css[k], p.rope, p.eps) : bf2f(raw);
      v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
      v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
      const uint16_t q8 = (uint16_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)locs[k] * 128 + head * 64 + 2 * lane) = q8;
      const int bb = 64 * head + 2 * lane, ch = bb >> 4;
      uint8_t* tile = kind ? vn : kn;
      const int sw = kind ? vswz(j, ch) : kswz(j, ch);
      *reinterpret_cast<uint16_t*>(tile + j * 128 + 16 * sw + (bb & 15)) = q8;
    }
  }
  cp_wait_all();
  __syncthreads();

  // this warp's rows: g -> token 2 tp, g + 8 -> token 2 tp + 1, both q head 8 hw + g
  uint32_t qa[4][2], qb[4][2];
  {
    const uint32_t* ra = reinterpret_cast<const uint32_t*>(qs + ((2 * tp) * 16 + 8 * hw + g) * 128);
    const uint32_t* rb = reinterpret_cast<const uint32_t*>(qs + ((2 * tp + 1) * 16 + 8 * hw + g) * 128);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      qa[k][0] = ra[8 * t + 2 * k]; qa[k][1] = ra[8 * t + 2 * k + 1];
      qb[k][0] = rb[8 * t + 2 * k]; qb[k][1] = rb[8 * t + 2 * k + 1];
    }
  }
  const float cl2 = p.qk_sm * LOG2E;
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m[2] = {neg_inf(), neg_inf()}, l[2] = {0.f, 0.f};
  // token j sees old slot k iff k < n_old and kstart + k >= seq + j - (WIN - 1)
  const int lo0 = seq + 2 * tp - (WIN - 1) - kstart, lo1 = lo0 + 1;
  if (any_old) {
    const int nch = min(GG::NCH, (n_old - base + 31) / 32);
    for (int c = 0; c < nch; ++c)
      chunk32(Ks, Vs, 32 * c, hw, qa, qb, o, m, l, cl2, [&](int r, int k) {
        const int gk = base + k;
        return gk < n_old && gk >= (r ? lo1 : lo0);
      });
  }
  if (tail)   // new key i (position seq + i) is visible to token j iff i <= j
    chunk32(kn, vn, 0, hw, qa, qb, o, m, l, cl2, [&](int r, int k) { return k <= 2 * tp + r; });
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    l[r] += __shfl_xor_sync(FULL, l[r], 1);
    l[r] += __shfl_xor_sync(FULL, l[r], 2);
  }

  // stage o x 2^m and l x 2^m (absolute frame) in smem (the K | V region), then red.add into the request's accumulators
  __syncthreads();
  float* st_o = reinterpret_cast<float*>(smem);                 // [128][64]
  float* st_l = st_o + ROWS * 64;                               // [128]
  {
    const float f0 = (m[0] == neg_inf()) ? 0.f : ex2(fminf(m[0], 126.f));
    const float f1 = (m[1] == neg_inf()) ? 0.f : ex2(fminf(m[1], 126.f));
    const int R0 = (2 * tp) * 16 + 8 * hw + g, R1 = R0 + 16;
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int d = 8 * (2 * t + e) + jn;
        st_o[R0 * 64 + d] = o[jn][e] * f0;
        st_o[R1 * 64 + d] = o[jn][2 + e] * f1;
      }
    }
    if (t == 0) { st_l[R0] = l[0] * f0; st_l[R1] = l[1] * f1; }
  }
  __syncthreads();
  if (any_old || tail) {
    float* ao = p.acc_o + (size_t)b * ROWS * 64;
#pragma unroll
    for (int i = 0; i < ROWS * 16 / NTHR; ++i) {
      const int idx = tid + NTHR * i;                            // (row, float4 column)
      red4(ao + (size_t)idx * 4, reinterpret_cast<const float4*>(st_o)[idx]);
    }
    if (tid < ROWS) atomicAdd(p.acc_l + (size_t)b * ROWS + tid, st_l[tid]);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
  __syncthreads();
  if (misc[0] == 0) return;

  // last CTA of request b: out = RMSNorm(o / l) x sigmoid(gate), bf16; re-zero the accumulators
  constexpr int NI = ROWS * 16 / NTHR;
  float4 vv[NI];
  float dn[NI];
  uint2 gr[NI];
#pragma unroll
  for (int i = 0; i < NI; ++i) {   // v2: every load of the finalize issued before any use
    const int idx = tid + NTHR * i, row = idx >> 4, mu = idx & 15;
    const size_t tok = (size_t)b * L + (row >> 4);
    vv[i] = __ldcg(reinterpret_cast<const float4*>(p.acc_o + ((size_t)b * ROWS + row) * 64 + 4 * mu));
    dn[i] = (mu == 0) ? __ldcg(p.acc_l + (size_t)b * ROWS + row) : 0.f;
    gr[i] = __ldcg(reinterpret_cast<const uint2*>(p.qkv + tok * p.stride_qkv + (size_t)(HQ + 2 * HKV) * 64 + (size_t)(row & 15) * 64 + 4 * mu));
  }
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int idx = tid + NTHR * i, row = idx >> 4, mu = idx & 15, j = row >> 4, h = row & 15;
    __stcg(reinterpret_cast<float4*>(p.acc_o + ((size_t)b * ROWS + row) * 64 + 4 * mu), make_float4(0.f, 0.f, 0.f, 0.f));
    if (mu == 0) __stcg(p.acc_l + (size_t)b * ROWS + row, 0.f);
    const float4 v = vv[i];
    const float den = __shfl_sync(FULL, dn[i], lane & ~15);
    const size_t tok = (size_t)b * L + j;
    const uint2 graw = gr[i];
    const float dd = (den > 0.f) ? den : 1.f;
    const float y0 = bfr(v.x / dd), y1 = bfr(v.y / dd), y2 = bfr(v.z / dd), y3 = bfr(v.w / dd);
    float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    ss += __shfl_xor_sync(FULL, ss, 8);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 1);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    const float2 g01 = bf2f(graw.x), g23 = bf2f(graw.y);
    const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0 * rs * (1.f / (1.f + __expf(-g01.x))), y1 * rs * (1.f / (1.f + __expf(-g01.y))));
    const __nv_bfloat162 o23 = __floats2bfloat162_rn(y2 * rs * (1.f / (1.f + __expf(-g23.x))), y3 * rs * (1.f / (1.f + __expf(-g23.y))));
    *reinterpret_cast<uint2*>(p.out + tok * (HQ * 64) + (size_t)h * 64 + 4 * mu) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&o01), *reinterpret_cast<const uint32_t*>(&o23));
  }
  if (tid == 0) p.counters[b] = 0;
}

template <int NS>
static void launch(const Params& P, int B, cudaStream_t st) {
  constexpr int SM = G<NS>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ksv_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    attr = true;
  }
  ksv_kernel<NS><<<dim3((unsigned)NS, (unsigned)B, 1), NTHR, SM, st>>>(P);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NS>
static std::vector<int64_t> info() {
  constexpr int SM = G<NS>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ksv_kernel<NS>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ksv_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ksv_kernel<NS>, NTHR, SM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
}
}  // namespace ksv

void ksv_forward(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor acc_o, torch::Tensor acc_l, torch::Tensor counters, torch::Tensor out, int64_t NS,
                 double qk_sm, double eps, int64_t rope) {
  ksv::Params P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.seq_lens = seq_lens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.positions = positions.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  P.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  P.cos_sin = cos_sin.data_ptr<float>();
  P.acc_o = acc_o.data_ptr<float>();
  P.acc_l = acc_l.data_ptr<float>();
  P.counters = counters.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.stride_qkv = (int)qkv.stride(0);
  P.pt_cols = (int)page_table.size(1);
  P.rope = (int)rope;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  const int B = (int)seq_lens.size(0);
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (NS == 8) ksv::launch<8>(P, B, st);
  else if (NS == 4) ksv::launch<4>(P, B, st);
  else TORCH_CHECK(false, "ksv: NS must be 4 or 8");
}

std::vector<int64_t> ksv_info(int64_t NS) {
  return NS == 8 ? ksv::info<8>() : ksv::info<4>();
}

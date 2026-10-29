"""FX-S: one-launch verify-SLIDING attention for B = 16 / 32, q_len 8 (notes/study/fresh_eyes_oct29b.md s3 mechanism 1).

Grid (NS, B), 256 threads = 2 warpgroups (WG h = KV head h, 64 rows = 8 tokens x 8 q heads, row r = 8 j + hh).
- Before griddepcontrol.wait: page-table slice -> every K/V row of the CTA's tiles by 16-B cp.async into FP8 smem
  (128-B rows, 128B swizzle, whole share in flight, one mbarrier per 64-key tile), V^T ones rows, new-key tile zeroed.
- After the wait: q RMSNorm + RoPE -> e4m3 Q tiles (64B swizzle), the tail CTA writes the 8 new K/V rows to the cache and
  into its new-key tile (keys 0..7 of an extra tile; causal mask inside it).
- Per tile and WG: QK = 2 x wgmma m64n64k32 e4m3 SS (K head half of the 128-B row), V^T (key-permuted, + 8 ones rows) by
  the WG while QK runs, exact running max with offset 8 (P' = 2^(s - max + 8) <= 256, e4m3: today's FA4 PV8 numerics),
  PV = 2 x wgmma m64n72k32 RS (cols 64..71 = row sums of the e4m3 P).
- Partial (offset, l, O) by plain 16-B stores to a [B][NS][128][64] slab, one acq_rel ticket per (request, WG, warp):
  the last-arriving warp merges its 16 rows over the NS partials, o-RMSNorm x sigmoid(gate), bf16 out, counter reset.
MODE 0 = null consumer (gathers + prep + slab + ticket merge, no QK / softmax / PV): the step-0 skeleton.
"""

FXS_NS = (4, 8, 16)

FXS_CPP_SRC = r"""
#include <torch/extension.h>
void fxs_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, int64_t mode, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope);
std::vector<int64_t> fxs_info(int64_t NS, int64_t mode);
"""

FXS_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define FXS_SM90 1
#else
#define FXS_SM90 0
#endif

namespace fxs {

constexpr int HQ = 16, HKV = 2, L = 8, WIN = 2048;
constexpr int TILE = 64, ROWB = 128, TB = TILE * ROWB;     // 8 KB per K (or V) tile: 64 keys x 2 heads x 64 dims e4m3
constexpr int VT_BYTES = 72 * 64;                          // V^T per (head, buffer): 64 dims + 8 ones rows x 64 keys, 64B swizzle
constexpr int Q_BYTES = 64 * 64;                           // e4m3 Q per head: 64 rows (8 tokens x 8 heads) x 64 dims, 64B swizzle
constexpr int NTHR = 256;
constexpr int MAXT = (WIN - 1 + TILE - 1) / TILE + 1;      // 32 old tiles (2047 keys) + the new-key tile
constexpr float LOG2E = 1.4426950408889634f;
constexpr float POF = 8.f;                                 // P' = 2^(s - max + 8) <= 256 < 448
constexpr float FP8_MAX = 448.f;
constexpr unsigned FULL = 0xffffffffu;
constexpr uint64_t HI64 = ((uint64_t)(512 >> 4) | (2ull << 30)) << 32;    // SBO 512, 64B swizzle
constexpr uint64_t HI128 = ((uint64_t)(1024 >> 4) | (1ull << 30)) << 32;  // SBO 1024, 128B swizzle

template <int NS> struct G {
  static constexpr int CAP = (MAXT + NS - 1) / NS;         // tiles per CTA
  static constexpr int OFF_V = CAP * TB;
  static constexpr int OFF_VT = 2 * CAP * TB;
  static constexpr int OFF_Q = OFF_VT + 4 * VT_BYTES;
  static constexpr int OFF_BAR = OFF_Q + 2 * Q_BYTES;
  static constexpr int OFF_MISC = OFF_BAR + CAP * 8;
  static constexpr int SMEM = OFF_MISC + 64 + 1024;        // + 1 KB alignment slack
  static constexpr int MINB = (NS >= 8) ? 2 : 1;
};

struct Params {
  const __nv_bfloat16* qkv;   // [B * 8, stride] q | k | v | gate
  const int* cache_seqlens;   // [B] old length + 8
  const int64_t* cache_loc;   // [B * 8]
  const int64_t* positions;   // [B * 8]
  const int* page_table;      // [B, pt_cols] (page size 1)
  uint8_t* k_cache;           // e4m3 rows of HKV * 64 B
  uint8_t* v_cache;
  const float* cos_sin;       // [P, 64] cos | sin
  float* part_o;              // [B][NS][128][64] fp32 O (relative to the row offset)
  float2* part_ml;            // [B][NS][128] (offset, l) in the exp2 domain
  int* counters;              // [B * 8] tickets per (request, WG, warp), zero between calls (self-resetting)
  __nv_bfloat16* out;         // [B * 8, HQ * 64]
  long long* prof;            // lab: [CTAs][8] %globaltimer stamps, or nullptr
  int stride_qkv, pt_cols, B, trig, rope;
  float qk_sm, eps;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(FULL, v, d);
  return v;
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(FULL, v, 1));
  return fmaxf(v, __shfl_xor_sync(FULL, v, 2));
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float2 bf2f(uint32_t v) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v)); }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ uint32_t pk4(float a, float b, float c, float d) {   // 4 x e4m3 (satfinite), a in the low byte
  uint32_t r;
  asm("{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\ncvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\nmov.b32 %0, {lo, hi};\n}"
      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
__device__ __forceinline__ uint16_t e4m3x2(float2 v) {
  v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
  v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
  return (uint16_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float2 norm_rope(uint32_t raw, float2 cs, int rope, float eps) {   // = the FA4 prologue's rounding
  const float2 x = bf2f(raw);
  const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
  const float rs = rsqrtf(var + eps);
  const float e = bfr(x.x * rs), o = bfr(x.y * rs);
  if (!rope) return make_float2(e, o);
  return make_float2(bfr(__fmaf_rn(e, cs.x, -__fmul_rn(o, cs.y))), bfr(__fmaf_rn(e, cs.y, __fmul_rn(o, cs.x))));
}
__device__ __forceinline__ long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define FXS_RF(x) asm volatile("" : "+f"(x))
#define FXS_RR(x) asm volatile("" : "+r"(x))

#if FXS_SM90
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nFXSW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra FXSW_%=;\n}\n"
               ::"r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void sts32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory"); }
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ int atom_add_acqrel(int* a, int v) {
  int old;
  asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void stcg4(float* a, float x, float y, float z, float w) {
  asm volatile("st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
#define FXS_D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define FXS_R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define FXS_R36 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}"
__device__ __forceinline__ void wg_qk(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " FXS_R32 ", %32, %33, p, 1, 1;\n}\n"
               : FXS_D32(d) : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wg_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 " FXS_R36 ", {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : FXS_D32(d), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}
#endif

template <int NS, int MODE>
__global__ void __launch_bounds__(NTHR, G<NS>::MINB) fxs_kernel(const __grid_constant__ Params p) {
#if FXS_SM90
  using GG = G<NS>;
  constexpr int CAP = GG::CAP;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024u - (sa(smem_raw) & 1023u)) & 1023u);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + GG::OFF_BAR);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int s = blockIdx.x, b = blockIdx.y;
  const int cta = b * NS + s;
  long long* const prof = p.prof;
#define FXSP(k) do { if (prof != nullptr && tid == 0) prof[(size_t)cta * 8 + (k)] = gtimer(); } while (0)
  if (p.trig == 2) pdl_trigger();
  FXSP(0);
  const int seq_old = max(__ldg(p.cache_seqlens + b) - L, 0);
  const int n_old = min(seq_old, WIN - 1);
  const int kstart = seq_old - n_old;
  const int NT = (n_old + TILE - 1) / TILE + 1;
  const int t0 = (NT * s) / NS, t1 = (NT * (s + 1)) / NS;
  const int cnt = t1 - t0;                                     // local tiles (<= CAP)
  const bool tail = (t1 == NT);                                // owns the new-key tile = local slot cnt - 1
  const int nold = tail ? cnt - 1 : cnt;
  if (tid < CAP) mbar_init(full + tid, NTHR);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  {   // whole share in flight: page-table slice, then 4 x 16-B cp.async per thread and tile
    const int ch = tid & 7, r0 = tid >> 3;
    const int* pt = p.page_table + (size_t)b * p.pt_cols;
    int rows[2 * CAP];
#pragma unroll
    for (int i = 0; i < CAP; ++i)
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int gs = (t0 + i) * TILE + r0 + 32 * k;
        rows[2 * i + k] = (i < nold && gs < n_old && kstart + gs < p.pt_cols) ? __ldg(pt + kstart + gs) : -1;
      }
    const uint32_t kb0 = sa(smem) + r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
#pragma unroll
    for (int i = 0; i < CAP; ++i) {
      if (i < nold) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int rw = rows[2 * i + k];
          const bool ok = rw >= 0;
          const size_t go = (size_t)(ok ? rw : 0) * ROWB + 16 * ch;
          const uint32_t dk = kb0 + (uint32_t)(i * TB + k * 32 * ROWB);
          cp_async16(dk, p.k_cache + go, ok ? 16 : 0);
          cp_async16(dk + GG::OFF_V, p.v_cache + go, ok ? 16 : 0);
        }
        cp_async_mbar_arrive(full + i);
      }
    }
  }
  if (tid < 128)   // rows 64..71 of the four V^T buffers: e4m3 ones (0x38), every byte equal (no swizzle)
    *reinterpret_cast<uint4*>(smem + GG::OFF_VT + (tid >> 5) * VT_BYTES + 4096 + (tid & 31) * 16) =
        make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  if (tail) {   // new-key tile: zero K / V (rows 8..63 stay zero)
    uint8_t* nk = smem + (cnt - 1) * TB;
    for (int e = tid; e < TB / 16; e += NTHR) {
      reinterpret_cast<uint4*>(nk)[e] = make_uint4(0u, 0u, 0u, 0u);
      reinterpret_cast<uint4*>(nk + GG::OFF_V)[e] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  const int jt = warp;                                         // prep: warp = token
  float2 cs = make_float2(1.f, 0.f);
  if (p.rope) {
    const long long pos = __ldg(p.positions + b * L + jt);
    cs = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
  }
  const long long loc = tail ? __ldg(p.cache_loc + b * L + jt) : 0;
  FXSP(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  FXSP(2);
  {
    const __nv_bfloat16* xr = p.qkv + (size_t)(b * L + jt) * p.stride_qkv;
    uint32_t qr[HQ], nr[4];
#pragma unroll
    for (int hd = 0; hd < HQ; ++hd) qr[hd] = __ldcg(reinterpret_cast<const unsigned int*>(xr + hd * 64) + lane);
    if (tail) {
#pragma unroll
      for (int k = 0; k < 4; ++k) nr[k] = __ldcg(reinterpret_cast<const unsigned int*>(xr + (HQ + k) * 64) + lane);   // k0 k1 v0 v1
    }
    const int c = lane >> 3, bo = (2 * lane) & 15;
#pragma unroll
    for (int hd = 0; hd < HQ; ++hd) {
      const uint16_t q8 = e4m3x2(norm_rope(qr[hd], cs, p.rope, p.eps));
      const int h = hd >> 3, r = 8 * jt + (hd & 7);
      *reinterpret_cast<uint16_t*>(smem + GG::OFF_Q + h * Q_BYTES + r * 64 + ((c ^ ((r >> 1) & 3)) << 4) + bo) = q8;
    }
    if (tail) {
      uint8_t* nk = smem + (cnt - 1) * TB;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int kind = k >> 1, head = k & 1;
        const uint16_t e8 = e4m3x2(kind ? bf2f(nr[k]) : norm_rope(nr[k], cs, p.rope, p.eps));
        *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)loc * ROWB + head * 64 + 2 * lane) = e8;
        const int d = head * 64 + 2 * lane, cc = d >> 4;
        *reinterpret_cast<uint16_t*>(nk + (kind ? GG::OFF_V : 0) + jt * ROWB + ((cc ^ (jt & 7)) << 4) + (d & 15)) = e8;
      }
    }
  }
  fence_proxy_async();
  __syncthreads();
  FXSP(3);

  const int h = warp >> 2, w = warp & 3, g = lane >> 2, tig = lane & 3;
  float acc[36];
#pragma unroll
  for (int e = 0; e < 36; ++e) { acc[e] = 0.f; FXS_RF(acc[e]); }
  float cr0 = neg_inf(), cr1 = neg_inf();
  if constexpr (MODE == 1) {
    const float ssc = p.qk_sm * LOG2E;
    const uint32_t qlo = __shfl_sync(FULL, ((sa(smem) + GG::OFF_Q + h * Q_BYTES) >> 4) | 0x10000u, 0);
    const uint32_t klo = __shfl_sync(FULL, ((sa(smem) + h * 64) >> 4) | 0x10000u, 0);
    const uint32_t vtlo = __shfl_sync(FULL, ((sa(smem) + GG::OFF_VT + h * 2 * VT_BYTES) >> 4) | 0x10000u, 0);
    const int vkey = 16 * w + 8 * ((lane >> 3) & 1) + (lane & 7);
    const uint32_t vld0 = sa(smem) + GG::OFF_V + vkey * ROWB + (((4 * h + (lane >> 4)) ^ (lane & 7)) << 4);
    const uint32_t vst = sa(smem) + GG::OFF_VT + h * 2 * VT_BYTES + 128 * g + ((w ^ (g & 3)) << 4) + 4 * tig;
    const uint32_t vsel_a = (g >= 4) ? 0x7531u : 0x6420u, vsel_b = (g >= 4) ? 0x6420u : 0x7531u;
    const uint32_t vst_a = vst + ((g >= 4) ? 64u : 0u), vst_b = vst + ((g >= 4) ? 0u : 64u);
    const int j0 = 2 * w, j1 = 2 * w + 1;
    const int lo0 = n_old - (WIN - 1) + j0, lo1 = lo0 + 1;     // old key gs is in token j's window iff gs >= n_old - 2047 + j
    for (int i = 0; i < cnt; ++i) {
      const int buf = i & 1;
      const bool isnew = tail && (i == cnt - 1);
      if (!isnew) mbar_wait(full + i, 0u);
      fence_proxy_async();
      float sc[32];
      const uint32_t kl = klo + (uint32_t)(i * (TB >> 4));
      wg_fence();
      wg_qk(sc, HI64 | qlo, HI128 | kl, 0);
      wg_qk(sc, HI64 | (qlo + 2u), HI128 | (kl + 2u), 1);
      wg_commit();
      {   // this tile's V (head h) -> key-permuted V^T (buffer buf) while QK runs
        const uint32_t vsrc = vld0 + (uint32_t)(i * TB);
        const uint32_t da = vst_a + (uint32_t)(buf * VT_BYTES), db = vst_b + (uint32_t)(buf * VT_BYTES);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          uint32_t r[4];
          ldsm_x4_t(r, u ? (vsrc ^ 32u) : vsrc);
          sts32(da + 1024 * (2 * u), prmt(r[0], r[1], vsel_a));
          sts32(db + 1024 * (2 * u), prmt(r[0], r[1], vsel_b));
          sts32(da + 1024 * (2 * u + 1), prmt(r[2], r[3], vsel_a));
          sts32(db + 1024 * (2 * u + 1), prmt(r[2], r[3], vsel_b));
        }
      }
      wg_wait0();
#pragma unroll
      for (int e = 0; e < 32; ++e) FXS_RF(sc[e]);
      const int tg = t0 + i;
      if (isnew) {                                             // new keys 0..7: key c visible to token j iff c <= j
#pragma unroll
        for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int cc = 8 * nt8 + 2 * tig + e;
            if (cc > j0) sc[4 * nt8 + e] = neg_inf();
            if (cc > j1) sc[4 * nt8 + 2 + e] = neg_inf();
          }
      } else if (tg * TILE < lo1 || (tg + 1) * TILE > n_old) { // window start / last old tile
#pragma unroll
        for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int gs = tg * TILE + 8 * nt8 + 2 * tig + e;
            if (gs >= n_old || gs < lo0) sc[4 * nt8 + e] = neg_inf();
            if (gs >= n_old || gs < lo1) sc[4 * nt8 + 2 + e] = neg_inf();
          }
      }
      float m0 = neg_inf(), m1 = neg_inf();
#pragma unroll
      for (int nt8 = 0; nt8 < 8; ++nt8) {
        m0 = fmaxf(m0, fmaxf(sc[4 * nt8], sc[4 * nt8 + 1]));
        m1 = fmaxf(m1, fmaxf(sc[4 * nt8 + 2], sc[4 * nt8 + 3]));
      }
      m0 = quad_max(m0) * ssc;
      m1 = quad_max(m1) * ssc;
      const float n0 = fmaxf(cr0, m0 - POF), n1 = fmaxf(cr1, m1 - POF);
      const float a0 = (n0 == neg_inf()) ? 1.f : ex2(cr0 - n0), a1 = (n1 == neg_inf()) ? 1.f : ex2(cr1 - n1);
#pragma unroll
      for (int t = 0; t < 9; ++t) { acc[4 * t] *= a0; acc[4 * t + 1] *= a0; acc[4 * t + 2] *= a1; acc[4 * t + 3] *= a1; }
      cr0 = n0; cr1 = n1;
      const float u0 = (n0 == neg_inf()) ? 0.f : n0, u1 = (n1 == neg_inf()) ? 0.f : n1;
      uint32_t pa[2][4];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {   // P = e4m3(2^(s ssc - u)) of keys 32 kk .. +31, in the PV A-fragment (key-permuted) order
        const float* sv = sc + 16 * kk;
        pa[kk][0] = pk4(ex2(fmaf(sv[0], ssc, -u0)), ex2(fmaf(sv[1], ssc, -u0)), ex2(fmaf(sv[4], ssc, -u0)), ex2(fmaf(sv[5], ssc, -u0)));
        pa[kk][1] = pk4(ex2(fmaf(sv[2], ssc, -u1)), ex2(fmaf(sv[3], ssc, -u1)), ex2(fmaf(sv[6], ssc, -u1)), ex2(fmaf(sv[7], ssc, -u1)));
        pa[kk][2] = pk4(ex2(fmaf(sv[8], ssc, -u0)), ex2(fmaf(sv[9], ssc, -u0)), ex2(fmaf(sv[12], ssc, -u0)), ex2(fmaf(sv[13], ssc, -u0)));
        pa[kk][3] = pk4(ex2(fmaf(sv[10], ssc, -u1)), ex2(fmaf(sv[11], ssc, -u1)), ex2(fmaf(sv[14], ssc, -u1)), ex2(fmaf(sv[15], ssc, -u1)));
      }
#pragma unroll
      for (int e = 0; e < 36; ++e) FXS_RF(acc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) FXS_RR(pa[e >> 2][e & 3]);
      fence_proxy_async();                                     // V^T (st.shared) -> async proxy
      bar_named(1 + h, 128);                                   // all 4 warps' V^T rows written
      const uint32_t vl = vtlo + (uint32_t)(buf * (VT_BYTES >> 4));
      wg_fence();
      wg_pv(acc, pa[0], HI64 | vl);
      wg_pv(acc, pa[1], HI64 | (vl + 2u));
      wg_commit();
      wg_wait0();
#pragma unroll
      for (int e = 0; e < 36; ++e) FXS_RF(acc[e]);
    }
  } else {   // null consumer: every tile landed, one smem byte per thread kept live
    for (int i = 0; i < cnt; ++i)
      if (!(tail && i == cnt - 1)) mbar_wait(full + i, 0u);
    if (cnt > 0) acc[0] = (float)smem[tid] * 0.f;
    cr0 = 0.f; cr1 = 0.f;
    acc[32] = 1.f; acc[34] = 1.f;
  }
  if (tid == 0) FXSP(4);
  {   // partial: plain 16-B stores (lanes tig, tig ^ 1 swap so each holds 4 consecutive dims)
    const int R0 = (2 * w) * 16 + 8 * h + g, R1 = R0 + 16;
    const size_t pb = ((size_t)b * NS + s) * 128;
    float* a0p = p.part_o + (pb + R0) * 64;
    float* a1p = p.part_o + (pb + R1) * 64;
    const bool odd = tig & 1;
#pragma unroll
    for (int t = 0; t < 8; t += 2) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float x0 = acc[4 * t + 2 * r], x1 = acc[4 * t + 2 * r + 1];
        const float y0 = acc[4 * (t + 1) + 2 * r], y1 = acc[4 * (t + 1) + 2 * r + 1];
        const float s0 = odd ? x0 : y0, s1 = odd ? x1 : y1;
        const float q0 = __shfl_xor_sync(FULL, s0, 1), q1 = __shfl_xor_sync(FULL, s1, 1);
        float* ap = (r ? a1p : a0p) + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1);
        stcg4(ap, odd ? q0 : x0, odd ? q1 : x1, odd ? y0 : q0, odd ? y1 : q1);
      }
    }
    if (tig == 0) {
      __stcg(p.part_ml + pb + R0, make_float2(cr0, acc[32]));
      __stcg(p.part_ml + pb + R1, make_float2(cr1, acc[34]));
    }
  }
  __syncwarp();
  int last = 0;
  if (lane == 0) last = (atom_add_acqrel(p.counters + b * 8 + warp, 1) == NS - 1) ? 1 : 0;
  last = __shfl_sync(FULL, last, 0);
  FXSP(5);
  if (!last) return;
  {   // last arriver of (b, h, w): merge its 16 rows (2 tokens x 8 heads) over the NS partials
    const int rr = lane >> 1, hf = lane & 1;
    const int j = 2 * w + (rr >> 3), qh = 8 * h + (rr & 7);
    const int R = j * 16 + qh;
    const uint4* gp = reinterpret_cast<const uint4*>(p.qkv + (size_t)(b * L + j) * p.stride_qkv + (HQ + 2 * HKV) * 64 + qh * 64 + 32 * hf);
    uint4 gr[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) gr[e] = __ldcg(gp + e);
    float2 ml[NS];
#pragma unroll
    for (int q = 0; q < NS; ++q) ml[q] = __ldcg(p.part_ml + ((size_t)b * NS + q) * 128 + R);
    float M = neg_inf();
#pragma unroll
    for (int q = 0; q < NS; ++q) M = fmaxf(M, ml[q].x);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float4 a[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) a[e] = make_float4(0.f, 0.f, 0.f, 0.f);
    float den = 0.f;
    constexpr int QB = (NS >= 16) ? 1 : 2;
#pragma unroll
    for (int q0 = 0; q0 < NS; q0 += QB) {
      float4 v[QB][8];
#pragma unroll
      for (int qq = 0; qq < QB; ++qq)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          v[qq][e] = __ldcg(reinterpret_cast<const float4*>(p.part_o + (((size_t)b * NS + q0 + qq) * 128 + R) * 64 + 32 * hf) + e);
#pragma unroll
      for (int qq = 0; qq < QB; ++qq) {
        const float wq = (ml[q0 + qq].x == neg_inf()) ? 0.f : ex2(ml[q0 + qq].x - Mu);
        den = fmaf(wq, ml[q0 + qq].y, den);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          a[e].x = fmaf(wq, v[qq][e].x, a[e].x); a[e].y = fmaf(wq, v[qq][e].y, a[e].y);
          a[e].z = fmaf(wq, v[qq][e].z, a[e].z); a[e].w = fmaf(wq, v[qq][e].w, a[e].w);
        }
      }
    }
    const float inv = (den > 0.f) ? 1.f / den : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      a[e].x = bfr(a[e].x * inv); a[e].y = bfr(a[e].y * inv); a[e].z = bfr(a[e].z * inv); a[e].w = bfr(a[e].w * inv);
      ss += a[e].x * a[e].x + a[e].y * a[e].y + a[e].z * a[e].z + a[e].w * a[e].w;
    }
    ss += __shfl_xor_sync(FULL, ss, 1);
    const float rn = rsqrtf(ss * (1.f / 64.f) + p.eps);
    uint4* op = reinterpret_cast<uint4*>(p.out + (size_t)(b * L + j) * (HQ * 64) + qh * 64 + 32 * hf);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint32_t gw[4] = {gr[e].x, gr[e].y, gr[e].z, gr[e].w};
      const float yv[8] = {a[2 * e].x, a[2 * e].y, a[2 * e].z, a[2 * e].w, a[2 * e + 1].x, a[2 * e + 1].y, a[2 * e + 1].z, a[2 * e + 1].w};
      uint32_t ow[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 gg = bf2f(gw[k]);
        const __nv_bfloat162 o2 = __floats2bfloat162_rn(yv[2 * k] * rn * (1.f / (1.f + __expf(-gg.x))),
                                                        yv[2 * k + 1] * rn * (1.f / (1.f + __expf(-gg.y))));
        ow[k] = *reinterpret_cast<const uint32_t*>(&o2);
      }
      op[e] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
    }
    if (lane == 0) p.counters[b * 8 + warp] = 0;               // self-reset (all NS splits of (b, h, w) arrived)
    if (prof != nullptr && lane == 0) atomicMax(reinterpret_cast<unsigned long long*>(prof + (size_t)(b * NS) * 8 + 6),
                                                (unsigned long long)gtimer());
    if (prof != nullptr && lane == 0) atomicMax(reinterpret_cast<unsigned long long*>(prof + (size_t)(b * NS) * 8 + 7),
                                                (unsigned long long)gtimer());
  }
#undef FXSP
#endif
}

template <int NS, int MODE>
static void launch(const Params& P, bool pdl, cudaStream_t st) {
  constexpr int SM = G<NS>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fxs_kernel<NS, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fxs_kernel<NS, MODE>, P));
}

template <int NS, int MODE>
static std::vector<int64_t> info() {
  constexpr int SM = G<NS>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, fxs_kernel<NS, MODE>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(fxs_kernel<NS, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fxs_kernel<NS, MODE>, NTHR, SM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
}

}  // namespace fxs

void fxs_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, int64_t mode, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {
  const at::cuda::CUDAGuard guard(qkv.device());
  fxs::Params P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.cache_seqlens = cache_seqlens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.positions = positions.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = k_cache.data_ptr<uint8_t>();
  P.v_cache = v_cache.data_ptr<uint8_t>();
  P.cos_sin = cos_sin.data_ptr<float>();
  P.part_o = part_o.data_ptr<float>();
  P.part_ml = reinterpret_cast<float2*>(part_ml.data_ptr<float>());
  P.counters = counters.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.prof = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  P.stride_qkv = (int)qkv.stride(0);
  P.pt_cols = (int)page_table.size(1);
  P.B = (int)cache_seqlens.numel();
  P.trig = (int)trig;
  P.rope = (int)rope;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  TORCH_CHECK(P.B * 8 <= counters.numel(), "fxs: counters too small");
  TORCH_CHECK(part_o.numel() >= (int64_t)P.B * NS * 128 * 64 && part_ml.numel() >= (int64_t)P.B * NS * 128 * 2, "fxs: slab too small");
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const bool pd = pdl != 0;
#define FXS_GO(n, m) if (NS == n && mode == m) { fxs::launch<n, m>(P, pd, st); return; }
  FXS_GO(4, 0) FXS_GO(4, 1) FXS_GO(8, 0) FXS_GO(8, 1) FXS_GO(16, 0) FXS_GO(16, 1)
#undef FXS_GO
  TORCH_CHECK(false, "fxs: unsupported NS / mode");
}

std::vector<int64_t> fxs_info(int64_t NS, int64_t mode) {
#define FXS_GI(n, m) if (NS == n && mode == m) return fxs::info<n, m>();
  FXS_GI(4, 0) FXS_GI(4, 1) FXS_GI(8, 0) FXS_GI(8, 1) FXS_GI(16, 0) FXS_GI(16, 1)
#undef FXS_GI
  TORCH_CHECK(false, "fxs: unsupported NS / mode");
  return {};
}
"""


def build(arch="90a", name=None):
    import hashlib
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((FXS_CPP_SRC + FXS_CUDA_SRC + arch).encode()).hexdigest()[:10]
    return load_inline(name=name or f"fxs_{tag}", cpp_sources=FXS_CPP_SRC, cuda_sources=FXS_CUDA_SRC,
                       functions=["fxs_forward", "fxs_info"],
                       extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)


def call(mod, data, cos_sin, ns, mode=1, counters=None, prof=None, ws=None, trig=1, pdl=True, eps=1e-5):
    import torch
    cfg = data["config"]
    B = cfg["batch"]
    qkv = data["qkv"].view(B * 8, -1)
    dev = qkv.device
    if ws is None:
        po = torch.empty((B * ns * 128 * 64,), dtype=torch.float32, device=dev)
        pml = torch.empty((B * ns * 128 * 2,), dtype=torch.float32, device=dev)
    else:
        po, pml = ws
    if counters is None:
        counters = torch.zeros(1024, dtype=torch.int32, device=dev)
    out = torch.empty((B * 8, 16 * 64), dtype=torch.bfloat16, device=dev)
    pr = prof if prof is not None else torch.empty((0,), dtype=torch.int64, device=dev)
    mod.fxs_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                    data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml, counters, out, pr,
                    int(ns), int(mode), float(cfg["sm_scale"] * cfg["qk_scale_factor"]), float(eps), int(trig),
                    1 if pdl else 0, 1 if cfg["rope_theta"] else 0)
    return out

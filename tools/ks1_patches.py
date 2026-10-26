"""ks1_patches.py: KS1 = ONE-launch sliding-window attention for the decode layer's B = 1 sliding shapes (decode 1: T = q_len = 1;
verify 1: T = q_len = 8), replacing FA4 v8's three PDL-chained launches there (Triton prologue -> CuTe FA4 core -> Triton combine
+ norm-gate). Exact text patches against kernels/muse_decode_layer/k3_v29.py (style of tools/k2a_v47_patches.py: every `old`
occurs exactly once inside its scope).

KS1 kernel (one launch, PDL secondary of the qkv projection; grid = (NS + 1 splits, 2 KV heads, NTG token groups), 256 threads;
splits 0..NS-1 hold 2048 / NS old-key slots of the window each, split NS is the new-key CTA):
  * pre-wait (kernel inputs only, overlapping the qkv projection): cache_seqlens -> page-table rows of this CTA's key slots ->
    the KV head's 64-B K and V rows of every old key in the window (registers), plus the RoPE (cos, sin) entries of the block's
    positions and the new tokens' cache_loc. Nothing written by the predecessor is read before griddepcontrol.wait.
  * post-wait: qkv rows (q of this CTA's rows, the new k / v on the new-key CTA, the gate of the merge rows) from L2; old K / V
    -> fp16 smem tiles (e4m3 -> f16 exact, XOR-swizzled 16-B chunks); QK-RMSNorm + RoPE of q (SGLang FusedQKNorm arithmetic:
    bf16 after the norm and after RoPE; q kept at that bf16 value, finer than E4M3); the new-key CTA computes the new K
    (norm + RoPE) / V, clamps, rounds to E4M3, writes the cache rows (token group 0 only; bytes equal to r1's) and attends them
    from smem (nothing re-reads the rows it wrote).
  * per warp: m16n8k16 FP16 mma QK over its 16 (NS 16) / 32 (NS 8) key slots, masked (window, causal, length), one-pass softmax
    in the exp2 domain, P in fp16, PV mma with ldmatrix.trans V fragments (the new-key CTA: warp 0 over the 16-slot new tile).
  * in-CTA merge of the 8 warps (smem) -> one partial (fp32 O + max + sum, or normalized fp16 O: KS1_PHALF) per (group, split);
    ticket on a self-resetting per-group counter (KS1_SYNC 1: bar.sync + one atom.acq_rel.gpu, no fence.sc); the last-arriving
    CTA merges the NS + 1 partials in split order (deterministic), rounds to bf16, RMS-norms each head row, multiplies by
    sigmoid(gate) (FA4 combine arithmetic) and writes the output; launch_dependents right after the wait (KS1_TRIG 1) lets the
    o projection prefetch its weights during KS1.
  * optional per-CTA %globaltimer stamps (lab: KS1_PROF_BUF; kernels/muse_attention_decode/ks1_lab.py).

Layer glue: KS1_ENABLE / KS1_ROUTE / KS1_GEOM / KS1_TRIG / KS1_SYNC / KS1_PHALF / KS1_PDL are read at call time (layer_ab2.py
"set" arms); prepare builds KS1 (own load_inline, KS1_INSTS), self-tests every built geometry against the FP32 reference and the
r1 (stock-kernel) path on synthetic B = 1 sliding cases (error <= 1.25 x r1's + 5e-3 with both partial formats, new cache bytes
equal to r1's, tickets back at zero, a call with the other trigger / fence.sc protocol bit-identical, graph replay identical) and
sets state["ks1"]; any failure keeps today's FA4 v8 route. A host exception at call time disables KS1 (_k3_disable).
KS1_P1_CARVEOUT (prepare, default off) sets P1's preferred smem carveout so KS1 CTAs co-reside with the qkv GEMV at T = 1: it
does (KS1 entry -13.6 us before the wait returns, post-wait chain 3.7 us) but costs ~3 us elsewhere in the layer -> off.

Measured (box, layer_ab2 4 seeds x 4 pairs, shapes 0 / 6): decode 1 sliding 87.7 -> 83.5 us, verify 1 sliding 136.4 -> 130.5 us.

    python -I tools/ks1_patches.py [k3_v29.py] [out.py]   # asserts every `old` occurs exactly once in its scope
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v29.py"

KS1_CPP_SRC = r"""
#include <torch/extension.h>
void ks1_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out,
                 torch::Tensor prof, int64_t L, int64_t NS, int64_t NTG, double qk_sm, double eps, int64_t trig,
                 int64_t pdl, int64_t sync, int64_t phalf);
std::vector<int64_t> ks1_info(int64_t L, int64_t NS, int64_t NTG);
"""

KS1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define KS1_ARCH_OK 1
#else
#define KS1_ARCH_OK 0
#endif

namespace ks1 {

constexpr int HQ = 16, HKV = 2, WIN = 2048;    // tp8 shard of Muse Spark: 16 q heads, 2 KV heads, head dim 64, window 2048
constexpr int NWARP = 8, NTHR = NWARP * 32;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float FP8_MAX = 448.0f;
constexpr unsigned FULL = 0xffffffffu;

struct Params {
  const __nv_bfloat16* qkv;   // [L, stride] q | k | v | gate
  const int* cache_seqlens;   // [1] old length + L
  const int64_t* cache_loc;   // [L]
  const int64_t* positions;   // [L]
  const int* page_table;      // [1, pt_cols]
  uint8_t* k_cache;           // e4m3 rows of HKV * 64 B (page size 1)
  uint8_t* v_cache;
  const float* cos_sin;       // [P, 64] cos | sin
  float* part_o;              // [2 NTG][NS][ROWS][64]
  float2* part_ml;            // [2 NTG][NS][ROWS] (max, sum) in the exp2 domain
  int* counters;              // [2 NTG] arrival tickets, zero between calls (self-resetting)
  __nv_bfloat16* out;         // [L, HQ * 64]
  long long* prof;            // lab: [CTAs][8] %globaltimer stamps, or nullptr
  int stride_qkv;             // elements
  int pt_cols;
  int trig;                   // 0: implicit trigger at exit; 1: launch_dependents right after the wait; 2: at entry
  int phalf;                  // 1: partials as normalized fp16 O (else unnormalized fp32)
  int sync;                   // ticket protocol: 0 every thread fence.sc + relaxed atomic (+ fence.sc in the merger);
                              // 1 bar.sync + one atom.acq_rel.gpu; 2 bar.sync + fence.acq_rel + relaxed atomic (+ fence.acq_rel)
  float qk_sm;                // sm_scale * qk_scale_factor
  float eps;
};

template <int L, int NS, int NTG> struct Geo {
  static constexpr int KPS = WIN / NS;                        // old-key slots per split (NS * KPS = 2048 >= WIN - 1)
  static constexpr int TPG = L / NTG;                         // tokens per group (rows = 8 heads x TPG)
  static constexpr int ROWS = 8 * TPG;
  static constexpr int KPW = KPS / NWARP;                     // key slots per warp
  static constexpr int NT = KPW / 8;                          // n-tiles of 8 keys per warp
  static constexpr int TPK = (NTHR >= KPS) ? NTHR / KPS : 1;  // loader threads per key slot
  static constexpr int KPT = (KPS > NTHR) ? KPS / NTHR : 1;   // key slots per loader thread
  static constexpr int CPT = 4 / TPK;                         // 16-B chunks of the 64-B K (and V) head row per thread and slot
  static constexpr int OFF_VH = KPS * 128;
  static constexpr int OFF_KN = 2 * KPS * 128;
  static constexpr int OFF_VN = OFF_KN + 2048;
  static constexpr int OFF_Q = OFF_VN + 2048;
  static constexpr int OFF_ML = OFF_Q + 2048;
  static constexpr int OFF_MISC = OFF_ML + 2 * NWARP * 16 * 4;
  static constexpr int SMEM = OFF_MISC + 128;
  static_assert(KPW % 16 == 0 && TPK <= 4 && (4 % TPK) == 0, "key geometry");
  static_assert(NWARP * 16 * 64 * 4 <= 2 * KPS * 128, "merge scratch aliases the K / V tiles");
  static_assert(L % NTG == 0 && (TPG == 1 || TPG == 2), "token groups of 1 or 2 tokens");
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
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t v) {   // low 16 bits = two e4m3 -> f16x2 (exact)
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xFFFFu), __NV_E4M3);
  return (uint32_t)r.x | ((uint32_t)r.y << 16);
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 bf2f(uint32_t v) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if KS1_ARCH_OK
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const uint8_t* p) {
#if KS1_ARCH_OK
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
#endif
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const uint8_t* p) {
#if KS1_ARCH_OK
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
#endif
}
__device__ __forceinline__ uint4 ldg_nc16(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void pdl_wait() {
#if KS1_ARCH_OK
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if KS1_ARCH_OK
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
__device__ __forceinline__ int atom_add_acqrel(int* a, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void fence_acqrel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define KS1P(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[cta * 16 + (k)] = gtime(); } while (0)

// 8-dim fp16 chunk ch of key / row x (128-B rows of 64 fp16) sits at physical chunk ch ^ (x & 7): conflict-free ldmatrix
__device__ __forceinline__ int swz(int x, int ch) { return x * 128 + 16 * (ch ^ (x & 7)); }

// One online-softmax segment of NT n-tiles (8 keys each) starting at key k0 of the fp16 tiles kb / vb.
// valid(k, j): key k of the tiles visible to the rows of token slot j (j = 0: rows g, j = 1: rows g + 8).
template <int NT, typename F>
__device__ __forceinline__ void segment(const uint8_t* kb, const uint8_t* vb, const int k0, const uint32_t (&qa)[4][4],
                                        float (&o)[8][4], float (&m)[2], float (&ls)[2], const float cl2, F valid) {
  const int lane = threadIdx.x & 31, t = lane & 3;
  float sc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    sc[j][0] = 0.f; sc[j][1] = 0.f; sc[j][2] = 0.f; sc[j][3] = 0.f;
    const int key = k0 + 8 * j + (lane & 7);
    uint32_t b[4], c[4];
    ldsm_x4(b, kb + swz(key, lane >> 3));
    ldsm_x4(c, kb + swz(key, 4 + (lane >> 3)));
    mma16816(sc[j], qa[0], b[0], b[1]);
    mma16816(sc[j], qa[1], b[2], b[3]);
    mma16816(sc[j], qa[2], c[0], c[1]);
    mma16816(sc[j], qa[3], c[2], c[3]);
  }
  float x0 = neg_inf(), x1 = neg_inf();
#pragma unroll
  for (int j = 0; j < NT; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int k = k0 + 8 * j + 2 * t + e;
      sc[j][e] = valid(k, 0) ? sc[j][e] * cl2 : neg_inf();
      sc[j][2 + e] = valid(k, 1) ? sc[j][2 + e] * cl2 : neg_inf();
      x0 = fmaxf(x0, sc[j][e]);
      x1 = fmaxf(x1, sc[j][2 + e]);
    }
  }
  x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, 1));
  x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, 2));
  x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, 1));
  x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, 2));
  const float n0 = fmaxf(m[0], x0), n1 = fmaxf(m[1], x1);
  const float u0 = (n0 == neg_inf()) ? 0.f : n0, u1 = (n1 == neg_inf()) ? 0.f : n1;
  const float a0 = ex2(m[0] - u0), a1 = ex2(m[1] - u1);
  m[0] = n0; m[1] = n1;
  ls[0] *= a0; ls[1] *= a1;
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk) {
    uint32_t pa[4];
    {
      const float p00 = ex2(sc[2 * kk][0] - u0), p01 = ex2(sc[2 * kk][1] - u0);
      const float p02 = ex2(sc[2 * kk][2] - u1), p03 = ex2(sc[2 * kk][3] - u1);
      const float p10 = ex2(sc[2 * kk + 1][0] - u0), p11 = ex2(sc[2 * kk + 1][1] - u0);
      const float p12 = ex2(sc[2 * kk + 1][2] - u1), p13 = ex2(sc[2 * kk + 1][3] - u1);
      ls[0] += (p00 + p01) + (p10 + p11);
      ls[1] += (p02 + p03) + (p12 + p13);
      pa[0] = pack_h2(p00, p01); pa[1] = pack_h2(p02, p03); pa[2] = pack_h2(p10, p11); pa[3] = pack_h2(p12, p13);
    }
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      const int key = k0 + 16 * kk + 8 * ((lane >> 3) & 1) + (lane & 7);
      uint32_t b[4];
      ldsm_x4_t(b, vb + swz(key, 2 * np + (lane >> 4)));
      mma16816(o[2 * np], pa, b[0], b[1]);
      mma16816(o[2 * np + 1], pa, b[2], b[3]);
    }
  }
}

template <int L, int NS, int NTG>
__global__ void __launch_bounds__(NTHR, 2) ks1_kernel(const __grid_constant__ Params p) {
#if KS1_ARCH_OK
  using G = Geo<L, NS, NTG>;
  constexpr int TPG = G::TPG, ROWS = G::ROWS, KPS = G::KPS, KPW = G::KPW;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* kh = smem;
  uint8_t* vh = smem + G::OFF_VH;
  uint8_t* kn = smem + G::OFF_KN;
  uint8_t* vn = smem + G::OFF_VN;
  uint8_t* qs = smem + G::OFF_Q;
  float* mw = reinterpret_cast<float*>(smem + G::OFF_ML);
  float* lw = mw + NWARP * 16;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, h = blockIdx.y, tg = blockIdx.z;
  const int grp = h * NTG + tg;
  const int cta = (tg * HKV + h) * (NS + 1) + s;
  const bool tail = (s == NS);                                  // the new-key CTA (no old keys)
  if (p.trig == 2) pdl_trigger();
  KS1P(0);

  // ---------------- pre-wait: kernel inputs only (nothing the predecessor writes)
  const int seq = max(__ldg(p.cache_seqlens) - L, 0);         // cached length before this block
  const int n_old = min(seq, WIN - 1);                          // old keys in the window of the block's first token
  const int kstart = seq - n_old;
  const int part = tid % G::TPK;
  uint4 kr[G::KPT][G::CPT], vr[G::KPT][G::CPT];
  {
    int prow[G::KPT];
#pragma unroll
    for (int kk = 0; kk < G::KPT; ++kk) {
      const int slot = tid / G::TPK + kk * (NTHR / G::TPK);
      const int gs = s * KPS + slot;
      prow[kk] = (!tail && gs < n_old && kstart + gs < p.pt_cols) ? __ldg(p.page_table + kstart + gs) : -1;
    }
#pragma unroll
    for (int kk = 0; kk < G::KPT; ++kk) {
      const bool ok = prow[kk] >= 0;
      const uint8_t* kp = p.k_cache + (size_t)(ok ? prow[kk] : 0) * (HKV * 64) + h * 64 + 16 * part * G::CPT;
      const uint8_t* vp = p.v_cache + (size_t)(ok ? prow[kk] : 0) * (HKV * 64) + h * 64 + 16 * part * G::CPT;
#pragma unroll
      for (int c = 0; c < G::CPT; ++c) {
        kr[kk][c] = ok ? ldg_nc16(kp + 16 * c) : make_uint4(0u, 0u, 0u, 0u);
        vr[kk][c] = ok ? ldg_nc16(vp + 16 * c) : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
  // RoPE (cos, sin) of this lane's pair for the tokens of its q rows (j < TPG) and, on the new-key CTA, of token `warp`
  float2 csq[TPG];
#pragma unroll
  for (int j = 0; j < TPG; ++j) {
    const long long pos = __ldg(p.positions + tg * TPG + j);
    csq[j] = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
  }
  float2 csk = make_float2(1.f, 0.f);
  long long loc = 0;
  if (tail && warp < L) {
    const long long pos = __ldg(p.positions + warp);
    csk = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
    loc = __ldg(p.cache_loc + warp);
  }
  KS1P(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  KS1P(2);

  // ---------------- post-wait: the projection output (L2), then the smem tiles
  const size_t sq = (size_t)p.stride_qkv;
  uint32_t qraw[TPG];
#pragma unroll
  for (int j = 0; j < TPG; ++j)
    qraw[j] = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + (size_t)(tg * TPG + j) * sq + (size_t)(h * 8 + warp) * 64) + lane);
  uint32_t kraw = 0u, vraw = 0u;
  if (tail && warp < L) {
    kraw = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + (size_t)warp * sq + (size_t)(HQ + h) * 64) + lane);
    vraw = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + (size_t)warp * sq + (size_t)(HQ + HKV + h) * 64) + lane);
  }
  constexpr int DPT = (ROWS == 8) ? 2 : 4;                      // merge item (one per thread): row mr, dims DPT mu .. + DPT - 1
  constexpr int UPR = 64 / DPT;                                 // threads per row
  static_assert(ROWS * UPR == NTHR, "one merge item per thread");
  const int mr = tid / UPR, mu = tid % UPR;
  uint32_t graw[DPT / 2];
  {
    const __nv_bfloat16* gp = p.qkv + (size_t)(tg * TPG + (mr >> 3)) * sq + (size_t)(HQ + 2 * HKV) * 64 +
                              (size_t)(h * 8 + (mr & 7)) * 64 + DPT * mu;
    if constexpr (DPT == 4) {
      const uint2 v = __ldcg(reinterpret_cast<const uint2*>(gp));
      graw[0] = v.x;
      graw[DPT / 2 - 1] = v.y;
    } else {
      graw[0] = __ldcg(reinterpret_cast<const unsigned int*>(gp));
    }
  }
  // old K / V rows -> fp16 tiles
#pragma unroll
  for (int kk = 0; kk < (tail ? 0 : G::KPT); ++kk) {
    const int slot = tid / G::TPK + kk * (NTHR / G::TPK);
#pragma unroll
    for (int c = 0; c < G::CPT; ++c) {
      const int ch = part * G::CPT + c;                           // fp8 chunk: dims 16 ch .. 16 ch + 15
      const uint4 a = kr[kk][c], b = vr[kk][c];
      *reinterpret_cast<uint4*>(kh + swz(slot, 2 * ch)) =
          make_uint4(e4m3x2_to_h2(a.x), e4m3x2_to_h2(a.x >> 16), e4m3x2_to_h2(a.y), e4m3x2_to_h2(a.y >> 16));
      *reinterpret_cast<uint4*>(kh + swz(slot, 2 * ch + 1)) =
          make_uint4(e4m3x2_to_h2(a.z), e4m3x2_to_h2(a.z >> 16), e4m3x2_to_h2(a.w), e4m3x2_to_h2(a.w >> 16));
      *reinterpret_cast<uint4*>(vh + swz(slot, 2 * ch)) =
          make_uint4(e4m3x2_to_h2(b.x), e4m3x2_to_h2(b.x >> 16), e4m3x2_to_h2(b.y), e4m3x2_to_h2(b.y >> 16));
      *reinterpret_cast<uint4*>(vh + swz(slot, 2 * ch + 1)) =
          make_uint4(e4m3x2_to_h2(b.z), e4m3x2_to_h2(b.z >> 16), e4m3x2_to_h2(b.w), e4m3x2_to_h2(b.w >> 16));
    }
  }
  // q rows warp + 8 j: SGLang FusedQKNorm arithmetic (fp32 RMS over 64 dims, bf16, RoPE on the bf16 pair, bf16); fp16 tile
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int r = warp + 8 * j;
    uint32_t hv = 0u;
    if (j < TPG) {
      const float2 x = bf2f(qraw[j < TPG ? j : 0]);
      const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
      const float rs = rsqrtf(var + p.eps);
      const float e = bfr(x.x * rs), o = bfr(x.y * rs);
      const float2 c = csq[j < TPG ? j : 0];
      const float e2 = bfr(__fmaf_rn(e, c.x, -__fmul_rn(o, c.y)));
      const float o2 = bfr(__fmaf_rn(e, c.y, __fmul_rn(o, c.x)));
      hv = pack_h2(e2, o2);
    }
    *reinterpret_cast<uint32_t*>(qs + r * 128 + 16 * ((lane >> 2) ^ (r & 7)) + 4 * (lane & 3)) = hv;
  }
  if (tail) {   // new keys: rows warp, warp + 8 of the 16-slot tile (rows >= L stay zero)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int r = warp + 8 * j;
      uint32_t kh2 = 0u, vh2 = 0u;
      if (r < L) {
        const float2 x = bf2f(kraw);
        const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
        const float rs = rsqrtf(var + p.eps);
        const float e = bfr(x.x * rs), o = bfr(x.y * rs);
        float e2 = bfr(__fmaf_rn(e, csk.x, -__fmul_rn(o, csk.y)));
        float o2 = bfr(__fmaf_rn(e, csk.y, __fmul_rn(o, csk.x)));
        e2 = fminf(fmaxf(e2, -FP8_MAX), FP8_MAX);
        o2 = fminf(fmaxf(o2, -FP8_MAX), FP8_MAX);
        float2 v = bf2f(vraw);
        v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
        v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
        const uint32_t k8 = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(e2, o2), __NV_SATFINITE, __NV_E4M3);
        const uint32_t v8 = (uint32_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
        if (tg == 0) {
          *reinterpret_cast<uint16_t*>(p.k_cache + (size_t)loc * (HKV * 64) + h * 64 + 2 * lane) = (uint16_t)k8;
          *reinterpret_cast<uint16_t*>(p.v_cache + (size_t)loc * (HKV * 64) + h * 64 + 2 * lane) = (uint16_t)v8;
        }
        kh2 = e4m3x2_to_h2(k8);
        vh2 = e4m3x2_to_h2(v8);
      }
      *reinterpret_cast<uint32_t*>(kn + r * 128 + 16 * ((lane >> 2) ^ (r & 7)) + 4 * (lane & 3)) = kh2;
      *reinterpret_cast<uint32_t*>(vn + r * 128 + 16 * ((lane >> 2) ^ (r & 7)) + 4 * (lane & 3)) = vh2;
    }
  }
  KS1P(3);
  __syncthreads();
  KS1P(4);

  // ---------------- attention: warp = key slots [warp KPW, (warp + 1) KPW) of this split; the new-key CTA: warp 0, 16 slots
  const float cl2 = p.qk_sm * LOG2E;
  uint32_t qa[4][4];
  {
    const int qr = (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) ldsm_x4(qa[ks], qs + swz(qr, 2 * ks + (lane >> 4)));
  }
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m[2] = {neg_inf(), neg_inf()}, ls[2] = {0.f, 0.f};
  const int gbase = s * KPS;                                     // window slot of key 0 of this CTA's tile
  const int l0 = tg * TPG;                                       // token of rows g (rows g + 8: l0 + 1 when TPG = 2)
  if (!tail) {
    segment<G::NT>(kh, vh, warp * KPW, qa, o, m, ls, cl2, [&](int k, int j) {
      const int gs = gbase + k;                                  // key at absolute position kstart + gs
      const int l = l0 + ((TPG == 2) ? j : 0);
      return gs < n_old && gs >= n_old + l + 1 - WIN;            // > position(l) - WIN (window of token l)
    });
  } else if (warp == 0) {
    segment<2>(kn, vn, 0, qa, o, m, ls, cl2, [&](int k, int j) {
      const int l = l0 + ((TPG == 2) ? j : 0);
      return k < L && k <= l;                                    // causal inside the block
    });
  }
  ls[0] += __shfl_xor_sync(FULL, ls[0], 1);
  ls[0] += __shfl_xor_sync(FULL, ls[0], 2);
  ls[1] += __shfl_xor_sync(FULL, ls[1], 1);
  ls[1] += __shfl_xor_sync(FULL, ls[1], 2);

  KS1P(5);
  // ---------------- in-CTA merge of the 8 warps (scratch over the K / V tiles), then this split's partial
  __syncthreads();
  float* ow = reinterpret_cast<float*>(smem);                   // [NWARP][16][64], dims XOR 8 (row & 7)
  if (t == 0) {
    mw[warp * 16 + g] = m[0]; lw[warp * 16 + g] = ls[0];
    mw[warp * 16 + g + 8] = m[1]; lw[warp * 16 + g + 8] = ls[1];
  }
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = (8 * n + 2 * t) ^ (8 * g);
    *reinterpret_cast<float2*>(ow + (warp * 16 + g) * 64 + d) = make_float2(o[n][0], o[n][1]);
    if (ROWS == 16) *reinterpret_cast<float2*>(ow + (warp * 16 + g + 8) * 64 + d) = make_float2(o[n][2], o[n][3]);
  }
  __syncthreads();
  KS1P(6);
  uint16_t* ph = reinterpret_cast<uint16_t*>(p.part_o);       // phalf: normalized partial O in fp16 (half the merge bytes)
  {
    float M = neg_inf();
#pragma unroll
    for (int w = 0; w < NWARP; ++w) M = fmaxf(M, mw[w * 16 + mr]);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
    float lt = 0.f;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) {
      const float f = ex2(mw[w * 16 + mr] - Mu);
      lt = fmaf(f, lw[w * 16 + mr], lt);
      const float* src = ow + (w * 16 + mr) * 64 + ((DPT * mu) ^ (8 * (mr & 7)));
      if constexpr (DPT == 4) {
        const float4 x = *reinterpret_cast<const float4*>(src);
        acc[0] = fmaf(f, x.x, acc[0]); acc[1] = fmaf(f, x.y, acc[1]); acc[DPT - 2] = fmaf(f, x.z, acc[DPT - 2]);
        acc[DPT - 1] = fmaf(f, x.w, acc[DPT - 1]);
      } else {
        const float2 x = *reinterpret_cast<const float2*>(src);
        acc[0] = fmaf(f, x.x, acc[0]); acc[1] = fmaf(f, x.y, acc[1]);
      }
    }
    const size_t prow = ((size_t)grp * (NS + 1) + s) * ROWS + mr;
    if (p.phalf) {
      const float inv = (lt > 0.f) ? 1.f / lt : 0.f;
      if constexpr (DPT == 4)
        __stcg(reinterpret_cast<uint2*>(ph + prow * 64 + DPT * mu),
               make_uint2(pack_h2(acc[0] * inv, acc[1] * inv), pack_h2(acc[DPT - 2] * inv, acc[DPT - 1] * inv)));
      else
        __stcg(reinterpret_cast<unsigned int*>(ph + prow * 64 + DPT * mu), pack_h2(acc[0] * inv, acc[1] * inv));
    } else {
      if constexpr (DPT == 4)
        __stcg(reinterpret_cast<float4*>(p.part_o + prow * 64 + DPT * mu), make_float4(acc[0], acc[1], acc[DPT - 2], acc[DPT - 1]));
      else
        __stcg(reinterpret_cast<float2*>(p.part_o + prow * 64 + DPT * mu), make_float2(acc[0], acc[1]));
    }
    if (mu == 0) __stcg(p.part_ml + prow, make_float2(M, lt));
  }
  KS1P(7);
  if (p.sync == 0) __threadfence();
  __syncthreads();
  if (tid == 0) {
    int tk;
    if (p.sync == 1) {
      tk = atom_add_acqrel(p.counters + grp, 1);
    } else {
      if (p.sync == 2) fence_acqrel();
      tk = atomicAdd(p.counters + grp, 1);
      if (p.sync == 2 && tk == NS) fence_acqrel();
    }
    misc[0] = (tk == NS) ? 1 : 0;                                // NS + 1 arrivals per group
  }
  __syncthreads();
  KS1P(8);
  if (misc[0] == 0) return;

  // ---------------- last CTA of the group: merge the NS + 1 partials in split order, bf16, RMSNorm x sigmoid(gate)
  if (p.sync == 0) __threadfence();
  {
    float2 ml[NS + 1];
    float ov[NS + 1][DPT];
#pragma unroll
    for (int q = 0; q <= NS; ++q) {
      const size_t prow = ((size_t)grp * (NS + 1) + q) * ROWS + mr;
      ml[q] = __ldcg(p.part_ml + prow);
      if (p.phalf) {
        if constexpr (DPT == 4) {
          const uint2 v = __ldcg(reinterpret_cast<const uint2*>(ph + prow * 64 + DPT * mu));
          const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
          const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
          ov[q][0] = a.x; ov[q][1] = a.y; ov[q][DPT - 2] = b.x; ov[q][DPT - 1] = b.y;
        } else {
          const unsigned int v = __ldcg(reinterpret_cast<const unsigned int*>(ph + prow * 64 + DPT * mu));
          const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v));
          ov[q][0] = a.x; ov[q][1] = a.y;
        }
      } else {
        if constexpr (DPT == 4) {
          const float4 v = __ldcg(reinterpret_cast<const float4*>(p.part_o + prow * 64 + DPT * mu));
          ov[q][0] = v.x; ov[q][1] = v.y; ov[q][DPT - 2] = v.z; ov[q][DPT - 1] = v.w;
        } else {
          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.part_o + prow * 64 + DPT * mu));
          ov[q][0] = v.x; ov[q][1] = v.y;
        }
      }
    }
    float M = neg_inf();
#pragma unroll
    for (int q = 0; q <= NS; ++q) M = fmaxf(M, ml[q].x);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float den = 0.f;
    float a[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) a[i] = 0.f;
#pragma unroll
    for (int q = 0; q <= NS; ++q) {
      const float w = ex2(ml[q].x - Mu);
      const float wl = w * ml[q].y;                              // split weight: e^(m_q - M) l_q
      const float wa = p.phalf ? wl : w;                         // fp16 partials are normalized, fp32 ones are not
      den += wl;
#pragma unroll
      for (int i = 0; i < DPT; ++i) a[i] = fmaf(wa, ov[q][i], a[i]);
    }
    const float dd = (den > 0.f) ? den : 1.f;
    float y[DPT];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < DPT; ++i) { y[i] = bfr(a[i] / dd); ss = fmaf(y[i], y[i], ss); }
#pragma unroll
    for (int d = UPR / 2; d > 0; d >>= 1) ss += __shfl_xor_sync(FULL, ss, d);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    uint32_t ow2[DPT / 2];
#pragma unroll
    for (int i = 0; i < DPT / 2; ++i) {
      const float2 gg = bf2f(graw[i]);
      const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * i] * rs * (1.f / (1.f + __expf(-gg.x))),
                                                      y[2 * i + 1] * rs * (1.f / (1.f + __expf(-gg.y))));
      ow2[i] = *reinterpret_cast<const uint32_t*>(&o2);
    }
    __nv_bfloat16* op = p.out + (size_t)(tg * TPG + (mr >> 3)) * (HQ * 64) + (size_t)(h * 8 + (mr & 7)) * 64 + DPT * mu;
    if constexpr (DPT == 4) *reinterpret_cast<uint2*>(op) = make_uint2(ow2[0], ow2[DPT / 2 - 1]);
    else *reinterpret_cast<uint32_t*>(op) = ow2[0];
  }
  if (tid == 0) p.counters[grp] = 0;                             // self-reset (every split of the group has arrived)
  KS1P(9);
#endif
}

template <int L, int NS, int NTG>
static void launch(const Params& P, bool pdl, cudaStream_t st) {
  using G = Geo<L, NS, NTG>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(NS + 1), (unsigned)HKV, (unsigned)NTG);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = G::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks1_kernel<L, NS, NTG>, P));
}

template <int L, int NS, int NTG>
static std::vector<int64_t> info_t() {
  using G = Geo<L, NS, NTG>;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks1_kernel<L, NS, NTG>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks1_kernel<L, NS, NTG>, NTHR, G::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)G::SMEM, (int64_t)occ};
}

}  // namespace ks1

#define KS1_INSTS @INSTS@

void ks1_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out,
                 torch::Tensor prof, int64_t L, int64_t NS, int64_t NTG, double qk_sm, double eps, int64_t trig,
                 int64_t pdl, int64_t sync, int64_t phalf) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1, "ks1: qkv");
  TORCH_CHECK(qkv.size(0) == L && qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64, "ks1: qkv shape");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.numel() == 1, "ks1: B = 1 int32 cache_seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == L && cache_loc.is_contiguous(), "ks1: cache_loc");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == L && positions.is_contiguous(), "ks1: positions");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == 1 &&
              page_table.is_contiguous(), "ks1: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "ks1: caches");
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous() && cos_sin.dim() == 2 && cos_sin.size(1) == 64,
              "ks1: cos_sin");
  const int64_t rows = 8 * (L / NTG);
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_o.numel() >= 2 * NTG * (NS + 1) * rows * 64, "ks1: part_o");
  TORCH_CHECK(part_ml.scalar_type() == at::kFloat && part_ml.numel() >= 2 * NTG * (NS + 1) * rows * 2, "ks1: part_ml");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= 2 * NTG, "ks1: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == L * ks1::HQ * 64, "ks1: out");
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 16 * 2 * NTG * (NS + 1)), "ks1: prof");
  const at::cuda::CUDAGuard guard(qkv.device());
  ks1::Params P;
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
  P.trig = (int)trig;
  P.sync = (int)sync;
  P.phalf = (int)phalf;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define KS1_X(L_, NS_, NTG_) if (L == L_ && NS == NS_ && NTG == NTG_) { ks1::launch<L_, NS_, NTG_>(P, pdl != 0, st); return; }
  KS1_INSTS
#undef KS1_X
  TORCH_CHECK(false, "ks1: geometry not built");
}

std::vector<int64_t> ks1_info(int64_t L, int64_t NS, int64_t NTG) {
#define KS1_X(L_, NS_, NTG_) if (L == L_ && NS == NS_ && NTG == NTG_) return ks1::info_t<L_, NS_, NTG_>();
  KS1_INSTS
#undef KS1_X
  return {};
}
"""

# Python glue, inserted into the layer verbatim (the lab exec()s the same text with k1_fa4v8's helpers).
KS1_PY = r'''
KS1_ENABLE = True          # master switch (read at call time; False at prepare also skips the KS1 build and self-tests)
KS1_ROUTE = {1: True, 8: True}       # q_len -> KS1 on B = 1 sliding layers (read at call time)
KS1_GEOM = {1: (16, 1), 8: (16, 4)}  # q_len -> (NS key splits, NTG token groups) (read at call time; must be in KS1_INSTS)
KS1_TRIG = {1: 1, 8: 1}              # q_len -> griddepcontrol.launch_dependents: 0 implicit at exit, 1 right after the wait, 2 at entry
KS1_PDL = True                       # PDL launch: the pre-wait gathers overlap the qkv projection (read at call time)
KS1_SYNC = {1: 1, 8: 1}              # q_len -> ticket protocol (0 fence.sc, 1 atom.acq_rel, 2 fence.acq_rel; read at call time)
KS1_PHALF = {1: False, 8: False}     # q_len -> partials as normalized fp16 (half the merge bytes; measured neutral, so fp32 as FA4)
KS1_P1_CARVEOUT = -1                 # prepare: >= 0 sets P1's preferred smem carveout (%) so KS1 CTAs can co-reside with the qkv GEMV
KS1_INSTS = ((1, 16, 1), (8, 16, 4))   # (q_len, NS, NTG) built + self-tested in prepare (lab: (1, 8, 1), (8, 8, 4), (8, 8, 8) too)
KS1_PREPARE_BY_S = 230.0             # no KS1 build / self-test starts after this many seconds of prepare (limit 300 s)
KS1_PROF_BUF = None                  # lab only: int64 CUDA tensor [>= 16 x CTAs] -> per-CTA %globaltimer stamps of the next calls
KS1_EPS = 1e-5
# (ctx, q_len, seed, row0) on B = 1, window 2048: full windows, window edges, short caches, cache row 0 as a new-token row
KS1_SELF_TEST_CASES = ((3000, 1, 901, False), (3000, 8, 902, False), (2047, 1, 903, False), (2047, 8, 904, False),
                       (2048, 1, 905, False), (2040, 8, 906, False), (2049, 8, 907, False), (100, 1, 908, False),
                       (5, 8, 909, False), (1, 1, 910, False), (4000, 1, 911, True), (4000, 8, 912, True))
_KS1 = {"mod": None, "cnt": {}, "build_s": 0.0}


def _ks1_build():
    """KS1's own load_inline (KS1_INSTS instantiations). Returns the module (cached)."""
    if _KS1["mod"] is not None:
        return _KS1["mod"]
    from torch.utils.cpp_extension import load_inline
    insts = " ".join(f"KS1_X({l}, {n}, {g})" for l, n, g in KS1_INSTS)
    cuda = KS1_CUDA_SRC.replace("@INSTS@", insts)
    tag = hashlib.sha1((KS1_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    _KS1["mod"] = load_inline(name=f"ks1_{tag}", cpp_sources=KS1_CPP_SRC, cuda_sources=cuda,
                              functions=["ks1_forward", "ks1_info"],
                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                              verbose=False)
    _KS1["build_s"] = time.time() - t0
    return _KS1["mod"]


def _ks1_counters(dev):
    """Persistent zeroed int32 tickets (the kernel resets them); created eagerly in prepare (never under capture)."""
    key = str(dev)
    if key not in _KS1["cnt"]:
        _KS1["cnt"][key] = torch.zeros(64, dtype=torch.int32, device=dev)
    return _KS1["cnt"][key]


def _ks1_call(qkv, data, cos_sin, geom=None, trig=None, pdl=None, sync=None, phalf=None):
    """One KS1 launch on the layer's qkv [L, 2304] (B = 1): writes the new K / V cache rows, returns [L, 1024] bf16."""
    cfg = data["config"]
    L = cfg["q_len"]
    ns, ntg = geom if geom is not None else KS1_GEOM[L]
    trig = KS1_TRIG.get(L, 0) if trig is None else trig
    pdl = KS1_PDL if pdl is None else pdl
    sync = KS1_SYNC.get(L, 1) if sync is None else sync
    phalf = KS1_PHALF.get(L, False) if phalf is None else phalf
    dev = qkv.device
    rows = 8 * (L // ntg)
    po = torch.empty((2 * ntg * (ns + 1) * rows * 64,), dtype=torch.float32, device=dev)
    pml = torch.empty((2 * ntg * (ns + 1) * rows * 2,), dtype=torch.float32, device=dev)
    out = torch.empty((L, 16 * 64), dtype=torch.bfloat16, device=dev)
    prof = KS1_PROF_BUF if KS1_PROF_BUF is not None else torch.empty((0,), dtype=torch.int64, device=dev)
    _KS1["mod"].ks1_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                            _ks1_counters(dev), out, prof, L, ns, ntg, float(cfg["sm_scale"] * cfg["qk_scale_factor"]),
                            KS1_EPS, int(trig), 1 if pdl else 0, int(sync), 1 if phalf else 0)
    return out


def _ks1_wanted(qkv, data, cfg, state):
    """KS1 for this call? B = 1 sliding (window 2048, RoPE theta 500000), q_len routed in KS1_ROUTE, page 1, unit KV scale,
    tp8 heads, the dtypes / layouts the kernel reads; needs the prepare self-test. Shape / config only (graph-safe)."""
    if not (KS1_ENABLE and state.get("ks1") and _KS1["mod"] is not None):
        return False
    L = cfg["q_len"]
    if not (cfg["batch"] == 1 and KS1_ROUTE.get(L) and cfg["window"] == 2048 and cfg["rope_theta"] == ROPE_THETA
            and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2 and (L, *KS1_GEOM.get(L, (0, 0))) in state["ks1"]):
        return False
    cs = state.get("cos_sin")
    return (cs is not None and cs.dtype == torch.float32 and cs.is_contiguous() and cs.dim() == 2 and cs.shape[1] == 64
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.shape[0] == L
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2 and data["page_table"].is_contiguous()
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].numel() == 1
            and data["cache_loc"].dtype == torch.int64 and data["cache_loc"].is_contiguous()
            and data["positions"].dtype == torch.int64 and data["positions"].is_contiguous())


def _ks1_run(qkv, data, state):
    return _ks1_call(qkv, data, state["cos_sin"])


def _ks1_case(ctx, length, seed, row0, device):
    return _k1fa4_case(1, ctx, length, 2048, seed, row0, device)


def _ks1_self_test(cos_sin, device, geoms, verbose=True):
    """Per case and geometry: rel L2 vs the FP32 reference <= 1.25 x the r1 path's + 5e-3, finite, the new cache bytes equal to
    r1's, tickets back at zero, a second call (other PDL trigger) bit-identical; then one CUDA-graph replay per q_len identical
    to the eager call. Returns the set of (q_len, NS, NTG) geometries that passed every case of their q_len."""
    cnt = _ks1_counters(device)
    bad = set()
    worst = {}
    for ctx, length, seed, row0 in KS1_SELF_TEST_CASES:
        mine = [(l, n, g) for l, n, g in geoms if l == length and (l, n, g) not in bad]
        if not mine:
            continue
        ref = _k1fa4_reference(_ks1_case(ctx, length, seed, row0, device)).view(length, -1)
        d1 = _ks1_case(ctx, length, seed, row0, device)
        o1 = _k1fa4_r1_path(d1).float().view(length, -1)
        torch.cuda.synchronize(device)
        e1 = float((o1 - ref).norm() / ref.norm())
        rows = d1["cache_loc"]
        k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
        v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
        for geo in mine:
            dk = _ks1_case(ctx, length, seed, row0, device)
            qkv = dk["qkv"].view(length, -1)
            o_a = _ks1_call(qkv, dk, cos_sin, geo[1:], 0, KS1_PDL).clone()
            o_b = _ks1_call(qkv, dk, cos_sin, geo[1:], 1, KS1_PDL, 0)   # other trigger, legacy fence.sc tickets
            o_c = _ks1_call(qkv, dk, cos_sin, geo[1:], 0, KS1_PDL, 2, not KS1_PHALF.get(length, False))   # other partial format
            torch.cuda.synchronize(device)
            ek = float((o_a.float() - ref).norm() / ref.norm())
            ec = float((o_c.float() - ref).norm() / ref.norm())
            kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
            vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
            nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
            tick = int(cnt[:16].abs().sum())
            same = bool(torch.equal(o_a, o_b))
            ok = (bool(torch.isfinite(o_a).all()) and bool(torch.isfinite(o_c).all()) and max(ek, ec) <= 1.25 * e1 + 5e-3
                  and nbad == 0 and tick == 0 and same)
            worst[geo] = max(worst.get(geo, (0.0, 0.0)), (ek, e1))
            if not ok:
                bad.add(geo)
            if verbose or not ok:
                print(f"[ks1] self-test L{length} NS{geo[1]} NTG{geo[2]} ctx{ctx}{' row0' if row0 else ''}: {'PASS' if ok else 'FAIL'} "
                      f"err {ek:.4e} / other partials {ec:.4e} (r1 {e1:.4e}) cache bytes != r1 {nbad} tickets {tick} repeat identical {same}", flush=True)
            del dk, o_a, o_b, o_c
    ok_geo = {geo for geo in geoms if geo not in bad and geo in worst}
    for geo in sorted(set(geoms)):
        w = worst.get(geo, (float("nan"), float("nan")))
        print(f"[ks1] geometry L{geo[0]} NS{geo[1]} NTG{geo[2]}: {'ok' if geo in ok_geo else 'FAILED'} (worst err {w[0]:.3e} vs r1 "
              f"{w[1]:.3e})", flush=True)
    for length in sorted({l for l, _, _ in ok_geo}):
        geo = KS1_GEOM.get(length)
        if geo is None or (length, *geo) not in ok_geo:
            continue
        d = _ks1_case(3000, length, 920 + length, False, device)
        qkv = d["qkv"].view(length, -1)
        o1 = _ks1_call(qkv, d, cos_sin, geo).clone()
        torch.cuda.synchronize(device)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            og = _ks1_call(qkv, d, cos_sin, geo)
        gr.replay()
        gr.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1)) and int(cnt[:16].abs().sum()) == 0
        print(f"[ks1] graph check L{length} NS{geo[0]} NTG{geo[1]}: replay identical {same}", flush=True)
        del gr, og
        if not same:
            ok_geo = {x for x in ok_geo if x[0] != length}
    return ok_geo


def _ks1_prepare(state, t0):
    """Build KS1 and self-test every KS1_INSTS geometry; state["ks1"] = {(q_len, NS, NTG) that passed}: an empty set keeps
    today's FA4 v8 route. Skipped when KS1_ENABLE is off at prepare
    or prepare already took KS1_PREPARE_BY_S."""
    state["ks1"] = set()
    print(f"[ks1] levers: KS1_ENABLE={KS1_ENABLE} route {KS1_ROUTE} geom {KS1_GEOM} trig {KS1_TRIG} pdl {KS1_PDL} insts {KS1_INSTS}",
          flush=True)
    if not KS1_ENABLE:
        return
    if time.time() - t0 > KS1_PREPARE_BY_S:
        print(f"[ks1] skipped: prepare already took {time.time() - t0:.1f}s (> {KS1_PREPARE_BY_S}s): FA4 v8 stays", flush=True)
        return
    device = state["router"].device
    t1 = time.time()
    try:
        mod = _ks1_build()
        for l, n, g in KS1_INSTS:
            regs, local, smem_b, occ = (int(x) for x in mod.ks1_info(l, n, g))
            print(f"[ks1] L{l} NS{n} NTG{g}: {regs} regs, {local} B local, {smem_b} B smem, {occ} CTAs/SM", flush=True)
        ok_geo = _ks1_self_test(state["cos_sin"], device, KS1_INSTS, verbose=False)
    except Exception as ex:  # noqa: BLE001
        print(f"[ks1] build / self-test FAILED {repr(ex)[-2000:]}: FA4 v8 stays", flush=True)
        ok_geo = set()
    state["ks1"] = set(ok_geo)   # (q_len, NS, NTG) that passed: _ks1_wanted needs (L, *KS1_GEOM[L]) in it
    if ok_geo and KS1_P1_CARVEOUT >= 0 and _P1_MOD is not None and hasattr(_P1_MOD, "p1s_carveout"):
        _P1_MOD.p1s_carveout(int(KS1_P1_CARVEOUT))   # SMs running the qkv GEMV keep room for KS1's smem (PDL co-residency)
        print(f"[ks1] P1 preferred smem carveout {KS1_P1_CARVEOUT}%", flush=True)
    live = {l: tuple(KS1_GEOM[l]) for l in KS1_ROUTE if (l, *KS1_GEOM.get(l, (0, 0))) in ok_geo}
    print(f"[ks1] route {'live ' + str(live) if live else 'FAILED: FA4 v8 stays'} (build {_KS1['build_s']:.1f}s, total "
          f"{time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
'''

SECTION = ("# ---------------------------------------------------------------- KS1 (tools/ks1_patches.py): one-launch sliding attention "
           "at B = 1 (decode 1 / verify 1)\n"
           'KS1_CPP_SRC = r"""' + KS1_CPP_SRC + '"""\n\n'
           'KS1_CUDA_SRC = r"""' + KS1_CUDA_SRC + '"""\n' + KS1_PY + "\n\n")

KFN = ("def kernel_fn(data, state):", "\n    # Post-attention boundary")
P1CPP = ('P1_CPP_SRC = r"""', '"""')
P1CU = ('P1_CUDA_SRC = r"""', '"""')
PATCHES = [
    ("KS1 section before _k3_disable", ("def _k3_disable(state, key, value, piece, ex):", "\ndef kernel_fn("),
     "def _k3_disable(state, key, value, piece, ex):\n", SECTION + "def _k3_disable(state, key, value, piece, ex):\n"),
    ("prepare: KS1 build + self-test", ("def prepare(weights):", "\ndef "),
     "    _g2_prepare(state, t0)\n",
     "    _g2_prepare(state, t0)\n    _ks1_prepare(state, t0)   # KS1: one-launch B = 1 sliding attention (FA4 v8 stays on failure)\n"),
    ("kernel_fn: KS1 route ahead of FA4 v8", KFN,
     "    o = None\n    if _v18_fa4_wanted(qkv, data, cfg, state):\n",
     "    o = None\n"
     "    if _ks1_wanted(qkv, data, cfg, state):\n"
     "        try:\n"
     "            # KS1: QK-norm + RoPE, FP8 K/V store, split attention, in-launch merge and RMSNorm x sigmoid(gate) in ONE PDL launch\n"
     "            # (B = 1 sliding, q_len 1 / 8). Any host exception keeps FA4 v8 from now on.\n"
     "            o = _ks1_run(qkv, data, state).view(tokens, hq * d)\n"
     "        except Exception as ex:  # noqa: BLE001\n"
     '            _k3_disable(state, "ks1", set(), "KS1", ex)\n'
     "            o = None\n"
     "    if o is None and _v18_fa4_wanted(qkv, data, cfg, state):\n"),
    ("P1 C++: p1s_carveout declaration", P1CPP, "std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps);\n",
     "std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps);\nvoid p1s_carveout(int64_t pct);\n"),
    ("P1 CUDA: p1s_carveout", P1CU, "  return {0, 0, 0};\n}\n",
     "  return {0, 0, 0};\n}\n\n"
     "// KS1: preferred shared-memory carveout (percent) of the layer's P1 kernel, so a PDL secondary with large smem can co-reside\n"
     "void p1s_carveout(int64_t pct) {\n"
     "  C10_CUDA_CHECK(cudaFuncSetAttribute(p1::p1s_kernel<1, 0>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)pct));\n"
     "}\n"),
    ("p1_build: export p1s_carveout", ("def p1_build(", "\ndef "), '["p1s_gemv", "p1s_info", "p1_unpack"]',
     '["p1s_gemv", "p1s_info", "p1_unpack", "p1s_carveout"]'),
]


def _scope(text, scope):
    a = text.index(scope[0])
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"ks1: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patches=PATCHES):
    for what, scope, old, new in patches:
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"ks1 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v29_ks1_check.py")
    text = src.read_text(encoding="utf-8")
    for what, scope, old, _ in PATCHES:
        a, b = _scope(text, scope)
        print(f"  {text[a:b].count(old)}x  {what}")
    patched = apply(text)
    out.write_bytes(patched.encode("utf-8"))   # LF as read (Windows text mode would write CRLF)
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

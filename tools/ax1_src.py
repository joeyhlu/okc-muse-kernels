"""AX1: one-launch FULL attention at B = 1 (decode q_len 1), team joeyhlu's own code (2026-10-30).

Replaces the 3-launch FA4 chain (prologue + FA4 core + combine_gate) at batch 1 / full layer / page 1 / kv_scale 1.
Structure (ideas from the public vjeux attention notes; every line here is ours):
  * grid = active splits (CTA s: keys [s*chunk, min(W, (s+1)*chunk)) of the page-table width W, chunk a multiple
    of 64): the key range comes from W (a shape), so page indices load before cache_seqlens lands; keys past the
    real length are masked and never read.
  * 128 threads, warp = (KV head, key half of each 64-key tile); cp.async ring of ST stages (K and V 8 KB each),
    page indices by 4-byte cp.async 2 * ST - 2 tiles ahead of their K / V.
  * QK and PV on f16 mma.sync m16n8k16: K / V e4m3 converted in registers (K: one 16-B LDS per 8 keys, dims mapped
    so a thread's 16 bytes are its B fragment; V: ldmatrix.trans + prmt), query = bf16(rms(q) * qk_scale) * sm_scale
    * log2e in f16 (finer than the e4m3 queries the bound allows).
  * the CTA owning position seq builds the new key / value (RMSNorm, e4m3), writes the cache row and attends it
    from shared memory.
  * merge: o * 2^m, l * 2^m (absolute frame; QK-norm bounds the logits) by red.global.add.v4 into copy s % nc,
    one acq_rel ticket per CTA; the LAST arriving CTA sums the copies, normalizes, RMSNorm * sigmoid(gate),
    stores and zeroes the workspace (no spinning, no residency requirement).
"""

AX1_CPP_SRC = r"""
#include <torch/extension.h>
void ax1_forward(torch::Tensor qkv, torch::Tensor seqlens, torch::Tensor cloc, torch::Tensor pt, torch::Tensor kc,
                 torch::Tensor vc, torch::Tensor wo, torch::Tensor wl, torch::Tensor cnt, torch::Tensor out,
                 int64_t L, int64_t chunk, int64_t nsplit, int64_t nc, double sc2, double qks, double eps,
                 int64_t st, int64_t pdl);
std::vector<int64_t> ax1_info(int64_t L, int64_t st);
"""

AX1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector>

namespace ax1 {

constexpr int TILE = 64, ROWB = 128, HQ = 16, HKV = 2, NTHR = 128;
constexpr unsigned FULL = 0xffffffffu;

struct Params {
  const __nv_bfloat16* qkv;   // [L, sq]: q (16 x 64) | k (2 x 64) | v (2 x 64) | gate (16 x 64)
  const int* seqlens;         // [1] cached length + L
  const int64_t* cloc;        // [L] cache rows of the new tokens
  const int* pt;              // [1, W] page table (page size 1)
  uint8_t* kc;                // e4m3 rows of HKV * 64 bytes
  uint8_t* vc;
  float* wo;                  // [nc][16 L][64] fp32 accumulators (zero between calls)
  float* wl;                  // [nc][16 L]
  int* cnt;                   // [1] arrival ticket (zero between calls)
  __nv_bfloat16* out;         // [L, 16 * 64]
  int sq;                     // qkv row stride (elements)
  int W;                      // page-table width: the key domain of the split partition
  int chunk;                  // keys per split (multiple of 64)
  int nsplit;                 // active splits = grid
  int nc;                     // workspace copies (power of two)
  float sc2;                  // sm_scale * log2(e)
  float qks;                  // qk_scale_factor
  float eps;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(uint32_t d, const void* s, int n) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(s), "r"(n) : "memory");
}
__device__ __forceinline__ void cpa4(uint32_t d, const void* s) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(s) : "memory");
}
__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpw() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float ninf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t h2(uint32_t v) {   // two e4m3 (low 16 bits) -> f16x2, low byte -> low half
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xFFFFu), __NV_E4M3);
  return (uint32_t)r.x | ((uint32_t)r.y << 16);
}
__device__ __forceinline__ uint32_t ph2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  // m16n8k16 f16 -> f32, A rows 8..15 zero (a1 = a3 = 0)
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void red4(float* a, float x, float y, float z, float w) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
#else   // (portable build for the local sm_86 functional test)
  atomicAdd(a, x); atomicAdd(a + 1, y); atomicAdd(a + 2, z); atomicAdd(a + 3, w);
#endif
}
__device__ __forceinline__ void red1(float* a, float x) { asm volatile("red.global.add.f32 [%0], %1;\n" ::"l"(a), "f"(x) : "memory"); }
__device__ __forceinline__ int atom_acqrel(int* a, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_go() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
// 128-byte tile row (2 KV heads x 64 e4m3), 16-byte chunk c (0..7): XOR swizzle with bit 2 = key parity, so the
// 8 lanes of a 128-bit LDS phase (keys k, k + 1 x chunks 4 hk .. 4 hk + 3) hit 8 distinct bank groups, and any 8
// consecutive keys of one chunk (an ldmatrix matrix) do too
__device__ __forceinline__ int swz(int key, int c) { return key * ROWB + ((c ^ (((key & 1) << 2) | ((key >> 1) & 3))) << 4); }

template <int ST> struct Lay1 {
  static constexpr int OFF_V = ST * TILE * ROWB;
  static constexpr int OFF_IDX = 2 * ST * TILE * ROWB;     // [2 ST][64] page indices
  static constexpr int OFF_QH = OFF_IDX + 2 * ST * TILE * 4;   // f16 query [16][64]
  static constexpr int OFF_KN = OFF_QH + 16 * 64 * 2;      // new key [128] e4m3, new value [128]
  static constexpr int OFF_MISC = OFF_KN + 2 * ROWB;
  static constexpr int SMEM = OFF_MISC + 64;
  static_assert(ST >= 2, "ring");   // (ST 2: one tile in flight per CTA, the next issued after the current landed)
  static_assert(2 * 16 * 64 * 4 <= ST * TILE * ROWB, "merge scratch fits the K ring");
};

template <int ST>
__global__ void __launch_bounds__(NTHR, 2) fb1_l1(const __grid_constant__ Params p) {
  using Y = Lay1<ST>;
  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* Ks = sm;
  uint8_t* Vs = sm + Y::OFF_V;
  int* idx = reinterpret_cast<int*>(sm + Y::OFF_IDX);
  uint8_t* qh = sm + Y::OFF_QH;
  uint8_t* kn = sm + Y::OFF_KN;
  uint8_t* vn = kn + ROWB;
  volatile int* misc = reinterpret_cast<volatile int*>(sm + Y::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x;
  const int start = s * p.chunk, end = min(p.W, start + p.chunk);
  const int ntiles = (end - start + TILE - 1) / TILE;

  auto issue_idx = [&](int y) {
    if (tid < TILE && y < ntiles) {
      const int pos = start + y * TILE + tid;
      if (pos < end) cpa4(sa(idx + (y % (2 * ST)) * TILE + tid), p.pt + pos);
    }
  };
  // group 0: page indices of tiles 0 .. ST-2 and the length; group 1: tiles ST-1 .. 2 ST-3
#pragma unroll
  for (int y = 0; y < ST - 1; ++y) issue_idx(y);
  if (tid == TILE) cpa4(sa((const void*)(misc)), p.seqlens);
  cpc();
#pragma unroll
  for (int y = ST - 1; y < 2 * ST - 2; ++y) issue_idx(y);
  cpc();
  cpw<1>();
  __syncthreads();
  const int seq = misc[0] - 1;                 // cached keys; position seq is the block's new key
  const int kend = min(end, seq + 1);          // keys this split attends
  const int cend = min(end, seq);              // cached keys this split reads
  const bool owner = seq >= start && seq < end;

  const int cc = tid & 7, cr0 = tid >> 3;      // copy role: 16-B chunk cc of rows cr0 + 16 k
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % ST;
    const int* ix = idx + (x % (2 * ST)) * TILE;
    uint8_t* kd = Ks + st * TILE * ROWB;
    uint8_t* vd = Vs + st * TILE * ROWB;
#pragma unroll
    for (int k = 0; k < TILE / 16; ++k) {
      const int r = cr0 + 16 * k, pos = start + x * TILE + r;
      const int o = swz(r, cc);
      if (pos < cend) {
        const size_t row = (size_t)(uint32_t)ix[r] * ROWB + cc * 16;
        cpa16(sa(kd + o), p.kc + row, 16);
        cpa16(sa(vd + o), p.vc + row, 16);
      } else if (pos >= kend) {   // past the keys: zero-fill (masked out of S; V must stay finite)
        cpa16(sa(kd + o), p.kc, 0);
        cpa16(sa(vd + o), p.vc, 0);
      } else if (with_new) {      // the new key (pos == seq)
        *reinterpret_cast<uint4*>(kd + o) = *reinterpret_cast<const uint4*>(kn + cc * 16);
        *reinterpret_cast<uint4*>(vd + o) = *reinterpret_cast<const uint4*>(vn + cc * 16);
      }
    }
  };
#pragma unroll
  for (int x = 0; x < ST - 1; ++x) {
    if (x < ntiles) issue_kv(x, false);
    cpc();
  }

  pdl_wait();   // qkv is the preceding kernel's output: nothing above reads it
  // ---- query: thread -> head tid >> 3, dims 8 (tid & 7) .. + 7: bf16(rms(q) * qk_scale) * sm_scale * log2e in f16
  {
    const int r = tid >> 3, c8 = (tid & 7) * 8;
    const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p.qkv + r * 64 + c8));
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
    float x[8];
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(b2[e]);
      x[2 * e] = f.x; x[2 * e + 1] = f.y;
      ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
    }
    ss += __shfl_xor_sync(FULL, ss, 1);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 4);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps) * p.qks;
    uint4 w;
    uint32_t* wp = reinterpret_cast<uint32_t*>(&w);
#pragma unroll
    for (int e = 0; e < 4; ++e) wp[e] = ph2(bfr(x[2 * e] * rs) * p.sc2, bfr(x[2 * e + 1] * rs) * p.sc2);
    *reinterpret_cast<uint4*>(qh + r * 128 + c8 * 2) = w;
  }
  // ---- the new key / value (owner CTA): warp 0 keys, warp 1 values; lane: head lane >> 4, dims 4 (lane & 15) .. + 3
  if (owner && warp < 2) {
    const int hh = lane >> 4, d0 = 4 * (lane & 15);
    const uint2 raw = __ldcg(reinterpret_cast<const uint2*>(p.qkv + (HQ + warp * HKV + hh) * 64 + d0));
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    float y0 = a.x, y1 = a.y, y2 = b.x, y3 = b.y;
    if (warp == 0) {
      float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
      ss += __shfl_xor_sync(FULL, ss, 1);
      ss += __shfl_xor_sync(FULL, ss, 2);
      ss += __shfl_xor_sync(FULL, ss, 4);
      ss += __shfl_xor_sync(FULL, ss, 8);
      const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
      y0 *= rs; y1 *= rs; y2 *= rs; y3 *= rs;
    }
    const uint32_t lo = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(y0, y1), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(y2, y3), __NV_SATFINITE, __NV_E4M3);
    const uint32_t w = lo | (hi << 16);
    *reinterpret_cast<uint32_t*>((warp == 0 ? kn : vn) + hh * 64 + d0) = w;
    const long long row = __ldg(p.cloc);
    *reinterpret_cast<uint32_t*>((warp == 0 ? p.kc : p.vc) + (size_t)row * ROWB + hh * 64 + d0) = w;
  }
  __syncthreads();
  if (owner) {   // the new row into an already issued tile (later tiles take it in issue_kv)
    const int xn = (seq - start) / TILE, r = (seq - start) % TILE;
    if (xn < ST - 1 && tid < 16) {
      const int c = tid & 7;
      uint8_t* d = (tid < 8 ? Ks : Vs) + (xn % ST) * TILE * ROWB + swz(r, c);
      *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>((tid < 8 ? kn : vn) + c * 16);
    }
  }

  // ---- consumer: warp = (KV head hk, key half kh of each tile); rows g = query heads 8 hk + g
  const int hk = warp & 1, kh = warp >> 1;
  uint32_t qa[8];   // k-step s: a0 = qa[2 s] (dims 16 t + 4 s + 0, 1), a2 = qa[2 s + 1] (dims + 2, 3)
  {
    const uint4 q0 = *reinterpret_cast<const uint4*>(qh + (hk * 8 + g) * 128 + 32 * t);
    const uint4 q1 = *reinterpret_cast<const uint4*>(qh + (hk * 8 + g) * 128 + 32 * t + 16);
    qa[0] = q0.x; qa[1] = q0.y; qa[2] = q0.z; qa[3] = q0.w; qa[4] = q1.x; qa[5] = q1.y; qa[6] = q1.z; qa[7] = q1.w;
  }
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m = ninf(), l = 0.f;
  const int vrow = 8 * ((lane >> 3) & 1) + (lane & 7);   // ldmatrix row of this lane (key within a 16-key step)
  const int vch = hk * 4 + (lane >> 4);                   // + 2 cc: chunk of this lane's matrix

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    cpw<ST - 2>();
    __syncthreads();
    if (i + ST - 1 < ntiles) issue_kv(i + ST - 1, true);
    issue_idx(i + 2 * ST - 2);
    cpc();
    const uint8_t* K = Ks + (i % ST) * TILE * ROWB;
    const uint8_t* V = Vs + (i % ST) * TILE * ROWB;
    const int t0 = start + i * TILE + 32 * kh;
    float sc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * kh + 8 * nt + g;
      const uint4 kw = *reinterpret_cast<const uint4*>(K + swz(key, hk * 4 + t));
      sc[nt][0] = 0.f; sc[nt][1] = 0.f; sc[nt][2] = 0.f; sc[nt][3] = 0.f;
      mma(sc[nt], qa[0], qa[1], h2(kw.x), h2(kw.x >> 16));
      mma(sc[nt], qa[2], qa[3], h2(kw.y), h2(kw.y >> 16));
      mma(sc[nt], qa[4], qa[5], h2(kw.z), h2(kw.z >> 16));
      mma(sc[nt], qa[6], qa[7], h2(kw.w), h2(kw.w >> 16));
    }
    if (t0 + 32 > kend) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e)
          if (t0 + 8 * nt + 2 * t + e >= kend) sc[nt][e] = ninf();
    }
    float mx = fmaxf(fmaxf(fmaxf(sc[0][0], sc[0][1]), fmaxf(sc[1][0], sc[1][1])),
                     fmaxf(fmaxf(sc[2][0], sc[2][1]), fmaxf(sc[3][0], sc[3][1])));
    mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 2));
    const float mn = fmaxf(m, mx);
    if (mn != ninf()) {   // (quad-uniform; warp-uniform in practice: whole-warp masks only on the last tile)
      const float al = ex2(m - mn);
      m = mn;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        sc[nt][0] = ex2(sc[nt][0] - mn);
        sc[nt][1] = ex2(sc[nt][1] - mn);
        ls += sc[nt][0] + sc[nt][1];
      }
      l = fmaf(l, al, ls);
#pragma unroll
      for (int n = 0; n < 8; ++n) { o[n][0] *= al; o[n][1] *= al; }
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint32_t a0 = ph2(sc[2 * ks][0], sc[2 * ks][1]);
        const uint32_t a2 = ph2(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
        const int key = 32 * kh + 16 * ks + vrow;
#pragma unroll
        for (int cp = 0; cp < 2; ++cp) {
          uint32_t r[4];
          ldsm4t(r, sa(V + swz(key, vch + 2 * cp)));
          // r0 / r1: chunk 2 cp keys (2t, 2t+1) / (2t+8, 2t+9), bytes (k, d) (k, d+1) (k+1, d) (k+1, d+1), d = 16c + 2g;
          // r2 / r3: chunk 2 cp + 1. prmt 0x3120 -> low half even dim d, high half odd dim d + 1
#pragma unroll
          for (int hf = 0; hf < 2; ++hf) {
            const uint32_t e0 = __byte_perm(r[2 * hf], 0u, 0x3120), e1 = __byte_perm(r[2 * hf + 1], 0u, 0x3120);
            const int c = 2 * cp + hf;
            mma(o[2 * c], a0, a2, h2(e0), h2(e1));
            mma(o[2 * c + 1], a0, a2, h2(e0 >> 16), h2(e1 >> 16));
          }
        }
      }
    }
  }
  pdl_go();
  l += __shfl_xor_sync(FULL, l, 1);
  l += __shfl_xor_sync(FULL, l, 2);
  cpw<0>();
  __syncthreads();   // the ring is free: merge scratch
  float* mb = reinterpret_cast<float*>(Ks);   // [2 kh][16 rows][64]
  float* ml = reinterpret_cast<float*>(Vs);   // [2 kh][16 rows]
  {
    const float f = (m == ninf()) ? 0.f : ex2(fminf(m, 126.f));
    const int row = hk * 8 + g;
    // n-tile 2c (even dims) holds dims 16c + 4t + {0, 2}, n-tile 2c + 1 dims 16c + 4t + {1, 3}
#pragma unroll
    for (int c = 0; c < 4; ++c)
      *reinterpret_cast<float4*>(mb + (kh * 16 + row) * 64 + 16 * c + 4 * t) =
          make_float4(o[2 * c][0] * f, o[2 * c + 1][0] * f, o[2 * c][1] * f, o[2 * c + 1][1] * f);
    if (t == 0) ml[kh * 16 + row] = l * f;
  }
  __syncthreads();
  const int r = tid >> 3, part = tid & 7;
  {
    const int cp = s & (p.nc - 1);
    const float4 a0 = *reinterpret_cast<const float4*>(mb + r * 64 + 8 * part);
    const float4 a1 = *reinterpret_cast<const float4*>(mb + r * 64 + 8 * part + 4);
    const float4 b0 = *reinterpret_cast<const float4*>(mb + (16 + r) * 64 + 8 * part);
    const float4 b1 = *reinterpret_cast<const float4*>(mb + (16 + r) * 64 + 8 * part + 4);
    float* w = p.wo + ((size_t)cp * 16 + r) * 64 + 8 * part;
    red4(w, a0.x + b0.x, a0.y + b0.y, a0.z + b0.z, a0.w + b0.w);
    red4(w + 4, a1.x + b1.x, a1.y + b1.y, a1.z + b1.z, a1.w + b1.w);
    if (part == 0) red1(p.wl + cp * 16 + r, ml[r] + ml[16 + r]);
  }
  __syncthreads();
  if (tid == 0) misc[1] = (atom_acqrel(p.cnt, 1) == p.nsplit - 1) ? 1 : 0;
  __syncthreads();
  if (misc[1] == 0) return;
  // ---- last CTA: sum the copies, normalize, bf16, RMSNorm * sigmoid(gate), store; zero the workspace
  const uint4 graw = __ldcg(reinterpret_cast<const uint4*>(p.qkv + (HQ + 2 * HKV) * 64 + r * 64 + 8 * part));
  float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float ls = 0.f;
  {   // every copy's loads in flight together (one L2 round trip; nc <= 8)
    float4 xa[8], xb[8];
    float lc[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int c = u;
      const bool on = c < p.nc;
      float* w = p.wo + ((size_t)(on ? c : 0) * 16 + r) * 64 + 8 * part;
      xa[u] = on ? __ldcg(reinterpret_cast<const float4*>(w)) : make_float4(0.f, 0.f, 0.f, 0.f);
      xb[u] = on ? __ldcg(reinterpret_cast<const float4*>(w + 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
      lc[u] = on ? __ldcg(p.wl + c * 16 + r) : 0.f;
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int c = u;
      if (c < p.nc) {
        y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;
        y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;
        ls += lc[u];
        float* w = p.wo + ((size_t)c * 16 + r) * 64 + 8 * part;
        __stcg(reinterpret_cast<float4*>(w), make_float4(0.f, 0.f, 0.f, 0.f));
        __stcg(reinterpret_cast<float4*>(w + 4), make_float4(0.f, 0.f, 0.f, 0.f));
        if (part == 0) __stcg(p.wl + c * 16 + r, 0.f);
      }
    }
  }
  const float inv = ls > 0.f ? 1.f / ls : 0.f;
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) { y[e] = bfr(y[e] * inv); ss = fmaf(y[e], y[e], ss); }
  ss += __shfl_xor_sync(FULL, ss, 1);
  ss += __shfl_xor_sync(FULL, ss, 2);
  ss += __shfl_xor_sync(FULL, ss, 4);
  const float rn = rsqrtf(ss * (1.f / 64.f) + p.eps);
  const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
  uint4 ov;
  uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 gt = __bfloat1622float2(g2[e]);
    const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * e] * rn / (1.f + __expf(-gt.x)), y[2 * e + 1] * rn / (1.f + __expf(-gt.y)));
    ow[e] = *reinterpret_cast<const uint32_t*>(&o2);
  }
  *reinterpret_cast<uint4*>(p.out + r * 64 + 8 * part) = ov;
  if (tid == 0) p.cnt[0] = 0;
}

// ======================= q_len 8: FP8 wgmma (sm_90a; the core of our FX-S kernel, kernels/lab/fxs_src.py) =======================
// 256 threads = 2 warpgroups, WG h = KV head h, wgmma rows r = 8 j + hh (token j, query head 8 h + hh); warp w of the WG
// holds rows 16 w + g (token 2 w) and 16 w + g + 8 (token 2 w + 1). Per 64-key tile: QK = 2 x m64n64k32 e4m3 SS (K head
// half of the 128B-swizzled row), V^T (key-permuted, + 8 rows of e4m3 ones) by the WG while QK runs, exact running max
// with offset 8 (P = 2^(s - max + 8) <= 256 in e4m3: the FA4 PV8 numerics), PV = 2 x m64n72k32 RS (cols 64..71 = row
// sums). Merge: o 2^off, l 2^off by red.v4 into copy s % nc; arrival ticket (epoch-alternating counters); CTAs that own
// output rows (ceil(128 / nsplit) each) wait for all arrivals, sum the copies, finalize and zero them.
constexpr int TB = TILE * ROWB;
constexpr int VTB = 72 * 64;
constexpr int QB8 = 64 * 64;
constexpr int NTH8 = 256;
constexpr uint64_t HI64 = ((uint64_t)(512 >> 4) | (2ull << 30)) << 32;    // SBO 512, 64B swizzle
constexpr uint64_t HI128 = ((uint64_t)(1024 >> 4) | (1ull << 30)) << 32;  // SBO 1024, 128B swizzle

template <int ST> struct Lay8 {
  static constexpr int OFF_V = ST * TB;
  static constexpr int OFF_VT = 2 * ST * TB;               // [head][buffer] V^T
  static constexpr int OFF_Q = OFF_VT + 4 * VTB;           // [head] e4m3 Q
  static constexpr int OFF_IDX = OFF_Q + 2 * QB8;
  static constexpr int OFF_KN = OFF_IDX + 2 * ST * TILE * 4;   // new keys [8][128], new values [8][128]
  static constexpr int OFF_MISC = OFF_KN + 2 * 8 * ROWB;
  static constexpr int SMEM = OFF_MISC + 64 + 1024;        // + 1 KB alignment slack (128B swizzle atoms)
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define AX1_SM90 1
#else
#define AX1_SM90 0
#endif
#if AX1_SM90
__device__ __forceinline__ void fpa() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wgf() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgc() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgw0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void barn(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t pk4(float a, float b, float c, float d) {   // 4 x e4m3 (satfinite), a in the low byte
  uint32_t r;
  asm("{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\ncvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\nmov.b32 %0, {lo, hi};\n}\n"
      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
#define AX1_RF(x) asm volatile("" : "+f"(x))
#define AX1_RR(x) asm volatile("" : "+r"(x))
#define AX1_D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define AX1_R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define AX1_R36 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}"
__device__ __forceinline__ void wg_qk(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " AX1_R32 ", %32, %33, p, 1, 1;\n}\n"
               : AX1_D32(d) : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wg_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 " AX1_R36 ", {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : AX1_D32(d), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}
#endif

template <int ST>
__global__ void __launch_bounds__(NTH8, 2) fb1_l8(const __grid_constant__ Params p) {
#if AX1_SM90
  using Y = Lay8<ST>;
  constexpr int L = 8;
  extern __shared__ uint8_t sm_raw[];
  uint8_t* sm = sm_raw + ((1024u - (sa(sm_raw) & 1023u)) & 1023u);
  uint8_t* Ks = sm;
  int* idx = reinterpret_cast<int*>(sm + Y::OFF_IDX);
  uint8_t* kn = sm + Y::OFF_KN;
  uint8_t* vn = kn + 8 * ROWB;
  volatile int* misc = reinterpret_cast<volatile int*>(sm + Y::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int s = blockIdx.x;
  const int start = s * p.chunk, end = min(p.W, start + p.chunk);
  const int ntiles = (end - start + TILE - 1) / TILE;

  auto issue_idx = [&](int y) {
    if (tid < TILE && y < ntiles) {
      const int pos = start + y * TILE + tid;
      if (pos < end) cpa4(sa(idx + (y % (2 * ST)) * TILE + tid), p.pt + pos);
    }
  };
#pragma unroll
  for (int y = 0; y < ST - 1; ++y) issue_idx(y);
  if (tid == TILE) cpa4(sa((const void*)(misc)), p.seqlens);
  cpc();
#pragma unroll
  for (int y = ST - 1; y < 2 * ST - 2; ++y) issue_idx(y);
  cpc();
  cpw<1>();
  __syncthreads();
  const int seq = misc[0] - L;                 // cached keys; positions seq .. seq + 7 are the block's new keys
  const int kend = min(end, seq + L);
  const int cend = min(end, seq);

  const int cc = tid & 7, crw = tid >> 3;      // copy role: 16-B chunk cc of rows crw, crw + 32
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % ST;
    const int* ix = idx + (x % (2 * ST)) * TILE;
    uint8_t* kd = Ks + st * TB;
    uint8_t* vd = Ks + Y::OFF_V + st * TB;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int r = crw + 32 * k, pos = start + x * TILE + r;
      const int o = r * ROWB + ((cc ^ (r & 7)) << 4);
      if (pos < cend) {
        const size_t row = (size_t)(uint32_t)ix[r] * ROWB + cc * 16;
        cpa16(sa(kd + o), p.kc + row, 16);
        cpa16(sa(vd + o), p.vc + row, 16);
      } else if (pos >= kend) {
        cpa16(sa(kd + o), p.kc, 0);
        cpa16(sa(vd + o), p.vc, 0);
      } else if (with_new) {
        *reinterpret_cast<uint4*>(kd + o) = *reinterpret_cast<const uint4*>(kn + (pos - seq) * ROWB + cc * 16);
        *reinterpret_cast<uint4*>(vd + o) = *reinterpret_cast<const uint4*>(vn + (pos - seq) * ROWB + cc * 16);
      }
    }
  };
#pragma unroll
  for (int x = 0; x < ST - 1; ++x) {
    if (x < ntiles) issue_kv(x, false);
    cpc();
  }
  if (tid < 128)   // rows 64..71 of the four V^T buffers: e4m3 ones (0x38; every byte equal, no swizzle)
    *reinterpret_cast<uint4*>(sm + Y::OFF_VT + (tid >> 5) * VTB + 4096 + (tid & 31) * 16) =
        make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);

  pdl_wait();
  // the barrier epoch AFTER the wait: a back-to-back (PDL) launch of this kernel must see the previous one's bump
  const int ep = (tid == 0) ? __ldcg(p.cnt + 6) : 0;
  {   // warp = token: q RMSNorm -> bf16 -> e4m3 (the FA4 prologue's rounding); new K (RMSNorm) / V -> e4m3
    const int jt = warp;
    const __nv_bfloat16* xr = p.qkv + (size_t)jt * p.sq;
    uint32_t qr[HQ];
#pragma unroll
    for (int hd = 0; hd < HQ; ++hd) qr[hd] = __ldcg(reinterpret_cast<const unsigned int*>(xr + hd * 64) + lane);
    const bool mine = seq + L > start && seq < end;   // this CTA holds some new key
    uint32_t nr[4] = {0u, 0u, 0u, 0u};
    if (mine) {
#pragma unroll
      for (int k = 0; k < 4; ++k) nr[k] = __ldcg(reinterpret_cast<const unsigned int*>(xr + (HQ + k) * 64) + lane);   // k0 k1 v0 v1
    }
    const int c = lane >> 3, bo = (2 * lane) & 15;
#pragma unroll
    for (int hd = 0; hd < HQ; ++hd) {
      const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qr[hd]));
      float ss = fmaf(x.x, x.x, x.y * x.y);
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) ss += __shfl_xor_sync(FULL, ss, d);
      const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
      const uint16_t q8 = (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(bfr(x.x * rs), bfr(x.y * rs)), __NV_SATFINITE, __NV_E4M3);
      const int h = hd >> 3, r = 8 * jt + (hd & 7);
      *reinterpret_cast<uint16_t*>(sm + Y::OFF_Q + h * QB8 + r * 64 + ((c ^ ((r >> 1) & 3)) << 4) + bo) = q8;
    }
    if (mine) {
      const long long loc = __ldg(p.cloc + jt);
      const bool wr = seq + jt >= start && seq + jt < end;   // each new row is written by exactly one CTA
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nr[k]));
        if (k < 2) {
          float ss = fmaf(x.x, x.x, x.y * x.y);
#pragma unroll
          for (int d = 16; d > 0; d >>= 1) ss += __shfl_xor_sync(FULL, ss, d);
          const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
          x.x *= rs; x.y *= rs;
        }
        const uint16_t e8 = (uint16_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
        const int head = k & 1;
        *reinterpret_cast<uint16_t*>((k < 2 ? kn : vn) + jt * ROWB + head * 64 + 2 * lane) = e8;
        if (wr) *reinterpret_cast<uint16_t*>((k < 2 ? p.kc : p.vc) + (size_t)loc * ROWB + head * 64 + 2 * lane) = e8;
      }
    }
  }
  __syncthreads();
  if (tid < 128) {   // new rows into the tiles issued above (later tiles take them in issue_kv)
    const int kind = tid >> 6, jn = (tid >> 3) & 7, c = tid & 7, pos = seq + jn;
    if (pos >= start && pos < kend) {
      const int x = (pos - start) / TILE, r = (pos - start) % TILE;
      if (x < ST - 1)
        *reinterpret_cast<uint4*>(Ks + (kind ? Y::OFF_V : 0) + x * TB + r * ROWB + ((c ^ (r & 7)) << 4)) =
            *reinterpret_cast<const uint4*>((kind ? vn : kn) + jn * ROWB + c * 16);
    }
  }
  fpa();

  const int h = warp >> 2, w = warp & 3, g = lane >> 2, tig = lane & 3;
  const float ssc = p.sc2 * p.qks;             // raw score -> log2 units
  float acc[36];
#pragma unroll
  for (int e = 0; e < 36; ++e) { acc[e] = 0.f; AX1_RF(acc[e]); }
  float cr0 = ninf(), cr1 = ninf();
  const uint32_t qlo = __shfl_sync(FULL, ((sa(sm) + Y::OFF_Q + h * QB8) >> 4) | 0x10000u, 0);
  const uint32_t klo = __shfl_sync(FULL, ((sa(sm) + h * 64) >> 4) | 0x10000u, 0);
  const uint32_t vtlo = __shfl_sync(FULL, ((sa(sm) + Y::OFF_VT + h * 2 * VTB) >> 4) | 0x10000u, 0);
  const int vkey = 16 * w + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld0 = sa(sm) + Y::OFF_V + vkey * ROWB + (((4 * h + (lane >> 4)) ^ (lane & 7)) << 4);
  const uint32_t vst = sa(sm) + Y::OFF_VT + h * 2 * VTB + 128 * g + ((w ^ (g & 3)) << 4) + 4 * tig;
  const uint32_t vsel_a = (g >= 4) ? 0x7531u : 0x6420u, vsel_b = (g >= 4) ? 0x6420u : 0x7531u;
  const uint32_t vst_a = vst + ((g >= 4) ? 64u : 0u), vst_b = vst + ((g >= 4) ? 0u : 64u);
  const int lim0 = seq + 2 * w, lim1 = lim0 + 1;   // key pos visible to row g / g + 8 iff pos <= lim

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    const int st = i % ST, buf = i & 1;
    cpw<ST - 2>();
    fpa();
    __syncthreads();
    if (i + ST - 1 < ntiles) issue_kv(i + ST - 1, true);
    issue_idx(i + 2 * ST - 2);
    cpc();
    float sc[32];
    const uint32_t kl = klo + (uint32_t)(st * (TB >> 4));
    wgf();
    wg_qk(sc, HI64 | qlo, HI128 | kl, 0);
    wg_qk(sc, HI64 | (qlo + 2u), HI128 | (kl + 2u), 1);
    wgc();
    {   // this tile's V (head h) -> key-permuted V^T (buffer buf) while QK runs
      const uint32_t vsrc = vld0 + (uint32_t)(st * TB);
      const uint32_t da = vst_a + (uint32_t)(buf * VTB), db = vst_b + (uint32_t)(buf * VTB);
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t r[4];
        ldsm4t(r, u ? (vsrc ^ 32u) : vsrc);
        sts32(da + 1024 * (2 * u), __byte_perm(r[0], r[1], vsel_a));
        sts32(db + 1024 * (2 * u), __byte_perm(r[0], r[1], vsel_b));
        sts32(da + 1024 * (2 * u + 1), __byte_perm(r[2], r[3], vsel_a));
        sts32(db + 1024 * (2 * u + 1), __byte_perm(r[2], r[3], vsel_b));
      }
    }
    wgw0();
#pragma unroll
    for (int e = 0; e < 32; ++e) AX1_RF(sc[e]);
    const int t0 = start + i * TILE;
    if (t0 + TILE - 1 > lim0) {   // tiles reaching the block's own keys (or past the range): causal mask
#pragma unroll
      for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 8 * nt8 + 2 * tig + e;
          if (pos > lim0) sc[4 * nt8 + e] = ninf();
          if (pos > lim1) sc[4 * nt8 + 2 + e] = ninf();
        }
    }
    float m0 = ninf(), m1 = ninf();
#pragma unroll
    for (int nt8 = 0; nt8 < 8; ++nt8) {
      m0 = fmaxf(m0, fmaxf(sc[4 * nt8], sc[4 * nt8 + 1]));
      m1 = fmaxf(m1, fmaxf(sc[4 * nt8 + 2], sc[4 * nt8 + 3]));
    }
    m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 2)) * ssc;
    m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 2)) * ssc;
    const float n0 = fmaxf(cr0, m0 - 8.f), n1 = fmaxf(cr1, m1 - 8.f);
    const float a0 = (n0 == ninf()) ? 1.f : ex2(cr0 - n0), a1 = (n1 == ninf()) ? 1.f : ex2(cr1 - n1);
#pragma unroll
    for (int t = 0; t < 9; ++t) { acc[4 * t] *= a0; acc[4 * t + 1] *= a0; acc[4 * t + 2] *= a1; acc[4 * t + 3] *= a1; }
    cr0 = n0; cr1 = n1;
    const float u0 = (n0 == ninf()) ? 0.f : n0, u1 = (n1 == ninf()) ? 0.f : n1;
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
    for (int e = 0; e < 36; ++e) AX1_RF(acc[e]);
#pragma unroll
    for (int e = 0; e < 8; ++e) AX1_RR(pa[e >> 2][e & 3]);
    fpa();                    // V^T (st.shared) -> async proxy
    barn(1 + h, 128);         // all 4 warps' V^T rows written
    const uint32_t vl = vtlo + (uint32_t)(buf * (VTB >> 4));
    wgf();
    wg_pv(acc, pa[0], HI64 | vl);
    wg_pv(acc, pa[1], HI64 | (vl + 2u));
    wgc();
    wgw0();
#pragma unroll
    for (int e = 0; e < 36; ++e) AX1_RF(acc[e]);
  }
  pdl_go();
  {   // o 2^off, l 2^off -> copy s % nc (lanes tig, tig ^ 1 swap so each lane adds 4 consecutive dims)
    const int cp = s & (p.nc - 1);
    const int R0 = (2 * w) * 16 + 8 * h + g, R1 = R0 + 16;
    const float f0 = (cr0 == ninf()) ? 0.f : ex2(fminf(cr0, 126.f)), f1 = (cr1 == ninf()) ? 0.f : ex2(fminf(cr1, 126.f));
    float* w0 = p.wo + ((size_t)cp * 128 + R0) * 64;
    float* w1 = p.wo + ((size_t)cp * 128 + R1) * 64;
    const bool odd = tig & 1;
#pragma unroll
    for (int t = 0; t < 8; t += 2) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float f = r ? f1 : f0;
        const float x0 = acc[4 * t + 2 * r] * f, x1 = acc[4 * t + 2 * r + 1] * f;
        const float y0 = acc[4 * (t + 1) + 2 * r] * f, y1 = acc[4 * (t + 1) + 2 * r + 1] * f;
        const float s0 = odd ? x0 : y0, s1 = odd ? x1 : y1;
        const float q0 = __shfl_xor_sync(FULL, s0, 1), q1 = __shfl_xor_sync(FULL, s1, 1);
        red4((r ? w1 : w0) + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? q0 : x0, odd ? q1 : x1, odd ? y0 : q0, odd ? y1 : q1);
      }
    }
    if (tig == 0) {
      red1(p.wl + cp * 128 + R0, acc[32] * f0);
      red1(p.wl + cp * 128 + R1, acc[34] * f1);
    }
  }
  const int rpc = (128 + p.nsplit - 1) / p.nsplit;
  const int rlo = min(128, s * rpc), rhi = min(128, rlo + rpc);
  __syncthreads();
  if (tid == 0) {
    int* c = p.cnt + 4 + (ep & 1);
    const int old = atom_acqrel(c, 1);
    int go = 1;
    if (old == p.nsplit - 1) {   // last arrival: reset the other counter (next launch's), advance the epoch
      p.cnt[4 + ((ep & 1) ^ 1)] = 0;
      p.cnt[6] = ep + 1;
    } else if (rlo < rhi) {
      int v = 0;
      unsigned n = 0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
        if (v >= p.nsplit) break;
        if (++n > (1u << 22)) { atomicExch(p.cnt + 7, 1); go = 0; break; }   // sticky flag (prepare checks it)
        __nanosleep(64);
      }
    }
    misc[1] = go && (rlo < rhi);
  }
  __syncthreads();
  if (!misc[1]) return;
  const int part = tid & 7;
  for (int rb = rlo; rb < rhi; rb += NTH8 / 8) {
    const int R = rb + (tid >> 3);
    const bool act = R < rhi;
    const int j = act ? (R >> 4) : 0, hh = R & 15;
    const uint4 graw = act ? __ldcg(reinterpret_cast<const uint4*>(p.qkv + (size_t)j * p.sq + (HQ + 2 * HKV) * 64 + hh * 64 + 8 * part))
                           : make_uint4(0u, 0u, 0u, 0u);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float ls = 0.f;
    if (act) {
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const bool on = u < p.nc;
        const float* src = p.wo + ((size_t)(on ? u : 0) * 128 + R) * 64 + 8 * part;
        xa[u] = on ? __ldcg(reinterpret_cast<const float4*>(src)) : make_float4(0.f, 0.f, 0.f, 0.f);
        xb[u] = on ? __ldcg(reinterpret_cast<const float4*>(src + 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
        lc[u] = on ? __ldcg(p.wl + u * 128 + R) : 0.f;
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        if (u < p.nc) {
          y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;
          y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;
          ls += lc[u];
          float* dst = p.wo + ((size_t)u * 128 + R) * 64 + 8 * part;
          __stcg(reinterpret_cast<float4*>(dst), make_float4(0.f, 0.f, 0.f, 0.f));
          __stcg(reinterpret_cast<float4*>(dst + 4), make_float4(0.f, 0.f, 0.f, 0.f));
          if (part == 0) __stcg(p.wl + u * 128 + R, 0.f);
        }
      }
    }
    const float inv = ls > 0.f ? 1.f / ls : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = bfr(y[e] * inv); ss = fmaf(y[e], y[e], ss); }
    ss += __shfl_xor_sync(FULL, ss, 1);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 4);
    const float rn = rsqrtf(ss * (1.f / 64.f) + p.eps);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * e] * rn / (1.f + __expf(-gt.x)), y[2 * e + 1] * rn / (1.f + __expf(-gt.y)));
      ow[e] = *reinterpret_cast<const uint32_t*>(&o2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (size_t)j * (HQ * 64) + hh * 64 + 8 * part) = ov;
  }
#endif
}

template <int ST>
static void launch8(const Params& P, bool pdl, cudaStream_t stream) {
  constexpr int SM = Lay8<ST>::SMEM;
  static int cap = 0;
  if (cap == 0) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fb1_l8<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    int occ = 0, dev = 0, nsm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fb1_l8<ST>, NTH8, SM));
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    cap = occ * nsm;
  }
  TORCH_CHECK(P.nsplit <= cap, "ax1: q_len 8 grid must be resident (cooperative finalize)");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P.nsplit, 1, 1);
  cfg.blockDim = dim3(NTH8, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fb1_l8<ST>, P));
}

template <int ST>
static std::vector<int64_t> info8() {
  constexpr int SM = Lay8<ST>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, fb1_l8<ST>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(fb1_l8<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fb1_l8<ST>, NTH8, SM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
}

template <int ST>
static void launch1(const Params& P, bool pdl, cudaStream_t stream) {
  constexpr int SM = Lay1<ST>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(fb1_l1<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P.nsplit, 1, 1);
  cfg.blockDim = dim3(NTHR, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fb1_l1<ST>, P));
}

template <int ST>
static std::vector<int64_t> info1() {
  constexpr int SM = Lay1<ST>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, fb1_l1<ST>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(fb1_l1<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fb1_l1<ST>, NTHR, SM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
}

}  // namespace ax1

void ax1_forward(torch::Tensor qkv, torch::Tensor seqlens, torch::Tensor cloc, torch::Tensor pt, torch::Tensor kc,
                 torch::Tensor vc, torch::Tensor wo, torch::Tensor wl, torch::Tensor cnt, torch::Tensor out,
                 int64_t L, int64_t chunk, int64_t nsplit, int64_t nc, double sc2, double qks, double eps,
                 int64_t st, int64_t pdl) {
  TORCH_CHECK(L == 1 || L == 8, "ax1: q_len 1 / 8");
  TORCH_CHECK(qkv.size(0) == L, "ax1: batch 1");
  TORCH_CHECK(chunk % 64 == 0 && chunk > 0 && nsplit > 0 && nc > 0 && nc <= 8 && (nc & (nc - 1)) == 0, "ax1: plan");
  TORCH_CHECK((int64_t)pt.size(1) <= chunk * nsplit && (int64_t)pt.size(1) > chunk * (nsplit - 1), "ax1: plan vs width");
  ax1::Params P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.seqlens = seqlens.data_ptr<int>();
  P.cloc = cloc.data_ptr<int64_t>();
  P.pt = pt.data_ptr<int>();
  P.kc = reinterpret_cast<uint8_t*>(kc.data_ptr());
  P.vc = reinterpret_cast<uint8_t*>(vc.data_ptr());
  P.wo = wo.data_ptr<float>();
  P.wl = wl.data_ptr<float>();
  P.cnt = cnt.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.sq = (int)qkv.stride(0);
  P.W = (int)pt.size(1);
  P.chunk = (int)chunk;
  P.nsplit = (int)nsplit;
  P.nc = (int)nc;
  P.sc2 = (float)sc2;
  P.qks = (float)qks;
  P.eps = (float)eps;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (L == 1 && st == 2) ax1::launch1<2>(P, pdl != 0, stream);
  else if (L == 1 && st == 3) ax1::launch1<3>(P, pdl != 0, stream);
  else if (L == 1 && st == 4) ax1::launch1<4>(P, pdl != 0, stream);
  else if (L == 8 && st == 3) ax1::launch8<3>(P, pdl != 0, stream);
  else if (L == 8 && st == 4) ax1::launch8<4>(P, pdl != 0, stream);
  else TORCH_CHECK(false, "ax1: stages 3 / 4");
}

std::vector<int64_t> ax1_info(int64_t L, int64_t st) {
  if (L == 8) return st == 3 ? ax1::info8<3>() : ax1::info8<4>();
  return st == 2 ? ax1::info1<2>() : st == 3 ? ax1::info1<3>() : ax1::info1<4>();
}
"""

AX1_ENABLE = True
AX1_ROUTE = {1: True, 8: True}        # q_len -> live (B = 1, full, page 1, kv_scale 1)
AX1_SPLITS = {1: 252, 8: 252}         # target CTAs (2 / SM at 132 SMs, 21/22 headroom)
AX1_NC = {1: 8, 8: 8}                 # workspace copies
AX1_ST = {1: 3, 8: 3}                 # cp.async ring stages
AX1_PDL = True
AX1_SELF_TEST = ((1, 3000, 701, False, 0), (1, 80000, 702, True, 0), (1, 10, 703, False, 0), (1, 1, 704, False, 0),
                 (1, 64, 705, False, 0), (1, 127, 706, False, 37), (1, 20000, 707, False, 300), (1, 200000, 708, False, 0),
                 (8, 3000, 711, False, 0), (8, 80000, 712, True, 0), (8, 10, 713, False, 0), (8, 1, 714, False, 0),
                 (8, 60, 715, False, 0), (8, 127, 716, False, 37), (8, 20000, 717, False, 300), (8, 200000, 718, False, 0))
AX1_TOL = {1: 1e-2, 8: 2e-2}         # q_len 1 vs the bf16-query reference, q_len 8 vs the e4m3-query one (e4m3 P, as FA4 PV8)
_AX1 = {"mod": None, "ws": {}, "build_s": 0.0}


def _ax1_build(arch="90a"):
    if _AX1["mod"] is not None:
        return _AX1["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((AX1_CPP_SRC + AX1_CUDA_SRC + arch).encode()).hexdigest()[:10]
    t0 = time.time()
    _AX1["mod"] = load_inline(name=f"ax1_{tag}", cpp_sources=AX1_CPP_SRC, cuda_sources=AX1_CUDA_SRC,
                              functions=["ax1_forward", "ax1_info"],
                              extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                              verbose=False)
    _AX1["build_s"] = time.time() - t0
    return _AX1["mod"]


def _ax1_ws(dev):
    key = str(dev)
    if key not in _AX1["ws"]:
        _AX1["ws"][key] = (torch.zeros(8 * 128 * 64, dtype=torch.float32, device=dev),
                           torch.zeros(8 * 128, dtype=torch.float32, device=dev),
                           torch.zeros(16, dtype=torch.int32, device=dev))
    return _AX1["ws"][key]


def _ax1_plan(width, ns):
    tiles = (width + 63) // 64
    chunk = 64 * (-(-tiles // max(1, ns)))
    return chunk, -(-width // chunk)


def _ax1_call(qkv, data, cfg, ns=None, nc=None, st=None, pdl=None):
    L = cfg["q_len"]
    dev = qkv.device
    wo, wl, cnt = _ax1_ws(dev)
    pt = data["page_table"]
    chunk, nsplit = _ax1_plan(pt.shape[1], ns or AX1_SPLITS[L])
    out = torch.empty((L, 16 * 64), dtype=torch.bfloat16, device=dev)
    _AX1["mod"].ax1_forward(qkv, data["cache_seqlens"], data["cache_loc"], pt, data["k_cache"].view(torch.uint8),
                            data["v_cache"].view(torch.uint8), wo, wl, cnt, out, L, chunk, nsplit, int(nc or AX1_NC[L]),
                            float(cfg["sm_scale"] * 1.4426950408889634), float(cfg["qk_scale_factor"]), 1e-5,
                            int(st or AX1_ST[L]), 1 if (AX1_PDL if pdl is None else pdl) else 0)
    return out


def _ax1_wanted(qkv, data, cfg, state):
    if not (AX1_ENABLE and state.get("ax1") and _AX1["mod"] is not None and AX1_ROUTE.get(cfg["q_len"])):
        return False
    pt = data["page_table"]
    return (cfg["batch"] == 1 and not cfg["window"] and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0
            and cfg["head_dim"] == 64 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.shape[0] == cfg["q_len"]
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and pt.dtype == torch.int32 and pt.dim() == 2 and pt.is_contiguous() and pt.shape[0] == 1
            and cfg["q_len"] <= pt.shape[1] < (1 << 30)
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].is_contiguous()
            and data["cache_loc"].dtype == torch.int64 and data["cache_loc"].is_contiguous())


def _ax1_ref(L, d):
    """Reference on the data after a call (the cache holds the call's own new rows, as the judge checks)."""
    if L == 1:
        return _k1fa4_reference(d).view(L, -1)
    rows = d["cache_loc"]
    kn = d["k_cache"].view(torch.uint8).view(-1, 2, 64)[rows].view(1, L, 2, 64)
    vn = d["v_cache"].view(torch.uint8).view(-1, 2, 64)[rows].view(1, L, 2, 64)
    return _vfa_ref_attention(d, kn, vn).view(L, -1)


def _ax1_case(L, ctx, seed, row0, extra, dev):
    d = _k1fa4_case(1, ctx, L, 0, seed, row0, dev)
    if extra:   # a page table wider than the request (the split plan uses the width): spec miss
        pt = d["page_table"]
        d["page_table"] = torch.cat([pt, torch.full((1, extra), -1, dtype=pt.dtype, device=dev)], 1).contiguous()
    return d


def _ax1_self_test(dev, verbose=True, only=None):
    wo, wl, cnt = _ax1_ws(dev)
    worst = 0.0
    for L, ctx, seed, row0, extra in AX1_SELF_TEST:
        if only is not None and L != only:
            continue
        d = _ax1_case(L, ctx, seed, row0, extra, dev)
        cfg = d["config"]
        x = d["qkv"].view(L, -1)
        k = x[:, 16 * 64:18 * 64].float().view(L, 2, 64)
        kq = (k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)).to(FP8).float()
        vq = x[:, 18 * 64:20 * 64].float().view(L, 2, 64).to(FP8).float()
        out = _ax1_call(x, d, cfg)
        o2 = _ax1_call(x, d, cfg)
        torch.cuda.synchronize(dev)
        ref = _ax1_ref(L, d)
        err = float((out.float() - ref).norm() / ref.norm())
        rows = d["cache_loc"]
        kc = d["k_cache"].view(-1, 2, 64)[rows].float()
        vc = d["v_cache"].view(-1, 2, 64)[rows].float()
        ek = float((kc - kq).norm() / kq.norm())
        ev = float((vc - vq).norm() / vq.norm())
        clean = not (int(cnt[0]) or int(cnt[7]) or float(wo.abs().sum()) or float(wl.abs().sum()))
        same = bool(torch.equal(out, o2))
        ok = bool(torch.isfinite(out.float()).all()) and err < AX1_TOL[L] and ek < 1e-2 and ev < 1e-2 and clean
        worst = max(worst, err)
        if verbose or not ok:
            print(f"[ax1] self-test L{L} ctx{ctx}{' row0' if row0 else ''} +{extra}: {'PASS' if ok else 'FAIL'} err {err:.3e} "
                  f"cache k {ek:.1e} v {ev:.1e} ws {'clean' if clean else 'DIRTY'} repeat-identical {same}", flush=True)
        if not ok:
            raise RuntimeError(f"ax1 self-test L{L} ctx{ctx}")
    for L in ([only] if only is not None else sorted(k for k, v in AX1_ROUTE.items() if v)):   # graph: replay = eager, then fresh inputs
        d = _ax1_case(L, 3000, 720 + L, False, 0, dev)
        x = d["qkv"].view(L, -1)
        o1 = _ax1_call(x, d, d["config"]).clone()
        torch.cuda.synchronize(dev)
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            og = _ax1_call(x, d, d["config"])
        gph.replay()
        torch.cuda.synchronize(dev)
        same = float((og.float() - o1.float()).norm() / o1.float().norm()) < 2e-3
        f = _ax1_case(L, 3000, 730 + L, False, 0, dev)
        for key in ("qkv", "k_cache", "v_cache", "page_table", "cache_seqlens", "cache_loc"):
            d[key].copy_(f[key])
        og.fill_(float("nan"))
        gph.replay()
        torch.cuda.synchronize(dev)
        ref2 = _ax1_ref(L, d)
        e2 = float((og.float() - ref2).norm() / ref2.norm())
        clean = not (int(cnt[0]) or int(cnt[7]) or float(wo.abs().sum()) or float(wl.abs().sum()))
        print(f"[ax1] graph check L{L}: replay identical {same}, fresh inputs err {e2:.3e}, ws {'clean' if clean else 'DIRTY'}",
              flush=True)
        if not (same and e2 < AX1_TOL[L] and clean):
            raise RuntimeError("ax1 graph check failed")
        del gph
    return worst


AX1_TUNE = True                       # prepare-time plan pick on a synthetic scored-shape input (A23)
AX1_TUNE_CT = {1: (3, 4, 5, 6, 8), 8: (5, 6, 7, 8, 10)}   # tiles per split
AX1_TUNE_ST = {1: (2, 3, 4), 8: (3, 4)}   # ring depth: in flight per CTA = (ST - 1) x 16 KB
AX1_TUNE_NC = {1: (4, 8), 8: (8,)}
AX1_TUNE_CTX = 80000
AX1_TUNE_ROUNDS = 25


def _ax1_tune(state, dev, L):
    """Event-timed graph replays after a cold-L2 read, rotated rounds; returns {name: median us}. Picks the fastest AX1
    plan; AX1 stays off at q_len L if the FA4 chain (the shipped route) is faster in the same harness."""
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev).fill_(1.0)
    d = _ax1_case(L, AX1_TUNE_CTX, 900 + L, False, 0, dev)
    x = d["qkv"].view(L, -1)
    tiles = (d["page_table"].shape[1] + 63) // 64
    cands = {}
    occ = {st: int(_AX1["mod"].ax1_info(L, st)[3]) for st in AX1_TUNE_ST[L]}
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    for ct in AX1_TUNE_CT[L]:
        for st in AX1_TUNE_ST[L]:
            ns = -(-tiles // ct)
            if ns > occ[st] * sms and (L == 8 or ns > 2 * occ[st] * sms):   # (q_len 8: resident grid; q_len 1: <= 2 waves)
                continue
            for nc in AX1_TUNE_NC[L]:
                if nc != AX1_NC[L] and st != 3:
                    continue
                cands[f"ax1 ct{ct} st{st} nc{nc}"] = (lambda ns=ns, st=st, nc=nc: _ax1_call(x, d, d["config"], ns=ns, st=st, nc=nc))
    if state.get("fa4v6") and K1FA4_STATE.get("ready"):
        cands["fa4 chain"] = lambda: _v18_fa4_core(x, d)
    graphs = {}
    for name, fn in cands.items():
        fn()
        torch.cuda.synchronize(dev)
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            fn()
        graphs[name] = gph
    ts = {n: [] for n in graphs}
    names = list(graphs)
    for rnd in range(AX1_TUNE_ROUNDS):
        for k in range(len(names)):
            n = names[(k + rnd) % len(names)]
            flush.sum()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            graphs[n].replay()
            e1.record()
            torch.cuda.synchronize(dev)
            ts[n].append(e0.elapsed_time(e1) * 1000.0)
    med = {n: sorted(v)[len(v) // 2] for n, v in ts.items()}
    del graphs, flush
    best = min((n for n in med if n.startswith("ax1")), key=lambda n: med[n])
    ct, st, nc = (int(w[2:]) for w in best.split()[1:])
    AX1_SPLITS[L] = -(-tiles // ct)
    AX1_ST[L] = st
    AX1_NC[L] = nc
    if "fa4 chain" in med and med["fa4 chain"] < med[best]:
        AX1_ROUTE[L] = False
    print(f"[ax1] tune L{L} ctx{AX1_TUNE_CTX} (median us, raw replay incl. graph overhead): "
          + ", ".join(f"{n} {med[n]:.2f}" for n in sorted(med, key=lambda n: med[n]))
          + f" -> {best} (splits {AX1_SPLITS[L]}, stages {st}, copies {nc}), route {'AX1' if AX1_ROUTE[L] else 'FA4 chain'}", flush=True)
    return med


def _ax1_prepare(state, t0):
    state["ax1"] = False
    if not AX1_ENABLE:
        return
    dev = torch.empty(0, device="cuda").device
    try:
        _pt1_wait("ax1")
        if _PT1["errors"].get("ax1"):
            raise RuntimeError(f"build failed: {_PT1['errors']['ax1']}")
        mod = _ax1_build()
        info = {(L, st): list(mod.ax1_info(L, st)) for L in (1, 8) for st in ((2, 3, 4) if L == 1 else (3, 4))}
        worst = 0.0
        for L in sorted(k for k, v in AX1_ROUTE.items() if v):
            try:
                worst = max(worst, _ax1_self_test(dev, only=L))
            except Exception as ex:
                AX1_ROUTE[L] = False
                print(f"[ax1] q_len {L} OFF (self-test): {ex!r}"[:1200], flush=True)
                _ax1_ws(dev)[0].zero_(); _ax1_ws(dev)[1].zero_(); _ax1_ws(dev)[2].zero_()
        if not any(AX1_ROUTE.values()):
            raise RuntimeError("no q_len passed")
        state["ax1"] = True
        if AX1_TUNE:
            for L in sorted(k for k, v in AX1_ROUTE.items() if v):
                try:
                    _ax1_tune(state, dev, L)
                except Exception as ex:
                    print(f"[ax1] tune q_len {L} skipped: {ex!r}"[:600], flush=True)
        print(f"[ax1] LIVE: B = 1 full q_len {sorted(k for k, v in AX1_ROUTE.items() if v)}, splits {AX1_SPLITS}, copies {AX1_NC}, "
              f"stages {AX1_ST}, pdl {AX1_PDL}, info (regs, local, smem, occ) {info}, worst err {worst:.3e}, "
              f"build {_AX1['build_s']:.0f}s, prepare so far {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:
        print(f"[ax1] FAILED, the FA4 chain stays: {ex!r}"[:1500], flush=True)
        state["ax1"] = False

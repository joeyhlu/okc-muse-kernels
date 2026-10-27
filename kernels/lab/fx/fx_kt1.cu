// FX kill test KT1 (notes/study/attn_full_independent.md section 6): G = FX producer + checksum/null consumers,
// P = G + V e4m3->bf16 converter + the FX consumer instruction stream (QK wgmma e4m3 RS, unified-max softmax,
// PV wgmma bf16 RS + row sums). Timing only: Q fragments and wgmma descriptors are placeholders (in-bounds,
// no swizzle); outputs go to scratch. One CTA per SM, 512 threads, stream-K over the request-major tile list.
// FX_PORTABLE (sm_80/86 local protocol test): no wgmma / proxy fences / vector red, smaller rings.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace fxkt {
#ifndef FX_ST
#define FX_ST 6
#endif
#ifndef FX_VST
#define FX_VST 3
#endif
constexpr int TILE = 64, ST = FX_ST, VST = FX_VST, PTD = 8, MAXB = 64, NTHR = 512;
constexpr int KV_STAGE = 2 * TILE * 128;             // K 8 KB (head h at h*4096, 64 B per key) | V 8 KB raw rows
constexpr int VB_STAGE = 2 * TILE * 64 * 2;           // bf16 V: head h at h*8192, 128 B per key
constexpr int OFF_VB = ST * KV_STAGE;
constexpr int OFF_ONES = OFF_VB + VST * VB_STAGE;     // 1 KB (256 B used)
constexpr int OFF_PT = OFF_ONES + 1024;
constexpr int OFF_TP = OFF_PT + PTD * TILE * 4;
constexpr int OFF_BAR = OFF_TP + (MAXB + 8) * 4;
constexpr int NBAR = 2 * ST + 2 * VST + PTD;
constexpr int SMEM = OFF_BAR + 8 * NBAR + 64;
constexpr unsigned FULL = 0xffffffffu;
static_assert(OFF_BAR % 8 == 0, "mbarrier alignment");

struct Params {
  const int* page_table; const uint8_t* k_cache; const uint8_t* v_cache; const int* seq_lens;
  const __nv_bfloat16* qkv; float* acc; int* counters; unsigned long long* csum;
  int B, L, pt_stride; float c;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void bar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void bar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_wait(uint32_t b, uint32_t ph) {
#ifdef FX_PORTABLE
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int nb) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(nb) : "memory");
}
__device__ __forceinline__ void cp4(uint32_t dst, const void* src, int nb) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(dst), "l"(src), "r"(nb) : "memory");
}
__device__ __forceinline__ void cp_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void fence_async() {
#ifndef FX_PORTABLE
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}
// wgmma smem descriptor, no swizzle (mode 0): start, leading and stride byte offsets (all >> 4)
__device__ __forceinline__ uint64_t desc(uint32_t addr, int lbo, int sbo) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}

#define D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"

template <int SD> __device__ __forceinline__ void qk_k32(float (&d)[32], const uint32_t (&a)[4], uint64_t db) {
#ifndef FX_PORTABLE
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, %37, 1, 1;\n"
               : D32(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(SD));
#else
  if (SD == 0) { for (int i = 0; i < 32; ++i) d[i] = __uint_as_float((a[i & 3] & 0x007fffffu) | 0x3f000000u); }
#endif
}
__device__ __forceinline__ void pv_k16(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#ifndef FX_PORTABLE
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1, 1;\n"
               : D32(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#else
  d[0] += __uint_as_float(a0 & 0x007fffffu);
#endif
}
__device__ __forceinline__ void rs_k16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#ifndef FX_PORTABLE
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1, 0;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#else
  d[0] += __uint_as_float(a1 & 0x007fffffu);
#endif
}
__device__ __forceinline__ void wg_fence() {
#ifndef FX_PORTABLE
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit() {
#ifndef FX_PORTABLE
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void wg_wait() {
#ifndef FX_PORTABLE
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
#endif
}
template <int N> __device__ __forceinline__ void fence_regs(uint32_t (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(d[i]) :: "memory");
}
template <int N> __device__ __forceinline__ void fence_acc(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
__device__ __forceinline__ void red4(float* a, float x, float y, float z, float w) {
#ifndef FX_PORTABLE
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
#else
  atomicAdd(a, x); atomicAdd(a + 1, y); atomicAdd(a + 2, z); atomicAdd(a + 3, w);
#endif
}

// MODE 0 = G (producer + checksum / null consumers), MODE 1 = P (+ converter + FX consumer instruction stream)
template <int MODE>
__global__ void __launch_bounds__(NTHR, 1) fx_kt(const Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = tid >> 5;
  const uint32_t s0 = sa(smem);
  int* tp = reinterpret_cast<int*>(smem + OFF_TP);
  const uint32_t bar_full = s0 + OFF_BAR, bar_empty = bar_full + 8 * ST, bar_vready = bar_empty + 8 * ST,
                 bar_vempty = bar_vready + 8 * VST, bar_pt = bar_vempty + 8 * VST;
  if (warp == 0) {                                    // request-major tile prefix: request b owns [tp[b], tp[b+1])
    int run = 0;
    for (int base = 0; base < p.B; base += 32) {
      const int b = base + lane;
      int x = (b < p.B) ? (__ldg(p.seq_lens + b) + TILE - 1) / TILE : 0;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int y = __shfl_up_sync(FULL, x, d); if (lane >= d) x += y; }
      if (b < p.B) tp[b + 1] = run + x;
      run += __shfl_sync(FULL, x, 31);
    }
    if (lane == 0) tp[0] = 0;
  }
  if (tid == 32) {
    for (int s = 0; s < ST; ++s) { bar_init(bar_full + 8 * s, 32); bar_init(bar_empty + 8 * s, 2 + 4); }
    for (int s = 0; s < VST; ++s) { bar_init(bar_vready + 8 * s, 4); bar_init(bar_vempty + 8 * s, 2); }
    for (int s = 0; s < PTD; ++s) bar_init(bar_pt + 8 * s, 32);
#ifndef FX_PORTABLE
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
  }
  __syncthreads();
  const int T = tp[p.B];
  const int t0 = (int)(((long long)blockIdx.x * T) / gridDim.x);
  const int t1 = (int)(((long long)(blockIdx.x + 1) * T) / gridDim.x);
  const int n = t1 - t0, npad = n + (n & 1);          // even: P's loop has a fixed wgmma sequence
  if (npad == 0) return;                              // uniform per CTA (never at the judge shapes: T >> grid)
  int b0 = 0;
  while (b0 + 1 < p.B && tp[b0 + 1] <= t0) ++b0;

  if (wg == 3) {                                      // ---------------- producer: warp 12 (13-15 idle)
    if (warp != 12) return;
    int* pts = reinterpret_cast<int*>(smem + OFF_PT);
    int pb = b0, kb = b0;                             // request of the page-table stream / of the KV stream
    auto tile_of = [&](int i, int& b, int& j, int& nv) {   // tile i of this CTA -> (request, tile in request, valid keys)
      const int t = t0 + i;
      if (t >= t1) { j = 0; nv = 0; return; }
      while (b + 1 < p.B && t >= tp[b + 1]) ++b;
      j = t - tp[b];
      nv = min(TILE, __ldg(p.seq_lens + b) - j * TILE);
    };
    auto pt_issue = [&](int i) {
      const int slot = i % PTD;
      int j, nv;
      tile_of(i, pb, j, nv);
      const int* src = p.page_table + (size_t)pb * p.pt_stride + (size_t)j * TILE;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = 2 * lane + e;
        cp4(sa(pts + slot * TILE + key), (key < nv) ? (const void*)(src + key) : (const void*)p.page_table, (key < nv) ? 4 : 0);
      }
      cp_arrive(bar_pt + 8 * slot);
    };
    for (int i = 0; i < PTD - 1; ++i) pt_issue(i);
    for (int i = 0; i < npad; ++i) {
      const int s = i % ST, slot = i % PTD;
      int j, nv;
      tile_of(i, kb, j, nv);
      if (i >= ST) bar_wait(bar_empty + 8 * s, ((i / ST) - 1) & 1);
      bar_wait(bar_pt + 8 * slot, (i / PTD) & 1);
      const uint32_t stg = s0 + s * KV_STAGE;
#pragma unroll
      for (int r = 0; r < 16; ++r) {                  // 4 whole 128-B rows per instruction, 8 lanes per row
        const int key = 4 * r + (lane >> 3), ch = lane & 7;
        const bool ok = key < nv;
        const size_t off = ok ? (size_t)pts[slot * TILE + key] * 128 + 16 * ch : 0;
        const int kdst = (ch >> 2) * 4096 + key * 64 + 16 * ((ch & 3) ^ (key & 3));
        cp16(stg + kdst, p.k_cache + off, ok ? 16 : 0);
        cp16(stg + 8192 + key * 128 + 16 * ch, p.v_cache + off, ok ? 16 : 0);
      }
      cp_arrive(bar_full + 8 * s);
      pt_issue(i + PTD - 1);
    }
    asm volatile("cp.async.wait_all;" ::: "memory");  // no exit with copies in flight
    return;
  }
  if (wg == 2) {                                      // ---------------- converters (P) / releasers (G)
    const int t = tid - 256, key = t >> 1, h = t & 1;
    if (MODE == 1 && t < 64) reinterpret_cast<uint32_t*>(smem + OFF_ONES)[t] = 0x3f803f80u;   // bf16 1.0 pairs
    for (int i = 0; i < npad; ++i) {
      const int s = i % ST, v = i % VST;
      bar_wait(bar_full + 8 * s, (i / ST) & 1);
      if (MODE == 1) {
        if (i >= VST) bar_wait(bar_vempty + 8 * v, ((i / VST) - 1) & 1);
        const uint4* src = reinterpret_cast<const uint4*>(smem + s * KV_STAGE + 8192 + key * 128 + h * 64);
        uint4* dst = reinterpret_cast<uint4*>(smem + OFF_VB + v * VB_STAGE + h * 8192 + key * 128);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 w = src[q];
          const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
          uint32_t o[8];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(ww[e] & 0xFFFFu), __NV_E4M3);
            const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(ww[e] >> 16), __NV_E4M3);
            const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&lo));
            const float2 c = __half22float2(*reinterpret_cast<const __half2*>(&hi));
            o[2 * e] = pack_bf16(a.x, a.y);
            o[2 * e + 1] = pack_bf16(c.x, c.y);
          }
          dst[(2 * q) ^ (key & 7)] = make_uint4(o[0], o[1], o[2], o[3]);
          dst[(2 * q + 1) ^ (key & 7)] = make_uint4(o[4], o[5], o[6], o[7]);
        }
        fence_async();
      }
      __syncwarp();
      if (lane == 0) {
        if (MODE == 1) bar_arrive(bar_vready + 8 * v);
        bar_arrive(bar_empty + 8 * s);
      }
    }
    return;
  }
  // ---------------- consumers: WG h = kv head h
  const int h = wg, ct = tid & 127;
  float o[32], l[4];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) l[i] = 0.f;
  if (MODE == 0) {                                    // G: WG0 checksums every landed byte, both WGs release
    unsigned long long cs = 0;
    for (int i = 0; i < npad; ++i) {
      const int s = i % ST;
      bar_wait(bar_full + 8 * s, (i / ST) & 1);
      if (h == 0) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(smem + s * KV_STAGE);
#pragma unroll 8
        for (int k = 0; k < 32; ++k) cs += w[k * 128 + ct];
      }
      __syncwarp();
      if (ct == 0) bar_arrive(bar_empty + 8 * s);
    }
    if (h == 0 && p.csum) atomicAdd(p.csum, cs);
  } else {
    uint32_t qa[2][4], pA[16], T[16];
    {
      const __nv_bfloat16* q = p.qkv + (size_t)(b0 * p.L + (ct % p.L)) * 2304 + h * 512 + ((ct / p.L) & 7) * 64;
#pragma unroll
      for (int k = 0; k < 2; ++k)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q + 2 * (4 * k + r)));
          const float2 y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q + 2 * (4 * k + r) + 16));
          qa[k][r] = (uint32_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3) |
                     ((uint32_t)__nv_cvt_float2_to_fp8x2(y, __NV_SATFINITE, __NV_E4M3) << 16);
        }
    }
    const float c = p.c, m0 = 31.6f, m1 = 31.6f;
    float sA[32];
    const uint32_t kbase = s0 + h * 4096, vbase = s0 + OFF_VB + h * 8192;
    const uint64_t ones = desc(s0 + OFF_ONES, 128, 128);
    auto softmax = [&](float (&s)[32], uint32_t (&pp)[16]) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float mm = (j & 1) ? m1 : m0;
        pp[j] = pack_bf16(ex2(fmaf(s[2 * j], c, -mm)), ex2(fmaf(s[2 * j + 1], c, -mm)));
      }
    };
    auto release_k = [&](int i) { if (ct == 0 && i < npad) bar_arrive(bar_empty + 8 * (i % ST)); };
    auto release_v = [&](int i) { if (ct == 0) bar_arrive(bar_vempty + 8 * (i % VST)); };
    {
      bar_wait(bar_full, 0);
      fence_async();
      const uint64_t d0 = desc(kbase, 128, 256), d1 = desc(kbase + 2048, 128, 256);
      wg_fence();
      qk_k32<0>(sA, qa[0], d0);
      qk_k32<1>(sA, qa[1], d1);
      wg_commit();
      wg_wait<0>();
      fence_acc(sA);
      release_k(0);
      softmax(sA, pA);
    }
    for (int i = 0; i < npad; ++i) {
      const int st = (i + 1) % ST, v = i % VST;
      uint64_t dq0 = desc(kbase + st * KV_STAGE, 128, 256), dq1 = desc(kbase + st * KV_STAGE + 2048, 128, 256);
      const uint32_t vt = vbase + v * VB_STAGE;
      uint64_t dv0 = desc(vt, 128, 256), dv1 = desc(vt + 2048, 128, 256), dv2 = desc(vt + 4096, 128, 256),
               dv3 = desc(vt + 6144, 128, 256);
      asm volatile("" : "+l"(dq0), "+l"(dq1), "+l"(dv0), "+l"(dv1), "+l"(dv2), "+l"(dv3));
      if (i + 1 < npad) bar_wait(bar_full + 8 * st, ((i + 1) / ST) & 1);   // last QK is a dummy on a stale slot
      bar_wait(bar_vready + 8 * v, (i / VST) & 1);
      fence_async();
      fence_regs(pA);
      wg_fence();
      qk_k32<0>(sA, qa[0], dq0);
      qk_k32<1>(sA, qa[1], dq1);
      wg_commit();
      pv_k16(o, pA[0], pA[1], pA[2], pA[3], dv0);
      rs_k16(l, pA[0], pA[1], pA[2], pA[3], ones);
      pv_k16(o, pA[4], pA[5], pA[6], pA[7], dv1);
      rs_k16(l, pA[4], pA[5], pA[6], pA[7], ones);
      pv_k16(o, pA[8], pA[9], pA[10], pA[11], dv2);
      rs_k16(l, pA[8], pA[9], pA[10], pA[11], ones);
      pv_k16(o, pA[12], pA[13], pA[14], pA[15], dv3);
      rs_k16(l, pA[12], pA[13], pA[14], pA[15], ones);
      wg_commit();
      wg_wait<1>();
      fence_acc(sA);
      release_k(i + 1);
      softmax(sA, T);
      wg_wait<0>();
      fence_regs(pA);
      release_v(i);
#pragma unroll
      for (int j = 0; j < 16; ++j) pA[j] = T[j];
      fence_regs(pA);
    }
    wg_wait<0>();
    fence_acc(o);
    fence_acc(l);
  }
  // flush (both modes): plain FP32 sums into the request accumulator, then the arrival counter
  float* acc = p.acc + ((size_t)b0 * 2 + h) * 128 * 36 + ct * 36;
#pragma unroll
  for (int k = 0; k < 8; ++k) red4(acc + 4 * k, o[4 * k], o[4 * k + 1], o[4 * k + 2], o[4 * k + 3]);
  red4(acc + 32, l[0], l[1], l[2], l[3]);
  __threadfence();
  if (ct == 0) atomicAdd(p.counters + b0, 1);
}

__global__ void fx_l2flush(const uint4* __restrict__ buf, size_t n, unsigned* sink) {
  uint32_t a = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = __ldcg(buf + i);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (a == 0x9e3779b9u) sink[0] = a;
}
template __global__ void fx_kt<0>(const Params);
template __global__ void fx_kt<1>(const Params);
}  // namespace fxkt

#ifdef FX_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void kt_init() {
  cudaFuncSetAttribute(fxkt::fx_kt<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, fxkt::SMEM);
  cudaFuncSetAttribute(fxkt::fx_kt<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, fxkt::SMEM);
}
int64_t kt_smem() { return fxkt::SMEM; }
void kt_run(int64_t mode, at::Tensor qkv, at::Tensor seq_lens, at::Tensor page_table, at::Tensor k_cache,
            at::Tensor v_cache, at::Tensor acc, at::Tensor counters, at::Tensor csum, int64_t B, int64_t L, int64_t grid,
            double c) {
  fxkt::Params p;
  p.page_table = page_table.data_ptr<int>();
  p.k_cache = reinterpret_cast<const uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<const uint8_t*>(v_cache.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.acc = acc.data_ptr<float>();
  p.counters = counters.data_ptr<int>();
  p.csum = csum.numel() ? reinterpret_cast<unsigned long long*>(csum.data_ptr()) : nullptr;
  p.B = (int)B; p.L = (int)L; p.pt_stride = (int)page_table.size(1); p.c = (float)c;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (mode == 0) fxkt::fx_kt<0><<<(int)grid, fxkt::NTHR, fxkt::SMEM, stream>>>(p);
  else fxkt::fx_kt<1><<<(int)grid, fxkt::NTHR, fxkt::SMEM, stream>>>(p);
}
void kt_flush(at::Tensor buf, at::Tensor sink, int64_t grid) {
  fxkt::fx_l2flush<<<(int)grid, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(buf.data_ptr()), (size_t)buf.numel() / 16, reinterpret_cast<unsigned*>(sink.data_ptr()));
}
#endif

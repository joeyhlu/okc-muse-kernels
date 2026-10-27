// FX skeleton: compile-only check (sm_90a) of the verify-full structure proposed in attn_full_independent.md.
// 1 CTA/SM, 512 threads: WG0/WG1 = consumers (kv head 0/1), WG2 = V FP8->BF16 converters, WG3 warp 0 = producer.
// Consumer: RS FP8 QK (m64n64k32 x2) -> max-free softmax (FFMA + EX2 + bf16x2 pack, no max / rescale) ->
// RS BF16 PV (m64n64k16 x4, B transposed) + row sums by m64n8k16 x4 against a ones tile; S and P double-buffered,
// loop unrolled x2 so every wgmma operand register set is static. Descriptors / swizzles are placeholders (compile only).
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace fx {
constexpr int TILE = 64, ST = 6, VST = 3, PTD = 8;
constexpr int KV_STAGE = 2 * TILE * 128;          // K 8 KB | V 8 KB (both heads, raw 128-B rows)
constexpr int VB_STAGE = 2 * TILE * 64 * 2;        // BF16 V, 2 heads x 64 keys x 64 dims = 16 KB
constexpr int OFF_VB = ST * KV_STAGE;
constexpr int OFF_ONES = OFF_VB + VST * VB_STAGE;  // 16 keys x 8 cols bf16 ones = 256 B
constexpr int OFF_PT = OFF_ONES + 1024;
constexpr int OFF_BAR = OFF_PT + PTD * TILE * 4;
constexpr int SMEM = OFF_BAR + 8 * (ST * 2 + VST * 2 + PTD) + 64;

struct Params {
  const int* page_table; const uint8_t* k_cache; const uint8_t* v_cache; const int* seq_lens;
  const __nv_bfloat16* qkv; float* acc; int* counters; int B, pt_stride, ntiles; float c;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void bar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void bar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void bar_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int nb) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(nb) : "memory");
}
__device__ __forceinline__ void cp_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ uint64_t desc(uint32_t addr, int lbo, int sbo) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}

#define D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"

// S (m64n64, fp32) = Q (RS, e4m3) * K^T (smem, K-major e4m3); scale-d as an immediate (0 on the first k-step)
template <int SD> __device__ __forceinline__ void qk_k32(float (&d)[32], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, %37, 1, 1;\n"
               : D32(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(SD));
}
// O (m64n64, fp32) += P (RS, bf16) * V (smem, MN-major bf16 -> imm-trans-b = 1)
__device__ __forceinline__ void pv_k16(float (&d)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1, 1;\n"
               : D32(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
// l (m64n8, fp32) += P * ones
__device__ __forceinline__ void rs_k16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1, 0;\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_regs(uint32_t (&d)[N]) {   // keep A operands live (no coalescing)
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(d[i]) :: "memory");
}
template <int N> __device__ __forceinline__ void fence_acc(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

__global__ void __launch_bounds__(512, 1) fx_verify(const Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  const uint32_t s0 = sa(smem);
  const uint32_t bar_full = s0 + OFF_BAR, bar_empty = bar_full + 8 * ST, bar_vready = bar_empty + 8 * ST,
                 bar_vempty = bar_vready + 8 * VST, bar_pt = bar_vempty + 8 * VST;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { bar_init(bar_full + 8 * s, 32); bar_init(bar_empty + 8 * s, 2 + 4); }
    for (int s = 0; s < VST; ++s) { bar_init(bar_vready + 8 * s, 4); bar_init(bar_vempty + 8 * s, 2); }
    for (int s = 0; s < PTD; ++s) bar_init(bar_pt + 8 * s, 32);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int n = p.ntiles;        // tiles of 64 keys owned by this CTA (stream-K range; one request here)
  const int b = blockIdx.x % p.B;

  if (wg == 3) {                 // ---------------- producer (warp 12); warps 13-15 retire
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    if ((tid >> 5) != 12) return;
    const int* pt = p.page_table + (size_t)b * p.pt_stride + (size_t)blockIdx.x * n * TILE;
    int* pts = reinterpret_cast<int*>(smem + OFF_PT);
    auto pt_issue = [&](int i) {           // 64 entries = 16 lanes x 16 B, PTD tiles ahead
      const int slot = i % PTD;
      if (lane < 16) cp16(sa(pts + slot * TILE + 4 * lane), pt + (size_t)i * TILE + 4 * lane, (i < n) ? 16 : 0);
      cp_arrive(bar_pt + 8 * slot);
    };
    for (int i = 0; i < PTD - 1; ++i) pt_issue(i);
    for (int i = 0; i < n; ++i) {
      const int s = i % ST, slot = i % PTD;
      if (i >= ST) bar_wait(bar_empty + 8 * s, ((i / ST) - 1) & 1);
      bar_wait(bar_pt + 8 * slot, (i / PTD) & 1);
      pt_issue(i + PTD - 1);
      const uint32_t stg = s0 + s * KV_STAGE;
#pragma unroll
      for (int r = 0; r < 16; ++r) {       // 4 whole 128-B rows per instruction, 8 lanes per row
        const int key = 4 * r + (lane >> 3), ch = lane & 7;
        const int pg = pts[slot * TILE + key];
        const size_t off = (size_t)pg * 128 + 16 * ch;
        const int kdst = (ch < 4) ? (key * 64 + 16 * (ch ^ (key & 3))) : (4096 + key * 64 + 16 * ((ch - 4) ^ (key & 3)));
        cp16(stg + kdst, p.k_cache + off, 16);                       // K: head h tile = [64 keys x 64 B], K-major
        cp16(stg + 8192 + key * 128 + 16 * ch, p.v_cache + off, 16);  // V: raw rows (converted by WG2)
      }
      cp_arrive(bar_full + 8 * s);
    }
    return;
  }
  if (wg == 2) {                 // ---------------- converters: V e4m3 [key][2x64] -> bf16 [head][key][64], MN-major
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    const int t = tid - 256, key = t >> 1, h = t & 1;
    for (int i = 0; i < n; ++i) {
      const int s = i % ST, v = i % VST;
      bar_wait(bar_full + 8 * s, (i / ST) & 1);
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
      fence_async();             // generic-proxy smem writes -> visible to wgmma (async proxy)
      __syncwarp();
      if (lane == 0) { bar_arrive(bar_vready + 8 * v); bar_arrive(bar_empty + 8 * s); }
    }
    return;
  }
  // ---------------- consumers: WG h = kv head h, 64 rows = 8 q heads x 8 tokens
  asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
  const int h = wg;
  uint32_t qa[2][4];
  {
    const __nv_bfloat16* q = p.qkv + (size_t)b * 8 * 2304 + h * 512 + (tid & 127);
#pragma unroll
    for (int k = 0; k < 2; ++k)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q + 64 * (4 * k + r)));
        const float2 y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q + 64 * (4 * k + r) + 2304));
        qa[k][r] = (uint32_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3) |
                   ((uint32_t)__nv_cvt_float2_to_fp8x2(y, __NV_SATFINITE, __NV_E4M3) << 16);
      }
  }
  const float c = p.c;
  const float m0 = 31.6f, m1 = 31.6f;      // per-row offsets (half of the QK-norm bound) for the thread's 2 rows
  float o[32], l[4], sA[32], sB[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) l[i] = 0.f;
  uint32_t pA[16], pB[16];
  const uint64_t ones = desc(s0 + OFF_ONES, 128, 256);

  auto softmax = [&](float (&s)[32], uint32_t (&pp)[16]) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float mm = (j & 1) ? m1 : m0;
      pp[j] = pack_bf16(ex2(fmaf(s[2 * j], c, -mm)), ex2(fmaf(s[2 * j + 1], c, -mm)));
    }
  };
  // FA3 order: [QK(i+1) -> S, PV(i) <- P] ; wait<1> retires QK(i+1) ; softmax(S) -> T while PV(i) runs ;
  // wait<0> ; P = T. All barrier waits and every descriptor of the step are done BEFORE the first wgmma of the
  // step, so no non-wgmma instruction defines a wgmma input while a group is in flight (C7513), and nothing but
  // the retired S is read between issue and wait (C7514).
  const uint32_t kbase = s0 + h * 4096, vbase = s0 + OFF_VB + h * 8192;
  uint32_t T[16];
  auto release_k = [&](int i) { if ((tid & 127) == 0 && i < n) bar_arrive(bar_empty + 8 * (i % ST)); };
  auto release_v = [&](int i) { if ((tid & 127) == 0) bar_arrive(bar_vempty + 8 * (i % VST)); };
  {
    bar_wait(bar_full, 0);
    fence_async();
    const uint64_t d0 = desc(kbase, 16, 512);
    wg_fence();
    qk_k32<0>(sA, qa[0], d0);
    qk_k32<1>(sA, qa[1], d0 + 2);
    wg_commit();
    wg_wait<0>();
    fence_acc(sA);
    release_k(0);
    softmax(sA, pA);
  }
  for (int i = 0; i < n; ++i) {
    const int st = (i + 1) % ST, v = i % VST;
    uint64_t dq0 = desc(kbase + st * KV_STAGE, 16, 512), dq1 = dq0 + 2;
    uint64_t dv0 = desc(vbase + v * VB_STAGE, 1024, 128), dv1 = dv0 + 128, dv2 = dv0 + 256, dv3 = dv0 + 384;
    asm volatile("" : "+l"(dq0), "+l"(dq1), "+l"(dv0), "+l"(dv1), "+l"(dv2), "+l"(dv3));   // materialize before issue
    if (i + 1 < n) bar_wait(bar_full + 8 * st, ((i + 1) / ST) & 1);   // i = n-1: dummy QK on a stale slot
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
    fence_regs(pA);                  // P stays live until PV(i) retired, so T cannot reuse its registers
    release_v(i);
#pragma unroll
    for (int j = 0; j < 16; ++j) pA[j] = T[j];
    fence_regs(pA);
  }
  wg_wait<0>();
  fence_acc(o);
  fence_acc(l);
  // flush: plain FP32 sums into the request's accumulator (unified offset -> no LSE merge)
  float* acc = p.acc + ((size_t)b * 2 + h) * 64 * 72 + (tid & 127) * 36;
#pragma unroll
  for (int k = 0; k < 8; ++k)
    asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(acc + 4 * k), "f"(o[4 * k]), "f"(o[4 * k + 1]), "f"(o[4 * k + 2]), "f"(o[4 * k + 3]) : "memory");
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(acc + 32), "f"(l[0]), "f"(l[1]), "f"(l[2]), "f"(l[3]) : "memory");
  __threadfence();
  if ((tid & 127) == 0) atomicAdd(p.counters + b, 1);
}
}  // namespace fx

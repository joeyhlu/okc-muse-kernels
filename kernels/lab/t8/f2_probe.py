"""R-t8-2 probes (notes/memory/gpu_requests.md; design notes/study/t8_design.md s8): F2, the phase-signalled T<=8 front.
LAB FILE: box runs only (tools/box/t8_f2.py).

Three kernels in one stream, standard PDL, today's attention kernel kept (emulated here with its grid and SMEM):
  f2a_probe  qkv conveyor: 132 CTAs x 320 threads (warp 0 producer, warp 1 control, 8 consumer warps), split-K 8
             granules of 24 KB (K-chunk s % 8), per-granule red.add into FP32 (no fence), ONE release per SM at the phase
             end, one spin, each SM converts its share to BF16 (KS1's input) and zeroes it. Triggers dependents at entry.
  ks1e_probe KS1 stand-in: KS1's grid ((NS + 1) x 2 x NTG = 34 at T=1, NS16; 72 at T=8, NS8 NTG4), 256 threads, KS1's
             dynamic SMEM (40,064 / 72,832 B); KV window rows gathered into registers BEFORE griddepcontrol.wait (as KS1),
             then wait, trigger, q/k/v read, the LOADED chain as a spin (SO1: 6.9 / 7.6 us), CTA 0 writes the o-input.
  f2b_probe  o + boundary + router + pooler conveyor, PDL secondary: its producer streams o / router / pooler weights
             BEFORE griddepcontrol.wait (under KS1's chain); after the wait: o, ONE moments release + spin, s1 / h, ONE
             release + spin, router red.add.v4 (one batch), pooler split-K 4 with per-granule red.add (no fence), ONE
             release + spin, the routing-tail read (logits + p), exit. The last CTA resets counters.
MODE 0: null consumers, no waits / epilogues (the pure two-kernel stream). MODE 1: + every dependency, null consumers.
MODE 2: MODE 1 + P1's 12-bit decode + mma.sync instruction stream on the staged slots (router slots: T FMAs per float).
Bounded waits (err flags, no trap). Marks (%globaltimer) per CTA when marks != nullptr.
"""
import hashlib
import time

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace f2p {
constexpr unsigned FULL = 0xffffffffu;
constexpr int SLOT = 24576, NCW = 8, NT = 32 * (NCW + 2), G = 132, RSL = 63488;
constexpr long long BUDGET = 1ll << 31;
enum { CQ = 0, CM1 = 1, CH = 2, CP = 3, CEXA = 4, CEXB = 5, NC = 8 };
constexpr uint32_t KC4 = 0x38383838u, CMASK = 0x87008700u;

__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int smid() { unsigned v; asm volatile("mov.u32 %0, %%smid;" : "=r"(v)); return (int)v; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_tx(uint32_t b, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) if (clock64() - t0 > BUDGET) { atomicExch(err, 1); atomicExch(err + 1, 1); return; }
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_n(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
__device__ __forceinline__ uint64_t evict_first() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_rel(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void spin_ge(const int* p, int target, int* err) {
  if (ld_acq(p) >= target) return;
  const long long t0 = clock64();
  while (ld_acq(p) < target) { __nanosleep(32); if (clock64() - t0 > BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; } }
}
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async;" ::: "memory"); }
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" :: "n"(32 * NCW) : "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t rotr(uint32_t x, int r) { return __funnelshift_r(x, x, r); }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// P1's 12-bit super-block decode (16 rows x 64 K = 1536 B, a lane's three 16-B chunks) -> 4 k-tiles of A fragments,
// each fed to one mma against B fragments from the staged activation (kc = this slot's K offset in the staged chunk)
__device__ __forceinline__ void dense_slot(const uint8_t* sl, const uint8_t* act, int rowb, int T, int cw, int lane, float* acc) {
  const int g8 = lane >> 2, t4 = lane & 3;
  const uint8_t* xr = act + (size_t)(g8 % T) * rowb;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int sb = 2 * cw + k;   // 16 super-blocks per 24 KB slot, 2 per consumer warp
    const uint4* pp = reinterpret_cast<const uint4*>(sl + sb * 1536) + lane;
    const uint4 p0 = pp[0], p1 = pp[32], p2 = pp[64];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const uint4 bc = kt < 2 ? p0 : p1;
      const uint32_t w0 = (kt & 1) ? bc.z : bc.x, w1 = (kt & 1) ? bc.w : bc.y;
      const uint32_t cw2 = kt == 0 ? p2.x : kt == 1 ? p2.y : kt == 2 ? p2.z : p2.w;
      const uint32_t a0 = __byte_perm(w0, KC4, 0x4140) | (cw2 & CMASK);
      const uint32_t a1 = __byte_perm(w0, KC4, 0x4342) | (rotr(cw2, 4) & CMASK);
      const uint32_t a2 = __byte_perm(w1, KC4, 0x4140) | (rotr(cw2, 8) & CMASK);
      const uint32_t a3 = __byte_perm(w1, KC4, 0x4342) | (rotr(cw2, 12) & CMASK);
      const int kk = (sb * 64 + kt * 16 + 2 * t4) * 2;
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xr + (kk % rowb));
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(xr + ((kk + 16) % rowb));
      mma(acc, a0, a1, a2, a3, b0, b1);
    }
  }
}

// ======================================== F2a: qkv ========================================
struct PA {
  const uint8_t* qkv; const __nv_bfloat16* x; float* qacc; __nv_bfloat16* qkvb;
  int* cnt; int* err; long long* marks; int* stages; int T;
};
template <int NSLOT, int AB, int MODE>
__global__ void __launch_bounds__(NT, 2) f2a_probe(const __grid_constant__ PA p) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* ring = sm;
  uint8_t* act = sm + NSLOT * SLOT;
  float* red = reinterpret_cast<float*>(act + AB);
  uint64_t* bars = reinterpret_cast<uint64_t*>(red + NCW * 128);   // full, empty, abar
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, s = blockIdx.x, T = p.T;
  long long* mk = p.marks ? p.marks + (size_t)s * 16 : nullptr;
  if (tid == 0) { pdl_trigger(); if (mk) { mk[0] = gt(); mk[15] = smid(); } }
  const uint32_t full = sa(bars), empty = sa(bars + NSLOT), abar = sa(bars + 2 * NSLOT);
  if (tid == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(full + 8 * i, 1); mbar_init(empty + 8 * i, NCW); }
    mbar_init(abar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int c = s & 7, nc = (c < 4) ? 17 : 16, u = s >> 3;
  const int nq = (143 - u) / nc + 1;
  if (warp == 0) {
    if (lane == 0) {
      const uint64_t pol = evict_first();
      long long wt = 0;
      for (int i = 0; i < nq; ++i) {
        const int slot = i % NSLOT;
        const long long t0 = mk ? gt() : 0;
        mbar_wait(empty + 8 * slot, ((i / NSLOT) & 1) ^ 1, p.err);
        if (mk) wt += gt() - t0;
        mbar_tx(full + 8 * slot, SLOT);
        bulk(sa(ring + slot * SLOT), p.qkv + ((size_t)c * 144 + u + nc * i) * SLOT, SLOT, full + 8 * slot, pol);
        if (i == 0 && mk) mk[1] = gt();
      }
      if (mk) mk[11] = wt;
    }
  } else if (warp == 1) {
    if (MODE >= 1 && lane == 0) {
      mbar_tx(abar, T * 2048);
      for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 2048), p.x + (size_t)t * 8192 + c * 1024, 2048, abar);
    }
  } else {
    const int cw = warp - 2, ct = tid - 64;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int nst = 0;
    for (int i = 0; i < nq; ++i) {
      const int slot = i % NSLOT;
      if (MODE >= 1 && i == 0) mbar_wait(abar, 0, p.err);
      mbar_wait(full + 8 * slot, (i / NSLOT) & 1, p.err);
      if (MODE == 2) dense_slot(ring + slot * SLOT, act, 2048, T, cw, lane, acc);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + 8 * slot);
      ++nst;
      if (MODE >= 1) {   // granule epilogue: CTA reduction barrier + red.add of 16 x T partials, no fence
        cbar();
        if (ct < 16 * T) atomicAdd(p.qacc + (size_t)(ct >> 4) * 2304 + (u + nc * i) * 16 + (ct & 15), acc[0] * 1e-30f + 1e-30f);
      }
    }
    if (MODE >= 1) {   // phase end: ONE release per SM, one spin, BF16 conversion + zeroing of this SM's share
      cbar();
      if (ct == 0) { __threadfence(); red_rel(p.cnt + CQ, 1); if (mk) mk[2] = gt(); spin_ge(p.cnt + CQ, G, p.err); if (mk) mk[3] = gt(); }
      cbar();
      for (int k = s + G * ct; k < T * 2304; k += G * 32 * NCW) {
        p.qkvb[k] = __float2bfloat16_rn(__ldcg(p.qacc + k));
        p.qacc[k] = 0.f;
      }
    }
    if (ct == 0) { if (p.stages) p.stages[s] = nst; if (mk) mk[10] = gt(); }
  }
  __syncthreads();
  if (MODE >= 1 && tid == 0) {
    __threadfence();
    if (atomicAdd(p.cnt + CEXA, 1) == G - 1) { __threadfence(); p.cnt[CQ] = 0; __threadfence(); p.cnt[CEXA] = 0; }
  }
}

// ======================================== KS1 stand-in ========================================
struct PE {
  const uint8_t* kc; const uint8_t* vc; const int* ptab; const __nv_bfloat16* qkvb; __nv_bfloat16* oin;
  unsigned* sink; long long* marks; int T, seq, nkey; long long spin;
};
template <int SMEMB>
__global__ void __launch_bounds__(256, 2) ks1e_probe(const __grid_constant__ PE p) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x;
  const int cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  const int ncta = gridDim.x * gridDim.y * gridDim.z;
  long long* mk = p.marks ? p.marks + (size_t)(132 + cta) * 16 : nullptr;
  if (tid == 0 && mk) { mk[0] = gt(); mk[15] = smid(); }
  // KV window rows of this CTA's keys -> registers BEFORE the PDL wait (as KS1); 16 x 16 B per thread
  uint4 kv[16];
  const int keys = (p.nkey + ncta - 1) / ncta;
#pragma unroll
  for (int k = 0; k < 16; ++k) {   // piece q: key q / 16, K or V by bit 3, 16-B piece q % 8 of the 128-B row (both heads)
    const int q = tid * 16 + k;
    const int key = cta * keys + (q >> 4) % (keys > 0 ? keys : 1);
    const int pos = p.seq - p.nkey + (key < p.nkey ? key : p.nkey - 1);
    const int pg = p.ptab[pos < 0 ? 0 : pos];
    kv[k] = __ldg(reinterpret_cast<const uint4*>((((q >> 3) & 1) ? p.vc : p.kc) + (size_t)pg * 128) + (q & 7));
  }
  sm[tid] = (uint8_t)tid;   // touch the dynamic SMEM (KS1's footprint is what matters for co-residency)
  pdl_wait();
  if (tid == 0) { pdl_trigger(); if (mk) mk[1] = gt(); }
  unsigned acc = 0u;
  for (int k = tid; k < p.T * 1280; k += 256) acc ^= __bfloat16_as_ushort(p.qkvb[(size_t)(k / 1280) * 2304 + (k % 1280)]);
#pragma unroll
  for (int k = 0; k < 16; ++k) acc ^= kv[k].x ^ kv[k].y ^ kv[k].z ^ kv[k].w;
  acc ^= sm[(tid * 7) & 255];
  const long long t0 = clock64();
  while (clock64() - t0 < p.spin) {}
  if (cta == 0) {
    for (int k = tid; k < p.T * 1024; k += 256)
      p.oin[k] = __float2bfloat16_rn(__bfloat162float(p.qkvb[(size_t)(k / 1024) * 2304 + 1280 + (k % 1024)]) + (acc == 0x12345678u ? 1.f : 0.f));
  }
  if (acc == 0x12345678u) p.sink[0] = acc;
  if (tid == 0 && mk) mk[2] = gt();
}

// ======================================== F2b: o + boundary + router + pooler ========================================
struct PB {
  const uint8_t* ow; const uint8_t* rw; const uint8_t* pw;
  const __nv_bfloat16* oin; const __nv_bfloat16* x;
  float* mom1; float* s1; __nv_bfloat16* hb; float* logits; float* pacc;
  int* cnt; int* err; long long* marks; int* stages; int T;
};
template <int NSLOT, int AB, int MODE>
__global__ void __launch_bounds__(NT, 2) f2b_probe(const __grid_constant__ PB p) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* ring = sm;
  uint8_t* act = sm + NSLOT * SLOT;
  float* red = reinterpret_cast<float*>(act + AB);
  uint64_t* bars = reinterpret_cast<uint64_t*>(red + NCW * 128);   // full, empty, abar, afree
  __shared__ float s_mom[8][4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, s = blockIdx.x, T = p.T;
  long long* mk = p.marks ? p.marks + (size_t)(208 + s) * 16 : nullptr;
  if (tid == 0 && mk) { mk[0] = gt(); mk[15] = smid(); }
  const uint32_t full = sa(bars), empty = sa(bars + NSLOT), abar = sa(bars + 2 * NSLOT), afree = sa(bars + 2 * NSLOT + 1);
  if (tid == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(full + 8 * i, 1); mbar_init(empty + 8 * i, NCW); }
    mbar_init(abar, 1);
    mbar_init(afree, NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) (&s_mom[0][0])[tid] = 0.f;
  __syncthreads();
  const int no = (511 - s) / G + 1;
  const int qq = s & 3, u4 = s >> 2, npl = (255 - u4) / 33 + 1;
  const int nsl = no + 3 + 2 * npl;
  if (warp == 0) {   // producer: weights do not depend on the attention, so they stream before the PDL wait
    if (lane == 0) {
      const uint64_t pol = evict_first();
      long long wt = 0;
      for (int i = 0; i < nsl; ++i) {
        const uint8_t* src;
        uint32_t bytes = SLOT;
        if (i < no) src = p.ow + (size_t)(s + G * i) * SLOT;
        else if (i < no + 3) { const int j = i - no; src = p.rw + (size_t)s * RSL + (size_t)j * SLOT; bytes = j < 2 ? SLOT : RSL - 2 * SLOT; }
        else { const int j = i - no - 3; src = p.pw + (((size_t)qq * 256 + u4 + 33 * (j >> 1)) * 2 + (j & 1)) * SLOT; }
        const int slot = i % NSLOT;
        const long long t0 = mk ? gt() : 0;
        mbar_wait(empty + 8 * slot, ((i / NSLOT) & 1) ^ 1, p.err);
        if (mk) wt += gt() - t0;
        mbar_tx(full + 8 * slot, bytes);
        bulk(sa(ring + slot * SLOT), src, bytes, full + 8 * slot, pol);
        if (i == 0 && mk) mk[1] = gt();
      }
      if (mk) mk[11] = wt;
    }
  } else if (warp == 1) {   // control: PDL wait (attention done), o-input staging; h quarter after the h hop
    if (lane == 0) {
      pdl_wait();
      if (mk) mk[2] = gt();
      if (MODE >= 1) {
        fence_async();
        mbar_tx(abar, T * 2048);
        for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 2048), p.oin + (size_t)t * 1024, 2048, abar);
        spin_ge(p.cnt + CH, G, p.err);
        if (mk) mk[6] = gt();
        mbar_wait(afree, 0, p.err);
        fence_async();
        mbar_tx(abar, T * 4096);
        for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 4096), p.hb + (size_t)t * 8192 + qq * 2048, 4096, abar);
      }
    }
  } else {
    const int cw = warp - 2, ct = tid - 64;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int nst = 0;
    if (MODE >= 1) {   // this SM's share of the logits / pooler accumulators, ordered before any red.add by the hops
      for (int k = s + G * ct; k < T * 256; k += G * 32 * NCW) p.logits[k] = 0.f;
      for (int k = s + G * ct; k < T * 4096; k += G * 32 * NCW) p.pacc[k] = 0.f;
    }
    for (int i = 0; i < nsl; ++i) {
      const int slot = i % NSLOT;
      if (MODE >= 1) {
        if (i == 0) mbar_wait(abar, 0, p.err);
        if (i == no + 3) mbar_wait(abar, 1, p.err);
      }
      mbar_wait(full + 8 * slot, (i / NSLOT) & 1, p.err);
      if (MODE == 2) {
        if (i < no) dense_slot(ring + slot * SLOT, act, 2048, T, cw, lane, acc);
        else if (i < no + 3) {   // FP32 router columns: T FMAs per weight on the CUDA cores
          const float4* wv = reinterpret_cast<const float4*>(ring + slot * SLOT) + ct;
          for (int k = 0; k < 6; ++k) {
            const float4 w = wv[k * 256];
#pragma unroll
            for (int t = 0; t < 8; ++t) if (t < T) acc[1] = fmaf(w.x + w.y, w.z + w.w, acc[1]);
          }
        } else dense_slot(ring + slot * SLOT, act, 4096, T, cw, lane, acc);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + 8 * slot);
      ++nst;
      if (MODE == 0) continue;
      if (i < no) {
        cbar();
        if (ct < 4 * T) atomicAdd(&s_mom[ct >> 2][ct & 3], 1e-3f + acc[0] * 1e-30f);
        if (i == no - 1) {   // o done: ONE moments release + spin, then s1 / h for own channels, ONE release
          cbar();
          if (ct < 4 * T) atomicAdd(p.mom1 + ct, s_mom[ct >> 2][ct & 3]);
          cbar();
          if (ct == 0) { __threadfence(); red_rel(p.cnt + CM1, 1); if (mk) mk[3] = gt(); spin_ge(p.cnt + CM1, G, p.err); if (mk) mk[4] = gt(); }
          cbar();
          for (int k = ct; k < no * 16 * T; k += 32 * NCW) {
            const int t = k % T, ch = (s + G * (k / (16 * T))) * 16 + (k / T) % 16;
            const float v = 0.9f * __bfloat162float(p.x[(size_t)t * 8192 + ch]) + rsqrtf(__ldcg(p.mom1 + t * 4) + 1e-5f);
            p.s1[(size_t)t * 8192 + ch] = v;
            p.hb[(size_t)t * 8192 + ch] = __float2bfloat16_rn(v);
          }
          __syncwarp();
          if (lane == 0) mbar_arrive(afree);
          cbar();
          if (ct == 0) { __threadfence(); red_rel(p.cnt + CH, 1); }
        }
      } else if (i == no + 2) {   // router: one red.add.v4 batch of this SM's partial logits
        cbar();
        if (ct < 64 * T) atomicAdd(reinterpret_cast<float4*>(p.logits) + ct, make_float4(1e-3f, 1e-3f, 1e-3f, acc[1] * 1e-30f));
      } else if (i >= no + 3 && ((i - no - 3) & 1)) {   // pooler granule: reduction barrier + red.add, no fence
        cbar();
        const int tile = u4 + 33 * ((i - no - 3) >> 1);
        if (ct < 16 * T) atomicAdd(p.pacc + (size_t)(ct >> 4) * 4096 + tile * 16 + (ct & 15), 1e-3f + acc[2] * 1e-30f);
      }
    }
    if (MODE >= 1) {   // pooler phase end: ONE release, one spin, the routing-tail read (logits + p), as the tail would
      cbar();
      if (ct == 0) { __threadfence(); red_rel(p.cnt + CP, 1); if (mk) mk[7] = gt(); spin_ge(p.cnt + CP, G, p.err); if (mk) mk[8] = gt(); }
      cbar();
      float v = 0.f;
      for (int k = ct; k < T * 256; k += 32 * NCW) v += __ldcg(p.logits + k);
      for (int k = s * 32 + ct; k < T * 4096; k += G * 32) v += __ldcg(p.pacc + k);
      if (v == 1234.5f) p.s1[0] = v;
    }
    if (ct == 0) { if (p.stages) p.stages[s] = nst; if (mk) mk[10] = gt(); }
  }
  __syncthreads();
  if (MODE >= 1 && tid == 0) {
    __threadfence();
    if (atomicAdd(p.cnt + CEXB, 1) == G - 1) {
      __threadfence();
      p.cnt[CM1] = 0; p.cnt[CH] = 0; p.cnt[CP] = 0;
      for (int i = 0; i < 32; ++i) p.mom1[i] = 0.f;
      __threadfence();
      p.cnt[CEXB] = 0;
    }
  }
}

template <typename K> static void set_smem(K k, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}
template <int NSLOT, int AB> constexpr int smem_c() { return NSLOT * SLOT + AB + NCW * 512 + (2 * NSLOT + 2) * 8; }
template <typename K, typename P> static void launch(K k, dim3 grid, int threads, int smem, const P& prm, bool pdl, cudaStream_t st) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, prm));
}
template <typename K> static void attrs_of(K k, int threads, int smem, std::vector<int64_t>& v) {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, k));
  set_smem(k, smem);
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, threads, smem));
  v.push_back(a.numRegs); v.push_back((int64_t)a.localSizeBytes); v.push_back((int64_t)a.sharedSizeBytes + smem); v.push_back(occ);
}
}  // namespace f2p

using namespace f2p;
#define U8(t) reinterpret_cast<const uint8_t*>(t.data_ptr())
#define BF(t) reinterpret_cast<__nv_bfloat16*>(t.data_ptr())

// mode: 0 = f2a + f2b streams only; 1 / 2 = f2a -> KS1 stand-in -> f2b with dependencies (2: + instruction stream)
void f2_run(int64_t T, int64_t mode, torch::Tensor qkv, torch::Tensor ow, torch::Tensor rw, torch::Tensor pw, torch::Tensor kc,
            torch::Tensor vc, torch::Tensor ptab, torch::Tensor x, torch::Tensor qacc, torch::Tensor qkvb, torch::Tensor oin,
            torch::Tensor mom1, torch::Tensor s1, torch::Tensor hb, torch::Tensor logits, torch::Tensor pacc, torch::Tensor cnt,
            torch::Tensor err, torch::Tensor marks, torch::Tensor stages, torch::Tensor sink, int64_t seq, int64_t spin,
            int64_t use_marks) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  long long* mk = use_marks ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  int* sg = use_marks ? stages.data_ptr<int>() : nullptr;
  PA pa{U8(qkv), BF(x), qacc.data_ptr<float>(), BF(qkvb), cnt.data_ptr<int>(), err.data_ptr<int>(), mk, sg, (int)T};
  PE pe{U8(kc), U8(vc), ptab.data_ptr<int>(), BF(qkvb), BF(oin), reinterpret_cast<unsigned*>(sink.data_ptr()), mk, (int)T,
        (int)seq, 2048, spin};
  PB pb{U8(ow), U8(rw), U8(pw), BF(oin), BF(x), mom1.data_ptr<float>(), s1.data_ptr<float>(), BF(hb), logits.data_ptr<float>(),
        pacc.data_ptr<float>(), cnt.data_ptr<int>(), err.data_ptr<int>(), mk, sg ? sg + 132 : nullptr, (int)T};
  const bool ks = mode >= 1;
  if (T <= 1) {
    constexpr int SA = smem_c<7, 2048>(), SB = smem_c<7, 4096>();
    if (mode == 0) launch(f2a_probe<7, 2048, 0>, dim3(G), NT, SA, pa, false, st);
    else if (mode == 1) launch(f2a_probe<7, 2048, 1>, dim3(G), NT, SA, pa, false, st);
    else launch(f2a_probe<7, 2048, 2>, dim3(G), NT, SA, pa, false, st);
    if (ks) launch(ks1e_probe<40064>, dim3(17, 2, 1), 256, 40064, pe, true, st);
    if (mode == 0) launch(f2b_probe<7, 4096, 0>, dim3(G), NT, SB, pb, true, st);
    else if (mode == 1) launch(f2b_probe<7, 4096, 1>, dim3(G), NT, SB, pb, true, st);
    else launch(f2b_probe<7, 4096, 2>, dim3(G), NT, SB, pb, true, st);
  } else {
    constexpr int SA = smem_c<5, 16384>(), SB = smem_c<4, 32768>();
    if (mode == 0) launch(f2a_probe<5, 16384, 0>, dim3(G), NT, SA, pa, false, st);
    else if (mode == 1) launch(f2a_probe<5, 16384, 1>, dim3(G), NT, SA, pa, false, st);
    else launch(f2a_probe<5, 16384, 2>, dim3(G), NT, SA, pa, false, st);
    if (ks) launch(ks1e_probe<72832>, dim3(9, 2, 4), 256, 72832, pe, true, st);
    if (mode == 0) launch(f2b_probe<4, 32768, 0>, dim3(G), NT, SB, pb, true, st);
    else if (mode == 1) launch(f2b_probe<4, 32768, 1>, dim3(G), NT, SB, pb, true, st);
    else launch(f2b_probe<4, 32768, 2>, dim3(G), NT, SB, pb, true, st);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> f2_attrs() {
  std::vector<int64_t> v;
  attrs_of(f2a_probe<7, 2048, 0>, NT, smem_c<7, 2048>(), v); attrs_of(f2a_probe<7, 2048, 1>, NT, smem_c<7, 2048>(), v);
  attrs_of(f2a_probe<7, 2048, 2>, NT, smem_c<7, 2048>(), v);
  attrs_of(f2b_probe<7, 4096, 0>, NT, smem_c<7, 4096>(), v); attrs_of(f2b_probe<7, 4096, 1>, NT, smem_c<7, 4096>(), v);
  attrs_of(f2b_probe<7, 4096, 2>, NT, smem_c<7, 4096>(), v);
  attrs_of(f2a_probe<5, 16384, 0>, NT, smem_c<5, 16384>(), v); attrs_of(f2a_probe<5, 16384, 1>, NT, smem_c<5, 16384>(), v);
  attrs_of(f2a_probe<5, 16384, 2>, NT, smem_c<5, 16384>(), v);
  attrs_of(f2b_probe<4, 32768, 0>, NT, smem_c<4, 32768>(), v); attrs_of(f2b_probe<4, 32768, 1>, NT, smem_c<4, 32768>(), v);
  attrs_of(f2b_probe<4, 32768, 2>, NT, smem_c<4, 32768>(), v);
  attrs_of(ks1e_probe<40064>, 256, 40064, v); attrs_of(ks1e_probe<72832>, 256, 72832, v);
  return v;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void f2_run(int64_t T, int64_t mode, torch::Tensor qkv, torch::Tensor ow, torch::Tensor rw, torch::Tensor pw, torch::Tensor kc,
            torch::Tensor vc, torch::Tensor ptab, torch::Tensor x, torch::Tensor qacc, torch::Tensor qkvb, torch::Tensor oin,
            torch::Tensor mom1, torch::Tensor s1, torch::Tensor hb, torch::Tensor logits, torch::Tensor pacc, torch::Tensor cnt,
            torch::Tensor err, torch::Tensor marks, torch::Tensor stages, torch::Tensor sink, int64_t seq, int64_t spin,
            int64_t use_marks);
std::vector<int64_t> f2_attrs();
"""

ATTR_NAMES = ["f2a T1 m0", "f2a T1 m1", "f2a T1 m2", "f2b T1 m0", "f2b T1 m1", "f2b T1 m2",
              "f2a T8 m0", "f2a T8 m1", "f2a T8 m2", "f2b T8 m0", "f2b T8 m1", "f2b T8 m2", "ks1e T1", "ks1e T8"]
CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def build(name="f2_probe", verbose=False):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CPP_SRC + CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["f2_run", "f2_attrs"],
                      extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0


def a_expected(s, G=132):
    c, u = s & 7, s >> 3
    return len(range(u, 144, 17 if c < 4 else 16))


def b_expected(s, G=132):
    return len(range(s, 512, G)) + 3 + 2 * len(range(s >> 2, 256, 33))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "f2_probe_gen.cu")
    out.write_text(CUDA_SRC, encoding="utf-8")
    print(f"wrote {out} ({len(CUDA_SRC)} chars)")

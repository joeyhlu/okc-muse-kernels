"""R-cv-1 probe (notes/memory/gpu_requests.md; notes/study/fresh_c8v2.md): the C8-F front conveyor at decode-layer sizes, with
the R-t8-1 fenced hops (LL=0, verbatim from kernels/lab/t8/c8_probe.py) next to fence-free LL hops (LL=1). LAB FILE: box runs
only (tools/box/cv_probe.py).

Why: R-t8-1 killed C8-F at c8f_d 51.6 / 65.6 us (T 1 / 8) vs a 22.6 us no-wait stream. Every hop in that skeleton was a
`__threadfence` + `red.release` issued while ~192 KB per SM of bulk copies were in flight; R-dsm-0b / 0c later priced that
fence at 4.6-6.9 us (it waits for the SM's in-flight bulk copies) while a plain dependent L2 hop is 0.5-0.9 us.

c8f_probe<NSLOT, AB, MODE, LL>: 132 CTAs (one per SM) x 576 threads: warp 0 producer (lane 0 issues cp.async.bulk, L2
evict_first), warp 1 control, warps 2..17 consumers. Stream order per SM (unchanged from R-t8-1): qkv (K-chunk s % 8, row tiles
u, u + nc, ...; q/k/v tiles 0..79 before gate tiles 80..143), o tiles s, s + 132, ..., router column slice, pooler quarter.
  - MODE 0: null consumers, no waits (the stream).
  - MODE 1, LL 0: R-t8-1's c8f_d (qkv red.add + fenced counters, 8 attention CTAs, merger CTA 0, fenced moment hop, fenced h hop).
  - MODE 1, LL 1: the same work with LL words (8 B = 4 B payload | 4 B epoch, st.relaxed.gpu / ld.relaxed.gpu, no MEMBAR):
      qkv: every split writes its 16 x T partials to its own slot; the control warp of each c = 0 SM reduces its tiles' 8 splits in
           fixed order and writes the final q/k/v/gate words (no atomics);
      attention: 8 CTAs (s < 8) wait the final q/k/v words, gather KV rows, compute spin, write (m, l, o) partials;
      merge: 16 head mergers (s = 8..23) wait their head's 8 partials + gate words, write o-input (bf16x2 words);
      o: consumers stage the o-input themselves (LL loads -> smem); moments as per-SM words, every SM sums all 132;
      h: own channels as bf16x2 words; consumers stage the pooler quarter + router slice at the router start;
      router / pooler partials: per-SM LL words (no readers here, cost only).
    The epoch is read once per launch from device memory and bumped by the last CTA out, so the same graph replays correctly.
Bounded waits everywhere (err flags, no trap). Marks (%globaltimer) per CTA when marks != nullptr.
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

namespace c8v {
typedef unsigned long long u64;
constexpr unsigned FULL = 0xffffffffu;
constexpr int SLOTF = 24576, NCW = 16, NT = 32 * (NCW + 2), G = 132, NCT = 32 * NCW;
constexpr int RSL = 63488;               // router column slice per SM: 256 x 62 FP32
constexpr long long BUDGET = 1ll << 31;  // clock64 cycles per wait (~1 s) -> err flag, no trap
enum { FQ0 = 0, FQ1 = 1, FQG = 2, FATT = 3, FOIN = 4, FM1 = 5, FH = 6, FEXIT = 7, NCF = 8 };

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
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory"); }   // consumer warps only

// ---- LL words: low 32 bits payload, high 32 bits epoch. One aligned 64-bit relaxed access = single-copy atomic; no fences.
__device__ __forceinline__ void st_ll(u64* p, uint32_t v, uint32_t ep) {
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" :: "l"(p), "l"(((u64)ep << 32) | (u64)v) : "memory");
}
__device__ __forceinline__ u64 ld_ll(const u64* p) {
  u64 w; asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(w) : "l"(p) : "memory"); return w;
}
__device__ __noinline__ u64 spin_ll(const u64* p, uint32_t ep, int* err) {
  const long long t0 = clock64();
  for (int k = 0;; ++k) {
    const u64 w = ld_ll(p);
    if ((uint32_t)(w >> 32) == ep) return w;
    if (k > 32) __nanosleep(20);
    if (clock64() - t0 > BUDGET) { atomicExch(err, 1); atomicExch(err + 3, 1); return (u64)ep << 32; }
  }
}
// n words, thread ti of nthr; addr(i) -> word pointer; use(i, payload). 8 loads in flight per thread, then the checks.
template <class A, class F>
__device__ __forceinline__ void ll_read(int n, int ti, int nthr, uint32_t ep, int* err, A addr, F use) {
  for (int i0 = ti; i0 < n; i0 += 8 * nthr) {
    u64 w[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { const int i = i0 + k * nthr; w[k] = (i < n) ? ld_ll(addr(i)) : 0ull; }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = i0 + k * nthr;
      if (i < n) {
        if ((uint32_t)(w[k] >> 32) != ep) w[k] = spin_ll(addr(i), ep, err);
        use(i, (uint32_t)w[k]);
      }
    }
  }
}
// R-cv-2: the same read, but a stale batch is re-polled together (one round trip per pass for all 8 words), not word by word
template <class A, class F>
__device__ __forceinline__ void ll_read2(int n, int ti, int nthr, uint32_t ep, int* err, A addr, F use) {
  for (int i0 = ti; i0 < n; i0 += 8 * nthr) {
    u64 w[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { const int i = i0 + k * nthr; w[k] = (i < n) ? ld_ll(addr(i)) : ((u64)ep << 32); }
    const long long t0 = clock64();
    for (int pass = 0;; ++pass) {
      bool ok = true;
#pragma unroll
      for (int k = 0; k < 8; ++k) ok &= (uint32_t)(w[k] >> 32) == ep;
      if (ok) break;
      if (clock64() - t0 > BUDGET) { atomicExch(err, 1); atomicExch(err + 3, 1); break; }
      if (pass > 32) __nanosleep(20);
#pragma unroll
      for (int k = 0; k < 8; ++k) { const int i = i0 + k * nthr; if ((uint32_t)(w[k] >> 32) != ep) w[k] = ld_ll(addr(i)); }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) { const int i = i0 + k * nthr; if (i < n) use(i, (uint32_t)w[k]); }
  }
}
template <int LL, class A, class F>
__device__ __forceinline__ void ll_rd(int n, int ti, int nthr, uint32_t ep, int* err, A addr, F use) {
  if constexpr (LL >= 2) ll_read2(n, ti, nthr, ep, err, addr, use);
  else ll_read(n, ti, nthr, ep, err, addr, use);
}
__device__ __forceinline__ uint32_t pk2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

struct PF {
  const uint8_t* qkv; const uint8_t* ow; const uint8_t* rw; const uint8_t* pw;   // [8][144][SLOTF], [512][SLOTF], [132][RSL], [4][256][2][SLOTF]
  const uint8_t* kc; const uint8_t* vc; const int* ptab;                          // [npages][128] each, [npages]
  const __nv_bfloat16* x;                                                         // [T][8192]
  float* qacc; float* part; __nv_bfloat16* oin; float* mom1;                      // [T][2304], [8][T][16][66], [T][1024], [8][4]
  float* s1; __nv_bfloat16* hb; float* logits; float* pacc;                       // [T][8192] x2, [T][256], [T][4096]
  u64* qp; u64* qf; u64* ap; u64* oi; u64* mp; u64* hl; u64* lp; u64* pp;        // LL words (see c8f_run)
  unsigned* ep;
  int* cnt; int* err; long long* marks; int* stages; unsigned* sink;
  int T, seq; long long spin;
};

template <int NSLOT, int AB, int MODE, int LL>
__global__ void __launch_bounds__(NT, 1) c8f_probe(const __grid_constant__ PF p) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* ring = sm;
  uint8_t* act = sm + NSLOT * SLOTF;
  uint64_t* bars = reinterpret_cast<uint64_t*>(act + AB);   // full, empty, abar, afree
  __shared__ float s_mom[8][4], s_tot[8][4];
  __shared__ uint32_t s_ep;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, s = blockIdx.x, T = p.T;
  long long* mk = p.marks ? p.marks + (size_t)s * 16 : nullptr;
  if (tid == 0 && mk) { mk[0] = gt(); mk[15] = smid(); }
  const uint32_t full = sa(bars), empty = sa(bars + NSLOT), abar = sa(bars + 2 * NSLOT), afree = sa(bars + 2 * NSLOT + 1);
  if (tid == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(full + 8 * i, 1); mbar_init(empty + 8 * i, NCW); }
    mbar_init(abar, 1);
    mbar_init(afree, NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (LL) { unsigned e; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(e) : "l"(p.ep) : "memory"); s_ep = e + 1u; }
  }
  if (tid < 32) { (&s_mom[0][0])[tid] = 0.f; (&s_tot[0][0])[tid] = 0.f; }
  __syncthreads();
  const uint32_t ep = LL ? s_ep : 0u;
  const int c = s & 7, nc = (c < 4) ? 17 : 16, u = s >> 3;
  const int nq = (143 - u) / nc + 1;                       // qkv tiles u, u + nc, ... (< 144)
  const int nqv = (79 - u) / nc + 1;                       // of which q/k/v tiles (< 80)
  const int no = (511 - s) / G + 1;                        // o tiles s, s + 132, ... (< 512)
  const int qq = s & 3, u4 = s >> 2;
  const int npl = (255 - u4) / 33 + 1;                     // pooler tiles u4, u4 + 33, ... (< 256)
  const int nsl = nq + no + 3 + 2 * npl;
  if (warp == 0) {   // ---------------- producer
    if (lane == 0) {
      const uint64_t pol = evict_first();
      long long wt = 0;
      for (int i = 0; i < nsl; ++i) {
        const uint8_t* src;
        uint32_t bytes = SLOTF;
        if (i < nq) src = p.qkv + ((size_t)c * 144 + u + nc * i) * SLOTF;
        else if (i < nq + no) src = p.ow + (size_t)(s + G * (i - nq)) * SLOTF;
        else if (i < nq + no + 3) { const int j = i - nq - no; src = p.rw + (size_t)s * RSL + (size_t)j * SLOTF; bytes = j < 2 ? SLOTF : RSL - 2 * SLOTF; }
        else { const int j = i - nq - no - 3; src = p.pw + (((size_t)qq * 256 + u4 + 33 * (j >> 1)) * 2 + (j & 1)) * SLOTF; }
        const int slot = i % NSLOT;
        const long long t0 = mk ? gt() : 0;
        mbar_wait(empty + 8 * slot, ((i / NSLOT) & 1) ^ 1, p.err);
        if (mk) wt += gt() - t0;
        mbar_tx(full + 8 * slot, bytes);
        bulk(sa(ring + slot * SLOTF), src, bytes, full + 8 * slot, pol);
        if (i == 0 && mk) mk[1] = gt();
      }
      if (mk) mk[11] = wt;
    }
  } else if (warp == 1) {   // ---------------- control
    if (MODE >= 1 && lane == 0) {
      mbar_tx(abar, T * 2048);
      for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 2048), p.x + (size_t)t * 8192 + c * 1024, 2048, abar);
    }
    if (MODE >= 1 && !LL && lane == 0) {   // R-t8-1: o-input (after cnt_oin), then h quarter (after cnt_h)
      spin_ge(p.cnt + FOIN, 1, p.err);
      if (mk) mk[6] = gt();
      for (int i = s; i < T * 2304; i += G) p.qacc[i] = 0.f;   // qacc is dead after the o-input: zero this SM's share
      mbar_wait(afree, 0, p.err);
      fence_async();
      mbar_tx(abar, T * 2048);
      for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 2048), p.oin + (size_t)t * 1024, 2048, abar);
      spin_ge(p.cnt + FH, G, p.err);
      if (mk) mk[9] = gt();
      mbar_wait(afree, 1, p.err);
      fence_async();
      mbar_tx(abar, T * 4096);
      for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 4096), p.hb + (size_t)t * 8192 + qq * 2048, 4096, abar);
    }
    if (MODE >= 1 && LL >= 2) {   // R-cv-2 reducer: tiles s and s + 132 (one per SM), the 8 splits re-polled together
      for (int tile = s; tile < 144; tile += G) {
        for (int e = lane; e < 16 * T; e += 32) {
          float acc = 0.f;
          ll_read2(8, 0, 1, ep, p.err, [&](int k) { return p.qp + ((size_t)(k * 144 + tile) * T) * 16 + e; },
                   [&](int k, uint32_t v) { acc += __uint_as_float(v); });
          st_ll(p.qf + (size_t)tile * T * 16 + e, __float_as_uint(acc), ep);
        }
      }
      if (lane == 0 && mk) mk[12] = gt();
    }
    if (MODE >= 1 && LL == 1 && c == 0) {   // reducer: this SM's tiles u, u + 17, ...: 8 split partials in fixed order -> final words
      for (int tile = u; tile < 144; tile += 17) {
        for (int e = lane; e < 16 * T; e += 32) {
          u64 w[8];
#pragma unroll
          for (int k = 0; k < 8; ++k) w[k] = ld_ll(p.qp + ((size_t)(k * 144 + tile) * T) * 16 + e);
          float acc = 0.f;
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            if ((uint32_t)(w[k] >> 32) != ep) w[k] = spin_ll(p.qp + ((size_t)(k * 144 + tile) * T) * 16 + e, ep, p.err);
            acc += __uint_as_float((uint32_t)w[k]);
          }
          st_ll(p.qf + (size_t)tile * T * 16 + e, __float_as_uint(acc), ep);
        }
      }
      if (lane == 0 && mk) mk[12] = gt();
    }
  } else {   // ---------------- consumers
    const int ct = tid - 64;
    uint4 kv[8];
    const bool att = MODE >= 1 && s < 8;
    if (att) {   // KV window rows for this split: 256 keys, K (even threads) or V (odd threads), 128 B rows (both heads)
      const int key = ct >> 1, pos = p.seq - 2048 + s * 256 + key;
      const int pg = p.ptab[pos < 0 ? 0 : pos];
      const uint4* row = reinterpret_cast<const uint4*>(((ct & 1) ? p.vc : p.kc) + (size_t)pg * 128);
#pragma unroll
      for (int k = 0; k < 8; ++k) kv[k] = __ldg(row + k);
    }
    int nst = 0;
    unsigned sink = 0u;
    uint32_t* actw = reinterpret_cast<uint32_t*>(act);
    for (int i = 0; i < nsl; ++i) {
      const int slot = i % NSLOT;
      if (MODE >= 1) {
        if (i == 0) mbar_wait(abar, 0, p.err);                       // x chunk
        if (!LL) {
          if (i == nq) mbar_wait(abar, 1, p.err);                    // o-input
          if (i == nq + no + 3) mbar_wait(abar, 0, p.err);           // h quarter (third completion)
        } else {
          if (i == nq) {   // o-input: 16 heads x 32 bf16x2 words per token, staged by the consumers themselves
            cbar();
            ll_rd<LL>(512 * T, ct, NCT, ep, p.err, [&](int k) { return p.oi + k; }, [&](int k, uint32_t v) { actw[k] = v; });
            cbar();
            if (ct == 0 && mk) mk[6] = gt();
          }
          if (i == nq + no) {   // h: pooler quarter (1024 bf16x2 words per token) + router column slice (31 words per token)
            cbar();
            ll_rd<LL>(1024 * T, ct, NCT, ep, p.err,
                    [&](int k) { return p.hl + (size_t)(k >> 10) * 4096 + qq * 1024 + (k & 1023); }, [&](int k, uint32_t v) { actw[k] = v; });
            ll_rd<LL>(31 * T, ct, NCT, ep, p.err,
                    [&](int k) { return p.hl + (size_t)(k / 31) * 4096 + s * 31 + (k % 31); }, [&](int k, uint32_t v) { sink ^= v; });
            cbar();
            if (ct == 0 && mk) mk[9] = gt();
          }
        }
      }
      mbar_wait(full + 8 * slot, (i / NSLOT) & 1, p.err);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + 8 * slot);
      ++nst;
      if (ct == 0 && mk) { if (i == nqv - 1) mk[13] = gt(); if (i == nq - 1) mk[14] = gt(); }   // R-cv-3: slot landed + consumed
      if (MODE == 0) continue;
      if (i < nq) {   // qkv granule epilogue
        const int tile = u + nc * i;
        if (!LL) {   // R-t8-1: reduction barrier, red.add 16 x T partials, group counter
          cbar();
          if (ct < 16 * T) atomicAdd(p.qacc + (size_t)(ct / 16) * 2304 + tile * 16 + (ct & 15), 1e-30f);
          cbar();
          if (ct == 0) { __threadfence(); red_rel(p.cnt + (tile < 40 ? FQ0 : tile < 80 ? FQ1 : FQG), 1); }
        } else {     // reduction barrier, 16 x T partial words into this split's slot
          cbar();
          if (ct < 16 * T) st_ll(p.qp + ((size_t)(c * 144 + tile) * T + ct / 16) * 16 + (ct & 15), __float_as_uint((c + 1) * 1e-3f), ep);
        }
        if (att && i == nqv - 1) {   // attention after this SM's last q/k/v tile
          float qs = 0.f;
          if (!LL) {
            if (ct == 0) { if (mk) mk[2] = gt(); spin_ge(p.cnt + FQ0, 320, p.err); spin_ge(p.cnt + FQ1, 320, p.err); if (mk) mk[3] = gt(); }
            cbar();
            for (int k = ct; k < 1280 * T; k += NCT) qs += __ldcg(p.qacc + (k / 1280) * 2304 + (k % 1280));
          } else {   // final q/k/v words of tiles 0..79 ([tile][t][16] -> contiguous)
            if (ct == 0 && mk) mk[2] = gt();
            ll_rd<LL>(1280 * T, ct, NCT, ep, p.err, [&](int k) { return p.qf + k; }, [&](int k, uint32_t v) { qs += __uint_as_float(v); });
            cbar();
            if (ct == 0 && mk) mk[3] = gt();
          }
#pragma unroll
          for (int k = 0; k < 8; ++k) sink ^= kv[k].x ^ kv[k].y ^ kv[k].z ^ kv[k].w;
          const long long t0 = clock64();
          while (clock64() - t0 < p.spin) {}
          if (!LL) {
            float* pp = p.part + (size_t)s * T * 16 * 66;
            for (int k = ct; k < T * 16 * 66; k += NCT) pp[k] = qs * 1e-30f + (float)k;
            cbar();
            if (ct == 0) { __threadfence(); red_rel(p.cnt + FATT, 1); if (mk) mk[4] = gt(); }
          } else {
            u64* pp = p.ap + (size_t)s * T * 16 * 66;
            for (int k = ct; k < T * 16 * 66; k += NCT) st_ll(pp + k, __float_as_uint(qs * 1e-30f + (float)k), ep);
            if (ct == 0 && mk) mk[4] = gt();
          }
        }
        if (!LL && att && s == 0 && i == nq - 1) {   // R-t8-1 merger: after its gate tiles, the 8 partials and the gate rows -> o-input
          if (ct == 0) { spin_ge(p.cnt + FATT, 8, p.err); spin_ge(p.cnt + FQG, 512, p.err); }
          cbar();
          float acc = 0.f;
          for (int k = ct; k < 8 * T * 16 * 66; k += NCT) acc += __ldcg(p.part + k);
          for (int k = ct; k < T * 1024; k += NCT) {
            const float gv = __ldcg(p.qacc + (k / 1024) * 2304 + 1280 + (k % 1024));
            p.oin[k] = __float2bfloat16_rn(acc * 1e-30f + gv);
          }
          cbar();
          if (ct == 0) { __threadfence(); red_rel(p.cnt + FOIN, 1); if (mk) mk[5] = gt(); }
        }
        if (LL && s >= 8 && s < 24 && i == nq - 1) {   // head merger h = s - 8: 8 partials of head h + its 64 gate rows
          const int h = s - 8, n1 = T * 66;
          float acc = 0.f, gs = 0.f;
          ll_rd<LL>(8 * n1, ct, NCT, ep, p.err,
                  [&](int k) { const int a = k / n1, r = k % n1; return p.ap + ((size_t)(a * T + r / 66) * 16 + h) * 66 + r % 66; },
                  [&](int k, uint32_t v) { acc += __uint_as_float(v); });
          ll_rd<LL>(64 * T, ct, NCT, ep, p.err,
                  [&](int k) { const int j = k / (16 * T), r = k % (16 * T); return p.qf + (size_t)(80 + 4 * h + j) * T * 16 + r; },
                  [&](int k, uint32_t v) { gs += __uint_as_float(v); });
          cbar();   // (a real merge reduces (m, l, o) across threads here)
          if (ct < 32 * T) st_ll(p.oi + (size_t)(ct >> 5) * 512 + h * 32 + (ct & 31), pk2(acc * 1e-30f + 1.f, gs * 1e-30f + 1.f), ep);
          if (ct == 0 && mk) mk[5] = gt();
        }
        if (i == nq - 1) { __syncwarp(); if (lane == 0) mbar_arrive(afree); }   // x chunk released
      } else if (i < nq + no) {   // o granule: moment partials (16 rows x T) into s_mom
        cbar();
        if (ct < 4 * T) atomicAdd(&s_mom[ct >> 2][ct & 3], 1e-3f);
        if (i == nq + no - 1) {   // end of the o phase: moments -> hop, then s1 / h for own channels -> hop
          cbar();
          if (!LL) {
            for (int k = ct; k < T * 256; k += NCT) if ((k % G) == s) p.logits[k] = 0.f;   // distributed zeroing
            for (int k = s + G * ct; k < T * 4096; k += G * NCT) p.pacc[k] = 0.f;
            if (ct < 4 * T) atomicAdd(p.mom1 + ct, s_mom[ct >> 2][ct & 3]);
            cbar();
            if (ct == 0) { __threadfence(); red_rel(p.cnt + FM1, 1); if (mk) mk[7] = gt(); spin_ge(p.cnt + FM1, G, p.err); if (mk) mk[8] = gt(); }
            cbar();
            for (int k = ct; k < no * 16 * T; k += NCT) {
              const int t = k % T, ch = (s + G * (k / (16 * T))) * 16 + (k / T) % 16;
              const float m0 = __ldcg(p.mom1 + t * 4), m3 = __ldcg(p.mom1 + t * 4 + 3);
              const float v = 0.9f * __bfloat162float(p.x[(size_t)t * 8192 + ch]) + rsqrtf(m0 + m3 + 1e-5f);
              p.s1[(size_t)t * 8192 + ch] = v;
              p.hb[(size_t)t * 8192 + ch] = __float2bfloat16_rn(v);
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(afree);   // o-input released
            cbar();
            if (ct == 0) { __threadfence(); red_rel(p.cnt + FH, 1); }
          } else {
            if (ct < 4 * T) st_ll(p.mp + (size_t)s * T * 4 + ct, __float_as_uint(s_mom[ct >> 2][ct & 3]), ep);
            if (ct == 0 && mk) mk[7] = gt();
            ll_rd<LL>(G * 4 * T, ct, NCT, ep, p.err, [&](int k) { return p.mp + k; },
                    [&](int k, uint32_t v) { const int r = k % (4 * T); atomicAdd(&s_tot[r >> 2][r & 3], __uint_as_float(v)); });
            cbar();
            if (ct == 0 && mk) mk[8] = gt();
            for (int k = ct; k < no * 8 * T; k += NCT) {   // channel pairs of the own o tiles
              const int t = k % T, pr = k / T, ch = (s + G * (pr >> 3)) * 16 + 2 * (pr & 7);
              const float r = rsqrtf(s_tot[t][0] + s_tot[t][3] + 1e-5f);
              const float v0 = 0.9f * __bfloat162float(p.x[(size_t)t * 8192 + ch]) + r;
              const float v1 = 0.9f * __bfloat162float(p.x[(size_t)t * 8192 + ch + 1]) + r;
              p.s1[(size_t)t * 8192 + ch] = v0;
              p.s1[(size_t)t * 8192 + ch + 1] = v1;
              st_ll(p.hl + (size_t)t * 4096 + (ch >> 1), pk2(v0, v1), ep);
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(afree);
          }
        }
      } else if (i < nq + no + 3) {   // router slice: partial logits after the last router stage
        if (i == nq + no + 2) {
          cbar();
          if (!LL) { if (ct < 64 * T) atomicAdd(reinterpret_cast<float4*>(p.logits) + ct, make_float4(1e-3f, 1e-3f, 1e-3f, 1e-3f)); }
          else { for (int k = ct; k < 256 * T; k += NCT) st_ll(p.lp + (size_t)s * 256 * T + k, __float_as_uint(1e-3f), ep); }
        }
      } else {   // pooler granule (2 slots): reduction barrier + 16 x T partials
        const int j = i - nq - no - 3;
        if (j & 1) {
          cbar();
          const int tile = u4 + 33 * (j >> 1);
          if (!LL) { if (ct < 16 * T) atomicAdd(p.pacc + (size_t)(ct / 16) * 4096 + tile * 16 + (ct & 15), 1e-3f); }
          else if (ct < 16 * T) st_ll(p.pp + ((size_t)(qq * 256 + tile) * T + ct / 16) * 16 + (ct & 15), __float_as_uint(1e-3f), ep);
        }
      }
    }
    if (att && sink == 0x12345678u) p.sink[0] = sink;
    if (LL && sink == 0x12345679u) p.sink[1] = sink;
    if (ct == 0) { if (p.stages) p.stages[s] = nst; if (mk) mk[10] = gt(); }
  }
  __syncthreads();
  if (MODE >= 1 && tid == 0) {
    if (!LL) {
      __threadfence();
      const int old = atomicAdd(p.cnt + FEXIT, 1);
      if (old == G - 1) {
        __threadfence();
        for (int i = 0; i < FEXIT; ++i) p.cnt[i] = 0;
        for (int i = 0; i < 32; ++i) p.mom1[i] = 0.f;
        __threadfence();
        p.cnt[FEXIT] = 0;
      }
    } else {   // the last CTA out publishes the next epoch (visible to the next launch at the kernel boundary)
      const int old = atomicAdd(p.cnt + FEXIT, 1);
      if (old == G - 1) { p.cnt[FEXIT] = 0; *p.ep = ep; }
    }
  }
}

template <typename K> static void set_smem(K k, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}
template <int NSLOT, int AB> constexpr int smem_f() { return NSLOT * SLOTF + AB + (2 * NSLOT + 2) * 8; }

template <int NSLOT, int AB, int MODE, int LL> static void launch_f(const PF& pf, cudaStream_t st) {
  constexpr int SM = smem_f<NSLOT, AB>();
  static bool once = false;
  if (!once) { set_smem(c8f_probe<NSLOT, AB, MODE, LL>, SM); once = true; }
  c8f_probe<NSLOT, AB, MODE, LL><<<G, NT, SM, st>>>(pf);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <typename K> static void attrs_of(K k, int smem, std::vector<int64_t>& v) {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, k));
  set_smem(k, smem);
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, NT, smem));
  v.push_back(a.numRegs); v.push_back((int64_t)a.localSizeBytes); v.push_back((int64_t)a.sharedSizeBytes + smem); v.push_back(occ);
}
}  // namespace c8v

using namespace c8v;
#define U8(t) reinterpret_cast<const uint8_t*>(t.data_ptr())
#define W64(t) reinterpret_cast<u64*>(t.data_ptr())

void c8f_run(int64_t T, int64_t mode, int64_t ll, torch::Tensor qkv, torch::Tensor ow, torch::Tensor rw, torch::Tensor pw,
             torch::Tensor kc, torch::Tensor vc, torch::Tensor ptab, torch::Tensor x, torch::Tensor qacc, torch::Tensor part,
             torch::Tensor oin, torch::Tensor mom1, torch::Tensor s1, torch::Tensor hb, torch::Tensor logits, torch::Tensor pacc,
             torch::Tensor llbuf, torch::Tensor epw, torch::Tensor cnt, torch::Tensor err, torch::Tensor marks, torch::Tensor stages,
             torch::Tensor sink, int64_t seq, int64_t spin, int64_t use_marks) {
  PF pf;
  pf.qkv = U8(qkv); pf.ow = U8(ow); pf.rw = U8(rw); pf.pw = U8(pw); pf.kc = U8(kc); pf.vc = U8(vc); pf.ptab = ptab.data_ptr<int>();
  pf.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  pf.qacc = qacc.data_ptr<float>(); pf.part = part.data_ptr<float>(); pf.oin = reinterpret_cast<__nv_bfloat16*>(oin.data_ptr());
  pf.mom1 = mom1.data_ptr<float>(); pf.s1 = s1.data_ptr<float>(); pf.hb = reinterpret_cast<__nv_bfloat16*>(hb.data_ptr());
  pf.logits = logits.data_ptr<float>(); pf.pacc = pacc.data_ptr<float>();
  // LL regions inside llbuf (u64 words, sized for T = 8): qp [8][144][T][16], qf [144][T][16], ap [8][T][16][66], oi [T][512],
  // mp [132][T][4], hl [T][4096], lp [132][T][256], pp [4][256][T][16]
  u64* b = W64(llbuf);
  const size_t t8 = 8;
  pf.qp = b; b += 8 * 144 * t8 * 16;
  pf.qf = b; b += 144 * t8 * 16;
  pf.ap = b; b += 8 * t8 * 16 * 66;
  pf.oi = b; b += t8 * 512;
  pf.mp = b; b += 132 * t8 * 4;
  pf.hl = b; b += t8 * 4096;
  pf.lp = b; b += 132 * t8 * 256;
  pf.pp = b; b += 4 * 256 * t8 * 16;
  TORCH_CHECK((int64_t)(b - W64(llbuf)) <= llbuf.numel(), "llbuf too small");
  pf.ep = reinterpret_cast<unsigned*>(epw.data_ptr());
  pf.cnt = cnt.data_ptr<int>(); pf.err = err.data_ptr<int>();
  pf.marks = use_marks ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  pf.stages = use_marks ? stages.data_ptr<int>() : nullptr;
  pf.sink = reinterpret_cast<unsigned*>(sink.data_ptr());
  pf.T = (int)T; pf.seq = (int)seq; pf.spin = spin;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (T <= 1) {
    if (mode == 0) launch_f<8, 4096, 0, 0>(pf, st); else if (!ll) launch_f<8, 4096, 1, 0>(pf, st);
    else if (ll == 1) launch_f<8, 4096, 1, 1>(pf, st); else launch_f<8, 4096, 1, 2>(pf, st);
  } else {
    if (mode == 0) launch_f<7, 32768, 0, 0>(pf, st); else if (!ll) launch_f<7, 32768, 1, 0>(pf, st);
    else if (ll == 1) launch_f<7, 32768, 1, 1>(pf, st); else launch_f<7, 32768, 1, 2>(pf, st);
  }
}

int64_t ll_words() { return 8 * 144 * 8 * 16 + 144 * 8 * 16 + 8 * 8 * 16 * 66 + 8 * 512 + 132 * 8 * 4 + 8 * 4096 + 132 * 8 * 256 + 4 * 256 * 8 * 16; }

std::vector<int64_t> c8v_attrs() {
  std::vector<int64_t> v;
  attrs_of(c8f_probe<8, 4096, 0, 0>, smem_f<8, 4096>(), v); attrs_of(c8f_probe<8, 4096, 1, 0>, smem_f<8, 4096>(), v);
  attrs_of(c8f_probe<8, 4096, 1, 1>, smem_f<8, 4096>(), v); attrs_of(c8f_probe<8, 4096, 1, 2>, smem_f<8, 4096>(), v);
  attrs_of(c8f_probe<7, 32768, 0, 0>, smem_f<7, 32768>(), v); attrs_of(c8f_probe<7, 32768, 1, 0>, smem_f<7, 32768>(), v);
  attrs_of(c8f_probe<7, 32768, 1, 1>, smem_f<7, 32768>(), v); attrs_of(c8f_probe<7, 32768, 1, 2>, smem_f<7, 32768>(), v);
  return v;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void c8f_run(int64_t T, int64_t mode, int64_t ll, torch::Tensor qkv, torch::Tensor ow, torch::Tensor rw, torch::Tensor pw,
             torch::Tensor kc, torch::Tensor vc, torch::Tensor ptab, torch::Tensor x, torch::Tensor qacc, torch::Tensor part,
             torch::Tensor oin, torch::Tensor mom1, torch::Tensor s1, torch::Tensor hb, torch::Tensor logits, torch::Tensor pacc,
             torch::Tensor llbuf, torch::Tensor epw, torch::Tensor cnt, torch::Tensor err, torch::Tensor marks, torch::Tensor stages,
             torch::Tensor sink, int64_t seq, int64_t spin, int64_t use_marks);
int64_t ll_words();
std::vector<int64_t> c8v_attrs();
"""

ATTR_NAMES = ["c8f T1 m0", "c8f T1 m1", "c8f T1 m1 LL", "c8f T1 m1 LL2", "c8f T8 m0", "c8f T8 m1", "c8f T8 m1 LL", "c8f T8 m1 LL2"]
CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def build(name="c8v2_probe", verbose=False):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CPP_SRC + CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                      functions=["c8f_run", "ll_words", "c8v_attrs"], extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0


def f_expected_stages(s, G=132):
    c, u = s & 7, s >> 3
    nc = 17 if c < 4 else 16
    nq = len(range(u, 144, nc))
    no = len(range(s, 512, G))
    npl = len(range(s >> 2, 256, 33))
    return nq + no + 3 + 2 * npl


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "c8v2_probe_gen.cu")
    out.write_text(CUDA_SRC, encoding="utf-8")
    print(f"wrote {out} ({len(CUDA_SRC)} chars)")

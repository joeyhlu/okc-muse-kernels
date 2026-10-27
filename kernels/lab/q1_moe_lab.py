"""Q1: one-launch two-wave MoE for T <= 8 (notes/study/moe_structure_fresh.md), step-0 lab.

One persistent launch per call, G = SMs x 2 CTAs (all resident), 320 threads: warps 0-7 consumers, warp 8 producer,
warp 9 epilogue. Weights are K2A's stage-repacked tensors (ws13 / ws2, 128-row items x 16 K-stages of 18432 B) and the
consumer is K2A v3.3's NC1 inner loop (deq4 / bmul2 / consume_kb, two mma chains), so any time difference against K2A is
the skeleton, not the math.

Skeleton (what is new):
- lean prologue: <= 64 ids -> counts / offsets / route lists / touched order in one warp (no smem atomics);
- two waves (EPI): touched experts split into A (first n - max(1, n/3)) and B (the rest); every CTA runs its static
  Stream-K share of W13(A), W13(B), W2(A), W2(B) back to back in one ring. W2(A)'s inter dependency was met a whole
  W13(B) share earlier, W2(B)'s during W2(A): the stream never drains between W13 and W2;
- W2 weight copies are issued ahead of the dependency; only the inter (B operand) copies are deferred (bounded FIFO);
- W13 split items: K2A's h atomics + last-contributor SwiGLU finalize on the epilogue warp;
- W2: whole items store y, split items atomically add (no split finalize); the post-norm sums of squares are computed
  by the finalize from the full y rows (one L2 batch), so W2's end is not a finalize chain;
- finalize in the same launch: CTA b < 4M takes (token b / 4, quarter b % 4) after one monotonic counter.
Counters are monotonic (targets from a per-call generation ticket), split-item counters and h self-reset; bounded waits
set an error flag and never trap.

mode bits (lab arms): 1 null consumers (no math / flush), 2 no X / inter copies, 4 no dependency waits, 8 no finalize.
Builds for sm_90a (bulk copies + expect_tx) and sm_80/86 (cp.async + noinc arrivals; local correctness only).
"""
import hashlib
import statistics
import time

import torch

Q1_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace q1 {
constexpr int KDIM = 4096;
constexpr int SW = 18432;            // weight bytes per stage: 128 rows x 256 K (codes 16384 + scales 2048)
constexpr int XROW = 528;            // staged B row: 256 bf16 + 16 B pad (conflict-free LDS.128)
constexpr int XST = XROW / 2;
constexpr int MMAX = 8;              // tokens per call
constexpr int PMAX = 8 * MMAX;       // ids per call = max routes
constexpr int RS = PMAX;             // route stride of the inter buffer [16 K-stages][RS routes][XST]
constexpr int STG = SW + 8 * XROW;   // one slot: weights + up to 8 B rows
constexpr unsigned FULL = 0xffffffffu;
constexpr long long WAIT_BUDGET = 1ll << 30;   // clock64 cycles per wait (~0.5 s) -> error flag
#if __CUDA_ARCH__ >= 900
constexpr int FULLCNT = 1;           // lane 0's arrive.expect_tx
#else
constexpr int FULLCNT = 32;          // 32 producer lanes' cp.async noinc arrivals
#endif

struct Args {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint8_t* ws13; const uint8_t* ws2;
  const float* gsc13; const float* gsc2; const float* nw;
  float* h; __nv_bfloat16* inter; float* y; int* cnt; unsigned* sync; int* err;
  __nv_bfloat16* out; long long* marks;
  int M, lo, mode;
};

struct Plan {
  int n, nA, R;
  int grp[32], cnt[32], off[32];
};

// ---------------- math (K2A v3.3, unchanged) ----------------
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
// One kb (16 B of codes per lane) of this warp's 16-row tile; B from the staged rows (K2A consume_kb<1>).
__device__ __forceinline__ void consume_kb1(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow,
                                            int kloc, int t, float* acc) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  const uint4* xp = reinterpret_cast<const uint4*>(xrow + kloc + 16 * t);
  const uint4 x0 = xp[0], x1 = xp[1];
  mma16816(acc, ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
  mma16816(acc, ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
  mma16816(acc, ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
  mma16816(acc, ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
}

// ---------------- barriers / copies ----------------
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
#endif
  return ok != 0;
}
__device__ __forceinline__ void set_err(int* err, int code) {
  if (atomicExch(err, 1) == 0) atomicExch(err + 1, code);
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, int code) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  int n = 0;
  while (!mbar_try(b, par)) {
    if ((++n & 255) == 0) {
      if (*reinterpret_cast<volatile int*>(err) != 0) return;
      if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return; }
    }
  }
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ bool reached(unsigned v, unsigned tgt) { return (int)(v - tgt) >= 0; }
// bounded spin until *p >= tgt (monotonic counter); false on timeout / error
__device__ __forceinline__ bool spin_ge(const unsigned* p, unsigned tgt, int* err, int code) {
  if (reached(ld_acq(p), tgt)) return true;
  const long long t0 = clock64();
  while (true) {
    __nanosleep(32);
    if (reached(ld_acq(p), tgt)) return true;
    if (*reinterpret_cast<volatile int*>(err) != 0) return false;
    if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return false; }
  }
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void fence_proxy_async() {
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async;\n" ::: "memory");
#endif
}
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
#else
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void warp_copy(int lane, uint32_t dst, const void* src, uint32_t bytes) {
  const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
  for (uint32_t o = 16u * lane; o < bytes; o += 512u) cpa16(dst + o, s + o);
}
#endif

// ---------------- plan ----------------
// phase p: 0 W13(A), 1 W13(B), 2 W2(A), 3 W2(B); this CTA's static Stream-K share [beg, end) of the phase's stages
__device__ __forceinline__ void phase_range(const Plan& S, int p, int b, int G, int& g0, int& nrb, int& beg, int& end) {
  const bool a = (p == 0 || p == 2);
  g0 = a ? 0 : S.nA;
  const int ng = a ? S.nA : S.n - S.nA;
  nrb = (p < 2) ? 64 : 32;
  const long long sp = (long long)ng * nrb * 16;
  beg = (int)(sp * b / G);
  end = (int)(sp * (b + 1) / G);
}

// ---------------- the kernel ----------------
template <int NST>
__global__ void __launch_bounds__(320, 2) q1_kernel(const Args A) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_ids[PMAX];
  __shared__ int16_t s_list[PMAX];   // route c -> id position p (token p >> 3)
  __shared__ int16_t s_rt[PMAX];     // id position p -> route c, or -1
  __shared__ Plan S;
  __shared__ int s_gen;
  __shared__ float s_red[8][8];
  __shared__ float s_sc[8];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = blockIdx.x, G = gridDim.x;
  const int M = A.M, NP = 8 * A.M;
  const int mode = A.mode;
  long long* mk = A.marks ? A.marks + (size_t)b * 16 : nullptr;
  const uint32_t ring_a = sa(smem), full_a = sa(bars), empty_a = sa(bars + NST);

  // ---------------- prologue ----------------
  if (tid == 0) {
    if (mk) mk[0] = gtime();
    for (int i = 0; i < NST; ++i) { mbar_init(full_a + 8 * i, FULLCNT); mbar_init(empty_a + 8 * i, 8); }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    s_gen = -1;
  }
  if (tid < NP) {
    const int v = __ldg(A.ids + tid) - A.lo;
    s_ids[tid] = (v >= 0 && v < 32) ? v : -1;
  }
  __syncthreads();
  if (warp == 0) {
    // lane = local expert: count, exclusive prefix, route list in increasing id position, touched order
    const int e = lane;
    int c = 0;
    for (int p = 0; p < NP; ++p) c += (s_ids[p] == e);
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    const unsigned msk = __ballot_sync(FULL, c > 0);
    S.cnt[e] = c; S.off[e] = off;
    s_rt[lane] = -1; s_rt[lane + 32] = -1;
    __syncwarp();
    int r = 0;
    for (int p = 0; p < NP; ++p)
      if (s_ids[p] == e) { s_list[off + r] = (int16_t)p; s_rt[p] = (int16_t)(off + r); ++r; }
    const int n = __popc(msk);
    if (c > 0) S.grp[__popc(msk & ((1u << lane) - 1u))] = e;
    if (lane == 31) {
      S.n = n;
      S.nA = n - ((n >= 2) ? max(1, n / 3) : 0);
      S.R = v;
    }
  }
  __syncthreads();

  if (warp == 8) {
    // ---------------- producer ----------------
    const bool noX = (mode & 2) != 0, nodep = (mode & 4) != 0;
#if __CUDA_ARCH__ >= 900
    const uint64_t pol = policy_evict_first();
#endif
    int it = 0;
    int nd = 0;
    uint32_t dDst[NST], dBar[NST], dB[NST];
    const __nv_bfloat16* dSrc[NST];
    int dDep[NST];
    bool dok0 = nodep, dok1 = nodep;
    unsigned tgt = 0;
    bool have_tgt = false;
    auto dep_target = [&]() -> unsigned {
      if (!have_tgt) {
        const long long t0 = clock64();
        while (*reinterpret_cast<volatile int*>(&s_gen) < 0) {
          if (clock64() - t0 > WAIT_BUDGET) { if (lane == 0) set_err(A.err, 5); break; }
        }
        tgt = (unsigned)(*reinterpret_cast<volatile int*>(&s_gen) + 1) * (unsigned)G;
        have_tgt = true;
      }
      return tgt;
    };
    // blocking flush of the deferred inter copies, in order
    auto flush = [&]() {
#pragma unroll
      for (int d = 0; d < NST; ++d) {
        if (d < nd) {
          const int dp = dDep[d];
          if (!(dp ? dok1 : dok0)) {
            const unsigned tg = dep_target();
            bool ok = true;
            if (lane == 0) ok = spin_ge(A.sync + 1 + dp, tg, A.err, 3);
            __syncwarp();
            if (dp) dok1 = true; else dok0 = true;
            if (mk && lane == 0) mk[8 + dp] = gtime();
            fence_proxy_async();
            (void)ok;
          }
#if __CUDA_ARCH__ >= 900
          if (lane == 0) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
#else
          warp_copy(lane, dDst[d], dSrc[d], dB[d]);
          cpa_arrive(dBar[d]);
#endif
        }
      }
      nd = 0;
    };
    bool first = true;
    for (int p = 0; p < 4; ++p) {
      int g0, nrb, beg, end;
      phase_range(S, p, b, G, g0, nrb, beg, end);
      const int md = (p < 2) ? 1 : 2;
      const int dp = (p == 2) ? 0 : 1;
      const uint8_t* wsrc = (md == 1) ? A.ws13 : A.ws2;
      const int per = nrb * 16;
      for (int x = beg; x < end; ++x, ++it) {
        const int ge = x / per, l = x - ge * per;
        const int e = S.grp[g0 + ge], rb = l >> 4, s = l & 15;
        const int ntok = S.cnt[e], c0 = S.off[e];
        const int slot = it % NST;
        const uint32_t par = ((it / NST) & 1) ^ 1;
        if (it >= NST) {
          if (nd > 0) flush();   // never block on a slot while copies are deferred
          mbar_wait(empty_a + 8 * slot, par, A.err, 2);
        }
        const uint32_t dst = ring_a + slot * STG, fb = full_a + 8 * slot;
        const uint32_t xbytes = noX ? 0u : (uint32_t)((md == 1) ? ntok * 512 : ntok * XROW);
        const uint8_t* src = wsrc + ((size_t)(e * nrb + rb) * 16 + s) * SW;
        bool defer = false;
        const __nv_bfloat16* isrc = A.inter + ((size_t)s * RS + c0) * XST;
        if (md == 2 && !noX) {
          bool ok = dp ? dok1 : dok0;
          if (!ok) {   // non-blocking probe
            if (mk && lane == 0 && mk[12 + dp] == 0) mk[12 + dp] = gtime();   // first time this dependency is needed
            unsigned v = 0;
            if (lane == 0) v = ld_acq(A.sync + 1 + dp);
            v = __shfl_sync(FULL, v, 0);
            ok = reached(v, dep_target());
            if (ok) {
              if (dp) dok1 = true; else dok0 = true;
              if (mk && lane == 0) mk[8 + dp] = gtime();
              fence_proxy_async();
            }
          }
          defer = !ok || nd > 0;
        }
#if __CUDA_ARCH__ >= 900
        if (lane == 0) {
          mbar_expect_tx(fb, SW + xbytes);
          bulk_g2s_ef(dst, src, SW, fb, pol);
        }
        __syncwarp();
        if (!noX) {
          if (md == 1) {
            if (lane >= 1 && lane <= ntok) {
              const int tok = ((int)s_list[c0 + lane - 1]) >> 3;
              bulk_g2s(dst + SW + (lane - 1) * XROW, A.x + (size_t)tok * KDIM + s * 256, 512, fb);
            }
          } else if (!defer) {
            if (lane == 0) bulk_g2s(dst + SW, isrc, xbytes, fb);
          }
        }
#else
        warp_copy(lane, dst, src, SW);
        if (!noX && md == 1) {
          for (int n = 0; n < ntok; ++n) {
            const int tok = ((int)s_list[c0 + n]) >> 3;
            warp_copy(lane, dst + SW + n * XROW, A.x + (size_t)tok * KDIM + s * 256, 512);
          }
        } else if (!noX && !defer) {
          warp_copy(lane, dst + SW, isrc, xbytes);
        }
        if (!defer) cpa_arrive(fb);
#endif
        if (defer) {
#pragma unroll
          for (int d = 0; d < NST; ++d)
            if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = isrc; dB[d] = xbytes; dDep[d] = dp; }
          ++nd;
        }
        if (first) { first = false; if (mk && lane == 0) mk[1] = gtime(); }
      }
    }
    if (nd > 0) flush();
    if (mk && lane == 0) mk[10] = gtime();
    return;
  }

  if (warp == 9) {
    // ---------------- epilogue warp ----------------
    if (lane == 0) {
      const unsigned old = atomicAdd(A.sync + 0, 1u);   // generation ticket: every CTA once per call
      *reinterpret_cast<volatile int*>(&s_gen) = (int)(old / (unsigned)G);
    }
    const bool nullc = (mode & 1) != 0;
    int nsplit = 0;
    for (int p = 0; p < 2; ++p) {
      int g0, nrb, beg, end;
      phase_range(S, p, b, G, g0, nrb, beg, end);
      const int per = nrb * 16;
      for (int x = beg; x < end;) {
        const int ge = x / per, l = x - ge * per;
        const int e = S.grp[g0 + ge], rb = l >> 4, s0 = l & 15;
        const int s1 = min(16, s0 + (end - x));
        x += s1 - s0;
        if ((s0 == 0 && s1 == 16) || nullc) continue;
        const int ntok = S.cnt[e], c0 = S.off[e];
        bar_sync(4 + nsplit, 288);
        ++nsplit;
        __threadfence();   // cumulative over the consumers' h atomics (ordered before us by the barrier)
        int last = 0;
        if (lane == 0) {
          int* cp = A.cnt + e * 64 + rb;
          const int old = atomicAdd(cp, s1 - s0);
          last = (old + (s1 - s0) == 16);
          if (last) atomicExch(cp, 0);
        }
        last = __shfl_sync(FULL, last, 0);
        if (last) {
          __threadfence();
          const int col = rb * 64 + 2 * lane;
          float2 gv[MMAX], uv[MMAX];
#pragma unroll
          for (int n = 0; n < MMAX; ++n) {
            if (n < ntok) {
              const float* hp = A.h + (size_t)(c0 + n) * (2 * KDIM) + col;
              gv[n] = __ldcg(reinterpret_cast<const float2*>(hp));
              uv[n] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
            }
          }
#pragma unroll
          for (int n = 0; n < MMAX; ++n) {
            if (n < ntok) {
              const int c = c0 + n;
              float* hp = A.h + (size_t)c * (2 * KDIM) + col;
              *reinterpret_cast<uint32_t*>(A.inter + ((size_t)(col >> 8) * RS + c) * XST + (col & 255)) =
                  pack_bf2(swiglu(gv[n].x, uv[n].x), swiglu(gv[n].y, uv[n].y));
              *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
              *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
            }
          }
        }
      }
      bar_sync(1 + p, 288);   // the consumers are done with this W13 wave (inter stores / h atomics issued)
      __threadfence();
      if (lane == 0) atomicAdd(A.sync + 1 + p, 1u);
    }
    bar_sync(3, 288);         // the consumers are done with W2 (y stores / atomics issued)
    __threadfence();
    if (lane == 0) {
      atomicAdd(A.sync + 3, 1u);
      if (mk) mk[6] = gtime();
    }
    return;
  }

  // ---------------- consumers (warps 0-7: warp w = tile w of the 128-row item) ----------------
  const bool nullc = (mode & 1) != 0;
  {
    // y rows [0, R) -> 0 (W2 split items add into them; W2 starts only after every CTA passed its W13(A))
    const size_t n4 = (size_t)S.R * KDIM / 4;
    float4* yp = reinterpret_cast<float4*>(A.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)b * 256 + tid; i < n4; i += (size_t)G * 256) yp[i] = z;
  }
  const int g = lane >> 2, t = lane & 3;
  int it = 0, nsplit = 0;
  for (int p = 0; p < 4; ++p) {
    int g0, nrb, beg, end;
    phase_range(S, p, b, G, g0, nrb, beg, end);
    const int md = (p < 2) ? 1 : 2;
    const int per = nrb * 16;
    for (int x = beg; x < end;) {
      const int ge = x / per, l = x - ge * per;
      const int e = S.grp[g0 + ge], rb = l >> 4, s0 = l & 15;
      const int s1 = min(16, s0 + (end - x));
      const int ntok = S.cnt[e], c0 = S.off[e];
      float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
      for (int s = s0; s < s1; ++s, ++it) {
        const int slot = it % NST;
        mbar_wait(full_a + 8 * slot, (it / NST) & 1, A.err, 1);
        if (!nullc) {
          const uint8_t* st = smem + slot * STG;
          const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
          const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
          const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;
          uint4 wv[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
          const uint32_t sv0 = sp[0], sv1 = sp[32];
          consume_kb1(wv[0], sv0, 0, xrow, 0, t, acc);
          consume_kb1(wv[1], sv0, 1, xrow, 64, t, acc);
          consume_kb1(wv[2], sv1, 0, xrow, 128, t, acc2);
          consume_kb1(wv[3], sv1, 1, xrow, 192, t, acc2);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_a + 8 * slot);
      }
      const bool whole = (s0 == 0 && s1 == 16);
      if (!nullc) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += acc2[i];
        const int tile = rb * 8 + warp;
        if (md == 1) {
          const float gs = __ldg(A.gsc13 + e);
          const int col = 8 * tile + g;
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int n = 2 * t + hh;
            if (n < ntok) {
              const int c = c0 + n;
              const float v0 = acc[hh] * gs, v1 = acc[2 + hh] * gs;
              if (whole) {
                A.inter[((size_t)(col >> 8) * RS + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));
              } else {
                atomicAdd(A.h + (size_t)c * (2 * KDIM) + col, v0);
                atomicAdd(A.h + (size_t)c * (2 * KDIM) + KDIM + col, v1);
              }
            }
          }
          if (!whole) { bar_arrive(4 + nsplit, 288); ++nsplit; }
        } else {
          const float gs = __ldg(A.gsc2 + e);
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int n = 2 * t + hh;
            if (n < ntok) {
              float* yp = A.y + (size_t)(c0 + n) * KDIM + 16 * tile + g;
              const float v0 = acc[hh] * gs, v1 = acc[2 + hh] * gs;
              if (whole) { yp[0] = v0; yp[8] = v1; }
              else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
            }
          }
        }
      }
      x += s1 - s0;
    }
    if (p < 2) bar_arrive(1 + p, 288);
    if (p == 3) bar_arrive(3, 288);
    if (mk && tid == 0) mk[2 + p] = gtime();
  }

  // ---------------- finalize: post-expert norm + weighted sum, item (token b / 4, quarter b % 4) ----------------
  if ((mode & 8) || b >= 4 * M) return;
  {
    if (tid == 0) {
      const long long t0 = clock64();
      while (*reinterpret_cast<volatile int*>(&s_gen) < 0) {
        if (clock64() - t0 > WAIT_BUDGET) { set_err(A.err, 5); break; }
      }
      const unsigned tg = (unsigned)(*reinterpret_cast<volatile int*>(&s_gen) + 1) * (unsigned)G;
      spin_ge(A.sync + 3, tg, A.err, 4);
      if (mk) mk[11] = gtime();
    }
    bar_sync(8, 256);
    const int tk = b >> 2, q = b & 3;
    // local routes of the token, compacted (route id, slot)
    int nl = 0;
    int lr[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = s_rt[tk * 8 + j];
      lr[j] = 0;
      if (c >= 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) if (i == nl) lr[i] = (c << 3) | j;
        ++nl;
      }
    }
    // pass 1: sums of squares of z = bf16(y) * (1 + w) over the full rows, 2 routes per batch
    float ssp[8];
#pragma unroll
    for (int j0 = 0; j0 < 8; j0 += 2) {
      ssp[j0] = 0.f; ssp[j0 + 1] = 0.f;
      if (j0 < nl) {
        float4 v[2][4];
#pragma unroll
        for (int jj = 0; jj < 2; ++jj)
#pragma unroll
          for (int k = 0; k < 4; ++k)
            v[jj][k] = (j0 + jj < nl)
                ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)(lr[j0 + jj] >> 3) * KDIM + 1024 * k + 4 * tid))
                : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float4 nk = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * k + 4 * tid));
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            float zz, a = ssp[j0 + jj];
            zz = bf16r(v[jj][k].x) * nk.x; a += zz * zz;
            zz = bf16r(v[jj][k].y) * nk.y; a += zz * zz;
            zz = bf16r(v[jj][k].z) * nk.z; a += zz * zz;
            zz = bf16r(v[jj][k].w) * nk.w; a += zz * zz;
            ssp[j0 + jj] = a;
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float a = ssp[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(FULL, a, o);
      if (lane == 0) s_red[warp][j] = a;
    }
    bar_sync(8, 256);
    if (tid < 8) {
      float a = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) a += s_red[w][tid];
      int lv = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) if (i == tid) lv = lr[i];
      s_sc[tid] = (tid < nl) ? rsqrtf(a * (1.f / 4096.f) + 1e-8f) * __ldg(A.wts + tk * 8 + (lv & 7)) : 0.f;
    }
    bar_sync(8, 256);
    // pass 2: this quarter's columns, all local routes in one batch
    const float4 w = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * q + 4 * tid));
    float4 vq[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      vq[j] = (j < nl) ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)(lr[j] >> 3) * KDIM + 1024 * q + 4 * tid))
                       : make_float4(0.f, 0.f, 0.f, 0.f);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float r = s_sc[j];
      a0 += bf16r(vq[j].x) * w.x * r; a1 += bf16r(vq[j].y) * w.y * r;
      a2 += bf16r(vq[j].z) * w.z * r; a3 += bf16r(vq[j].w) * w.w * r;
    }
    uint2 o;
    o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
    *reinterpret_cast<uint2*>(A.out + (size_t)tk * KDIM + 1024 * q + 4 * tid) = o;
    if (mk && tid == 0) mk[7] = gtime();
  }
}

template <int NST>
constexpr int smem_bytes() { return NST * STG + 2 * NST * 8; }

template <int NST>
void launch(const Args& a, int G) {
  constexpr int sm = smem_bytes<NST>();
  cudaFuncSetAttribute(q1_kernel<NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaFuncSetAttribute(q1_kernel<NST>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  q1_kernel<NST><<<G, 320, sm, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int NST>
std::vector<int64_t> attrs() {
  constexpr int sm = smem_bytes<NST>();
  cudaFuncSetAttribute(q1_kernel<NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaFuncSetAttribute(q1_kernel<NST>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, q1_kernel<NST>);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, q1_kernel<NST>, 320, sm);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + sm), occ};
}
}  // namespace q1

void q1_run(int64_t cfg, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, torch::Tensor inter,
            torch::Tensor y, torch::Tensor cnt, torch::Tensor sync, torch::Tensor err, torch::Tensor out,
            torch::Tensor marks, int64_t M, int64_t lo, int64_t G, int64_t mode) {
  TORCH_CHECK(M >= 1 && M <= q1::MMAX, "q1: M out of range");
  q1::Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ids = ids.data_ptr<int>(); a.wts = wts.data_ptr<float>();
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr()); a.ws2 = reinterpret_cast<const uint8_t*>(ws2.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>(); a.gsc2 = gsc2.data_ptr<float>(); a.nw = nw.data_ptr<float>();
  a.h = h.data_ptr<float>(); a.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); a.y = y.data_ptr<float>();
  a.cnt = cnt.data_ptr<int>(); a.sync = reinterpret_cast<unsigned*>(sync.data_ptr<int>()); a.err = err.data_ptr<int>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.marks = marks.numel() >= G * 16 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  a.M = (int)M; a.lo = (int)lo; a.mode = (int)mode;
  switch (cfg) {
    case 3: q1::launch<3>(a, (int)G); break;
    case 4: q1::launch<4>(a, (int)G); break;
    default: TORCH_CHECK(false, "q1: bad cfg");
  }
}
std::vector<int64_t> q1_attrs(int64_t cfg) {
  switch (cfg) {
    case 3: return q1::attrs<3>();
    case 4: return q1::attrs<4>();
    default: return {-1};
  }
}
"""

Q1_CPP = r"""
#include <torch/extension.h>
#include <vector>
void q1_run(int64_t cfg, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, torch::Tensor inter,
            torch::Tensor y, torch::Tensor cnt, torch::Tensor sync, torch::Tensor err, torch::Tensor out,
            torch::Tensor marks, int64_t M, int64_t lo, int64_t G, int64_t mode);
std::vector<int64_t> q1_attrs(int64_t cfg);
"""

Q1_EXPERT_BYTES = 768 * 36864    # 28.31 MB per touched expert (W13 + W2, codes + scales)
Q1_CFGS = (3, 4)                 # ring stages


def q1_build(arch="sm_90a", verbose=False):
    from torch.utils.cpp_extension import load_inline
    gen = "-gencode=arch=compute_90a,code=sm_90a" if arch == "sm_90a" else "-gencode=arch=compute_86,code=sm_86"
    tag = hashlib.sha1((Q1_CPP + Q1_CUDA + arch).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"q1_moe_{tag}", cpp_sources=Q1_CPP, cuda_sources=Q1_CUDA, functions=["q1_run", "q1_attrs"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", gen], verbose=verbose)
    return mod, time.time() - t0


def q1_bufs(dev, G):
    return {
        "h": torch.zeros((64, 8192), dtype=torch.float32, device=dev),
        "inter": torch.zeros((16 * 64 * 264,), dtype=torch.bfloat16, device=dev),
        "y": torch.zeros((64, 4096), dtype=torch.float32, device=dev),
        "cnt": torch.zeros((32 * 64,), dtype=torch.int32, device=dev),
        "sync": torch.zeros((8,), dtype=torch.int32, device=dev),
        "err": torch.zeros((8,), dtype=torch.int32, device=dev),
        "marks": torch.zeros((G * 16,), dtype=torch.int64, device=dev),
        "nomark": torch.zeros((0,), dtype=torch.int64, device=dev),
    }


def q1_fn(mod, st, bufs, data, cfg, G, mode=0, mark=False):
    """Closure for one call (graph-capturable): returns out [M, 4096] bf16."""
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    lo = int(data["config"]["local_start"])
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    mk = bufs["marks"] if mark else bufs["nomark"]

    def f():
        mod.q1_run(cfg, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], bufs["h"],
                   bufs["inter"], bufs["y"], bufs["cnt"], bufs["sync"], bufs["err"], out, mk, m, lo, G, mode)
        return out
    return f


def q1_case(bs, seed, dev, dist="normal"):
    """generate_input's normal / ep8 / busiest routing math (our own copy; seeds disjoint from the board's)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    if dist == "normal":
        z = torch.randn((256,), generator=g, device=dev)
        u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    else:
        scores = scores + torch.rand((256,), generator=g, device=dev).pow(6) * 4
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def q1_touched(data):
    ids = data["topk_ids"].long() - int(data["config"]["local_start"])
    loc = ids[(ids >= 0) & (ids < 32)]
    return int(torch.unique(loc).numel())


def q1_rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


class Q1Timer:
    """The judge clone's timer: 256 MB read before every replay (cold L2), events around graph.replay()."""

    def __init__(self):
        self.flush_buf = torch.ones((256 << 20) // 4, dtype=torch.float32, device="cuda")
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")
        self.stream = torch.cuda.Stream()

    def capture(self, fn):
        with torch.cuda.stream(self.stream):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                out = fn()
        torch.cuda.synchronize()
        return gr, out

    def time(self, gr):
        with torch.cuda.stream(self.stream):
            torch.sum(self.flush_buf, dim=0, out=self.sink)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    def overhead(self, n=100):
        tiny = torch.zeros(1, device="cuda")
        with torch.cuda.stream(self.stream):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                tiny.add_(0)
        torch.cuda.synchronize()
        return statistics.median(self.time(gr) for _ in range(n))


def q1_marks_summary(bufs, G, t_base=None):
    mv = bufs["marks"].view(-1, 16)[:G].tolist()
    t0 = min(r[0] for r in mv) if t_base is None else t_base

    def col(k, only=None):
        v = [(r[k] - t0) / 1e3 for r in mv if r[k] > 0 and (only is None or only(r))]
        return sorted(v)

    def p50max(v):
        return (statistics.median(v), v[-1]) if v else (float("nan"), float("nan"))
    out = {}
    for name, k in (("first_copy", 1), ("w13a_end", 2), ("w13b_end", 3), ("w2a_end", 4), ("w2_end", 5),
                    ("sync2_pub", 6), ("fin_end", 7), ("depA", 8), ("depB", 9), ("prod_done", 10), ("fin_seen", 11),
                    ("needA", 12), ("needB", 13)):
        out[name] = p50max(col(k))
    # producer stall on a dependency: satisfied - first needed (0 when it was met at the first probe)
    for nm, a, b2 in (("stallA", 12, 8), ("stallB", 13, 9)):
        v = sorted((r[b2] - r[a]) / 1e3 for r in mv if r[a] > 0 and r[b2] > 0)
        out[nm] = p50max(v) if v else (0.0, 0.0)
    return out


def q1_lab(st, k2a_fn, k2a_state, log=print, rounds=30, inputs=3, seeds0=919001, sizes=(1, 8), arch="sm_90a"):
    """Step-0 gate (notes/memory/gpu_requests.md R-moe1-0): arms K (shipped K2A), Q3 / Q4 (one-launch two-wave, 3 / 4
    ring stages), Q0 (Q3 with null consumers), S (Q3 pure stream: null consumers, no X / inter, no deps, no finalize)."""
    dev = st["ws13"].device
    mod, dt = q1_build(arch)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    occ = {}
    for cfg in Q1_CFGS:
        a = list(mod.q1_attrs(cfg))
        occ[cfg] = a[3]
        log(f"[q1] cfg {cfg} attrs (regs, local, smem, occ) {a}")
    log(f"[q1] built in {dt:.0f}s, {sms} SMs")
    timer = Q1Timer()
    ov = timer.overhead()
    log(f"[q1] graph overhead {ov:.2f} us")
    rows = []
    for bs in sizes:
        for r in range(inputs):
            d = q1_case(bs, seeds0 + 1000 * bs + r, dev)
            ne = q1_touched(d)
            B = ne * Q1_EXPERT_BYTES
            spec = {}
            for cfg in Q1_CFGS:
                if occ[cfg] >= 1:   # every CTA must be resident (in-kernel waits): G = SMs x min(occ, 2)
                    spec[f"Q{cfg}"] = (cfg, min(occ[cfg], 2) * sms, 0)
            if "Q3" in spec:
                spec["Q0"] = (3, spec["Q3"][1], 1)
                spec["S"] = (3, spec["Q3"][1], 15)
            bufs = {nm: q1_bufs(dev, v[1]) for nm, v in spec.items()}
            arms = {"K": lambda d=d: k2a_fn(d, k2a_state)}
            arms.update({nm: q1_fn(mod, st, bufs[nm], d, *v) for nm, v in spec.items()})
            ref = arms["K"]().float().clone()
            torch.cuda.synchronize()
            rel, errs = {}, {}
            for nm in spec:
                if spec[nm][2] != 0:
                    continue
                bufs[nm]["err"].zero_()
                o1 = arms[nm]().float().clone()
                o2 = arms[nm]().float().clone()
                torch.cuda.synchronize()
                rel[nm] = (q1_rel(o1, ref), q1_rel(o2, ref), bool(torch.isfinite(o1).all()))
                errs[nm] = [int(v) for v in bufs[nm]["err"].tolist()[:2]]
            graphs = {nm: timer.capture(fn)[0] for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            tl = {}
            for nm in ("Q3", "Q0"):
                if nm not in spec:
                    continue
                cfg, G, md = spec[nm]
                bufs[nm]["marks"].zero_()
                f = q1_fn(mod, st, bufs[nm], d, cfg, G, md, mark=True)
                with torch.cuda.stream(timer.stream):
                    torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                    f()
                torch.cuda.synchronize()
                tl[nm] = q1_marks_summary(bufs[nm], G)
            e_all = {nm: [int(v) for v in bufs[nm]["err"].tolist()[:2]] for nm in spec}
            rows.append({"bs": bs, "ne": ne, "B": B, "us": us, "rel": rel, "err": e_all, "tl": tl})
            log(f"[q1] bs={bs} r={r} experts {ne} ({B / 1e6:.1f} MB): " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items())
                + " | " + " ".join(f"{nm}/K {us[nm] / us['K']:.3f}" for nm in us if nm != "K")
                + f" | B/4.4TB/s {B / 4.4e6:.1f}")
            log("[q1]   rel vs K (call 1, call 2, finite): " + " ".join(f"{nm} {a:.1e}/{b2:.1e}/{fi}" for nm, (a, b2, fi) in rel.items())
                + f" | err {e_all}")
            for nm, s in tl.items():
                log(f"[q1]   TL {nm}: " + " ".join(f"{k} {a:.1f}/{b2:.1f}" for k, (a, b2) in s.items()))
            del graphs, bufs
            torch.cuda.empty_cache()
    # gate (pre-registered R-moe1-0)
    log("[q1] GATE (median over routings of best(Q3, Q4) / K): bs1 <= 0.85 and bs8 <= 0.92, rel <= 1e-3, err 0")
    verdict = True
    for bs in sizes:
        rr = [x for x in rows if x["bs"] == bs]
        if not rr:
            continue
        ratios, ok_c = [], True
        for x in rr:
            qs = [nm for nm in ("Q3", "Q4") if nm in x["us"]]
            if not qs:
                ok_c = False
                continue
            best = min(qs, key=lambda nm: x["us"][nm])
            ratios.append(x["us"][best] / x["us"]["K"])
            for nm in qs:
                a, b2, fi = x["rel"].get(nm, (1.0, 1.0, False))
                ok_c &= (a <= 1e-3 and b2 <= 1e-3 and fi and x["err"][nm][0] == 0)
        mr = statistics.median(ratios) if ratios else float("nan")
        lim = 0.85 if bs == 1 else 0.92
        okb = ok_c and mr <= lim
        if bs in (1, 8):
            verdict &= okb
        else:
            ok_c and log(f"[q1]   (bs={bs} is information only)")
        log(f"[q1]   bs={bs}: median best/K {mr:.3f} (limit {lim}), correctness {'ok' if ok_c else 'FAIL'} -> "
            f"{'pass' if okb else 'FAIL'}")
    log(f"[q1] GATE {'GO' if verdict else 'NO-GO'}")
    del timer
    torch.cuda.empty_cache()
    return rows

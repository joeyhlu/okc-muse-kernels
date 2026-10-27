"""Q2: one-launch two-wave MoE for T <= 8, step-0 lab (request R-moe1-1; notes/study/moe_structure_fresh.md).

Q1 (R-moe1-0, killed) showed the one-launch skeleton itself costs ~4.5 us over a pure stream (K2A's chain ~16 us) but lost on
measured implementation costs. Q2 = Q1's schedule (G = SMs x 2, static Stream-K shares of W13(A) W13(B) W2(A) W2(B) in one
ring, K2A's NC1 consumer and stage layout) with each diagnosed cost removed:
- W13 epilogue: every item (whole or split) red.adds its FP32 gate / up rows into h [16 K-stages][RS routes][512]; no split
  counters, no last-contributor chain (Q1: ~2 extra loaded RTTs before each wave signal);
- W2 B operand: the producer bulk-copies the stage's h slice (ntok x 2 KB) into a 2-slot staging ring; a converter warp
  (warp 10) applies SwiGLU + bf16 rounding into the ring slot's B rows and arrives on the slot's full barrier;
- dependency polls are async: a 16 B bulk copy of the counter line into smem, checked with try_wait between stages (Q1's
  probe used the loaded value at once and blocked the producer one loaded RTT, ~3 us, per probe);
- release-only fences (fence.release.gpu + red, no CCTL.IVALL); global scales and the post-norm weights in smem /
  registers, so no consumer or finalize load sits behind an L1 invalidation;
- ballot routing in one warp (no serial smem loops; Q1's first copy was at 2.3 / 4.9 us at bs 1 / 8);
- wave split nB = max(1, round(0.4 n)) (more slack for dependency A);
- finalize: nw / weights / route lists prefetched before the counter wait; h zeroed after it (self-reset).
352 threads: warps 0-7 consumers (+ finalize), 8 producer, 9 wave signals, 10 SwiGLU converter.
mode bits (lab arms): 1 null consumers, 2 no X / h copies, 4 no dependency waits, 8 no finalize.
Builds for sm_90a (bulk copies) and sm_80/86 (cp.async, blocking polls; local correctness only).
"""
import hashlib
import statistics
import time

import torch

Q2_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace q2 {
constexpr int KDIM = 4096;
constexpr int SW = 18432;            // weight bytes per stage: 128 rows x 256 K
constexpr int XROW = 528;            // staged B row: 256 bf16 + 16 B pad
constexpr int XST = XROW / 2;
constexpr int MMAX = 8;
constexpr int PMAX = 8 * MMAX;
constexpr int RS = PMAX;             // route stride of h [16][RS][512]
constexpr int STG = SW + 8 * XROW;   // ring slot
#ifndef Q2_NST
#define Q2_NST 3
#endif
constexpr int NST = Q2_NST;          // ring stages (sm_86 local builds: 2, smem cap 99 KB)
constexpr int NHS = 2;               // h staging slots
constexpr int HSB = 8 * 2048;        // h staging slot: <= 8 routes x (256 gate + 256 up) fp32
constexpr int NTH = 352;
constexpr unsigned FULL = 0xffffffffu;
constexpr long long WAIT_BUDGET = 1ll << 30;
#if __CUDA_ARCH__ >= 900
constexpr int FULLCNT = 2;           // producer arrive.expect_tx + (producer | converter) arrive
constexpr int HCNT = 1;
#else
constexpr int FULLCNT = 33;          // 32 noinc cp.async arrivals + (producer | converter) arrive
constexpr int HCNT = 32;
#endif

struct Args {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint8_t* ws13; const uint8_t* ws2;
  const float* gsc13; const float* gsc2; const float* nw;
  float* h; float* y; unsigned* sync; int* err;
  __nv_bfloat16* out; long long* marks;
  int M, lo, mode;
};

struct Plan {
  int n, nA, R;
  int grp[32], cnt[32], off[32];
  float gs13[32], gs2[32];
};

// ---------------- math (K2A v3.3) ----------------
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {
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

// ---------------- barriers / copies / fences ----------------
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
// K2A's wait: no global loads in the spin (a load under a full ring costs one loaded RTT)
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, int code) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return; }
  }
}
__device__ __forceinline__ unsigned ld_rlx(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void fence_acqrel() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ void red_release(unsigned* p) {
  asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p) : "memory");
}
__device__ __forceinline__ void red_f32(float* p, float v) {
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n" :: "l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ bool reached(unsigned v, unsigned tgt) { return (int)(v - tgt) >= 0; }
__device__ __forceinline__ bool spin_ge(const unsigned* p, unsigned tgt, int* err, int code) {
  const long long t0 = clock64();
  while (!reached(ld_rlx(p), tgt)) {
    __nanosleep(32);
    if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return false; }
  }
  fence_acqrel();
  return true;
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

__device__ __forceinline__ void phase_range(const Plan& S, int p, int b, int G, int& g0, int& nrb, int& beg, int& end) {
  const bool a = (p == 0 || p == 2);
  g0 = a ? 0 : S.nA;
  const int ng = a ? S.nA : S.n - S.nA;
  nrb = (p < 2) ? 64 : 32;
  const long long sp = (long long)ng * nrb * 16;
  beg = (int)(sp * b / G);
  end = (int)(sp * (b + 1) / G);
}

__global__ void __launch_bounds__(NTH, 2) q2_kernel(const Args A) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int16_t s_list[PMAX];
  __shared__ int16_t s_rt[PMAX];
  __shared__ Plan S;
  __shared__ int s_gen;
  __shared__ __align__(16) unsigned s_poll[4];
  __shared__ float s_red[8][8];
  __shared__ float s_sc[8];
  __shared__ float s_w8[8];
  uint8_t* hstage = smem + NST * STG;
  uint64_t* bars = reinterpret_cast<uint64_t*>(hstage + NHS * HSB);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = blockIdx.x, G = gridDim.x;
  const int M = A.M, NP = 8 * A.M;
  const int mode = A.mode;
  long long* mk = A.marks ? A.marks + (size_t)b * 16 : nullptr;
  const uint32_t ring_a = sa(smem), hst_a = sa(hstage);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST), hfull_a = sa(bars + 2 * NST),
                 hempty_a = sa(bars + 2 * NST + NHS), poll_a = sa(bars + 2 * NST + 2 * NHS);

  // ---------------- prologue ----------------
  if (tid == 0) {
    if (mk) mk[0] = gtime();
    for (int i = 0; i < NST; ++i) { mbar_init(full_a + 8 * i, FULLCNT); mbar_init(empty_a + 8 * i, 8); }
    for (int i = 0; i < NHS; ++i) { mbar_init(hfull_a + 8 * i, HCNT); mbar_init(hempty_a + 8 * i, 1); }
    mbar_init(poll_a, 1);
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    s_gen = -1;
  }
  if (warp == 1) { S.gs13[lane] = __ldg(A.gsc13 + lane); S.gs2[lane] = __ldg(A.gsc2 + lane); }
  if (warp == 0) {
    // ballot routing: lane holds id positions p = lane and p = lane + 32
    int e0 = -1, e1 = -1;
    if (lane < NP) { const int v = __ldg(A.ids + lane) - A.lo; e0 = (v >= 0 && v < 32) ? v : -1; }
    if (lane + 32 < NP) { const int v = __ldg(A.ids + lane + 32) - A.lo; e1 = (v >= 0 && v < 32) ? v : -1; }
    int fh = 0, c = 0;
#pragma unroll
    for (int e = 0; e < 32; ++e) {
      const unsigned b0 = __ballot_sync(FULL, e0 == e), b1 = __ballot_sync(FULL, e1 == e);
      if (lane == e) { fh = __popc(b0); c = fh + __popc(b1); }
    }
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    const unsigned msk = __ballot_sync(FULL, c > 0);
    S.cnt[lane] = c; S.off[lane] = off;
    if (c > 0) S.grp[__popc(msk & ((1u << lane) - 1u))] = lane;
    const unsigned lt = (1u << lane) - 1u;
    const unsigned m0 = __match_any_sync(FULL, e0), m1 = __match_any_sync(FULL, e1);
    const int off0 = __shfl_sync(FULL, off, e0 & 31), off1 = __shfl_sync(FULL, off, e1 & 31);
    const int fh1 = __shfl_sync(FULL, fh, e1 & 31);
    if (lane < NP) {
      int r = -1;
      if (e0 >= 0) { r = off0 + __popc(m0 & lt); s_list[r] = (int16_t)lane; }
      s_rt[lane] = (int16_t)r;
    }
    if (lane + 32 < NP) {
      int r = -1;
      if (e1 >= 0) { r = off1 + fh1 + __popc(m1 & lt); s_list[r] = (int16_t)(lane + 32); }
      s_rt[lane + 32] = (int16_t)r;
    }
    if (lane == 31) {
      const int n = __popc(msk);
      const int nb = (n >= 2) ? max(1, (2 * n + 2) / 5) : 0;   // round(0.4 n)
      S.n = n; S.nA = n - nb; S.R = v;
    }
  }
  __syncthreads();

  if (warp == 8) {
    // ---------------- producer ----------------
    const bool noX = (mode & 2) != 0, nodep = (mode & 4) != 0;
#if __CUDA_ARCH__ >= 900
    const uint64_t pol = policy_evict_first();
#endif
    bool dok0 = nodep, dok1 = nodep;
    bool pin = false;            // async counter poll in flight
    uint32_t pph = 0;
    unsigned tgt = 0;
    bool have_tgt = false;
    // FIFO (depth <= NST = 3) of W2 stages whose h slice is not yet issued, kept in registers
    int nd = 0, w2i_issued = 0;
    const float* fs0 = nullptr; const float* fs1 = nullptr; const float* fs2 = nullptr;
    uint32_t fb0 = 0, fb1 = 0, fb2 = 0;
    int fd0 = 0, fd1 = 0, fd2 = 0, fi0 = 0, fi1 = 0, fi2 = 0;
    bool need0 = false, need1 = false;
#define Q2_TGT()                                                                                     \
    do {                                                                                             \
      if (!have_tgt) {                                                                               \
        const long long t0_ = clock64();                                                             \
        while (*reinterpret_cast<volatile int*>(&s_gen) < 0) {                                       \
          if (clock64() - t0_ > WAIT_BUDGET) { if (lane == 0) set_err(A.err, 5); break; }            \
        }                                                                                            \
        tgt = (unsigned)(*reinterpret_cast<volatile int*>(&s_gen) + 1) * (unsigned)G;                 \
        have_tgt = true;                                                                             \
      }                                                                                              \
    } while (0)
#if __CUDA_ARCH__ >= 900
#define Q2_POLL_ISSUE()                                                                              \
    do { if (lane == 0) { mbar_expect_tx(poll_a, 16); bulk_g2s(sa(s_poll), A.sync, 16, poll_a); } pin = true; } while (0)
#define Q2_POLL_TRY()                                                                                \
    do {                                                                                             \
      if (pin && __shfl_sync(FULL, (int)mbar_try(poll_a, pph), 0)) {                                                            \
        pph ^= 1; pin = false;                                                                       \
        const unsigned v1_ = *reinterpret_cast<volatile unsigned*>(s_poll + 1);                      \
        const unsigned v2_ = *reinterpret_cast<volatile unsigned*>(s_poll + 2);                      \
        Q2_TGT();                                                                                    \
        const bool o0_ = dok0, o1_ = dok1;                                                           \
        if (reached(v1_, tgt)) dok0 = true;                                                          \
        if (reached(v2_, tgt)) dok1 = true;                                                          \
        if ((dok0 && !o0_) || (dok1 && !o1_)) {                                                      \
          fence_acqrel(); fence_proxy_async();                                                       \
          if (mk && lane == 0) { if (dok0 && !o0_) mk[8] = gtime(); if (dok1 && !o1_) mk[9] = gtime(); } \
        }                                                                                            \
      }                                                                                              \
    } while (0)
#define Q2_ISSUE_H(SRC, BYTES)                                                                       \
    do {                                                                                             \
      const int hs_ = w2i_issued % NHS;                                                              \
      if (w2i_issued >= NHS) mbar_wait(hempty_a + 8 * hs_, ((w2i_issued / NHS) & 1) ^ 1, A.err, 6);  \
      if (lane == 0) { mbar_expect_tx(hfull_a + 8 * hs_, (BYTES));                                   \
                       if (BYTES) bulk_g2s(hst_a + hs_ * HSB, (SRC), (BYTES), hfull_a + 8 * hs_); }  \
      ++w2i_issued;                                                                                  \
    } while (0)
#else
#define Q2_POLL_ISSUE() do { pin = true; } while (0)
#define Q2_POLL_TRY()                                                                                \
    do {                                                                                             \
      if (pin) {                                                                                     \
        pin = false;                                                                                 \
        unsigned v1_ = 0, v2_ = 0;                                                                   \
        if (lane == 0) { v1_ = ld_rlx(A.sync + 1); v2_ = ld_rlx(A.sync + 2); }                       \
        v1_ = __shfl_sync(FULL, v1_, 0); v2_ = __shfl_sync(FULL, v2_, 0);                            \
        Q2_TGT();                                                                                    \
        const bool o0_ = dok0, o1_ = dok1;                                                           \
        if (reached(v1_, tgt)) dok0 = true;                                                          \
        if (reached(v2_, tgt)) dok1 = true;                                                          \
        if ((dok0 && !o0_) || (dok1 && !o1_)) {                                                      \
          fence_acqrel();                                                                            \
          if (mk && lane == 0) { if (dok0 && !o0_) mk[8] = gtime(); if (dok1 && !o1_) mk[9] = gtime(); } \
        }                                                                                            \
      }                                                                                              \
    } while (0)
#define Q2_ISSUE_H(SRC, BYTES)                                                                       \
    do {                                                                                             \
      const int hs_ = w2i_issued % NHS;                                                              \
      if (w2i_issued >= NHS) mbar_wait(hempty_a + 8 * hs_, ((w2i_issued / NHS) & 1) ^ 1, A.err, 6);  \
      warp_copy(lane, hst_a + hs_ * HSB, (SRC), (BYTES));                                            \
      cpa_arrive(hfull_a + 8 * hs_);                                                                 \
      ++w2i_issued;                                                                                  \
    } while (0)
#endif
    // pop FIFO heads whose dependency is met; heads of stages <= UPTO are waited for (their slot is reused next)
#define Q2_DRAIN(UPTO)                                                                               \
    do {                                                                                             \
      while (nd > 0) {                                                                               \
        bool ok_ = fd0 ? dok1 : dok0;                                                                \
        if (!ok_ && fi0 <= (UPTO)) {                                                                 \
          const long long t0_ = clock64();                                                           \
          while (!(fd0 ? dok1 : dok0)) {                                                             \
            if (!pin) Q2_POLL_ISSUE();                                                               \
            Q2_POLL_TRY();                                                                           \
            if (clock64() - t0_ > WAIT_BUDGET) { if (lane == 0) set_err(A.err, 3); dok0 = dok1 = true; } \
          }                                                                                          \
          ok_ = true;                                                                                \
        }                                                                                            \
        if (!ok_) break;                                                                             \
        Q2_ISSUE_H(fs0, fb0);                                                                        \
        fs0 = fs1; fb0 = fb1; fd0 = fd1; fi0 = fi1;                                                  \
        fs1 = fs2; fb1 = fb2; fd1 = fd2; fi1 = fi2;                                                  \
        --nd;                                                                                        \
      }                                                                                              \
    } while (0)
    int it = 0;
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
        Q2_POLL_TRY();
        Q2_DRAIN(-1);
        if (it >= NST) {
          Q2_DRAIN(it - NST);   // the reused slot's consumers need its h copy before they can free it
          mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, A.err, 2);
        }
        const uint32_t dst = ring_a + slot * STG, fb = full_a + 8 * slot;
        const uint8_t* src = wsrc + ((size_t)(e * nrb + rb) * 16 + s) * SW;
        const uint32_t xbytes = (md == 1 && !noX) ? (uint32_t)(ntok * 512) : 0u;
#if __CUDA_ARCH__ >= 900
        if (lane == 0) { mbar_expect_tx(fb, SW + xbytes); bulk_g2s_ef(dst, src, SW, fb, pol); }
        __syncwarp();
        if (xbytes && lane >= 1 && lane <= ntok) {
          const int tok = ((int)s_list[c0 + lane - 1]) >> 3;
          bulk_g2s(dst + SW + (lane - 1) * XROW, A.x + (size_t)tok * KDIM + s * 256, 512, fb);
        }
#else
        warp_copy(lane, dst, src, SW);
        if (xbytes) {
          for (int n = 0; n < ntok; ++n) {
            const int tok = ((int)s_list[c0 + n]) >> 3;
            warp_copy(lane, dst + SW + n * XROW, A.x + (size_t)tok * KDIM + s * 256, 512);
          }
        }
        cpa_arrive(fb);
#endif
        if (md == 1) {
          __syncwarp();
          if (lane == 0) mbar_arrive(fb);   // W13: the second arrival is ours (W2: the converter's)
        } else {
          const float* hsrc = A.h + ((size_t)s * RS + c0) * 512;
          const uint32_t hb = noX ? 0u : (uint32_t)(ntok * 2048);
          bool ok = dp ? dok1 : dok0;
          if (!ok) {
            if (!(dp ? need1 : need0)) {   // first time this dependency is needed (marks: register flag, no load)
              if (dp) need1 = true; else need0 = true;
              if (mk && lane == 0) mk[12 + dp] = gtime();
            }
            if (!pin) Q2_POLL_ISSUE();
          }
          if (ok && nd == 0) {
            Q2_ISSUE_H(hsrc, hb);
          } else {
            if (nd == 0) { fs0 = hsrc; fb0 = hb; fd0 = dp; fi0 = it; }
            else if (nd == 1) { fs1 = hsrc; fb1 = hb; fd1 = dp; fi1 = it; }
            else { fs2 = hsrc; fb2 = hb; fd2 = dp; fi2 = it; }
            ++nd;
          }
        }
        if (first) { first = false; if (mk && lane == 0) mk[1] = gtime(); }
      }
    }
    Q2_DRAIN(1 << 30);
    if (pin) {   // retire an outstanding poll before exit (its bytes target our smem)
#if __CUDA_ARCH__ >= 900
      mbar_wait(poll_a, pph, A.err, 7);
#endif
    }
    if (mk && lane == 0) mk[10] = gtime();
#undef Q2_TGT
#undef Q2_POLL_ISSUE
#undef Q2_POLL_TRY
#undef Q2_ISSUE_H
#undef Q2_DRAIN
    return;
  }

  if (warp == 9) {
    // ---------------- wave signals ----------------
    if (lane == 0) {
      const unsigned old = atomicAdd(A.sync + 0, 1u);   // generation ticket: every CTA once per call
      *reinterpret_cast<volatile int*>(&s_gen) = (int)(old / (unsigned)G);
    }
    bar_sync(1, 288);   // consumers issued their W13(A) h reds
    if (lane == 0) red_release(A.sync + 1);
    bar_sync(2, 288);   // W13(B)
    if (lane == 0) red_release(A.sync + 2);
    bar_sync(3, 288);   // W2 y stores / reds
    if (lane == 0) { red_release(A.sync + 3); if (mk) mk[6] = gtime(); }
    return;
  }

  if (warp == 10) {
    // ---------------- SwiGLU converter: h slice (fp32 gate | up) -> bf16 B rows of the ring slot ----------------
    int it = 0, w2i = 0;
    for (int p = 0; p < 4; ++p) {
      int g0, nrb, beg, end;
      phase_range(S, p, b, G, g0, nrb, beg, end);
      if (p < 2) { it += end - beg; continue; }
      const int per = nrb * 16;
      for (int x = beg; x < end; ++x, ++it, ++w2i) {
        const int ge = x / per;
        const int e = S.grp[g0 + ge];
        const int ntok = S.cnt[e];
        const int slot = it % NST, hs = w2i % NHS;
        mbar_wait(hfull_a + 8 * hs, (w2i / NHS) & 1, A.err, 8);
        if (!(mode & 3)) {
          const float* hp = reinterpret_cast<const float*>(hstage + hs * HSB);
          __nv_bfloat16* xr = reinterpret_cast<__nv_bfloat16*>(smem + slot * STG + SW);
          for (int n = 0; n < ntok; ++n) {
            const float4* g4 = reinterpret_cast<const float4*>(hp + n * 512) + 2 * lane;
            const float4* u4 = reinterpret_cast<const float4*>(hp + n * 512 + 256) + 2 * lane;
            const float4 ga = g4[0], gb = g4[1], ua = u4[0], ub = u4[1];
            uint4 o;
            o.x = pack_bf2(swiglu(ga.x, ua.x), swiglu(ga.y, ua.y));
            o.y = pack_bf2(swiglu(ga.z, ua.z), swiglu(ga.w, ua.w));
            o.z = pack_bf2(swiglu(gb.x, ub.x), swiglu(gb.y, ub.y));
            o.w = pack_bf2(swiglu(gb.z, ub.z), swiglu(gb.w, ub.w));
            *reinterpret_cast<uint4*>(xr + n * XST + 8 * lane) = o;
          }
        }
        __syncwarp();
        if (lane == 0) { mbar_arrive(full_a + 8 * slot); mbar_arrive(hempty_a + 8 * hs); }
      }
    }
    return;
  }

  // ---------------- consumers (warps 0-7: warp w = tile w of the 128-row item) ----------------
  const bool nullc = (mode & 1) != 0;
  {
    const size_t n4 = (size_t)S.R * KDIM / 4;
    float4* yp = reinterpret_cast<float4*>(A.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)b * 256 + tid; i < n4; i += (size_t)G * 256) yp[i] = z;
  }
  const int g = lane >> 2, t = lane & 3;
  int it = 0;
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
          const float gs = S.gs13[e];
          const int col = 8 * tile + g;
          float* hb = A.h + ((size_t)(col >> 8) * RS + c0) * 512 + (col & 255);
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int n = 2 * t + hh;
            if (n < ntok) {
              red_f32(hb + (size_t)n * 512, acc[hh] * gs);
              red_f32(hb + (size_t)n * 512 + 256, acc[2 + hh] * gs);
            }
          }
        } else {
          const float gs = S.gs2[e];
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int n = 2 * t + hh;
            if (n < ntok) {
              float* yp = A.y + (size_t)(c0 + n) * KDIM + 16 * tile + g;
              const float v0 = acc[hh] * gs, v1 = acc[2 + hh] * gs;
              if (whole) { yp[0] = v0; yp[8] = v1; }
              else { red_f32(yp, v0); red_f32(yp + 8, v1); }
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
    const int tk = b >> 2, q = b & 3;
    // prefetch everything that does not depend on W2: routes, weights, (1 + w)
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
    if (tid < 8) s_w8[tid] = __ldg(A.wts + tk * 8 + tid);
    float4 nwv[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) nwv[k] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * k + 4 * tid));
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
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            float zz, a = ssp[j0 + jj];
            zz = bf16r(v[jj][k].x) * nwv[k].x; a += zz * zz;
            zz = bf16r(v[jj][k].y) * nwv[k].y; a += zz * zz;
            zz = bf16r(v[jj][k].z) * nwv[k].z; a += zz * zz;
            zz = bf16r(v[jj][k].w) * nwv[k].w; a += zz * zz;
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
      s_sc[tid] = (tid < nl) ? rsqrtf(a * (1.f / 4096.f) + 1e-8f) * s_w8[lv & 7] : 0.f;
    }
    bar_sync(8, 256);
    const float4 w = (q == 0) ? nwv[0] : (q == 1) ? nwv[1] : (q == 2) ? nwv[2] : nwv[3];
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
    // self-reset: h back to zero for the next call (every CTA's W2 h copies landed before sync2)
    {
      const int K3 = 4 * M;
      const size_t per_s = (size_t)S.R * 512 / 4;   // float4 per K-stage
      const size_t tot = 16 * per_s;
      float4* hz = reinterpret_cast<float4*>(A.h);
      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      for (size_t i = (size_t)b * 256 + tid; i < tot; i += (size_t)K3 * 256) {
        const size_t s = i / per_s, r = i - s * per_s;
        hz[s * (RS * 512 / 4) + r] = z;
      }
    }
    if (mk && tid == 0) mk[7] = gtime();
  }
}

constexpr int smem_bytes() { return NST * STG + NHS * HSB + (2 * NST + 2 * NHS + 1) * 8; }

void launch(const Args& a, int G) {
  constexpr int sm = smem_bytes();
  cudaFuncSetAttribute(q2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaFuncSetAttribute(q2_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  q2_kernel<<<G, NTH, sm, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
std::vector<int64_t> attrs() {
  constexpr int sm = smem_bytes();
  cudaFuncSetAttribute(q2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaFuncSetAttribute(q2_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, q2_kernel);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, q2_kernel, NTH, sm);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + sm), occ};
}
}  // namespace q2

void q2_run(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, torch::Tensor y,
            torch::Tensor sync, torch::Tensor err, torch::Tensor out, torch::Tensor marks,
            int64_t M, int64_t lo, int64_t G, int64_t mode) {
  TORCH_CHECK(M >= 1 && M <= q2::MMAX, "q2: M out of range");
  q2::Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ids = ids.data_ptr<int>(); a.wts = wts.data_ptr<float>();
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr()); a.ws2 = reinterpret_cast<const uint8_t*>(ws2.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>(); a.gsc2 = gsc2.data_ptr<float>(); a.nw = nw.data_ptr<float>();
  a.h = h.data_ptr<float>(); a.y = y.data_ptr<float>();
  a.sync = reinterpret_cast<unsigned*>(sync.data_ptr<int>()); a.err = err.data_ptr<int>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.marks = marks.numel() >= G * 16 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  a.M = (int)M; a.lo = (int)lo; a.mode = (int)mode;
  q2::launch(a, (int)G);
}
std::vector<int64_t> q2_attrs() { return q2::attrs(); }
"""

Q2_CPP = r"""
#include <torch/extension.h>
#include <vector>
void q2_run(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, torch::Tensor y,
            torch::Tensor sync, torch::Tensor err, torch::Tensor out, torch::Tensor marks,
            int64_t M, int64_t lo, int64_t G, int64_t mode);
std::vector<int64_t> q2_attrs();
"""

Q2_EXPERT_BYTES = 768 * 36864    # 28.31 MB per touched expert


def q2_build(arch="sm_90a", verbose=False):
    from torch.utils.cpp_extension import load_inline
    gen = (["-gencode=arch=compute_90a,code=sm_90a"] if arch == "sm_90a"
           else ["-gencode=arch=compute_86,code=sm_86", "-DQ2_NST=2"])
    tag = hashlib.sha1((Q2_CPP + Q2_CUDA + arch).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"q2_moe_{tag}", cpp_sources=Q2_CPP, cuda_sources=Q2_CUDA, functions=["q2_run", "q2_attrs"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills"] + gen, verbose=verbose)
    return mod, time.time() - t0


def q2_bufs(dev, G):
    return {
        "h": torch.zeros((16 * 64 * 512,), dtype=torch.float32, device=dev),
        "y": torch.zeros((64, 4096), dtype=torch.float32, device=dev),
        "sync": torch.zeros((8,), dtype=torch.int32, device=dev),
        "err": torch.zeros((8,), dtype=torch.int32, device=dev),
        "marks": torch.zeros((G * 16,), dtype=torch.int64, device=dev),
        "nomark": torch.zeros((0,), dtype=torch.int64, device=dev),
    }


def q2_fn(mod, st, bufs, data, G, mode=0, mark=False):
    """Closure for one call (graph-capturable): returns out [M, 4096] bf16."""
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    lo = int(data["config"]["local_start"])
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    mk = bufs["marks"] if mark else bufs["nomark"]

    def f():
        mod.q2_run(x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], bufs["h"], bufs["y"],
                   bufs["sync"], bufs["err"], out, mk, m, lo, G, mode)
        return out
    return f


def q2_case(bs, seed, dev, dist="normal"):
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


def q2_touched(data):
    ids = data["topk_ids"].long() - int(data["config"]["local_start"])
    loc = ids[(ids >= 0) & (ids < 32)]
    return int(torch.unique(loc).numel())


def q2_rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


class Q2Timer:
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


def q2_marks_summary(bufs, G):
    mv = bufs["marks"].view(-1, 16)[:G].tolist()
    t0 = min(r[0] for r in mv)

    def col(k):
        return sorted((r[k] - t0) / 1e3 for r in mv if r[k] > 0)

    def p50max(v):
        return (statistics.median(v), v[-1]) if v else (float("nan"), float("nan"))
    out = {}
    for name, k in (("first_copy", 1), ("w13a_end", 2), ("w13b_end", 3), ("w2a_end", 4), ("w2_end", 5),
                    ("sync2_pub", 6), ("fin_end", 7), ("depA", 8), ("depB", 9), ("prod_done", 10), ("fin_seen", 11),
                    ("needA", 12), ("needB", 13)):
        out[name] = p50max(col(k))
    for nm, a, b2 in (("stallA", 12, 8), ("stallB", 13, 9)):
        v = sorted((r[b2] - r[a]) / 1e3 for r in mv if r[a] > 0 and r[b2] > 0)
        out[nm] = p50max(v) if v else (0.0, 0.0)
    return out


def q2_lab(st, k2a_fn, k2a_state, log=print, rounds=60, inputs=4, seeds0=919001, sizes=(1, 4, 8), arch="sm_90a"):
    """R-moe1-1: arms K (shipped K2A), Q (Q2), Q0 (Q2 null consumers), S (Q2 pure stream). Same routings as R-moe1-0."""
    dev = st["ws13"].device
    mod, dt = q2_build(arch)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    a = list(mod.q2_attrs())
    log(f"[q2] attrs (regs, local, smem, occ) {a}; built in {dt:.0f}s, {sms} SMs")
    G = min(a[3], 2) * sms
    if a[3] < 1:
        log("[q2] cannot be resident: lab skipped")
        return []
    timer = Q2Timer()
    ov = timer.overhead()
    log(f"[q2] graph overhead {ov:.2f} us; G {G}")
    rows = []
    for bs in sizes:
        for r in range(inputs):
            d = q2_case(bs, seeds0 + 1000 * bs + r, dev)
            ne = q2_touched(d)
            B = ne * Q2_EXPERT_BYTES
            spec = {"Q": 0, "Q0": 1, "S": 15}
            bufs = {nm: q2_bufs(dev, G) for nm in spec}
            arms = {"K": lambda d=d: k2a_fn(d, k2a_state)}
            arms.update({nm: q2_fn(mod, st, bufs[nm], d, G, md) for nm, md in spec.items()})
            ref = arms["K"]().float().clone()
            torch.cuda.synchronize()
            o1 = arms["Q"]().float().clone()
            o2 = arms["Q"]().float().clone()
            torch.cuda.synchronize()
            rel = (q2_rel(o1, ref), q2_rel(o2, ref), bool(torch.isfinite(o1).all()))
            graphs = {nm: timer.capture(fn)[0] for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            o3 = graphs and arms["Q"]().float()
            torch.cuda.synchronize()
            rel3 = q2_rel(o3, ref)
            tl = {}
            for nm in ("Q", "Q0"):
                bufs[nm]["marks"].zero_()
                f = q2_fn(mod, st, bufs[nm], d, G, spec[nm], mark=True)
                with torch.cuda.stream(timer.stream):
                    torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                    f()
                torch.cuda.synchronize()
                tl[nm] = q2_marks_summary(bufs[nm], G)
            e_all = {nm: [int(v) for v in bufs[nm]["err"].tolist()[:2]] for nm in spec}
            rows.append({"bs": bs, "ne": ne, "B": B, "us": us, "rel": rel, "rel3": rel3, "err": e_all, "tl": tl})
            log(f"[q2] bs={bs} r={r} experts {ne} ({B / 1e6:.1f} MB): " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items())
                + " | " + " ".join(f"{nm}/K {us[nm] / us['K']:.3f}" for nm in us if nm != "K")
                + f" | B/4.4TB/s {B / 4.4e6:.1f}")
            log(f"[q2]   rel Q vs K {rel[0]:.1e}/{rel[1]:.1e}/{rel[2]} after-timing {rel3:.1e} | err {e_all}")
            for nm, s in tl.items():
                log(f"[q2]   TL {nm}: " + " ".join(f"{k} {a2:.1f}/{b2:.1f}" for k, (a2, b2) in s.items()))
            del graphs, bufs
            torch.cuda.empty_cache()
    log("[q2] GATE (R-moe1-1: median Q/K bs1 <= 0.85 and bs8 <= 0.92, rel <= 1e-3, err 0; else KILL)")
    verdict = True
    for bs in sizes:
        rr = [x for x in rows if x["bs"] == bs]
        if not rr:
            continue
        ratios = [x["us"]["Q"] / x["us"]["K"] for x in rr]
        ok_c = all(x["rel"][0] <= 1e-3 and x["rel"][1] <= 1e-3 and x["rel"][2] and x["rel3"] <= 1e-3
                   and x["err"]["Q"][0] == 0 for x in rr)
        mr = statistics.median(ratios)
        lim = 0.85 if bs == 1 else 0.92
        okb = ok_c and mr <= lim
        if bs in (1, 8):
            verdict &= okb
        log(f"[q2]   bs={bs}: median Q/K {mr:.3f} (limit {lim}{'' if bs in (1, 8) else ', info'}), correctness "
            f"{'ok' if ok_c else 'FAIL'} -> {'pass' if okb else 'FAIL'}")
    log(f"[q2] GATE {'GO' if verdict else 'NO-GO'}")
    del timer
    torch.cuda.empty_cache()
    return rows

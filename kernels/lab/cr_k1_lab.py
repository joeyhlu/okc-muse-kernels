"""CR-K1: step-0 lab of the clean-room MoE (notes/study/moe_cleanroom.md), W13 phase only, timing + correctness.

One launch, 1 CTA/SM x 544 threads: 16 consumer warps + 1 producer warp, one 4-slot ring.
  slot = 512 W13 rows x 128 K of one expert (36,864 B: warp w owns intermediate dims d0 = 256 b + 16 w as two m16 tiles,
         gate rows d0..d0+15 and up rows 4096 + d0.., packed with K2A's BITPOS3 words / re-encoded scales)
         + the expert's X slice for this K-stage, <= 64 tokens x (128 + 8 pad) bf16 (one bulk copy).
  consumer: every warp decodes its 2 tiles once per stage and runs ALL n8 token groups of the pass on them
         (m16n8k16 mma.sync, B straight from the X slot, 2 mma per B load); no hot/cold roles.
  plan (host, lab only): units = (expert, 512-row block, 64-token pass) x 32 K-stages, priced per stage max(1, ng * kappa);
         CTA c takes the c-th 1/G of the total cost as a contiguous stage range. A range end inside a unit makes a partial
         piece: FP32 partial to scratch, stage counter, the last contributor sums the others' partials and does SwiGLU.
         Each CTA runs its partial pieces FIRST, so last-arriver work happens early, not in the tail.
  out: inter [routes][4096] bf16 = silu(x Wg^T) * (x Wu^T) per routed (token, expert).
mode bit 0: null consumers (skeleton: plan + ring + weight + X copies only).
sm_90a: bulk copies (evict_first on single-pass weights). sm_86 (local correctness only): cp.async, 1-slot ring.
"""
import hashlib
import math
import statistics
import time

import torch

CR_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace crk {
#ifndef CR_NST
#define CR_NST 4
#endif
#ifndef CR_NGMAX
#define CR_NGMAX 8
#endif
constexpr int NST = CR_NST, NGMAX = CR_NGMAX;
#ifndef CR_NCW
#define CR_NCW 16
#endif
constexpr int NCW = CR_NCW;                // 16: 1 CTA/SM, 4 consumer WGs + 1 producer WG (setmaxnreg); 8: 2 CTAs/SM, 8 + 1 warps
constexpr int NTH = 32 * (NCW + (NCW == 16 ? 4 : 1));
constexpr int MINB = NCW == 16 ? 1 : 2;
constexpr int HB = 16 / NCW;               // CR blocks per 512-row packed block
#ifndef CR_PREG
#define CR_PREG 24
#endif
#ifndef CR_CREG
#define CR_CREG 112
#endif
constexpr int CB = NCW * 2048;             // codes per slot: NCW warps x 4 parts x 32 lanes x 16 B
constexpr int WB = NCW * 2304;             // + scales NCW x 32 lanes x 8 B
constexpr int XRB = 272;                   // X row bytes per stage: 128 bf16 + 16 B pad (conflict-free B loads)
constexpr int XB = 8 * NGMAX * XRB;
constexpr int SLOT = WB + XB;
constexpr int OFF_BAR = NST * SLOT;
constexpr int SMEM = OFF_BAR + 2 * NST * 8;
constexpr int PMAX = 48;
constexpr int PART = 512 * 64;             // floats per scratch slot (16 warps x 2 tiles x 8 groups x 4 x 32 lanes)
constexpr long long WAIT_BUDGET = 1ll << 31;
#if __CUDA_ARCH__ >= 900
constexpr int FULLCNT = 1;
#else
constexpr int FULLCNT = 32;
#endif

struct Args {
  const uint8_t* ws;        // [32][16][32][WB]
  const uint8_t* xp;        // packed X
  const int* units;         // [nu][8]: e, b, ntok, ng, rbase, xoff, multipass, 0
  const int* cuts;          // [G + 1] global stage cut points
  const float* gsc;         // [32] w13_gs * 256
  __nv_bfloat16* inter;     // [R][4096]
  float* scratch;           // [G][2][PART]
  int* cnt;                 // [nu] self-resetting stage counters
  int* err; long long* marks;
  int G, mode;
  // full-call fields: phase 0 = W13 (+ SwiGLU, inter into xp2 in W2's X layout, zero ss), 1 = W2 (+ post-norm partials)
  int phase, nbe, R;
  uint8_t* xp2; float* ss; __nv_bfloat16* ybuf; const float* nw;
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 words: o[j] = codes (2j, 2j + 1)
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
__device__ __forceinline__ bool mbar_wait(uint32_t b, uint32_t par, int* err, int code) {
  if (mbar_try(b, par)) return true;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return false; }
  }
  return true;
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_h(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
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
__device__ __forceinline__ float silu_mul(float g, float u) { return g / (1.f + __expf(-g)) * u; }
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ void pdl_launch() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// One piece's stage loop, NG token groups (8 tokens each) compiled in: in-place accumulation, no predicated mma.
template <int NG>
__device__ __forceinline__ void consume_piece(const uint8_t* smem, uint32_t full_a, uint32_t empty_a, int* err, int s0,
                                              int s1, int& it, bool& dead, bool null_c, int warp, int lane,
                                              long long* mk, int tid, float (&acc)[2][NGMAX][4]) {
  const int g = lane >> 2, t = lane & 3;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    if (!dead && !mbar_wait(full_a + 8 * slot, (it / NST) & 1, err, 1)) dead = true;
    if (it == 0 && mk && tid == 0) mk[2] = gtime();
    if (!null_c) {
      const uint8_t* sb = smem + slot * SLOT;
      const uint4* cw = reinterpret_cast<const uint4*>(sb) + warp * 128 + lane;
      const uint2 sc = reinterpret_cast<const uint2*>(sb + CB)[warp * 32 + lane];
      const uint8_t* xs = sb + WB + g * XRB + t * 32;
#pragma unroll
      for (int kbo = 0; kbo < 2; ++kbo) {
        const uint4 q0 = cw[32 * kbo], q1 = cw[32 * (2 + kbo)];
        const uint32_t Sg0 = __byte_perm(sc.x, 0u, 0x4040u + 0x0202u * kbo) * 16u + 0x75807580u;
        const uint32_t S80 = __byte_perm(sc.x, 0u, 0x4141u + 0x0202u * kbo) * 16u + 0x75807580u;
        const uint32_t Sg1 = __byte_perm(sc.y, 0u, 0x4040u + 0x0202u * kbo) * 16u + 0x75807580u;
        const uint32_t S81 = __byte_perm(sc.y, 0u, 0x4141u + 0x0202u * kbo) * 16u + 0x75807580u;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t a0g[4], a08[4], a1g[4], a18[4];
          deq4(h ? q0.y : q0.x, a0g); deq4(h ? q0.w : q0.z, a08);
          deq4(h ? q1.y : q1.x, a1g); deq4(h ? q1.w : q1.z, a18);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            a0g[j] = bmul2(a0g[j], Sg0); a08[j] = bmul2(a08[j], S80);
            a1g[j] = bmul2(a1g[j], Sg1); a18[j] = bmul2(a18[j], S81);
          }
#pragma unroll
          for (int gi = 0; gi < NG; ++gi) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xs + gi * 8 * XRB + kbo * 128 + h * 16);
            mma16816(acc[0][gi], a0g[0], a08[0], a0g[1], a08[1], xv.x, xv.y);
            mma16816(acc[1][gi], a1g[0], a18[0], a1g[1], a18[1], xv.x, xv.y);
            mma16816(acc[0][gi], a0g[2], a08[2], a0g[3], a08[3], xv.z, xv.w);
            mma16816(acc[1][gi], a1g[2], a18[2], a1g[3], a18[3], xv.z, xv.w);
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
}

__global__ void __launch_bounds__(NTH, MINB) cr_k1(const Args A) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_pu[PMAX], s_ps0[PMAX], s_ps1[PMAX], s_pslot[PMAX];
  __shared__ int s_np, s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x;
  long long* mk = A.marks ? A.marks + (size_t)c * 8 : nullptr;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
  const uint32_t ring_a = sa(smem), full_a = sa(bars), empty_a = sa(bars + NST);
  if (tid == 0) {
    if (mk) mk[0] = gtime();
    if (A.phase == 0) pdl_launch();
    for (int i = 0; i < NST; ++i) { mbar_init(full_a + 8 * i, FULLCNT); mbar_init(empty_a + 8 * i, NCW); }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    // piece list: partial pieces first (slot 0 = the piece holding cuts[c], slot 1 = the one holding cuts[c+1]-1)
    const int a = A.cuts[c], bnd = A.cuts[c + 1];
    int np = 0;
    if (bnd > a) {
      const int ua = a >> 5, ub = (bnd - 1) >> 5;
      if (ua == ub) {
        const int s0 = a & 31, s1 = bnd - 32 * ua;
        s_pu[0] = ua; s_ps0[0] = s0; s_ps1[0] = s1; s_pslot[0] = (s0 == 0 && s1 == 32) ? -1 : 0; np = 1;
      } else {
        const bool hp = (a & 31) != 0, tp = (bnd - 32 * ub) != 32;
        if (hp) { s_pu[np] = ua; s_ps0[np] = a & 31; s_ps1[np] = 32; s_pslot[np] = 0; ++np; }
        if (tp) { s_pu[np] = ub; s_ps0[np] = 0; s_ps1[np] = bnd - 32 * ub; s_pslot[np] = 1; ++np; }
        for (int u = hp ? ua + 1 : ua; u <= (tp ? ub - 1 : ub); ++u) {
          if (np >= PMAX) { set_err(A.err, 9); break; }
          s_pu[np] = u; s_ps0[np] = 0; s_ps1[np] = 32; s_pslot[np] = -1; ++np;
        }
      }
    }
    s_np = np;
  }
  __syncthreads();
  const int np = s_np;
  if (A.phase == 1) pdl_wait();                      // K1's inter (xp2) and zeroed ss are visible after this
  else if (A.ss) for (int i = c * NTH + tid; i < A.R; i += A.G * NTH) A.ss[i] = 0.f;

  if (warp >= NCW) {
#if __CUDA_ARCH__ >= 900 && CR_NCW == 16
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(CR_PREG) : "memory");
#endif
    if (warp != NCW) return;
    // ---------------- producer ----------------
#if __CUDA_ARCH__ >= 900
    const uint64_t pol = policy_evict_first();
#endif
    int it = 0;
    bool dead = false;                      // after one bounded-wait timeout: stop waiting (err is set), never hang
    for (int p = 0; p < np; ++p) {
      const int* U = A.units + 8 * s_pu[p];
      const int e = U[0], b = U[1], ng = U[3], xoff = U[5], multi = U[6];
      const uint32_t xbytes = (uint32_t)(8 * ng * XRB);
      for (int s = s_ps0[p]; s < s_ps1[p]; ++s, ++it) {
        const int slot = it % NST;
        if (it >= NST && !dead && !mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, A.err, 2)) dead = true;
        const uint8_t* wsrc = A.ws + ((size_t)((e * A.nbe + b / HB) * 32 + s)) * 36864 + (b % HB) * CB;
        const uint8_t* ssrc = A.ws + ((size_t)((e * A.nbe + b / HB) * 32 + s)) * 36864 + 32768 + (b % HB) * (NCW * 256);
        const uint8_t* xsrc = A.xp + (size_t)xoff + (size_t)s * xbytes;
        const uint32_t dst = ring_a + slot * SLOT;
#if __CUDA_ARCH__ >= 900
        if (lane == 0) {
          mbar_expect_tx(full_a + 8 * slot, WB + xbytes);
          if (NCW == 16) {
            if (multi) bulk_g2s(dst, wsrc, WB, full_a + 8 * slot);
            else bulk_g2s_h(dst, wsrc, WB, full_a + 8 * slot, pol);
          } else {
            if (multi) { bulk_g2s(dst, wsrc, CB, full_a + 8 * slot); bulk_g2s(dst + CB, ssrc, WB - CB, full_a + 8 * slot); }
            else { bulk_g2s_h(dst, wsrc, CB, full_a + 8 * slot, pol); bulk_g2s_h(dst + CB, ssrc, WB - CB, full_a + 8 * slot, pol); }
          }
          bulk_g2s(dst + WB, xsrc, xbytes, full_a + 8 * slot);
        }
#else
        warp_copy(lane, dst, wsrc, CB);
        warp_copy(lane, dst + CB, ssrc, WB - CB);
        warp_copy(lane, dst + WB, xsrc, xbytes);
        cpa_arrive(full_a + 8 * slot);
#endif
        if (it == 0 && mk && lane == 0) mk[1] = gtime();
      }
    }
    if (mk && lane == 0) mk[4] = gtime();
    return;
  }

  // ---------------- consumers ----------------
#if __CUDA_ARCH__ >= 900 && CR_NCW == 16
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(CR_CREG) : "memory");
#endif
  const int g = lane >> 2, t = lane & 3;
  const bool null_c = (A.mode & 1) != 0;
  int it = 0;
  bool dead = false;
  float acc[2][NGMAX][4];
  for (int p = 0; p < np; ++p) {
    const int u = s_pu[p];
    const int* U = A.units + 8 * u;
    const int e = U[0], b = U[1], ntok = U[2], ng = U[3], rbase = U[4], xoff = U[5];
#pragma unroll
    for (int ti = 0; ti < 2; ++ti)
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[ti][gi][i] = 0.f;
    {
      const int s0 = s_ps0[p], s1 = s_ps1[p];
#define CR_CP(N) consume_piece<N>(smem, full_a, empty_a, A.err, s0, s1, it, dead, null_c, warp, lane, mk, tid, acc)
      switch (ng) {
        case 1: CR_CP(1); break;
        case 2: CR_CP(2); break;
        case 3: CR_CP(3); break;
        case 4: CR_CP(4); break;
        case 5: CR_CP(5); break;
        case 6: CR_CP(6); break;
#if CR_NGMAX >= 7
        case 7: CR_CP(7); break;
#endif
#if CR_NGMAX >= 8
        default: CR_CP(8); break;
#else
        default: CR_CP(6); break;
#endif
      }
#undef CR_CP
    }
    if (null_c) continue;
    // ---------------- piece epilogue ----------------
    const int pslot = s_pslot[p];
    const float gs = A.gsc[e];
    const int d0 = 16 * NCW * b + 16 * warp;
    bool fin = pslot < 0;
    if (!fin) {
      float* my = A.scratch + ((size_t)(c * 2 + pslot)) * PART + (size_t)warp * (2 * NGMAX * 4 * 32);
#pragma unroll
      for (int ti = 0; ti < 2; ++ti)
#pragma unroll
        for (int gi = 0; gi < NGMAX; ++gi)
          if (gi < ng) {
#pragma unroll
            for (int i = 0; i < 4; ++i) my[((ti * NGMAX + gi) * 4 + i) * 32 + lane] = acc[ti][gi][i];
          }
      bar_sync(1, 32 * NCW);
      if (tid == 0) {
        __threadfence();
        const int n = s_ps1[p] - s_ps0[p];
        const int old = atomicAdd(A.cnt + u, n);
        s_last = (old + n == 32);
        if (old + n > 32) set_err(A.err, 5);
        __threadfence();
      }
      bar_sync(1, 32 * NCW);
      fin = s_last != 0;
      if (fin) {
        // add the other contributors' partials (CTAs whose ranges intersect unit u)
        const int lo = 32 * u, hi = 32 * u + 32;
        int c0 = 0, c1 = A.G;                       // first c' with cuts[c' + 1] > lo
        while (c0 < c1) { const int m = (c0 + c1) >> 1; if (A.cuts[m + 1] > lo) c1 = m; else c0 = m + 1; }
        for (int cc = c0; cc < A.G && A.cuts[cc] < hi; ++cc) {
          if (cc == c || A.cuts[cc + 1] <= A.cuts[cc]) continue;
          const int os = ((A.cuts[cc] >> 5) == u) ? 0 : 1;
          const float* o = A.scratch + ((size_t)(cc * 2 + os)) * PART + (size_t)warp * (2 * NGMAX * 4 * 32);
#pragma unroll
          for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int gi = 0; gi < NGMAX; ++gi)
              if (gi < ng) {
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[ti][gi][i] += __ldcg(o + ((ti * NGMAX + gi) * 4 + i) * 32 + lane);
              }
        }
        if (tid == 0) A.cnt[u] = 0;
      }
    }
    if (fin && A.phase == 0) {
      __nv_bfloat16* x2 = A.xp2 ? reinterpret_cast<__nv_bfloat16*>(A.xp2 + xoff) : nullptr;
      const int sstr = 8 * ng * (XRB / 2);          // elements per K-stage of this pass in W2's X layout
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
        if (gi < ng) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int tok = 8 * gi + 2 * t + (i & 1);
            const int dim = d0 + g + 8 * (i >> 1);
            if (tok < ntok) {
              const __nv_bfloat16 v = __float2bfloat16_rn(silu_mul(acc[0][gi][i] * gs, acc[1][gi][i] * gs));
              if (A.inter) A.inter[(size_t)(rbase + tok) * 4096 + dim] = v;
              if (x2) x2[(size_t)(dim >> 7) * sstr + tok * (XRB / 2) + (dim & 127)] = v;
            }
          }
        }
    }
    if (fin && A.phase == 1) {
      const int o0 = 32 * NCW * b + 32 * warp;
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
        if (gi < ng) {
          float q0 = 0.f, q1 = 0.f;
#pragma unroll
          for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int tok = 8 * gi + 2 * t + (i & 1);
              const int row = o0 + 16 * ti + g + 8 * (i >> 1);
              const float yb = bf16r(bf16r(acc[ti][gi][i] * gs) * A.nw[row]);
              if (i & 1) q1 += yb * yb; else q0 += yb * yb;
              if (tok < ntok) A.ybuf[(size_t)(rbase + tok) * 4096 + row] = __float2bfloat16_rn(yb);
            }
#pragma unroll
          for (int m = 4; m < 32; m <<= 1) {
            q0 += __shfl_xor_sync(0xffffffffu, q0, m);
            q1 += __shfl_xor_sync(0xffffffffu, q1, m);
          }
          const int tk = 8 * gi + 2 * t;
          if (g == 0) {
            if (tk < ntok) atomicAdd(A.ss + rbase + tk, q0);
            if (tk + 1 < ntok) atomicAdd(A.ss + rbase + tk + 1, q1);
          }
        }
    }
  }
  if (mk && tid == 0) mk[3] = gtime();
}
// K3: out[t] = sum over t's local routes of w * rsqrt(mean(yb^2) + eps) * yb (yb = bf16(bf16(y) * (1 + w_post)))
__global__ void __launch_bounds__(256) cr_fin(const __nv_bfloat16* ybuf, const float* ss, const int* tokr,
                                              const float* tokw, __nv_bfloat16* out) {
  pdl_wait();
  const int tt = blockIdx.x, d = blockIdx.y * 1024 + threadIdx.x * 4;
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < 8; ++k) {
    const int r = tokr[tt * 8 + k];
    if (r < 0) continue;
    const float sc = tokw[tt * 8 + k] * rsqrtf(ss[r] * (1.f / 4096.f) + 1e-8f);
    const uint2 v = *reinterpret_cast<const uint2*>(ybuf + (size_t)r * 4096 + d);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&v.x), b2 = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
    o[0] += sc * __low2float(a); o[1] += sc * __high2float(a); o[2] += sc * __low2float(b2); o[3] += sc * __high2float(b2);
  }
  __nv_bfloat162 r0 = __floats2bfloat162_rn(o[0], o[1]), r1 = __floats2bfloat162_rn(o[2], o[3]);
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&r0); w.y = *reinterpret_cast<uint32_t*>(&r1);
  *reinterpret_cast<uint2*>(out + (size_t)tt * 4096 + d) = w;
}
}  // namespace crk

template <typename K, typename... Args>
static void cr_launch(K kern, dim3 grid, dim3 block, size_t smem, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

void cr_full(torch::Tensor x1, torch::Tensor ws13, torch::Tensor units1, torch::Tensor cuts1, torch::Tensor gsc13,
             torch::Tensor xp2, torch::Tensor ws2, torch::Tensor units2, torch::Tensor cuts2, torch::Tensor gsc2,
             torch::Tensor nw, torch::Tensor ss, torch::Tensor ybuf, torch::Tensor tokr, torch::Tensor tokw,
             torch::Tensor out, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err, int64_t G, int64_t R,
             int64_t pdl) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(crk::cr_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, crk::SMEM));
    attr = true;
  }
  crk::Args A = {};
  A.ws = ws13.data_ptr<uint8_t>(); A.xp = x1.data_ptr<uint8_t>(); A.units = units1.data_ptr<int>();
  A.cuts = cuts1.data_ptr<int>(); A.gsc = gsc13.data_ptr<float>(); A.inter = nullptr;
  A.scratch = scratch.data_ptr<float>(); A.cnt = cnt.data_ptr<int>(); A.err = err.data_ptr<int>(); A.marks = nullptr;
  A.G = (int)G; A.mode = 0; A.phase = 0; A.nbe = 16 / crk::HB * crk::HB == 16 ? 16 : 16; A.R = (int)R;
  A.xp2 = xp2.data_ptr<uint8_t>(); A.ss = ss.data_ptr<float>(); A.ybuf = nullptr; A.nw = nullptr;
  cr_launch(crk::cr_k1, dim3((unsigned)G), dim3(crk::NTH), crk::SMEM, false, A);
  crk::Args B = A;
  B.ws = ws2.data_ptr<uint8_t>(); B.xp = xp2.data_ptr<uint8_t>(); B.units = units2.data_ptr<int>();
  B.cuts = cuts2.data_ptr<int>(); B.gsc = gsc2.data_ptr<float>(); B.cnt = cnt.data_ptr<int>() + 4096;
  B.phase = 1; B.nbe = 8; B.xp2 = nullptr;
  B.ybuf = reinterpret_cast<__nv_bfloat16*>(ybuf.data_ptr<at::BFloat16>()); B.nw = nw.data_ptr<float>();
  cr_launch(crk::cr_k1, dim3((unsigned)G), dim3(crk::NTH), crk::SMEM, pdl != 0, B);
  const int M = (int)out.size(0);
  cr_launch(crk::cr_fin, dim3((unsigned)M, 4), dim3(256), 0, pdl != 0,
            (const __nv_bfloat16*)B.ybuf, (const float*)ss.data_ptr<float>(), (const int*)tokr.data_ptr<int>(),
            (const float*)tokw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
}

void cr_run(torch::Tensor xp, torch::Tensor ws, torch::Tensor units, torch::Tensor cuts, torch::Tensor gsc,
            torch::Tensor inter, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err, torch::Tensor marks,
            int64_t G, int64_t mode) {
  crk::Args A;
  A.ws = ws.data_ptr<uint8_t>(); A.xp = xp.data_ptr<uint8_t>(); A.units = units.data_ptr<int>();
  A.cuts = cuts.data_ptr<int>(); A.gsc = gsc.data_ptr<float>();
  A.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr<at::BFloat16>());
  A.scratch = scratch.data_ptr<float>(); A.cnt = cnt.data_ptr<int>(); A.err = err.data_ptr<int>();
  A.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr<int64_t>()) : nullptr;
  A.G = (int)G; A.mode = (int)mode;
  A.phase = 0; A.nbe = 16; A.R = 0; A.xp2 = nullptr; A.ss = nullptr; A.ybuf = nullptr; A.nw = nullptr;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(crk::cr_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, crk::SMEM));
    attr = true;
  }
  crk::cr_k1<<<(unsigned)G, crk::NTH, crk::SMEM, at::cuda::getCurrentCUDAStream()>>>(A);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> cr_attrs() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, crk::cr_k1));
  C10_CUDA_CHECK(cudaFuncSetAttribute(crk::cr_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, crk::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, crk::cr_k1, crk::NTH, crk::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)crk::SMEM, (int64_t)occ, (int64_t)crk::NGMAX,
          (int64_t)crk::NST, (int64_t)crk::NCW};
}
"""

CR_CPP = r"""
#include <torch/extension.h>
#include <vector>
void cr_run(torch::Tensor xp, torch::Tensor ws, torch::Tensor units, torch::Tensor cuts, torch::Tensor gsc,
            torch::Tensor inter, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err, torch::Tensor marks,
            int64_t G, int64_t mode);
std::vector<int64_t> cr_attrs();
void cr_full(torch::Tensor x1, torch::Tensor ws13, torch::Tensor units1, torch::Tensor cuts1, torch::Tensor gsc13,
             torch::Tensor xp2, torch::Tensor ws2, torch::Tensor units2, torch::Tensor cuts2, torch::Tensor gsc2,
             torch::Tensor nw, torch::Tensor ss, torch::Tensor ybuf, torch::Tensor tokr, torch::Tensor tokw,
             torch::Tensor out, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err, int64_t G, int64_t R,
             int64_t pdl);
"""

CR_WB = 36864
CR_XRB = 272
CR_PART = 512 * 64


def cr_build(arch="sm_90a", ncw=16, ngmax=8, nst=4, verbose=False):
    from torch.utils.cpp_extension import load_inline
    gen = (["-gencode=arch=compute_90a,code=sm_90a", f"-DCR_NST={nst}"] if arch == "sm_90a"
           else ["-gencode=arch=compute_86,code=sm_86", "-DCR_NST=1"])
    gen = gen + [f"-DCR_NGMAX={ngmax}", f"-DCR_NCW={ncw}"]
    tag = hashlib.sha1((CR_CPP + CR_CUDA + arch + str((ncw, ngmax, nst))).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"cr_k1_{tag}", cpp_sources=CR_CPP, cuda_sources=CR_CUDA, functions=["cr_run", "cr_attrs", "cr_full"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills"] + gen,
                      verbose=verbose)
    return mod, time.time() - t0


# ---- generic NVFP4 decode math (BITPOS3 words + exact E4M3 re-encode; the same lossless math K2A's prepare uses) ----
CR_BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (0, 1, 2, 9), (16, 17, 18, 25),
              (26, 27, 28, 3), (10, 11, 12, 19), (4, 5, 14, 13), (20, 21, 30, 29))
_CR_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def cr_dequant(packed, sf, gs):
    grid = torch.tensor(_CR_GRID, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)
    return vals.view(*vals.shape[:-2], k)


def cr_reencode_scales(sf_u8):
    """E4M3 bytes -> (b', neg, zero, has_nan); b' = (e+10)<<3 | f so that bf16 bits b'*16 + 0x7580 = |sf| * 2^118."""
    b = sf_u8.to(torch.int32)
    sign, ex, m = b >> 7, (b >> 3) & 15, b & 7
    nan = (ex == 15) & (m == 7)
    zero = (ex == 0) & (m == 0)
    sub = (ex == 0) & (m != 0)
    lut_e5 = torch.tensor([0, 1, 2, 2, 3, 3, 3, 3], dtype=torch.int32, device=b.device)
    lut_f = torch.tensor([0, 0, 0, 4, 0, 2, 4, 6], dtype=torch.int32, device=b.device)
    bp = ((ex + 3) << 3) | m
    bp = torch.where(sub, (lut_e5[m.long()] << 3) | lut_f[m.long()], bp)
    bp = torch.where(zero, torch.full_like(bp, 24), bp)
    return bp.to(torch.uint8), (sign == 1) & ~zero, zero, bool(nan.any())


def cr_pack_words(codes, bitpos=CR_BITPOS3):
    c = codes.to(torch.int64)
    w = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for i, (p0, p1, p2, ps) in enumerate(bitpos):
        ci = c[..., i]
        w |= ((ci & 1) << p0) | (((ci >> 1) & 1) << p1) | (((ci >> 2) & 1) << p2) | (((ci >> 3) & 1) << ps)
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w)
    return w.to(torch.int32)


def cr_prepack_rows(w, sf):
    """w uint8 [N, K/2], sf e4m3 [N, K/16] -> (wq int32 [N/16, K/64, 32, 4], sq uint8 [N/16, K/128, 32, 4]); natural row order.
    words (x, y) = row g at k = 64 kb + 16 t + (0..7, 8..15), (z, w) = row g + 8; sq bytes 2 par + hi of group 4 kb + t."""
    n, k2 = w.shape
    k = 2 * k2
    rt, nkb = n // 16, k // 64
    sfb = sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf
    bp, neg, zero, bad = cr_reencode_scales(sfb)
    if bad:
        raise ValueError("NaN block scale in NVFP4 weights")
    codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(n, k)
    if bool(neg.any()) or bool(zero.any()):
        cb = codes.view(n, k // 16, 16)
        cb = torch.where(neg.unsqueeze(-1), cb ^ 8, cb)
        cb = torch.where(zero.unsqueeze(-1), torch.zeros_like(cb), cb)
        codes = cb.reshape(n, k)
    c = codes.view(rt, 2, 8, nkb, 4, 2, 8).permute(0, 3, 2, 4, 1, 5, 6).reshape(rt, nkb, 32, 4, 8)
    wq = cr_pack_words(c)
    sq = bp.view(rt, 2, 8, nkb // 2, 2, 4).permute(0, 3, 2, 5, 4, 1).reshape(rt, nkb // 2, 32, 4)
    return wq.contiguous(), sq.contiguous()


def cr_prepack(w13, w13_sf, gs13):
    """w13 uint8 [E, 8192, 2048], w13_sf e4m3 [E, 8192, 256] -> ws uint8 [E][16 b][32 s][36864] (CR slot layout),
    gsc float32 [E] (= gs * 256: the decoded weights carry 2^-8)."""
    E = w13.shape[0]
    ws = torch.empty((E, 16, 32, CR_WB), dtype=torch.uint8, device=w13.device)
    for e in range(E):
        wq, sq = cr_prepack_rows(w13[e], w13_sf[e])   # [512, 64, 32, 4] i32, [512, 32, 32, 4] u8
        wq = wq.view(2, 16, 16, 32, 2, 32, 4)            # (gate|up, b, w, s, kbo, lane, word)
        codes = wq.permute(1, 3, 2, 0, 4, 5, 6).contiguous().view(16, 32, 16 * 4 * 32 * 4)    # (b, s, w, tile, kbo, lane, word)
        sq32 = sq.contiguous().view(torch.int32).view(2, 16, 16, 32, 32)                      # (tile, b, w, s, lane)
        scl = sq32.permute(1, 3, 2, 4, 0).contiguous().view(16, 32, 16 * 32 * 2)              # (b, s, w, lane, tile)
        ws[e, :, :, :32768] = codes.view(torch.uint8).view(16, 32, 32768)
        ws[e, :, :, 32768:] = scl.view(torch.uint8).view(16, 32, 4096)
    return ws, (gs13.float() * 256.0).contiguous()


def cr_prepack_w2(w2, w2_sf, gs2):
    """w2 uint8 [E, 4096, 2048] -> ws2 uint8 [E][8 b][32 s][36864]: warp w of block b owns output rows 512 b + 32 w .. + 31
    as two m16 tiles (K = the 4096 intermediate dims, same K permutation and scale words as W13)."""
    E = w2.shape[0]
    ws = torch.empty((E, 8, 32, CR_WB), dtype=torch.uint8, device=w2.device)
    for e in range(E):
        wq, sq = cr_prepack_rows(w2[e], w2_sf[e])   # [256, 64, 32, 4], [256, 32, 32, 4]
        codes = wq.view(8, 16, 2, 32, 2, 32, 4).permute(0, 3, 1, 2, 4, 5, 6).contiguous().view(8, 32, 16 * 4 * 32 * 4)
        sq32 = sq.contiguous().view(torch.int32).view(8, 16, 2, 32, 32)
        scl = sq32.permute(0, 3, 1, 4, 2).contiguous().view(8, 32, 16 * 32 * 2)
        ws[e, :, :, :32768] = codes.view(torch.uint8).view(8, 32, 32768)
        ws[e, :, :, 32768:] = scl.view(torch.uint8).view(8, 32, 4096)
    return ws, (gs2.float() * 256.0).contiguous()


def cr_plan_w2(d, P1, G, kappa=0.22, nb2=8):
    """W2 units from the W13 plan's (expert, pass) table (same token passes, X = K1's packed inter), equal-cost cuts,
    and the finalize's per-token route table."""
    dev = P1["units"].device
    u1 = P1["units"].cpu().tolist()
    seen, order = set(), []
    for e, b, nt, ng, rb, xo, multi, _ in u1:
        if (e, rb) not in seen:
            seen.add((e, rb))
            order.append((e, nt, ng, rb, xo, multi))
    units = []
    for e in dict.fromkeys(o[0] for o in order):
        ps = [o for o in order if o[0] == e]
        for b in range(nb2):
            for (_, nt, ng, rb, xo, multi) in ps:
                units.append([e, b, nt, ng, rb, xo, multi, 0])
    cost = [max(1.0, u[3] * kappa) for u in units]
    tot = 32.0 * sum(cost)
    cuts, cum, ui = [0], 0.0, 0
    for i in range(1, G):
        target = tot * i / G
        while ui < len(units) and cum + 32 * cost[ui] <= target:
            cum += 32 * cost[ui]
            ui += 1
        s_ = 0 if ui >= len(units) else min(32, int((target - cum) / cost[ui] + 0.5))
        cuts.append(max(cuts[-1], 32 * ui + s_))
    cuts.append(32 * len(units))
    rmap = {te: r for r, te in enumerate(P1["route_tok"])}
    ids = d["topk_ids"].long().cpu()
    wts = d["topk_weights"].float().cpu()
    M = ids.shape[0]
    tokr = torch.full((M, 8), -1, dtype=torch.int32)
    tokw = torch.zeros((M, 8), dtype=torch.float32)
    for tt in range(M):
        for k in range(8):
            e = int(ids[tt, k])
            if e < 32:
                tokr[tt, k] = rmap[(tt, e)]
                tokw[tt, k] = wts[tt, k]
    return {"units": torch.tensor(units, dtype=torch.int32, device=dev).reshape(-1, 8),
            "cuts": torch.tensor(cuts, dtype=torch.int32, device=dev), "tokr": tokr.to(dev), "tokw": tokw.to(dev)}


def cr_ref_full(d, w, dequant):
    """FP32 reference of the whole MoE (the problem's math; post-norm on bf16(y * (1 + w)))."""
    x = d["hidden_states"].float()
    ids = d["topk_ids"].long()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not 0 <= e < 32:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = (inter @ dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y.to(torch.bfloat16).float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * d["topk_weights"][tok, slot].float().unsqueeze(1))
    return out


class CrFull:
    """CR full call: K1 (W13 + SwiGLU -> packed inter) -> PDL -> K2 (W2 + post-norm partials) -> PDL -> K3 finalize.
    Plans / X pack are host-built (lab only)."""

    def __init__(self, st, ws2, gsc2, nw, pdl=True):
        self.st, self.ws2, self.gsc2, self.nw, self.pdl = st, ws2, gsc2, nw, pdl

    def prep(self, d, P1, P2):
        dev = self.ws2.device
        R = max(P1["R"], 1)
        bufs = {"xp2": torch.zeros_like(P1["xp"]), "ss": torch.zeros((R,), dtype=torch.float32, device=dev),
                "ybuf": torch.zeros((R, 4096), dtype=torch.bfloat16, device=dev),
                "out": torch.empty((d["hidden_states"].shape[0], 4096), dtype=torch.bfloat16, device=dev)}
        return bufs

    def fn(self, P1, P2, B):
        st = self.st

        def f():
            st.mod.cr_full(P1["xp"], st.ws, P1["units"], P1["cuts"], st.gsc, B["xp2"], self.ws2, P2["units"], P2["cuts"],
                           self.gsc2, self.nw, B["ss"], B["ybuf"], P2["tokr"], P2["tokw"], B["out"], st.scratch, st.cnt,
                           st.err, st.G, max(P1["R"], 1), int(self.pdl))
        return f


def cr_check_full(full, d, w, P1, P2, dequant):
    B = full.prep(d, P1, P2)
    full.st.err.zero_()
    full.fn(P1, P2, B)()
    torch.cuda.synchronize()
    ref = cr_ref_full(d, w, dequant)
    got = B["out"].float()
    diff = got - ref
    rel = float(diff.norm() / ref.norm().clamp(min=1e-30))
    rn = ref.norm(dim=1)
    worst = float((diff.norm(dim=1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
    return rel, worst, [int(v) for v in full.st.err.tolist()], int(full.st.cnt.abs().sum()), bool(torch.isnan(got).any()), B


def cr_case(bs, seed, dev):
    """The board's normal / ep8 / busiest routing law (our own copy; seeds disjoint from the judge's)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


# decode_layer routings (tools/box/m32_run.py, box_out/m32/run.json: 5 router seeds x 2 inputs): local expert token counts
CR_LAYER_COUNTS = {
    32: ([26, 11, 7, 6, 3, 3, 2, 1, 1, 1], [24, 16, 10, 6, 4, 3, 3], [22, 19, 11, 4, 3, 3, 3, 1, 1, 1], [23, 13, 11, 10]),
    16: ([14, 7, 5, 5, 3, 2, 1], [14, 12, 4, 2, 1, 1, 1, 1], [9, 4, 4, 4, 3, 1, 1, 1, 1, 1, 1]),
}


def cr_case_counts(bs, counts, seed, dev):
    """A routing with the given local-expert token counts (layer-shaped): random local experts, random distinct tokens per
    expert (<= 8 slots per token), the remaining slots filled with distinct remote experts (>= 32)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((bs, 4096), generator=torch.Generator(device=dev).manual_seed(seed), device=dev).to(torch.bfloat16)
    experts = torch.randperm(32, generator=g)[:len(counts)].tolist()
    slots = [[] for _ in range(bs)]
    for e, cn in zip(experts, counts):
        order = torch.randperm(bs, generator=g).tolist()
        order.sort(key=lambda tt: len(slots[tt]))
        free = [tt for tt in order if len(slots[tt]) < 8][:cn]
        assert len(free) == cn, "counts exceed 8 slots per token"
        for tt in free:
            slots[tt].append(e)
    ids = torch.empty((bs, 8), dtype=torch.int32)
    for tt in range(bs):
        rem = (32 + torch.randperm(224, generator=g)[:8 - len(slots[tt])]).tolist()
        row = slots[tt] + rem
        perm = torch.randperm(8, generator=g).tolist()
        ids[tt] = torch.tensor([row[i] for i in perm], dtype=torch.int32)
    wts = torch.rand((bs, 8), generator=g) + 0.05
    wts = wts / wts.sum(-1, keepdim=True)
    return {"hidden_states": x, "topk_ids": ids.to(dev).contiguous(), "topk_weights": wts.float().to(dev).contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def cr_plan(d, G, kappa=0.22, tpass=64, nb=16):
    """Host plan (lab only): units in hot-first expert order, (expert, block, pass); contiguous equal-cost stage ranges."""
    ids = d["topk_ids"].long().cpu()
    x = d["hidden_states"]
    dev = x.device
    M = ids.shape[0]
    toks = {e: [] for e in range(32)}
    for tt in range(M):
        for k in range(8):
            e = int(ids[tt, k])
            if e < 32:
                toks[e].append(tt)
    experts = sorted((e for e in range(32) if toks[e]), key=lambda e: (-len(toks[e]), e))
    passes, units, route_tok, xparts = {}, [], [], []
    rbase = xoff = 0
    for e in experts:
        T = toks[e]
        npass = math.ceil(len(T) / tpass)
        for p in range(npass):
            tl = T[tpass * p:tpass * (p + 1)]
            ng = math.ceil(len(tl) / 8)
            passes[(e, p)] = (len(tl), ng, rbase, xoff, npass > 1)
            xparts.append((tl, 8 * ng))
            route_tok.extend((tt, e) for tt in tl)
            rbase += len(tl)
            xoff += 32 * 8 * ng * CR_XRB
    for e in experts:
        npass = math.ceil(len(toks[e]) / tpass)
        for b in range(nb):
            for p in range(npass):
                nt, ng, rb, xo, multi = passes[(e, p)]
                units.append([e, b, nt, ng, rb, xo, int(multi), 0])
    cost = [max(1.0, u[3] * kappa) for u in units]
    tot = 32.0 * sum(cost)
    cuts, cum, ui = [0], 0.0, 0
    for i in range(1, G):
        target = tot * i / G
        while ui < len(units) and cum + 32 * cost[ui] <= target:
            cum += 32 * cost[ui]
            ui += 1
        s = 0 if ui >= len(units) else min(32, int((target - cum) / cost[ui] + 0.5))
        cuts.append(max(cuts[-1], 32 * ui + s))
    cuts.append(32 * len(units))
    # X pack: per (expert, pass) [32 stages][tpad][136] bf16 (128 K + 8 pad), zero-padded tokens
    chunks = []
    for tl, tpad in xparts:
        X = torch.zeros((tpad, 4096), dtype=torch.bfloat16, device=dev)
        X[:len(tl)] = x[torch.tensor(tl, dtype=torch.long, device=dev)]
        X = torch.nn.functional.pad(X.view(tpad, 32, 128).permute(1, 0, 2), (0, 8)).contiguous()
        chunks.append(X.view(torch.uint8).reshape(-1))
    xp = torch.cat(chunks) if chunks else torch.zeros((16,), dtype=torch.uint8, device=dev)
    w13_bytes = len(experts) * 16 * 32 * CR_WB
    return {"units": torch.tensor(units, dtype=torch.int32, device=dev).reshape(-1, 8),
            "cuts": torch.tensor(cuts, dtype=torch.int32, device=dev), "xp": xp, "R": rbase, "route_tok": route_tok,
            "experts": experts, "cnts": [len(toks[e]) for e in experts], "nunits": len(units), "bytes": w13_bytes,
            "cost": tot, "max_cta_cost": max((cuts[i + 1] - cuts[i]) for i in range(G))}


def cr_ref(d, w, P, dequant):
    """FP32 reference inter per planned route: silu(x Wg^T) * (x Wu^T)."""
    x = d["hidden_states"].float()
    R = P["R"]
    ref = torch.zeros((max(R, 1), 4096), dtype=torch.float32, device=x.device)
    by_e = {}
    for r, (tt, e) in enumerate(P["route_tok"]):
        by_e.setdefault(e, []).append((r, tt))
    for e, lst in by_e.items():
        rr = torch.tensor([a for a, _ in lst], dtype=torch.long, device=x.device)
        tk = torch.tensor([b for _, b in lst], dtype=torch.long, device=x.device)
        W = dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e])
        h = x[tk] @ W.T
        ref[rr] = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        del W
    return ref


class CrTimer:
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
                fn()
        torch.cuda.synchronize()
        return gr

    def time(self, gr):
        with torch.cuda.stream(self.stream):
            torch.sum(self.flush_buf, dim=0, out=self.sink)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    def overhead(self, n=60):
        tiny = torch.zeros(1, device="cuda")
        with torch.cuda.stream(self.stream):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                tiny.add_(0)
        torch.cuda.synchronize()
        return statistics.median(self.time(gr) for _ in range(n))


class CrState:
    """Device buffers + module for the CR-K1 lab."""

    def __init__(self, mod, ws, gsc, G, dev):
        self.mod, self.ws, self.gsc, self.G = mod, ws, gsc, G
        self.scratch = torch.zeros((G * 2 * CR_PART,), dtype=torch.float32, device=dev)
        self.cnt = torch.zeros((8192,), dtype=torch.int32, device=dev)
        self.err = torch.zeros((4,), dtype=torch.int32, device=dev)
        self.marks = torch.zeros((G * 8,), dtype=torch.int64, device=dev)
        self.nomark = torch.zeros((0,), dtype=torch.int64, device=dev)

    def fn(self, P, inter, mode, mk=False):
        def f():
            self.mod.cr_run(P["xp"], self.ws, P["units"], P["cuts"], self.gsc, inter, self.scratch, self.cnt, self.err,
                            self.marks if mk else self.nomark, self.G, mode)
        return f


def cr_check(st, d, w, P, dequant):
    """Run once (real consumer), compare inter with the FP32 reference; returns (rel, worst_row_rel, err, cnt_clean)."""
    inter = torch.full((max(P["R"], 1), 4096), float("nan"), dtype=torch.bfloat16, device=st.ws.device)
    st.err.zero_()
    st.fn(P, inter, 0)()
    torch.cuda.synchronize()
    ref = cr_ref(d, w, P, dequant)
    got = inter.float()
    diff = got - ref
    rel = float(diff.norm() / ref.norm().clamp(min=1e-30))
    rown = ref.norm(dim=1)
    floor = 0.1 * rown.mean()
    worst = float((diff.norm(dim=1) / torch.maximum(rown, floor)).max())
    nan = bool(torch.isnan(got).any())
    return rel, worst, [int(v) for v in st.err.tolist()], int(st.cnt.abs().sum()), nan, inter


def cr_rate_lab(st, G, timer, ov, log, tpass=64, nb=16, tag="R", ntoks=(8, 16, 32, 48, 64), rounds=12):
    """Consumer-rate constant (R1 rule): every unit is expert 0 / block 0 (1.15 MB, L2-resident after first touch), 384
    units x 32 stages = 453 MB of slot copies, ntok tokens on the expert. Prints S (null) and real us and GB/s per ng."""
    dev = st.ws.device
    out = {}
    for ntok in ntoks:
        if ntok > tpass:
            continue
        ids = torch.full((ntok, 8), 100, dtype=torch.int32, device=dev)
        ids[:, 0] = 0
        ids[:, 1:] = (100 + torch.arange(7, device=dev)).int()
        d = {"hidden_states": torch.randn((ntok, 4096), device=dev).to(torch.bfloat16), "topk_ids": ids,
             "topk_weights": torch.full((ntok, 8), 0.125, device=dev), "config": {}}
        P = cr_plan(d, G, tpass=tpass, nb=nb)
        nun = 384 * (16 // nb) if nb <= 16 else 384 * 2
        u = P["units"][:1].repeat(nun, 1).contiguous()
        u[:, 1] = 0
        P["units"] = u
        tot = 32 * nun
        P["cuts"] = torch.tensor([tot * i // G for i in range(G)] + [tot], dtype=torch.int32, device=dev)
        inter = torch.zeros((max(P["R"], 1), 4096), dtype=torch.bfloat16, device=dev)
        st.cnt.zero_()
        gS, gR = timer.capture(st.fn(P, inter, 1)), timer.capture(st.fn(P, inter, 0))
        tS, tR = [], []
        for _ in range(rounds):
            tS.append(timer.time(gS) - ov)
            tR.append(timer.time(gR) - ov)
        uS, uR = statistics.median(tS), statistics.median(tR)
        byt = tot * (CR_WB if nb == 16 else CR_WB // 2)
        out[ntok] = (uS, uR)
        log(f"[cr] rate {tag} ntok {ntok} (ng {u[0, 3].item()}): S {uS:.1f} us {byt / uS / 1e6:.2f} TB/s | real {uR:.1f} us "
            f"{byt / uR / 1e6:.2f} TB/s | real/S {uR / uS:.3f} | err {st.err.tolist()}")
        st.err.zero_()
        st.cnt.zero_()
        del gS, gR
    return out


def cr_lab(w, dequant=cr_dequant, k_full=None, k1_only=None, k_reset=None, log=print, rounds=20,
           cases=((256, 4), (128, 3), (32, 2), (16, 2), (8, 2), (1, 3), ("L32", 4), ("L16", 3)), seed0=770001,
           kappas=(0.22, 0.30)):
    """Arms per routing: K (shipped K2A full call), K1 (shipped K2A W13 kernel(s) alone), S (CR null consumers),
    R (CR real, kappa[0]), Rk (CR real, kappa[1]), V (2 CTAs/SM x 8 consumer warps, 256-row slots, <= 48 tokens/pass,
    kappa[0]). Correctness of R / Rk / V on every routing. Prints the pre-registered gate."""
    dev = w["w13"].device
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:                      # both builds at once (prepare-time budget on the judge)
        f16, f8 = ex.submit(cr_build), ex.submit(cr_build, ncw=8, ngmax=6, nst=3)
        mod, dt = f16.result()
        mod8, dt8 = f8.result()
    a = list(mod.cr_attrs())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    log(f"[cr] built in {dt:.0f}s; attrs (regs, local, smem, occ, ngmax, nst, ncw) {a}; {sms} SMs")
    a8 = list(mod8.cr_attrs())
    log(f"[cr] V built in {dt8:.0f}s; attrs {a8}")
    use_v = a8[3] >= 2
    if not use_v:
        log("[cr] V variant: occupancy < 2, arm dropped")
    if a[3] < 1 or a[0] != 96:
        log(f"[cr] ABORT: occupancy {a[3]} or launch registers {a[0]} != 96 (setmaxnreg 24 / 112 assumes a 96-reg CTA pool)")
        return []
    t0 = time.time()
    ws, gsc = cr_prepack(w["w13"][:32], w["w13_sf"][:32], w["w13_gs"][:32])
    torch.cuda.synchronize()
    log(f"[cr] prepack {time.time() - t0:.1f}s")
    ws2, gsc2 = cr_prepack_w2(w["w2"][:32], w["w2_sf"][:32], w["w2_gs"][:32])
    nw = (w["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous()
    G = sms
    st = CrState(mod, ws, gsc, G, dev)
    full = CrFull(st, ws2, gsc2, nw, pdl=True)
    st8 = CrState(mod8, ws, gsc, 2 * G, dev) if use_v else None
    timer = CrTimer()
    ov = timer.overhead()
    log(f"[cr] graph overhead {ov:.2f} us")
    try:
        cr_rate_lab(st, G, timer, ov, log)
        if use_v:
            cr_rate_lab(st8, 2 * G, timer, ov, log, tpass=48, nb=32, tag="V", ntoks=(8, 16, 32, 48))
    except Exception as ex:  # noqa: BLE001
        log(f"[cr] rate lab failed: {ex!r}")
    rows = []
    for bs, ncase in cases:
        for r in range(ncase):
            if isinstance(bs, str):
                d = cr_case_counts(int(bs[1:]), CR_LAYER_COUNTS[int(bs[1:])][r], seed0 + 500 + r, dev)
            else:
                d = cr_case(bs, seed0 + 1000 * bs + r, dev)
            Ps = [cr_plan(d, G, kappa=k) for k in kappas]
            P = Ps[0]
            P8 = cr_plan(d, 2 * G, kappa=kappas[0], tpass=48, nb=32) if use_v else None
            log(f"[cr] bs={bs} r={r}: {len(P['experts'])} experts, tokens {P['cnts'][:8]}{'...' if len(P['cnts']) > 8 else ''}; "
                f"{P['nunits']} units, W13 {P['bytes'] / 1e6:.0f} MB, stages/CTA max {P['max_cta_cost']}")
            rel, worst, e0, cl, nan, _ = cr_check(st, d, w, P, dequant)
            rel2, worst2, e2, cl2, nan2, _ = cr_check(st, d, w, Ps[1], dequant)
            ok = (rel <= 1e-2 and worst <= 5e-2 and e0[0] == 0 and cl == 0 and not nan and
                  rel2 <= 1e-2 and e2[0] == 0 and cl2 == 0 and not nan2)
            relv, ev = float("nan"), [0]
            if use_v:
                relv, worstv, ev, clv, nanv, _ = cr_check(st8, d, w, P8, dequant)
                ok = ok and relv <= 1e-2 and worstv <= 5e-2 and ev[0] == 0 and clv == 0 and not nanv
            log(f"[cr]   check: rel {rel:.2e} worst-row {worst:.2e} err {e0} cnt-clean {cl == 0} nan {nan} | "
                f"k2: rel {rel2:.2e} err {e2} | V: rel {relv:.2e} err {ev} | {'OK' if ok else 'FAIL'}")
            if not ok:
                rows.append({"bs": bs, "bad": True})
                st.err.zero_(); st.cnt.zero_()
                continue
            P2 = cr_plan_w2(d, P, G, kappa=kappas[0])
            relF, worstF, eF, clF, nanF, BF = cr_check_full(full, d, w, P, P2, dequant)
            okF = relF <= 1.2e-2 and worstF <= 6e-2 and eF[0] == 0 and clF == 0 and not nanF
            relK = float("nan")
            if k_full is not None:
                try:
                    ko = k_full(d)
                    torch.cuda.synchronize()
                    relK = float((BF["out"].float() - ko.float()).norm() / ko.float().norm().clamp(min=1e-30))
                except Exception as ex:  # noqa: BLE001
                    log(f"[cr]   K2A output for rel failed: {ex!r}")
            log(f"[cr]   full check: rel {relF:.2e} worst-row {worstF:.2e} rel-vs-K2A {relK:.2e} err {eF} "
                f"cnt-clean {clF == 0} nan {nanF} -> {'OK' if okF else 'FAIL'}")
            st.err.zero_(); st.cnt.zero_()
            inter = torch.zeros((max(P["R"], 1), 4096), dtype=torch.bfloat16, device=dev)
            arms = {"S": st.fn(P, inter, 1), "R": st.fn(P, inter, 0), "Rk": st.fn(Ps[1], inter, 0)}
            if okF:
                arms["F"] = full.fn(P, P2, BF)
            if use_v:
                arms["V"] = st8.fn(P8, inter, 0)
                arms["SV"] = st8.fn(P8, inter, 1)
            if k_full is not None:
                arms["K"] = (lambda d=d: k_full(d))
            graphs = {nm: timer.capture(fn) for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            del graphs
            if k1_only is not None:
                try:
                    g1 = timer.capture(lambda d=d: k1_only(d))
                    us["K1"] = statistics.median(max(timer.time(g1) - ov, 1e-3) for _ in range(max(8, rounds // 2)))
                    del g1
                except Exception as ex:  # noqa: BLE001
                    log(f"[cr]   K1 arm failed: {ex!r}")
                    us["K1"] = float("nan")
                if k_reset is not None:
                    k_reset()
            # one marked run of R: per-CTA consumer ends
            st.marks.zero_()
            with torch.cuda.stream(timer.stream):
                torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                st.fn(P, inter, 0, mk=True)()
            torch.cuda.synchronize()
            mv = st.marks.view(G, 8).tolist()
            t00 = min(m[0] for m in mv)
            ends = sorted((m[3] - t00) / 1e3 for m in mv if m[3] > 0)
            first = sorted((m[2] - t00) / 1e3 for m in mv if m[2] > 0)
            e1 = [int(v) for v in st.err.tolist()]
            row = {"bs": bs, "us": us, "B": P["bytes"], "cnts": P["cnts"], "err": e1, "rel": rel, "worst": worst, "relF": relF,
                   "end": (statistics.median(ends), ends[-1]) if ends else (0, 0),
                   "first": (statistics.median(first), first[-1]) if first else (0, 0)}
            rows.append(row)
            log(f"[cr]   us: " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items()) +
                f" | W13 rate S {P['bytes'] / us['S'] / 1e6:.2f} R {P['bytes'] / us['R'] / 1e6:.2f}"
                f"{(' V %.2f SV %.2f' % (P['bytes'] / us['V'] / 1e6, P['bytes'] / us['SV'] / 1e6)) if 'V' in us else ''} TB/s | "
                f"R/K1 {us['R'] / us.get('K1', float('nan')):.3f} | marks first p50/max {row['first'][0]:.1f}/"
                f"{row['first'][1]:.1f} end p50/max {row['end'][0]:.1f}/{row['end'][1]:.1f} us | err {e1}")
            st.err.zero_()
            del inter
    log("[cr] GATE (pre-registered R-cr-1 = R-cr-0's rule, medians): GO if (min(R,Rk)+2)/K1 <= 0.80 at bs256 AND <= 0.88 at bs128 AND "
        "R stream rate at bs8/bs32 >= 3.6 TB/s AND layer L16 / L32 <= 0.95; KILL if bs256 > 0.90 or stream rate < 3.4; "
        "else REVISE")
    res = {}
    for bs in sorted({x["bs"] for x in rows}, key=str):
        rr = [x for x in rows if x["bs"] == bs and not x.get("bad")]
        if not rr or any(x.get("bad") for x in rows if x["bs"] == bs):
            log(f"[cr]   bs={bs}: INVALID ({len(rr)} valid)")
            res[bs] = None
            continue
        best = [min(x["us"]["R"], x["us"]["Rk"], x["us"].get("V", 9e9)) for x in rr]
        g = statistics.median((b_ + 2.0) / x["us"].get("K1", float("nan")) for b_, x in zip(best, rr))
        rate = statistics.median(x["B"] / min(x["us"]["R"], x["us"].get("V", 9e9)) / 1e6 for x in rr)
        res[bs] = (g, rate)
        log(f"[cr]   bs={bs}: (best R + 2)/K1 {g:.3f} | R {statistics.median(x['us']['R'] for x in rr):.1f} "
            f"Rk {statistics.median(x['us']['Rk'] for x in rr):.1f} V {statistics.median(x['us'].get('V', float('nan')) for x in rr):.1f} "
            f"SV {statistics.median(x['us'].get('SV', float('nan')) for x in rr):.1f} S {statistics.median(x['us']['S'] for x in rr):.1f} "
            f"K1 {statistics.median(x['us'].get('K1', float('nan')) for x in rr):.1f} us | R W13 rate {rate:.2f} TB/s")
    # full call (diagnostic, pre-registered in R-cr-1): (F + 3) / K per board bs, projected MoE board vs K2A's 1.93
    lnsum, nsh = 0.0, 0
    for bs in (1, 8, 16, 32, 128, 256, "L16", "L32"):
        rr = [x for x in rows if x["bs"] == bs and not x.get("bad") and "F" in x["us"] and "K" in x["us"]]
        if not rr:
            continue
        fk = statistics.median((x["us"]["F"] + 3.0) / x["us"]["K"] for x in rr)
        log(f"[cr]   full bs={bs}: F {statistics.median(x['us']['F'] for x in rr):.1f} K {statistics.median(x['us']['K'] for x in rr):.1f} us "
            f"| (F + 3) / K {fk:.3f} | rel {max(x['relF'] for x in rr):.2e}")
        if not isinstance(bs, str):
            lnsum += math.log(fk)
            nsh += 1
    if nsh == 6:
        log(f"[cr] FULL-CALL projection: MoE board 1.93 x {math.exp(-lnsum / 6):.3f} = {1.93 * math.exp(-lnsum / 6):.3f} "
            f"(host plan + X pack outside timing; +3 us allowance included)")
    ok = all(v is not None for v in res.values()) and 256 in res and 128 in res
    if not ok:
        dec = "INVALID (fix, rerun under the same rule)"
    else:
        srate = min(res[b][1] for b in (8, 32) if res.get(b))
        lay = max((res[b][0] for b in ("L32", "L16") if res.get(b)), default=9.0)
        if res[256][0] > 0.90 or srate < 3.4:
            dec = "KILL (CR uniform-consumer structure; add trap)"
        elif res[256][0] <= 0.80 and res[128][0] <= 0.88 and srate >= 3.6 and lay <= 0.95:
            dec = "GO (build CR K2 + finalize; board check file)"
        else:
            dec = f"REVISE (one revision guided by S / marks, one rerun; layer worst {lay:.3f})"
    log(f"[cr] DECISION {dec}")
    del timer
    torch.cuda.empty_cache()
    return rows

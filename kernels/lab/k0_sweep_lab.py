"""K0: the SWEEP kill test (notes/study/moe_redesign_independent.md s5), filler arms at bs 1 / 8 (G-small).

One launch per call. Every virtual CTA v (V = grid x GROUPS) gets a balanced lockstep share of every touched expert:
16-row full-K granules, W13 shares then W2 shares, extras rotated so cumulative shares differ by <= 1. The CTA's
sequence is cut into 8-granule items plus 4 / 2 / 1-granule items that are K-split inside the CTA. Each ring slot always
holds 8 tile-stages: position p = granule (p mod g) at K-stage j*(8/g) + p div g, so every consumer warp is busy.
Consumer = K2A v3.3's NC1 inner loop (consume_kb<1>, two mma chains) and its flush math. Weights are K2A's stage-repacked
tensors (ws13 / ws2). X / inter come from a per-slot row pool. W2 waits per expert on cnt13[e] == 512 (lockstep: should
never block). Finalize in the same launch: done counter, then each CTA computes its column slice. Self-resetting
counters, bounded waits (error flag, no trap).

mode bit 0: null consumers (no math, no stores; counters kept), bit 1: no X copies.
"""
import hashlib
import statistics
import time

import torch

K0_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace k0 {
constexpr int KDIM = 4096;
constexpr int M3 = 320;
constexpr int RS = 8 * M3;            // route stride of inter / y (K2A's layout)
constexpr int SW = 18432;
constexpr int XROW = 528;
constexpr int XST = XROW / 2;
constexpr int MMAX = 32;
constexpr int RMAX = 8 * MMAX;
constexpr unsigned FULL = 0xffffffffu;
constexpr long long WAIT_BUDGET = 1ll << 28;   // ~0.15 s per wait; after the first timeout every wait bails out

struct Args {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint8_t* ws13; const uint8_t* ws2;
  const float* gsc13; const float* gsc2; const float* nw;
  __nv_bfloat16* inter; float* y; float* ss; int* cnt; int* err; __nv_bfloat16* out; long long* marks;
  int M, lo, mode;
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout (K2A v3)
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
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  int n = 0;
  while (!mbar_try(b, par)) {
    if ((++n & 255) == 0) {
      if (*reinterpret_cast<volatile int*>(err) != 0) return;
      if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }
    }
  }
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
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__device__ __forceinline__ void spin_ge(const int* p, int target, int* err) {
  if (ld_acquire(p) >= target) return;
  const long long t0 = clock64();
  while (ld_acquire(p) < target) {
    __nanosleep(64);
    if (*reinterpret_cast<volatile int*>(err) != 0) return;
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 3, 1); return; }
  }
}
// One kb (16 B of codes) of this warp's tile, B from the staged X rows (K2A v3 consume_kb<1>).
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

// Balanced lockstep share of phase p (p < ne: W13 of order index p, else W2 of p - ne) for virtual CTA v of V.
__device__ __forceinline__ void phase_share(int p, int ne, int V, int v, int& o, int& n) {
  const int m = p >= ne ? 1 : 0;
  const int P = m ? 256 : 512;
  const int q = P / V, r = P % V;
  const long long R = (long long)min(p, ne) * (512 % V) + (long long)max(0, p - ne) * (256 % V);
  const int w = (int)(R % V);
  const int dv = ((v - w) % V + V) % V;
  n = q + (dv < r ? 1 : 0);
  int c;
  if (w + r <= V) c = min(max(v - w, 0), r);
  else c = min(v, w + r - V) + max(0, v - w);
  o = q * v + c;
}
// item idx of a sequence of N granules: full 8-granule items, then 4 / 2 / 1 by the bits of N mod 8
__device__ __forceinline__ int item_g(int N, int idx) {
  const int full = N >> 3, rem = N & 7;
  if (idx < full) return 8;
  idx -= full;
  if (rem & 4) { if (idx == 0) return 4; --idx; }
  if (rem & 2) { if (idx == 0) return 2; --idx; }
  if (rem & 1) { if (idx == 0) return 1; }
  return 0;
}

struct Sh {
  int cnt[32], off[33], el[32], run[32];
  int ne, R;
  short list[RMAX];
  short tr[MMAX][8];
  int tn[MMAX];
  int po[2][2][32], pn[2][2][32], pc[2][2][33];   // [group][matrix][phase]
  unsigned long long cend;
};

__device__ __forceinline__ void lookup(const Sh& S, int gq, int m, int i, int& k, int& gi) {
  k = 0;
  while (k + 1 < S.ne && S.pc[gq][m][k + 1] <= i) ++k;
  gi = S.po[gq][m][k] + (i - S.pc[gq][m][k]);
}

template <int NST, int XP, int GROUPS>
__global__ void __launch_bounds__(32 * (9 * GROUPS + 1), (GROUPS == 1) ? 2 : 1) k0_kernel(const Args A) {
  constexpr int SLOT = SW + XP * XROW;
  constexpr int RING = NST * SLOT;
  constexpr int GS = RING + 16 * NST + 32 * NST + 4096;   // ring | full,empty bars | hdr | red (one buffer)
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Sh S;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int G = gridDim.x, V = G * GROUPS;
  if (tid == 0 && A.marks) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
    A.marks[(size_t)blockIdx.x * 8 + 0] = gtime();
    A.marks[(size_t)blockIdx.x * 8 + 5] = smid;
  }
  // ---------------- prologue: routing (every CTA, deterministic) ----------------
  if (tid < 32) { S.cnt[tid] = 0; S.run[tid] = 0; }
  if (tid < MMAX) S.tn[tid] = 0;
  if (tid == 0) S.cend = 0ull;
  if (tid < GROUPS) {
    uint8_t* gb = smem + tid * GS;
    uint64_t* bars = reinterpret_cast<uint64_t*>(gb + RING);
    for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + i), 1); mbar_init(sa(bars + NST + i), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int NP = A.M * 8;
  for (int i = tid; i < NP; i += blockDim.x) {
    const int e = (A.ids[i] - A.lo) & 255;
    if (e < 32) atomicAdd(&S.cnt[e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    const int c = S.cnt[lane];
    int incl = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int u = __shfl_up_sync(FULL, incl, d); if (lane >= d) incl += u; }
    S.off[lane] = incl - c;
    if (lane == 31) { S.off[32] = incl; S.R = incl; }
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
      const int cj = __shfl_sync(FULL, c, j);
      rank += (cj > c || (cj == c && j < lane)) ? 1 : 0;
    }
    if (c > 0) S.el[rank] = lane;   // order: descending count, ties by id
    const int ne = __popc(__ballot_sync(FULL, c > 0));
    if (lane == 0) S.ne = ne;
    __syncwarp();
    for (int base = 0; base < NP; base += 32) {
      const int i = base + lane;
      int e = -1;
      if (i < NP) { const int ee = (A.ids[i] - A.lo) & 255; if (ee < 32) e = ee; }
      const bool loc = e >= 0;
      const unsigned mm = __match_any_sync(FULL, e);
      const int rk = __popc(mm & ((1u << lane) - 1u));
      const unsigned lb = __ballot_sync(FULL, loc);
      const unsigned tg = 0xFFu << (lane & ~7);
      const int run = loc ? S.run[e] : 0;
      __syncwarp();
      if (loc) {
        const int r = S.off[e] + run + rk;
        S.list[r] = (short)i;
        const int t = i >> 3;
        S.tr[t][__popc(lb & tg & ((1u << lane) - 1u))] = (short)r;
        if (rk == __popc(mm) - 1) S.run[e] = run + __popc(mm);
      }
      if ((lane & 7) == 0 && i < NP) S.tn[i >> 3] = __popc(lb & tg);
      __syncwarp();
    }
  }
  __syncthreads();
  const int ne = S.ne;
  if (tid < GROUPS * 64) {
    const int gq = tid >> 6, idx = tid & 63, m = idx >> 5, k = idx & 31;
    if (k < ne) {
      int o, n;
      phase_share(m * ne + k, ne, V, blockIdx.x * GROUPS + gq, o, n);
      S.po[gq][m][k] = o;
      S.pn[gq][m][k] = n;
    }
  }
  __syncthreads();
  if (tid < GROUPS * 2) {
    const int gq = tid >> 1, m = tid & 1;
    int c = 0;
    for (int k = 0; k < ne; ++k) { S.pc[gq][m][k] = c; c += S.pn[gq][m][k]; }
    S.pc[gq][m][ne] = c;
  }
  __syncthreads();
  const bool nullc = (A.mode & 1) != 0, nox = (A.mode & 2) != 0;

  if (warp < 9 * GROUPS) {
    const int gq = warp / 9, wi = warp % 9;
    uint8_t* gb = smem + gq * GS;
    uint8_t* ring = gb;
    uint64_t* bars = reinterpret_cast<uint64_t*>(gb + RING);
    int* hdr = reinterpret_cast<int*>(gb + RING + 16 * NST);
    float4* red = reinterpret_cast<float4*>(gb + RING + 16 * NST + 32 * NST);
    const uint32_t ring_a = sa(ring), full_a = sa(bars), empty_a = sa(bars + NST);
    if (wi == 8) {
      // ---------------- producer ----------------
      const uint64_t pol = policy_evict_first();
      const int p = lane & 7;
      const bool vp = lane < 8;
      int it = 0, ready = -1;
      for (int m = 0; m < 2; ++m) {
        const int N = S.pc[gq][m][ne];
        const uint8_t* wsrc = m ? A.ws2 : A.ws13;
        const int NG = m ? 256 : 512;
        int start = 0;
        for (int ii = 0;; ++ii) {
          const int g = item_g(N, ii);
          if (g == 0) break;
          const int kpg = 8 / g;
          for (int j = 0; j < 2 * g; ++j, ++it) {
            const int slot = it % NST;
            const uint32_t par = ((it / NST) & 1) ^ 1;
            const int s = j * kpg + p / g;
            int k, gi;
            lookup(S, gq, m, start + (p % g), k, gi);
            const int e = S.el[k], T = S.cnt[e], c0 = S.off[e];
            const int pk = __shfl_up_sync(FULL, k, 1), ps = __shfl_up_sync(FULL, s, 1), pg = __shfl_up_sync(FULL, gi, 1);
            const bool whead = vp && (p == 0 || pk != k || ps != s || pg + 1 != gi);   // granule-major: [e][s][gi]
            const bool xhead = vp && (p == 0 || pk != k || ps != s);
            const unsigned wm = __ballot_sync(FULL, whead) & 0xFFu, xm = __ballot_sync(FULL, xhead) & 0xFFu;
            int L = 0;
            if (whead) { const unsigned rest = wm & ~((2u << p) - 1u); L = (rest ? __ffs(rest) - 1 : 8) - p; }
            const int xr = xhead ? T : 0;
            int xs = xr;
#pragma unroll
            for (int d = 1; d < 8; d <<= 1) { const int u = __shfl_up_sync(FULL, xs, d); if (lane >= d) xs += u; }
            const int xoh = xs - xr;   // run head's row offset in the pool
            const int rows = xhead ? max(0, min(T, XP - xoh)) : 0;
            if (xhead && rows < T) atomicExch(A.err + 1, 1);   // X pool overflow (rows dropped; timing only)
            const int hp = 31 - __clz(xm & ((2u << p) - 1u));
            const int xo = __shfl_sync(FULL, xoh, hp);
            const int rb = m ? 528 : 512;
            int xb = nox ? 0 : rows * rb;
#pragma unroll
            for (int d = 16; d >= 1; d >>= 1) xb += __shfl_xor_sync(FULL, xb, d);
            if (m == 1) {   // W2: every expert in this slot must have all 512 W13 granules done (lockstep: ~never waits)
              int mk = vp ? k : -1;
              mk = __reduce_max_sync(FULL, mk);
              if (mk > ready) {
                if (lane == 0) {
                  for (int kk = ready + 1; kk <= mk; ++kk) spin_ge(A.cnt + S.el[kk], 512, A.err);
                  asm volatile("fence.proxy.async.global;\n" ::: "memory");
                }
                ready = mk;
                __syncwarp();
              }
            }
            mbar_wait(empty_a + 8 * slot, par, A.err);
            if (vp) hdr[slot * 8 + p] = min(xo, XP - 1);
            __syncwarp();
            const uint32_t dst = ring_a + slot * SLOT, fb = full_a + 8 * slot;
            if (lane == 0) {
              mbar_expect_tx(fb, SW + xb);
              if (it == 0 && gq == 0 && A.marks) A.marks[(size_t)blockIdx.x * 8 + 1] = gtime();
            }
            __syncwarp();
            if (whead)   // L consecutive granules of one (expert, K-stage): one contiguous copy of L x 2304 B
              bulk_g2s_ef(dst + p * 2304, wsrc + (((size_t)e * 16 + s) * NG + gi) * 2304, L * 2304, fb, pol);
            if (!nox) {
              if (m == 0) {
                if (xhead)
                  for (int jr = 0; jr < rows; ++jr)
                    bulk_g2s(dst + SW + (xoh + jr) * XROW,
                             A.x + (size_t)(S.list[c0 + jr] >> 3) * KDIM + s * 256, 512, fb);
              } else {   // inter written by other CTAs: lane 0 (which took the acquire + proxy fence) copies every run
                for (unsigned mmk = xm; mmk; mmk &= mmk - 1) {
                  const int h = __ffs(mmk) - 1;
                  const int hs = __shfl_sync(FULL, s, h), hc = __shfl_sync(FULL, c0, h);
                  const int hx = __shfl_sync(FULL, xoh, h), hr = __shfl_sync(FULL, rows, h);
                  if (lane == 0 && hr > 0)
                    bulk_g2s(dst + SW + hx * XROW, A.inter + ((size_t)hs * RS + hc) * XST, hr * 528, fb);
                }
              }
            }
            __syncwarp();
          }
          start += g;
        }
      }
    } else {
      // ---------------- consumers: warp = position p ----------------
      const int p = wi, g8 = lane >> 2, t = lane & 3;
      int it = 0;
      for (int m = 0; m < 2; ++m) {
        const int N = S.pc[gq][m][ne];
        int start = 0;
        for (int ii = 0;; ++ii) {
          const int g = item_g(N, ii);
          if (g == 0) break;
          float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
          for (int j = 0; j < 2 * g; ++j, ++it) {
            const int slot = it % NST;
            mbar_wait(full_a + 8 * slot, (it / NST) & 1, A.err);
            if (!nullc) {
              const uint8_t* st = ring + slot * SLOT;
              const int xo = hdr[slot * 8 + p];
              const uint4* wp = reinterpret_cast<const uint4*>(st + p * 2304) + lane;
              const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + p * 2304 + 2048) + lane;
              const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW + xo * XROW) + g8 * XST;
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
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[i] += acc2[i];
          if (g < 8) {   // K-split item: the 8/g warps of a granule meet in smem
            red[p * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
            asm volatile("bar.sync %0, 256;\n" :: "r"(1 + gq) : "memory");
            if (p < g) {
              for (int kk = 1; kk < 8 / g; ++kk) {
                const float4 v = red[(p + kk * g) * 32 + lane];
                acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
              }
            }
            asm volatile("bar.sync %0, 256;\n" :: "r"(1 + gq) : "memory");
          }
          if (p < g) {
            int k, gi;
            lookup(S, gq, m, start + p, k, gi);
            const int e = S.el[k], T = S.cnt[e], c0 = S.off[e];
            if (!nullc) {
              if (m == 0) {   // SwiGLU -> inter (K2A flush_item<1, 1>, whole item)
                const float gs = __ldg(A.gsc13 + e);
                const int col = 8 * gi + g8;
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                  const int n = 2 * t + hh;
                  if (n < T)
                    A.inter[((size_t)(col >> 8) * RS + c0 + n) * XST + (col & 255)] =
                        __float2bfloat16_rn(swiglu(acc[hh] * gs, acc[2 + hh] * gs));
                }
              } else {        // y + sum of squares of bf16(y) * (1 + w) (K2A flush_item<1, 2>, whole item)
                const float gs = __ldg(A.gsc2 + e);
                const float w0 = __ldg(A.nw + 16 * gi + g8), w1 = __ldg(A.nw + 16 * gi + g8 + 8);
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                  const int n = 2 * t + hh;
                  const float v0 = acc[hh] * gs, v1 = acc[2 + hh] * gs;
                  if (n < T) {
                    float* yp = A.y + (size_t)(c0 + n) * KDIM + 16 * gi + g8;
                    yp[0] = v0; yp[8] = v1;
                  }
                  const float z0 = bf16r(v0) * w0, z1 = bf16r(v1) * w1;
                  float pz = z0 * z0 + z1 * z1;
                  pz += __shfl_xor_sync(FULL, pz, 4);
                  pz += __shfl_xor_sync(FULL, pz, 8);
                  pz += __shfl_xor_sync(FULL, pz, 16);
                  if (g8 == 0 && n < T) atomicAdd(A.ss + c0 + n, pz);
                }
              }
            }
            if (m == 0) {
              __threadfence();
              __syncwarp();
              if (lane == 0) atomicAdd(A.cnt + e, 1);
            }
          }
          start += g;
        }
      }
      if (lane == 0) atomicMax(&S.cend, (unsigned long long)gtime());
    }
  }
  __syncthreads();
  // ---------------- finalize (same launch) ----------------
  if (tid == 0) {
    if (A.marks) A.marks[(size_t)blockIdx.x * 8 + 2] = (long long)S.cend;
    __threadfence();
    atomicAdd(A.cnt + 32, 1);
    spin_ge(A.cnt + 32, G, A.err);
    if (A.marks) A.marks[(size_t)blockIdx.x * 8 + 3] = gtime();
  }
  __syncthreads();
  {
    const int c0 = (int)((long long)KDIM * blockIdx.x / G), c1 = (int)((long long)KDIM * (blockIdx.x + 1) / G);
    const int nc = c1 - c0;
    for (int q = tid; q < A.M * nc; q += blockDim.x) {
      const int tk = q / nc, col = c0 + q % nc;
      float o = 0.f;
      const int nr = S.tn[tk];
      for (int jj = 0; jj < nr; ++jj) {
        const int r = S.tr[tk][jj];
        const float sc = rsqrtf(__ldcg(A.ss + r) * (1.f / 4096.f) + 1e-8f);
        o += __ldg(A.wts + S.list[r]) * bf16r(__ldcg(A.y + (size_t)r * KDIM + col)) * __ldg(A.nw + col) * sc;
      }
      A.out[(size_t)tk * KDIM + col] = __float2bfloat16_rn(o);
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const int old = atomicAdd(A.cnt + 33, 1);
    if (old == G - 1) {   // last CTA out: reset counters and sums of squares for the next call
      for (int i = 0; i < 33; ++i) A.cnt[i] = 0;
      for (int i = 0; i < S.R; ++i) A.ss[i] = 0.f;
      __threadfence();
      A.cnt[33] = 0;
    }
    if (A.marks) A.marks[(size_t)blockIdx.x * 8 + 4] = gtime();
  }
}

template <int NST, int XP, int GROUPS>
int smem_bytes() { return GROUPS * (NST * (SW + XP * XROW) + 16 * NST + 32 * NST + 4096); }

template <int NST, int XP, int GROUPS>
void launch(const Args& a, int G) {
  const int sm = smem_bytes<NST, XP, GROUPS>();
  cudaFuncSetAttribute(k0_kernel<NST, XP, GROUPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaFuncSetAttribute(k0_kernel<NST, XP, GROUPS>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  k0_kernel<NST, XP, GROUPS><<<G, 32 * (9 * GROUPS + 1), sm, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int NST, int XP, int GROUPS>
std::vector<int64_t> attrs() {
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, k0_kernel<NST, XP, GROUPS>);
  const int sm = smem_bytes<NST, XP, GROUPS>();
  cudaFuncSetAttribute(k0_kernel<NST, XP, GROUPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaFuncSetAttribute(k0_kernel<NST, XP, GROUPS>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k0_kernel<NST, XP, GROUPS>, 32 * (9 * GROUPS + 1), sm);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + sm), occ};
}
}  // namespace k0

void k0_run(int64_t cfg, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y,
            torch::Tensor ss, torch::Tensor cnt, torch::Tensor err, torch::Tensor out, torch::Tensor marks,
            int64_t M, int64_t lo, int64_t G, int64_t mode) {
  k0::Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ids = ids.data_ptr<int>(); a.wts = wts.data_ptr<float>();
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr()); a.ws2 = reinterpret_cast<const uint8_t*>(ws2.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>(); a.gsc2 = gsc2.data_ptr<float>(); a.nw = nw.data_ptr<float>();
  a.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); a.y = y.data_ptr<float>(); a.ss = ss.data_ptr<float>();
  a.cnt = cnt.data_ptr<int>(); a.err = err.data_ptr<int>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  a.M = (int)M; a.lo = (int)lo; a.mode = (int)mode;
  switch (cfg) {
    case 0: k0::launch<4, 8, 1>(a, (int)G); break;
    case 1: k0::launch<4, 8, 2>(a, (int)G); break;
    case 2: k0::launch<2, 64, 1>(a, (int)G); break;
    case 3: k0::launch<2, 64, 2>(a, (int)G); break;
    default: TORCH_CHECK(false, "k0: bad cfg");
  }
}
std::vector<int64_t> k0_attrs(int64_t cfg) {
  switch (cfg) {
    case 0: return k0::attrs<4, 8, 1>();
    case 1: return k0::attrs<4, 8, 2>();
    case 2: return k0::attrs<2, 64, 1>();
    case 3: return k0::attrs<2, 64, 2>();
    default: return {-1};
  }
}
"""

K0_CPP = r"""
#include <torch/extension.h>
#include <vector>
void k0_run(int64_t cfg, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y,
            torch::Tensor ss, torch::Tensor cnt, torch::Tensor err, torch::Tensor out, torch::Tensor marks,
            int64_t M, int64_t lo, int64_t G, int64_t mode);
std::vector<int64_t> k0_attrs(int64_t cfg);
"""

K0_EXPERT_BYTES = 768 * 36864    # 28.31 MB per touched expert


def k0_build(verbose=False):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((K0_CPP + K0_CUDA).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"k0_sweep_{tag}", cpp_sources=K0_CPP, cuda_sources=K0_CUDA, functions=["k0_run", "k0_attrs"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"], verbose=verbose)
    return mod, time.time() - t0


def k0_repack(ws):
    """K2A stage layout [E][NRB][16 K-stages][18432 B] -> granule-major [E][16][NRB*8][2304 B] (lossless permutation):
    granule gi = rb*8 + w, 2048 B codes + 256 B scales, so a slot's consecutive granules of one (expert, K-stage) are one copy."""
    e, nrb = ws.shape[0], ws.shape[1]
    codes = ws[..., :16384].reshape(e, nrb, 16, 8, 2048)
    sc = ws[..., 16384:].reshape(e, nrb, 16, 8, 256)
    return torch.cat((codes, sc), dim=-1).permute(0, 2, 1, 3, 4).reshape(e, 16, nrb * 8, 2304).contiguous()


def k0_case(bs, seed, dev):
    """generate_input's normal / ep8 / busiest routing math (our own copy)."""
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


class K0Timer:
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


def k0_lab(st, k2a_fn, k2a_state, log=print, rounds=30, inputs=3, seeds0=777001):
    """G-small: arms K (shipped K2A), F2 (V=264, 2 CTAs/SM), F1 (V=264, 132 CTAs x 2 groups), F3 (V=132, 1 group/SM),
    S (F2 schedule, null consumers + X copies), S0 (null consumers, no X). Prints per routing and the gate."""
    dev = st["ws13"].device
    mod, dt = k0_build()
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    occ = {}
    for cfg in range(4):
        a = list(mod.k0_attrs(cfg))
        occ[cfg] = a[3]
        log(f"[k0] cfg {cfg} attrs (regs, local, smem, occ) {a}")
    log(f"[k0] built in {dt:.0f}s, {sms} SMs")
    ws13g, ws2g = k0_repack(st["ws13"]), k0_repack(st["ws2"])
    torch.cuda.synchronize()
    inter = torch.zeros((16 * 8 * 320 * 264,), dtype=torch.bfloat16, device=dev)
    yb = torch.zeros((8 * 320, 4096), dtype=torch.float32, device=dev)
    ss = torch.zeros((8 * 320,), dtype=torch.float32, device=dev)
    cnt = torch.zeros((64,), dtype=torch.int32, device=dev)
    err = torch.zeros((8,), dtype=torch.int32, device=dev)
    marks = torch.zeros((2 * sms * 8,), dtype=torch.int64, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    timer = K0Timer()
    ov = timer.overhead()
    log(f"[k0] graph overhead {ov:.2f} us")
    rows = []
    for bs in (1, 8):
        for r in range(inputs):
            d = k0_case(bs, seeds0 + 1000 * bs + r, dev)
            ids = d["topk_ids"]
            cnts = torch.bincount((ids.long() & 255)[(ids.long() & 255) < 32].flatten(), minlength=32)
            ne = int((cnts > 0).sum())
            B = ne * K0_EXPERT_BYTES
            out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
            base_cfg = 0 if bs == 1 else 2

            def k0f(cfg, G, mode, mk=False, d=d, out=out):
                def f():
                    mod.k0_run(cfg, d["hidden_states"], d["topk_ids"], d["topk_weights"], ws13g, ws2g,
                               st["gsc13"], st["gsc2"], st["nw"], inter, yb, ss, cnt, err, out, marks if mk else nomark,
                               bs, 0, G, mode)
                    return out
                return f
            spec = {"F2": (base_cfg, 2 * sms, 0), "F1": (base_cfg + 1, sms, 0), "F3": (base_cfg, sms, 0),
                    "S": (base_cfg, 2 * sms, 1), "S0": (base_cfg, 2 * sms, 3)}
            # every CTA must be resident (W2 and done spins wait on all of them): drop arms that cannot be
            spec = {nm: v for nm, v in spec.items() if occ[v[0]] * sms >= v[1]}
            arms = {"K": lambda d=d: k2a_fn(d, k2a_state)}
            arms.update({nm: k0f(*v) for nm, v in spec.items()})
            err.zero_()
            ref = arms["K"]().float().clone()
            torch.cuda.synchronize()
            rel = {}
            for nm in ("F2", "F1", "F3"):
                if nm not in arms:
                    rel[nm] = float("nan")
                    continue
                o = arms[nm]().float()
                torch.cuda.synchronize()
                rel[nm] = float((o - ref).norm() / ref.norm().clamp(min=1e-30))
            graphs = {nm: timer.capture(fn)[0] for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            # one marked eager run per K0 arm: consumer-end spread over CTAs (us)
            spread = {}
            for nm, (cfg, G, mode) in spec.items():
                if nm == "S":
                    continue
                marks.zero_()
                with torch.cuda.stream(timer.stream):
                    torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                    k0f(cfg, G, mode, mk=True)()
                torch.cuda.synchronize()
                mv = marks.view(-1, 8)[:G].tolist()
                t0 = min(x[0] for x in mv)
                ce = sorted((x[2] - t0) / 1e3 for x in mv)
                fc = sorted((x[1] - t0) / 1e3 for x in mv if x[1] > 0)
                spread[nm] = (statistics.median(ce), ce[-1], ce[-1] - statistics.median(ce),
                              statistics.median(fc) if fc else float("nan"))
            e = [int(v) for v in err.tolist()[:4]]
            rows.append({"bs": bs, "ne": ne, "B": B, "us": us, "rel": rel, "spread": spread, "err": e})
            log(f"[k0] bs={bs} r={r} experts {ne} ({B / 1e6:.1f} MB) err {e}: " +
                " ".join(f"{nm} {v:.1f}" for nm, v in us.items()) +
                f" | rel vs K: F2 {rel['F2']:.1e} F1 {rel['F1']:.1e} F3 {rel['F3']:.1e}")
            log("[k0]   consumer end p50/max/spread, first copy p50 (us): " +
                " ".join(f"{nm} {a:.1f}/{b:.1f}/{c:.2f}/{f:.2f}" for nm, (a, b, c, f) in spread.items()))
            del graphs
    # gate (pre-registered): per routing, best of F1/F2 <= B/3.7 TB/s + 2.5 us, spread <= 1.5 us; medians vs K
    log("[k0] GATE G-small (per routing: bar = B/3.7 TB/s + 2.5 us; best F = min(F1, F2))")
    ok_all = True
    for bs in (1, 8):
        rr = [x for x in rows if x["bs"] == bs]
        lim_k = 0.80 if bs == 1 else 0.90
        ratios = []
        for x in rr:
            cands = [nm for nm in ("F1", "F2") if nm in x["us"]] or [nm for nm in ("F3",) if nm in x["us"]]
            bestnm = min(cands, key=lambda nm: x["us"][nm])
            best = x["us"][bestnm]
            bar = x["B"] / 3.7e12 * 1e6 + 2.5
            sp = x["spread"][bestnm][2]
            ratios.append(best / x["us"]["K"])
            ok = best <= bar and sp <= 1.5 and x["err"][0] == 0
            ok_all &= ok
            log(f"[k0]   bs={bs} ne={x['ne']}: best {bestnm} {best:.1f} us vs bar {bar:.1f} | spread {sp:.2f} us | "
                f"x K {best / x['us']['K']:.3f} | S0 {x['us'].get('S0', float('nan')):.1f} "
                f"S {x['us'].get('S', float('nan')):.1f} K {x['us']['K']:.1f} -> {'pass' if ok else 'FAIL'}")
        mr = statistics.median(ratios)
        okk = mr <= lim_k
        ok_all &= okk
        log(f"[k0]   bs={bs}: median best/K {mr:.3f} (limit {lim_k}) -> {'pass' if okk else 'FAIL'}")
    log(f"[k0] G-small {'GO' if ok_all else 'NO-GO'}")
    del ws13g, ws2g, inter, yb, timer
    torch.cuda.empty_cache()
    return rows

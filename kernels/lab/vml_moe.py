"""VML1: one persistent launch for the T >= 17 MoE (phase A = W13 + SwiGLU, phase B = W2) + a finalize launch, on K2A's
ws13 / ws2 stage layouts (notes/study/vm_large.md s2). Our implementation of the public "WGS" structure (vjeux/okc-2026,
Apache-2.0, ideas only) with the R-vml-1 consumer (kernels/lab/vml_probe.py).

vm_main: grid = SMs, 576 threads (16 consumer warps = 4 WGs, weight producer warp 16, x producer warp 17), 1 CTA/SM.
  1. routing, redundantly in every CTA: match_any histograms per (warp, expert), warp-0 scans -> jobs (experts split into
     ceil(n / 64) balanced jobs), route rows in flat (token, slot) order; CTA 0 writes rmap (token, slot) -> route row.
  2. cost-weighted stream-K ranges over 256-K stages (phase A: job x 32 units x 16 stages, B: job x 16 x 16), cost by
     n-tiles from the R-vml-1 stage times; no integer divides (magic reciprocals).
  3. x gather: every CTA copies its share of the padded job rows into xA (K2A xw_u32 layout per job and stage), then
     red.release xready[job] += rows (padding rows counted, not written).
  4. phase A then B on one continuous weight ring (the weight producer runs ahead across the phase boundary); the x
     producer acquires xready (A) / doneA[job][stage] == 8 (B) before each copy. A unit cut between CTAs: red.add.v4.f32
     into the fixup slot of the CTA holding its first stage; the WG that completes the stage count (16) reads back,
     zeroes the slot and runs the epilogue. Phase A epilogue: SwiGLU -> bf16 straight into xB (phase B's x layout), then
     red.release doneA[job][unit / 2]. Phase B epilogue: y rows (f32, x gs2).
vm_fin: grid = M, 256 threads: K2A K3 numerics (bf16r(y) * nw, ss, rsqrt(ss / 4096 + 1e-8) * wt, slot order), and block 0
  resets xready / doneA / fixup counters for the next call. Bounded waits only (error flag, no trap).
"""
import hashlib
import time

import torch

E_LOC, TOPK, KDIM = 32, 8, 4096
MMAX = 320
JMAX = 80
RMAX = MMAX * TOPK
SLOTS = 136
FIXW = 8 * 128 * 4
COST = (128, 135, 147, 160, 171, 182, 193, 204)   # R-vml-1 stage us x 100 by n-tiles


def _mag(d):
    return -(-(1 << 40) // d)


def _wg_spec(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
    body = ("{\\n.reg .b64 dd;\\nmov.b64 dd, {%%%d, %%%d};\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 "
            "{%s}, {%s}, dd, %d, 1, 1, 0;\\n}\\n")
    s = ""
    for nm, sd, cons in (("wgmma_acc", 1, plus), ("wgmma_zero", 0, eq)):
        s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
              "uint32_t a3, uint32_t lo, uint32_t hi) {\n  asm volatile(\"" % (nm, n)
              + body % (na + 4, na + 5, n, outs, a, sd)
              + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}\n" % cons)
    return s


CUDA_T = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace vm {
constexpr int KDIM = 4096, E_LOC = 32, TOPK = 8;
constexpr int SW = 18432;
constexpr int US = 2 * SW;
constexpr int NS = 3, NX = 3;
constexpr int XS = 64 * 512;
constexpr int NCW = 16;
constexpr int NTH = 32 * NCW + 64;
constexpr int JMAX = @JMAX@, GMAX = 136, SLOTS = @SLOTS@, FIXW = @FIXW@;
constexpr int BAR_OFF = NS * US + NX * XS;
constexpr int TAB_OFF = BAR_OFF + 8 * 2 * (NS + NX);
constexpr long long WAIT_BUDGET = 1ll << 31;

struct Tab {
  int nj, R, Rpad, pad0;
  int wlast[4];
  int jE[JMAX], jC[JMAX], jS[JMAX + 1], jrow[JMAX + 1];
  uint32_t cpA[JMAX + 1], cpB[JMAX + 1];
  int bA[GMAX + 1], bB[GMAX + 1];
  int est[E_LOC];
};
constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);

struct Args {
  const __nv_bfloat16* x; const int* ids; const float* wts; int M; int ls;
  const uint8_t* ws13; const uint8_t* ws2; const float* gsc13; const float* gsc2; const float* nw;
  __nv_bfloat16* xA; __nv_bfloat16* xB; float* y; float* fix; int* cnt; int* xready; int* doneA; int* rmap;
  __nv_bfloat16* out; int* err; long long* marks; unsigned long long magG; int flags;
uint32_t cost[8]; unsigned long long mag[8];
};
__device__ __forceinline__ int split_point(const Args& A, int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;
  return ((A.flags & 1) && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}

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
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
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
      if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); return; }
    }
  }
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void wait_flag(const int* p, int target, int* err) {
  if (ld_acquire(p) >= target) return;
  const long long t0 = clock64();
  while (ld_acquire(p) < target) {
    __nanosleep(64);
    if (*reinterpret_cast<volatile int*>(err) != 0) return;
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 2); return; }
  }
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int atom_acqrel_add(int* p, int v) {
  int o;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(o) : "l"(p), "r"(v) : "memory");
  return o;
}
__device__ __forceinline__ void red_add_v4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.global;\n" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
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
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_acc(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <int N> __device__ __forceinline__ void wgmma_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
@WGSPECS@
__device__ __forceinline__ uint32_t mdiv(uint32_t a, unsigned long long m) {
  const unsigned long long lo = (unsigned long long)a * (m & 0xFFFFFFFFull);
  const unsigned long long hi = (unsigned long long)a * (m >> 32);
  return (uint32_t)((hi + (lo >> 32)) >> 8);
}
__device__ __forceinline__ int excl_scan(int v, int lane) {
  int s = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, s, o); if (lane >= o) s += n; }
  return s - v;
}

__device__ __forceinline__ int stage_at(const Args& A, const Tab& T, int pb, int c, int G, unsigned long long magG) {
  const int nj = T.nj;
  if (nj == 0) return 0;
  if (c >= G) return nj << (9 - pb);
  const uint32_t* cp = pb ? T.cpB : T.cpA;
  const uint32_t x = mdiv(cp[nj] * (uint32_t)c, magG);
  int lo = 0, hi = nj;
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (cp[mid] <= x) lo = mid; else hi = mid; }
  const int nt = (T.jC[lo] + 7) >> 3;
  uint32_t st = mdiv(x - cp[lo], A.mag[nt - 1]);
  const uint32_t ns = 512u >> pb;
  if (st > ns - 1) st = ns - 1;
  return (lo << (9 - pb)) + (int)st;
}
__device__ __forceinline__ int cta_of(const Tab& T, int pb, int s, int G) {
  const int* b = pb ? T.bB : T.bA;
  int lo = 0, hi = G;
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (b[mid] <= s) lo = mid; else hi = mid; }
  return lo;
}
__device__ void route(const Args& A, Tab& T, int* hist, uint16_t* ent) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int E8 = A.M * TOPK;
  const int C = (((E8 + 15) >> 4) + 31) & ~31;
  const int e0 = warp * C, e1 = min(E8, e0 + C);
  int xv[5];
#pragma unroll
  for (int k = 0; k < 5; ++k) {
    const int e = e0 + 32 * k + lane;
    xv[k] = -1;
    if (warp < NCW && 32 * k < C && e < e1) { const int id = __ldg(A.ids + e) - A.ls; xv[k] = ((unsigned)id < (unsigned)E_LOC) ? id : -1; }
  }
  for (int i = tid; i < 16 * 33; i += NTH) hist[i] = 0;
  __syncthreads();
  if (warp < NCW) {
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      if (e0 + 32 * k >= e1) break;
      const int x = xv[k];
      const unsigned m = __match_any_sync(0xffffffffu, x);
      if (x >= 0 && lane == __ffs(m) - 1) hist[warp * 33 + x] += __popc(m);
      __syncwarp();
    }
  }
  __syncthreads();
  if (warp == 0) {
    int tot = 0;
    for (int w = 0; w < NCW; ++w) { const int h = hist[w * 33 + lane]; hist[w * 33 + lane] = tot; tot += h; }
    const int est = excl_scan(tot, lane);
    const int nj = (tot + 63) >> 6;
    int base = 64;
    if (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }
    int prow = 0;
    for (int k = 0; k < nj; ++k) { const int sz = min(base, tot - k * base); prow += (sz + 7) & ~7; }
    const int jst = excl_scan(nj, lane), rst = excl_scan(prow, lane);
    int r = rst;
    for (int k = 0; k < nj; ++k) {
      const int j = jst + k, sz = min(base, tot - k * base);
      T.jE[j] = lane; T.jC[j] = sz; T.jS[j] = est + k * base; T.jrow[j] = r;
      r += (sz + 7) & ~7;
    }
    T.est[lane] = est;
    if (lane == 31) {
      T.nj = jst + nj; T.R = est + tot; T.Rpad = rst + prow;
      T.jS[jst + nj] = est + tot; T.jrow[jst + nj] = rst + prow;
    }
  }
  __syncthreads();
  if (warp < NCW) {
#pragma unroll
    for (int k = 0; k < 5; ++k) {
      if (e0 + 32 * k >= e1) break;
      const int e = e0 + 32 * k + lane, x = xv[k];
      const unsigned m = __match_any_sync(0xffffffffu, x);
      const int rank = __popc(m & ((1u << lane) - 1u));
      int r = -1;
      if (x >= 0) { r = T.est[x] + hist[warp * 33 + x] + rank; ent[r] = (uint16_t)e; }
      if (blockIdx.x == 0 && e < e1) A.rmap[e] = r;
      __syncwarp();
      if (x >= 0 && lane == __ffs(m) - 1) hist[warp * 33 + x] += __popc(m);
      __syncwarp();
    }
  }
  __syncthreads();
  if (warp == 0) {
    uint32_t accA = 0, accB = 0;
    for (int j0 = 0; j0 < T.nj; j0 += 32) {
      const int j = j0 + lane;
      const int ca = (j < T.nj) ? (int)A.cost[((T.jC[j] + 7) >> 3) - 1] : 0;
      const int sc = excl_scan(ca, lane);
      if (j < T.nj) { T.cpA[j] = accA + 512u * (uint32_t)sc; T.cpB[j] = accB + 256u * (uint32_t)sc; }
      const int tt = __shfl_sync(0xffffffffu, sc + ca, 31);
      accA += 512u * (uint32_t)tt; accB += 256u * (uint32_t)tt;
    }
    if (lane == 0) { T.cpA[T.nj] = accA; T.cpB[T.nj] = accB; }
  }
  __syncthreads();
  const int G = gridDim.x;
  if (tid <= G) { T.bA[tid] = stage_at(A, T, 0, tid, G, A.magG); T.bB[tid] = stage_at(A, T, 1, tid, G, A.magG); }
  __syncthreads();
}

__device__ void gather(const Args& A, const Tab& T, const uint16_t* ent) {
  const int tid = threadIdx.x, G = gridDim.x, c = blockIdx.x;
  const int q0 = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
  const int q1 = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
  const int s = tid >> 5, kk = (tid & 31) * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
  uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
  int j = 0;
  constexpr int GB = 8;
  for (int qq = q0; qq < q1; qq += GB) {
    uint4 v[GB];
    int off[GB];
#pragma unroll
    for (int k = 0; k < GB; ++k) {
      const int q = qq + k;
      off[k] = -1;
      if (q < q1) {
        while (T.jrow[j + 1] <= q) ++j;
        const int r = q - T.jrow[j], cn = T.jC[j];
        if (r < cn) {
          const int npad = (cn + 7) & ~7;
          off[k] = T.jrow[j] * 2048 + s * npad * 128 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
          v[k] = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + tid);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (off[k] >= 0) {
        uint32_t* d = xa + off[k];
        d[0] = v[k].x; d[32] = v[k].y; d[64] = v[k].z; d[96] = v[k].w;
      }
  }
  fence_proxy_async();
  bar_sync(5, 32 * NCW);
  if (tid == 0) {
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (T.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, T.jrow[jj + 1]);
      red_release_add(A.xready + jj, e - q);
      q = e;
    }
  }
}

__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {
  const uint64_t pol = policy_evict_first();
  const int c = blockIdx.x;
  int it = 0;
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
    for (int i = 0; i < g1 - g0; ++i, ++it) {
      const int gs = (sp + i < g1) ? sp + i : sp + i - (g1 - g0);
      const int u = gs >> 4, s = gs & 15;
      const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);
      const int e = T.jE[j];
      const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW
                              : A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;
      const int sl = it % NS;
      if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
      const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
      mbar_expect_tx(fb, US);
      bulk_g2s_ef(dst, src, SW, fb, pol);
      bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);
    }
  }
}
__device__ void prod_x(const Args& A, const Tab& T, uint32_t xring, uint32_t bars) {
  const int c = blockIdx.x;
  int it = 0;
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
    int lastj = -1;
    for (int i = 0; i < g1 - g0; ++i, ++it) {
      const int gs = (sp + i < g1) ? sp + i : sp + i - (g1 - g0);
      const int u = gs >> 4, s = gs & 15;
      const int j = pb ? (u >> 4) : (u >> 5);
      const int npad = (T.jC[j] + 7) & ~7;
      if (!pb) {
        if (j != lastj) { wait_flag(A.xready + j, npad, A.err); fence_proxy_async(); lastj = j; }
      } else {
        wait_flag(A.doneA + j * 16 + s, 8, A.err);
        fence_proxy_async();
      }
      const uint32_t bytes = (uint32_t)npad * 512u;
      const __nv_bfloat16* src = (pb ? A.xB : A.xA) + (size_t)T.jrow[j] * KDIM + (size_t)s * npad * 256;
      const int sl = it % NX;
      if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
      const uint32_t fb = bars + 8 * (2 * NS + sl);
      mbar_expect_tx(fb, bytes);
      bulk_g2s(xring + sl * XS, src, bytes, fb);
    }
  }
}

__device__ __forceinline__ void dec_half(uint32_t w0, uint32_t w8, uint32_t Sg, uint32_t S8, uint32_t* hh) {
  deq4(w0, hh); deq4(w8, hh + 4);
#pragma unroll
  for (int j = 0; j < 4; ++j) { hh[j] = bmul2(hh[j], Sg); hh[4 + j] = bmul2(hh[4 + j], S8); }
}
template <int N, bool Z>
__device__ __forceinline__ void mma2(float* d, const uint32_t* hh, uint32_t dlo, uint32_t dhi) {
  if constexpr (Z) wgmma_zero<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  else wgmma_acc<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + 16, dhi);
}
template <int N>
__device__ __forceinline__ void stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
                                      uint32_t (&hb)[2][8], int q, int wp, int lane, bool first) {
  const int ws = it % NS, xs = it % NX;
  mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  const uint8_t* st = smem + ws * US + (q >> 1) * SW;
  const int tile = (q & 1) * 4 + wp;
  const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
  uint4 c[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
  const uint32_t sw0 = sp[0], sw1 = sp[32];
  __syncwarp();
  if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
  mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
  __syncwarp();
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);
  const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
    const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
    const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t* hh = hb[hf];
      dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
      wg_fence();
      if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);
      else mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
      wg_commit();
      wg_wait<1>();
      if (i == 0 && hf == 0 && prevx >= 0) {
        if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
        prevx = -1;
      }
    }
  }
  prevx = xs;
  ++it;
}
template <bool PB, int N>
__device__ __forceinline__ void epilogue(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d) {
  const int g = lane >> 2, t = lane & 3, wt = (wp << 5) | lane;
  const int e = T.jE[j], cn = T.jC[j];
  const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;
  if constexpr (!PB) {
    const float gs = __ldg(A.gsc13 + e);
    const int npad = (cn + 7) & ~7;
    const int col = 8 * tg + g, s2 = col >> 8, kk = col & 255;
    const int i = kk >> 6, tt = (kk >> 4) & 3, qq = (kk >> 1) & 7, b = kk & 1;
    __nv_bfloat16* xb = A.xB + (size_t)T.jrow[j] * KDIM + (size_t)s2 * npad * 256;
#pragma unroll
    for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nt + 2 * t + hh;
        if (n < cn)
          xb[((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + hh)) * 4 + tt) * 2 + b] =
              __float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
      }
    fence_proxy_async();
    bar_sync(1 + q, 128);
    if (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);
  } else {
    const float gs = __ldg(A.gsc2 + e);
    const int h0 = 16 * tg + g;
    float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
    for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nt + 2 * t + hh;
        if (n < cn) {
          yb[(size_t)n * KDIM + h0] = d[4 * nt + hh] * gs;
          yb[(size_t)n * KDIM + h0 + 8] = d[4 * nt + 2 + hh] * gs;
        }
      }
  }
}
template <bool PB, int N>
__device__ __forceinline__ void seg(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int u, int s0, int s1,
                                    int q, int wp, int lane) {
  float d[N / 2];
  uint32_t hb[2][8];
  int prevx = -1;
  stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true);
#pragma unroll 1
  for (int s = s0 + 1; s < s1; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false);
  wg_wait<0>();
  if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
  constexpr int UBL = PB ? 4 : 5;
  const int j = u >> UBL, ub = u & ((1 << UBL) - 1);
  if (!(s0 == 0 && s1 == 16)) {
    const int wt = (wp << 5) | lane;
    const int slot = cta_of(T, PB, u << 4, gridDim.x);
    float* fb = A.fix + ((size_t)((PB * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);
    bar_sync(1 + q, 128);
    int* fc = A.cnt + (PB * SLOTS + slot) * 4 + q;
    if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);
    bar_sync(1 + q, 128);
    if (!T.wlast[q]) return;
    fence_acq_rel();
#pragma unroll
    for (int nt = 0; nt < N / 8; ++nt) {
      const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
      *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
      d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
    }
    if (wt == 0) *fc = 0;
  }
  epilogue<PB, N>(A, T, j, ub, q, wp, lane, d);
}
template <bool PB>
__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane) {
  const int c = blockIdx.x;
  const int g0 = PB ? T.bB[c] : T.bA[c], g1 = PB ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, PB, g0, g1);
#pragma unroll 1
  for (int gs = sp, ge = g1; ;) {
    if (gs >= ge) {
      if (ge == g1 && sp > g0) { gs = g0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    const int j = PB ? (u >> 4) : (u >> 5);
    if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * 32 + 8 + 8 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;
    switch ((T.jC[j] + 7) >> 3) {
      case 1: seg<PB, 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;
      case 2: seg<PB, 16>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;
      case 3: seg<PB, 24>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;
      case 4: seg<PB, 32>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;
      case 5: seg<PB, 40>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;
      case 6: seg<PB, 48>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;
      case 7: seg<PB, 56>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;
      default: seg<PB, 64>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;
    }
    gs += s1 - s0;
  }
}

__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {
  extern __shared__ __align__(1024) uint8_t smem[];
  Tab& T = *reinterpret_cast<Tab*>(smem + TAB_OFF);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = sa(smem + BAR_OFF);
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
    for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (A.marks) A.marks[(size_t)blockIdx.x * 32] = gtime();
  }
  int* hist = reinterpret_cast<int*>(smem + NS * US);
  uint16_t* ent = reinterpret_cast<uint16_t*>(smem + NS * US + 4096);
  route(A, T, hist, ent);
  if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 7] = gtime();
  if (warp >= NCW) {
    if (warp == NCW) {
      if (lane == 0) prod_w(A, T, sa(smem), bars);
    } else {
      bar_sync(6, 32 * NCW + 32);
      if (lane == 0) prod_x(A, T, sa(smem + NS * US), bars);
    }
    return;
  }
  gather(A, T, ent);
  bar_arrive(6, 32 * NCW + 32);
  if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 5] = gtime();
  const int q = warp >> 2, wp = warp & 3;
  int it = 0;
  phase<false>(A, T, smem, bars, it, q, wp, lane);
  if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * 32 + 6] = gtime();
  phase<true>(A, T, smem, bars, it, q, wp, lane);
  if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 1 + q] = gtime();
}

__global__ void __launch_bounds__(256) vm_fin(const Args A) {
  __shared__ int s_r[8];
  __shared__ float s_w[8];
  __shared__ float s_ss[8][8];
  __shared__ float s_fr[8];
  const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (t == 0) {
    for (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;
    for (int i = tid; i < JMAX * 16; i += 256) A.doneA[i] = 0;
    for (int i = tid; i < 2 * SLOTS * 4; i += 256) A.cnt[i] = 0;
  }
  if (tid < 8) { s_r[tid] = A.rmap[t * 8 + tid]; s_w[tid] = __ldg(A.wts + t * 8 + tid); }
  __syncthreads();
  float ss[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    ss[k] = 0.f;
    const int r = s_r[k];
    if (r >= 0) {
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const int col = 1024 * m + 4 * tid;
        const float4 v = __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)r * KDIM + col));
        const float4 w = __ldg(reinterpret_cast<const float4*>(A.nw + col));
        float f;
        f = bf16r(v.x) * w.x; ss[k] += f * f; f = bf16r(v.y) * w.y; ss[k] += f * f;
        f = bf16r(v.z) * w.z; ss[k] += f * f; f = bf16r(v.w) * w.w; ss[k] += f * f;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
  }
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < 8; ++k) s_ss[warp][k] = ss[k];
  }
  __syncthreads();
  if (tid < 8) {
    float s = 0.f;
    for (int w = 0; w < 8; ++w) s += s_ss[w][tid];
    s_fr[tid] = (s_r[tid] >= 0) ? rsqrtf(s * (1.f / 4096.f) + 1e-8f) * s_w[tid] : 0.f;
  }
  __syncthreads();
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const int col = 1024 * m + 4 * tid;
    const float4 w = __ldg(reinterpret_cast<const float4*>(A.nw + col));
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = s_r[k];
      if (r >= 0) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)r * KDIM + col));
        const float fr = s_fr[k];
        a0 += bf16r(v.x) * w.x * fr; a1 += bf16r(v.y) * w.y * fr;
        a2 += bf16r(v.z) * w.z * fr; a3 += bf16r(v.w) * w.w * fr;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a0, a1), o1 = __floats2bfloat162_rn(a2, a3);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0); o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(A.out + (size_t)t * KDIM + col) = o;
  }
}
}

void vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V) {
  vm::Args a;
  const int64_t M = V[0], G = V[2], mask = V[4];
  a.x = reinterpret_cast<const __nv_bfloat16*>(T[0].data_ptr()); a.ids = T[1].data_ptr<int>(); a.wts = T[2].data_ptr<float>();
  a.M = (int)M; a.ls = (int)V[1];
  a.ws13 = reinterpret_cast<const uint8_t*>(T[3].data_ptr()); a.ws2 = reinterpret_cast<const uint8_t*>(T[4].data_ptr());
  a.gsc13 = T[5].data_ptr<float>(); a.gsc2 = T[6].data_ptr<float>(); a.nw = T[7].data_ptr<float>();
  a.xA = reinterpret_cast<__nv_bfloat16*>(T[8].data_ptr()); a.xB = reinterpret_cast<__nv_bfloat16*>(T[9].data_ptr());
  a.y = T[10].data_ptr<float>(); a.fix = T[11].data_ptr<float>(); a.cnt = T[12].data_ptr<int>();
  a.xready = T[13].data_ptr<int>(); a.doneA = T[14].data_ptr<int>(); a.rmap = T[15].data_ptr<int>();
  a.out = reinterpret_cast<__nv_bfloat16*>(T[16].data_ptr()); a.err = T[17].data_ptr<int>();
  a.marks = T[18].numel() >= G * 32 ? reinterpret_cast<long long*>(T[18].data_ptr()) : nullptr;
  a.magG = (unsigned long long)V[3]; a.flags = (int)V[5];
  for (int k = 0; k < 8; ++k) { a.cost[k] = (uint32_t)V[6 + k]; a.mag[k] = ((1ull << 40) + V[6 + k] - 1) / V[6 + k]; }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (mask & 1) {
    cudaFuncSetAttribute(vm::vm_main, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
    vm::vm_main<<<(int)G, vm::NTH, vm::SMEM, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (mask & 2) {
    vm::vm_fin<<<(int)M, 256, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
"""
CUDA = (CUDA_T.replace("@WGSPECS@", "".join(_wg_spec(n) for n in (8, 16, 24, 32, 40, 48, 56, 64)))
        .replace("@JMAX@", str(JMAX)).replace("@SLOTS@", str(SLOTS)).replace("@FIXW@", str(FIXW))
        )

CPP = r"""
#include <torch/extension.h>
#include <vector>
void vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V);
"""
FLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def build(verbose=False):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CPP + CUDA + " ".join(FLAGS)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vml_moe_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["vml_run"],
                      extra_cuda_cflags=FLAGS, verbose=verbose)
    return mod, time.time() - t0


def alloc(dev, G):
    z = lambda n, dt=torch.int32: torch.zeros(n, dtype=dt, device=dev)  # noqa: E731
    xrows = RMAX + 8 * JMAX
    assert G <= SLOTS and G <= 136
    return {"xA": z((xrows * KDIM,), torch.bfloat16), "xB": z((xrows * KDIM,), torch.bfloat16),
            "y": z((RMAX * KDIM,), torch.float32), "fix": z((2 * SLOTS * 4 * FIXW,), torch.float32),
            "cnt": z((2 * SLOTS * 4,)), "xready": z((JMAX,)), "doneA": z((JMAX * 16,)), "rmap": z((RMAX,)),
            "err": z((4,)), "nomark": z((0,), torch.int64), "G": G, "magG": _mag(G)}


FLAGS_DEFAULT = 1   # bit 0: phase A trailing partial unit first


def call(mod, k, sc, x, ids, wts, ls=0, marks=None, mask=3, out=None, flags=None, cost=None):
    m = x.shape[0]
    assert m <= MMAX
    flags = FLAGS_DEFAULT if flags is None else flags
    if out is None:
        out = torch.empty((m, KDIM), dtype=torch.bfloat16, device=x.device)
    mod.vml_run([x, ids, wts, k["ws13"], k["ws2"], k["gsc13"], k["gsc2"], k["nw"], sc["xA"], sc["xB"], sc["y"], sc["fix"],
                 sc["cnt"], sc["xready"], sc["doneA"], sc["rmap"], out, sc["err"], sc["nomark"] if marks is None else marks],
                [m, ls, sc["G"], sc["magG"], mask, flags] + list(COST if cost is None else cost))
    return out

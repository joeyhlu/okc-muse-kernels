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
Lab v9: marks cols 24..28 from the x producer (B doneA wait ns / count, A xready wait ns, first / last B acquire); only with marks.
Lab v13: compile-time VM_TRICKS (default 1; build(defines=["-DVM_TRICKS=0"]) = the v12 consumer) on the RS-wgmma stage
(nt >= 3 with bit 7): codes one 4-step group ahead (weight slot freed after the last group's load), deq4's two left shifts as
multiplies by opaque constants (FMA pipe), one PRMT + IMAD for both rows' scales with half-broadcast multiplies, x
descriptor = slot-0 descriptor + slot offset (R-vml-15 probe ALL: -1.1..-3.4 % per stage; bitwise-identical products).
Lab v12: compile-time VM_DEFER = 1 (build(defines=["-DVM_DEFER=1"])): a phase-A whole unit's proxy fence + WG barrier +
doneA red.release are not issued after its SwiGLU stores but kept in a per-WG pending record and flushed behind commit
group 3 of the WG's next segment's first stage (mma.sync path: after its first stage), and at the end of phase A (never
across the A -> B boundary: phase B's x waits on these releases). vml_vs_wgs.md s4 #1, the doneA part.
Lab v11: marks cols 29..31 = %smid and two L2 latency probes (cycles per dependent ld.cg, addresses 4 KB apart); marks only.
Lab v10: flags bit 9 = each job's 16 phase-B stages in order of predicted A-unit readiness (afin / bperm_lanes, computed by the
x-producer warp's idle lanes during phase A; consumers unchanged: only the K order of the B sums changes).
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
  int bready, pad1;                 // bit 9: B stage orders below are written (x-producer lanes 1-31, during phase A)
  uint32_t brd[16];                 // bit 9 scratch: predicted readiness of one job's 16 B stages
  uint8_t bpos[16];                 // bit 9 scratch: order position -> stage
  unsigned long long bperm[JMAX];   // bit 9: per job, 16 x 4 bits: B order position -> stage
};
constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);

struct Args {
  const __nv_bfloat16* x; const int* ids; const float* wts; int M; int ls;
  const uint8_t* ws13; const uint8_t* ws2; const float* gsc13; const float* gsc2; const float* nw;
  __nv_bfloat16* xA; __nv_bfloat16* xB; float* y; float* fix; int* cnt; int* xready; int* doneA; int* rmap; int* gbar;
  __nv_bfloat16* out; int* err; long long* marks; unsigned long long magG; int flags;
uint32_t cost[16]; unsigned long long mag[16];
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
__device__ __noinline__ long long lat_probe(const uint8_t* p, int flags) {   // marks only: cycles per dependent L2 load
  const unsigned z = (unsigned)flags >> 30;   // 0, opaque to the compiler
  unsigned v;
  asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  const long long t0 = clock64();
#pragma unroll 1
  for (int k = 0; k < 8; ++k) asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p + (v & z)) : "memory");
  asm volatile("add.u32 %0, %0, 0;\n" : "+r"(v));
  return (clock64() - t0 + (long long)(v & z)) / 7;
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
  uint32_t st = mdiv(x - cp[lo], A.mag[8 * pb + nt - 1]);
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
__device__ void route(const Args& A, Tab& T, int* hist, uint16_t* ent, uint32_t* fin) {
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
      const int nt = (j < T.nj) ? ((T.jC[j] + 7) >> 3) - 1 : 0;
      const int ca = (j < T.nj) ? (int)A.cost[nt] : 0, cb = (j < T.nj) ? (int)A.cost[8 + nt] : 0;
      const int sc = excl_scan(ca, lane), sd = excl_scan(cb, lane);
      if (j < T.nj) { T.cpA[j] = accA + 512u * (uint32_t)sc; T.cpB[j] = accB + 256u * (uint32_t)sd; }
      accA += 512u * (uint32_t)__shfl_sync(0xffffffffu, sc + ca, 31);
      accB += 256u * (uint32_t)__shfl_sync(0xffffffffu, sd + cb, 31);
    }
    if (lane == 0) { T.cpA[T.nj] = accA; T.cpB[T.nj] = accB; }
  }
  __syncthreads();
  const int G = gridDim.x;
  if (tid <= G) { T.bA[tid] = stage_at(A, T, 0, tid, G, A.magG); T.bB[tid] = stage_at(A, T, 1, tid, G, A.magG); }
  if (tid == 0) T.bready = 0;
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

__device__ void gather_warp(const Args& A, const Tab& T, const uint16_t* ent, int lane) {
  const int G = gridDim.x, c = blockIdx.x;
  const int q0 = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
  const int q1 = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
  const int kk = lane * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
  uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
  int j = 0;
  for (int q = q0; q < q1; ++q) {
    while (T.jrow[j + 1] <= q) ++j;
    const int r = q - T.jrow[j], cn = T.jC[j];
    if (r >= cn) continue;
    const int npad = (cn + 7) & ~7;
    const uint4* src = reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + lane;
    uint4 v[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) v[m] = __ldg(src + 32 * m);
    uint32_t* d = xa + T.jrow[j] * 2048 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
#pragma unroll
    for (int m = 0; m < 16; ++m) {
      uint32_t* dm = d + m * npad * 128;
      dm[0] = v[m].x; dm[32] = v[m].y; dm[64] = v[m].z; dm[96] = v[m].w;
    }
  }
  fence_proxy_async();
  __syncwarp();
  if (lane == 0) {
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (T.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, T.jrow[jj + 1]);
      red_release_add(A.xready + jj, e - q);
      q = e;
    }
  }
}
__device__ __forceinline__ int nfill_of(const Args& A, const Tab& T) {
  return (A.flags & 64) ? min(NX - 1, T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) : 0;
}
// consumers fill the x slots of the CTA's first nf phase-A stages straight from x (no xA round trip)
__device__ void self_fill(const Args& A, const Tab& T, uint8_t* smem, uint32_t bars, const uint16_t* ent, int nf) {
  const int tid = threadIdx.x, c = blockIdx.x;
  const int g0 = T.bA[c], g1 = T.bA[c + 1], sp = split_point(A, 0, g0, g1);
  for (int f = 0; f < nf; ++f) {
    const int gs = (sp + f < g1) ? sp + f : sp + f - (g1 - g0);
    const int j = gs >> 9, st = gs & 15, cn = T.jC[j];
    uint32_t* xs = reinterpret_cast<uint32_t*>(smem + NS * US + f * XS);
    for (int w = tid; w < cn * 32; w += 32 * NCW) {
      const int r = w >> 5, ch = w & 31, i = ch >> 3, t = (ch >> 1) & 3, qb = (ch & 1) * 4;
      const uint4 v = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM + 256 * st) + ch);
      uint32_t* d = xs + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
      d[0] = v.x; d[32] = v.y; d[64] = v.z; d[96] = v.w;
    }
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  bar_sync(5, 32 * NCW);
  if (tid == 0)
    for (int f = 0; f < nf; ++f) mbar_arrive(bars + 8 * (2 * NS + f));
}

// bit 9: B stage order by predicted A readiness. afin = predicted finish (16.16 fraction of its owner's range, in the owner's
// phase-A processing order, max over owners) of A unit U; a B stage of job j reads A units 2s / 2s + 1. Pure function of the
// global plan (bA), so every CTA that holds part of a B unit derives the same order. Computed by lanes 1-31 of the x-producer
// warp while phase A runs (no prologue cost, R-vml-5's lesson); the producers wait for bready before their first B stage.
__device__ __forceinline__ uint32_t afin(const Args& A, const Tab& T, int U, int G) {
  const int a = U << 4, b = a + 16;
  int lo = 0, hi = G - 1;
  while (lo < hi) { const int m = (lo + hi) >> 1; if (T.bA[m + 1] > a) hi = m; else lo = m + 1; }
  uint32_t f = 0;
  for (int c = lo; c < G && T.bA[c] < b; ++c) {
    const int g0 = T.bA[c], g1 = T.bA[c + 1];
    if (g1 <= g0) continue;
    const int sp = split_point(A, 0, g0, g1), l2 = max(a, g0), h2 = min(b, g1);
    if (h2 <= l2) continue;
    const int pos = (l2 < sp) ? (g1 - sp) + (min(h2, sp) - 1 - g0) : h2 - 1 - sp;
    const uint32_t fc = (uint32_t)(((unsigned long long)(pos + 1) << 16) / (unsigned)(g1 - g0));
    f = fc > f ? fc : f;
  }
  return f;
}
__device__ void bperm_lanes(const Args& A, Tab& T, int lane) {
  const unsigned mask = 0xfffffffeu;
  const int c = blockIdx.x, G = gridDim.x, g0 = T.bB[c], g1 = T.bB[c + 1];
  if (g1 > g0) {
    for (int j = g0 >> 8; j <= (g1 - 1) >> 8; ++j) {
      for (int s = lane - 1; s < 16; s += 31) {
        const uint32_t f0 = afin(A, T, 32 * j + 2 * s, G), f1 = afin(A, T, 32 * j + 2 * s + 1, G);
        T.brd[s] = f0 > f1 ? f0 : f1;
      }
      __syncwarp(mask);
      for (int s = lane - 1; s < 16; s += 31) {
        const uint32_t r = T.brd[s];
        int rank = 0;
        for (int t = 0; t < 16; ++t) { const uint32_t q = T.brd[t]; rank += (q < r) || (q == r && t < s); }
        T.bpos[rank] = (uint8_t)s;
      }
      __syncwarp(mask);
      if (lane == 1) {
        unsigned long long pm = 0;
        for (int r = 0; r < 16; ++r) pm |= (unsigned long long)T.bpos[r] << (4 * r);
        T.bperm[j] = pm;
      }
      __syncwarp(mask);
    }
  }
  if (lane == 1) { __threadfence_block(); *reinterpret_cast<volatile int*>(&T.bready) = 1; }
}
__device__ __forceinline__ void bperm_wait(const Args& A, const Tab& T) {
  if (*reinterpret_cast<const volatile int*>(&T.bready) != 0) { __threadfence_block(); return; }
  const long long t0 = clock64();
  while (*reinterpret_cast<const volatile int*>(&T.bready) == 0) {
    __nanosleep(64);
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 3); break; }
  }
  __threadfence_block();
}
__device__ __forceinline__ int bstage(const Args& A, const Tab& T, int j, int s) {
  return (A.flags & 512) ? (int)((T.bperm[j] >> (4 * s)) & 15ull) : s;
}

__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {
  const uint64_t pol = policy_evict_first();
  const int c = blockIdx.x;
  int it = 0;
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
    if (pb && (A.flags & 512)) bperm_wait(A, T);
    for (int i = 0; i < g1 - g0; ++i, ++it) {
      const int gs = (sp + i < g1) ? sp + i : sp + i - (g1 - g0);
      const int u = gs >> 4;
      const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);
      const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);
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
  const int c = blockIdx.x, nf = nfill_of(A, T);
  int it = 0;
  long long wA = 0, wB = 0, tB0 = 0, tB1 = 0;   // marks only (v9): readiness waits of this CTA's x producer
  int nB = 0;
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
    if (pb && (A.flags & 512)) bperm_wait(A, T);
    int lastj = -1;
    for (int i = 0; i < g1 - g0; ++i, ++it) {
      const int gs = (sp + i < g1) ? sp + i : sp + i - (g1 - g0);
      const int u = gs >> 4;
      const int j = pb ? (u >> 4) : (u >> 5);
      const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);
      const int npad = (T.jC[j] + 7) & ~7;
      if (!pb && i < nf) continue;
      if (!pb) {
        if (j != lastj) {
          if (A.marks && ld_acquire(A.xready + j) < npad) {
            const long long t = gtime();
            wait_flag(A.xready + j, npad, A.err);
            wA += gtime() - t;
          } else {
            wait_flag(A.xready + j, npad, A.err);
          }
          fence_proxy_async(); lastj = j;
        }
      } else {
        if (A.marks && ld_acquire(A.doneA + j * 16 + s) < 8) {
          const long long t = gtime();
          wait_flag(A.doneA + j * 16 + s, 8, A.err);
          wB += gtime() - t; ++nB;
        } else {
          wait_flag(A.doneA + j * 16 + s, 8, A.err);
        }
        fence_proxy_async();
        if (A.marks) { tB1 = gtime(); if (tB0 == 0) tB0 = tB1; }
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
  if (A.marks) {   // cols 24..28: B doneA wait ns, B stages that waited, A xready wait ns, first / last B x acquire (gtime)
    long long* m = A.marks + (size_t)c * 32;
    m[24] = wB; m[25] = nB; m[26] = wA; m[27] = tB0; m[28] = tB1;
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
#ifndef VM_DEFER
#define VM_DEFER 0
#endif
#ifndef VM_TRICKS
#define VM_TRICKS 1
#endif
#ifndef VM_TRICKS_DQ
#define VM_TRICKS_DQ 1   // 0: VM_TRICKS without the opaque-multiply dequant (2 registers fewer)
#endif
__device__ __forceinline__ void deq4_dq(uint32_t w, uint32_t* o, uint32_t k64, uint32_t k4) {   // == deq4 (same bits)
  o[0] = w & 0x81C081C0u;
  o[1] = (w * k64) & 0x81C081C0u;                                  // rotl 6: the mask keeps no wrapped bit
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w * k4) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {   // a x (b.lo, b.lo): HMUL2 .H0_H0
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {   // a x (b.hi, b.hi): HMUL2 .H1_H1
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
// w0 = row g, w8 = row g + 8 (2 k16 steps each); sgp = (row g scale, row g + 8 scale) as one bf16x2
__device__ __forceinline__ void dec_half2(uint32_t w0, uint32_t w8, uint32_t sw2, uint32_t par, uint32_t* hh, uint32_t k16,
                                          uint32_t k64, uint32_t k4) {
#if VM_TRICKS_DQ
  deq4_dq(w0, hh, k64, k4); deq4_dq(w8, hh + 4, k64, k4);
#else
  deq4(w0, hh); deq4(w8, hh + 4);
#endif
  const uint32_t sb = 2u * par;
  const uint32_t sgp = __byte_perm(sw2, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16 + 0x75807580u;
#pragma unroll
  for (int j = 0; j < 4; ++j) { hh[j] = bmul2l(hh[j], sgp); hh[4 + j] = bmul2h(hh[4 + j], sgp); }
}
struct Pend { int i; };   // doneA index (job x 16 + unit / 2) whose release is still pending (< 0: none); WG-uniform
__device__ __forceinline__ void pend_flush(const Args& A, Pend& pd, int q, int wp, int lane) {
  if (pd.i >= 0) {
    fence_proxy_async();
    bar_sync(1 + q, 128);
    if (((wp << 5) | lane) == 0) red_release_add(A.doneA + pd.i, 1);
    pd.i = -1;
  }
}
template <int N>
__device__ __forceinline__ void stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
                                      uint32_t (&hb)[2][8], int q, int wp, int lane, bool first, Pend& pd) {
  const int ws = it % NS, xs = it % NX;
  mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  const uint8_t* st = smem + ws * US + (q >> 1) * SW;
  const int tile = (q & 1) * 4 + wp;
  const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
  uint4 c[4];
#if VM_TRICKS
  c[0] = cp[0];
  const uint32_t sw0 = sp[0], sw1 = sp[32];
#else
#pragma unroll
  for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
  const uint32_t sw0 = sp[0], sw1 = sp[32];
  __syncwarp();
  if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
#endif
  mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
  __syncwarp();
#if VM_TRICKS
  const uint32_t zo = (uint32_t)A.flags >> 30;   // 0 (flags < 2^30), opaque to the compiler
  const uint32_t k16 = 16u + zo, k64 = VM_TRICKS_DQ ? 64u + zo : 64u, k4 = VM_TRICKS_DQ ? 4u + zo : 4u;
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US), 4096u);
  const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);
#else
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);
  const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#endif
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
#if VM_TRICKS
    if (i < 3) c[i + 1] = cp[32 * (i + 1)];   // the next 4-step group's codes, one group ahead
#else
    const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
    const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#endif
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t* hh = hb[hf];
#if VM_TRICKS
      dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16, k64, k4);
      if (i == 3 && hf == 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // all codes in registers
#else
      dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
#endif
      wg_fence();
      if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);
      else mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
      wg_commit();
      wg_wait<1>();
      if (i == 0 && hf == 0 && prevx >= 0) {
        if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
        prevx = -1;
      }
      if constexpr (VM_DEFER) {
        if (first && i == 1 && hf == 1) pend_flush(A, pd, q, wp, lane);   // behind commit group 3
      }
    }
  }
  prevx = xs;
  ++it;
}
__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// small jobs (nt <= 2): mma.sync m16n8k16 per warp (its 16-row tile), B by ldmatrix.x4 from the shared x slot, 2 chains
template <int N>
__device__ __forceinline__ void stage_m(const Args& A, uint8_t* smem, uint32_t bars, int& it, float (&d2)[2][N / 2], int q,
                                        int wp, int lane) {
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
  const uint32_t xb = sa(smem + NS * US + xs * XS) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
    const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
    const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t hh[8];
      dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
#pragma unroll
      for (int nt = 0; nt < N / 8; ++nt) {
        uint32_t b0, b1, b2, b3;
        ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
        mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);
        mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);
      }
    }
  }
  __syncwarp();
  if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
  ++it;
}
template <bool PB, int N>
__device__ __forceinline__ void epilogue(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d,
                                         Pend& pd) {
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
    if constexpr (VM_DEFER) {
      pd.i = j * 16 + (ub >> 1);
    } else {
      fence_proxy_async();
      bar_sync(1 + q, 128);
      if (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);
    }
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
                                    int q, int wp, int lane, Pend& pd) {
  float d[N / 2];
  bool sync_path = false;
  if constexpr (N <= 16) sync_path = (A.flags & 128) != 0;
  if (sync_path) {
    float d2[2][N / 2];
#pragma unroll
    for (int k = 0; k < N / 2; ++k) { d2[0][k] = 0.f; d2[1][k] = 0.f; }
    stage_m<N>(A, smem, bars, it, d2, q, wp, lane);
    if constexpr (VM_DEFER) pend_flush(A, pd, q, wp, lane);
#pragma unroll 1
    for (int s = s0 + 1; s < s1; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane);
#pragma unroll
    for (int k = 0; k < N / 2; ++k) d[k] = d2[0][k] + d2[1][k];
  } else {
    uint32_t hb[2][8];
    int prevx = -1;
    stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd);
#pragma unroll 1
    for (int s = s0 + 1; s < s1; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd);
    wg_wait<0>();
    if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
  }
  constexpr int UBL = PB ? 4 : 5;
  const int j = u >> UBL, ub = u & ((1 << UBL) - 1);
  if (!(s0 == 0 && s1 == 16)) {
    const int wt = (wp << 5) | lane;
    const int slot = cta_of(T, PB, u << 4, gridDim.x);
    float* fb = A.fix + ((size_t)((PB * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;
    int* fc = A.cnt + (PB * SLOTS + slot) * 4 + q;
#pragma unroll
    for (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);
    bar_sync(1 + q, 128);
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
  epilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);
}
template <bool PB>
__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane) {
  const int c = blockIdx.x;
  const int g0 = PB ? T.bB[c] : T.bA[c], g1 = PB ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, PB, g0, g1);
  Pend pd{-1};
#pragma unroll 1
  for (int gs = sp, ge = g1; ;) {
    if (gs >= ge) {
      if (ge == g1 && sp > g0) { gs = g0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    const int j = PB ? (u >> 4) : (u >> 5);
    if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * 32 + 8 + 8 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;
    switch ((T.jC[j] + 7) >> 3) {
      case 1: seg<PB, 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
      case 2: seg<PB, 16>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
      case 3: seg<PB, 24>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
      case 4: seg<PB, 32>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
      case 5: seg<PB, 40>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
      case 6: seg<PB, 48>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
      case 7: seg<PB, 56>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
      default: seg<PB, 64>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
    }
    gs += s1 - s0;
  }
  if constexpr (VM_DEFER && !PB) pend_flush(A, pd, q, wp, lane);   // before phase B: its x waits on these releases
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
    if (A.marks) {   // v11: SM id + two L2 latency probes (4 KB apart: H200 partition class), then the start mark
      long long* m = A.marks + (size_t)blockIdx.x * 32;
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
      m[29] = smid;
      m[30] = lat_probe(A.ws13, A.flags);
      m[31] = lat_probe(A.ws13 + 4096, A.flags);
      m[0] = gtime();
    }
  } else if (tid == 32 && (A.flags & 2)) {
    const uint32_t tot = (uint32_t)A.M * (KDIM * 2), S = (mdiv(tot + gridDim.x - 1, A.magG) + 15u) & ~15u, off = blockIdx.x * S;
    if (off < tot)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(reinterpret_cast<const uint8_t*>(A.x) + off),
                   "r"(min(S, tot - off)) : "memory");
  }
  if (A.flags & 256) asm volatile("griddepcontrol.wait;\n" ::: "memory");   // PDL: ids / x come from the previous kernel
  uint8_t* scr = smem + NS * US + (NX - 1) * XS;
  int* hist = reinterpret_cast<int*>(scr);
  uint16_t* ent = reinterpret_cast<uint16_t*>(scr + 4096);
  route(A, T, hist, ent, reinterpret_cast<uint32_t*>(scr + 12288));
  if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 7] = gtime();
  if (warp >= NCW) {
    if (warp == NCW) {
      if (A.flags & 4) bar_sync(7, 32 * NCW + 32);
      if (lane == 0) prod_w(A, T, sa(smem), bars);
    } else {
      if (A.flags & 64) gather_warp(A, T, ent, lane);
      bar_sync(6, 32 * NCW + 32);
      if (lane == 0) prod_x(A, T, sa(smem + NS * US), bars);
      else if (A.flags & 512) bperm_lanes(A, T, lane);
    }
    return;
  }
  if (A.flags & 64) self_fill(A, T, smem, bars, ent, nfill_of(A, T));
  else gather(A, T, ent);
  bar_arrive(6, 32 * NCW + 32);
  if (A.flags & 4) bar_arrive(7, 32 * NCW + 32);
  if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 5] = gtime();
  const int q = warp >> 2, wp = warp & 3;
  int it = 0;
  phase<false>(A, T, smem, bars, it, q, wp, lane);
  if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * 32 + 6] = gtime();
  phase<true>(A, T, smem, bars, it, q, wp, lane);
  if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 1 + q] = gtime();
}

// finalize (post-expert norm of every local route's y row, routing weight, sum in slot order = K2A K3 numerics; resets the
// counters for the next call). VG-F's structure (glue builder, tools/vg_patches.py): the token's local slots compacted
// first, then the y loads of up to 4 routes in flight at once; PDL launch, the wait after the norm-weight loads.
__global__ void __launch_bounds__(256) vm_fin(const Args A) {
  __shared__ int s_r[8], s_r8[8];
  __shared__ float s_w[8], s_w8[8];
  __shared__ int s_n;
  __shared__ float s_ss[2][8][4];
  const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float4 w4[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * tid));
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  if (t == 0) {
    for (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;
    for (int i = tid; i < JMAX * 16; i += 256) A.doneA[i] = 0;
    for (int i = tid; i < 2 * SLOTS * 4; i += 256) A.cnt[i] = 0;
  }
  if (tid < 8) { s_r8[tid] = __ldcg(A.rmap + t * 8 + tid); s_w8[tid] = __ldg(A.wts + t * 8 + tid); }
  __syncthreads();
  if (tid == 0) {
    int n = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = s_r8[k];
      if (r >= 0) { s_r[n] = r; s_w[n] = s_w8[k]; ++n; }
    }
    s_n = n;
  }
  __syncthreads();
  const int nloc = s_n;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  float a[4][4];
#pragma unroll
  for (int m = 0; m < 4; ++m) { a[m][0] = 0.f; a[m][1] = 0.f; a[m][2] = 0.f; a[m][3] = 0.f; }
  int par = 0;
  for (int c0 = 0; c0 < nloc; c0 += 4, par ^= 1) {
    float4 v[4][4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;
#pragma unroll
      for (int m = 0; m < 4; ++m)
        v[jj][m] = (r >= 0) ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * tid)) : z4;
    }
    float ss[4];
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      ss[jj] = 0.f;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        float f;
        f = bf16r(v[jj][m].x) * w4[m].x; ss[jj] += f * f; f = bf16r(v[jj][m].y) * w4[m].y; ss[jj] += f * f;
        f = bf16r(v[jj][m].z) * w4[m].z; ss[jj] += f * f; f = bf16r(v[jj][m].w) * w4[m].w; ss[jj] += f * f;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss[jj] += __shfl_xor_sync(0xffffffffu, ss[jj], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) s_ss[par][warp][jj] = ss[jj];
    }
    __syncthreads();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      if (c0 + jj < nloc) {
        float sm = 0.f;
        for (int w = 0; w < 8; ++w) sm += s_ss[par][w][jj];
        const float fr = rsqrtf(sm * (1.f / 4096.f) + 1e-8f) * s_w[c0 + jj];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          a[m][0] += bf16r(v[jj][m].x) * w4[m].x * fr; a[m][1] += bf16r(v[jj][m].y) * w4[m].y * fr;
          a[m][2] += bf16r(v[jj][m].z) * w4[m].z * fr; a[m][3] += bf16r(v[jj][m].w) * w4[m].w * fr;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a[m][0], a[m][1]), o1 = __floats2bfloat162_rn(a[m][2], a[m][3]);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0); o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(A.out + (size_t)t * KDIM + 1024 * m + 4 * tid) = o;
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
  a.xready = T[13].data_ptr<int>(); a.doneA = T[14].data_ptr<int>(); a.rmap = T[15].data_ptr<int>(); a.gbar = T[19].data_ptr<int>();
  a.out = reinterpret_cast<__nv_bfloat16*>(T[16].data_ptr()); a.err = T[17].data_ptr<int>();
  a.marks = T[18].numel() >= G * 32 ? reinterpret_cast<long long*>(T[18].data_ptr()) : nullptr;
  a.magG = (unsigned long long)V[3]; a.flags = (int)V[5];
  for (int k = 0; k < 16; ++k) { a.cost[k] = (uint32_t)V[6 + k]; a.mag[k] = ((1ull << 40) + V[6 + k] - 1) / V[6 + k]; }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (mask & 1) {
    cudaFuncSetAttribute(vm::vm_main, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
    cudaLaunchConfig_t cm = {};
    cm.gridDim = dim3((unsigned)G); cm.blockDim = dim3(vm::NTH); cm.dynamicSmemBytes = vm::SMEM; cm.stream = stream;
    cudaLaunchAttribute am[1];
    am[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    am[0].val.programmaticStreamSerializationAllowed = 1;
    cm.attrs = am; cm.numAttrs = (V[5] & 256) ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cm, vm::vm_main, a));
  }
  if ((mask & 2) && M > 0) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)M); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0; cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));
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


def build(verbose=False, defines=()):
    """defines: compile-time variants, e.g. ("-DVM_DEFER=1",)."""
    from torch.utils.cpp_extension import load_inline
    fl = FLAGS + list(defines)
    tag = hashlib.sha1((CPP + CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vml_moe_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["vml_run"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


def alloc(dev, G):
    z = lambda n, dt=torch.int32: torch.zeros(n, dtype=dt, device=dev)  # noqa: E731
    xrows = RMAX + 8 * JMAX
    assert G <= SLOTS and G <= 136
    return {"xA": z((xrows * KDIM,), torch.bfloat16), "xB": z((xrows * KDIM,), torch.bfloat16),
            "y": z((RMAX * KDIM,), torch.float32), "fix": z((2 * SLOTS * 4 * FIXW,), torch.float32),
            "cnt": z((2 * SLOTS * 4,)), "xready": z((JMAX,)), "doneA": z((JMAX * 16,)), "rmap": z((RMAX,)),
            "err": z((4,)), "gbar": z((2,)), "nomark": z((0,), torch.int64), "G": G, "magG": _mag(G)}


FLAGS_DEFAULT = 1   # bit 0: phase A trailing partial unit first, bit 1: x rows -> L2 at kernel start, bit 2: weights after gather, (bits 3 / 4 / 5 killed: R-vml-4 / R-vml-5), bit 6: consumers self-fill the first NX-1 x slots, x producer warp gathers, bit 7: mma.sync for nt <= 2 segments, bit 8: vm_main as a PDL secondary, bit 9: phase-B stage order by predicted A readiness


def call(mod, k, sc, x, ids, wts, ls=0, marks=None, mask=3, out=None, flags=None, cost=None):
    m = x.shape[0]
    assert m <= MMAX
    flags = FLAGS_DEFAULT if flags is None else flags
    if out is None:
        out = torch.empty((m, KDIM), dtype=torch.bfloat16, device=x.device)
    mod.vml_run([x, ids, wts, k["ws13"], k["ws2"], k["gsc13"], k["gsc2"], k["nw"], sc["xA"], sc["xB"], sc["y"], sc["fix"],
                 sc["cnt"], sc["xready"], sc["doneA"], sc["rmap"], out, sc["err"], sc["nomark"] if marks is None else marks, sc["gbar"]],
                [m, ls, sc["G"], sc["magG"], mask, flags] + list(COST + COST if cost is None else cost))
    return out

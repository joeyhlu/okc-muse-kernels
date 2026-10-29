"""dsm_bench.py: R-dsm-0, split-merge latency L2 vs DSMEM under a saturating weight stream (H200, standalone exe).

Question (notes/memory/gpu_requests.md R-dsm-0): VML's phase-A K-split merge is red.add -> atom.acq_rel counter ->
fence -> last-arriver ld.cg read-back (~4 dependent L2 hops queued behind the weight stream). Is a thread-block-cluster
DSMEM merge far faster under the stream, and does cluster launch keep the stream and the co-resident CTA count?
Abstract model checked by the rtt arm: merge latency ~= dependent L2 hops x per-hop RTT (both measured here).

One kernel, 1 CTA per SM (~218 KB smem), 576 threads: 4 merge warpgroups (q = 0..3, 64 rows x N each, as ship's
4 consumer WGs) + 1 producer warp (cp.async.bulk 6 x 24 KB ring, L2::evict_first as ship, interleaved front over a
1 GiB cold buffer, until the merges are done) + 1 release warp. Pairs = CTAs (2k, 2k+1): cluster neighbours when C > 1.
Every pair-iteration both CTAs spin to the same %globaltimer target (40 us apart, 24 iterations, first dropped), then:
  l2  : ship path verbatim (red.global.add.v4.f32 partials, bar, atom.acq_rel.gpu counter, bar, last arriver
        fence.acq_rel.gpu + __ldcg read-back + zero store + counter reset)
  sa  : partner st.async.shared::cluster.mbarrier::complete_tx v4 into the owner's smem; owner waits its mbarrier
  sc  : partner st.shared::cluster v4, bar, fence.acq_rel.cluster + remote mbarrier.arrive.release.cluster
  lc  : partner st.shared (own smem), bar, fence + remote arrive; owner reads with mapa + ld.shared::cluster
  rtt : per CTA, 16 dependent ld.global.cg (L2 pointer chase, 1 warm hop) and 16 dependent atomicAdd: per-hop RTT
  none: spin only (stream bandwidth reference)
Latency = usable (consumer's %globaltimer after its sum is in registers, WG barrier) - first ready of the pair
(min of both CTAs' ready stamps, same q); also the consumer's own clock64 ready -> usable. Sums are exact small
integers and checked. A 300 us gate kernel precedes every timed launch. Stream BW = bytes / (last stop - first start).

  python3 -u kernels/lab/dsm_bench.py gen OUT.cu
  python3 -u kernels/lab/dsm_bench.py all [--out=box_out/dsm0/run.json]     gen + nvcc + run + verdict (GPU)
  python3 -u kernels/lab/dsm_bench.py summary RUN.json
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
NS_ = [8, 32, 72]
CS = [1, 2, 4, 8]
MODES = {0: "none", 1: "l2", 3: "sa", 4: "sc", 5: "lc", 6: "rtt"}

SRC = r"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
constexpr int NQ = 4, NMT = NQ * 128, BLOCK = NMT + 64;
constexpr int NS = 6, SB = 24576, NMAX = 72, RECV = 256 * NMAX * 4;
constexpr int SMEM = NS * SB + RECV + 512;
constexpr int M = 24;
constexpr long long PERIOD = 40000, WARM = 40000, LEAD = 20000;
constexpr unsigned long long TMO = 20000000ull;

struct P {
  const unsigned char* wbuf; long long nchunk; int stream, mode, C, ns, sb;
  float* fix; int* cnt; unsigned long long* hs;
  long long *rdy, *use, *clk, *rtt; unsigned long long* bw; unsigned* err; unsigned* chase; long long* tick;
};

__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, unsigned r) {
  uint32_t o; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(r)); return o;
}
__device__ __forceinline__ unsigned crank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ void mbar_init(uint32_t b, unsigned n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_arrive_remote(uint32_t b) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, unsigned par) {
  unsigned ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ bool mbar_try_cl(uint32_t b, unsigned par) {
  unsigned ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
template <bool CL> __device__ __forceinline__ void mbar_wait(uint32_t b, unsigned par, unsigned* err) {
  if (CL ? mbar_try_cl(b, par) : mbar_try(b, par)) return;
  const long long t0 = gtime();
  while (!(CL ? mbar_try_cl(b, par) : mbar_try(b, par))) if (gtime() - t0 > (long long)TMO) { atomicOr(err, 1u); return; }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, unsigned bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void red_add_v4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ int atom_acqrel_add(int* p, int v) {
  int o; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o;
}
__device__ __forceinline__ void fence_acq_rel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ float val(int role, int it, int e) { return (float)(((e * 3) + it * 5 + role * 7) & 255); }

template <int N>
__global__ void __launch_bounds__(BLOCK, 1) dsm_kernel(P p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  float* recv = reinterpret_cast<float*>(smem + NS * SB);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NS * SB + RECV);
  uint64_t* empty = full + NS;
  uint64_t* dbar = empty + NS;
  uint64_t* cbar = dbar + NQ;
  volatile int* tag = reinterpret_cast<volatile int*>(cbar + NQ);
  volatile int* stop = tag + NS;
  volatile int* wlast = stop + 1;
  volatile long long* sbase = reinterpret_cast<volatile long long*>(smem + NS * SB + RECV + 384);
  const int tid = threadIdx.x, G = gridDim.x, blk = blockIdx.x, pr = blk >> 1, role = blk & 1;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(su32(&full[s]), 1); mbar_init(su32(&empty[s]), 1); }
    for (int q = 0; q < NQ; ++q) { mbar_init(su32(&dbar[q]), 1); mbar_init(su32(&cbar[q]), 1); }
    *stop = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (p.C > 1) cluster_sync();
  const unsigned prk = p.C > 1 ? (crank() ^ 1u) : 0u;      // pair partner's cluster rank
  if (tid == NMT) {                                        // producer
    if (p.stream) {
      uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
      const long long t0 = gtime();
      long long k = 0; int slot = 0; unsigned par = 0;
      for (;;) {
        if (*stop) break;
        mbar_wait<false>(su32(&empty[slot]), par ^ 1u, p.err);
        tag[slot] = 0;
        mbar_expect_tx(su32(&full[slot]), p.sb);
        bulk_g2s(su32(smem + slot * p.sb), p.wbuf + ((blk + k * G) % p.nchunk) * p.sb, p.sb, su32(&full[slot]), pol);
        ++k;
        if (++slot == p.ns) { slot = 0; par ^= 1u; }
      }
      mbar_wait<false>(su32(&empty[slot]), par ^ 1u, p.err);
      tag[slot] = -1; mbar_arrive(su32(&full[slot]));
      const long long t1 = gtime();
      p.bw[blk * 3] = t0; p.bw[blk * 3 + 1] = t1; p.bw[blk * 3 + 2] = (unsigned long long)k * p.sb;
    } else {
      tag[0] = -1; mbar_arrive(su32(&full[0]));
    }
  } else if (tid == NMT + 32) {                            // ring release
    int slot = 0; unsigned par = 0;
    for (;;) {
      mbar_wait<false>(su32(&full[slot]), par, p.err);
      if (tag[slot] < 0) break;
      mbar_arrive(su32(&empty[slot]));
      if (++slot == p.ns) { slot = 0; par ^= 1u; }
    }
  } else if (tid < NMT) {                                  // merge warpgroups
    const int q = tid >> 7, wt = tid & 127;
    if (tid == 0) {
      long long mn = 1ll << 60, last = gtime();
      for (int c = 0; c < 9;) { const long long t = gtime(); if (t != last) { if (c) mn = min(mn, t - last); last = t; ++c; } }
      p.tick[blk] = mn;
      volatile unsigned long long* hs = p.hs + 2 * pr;
      const long long te = gtime();
      bool ok = true;
      if (role) {
        hs[0] = (unsigned long long)te; __threadfence();
        while (hs[1] == 0) if (gtime() - te > (long long)TMO) { ok = false; break; }
        *sbase = (long long)hs[1];
      } else {
        while (hs[0] == 0) if (gtime() - te > (long long)TMO) { ok = false; break; }
        const long long b = max(te, (long long)hs[0]) + LEAD;
        hs[1] = (unsigned long long)b; __threadfence();
        *sbase = b;
      }
      if (!ok) { atomicOr(p.err, 2u); *sbase = gtime(); }
    }
    bar_sync(5, NMT);
    const long long base = *sbase;
    const int mode = p.mode;
    constexpr int NV = N / 8, QW = 64 * N;
    float d[4 * NV];
    float* fb = p.fix + (size_t)(pr * NQ + q) * QW + wt * 4;
    int* fc = p.cnt + pr * NQ + q;
    const uint32_t rq = su32(recv + q * QW + wt * 4);
    const uint32_t rbar_d = p.C > 1 ? mapa(su32(&dbar[q]), prk) : 0u, rbar_c = p.C > 1 ? mapa(su32(&cbar[q]), prk) : 0u;
    const uint32_t rrq = p.C > 1 ? mapa(rq, prk) : 0u;
    if (mode == 3 && role == 0 && wt == 0) mbar_expect_tx(su32(&dbar[q]), QW * 4);
    for (int it = 0; it < M; ++it) {
#pragma unroll
      for (int nt = 0; nt < NV; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) d[4 * nt + c] = val(role, it, q * QW + nt * 512 + wt * 4 + c);
      const long long tgt = base + WARM + it * PERIOD;
      if (wt == 0) while (gtime() < tgt) { }
      bar_sync(1 + q, 128);
      const long long cr = clock64();
      const size_t ri = ((size_t)blk * M + it) * NQ + q, ui = ((size_t)pr * M + it) * NQ + q;
      if (wt == 0) p.rdy[ri] = gtime();
      float chk = 0.f; bool cons = false;
      if (mode == 1) {
#pragma unroll
        for (int nt = 0; nt < NV; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);
        bar_sync(1 + q, 128);
        if (wt == 0) wlast[q] = (atom_acqrel_add(fc, 1) + 1 == 2);
        bar_sync(1 + q, 128);
        if (wlast[q]) {
          fence_acq_rel();
#pragma unroll
          for (int nt = 0; nt < NV; ++nt) {
            const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
            *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
            chk += (v.x + v.y) + (v.z + v.w);
          }
          if (wt == 0) *fc = 0;
          cons = true;
        }
      } else if (mode >= 3 && mode <= 5) {
        if (role) {
          if (mode == 3) {
#pragma unroll
            for (int nt = 0; nt < NV; ++nt)
              asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                           :: "r"(rrq + nt * 2048), "f"(d[4 * nt]), "f"(d[4 * nt + 1]), "f"(d[4 * nt + 2]), "f"(d[4 * nt + 3]),
                              "r"(rbar_d) : "memory");
          } else {
#pragma unroll
            for (int nt = 0; nt < NV; ++nt) {
              if (mode == 4)
                asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(rrq + nt * 2048), "f"(d[4 * nt]),
                             "f"(d[4 * nt + 1]), "f"(d[4 * nt + 2]), "f"(d[4 * nt + 3]) : "memory");
              else
                asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(rq + nt * 2048), "f"(d[4 * nt]),
                             "f"(d[4 * nt + 1]), "f"(d[4 * nt + 2]), "f"(d[4 * nt + 3]) : "memory");
            }
            bar_sync(1 + q, 128);
            if (wt == 0) { asm volatile("fence.acq_rel.cluster;" ::: "memory"); mbar_arrive_remote(rbar_c); }
          }
        } else {
          mbar_wait<true>(su32(mode == 3 ? &dbar[q] : &cbar[q]), it & 1, p.err);
#pragma unroll
          for (int nt = 0; nt < NV; ++nt) {
            float4 v;
            if (mode == 5)
              asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                           : "r"(rrq + nt * 2048) : "memory");
            else
              asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                           : "r"(rq + nt * 2048) : "memory");
            chk += ((v.x + d[4 * nt]) + (v.y + d[4 * nt + 1])) + ((v.z + d[4 * nt + 2]) + (v.w + d[4 * nt + 3]));
          }
          cons = true;
        }
      } else if (mode == 6 && wt == 0) {
        const unsigned* ch = p.chase + (size_t)(blk * NQ + q) * 512;
        int* ac = reinterpret_cast<int*>(p.chase + (size_t)(G * NQ + blk * NQ + q) * 512);
        unsigned v = __ldcg(ch);
        const long long t0 = gtime();
#pragma unroll 1
        for (int h = 0; h < 16; ++h) v = __ldcg(ch + v * 32);
        const long long t1 = gtime();
        int w = (int)v & 15;
#pragma unroll 1
        for (int h = 0; h < 16; ++h) w = atomicAdd(ac + (w & 15) * 32, 1) & 15;
        const long long t2 = gtime();
        int* ap = reinterpret_cast<int*>(p.chase + (size_t)(G * NQ + (blk ^ 1) * NQ + q) * 512);   // partner's lines
        int x = w;
#pragma unroll 1
        for (int h = 0; h < 8; ++h) { const int o = atomicAdd(ap + (x & 15) * 32, 1); x = __ldcg(ap + (o & 15) * 32); }
        const long long t3 = gtime();
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        const long long t4 = gtime();
        asm volatile("fence.acq_rel.cluster;" ::: "memory");
        const long long t5 = gtime();
        long long* rr = p.rtt + ri * 5;
        rr[0] = t1 - t0; rr[1] = t2 - t1; rr[2] = t3 - t2; rr[3] = t4 - t3; rr[4] = t5 - t4;
        if (v == 0xdeadbeefu && w == 99 && x == 7) atomicOr(p.err, 16u);
      }
      if (cons) {
        bar_sync(1 + q, 128);
        if (wt == 0) { p.use[ui] = gtime(); p.clk[ui] = clock64() - cr; }
        float ex = 0.f;
#pragma unroll
        for (int nt = 0; nt < NV; ++nt)
#pragma unroll
          for (int c = 0; c < 4; ++c) { const int e = q * QW + nt * 512 + wt * 4 + c; ex += val(0, it, e) + val(1, it, e); }
        if (chk != ex) atomicOr(p.err, 4u);
        if (mode == 3 && wt == 0 && it + 1 < M) mbar_expect_tx(su32(&dbar[q]), QW * 4);
      }
    }
    bar_sync(5, NMT);
    if (tid == 0) *stop = 1;
  }
  __syncthreads();
  if (p.C > 1) cluster_sync();
}

// R-dsm-0c (lead): does a fence wait for every CTA's bulk copies on the SM or only its own? 2 CTAs per SM (grid 2G,
// smem sized so exactly 2 fit); per SM the first arriver is Q (streams, no fences), the second P (thread 0 times lone
// fences / an acq_rel ticket atomic every 5 us). qs: Q streams; ps: P streams itself too (own copies, as R-dsm-0b).
constexpr int XM = 160, XSM = 3 * 36864 + 256;
struct PX { const unsigned char* wbuf; long long nchunk; int ns, sb, qs, ps; int* smc; int* done; int* tick;
            long long* fa; long long* fs; long long* ta; unsigned long long* bw; unsigned* err; };
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__global__ void __launch_bounds__(96, 2) xcta_kernel(PX p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + 3 * 36864);
  uint64_t* empty = full + 3;
  volatile int* tag = reinterpret_cast<volatile int*>(empty + 3);
  volatile int* role = tag + 3;
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < 3; ++s) { mbar_init(su32(&full[s]), 1); mbar_init(su32(&empty[s]), 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    *role = atomicAdd(p.smc + smid(), 1);                  // 0 = Q, 1 = P, >= 2 idle
  }
  __syncthreads();
  const int r = *role;
  const unsigned sm = smid();
  volatile int* dn = p.done + sm;
  const bool strm = (r == 0 && p.qs) || (r == 1 && p.ps);
  if (tid == 32 && strm) {                                 // producer (Q, or P when ps)
    uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const long long t0 = gtime();
    long long k = 0; int slot = 0; unsigned par = 0;
    for (;;) {
      if (*dn || gtime() - t0 > 3000000) break;
      mbar_wait<false>(su32(&empty[slot]), par ^ 1u, p.err);
      tag[slot] = 0;
      mbar_expect_tx(su32(&full[slot]), p.sb);
      bulk_g2s(su32(smem + slot * p.sb), p.wbuf + ((blockIdx.x + k * gridDim.x) % p.nchunk) * p.sb, p.sb, su32(&full[slot]), pol);
      ++k;
      if (++slot == p.ns) { slot = 0; par ^= 1u; }
    }
    mbar_wait<false>(su32(&empty[slot]), par ^ 1u, p.err);
    tag[slot] = -1; mbar_arrive(su32(&full[slot]));
    p.bw[blockIdx.x * 3] = t0; p.bw[blockIdx.x * 3 + 1] = gtime(); p.bw[blockIdx.x * 3 + 2] = (unsigned long long)k * p.sb;
  } else if (tid == 64 && strm) {                          // release
    int slot = 0; unsigned par = 0;
    for (;;) {
      mbar_wait<false>(su32(&full[slot]), par, p.err);
      if (tag[slot] < 0) break;
      mbar_arrive(su32(&empty[slot]));
      if (++slot == p.ns) { slot = 0; par ^= 1u; }
    }
  } else if (tid == 0 && r == 1) {                         // P: timed fences
    int* tk = p.tick + sm * 32;
    const long long tb = gtime() + 50000;
    for (int i = 0; i < XM; ++i) {
      while (gtime() < tb + i * 5000ll) { }
      const long long a = gtime();
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      const long long b = gtime();
      asm volatile("fence.sc.gpu;" ::: "memory");
      const long long c = gtime();
      const int o = atom_acqrel_add(tk, 1);
      const long long d = gtime();
      if (o == -12345) atomicOr(p.err, 16u);
      p.fa[sm * XM + i] = b - a; p.fs[sm * XM + i] = c - b; p.ta[sm * XM + i] = d - c;
    }
    *dn = 1; __threadfence();
  }
  __syncthreads();
}

__global__ void gate_kernel(long long ns) { const long long t0 = gtime(); while (gtime() - t0 < ns) { } }

static int argi(int argc, char** argv, const char* k, int d) {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], k)) return atoi(argv[i + 1]);
  return d;
}
static double pct(std::vector<double> v, double f) {
  if (v.empty()) return -1; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(f * v.size()))];
}

template <int N>
static void run_c(int C, std::vector<int> modes, std::vector<int> streams, int ns, int sb, int G, int reps, double clk_ghz,
                  const P& p0, size_t pairs_max) {
  auto kern = dsm_kernel<N>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  {
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cf = {}; cf.blockDim = dim3(BLOCK); cf.dynamicSmemBytes = SMEM;
    cf.gridDim = dim3(G / C * C); cf.attrs = at; cf.numAttrs = 1;
    int ncl = 0; CK(cudaOccupancyMaxActiveClusters(&ncl, kern, &cf));
    int grid = std::min(ncl * C, G / 2 * 2); grid = grid / std::max(C, 2) * std::max(C, 2);
    cf.gridDim = dim3(grid); cf.numAttrs = C > 1 ? 1 : 0;
    const int pairs = grid / 2;
    for (int mode : modes) for (int stream : streams) {
      std::vector<double> lq, lp, lc, r0, r1, r2, r3, r4, bws; unsigned errs = 0; long long tick = 1ll << 60;
      for (int r = 0; r < reps; ++r) {
        P p = p0; p.mode = mode; p.stream = stream; p.C = C; p.ns = ns; p.sb = sb;
        p.nchunk = p0.nchunk * SB / sb;                    // p0.nchunk counts SB-byte chunks of the 1 GiB buffer
        CK(cudaMemset(p.hs, 0, pairs_max * 16)); CK(cudaMemset(p.err, 0, 4)); CK(cudaMemset(p.bw, 0, (size_t)G * 24));
        CK(cudaMemset(p.rdy, 0xFF, (size_t)G * M * NQ * 8)); CK(cudaMemset(p.use, 0xFF, pairs_max * M * NQ * 8));
        gate_kernel<<<1, 32>>>(300000); CK(cudaGetLastError());
        CK(cudaLaunchKernelEx(&cf, kern, p)); CK(cudaDeviceSynchronize());
        std::vector<long long> rdy((size_t)grid * M * NQ), use((size_t)pairs * M * NQ), clk(use.size()), rtt(rdy.size() * 5), tk(grid);
        std::vector<unsigned long long> bw((size_t)grid * 3); unsigned e = 0;
        CK(cudaMemcpy(rdy.data(), p.rdy, rdy.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(use.data(), p.use, use.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(clk.data(), p.clk, clk.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(rtt.data(), p.rtt, rtt.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(tk.data(), p.tick, tk.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bw.data(), p.bw, bw.size() * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&e, p.err, 4, cudaMemcpyDeviceToHost));
        errs |= e;
        for (int b = 0; b < grid; ++b) tick = std::min(tick, tk[b]);
        if (stream) {
          unsigned long long t0 = ~0ull, t1 = 0, by = 0;
          for (int b = 0; b < grid; ++b) { t0 = std::min(t0, bw[b * 3]); t1 = std::max(t1, bw[b * 3 + 1]); by += bw[b * 3 + 2]; }
          if (t1 > t0) bws.push_back((double)by / (double)(t1 - t0) / 1e3);   // TB/s
        }
        for (int pr = 0; pr < pairs; ++pr) for (int it = 1; it < M; ++it) {
          long long umax = -1, rmin = 1ll << 62; bool miss = false;
          for (int q = 0; q < NQ; ++q) {
            const size_t ui = ((size_t)pr * M + it) * NQ + q;
            const long long ra = rdy[((size_t)(2 * pr) * M + it) * NQ + q], rb = rdy[((size_t)(2 * pr + 1) * M + it) * NQ + q];
            const long long rq = std::min(ra, rb);
            rmin = std::min(rmin, rq);
            if (mode == 6) {
              for (int s = 0; s < 2; ++s) {
                const long long* rr = &rtt[(((size_t)(2 * pr + s) * M + it) * NQ + q) * 5];
                r0.push_back(rr[0] / 16.0); r1.push_back(rr[1] / 16.0); r2.push_back(rr[2] / 8.0);
                r3.push_back((double)rr[3]); r4.push_back((double)rr[4]);
              }
              continue;
            }
            if (mode == 0) continue;
            if (use[ui] < 0) { miss = true; continue; }
            lq.push_back((double)(use[ui] - rq)); lc.push_back(clk[ui] / clk_ghz);
            umax = std::max(umax, use[ui]);
          }
          if (miss) errs |= 8u;
          if (umax >= 0 && !miss) lp.push_back((double)(umax - rmin));
        }
      }
      double bwm = bws.empty() ? -1 : pct(bws, 0.5);
      printf("{\"N\":%d,\"C\":%d,\"ns\":%d,\"sb\":%d,\"clusters\":%d,\"coresident\":%d,\"grid\":%d,\"mode\":%d,\"stream\":%d,"
             "\"TBps\":%.3f,\"q_p50\":%.0f,\"q_p95\":%.0f,\"pair_p50\":%.0f,\"pair_p95\":%.0f,\"clk_p50\":%.0f,\"clk_p95\":%.0f,"
             "\"ld_p50\":%.0f,\"ld_p95\":%.0f,\"atom_p50\":%.0f,\"atom_p95\":%.0f,\"ar_p50\":%.0f,\"ar_p95\":%.0f,"
             "\"fg_p50\":%.0f,\"fg_p95\":%.0f,\"fc_p50\":%.0f,\"fc_p95\":%.0f,\"n\":%zu,\"tick_ns\":%lld,\"err\":%u}\n",
             N, C, ns, sb, ncl, ncl * C, grid, mode, stream, bwm, pct(lq, .5), pct(lq, .95), pct(lp, .5), pct(lp, .95),
             pct(lc, .5), pct(lc, .95), pct(r0, .5), pct(r0, .95), pct(r1, .5), pct(r1, .95), pct(r2, .5), pct(r2, .95),
             pct(r3, .5), pct(r3, .95), pct(r4, .5), pct(r4, .95), lq.size() + r0.size(), tick, errs);
      fflush(stdout);
    }
  }
}

template <int N>
static void run_n(int G, int reps, double ghz, const P& p, size_t pm) {
  for (int C : {1, 2, 4, 8})
    run_c<N>(C, C == 1 ? std::vector<int>{0, 1, 6} : std::vector<int>{0, 1, 3, 4, 5, 6}, {1, 0}, NS, SB, G, reps, ghz, p, pm);
}

// R-dsm-0b: background in-flight bytes per SM (ring slots x slot bytes) vs dependent-latency and stream bandwidth
template <int N>
static void sweep_n(int G, int reps, double ghz, const P& p, size_t pm) {
  const int pts[][2] = {{1, 16384}, {1, 32768}, {1, 49152}, {2, 36864}, {3, 36864}, {4, 36864}};
  for (auto& pt : pts) {
    run_c<N>(1, {0, 1, 6}, {1}, pt[0], pt[1], G, reps, ghz, p, pm);
    run_c<N>(2, {3, 4}, {1}, pt[0], pt[1], G, reps, ghz, p, pm);
  }
  run_c<N>(1, {1, 6}, {0}, 3, 36864, G, reps, ghz, p, pm);
}

static void run_xcta(int G, int reps, unsigned char* wb, long long WB) {
  CK(cudaFuncSetAttribute(xcta_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, XSM));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, xcta_kernel, 96, XSM));
  const int NSM = 256;
  PX p = {}; p.wbuf = wb;
  CK(cudaMalloc(&p.smc, NSM * 4)); CK(cudaMalloc(&p.done, NSM * 4)); CK(cudaMalloc(&p.tick, NSM * 128)); CK(cudaMemset(p.tick, 0, NSM * 128));
  CK(cudaMalloc(&p.fa, NSM * XM * 8)); CK(cudaMalloc(&p.fs, NSM * XM * 8)); CK(cudaMalloc(&p.ta, NSM * XM * 8));
  CK(cudaMalloc(&p.bw, (size_t)2 * G * 24)); CK(cudaMalloc(&p.err, 4));
  const int pts[][2] = {{1, 16384}, {2, 36864}, {3, 36864}};
  const int cs[][2] = {{1, 0}, {0, 1}, {1, 1}};
  for (int ci = -1; ci < 9; ++ci) {
    const int ns = ci < 0 ? 1 : pts[ci / 3][0], sb = ci < 0 ? 16384 : pts[ci / 3][1];
    const int qs = ci < 0 ? 0 : cs[ci % 3][0], ps = ci < 0 ? 0 : cs[ci % 3][1];
    std::vector<double> fa, fs, ta, bws; unsigned errs = 0; int pairs = 0;
    for (int r = 0; r < reps; ++r) {
      p.ns = ns; p.sb = sb; p.qs = qs; p.ps = ps; p.nchunk = WB / sb;
      CK(cudaMemset(p.smc, 0, NSM * 4)); CK(cudaMemset(p.done, 0, NSM * 4)); CK(cudaMemset(p.err, 0, 4));
      CK(cudaMemset(p.bw, 0, (size_t)2 * G * 24));
      gate_kernel<<<1, 32>>>(300000); CK(cudaGetLastError());
      xcta_kernel<<<2 * G, 96, XSM>>>(p); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      std::vector<int> smc(NSM); std::vector<long long> a(NSM * XM), b(NSM * XM), c(NSM * XM);
      std::vector<unsigned long long> bw((size_t)2 * G * 3); unsigned e = 0;
      CK(cudaMemcpy(smc.data(), p.smc, NSM * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(a.data(), p.fa, a.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), p.fs, b.size() * 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(c.data(), p.ta, c.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(bw.data(), p.bw, bw.size() * 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&e, p.err, 4, cudaMemcpyDeviceToHost)); errs |= e;
      pairs = 0;
      for (int s = 0; s < NSM; ++s) {
        if (smc[s] > 2) errs |= 32u;
        if (smc[s] != 2) continue;
        ++pairs;
        for (int i = 5; i < XM; ++i) { fa.push_back(a[s * XM + i]); fs.push_back(b[s * XM + i]); ta.push_back(c[s * XM + i]); }
      }
      unsigned long long t0 = ~0ull, t1 = 0, by = 0;
      for (int k = 0; k < 2 * G; ++k) if (bw[k * 3 + 2]) { t0 = std::min(t0, bw[k * 3]); t1 = std::max(t1, bw[k * 3 + 1]); by += bw[k * 3 + 2]; }
      if (t1 > t0) bws.push_back((double)by / (double)(t1 - t0) / 1e3);
    }
    printf("{\"x\":1,\"occ\":%d,\"sm_pairs\":%d,\"ns\":%d,\"sb\":%d,\"qs\":%d,\"ps\":%d,\"TBps\":%.3f,\"fa_p50\":%.0f,\"fa_p95\":%.0f,"
           "\"fs_p50\":%.0f,\"fs_p95\":%.0f,\"ta_p50\":%.0f,\"ta_p95\":%.0f,\"n\":%zu,\"err\":%u}\n", occ, pairs, ns, sb, qs, ps,
           bws.empty() ? -1.0 : pct(bws, .5), pct(fa, .5), pct(fa, .95), pct(fs, .5), pct(fs, .95), pct(ta, .5), pct(ta, .95),
           fa.size(), errs);
    fflush(stdout);
  }
}

int main(int argc, char** argv) {
  const int reps = argi(argc, argv, "--reps", 3);
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  const int G = pr.multiProcessorCount;
  int clk_khz = 0; CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  const long long WB = 1ll << 30;
  const size_t pairs_max = G / 2 + 1;
  P p = {};
  unsigned char* wb; CK(cudaMalloc(&wb, WB)); CK(cudaMemset(wb, 1, WB));
  p.wbuf = wb; p.nchunk = WB / SB;
  CK(cudaMalloc(&p.fix, pairs_max * NQ * 64 * NMAX * 4)); CK(cudaMemset(p.fix, 0, pairs_max * NQ * 64 * NMAX * 4));
  CK(cudaMalloc(&p.cnt, pairs_max * NQ * 4)); CK(cudaMemset(p.cnt, 0, pairs_max * NQ * 4));
  CK(cudaMalloc(&p.hs, pairs_max * 16));
  CK(cudaMalloc(&p.rdy, (size_t)G * M * NQ * 8)); CK(cudaMalloc(&p.use, pairs_max * M * NQ * 8));
  CK(cudaMalloc(&p.clk, pairs_max * M * NQ * 8)); CK(cudaMalloc(&p.rtt, (size_t)G * M * NQ * 40));
  CK(cudaMalloc(&p.bw, (size_t)G * 24)); CK(cudaMalloc(&p.err, 4)); CK(cudaMalloc(&p.tick, (size_t)G * 8));
  const size_t nch = (size_t)2 * G * NQ * 512;
  std::vector<unsigned> ch(nch, 0);
  for (size_t g = 0; g < (size_t)G * NQ; ++g) for (int l = 0; l < 16; ++l) ch[g * 512 + l * 32] = (5 * l + 1) % 16;
  CK(cudaMalloc(&p.chase, nch * 4)); CK(cudaMemcpy(p.chase, ch.data(), nch * 4, cudaMemcpyHostToDevice));
  printf("{\"dev\":\"%s\",\"G\":%d,\"clk_khz\":%d,\"smem\":%d,\"block\":%d}\n", pr.name, G, clk_khz, SMEM, BLOCK);
  const double ghz = clk_khz / 1e6;
  if (argi(argc, argv, "--xcta", 0)) {
    run_xcta(G, reps, wb, WB);
    return 0;
  }
  if (argi(argc, argv, "--sweep", 0)) {
    sweep_n<32>(G, reps, ghz, p, pairs_max);
    sweep_n<72>(G, reps, ghz, p, pairs_max);
    return 0;
  }
  run_n<8>(G, reps, ghz, p, pairs_max);
  run_n<32>(G, reps, ghz, p, pairs_max);
  run_n<72>(G, reps, ghz, p, pairs_max);
  return 0;
}
"""


def gen():
    return SRC


def find_nvcc():
    nv = shutil.which("nvcc")
    if nv:
        return nv
    for c in ["/usr/local/cuda/bin/nvcc"]:
        if pathlib.Path(c).exists():
            return c
    from torch.utils.cpp_extension import CUDA_HOME
    return str(pathlib.Path(CUDA_HOME) / "bin/nvcc")


def verdict(rows):
    """PASS iff some C > 1 has, under the stream, for every N: best DSMEM q_p95 <= 1000 ns and L2 q_p50 >= 3 x DSMEM
    q_p50 (same C), co-resident >= 128, stream BW (mode none) within 2 % of C = 1, err 0 everywhere."""
    out = []
    key = {(r["N"], r["C"], r["mode"], r["stream"]): r for r in rows if "mode" in r}
    errs = [r for r in rows if r.get("err")]
    bw1 = key.get((8, 1, 0, 1), {}).get("TBps", -1)
    out.append(f"BW C1 {bw1:.3f} TB/s (need >= 4.3)" + ("" if bw1 >= 4.3 else "  [stream not saturating]"))
    ok_any = False
    for C in CS[1:]:
        r0 = key.get((8, C, 0, 1))
        if not r0:
            continue
        bwc = r0["TBps"]
        co = r0["coresident"]
        conds = [co >= 128, bwc >= 0.98 * bw1]
        line = [f"C{C}: coresident {co}, BW {bwc:.3f} ({(bwc / bw1 - 1) * 100:+.1f} %)"]
        for n in NS_:
            l2 = key.get((n, C, 1, 1), {})
            ds = [key.get((n, C, m, 1), {}) for m in (3, 4, 5)]
            ds = [d for d in ds if d.get("q_p95", -1) > 0]
            if not ds or l2.get("q_p50", -1) <= 0:
                conds.append(False)
                line.append(f"N{n} missing")
                continue
            b = min(ds, key=lambda d: d["q_p95"])
            ratio = l2["q_p50"] / max(b["q_p50"], 1)
            conds += [b["q_p95"] <= 1000, ratio >= 3]
            line.append(f"N{n} L2 {l2['q_p50']:.0f}/{l2['q_p95']:.0f} vs {MODES[b['mode']]} {b['q_p50']:.0f}/{b['q_p95']:.0f} "
                        f"ns (x{ratio:.1f})")
        ok = all(conds) and not errs
        ok_any |= ok
        out.append(("PASS " if ok else "fail ") + "; ".join(line))
    out.append(f"errors: {len(errs)} rows" if errs else "errors: none")
    out.append("VERDICT " + ("PASS" if ok_any and bw1 >= 4.3 else "KILL"))
    return out


def summarize(rows):
    for r in rows:
        if r.get("x"):
            who = {(0, 0): "idle", (1, 0): "Q streams (other CTA)", (0, 1): "P streams (own)", (1, 1): "both"}[(r["qs"], r["ps"])]
            print(f"xcta ring {r['ns']}x{r['sb'] // 1024}K {who:22s} BW {r['TBps']:6.3f}  fence.acq_rel {r['fa_p50']:.0f}/{r['fa_p95']:.0f}"
                  f"  fence.sc {r['fs_p50']:.0f}/{r['fs_p95']:.0f}  atom.acq_rel {r['ta_p50']:.0f}/{r['ta_p95']:.0f} ns"
                  f"  occ {r['occ']} sm_pairs {r['sm_pairs']} n {r['n']} err {r['err']}")
            continue
        if "mode" not in r:
            print(json.dumps(r))
            continue
        m = MODES[r["mode"]]
        s = (f"N{r['N']:>2} C{r['C']} co{r['coresident']:>3} g{r['grid']:>3} {m:>4} st{r['stream']} BW {r['TBps']:6.3f}")
        if "ns" in r:
            s = f"ring {r['ns']}x{r['sb'] // 1024}K={r['ns'] * r['sb'] // 1024:>3}K " + s
        if r["mode"] == 6:
            s += f"  ld {r['ld_p50']:.0f}/{r['ld_p95']:.0f}  atom {r['atom_p50']:.0f}/{r['atom_p95']:.0f} ns/hop"
            if "ar_p50" in r:
                s += (f"  atom+rb {r['ar_p50']:.0f}/{r['ar_p95']:.0f}  fence.gpu {r['fg_p50']:.0f}/{r['fg_p95']:.0f}"
                      f"  fence.cl {r['fc_p50']:.0f}/{r['fc_p95']:.0f}")
        elif r["mode"]:
            s += (f"  q {r['q_p50']:.0f}/{r['q_p95']:.0f}  pair {r['pair_p50']:.0f}/{r['pair_p95']:.0f}"
                  f"  clk {r['clk_p50']:.0f}/{r['clk_p95']:.0f} ns")
        s += f"  n{r['n']} tick{r['tick_ns']} err{r['err']}"
        print(s)
    if any(r.get("x") for r in rows) or any("ns" in r and r["ns"] * r["sb"] != 6 * 24576 for r in rows if "mode" in r):
        return   # R-dsm-0b sweep: no R-dsm-0 verdict
    for line in verdict(rows):
        print(line)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "gen":
        pathlib.Path(argv[2]).write_text(gen())
        return
    if cmd == "summary":
        summarize(json.loads(pathlib.Path(argv[2]).read_text()))
        return
    out, extra = "box_out/dsm0/run.json", []
    for a in argv[2:]:
        if a.startswith("--out="):
            out = a[6:]
        if a == "--sweep":
            extra = ["--sweep", "1"]
        if a == "--xcta":
            extra = ["--xcta", "1"]
    tag = pathlib.Path(out).stem
    wd = ROOT / "box_out/dsm0"
    wd.mkdir(parents=True, exist_ok=True)
    cu, exe = wd / "dsm.cu", wd / "dsm_bench"
    cu.write_text(gen())
    t0 = time.time()
    r = subprocess.run([find_nvcc(), "-O3", "-arch=sm_90a", "-std=c++17", "-Xptxas=-v", "-o", str(exe), str(cu)],
                       capture_output=True, text=True)
    (wd / "ptxas.log").write_text(r.stdout + r.stderr)
    print(f"[nvcc] rc {r.returncode} {time.time() - t0:.0f} s")
    if r.returncode:
        print((r.stdout + r.stderr)[-1500:])
        sys.exit(1)
    p = subprocess.run([str(exe), "--reps", "3", *extra], capture_output=True, text=True, timeout=150)
    (wd / f"{tag}.log").write_text(p.stdout + p.stderr)
    rows = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
    (ROOT / out).write_text(json.dumps(rows, indent=0))
    print(f"[run] rc {p.returncode} rows {len(rows)}" + (f" stderr {p.stderr[-400:]}" if p.returncode else ""))
    summarize(rows)


if __name__ == "__main__":
    main(sys.argv)

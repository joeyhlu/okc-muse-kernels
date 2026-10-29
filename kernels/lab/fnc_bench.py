"""fnc_bench.py: lever FNC (R-fnc-1), which sync ops pay the SM-wide in-flight tax (R-dsm-0c), and what the exit
flush of a fence-free predecessor (sbb1-like cluster grid) costs a dependent that prefetches pre-wait (hop_d8s model).

Part X (xcta-style, 2 CTAs / SM, grid 2G, smem sized for exactly 2): per SM the first arriver Q streams, the second P
(thread 0) times one sync op at a time, 1.5 us apart so Q's ring refills. Q modes: idle / bulk ring 1x16K, 2x36K,
3x36K (cp.async.bulk, evict_first, 1 GiB cold) / LDG 12K, 36K in flight (96 threads x 8 or 24 ld.global.nc.v4
into registers, as GD's XP-6 pre-wait register PF). P ops (SASS, nvcc 13.4 sm_90a):
  sc   fence.sc.gpu        (= __threadfence: MEMBAR.SC.GPU + ERRBAR + CCTL.IVALL; ship writer AND merger side)
  ar   fence.acq_rel.gpu   (MEMBAR.ALL.GPU + ERRBAR + CCTL.IVALL; = fence.acq_rel.cluster, barrier.cluster.arrive)
  acq  fence.acquire.gpu   (CCTL.IVALL only)       <- merger-side candidate
  rel  fence.release.gpu   (MEMBAR.ALL.GPU + ERRBAR, no CCTL)
  cta  fence.acq_rel.cta   (MEMBAR.ALL.CTA)
  atr  atom.relaxed.gpu    (ATOMG only; RTT reference)
  ata  atom.acquire.gpu    (ATOMG + CCTL.IVALL) <- ticket-side candidate
  lda  ld.acquire.gpu      (LDG.STRONG.GPU + CCTL.IVALL)
Part H (exit flush): P spinner (132 x 256, trigger at entry) -> A (sbb1 verbatim exchange: T*16 CTAs x 64, 16-CTA
cluster, relaxed arrive + st.async + mbarrier, NO fence) -> B (264 x 256, occ 2, PDL). B pre-wait PF of pfb bytes per CTA:
pf0 none / ent bulk at entry / aft bulk right after griddepcontrol.wait / reg LDG registers at entry. Marks per replay:
hop = min B release - A last end; rdy = max B data-ready - A last end; lead = A last end - median B entry.

  python3 -u kernels/lab/fnc_bench.py gen OUT.cu
  python3 -u kernels/lab/fnc_bench.py all [--out=box_out/fnc0/run.json]
  python3 -u kernels/lab/fnc_bench.py summary RUN.json
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OPS = ["sc", "ar", "acq", "rel", "cta", "atr", "ata", "lda"]

SRC = r"""
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("{\"cuda_error\":\"%s\",\"line\":%d}\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ void mbar_init(uint32_t b, unsigned n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t b, unsigned par) {
  unsigned d = 0;
  while (!d) asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                          : "=r"(d) : "r"(b), "r"(par) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, unsigned bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 ldnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void spin(long long ns) { const long long t = gt(); while (gt() - t < ns) { } }
__device__ __forceinline__ unsigned use(unsigned o) { unsigned z; asm volatile("add.u32 %0, %1, 0;" : "=r"(z) : "r"(o)); return z; }

// ------------------------------------------------------------------ Part X
constexpr int XM = 100, NOP = 8, XSM = 3 * 36864 + 256;
struct PX { const unsigned char* wb; long long nchunk; int qm, ns, sb, nld; int* smc; int* done; unsigned* tick;
            long long* t; unsigned long long* bw; unsigned* err; };
template <int NL> __device__ __forceinline__ void ldg_stream(const PX& p, volatile int* dn, unsigned long long* bwslot) {
  const int tid = threadIdx.x;
  const long long t0 = gt();
  unsigned acc = 0; long long k = 0;
  const long long CH = 96 * NL * 16;
  for (;;) {
    if (*dn || gt() - t0 > 3000000) break;
    const unsigned char* base = p.wb + ((blockIdx.x + k * gridDim.x) % (((1ll << 30) / CH))) * CH;
    uint4 v[NL];
#pragma unroll
    for (int j = 0; j < NL; ++j) v[j] = ldnc(base + ((size_t)j * 96 + tid) * 16);
#pragma unroll
    for (int j = 0; j < NL; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
    ++k;
  }
  if (acc == 0x9e3779b9u) atomicOr(p.err, 64u);
  if (tid == 0) { bwslot[0] = t0; bwslot[1] = gt(); bwslot[2] = (unsigned long long)k * CH; }
}
__global__ void __launch_bounds__(96, 2) xk(PX p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + 3 * 36864);
  uint64_t* empty = full + 3;
  volatile int* tag = reinterpret_cast<volatile int*>(empty + 3);
  volatile int* role = tag + 3;
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < 3; ++s) { mbar_init(su32(&full[s]), 1); mbar_init(su32(&empty[s]), 1); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    *role = atomicAdd(p.smc + smid(), 1);
  }
  __syncthreads();
  const int r = *role;
  const unsigned sm = smid();
  volatile int* dn = p.done + sm;
  unsigned long long* bws = p.bw + blockIdx.x * 3;
  if (r == 0 && p.qm == 2) {                                  // Q: LDG register stream (all 96 threads)
    if (p.nld == 8) ldg_stream<8>(p, dn, bws); else ldg_stream<24>(p, dn, bws);
  } else if (r == 0 && p.qm == 1 && tid == 32) {              // Q: bulk ring producer
    uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    const long long t0 = gt();
    long long k = 0; int slot = 0; unsigned par = 0;
    for (;;) {
      if (*dn || gt() - t0 > 3000000) break;
      mbar_wait(su32(&empty[slot]), par ^ 1u);
      tag[slot] = 0;
      mbar_expect_tx(su32(&full[slot]), p.sb);
      bulk_g2s(su32(smem + slot * p.sb), p.wb + ((blockIdx.x + k * gridDim.x) % p.nchunk) * p.sb, p.sb, su32(&full[slot]), pol);
      ++k;
      if (++slot == p.ns) { slot = 0; par ^= 1u; }
    }
    mbar_wait(su32(&empty[slot]), par ^ 1u);
    tag[slot] = -1; mbar_arrive(su32(&full[slot]));
    bws[0] = t0; bws[1] = gt(); bws[2] = (unsigned long long)k * p.sb;
  } else if (r == 0 && p.qm == 1 && tid == 64) {              // Q: release
    int slot = 0; unsigned par = 0;
    for (;;) {
      mbar_wait(su32(&full[slot]), par);
      if (tag[slot] < 0) break;
      mbar_arrive(su32(&empty[slot]));
      if (++slot == p.ns) { slot = 0; par ^= 1u; }
    }
  } else if (r == 1 && tid == 0) {                            // P: timed sync ops
    unsigned* tk = p.tick + sm * 32;
    const long long tb = gt() + 50000;
    unsigned chk = 0;
    for (int i = 0; i < XM; ++i) {
      while (gt() < tb + i * 25000ll) { }
      long long* o = p.t + ((size_t)sm * XM + i) * NOP;
      long long a, b; unsigned v;
      a = gt(); asm volatile("fence.sc.gpu;" ::: "memory"); b = gt(); o[0] = b - a; spin(1500);
      a = gt(); asm volatile("fence.acq_rel.gpu;" ::: "memory"); b = gt(); o[1] = b - a; spin(1500);
      a = gt(); asm volatile("fence.acquire.gpu;" ::: "memory"); b = gt(); o[2] = b - a; spin(1500);
      a = gt(); asm volatile("fence.release.gpu;" ::: "memory"); b = gt(); o[3] = b - a; spin(1500);
      a = gt(); asm volatile("fence.acq_rel.cta;" ::: "memory"); b = gt(); o[4] = b - a; spin(1500);
      a = gt(); asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], 1;" : "=r"(v) : "l"(tk) : "memory"); v = use(v); b = gt(); o[5] = b - a; chk += v; spin(1500);
      a = gt(); asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], 1;" : "=r"(v) : "l"(tk) : "memory"); v = use(v); b = gt(); o[6] = b - a; chk += v; spin(1500);
      a = gt(); asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(tk) : "memory"); v = use(v); b = gt(); o[7] = b - a; chk += v; spin(1500);
    }
    if (chk == 0xdeadbeefu) atomicOr(p.err, 16u);
    *dn = 1; __threadfence();
  }
  __syncthreads();
}
__global__ void gate_kernel(long long ns) { const long long t0 = gt(); while (gt() - t0 < ns) { } }

static double pct(std::vector<double> v, double f) {
  if (v.empty()) return -1; std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(f * v.size()))];
}
static void run_x(int G, int reps, unsigned char* wb, long long WB) {
  CK(cudaFuncSetAttribute(xk, cudaFuncAttributeMaxDynamicSharedMemorySize, XSM));
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, xk, 96, XSM));
  const int NSM = 256;
  PX p = {}; p.wb = wb;
  CK(cudaMalloc(&p.smc, NSM * 4)); CK(cudaMalloc(&p.done, NSM * 4)); CK(cudaMalloc(&p.tick, NSM * 128)); CK(cudaMemset(p.tick, 0, NSM * 128));
  CK(cudaMalloc(&p.t, (size_t)NSM * XM * NOP * 8)); CK(cudaMalloc(&p.bw, (size_t)2 * G * 24)); CK(cudaMalloc(&p.err, 4));
  const int modes[][4] = {{0, 1, 16384, 0}, {1, 1, 16384, 0}, {1, 2, 36864, 0}, {1, 3, 36864, 0}, {2, 1, 16384, 8}, {2, 1, 16384, 24}};
  for (int mi = 0; mi < 6; ++mi) {
    p.qm = modes[mi][0]; p.ns = modes[mi][1]; p.sb = modes[mi][2]; p.nld = modes[mi][3]; p.nchunk = WB / p.sb;
    std::vector<double> v[NOP], bws; unsigned errs = 0; int pairs = 0;
    for (int r = 0; r < reps; ++r) {
      CK(cudaMemset(p.smc, 0, NSM * 4)); CK(cudaMemset(p.done, 0, NSM * 4)); CK(cudaMemset(p.err, 0, 4));
      CK(cudaMemset(p.bw, 0, (size_t)2 * G * 24));
      gate_kernel<<<1, 32>>>(300000); CK(cudaGetLastError());
      xk<<<2 * G, 96, XSM>>>(p); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      std::vector<int> smc(NSM); std::vector<long long> t((size_t)NSM * XM * NOP); std::vector<unsigned long long> bw((size_t)2 * G * 3);
      unsigned e = 0;
      CK(cudaMemcpy(smc.data(), p.smc, NSM * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(t.data(), p.t, t.size() * 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(bw.data(), p.bw, bw.size() * 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&e, p.err, 4, cudaMemcpyDeviceToHost)); errs |= e;
      pairs = 0;
      for (int s = 0; s < NSM; ++s) {
        if (smc[s] > 2) errs |= 32u;
        if (smc[s] != 2) continue;
        ++pairs;
        for (int i = 5; i < XM; ++i) for (int k = 0; k < NOP; ++k) v[k].push_back((double)t[((size_t)s * XM + i) * NOP + k]);
      }
      unsigned long long t0 = ~0ull, t1 = 0, by = 0;
      for (int k = 0; k < 2 * G; ++k) if (bw[k * 3 + 2]) { t0 = std::min(t0, bw[k * 3]); t1 = std::max(t1, bw[k * 3 + 1]); by += bw[k * 3 + 2]; }
      if (t1 > t0) bws.push_back((double)by / (double)(t1 - t0) / 1e3);
    }
    printf("{\"part\":\"x\",\"qm\":%d,\"ns\":%d,\"sb\":%d,\"nld\":%d,\"occ\":%d,\"pairs\":%d,\"TBps\":%.3f,\"err\":%u,\"n\":%zu",
           p.qm, p.ns, p.sb, p.nld, occ, pairs, bws.empty() ? -1.0 : pct(bws, .5), errs, v[0].size());
    const char* nm[NOP] = {"sc", "ar", "acq", "rel", "cta", "atr", "ata", "lda"};
    for (int k = 0; k < NOP; ++k) printf(",\"%s_p50\":%.0f,\"%s_p95\":%.0f", nm[k], pct(v[k], .5), nm[k], pct(v[k], .95));
    printf("}\n"); fflush(stdout);
  }
}

// ------------------------------------------------------------------ Part H
#define NT 64
#define CS 16
__global__ void p_kernel(float* buf, long long spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  spin(spin_ns);
  buf[blockIdx.x * blockDim.x + threadIdx.x] += 1.f;
}
__global__ void __launch_bounds__(NT) a_kernel(const float* __restrict__ in, float* __restrict__ out, long long* marks) {
  __shared__ __align__(16) float partials[CS][4];
  __shared__ float warp_red[NT / 32];
  __shared__ __align__(8) uint64_t bar;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t rank = 0;
  const uint32_t bar_a = su32(&bar);
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar_a));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_a), "r"((CS - 1) * 16));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const size_t i0 = ((size_t)blockIdx.x * NT + tid) * 8;
  const float4 x0 = *reinterpret_cast<const float4*>(in + i0), x1 = *reinterpret_cast<const float4*>(in + i0 + 4);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  float s = x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w + x1.x * x1.x + x1.y * x1.y + x1.z * x1.z + x1.w * x1.w;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) warp_red[warp] = s;
  __syncthreads();
  const float mine = warp_red[0] + warp_red[1];
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (tid < CS) {
    if (tid == (int)rank) {
      *reinterpret_cast<float4*>(&partials[rank][0]) = make_float4(mine, mine, mine, mine);
    } else {
      uint32_t dst, rbar;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dst) : "r"(su32(&partials[rank][0])), "r"(tid));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(bar_a), "r"(tid));
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                   ::"r"(dst), "f"(mine), "f"(mine), "f"(mine), "f"(mine), "r"(rbar) : "memory");
    }
  }
  {
    uint32_t done = 0;
    while (!done)
      asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                   " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bar_a) : "memory");
  }
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int q = 0; q < CS; ++q) tot += partials[q][0];
  const float r = rsqrtf(tot * (1.f / 8192.f) + 1e-6f);
  *reinterpret_cast<float4*>(out + i0) = make_float4(x0.x * r, x0.y * r, x0.z * r, x0.w * r);
  *reinterpret_cast<float4*>(out + i0 + 4) = make_float4(x1.x * r, x1.y * r, x1.z * r, x1.w * r);
  __syncthreads();
  if (tid == 0) marks[blockIdx.x] = gt();
}
struct BP { const float* aout; int na; const unsigned char* wb; long long nchunk; int pfb, pfm, rot; float* o; long long* m; };
constexpr int BSM = 65536 + 64;
__global__ void __launch_bounds__(256, 2) b_kernel(BP p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem + 65536);
  const int tid = threadIdx.x;
  const long long t0 = gt();
  const unsigned char* src = p.wb + ((blockIdx.x + (long long)p.rot * gridDim.x) % p.nchunk) * 65536;
  uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  if (tid == 0) { mbar_init(su32(bar), 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  if (p.pfm == 1 && tid == 0) { mbar_expect_tx(su32(bar), p.pfb); bulk_g2s(su32(smem), src, p.pfb, su32(bar), pol); }
  uint4 v[16];
  const int nld = p.pfb / 4096;                        // 256 threads x 16 B per load step
  if (p.pfm == 3) {
#pragma unroll
    for (int j = 0; j < 16; ++j) if (j < nld) v[j] = ldnc(src + ((size_t)j * 256 + tid) * 16);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const long long t1 = gt();
  if (p.pfm == 2 && tid == 0) { mbar_expect_tx(su32(bar), p.pfb); bulk_g2s(su32(smem), src, p.pfb, su32(bar), pol); }
  unsigned x = 0;
  if (p.pfm == 1 || p.pfm == 2) { mbar_wait(su32(bar), 0); x = reinterpret_cast<const unsigned*>(smem)[tid]; }
  if (p.pfm == 3) {
#pragma unroll
    for (int j = 0; j < 16; ++j) if (j < nld) x ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  const int gi = blockIdx.x * blockDim.x + tid;
  p.o[gi] = p.aout[gi % p.na] + (float)(x & 1u);
  __syncthreads();
  if (tid == 0) { const long long t2 = gt(); p.m[3 * blockIdx.x] = t0; p.m[3 * blockIdx.x + 1] = t1; p.m[3 * blockIdx.x + 2] = t2; }
}
static void launch_pdl(const void* f, dim3 g, dim3 b, size_t sm, cudaStream_t s, void** args, int cl) {
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = sm; cfg.stream = s;
  cudaLaunchAttribute at[2]; int na = 0;
  at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[na].val.programmaticStreamSerializationAllowed = 1; ++na;
  if (cl) { at[na].id = cudaLaunchAttributeClusterDimension; at[na].val.clusterDim.x = cl; at[na].val.clusterDim.y = 1; at[na].val.clusterDim.z = 1; ++na; }
  cfg.attrs = at; cfg.numAttrs = na;
  CK(cudaLaunchKernelExC(&cfg, f, args));
}
static void run_h(int G, int reps, unsigned char* wb, long long WB) {
  CK(cudaFuncSetAttribute(a_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CK(cudaFuncSetAttribute(b_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BSM));
  const int NB = 2 * G;
  float *pbuf, *ain, *aout, *bo; long long *am, *bm;
  CK(cudaMalloc(&pbuf, 132 * 256 * 4)); CK(cudaMalloc(&ain, 8 * 16 * 64 * 8 * 4)); CK(cudaMalloc(&aout, 8 * 16 * 64 * 8 * 4));
  CK(cudaMemset(ain, 0, 8 * 16 * 64 * 8 * 4)); CK(cudaMalloc(&bo, NB * 256 * 4)); CK(cudaMalloc(&am, 8 * 16 * 8)); CK(cudaMalloc(&bm, NB * 24));
  cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  const int arms[][2] = {{0, 0}, {1, 24576}, {2, 24576}, {3, 24576}, {1, 57344}, {2, 57344}, {3, 57344}};
  const int Ts[] = {1, 8};
  int rot = 0;
  for (int T : Ts) for (int ai = 0; ai < 7; ++ai) {
    const int pfm = arms[ai][0], pfb = arms[ai][1] ? arms[ai][1] : 4096;
    std::vector<double> hop, rdy, lead, rdym;
    for (int r = 0; r < reps + 5; ++r) {
      gate_kernel<<<1, 32, 0, s>>>(100000);
      long long spin_ns = 6000;
      p_kernel<<<132, 256, 0, s>>>(pbuf, spin_ns);
      const float* aip = ain; float* aop = aout; long long* amp = am;
      void* aa[] = {(void*)&aip, (void*)&aop, (void*)&amp};
      launch_pdl((const void*)a_kernel, dim3(T * CS), dim3(NT), 0, s, aa, CS);
      BP bp = {aout, T * CS * NT * 8, wb, WB / 65536, pfb, pfm, rot++, bo, bm};
      void* ba[] = {(void*)&bp};
      launch_pdl((const void*)b_kernel, dim3(NB), dim3(256), BSM, s, ba, 0);
      CK(cudaStreamSynchronize(s));
      if (r < 5) continue;
      std::vector<long long> a(T * CS), b(NB * 3);
      CK(cudaMemcpy(a.data(), am, a.size() * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), bm, b.size() * 8, cudaMemcpyDeviceToHost));
      const long long aend = *std::max_element(a.begin(), a.end());
      long long relmin = 1ll << 62, rdymax = 0; std::vector<double> ent, rd;
      for (int k = 0; k < NB; ++k) { relmin = std::min(relmin, b[3 * k + 1]); rdymax = std::max(rdymax, b[3 * k + 2]); ent.push_back((double)b[3 * k]); rd.push_back((double)b[3 * k + 2]); }
      hop.push_back((double)(relmin - aend)); rdy.push_back((double)(rdymax - aend)); lead.push_back((double)aend - pct(ent, .5));
      rdym.push_back(pct(rd, .5) - (double)aend);
    }
    printf("{\"part\":\"h\",\"T\":%d,\"pfm\":%d,\"pfb\":%d,\"n\":%zu,\"hop_p50\":%.0f,\"hop_p90\":%.0f,\"rdy_p50\":%.0f,\"rdy_p90\":%.0f,"
           "\"rdymed_p50\":%.0f,\"lead_p50\":%.0f}\n", T, pfm, arms[ai][1], hop.size(), pct(hop, .5), pct(hop, .9), pct(rdy, .5), pct(rdy, .9),
           pct(rdym, .5), pct(lead, .5));
    fflush(stdout);
  }
}
static int argi(int argc, char** argv, const char* k, int d) {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], k)) return atoi(argv[i + 1]);
  return d;
}
int main(int argc, char** argv) {
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  const int G = pr.multiProcessorCount;
  const long long WB = 1ll << 30;
  unsigned char* wb; CK(cudaMalloc(&wb, WB)); CK(cudaMemset(wb, 1, WB));
  printf("{\"dev\":\"%s\",\"G\":%d}\n", pr.name, G); fflush(stdout);
  const int part = argi(argc, argv, "--part", 3);
  if (part & 1) run_x(G, argi(argc, argv, "--reps", 3), wb, WB);
  if (part & 2) run_h(G, argi(argc, argv, "--hreps", 40), wb, WB);
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


def summarize(rows):
    xs = {(r["qm"], r["ns"], r["sb"], r["nld"]): r for r in rows if r.get("part") == "x"}
    for k, r in xs.items():
        lab = {0: "idle", 1: f"bulk {r['ns']}x{r['sb'] // 1024}K", 2: f"LDG {r['nld'] * 96 * 16 // 1024}K"}[r["qm"]]
        print(f"X {lab:14s} BW {r['TBps']:6.3f} err {r['err']} | " + " ".join(f"{o} {r[o + '_p50']:.0f}/{r[o + '_p95']:.0f}" for o in OPS))
    for r in rows:
        if r.get("part") == "h":
            arm = ["pf0", "ent", "aft", "reg"][r["pfm"]]
            print(f"H T{r['T']} {arm} {r['pfb'] // 1024:3d}K/CTA  hop {r['hop_p50']:.0f}/{r['hop_p90']:.0f}  rdy(max) {r['rdy_p50']:.0f}/{r['rdy_p90']:.0f}"
                  f"  rdy(med) {r['rdymed_p50']:.0f}  lead {r['lead_p50']:.0f} ns")
    v = verdict(rows)
    if v:
        print(v)


def verdict(rows):
    x = {(r["qm"], r["ns"]): r for r in rows if r.get("part") == "x"}
    r72 = x.get((1, 2))
    if not r72 or r72["err"]:
        return "VERDICT: no valid 72K row"
    sc, acq = r72["sc_p50"], r72["acq_p50"]
    go = acq <= 0.5 * sc
    return f"VERDICT merger-side fence.acquire vs fence.sc at 72K: {acq:.0f} vs {sc:.0f} ns -> {'GO' if go else 'KILL'} (rule <= 50 %)"


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "gen":
        pathlib.Path(argv[2]).write_text(gen())
        return
    if cmd == "summary":
        summarize(json.loads(pathlib.Path(argv[2]).read_text()))
        return
    out = "box_out/fnc0/run.json"
    for a in argv[2:]:
        if a.startswith("--out="):
            out = a[6:]
    wd = ROOT / "box_out/fnc0"
    wd.mkdir(parents=True, exist_ok=True)
    cu, exe = wd / "fnc.cu", wd / "fnc_bench"
    cu.write_text(gen())
    t0 = time.time()
    r = subprocess.run([find_nvcc(), "-O3", "-arch=sm_90a", "-std=c++17", "-Xptxas=-v", "-o", str(exe), str(cu)],
                       capture_output=True, text=True)
    (wd / "ptxas.log").write_text(r.stdout + r.stderr)
    print(f"[nvcc] rc {r.returncode} {time.time() - t0:.0f} s")
    if r.returncode:
        print((r.stdout + r.stderr)[-1500:])
        sys.exit(1)
    p = subprocess.run([str(exe)], capture_output=True, text=True, timeout=150)
    (wd / f"{pathlib.Path(out).stem}.log").write_text(p.stdout + p.stderr)
    rows = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
    (ROOT / out).write_text(json.dumps(rows, indent=0))
    print(f"[run] rc {p.returncode} rows {len(rows)}" + (f" stderr {p.stderr[-400:]}" if p.returncode else ""))
    summarize(rows)


if __name__ == "__main__":
    main(sys.argv)

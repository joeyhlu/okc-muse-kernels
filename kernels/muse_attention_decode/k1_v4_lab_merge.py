"""K1 v4 lab: why does the last-arriving CTA read freshly written partials at ~20 GB/s? (k1_v3 PROF stamps)
  python3 kernels/muse_attention_decode/k1_v4_lab_merge.py
A) fold: NP producer CTAs each write PER bytes, (optional __threadfence), atomicAdd; the LAST CTA reads all NP*PER
   bytes with load mode M (0 default ld.global, 1 ld.global.cg, 2 ld.volatile, 3 ld.relaxed.gpu, 4 ld.acquire.gpu)
   and U float4 loads in flight per thread. Time = %globaltimer around the read phase of the last CTA.
B) same bytes read by ONE CTA of a separate, later kernel (L2-resident, no cross-SM freshness).
C) DSMEM: cluster of NP CTAs each holding PER bytes in smem; after cluster.sync, rank 0 reads everything (C0) or
   every rank reads 1/NP of every peer (C1, the distributed merge pattern)."""
import statistics

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
namespace cg = cooperative_groups;
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
template <int M> __device__ __forceinline__ float4 ld4(const float4* p) {
  float4 v;
  if (M == 0) asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  if (M == 1) asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  if (M == 2) asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  if (M == 3) asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  if (M == 4) { asm volatile("ld.acquire.gpu.global.f32 %0, [%1];" : "=f"(v.x) : "l"(p));
                asm volatile("ld.relaxed.gpu.global.v2.f32 {%0,%1}, [%2];" : "=f"(v.y), "=f"(v.z) : "l"(((const float*)p) + 2)); v.w = 0.f; }
  return v;
}
template <int M, int U>
__device__ __forceinline__ float readall(const float4* src, int n4) {
  float acc = 0.f;
  for (int base = threadIdx.x; base < n4; base += blockDim.x * U) {
    float4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) { const int i = base + u * blockDim.x; v[u] = (i < n4) ? ld4<M>(src + i) : make_float4(0, 0, 0, 0); }
#pragma unroll
    for (int u = 0; u < U; ++u) acc += v[u].x + v[u].y + v[u].z + v[u].w;
  }
  return acc;
}
template <int M, int U, int FENCE>
__global__ void fold(float4* part, int* cnt, unsigned long long* t, float* sink, int per4) {
  float4* mine = part + (size_t)blockIdx.x * per4;
  for (int i = threadIdx.x; i < per4; i += blockDim.x) mine[i] = make_float4(i, blockIdx.x, 1.f, 2.f);
  if (FENCE) __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = (atomicAdd(cnt, 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  const unsigned long long t0 = gt();
  float acc = readall<M, U>(part, per4 * gridDim.x);
  __syncthreads();
  const unsigned long long t1 = gt();
  if (threadIdx.x == 0) { t[0] = t1 - t0; *cnt = 0; }
  sink[threadIdx.x] = acc;
}
template <int U>
__global__ void later(const float4* part, unsigned long long* t, float* sink, int n4) {
  const unsigned long long t0 = gt();
  float acc = readall<0, U>(part, n4);
  __syncthreads();
  const unsigned long long t1 = gt();
  if (threadIdx.x == 0) t[0] = t1 - t0;
  sink[threadIdx.x] = acc;
}
template <int DIST>
__global__ void dsm(unsigned long long* t, float* sink, int per4) {
  extern __shared__ float4 sm[];
  cg::cluster_group cl = cg::this_cluster();
  for (int i = threadIdx.x; i < per4; i += blockDim.x) sm[i] = make_float4(i, blockIdx.x, 1.f, 2.f);
  cl.sync();
  const unsigned long long t0 = gt();
  float acc = 0.f;
  const int np = cl.num_blocks(), me = cl.block_rank();
  if (DIST == 0) {
    if (me == 0)
      for (int r = 0; r < np; ++r) { const float4* peer = cl.map_shared_rank(sm, r);
        for (int i = threadIdx.x; i < per4; i += blockDim.x) { float4 v = peer[i]; acc += v.x + v.y + v.z + v.w; } }
  } else {
    const int sl = per4 / np;
    for (int r = 0; r < np; ++r) { const float4* peer = cl.map_shared_rank(sm, r) + me * sl;
      for (int i = threadIdx.x; i < sl; i += blockDim.x) { float4 v = peer[i]; acc += v.x + v.y + v.z + v.w; } }
  }
  __syncthreads();
  const unsigned long long t1 = gt();
  cl.sync();
  if (threadIdx.x == 0 && me == 0) t[0] = t1 - t0;
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
#define FOLD(M, U, F) case (M * 100 + U * 10 + F): fold<M, U, F><<<np, nthr>>>(part, cnt, tp, sk, per4); break;
torch::Tensor run(int64_t kind, int64_t mode, int64_t np, int64_t per_bytes, int64_t nthr) {
  const int per4 = (int)(per_bytes / 16);
  auto part = torch::empty({np * per4 * 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto cntt = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto t = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto sink = torch::empty({64 * 1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  float4* pp = (float4*)part.data_ptr<float>(); int* cnt = cntt.data_ptr<int>();
  unsigned long long* tp = (unsigned long long*)t.data_ptr<int64_t>(); float* sk = sink.data_ptr<float>();
  if (kind == 0) {
    float4* part = pp;
    switch (mode) {
      FOLD(0,1,1) FOLD(0,4,1) FOLD(0,8,1) FOLD(1,1,1) FOLD(1,4,1) FOLD(1,8,1) FOLD(2,4,1) FOLD(2,8,1)
      FOLD(3,4,1) FOLD(3,8,1) FOLD(4,8,1) FOLD(0,8,0) FOLD(1,8,0)
      default: TORCH_CHECK(false, "mode");
    }
  } else if (kind == 1) {
    fold<0, 8, 1><<<np, nthr>>>(pp, cnt, tp, sk, per4);
    if (mode == 1) later<1><<<1, nthr>>>(pp, tp, sk, (int)(np * per4));
    if (mode == 4) later<4><<<1, nthr>>>(pp, tp, sk, (int)(np * per4));
    if (mode == 8) later<8><<<1, nthr>>>(pp, tp, sk, (int)(np * per4));
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)np); cfg.blockDim = dim3((unsigned)nthr); cfg.dynamicSmemBytes = (size_t)per_bytes;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = (unsigned)np; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    if (mode == 0) { cudaFuncSetAttribute(dsm<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)per_bytes);
      cudaFuncSetAttribute(dsm<0>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsm<0>, tp, sk, per4)); }
    else { cudaFuncSetAttribute(dsm<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)per_bytes);
      cudaFuncSetAttribute(dsm<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsm<1>, tp, sk, per4)); }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return t;
}
"""
mod = load_inline("k1v4_labmerge", "torch::Tensor run(int64_t kind, int64_t mode, int64_t np, int64_t per_bytes, int64_t nthr);",
                  cuda_sources=SRC, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"])
NAMES = {0: "default", 1: "cg", 2: "volatile", 3: "relaxed.gpu", 4: "acquire+relaxed"}


def med(kind, mode, np_, per, nthr, reps=15):
    ts = []
    for _ in range(reps):
        t = mod.run(kind, mode, np_, per, nthr)
        torch.cuda.synchronize()
        ts.append(int(t.item()))
    return statistics.median(ts)


for np_, per, tag in ((8, 32768, "L=8 sliding: 8 splits x 32 KB"), (16, 4096, "L=1 sliding: 16 splits x 4 KB")):
    tot = np_ * per
    print(f"=== {tag} ({tot // 1024} KB)", flush=True)
    for nthr in (256, 1024):
        for code in (11, 41, 81, 111, 141, 181, 241, 281, 341, 381, 481, 80, 180):
            m, u, f = code // 100, (code // 10) % 10, code % 10
            ns = med(0, code, np_, per, nthr)
            print(f"LAB fold thr={nthr:4d} {NAMES[m]:16s} U={u} fence={f}: {ns / 1000:6.2f} us  {tot / max(ns, 1):6.1f} GB/s", flush=True)
        for u in (1, 4, 8):
            ns = med(1, u, np_, per, nthr)
            print(f"LAB later-kernel 1 CTA thr={nthr:4d} U={u}: {ns / 1000:6.2f} us  {tot / max(ns, 1):6.1f} GB/s", flush=True)
    for d in (0, 1):
        ns = med(2, d, np_, per, 256)
        print(f"LAB DSMEM cluster={np_} {'rank0 reads all' if d == 0 else 'distributed 1/np each'}: {ns / 1000:6.2f} us"
              f"  {tot / max(ns, 1):6.1f} GB/s", flush=True)

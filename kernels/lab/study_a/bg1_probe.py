"""study-a BG1 load-path probe (box / lab only; no judge file).

Question: how fast can one kernel stream a contiguous packed weight matrix of the T <= 8 dense sizes (6.3 / 12.6 / 28.3 /
50.3 MB) from cold HBM when every CTA owns ONE contiguous slice? Arms (1 CTA per SM, grid = #SMs):
  ldg    : 512 threads, plain 16 B no-allocate loads, UNR in flight per thread (UNR = 4 / 8 -> 32 / 64 KB per SM)
  bulk   : 1 producer lane issues cp.async.bulk chunks (CHUNK 12 / 24 KB) into a STAGES ring, mbarrier full / empty,
           7 consumer warps only wait + release (pure stream)
  bulkc  : same, and the consumers read every staged byte with LDS.128 (what a decode consumer must at least do)
Duration = last CTA end - first CTA start (%globaltimer), median of 9 runs, L2 flushed (256 MB write) before each run.
Prints one line per (arm, size): us and TB/s.
"""
import statistics

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint4 ldg_na(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(b)) : "memory");
}
// bounded wait (lab): gives up after ~2^26 polls and raises the error flag; no trap
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t parity, int* err) {
  uint32_t ok = 0;
  for (int i = 0; i < (1 << 26); ++i) {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(sa(b)), "r"(parity) : "memory");
    if (ok) return;
  }
  atomicExch(err, 1);
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}

__device__ __forceinline__ void stamp(unsigned long long* tm, uint64_t t0) {
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin(tm, (unsigned long long)t0);
    atomicMax(tm + 1, (unsigned long long)gtime());
  }
}

template <int UNR>
__global__ void __launch_bounds__(512, 1) probe_ldg(const uint8_t* __restrict__ w, long long bytes,
                                                    unsigned long long* tm, uint32_t* sink) {
  const uint64_t t0 = gtime();
  const long long per = (bytes / gridDim.x) & ~15LL;
  const long long a = per * blockIdx.x, b = blockIdx.x == gridDim.x - 1 ? bytes : a + per;
  uint32_t acc = 0;
  for (long long o = a + threadIdx.x * 16LL; o < b; o += (long long)UNR * 512 * 16) {
    uint4 v[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const long long p = o + (long long)u * 512 * 16;
      v[u] = p < b ? ldg_na(w + p) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9E3779B9u) sink[blockIdx.x] = acc;
  stamp(tm, t0);
}

// 256 threads: warp 0 lane 0 = producer, warps 1-7 = consumers
template <bool CONSUME>
__global__ void __launch_bounds__(256, 1) probe_bulk(const uint8_t* __restrict__ w, long long bytes, int chunk, int stages,
                                                     unsigned long long* tm, uint32_t* sink, int* err) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t full[8], empty[8];
  const uint64_t t0 = gtime();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const long long per = (bytes / gridDim.x) & ~15LL;
  const long long a = per * blockIdx.x, b = blockIdx.x == gridDim.x - 1 ? bytes : a + per;
  const int n = (int)((b - a + chunk - 1) / chunk);
  if (tid == 0) {
    for (int s = 0; s < stages; ++s) { mb_init(&full[s], 1); mb_init(&empty[s], 7); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  uint32_t acc = 0;
  if (warp == 0) {
    if (lane == 0) {
      for (int i = 0; i < n; ++i) {
        const int s = i % stages;
        if (i >= stages) mb_wait(&empty[s], ((i / stages) - 1) & 1, err);
        const long long o = a + (long long)i * chunk;
        const uint32_t len = (uint32_t)min((long long)chunk, b - o);
        mb_expect(&full[s], len);
        bulk_g2s(smem + (size_t)s * chunk, w + o, len, &full[s]);
      }
    }
  } else {
    const int ct = tid - 32;
    for (int i = 0; i < n; ++i) {
      const int s = i % stages;
      mb_wait(&full[s], (i / stages) & 1, err);
      if (CONSUME) {
        const long long o = a + (long long)i * chunk;
        const int len = (int)min((long long)chunk, b - o);
        const uint8_t* st = smem + (size_t)s * chunk;
        for (int off = ct * 16; off < len; off += 224 * 16) {
          const uint4 v = *reinterpret_cast<const uint4*>(st + off);
          acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
      }
      __syncwarp();
      if (lane == 0) mb_arrive(&empty[s]);
    }
  }
  if (acc == 0x9E3779B9u) sink[blockIdx.x] = acc;
  stamp(tm, t0);
}

void probe_run(int64_t arm, torch::Tensor w, int64_t bytes, int64_t p0, int64_t p1, torch::Tensor tm, torch::Tensor sink,
               torch::Tensor err) {
  const int S = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint8_t* wp = w.data_ptr<uint8_t>();
  auto* tp = reinterpret_cast<unsigned long long*>(tm.data_ptr<int64_t>());
  uint32_t* sp = reinterpret_cast<uint32_t*>(sink.data_ptr<int32_t>());
  int* ep = err.data_ptr<int32_t>();
  if (arm == 0) {
    if (p0 == 4) probe_ldg<4><<<S, 512, 0, stream>>>(wp, bytes, tp, sp);
    else probe_ldg<8><<<S, 512, 0, stream>>>(wp, bytes, tp, sp);
  } else {
    const int chunk = (int)p0, stages = (int)p1;
    const size_t smem = (size_t)chunk * stages;
    if (arm == 1) {
      cudaFuncSetAttribute(probe_bulk<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      probe_bulk<false><<<S, 256, smem, stream>>>(wp, bytes, chunk, stages, tp, sp, ep);
    } else {
      cudaFuncSetAttribute(probe_bulk<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      probe_bulk<true><<<S, 256, smem, stream>>>(wp, bytes, chunk, stages, tp, sp, ep);
    }
  }
}
"""
CPP_SRC = "void probe_run(int64_t arm, torch::Tensor w, int64_t bytes, int64_t p0, int64_t p1, torch::Tensor tm, " \
          "torch::Tensor sink, torch::Tensor err);"


def main():
    ext = load_inline(name="sa_bg1_probe", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["probe_run"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"], verbose=False)
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(dev), flush=True)
    sizes = {"unpool_slice": 6_291_456, "o": 12_582_912, "qkv": 28_311_552, "pooler": 50_331_648}
    w = torch.randint(0, 256, (max(sizes.values()),), dtype=torch.uint8, device=dev)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    tm = torch.zeros(2, dtype=torch.int64, device=dev)
    sink = torch.zeros(1024, dtype=torch.int32, device=dev)
    err = torch.zeros(1, dtype=torch.int32, device=dev)
    arms = [("ldg u4", 0, 4, 0), ("ldg u8", 0, 8, 0)]
    for chunk in (12288, 24576):
        for stages in (4, 6, 8):
            if chunk * stages <= 200 * 1024:
                arms.append((f"bulk {chunk // 1024}K x{stages}", 1, chunk, stages))
                arms.append((f"bulkc {chunk // 1024}K x{stages}", 2, chunk, stages))
    for name, nbytes in sizes.items():
        for label, arm, p0, p1 in arms:
            us = []
            for rep in range(10):
                flush.fill_(rep & 0xFF)
                tm[0] = 2 ** 62
                tm[1] = 0
                ext.probe_run(arm, w, nbytes, p0, p1, tm, sink, err)
                torch.cuda.synchronize()
                if rep:
                    us.append((tm[1] - tm[0]).item() / 1000.0)
            med = statistics.median(us)
            print(f"bg1 {name:12s} {nbytes / 1e6:5.1f} MB  {label:14s} {med:7.2f} us  {nbytes / med / 1e6:5.2f} TB/s"
                  f"  (min {min(us):.2f})", flush=True)
    print("bg1 err flag", int(err.item()), flush=True)


if __name__ == "__main__":
    main()

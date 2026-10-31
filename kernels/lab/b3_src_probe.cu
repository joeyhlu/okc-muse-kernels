// R-b3-0 probe (not yet run: the WSL service timed out on 2026-10-31). Build: nvcc -arch=sm_90a -cubin -DARM=0|1 b3_src_probe.cu
// Expected: ARM 0 compiles (UBLKRED), ARM 1 is rejected by ptxas (cp.reduce.async.bulk has no .global source).
// B3 $0 probe: which source state spaces does cp.reduce.async.bulk (.add.f32, global dst) accept on sm_90a?
#include <cstdint>
#ifndef ARM
#define ARM 0
#endif
extern "C" __global__ void probe(float* y, const float* g) {
  __shared__ __align__(128) float s[256];
  s[threadIdx.x] = threadIdx.x;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
#if ARM == 0   // shared::cta source (the only documented form)
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 1024;" :: "l"(y), "r"((uint32_t)__cvta_generic_to_shared(s)) : "memory");
#elif ARM == 1 // global source (would avoid the smem staging copy)
    asm volatile("cp.reduce.async.bulk.global.global.bulk_group.add.f32 [%0], [%1], 1024;" :: "l"(y), "l"(g) : "memory");
#endif
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
}

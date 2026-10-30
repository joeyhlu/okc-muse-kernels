// BRS SASS probe (sm_90a, R-brs-0 / A31): ship split-share count vs Muse (a) bulk-reduce arm (brs_muse) vs the PTX-correct bulk arm
// (brs_ptx = + fence.release.gpu). Build: nvcc -arch=sm_90a -cubin -o p.cubin brs_sass_probe.cu; cuobjdump -sass p.cubin | grep MEMBAR
#include <cstdint>
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ship: red.add.v4 + bar + atom.acq_rel (writer) ; last arriver fence.acq_rel + ld.cg
extern "C" __global__ void ship(float* fix, int* cnt, float* out) {
  int t = threadIdx.x;
  float4 v = make_float4(t, t + 1, t + 2, t + 3);
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(fix + 4 * t), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
  __shared__ int last;
  __syncthreads();
  if (t == 0) { int o; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;" : "=r"(o) : "l"(cnt) : "memory"); last = (o == 3); }
  __syncthreads();
  if (last) { asm volatile("fence.acq_rel.gpu;" ::: "memory"); out[t] = __ldcg(fix + 4 * t); }
}

// Muse (a): st.shared -> proxy fence -> bar -> lane0 bulk reduce -> commit -> wait_group 0 -> fence.proxy.async.global
//           -> atom.relaxed.gpu ; last: fence.acquire.gpu -> ld.cg
template <bool REL>
__device__ __forceinline__ void brs_body(float* fix, int* cnt, float* out) {
  __shared__ __align__(128) float st[128 * 4];
  __shared__ int last;
  int t = threadIdx.x;
  reinterpret_cast<float4*>(st)[t] = make_float4(t, t + 1, t + 2, t + 3);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (t == 0) {
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" :: "l"(fix), "r"(sa(st)), "r"(128 * 16) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    asm volatile("fence.proxy.async.global;" ::: "memory");
    if (REL) asm volatile("fence.release.gpu;" ::: "memory");
    int o; asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], 1;" : "=r"(o) : "l"(cnt) : "memory");
    if (o == 3) asm volatile("fence.acquire.gpu;" ::: "memory");
    last = (o == 3);
  }
  __syncthreads();
  if (last) out[t] = __ldcg(fix + 4 * t);
}
extern "C" __global__ void brs_muse(float* fix, int* cnt, float* out) { brs_body<false>(fix, cnt, out); }
extern "C" __global__ void brs_ptx(float* fix, int* cnt, float* out) { brs_body<true>(fix, cnt, out); }

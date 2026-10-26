"""Dense3p: dense3s with RAW-only copy enqueue/consume lookahead.

RAW fills commit X and W as one copy group without waiting. Consumption waits
for the current group while allowing the next tile to remain pending, then
publishes through the existing proxy fence and CTA barrier. Last tile waits0.
PACKED keeps its synchronous fill path; SW128 layout, split policy, MMA ring,
WGMMA waits/fences, math and linear output epilogue are unchanged. Untimed here.

prepare(w[,force_raw=True]), prepare_raw(w), run(x,state,out=None). Dense2 is
imported normally only for its frozen weight packer; its kernels are not used.
prepare does weight-only packing/workspace allocation. build() compiles a small
CUDA extension; run lazily builds on first warm call if needed. No activations,
outputs, raw pointers or descriptors are cached by prepare. Live pointers are
obtained by the launcher every call. Supply out for allocation-free graph use.

One fixed 64x64x64 geometry, one 128-thread warpgroup, three decoded W/X slots.
Weights are reconstructed eight BF16 values at a time into 16-byte shared
swizzled stores; there are no intermediate layout-conversion scratch tiles.
Keep one WGMMA group pending while filling stage i+2. Only the epilogue waits
for all groups. X cp.async overlaps weight load/decode within a stage fill.
Aligned raw W also uses cp.async. RAW pre-enqueues tiles 0/1, waits1 at each
consume except last wait0, and enqueues i+2 only after wg_wait<1> retires i-1.
PACKED still waits for all its copies inside each fill, as in dense3s.
T<=64 has four K splits plus an FP32 reduction, matching dense2's policy.
T>64 is full K and direct output. BF16 operands and FP32 accumulation.

CPU checks (no torch/CUDA imports):
    python kernels/lab/fresh_packed_dense3p.py --smoke
GPU compile/correctness/performance are for the parent; not established here.
"""

import hashlib

_MODULE = None
_CPP = r"""
#include <torch/extension.h>
#include <vector>
void dense3_launch(torch::Tensor x, torch::Tensor storage, torch::Tensor meta,
                   torch::Tensor out, torch::Tensor partials, bool packed);
std::vector<int64_t> dense3_resources(bool packed);
"""

_CUDA = r"""
// Native WGMMA instruction spelling and core-layout descriptor follow NVIDIA
// PTX ISA and CUTLASS cute/arch/mma_sm90_gmma.hpp. No CUTLASS runtime needed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

namespace d3 {
constexpr int TILE = 64, STAGES = 3, THREADS = 128;
struct Params {
  const uint16_t* x;
  const void* storage;
  const uint32_t* meta;
  void* dest;
  int M, N, K, kg, ng;
};
struct __align__(1024) Shared {
  uint16_t x[STAGES][TILE*TILE];
  uint16_t w[STAGES][TILE*TILE];
};
static_assert(sizeof(Shared) == 49152, "fixed three-slot ring");

__device__ __forceinline__ uint32_t sa(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// CUTLASS Major-K SW128: byte offset b=128*row+2*k, then b^((b&0x380)>>3).
// Both operands remain [MN,K]; each K8 vector stays contiguous and 16B aligned.
__host__ __device__ __forceinline__ int core_half(int row, int k) {
  return row*64 + (k ^ ((row&7)<<3));
}
__device__ __forceinline__ uint64_t descriptor(const uint16_t* ptr) {
  // 1024B-aligned tile bases give swizzle base-offset zero, including K16 slices.
  return uint64_t((sa(ptr) & 0x3ffffu) >> 4)
       | (uint64_t(16u >> 4) << 16) | (uint64_t(1024u >> 4) << 32)
       | (uint64_t(1) << 62);
}
__device__ __forceinline__ void store_core(uint16_t* dst, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
    :: "r"(sa(dst)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ void copy16(uint16_t* dst, const uint16_t* src, bool valid) {
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
    :: "r"(sa(dst)), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void copy_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void copy_wait() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
template<int N> __device__ __forceinline__ void copy_wait_pending() {
  asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ void proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template<int N> __device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}
@MMA@

__device__ __forceinline__ uint4 raw8(const uint16_t* p, int row, int k, int rows, int K) {
  if (row < rows && k+7 < K && K%8 == 0 && (reinterpret_cast<uintptr_t>(p)&15u)==0)
    return *reinterpret_cast<const uint4*>(p + size_t(row)*K + k);
  // Ragged stride/tail: never issue an unaligned vector load or an OOB scalar.
  uint32_t v[4];
  #pragma unroll
  for (int j=0; j<4; ++j) {
    uint32_t a=0, b=0;
    if (row < rows && k+2*j < K) a=p[size_t(row)*K+k+2*j];
    if (row < rows && k+2*j+1 < K) b=p[size_t(row)*K+k+2*j+1];
    v[j] = a | (b << 16);
  }
  return make_uint4(v[0],v[1],v[2],v[3]);
}
__device__ __forceinline__ uint32_t mask_pair(uint32_t v, bool row_ok, int k, int K) {
  return v & ((row_ok && k<K ? 0xffffu : 0u) | (row_ok && k+1<K ? 0xffff0000u : 0u));
}
template<bool PACKED> __device__ __forceinline__ void fill(
    const Params& p, Shared& sm, int stage, int kb, int nb, int mb) {
  const int tid=threadIdx.x;
  // Start activation copies before the synchronous weight load/decode work.
  #pragma unroll
  for (int q=0; q<4; ++q) {
    const int chunk=tid+q*128, r=chunk/8, c=(chunk%8)*8;
    const int row=mb*64+r, k=kb*64+c;
    uint16_t* dst=sm.x[stage]+core_half(r,c);
    if (p.K%8 == 0 && k+7 < p.K && (reinterpret_cast<uintptr_t>(p.x)&15u)==0) {
      const bool ok=row<p.M;
      const uint16_t* src=p.x+(ok ? size_t(row)*p.K+k : 0);
      copy16(dst,src,ok);
    } else store_core(dst,raw8(p.x,row,k,p.M,p.K));
  }
  if constexpr (PACKED) copy_commit();
  uint32_t tag=0;
  const uint32_t* payload=nullptr;
  if constexpr (PACKED) {
    tag=p.meta[kb*p.ng+nb];
    payload=reinterpret_cast<const uint32_t*>(p.storage)+size_t(tag>>9)*512;
  }
  #pragma unroll
  for (int q=0; q<4; ++q) {
    const int chunk=tid+q*128, r=chunk/8, c=(chunk%8)*8;
    const int row=nb*64+r, k=kb*64+c;
    uint4 v;
    if constexpr (PACKED) {
      if (tag & 1u) v=*reinterpret_cast<const uint4*>(payload+r*32+c/2);
      else {
        const uint2 sm8=*reinterpret_cast<const uint2*>(payload+r*16+c/4);
        const uint32_t ex8=payload[1024+r*8+c/8];
        const uint32_t base=(tag>>1)&255;
        uint32_t pairs[4];
        #pragma unroll
        for (int j=0; j<4; ++j) {
          const uint32_t smword = j<2 ? sm8.x : sm8.y;
          const int shift=(j%2)*16;
          const uint32_t a=(smword>>shift)&255, b=(smword>>(shift+8))&255;
          const uint32_t ea=base+((ex8>>(j*8))&15), eb=base+((ex8>>(j*8+4))&15);
          const uint32_t va=((a&128)<<8)|(ea<<7)|(a&127);
          const uint32_t vb=((b&128)<<8)|(eb<<7)|(b&127);
          pairs[j]=va|(vb<<16);
        }
        v=make_uint4(pairs[0],pairs[1],pairs[2],pairs[3]);
      }
      if (row>=p.N || k+7>=p.K) {
        v.x=mask_pair(v.x,row<p.N,k,p.K); v.y=mask_pair(v.y,row<p.N,k+2,p.K);
        v.z=mask_pair(v.z,row<p.N,k+4,p.K); v.w=mask_pair(v.w,row<p.N,k+6,p.K);
      }
    } else {
      const uint16_t* raw=reinterpret_cast<const uint16_t*>(p.storage);
      if (p.K%8 == 0 && k+7 < p.K && (reinterpret_cast<uintptr_t>(raw)&15u)==0) {
        const bool ok=row<p.N;
        const uint16_t* src=raw+(ok ? size_t(row)*p.K+k : 0);
        copy16(sm.w[stage]+core_half(r,c),src,ok);
        continue;
      }
      v=raw8(raw,row,k,p.N,p.K);
    }
    // One direct 16B core-layout store for eight exact BF16 values.
    store_core(sm.w[stage]+core_half(r,c),v);
  }
  if constexpr (!PACKED) copy_commit();
  if constexpr (PACKED) {
    copy_wait();
    // Each writer orders its generic shared stores before the async proxy;
    // the CTA barrier orders all writers' fences before subsequent WGMMA use.
    proxy_fence();
    __syncthreads();
  }
}

template<bool PACKED, bool FLOAT_OUT, int SK>
__global__ __launch_bounds__(128) void gemm(Params p) {
  __shared__ Shared sm;
  const int nb=blockIdx.x, mb=blockIdx.y, split=blockIdx.z;
  const int per=(p.kg+SK-1)/SK, first=split*per;
  const int count=max(0,min(per,p.kg-first));
  float acc[32];
  #pragma unroll
  for (int j=0;j<32;++j) acc[j]=0.f;
  @FENCE_ACC@
  if (count>0) fill<PACKED>(p,sm,0,first,nb,mb);
  if (count>1) fill<PACKED>(p,sm,1,first+1,nb,mb);
  for (int i=0;i<count;++i) {
    const int s=i%3;
    if constexpr (!PACKED) {
      // One committed X+W group per tile. Keep i+1 pending, but drain the last.
      if (i+1<count) copy_wait_pending<1>();
      else copy_wait_pending<0>();
      proxy_fence();
      __syncthreads();
    }
    wg_fence();
    #pragma unroll
    for (int j=0;j<4;++j)
      mma(acc,descriptor(sm.w[s]+j*16),descriptor(sm.x[s]+j*16));
    wg_commit();
    wg_wait<1>();
    // wait<1> retires tile i-1. Stage (i+2)%3 previously held i-1,
    // so it is reusable while tile i remains in flight.
    if (i+2<count) fill<PACKED>(p,sm,(i+2)%3,first+i+2,nb,mb);
  }
  wg_wait<0>();
  @FENCE_ACC@
  __syncthreads();
  // Reuse the retired X ring for a coalesced transposed-output epilogue.
  float* ep=reinterpret_cast<float*>(sm.x);
  const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
  const int r=warp*16+lane/4;
  #pragma unroll
  for (int j=0;j<8;++j) {
    const int t=8*j+2*(lane%4);
    ep[t*64+r]=acc[4*j]; ep[(t+1)*64+r]=acc[4*j+1];
    ep[t*64+r+8]=acc[4*j+2]; ep[(t+1)*64+r+8]=acc[4*j+3];
  }
  __syncthreads();
  for (int q=tid;q<4096;q+=128) {
    const int t=mb*64+q/64, n=nb*64+q%64;
    if (t<p.M && n<p.N) {
      const size_t dst=(SK>1 ? size_t(split)*64*p.N : 0)+size_t(t)*p.N+n;
      if constexpr (FLOAT_OUT) reinterpret_cast<float*>(p.dest)[dst]=ep[q];
      else reinterpret_cast<__nv_bfloat16*>(p.dest)[dst]=__float2bfloat16_rn(ep[q]);
    }
  }
}
template<bool FLOAT_OUT> __global__ void reduce(const float* part, void* out, int M, int N) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<M*N) {
    float v=part[i];
    #pragma unroll
    for (int s=1;s<4;++s) v+=part[size_t(s)*64*N+i];
    if constexpr (FLOAT_OUT) reinterpret_cast<float*>(out)[i]=v;
    else reinterpret_cast<__nv_bfloat16*>(out)[i]=__float2bfloat16_rn(v);
  }
}
template<bool PACKED> void launch(const torch::Tensor& x,const torch::Tensor& storage,
    const torch::Tensor& meta,const torch::Tensor& out,const torch::Tensor& part) {
  Params p;
  p.x=reinterpret_cast<const uint16_t*>(x.data_ptr()); p.storage=storage.data_ptr();
  p.meta=PACKED ? reinterpret_cast<const uint32_t*>(meta.data_ptr()) : nullptr;
  p.M=int(x.size(0));p.N=int(out.size(1));p.K=int(x.size(1));p.kg=(p.K+63)/64;p.ng=(p.N+63)/64;
  auto stream=at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  if (p.M<=64) {
    p.dest=part.data_ptr();
    gemm<PACKED,true,4><<<dim3(p.ng,(p.M+63)/64,4),128,0,stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if(out.scalar_type()==at::kFloat) reduce<true><<<(p.M*p.N+255)/256,256,0,stream>>>(part.data_ptr<float>(),out.data_ptr(),p.M,p.N);
    else reduce<false><<<(p.M*p.N+255)/256,256,0,stream>>>(part.data_ptr<float>(),out.data_ptr(),p.M,p.N);
  } else {
    p.dest=out.data_ptr();
    if(out.scalar_type()==at::kFloat) gemm<PACKED,true,1><<<dim3(p.ng,(p.M+63)/64,1),128,0,stream>>>(p);
    else gemm<PACKED,false,1><<<dim3(p.ng,(p.M+63)/64,1),128,0,stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace d3

void dense3_launch(torch::Tensor x, torch::Tensor storage, torch::Tensor meta,
                   torch::Tensor out, torch::Tensor partials, bool packed) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.is_contiguous(),"dense3 x");
  TORCH_CHECK(out.is_cuda() && out.device()==x.device() && out.is_contiguous() && out.dim()==2,"dense3 out");
  TORCH_CHECK(out.scalar_type()==at::kFloat || out.scalar_type()==at::kBFloat16,"dense3 out dtype");
  TORCH_CHECK(storage.is_cuda() && storage.device()==x.device() && storage.is_contiguous(),"dense3 weights");
  TORCH_CHECK(partials.is_cuda() && partials.device()==x.device() && partials.scalar_type()==at::kFloat && partials.is_contiguous(),"dense3 scratch");
  TORCH_CHECK(x.size(0)>=1 && x.size(0)<=256 && out.size(0)==x.size(0),"dense3 T");
  TORCH_CHECK(partials.numel()>=4*64*out.size(1),"dense3 scratch capacity");
  if(packed) {
    TORCH_CHECK(storage.scalar_type()==at::kInt && meta.scalar_type()==at::kInt && meta.is_cuda() && meta.device()==x.device() && meta.is_contiguous(),"dense3 packed/meta");
    TORCH_CHECK(meta.numel()==((x.size(1)+63)/64)*((out.size(1)+63)/64),"dense3 descriptor shape");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(storage.data_ptr())&15u)==0,"dense3 packed alignment");
  } else TORCH_CHECK(storage.scalar_type()==at::kBFloat16 && storage.numel()==out.size(1)*x.size(1),"dense3 raw weights");
  c10::cuda::CUDAGuard guard(x.device());
  if(packed) d3::launch<true>(x,storage,meta,out,partials);
  else d3::launch<false>(x,storage,meta,out,partials);
}
std::vector<int64_t> dense3_resources(bool packed) {
  cudaFuncAttributes a;
  if(packed) C10_CUDA_CHECK(cudaFuncGetAttributes(&a,d3::gemm<true,false,1>));
  else C10_CUDA_CHECK(cudaFuncGetAttributes(&a,d3::gemm<false,false,1>));
  return {int64_t(a.numRegs),int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),int64_t(a.maxThreadsPerBlock)};
}
"""


def cuda_source():
    """Generate only fixed m64n64k16 inline-PTX operands, no source-module edits."""
    regs = ",".join(f"%{i}" for i in range(32))
    outputs = ", ".join(f'"+f"(a[{i}])' for i in range(32))
    mma = ("__device__ __forceinline__ void mma(float* a, uint64_t da, uint64_t db) {\n"
           '  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; '
           f'wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {{{regs}}}, '
           '%32, %33, p, 1, 1, 0, 0; }"\n'
           f'    : {outputs} : "l"(da), "l"(db) : "memory");\n}}')
    fence = f'asm volatile("" : {outputs.replace("a[", "acc[")} :: "memory");'
    return _CUDA.replace("@MMA@", mma).replace("@FENCE_ACC@", fence)


def build(verbose=False):
    """Compile via installed nvcc; does not launch or create input descriptors."""
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline
        source = cuda_source()
        digest = hashlib.sha256((_CPP + source).encode()).hexdigest()[:12]
        _MODULE = load_inline(
            name="fresh_packed_dense3p_" + digest, cpp_sources=_CPP, cuda_sources=source,
            functions=["dense3_launch", "dense3_resources"], with_cuda=True,
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo",
                               "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"],
            verbose=verbose,
        )
    return _MODULE


def prepare(w, *, force_raw=False):
    """Weight-only dense2 packing; no GEMM compile, activation work or descriptors."""
    import fresh_packed_dense2 as D2
    state = D2.prepare(w, force_raw=force_raw)
    stats = dict(state["stats"])
    stats.update(implementation="cuda_wgmma_sw128_ring3_raw_lookahead_initfence", shared_tile_bytes=49152,
                 weight_core_store_bytes=16, decoded_weight_stages=3)
    state["stats"] = state["compression_stats"] = stats
    return state


def prepare_raw(w):
    return prepare(w, force_raw=True)


def run(x, state, out=None):
    """Warm build outside graph capture. Provided-out path allocates no tensors."""
    import torch
    n, k = state["n"], state["k"]
    if (x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16
            or not x.is_cuda or not x.is_contiguous() or x.device != state["data"].device):
        raise ValueError("x must be contiguous CUDA BF16 [T,K] on prepared device")
    m = int(x.shape[0])
    if not 1 <= m <= 256:
        raise ValueError("T=1..256 required")
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if (tuple(out.shape) != (m, n) or out.dtype not in (torch.bfloat16, torch.float32)
            or not out.is_contiguous() or out.device != x.device):
        raise ValueError("out must be contiguous BF16/FP32 [T,N] on x.device")
    if out is x or out is state["data"] or out is state["partials"]:
        raise ValueError("output must not alias input/state")
    module = build()
    meta = state["meta"] if state["packed"] else state["data"]
    module.dense3_launch(x, state["data"], meta, out, state["partials"], state["packed"])
    return out


def smoke():
    """CPU copy-group/slot proof plus SW128/descriptors/exact data and epilogue."""
    import ast
    import random
    import struct
    from pathlib import Path
    import fresh_packed_dense2 as D2
    ast.parse(Path(__file__).read_text(encoding="utf-8"))
    src = cuda_source()
    assert "@MMA@" not in src and "@FENCE_ACC@" not in src
    assert "st.shared.v4.u32" in src and "st.shared.b16" not in src
    assert src.count("wg_wait<0>();") == 1
    assert ':: "memory"' in src
    operand_fence = 'asm volatile("" : "+f"(acc[0])'
    assert src.count(operand_fence) == 2
    assert (src.index('for (int j=0;j<32;++j) acc[j]=0.f;')
            < src.index(operand_fence) < src.index('if (count>0) fill<PACKED>'))
    assert src.count('copy16(sm.w[stage]+core_half(r,c),src,ok);') == 1
    assert src.count('if constexpr (!PACKED) copy_commit();') == 1
    assert src.count('if constexpr (PACKED) copy_commit();') == 1
    assert (src.index('copy16(sm.w[stage]+core_half(r,c),src,ok);')
            < src.index('if constexpr (!PACKED) copy_commit();')
            < src.index('copy_wait();'))
    assert src.count('copy_wait_pending<1>();') == 1
    assert src.count('copy_wait_pending<0>();') == 1
    consume = src[src.index('  for (int i=0;i<count;++i) {'):src.index('  wg_wait<0>();')]
    assert (consume.index('if (i+1<count) copy_wait_pending<1>();')
            < consume.index('else copy_wait_pending<0>();')
            < consume.index('proxy_fence();') < consume.index('__syncthreads();')
            < consume.index('wg_fence();') < consume.index('wg_wait<1>();')
            < consume.index('if (i+2<count) fill<PACKED>'))
    assert 'struct __align__(1024) Shared' in src
    assert 'return row*64 + (k ^ ((row&7)<<3));' in src
    assert '(uint64_t(16u >> 4) << 16)' in src
    assert '(uint64_t(1024u >> 4) << 32)' in src
    assert '(uint64_t(1) << 62)' in src
    assert 'descriptor(sm.w[s]+j*16),descriptor(sm.x[s]+j*16)' in src
    def core(r, k):
        return r * 64 + (k ^ ((r & 7) << 3))
    assert sorted(core(r, k) for r in range(64) for k in range(64)) == list(range(4096))
    # Independent byte-address Swizzle<3,4,3> and canonical Major-K layout.
    def swizzle_byte(b):
        return b ^ ((b & 0x380) >> 3)
    labels = [None] * 4096
    for r in range(64):
        for k in range(64):
            canonical = (r % 8) * 64 + (r // 8) * 512 + (k % 8) + (k // 8) * 8
            assert 2 * core(r, k) == swizzle_byte(2 * canonical)
            assert swizzle_byte(swizzle_byte(2 * canonical)) == 2 * canonical
            labels[core(r, k)] = (r, k)
    for r in range(64):
        for k in range(0, 64, 8):
            assert core(r, k) % 8 == 0
            assert [core(r, k + j) for j in range(8)] == list(range(core(r, k), core(r, k) + 8))
    # Each eight-lane copy group writes 32 different banks with four words/lane.
    # Test all warps/q iterations, with the unchanged copy lane assignment.
    for warp in range(4):
        for q in range(4):
            for group in range(4):
                banks = []
                for lane in range(group * 8, group * 8 + 8):
                    chunk = warp * 32 + lane + q * 128
                    r, c = chunk // 8, (chunk % 8) * 8
                    banks.extend((core(r, c) // 2 + v) % 32 for v in range(4))
                assert sorted(banks) == list(range(32))
    # Reconstruct every logical BF16 address from descriptor fields and the
    # PTX Major-K B128 canonical atom, rather than from core() at the read side.
    # A=W[N,K], B(k,t)=X[t,k]; both descriptors use this same K-major layout.
    def make_desc(start):
        return ((start & 0x3ffff) >> 4) | (1 << 16) | (64 << 32) | (1 << 62)
    for shared_base in (0, 1024):
        for operand_offset in (0, 24576):
            for stage in range(3):
                tile_base = shared_base + operand_offset + stage * 8192
                assert tile_base % 1024 == 0
                for j in range(4):
                    desc = make_desc(tile_base + 32 * j)
                    start = (desc & 0x3fff) << 4
                    leading = ((desc >> 16) & 0x3fff) << 4
                    stride = ((desc >> 32) & 0x3fff) << 4
                    assert leading == 16 and stride == 1024
                    assert (desc >> 49) & 7 == 0 and (desc >> 62) & 3 == 1
                    assert start == tile_base + 32 * j
                    for r in range(64):
                        for kk in range(16):
                            logical = start + (r % 8) * 128 + (r // 8) * stride + 2 * kk
                            addr = swizzle_byte(logical)
                            assert addr % 2 == 0
                            assert tile_base <= addr < tile_base + 8192
                            assert labels[(addr - tile_base) // 2] == (r, 16 * j + kk)
    # Model the new raw copy's exact 16B/zero-fill behavior, plus its unchanged
    # scalar fallback. No vector copy may read beyond its real source row.
    raw_counts = dict(vector=0, zero=0, scalar=0)
    for n, k, base_mod16 in [(256, 256, 0), (65, 72, 0), (129, 259, 0),
                              (65, 72, 2), (1, 8, 0), (1, 7, 0)]:
        values = [(i * 40503 + 197) & 65535 for i in range(n * k)]
        if n * k == 65536:
            assert len(set(values)) == 65536  # every possible BF16 bit pattern
        for nb in range((n + 63) // 64):
            for kb in range((k + 63) // 64):
                shmem = [None] * 4096
                for tid in range(128):
                    for q in range(4):
                        chunk = tid + q * 128
                        r, c = chunk // 8, (chunk % 8) * 8
                        row, kk = nb * 64 + r, kb * 64 + c
                        if k % 8 == 0 and kk + 7 < k and base_mod16 == 0:
                            ok = row < n
                            source = row * k + kk if ok else 0
                            assert (source * 2 + base_mod16) % 16 == 0
                            assert core(r, c) * 2 % 16 == 0
                            if ok:
                                assert source + 8 <= n * k
                                v = values[source:source + 8]
                                raw_counts["vector"] += 1
                            else:
                                v = [0] * 8
                                raw_counts["zero"] += 1
                        else:
                            v = [values[row * k + kk + j]
                                 if row < n and kk + j < k else 0 for j in range(8)]
                            raw_counts["scalar"] += 1
                        shmem[core(r, c):core(r, c) + 8] = v
                for r in range(64):
                    for c in range(64):
                        row, kk = nb * 64 + r, kb * 64 + c
                        expected = values[row * k + kk] if row < n and kk < k else 0
                        assert shmem[core(r, c)] == expected
    assert all(raw_counts.values())
    rng = random.Random(3)
    # Include every BF16 code through compact tiles at every exponent base group.
    vals = [0] * 65536
    for kb in range(4):
        for nb in range(4):
            for lane in range(4096):
                sm, exp = lane & 255, 16 * (kb * 4 + nb) + lane // 256
                vals[(nb * 64 + lane // 64) * 256 + kb * 64 + lane % 64] = ((sm & 128) << 8) | (exp << 7) | (sm & 127)
    cases = [(vals, 256, 256), ([0x3C23] * (127 * 257), 127, 257),
             ([rng.randrange(65536) for _ in range(129 * 259)], 129, 259)]
    mixed = [0x3C23] * (128 * 128)
    mixed[0] = 0x4E80  # large exponent in just one tile
    cases.append((mixed, 128, 128))
    packed_cases = 0
    for values, n, k in cases:
        fmt = D2.pack_bits(values, n, k)
        assert D2.unpack_bits(fmt) == values
        packed_cases += int(fmt.packed)
        ng, kg = (n + 63) // 64, (k + 63) // 64
        for tile in range(ng * kg):
            tag = fmt.descriptors[tile] if fmt.packed else 0
            kb, nb = divmod(tile, ng)
            offset, base, raw = (tag >> 9) * 2048, (tag >> 1) & 255, tag & 1
            shmem = [None] * 4096
            for tid in range(128):
                for q in range(4):
                    chunk = tid + q * 128
                    r, c = chunk // 8, (chunk % 8) * 8
                    if not fmt.packed:
                        row = nb * 64 + r
                        v = [values[row * k + kb * 64 + c + j]
                             if row < n and kb * 64 + c + j < k else 0 for j in range(8)]
                    elif raw:
                        v = list(struct.unpack_from("<8H", fmt.data, offset + (r * 64 + c) * 2))
                    else:
                        smx, smy = struct.unpack_from("<II", fmt.data, offset + r * 64 + c)
                        ex = struct.unpack_from("<I", fmt.data, offset + 4096 + r * 32 + c // 2)[0]
                        v = []
                        for j in range(4):
                            word = smx if j < 2 else smy
                            shift = (j % 2) * 16
                            a, b = (word >> shift) & 255, (word >> (shift + 8)) & 255
                            ea, eb = base + ((ex >> (j * 8)) & 15), base + ((ex >> (j * 8 + 4)) & 15)
                            pair = ((a & 128) << 8) | (ea << 7) | (a & 127)
                            pair |= (((b & 128) << 8) | (eb << 7) | (b & 127)) << 16
                            v.extend((pair & 65535, pair >> 16))
                    for j in range(8):
                        row, kk = nb * 64 + r, kb * 64 + c + j
                        v[j] = v[j] if row < n and kk < k else 0
                    shmem[core(r, c):core(r, c) + 8] = v
            for r in range(64):
                for c in range(64):
                    row, kk = nb * 64 + r, kb * 64 + c
                    expected = values[row * k + kk] if row < n and kk < k else 0
                    assert shmem[core(r, c)] == expected
    assert packed_cases >= 3
    # RAW per-thread group/slot proof. Record committed groups even if already
    # complete (including empty ragged groups): wait counts groups, not tiles
    # whose transfers happen to remain physically in flight. Each wait also
    # retires all older committed groups, independent of early completions.
    full = (1 << 128) - 1
    for count in range(130):
        for mode in range(4):
            slots, groups, pending_mma, retired, published = {}, [], [], set(), set()
            def enqueue(tile):
                stage = tile % 3
                old = slots.get(stage)
                assert old not in pending_mma
                assert all(g[0] != old or g[1] == full for g in groups)
                if old is not None:
                    assert old in retired
                slots[stage] = tile
                # 0: all copies initially pending; 1: already complete;
                # 2: mixed async/zero/generic readiness across all 128 threads;
                # 3: alternating empty/scalar-only and initially pending groups.
                done = (0 if mode == 0 else full if mode == 1 else
                        sum(1 << t for t in range(128) if (t + tile) % 3) if mode == 2 else
                        full if tile % 2 else 0)
                groups.append([tile, done])  # exactly one X+W commit per thread
                assert len(groups) <= 2
            for tile in range(min(2, count)):
                enqueue(tile)
            consumed = []
            for i in range(count):
                allowance = 1 if i + 1 < count else 0
                assert [g[0] for g in groups] == list(range(i, min(count, i + 2)))
                complete_count = len(groups) - allowance
                completed = groups[:complete_count]
                for g in completed:
                    g[1] = full
                assert completed[0] == [i, full]
                groups = groups[complete_count:]
                assert slots[i % 3] == i
                # Each writer completed its current copies/generic stores;
                # proxy fence plus CTA barrier publishes every writer's data.
                published.add(i)
                assert i in published
                pending_mma.append(i)
                retired.update(pending_mma[:-1])  # wg_wait<1>
                pending_mma = pending_mma[-1:]
                consumed.append(i)
                if i + 2 < count:
                    old = slots.get((i + 2) % 3)
                    assert old is None if i == 0 else old == i - 1
                    enqueue(i + 2)
            assert not groups  # final tile's copy wait0; also true for count0
            retired.update(pending_mma)  # unchanged final wg_wait<0>
            pending_mma.clear()
            assert consumed == list(range(count))
            assert retired == set(range(count))
    # Original MMA-ring invariant also holds for the unchanged PACKED schedule.
    for count in range(130):
        slots, pending = {}, []
        for i in range(min(2, count)):
            slots[i % 3] = i
        for i in range(count):
            assert slots[i % 3] == i
            pending.append(i)
            pending = pending[-1:]
            if i + 2 < count:
                s = (i + 2) % 3
                assert slots.get(s) not in pending
                slots[s] = i + 2
        pending.clear()  # final wait<0>, before shared epilogue reuse
    # PTX accumulator coordinates: every [token,output] cell exactly once.
    cells = []
    for tid in range(128):
        warp, lane = divmod(tid, 32)
        r = warp * 16 + lane // 4
        for j in range(8):
            t = j * 8 + 2 * (lane % 4)
            cells.extend([(t, r), (t + 1, r), (t, r + 8), (t + 1, r + 8)])
    assert len(cells) == len(set(cells)) == 4096
    assert set(cells) == {(t, n) for t in range(64) for n in range(64)}
    print("PASS CPU dense3p: RAW per-thread copy groups/drain/slot lifetimes, SW128/descriptors/all BF16 bits, packed ring and epilogue")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.parse_args()
    smoke()

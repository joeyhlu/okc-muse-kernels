#!/usr/bin/env python3
"""MSKEL: standalone, NO layer math and NO submission/production integration.

Lead: python3 kernels/lab/mega_skel.py --out=box_out/mskel/run.json
CPU:  python -B kernels/lab/mega_skel.py --dry
Compile/SASS (no GPU): bash tools/local/mskel_compile.py (under cudaenv.sh).

Decimal MB for the requested weights/pool; judge flush is 256 MiB. T=8's
7.5 experts means seven full synthetic experts and a half-sized eighth.
All arms use the same 16-byte LDG/XOR stream, descriptors and payloads.
No imports/builds/allocations occur until main; --dry does not import torch.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time

THREADS = 256
VECTORS = 8                         # 32 KiB/CTA ring stage, 32 regs/thread
STAGE_BYTES = THREADS * VECTORS * 16
PREFETCH_CAP = 20_000_000           # per transition, across the entire grid
POOL_BYTES = 64_000_000
ATTN_ROWS = 8192                    # 2048 keys * K/V * two KV heads
ATTN_BYTES = ATTN_ROWS * 128
FLUSH_BYTES = 256 * 1024 * 1024
SLOTS = 25
COUNTERS = 16
PHASES = ["qkv", "attention", "o_proj", "boundary", "pooler", "router", "top8"]
PHASES += [f"{kind}_{e}" for e in range(8) for kind in ("w13", "w2")]
PHASES += ["unpooler", "final_boundary"]
ARM_IDS = {"A": 0, "A0": 1, "B": 2, "S": 3}
CUDA_FLAGS = ["-O3", "-std=c++17", "-lineinfo",
              "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v,-warn-spills"]


def even(n, rank, count):
    """Exact half-open split, at most one 16-byte vector of imbalance."""
    return n * rank // count, n * (rank + 1) // count


def make_plan(tokens, grid):
    if tokens not in (1, 8) or grid not in (132, 264):
        raise ValueError("MSKEL only implements T=1/8, G=132/264")
    experts = [1, 1] if tokens == 1 else [1] * 7 + [0.5]
    sizes = {0: 28_300_000, 2: 12_600_000, 4: 50_300_000,
             5: 8_400_000, 23: 6_300_000}
    for e, fraction in enumerate(experts):
        sizes[7 + 2 * e] = int(18_900_000 * fraction)
        sizes[8 + 2 * e] = int(9_400_000 * fraction)
    # Descriptors are [offset_in_uint4, count_in_uint4], phase-major.
    desc = [[[0, 0] for _ in range(grid)] for _ in range(SLOTS)]
    ranges, offset = {}, 0
    for phase in sorted(sizes):
        n = sizes[phase] // 16
        assert n * 16 == sizes[phase]
        ranks = list(range(32, grid)) if phase == 4 else list(range(32)) if phase == 5 else list(range(grid))
        for rank, cta in enumerate(ranks):
            lo, hi = even(n, rank, len(ranks))
            desc[phase][cta] = [offset + lo, hi - lo]
        ranges[str(phase)] = [offset, n]
        offset += n
    # Page-size-one index slabs, one head per CTA parity; K then V per head.
    for cta in range(grid):
        lo, hi = even(4096, cta // 2, grid // 2)
        desc[1][cta] = [(cta % 2) * 4096 + lo, hi - lo]  # rows, not uint4
    total = offset * 16 + ATTN_BYTES
    pure = [list(even(total // 16, c, grid)) for c in range(grid)]
    pure = [[lo, hi - lo] for lo, hi in pure]
    return dict(tokens=tokens, grid=grid, experts=len(experts), fractions=experts,
                router_ctas=32, desc=desc, pure=pure, sizes=sizes, ranges=ranges,
                weight_bytes=offset * 16, attention_bytes=ATTN_BYTES, total_bytes=total,
                pf_per_cta=(PREFETCH_CAP // grid // 128) * 128)


CUDA = r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
namespace ms {
constexpr int NT=256, U=8, NV=NT*U, NP=25, NC=16;
constexpr unsigned long long WAIT_NS=50000000ull;  // bounded 50 ms, no trap
struct Span { unsigned long long off, n; };
struct Params {
  const uint4* arena; const uint4* pool; const unsigned* indices;
  const Span* desc; const Span* pure;
  unsigned* counters; unsigned* ready; unsigned* epoch; unsigned* error;
  unsigned* checks; unsigned* partial; unsigned* combined;
  float* norms; unsigned long long* marks;
  int g, experts, pf_cap;
};
struct Shared { unsigned warp[8], last, ok, ep; };
__device__ __forceinline__ unsigned long long now() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); return t;
}
__device__ __forceinline__ unsigned acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory"); return v;
}
__device__ __forceinline__ void release(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;"::"l"(p),"r"(v):"memory");
}
__device__ __forceinline__ unsigned arrive(unsigned* p) {
  unsigned v; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;":"=r"(v):"l"(p):"memory"); return v;
}
__device__ __forceinline__ uint4 load16(const uint4* p) {
  uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
    :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory"); return v;
}
__device__ __forceinline__ unsigned fold(uint4 v) { return v.x^v.y^v.z^v.w; }
__device__ __forceinline__ Span span(const Params& p, int ph) { return p.desc[ph*p.g+blockIdx.x]; }
__device__ __forceinline__ unsigned reduce(unsigned x, Shared& s) {
  x=__reduce_xor_sync(0xffffffffu,x);
  if ((threadIdx.x&31)==0) s.warp[threadIdx.x/32]=x;
  __syncthreads();
  if (threadIdx.x==0) { x=0; for(int i=0;i<8;++i) x^=s.warp[i]; s.warp[0]=x; }
  __syncthreads(); return s.warp[0];
}
template<bool MK> __device__ __forceinline__ void mark(const Params& p,int ph,int end) {
  if constexpr(MK) if(threadIdx.x==0) p.marks[(ph*p.g+blockIdx.x)*2+end]=now();
}
__device__ __forceinline__ void save(const Params& p,int ph,unsigned x,Shared& s) {
  x=reduce(x,s); if(threadIdx.x==0) p.checks[ph*p.g+blockIdx.x]=x;
}
// Eight outstanding independent 16-byte loads, then eight four-word XOR folds.
// Tail predicates do not read beyond the exact descriptor (no padded weight reads).
__device__ __forceinline__ unsigned stage(const uint4* src,Span d,unsigned long long k) {
  uint4 v[U];
  #pragma unroll
  for(int u=0;u<U;++u) {
    unsigned long long j=k+threadIdx.x+u*NT;
    v[u]=j<d.n ? load16(src+d.off+j) : make_uint4(0,0,0,0);
  }
  unsigned x=0;
  #pragma unroll
  for(int u=0;u<U;++u) x^=fold(v[u]);
  return x;
}
__device__ __forceinline__ unsigned stream(const uint4* src,Span d,unsigned long long begin=0) {
  unsigned x=0; for(unsigned long long k=begin;k<d.n;k+=NV) x^=stage(src,d,k); return x;
}
__device__ __forceinline__ unsigned gather(const Params& p,unsigned begin=0,unsigned end=0xffffffffu) {
  Span d=span(p,1); unsigned x=0;
  for(unsigned j=begin+threadIdx.x;j<min((unsigned)d.n*8,end);j+=NT) {
    unsigned row=p.indices[d.off+j/8]; x^=fold(load16(p.pool+(size_t)row*8+j%8));
  }
  return x;
}
// An invocation has a finite issue budget, not an unbounded repeated prefetch.
// Every CTA can issue at most floor(20MB/G/128)*128 bytes for one transition.
struct PF { int ph; unsigned pos, limit; };
__device__ __forceinline__ PF pf(const Params& p,int ph) {
  unsigned bytes=ph<0 ? 0 : (unsigned)(span(p,ph).n*(ph==1?128:16));
  return PF{ph,0,min(bytes,(unsigned)p.pf_cap)};
}
template<bool PRE> __device__ __forceinline__ void prefetch(const Params& p,PF& f) {
  if constexpr(PRE) {
    if(f.pos>=f.limit) return;
    Span d=span(p,f.ph); const char* ptr; unsigned bytes;
    if(f.ph==1) { // random page-size-one gather: prefetch one actual 128-byte row
      ptr=(const char*)p.pool+(size_t)p.indices[d.off+f.pos/128]*128; bytes=128;
    } else { ptr=(const char*)(p.arena+d.off)+f.pos; bytes=min(8192u,f.limit-f.pos); }
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"::"l"(ptr),"r"(bytes):"memory");
    f.pos+=bytes;
  }
}
// Each counter is used once per epoch. Last arriver resets count BEFORE release.
// Two banks + full epoch token prevent graph-replay ABA; stream ordering prevents
// a new invocation from overtaking a previous invocation's last reader.
__device__ __forceinline__ bool ticket(const Params& p,int id,int count,unsigned ep,Shared& s) {
  __syncthreads();
  if(threadIdx.x==0) s.last=(arrive(p.counters+(ep&1)*NC+id)==(unsigned)(count-1));
  __syncthreads(); return s.last;
}
__device__ __forceinline__ void publish(const Params& p,int id,unsigned ep) {
  if(threadIdx.x==0) { p.counters[(ep&1)*NC+id]=0; release(p.ready+(ep&1)*NC+id,ep); }
}
template<bool PRE> __device__ __forceinline__ bool wait(const Params& p,int id,unsigned ep,PF& f,Shared& s) {
  if(threadIdx.x==0) {
    unsigned long long start=now(); unsigned backoff=32; s.ok=1;
    while(acquire(p.ready+(ep&1)*NC+id)!=ep) {
      if(acquire(p.error) || now()-start>WAIT_NS) {
        atomicCAS(p.error,0u,(unsigned)(id+1)); s.ok=0; break;
      }
      prefetch<PRE>(p,f); __nanosleep(backoff); backoff=min(backoff*2,256u);
    }
  }
  __syncthreads(); return s.ok; // uniform return; no CTA abandons a CTA barrier
}
template<bool PRE> __device__ __forceinline__ bool barrier(const Params& p,int id,unsigned ep,int next,Shared& s) {
  if(ticket(p,id,p.g,ep,s)) publish(p,id,ep);
  PF f=pf(p,next); return wait<PRE>(p,id,ep,f,s);
}
template<bool MK> __device__ __forceinline__ void weights(const Params& p,int ph,Shared& s,unsigned prefix=0,int begin=0) {
  mark<MK>(p,ph,0); unsigned x=prefix^stream(p.arena,span(p,ph),begin);
  save(p,ph,x,s); mark<MK>(p,ph,1);
}
template<bool PRE,bool MK> __device__ __forceinline__ bool attention(const Params& p,unsigned ep,Shared& s,unsigned prefix=0,unsigned begin=0) {
  mark<MK>(p,1,0); unsigned x=prefix^gather(p,begin); x=reduce(x,s);
  if(threadIdx.x==0) { p.partial[blockIdx.x]=x; p.checks[p.g+blockIdx.x]=x; }
  if(ticket(p,1,p.g,ep,s)) {
    // Last CTA, threads [0,256): each covers j=tid,tid+256,... <G/2.
    for(int head=0;head<2;++head) {
      x=0; for(int j=threadIdx.x;j<p.g/2;j+=NT) x^=p.partial[2*j+head];
      x=reduce(x,s); if(threadIdx.x==0) p.combined[head]=x;
    }
    __syncthreads(); publish(p,1,ep);
  }
  PF f=pf(p,2); bool ok=wait<PRE>(p,1,ep,f,s); mark<MK>(p,1,1); return ok;
}
template<bool PRE,bool MK> __device__ __forceinline__ bool boundary(const Params& p,int ph,int id,unsigned ep,int next,Shared& s) {
  mark<MK>(p,ph,0);
  if(threadIdx.x==0) p.norms[(ph==24?p.g:0)+blockIdx.x]=(float)(blockIdx.x+1);
  if(!barrier<PRE>(p,id,ep,next,s)) return false;
  // ALL CTAs, threads [0,256): cover EVERY partial, including 256..263 at G264.
  float sum=0; for(int j=threadIdx.x;j<p.g;j+=NT) sum+=p.norms[(ph==24?p.g:0)+j];
  // Integer sums <= 34980 are exactly representable; xor stores prove all reads.
  save(p,ph,__float_as_uint(sum),s); mark<MK>(p,ph,1); return true;
}
template<bool PRE,bool MK> __device__ __forceinline__ bool pool_router(const Params& p,unsigned ep,Shared& s,unsigned prefix=0,int begin=0) {
  PF f=pf(p,7); // one cursor/budget across router AND pooler-join waits
  // The last router CTA can change on every replay. Clear ALL previous winners.
  if constexpr(MK) if(threadIdx.x==0) {
    p.marks[(6*p.g+blockIdx.x)*2]=0; p.marks[(6*p.g+blockIdx.x)*2+1]=0;
  }
  int ph=blockIdx.x<32?5:4; weights<MK>(p,ph,s,prefix,begin);
  if(blockIdx.x<32) {
    if(threadIdx.x==0) p.partial[blockIdx.x]=p.checks[ph*p.g+blockIdx.x];
    if(ticket(p,6,32,ep,s)) {
      mark<MK>(p,6,0);
      if(threadIdx.x==0) {
        unsigned used=0, out=0;
        for(int pick=0;pick<8;++pick) {
          unsigned best=0; int who=-1;
          for(int j=0;j<32;++j) if(!(used&(1u<<j)) && (who<0 || p.partial[j]>best)) { best=p.partial[j]; who=j; }
          used|=1u<<who; out^=best;
        }
        p.combined[2]=out;
      }
      mark<MK>(p,6,1); publish(p,6,ep);
    }
    if(!wait<PRE>(p,6,ep,f,s)) return false;
  }
  if(ticket(p,3,p.g,ep,s)) publish(p,3,ep);
  return wait<PRE>(p,3,ep,f,s);
}
template<bool PRE,bool MK> __global__ __launch_bounds__(NT,2) void persistent(Params p) {
  __shared__ Shared s;
  if(threadIdx.x==0) { s.ep=p.epoch[blockIdx.x]+1; p.epoch[blockIdx.x]=s.ep; }
  __syncthreads(); unsigned ep=s.ep;
  weights<MK>(p,0,s);
  if(!barrier<PRE>(p,0,ep,1,s)) return;
  if(!attention<PRE,MK>(p,ep,s)) return;
  weights<MK>(p,2,s);
  if(!boundary<PRE,MK>(p,3,2,ep,blockIdx.x<32?5:4,s)) return;
  if(!pool_router<PRE,MK>(p,ep,s)) return;
  for(int e=0;e<p.experts;++e) {
    weights<MK>(p,7+2*e,s);
    if(!barrier<PRE>(p,8+e,ep,8+2*e,s)) return;
    weights<MK>(p,8+2*e,s); // NO W2(e) barrier: can overlap W13(e+1)
  }
  if(!barrier<PRE>(p,4,ep,23,s)) return;
  weights<MK>(p,23,s);
  if(!boundary<PRE,MK>(p,24,5,ep,-1,s)) return;
}
// PDL graph: nine nodes, matching the coarse phases of the current chain.
// W13 and W2 are separate nodes; both use the same per-expert even byte cuts.
// First TWO register-ring stages load before griddepcontrol.wait. Nothing reads
// predecessor outputs there. No dependency on concurrent kernel scheduling.
template<int PH> __global__ __launch_bounds__(NT,2) void chain(Params p) {
  __shared__ Shared s;
  constexpr bool simple=(PH==0 || PH==2 || PH==4 || PH==7 || PH==8 || PH==23);
  int ph=PH==4?(blockIdx.x<32?5:4):PH;
  unsigned prefix=0;
  if constexpr(simple) {
    Span d=span(p,ph); prefix=stage(p.arena,d,0)^stage(p.arena,d,NV);
  }
  // The synthetic KV pool is prepared/immutable, so its gathers are independent.
  if constexpr(PH==1) prefix=gather(p,0,2*NV);
  asm volatile("griddepcontrol.wait;":::"memory");
  if(threadIdx.x==0) {
    s.ep=p.epoch[blockIdx.x];
    if constexpr(PH==0) p.epoch[blockIdx.x]=++s.ep;
  }
  __syncthreads(); unsigned ep=s.ep;
  // Every CTA signals AFTER its incoming dependency is complete, BEFORE its
  // body/barriers. All current CTAs must have signaled before successors start;
  // no unscheduled upstream block is needed by a resident downstream waiter.
  asm volatile("griddepcontrol.launch_dependents;":::"memory");
  if constexpr(PH==0 || PH==2 || PH==23) weights<false>(p,PH,s,prefix,2*NV);
  if constexpr(PH==1) if(!attention<false,false>(p,ep,s,prefix,2*NV)) return;
  if constexpr(PH==3 || PH==24) if(!boundary<false,false>(p,PH,PH==3?2:5,ep,-1,s)) return;
  if constexpr(PH==4) if(!pool_router<false,false>(p,ep,s,prefix,2*NV)) return;
  if constexpr(PH==7 || PH==8) {
    for(int e=0;e<p.experts;++e) {
      // W2 additionally checks exactly its matching W13 counter (PDL supplies
      // the stronger coarse-node ordering in B, as in a separate W13/W2 chain).
      PF f=pf(p,-1);
      if constexpr(PH==8) if(!wait<false>(p,8+e,ep,f,s)) return;
      weights<false>(p,PH+2*e,s,e==0?prefix:0,e==0?2*NV:0);
      if constexpr(PH==7) { if(ticket(p,8+e,p.g,ep,s)) publish(p,8+e,ep); }
    }
  }
  // Observable epilogue after trigger (also catches stale epochs in replay).
  __syncthreads();
  if(threadIdx.x==0) p.combined[3+blockIdx.x]=ep;
}
__global__ __launch_bounds__(NT,2) void pure_stream(Params p) {
  __shared__ Shared s; save(p,0,stream(p.arena,p.pure[blockIdx.x]),s);
}
__device__ __forceinline__ unsigned mix(unsigned v) {
  v^=v>>16; v*=0x7feb352du; v^=v>>15; v*=0x846ca68bu; return v^(v>>16);
}
__global__ void init(unsigned* p,size_t n,unsigned seed) {
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) p[i]=mix((unsigned)i^seed);
}
__global__ void gate(unsigned long long ns) { if(threadIdx.x==0) { auto t=now(); while(now()-t<ns) __nanosleep(64); } }
} // namespace ms
#ifndef MSKEL_CUBIN
extern "C" int mskel_params_size() { return sizeof(ms::Params); }
extern "C" int mskel_init(void* dst,size_t bytes,unsigned seed,void* stream) {
  ms::init<<<132,256,0,(cudaStream_t)stream>>>((unsigned*)dst,bytes/4,seed); return cudaGetLastError();
}
extern "C" int mskel_gate(void* stream) {
  ms::gate<<<1,32,0,(cudaStream_t)stream>>>(300000ull); return cudaGetLastError();
}
static const void* kernel(int arm,int marks) {
  if(arm==0) return marks?(const void*)ms::persistent<true,true>:(const void*)ms::persistent<true,false>;
  if(arm==1) return marks?(const void*)ms::persistent<false,true>:(const void*)ms::persistent<false,false>;
  return (const void*)ms::pure_stream;
}
extern "C" int mskel_resources(int arm,int marks,int* values) {
  const void* fn=kernel(arm,marks); cudaFuncAttributes a{}; int occ=0;
  cudaError_t e=cudaFuncGetAttributes(&a,fn); if(e!=cudaSuccess) return e;
  e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,fn,256,0);
  values[0]=a.numRegs; values[1]=(int)a.sharedSizeBytes; values[2]=(int)a.localSizeBytes; values[3]=occ; return e;
}
extern "C" int mskel_launch(ms::Params p,int arm,int marks,void* stream) {
  if(arm<2) {
    void* args[]={&p}; return cudaLaunchCooperativeKernel(kernel(arm,marks),p.g,256,args,0,(cudaStream_t)stream);
  }
  if(arm==3) { ms::pure_stream<<<p.g,256,0,(cudaStream_t)stream>>>(p); return cudaGetLastError(); }
  cudaLaunchAttribute attr{}; attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t cfg{}; cfg.gridDim=dim3(p.g); cfg.blockDim=dim3(256); cfg.stream=(cudaStream_t)stream;
  cfg.attrs=&attr; cfg.numAttrs=1;
  #define NODE(PH) { cudaError_t e=cudaLaunchKernelEx(&cfg,ms::chain<PH>,p); if(e!=cudaSuccess) return e; }
  NODE(0) NODE(1) NODE(2) NODE(3) NODE(4) NODE(7) NODE(8) NODE(23) NODE(24)
  #undef NODE
  return cudaSuccess;
}
#else
// Explicit instantiation forces cubin-only compilation of EVERY runtime route.
template __global__ void ms::persistent<true,true>(ms::Params);
template __global__ void ms::persistent<true,false>(ms::Params);
template __global__ void ms::persistent<false,true>(ms::Params);
template __global__ void ms::persistent<false,false>(ms::Params);
#define INST(PH) template __global__ void ms::chain<PH>(ms::Params);
INST(0) INST(1) INST(2) INST(3) INST(4) INST(7) INST(8) INST(23) INST(24)
#undef INST
#endif
'''


class Params(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in
                ("arena", "pool", "indices", "desc", "pure", "counters", "ready",
                 "epoch", "error", "checks", "partial", "combined", "norms", "marks")]
    _fields_ += [(name, C.c_int) for name in ("g", "experts", "pf_cap")]


def cuda_source():
    """The sole generator; compile-only and runtime use this exact source."""
    return CUDA


def prepare(plan, alloc, payload):
    """Shared real/dry allocation path. There is no base-layer state or module."""
    g = plan["grid"]
    state = dict(payload)
    state.update(desc=alloc((SLOTS, g, 2), "i64", plan["desc"]),
                 pure=alloc((g, 2), "i64", plan["pure"]),
                 counters=alloc((2, COUNTERS), "i32"), ready=alloc((2, COUNTERS), "i32"),
                 epoch=alloc((g,), "i32"), error=alloc((1,), "i32"),
                 checks=alloc((SLOTS, g), "i32"), partial=alloc((g,), "i32"),
                 combined=alloc((g + 3,), "i32"), norms=alloc((2, g), "f32"),
                 marks=alloc((SLOTS, g, 2), "i64"))
    state["params"] = Params(*(int(state[n].data_ptr()) for n, _ in Params._fields_[:14]),
                             g, plan["experts"], plan["pf_per_cta"])
    state["plan"] = plan
    return state


def select(bank, arm, grid, marked=False):
    if arm not in ARM_IDS or (marked and arm not in ("A", "A0")):
        raise ValueError("invalid arm/marks selection")
    state = bank[(grid, arm, marked)]
    return state["params"], ARM_IDS[arm], int(marked)


def dry_run():
    """Exercise the actual prepare/bank/selector code, exact covers and replay ABA."""
    class Fake:
        serial = 0
        def __init__(self, shape, dtype, data=None):
            self.shape, self.dtype, self.data = shape, dtype, data
            Fake.serial += 1
            self.ptr = Fake.serial * 256
        def data_ptr(self):
            return self.ptr
    checked = []
    for t in (1, 8):
        bank = {}
        for g in (132, 264):
            p = make_plan(t, g)
            for ph, b in p["sizes"].items():
                spans = sorted((o, n) for o, n in p["desc"][ph] if n)
                cursor = p["ranges"][str(ph)][0]
                for o, n in spans:
                    assert o == cursor
                    cursor += n
                assert sum(n for _, n in spans) * 16 == b
                assert max(n for _, n in spans) - min(n for _, n in spans) <= 1
            assert sum(n for _, n in p["desc"][1]) == ATTN_ROWS
            for head in range(2):
                rows = [r for c in range(head, g, 2) for r in range(p["desc"][1][c][0], sum(p["desc"][1][c]))]
                assert rows == list(range(head * 4096, (head + 1) * 4096))
            assert sum(n for _, n in p["pure"]) * 16 == p["total_bytes"]
            assert p["pf_per_cta"] * g <= PREFETCH_CAP
            # Thread-strided reductions must include every j, especially G264.
            assert sorted(j for tid in range(THREADS) for j in range(tid, g, THREADS)) == list(range(g))
            payload = {k: Fake((1,), "i32") for k in ("arena", "pool", "indices")}
            for arm, mk in [(a, False) for a in ARM_IDS] + [("A", True), ("A0", True)]:
                bank[(g, arm, mk)] = prepare(p, Fake, payload)
                params, ai, mi = select(bank, arm, g, mk)
                assert params.g == g and ai == ARM_IDS[arm] and mi == mk
                assert params.experts == (2 if t == 1 else 8)
            # Model last-arriver reset and stale token rejection for 100 replays.
            ready = [[0] * COUNTERS for _ in range(2)]
            counts = [[0] * COUNTERS for _ in range(2)]
            rng = random.Random(13)
            for ep in range(1, 101):
                for slot, n in [(0, g), (1, g), (2, g), (3, g), (4, g), (5, g), (6, 32)] + [(8+e, g) for e in range(p["experts"])]:
                    b = ep & 1
                    assert ready[b][slot] != ep and counts[b][slot] == 0
                    order = list(range(n)); rng.shuffle(order)
                    for i, _ in enumerate(order):
                        old = counts[b][slot]; counts[b][slot] += 1
                        if old == n - 1:
                            counts[b][slot] = 0
                            ready[b][slot] = ep
                        assert (ready[b][slot] == ep) == (i == n - 1)
            checked.append(dict(T=t, G=g, total_bytes=p["total_bytes"], experts=p["experts"]))
        try:
            select(bank, "missing", 132)
            raise AssertionError("bad selector accepted")
        except ValueError:
            pass
    assert "__trap" not in CUDA
    assert "-gencode=arch=compute_90a,code=sm_90a" in CUDA_FLAGS
    assert len(PHASES) == SLOTS
    return dict(status="PASS", cases=checked, prepare_paths=24, replay_epochs=100,
                layer_state_accesses=0, generated_cuda_sha256=hashlib.sha256(cuda_source().encode()).hexdigest())


def build_shared(directory, timeout=150):
    directory.mkdir(parents=True, exist_ok=True)
    src, lib = directory / "mskel.cu", directory / "mskel.so"
    src.write_text(cuda_source(), encoding="utf-8", newline="\n")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc missing; run under tools/local/cudaenv.sh or an H200 CUDA development image")
    command = [nvcc, *CUDA_FLAGS, "--shared", "-Xcompiler=-fPIC", "--cudart=shared", str(src), "-o", str(lib)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("nvcc failed: " + result.stderr[-6000:])
    dll = C.CDLL(str(lib.resolve()))
    dll.mskel_launch.argtypes = [Params, C.c_int, C.c_int, C.c_void_p]
    dll.mskel_launch.restype = C.c_int
    dll.mskel_resources.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_int)]
    dll.mskel_resources.restype = C.c_int
    dll.mskel_init.argtypes = [C.c_void_p, C.c_size_t, C.c_uint, C.c_void_p]
    dll.mskel_init.restype = C.c_int
    dll.mskel_gate.argtypes = [C.c_void_p]
    dll.mskel_gate.restype = C.c_int
    if dll.mskel_params_size() != C.sizeof(Params):
        raise RuntimeError("C/Python Params ABI mismatch")
    return dll, command


def check_cuda(code):
    if code:
        raise RuntimeError(f"CUDA runtime error {code}; aborting without another replay")


def expected_checks(plan, indices):
    """Independent CPU oracle for EVERY vector read; never used by a timed call."""
    import numpy as np
    def checksum(off, n, seed):
        v = np.arange(off * 4, (off + n) * 4, dtype=np.uint32) ^ np.uint32(seed)
        v ^= v >> 16; v *= np.uint32(0x7feb352d)
        v ^= v >> 15; v *= np.uint32(0x846ca68b); v ^= v >> 16
        return int(np.bitwise_xor.reduce(v, initial=np.uint32(0)))
    g = plan["grid"]
    out = np.zeros((SLOTS, g), dtype=np.uint32)
    for ph in plan["sizes"]:
        for c, (off, n) in enumerate(plan["desc"][ph]):
            out[ph, c] = checksum(off, n, 71)
    for c, (off, n) in enumerate(plan["desc"][1]):
        x = 0
        for row in indices[off:off+n]:
            x ^= checksum(row*8, 8, 113)
        out[1, c] = x
    sums = np.array([sum(j+1 for j in range(tid, g, THREADS)) for tid in range(THREADS)], dtype=np.float32)
    bnd = np.bitwise_xor.reduce(sums.view(np.uint32))
    out[3, :] = bnd; out[24, :] = bnd
    pure = np.array([checksum(o, n, 71) for o, n in plan["pure"]], dtype=np.uint32)
    return out, pure


def validate(state, expected, pure, arm):
    import numpy as np
    if int(state["error"].item()):
        raise RuntimeError(f"bounded wait failed, counter/error id {state['error'].item()}")
    if np.any(state["counters"].cpu().numpy()):
        raise RuntimeError("arrival counters were not reset")
    got = state["checks"].cpu().numpy().view(np.uint32)
    want = expected if arm != "S" else np.vstack([pure, np.zeros_like(expected[1:])])
    if not np.array_equal(got, want):
        at = np.argwhere(got != want)[0].tolist()
        raise RuntimeError(f"checksum mismatch arm={arm}, [phase,cta]={at}: {got[tuple(at)]} != {want[tuple(at)]}")
    if arm != "S":
        combined = state["combined"].cpu().numpy().view(np.uint32)
        attn = [int(np.bitwise_xor.reduce(expected[1, h::2])) for h in (0, 1)]
        top = int(np.bitwise_xor.reduce(np.sort(expected[5, :32])[-8:]))
        if list(combined[:3]) != attn + [top]:
            raise RuntimeError("last-arriver attention/top8 combine mismatch")
        epoch = state["epoch"].cpu().numpy().view(np.uint32)
        if not np.all(epoch == epoch[0]):
            raise RuntimeError("CTA epoch disagreement")
        if arm == "B" and not np.all(combined[3:] == epoch):
            raise RuntimeError("PDL epilogue observed a stale epoch")


def timeline(marks, plan):
    starts = [int(v) for v in marks[0, :, 0] if v]
    origin = min(starts)
    rows = {}
    for ph, name in enumerate(PHASES):
        active = [(int(a), int(b)) for a, b in marks[ph] if a and b]
        if active:
            assert all(b >= a for a, b in active)
            rows[name] = dict(first_start_us=(min(a for a, _ in active)-origin)/1000,
                              last_end_us=(max(b for _, b in active)-origin)/1000,
                              active_ctas=len(active))
    return rows


def decision(rows):
    # Pre-register per-grid decisions. Do not pick a different G for each shape.
    result = {}
    for g in (132, 264):
        a = {(r["T"]): r["median_us"] for r in rows if r["arm"] == "A" and r["G"] == g and not r["marked"]}
        if len(a) != 2:
            result[str(g)] = "PENDING"
        elif a[1] > 52:
            result[str(g)] = "NO_BUILD: T1 > 52 us"
        elif a[1] <= 44 and a[8] <= 78:
            result[str(g)] = "WORTH_BUILDING: both registered skeleton thresholds pass"
        else:
            result[str(g)] = "INCONCLUSIVE: neither registered rule passed"
    return result


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")


def run(args, report):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("H200 required for measurements; --dry is CPU-only")
    dev = torch.cuda.get_device_properties(torch.cuda.current_device())
    if "H200" not in dev.name or dev.major != 9 or dev.multi_processor_count != 132:
        raise RuntimeError(f"requires a 132-SM H200, got {dev.name} / {dev.multi_processor_count} SMs")
    report["device"] = dict(name=dev.name, sms=dev.multi_processor_count, torch=torch.__version__, cuda=torch.version.cuda)
    dll, command = build_shared(Path(args.out).parent / "build")
    report["compile_command"] = command
    resources = {}
    for arm in ("A", "A0", "S"):
        for mk in ((False, True) if arm != "S" else (False,)):
            vals = (C.c_int * 4)()
            check_cuda(dll.mskel_resources(ARM_IDS[arm], mk, vals))
            resources[f"{arm}/marks={mk}"] = dict(zip(("regs", "smem_bytes", "local_bytes", "ctas_per_sm"), vals))
            if arm != "S" and vals[3] < 2:
                raise RuntimeError("G264 cannot be safely resident; resource gate failed")
    report["resources"] = resources
    stream = torch.cuda.Stream()
    flush = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    flush_sink = torch.empty((), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    def alloc(shape, dtype, data=None):
        ty = {"i32": torch.int32, "i64": torch.int64, "f32": torch.float32}[dtype]
        return torch.zeros(shape, dtype=ty, device="cuda") if data is None else torch.tensor(data, dtype=ty, device="cuda")
    rng = random.Random(20261102)
    indices = rng.sample(range(POOL_BYTES // 128), ATTN_ROWS)
    for t in (1, 8):
        p132 = make_plan(t, 132)
        with torch.cuda.stream(stream):
            arena = torch.empty(p132["total_bytes"], dtype=torch.uint8, device="cuda")
            pool = torch.empty(POOL_BYTES, dtype=torch.uint8, device="cuda")
            check_cuda(dll.mskel_init(arena.data_ptr(), arena.numel(), 71, stream.cuda_stream))
            check_cuda(dll.mskel_init(pool.data_ptr(), pool.numel(), 113, stream.cuda_stream))
            payload = dict(arena=arena, pool=pool, indices=alloc((ATTN_ROWS,), "i32", indices))
            bank = {}
            for g in (132, 264):
                p = make_plan(t, g)
                for arm, mk in [(a, False) for a in ARM_IDS] + [("A", True)]:
                    bank[(g, arm, mk)] = prepare(p, alloc, payload)
        stream.synchronize()
        expected = {g: expected_checks(make_plan(t, g), indices) for g in (132, 264)}
        graphs, samples, timelines, raw_marks = {}, {}, {}, {}
        # Eager/graph checksum checks exercise both parity banks BEFORE timing.
        for key, state in bank.items():
            g, arm, mk = key
            with torch.cuda.stream(stream):
                for _ in range(3):
                    check_cuda(dll.mskel_launch(*select(bank, arm, g, mk), stream.cuda_stream))
            stream.synchronize()
            validate(state, *expected[g], arm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                check_cuda(dll.mskel_launch(*select(bank, arm, g, mk), stream.cuda_stream))
            graphs[key], samples[key] = graph, []
            timelines[key], raw_marks[key] = [], []
        keys = list(bank)
        for rep in range(args.replays + 3):
            # Deterministic rotating order avoids assigning drift to an arm.
            order = keys[rep % len(keys):] + keys[:rep % len(keys)]
            for key in order:
                state = bank[key]
                with torch.cuda.stream(stream):
                    torch.sum(flush, dim=0, out=flush_sink) # reads all 256 MiB
                    check_cuda(dll.mskel_gate(stream.cuda_stream)) # globaltimer 300 us
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(stream); graphs[key].replay(); end.record(stream)
                end.synchronize()
                if int(state["error"].item()):
                    raise RuntimeError(f"wait timeout during replay {rep} arm {key}")
                if rep >= 3:
                    samples[key].append(start.elapsed_time(end) * 1000)
                    if key[2]:
                        marks = state["marks"].cpu().numpy()
                        timelines[key].append(timeline(marks, state["plan"]))
                        raw_marks[key].append(marks.tolist())
                if rep in (0, 1, 2, args.replays + 2):
                    validate(state, *expected[key[0]], key[1])
        for key in keys:
            g, arm, mk = key
            row = dict(T=t, G=g, arm=arm, marked=mk, median_us=statistics.median(samples[key]),
                       min_us=min(samples[key]), max_us=max(samples[key]), samples_us=samples[key],
                       replays=len(samples[key]), total_bytes=bank[key]["plan"]["total_bytes"], label="MEASURED")
            if mk:
                names = timelines[key][0]
                row["timeline_us"] = {n: {f: statistics.median(x[n][f] for x in timelines[key])
                                           for f in ("first_start_us", "last_end_us", "active_ctas")} for n in names}
                row["globaltimer_marks_ns"] = raw_marks[key]
                row["timeline_replays"] = timelines[key]
            report["rows"].append(row)
        report["decision"] = decision(report["rows"])
        write_json(Path(args.out), report)
        # Drop references, never reinitialize or reset state between replays.
        graphs.clear(); bank.clear()
    report["status"] = "PASS"
    print("| T | G | arm | marks | MEASURED median us |")
    print("|---|---|---|---|---|")
    for r in report["rows"]:
        print(f"| {r['T']} | {r['G']} | {r['arm']} | {r['marked']} | {r['median_us']:.3f} |")
    for r in report["rows"]:
        if r["marked"]:
            print(f"\nA timeline T={r['T']} G={r['G']} (separate marked graph):")
            print("| phase | MEASURED first start us | MEASURED last end us | active CTAs |")
            print("|---|---|---|---|")
            for n, v in r["timeline_us"].items():
                print(f"| {n} | {v['first_start_us']:.3f} | {v['last_end_us']:.3f} | {v['active_ctas']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="box_out/mskel/run.json")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--replays", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=450)
    args = ap.parse_args()
    if args.dry:
        print(json.dumps(dry_run(), indent=2)); return
    if args.replays < 30 or not 1 <= args.timeout <= 470:
        ap.error("at least 30 replays, wall timeout in [1,470] seconds")
    report = dict(status="RUNNING", cpu_gate=dry_run(), rows=[], decision={},
                  protocol=dict(flush_read_bytes=FLUSH_BYTES, spin_gate_ns=300000, replays=args.replays,
                                events="outside captured graph, flush/gate excluded", empty_graph_subtraction=False,
                                marks="separate graphs, no marks in primary decision", seed=20261102),
                  budgets={str(t): {k: make_plan(t, 132)[k] for k in ("weight_bytes", "attention_bytes", "total_bytes", "fractions")}
                           for t in (1, 8)})
    t0 = time.monotonic()
    def expired():
        # Handles a hung CUDA call as well as compilation. No trap, no GPU reset.
        report["status"] = "TIMEOUT"; report["elapsed_s"] = time.monotonic()-t0
        write_json(Path(args.out), report)
        os._exit(124)
    watchdog = threading.Timer(args.timeout, expired); watchdog.daemon = True; watchdog.start()
    try:
        run(args, report)
    except BaseException as exc:
        report["status"] = "FAIL"; report["error"] = repr(exc)
        raise
    finally:
        watchdog.cancel(); report["elapsed_s"] = time.monotonic()-t0
        write_json(Path(args.out), report)


if __name__ == "__main__":
    main()

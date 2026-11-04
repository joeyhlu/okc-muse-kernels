"""S6A2: anchored, default-off register-fed MoE build; no CUDA execution here.

python -B tools/s6a2s_patches.py --check
python -B tools/s6a2s_patches.py --ship --check
The lab bank is requested with S6A2_BUILD=True before prepare(). The two
selectors may then be changed independently between captures. Default prepare
does not build the bank. --ship removes the obsolete pre-TINYS source, not T1.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"


def sub(s, old, new, count=1):
    if s.count(old) != count:
        raise ValueError(f"S6A2 anchor count {s.count(old)} != {count}: {old[:100]!r}")
    return s.replace(old, new)


CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
namespace s6a2s {
struct Params {
 const __nv_bfloat16 *x, *nw;
 const int *ids; const float *tw;
 const unsigned *mask;
 const unsigned char *wa, *wb;
 const float *ga, *gb;
 __nv_bfloat16 *h, *y, *out;
 int m, grid;
};
__device__ __forceinline__ unsigned opaque(unsigned v) {
 unsigned r; asm("mov.b32 %0, %1;" : "=r"(r) : "r"(v)); return r;
}
template<bool HI> __device__ __forceinline__ unsigned mul(unsigned a, unsigned s) {
 const __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&s);
 const __nv_bfloat162 f = HI ? __high2bfloat162(v) : __low2bfloat162(v);
 const __nv_bfloat162 o = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), f);
 return *reinterpret_cast<const unsigned*>(&o);
}
// The MMA and the weight/scale loads are macros, not inline functions: the hot SASS rows
// then carry exactly one source frame (the statement inside s6a2s_common_body). With an
// inlined helper frame nvdisasm -gi dropped the pseudo-file name for whole chunk
// instances (24 of 32 MMAs attributed), so the gate's line map was incomplete.
#define S6A2S_MMA(c, a, b, d, e, x, y) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
 : "+f"((c)[0]), "+f"((c)[1]), "+f"((c)[2]), "+f"((c)[3]) \
 : "r"(a), "r"(b), "r"(d), "r"(e), "r"(x), "r"(y))
__device__ __forceinline__ int nth(unsigned bits, int n) {
 for (int j=0; j<n; ++j) bits &= bits-1;
 return __ffs(bits)-1;
}
__device__ __forceinline__ unsigned membership(Params p, int lane) {
 unsigned bits=0;
 if (lane < p.m) {
  if (p.mask) bits=__ldg(p.mask+lane);
  else {
   #pragma unroll
   for (int s=0;s<8;++s) {
    const int e=__ldg(p.ids+lane*8+s);
    if (unsigned(e)<32u) bits |= 1u<<e;
   }
  }
 }
 return bits;
}
// Each lane receives four words for a K=64 chunk. Adjacent lane groups share
// a scale across all four words (the shipped KPERM3 physical-K permutation).
// wt / st point at this lane's chunk 0 of the current tile row; every chunk is a
// compile-time immediate offset from them (row stride 36864 B = 2304 uint4 =
// 9216 words; chunk stride 512 B = 32 uint4; scale block at +32768 B). Same
// addresses as before: base+row*36864+cc*512+lane*16, scales (cc/2)*32+lane.
// The predicated load writes fresh registers and leaves them undefined when ok==0:
// a row >= end is only ever prefetched, never consumed (the tile loop exits
// first), so no zero fill and no copy of the old value is needed.
#define S6A2S_WLOAD(v, wt, ok, off) asm volatile("{\n .reg .pred q;\n setp.ne.b32 q,%5,0;\n @q ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4+%6];\n}" \
 : "=r"((v).x), "=r"((v).y), "=r"((v).z), "=r"((v).w) : "l"(wt), "r"(int(ok)), "n"(off))
#define S6A2S_SLOAD(v, st, ok, off) asm volatile("{\n .reg .pred q;\n setp.ne.b32 q,%2,0;\n @q ld.global.nc.u32 %0,[%1+%3];\n}" \
 : "=r"(v) : "l"(st), "r"(int(ok)), "n"(off))
// Prologue form of the same loads (outside the common body).
template<int N, int C> __device__ __forceinline__ uint4 wload(const uint4 *wt, int ok) {
 constexpr int OFF=((C/(64/N))*2304+(C%(64/N))*32)*16;
 uint4 v;
 S6A2S_WLOAD(v, wt, ok, OFF);
 return v;
}
template<int N, int C> __device__ __forceinline__ unsigned sload(const unsigned *st, int ok) {
 static_assert(C%2==0, "paired scale load");
 // The two K=64 scale pairs are adjacent for one lane: one LDG.32 per 2 chunks.
 constexpr int OFF=((C/(64/N))*9216+((C%(64/N))/2)*32)*4;
 unsigned v;
 S6A2S_SLOAD(v, st, ok, OFF);
 return v;
}
template<int N, bool PB>
__global__ void __launch_bounds__(N*32, 16/N) mat(Params p) {
 constexpr int NW=N, CP=64/N, NH=N/8, ROWS=PB?256:512;
 __shared__ float4 red[NW][NH][32];
 const int tid=threadIdx.x, warp=tid>>5, lane=tid&31, g=lane>>2, t=lane&3;
 if(tid==0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 asm volatile("griddepcontrol.wait;" ::: "memory");
 const unsigned my=membership(p,lane);
 const unsigned active=__reduce_or_sync(0xffffffffu,my);
 const int jobs=__popc(active)*ROWS;
 const int begin=jobs*int(blockIdx.x)/p.grid, end=jobs*(int(blockIdx.x)+1)/p.grid;
 // A whole CTA may have no work. No global counters, reset obligations, or
 // partial-CTA barriers follow this return; the next launch is the dependency.
 if (begin==end) return;
 uint4 xr[16]; // 64 32-bit input registers; ALL uses below have constant indices.
 for (int first=begin; first<end;) {
  const int rank=first/ROWS, e=nth(active,rank), stop=min(end,(rank+1)*ROWS);
  const int tile0=first%ROWS, tile1=stop-rank*ROWS;
  const unsigned tokens=__ballot_sync(0xffffffffu,(my & (1u<<e))!=0);
  const int count=__popc(tokens);
  const unsigned char *base=(PB?p.wb:p.wa)+size_t(e)*ROWS*36864;
  #pragma unroll
  for (int h=0;h<NH;++h) {
   const int token=nth(tokens,g+8*h);
   const __nv_bfloat16 *src=PB ? p.h+(size_t(e)*N+max(token,0))*4096
                             : p.x+size_t(max(token,0))*4096;
   #pragma unroll
   for (int c=0;c<CP;++c) {
    const uint4 *v=reinterpret_cast<const uint4*>(src+warp*CP*64+c*64+t*16);
    xr[h*CP*2+c*2]=g+8*h<count ? __ldg(v) : make_uint4(0,0,0,0);
    xr[h*CP*2+c*2+1]=g+8*h<count ? __ldg(v+1) : make_uint4(0,0,0,0);
   }
  }
  const uint4 *wb=reinterpret_cast<const uint4*>(base)+warp*CP*32+lane;
  const unsigned *sb=reinterpret_cast<const unsigned*>(base+32768)+warp*(CP/2)*32+lane;
  const uint4 *wt0=wb+size_t(tile0)*2304;
  const unsigned *st0=sb+size_t(tile0)*9216;
  uint4 w0=wload<N,0>(wt0,tile0+0/CP<tile1);
  uint4 w1=wload<N,1>(wt0,tile0+1/CP<tile1);
  uint4 w2=wload<N,2>(wt0,tile0+2/CP<tile1);
  uint4 w3=wload<N,3>(wt0,tile0+3/CP<tile1);
  uint4 w4=wload<N,4>(wt0,tile0+4/CP<tile1);
  unsigned s0=sload<N,0>(st0,tile0+0/CP<tile1);
  unsigned s1=sload<N,2>(st0,tile0+2/CP<tile1);
  unsigned s2=sload<N,4>(st0,tile0+4/CP<tile1);
  for (int tile=tile0;tile<tile1;++tile) {
   float acc[NH][4]={};
#line 6100 "s6a2s_common_body"
   const uint4 *wt=wb+size_t(tile)*2304;
   const unsigned *st=sb+size_t(tile)*9216;
   #pragma unroll
   for (int c=0;c<CP;++c) {
    const unsigned sc=__byte_perm(s0,0u,(c&1)?0x4342u:0x4140u)*16u+0x5A805A80u;
    const unsigned ww[4]={w0.x,w0.y,w0.z,w0.w};
    #pragma unroll
    for (int s=0;s<4;++s) {
     const unsigned q=ww[s], M=0x81C081C0u;
     const unsigned r0=q&M, r1=__funnelshift_l(q,q,10)&M;
     const unsigned r2=__funnelshift_l(q,q,13)&M;
     const unsigned r3=((q&0x001B001Bu)*opaque(0x1084u))&M;
     const unsigned a0=mul<false>(r0,sc), a1=mul<true>(r1,sc);
     const unsigned a2=mul<false>(r2,sc), a3=mul<true>(r3,sc);
     #pragma unroll
     for (int h=0;h<NH;++h) {
      const uint4 a=xr[h*CP*2+c*2], b=xr[h*CP*2+c*2+1];
      const unsigned bb[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
      S6A2S_MMA(acc[h],a0,a1,a2,a3,bb[s*2],bb[s*2+1]);
     }
    }
    // Rotation is straight-line scalar renaming, never a dynamic local array.
    w0=w1; w1=w2; w2=w3; w3=w4;
    S6A2S_WLOAD(w4,wt,tile+(c+5)/CP<tile1,(((c+5)/CP)*2304+((c+5)%CP)*32)*16);
    if constexpr (c%2==1) {
     s0=s1; s1=s2; S6A2S_SLOAD(s2,st,tile+(c+5)/CP<tile1,(((c+5)/CP)*9216+(((c+5)%CP)/2)*32)*4);
    }
   }
#line 6200 "s6a2s_reduce_epilogue"
   #pragma unroll
   for (int h=0;h<NH;++h) red[warp][h][lane]=make_float4(acc[h][0],acc[h][1],acc[h][2],acc[h][3]);
   __syncthreads();
   // Every warp owns one token; all four SMSPs participate in the short tail. The owner
   // search (a runtime-length bit loop) runs only inside the guard: warps without a
   // token skip it instead of paying about 60 instructions per tile.
   if (warp<count && lane<(PB?16:8)) {
    const int token=nth(tokens,warp);
    const int producer=(lane&7)*4+(warp&7)/2;
    float4 v=red[0][warp/8][producer];
    // K partitions are still summed strictly in order 1..NW-1 (bitwise the same tree
    // as a plain loop), but at most four 16 B loads are in flight per thread: the
    // __syncwarp between groups stops ptxas from hoisting all NW-1 loads (up to
    // 60 live registers) above the adds, which spilled the prefetched weights.
    #pragma unroll
    for (int k=1;k<NW;k+=4) {
     float4 u[4];
     #pragma unroll
     for (int j=0;j<4;++j) if (k+j<NW) u[j]=red[k+j][warp/8][producer];
     #pragma unroll
     for (int j=0;j<4;++j) if (k+j<NW) { v.x+=u[j].x; v.y+=u[j].y; v.z+=u[j].z; v.w+=u[j].w; }
     __syncwarp(PB?0xffffu:0xffu); // exactly the lanes inside the guard below
    }
    if constexpr (PB) {
     const float z=lane<8 ? ((warp&1)?v.y:v.x) : ((warp&1)?v.w:v.z);
     p.y[(size_t(e)*N+token)*4096+tile*16+lane]=__float2bfloat16_rn(z*__ldg(p.gb+e));
    } else {
     const float scale=__ldg(p.ga+e);
     const float gate=__bfloat162float(__float2bfloat16_rn(((warp&1)?v.y:v.x)*scale));
     const float up=__bfloat162float(__float2bfloat16_rn(((warp&1)?v.w:v.z)*scale));
     // SwiGLU as in the shipped decode kernels (SILU_DIV = __fdividef, __expf). An IEEE
     // divide compiles to a CALL into the slow-path routine, which makes ptxas spill the
     // live prefetch registers around it.
     p.h[(size_t(e)*N+token)*4096+tile*8+lane]=__float2bfloat16_rn(__fdividef(gate,1.f+__expf(-gate))*up);
    }
   }
   __syncthreads(); // all N*32 threads; protects red before the next tile reuses it
  }
  first=stop;
 }
}
#line 6300 "s6a2s_finalize"
__global__ void __launch_bounds__(256,1) finish(Params p) {
 __shared__ float sums[8];
 const int token=blockIdx.x, tid=threadIdx.x, lane=tid&31, warp=tid>>5;
 if(tid==0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
 asm volatile("griddepcontrol.wait;" ::: "memory");
 float out[16]={};
 #pragma unroll 1
 for (int slot=0;slot<8;++slot) {
  const int e=__ldg(p.ids+token*8+slot);
  if (unsigned(e)<32u) {
   float v[16], ss=0;
   #pragma unroll
   for (int i=0;i<16;++i) {
    const int col=tid+i*256;
    v[i]=__bfloat162float(__ldg(p.y+(size_t(e)*p.m+token)*4096+col))*__bfloat162float(__ldg(p.nw+col));
    ss=fmaf(v[i],v[i],ss);
   }
   #pragma unroll
   for(int off=16;off;off>>=1) ss+=__shfl_xor_sync(0xffffffffu,ss,off);
   if(lane==0) sums[warp]=ss;
   __syncthreads();
   float total=0;
   #pragma unroll
   for(int w=0;w<8;++w) total+=sums[w];
   const float factor=__ldg(p.tw+token*8+slot)*rsqrtf(total*(1.f/4096.f)+1.e-8f);
   #pragma unroll
   for(int i=0;i<16;++i) out[i]+=v[i]*factor;
   __syncthreads(); // all 256 threads, including tokens with zero local experts
  }
 }
 #pragma unroll
 for(int i=0;i<16;++i) p.out[size_t(token)*4096+tid+i*256]=__float2bfloat16_rn(out[i]);
}
} // namespace
#ifdef S6A2_OFFLINE
#if S6A2_ONLY_N == 8
template __global__ void s6a2s::mat<8,false>(s6a2s::Params);
template __global__ void s6a2s::mat<8,true>(s6a2s::Params);
#else
template __global__ void s6a2s::mat<16,false>(s6a2s::Params);
template __global__ void s6a2s::mat<16,true>(s6a2s::Params);
#endif
#else
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
void s6a2s_run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor mask,
 torch::Tensor wa, torch::Tensor wb, torch::Tensor ga, torch::Tensor gb, torch::Tensor nw,
 torch::Tensor h, torch::Tensor y, torch::Tensor out, int64_t nsm) {
 const int n=x.size(0), grid=nsm*(n==8?2:1);
 TORCH_CHECK(n==8 || n==16, "S6A2 exact T=8/16 only");
 TORCH_CHECK(x.is_contiguous() && ids.is_contiguous() && tw.is_contiguous(), "S6A2 layout");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && ids.scalar_type()==at::kInt && tw.scalar_type()==at::kFloat, "S6A2 dtype");
 s6a2s::Params p{(const __nv_bfloat16*)x.data_ptr(),(const __nv_bfloat16*)nw.data_ptr(),ids.data_ptr<int>(),tw.data_ptr<float>(),
 mask.numel()?(const unsigned*)mask.data_ptr():nullptr,wa.data_ptr<unsigned char>(),wb.data_ptr<unsigned char>(),
 ga.data_ptr<float>(),gb.data_ptr<float>(),(__nv_bfloat16*)h.data_ptr(),(__nv_bfloat16*)y.data_ptr(),(__nv_bfloat16*)out.data_ptr(),n,grid};
 auto stream=at::cuda::getCurrentCUDAStream();
 // Same-stream PDL: the device wait enforces complete A -> B -> finish.
 // Launch attributes follow the existing vms host launcher exactly.
 cudaLaunchAttribute at[1];
 at[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
 at[0].val.programmaticStreamSerializationAllowed=1;
 cudaLaunchConfig_t lc={};
 lc.gridDim=dim3(grid); lc.blockDim=dim3(n*32); lc.dynamicSmemBytes=0;
 lc.stream=stream; lc.attrs=at; lc.numAttrs=1;
 if(n==8) {
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc,s6a2s::mat<8,false>,p));
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc,s6a2s::mat<8,true>,p));
 } else {
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc,s6a2s::mat<16,false>,p));
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc,s6a2s::mat<16,true>,p));
 }
 lc.gridDim=dim3(n); lc.blockDim=dim3(256);
 C10_CUDA_CHECK(cudaLaunchKernelEx(&lc,s6a2s::finish,p));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif
'''

# C++ loop counters cannot be non-type template arguments, even with unroll.
# Generate the CP-dependent statement sequence at source construction time.
def cuda_source():
    s = CUDA
    start = s.index('   #pragma unroll\n   for (int c=0;c<CP;++c) {', s.index('#line 6100'))
    end = s.index('#line 6200', start)
    body = s[start:end]
    body = body[body.index('    const unsigned sc='):body.rfind('   }')]
    def chunks(n):
        return ''.join('   { constexpr int c='+str(c)+';\n'+body+'   }\n' for c in range(64//n))
    # The only branch is compile-time N, common to every warp of the CTA.
    return s[:start] + '   if constexpr (N==8) {\n' + chunks(8) + '   } else {\n' + chunks(16) + '   }\n' + s[end:]


CPP = '''#include <torch/extension.h>
void s6a2s_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
'''

RUNTIME = r'''
# S6A2_BEGIN: all new globals are inert unless a prepare bank is requested.
S6A2_T16=False
S6A2_T8=False
S6A2_BUILD=False
_S6A2_MOD=None
def _s6a2s_selected(n):
 return (n==16 and S6A2_T16) or (n==8 and S6A2_T8)
def _s6a2s_requested():
 return S6A2_BUILD or S6A2_T16 or S6A2_T8
def _s6a2s_replace(s,a,b,n=1):
 if s.count(a)!=n:
  raise RuntimeError(("S6A2 anchor",a[:80],s.count(a),n))
 return s.replace(a,b)
def _s6a2s_sources():
 # Copy the incumbent tail. Selection, tie handling, and routing are unchanged.
 cu=V8_TAIL_CUDA_SRC
 cu=_s6a2s_replace(cu,"int* __restrict__ lid,\nfloat eps", "int* __restrict__ lid, unsigned* membership,\nfloat eps",3)
 a="const int ls = __ldcg(lstart);\n#pragma unroll\nfor (int k = 0; k < TK; ++k) {"
 cu=_s6a2s_replace(cu,a,a.replace("#pragma unroll","unsigned local_mask=0;\n#pragma unroll"),2)
 a="lid[t * TK + k] = rel < 32 ? rel : -1;\n}"
 cu=_s6a2s_replace(cu,a,a.replace("\n}","\nif (rel<32) local_mask |= 1u<<rel;\n}\nmembership[t]=local_mask;"),2)
 a="const float inv = rscale / (sum + deps);\nif (lane < TK) {"
 cu=_s6a2s_replace(cu,a,a.replace("\nif", "\nunsigned local_bit=0;\nif"))
 a="lid[t * TK + lane] = rel < 32 ? rel : -1;\n}\nreturn;"
 b=a.replace("\n}\nreturn;","\nlocal_bit=rel<32 ? 1u<<rel : 0u;\n}\nconst unsigned local_mask=__reduce_or_sync(0xffffffffu,local_bit);\nif(lane==0) membership[t]=local_mask;\nreturn;")
 cu=_s6a2s_replace(cu,a,b)
 a="torch::Tensor lid, double eps"
 cu=_s6a2s_replace(cu,a,"torch::Tensor lid, torch::Tensor membership, double eps")
 cu=_s6a2s_replace(cu,"lid.data_ptr<int>(), (float)eps", "lid.data_ptr<int>(), (unsigned*)membership.data_ptr<int>(), (float)eps",4)
 cpp=_s6a2s_replace(V8_TAIL_CPP_SRC,a,"torch::Tensor lid, torch::Tensor membership, double eps")
 return S6A2_CUDA+"\n"+cu,S6A2_CPP+"\n"+cpp
def _s6a2s_build(loader=None):
 global _S6A2_MOD
 if _S6A2_MOD is not None:
  return _S6A2_MOD
 if loader is None:
  from torch.utils.cpp_extension import load_inline
  loader=load_inline
 cu,cpp=_s6a2s_sources()
 flags=["-O3","-std=c++20","-lineinfo","-gencode=arch=compute_90a,code=sm_90a","-Xptxas=-v,-warn-spills"]
 tag=hashlib.sha1((cu+cpp+" ".join(flags)).encode()).hexdigest()[:12]
 _S6A2_MOD=loader(name="s6a2s_"+tag,cpp_sources=cpp,cuda_sources=cu,
  functions=["s6a2s_run","v8_tail","v8_router32"],extra_cflags=["-std=c++20"],extra_cuda_cflags=flags,verbose=False)
 return _S6A2_MOD
def _s6a2s_pack(w,sf,gate_up):
 # Reuse exact code/sign/subnormal mapping and the shipped KPERM3 bit encoder.
 n=w.shape[0]
 bp,neg,zero,bad=_k2a_reencode_scales(sf.view(torch.uint8))
 if bad:
  raise ValueError("S6A2 NaN E4M3 scale")
 codes=torch.stack((w.view(torch.uint8)&15,w.view(torch.uint8)>>4),-1).reshape(n,256,16)
 codes=torch.where(neg.unsqueeze(-1),codes^8,codes)
 codes=torch.where(zero.unsqueeze(-1),torch.zeros_like(codes),codes)
 rows=_k2a_tile_row_perm(n,gate_up,device=w.device)
 cu=codes.reshape(n,4096)[rows].contiguous().view(n//16,16,4096)
 words=_vms_pack_words(cu).view(torch.uint8).view(n//16,32768)
 # [tile,chunk-pair,lane,chunk-in-pair,row-half] -> one 32-bit scale load.
 sb=bp[rows].view(n//16,2,8,32,2,4).permute(0,3,2,5,4,1).contiguous().view(n//16,4096)
 return torch.cat((words,sb),dim=1).contiguous()
def _s6a2s_prepare(state,weights,t0):
 if not _s6a2s_requested():
  return
 # These state/weight accessors are copied from the real prepare hook at 11746.
 st=state["k2a33"]["st"]
 w={k:weights[k]for k in("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")}|{"post_norm_w":weights["post_expert_norm"]}
 dev=weights["w13"].device
 if tuple(w["w13"].shape)!=(32,8192,2048) or tuple(w["w2"].shape)!=(32,4096,2048):
  raise ValueError("S6A2 requires 32 local 4096-wide experts")
 if state["v8"] is None:
  raise ValueError("S6A2 requires the exact top-8 tail")
 _pt1_wait("s6a2s")
 mod=_s6a2s_build()
 wa=torch.empty((32,512,36864),dtype=torch.uint8,device=dev)
 wb=torch.empty((32,256,36864),dtype=torch.uint8,device=dev)
 for e in range(32):
  if time.time()-t0>=295:
   raise RuntimeError("S6A2 prepare exceeded 295 s budget; do not enable")
  wa[e]=_s6a2s_pack(w["w13"][e],w["w13_sf"][e],True)
  wb[e]=_s6a2s_pack(w["w2"][e],w["w2_sf"][e],False)
 bank={"mod":mod,"wa":wa,"wb":wb,
  "ga":(w["w13_gs"].float().reshape(32)*float(2**62)).contiguous(),
  "gb":(w["w2_gs"].float().reshape(32)*float(2**62)).contiguous(),
  "nw":(1.0+w["post_norm_w"].float()).to(torch.bfloat16).contiguous(),
  "nsm":torch.cuda.get_device_properties(0).multi_processor_count,"arms":{}}
 for n in (8,16):
  bank["arms"][n]={"h":torch.empty((32,n,4096),dtype=torch.bfloat16,device=dev),
   "y":torch.empty((32,n,4096),dtype=torch.bfloat16,device=dev),
   "mask":torch.empty((0,),dtype=torch.int32,device=dev),"ids":None}
 torch.cuda.synchronize()
 if time.time()-t0>=300:
  raise RuntimeError("S6A2 prepare exceeded 300 s")
 st["s6a2s"]=bank
def _s6a2s_bank(state,n):
 if not _s6a2s_selected(n):
  return None
 # During the incumbent's prepare self-tests no bank exists yet.
 k33=state.get("k2a33")
 if k33 is None or "s6a2s" not in k33["st"]:
  return None
 return k33["st"]["s6a2s"]
def _s6a2s_run(state,x,ids,tw):
 x=x.contiguous()
 if x.data_ptr()%16:
  x=x.clone()
 b=_s6a2s_bank(state,x.shape[0])
 if b is None:
  raise RuntimeError("S6A2 selected without a prepare bank")
 a=b["arms"][x.shape[0]]
 # Standalone calls and alternative exact front paths derive membership in A/B;
 # only use a tail mask when it belongs to this call's exact ids tensor.
 mask=a["mask"] if ids is a["ids"] else torch.empty((0,),dtype=torch.int32,device=x.device)
 a["ids"]=None # consume the tail association; reusing/mutating ids alone cannot reuse a stale mask
 out=torch.empty((x.shape[0],4096),dtype=torch.bfloat16,device=x.device)
 b["mod"].s6a2s_run(x.contiguous(),ids.to(torch.int32).contiguous(),tw.to(torch.float32).contiguous(),mask,
  b["wa"],b["wb"],b["ga"],b["gb"],b["nw"],a["h"],a["y"],out,b["nsm"])
 return out
# S6A2_END
'''


def node_source(s, name):
    node = next(n for n in ast.parse(s).body if isinstance(n, (ast.FunctionDef, ast.Assign)) and
                (getattr(n, 'name', None) == name or any(isinstance(t, ast.Name) and t.id == name for t in getattr(n, 'targets', []))))
    return ast.get_source_segment(s, node)


def edits(s):
    # Clone the proven tail wrapper, changing only module, mask argument and name.
    tail = node_source(s, '_v8_tail')
    tail = sub(tail, 'def _v8_tail(', 'def _s6a2s_tail(')
    tail = sub(tail, ' v8=state["v8"]', ' v8=state["v8"]\n bank=_s6a2s_bank(state,y.shape[0])\n arm=bank["arms"][y.shape[0]]\n membership=torch.empty((y.shape[0],),dtype=torch.int32,device=y.device)')
    tail = sub(tail, 'v8["mod"].v8_tail(', 'bank["mod"].v8_tail(')
    tail = sub(tail, 'xn,tw_,tid_,lid_,', 'xn,tw_,tid_,lid_,membership,')
    tail = sub(tail, ' return xn,tw,tid,lid', ' arm["mask"],arm["ids"]=membership,lid\n return xn,tw,tid,lid')
    block = RUNTIME + '\nS6A2_CUDA=r"""' + cuda_source() + '"""\nS6A2_CPP=r"""' + CPP + '"""\n' + tail + '\n'
    return [
        ('def _v8_tail(y,logits,state,data,ftf=None):\n', 'def _v8_tail(y,logits,state,data,ftf=None):\n if _s6a2s_bank(state,y.shape[0]) is not None:\n  return _s6a2s_tail(y,logits,state,data,ftf)\n'),
        ('def _v8_experts(expert_x,local_ids,topk_weights,state,cfg,tokens):\n', 'def _v8_experts(expert_x,local_ids,topk_weights,state,cfg,tokens):\n if _s6a2s_selected(tokens):\n  if not(cfg["local_experts"]==K2A_E_LOC and cfg["top_k"]==K2A_TOP_K and cfg["n_experts"]==N_EXPERTS):\n   raise ValueError("S6A2 unsupported expert configuration")\n  return _s6a2s_run(state,expert_x,local_ids,topk_weights)\n'),
        (' if VMS_ENABLE:\n', ' if _s6a2s_requested():\n  jobs.append(("s6a2s",_s6a2s_build))\n if VMS_ENABLE:\n'),
        (' _tinys_prepare()\n', ' _tinys_prepare()\n _s6a2s_prepare(state,weights,t0)\n'),
        ('def _v11_k2a33_call(x,local_ids,topk_weights,st):\n', block + '\ndef _v11_k2a33_call(x,local_ids,topk_weights,st):\n'),
    ]


def apply(s, ship=False):
    original = s
    changes = edits(original)
    for a, b in changes:
        s = sub(s, a, b)
    if ship:
        # _hyblo_build uses _TINYS_CUDA in this base. The older complete CUDA
        # string is superseded and unreferenced; deleting it preserves T1.
        old = node_source(s, '_HYBLO_CUDA')
        if s.count('_HYBLO_CUDA') != 1:
            raise ValueError('cannot remove referenced _HYBLO_CUDA in chained base')
        s = sub(s, old + '\n', '')
    ast.parse(s)
    if ship and len(s.encode()) >= 1_000_000:
        raise ValueError(f'SHIP too large: {len(s.encode())}')
    return s


def check_off(base, lab):
    # Exact inverse of every scoped edit, including the prepare-job hook. This
    # is stronger than comparing ASTs: all original source bytes must return.
    restored = lab
    for a, b in reversed(edits(base)):
        restored = sub(restored, b, a)
    assert restored == base, 'selectors-off projection differs from base'
    ns = {}
    tree = ast.parse(lab)
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('S6A2_T8','S6A2_T16','S6A2_BUILD') for t in n.targets):
            exec(compile(ast.Module([n], type_ignores=[]), '<selector>', 'exec'), ns)
    assert not any(ns[k] for k in ('S6A2_T8','S6A2_T16','S6A2_BUILD'))
    return hashlib.sha256(restored.encode()).hexdigest()


def line_changes(base, out):
    return [dict(base=[i+1,j], output=[k+1,l], kind=tag)
            for tag,i,j,k,l in difflib.SequenceMatcher(None,base.splitlines(),out.splitlines(),autojunk=False).get_opcodes() if tag!='equal']


def manifest(out):
    lines=out.splitlines(); tree=ast.parse(out)
    def locations(fragment): return [i+1 for i,l in enumerate(lines) if fragment in l]
    functions={n.name:[n.lineno,n.end_lineno] for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_s6a2s_')}
    return dict(selectors={k:locations(k+'=False') for k in ('S6A2_T8','S6A2_T16','S6A2_BUILD')},
                functions=functions,
                T8_launches=locations('cudaLaunchKernelEx(&lc,s6a2s::mat<8,'),
                T16_launches=locations('cudaLaunchKernelEx(&lc,s6a2s::mat<16,'),
                common_body=[locations('#line 6100')[0],locations('#line 6200')[0]-1],
                reduction_epilogue=[locations('#line 6200')[0],locations('#line 6300')[0]-1],
                prepare_job=locations('jobs.append(("s6a2s",_s6a2s_build))'),
                prepare_hook=locations(' _s6a2s_prepare(state,weights,t0)'),
                tail_hook=locations(' if _s6a2s_bank(state,y.shape[0]) is not None:'),
                dispatch_hook=locations(' if _s6a2s_selected(tokens):'))


def accounting():
    return dict(label='EST source-operation accounting, NOT measured SASS or timing',
        weight_stream_grows=False,weight_tile_bytes_before=36864,weight_tile_bytes_after=36864,
        packed_word_bytes_with_scale_before=4.5,packed_word_bytes_with_scale_after=4.5,
        warp_K64_bytes_before=576,warp_K64_bytes_after=576,
        T8=dict(CTA_threads_before=288,compute_warps_before=8,CTA_threads_after=256,
                resident_compute_warps_target=16,K_per_warp_before=512,K_per_warp_after=512,
                words_per_thread_tile_before=32,words_per_thread_tile_after=32,
                weight_scale_bytes_per_compute_thread_before=144,weight_scale_bytes_per_compute_thread_after=144,
                unpack_scale_ops_per_word_before='DQ7: 10 live source primitives + 4 scale-mul + 4/4 scale decode = 15; SASS can fuse primitives (unmeasured)',
                unpack_scale_ops_per_word_after='8 + 4 + 2/4 = 12.5',
                MMA_ops_per_word_before=1,MMA_ops_per_word_after=1,
                body_ops_per_thread_K64_before='4*10 DQ7 primitives + 4*4 scale-mul + 4 scale-broadcast + 4 MMA + 1 LDS.128 + 0.125 scale LDS.128 = 65.125 source operations, before compiler fusion',
                body_ops_per_thread_K64_after='4*8 unpack + 4*4 scale-mul + 2 scale-pair + 4 MMA + 1 LDG.128 + 0.5 LDG.32 = 55.5 (excludes address/control/epilogue)',
                steady_activation_load_bytes_before=0,steady_activation_load_bytes_after=0,
                reduction_smem_bytes_after=4096,total_instruction_target_per_word=17.),
        T16=dict(CTA_threads_before=288,compute_warps_before=8,CTA_threads_after=512,
                 resident_compute_warps_target=16,K_per_warp_before=512,K_per_warp_after=256,
                 words_per_thread_tile_before=32,words_per_thread_tile_after=16,
                 weight_scale_bytes_per_compute_thread_before=144,weight_scale_bytes_per_compute_thread_after=72,
                 unpack_scale_ops_per_word_before='8 + 4 + 2/4 = 12.5',unpack_scale_ops_per_word_after='8 + 4 + 2/4 = 12.5',
                 MMA_ops_per_word_before='1 for count<=8; 2 for count>8',MMA_ops_per_word_after=2,
                 body_ops_per_thread_K64_before='4*8 unpack + 4*4 scale-mul + 2 scale-pair + (4 or 8) MMA + 1 LDS.128 + 0.125 scale LDS.128 + (0 or 2) second-half LDS.128 = 55.125 cold / 61.125 hot source operations',
                 body_ops_per_thread_K64_after='4*8 unpack + 4*4 scale-mul + 2 scale-pair + 8 MMA + 1 LDG.128 + 0.5 LDG.32 = 59.5 (excludes address/control/epilogue)',
                 steady_activation_load_bytes_per_hot_tile_before=65536,steady_activation_load_bytes_after=0,
                 reduction_smem_bytes_after=16384,total_instruction_target_per_word=18.5),
        boundary_scratch=dict(membership_bytes_per_token=4,intermediate_BF16_bytes_per_live_route=8192,
                              W2_BF16_bytes_per_live_route=8192,FP32_global_K_partials=0,
                              reduction_stores_per_tile_T8=4096,reduction_stores_per_tile_T16=16384,
                              finalize_threads=256,finalize_elements_per_thread=16),
        limitations=['before/after complete SASS instruction counts require offline compilation',
                     'source vector loads and bitwise primitives may split/fuse in SASS; do not price source-operation deltas as time',
                     'the new N16 template always computes both padded token halves',
                     'extra launch edges, input fills, epilogue and normalization remain on the dependency chain'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--out',type=Path)
    p.add_argument('--ship',action='store_true')
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    out=args.out or ROOT/('kernels/lab/k3_s6a2s_ship.py' if args.ship else 'kernels/lab/k3_s6a2s_ab.py')
    if out.resolve()==args.base.resolve():
        p.error('base is read-only')
    raw=args.base.read_bytes()
    if b'\r' in raw:
        p.error('base must be LF; do not silently change chained source')
    s=raw.decode('utf-8')
    lab=apply(s)
    digest=check_off(s,lab)
    new=apply(s,True) if args.ship else lab
    if args.check:
        compile(new,str(out),'exec')
    out.write_bytes(new.encode('utf-8'))
    print(json.dumps(dict(output=str(out),base_bytes=len(raw),bytes=len(new.encode()),net_bytes=len(new.encode())-len(raw),
                         default_off_sha256=digest,changes=line_changes(s,new),per_bit=manifest(new)),indent=2))


if __name__=='__main__':
    main()

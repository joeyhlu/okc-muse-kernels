"""S6A5a anchored, default-off dense producer / SS-WGMMA experiment.

Only this generator owns the lab vehicle. --check is read-only, including with
--ship; --base/--out allow chaining. No imports of the GPU base during generation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "kernels/lab/k3_s6a5a_ab.py"

CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace s6a5a {
struct Params {
 const uint4* w;
 const uint32_t *esc, *eo;
 const int *offsets, *plan;
 const __nv_bfloat16* x;
 void* y;
 float* partial;
 unsigned* count;
 const unsigned* tb;
 int K,N,f32,pdl;
};
__device__ __forceinline__ unsigned sa(const void* p) {return (unsigned)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void csync() {
 asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ unsigned remote(unsigned a,unsigned r) {
 unsigned b; asm volatile("mapa.shared::cluster.u32 %0,%1,%2;" : "=r"(b):"r"(a),"r"(r)); return b;
}
__device__ __forceinline__ float rload(unsigned a) {
 float x; asm volatile("ld.shared::cluster.f32 %0,[%1];":"=f"(x):"r"(a):"memory"); return x;
}
__device__ __forceinline__ unsigned rloadu(unsigned a) {
 unsigned x; asm volatile("ld.shared::cluster.u32 %0,[%1];":"=r"(x):"r"(a):"memory"); return x;
}
__device__ __forceinline__ void init(unsigned a,int n) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(a),"r"(n):"memory");
}
__device__ __forceinline__ void arrive(unsigned a) {
 #line 10301 "s6a5a_producer_protocol"
 asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(a):"memory");
}
__device__ __forceinline__ void wait(unsigned a,unsigned phase) {
 // Only CTA-local producer/consumer waits. There is no grid / other-cluster poll.
 asm volatile("{.reg .pred p; AGAIN: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra AGAIN;}"
              ::"r"(a),"r"(phase):"memory");
}
 #line 20001 "s6a5a_pipeline"
__device__ __forceinline__ uint64_t desc(unsigned a) {
 return uint64_t((a&0x3ffff)>>4)|(uint64_t(1)<<16)|(uint64_t(64)<<32)|(uint64_t(1)<<62);
}
template<int T> __device__ __forceinline__ void mma(float* d,uint64_t a,uint64_t b);
@MMA@
template<int T> struct Cfg {
 static constexpr int STAGE=16384+T*128, RING=3*STAGE, SMEM=RING+1024+64;
 static_assert(T*128*4<=RING,"drained ring holds the complete FP32 CTA partial");
};
__device__ __forceinline__ void fetch(const Params& p,int stage,int lane,uint4 (&v)[6]) {
 #pragma unroll
 for(int q=0;q<6;++q) v[q]=__ldg(p.w+size_t(stage)*768+q*128+lane);
}
__device__ __forceinline__ void decode_store(const uint4 (&v)[6],unsigned tb,unsigned ws,int lane) {
 #line 10001 "s6a5a_producer_decode"
 // GD's exact 9-instruction / eight-value decode, with independent producer registers.
 unsigned swbase=ws+lane*128+(lane&7)*16;
 #pragma unroll
 for(int half=0;half<2;++half) {
  const uint32_t L[8]={v[half*3].x,v[half*3].y,v[half*3].z,v[half*3].w,
                      v[half*3+1].x,v[half*3+1].y,v[half*3+1].z,v[half*3+1].w};
  const uint32_t Q[4]={v[half*3+2].x,v[half*3+2].y,v[half*3+2].z,v[half*3+2].w};
  #pragma unroll
  for(int q=0;q<4;++q) {
   unsigned h0=(Q[q]&0x87878787u)+tb;
   unsigned h1=(__funnelshift_r(Q[q],Q[q],4)&0x87878787u)+tb;
   uint4 d=make_uint4(__byte_perm(L[q*2],h0,0x5140),__byte_perm(L[q*2],h0,0x7362),
                     __byte_perm(L[q*2+1],h1,0x5140),__byte_perm(L[q*2+1],h1,0x7362));
   unsigned dst=swbase^((half*4+q)*16);
   asm volatile("st.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(dst),"r"(d.x),"r"(d.y),"r"(d.z),"r"(d.w):"memory");
  }
 }
}
 #line 20001 "s6a5a_pipeline"
template<int T> __device__ __forceinline__ void produce(const Params& p,uint4 (&v)[6],
 int i,int len,int stage0,int k0,unsigned base,unsigned bars,int lane,unsigned tb) {
 #line 10101 "s6a5a_producer_stage"
 if(i>=len) return; // Uniform over the entire producer WG; no CTA exits.
 int s=i%3; unsigned ws=base+s*Cfg<T>::STAGE, xs=ws+16384;
 if(i>=3) wait(bars+24+s*8,((i/3)-1)&1);
 // SASS gate attributes the complete producer stage, including escapes and X.
 decode_store(v,tb,ws,lane);
 asm volatile("bar.sync 1,128;":::"memory");
 int stage=stage0+i;
 unsigned eb=p.eo[stage],ee=p.eo[stage+1];
 for(unsigned e=eb+lane;e<ee;e+=128) {
  unsigned z=p.esc[e], pos=z&8191, r=pos>>6, k=pos&63;
  unsigned a=ws+r*128+(((k>>3)^(r&7))*16)+(k&7)*2;
  unsigned h; asm volatile("ld.shared.u16 %0,[%1];":"=r"(h):"r"(a):"memory");
  h^=(z>>13)<<8;
  asm volatile("st.shared.u16 [%0],%1;"::"r"(a),"r"(h):"memory");
 }
 #pragma unroll
 for(int j=0;j<T/16;++j) {
  int z=lane+j*128, t=z/8, k=z%8;
  uint4 x=__ldg(reinterpret_cast<const uint4*>(p.x+size_t(t)*p.K+(k0+i)*64)+k);
  unsigned a=xs+t*128+((k^(t&7))*16);
  asm volatile("st.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(a),"r"(x.x),"r"(x.y),"r"(x.z),"r"(x.w):"memory");
 }
 asm volatile("bar.sync 1,128;\nfence.proxy.async.shared::cta;":::"memory");
 arrive(bars+s*8); // All 128 producers publish after patches and X stores.
 if(i+4<len) {
  #pragma unroll
  for(int q=0;q<6;++q) v[q]=__ldg(p.w+size_t(stage0+i+4)*768+q*128+lane);
 }
}
 #line 20001 "s6a5a_pipeline"
template<int T> __global__ __launch_bounds__(384,1)
void kernel(const __grid_constant__ Params p) {
 extern __shared__ __align__(1024) unsigned char raw[];
 unsigned base=(sa(raw)+1023)&~1023u, bars=base+Cfg<T>::RING;
 float* red=reinterpret_cast<float*>(raw+(base-sa(raw)));
 unsigned* last=reinterpret_cast<unsigned*>(raw+(bars+48-sa(raw)));
 int tid=threadIdx.x,rank=blockIdx.x&1,cid=blockIdx.x/2;
 if(p.pdl) asm volatile("griddepcontrol.wait;":::"memory");
 csync(); // Both CTAs exist before any remote-shared access, even empty ranges.
 for(int job=p.offsets[cid];job<p.offsets[cid+1];++job) {
  const int* a=p.plan+job*7;
  int tile=a[0],lo=a[1],hi=a[2],slot=a[3],first=a[4],parts=a[5],counter=a[6];
  int mid=(lo+hi)/2,k0=rank?mid:lo,k1=rank?hi:mid,len=k1-k0;
  if(tid==0) {
   for(int s=0;s<3;++s) {init(bars+s*8,128);init(bars+24+s*8,256);}
   *last=0;
   asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  float d[T/2]; // Live only on consumers; producer slot lifetimes are disjoint.
  if(tid<128) {
   // Four *separately named* 24-register slots; no dynamically indexed local array.
   uint4 v0[6],v1[6],v2[6],v3[6]; int stage0=tile*(p.K/64)+k0;
   unsigned tb=p.tb[tile];
   if(len>0) fetch(p,stage0,tid,v0);
   if(len>1) fetch(p,stage0+1,tid,v1);
   if(len>2) fetch(p,stage0+2,tid,v2);
   if(len>3) fetch(p,stage0+3,tid,v3);
   for(int i=0;i<len;i+=4) {
    produce<T>(p,v0,i,len,stage0,k0,base,bars,tid,tb);
    produce<T>(p,v1,i+1,len,stage0,k0,base,bars,tid,tb);
    produce<T>(p,v2,i+2,len,stage0,k0,base,bars,tid,tb);
    produce<T>(p,v3,i+3,len,stage0,k0,base,bars,tid,tb);
   }
  } else {
   #pragma unroll
   for(int j=0;j<T/2;++j) d[j]=0.f;
   int g=(tid-128)/128;
   for(int i=0;i<len;++i) {
    unsigned ws=base+(i%3)*Cfg<T>::STAGE,xs=ws+16384;
    wait(bars+(i%3)*8,(i/3)&1);
    #pragma unroll
    for(int j=0;j<T/2;++j) asm volatile("":"+f"(d[j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
    #pragma unroll
    for(int q=0;q<4;++q) mma<T>(d,desc(ws+g*8192+q*32),desc(xs+q*32));
    asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;":::"memory");
    #pragma unroll
    for(int j=0;j<T/2;++j) asm volatile("":"+f"(d[j])::"memory");
    arrive(bars+24+(i%3)*8); // All 256 consumers release, only after operand retirement.
   }
  }
  __syncthreads(); // Uniform CTA rendezvous before aliasing the drained stage ring.
  if(tid>=128) {
   int g=(tid-128)/128;
   int lane=tid&31,warp=((tid-128)&127)/32;
   #pragma unroll
   for(int j=0;j<T/2;++j) {
    int r=g*64+warp*16+lane/4+((j&2)?8:0), t=(j/4)*8+(lane%4)*2+(j&1);
    red[t*128+r]=d[j];
   }
  }
  __syncthreads();
  csync(); // Both red tiles ready; no producer or MMA can still reference the ring.
  unsigned peer=remote(base,rank^1);
  for(int j=rank*(T*64)+tid;j<(rank+1)*(T*64);j+=384) {
   // Rank-independent add order, including the zero-K half of one-stage fragments.
   float local=red[j],other=rload(peer+j*4),v=rank?other+local:local+other;
   if(parts==1) {
    size_t dst=size_t(j/128)*p.N+tile*128+j%128;
    if(p.f32) reinterpret_cast<float*>(p.y)[dst]=v;
    else reinterpret_cast<__nv_bfloat16*>(p.y)[dst]=__float2bfloat16_rn(v);
   } else p.partial[size_t(slot)*T*128+j]=v;
  }
  if(parts>1) __threadfence(); // Every writer fences its own global partial stores.
  __syncthreads(); csync();
  if(parts>1 && rank==0 && tid==0) {
   unsigned old;
   asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p.count+counter):"memory");
   *last=(old==unsigned(parts-1));
  }
  __syncthreads(); csync();
  bool final=parts>1 && rloadu(remote(bars+48,0));
  if(final) {
   // Acquire on BOTH CTAs; prior contributors never wait for this resolver.
   asm volatile("fence.acquire.gpu;":::"memory");
   for(int j=rank*(T*64)+tid;j<(rank+1)*(T*64);j+=384) {
    float v=0.f;
    for(int z=0;z<parts;++z) v+=p.partial[size_t(first+z)*T*128+j];
    size_t dst=size_t(j/128)*p.N+tile*128+j%128;
    if(p.f32) reinterpret_cast<float*>(p.y)[dst]=v;
    else reinterpret_cast<__nv_bfloat16*>(p.y)[dst]=__float2bfloat16_rn(v);
   }
  }
  __syncthreads(); csync();
  if(final && rank==0 && tid==0) atomicExch(p.count+counter,0u);
  if(tid==0) for(int s=0;s<6;++s)
   asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(bars+s*8):"memory");
  __syncthreads(); csync(); // Lifetime/reset barrier, also on all non-finalizing CTAs.
 }
 csync();
 if(p.pdl) asm volatile("griddepcontrol.launch_dependents;":::"memory");
}
template<int T> int launch(Params p,cudaStream_t stream) {
 auto fn=kernel<T>;
 cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg<T>::SMEM);
 if(e!=cudaSuccess) return int(e);
 cudaLaunchConfig_t cfg={}; cfg.gridDim=dim3(132);cfg.blockDim=dim3(384);
 cfg.dynamicSmemBytes=Cfg<T>::SMEM;cfg.stream=stream;
 cudaLaunchAttribute attrs[2]={}; attrs[0].id=cudaLaunchAttributeClusterDimension;
 attrs[0].val.clusterDim.x=2;attrs[0].val.clusterDim.y=1;attrs[0].val.clusterDim.z=1;
 attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
 attrs[1].val.programmaticStreamSerializationAllowed=p.pdl;
 cfg.attrs=attrs;cfg.numAttrs=2;
 return int(cudaLaunchKernelEx(&cfg,fn,p));
}
} // namespace s6a5a
extern "C" int s6a5a_launch(int T,s6a5a::Params p,cudaStream_t st) {
 switch(T) {case 16:return s6a5a::launch<16>(p,st);case 32:return s6a5a::launch<32>(p,st);
 case 128:return s6a5a::launch<128>(p,st);default:return int(cudaErrorInvalidValue);}
}
#ifndef S6A5A_STANDALONE
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int64_t s6a5a_run(torch::Tensor w,torch::Tensor esc,torch::Tensor eo,torch::Tensor offsets,
 torch::Tensor plan,torch::Tensor partial,torch::Tensor count,torch::Tensor x,torch::Tensor y,torch::Tensor tb,int64_t pdl) {
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.is_contiguous(),"s6a5a X");
 TORCH_CHECK(y.is_cuda() && y.is_contiguous() && (y.scalar_type()==at::kFloat || y.scalar_type()==at::kBFloat16),"s6a5a Y");
 TORCH_CHECK(x.dim()==2 && y.dim()==2 && y.size(0)==x.size(0),"s6a5a dimensions");
 TORCH_CHECK(x.size(1)%64==0 && y.size(1)%128==0 && offsets.numel()==67,"s6a5a plan dimensions");
 TORCH_CHECK(tb.numel()==y.size(1)/128 && tb.scalar_type()==at::kInt && tb.is_cuda(),"s6a5a exponent windows");
 s6a5a::Params p={reinterpret_cast<const uint4*>(w.data_ptr()),reinterpret_cast<const unsigned*>(esc.data_ptr()),
 reinterpret_cast<const unsigned*>(eo.data_ptr()),offsets.data_ptr<int>(),plan.data_ptr<int>(),
 reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),y.data_ptr(),partial.data_ptr<float>(),
 reinterpret_cast<unsigned*>(count.data_ptr()),reinterpret_cast<const unsigned*>(tb.data_ptr()),
 int(x.size(1)),int(y.size(1)),y.scalar_type()==at::kFloat,int(pdl)};
 int e=s6a5a_launch(int(x.size(0)),p,at::cuda::getCurrentCUDAStream().stream());
 TORCH_CHECK(e==0,"s6a5a launch ",e); return e;
}
#endif
'''

PYTHON = r'''
# S6A5A BEGIN -- selectors change call-time routing only; prepare request is separate.
S6A5A = 0 # bits: qkv=1, o=2, cat(pooler + router hi/lo)=4, unpooler=8
S6A5A_T = (16,32,128)
S6A5A_GEMM = ("qkv","o","cat","unpooler")
S6A5A_PREPARE = False
S6A5A_TWIN = 0
S6A5A_PDL = True
_S6A5A_BITS = {"qkv":1,"o":2,"cat":4,"unpooler":8}
_S6A5A_CACHE = {}
def _s6a5a_plan(n,k):
 assert n%128==0 and k%64==0
 nt,nk=n//128,k//64
 ranges=[(i*nt*nk//66,(i+1)*nt*nk//66) for i in range(66)]
 owners=[[] for _ in range(nt)]
 for c,(lo,hi) in enumerate(ranges):
  if lo==hi: continue
  for t in range(lo//nk,(hi+nk-1)//nk):
   owners[t].append(c)
 first=[]; slots=0; counters=0; ci=[]
 for cs in owners:
  first.append(slots if len(cs)>1 else -1)
  ci.append(counters if len(cs)>1 else -1)
  if len(cs)>1: slots+=len(cs);counters+=1
 rows=[];offsets=[0]
 for c,(lo,hi) in enumerate(ranges):
  if lo==hi: offsets.append(len(rows));continue
  for t in range(lo//nk,(hi+nk-1)//nk):
   cs=owners[t]; slot=first[t]+cs.index(c) if len(cs)>1 else -1
   rows.append((t,max(lo,t*nk)-t*nk,min(hi,(t+1)*nk)-t*nk,slot,first[t],len(cs),ci[t]))
  offsets.append(len(rows))
 return offsets,rows,slots,counters
def _s6a5a_encode(bits):
 # Lossless row-major P12; GD's sign placement / byte-permute arithmetic, new wire order.
 import numpy as np
 n,k=bits.shape
 assert n%128==0 and k%64==0
 b=np.asarray(bits,dtype=np.uint16).reshape(n//128,128,k//64,64).transpose(0,2,1,3).copy()
 b=b.reshape(-1,128,8,8).astype(np.uint32)
 h=(b>>8)&127
 # Distinct windows keep pooler / router-hi / tiny router-lo exponent populations separate.
 hist=[np.bincount(tile.reshape(-1),minlength=128) for tile in h.reshape(n//128,-1)]
 base=np.array([np.convolve(v,np.ones(8,dtype=np.int64),mode="valid").argmax() for v in hist],dtype=np.uint32)
 sb=np.repeat(base,k//64)
 off=h.astype(np.int32)-sb[:,None,None,None].astype(np.int32)
 bad=(off<0)|(off>7); off=np.where(bad,0,off).astype(np.uint32)
 sg=b>>15; low=b&255
 a=off[...,:4]|(sg[...,4:][...,[3,0,1,2]]<<3)|(off[...,4:]<<4)|(sg[...,:4]<<7)
 shifts=np.arange(4,dtype=np.uint32)*8
 codes=(a<<shifts).sum(-1,dtype=np.uint32)
 lows=(low.reshape(*low.shape[:-1],2,4)<<shifts).sum(-1,dtype=np.uint32)
 planes=[]
 for half in range(2):
  ll=lows[:,:,half*4:(half+1)*4].reshape(-1,128,8)
  planes.extend((ll[:,:,:4],ll[:,:,4:],codes[:,:,half*4:(half+1)*4]))
 packed=np.stack(planes,axis=1).copy()
 pos=np.nonzero(bad.reshape(-1,8192))
 xor=(h.reshape(-1,8192)[pos]^sb[pos[0]]).astype(np.uint32)
 esc=(pos[1].astype(np.uint32)|(xor<<13)).copy()
 counts=np.bincount(pos[0],minlength=len(b))
 eo=np.concatenate((np.zeros(1,dtype=np.int64),counts.cumsum())).astype(np.uint32)
 return packed,esc,eo,base*np.uint32(0x01010101)
def _s6a5a_pack(w):
 import numpy as np
 bits=w.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
 packed,esc,eo,tb=_s6a5a_encode(bits)
 return (torch.from_numpy(packed.view(np.int32)).to(w.device),
         torch.from_numpy(esc.view(np.int32)).to(w.device),torch.from_numpy(eo.view(np.int32)).to(w.device),
         torch.from_numpy(tb.view(np.int32)).to(w.device))
def _s6a5a_build(twin):
 if twin not in _S6A5A_CACHE:
  from torch.utils.cpp_extension import load_inline
  src=_S6A5A_CUDA.replace("namespace s6a5a {",f"namespace s6a5a_t{twin} {{").replace("s6a5a::",f"s6a5a_t{twin}::")
  tag=hashlib.sha256(src.encode()).hexdigest()[:12]
  _S6A5A_CACHE[twin]=load_inline(name="s6a5a_"+tag,cpp_sources=_S6A5A_CPP,cuda_sources=src,
   functions=["s6a5a_run"],extra_cuda_cflags=["-O3","-lineinfo","-gencode=arch=compute_90a,code=sm_90a",
   "--expt-relaxed-constexpr","--maxrregcount=160"],verbose=False)
 return _S6A5A_CACHE[twin]
def _s6a5a_prepare(state):
 state["s6a5a"]={}
 if not (S6A5A_PREPARE or S6A5A): return
 # The exact accessor is the base's _dg1_weight, including v8/wcat and qkv_gate.
 for twin in range(_S6A5A_NBANKS):
  H={"mod":_s6a5a_build(twin),"route":{},"w":{}}
  for proj in S6A5A_GEMM:
   w=_dg1_weight(state,proj)
   if w is None: raise RuntimeError("S6A5a missing weight: "+proj)
   n,k=w.shape; packed=_s6a5a_pack(w);H["w"][proj]=packed
   offsets,rows,slots,counters=_s6a5a_plan(n,k)
   dev=w.device
   off=torch.tensor(offsets,dtype=torch.int32,device=dev)
   plan=torch.tensor(rows,dtype=torch.int32,device=dev)
   for T in S6A5A_T:
    if T not in (16,32,128): raise ValueError("S6A5a unsupported T")
    H["route"][(proj,T)]={"n":n,"k":k,"offsets":off,"plan":plan,
     "partial":torch.empty((slots,T,128),dtype=torch.float32,device=dev),
     "count":torch.zeros((counters,),dtype=torch.int32,device=dev),"bytes":slots*T*128*4}
  state["s6a5a"][twin]=H
def _s6a5a_live(proj,T):
 return bool(S6A5A&_S6A5A_BITS[proj]) and T in S6A5A_T and proj in S6A5A_GEMM
def _s6a5a_gemm(state,proj,x):
 T=x.shape[0]
 if not _s6a5a_live(proj,T): return None
 H=state["s6a5a"][S6A5A_TWIN];r=H["route"][(proj,T)]
 assert x.dtype==torch.bfloat16 and x.is_contiguous() and x.shape[1]==r["k"]
 out=torch.empty((T,r["n"]),dtype=torch.float32 if proj=="cat" else torch.bfloat16,device=x.device)
 w,esc,eo,tb=H["w"][proj]
 H["mod"].s6a5a_run(w,esc,eo,r["offsets"],r["plan"],r["partial"],r["count"],x,out,tb,int(S6A5A_PDL))
 return out
def _s6a5a_tail(state,x,data,tail):
 if tail!="cat": return None
 y=_s6a5a_gemm(state,"cat",x)
 return None if y is None else _v8_tail(y,None,state,data)
def s6a5a_arm(name):
 # Same-state call-time bank switch, as in padz_run.py. Set PREPARE before prepare().
 global S6A5A,S6A5A_T,S6A5A_GEMM,S6A5A_TWIN
 S6A5A=0;S6A5A_T=(16,32,128);S6A5A_GEMM=tuple(_S6A5A_BITS);S6A5A_TWIN=0
 if name in ("ship","mid1","mid2"): return
 if name.endswith("_twin"): S6A5A_TWIN=1;name=name[:-5]
 if name=="all": S6A5A=15;return
 for proj,bit in _S6A5A_BITS.items():
  for T in (16,32,128):
   if name==proj+str(T): S6A5A=bit;S6A5A_T=(T,);S6A5A_GEMM=(proj,);return
 raise ValueError("unknown S6A5a arm: "+name)
# S6A5A END
'''

CPP = "int64_t s6a5a_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);"

def mma_spec(t):
    nr = t // 2
    regs = ",".join(f"%{i}" for i in range(nr))
    outputs = ",".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f'template<> __device__ __forceinline__ void mma<{t}>(float* d,uint64_t a,uint64_t b) {{\n'
            f' asm volatile("{{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n{t}k16.f32.bf16.bf16 '
            f'{{{regs}}},%{nr},%{nr+1},p,1,1,0,0;}}" : {outputs} : "l"(a),"l"(b));\n}}\n')

def cuda_source():
    return CUDA.replace("@MMA@", "".join(mma_spec(t) for t in (16, 32, 128)))

def sub(src, old, new):
    assert src.count(old) == 1, f"anchor count {src.count(old)}: {old[:100]!r}"
    return src.replace(old, new)

def generate(src, ship=False):
    assert "# S6A5A BEGIN" not in src, "already patched"
    assert src.startswith("# okc: problem=muse_decode_layer\n# nonce ")
    changes = []
    # Replace statements at their existing route priority; no wrapper module mutation.
    for var, proj, x in (("qkv", "qkv", 'data["attn_input"]'), ("attn_output", "o", "o"),
                         ("ffn_output", "unpooler", "expert_output")):
        old = f' {var}=_rq_gemm(state,"{proj}",{x})\n'
        new = f' {var}=_s6a5a_gemm(state,"{proj}",{x})\n if {var} is None:\n '+old
        src = sub(src, old, new)
        changes.append({"bit": {"qkv":1,"o":2,"unpooler":8}[proj], "anchor": old.strip()})
    old = ' g2out=_rq_pooler_tail(state,ffn_input,data,tail)\n'
    src = sub(src, old, ' g2out=_s6a5a_tail(state,ffn_input,data,tail)\n if g2out is None:\n '+old)
    changes.append({"bit":4,"anchor":old.strip()})
    src = sub(src, ' _tinys_prepare()\n', ' _tinys_prepare()\n _s6a5a_prepare(state)\n')
    payload = (f"_S6A5A_NBANKS={1 if ship else 2}\n_S6A5A_CPP={CPP!r}\n_S6A5A_CUDA=r'''\n"+
               cuda_source()+"\n'''\n"+PYTHON)
    if ship:
        # Lab arm enumeration is superseded by explicit production selectors.
        payload = payload[:payload.index("def s6a5a_arm(")]+"# S6A5A END\n"
    src = sub(src, "def kernel_fn(data,state):\n", payload+"\ndef kernel_fn(data,state):\n")
    lines=src.splitlines()
    for change in changes:
        token={1:' qkv=_s6a5a_gemm(',2:' attn_output=_s6a5a_gemm(',4:' g2out=_s6a5a_tail(',8:' ffn_output=_s6a5a_gemm('}[change["bit"]]
        change["generated_lines"]=[i+1 for i,line in enumerate(lines) if line.startswith(token)]
    changes.append({"bit":"common","prepare_line":next(i+1 for i,line in enumerate(lines) if line==' _s6a5a_prepare(state)'),
                    "payload_first_line":next(i+1 for i,line in enumerate(lines) if line.startswith('_S6A5A_NBANKS=')),
                    "payload_last_line":next(i+1 for i,line in enumerate(lines) if line=='# S6A5A END')})
    ast.parse(src)
    assert src.isascii() and "\r" not in src
    if ship: assert len(src.encode()) < 1_000_000, "ship exceeds byte cap; do not accumulate another layer"
    return src, changes

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,default=BASE)
    ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--ship",action="store_true")
    ap.add_argument("--check",action="store_true",help="generate/AST/size gates in memory; never write")
    args=ap.parse_args()
    assert args.base.resolve()!=args.out.resolve(), "never overwrite the input base"
    old=args.base.read_bytes().decode("ascii").replace("\r\n","\n")
    new,changes=generate(old,args.ship)
    if not args.check: args.out.write_bytes(new.encode("ascii"))
    print(json.dumps({"ok":True,"ship":args.ship,"check":args.check,"bytes":len(new),"net_bytes":len(new)-len(old),
      "base_sha256":hashlib.sha256(args.base.read_bytes()).hexdigest(),"changes":changes},indent=2))

if __name__=="__main__": main()

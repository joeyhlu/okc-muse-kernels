"""S6A5b: raw BF16 T256, 192 rows, 3 consumer WGs + one TMA warp.

Only generates kernels/lab/k3_s6a5b_ab.py. S6A5B defaults to zero:
1 cat (pooler + FP32 router hi/lo), 2 unpooler, 4 qkv, 8 o.
S6A5B_ARM selects compiled twins 1/2 in one prepared state (padz convention).
--ship removes the second build; --bits selects any subset (default zero).
No CUDA driver calls, raw pointers or tensor maps are stored by prepare.
"""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "kernels/lab/k3_s6a5b_ab.py"

CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef S6A5B_DEVICE_ONLY
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#endif
namespace s6a5b {
// Dynamic allocation includes 1024 B alignment slack and 64 B barrier/control.
constexpr int WB=192*64*2, XB=256*64*2, SB=WB+XB;
constexpr int RING=3*SB, SMEM=RING+1024+64, THREADS=416;
static_assert(SMEM<=227*1024 && 16*384*16<=RING, "resource contract");
struct P { void* y; float* partial; int* count; const int* plan; int n, k; };
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void init(uint32_t b,int n) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b),"r"(n):"memory");
}
__device__ __forceinline__ void arrive(uint32_t b) {
 asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b):"memory");
}
__device__ __forceinline__ void expect(uint32_t b) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b),"r"(SB):"memory");
}
__device__ __forceinline__ void wait(uint32_t b,int phase) {
 asm volatile("{ .reg .pred p; Lwait: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra Lwait; }"
              :: "r"(b),"r"(phase):"memory");
}
__device__ __forceinline__ void tma(uint32_t dst,const CUtensorMap* tm,int k,int r,uint32_t b) {
 asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
              :: "r"(dst),"l"(tm),"r"(b),"r"(k),"r"(r):"memory");
}
__device__ __forceinline__ uint64_t desc(uint32_t a) {
 return ((uint64_t)(a&0x3ffff)>>4) | (1ull<<16) | (64ull<<32) | (1ull<<62);
}
__device__ __forceinline__ void cluster_sync() {
 asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t remote(uint32_t a,int r) {
 uint32_t b; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r)); return b;
}
__device__ __forceinline__ void remote4(uint32_t a,float x,float y,float z,float w) {
 asm volatile("st.shared::cluster.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(a),"f"(x),"f"(y),"f"(z),"f"(w):"memory");
}
__device__ __forceinline__ int read_remote(uint32_t a) {
 int x; asm volatile("ld.shared::cluster.u32 %0, [%1];":"=r"(x):"r"(a):"memory"); return x;
}
__device__ __forceinline__ int publish(int* p) {
 int x; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;":"=r"(x):"l"(p):"memory"); return x;
}
__device__ __forceinline__ void acquire() { asm volatile("fence.acquire.gpu;" ::: "memory"); }
__device__ __forceinline__ void release() { asm volatile("fence.release.gpu;" ::: "memory"); }
__device__ __forceinline__ void mma(float* d,uint64_t a,uint64_t b) {
@MMA@
}
template<bool F32> __device__ __forceinline__ void output(P p,int tile,int t,int r,float v) {
 const int row=tile*192+r;
 if(row<p.n) {
  if constexpr(F32) ((float*)p.y)[t*p.n+row]=v;
  else ((__nv_bfloat16*)p.y)[t*p.n+row]=__float2bfloat16_rn(v);
 }
}
template<int Rows,int Tokens,int Kstep,int ConsumerWG,int ProducerWarp,bool F32>
__global__ __launch_bounds__(416,1)
void Dense(const __grid_constant__ CUtensorMap wm,const __grid_constant__ CUtensorMap xm,P p) {
 static_assert(Rows==192 && Tokens==256 && Kstep==64 && ConsumerWG==3 && ProducerWarp==1,"specialization");
 extern __shared__ __align__(1024) uint8_t raw[];
 const uint32_t base=(sa(raw)+1023u)&~1023u;
 uint8_t* mem=raw+base-sa(raw);
 const int tid=threadIdx.x, rank=blockIdx.x&1, cid=blockIdx.x/2;
 const uint32_t full=base+RING, empty=full+24;
 // At most two fragments per range for these four shapes. Plan slots are
 // [tile,k0,k1,workspace_slot,first_slot,parts]; padded entries have tile=-1.
 #pragma unroll 1
 for(int frag=0;frag<2;++frag) {
  const int* q=p.plan+(cid*2+frag)*6;
  if(q[0]<0) break; // cluster-uniform; previous lifetime barrier completed
  const int mid=(q[1]+q[2])/2;
  const int k0=rank?mid:q[1], kend=rank?q[2]:mid, nk=kend-k0;
  if(tid==0) {
   #pragma unroll
   for(int s=0;s<3;++s) { init(full+8*s,1); init(empty+8*s,3); }
   asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  float acc[128];
  if(tid>=384) {
   if(tid==384) {
    #pragma unroll 1
    for(int i=0;i<nk;++i) {
     const int s=i%3;
     if(i>=3) wait(empty+8*s,((i/3)-1)&1);
     expect(full+8*s);
     tma(base+s*SB,&wm,(k0+i)*64,q[0]*192,full+8*s);
     tma(base+s*SB+WB,&xm,(k0+i)*64,0,full+8*s);
    }
   }
  } else {
   #pragma unroll
   for(int j=0;j<128;++j) acc[j]=0.f;
   #pragma unroll 1
   for(int i=0;i<nk;++i) {
    const int s=i%3;
    wait(full+8*s,(i/3)&1);
    const uint64_t a=desc(base+s*SB+(tid/128)*8192), b=desc(base+s*SB+WB);
    #pragma unroll
    for(int j=0;j<128;++j) asm volatile("":"+f"(acc[j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    // S6A5B_K64_BEGIN: 4 instructions per WG, 12 per CTA per K64.
    #pragma unroll
    for(int kk=0;kk<4;++kk) mma(acc,a+2*kk,b+2*kk);
    // S6A5B_K64_END
    asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 1;" ::: "memory");
    #pragma unroll
    for(int j=0;j<128;++j) asm volatile("":"+f"(acc[j])::"memory");
    if(i>0 && (tid&127)==0) arrive(empty+8*((i-1)%3));
   }
   asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
   #pragma unroll
   for(int j=0;j<128;++j) asm volatile("":"+f"(acc[j])::"memory");
  }
  // Both roles join at the same lexical collective; both rings must be drained.
  __syncthreads();
  cluster_sync();
  if(tid<384) {
   const uint32_t peer=remote(base,1-rank);
   #pragma unroll
   for(int j=0;j<32;++j) if(j/16!=rank)
    remote4(peer+((j%16)*384+tid)*16,acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
  }
  cluster_sync();
  if(tid<384) {
   const float4* rb=(const float4*)mem;
   const int r=(tid/128)*64+((tid/32)&3)*16+(tid&31)/4;
   #pragma unroll
   for(int j=0;j<32;++j) if(j/16==rank) {
    const float4 v=rb[(j%16)*384+tid];
    const int t=j*8+2*(tid&3);
    const float a=__fadd_rn(acc[4*j],v.x), b=__fadd_rn(acc[4*j+1],v.y);
    const float c=__fadd_rn(acc[4*j+2],v.z), d=__fadd_rn(acc[4*j+3],v.w);
    if(q[5]==1) {
     output<F32>(p,q[0],t,r,a); output<F32>(p,q[0],t+1,r,b);
     output<F32>(p,q[0],t,r+8,c); output<F32>(p,q[0],t+1,r+8,d);
    } else {
     float* dst=p.partial+(size_t)q[3]*192*256;
     dst[t*192+r]=a; dst[(t+1)*192+r]=b;
     dst[t*192+r+8]=c; dst[(t+1)*192+r+8]=d;
    }
   }
  }
  // Every writer releases its global stores, then rank 0 publishes one counter.
  release();
  __syncthreads();
  cluster_sync();
  if(rank==0 && tid==0) {
   *(int*)(mem+RING+48)=q[5]>1 && publish(p.count+q[0])==q[5]-1;
  }
  cluster_sync();
  const int last=read_remote(remote(base+RING+48,0));
  if(last) {
   acquire(); // acquire in every reader, including rank 1, after counter acquire
   // 384 consumers cover each owned 128*192 half; producer does no epilogue work.
   if(tid<384) for(int z=tid;z<128*192;z+=384) {
    const int t=rank*128+z/192, r=z%192;
    float v=p.partial[(size_t)q[4]*192*256+t*192+r];
    #pragma unroll 1
    for(int s=1;s<q[5];++s) v=__fadd_rn(v,p.partial[(size_t)(q[4]+s)*192*256+t*192+r]);
    output<F32>(p,q[0],t,r,v);
   }
   __syncthreads();
   cluster_sync();
   if(rank==0 && tid==0) atomicExch(p.count+q[0],0);
  }
  // All 416 threads of both CTAs remain alive through all DSMEM reads and reset.
  cluster_sync();
  // The drained ring held generic DSMEM stores; order them before TMA reuse.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if(tid==0) {
   #pragma unroll
   for(int s=0;s<3;++s) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+8*s):"memory");
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(empty+8*s):"memory");
   }
  }
  __syncthreads();
 }
}
} // namespace s6a5b
#ifdef S6A5B_DEVICE_ONLY
template __global__ void s6a5b::Dense<192,256,64,3,1,false>(const __grid_constant__ CUtensorMap,const __grid_constant__ CUtensorMap,s6a5b::P);
template __global__ void s6a5b::Dense<192,256,64,3,1,true>(const __grid_constant__ CUtensorMap,const __grid_constant__ CUtensorMap,s6a5b::P);
#else
typedef CUresult (*s6_encode)(CUtensorMap*,CUtensorMapDataType,cuuint32_t,void*,const cuuint64_t*,const cuuint64_t*,
 const cuuint32_t*,const cuuint32_t*,CUtensorMapInterleave,CUtensorMapSwizzle,CUtensorMapL2promotion,CUtensorMapFloatOOBfill);
static void s6_map(CUtensorMap* tm,torch::Tensor a,int boxrows) {
 void* entry=nullptr; cudaDriverEntryPointQueryResult q;
 #if CUDART_VERSION >= 12050
 C10_CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",&entry,12000,cudaEnableDefault,&q));
 #else
 C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&entry,cudaEnableDefault,&q));
 #endif
 TORCH_CHECK(q==cudaDriverEntryPointSuccess && entry,"s6a5b: map entry");
 const cuuint64_t dims[2]={(cuuint64_t)a.size(1),(cuuint64_t)a.size(0)}, strides[1]={(cuuint64_t)a.stride(0)*2};
 const cuuint32_t box[2]={64u,(cuuint32_t)boxrows}, es[2]={1,1};
 TORCH_CHECK(((s6_encode)entry)(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,a.data_ptr(),dims,strides,box,es,
  CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,
  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS,"s6a5b: map encode");
}
template<bool F32> static void s6_launch(torch::Tensor w,torch::Tensor x,torch::Tensor y,torch::Tensor plan,
 torch::Tensor partial,torch::Tensor count) {
 auto kern=s6a5b::Dense<192,256,64,3,1,F32>;
 static bool attr=false;
 if(!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,s6a5b::SMEM)); attr=true; }
 CUtensorMap wm,xm; // Encoded only inside the timed invocation; never in prepare.
 s6_map(&wm,w,192); s6_map(&xm,x,256);
 s6a5b::P p={y.data_ptr(),partial.data_ptr<float>(),count.data_ptr<int>(),plan.data_ptr<int>(),(int)y.size(1),(int)x.size(1)};
 cudaLaunchConfig_t cfg={};
 cfg.gridDim=dim3(132); cfg.blockDim=dim3(416); cfg.dynamicSmemBytes=s6a5b::SMEM;
 cfg.stream=at::cuda::getCurrentCUDAStream().stream();
 cudaLaunchAttribute attr={}; attr.id=cudaLaunchAttributeClusterDimension;
 attr.val.clusterDim.x=2; attr.val.clusterDim.y=attr.val.clusterDim.z=1;
 cfg.attrs=&attr; cfg.numAttrs=1;
 C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,kern,wm,xm,p));
}
void s6a5b_run(torch::Tensor w,torch::Tensor x,torch::Tensor y,torch::Tensor plan,torch::Tensor partial,torch::Tensor count) {
 for(const auto& a:{w,x,y,plan,partial,count})
  TORCH_CHECK(a.is_cuda() && a.device()==x.device() && a.is_contiguous(),"s6a5b: device/contiguous");
 TORCH_CHECK(w.scalar_type()==at::kBFloat16 && x.scalar_type()==at::kBFloat16,"s6a5b: BF16");
 TORCH_CHECK(x.dim()==2 && w.dim()==2 && y.dim()==2 && x.size(0)==256 && y.size(0)==256,"s6a5b: T256");
 TORCH_CHECK(x.size(1)==w.size(1) && x.size(1)%64==0 && w.size(0)==((y.size(1)+191)/192)*192,"s6a5b: NK");
 TORCH_CHECK(plan.scalar_type()==at::kInt && plan.numel()==66*2*6 && count.scalar_type()==at::kInt && count.numel()==w.size(0)/192,"s6a5b: plan");
 TORCH_CHECK(partial.scalar_type()==at::kFloat && partial.numel()>=132*192*256,"s6a5b: partial");
 if(y.scalar_type()==at::kFloat) s6_launch<true>(w,x,y,plan,partial,count);
 else { TORCH_CHECK(y.scalar_type()==at::kBFloat16,"s6a5b: output"); s6_launch<false>(w,x,y,plan,partial,count); }
}
#endif
'''

CPP = '#include <torch/extension.h>\nvoid s6a5b_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);\n'

PYTHON = '''
# S6A5b, default-off per-projection selectors. Bank/control structure follows padz.
S6A5B = 0
S6A5B_ARM = 1
S6A5B_BUILD = 15
_S6A5B_KEYS = (1, 2)
_S6A5B_BITS = {"cat":1,"unpooler":2,"qkv":4,"o":8}
_S6A5B = {k:{"mod":None,"err":None,"build_s":0.0} for k in _S6A5B_KEYS}
_S6A5B_FLAGS = ("-O3","-std=c++20","-gencode=arch=compute_90a,code=sm_90a","--maxrregcount=144","-Xptxas=-v","-lineinfo")
def _s6a5b_plan(n,k):
    nt,nk=(n+191)//192,k//64
    assert n>0 and k%64==0 and nt<=66
    frags=[]
    for c in range(66):
        a,b=(nt*nk*c)//66,(nt*nk*(c+1))//66
        start=len(frags)
        while a<b:
            tile=a//nk
            end=min(b,(tile+1)*nk)
            frags.append([c,tile,a%nk,end-tile*nk])
            a=end
        assert len(frags)-start<=2
    bytile={t:[i for i,f in enumerate(frags) if f[1]==t] for t in range(nt)}
    plan=[[[-1,0,0,0,0,0] for _ in range(2)] for _ in range(66)]
    used=[0]*66
    for slot,(c,t,a,b) in enumerate(frags):
        ids=bytile[t]
        assert ids==list(range(ids[0],ids[-1]+1))
        plan[c][used[c]]=[t,a,b,slot,ids[0],len(ids)]
        used[c]+=1
    return plan,len(frags)
def _s6a5b_build():
    if not S6A5B_BUILD:
        return
    from torch.utils.cpp_extension import load_inline
    def one(k):
        H=_S6A5B[k]
        if H["mod"] is not None or H["err"] is not None:
            return
        try:
            src=_S6A5B_CUDA+"\\n// twin "+str(k)+"\\n"
            tag=hashlib.sha1((_S6A5B_CPP+src+str(_S6A5B_FLAGS)).encode()).hexdigest()[:12]
            t0=time.time()
            H["mod"]=load_inline(name=f"s6a5b_{tag}",cpp_sources=_S6A5B_CPP,cuda_sources=src,
                                 functions=["s6a5b_run"],extra_cuda_cflags=list(_S6A5B_FLAGS),verbose=False)
            H["build_s"]=time.time()-t0
        except Exception as ex:
            H["err"]=repr(ex)[-1000:]
    for k in _S6A5B_KEYS:
        one(k)
def _s6a5b_call(H,p,x,y):
    H["mod"].s6a5b_run(p["w"],x,y,p["plan"],p["partial"],p["count"])
def _s6a5b_validate(H,p,w,state,proj):
    dev=w.device
    gen=torch.Generator(device=dev).manual_seed(61425)
    for seed in range(2):
        x=torch.randn((256,w.shape[1]),generator=gen,device=dev).to(torch.bfloat16)
        if seed:
            x=x*0.125
        ref=torch.mm(x,w.t(),out_dtype=torch.float32) if proj=="cat" else torch.mm(x,w.t())
        ref64=x.double()@w.double().t()
        last=None
        for replay in range(2):
            y=torch.full((256,w.shape[0]),float("nan"),device=dev,dtype=ref.dtype)
            _s6a5b_call(H,p,x,y)
            torch.cuda.synchronize()
            assert bool(torch.isfinite(y).all()) and bool((p["count"]==0).all()),"finite/counter reset"
            err=float((y.double()-ref64).norm()/ref64.norm().clamp_min(1e-30))
            cur=float((ref.double()-ref64).norm()/ref64.norm().clamp_min(1e-30))
            assert err<=max(1e-5,cur*1.05+1e-6),("FP64 error",proj,err,cur)
            assert float((y.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-30))<0.005,"cuBLAS error"
            if last is not None:
                assert torch.equal(y,last),"repeat"
            last=y
            if proj=="cat":
                data={"local_start":torch.tensor([seed*32],dtype=torch.int32,device=dev)}
                got,expected=_v8_tail(y,None,state,data),_v8_tail(ref,None,state,data)
                assert torch.equal(got[2],expected[2]) and torch.equal(got[3],expected[3]),"top-8 interface"
                assert torch.allclose(got[1],expected[1],rtol=2e-4,atol=2e-6),"route weights"
                assert torch.allclose(got[0].float(),expected[0].float(),rtol=0.03,atol=0.02),"pooler rounding"
def _s6a5b_prepare(state):
    state["s6a5b"]={k:{} for k in _S6A5B_KEYS}
    if not S6A5B_BUILD:
        return
    _pt1_wait("s6a5b")
    _s6a5b_build()
    for proj,bit in _S6A5B_BITS.items():
        if not S6A5B_BUILD&bit:
            continue
        # Same accessor as _lt_prepare, base:22827, including v8/wcat.
        w=_dg1_weight(state,proj)
        if w is None:
            continue
        n,k=w.shape
        assert (n,k)=={"cat":(4608,8192),"unpooler":(8192,4096),"qkv":(2304,8192),"o":(8192,1024)}[proj]
        assert w.dtype==torch.bfloat16 and w.is_contiguous()
        plan,nfrag=_s6a5b_plan(n,k)
        padded=torch.zeros((((n+191)//192)*192,k),dtype=torch.bfloat16,device=w.device)
        padded[:n].copy_(w)
        pt=torch.tensor(plan,dtype=torch.int32,device=w.device)
        for arm,H in _S6A5B.items():
            if H["mod"] is None:
                print(f"[s6a5b] {proj} twin {arm} unavailable: {H['err']}",flush=True)
                continue
            p={"w":padded,"plan":pt,"n":n,"k":k,"nfrag":nfrag,"device":w.device,
               "partial":torch.empty((132,256,192),dtype=torch.float32,device=w.device),
               "count":torch.zeros(((n+191)//192,),dtype=torch.int32,device=w.device)}
            try:
                _s6a5b_validate(H,p,w,state,proj)
                state["s6a5b"][arm][proj]=p
                print(f"[s6a5b] LIVE {proj}/256 twin {arm}: Dense<192,256,64,3,1>",flush=True)
            except Exception as ex:
                print(f"[s6a5b] OFF {proj}/256 twin {arm}: {ex!r}",flush=True)
def _s6a5b(state,proj,x):
    if x.shape[0]!=256 or not S6A5B&_S6A5B_BITS[proj]:
        return None
    p=state.get("s6a5b",{}).get(S6A5B_ARM,{}).get(proj)
    if p is None or x.dtype!=torch.bfloat16 or x.dim()!=2 or x.shape[1]!=p["k"]:
        return None
    if x.device!=p["device"] or not x.is_contiguous() or x.data_ptr()%16:
        return None
    out=torch.empty((256,p["n"]),dtype=torch.float32 if proj=="cat" else torch.bfloat16,device=x.device)
    _s6a5b_call(_S6A5B[S6A5B_ARM],p,x,out)
    return out
def _s6a5b_cat(state,x,data,tail):
    if tail!="cat":
        return None
    y=_s6a5b(state,"cat",x)
    return None if y is None else _v8_tail(y,None,state,data)
'''


def cuda_source():
    operands = ','.join('%'+str(i) for i in range(128))
    outputs = ','.join(f'"+f"(d[{i}])' for i in range(128))
    mma = (' asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; '
           'wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {'+operands+
           '}, %128, %129, p, 1, 1, 0, 0; }"\n : '+outputs+'\n : "l"(a),"l"(b));')
    return CUDA.replace('@MMA@', mma)


def sub(src, old, new):
    assert src.count(old) == 1, (old, src.count(old))
    return src.replace(old, new)


def generate(src, ship=False, bits=0):
    assert bits in range(16), bits
    assert '# okc: problem=muse_decode_layer' == src.splitlines()[0]
    src = sub(src, ' for name,fn in jobs:\n', ' if S6A5B_BUILD:\n  jobs.append(("s6a5b",_s6a5b_build))\n for name,fn in jobs:\n')
    src = sub(src, ' _tinys_prepare()\n', ' _tinys_prepare()\n _s6a5b_prepare(state)\n')
    for var, proj, x in [('qkv','qkv','data["attn_input"]'), ('attn_output','o','o'), ('ffn_output','unpooler','expert_output')]:
        old = f' {var}=_rq_gemm(state,"{proj}",{x})\n'
        src = sub(src, old, f' {var}=_s6a5b(state,"{proj}",{x})\n if {var} is None:\n '+old)
    old = ' g2out=_rq_pooler_tail(state,ffn_input,data,tail)\n'
    src = sub(src, old, ' g2out=_s6a5b_cat(state,ffn_input,data,tail)\n if g2out is None:\n '+old)
    py = PYTHON.replace('_S6A5B_KEYS = (1, 2)', '_S6A5B_KEYS = (1,)') if ship else PYTHON
    py = py.replace('S6A5B = 0\n', f'S6A5B = {bits}\n')
    if ship and bits:
        py = py.replace('S6A5B_BUILD = 15\n', f'S6A5B_BUILD = {bits}\n')
    src += '\n'+py+'\n_S6A5B_CPP = '+repr(CPP)+'\n_S6A5B_CUDA = r"""'+cuda_source()+'"""\n'
    src = sub(src, src.splitlines()[1], '# nonce s6a5b-ab k3_tinys7_c1 + raw BF16 Dense192x256; S6A5B=0')
    ast.parse(src)
    assert src.isascii() and '\r' not in src
    assert len(src.encode()) < 1_000_000, len(src.encode())
    return src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ship', action='store_true')
    ap.add_argument('--bits', type=int, choices=range(16), default=0)
    ap.add_argument('--check', action='store_true', help='verify existing output without writing')
    args = ap.parse_args()
    base = BASE.read_bytes().decode('ascii')
    ship,bits=args.ship,args.bits
    if args.check:
        assignments={n.targets[0].id:n.value for n in ast.parse(OUT.read_text()).body
                     if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        ship=ast.literal_eval(assignments['_S6A5B_KEYS'])==(1,)
        bits=ast.literal_eval(assignments['S6A5B'])
    src = generate(base, ship, bits)
    if args.check:
        assert OUT.read_bytes() == src.encode('ascii'), 'stale generated file'
    else:
        OUT.write_bytes(src.encode('ascii'))
    print({'bytes':len(src),'net_bytes':len(src)-len(base),'headroom':1_000_000-len(src),
           'sha256':hashlib.sha256(src.encode()).hexdigest(),'ship':ship,'bits':bits})


if __name__ == '__main__':
    main()

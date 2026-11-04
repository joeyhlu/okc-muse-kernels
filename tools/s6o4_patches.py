#!/usr/bin/env python3
"""S6O4: anchored, default-off deferred-scalar T<=8 lab and ship generator.

Only the requested output is written. --check also removes every insertion and
requires byte equality with the base (after newline normalization), then checks
the selector-off prepare/kernel ASTs. No importing the layer or CUDA execution.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"


def sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:120], s.count(old), count)
    return s.replace(old, new)


# A paired activation reuses each exact P12 weight load. The GD decode and
# escape loop are copied unchanged. DS=2 uses one exponent-window byte per
# fragment lane: this makes adjacent hi/lo rows losslessly packable even though
# their exponent windows differ by ~8. There is no weight precision conversion.
CUDA_ARGS = r'''
  int nbx;
  const uint16_t* xlo;
  const uint32_t* bases;
  const float* as; const float* beta;
  float* avec; float* kg; float* moments;
'''
CUDA_STORE = r'''
if constexpr (DS == 2) {
  if (row < 8192) {
    A.avec[(size_t)tok * 8192 + row] = v;
    ds_tile[tok * 64 + (row & 63)] = v;
  } else if ((row & 1) == 0) {
    // Adjacent rows differ in lane bit 2; both are in this CTA's red tile.
    A.kg[(size_t)tok * 4352 + (row - 8192) / 2] = v + cta_value(i ^ 4);
  }
} else if constexpr (OUT == 1) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + row] = v;
'''
CUDA_MOMENTS = r'''
    if constexpr (DS == 2) {
      __syncthreads();
      if (bx < 128 && warp < A.M) {
        float aa = 0.f, sa = 0.f, bb = 0.f;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
          int col = bx * 64 + lane + j * 32;
          float av = ds_tile[warp * 64 + lane + j * 32];
          float bv = __fmul_rn(A.beta[col], av);
          float sv = A.as[(size_t)warp * 8192 + col];
          aa = fmaf(av, av, aa); sa = fmaf(sv, bv, sa); bb = fmaf(bv, bv, bb);
        }
        #pragma unroll
        for (int d = 16; d; d >>= 1) {
          aa += __shfl_xor_sync(0xffffffffu, aa, d);
          sa += __shfl_xor_sync(0xffffffffu, sa, d);
          bb += __shfl_xor_sync(0xffffffffu, bb, d);
        }
        if (lane == 0) {
          float* p = A.moments + ((size_t)warp * 128 + bx) * 3;
          p[0] = aa; p[1] = sa; p[2] = bb;
        }
      }
    }
'''

CUDA_EXTRA = r'''
// The body below is vf_router32_kernel<MT>, copied from SBB1_CU by the
// generator. Only block indices become a persistent job id.
@ROUTER@

template<int MT>
__global__ void __launch_bounds__(256, (MT == 1 ? 2 : 1)) branch_b(const __grid_constant__ Args A) {
  // Virtual GD grid has 256 tiles. The physical side branch has 96 CTAs.
  for (int bx = blockIdx.x; bx < 256; bx += gridDim.x) {
    gd_tile<1,1,1,8,16,2,1,2,1,0,0,1>(A, bx);
    __syncthreads(); // all readers of red/redlo finish before tile reuse
  }
  for (int job = blockIdx.x; job < 128; job += gridDim.x)
    router_tile<MT>(A.x32, A.w32, A.part, A.M, job);
}
__global__ void __launch_bounds__(256, 2) extended_o(const __grid_constant__ Args A) {
  gd_tile<1,1,4,2,8,2,1,2,6,0,0,2>(A, blockIdx.x);
}
__global__ void __launch_bounds__(64, 8) input_u(const float* s, const float* alpha,
    const float* aw, uint16_t* hi, uint16_t* lo, float* u, float* as, float* ss) {
  __shared__ float sums[2];
  int col = blockIdx.x * 64 + threadIdx.x, tok = blockIdx.y;
  size_t off = (size_t)tok * 8192 + col;
  float a = __fmul_rn(alpha[col], s[off]), v = __fmul_rn(aw[col], s[off]);
  __nv_bfloat16 h = __float2bfloat16_rn(v);
  reinterpret_cast<__nv_bfloat16*>(hi)[off] = h;
  reinterpret_cast<__nv_bfloat16*>(lo)[off] = __float2bfloat16_rn(v - __bfloat162float(h));
  u[off] = v; as[off] = a;
  float x = a * a;
  #pragma unroll
  for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffffu, x, d);
  if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = x;
  __syncthreads();
  if (threadIdx.x == 0) ss[tok * 128 + blockIdx.x] = sums[0] + sums[1];
}
struct Combine {
  const float *as, *beta, *a, *kg, *mom, *ss, *pu, *ru;
  float *scalars, *stream, *logits;
  __nv_bfloat16* q;
  int M;
};
__global__ void __launch_bounds__(256, 2) combine_e(const __grid_constant__ Combine A) {
  __shared__ float red[4][8];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, b = blockIdx.x;
  // CTAs [0,M) reduce one token each. Others still join the grid barrier.
  if (b < A.M) {
    float aa=0.f, sa=0.f, bb=0.f, ss=0.f;
    if (tid < 128) {
      const float* p = A.mom + ((size_t)b * 128 + tid) * 3;
      aa=p[0]; sa=p[1]; bb=p[2]; ss=A.ss[b * 128 + tid];
    }
    #pragma unroll
    for (int d=16; d; d>>=1) {
      aa+=__shfl_xor_sync(0xffffffffu,aa,d); sa+=__shfl_xor_sync(0xffffffffu,sa,d);
      bb+=__shfl_xor_sync(0xffffffffu,bb,d); ss+=__shfl_xor_sync(0xffffffffu,ss,d);
    }
    if (lane==0) { red[0][warp]=aa; red[1][warp]=sa; red[2][warp]=bb; red[3][warp]=ss; }
    __syncthreads();
    if (tid==0) {
      aa=sa=bb=ss=0.f;
      #pragma unroll
      for (int w=0; w<8; ++w) { aa+=red[0][w]; sa+=red[1][w]; bb+=red[2][w]; ss+=red[3][w]; }
      float ca=rsqrtf(__fadd_rn(aa/8192.f,1e-8f));
      float n=__fadd_rn(__fadd_rn(ss,__fmul_rn(2.f*ca,sa)),__fmul_rn(__fmul_rn(ca,ca),bb));
      // A negative/nonfinite norm fails numerics visibly; never silently clamp it.
      A.scalars[b*2]=ca; A.scalars[b*2+1]=rsqrtf(__fadd_rn(n/8192.f,1e-5f));
    }
  }
  // CUDA cooperative launch certifies whole-grid residency. No custom spin,
  // counter generation, early exit, cleanup/reset loop, or trap is involved.
  cooperative_groups::this_grid().sync();
  for (int idx=b*256+tid; idx<A.M*8192; idx+=gridDim.x*256) {
    int t=idx/8192, c=idx%8192; float ca=A.scalars[t*2];
    A.stream[idx]=__fadd_rn(A.as[idx],__fmul_rn(ca,__fmul_rn(A.beta[c],A.a[idx])));
  }
  for (int idx=b*256+tid; idx<A.M*4352; idx+=gridDim.x*256) {
    int t=idx/4352, c=idx%4352; float ca=A.scalars[t*2], c1=A.scalars[t*2+1], v;
    if (c<4096) {
      v=__fmul_rn(c1,__fadd_rn(A.pu[t*4096+c],__fmul_rn(ca,A.kg[idx])));
      A.q[t*4096+c]=__float2bfloat16_rn(v);
    } else {
      float r=0.f;
      #pragma unroll
      for (int k=0; k<8; ++k) r+=A.ru[(k*A.M+t)*256+c-4096];
      v=__fmul_rn(c1,__fadd_rn(r,__fmul_rn(ca,A.kg[idx])));
      A.logits[t*256+c-4096]=v;
    }
  }
}
} // namespace s6o4

// Tensor addresses are resolved only in the timed call, never in prepare.
int s6_run_raw(int op, void** p, int M, unsigned tb, cudaStream_t st) {
  using namespace s6o4;
  if (op==0) {
    input_u<<<dim3(128,M),64,0,st>>>((float*)p[0],(float*)p[1],(float*)p[2],
      (uint16_t*)p[3],(uint16_t*)p[4],(float*)p[5],(float*)p[6],(float*)p[7]);
  } else if (op==1 || op==2) {
    Args a={}; a.w=(uint4*)p[0]; a.esc=(uint32_t*)p[1]; a.x=(uint16_t*)p[2]; a.M=M;
    a.tb=tb; a.trig=0;
    if (op==1) {
      a.xlo=(uint16_t*)p[3]; a.out=p[4]; a.x32=(float*)p[5]; a.w32=(float*)p[6]; a.part=(float*)p[7];
      a.K=8192; a.ldo=4096; a.nbx=256;
      if (M==1) branch_b<1><<<96,256,0,st>>>(a); else branch_b<8><<<96,256,0,st>>>(a);
    } else {
      a.bases=(uint32_t*)p[3]; a.as=(float*)p[4]; a.beta=(float*)p[5];
      a.avec=(float*)p[6]; a.kg=(float*)p[7]; a.moments=(float*)p[8]; a.K=1024; a.nbx=264;
      cudaLaunchConfig_t cfg={}; cfg.gridDim=dim3(264); cfg.blockDim=dim3(256); cfg.stream=st;
      cudaLaunchAttribute attr={}; attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
      attr.val.programmaticStreamSerializationAllowed=1; cfg.attrs=&attr; cfg.numAttrs=1;
      int err=(int)cudaLaunchKernelEx(&cfg,extended_o,a); if(err) return err;
    }
  } else if (op==3) {
    Combine a={(float*)p[0],(float*)p[1],(float*)p[2],(float*)p[3],(float*)p[4],(float*)p[5],
      (float*)p[6],(float*)p[7],(float*)p[8],(float*)p[9],(float*)p[10],(__nv_bfloat16*)p[11],M};
    void* args[]={&a};
    int e=(int)cudaLaunchCooperativeKernel((void*)combine_e,dim3(32),dim3(256),args,0,st);
    if(e) return e;
  } else return (int)cudaErrorInvalidValue;
  return (int)cudaGetLastError();
}
int s6_info_raw(int k, int64_t* r) {
  using namespace s6o4;
  const void* f = k==0 ? (const void*)input_u : k==1 ? (const void*)branch_b<1> :
    k==2 ? (const void*)branch_b<8> : k==3 ? (const void*)extended_o : (const void*)combine_e;
  cudaFuncAttributes a; int e=(int)cudaFuncGetAttributes(&a,f); if(e) return e;
  int occ=0; e=(int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,f,k==0?64:256,0);
  r[0]=a.numRegs; r[1]=a.localSizeBytes; r[2]=a.sharedSizeBytes; r[3]=occ; return e;
}
'''

CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <vector>
int s6_run_raw(int,void**,int,unsigned,cudaStream_t);
int s6_info_raw(int,int64_t*);
void s6_run(int64_t op,std::vector<torch::Tensor> a,int64_t m,int64_t tb) {
  const int need=op==0?8:op==1?8:op==2?9:op==3?12:0;
  TORCH_CHECK(need && a.size()==need && m>=1 && m<=8,"s6o4: op/arguments/T");
  void* p[12]; for(int i=0;i<need;++i) {
    TORCH_CHECK(a[i].is_cuda() && a[i].is_contiguous(),"s6o4: CUDA contiguous tensor required");
    bool bf=op==0?(i==3||i==4):op==1?(i==2||i==3):op==2?i==2:i==11;
    bool ii=(op==1 && i<2)||(op==2 && (i<2||i==3));
    TORCH_CHECK(a[i].scalar_type()==(bf?at::kBFloat16:ii?at::kInt:at::kFloat),"s6o4: dtype at ",op,"/",i);
    TORCH_CHECK(a[i].device()==a[0].device(),"s6o4: mixed devices");
    p[i]=a[i].data_ptr();
  }
  auto size=[&](int i,int64_t n){TORCH_CHECK(a[i].numel()==n,"s6o4: size at ",op,"/",i);};
  if(op==0) {
    for(int i=0;i<7;++i) size(i,(i==1||i==2)?8192:m*8192); size(7,m*128);
  } else if(op==1) {
    size(0,12582912); size(1,65536); size(2,m*8192); size(3,m*8192); size(4,m*4096);
    size(5,m*8192); size(6,256*8192); size(7,8*m*256);
  } else if(op==2) {
    size(0,6488064); size(1,67584); size(2,m*1024); size(3,33792); size(4,m*8192);
    size(5,8192); size(6,m*8192); size(7,m*4352); size(8,m*128*3); size(9,8192);
  } else {
    const int64_t sz[]={m*8192,8192,m*8192,m*4352,m*128*3,m*128,m*4096,8*m*256,m*2,m*8192,m*256,m*4096};
    for(int i=0;i<12;++i) size(i,sz[i]);
  }
  int e=s6_run_raw((int)op,p,(int)m,(unsigned)tb,at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(!e,"s6o4: ",cudaGetErrorString((cudaError_t)e));
}
std::vector<int64_t> s6_info(int64_t k) {
  TORCH_CHECK(k>=0 && k<5,"s6o4: kernel index"); std::vector<int64_t> r(4);
  int e=s6_info_raw((int)k,r.data()); TORCH_CHECK(!e,"s6o4 info: ",e); return r;
}
'''

SOURCE_BUILDER = r'''
def _s6o4_src():
 s=GD_CUDA_SRC.split("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>\nstatic int launch",1)[0]
 def rep(a,b,n=1):
  nonlocal s
  assert s.count(a)==n,(a[:90],s.count(a),n)
  s=s.replace(a,b)
 rep("#include <cuda_runtime.h>","#include <cuda_runtime.h>\n#include <cooperative_groups.h>")
 rep("namespace gd {","namespace s6o4 {")
 rep("struct Args {","struct Args {"+_S6O4_ARGS)
 rep("int VF = 0, int CAT = 0>","int VF = 0, int CAT = 0, int DS = 0>")
 rep("__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A)",
     "__device__ __forceinline__ void gd_tile(const Args& A, int bx)")
 s=s.replace("(int)blockIdx.x","bx").replace("(int)(gridDim.x * gridDim.y)","A.nbx")
 rep("  __shared__ float red[NW][MT * 4][32];", "  __shared__ float red[NW][MT * 4][32];\n"
     "  __shared__ float redlo[DS == 1 ? NW : 1][MT * 4][32];\n"
     "  __shared__ float ds_tile[DS == 2 ? 8 * 64 : 1];")
 rep("const uint32_t em = lde", "const uint32_t ds_tb = DS == 2 ? A.bases[((bx * WR + warp / WK) * 32) + lane] : A.tb;\nconst uint32_t em = lde")
 rep("(CAT && s >= KUP) ? A.tbr : A.tb", "(CAT && s >= KUP) ? A.tbr : ds_tb")
 rep("uint4 X[XB][MT][2];", "uint4 X[XB][MT][2];\nuint4 XL[DS == 1 ? XB : 1][MT][2];")
 # Clone every pipeline x load, with identical offsets into the lo tensor.
 lines=[]
 for line in s.splitlines():
  lines.append(line)
  if line.startswith("X[") and " = ldxk<XP>(" in line:
   lhs,rhs=line.split(" = ",1)
   arg=rhs[len("ldxk<XP>("):-2]
   rhs="ldxk<XP>(reinterpret_cast<const uint4*>(A.xlo) + (("+arg+") - reinterpret_cast<const uint4*>(A.x)));"
   lines.append("if constexpr (DS == 1) "+lhs.replace("X[","XL[",1)+" = "+rhs)
 s="\n".join(lines)+"\n"
 rep("float acc[NC][MT][4];", "float acc[NC][MT][4];\nfloat acl[DS == 1 ? NC : 1][MT][4];")
 rep("for (int i = 0; i < 4; ++i) acc[c][mt][i] = 0.f;", "for (int i = 0; i < 4; ++i) { acc[c][mt][i] = 0.f; if constexpr (DS == 1) acl[c][mt][i] = 0.f; }")
 a="mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);"
 rep(a,a+"\nif constexpr (DS == 1) {\nconst uint4 lv = XL[xs][mt][q >> 1];\n"
     "mma16816(acl[q % NC][mt], a[4*q], a[4*q+1], a[4*q+2], a[4*q+3], (q&1)?lv.z:lv.x, (q&1)?lv.w:lv.y);\n}")
 a="if constexpr (CAT > 0) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;"
 rep(a,a+"\nif constexpr (DS == 1) redlo[warp][mt * 4 + i][lane] = acl[0][mt][i] + acl[1][mt][i];")
 a="for (int k2 = 0; k2 < WK; ++k2) v += red[w2 * WK + k2][mt * 4 + c][l];\nreturn v;"
 rep(a,"for (int k2 = 0; k2 < WK; ++k2) v += red[w2 * WK + k2][mt * 4 + c][l];\n"
     "if constexpr (DS == 1) { float vl=0.f;\n#pragma unroll\nfor (int k2=0;k2<WK;++k2) vl+=redlo[w2*WK+k2][mt*4+c][l];\nv+=vl; }\nreturn v;")
 rep("if constexpr (OUT == 1) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + row] = v;",_S6O4_STORE)
 a="    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));"
 rep(a,a+_S6O4_MOMENTS)
 return s+_S6O4_EXTRA
'''

PYTHON = r'''
S6O4=0
S6O4_BUILD=False
_S6O4_BANK={}

def _s6o4_build():
 from torch.utils.cpp_extension import load_inline
 src=_s6o4_src()
 tag=hashlib.sha1((_S6O4_CPP+src).encode()).hexdigest()[:12]
 return load_inline(name=f"s6o4_{tag}",cpp_sources=_S6O4_CPP,cuda_sources=src,
  functions=["s6_run","s6_info"],extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a","-Xptxas=-v"],verbose=False)

def _s6o4_vectors(state):
 # Exact base accessors: _sbb1_prepare, not a guessed state/module handle.
 op=state["attn_boundary"]
 op._ensure_cached_weights(torch.bfloat16)
 dense=state["dense"]
 assert not op._post_norm.has_weight and op._pre_norm.has_weight and op._use_residual_gate
 a=op._cc_alpha.float().contiguous().reshape(-1)
 b=op._cc_beta.float().contiguous().reshape(-1)
 wf=op._cc_pre_weight.to(torch.bfloat16).contiguous().reshape(-1)
 if a.numel()==1:
  a=a.expand(8192).contiguous(); b=b.expand(8192).contiguous()
 assert a.numel()==b.numel()==wf.numel()==8192
 assert tuple(dense["o"].shape)==(8192,1024) and tuple(dense["pooler"].shape)==(4096,8192)
 assert tuple(state["router"].shape)==(256,8192)
 return {"a":a,"b":b,"wf":wf,"aw":(a.double()*wf.double()).float().contiguous(),
  "bw":(b.double()*wf.double()).float().contiguous(),"a2":a*a,"ab":a*b,"b2":b*b}

def _s6o4_compose(state,v):
 dense=state["dense"]
 o=dense["o"]
 d=(v["b"].double()*v["wf"].double())[:,None]*o.double()
 rows=[o]; bases=[torch.full((8192,),_gd_window_base(o),dtype=torch.int32,device=o.device)]
 errs={}
 for name,w in (("pooler",dense["pooler"]),("router",state["router"])):
  z=w.double()@d
  hi=z.to(torch.bfloat16); lo=(z-hi.double()).to(torch.bfloat16)
  rel=float((hi.double()+lo.double()-z).norm()/z.norm().clamp_min(1e-300))
  assert rel<2**-14,(name,rel)
  errs[name]=rel
  pair=torch.stack((hi,lo),dim=1).reshape(-1,1024).contiguous()
  hb,lb=_gd_window_base(hi),_gd_window_base(lo)
  bas=torch.empty((hi.shape[0],2),dtype=torch.int32,device=o.device); bas[:,0]=hb; bas[:,1]=lb
  rows.append(pair); bases.append(bas.reshape(-1))
 return torch.cat(rows,dim=0).contiguous(),torch.cat(bases),errs

def _s6o4_alloc(dev):
 # Per-token-size banks: no tensor derived from data survives a call without
 # being fully overwritten. Stable addresses are ordinary graph scratch.
 bank={}
 for m in range(1,9):
  def e(shape,dtype=torch.float32): return torch.empty(shape,dtype=dtype,device=dev)
  bank[m]={"hi":e((m,8192),torch.bfloat16),"lo":e((m,8192),torch.bfloat16),
   "u":e((m,8192)),"as":e((m,8192)),"ss":e((m,128)),"pu":e((m,4096)),
   "ru":e((8,m,256)),"a":e((m,8192)),"kg":e((m,4352)),"mom":e((m,128,3)),
   "sc":e((m,2)),"stream":e((m,8192)),"logits":e((m,256)),"q":e((m,4096),torch.bfloat16)}
 return bank

def _s6o4_prepare_data(state):
 v=_s6o4_vectors(state)
 rows,bases,errs=_s6o4_compose(state,v)
 # Copy the proven pooler pack/accessor. Both T1 and T8 use (WR,WK,KU,S)=(1,8,16,1).
 st=state["gd"]
 r=st["route"].get(("pooler",8))
 assert r is not None and r["N"]==4096 and r["K"]==8192 and r["nbx"]==256
 assert r["desc"].startswith("m1(1,8,16,1)s2x1")
 assert state["v8"] is not None
 pk=_s6o4_pack(rows,bases)
 return {"v":v,"o":pk,"p":r,"router":state["router"],"sc":_s6o4_alloc(rows.device),"errors":errs}

def _s6o4_prepare(state,t0):
 if not (S6O4_BUILD or S6O4): return
 t1=time.time()
 h=_s6o4_prepare_data(state)
 h["mod"]=_s6o4_build()
 h["info"]=[list(h["mod"].s6_info(k)) for k in range(5)]
 dev=state["router"].device
 props=torch.cuda.get_device_properties(dev)
 torch.cuda.synchronize()
 _s6o4_install(state,h,props.multi_processor_count,t0,t1)

def _s6o4_install(state,h,nsm,t0,t1):
 assert len(h["info"])==5 and all(r[1]==0 for r in h["info"]),("s6o4 local memory/spills",h["info"])
 assert h["info"][4][3]*nsm>=32,"s6o4 E cooperative residency"
 h["prepare_seconds"]=time.time()-t1
 h["total_prepare_seconds"]=time.time()-t0
 assert h["total_prepare_seconds"]<300,("s6o4 prepare deadline",h["total_prepare_seconds"])
 state["s6o4"]=h
 _S6O4_BANK[id(state)]=h
 print(f"[s6o4] prepare {h['prepare_seconds']:.3f}s total {h['total_prepare_seconds']:.3f}s info {h['info']} composition {h['errors']}",flush=True)

def _s6o4_select(state,tokens):
 if not S6O4 or not 1<=tokens<=8: return None
 assert S6O4==1,"S6O4 accepts only 0 (base) and 1 (DS)"
 h=state["s6o4"]
 return h,h["sc"][tokens]

def _s6o4_begin(data,state):
 selected=_s6o4_select(state,data["config"]["tokens"])
 if selected is None: return None
 h,z=selected; m=data["config"]["tokens"]; v=h["v"]; r=h["p"]
 main_stream,side_stream=torch.cuda.current_stream(),state["side_stream"]
 side_stream.wait_stream(main_stream)
 with torch.cuda.stream(side_stream):
  h["mod"].s6_run(0,[data["stream"],v["a"],v["aw"],z["hi"],z["lo"],z["u"],z["as"],z["ss"]],m,0)
  h["mod"].s6_run(1,[r["w"],r["esc"],z["hi"],z["lo"],z["pu"],z["u"],h["router"],z["ru"]],m,r["tb"])
 if not torch.cuda.is_current_stream_capturing():
  data["stream"].record_stream(side_stream)
 return selected

def _s6o4_finish(selected,o,state,data):
 h,z=selected; m=data["config"]["tokens"]; v=h["v"]; r=h["o"]
 # as is produced by the side-branch input kernel. A full side join here
 # would serialize B before O; compute a1*s in O directly from the stream.
 h["mod"].s6_run(2,[r["w"],r["esc"],o,r["bases"],data["stream"],v["b"],z["a"],z["kg"],z["mom"],v["a"]],m,0)
 main_stream,side_stream=torch.cuda.current_stream(),state["side_stream"]
 main_stream.wait_stream(side_stream)
 if not torch.cuda.is_current_stream_capturing():
  for tensor in (z["pu"],z["ru"],z["ss"],z["as"]): tensor.record_stream(main_stream)
 h["mod"].s6_run(3,[z["as"],v["b"],z["a"],z["kg"],z["mom"],z["ss"],z["pu"],z["ru"],z["sc"],z["stream"],z["logits"],z["q"]],m,0)
 return z["stream"],_v8_tail(z["q"],z["logits"],state,data)
'''


def get_assign(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def function_source(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"


def pack_source(base):
    s = function_source(base, "_gd_pack")
    s = sub(s, "def _gd_pack(wv,wr,wk,ku,s,base):", "def _s6o4_pack(wv,bases):\n wr,wk,ku,s=4,2,8,1")
    s = sub(s, " d=h7-base", " lane=torch.arange(32,device=dev)\n row=(torch.arange(nrt,device=dev)[:,None]*16+(lane>>2)[None,:])\n base=bases[row].reshape(nrt,1,32,1,1)\n d=h7-base")
    s = sub(s, "(h7[rt,u,ln,qq,jv]^base)", "(h7[rt,u,ln,qq,jv]^base[rt,0,ln,0,0])")
    s = sub(s, '"tb":base*0x01010101', '"tb":0,"bases":_gd_i32((base.reshape(nrt,32).to(torch.int64)*0x01010101)).contiguous()')
    return s


def runtime(base):
    cu = get_assign(base, "SBB1_HOST")
    start = cu.index("template <int MT>\n__global__ void __launch_bounds__(256, 6) vf_router32_kernel")
    end = cu.index("// x through L1", start)
    router = cu[start:end]
    router = sub(router, "__global__ void __launch_bounds__(256, 6) vf_router32_kernel", "__device__ __forceinline__ void router_tile")
    router = sub(router, "float* __restrict__ part, int m)", "float* __restrict__ part, int m, int job)")
    router = router.replace("blockIdx.y", "(job / 16)").replace("blockIdx.x", "(job % 16)")
    # O needs input stream/alpha, not side-produced as; otherwise O races U.
    args = CUDA_ARGS.replace("const float* as;", "const float* as; const float* alpha;")
    moments = CUDA_MOMENTS.replace("float sv = A.as[(size_t)warp * 8192 + col];", "float sv = __fmul_rn(A.alpha[col], A.as[(size_t)warp * 8192 + col]);")
    extra = CUDA_EXTRA.replace("@ROUTER@", router).replace("a.moments=(float*)p[8];", "a.moments=(float*)p[8]; a.alpha=(float*)p[9];")
    cpp = CPP.replace("op==2?9", "op==2?10")
    constants = dict(_S6O4_ARGS=args, _S6O4_STORE=CUDA_STORE, _S6O4_MOMENTS=moments,
                     _S6O4_EXTRA=extra, _S6O4_CPP=cpp)
    return "\n# S6O4 BEGIN\n" + "".join(f"{k}={v!r}\n" for k, v in constants.items()) + SOURCE_BUILDER + pack_source(base) + PYTHON + "# S6O4 END\n"


def generate(base, ship=False):
    additions = runtime(base)
    # Exactly three anchored base edits; all old branches stay verbatim.
    edits = []
    a = " _tinys_prepare()\n builds="
    b = " _tinys_prepare()\n _s6o4_prepare(state,t0)\n builds="
    edits.append((a, b))
    a = ' dense=state["dense"]\n qkv=_rq_gemm'
    b = ' dense=state["dense"]\n _ds=_s6o4_begin(data,state)\n qkv=_rq_gemm'
    edits.append((a, b))
    begin = base.index(' attn_output=_rq_gemm(state,"o",o)\n')
    end = base.index(' _SU1_FLAG["k3trig"]=_su1_k3trig(state,tokens)\n', begin)
    old = base[begin:end]
    new = (' if _ds is not None:\n'
           '  stream,_ds_tail=_s6o4_finish(_ds,o,state,data)\n'
           '  expert_x,topk_weights,topk_ids,local_ids=_ds_tail\n'
           ' else:\n' + "".join(" " + line + "\n" for line in old.splitlines()))
    edits.append((old, new))
    out = base
    for a, b in edits:
        out = sub(out, a, b)
    out += additions
    # Ship is default-off too. Select S6O4=1 before prepare/capture to enable.
    # It shares the base source strings instead of duplicating the GD kernel.
    return out, edits, additions


def check_output(base, out, edits, additions):
    assert out.endswith(additions)
    restored = out[:-len(additions)]
    for a, b in reversed(edits):
        restored = sub(restored, b, a)
    assert restored == base, "selector-off restoration differs from base"
    compile(out, "s6o4 generated", "exec")
    ns = {"GD_CUDA_SRC": get_assign(base, "GD_CUDA_SRC")}
    tree = ast.parse(additions)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("_S6O4_") and isinstance(n.value, ast.Constant)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_s6o4_src"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "s6o4 source builder", "exec"), ns)
    cu = ns["_s6o4_src"]()
    assert "@ROUTER@" not in cu and "gd_kernel" not in cu
    assert cu.count("cooperative_groups::this_grid().sync()") == 1
    assert "__trap" not in cu
    # Default OFF executes only two pure return paths. All base function bodies
    # and all original source strings are unchanged after removing those hooks.
    at = ast.parse(additions)
    defaults = {n.targets[0].id: ast.literal_eval(n.value) for n in at.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("S6O4", "S6O4_BUILD")}
    assert defaults == {"S6O4": 0, "S6O4_BUILD": False}
    for path in ("_s6o4_prepare", "_s6o4_select"):
        first = next(n for n in at.body if isinstance(n, ast.FunctionDef) and n.name == path).body[0]
        assert isinstance(first, ast.If) and isinstance(first.body[0], ast.Return)
    return dict(status="PASS", selector_default=0, off_restoration_sha256=hashlib.sha256(base.encode()).hexdigest(),
                prepare_default_noop=True, bytes=len(out.encode()), cuda_bytes=len(cu.encode()), edits=len(edits), ship_limit=len(out.encode()) < 1_000_000)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--out", type=Path)
    p.add_argument("--ship", action="store_true")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    base = a.base.read_text(encoding="utf-8")
    out, edits, additions = generate(base, a.ship)
    report = check_output(base, out, edits, additions)
    if a.ship:
        assert report["ship_limit"], f"ship exceeds 1 MB: {report['bytes']}"
    dest = a.out or ROOT / ("kernels/lab/k3_s6o4_ship.py" if a.ship else "kernels/lab/k3_s6o4_ab.py")
    dest.write_text(out, encoding="utf-8", newline="\n")
    report["out"] = str(dest)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

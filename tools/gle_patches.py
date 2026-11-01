"""Generate the default-off native GLE laboratory arm. No GPU work here.

Design credit: keell0renz's deferred normalization (Apache-2.0), built on
alygg77's projection work (Apache-2.0). CUDA implementation uses our GD core.
The FP32 router retains SBB's intervening BF16 rounding, rather than silently
using the reference port's different residual arithmetic.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/lab/k3_gle_ab.py"


def sub(s, a, b):
    assert s.count(a) == 1, ("GLE exact anchor", a[:100], s.count(a))
    return s.replace(a, b, 1)


EPILOGUE = r'''
// GLE: only OUT=2, the exact (o,1) GD geometry, enters this epilogue.
if constexpr (OUT == 2) {
  static_assert(MT == 1 && WR == 1 && WK == 8 && S == 1, "GLE o route");
  if (tok == 0) {
    const float p = __bfloat162float(__float2bfloat16_rn(v));
    const int gi = A.ep.channel ? row : 0;
    const float a = A.ep.alpha[gi] * A.ep.stream[row];
    const float b = A.ep.beta[gi] * p;
    const float pre = __bfloat162float(A.ep.pre[row]);
    A.ep.raw[row] = a;
    A.ep.raw[8192 + row] = p;
    A.ep.uv[row] = __float2bfloat16_rn(a * pre);
    A.ep.uv[8192 + row] = __float2bfloat16_rn(b * pre);
    gle_q[0][row & 15] = p * p;
    gle_q[1][row & 15] = a * a;
    gle_q[2][row & 15] = a * b;
    gle_q[3][row & 15] = b * b;
  }
}
'''

FINISH = r'''
    if constexpr (OUT == 2) {
      // All 256 threads remain here. The 16 token-zero writers covered q.
      __syncthreads();
      if (threadIdx.x < 32) {
        float q[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) q[k] = lane < 16 ? gle_q[k][lane] : 0.f;
        #pragma unroll
        for (int d = 8; d; d >>= 1)
          #pragma unroll
          for (int k = 0; k < 4; ++k) q[k] += __shfl_xor_sync(0xffffffffu, q[k], d);
        if (lane == 0) A.ep.mom[blockIdx.x] = make_float4(q[0], q[1], q[2], q[3]);
      }
    }
'''

COMMON = r'''
struct Gle {
  const float *stream, *alpha, *beta;
  const __nv_bfloat16* pre;
  __nv_bfloat16* uv;
  float* raw;
  float4* mom;
  int channel;
};
__device__ __forceinline__ float gle_bf(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
// A warp reduces 512 sixteen-column moments. No counters or cross-CTA waits.
__device__ __forceinline__ float2 gle_norm(const float4* mom, float pe, float qe) {
  float4 q = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = threadIdx.x & 31; i < 512; i += 32) {
    const float4 a = __ldcg(mom + i);
    q.x += a.x; q.y += a.y; q.z += a.z; q.w += a.w;
  }
  #pragma unroll
  for (int d = 16; d; d >>= 1) {
    q.x += __shfl_xor_sync(0xffffffffu, q.x, d);
    q.y += __shfl_xor_sync(0xffffffffu, q.y, d);
    q.z += __shfl_xor_sync(0xffffffffu, q.z, d);
    q.w += __shfl_xor_sync(0xffffffffu, q.w, d);
  }
  const float r1 = rsqrtf(q.x * (1.f / 8192.f) + pe);
  const float ss = fmaf(r1, fmaf(r1, q.w, 2.f * q.z), q.y);
  return make_float2(r1, rsqrtf(fmaxf(ss, 0.f) * (1.f / 8192.f) + qe));
}
'''

ROUTER = r'''
// Two virtual rows: raw alpha*s and BF16 P, both stored as FP32.
// Reconstruct the BASE h before FP32 FMA: BF16(r1*P) is nonlinear.
// Projecting u and v separately here would change the router input semantics.
__global__ void __launch_bounds__(256, 6) gle_router(
    Gle e, const float* w, float* part, float* ns, float* hr, float pe, float qe) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = blockIdx.y * 1024;
  const float2 r = gle_norm(e.mom, pe, qe);
  __shared__ float4 xs[256];
  const int c0 = k0 + 4 * threadIdx.x;
  float hv[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = c0 + j;
    const float n = gle_bf(e.raw[8192 + c] * r.x);
    const float s = fmaf(e.beta[e.channel ? c : 0], n, e.raw[c]);
    hv[j] = (s * r.y) * __bfloat162float(e.pre[c]);
    // Only x=0 owns the residual, with y=0..7 owning disjoint 1024 columns.
    if (blockIdx.x == 0) { ns[c] = s; hr[c] = hv[j]; }
  }
  xs[threadIdx.x] = make_float4(hv[0], hv[1], hv[2], hv[3]);
  __syncthreads();
  #pragma unroll 1
  for (int h = 0; h < 2; ++h) {
    const int row = blockIdx.x * 16 + h * 8 + warp;
    const float4* wp = reinterpret_cast<const float4*>(w + (long)row * 8192 + k0);
    float acc = 0.f;
    #pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
      float4 wv[2];
      #pragma unroll
      for (int i = 0; i < 2; ++i) wv[i] = __ldcs(wp + (hh * 2 + i) * 32 + lane);
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const float4 x = xs[(hh * 2 + i) * 32 + lane];
        acc = fmaf(wv[i].x, x.x, acc); acc = fmaf(wv[i].y, x.y, acc);
        acc = fmaf(wv[i].z, x.z, acc); acc = fmaf(wv[i].w, x.w, acc);
      }
    }
    #pragma unroll
    for (int d = 16; d; d >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, d);
    if (lane == 0) part[blockIdx.y * 256 + row] = acc;
  }
}
'''

RAW_HOST = r'''
using namespace gdgle;
extern "C" int gle_o_raw(const void* w, const void* esc, const void* x, void* out,
    uint32_t tb, Gle e, cudaStream_t st) {
  Args a = {}; a.w = (const uint4*)w; a.esc = (const uint32_t*)esc;
  a.x = (const uint16_t*)x; a.out = out; a.M = 1; a.K = 1024; a.ldo = 8192;
  a.tb = tb; a.trig = 1; a.ep = e;
  return launch<2,1,1,8,2,2,1,4,6>(a, 512, 1, 3, st);
}
extern "C" int gle_p_raw(const void* w, const void* esc, const void* x, void* out,
    uint32_t tb, cudaStream_t st) {
  Args a = {}; a.w = (const uint4*)w; a.esc = (const uint32_t*)esc;
  a.x = (const uint16_t*)x; a.out = out; a.M = 2; a.K = 8192; a.ldo = 4096;
  a.tb = tb; a.trig = 1;
  return launch<1,1,1,8,16,2,1,2,7>(a, 256, 1, 3, st);
}
extern "C" int gle_r_raw(Gle e, const float* w, float* part, float* ns, float* hr,
    float pe, float qe, cudaStream_t st) {
  // Ordinary side-stream launch AFTER the o event; no PDL dependency here.
  gle_router<<<dim3(16,8),256,0,st>>>(e,w,part,ns,hr,pe,qe);
  return (int)cudaGetLastError();
}
extern "C" int gle_t_raw(const float* y, const float* lg, const void* nw,
    const float* bias, const int* ls, void* xn, float* tw, int* ids, int* lids,
    const float4* mom, float pe, float qe, cudaStream_t st) {
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(1); cfg.blockDim = dim3(768); cfg.stream = st;
  cudaLaunchAttribute at = {}; at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at.val.programmaticStreamSerializationAllowed = 1; cfg.attrs = &at; cfg.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&cfg, gle_tail<true,8,8>, (const void*)y, 4096L,
    lg, (const float*)nullptr, 256L, 8, 256L, (const __nv_bfloat16*)nw, bias, ls,
    (__nv_bfloat16*)xn, tw, ids, lids, 1.e-5f, 0.1767766952966369f, 1.f, 1.e-15f, mom, pe, qe);
}
'''

# A single tensor-vector ABI avoids baking tensor pointers into prepare().
CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
struct Gle {
 const float *stream,*alpha,*beta; const __nv_bfloat16* pre;
 __nv_bfloat16* uv; float* raw; float4* mom; int channel;
};
extern "C" int gle_o_raw(const void*,const void*,const void*,void*,uint32_t,Gle,cudaStream_t);
extern "C" int gle_p_raw(const void*,const void*,const void*,void*,uint32_t,cudaStream_t);
extern "C" int gle_r_raw(Gle,const float*,float*,float*,float*,float,float,cudaStream_t);
extern "C" int gle_t_raw(const float*,const float*,const void*,const float*,const int*,void*,float*,int*,int*,const float4*,float,float,cudaStream_t);
void gle_run(int64_t kind, std::vector<torch::Tensor> t, int64_t tb, double pe, double qe) {
 TORCH_CHECK(t.size()==19, "GLE tensor bank");
 auto p = [&](int i) { return t[i].data_ptr(); };
 Gle e = {(const float*)p(0),(const float*)p(1),(const float*)p(2),
  (const __nv_bfloat16*)p(3),(__nv_bfloat16*)p(4),(float*)p(5),(float4*)p(6),t[1].numel()==8192};
 auto st=at::cuda::getCurrentCUDAStream(); int err=-1;
 if(kind==0) err=gle_o_raw(p(7),p(8),p(9),p(10),(uint32_t)tb,e,st);
 if(kind==1) err=gle_p_raw(p(7),p(8),p(4),p(11),(uint32_t)tb,st);
 if(kind==2) err=gle_r_raw(e,(const float*)p(7),(float*)p(12),(float*)p(13),(float*)p(14),pe,qe,st);
 if(kind==3) err=gle_t_raw((const float*)p(11),(const float*)p(12),p(7),(const float*)p(8),(const int*)p(9),
  p(15),(float*)p(16),(int*)p(17),(int*)p(18),(const float4*)p(6),pe,qe,st);
 TORCH_CHECK(err==0,"GLE CUDA error ",err);
}
'''

PYTHON = r'''
# GLE native in-file A/B, default off. keell0renz / alygg77, Apache-2.0.
GLE_T1 = False
_GLE = {"mod": None}

def _gle_byte_audit():
 # GD's T=1 xo clamp broadcasts one row across eight lane groups. At M=2
 # the same instructions touch two distinct rows. Count coalesced bytes,
 # not the inflated sum of per-lane requested bytes.
 return {"weight_per_warp_stage":1536,"x_before":128,"x_after":256,
  "pooler_x_before":256*8*16*128,"pooler_x_after":256*8*16*256,
  "eligible":False,"reason":"two-row GD pooler grows distinct activation bytes per stage"}

def _gle_build():
 from torch.utils.cpp_extension import load_inline
 if _GLE["mod"] is None:
  tag=hashlib.sha1((GLE_CPP+GLE_CUDA).encode()).hexdigest()[:12]
  _GLE["mod"]=load_inline(name=f"gle_{tag}",cpp_sources=GLE_CPP,cuda_sources=GLE_CUDA,
   functions=["gle_run"],extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a"],verbose=False)
 return _GLE["mod"]

def _gle_bank(state):
 # Accessors copied from _gd_live, _sbb1_call and _v8_tail; no guessed MoE keys.
 ro=_gd_live(state,"o",1)
 rp=_gd_live(state,"pooler",1)
 if ro is None or rp is None or _SBB1["st"] is None or state.get("v8") is None:
  raise RuntimeError("GLE needs live base GD/o1, GD/pooler1, SBB and tail")
 pw,qw,al,bt,isc,pe,qe=_SBB1["st"]["w"]
 if pw.numel() or isc or qw.numel()!=8192 or al.numel() not in (1,8192) or bt.numel()!=al.numel():
  raise RuntimeError("GLE only supports the base unweighted post-attention norm")
 assert GD_ROUTE[("o",1)]==(1,1,8,2,2,1,6)
 assert GD_ROUTE[("pooler",1)]==(1,1,8,16,2,1,7)
 assert ro["nbx"]==512 and rp["nbx"]==256 and ro["K"]==1024 and rp["K"]==8192
 assert _SBB1["st"]["ok"][1] and _VF["r8"] and VF_ROUTER[1]=="lean"
 assert _ft_flag(1)==8, "GLE copies the live RW=8 tail"
 return {"mod":_gle_build(),"o":ro,"pooler":rp,"pre":qw,"alpha":al,"beta":bt,
  "pe":pe,"qe":qe,"v8":state["v8"],"router":state["router"],"ok":False}

def _gle_selected(state,tokens):
 if not GLE_T1 or tokens!=1:
  return None
 b=state.get("gle")
 if b is None or not b["ok"]:
  raise RuntimeError("GLE_T1 requested but prepare gates did not pass")
 return b

def _gle_buffers(b,x,stream):
 dev=x.device
 def alloc(shape,dtype): return torch.empty(shape,dtype=dtype,device=dev)
 # Allocated per call: CUDA graph capture owns stable storage; no shared scratch.
 uv=alloc((2,8192),torch.bfloat16); raw=alloc((2,8192),torch.float32)
 mom=alloc((512,4),torch.float32); P=alloc((1,8192),torch.bfloat16)
 y=alloc((2,4096),torch.float32); lg=alloc((8,1,256),torch.float32)
 ns=alloc((1,8192),torch.float32); hr=alloc((1,8192),torch.float32)
 xn=alloc((1,4096),torch.bfloat16); tw=alloc((1,8),torch.float32)
 ids=alloc((1,8),torch.int32); lids=alloc((1,8),torch.int32)
 return [stream,b["alpha"],b["beta"],b["pre"],uv,raw,mom,x,x,x,P,y,lg,ns,hr,xn,tw,ids,lids]

def _gle_boundary(b,x,stream,data,state):
 t=_gle_buffers(b,x,stream)
 def run(k,route=None):
  b["mod"].gle_run(k,t,route["tb"] if route else 0,b["pe"],b["qe"])
 ro,rp=b["o"],b["pooler"]
 t[7:10]=[ro["w"],ro["esc"],x]; run(0,ro)
 main_stream,side_stream=torch.cuda.current_stream(),state["side_stream"]
 # Event is recorded after o, before pooler: side waits for complete raw/mom.
 side_stream.wait_stream(main_stream)
 t[7:10]=[rp["w"],rp["esc"],x]; run(1,rp)
 with torch.cuda.stream(side_stream):
  t[7]=b["router"]; run(2)
 main_stream.wait_stream(side_stream)
 if not torch.cuda.is_current_stream_capturing():
  for tensor in (t[5],t[6],t[12],t[13],t[14]): tensor.record_stream(side_stream)
 t[7:10]=[b["v8"]["nw"],b["v8"]["bias"],data["local_start"]]; run(3)
 return (t[15],t[16],t[17],t[18]),t[13],t

def _gle_selftest(state,b):
 dev=state["router"].device
 worst={k:0. for k in ("P","stream","router_h","logits","expert_x","next_attn","next_stream")}
 for seed in (5253,4545,909):
  gen=torch.Generator(device=dev).manual_seed(seed)
  x=torch.randn((1,1024),generator=gen,device=dev).to(torch.bfloat16)
  s=torch.randn((1,8192),generator=gen,device=dev)
  data={"local_start":torch.tensor([seed%8*32],dtype=torch.int32,device=dev)}
  p=_gd(state,"o",x)
  h,ns,hr=_v9_bnd("attn",s,p,torch.float32,state,1)
  y=_gd(state,"pooler",h)
  lg=_v8_logits("f32",h,hr,state)
  ref=_v8_tail(y,lg,state,data)
  got,gs,t=_gle_boundary(b,x,s,data,state)
  z=torch.randn((1,8192),generator=gen,device=dev).to(torch.bfloat16)
  n0,s0,_=_v9_bnd("ffn",ns,z,None,state,1)
  n1,s1,_=_v9_bnd("ffn",gs,z,None,state,1)
  pairs={"P":(t[10],p),"stream":(gs,ns),"router_h":(t[14],hr),"logits":(t[12],lg),
   "expert_x":(got[0],ref[0]),"next_attn":(n1,n0),"next_stream":(s1,s0)}
  errors={k:float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1.e-30)) for k,(a,r) in pairs.items()}
  finite=all(bool(torch.isfinite(a).all()) for a,_ in pairs.values())
  same=bool(torch.equal(got[2],ref[2])) and bool(torch.equal(got[3],ref[3]))
  print(f"[gle] seed {seed} rel vs base {errors} top8 identical {same} finite {finite}",flush=True)
  assert finite and same and errors["P"]==0 and errors["stream"]<1.e-4 and errors["router_h"]<1.e-4
  assert errors["logits"]<1.e-4 and errors["expert_x"]<1.e-2 and errors["next_attn"]<1.e-2 and errors["next_stream"]<1.e-4
  for k,e in errors.items(): worst[k]=max(worst[k],e)
 b["errors"]=worst
 return True

def _gle_prepare(state):
 # Standing byte rule kills this arm before compilation or any new GPU work.
 # Keep the native implementation reviewable, but never silently enable it.
 audit=_gle_byte_audit()
 state["gle"]={"ok":False,"audit":audit}
 if not audit["eligible"]:
  print(f"[gle] DROPPED by byte gate: {audit}",flush=True)
  return
 b=_gle_bank(state)
 state["gle"]=b
 b["ok"]=_gle_selftest(state,b)
'''


def cuda(s):
    tree = ast.parse(s)
    vals = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    gd = vals["GD_CUDA_SRC"]
    gd = gd[:gd.index("__global__ void unpack_kernel(")]
    gd = sub(gd, "namespace gd {\n", "namespace gdgle {\n" + COMMON)
    gd = sub(gd, "struct Args {\n", "struct Args {\nGle ep;\n")
    gd = sub(gd, "  __shared__ float red[NW][MT * 4][32];", "  __shared__ float red[NW][MT * 4][32];\n  __shared__ float gle_q[OUT == 2 ? 4 : 1][OUT == 2 ? 16 : 1];")
    gd = sub(gd, "if (tok < A.M) {\nif constexpr (OUT == 1)", "if (tok < A.M) {\n" + EPILOGUE + "if constexpr (OUT == 1)")
    gd = sub(gd, "    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n",
             "    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n" + FINISH)
    v8 = vals["V8_TAIL_CUDA_SRC"]
    helper = v8[v8.index("template <int EPT, bool CG>"):v8.index("template <bool Y32, int NSMAX, int RW>")]
    tail = v8[v8.index("template <bool Y32, int NSMAX, int RW>"):v8.index("template <int MT>\n__global__ void __launch_bounds__(256) router32_kernel")]
    tail = sub(tail, "tail_kernel3(", "gle_tail(")
    tail = sub(tail, "float eps, float mult, float rscale, float deps) {", "float eps, float mult, float rscale, float deps, const float4* mom, float pe, float qe) {")
    tail = sub(tail, "float v[8];\nif (Y32) {", "const float2 rr=gle_norm(mom,pe,qe);\nfloat v[8];\nif (Y32) {")
    tail = sub(tail, "for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16(v[i]));",
               "for (int i = 0; i < 8; ++i) v[i] = gle_bf((v[i] + rr.x * __ldcg(reinterpret_cast<const float*>(y) + 4096 + tx * 8 + i)) * rr.y);")
    return gd + "\nconstexpr int D=4096, NE=256, TK=8, NT=512;\n" + ROUTER + helper + tail + "\n}\n" + RAW_HOST


def generate(s):
    c = cuda(s)
    assert "'''" not in CPP and "'''" not in c
    addon = "\nGLE_CPP = r'''" + CPP + "'''\nGLE_CUDA = r'''" + c + "'''\n" + PYTHON
    s = sub(s, " _rq_prepare(state)\n", " _rq_prepare(state)\n _gle_prepare(state)\n")
    start = ' attn_output=_rq_gemm(state,"o",o)\n'
    end = ' _SU1_FLAG["k3trig"]=_su1_k3trig(state,tokens)\n'
    assert s.count(start) == s.count(end) == 1
    a,b=s.index(start),s.index(end)
    old=s[a:b]
    new=' gle=_gle_selected(state,tokens)\n if gle is not None:\n  gout,stream,_=_gle_boundary(gle,o,data["stream"],data,state)\n  expert_x,topk_weights,topk_ids,local_ids=gout\n else:\n'
    s=s[:a]+new+''.join(' '+line+'\n' for line in old.splitlines())+s[b:]+addon
    ast.parse(s)
    assert '__trap(' not in c and '__trap()' not in c
    return s


def main():
    s=BASE.read_text(encoding="utf-8")
    result=generate(s)
    OUT.write_bytes(result.encode("utf-8"))
    print(f"GLE generated: {len(result.encode('utf-8'))} bytes, GLE_T1=False")


if __name__ == "__main__":
    main()

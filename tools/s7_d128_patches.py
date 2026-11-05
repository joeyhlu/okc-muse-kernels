#!/usr/bin/env bash
""":"
# Self-contained Linux compile/CPU/SASS gate; no device is made visible.
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
exec python -I -B "$0" --linux-gate "$@"
":"""
from __future__ import annotations

"""S7-D128: one default-off, fused lossless RQ cat/128 experiment.

Run --generate, --check, or --report with Python. Run this SAME file with
bash under tools/local/cudaenv.sh for the CPU/compile/SASS gate. Results go
to s7_d128_patches.json; Linux compiler products remain in a fresh /tmp dir.
No GPU, judge, git, notes, secret, or base-file mutation is performed here.

Kill reasons FIRST: RS decode latency can erase the stream saving; the
larger cluster kernel can spill or fail one-wave admission. The gate checks
zero spills, and prepare requires 36 active clusters before enabling it.
This is not RQP128: pooler AND router hi/lo stay in one FP32-output launch.
This is not R-dnx-1: packed weights decode directly into RS registers; no
decoded-weight STS ring. VD's dedicated DSMEM push reduction is retained.

EST accounting: W per k64 tile 16384 -> 12288 B; 4608*8192 BF16 weights
75497472 -> 56623104 B plus 442368 B cursor lists and 144 B base table.
Gross stream saving = 18431856/4.64e6 = 3.972 us, NOT a net prediction:
T128 is co-bound. The port model from notes/study/t128_plan.md:43-49 gives
43*(81920-73728)/(128*1520*eff) = 1.81..2.01 us at eff=1..0.9. Charging
one measured PDL edge (0.84 us) leaves an EST 0.97..1.17 us ceiling before
exposed decode/synchronization. Thus >=0.5 us is possible only if added
exposed work is <=0.47..0.67 us; a GPU measurement is mandatory.
Source-level added decode/load issue alone is 43*2*40/1520 = 2.263 us at
one issue/cycle/SMSP, plus 2*4*32/1520 = 0.168 us for up to four escapes;
most must overlap WGMMA for a win. These are EST issue budgets, not SASS
measurements; the port efficiency and issue throughput are model assumptions.
MEASURED cat = 25.8/24.8 us (v16s/f, kernel_baseline.md:99/109).
The cat gates tail/routing/MoE; no side-stream join or extra launch is added.
DSMEM payload (4718592 B), FP32 output (2359296 B), and split assignment are
unchanged from VD. Mainloop smem traffic 81920 -> 73728 B/k64 stage.

Protocol: cluster rank ks=blockIdx.x%3, nb=blockIdx.x//3. All 384 threads
publish/acquire cluster barrier initialization; producers 256..383 then
exit. Math 0..255 drain WGMMA and issue every DSMEM send. Each receiver
owns j in [floor(16*ks/3),floor(16*(ks+1)/3)), receiving exactly
2*(jhi-jlo)*256*16 bytes. All 256 math threads wait, combine two peers,
transpose, and store their owned tokens. There are no global counters,
zeroing loops, or resets in this arm. Shared barriers are initialized on
every launch. A failed bounded receive sets err and writes NaNs, never
traps; the validator cannot admit a failed receive. All existing RQ paths
and self-tests are retained verbatim except compile-time CAT=false branches.
"""

import argparse
import ast
import contextlib
import copy
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "box_out/s7/k3_d128_v1.py"
REPORT = Path(__file__).with_suffix(".json")
CAT_BIT = 1 << 27
CAT_FL = 42999939 | CAT_BIT
LIMIT = 1_000_000


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sub(s, old, new, label):
    require(s.count(old) == 1, f"{label}: expected one anchor, found {s.count(old)}")
    return s.replace(old, new, 1)


PYTHON_ARM = '''# S7-D128 bit 1: fused P12 cat/128, default OFF; validated bank built at prepare.
S7_D128 = 0
S7_D128_PREPARE = True
_S7D128_DEADLINE = 265.0

def _s7d128_deadline(t0):
 if time.time()-t0 >= _S7D128_DEADLINE:
  raise RuntimeError("S7-D128 prepare deadline")

def _s7d128_pack(w,t0):
 # Exact existing _dg1_weight accessor supplies v8/wcat. Separate BF16
 # component bases keep the low router component lossless and cursor-small.
 if w.shape!=(D_EXPERT+2*N_EXPERTS,HIDDEN) or w.dtype!=torch.bfloat16:
  raise ValueError("S7-D128 cat layout")
 lays,bases,escs=[],[],[]
 for a,b in ((0,D_EXPERT),(D_EXPERT,D_EXPERT+N_EXPERTS),(D_EXPERT+N_EXPERTS,D_EXPERT+2*N_EXPERTS)):
  _s7d128_deadline(t0)
  lay,tab,esc=_rq_pack_weight(w[a:b].contiguous())
  esc=esc.clone()
  esc[:,0]+=a//16
  lays.append(lay)
  bases.extend([tab]*((b-a)//128))
  escs.append(esc)
 lay=torch.cat(lays,dim=0).contiguous()
 esc6=torch.cat(escs,dim=0).contiguous()
 esc,mx=_rqesc2(esc6,w.shape[0]//128,w.shape[1]//64,3,w.device)
 if esc is None:
  raise ValueError("S7-D128 escape cursor overflow: %d"%mx)
 tab=torch.tensor(bases,dtype=torch.int32,device=w.device)
 wb=sum(t.numel()*t.element_size() for t in (lay,esc,tab))
 if wb>=w.numel()*w.element_size():
  raise ValueError("S7-D128 packed stream did not shrink")
 return {"lay":lay,"esc":esc,"tab":tab,"weight_bytes":wb,"max_escape":mx}

def _s7d128_call(r,x,out,info=0):
 return r["mod"].rqc_run(r["lay"],x,out,r["esc"],r["tab"],r["err"],info)

def _s7d128_validate(r,w,state,t0):
 # Padz convention: poison, fresh inputs, repeated calls, reference, and
 # explicit ok flag. Existing RQ, VD and routing self-tests remain intact.
 worst=0.0
 for t,scale in ((128,1.0),(125,30.0),(65,1.0),(1,1.0)):
  _s7d128_deadline(t0)
  x,_=_gd_test_input(t,w.shape[1],71283+t,w.device,outliers=scale!=1.0)
  ref=x.double()@w.double().t()
  outs=[]
  for rep in range(2):
   y=torch.full((t,w.shape[0]),float("nan"),dtype=torch.float32,device=w.device)
   r["err"].zero_()
   if _s7d128_call(r,x,y)!=0:
    raise ValueError("S7-D128 launch rejected")
   torch.cuda.synchronize()
   e=float((y.double()-ref).norm()/ref.norm().clamp(min=1e-30))
   row=float(((y.double()-ref).norm(dim=1)/ref.norm(dim=1).clamp(min=1e-30)).max())
   if int(r["err"][0]) or not bool(torch.isfinite(y).all()) or e>=1e-4 or row>=5e-4:
    raise ValueError("S7-D128 FP64/self-test: %.3g %.3g"%(e,row))
   worst=max(worst,e)
   outs.append(y)
  if not torch.equal(outs[0],outs[1]):
   raise ValueError("S7-D128 non-deterministic output")
  if t==128:
   _s7d128_deadline(t0)
   today=torch.mm(x,w.t(),out_dtype=torch.float32)
   for start in (0,96,160):
    d={"local_start":torch.tensor([start],dtype=torch.int32,device=w.device)}
    a=_v8_tail(outs[0],None,state,d)
    b=_v8_tail(today,None,state,d)
    logits64=x.double()@state["router"].double().t()
    logits=outs[0][:,D_EXPERT:D_EXPERT+N_EXPERTS]+outs[0][:,D_EXPERT+N_EXPERTS:]
    ids_ok,we,lids_ok=_tie1_compare(a[2],a[1],a[3],b[2],b[1],b[3],logits64,state["v8"]["bias"],
     logits,state["local_id_table"],start)
    xe=float((a[0].float()-b[0].float()).norm()/b[0].float().norm().clamp(min=1e-30))
    if not(ids_ok and lids_ok and we<1e-3 and xe<1e-2):
     raise ValueError("S7-D128 cat routing self-test")
 return worst

def _s7d128_prepare(state,t0):
 bank=state["s7_d128"]={1:{"ok":False}}
 if not S7_D128_PREPARE:
  return
 try:
  _s7d128_deadline(t0)
  w=_dg1_weight(state,"cat")
  if w is None:
   raise ValueError("S7-D128 missing cat")
  r=_s7d128_pack(w,t0)
  r.update(mod=_rq_build(),err=torch.zeros(1,dtype=torch.int32,device=w.device),device=w.device)
  clusters=_s7d128_call(r,w,w,4)
  if clusters<w.shape[0]//128:
   raise ValueError("S7-D128 cannot admit one wave: %d"%clusters)
  if _s7d128_call(r,w,w,2)!=0:
   raise ValueError("S7-D128 nonzero local memory")
  r["max_clusters"]=clusters
  r["rel64"]=_s7d128_validate(r,w,state,t0)
  r["ok"]=True
  bank[1]=r
  print("[s7d128] cat128 LIVE bytes=%d clusters=%d rel64=%.3g"%(r["weight_bytes"],clusters,r["rel64"]),flush=True)
 except Exception as ex:
  bank[1]["error"]=str(ex)
  print("[s7d128] cat128 OFF: %s"%str(ex)[:250],flush=True)

def _s7d128_cat(state,x):
 if S7_D128!=1 or x.dim()!=2 or x.shape!=(128,HIDDEN) or x.dtype!=torch.bfloat16:
  return None
 r=state.get("s7_d128",{}).get(1)
 if r is None or not r["ok"] or x.device!=r["device"] or not x.is_contiguous() or x.data_ptr()%16:
  return None
 out=torch.empty((128,D_EXPERT+2*N_EXPERTS),dtype=torch.float32,device=x.device)
 if _s7d128_call(r,x,out)!=0:
  raise RuntimeError("S7-D128 selected launch failed")
 return out

'''


CUDA_HELPERS = r'''
// S7-D128: VD's dedicated DSMEM push epilogue, with RQ RS accumulators.
__device__ __forceinline__ uint32_t rqc_mapa(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
__device__ __forceinline__ void rqc_push(uint32_t a, float x, float y, float z, float w, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n"
    :: "r"(a), "r"(__float_as_uint(x)), "r"(__float_as_uint(y)), "r"(__float_as_uint(z)), "r"(__float_as_uint(w)), "r"(b) : "memory");
}
__device__ __forceinline__ bool rqc_wait(uint32_t b, uint32_t* err) {
  const unsigned long long t0=clock64();
  for (;;) {
    uint32_t ok;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }\n"
      : "=r"(ok) : "r"(b) : "memory");
    if (ok) return true;
    if (clock64()-t0>2000000000ull) { atomicExch(err,2u); return false; }
  }
}
'''


CUDA_EPILOGUE = r'''  if constexpr (CAT) {
    // All 256 math threads retire the RS pipeline before reusing the ring.
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    const uint32_t rb=sbase+C::CAT_RB, rbar=sbase+C::CAT_RBAR;
    const int jlo=ks*16/3, jhi=(ks+1)*16/3;
    #pragma unroll
    for (int peer=0;peer<3;++peer) {
      if (peer==ks) continue;
      const int plo=peer*16/3, phi=(peer+1)*16/3, slot=ks<peer?ks:ks-1;
      const uint32_t dst=rqc_mapa(rb,peer), mb=rqc_mapa(rbar,peer);
      #pragma unroll
      for (int j=0;j<16;++j) {
        if (j<plo || j>=phi) continue;
        rqc_push(dst+((slot*6+j-plo)*256+tid)*16,D[4*j],D[4*j+1],D[4*j+2],D[4*j+3],mb);
      }
    }
    const bool good=rqc_wait(rbar,p.sink);
    const float4* recv=reinterpret_cast<const float4*>(smem+(sbase-sraw)+C::CAT_RB);
    float* tile=reinterpret_cast<float*>(smem+(sbase-sraw));
    const int row=64*g+16*w+(lane>>2), t4=lane&3;
    #pragma unroll
    for (int j=0;j<16;++j) {
      if (j<jlo || j>=jhi) continue;
      float a=D[4*j],b=D[4*j+1],c=D[4*j+2],d=D[4*j+3];
      #pragma unroll
      for (int s=0;s<2;++s) {
        if (good) {
          const float4 q=recv[(s*6+j-jlo)*256+tid];
          a+=q.x; b+=q.y; c+=q.z; d+=q.w;
        }
      }
      if (!good) a=b=c=d=__int_as_float(0x7fffffff);
      const int t=8*(j-jlo)+2*t4;
      tile[t*132+row]=a; tile[(t+1)*132+row]=b;
      tile[t*132+row+8]=c; tile[(t+1)*132+row+8]=d;
    }
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
    const int tlo=8*jlo, thi=min(8*jhi,p.m), nt=max(0,thi-tlo);
    #pragma unroll 1
    for (int i=tid;i<nt*32;i+=256) {
      const int t=i/32,c=i%32;
      const uint4 v=*reinterpret_cast<const uint4*>(tile+t*132+4*c);
      *reinterpret_cast<uint4*>(p.cat_out+(size_t)(tlo+t)*p.N+nb*128+4*c)=v;
    }
    return; // no counters or output-zero loop in the cluster arm
  }
'''


CUDA_HOST = r'''
// S7-D128's only new instance. Regular rq_run and its selectors are unchanged.
int64_t rqc_run(torch::Tensor W,torch::Tensor x,torch::Tensor out,torch::Tensor esc,
               torch::Tensor tab,torch::Tensor err,int64_t info) {
  constexpr int FL=177217667;
  using C=rq_C<128,5,1,0,0,0,FL,1>;
  auto k=rq_kernel<128,5,1,0,0,0,FL,1>;
  if (!W.is_cuda() || W.scalar_type()!=torch::kUInt8 || W.dim()!=4 || W.size(0)!=36 ||
      W.size(1)!=128 || W.size(2)!=8 || W.size(3)!=1536 || !W.is_contiguous()) return -40;
  static bool once=false;
  if (!once) {
    if (cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM)!=cudaSuccess) { cudaGetLastError(); return -41; }
    once=true;
  }
  cudaLaunchAttribute cat_attrs[1]={};
  cat_attrs[0].id=cudaLaunchAttributeClusterDimension;
  cat_attrs[0].val.clusterDim.x=3; cat_attrs[0].val.clusterDim.y=1; cat_attrs[0].val.clusterDim.z=1;
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(108,1,1); cfg.blockDim=dim3(384,1,1);
  cfg.dynamicSmemBytes=C::SMEM; cfg.stream=at::cuda::getCurrentCUDAStream(); cfg.attrs=cat_attrs; cfg.numAttrs=1;
  if (info) {
    if (info==4) {
      int n=0;
      if (cudaOccupancyMaxActiveClusters(&n,k,&cfg)!=cudaSuccess) { cudaGetLastError(); return -42; }
      return n;
    }
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a,k)!=cudaSuccess) { cudaGetLastError(); return -43; }
    return info==1?a.numRegs:info==2?(int)a.localSizeBytes:C::SMEM;
  }
  if (!x.is_cuda() || x.scalar_type()!=torch::kBFloat16 || x.dim()!=2 || x.size(0)<1 || x.size(0)>128 ||
      x.size(1)!=8192 || !x.is_contiguous() || (uintptr_t)x.data_ptr()%16) return -44;
  if (!out.is_cuda() || out.scalar_type()!=torch::kFloat32 || out.dim()!=2 || out.size(0)!=x.size(0) ||
      out.size(1)!=4608 || !out.is_contiguous() || (uintptr_t)out.data_ptr()%16) return -45;
  if (!esc.is_cuda() || esc.scalar_type()!=torch::kInt32 || esc.numel()!=3*36*8*32*4 || !esc.is_contiguous() ||
      !tab.is_cuda() || tab.scalar_type()!=torch::kInt32 || tab.numel()!=36 || !tab.is_contiguous() ||
      !err.is_cuda() || err.scalar_type()!=torch::kInt32 || err.numel()<1 || !err.is_contiguous()) return -46;
  if (W.device()!=x.device() || out.device()!=x.device() || esc.device()!=x.device() || tab.device()!=x.device() || err.device()!=x.device()) return -47;
  rq_P p={};
  p.W=(const uint8_t*)W.data_ptr(); p.esc=p.esc2=(const uint32_t*)esc.data_ptr();
  p.cat_tab=(const uint32_t*)tab.data_ptr(); p.cat_out=out.data_ptr<float>();
  p.sink=(uint32_t*)err.data_ptr(); p.m=(int)x.size(0); p.N=4608; p.K=8192; p.KC=128; p.NB=36;
  rq_Tmap tx,to={};
  int e=rq_map(&tx,x.data_ptr(),p.K,p.m,128);
  if (e) return e;
  const cudaError_t launch=cudaLaunchKernelEx(&cfg,k,p,tx,to);
  return launch==cudaSuccess?0:(int)launch;
}
'''


def build_text(src):
    require("\r" not in src and src.isascii(), "base is not ASCII/LF")
    lines = src.splitlines(keepends=True)
    require(lines[0].startswith("# okc:") and lines[1].startswith("# nonce "), "header")
    lines[1] = "# nonce d128-v1 (tinys7_c1 + fused P12/FP32 cat128; S7_D128=0; tools/s7_d128_patches.py)\n"
    s = "".join(lines)
    s = sub(s, " _tinys_prepare()\n", " _tinys_prepare()\n _s7d128_prepare(state,t0)\n", "prepare hook after all mandatory base work")
    s = sub(s, 'functions=["rq_run"],extra_cuda_cflags=', 'functions=["rq_run","rqc_run"],extra_cuda_cflags=', "RQ exports")
    decl = 'int64_t rqc_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);'
    a = next(l for l in s.splitlines() if l.startswith('_rq_cpp='))
    s = sub(s, a, a[:-1] + decl + '"', "RQ declaration")
    s = sub(s, 'def _rq_pooler_tail(state,x,data,tail):\n if tail!="cat":\n  return None\n',
            'def _rq_pooler_tail(state,x,data,tail):\n if tail!="cat":\n  return None\n y32=_s7d128_cat(state,x)\n if y32 is not None:\n  return _v8_tail(y32,None,state,data)\n', "fused cat route")
    s = sub(s, '_rq_cuda=r"""\n', PYTHON_ARM + '_rq_cuda=r"""\n', "arm Python")
    start = s.index('_rq_cuda=r"""\n') + len('_rq_cuda=r"""\n')
    end = s.index('\n"""\n\n_rq_variants=', start)
    cu = s[start:end]
    cu = sub(cu, '  uint32_t Tb;\n', '  uint32_t Tb;\n  const uint32_t* cat_tab;\n  float* cat_out;\n', "cat parameters")
    cu = sub(cu, '// Ring: ST k64 slots', CUDA_HELPERS + '\n// Ring: ST k64 slots', "DSMEM helpers")
    a = next(l for l in cu.splitlines() if l.startswith('  static constexpr int SMEM ='))
    b = a.replace('int SMEM =', 'int BASE_SMEM =')
    b += '\n  static constexpr int CAT_RBAR=ST*(XBS+WBS)+24*ST+16;\n'
    b += '  static constexpr int CAT_RB=(CAT_RBAR+8+1023)&~1023;\n'
    b += f'  static constexpr int SMEM=(FL & {CAT_BIT})?1024+CAT_RB+2*6*256*16:BASE_SMEM;'
    cu = sub(cu, a, b, "cluster receive buffer")
    cu = sub(cu, '  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;',
             f'  constexpr bool CAT=(FL & {CAT_BIT})!=0;\n  static_assert(!CAT || (NT==128 && ST==5 && GS==1 && PRB==0 && WM==0 && INS==0 && MS==1 && FL=={CAT_FL}), "S7-D128 exact cat instance");\n  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0 && !CAT;', "disable atomic zeroing only in CAT")
    a = '  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;'
    b = '  const int nb = CAT ? (int)blockIdx.x/3 : (HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x), hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = CAT ? (int)blockIdx.x%3 : (int)blockIdx.y, KS = CAT ? 3 : (int)gridDim.y;'
    cu = sub(cu, a, b, "cluster grid indices")
    a = '  __syncthreads();\n  const unsigned long long cta ='
    b = r'''  if constexpr (CAT) {
    if (tid==0) {
      const uint32_t rb=sbase+C::CAT_RBAR;
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(rb) : "memory");
      rq_expect(rb,2*((ks+1)*16/3-ks*16/3)*256*16);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
  }
  __syncthreads();
  if constexpr (CAT) asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  const unsigned long long cta ='''
    cu = sub(cu, a, b, "receive barrier initialization")
    a = '    return;\n  }\n  // ---------------- math warpgroups:'
    b = '    if constexpr (CAT) asm volatile("barrier.cluster.wait.acquire;\\n" ::: "memory");\n' + a
    cu = sub(cu, a, b, "producer cluster lifetime")
    cu = sub(cu, '  uint32_t Tb = p.Tb;', '  uint32_t Tb = CAT ? __ldg(p.cat_tab+nb) : p.Tb;', "component-specific exponent base")
    a = '  const unsigned long long gtl1 = INS ? rq_gt() : 0;'
    cu = sub(cu, a, CUDA_EPILOGUE + a, "FP32 DSMEM epilogue")
    cu += '\n' + CUDA_HOST
    return s[:start] + cu + s[end:]


def definition(src, name):
    tree = ast.parse(src)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(found) == 1, f"definition {name}")
    return found[0]


def literal(src, name):
    tree = ast.parse(src)
    found = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(found) == 1, f"literal {name}")
    return ast.literal_eval(found[0].value)


def load_functions(src, names, ns):
    nodes = [definition(src, n) for n in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(OUT), "exec"), ns)


def cuda_source(src):
    ns = {"_rq_cuda": literal(src, "_rq_cuda"), "_rq_ms": literal(src, "_rq_ms")}
    load_functions(src, ["_rqwg", "_rqss", "_rqsrc"], ns)
    return ns["_rqsrc"](literal(src, "_rq_variants"))


def changes(src, out):
    a,b=src.splitlines(),out.splitlines()
    return [dict(bit=1,base=[i+1,j],variant=[k+1,l],kind=tag)
            for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes() if tag!="equal"]


def static_gate(src, out):
    require(out == build_text(src), "generated file differs from anchored build")
    require(len(out.encode()) < LIMIT and "\r" not in out and out.isascii(), "bytes/LF/ASCII")
    ast.parse(out); compile(out,str(OUT),"exec")
    require(literal(out,"S7_D128")==0, "selector must default off")
    require(literal(out,"_rq_variants")==literal(src,"_rq_variants"), "base RQ instances changed")
    for name in ("VD_INSTS","VD_ROUTE","LT_PIN","_rq_picks","_rq_ks","RQ_PACK"):
        require(literal(out,name)==literal(src,name),f"unchanged route {name}")
    allowed={"prepare","_rq_build","_rq_pooler_tail"}
    old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}
    new={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(out).body if isinstance(n,ast.FunctionDef)}
    require([k for k in old if old[k]!=new.get(k) and k not in allowed]==[], "unexpected Python function change")
    cu=cuda_source(out)
    require("@@" not in cu and "__trap" not in CUDA_HELPERS+CUDA_EPILOGUE+CUDA_HOST,"CUDA markers/trap")
    require(str(CAT_FL) in CUDA_HOST,"wrong flag literal")
    return dict(status="PASS",bytes=len(out.encode()),headroom=LIMIT-len(out.encode()),
                sha256=hashlib.sha256(out.encode()).hexdigest(),changed_lines=changes(src,out),
                unchanged_python_functions=len(old)-len(allowed),selector_default=0)


# CPU and Linux gates below never import the full submission or create a GPU context.

class TensorShape:
    """Strict shape-only tensor. Numerical operations are explicitly mocked."""
    def __init__(self, shape, dtype="bf16", data=None, zero=False, device="cpu-dry"):
        self.shape=tuple(shape); self.dtype=dtype; self.data=data; self.zero=zero
        self.device=device; self.alignment=16; self.contig=True
    def contiguous(self): return self
    def is_contiguous(self): return self.contig
    def dim(self): return len(self.shape)
    def numel(self):
        import math
        return math.prod(self.shape)
    def element_size(self): return {"bf16":2,"f32":4,"f64":8,"i32":4,"u8":1,"bool":1}[self.dtype]
    def data_ptr(self): return self.alignment
    def clone(self): return TensorShape(self.shape,self.dtype,None if self.data is None else self.data.copy(),self.zero,self.device)
    def to(self,dtype): return TensorShape(self.shape,dtype,self.data,self.zero,self.device)
    def double(self): return self.to("f64")
    def float(self): return self.to("f32")
    def t(self):
        require(self.dim()==2,"mock transpose rank")
        return TensorShape(self.shape[::-1],self.dtype,zero=self.zero,device=self.device)
    def zero_(self):
        self.zero=True
        if self.data is not None: self.data.fill(0)
        return self
    def __getitem__(self,key):
        keys=key if isinstance(key,tuple) else (key,)
        require(len(keys)<=self.dim(),"mock indexing rank")
        shape=[]
        for n,k in zip(self.shape,keys+(slice(None),)*(self.dim()-len(keys))):
            if isinstance(k,int): require(-n<=k<n,"mock index bound")
            elif isinstance(k,slice): shape.append(len(range(*k.indices(n))))
            else: raise AssertionError(f"unsupported mock index {k!r}")
        data=None if self.data is None else self.data[key]
        return TensorShape(shape,self.dtype,data,self.zero,self.device)
    def __setitem__(self,key,value):
        require(self.data is not None,"mock writes require concrete tiny data")
        self.data[key]=value.data if isinstance(value,TensorShape) else value
    def __iadd__(self,value):
        require(self.data is not None,"mock iadd requires data")
        self.data+=value
        return self
    def __matmul__(self,other):
        require(self.dim()==other.dim()==2 and self.shape[1]==other.shape[0],"mock matmul dimensions")
        return TensorShape((self.shape[0],other.shape[1]),self.dtype,device=self.device)
    def __sub__(self,other):
        require(self.shape==other.shape,"mock difference shape")
        return TensorShape(self.shape,self.dtype,zero=True,device=self.device)
    def __add__(self,other):
        require(self.shape==other.shape,"mock sum shape")
        return TensorShape(self.shape,self.dtype,device=self.device)
    def norm(self,dim=None):
        shape=() if dim is None else self.shape[:dim]+self.shape[dim+1:]
        return TensorShape(shape,self.dtype,zero=self.zero,device=self.device)
    def clamp(self,min):
        return TensorShape(self.shape,self.dtype,zero=False,device=self.device)
    def __truediv__(self,other):
        require(self.shape==other.shape,"mock divide shape")
        return TensorShape(self.shape,self.dtype,zero=self.zero,device=self.device)
    def max(self): return TensorShape((),self.dtype,zero=self.zero,device=self.device)
    def __float__(self):
        require(self.shape==(),"mock scalar float")
        return 0.0 if self.zero else 1.0
    def __int__(self):
        require(self.shape==(),"mock scalar int")
        return int(self.data) if self.data is not None else (0 if self.zero else 1)


def dry_run(base, out):
    """Execute actual added prepare/pack/validator/select functions on real AST layouts.

    Only tensors, existing pack arithmetic, CUDA, occupancy, routing math and
    numeric comparisons are mocked; every added Python branch is executed.
    No dict/attribute auto-creation or missing-key fallback is permitted.
    """
    import numpy as np
    trace={"packs":[],"launches":[],"routing":0}
    mode={"clusters":42,"local":0,"overflow":False,"pack_fail":False,"huge":False,"launch_error":False,"build_fail":False}
    def cat(ts,dim=0):
        shape=list(ts[0].shape)
        require(all(t.dim()==len(shape) and t.dtype==ts[0].dtype for t in ts),"mock cat dtype/rank")
        for d in range(len(shape)):
            if d!=dim: require(all(t.shape[d]==shape[d] for t in ts),"mock cat axes")
        shape[dim]=sum(t.shape[dim] for t in ts)
        data=np.concatenate([t.data for t in ts],axis=dim) if all(t.data is not None for t in ts) else None
        return TensorShape(shape,ts[0].dtype,data,device=ts[0].device)
    def tensor(v,dtype,device):
        a=np.asarray(v,dtype=np.int32 if dtype=="i32" else np.float32)
        return TensorShape(a.shape,dtype,a,device=device)
    def zeros(n,dtype,device):
        shape=(n,) if isinstance(n,int) else n
        return TensorShape(shape,dtype,np.zeros(shape,dtype=np.int32),True,device)
    def mm(x,w,out_dtype):
        y=x@w
        return y.to(out_dtype)
    torch=SimpleNamespace(bfloat16="bf16",float32="f32",int32="i32",cat=cat,tensor=tensor,zeros=zeros,
        empty=lambda shape,dtype,device:TensorShape(shape,dtype,device=device),
        full=lambda shape,value,dtype,device:TensorShape(shape,dtype,device=device),
        mm=mm,isfinite=lambda t:SimpleNamespace(all=lambda:True),
        equal=lambda a,b:a.shape==b.shape and a.dtype==b.dtype,
        cuda=SimpleNamespace(synchronize=lambda:None))
    weights={k:TensorShape(shape,dtype) for k,shape,dtype in (
        ("qkv_gate",(2304,8192),"bf16"),("o",(8192,1024),"bf16"),
        ("pooler",(4096,8192),"bf16"),("unpooler",(8192,4096),"bf16"),
        ("router",(256,8192),"f32"),("router_bias",(256,),"f32"))}
    stnode=next(n.value for n in definition(base,"_r1_prepare").body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="state" for t in n.targets))
    keep=("dense","router","local_id_table")
    expr=ast.Dict(keys=[k for k in stnode.keys if isinstance(k,ast.Constant) and k.value in keep],
                  values=[v for k,v in zip(stnode.keys,stnode.values) if isinstance(k,ast.Constant) and k.value in keep])
    require(len(expr.keys)==3,"real base state layout changed")
    scope={"weights":weights,"router":weights["router"],"device":"cpu-dry","torch":torch,
           "_local_id_table":lambda device:TensorShape((65536,),"i32",device=device)}
    state=eval(compile(ast.fix_missing_locations(ast.Expression(expr)),str(BASE),"eval"),scope)
    v8node=next(n.value for n in definition(base,"_v8_tail_prepare").body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=="v8" for t in n.targets))
    scope.update(hi=TensorShape((256,8192)),lo=TensorShape((256,8192)))
    state["v8"]={k.value:eval(compile(ast.Expression(v),str(BASE),"eval"),scope)
                  for k,v in zip(v8node.keys,v8node.values) if k.value in ("wcat","bias")}
    original_state=state
    ns={"time":time,"torch":torch,"S7_D128":0,"S7_D128_PREPARE":True,"_S7D128_DEADLINE":265.0,
        "HIDDEN":literal(base,"HIDDEN"),"D_EXPERT":literal(base,"D_EXPERT"),"N_EXPERTS":literal(base,"N_EXPERTS")}
    names=[n.name for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name.startswith("_s7d128_")]
    load_functions(out,names+["_dg1_weight","_rq_pooler_tail"],ns)
    def pack(w):
        if mode["pack_fail"]: raise ValueError("mock existing pack failure")
        n,k=w.shape
        require(k==8192 and n in (4096,256),"component shape mismatch")
        i=len(trace["packs"])%3
        trace["packs"].append((n,k))
        return TensorShape((n//128,k//64,8,1536),"u8"), (58-i)*0x01010101, tensor([[0,0,0,0,0,1]],"i32",w.device)
    def esc2(esc,nb,kc,ks,dev):
        require((nb,kc,ks)==(36,128,3),"escape layout mismatch")
        require(esc.data[:,0].tolist()==[0,256,272],"component escape offsets wrong")
        if mode["overflow"]: return None,5
        n=3*36*8*32*4
        if mode["huge"]: n*=1000
        return TensorShape((n,),"i32",device=dev),1
    class Module:
        def rqc_run(self,w,x,y,e,t,err,info):
            require(w.shape==(36,128,8,1536) and e.shape==(3*36*8*32*4,) and t.shape==(36,),"new host ABI packing")
            require(t.data.tolist()==[58*0x01010101]*32+[57*0x01010101]*2+[56*0x01010101]*2,"component table contents")
            if info: return mode["clusters"] if info==4 else mode["local"]
            require(x.dtype=="bf16" and x.shape[1]==8192 and y.dtype=="f32" and y.shape==(x.shape[0],4608),"new host ABI output")
            trace["launches"].append(x.shape[0])
            return -44 if mode["launch_error"] else 0
    mod=Module()
    def build():
        if mode["build_fail"]: raise ValueError("mock build failure")
        return mod
    def tail(y,logits,st,d):
        require(logits is None and y.dtype=="f32" and y.shape[1]==4608,"fused cat tail precision")
        require(st["v8"]["bias"].shape==(256,) and d["local_start"].shape==(1,),"real tail state layout")
        trace["routing"]+=1
        n=y.shape[0]
        return (TensorShape((n,4096)),TensorShape((n,8),"f32"),TensorShape((n,8),"i32"),TensorShape((n,8),"i32"))
    def tie(*args):
        require(len(args)==11 and args[6].shape==(128,256) and args[9] is original_state["local_id_table"],"routing validator ABI")
        return True,0.0,True
    ns.update(_rq_pack_weight=pack,_rqesc2=esc2,_rq_build=build,_v8_tail=tail,_tie1_compare=tie,
              _rq_gemm=lambda *args:None,
              _gd_test_input=lambda m,k,seed,device,outliers=False:(TensorShape((m,k),device=device),TensorShape((m,),"bool",device=device)))
    logs=io.StringIO()
    with contextlib.redirect_stdout(logs): ns["_s7d128_prepare"](state,time.time())
    require(state["s7_d128"][1]["ok"],logs.getvalue())
    require(trace["launches"]==[128,128,125,125,65,65,1,1] and trace["routing"]==6,"validator case coverage")
    require(trace["packs"]==[(4096,8192),(256,8192),(256,8192)],"component pack coverage")
    successful_r=state["s7_d128"][1]
    selection_count=0
    for selector in (0,1,2,3):
        ns["S7_D128"]=selector
        for t in (1,8,16,32,64,65,127,128,129,256):
            x=TensorShape((t,8192))
            y=ns["_s7d128_cat"](state,x)
            require((y is not None)==(selector==1 and t==128),f"selector {selector} T={t}")
            selection_count+=1
    ns["S7_D128"]=1
    for x in (TensorShape((128,8192),"f32"),TensorShape((128,4096)),TensorShape((128,8192),device="different"),TensorShape((8192,))):
        require(ns["_s7d128_cat"](state,x) is None,"invalid x route")
    x=TensorShape((128,8192)); x.contig=False
    require(ns["_s7d128_cat"](state,x) is None,"noncontiguous x")
    x=TensorShape((128,8192)); x.alignment=3
    require(ns["_s7d128_cat"](state,x) is None,"misaligned x")
    require(ns["_s7d128_cat"]({},TensorShape((128,8192))) is None,"missing bank")
    d={"local_start":tensor([0],"i32","cpu-dry")}
    require(ns["_rq_pooler_tail"](state,TensorShape((128,8192)),d,"cat") is not None,"live call-chain hook")
    require(ns["_rq_pooler_tail"](state,TensorShape((128,8192)),d,"sep") is None,"separate tail untouched")
    ns["S7_D128"]=0
    require(ns["_rq_pooler_tail"](state,TensorShape((128,8192)),d,"cat") is None,"off call-chain")
    failures=[]
    for label,key,value in (("occupancy","clusters",35),("local memory","local",8),("escape overflow","overflow",True),
                            ("pack exception","pack_fail",True),("stream did not shrink","huge",True),
                            ("launch error","launch_error",True),("build error","build_fail",True)):
        old=mode[key]; mode[key]=value
        with contextlib.redirect_stdout(io.StringIO()): ns["_s7d128_prepare"](state,time.time())
        require(not state["s7_d128"][1]["ok"] and "error" in state["s7_d128"][1],label+" not rejected")
        failures.append(label); mode[key]=old
    for label,t0,mutate in (("deadline",time.time()-270,None),("missing cat",time.time(),"missing"),
                             ("bad layout",time.time(),"layout")):
        oldv8=state["v8"]
        if mutate=="missing": state["v8"]=None
        if mutate=="layout": state["v8"]={"wcat":TensorShape((4352,8192))}
        with contextlib.redirect_stdout(io.StringIO()): ns["_s7d128_prepare"](state,t0)
        require(not state["s7_d128"][1]["ok"] and "error" in state["s7_d128"][1],label+" not rejected")
        state["v8"]=oldv8; failures.append(label)
    ns["S7_D128_PREPARE"]=False
    ns["_s7d128_prepare"](state,time.time())
    require(state["s7_d128"]=={1:{"ok":False}},"prepare OFF bank")
    state["s7_d128"][1]=successful_r
    return dict(status="PASS",selection_cases=selection_count,validation_cases=4,repeated_launches=8,
                router_cases=3,negative_prepare_cases=failures,weight_bytes=successful_r["weight_bytes"],
                state_layout_sources={"_r1_prepare":definition(base,"_r1_prepare").lineno,
                                      "_v8_tail_prepare":definition(base,"_v8_tail_prepare").lineno},
                numerical_status="NOT RUN: strict tensor/CUDA/numeric mocks; real added Python functions executed")


def protocol_gate():
    """Exhaustively enumerate all threads, owned values, receive slots and output rows."""
    from collections import Counter
    ownership=Counter(); received={}; sent=0
    for dest in range(3):
        lo,hi=dest*16//3,(dest+1)*16//3
        slots=set()
        for sender in range(3):
            if sender==dest: continue
            slot=sender if sender<dest else sender-1
            for tid in range(256):
                for j in range(lo,hi):
                    ix=(slot*6+j-lo)*256+tid
                    require(ix not in slots,"DSMEM collision")
                    slots.add(ix); sent+=16
        expected=2*(hi-lo)*256*16
        require(len(slots)*16==expected,"receive expected_tx mismatch")
        require(max(slots)*16+16<=49152,"receive-buffer overflow")
        received[dest]=expected
        for tid in range(256):
            warp,lane=divmod(tid,32); g,w=divmod(warp,4)
            row=64*g+16*w+(lane>>2)
            for j in range(lo,hi):
                for t,r in ((8*j+2*(lane&3),row),(8*j+2*(lane&3)+1,row),
                            (8*j+2*(lane&3),row+8),(8*j+2*(lane&3)+1,row+8)):
                    ownership[(t,r)]+=1
        for t in (1,2,8,32,40,64,65,80,81,125,127,128):
            seen=set(); n=max(0,min(8*hi,t)-8*lo)
            for tid in range(256):
                for i in range(tid,n*32,256):
                    tt,c=divmod(i,32)
                    for v in range(4):
                        seen.add((8*lo+tt,4*c+v))
            expected_set={(tt,r) for tt in range(8*lo,min(8*hi,t)) for r in range(128)}
            require(seen==expected_set,"partial T store coverage")
        # Ring is dead before the transpose; separate receive buffer never aliases it.
        require((hi-lo)*8*132*4<=5*128*128,"transpose exceeds drained x ring")
    require(len(ownership)==128*128 and set(ownership.values())=={1},"accumulator mapping coverage")
    require(sent==2*128*128*4,"DSMEM bytes differ from VD")
    # Split-K source intervals and zero/cleanup coverage are independent of scheduling.
    intervals=[(s*128//3,(s+1)*128//3) for s in range(3)]
    require([i for a,b in intervals for i in range(a,b)]==list(range(128)),"K split coverage")
    raw=5*(16384+12288)+24*5+16
    rb=(raw+8+1023)&~1023
    smem=1024+rb+49152
    require(smem==194560 and rb>=raw+8 and smem<232448,"shared layout")
    return dict(status="PASS",CTAs=108,clusters=36,threads=384,math_threads=[0,255],producer_threads=[256,383],
                receive_bytes_per_rank=received,DSMEM_bytes=sent*36,output_values=128*4608,
                k_intervals=intervals,shared_bytes=smem,global_counters=0,reset_loops=0,
                coverage="all cluster participants initialize/acquire; all math threads push, wait, combine, transpose and store")


def build_dry_run(out):
    """Run the real _rq_build function with only its compiler call intercepted."""
    import builtins
    calls=[]; marker=object()
    def load_inline(**kw):
        calls.append(kw)
        require(kw["functions"]==["rq_run","rqc_run"],"wrong extension exports")
        for name in kw["functions"]:
            decl=re.search(r"int64_t\s+"+name+r"\(([^)]*)\)",kw["cpp_sources"])
            impl=re.search(r"int64_t\s+"+name+r"\(([^)]*)\)",kw["cuda_sources"])
            require(decl is not None and impl is not None,"missing host declaration/definition "+name)
            require(len(decl[1].split(","))==len(impl[1].split(",")),"extension ABI arity "+name)
        require(kw["cuda_sources"]==cuda_source(out),"real build source differs from compile gate")
        return marker
    original_import=builtins.__import__
    def importer(name,globals=None,locals=None,fromlist=(),level=0):
        if name=="torch.utils.cpp_extension": return SimpleNamespace(load_inline=load_inline)
        return original_import(name,globals,locals,fromlist,level)
    ns={name:literal(out,name) for name in ("_rq_cuda","_rq_cpp","_rq_flags","_rq_variants","_rq_ms","_rq_build_cache")}
    ns["__builtins__"]=dict(vars(builtins),__import__=importer)
    load_functions(out,["_rqwg","_rqss","_rqsrc","_rq_build"],ns)
    require(ns["_rq_build"]() is marker and ns["_rq_build"]() is marker and len(calls)==1,"build cache path")
    ns["_rq_build_cache"]["error"]="CPU test cached failure"
    try:
        ns["_rq_build"]()
    except RuntimeError as ex:
        require(str(ex)=="CPU test cached failure","cached build error changed")
    else: raise AssertionError("cached error was ignored")
    return dict(status="PASS",exports=calls[0]["functions"],source_sha256=hashlib.sha256(calls[0]["cuda_sources"].encode()).hexdigest(),
                compiler_invocations=0,compiler_call="explicit CPU double",warm_cache="PASS",cached_failure="PASS")


def numpy_pack_gate():
    """Independent bit-level P12/table/escape and fragment-layout round trip."""
    import numpy as np
    rng=np.random.default_rng(71283)
    n,k=384,192
    # Distinct component exponent ranges plus exact rare outliers/signs/low bytes.
    weights=rng.integers(0,65536,(n,k),dtype=np.uint16)
    for g,base in enumerate((56,47,36)):
        weights[g*128:(g+1)*128] = (weights[g*128:(g+1)*128]&0x80ff) | (rng.integers(base,base+8,(128,k),dtype=np.uint16)<<8)
    weights[0,0]=0; weights[129,137]=0xffff; weights[300,7]=0x7f80
    fr=weights.reshape(n//16,2,8,k//64,4,2,4,2).transpose(0,3,4,2,6,5,1,7).reshape(n//16,k//64,4,32,8)
    decoded=np.empty_like(fr)
    offsets=(0,8,16,24,4,12,20,28); signs=(7,15,23,31,11,19,27,3)
    escape_count=0
    for g,base in enumerate((56,47,36)):
        a=fr[g*8:(g+1)*8].astype(np.uint32)
        h=(a>>8)&127; sg=a>>15; d=h.astype(np.int32)-base
        valid=(d>=0)&(d<8); d=np.where(valid,d,0).astype(np.uint32)
        c=np.zeros(a.shape[:-1],np.uint32)
        for v in range(8): c|=(d[...,v]<<offsets[v])|(sg[...,v]<<signs[v])
        reconstructed=np.empty_like(a)
        for v in range(8):
            off=(c>>offsets[v])&7; sign=(c>>signs[v])&1
            reconstructed[...,v]=((base+off|(sign<<7))<<8)|(a[...,v]&255)
        esc=np.argwhere(~valid)
        for ix in esc:
            idx=tuple(ix); reconstructed[idx]^=(h[idx]^base)<<8
        escape_count+=len(esc)
        require(np.array_equal(a,reconstructed),"P12 word/escape round trip")
        decoded[g*8:(g+1)*8]=reconstructed
    back=decoded.reshape(n//16,k//64,4,8,4,2,2,2).transpose(0,6,3,1,2,5,4,7).reshape(n,k)
    require(np.array_equal(back,weights),"fragment layout round trip")
    return dict(status="PASS",weights=n*k,components=3,escaped_values=escape_count,
                note="independent NumPy bit model; Linux gate also executes real torch packers")


def torch_pack_gate(base,out):
    """Linux CPU-only: execute the actual base pack/roundtrip and new component join."""
    import torch
    require(not torch.cuda.is_initialized(),"CPU gate unexpectedly initialized CUDA")
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    ns=dict(torch=torch,time=time,HIDDEN=8192,D_EXPERT=4096,N_EXPERTS=256,_S7D128_DEADLINE=265.0,
            _rq_os=literal(base,"_rq_os"),_rq_ss=literal(base,"_rq_ss"))
    load_functions(base,["_rq_c12","_rq_unc12","_rq_pack_nat","_rqpack","_rqunpack","_rq_pack_weight","_rqesc2"],ns)
    load_functions(out,["_s7d128_pack","_s7d128_deadline"],ns)
    gen=torch.Generator(device="cpu").manual_seed(71283)
    pool=(torch.randn((4096,8192),generator=gen)/8192**0.5).to(torch.bfloat16)
    router=torch.randn((256,8192),generator=gen)/8192**0.5
    hi=router.to(torch.bfloat16); lo=(router-hi.float()).to(torch.bfloat16)
    w=torch.cat((pool,hi,lo),dim=0).contiguous()
    start=time.time(); r=ns["_s7d128_pack"](w,start)
    require(r["lay"].shape==(36,128,8,1536) and r["esc"].shape==(3*36*8*32*4,),"real torch packed layout")
    require(r["tab"].shape==(36,) and len(set(r["tab"][:32].tolist()))==1,"real table layout")
    require(r["weight_bytes"]==57065616,"real torch bytes")
    require(not torch.cuda.is_initialized(),"CPU gate initialized CUDA")
    return dict(status="PASS",seconds=time.time()-start,max_escape=r["max_escape"],
                bases=[int(r["tab"][i]) for i in (0,32,34)],weight_bytes=r["weight_bytes"],
                device=str(r["lay"].device),roundtrip="real _rq_pack_weight checks all 37,748,736 BF16 weights")


def route_report(base,out):
    def ln(src,text):
        rows=[i+1 for i,l in enumerate(src.splitlines()) if text in l]
        require(rows,"missing route anchor "+text)
        return rows[0]
    return dict(
        v16=dict(shapes=["v16s","v16f"],
            qkv=dict(instance="rq_kernel<128,5,1,0,0,0,42999939,1>",KS=7,source=f"{BASE.relative_to(ROOT).as_posix()}:{ln(base,'RQ_SEL=')}",
                     measured="30.8 us aggregate qkv+unpooler; no individual split inferred"),
            unpooler=dict(instance="rq_kernel<128,5,1,0,0,0,42999939,1>",KS=2,source=f"{BASE.relative_to(ROOT).as_posix()}:{ln(base,'_rq_ks=')}"),
            cat=dict(instance="vd::vd_kernel<2,128,5,3,1,1>",source=f"{BASE.relative_to(ROOT).as_posix()}:{ln(base,'case 0: return vd_launch_t<')}",
                     measured_us=[25.8,24.8],measured_source="notes/study/evo/s7/kernel_baseline.md:99,109"),
            o=dict(instance="nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT",pin=literal(base,"LT_PIN")[("o",128)],layout=literal(base,"ULP_LAYOUT")[("o",128)],
                   source=f"{BASE.relative_to(ROOT).as_posix()}:{ln(base,'LT_PIN=')}",measured_us=[8.7,7.6]),
            new_cat=dict(instance=f"rq_kernel<128,5,1,0,0,0,{CAT_FL},1>",KS=3,
                         source=f"{OUT.relative_to(ROOT).as_posix()}:{ln(out,'auto k=rq_kernel<128,5,1,0,0,0,FL,1>')}",
                         route=f"{OUT.relative_to(ROOT).as_posix()}:{ln(out,'def _s7d128_cat')}",selector="S7_D128=1",measured_us=None)),
        v32=dict(shapes=["v32s","v32f"],changed=False,
            cat=dict(pin=literal(base,"LT_PIN")[("cat",256)],instance="nvjet_sm90_tss_72x128_64x8_1x2_h_bz_TNN",measured_us=[32.6,32.4]),
            unpooler=dict(pin=literal(base,"LT_PIN")[("unpooler",256)],layout=literal(base,"ULP_LAYOUT")[("unpooler",256)],
                          instance="nvjet_sm90_tst_128x128_64x6_1x2_h_bz_coopA_NNT",measured_us=[24.6,24.7]),
            qkv=dict(pin=literal(base,"LT_PIN")[("qkv",256)],instance="nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN",measured_us=[20.3,20.4]),
            o=dict(route="_v12_gemm -> _gemm -> _bf16_gemm_dispatch_impl (no LT_PIN o256)",
                   source=f"{BASE.relative_to(ROOT).as_posix()}:{definition(base,'_v12_gemm').lineno}",
                   instance="nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT",measured_us=[9.3,9.4]),
            pins_source=f"{BASE.relative_to(ROOT).as_posix()}:{ln(base,'LT_PIN=')}",
            measured_source="notes/study/evo/s7/kernel_baseline.md:116-131"))


def accounting():
    gross=(75497472-57065616)/4.64e6
    issue=43*2*40/1520
    escape=2*4*32/1520
    port=[43*(81920-73728)/(128*1520*eff) for eff in (1.0,0.9)]
    return dict(label="EST, conditional source-level arithmetic, NOT a measurement",
        baseline_us=dict(v16s=25.8,v16f=24.8,label="MEASURED",source="kernel_baseline.md:99/109"),
        weight_bytes_per_stage=dict(before=16384,after=12288,grows=False),
        x_bytes_per_stage=dict(before=16384,after=16384),
        smem_bytes_per_stage=dict(before=81920,after=73728),
        weight_bytes_total=dict(before=75497472,after=57065616,after_includes_escape_and_table=True),
        payload_thread_instructions_per_stage=dict(before="0 LDS weight decode + 0 unpack ALU + 4 SS WGMMA",
            after="3 LDS.128 + 36 unpack ALU + cursor control/rare patches + 4 RS WGMMA",
            modeled_added_issue_charge=40,exact_dynamic_SASS="pending Linux gate; escape/wait-dependent",
            common="one full-barrier wait + WGMMA fence/commit/wait per stage; address/control instructions compiler-dependent",
            release_before="2 empty-barrier arrivals per CTA/stage (WG leaders)",
            release_after="8 W-empty + 8 x-empty arrivals per CTA/stage (warp leaders), plus 2 syncwarp sites per math thread"),
        epilogue=dict(DSMEM_bytes_before_after=4718592,FP32_output_bytes_before_after=2359296,
            per_math_thread_send_vectors=[11,11,10],per_math_thread_receive_vectors=[10,10,12],
            per_math_thread_FP32_adds=[40,40,48],per_math_thread_staging_stores=[20,20,24],
            global_vector_stores_per_thread=[5,5,6],new_global_scratch_bytes=0,
            note="all figures per rank 0/1/2; same source-level epilogue payload as VD"),
        estimate_us=dict(gross_stream=gross,smem_port_ceiling=port,PDL_charge=0.84,
            ceiling_after_PDL_before_decode=[p-0.84 for p in port],net="UNMEASURED",
            all_extra_decode_issue=issue,escape_allowance=escape,
            maximum_exposed_extra_cost_for_0_5us=[p-0.84-0.5 for p in port],
            arithmetic="stream=(75497472-57065616)/4.64e6; port=43*(81920-73728)/(128*1520*eff), eff=1 or 0.9; ceiling=port-0.84",
            assumptions="128 B/SM/clock modeled port; 90-100% port efficiency; one issue/cycle/SMSP issue budget; added issue must mostly overlap WGMMA",
            source="notes/study/t128_plan.md:43-49; measured clock/PDL constants in task"),
        kill_reasons=["decode dependencies or PDL exposure can consume the conditional residual",
                      "register/cluster admission loss or compiler spills"],
        closest_DNR=["R-dnx-1 (avoided decoded-weight STS ring)","RQP128 (kept fused hi/lo router; no side-stream rejoin)"],
        critical_path="cat's FP32 result feeds tail/routing/MoE; reduced weight stream is inside that dependency; existing PDL overlap charged")


def ptxas_resources(stderr):
    rows={}; current=None
    for line in stderr.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m: current=m[1]; rows[current]={}
        m=re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",line)
        if m and current: rows[current].update(zip(("stack","spill_stores","spill_loads"),map(int,m.groups())))
        m=re.search(r"Used (\d+) registers",line)
        if m and current: rows[current]["registers"]=int(m[1])
    return rows


def sass_rows(text):
    from collections import Counter
    rows={}
    for m in re.finditer(r"Function\s*:\s*(\S+)(.*?)(?=Function\s*:|\Z)",text,re.S):
        name,body=m.groups()
        if "rq_kernel" not in name: continue
        ins=[s.strip() for s in re.findall(r"/\*[0-9a-fA-F]+\*/\s+([^;\n]+);",body)]
        require(ins,"empty SASS")
        ops=[re.sub(r"^@!?P\d+\s+","",s).split()[0] for s in ins]
        rows[name]=dict(static_sites=len(ins),opcodes=dict(Counter(ops)),instructions=ins,
                        local_memory_sites=[s for s in ins if re.search(r"\b(?:LDL|STL)\b",s)])
    require(rows,"no RQ SASS")
    return rows


def compile_cuda(src,tag,work):
    # Same full-production-source compile convention as the proven s7_rt gate.
    import torch
    import sysconfig
    from torch.utils.cpp_extension import include_paths
    cu,obj=work/(tag+".cu"),work/(tag+".o")
    cu.write_bytes(src.encode())
    cmd=["nvcc","-c","-std=c++20","-arch=sm_90a","-O3","--expt-relaxed-constexpr","-DNDEBUG","-Xptxas=-v",
         str(cu),"-o",str(obj),"-DTORCH_EXTENSION_NAME=s7_d128_compile","-DTORCH_API_INCLUDE_EXTENSION_H",
         "-D_GLIBCXX_USE_CXX11_ABI=%d"%int(torch._C._GLIBCXX_USE_CXX11_ABI),"--compiler-options","-fPIC"]
    cmd += ["-I"+p for p in include_paths(device_type="cuda")+[sysconfig.get_path("include")]]
    print(f"[s7d128] compiling {tag} (no GPU)",flush=True)
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    (work/(tag+".ptxas.log")).write_bytes((proc.stdout+proc.stderr).encode())
    require(proc.returncode==0,f"{tag} nvcc failed: {proc.stderr[-6000:]}")
    sass=subprocess.check_output(["cuobjdump","-sass",str(obj)],text=True,timeout=60)
    res=subprocess.check_output(["cuobjdump","-res-usage",str(obj)],text=True,timeout=60)
    (work/(tag+".sass")).write_bytes(sass.encode())
    (work/(tag+".resources.txt")).write_bytes(res.encode())
    return dict(command=cmd,resources=ptxas_resources(proc.stderr),resource_dump=res,
                warnings=[l for l in proc.stderr.splitlines() if "warning" in l.lower()]),sass_rows(sass)


def linux_checks(base,out,report):
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    require(os.name!="nt" and shutil.which("nvcc") and shutil.which("cuobjdump"),"Linux CUDA toolchain required")
    report["torch_cpu"]=torch_pack_gate(base,out)
    work=Path(tempfile.mkdtemp(prefix="s7_d128_gate_"))
    report["compile_work_directory"]=str(work)
    report["nvcc_version"]=subprocess.check_output(["nvcc","--version"],text=True)
    bc,bs=compile_cuda(cuda_source(base),"rq_base",work)
    report["base_compile"]=bc
    nc,ns=compile_cuda(cuda_source(out),"rq_d128",work)
    report["variant_compile"]=nc
    target=(128,5,1,0,0,0,CAT_FL,1)
    names=[n for n in ns if tuple(map(int,re.findall(r"Li(\d+)E",n)))==target]
    require(len(names)==1,"missing/ambiguous new instantiation")
    name=names[0]; r=nc["resources"].get(name,{})
    report["new_instance"]=dict(name=name,resources=r,SASS=ns[name])
    require({"registers","stack","spill_stores","spill_loads"}<=r.keys(),"missing ptxas resource data")
    require(r["spill_stores"]==r["spill_loads"]==r["stack"]==0,"new instance spills or uses stack")
    require(not ns[name]["local_memory_sites"],"new instance SASS contains local-memory access")
    # Parameter struct grows only at its tail; descriptor parameter offsets can
    # change, so compare opcode sequences/resources and retain full SASS for review.
    controls={}
    for old in bs:
        require(old in ns,"base specialization lost")
        same_ops=[re.sub(r"^@!?P\d+\s+","",s).split()[0] for s in bs[old]["instructions"]]==[
                  re.sub(r"^@!?P\d+\s+","",s).split()[0] for s in ns[old]["instructions"]]
        controls[old]=dict(same_opcode_sequence=same_ops,base_resources=bc["resources"][old],new_resources=nc["resources"][old])
        require(same_ops and bc["resources"][old]==nc["resources"][old],"default-off RQ codegen/resource drift")
    report["default_off_SASS_controls"]=controls
    import torch
    require(not torch.cuda.is_initialized(),"compile/CPU gate initialized CUDA")
    report["compile_status"]="PASS_ZERO_SPILLS"


def cpu_checks(base,out):
    return dict(status="CPU_PASS_CUDA_NOT_RUN",GPU_executed=False,
        base_sha256=hashlib.sha256(base.encode()).hexdigest(),variant=static_gate(base,out),
        dry_run=dry_run(base,out),build_dry_run=build_dry_run(out),protocol=protocol_gate(),numpy_pack=numpy_pack_gate(),
        accounting=accounting(),routes=route_report(base,out),
        compile_status="NOT RUN: WSL/NVCC unavailable in implementer sandbox",
        GPU_self_test="NOT RUN",prepare_duration="NOT MEASURED; deadline checks at 265 s; H200 validation required")


def main():
    ap=argparse.ArgumentParser(description="S7-D128 anchored generator and offline Linux CPU/compile/SASS gate")
    mode=ap.add_mutually_exclusive_group(required=True)
    for flag in ("generate","check","report","linux-gate"):
        mode.add_argument("--"+flag,action="store_true")
    args=ap.parse_args()
    before=BASE.read_bytes()
    require(len(before)==965632,"unexpected frozen base size")
    base=before.decode("ascii")
    out=build_text(base)
    if not args.generate:
        require(OUT.read_bytes()==out.encode("ascii"),"run --generate; checked variant does not match")
    report=cpu_checks(base,out)
    code=0
    if args.generate:
        OUT.parent.mkdir(parents=True,exist_ok=True)
        OUT.write_bytes(out.encode("ascii"))
    if args.linux_gate:
        try:
            linux_checks(base,out,report)
            report["status"]="COMPILE_PASS_GPU_NOT_RUN"
        except Exception as ex:
            import traceback
            report.update(status="GATE_FAIL",error=str(ex),traceback=traceback.format_exc())
            code=1
    require(BASE.read_bytes()==before,"base changed during gates")
    if not args.report:
        REPORT.write_bytes((json.dumps(report,indent=2)+"\n").encode())
    if args.report:
        print(json.dumps(report,indent=2))
    else:
        print(json.dumps(dict(status=report["status"],variant=str(OUT.relative_to(ROOT)),bytes=report["variant"]["bytes"],
            CPU="PASS",prepare_dry_run="PASS",protocol="PASS",compile=report["compile_status"],GPU="NOT RUN",
            report=str(REPORT.relative_to(ROOT)),error=report.get("error")),indent=2))
    return code


if __name__=="__main__":
    raise SystemExit(main())

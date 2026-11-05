"""S7-D32: two default-off SK2 occupancy experiments; no GPU work in this tool.

  python -B tools/s7_d32_patches.py              # generate both, selectors OFF
  python -B tools/s7_d32_patches.py --enable     # generate both, each own arm ON
  python -B tools/s7_d32_patches.py --check      # CPU gates, no output writes
  bash tools/local/s7_d32_gate.sh               # Linux compile/link/SASS, no GPU

Only the named variant files are written by generation. --json is reserved for
the requested Linux gate report. Temporary compiler artifacts are retained.

Pre-mortem: (1) extra L2 activation traffic can consume the occupancy gain;
(2) smaller per-CTA rings / the extra qkv CTAs can expose a tail. Neither is a
measured win. Runtime admission requires zero local memory and >=2 CTAs/SM.
R-sk2-1 is the nearest precedent; this changes the two *shipped* SK2 geometries,
not its killed cat/o routes, R-rat's GD routes, or GD-A/register rings.

v1 / bit 1: unpooler32, NT 4->2, R 64->32, grid 128->256, NSTAGE 3 kept.
v2 / bit 2: qkv32, NT 3->2, R 24->16, grid 96->144, NSTAGE 4->2.
The latter keeps UPW=2: it never activates SK2's early-release UPW=1 branch.
No weight-format or activation-precision change. No kernel-body/protocol edit.

The source-instruction accounting below counts explicit inline PTX operations,
not SASS or all C++ address/control instructions. The Linux gate supplies SASS.
The CPU dry run executes the real base state comprehension, real stream/table
builders and new prepare/selection paths; CUDA and numerical tensor operations
are mocks, explicitly not a GPU numerical check. This follows padz's independent
module/bank + prepare validation + call-time selector pattern.
"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUTS = {v: ROOT / f"box_out/s7/k3_d32_v{v}.py" for v in (1, 2)}
SHELL = ROOT / "tools/local/s7_d32_gate.sh"
LIMIT = 1_000_000


def sub(s, old, new, count=1):
    assert s.count(old) == count, ("anchor", old[:120], s.count(old), count)
    return s.replace(old, new)


def value(source, name):
    nodes = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, (name, len(nodes))
    return ast.literal_eval(nodes[0])


def expression(source, name):
    nodes = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, (name, len(nodes))
    return ast.dump(nodes[0], include_attributes=False)


# Record: bit -> (cfg, weight key, N, R, NT, NWR, WSK, UPW, NSTAGE).
CONFIGS = {1: (98, "unpooler", 8192, 32, 2, 2, 2, 2, 3),
           2: (99, "qkv_gate", 2304, 16, 2, 1, 4, 2, 2)}
ORIGINAL = {1: (11, "unpooler", 8192, 64, 4, 2, 2, 2, 3),
            2: (6, "qkv_gate", 2304, 24, 3, 1, 4, 2, 4)}

LAB = r'''
# S7-D32 BEGIN: independent SK2 module/bank; default-off, no live-module edits.
D32_BITS=0
D32_VARIANT=@VARIANT@
_D32_CFG=@CONFIG@
_D32={"mod":None,"err":None,"secs":0.0,"bank":{},"ok":False,"info":None}
def _d32_validate():
 if D32_BITS not in (0,D32_VARIANT):
  raise ValueError("D32_BITS must be 0 or this file's D32_VARIANT")
def _d32_src():
 s=_SK2_CUDA
 old="__launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)"
 assert s.count(old)==1
 s=s.replace(old,old[:-2]+"2)")
 a=s.index("if (cfg == 6 && tokr == 32) {")
 b=s.index('TORCH_CHECK(false, "sk2: no config");',a)
 ci,layer,n,R,nt,nwr,wsk,upw,st=_D32_CFG
 args=f"2, 32, {nt}, {nwr}, {wsk}, {upw}, {st}, true, false, 1, 1, __nv_bfloat16"
 s=s[:a]+f"if (cfg == {ci} && tokr == 32) {{ launch_sk2<{args}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}\n"+s[b:]
 # Query the actual compiled specialization and actual pmax, never inferred occupancy.
 s+='\nstd::vector<int64_t> d32_info(int64_t pmax) {\n'
 s+=f' auto kern=sk2_kernel<{args}, false, false>;\n'
 nw=nwr*wsk
 ne=min(st,4)
 wb=nw*upw*nt*768
 xb=wsk*upw*32*128
 ss=(wb+xb+1023)&~1023
 cs=(32*(R+1)+3)&~3
 sm=1024+st*ss+16*st+32+ne*cs*4+128
 s+=f' const size_t sm={sm}+(size_t)pmax*8;\n'
 s+=' cudaFuncAttributes fa; C10_CUDA_CHECK(cudaFuncGetAttributes(&fa,kern));\n'
 s+=' C10_CUDA_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm));\n'
 s+=f' int occ=0; C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,kern,{(nw+1+ne)*32},sm));\n'
 s+=' return {(int64_t)fa.numRegs,(int64_t)fa.localSizeBytes,(int64_t)occ,(int64_t)sm};\n}\n'
 return '#include <vector>\n'+s
def _d32_compile(**kw):
 from torch.utils.cpp_extension import load_inline
 return load_inline(**kw)
def _d32_build():
 _d32_validate()
 if not D32_BITS:
  return
 t1=time.time()
 try:
  cu=_d32_src()
  cpp=_SK2_CPP+'\n#include <vector>\nstd::vector<int64_t> d32_info(int64_t pmax);\n'
  flags=list(_SK2_FLAGS)+['-std=c++20']
  tag=hashlib.sha1((cpp+cu+' '.join(flags)).encode()).hexdigest()[:12]
  kw=dict(cpp_sources=cpp,cuda_sources=cu,functions=['sk2','sk2_set_pdl','d32_info'],
   extra_cflags=['-std=c++20'],extra_cuda_cflags=flags,verbose=False)
  try:
   _D32['mod']=_d32_compile(name=f'd32_{tag}',no_implicit_headers=True,**kw)
  except TypeError:
   _D32['mod']=_d32_compile(name=f'd32_{tag}i',**kw)
 except Exception as ex:
  _D32['err']=repr(ex)[-600:]
 _D32['secs']=time.time()-t1
def _d32_call(x,n,k,pdl,out=None):
 ci,stream,tab,pmax,b4=_D32['bank'][D32_VARIANT]
 if out is None:
  out=torch.empty((x.shape[0],n),dtype=torch.bfloat16,device=x.device)
 mod=_D32['mod']
 mod.sk2_set_pdl(pdl)
 mod.sk2(stream,x,out,tab,pmax,b4,x.shape[0],n,k,ci,32,0,_SK2['dummy'])
 return out
def _d32_selftest(w,n,k,gen):
 # The original SK2 self-test is kept. This gate additionally checks fresh x,
 # repeat determinism, NaN-poisoned complete output coverage and the stock arm.
 name='unpooler32_r144' if D32_VARIANT==1 else 'qkv32_r208'
 if not _SK2['ok'].get(name):
  raise RuntimeError('D32 stock SK2 control unavailable')
 worst=0.0
 for scale in (0.5,2.0):
  x=(torch.randn((32,k),generator=gen,device=w.device)*scale).to(torch.bfloat16)
  ref=x.float()@w.float().t()
  stock=_sk2_run(name,x,n,k,0)
  outs=[]
  for p in (0,2,2):
   out=torch.full((32,n),float('nan'),dtype=torch.bfloat16,device=w.device)
   outs.append(_d32_call(x,n,k,p,out))
  torch.cuda.synchronize()
  e=max(float((o.float()-ref).norm()/ref.norm().clamp_min(1e-30))for o in outs)
  es=max(float((o.float()-stock.float()).norm()/stock.float().norm().clamp_min(1e-30))for o in outs)
  rep=all(torch.equal(o,outs[0])for o in outs)
  if not(e<4e-3 and es<2e-3 and rep and all(bool(torch.isfinite(o.float()).all())for o in outs)):
   raise RuntimeError(f'D32 self-test rel {e} stock {es} repeat {rep}')
  worst=max(worst,e,es)
 return worst
def _d32_prepare(state,t0):
 _d32_validate()
 _D32['bank'],_D32['ok'],_D32['info']={},False,None
 if not D32_BITS:
  return
 try:
  _pt1_wait('d32')
  if time.time()-t0>260:
   raise RuntimeError('D32 prepare budget guard (260 s)')
  if _D32['mod']is None and _D32['err']is None:
   _d32_build()
  if _D32['mod']is None:
   raise RuntimeError(f"D32 build failed: {_D32['err']}")
  dense=state['dense']
  dev=dense['qkv_gate'].device
  ci,layer,n,R,nt,nwr,wsk,upw,st=_D32_CFG
  w=dense[layer].contiguous()
  k=_SK2_K[layer]
  if w.dtype!=torch.bfloat16 or tuple(w.shape)!=(n,k):
   raise RuntimeError('D32 weight shape/dtype mismatch')
  comp=_sk2_compress(w)
  if comp is None:
   raise RuntimeError('D32 weights not losslessly packable')
  tab,pmax=_sk2_table(comp,n,R)
  info=_D32['mod'].d32_info(pmax)
  _D32['info']=tuple(info)
  if info[1]!=0 or info[2]<2:
   raise RuntimeError(f'D32 zero-local/two-CTA gate failed: {info}')
  _D32['bank'][D32_VARIANT]=(ci,_sk2_stream(comp['packed'],n,R,nt,nwr,wsk,upw),tab,pmax,comp['base4'])
  gen=torch.Generator(device=dev).manual_seed(20261105)
  worst=_d32_selftest(w,n,k,gen)
  if time.time()-t0>280:
   raise RuntimeError('D32 prepare budget guard (280 s)')
  _D32['ok']=True
  print(f"[d32] LIVE bit {D32_VARIANT}: info {info}, rel {worst:.2e}, nvcc {_D32['secs']:.1f}s",flush=True)
 except Exception as ex:
  _D32['ok']=False
  _D32['err']=repr(ex)[-600:]
  print(f"[d32] OFF bit {D32_VARIANT}: {_D32['err']}",flush=True)
def _d32_wanted(proj,x):
 _d32_validate()
 return bool(D32_BITS and proj==('unpooler' if D32_VARIANT==1 else 'qkv')
  and x.dim()==2 and x.shape[0]==32)
def _d32_run_checked(proj,x):
 if not _D32['ok'] or D32_VARIANT not in _D32['bank']:
  raise RuntimeError(f"D32 enabled but not prepared/live: {_D32['err']}")
 ci,layer,n,R,nt,nwr,wsk,upw,st=_D32_CFG
 k=_SK2_K[layer]
 if not(x.dtype==torch.bfloat16 and x.shape[1]==k and x.is_contiguous()and x.data_ptr()%16==0):
  raise RuntimeError('D32 invalid activation')
 return _d32_call(x,n,k,SK2_PDL.get(proj,0))
# S7-D32 END
'''


def generate(base, variant, enable=False):
    bit = variant
    lab = LAB.replace('@VARIANT@', str(bit)).replace('@CONFIG@', repr(CONFIGS[bit]))
    if enable:
        lab = sub(lab, 'D32_BITS=0\n', f'D32_BITS={bit}\n')
    lines = base.splitlines(keepends=True)
    assert lines[0] == '# okc: problem=muse_decode_layer\n' and lines[1].startswith('# nonce ')
    lines[1] = f'# nonce d32-v{variant} (S7 SK2 {CONFIGS[bit][1]} two-CTA geometry; selector {bit if enable else 0})\n'
    s = ''.join(lines)
    s = sub(s, 'def _sk2(state,proj,x):\n', lab + '\ndef _sk2(state,proj,x):\n'
            ' if _d32_wanted(proj,x):\n  return _d32_run_checked(proj,x)\n')
    s = sub(s, '  jobs.append(("sk2",_sk2_build))\n',
            '  jobs.append(("sk2",_sk2_build))\n if D32_BITS:\n  jobs.append(("d32",_d32_build))\n')
    # After the entire base prepare sequence, so no existing time guard shifts.
    s = sub(s, ' _tinys_prepare()\n', ' _tinys_prepare()\n _d32_prepare(state,t0)\n')
    assert '\r' not in s and len(s.encode()) < LIMIT
    compile(s, f'<d32-v{variant}>', 'exec')
    return s


def namespace(source, torch=None):
    funcs = {'_sk2_stream', '_sk2_table', '_sk2_run', '_sk2', '_sk2_build',
             '_gd_key', '_gd_nk', '_gd_cat_key', '_gd_bucket'}
    constants = {'HIDDEN','D_EXPERT','N_EXPERTS','_SK2_CUDA','_SK2_CPP','_SK2_FLAGS',
                 '_SK2_CFGS','_SK2_PROJ','_SK2_K','_SK2','SK2_T','SK2_ON','SK2_ENABLE','SK2_PDL',
                 'GD_ROUTE','GD_CAT_ROUTE','GD_DIMS','GD_SMS','GD_CAT_UNITS','DG1_GEOMETRY',
                 'UNPOOL_SLICE','SLICED_UNPOOL_MAX_TOKENS','VD_ROUTE','VD_BUCKETS'}
    nodes=[]
    for n in ast.parse(source).body:
        if isinstance(n,ast.FunctionDef) and (n.name in funcs or n.name.startswith('_d32_')):
            nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and
             (t.id in constants or t.id.startswith('_D32') or t.id.startswith('D32_')) for t in n.targets):
            nodes.append(n)
    ns={'torch':torch,'time':time,'hashlib':hashlib,'_nb64':lambda t:33<=t<=64}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real-base-and-d32-ast>','exec'),ns)
    return ns


def accounting(cfg):
    ci,layer,n,R,nt,nwr,wsk,upw,st=cfg
    k=8192 if layer=='qkv_gate' else 4096
    nw=nwr*wsk
    ne=min(st,4)
    wb=nw*upw*nt*768
    xb=wsk*upw*32*128
    ss=(wb+xb+1023)&~1023
    stages=k//(wsk*upw*64)
    return dict(cfg=ci,N=n,K=k,R=R,CTAs=n//R,threads=32*(nw+1+ne),stages=stages,
                ring_stages=st,weight_bytes_per_stage=wb,activation_bytes_per_stage=xb,
                weight_bytes_total=wb*stages*(n//R),activation_L2_request_bytes=xb*stages*(n//R),
                dynamic_smem_without_ptab=1024+st*ss+16*st+32+ne*32*(R+1)*4+128,
                ptab_bytes='8*pmax per CTA (data-dependent, L2)',
                explicit_PTX_per_consumer_thread_per_stage=dict(
                    weight_LDS=2*nt*upw,x_LDSM=4*2*upw,decode_PRMT_LOP3=12*nt*upw,
                    MMA=4*2*nt*upw),
                explicit_PTX_consumer_warp_instructions_total=(22*nt+8)*upw*stages*nw*(n//R),
                excluded_instructions='address/loop arithmetic, waits, escape-dependent instructions; SASS gate reports static totals')


def protocol(cfg):
    _,_,n,R,nt,nwr,wsk,upw,st=cfg
    nw=nwr*wsk
    ne=min(st,4)
    nc=nw*32
    # Exhaustively prove epilogue ownership at M=32, with the actual source maps.
    staged=set()
    for tid in range(nc):
        warp,lane=divmod(tid,32)
        wr,wk=divmod(warp,wsk)
        g,t4=divmod(lane,4)
        for mt in range(2):
            for nn in range(nt):
                row=(wr*nt+nn)*8+2*t4
                tok=mt*16+g
                for dr,dt in ((0,0),(1,0),(0,8),(1,8)):
                    key=(wk,tok+dt,row+dr)
                    assert key not in staged
                    staged.add(key)
    assert len(staged)==wsk*32*R
    combined=[(idx//(R//4),(idx%(R//4))*4+c)
              for tid in range(nc) for idx in range(tid,(R//4)*32,nc) for c in range(4)]
    assert len(set(combined))==len(combined)==32*R
    corr=[i for lane in range(32) for i in range(lane,32*(R+1),32)]
    assert sorted(corr)==list(range(32*(R+1)))
    rp=R+((40-(R%32))%32)
    ring_bytes=st*(((nw*upw*nt*768+wsk*upw*32*128)+1023)&~1023)
    assert wsk*32*rp*4<=ring_bytes,'epilogue red overlaps correction storage'
    return dict(CTAs=[0,n//R-1],tid0='init full[0:P], expect+prefetch W, wait, TMA X, ptab; then full[P:NS], empty[0:NS], cbar/qbar',
                zero_row_tids=[32,63],zero_row_bytes=128,producer_tids=[nc,nc+31],
                producer_issue_tid=nc,escape_tids=[nc+32,(nw+1+ne)*32-1],
                corr_zero_elements_per_escape_warp=32*(R+1),empty_arrivals_per_stage=nw+ne,
                cbar_arrivals=32*ne,red_and_combine_tids=[0,nc-1],red_float_elements=len(staged),
                output_elements=len(combined),
                cleanup='No persistent counters/resets. Producer exits after last issue; each escape thread arrives cbar after all stages; consumers wait cbar and two NC-thread barriers before combine. Every CTA retains all roles; UPW=2 keeps release after all LDS/MMA. Smem alias reuse follows final full wait and cbar.')


def ring_proof(cfg):
    """Random interleavings of every role's existing stage-loop protocol."""
    import random
    _,layer,n,R,nt,nwr,wsk,upw,st=cfg
    k=8192 if layer=='qkv_gate' else 4096
    stages=k//(wsk*upw*64)
    readers=nwr*wsk+min(st,4)
    completed=0
    for pdl in (0,1,2,3,4):
        P=min(pdl,st,stages) if pdl>0 else 1
        for seed in range(64):
            rng=random.Random(seed+100*pdl)
            issued=P; ready=set(); pending=set(range(P)); cursor=[0]*readers
            arrivals={i:set() for i in range(stages)}
            while min(cursor)<stages:
                options=[('land',i) for i in pending]
                if issued<stages and (issued<st or len(arrivals[issued-st])==readers):
                    options.append(('issue',issued))
                options.extend(('read',r) for r,s in enumerate(cursor) if s in ready)
                assert options,('deadlock',cfg,pdl,seed,issued,cursor)
                action,i=rng.choice(options)
                if action=='issue':
                    assert i<st or len(arrivals[i-st])==readers
                    pending.add(i); issued+=1
                elif action=='land':
                    pending.remove(i); ready.add(i)
                else:
                    stage=cursor[i]
                    assert i not in arrivals[stage]
                    arrivals[stage].add(i); cursor[i]+=1
            assert issued==stages and all(len(a)==readers for a in arrivals.values())
            completed+=1
    return dict(interleavings=completed,stages=stages,readers_per_stage=readers,
                outcome='all stage releases covered once; no slot reuse before every consumer/escape warp releases')


class Array:
    """Minimal NumPy adapter for the unchanged base stream/table builders."""
    def __init__(self,a):
        import numpy as np
        self.a=np.asarray(a)
        self.device='cpu'
    @property
    def shape(self): return self.a.shape
    def view(self,*shape):
        import numpy as np
        return Array(self.a.view(shape[0]) if len(shape)==1 and isinstance(shape[0],type) else self.a.reshape(*shape))
    def contiguous(self): return Array(self.a.copy())
    def long(self):
        import numpy as np
        return Array(self.a.astype(np.int64))
    def to(self,dtype): return Array(self.a.astype(dtype))
    def numel(self): return self.a.size
    def max(self): return Array(self.a.max())
    def item(self): return self.a.item()
    def __getitem__(self,key):
        if isinstance(key,tuple): key=tuple(k.a if isinstance(k,Array) else k for k in key)
        elif isinstance(key,Array): key=key.a
        return Array(self.a[key])
    def __setitem__(self,key,v):
        if isinstance(key,tuple): key=tuple(k.a if isinstance(k,Array) else k for k in key)
        elif isinstance(key,Array): key=key.a
        self.a[key]=v.a if isinstance(v,Array) else v
    def _op(self,v,fn): return Array(fn(self.a,v.a if isinstance(v,Array) else v))
    def __add__(self,v): return self._op(v,lambda a,b:a+b)
    __radd__=__add__
    def __sub__(self,v): return self._op(v,lambda a,b:a-b)
    def __mul__(self,v): return self._op(v,lambda a,b:a*b)
    __rmul__=__mul__
    def __floordiv__(self,v): return self._op(v,lambda a,b:a//b)
    def __mod__(self,v): return self._op(v,lambda a,b:a%b)
    def __lshift__(self,v): return self._op(v,lambda a,b:a<<b)
    def __or__(self,v): return self._op(v,lambda a,b:a|b)
    def __lt__(self,v): return self._op(v,lambda a,b:a<b)


class Scalar(float):
    def clamp_min(self,v): return Scalar(max(self,v))


class Tensor:
    """Shape/initialization-only CUDA tensor mock, NOT numerical evidence."""
    def __init__(self,shape,dtype='bf16',device='cpu',initialized=True):
        self.shape=tuple(shape); self.dtype=dtype; self.device=device; self.initialized=initialized
    def dim(self): return len(self.shape)
    def data_ptr(self): return 16
    def is_contiguous(self): return True
    def contiguous(self): return self
    def float(self): return Tensor(self.shape,'f32',self.device,self.initialized)
    def to(self,dtype): return Tensor(self.shape,dtype,self.device,self.initialized)
    def t(self): return Tensor(self.shape[::-1],self.dtype,self.device,self.initialized)
    def __mul__(self,v): return Tensor(self.shape,self.dtype,self.device,self.initialized)
    def __sub__(self,v):
        assert self.shape==v.shape
        return Tensor(self.shape,self.dtype,self.device,self.initialized and v.initialized)
    def __matmul__(self,v):
        assert self.shape[1]==v.shape[0]
        return Tensor((self.shape[0],v.shape[1]),self.dtype,self.device)
    def norm(self): return Scalar(0.0)  # Mock arithmetic: never reported as correctness.


def base_state(base):
    """Evaluate state['dense']'s exact value from _r1_prepare, not a guessed layout."""
    fn=next(n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef) and n.name=='_r1_prepare')
    dic=next(n.value for n in ast.walk(fn) if isinstance(n,ast.Assign) and
             any(isinstance(t,ast.Name) and t.id=='state' for t in n.targets))
    field=next(v for k,v in zip(dic.keys,dic.values) if isinstance(k,ast.Constant) and k.value=='dense')
    shapes={'qkv_gate':(2304,8192),'o':(8192,1024),'pooler':(4096,8192),'unpooler':(8192,4096)}
    weights={n:Tensor(sh) for n,sh in shapes.items()}
    return {'dense':eval(compile(ast.Expression(field),'<base-state-dense>','eval'),{'weights':weights})}


def cpu_dry_run(base,source):
    import numpy as np
    calls=[]; builds=[]; waits=[]
    class Generator:
        def __init__(self,**kw): pass
        def manual_seed(self,seed): return self
    def zeros(*shape,dtype=None,device=None):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)): shape=shape[0]
        return Array(np.zeros(shape,dtype=dtype))
    torch=types.SimpleNamespace(bfloat16='bf16',float32='f32',int32=np.int32,long=np.int64,
       int64=np.int64,arange=lambda n,**kw:Array(np.arange(n)),zeros=zeros,
       bincount=lambda x,**kw:Array(np.bincount(x.a.astype(np.int64),**kw)),
       argsort=lambda x:Array(np.argsort(x.a,kind='stable')),cumsum=lambda x,dim:Array(np.cumsum(x.a,axis=dim)),
       Generator=Generator,empty=lambda sh,**kw:Tensor(sh,initialized=False,**kw),
       full=lambda sh,v,**kw:Tensor(sh,initialized=False,**kw),
       randn=lambda sh,**kw:Tensor(sh,'f32',kw.get('device','cpu')),
       equal=lambda a,b:a.initialized and b.initialized and a.shape==b.shape,
       isfinite=lambda a:types.SimpleNamespace(all=lambda:a.initialized),
       cuda=types.SimpleNamespace(synchronize=lambda:None))
    ns=namespace(source,torch); ns['print']=lambda *a,**kw:None
    variant=ns['D32_VARIANT']; cfg=CONFIGS[variant]
    _,layer,n,R,nt,nwr,wsk,upw,st=cfg
    k=ns['_SK2_K'][layer]
    class Module:
        info=(96,0,2,100000)  # SYNTHETIC path inputs, not measured resources.
        def d32_info(self,pmax): return self.info
        def sk2_set_pdl(self,pdl): self.pdl=pdl
        def sk2(self,stream,x,out,tab,pmax,b4,M,N,K,ci,tokr,dg,dummy):
            assert (M,N,K,tokr)==(32,n,k,32) and ci==cfg[0]
            assert stream.shape[:5]==(k//(wsk*upw*64),n//R,nwr*wsk,upw,nt)
            assert tab.shape==(n//R,pmax,2) and dummy is ns['_SK2']['dummy']
            assert out.shape==(M,N)
            out.initialized=True
            calls.append((ci,self.pdl,x.shape,out.shape))
    mod=Module()
    def compiler(**kw):
        assert '-std=c++20' in kw['extra_cflags'] and '-std=c++20' in kw['extra_cuda_cflags']
        assert kw['functions']==['sk2','sk2_set_pdl','d32_info']
        assert kw['cuda_sources']==ns['_d32_src']()
        builds.append(kw['name'])
        return mod
    ns['_d32_compile']=compiler
    ns['_pt1_wait']=lambda name:waits.append(name)
    def compress(w):
        N,K=w.shape
        # One identity per 8x64 weight tile. Executes the real stream indexing.
        packed=Array(np.arange((N//8)*(K//64),dtype=np.int64).reshape(N//8,K//64,1))
        rows=np.arange(N,dtype=np.int64)[::7][::-1]
        cols=(rows*257+63)%K
        return {'packed':packed,'rows':Array(rows),'cols':Array(cols),
                'delta':Array(np.full(rows.size,0.001,np.float32)),'base4':123}
    ns['_sk2_compress']=compress
    stock_name='unpooler32_r144' if variant==1 else 'qkv32_r208'
    ns['_SK2']['dummy']=object()
    for name in ns['_SK2_CFGS']:
        ns['_SK2']['ok'][name]=True
        ns['_SK2']['w'][name]=('stock weights',name)
    # Base numerical CUDA call is mocked; base wrappers and layout are real AST.
    ns['_sk2_run']=lambda name,x,N,K,pdl:Tensor((32,N))
    stock_snapshot=copy.copy(ns['_SK2']['w'])
    state=base_state(base)
    ns['D32_BITS']=0
    ns['_d32_build'](); ns['_d32_prepare'](state,time.time())
    assert not builds and not waits and not ns['_D32']['bank']
    assert not ns['_d32_wanted']('qkv',Tensor((32,8192)))
    ns['D32_BITS']=variant
    proj='unpooler' if variant==1 else 'qkv'
    try:
        ns['_sk2'](state,proj,Tensor((32,k)))
        raise AssertionError('enabled unprepared arm silently accepted')
    except RuntimeError as e:
        assert 'not prepared/live' in str(e)
    ns['_d32_prepare'](state,time.time())
    assert ns['_D32']['ok'],ns['_D32']['err']
    assert len(builds)==1 and len(calls)==6 and waits==['d32']
    assert ns['_SK2']['w']==stock_snapshot
    assert set(ns['_D32']['bank'])=={variant}
    ns['_sk2'](state,proj,Tensor((32,k)))
    assert len(calls)==7 and calls[-1][1]==ns['SK2_PDL'][proj]
    for T in (1,8,9,16,17,31,33,48,64,128,256):
        assert not ns['_d32_wanted'](proj,Tensor((T,k)))
        assert ns['_sk2'](state,proj,Tensor((T,k))) is None
    assert not ns['_d32_wanted']('o',Tensor((32,1024)))
    other='qkv' if variant==1 else 'unpooler'
    _,on,ok=ns['_SK2_PROJ'][other]
    assert ns['_sk2'](state,other,Tensor((32,ok))).shape==(32,on)
    assert len(calls)==7,'untouched projection dispatched D32'
    for bad in (Tensor((32,k-1)),Tensor((32,k),'f32')):
        try:
            ns['_d32_run_checked'](proj,bad)
            raise AssertionError('bad activation accepted')
        except RuntimeError: pass
    # Execute both real stream/table helpers, then invert their coordinates.
    ci,stream,tab,pmax,b4=ns['_D32']['bank'][variant]
    flat=stream.a.reshape(-1)
    assert np.array_equal(np.sort(flat),np.arange((n//8)*(k//64)))
    # Verify the CUDA (stage, CTA, warp, uu, nt) address, not merely a permutation.
    for ss in range(k//(wsk*upw*64)):
        for cc in range(n//R):
            for ww in range(nwr*wsk):
                for uu in range(upw):
                    for nn in range(nt):
                        tile=cc*(R//8)+(ww//wsk)*nt+nn
                        unit=ss*(wsk*upw)+(ww%wsk)*upw+uu
                        assert int(stream.a[ss,cc,ww,uu,nn,0])==tile*(k//64)+unit
    comp=compress(state['dense'][layer])
    recovered=[]
    for c in range(n//R):
        cnt=int(tab.a[c,0,0])
        assert cnt<pmax
        for slot in range(1,cnt+1):
            ent=int(tab.a[c,slot,0]); recovered.append((c*R+(ent>>16),ent&65535,int(tab.a[c,slot,1])))
    expected=sorted(zip(comp['rows'].a.tolist(),comp['cols'].a.tolist(),comp['delta'].a.view(np.int32).tolist()))
    assert sorted(recovered)==expected
    empty=dict(comp,rows=Array(np.array([],np.int64)),cols=Array(np.array([],np.int64)),delta=Array(np.array([],np.float32)))
    et,ep=ns['_sk2_table'](empty,n,R)
    assert ep==2 and not et.a.any()
    failures=[]
    bad_dtype={'dense':dict(state['dense'],**{layer:Tensor((n,k),'f32')})}
    bad_shape={'dense':dict(state['dense'],**{layer:Tensor((n-1,k))})}
    for label,stt,info,dt,comp_fn in (
        ('missing_dense',{},(96,0,2,100000),0,compress),
        ('missing_qkv',{'dense':{}},(96,0,2,100000),0,compress),
        ('weight_dtype',bad_dtype,(96,0,2,100000),0,compress),
        ('weight_shape',bad_shape,(96,0,2,100000),0,compress),
        ('local_memory',state,(96,4,2,100000),0,compress),
        ('one_CTA',state,(96,0,1,100000),0,compress),
        ('timeout',state,(96,0,2,100000),270,compress),
        ('unpackable',state,(96,0,2,100000),0,lambda w:None)):
        mod.info=info; ns['_sk2_compress']=comp_fn
        ns['_d32_prepare'](stt,time.time()-dt)
        assert not ns['_D32']['ok'],label
        failures.append(label)
    ns['_sk2_compress']=compress; mod.info=(96,0,2,100000)
    ns['_D32']['mod']=None; ns['_D32']['err']=None
    def broken(**kw): raise OSError('synthetic compile failure')
    ns['_d32_compile']=broken; ns['_d32_prepare'](state,time.time())
    assert not ns['_D32']['ok'] and 'synthetic compile failure' in ns['_D32']['err']
    ns['_D32']['mod']=None; ns['_D32']['err']=None
    def legacy(**kw):
        if 'no_implicit_headers' in kw: raise TypeError('legacy API')
        return compiler(**kw)
    ns['_d32_compile']=legacy; ns['_d32_prepare'](state,time.time())
    assert ns['_D32']['ok']
    ns['_SK2']['ok'][stock_name]=False
    ns['_d32_prepare'](state,time.time())
    assert not ns['_D32']['ok'] and 'control unavailable' in ns['_D32']['err']
    ns['_SK2']['ok'][stock_name]=True
    # Test the self-test rejection and the guard *after* self-testing too.
    real_selftest=ns['_d32_selftest']
    ns['_d32_selftest']=lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic numeric failure'))
    ns['_d32_prepare'](state,time.time())
    assert not ns['_D32']['ok'] and 'synthetic numeric failure' in ns['_D32']['err']
    ticks=iter((0.0,281.0))
    ns['time']=types.SimpleNamespace(time=lambda:next(ticks))
    ns['_d32_selftest']=lambda *a:0.0
    ns['_d32_prepare'](state,0.0)
    assert not ns['_D32']['ok'] and '280 s' in ns['_D32']['err']
    ns['time']=time; ns['_d32_selftest']=real_selftest
    # Run the actual newly inserted PT1 branch with a CPU-only jobs list.
    pt1=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_pt1_start')
    branch=next(n for n in pt1.body if isinstance(n,ast.If) and ast.unparse(n.test)=='D32_BITS')
    for bit in (0,variant):
        ns.update(D32_BITS=bit,jobs=[])
        exec(compile(ast.Module(body=[branch],type_ignores=[]),'<real-PT1-D32-branch>','exec'),ns)
        assert ns['jobs']==([('d32',ns['_d32_build'])] if bit else [])
    ns['D32_BITS']=3
    try:
        ns['_d32_validate'](); raise AssertionError('invalid bit accepted')
    except ValueError: pass
    ns['D32_BITS']=0
    assert not ns['_d32_wanted'](proj,Tensor((32,k)))
    return dict(status='PASS',state_expression='real _r1_prepare dense comprehension',
        prepare='real new Python paths; CUDA and numerical tensor operations mocked',
        stream_weight_tiles=len(flat),escape_records=len(recovered),empty_escape_table=True,
        self_test_paths='fresh x, NaN output, pdl 0/2/2, stock comparison executed with shape mocks',
        failures=failures+['compile_failure','legacy_build_API','unprepared_arm','bad_activation','invalid_selector',
                          'stock_not_live','numeric_rejection','post_selftest_timeout'],
        bank='independent _D32; _SK2 weight bank unchanged',GPU_numerics='NOT RUN')


def change_lines(base,source):
    changes=[]
    for tag,a,b,c,d in difflib.SequenceMatcher(a=base.splitlines(),b=source.splitlines(),autojunk=False).get_opcodes():
        if tag!='equal': changes.append(dict(kind=tag,base=[a+1,b],variant=[c+1,d]))
    return changes


def route_map(base):
    ns=namespace(base)
    loc={t.id:n.lineno for n in ast.parse(base).body if isinstance(n,ast.Assign)
         for t in n.targets if isinstance(t,ast.Name)}
    rows=[]
    for T in (16,32):
        for proj in ('qkv','o','cat','unpooler'):
            if T==32 and proj in ('qkv','unpooler'):
                name=ns['_SK2_PROJ'][proj][0]
                cfg=ORIGINAL[2 if proj=='qkv' else 1]
                _,_,_,_,nt,nwr,wsk,upw,st=cfg
                route=(2,32,nt,nwr,wsk,upw,st,True,False,1,1,'__nv_bfloat16',False,False)
                family='SK2'; table='_SK2_CFGS'
            elif proj=='cat':
                key=ns['_gd_cat_key'](ns['GD_CAT_ROUTE'][T])
                route=key[:9]+(0,key[9]); family='GD_CAT'; table='GD_CAT_ROUTE'
            elif proj=='o' and T==32:
                route=ns['DG1_GEOMETRY'][(proj,T)][:3]; family='DG1'; table='DG1_GEOMETRY'
            else:
                route=ns['_gd_key'](*ns['_gd_nk'](proj,T),ns['GD_ROUTE'][(proj,T)])+(0,0)
                family='GD'; table='GD_ROUTE'
            rows.append(dict(T=T,role=proj,family=family,instance=route,
                             route_table=table,source=f'{BASE.relative_to(ROOT).as_posix()}:{loc[table]}'))
    return rows


def cpu_gate(base,sources):
    report={'CPU':'PASS','GPU':'NOT RUN','base_sha256':hashlib.sha256(base.encode()).hexdigest(),
            'routes':route_map(base),'variants':{}}
    funcs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(base).body if isinstance(n,ast.FunctionDef)}
    for v,s in sources.items():
        enabled=value(s,'D32_BITS')!=0
        assert s==generate(base,v,enabled),'stale output'
        assert s.splitlines()[1].startswith(f'# nonce d32-v{v} (')
        for key in ('_SK2_CUDA','_SK2_CPP','_SK2_CFGS','_SK2_PROJ','GD_ROUTE','GD_CAT_ROUTE','DG1_GEOMETRY','VD_ROUTE'):
            assert expression(base,key)==expression(s,key),(v,key)
        after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
        for key in funcs:
            if key not in ('prepare','_pt1_start','_sk2'):
                assert funcs[key]==after[key],(v,'unexpected function change',key)
        ns=namespace(s)
        cu=ns['_d32_src']()
        # New module contains exactly the requested template, never the old configs.
        assert 'if (cfg == 6 ' not in cu and 'if (cfg == 11 ' not in cu
        assert cu.count('launch_sk2<2, 32,')==1
        assert '(NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 2)' in cu
        before=accounting(ORIGINAL[v]); after_acc=accounting(CONFIGS[v])
        assert before['weight_bytes_total']==after_acc['weight_bytes_total']
        assert after_acc['weight_bytes_per_stage']<=before['weight_bytes_per_stage']
        report['variants'][v]=dict(bytes=len(s.encode()),enabled=enabled,sha256=hashlib.sha256(s.encode()).hexdigest(),
          dry_run=cpu_dry_run(base,s),before=before,after=after_acc,protocol=protocol(CONFIGS[v]),ring=ring_proof(CONFIGS[v]),
          changes=change_lines(base,s),WEIGHT_STREAM_GROWTH=False)
    # Real py_compile with all bytecode writes intercepted; no __pycache__ files.
    import py_compile
    from unittest.mock import patch
    for path in (Path(__file__),*OUTS.values()):
        with patch('importlib._bootstrap_external._write_atomic',lambda *a,**kw:None):
            py_compile.compile(str(path),cfile=str(path)+'.discarded-pyc',doraise=True)
    assert b'\r' not in SHELL.read_bytes() and SHELL.read_bytes().startswith(b'#!/bin/bash\n')
    report['py_compile']='PASS (bytecode writes intercepted)'
    report['shell_LF']='PASS'
    return report


def resources(log):
    found={}
    for block in re.split(r'ptxas info\s*: Compiling entry function ',log)[1:]:
        name=block.split("'",2)[1]
        reg=re.search(r'Used (\d+) registers',block)
        spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block)
        assert reg and spill,block
        found[name]=dict(registers=int(reg[1]),stack=int(spill[1]),spill_stores=int(spill[2]),spill_loads=int(spill[3]))
    assert found,log
    return found


def sass_functions(sass):
    return {b.splitlines()[0].strip():[l.strip() for l in b.splitlines() if re.match(r'\s*/\*',l)]
            for b in re.split(r'Function\s*:\s*',sass)[1:]}


def linux_gate(base,sources,report):
    """Host compile/link + sm_90a SASS only; never loads an extension or GPU."""
    import tempfile
    import sysconfig
    import torch
    from torch.utils.cpp_extension import include_paths,library_paths
    build=Path(tempfile.mkdtemp(prefix='s7-d32-gate-'))
    report['build_dir']=str(build)
    def run(cmd):
        p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        report.setdefault('commands',[]).append(dict(argv=cmd,returncode=p.returncode))
        if p.returncode: raise RuntimeError(repr(cmd)+'\n'+p.stdout[-16000:])
        return p.stdout
    inc=['-I'+p for p in include_paths(device_type='cuda')]+['-I'+sysconfig.get_path('include')]
    abi=f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}'
    flags=['-O3','-std=c++20','-arch=sm_90a','-Xptxas=-v,-warn-spills','-Xcompiler=-fPIC',
           '--expt-relaxed-constexpr','-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__',
           '-D__CUDA_NO_BFLOAT16_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__',abi]
    report['nvcc_version']=run(['nvcc','--version'])
    baseline=value(base,'_SK2_CUDA')
    assembly={}; parsed={}
    texts={'base':baseline,'off':value(sources[1],'_SK2_CUDA')}
    texts.update({f'v{v}':namespace(s)['_d32_src']() for v,s in sources.items()})
    for name,cu in texts.items():
        src,obj=build/(name+'.cu'),build/(name+'.o')
        src.write_bytes(cu.encode())
        log=run(['nvcc',*flags,*inc,'-c',str(src),'-o',str(obj)])
        (build/(name+'.compile.log')).write_bytes(log.encode())
        parsed[name]=resources(log)
        sass=run(['cuobjdump','--dump-sass',str(obj)])
        (build/(name+'.sass')).write_bytes(sass.encode())
        assembly[name]=sass_functions(sass)
        res=run(['cuobjdump','--dump-resource-usage',str(obj)])
        (build/(name+'.resources.txt')).write_bytes(res.encode())
    assert assembly['base']==assembly['off'],'default SASS/encoding changed'
    assert parsed['base']==parsed['off'],'default ptxas resources changed'
    report['default_SASS_identical']=True
    report['ptxas']=parsed
    for v in (1,2):
        kernels={n:r for n,r in parsed[f'v{v}'].items() if 'sk2_kernel' in n}
        assert len(kernels)==1,kernels
        symbol,r=next(iter(kernels.items()))
        assert r['stack']==r['spill_stores']==r['spill_loads']==0,(v,r)
        acc=accounting(CONFIGS[v]); threads=acc['threads']
        rounded=((r['registers']+7)//8)*8
        assert rounded*threads*2<=65536,(v,'register occupancy',r,threads)
        # Real pmax is tested by cudaOccupancyMaxActiveBlocks... in prepare.
        assert acc['dynamic_smem_without_ptab']*2<232448
        lines=assembly[f'v{v}'][symbol]
        ops={op:sum(bool(re.search(r'\b'+op+r'(?:\.|\b)',l)) for l in lines)
             for op in ('LDG','LDS','LDSM','HMMA','PRMT','LOP3','LDL','STL','BAR','MBAR')}
        assert ops['LDL']==ops['STL']==0,(v,ops)
        report['variants'][v]['compile']=dict(status='PASS',symbol=symbol,resources=r,
          SASS_static_instruction_lines=len(lines),SASS_op_counts=ops,
          occupancy='register/smem feasibility passed; actual pmax occupancy checked in prepare')
        cpp=value(base,'_SK2_CPP')+'\n#include <vector>\nstd::vector<int64_t> d32_info(int64_t);\n'
        cpp+='PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("sk2",&sk2); m.def("sk2_set_pdl",&sk2_set_pdl); m.def("d32_info",&d32_info); }\n'
        host=build/f'v{v}.cpp'; host.write_bytes(cpp.encode())
        hostobj=build/f'v{v}.host.o'
        cxx=os.environ.get('CXX','c++')
        run([cxx,'-O2','-std=c++20','-fPIC',abi,'-DTORCH_EXTENSION_NAME=d32_gate',*inc,'-c',str(host),'-o',str(hostobj)])
        libs=['-L'+p for p in library_paths(device_type='cuda')]
        run([cxx,'-shared',str(hostobj),str(build/f'v{v}.o'),*libs,'-lc10','-ltorch','-ltorch_cpu',
             '-ltorch_python','-lcudart','-lc10_cuda','-ltorch_cuda','-o',str(build/f'v{v}.so')])
    report['CUDA_compile_host_link_SASS']='PASS'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enable',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--linux-gate',action='store_true')
    p.add_argument('--json',type=Path)
    args=p.parse_args()
    if args.json:
        assert args.linux_gate and args.json.resolve()==SHELL.with_suffix('.json').resolve()
    original=BASE.read_bytes(); base=original.decode()
    assert b'\r' not in original and len(original)==965632,(len(original),'base bytes drifted')
    if not args.check:
        for v,path in OUTS.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(generate(base,v,args.enable).encode())
    sources={v:path.read_bytes().decode() for v,path in OUTS.items()}
    report={}
    try:
        report=cpu_gate(base,sources)
        if args.linux_gate:
            linux_gate(base,sources,report)
        else:
            report['CUDA_compile_host_link_SASS']='NOT RUN: use tools/local/s7_d32_gate.sh under cudaenv.sh'
        report['status']='PASS' if args.linux_gate else 'CPU PASS; CUDA PENDING'
    except Exception as ex:
        report['status']='FAIL'; report['error']=repr(ex)
        raise
    finally:
        assert BASE.read_bytes()==original,'base changed during invocation'
        if args.json:
            args.json.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()

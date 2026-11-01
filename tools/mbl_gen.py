#!/usr/bin/env python3
""":"
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
exec python3 -B -I "$0" --linux-gates "$@"
":"""
"""MBL: exact-source MoE extraction; CPU dry run by default, no GPU execution.

Only writes mb_layer.py and mbl_report.md. This file is also the LF-only bash
compile/SASS runner, avoiding a fourth file outside the task's allowlist.
Linux gate JSON is embedded in the allowed report, not a separate sidecar.
"""
import ast
import builtins
import contextlib
import hashlib
import io
import json
import math
import pathlib
import subprocess
import sys
import symtable
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/muse_moe_w4a16/mb_layer.py"
REPORT = ROOT / "box_out/reset/mbl_report.md"
MANIFEST = []


def check(ok, why):
    if not ok:
        raise AssertionError(why)


def between(src, first, after, label):
    # Same count-before-replace discipline as tools/padz_patches.py.
    check(src.count(first) == 1, (label, "start anchor", first))
    check(src.count(after) == 1, (label, "end anchor", after))
    a, b = src.index(first), src.index(after)
    check(a < b, label)
    text = src[a:b]
    MANIFEST.append(dict(item=label, first=first, after=after,
                         base_line=src.count("\n", 0, a) + 1,
                         lines=text.count("\n"), sha256=hashlib.sha256(text.encode()).hexdigest()))
    return text


def node_text(src, name):
    nodes = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            nodes.append(n)
    check(len(nodes) == 1, (name, "unique top-level AST anchor"))
    n = nodes[0]
    lines = src.splitlines(keepends=True)
    text = "".join(lines[n.lineno - 1:n.end_lineno])
    MANIFEST.append(dict(item=name, base_line=n.lineno, lines=n.end_lineno-n.lineno+1,
                         first=lines[n.lineno-1].rstrip(), sha256=hashlib.sha256(text.encode()).hexdigest()))
    return text


ADAPTER = r'''
# MBL board adapter. CUDA and native helpers above are copied verbatim.
# The base VMS launcher hard-rejects M > 16 (MAXP=128); edges 17..24 use bf2.
MBL_EDGES=(1,2,4,5,8,9,12,13,16,17,24,25,32,33,48,49,63,64,128,192,193,256,257,320)
MBL_ARMS=("tiny","tiny2","vms_hi","vms","bf2","w72","hyb","hyb7")

def _mbl_status(name,good,reason):
 print(f"[mbl] {'LIVE' if good else 'FAIL'} {name}: {reason}",flush=True)

def _mbl_candidates(m):
 rep=_k2a_bucket_of(m)
 if rep is None:
  raise ValueError("MBL supports 1 <= M <= 320")
 if m<=4:
  return ("tiny","vms","bf2")
 if m<=max(HYBLO_T):
  return ("tiny2","vms","bf2")
 if m<=VMS_MAXT:
  return ("vms_hi","vms","bf2")
 if m<HYB_T[0]:
  return ("bf2",)
 return (("hyb7","hyb","w72","bf2")if m>=HYB_T7 else("hyb","w72","bf2"))

def _mbl_builds():
 # Native flags and native load_inline host functions, including both VMS builds.
 # Like the base's _pt1 build jobs, families overlap; native HYBLO itself still
 # builds tiny then tiny2 sequentially (its documented compiler-memory guard).
 def one(name,fn):
  try:
   fn()
  except Exception as ex:
   _mbl_status(name,False,"build "+repr(ex)[:600])
 jobs=[threading.Thread(target=one,args=(name,fn),daemon=True)
       for name,fn in(("VMS",_vms_build),("VML",_vml_build_job),("HYB",_hyb_build),("HYBLO",_hyblo_build))]
 for job in jobs:
  job.start()
 for job in jobs:
  job.join()
 mods={"vms":_VMS["mod"],"vms_hi":_VMS["mod_hi"],"hyb":_HYB["mod"],"hyb7":_HYB["mod7"],
       "tiny":_HYBLO["mods"].get("tiny"),"tiny2":_HYBLO["mods"].get("tiny2"),
       "bf2":_VML["mods"].get("bf2"),"w72":_VML["mods"].get("w72")}
 for name,mod in mods.items():
  if mod is None:
   _mbl_status(name,False,"module unavailable")
  elif name in("tiny","tiny2","hyb","hyb7"):
   mod.okc_set_pdl(int(HYBLO_PDL if name.startswith("tiny")else HYB_PDL))
  elif name in("bf2","w72"):
   _vml_cfg(mod) # initialize native host metadata before capture
 return mods

def _mbl_pack_bank(w,dev):
 # A native layer bank contains exactly 32 local experts; no guessed state key.
 st={}
 _mbl_pack_stage(st,w,dev)
 try:
  _mbl_pack_vms(st,w,dev)
 except Exception as ex:
  _mbl_status("vms-pack",False,repr(ex)[:400])
 try:
  if not _hyb_prmt_ok(w["w13_sf"],w["w2_sf"]):
   raise ValueError("HYB PRMT scale range")
  _mbl_pack_hyb(st,w,dev)
  st["hyb"]=_HYB["w"]
  _mbl_pack_hyblo(st,w,dev)
  st["hyblo"]=_HYBLO["w"]
 except Exception as ex:
  _mbl_status("hyb/hyblo-pack",False,repr(ex)[:400])
 return st

def _mbl_banks(weights,dev):
 # Prepare every expert once. Views cover ALL valid local_start values, including
 # unaligned windows; no repack, id conversion, tensor allocation or caching in kernel_fn.
 E=weights["w13"].shape[0]
 if E!=256:
  raise ValueError("board weights must contain 256 global experts")
 chunks=[]
 keys=("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")
 for lo in range(0,E,32):
  w={k:weights[k][lo:lo+32]for k in keys}
  w["post_norm_w"]=weights["post_norm_w"]
  chunks.append(_mbl_pack_bank(w,dev))
 shared=chunks[0]
 global_k={k:torch.cat([b[k]for b in chunks],dim=0)for k in("ws13","ws2","gsc13","gsc2")}
 have_vms=all("vms"in b for b in chunks)
 have_hyb=all("hyb"in b for b in chunks)
 have_tiny=all("hyblo"in b for b in chunks)
 if have_vms:
  global_v={k:torch.cat([b["vms"][k]for b in chunks],dim=0)for k in("wA","wB","gsA","gsB")}
 if have_hyb:
  global_h={j:torch.cat([b["hyb"][j]for b in chunks]).view(E,-1)for j in(2,5,12,13)}
 if have_tiny:
  global_t={j:torch.cat([b["hyblo"][j]for b in chunks]).view(E,-1)for j in(0,1,3,4,8,9,10,11)}
 banks={}
 for lo in range(E-31):
  st={k:t[lo:lo+32]for k,t in global_k.items()}
  st["nw"]=shared["nw"]
  if have_vms:
   st["vms"]=dict(shared["vms"],**{k:t[lo:lo+32]for k,t in global_v.items()})
  if have_hyb:
   st["hyb"]=list(shared["hyb"])
   for j,t in global_h.items():
    st["hyb"][j]=t[lo:lo+32].reshape(-1)
  if have_tiny:
   st["hyblo"]=list(st["hyb"])
   for j,t in global_t.items():
    st["hyblo"][j]=t[lo:lo+32].reshape(-1)
  banks[lo]={"k2a33":{"st":st}}
 # Avoid retaining the last chunk's duplicate global weight handles.
 _HYB["w"]=_HYBLO["w"]=None
 return banks

def _mbl_call(name,data,state):
 x,ids,tw=data["hidden_states"],data["topk_ids"],data["topk_weights"]
 m=x.shape[0]
 lo=int(data["config"]["local_start"])
 st=state["banks"][lo]["k2a33"]["st"]
 out=state["out"][m]
 mod=state["mods"][name]
 if name in("vms","vms_hi"):
  vs=st["vms"]
  mod.run(x,ids,tw,vs["wA"],vs["wB"],vs["gsA"],vs["gsB"],vs["nw"],vs["inter"],vs["ybuf"],vs["ss"],vs["ctr"],
  out,vs["nomarks"],m,K2A_POST_NORM_EPS,1,lo)
 elif name in("bf2","w72"):
  rep=_k2a_bucket_of(m)
  _vml_call(mod,st,state["vml_sc"][name],x,ids,tw,ls=lo,out=out,
  flags=VML_FLAGS.get(rep,193),cost=VML_COST.get(rep,VML_COST[32]))
 else:
  mod.okc_run(st["hyblo"]if name.startswith("tiny")else st["hyb"],state["h_sc"],x,ids,tw,out,lo,32,state["nsm"])
 return out

def kernel_fn(data,state):
 m=data["hidden_states"].shape[0]
 return _mbl_call(state["route"][m],data,state)

def _mbl_input(m,seed,dev,dist="normal",lo=0):
 # Native layer probes retain their seeds and distributions; board inputs keep
 # GLOBAL ids, never -1, and include no-local, all-local and mixed-zero rows.
 d=_hyblo_inputs(m,seed,dev,dist)if dist in("none","conc")else _k2a36_synth_inputs(m,seed,dev,dist)
 ids=d["topk_ids"]
 ids=torch.where(ids<0,torch.full_like(ids,32),ids)
 if dist=="mixed":
  ids[::2]=torch.arange(32,40,dtype=torch.int32,device=dev)
 d["topk_ids"]=((ids+lo)%256).to(torch.int32).contiguous()
 if dist=="skewed":
  # Additional outlier channels, like the board's skewed distribution.
  d["hidden_states"][:,::128]*=8
 d["config"]["local_start"]=lo
 return d

def _mbl_cases():
 cases=list(HYBLO_SELF_TEST)+list(VMS_SELF_TEST_CASES)+list(MBL_VMS_HI_TEST)+list(HYB_SELF_TEST)
 for cs in MBL_VML_TEST.values():
  cases.extend(cs)
 cases.extend((m,1000+m,"normal")for m in MBL_EDGES)
 cases.extend((m,2000+m,"none")for m in MBL_EDGES)
 cases.extend((m,3000+m,"mixed")for m in(1,8,16,32,128,256))
 cases.extend((m,4000+m,"conc")for m in(1,8,16,24,32,128,256,320))
 return tuple(dict.fromkeys(cases))

def _mbl_health(name,state):
 st=state["banks"][0]["k2a33"]["st"]
 if name in("vms","vms_hi"):
  return int(st["vms"]["ctr"][2])==0
 if name in("bf2","w72"):
  return int(state["vml_sc"][name]["err"][0])==0
 return all(bool((f[n:].view(vt)==sv).all())for f,n,vt,sv in state["h_full"])

def _mbl_compare(got,ref,data,state,name,prev=None):
 tot,row=_k2a_rel_err(got,ref)
 repeat=0.0 if prev is None else _k2a_rel_err(got,prev)[0]
 ids=data["topk_ids"]
 lo=data["config"]["local_start"]
 empty=~((ids>=lo)&(ids<lo+32)).any(dim=1)
 zero=bool((got[empty]==0).all())
 ok=(got.dtype==torch.bfloat16 and tuple(got.shape)==tuple(ref.shape)and
     tot<1e-2 and row<5e-2 and repeat<1e-3 and zero and
     bool(torch.isfinite(got.float()).all())and _mbl_health(name,state))
 if not ok:
  raise RuntimeError(f"rel={tot:.3e} row={row:.3e} repeat={repeat:.3e} zero={zero}")

def _mbl_validate(state,weights):
 # A failure removes an engine for the whole bank family, visibly. Fallbacks
 # get the SAME cases before route selection; there is no timed exception path.
 good={k:v is not None for k,v in state["mods"].items()}
 st=state["banks"][0]["k2a33"]["st"]
 for name,need in(("vms","vms"),("vms_hi","vms"),("tiny","hyblo"),("tiny2","hyblo"),("hyb","hyb"),("hyb7","hyb")):
  good[name]=good[name]and need in st
 dev=weights["w13"].device
 for m,seed,dist in _mbl_cases():
  d=_mbl_input(m,seed,dev,dist)
  ref=_k2a_ref_fp32(d,weights)
  seen,noise={},{}
  for name in _mbl_candidates(m):
   if not good[name]:
    continue
   try:
    prev=None
    for _ in range(2):
     got=_mbl_call(name,d,state)
     torch.cuda.synchronize()
     _mbl_compare(got,ref,d,state,name,prev)
     noise[name]=0.0 if prev is None else _k2a_rel_err(got,prev)[0]
     prev=got.clone()
    seen[name]=prev
   except Exception as ex:
    good[name]=False
    _mbl_status(name,False,f"M={m} {dist}: {ex!r}")
  # Native applicable cross-engine gates: HYB7 versus HYB's repeat noise;
  # w72 versus bf2; HYB/HYBLO versus the existing VML/VMS engines.
  for name,other,bound in(("hyb7","hyb",max(2*noise.get("hyb",0.0),1e-6)),
                         ("w72","bf2",2e-3),("hyb","w72",1e-2),
                         ("tiny","vms",1e-2),("tiny2","vms",1e-2)):
   if name in seen and other in seen and good[name]and good[other]:
    err=_k2a_rel_err(seen[name],seen[other])[0]
    if err>bound:
     good[name]=False
     _mbl_status(name,False,f"M={m} {dist}: vs {other} {err:.3e} > {bound:.3e}")
 # Global ids and weight-bank indexing are checked on nonzero AND unaligned
 # starts, all scored shapes. No expanded weight cache survives prepare.
 for lo in(1,32,224):
  for m in(1,8,16,32,128,256):
   d=_mbl_input(m,5000+lo+m,dev,"normal",lo)
   ref=_k2a_ref_fp32(d,weights)
   for name in _mbl_candidates(m):
    if good[name]:
     try:
      got=_mbl_call(name,d,state)
      torch.cuda.synchronize()
      _mbl_compare(got,ref,d,state,name)
     except Exception as ex:
      good[name]=False
      _mbl_status(name,False,f"local_start={lo} M={m}: {ex!r}")
 state["good"]=good
 _mbl_select(state)
 # Exercise capture and repeated fresh-input replays with persistent counters.
 for m in(1,8,16,32,128,256):
  while True:
   d=_mbl_input(m,6000+m,dev)
   name=state["route"][m]
   try:
    warm=torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
     kernel_fn(d,state)
    torch.cuda.current_stream().wait_stream(warm)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     out=kernel_fn(d,state)
    for dist in("normal","none","conc","normal"):
     fresh=_mbl_input(m,7000+m,dev,dist)
     for key in("hidden_states","topk_ids","topk_weights"):
      d[key].copy_(fresh[key])
     graph.replay()
     torch.cuda.synchronize()
     _mbl_compare(out,_k2a_ref_fp32(d,weights),d,state,name)
    break
   except Exception as ex:
    good[name]=False
    _mbl_status(name,False,f"graph M={m}: {ex!r}")
    _mbl_select(state)
 for name in MBL_ARMS:
  _mbl_status(name,good[name],"native cases + FP32 board equation + edges/global ids; selected="+
              str([m for m in(1,8,16,32,128,256)if state["route"][m]==name]))

def _mbl_select(state):
 for m in range(1,321):
  live=next((name for name in _mbl_candidates(m)if state["good"][name]),None)
  if live is None:
   raise RuntimeError(f"FAIL no validated VMS/VML fallback for M={m}")
  state["route"][m]=live

def prepare(weights):
 t0=time.time()
 dev=weights["w13"].device
 nsm=torch.cuda.get_device_properties(dev).multi_processor_count
 if nsm>min(_HYB_SLOTS,_vml_SLOTS):
  raise ValueError("unsupported SM count")
 result={}
 def build():
  try:
   result["mods"]=_mbl_builds()
  except Exception as ex:
   result["error"]=ex
 job=threading.Thread(target=build,name="mbl-build",daemon=True)
 job.start()
 try:
  banks=_mbl_banks(weights,dev)
 finally:
  job.join()
 if "error"in result:
  raise RuntimeError("MBL build orchestration failed")from result["error"]
 mods=result["mods"]
 for bank in banks.values():
  vs=bank["k2a33"]["st"].get("vms")
  if vs is not None:
   vs["mod"],vs["mod_hi"]=mods["vms"],mods["vms_hi"]
 sc={"bf2":_vml_alloc(dev,nsm),"w72":_vml_alloc(dev,nsm)}
 hsc,hfull=_hyb_scratch(dev)
 output=torch.empty((320,4096),dtype=torch.bfloat16,device=dev)
 state={"mods":mods,"banks":banks,"vml_sc":sc,"h_sc":hsc,"h_full":hfull,"nsm":nsm,
        "route":{},"out":{m:output[:m]for m in range(1,321)}}
 _mbl_validate(state,weights)
 elapsed=time.time()-t0
 print(f"[mbl] routes { {m:state['route'][m]for m in(1,8,16,32,128,256)} } prepare={elapsed:.1f}s",flush=True)
 if elapsed>=295:
  raise RuntimeError("MBL prepare exceeded its 295s safety budget")
 return state
'''


def generate():
    MANIFEST.clear()
    src = BASE.read_text(encoding="utf-8")
    parts = ['# okc: problem=muse_moe_w4a16\n# Generated by python -B -I tools/mbl_gen.py; diagnostic, no tuning.\n',
             node_text(src, "THIRD_PARTY_NOTICES"),
             'import hashlib\nimport threading\nimport time\nimport torch\n_K3_BUILD_S={}\nR1_LOCAL_ID_TABLE=True\n']
    parts.append(between(src, 'K2A_E_LOC=32\n', 'V8_ROUTER_ROUTE=', 'K2A helpers'))
    parts.append(between(src, 'K2A31_BITPOS3=', '_GFT_NOTICE=', 'NVFP4 packing'))
    parts.append('K2A33_SW3=18432\n')
    parts.append(node_text(src, '_k2a33_stage_repack'))
    parts.append(between(src, 'VMS_CUDA_SRC=r"""', 'def _vms_prepare(', 'VMS CUDA/build/pack/call'))
    parts.append(node_text(src, 'VMS_SELF_TEST_CASES'))
    parts.append(between(src, '_HYB_NOTICE=r"""', 'def _hyb_prepare(', 'HYB CUDA/build/pack/call'))
    parts.append(between(src, '_HYBLO_NOTICE=r"""', 'def _hyblo_prepare(', 'HYBLO CUDA/build/pack/call'))
    parts.append(between(src, '_vml_E_LOC,_vml_TOPK,_vml_KDIM=', 'def _vml_prepare(', 'VML CUDA/build/pack/call'))
    parts.append(node_text(src, '_k2a36_synth_inputs'))
    vp=node_text(src,'_vms_prepare')
    hi=[n.iter for n in ast.walk(ast.parse(vp)) if isinstance(n,ast.For)
        and isinstance(n.iter,ast.Tuple) and ast.literal_eval(n.iter)[0]==(12,61,'skewed')]
    check(len(hi)==1,'native VMS hi self-test anchor')
    parts.append('MBL_VMS_HI_TEST='+ast.get_source_segment(vp,hi[0])+'\n')
    vl=node_text(src,'_vml_prepare')
    cases=[n.value for n in ast.walk(ast.parse(vl)) if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name)and t.id=='cases'for t in n.targets)]
    check(len(cases)==1,'native VML self-test anchor')
    parts.append('MBL_VML_TEST='+ast.get_source_segment(vl,cases[0])+'\n')
    # Verbatim native host allocation/repack statements, wrapped without reindent.
    pack = between(src, '  ws13=torch.empty((K2A_E_LOC,64,16,K2A33_SW3)', ' st=dict(\n ws13=ws13,ws2=ws2,', 'VML bank repack')
    fields = between(src, ' ws13=ws13,ws2=ws2,\n', ' h=torch.zeros((8*K2A33_M3,2*K2A_D_EXPERT)', 'VML bank fields')
    parts.append('def _mbl_pack_stage(st,weights,dev):\n if True:\n'+pack+' st.update(\n'+fields+' )\n')
    vpack = between(src, '  b13,b2=w["w13_sf"].view(torch.uint8)', '  if VMS_HI_ON and st["vms"].get("mod_hi")is not None:', 'VMS bank/scratch')
    parts.append('def _mbl_pack_vms(st,w,dev):\n mod=_VMS["mod"]\n if True:\n'+vpack)
    # Scope first because the row-construction anchors occur in both HYB versions.
    hp = node_text(src, '_hyb_prepare')
    hpack = between(hp, '  I=H=4096\n', '  _HYB["sc"],_HYB["full"]=_hyb_scratch(dev)\n', 'HYB bank')
    parts.append('def _mbl_pack_hyb(st,w,dev):\n if True:\n'+hpack)
    lp = node_text(src, '_hyblo_prepare')
    lpack = between(lp, '  I=H=4096\n', '  _HYBLO["sc"],_HYBLO["full"]=_hyb_scratch(dev)\n', 'HYBLO bank')
    parts.append('def _mbl_pack_hyblo(st,w,dev):\n if True:\n'+lpack)
    parts.append(ADAPTER)
    out = '\n'.join(parts)
    ast.parse(out)
    compile(out, str(OUT), 'exec')
    check(len(out.encode()) < 1_000_000, 'submission size')
    check('\r' not in out, 'LF')
    check('importlib' not in out and 'sys.modules' not in out, 'no module injection')
    check(out.isascii(), 'ASCII submission')
    check(out.splitlines()[0]=='# okc: problem=muse_moe_w4a16', 'problem marker')
    # Every CUDA literal is byte-identical to the named base (including notices).
    for name in ('VMS_CUDA_SRC','VMS_CPP_SRC','_HYB_MOE_CUDA','_HYB_MOE_CPP','_HYBLO_CUDA','_vml_CUDA_T','_vml_CPP',
                 'THIRD_PARTY_NOTICES','_HYB_NOTICE','_HYBLO_NOTICE'):
        check(node_text(out,name) == node_text(src,name), name+' verbatim')
    table=symtable.symtable(out,str(OUT),'exec')
    known={s.get_name()for s in table.get_symbols()if s.is_assigned()or s.is_imported()or s.is_namespace()}|set(vars(builtins))
    def globals_gate(scope):
        for s in scope.get_symbols():
            check(not(s.is_global()and s.is_referenced()and s.get_name()not in known),
                  ('unbound global',scope.get_name(),s.get_name()))
        for child in scope.get_children(): globals_gate(child)
    globals_gate(table)
    return out


# CPU-only symbolic tensors: execute real repack bodies and real prepare glue,
# checking dimensions and dictionary accesses without allocating 50 GB or CUDA.
class DType:
    def __init__(self,name,size): self.name,self.size=name,size
    def __repr__(self): return self.name


class Tensor:
    def __init__(self,shape=(),dtype=None):
        self.shape=tuple(shape) if isinstance(shape,(list,tuple)) else (shape,)
        self.dtype=dtype or DT['float32']; self.device='cuda'
    def numel(self): return math.prod(self.shape)
    def view(self,*shape):
        if len(shape)==1 and isinstance(shape[0],DType):
            d=shape[0]; s=list(self.shape)
            if s: s[-1]=s[-1]*self.dtype.size//d.size
            return Tensor(s,d)
        return self.reshape(*shape)
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(list,tuple)): shape=shape[0]
        shape=list(shape)
        if -1 in shape:
            i=shape.index(-1); shape[i]=self.numel()//math.prod(s for s in shape if s!=-1)
        check(math.prod(shape)==self.numel(),('reshape',self.shape,shape))
        return Tensor(shape,self.dtype)
    def permute(self,*axes): return Tensor(tuple(self.shape[i]for i in axes),self.dtype)
    def contiguous(self): return self
    def clone(self): return Tensor(self.shape,self.dtype)
    def to(self,dtype=None,**kw): return Tensor(self.shape,dtype if isinstance(dtype,DType) else self.dtype)
    def float(self): return self.to(DT['float32'])
    def long(self): return self.to(DT['int64'])
    def zero_(self): return self
    def fill_(self,v): return self
    def copy_(self,v): check(self.shape==v.shape,('copy',self.shape,v.shape)); return self
    def repeat(self,*reps):
        s=(1,)*(len(reps)-len(self.shape))+self.shape
        return Tensor(tuple(n*r for n,r in zip(s,reps)),self.dtype)
    def all(self,*a,**k): return True
    def any(self,dim=None): return Tensor((self.shape[0],),DT['bool']) if dim is not None else False
    def __bool__(self): return True
    def __int__(self): return 0
    def __float__(self): return 0.
    def tolist(self): return list(range(self.numel()))
    def index_select(self,dim,index):
        s=list(self.shape); s[dim]=index.numel(); return Tensor(s,self.dtype)
    def unsqueeze(self,dim):
        s=list(self.shape); s.insert(dim if dim>=0 else len(s)+1+dim,1); return Tensor(s,self.dtype)
    def flatten(self,start_dim=0,end_dim=-1):
        a=start_dim%len(self.shape); b=end_dim%len(self.shape)
        return Tensor(self.shape[:a]+(math.prod(self.shape[a:b+1]),)+self.shape[b+1:],self.dtype)
    def __getitem__(self,key):
        if isinstance(key,Tensor): return Tensor(key.shape+self.shape[1:],self.dtype)
        if not isinstance(key,tuple): key=(key,)
        keys=list(key)
        if any(k is Ellipsis for k in keys):
            p=next(i for i,k in enumerate(keys)if k is Ellipsis)
            keys[p:p+1]=[slice(None)]*(len(self.shape)-sum(k is not None and k is not Ellipsis for k in keys))
        s=[]; axis=0
        for k in keys:
            if k is None: s.append(1); continue
            n=self.shape[axis]; axis+=1
            if isinstance(k,slice): s.append(len(range(*k.indices(n))))
            elif isinstance(k,Tensor): s.extend(k.shape)
        return Tensor(tuple(s)+self.shape[axis:],self.dtype)
    def __setitem__(self,k,v): pass
    def binary(self,other):
        if not isinstance(other,Tensor): return Tensor(self.shape,self.dtype)
        a,b=list(self.shape),list(other.shape)
        a=[1]*(max(len(a),len(b))-len(a))+a; b=[1]*(len(a)-len(b))+b
        check(all(x==y or x==1 or y==1 for x,y in zip(a,b)),('broadcast',a,b))
        return Tensor(tuple(max(x,y)for x,y in zip(a,b)),self.dtype)
    __add__=__radd__=__sub__=__rsub__=__mul__=__rmul__=__truediv__=__floordiv__=__mod__=binary
    __and__=__or__=__xor__=__lshift__=__rshift__=__ge__=__gt__=__le__=__lt__=__eq__=__ne__=binary
    def __invert__(self): return self
    def __neg__(self): return self
    def __pow__(self,x): return self


DT={n:DType(n,s)for n,s in [('uint8',1),('int8',1),('float8_e4m3fn',1),('int16',2),('bfloat16',2),
                           ('int32',4),('int64',8),('float32',4),('bool',1)]}


class FakeModule:
    def __init__(self,name,calls): self.name,self.calls=name,calls
    def okc_set_pdl(self,v): self.pdl=v
    def vml_cfg(self): return [0,9 if 'w72' in self.name else 8,0,0,0,0,0,0,0,0,0]
    def run(self,*a):
        check(a[14]==a[0].shape[0] and a[14]<=16,('VMS args',a[14]))
        check(a[3].shape[0]==32 and a[4].shape[0]==32,'VMS bank')
        self.calls.append((self.name,a[0].shape[0],a[-1]))
    def okc_run(self,w,sc,x,ids,tw,out,lo,nloc,g):
        check(len(w)==14 and len(sc)==3 and nloc==32 and g==132,'HYB args')
        self.calls.append((self.name,x.shape[0],lo))
    def vml_run(self,t,v):
        check(len(t)==20 and t[3].shape[0]==32 and v[5]==193,'VML args')
        self.calls.append((self.name,v[0],v[1]))


def fake_torch(specs,calls):
    t=types.SimpleNamespace(**DT,Tensor=Tensor)
    t.empty=t.zeros=lambda shape,**kw:Tensor(shape,kw.get('dtype'))
    t.zeros_like=lambda x,**kw:Tensor(x.shape,kw.get('dtype',x.dtype))
    t.full_like=lambda x,v,**kw:Tensor(x.shape,kw.get('dtype',x.dtype))
    t.full=lambda shape,v,**kw:Tensor(shape,kw.get('dtype'))
    t.arange=lambda start,end=None,**kw:Tensor((start if end is None else end-start,),kw.get('dtype',DT['int64']))
    t.tensor=lambda values,**kw:Tensor((len(values),),kw.get('dtype'))
    def cat(xs,dim=0):
        s=list(xs[0].shape); dim%=len(s); s[dim]=sum(x.shape[dim]for x in xs)
        return Tensor(s,xs[0].dtype)
    def stack(xs,dim=0):
        s=list(xs[0].shape); dim=dim if dim>=0 else len(s)+1+dim; s.insert(dim,len(xs)); return Tensor(s,xs[0].dtype)
    t.cat,t.stack=cat,stack
    t.where=lambda c,a,b:a.binary(b)
    t.isfinite=lambda x:x
    class Stream:
        def wait_stream(self,s): pass
    class Graph:
        def replay(self): pass
    t.cuda=types.SimpleNamespace(get_device_properties=lambda d:types.SimpleNamespace(multi_processor_count=132),
        synchronize=lambda:None,Stream=Stream,CUDAGraph=Graph,current_stream=lambda:Stream(),
        stream=lambda s:contextlib.nullcontext(),graph=lambda g:contextlib.nullcontext())
    def build(**kw):
        specs.append(kw)
        fl=kw['extra_cuda_cflags']; name=kw['name']
        if '-DVM_JN=72' in fl: name+='w72'
        return FakeModule(name,calls)
    return t,build


def cpu_dry_run(out):
    specs,calls=[],[]
    torch,build=fake_torch(specs,calls)
    def importer(name,*a,**kw):
        if name=='torch': return torch
        if name=='torch.utils.cpp_extension': return types.SimpleNamespace(load_inline=build)
        return builtins.__import__(name,*a,**kw)
    ns={'__builtins__':dict(vars(builtins),__import__=importer)}
    exec(compile(out,str(OUT),'exec'),ns)
    def inp(m,seed,dev,dist='normal',lo=0):
        return dict(hidden_states=Tensor((m,4096),DT['bfloat16']),topk_ids=Tensor((m,8),DT['int32']),
                    topk_weights=Tensor((m,8)),config=dict(local_start=lo,local_count=32))
    # Numerical work alone is stubbed. Native allocations, repacks, bank views,
    # arm selection, launch argument construction and prepare control flow run.
    ns['_hyb_kperm_src']=lambda:Tensor((256,),DT['int64'])
    ns['_k2a_ref_fp32']=lambda d,w:Tensor(d['hidden_states'].shape,DT['bfloat16'])
    ns['_k2a_rel_err']=lambda a,b:(0.,0.)
    ns['_k2a_synth_inputs']=inp
    ns['_k2a36_synth_inputs']=inp
    weights={k:Tensor(s,DT[d])for k,s,d in (
        ('w13',(256,8192,2048),'uint8'),('w13_sf',(256,8192,256),'float8_e4m3fn'),('w13_gs',(256,),'float32'),
        ('w2',(256,4096,2048),'uint8'),('w2_sf',(256,4096,256),'float8_e4m3fn'),('w2_gs',(256,),'float32'),
        ('post_norm_w',(4096,),'bfloat16'))}
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        st=ns['prepare'](weights)
    check(all(st['good'].values()),log.getvalue())
    check(len(st['banks'])==225 and len(st['route'])==320,'bank/route coverage')
    check([st['route'][m]for m in(1,8,16,32,128,256)]==['tiny','tiny2','vms_hi','bf2','hyb','hyb7'],
          'scored routes')
    check(st['route'][17]==st['route'][24]=='bf2','unsafe VMS edge excluded')
    for lo in(0,1,31,32,127,224):
        for m in range(1,321):
            o=ns['kernel_fn'](inp(m,0,'cuda',lo=lo),st)
            check(o.shape==(m,4096),'output shape')
    for missing in ('tiny','tiny2','vms_hi','hyb7','hyb','w72','vms'):
        old=dict(st['good']); st['good'][missing]=False; ns['_mbl_select'](st)
        check(missing not in st['route'].values(),missing+' fallback')
        st['good']=old
    compare=ns['_mbl_compare']
    def fail_primaries(got,ref,data,state,name,prev=None):
        if name in('tiny','tiny2','vms_hi','hyb','hyb7'):
            raise RuntimeError('injected numeric failure')
        return compare(got,ref,data,state,name,prev)
    ns['_mbl_compare']=fail_primaries
    with contextlib.redirect_stdout(log): ns['_mbl_validate'](st,weights)
    check([st['route'][m]for m in(1,8,16,32,128,256)]==['vms','vms','vms','bf2','w72','w72'],
          'validation failure falls back')
    ns['_mbl_compare']=compare
    # Exercise graph-only failure, then require recapture/validation of fallback.
    old_graph=torch.cuda.graph
    failed=[False]
    def fail_first_graph(g):
        if not failed[0]:
            failed[0]=True
            raise RuntimeError('injected capture failure')
        return old_graph(g)
    torch.cuda.graph=fail_first_graph
    with contextlib.redirect_stdout(log): ns['_mbl_validate'](st,weights)
    check(st['route'][1]=='vms','graph failure recaptures fallback')
    torch.cuda.graph=old_graph
    st['good']={n:n=='bf2'for n in st['good']}; ns['_mbl_select'](st)
    check(set(st['route'].values())=={'bf2'},'all failed -> VML')
    st['good']['bf2']=False
    try: ns['_mbl_select'](st)
    except RuntimeError: pass
    else: raise AssertionError('no validated engine must fail closed')
    check(len(specs)==8,('native build count',len(specs)))
    return dict(status='PASS',banks=225,shapes=320,launches=len(calls),native_builds=len(specs),
                failure_gates=['per-engine routing','numeric failure','graph failure/recapture','no engine -> raise'],
                numeric='NOT RUN: symbolic CPU tensors; no torch/CUDA installed'),specs


def linux_gates(out,specs):
    # Compile/load only; no kernel launches, GPU properties, pod or judge calls.
    import os
    import torch
    from torch.utils.cpp_extension import load_inline
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
    os.environ.setdefault('MAX_JOBS','2')
    results=[]
    for spec in specs:
        record={'name':spec['name'],'flags':spec['extra_cuda_cflags']}
        try:
            mod=load_inline(**spec)
            p=subprocess.run(['cuobjdump','--dump-sass',mod.__file__],capture_output=True,text=True,check=True)
            check('Function :' in p.stdout,'empty SASS')
            resources=subprocess.run(['cuobjdump','--dump-resource-usage',mod.__file__],
                                     capture_output=True,text=True,check=True)
            record.update(status='PASS',sass_sha256=hashlib.sha256(p.stdout.encode()).hexdigest(),
                          functions=p.stdout.count('Function :'),resources=resources.stdout)
        except Exception as ex:
            record.update(status='FAIL',error=repr(ex)[-3000:])
        results.append(record)
    return {'status':'PASS'if all(r['status']=='PASS'for r in results)else'FAIL','builds':results,
            'gpu_numerics':'NOT RUN: compile/SASS only'}


def report(out,dry,linux):
    # Exact line inventory derives from the actual generated artifact.
    lines=out.splitlines()
    def line(prefix): return next(i+1 for i,s in enumerate(lines)if s.startswith(prefix))
    functions={n.name:(n.lineno,n.end_lineno)for n in ast.parse(out).body if isinstance(n,ast.FunctionDef)}
    routes=[(1,'tiny','_HYBLO_FLAGS + tiny',7343),(8,'tiny2','_HYBLO_FLAGS + tiny2',7343),
            (16,'vms_hi','VMS_T16=1 / NS=2 / ER=1',4686),(32,'bf2','VML_VAR=bf2 / flags=193',9767),
            (128,'hyb','HYB_FLAGS / moe_fused<true>',6151),(256,'hyb7','HYB_FLAGS + HYB_F7 / moe_fused<true>',6151)]
    rows=['| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |',
          '|---|---|---|---|---|---|---|',
          '| MBL diagnostic | 1, 8, 16, 32, 128, 256 | Not measured for this board file | No saving claimed | Fresh board timing required; unchanged loops | Process traps: board-to-layer carry | CPU PASS; Linux '+linux['status']+'; GPU/check pending |','',
          f'Generated `{OUT.relative_to(ROOT).as_posix()}`: **{len(out.encode()):,} bytes**, LF, below 1,000,000 bytes.',
          'Kill reasons first: board routing can erase the layer advantage; standalone launches remove the layer\'s PDL context. Diagnostic only; no optimization or microsecond estimate.',
          '', '| scored bs | preferred engine (subject to printed validation) | exact specialization | base source line |',
          '|---|---|---|---|']
    rows += [f'| {m} | {name} | {flags} | kernels/lab/k3_diet_rq128.py:{ln} |'for m,name,flags,ln in routes]
    rows += ['', 'Source discrepancies: VMS_MAXT=16 (:4724), CUDA MAXP=128 (:3962), hard m<=16 check (:4689); M=17..24 uses validated bf2, not an unsafe VMS call. HYB M=49..63 runs moe_fused<false>; OKC_BIGM=64 (:4965). HYBLO_T=(1,8) is a scored-shape whitelist in the layer (:7497); the board extends its native tiny/tiny2 specialization to edges 2..4/5..7.',
             'Bytes/instructions: no changed CUDA hot loop, before == after byte-for-byte (SHA gates). VMS weight unit = 32,768 code + 4,096 scale = 36,864 B; VML stage = 16,384 + 2,048 = 18,432 B per half; HYB WGS copy = 8,192 + 1,024 = 9,216 B per subblock. Instruction delta/thread = 0 by identical CUDA and flags, SASS verification pending. No protocol change: all CTA/thread cleanup, reset, combine loops and launch geometry retained.',
             'Host changes: board global ids pass through native lo arguments; all 256 expert weights are losslessly repacked by the native 32-expert statements, then exposed as 225 local-window views. Scratch and outputs allocated in prepare; timed route performs no host sync or GPU allocation. Native build families overlap with repacking, following the base _pt1 job pattern (:10184); HYBLO tiny/tiny2 remain sequential inside their native builder.',
             'Validation: native HYBLO/VMS/VMS-hi/HYB case tables and VML prepare cases; scored shapes and bucket boundaries; normal/skewed/all-local/no-local/mixed-zero routings; nonzero/unaligned starts; native applicable cross-engine error bounds; repeat and fresh-input graph replay checks, including recapture of a fallback. Independent FP32 equation copied from the layer agrees structurally with the board reference; no reference import. Legacy K2A-vs-engine comparisons are inapplicable without K2A; FP32/repeat/error/guard checks are kept. Every engine prints LIVE/FAIL and failures select validated VMS/VML; if none passes prepare raises.',
             'Limitations: CPU dry run is symbolic, not a numerical proof. The 256-expert repack cost and 300s prepare limit remain unmeasured. Prior layer prepare took 201.2 s with HYBLO reported as the 198-202 s long pole (MEASURED, notes/lab.md:1183). Linux compile/SASS status: '+linux['status']+'. H200 correctness/timing and judge check NOT RUN.',
             '', 'Exact changed/new host lines in mb_layer.py (all remaining engine bodies are verbatim):']
    for n,(start,end) in functions.items():
        if n.startswith('_mbl_')or n in('prepare','kernel_fn'):
            rows.append(f'- `{n}`: `{OUT.relative_to(ROOT).as_posix()}:{start}-{end}`.')
    rows += ['', 'Copied engine specialization anchors in mb_layer.py:',
             f'- HYBLO tiny / tiny2: `_HYBLO_KEYS` :{line("_HYBLO_KEYS=")}, native build :{line("def _hyblo_build(")}.',
             f'- VMS hi: native build :{line("def _vms_build(")}; launcher :{line("void run(torch::Tensor x,")}.',
             f'- VML bf2 / w72: `_VML_DEFS` :{line("_VML_DEFS=")}, `vm_main` :{line("__global__ void __launch_bounds__(NTH, 1) vm_main")}, `vm_fin` :{line("__global__ void __launch_bounds__(256) vm_fin")}.',
             f'- HYB mod / mod7: `HYB_FLAGS` :{line("HYB_FLAGS=")}, native build :{line("def _hyb_build(")}, `moe_fused<true>` launcher :{line("if (p.M >= OKC_BIGM) cudaLaunchKernelEx")}.']
    rows += ['', 'Generator + CPU gates: `python -B -I tools/mbl_gen.py`.',
             'Linux compile/SASS runner (this generator is also a self-contained LF bash script):',
             '```bash', 'MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" bash "<repo>/tools/mbl_gen.py"', '```',
             'The three-file allowlist prevents a separate .sh/.json: the shell entry is in mbl_gen.py and gate JSON is embedded here. No WSL invocation was attempted in this sandbox.',
             '', '```json',json.dumps({'cpu':dry,'linux':linux,'source_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest()},indent=2),'```','']
    return '\n'.join(rows)


def main():
    out=generate()
    dry,specs=cpu_dry_run(out)
    linux=linux_gates(out,specs)if '--linux-gates'in sys.argv else {'status':'NOT RUN'}
    OUT.write_bytes(out.encode())
    REPORT.write_bytes(report(out,dry,linux).encode())
    print(json.dumps({'file':str(OUT),'bytes':len(out.encode()),'cpu':dry,'linux':linux['status']}))
    if linux['status']=='FAIL': raise SystemExit(1)


if __name__=='__main__':
    main()

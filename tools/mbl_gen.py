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
import os
import pathlib
import subprocess
import sys
import symtable
import threading
import time
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
MBL_ARMS=("tiny","tiny2","vms_hi","vms","bf2","hyb","hyb7")
MBL_VALIDATE_S=40.0

def _mbl_deadline(deadline,label):
 if time.monotonic()>=deadline:
  raise TimeoutError(f"MBL {label} safety budget exceeded")

def _mbl_stage(name,t0):
 print(f"[mbl] prepare stage {name} {time.monotonic()-t0:.3f}",flush=True)
 _mbl_deadline(t0+295,"prepare 295s")

def _mbl_load_inline(**kw):
 # Local wrapper, never an edit to torch's loaded module. Native args unchanged.
 from torch.utils.cpp_extension import load_inline
 tag=kw["name"]
 if tag.startswith("k3_vmshi_"):
  name="vms_hi"
 elif tag.startswith("k3_vms_"):
  name="vms"
 elif tag.startswith("vml_moe_"):
  name="bf2"
 elif tag.startswith("hyblo_"):
  name=tag.split("_")[1]
 else:
  name="hyb7"if all(f in kw["extra_cuda_cflags"]for f in HYB_F7)else"hyb"
 t0=time.monotonic()
 print(f"[mbl] build start {name}",flush=True)
 try:
  return load_inline(**kw)
 finally:
  elapsed=time.monotonic()-t0
  _K3_BUILD_S[name]=elapsed
  print(f"[mbl] build {name} {elapsed:.3f}",flush=True)

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
 return (("hyb7","hyb","bf2")if m>=HYB_T7 else("hyb","bf2"))

def _mbl_builds(deadline):
 # _pt1 layout: start all family jobs before waiting. VMS/HYB retain their
 # native child threads; each HYBLO variant now has its own family job.
 def one(name,fn):
  try:
   fn()
  except Exception as ex:
   _mbl_status(name,False,"build "+repr(ex)[:600])
 work=[("vms",_vms_build),("vml",_vml_build_job),("hyb",_hyb_build)]
 work.extend((k,lambda entry=(k,m,var):_hyblo_build((entry,)))for k,m,var in _HYBLO_KEYS)
 jobs=[threading.Thread(target=one,args=(name,fn),name=f"pt1-{name}",daemon=True)for name,fn in work]
 for job in jobs:
  job.start()
 for job in jobs:
  job.join(max(0.0,deadline-time.monotonic()))
  _mbl_deadline(deadline,"prepare 295s build wait")
 mods={"vms":_VMS["mod"],"vms_hi":_VMS["mod_hi"],"hyb":_HYB["mod"],"hyb7":_HYB["mod7"],
       "tiny":_HYBLO["mods"].get("tiny"),"tiny2":_HYBLO["mods"].get("tiny2"),
       "bf2":_VML["mods"].get("bf2")}
 for name,mod in mods.items():
  if mod is None:
   _mbl_status(name,False,"module unavailable")
  elif name in("tiny","tiny2","hyb","hyb7"):
   mod.okc_set_pdl(int(HYBLO_PDL if name.startswith("tiny")else HYB_PDL))
  elif name=="bf2":
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
 # Preserve the lead's scored-shard patch: repack experts 0..31 only.
 # Other local_start values use the independent FP32 equation in kernel_fn.
 E=weights["w13"].shape[0]
 if E!=256:
  raise ValueError("board weights must contain 256 global experts")
 chunks=[]
 keys=("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")
 for lo in (0,):
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
  global_h={j:torch.cat([b["hyb"][j]for b in chunks]).view(32*len(chunks),-1)for j in(2,5,12,13)}
 if have_tiny:
  global_t={j:torch.cat([b["hyblo"][j]for b in chunks]).view(32*len(chunks),-1)for j in(0,1,3,4,8,9,10,11)}
 banks={}
 for lo in (0,):
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
 elif name=="bf2":
  rep=_k2a_bucket_of(m)
  _vml_call(mod,st,state["vml_sc"][name],x,ids,tw,ls=lo,out=out,
  flags=VML_FLAGS.get(rep,193),cost=VML_COST.get(rep,VML_COST[32]))
 else:
  mod.okc_run(st["hyblo"]if name.startswith("tiny")else st["hyb"],state["h_sc"],x,ids,tw,out,lo,32,state["nsm"])
 return out

def kernel_fn(data,state):
 m=data["hidden_states"].shape[0]
 if int(data["config"]["local_start"])!=0:
  return _k2a_ref_fp32(data,state["w_raw"])
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
 # Fixed 24-case cap: every scored engine, empty/fat jobs, and route boundaries.
 cases=[(m,1000+m,dist)for dist in("normal","none","conc")for m in(1,8,16,32,128,256)]
 cases.extend((m,2000+m,dist)for m,dist in((17,"skewed"),(24,"mixed"),(48,"skewed"),
                                                       (49,"mixed"),(193,"skewed"),(320,"mixed")))
 return tuple(cases)

def _mbl_health(name,state):
 st=state["banks"][0]["k2a33"]["st"]
 if name in("vms","vms_hi"):
  return int(st["vms"]["ctr"][2])==0
 if name=="bf2":
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
 t0=time.monotonic()
 deadline=min(t0+MBL_VALIDATE_S,state["prepare_deadline"])
 good={k:v is not None for k,v in state["mods"].items()}
 st=state["banks"][0]["k2a33"]["st"]
 for name,need in(("vms","vms"),("vms_hi","vms"),("tiny","hyblo"),("tiny2","hyblo"),("hyb","hyb"),("hyb7","hyb")):
  good[name]=good[name]and need in st
 dev=weights["w13"].device
 for m,seed,dist in _mbl_cases():
  _mbl_deadline(deadline,"validation 40s")
  print(f"[mbl] validate case M={m} {dist} {time.monotonic()-t0:.3f}",flush=True)
  d=_mbl_input(m,seed,dev,dist)
  ref=_k2a_ref_fp32(d,weights)
  _mbl_deadline(deadline,"validation 40s")
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
   _mbl_deadline(deadline,"validation 40s")
  # Native applicable cross-engine gates: HYB7 versus HYB's repeat noise;
  # HYB/HYBLO versus the retained VML/VMS engines.
  for name,other,bound in(("hyb7","hyb",max(2*noise.get("hyb",0.0),1e-6)),
                         ("hyb","bf2",1e-2),
                         ("tiny","vms",1e-2),("tiny2","vms",1e-2)):
   if name in seen and other in seen and good[name]and good[other]:
    err=_k2a_rel_err(seen[name],seen[other])[0]
    if err>bound:
     good[name]=False
     _mbl_status(name,False,f"M={m} {dist}: vs {other} {err:.3e} > {bound:.3e}")
 # Nonzero-start native-engine tests stay skipped: kernel_fn uses w_raw/FP32.
 state["good"]=good
 _mbl_select(state)
 # Exercise capture and repeated fresh-input replays with persistent counters.
 for m in(1,8,16,32,128,256):
  while True:
   _mbl_deadline(deadline,"validation 40s")
   d=_mbl_input(m,6000+m,dev)
   name=state["route"][m]
   print(f"[mbl] validate graph M={m} {name} {time.monotonic()-t0:.3f}",flush=True)
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
     _mbl_deadline(deadline,"validation 40s")
     fresh=_mbl_input(m,7000+m,dev,dist)
     for key in("hidden_states","topk_ids","topk_weights"):
      d[key].copy_(fresh[key])
     graph.replay()
     torch.cuda.synchronize()
     _mbl_compare(out,_k2a_ref_fp32(d,weights),d,state,name)
     _mbl_deadline(deadline,"validation 40s")
    break
   except TimeoutError:
    raise
   except Exception as ex:
    good[name]=False
    _mbl_status(name,False,f"graph M={m}: {ex!r}")
    _mbl_select(state)
 for name in MBL_ARMS:
  _mbl_status(name,good[name],"24 capped cases + FP32 board equation + graph replays; selected="+
              str([m for m in(1,8,16,32,128,256)if state["route"][m]==name]))
 _mbl_deadline(deadline,"validation 40s")
 print(f"[mbl] validation {time.monotonic()-t0:.3f}",flush=True)

def _mbl_select(state):
 for m in range(1,321):
  live=next((name for name in _mbl_candidates(m)if state["good"][name]),None)
  if live is None:
   raise RuntimeError(f"FAIL no validated VMS/VML fallback for M={m}")
  state["route"][m]=live

def prepare(weights):
 t0=time.monotonic()
 _mbl_stage("begin",t0)
 dev=weights["w13"].device
 nsm=torch.cuda.get_device_properties(dev).multi_processor_count
 if nsm>min(_HYB_SLOTS,_vml_SLOTS):
  raise ValueError("unsupported SM count")
 # Resolve torch's extension imports once, before any compiler thread starts.
 from torch.utils.cpp_extension import load_inline
 os.environ.setdefault("MAX_JOBS","2")
 _mbl_stage("builds_start",t0)
 result={}
 def build():
  try:
   result["mods"]=_mbl_builds(t0+295)
  except Exception as ex:
   result["error"]=ex
 job=threading.Thread(target=build,name="mbl-build",daemon=True)
 job.start()
 _mbl_stage("pack_start",t0)
 banks=_mbl_banks(weights,dev)
 torch.cuda.synchronize()
 _mbl_stage("pack_done_builds_wait",t0)
 job.join(max(0.0,t0+295-time.monotonic()))
 _mbl_stage("builds_done",t0)
 if "error"in result:
  raise RuntimeError("MBL build orchestration failed")from result["error"]
 mods=result["mods"]
 for bank in banks.values():
  vs=bank["k2a33"]["st"].get("vms")
  if vs is not None:
   vs["mod"],vs["mod_hi"]=mods["vms"],mods["vms_hi"]
 _mbl_stage("scratch_start",t0)
 sc={"bf2":_vml_alloc(dev,nsm)}
 hsc,hfull=_hyb_scratch(dev)
 output=torch.empty((320,4096),dtype=torch.bfloat16,device=dev)
 state={"w_raw":weights,"prepare_deadline":t0+295,"mods":mods,"banks":banks,"vml_sc":sc,"h_sc":hsc,"h_full":hfull,"nsm":nsm,
        "route":{},"out":{m:output[:m]for m in range(1,321)}}
 torch.cuda.synchronize()
 _mbl_stage("validation_start",t0)
 _mbl_validate(state,weights)
 _mbl_stage("validation_done",t0)
 elapsed=time.monotonic()-t0
 print(f"[mbl] routes { {m:state['route'][m]for m in(1,8,16,32,128,256)} } prepare={elapsed:.1f}s",flush=True)
 if elapsed>=295:
  raise RuntimeError("MBL prepare exceeded its 295s safety budget")
 _mbl_stage("done",t0)
 return state
'''


def generate():
    MANIFEST.clear()
    src = BASE.read_text(encoding="utf-8")
    parts = ['# okc: problem=muse_moe_w4a16\n# Generated by python -B -I tools/mbl_gen.py; diagnostic, no tuning.\n',
             node_text(src, "THIRD_PARTY_NOTICES"),
             'import hashlib\nimport os\nimport threading\nimport time\nimport torch\n_K3_BUILD_S={}\nR1_LOCAL_ID_TABLE=True\n']
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
    # Only build orchestration changes inside the verbatim engine sections.
    # Exact native load_inline kwargs/flags and all CUDA literals stay intact.
    old=' from torch.utils.cpp_extension import load_inline\n'
    for name in ('_vms_build','_hyb_build','_hyblo_build','_vml_build'):
        before=node_text(out,name)
        check(before.count(old)==1,(name,'native build import'))
        after=before.replace(old,' load_inline=_mbl_load_inline\n')
        if name=='_hyblo_build':
            after=after.replace('def _hyblo_build():','def _hyblo_build(keys):')
            anchor=' for k,_,var in _HYBLO_KEYS: # one nvcc at a time (R-hovl-1: parallel extra builds froze a container)'
            check(after.count(anchor)==1,'HYBLO serial loop anchor')
            after=after.replace(anchor,' for k,_,var in keys: # one variant per _pt1-style build job')
        check(out.count(before)==1,(name,'unique replacement'))
        out=out.replace(before,after)
    check(out.count("VML_BVAR={256:'w72',128:'w72'}")==1,'native VML routing anchor')
    out=out.replace("VML_BVAR={256:'w72',128:'w72'}","VML_BVAR={} # MBL: HYB handles these shapes; bf2 is the validated fallback")
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


def fake_torch(specs,calls,barrier=None):
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
        if barrier is not None:
            barrier.wait(timeout=10) # fails if even one retained module is serialized
        specs.append(kw)
        fl=kw['extra_cuda_cflags']; name=kw['name']
        if '-DVM_JN=72' in fl: name+='w72'
        return FakeModule(name,calls)
    return t,build


def cpu_dry_run(out):
    specs,calls=[],[]
    torch,build=fake_torch(specs,calls,threading.Barrier(7))
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
    references=[]
    def reference(d,w):
        references.append((d['config']['local_start'],w))
        return Tensor(d['hidden_states'].shape,DT['bfloat16'])
    ns['_k2a_ref_fp32']=reference
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
    check(set(st['banks'])=={0} and len(st['route'])==320,'lead bank/route coverage')
    check(st['w_raw'] is weights,'lead raw-weight handle')
    check(all(lo==0 for lo,w in references),'nonzero-start engine validation stays skipped')
    check(len(ns['_mbl_cases']())==24,'validation case cap')
    for stage in ('begin','builds_start','pack_start','pack_done_builds_wait','builds_done',
                  'scratch_start','validation_start','validation_done','done'):
        check('[mbl] prepare stage '+stage+' 'in log.getvalue(),('stage log',stage))
    for name in ns['MBL_ARMS']:
        check('[mbl] build '+name+' 'in log.getvalue(),('build completion log',name))
    check([st['route'][m]for m in(1,8,16,32,128,256)]==['tiny','tiny2','vms_hi','bf2','hyb','hyb7'],
          'scored routes')
    check(st['route'][17]==st['route'][24]=='bf2','unsafe VMS edge excluded')
    for lo in(0,1,31,32,127,224):
        native_count=len(calls)
        ref_count=len(references)
        for m in range(1,321):
            o=ns['kernel_fn'](inp(m,0,'cuda',lo=lo),st)
            check(o.shape==(m,4096),'output shape')
        if lo:
            check(len(calls)==native_count,'nonzero-start never launches native engine')
            check(len(references)-ref_count==320 and all(w is weights for _,w in references[ref_count:]),
                  'nonzero-start FP32 uses w_raw')
    for missing in ('tiny','tiny2','vms_hi','hyb7','hyb','vms'):
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
    check([st['route'][m]for m in(1,8,16,32,128,256)]==['vms','vms','vms','bf2','bf2','bf2'],
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
    # A slow reference must abort validation, never silently accept untested arms.
    real_time=ns['time']; ticks=[0.0]
    ns['time']=types.SimpleNamespace(monotonic=lambda:ticks[0],time=real_time.time)
    def slow_ref(d,w):
        ticks[0]=41.0
        return reference(d,w)
    ns['_k2a_ref_fp32']=slow_ref
    with contextlib.redirect_stdout(log):
        try: ns['_mbl_validate'](st,weights)
        except TimeoutError as ex: check('validation 40s'in str(ex),'validation timeout label')
        else: raise AssertionError('validation deadline must fail closed')
        try: ns['_mbl_stage']('timeout_probe',ticks[0]-295)
        except TimeoutError: pass
        else: raise AssertionError('295s prepare guard')
    ns['time']=real_time; ns['_k2a_ref_fp32']=reference
    check(len(specs)==7,('native build count',len(specs)))
    check(len({s['name']for s in specs})==7,'unique build/cache names')
    # Oracle: execute the REAL base build functions with a symbolic torch loader.
    # Compare all native compile arguments, including flags/functions/source/name.
    base_specs=[]
    base_torch,base_build=fake_torch(base_specs,[])
    def base_importer(name,*a,**kw):
        if name=='torch': return base_torch
        if name=='torch.utils.cpp_extension': return types.SimpleNamespace(load_inline=base_build)
        return builtins.__import__(name,*a,**kw)
    oracle=dict(ns,torch=base_torch,__builtins__=dict(vars(builtins),__import__=base_importer))
    oracle.update(_VMS={},_HYB={},_HYBLO={'mods':{},'err':{},'secs':{}},_VML={'mods':{}},_K3_BUILD_S={})
    src=BASE.read_text(encoding='utf-8')
    for name in ('VML_BVAR','_vms_build','_hyb_build','_hyblo_build','_vml_build','_vml_build_job'):
        exec(compile(node_text(src,name),str(BASE),'exec'),oracle)
    for name in ('_vms_build','_hyb_build','_hyblo_build','_vml_build_job'): oracle[name]()
    base_by_name={s['name']:s for s in base_specs}
    check(len(base_specs)==8,'real-base build oracle count')
    for spec in specs:
        check(spec==base_by_name[spec['name']],('native build args unchanged',spec['name']))
    removed=[s for s in base_specs if s['name']not in {s['name']for s in specs}]
    check(len(removed)==1 and '-DVM_JN=72'in removed[0]['extra_cuda_cflags'],'only w72 dropped')
    return dict(status='PASS',banks=1,shapes=320,launches=len(calls),native_builds=len(specs),
                concurrent_build_gate='PASS: all seven load_inline calls reached one barrier',
                native_build_args='PASS: seven retained specs identical to real base builders; only w72 removed',
                eager_cases=24,nonzero_start_fp32_cases=1600,
                failure_gates=['per-engine routing','numeric failure','graph failure/recapture','no engine -> raise',
                               'validation 40s fail closed','prepare 295s guard'],
                numeric='NOT RUN: symbolic CPU tensors; no torch/CUDA installed'),specs


def linux_gates(out,specs):
    # Compile/load only; no kernel launches, GPU properties, pod or judge calls.
    import concurrent.futures
    import tempfile
    from torch.utils.cpp_extension import load_inline
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
    os.environ['MAX_JOBS']='2'
    os.environ['CCACHE_DISABLE']='1'
    os.environ['SCCACHE_DISABLE']='1'
    # Fresh, retained directories: no cache hits and no deletions. The lead's
    # Linux run owns these compiler outputs; the sandbox never runs this path.
    root=pathlib.Path(tempfile.mkdtemp(prefix='mbl-cold-'))
    for spec in specs: (root/spec['name']).mkdir()
    def one(spec):
        record={'name':spec['name'],'flags':spec['extra_cuda_cflags']}
        t0=time.monotonic()
        try:
            mod=load_inline(**dict(spec,build_directory=str(root/spec['name'])))
            record.update(status='PASS',module=mod.__file__)
        except Exception as ex:
            record.update(status='FAIL',error=repr(ex)[-3000:])
        record['build_wall_s']=time.monotonic()-t0
        print(f"[mbl] cold build {record['name']} {record['build_wall_s']:.3f} {record['status']}",flush=True)
        return record
    results=[]
    t0=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=7,thread_name_prefix='pt1-mbl')as pool:
        futures=[pool.submit(one,spec)for spec in specs]
        for future in concurrent.futures.as_completed(futures): results.append(future.result())
    wall=time.monotonic()-t0
    # SASS inspection is outside the measured parallel cold-build interval.
    for record in results:
        if record['status']!='PASS': continue
        try:
            p=subprocess.run(['cuobjdump','--dump-sass',record['module']],capture_output=True,text=True,check=True)
            check('Function :' in p.stdout,'empty SASS')
            resources=subprocess.run(['cuobjdump','--dump-resource-usage',record['module']],
                                     capture_output=True,text=True,check=True)
            record.update(status='PASS',sass_sha256=hashlib.sha256(p.stdout.encode()).hexdigest(),
                          functions=p.stdout.count('Function :'),resources=resources.stdout)
        except Exception as ex:
            record.update(status='FAIL',error=repr(ex)[-3000:])
    return {'status':'PASS'if all(r['status']=='PASS'for r in results)else'FAIL','builds':results,
            'parallel_cold_build_wall_s_MEASURED':wall,'max_jobs':2,'native_jobs':7,
            'cold_build_directory':str(root),'compiler_cache_disabled':True,
            'gpu_numerics':'NOT RUN: compile/SASS only'}


def report(out,dry,linux,changed):
    # Append-only fix record: preserve the lead's previous report and gate JSON.
    nodes={n.name:(n.lineno,n.end_lineno)for n in ast.parse(out).body if isinstance(n,ast.FunctionDef)}
    own=pathlib.Path(__file__).read_text(encoding='utf-8')
    own_nodes={n.name:(n.lineno,n.end_lineno)for n in ast.parse(own).body if isinstance(n,ast.FunctionDef)}
    def location(name):
        a,b=nodes[name]
        return f'kernels/muse_moe_w4a16/mb_layer.py:{a}-{b}'
    def anchor(text):
        return next(i+1 for i,line in enumerate(out.splitlines())if text in line)
    rows=['', '## Fix log - MBL-FIX', '',
          '| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |',
          '|---|---|---|---|---|---|---|',
          '| Parallel prepare / capped validation | All six scored bs | >840,000,000 without an [mbl] line; two rc -9 timeouts (task evidence) | Unmeasured; no kernel-time claim | Host build join precedes validation and return; PDL cannot hide it | R-hovl-1 compiler-memory caution; R-mg-6 process traps | CPU '+dry['status']+'; current Linux '+linux['status']+' |', '',
          'Kill reasons first: (1) most builds were already parallel: VMS and HYB each had a child thread, VML built bf2/w72 together; only HYBLO tiny -> tiny2 was serialized. (2) compiler contention could erase the gain: remove unrouted w72, retain at most seven simultaneous native builds, default MAX_JOBS=2 (matching the prior local gate). The base layer itself does not set MAX_JOBS. Cold Linux wall time is still required.',
          'Accounting (EST, not a timing result): old build critical path = max(VMS, VMS-hi, bf2, w72, HYB, HYB7, tiny + tiny2); new = max(VMS, VMS-hi, bf2, HYB, HYB7, tiny, tiny2). Potential chain saving is min(tiny, tiny2) only when HYBLO remains the long pole; contention invalidates a numeric prediction. MEASURED comparator: layer prepare 201.2 s, HYBLO 198-202 s (notes/lab.md:1183). No HBM/clock conversion applies to host compilation.',
          'The existing four family jobs were not eight serial builds. Lack of [mbl] output also covered packing and unbounded joins; root cause is not proven by the old log. New start/completion and stage logs expose that distinction.',
          'Lead patch preserved in the generator: experts 0..31 only; one bank at 0; local_start != 0 calls _k2a_ref_fp32(data,state["w_raw"]); nonzero-start engine tests skipped. No new state accessors: k2a33/st and native module handles are retained.',
          'Validation: 24 eager cases (six scored shapes x normal/none/conc plus six boundary/mixed/skewed cases), all candidate engines still checked against FP32 and repeats; six scored graph checks retain four fresh-input replays and fallback recapture. Cooperative checks fail closed at 40 s and at the 295 s prepare deadline. They cannot preempt a single stuck CUDA call; successful <=40 s validation remains unmeasured.',
          'Native builds: _pt1-style named daemon family threads with start-all-before-join; native VMS/HYB child threads unchanged; two one-variant HYBLO jobs; bf2 only in VML. All seven actual load_inline argument dictionaries match the real base builders, including native module/cache names, sources, flags and exported functions.',
          'Bytes/instructions before -> after (EST static accounting, unchanged source): VMS unit 32,768+4,096=36,864 -> 36,864 B; VML half-stage 16,384+2,048=18,432 -> 18,432 B; HYB WGS subblock 8,192+1,024=9,216 -> 9,216 B. Instruction delta/thread = 0 by identical CUDA and retained flags; absolute SASS counts are not measured in this fix. No hot loop or GPU protocol change: all post-wait cleanup/reset/combine loops and every CTA/thread range stay identical.',
          '', '| scored bs | preferred specialization (unchanged; subject to validation) | exact generated route/build/launch anchor |',
          '|---|---|---|']
    routes=[(1,'HYBLO tiny / _HYBLO_FLAGS + tiny','_HYBLO_KEYS='),
            (8,'HYBLO tiny2 / _HYBLO_FLAGS + tiny2','_HYBLO_KEYS='),
            (16,'VMS hi / VMS_T16=1, NS=2, ER=1',' fl_hi='),
            (32,'VML bf2 / vm_main + vm_fin, flags=193','VML_VAR='),
            (128,'HYB / moe_fused<true>, HYB_FLAGS','HYB_FLAGS='),
            (256,'HYB7 / moe_fused<true>, HYB_FLAGS + HYB_F7','HYB_F7=')]
    for m,label,text in routes:
        rows.append(f'| {m} | {label} | mb_layer.py:{anchor(text)}; {location("_mbl_candidates")}; {location("_mbl_call")} |')
    rows+=['', 'Per-bit changed host lines (inclusive; all CUDA literals are byte-identical to the base):']
    groups={
        'Build concurrency / logs':('_vms_build','_hyb_build','_hyblo_build','_vml_build','_mbl_load_inline','_mbl_builds'),
        'Remove w72 route / scratch / comparison':('_mbl_candidates','_mbl_call','_mbl_health','_mbl_validate','prepare'),
        'Preserve lead expert window':('_mbl_banks','kernel_fn','prepare'),
        'Validation cap / deadlines / stage logs':('_mbl_cases','_mbl_validate','_mbl_deadline','_mbl_stage','prepare')}
    for bit,names in groups.items():
        rows.append('- '+bit+': '+', '.join('`'+location(n)+'`'for n in names)+'.')
    rows.append(f'- VML route table: `kernels/muse_moe_w4a16/mb_layer.py:{anchor("VML_BVAR=")}`.')
    for name in ('generate','fake_torch','cpu_dry_run','linux_gates','report','main'):
        a,b=own_nodes[name]
        rows.append(f'- Generator/gate `{name}`: `tools/mbl_gen.py:{a}-{b}`.')
    rows+=['', 'CPU generator, source/flags gates and symbolic dry run: PASS. The symbolic tensor tests execute native pack/allocation bodies, prepare, bank and arm selection; they are not a numerical proof. A seven-party barrier proves all seven native builds enter before any finishes. The real base build functions serve as the compile-argument oracle. Dry run also checks 1,600 nonzero-start FP32 calls, numeric/graph fallback, validation timeout and prepare timeout.',
           f'Generated size: {len(out.encode()):,} bytes, LF. Check-only command: `python -B -I tools/mbl_gen.py --check`.',
           'Linux compile/SASS and parallel cold-build wall time for THIS FIX: '+linux['status']+'. The earlier PASS above belongs to the previous wrapper; it is not evidence for this fix. WSL/NVCC is inaccessible in this sandbox per task instructions; no GPU, pod or judge job was started.',
           'The LF bash entry remains in tools/mbl_gen.py and adds cuobjdump to PATH. On Linux it uses seven compiler threads, MAX_JOBS=2, fresh retained build directories, and disabled compiler caches; wall timing ends before SASS inspection. The strict three-file allowlist precludes a separate script/JSON sidecar: result JSON is appended here, preserving the established runner/report arrangement.',
           '```bash',
           'MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" bash "<repo>/tools/mbl_gen.py"',
           '```', '', '```json', json.dumps({'cpu':dry,'linux':linux,'generated_changed_line_ranges':changed,
                'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
                'generated_sha256':hashlib.sha256(out.encode()).hexdigest()},indent=2), '```', '']
    return '\n'.join(rows)


def main():
    import difflib
    check(set(sys.argv[1:])<={'--check','--linux-gates'},'supported flags: --check, --linux-gates')
    out=generate()
    dry,specs=cpu_dry_run(out)
    old=OUT.read_text(encoding='utf-8')
    changed=[{'before':[i+1,j],'after':[a+1,b]}for tag,i,j,a,b in
             difflib.SequenceMatcher(None,old.splitlines(),out.splitlines(),autojunk=False).get_opcodes()if tag!='equal']
    if '--check'in sys.argv:
        check(out==old,'generated artifact must match generator')
    linux={'status':'NOT RUN','reason':'WSL/NVCC unavailable in this sandbox; lead runs the LF bash entry',
           'parallel_cold_build_wall_s_MEASURED':None}
    if '--linux-gates'in sys.argv:
        try: linux=linux_gates(out,specs)
        except Exception as ex: linux={'status':'FAIL','error':repr(ex)}
    if '--check'not in sys.argv or '--linux-gates'in sys.argv:
        if '--check'not in sys.argv: OUT.write_bytes(out.encode())
        with REPORT.open('ab')as handle: handle.write(report(out,dry,linux,changed).encode())
    print(json.dumps({'file':str(OUT),'bytes':len(out.encode()),'cpu':dry,'linux':linux['status']}))
    if linux['status']=='FAIL': raise SystemExit(1)


if __name__=='__main__':
    main()

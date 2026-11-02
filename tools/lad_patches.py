#!/usr/bin/env python3
"""Generate the timing-only HYB ladder; no GPU work or edits to the base."""
import ast, difflib, hashlib, importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'kernels/muse_moe_w4a16/mb_layer.py'
DEST=ROOT/'kernels/muse_moe_w4a16/mb_layer_lad.py'
spec=importlib.util.spec_from_file_location('bt4_patch',ROOT/'tools/bt4_patches.py')
bt4=importlib.util.module_from_spec(spec); spec.loader.exec_module(bt4)
environment=bt4.environment
# Reuse the exact BT4 H4 edit list and L2R's exact unit address expression.
L2_EDITS=[
 ('uint32_t zero;\nconst __nv_bfloat16* x;', 'uint32_t zero; const int* l2r_mode;\nconst __nv_bfloat16* x;'),
 ('p.zero = 0;\np.x =', 'p.zero = 0;\np.l2r_mode = (const int*)w[8].data_ptr();\np.x ='),
 ('const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;',
  'const size_t l2r_unit = (size_t)(((S.jE[j] << ubl) + ub) * 16 + st);\nconst unsigned char* src = base + ((*p.l2r_mode & 1) ? l2r_unit % 112 : l2r_unit) * (size_t)UNIT_BYTES;'),
]
G_EDITS=[
 ('if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }',
  'if (!pb && !(OKC_WGSX & 8) && !(*p.l2r_mode & 4)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }'),
 ('wgs_gather(p, S);\nDBG(7);', 'if (!(*p.l2r_mode & 4)) wgs_gather(p, S);\nDBG(7);'),
]
# BT4 retains its own bit-4 snapshot; ladder bit 1 is translated on host.
EDITS=L2_EDITS+bt4.HYB_EDITS+G_EDITS
EXT=r'''
# LAD_BEGIN -- timing probes; output of nonzero arms is invalid.
_LAD_EDITS = @EDITS@
_lad_original_builds = _mbl_builds

def lad_sources(engine, instrument=True):
 if engine not in ("hyb", "hyb7"): raise ValueError(engine)
 src = _HYB_MOE_CUDA
 if instrument:
  for old,new in _LAD_EDITS:
   assert src.count(old)==1, (src.count(old), old[:80])
   src=src.replace(old,new,1)
 cpp=_HYB_MOE_CPP + ("\nvoid bt4_set_mask(int64_t mask);\n" if instrument else "")
 return src,cpp,list(HYB_FLAGS)+(list(HYB_F7) if engine=="hyb7" else [])

def _mbl_builds(deadline):
 mods=_lad_original_builds(deadline)
 for engine in ("hyb","hyb7"):
  _mbl_deadline(deadline,"LAD instrumented build")
  src,cpp,flags=lad_sources(engine)
  tag=hashlib.sha1((cpp+src+" ".join(flags)).encode()).hexdigest()[:12]
  mod=_mbl_load_inline(name="lad_"+engine+"_"+tag,cpp_sources=cpp,cuda_sources=src,
                       functions=["okc_run","okc_set_pdl","bt4_set_mask"],extra_cuda_cflags=flags,verbose=False)
  mod.bt4_set_mask(0); mod.okc_set_pdl(0)
  mods[engine]=mod
 return mods

def _lad_init(state,dev):
 flag=torch.zeros(1,dtype=torch.int32,device=dev)
 st=state["banks"][0]["k2a33"]["st"]
 st["hyb"]=list(st["hyb"]); st["hyb"][8]=flag
 state["lad_flag"]=flag

def lad_arm(state,mask,m):
 if mask & ~7 or m not in (128,256): raise ValueError((mask,m))
 flag=torch.zeros(1,dtype=torch.int32,device=state["out"][m].device)
 flag.fill_(mask); actual=int(flag.item())
 assert actual==mask
 print(f"[lad] flag readback={actual} M={m}",flush=True)
 st=dict(state["banks"][0]["k2a33"]["st"])
 st["hyb"]=list(st["hyb"]); st["hyb"][8]=flag
 bank=dict(state["banks"][0]); bank["k2a33"]=dict(bank["k2a33"],st=st)
 arm=dict(state); arm["banks"]=dict(state["banks"]); arm["banks"][0]=bank
 arm["h_sc"],arm["h_full"]=_hyb_scratch(flag.device)
 return arm

def lad_capture_mask(state,mask):
 assert int(state["banks"][0]["k2a33"]["st"]["hyb"][8].item())==mask
 # BT4's kernel-argument snapshot is set immediately before graph capture.
 for name in ("hyb","hyb7"):
  state["mods"][name].okc_set_pdl(0)
  state["mods"][name].bt4_set_mask(4 if mask & 2 else 0)
# LAD_END
'''
def once(s,a,b):
 n=s.count(a)
 if n!=1: raise AssertionError((n,a[:100]))
 return s.replace(a,b,1)
def generate(base):
 text=once(base,' _mbl_validate(state,weights)\n',' _lad_init(state,dev)\n _mbl_validate(state,weights)\n')
 text=text.rstrip()+'\n'+EXT.replace('@EDITS@',repr(EDITS))
 ast.parse(text); return text

def extension_env(source):
 env=bt4.environment(source); env.update(_mbl_builds=lambda deadline:{},_mbl_call=lambda *a:None)
 exec(compile(source[source.index('# LAD_BEGIN'):],'<lad extension>','exec'),env); return env

def cpu_gates(base,variant):
 assert variant==generate(base)
 a,b=bt4.environment(base),bt4.environment(variant)
 for key in ('VMS_CUDA_SRC','_HYB_MOE_CUDA','_HYB_MOE_CPP','_HYBLO_CUDA','_vml_CUDA_T','_vml_CUDA','_vml_CPP'):
  assert a[key]==b[key],key
 env=extension_env(variant)
 for engine in ('hyb','hyb7'):
  src=env['lad_sources'](engine)[0]
  for old,new in reversed(EDITS): src=once(src,new,old)
  assert src==a['_HYB_MOE_CUDA']
 for n in (2,88,1408): assert sorted(i for t in range(256) for i in range(t,n,256))==list(range(n))
 return {'status':'PASS','base_cuda_globals_equal':True,'instrumentation_reversible':True,'reset_coverage':True,'sha256':hashlib.sha256(variant.encode()).hexdigest()}

def change_report(base,variant):
 env=extension_env(variant); out={}
 for name in ('hyb','hyb7'):
  old=env['lad_sources'](name,False)[0].splitlines(); new=env['lad_sources'](name)[0].splitlines()
  out[name]=[{'kind':tag,'base_cuda_lines':[i+1,j],'variant_cuda_lines':[k+1,l], 'before':old[i:j],'after':new[k:l]} for tag,i,j,k,l in difflib.SequenceMatcher(a=old,b=new,autojunk=False).get_opcodes() if tag!='equal']
 return out
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('--check',action='store_true'); ap.add_argument('--changes',action='store_true'); args=ap.parse_args()
 base=BASE.read_text(encoding='utf8'); variant=generate(base)
 if args.check: assert DEST.read_bytes()==variant.encode()
 else: DEST.write_bytes(variant.encode())
 result=cpu_gates(base,variant)
 if args.changes: result['changes']=change_report(base,variant)
 print(json.dumps(result,indent=2))

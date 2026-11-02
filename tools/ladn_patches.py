#!/usr/bin/env python3
"""Generate the timing-only HYB ladder; no GPU work or edits to the base."""
import ast, difflib, hashlib, importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'kernels/muse_moe_w4a16/mb_layer.py'
DEST=ROOT/'kernels/muse_moe_w4a16/mb_layer_ladn.py'
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
M_HELPER = r'''__device__ __forceinline__ void ladn_ldsm(uint32_t addr, uint32_t& b0, uint32_t& b1) {
asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
}
__device__ __forceinline__ void ladn_mma(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
'''
M_OLD = r'''if (FIRST && ci == 0 && jj == 0)
WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
else
WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));'''
M_NEW = r'''if constexpr (NTT <= 2) {
#pragma unroll
for (int n = 0; n < NTT; ++n) {
if (FIRST && ci == 0 && jj == 0) { d[n][0] = d[n][1] = d[n][2] = d[n][3] = 0.f; }
uint32_t b0, b1;
// One 256-byte K16 x N8 tile; x2 uses lanes 0..15 as row addresses.
ladn_ldsm(xring_s + xs * WGS_XS + n * 4096 + (lg * 4 + h * GS + jj) * 256 + (lane & 15) * 16, b0, b1);
ladn_mma(d[n], af[ci & 1][jj], b0, b1);
}
} else {
if (FIRST && ci == 0 && jj == 0)
WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
else
WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3], xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
}
'''
M_NEW = '#if defined(OKC_LADM_M) && OKC_LADM_M\n' + M_NEW + '\n#else\n' + M_OLD + '\n#endif'
M_EDITS=[('template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg', M_HELPER+'template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg'), (M_OLD,M_NEW)]
EDITS=L2_EDITS+bt4.HYB_EDITS+G_EDITS
EXT=r'''
# LADN_BEGIN -- timing probes; output of nonzero arms is invalid.
_LADN_EDITS = @EDITS@
_LADN_M_EDITS = @M_EDITS@
_ladn_original_builds = _mbl_builds

def ladn_sources(engine, instrument=True, m=False):
 if engine not in ("hyb", "hyb7"): raise ValueError(engine)
 src = _HYB_MOE_CUDA
 if instrument:
  for old,new in _LADN_EDITS:
   assert src.count(old)==1, (src.count(old), old[:80])
   src=src.replace(old,new,1)
 if m:
  assert instrument
  for old,new in _LADN_M_EDITS:
   assert src.count(old)==1, (src.count(old),old[:80])
   src=src.replace(old,new,1)
 cpp=_HYB_MOE_CPP + ("\nvoid bt4_set_mask(int64_t mask);\n" if instrument else "")
 return src,cpp,list(HYB_FLAGS)+(list(HYB_F7) if engine=="hyb7" else [])+(["-DOKC_LADM_M=1"] if m else [])

def _mbl_builds(deadline):
 mods=_ladn_original_builds(deadline)
 for engine in ("hyb","hyb7"):
  for m in (False, True):
   _mbl_deadline(deadline,"LADN instrumented build")
   src,cpp,flags=ladn_sources(engine,m=m)
   tag=hashlib.sha1((cpp+src+" ".join(flags)).encode()).hexdigest()[:12]
   mod=_mbl_load_inline(name="ladn_"+engine+("m_" if m else "_")+tag,cpp_sources=cpp,cuda_sources=src,
                        functions=["okc_run","okc_set_pdl","bt4_set_mask"],extra_cuda_cflags=flags,verbose=False)
   mod.bt4_set_mask(0); mod.okc_set_pdl(0)
   mods[engine+("_m" if m else "")]=mod
 return mods

def _ladn_init(state,dev):
 flag=torch.zeros(1,dtype=torch.int32,device=dev)
 st=state["banks"][0]["k2a33"]["st"]
 st["hyb"]=list(st["hyb"]); st["hyb"][8]=flag
 state["ladn_flag"]=flag

def ladn_arm(state,mask,m):
 if mask & ~15 or m not in (128,256): raise ValueError((mask,m))
 flag=torch.zeros(1,dtype=torch.int32,device=state["out"][m].device)
 flag.fill_(mask); actual=int(flag.item())
 assert actual==mask
 print(f"[ladn] flag readback={actual} M={m}",flush=True)
 st=dict(state["banks"][0]["k2a33"]["st"])
 st["hyb"]=list(st["hyb"]); st["hyb"][8]=flag
 bank=dict(state["banks"][0]); bank["k2a33"]=dict(bank["k2a33"],st=st)
 arm=dict(state); arm["banks"]=dict(state["banks"]); arm["banks"][0]=bank
 arm["mods"]=dict(state["mods"])
 if mask & 8:
  for name in ("hyb","hyb7"): arm["mods"][name]=state["mods"][name+"_m"]
 arm["h_sc"],arm["h_full"]=_hyb_scratch(flag.device)
 return arm

def ladn_capture_mask(state,mask):
 assert int(state["banks"][0]["k2a33"]["st"]["hyb"][8].item())==mask
 # BT4's kernel-argument snapshot is set immediately before graph capture.
 for name in ("hyb","hyb7"):
  state["mods"][name].okc_set_pdl(0)
  state["mods"][name].bt4_set_mask(4 if mask & 2 else 0)
# LADN_END
'''
def once(s,a,b):
 n=s.count(a)
 if n!=1: raise AssertionError((n,a[:100]))
 return s.replace(a,b,1)
def generate(base):
 text=once(base,' _mbl_validate(state,weights)\n',' _ladn_init(state,dev)\n _mbl_validate(state,weights)\n')
 text=text.rstrip()+'\n'+EXT.replace('@EDITS@',repr(EDITS)).replace('@M_EDITS@',repr(M_EDITS))
 ast.parse(text); return text

def extension_env(source):
 env=bt4.environment(source); env.update(_mbl_builds=lambda deadline:{},_mbl_call=lambda *a:None)
 exec(compile(source[source.index('# LADN_BEGIN'):],'<lad extension>','exec'),env); return env

def cpu_gates(base,variant):
 assert variant==generate(base)
 a,b=bt4.environment(base),bt4.environment(variant)
 for key in ('VMS_CUDA_SRC','_HYB_MOE_CUDA','_HYB_MOE_CPP','_HYBLO_CUDA','_vml_CUDA_T','_vml_CUDA','_vml_CPP'):
  assert a[key]==b[key],key
 env=extension_env(variant)
 for engine in ('hyb','hyb7'):
  src=env['ladn_sources'](engine)[0]
  msrc=env['ladn_sources'](engine,m=True)[0]
  assert M_NEW in msrc and M_HELPER not in src
  assert '*p.l2r_mode' not in M_NEW and 'if constexpr (NTT <= 2)' in M_NEW
  assert M_OLD in M_NEW and 'x2.shared.b16' in M_HELPER
  for old,new in reversed(M_EDITS): msrc=once(msrc,new,old)
  assert msrc==src
  assert '-DOKC_LADM_M=1' in env['ladn_sources'](engine,m=True)[2]
  assert '-DOKC_LADM_M=1' not in env['ladn_sources'](engine)[2]
  for old,new in reversed(EDITS): src=once(src,new,old)
  assert src==a['_HYB_MOE_CUDA']
 for n in (2,88,1408): assert sorted(i for t in range(256) for i in range(t,n,256))==list(range(n))
 return {'status':'PASS','base_cuda_globals_equal':True,'instrumentation_reversible':True,'reset_coverage':True,'sha256':hashlib.sha256(variant.encode()).hexdigest()}

def change_report(base,variant):
 env=extension_env(variant); out={}
 for name,m in (('hyb',False),('hyb',True),('hyb7',False),('hyb7',True)):
  old=env['ladn_sources'](name,False)[0].splitlines(); new=env['ladn_sources'](name,m=m)[0].splitlines()
  name += '_m' if m else '_base'
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

"""Build the default-off QCAP GD experiment; --dry-run runs CPU gates."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'kernels/lab/k3_diet_rq128.py'
OUT = ROOT / 'kernels/lab/k3_qcap_ab.py'

def sub(s, a, b):
    assert s.count(a) == 1, (a[:80], s.count(a))
    return s.replace(a, b)

GLUE = '''
QCAP = 0
QCAP_KEYS = (112, 120)
def _qcap_key(cap):
 return _gd_key(*_gd_nk("qkv",1),GD_ROUTE[("qkv",1)])+(0,cap)
def _qcap_prepare(state):
 st=state.get("gd")
 if st is None:
  return
 st["qcap"]={}
 ids={k:i for i,k in enumerate(_GD["keys"])}
 for b in (1,8):
  base=st["route"].get(("qkv",b))
  if base is None:
   continue
  w=state["dense"][GD_WEIGHT["qkv"]][:base["N"]].contiguous()
  for cap in QCAP_KEYS:
   try:
    inst=ids[_qcap_key(cap)]
    info=st["mod"].gd_info(inst)
    if info[1]>64 or info[2]*torch.cuda.get_device_properties(w.device).multi_processor_count<cap:
     raise RuntimeError(f"resources {list(info)} for {cap} CTAs")
    r=dict(base,inst=inst,desc=base["desc"]+f"-qcap{cap}")
    ok,msg=_gd_selftest(r,w,b)
    if not ok:
     raise RuntimeError(msg)
    st["qcap"][(b,cap)]=r
    print(f"[qcap] self-test qkv/{b}/{cap}: ok ({msg}); resources {list(info)}",flush=True)
   except Exception as ex:
    print(f"[qcap] FALLBACK qkv/{b}/{cap}: {ex!r}",flush=True)
'''

LINUX = r'''#!/usr/bin/env bash
set -uo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
cd "<repo>"
export PYTHONDONTWRITEBYTECODE=1
export QCAP_RESULT="$(dirname "$0")/qcap_results.json"
export QCAP_WORK="$(mktemp -d)"
python - <<'PY'
import ast,json,os,pathlib,subprocess,sys
results={}
def run(name,cmd):
 p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 results[name]={'exit':p.returncode,'output':p.stdout}
 print(name,p.returncode,flush=True)
 return p.returncode
run('generator',[sys.executable,'-I','-B','tools/qcap_patches.py'])
run('dry_run',[sys.executable,'-I','-B','tools/qcap_patches.py','--dry-run'])
run('lead_gate',[sys.executable,'-I','-B','tools/lead_gate.py','file','kernels/lab/k3_qcap_ab.py'])
tree=ast.parse(pathlib.Path('kernels/lab/k3_qcap_ab.py').read_text())
src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GD_CUDA_SRC' for t in n.targets))
keys=[(0,1,1,4,8,2,4,5,1),(0,1,1,4,8,2,4,5,1,0,112),(0,1,1,4,8,2,4,5,1,0,120)]
for macro in ('INST','INFO'):
 text='\n'.join('    GD_%s(%d, %s)'%(macro,i,', '.join(map(str,k))) for i,k in enumerate(keys))
 src=src.replace('@'+macro+'@',text)
w=pathlib.Path(os.environ['QCAP_WORK']); cu=w/'qcap.cu'; obj=w/'qcap.o'; cu.write_text(src)
if not run('nvcc',['nvcc','-O3','-arch=sm_90a','--resource-usage','-c',str(cu),'-o',str(obj)]):
 run('cuobjdump_resources',['cuobjdump','--dump-resource-usage',str(obj)])
 run('cuobjdump_sass',['cuobjdump','--dump-sass',str(obj)])
pathlib.Path(os.environ['QCAP_RESULT']).write_text(json.dumps(results,indent=2))
sys.exit(any(v['exit'] for v in results.values()))
PY
'''

def generate(s):
    s=sub(s,'_gd_prepare(state,t0)\n _gd_cat_prepare(state,t0)','_gd_prepare(state,t0)\n _qcap_prepare(state)\n _gd_cat_prepare(state,t0)')
    s=sub(s,'def _gd_bucket(T):',GLUE+'\ndef _gd_bucket(T):')
    s=sub(s,' for key in _gd_cat_keys():\n',' for cap in QCAP_KEYS:\n  key=_qcap_key(cap)\n  if key not in keys:\n   keys.append(key)\n for key in _gd_cat_keys():\n')
    s=sub(s,' r=st["route"].get((proj,b))\n if r is None or tokens>r["T"]',' r=st["route"].get((proj,b))\n if proj=="qkv" and b in (1,8) and QCAP in QCAP_KEYS and r is not None:\n  r=st.get("qcap",{}).get((b,QCAP))\n  if r is None:\n   raise RuntimeError(f"QCAP {QCAP} qkv/{b} was not prepared or failed self-test")\n if r is None or tokens>r["T"]')
    s=sub(s,'int VF = 0, int CAT = 0>\n__global__ void __launch_bounds__','int VF = 0, int CAT = 0, int CAP = 0>\n__global__ void __launch_bounds__')
    s=sub(s,'''const int split = S > 1 ? (int)blockIdx.y : 0;
const int G = (int)(gridDim.x * gridDim.y) * NW;
const int gw = ((int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x) * NW + warp;''','''const int physical_bx = (int)blockIdx.x;
for (int logical_bx = physical_bx; logical_bx < (CAP ? 144 : (int)gridDim.x); logical_bx += (int)gridDim.x) {
const int split = S > 1 ? (int)blockIdx.y : 0;
const int logical_nbx = CAP ? 144 : (int)gridDim.x;
const int G = logical_nbx * (int)gridDim.y * NW;
const int gw = ((int)blockIdx.y * logical_nbx + logical_bx) * NW + warp;''')
    s=sub(s,'const int row = ((int)blockIdx.x * WR + w2) * 16 + gg + 8 * (c >> 1);','const int row = (logical_bx * WR + w2) * 16 + gg + 8 * (c >> 1);')
    s=sub(s,'for (int i = (int)threadIdx.x; i < NO; i += NT) st_async(rb + 4u * (uint32_t)i, cta_value(i), rm);\nreturn;\n}\nfloat own','for (int i = (int)threadIdx.x; i < NO; i += NT) st_async(rb + 4u * (uint32_t)i, cta_value(i), rm);\n} else {\nfloat own')
    s=sub(s,'''store(i, ok ? v : __int_as_float(0x7fc00000));
}
}
#endif
}
}
template <int OUT''','''store(i, ok ? v : __int_as_float(0x7fc00000));
}
}
}
#endif
}
#if GD_SM90
if constexpr (CAP && S > 1) { if (logical_bx + (int)gridDim.x < 144) { cl_arrive(); cl_wait(); } }
#endif
}
}
template <int OUT''')
    s=sub(s,'int XP, int CAT = 0>\nstatic int launch','int XP, int CAT = 0, int CAP = 0>\nstatic int launch')
    s=sub(s,'cfg.gridDim = dim3((unsigned)nbx, (unsigned)S, (unsigned)nz);','cfg.gridDim = dim3((unsigned)(CAP ? CAP / S : nbx), (unsigned)S, (unsigned)nz);')
    s=sub(s,'gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>, A)','gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT, CAP>, A)')
    s=sub(s,'int XP, int CAT = 0>\nstatic int info','int XP, int CAT = 0, int CAP = 0>\nstatic int info')
    s=sub(s,'auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>;','auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT, CAP>;')
    return s

def dryrun(s):
    tree=ast.parse(s)
    wanted={'_gd_key','_gd_nk','_gd_bucket','_gd_keys','_gd_live','_qcap_key','_qcap_prepare'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes}==wanted
    class W:
        device='cpu'
        def __getitem__(self,k): return self
        def contiguous(self): return self
    class Mod:
        def gd_info(self,i): return [32,0,2,4096]
    class CUDA:
        @staticmethod
        def get_device_properties(d): return type('P',(),{'multi_processor_count':132})()
    class Torch: cuda=CUDA()
    route_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GD_ROUTE' for t in n.targets))
    all_routes=ast.literal_eval(route_node.value)
    routes={k:all_routes[k] for k in (('qkv',1),('qkv',8))}
    assert routes[('qkv',1)]==routes[('qkv',8)]==(1,1,4,8,2,4,1)
    env={'HIDDEN':8192,'UNPOOL_SLICE':1024,'SLICED_UNPOOL_MAX_TOKENS':8,'GD_SMS':132,'GD_DIMS':{'qkv':(2304,8192)},'GD_ROUTE':routes,'GD_WEIGHT':{'qkv':'qkv_gate'},'QCAP_KEYS':(112,120),'QCAP':0,'GD_ENABLE':True,'_nb64':lambda t:False,'_gd_cat_keys':lambda:[],'_gd_selftest':lambda r,w,b:(True,'mock'),'torch':Torch(),'NB64_GD_B':{},'NB64_USE':{}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<qcap-dryrun>','exec'),env)
    keys=env['_gd_keys'](); env['_GD']={'keys':keys}
    m=Mod(); b={'mod':m,'inst':0,'N':2304,'K':8192,'T':8,'slice':False,'desc':'base'}
    st={'dense':{'qkv_gate':W()},'gd':{'mod':m,'route':{('qkv',1):b,('qkv',8):b}}}
    env['_qcap_prepare']({'gd':None}); env['_qcap_prepare'](st)
    assert len(st['gd']['qcap'])==4
    for cap in (0,112,120):
        env['QCAP']=cap
        for t in (1,2,5,8):
            r=env['_gd_live'](st,'qkv',t)
            assert r is not None and (r['inst']==0 if cap==0 else r['inst']==keys.index(env['_qcap_key'](cap)))
    assert env['_qcap_key'](112)==(0,1,1,4,8,2,4,5,1,0,112)
    env['_gd_selftest']=lambda r,w,b:(False,'expected failure')
    env['_qcap_prepare'](st)
    assert st['gd']['qcap']=={}
    try:
        env['_gd_live'](st,'qkv',1)
        raise AssertionError('failed variant was silently selected')
    except RuntimeError:
        pass
    st['gd']['route'].clear()
    env['_qcap_prepare'](st)
    assert st['gd']['qcap']=={}
    for n in (28,30):
        assert sorted(x for bx in range(n) for x in range(bx,144,n))==list(range(144))
    print('CPU dry run PASS: prepare, keys, arm selection, tile coverage')

def main():
    if '--emit-linux' in sys.argv:
        sys.stdout.buffer.write(LINUX.encode('utf-8')); return
    s=generate(BASE.read_text(encoding='utf-8')); ast.parse(s)
    if '--dry-run' not in sys.argv:
        OUT.write_text(s,encoding='utf-8',newline='\n')
        print(f'generated {OUT} ({len(s.encode())} bytes)')
    dryrun(s)

if __name__=='__main__': main()

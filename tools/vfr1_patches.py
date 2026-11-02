"""VFR1 in-file lab. Only OUT is written; no CUDA, imports of the layer or caches.

python -B tools/vfr1_patches.py [--check]
The CPU doubles validate Python plumbing, NOT GPU arithmetic or performance.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import time
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c1.py"
OUT = ROOT / "kernels/lab/k3_vfr1_ab.py"


def sub(text, old, new):
    assert text.count(old) == 1, ("anchor must occur exactly once", text.count(old), old[:120])
    return text.replace(old, new, 1)


def function(text, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, name
    return nodes[0]


def literal(text, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


def replace_function(text, name, new):
    node = function(text, name)
    old = ast.get_source_segment(text, node)
    return sub(text, old, new.strip())


SELECTOR = '''def _vfr_fused(state,ffn_input,router_input):
 if not(VF_GDR and _VFR["ok"])or router_input is None or router_input.dtype!=torch.float32:
  return None
 T=ffn_input.shape[0]
 if not(1<=T<=8 and router_input.is_contiguous()and ffn_input.is_contiguous()):
  return None
 vf=_vfr1_choice(T)
 return None if not vf else _vfr_gdr(state,ffn_input,router_input,vf)
'''

LAB = '''# VFR1: host-only selectors, read at call / CUDA-graph capture time.
VFR1=0
VFR1_T8=0
_VFR1={"bank":{},"tests":[],"errors":{}}
def _vfr1_choice(T):
 if VFR1 not in(0,2,3)or VFR1_T8 not in(0,1):
  raise ValueError("VFR1 must be 0/2/3; VFR1_T8 must be 0/1")
 if T==1:
  H=_VFR1["bank"].get((1,VFR1))
  return VFR1 if H is not None and H["ok"]else 0
 if T==8 and VFR1==3 and VFR1_T8:
  H=_VFR1["bank"].get((8,3))
  return 3 if H is not None and H["ok"]else 2
 return 2 if 2<=T<=8 else 0
def _vfr1_bits(a,b):
 return a.shape==b.shape and a.dtype==b.dtype and bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
def _vfr1_logits(part):
 # tail_kernel3<false,8,8>: qa[0], then qa[1..7], then qb=+0.
 lg=part[0].clone()
 for q in range(1,8):
  lg=lg+part[q]
 return lg+0.0
def _vfr1_input(m,case,dev):
 gen=torch.Generator(device=dev).manual_seed(91301+37*m+case)
 z=torch.randn((m,HIDDEN),generator=gen,device=dev,dtype=torch.float32)
 if case==2:
  z[:,::257]*=64.0
 elif case==3:
  z*=0.0001
  z[:,0]=128.0
  z[:,4095]=-128.0
 elif case==4:
  z.zero_()
 z=(z*torch.rsqrt(z.pow(2).mean(-1,keepdim=True).clamp_min(1e-20))).contiguous()
 return(z*0.7).to(torch.bfloat16).contiguous(),z
def _vfr1_prepare(state):
 # Same bank/ok/tests/errors discipline as the padz in-file lab. No repack.
 _VFR1["bank"]={(m,vf):{"ok":False,"info":None}for m,vf in((1,2),(1,3),(8,3))}
 _VFR1["tests"].clear()
 _VFR1["errors"].clear()
 for key,H in _VFR1["bank"].items():
  m,vf=key
  try:
   assert _VFR["ok"]and state.get("v8")is not None and _SBB1.get("mod")is not None,"ship VFR/tail/SBB1 unavailable"
   r=_gd_live(state,"pooler",m)
   assert r is not None and r["desc"].startswith("m1(1,8,16,1)s2x"),"pooler geometry unavailable"
   assert(r["N"],r["K"],r["nbx"])==(4096,8192,256),"pooler dimensions changed"
   H["info"]=list(r["mod"].gdr_info(vf))
   assert H["info"][1]==0 and H["info"][2]>=2,"local memory or occupancy < 2"
   dev=state["router"].device
   good=True
   for case in range(5):
    h,z=_vfr1_input(m,case,dev)
    d={"local_start":torch.tensor([(0,64,224,32,160)[case]],dtype=torch.int32,device=dev)}
    y0=_gd(state,"pooler",h)
    assert y0 is not None,"ship pooler unavailable"
    p0=torch.full((8,m,N_EXPERTS),float("nan"),dtype=torch.float32,device=dev)
    # T=1 ship expression from _v8_logits; T=8 ship is the existing VF2 instance.
    if m==1:
     _SBB1["mod"].vf_router32(z,state["router"],p0,0)
    else:
     y0,p0=_vfr_gdr(state,h,z,2,check=True)
    ref_tail=_v8_tail(y0,p0,state,d)
    ref_logits=_vfr1_logits(p0)
    for captured in(False,True):
     if captured:
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):
       y,p=_vfr_gdr(state,h,z,vf)
       got_tail=_v8_tail(y,p,state,d)
      graph.replay()
      graph.replay()
     else:
      y,p=_vfr_gdr(state,h,z,vf,check=True)
      got_tail=_v8_tail(y,p,state,d)
     lg=_vfr1_logits(p)
     torch.cuda.synchronize()
     checks={"pooler":_vfr1_bits(y0,y),"partials":_vfr1_bits(p0,p),"logits":_vfr1_bits(ref_logits,lg),
      "top8_ids":_vfr1_bits(ref_tail[2],got_tail[2]),"top8_weights":_vfr1_bits(ref_tail[1],got_tail[1]),
      "local_ids":_vfr1_bits(ref_tail[3],got_tail[3]),"expert_input":_vfr1_bits(ref_tail[0],got_tail[0]),
      "finite":bool(torch.isfinite(y).all()and torch.isfinite(p).all()and torch.isfinite(lg).all())}
     _VFR1["tests"].append({"T":m,"vf":vf,"case":case,"graph":captured,**checks})
     good&=all(checks.values())
     if not all(checks.values()):
      print(f"[vfr1] FAIL T={m} vf={vf} case={case} graph={captured}: {checks}",flush=True)
   assert good,"bitwise/finite gate failed; see per-case checks"
   H["ok"]=True
  except Exception as ex:
   H["ok"]=False
   _VFR1["errors"][key]=repr(ex)
   print(f"[vfr1] OFF T={m} vf={vf}: {ex!r}",flush=True)
  print(f"[vfr1] T={m} vf={vf} {'LIVE' if H['ok']else 'OFF'} regs/local/occ/smem {H['info']}",flush=True)
 # Do not change the user's selectors, the ship VFR gate, or neighbour PDL flags.
'''


def generate(base):
    text = sub(base, ' _vfr_prepare(state)\n', ' _vfr_prepare(state)\n _vfr1_prepare(state)\n')
    # Only HOST dispatch changes. The entire gd namespace/kernel stays verbatim.
    text = sub(text, 'if (vf == 2)\nTORCH_CHECK(x32.scalar_type()',
               'TORCH_CHECK(vf == 1 || vf == 2 || vf == 3, "gdr_run: invalid vf");\nif (vf >= 2)\nTORCH_CHECK(x32.scalar_type()')
    text = sub(text, '(int)nbx, (int)pdl, vf == 2 ? x32.data_ptr<float>() : nullptr,\n'
               'vf == 2 ? w32.data_ptr<float>() : nullptr, vf == 2 ? part.data_ptr<float>() : nullptr, mk,',
               '(int)nbx, (int)pdl, vf >= 2 ? x32.data_ptr<float>() : nullptr,\n'
               'vf >= 2 ? w32.data_ptr<float>() : nullptr, vf >= 2 ? part.data_ptr<float>() : nullptr, mk,')
    branch = '''if (vf == 3) {
static bool set3 = false;
if (!set3) {
const int e = (int)cudaFuncSetAttribute(gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 7, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, GDR_DS);
if (e) return e;
set3 = true;
}
cfg.dynamicSmemBytes = GDR_DS;
return (int)cudaLaunchKernelEx(&cfg, gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 7, 2>, A);
}
'''
    text = sub(text, 'cfg.numAttrs = na;\nif (vf == 2) {', 'cfg.numAttrs = na;\n' + branch + 'if (vf == 2) {')
    text = sub(text, 'auto fn = vf == 2 ? gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2> : gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 1>;\nconst int ds = vf == 2 ? GDR_DS : 0;',
               'auto fn = vf == 3 ? gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 7, 2> :\n'
               'vf == 2 ? gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2> : gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 1>;\n'
               'const int ds = vf >= 2 ? GDR_DS : 0;')
    text = sub(text, 'def _vfr_gdr(state,x,z,vf):', 'def _vfr_gdr(state,x,z,vf,check=False):')
    text = sub(text, ' if r is None or _gd_bucket(T)!=8:', ' if r is None or _gd_bucket(T)not in(1,8):')
    text = sub(text, ' z if vf==2 else _VFR["nomk"],state["router"],part,mk)',
               ' z if vf>=2 else _VFR["nomk"],state["router"],part,mk)')
    text = sub(text, ' mk=_VFR["marks"]if VF_GDM else _VFR["nomk"]\n',
               ' if check:\n  y.fill_(float("nan"))\n  part.fill_(float("nan"))\n mk=_VFR["marks"]if VF_GDM else _VFR["nomk"]\n')
    text = replace_function(text, '_vfr_fused', SELECTOR)
    text = sub(text, 'def _vfr_prepare(state):\n', LAB + '\ndef _vfr_prepare(state):\n')
    compile(text, str(OUT), 'exec')
    return text


def load_definitions(text, ns, names):
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), '<vfr1-extracted>', 'exec'), ns)


def config_namespace(text):
    ns = {"HIDDEN":8192,"D_EXPERT":4096,"N_EXPERTS":256,"UNPOOL_SLICE":1024,
          "SLICED_UNPOOL_MAX_TOKENS":8}
    names = {'GD_ROUTE','GD_DIMS','GD_WEIGHT','GD_SMS','GD_CAT_ROUTE','GD_CAT_UNITS',
             'GD_ENABLE','GD_BUILD','GD_PREPARE_BY_S','GD_ONE_WAVE','GD_PDL','GD_TRIG','GD_REQUIRE'}
    found = set()
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            exec(compile(ast.Module([node], []), '<base-config>', 'exec'), ns)
            found.add(node.targets[0].id)
    assert found == names, names-found
    load_definitions(text, ns, {'_gd_key','_gd_keys','_gd_nk','_gd_cat_key','_gd_cat_keys'})
    return ns


def builder_inputs(text):
    """Execute the real _gd_build, replacing only its import with injected load_inline."""
    ns = config_namespace(text)
    ns.update(_GD={"mod":None,"keys":None},_pt1_wait=lambda k:None,time=time,hashlib=hashlib,
              GD_CPP_SRC=literal(text,'GD_CPP_SRC'),GD_CUDA_SRC=literal(text,'GD_CUDA_SRC'),
              load_inline=lambda **kw:kw)
    node = function(text, '_gd_build')
    imports = [n for n in node.body if isinstance(n,ast.ImportFrom)]
    assert len(imports)==1 and imports[0].module=='torch.utils.cpp_extension'
    node.body.remove(imports[0])
    exec(compile(ast.Module([node],[]),'<gd-build>','exec'),ns)
    return ns['_gd_build']()


def gates(base, new):
    compile(new,str(OUT),'exec')
    a, b = literal(base,'GD_CUDA_SRC'), literal(new,'GD_CUDA_SRC')
    assert a.split('using namespace gd;')[0] == b.split('using namespace gd;')[0]
    changed = {'prepare','_vfr_gdr','_vfr_fused'}
    fn0 = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(base).body if isinstance(n,ast.FunctionDef)}
    fn1 = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(new).body if isinstance(n,ast.FunctionDef)}
    assert {k for k,v in fn0.items() if fn1.get(k)!=v} == changed
    constants1 = {n.targets[0].id:n.value.value for n in ast.parse(new).body
                  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                  and isinstance(n.value,ast.Constant)}
    for node in ast.parse(base).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name = node.targets[0].id
                if name not in ('GD_CPP_SRC','GD_CUDA_SRC'):
                    assert constants1[name]==node.value.value,name
    assert literal(new,'VFR1')==literal(new,'VFR1_T8')==0
    assert len(new.encode('ascii'))<1_000_000
    assert new.splitlines()[0]=='# okc: problem=muse_decode_layer'
    ns = config_namespace(base)
    r1,r8 = (ns['GD_ROUTE'][('pooler',t)] for t in (1,8))
    assert (r1[1],r1[2],r1[3],r1[5])==(r8[1],r8[2],r8[3],r8[5])==(1,8,16,1)
    old, ab = builder_inputs(base), builder_inputs(new)
    assert old['extra_cuda_cflags']==ab['extra_cuda_cflags']
    assert old['functions']==ab['functions']
    assert '@INST@' not in ab['cuda_sources'] and '@INFO@' not in ab['cuda_sources']
    # Source-level per-output FMA coordinate order, including warp remapping.
    for m in (1,8):
        seen=set()
        for cta in range(256):
            bx,sl=cta&31,cta>>5
            for warp in range(8):
                for j in range(4):
                    tok=4*(warp&1)+j
                    if tok>=m: continue
                    for row in (bx*8+2*(warp>>1),bx*8+2*(warp>>1)+1):
                        for lane in range(32):
                            old_order=[sl*1024+4*((hh*2+i)*32+lane)+q for hh in range(4) for i in range(2) for q in range(4)]
                            new_order=[sl*1024+4*(i*32+lane)+q for i in range(8) for q in range(4)]
                            assert old_order==new_order
                        assert (sl,tok,row) not in seen
                        seen.add((sl,tok,row))
        assert len(seen)==8*m*256
    return {"python_compile":"PASS","unchanged_kernel_body":"PASS","unchanged_functions":len(fn0)-len(changed),
            "same_pooler_pack":"PASS","fma_order_and_unique_partial_coverage":"PASS",
            "builder_expressions":"PASS","base_sha256":hashlib.sha256(base.encode()).hexdigest(),
            "lab_bytes":len(new.encode('ascii'))}


def cpu_dry_run(base,new):
    """Real base _gd_prepare builds the routes; real _v8_logits/_v8_tail execute.

    Strict module signatures + scalar tensor doubles exercise all new Python,
    including capture and injected failures. They do not simulate CUDA math.
    """
    class Tensor:
        def __init__(self,shape=(),dtype='f32',value=1.0):
            self.shape=tuple(shape) if isinstance(shape,(tuple,list)) else (shape,)
            self.dtype,self.device,self.value=dtype,'cpu-double',value
        def to(self,dtype): return Tensor(self.shape,dtype,self.value)
        def float(self): return self.to('f32')
        def contiguous(self): return self
        def detach(self): return self
        def clone(self): return Tensor(self.shape,self.dtype,self.value)
        def view(self,dtype): return self.to(dtype)
        def dim(self): return len(self.shape)
        def numel(self): return math.prod(self.shape)
        def stride(self,i): return math.prod(self.shape[i+1:])
        def data_ptr(self): return 4096
        def is_contiguous(self): return True
        def fill_(self,v): self.value=v; return self
        def zero_(self): return self.fill_(0)
        def all(self): return bool(self.value)
        def pow(self,n): return Tensor(self.shape,self.dtype,self.value**n)
        def mean(self,dim,keepdim=False): return Tensor(self.shape[:-1]+((1,) if keepdim else ()),self.dtype,self.value)
        def clamp_min(self,v): return Tensor(self.shape,self.dtype,max(self.value,v))
        def __getitem__(self,key):
            if isinstance(key,int): return Tensor(self.shape[1:],self.dtype,self.value)
            if isinstance(key,slice): return Tensor((min(key.stop or self.shape[0],self.shape[0]),)+self.shape[1:],self.dtype,self.value)
            return self.clone()
        def __setitem__(self,key,value): pass
        def __mul__(self,v): return Tensor(self.shape,self.dtype,self.value*(v.value if isinstance(v,Tensor) else v))
        def __imul__(self,v): self.value*=v; return self
        def __add__(self,v): return Tensor(self.shape,self.dtype,self.value+(v.value if isinstance(v,Tensor) else v))
    class Generator:
        def __init__(self,**kw): pass
        def manual_seed(self,n): return self
    class Graph:
        def replay(self): pass
    torch=types.SimpleNamespace(bfloat16='bf16',float32='f32',int16='i16',int32='i32',int64='i64',uint8='u8',Generator=Generator,
        empty=lambda shape,dtype='f32',**kw:Tensor(shape,dtype,float('nan')),
        empty_like=lambda t:Tensor(t.shape,t.dtype,float('nan')),
        zeros=lambda shape,dtype='f32',**kw:Tensor(shape,dtype,0),
        full=lambda shape,v,dtype='f32',**kw:Tensor(shape,dtype,v),
        tensor=lambda v,dtype='f32',**kw:Tensor((len(v),),dtype,v[0]),
        randn=lambda shape,**kw:Tensor(shape),rsqrt=lambda t:Tensor(t.shape,t.dtype,1/math.sqrt(t.value)),
        equal=lambda a,b:a.shape==b.shape and a.dtype==b.dtype and a.value==b.value,
        isfinite=lambda t:Tensor((),value=math.isfinite(t.value)),
        cuda=types.SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,CUDAGraph=Graph,
            graph=lambda g:contextlib.nullcontext(),get_device_properties=lambda d:types.SimpleNamespace(multi_processor_count=132)))
    calls=[]
    fault={"key":None,"kind":None}
    class Module:
        def gd_info(self,inst): return [64,0,8,4096]
        def gdr_info(self,vf):
            if fault['kind']=='info' and vf==3: raise RuntimeError('injected info failure')
            return [64,4 if fault['kind']=='spill' and vf==3 else 0,1 if fault['kind']=='occ' and vf==3 else 2,4096]
        def gd_unpack(self,w,esc,out,k,G,nbx,wr,wk,ku,tb): out.value=1.0
        def gd_run(self,inst,w,esc,x,out,T,K,N,tb,nbx,nz,pdl): out.value=x.value
        def vf_router32(self,z,w,p,kind):
            assert w.shape==(256,8192) and kind==0
            p.value=z.value
        def v8_router32(self,z,w,p): p.value=z.value
        def gdr_run(self,vf,w,esc,x,y,T,K,N,tb,nbx,pdl,z,router,part,mk):
            assert (K,N,nbx)==(8192,4096,256) and pdl==3
            assert router.shape==(256,8192) and z.shape==(T,8192)
            calls.append((T,vf))
            y.value,part.value=x.value,z.value
            if fault['key']==(T,vf):
                if fault['kind']=='raise': raise RuntimeError('injected launch failure')
                if fault['kind']=='partial': part.value+=1
                if fault['kind']=='pooler': y.value+=1
                if fault['kind']=='nan': part.value=float('nan')
        def v8_tail(self,y,lg,la,lb,l_ld,ns,sstride,nw,bias,ls,xn,tw,tid,lid,eps,mult,rscale,deps,pdl):
            assert nw.shape==(4096,) and bias.shape==(256,) and ns==8
            xn.value,tw.value,tid.value,lid.value=y.value,lg.value,lg.value,ls.value
    mod=Module()
    ns=config_namespace(base)
    ns.update(torch=torch,time=time,_gd_build=lambda:mod,_gd_window_base=lambda w:0,
        _gd_selftest=lambda *a:(True,'CPU double'),NB64_BUILD={},_GD_SEEN=set(),
        _GD={"keys":ns['_gd_keys'](),"build_s":0},_nb64=lambda T:False,_SBB1={"mod":mod},
        _VF={"r8":True},VF_ROUTER=literal(base,'VF_ROUTER'),V9_TAIL_PDL=True,TL1_TAIL2=True,
        FT_OK=True,FT_TAIL3=literal(base,'FT_TAIL3'),RMS_EPS=1e-6,TOP_K=8,
        VF_GDR=True,VF_GDM=False,_VFR={"ok":True,"marks":Tensor((1024,),'i64'),"nomk":Tensor((0,),'i64'),"seen":set()},
        VFR1=0,VFR1_T8=0,_VFR1={"bank":{},"tests":[],"errors":{}})
    # Evaluate the exact pack return dict and base state entries, not guessed accessors.
    pack_return=next(n.value for n in ast.walk(function(base,'_gd_pack')) if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict))
    def pack(w,wr,wk,ku,s,base_):
        nbx=w.shape[0]//16//wr
        env=dict(v=Tensor((nbx*wk*ku*3*32*4,),'i32'),esc=Tensor((nbx*wk*32,),'i32'),base=base_,G=nbx*wk*s,nbx=nbx,nesc=0)
        return eval(compile(ast.Expression(pack_return),'<base-pack>','eval'),env)
    ns['_gd_pack']=pack
    weights={name:Tensor(shape,'bf16') for name,shape in {'qkv_gate':(2304,8192),'o':(8192,1024),'pooler':(4096,8192),'unpooler':(8192,4096)}.items()}
    weights.update(router=Tensor((256,8192)),router_bias=Tensor((256,)))
    state_node=next(n.value for n in function(base,'_r1_prepare').body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='state' for t in n.targets))
    state={ast.literal_eval(k):eval(compile(ast.Expression(v),'<base-state>','eval'),{'weights':weights,'router':weights['router']})
           for k,v in zip(state_node.keys,state_node.values) if ast.literal_eval(k) in ('dense','router')}
    state['pre_expert_norm']=types.SimpleNamespace(_get_effective_weight=lambda:Tensor((4096,),'bf16'))
    v8_node=next(n.value for n in function(base,'_v8_tail_prepare').body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='v8' for t in n.targets))
    state['v8']={ast.literal_eval(k):eval(compile(ast.Expression(v),'<base-v8>','eval'),{'state':state,'weights':weights,'mod':mod,'torch':torch})
                 for k,v in zip(v8_node.keys,v8_node.values) if ast.literal_eval(k) in ('mod','nw','bias')}
    names={'_gd_prepare','_gd_require','_gd_bucket','_gd_live','_gd','_v8_logits','_v8_tail','_ft_flag','_vfr_gdr','_vfr_fused'}
    names|={n.name for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name.startswith('_vfr1_')}
    load_definitions(new,ns,names)
    rows=[]
    with contextlib.redirect_stdout(io.StringIO()):
        ns['_gd_prepare'](state,time.time())
        r1,r8=(state['gd']['route'][('pooler',m)] for m in (1,8))
        assert r1['w'] is r8['w'] and r1['esc'] is r8['esc']
        ns['_vfr1_prepare'](state)
        assert all(h['ok'] for h in ns['_VFR1']['bank'].values()),ns['_VFR1']['errors']
        assert len(ns['_VFR1']['tests'])==30
        # Call/capture selection, with every scored T and intermediate bucket row.
        for arm in (0,2,3):
            for t8 in (0,1):
                ns.update(VFR1=arm,VFR1_T8=t8)
                for m in (1,2,5,8,16,32,128,256):
                    out=ns['_vfr_fused'](state,Tensor((m,8192),'bf16'),Tensor((m,8192)))
                    expected=arm if m==1 else 3 if m==8 and arm==3 and t8 else 2 if 2<=m<=8 else 0
                    assert (out is not None)==bool(expected),(arm,t8,m)
                    if expected: assert calls[-1]==(m,expected)
        for kind in ('partial','pooler','nan','raise','info','spill','occ'):
            fault.update(key=(1,3),kind=kind)
            ns['_vfr1_prepare'](state)
            assert not ns['_VFR1']['bank'][(1,3)]['ok'],kind
            assert ns['_VFR1']['bank'][(1,2)]['ok'],kind
            ns.update(VFR1=3,VFR1_T8=1)
            assert ns['_vfr1_choice'](1)==0
            if kind in ('info','spill','occ'): assert ns['_vfr1_choice'](8)==2
            rows.append(kind)
        fault.update(key=None,kind=None)
        # Missing ship dependencies, absent/wrong routes, and repeat prepare reset.
        for missing in ('gd','v8'):
            saved=state[missing]; state[missing]=None
            ns['_vfr1_prepare'](state)
            assert not any(h['ok'] for h in ns['_VFR1']['bank'].values())
            state[missing]=saved
        saved=ns['_SBB1']['mod']; ns['_SBB1']['mod']=None
        ns['_vfr1_prepare'](state)
        assert not any(h['ok'] for h in ns['_VFR1']['bank'].values())
        ns['_SBB1']['mod']=saved
        ns['_VFR']['ok']=False
        ns['_vfr1_prepare'](state)
        assert not any(h['ok'] for h in ns['_VFR1']['bank'].values())
        ns['_VFR']['ok']=True
        saved=r1['desc']; r1['desc']='wrong geometry'
        ns['_vfr1_prepare'](state)
        assert not ns['_VFR1']['bank'][(1,2)]['ok'] and ns['_VFR1']['bank'][(8,3)]['ok']
        r1['desc']=saved
        saved=r1['N']; r1['N']=1
        ns['_vfr1_prepare'](state)
        assert not ns['_VFR1']['bank'][(1,3)]['ok']
        r1['N']=saved
        ns['_vfr1_prepare'](state)
        assert all(h['ok'] for h in ns['_VFR1']['bank'].values())
        for arm,t8 in ((1,0),(3,2)):
            ns.update(VFR1=arm,VFR1_T8=t8)
            try: ns['_vfr1_choice'](1)
            except ValueError: pass
            else: raise AssertionError('invalid selector accepted')
        ns.update(VFR1=0,VFR1_T8=0)
    return {"status":"PASS (CPU doubles; no CUDA arithmetic)","prepare_tests":30,"selector_cases":48,
            "faults_disabled":rows,"real_base_gd_prepare":True,"shared_pack_identity":True,
            "missing_state_and_route_guards":"PASS","repeat_prepare_and_invalid_selectors":"PASS"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    base=BASE.read_bytes().decode('ascii').replace('\r\n','\n')
    new=generate(base)
    result={'gates':gates(base,new),'dry_run':cpu_dry_run(base,new)}
    if args.check:
        assert OUT.read_bytes()==new.encode('ascii'),'stale generated lab'
    else:
        OUT.write_bytes(new.encode('ascii'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

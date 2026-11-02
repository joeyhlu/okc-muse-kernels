"""HPF: default-off, in-file progressive HYB finalize on the MOEXR lab.

Only the named lab and report are emitted. --check performs CPU gates without
writes. CUDA compilation is delegated to tools/local/hpf_compile.py.
"""
from __future__ import annotations
import argparse
import ast
import builtins
import contextlib
import difflib
import hashlib
import io
import json
from pathlib import Path
import random
import re
import runpy
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/lab/k3_moexr_ab.py'
LAB = ROOT / 'kernels/lab/k3_hpf_ab.py'
REPORT = ROOT / 'box_out/reset/hpf_report.md'


def once(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new, 1)


def literal(s, name):
    matches = [n.value for n in ast.parse(s).body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(matches) == 1, name
    return ast.literal_eval(matches[0])


def nodes(s, names):
    body = [n for n in ast.parse(s).body if
            (isinstance(n, ast.FunctionDef) and n.name in names) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    return compile(ast.Module(body=body, type_ignores=[]), '<hpf-cpu>', 'exec')


def function(s, name):
    n, = [n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == name]
    return '\n'.join(s.splitlines()[n.lineno-1:n.end_lineno])


# 128 threads of ONE completing WG enter this helper; no CTA-wide barrier,
# ring storage, producer state, or another WG's scratch is used here.
COMPLETE = r'''
// HPF: the counter covers ALL B column blocks of ALL jobs for this expert.
__device__ __noinline__ void hpf_finalize(const Params& p, Smem& S, int e) {
const int wt = threadIdx.x & 127, wg = threadIdx.x >> 7;
const int lane = wt & 31, wp = wt >> 5;
const int total = S.cnt[e];
asm volatile("fence.acquire.gpu;\n" ::: "memory");
// One completed expert's actual rows, never a final pass over all tokens.
for (int q = S.off[e]; q < S.off[e] + total; ++q) {
 const int route = S.ent[q], tk = route >> 3;
 float4 a[4], b[4];
 // Four virtual groups of 128 reproduce ship's 512-thread RMS tree exactly.
 #pragma unroll
 for (int h = 0; h < 4; ++h) {
  const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + (wt + h * 128) * 2;
  a[h] = __ldcg(src); b[h] = __ldcg(src + 1);
  float ps = a[h].x*a[h].x + a[h].y*a[h].y + a[h].z*a[h].z + a[h].w*a[h].w
           + b[h].x*b[h].x + b[h].y*b[h].y + b[h].z*b[h].z + b[h].w*b[h].w;
  #pragma unroll
  for (int d = 16; d > 0; d >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, d);
  if (lane == 0) S.hred[wg][h * 4 + wp] = ps;
 }
 asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
 if (wt == 0) {
  float sum = 0.f;
  for (int w = 0; w < NW; ++w) sum += S.hred[wg][w];
  S.hred[wg][0] = p.topk_w[route] * rsqrtf(sum * (1.f / HD) + 1e-8f);
 }
 asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
 const float scale = S.hred[wg][0];
 #pragma unroll
 for (int h = 0; h < 4; ++h) {
  const float v[8] = {a[h].x,a[h].y,a[h].z,a[h].w,b[h].x,b[h].y,b[h].z,b[h].w};
  #pragma unroll
  for (int c = 0; c < 8; ++c) {
   const float value = scale * v[c];
   asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n"
    ::"l"(p.hacc + (size_t)tk * HD + (wt + h * 128) * 8 + c), "f"(value) : "memory");
  }
 }
 // Every lane drains its atomics before the leader publishes completion.
 asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
 asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
 if (wt == 0) {
  int need = 0;
  #pragma unroll
  for (int sl = 0; sl < 8; ++sl) need += S.pos[tk * 8 + sl] >= 0;
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n"
   : "=r"(old) : "l"(p.htok + tk), "r"(1) : "memory");
  S.hlast[wg] = old + 1 == need;
 }
 asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
 if (S.hlast[wg]) {
  asm volatile("fence.acquire.gpu;\n" ::: "memory");
  #pragma unroll
  for (int h = 0; h < 4; ++h) {
   float4* src = reinterpret_cast<float4*>(p.hacc + (size_t)tk * HD) + (wt + h * 128) * 2;
   const float4 va = __ldcg(src), vb = __ldcg(src + 1);
   const float v[8] = {va.x,va.y,va.z,va.w,vb.x,vb.y,vb.z,vb.w};
   uint4 out;
   uint32_t* words = reinterpret_cast<uint32_t*>(&out);
   #pragma unroll
   for (int c = 0; c < 4; ++c) {
    __nv_bfloat162 pair = __floats2bfloat162_rn(v[c*2],v[c*2+1]);
    words[c] = *reinterpret_cast<uint32_t*>(&pair);
   }
   *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + (wt + h * 128) * 8) = out;
   *src = make_float4(0.f,0.f,0.f,0.f);
   *(src + 1) = make_float4(0.f,0.f,0.f,0.f);
  }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (wt == 0) p.htok[tk] = 0;
 }
 // No lane can reuse hred/hlast before this row's other lanes finish.
 asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
}
}
__device__ __forceinline__ void hpf_complete(const Params& p, Smem& S, int e) {
const int wt = threadIdx.x & 127, wg = threadIdx.x >> 7, total = S.cnt[e];
const int target = 64 * (total > SK_T ? (total + FT_T - 1) / FT_T : 1);
if (wt == 0) {
 int old;
 asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n"
  : "=r"(old) : "l"(p.mfl + MF_CE + mf_gen(S) * 32 + e), "r"(1) : "memory");
 S.hlast[wg] = old + 1 == target;
}
asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
if (S.hlast[wg]) {
 // Completion can be published while the NEXT unit has one in-flight group.
 // Drain before an out-of-line call can save live WGMMA accumulator registers.
 wg_wait<0>();
 hpf_finalize(p, S, e);
}
}
'''


def cuda_source(cu):
    edits = []
    def patch(label, old, new):
        nonlocal cu
        edits.append((label, old, new))
        cu = once(cu, old, new)
    patch('parameters', 'float* sspart;\n};', 'float* sspart;\nfloat* hacc;\nint* htok;\n};')
    patch('private WG scratch', 'int wglast[4];', 'int wglast[4];\nfloat hred[4][16];\nint hlast[4];')
    patch('expert-completion helper', 'struct WgsPend { int kind, u, slot, nst; };', COMPLETE+'\nstruct WgsPend { int kind, u, slot, nst; };')
    for name in ('wgs_flush', 'wgs_flush_end'):
        start = '__device__ __forceinline__ void '+name+'('
        a = cu.index(start)
        b = cu.index('\nstruct WgsEpi' if name == 'wgs_flush' else '\ntemplate <bool PB, int NTT>', a)
        old = cu[a:b]
        new = once(old,
          'if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.mfl + MF_CE + pd.slot), "r"(1) : "memory");',
          'hpf_complete(p, S, S.jE[j]);')
        new = once(new,
          'if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), "r"(1) : "memory");',
          'hpf_complete(p, S, S.jE[j]);')
        patch(name+' full/split publication', old, new)
    patch('zero-local output before B', 'route(p, S);\n__threadfence_block();', '''route(p, S);
// HPF zero-local tokens have no completion owner; their prologue owner writes 0.
for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
 int n = 0;
 #pragma unroll
 for (int sl = 0; sl < 8; ++sl) n += S.pos[tk * 8 + sl] >= 0;
 if (n == 0) {
  for (int c = tid; c < HD; c += NT) p.out[(size_t)tk * HD + c] = __float2bfloat16(0.f);
 }
}
__threadfence_block();''')
    a = cu.index('{\nstatic_assert(HD == NT * 8, "8 columns per thread");')
    b = cu.index('DBG(5);', a)
    patch('remove final token pass', cu[a:b], '// HPF: all nonempty outputs were finalized at expert completion.\n')
    patch('scratch bindings', 'p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;', '''static_assert(OKC_MFIN && OKC_SC6 && OKC_PLAN, "HPF requires smp");
TORCH_CHECK((p.M == 128 || p.M == 256) && scratch.size() == 5, "HPF shape/scratch");
TORCH_CHECK(scratch[3].scalar_type() == torch::kFloat32 && scratch[3].numel() >= okc::MMAX * okc::HD, "HPF accumulator");
TORCH_CHECK(scratch[4].scalar_type() == torch::kInt32 && scratch[4].numel() >= okc::MMAX, "HPF token counters");
p.hacc = (float*)scratch[3].data_ptr();
p.htok = (int*)scratch[4].data_ptr();
p.dbg = nullptr;''')
    assert 'for (int tk = blockIdx.x;' in cu and 'float ps[8];' not in cu
    assert cu.count('hpf_complete(p, S, S.jE[j]);') == 4
    assert '__trap' not in cu
    return cu, edits


PYTHON = '''# HPF is read at call/capture time. Set MOEXR=True for smp controls.
HPF = False
_HPF = {t: dict(_HYB, mod=None, mod7=None, ok=False, sc=None, full=None,
               attempted=False, checks=[], err=None, pdl=-1, pdl7=-1) for t in (128,256)}
def _hpf_flags(t):
 return list(HYB_FLAGS)+(list(HYB_F7)if t==256 else[])+['-DOKC_SC6=1','-DOKC_MFIN=1','-DOKC_PLAN=1','-DOKC_HPF=1']
def _hpf_build(t,H):
 if H['attempted']:
  return
 from torch.utils.cpp_extension import load_inline
 H['attempted']=True
 try:
  fl=_hpf_flags(t)
  tag=hashlib.sha1((_HYB_MOE_CPP+_HPF_CUDA+' '.join(fl)).encode()).hexdigest()[:12]
  start=time.time()
  H['mod']=load_inline(name=f'hyb_hpf_{tag}',cpp_sources=_HYB_MOE_CPP,cuda_sources=_HPF_CUDA,
   functions=['okc_run','okc_set_pdl'],extra_cuda_cflags=fl,verbose=False)
  H['build_s']=time.time()-start
 except Exception as ex:
  H['mod']=None
  H['err']=repr(ex)[:500]
def _hpf_scratch(dev):
 sc,full=_moexr_scratch(dev,True)
 for n,dt in ((_HYB_MMAX*4096,torch.float32),(_HYB_MMAX,torch.int32)):
  f=torch.zeros(n+_HYB_GUARD,dtype=dt,device=dev)
  f[n:].view(torch.int32).fill_(0x5EADBEEF)
  sc.append(f[:n])
  full.append((f,n,torch.int32,0x5EADBEEF))
 return sc,full
def _hpf_cases(t,dev):
 for name in ('normal','skewed','conc69','eight_local','none','one'):
  d=_k2a_self_test_inputs(t,8461+t,dev,name if name in ('normal','skewed')else 'normal')
  if name not in ('normal','skewed'):
   rows=[]
   for tk in range(t):
    rows.append(list(range(8))if name=='eight_local'else
      ([0]+[-1]*7 if name=='one' or(name=='conc69' and tk<69)else [-1]*8))
   d['topk_ids']=torch.tensor(rows,dtype=torch.int32,device=dev)
  yield name,{'hidden_states':d['hidden_states'].contiguous(),'topk_ids':d['topk_ids'].to(torch.int32).contiguous(),
   'topk_weights':d['topk_weights'].float().contiguous(),'config':{'local_start':0,'local_count':K2A_E_LOC}}
def _hpf_clean(H):
 return (_moexr_guards(H) and int(H['sc'][0][8+1024+_HYB_EMAX+65])==0
  and bool((H['sc'][3]==0).all()) and bool((H['sc'][4]==0).all()))
def _hpf_compare(a,b,ref,ship,ship2,control):
 tot,row=_k2a_rel_err(a,ref)
 ev,er=_k2a_rel_err(a,ship)
 rp,_=_k2a_rel_err(b,a)
 sr,_=_k2a_rel_err(ship2,ship)
 ce,cr=_k2a_rel_err(a,control)
 finite=bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all())
 ok=finite and tot<1e-2 and row<5e-2 and ev<2e-3 and er<1e-2 and ce<2e-3 and cr<1e-2 and rp<1e-3
 return dict(ok=bool(ok),fp32=tot,row=row,ship=ev,ship_row=er,smp=ce,smp_row=cr,repeat=rp,ship_repeat=sr,finite=finite)
def _hpf_prepare(st,w,dev):
 # Accessor at the hook is copied verbatim from MOEXR / padz. st is the real inner state.
 for t,H in _HPF.items():
  H['ok']=False
  H['checks']=[]
  if not HYB_ENABLE or not _HYB['ok'] or not _MOEXR_SHIP['ok']:
   H['err']='smp / ship control unavailable'
   continue
  try:
   _hpf_build(t,H)
   if H['mod'] is None:
    continue
   H['w'],H['nsm']=_HYB['w'],_HYB['nsm']
   H['sc'],H['full']=_hpf_scratch(dev)
   good=True
   for name,d in _hpf_cases(t,dev):
    x,ids,tw=d['hidden_states'],d['topk_ids'],d['topk_weights']
    ref=_k2a_ref_fp32(d,w)
    ship=_moexr_call_bank(_MOEXR_SHIP,x,ids,tw).clone()
    ship2=_moexr_call_bank(_MOEXR_SHIP,x,ids,tw).clone()
    control=_moexr_call_bank(_HYB,x,ids,tw).clone()
    a=_moexr_call_bank(H,x,ids,tw).clone()
    b=_moexr_call_bank(H,x,ids,tw).clone()
    torch.cuda.synchronize()
    row=_hpf_compare(a,b,ref,ship,ship2,control)
    row.update(case=name,graph=False,clean=bool(_hpf_clean(H)))
    good &= row['ok'] and row['clean']
    H['checks'].append(row)
    # Capture exercises the same bank/launch body; eight launches cover both
    # MFIN parities repeatedly, without host resets between graph replays.
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     gout=_moexr_call_bank(H,x,ids,tw)
    for replay in range(8):
     graph.replay()
     torch.cuda.synchronize()
     grow=_hpf_compare(gout,a,ref,ship,ship2,control)
     grow.update(case=name,graph=True,replay=replay,clean=bool(_hpf_clean(H)))
     H['checks'].append(grow)
     good &= grow['ok'] and grow['clean']
   H['ok']=bool(good)
   if not H['ok']:
    H['err']='HPF numerical / graph / scratch gate failed'
   else:
    H['err']=None
  except Exception as ex:
   H['ok']=False
   H['err']=repr(ex)[:500]
  print(f"[hpf] T={t} {'LIVE' if H['ok'] else 'OFF'} {H['err']} checks={len(H['checks'])}",flush=True)
def _hpf_bank(m):
 if HPF and MOEXR and m in _HPF:
  H=_HPF[m]
  if not H['ok']:
   raise RuntimeError(f"HPF T={m} disabled: {H['err']}")
  return H
 return _moexr_bank()
'''


def generate(base):
    cu, _ = cuda_source(literal(base, '_HYB_MOE_CUDA'))
    assert 'HPF = ' not in base and '"""' not in cu
    s = once(base, 'def _moexr_bank():', '_HPF_CUDA=r"""'+cu+'"""\n'+PYTHON+'\ndef _moexr_bank():')
    s = once(s, 'return _moexr_call_bank(_moexr_bank(),x,ids,wts)', 'return _moexr_call_bank(_hpf_bank(x.shape[0]),x,ids,wts)')
    s = once(s, 'def _hyb_on(m):\n H=_moexr_bank()', 'def _hyb_on(m):\n H=_hpf_bank(m)')
    hook = '  _moexr_ship_prepare(state["k2a33"]["st"],{k:weights[k]for k in("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")}|{"post_norm_w":weights["post_expert_norm"]},weights["w13"].device)'
    s = once(s, hook, hook+'\n'+hook.replace('_moexr_ship_prepare(', '_hpf_prepare('))
    compile(s, str(LAB), 'exec')
    return s


def source_gates(base, source):
    original = ast.parse(base).body
    new = ast.parse(source).body
    def key(n):
        if isinstance(n,ast.FunctionDef): return n.name
        if isinstance(n,ast.Assign): return ','.join(ast.dump(t) for t in n.targets)
        return ast.dump(n)
    additions = [n for n in new if key(n) not in {key(o) for o in original}]
    kept = [n for n in new if n not in additions]
    assert len(kept)==len(original)
    changed=[]
    for a,b in zip(original,kept):
        if ast.dump(a)!=ast.dump(b): changed.append(key(a))
    assert changed==['_hyb_call','_hyb_on','prepare'],changed
    for name in ('_HYB_MOE_CUDA','_HYB_MOE_CPP','HYB_FLAGS','HYB_F7'):
        assert literal(base,name)==literal(source,name)
    assert literal(source,'HPF') is False
    cu,edits=cuda_source(literal(base,'_HYB_MOE_CUDA'))
    assert cu==literal(source,'_HPF_CUDA')
    # Reverse every exact CUDA anchor; this also certifies no weight-loop edit.
    reverse=cu
    for _,old,newtext in reversed(edits): reverse=once(reverse,newtext,old)
    assert reverse==literal(base,'_HYB_MOE_CUDA')
    # Lexical delimiter gate catches malformed anchor assembly before NVCC.
    def delimiters(text):
        lex=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"', '', text)
        stack=[]
        for c in lex:
            if c in '([{':stack.append(c)
            elif c in ')]}':assert stack and stack.pop()=='([{'[')]}'.index(c)]
        return stack
    # Existing #if alternatives have two redundant lexical open braces.
    assert delimiters(cu)==delimiters(literal(base,'_HYB_MOE_CUDA'))
    assert not delimiters(COMPLETE)
    return dict(status='PASS',changed_functions=changed,new_hpf_builds=2,
                unchanged_cuda_and_flags=True,weight_loops_identical=True,
                sha256=hashlib.sha256(source.encode()).hexdigest(),bytes=len(source.encode()))


def protocol_gate():
    """Event model: partial B tiles, multi-job experts, arbitrary finalize order,
    tokens with 0..8 routes, duplicate slots, and repeated launches. CPU only.
    """
    rng=random.Random(8461)
    cases=0
    for t in (128,256):
        for kind in ('random','none','one','eight','conc69','duplicates'):
            ids=[[(-1 if kind=='none' else 0 if kind=='one' else
                   sl if kind=='eight' else 0 if kind=='duplicates' else
                   (0 if tk<69 and sl==0 else -1) if kind=='conc69' else
                   rng.choice([-1]*7+list(range(32)))) for sl in range(8)] for tk in range(t)]
            if kind=='one': ids=[[0]+[-1]*7 for _ in range(t)]
            routes={e:[(tk,sl) for tk in range(t) for sl in range(8) if ids[tk][sl]==e] for e in range(32)}
            acc=[0]*t; counters=[0]*t; ce=[[0]*32 for _ in range(2)]
            for replay in range(6):
                gen=replay%2; ce[1-gen]=[0]*32
                events=[]; totals={}
                for e,rs in routes.items():
                    if not rs: continue
                    totals[e]=64*((len(rs)+63)//64 if len(rs)>8 else 1)
                    events += [(e,k) for k in range(totals[e])]
                rng.shuffle(events)
                ready=[]; published=[sum(v>=0 for v in row)==0 for row in ids]
                done=set()
                # Finalizer rows may be interleaved independently of B tickets.
                while events or ready:
                    if events and (not ready or rng.randrange(3)):
                        e,k=events.pop();done.add((e,k));ce[gen][e]+=1
                        if ce[gen][e]==totals[e]:
                            assert all((e,k) in done for k in range(totals[e]))
                            ready.extend((e,tk,sl) for tk,sl in routes[e])
                    else:
                        e,tk,sl=ready.pop(rng.randrange(len(ready)))
                        assert not published[tk]
                        acc[tk]+=1;counters[tk]+=1
                        need=sum(v>=0 for v in ids[tk])
                        if counters[tk]==need:
                            assert acc[tk]==need
                            published[tk]=True;acc[tk]=0;counters[tk]=0
                assert all(published) and not any(acc) and not any(counters)
                cases+=1
    return dict(status='PASS',replays=cases,model_only=True)


def numerical_gate():
    """CPU arithmetic model of the unchanged RMS tree and changed accumulation.
    Not a CUDA/H200 accuracy test. Check the absolute bound under cancellation.
    """
    import numpy as np
    rng=np.random.default_rng(8461)
    u=2.**-24;gamma=8*u/(1-8*u);tiny=np.finfo(np.float32).tiny
    worst=0.;cases=0
    for t in (128,256):
        for kind in ('normal','cancel','tiny'):
            y=rng.standard_normal((8,4096)).astype(np.float32)
            if kind=='cancel':y[1::2]=-y[::2]
            if kind=='tiny':y*=np.float32(1e-42)
            def tree(v):
                v=v.reshape(8,512,8)
                ps=(v[:,:,0]*v[:,:,0]).astype(np.float32)
                for c in range(1,8):ps=(ps+v[:,:,c]*v[:,:,c]).astype(np.float32)
                ps=ps.reshape(8,16,32)
                for d in (16,8,4,2,1):ps=(ps+ps[:,:,np.arange(32)^d]).astype(np.float32)
                tot=np.zeros(8,dtype=np.float32)
                for w in range(16):tot=(tot+ps[:,w,0]).astype(np.float32)
                return tot
            base=tree(y)
            # HPF's h*128+wt maps each virtual thread to the same eight columns.
            reconstructed=np.empty_like(y)
            for h in range(4):
                for wt in range(128):
                    c=(wt+h*128)*8;reconstructed[:,c:c+8]=y[:,c:c+8]
            assert np.array_equal(base,tree(reconstructed))
            r=(np.float32(.125)/np.sqrt(base*np.float32(1/4096)+np.float32(1e-8))).astype(np.float32)
            terms=r.astype(np.float64)[:,None]*y.astype(np.float64)
            ship=np.zeros(4096,dtype=np.float32)
            for k in range(8):ship=(terms[k]+ship.astype(np.float64)).astype(np.float32)
            for _ in range(8):
                order=rng.permutation(8);out=np.zeros(4096,dtype=np.float32)
                for k in order:
                    term=terms[k].astype(np.float32)
                    term[np.abs(term)<tiny]=0 # global FP32 RED flushes subnormals
                    out=(out+term).astype(np.float32)
                    out[np.abs(out)<tiny]=0
                delta=np.abs(out.astype(np.float64)-ship.astype(np.float64))
                bound=2*gamma*np.sum(np.abs(terms),axis=0)+16*tiny
                assert np.all(delta<=bound),kind
                worst=max(worst,float(np.max(delta/np.maximum(bound,tiny))));cases+=1
    return dict(status='PASS',cpu_cases=cases,worst_bound_fraction=worst,
                bound='2*gamma8*sum(abs(r*y)) + 16*FLT_MIN; before BF16',device_validation=False)


def dry_run(base, source):
    """Execute all new prepare Python, strict bank layouts and failure branches.
    Tensor/CUDA arithmetic is doubled; missing dictionary keys are real errors.
    """
    helpers=runpy.run_path(str(ROOT/'tools/moexr_patches.py'))
    C=helpers['C']; Meta=C['Meta']; DT=C['DT']; torch=C['meta_torch']()
    env={'torch':torch,'time':time,'hashlib':hashlib,'__builtins__':dict(vars(builtins))}
    names={'_HYB','_MOEXR_SHIP','MOEXR','_HYB_MOE_CPP','_HYB_MMAX','_HYB_EMAX','_HYB_FIXF','_HYB_XROWS',
           '_HYB_GUARD','_HYB_SLOTS','_hyb_scratch','_moexr_scratch','_moexr_guards',
           'HYB_FLAGS','HYB_F7','HYB_T7','HYB_T','HYB_PDL','HYB_ENABLE','K2A_E_LOC',
           '_moexr_bank','_moexr_call_bank','_hyb_call','_hyb_on','HPF','_HPF','_HPF_CUDA'}
    names|={n.name for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name.startswith('_hpf_')}
    exec(nodes(source,names),env)
    builds=[];launches=[];input_rows=[];fault={'build':False,'numeric':False,'dirty':False,'guard':False,'nonfinite':False,'reference':False}
    class Mod:
        def __init__(self,t=0): self.t=t
        def okc_set_pdl(self,pdl): assert pdl in (0,1)
        def okc_run(self,w,sc,x,ids,tw,out,lo,nloc,grid):
            assert len(w)==14 and (lo,nloc,grid)==(0,32,132)
            assert x.shape==out.shape and ids.shape==tw.shape==(x.shape[0],8)
            assert len(sc)==(5 if self.t else 3)
            if self.t:
                assert x.shape[0]==self.t
                assert sc[3].numel()==384*4096 and sc[4].numel()==384
                out.tag='bad' if fault['numeric'] else None
            launches.append((self.t,x.shape[0]))
    def compiler(**kw):
        t=256 if '-DOKC_HYBST=7' in kw['extra_cuda_cflags'] else 128
        builds.append(t)
        assert kw['cuda_sources']==literal(source,'_HPF_CUDA')
        assert kw['extra_cuda_cflags']==env['_hpf_flags'](t)
        if fault['build']: raise RuntimeError('injected compiler failure')
        return Mod(t)
    orig_import=builtins.__import__
    env['__builtins__']['__import__']=lambda name,*a,**k:SimpleNamespace(load_inline=compiler) if name=='torch.utils.cpp_extension' else orig_import(name,*a,**k)
    def tensor(rows,dtype,device):
        input_rows.append(rows)
        return Meta((len(rows),len(rows[0])),dtype)
    torch.tensor=tensor
    env['_k2a_self_test_inputs']=lambda t,seed,dev,dist:{'hidden_states':Meta((t,4096),DT.bfloat16),'topk_ids':Meta((t,8),DT.int32),'topk_weights':Meta((t,8))}
    def reference(d,w):
        if fault['reference']: raise RuntimeError('injected reference error')
        return Meta(d['hidden_states'].shape)
    env['_k2a_ref_fp32']=reference
    env['_k2a_rel_err']=lambda a,b:(1.,1.) if a.tag=='bad' else (0.,0.)
    torch.isfinite=lambda a:Meta(a.shape,value=not fault['nonfinite'])
    # Real allocation and real clean predicate execute. Only view aliasing is
    # represented by metadata; verify the sentinel predicate independently.
    env['_moexr_guards']=lambda H:not fault['guard'] and all(f.numel()==n+env['_HYB_GUARD'] for f,n,vt,sv in H['full'])
    clean=env['_hpf_clean']
    env['_hpf_clean']=lambda H:clean(H) and not fault['dirty']
    for key in ('_HYB','_MOEXR_SHIP'):
        H=env[key];H.update(ok=True,w=list(range(14)),sc=[Meta(),Meta(),Meta()],nsm=132,mod=Mod(),mod7=Mod())
    # Execute the real constructor and prepare hooks using the proven MOEXR
    # integration harness, then execute HPF's hook with that exact inner object.
    with contextlib.redirect_stdout(io.StringIO()):
        baseline=helpers['dry_run'](base,base,False)
        integration=helpers['integration_dry_run'](base,source)
    diet=runpy.run_path(str(ROOT/'tools/diet_patches.py'))
    denv,_=diet['cpu_namespace'](base);ft=diet['FakeTensor']
    denv['_k2a33_build_ext']=lambda *a:(SimpleNamespace(attrs3v=lambda i:[0,0,0,2,0,0,0,2]),0)
    mw={k:ft((32,4096),'bf16') for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs','post_norm_w')}
    with contextlib.redirect_stdout(io.StringIO()):
        inner=denv['_k2a33_prepare_st'](mw,share={'ws13':ft(),'ws2':ft()})
    weights={k:mw[k] for k in mw if k!='post_norm_w'}|{'post_expert_norm':mw['post_norm_w']}
    hook=next(n for n in ast.walk(ast.parse(function(source,'prepare'))) if isinstance(n,ast.Expr)
              and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_hpf_prepare')
    hookcode=compile(ast.Module(body=[hook],type_ignores=[]),'<real-hpf-hook>','exec')
    actual=env['_hpf_prepare'];seen=[]
    def checked(st,w,dev):
        assert st is inner and set(w)==set(mw)
        seen.append(True);return actual(st,w,dev)
    env['_hpf_prepare']=checked
    env['weights']=weights
    def prep():
        with contextlib.redirect_stdout(io.StringIO()):
            exec(hookcode,env,{'state':{'k2a33':{'st':inner}},'weights':weights})
    prep()
    assert all(H['ok'] for H in env['_HPF'].values()), [H['err'] for H in env['_HPF'].values()]
    assert builds==[128,256]
    assert all(len(H['checks'])==54 for H in env['_HPF'].values())
    assert any(sum(row[0]==0 for row in rows)==69 for rows in input_rows)
    assert any(len(rows)==256 and all(row==list(range(8)) for row in rows) for rows in input_rows)
    for moexr in (False,True):
        env['MOEXR']=moexr
        for hpf in (False,True):
            env['HPF']=hpf
            for t in (1,8,16,32,49,128,192,193,256,320,384):
                H=env['_hpf_bank'](t)
                assert H is (env['_HPF'][t] if moexr and hpf and t in (128,256) else env['_HYB' if moexr else '_MOEXR_SHIP'])
                assert bool(env['_hyb_on'](t))==(49<=t<=320)
                if 49<=t<=320:
                    env['_hyb_call'](Meta((t,4096)),Meta((t,8)),Meta((t,8)))
    for key in ('numeric','dirty','guard','nonfinite','reference'):
        fault[key]=True;prep()
        assert all(not H['ok'] for H in env['_HPF'].values())
        env.update(MOEXR=True,HPF=True)
        for t in (128,256):
            try: env['_hpf_bank'](t)
            except RuntimeError: pass
            else: raise AssertionError('failed arm was enabled')
        fault[key]=False
    assert builds==[128,256], 'prepare rebuilt modules'
    env['_HYB']['ok']=False;prep();assert all(not H['ok'] for H in env['_HPF'].values())
    env['_HYB']['ok']=True;env['_MOEXR_SHIP']['ok']=False;prep()
    env['_MOEXR_SHIP']['ok']=True;env['HYB_ENABLE']=False;prep()
    env['HYB_ENABLE']=True;fault['build']=True
    for H in env['_HPF'].values():H.update(attempted=False,mod=None)
    prep();prep();assert builds==[128,256,128,256]
    assert all(not H['ok'] for H in env['_HPF'].values())
    try: exec(hookcode,env,{'state':{'k2a33':{}},'weights':weights})
    except KeyError as ex: assert ex.args==('st',)
    else: raise AssertionError('missing st was not detected')
    return dict(status='PASS',baseline=baseline['status'],integration=integration,
                builds=builds,mock_launches=len(launches),real_hook_calls=len(seen),
                paths=['build and reuse','scratch allocation','all six routing cases','eager/graph validation',
                       'selector cross product','numeric/dirty/guard/nonfinite/reference failures','missing st','disabled HYB/ship','compile failure'],
                limitation='CUDA/ref leaves and graph replay are doubles, not GPU correctness evidence')


def cpu_gates(base,source):
    return dict(source=source_gates(base,source),prepare=dry_run(base,source),protocol=protocol_gate(),numerics=numerical_gate())


def report(base,source,gates):
    # Filled below with the source-derived report; keep deterministic output.
    return report_text(base,source,gates)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
    base=BASE.read_text(encoding='utf-8');source=generate(base);gates=cpu_gates(base,source)
    if args.check:
        assert LAB.read_bytes()==source.encode(),'stale lab'
        assert REPORT.read_bytes()==report(base,source,gates).encode(),'stale report'
    else:
        LAB.write_bytes(source.encode());REPORT.write_bytes(report(base,source,gates).encode())
    print(json.dumps(gates,indent=2))


def report_text(base,source,gates):
    def line(s,needle):
        assert needle in s,needle
        return s[:s.index(needle)].count('\n')+1
    cu=literal(source,'_HPF_CUDA');oldcu=literal(base,'_HYB_MOE_CUDA')
    offset=line(source,'_HPF_CUDA=r"""')-1
    def cl(needle):return offset+line(cu,needle)
    hunks=[]
    oldoffset=next(n.lineno for n in ast.parse(base).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_HYB_MOE_CUDA' for t in n.targets))-1
    for op,a,b,c,d in difflib.SequenceMatcher(a=oldcu.splitlines(),b=cu.splitlines(),autojunk=False).get_opcodes():
        if op=='insert':hunks.append(f'insert before base CUDA {oldoffset+a+1} -> HPF CUDA {offset+c+1}-{offset+d}')
        elif op!='equal':hunks.append(f'{op}: base CUDA {oldoffset+a+1}-{oldoffset+b} -> HPF CUDA {offset+c+1}-{offset+d}' if d>c else f'{op}: base CUDA {oldoffset+a+1}-{oldoffset+b} removed before HPF line {offset+c+1}')
    ranges=[]
    for n in ast.parse(source).body:
        if isinstance(n,ast.FunctionDef) and (n.name.startswith('_hpf_') or n.name in ('_hyb_call','_hyb_on')):
            ranges.append(f'{n.name}:{n.lineno}-{n.end_lineno}')
    hookline=line(source,'  _hpf_prepare(state["k2a33"]["st"]')
    return f'''| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |
|---|---|---|---|---|---|---|
| HPF progressive finalize | T=128 | LAD L=154.524; L→LF=3.640, VOID controls (box_out/lad/summary.md:7-13); current smp absolute time unavailable | Net unknown; no positive timing claim | Can overlap completed experts with trailing B; last expert's finalization remains exposed, and may stall the weight ring | R-hybst-2; R-mg-6 | BUILT; HPF=False; GPU qualification pending |
| HPF progressive finalize | T=256 | LAD L=192.216; L→LF=6.244 (box_out/lad/summary.md:21-27), board diagnostic only | Net unknown; mod7 already reuses y loads | Removes the final all-token pass; no additional grid barrier or launch removed from MFIN | R-hybst-2; conserved coupling model | BUILT; HPF=False; GPU qualification pending |

Kill reasons first: (1) a hot expert's last WG serializes up to all its routes, and its paused ring consumer can delay unfinished B; (2) accumulator atomics/fences and live-register/call-frame growth can erase the tail reduction. These are qualification risks, not reasons to refuse the explicitly requested build.

WEIGHT STREAM: NO GROWTH. EST source accounting, before=after: SC6 stage = 4*(8192+768)=35,840 B; A job=32*16*35,840=18,350,080 B; B job=16*16*35,840=9,175,040 B. All stage/weight-loop source is identical. Only B-completion publication and finalize change. The original CUDA/flags remain byte-identical and the original module hashes are unchanged.

Exact scored routes: HPF can select only T128/T256 (both sliding/full). At T128/T256, MOEXR=True, HPF=False uses the original _HYB mod/mod7 through base _moexr_call_bank at kernels/lab/k3_moexr_ab.py:{line(base,'def _moexr_call_bank(')}. HPF=True uses _HPF[128]/_HPF[256] (SC6=1/MFIN=1/PLAN=1, T256 also HYBST=7/HN=1), both moe_fused<true> at kernels/lab/k3_hpf_ab.py:{cl('if (p.M >= OKC_BIGM) cudaLaunchKernelEx')}. Phase B is wgs_phase<true> -> wgs_seg<true,NTT>, NTT=(S.jC[j]+7)>>3, cases 1..8 at {cl('switch (ntt)')}; routing determines each NTT, not batch size. The full/split completion paths in wgs_flush<true> and wgs_flush_end<true> now call hpf_complete. The stage math/copy body is unchanged; its completion call graph gains the operations below. Exact compiled counts/resources await NVCC.

Mechanism at kernels/lab/k3_hpf_ab.py:{cl('__device__ __forceinline__ void hpf_complete')}: a WG leader atom.acq_rel increments the existing parity/expert counter; target=64*ceil(cnt/64) for cnt>8, otherwise64. This covers 16 B column units*4 WGs for EVERY token chunk. The elected WG processes S.off[e]..S.off[e]+S.cnt[e], using S.ent[q] for token/slot. No partially complete tile is normalized. The B epilogue already applies (1+post_norm_w); HPF reads that ybuf and does not apply it again.

Finalize payload (EST source counts, not measured HBM transactions): let B=4096*4=16,384 B, R=real local routes, A=tokens with local routes. Excluding identical BF16 output 8192*T B, base128=2BR, base256=BR; HPF=BR y reads+BR FP32 RED operands+BA accumulator reads+BA zero stores. Delta128=2BA, delta256=BR+2BA. For illustrative R=A=T, additional payload is 4,194,304 / 12,582,912 B at T128/256. Accumulator/counter traffic is scratch/L2 traffic, NOT weight-stream growth; internal atomic read-modify-write traffic is additional. No HBM-rate conversion or net saving is asserted because residency/exposure are unmeasured. No PDL-edge or flag-hop saving is credited.

Scratch (EST source bytes): each HPF bank adds 384*4096*4=6,291,456 B FP32 accumulation plus 384*4=1,536 B token counters, each with 65,536 guard elements; MFIN and existing y/split buffers are isolated per bank. New shared storage=4*16*4+4*4=272 B before 128-byte xbuf alignment (total dynamic-smem increase 256 or384 B). No weight repack or stage byte change.

Instruction accounting below is EST source-level operations per participating thread; instruction fusion, vectorization and actual SASS counts remain unmeasured. Original finalizer: 512 threads/row/route, 2 LDG.128 (128 re-reads another2; 256 retains), eight square products/seven adds, five SHFL+five adds; lane0 stores a warp sum, one reducer does16 shared loads/adds; each thread computes scale and eight FMA-like combines. HPF: 128 threads/row/route, eight LDG.128, 32 square products/28 adds, 20 SHFL+20 adds, lane0 writes four shared sums; wt0 performs the SAME ordered16-warp sum and one scale; each thread adds32 FMUL+32 RED, one acquire-release fence, and five WG barriers per route. Total row arithmetic for RMS is unchanged; scale is shared. Each expert completion adds one leader acq_rel ATOM, one shared flag write and one128-thread barrier (replacing one leader release RED); only the elected WG drains the one possibly in-flight WGMMA group with wg_wait<0> before its noinline finalizer call, and adds an acquire fence. This avoids call-ABI saves of live asynchronous accumulator registers. Token publication adds one leader ATOM, eight local-slot tests, flag store. Last-token threads add eight LDG.128, eight zero STG.128, 16 paired BF16 conversions/four output STG.128, one acquire fence and one WG barrier; leader resets one counter. Base output is four paired conversions/one STG.128 per512-thread token plus its final CTA barrier. The zero-local prologue adds eight shared slot tests/thread/token; only empty tokens write eight BF16 zeros/thread. No final token loop remains.

Protocol proof: all CTAs' compute threads0..511 still execute other-parity cleanup BEFORE B: i=CTA*512+tid, stride=G*512 covers32 expert counters+96 xready+JMAX*16 doneAs (JMAX=88, total1536 ints, CTAs0..2 at G=132). Helper threads512..543 and544..575 retain their original producer exits and own no cleanup. No compute CTA exits early and no grid barrier is added. After the new wg_wait<0>, all128 lanes continue into finalization; none exits or skips any split reset. Other WGs continue independently.

Post-wait split cleanup is retained in BOTH flush variants: for WG w, threads128w..128w+127 (wt0..127) load/combine/reset floats4*wt..4*wt+3 per nt at stride512, all nt0..NTT-1; wt0 resets fc before publishing B completion. Phase-A flags and split rules are unchanged. The elected expert WG covers every q in that expert, every column via (wt+h*128)*8+c for h0..3,c0..7. Unchanged B stores -> WG barrier -> acq_rel expert counter chain -> WG barrier/acquire publishes all rows, including cross-CTA column contributions.

Token cleanup/combine: every contributing WG's128 lanes issue32 reductions each, then each lane fences, then WG barrier, then leader acq_rel token increment. Target counts ALL valid local slots (including duplicate expert slots); no local experts means no ticket. The last WG acquires and reads all4096 columns, writes BF16, zeroes all4096 accumulator elements, barriers, and wt0 zeroes the token counter. Earlier contributors have finished every global reduction before their tickets; none reads/writes the accumulator after its ticket, so last-owner zeroing cannot race them. A final WG barrier protects hred/hlast reuse. Counter/data reset completes before kernel completion; MFIN parity and separate banks remain graph-safe under the base's ordered-stream/nonconcurrent scratch contract. Zero-local output owners: CTA=tk%G, threads0..511, columns tid+512*k,k0..7, in the prologue. There is no new polling wait or __trap. Existing split waits are unchanged.

Numerics: four virtual128-thread groups preserve the original512-thread column grouping, shuffle tree and ordered16-warp RMS sum. For identical finite normal scales, u=2^-24, gamma8=8u/(1-8u), absolute FP32 order/product difference <=2*gamma8*sum(abs(r*y)); global FP32 RED's subnormal flushing adds conservative16*FLT_MIN per component. This is an absolute bound before BF16, not a relative guarantee near cancellation; BF16 conversion adds each result's half-ulp rounding and can change adjacent BF16 values. Actual compiler RMS equivalence and all full-MoE errors must pass prepare.

Prepare gates BUILT, NOT GPU-RUN: for each T128/256, normal/skewed/one-local/no-local/69-token-single-expert/eight-local routes (all eight experts have128/256 tokens), compare against real _k2a_ref_fp32, stock _MOEXR_SHIP, and smp _HYB; FP32 total<1e-2,row<5e-2; ship/smp total<2e-3,row<1e-2; repeats<1e-3; finite and all scratch/guards clean. Each case captures and checks eight graph replays without host resets. Each bank records54 check rows. Any failure disables that shape; requesting a disabled HPF arm raises visibly instead of timing fallback. Control modules and scratch remain available with HPF=False; MOEXR=False never selects HPF.

CPU RUN/PASS: in-memory Python compile; exact-anchor/reverse-patch/AST/lexical-delimiter checks; original MOEXR prepare dry run; real base state constructor and exact nested hook; all new allocation/build/case/compare/selector paths, failure injections and capture bodies (CUDA/graph leaves doubled);72 shuffled protocol replays including multi-job/duplicate experts;48 CPU arithmetic order/cancellation/subnormal cases, worst bound fraction={gates['numerics']['worst_bound_fraction']:.6f}. These do NOT certify device memory ordering or accuracy. Lab bytes={len(source.encode()):,}; SHA256={gates['source']['sha256']}.

Per-bit changes (only bit HPF; no unrelated bits): added _HPF_CUDA at lab line{offset+1} through{offset+len(cu.splitlines())}; HPF=False at{line(source,'HPF = False')}; bank at{line(source,'_HPF = {t:')}; actual state hook at{hookline}. Exact CUDA delta intervals: {'; '.join(hunks)}. Python intervals in kernels/lab/k3_hpf_ab.py: {'; '.join(ranges)}. Deleted-token-pass lines refer to the original base; the control copy remains unchanged in the lab.

Not run: NVCC, cuobjdump, real H200 accuracy, CUDA graphs or timings. WSL is known unavailable in this sandbox and was not attempted. No GPU/pod/judge jobs, git writes, notes edits or secret access. Process deviation: the generator was briefly removed by a replacement patch and immediately recreated; no file remains deleted.

Lead command (LF bash/Python polyglot; CPU and offline compilation only):
```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" bash "<repo>/tools/local/hpf_compile.py"
```
It adds cuobjdump's requested directory to PATH, runs every CPU gate, compiles base/lab ship0/ship7/smp0/smp7 plus HPF0/HPF7, compares every unchanged HYB control's exact SASS/resource maps, records both BIG kernel instances and the HPF helper's registers/spills/stack, and emits tools/local/hpf_compile.json next to the script. Resource gate: kernel<=112 registers, no kernel spill growth, zero helper spills; noninline ABI stack frames are recorded, not hidden. All other original CUDA literals/flags are AST-identical and retain their cached modules; they are not separately SASS-recompiled. Temporary compiler artifacts remain under /tmp; nothing is deleted. JSON is a requested future lead-run output and was not created in this session.
'''


if __name__=='__main__':main()

#!/usr/bin/env python3
"""BT4b: TINY/TINY2/VMS timing-only intercept probes; no output-null arm is a model.
Timer, gate, flush, ceiling, module loading copied verbatim from moephase.py.
"""
import argparse
import ast
import contextlib
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics as stats
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SHAPES = (1, 8, 16)
KEYS = ('w13', 'w13_sf', 'w2', 'w2_sf')
GRID, UNIT, HALF = 132, 36864, 18432


CUDA = r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
constexpr int U=36864, H=18432, S=3*U+24;
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void init(unsigned b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(b) : "memory");
}
__device__ __forceinline__ void issue(unsigned dst,const unsigned char* src,unsigned b,int n,unsigned long long pol) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(b),"r"(n) : "memory");
  int a=n<H?n:H, z=n-a;
  if(a) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" :: "r"(dst),"l"(src),"r"(a),"r"(b),"l"(pol) : "memory");
  if(z) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" :: "r"(dst+H),"l"(src+H),"r"(z),"r"(b),"l"(pol) : "memory");
}
__device__ __forceinline__ void wait(unsigned b,int phase) {
  unsigned done;
  do { asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(b),"r"(phase) : "memory"); } while(!done);
}
__global__ void stream_weights(const unsigned char* p,size_t bytes,unsigned* sink) {
  extern __shared__ __align__(128) unsigned char sm[];
  if(threadIdx.x != 0) return;
  unsigned bars=sa(sm+3*U);
  for(int i=0;i<3;i++) init(bars+8*i);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  size_t stages=(bytes+U-1)/U;
  size_t begin=stages*blockIdx.x/gridDim.x, end=stages*(blockIdx.x+1)/gridDim.x;
  size_t count=end-begin; unsigned sum=0;
  for(size_t j=0;j<count+3;j++) {
    if(j>=3 && j-3<count) {
      size_t k=j-3; int slot=k%3;
      wait(bars+slot*8,(k/3)&1);
      sum ^= *reinterpret_cast<volatile unsigned*>(sm+slot*U);
    }
    if(j<count) {
      size_t off=(begin+j)*U; int n=(int)((bytes-off)<U?(bytes-off):U);
      issue(sa(sm+(j%3)*U),p+off,bars+(j%3)*8,n,pol);
    }
  }
  for(int i=0;i<3;i++) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bars+i*8) : "memory");
  sink[blockIdx.x]=sum;
}
extern "C" int mp_init() { return (int)cudaFuncSetAttribute(stream_weights,cudaFuncAttributeMaxDynamicSharedMemorySize,S); }
extern "C" int mp_run(uint64_t p,uint64_t bytes,uint64_t sink,uint64_t stream) {
  stream_weights<<<132,1,S,(cudaStream_t)stream>>>((const unsigned char*)p,(size_t)bytes,(unsigned*)sink);
  return (int)cudaGetLastError();
}
'''

def phase(label, seconds):
    @contextlib.contextmanager
    def manager():
        print('[mp1] ' + label, flush=True)
        def expired(_s, _f):
            raise TimeoutError(label)
        previous = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        started = time.monotonic()
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(seconds, previous_timer[0]) if previous_timer[0] else seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            if previous_timer[0]:
                signal.setitimer(signal.ITIMER_REAL, max(0.001,previous_timer[0]-(time.monotonic()-started)),previous_timer[1])
    return manager()

def touched(ids):
    return sorted({int(i) for i in ids.reshape(-1).tolist() if 0 <= int(i) < 32})

def lengths(n):
    assert n > 0 and n % 16 == 0
    return {'minus4': ((n * 96 // 100) // 16) * 16,
            'base': n, 'plus10': n + ((n // 10) // 16) * 16}

def bank(weights, ids, torch):
    # Byte view preserves the raw E4M3 bit patterns; no numerical conversion.
    pieces = [weights[k][e].contiguous().view(torch.uint8).reshape(-1)
              for e in ids for k in KEYS]
    assert pieces
    b = torch.cat(pieces)
    n = b.numel()
    sizes = lengths(n)
    return torch.cat((b, b[:sizes['plus10'] - n])), sizes

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def build():
    # Temporary build artifacts only; no torch extension or source-tree cache.
    directory = Path(tempfile.mkdtemp(prefix='mp1_'))
    src, lib = directory/'stream.cu', directory/'stream.so'
    src.write_text(CUDA)
    subprocess.run(['nvcc', '-gencode=arch=compute_90a,code=sm_90a', '-O3', '-std=c++17', '-shared',
                    '-Xcompiler', '-fPIC', str(src), '-o', str(lib)], check=True, timeout=90)
    dll = ctypes.CDLL(str(lib))
    dll.mp_init.restype = ctypes.c_int
    dll.mp_run.argtypes = [ctypes.c_uint64]*4
    dll.mp_run.restype = ctypes.c_int
    rc = dll.mp_init()
    if rc: raise RuntimeError('cudaFuncSetAttribute: %d' % rc)
    return directory, dll

class Timer:
    # m16xa_micro.py Timer: same flush, calibrated gate, graph and event order.
    def __init__(self, torch):
        self.torch = torch
        self.stream = torch.cuda.Stream()
        self.flush_buf = torch.ones(256*1024*1024//4, dtype=torch.float32,device='cuda')
        self.sink = torch.empty((),dtype=torch.float32,device='cuda')
        self.gate_cyc = None
    def capture(self,fn):
        t=self.torch
        with t.cuda.stream(self.stream):
            fn(); t.cuda.synchronize()
            g=t.cuda.CUDAGraph()
            with t.cuda.graph(g,stream=self.stream): fn()
        t.cuda.synchronize()
        return g
    def gate(self):
        t=self.torch
        if self.gate_cyc is None:
            cyc=int(300*1980)
            for _ in range(3):
                a,b=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
                a.record(); t.cuda._sleep(cyc); b.record(); b.synchronize()
                cyc=int(cyc*300/max(a.elapsed_time(b)*1e3,1.0))
            self.gate_cyc=cyc
        t.cuda._sleep(self.gate_cyc)
    def replay(self,g):
        t=self.torch
        with t.cuda.stream(self.stream):
            t.sum(self.flush_buf,dim=0,out=self.sink)
            self.gate()
            a,b=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record()
        b.synchronize()
        return a.elapsed_time(b)*1e3

def percentile(v, p):
    v = sorted(v); x=(len(v)-1)*p; i=int(x)
    return v[i] + (v[min(i+1,len(v)-1)]-v[i])*(x-i)

def summarize(values, overhead):
    raw = stats.median(values)
    return {'raw_us': raw, 'us': raw-overhead, 'min_raw_us': min(values),
            'p90_raw_us': percentile(values,.9), 'min_us': min(values)-overhead,
            'p90_us': percentile(values,.9)-overhead, 'replays_raw_us': values}


# Bit 1 IMPLIES the AB1 waiver without requiring bit 0 to be set.
ARMS = {'base':0, 'AB1':1, 'AB2':2, 'COMBSKIP':4, 'FIN':8,
        'AB1+COMBSKIP':5, 'AB2+FIN':10}
EXPECTED_ROUTES = {1:'tiny', 8:'tiny2', 16:'vms_hi'}
GATES = {'controls':{'twin_us':1.0}, 'AB1_kill':1.2, 'AB1_pass':1.8,
         'AB2_minus_AB1':1.0, 'COMBSKIP':{1:.7,8:1.5,16:1.5},
         'FIN':{1:1.0,8:1.8,16:2.5}}


def disable_pdl(state):
    # Copied from BT4; explicit extra method keeps the new mask API reachable.
    for name in ('tiny','tiny2','hyb','hyb7'):
        state['mods'][name].okc_set_pdl(0)
    class VmsNoPDL:
        def __init__(self, mod): self.mod=mod
        def run(self, *args):
            assert len(args)==18, 'VMS run ABI drift'
            args=list(args); args[-2]=0
            return self.mod.run(*args)
        def bt4b_set_mask(self, v): return self.mod.bt4b_set_mask(v)
    for name in ('vms','vms_hi'):
        state['mods'][name]=VmsNoPDL(state['mods'][name])


def verify_routes(state):
    for bs,name in EXPECTED_ROUTES.items():
        assert state['route'][bs] == name, ('fallback invalidates probe',bs,state['route'][bs],name)


def correctness(layer, state, data):
    """Copied BT4 flag-zero comparator: eager, graph, stock FP32 equation."""
    bs = data['hidden_states'].shape[0]
    name = state['route'][bs]
    ref = layer._k2a_ref_fp32(data,state['w_raw'])
    previous = layer.kernel_fn(data,state).clone()
    layer.torch.cuda.synchronize()
    layer._mbl_compare(previous,ref,data,state,name)
    graph = layer.torch.cuda.CUDAGraph()
    with layer.torch.cuda.graph(graph):
        got = layer.kernel_fn(data,state)
    graph.replay()
    layer.torch.cuda.synchronize()
    layer._mbl_compare(got,ref,data,state,name,previous)
    return {'relative_fp32':float(layer._k2a_rel_err(got,ref)[0])}


def protocol_snapshot(state, bs, mask):
    """Synchronous, UNTIMED. Check every replay's generation and next flag bank."""
    if bs <= 8:
        ctr = state['h_sc'][0]
        gen, nj = int(ctr[3].item()), 88
    else:
        st = state['banks'][0]['k2a33']['st']
        ctr = st['vms']['ctr']
        gen, nj = int(ctr[0].item()), 64
        assert int(ctr[2].item()) == 0, 'VMS bounded wait timed out'
    par = (gen & 1) if mask & 8 else 0
    ready = ctr[8+par*nj:8+(par+1)*nj]
    assert int(ready.count_nonzero().item()) == 0, 'next doneA bank was not cleared'
    if mask & 8:
        assert int(ctr[4+(gen&1)].item()) == 0, 'next exit counter was not cleared'
        assert int(ctr[4+((gen-1)&1)].item()) == state['nsm'], 'lost CTA exit'
    if bs <= 8:
        assert int(ctr[0].item()) == int(ctr[1].item()) == 0
        assert all(bool((f[n:].view(vt)==sv).all()) for f,n,vt,sv in state['h_full']), 'scratch guard'
    return {'generation':gen,'next_bank_zero':True,'mask':mask}


def capture_arms(layer, state, data, timer):
    """Flag set/read BEFORE each graph; 3 consecutive replays of EVERY twin."""
    bs = data['hidden_states'].shape[0]
    graphs, checks, snapshots = {}, {}, {}
    for name,mask in ARMS.items():
        for suffix in ('','_twin'):
            arm = layer.bt4b_arm(state,mask,bs)
            assert int(state['bt4b_flag'].item()) == mask
            timer.torch.cuda.synchronize()  # scratch zeros finish before capture stream
            graph = timer.capture(lambda arm=arm:layer.kernel_fn(data,arm))
            # Capture may execute an eager warmup; only replay increments below
            # are compared. A fresh FIN graph has already warmed once.
            previous = protocol_snapshot(arm,bs,mask)['generation']
            replays = []
            for replay in range(3):
                with timer.torch.cuda.stream(timer.stream): graph.replay()
                timer.torch.cuda.synchronize()
                snap = protocol_snapshot(arm,bs,mask)
                assert snap['generation'] == previous+1, ('generation',name,suffix,replay,snap,previous)
                previous = snap['generation']; replays.append(snap)
            label = name+suffix
            graphs[label] = graph
            checks[label] = replays
            snapshots[label] = mask
            print(f'[bt4b] replay-safe bs={bs} arm={label} flag={mask} x3',flush=True)
    # All captures complete with global host holder at 10. Already-captured
    # flag-zero graph MUST still execute mask0 (kernel-argument snapshot).
    return graphs, {'three_replays':checks,'captured_masks':snapshots}


def measure_graphs(timer, graphs, overhead):
    # Copied verbatim from stage 1: rotated 30 reps, paired seed-median twins.
    values = {a:[] for a in graphs}
    for rep in range(30):
        order = list(graphs)
        order = order[rep % len(order):] + order[:rep % len(order)]
        for arm in order:
            values[arm].append(timer.replay(graphs[arm]))
    times = {a:summarize(v, overhead) for a,v in values.items()}
    twins = {a:stats.median(x-y for x,y in zip(values[a],values[a+'_twin']))
             for a in graphs if not a.endswith('_twin')}
    return {'times':times, 'twins_us':twins,
            'control_ok':all(abs(v) <= GATES['controls']['twin_us'] for v in twins.values())}


def difference(row, positive, negative):
    ts = row['times']
    return stats.median(a-b for a,b in zip(ts[positive]['replays_raw_us'],ts[negative]['replays_raw_us']))


def verdicts(result):
    rows = result['shapes']
    complete = all(str(bs) in rows and len(rows[str(bs)]) == 4 and
                   all('times' in r for r in rows[str(bs)]) for bs in SHAPES)
    ok = complete and all(r['control_ok'] for rr in rows.values() for r in rr)
    if not complete:
        return {'status':'VOID','reason':'incomplete seed/shape/arm set'}
    med = {str(bs): {arm: stats.median(difference(r,'base',arm) for r in rows[str(bs)])
                    for arm in ARMS if arm != 'base'} for bs in SHAPES}
    ab2 = {str(bs):stats.median(difference(r,'AB1','AB2') for r in rows[str(bs)]) for bs in SHAPES}
    if med['1']['AB1'] < 1.2 and med['8']['AB1'] < 1.2:
        ab1 = 'KILL'
    elif med['1']['AB1'] >= 1.8 and med['8']['AB1'] >= 1.8:
        ab1 = 'PASS'
    else:
        ab1 = 'MIXED (PASS requires both bs1 and bs8 >=1.8)'
    numeric = {'AB1':ab1,
               'AB1_per_shape':{str(bs):('PASS' if med[str(bs)]['AB1']>=1.8 else 'KILL' if med[str(bs)]['AB1']<1.2 else 'BETWEEN') for bs in (1,8)},
               'AB2_minus_AB1':{str(bs):('PASS' if ab2[str(bs)]>=1 else 'BELOW_1_US') for bs in SHAPES}}
    for arm in ('COMBSKIP','FIN'):
        numeric[arm] = {str(bs):('KILL' if med[str(bs)][arm]<GATES[arm][bs] else 'PASS') for bs in SHAPES}
    return {'status':'VALID' if ok else 'VOID','reason':None if ok else 'any arm seed-median A/A >1.0 us',
            'median_savings_us':med,'AB2_minus_AB1_us':ab2,'numeric_only':numeric,
            'verdicts':numeric if ok else 'ALL VOID', 'ROUTE0':'optional, not implemented'}


def save(result,out):
    result['verdicts'] = verdicts(result)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines = ['| item | bs1 net us | bs8 net us | bs16 net us | verdict |',
             '|---|---:|---:|---:|---|']
    v = result['verdicts']
    if 'median_savings_us' in v:
        for arm in ARMS:
            if arm=='base': continue
            val=[v['median_savings_us'][str(b)][arm] for b in SHAPES]
            lines.append('| %s | %.3f | %.3f | %.3f | %s |' % (arm,*val,v['status']))
        lines.append('\nAB2-AB1 us: '+json.dumps(v['AB2_minus_AB1_us']))
        lines.append('\nNumeric gates (non-actionable when VOID): '+json.dumps(v['numeric_only']))
    lines.append('\nStatus: '+v['status']+'. Outputs from nonzero flags are garbage; standalone bounds only.')
    if result.get('error'): lines.append('\nError: '+result['error'])
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(out):
    import torch
    out.mkdir(parents=True,exist_ok=True)
    result = {'status':'RUNNING','shapes':{},'arms':ARMS,'gates':GATES,'prepare_count':0}
    save(result,out)
    try:
        with phase('BT4b total',620):
            with phase('one prepare, flag-zero validation',295):
                sys.path.insert(0,str(ROOT/'problems/muse_moe_w4a16'))
                ref=module(ROOT/'problems/muse_moe_w4a16/reference.py','bt4b_reference')
                layer=module(ROOT/'kernels/muse_moe_w4a16/mb_layer_bt4b.py','bt4b_layer')
                weights=ref.generate_weights(ref.WEIGHTS_SEED)
                state=layer.prepare(weights)
                result['prepare_count']+=1
                verify_routes(state)
                assert int(state['bt4b_flag'].item())==0
                disable_pdl(state)
            with phase('flag-zero reference/graph check',60):
                result['correctness']={str(bs):correctness(layer,state,ref.generate_input(bs,'ep8',0,dist='normal')) for bs in SHAPES}
            timer=Timer(torch)
            empty=timer.capture(lambda:None)
            overhead=stats.median(timer.replay(empty) for _ in range(30))
            result.update(overhead_us=overhead,gate_cycles=timer.gate_cyc,
                          routes={str(bs):state['route'][bs] for bs in SHAPES})
            for bs in SHAPES:
                result['shapes'][str(bs)]=[]
                for seed in range(4):
                    with phase(f'bt4b bs={bs} seed={seed}',55):
                        data=ref.generate_input(bs,'ep8',seed,dist='normal')
                        graphs,check=capture_arms(layer,state,data,timer)
                        row={'seed':seed,'protocol':check}
                        result['shapes'][str(bs)].append(row)
                        row.update(measure_graphs(timer,graphs,overhead))
                        # Verify counters after all flags have been interleaved.
                        for name,mask in ARMS.items():
                            arm=state['bt4b_arms'][(mask,bs)]
                            protocol_snapshot(arm,bs,mask)
                        print('[bt4b] paired '+json.dumps({'bs':bs,'seed':seed,'twins_us':row['twins_us'],
                              'saving_us':{a:difference(row,'base',a) for a in ARMS if a!='base'}}),flush=True)
                        save(result,out)
            result['status']='COMPLETE'
    except BaseException:
        import traceback
        result.update(status='FAIL',error=traceback.format_exc())
        raise
    finally:
        save(result,out)


def dry():
    """CPU only; BT4 shape stubs + the REAL base AST constructors/accessors.

    No CUDA/torch imports, allocations, permissive missing-method mocks or files.
    Every new build/init/arm path runs, including all masks and cached selections.
    """
    import hashlib
    patch=module(ROOT/'tools/bt4b_patches.py','bt4b_patch_cpu')
    base=patch.BASE.read_text(encoding='utf-8')
    generated=patch.DEST.read_text(encoding='utf-8')
    report={'source_gates':patch.cpu_gates(base,generated)}
    functions={n.name:n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef)}
    def expression(node,env):
        return eval(compile(ast.Expression(node),'<real base expression>','eval'),env)
    env=patch.environment(base)
    calls,builds=[],[]

    class Tensor:
        sizes={'uint8':1,'int16':2,'bfloat16':2,'int32':4,'float32':4,'int64':8}
        def __init__(self,shape,dtype='float32',device='cpu',value=0):
            self.shape=(shape,) if isinstance(shape,int) else tuple(shape)
            self.dtype,self.device,self.value=dtype,device,value
        def numel(self): return math.prod(self.shape)
        def contiguous(self): return self
        def fill_(self,v): self.value=v; return self
        def item(self): assert self.numel()==1; return self.value
        def view(self,dtype):
            return Tensor(self.numel()*self.sizes[self.dtype]//self.sizes[dtype],dtype,self.device,self.value)
        def __getitem__(self,index):
            if isinstance(index,int):
                assert 0<=index<self.shape[0]
                return Tensor(self.shape[1:],self.dtype,self.device,self.value)
            assert isinstance(index,slice)
            return Tensor((len(range(*index.indices(self.shape[0]))),)+self.shape[1:],self.dtype,self.device,self.value)

    class Torch:
        int32='int32'; int16='int16'; int64='int64'; float32='float32'; bfloat16='bfloat16'; uint8='uint8'
        class cuda:
            @staticmethod
            def get_device_properties(dev):
                class Properties: multi_processor_count=132
                return Properties()
        @staticmethod
        def zeros(shape,dtype='float32',device='cpu'): return Tensor(shape,dtype,device)
        empty=zeros
        @staticmethod
        def zeros_like(t): return Tensor(t.shape,t.dtype,t.device)

    class Mod:
        def __init__(self,name): self.name=name; self.mask=0; self.pdl=0
        def okc_set_pdl(self,v): self.pdl=int(v)
        def bt4b_set_mask(self,v): self.mask=int(v)
        def vml_cfg(self): return (64,8,3,64,16,0,0,0,1,0,0)
        def vml_run(self,T,V): calls.append(('vml',V[0]))
        def okc_run(self,W,S,x,ids,tw,out,lo,nloc,nsm):
            assert len(W)==14 and len(S)==3 and lo==0 and nloc==32 and nsm==132
            assert self.pdl==0
            calls.append((self.name,x.shape[0],self.mask))
        def run(self,*args):
            assert len(args)==18 and args[-2]==0
            m=args[-4]
            if self.mask&8: assert args[11].numel()>=136
            assert args[8].shape==(128,4096) and args[9].shape==(128,4096)
            assert args[10].shape==(256,)
            calls.append((self.name,m,self.mask))

    env.update(torch=Torch,hashlib=hashlib)
    needed=('_k2a_bucket_of','_vml_mag','_vml_alloc','_hyb_scratch','_vml_cfg',
            '_vml_cost_ext','_vml_pool_ext','_vml_seg_ext','_vml_pdel_ext','_vml_call',
            '_mbl_candidates','_mbl_select','_mbl_call','kernel_fn')
    for name in needed:
        exec(compile(ast.Module([functions[name]],type_ignores=[]),'<real base function>','exec'),env)
    mods={name:Mod(name) for name in env['MBL_ARMS']}
    env.update(_VMS={'mod':mods['vms'],'mod_hi':mods['vms_hi']},
               _HYB={'mod':mods['hyb'],'mod7':mods['hyb7']},
               _HYBLO={'mods':{'tiny':mods['tiny'],'tiny2':mods['tiny2']}},
               _VML={'mods':{'bf2':mods['bf2']}})
    mods_expr=next(n.value for n in functions['_mbl_builds'].body if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id=='mods' for t in n.targets))
    env['_mbl_builds']=lambda deadline:expression(mods_expr,env)
    def build_stub(**kw):
        engine=next(e for e in ('vms_hi','tiny2','tiny') if kw['name'].startswith('bt4b_'+e+'_'))
        src,cpp,flags=env['bt4b_sources'](engine)
        assert kw['cuda_sources']==src and kw['cpp_sources']==cpp and kw['extra_cuda_cflags']==flags
        assert 'bt4b_set_mask' in kw['functions']
        builds.append(engine)
        return Mod(engine)
    env['_mbl_load_inline']=build_stub
    env['_mbl_deadline']=lambda deadline,label:None
    exec(compile(generated[generated.index('# BT4B_BEGIN'):],'<new prepare paths>','exec'),env)
    mods=env['_mbl_builds'](999)
    assert builds==['tiny','tiny2','vms_hi']

    # New scratch dimensions MUST come from _mbl_pack_vms's real dictionary.
    vs_expr=next(n.value for n in ast.walk(functions['_mbl_pack_vms']) if isinstance(n,ast.Assign) and
                 any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='st' for t in n.targets))
    vs={}
    env['dev']='cpu'
    for key,val in zip(vs_expr.keys,vs_expr.values):
        k=ast.literal_eval(key)
        vs[k]=expression(val,env) if k in ('inter','ybuf','ss','ctr','nomarks') else Tensor(1)
    st={k:Tensor(1) for k in ('ws13','ws2','gsc13','gsc2','nw')}
    st.update(vms=vs,hyb=[Tensor(1) for _ in range(14)],hyblo=[Tensor(1) for _ in range(14)])
    bank_expr=next(n.value for n in ast.walk(functions['_mbl_banks']) if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='banks' for t in n.targets))
    env['st']=st
    banks={0:expression(bank_expr,env)}
    sc={'bf2':env['_vml_alloc']('cpu',132)}
    hsc,hfull=env['_hyb_scratch']('cpu')
    env.update(weights={},t0=0,mods=mods,banks=banks,sc=sc,hsc=hsc,hfull=hfull,nsm=132,output=Tensor((320,4096),'bfloat16'))
    state_expr=next(n.value for n in functions['prepare'].body if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='state' for t in n.targets))
    state=expression(state_expr,env)
    state['good']={k:True for k in mods}
    env['_mbl_select'](state); verify_routes(state)
    env['_bt4b_init'](state,'cpu')
    assert state['bt4b_flag'].item()==0
    disable_pdl(state)
    tested=[]
    scratch_ids=set()
    for bs in SHAPES:
        data={'hidden_states':Tensor((bs,4096),'bfloat16'),'topk_ids':Tensor((bs,8),'int32'),
              'topk_weights':Tensor((bs,8)),'config':{'local_start':0}}
        for name,mask in ARMS.items():
            arm=env['bt4b_arm'](state,mask,bs)
            assert env['kernel_fn'](data,arm) is state['out'][bs]
            assert calls[-1]==(EXPECTED_ROUTES[bs],bs,mask)
            if bs<=8:
                scratch=arm['h_sc'][0]
                assert scratch is not state['h_sc'][0]
            else:
                av=arm['banks'][0]['k2a33']['st']['vms']
                scratch=av['ctr']
                assert all(av[k] is not vs[k] for k in ('ctr','inter','ybuf','ss'))
                assert all(av[k] is vs[k] for k in ('wA','wB','nw'))
            assert id(scratch) not in scratch_ids
            scratch_ids.add(id(scratch))
            assert env['bt4b_arm'](state,mask,bs) is arm
            tested.append((bs,name,mask))
    for mask,bs in ((16,16),(3,1),(0,32),(-1,1)):
        try: env['bt4b_arm'](state,mask,bs)
        except ValueError: pass
        else: raise AssertionError('invalid arm accepted')
    bad=dict(state,banks={0:{'k2a':{'st':st}}},bt4b_arms={})
    try: env['bt4b_arm'](bad,8,16)
    except KeyError: pass
    else: raise AssertionError('invented bank key accepted')
    bad_vs=dict(vs); bad_vs.pop('ss')
    bad=dict(state,banks={0:{'k2a33':{'st':dict(st,vms=bad_vs)}}},bt4b_arms={})
    try: env['bt4b_arm'](bad,8,16)
    except KeyError: pass
    else: raise AssertionError('missing real scratch key accepted')

    # Exercise capture setup with explicit fake graph snapshots (no CUDA).
    # This runs the actual harness arm loop and all 42*3 replay calls.
    class Cuda:
        @staticmethod
        def stream(s): return contextlib.nullcontext()
        @staticmethod
        def synchronize(): pass
    class TimerStub:
        class torch: cuda=Cuda
        stream=None
        def capture(self,fn):
            fn()
            class Graph:
                def replay(self): pass
            return Graph()
    class Layer:
        bt4b_arm=staticmethod(env['bt4b_arm'])
        kernel_fn=staticmethod(env['kernel_fn'])
    # Snapshot checking itself needs real GPU memory. Replace only that I/O
    # boundary; state construction, dispatch and flag selection stay real.
    old_snapshot=globals()['protocol_snapshot']
    counter={}
    def snapshot_stub(arm,bs,mask):
        key=(bs,mask); counter[key]=counter.get(key,0)+1
        return {'generation':counter[key]}
    globals()['protocol_snapshot']=snapshot_stub
    try:
        for bs in SHAPES:
            data={'hidden_states':Tensor((bs,4096),'bfloat16'),'topk_ids':Tensor((bs,8),'int32'),
                  'topk_weights':Tensor((bs,8)),'config':{'local_start':0}}
            graphs,checks=capture_arms(Layer,state,data,TimerStub())
            assert len(graphs)==14 and all(len(v)==3 for v in checks['three_replays'].values())
    finally:
        globals()['protocol_snapshot']=old_snapshot
    # Gate outcomes + strict VOID rule at both sides of the 1.0-us boundary.
    fake={'shapes':{str(bs):[] for bs in SHAPES}}
    for bs in SHAPES:
        for seed in range(4):
            times={a:{'replays_raw_us':[30.0-(0 if a=='base' else 3.0)]*30} for a in ARMS}
            fake['shapes'][str(bs)].append({'times':times,'control_ok':True})
    assert verdicts(fake)['status']=='VALID'
    fake['shapes']['16'][2]['control_ok']=False
    assert verdicts(fake)['status']=='VOID'
    report.update(status='PASS',builds=builds,prepare_paths=tested,real_base_state_ast=True,
                  invalid_keys_rejected=True,independent_scratch=len(scratch_ids),
                  capture_paths=42,replay_paths=126,limitations='CPU shape stubs do not prove CUDA correctness or liveness')
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('dry','run'))
    p.add_argument('--out',type=Path,default=ROOT/'box_out/bt4b')
    args=p.parse_args()
    if args.mode=='dry': dry()
    else: run(args.out)


if __name__=='__main__':
    main()



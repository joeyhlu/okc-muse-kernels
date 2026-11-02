#!/usr/bin/env python3
"""MP1: cold graph measurements of the board file and raw-weight streams.

The -4% arm deliberately omits the final 4%; the baseline reads every byte.
The +10% arm rereads a prefix. No output from either arm is a model result.
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
import sqlite3
import statistics as stats
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SHAPES = (1, 8, 16, 32, 128, 256)
KEYS = ('w13', 'w13_sf', 'w2', 'w2_sf')
GRID, UNIT, HALF = 132, 36864, 18432

# One issuing thread per CTA, three outstanding transactions, one checksum word
# per completed stage.  All payload bytes are transferred even though the
# checksum samples just one word per stage.  All copy lengths are multiples of
# 16; the final (short) stage is handled by the same barriers.
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
    subprocess.run(['nvcc', '-arch=sm_90a', '-O3', '-std=c++17', '-shared',
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


def power_sample(timer, graph):
    p = subprocess.Popen(['nvidia-smi','--query-gpu=clocks.sm,power.draw',
                          '--format=csv,noheader,nounits','-lms','50'],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        start=time.monotonic()
        while time.monotonic()-start < 2:
            timer.replay(graph)
    finally:
        p.terminate()
        try: output,_=p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill(); output,_=p.communicate()
    rows=[]
    for line in output.splitlines():
        try: rows.append(tuple(float(s.strip()) for s in line.split(',')))
        except ValueError: continue
    if not rows: raise RuntimeError('nvidia-smi returned no clock/power samples')
    return {'sm_mhz':stats.median(r[0] for r in rows),
            'power_w':stats.median(r[1] for r in rows), 'samples':len(rows)}


def run(out):
    import torch
    out.mkdir(parents=True, exist_ok=True)
    with phase('load and prepare', 300):
        sys.path.insert(0, str(ROOT/'problems/muse_moe_w4a16'))
        ref=module(ROOT/'problems/muse_moe_w4a16/reference.py','mp1_reference')
        layer=module(ROOT/'kernels/muse_moe_w4a16/mb_layer.py','mp1_layer')
        weights=ref.generate_weights(ref.WEIGHTS_SEED)
        state=layer.prepare(weights)
        tmp,dll=build()
        timer=Timer(torch)
        empty=timer.capture(lambda: None)
        overhead=stats.median(timer.replay(empty) for _ in range(30))
        print('[mp1] raw empty graph %.3f us' % overhead, flush=True)
    result={'overhead_raw_us':overhead, 'overhead_definition':'empty CUDA graph between CUDA events',
            'routing_share':0.75, 'shapes':{}, 'chains':{}}
    trace_graphs={}
    for bs in SHAPES:
        with phase('measure bs=%d, seeds 0..7' % bs, 75):
            rows=[]
            for seed in range(8):
                data=ref.generate_input(bs,'ep8',seed,dist='normal')
                ids=touched(data['topk_ids'].cpu().numpy())
                buf,sizes=bank(weights,ids,torch)
                sink=torch.empty(GRID,dtype=torch.int32,device='cuda')
                torch.cuda.synchronize()  # default-stream bank/input writes precede timer stream
                def stream(n):
                    rc=dll.mp_run(buf.data_ptr(),n,sink.data_ptr(),timer.stream.cuda_stream)
                    if rc: raise RuntimeError('stream launch: %d' % rc)
                graphs={'ours':timer.capture(lambda:layer.kernel_fn(data,state))}
                for arm,n in sizes.items():
                    graphs[arm]=timer.capture(lambda n=n:stream(n))
                vals={k:[] for k in graphs}
                for rep in range(30):
                    order=list(graphs); order=order[rep%len(order):]+order[:rep%len(order)]
                    for arm in order: vals[arm].append(timer.replay(graphs[arm]))
                row={'seed':seed,'experts':ids,'bytes':sizes,
                     'times':{k:summarize(v,overhead) for k,v in vals.items()}}
                for arm in sizes:
                    us=row['times'][arm]['us']
                    if us <= 0: raise RuntimeError('nonpositive overhead-subtracted stream time')
                    row['times'][arm]['TBps']=sizes[arm]/(us*1e6)
                rows.append(row)
                if seed==0:
                    trace_graphs[bs]=(graphs['ours'],data)
                    with phase('clocks bs=%d ours' % bs, 12): pc=power_sample(timer,graphs['ours'])
                    with phase('clocks bs=%d stream' % bs, 12): ps=power_sample(timer,graphs['base'])
            result['shapes'][str(bs)]={'seeds':rows,'clock_ours':pc,'clock_stream':ps,
                                      'route':state['route'][bs]}
            (out/'results.json').write_text(json.dumps(result,indent=2))
    with phase('nsys graph node traces', 80):
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for bs in SHAPES:
            g,_=trace_graphs[bs]
            for r in range(20):
                with torch.cuda.stream(timer.stream): torch.sum(timer.flush_buf,dim=0,out=timer.sink)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push('G|mp1|%d|%d' % (bs,r))
                with torch.cuda.stream(timer.stream): g.replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    (out/'results.json').write_text(json.dumps(result,indent=2))
    render(result,out)


def render(result,out):
    lines=['| bs | touched experts mean[min-max] | MB | stream ceiling us (sd) | ours us (sd) | gap us | gap % of ours | SM MHz ours / ceiling | W ours / ceiling |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for bs in SHAPES:
        s=result['shapes'][str(bs)]; rows=s['seeds']
        counts=[len(r['experts']) for r in rows]
        a=[r['times']['ours']['us'] for r in rows]; b=[r['times']['base']['us'] for r in rows]
        am,bm=stats.median(a),stats.median(b); gap=am-bm
        c,d=s['clock_ours'],s['clock_stream']
        lines.append('| %d | %.2f [%d–%d] | %.2f | %.2f (%.2f) | %.2f (%.2f) | %.2f | %.1f%% | %.0f / %.0f | %.1f / %.1f |' %
                     (bs,stats.mean(counts),min(counts),max(counts),stats.mean(r['bytes']['base'] for r in rows)/1e6,
                      bm,stats.stdev(b),am,stats.stdev(a),gap,100*gap/am,c['sm_mhz'],d['sm_mhz'],c['power_w'],d['power_w']))
    lines += ['', 'Times are medians of eight seed medians, with sample SD across seeds; MB is decimal. '
              'Raw times, min/p90, per-seed bytes, TB/s, and ± byte arms are in results.json. '
              'The baseline transfers all bytes of all touched experts. The -4% slope arm omits a suffix; '
              'the +10% arm rereads a prefix. ROUTING_SHARE=0.75 is the reference analytical scoring '
              'factor, not applied to this full-coverage empirical measurement.',
              'The subtracted overhead is %.3f us (raw empty CUDA graph between events).' % result['overhead_raw_us']]
    for bs in SHAPES:
        rows=result['shapes'][str(bs)]['seeds']
        fields=[]
        for arm in ('minus4','base','plus10'):
            fields.append('%s %.2f us, %.2f TB/s' % (arm,stats.median(r['times'][arm]['us'] for r in rows),
                                                      stats.median(r['times'][arm]['TBps'] for r in rows)))
        lines.append('bs %d stream: %s.' % (bs,'; '.join(fields)))
    for bs in SHAPES:
        lines += ['', '### bs %d, route %s' % (bs,result['shapes'][str(bs)]['route']),
                  '| kernel / inferred phase | start us | duration us | gap us |', '|---|---:|---:|---:|']
        for k in result.get('chains',{}).get(str(bs),[]):
            lines.append('| %s / %s | %.3f | %.3f | %.3f |' % (k['name'].replace('|','/'),k['phase'],k['start_us'],k['duration_us'],k['gap_us']))
        if not result.get('chains',{}).get(str(bs)): lines.append('| trace unavailable | | | |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def analyze(out, sqlite):
    # The proven NVTX graph-to-launch association handles graph replays and streams.
    sys.path.insert(0,str(ROOT/'tools/box'))
    import layer_budget
    result=json.loads((out/'results.json').read_text())
    by=layer_budget.load_sqlite(str(sqlite))
    for bs in SHAPES:
        replays=[by.get('G|mp1|%d|%d' % (bs,r),[]) for r in range(20)]
        if any(not x for x in replays): raise RuntimeError('missing trace replay for bs %d' % bs)
        if any([k['name'] for k in x] != [k['name'] for k in replays[0]] for x in replays):
            raise RuntimeError('kernel order varies for bs %d' % bs)
        chain=[]
        for i,k in enumerate(replays[0]):
            name=k['name']; low=name.lower()
            kind=next((x for x in ('route','main','fin','combine','norm') if x in low),'unknown')
            starts=[(x[i]['start']-x[0]['start'])/1000 for x in replays]
            dur=[(x[i]['end']-x[i]['start'])/1000 for x in replays]
            gap=[0 if i==0 else (x[i]['start']-x[i-1]['end'])/1000 for x in replays]
            chain.append({'name':name,'phase':kind,'start_us':stats.median(starts),
                          'duration_us':stats.median(dur),'gap_us':stats.median(gap),'replays':len(replays)})
        result['chains'][str(bs)]=chain
    (out/'results.json').write_text(json.dumps(result,indent=2)); render(result,out)
    print('[mp1] trace analysis complete',flush=True)


def dry():
    import numpy as np
    # Execute the base file's actual entrypoints with stand-in tensors/modules.
    # Extract just the two top-level function bodies, not the huge source file.
    snippets=[]; active=False
    with (ROOT/'kernels/muse_moe_w4a16/mb_layer.py').open(encoding='utf-8') as source:
        for line in source:
            if line.startswith(('def _mbl_call(', 'def kernel_fn(')):
                active=True; snippets.append(line); continue
            if active and line.strip() and not line[0].isspace():
                active=False
            if active: snippets.append(line)
    tree=ast.parse(''.join(snippets))
    funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('kernel_fn','_mbl_call')]
    assert len(funcs)==2
    calls=[]
    class Mod:
        def run(self,*args): calls.append('vms')
        def okc_run(self,*args): calls.append('other')
    vs={k:object() for k in ('wA','wB','gsA','gsB','nw','inter','ybuf','ss','ctr','nomarks')}
    st={'vms':vs,'hyblo':object(),'hyb':object()}
    names=['tiny','tiny2','vms','vms_hi','bf2','hyb']
    state={'banks':{0:{'k2a33':{'st':st}}},'out':{m:object() for m in SHAPES},
           'mods':{n:Mod() for n in names},'vml_sc':{'bf2':object()},'h_sc':object(),
           'nsm':132,'route':dict(zip(SHAPES,names))}
    env={'K2A_POST_NORM_EPS':1e-8,'VML_FLAGS':{},'VML_COST':{32:0},
         '_k2a_bucket_of':lambda m:32,'_vml_call':lambda *a,**kw:calls.append('bf2')}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<base functions>','exec'),env)
    for m in SHAPES:
        d={'hidden_states':np.zeros((m,1)), 'topk_ids':np.zeros((m,8)),
           'topk_weights':np.zeros((m,8)), 'config':{'local_start':0}}
        assert env['kernel_fn'](d,state) is state['out'][m]
    assert calls==['other','other','vms','vms','bf2','other']
    class FakeTorch:
        uint8=np.uint8
        @staticmethod
        def cat(xs): return np.concatenate(xs)
    # numpy's view and contiguous interface differ; thin tensor wrapper.
    class T:
        def __init__(self,a): self.a=a
        def __getitem__(self,i): return T(self.a[i])
        def contiguous(self): return self
        def view(self,d): return T(self.a.view(d))
        def reshape(self,*shape): return T(self.a.reshape(*shape))
        def numel(self): return self.a.size
    FakeTorch.cat=staticmethod(lambda xs:T(np.concatenate([x.a for x in xs])))
    w={k:T(np.arange(32*64,dtype=np.uint8).reshape(32,64)) for k in KEYS}
    ids=touched(np.array([[0,3,32,255,-1,3]])); assert ids==[0,3]
    b,sizes=bank(w,ids,FakeTorch); assert sizes['base']==512 and b.numel()==sizes['plus10']
    for n in (16, 512, 132*UNIT+16, 32*28311552):
        spans=[]; stages=(n+UNIT-1)//UNIT
        for c in range(GRID):
            lo=stages*c//GRID; hi=stages*(c+1)//GRID
            spans.extend((j*UNIT,min(n,(j+1)*UNIT)) for j in range(lo,hi))
        assert spans and spans[0][0]==0 and spans[-1][1]==n
        assert all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
    print('CPU dry PASS: real base entrypoint arms, raw bank, coverage, tail, slope sizes',flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('run','analyze','dry','cuda'))
    p.add_argument('--out',type=Path,default=ROOT/'box_out/mp1'); p.add_argument('--sqlite',type=Path)
    a=p.parse_args()
    if a.mode=='dry': dry()
    elif a.mode=='cuda': print(CUDA)
    elif a.mode=='run': run(a.out)
    else:
        if not a.sqlite: p.error('--sqlite required')
        with phase('analyze nsys sqlite',60): analyze(a.out,a.sqlite)


if __name__=='__main__': main()

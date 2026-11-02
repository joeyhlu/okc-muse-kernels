#!/usr/bin/env python3
"""BT4: timing-only intercept probes; no output-null arm is a model.
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
SHAPES = (1, 8, 16, 32, 128, 256)
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


GATES = {
    'H1': {'bs16_us':2.0, 'bs32_us':3.0,
           'rule':'KILL hot-token family only if |gap(max>=8)-gap(max2)| is below both gates; all four seeds required'},
    'H2': {'all_bs_us':1.0, 'rule':'KILL if tax < 1 us at EVERY scored bs'},
    'H3': {'bs32_us':2.5, 'rule':'KILL if net gated saving (reset included) < 2.5 us'},
    'H4': {'bs128_us':2.5, 'bs256_us':2.5,
           'rule':'KILL if net gated saving < 2.5 us at EITHER bs'},
    'H5': {'status':'NOT IMPLEMENTED (optional)', 'gates_us':[3.3,6.8,8.0,8.0]},
    'controls': {'twin_us':1.0, 'rule':'VOID that family if any arm A/A seed median differs > 1 us'},
}
EXPECTED_ROUTES = {1:'tiny',8:'tiny2',16:'vms_hi',32:'bf2',128:'hyb',256:'hyb7'}


class InfeasibleRouting(ValueError):
    pass


def hot_routing(ids, weights, maximum):
    """Numpy construction preserving global touched set and local-pair count.

    A cap c is feasible iff n-1+c <= P <= n*c, c<=bs, P<=8*bs.
    For >=8 choose max(8,ceil(P/n)); keep every other expert at least once.
    Bipartite degree realization: put each expert on the least-loaded tokens.
    No token gets that expert twice, and loads differ by at most one.
    """
    import numpy as np
    ids = np.asarray(ids, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float32)
    assert ids.ndim == 2 and ids.shape[1] == 8 and weights.shape == ids.shape
    local = (ids >= 0) & (ids < 32)
    experts = np.unique(ids[local])
    n, pairs, bs = len(experts), int(local.sum()), len(ids)
    cap = max(8, math.ceil(pairs/n)) if maximum == '8+' and n else int(maximum) if maximum != '8+' else 8
    if n == 0 or cap > bs or not n-1+cap <= pairs <= n*cap or pairs > 8*bs:
        raise InfeasibleRouting(f'P={pairs}, n={n}, bs={bs}, max={maximum}: need n-1+cap <= P <= n*cap and cap<=bs')
    degrees = np.ones(n, dtype=np.int32)
    degrees[0] = cap
    remain = pairs - int(degrees.sum())
    if n > 1:
        quotient, extra = divmod(remain, n-1)
        degrees[1:] += quotient
        degrees[1:1+extra] += 1
    else:
        assert remain == 0
    rows = [[] for _ in range(bs)]
    loads = np.zeros(bs, dtype=np.int32)
    for j in np.argsort(-degrees, kind='stable'):
        selected = np.argsort(loads, kind='stable')[:int(degrees[j])]
        for t in selected:
            rows[int(t)].append(int(experts[j]))
            loads[t] += 1
    assert int(loads.max()) <= 8
    result = np.empty_like(ids)
    for t, row in enumerate(rows):
        result[t] = row + list(range(32, 32+8-len(row)))
    # Preserve slot weights, normalize after routing edits; hidden_states stay shared.
    tw = np.maximum(weights, np.finfo(np.float32).tiny)
    tw = tw / tw.sum(axis=1, keepdims=True)
    validate_hot(ids, result, tw, maximum)
    return result, tw


def validate_hot(original, ids, tw, maximum):
    import numpy as np
    assert ((ids >= 0) & (ids < 256)).all()
    assert all(len(set(map(int,row))) == 8 for row in ids)
    old_local = original[(original >= 0) & (original < 32)]
    new_local = ids[(ids >= 0) & (ids < 32)]
    assert np.array_equal(np.unique(old_local), np.unique(new_local))
    assert len(old_local) == len(new_local)
    count = np.bincount(new_local, minlength=32)
    assert count.max() >= 8 if maximum == '8+' else count.max() == int(maximum)
    assert np.isfinite(tw).all() and (tw > 0).all()
    np.testing.assert_allclose(tw.sum(1), 1, atol=2e-7)
    return {'pairs':len(new_local), 'touched':len(np.unique(new_local)),
            'max_n':int(count.max()), 'counts':count.tolist()}


def measure(timer, functions, overhead):
    """Second independent graph for EVERY arm; rotate all arms every repetition."""
    graphs = {}
    for arm, fn in functions.items():
        graphs[arm] = timer.capture(fn)
        graphs[arm+'_twin'] = timer.capture(fn)
    return measure_graphs(timer, graphs, overhead)


def measure_graphs(timer, graphs, overhead):
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
    # Paired per-repetition differences; empty overhead cancels for this contrast.
    ts = row['times']
    return stats.median(a-b for a,b in zip(ts[positive]['replays_raw_us'],ts[negative]['replays_raw_us']))


def disable_pdl(state):
    # Access is exactly mb_layer._mbl_builds/_mbl_call's state['mods'][name].
    for name in ('tiny','tiny2','hyb','hyb7'):
        state['mods'][name].okc_set_pdl(0)
    # Base _mbl_call passes pdl=1 positionally to VMS. Two calls in H2's B
    # graph share scratch, so an early dependent launch must not overlap them.
    # Keep kernel_fn and CUDA untouched; adapt only the host run() parameter.
    class VmsNoPDL:
        def __init__(self, mod): self.mod=mod
        def run(self, *args):
            assert len(args)==18, 'VMS run ABI drift (mb_layer.py:1139-1141)'
            args=list(args)
            args[-2]=0
            return self.mod.run(*args)
    for name in ('vms','vms_hi'):
        state['mods'][name]=VmsNoPDL(state['mods'][name])


def verify_routes(state):
    for bs, name in EXPECTED_ROUTES.items():
        assert state['route'][bs] == name, ('fallback invalidates probe', bs, state['route'][bs], name)
    for bs in range(25,49):
        assert state['route'][bs] == 'bf2', (bs,state['route'][bs])


def correctness(layer, state, data):
    """Flag-zero only: same base comparator, eager plus graph replay."""
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
    return got.clone()


def verdicts(result):
    out = {}
    hot = result['H1']
    diffs = {}
    for bs in (16,32):
        rows = hot.get(str(bs),[])
        valid = [r for r in rows if all(p in r['profiles'] and 'times' in r['profiles'][p] for p in ('2','8+'))]
        if len(valid) == 4:
            diffs[str(bs)] = stats.median(r['profiles']['8+']['gap_minus_030n_us']-r['profiles']['2']['gap_minus_030n_us'] for r in valid)
    if len(diffs) < 2:
        out['H1'] = {'verdict':'VOID', 'reason':'fixed P/n constraints make one or more required seed profiles infeasible, or run incomplete', 'contrasts_us':diffs}
    else:
        control = all(p['control_ok'] for rows in hot.values() for r in rows for p in r['profiles'].values() if 'times' in p)
        out['H1'] = {'verdict':'VOID' if not control else 'KILL' if abs(diffs['16']) < 2 and abs(diffs['32']) < 3 else 'PASS', 'contrasts_us':diffs}
    cold = result['H2']
    taxes = {b:stats.median(r['tax_us'] for r in rows) for b,rows in cold.items() if len(rows)==4}
    good = all(r['control_ok'] for rows in cold.values() for r in rows)
    out['H2'] = {'verdict':'VOID' if len(taxes)!=6 or not good else 'KILL' if all(t < 1 for t in taxes.values()) else 'PASS', 'tax_us':taxes}
    for probe, shapes in (('H3',(32,)),('H4',(128,256))):
        rows_by_bs = result[probe]
        deltas = {b:stats.median(r['saving_us'] for r in rows) for b,rows in rows_by_bs.items() if len(rows)==4}
        good = all(r['control_ok'] for rows in rows_by_bs.values() for r in rows)
        out[probe] = {'verdict':'VOID' if not good or any(str(b) not in deltas for b in shapes) else
                      'KILL' if any(deltas[str(b)] < 2.5 for b in shapes) else 'PASS', 'net_saving_us':deltas}
    return out


def save(result, out):
    result['verdicts'] = verdicts(result)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines = ['# BT4 timing-only intercept probes', '',
             'Pre-registered gates (microseconds; net savings include reset launches):',
             '```json',json.dumps(GATES,indent=2),'```','',
             'Protocol: 256 MiB flush, calibrated 300 us gate, empty-graph overhead subtracted, '
             '30 rotated repetitions, 4 seeds, every arm has an independent A/A graph.',
             'PASS identifies an upper-bound timing signal, not a usable kernel. '
             'H2 includes all warmable state, including any carried weight lines.',
             f"Run status: {result.get('status','RUNNING')}; empty-graph overhead: {result.get('overhead_us','pending')} us.", '',
             '| probe | bs | seed/profile | measured values us | control |', '|---|---:|---|---|---|']
    for bs, rows in result['H1'].items():
        for r in rows:
            for p, v in r['profiles'].items():
                txt = (f"ours {v['times']['ours']['us']:.3f}; ceiling {v['times']['ceiling']['us']:.3f}; "
                       f"gap {v['gap_us']:.3f}; gap-0.30n {v['gap_minus_030n_us']:.3f}") if 'times' in v else v['reason']
                lines.append(f"| H1 | {bs} | {r['seed']}/{p} | {txt} | {v.get('control_ok','INFEASIBLE')} |")
    for bs, rows in result['H2'].items():
        for r in rows:
            ts = r['times']
            txt = f"A1 {ts['A1']['us']:.3f}; A2 {ts['A2']['us']:.3f}; B {ts['B']['us']:.3f}; tax {r['tax_us']:.3f}"
            lines.append(f"| H2 | {bs} | {r['seed']} | {txt} | {r['control_ok']} |")
    for probe in ('H3','H4'):
        for bs,rows in result[probe].items():
            for r in rows:
                ts = r['times']
                txt = f"base {ts['base']['us']:.3f}; probe {ts['probe']['us']:.3f}; net saving {r['saving_us']:.3f}"
                if 'reset_only' in ts:
                    txt += f"; reset-only {ts['reset_only']['us']:.3f} (diagnostic)"
                lines.append(f"| {probe} | {bs} | {r['seed']} | {txt} | {r['control_ok']} |")
    for name, v in result['verdicts'].items():
        lines += ['', f"{v['verdict']} {name}: {json.dumps(v,sort_keys=True)}"]
    if result.get('error'):
        lines += ['', 'ERROR: '+result['error']]
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(out):
    import numpy as np
    import torch
    out.mkdir(parents=True,exist_ok=True)
    result = {'gates':GATES,'H1':{},'H2':{},'H3':{},'H4':{},'status':'RUNNING',
              'flag_bits':{'NOFIN_VML':2,'NOFINAL_HYB':4}, 'H5':'optional, not implemented'}
    save(result,out)
    try:
        with phase('BT4 total', 620):
            with phase('load weights and base prepare ONCE',295):
                sys.path.insert(0,str(ROOT/'problems/muse_moe_w4a16'))
                ref = module(ROOT/'problems/muse_moe_w4a16/reference.py','bt4_reference')
                base = module(ROOT/'kernels/muse_moe_w4a16/mb_layer.py','bt4_base')
                weights = ref.generate_weights(ref.WEIGHTS_SEED)
                base_state = base.prepare(weights)
                verify_routes(base_state)
            with phase('variant prepare ONCE, flag 0 validation unchanged',295):
                variant = module(ROOT/'kernels/muse_moe_w4a16/mb_layer_bt4.py','bt4_variant')
                state = variant.prepare(weights)
                verify_routes(state)
                assert state['bt4_flag'].item() == 0
                disable_pdl(base_state)
                disable_pdl(state)
            with phase('flag 0 correctness replays, before enabling any probe',60):
                result['correctness'] = {}
                for bs in SHAPES:
                    d = ref.generate_input(bs,'ep8',0,dist='normal')
                    a = correctness(base,base_state,d)
                    b = correctness(variant,state,d)
                    err = variant._k2a_rel_err(b,a)[0]
                    assert err < 1e-3, ('flag zero diverged',bs,err)
                    result['correctness'][str(bs)] = {'base_relative':err,'bitwise':bool(torch.equal(a,b))}
                print('[bt4] flag-0 correct; now permitting timing-only flags',flush=True)
            with phase('ceiling compile + timer calibration',90):
                tmp,dll = build()
                timer = Timer(torch)
                empty = timer.capture(lambda:None)
                overhead = stats.median(timer.replay(empty) for _ in range(30))
                result['overhead_us'] = overhead
                result['gate_cycles'] = timer.gate_cyc
                result['routes'] = {str(b):state['route'][b] for b in SHAPES}
                print(f'[bt4] empty overhead={overhead:.3f} us gate={timer.gate_cyc} cycles',flush=True)
            for bs in (16,32):
                result['H1'][str(bs)] = []
                with phase(f'H1 bs={bs}',55):
                    for seed in range(4):
                        d = ref.generate_input(bs,'ep8',seed,dist='normal')
                        orig = d['topk_ids'].cpu().numpy()
                        wt = d['topk_weights'].cpu().numpy()
                        experts = touched(orig)
                        buf,sizes = bank(weights,experts,torch)
                        sink = torch.empty(GRID,dtype=torch.int32,device='cuda')
                        torch.cuda.synchronize()
                        def ceiling():
                            rc = dll.mp_run(buf.data_ptr(),sizes['base'],sink.data_ptr(),timer.stream.cuda_stream)
                            if rc: raise RuntimeError('stream launch %d'%rc)
                        row = {'seed':seed,'experts':experts,'bytes':sizes['base'],'profiles':{},
                               'original_pairs':int(((orig>=0)&(orig<32)).sum())}
                        result['H1'][str(bs)].append(row)
                        graphs, retained = {}, []
                        for profile in ('2','5','8+'):
                            try:
                                ids,tw = hot_routing(orig,wt,profile)
                            except InfeasibleRouting as ex:
                                row['profiles'][profile] = {'status':'INFEASIBLE','reason':str(ex)}
                                print(f'[bt4] H1 bs={bs} seed={seed} max={profile}: {ex}',flush=True)
                                continue
                            data = dict(d,topk_ids=torch.as_tensor(ids,device='cuda'),topk_weights=torch.as_tensor(tw,device='cuda'))
                            assert data['hidden_states'] is d['hidden_states']
                            retained.append(data)
                            torch.cuda.synchronize()
                            row['profiles'][profile] = validate_hot(orig,ids,tw,profile)
                            for arm,fn in (('ours',lambda data=data:base.kernel_fn(data,base_state)),('ceiling',ceiling)):
                                for suffix in ('','_twin'):
                                    graphs[profile+'/'+arm+suffix] = timer.capture(fn)
                        # All profiles and their twins rotate together, eliminating profile-order drift.
                        if graphs:
                            all_values = {a:[] for a in graphs}
                            for rep in range(30):
                                order = list(graphs); order = order[rep%len(order):]+order[:rep%len(order)]
                                for arm in order: all_values[arm].append(timer.replay(graphs[arm]))
                            for p in ('2','5','8+'):
                                if 'status' in row['profiles'][p]: continue
                                vals = {a.split('/')[1]:v for a,v in all_values.items() if a.startswith(p+'/')}
                                r = row['profiles'][p]
                                r['times'] = {a:summarize(v,overhead) for a,v in vals.items()}
                                r['twins_us'] = {a:stats.median(x-y for x,y in zip(vals[a],vals[a+'_twin'])) for a in ('ours','ceiling')}
                                r['control_ok'] = all(abs(x)<=1 for x in r['twins_us'].values())
                                r['gap_us'] = difference(r,'ours','ceiling')
                                r['gap_minus_030n_us'] = r['gap_us']-0.30*len(experts)
                        save(result,out)
                        print(f'[bt4] H1 bs={bs} seed={seed} done',flush=True)
            for bs in SHAPES:
                result['H2'][str(bs)] = []
                with phase(f'H2 bs={bs}',40):
                    for seed in range(4):
                        d1 = ref.generate_input(bs,'ep8',seed,dist='normal')
                        d2 = ref.generate_input(bs,'ep8',seed+4,dist='normal')
                        torch.cuda.synchronize()
                        def a1(): return base.kernel_fn(d1,base_state)
                        def a2(): return base.kernel_fn(d2,base_state)
                        def both(): a1(); a2()
                        r = measure(timer,{'A1':a1,'A2':a2,'B':both},overhead)
                        ts = r['times']
                        r.update(seed=seed,seed2=seed+4,
                                 tax_us=stats.median(x+y-z-overhead for x,y,z in zip(ts['A1']['replays_raw_us'],ts['A2']['replays_raw_us'],ts['B']['replays_raw_us'])))
                        result['H2'][str(bs)].append(r)
                        save(result,out)
                        print(f"[bt4] H2 bs={bs} seed={seed} tax={r['tax_us']:.3f}",flush=True)
            for probe,mask,shapes in (('H3',2,(32,)),('H4',4,(128,256))):
                for bs in shapes:
                    result[probe][str(bs)] = []
                    with phase(f'{probe} bs={bs}',50):
                        for seed in range(4):
                            d = ref.generate_input(bs,'ep8',seed,dist='normal')
                            graphs = {}
                            for arm,bit in (('base',0),('probe',mask)):
                                view = variant.bt4_arm(state,bit,bs)
                                torch.cuda.synchronize()
                                fn = lambda view=view:variant.kernel_fn(d,view)
                                graphs[arm] = timer.capture(fn)
                                graphs[arm+'_twin'] = timer.capture(fn)
                            # Mutate the holder after capture. Graph replay must use the captured
                            # scalar/mask, never this live word. Repeated replays catch stale counters.
                            variant.bt4_arm(state,0,bs)
                            r = measure_graphs(timer,graphs,overhead)
                            r.update(seed=seed,mask=mask,saving_us=difference(r,'base','probe'))
                            probe_state = state['bt4_arms'][(mask,bs)]
                            assert variant._mbl_health(state['route'][bs],probe_state), (probe,bs,'guard/error counter')
                            if probe=='H3':
                                view = state['bt4_arms'][(mask,bs)]
                                extra = measure(timer,{'reset_only':lambda:variant.bt4_reset_only(d,view)},overhead)
                                r['times'].update(extra['times'])
                                r['twins_us'].update(extra['twins_us'])
                                r['control_ok'] &= extra['control_ok']
                            result[probe][str(bs)].append(r)
                            save(result,out)
                            print(f"[bt4] {probe} bs={bs} seed={seed} net saving={r['saving_us']:.3f}",flush=True)
            result['status'] = 'COMPLETE'
    except BaseException as ex:
        result['status'] = 'FAILED'
        result['error'] = repr(ex)
        raise
    finally:
        save(result,out)
        for name,v in result['verdicts'].items():
            print(f"[bt4] {v['verdict']} {name}: {v}",flush=True)


def dry():
    """CPU-only: numpy routings plus real AST state constructors/accessors.

    No permissive __getattr__, no torch/CUDA import, no files written. GPU RNG
    is not reproducible by numpy; real seed feasibility is logged on the pod.
    """
    import hashlib
    import numpy as np
    patch = module(ROOT/'tools/bt4_patches.py','bt4_patch_cpu')
    base = patch.BASE.read_text(encoding='utf-8')
    generated = patch.DEST.read_text(encoding='utf-8')
    report = {'source_gates':patch.cpu_gates(base,generated)}
    tree = ast.parse(base)
    functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    def expression(node, env):
        return eval(compile(ast.Expression(node),'<real base expression>','eval'),env)
    env = patch.environment(base)
    calls, builds = [], []

    class Tensor:
        # Shape-only stand-in avoids allocating the multi-GB real weight bank.
        sizes = {'uint8':1,'int16':2,'bfloat16':2,'int32':4,'float32':4,'int64':8}
        def __init__(self, shape, dtype='float32', device='cpu', value=0):
            self.shape = (shape,) if isinstance(shape,int) else tuple(shape)
            self.dtype,self.device,self.value = dtype,device,value
        def numel(self): return math.prod(self.shape)
        def contiguous(self): return self
        def fill_(self,value): self.value=value; return self
        def item(self): assert self.numel()==1; return self.value
        def reshape(self,*shape):
            shape = list(shape[0] if len(shape)==1 and isinstance(shape[0],(tuple,list)) else shape)
            if -1 in shape: shape[shape.index(-1)] = self.numel()//math.prod(x for x in shape if x!=-1)
            assert math.prod(shape)==self.numel()
            return Tensor(shape,self.dtype,self.device,self.value)
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
        @staticmethod
        def zeros(shape,dtype='float32',device='cpu'): return Tensor(shape,dtype,device)
        empty=zeros
        @staticmethod
        def cat(ts,dim=0):
            assert dim==0 and all(t.dtype==ts[0].dtype and t.shape[1:]==ts[0].shape[1:] for t in ts)
            return Tensor((sum(t.shape[0] for t in ts),)+ts[0].shape[1:],ts[0].dtype)

    class Mod:
        def __init__(self,name): self.name=name; self.mask=0; self.pdl=0
        def okc_set_pdl(self,v): self.pdl=int(v)
        def bt4_set_mask(self,v): self.mask=int(v)
        def vml_cfg(self): return (64,8,3,64,16,0,0,0,1,0,0)
        def bt4_vml_reset(self,T,M):
            assert len(T)==5 and 25<=M<=48 and T[3].value==-1
            calls.append(('reset',M))
        def vml_run(self,T,V):
            assert len(T)==20 and len(V)==22 and V[4] in (1,3) and V[5]&256==0
            calls.append(('vml',V[0],V[4]))
        def okc_run(self,W,S,x,ids,tw,out,lo,nloc,nsm):
            assert len(W)==14 and len(S)==3 and lo==0 and nloc==32 and nsm==132
            calls.append((self.name,x.shape[0],self.mask))
        def run(self,*args): calls.append(('vms',args[-4],args[-2]))

    env.update(torch=Torch,hashlib=hashlib)
    needed = ('_k2a_bucket_of','_vml_mag','_vml_alloc','_hyb_scratch','_vml_cfg',
              '_vml_cost_ext','_vml_pool_ext','_vml_seg_ext','_vml_pdel_ext','_vml_call',
              '_mbl_candidates','_mbl_select','_mbl_call','kernel_fn')
    for name in needed:
        exec(compile(ast.Module([functions[name]],type_ignores=[]),'<real base function>','exec'),env)
    # Unpack simple literal multi-target assignments used by the allocator.
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) for t in n.targets):
            try: ast.literal_eval(n.value)
            except (ValueError,TypeError): continue
            exec(compile(ast.Module([n],type_ignores=[]),'<real base constants>','exec'),env)
    mods = {name:Mod(name) for name in env['MBL_ARMS']}
    env.update(_VMS={'mod':mods['vms'],'mod_hi':mods['vms_hi']},
               _HYB={'mod':mods['hyb'],'mod7':mods['hyb7']},
               _HYBLO={'mods':{'tiny':mods['tiny'],'tiny2':mods['tiny2']}},
               _VML={'mods':{'bf2':mods['bf2']}})
    mods_expr = next(n.value for n in functions['_mbl_builds'].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mods' for t in n.targets))
    env['_mbl_builds'] = lambda deadline:expression(mods_expr,env)
    def build_stub(**kw):
        assert set(kw)=={'name','cpp_sources','cuda_sources','functions','extra_cuda_cflags','verbose'}
        engine = kw['name'].split('_')[1]
        assert engine in ('bf2','hyb','hyb7')
        assert kw['cuda_sources'] == env['bt4_sources'](engine)[0]
        builds.append(engine)
        return Mod(engine)
    env['_mbl_load_inline']=build_stub
    env['_mbl_deadline']=lambda deadline,label:None
    exec(compile(generated[generated.index('# BT4_BEGIN'):],'<new prepare paths>','exec'),env)
    mods=env['_mbl_builds'](999)
    assert builds==['bf2','hyb','hyb7']

    # Construct the bank nesting from the REAL _mbl_banks assignment, not a
    # parallel guessed {'k2a':...} layout. Payload keys come from actual accessors.
    st={k:Tensor(1) for k in ('ws13','ws2','gsc13','gsc2','nw')}
    vs_keys={n.slice.value for n in ast.walk(functions['_mbl_call'])
             if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='vs' and isinstance(n.slice,ast.Constant)}
    st['vms']={k:Tensor(1) for k in vs_keys}
    st['hyb']=[Tensor(1) for _ in range(14)]; st['hyblo']=[Tensor(1) for _ in range(14)]
    bank_expr=next(n.value for n in ast.walk(functions['_mbl_banks']) if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='banks' for t in n.targets))
    env.update(st=st)
    banks={0:expression(bank_expr,env)}
    sc={'bf2':env['_vml_alloc']('cpu',132)}
    hsc,hfull=env['_hyb_scratch']('cpu')
    env.update(weights={},t0=0,mods=mods,banks=banks,sc=sc,hsc=hsc,hfull=hfull,nsm=132,output=Tensor((320,4096),'bfloat16'))
    state_expr=next(n.value for n in functions['prepare'].body if isinstance(n,ast.Assign) and
                    any(isinstance(t,ast.Name) and t.id=='state' for t in n.targets))
    state=expression(state_expr,env)
    state['good']={k:True for k in mods}
    env['_mbl_select'](state)
    verify_routes(state)
    env['_bt4_init'](state,'cpu')
    assert state['bt4_flag'].item()==0
    disable_pdl(state)
    tested=[]
    for bs in (1,8,16,25,32,48,49,128,193,256):
        data={'hidden_states':Tensor((bs,4096),'bfloat16'), 'topk_ids':Tensor((bs,8),'int32'),
              'topk_weights':Tensor((bs,8)), 'config':{'local_start':0}}
        for mask in (0,2,4,6):
            arm=env['bt4_arm'](state,mask,bs)
            out=env['kernel_fn'](data,arm)
            assert out is state['out'][bs]
            if bs<=16 and bs>8: assert calls[-1]==('vms',bs,0)
            if mask&2 and 25<=bs<=48:
                assert arm['vml_sc']['bf2'] is not state['vml_sc']['bf2']
                assert calls[-2:]==[('reset',bs),('vml',bs,1)]
            if mask&4 and bs>=49: assert arm['h_sc'] is not state['h_sc']
            tested.append((bs,mask))
            assert env['bt4_arm'](state,mask,bs) is arm
    # Negative key regression: the real accessor must fail, not silently fallback.
    bad=dict(state,banks={0:{'k2a':{'st':st}}})
    data['hidden_states']=Tensor((256,4096),'bfloat16')
    try: env['kernel_fn'](data,bad)
    except KeyError: pass
    else: raise AssertionError('incorrect bank key accepted')
    try: env['bt4_arm'](state,16,32)
    except ValueError: pass
    else: raise AssertionError('unimplemented NCAP8 accepted')

    # The copied raw-byte ceiling bank executes with real weight dimensions.
    w={'w13':Tensor((256,8192,2048),'uint8'),'w13_sf':Tensor((256,8192,256),'uint8'),
       'w2':Tensor((256,4096,2048),'uint8'),'w2_sf':Tensor((256,4096,256),'uint8')}
    for n in (1,8,15,32):
        buf,sizes=bank(w,list(range(n)),Torch)
        assert sizes['base']==n*28311552 and buf.numel()==sizes['plus10']

    passed=impossible=0
    for bs in (16,32):
        for seed in range(4):
            rng=np.random.default_rng(seed+bs*100)
            # Numpy stand-in for reference.generate_input, including busiest rotation.
            _=rng.normal(size=(bs,4096))
            scores=rng.random((bs,256))+0.03*rng.normal(size=256)/np.sqrt(-np.log(np.maximum(rng.random(256),1e-12)))
            ids=np.argsort(-scores,axis=1)[:,:8].astype(np.int32)
            shard=np.bincount((ids//32).ravel(),minlength=8).argmax()
            ids=(ids-32*shard)%256
            tw=rng.random((bs,8),dtype=np.float32)
            for maximum in ('2','5','8+'):
                try: hot_routing(ids,tw,maximum); passed+=1
                except InfeasibleRouting: impossible+=1
        # Exercise EVERY feasible degree total, including dense top8 and sparse rows.
        for n in range(1,33):
            for maximum in ('2','5','8+'):
                for pairs in range(n,min(bs*8,n*bs)+1):
                    cap=max(8,math.ceil(pairs/n)) if maximum=='8+' else int(maximum)
                    feasible=cap<=bs and n-1+cap<=pairs<=n*cap
                    if not feasible: continue
                    # Obtain arbitrary valid seed ids with the intended n and P.
                    original=np.tile(np.arange(32,40,dtype=np.int32),(bs,1))
                    for i in range(pairs):
                        t=i//8; sl=i%8
                        original[t,sl]=i%n
                    # Small n needs a degree-realizable seed, not row-major duplicates.
                    if n<8:
                        original=np.tile(np.arange(32,40,dtype=np.int32),(bs,1))
                        for i in range(pairs): original[i//n,i%n]=i%n
                    new,tw=hot_routing(original,np.ones((bs,8),dtype=np.float32),maximum)
                    validate_hot(original,new,tw,maximum)
                    passed+=1
    # Prove rmap y-reset coverage against the real vm_fin four-float4 mapping.
    cols=[1024*m+4*t+k for t in range(256) for m in range(4) for k in range(4)]
    assert sorted(cols)==list(range(4096))
    # A single input may have no feasible max-2 redistribution. Never falsify P/n.
    ids=np.tile(np.arange(8,dtype=np.int32),(16,1))
    try: hot_routing(ids,np.ones_like(ids,dtype=np.float32),'2')
    except InfeasibleRouting: pass
    else: raise AssertionError('infeasible P > 2*n accepted')
    report.update(status='PASS',prepare_paths=tested,builds=builds,original_state_ast=True,
                  routing_passed=passed,routing_infeasible=impossible,reset_indices_exact=True,
                  gpu_rng_note='numpy stand-ins exercise constraints; real CUDA seed counts are checked at run time')
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('dry','run'))
    parser.add_argument('--out',type=Path,default=ROOT/'box_out/bt4')
    args=parser.parse_args()
    if args.mode=='dry': dry()
    else: run(args.out)


if __name__=='__main__':
    main()

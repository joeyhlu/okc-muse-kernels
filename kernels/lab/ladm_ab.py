#!/usr/bin/env python3
"""LAD timing-only: invalid outputs for L/F/G; never a submission."""
import ast, json, math, random, statistics as stats, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'kernels/lab'))
from moephase import Timer, phase, module, percentile
EXPECTED_ROUTES={1:'tiny',8:'tiny2',16:'vms_hi',32:'bf2',128:'hyb',256:'hyb7'}
def verify_routes(state):
    for bs,name in EXPECTED_ROUTES.items(): assert state['route'][bs]==name,(bs,state['route'][bs],name)
ARMS={'A':0,'L':1,'LF':3,'LG':5,'LFG':7,'F':2,'G':4,'M':8,'LM':9,'LFM':11,'LFGM':15}

def dry():
    """CPU-only: numpy routings plus real AST state constructors/accessors.

    No permissive __getattr__, no torch/CUDA import, no files written. GPU RNG
    is not reproducible by numpy; real seed feasibility is logged on the pod.
    """
    import hashlib
    import numpy as np
    # Execute the proven F1 dry run, including the real _mbl_banks function.
    from l2r_ab import dry as bank_dry
    bank_dry()
    patch = module(ROOT/'tools/ladm_patches.py','ladm_patch_cpu')
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
        assert engine in ('hyb','hyb7')
        assert kw['cuda_sources'] == env['ladm_sources'](engine)[0]
        builds.append(engine)
        return Mod(engine)
    env['_mbl_load_inline']=build_stub
    env['_mbl_deadline']=lambda deadline,label:None
    exec(compile(generated[generated.index('# LADM_BEGIN'):],'<new prepare paths>','exec'),env)
    mods=env['_mbl_builds'](999)
    assert builds==['hyb','hyb7']

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
    env['_ladm_init'](state,'cpu')
    assert state['ladm_flag'].item()==0
    tested=[]
    for bs in (128,256):
        data={'hidden_states':Tensor((bs,4096),'bfloat16'), 'topk_ids':Tensor((bs,8),'int32'),
              'topk_weights':Tensor((bs,8)), 'config':{'local_start':0}}
        for mask in range(16):
            arm=env['ladm_arm'](state,mask,bs)
            env['ladm_capture_mask'](arm,mask)
            assert arm['banks'][0]['k2a33']['st']['hyb'][8].item()==mask
            assert arm['h_sc'] is not state['h_sc']
            assert env['kernel_fn'](data,arm) is state['out'][bs]
            assert calls[-1]==(state['route'][bs],bs,4 if mask&2 else 0)
            tested.append((bs,mask))
    assert state['ladm_flag'].item()==0
    assert state['banks'][0]['k2a33']['st']['hyb'][8] is state['ladm_flag']
    bad=dict(state,banks={0:{'k2a':{'st':st}}})
    try: env['ladm_arm'](bad,0,128)
    except KeyError: pass
    else: raise AssertionError('incorrect bank key accepted')
    for m in (0,15,16):
        try: env['ladm_arm'](state,m,1 if m<16 else 128)
        except ValueError: pass
        else: raise AssertionError('invalid arm accepted')
    h=job_histogram(np.array([0]*65+[1]*8+[33]*4),0,32)
    assert h['jobs_by_token_count']=={1:1,8:1,64:1} and h['small_weight_stages']==1536 and h['weight_stages_total']==2304
    print(json.dumps({'status':'PASS','source_gates':report['source_gates'], 'builds':builds,
                      'base_state_ast':True,'arms':tested},indent=2))

def bootstrap(values):
    rng=random.Random(20261102)
    draws=[stats.median(rng.choices(values,k=len(values))) for _ in range(10000)]
    return {'median_us':stats.median(values),'ci95_us':[percentile(draws,.025),percentile(draws,.975)]}

def job_histogram(ids, lo, nloc):
    # route() uses SK_T=8, FT_T=64 for both HYB flag sets.
    from collections import Counter
    counts=Counter(int(v) for v in ids.reshape(-1).tolist() if lo <= int(v) < lo+nloc)
    jobs=[]
    for count in counts.values():
        jobs.extend([min(64,count-k) for k in range(0,count,64)])
    hist=dict(sorted(Counter(jobs).items()))
    small=sum(v for k,v in hist.items() if k<=16)
    # route jobs each have 32*16 phase-A and 16*16 phase-B weight stages.
    return {'jobs_by_token_count':hist,'jobs_le_16':small,'jobs_total':len(jobs),
            'small_weight_stages':small*768,'weight_stages_total':len(jobs)*768,
            'small_stage_share':small/len(jobs) if jobs else None}

def run():
    import torch
    out=ROOT/'box_out/ladm'; out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'problems/muse_moe_w4a16'))
    ref=module(ROOT/'problems/muse_moe_w4a16/reference.py','ladm_reference')
    layer=module(ROOT/'kernels/muse_moe_w4a16/mb_layer_ladm.py','ladm_layer')
    with phase('one prepare, flag 0',295):
        weights=ref.generate_weights(ref.WEIGHTS_SEED); state=layer.prepare(weights)
    for bs,name in ((128,'hyb'),(256,'hyb7')): assert state['route'][bs]==name
    timer=Timer(torch)
    with phase('overhead calibration',20):
        empty=timer.capture(lambda:None); overhead=stats.median(timer.replay(empty) for _ in range(30))
    result={'status':'RUNNING','overhead_us':overhead,'flags':ARMS,'device_flags':{},'shapes':{},'note':'diagnostic probes; nonzero arms invalid; resets included'}
    for bs in (128,256):
        states={(a,t):layer.ladm_arm(state,mask,bs) for a,mask in ARMS.items() for t in (0,1)}
        for (a,t),s in states.items():
            v=int(s['banks'][0]['k2a33']['st']['hyb'][8].item())
            print('[ladm] device flag',bs,a,t,v,flush=True); assert v==ARMS[a]
            result['device_flags'][f'{bs}:{a}:{t}']=v
        rows=[]
        with phase('bs %d four seeds 30 rotated reps'%bs,180):
            for seed in range(4):
                data=ref.generate_input(bs,'ep8',seed,dist='normal')
                hist=job_histogram(data['topk_ids'].cpu(),data['config']['local_start'],data['config']['local_count'])
                print('[ladm] job histogram',bs,seed,hist,flush=True)
                graphs={}
                for key,s in states.items():
                    layer.ladm_capture_mask(s,ARMS[key[0]])
                    graphs[key]=timer.capture(lambda s=s:layer.kernel_fn(data,s))
                labels=list(graphs); times={key:[] for key in labels}
                for rep in range(30):
                    k=rep%len(labels)
                    for key in labels[k:]+labels[:k]: times[key].append(timer.replay(graphs[key])-overhead)
                med={key:stats.median(values) for key,values in times.items()}
                assert min(med.values())>0
                row={'seed':seed,'job_histogram':hist,'times':{a:[med[(a,0)],med[(a,1)]] for a in ARMS},
                     'replays':{a:[times[(a,0)],times[(a,1)]] for a in ARMS}}
                rows.append(row); print('[ladm] bs',bs,'seed',seed,row['times'],flush=True)
        void={a:any(abs(r['times'][a][0]-r['times'][a][1])>1.0 for r in rows) for a in ARMS}
        # A itself is a required control for every contrast.
        delta={a:bootstrap([stats.mean(r['times'][a])-stats.mean(r['times']['A']) for r in rows]) for a in ARMS}
        median={a:stats.median(stats.mean(r['times'][a]) for r in rows) for a in ARMS}
        valid={a:not (void['A'] or void[a]) for a in ARMS}
        rungs={}
        for left,right in (('A','L'),('L','LF'),('LF','LFG')):
            v=bootstrap([stats.mean(r['times'][left])-stats.mean(r['times'][right]) for r in rows])
            v['valid']=valid[left] and valid[right]; rungs[left+' -> '+right]=v
        contrasts={}
        for left,right in (('A','M'),('L','LM'),('LF','LFM'),('LFG','LFGM')):
            v=bootstrap([stats.mean(r['times'][left])-stats.mean(r['times'][right]) for r in rows])
            v['valid']=valid[left] and valid[right]
            contrasts[left+' -> '+right]=v
        result['shapes'][str(bs)]={'m_saving_us':contrasts,'seeds':rows,'median_us':median,'delta_vs_A':delta,'void':void,'valid_vs_A':valid,'ladder_saving_us':rungs,'residual_fixed_floor_us':median['LFG'] if valid['LFG'] else None}
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    result['status']='COMPLETE'
    lines=['# LADM diagnostic', '', 'Savings and deltas include the ordered reset, and are not proven layer gains.', '']
    for bs in ('128','256'):
        s=result['shapes'][bs]; lines += ['## bs'+bs,'','Medians (us): '+str(s['median_us']), '', 'Delta vs A, seed-bootstrap 95% CIs (us): '+str(s['delta_vs_A']), '', 'VOID controls: '+str(s['void']), '', 'Ladder savings (us): '+str(s['ladder_saving_us']), '', 'Residual fixed floor (LFG, us): '+str(s['residual_fixed_floor_us']),'']
    lines += ['| Rung | bs128 saving (us) | bs256 saving (us) |', '|---|---:|---:|']
    for rung in ('A -> L','L -> LF','LF -> LFG'):
        def display(bs):
            v=result['shapes'][bs]['ladder_saving_us'][rung]
            return ('%.2f %s' % (v['median_us'],v['ci95_us'])) if v['valid'] else 'VOID'
        lines.append('| %s | %s | %s |' % (rung,display('128'),display('256')))
    for label,rung in (('progressive finalize','L -> LF'),('prologue','LF -> LFG')):
        vals=[(bs,result['shapes'][bs]['ladder_saving_us'][rung]) for bs in ('128','256')]
        eligible=[v['median_us'] >= (4 if bs=='128' else 6) for bs,v in vals if v['valid']]
        verdict='GO-to-build' if any(eligible) else ('NO-GO' if len(eligible)==2 else 'VOID / insufficient valid shapes')
        lines.append('%s: %s (net threshold: >=4 us at bs128 or >=6 us at bs256).' % (label,verdict))
    v=result['shapes']['128']['m_saving_us']
    go=v['L -> LM']['valid'] and v['A -> M']['valid'] and v['L -> LM']['median_us']>=8 and v['A -> M']['median_us']>=3
    lines += ['M contrasts (saving us, seed-bootstrap CIs):',str({bs:result['shapes'][bs]['m_saving_us'] for bs in ('128','256')}), 'Pre-registered M verdict: '+('PAY' if go else 'NO-GO or VOID') ]
    (out/'summary.md').write_text('\n'.join(lines)+'\n'); (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    if sys.argv[1:]==['dry']: dry()
    elif sys.argv[1:]==['run']: run()
    else: raise SystemExit('usage: ladm_ab.py dry|run')

#!/usr/bin/env python3
"""F1 timing-only: run only on the authorized pod. B/C outputs are invalid."""
import ast
import json
import random
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'kernels/lab'))
from moephase import Timer, phase, module, percentile

UNIT_BYTES = 36864
EXPERT_BYTES = 768 * UNIT_BYTES
ARMS = ('A', 'B', 'C', 'D')
MODES = (0, 1, 2, 0)

def arm_state(state, flag):
    # Same expression as _mbl_call; clone just its mapping/list, not the banks.
    old = state['banks'][0]['k2a33']['st']
    st = dict(old)
    st['hyb'] = list(old['hyb'])
    st['hyb'][8] = flag
    bank = dict(state['banks'][0])
    bank['k2a33'] = dict(bank['k2a33'], st=st)
    result = dict(state)
    result['banks'] = dict(state['banks'], **{})
    result['banks'][0] = bank
    return result

def routing(ids, lo):
    counts = [0] * 32
    for row in ids:
        for item in row:
            e = int(item) - lo
            if 0 <= e < 32:
                counts[e] += 1
    jobs = [(n + 63) // 64 for n in counts]
    return {'counts': counts, 'jobs_per_expert': jobs,
            'distribution': dict(sorted(Counter(jobs).items())),
            'extra_passes': sum(max(0, j-1) for j in jobs),
            'touched': sum(j > 0 for j in jobs), 'jobs': sum(jobs)}

def bootstrap(rows, arm):
    pairs = [(r['times'][arm] - r['times']['A']) for r in rows]
    rng = random.Random(20261101)
    draws = [stats.median(rng.choices(pairs, k=len(pairs))) for _ in range(10000)]
    return {'median_paired_us': stats.median(pairs),
            'ci95_us': [percentile(draws, .025), percentile(draws, .975)]}

def dry():
    # Execute the *actual* base routing/dispatch functions with stand-ins.
    p = ROOT / 'kernels/muse_moe_w4a16/mb_layer.py'
    tree = ast.parse(p.read_text(encoding='utf-8'))
    funcs = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in ('_mbl_banks', '_mbl_call', 'kernel_fn')]
    assert len(funcs) == 3
    seen = []
    class Mod:
        def okc_run(self, *args): seen.append(args)
    class Shape:
        def __init__(self, m): self.shape = (m, 4096)
    class Tensor:
        shape=(256,)
        def __getitem__(self, item): return self
        def view(self, *args): return self
        def reshape(self, *args): return self
    class FakeTorch:
        @staticmethod
        def cat(items, *args, **kwargs): return Tensor()
    ph=Tensor()
    fake_weights={k:Tensor() for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs','post_norm_w')}
    def pack(w, dev):
        assert set(w)==set(fake_weights)
        return dict({k:Tensor() for k in ('ws13','ws2','gsc13','gsc2','nw')}, hyb=[ph]*14)
    env={'torch':FakeTorch, '_mbl_pack_bank':pack, '_HYB':{}, '_HYBLO':{}}
    exec(compile(ast.Module(body=funcs,type_ignores=[]), str(p), 'exec'), env)
    banks=env['_mbl_banks'](fake_weights, 'cpu')
    assert banks[0]['k2a33']['st']['hyb'][8] is ph
    # Make later elements distinct so routing assertions also test list identity.
    banks[0]['k2a33']['st']['hyb'] = list(range(14))
    state = {'banks': banks, 'out': {128: object(), 256: object()},
             'mods': {'hyb': Mod(), 'hyb7': Mod()}, 'route': {128:'hyb', 256:'hyb7'},
             'h_sc': object(), 'nsm': 132}
    for m in (128, 256):
        for mode in MODES:
            flag = [mode]
            s = arm_state(state, flag)
            data = {'hidden_states':Shape(m), 'topk_ids': object(), 'topk_weights': object(), 'config':{'local_start':0}}
            assert env['kernel_fn'](data,s) is state['out'][m]
            assert seen[-1][0][8] is flag and seen[-1][0][12:] == [12,13]
    assert state['banks'][0]['k2a33']['st']['hyb'][8] == 8
    for unit in (0, 111, 112, 113, 500, 16383):
        assert (unit % 112) * UNIT_BYTES + UNIT_BYTES <= 112 * UNIT_BYTES
    assert routing([[0]*64, [0]*64, [1]*65], 0)['extra_passes'] == 2
    print('[f1] CPU dry PASS: base dispatch, all arms, bank isolation, aliases, route accounting', flush=True)

def run():
    import torch
    out = ROOT/'box_out/f1'
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT/'problems/muse_moe_w4a16'))
    ref = module(ROOT/'problems/muse_moe_w4a16/reference.py', 'f1_reference')
    layer = module(ROOT/'kernels/muse_moe_w4a16/mb_layer_l2r.py', 'f1_layer')
    with phase('prepare and validate mode 0', 295):
        weights = ref.generate_weights(ref.WEIGHTS_SEED)
        state = layer.prepare(weights)
    for bs,name in ((128,'hyb'),(256,'hyb7')):
        assert state['route'][bs] == name, (bs, state['route'][bs])
    timer = Timer(torch)
    with phase('calibrate overhead', 20):
        empty = timer.capture(lambda: None)
        overhead = stats.median(timer.replay(empty) for _ in range(30))
    # One binary and one prepared state: separate fixed device flags per graph.
    flags = {a: torch.zeros(1, dtype=torch.int32, device='cuda') for a in ARMS}
    states = {a: arm_state(state, flags[a]) for a in ARMS}
    with phase('mode 0 correctness replay', 25):
        for bs in (128,256):
            data=ref.generate_input(bs,'ep8',0,dist='normal')
            x=layer.kernel_fn(data,states['A']).clone()
            y=layer.kernel_fn(data,states['D']).clone()
            torch.cuda.synchronize()
            if not torch.allclose(x,y,rtol=1e-3,atol=1e-3,equal_nan=False):
                raise RuntimeError('mode 0 A/D replay mismatch')
    for a,m in zip(ARMS,MODES): flags[a].fill_(m)
    for a in ARMS:
        val = int(flags[a].item())
        print('[f1] flag', a, val, flush=True)
        assert val == dict(zip(ARMS, MODES))[a]
    result = {'overhead_us':overhead, 'shapes':{}, 'flags':dict(zip(ARMS, MODES)),
              'note':'B/C invalid output. Flags are independent device tensors; all arms share patched binary, scratch and banks; graphs replay serially.'}
    for bs in (128,256):
        rows=[]
        with phase('bs=%d seeds 0..7'%bs, 95):
            for seed in range(8):
                data=ref.generate_input(bs,'ep8',seed,dist='normal')
                route=routing(data['topk_ids'].cpu().tolist(),int(data['config']['local_start']))
                torch.cuda.synchronize()
                graphs={a:timer.capture(lambda a=a:layer.kernel_fn(data,states[a])) for a in ARMS}
                times={a:[] for a in ARMS}
                for rep in range(30):
                    order=ARMS[rep%4:]+ARMS[:rep%4]
                    for a in order: times[a].append(timer.replay(graphs[a])-overhead)
                med={a:stats.median(times[a]) for a in ARMS}
                if min(med.values()) <= 0: raise RuntimeError('nonpositive measured time')
                # Ideal HBM accounting: A all jobs, B one pass/expert, C zero.
                sizes={'A':route['jobs']*EXPERT_BYTES,'B':route['touched']*EXPERT_BYTES,'C':0,'D':route['jobs']*EXPERT_BYTES}
                rows.append({'seed':seed,'route':route,'times':med,'replays_us':times,'ideal_hbm_bytes':sizes,
                             'ideal_hbm_TBps':{a:sizes[a]/(med[a]*1e6) for a in ARMS}})
                print('[f1] bs=%d seed=%d us=%s jobs=%s'%(bs,seed,med,route['distribution']),flush=True)
        result['shapes'][str(bs)]={'seeds':rows,'median_us':{a:stats.median(r['times'][a] for r in rows) for a in ARMS},
                                    'delta_vs_A':{a:bootstrap(rows,a) for a in ('B','C','D')}}
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    render(result,out)

def render(result,out):
    header = ['# F1 timing-only', '', 'Pre-registered: PASS-to-build if B or C saves >=15 us at bs256 and >=5 us at bs128, and |D-A| <=1.0 us at both shapes. KILL read-once if B saves <5 us at bs256; 5 to <15 us: report only.',
              '', 'B aliases later jobs; C aliases all jobs. All four arms use the same patched binary; B/C outputs are invalid. Savings = A-B or A-C. Per-bank 4,128,768 B windows may be evicted; HBM bytes/TB/s below are ideal estimates, not counters.', '']
    d = {bs: result['shapes'][bs]['delta_vs_A'] for bs in ('128','256')}
    stable = all(abs(d[bs]['D']['median_paired_us']) <= 1 for bs in d)
    savings = {a:{bs:-d[bs][a]['median_paired_us'] for bs in d} for a in ('B','C')}
    passed = stable and any(savings[a]['256'] >= 15 and savings[a]['128'] >= 5 for a in savings)
    verdict = ('VOID: twin control' if not stable else 'PASS-to-build' if passed else
               'KILL read-once family' if savings['B']['256'] < 5 else 'REPORT ONLY')
    header += ['Verdict: '+verdict, '']
    for bs in ('128','256'):
        s=result['shapes'][bs]
        header += ['## bs'+bs, '', 'Medians (us): '+str(s['median_us']), '', 'Paired deltas (arm - A, us; seed bootstrap 95% CI): '+str(s['delta_vs_A']), '',
                   'Seed routing and ideal HBM bytes / TB/s:', '']
        for r in s['seeds']:
            header.append('- seed %d jobs/expert=%s extra=%d bytes=%s TB/s=%s' % (r['seed'],r['route']['jobs_per_expert'],r['route']['extra_passes'],r['ideal_hbm_bytes'],r['ideal_hbm_TBps']))
        header.append('')
    (out/'summary.md').write_text('\n'.join(header)+'\n')
    print('[f1] summary written',flush=True)

if __name__ == '__main__':
    if sys.argv[1:] == ['dry']: dry()
    elif sys.argv[1:] == ['run']: run()
    else: raise SystemExit('usage: l2r_ab.py dry|run')

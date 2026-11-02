#!/usr/bin/env python3
"""SEG timing-only bound. `dry` is CPU-only; `run` is lead-run H200-only.

Each arm and A/A twin is a distinct compiled module with its own scratch.
One full base prepare, seven lightweight module prepares sharing immutable
weights, all within one 295 s budget. No prepare inside a seed/shape loop.
BT4 Timer supplies 256 MiB flush, calibrated 300 us gate and graph events.
Positive deltas mean base minus probe. Seed-cluster bootstrap (four seeds),
not a bootstrap of 120 supposedly independent replays. Any seed/shape twin
drift >1 us voids that arm; base drift voids every comparison in the batch.
SEG1 is a diagnostic bound, not a numerically valid implementation.
"""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as stats
import sys
import time
from types import SimpleNamespace

from bt4_ab import Timer, module, percentile, phase, summarize

ROOT = Path(__file__).resolve().parents[2]
SHAPES = (25, 32, 40)
PROBES = ('SEG0', 'SEG1', 'SEG2')
ARMS = ('base', 'base_twin', 'SEG0', 'SEG0_twin', 'SEG1', 'SEG1_twin', 'SEG2', 'SEG2_twin')
SEEDS = (0, 1, 2, 3)
GATES = {'kill_SEG1_bs32_us': 3.0, 'pass_to_build_SEG1_bs32_us': 8.0,
         'void_twin_seed_median_us': 1.0, 'reps': 30, 'replay_checks': 3,
         'prepare_s': 295, 'batch_s': 700}


def prepare_all(layer, weights):
    started = time.monotonic()
    shared = layer.prepare(weights)
    assert all(shared['route'][bs] == 'bf2' for bs in SHAPES)
    states = {name: layer.seg_prepare_module(shared, name, started+295) for name in ARMS}
    elapsed = time.monotonic() - started
    assert elapsed < 295, ('total prepare budget', elapsed)
    assert len({id(s['mods']['bf2']) for s in states.values()}) == len(ARMS)
    assert len({id(s['vml_sc']['bf2']) for s in states.values()}) == len(ARMS)
    return states, elapsed


def bootstrap_ci(values):
    # Exhaustive ordinary seed bootstrap: 4^4=256 resamples, deterministic.
    assert len(values) == 4
    samples = [stats.median(values[i] for i in indices)
               for indices in itertools.product(range(4), repeat=4)]
    return [percentile(samples, .025), percentile(samples, .975)]


def measure(timer, graphs, overhead):
    values = {name: [] for name in ARMS}
    for rep in range(30):
        offset = rep % len(ARMS)
        for name in ARMS[offset:] + ARMS[:offset]:
            values[name].append(timer.replay(graphs[name]))
    times = {name: summarize(v, overhead) for name, v in values.items()}
    twins = {a: stats.median(x-y for x, y in zip(values[a], values[a+'_twin']))
             for a in ('base',) + PROBES}
    delta = {}
    for a in PROBES:
        # Paired within repetition, average the two independently loaded modules.
        delta[a] = stats.median((b+bt-p-pt)/2 for b, bt, p, pt in
                               zip(values['base'], values['base_twin'], values[a], values[a+'_twin']))
    return {'times': times, 'twin_seed_median_us': twins, 'paired_saving_us': delta}


def aggregate(rows):
    summary = {}
    for bs in SHAPES:
        seeds = rows[str(bs)]
        assert sorted(r['seed'] for r in seeds) == list(SEEDS)
        result = {'median_us': {a: stats.median(r['times'][a]['us'] for r in seeds) for a in ARMS},
                  'arms': {}}
        for a in PROBES:
            values = [r['paired_saving_us'][a] for r in seeds]
            bad = [{'bs': int(shape), 'seed': r['seed'], 'arm': k, 'twin_us': r['twin_seed_median_us'][k]}
                   for shape, observations in rows.items() for r in observations for k in ('base', a)
                   if abs(r['twin_seed_median_us'][k]) > 1.0]
            median = stats.median(values)
            verdict = 'VOID' if bad else 'REPORT ONLY'
            if a == 'SEG1' and bs == 32 and not bad:
                verdict = 'KILL segment-end family' if median < 3.0 else 'PASS-to-build' if median >= 8.0 else 'REPORT ONLY'
            result['arms'][a] = {'saving_us': median, 'seed_bootstrap_95_ci_us': bootstrap_ci(values),
                                 'seed_savings_us': values, 'void_reasons': bad, 'verdict': verdict}
        summary[str(bs)] = result
    return summary


def replay_health(layer, state, bs, output, torch):
    sc = state['vml_sc']['bf2']  # exact base accessor, mb_layer.py:6332
    assert layer._mbl_health('bf2', state), ('VML error flag', bs)
    # vm_fin clears these exact live prefixes; oversized allocated tails unused.
    for key, n in (('err', 4), ('xready', layer._vml_JMAX),
                   ('doneA', layer._vml_JMAX*16), ('cnt', 2*layer._vml_SLOTS*4)):
        assert int(torch.count_nonzero(sc[key][:n]).item()) == 0, (bs, key)
    assert int(torch.count_nonzero(sc['y']).item()) == 0, ('B partials not rearmed', bs)
    return {'control_zero': True, 'y_zero': True, 'finite_output': bool(torch.isfinite(output).all().item())}


def route_instances(data, torch):
    """Untimed observation of actual seg<PB,N> cases for this router seed."""
    lo = int(data['config']['local_start'])
    ids = data['topk_ids'].reshape(-1)
    local = ids[(ids >= lo) & (ids < lo+32)]-lo
    counts = torch.bincount(local.to(torch.int64), minlength=32).cpu().tolist()
    # bs<=40, below bf2's 64-token job size: each expert occupies one job.
    buckets = sorted({8*((n+7)//8) for n in counts if n})
    return {'engine':'bf2', 'kernels':['vm_main','vm_fin'], 'expert_token_counts':counts,
            'seg_N':buckets, 'stage_m_N':[n for n in buckets if n<=16],
            'RS_wgmma_N':[n for n in buckets if n>16],
            'dispatch':'mb_layer.py:5521-5526; shape route:6187-6188; defines:5991'}


def save(out, result):
    out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


def run(out):
    import torch
    result = {'status': 'RUNNING', 'gates': GATES, 'rows': {},
              'timing_units': 'MEASURED microseconds, CUDA events; savings are paired measured differences',
              'segments_per_CTA': 'omitted: enabling marks would alter the measured build',
              'numeric_warning': 'all SEG arms timing-only; SEG1 discards partials'}
    save(out, result)
    started = time.monotonic()
    try:
        with phase('SEG total wall', 690):
            # This budget covers weights plus ALL module prepares, not 295 per arm.
            with phase('SEG weights and all prepares', 295):
                sys.path.insert(0, str(ROOT/'problems/muse_moe_w4a16'))
                ref = module(ROOT/'problems/muse_moe_w4a16/reference.py', 'seg_reference')
                layer = module(ROOT/'kernels/muse_moe_w4a16/mb_layer_seg.py', 'seg_layer')
                weights = ref.generate_weights(ref.WEIGHTS_SEED)
                states, elapsed = prepare_all(layer, weights)
                result['prepare_s'] = elapsed
                result['module_cuda_sha256'] = {a: hashlib.sha256(layer.seg_sources(a)[0].encode()).hexdigest() for a in ARMS}
                result['routes'] = {str(bs): states['base']['route'][bs] for bs in SHAPES}
            timer = Timer(torch)
            empty = timer.capture(lambda: None)
            overhead = stats.median(timer.replay(empty) for _ in range(30))
            result.update(overhead_us=overhead, gate_cycles=timer.gate_cyc)
            for bs in SHAPES:
                rows = result['rows'][str(bs)] = []
                for seed in SEEDS:
                    data = ref.generate_input(bs, 'ep8', seed, dist='normal')
                    torch.cuda.synchronize()  # publish default-stream inputs before capture stream
                    graphs, checks = {}, {}
                    for name in ARMS:
                        state = layer.seg_arm(states, name, bs)
                        assert int(data['config']['local_start']) == 0
                        graphs[name] = timer.capture(lambda state=state: layer.kernel_fn(data, state))
                        checks[name] = []
                        for replay in range(3):
                            with torch.cuda.stream(timer.stream):
                                graphs[name].replay()
                            torch.cuda.synchronize()
                            checks[name].append(replay_health(layer, state, bs, state['out'][bs], torch))
                        if name in ('base', 'base_twin'):
                            assert all(c['finite_output'] for c in checks[name])
                    # Base twins also pass the real base numeric comparator; probe equality is not required.
                    err = layer._k2a_rel_err(states['base']['out'][bs], states['base_twin']['out'][bs])[0]
                    assert err < 1e-3, ('base twin numeric mismatch', bs, seed, err)
                    row = measure(timer, graphs, overhead)
                    row.update(seed=seed, replay_checks=checks, base_twin_relative_error=err,
                               route_instances=route_instances(data,torch))
                    # Repeat control-health check after timing; SEG1 fix floats intentionally persist.
                    row['post_timing_health'] = {a: replay_health(layer, states[a], bs, states[a]['out'][bs], torch) for a in ARMS}
                    rows.append(row)
                    print('[seg]', bs, seed, row['paired_saving_us'], row['twin_seed_median_us'], flush=True)
                    save(out, result)
            result['summary'] = aggregate(result['rows'])
            result['status'] = 'COMPLETE'
            print(json.dumps(result['summary'], indent=2), flush=True)
    except BaseException as exc:
        result.update(status='VOID', error=repr(exc))
        raise
    finally:
        result['wall_s'] = time.monotonic()-started
        save(out, result)
    return result


def dry():
    """Execute new prepare/selection paths against real base AST constructors.

    Strict tensor/module stand-ins; no CUDA/torch/numpy import, file writes or
    permissive attribute fallback. Only native build and weight payload packing
    are substituted. Bank construction and every launch argument are real code.
    """
    sys.path.insert(0, str(ROOT/'tools'))
    import seg_patches as patch
    base = patch.BASE.read_text(encoding='utf-8')
    generated = patch.DEST.read_text(encoding='utf-8')
    report = patch.cpu_gates(base, generated)
    tree = ast.parse(base)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    env = patch.environment(base)
    builds, calls = [], []

    class Tensor:
        def __init__(self, shape, dtype='float32', device='cpu', value=0):
            self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
            self.dtype, self.device, self.value = dtype, device, value
        def __getitem__(self, index):
            if isinstance(index, int):
                assert 0 <= index < self.shape[0]
                return Tensor(self.shape[1:], self.dtype, self.device, self.value)
            assert isinstance(index, slice)
            return Tensor((len(range(*index.indices(self.shape[0]))),)+self.shape[1:], self.dtype, self.device, self.value)
        def __int__(self):
            assert not self.shape or math.prod(self.shape) == 1
            return int(self.value)

    class Torch:
        int32='int32'; int64='int64'; bfloat16='bfloat16'; float32='float32'
        @staticmethod
        def zeros(shape, dtype='float32', device='cpu'): return Tensor(shape, dtype, device)
        empty = zeros
        @staticmethod
        def cat(ts, dim=0):
            assert dim == 0 and ts and all(t.shape[1:] == ts[0].shape[1:] for t in ts)
            return Tensor((sum(t.shape[0] for t in ts),)+ts[0].shape[1:], ts[0].dtype)

    class Mod:
        def __init__(self, name): self.name = name
        def vml_cfg(self): return (64, 8, 3, 64, 16, 0, 0, 0, 1, 0, 0)
        def vml_run(self, tensors, values):
            assert len(tensors) == 20 and len(values) == 22
            assert values[0] in SHAPES and values[1] == 0 and values[2] == 132
            assert values[4:6] == [3, 193], values  # vm_main + vm_fin, no external PDL
            assert tensors[3:8] == [env['banks'][0]['k2a33']['st'][k] for k in ('ws13','ws2','gsc13','gsc2','nw')]
            calls.append((self.name, values[0]))

    env.update(torch=Torch, hashlib=hashlib, time=time)
    needed = ('_k2a_bucket_of','_vml_mag','_vml_alloc','_vml_cfg','_vml_cost_ext','_vml_pool_ext',
              '_vml_seg_ext','_vml_pdel_ext','_vml_call','_mbl_banks','_mbl_candidates',
              '_mbl_select','_mbl_call','kernel_fn','_mbl_deadline','_mbl_health')
    for name in needed:
        exec(compile(ast.Module([functions[name]], type_ignores=[]), '<real base '+name+'>', 'exec'), env)
    # Payload packing is GPU-only. The REAL _mbl_banks executes its slicing,
    # concatenation and banks[lo]={"k2a33":{"st":st}} without replacement.
    def pack_stub(w, dev):
        assert w['w13'].shape[0] == 32 and dev == 'cpu'
        return {k: Tensor((32, 16)) for k in ('ws13','ws2','gsc13','gsc2','nw')}
    env.update(_mbl_pack_bank=pack_stub, _HYB={'w':None}, _HYBLO={'w':None})
    weights = {k: Tensor((256,16)) for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs')}
    weights['post_norm_w'] = Tensor(4096)
    banks = env['_mbl_banks'](weights, 'cpu')
    mods = {name: Mod(name) for name in env['MBL_ARMS']}
    env.update(weights=weights, t0=time.monotonic(), mods=mods, banks=banks,
               sc={'bf2':env['_vml_alloc']('cpu',132)}, hsc=[], hfull=[], nsm=132,
               output=Tensor((320,4096),'bfloat16'))
    state_expr = next(n.value for n in functions['prepare'].body if isinstance(n,ast.Assign) and
                      any(isinstance(t,ast.Name) and t.id=='state' for t in n.targets))
    shared = eval(compile(ast.Expression(state_expr), '<real prepare state>', 'eval'), env)
    shared['good'] = {name:True for name in mods}
    env['_mbl_select'](shared)
    def build_stub(**kw):
        name = next(a for a in sorted(ARMS, key=len, reverse=True) if kw['name'].startswith('vml_moe_seg_'+a.lower()+'_'))
        src, cpp, flags = env['seg_sources'](name)
        assert kw['cuda_sources'] == src and kw['cpp_sources'] == cpp and kw['extra_cuda_cflags'] == flags
        assert kw['functions'] == ['vml_run','vml_cfg']
        builds.append(name)
        return Mod(name)
    env['_mbl_load_inline'] = build_stub
    exec(compile(generated[generated.index('# SEG_BEGIN'):], '<new prepare paths>', 'exec'), env)
    prepare_calls = []
    def base_prepare(w):
        assert w is weights
        prepare_calls.append(1)
        return shared
    layer = SimpleNamespace(prepare=base_prepare, seg_prepare_module=env['seg_prepare_module'])
    states, elapsed = prepare_all(layer, weights)
    assert prepare_calls == [1] and builds == list(ARMS[1:])
    tested = []
    for bs in SHAPES:
        data = {'hidden_states':Tensor((bs,4096),'bfloat16'), 'topk_ids':Tensor((bs,8),'int32'),
                'topk_weights':Tensor((bs,8)), 'config':{'local_start':0}}
        for name in ARMS:
            state = env['seg_arm'](states,name,bs)
            for _ in range(3):
                assert env['kernel_fn'](data,state) is state['out'][bs]
                assert env['_mbl_health']('bf2',state)
            tested.append((bs,name))
    # An invented key must fail locally, never fall back to an unrelated path.
    bad = dict(states['base'], banks={0:{'k2a':{'st':{}}}})
    try: env['kernel_fn'](data,bad)
    except KeyError: pass
    else: raise AssertionError('invented bank accessor passed')
    for name, bs in (('bogus',32), ('SEG0',24)):
        try: env['seg_arm'](states,name,bs)
        except ValueError: pass
        else: raise AssertionError('invalid arm/shape passed')
    try: env['seg_prepare_module'](shared,'SEG0',time.monotonic()+100)
    except ValueError: pass
    else: raise AssertionError('duplicate module prepare accepted')
    # Exercise rotated timer and report/VOID paths without a CUDA import.
    class FakeTimer:
        def replay(self, name): return {'base':20,'SEG0':19,'SEG1':11,'SEG2':18}[name.removesuffix('_twin')]
    rows = {str(bs):[dict(measure(FakeTimer(),dict(zip(ARMS,ARMS)),2),seed=s) for s in SEEDS] for bs in SHAPES}
    summary = aggregate(rows)
    assert summary['32']['arms']['SEG1']['verdict'] == 'PASS-to-build'
    rows['32'][0]['twin_seed_median_us']['SEG1'] = 1.01
    assert aggregate(rows)['32']['arms']['SEG1']['verdict'] == 'VOID'
    rows['32'][0]['twin_seed_median_us']['SEG1'] = 0
    for row in rows['32']: row['paired_saving_us']['SEG1'] = 2.99
    assert aggregate(rows)['32']['arms']['SEG1']['verdict'] == 'KILL segment-end family'
    report.update(status='PASS', real_base_bank_constructor=True, real_prepare_state_expression=True,
                  prepare_calls=len(prepare_calls), module_builds=builds, paths=tested,
                  dry_launches=len(calls), dry_prepare_s=elapsed, timing_statistics_checks=True,
                  limitation='strict CPU substitutes do not prove CUDA ordering or GPU performance')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('dry','run'))
    parser.add_argument('--out', type=Path, default=ROOT/'box_out/seg')
    args = parser.parse_args()
    if args.mode == 'dry': dry()
    else: run(args.out)


if __name__ == '__main__':
    main()

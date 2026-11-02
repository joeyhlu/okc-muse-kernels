#!/usr/bin/env python3
"""SL1 harness (TINY / TINY2, bs 1 and 8): CORRECTNESS FIRST, then hang-free replays, then paired timing.

Variant: kernels/muse_moe_w4a16/mb_layer_sl1.py (tools/sl1_patches.py). One prepare(); prepare's flag-0 validation
runs on the SL1 binaries. Arms (Params.sl1): base 0, TM 1 (A->B decoupling), PC 2 (progressive combine), TM+PC 3;
diagnostics TM_NOPF 5 / TM+PC_NOPF 7 are correctness/protocol only. Every (mask, M) arm owns a fresh scratch.
Correctness (bs 1, 8 x seeds 0..3 normal + 2 skewed + one no-local routing): each arm's eager output and 3 graph
replays vs base (flag 0) rel-L2 <= 2e-3, vs the problem reference (problems/muse_moe_w4a16 reference.check_implementation,
REL_L2_TOL 1.5 %), finite, plus a counter snapshot after EVERY launch (generation +1, next banks zero, used banks
exactly 512 per job / 256 x L_t per token, scratch guards). Timing protocol = bt4b_ab.py (256 MB flush, calibrated
300 us gate, empty-graph overhead subtracted, 30 rotated reps, seeds 0..3 via generate_input(bs, "ep8", seed,
dist="normal"), A/A twin per arm); Timer / phase / module / percentile / summarize / measure_graphs are verbatim.
"""
import argparse
import ast
import contextlib
import importlib.util
import json
import math
import random
from pathlib import Path
import signal
import statistics as stats
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SHAPES = (1, 8)
VARIANT_ARMS = {'base': 0, 'TM': 1, 'PC': 2, 'TM+PC': 3}
CONTROL_ARMS = {'ship': 0}  # unpatched ship binaries (flag-0 semantics): flag-0 cost of the SL1 binary
ARMS = dict(VARIANT_ARMS, **CONTROL_ARMS)  # timed arms (each with an A/A twin)
DIAG_ARMS = {'TM_NOPF': 5, 'TM+PC_NOPF': 7}
ALL_ARMS = dict(ARMS, **DIAG_ARMS)


def get_arm(layer, state, name, bs):
    if name in CONTROL_ARMS:
        return layer.sl1_ship_arm(state, bs)
    arm = layer.sl1_arm(state, ALL_ARMS[name], bs)
    assert int(state['sl1_flag'].item()) == ALL_ARMS[name]
    return arm
EXPECTED_ROUTES = {1: 'tiny', 8: 'tiny2'}
JMAX, MMAX, EMAX, RB_A, RB_B = 88, 384, 3072, 512, 256
# MEASURED bounds (box_out/bt4b, net us saved, bs1 / bs8; garbage outputs) for the matching arm.
BOUNDS = {'TM': {'src': 'AB1', 1: 1.21, 8: 1.86}, 'PC': {'src': 'FIN', 1: 1.15, 8: 1.64},
          'TM+PC': {'src': 'AB2+FIN', 1: 2.47, 8: 3.07}, 'TM+PC_alt': {'src': 'AB1+COMBSKIP', 1: 1.84, 8: 2.72}}
GATES = {'controls': {'twin_us': 1.0}, 'rel_vs_base': 2e-3, 'ref_rel_l2': 1.5e-2,
         'pass_combined_us': {1: 1.2, 8: 1.5}, 'capture_fraction': 0.5}
CORR_CASES = tuple(('normal', s) for s in range(4)) + (('skewed', 0), ('skewed', 1), ('none', 0), ('conc', 0))


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

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

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


# ---------------------------------------------------------------------------------------------------------------
def disable_pdl(state):
    # Ordered replay for every engine (the SL1 binaries also refuse a nonzero flag with PDL on).
    for name in ('tiny', 'tiny2', 'hyb', 'hyb7'):
        if state['mods'].get(name) is not None:
            state['mods'][name].okc_set_pdl(0)
    class VmsNoPDL:
        def __init__(self, mod): self.mod = mod
        def run(self, *args):
            assert len(args) == 18, 'VMS run ABI drift'
            args = list(args); args[-2] = 0
            return self.mod.run(*args)
    for name in ('vms', 'vms_hi'):
        if state['mods'].get(name) is not None:
            state['mods'][name] = VmsNoPDL(state['mods'][name])


def verify_routes(state):
    for bs, name in EXPECTED_ROUTES.items():
        assert state['route'][bs] == name, ('fallback invalidates SL1', bs, state['route'][bs], name)


def tolist(x):
    return x.cpu().tolist() if hasattr(x, 'cpu') else list(x)


def routing_stats(ids_rows, lo=0):
    """njS (distinct local experts) and L_t (local pairs per token) from topk_ids rows (python ints)."""
    local = sorted({e - lo for row in ids_rows for e in row if 0 <= e - lo < 32})
    L = [sum(1 for e in row if 0 <= e - lo < 32) for row in ids_rows]
    return {'njS': len(local), 'L': L, 'experts': local}


def snapshot_lists(ints, ss0, G, mask, rs, M):
    """Counter / bank invariants after a COMPLETED launch (pure python lists; GPU path and dry run share it).
    ints = scratch[0] int32 words, ss0 = scratch[1][:EMAX] floats. Returns the next launch's generation."""
    assert len(ints) >= 8 + 1024 + EMAX
    ss1_bits = ints[8 + 1024:8 + 1024 + EMAX]
    def zero_ss(par):
        return all(v == 0 for v in ss1_bits) if par else all(v == 0.0 for v in ss0[:EMAX])
    dA = lambda b: ints[8 + b * JMAX:8 + (b + 1) * JMAX]
    tk = lambda b: ints[8 + 2 * JMAX + b * MMAX:8 + 2 * JMAX + (b + 1) * MMAX]
    assert ints[0] == ints[1] == ints[2] == 0, ('ctr[0..2]', ints[:3])
    if mask & 2:
        assert ints[6] % G == 0, ('PC exit counter not a multiple of G', ints[6], G)
        gen = ints[6] // G
        nb = gen & 1
        assert ints[3] == 0 and ints[4] == 0 and ints[5] == 0, ('PC touched ship counters', ints[3:6])
        assert all(v == 0 for v in dA(nb)) and all(v == 0 for v in tk(nb)), 'next PC bank not zero'
        assert zero_ss(nb), 'next ss parity not zero'
        if gen >= 1:
            ub = (gen - 1) & 1
            want = [RB_A] * rs['njS'] + [0] * (JMAX - rs['njS'])
            assert dA(ub) == want, ('used doneA bank', dA(ub)[:rs['njS'] + 2], rs['njS'])
            wt = [RB_B * x for x in rs['L']] + [0] * (MMAX - M)
            assert tk(ub) == wt, ('used token bank', tk(ub)[:M], rs['L'])
    else:
        gen = ints[3]
        assert ints[6] == 0, 'flag-0/TM arm touched the PC exit counter'
        assert ints[4 + (gen & 1)] == 0, ('next barrier counter', ints[4:6], gen)
        if gen >= 1:
            assert ints[4 + ((gen - 1) & 1)] == G, ('lost CTA at the barrier', ints[4:6], gen)
        assert all(v == 0 for v in dA(0)), 'ship end zeroing of doneA bank 0'
        assert all(v == 0 for v in dA(1)) and all(v == 0 for v in tk(0)) and all(v == 0 for v in tk(1)), 'PC banks touched'
        assert zero_ss(gen & 1), 'next ss parity not zero'
    return gen


def snapshot(arm, bs, mask, rs, G):
    """Synchronous, UNTIMED readback of the arm's own scratch."""
    gen = snapshot_lists(tolist(arm['h_sc'][0]), tolist(arm['h_sc'][1][:EMAX]), G, mask, rs, bs)
    assert all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in arm['h_full']), 'scratch guard'
    return gen


def make_input(ref, torch, bs, dist, seed):
    if dist == 'none':  # no local pair at all (njS = 0): every token routes to experts 32..39 only
        d = ref.generate_input(bs, 'ep8', seed, dist='normal')
        d['topk_ids'] = (torch.arange(32, 40, dtype=torch.int32, device=d['topk_ids'].device)
                         .repeat(bs, 1).contiguous())
        return d
    if dist == 'conc':  # every token on local experts 0..7: cnt = bs per job, L_t = 8 (prefetch / counter worst case)
        d = ref.generate_input(bs, 'ep8', seed, dist='normal')
        d['topk_ids'] = (torch.arange(0, 8, dtype=torch.int32, device=d['topk_ids'].device)
                         .repeat(bs, 1).contiguous())
        return d
    return ref.generate_input(bs, 'ep8', seed, dist=dist)


def correctness(layer, state, ref, weights, torch):
    """Every arm incl. diagnostics; eager + 3 graph replays; vs base and vs the problem reference."""
    out = {}
    G = state['nsm']
    for bs in SHAPES:
        for dist, seed in CORR_CASES:
            data = make_input(ref, torch, bs, dist, seed)
            rs = routing_stats(tolist(data['topk_ids']))
            arm0 = layer.sl1_arm(state, 0, bs)
            base = layer.kernel_fn(data, arm0).clone()
            torch.cuda.synchronize()
            snapshot(arm0, bs, 0, rs, G)
            again = layer.kernel_fn(data, arm0).clone()
            torch.cuda.synchronize()
            snapshot(arm0, bs, 0, rs, G)
            row = {'njS': rs['njS'], 'L': rs['L'], 'base_repeat_rel': float(layer._k2a_rel_err(again, base)[0])}
            for name, mask in ALL_ARMS.items():
                arm = get_arm(layer, state, name, bs)
                got = layer.kernel_fn(data, arm).clone()
                torch.cuda.synchronize()
                gen = snapshot(arm, bs, mask, rs, G)
                outs = [got]
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    layer.kernel_fn(data, arm)
                torch.cuda.synchronize()
                gen = snapshot(arm, bs, mask, rs, G)  # capture does not launch
                for rep in range(3):
                    graph.replay()
                    torch.cuda.synchronize()
                    g2 = snapshot(arm, bs, mask, rs, G)
                    assert g2 == gen + 1, ('generation', name, bs, dist, seed, rep, gen, g2)
                    gen = g2
                    outs.append(state['out'][bs].clone())
                rels = [float(layer._k2a_rel_err(o, base)[0]) if dist != 'none' else float((o.float() - base.float()).abs().max())
                        for o in outs]
                finite = all(bool(torch.isfinite(o.float()).all()) for o in outs)
                if dist == 'none':
                    ok_ref, msg = True, 'no local pair: exact zeros required'
                    assert all(bool((o == 0).all()) for o in outs) and bool((base == 0).all()), ('none not zero', name)
                else:
                    ok_ref, msg = ref.check_implementation(data, outs[-1], weights)
                    assert max(rels) <= GATES['rel_vs_base'], ('variant vs base', name, bs, dist, seed, rels)
                assert finite and ok_ref, ('reference / finite', name, bs, dist, seed, msg, finite)
                empty = [t for t, x in enumerate(rs['L']) if x == 0]
                assert all(bool((o[empty] == 0).all()) for o in outs), ('empty tokens not zero', name)
                row[name] = {'rel_vs_base': rels, 'ref_check': [bool(ok_ref), str(msg)[:120]], 'finite': finite}
                del graph
                print(f'[sl1] correct bs={bs} {dist}:{seed} arm={name} rel_vs_base_max={max(rels):.3e}', flush=True)
            out[f'{bs}:{dist}:{seed}'] = row
    return out


def capture_arms(layer, state, data, timer, rs, base_out):
    """Flag set/read BEFORE each graph; 3 consecutive checked replays of EVERY arm and twin."""
    bs = data['hidden_states'].shape[0]
    G = state['nsm']
    graphs, checks = {}, {}
    for name, mask in ARMS.items():
        for suffix in ('', '_twin'):
            arm = get_arm(layer, state, name, bs)
            timer.torch.cuda.synchronize()
            graph = timer.capture(lambda arm=arm: layer.kernel_fn(data, arm))
            previous = snapshot(arm, bs, mask, rs, G)
            reps = []
            for replay in range(3):
                with timer.torch.cuda.stream(timer.stream):
                    graph.replay()
                timer.torch.cuda.synchronize()
                g2 = snapshot(arm, bs, mask, rs, G)
                assert g2 == previous + 1, ('generation', name, suffix, replay, previous, g2)
                rel = float(layer._k2a_rel_err(state['out'][bs], base_out)[0]) if base_out is not None else 0.0
                assert rel <= GATES['rel_vs_base'], ('replay output', name, suffix, replay, rel)
                previous = g2
                reps.append({'generation': g2, 'rel_vs_base': rel})
            graphs[name + suffix] = graph
            checks[name + suffix] = reps
            print(f'[sl1] replay-safe bs={bs} arm={name}{suffix} flag={mask} x3', flush=True)
    return graphs, checks


def difference(row, positive, negative):
    ts = row['times']
    return stats.median(a - b for a, b in zip(ts[positive]['replays_raw_us'], ts[negative]['replays_raw_us']))


def bootstrap(values, draws=2000, seed=1234):
    rnd = random.Random(seed)
    if not values:
        return [None, None]
    meds = sorted(stats.median(rnd.choice(values) for _ in values) for _ in range(draws))
    return [meds[int(0.025 * (draws - 1))], meds[int(0.975 * (draws - 1))]]


def verdicts(result):
    rows = result.get('shapes', {})
    complete = all(str(bs) in rows and len(rows[str(bs)]) == 4 and all('times' in r for r in rows[str(bs)]) for bs in SHAPES)
    if not complete or result.get('correctness_status') != 'PASS':
        return {'status': 'VOID', 'reason': 'incomplete seed/shape/arm set or correctness not PASS'}
    ok = all(r['control_ok'] for rr in rows.values() for r in rr)
    per, med, ci, frac = {}, {}, {}, {}
    for bs in SHAPES:
        k = str(bs)
        per[k] = {arm: [difference(r, 'base', arm) for r in rows[k]] for arm in VARIANT_ARMS if arm != 'base'}
        per[k]['flag0_cost_vs_ship'] = [difference(r, 'base', 'ship') for r in rows[k]]
        per[k]['TM+PC_vs_ship'] = [difference(r, 'ship', 'TM+PC') for r in rows[k]]
        med[k] = {arm: stats.median(v) for arm, v in per[k].items()}
        ci[k] = {arm: bootstrap(v) for arm, v in per[k].items()}
        frac[k] = {arm: med[k][arm] / BOUNDS[arm][bs] for arm in ('TM', 'PC', 'TM+PC')}
        frac[k]['TM+PC_vs_AB1+COMBSKIP'] = med[k]['TM+PC'] / BOUNDS['TM+PC_alt'][bs]
    twins = max(abs(v) for rr in rows.values() for r in rr for v in r['twins_us'].values())
    passed = all(med[str(bs)]['TM+PC'] >= GATES['pass_combined_us'][bs] for bs in SHAPES)
    return {'status': 'VALID' if ok else 'VOID', 'reason': None if ok else 'an arm seed A/A twin > 1.0 us',
            'median_saving_us': med, 'per_seed_saving_us': per, 'bootstrap_ci95_us': ci,
            'capture_fraction_of_bound': frac, 'max_abs_twin_us': twins,
            'SL_rule': ('PASS' if passed else 'KILL') if ok else 'VOID',
            'rule_text': 'PASS iff TM+PC saves >= 1.2 us at bs1 AND >= 1.5 us at bs8, correct outputs, |A/A| <= 1.0 us'}


def save(result, out):
    result['verdicts'] = verdicts(result)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    v = result['verdicts']
    lines = ['# SL1 (TINY / TINY2) result', '', 'Correctness: ' + str(result.get('correctness_status')), '']
    if 'median_saving_us' in v:
        lines += ['| arm | bs1 net us [CI95] | bs8 net us [CI95] | bound bs1 / bs8 | capture bs1 / bs8 |', '|---|---|---|---|---|']
        for arm in ('TM', 'PC', 'TM+PC'):
            m, c, f = v['median_saving_us'], v['bootstrap_ci95_us'], v['capture_fraction_of_bound']
            lines.append('| %s | %.3f [%s] | %.3f [%s] | %s %.2f / %.2f | %.2f / %.2f |' % (
                arm, m['1'][arm], ', '.join('%.2f' % x for x in c['1'][arm]), m['8'][arm],
                ', '.join('%.2f' % x for x in c['8'][arm]), BOUNDS[arm]['src'], BOUNDS[arm][1], BOUNDS[arm][8],
                f['1'][arm], f['8'][arm]))
        m, c = v['median_saving_us'], v['bootstrap_ci95_us']
        for key, label in (('flag0_cost_vs_ship', 'flag-0 SL1 binary minus unpatched ship (us, + = slower)'),
                           ('TM+PC_vs_ship', 'TM+PC saving vs the unpatched ship (us)')):
            lines.append('\n%s: bs1 %.3f [%s], bs8 %.3f [%s]' % (label, m['1'][key], ', '.join('%.2f' % x for x in c['1'][key]),
                                                             m['8'][key], ', '.join('%.2f' % x for x in c['8'][key])))
        lines.append('\nmax |A/A twin| us: %.3f; status %s; SL rule: %s (%s)' % (v['max_abs_twin_us'], v['status'], v['SL_rule'], v['rule_text']))
    else:
        lines.append('Status: ' + v['status'] + ' (' + str(v.get('reason')) + ')')
    if result.get('error'):
        lines.append('\nError: ' + result['error'][-2000:])
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(out):
    import torch
    out.mkdir(parents=True, exist_ok=True)
    result = {'status': 'RUNNING', 'shapes': {}, 'arms': ARMS, 'diag_arms': DIAG_ARMS, 'gates': GATES, 'bounds': BOUNDS}
    save(result, out)
    try:
        with phase('SL1 total', 620):
            with phase('one prepare, flag-zero validation on the SL1 binaries', 295):
                sys.path.insert(0, str(ROOT / 'problems/muse_moe_w4a16'))
                ref = module(ROOT / 'problems/muse_moe_w4a16/reference.py', 'sl1_reference')
                layer = module(ROOT / 'kernels/muse_moe_w4a16/mb_layer_sl1.py', 'sl1_layer')
                weights = ref.generate_weights(ref.WEIGHTS_SEED)
                state = layer.prepare(weights)
                verify_routes(state)
                assert int(state['sl1_flag'].item()) == 0
                disable_pdl(state)
            with phase('correctness: 7 arms x 2 bs x 8 routings, eager + 3 replays each', 150):
                result['correctness'] = correctness(layer, state, ref, weights, torch)
                result['correctness_status'] = 'PASS'
                save(result, out)
            timer = Timer(torch)
            empty = timer.capture(lambda: None)
            overhead = stats.median(timer.replay(empty) for _ in range(30))
            result.update(overhead_us=overhead, gate_cycles=timer.gate_cyc, routes={str(bs): state['route'][bs] for bs in SHAPES})
            for bs in SHAPES:
                result['shapes'][str(bs)] = []
                for seed in range(4):
                    with phase(f'sl1 bs={bs} seed={seed}', 60):
                        data = ref.generate_input(bs, 'ep8', seed, dist='normal')
                        rs = routing_stats(tolist(data['topk_ids']))
                        arm0 = layer.sl1_arm(state, 0, bs)
                        base_out = layer.kernel_fn(data, arm0).clone()
                        torch.cuda.synchronize()
                        snapshot(arm0, bs, 0, rs, state['nsm'])
                        graphs, check = capture_arms(layer, state, data, timer, rs, base_out)
                        row = {'seed': seed, 'njS': rs['njS'], 'L': rs['L'], 'protocol': check}
                        result['shapes'][str(bs)].append(row)
                        row.update(measure_graphs(timer, graphs, overhead))
                        for name, mask in ARMS.items():
                            key = ('ship', bs) if name in CONTROL_ARMS else (mask, bs)
                            snapshot(state['sl1_arms'][key], bs, mask, rs, state['nsm'])
                        print('[sl1] paired ' + json.dumps({'bs': bs, 'seed': seed, 'njS': rs['njS'], 'twins_us': row['twins_us'],
                              'saving_us': {a: difference(row, 'base', a) for a in ARMS if a != 'base'}}), flush=True)
                        # ('ship' saving < 0 means the unpatched ship is SLOWER than the SL1 binary at flag 0)
                        del graphs
                        save(result, out)
            result['status'] = 'COMPLETE'
    except BaseException:
        import traceback
        result.update(status='FAIL', error=traceback.format_exc())
        raise
    finally:
        save(result, out)


# ---------------------------------------------------------------------------------------------------------------
def dry():
    """CPU only: generated-source gates + the REAL base state/dispatch functions + the generated SL1 block,
    driven with shape stubs (no torch, no CUDA). Every new prepare / arm / capture / snapshot / verdict path runs."""
    import hashlib
    import threading
    patch = module(ROOT / 'tools/sl1_patches.py', 'sl1_patch_cpu')
    base = patch.BASE.read_text(encoding='utf-8')
    generated = patch.DEST.read_text(encoding='utf-8')
    report = {'source_gates': patch.cpu_gates(base, generated)}
    functions = {n.name: n for n in ast.parse(base).body if isinstance(n, ast.FunctionDef)}
    def expression(node, env):
        return eval(compile(ast.Expression(node), '<real base expression>', 'eval'), env)
    env = constants(base)
    calls, builds = [], []

    class Tensor:
        sizes = {'uint8': 1, 'int16': 2, 'bfloat16': 2, 'int32': 4, 'float32': 4, 'int64': 8}
        def __init__(self, shape, dtype='float32', device='cpu', value=0):
            self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
            self.dtype, self.device, self.value = dtype, device, value
        def numel(self): return math.prod(self.shape)
        def contiguous(self): return self
        def fill_(self, v): self.value = v; return self
        def item(self): assert self.numel() == 1; return self.value
        def view(self, dtype):
            return Tensor(self.numel() * self.sizes[self.dtype] // self.sizes[dtype], dtype, self.device, self.value)
        def __getitem__(self, index):
            if isinstance(index, int):
                assert 0 <= index < self.shape[0]
                return Tensor(self.shape[1:], self.dtype, self.device, self.value)
            assert isinstance(index, slice)
            return Tensor((len(range(*index.indices(self.shape[0]))),) + self.shape[1:], self.dtype, self.device, self.value)

    class Torch:
        int32 = 'int32'; int16 = 'int16'; int64 = 'int64'; float32 = 'float32'; bfloat16 = 'bfloat16'; uint8 = 'uint8'
        class cuda:
            @staticmethod
            def get_device_properties(dev):
                class Properties: multi_processor_count = 132
                return Properties()
        @staticmethod
        def zeros(shape, dtype='float32', device='cpu'): return Tensor(shape, dtype, device)
        empty = zeros

    class Mod:
        def __init__(self, name): self.name = name; self.flag = 0; self.pdl = 1
        def okc_set_pdl(self, v): self.pdl = int(v)
        def sl1_set(self, v): self.flag = int(v)
        def okc_run(self, W, S, x, ids, tw, out, lo, nloc, nsm):
            assert len(W) == 14 and len(S) == 3 and lo == 0 and nloc == 32 and nsm == 132
            # host checks of the patched okc_run (HOST_CHECKS), mirrored
            assert not (self.flag & ~7) and (not (self.flag & 4) or (self.flag & 1)), 'invalid flag'
            assert not self.flag or not self.pdl, 'sl1 needs PDL off'
            assert S[0].shape == (8 + 1024 + EMAX,) and S[0].dtype == 'int32'
            calls.append((self.name, x.shape[0], self.flag, id(S[0])))
        def run(self, *args):
            raise AssertionError('VMS must not be routed at bs <= 8')

    env.update(torch=Torch, hashlib=hashlib, threading=threading, time=time)
    needed = ('_k2a_bucket_of', '_hyb_scratch', '_mbl_candidates', '_mbl_select', '_mbl_call', 'kernel_fn', '_mbl_deadline')
    for name in needed:
        exec(compile(ast.Module([functions[name]], type_ignores=[]), '<real base function>', 'exec'), env)
    mods_expr = next(n.value for n in functions['_mbl_builds'].body if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'mods' for t in n.targets))
    ship = {name: Mod(name) for name in env['MBL_ARMS']}
    env.update(_VMS={'mod': ship['vms'], 'mod_hi': ship['vms_hi']}, _HYB={'mod': ship['hyb'], 'mod7': ship['hyb7']},
               _HYBLO={'mods': {'tiny': ship['tiny'], 'tiny2': ship['tiny2']}}, _VML={'mods': {'bf2': ship['bf2']}})
    env['_mbl_builds'] = lambda deadline: expression(mods_expr, env)
    lock = threading.Lock()
    def build_stub(**kw):
        engine = next(e for e in ('tiny2', 'tiny') if kw['name'].startswith('sl1_' + e + '_'))
        src, cpp, flags = env['sl1_sources'](engine)
        assert kw['cuda_sources'] == src and kw['cpp_sources'] == cpp and kw['extra_cuda_cflags'] == flags
        assert kw['functions'] == ['okc_run', 'okc_set_pdl', 'sl1_set']
        assert 'int sl1;' in src and 'void sl1_set(int64_t v);' in cpp and '-gencode=arch=compute_90a,code=sm_90a' in flags
        with lock:
            builds.append(engine)
        return Mod('sl1_' + engine)
    env['_mbl_load_inline'] = build_stub
    exec(compile(generated[generated.index('# SL1_BEGIN'):], '<generated SL1 block>', 'exec'), env)
    mods = env['_mbl_builds'](time.monotonic() + 60)
    assert sorted(builds) == ['tiny', 'tiny2'] and mods['tiny'].name == 'sl1_tiny' and mods['tiny2'].name == 'sl1_tiny2'
    assert mods['tiny'].flag == 0 and mods['tiny'].pdl == int(env['HYBLO_PDL'])
    # Real prepare state dict (AST) + real scratch allocator shapes.
    st = {'hyblo': [Tensor(1) for _ in range(14)], 'hyb': [Tensor(1) for _ in range(14)]}
    banks = {0: {'k2a33': {'st': st}}}
    hsc, hfull = env['_hyb_scratch']('cpu')
    env.update(weights={}, t0=0, mods=mods, banks=banks, sc={'bf2': None}, hsc=hsc, hfull=hfull, nsm=132,
               output=Tensor((320, 4096), 'bfloat16'))
    state_expr = next(n.value for n in functions['prepare'].body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == 'state' for t in n.targets))
    state = expression(state_expr, env)
    state['good'] = {k: True for k in mods}
    env['_mbl_select'](state)
    verify_routes(state)
    env['_sl1_init'](state, 'cpu')
    assert state['sl1_flag'].item() == 0
    disable_pdl(state)
    seen = set()
    for bs in SHAPES:
        data = {'hidden_states': Tensor((bs, 4096), 'bfloat16'), 'topk_ids': Tensor((bs, 8), 'int32'),
                'topk_weights': Tensor((bs, 8)), 'config': {'local_start': 0}}
        for name, mask in ALL_ARMS.items():
            ship = name in CONTROL_ARMS
            arm = env['sl1_ship_arm'](state, bs) if ship else env['sl1_arm'](state, mask, bs)
            assert env['kernel_fn'](data, arm) is state['out'][bs]
            eng, m, flag, sid = calls[-1]
            want = EXPECTED_ROUTES[bs] if ship else 'sl1_' + EXPECTED_ROUTES[bs]
            assert (eng, m, flag) == (want, bs, mask), calls[-1]
            assert sid != id(state['h_sc'][0]) and (sid, mask) not in seen
            seen.add((sid, mask))
            assert (env['sl1_ship_arm'](state, bs) if ship else env['sl1_arm'](state, mask, bs)) is arm
            if ship:
                assert arm['mods']['tiny'].name == 'tiny' and state['mods']['tiny'].name == 'sl1_tiny'
                assert arm['mods']['tiny'].pdl == 0
    assert len({s for s, _ in seen}) == len(SHAPES) * len(ALL_ARMS), 'scratch shared between arms'
    for mask, bs in ((4, 1), (6, 8), (8, 1), (-1, 1), (1, 9), (1, 0)):
        try:
            env['sl1_arm'](state, mask, bs)
        except ValueError:
            pass
        else:
            raise AssertionError(('invalid arm accepted', mask, bs))
    # capture path with fake graphs + snapshot stub (snapshot itself is exercised below on modelled scratch).
    class Cuda:
        @staticmethod
        def stream(s): return contextlib.nullcontext()
        @staticmethod
        def synchronize(): pass
    class TimerStub:
        class torch: cuda = Cuda
        stream = None
        def capture(self, fn):
            fn()
            class Graph:
                def replay(self): pass
            return Graph()
    class Layer:
        sl1_arm = staticmethod(env['sl1_arm'])
        sl1_ship_arm = staticmethod(env['sl1_ship_arm'])
        kernel_fn = staticmethod(env['kernel_fn'])
    counter = {}
    def snap_stub(arm, bs, mask, rs, G):
        k = (bs, mask, id(arm)); counter[k] = counter.get(k, 0) + 1
        return counter[k]
    g_snap = globals()['snapshot']
    globals()['snapshot'] = snap_stub
    try:
        for bs in SHAPES:
            data = {'hidden_states': Tensor((bs, 4096), 'bfloat16'), 'topk_ids': Tensor((bs, 8), 'int32'),
                    'topk_weights': Tensor((bs, 8)), 'config': {'local_start': 0}}
            graphs, checks = capture_arms(Layer, state, data, TimerStub(), {'njS': 1, 'L': [1] * bs}, None)
            assert len(graphs) == 2 * len(ARMS) and all(len(v) == 3 for v in checks.values())
    finally:
        globals()['snapshot'] = g_snap
    report['snapshot_model'] = snapshot_model()
    # verdict logic on both sides of every gate
    fake = {'correctness_status': 'PASS', 'shapes': {}}
    for bs in SHAPES:
        fake['shapes'][str(bs)] = []
        for seed in range(4):
            times = {a: {'replays_raw_us': [30.0 - (0 if a.startswith('base') else 2.0)] * 30} for a in list(ARMS) + [a + '_twin' for a in ARMS]}
            fake['shapes'][str(bs)].append({'times': times, 'control_ok': True, 'twins_us': {a: 0.0 for a in ARMS}})
    v = verdicts(fake)
    assert v['status'] == 'VALID' and v['SL_rule'] == 'PASS', v
    fake['shapes']['8'][1]['times']['TM+PC']['replays_raw_us'] = [29.0] * 30
    fake['shapes']['8'][2]['times']['TM+PC']['replays_raw_us'] = [29.0] * 30
    fake['shapes']['8'][3]['times']['TM+PC']['replays_raw_us'] = [29.0] * 30
    assert verdicts(fake)['SL_rule'] == 'KILL'
    fake['shapes']['1'][0]['control_ok'] = False
    assert verdicts(fake)['status'] == 'VOID'
    fake['correctness_status'] = 'FAIL'
    assert verdicts(fake)['status'] == 'VOID'
    report.update(status='PASS', builds=sorted(builds), arms_dispatched=len(seen), real_base_state_ast=True,
                  invalid_arms_rejected=True, capture_paths=2 * len(ARMS) * len(SHAPES),
                  limitations='CPU stubs prove dispatch/scratch/flag plumbing and the protocol models, not CUDA correctness or liveness')
    print(json.dumps(report, indent=2))
    return report


def constants(source):
    """Literal and constant-expression globals of the base module (bt4b_patches.environment rule), no torch."""
    env = {}
    safe = (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv,
            ast.Div, ast.Mod, ast.LShift, ast.RShift, ast.USub, ast.UAdd, ast.Tuple, ast.List)
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            if not all(isinstance(n, safe) for n in ast.walk(node.value)):
                continue
            try:
                value = eval(compile(ast.Expression(node.value), '<constant>', 'eval'), {'__builtins__': {}}, env)
            except (NameError, TypeError):
                continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                env[t.id] = value
            elif isinstance(t, ast.Tuple) and all(isinstance(x, ast.Name) for x in t.elts):
                env.update((x.id, v) for x, v in zip(t.elts, value))
    return env


def snapshot_model():
    """Drive snapshot_lists() with scratch images produced by the P5/P6 protocol model (and corrupted ones)."""
    G, M = 132, 8
    rs = {'njS': 3, 'L': [1, 0, 2, 1, 0, 0, 3, 1]}
    def image(mask, launches):
        ints = [0] * (8 + 1024 + EMAX + 64)
        ss0 = [0.0] * EMAX
        for g in range(launches):
            if mask & 2:
                ob, ub = (g + 1) & 1, g & 1
                for i in range(JMAX):
                    ints[8 + ob * JMAX + i] = 0
                for i in range(MMAX):
                    ints[8 + 2 * JMAX + ob * MMAX + i] = 0
                for j in range(rs['njS']):
                    ints[8 + ub * JMAX + j] = RB_A
                for t in range(M):
                    ints[8 + 2 * JMAX + ub * MMAX + t] = RB_B * rs['L'][t]
                ints[6] += G
            else:
                ints[4 + ((g + 1) & 1)] = 0
                ints[4 + (g & 1)] = G
                ints[3] += 1
            # ss: the used parity gets garbage, the other parity is zeroed (ship)
            if g & 1:
                for i in range(EMAX):
                    ints[8 + 1024 + i] = 1065353216  # 1.0f bits
                ss0 = [0.0] * EMAX
            else:
                ss0 = [1.0] * EMAX
                for i in range(EMAX):
                    ints[8 + 1024 + i] = 0
        return ints, ss0
    checked = 0
    for mask in (0, 1, 2, 3, 5, 7):
        for launches in range(0, 5):
            ints, ss0 = image(mask, launches)
            gen = snapshot_lists(ints, ss0, G, mask, rs, M)
            assert gen == launches
            checked += 1
            if launches:
                bad = list(ints)
                if mask & 2:
                    bad[8 + 2 * JMAX + ((launches - 1) & 1) * MMAX + 6] -= 1   # a lost unit of token 6
                else:
                    bad[4 + ((launches - 1) & 1)] -= 1                            # a lost barrier arrival
                try:
                    snapshot_lists(bad, ss0, G, mask, rs, M)
                except AssertionError:
                    checked += 1
                else:
                    raise AssertionError(('corruption accepted', mask, launches))
    return f'PASS: {checked} modelled scratch images (valid accepted, corrupted rejected)'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mode', choices=('dry', 'run'))
    p.add_argument('--out', type=Path, default=ROOT / 'box_out/sl1')
    a = p.parse_args()
    if a.mode == 'dry':
        dry()
    else:
        run(a.out)


if __name__ == '__main__':
    main()

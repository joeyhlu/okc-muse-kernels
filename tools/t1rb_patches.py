"""T1RB: three default-off latency trims on the exact TINYS7 ship file.

Generate only kernels/lab/k3_t1rb_ab.py. --check is read-only (including
py_compile, whose bytecode sink is in memory). --linux-gate is compile-only;
it never loads a CUDA extension or creates a CUDA context.

Kill reasons, before performance claims:
  b1: landing KV before the wait can delay Q; conversion may already overlap.
  b2: random-gather throughput may hide Q; Q's longer live range may spill.
  b4: selection already hides lstart; its longer live range may cost registers.
No weight-stream bytes change. This is an explicitly requested rider build.

Source accounting (EST; final SASS/resources are recorded by the Linux gate):
  b1, each old-key CTA: 128 keys * 64 channels * 2 (K,V) = 16,384 B
  global input, 32,768 B FP16 shared stores, unchanged. Per thread: 2 chunks
  * (16 fp8x2 conversions + 4 STS.128) = 32 conversions + 8 stores, unchanged.
  b2, each tile: 64 * 128 * 2 = 16,384 B K/V and 64 * 4 = 256 B indices,
  unchanged. Each of 128 threads issues 8 cp.async.16; tid 0..63 additionally
  issues 1 cp.async.4 per valid index tile. ST=2 boot indices 512 -> 768 B
  (plus the same 4 B seqlen); boot KV stays 16,384 B. Tile 1's 16,384 B still
  issues AFTER wait. A three-tile CTA keeps 49,152 B KV + 768 B indices;
  commits/thread 6 -> 7, waits unchanged (new wait immediates), two extra
  source predicates i<ST/i>0 per iteration; barriers and Q LDG.128 unchanged.
  b4: no loop changes, no weight-byte change. Surviving routing threads
  512..543 keep one 4-B lstart load each; source hoisting exposes that load to
  512..767 (128 -> at most 1,024 lane-request B, SAME 4-B address). The other
  warps' dead loads can be eliminated; compiled instruction count is pending.

Post-wait protocol/cleanup (unchanged text, checked below):
  KS1: every CTA's threads 0..255 reach B1 before segment, the post-segment
  barrier, ow stores (all threads), mw/lw stores (lane%4==0), and eight-warp
  combine. All 17 CTAs/group publish red.add then ticket; non-last CTAs return.
  The last CTA's full 0..255 cover ROWS*64 O values: DPT=2/4, UPR=32/16,
  mr=tid/UPR, mu=tid%UPR; mu=0 resets each denominator, tid=0 resets counter.
  Groups=(head,tg): 2 at d1s, 8 at v1s; each independently keeps a last CTA.
  AX1: all CTAs' threads 0..127 keep wait_group/barrier before shared reuse,
  kh=warp/2 folds the two halves, red4/red1, and ticket. Only non-last exits.
  Last CTA 0..127 covers 16 rows * 8 parts * 8 floats, for each copy 0..nc-1;
  part=0 resets each denominator, tid=0 resets cnt[0]. Owner threads 0..15
  patch the one 256-B new K/V row in any boot slot before the loop barrier.
  Tail: each token CTA keeps routing threads 512..767 through bar 2; 544..767
  arrive then exit, 512..543 do the 19-step network and 8 picks, 512..519 store
  tw/tid/lid. Norm threads 0..511 keep bar 1, 16-warp sum and all xn stores.
  No CTA/thread exits, combine/reset loops, counters or barriers are removed.
"""
import argparse
import ast
import builtins
import collections
import hashlib
import json
import pathlib
import re
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "kernels/lab/k3_t1rb_ab.py"
CHECK = ROOT / "tools/box/t1rb_check.py"

# Kept in the generated file: no import of this generator at prepare/call time.
RUNTIME = r'''
# T1RB BEGIN. Ship strings/cache keys/scratch/TINYS remain untouched.
T1RB = 0
_T1RB_BANK = {}
_T1RB_VSL = (('ks1',1,16,1,1), ('ks1',8,16,4,1))

def _t1rb_once(s,a,b):
 if s.count(a)!=1:
  raise RuntimeError(('T1RB source anchor',a[:100],s.count(a)))
 return s.replace(a,b)

def _t1rb_region(s,start,end,edit):
 i=s.index(start)
 j=s.index(end,i)
 return s[:i]+edit(s[i:j])+s[j:]

def _t1rb_ks1(s):
 a='#pragma unroll\nfor (int kk = 0; kk < (oldc ? G::KPT : 0); ++kk) {'
 i=s.index(a)
 j=s.index('#pragma unroll\nfor (int j = 0; j < 2; ++j)',i)
 loop=s[i:j]
 s=_t1rb_once(s,loop,'')
 return _t1rb_once(s,'KS1P(1);\npdl_wait();',loop+'KS1P(1);\npdl_wait();')

def _t1rb_ax1(s):
 def edit(x):
  changes=(
   ('y < 2 * ST - 2; ++y) issue_idx(y);','y < 2 * ST - 1; ++y) issue_idx(y);'),
   ('cpw<1>();\n__syncthreads();\nAX1P(1);','cpw<0>();\n__syncthreads();\nAX1P(1);'),
   ('pdl_wait();\nif (p.trig == 1) pdl_go();',
    'pdl_wait();\nif (p.trig == 1) pdl_go();\n'
    'const uint4 qv = __ldcg(reinterpret_cast<const uint4*>(p.qkv + (tid >> 3) * 64 + (tid & 7) * 8));\n'
    'if (ST - 1 < ntiles) issue_kv(ST - 1, false);\ncpc();'),
   ('const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p.qkv + r * 64 + c8));','const uint4 v = qv;'),
   ('if (xn < ST - 1 && tid < 16)', 'if (xn < ST && tid < 16)'),
   ('cpw<ST - 2>();\n__syncthreads();\nif (i + ST - 1 < ntiles) issue_kv(i + ST - 1, true);\nissue_idx(i + 2 * ST - 2);',
    'if (i < ST) cpw<ST - 1>(); else cpw<ST - 2>();\n__syncthreads();\n'
    'if (i > 0 && i + ST - 1 < ntiles) issue_kv(i + ST - 1, true);\nissue_idx(i + 2 * ST - 1);'))
  for a,b in changes:
   x=_t1rb_once(x,a,b)
  return x
 return _t1rb_region(s,'template <int ST>\n__global__ void __launch_bounds__(NTHR, 2) fb1_l1(',
                     'constexpr int TB = TILE * ROWB;',edit)

def _t1rb_tail(s):
 def edit(x):
  x=_t1rb_once(x,'const int ls = __ldcg(lstart);\n','')
  return _t1rb_once(x,'ft_ld<EPT, false>(bias + e0, bs);\n',
                      'ft_ld<EPT, false>(bias + e0, bs);\nconst int ls = __ldcg(lstart);\n')
 return _t1rb_region(s,'__global__ void __launch_bounds__(NT + 32 * RW) tail_kernel3(',
                     'template <int MT>\n__global__',edit)

def _t1rb_jobs(changed=True):
 # KS1 builds only the two selected VSL specializations. This module never
 # handles KS3 or DG1, so their folded translation units need not be duplicated.
 ks=KS1_CUDA_SRC.replace('@INSTS@','').replace('@MINSTS@',
     ' '.join(f'KS1_XM({l}, {n}, {g}, {m})' for _,l,n,g,m in _T1RB_VSL)).replace('@M3INSTS@','')
 ax=AX1_CUDA_SRC
 tail=V8_TAIL_CUDA_SRC
 if changed:
  ks,ax,tail=_t1rb_ks1(ks),_t1rb_ax1(ax),_t1rb_tail(tail)
 fl=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-std=c++20']
 return {1:(KS1_CPP_SRC,ks,list(fl),['ks1_forward','ks1_info','ks3_forward','ks3_info']),
         2:(AX1_CPP_SRC,ax,list(fl),['ax1_forward','ax1_info']),
         4:(V8_TAIL_CPP_SRC,tail,list(fl)+['--expt-relaxed-constexpr'],['v8_tail','v8_router32'])}

def _t1rb_load(bit,job):
 from torch.utils.cpp_extension import load_inline
 cpp,cu,flags,fns=job
 tag=hashlib.sha1((cpp+cu+' '.join(flags)).encode()).hexdigest()[:12]
 return load_inline(name=f't1rb_b{bit}_{tag}',cpp_sources=cpp,cuda_sources=cu,
                    functions=fns,extra_cflags=['-O3','-std=c++20'],extra_cuda_cflags=flags,verbose=False)

def _t1rb_prepare(state):
 # Exact handles: _vsl_call/_ax1_call/_v8_tail. No invented MoE accessors.
 if _KS1['mod'] is None or _AX1['mod'] is None or state.get('v8') is None:
  raise RuntimeError('T1RB: ship attention/tail prepare unavailable')
 if state['v8']['mod'] is None:
  raise RuntimeError('T1RB: ship tail module unavailable')
 jobs=_t1rb_jobs()
 for bit,job in jobs.items():
  if bit not in _T1RB_BANK:
   _T1RB_BANK[bit]=_t1rb_load(bit,job)

def _t1rb_select(bit,ship,eligible):
 if type(T1RB) is not int or T1RB<0 or T1RB & ~7:
  raise ValueError(('T1RB',T1RB))
 if not eligible or not (T1RB & bit):
  return ship
 if bit not in _T1RB_BANK:
  raise RuntimeError(('T1RB module was not built in prepare',bit))
 return _T1RB_BANK[bit]
# T1RB END
'''


def once(s, a, b):
    assert s.count(a) == 1, (a[:100], s.count(a))
    return s.replace(a, b)


def function_text(s, name):
    node = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(s, node)


def patch_function(s, name, a, b):
    old = function_text(s, name)
    return once(s, old, once(old, a, b))


EDITS = (
    ('prepare', ' _tinys_prepare()\n', ' _tinys_prepare()\n _t1rb_prepare(state)\n'),
    ('_vsl_call', '_KS1["mod"].ks1_forward(',
     '_t1rb_select(1,_KS1["mod"],v in _T1RB_VSL).ks1_forward('),
    ('_ax1_call', "_AX1['mod'].ax1_forward(",
     "_t1rb_select(2,_AX1['mod'],L==1 and qkv.shape[0]<=8).ax1_forward("),
    ('_v8_tail', 'v8["mod"].v8_tail(',
     '_t1rb_select(4,v8["mod"],tokens<=8).v8_tail('),
)


def generate(base):
    s = base
    for name, a, b in EDITS:
        s = patch_function(s, name, a, b)
    return s + RUNTIME


def namespace(source):
    """Read real AST constants/functions; never import the GPU layer."""
    ns = {'hashlib': hashlib, '__builtins__': dict(vars(builtins))}
    functions = {'_vsl_call', '_ax1_call', '_ax1_plan', '_v8_tail', '_ft_flag'}
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign):
            try:
                val = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                # Pure constant arithmetic only; no calls, attributes or lookups.
                if not all(isinstance(x, (ast.Expression, ast.Constant, ast.Name, ast.Load,
                                         ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop,
                                         ast.Tuple, ast.List, ast.Dict)) for x in ast.walk(n.value)):
                    continue
                try:
                    val = eval(compile(ast.Expression(n.value), '<constant>', 'eval'), ns)
                except (NameError, TypeError):
                    continue
            for t in n.targets:
                if isinstance(t, ast.Name):
                    ns[t.id] = val
        elif isinstance(n, ast.FunctionDef) and (n.name.startswith('_t1rb_') or n.name in functions):
            exec(compile(ast.Module([n], []), '<real layer function>', 'exec'), ns)
    return ns


class Tensor:
    def __init__(self, shape=(1,), dtype='bf16'):
        self.shape, self.dtype, self.device = tuple(shape), dtype, 'cpu-mock'

    def dim(self): return len(self.shape)
    def stride(self, axis):
        import math
        return math.prod(self.shape[axis + 1:])
    def view(self, *args): return self


class Module:
    def __init__(self, name): self.name, self.calls = name, []
    def ks1_forward(self, *args): self.calls.append(('ks1_forward', args))
    def ks3_forward(self, *args): self.calls.append(('ks3_forward', args))
    def ax1_forward(self, *args): self.calls.append(('ax1_forward', args))
    def v8_tail(self, *args): self.calls.append(('v8_tail', args))


def record_keys(source, function, target):
    node = ast.parse(function_text(source, function))
    a = next(n for n in ast.walk(node) if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Dict) and any(ast.unparse(t) == target for t in n.targets))
    return [ast.literal_eval(k) for k in a.value.keys]


def dry_run(base, generated):
    ns = namespace(generated)
    ship = {b: Module('ship' + str(b)) for b in (1, 2, 4)}
    ns['_KS1']['mod'], ns['_AX1']['mod'] = ship[1], ship[2]
    # Schema is taken from the base's actual prepare dictionary, not invented.
    v8 = {k: Tensor() for k in record_keys(base, '_v8_tail_prepare', 'v8')}
    v8['mod'] = ship[4]
    state = {'v8': v8}
    built = []

    def load_inline(**kw):
        bit = int(kw['name'].split('_')[1][1:])
        assert kw['cpp_sources'] and kw['cuda_sources'] and kw['functions']
        assert '-std=c++20' in kw['extra_cflags'] and '-std=c++20' in kw['extra_cuda_cflags']
        built.append(bit)
        return Module('b' + str(bit))

    def guarded_import(name, *args, **kw):
        if name == 'torch.utils.cpp_extension':
            return types.SimpleNamespace(load_inline=load_inline)
        return builtins.__import__(name, *args, **kw)

    ns['__builtins__']['__import__'] = guarded_import
    ns['torch'] = types.SimpleNamespace(
        bfloat16='bf16', float32='f32', int32='i32', int64='i64', uint8='u8',
        empty=lambda shape, **kw: Tensor((shape,) if isinstance(shape, int) else shape, kw['dtype']))
    ns['_vsl_ws'] = lambda dev: (Tensor(), Tensor())
    ns['_ks1_counters'] = lambda dev: Tensor()
    ns['_ax1_ws'] = lambda dev: (Tensor(), Tensor(), Tensor())
    ns['FT_OK'] = True
    # Execute the exact added prepare call after the base's _tinys_prepare call.
    prep = ast.parse(function_text(generated, 'prepare')).body[0]
    stmt = next(n for n in prep.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id == '_t1rb_prepare')
    exec(compile(ast.Module([stmt], []), '<real prepare insertion>', 'exec'), dict(ns, state=state))
    ns['_t1rb_prepare'](state)  # repeated prepare must reuse all three modules
    assert built == [1, 2, 4], built
    old_handles = ns['_KS1']['mod'], ns['_AX1']['mod'], state['v8']['mod']
    checked = 0
    for bits in range(8):
        ns['T1RB'] = bits
        for T in (1, 8, 16, 32, 128, 256):
            cfg = {'batch': 1, 'q_len': T, 'sm_scale': 0.125, 'qk_scale_factor': 1.0}
            data = {k: Tensor() for k in ('cache_seqlens','cache_loc','positions','k_cache','v_cache','local_start')}
            data.update(config=cfg, page_table=Tensor((1, 80000)))
            qkv = Tensor((T, 2304))
            if T in (1, 8):
                v = ns['_T1RB_VSL'][0 if T == 1 else 1]
                out = ns['_vsl_call'](qkv, data, Tensor(), v)
                md = ns['_T1RB_BANK'][1] if bits & 1 else ship[1]
                assert md.calls[-1][0] == 'ks1_forward' and md.calls[-1][1][11] is out
                out = ns['_ax1_call'](qkv, data, cfg)
                md = ns['_T1RB_BANK'][2] if bits & 2 and T == 1 else ship[2]
                assert md.calls[-1][0] == 'ax1_forward' and md.calls[-1][1][9] is out
                if T == 1:
                    assert md.calls[-1][1][10:14] == (1, 192, 417, 8)
            for logits in (None, Tensor((T, 256)), Tensor((8, T, 256))):
                outs = ns['_v8_tail'](Tensor((T, 4608)), logits, state, data)
                md = ns['_T1RB_BANK'][4] if bits & 4 and T <= 8 else ship[4]
                assert md.calls[-1][1][10:14] == outs
                assert md.calls[-1][1][-1] == 11
                checked += 1
        # Non-target KS3 must still use the original handle.
        cfg = {'batch': 16, 'q_len': 1, 'sm_scale': 0.125, 'qk_scale_factor': 1.0, 'rope_theta': 10000.0}
        data['config'] = cfg
        ns['_vsl_call'](Tensor((16, 2304)), data, Tensor(), ('ks3', 16, 1))
        assert ship[1].calls[-1][0] == 'ks3_forward'
        assert old_handles == (ns['_KS1']['mod'], ns['_AX1']['mod'], state['v8']['mod'])
    for bad in (-1, 8, 1.0, True, '1'):
        ns['T1RB'] = bad
        try:
            ns['_t1rb_select'](1, ship[1], True)
        except ValueError:
            pass
        else:
            raise AssertionError(('invalid selector accepted', bad))
    ns['T1RB'] = 1
    saved = ns['_T1RB_BANK'].pop(1)
    try:
        ns['_t1rb_select'](1, ship[1], True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('unbuilt module silently fell back')
    ns['_T1RB_BANK'][1] = saved
    for st in ({}, {'v8': None}, {'v8': {'mod': None}}):
        try:
            ns['_t1rb_prepare'](st)
        except RuntimeError:
            pass
        else:
            raise AssertionError('missing tail handle accepted')
    for name in ('_KS1', '_AX1'):
        original = ns[name]['mod']
        ns[name]['mod'] = None
        try:
            ns['_t1rb_prepare'](state)
        except RuntimeError:
            pass
        else:
            raise AssertionError(('missing attention handle accepted', name))
        finally:
            ns[name]['mod'] = original
    return {'built': built, 'selector_values': list(range(8)), 'tail_dispatches': checked,
            'real_state_keys': list(v8), 'handles_unchanged': True}


def cp_async_model():
    """Worst-case legal wait_group completion; model both index and KV rings.

    Empty commits count as groups. Include 0/1/2 tiles, final partial tiles,
    owner in either preissued tile, and enough turns to wrap both rings.
    No barrier, atomic, early exit or cleanup is added/removed by Fix 1.
    """
    cases = 0
    for st in (2, 3, 4):
        for tiles in range(0, 18):
            groups, idx, kv, consumed, last_read = [], {}, {}, set(), {}

            def commit(ops): groups.append(list(ops))

            def wait(n):
                while len(groups) > n:
                    for kind, tile in groups.pop(0):
                        ring = idx if kind == 'idx' else kv
                        slot = tile % (2 * st if kind == 'idx' else st)
                        old = ring.get(slot)
                        if kind == 'kv' and old is not None:
                            assert old in consumed, ('overwrite live KV', st, tiles, old)
                        if kind == 'idx' and old is not None:
                            assert old in last_read, ('overwrite unread index', st, tiles, old)
                        ring[slot] = tile

            def idx_op(tile): return [('idx', tile)] if tile < tiles else []

            def kv_op(tile):
                if tile >= tiles: return []
                assert idx.get(tile % (2 * st)) == tile, ('idx not landed', st, tiles, tile)
                last_read[tile] = True
                return [('kv', tile)]

            commit([op for y in range(st - 1) for op in idx_op(y)])
            commit([op for y in range(st - 1, 2 * st - 1) for op in idx_op(y)])
            wait(0)
            for x in range(st - 1): commit(kv_op(x))
            commit(kv_op(st - 1))  # after PDL, concurrent with Q
            for i in range(tiles):
                wait(st - 1 if i < st else st - 2)
                assert kv.get(i % st) == i, ('KV not landed', st, tiles, i)
                ops = kv_op(i + st - 1) if i > 0 else []
                commit(ops + idx_op(i + 2 * st - 1))
                consumed.add(i)
            wait(0)  # unchanged drain before smem is reused by merge
            assert consumed == set(range(tiles))
            cases += 1
    # Exhaust the owner row's initial slots and all 64 row offsets. All global
    # cp.async writers MUST skip exactly the new row when with_new=false.
    # The extended ST (not ST-1) patch is necessary for tile 1 at ST=2.
    for st in (2, 3, 4):
        for seq in range(64 * (st + 2)):
            owner_tile, owner_row = divmod(seq, 64)
            writers = collections.Counter()
            for tile in range(st):
                for tid in range(128):
                    for k in range(4):
                        row = (tid >> 3) + 16 * k
                        pos = tile * 64 + row
                        if pos != seq:  # old rows load, padding zero-fills
                            for kind in ('K', 'V'):
                                writers[kind, tile, row, tid & 7] += 1
            if owner_tile < st:
                for tid in range(16):
                    writers['K' if tid < 8 else 'V', owner_tile, owner_row, tid & 7] += 1
            assert len(writers) == st * 64 * 8 * 2 and set(writers.values()) == {1}
    return cases


def py_compile_memory(paths):
    import importlib._bootstrap_external as be
    import py_compile
    from unittest.mock import patch
    written = []
    with patch.object(be, '_write_atomic', side_effect=lambda path, data, mode=0o666: written.append(len(data))):
        for p in paths:
            # Existing parent; py_compile never makes a __pycache__ directory.
            py_compile.compile(str(p), cfile=str(OUT), doraise=True)
    assert len(written) == len(paths) and all(written)
    return len(written)


def cpu_gates(base, generated):
    assert generated.endswith(RUNTIME)
    restored = generated[:-len(RUNTIME)]
    for name, a, b in reversed(EDITS):
        restored = patch_function(restored, name, b, a)
    assert restored == base, 'outside permitted selectors/prepare insertion changed'
    bns, ns = namespace(base), namespace(generated)
    for name in bns:
        if isinstance(bns[name], str) and ('CUDA' in name or 'CPP' in name):
            assert ns[name] == bns[name], name
    jobs, controls = ns['_t1rb_jobs'](), ns['_t1rb_jobs'](False)
    assert set(jobs) == {1, 2, 4} and ns['T1RB'] == 0
    for bit in jobs:
        assert jobs[bit][1] != controls[bit][1]
    ax, oldax = jobs[2][1], controls[2][1]
    # Entire merge/counter/reset tail is byte-for-byte unchanged.
    assert ax[ax.index('AX1P(3);'):] == oldax[oldax.index('AX1P(3);'):]
    ks, oldks = jobs[1][1], controls[1][1]
    assert ks[ks.index('KS1P(3);'):] == oldks[oldks.index('KS1P(3);'):]
    tail, oldtail = jobs[4][1], controls[4][1]
    assert tail[tail.index('const uint4 k0 ='): ] == oldtail[oldtail.index('const uint4 k0 ='):]
    assert ax.count('__syncthreads();') == oldax.count('__syncthreads();')
    assert ax.count('return;') == oldax.count('return;')
    # The pod file's CPU path must also run as part of the single Linux gate.
    check_ns = {'__name__': 't1rb_checker_cpu', '__file__': str(CHECK)}
    exec(compile(CHECK.read_text(encoding='utf-8'), str(CHECK), 'exec'), check_ns)
    pod_cpu = check_ns['cpu_check']()
    return {'default_identity': True, 'original_cuda_literals_identical': True,
            'tinys_unchanged': ns['_TINYS_CUDA'] == bns['_TINYS_CUDA'],
            'dry_run': dry_run(base, generated), 'cp_async_cases': cp_async_model(),
            'pod_cpu': pod_cpu}


def resource_records(log):
    records = {}
    # Each Function properties block has a spill report even for zero spills.
    for m in re.finditer(r'Function properties for (\S+)\n(.*?)(?=ptxas info\s*: Compiling entry function|\Z)', log, re.S):
        sp = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', m[2])
        regs = re.search(r'Used (\d+) registers', m[2])
        if sp:
            records[m[1]] = {'spill_stores': int(sp[1]), 'spill_loads': int(sp[2]),
                             'registers': int(regs[1]) if regs else None}
    return records


def linux_gate(generated, result_path, result):
    import os
    import shutil
    import subprocess
    import sysconfig
    import tempfile
    result.update(cuda_status='FAIL', builds={})
    try:
        assert sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        import torch
        import torch.utils.cpp_extension as ce
        nvcc, cobj, host = shutil.which('nvcc'), shutil.which('cuobjdump'), shutil.which('c++')
        assert nvcc and cobj and host, 'nvcc/cuobjdump/c++ required'
        inc = sum((['-I', p] for p in ce.include_paths(device_type='cuda')), [])
        inc += ['-I', sysconfig.get_paths()['include']]
        out = pathlib.Path(tempfile.mkdtemp(prefix='t1rb_gate_'))  # retained; never delete artifacts
        result['artifacts'] = str(out)
        abi = str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
        common = ['-DTORCH_EXTENSION_NAME=t1rb_gate', '-DTORCH_API_INCLUDE_EXTENSION_H',
                  '-D_GLIBCXX_USE_CXX11_ABI=' + abi] + inc
        nvbase = [nvcc, '-c', '-std=c++20', '--expt-relaxed-constexpr', '-Xcompiler', '-fPIC',
                  '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
                  '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__'] + common
        ns = namespace(generated)
        for changed in (False, True):
            for bit, (cpp, cu, flags, fns) in ns['_t1rb_jobs'](changed).items():
                key = ('b' if changed else 'ship') + str(bit)
                hp, cp, ob = out / (key + '.cpp'), out / (key + '.cu'), out / (key + '.o')
                binding = '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n' + ''.join(
                    f'm.def("{f}", torch::wrap_pybind_function({f}));\n' for f in fns) + '}\n'
                hp.write_bytes(('#include <torch/extension.h>\n' + cpp + binding).encode())
                cp.write_bytes(('#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n' + cu).encode())
                hc = [host, '-O3', '-std=c++20', '-fPIC', '-fsyntax-only'] + common + [str(hp)]
                hr = subprocess.run(hc, capture_output=True, text=True)
                cmd = nvbase + flags + ['-Xptxas=-v', str(cp), '-o', str(ob)]
                cr = subprocess.run(cmd, capture_output=True, text=True)
                log = cr.stdout + cr.stderr
                row = {'host_command': hc, 'host_exit': hr.returncode, 'host_log': hr.stdout + hr.stderr,
                       'nvcc_command': cmd, 'nvcc_exit': cr.returncode, 'ptxas_log': log,
                       'resources': resource_records(log)}
                if cr.returncode == 0:
                    sass = subprocess.run([cobj, '-sass', str(ob)], capture_output=True, text=True)
                    res = subprocess.run([cobj, '-res-usage', str(ob)], capture_output=True, text=True)
                    row.update(sass_exit=sass.returncode, resources_exit=res.returncode,
                               sass=sass.stdout, resource_usage=res.stdout,
                               instruction_counts=dict(collections.Counter(re.findall(
                                   r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)', sass.stdout))))
                result['builds'][key] = row
                result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
                print(key, 'host', hr.returncode, 'nvcc', cr.returncode, flush=True)
        for bit in (1, 2, 4):
            old, new = result['builds']['ship' + str(bit)], result['builds']['b' + str(bit)]
            for row in (old, new):
                assert row['host_exit'] == row['nvcc_exit'] == row.get('sass_exit') == row.get('resources_exit') == 0
                assert row['resources'] and row['instruction_counts'], 'empty ptxas/SASS output'
            assert old['resources'].keys() == new['resources'].keys()
            target = {1: 'ks1_kernel', 2: 'fb1_l1', 4: 'tail_kernel3'}[bit]
            assert any(target in symbol for symbol in new['resources']), ('missing instantiation', bit)
            for symbol, r in new['resources'].items():
                for field in ('spill_stores', 'spill_loads'):
                    assert r[field] <= old['resources'][symbol][field], (bit, symbol, field, r)
            # Spill opcodes are included independently, catching parser mistakes.
            for op in ('LDL', 'STL'):
                def count(row): return sum(v for k, v in row['instruction_counts'].items() if k.split('.')[0] == op)
                assert count(new) <= count(old), (bit, op, 'new local instructions')
        result.update(cuda_status='PASS', no_new_spills=True)
    except Exception as ex:
        result['error'] = repr(ex)
    finally:
        result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('T1RB Linux gate:', result['cuda_status'], result_path, flush=True)
    if result['cuda_status'] != 'PASS':
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--linux-gate', type=pathlib.Path)
    a = p.parse_args()
    base = BASE.read_text(encoding='utf-8')
    generated = generate(base)
    try:
        result = cpu_gates(base, generated)
        if a.check or a.dry_run or a.linux_gate:
            assert OUT.read_bytes() == generated.encode(), 'generated file is stale or not LF'
        else:
            OUT.write_bytes(generated.encode())
        result['py_compile_files'] = py_compile_memory([pathlib.Path(__file__), OUT, CHECK])
        assert b'\r' not in (ROOT / 'tools/local/t1rb_gate.sh').read_bytes()
        print(json.dumps(result, indent=2))
    except Exception as ex:
        if a.linux_gate:
            a.linux_gate.write_text(json.dumps({'cpu': 'FAIL', 'error': repr(ex)}) + '\n', encoding='utf-8')
        raise
    if a.linux_gate:
        linux_gate(generated, a.linux_gate, result)


if __name__ == '__main__':
    main()

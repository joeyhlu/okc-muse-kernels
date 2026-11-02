#!/bin/bash
""":"
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
exec python -B "$0" --linux-gates "$@"
":"""
"""RID2 generator and CPU/compile-only gates. Bash entry runs Linux gates.

Only the named lab file and report are written by the normal CPU invocation.
The lead's Linux invocation preserves compiler scratch under /tmp (no deletion).
JSON is embedded in the named report; no unapproved JSON sidecar is created.
"""
import argparse
import ast
import builtins
import contextlib
import difflib
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexs_c1.py"
SOURCE = ROOT / "kernels/lab/k3_rq32_ab.py"
OUT = ROOT / "kernels/lab/k3_rid2_ab.py"
REPORT = ROOT / "box_out/reset/rid2_report.md"
Q32 = ('t32s6hzw', 32, 6, 1, 0, 0, 0, 9437187)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def sub(text, old, new, label):
    check(text.count(old) == 1, f"{label}: anchor occurs {text.count(old)} times")
    return text.replace(old, new, 1)


def nodes(text):
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node
    return result


def segment(text, name):
    return ast.get_source_segment(text, nodes(text)[name])


GLOBALS = '''# RID2: all selectors default off; changing them affects the next call/capture.
RID_HN = False  # Idempotent: the candidate already enables hn in HYB_F7.
RID_Q32 = False
RID_HYA1F = False
RID_Q32_PREPARE = False  # Opt in before prepare, then toggle RID_Q32 for A/B.
_RID2_Q32_VARIANTS = [('t32s6hzw',32,6,1,0,0,0,9437187)]
_RID2_Q32 = {"mod":None,"error":None,"build_s":0.0,"prepare_s":0.0}

'''


def make(base, source):
    """Clone only the proven RQ prepare/build skeleton; leave base arms intact."""
    bn, sn = nodes(base), nodes(source)
    check(Q32 in ast.literal_eval(sn['_rq_variants'].value), 'upstream q32 tuple')
    for name in ('_rq_cuda', '_rqsrc', '_rqwg', '_rqss', '_rq_pack_weight',
                 '_rq_validate', '_rq_weight', '_rq_call', '_rq_gemm'):
        check(ast.dump(bn[name]) == ast.dump(sn[name]), f'RQ upstream drift: {name}')
    # The policy-only hn transform is already present, including MOEXS F7 flags.
    check('HYB_F7=("-DOKC_HYBST=7","-DOKC_HN=1")' in base, 'hn flag missing')
    check('return q < n - 1 ? 0x1000000000000000ull : 0x12F0000000000000ull;' in base,
          'hn policy is not the proven policy-only transform')
    check('list(HYB_F7) if f7 else []' in segment(base, '_moexs_flags'), 'MOEXS inherits hn')

    build = segment(base, '_rq_build')
    build = sub(build, 'def _rq_build():', 'def _rid2_q32_build():', 'build name')
    build = build.replace('_rq_build_cache', '_RID2_Q32')
    build = sub(build, '_rqsrc(_rq_variants)', '_rqsrc(_RID2_Q32_VARIANTS)', 'build instance')
    build = sub(build, 'name="rq_layer_"+rq_tag', 'name="rq_rid2_"+rq_tag', 'module name')
    build = sub(build, '  try:\n', '  rid2_t0=time.time()\n  try:\n', 'build timer')
    build = sub(build, '   raise\n', '   raise\n  finally:\n   _RID2_Q32["build_s"]=time.time()-rid2_t0\n', 'build elapsed')

    prep = segment(base, '_rq_prepare')
    prep = sub(prep, 'def _rq_prepare(state):\n rq_keys=_rq_keys()',
               'def _rid2_q32_prepare(state):\n if not(RID_Q32_PREPARE or RID_Q32):\n  return\n'
               ' _pt1_wait("rid2_q32")\n rq_keys=(("qkv",32),)', 'prepare opt-in')
    prep = sub(prep, 'rq_bank=state["rq_bank"]={}', 'rq_bank=state["rq_bank"]', 'preserve base bank')
    prep = sub(prep, 'rq_mod=_rq_build()', 'rq_mod=_rid2_q32_build()', 'prepare build')
    prep = sub(prep, 'rq_i,rq_name=_rq_picks[rq_t]', 'rq_i,rq_name=0,"t32s6hzw-k7"', 'prepare pick')
    check(prep.count('_rq_variants[rq_i][7]') == 2, 'prepare flags count')
    prep = prep.replace('_rq_variants[rq_i][7]', '_RID2_Q32_VARIANTS[rq_i][7]')
    prep += '\n _RID2_Q32["prepare_s"]=_rq_time.monotonic()-rq_t0\n'
    prep += (' print("[rid2] q32 live=%s nvcc=%.3fs prepare=%.3fs"%'
             '(("qkv",32)in rq_bank,_RID2_Q32["build_s"],_RID2_Q32["prepare_s"]),flush=True)\n')

    out = sub(base, 'MOEXS = True\n', GLOBALS + 'MOEXS = True\n', 'selector globals')
    out = sub(out, 'def _moexs_select(m):\n',
              'def _moexs_select(m):\n'
              '    if RID_HN and m == 256:\n'
              '        assert "-DOKC_HN=1" in HYB_F7, "RID_HN requires existing hn build"\n', 'hn MOEXS')
    out = sub(out, ' if _HYB[k]!=pdl:\n',
              ' if RID_HN and m==256:\n'
              '  assert k=="pdl7" and "-DOKC_HN=1" in HYB_F7, "RID_HN mod7 unavailable"\n'
              ' if _HYB[k]!=pdl:\n', 'hn fallback')
    out = sub(out, ' for name,fn in jobs:\n',
              ' if RID_Q32_PREPARE or RID_Q32:\n'
              '  jobs.append(("rid2_q32",_rid2_q32_build))\n'
              ' for name,fn in jobs:\n', 'PT1 optional q32')
    out = sub(out, ' _rq_prepare(state)\n',
              ' _rq_prepare(state)\n _rid2_q32_prepare(state)\n', 'prepare hook')
    out = sub(out, ' r=HYA_ROUTE.get(_hya_key(cfg))\n',
              ' r=True if(RID_HYA1F and _hya_key(cfg)==("full",1,1))else HYA_ROUTE.get(_hya_key(cfg))\n', 'HYA d1f')
    out = sub(out, 'def _rq_gemm(state,proj,x):\n', build + '\n\n' + prep + '\n\ndef _rq_gemm(state,proj,x):\n', 'q32 bank functions')
    out = sub(out, ' if rq_key not in frozenset(tuple(x)for x in RQ_SEL):\n',
              ' rid2_q32=RID_Q32 and rq_key==("qkv",32)\n'
              ' if rid2_q32 and rq_key not in state.get("rq_bank",{}):\n'
              '  raise RuntimeError("RID_Q32 unavailable: enable RID_Q32_PREPARE before prepare and check validation")\n'
              ' if not rid2_q32 and rq_key not in frozenset(tuple(x)for x in RQ_SEL):\n', 'q32 capture selector')
    return out


def exec_nodes(text, names, env):
    tree = nodes(text)
    for name in names:
        exec(compile(ast.Module([tree[name]], []), str(OUT), 'exec'), env)


RQ_DEFS = ('_rq_cuda', '_rq_variants', '_rq_ms', '_rq_flags', '_rq_cpp',
           '_rqwg', '_rqss', '_rqsrc', 'RQ_SEL', 'RQ_PACK', '_rq_picks',
           '_rq_ks', '_rq_tol', '_rq_budget_s', '_rq_build_cache', '_rq_build',
           '_rq_keys', '_rq_weight', '_rq_call', '_rq_prepare', '_rq_gemm')


class Tensor:
    """Strict shape-only boundary double; no invented layer-state key fallback."""
    def __init__(self, shape, dtype='bf16', device='cpu'):
        self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        self.dtype, self.device = dtype, device

    def contiguous(self): return self
    def is_contiguous(self): return True
    def dim(self): return len(self.shape)
    def stride(self, axis): return math.prod(self.shape[axis+1:])
    def numel(self): return math.prod(self.shape)
    def element_size(self): return {'bf16': 2, 'u8': 1, 'i32': 4, 'i64': 8, 'f32': 4}[self.dtype]
    def data_ptr(self): return 16


def dry_run(base, out):
    calls, pack_calls, validations, waited = [], [], [], []
    fault = {'build': False, 'pack': False, 'escape': False, 'residency': False, 'validation': False}
    def loader(**kw):
        calls.append(kw)
        if fault['build']: raise RuntimeError('injected build failure')
        return NS(rq_run=lambda *a: (0 if fault['residency'] else 1) if a[-1] == 4 else 0)
    def importer(name, *args, **kw):
        if name == 'torch.utils.cpp_extension': return NS(load_inline=loader)
        return builtins.__import__(name, *args, **kw)
    torch = NS(bfloat16='bf16', int32='i32', int64='i64', float32='f32',
               empty=Tensor, zeros=Tensor,
               Generator=lambda **kw: NS(manual_seed=lambda n: n),
               cuda=NS(get_device_properties=lambda dev: NS(multi_processor_count=132)))
    env = {'__builtins__': dict(vars(builtins), __import__=importer), 'torch': torch, 'time': time}
    exec_nodes(out, RQ_DEFS, env)
    exec_nodes(out, ('RID_HN', 'RID_Q32', 'RID_HYA1F', 'RID_Q32_PREPARE',
                    '_RID2_Q32_VARIANTS', '_RID2_Q32', '_rid2_q32_build', '_rid2_q32_prepare'), env)
    def pack(w):
        pack_calls.append(w.shape)
        if fault['pack']: raise ValueError('injected lossless gate failure')
        return Tensor((w.numel()*3//2,), 'u8'), 0, object()
    def validate(p, w, t, gen):
        validations.append((p['rq_name'], t, gen))
        return not fault['validation'], 'CPU boundary double'
    env.update(_rq_pack_weight=pack, _rq_validate=validate,
               _rqesc=lambda *a: (Tensor((128,), 'i32'),),
               _rqesc2=lambda *a: (None, 99) if fault['escape'] else (Tensor((128,), 'i32'), 0),
               _pt1_wait=waited.append)
    # Derive the real dense state constructor from _r1_prepare, never invent its keys.
    constructor = next(n.value for n in ast.walk(nodes(base)['_r1_prepare'])
                       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'state' for t in n.targets))
    dense_expr = next(v for k, v in zip(constructor.keys, constructor.values)
                      if isinstance(k, ast.Constant) and k.value == 'dense')
    weights = {'qkv_gate': Tensor((2304,8192)), 'o': Tensor((8192,1024)),
               'pooler': Tensor((4096,8192)), 'unpooler': Tensor((8192,4096))}
    state = {'dense': eval(compile(ast.Expression(dense_expr), str(BASE), 'eval'), {'weights': weights})}
    # Execute the actual two prepare call sites, as in the padz capture-the-real-state pattern.
    hooks = [n for n in nodes(out)['prepare'].body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name) and n.value.func.id in ('_rq_prepare','_rid2_q32_prepare')]
    check(len(hooks) == 2, 'prepare hooks')
    env['state'] = state
    exec(compile(ast.Module(hooks, []), str(OUT), 'exec'), env)
    check(set(state['rq_bank']) == {('qkv',128),('unpooler',128)}, 'off bank equals base')
    check(len(calls) == 1 and len(pack_calls) == 2 and not waited, 'off adds no build/pack/wait')
    initial = dict(state['rq_bank'])
    env['RID_Q32'] = True
    try:
        env['_rq_gemm'](state, 'qkv', Tensor((32,8192)))
    except RuntimeError:
        pass
    else:
        raise AssertionError('unprepared q32 silently fell back')
    env['RID_Q32'] = False
    env['RID_Q32_PREPARE'] = True
    env['_rid2_q32_prepare'](state)
    check(state['rq_bank'][('qkv',32)]['rq_name'] == 't32s6hzw-k7', 'q32 pick')
    check(state['rq_bank'][('qkv',32)]['rq_i'] == 0, 'q32 isolated module index')
    check(all(state['rq_bank'][k] is v for k,v in initial.items()), 'existing picks not rebuilt')
    check(len(calls) == 2 and len(pack_calls) == 3 and waited == ['rid2_q32'], 'only q32 extra work')
    check([v[1] for v in validations] == [128,128,32], 'validation shapes')
    check('rq_go<32, 6, 1, 0, 0, 0, 9437187, 1>' in calls[-1]['cuda_sources'], 'exact q32 instance')
    # Call-time selectors do no packing, compilation or preparation.
    counts = len(calls), len(pack_calls), len(validations)
    for enabled in (False,True,False,True):
        env['RID_Q32'] = enabled
        for p,t in itertools.product(('qkv','o','unpooler'), (1,8,16,32,128,256)):
            got = env['_rq_gemm'](state,p,Tensor((t,weights['qkv_gate' if p=='qkv' else p].shape[1])))
            check((got is not None) == ((p,t) in env['RQ_SEL'] or (enabled and (p,t)==('qkv',32))), 'call selector scope')
    check(counts == (len(calls),len(pack_calls),len(validations)), 'call-time preparation')
    # Failure arms keep the validated baseline entries; selecting a failed bank is loud.
    for key in fault:
        fault[key] = True
        env['_RID2_Q32'].update(mod=None,error=None)
        state['rq_bank'] = dict(initial)
        env['_rid2_q32_prepare'](state)
        check(set(state['rq_bank']) == set(initial), f'{key}: failed arm leaked')
        fault[key] = False
    # Enabling RID_Q32 before prepare is also an explicit prepare opt-in.
    env['_RID2_Q32'].update(mod=None,error=None)
    env['RID_Q32_PREPARE'], env['RID_Q32'] = False, True
    env['_rid2_q32_prepare'](state)
    check(('qkv',32) in state['rq_bank'], 'selector-before-prepare path')
    # Real _pt1_start, fake threads (no function or external command is started).
    class Thread:
        def __init__(self, target, name, daemon): self.name = name
        def start(self): pass
    env['threading'] = NS(Thread=Thread)
    pt = nodes(out)['_pt1_start']
    for n in ast.walk(pt):
        if isinstance(n, ast.Name) and n.id not in env and n.id not in ('jobs','name','fn','th','print'):
            env[n.id] = (lambda *a: None) if n.id.startswith('_') else False
    env['PT1_PREFETCH'] = True
    env['_pt1_target'] = lambda *a: None
    exec_nodes(out, ('_pt1_start',), env)
    for available, enabled in itertools.product((False,True), repeat=2):
        env['RID_Q32_PREPARE'], env['RID_Q32'] = available, enabled
        env['_PT1'] = {'threads':{}}
        env['_pt1_start']()
        check(('rid2_q32' in env['_PT1']['threads']) == (available or enabled), 'PT1 capability gate')

    # Execute real HYA predicates at every scored shape, plus unsupported inputs.
    exec_nodes(out, ('HYA_ENABLE','HYA_ROUTE','HYA_DEFAULT','_hya_key','_hya_wanted',
                     'HYB_F7','HYB_T7','MOEXS','_MOEXS_PREPARING','_moexs_flags','HYB_FLAGS','_moexs_select'), env)
    env.update(_HYA={'mod':object(),'kvs':True}, FP8='fp8', _hyb_on=lambda m: True)
    env['_MOEXS'] = {False:{'ok':True},True:{'ok':True}}
    baseline = dict(env)
    exec_nodes(base, ('_hya_wanted','_moexs_select'), baseline)
    for hn,q32,hya in itertools.product((False,True),repeat=3):
        env.update(RID_HN=hn,RID_Q32=q32,RID_HYA1F=hya)
        for m in (1,8,16,32,128,256,320):
            check(env['_moexs_select'](m) is baseline['_moexs_select'](m), 'hn identity')
        for B,L in ((1,1),(8,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
            for window in (0,2048):
                cfg=dict(batch=B,q_len=L,window=window,num_q_heads=16,num_kv_heads=2,page_size=1,head_dim=64,kv_scale=1.)
                data=dict(k_cache=Tensor((90000,2,64),'fp8'),v_cache=Tensor((90000,2,64),'fp8'),
                          page_table=Tensor((B,80000),'i32'),seq_lens=Tensor((B,),'i32'),
                          cache_loc=Tensor((B*L,),'i64'),k_scale=Tensor((1,),'f32'),v_scale=Tensor((1,),'f32'))
                x=Tensor((B*L,2304))
                for first in (False,True):
                    old=baseline['_hya_wanted'](x,data,cfg,{'hya':True},first)
                    got=env['_hya_wanted'](x,data,cfg,{'hya':True},first)
                    check(got == (True if hya and (B,L,window)==(1,1,0) else old), 'HYA route scope')
                    check(not env['_hya_wanted'](x,data,cfg,{'hya':False},first), 'HYA readiness guard')
    check('-DOKC_HN=1' in env['_moexs_flags'](True) and '-DOKC_HN=1' not in env['_moexs_flags'](False), 'hn flag scope')
    # The other HN call site is the original HYB fallback (MOEXS disabled).
    exec_nodes(out, ('_hyb_call',), env)
    baseline_call = dict(env)
    exec_nodes(base, ('_hyb_call',), baseline_call)
    launched = []
    modules = {key:NS(okc_set_pdl=lambda p: None,
                      okc_run=lambda *args, key=key: launched.append(key)) for key in ('mod','mod7')}
    for hn,m in itertools.product((False,True),(1,8,16,32,128,192,193,256,320)):
        env.update(RID_HN=hn,MOEXS=False,HYB_PDL=True,
                   _HYB=dict(modules, w=object(),sc=object(),nsm=132,pdl=-1,pdl7=-1))
        baseline_call.update(_HYB=env['_HYB'],HYB_PDL=True)
        env['_hyb_call'](Tensor((m,4096)),None,None)
        baseline_call['_hyb_call'](Tensor((m,4096)),None,None)
        check(launched[-2] == launched[-1] == ('mod7' if m>=193 else 'mod'), 'HN fallback identity')
    env['RID_HN']=True
    env['_HYB']['mod7']=None
    try:
        env['_hyb_call'](Tensor((256,4096)),None,None)
    except AssertionError:
        pass
    else:
        raise AssertionError('HN absent mod7 was not rejected')
    return {'status':'PASS','state_constructor':'real _r1_prepare dense dict AST',
            'prepare_hooks':'actual _rq_prepare + _rid2_q32_prepare AST',
            'scored_route_combinations':8,'failure_paths':list(fault),
            'gpu_boundaries':'shape-only torch/compiler/codec/numerics doubles; no GPU execution'}


class DefaultArm(ast.NodeTransformer):
    """Erase only the added disabled arms, then compare with the real base AST."""
    def visit_Name(self,node):
        if isinstance(node.ctx,ast.Load) and node.id in ('RID_HN','RID_Q32','RID_Q32_PREPARE','RID_HYA1F','rid2_q32'):
            return ast.Constant(False)
        return node

    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='rid2_q32' for t in node.targets): return None
        return self.generic_visit(node)

    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_rid2_q32_prepare':
            return None
        return self.generic_visit(node)

    def visit_UnaryOp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.op,ast.Not) and isinstance(node.operand,ast.Constant) and isinstance(node.operand.value,bool):
            return ast.Constant(not node.operand.value)
        return node

    def visit_BoolOp(self,node):
        node=self.generic_visit(node)
        # Only leading bool constants: preserve Python's short-circuit evaluation.
        while node.values and isinstance(node.values[0],ast.Constant) and isinstance(node.values[0].value,bool):
            value=node.values.pop(0).value
            if isinstance(node.op,ast.And) and not value: return ast.Constant(False)
            if isinstance(node.op,ast.Or) and value: return ast.Constant(True)
        if not node.values: return ast.Constant(isinstance(node.op,ast.And))
        return node.values[0] if len(node.values)==1 else node

    def visit_If(self,node):
        node=self.generic_visit(node)
        return (node.body if node.test.value else node.orelse) if isinstance(node.test,ast.Constant) and isinstance(node.test.value,bool) else node

    def visit_IfExp(self,node):
        node=self.generic_visit(node)
        return (node.body if node.test.value else node.orelse) if isinstance(node.test,ast.Constant) and isinstance(node.test.value,bool) else node


def cpu_gates(base, out, source):
    check(len(out.encode('ascii')) < 990000, 'strict size cap')
    check('\r' not in out and out.startswith('# okc: problem=muse_decode_layer\n'), 'LF/header')
    compile(out, str(OUT), 'exec')  # no pyc write
    bn, on = nodes(base), nodes(out)
    changed = {k for k in bn if ast.dump(bn[k]) != ast.dump(on[k])}
    check(changed == {'_moexs_select','_hyb_call','_pt1_start','prepare','_hya_wanted','_rq_gemm'}, f'changed AST scope {changed}')
    for name in changed:
        check(ast.dump(DefaultArm().visit(on[name])) == ast.dump(DefaultArm().visit(bn[name])), f'default semantic identity: {name}')
    check(make(base,source) == out, 'deterministic generation')
    # Preserve all device source, constants, notices, original builds and validations.
    for key in bn:
        if key not in changed: check(ast.dump(bn[key]) == ast.dump(on[key]), key)
    old, new = {}, {}
    exec_nodes(base, RQ_DEFS[:9], old)
    exec_nodes(out, RQ_DEFS[:9], new)
    check(old['_rqsrc'](old['_rq_variants']) == new['_rqsrc'](new['_rq_variants']), 'default RQ build bytes')
    with contextlib.redirect_stdout(io.StringIO()):
        dry = dry_run(base,out)
    return {'syntax_ascii_lf_size':'PASS','determinism':'PASS','unchanged_device_sources':'PASS',
            'default_RQ_build_source_identical':'PASS','all_default_function_ASTs_equal':'PASS',
            'changed_existing_functions':sorted(changed),'dry_run':dry}


def linux_gates(out):
    """Compile-only. No torch CUDA allocation, import of the layer, or GPU launch."""
    import torch.utils.cpp_extension as ce
    ns = {}
    exec_nodes(out, RQ_DEFS[:9] + ('_RID2_Q32_VARIANTS',), ns)
    reference = {}
    exec_nodes(SOURCE.read_text(), RQ_DEFS[:9], reference)
    inc = [arg for p in ce.include_paths(device_type='cuda') for arg in ('-I',p)]
    inc += ['-I',sysconfig.get_paths()['include']]
    common = ['-cubin','-std=c++20','-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__',
              '-D__CUDA_NO_BFLOAT16_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__',
              '--expt-relaxed-constexpr','-DTORCH_EXTENSION_NAME=rid2_gate','-DTORCH_API_INCLUDE_EXTENSION_H',
              '-D_GLIBCXX_USE_CXX11_ABI=1','-Xptxas=-v']
    work = Path(tempfile.mkdtemp(prefix='rid2_compile_'))  # retained, never delete
    result = {'status':'FAIL','scratch':str(work),'jobs':{}}
    # Reusing one input pathname avoids source-path-derived SASS naming differences.
    jobs = {'baseline':ns['_rqsrc'](ns['_rq_variants']),
            'rid2_q32':ns['_rqsrc'](ns['_RID2_Q32_VARIANTS']),
            'proven_q32':reference['_rqsrc']([Q32])}
    check(jobs['rid2_q32'] == jobs['proven_q32'], 'Q32 device code exact upstream')
    for name, src in jobs.items():
        cu, cubin = work/'input.cu', work/(name+'.cubin')
        cu.write_bytes(('#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n'+src).encode())
        started = time.monotonic()
        proc = subprocess.run(['nvcc']+common+inc+ns['_rq_flags']+[str(cu),'-o',str(cubin)], capture_output=True,text=True,timeout=300)
        row = result['jobs'][name] = {'compile_s':time.monotonic()-started,'returncode':proc.returncode,'log':proc.stderr}
        if proc.returncode: continue
        dump = subprocess.run(['cuobjdump','-sass',str(cubin)],capture_output=True,text=True,timeout=60,check=True).stdout
        # Normalize only addresses and instruction encodings; preserve every opcode/operand.
        sass = [re.sub(r'/\*.*?\*/','',s).strip() for s in dump.splitlines() if re.match(r'\s*/\*[0-9a-f]+\*/\s+\S',s)]
        row.update(sass_instructions=len(sass),sass_sha256=hashlib.sha256('\n'.join(sass).encode()).hexdigest(),
                   registers=[int(n) for n in re.findall(r'Used (\d+) registers',proc.stderr)],
                   stack_spills=[list(map(int,m)) for m in re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',proc.stderr)],
                   local_or_trap=bool(re.search(r'\b(?:LDL|STL|BPT\.TRAP)\b','\n'.join(sass))))
        (work/(name+'.sass')).write_bytes(dump.encode())
    good = all(r['returncode']==0 and r.get('sass_instructions',0)>0 and r.get('registers')
               and r.get('stack_spills') and not any(any(v) for v in r['stack_spills']) and not r['local_or_trap']
               for r in result['jobs'].values())
    a,b = result['jobs']['rid2_q32'],result['jobs']['proven_q32']
    good = good and a.get('sass_sha256') == b.get('sass_sha256') and a.get('registers') == b.get('registers')
    result['status'] = 'PASS' if good else 'FAIL'
    return result


def report(base,out,gates,cuda):
    changes = []
    for tag,a,b,c,d in difflib.SequenceMatcher(None,base.splitlines(),out.splitlines(),autojunk=False).get_opcodes():
        if tag!='equal': changes.append({'base':f'{a+1}-{b}' if b>a else f'after {a}', 'lab':f'{c+1}-{d}', 'kind':tag})
    lines=out.splitlines()
    def line(needle):
        found=[i+1 for i,s in enumerate(lines) if s.startswith(needle)]
        check(len(found)==1, f'report anchor {needle!r}: {found}')
        return found[0]
    on=nodes(out)
    def span(name): return f'{on[name].lineno}-{on[name].end_lineno}'
    qjob=line('  jobs.append(("rid2_q32",_rid2_q32_build))')
    hnlaunch=next(i+1 for i,s in enumerate(lines) if on['_HYB_MOE_CUDA'].lineno<=i+1<=on['_HYB_MOE_CUDA'].end_lineno
                  and s.startswith('if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);'))
    per_bit={
        'RID_HN':{'selector':line('RID_HN = False'),'MOEXS guard':f'{line("    if RID_HN and m == 256:")}-{line("    if RID_HN and m == 256:")+1}',
                  'HYB fallback guard':f'{line(" if RID_HN and m==256:")}-{line(" if RID_HN and m==256:")+1}'},
        'RID_Q32':{'selector':line('RID_Q32 = False'),'prepare capability/cache':f'{line("RID_Q32_PREPARE = False")}-{line("_RID2_Q32 = ")}',
                   'PT1 job':f'{qjob-1}-{qjob}',
                   'prepare hook':line(' _rid2_q32_prepare(state)'), 'build':span('_rid2_q32_build'),
                   'bank construction':span('_rid2_q32_prepare'),'call selector':f'{line(" rid2_q32=RID_Q32")}-{line(" if not rid2_q32 and")}'},
        'RID_HYA1F':{'selector':line('RID_HYA1F = False'),'route':line(' r=True if(RID_HYA1F')},
    }
    data = dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),lab_sha256=hashlib.sha256(out.encode()).hexdigest(),
                bytes=len(out.encode()),cpu=gates,cuda=cuda,per_bit_lab_lines=per_bit,changed_lines=changes)
    text = '''| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |
|---|---|---|---|---|---|---|
| RID_HN; kills: already active; policy effect may be hidden by stream | T=256, v32s/f | Historical delta -0.15/-2.54, R-hybk-1, notes/lab.md:1117; current candidate timing not measured here | 0 = identical module + flags + launch | No new critical-path work removed | Stream geometry / L2 hints, R-b4m-0 | Already in base; explicit idempotent selector |
| RID_Q32; kills: issue work hidden; split reduction/escape overhead | d32s/f | Historical qkv 11.1/11.0 (profiles/k3_rq128_c1.3ad6bd2659ef4691.txt:88/106); combo5 layer delta +0.5/0.0, -5.4/-1.4 (task; notes/lab.md:1231); MOEXS value unmeasured | Net unknown; no fresh speed claim | qkv gates attention; stock sk2 qkv has no incoming PDL | R-sk2-1; R-gdd-1 | Optional arm, default off; GPU verification pending |
| RID_HYA1F; kills: below in-layer floor; changed floating-point reduction order | d1f only | Historical layer delta -0.77/-0.8, notes/lab.md:1137/974 | Net unknown | Existing HYA route takes precedence over AX1 | R-hyas-1 / R-hya-1b | Requested optional arm, default off |

All three selectors default False; MOEXS stays True. Default CUDA source, flags, original module names, prepare paths and device launches are unchanged. Default identity is a CPU/source claim, not a fresh GPU bitwise result.

RID_HN: the candidate already has HYB_F7=(-DOKC_HYBST=7,-DOKC_HN=1), and _moexs_flags(True) inherits it. Existing hk_pol selects 0x1000000000000000 on non-final expert passes and 0x12F0000000000000 on the last. The selector only checks this existing build at T=256. No second copy, scheduling change or nvcc job; expected bitwise relative to the same base execution (base atomic repeat variability remains). Added prepare work: zero builds/allocations/validation launches.

RID_Q32: set RID_Q32_PREPARE=True before prepare; then toggle RID_Q32 at call/capture time. Alternatively set RID_Q32=True before prepare. A selected but absent bank raises instead of silently timing sk2. The optional PT1 job compiles one separate rq_kernel<32,6,1,0,0,0,9437187,1>, runtime KS=7, grid (18,7), 384 threads, EST smem 1024+6*(4096+12288)+24*6+16=99488 B. It leaves the baseline RQ module/cache and T=128 bank entries untouched. New preparation copies the proven base prepare skeleton and exact state["dense"]["qkv_gate" if proj=="qkv" else proj] accessor; no state["k2a"] accessor. Packed weights remain lossless BF16; different FP32 partial/reduction order means tolerance class, not bitwise. Existing rel<=8e-3 and per-row<=4e-2 validation, poison/repeats, reset and residency checks remain. Full-layer tolerance is unverified on this bundle.

RQ32 added prepare work, only on opt-in: one optional parallel NVCC build; one qkv pack/unpack roundtrip; one escape list and cursor construction; one residency query; two reference matmuls and four validation launches. Exact added seconds unmeasured; _RID2_Q32["build_s"] and ["prepare_s"] plus the rid2 log expose them. No prepare-time timing sweep. The existing 60-second RQ admission guard is copied; wall time/300-second cap still requires the lead's cold-prepare measurement.

RID_HYA1F: override only _hya_key(cfg)==("full",1,1); normal HYA eligibility/self-test checks stay. Sliding T=1 keeps KS1. Existing HYA d1f tuned plan: splits=252, nc=8, coop=1; source and all workspace/reset protocols unchanged. Uses already prepared HYA; zero extra NVCC, allocations or validation launches. Non-bitwise tolerance class (historical relative layer difference 0.00396, box_out/hyas/summary.json); no new numerical claim.

No device hot loop was edited: before/after bytes per tile/stage and instructions per thread of every existing instance are identical (delta 0). The new q32 instance is byte-identical to the requested upstream specialization. Source accounting (EST, not measured SASS): rq FULL N128,K64 stage has W=12288 B plus escapes and X=4096 B; each consumer thread issues 3 vector W loads + 36 decode primitives + 4 RS-WGMMA = 43 counted primitives, plus original variable escape/control/fence/commit/wait work. Its producers issue one W bulk and one X TMA. Before/after versus the proven rq32 arm is exactly 16384 B and 43 primitives on both sides.

Route comparison (EST source counts, box_out/reset/rq32_report.md:29-33): existing sk2 qkv N24,K512 stage has W=4*2*3*768=18432 B, X=8*32*128=32768 B, total=51200 B; consumer/thread has 12 W LDS + 16 X LDSM + 144 decode + 48 MMA = 220 primitives plus escape/control work. Normalized to N128,K64, W+X is 34133.3 B versus rq 16384 B; X reuse is mainly L2, so no HBM saving is claimed. Actual total instruction counts require SASS. MOEXS's existing SC6 W stage remains 35840 B (4*8960), including for RID_HN=True; HYB fallback without SC6 remains 36864 B. RQ logical extra split output traffic is EST 4*7*32*2304=2064384 B, stream equivalent 2064384/4.64e6=0.445 us if all reaches HBM; this is a cost bound, not a net saving. No bytes were added to any existing sk2/GD/VMS/VML/HYB hot loop. No PDL edge or launch is removed, so no edge-time credit is claimed.

No early exit, barrier, flag, counter, reset or combine code was changed. Q32 inherits the FULL instance's existing protocol: residency admits all 18*7=126 CTAs against 132*occupancy; counter storage 8*18=144 int32 elements covers the two 18-element flags. CTA x=0..17, y=0..6 remain resident. Threads 320..383 zero their split's token interval [floor(y*32/7),floor((y+1)*32/7)) across 128 columns (16-byte stores, strided by 64 threads); thread 320 polls readiness, publishes zflag, and the last of the seven pollers resets both counters. Consumer threads 0..255 retain the original two-WG output staging/reduce-add and waits. Producer warps 8/9 and zeroing warps 10/11 retain all their original returns; no new CTA/thread exits exist. Existing prepare validation checks zero counters after each launch. HYB/HYA cleanup coverage follows exact source identity, with no new protocol change.

CPU gates execute actual prepare hook ASTs and the real base dense-state constructor, using strict shape-only tensor/compiler/codec/numeric boundary doubles. They cover opt-out, both opt-in methods, preserved T=128 entries, PT1 gating, call-time switching, unprepared selection, build/pack/escape/residency/numeric failures, all 8 selector combinations and scored HYA/MOEXS route scope. Source/AST checks preserve every device string, original validations and kernel_fn. These tests do not replace GPU numerical validation.

Only the three named files are created/edited. To respect that allowlist, tools/rid2_patches.py is also an LF Bash entrypoint; machine-readable results are embedded below in this named report rather than creating an extra adjacent JSON file. The lead's compile-only invocation preserves compiler scratch under /tmp, runs CPU gates then NVCC/SASS checks, and updates this report with JSON results. It does not launch a GPU kernel, pod or judge.

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" bash "<repo>/tools/rid2_patches.py"
```

CPU command: `python -B tools/rid2_patches.py --cpu-only`. WSL/NVCC, SASS, H200 eager/capture, fresh timings, and actual cold prepare seconds could not run in this sandbox.

'''
    text += f'Generated file: {len(out.encode())} B (<990000 B); increase {len(out.encode())-len(base.encode())} B. Per-bit exact lab changes:\n\n'
    for bit, locations in per_bit.items():
        text += '- '+bit+': '+ '; '.join(f'{k} {v}' for k,v in locations.items())+'.\n'
    text += '\nExact affected specializations (all references below are kernels/lab/k3_rid2_ab.py):\n\n'
    text += (f'- T=256 sliding/full: MOEXS smp/F7 -> okc::moe_fused<true>, '
             f'flags {on["_moexs_flags"].lineno}, selector {on["_moexs_select"].lineno}, '
             f'launch {hnlaunch}; '
             'HN reuses it. T=128 smp/F0 is unchanged. HYB mod7 fallback remains its original true instance.\n')
    text += (f'- T=32 sliding/full qkv: original launch_sk2<2,32,3,1,4,2,4,true,false,1,1,__nv_bfloat16> '
             f'at {line("if (cfg == 6 && tokr == 32)")} -> rq_kernel<32,6,1,0,0,0,9437187,1>, KS=7; '
             f'tuple {line("_RID2_Q32_VARIANTS = ")}, runtime selection {line(" rid2_q32=RID_Q32")}. '
             'T=1/8/16/128/256 qkv and every o/unpooler route remain unchanged.\n')
    text += (f'- T=1 full: ax1::fb1_l1<2> (launch {line("C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fb1_l1<ST>, P));")}) '
             f'-> HYA attn_kernel<1,false,11> (ksp[3], table {line("static const KernFn ksp[4]")}; '
             'mode 9 if its unchanged occupancy guard disables coop). At the scored full context, 252*64 keys is below the width, '
             'so the one-tile mode is not selected. Sliding T=1 stays KS1; T=8/16/32/128/256 attention routes remain unchanged.\n')
    text += '\nFull CPU/compile results and base-to-lab hunks:\n\n```json\n'
    REPORT.write_bytes((text+json.dumps(data,indent=2)+'\n```\n').encode('utf-8'))
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linux-gates',action='store_true')
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    base=BASE.read_bytes().decode('ascii').replace('\r\n','\n')
    source=SOURCE.read_bytes().decode('ascii').replace('\r\n','\n')
    out=make(base,source)
    gates=cpu_gates(base,out,source)
    OUT.write_bytes(out.encode('ascii'))
    lead=subprocess.run([sys.executable,'-B',str(ROOT/'tools/lead_gate.py'),'file',str(OUT),'--raw='+str(BASE)],
                        capture_output=True,text=True,check=False)
    check(lead.returncode==0, 'lead_gate failed: '+lead.stdout+lead.stderr)
    gates['lead_gate']={'status':'PASS','log':lead.stdout.strip()}
    cuda={'status':'NOT_RUN','reason':'WSL/NVCC unavailable in sandbox'}
    if args.linux_gates and not args.cpu_only:
        try:
            cuda=linux_gates(out)
        except Exception as ex:
            cuda={'status':'FAIL','error':repr(ex)}
    data=report(base,out,gates,cuda)
    print(json.dumps({'bytes':data['bytes'],'cpu':'PASS','cuda':cuda['status'],'report':str(REPORT)}))
    if cuda['status']=='FAIL': raise SystemExit(1)


if __name__=='__main__':
    main()

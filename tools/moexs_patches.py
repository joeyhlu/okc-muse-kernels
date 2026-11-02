#!/bin/bash
""":"
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
exec python3 -B "$0" --check --sass "$@"
": """
"""MOEXS single-source smp port. Only the named layer and report are outputs.

python -B tools/moexs_patches.py          # generate, CPU gates, report
python -B tools/moexs_patches.py --check  # reproduce and check without rewriting layer
bash tools/moexs_patches.py              # Linux offline compile/SASS; no GPU launches
The bash entry is LF and is intended to run under tools/local/cudaenv.sh.
Compiler intermediates are retained under /tmp by the lead's Linux invocation.
JSON is embedded in the allowed report, rather than writing a fourth file.
"""
import argparse
import ast
import builtins
import contextlib
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/lab/k3_diet_rq128.py'
OUT = ROOT / 'kernels/lab/k3_moexs_ab.py'
REPORT = ROOT / 'box_out/reset/moexs_report.md'
GEN = runpy.run_path(str(ROOT / 'tools/moex_patches.py'))
GATES = runpy.run_path(str(ROOT / 'tools/local/moex_compile.py'))
once, literal, nodes = GEN['once'], GEN['literal'], GATES['nodes']
BITS = ('SC6', 'MFIN', 'PLAN')


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node) + '\n'


def feature_view(cu, values):
    """Fold only our feature conditionals; preserve every unrelated directive.

    This is a token-identity gate, not a replacement for the NVCC SASS gate.
    """
    out, stack, emit = [], [], True
    for line in cu.splitlines(True):
        m = re.match(r'\s*#\s*(if|ifdef|ifndef|else|elif|endif)\b(.*)', line)
        if m:
            kind, expr = m[1], m[2].strip()
            if kind in ('if','ifdef','ifndef'):
                key = expr.lstrip('!').strip()
                controlled = key in values
                cond = ((not values[key]) if expr.startswith('!') else bool(values[key])) if controlled else True
                if controlled and kind in ('ifdef','ifndef'): cond = kind == 'ifdef'
                stack.append((controlled,emit,cond))
                if controlled: emit = emit and cond
                elif emit: out.append(line)
            elif kind == 'endif':
                controlled,parent,_ = stack.pop()
                if not controlled and emit: out.append(line)
                emit = parent
            elif kind == 'else':
                controlled,parent,cond = stack[-1]
                if controlled: emit = parent and not cond
                elif emit: out.append(line)
            else:
                assert not stack[-1][0], 'unexpected feature elif'
                if emit: out.append(line)
            continue
        if emit: out.append(line)
    assert not stack
    s=''.join(out)
    if not values.get('OKC_MFIN',0):
        for alias,real in (('mf_xr','p.xready'),('mf_da','p.doneAs')):
            s=once(s,f'#define {alias}(p, S) {real}\n','') if f'#define {alias}' in s else s
            s=s.replace(alias+'(p, S)',real)
    if '#define WGS_COPY_BYTES' in s:
        value=re.search(r'^#define WGS_COPY_BYTES (\w+)$',s,re.M)[1]
        s=once(s,'#define WGS_COPY_BYTES '+value+'\n','').replace('WGS_COPY_BYTES',value)
    # Drop comments/whitespace while preserving complete C++ string tokens.
    tokens=re.findall(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|[A-Za-z_]\w*|\d+|[^\s]',s,re.S)
    return [x for x in tokens if not x.startswith(('//','/*'))]


GLUE = r'''
MOEXS = False
_MOEXS_BITS = {"smp": (1, 1, 1, 0)}
_MOEXS = {f: {"mod": None, "ok": False, "sc": None, "full": None, "w": None,
              "pdl": -1, "err": None, "checks": [], "build_s": 0.0} for f in (False, True)}
_MOEXS_PREPARING = False

def _moexs_flags(f7):
    return list(HYB_FLAGS) + (list(HYB_F7) if f7 else []) + ["-DOKC_SC6=1", "-DOKC_MFIN=1", "-DOKC_PLAN=1"]

def _moexs_build_one(f7):
    from torch.utils.cpp_extension import load_inline
    H, t = _MOEXS[f7], time.time()
    H["ok"] = False
    try:
        fl = _moexs_flags(f7)
        tag = hashlib.sha1((_HYB_MOE_CPP + _HYB_MOE_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
        H["mod"] = load_inline(name="hyb_moexs_" + tag, cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYB_MOE_CUDA,
                              functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
        H["err"] = None
    except Exception as ex:
        H["mod"], H["err"] = None, repr(ex)[-600:]
    finally:
        H["build_s"] = time.time() - t

def _moexs_select(m):
    if not MOEXS or _MOEXS_PREPARING or m not in (128, 256):
        return None
    H = _MOEXS[m >= HYB_T7]
    if not _hyb_on(m) or not H["ok"]:
        raise RuntimeError("MOEXS smp disabled: " + str(H["err"]))
    return H

def _moexs_budget(t0):
    if time.time() - t0 >= 290.0:
        raise TimeoutError("MOEXS validation budget; bundle guards unchanged")

def _moexs_prepare(st, w, dev, t0):
    global _MOEXS_PREPARING
    _MOEXS_PREPARING = True
    t1 = time.time()
    try:
        for H in _MOEXS.values():
            H["ok"], H["w"], H["sc"], H["full"], H["checks"] = False, None, None, None, []
        if not _HYB["ok"]:
            raise RuntimeError("ship HYB did not pass")
        # Added jobs start with PT1, but do not delay guard-timed bundle prepares.
        # No synchronous rebuild, extra codec, or retry after a failed build.
        for f7, H in _MOEXS.items():
            key = "moexs7" if f7 else "moexs"
            th = _PT1["threads"].get(key)
            if th is None:
                H["err"] = "PT1 prefetch disabled/missing"
                continue
            th.join(max(0.0, 280.0 - (time.time() - t0)))
            if th.is_alive():
                H["err"] = "compile deadline exceeded"
        _moexs_budget(t0)
        # Reject unsupported scales; keep exactly two additional SC6 builds.
        if not (_moexs_sc6_ok(w["w13_sf"], w["w2_sf"]) and _hyb_prmt_ok(w["w13_sf"], w["w2_sf"])):
            raise ValueError("SC6/PRMT exact scale range failed")
        _, packed = _moexs_weights(w)
        for f7, H in _MOEXS.items():
            try:
                _moexs_budget(t0)
                if H["mod"] is None or H["err"]:
                    raise RuntimeError(H["err"] or "build unavailable")
                md = _HYB["mod7"] if f7 else _HYB["mod"]
                if md is None:
                    raise RuntimeError("matching ship build unavailable")
                _moexs_bind(H, "smp", dev, True, packed)
                refH = {"mod": md, "w": _HYB["w"], "sc": _HYB["sc"], "pdl": -1}
                good = True
                T = 256 if f7 else 128
                for dist, seed in (("normal", 991), ("skewed", 992)):
                    _moexs_budget(t0)
                    d = _k2a_self_test_inputs(T, seed, dev, dist)
                    x, ids, tw = d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(), d["topk_weights"].float().contiguous()
                    fp32 = _k2a_ref_fp32(d, w)
                    a = _moexs_run(refH, x, ids, tw).clone()
                    b = _moexs_run(refH, x, ids, tw).clone()
                    torch.cuda.synchronize()
                    good &= _moexs_check(H, "smp", f7, f"T={T}/{dist}", x, ids, tw, b, fp32, _k2a_rel_err(a, b)[0])
                _moexs_budget(t0)
                good &= _moexs_graph_check(H, "smp", f7, dev)
                _moexs_budget(t0)
                H["ok"], H["err"] = bool(good), None if good else "numerical/graph/device-error/guard failure"
            except Exception as ex:
                H["ok"], H["err"] = False, repr(ex)[-600:]
            print(f"[moexs] smp/F7={int(f7)} live={H['ok']} nvcc={H['build_s']:.2f}s err={H['err']}", flush=True)
    except Exception as ex:
        for H in _MOEXS.values():
            H["ok"], H["err"] = False, repr(ex)[-600:]
        print(f"[moexs] DISABLED {ex!r}"[:700], flush=True)
    finally:
        _MOEXS_PREPARING = False
        print(f"[moexs] extra prepare {time.time() - t1:.2f}s; elapsed {time.time() - t0:.2f}s", flush=True)
'''


def generate(base):
    cu = literal(base, '_HYB_MOE_CUDA')
    patched, edits, restored = GEN['patch_cuda'](cu, BITS)
    assert restored == cu and '__trap' not in patched
    text = once(base, '_HYB_MOE_CUDA=r"""' + cu + '"""', '_HYB_MOE_CUDA=r"""' + patched + '"""')
    pack = GEN['receipt'](7066, 7072) + '\n' + GEN['receipt'](7085, 7095)
    pack = pack.replace('def _sc6_', 'def _moexs_sc6_')
    names = ('_moex_pack_wgs', '_moex_weights', '_moex_scratch', '_moex_bind', '_moex_run',
             '_moex_error', '_moex_guards', '_moex_check', '_moex_graph_check')
    helpers = '\n'.join(function(GEN['GLUE'], name) for name in names)
    helpers = helpers.replace('_moex_', '_moexs_').replace('_MOEX_BITS', '_MOEXS_BITS')
    insertion = GEN['NOTICE'] + pack + GLUE + '\n' + helpers + '\n'
    text = once(text, 'def _hyb_build():\n', insertion + 'def _hyb_build():\n')
    text = once(text, ' for name,fn in jobs:\n',
                ' if HYB_ENABLE:\n  jobs += [("moexs",lambda: _moexs_build_one(False)),("moexs7",lambda: _moexs_build_one(True))]\n for name,fn in jobs:\n')
    text = once(text, ' for th in list(_PT1["threads"].values()):\n  th.join()\n',
                ' for name,th in list(_PT1["threads"].items()):\n  if name not in ("moexs","moexs7"):\n   th.join()\n')
    text = once(text, '[pt1] prefetch builds done:', '[pt1] bundle builds joined (MOEXS joins after guards):')
    # Copy the base HYBLO state and weight accessor verbatim. This hook is after
    # _gft_prepare / _rq_prepare, not inside the guard-sensitive K2A prepare.
    call = re.search(r'^  _hyblo_prepare\((.*)\)$', base, re.M)
    assert call and len(re.findall(r'^  _hyblo_prepare\(', base, re.M)) == 1
    hook = ' if state.get("k2a33")is not None:\n  _moexs_prepare(' + call[1] + ',t0)\n'
    text = once(text, ' _rq_prepare(state)\n', ' _rq_prepare(state)\n' + hook)
    anchor = 'def _hyb_call(x,ids,wts):\n m=x.shape[0]\n'
    text = once(text, anchor, anchor + ' H=_moexs_select(m)\n if H is not None:\n  return _moexs_run(H,x,ids,wts)\n')
    # Explicit lab requests fail outside the ship's catch-and-fallback block.
    anchor = ' if _hyb_on(x.shape[0]):\n'
    text = once(text, anchor, ' H=_moexs_select(x.shape[0])\n if H is not None:\n  return _moexs_run(H,x,local_ids.to(torch.int32).contiguous(),topk_weights.to(torch.float32).contiguous())\n' + anchor)
    compile(text, str(OUT), 'exec')
    assert len(text.encode()) < 990000 and '_MOEX_CUDA' not in text
    return text, edits


def source_gates(base, lab, edits):
    bt, lt = ast.parse(base), ast.parse(lab)
    ba = {n.targets[0].id:n for n in bt.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    la = {n.targets[0].id:n for n in lt.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    for name in ba:
        if name != '_HYB_MOE_CUDA':
            assert ast.dump(ba[name]) == ast.dump(la[name]), ('base assignment', name)
    bf = {n.name:n for n in bt.body if isinstance(n, ast.FunctionDef)}
    lf = {n.name:n for n in lt.body if isinstance(n, ast.FunctionDef)}
    changed = sorted(k for k in bf if ast.dump(bf[k]) != ast.dump(lf[k]))
    assert changed == ['_hyb_call', '_pt1_join_all', '_pt1_start', '_v11_k2a33_call', 'prepare'], changed
    for n in ast.walk(bt):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and any(k in n.value.lower() for k in ('apache','copyright','derived from')):
            assert n.value in [x.value for x in ast.walk(lt) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
    assert literal(lab, 'MOEXS') is False
    assert function(base, '_hyb_build') == function(lab, '_hyb_build')
    assert function(lab, 'prepare').index('_gft_prepare') < function(lab, 'prepare').index('_moexs_prepare')
    assert function(lab, 'prepare').index('_rq_prepare') < function(lab, 'prepare').index('_moexs_prepare')
    # Exactly reverse the three default-off grafts, including overlapping edits.
    restored = literal(lab, '_HYB_MOE_CUDA')
    for _, old, new in reversed(edits):
        restored = once(restored, new, old)
    defs = ''.join(f'#ifndef OKC_{b}\n#define OKC_{b} 0\n#endif\n' for b in BITS)
    restored = once(restored, defs + 'namespace okc {\n', 'namespace okc {\n')
    assert restored == literal(base, '_HYB_MOE_CUDA')
    off={f'OKC_{b}':0 for b in (*BITS,'YBF')}
    on=dict(off,OKC_SC6=1,OKC_MFIN=1,OKC_PLAN=1)
    assert feature_view(literal(lab,'_HYB_MOE_CUDA'),off) == feature_view(literal(base,'_HYB_MOE_CUDA'),off), 'off token drift'
    previous=(ROOT/'kernels/lab/k3_moex2_ab.py').read_text(encoding='utf-8')
    assert literal(base,'_HYB_MOE_CUDA') == literal(previous,'_HYB_MOE_CUDA')
    assert feature_view(literal(lab,'_HYB_MOE_CUDA'),on) == feature_view(literal(previous,'_MOEX_CUDA'),on), 'measured smp token drift'
    # Reuse tested SC6 arithmetic and protocol model against the actual candidate.
    view = lab.replace('_moexs_', '_moex_').replace('_HYB_MOE_CUDA', '_MOEX_CUDA')
    return {'status':'PASS', 'changed_base_functions':changed, 'exact_cuda_edits':len(edits),
            'reversible_cuda':True, 'off_tokens_identical':True, 'measured_smp_tokens_identical':True,
            'notices_preserved':True, 'base_sha256':sha(base), 'output_sha256':sha(lab),
            'base_bytes':len(base.encode()), 'output_bytes':len(lab.encode()),
            'cuda_growth_bytes_by_bit':{bit:sum(len(new.encode())-len(old.encode()) for b,old,new in edits if b==bit)
                                       +len(f'#ifndef OKC_{bit}\n#define OKC_{bit} 0\n#endif\n') for bit in BITS},
            'sc6':GATES['sc6_test'](view), 'protocol':GATES['protocol_test'](view)}


def dry_run(base, lab):
    """Execute new host paths with real base schema, allocators and numeric doubles."""
    Meta, DT = GATES['Meta'], GATES['DT']
    env = {'torch':GATES['meta_torch'](), 'hashlib':hashlib, 'time':time,
           '__builtins__':dict(vars(builtins))}
    constants = {'_HYB','_HYB_EMAX','_HYB_MMAX','_HYB_FIXF','_HYB_XROWS','_HYB_GUARD','_HYB_SLOTS',
                 '_hyb_scratch','_hyb_prmt_ok','HYB_T7','HYB_F7','HYB_FLAGS','HYB_SELF_TEST','HYB_PDL',
                 '_HYB_MOE_CPP','HYB_ENABLE','HYB_T'}
    exec(nodes(base, constants), env)
    names = {n.name for n in ast.parse(lab).body if isinstance(n, ast.FunctionDef) and n.name.startswith('_moexs_')}
    names |= {'MOEXS','_MOEXS','_MOEXS_BITS','_MOEXS_PREPARING','_HYB_MOE_CUDA','_hyb_call','_hyb_on','_v11_k2a33_call'}
    exec(nodes(lab, names), env)
    calls, launches, joins, inputs_seen = [], [], [], []
    fail_compile = [False]
    class Mod:
        def __init__(self, fl): self.flags, self.fail = fl, False
        def okc_set_pdl(self, pdl): assert pdl in (0,1)
        def okc_run(self, w, sc, x, ids, tw, out, lo, nloc, grid):
            assert len(w) == 14 and len(sc) == 3 and (lo,nloc,grid) == (0,32,132)
            assert ids.shape == tw.shape == (x.shape[0],8) and out.shape == x.shape
            out.tag = 'bad' if self.fail else None
            launches.append((self.flags, x.shape[0]))
    def compiler(**kw):
        assert kw['cuda_sources'] == literal(lab,'_HYB_MOE_CUDA')
        assert kw['cpp_sources'] == literal(base,'_HYB_MOE_CPP')
        assert kw['functions'] == ['okc_run','okc_set_pdl']
        calls.append(kw)
        if fail_compile[0]: raise RuntimeError('injected nvcc failure')
        return Mod(kw['extra_cuda_cflags'])
    imp = builtins.__import__
    env['__builtins__']['__import__'] = lambda name,*a,**kw: SimpleNamespace(load_inline=compiler) if name == 'torch.utils.cpp_extension' else imp(name,*a,**kw)
    class Th:
        def __init__(self, key): self.key, self.alive = key, False
        def join(self, timeout=None): joins.append((self.key, timeout))
        def is_alive(self): return self.alive
    env['_PT1'] = {'threads':{k:Th(k) for k in ('moexs','moexs7')}}
    def inputs(T, seed, dev, dist):
        inputs_seen.append((T,dist))
        return {'hidden_states':Meta((T,4096),DT.bfloat16),'topk_ids':Meta((T,8),DT.int32),
                'topk_weights':Meta((T,8)), 'config':{'local_start':0,'local_count':32}}
    env['_k2a_self_test_inputs'] = inputs
    env['_k2a_ref_fp32'] = lambda d,w: Meta(d['hidden_states'].shape)
    env['_k2a_rel_err'] = lambda a,b: (1.,1.) if a.tag == 'bad' else (0.,0.)
    ship = env['_HYB']
    ship.update(ok=True,nsm=132,mod=Mod([]),mod7=Mod(list(env['HYB_F7'])))
    # Execute the base's actual 14-entry weight-bank assignment, not a guessed list.
    hp = next(n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef) and n.name == '_hyb_prepare')
    bank = next(n for n in ast.walk(hp) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0]) == "_HYB['w']")
    w = {k:Meta((32,4096),DT.bfloat16) for k in ('w13','w13_gs','w2','w2_gs','post_norm_w')}
    w.update(w13_sf=Meta((32,8192,256),DT.uint8,64), w2_sf=Meta((32,4096,256),DT.uint8,126))
    env.update(w=w, ph=Meta((16,),DT.int32), lut=Meta((256,),DT.int32),
               wa=Meta((32*32*16*4*9216,),DT.uint8,64), wb=Meta((32*16*16*4*9216,),DT.uint8,64))
    exec(compile(ast.Module(body=[bank],type_ignores=[]),'<base HYB bank>','exec'),env)
    ship['sc'], ship['full'] = env['_hyb_scratch']('cpu')
    guards = env['_moexs_guards']
    # Metadata doubles cannot alias writes into huge guard allocations. Validate
    # actual extents here, and exercise the unchanged sentinel reader separately.
    env['_moexs_guards'] = lambda H: all(n == f.numel()-env['_HYB_GUARD'] for f,n,vt,sv in H['full'])
    captured = io.StringIO()
    def prep(weights=w, age=0):
        with contextlib.redirect_stdout(captured): env['_moexs_prepare']({},weights,'cpu',time.time()-age)
    for f in (False,True): env['_moexs_build_one'](f)
    assert len(calls) == 2
    prep()
    assert all(H['ok'] and len(H['checks']) == 4 for H in env['_MOEXS'].values())
    assert {(128,'normal'),(128,'skewed'),(256,'normal'),(256,'skewed')} <= set(inputs_seen)
    env.update(HYBLO_T=(), VMS_ENABLE=False)
    for switch in (False,True):
        env['MOEXS'] = switch
        for T in (1,8,16,32,49,127,128,192,193,256,320):
            selected = env['_moexs_select'](T)
            assert (selected is not None) == (switch and T in (128,256))
        for T in (128,256):
            d=inputs(T,0,'cpu','normal')
            env['_v11_k2a33_call'](d['hidden_states'],d['topk_ids'],d['topk_weights'],{})
            env['_hyb_call'](d['hidden_states'],d['topk_ids'],d['topk_weights'])
    env['_MOEXS'][False]['ok'] = False
    d=inputs(128,0,'cpu','normal')
    try: env['_v11_k2a33_call'](d['hidden_states'],d['topk_ids'],d['topk_weights'],{})
    except RuntimeError: pass
    else: raise AssertionError('explicit disabled arm was swallowed by fallback')
    for value in (63,127,128):
        w['w13_sf'].value=value; prep()
        assert all(not H['ok'] for H in env['_MOEXS'].values())
    assert len(calls)==2, 'range failure must not build a third codec'
    w['w13_sf'].value=64
    prep({})
    assert all(not H['ok'] and 'w13_sf' in H['err'] for H in env['_MOEXS'].values())
    prep(age=291)
    assert all(not H['ok'] for H in env['_MOEXS'].values())
    for H in env['_MOEXS'].values(): H['err']=None
    prep()
    env['_MOEXS'][True]['mod'].fail=True; prep()
    assert env['_MOEXS'][False]['ok'] and not env['_MOEXS'][True]['ok']
    env['_MOEXS'][True]['mod'].fail=False
    ship['mod7']=None; prep()
    assert not env['_MOEXS'][True]['ok']
    ship['ok']=False; prep()
    assert all(not H['ok'] for H in env['_MOEXS'].values())
    ship['ok']=True
    env['_PT1']['threads']['moexs'].alive=True; prep()
    assert not env['_MOEXS'][False]['ok']
    env['_PT1']['threads']={}; prep()
    assert all(not H['ok'] for H in env['_MOEXS'].values())
    fail_compile[0]=True
    env['_moexs_build_one'](False)
    assert env['_MOEXS'][False]['mod'] is None and not env['_MOEXS'][False]['ok']
    f=Meta((20,),DT.int32,0x5EADBEEF)
    assert guards({'full':[(f,10,DT.int32,0x5EADBEEF)]})
    f.value=1
    assert not guards({'full':[(f,10,DT.int32,0x5EADBEEF)]})
    return {'status':'PASS','initial_builds':2,'mock_launches':len(launches), 'checks_per_bank':4,
            'cases':sorted(set(inputs_seen)), 'schema':'real base _HYB, weight assignment and scratch allocator',
            'failure_paths':['build','scale range','missing key','missing ship mod7','ship disabled','budget','join timeout','PT1 disabled','numerical','guard','explicit selection'],
            'limitation':'metadata/numerical doubles, not GPU numerical or memory-order validation'}


def integration_dry_run(base, lab):
    """Run the real base state constructors and candidate prepare call chain."""
    diet = runpy.run_path(str(ROOT / 'tools/diet_patches.py'))
    env, calls = diet['cpu_namespace'](lab)
    env['_lt_flags'] = lambda:['-lcublasLt']
    # Every real PT1 compiler path is executed with the original recorder.
    with contextlib.redirect_stdout(io.StringIO()):
        env['_pt1_start'](); env['_pt1_join_all']()
    extras=[c for c in calls if c['name'].startswith('hyb_moexs_')]
    assert len(extras)==2
    for c in extras: assert c['cuda_sources']==literal(lab,'_HYB_MOE_CUDA')
    ft, fi = diet['FakeTensor'], diet['FakeImport']
    env['_norm']=lambda *a,**kw:fi('norm')
    env['_residual']=lambda *a,**kw:fi('residual')
    env['_marlin_bank']=lambda *a:('cpu mocked marlin allocation',)
    env['_local_id_table']=lambda *a:ft((256,),'i32')
    weights={k:ft((32,4096),'bf16') for k in env['Weights'].__annotations__}
    for k,dims in {'qkv_gate':(2304,8192),'o':(8192,1024),'pooler':(4096,8192),'unpooler':(8192,4096),'router':(256,8192)}.items():
        weights[k]=ft(dims,'f32' if k=='router' else 'bf16')
    env['_k2a33_build_ext']=lambda *a:(SimpleNamespace(attrs3v=lambda i:[0,0,0,2,0,0,0,2]),0)
    mw={k:weights[k] for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs')}
    mw['post_norm_w']=weights['post_expert_norm']
    with contextlib.redirect_stdout(io.StringIO()):
        inner=env['_k2a33_prepare_st'](mw,share={'ws13':ft(),'ws2':ft()})
    env['_v11_k2a33_prepare']=lambda *a:{'st':inner,'ok':{r:True for _,_,r in env['K2A_BUCKETS']}}
    seen=[]
    def hook(st,w,dev,t0):
        assert st is inner and w==mw
        seen.append('MOEXS real nested state')
    env['_moexs_prepare']=hook
    tree=ast.parse(function(lab,'prepare'))
    leaves={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and 'prepare' in n.func.id}
    for name in leaves-{'_r1_prepare','_v11_k2a33_prepare','_moexs_prepare'}:
        env[name]=lambda *a,_n=name,**kw:seen.append(_n)
    env['_pt1_start']=lambda:None
    env['_pt1_join_all']=lambda:None
    env['_v10_build_all']=lambda:None
    with contextlib.redirect_stdout(io.StringIO()): state=env['prepare'](weights)
    assert state['k2a33']['st'] is inner
    assert seen.count('MOEXS real nested state')==1
    assert seen.index('_gft_prepare') < seen.index('MOEXS real nested state')
    # The copied hook must fail loudly if supplied the historical wrong layout.
    hooknode=next(n for n in tree.body[0].body if isinstance(n,ast.If) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='_moexs_prepare' for x in ast.walk(n)))
    try: exec(compile(ast.Module(body=[hooknode],type_ignores=[]),'<hook>','exec'),env,{'state':{'k2a33':{}},'weights':weights,'t0':time.time()})
    except KeyError as ex: assert ex.args==('st',)
    else: raise AssertionError('nested state typo not caught')
    return {'status':'PASS','state_keys':sorted(state),'k2a33_inner_keys':sorted(inner),
            'extra_pt1_builds':len(extras),'guarded_prepares_before_hook':True,'missing_nested_key_detected':True}


def linux_sass(base, lab):
    """Offline NVCC, six builds. No CUDA module load or kernel launch."""
    assert sys.platform.startswith('linux'), 'Lead must use the Linux cudaenv.sh entry'
    import shutil
    import sysconfig
    import torch
    import torch.utils.cpp_extension as ce
    assert shutil.which('nvcc') and shutil.which('cuobjdump')
    work=Path(tempfile.mkdtemp(prefix='moexs-gates-'))
    includes=[v for p in ce.include_paths(device_type='cuda') for v in ('-I',p)] + ['-I',sysconfig.get_paths()['include']]
    common=['-std=c++20','-DTORCH_EXTENSION_NAME=moexs_gate','-DTORCH_API_INCLUDE_EXTENSION_H',
            '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
            '-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__','--expt-relaxed-constexpr']
    prelude='#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n'
    results,images={},{}
    for f7 in (False,True):
        fl=list(literal(base,'HYB_FLAGS'))+(list(literal(base,'HYB_F7')) if f7 else [])
        for arm,source in (('base',base),('off',lab),('smp',lab)):
            key=f'{arm}{int(f7)}'
            flags=fl+(['-DOKC_SC6=1','-DOKC_MFIN=1','-DOKC_PLAN=1'] if arm=='smp' else [])
            cu, cubin = work/'cuda.cu', work/(key+'.cubin')
            cu.write_bytes((prelude+literal(source,'_HYB_MOE_CUDA')).encode())
            t0=time.time()
            run=subprocess.run(['nvcc','-cubin',*common,*includes,*flags,'-Xptxas=-v',str(cu),'-o',str(cubin)],capture_output=True,text=True)
            row={'status':'FAIL','nvcc_s':time.time()-t0,'flags':flags,'ptxas':GATES['ptxas_entries'](run.stdout+run.stderr)}
            results[key]=row
            (work/(key+'.log')).write_bytes((run.stdout+run.stderr).encode())
            if run.returncode:
                row['error']=run.stderr[-4000:]; continue
            res=subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],capture_output=True,text=True,check=True).stdout
            sass=subprocess.run(['cuobjdump','--dump-sass',str(cubin)],capture_output=True,text=True,check=True).stdout
            (work/(key+'.sass')).write_bytes(sass.encode())
            images[key]=GATES['sass_image'](sass)
            row.update(status='PASS',resources=res,sass_sha256=sha(json.dumps(images[key],sort_keys=True)))
            entries={k:v for k,v in row['ptxas'].items() if 'moe_fused' in k}
            assert len(entries)==2 and all({'registers','stack','spill_stores','spill_loads'} <= set(v) for v in entries.values())
            print(f'[moexs compile] {key} {row["nvcc_s"]:.2f}s {entries}',flush=True)
    equal={str(f):'off'+str(f) in images and images['off'+str(f)]==images.get('base'+str(f)) for f in (0,1)}
    # Obtain dynamic shared size from the exact launch expression as a device
    # constant; this only compiles PTX and does not run a kernel.
    smem={}
    for f7 in (False,True):
        cu=literal(lab,'_HYB_MOE_CUDA')
        expr=GEN['between'](cu,'static size_t okc_smem_bytes() {','static int g_okc_pdl')
        expr=expr.split('return base + ',1)[1].split(';',1)[0]
        constant='extern "C" { __device__ unsigned long long moexs_smem = ((sizeof(okc::Smem)+127)/128)*128 + '+expr+'; }\n'
        p=work/'smem.cu'; out=work/('smem'+str(int(f7))+'.ptx')
        p.write_bytes((prelude+cu+constant).encode())
        fl=list(literal(base,'HYB_FLAGS'))+(list(literal(base,'HYB_F7')) if f7 else [])+['-DOKC_SC6=1','-DOKC_MFIN=1','-DOKC_PLAN=1']
        run=subprocess.run(['nvcc','-ptx',*common,*includes,*fl,str(p),'-o',str(out)],capture_output=True,text=True)
        assert run.returncode==0,run.stderr[-4000:]
        m=re.search(r'\.u64 moexs_smem = (\d+)',out.read_text())
        assert m,'dynamic shared constant missing'
        smem[str(int(f7))]=int(m[1])
    return {'status':'PASS' if all(equal.values()) and all(x['status']=='PASS' for x in results.values()) else 'FAIL',
            'builds':results,'off_sass_equal':equal,'dynamic_smem_bytes':smem,'artifacts':str(work),
            'added_cold_prepare_nvcc_s':'Not measured: standalone compiler durations exclude parallel contention/link/self-tests'}


def changed_lines(base, lab, edits):
    current=literal(base,'_HYB_MOE_CUDA')
    defs=''.join(f'#ifndef OKC_{b}\n#define OKC_{b} 0\n#endif\n' for b in BITS)
    current=once(current,'namespace okc {\n',defs+'namespace okc {\n')
    labels=[None]*len(current.splitlines())
    dstart=current[:current.index(defs)].count('\n')
    for i,bit in enumerate(BITS): labels[dstart+i*3:dstart+(i+1)*3]=[bit]*3
    for bit,old,new in edits:
        start=current[:current.index(old)].count('\n')
        a,b=old.splitlines(),new.splitlines()
        prior=labels[start:start+len(a)]; out=[]
        for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
            out.extend(prior[i:j] if tag=='equal' else [bit]*(l-k))
        labels[start:start+len(a)]=out
        current=once(current,old,new)
    offset=next(n.lineno for n in ast.parse(lab).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_HYB_MOE_CUDA' for t in n.targets))
    result={}
    for bit in BITS:
        spans=[]
        for i,label in enumerate(labels):
            if label!=bit: continue
            n=offset+i
            if spans and n==spans[-1][1]+1: spans[-1][1]=n
            else: spans.append([n,n])
        result[bit]=spans
    result['python']={n.name:[n.lineno,n.end_lineno] for n in ast.parse(lab).body if isinstance(n,ast.FunctionDef) and (n.name.startswith('_moexs_') or n.name in ('prepare','_hyb_call','_v11_k2a33_call','_pt1_start','_pt1_join_all'))}
    result['all_diff_hunks']=[{'base':[i+1,j],'candidate':[k+1,l]} for tag,i,j,k,l in
                              difflib.SequenceMatcher(None,base.splitlines(),lab.splitlines(),autojunk=False).get_opcodes() if tag!='equal']
    return result


def report(base,lab,edits,gates):
    def loc(needle):
        assert lab.count(needle)==1,needle
        return lab[:lab.index(needle)].count('\n')+1
    def culoc(needle):
        cu=literal(lab,'_HYB_MOE_CUDA')
        assert cu.count(needle)==1,needle
        return loc('_HYB_MOE_CUDA=r"""')+cu[:cu.index(needle)].count('\n')
    route=f'`kernels/lab/k3_moexs_ab.py:{loc("def _v11_k2a33_call(")}`'
    vmsline=loc(' md=vs["mod_hi"]')
    vmlline=loc("VML_VAR='bf2'")
    comp=gates['compile']
    compile_status=comp['status']
    lines=[
        '| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |',
        '|---|---|---|---|---|---|---|',
        f'| MOEXS smp | T=128 s/f; T=256 control | combo8: v16f ship 326.9, v32f ship 472.4; deltas v16s -2.0 / v16f -1.8; v32 ~0 (notes/reset/LOG.md:73-74) | No new net estimate; SC6 traffic ceiling = 786432 B/job / 4.64e6 B/us = 0.1695 us/job; >=0.5 at 3 exposed jobs | Stream bytes decrease; MFIN permits token completion before unrelated experts. Only combo8 demonstrates exposure; one seed, no independent attribution | R-b4m-0; R-hybst-2 FIN; R-hybtab-1 | Implemented requested measured rider; CPU PASS, compile {compile_status}, H200 pending |',
        '',
        'Kill reasons first: (1) byte savings may be hidden and planner end spread may cancel them; (2) F7 register pressure / completion polls may erase the gain. No new performance promotion. T=256 is a requested control, not a claimed win.',
        '',
        f'MEASURED file sizes: {len(base.encode()):,} -> {len(lab.encode()):,} B; {990000-len(lab.encode()):,} B below the requested 990,000 B ceiling. One HYB CUDA literal; original HYB builder and flags unchanged, plus exactly two smp PT1 builds. `MOEXS=False` by default. `True` selects smp only at T=128/256 before call/capture; existing captures retain their arm. Disabled explicit arms raise before the HYB fallback handler.',
        'MEASURED source growth by bit: SC6 +1,387 B; MFIN +3,818 B; PLAN +4,482 B (9,687 B CUDA total). EST packed weights for all32 experts: 32*(32+16)*16*4*9216=905,969,664 B -> same factors*8960=880,803,840 B, saving25,165,824 B. Both candidate banks share this packed weight list but own separate scratch.',
        '',
        f'Exact scored MoE routes (both sliding/full use {route}): T=1/8 are unchanged HYBLO tiny/tiny2 at `kernels/lab/k3_moexs_ab.py:{loc("_HYBLO_KEYS=")}`; T=16 uses VMS mod_hi at `kernels/lab/k3_moexs_ab.py:{vmsline}`; T=32 uses VML bf2 (`kernels/lab/k3_moexs_ab.py:{vmlline}`, dispatcher `{loc("def _vml_route(")}`). T=128 selects mod; T=256 selects mod7 via HYB_T7=193, `kernels/lab/k3_moexs_ab.py:{loc("def _hyb_call(")}`. MOEXS only replaces these last two with `_MOEXS[False/True]` at `{loc("def _moexs_select(")}`. Both launch `okc::moe_fused<true>` at `{culoc("if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);")}`, 576 threads/CTA and one CTA/SM; NTT=ceil(jC/8) selects `wgs_seg<PB,1..8>` at `{culoc("case 1: wgs_seg<PB, 1>")}` (NTT depends on routing, not T alone). Exact flags: `{loc("HYB_FLAGS=")}`, F7 adds HYBST=7/HN=1 at `{loc("HYB_F7=")}`, smp adds SC6/MFIN/PLAN=1 at `{loc("def _moexs_flags(")}`; all retain PADZ=1/FT_T=64.',
        '',
        'Prepare: new PT1 jobs start after all existing jobs; the old join skips only moexs/moexs7. Their bounded joins and validation occur after _sk2_prepare/_gft_prepare/_rq_prepare. All bundle guards are unchanged; 280 s join deadline and 290 s validation checkpoints disable incomplete banks. Added cold nvcc/prepare time is NOT MEASURED here; each bank records build_s and prints it. Two extra compiler jobs can still contend for CPU, so actual <300 s and unchanged guard outcomes require a cold H200 prepare.',
        '',
        'Self-tests: two calls per normal/skewed case at each bank\'s T=128/256, versus matching ship and FP32, plus alternating normal/skewed graph replays. Same MOEX tolerances (FP32 total<1e-2, row<5e-2; ship rel<=max(2*repeat,1e-3); repeat<1e-3), finite, guard and sticky device-error gates. SC6/PRMT range failure disables both banks; it never adds a fallback codec build. Real base state accessor is copied from HYBLO: state["k2a33"]["st"].',
        '',
        '| changed loop | EST bytes before -> after | EST source operations per participating thread before -> after (not SASS) |',
        '|---|---|---|',
        '| SC6 stage weight stream | 36,864 -> 35,840 B; physical slot remains 36,864 B; x-ring unchanged | producer leader: 1 expect_tx + 1 bulk copy -> same; consumers: scale loads LDS64 -> LDS32+LDS16, scale-pair expressions 16 -> 4 plus 5 AND/1 OR/2 multiplies; fused algebraic issue count 33 -> 18, before compiler CSE |',
        '| MFIN handoffs | no added weight/activation stream; flag addresses move to parity banks | existing acquire/release retained; volatile shared gen load and parity arithmetic added at each evaluation |',
        '| MFIN B unit/WG | 4 B release/WG = 16 B/unit, 1 B/stage amortized; SC6+smp combined 36,864 -> 35,841 B/stage plus readiness polling | each publishing WG adds 1 bar.sync; wt0 adds 1 red.release. Whole unit pending record adds gen load/address arithmetic. Split non-resolvers do not publish |',
        '| MFIN launch/finalize | +16,384 B persistent flags; other-parity clear 6,144 B replaces 12,288 B ss clear + 6,344 B old cleanup; +4 B atomic/CTA, 4 B per poll | tid0 old fence+barrier atomic+poll -> atomicInc; tid0..7 polls/token; +2 CTA barriers/token. Poll iterations unbounded in model, bounded 4,000,000 in code; no removed launch |',
        '| PLAN prefix/ranges | same 2*(J+1) shared words/totals/4 range words; stream payload unchanged | lane0 serial costs/prefix/4 searches -> all32 helper lanes, 10 shuffle-up + <=10 conditional adds +2 carry shuffles per 32 jobs/lane; 2 warp syncs; 1 boundary search/lane, lanes0..3 store. U=160/140 arithmetic added |',
        '',
        'Instruction numbers are EST source arithmetic from box_out/reset/moex_report.md, not measured speed. Stream intercept 1.9 us is not removed. MFIN removes one grid rendezvous (MEASURED 1.09 us, DO_NOT_REPEAT R-pdle-0), but adds per-expert polling/atomics; neither 0.84 us PDL nor 0.80 us flag hops are counted again. No aggregate instruction delta is claimed without SASS.',
        '',
        '| protocol cleanup/combine | CTA/thread ownership and coverage |',
        '|---|---|',
        '| generation and other-parity reset | every CTA tid0 increments MF_EXIT modulo 2G; all compute tids0..511 clear i=CTA*512+tid+k*G*512 for i<1536 (32 CE +96 XR +88*16 DA). Before phases; no compute CTA exits, even CTA>=T |',
        '| producers / zero jobs | tids512..543 weight, 544..575 x retain their original returns; neither owns output/reset. All32 x helper lanes run PLAN. nj==0 returns only from wgs_ranges with zero ranges; compute CTAs still reset/combine |',
        '| split combine and fc reset | resolving WG tids128*wg..128*wg+127 covers nt*512+wt*4..+3 partial elements; wt0 resets fc. Existing wgs_flush AND wgs_flush_end loops preserved. Kind2 nonresolver exits only that function; kind4 whole unit has no partial buffer |',
        '| completion / removed cleanup | each B job: 16 units*4 WG releases=64; whole owner or resolver only, barrier then release. Old ctr[0:2], doneA[88], xready[88], doneAs[1408] post-grid reset removed; live XR/DA use opposite-parity preclear; ctr/doneA/ss are unused by this WGS specialization |',
        '| finalizer / timeout | CTA tk%G owns tk; tids0..7 acquire its routes, all512 sync before y reads; each tid covers8/4096 columns. Collective timeout writes NaN, sticky error and continues; no partial CTA return or skipped cleanup. Same-stream scratch reuse only, as ship |',
        '',
        f'MEASURED prior MOEX compiler evidence (box_out/reset/moex_gate.log): smp mod=96 regs, stack/spill stores/loads=0/0/0 B; smp mod7=96 regs, 8/8/8 B. Both static shared=1024 B. These are prior-lab builds, NOT measurements of this new source. Current-build Linux gate: {compile_status}; current resource results, if available, are in the JSON below. The gate reports spills rather than assuming F7 is spill-free.',
        'CPU feature-fold gate: off tokens equal the current bundle and on tokens equal measured k3_moex2_ab smp with YBF=0, including HN/PADZ/F7. This rules out a functional port drift at the token level; it does not assert compiler-output identity. The Linux gate compares every raw SASS instruction/control word for both off builds.',
        '',
        'Exact candidate lines (one-based; source operations only, not all surrounding unchanged lines):',
        '```json',json.dumps(changed_lines(base,lab,edits),indent=2),'```',
        '',
        'Local CPU gates execute exact anchors/reversal, AST invariants, preserved notices, SC6 65,536-byte reconstruction and unchanged codes, 96 planner cases/parity/cleanup models, all new host paths/failure arms, actual PT1 builders, real base K2A state constructors, and copied prepare hook. Tensor doubles do not validate CUDA numerics or memory ordering.',
        '',
        f'NVCC/SASS/new resource usage: {compile_status}. Not run: H200 numerical/graph gates, cold prepare budget, paired performance. No GPU/pod/judge jobs or git writes were made. The three-file allowlist prevents a fourth .sh/.json: the generator doubles as LF bash script and JSON is retained in this authorized report and stdout. Linux compiler intermediates are retained under /tmp by the lead invocation.',
        '',
        '```bash',
        'MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" bash "<repo>/tools/moexs_patches.py"',
        '```','',
        '```json',json.dumps(gates,indent=2),'```',
    ]
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--sass',action='store_true')
    args=p.parse_args()
    base=BASE.read_text(encoding='utf-8')
    lab,edits=generate(base)
    gates={'source':source_gates(base,lab,edits), 'dry_run':dry_run(base,lab), 'integration':integration_dry_run(base,lab),
           'compile':{'status':'NOT RUN','reason':'Linux toolchain required; no WSL probe/GPU job'}}
    if args.check: assert OUT.read_bytes()==lab.encode(), 'generated output is stale'
    else: OUT.write_bytes(lab.encode())
    if args.sass:
        try: gates['compile']=linux_sass(base,lab)
        except Exception as ex: gates['compile']={'status':'FAIL','error':repr(ex)}
    REPORT.write_bytes(report(base,lab,edits,gates).encode())
    print(json.dumps(gates,indent=2))
    if gates['compile']['status']=='FAIL': raise SystemExit(1)


if __name__=='__main__':
    main()

#!/bin/bash
""":"
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
exec python3 -B "$0" --check --sass "$@"
: <<'# MOEXR_PYTHON'
": """
"""MOEXR: replace the bundle HYB builds, with a lab-only stock bank.

python -B tools/moexr_patches.py          # generate both files, CPU gates, report
python -B tools/moexr_patches.py --check  # deterministic regeneration + CPU gates
bash tools/moexr_patches.py              # offline Linux compile/SASS + CPU gates
Run the LF bash entry through tools/local/cudaenv.sh. No GPU is used.
Only the four task-named files are written. Gate JSON lives in the report;
the Linux runner retains compiler intermediates in /tmp (never deletes them).
"""
import argparse
import ast
import builtins
import contextlib
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/lab/k3_diet_rq128.py'
REFERENCE = ROOT / 'kernels/lab/k3_moexs_ab.py'
LAB = ROOT / 'kernels/lab/k3_moexr_ab.py'
SHIP = ROOT / 'kernels/muse_decode_layer/k3_moexr_c1.py'
REPORT = ROOT / 'box_out/reset/moexr_report.md'
S = runpy.run_path(str(ROOT / 'tools/moexs_patches.py'))
G, C = S['GEN'], S['GATES']
once, literal, function, nodes = G['once'], G['literal'], S['function'], C['nodes']
BITS = ('SC6', 'MFIN', 'PLAN')
DEFINES = ['-DOKC_SC6=1', '-DOKC_MFIN=1', '-DOKC_PLAN=1']


def replace_function(source, name, replacement):
    return once(source, function(source, name), replacement)


def generate(base, reference, ship):
    cu, edits, restored = G['patch_cuda'](literal(base, '_HYB_MOE_CUDA'), BITS)
    assert restored == literal(base, '_HYB_MOE_CUDA')
    assert cu == literal(reference, '_HYB_MOE_CUDA'), 'measured smp source drift'
    text = once(base, '_HYB_MOE_CUDA=r"""' + restored + '"""', '_HYB_MOE_CUDA=r"""' + cu + '"""')
    helpers = ''.join(function(reference, n) for n in
                      ('_moexs_sc6_ok', '_moexs_sc6_wgs', '_moexs_pack_wgs', '_moexs_scratch'))
    helpers = helpers.replace('_moexs_', '_moexr_')
    helpers += 'MOEXR = ' + str(ship) + '\n'
    if not ship:
        helpers += '_MOEXR_SHIP = ' + repr(literal(base, '_HYB')) + '\n'
    helpers += 'def _moexr_bank():\n return _HYB if MOEXR else ' + ('None' if ship else '_MOEXR_SHIP') + '\n'
    # These are the base call and guard bodies, parameterized like the padz bank.
    call = function(base, '_hyb_call').replace('def _hyb_call(x,ids,wts):', 'def _moexr_call_bank(H,x,ids,wts):')
    helpers += re.sub(r'\b_HYB\b', 'H', call)
    guards = function(base, '_hyb_guards_ok').replace('def _hyb_guards_ok():', 'def _moexr_guards(H):')
    helpers += re.sub(r'\b_HYB\b', 'H', guards)
    # Keep the same two-way compilation topology; failed attempts are not retried.
    # Every started thread is joined even if the ordinary module fails.
    build = '''def _moexr_build_bank(H@ARG@):
 if H.get("attempted"):
  return H["mod"]
 from torch.utils.cpp_extension import load_inline
 H["attempted"],H["ok"],H["err"]=True,False,None
 def one(f7):
  fl=list(HYB_FLAGS)+(list(HYB_F7)if f7 else[])@FLAGS@
  tag=hashlib.sha1((_HYB_MOE_CPP+_HYB_MOE_CUDA+" ".join(fl)).encode()).hexdigest()[:12]
  return load_inline(name=f"hyb_moe_{tag}",cpp_sources=_HYB_MOE_CPP,cuda_sources=_HYB_MOE_CUDA,
  functions=["okc_run","okc_set_pdl"],extra_cuda_cflags=fl,verbose=False)
 def b7():
  try:
   H["mod7"]=one(True)
  except Exception as ex:
   H["mod7"]=None
   H["err"]=repr(ex)[:500]
 t0=time.time()
 th=threading.Thread(target=b7,name="hyb7",daemon=True)
 th.start()
 try:
  H["mod"]=one(False)
 except Exception as ex:
  H["mod"]=None
  H["err"]=repr(ex)[:500]
 finally:
  th.join()
  H["build_s"]=time.time()-t0
 return H["mod"]
'''
    build = build.replace('@ARG@', '' if ship else ',smp')
    flags = '+' + repr(DEFINES)
    build = build.replace('@FLAGS@', flags if ship else '+(' + repr(DEFINES) + ' if smp else [])')
    helpers += build
    old_prep = function(base, '_hyb_prepare')
    prep = once(old_prep, 'def _hyb_prepare(st,w,dev):', 'def _moexr_prepare_bank(st,w,dev,H' + ('' if ship else ',smp') + '):')
    prep = re.sub(r'\b_HYB\b', 'H', prep)
    prep = once(prep, '   _hyb_build()\n', '   _moexr_build_bank(H' + ('' if ship else ',smp') + ')\n')
    prep = once(prep, '  nsm=torch.cuda.get_device_properties(dev).multi_processor_count\n',
                '  if H["mod"]is None' + (' or H["mod7"]is None' if ship else ' or (smp and H["mod7"]is None)') + ':\n   return\n'
                '  nsm=torch.cuda.get_device_properties(dev).multi_processor_count\n')
    check = '_moexr_sc6_ok(w["w13_sf"],w["w2_sf"])'
    prep = once(prep, '  I=H=4096\n', '  if ' + ('' if ship else 'smp and ') + 'not ' + check + ':\n   H["err"]="SC6 scale range"\n   return\n  I=HDIM=4096\n')
    # H was the base hidden dimension local; use HDIM now that H is the bank.
    prep = once(prep, 'torch.arange(H,device=dev)', 'torch.arange(HDIM,device=dev)')
    anchor = '  lut=(torch.arange(256,dtype=torch.int32,device=dev)'
    packing = '  wa,wb=_moexr_pack_wgs(wa),_moexr_pack_wgs(wb)\n'
    if not ship:
        packing = '  if smp:\n ' + packing
    prep = once(prep, anchor, packing + anchor)
    prep = once(prep, '  H["sc"],H["full"]=_hyb_scratch(dev)\n',
                '  H["sc"],H["full"]=_moexr_scratch(dev,' + ('True' if ship else 'smp') + ')\n')
    # Both normal and F7 tests run directly on the bank being prepared; MOEXR
    # never redirects prepare tests to the lab control.
    assert prep.count('_hyb_call(') == 2
    prep = prep.replace('_hyb_call(', '_moexr_call_bank(H,')
    prep = once(prep, '  gok=_hyb_guards_ok()\n', '  gok=_moexr_guards(H)\n')
    error = 'int(H["sc"][0][8+1024+_HYB_EMAX+65])==0'
    prep = once(prep, '  H["ok"]=bool(good and gok)\n',
                '  H["ok"]=bool(good and gok' + (' and ' if ship else ' and (not smp or (') +
                'H["mod7"]is not None and ' + error + (')\n' if ship else ')))\n'))
    helpers += prep
    helpers += 'def _hyb_prepare(st,w,dev):\n _moexr_prepare_bank(st,w,dev,_HYB' + ('' if ship else ',True') + ')\n'
    if not ship:
        helpers += 'def _moexr_ship_prepare(st,w,dev):\n _moexr_prepare_bank(st,w,dev,_MOEXR_SHIP,False)\n'
    text = replace_function(text, '_hyb_build',
                            G['NOTICE'] + helpers + 'def _hyb_build():\n return _moexr_build_bank(_HYB' + ('' if ship else ',True') + ')\n')
    # Remove the original prepare, now copied into the helper above.
    text = once(text, old_prep, '')
    text = replace_function(text, '_hyb_call', 'def _hyb_call(x,ids,wts):\n return _moexr_call_bank(_moexr_bank(),x,ids,wts)\n')
    text = replace_function(text, '_hyb_on', 'def _hyb_on(m):\n H=_moexr_bank()\n return HYB_ENABLE and H is not None and H["ok"]and HYB_T[0]<=m<=HYB_T[1]\n')
    if not ship:
        # Lab control compilation is deliberately AFTER all guard-timed prepares.
        # Exact nested accessor copied from the proven base HYBLO/padz pipeline.
        calls = re.findall(r'^  _hyblo_prepare\((.*)\)$', base, re.M)
        assert len(calls) == 1
        text = once(text, ' _rq_prepare(state)\n', ' _rq_prepare(state)\n if state.get("k2a33")is not None:\n  _moexr_ship_prepare(' + calls[0] + ')\n')
    nonce = next(line for line in base.splitlines() if line.startswith('# nonce '))
    text = once(text, nonce + '\n', '# nonce k3-moexr-' + ('c1' if ship else 'ab') + ' (bundle + sole smp HYB; tools/moexr_patches.py)\n')
    compile(text, '<moexr>', 'exec')
    assert len(text.encode()) < 1000000
    return text, edits


def source_gates(base, reference, lab, ship, edits):
    off = {f'OKC_{b}': 0 for b in (*BITS, 'YBF')}
    on = dict(off, OKC_SC6=1, OKC_MFIN=1, OKC_PLAN=1)
    changed = {}
    for label, source in (('lab', lab), ('ship', ship)):
        cu = literal(source, '_HYB_MOE_CUDA')
        assert cu == literal(reference, '_HYB_MOE_CUDA')
        assert S['feature_view'](cu, off) == S['feature_view'](literal(base, '_HYB_MOE_CUDA'), off)
        assert S['feature_view'](cu, on) == S['feature_view'](literal(reference, '_HYB_MOE_CUDA'), on)
        bt, nt = ast.parse(base), ast.parse(source)
        bf = {n.name: ast.dump(n) for n in bt.body if isinstance(n, ast.FunctionDef)}
        nf = {n.name: ast.dump(n) for n in nt.body if isinstance(n, ast.FunctionDef)}
        changed[label] = sorted(n for n in bf if bf[n] != nf[n])
        assert changed[label] == sorted(['_hyb_build', '_hyb_prepare', '_hyb_call', '_hyb_on'] + ([] if label == 'ship' else ['prepare']))
        assignments = {ast.dump(x.targets[0]):ast.dump(x) for x in nt.body if isinstance(x,ast.Assign)}
        for n in bt.body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id != '_HYB_MOE_CUDA':
                assert assignments.get(ast.dump(n.targets[0])) == ast.dump(n), n.targets[0].id
        assert function(base, '_v11_k2a33_call') == function(source, '_v11_k2a33_call')
        assert function(base, '_pt1_start') == function(source, '_pt1_start')
        assert function(base, '_pt1_join_all') == function(source, '_pt1_join_all')
        constants = [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        for n in ast.walk(ast.parse(base)):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and any(k in n.value.lower() for k in ('apache','copyright','derived from')):
                assert n.value in constants
    assert literal(ship, 'MOEXR') is True and literal(lab, 'MOEXR') is False
    assert '_MOEXR_SHIP' not in ship and '_moexr_ship_prepare' not in ship
    for name in ('_moexr_sc6_ok', '_moexr_sc6_wgs', '_moexr_pack_wgs', '_moexr_scratch'):
        assert function(ship, name).replace('_moexr_', '_moexs_') == function(reference, name.replace('_moexr_', '_moexs_'))
    view = ship.replace('_moexr_', '_moex_').replace('_HYB_MOE_CUDA', '_MOEX_CUDA')
    return {'status': 'PASS', 'changed_base_functions': changed, 'cuda_exact_moexs': True,
            'off_tokens_exact_bundle': True, 'notices_preserved': True,
            'sc6': C['sc6_test'](view), 'protocol': C['protocol_test'](view),
            'sizes': {n: len(s.encode()) for n,s in (('base',base),('moexs',reference),('lab',lab),('ship',ship))}}


def build_gates(base, reference, lab, ship):
    diet = runpy.run_path(str(ROOT / 'tools/diet_patches.py'))
    captures = {}
    for label, source in (('base',base),('moexs',reference),('lab',lab),('ship',ship)):
        env, calls = diet['capture_builds'](source)
        # Lab-only ship modules are lazy at the post-guard hook.
        if label == 'lab':
            # capture_builds converted the original recorder into a dict; use a
            # fresh full-module recorder to include the lazy bank.
            env, raw = diet['cpu_namespace'](source)
            env['_lt_flags'] = lambda: ['-lcublasLt']
            with contextlib.redirect_stdout(io.StringIO()):
                env['_pt1_start'](); env['_pt1_join_all']()
                env['_moexr_build_bank'](env['_MOEXR_SHIP'], False)
            calls = {c['name']: c for c in raw}
            assert len(calls) == len(raw)
        captures[label] = calls
    base_other = {n:c for n,c in captures['base'].items() if not n.startswith('hyb_moe_')}
    for label in ('lab','ship'):
        other = {n:c for n,c in captures[label].items() if not n.startswith('hyb_moe_')}
        assert other == base_other, ('non-HYB compiler input drift', label)
    expected = {}
    for f7 in (False,True):
        flags = list(literal(base, 'HYB_FLAGS')) + (list(literal(base, 'HYB_F7')) if f7 else []) + DEFINES
        expected[tuple(flags)] = True
    for label in ('lab','ship'):
        hyb = [c for n,c in captures[label].items() if n.startswith('hyb_moe_')]
        smp = [c for c in hyb if '-DOKC_SC6=1' in c['extra_cuda_cflags']]
        assert {tuple(c['extra_cuda_cflags']) for c in smp} == set(expected)
        assert len(hyb) == (4 if label == 'lab' else 2)
        assert all(c['cuda_sources'] == literal(reference, '_HYB_MOE_CUDA') for c in hyb)
    assert len(captures['ship']) == len(captures['base'])
    assert len(captures['moexs']) == len(captures['lab']) == len(captures['base']) + 2
    return {'status':'PASS', 'recorded_load_inline_calls':{n:len(v) for n,v in captures.items()},
            'HYB_builds':{'base':2,'moexs':4,'lab':4,'ship':2},
            'ship_non_HYB_inputs_exact':True, 'ship_flags_exact_moexs_smp':True,
            'lab_extra_build_placement':'after _rq_prepare; no added PT1 jobs'}


def dry_run(base, source, ship):
    """Run actual prepare/packing/selector bodies using strict tensor metadata.

    Numerical double only at CUDA/ref leaves; dictionaries, packing, tests,
    counters, guards, failure arms and launch selection execute real source.
    """
    Meta, DT = C['Meta'], C['DT']
    torch = C['meta_torch']()
    torch.float8_e4m3fn = C['DType']('float8_e4m3fn',1)
    torch.arange = lambda n,device=None,dtype=DT.int32: Meta((n,),dtype)
    torch.cuda.get_device_properties = lambda dev: SimpleNamespace(multi_processor_count=132)
    # Base row construction uses broadcasting/None indexing; do it with a tiny
    # numpy-backed index tensor, while all large expert arrays stay metadata.
    import numpy as np
    class Index:
        def __init__(self,a): self.a=np.asarray(a)
        def __getitem__(self,k): return Index(self.a[k])
        def __rmul__(self,v): return Index(v*self.a)
        def __add__(self,v): return Index(self.a+(v.a if isinstance(v,Index) else v))
        __radd__=__add__
        def reshape(self,*s): return Index(self.a.reshape(*s))
    arange = torch.arange
    torch.arange = lambda n,device=None,dtype=None: Index(np.arange(n)) if dtype is None else arange(n,device,dtype)
    stack = torch.stack
    torch.stack = lambda xs,dim=0: Index(np.stack([x.a for x in xs],axis=dim)) if isinstance(xs[0],Index) else stack(xs,dim)
    old_arange = torch.arange
    # .to/.view/.float return Meta; a local test subclass keeps LUT arithmetic
    # as metadata without modifying torch or an imported module.
    class Tensor(Meta):
        def __mul__(self,v): return self
        def view(self,*s):
            v=super().view(*s); return Tensor(v.shape,v.dtype,v.value,v.tag)
        def to(self,dt):
            v=super().to(dt); return Tensor(v.shape,v.dtype,v.value,v.tag)
    torch.arange = lambda n,device=None,dtype=None: Tensor((n,),dtype) if dtype is not None else old_arange(n,device,dtype)
    env = {'torch':torch, 'hashlib':hashlib, 'time':time, '__builtins__':dict(vars(builtins)),
           'threading':SimpleNamespace(Thread=C['Thread']), 'THIRD_PARTY_NOTICES':'cpu notices'}
    names = {'_HYB','_HYB_EMAX','_HYB_MMAX','_HYB_FIXF','_HYB_XROWS','_HYB_GUARD','_HYB_SLOTS',
             '_hyb_scratch','_hyb_prmt_ok','HYB_T7','HYB_F7','HYB_FLAGS','HYB_SELF_TEST','HYB_PDL',
             '_HYB_MOE_CPP','HYB_ENABLE','HYB_T','K2A_E_LOC', 'K2A_M_MAX','_k2a_bucket_of','K2A_BUCKETS',
             '_nb_vml_rep','NB_VML_REP','K2A33_PLAN','K2A33_HOT_T_BUCKET'}
    exec(nodes(base,names),env)
    names = {n.name for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name.startswith('_moexr_')}
    names |= {'_HYB_MOE_CUDA','MOEXR','_hyb_build','_hyb_prepare','_hyb_call','_hyb_on','_v11_k2a33_call'}
    if not ship: names.add('_MOEXR_SHIP')
    exec(nodes(source,names),env)
    calls,launches,inputs_seen,pack_calls=[],[],[],[]
    failures={'compile':None,'numeric':None,'sticky':False,'guard':False}
    class Mod:
        def __init__(self,flags): self.flags=flags
        def okc_set_pdl(self,pdl): assert pdl in (0,1)
        def okc_run(self,w,sc,x,ids,tw,out,lo,nloc,grid):
            assert len(w)==14 and len(sc)==3 and (lo,nloc,grid)==(0,32,132)
            assert ids.shape==tw.shape==(x.shape[0],8) and out.shape==x.shape
            smp='-DOKC_SC6=1' in self.flags
            assert w[12].numel()==32*32*16*4*(8960 if smp else 9216)
            assert w[13].numel()==32*16*16*4*(8960 if smp else 9216)
            assert sc[0].numel()==8+1024+env['_HYB_EMAX']+(4096 if smp else 0)
            f7='-DOKC_HYBST=7' in self.flags
            out.tag='bad' if failures['numeric'] in ('all', 'f7' if f7 else 'normal') else None
            if failures['sticky']: sc[0].value=1
            launches.append((smp,f7,x.shape[0]))
    def compiler(**kw):
        calls.append(kw)
        f7='-DOKC_HYBST=7' in kw['extra_cuda_cflags']
        if failures['compile'] in ('all','f7' if f7 else 'normal'): raise RuntimeError('injected nvcc failure')
        return Mod(kw['extra_cuda_cflags'])
    imp=builtins.__import__
    env['__builtins__']['__import__']=lambda name,*a,**kw: SimpleNamespace(load_inline=compiler) if name=='torch.utils.cpp_extension' else imp(name,*a,**kw)
    env['_pt1_wait']=lambda name:None
    env['_hyb_kperm_rows']=lambda rows:rows
    def pack(codes,sf,rows):
        # Unchanged legacy codec is a leaf; execute all NEW SC6 packing exactly.
        units=32 if sf.shape[0]==8192 else 16
        pack_calls.append(units)
        return Meta((units*16*4*9216,),DT.uint8,64)
    env['_hyb_wgs_copy']=pack
    def inputs(T,seed,dev,dist):
        inputs_seen.append((T,dist))
        return {'hidden_states':Meta((T,4096),DT.bfloat16),'topk_ids':Meta((T,8),DT.int32),'topk_weights':Meta((T,8))}
    env['_k2a_self_test_inputs']=inputs
    env['_k2a_ref_fp32']=lambda d,w:Meta(d['hidden_states'].shape)
    env['_k2a_rel_err']=lambda a,b:(1.,1.) if a.tag=='bad' else (0.,0.)
    env.update(VML_ENABLE=False,VML_BUCKETS={},_VML={'ok':{}},HYBLO_T=(),VMS_ENABLE=False,
               K2A36_ENABLE=False,SK1_ENABLE=False,_vml_MMAX=320)
    weights={k:Meta((32,),DT.bfloat16) for k in ('w13_gs','w2_gs','post_norm_w')}
    weights.update(w13=Meta((32,8192,2048),DT.uint8),w2=Meta((32,4096,2048),DT.uint8),
                   w13_sf=Meta((32,8192,256),DT.uint8,64),w2_sf=Meta((32,4096,256),DT.uint8,126))
    guard=env['_moexr_guards']
    # Huge float guard writes do not alias through the metadata-only view.
    # Check allocator extents here and execute the real sentinel predicate below.
    env['_moexr_guards']=lambda H:not failures['guard'] and all(n==f.numel()-env['_HYB_GUARD'] for f,n,vt,sv in H['full'])
    sentinel=Meta((20,),DT.int32,0x5EADBEEF)
    assert guard({'full':[(sentinel,10,DT.int32,0x5EADBEEF)]})
    sentinel.value=1
    assert not guard({'full':[(sentinel,10,DT.int32,0x5EADBEEF)]})
    capture=io.StringIO()
    st={'vms_ok':False}
    def prep(w=weights):
        with contextlib.redirect_stdout(capture): env['_hyb_prepare'](st,w,'cpu')
    prep()
    assert env['_HYB']['ok'], capture.getvalue()[-1500:]
    assert len(calls)==2 and len(pack_calls)==64
    expected={(T,d) for T,_,d in env['HYB_SELF_TEST']}
    assert set(inputs_seen)==expected
    assert all(smp for smp,f7,T in launches), 'base self-tests hit the lab stock bank'
    assert all((True,False,T) in launches for T,_,d in env['HYB_SELF_TEST'])
    assert all((True,True,T) in launches for T,_,d in env['HYB_SELF_TEST'] if T>=env['HYB_T7'])
    if not ship:
        with contextlib.redirect_stdout(capture): env['_moexr_ship_prepare'](st,weights,'cpu')
        assert env['_MOEXR_SHIP']['ok'],capture.getvalue()[-1500:]
        assert len(calls)==4
    initial_builds=len(calls)
    for enabled in (False,True):
        env['MOEXR']=enabled
        for T in (1,8,16,32,49,128,192,193,256,320,321):
            active=(enabled or not ship) and 49<=T<=320
            assert bool(env['_hyb_on'](T))==active
            if active:
                d=inputs(T,0,'cpu','normal')
                env['_v11_k2a33_call'](d['hidden_states'],d['topk_ids'],d['topk_weights'],st)
                assert launches[-1]==(enabled,T>=193,T)
    env['MOEXR']=True
    fallback=[]
    env['_vml_route']=lambda s,d:fallback.append('VML') or 'VML'
    env['_k2a33_call']=lambda *a,**kw:fallback.append('K2A') or 'K2A'
    def fallback_check():
        assert not env['_HYB']['ok']
        for T in (49,128,193,256,320):
            d=inputs(T,0,'cpu','normal')
            for vml in (False,True):
                env['VML_ENABLE']=vml
                rep=env['_nb_vml_rep'](T)
                env['VML_BUCKETS'][rep]=True;env['_VML']['ok'][rep]=True
                n=len(launches)
                assert env['_v11_k2a33_call'](d['hidden_states'],d['topk_ids'],d['topk_weights'],st)==('VML' if vml else 'K2A')
                assert len(launches)==n
        env['VML_ENABLE']=False
    for value in (63,127,128):
        weights['w13_sf'].value=value;prep();fallback_check()
    weights['w13_sf'].value=64
    prep({});fallback_check()
    for fault,value in (('numeric','all'),('numeric','f7'),('sticky',True),('guard',True)):
        failures[fault]=value;prep();fallback_check();failures[fault]=None if fault=='numeric' else False
        # F7 rejection intentionally removes that bank's handle; restore the
        # captured successful compiler result without launching another build.
        env['_HYB']['mod7']=Mod(list(literal(base,'HYB_FLAGS'))+list(literal(base,'HYB_F7'))+DEFINES)
    assert len(calls)==initial_builds, 'prepare retries added compiles'
    for mode in ('normal','f7','all'):
        env['_HYB'].clear();env['_HYB'].update(literal(base,'_HYB'))
        failures['compile']=mode
        before=len(calls)
        with contextlib.redirect_stdout(capture): env['_hyb_build']()
        prep();fallback_check();prep();fallback_check()
        assert len(calls)==before+2, 'failed PT1 builds must be attempted exactly once'
    env['HYB_ENABLE']=False
    before=len(calls);prep();assert len(calls)==before
    return {'status':'PASS','initial_builds':initial_builds,'self_test_cases':sorted(expected),
            'mock_launches':len(launches),'fallback_calls':len(fallback),
            'paths':['real HYB self-tests use smp mod/mod7','SC6 packing','MFIN scratch/guards','MOEXR both arms',
                     'normal/F7/both build failures with no retry','numeric normal/F7 failure','scale range',
                     'missing weight key','sticky device error','guard failure','HYB disabled','VML/K2A fallback'],
            'limitation':'CUDA numerical/reference leaves and legacy weight codec are doubles; no device validation'}


def integration_dry_run(base, lab):
    """Execute the real base K2A state constructor and the real prepare hook."""
    diet=runpy.run_path(str(ROOT/'tools/diet_patches.py'))
    env,_=diet['cpu_namespace'](lab)
    ft,fi=diet['FakeTensor'],diet['FakeImport']
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
    with contextlib.redirect_stdout(io.StringIO()): inner=env['_k2a33_prepare_st'](mw,share={'ws13':ft(),'ws2':ft()})
    env['_v11_k2a33_prepare']=lambda *a:{'st':inner,'ok':{r:True for _,_,r in env['K2A_BUCKETS']}}
    seen=[]
    def hook(st,w,dev):
        assert st is inner and w==mw
        seen.append('stock hook')
    env['_moexr_ship_prepare']=hook
    tree=ast.parse(function(lab,'prepare'))
    leaves={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and 'prepare' in n.func.id}
    for name in leaves-{'_r1_prepare','_v11_k2a33_prepare','_moexr_ship_prepare'}:
        env[name]=lambda *a,_n=name,**kw:seen.append(_n)
    env['_pt1_start']=env['_pt1_join_all']=env['_v10_build_all']=lambda:None
    with contextlib.redirect_stdout(io.StringIO()): state=env['prepare'](weights)
    assert state['k2a33']['st'] is inner and seen.count('stock hook')==1
    assert seen.index('_rq_prepare')<seen.index('stock hook')
    hooknode=next(n for n in tree.body[0].body if isinstance(n,ast.If) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='_moexr_ship_prepare' for x in ast.walk(n)))
    try: exec(compile(ast.Module(body=[hooknode],type_ignores=[]),'<hook>','exec'),env,{'state':{'k2a33':{}},'weights':weights})
    except KeyError as ex: assert ex.args==('st',)
    else: raise AssertionError('wrong nested accessor not caught')
    return {'status':'PASS','state_keys':sorted(state),'inner_state_keys':sorted(inner),
            'stock_hook_after_rq':True,'missing_nested_key_detected':True}


def linux_sass(base, reference, ship):
    """Four offline cubins: old/new smp mod and mod7; zero GPU calls."""
    assert sys.platform.startswith('linux'), 'Run the bash entry under cudaenv.sh'
    import shutil
    import subprocess
    import sysconfig
    import tempfile
    import torch
    import torch.utils.cpp_extension as ce
    assert shutil.which('nvcc') and shutil.which('cuobjdump')
    work=Path(tempfile.mkdtemp(prefix='moexr-gates-'))
    includes=[v for p in ce.include_paths(device_type='cuda') for v in ('-I',p)]+['-I',sysconfig.get_paths()['include']]
    common=['-std=c++20','-DTORCH_EXTENSION_NAME=moexr_gate','-DTORCH_API_INCLUDE_EXTENSION_H',
            '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
            '-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__','--expt-relaxed-constexpr']
    prelude='#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n'
    results,images={},{}
    cu=work/'cuda.cu'
    for f7 in (False,True):
        flags=list(literal(base,'HYB_FLAGS'))+(list(literal(base,'HYB_F7')) if f7 else [])+DEFINES
        for label,source in (('moexs',reference),('moexr',ship)):
            key=f'{label}{int(f7)}';cubin=work/(key+'.cubin')
            cu.write_bytes((prelude+literal(source,'_HYB_MOE_CUDA')).encode())
            t0=time.time()
            run=subprocess.run(['nvcc','-cubin',*common,*includes,*flags,'-Xptxas=-v',str(cu),'-o',str(cubin)],capture_output=True,text=True)
            output=run.stdout+run.stderr
            (work/(key+'.log')).write_bytes(output.encode())
            row={'status':'FAIL','nvcc_s':time.time()-t0,'flags':flags,'ptxas':C['ptxas_entries'](output)}
            results[key]=row
            if run.returncode:
                row['error']=run.stderr[-4000:];continue
            sass=subprocess.run(['cuobjdump','--dump-sass',str(cubin)],capture_output=True,text=True,check=True).stdout
            res=subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],capture_output=True,text=True,check=True).stdout
            (work/(key+'.sass')).write_bytes(sass.encode())
            images[key]=C['sass_image'](sass)
            assert images[key], 'empty SASS'
            row.update(status='PASS',resources=res,sass_sha256=S['sha'](json.dumps(images[key],sort_keys=True)))
            print(f'[moexr compile] {key}: {row["nvcc_s"]:.1f}s',flush=True)
    equal={str(f):'moexr'+str(f) in images and images['moexr'+str(f)]==images.get('moexs'+str(f)) for f in (0,1)}
    return {'status':'PASS' if all(equal.values()) and all(r['status']=='PASS' for r in results.values()) else 'FAIL',
            'builds':results,'smp_sass_exact_reference':equal,'artifacts':str(work),
            'prepare_delta':'NOT MEASURED (offline compilation is not parallel prepare)'}


def locations(base, source, edits):
    result=S['changed_lines'](base,source,edits)
    result['python']={n.name:[n.lineno,n.end_lineno] for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
                      and (n.name.startswith('_moexr_') or n.name in ('_hyb_build','_hyb_prepare','_hyb_call','_hyb_on'))}
    return result


def report(base,reference,lab,ship,edits,gates):
    def loc(s,needle):
        hits=[i for i,line in enumerate(s.splitlines(),1) if needle in line]
        assert hits,(needle,)
        return hits[0]
    def route(needle): return f'kernels/muse_decode_layer/k3_moexr_c1.py:{loc(ship,needle)}'
    sizes=gates['source']['sizes'];bc=gates['builds']['recorded_load_inline_calls']
    vms_route = route('md=vs["mod_hi"]')
    lines=[
        '| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |',
        '|---|---|---|---|---|---|---|',
        '| MOEXR smp integration | T=128 s/f, T=256 f | combo12 deltas -2.1/-2.65/-2.1; absolute combo12 baselines not supplied (notes/reset/LOG.md:128-130) | New runtime saving vs MOEXS: 0 (same CUDA/flags); byte ceiling 786432/4.64e6=0.1695 per job; 6 exposed jobs=1.017 | Existing paired layer evidence only; SC6 cuts stream bytes and MFIN permits earlier token completion; no additive attribution to hidden waits | R-b4m-0, R-hybst-2, R-hybtab-1 | CPU PASS; shipping build count restored; H200 pending |',
        '| Prepare contention | all | MEASURED 188600000 vs bundle 182700000 (66d782c2 / 3ad6bd26, notes/reset/LOG.md:57,138-140) | EST 0 extra hot-path us; 5.9 s recovery scenario=188.6-182.7, not a paired prediction | Two fewer nvcc workers; host prepare/guard clock precedes launches, so PDL cannot hide it | R-mg-6 process traps; combo2 degraded stack | Remove the two extra shipping builds |',
        '',
        'Below floor: T=256 sliding, MEASURED combo12 -0.2 us; no standalone claim.',
        '',
        'Kill reasons first: (1) compile contention may not dominate the critical worker after removal; (2) smp register pressure/planner tails can erase per-shape gains. No new kernel mechanism or additive speed claim is promoted. The requested already-measured smp implementation is copied byte-for-byte; runtime accounting relative to MOEXS is exactly zero changed bytes/instructions.',
        '',
        f'MEASURED file sizes (UTF-8/LF): bundle {sizes["base"]:,} B; MOEXS {sizes["moexs"]:,} B; lab {sizes["lab"]:,} B; ship {sizes["ship"]:,} B; generator {Path(__file__).stat().st_size:,} B. Both generated files are below 1,000,000 B. Report size is recorded by the final shell stat.',
        f'MEASURED CPU compiler-call capture: bundle {bc["base"]}, MOEXS {bc["moexs"]}, MOEXR ship {bc["ship"]}, MOEXR lab {bc["lab"]}. HYB subset 2 / 4 / 2 / 4 respectively. These count actual load_inline invocations through PT1 plus the lab post-guard hook; untouched lazy builders have identical source and inputs. Ship adds zero build jobs; all non-HYB compiler inputs match the bundle.',
        'Expected prepare delta: direction is lower contention; no new measured wall-time prediction. Recovering the observed 5.9 s difference would mean ~182.7 s (EST scenario), but the recorded 10-15 s worker slowdown is concurrent and must not be summed. SC6 still packs weights; cold host/cache variance and changed self-test routing prevent a guarantee below the 190 s guard.',
        '',
        f'Ship MOEXR=True at {route("MOEXR =")}; False disables HYB and uses VML/K2A (there is no hidden stock build). Lab MOEXR=False selects _MOEXR_SHIP, True selects _HYB; select before capture, as padz. The extra stock mod/mod7 compile only after _rq_prepare at kernels/lab/k3_moexr_ab.py:{loc(lab,"  _moexr_ship_prepare(state")}; this bank and hook are absent from ship. Lab has four HYB modules by necessity; only ship promises bundle build count.',
        f'Own HYB self-tests execute in {route("def _moexr_prepare_bank(")}: every original HYB_SELF_TEST case, twice on smp mod, plus F7 comparison at T>=193; original FP32/VML/repeat thresholds and guards retained. SC6 range, sticky MFIN error and missing/failed mod7 additionally disable smp. _hyb_on stays false on failure, and the unmodified dispatcher {route("def _v11_k2a33_call(")} takes its original VML/K2A branches. Each failed compiler is attempted once and every started hyb7 thread is joined. No explicit lab-arm exception at call time.',
        'The lab stock bank runs the same original self-tests on separate weights/scratch. The primary smp bank always self-tests directly, independent of MOEXR=False. HYBLO still uses the unchanged primary global weight slots/scales; its default tiny/tiny2 routes and source are unchanged.',
        '',
        f'Exact scored routes, both sliding/full: T=1/8 HYBLO tiny/tiny2 ({route("_HYBLO_KEYS=")}); T=16 VMS mod_hi ({vms_route}); T=32 VML bf2 ({route("VML_VAR=")}). T=128 smp mod, T=256 smp mod7 (HYB_T7=193, {route("def _moexr_call_bank(")}); both launch okc::moe_fused<true> ({route("if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);")}), 576 threads/CTA, one CTA/SM. wgs_seg<PB,1..8> at {route("case 1: wgs_seg<PB, 1>")} is chosen by NTT=ceil(job count/8), not solely by T. F7 keeps HYBST=7/HN=1; both keep PADZ=1/FT_T=64 and exactly SC6=MFIN=PLAN=1. The reference has no Python host planner: U=160/140 lives in device wgs_prefix/wgs_ranges/wgs_owner inside the unchanged CUDA literal. Host okc_run launch glue and the SC6 Python packing are preserved.',
        '',
        '| hot loop / stage | before -> after bytes | before -> after instructions per participating thread (EST source counts) |',
        '|---|---|---|',
        '| SC6 weight stage | 36,864 -> 35,840 B; physical slot stays 36,864 B, x ring unchanged | Producer leader 1 expect_tx + 1 bulk copy -> same; consumer LDS64 -> LDS32+LDS16; algebraic issue expressions 33 ->18 before CSE (16 scale-pair expressions ->4 plus 5 AND/1 OR/2 multiplies). Exact issued SASS deferred to Linux. |',
        '| MFIN B unit/WG | 4 B release/WG, 16 B/unit=1 B/stage; combined stream 36,864 ->35,841 B plus polling | Each publishing WG thread adds 1 bar.sync, wt0 adds 1 red.release; gen load/parity address arithmetic per evaluation. Split nonresolver publishes nothing. |',
        '| MFIN launch/finalize | +16,384 B persistent flags; 6,144 B opposite-parity clear replaces 12,288 B ss clear +6,344 B old cleanup; +4 B atomic/CTA; 4 B/read poll | tid0 fence+barrier atomic+poll -> atomicInc; tids0..7 poll/token, all512 add two CTA barriers/token; bounded 4,000,000 poll limit, not a static instruction total. No launch removed. |',
        '| PLAN | weight/activation stage bytes unchanged; same 2*(J+1) shared words/totals/4 range words | lane0 serial costs/prefix/4 searches ->all32 helper lanes: per32 jobs/lane, 10 shuffle-up +<=10 conditional adds +2 carry shuffles; 2 warp syncs; 1 boundary search/lane, lanes0..3 store. U=160/140 arithmetic added. Binary-search depth is data-dependent. |',
        '',
        'All hot-loop changes above are the already-measured MOEXS bits; MOEXR changes prepare plumbing only. Per expert/job weight saving: (32+16)*16*4*(9216-8960)=786,432 B; all32 experts: 25,165,824 B less. Stream law intercept 1.9 us is not removed. MFIN removes one grid rendezvous (MEASURED 1.09 us R-pdle-0) but adds polls/atomics; the 0.84 us PDL /0.80 us flag hops are not credited again. In-layer 1.51-1.53 GHz cannot turn source expressions into a reliable net gain without SASS/exposure. Combined known streamed bytes shrink; readiness polling is separately bounded, as in the measured smp arm.',
        '',
        '| post-wait cleanup / reset / combine | CTA/thread coverage proof (identical to measured smp) |',
        '|---|---|',
        '| Generation / opposite-parity reset | Every CTA tid0 atomicInc MF_EXIT modulo 2G. Compute tids0..511 clear i=CTA*512+tid+k*G*512 for i<1536 (32 CE+96 XR+88*16 DA), before phases. No compute CTA exits, including CTA>=T. |',
        '| Producers / empty routing | tids512..543 weight,544..575 x retain original returns, own no output/reset. All32 x lanes plan. nj=0 exits only wgs_ranges with zero ranges; all compute CTAs still reset/combine. |',
        '| Split combine / fc reset | Resolving WG tids128*wg..128*wg+127 cover nt*512+wt*4..+3; wt0 resets fc. wgs_flush and wgs_flush_end loops kept. Kind2 nonresolver exits only helper; kind4 whole unit has no partial buffer. |',
        '| Completion / old cleanup | 16 units*4 WG releases=64/job; whole owner or resolver only, barrier then release. Old ctr[0:2],doneA[88],xready[88],doneAs[1408] post-grid reset removed; live XR/DA get opposite-parity preclear. ctr/doneA/ss unused in WGS. |',
        '| Token combine / timeout | CTA tk%G owns tk; tids0..7 acquire its 8 routes, all512 sync before reading y, each tid covers8/4096 columns. Timeout collectively writes NaN/sticky error and continues; no partial CTA exit or skipped cleanup. Same-stream scratch reuse, as base. |',
        '',
        'CPU gates: exact single-use anchors/reversal, full smp CUDA identity, unchanged unrelated AST/constants/compiler inputs, original notices, exhaustive SC6 reconstruction, 96 planner/parity/cleanup cases, real bank construction and every new prepare branch, both arm selectors and all build/range/numerical/guard/sticky failures. Integration executes the base K2A state constructor and verifies state["k2a33"]["st"] in the copied hook. CUDA arithmetic/reference and legacy pack leaves use doubles; they do not establish H200 correctness.',
        f'Offline Linux compile/SASS: {gates["compile"]["status"]}. No WSL probe, nvcc, H200, pod, judge, or git mutation was run by the implementer. The LF generator doubles as the bash script, keeping the four-file edit limit. Run this exact command (four offline cubins: reference/candidate mod and mod7; no GPU):',
        '',
        '```bash',
        'MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" bash "<repo>/tools/moexr_patches.py"',
        '```',
        '',
        'The runner adds cuobjdump to PATH, repeats every CPU gate, compares raw SASS instruction/control words, records ptxas registers/stack/spills and retains /tmp artifacts. Results are JSON immediately below in this task-named report (no fifth output file). GPU numerical/stale/capture/timing and actual prepare time remain for the lead.',
        '',
        'Per-bit exact changed lines (inclusive generated-file ranges; plus all base/output diff hunks):',
        '```json',json.dumps({'lab':locations(base,lab,edits),'ship':locations(base,ship,edits)},indent=2),'```',
        '', 'Gate results:', '```json',json.dumps(gates,indent=2),'```',''
    ]
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--sass',action='store_true')
    args=parser.parse_args()
    base=BASE.read_text(encoding='utf-8');reference=REFERENCE.read_text(encoding='utf-8')
    lab,edits=generate(base,reference,False)
    ship,ship_edits=generate(base,reference,True)
    assert edits==ship_edits
    gates={'source':source_gates(base,reference,lab,ship,edits),'builds':build_gates(base,reference,lab,ship),
           'lab_dry_run':dry_run(base,lab,False),'ship_dry_run':dry_run(base,ship,True),
           'integration':integration_dry_run(base,lab),
           'compile':{'status':'NOT RUN','reason':'Linux cudaenv toolchain is inaccessible from this sandbox'}}
    for path,source in ((LAB,lab),(SHIP,ship)):
        if args.check: assert path.read_bytes()==source.encode(),f'stale output: {path}'
        else: path.write_bytes(source.encode())
    if args.sass:
        try: gates['compile']=linux_sass(base,reference,ship)
        except Exception as ex: gates['compile']={'status':'FAIL','error':repr(ex)}
    REPORT.write_bytes(report(base,reference,lab,ship,edits,gates).encode())
    print(json.dumps(gates,indent=2))
    if gates['compile']['status']=='FAIL': raise SystemExit(1)


if __name__=='__main__':
    main()
# MOEXR_PYTHON

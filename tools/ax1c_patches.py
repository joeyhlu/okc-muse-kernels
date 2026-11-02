"""AX1C lab generator. Only writes kernels/lab/k3_ax1c_ab.py; --check is read-only.

CPU gates execute the real AX1 prepare/state and call code with device operations
substituted, and model cp.async groups at their latest legal completion time.
No imports of the full layer, CUDA initialization, bytecode files, or GPU jobs.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c1.py"
OUT = ROOT / "kernels/lab/k3_ax1c_ab.py"
VARIANTS = [{"name": name, "set": {"AX1C": key}} for name, key in
            (("ship", 0), ("c1", 1), ("c2", 2), ("mid", 0), ("c12", 3), ("ship2", 0))]


def sub(text, old, new, label):
    assert text.count(old) == 1, (label, text.count(old), old[:100])
    return text.replace(old, new, 1)


# Scoped to fb1_l1; the q_len=8 kernel and every Lay1 offset are untouched.
C1_EDITS = (
    ("for (int y = 0; y < ST - 1; ++y) issue_idx(y);",
     "for (int y = 0; y < 2 * ST; ++y) issue_idx(y);"),
    ("#pragma unroll\nfor (int y = ST - 1; y < 2 * ST - 2; ++y) issue_idx(y);\ncpc();\n", ""),
    ("cpw<1>();\n__syncthreads();\nAX1P(1);", "cpw<0>();\n__syncthreads();\nAX1P(1);"),
    ("for (int x = 0; x < ST - 1; ++x) {", "for (int x = 0; x < ST; ++x) {"),
    ("if (xn < ST - 1 && tid < 16) {", "if (xn < ST && tid < 16) {"),
    ("cpw<ST - 2>();\n__syncthreads();\nif (i + ST - 1 < ntiles) issue_kv(i + ST - 1, true);\nissue_idx(i + 2 * ST - 2);\ncpc();",
     "cpw<ST - 1>();\n__syncthreads();"),
    ("}\n}\nAX1P(3);", "}\n// AX1C: all 128 threads finish consuming the stage before its reuse.\n"
     "__syncthreads();\nif (i + ST < ntiles) issue_kv(i + ST, true);\n"
     "issue_idx(i + 2 * ST);\ncpc(); // Empty groups at the tail preserve wait_group distance.\n}\nAX1P(3);"),
)


LAB = r'''
# AX1C lab: selectors are read only on the host, at call / capture time.
AX1C = 0
_AX1C = {k: {"mod": None, "ws": {}, "build_s": 0.0, "ok": False,
             "reason": "not prepared", "checks": [], "pairs": []} for k in (1, 2, 3)}
_AX1C_SHIP_CHECKS = []

def _ax1c_source(k):
    assert k in (0, 1, 2, 3), k
    src = AX1_CUDA_SRC
    if k & 1:
        a = src.index("template <int ST>\n__global__ void __launch_bounds__(NTHR, 2) fb1_l1")
        b = src.index("constexpr int TB = TILE * ROWB;", a)
        body = src[a:b]
        for old, new in _AX1C_EDITS:
            assert body.count(old) == 1, old[:100]
            body = body.replace(old, new, 1)
        src = src[:a] + body + src[b:]
    return src

def _ax1c_load(**kw):
    from torch.utils.cpp_extension import load_inline
    return load_inline(**kw)

def _ax1c_build(k):
    H = _AX1C[k]
    if H["mod"] is None:
        src = _ax1c_source(k)
        tag = hashlib.sha1((AX1_CPP_SRC + src + str(k)).encode()).hexdigest()[:12]
        t0 = time.time()
        H["mod"] = _ax1c_load(name=f"ax1c_{k}_{tag}", cpp_sources=AX1_CPP_SRC,
                              cuda_sources=src, functions=["ax1_forward", "ax1_info"],
                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                              verbose=False)
        H["build_s"] = time.time() - t0
    return H["mod"]

def _ax1c_ws(dev, H):
    key = str(dev)
    if key not in H['ws']:
        H['ws'][key] = (torch.zeros(8*128*64,dtype=torch.float32,device=dev),
                       torch.zeros(8*128,dtype=torch.float32,device=dev),
                       torch.zeros(16,dtype=torch.int32,device=dev))
    return H['ws'][key]

def _ax1c_select(cfg):
    k = int(AX1C)
    if k not in (0, 1, 2, 3):
        raise ValueError(f"AX1C must be 0..3, got {k}")
    H = _AX1C.get(k)
    if cfg['q_len'] == 1 and not cfg['window'] and H is not None and H['ok']:
        return H, (3, 2, 8, 0, 2, 1) if k & 2 else AX1_L1
    return _AX1, AX1_L1

def _ax1_call(qkv,data,cfg,ns=None,nc=None,st=None,pdl=None,pf=None,prof=None,trig=None,prew=None):
    H, l1 = _ax1c_select(cfg)
    return _ax1c_run(H,qkv,data,cfg,ns,nc,st,pdl,pf,prof,trig,prew,l1=l1)

def _ax1c_rel(a, b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30))

def _ax1c_pairs(ship, H, k, dev, ship_l1):
    # Same inputs and original reference/cache layout as padz and AX1 self-tests.
    plan = (3, 2, 8, 0, 2, 1) if k & 2 else ship_l1
    rows = []
    cases = [(ctx, seed, row0, extra, None, None)
             for L, ctx, seed, row0, extra in AX1_SELF_TEST if L == 1]
    # One-CTA checks remove cross-CTA atomic ordering, so load-only C1 MUST be bitwise.
    # Cover every prefill slot, idx-ring wrap, partial tiles, and poisoned pt suffix.
    cases += [(ctx, 900+ctx, True, 37, 1, st) for st in (2, 3, 4)
              for ctx in (1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 511, 512, 513)]
    for ctx, seed, row0, extra, ns, st in cases:
        d = _ax1_case(1,ctx,seed,row0,extra,dev)
        x = d['qkv'].view(1,-1)
        cfg = d['config']
        a = _ax1c_run(ship,x,d,cfg,ns=ns,st=st,l1=ship_l1).clone()
        a2 = _ax1c_run(ship,x,d,cfg,ns=ns,st=st,l1=ship_l1).clone()
        b = _ax1c_run(H,x,d,cfg,ns=ns,st=st,l1=plan).clone()
        torch.cuda.synchronize(dev)
        ref = _ax1_ref(1,d)
        row = {'ctx':ctx,'extra':extra,'ns':ns,'st':st,'ship_rel':_ax1c_rel(a,ref),
               'arm_rel':_ax1c_rel(b,ref),'vs_ship':_ax1c_rel(b,a),
               'ship_repeat':_ax1c_rel(a2,a),'bitwise':bool(torch.equal(a,b))}
        wo,wl,cnt = _ax1c_ws(dev,H)
        clean = not (int(cnt[0]) or int(cnt[7]) or float(wo.abs().sum()) or float(wl.abs().sum()))
        row['ok'] = bool(torch.isfinite(b.float()).all()) and clean and all(
            v < AX1_TOL[1] for v in (row['ship_rel'],row['arm_rel'],row['vs_ship']))
        if (k & 1) and ns == 1:
            row['ok'] = row['ok'] and row['bitwise']
        rows.append(row)
        H['pairs'] = rows
        print(f"[ax1c] arm{k} ctx{ctx}+{extra} ns{ns} st{st}: ship rel {row['ship_rel']:.3e}, "
              f"arm rel {row['arm_rel']:.3e}, vs ship {row['vs_ship']:.3e}, "
              f"ship repeat {row['ship_repeat']:.3e}, bitwise {row['bitwise']}, clean {clean} -> {row['ok']}",flush=True)
        if not row['ok']:
            raise RuntimeError(f"AX1C paired check: {row}")
    # The base graph case is ctx3000; also check the actual scored grid and an owner
    # in stage 1. Captured handles/workspace stay fixed when AX1C changes later.
    for ctx, extra, ns in ((80000,0,None),(129,37,1)):
        d = _ax1_case(1,ctx,1201,False,extra,dev)
        x = d['qkv'].view(1,-1)
        eager = _ax1c_run(H,x,d,d['config'],ns=ns,l1=plan).clone()
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _ax1c_run(H,x,d,d['config'],ns=ns,l1=plan)
        graph.replay()
        torch.cuda.synchronize(dev)
        replay = _ax1c_rel(out,eager)
        fresh = _ax1_case(1,ctx,1301,True,extra,dev)
        for name in ('qkv','k_cache','v_cache','page_table','cache_seqlens','cache_loc'):
            d[name].copy_(fresh[name])
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize(dev)
        arm_ref = _ax1c_rel(out,_ax1_ref(1,d))
        a = _ax1c_run(ship,x,d,d['config'],ns=ns,l1=ship_l1).clone()
        torch.cuda.synchronize(dev)
        ship_ref = _ax1c_rel(a,_ax1_ref(1,d))
        wo,wl,cnt = _ax1c_ws(dev,H)
        clean = not (int(cnt[0]) or int(cnt[7]) or float(wo.abs().sum()) or float(wl.abs().sum()))
        row = {'graph_ctx':ctx,'extra':extra,'replay':replay,'arm_rel':arm_ref,'ship_rel':ship_ref,
               'vs_ship':_ax1c_rel(out,a),'clean':clean}
        row['ok'] = bool(torch.isfinite(out.float()).all()) and clean and replay < 0.002 and all(
            v < AX1_TOL[1] for v in (arm_ref,ship_ref,row['vs_ship']))
        rows.append(row)
        print(f"[ax1c] arm{k} graph {row}",flush=True)
        if not row['ok']:
            raise RuntimeError(f"AX1C graph/stale check: {row}")
        del graph
    return rows

def _ax1c_tests(H, k, dev):
    # Run the BASE self-tests, including BOTH base graph checks, without rewriting
    # their numerical logic. The prepare wrapper binds _AX1 and AX1_L1 temporarily.
    import contextlib
    import io
    capture = io.StringIO()
    H['checks'] = []
    try:
        with contextlib.redirect_stdout(capture):
            for L in (1, 8):
                _ax1_self_test(dev,only=L)
    finally:
        H['checks'] = capture.getvalue().splitlines()
        for i, line in enumerate(H['checks']):
            baseline = _AX1C_SHIP_CHECKS[i] if i < len(_AX1C_SHIP_CHECKS) else 'baseline'
            print(f"[ax1c] arm{k} {line} | ship: {baseline}",flush=True)

def _ax1c_prepare(state, t0):
    global AX1C, AX1_L1, _AX1
    saved, ship_l1, ship = AX1C, AX1_L1, _AX1
    AX1C = 0
    _AX1C_SHIP_CHECKS.clear()
    for H in _AX1C.values():
        H.update(ok=False,reason='not prepared',checks=[],pairs=[])
    try:
        _ax1_prepare(state,t0)  # Exact base state['ax1'] accessor / fallback.
        if not state.get('ax1'):
            for H in _AX1C.values():
                H['reason'] = 'ship AX1 prepare failed'
            print('[ax1c] all arms OFF: ship AX1 prepare failed',flush=True)
            return
        dev = torch.empty(0,device='cuda').device
        try:
            _ax1c_tests(ship,0,dev)
            _AX1C_SHIP_CHECKS.extend(ship['checks'])
        except Exception as ex:
            for H in _AX1C.values():
                H['reason'] = f'ship full self-test failed: {ex!r}'
            for scratch in ship['ws'].values():
                for tensor in scratch:
                    tensor.zero_()
            print(f'[ax1c] all lab arms OFF: ship full self-test failed: {ex!r}',flush=True)
            return
        for k,H in _AX1C.items():
            try:
                _ax1c_build(k)
                _AX1 = H
                AX1_L1 = (3,2,8,0,2,1) if k & 2 else ship_l1
                _ax1c_ws(dev,H)
                _ax1c_tests(H,k,dev)
                _ax1c_pairs(ship,H,k,dev,ship_l1)
                H['info'] = {(L,st):list(H['mod'].ax1_info(L,st))
                             for L in (1,8) for st in ((2,3,4) if L==1 else (2,3,4,13))}
                H.update(ok=True,reason=None)
                print(f"[ax1c] arm{k} LIVE info (regs,local,smem,occ) {H['info']}",flush=True)
            except Exception as ex:
                H.update(ok=False,reason=repr(ex))
                print(f'[ax1c] arm{k} OFF: {ex!r}',flush=True)
                for scratch in H['ws'].values():
                    for tensor in scratch:
                        tensor.zero_()
            finally:
                _AX1, AX1_L1 = ship, ship_l1
    finally:
        _AX1, AX1_L1, AX1C = ship, ship_l1, saved
'''


def ax1_namespace(text, extra=None):
    """Only AX1 globals/functions: do not execute the entire 1 MB layer."""
    env = {'hashlib': hashlib, 'time': time}
    env.update(extra or {})
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', '')
            if name.startswith(('AX1', '_AX1')):
                exec(compile(ast.Module([node], []), '<AX1 assignments>', 'exec'), env)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith('_ax1'):
            exec(compile(ast.Module([node], []), '<real AX1 functions>', 'exec'), env)
    return env


def generate(base):
    env = ax1_namespace(base)
    cu = env['AX1_CUDA_SRC']
    a = cu.index('template <int ST>\n__global__ void __launch_bounds__(NTHR, 2) fb1_l1')
    b = cu.index('constexpr int TB = TILE * ROWB;', a)
    body = cu[a:b]
    for i, (old, new) in enumerate(C1_EDITS):
        body = sub(body, old, new, f'C1.{i}')
    assert body[body.index('AX1P(3);'):] == cu[a:b][cu[a:b].index('AX1P(3);'):]
    tree = ast.parse(base)
    call = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_ax1_call')
    original = ast.get_source_segment(base, call)
    run = sub(original, 'def _ax1_call(qkv,', 'def _ax1c_run(H,qkv,', 'explicit module')
    run = sub(run, 'trig=None,prew=None):', 'trig=None,prew=None,l1=None):', 'explicit geometry')
    run = sub(run, 'wo,wl,cnt=_ax1_ws(dev)', 'wo,wl,cnt=_ax1c_ws(dev,H)', 'scratch accessor')
    run = sub(run, 'L==1 and AX1_L1 is not None', 'L==1 and l1 is not None', 'plan guard')
    run = sub(run, '(int(v)for v in AX1_L1)', '(int(v)for v in l1)', 'plan values')
    run = sub(run, "_AX1['mod'].ax1_forward", "H['mod'].ax1_forward", 'module accessor')
    text = sub(base, original, run, 'base call')
    text = sub(text, "def _ax1_build(arch='90a'):",
               f'_AX1C_EDITS = {C1_EDITS!r}\n' + LAB + "\ndef _ax1_build(arch='90a'):", 'lab block')
    text = sub(text, ' _ax1_prepare(state,t0)\n', ' _ax1c_prepare(state,t0)\n', 'prepare hook')
    lines = text.splitlines()
    assert lines[0] == '# okc: problem=muse_decode_layer'
    nonce = [i for i,line in enumerate(lines[:4]) if line.startswith('# nonce ')]
    assert len(nonce) == 1, nonce
    lines[nonce[0]] = '# nonce k3-ax1c-ab (moexr_c1; AX1C=0 ship, 1 C1, 2 ct3, 3 C1+ct3; lab only)'
    text = '\n'.join(lines) + '\n'
    compile(text, str(OUT), 'exec')
    assert text.isascii() and '\r' not in text
    new = ax1_namespace(text)
    assert new['AX1C'] == 0 and new['AX1_CUDA_SRC'] == cu
    assert new['_ax1c_source'](0) == new['_ax1c_source'](2) == cu
    assert new['_ax1c_source'](1) == new['_ax1c_source'](3) == cu[:a]+body+cu[b:]
    assert '__trap' not in new['_ax1c_source'](1)
    # Non-AX1 top-level definitions must be AST-identical (including CuTe source).
    def other(s):
        return [ast.dump(n, include_attributes=False) for n in ast.parse(s).body
                if not (isinstance(n, ast.FunctionDef) and n.name.startswith('_ax1'))
                and not (isinstance(n, ast.Assign) and any(getattr(t, 'id', '').startswith(('AX1','_AX1')) for t in n.targets))]
    # The sole outer change is the real prepare call; normalize it for this gate.
    assert other(text.replace(' _ax1c_prepare(state,t0)\n',' _ax1_prepare(state,t0)\n')) == other(base)
    return text


def ring_gate():
    """Latest-legal cp.async completion; empty commits count just like real commits.

    Explicitly model all idx slots, two disjoint K/V stage allocations, owner
    patches, invalid pt suffixes, and stage reuse for ST=2/3/4.
    """
    count = 0
    for st in (2, 3, 4):
        for nt in range(1, 21):
            for valid in (1, 37, 63, 64):
                width = (nt-1)*64+valid
                for seq in sorted({0, min(width-1,63), min(width-1,64), width//2, width-1, -1, width+64}):
                    idx, slots, pending, used = {}, {}, [], set()

                    def finish(group):
                        for kind, tile in group:
                            if kind == 'idx':
                                idx[tile % (2*st)] = tile
                            else:
                                slots[tile % st] = tile

                    def wait(n):
                        while len(pending) > n:
                            finish(pending.pop(0))

                    pending.append([('idx',y) for y in range(min(2*st,nt))])
                    wait(0)
                    for x in range(st):
                        if x < nt:
                            assert idx[x % (2*st)] == x
                        pending.append([('kv',x)] if x < nt else [])
                    patched = seq // 64 if 0 <= seq < width and seq // 64 < st else None
                    for i in range(nt):
                        wait(st-1)
                        assert slots[i % st] == i
                        if seq >= 0 and i == seq//64 and i < st:
                            assert patched == i
                        used.add(i)
                        # CTA barrier here: the old tile has been fully consumed.
                        group = []
                        x = i+st
                        if x < nt:
                            assert idx[x % (2*st)] == x, (st,nt,i,idx)
                            assert x-st in used
                            group.append(('kv',x))
                        if i+2*st < nt:
                            group.append(('idx',i+2*st))
                        pending.append(group)
                    wait(0)  # Mandatory original drain before K/V -> merge scratch.
                    assert used == set(range(nt)) and not pending
                    count += 1
    return count


def cleanup_gate():
    # Post-wait merge scratch: all 128 lanes, four float4 stores per lane.
    merge, denom, output, weights, sums = set(),set(),set(),set(),set()
    for tid in range(128):
        lane,warp=tid&31,tid>>5
        hk,kh,g,t=warp&1,warp>>1,lane>>2,lane&3
        for c in range(4):
            for v in range(4): merge.add((kh*16+hk*8+g)*64+16*c+4*t+v)
        if t==0: denom.add(kh*16+hk*8+g)
        r,part=tid>>3,tid&7
        for v in range(8):
            output.add(r*64+8*part+v)
            for cp in range(8): weights.add(cp*1024+r*64+8*part+v)
        if part==0:
            for cp in range(8): sums.add(cp*16+r)
    assert merge==set(range(2048)) and denom==set(range(32))
    assert output==set(range(1024)) and weights==set(range(8192)) and sums==set(range(128))
    return {'merge_floats':len(merge),'merge_denoms':len(denom),'output_values':len(output),
            'finalizer_wo_resets':len(weights),'finalizer_wl_resets':len(sums),'counter_reset_threads':[0]}


class Tensor:
    """Strict device-free shape stub; only operations needed by host dispatch."""
    def __init__(self, shape=(), dtype='bf16', device='cuda', value=0):
        self.shape = (shape,) if isinstance(shape,int) else tuple(shape)
        self.dtype, self.device, self.value = dtype, device, value
    def view(self, *shape):
        if len(shape)==1 and isinstance(shape[0],str):
            return Tensor(self.shape,shape[0],self.device,self.value)
        return Tensor(shape,self.dtype,self.device,self.value)
    def zero_(self):
        self.value = 0
        return self
    def clone(self): return Tensor(self.shape,self.dtype,self.device,self.value)
    def float(self): return self
    def norm(self): return Tensor(value=abs(self.value))
    def abs(self): return Tensor(value=abs(self.value))
    def sum(self): return self
    def all(self): return bool(self.value)
    def clamp_min(self,v): return Tensor(value=max(v,self.value))
    def __sub__(self,b): return Tensor(value=self.value-b.value)
    def __truediv__(self,b): return Tensor(value=self.value/b.value)
    def __float__(self): return float(self.value)
    def __int__(self): return int(self.value)
    def __getitem__(self,k): return Tensor(value=self.value)
    def copy_(self,b): self.value=b.value; return self
    def fill_(self,v): self.value=v; return self
    def dim(self): return len(self.shape)
    def stride(self,i): return 1 if i==len(self.shape)-1 else self.shape[-1]
    def is_contiguous(self): return True


def dry_run(text):
    calls, builds, checks = [], [], []
    active_graph = []
    class Graph:
        def __init__(self): self.outputs=[]
        def __enter__(self): active_graph.append(self); return self
        def __exit__(self,*args): active_graph.pop()
        def replay(self):
            for t in self.outputs: t.value=1
    class Module:
        def __init__(self, name): self.name = name
        def ax1_forward(self, *args):
            calls.append((self.name,args))
            args[9].value=1
            if active_graph: active_graph[-1].outputs.append(args[9])
        def ax1_info(self, L, st): return [100,0,36160,4]
    fake_torch = SimpleNamespace(zeros=lambda shape,**kw:Tensor(shape,**kw),
                                 empty=lambda shape,**kw:Tensor(shape,**kw),
                                 float32='f32',int32='i32',int64='i64',uint8='u8',bfloat16='bf16',
                                 equal=lambda a,b:a.value==b.value,
                                 isfinite=lambda a:Tensor(value=__import__('math').isfinite(a.value)),
                                 cuda=SimpleNamespace(synchronize=lambda *a:None,CUDAGraph=Graph,graph=lambda g:g))
    e = ax1_namespace(text, {'torch':fake_torch,'FP8':'fp8'})
    ship = e['_AX1']
    ship['mod'] = Module('ship')
    e['_pt1_wait'] = lambda name: None
    e['_PT1'] = {'errors':{}}
    fail = {}
    def load(**kw):
        builds.append(kw)
        k = int(kw['name'].split('_')[1])
        if fail.get('build') == k: raise RuntimeError('injected compile failure')
        assert '-gencode=arch=compute_90a,code=sm_90a' in kw['extra_cuda_cflags']
        return Module(f'arm{k}')
    def self_test(dev,verbose=True,only=None):
        checks.append((e['_AX1']['mod'].name,only))
        if fail.get('self') == e['_AX1']['mod'].name: raise RuntimeError('injected self-test failure')
        if fail.get('ship8') and e['_AX1'] is ship and only==8: raise RuntimeError('injected ship q8 failure')
        print(f'[ax1] self-test L{only}: PASS err 0.000e+00')
        print(f'[ax1] graph check L{only}: fresh inputs err 0.000e+00')
        return 0.0
    real_pairs = e['_ax1c_pairs']
    def pairs(ship_H,H,k,dev,ship_l1):
        if fail.get('pairs') == k: raise RuntimeError('injected graph failure')
        assert ship_H is ship and H is e['_AX1C'][k]
        return real_pairs(ship_H,H,k,dev,ship_l1)
    # Dict key set / config fields come from the REAL case factory's return AST.
    tree = ast.parse(BASE.read_text(encoding='ascii'))
    factory = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_k1fa4_case')
    layout = next(n.value for n in ast.walk(factory) if isinstance(n,ast.Return))
    keys = [ast.literal_eval(k) for k in layout.keys]
    config = layout.values[keys.index('config')]
    config_keys = [ast.literal_eval(k) for k in config.keys]
    def case(L,ctx,seed,row0,extra,dev):
        d={k:Tensor(value=1) for k in keys}
        d['config']={k:0 for k in config_keys}
        d['config'].update(q_len=L,window=0,sm_scale=0.125,qk_scale_factor=5.4730064863875)
        d['page_table']=Tensor((1,ctx+L+extra),'i32')
        d['qkv']=Tensor((1,L,2304),value=1)
        return d
    e.update(_ax1c_load=load,_ax1_self_test=self_test,_ax1c_pairs=pairs,
             _ax1_case=case,_ax1_ref=lambda L,d:Tensor((L,1024),value=1))
    state = {}  # BASE _ax1_prepare creates state['ax1']; no invented state keys.
    with contextlib.redirect_stdout(io.StringIO()):
        e['_ax1c_prepare'](state,time.time())
    assert state == {'ax1':True} and e['_AX1'] is ship and e['AX1C']==0 and e['AX1_L1'] is None
    assert len(builds)==3 and len({b['name'] for b in builds})==3
    assert all(H['ok'] for H in e['_AX1C'].values())
    assert all((f'arm{k}',L) in checks for k in (1,2,3) for L in (1,8))
    def data(L=1,window=0,width=80000):
        return {'config':{'q_len':L,'window':window,'sm_scale':0.125,'qk_scale_factor':1.0,
                          'batch':1,'page_size':1,'kv_scale':1.0,'head_dim':64,'num_q_heads':16,'num_kv_heads':2},
                'page_table':Tensor((1,width),'i32'),'cache_seqlens':Tensor(1,'i32'),
                'cache_loc':Tensor(L,'i64'),'k_cache':Tensor((width,2,64),'fp8'),
                'v_cache':Tensor((width,2,64),'fp8')}
    routes = []
    for k in (0,1,2,3):
        e['AX1C']=k
        for L,window in ((1,0),(1,2048),(8,0)):
            d=data(L,window)
            assert e['_ax1_wanted'](Tensor((L,2304)),d,d['config'],state)==(L==1 and window==0)
            e['_ax1_call'](Tensor((L,2304)),d,d['config'])
            name,args=calls[-1]
            active = k and L==1 and window==0
            assert name == (f'arm{k}' if active else 'ship')
            ct3 = active and k&2
            assert args[11:13] == ((192,417) if ct3 else e['_ax1_plan'](80000,e['AX1_SPLITS'][L]))
            assert args[13] == 8 and args[17] == e['AX1_ST'][L]
            routes.append((k,L,window,name,args[11:13]))
        # Explicit ns overrides ct3; override parameters must survive forwarding.
        d=data()
        e['_ax1_call'](Tensor((1,2304)),d,d['config'],ns=1,nc=4,st=4,pdl=False,pf=1,trig=0,prew=0)
        assert calls[-1][1][11:14] == (80000,1,4)
        assert calls[-1][1][17:20] == (4,0,1) and calls[-1][1][21:] == (0,0)
    assert len({id(e['_ax1c_ws']('cuda',H)[0]) for H in [ship]+list(e['_AX1C'].values())}) == 4
    # Every failure path: compile, numeric/self, graph/pairs, baseline, disabled.
    for stage,k in (('build',1),('self','arm2'),('pairs',3)):
        fail.clear(); fail[stage]=k
        if stage=='build': e['_AX1C'][1]['mod']=None
        e['AX1C']=3
        with contextlib.redirect_stdout(io.StringIO()): e['_ax1c_prepare'](state,time.time())
        bad = int(str(k).replace('arm',''))
        assert not e['_AX1C'][bad]['ok'] and 'injected' in e['_AX1C'][bad]['reason']
        e['AX1C']=bad
        assert e['_ax1c_select'](data()['config'])[0] is ship
        assert e['_AX1'] is ship and e['AX1_L1'] is None
    fail.clear(); fail['ship8']=True
    with contextlib.redirect_stdout(io.StringIO()): e['_ax1c_prepare'](state,time.time())
    assert state['ax1'] is True and all('ship full self-test failed' in H['reason'] for H in e['_AX1C'].values())
    fail.clear(); fail['self']='ship'
    with contextlib.redirect_stdout(io.StringIO()): e['_ax1c_prepare'](state,time.time())
    assert state['ax1'] is False and not any(H['ok'] for H in e['_AX1C'].values())
    fail.clear(); e['AX1_ENABLE']=False
    with contextlib.redirect_stdout(io.StringIO()): e['_ax1c_prepare'](state,time.time())
    assert state['ax1'] is False
    e['AX1C']=99
    try: e['_ax1c_select'](data()['config'])
    except ValueError: pass
    else: raise AssertionError('invalid selector accepted')
    return {'routes':routes,'module_build_paths':3,'failure_paths':6,'base_state':list(state),
            'paired_cases_per_arm':54,'real_case_factory_keys':keys,
            'gpu_math':'device-free tensors; base numerical self-test stubbed; NEW paired/graph host paths executed'}


def gates(check=False):
    base = BASE.read_text(encoding='ascii')
    text = generate(base)
    if check:
        assert OUT.read_bytes() == text.encode('ascii'), 'generated lab differs; regenerate'
    result = {'ast':'PASS','anchors':'PASS','ship_cuda_identity':'PASS','non_ax1_ast_identity':'PASS',
              'ring_cases':ring_gate(),'cleanup_coverage':cleanup_gate(),'dry_run':dry_run(text),'bytes':len(text),
              'base_sha256':hashlib.sha256(base.encode()).hexdigest(),
              'lab_sha256':hashlib.sha256(text.encode()).hexdigest()}
    if not check: OUT.write_bytes(text.encode('ascii'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--variants',action='store_true')
    args = parser.parse_args()
    print(json.dumps(VARIANTS if args.variants else gates(args.check),indent=2))

"""Default-off GD-A, preserving every existing compiled source in k3_pure_c1.

python -B tools/gda_patches.py                 # generate the one named lab file
python -B tools/gda_patches.py --cpu-gate      # no CUDA, no output files
bash tools/local/gda_gate.sh                  # Linux compile/SASS gate, no GPU

GDA={'qkv16': 3, 'cat32': 3} before prepare builds both routes at ST=3..6.
Thereafter each selector is call-time, 0 uses the original module/record.
Kill risks: smem occupancy loss; weight latency hidden by other warps/PDL.
Weight bytes do NOT grow: 1536 B/warp/unit, 48 B/lane, before and after.
New smem traffic is 1536 B written + 1536 B read/warp/unit (not HBM).
Source-level lane transport: 3 LDG -> 3 CP + commit + wait + 3 LDS per unit;
KU commits (including empty tail groups), ST initial commits, one final drain.
Decode's 36 slots/unit and 4*MT MMA instructions are copied, not redesigned.
Address/loop instructions and actual registers must be read from the SASS gate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/muse_decode_layer/k3_pure_c1.py'
OUT = ROOT / 'kernels/lab/k3_gda_ab.py'
BEGIN = '\n# BEGIN GDA independent module (tools/gda_patches.py)\n'
END = '# END GDA independent module\n'
PREP = ' _rq_prepare(state)\n'
PREP_NEW = PREP + ' _gda_prepare(state)\n'
QCALL = ' r=_gd_live(state,proj,T)\n'
QCALL_NEW = QCALL + ' r=_gda_select(state,"qkv16",r) if proj=="qkv" and T==16 else r\n'
CCALL = 'def _gd_cat(state,x,data,tokens):\n r=_gd_cat_live(state,tokens)\n'
CCALL_NEW = CCALL + ' r=_gda_select(state,"cat32",r) if tokens==32 else r\n'


def sub(s, a, b, count=1):
    assert s.count(a) == count, (a[:160], s.count(a), count)
    return s.replace(a, b)


def literal(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def cuda_edits(cu):
    """Only transport and its launch allocation change; VF=0 for both targets."""
    init = '''uint4 R[ST][NP];
#pragma unroll
for (int i = 0; i < ST; ++i)
if (i < KU) {
#pragma unroll
for (int p = 0; p < NP; ++p)
R[i][p] = (XP & 2) ? ldwp(wp + (size_t)i * wstep + p * 32) : ldw(wp + (size_t)i * wstep + p * 32);
}'''
    init_new = '''static_assert(!WSM || (VF == 0 && ST >= 3 && ST <= 6), "GD-A supports VF0, ST3..6");
uint4 R[WSM ? 1 : ST][NP];
// Each lane owns [stage][warp][plane][lane]. No other lane reads its slot.
uint4* ring = reinterpret_cast<uint4*>(gd_vf_dsm);
#pragma unroll
for (int i = 0; i < ST; ++i) {
if (i < KU) {
#pragma unroll
for (int p = 0; p < NP; ++p) {
if constexpr (WSM)
vf_cp16(ring + ((i * NW + warp) * NP + p) * 32 + lane,
        wp + (size_t)i * wstep + p * 32);
else R[i][p] = (XP & 2) ? ldwp(wp + (size_t)i * wstep + p * 32) : ldw(wp + (size_t)i * wstep + p * 32);
}
}
if constexpr (WSM) asm volatile("cp.async.commit_group;\\n" ::: "memory");
}'''
    step = '''uint32_t a[16];
decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);
if (s + ST < KU) {
#pragma unroll
for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);
}'''
    step_new = '''if constexpr (WSM) {
asm volatile("cp.async.wait_group %0;\\n" :: "n"(ST - 1) : "memory");
#pragma unroll
for (int p = 0; p < NP; ++p) {
const unsigned addr = smem_u32(ring + ((slot * NW + warp) * NP + p) * 32 + lane);
asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n"
    : "=r"(R[0][p].x), "=r"(R[0][p].y), "=r"(R[0][p].z), "=r"(R[0][p].w)
    : "r"(addr) : "memory");
}
}
uint32_t a[16];
decode(R[WSM ? 0 : slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);
if (s + ST < KU) {
#pragma unroll
for (int p = 0; p < NP; ++p) {
if constexpr (WSM)
vf_cp16(ring + ((slot * NW + warp) * NP + p) * 32 + lane,
        wp + (size_t)(s + ST) * wstep + p * 32);
else R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);
}
}
// Empty commits past KU preserve the wait_group ST-1 invariant at the tail.
if constexpr (WSM) asm volatile("cp.async.commit_group;\\n" ::: "memory");'''
    drain = '''if constexpr (CAT > 0) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;
}
__syncthreads();'''
    edits = [
        ('int VF = 0, int CAT = 0>', 'int VF = 0, int CAT = 0, int WSM = 0>', 1),
        ('gd_kernel', 'gda_kernel', cu.count('gd_kernel')),
        (init, init_new, 1), (step, step_new, 1),
        (drain, drain.replace('__syncthreads();',
            'if constexpr (WSM) asm volatile("cp.async.wait_group 0;\\n" ::: "memory");\n__syncthreads();'), 1),
        ('gda_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>',
         'gda_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT, 1>', 2),
        ('cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);\ncfg.dynamicSmemBytes = 0;',
         'cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);\ncfg.dynamicSmemBytes = WR * WK * ST * 3 * 32 * 16;', 1),
        ('int occ = 0;\ne = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, 0);',
         '''constexpr int DS = WR * WK * ST * 3 * 32 * 16;
e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, DS);
if (e) return e;
int occ = 0;
e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, DS);''', 1),
    ]
    s = cu
    for a, b, n in edits:
        s = sub(s, a, b, n)
    # The entire post-loop reduction/epilogue remains identical, except drain.
    ep = 'auto cta_value = [&](int i) -> float {'
    assert cu[cu.index(ep):cu.index('static int launch')] == s[s.index(ep):s.index('static int launch')]
    return edits


LAB = '''GDA = {"qkv16": 0, "cat32": 0}
_GDA = {"mod": None, "keys": None, "build_s": 0.0}
def _gda_validate():
 if not isinstance(GDA,dict) or set(GDA)-{"qkv16","cat32"}:
  raise ValueError("GDA keys must be qkv16 / cat32")
 if any(type(v) is not int or v not in (0,3,4,5,6) for v in GDA.values()):
  raise ValueError("GDA stages must be 0 or 3..6")
def _gda_keys():
 q=_gd_key(*_gd_nk("qkv",16),GD_ROUTE[("qkv",16)])
 c=_gd_cat_key(GD_CAT_ROUTE[32])
 assert q==(0,2,1,4,8,1,4,5,0), q
 assert c==(0,4,1,8,18,1,1,2,0,2), c
 return [(name,st,k[:5]+(st,)+k[6:]) for name,k in (("qkv16",q),("cat32",c)) for st in (3,4,5,6)]
def _gda_src(keys):
 src=GD_CUDA_SRC
 for old,new,n in _GDA_EDITS:
  assert src.count(old)==n, old[:100]
  src=src.replace(old,new)
 inst="\\n".join("    GD_INST(%d, %s)"%(i,", ".join(map(str,k))) for i,(_,_,k) in enumerate(keys))
 info="\\n".join("    GD_INFO(%d, %s)"%(i,", ".join(map(str,k))) for i,(_,_,k) in enumerate(keys))
 return src.replace("@INST@",inst).replace("@INFO@",info)
def _gda_compile(**kwargs):
 from torch.utils.cpp_extension import load_inline
 return load_inline(**kwargs)
def _gda_build():
 keys=_gda_keys()
 if _GDA["mod"] is not None and _GDA["keys"]==keys:
  return _GDA["mod"],keys
 src=_gda_src(keys)
 flags=["-std=c++20","-O3","-gencode=arch=compute_90a,code=sm_90a","-Xptxas=-v,-warn-spills"]
 tag=hashlib.sha1((GD_CPP_SRC+src+" ".join(flags)).encode()).hexdigest()[:12]
 t=time.time()
 mod=_gda_compile(name=f"gda_{tag}",cpp_sources=GD_CPP_SRC,cuda_sources=src,
  functions=["gd_run","gd_info","gd_unpack","gdr_run","gdr_info","gdc_run","gd_unpack_cat"],
  extra_cflags=["-std=c++20"],extra_cuda_cflags=flags,verbose=False)
 _GDA.update(mod=mod,keys=keys,build_s=time.time()-t)
 return mod,keys
def _gda_prepare(state):
 _gda_validate()
 if not any(GDA.values()):
  return
 t=time.time()
 # Exact base accessors: _gd_live and _gd_cat_live; no repack/scratch changes.
 refs={"qkv16":_gd_live(state,"qkv",16),"cat32":_gd_cat_live(state,32)}
 for name,st in GDA.items():
  if st and refs[name] is None:
   raise RuntimeError("GDA base route not live: "+name)
 mod,keys=_gda_build()
 bank={}
 resources={}
 for inst,(name,st,key) in enumerate(keys):
  r=refs[name]
  if r is None:
   continue
  info=tuple(mod.gd_info(inst))  # also sets dynamic-smem attribute, before capture
  if info[0]>128 or info[1]!=0 or info[2]*GD_SMS<r["nbx"]*key[6]:
   raise RuntimeError(f"GDA resource gate {name}/{st}: {info}")
  # Reuse every packed pointer, base, stride and split field verbatim.
  bank[(name,st)]=dict(r,mod=mod,inst=inst,desc=r["desc"]+f" gda{st}")
  resources[f"{name}/{st}"]={"registers":info[0],"local_bytes":info[1],"occupancy":info[2],
   "static_smem":info[3],"dynamic_smem":key[2]*key[3]*st*1536}
 state["_gda"]={"bank":bank,"resources":resources,"prepare_s":time.time()-t,"build_s":_GDA["build_s"]}
 print(f"[gda] prepare {state['_gda']['prepare_s']:.3f}s build {_GDA['build_s']:.3f}s {resources}",flush=True)
def _gda_select(state,name,r):
 st=GDA.get(name,0)
 if type(st) is not int or st not in (0,3,4,5,6):
  raise ValueError("GDA stages must be 0 or 3..6")
 if not st:
  return r
 if r is None:
  raise RuntimeError("GDA base route not live: "+name)
 if "_gda" not in state or (name,st) not in state["_gda"]["bank"]:
  raise RuntimeError("Enable GDA before prepare to build "+name)
 return state["_gda"]["bank"][(name,st)]
'''


def generate(base):
    edits = cuda_edits(literal(base, 'GD_CUDA_SRC'))
    inserted = BEGIN + LAB + '_GDA_EDITS = ' + repr(edits) + '\n' + END
    out = sub(base, PREP, PREP_NEW)
    out = sub(out, QCALL, QCALL_NEW)
    out = sub(out, CCALL, CCALL_NEW)
    out = sub(out, 'def _gd_bucket(T):\n', inserted + 'def _gd_bucket(T):\n')
    ast.parse(out)
    return out


def namespace(source, torch):
    """Execute actual base functions + GDA, with no module/GPU import."""
    functions = {'_gd_key', '_gd_nk', '_gd_keys', '_gd_cat_key', '_gd_cat_keys',
                 '_gd_live', '_gd_cat_live', '_gd_bucket', '_gd', '_gd_cat', '_gd_cat_run'}
    constants = {'HIDDEN', 'D_EXPERT', 'N_EXPERTS', 'UNPOOL_SLICE', 'SLICED_UNPOOL_MAX_TOKENS',
                 'GD_SMS', 'GD_ROUTE', 'GD_DIMS', 'GD_CAT_ROUTE', 'GD_CAT_UNITS', 'GD_CPP_SRC',
                 'GD_CUDA_SRC', 'GD_ENABLE', 'GD_PDL', 'GD_TRIG', 'GD_CAT', 'GD_CAT_PDL',
                 '_GD_SEEN', '_GD_CAT_SEEN', 'GDA', '_GDA', '_GDA_EDITS', 'NB64_USE', 'NB64_GD_B'}
    nodes = []
    for n in ast.parse(source).body:
        if isinstance(n, ast.FunctionDef) and (n.name in functions or n.name.startswith('_gda_')):
            nodes.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets):
            nodes.append(n)
    ns = {'torch': torch, 'time': time, 'hashlib': hashlib, '_nb64': lambda t: 33 <= t <= 64}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<gd-base-ast>', 'exec'), ns)
    return ns


def base_state(base, ns, mod):
    """Evaluate the REAL base's r/state construction expressions (padz/tailw pattern)."""
    funcs = {n.name: n for n in ast.parse(base).body if isinstance(n, ast.FunctionDef)}
    state = {}
    for name, proj, b in (('_gd_prepare', 'qkv', 16), ('_gd_cat_prepare', 'cat', 32)):
        rt = ns['GD_ROUTE'][(proj,b)] if proj == 'qkv' else ns['GD_CAT_ROUTE'][b]
        key = ns['_gd_key'](*ns['_gd_nk'](proj,b),rt) if proj == 'qkv' else ns['_gd_cat_key'](rt)
        expr = next(n.value for n in ast.walk(funcs[name]) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'r' for t in n.targets))
        pk = {'w': object(), 'esc': object(), 'tb': 17, 'tbr': 18, 'nbx': 144 if proj == 'qkv' else 256}
        env = dict(ns, mod=mod, ids={key: 17}, key=key, pk=pk, n=2304, k=8192,
                   rt=rt, proj=proj, desc='base', state=state)
        r = eval(compile(ast.Expression(expr), '<base-record>', 'eval'), env)
        routes = {(proj,b): r} if proj == 'qkv' else {b: r}
        field = 'gd' if proj == 'qkv' else 'gd_cat'
        node = next(n for n in ast.walk(funcs[name]) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Subscript) and ast.unparse(t) == f"state['{field}']" for t in n.targets)
                    and not (isinstance(n.value, ast.Constant) and n.value.value is None))
        env.update(routes=routes)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<base-state>', 'exec'), env)
    return state


class FakeTensor:
    def __init__(self, shape, dtype='bf16', device='cpu'):
        self.shape, self.dtype, self.device = shape, dtype, device
    def dim(self):
        return len(self.shape)
    def is_contiguous(self):
        return True
    def data_ptr(self):
        return 16


def cpu_dry_run(base, out):
    calls, builds = [], []
    torch = types.SimpleNamespace(bfloat16='bf16', float32='f32', empty=FakeTensor)
    ns = namespace(out, torch)
    # These resource values are synthetic path-test inputs, NEVER measurements.
    # Suppress prepare diagnostics so they cannot be mistaken for ptxas output.
    ns['print'] = lambda *a, **kw: None
    class Module:
        def __init__(self, name):
            self.name = name
        def gd_info(self, inst):
            return (96 if inst < 4 else 128, 0, 5 if inst < 4 else 2, 7312 if inst < 4 else 32784)
        def gd_run(self, *args):
            calls.append((self.name, 'qkv', args))
        def gdc_run(self, *args):
            calls.append((self.name, 'cat', args))
    ship, opt = Module('ship'), Module('gda')
    def build(**kwargs):
        builds.append(kwargs)
        assert kwargs['cpp_sources'] == ns['GD_CPP_SRC']
        assert '-std=c++20' in kwargs['extra_cflags'] and '-std=c++20' in kwargs['extra_cuda_cflags']
        assert '@INST@' not in kwargs['cuda_sources']
        return opt
    ns['_gda_compile'] = build
    ns['_v8_tail'] = lambda y, part, state, data: (y, part)
    ns['_k3_disable'] = lambda *args: (_ for _ in ()).throw(AssertionError(('fallback',args)))
    state = base_state(base, ns, ship)
    saved = dict(state)
    ns['_gda_prepare'](state)
    assert state == saved and not builds and '_gda' not in state
    qref = state['gd']['route'][('qkv',16)]
    cref = state['gd_cat'][32]
    assert ns['_gda_select'](state,'qkv16',qref) is qref
    # Execute both real call sites, including both outputs of cat.
    def call(q, c):
        ns['GDA'] = {'qkv16':q,'cat32':c}
        ns['_gd'](state,'qkv',FakeTensor((16,8192)))
        ns['_gd_cat'](state,FakeTensor((32,8192)),{},32)
    call(0,0)
    baseline = calls[-2:]
    ns['GDA'] = {'qkv16':3,'cat32':3}
    try:
        ns['_gda_select'](state,'qkv16',qref)
        raise AssertionError('unprepared enabled route accepted')
    except RuntimeError:
        pass
    ns['_gda_prepare'](state)
    assert len(builds) == 1 and len(state['_gda']['bank']) == 8
    for name, ref in (('qkv16',qref),('cat32',cref)):
        for st in (3,4,5,6):
            r = state['_gda']['bank'][(name,st)]
            for k,v in ref.items():
                if k not in ('mod','inst','desc'):
                    assert r[k] is v, (name,st,k)
    for q in (0,3,4,5,6):
        for c in (0,3,4,5,6):
            call(q,c)
            for record, stage, ref in zip(calls[-2:],(q,c),baseline):
                assert record[0] == ('gda' if stage else 'ship')
                # Compare all pointer/scalar arguments except output allocations and inst.
                indices = (1,2,5,6,7,8,9,10,11) if record[1]=='qkv' else (1,2,6,7,8,9,10,11,12,13)
                assert all(record[2][i] == ref[2][i] for i in indices)
    ns['_gda_prepare'](state)  # cache-hit path
    assert len(builds) == 1
    # Disabled target, missing target, partial prepare and resource failure paths.
    for selection in ({'bad':3},{'qkv16':2},{'qkv16':True}):
        ns['GDA'] = selection
        try:
            ns['_gda_prepare'](state)
            raise AssertionError('bad selector accepted')
        except ValueError:
            pass
    ns['GDA'] = {'qkv16':0,'cat32':3}
    partial = dict(state,gd=None)
    ns['_gda_prepare'](partial)
    assert len(partial['_gda']['bank']) == 4
    ns['GDA'] = {'qkv16':3}
    try:
        ns['_gda_prepare'](partial)
        raise AssertionError('missing base accepted')
    except RuntimeError:
        pass
    good_info = opt.gd_info
    opt.gd_info = lambda i: (129,0,5,0)
    try:
        ns['_gda_prepare'](state)
        raise AssertionError('bad resources accepted')
    except RuntimeError:
        pass
    opt.gd_info = good_info
    # Shape guards: the selector applies only to exact scored T=16/T=32.
    ns['GDA'] = {'qkv16':3,'cat32':3}
    ns['_gd'](state,'qkv',FakeTensor((9,8192)))
    ns['_gd_cat'](state,FakeTensor((17,8192)),{},17)
    assert all(c[0]=='ship' for c in calls[-2:])
    return {'ok':True,'selector_pairs':25,'bank_variants':8,'build_calls':len(builds),
            'base_schema':'AST-evaluated _gd_prepare / _gd_cat_prepare', 'call_records':len(calls)}


def ring_proof():
    """Count pending groups; prove lane-exclusive slots, tail drain and output coverage."""
    rows = []
    for nw, ku, mt, splits in ((4,8,2,4),(8,18,4,1)):
        for st in (3,4,5,6):
            for warp in range(nw):
                for lane in range(32):
                    slots, pending = {}, []
                    def issue(unit):
                        pending.append(unit)
                    def wait(n):
                        while len(pending)>n:
                            unit=pending.pop(0)
                            if unit<ku:
                                for p in range(3):
                                    addr=(((unit%st)*nw+warp)*3+p)*32+lane
                                    slots[addr]=(unit,p,warp,lane)
                    for unit in range(st):
                        issue(unit)
                    for unit in range(ku):
                        wait(st-1)
                        for p in range(3):
                            addr=(((unit%st)*nw+warp)*3+p)*32+lane
                            assert slots[addr] == (unit,p,warp,lane)
                        issue(unit+st)
                    wait(0)
                    assert not pending
            nt, no = nw*32, mt*128
            indices = [i for tid in range(nt) for i in range(tid,no,nt)]
            assert sorted(indices)==list(range(no))
            # Every rank writes all of its red[] entries. No lane/CTA exits
            # before the per-lane drain or the unchanged __syncthreads().
            red = [(w,j,l) for w in range(nw) for l in range(32) for j in range(mt*4)]
            assert len(set(red))==nw*mt*4*32
            if splits>1:
                # Nonleaders send all NO values to their own pbuf plane, THEN
                # return. Rank0 owns own[j], the wait, ordered S-1 sum and store.
                planes=[(rank,i) for rank in range(1,splits) for i in indices]
                assert len(set(planes))==(splits-1)*no
                coverage={'all_CTAs':'threads 0..127: drain, red, CTA barrier',
                          'rank0_init':'thread 0: mbarrier expect_tx for 3*256*4 bytes',
                          'ranks1_3':'threads 0..127: send i=tid,tid+128 to pbuf[rank-1], then return',
                          'rank0':'threads 0..127: own/wait/combine/store i=tid,tid+128 (0..255)',
                          'reset':'none added; base per-launch mbarrier initialization retained'}
            else:
                partials=[i for tid in range(nt) for i in range(tid,no//2,nt)]
                assert sorted(partials)==list(range(no//2))
                coverage={'all_CTAs':'threads 0..255: park pooler red, final redr, drain, CTA barrier',
                          'pooler':'threads 0..255: combine/store i=tid,tid+256 (0..511)',
                          'router':'threads 0..255: ordered combine/store i=tid (0..255)',
                          'reset':'accumulators reset by all 256 threads at s=15; no new early exits or resets'}
            rows.append({'warps':nw,'KU':ku,'stages':st,'dynamic_smem':nw*st*1536,
                         'reduction_outputs':no,'cluster_ranks':splits,'post_wait_coverage':coverage})
    return {'ok':True,'variants':rows}


def default_identical(base, out):
    stripped = out[:out.index(BEGIN)]+out[out.index(END)+len(END):]
    for a,b in ((PREP,PREP_NEW),(QCALL,QCALL_NEW),(CCALL,CCALL_NEW)):
        stripped = sub(stripped,b,a)
    assert stripped == base, 'inverse diff changed base'
    count = 0
    for n in ast.parse(base).body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
            for t in n.targets:
                if isinstance(t,ast.Name):
                    assert literal(out,t.id)==n.value.value,t.id
                    count += 1
    bns, ons = namespace(base,None), namespace(out,None)
    def expanded(ns):
        keys=ns['_gd_keys']()
        return ns['GD_CUDA_SRC'].replace('@INST@','\n'.join('    GD_INST(%d, %s)'%(i,', '.join(map(str,k))) for i,k in enumerate(keys))).replace(
            '@INFO@','\n'.join('    GD_INFO(%d, %s)'%(i,', '.join(map(str,k))) for i,k in enumerate(keys)))
    bcu, ocu = expanded(bns), expanded(ons)
    assert bcu==ocu
    return {'ok':True,'unchanged_string_constants':count,'GD_expanded_sha256':hashlib.sha256(bcu.encode()).hexdigest()}, bcu, ocu


def py_compile_memory(paths):
    import py_compile
    from unittest.mock import patch
    result = {}
    for path in paths:
        with patch('importlib._bootstrap_external._write_atomic',side_effect=lambda name,data,mode: None):
            py_compile.compile(str(path), cfile=str(path)+'.gda-pyc', doraise=True)
        result[str(path.relative_to(ROOT))] = 'PASS (bytecode write suppressed)'
    return result


def cpu_gate(base, out):
    assert out==generate(base),'stale generated output'
    default,_,_ = default_identical(base,out)
    return {'default_identical':default,'dry_run':cpu_dry_run(base,out),'ring':ring_proof(),
            'py_compile':py_compile_memory([Path(__file__),OUT,ROOT/'tools/box/gda_check.py']),
            'shell_LF':b'\r' not in (ROOT/'tools/local/gda_gate.sh').read_bytes()}


def linux_gate(base, out, report):
    """Compile/link only. No CUDA initialization, no kernel launch, no cleanup deletes."""
    import tempfile
    import sysconfig
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
    build = Path(tempfile.mkdtemp(prefix='gda-gate-'))
    report['build_dir'] = str(build)
    ns = namespace(out,None)
    _, bcu, ocu = default_identical(base,out)
    keys=ns['_gda_keys']()
    sources={'base':bcu,'default':ocu,'gda':ns['_gda_src'](keys)}
    def run(cmd):
        p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        report.setdefault('commands',[]).append({'argv':cmd,'returncode':p.returncode})
        if p.returncode:
            raise RuntimeError(repr(cmd)+'\n'+p.stdout[-24000:])
        return p.stdout
    def resources(log):
        found={}
        for block in re.split(r'ptxas info\s*: Compiling entry function ',log)[1:]:
            name=block.split("'",2)[1]
            reg=re.search(r'Used (\d+) registers',block)
            spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block)
            smem=re.search(r'(\d+) bytes smem',block)
            assert reg and spill,block
            found[name]={'registers':int(reg[1]),'stack':int(spill[1]),'spill_stores':int(spill[2]),
                         'spill_loads':int(spill[3]),'static_smem':int(smem[1]) if smem else 0}
        assert found,log
        return found
    def sass_functions(text):
        found={}
        for block in re.split(r'Function\s*:\s*',text)[1:]:
            name=block.splitlines()[0].strip()
            found[name]=[re.sub(r'/\*.*?\*/','',l).strip() for l in block.splitlines()
                         if re.match(r'\s*/\*[0-9a-f]+\*/',l)]
        assert found
        return found
    def encoded_functions(text):
        # Include encoded scheduling words, not only disassembled mnemonics.
        return {b.splitlines()[0].strip():[l.strip() for l in b.splitlines() if re.match(r'\s*/\*',l)]
                for b in re.split(r'Function\s*:\s*',text)[1:]}
    inc=['-I'+p for p in include_paths(device_type='cuda')]+['-I'+sysconfig.get_path('include')]
    abi=f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}'
    common=['-std=c++20','-O3','-arch=sm_90a','-Xptxas=-v,-warn-spills','-Xcompiler=-fPIC',abi]
    report['nvcc_version']=run(['nvcc','--version'])
    assembly, parsed, encoded = {},{},{}
    for name,cu in sources.items():
        src,obj=build/(name+'.cu'),build/(name+'.o')
        src.write_bytes(cu.encode())
        log=run(['nvcc',*common,*inc,'-c',str(src),'-o',str(obj)])
        (build/(name+'.log')).write_bytes(log.encode())
        parsed[name]=resources(log)
        sass=run(['cuobjdump','--dump-sass',str(obj)])
        (build/(name+'.sass')).write_bytes(sass.encode())
        assembly[name]=sass_functions(sass)
        encoded[name]=encoded_functions(sass)
        res=run(['cuobjdump','--dump-resource-usage',str(obj)])
        (build/(name+'.resources.txt')).write_bytes(res.encode())
    report['ptxas']=parsed
    baseline=[n for n in assembly['base'] if 'gd_kernel' in n]
    assert baseline
    assert assembly['base']==assembly['default'],'default SASS differs'
    assert encoded['base']==encoded['default'],'default encoded scheduling words differ'
    assert parsed['base']==parsed['default'],'default resources differ'
    report['base_SASS_unchanged']=len(baseline)
    variants={}
    for name,st,key in keys:
        full=key[:9]+(0,)+(key[9:] if len(key)>9 else (0,))+(1,)
        token='gda_kernelI'+''.join(f'Li{x}E' for x in full)
        names=[n for n in parsed['gda'] if token in n]
        assert len(names)==1,(token,names)
        sym=names[0]
        r=dict(parsed['gda'][sym],dynamic_smem=key[2]*key[3]*st*1536,symbol=sym)
        lines=assembly['gda'][sym]
        r['SASS_instructions']=len(lines)
        r['SASS_memory_ops']={op:sum(bool(re.search(r'\b'+op+r'(?:\.|\b)',l)) for l in lines)
                              for op in ('LDG','LDGSTS','LDS','LDL','STL')}
        variants[f'{name}/{st}']=r
    report['variants']=variants
    # Full PyTorch binding host compile + shared-object link, C++20 on BOTH sides.
    host=build/'gda.cpp'
    exports=('gd_run','gd_info','gd_unpack','gdr_run','gdr_info','gdc_run','gd_unpack_cat')
    host.write_bytes((ns['GD_CPP_SRC']+'\nPYBIND11_MODULE(gda_gate,m) {\n'+
                     ''.join(f' m.def("{n}",&{n});\n' for n in exports)+'}\n').encode())
    import os
    cxx=os.environ.get('CXX','c++')
    run([cxx,'-std=c++20','-O2','-fPIC',abi,*inc,'-c',str(host),'-o',str(build/'host.o')])
    libs=['-L'+p for p in library_paths(device_type='cuda')]
    run([cxx,'-shared',str(build/'host.o'),str(build/'gda.o'),*libs,
         '-lc10','-ltorch','-ltorch_cpu','-ltorch_python','-lcudart','-lc10_cuda','-ltorch_cuda',
         '-o',str(build/'gda_gate.so')])
    report['host_CXX20_link']=True
    for name,r in variants.items():
        assert r['registers']<=128 and r['stack']==r['spill_stores']==r['spill_loads']==0,(name,r)
        assert r['SASS_memory_ops']['LDL']==r['SASS_memory_ops']['STL']==0,(name,r)
        assert r['SASS_memory_ops']['LDGSTS']>0 and r['SASS_memory_ops']['LDS']>0,(name,r)
    report['resource_gate']='PASS'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu-gate',action='store_true')
    ap.add_argument('--linux-gate',action='store_true')
    ap.add_argument('--json',type=Path)
    args=ap.parse_args()
    base=BASE.read_text(encoding='utf-8')
    if not (args.cpu_gate or args.linux_gate):
        out=generate(base)
        OUT.write_bytes(out.encode('utf-8'))
        print(f'generated {OUT.relative_to(ROOT)}: {len(out.encode())} bytes')
        return
    report={'status':'FAIL','CUDA':'NOT RUN'}
    try:
        out=OUT.read_text(encoding='utf-8')
        report['cpu']=cpu_gate(base,out)
        if args.linux_gate:
            linux_gate(base,out,report)
            report['CUDA']='PASS'
        report['status']='PASS'
    except Exception as ex:
        report['error']=repr(ex)
        raise
    finally:
        text=json.dumps(report,indent=2)
        print(text)
        if args.json:
            args.json.write_bytes((text+'\n').encode())


if __name__=='__main__':
    main()

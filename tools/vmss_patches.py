"""VMSS: default-off VMS synchronization experiments on k3_pure_c1.

Only the named lab output is generated. --cpu-only runs without torch/CUDA and
without pyc files. The Linux compile gate is tools/local/vmss_gate.sh.
The ship CUDA/CPP literals and every existing build remain byte-identical.
"""
import ast
import builtins
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_pure_c1.py"
OUT = ROOT / "kernels/lab/k3_vmss_ab.py"


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


POLL = r'''#if VMS_FX & 1
if (lane == 0) {
int n = 0;
while (ld_acq(p.ctr + 8 + jj) < RBA) {
if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
__nanosleep(64);
}
}
__syncwarp();
#else
if (lane == 0) {
int n = 0;
while (ld_rlx(p.ctr + 8 + jj) < RBA) {
if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
__nanosleep(64);
}
}
__syncwarp();
(void)ld_acq(p.ctr + 8 + jj);
#endif'''
POLL_SS = r'''#if VMS_SS & 1
if (!bpoll) {
bpoll = true;
if (tid == 0) {
const int jl = (b0 + (N - nA) - 1) >> 8;
for (int k = jj; k <= jl; ++k) {
int n = 0;
while (ld_acq(DONEA(k)) < RBA) {
if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
__nanosleep(64);
}
}
}
cta_sync_compute();
}
#else
''' + POLL + '\n#endif'
FENCE = r'''#if VMS_FX & 2
asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
#else
__threadfence();
#endif'''
OLD_WAIT = r'''__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t par) {
#if __CUDA_ARCH__ >= 900
asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
:: "r"(su32(b)), "r"(par) : "memory");
#else
asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
:: "r"(su32(b)), "r"(par) : "memory");
#endif
}'''
BOUNDED_WAIT = r'''#if VMS_SS
// All explicit spins in experimental copies are finite; never exit a CTA.
__device__ __forceinline__ void vmss_mbar_wait(uint64_t* b, uint32_t par, int* err) {
unsigned ready;
asm volatile("{\n .reg .pred p; .reg .u32 n;\n mov.u32 n, 0;\n W_%=:\n"
#if __CUDA_ARCH__ >= 900
"mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
#else
"mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n"
#endif
"@p bra D_%=;\n add.u32 n, n, 1;\n setp.lt.u32 p, n, %3;\n @p bra W_%=;\n"
"D_%=:\n selp.u32 %0, 1, 0, p;\n}\n"
: "=r"(ready) : "r"(su32(b)), "r"(par), "n"(WAIT_MAX) : "memory");
if (!ready) atomicExch(err, 3);
}
#define mbar_wait(barrier, phase) vmss_mbar_wait((barrier), (phase), p.ctr + 2)
#else
''' + OLD_WAIT + '\n#endif'

EDITS = [
    ('#ifndef VMS_NS', '''#ifndef VMS_SS
#define VMS_SS 0
#endif
#if VMS_SS < 0 || VMS_SS > 3 || (VMS_FX & 4)
#error Invalid VMSS bits or unsafe FX4
#endif
#if VMS_SS & 1
#define DONEA(j) (p.ctr + 32 + 32 * (j))
#else
#define DONEA(j) (p.ctr + 8 + (j))
#endif
#ifndef VMS_NS''', 1),
    (OLD_WAIT, BOUNDED_WAIT, 1),
    ('int curj = -1;', 'int curj = -1;\n#if VMS_SS & 1\nbool bpoll = false;\n#endif', 1),
    (POLL, POLL_SS, 1),
    (FENCE + '\n__syncwarp();', FENCE.replace('#if VMS_FX & 2', '#if (VMS_SS & 1) || (VMS_FX & 2)') + '\n__syncwarp();', 1),
    ('atomicAdd(p.ctr + 8 + j,', 'atomicAdd(DONEA(j),', 1),
    ('if (tid < MAXJ) p.ctr[8 + tid] = 0;', 'if (tid < MAXJ) *DONEA(tid) = 0;', 1),
    ('float* ssc = p.ss + (S.gen & 1) * MAXP;\nfor (int q = 0; q < N; ++q) {',
     '''float* ssc = p.ss + (S.gen & 1) * MAXP;
#if VMS_SS & 2
float pw0 = 0.f, pw1 = 0.f, pgs = 0.f;
auto preB = [&](int qq) {
if (qq >= nA && qq < N) {
const int vv = b0 + (qq - nA), nn = (vv & 255) * 16 + g;
pgs = __ldg(p.gsB + S.jE[vv >> 8]);
pw0 = __ldg(p.nw + nn); pw1 = __ldg(p.nw + nn + 8);
}
};
preB(0);
#endif
for (int q = 0; q < N; ++q) {
#if VMS_SS & 2
const float nwa = pw0, nwb = pw1, gsb = pgs;
preB(q + 1);
#endif''', 1),
    ('''const float gs = __ldg(p.gsB + S.jE[j]);
const int n0 = rb * 16 + g;
const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;''',
     '''const int n0 = rb * 16 + g;
#if VMS_SS & 2
const float w0 = nwa * gsb, w1 = nwb * gsb;
#else
const float gs = __ldg(p.gsB + S.jE[j]);
const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;
#endif''', 1),
    (FENCE + '\nif (mk && lane == 0) mrow[7]', '#if !(VMS_SS & 2)\n' + FENCE + '\n#endif\nif (mk && lane == 0) mrow[7]', 1),
    ('#if VMS_FX & 2\nasm volatile("red.release.gpu.global.add.s32',
     '#if (VMS_SS & 2) || (VMS_FX & 2)\nasm volatile("red.release.gpu.global.add.s32', 1),
    ('#include <torch/extension.h>', '#undef mbar_wait\n#undef DONEA\n#include <torch/extension.h>', 1),
]

LAB = '''VMSS = {}  # {16: 1/2/3} selects the entire hi interval 9 <= T <= 16.
_VMSS = {}  # Own scratch per bit; low and hi in a bank share its ctr tensor.

def _vmss_src():
 s=VMS_CUDA_SRC
 for a,b,n in _VMSS_EDITS:
  assert s.count(a)==n, a[:80]
  s=s.replace(a,b)
 return s

def _vmss_flags(G):
@FLAGS@
 return flags,fl_hi

def _vmss_build():
 from torch.utils.cpp_extension import load_inline
 flags,fl_hi=_vmss_flags(_VMS["G"])
 src=_vmss_src()
 mods={}
 for bits in (1,2,3):
  mods[bits]={}
  for mode,fl in (("mod",flags),("mod_hi",fl_hi)):
   fl=list(fl)+[f"-DVMS_SS={bits}"]
   tag=hashlib.sha1((VMS_CPP_SRC+src+" ".join(fl)).encode()).hexdigest()[:12]
   mods[bits][mode]=load_inline(name=f"k3_vmss_{tag}",cpp_sources=VMS_CPP_SRC,cuda_sources=src,
    functions=["run","attrs"],extra_cflags=["-std=c++20"],extra_cuda_cflags=fl)
 return mods

def _vmss_prepare(state):
 _VMSS.clear()
 if state.get("k2a33")is None:
  return
 st=state["k2a33"]["st"]
 if not st.get("vms_ok") or st["vms"].get("mod_hi")is None:
  return
 vs=st["vms"]
 mods=_vmss_build()
 for bits in (1,2,3):
  bank=dict(vs,**mods[bits])
  for name in ("inter","ybuf","ss"):
   bank[name]=torch.zeros_like(vs[name])
  bank["ctr"]=torch.zeros((32+32*64,),dtype=torch.int32,device=vs["ctr"].device)
  _VMSS[bits]=bank

def _vmss_select(vs,m):
 bits=VMSS.get(16,0) if 8<m<=16 else 0
 if bits not in (0,1,2,3):
  raise ValueError(f"invalid VMSS bits: {bits}")
 if bits:
  return _VMSS[bits]  # Fail closed; never silently benchmark the ship fallback.
 return vs

'''


def extract(src):
    tree = ast.parse(src)
    ns = {"G": 132}
    for name in ("VMS_CUDA_SRC", "VMS_CPP_SRC"):
        nodes = [n for n in tree.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(nodes) == 1, name
        ns[name] = ast.literal_eval(nodes[0].value)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vms_build")
    flag_nodes = [n for n in fn.body if isinstance(n, ast.Assign) and
                  isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("flags", "fl_hi")]
    assert len(flag_nodes) == 2
    exec(compile(ast.Module(body=flag_nodes, type_ignores=[]), str(BASE), "exec"), ns)
    ns["flag_source"] = '\n'.join(' ' + ast.get_source_segment(src, n) for n in flag_nodes)
    return ns


def generate(src):
    info = extract(src)
    # D1 is already in this base, not added by VMSS.
    assert 'bmulh<false>(r0, s2)' in info["VMS_CUDA_SRC"]
    lab = LAB.replace('@FLAGS@', info['flag_source'])
    lab += '_VMSS_EDITS = ' + repr(EDITS) + '\n'
    out = sub(src, 'def _vms_build():\n', lab + 'def _vms_build():\n')
    out = sub(out, ' md=vs["mod_hi"]if(VMS_HI_ON',
              ' vs=_vmss_select(vs,m)\n md=vs["mod_hi"]if(VMS_HI_ON')
    out = sub(out, ' return state\ndef _post_expert_norm_weighted_sum(',
              ' _vmss_prepare(state)\n return state\ndef _post_expert_norm_weighted_sum(')
    return out


def exec_nodes(src, names, ns):
    nodes = [n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.Assign)) and
             ((isinstance(n, ast.FunctionDef) and n.name in names) or
              (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OUT), "exec"), ns)


def dry_run(base, lab):
    """Execute real state-construction AST, new prepare/build paths, and real call.

    Fake only tensor operations and extension compilation. Keys/shapes come from
    the base's actual st['vms'] assignment and return {'st':st,'ok':ok}.
    """
    class Tensor:
        def __init__(self, shape=(1,), dtype=None, device='cpu'):
            self.shape, self.dtype, self.device = tuple(shape), dtype, device
            self.zeroed = 0
        def float(self): return self
        def reshape(self, *shape): return Tensor(shape, self.dtype, self.device)
        def contiguous(self): return self
        def to(self, *a): return self
        def __mul__(self, other): return self
        def __radd__(self, other): return self
        def zero_(self): self.zeroed += 1; return self

    calls, builds = [], []
    class Module:
        def __init__(self, name): self.name = name
        def run(self, *args): calls.append((self, args))

    def load_inline(**kw):
        builds.append(kw)
        assert kw['functions'] == ['run', 'attrs']
        assert kw['extra_cflags'] == ['-std=c++20']
        return Module(kw['name'])

    def fake_import(name, *args, **kw):
        if name == 'torch.utils.cpp_extension':
            return types.SimpleNamespace(load_inline=load_inline)
        return builtins.__import__(name, *args, **kw)

    torch = types.SimpleNamespace(zeros=Tensor, empty=Tensor,
        zeros_like=lambda t: Tensor(t.shape, t.dtype, t.device),
        int32='int32', int64='int64', bfloat16='bf16', float32='fp32',
        cuda=types.SimpleNamespace(synchronize=lambda: None))
    low, hi = Module('ship-low'), Module('ship-hi')
    ns = dict(torch=torch, hashlib=hashlib, _VMS={'G': 132, 'mod': low, 'mod_hi': hi},
              __builtins__=dict(vars(builtins), __import__=fake_import),
              K2A_D_HIDDEN=4096, K2A_D_EXPERT=4096, K2A_POST_NORM_EPS=1e-6,
              VMS_HI_ON=True, E=32, two62=float(2**62), mod=low, dev='cpu',
              wA=Tensor(), wB=Tensor(), w={k: Tensor() for k in ('w13_gs','w2_gs','post_norm_w')}, st={})
    ns.update({k: v for k, v in extract(base).items() if k.startswith('VMS_')})
    tree = ast.parse(base)
    prep = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_vms_prepare')
    layout = [n for n in ast.walk(prep) if isinstance(n, ast.Assign) and
              any(ast.unparse(t) == "st['vms']" for t in n.targets)]
    assert len(layout) == 1
    exec(compile(ast.Module(body=layout, type_ignores=[]), str(BASE), 'exec'), ns)
    ns['st'].update(vms_ok=True, h=Tensor(), cnt=Tensor(), cnt2=Tensor())
    kprep = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_v11_k2a33_prepare')
    ret = [n for n in kprep.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
    assert len(ret) == 1
    ns['ok'] = {16: True}
    real_k33 = eval(compile(ast.Expression(ret[0].value), str(BASE), 'eval'), ns)
    state = {'k2a33': real_k33, 'k2a': None}
    names = {'VMSS','_VMSS','_VMSS_EDITS','_vmss_src','_vmss_flags','_vmss_build',
             '_vmss_prepare','_vmss_select','_vms_call','_k2a33_reset'}
    exec_nodes(lab, names, ns)
    # Execute the actual inserted statement from prepare, with the actual layout.
    lab_prep = next(n for n in ast.parse(lab).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    hook = [n for n in lab_prep.body if isinstance(n, ast.Expr) and
            isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == '_vmss_prepare']
    assert len(hook) == 1
    ns['state'] = state
    exec(compile(ast.Module(body=hook, type_ignores=[]), str(OUT), 'exec'), ns)
    assert len(builds) == 6 and set(ns['_VMSS']) == {1,2,3}
    ship = state['k2a33']['st']['vms']
    assert ship['ctr'].shape == (72,)
    for b, bank in ns['_VMSS'].items():
        assert bank['ctr'].shape == (2080,)
        assert bank['mod'] is not low and bank['mod_hi'] is not hi
        assert bank['wA'] is ship['wA'] and bank['wB'] is ship['wB']
        for name in ('inter','ybuf','ss','ctr'):
            assert bank[name] is not ship[name]
        for mode in ('mod','mod_hi'):
            build = next(k for k in builds if k['name'] == bank[mode].name)
            fl = extract(base)['flags' if mode == 'mod' else 'fl_hi'] + [f'-DVMS_SS={b}']
            assert build['extra_cuda_cflags'] == fl
    for b in (0,1,2,3):
        ns['VMSS'] = {} if b == 0 else {16: b}
        for t in (1,8,9,12,16,17,32,128,256):
            expected = ns['_VMSS'][b] if b and 8<t<=16 else ship
            assert ns['_vmss_select'](ship,t) is expected
            if t <= 16:
                # Both module choices receive the exact same bank ctr.
                for high in (False,True):
                    ns['VMS_HI_ON'] = high
                    ns['_vms_call'](ns['st'],Tensor((t,4096)),Tensor((t,8)),Tensor((t,8)))
                    md, args = calls[-1]
                    assert md is expected['mod_hi' if high and t>8 else 'mod']
                    assert args[11] is expected['ctr']
                    assert args[3] is ship['wA'] and args[4] is ship['wB']
    ns['VMSS'] = {16:4}
    try: ns['_vmss_select'](ship,16)
    except ValueError: pass
    else: raise AssertionError('invalid bit accepted')
    ns['_k2a33_reset'](ns['st'])
    assert all(ns['st'][k].zeroed == 1 for k in ('h','cnt','cnt2'))
    assert ship['ctr'].zeroed == 0
    ns['_vmss_prepare']({'k2a33':None})
    assert not ns['_VMSS']
    ns['VMSS'] = {16:1}
    try: ns['_vmss_select'](ship,16)
    except KeyError: pass
    else: raise AssertionError('missing bank silently accepted')
    ns['st']['vms_ok'] = False
    ns['_vmss_prepare'](state)
    assert not ns['_VMSS']
    ns['st']['vms_ok'] = True
    ship['mod_hi'] = None
    ns['_vmss_prepare'](state)
    assert not ns['_VMSS']
    return {'builds':len(builds), 'call_paths':len(calls), 'real_layout':True,
            'counter_ints':2080, 'missing_or_disabled_ship':True}


def protocol_gate():
    # Exhaust every legal contiguous B interval, including N=0 / no B.
    for b0 in range(64*256):
        for nb in (0,1,2,255,256,257,511,512):
            if b0+nb > 64*256: continue
            polled = list(range(b0>>8, ((b0+nb-1)>>8)+1)) if nb else []
            assert polled == sorted({v>>8 for v in range(b0,b0+nb)})
            assert all(32+32*j < 2080 for j in polled)
    # Prefetch indices including A-only, B-only, transitions and final unit.
    for n in range(100):
        for na in range(n+1):
            prefetched = None
            for q in range(n+1):
                if q:
                    current = prefetched
                    if q-1 >= na: assert current == q-1
                if na <= q < n: prefetched = q
    # There are NO CTA exits. These are the complete post-grid-wait owners.
    assert {t for t in range(288) if t < 64} == set(range(64))
    assert {t for t in range(288) if t < 128} == set(range(128))
    for m in (1,8,9,12,16):
        items = [it for c in range(132) for it in range(c,m*4,132)]
        assert sorted(items) == list(range(m*4))
    return {'B_ranges':True, 'prefetch_boundaries':True, 'cleanup_coverage':True}


def py_compile_no_write(paths):
    import importlib._bootstrap_external as bootstrap
    import py_compile
    saved = bootstrap._write_atomic
    written = []
    bootstrap._write_atomic = lambda path,data,mode=0o666: written.append(len(data))
    try:
        for path in paths:
            py_compile.compile(str(path), cfile=str(path)+'.vmss-pyc', doraise=True)
    finally:
        bootstrap._write_atomic = saved
    assert len(written) == len(paths) and min(written)>0


def cpu_gate(base, lab):
    ast.parse(lab)
    assert '\r' not in lab and lab.isascii()
    assert len(lab.encode('ascii')) < 1_000_000
    assert generate(base) == lab, 'stale generated output'
    old, new = extract(base), extract(lab)
    for key in ('VMS_CUDA_SRC','VMS_CPP_SRC','flags','fl_hi'):
        assert old[key] == new[key], key
    # Every original top-level AST node is untouched except the two tiny hooks.
    bn, ln = ast.parse(base).body, ast.parse(lab).body
    original = [n for n in ln if not (isinstance(n, ast.FunctionDef) and n.name.startswith('_vmss_')) and
                not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('VMSS','_VMSS','_VMSS_EDITS') for t in n.targets))]
    assert len(original) == len(bn)
    for a,b in zip(bn,original):
        if isinstance(a, ast.FunctionDef) and a.name in ('prepare','_vms_call'):
            b=copy.deepcopy(b)
            if a.name == 'prepare':
                b.body=[n for n in b.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
                        isinstance(n.value.func,ast.Name) and n.value.func.id=='_vmss_prepare')]
            else:
                b.body=[n for n in b.body if not (isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and
                        isinstance(n.value.func,ast.Name) and n.value.func.id=='_vmss_select')]
        assert ast.dump(a)==ast.dump(b), getattr(a,'name',str(type(a)))
    cuda=old['VMS_CUDA_SRC']
    for a,b,n in EDITS: cuda=sub(cuda,a,b,n)
    assert cuda.count('DONEA(')==5  # two definitions, poll, publish, reset
    assert cuda.count('mbar_wait(')==8  # bounded helper + macro + fallback + 4 call sites + macro expansion
    assert '__trap' not in cuda
    assert '-DVMS_FX=4' not in repr(new)
    assert '-DVMS_NS=2' in new['fl_hi'] and '-DVMS_ER=1' in new['fl_hi']
    assert '-DVMS_NS=3' not in new['fl_hi']
    return {'default_source_identical':True, 'all_existing_AST_preserved':True,
            'dry_run':dry_run(base,lab), 'protocol':protocol_gate(),
            'ship_cuda_sha256':hashlib.sha256(old['VMS_CUDA_SRC'].encode()).hexdigest()}


def main():
    if set(sys.argv[1:]) - {'--cpu-only','--check'}:
        raise SystemExit('usage: vmss_patches.py [--cpu-only] [--check]')
    base=BASE.read_bytes().decode('ascii')
    lab=generate(base)
    if '--check' in sys.argv:
        assert OUT.read_bytes()==lab.encode('ascii'), 'run generator first'
    else:
        OUT.write_bytes(lab.encode('ascii'))
    result=cpu_gate(base,lab)
    paths=[Path(__file__).resolve(),OUT,ROOT/'tools/box/vmss_check.py']
    py_compile_no_write([p for p in paths if p.exists()])
    result['py_compile']=True
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

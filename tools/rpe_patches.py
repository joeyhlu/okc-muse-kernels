"""RPE: host-only programmatic router join; no device arithmetic edits.

Only writes kernels/lab/k3_rpe_ab.py. --check is read-only (including the CPU
dry run); --lines prints the exact generated edit ranges. GPU gates are in
tools/box/rpe_check.py and compile/SASS gates in tools/local/rpe_gate.sh.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_pure_c1.py"
OUT = ROOT / "kernels/lab/k3_rpe_ab.py"

EVENT_HOST = r'''
// RPE: process-lifetime, per-device events; initialization must precede capture.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <mutex>
#include <unordered_map>
namespace @NS@ {
static std::mutex event_mutex;
static std::unordered_map<int, cudaEvent_t> events;
static cudaEvent_t event(bool create) {
  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(event_mutex);
  auto it = events.find(device);
  if (it != events.end()) return it->second;
  TORCH_CHECK(create, "RPE event not initialized in prepare for this device");
  cudaStreamCaptureStatus status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(), &status));
  TORCH_CHECK(status == cudaStreamCaptureStatusNone, "RPE event creation during capture");
  cudaEvent_t ev = nullptr;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  events.emplace(device, ev);
  return ev;
}
}
void vf_router_event_init() { (void)@NS@::event(true); }
void vf_router_join() {
  C10_CUDA_CHECK(cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(), @NS@::event(false), 0));
}
void vf_router32_pe(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int m = (int)x.size(0);
  TORCH_CHECK(x.is_cuda() && w.device() == x.device() && part.device() == x.device(), "RPE devices");
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && part.scalar_type() == at::kFloat &&
    x.is_contiguous() && w.is_contiguous() && part.is_contiguous() && x.dim() == 2 && w.dim() == 2 &&
    x.size(1) == 8192 && w.size(0) == 256 && w.size(1) == 8192 && m >= 1 && m <= 8 &&
    part.numel() == 8L * m * 256 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
    reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "RPE router layout");
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeProgrammaticEvent;
  attr.val.programmaticEvent.event = @NS@::event(false);
  attr.val.programmaticEvent.flags = 0;
  attr.val.programmaticEvent.triggerAtBlockStart = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
@LAUNCH@
}
'''
DECL = ('\nvoid vf_router32_pe(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind);\n'
        'void vf_router_join();\nvoid vf_router_event_init();\n')
EXPORTS = ',"vf_router32_pe","vf_router_join","vf_router_event_init"'


def host(owner):
    choices = ([('kind == 2', 'vf_router32c_kernel<8>', 16, 128),
                ('kind == 1', 'vf_router32b_kernel<8>', 64, 128),
                ('m == 1', 'vf_router32_kernel<1>', 16, 256),
                (None, 'vf_router32_kernel<8>', 16, 256)] if owner == 'sbb' else
               [('m == 1', 'v8::router32_kernel<1>', 32, 256),
                (None, 'v8::router32_kernel<8>', 32, 256)])
    lines = ['  TORCH_CHECK(kind >= 0 && kind <= %d, "RPE kind");' % (2 if owner == 'sbb' else 0)]
    for i, (condition, kernel, gx, nt) in enumerate(choices):
        lines.append('  ' + (('if' if i == 0 else 'else if') + f' ({condition})' if condition else 'else') + ' {')
        lines.append(f'    cfg.gridDim = dim3({gx}, 8); cfg.blockDim = dim3({nt});')
        lines.append(f'    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, {kernel}, (const float*)x.data_ptr<float>(),')
        lines.append('      (const float*)w.data_ptr<float>(), part.data_ptr<float>(), m));\n  }')
    return EVENT_HOST.replace('@NS@', 'rpe_' + owner).replace('@LAUNCH@', '\n'.join(lines))


GLUE = '''# RPE BEGIN: call-time selector; prepare tests all supported router arms.
RPE = {}
def _rpe_owner(tokens, state):
 kind=VF_ROUTER.get(1 if tokens==1 else 8,"r32")
 if kind!="r32" and _VF["r8"]and _SBB1["mod"]is not None:
  return _SBB1["mod"], {"lean4":1,"lean5":2}.get(kind,0), "sbb"
 return state["v8"]["mod"], 0, "v8"
def _rpe_entry(tokens, state, mode, tail):
 bucket=1 if tokens==1 else 8 if 1<tokens<=8 else None
 if not RPE.get(bucket,0) or mode!="f32" or tail!="sep" or not V9_TAIL_PDL or _ft_flag(tokens) not in (4,8):
  return None
 mod,kind,owner=_rpe_owner(tokens,state)
 entry=state.get("rpe",{}).get((bucket,owner,kind))
 return entry if entry is not None and entry["ok"] and entry["mod"] is mod else None
def _rpe_logits(router_input,state,entry):
 part=torch.empty((8,router_input.shape[0],N_EXPERTS),dtype=torch.float32,device=router_input.device)
 entry["mod"].vf_router32_pe(router_input,state["router"],part,entry["kind"])
 return part
def _rpe_probe_call(state,ffn_input,router_input,data,entry,on,place):
 main_stream,side_stream=torch.cuda.current_stream(),state["side_stream"]
 side_stream.wait_stream(main_stream)
 if place!="side_first":
  expert_input=_v13_pooler(ffn_input,state["dense"],state)
 with torch.cuda.stream(side_stream):
  if on:
   logits=_rpe_logits(router_input,state,entry)
  else:
   logits=torch.empty((8,router_input.shape[0],N_EXPERTS),dtype=torch.float32,device=router_input.device)
   if entry["owner"]=="sbb":
    entry["mod"].vf_router32(router_input,state["router"],logits,entry["kind"])
   else:
    entry["mod"].v8_router32(router_input,state["router"],logits)
 if place=="side_first":
  expert_input=_v13_pooler(ffn_input,state["dense"],state)
 if on:
  entry["mod"].vf_router_join()
 else:
  main_stream.wait_stream(side_stream)
 if not torch.cuda.is_current_stream_capturing():
  logits.record_stream(main_stream)
 return (logits,)+_v8_tail(expert_input,logits,state,data)
def _rpe_same(a,b):
 return len(a)==len(b) and all(torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)) for x,y in zip(a,b))
def _rpe_prepare(state):
 state["rpe"]={}
 if state.get("v8") is None or not V9_TAIL_PDL:
  print("[rpe] OFF: v8 tail / PDL unavailable",flush=True)
  return
 dev=state["router"].device
 with torch.cuda.device(dev):
  owners=[("v8",state["v8"]["mod"],(0,))]
  if _VF["r8"]and _SBB1["mod"]is not None:
   owners.append(("sbb",_SBB1["mod"],(0,1,2)))
  for bucket in (1,8):
   for owner,mod,kinds in owners:
    for kind in kinds:
     entry={"mod":mod,"owner":owner,"kind":kind,"ok":False,"reason":None}
     state["rpe"][(bucket,owner,kind)]=entry
     try:
      if _ft_flag(bucket) not in (4,8):
       raise RuntimeError("tail_kernel3 unavailable")
      mod.vf_router_event_init()
      gen=torch.Generator(device=dev).manual_seed(7641+bucket)
      z=torch.randn((bucket,HIDDEN),generator=gen,device=dev)
      x=z.to(torch.bfloat16).contiguous()
      data={"local_start":torch.tensor([64],dtype=torch.int32,device=dev)}
      place=V8_ROUTER_PLACE.get(bucket,"side")
      main=torch.cuda.current_stream()
      probe=torch.cuda.Stream(device=dev)
      probe.wait_stream(main)
      with torch.cuda.stream(probe):
       ref=tuple(t.clone() for t in _rpe_probe_call(state,x,z,data,entry,False,place))
       got=_rpe_probe_call(state,x,z,data,entry,True,place)
      probe.synchronize()
      if not _rpe_same(ref,got):
       raise RuntimeError("eager partials / tail mismatch")
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph,stream=probe):
       got=_rpe_probe_call(state,x,z,data,entry,True,place)
      for trial in range(2):
       with torch.cuda.stream(probe):
        z.copy_(torch.randn((bucket,HIDDEN),generator=gen,device=dev))
        x.copy_(z.to(torch.bfloat16))
        ref=tuple(t.clone() for t in _rpe_probe_call(state,x,z,data,entry,False,place))
        graph.replay()
       probe.synchronize()
       if not _rpe_same(ref,got):
        raise RuntimeError("captured partials / tail mismatch on fresh input")
      entry["ok"]=True
     except Exception as ex:
      entry["reason"]=repr(ex)
      print(f"[rpe] OFF T={bucket} {owner}/{kind}: {ex!r}",flush=True)
     else:
      print(f"[rpe] ready T={bucket} {owner}/{kind}; selector defaults off",flush=True)
# RPE END
'''


def assignment(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node, ast.literal_eval(node.value)
    raise AssertionError(name)


def edits(base):
    changes = []
    for name, added in [('SBB1_HOST', host('sbb')), ('V8_TAIL_CUDA_SRC', host('v8')),
                        ('SBB1_DECL', DECL), ('V8_TAIL_CPP_SRC', DECL)]:
        node, _ = assignment(base, name)
        old = ast.get_source_segment(base, node)
        # Keep the original literal intact, including every device-source byte.
        changes.append((name, old, old + '+r"""' + added + '"""'))
    changes += [
        ('sbb exports', 'functions=["sbb1_run","vf_router32"]', 'functions=["sbb1_run","vf_router32"' + EXPORTS + ']'),
        ('v8 exports', 'functions=["v8_tail","v8_router32"]', 'functions=["v8_tail","v8_router32"' + EXPORTS + ']'),
        ('combined exports', '  "v8_tail","v8_router32"],', '  "v8_tail","v8_router32"' + EXPORTS + '],'),
        ('sbb cache name', 'load_inline("sbb1_k3t",', 'load_inline("sbb1_rpe_"+hashlib.sha1((SBB1_DECL+SBB1_CU+SBB1_HOST).encode()).hexdigest()[:10],'),
        ('prepare', ' f" prepare took {time.time() - t0:.1f}s",flush=True)\n return state\n',
         ' f" prepare took {time.time() - t0:.1f}s",flush=True)\n _rpe_prepare(state)\n return state\n'),
        ('helpers', 'def kernel_fn(data,state):', GLUE + 'def kernel_fn(data,state):'),
        ('select', '   side_stream.wait_stream(main_stream)\n',
         '   rpe_entry=_rpe_entry(tokens,state,mode,tail)\n   side_stream.wait_stream(main_stream)\n'),
        ('launch', '    logits=_v8_logits(mode,ffn_input,router_input,state)\n',
         '    logits=_rpe_logits(router_input,state,rpe_entry)if rpe_entry is not None else _v8_logits(mode,ffn_input,router_input,state)\n'),
        ('join', '   main_stream.wait_stream(side_stream)\n',
         '   if rpe_entry is not None:\n    rpe_entry["mod"].vf_router_join()\n   else:\n    main_stream.wait_stream(side_stream)\n'),
    ]
    return changes


def build(base):
    out = base
    # Apply call-site edits before helper insertion so anchors remain unique.
    for label, old, new in sorted(edits(base), key=lambda e: e[0] == 'helpers'):
        assert out.count(old) == 1, (label, out.count(old))
        out = out.replace(old, new)
    return out


def literal_namespace(src):
    env = {}
    def literal(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        return ast.literal_eval(node)
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                env[n.targets[0].id] = literal(n.value)
            except (ValueError, TypeError, KeyError):
                pass
    return env


def module_sources(src):
    """Build exactly the three affected load_inline inputs, without importing the layer."""
    env = literal_namespace(src)
    tree = ast.parse(src)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('_k2a33_wg_specs', '_k2a36_variant_cases')]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<rpe-source-builders>', 'exec'), env)
    cases, attrs = env['_k2a36_variant_cases'](env['K2A33_VARIANTS'])
    k33 = env['K2A33_KERNEL_SRC'].replace('@CASES3@', cases).replace('@ATTRS3@', attrs).replace('@WGSPECS@', env['_k2a33_wg_specs']())
    sources = {
        'sbb': (env['SBB1_DECL'], env['SBB1_CU'] + env['SBB1_HOST'], '_sbb1_build'),
        'v8': (env['V8_TAIL_CPP_SRC'], env['V8_TAIL_CUDA_SRC'], '_v8_tail_build'),
        'v10': ('\n'.join(env[n] for n in ('K2A33_CPP_SRC', 'R1B_CPP_SRC', 'V8_TAIL_CPP_SRC')),
                '\n'.join((k33, env['R1B_CUDA_SRC'], env['V8_TAIL_CUDA_SRC'])), '_v10_build_all'),
    }
    result = {}
    for owner, (cpp, cu, fn) in sources.items():
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        call = next(n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'load_inline')
        kw = {k.arg: k.value for k in call.keywords}
        result[owner] = {'cpp': cpp, 'cuda': cu, 'functions': ast.literal_eval(kw['functions']),
                         'cuda_flags': ast.literal_eval(kw['extra_cuda_cflags'])}
    return result


def cpu_dry_run(base, generated):
    """Execute every new prepare helper with strict dictionaries from base AST.

    Tensor/stream operations are CPU fakes, not a CUDA correctness simulation.
    Missing state keys, module methods, arm selection, and failed capture gating
    are real Python errors. No permissive __getattr__ is used.
    """
    class Tensor:
        def __init__(self, shape=(), dtype='f32', device='cuda:0'):
            self.shape, self.dtype, self.device = tuple(shape), dtype, device
        def to(self, dtype): return Tensor(self.shape, dtype, self.device)
        def contiguous(self): return self
        def clone(self): return Tensor(self.shape, self.dtype, self.device)
        def view(self, dtype): return self
        def copy_(self, other): assert self.shape == other.shape; return self
        def record_stream(self, stream): pass
        def dim(self): return len(self.shape)
        def stride(self, i):
            from math import prod
            return prod(self.shape[i + 1:])

    class Stream:
        def __init__(self, *a, **k): pass
        def wait_stream(self, other): log.append('full_wait')
        def synchronize(self): pass

    class Scope:
        def __init__(self, capture=False): self.capture = capture
        def __enter__(self):
            if self.capture: cuda.capturing = True
        def __exit__(self, *args):
            if self.capture: cuda.capturing = False

    class Graph:
        def replay(self): log.append('replay')

    class Mod:
        def vf_router_event_init(self):
            log.append('init')
            if fail[0] == 'init': raise RuntimeError('injected event creation rejection')
        def vf_router32_pe(self, x, w, part, kind):
            assert part.shape == (8, x.shape[0], 256)
            log.append(('pe', kind))
            if cuda.capturing and fail[0] == 'launch': raise RuntimeError('injected capture attribute rejection')
        def vf_router_join(self):
            log.append('join')
            if cuda.capturing and fail[0] == 'wait': raise RuntimeError('injected capture wait rejection')
        def vf_router32(self, x, w, part, kind): log.append(('plain', kind))
        def v8_router32(self, x, w, part): log.append('r32')
        def v8_tail(self, *args): log.append('tail')

    class Generator:
        def __init__(self, **kw): pass
        def manual_seed(self, seed): return self

    log, fail = [], [None]
    cuda = SimpleNamespace(capturing=False, current_stream=lambda: Stream(), Stream=Stream,
                           stream=lambda s: Scope(), device=lambda d: Scope(), CUDAGraph=Graph,
                           graph=lambda g, **k: Scope(True), is_current_stream_capturing=lambda: cuda.capturing)
    torch = SimpleNamespace(cuda=cuda, Generator=Generator, float32='f32', bfloat16='bf16', int32='i32', uint8='u8',
                            empty=lambda shape, **k: Tensor(shape, **k), randn=lambda shape, **k: Tensor(shape, device=k['device']),
                            tensor=lambda data, **k: Tensor((len(data),), **k), equal=lambda a, b: fail[0] != 'mismatch')
    env = literal_namespace(base)
    env.update(torch=torch, print=lambda *a, **k: None, _SBB1={'mod': Mod(), 'st': None}, _VF={'r8': True}, FT_OK=True,
               _gd=lambda state, name, x: Tensor((x.shape[0], 4096), 'bf16'), VF_GDM=False)
    names = {'_v8_tail', '_ft_flag', '_v13_pooler', '_v8_logits'}
    nodes = [n for n in ast.parse(generated).body if isinstance(n, ast.FunctionDef) and (n.name.startswith('_rpe_') or n.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<rpe-real-functions>', 'exec'), env)
    env['RPE'] = {}
    # Extract the real state's dict keys rather than inventing a fixture schema.
    tree = ast.parse(base)
    def schema(function, target):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        node = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in n.targets) and isinstance(n.value, ast.Dict))
        return {ast.literal_eval(k): Tensor() for k in node.value.keys}
    state = schema('_r1_prepare', 'state')
    state['v8'] = schema('_v8_tail_prepare', 'v8')
    state['v8']['mod'] = Mod()
    state['router'] = Tensor((256, 8192))
    state['side_stream'] = Stream()
    state['dense'] = {'pooler': Tensor((4096, 8192), 'bf16')}
    env['_rpe_prepare'](state)
    assert len(state['rpe']) == 8 and all(e['ok'] for e in state['rpe'].values()), state['rpe']
    assert log.count('replay') == 16
    # Execute the actual changed kernel_fn branch in isolation, with real helper
    # calls; default-off must issue precisely the same operation trace as base.
    def branch(source, name):
        fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
        node = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                    and ast.unparse(n.test) == "place == 'main'")
        body = node.orelse
        wrapper = ast.parse('def ' + name + '(state, tokens, mode, tail, place, ffn_input, router_input, dense):\n pass').body[0]
        wrapper.body = body + [ast.Return(value=ast.Name(id='logits', ctx=ast.Load()))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), '<rpe-live-branch>', 'exec'), env)
        return env[name]
    ship_branch, rpe_branch = branch(base, 'ship_branch'), branch(generated, 'rpe_branch')
    for t in (1, 8):
        for place in ('side', 'side_first'):
            args = (state, t, 'f32', 'sep', place, Tensor((t, 8192), 'bf16'), Tensor((t, 8192)), state['dense'])
            log.clear(); ship_branch(*args); before = list(log)
            log.clear(); rpe_branch(*args); assert before == log, (before, log)
            env['RPE'] = {1: 1, 8: 1}
            log.clear(); rpe_branch(*args)
            assert log.count('full_wait') == 1 and log.count('join') == 1
            assert any(isinstance(v, tuple) and v[0] == 'pe' for v in log)
            env['RPE'] = {}
    for tokens in (1, 2, 8):
        for kind in ('lean', 'lean4', 'lean5', 'r32'):
            bucket = 1 if tokens == 1 else 8
            env['VF_ROUTER'][bucket] = kind
            assert env['_rpe_entry'](tokens, state, 'f32', 'sep') is None
            env['RPE'] = {1: 1, 8: 1}
            e = env['_rpe_entry'](tokens, state, 'f32', 'sep')
            assert e and e['kind'] == {'lean4': 1, 'lean5': 2}.get(kind, 0)
            assert e['owner'] == ('v8' if kind == 'r32' else 'sbb')
            for mode, tail in (('stock', 'sep'), ('f32', None), ('f32', 'cat')):
                assert env['_rpe_entry'](tokens, state, mode, tail) is None
            env['RPE'] = {}
    env['RPE'] = {1: 1, 8: 1}
    assert env['_rpe_entry'](16, state, 'f32', 'sep') is None
    for failure in ('init', 'launch', 'wait', 'mismatch'):
        fail[0] = failure
        env['_rpe_prepare'](state)
        assert all(not e['ok'] and e['reason'] for e in state['rpe'].values())
        assert env['_rpe_entry'](1, state, 'f32', 'sep') is None
        assert not cuda.capturing
    fail[0] = None
    env['_VF']['r8'] = False
    env['_rpe_prepare'](state)
    assert len(state['rpe']) == 2 and all(e['ok'] for e in state['rpe'].values())
    env['V9_TAIL_PDL'] = False
    env['_rpe_prepare'](state)
    assert not state['rpe']
    env['V9_TAIL_PDL'] = True
    env['FT_OK'] = False
    env['_rpe_prepare'](state)
    assert all(not e['ok'] for e in state['rpe'].values())
    state['v8'] = None
    env['_rpe_prepare'](state)
    assert not state['rpe']
    return {'status': 'PASS', 'ready_entries': 8, 'capture_replay_paths': 16, 'event_init_rejection': True,
            'default_off_operation_trace_identical': True,
            'capture_launch_rejection': True, 'capture_wait_rejection': True, 'mismatch_rejection': True,
            'state_schema': 'AST of real _r1_prepare / _v8_tail_prepare', 'cuda_execution': False}


def checks(base, generated):
    assert generated == build(base), 'stale generated file'
    compile(generated, str(OUT), 'exec')
    restored = generated
    applied = sorted(edits(base), key=lambda e: e[0] == 'helpers')
    for label, old, new in reversed(applied):
        assert restored.count(new) == 1, label
        restored = restored.replace(new, old)
    assert restored == base, 'default source restoration differs'
    b, g = literal_namespace(base), literal_namespace(generated)
    for k, v in b.items():
        if isinstance(v, str) and ('__global__' in v or k.endswith(('CPP_SRC', 'DECL'))):
            assert g[k] == v + (host('sbb') if k == 'SBB1_HOST' else host('v8') if k == 'V8_TAIL_CUDA_SRC' else DECL if k in ('SBB1_DECL', 'V8_TAIL_CPP_SRC') else ''), k
    # No hot loop, barrier, early return, or old host function was changed.
    assert not any(s in host('sbb') + host('v8') for s in ('__global__', 'griddepcontrol', '__trap'))
    for owner, spec in module_sources(generated).items():
        assert {'vf_router32_pe', 'vf_router_join', 'vf_router_event_init'} <= set(spec['functions']), owner
    return {'default_source_restoration': True, 'original_device_sources_identical': True,
            'base_sha256': hashlib.sha256(base.encode()).hexdigest(), 'cpu_dry_run': cpu_dry_run(base, generated)}


def changed_lines(base, generated):
    rows = []
    for label, old, new in edits(base):
        start = generated.index(new)
        common = 0
        while common < min(len(old), len(new)) and old[common] == new[common]: common += 1
        tail = 0
        while tail < min(len(old), len(new)) - common and old[-tail-1] == new[-tail-1]: tail += 1
        line = generated.count('\n', 0, start + common) + 1
        end = start + len(new) - tail
        rows.append({'bit': label, 'first': line, 'last': generated.count('\n', 0, end) + 1})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--lines', action='store_true')
    args = parser.parse_args()
    base = BASE.read_text(encoding='utf-8')
    generated = build(base)
    result = checks(base, generated)
    if args.check:
        assert OUT.read_text(encoding='utf-8') == generated
    else:
        OUT.write_bytes(generated.encode('utf-8'))
    if args.lines: result['changed_lines'] = changed_lines(base, generated)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

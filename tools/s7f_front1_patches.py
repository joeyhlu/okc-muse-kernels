#!/usr/bin/env bash
""":"
# Bash/Python polyglot: this is also the self-contained Linux compile gate.
# --cpu-only exercises the wrapper without compiling or writing another file.
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST="9.0a"
export PYTHONDONTWRITEBYTECODE=1
export MAX_JOBS="${MAX_JOBS:-4}"
front1_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--cpu-only" ]]; then
  shift
  exec python -B "$0" --check-only "$@"
fi
exec python -B "$0" --check-only --cuda-gate --json "$front1_here/s7f_front1_gate.json" "$@"
":"""
from __future__ import annotations

# Front1: exact top-8 on the existing router side stream, T=1 only.
# Python writes only box_out/s7/k3_front1_v1.py; --check-only writes nothing.
# FRONT1_ON=0 stock; 1 split route/norm; 2 same-module combined control.
# No layer import or GPU discovery in the generator/CPU dry run.

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "box_out/s7/k3_front1_v1.py"


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def node(tree, name):
    found = [n for n in tree.body if
             isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    require(len(found) == 1, f"missing/duplicate definition: {name}")
    return found[0]


def replace_once(text, old, new):
    require(text.count(old) == 1, f"anchor count {text.count(old)}: {old[:100]}")
    return text.replace(old, new, 1)


DECL = r'''
void front1_route(torch::Tensor lg, torch::Tensor bias, torch::Tensor ls,
                  torch::Tensor tw, torch::Tensor tid, torch::Tensor lid);
void front1_norm(torch::Tensor y, torch::Tensor nw, torch::Tensor xn, double eps, int64_t pdl);
void front1_control(torch::Tensor y, torch::Tensor lg, torch::Tensor nw, torch::Tensor bias,
                    torch::Tensor ls, torch::Tensor xn, torch::Tensor tw, torch::Tensor tid,
                    torch::Tensor lid, double eps, int64_t pdl);
'''

HOST = r'''
static cudaLaunchConfig_t front1_config(int nt, int pdl, cudaLaunchAttribute& attr) {
  cudaLaunchConfig_t c = {};
  c.gridDim = dim3(1); c.blockDim = dim3(nt);
  c.stream = at::cuda::getCurrentCUDAStream();
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  c.attrs = &attr; c.numAttrs = 1;
  return c;
}
static void front1_route_check(torch::Tensor lg, torch::Tensor bias, torch::Tensor ls,
                               torch::Tensor tw, torch::Tensor tid, torch::Tensor lid) {
  TORCH_CHECK(lg.is_cuda() && lg.is_contiguous() && lg.scalar_type() == at::kFloat &&
              lg.dim() == 3 && lg.size(0) == 8 && lg.size(1) == 1 && lg.size(2) == 256,
              "front1: exact FP32 [8,1,256] router partials required");
  TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == at::kFloat &&
              bias.numel() == 256 && ls.is_cuda() && ls.is_contiguous() &&
              ls.scalar_type() == at::kInt && ls.numel() == 1, "front1: bias/local_start");
  TORCH_CHECK(tw.is_cuda() && tid.is_cuda() && lid.is_cuda() &&
              tw.is_contiguous() && tid.is_contiguous() && lid.is_contiguous() &&
              tw.scalar_type() == at::kFloat && tid.scalar_type() == at::kInt && lid.scalar_type() == at::kInt &&
              tw.numel() == 8 && tid.numel() == 8 && lid.numel() == 8, "front1: routing outputs");
}
static void front1_norm_check(torch::Tensor y, torch::Tensor nw, torch::Tensor xn) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kBFloat16 &&
              y.dim() == 2 && y.size(0) == 1 && y.size(1) == 4096 &&
              nw.is_cuda() && nw.is_contiguous() && nw.scalar_type() == at::kBFloat16 && nw.numel() == 4096 &&
              xn.is_cuda() && xn.is_contiguous() && xn.scalar_type() == at::kBFloat16 && xn.numel() == 4096,
              "front1: exact BF16 [1,4096] norm inputs/output required");
}
void front1_route(torch::Tensor lg, torch::Tensor bias, torch::Tensor ls,
                  torch::Tensor tw, torch::Tensor tid, torch::Tensor lid) {
  front1_route_check(lg, bias, ls, tw, tid, lid);
  cudaLaunchAttribute at; auto c = front1_config(256, 0, at);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c, front1::route_kernel<false, 8, 8>,
    (const void*)nullptr, (long)0, (const float*)lg.data_ptr<float>(), (const float*)nullptr,
    (long)256, 8, (long)256, (const __nv_bfloat16*)nullptr,
    (const float*)bias.data_ptr<float>(), (const int*)ls.data_ptr<int>(), (__nv_bfloat16*)nullptr,
    tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(),
    0.f, (float)0.1767766952966369, 1.f, (float)1e-15));
}
void front1_norm(torch::Tensor y, torch::Tensor nw, torch::Tensor xn, double eps, int64_t pdl) {
  front1_norm_check(y, nw, xn);
  cudaLaunchAttribute at; auto c = front1_config(512, pdl, at);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c, front1::norm_kernel<false, 8, 8>,
    (const void*)y.data_ptr(), (long)4096, (const float*)nullptr, (const float*)nullptr,
    (long)0, 0, (long)0, (const __nv_bfloat16*)nw.data_ptr(), (const float*)nullptr,
    (const int*)nullptr, (__nv_bfloat16*)xn.data_ptr(), (float*)nullptr, (int*)nullptr, (int*)nullptr,
    (float)eps, (float)0.1767766952966369, 1.f, (float)1e-15));
}
void front1_control(torch::Tensor y, torch::Tensor lg, torch::Tensor nw, torch::Tensor bias,
                    torch::Tensor ls, torch::Tensor xn, torch::Tensor tw, torch::Tensor tid,
                    torch::Tensor lid, double eps, int64_t pdl) {
  front1_norm_check(y, nw, xn); front1_route_check(lg, bias, ls, tw, tid, lid);
  cudaLaunchAttribute at; auto c = front1_config(768, pdl, at);
  C10_CUDA_CHECK(cudaLaunchKernelEx(&c, front1::control_kernel<false, 8, 8>,
    (const void*)y.data_ptr(), (long)4096, (const float*)lg.data_ptr<float>(), (const float*)nullptr,
    (long)256, 8, (long)256, (const __nv_bfloat16*)nw.data_ptr(),
    (const float*)bias.data_ptr<float>(), (const int*)ls.data_ptr<int>(), (__nv_bfloat16*)xn.data_ptr(),
    tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(),
    (float)eps, (float)0.1767766952966369, 1.f, (float)1e-15));
}
'''


def cuda_source(base):
    """Slice the actual tail3: no rewrite of its arithmetic or reduction order."""
    tree = ast.parse(base)
    tail = ast.literal_eval(node(tree, "V8_TAIL_CUDA_SRC").value)
    begin = tail.index("template <int EPT, bool CG>\n")
    kbegin = tail.index("template <bool Y32, int NSMAX, int RW>\n", begin)
    kend = tail.index("template <int MT>\n", kbegin)
    original = tail[kbegin:kend].rstrip()
    sig, body = original.split("constexpr int EPT = NE / (32 * RW);", 1)
    route_start = body.index("if (warp >= NT / 32) {\n")
    norm_start = body.index("const uint4 wq =", route_start)
    router = body[route_start + len("if (warp >= NT / 32) {\n"):norm_start]
    require(router.endswith("return;\n}\n"), "tail3 router scope changed")
    router = router[:-2]  # remove only the outer routing-role closing brace
    norm = body[norm_start:]
    require(norm.endswith("\n}"), "tail3 norm scope changed")
    norm = norm[:-2]
    route_sig = sig.replace("NT + 32 * RW", "32 * RW").replace("tail_kernel3", "route_kernel")
    norm_sig = sig.replace("NT + 32 * RW", "NT").replace("tail_kernel3", "norm_kernel")
    route = (route_sig + "constexpr int EPT = NE / (32 * RW);\n"
             "const int t = blockIdx.x, tx = threadIdx.x + NT, lane = tx & 31, warp = tx >> 5;\n"
             "__shared__ __align__(16) unsigned skey[NE];\n"
             "__shared__ __align__(16) float ssg[NE];\n" + router + "}\n")
    normal = (norm_sig +
              "const int t = blockIdx.x, tx = threadIdx.x, lane = tx & 31, warp = tx >> 5;\n"
              "__shared__ float red[NT / 32];\n" + norm + "\n}\n")
    control = original.replace("tail_kernel3", "control_kernel") + "\n"
    header = tail[:tail.index("namespace v8 {")]
    constants = "namespace front1 {\nconstexpr int D=4096, NE=256, TK=8, NT=512;\n"
    cuda = header + constants + tail[begin:kbegin] + control + route + normal + "}\n" + HOST
    require(router in cuda and norm in cuda, "arithmetic is not a verbatim slice")
    return cuda, {"router_body_sha256": hashlib.sha256(router.encode()).hexdigest(),
                  "norm_body_sha256": hashlib.sha256(norm.encode()).hexdigest()}


GLUE = r'''
# FRONT1 v1: 0 stock, 1 side top-8 + main norm, 2 same-module combined control.
FRONT1_ON=0
_FRONT1={"mod":None,"ok":False,"err":"","build_s":0.0}
def _front1_build():
 from torch.utils.cpp_extension import load_inline
 if _FRONT1["mod"] is not None:
  return _FRONT1["mod"]
 t=time.time()
 tag=hashlib.sha1((_FRONT1_CPP+_FRONT1_CUDA).encode()).hexdigest()[:12]
 _FRONT1["mod"]=load_inline(name=f"s7f_front1_{tag}",cpp_sources=_FRONT1_CPP,cuda_sources=_FRONT1_CUDA,
 functions=["front1_route","front1_norm","front1_control"],
 extra_cuda_cflags=["-O3","-lineinfo","-gencode=arch=compute_90a,code=sm_90a"],verbose=False)
 _FRONT1["build_s"]=time.time()-t
 return _FRONT1["mod"]
def _front1_select(state,tokens,mode,place,tail):
 if FRONT1_ON not in (1,2) or not _FRONT1["ok"] or state.get("v8") is None:
  return 0
 return FRONT1_ON if tokens==1 and mode=="f32" and place in ("side","side_first") and tail=="sep" else 0
def _front1_route(state,logits,data):
 v8=state["v8"]
 dev=logits.device
 tw=torch.empty((1,TOP_K),dtype=torch.float32,device=dev)
 tid=torch.empty((1,TOP_K),dtype=torch.int32,device=dev)
 lid=torch.empty((1,TOP_K),dtype=torch.int32,device=dev)
 _FRONT1["mod"].front1_route(logits,v8["bias"],data["local_start"],tw,tid,lid)
 return tw,tid,lid
def _front1_finish(y,state,routing):
 v8=state["v8"]
 xn=torch.empty((1,D_EXPERT),dtype=torch.bfloat16,device=y.device)
 _FRONT1["mod"].front1_norm(y,v8["nw"],xn,RMS_EPS,1 if V9_TAIL_PDL else 0)
 if not torch.cuda.is_current_stream_capturing():
  for tensor in routing:
   tensor.record_stream(torch.cuda.current_stream())
 return (xn,)+routing
def _front1_control(y,logits,state,data):
 v8=state["v8"]
 dev=y.device
 xn=torch.empty((1,D_EXPERT),dtype=torch.bfloat16,device=dev)
 tw=torch.empty((1,TOP_K),dtype=torch.float32,device=dev)
 tid=torch.empty((1,TOP_K),dtype=torch.int32,device=dev)
 lid=torch.empty((1,TOP_K),dtype=torch.int32,device=dev)
 _FRONT1["mod"].front1_control(y,logits,v8["nw"],v8["bias"],data["local_start"],xn,tw,tid,lid,
 RMS_EPS,1 if V9_TAIL_PDL else 0)
 return xn,tw,tid,lid
def _front1_prepare(state,t0):
 _FRONT1["ok"]=False
 _FRONT1["err"]=""
 if state.get("v8") is None or not FT_OK:
  _FRONT1["err"]="stock tail3 unavailable"
  return
 if time.time()-t0>250.0:
  _FRONT1["err"]="prepare time guard"
  return
 try:
  _front1_build()
  v8=state["v8"]
  dev=v8["bias"].device
  gen=torch.Generator(device=dev).manual_seed(73137)
  main,side=torch.cuda.current_stream(),state["side_stream"]
  for ls in (0,31,224,255):
   data={"local_start":torch.tensor([ls],dtype=torch.int32,device=dev)}
   for zero in (False,True):
    y=torch.randn((1,D_EXPERT),generator=gen,device=dev).to(torch.bfloat16)
    lg=torch.randn((8,1,N_EXPERTS),generator=gen,device=dev)
    if zero: lg.zero_()
    ref=_v8_tail(y,lg,state,data,ftf=8)
    control=_front1_control(y,lg,state,data)
    side.wait_stream(main)
    with torch.cuda.stream(side):
     routing=_front1_route(state,lg,data)
    main.wait_stream(side)
    got=_front1_finish(y,state,routing)
    torch.cuda.synchronize()
    if not all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(ref,got,control)):
     raise RuntimeError(f"front1 intermediate bitwise mismatch ls={ls} zero={zero}")
  _FRONT1["ok"]=True
 except Exception as ex:
  _FRONT1["err"]=repr(ex)[-600:]
 print(f"[front1] {'LIVE' if _FRONT1['ok'] else 'OFF'} bitwise gate; {_FRONT1['err']}",flush=True)
'''


def generate(base, enable=0):
    cuda, proof = cuda_source(base)
    src = replace_once(base, ' _tinys_prepare()\n', ' _tinys_prepare()\n _front1_prepare(state,t0)\n')
    src = replace_once(src, '  vgr=_vfr_fused(state,ffn_input,router_input)',
                       '  front1_arm=_front1_select(state,tokens,mode,place,tail)\n'
                       '  front1_routing=None\n  vgr=_vfr_fused(state,ffn_input,router_input)')
    src = replace_once(src, '    logits=_v8_logits(mode,ffn_input,router_input,state)\n',
                       '    logits=_v8_logits(mode,ffn_input,router_input,state)\n'
                       '    if front1_arm==1:\n     front1_routing=_front1_route(state,logits,data)\n')
    src = replace_once(src, '   expert_x,topk_weights,topk_ids,local_ids=_v8_tail(expert_input,logits,state,data)\n',
                       '   if front1_routing is not None:\n'
                       '    expert_x,topk_weights,topk_ids,local_ids=_front1_finish(expert_input,state,front1_routing)\n'
                       '   elif front1_arm==2:\n'
                       '    expert_x,topk_weights,topk_ids,local_ids=_front1_control(expert_input,logits,state,data)\n'
                       '   else:\n'
                       '    expert_x,topk_weights,topk_ids,local_ids=_v8_tail(expert_input,logits,state,data)\n')
    src += '\n_FRONT1_CPP=' + repr(DECL) + '\n_FRONT1_CUDA=' + repr(cuda) + '\n'
    src += GLUE.replace('FRONT1_ON=0\n', f'FRONT1_ON={enable}\n', 1)
    lines = src.splitlines()
    lines[1] = f'# nonce front1-v1 (exact side top-8; default selector {enable})'
    src = '\n'.join(lines) + '\n'
    require(len(src.encode()) < 1_000_000, 'submission exceeds 1 MB')
    compile(src, str(OUT), 'exec')
    tree0, tree1 = ast.parse(base), ast.parse(src)
    changed = {'prepare', 'kernel_fn'}
    for n in tree0.body:
        if isinstance(n, ast.FunctionDef) and n.name not in changed:
            require(ast.dump(n) == ast.dump(node(tree1, n.name)), f'untouched function changed: {n.name}')
    # No existing CUDA module changes; only the new independent module is compiled.
    for n in tree0.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    require(ast.literal_eval(node(tree1, t.id).value) == n.value.value, f'base literal changed: {t.id}')
    return src, proof


class Tensor:
    """Strict metadata tensor for Python wiring, never a GPU/math simulator."""
    def __init__(self, shape, dtype='float32', device='cpu-dry'):
        self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        self.dtype, self.device = dtype, device
    def to(self, dtype): return Tensor(self.shape, dtype, self.device)
    def float(self): return self.to('float32')
    def contiguous(self): return self
    def detach(self): return self
    def __sub__(self, other): return Tensor(self.shape, self.dtype, self.device)
    def dim(self): return len(self.shape)
    def stride(self, i):
        n = 1
        for d in self.shape[i+1:]: n *= d
        return n
    def record_stream(self, stream): stream.events.append('record')
    def zero_(self): return self


class Stream:
    def __init__(self, cuda, name): self.cuda, self.name, self.events = cuda, name, []
    def wait_stream(self, other): self.cuda.log.append(('wait', self.name, other.name))
    def __enter__(self):
        self.prev, self.cuda.current = self.cuda.current, self
        return self
    def __exit__(self, *args): self.cuda.current = self.prev


class FakeCUDA:
    def __init__(self):
        self.log, self.capturing = [], False
        self.current = Stream(self, 'main')
    def Stream(self, device): return Stream(self, 'side')
    def current_stream(self): return self.current
    def stream(self, stream): return stream
    def synchronize(self): pass
    def is_current_stream_capturing(self): return self.capturing


class FakeModule:
    def __init__(self, cuda): self.cuda = cuda
    def v8_tail(self, *args): self.cuda.log.append(('stock', self.cuda.current.name))
    def front1_route(self, *args):
        require(args[0].shape == (8,1,256), 'wrong router partial layout')
        self.cuda.log.append(('route', self.cuda.current.name))
    def front1_norm(self, *args):
        require(args[0].shape == (1,4096), 'wrong norm shape')
        self.cuda.log.append(('norm', self.cuda.current.name))
    def front1_control(self, *args): self.cuda.log.append(('control', self.cuda.current.name))


def cpu_dry_run(base, generated):
    """Execute real v8 state construction, all added prepare code and arm paths."""
    cuda = FakeCUDA()
    torch = types.SimpleNamespace(cuda=cuda, bfloat16='bfloat16', float32='float32', int32='int32')
    torch.empty = lambda shape, dtype='float32', device='cpu-dry': Tensor(shape, dtype, device)
    torch.randn = lambda shape, generator=None, device='cpu-dry': Tensor(shape, device=device)
    torch.tensor = lambda data, dtype, device: Tensor((len(data),), dtype, device)
    torch.cat = lambda ts, dim: Tensor((sum(t.shape[0] for t in ts), ts[0].shape[1]), ts[0].dtype)
    torch.Generator = lambda device: types.SimpleNamespace(manual_seed=lambda seed: object())
    torch.equal = lambda a,b: a.shape == b.shape and a.dtype == b.dtype
    module, captures = FakeModule(cuda), []
    def capture(**kw):
        captures.append(kw)
        require(kw['functions'] == ['front1_route','front1_norm','front1_control'], 'binding mismatch')
        return module
    import time
    ns = dict(torch=torch, hashlib=hashlib, time=time, load_inline=capture,
              _V10_MOD=module, V8_TAIL=True, FT_OK=True, V9_TAIL_PDL=True, TL1_TAIL2=True,
              D_EXPERT=4096, TOP_K=8, N_EXPERTS=256, RMS_EPS=1e-5, _ft_flag=lambda t:8)
    bt, gt = ast.parse(base), ast.parse(generated)
    # Exact constructor up to the second try (the original GPU self-test).
    constructor = copy.deepcopy(node(bt, '_v8_tail_prepare'))
    body = []
    for statement in constructor.body:
        body.append(statement)
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'v8' for t in statement.targets):
            break
    require(isinstance(body[-1], ast.Assign), 'v8 constructor anchor absent')
    constructor.body = body + [ast.Return(value=ast.Name(id='v8', ctx=ast.Load()))]
    ast.fix_missing_locations(constructor)
    exec(compile(ast.Module(body=[constructor, node(bt, '_v8_tail')], type_ignores=[]), str(BASE), 'exec'), ns)
    weights = {'pooler':Tensor((4096,8192),'bfloat16'), 'router':Tensor((256,8192)), 'router_bias':Tensor((256,))}
    state = {'pre_expert_norm':types.SimpleNamespace(_get_effective_weight=lambda: Tensor((4096,), 'bfloat16'))}
    state['v8'] = ns['_v8_tail_prepare'](weights, state)
    require(set(state['v8']) == {'mod','nw','bias','wcat'}, 'real v8 layout changed')
    # Copy the side_stream value from the real _r1_prepare state dictionary.
    r1 = node(bt, '_r1_prepare')
    values = [v for n in ast.walk(r1) if isinstance(n, ast.Dict) for k,v in zip(n.keys,n.values)
              if isinstance(k, ast.Constant) and k.value == 'side_stream']
    require(len(values) == 1, 'real side_stream accessor changed')
    state['side_stream'] = eval(compile(ast.Expression(values[0]), str(BASE), 'eval'), {'torch':torch,'device':'cpu-dry'})
    names = ['_FRONT1_CPP','_FRONT1_CUDA','FRONT1_ON','_FRONT1','_front1_build','_front1_select',
             '_front1_route','_front1_finish','_front1_control','_front1_prepare']
    nodes = [copy.deepcopy(node(gt,n)) for n in names]
    # Only the import is replaced in the CPU harness, as in padz/local build gates.
    builder = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == '_front1_build')
    require(isinstance(builder.body[0], ast.ImportFrom) and builder.body[0].module == 'torch.utils.cpp_extension', 'builder import changed')
    builder.body.pop(0)
    exec(compile(ast.Module(body=nodes,type_ignores=[]), str(OUT), 'exec'), ns)
    ns['_front1_prepare'](state, time.time())
    require(ns['_FRONT1']['ok'], 'new prepare dry run failed: '+ns['_FRONT1']['err'])
    ns['_front1_build']()
    require(len(captures)==1, 'module cache failed')
    require(sum(e==('route','side') for e in cuda.log)==8, 'route did not execute on existing side stream')
    require(sum(e==('norm','main') for e in cuda.log)==8, 'norm did not execute on main stream')
    arms=0
    for arm in (0,1,2,3):
        ns['FRONT1_ON']=arm
        for tokens in (1,2,8,16,32,128,256):
            for mode in ('f32','stock'):
                for place in ('side','side_first','main'):
                    for tail in ('sep','cat',None):
                        want=arm if arm in (1,2) and tokens==1 and mode=='f32' and place in ('side','side_first') and tail=='sep' else 0
                        require(ns['_front1_select'](state,tokens,mode,place,tail)==want, 'arm route mismatch')
                        arms+=1
    # Execute exceptional prepare paths and verify they really disable selection.
    for kind in ('missing_v8','stock_gate','time_guard','build_failure','check_failure'):
        ns['FT_OK']=kind!='stock_gate'
        saved=ns['_front1_build']
        old_equal=torch.equal
        if kind=='build_failure': ns['_front1_build']=lambda: (_ for _ in ()).throw(RuntimeError('injected compile failure'))
        if kind=='check_failure': torch.equal=lambda a,b:False
        ns['_front1_prepare']({'v8':None} if kind=='missing_v8' else state, time.time()-(251 if kind=='time_guard' else 0))
        require(not ns['_FRONT1']['ok'], 'failed prepare left arm enabled')
        ns['_front1_build'],torch.equal=saved,old_equal
    ns['FT_OK']=True
    ns['_front1_prepare'](state,time.time())
    # Capture branch: allocator record_stream must be skipped during capture.
    cuda.capturing=True
    ns['_front1_finish'](Tensor((1,4096),'bfloat16'),state,tuple(Tensor((1,8),d) for d in ('float32','int32','int32')))
    # Execute the ACTUAL modified kernel_fn front branch, not a model of its order.
    def find_front(statements):
        if any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='front1_arm'
               for t in n.targets) for n in statements):
            return statements
        for n in statements:
            for key in ('body','orelse','finalbody'):
                found=find_front(getattr(n,key,[]))
                if found: return found
        return None
    front=find_front(node(gt,'kernel_fn').body)
    require(front is not None,'actual kernel_fn split branch not found')
    front_code=compile(ast.Module(body=front,type_ignores=[]),str(OUT),'exec')
    ns.update(state=state,dense={},tail='sep',mode='f32',
              data={'local_start':Tensor((1,),'int32')})
    pipeline_cases=0
    for arm in (0,1,2):
        ns['FRONT1_ON']=arm
        for capture_mode in (False,True):
            cuda.capturing=capture_mode
            for place in ('side','side_first'):
                ns.update(tokens=1,place=place,ffn_input=Tensor((1,8192),'bfloat16'),router_input=Tensor((1,8192)))
                ns['_vfr_fused']=lambda *a:None
                def pooler(*a):
                    cuda.log.append(('pooler',cuda.current.name))
                    return Tensor((1,4096),'bfloat16')
                def logits(*a):
                    cuda.log.append(('router',cuda.current.name))
                    return Tensor((8,1,256))
                ns['_v13_pooler'],ns['_v8_logits']=pooler,logits
                cuda.log=[]
                exec(front_code,ns)
                events=cuda.log
                require(events.count(('wait','side','main'))==1 and events.count(('wait','main','side'))==1,
                        'fork/join count changed')
                if arm==1:
                    require(events.index(('router','side'))<events.index(('route','side'))<
                            events.index(('wait','main','side'))<events.index(('norm','main')),
                            'actual kernel_fn violated route/join/norm ordering')
                else:
                    require(not any(e[0] in ('route','norm') for e in events),'inactive arm executed split kernels')
                    require(('control' if arm==2 else 'stock','main') in events,'wrong combined control')
                require(ns['expert_x'].shape==(1,4096) and ns['local_ids'].dtype=='int32','pipeline outputs changed')
                pipeline_cases+=1
            ns.update(tokens=8,place='side_first',ffn_input=Tensor((8,8192),'bfloat16'),router_input=Tensor((8,8192)))
            ns['_vfr_fused']=lambda *a:(Tensor((8,4096),'bfloat16'),Tensor((8,8,256)))
            cuda.log=[]
            exec(front_code,ns)
            require(cuda.log==[('stock','main')],'T=8 fused route changed')
            pipeline_cases+=1
    cuda.capturing=False
    return dict(status='PASS', real_v8_keys=sorted(state['v8']), selector_cases=arms,
                prepare_cases=8, failure_paths=5, pipeline_cases=pipeline_cases, captured_build=captures[0],
                caveat='CPU wiring only; torch.equal checks metadata. GPU bitwise gate is retained, not executed.')


def protocol_gate(base, generated):
    cu, _ = cuda_source(base)
    require('bar.sync 1, %0;' in cu and 'bar.arrive 2, %0;' in cu and 'bar.sync 2, %0;' in cu, 'named barriers missing')
    require('const int t = blockIdx.x, tx = threadIdx.x + NT' in cu, 'routing virtual thread mapping changed')
    # Enumerate all actual thread owners, including the early-return routing warps.
    norm = {(tx*8+i) for tx in range(512) for i in range(8)}
    route = {((tx+512)-512) for tx in range(256)}
    require(norm == set(range(4096)) and route == set(range(256)), 'incomplete output coverage')
    require(set(range(32)) | set(range(32,256)) == set(range(256)), 'routing barrier coverage')
    tree = ast.parse(generated)
    k = ast.get_source_segment(generated,node(tree,'kernel_fn'))
    side = k.index('with torch.cuda.stream(side_stream):')
    route_pos = k.index('front1_routing=_front1_route')
    join = k.index('main_stream.wait_stream(side_stream)')
    finish = k.index('_front1_finish(expert_input')
    require(side < route_pos < join < finish, 'cross-stream producer/join/consumer order')
    return dict(status='PASS', norm_threads=512, route_threads=256, norm_elements=4096,
                route_elements=256, route_finish_threads=list(range(8)),
                barriers={'norm_1':'CTA 0, threads 0..511; no early exits',
                          'route_2':'CTA 0, threads 32..255 arrive then exit; 0..31 sync then select/write'},
                cleanup='No counters/flags/global reset/combine loops added; shared arrays overwritten each launch.')


def network_gate():
    network=((0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7),(0,1),(2,3),
             (4,5),(6,7),(2,4),(3,5),(1,4),(3,6),(1,2),(3,4),(5,6))
    rng=random.Random(73137)
    for case in range(2048):
        keys=[rng.randrange(1, 8 if case%2 else 2**32) for _ in range(256)]
        queues=[]
        for lane in range(32):
            q=[(keys[lane*8+j], lane*8+j) for j in range(8)]
            for i,j in network:
                if (-q[j][0],q[j][1]) < (-q[i][0],q[i][1]): q[i],q[j]=q[j],q[i]
            queues.append(q)
        selected=[]
        for _ in range(8):
            owner=min(range(32), key=lambda lane:(-queues[lane][0][0],lane))
            selected.append(queues[owner].pop(0))
            queues[owner].append((0, 999))
        require(selected==sorted([(k,j) for j,k in enumerate(keys)],key=lambda p:(-p[0],p[1]))[:8], 'top8 tie-order mismatch')
    return dict(status='PASS', cases=2048, note='same copied sorting network and lower-expert-ID tie break')


def changed_lines(base, generated):
    import difflib
    a,b=base.splitlines(),generated.splitlines()
    ranges=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if tag!='equal':
            ranges.append(dict(kind=tag,base_lines=[i+1,j],output_lines=[k+1,l]))
    tree=ast.parse(generated)
    refs={n.name:n.lineno for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_front1_')}
    refs.update({name:node(tree,name).lineno for name in ('_FRONT1_CPP','_FRONT1_CUDA','FRONT1_ON')})
    return dict(hunks=ranges,symbols=refs)


def accounting():
    return {
        'weight_stream_bytes_delta':0,
        'weight_stream_note':'No GD, router GEMV or MoE weight loop changed. Norm/bias reads below are unchanged.',
        'norm_per_token':{'y_read':8192,'nw_read':8192,'xn_write':8192,
                          'shared_requested_read':32768,'shared_write':64,
                          'before_after':'identical; 512 threads, 16 B from each input + 16 B output/thread'},
        'route_per_token':{'partials_read':8192,'bias_read':1024,'outputs_write':96,
                           'local_start_requested':128,'local_start_unique':4,
                           'shared_write_read':4096,
                           'before_after':'identical; 256 producer threads, 32 final selector threads'},
        'norm_source_ops_per_thread':{'before_after':'identical',
            'input_vector_loads':2,'square_accumulations':8,'shuffle_add_pairs':5,
            'shared_scalar_reduction_loads_adds':16,'rstd':1,'output_scale_bf16':8,'output_vector_stores':1},
        'route_source_ops_per_producer_thread':{'before_after':'identical',
            'coherent_partial_scalar_loads':8,'bias_loads':1,'partial_additions':8,
            'sigmoid_and_order_key':1,'shared_scalar_stores':2},
        'route_source_ops_per_finisher_thread':{'before_after':'same operations moved off main to side',
            'shared_vector_loads':4,'compare_exchange':19,'top8_rounds':8,
            'per_round':'1 redux, 1 ballot, 1 ffs, 2 shuffles, 21 shifts + 1 sentinel select',
            'weight_sum_adds':8,'ratio':1},
        'machine_instructions':'Compiler/SASS counts unavailable until Linux CUDA gate; source operations are not SASS counts.',
        'launches':{'before':1,'after':2,'critical_main':1,'added_side':1,'new_forks_joins':0},
        'scratch':'No new persistent counters, flags, partials or weight buffers. Same outputs; smem moves 2112 B combined -> 64 B norm + 2048 B route.'}


def cuda_gate(spec):
    """Compile only: load_inline's module contains no import-time GPU operations."""
    import os
    import shutil
    import subprocess
    import tempfile
    require(sys.platform=='linux','CUDA gate requires Linux under tools/local/cudaenv.sh')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
    require(shutil.which('nvcc') and shutil.which('cuobjdump'),'nvcc/cuobjdump missing from PATH')
    work=Path(tempfile.mkdtemp(prefix='s7f_front1_gate_'))
    os.environ['TORCH_EXTENSIONS_DIR']=str(work)
    from torch.utils.cpp_extension import load_inline
    flags=spec['extra_cuda_cflags']+['-Xptxas=-v']
    args=dict(spec,extra_cuda_cflags=flags,build_directory=str(work),verbose=True)
    module=load_inline(**args)
    sass=subprocess.check_output(['cuobjdump','-sass',module.__file__],text=True)
    resources=subprocess.check_output(['cuobjdump','-res-usage',module.__file__],text=True)
    (work/'front1.sass').write_bytes(sass.encode())
    (work/'front1.resources').write_bytes(resources.encode())
    chunks=re.split(r'Function\s*:\s*',sass)[1:]
    rows=[]
    for chunk in chunks:
        name,_,code=chunk.partition('\n')
        if not any(k in name for k in ('control_kernel','route_kernel','norm_kernel')):
            continue
        instructions=re.findall(r'/\*[0-9a-fA-F]+\*/\s+([^;]+);',code)
        require(instructions,'cuobjdump instruction parser found no instructions')
        require('ACQBULK' in code,'PDL wait missing: '+name)
        require('LDL' not in code and 'STL' not in code,'local memory instructions: '+name)
        # __ldcg predecessor-data reads must not precede the first wait.
        prefix=code.split('ACQBULK',1)[0]
        require(not re.search(r'LDG[^;]*STRONG\.GPU',prefix),'coherent input load above wait: '+name)
        rows.append(dict(function=name,static_instructions=len(instructions),
                         ld_global=sum('LDG' in i for i in instructions),
                         ld_shared=sum('LDS' in i for i in instructions),
                         st_shared=sum('STS' in i for i in instructions),
                         barriers=sum('BAR.' in i for i in instructions)))
    require(len(rows)==3,'expected exactly three instantiated front1 kernels')
    locals_=re.findall(r'LOCAL:\s*(\d+)',resources)
    require(locals_ and all(int(n)==0 for n in locals_),'nonzero/missing local-memory resources')
    norm_row=next(r for r in rows if 'norm_kernel' in r['function'])
    return dict(status='PASS',build_directory=str(work),module=module.__file__,kernels=rows,
                nvcc=subprocess.check_output(['nvcc','--version'],text=True),resources=resources,
                sass=str(work/'front1.sass'),gpu_execution=False,
                model_issue_budget_holds=norm_row['static_instructions']<=200,
                note='Static SASS instruction counts are not dynamic per-thread counts or timing evidence.')


def candidate_model():
    return {
        'measured':{'source':'notes/study/evo/s7/kernel_baseline.md:6,8,13,18,21,26',
                    'd1s_us':{'pooler':14.0,'router':5.8,'tail3':2.4},
                    'd1f_us':{'pooler':13.7,'router':5.7,'tail3':2.4}},
        'selected':{
            'kill_reasons_first':['Norm, rather than routing, may set the combined tail critical path.',
                                  'Side routing may contend with pooler or finish after its available slack.'],
            'mechanism':'Move the verbatim tail3 routing role to the already existing T=1 side stream; retain fork/join and pooler GD.',
            'estimate_label':'EST, conditional budget; not a measured gain or guaranteed bound',
            'norm_budget':'0.84 us edge + 800 dependent-load cycles/1520 + 800 issue cycles/1520 = 1.893 us',
            'saving_us':round(2.4-(.84+1600/1520),3),
            'clock_source':'standing rule 3: in-layer 1.51-1.53 GHz; use 1.52 GHz',
            'load_budget_source':'notes/lab.md:1173 R-mg-0 reports L2 round trip 600-800 cycles; 800 is a budget, not a cold-load guarantee',
            'issue_budget':'EST 4 norm warps/SMSP x at most 200 instructions = 800 slots; source operations in accounting(), SASS must validate',
            'side_slack':'EST d1s 14.0-(5.8+2.4+0.84)=4.96 us; d1f 13.7-(5.7+2.4+0.84)=4.76 us; durations are not aligned start/end measurements',
            'expected_value_pp':round((2.4-(.84+1600/1520))*.229,3),
            'critical_path':'Tail is after the main-stream join. Its routing cannot overlap pooler today; normal and routing roles DO overlap each other, hence the norm-dominance kill reason.',
            'closest_dead_rows':'R-study-b-2 / RPE / TAILFOLD; no new side stream, no programmatic event and no per-MoE-CTA top-k.',
            'falsifier':'Require >=0.5 us combined-tail versus norm-only standalone reduction, then paired in-layer controls; otherwise discard.'},
        'rejected':{
            'kill_reasons_first':['Single-SM data/issue throughput loses the present 16-SM fanout.',
                                  '1024 threads risk register pressure/spills.'],
            'idea':'One-CTA SBB with the existing 16-group floating-point reduction tree reproduced in shared memory.',
            'measured_us':2.5,'source':'notes/study/evo/s7/kernel_baseline.md:12,24',
            'estimate':'Unpriced: no defensible net saving after serializing 144 KiB onto one SM; a global flag-hop constant cannot price this DSMEM reduction.',
            'closest_dead_row':'SBBCS (8/4 CTA clusters); this would remove cluster communication entirely, but the bandwidth kill is likely.',
            'verdict':'DROP; no v2.'}}


def main():
    p=argparse.ArgumentParser(description='Front1 generator, CPU gates and compile-only Linux gate; no GPU execution.')
    p.add_argument('--enable',type=int,choices=(0,1,2),default=0)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--cuda-gate',action='store_true')
    p.add_argument('--json',type=Path)
    args=p.parse_args()
    report=dict(status='FAIL',cuda_compile='NOT RUN',gpu='NOT RUN')
    code=0
    try:
        base=BASE.read_bytes().decode('utf-8')
        src,proof=generate(base,args.enable)
        dry=cpu_dry_run(base,src)
        spec=dry.pop('captured_build')
        report.update(status='PASS',bytes=len(src.encode()),default_arm=args.enable,
                      base_sha256=hashlib.sha256(base.encode()).hexdigest(),proof=proof,
                      cpu_dry_run=dry,protocol=protocol_gate(base,src),top8=network_gate(),
                      output=str(OUT),accounting=accounting(),candidates=candidate_model(),
                      changed_lines=changed_lines(base,src))
        require(b'\r' not in Path(__file__).read_bytes(),'Linux gate must have LF line endings')
        if not args.check_only:
            OUT.parent.mkdir(parents=True,exist_ok=True)
            OUT.write_bytes(src.encode('utf-8'))
        elif args.enable==0 and OUT.exists():
            require(OUT.read_bytes()==src.encode('utf-8'),'generated file differs from deterministic rebuild')
        if args.cuda_gate:
            report['cuda_compile']=cuda_gate(spec)
    except Exception as ex:
        report['status']='FAIL'
        report['error']=repr(ex)
        code=1
    if args.json:
        args.json.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))
    return code


if __name__=='__main__':
    sys.exit(main())

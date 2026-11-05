"""S7 glue32: collapse SBB's logical cluster without changing its FP32 tree.

Only writes box_out/s7/k3_glue32_v{1,2}.py. The immutable row is the source.
v1: 1024 threads, one original 8-element lane per thread.
v2: 512 threads, two original lanes per thread (separate accumulators).
GLUE32_ON defaults to 0: 0=ship, 1=flattened cluster, 2=separately compiled ship twin.
The direct task's default-off requirement takes precedence over fact_s7's ON output.

Premortem: (1) fewer active SMs / register admission may cost the entire saving;
(2) the old DSM exchange can overlap other rows, whereas the new CTA barriers cannot.
The gross protocol opportunity is EST 0.52 us = MEASURED DSM flag 0.34 + cluster
barrier 0.18 (DO_NOT_REPEAT R-b4m-0). This is NOT a measured or guaranteed net win.
The standalone floor is 0.5 us; no in-layer gain is claimed before a paired test.
Closest kills: SBBCS kept DSM at cluster 4/8; R-bndx concerns T>=128; R-b1
duplicated the whole row in every consumer. This build does none of those.

Bytes per row (channel-wise gate, both weights present), unchanged:
post/pre norm weights 32768 B, alpha/beta 65536 B, residual 32768 B,
branch 16384 B, output 16384 B, new residual 32768 B = 196608 B.
Dense GEMM weight streams/stages/instructions are completely unchanged.
Old DSM payload: 16*15*16 = 3840 B/row. New DSM payload: zero.
New CTA-local shared stores: 32*16 + 16*16 + 8 = 776 B/row.
No global scratch, counters, flags, or persistent state. Every shared entry is
overwritten before use on each invocation, including CUDA graph replay.

Source-level instructions per original 8-element lane: 32 FFMA for moments,
24 FMUL before reduction (post weight, alpha, beta), 20 SHFL + 20 FADD;
the BF16 conversions/output arithmetic are copied verbatim from SBB.
v1 has one such lane/thread, v2 two. Old each thread additionally executes
8 LDS+8 FADD for two warp partials, then 64 LDS+64 FADD for 16 cluster partials
and both rsqrts. New only physical tid 0..15 does the former (one logical
rank each), and tid 0 does the latter and both rsqrts; all threads load 2 factors.
These are source operation counts, not SASS claims. The Linux gate records SASS.

Post-wait ownership/proof: virtual lane v=tid+pack*NT, NT=1024/NV, bijects
0..1023. lane 0 of each virtual warp stores warp_red[0..31][0..3]. All NT
threads reach barrier 1. Threads 0..15 combine warp pairs into partials[0..15]
and all NT reach barrier 2. Thread 0 sums ranks 0..15 in ship order, publishes
both factors, and all NT reach barrier 3. Each virtual lane stores its unique
8 branch/residual outputs. No early exit, global reset, async copy or cleanup.

CPU gates run without importing the row or torch: execute the REAL base
_sbb1_prepare with a strict metadata tensor double, then execute every added
build/prepare/selection path. Numerical tests compare the two reduction trees
using numpy FP32, including signed zeros/cancellation/wide magnitudes. GPU
self-tests remain in the generated files (eager, fresh inputs, poisoned outputs,
captured replay). Linux compile/SASS gates are in tools/local/s7f_glue32_gate.sh.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUTS = [ROOT / f"box_out/s7/k3_glue32_v{v}.py" for v in (1, 2)]


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sub(source, old, new, name):
    require(source.count(old) == 1, f"{name}: expected one exact anchor")
    return source.replace(old, new)


KERNEL = r'''
// Virtual rank/thread numbering is exactly SB_CS=16, SB_NT=64 in ship SBB.
template <int NV>
__global__ void __launch_bounds__(1024 / NV, 1) glue32_kernel(const SbbParams p) {
  constexpr int NT = 1024 / NV;
  const int tid = threadIdx.x, lane = tid & 31, row = blockIdx.x;
  __shared__ float warp_red[32][4];
  __shared__ float partials[16][4];
  __shared__ float factors[2];
  float pw[NV][8], qw[NV][8], bt[NV][8], st[NV][8], al[NV][8];
  float br[NV][8], val[NV][8], res[NV][8];
  if (p.trig_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for (int pack = 0; pack < NV; ++pack) {
    const int e0 = (tid + pack * NT) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      pw[pack][i] = 1.f; qw[pack][i] = 1.f; al[pack][i] = 1.f; bt[pack][i] = 1.f;
    }
    if (p.post_w) sb_ld8bf(p.post_w + e0, pw[pack]);
    if (p.pre_w) sb_ld8bf(p.pre_w + e0, qw[pack]);
    if (p.alpha) {
      if (p.gate_channel_wise) {
        sb_ld8f(p.alpha + e0, al[pack]); sb_ld8f(p.beta + e0, bt[pack]);
      } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) { al[pack][i] = p.alpha[0]; bt[pack][i] = p.beta[0]; }
      }
    }
    sb_ld8f(p.stream + (size_t)row * SB_H + e0, st[pack]);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  #pragma unroll
  for (int pack = 0; pack < NV; ++pack)
    sb_ld8bf(p.branch + (size_t)row * SB_H + (tid + pack * NT) * 8, br[pack]);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for (int pack = 0; pack < NV; ++pack) {
    float bsum = 0.f, rsum = 0.f, csum = 0.f, ssum = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float v = br[pack][i];
      float mv = v;
      if (p.post_w) mv *= pw[pack][i];
      const float r = al[pack][i] * st[pack][i];
      const float sb = bt[pack][i] * mv;
      val[pack][i] = v; res[pack][i] = r;
      bsum = fmaf(v, v, bsum);
      rsum = fmaf(r, r, rsum);
      csum = fmaf(r, sb, csum);
      ssum = fmaf(sb, sb, ssum);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      bsum += __shfl_xor_sync(0xffffffffu, bsum, o);
      rsum += __shfl_xor_sync(0xffffffffu, rsum, o);
      csum += __shfl_xor_sync(0xffffffffu, csum, o);
      ssum += __shfl_xor_sync(0xffffffffu, ssum, o);
    }
    if (lane == 0) {
      const int w = (tid + pack * NT) >> 5;
      warp_red[w][0] = bsum; warp_red[w][1] = rsum;
      warp_red[w][2] = csum; warp_red[w][3] = ssum;
    }
  }
  __syncthreads(); // Every virtual warp has published all four moments.
  if (tid < 16) {
    float4 mine = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
      mine.x += warp_red[2 * tid + w][0]; mine.y += warp_red[2 * tid + w][1];
      mine.z += warp_red[2 * tid + w][2]; mine.w += warp_red[2 * tid + w][3];
    }
    partials[tid][0] = mine.x; partials[tid][1] = mine.y;
    partials[tid][2] = mine.z; partials[tid][3] = mine.w;
  }
  __syncthreads(); // All 16 logical ranks are present; no DSM lifetime or fence.
  if (tid == 0) {
    float tb = 0.f, tr = 0.f, tc = 0.f, ts = 0.f;
    #pragma unroll
    for (int q = 0; q < 16; ++q) {
      tb += partials[q][0]; tr += partials[q][1]; tc += partials[q][2]; ts += partials[q][3];
    }
    const float post_rstd = rsqrtf(tb * (1.f / (float)SB_H) + p.post_eps);
    const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, ts, 2.f * tc), tr);
    const float pre_rstd = rsqrtf(fmaxf(pre_sum, 0.f) * (1.f / (float)SB_H) + p.pre_eps);
    factors[0] = post_rstd; factors[1] = pre_rstd;
  }
  __syncthreads(); // Every output owner sees both factors.
  const float post_rstd = factors[0], pre_rstd = factors[1];
  #pragma unroll
  for (int pack = 0; pack < NV; ++pack) {
    const int e0 = (tid + pack * NT) * 8;
    float ns[8], nz[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float n = val[pack][i] * post_rstd;
      if (p.post_w) n *= pw[pack][i];
      n = sb_rbf(n);
      const float u = fmaf(bt[pack][i], n, res[pack][i]);
      ns[i] = u;
      float z = u * pre_rstd;
      if (p.pre_w) z *= qw[pack][i];
      nz[i] = z;
    }
    const __nv_bfloat162 b0 = __floats2bfloat162_rn(nz[0], nz[1]), b1 = __floats2bfloat162_rn(nz[2], nz[3]);
    const __nv_bfloat162 b2 = __floats2bfloat162_rn(nz[4], nz[5]), b3 = __floats2bfloat162_rn(nz[6], nz[7]);
    uint4 q;
    q.x = *reinterpret_cast<const uint32_t*>(&b0); q.y = *reinterpret_cast<const uint32_t*>(&b1);
    q.z = *reinterpret_cast<const uint32_t*>(&b2); q.w = *reinterpret_cast<const uint32_t*>(&b3);
    *reinterpret_cast<uint4*>(p.out + (size_t)row * SB_H + e0) = q;
    sb_st8f(p.new_stream + (size_t)row * SB_H + e0, ns);
  }
}
'''

HOST = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void sbb1_run(torch::Tensor branch, torch::Tensor stream, torch::Tensor out, torch::Tensor new_stream, torch::Tensor router,
torch::Tensor post_w, torch::Tensor pre_w, torch::Tensor alpha, torch::Tensor beta, double post_eps,
double pre_eps, int64_t input_scale) {
  const int T = branch.size(0);
  TORCH_CHECK((T == 16 || T == 32) && input_scale == 0 && router.numel() == 0, "glue32: route");
  TORCH_CHECK(branch.is_cuda() && stream.is_cuda() && out.is_cuda() && new_stream.is_cuda(), "glue32: CUDA tensors");
  TORCH_CHECK(branch.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 &&
              stream.scalar_type() == at::kFloat && new_stream.scalar_type() == at::kFloat, "glue32: dtypes");
  TORCH_CHECK(branch.dim() == 2 && branch.size(1) == SB_H && stream.sizes() == branch.sizes() &&
              out.sizes() == branch.sizes() && new_stream.sizes() == branch.sizes(), "glue32: shape");
  TORCH_CHECK(branch.is_contiguous() && stream.is_contiguous() && out.is_contiguous() && new_stream.is_contiguous(), "glue32: layout");
  TORCH_CHECK((post_w.numel() == 0 || (post_w.numel() == SB_H && post_w.scalar_type() == at::kBFloat16)) &&
              (pre_w.numel() == 0 || (pre_w.numel() == SB_H && pre_w.scalar_type() == at::kBFloat16)), "glue32: norm weights");
  TORCH_CHECK((alpha.numel() == 0 || alpha.numel() == 1 || alpha.numel() == SB_H) &&
              beta.numel() == alpha.numel(), "glue32: gate shape");
  SbbParams p = {};
  p.branch = (const __nv_bfloat16*)branch.data_ptr(); p.stream = stream.data_ptr<float>();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.new_stream = new_stream.data_ptr<float>();
  p.post_w = post_w.numel() ? (const __nv_bfloat16*)post_w.data_ptr() : nullptr;
  p.pre_w = pre_w.numel() ? (const __nv_bfloat16*)pre_w.data_ptr() : nullptr;
  p.alpha = alpha.numel() ? alpha.data_ptr<float>() : nullptr;
  p.beta = beta.numel() ? beta.data_ptr<float>() : nullptr;
  p.gate_channel_wise = alpha.numel() == SB_H ? 1 : 0;
  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(T); cfg.blockDim = dim3(1024 / @NV@); cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr; cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, glue32_kernel<@NV@>, p));
}
'''

SUPPORT = '''
# glue32: default-off in-file bank; 1=virtual cluster, 2=ship module twin.
GLUE32_ON=0
_GLUE32={"mods":{},"bank":{},"err":None,"build_s":0.0}
def _glue32_sources():
 helpers=SBB1_CU.split("template <bool kPreStream, bool kRouter, bool kInputScale>")[0]
 return helpers+GLUE32_KERNEL+GLUE32_HOST
def _glue32_specs():
 decl=SBB1_DECL.split(" void vf_router32")[0]
 for arm,cu,cpp,fns in ((1,_glue32_sources(),decl,["sbb1_run"]),
                        (2,SBB1_CU+SBB1_HOST,SBB1_DECL,["sbb1_run","vf_router32"])):
  tag=hashlib.sha1((cpp+cu).encode()).hexdigest()[:12]
  yield arm,dict(name=f"glue32_{arm}_{tag}",cpp_sources=cpp,cuda_sources=cu,functions=fns,
                extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a"],verbose=False)
def _glue32_build():
 from torch.utils.cpp_extension import load_inline
 t=time.time()
 try:
  for arm,spec in _glue32_specs():
   _GLUE32["mods"][arm]=load_inline(**spec)
 except Exception as ex:
  _GLUE32["err"]=repr(ex)
 _GLUE32["build_s"]=time.time()-t
def _glue32_outputs(st,stream,branch):
 T=branch.shape[0]
 out=torch.full((T,HIDDEN),float("nan"),dtype=torch.bfloat16,device=branch.device)
 ns=torch.full((T,HIDDEN),float("nan"),dtype=torch.float32,device=branch.device)
 return out,ns,None
def _glue32_invoke(st,stream,branch,outs):
 pw,qw,al,bt,isc,pe,qe=st["w"]
 assert isc==0
 st["mod"].sbb1_run(branch.contiguous(),stream.contiguous(),outs[0],outs[1],st["e"],pw,qw,al,bt,pe,qe,isc)
def _glue32_selftest(st,refst,T):
 dev=st["e"].device
 gen=torch.Generator(device=dev).manual_seed(1931+T)
 stream=torch.randn((T,HIDDEN),generator=gen,device=dev)
 branch=torch.randn((T,HIDDEN),generator=gen,device=dev).to(torch.bfloat16)
 got=_glue32_outputs(st,stream,branch)
 ref=_glue32_outputs(refst,stream,branch)
 def compare():
  torch.cuda.synchronize()
  assert all(bool(torch.isfinite(got[i]).all()) and torch.equal(got[i],ref[i]) for i in (0,1)), "glue32 bitwise"
 # Fresh inputs and poisoned outputs; arithmetic and output coverage are separate gates.
 for scale in (1.,0.,0.03125,16.):
  stream.copy_(torch.randn((T,HIDDEN),generator=gen,device=dev)*scale)
  branch.copy_(torch.randn((T,HIDDEN),generator=gen,device=dev).to(torch.bfloat16)*scale)
  for out in got[:2]+ref[:2]: out.fill_(float("nan"))
  _glue32_invoke(refst,stream,branch,ref)
  _glue32_invoke(st,stream,branch,got)
  compare()
 # Warm up on a side stream, then capture the exact invocation with fresh inputs.
 side=torch.cuda.Stream()
 side.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(side):
  _glue32_invoke(st,stream,branch,got)
 torch.cuda.current_stream().wait_stream(side)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  _glue32_invoke(st,stream,branch,got)
 for scale in (0.,2.,0.25):
  stream.copy_(torch.randn((T,HIDDEN),generator=gen,device=dev)*scale)
  branch.copy_(torch.randn((T,HIDDEN),generator=gen,device=dev).to(torch.bfloat16)*scale)
  for out in got[:2]: out.fill_(float("nan"))
  _glue32_invoke(refst,stream,branch,ref)
  graph.replay()
  compare()
 return True
def _glue32_prepare(state,t0):
 _pt1_wait("glue32")
 _GLUE32["bank"]={}
 st=_SBB1["st"]
 if st is None or st["w"][4]!=0:
  return
 _GLUE32["bank"][0]=st
 for arm,mod in sorted(_GLUE32["mods"].items()):
  h=dict(st,mod=mod,ok={})
  for T in (16,32):
   if not st["ok"].get(T):
    continue
   try:
    h["ok"][T]=bool(_glue32_selftest(h,st,T))
   except Exception as ex:
    h["ok"][T]=False
    _GLUE32["err"]=repr(ex)
  _GLUE32["bank"][arm]=h
 print(f"[glue32] bank {[(a,h['ok']) for a,h in _GLUE32['bank'].items()]} error {_GLUE32['err']}",flush=True)
def _glue32_pick(tokens,router_dtype):
 if not GLUE32_ON or tokens not in (16,32) or router_dtype is not None:
  return None
 h=_GLUE32["bank"].get(GLUE32_ON)
 return h if h is not None and h["ok"].get(tokens) else None
'''


def build(base, version):
    require(version in (1, 2), "bad version")
    tree = ast.parse(base)
    assignments = {n.targets[0].id: n for n in tree.body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    # Prove source constants/critical arithmetic against the immutable row, not a hand-made stand-in.
    sbb = ast.literal_eval(assignments["SBB1_CU"].value)
    require("#define SB_CS 16" in sbb and "#define SB_NT (SB_H / (SB_CS * 8))" in sbb, "SBB geometry changed")
    for line in ("bsum = fmaf(v, v, bsum);", "rsum = fmaf(r, r, rsum);",
                 "csum = fmaf(r, sb, csum);", "ssum = fmaf(sb, sb, ssum);",
                 "const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, ts, 2.f * tc), tr);"):
        require(line in sbb and line in KERNEL, "numerical anchor " + line)
    # Compare the entire per-element arithmetic after specializing the ship's
    # <true,false,false> route and substituting only the virtual-pack indexing.
    def compact(text):
        return "".join(text.split())
    old_moment = sbb.split("float v = br[i];", 1)[1].split("ssum = fmaf(sb, sb, ssum);", 1)[0]
    old_moment = old_moment.replace("if (p.post_w && kInputScale) v = sb_rbf(v * pw[i]);", "")
    old_moment = old_moment.replace("p.post_w && !kInputScale", "p.post_w")
    new_moment = KERNEL.split("float v = br[pack][i];", 1)[1].split("ssum = fmaf(sb, sb, ssum);", 1)[0]
    require(compact(old_moment) == compact(new_moment.replace("[pack]", "")), "moment operation order")
    old_out = sbb.split("float n = val[i] * post_rstd;", 1)[1].split("nz[i] = z;", 1)[0]
    old_out = old_out.replace("p.post_w && !kInputScale", "p.post_w")
    new_out = KERNEL.split("float n = val[pack][i] * post_rstd;", 1)[1].split("nz[i] = z;", 1)[0]
    require(compact(old_out) == compact(new_out.replace("[pack]", "")), "output operation order")
    insertion = ("\nGLUE32_KERNEL=" + repr(KERNEL) + "\nGLUE32_HOST=" + repr(HOST.replace("@NV@", str(version)))
                 + "\n" + SUPPORT)
    source = sub(base, 'def _sbb1(stream,branch,router_dtype,state,tokens):\n st=_SBB1["st"]\n',
                 'def _sbb1(stream,branch,router_dtype,state,tokens):\n st=_glue32_pick(tokens,router_dtype)\n if st is None:\n  st=_SBB1["st"]\n',
                 "SBB arm selection")
    source = sub(source, '  jobs.append(("sbb1",_sbb1_build))\n',
                 '  jobs.append(("sbb1",_sbb1_build))\n  jobs.append(("glue32",_glue32_build))\n', "build job")
    source = sub(source, ' _sbb1_prepare(state,t0)\n',
                 ' _sbb1_prepare(state,t0)\n _glue32_prepare(state,t0)\n', "prepare hook")
    source += insertion
    lines = source.splitlines()
    require(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header")
    lines[1] = f"# nonce glue32-v{version} (virtual SBB cluster; default-off GLUE32_ON; exact tree)"
    source = "\n".join(lines) + "\n"
    require(source.isascii() and len(source.encode()) < 1_000_000, "ASCII / size")
    compile(source, str(OUTS[version - 1]), "exec")
    require(base.count("def _sbb1_prepare") == source.count("def _sbb1_prepare"), "self-tests retained")
    output_tree = ast.parse(source)
    changed = []
    for old, new in zip(tree.body, output_tree.body):
        if ast.dump(old) != ast.dump(new):
            require(isinstance(old, ast.FunctionDef), "unexpected changed base declaration")
            changed.append(old.name)
    require(set(changed) == {"_pt1_start", "prepare", "_sbb1"}, "only three intended base hooks changed")
    return source


def extract(source, names, env):
    tree = ast.parse(source)
    found = set()
    selected = []
    for node in tree.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in names:
            selected.append(node)
            found.add(name)
    require(found == set(names), f"missing definitions: {set(names) - found}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<real-row-slice>", "exec"), env)
    return env


class Tensor:
    """Strict shape/dtype double; it does NOT claim numerical GPU verification."""
    def __init__(self, shape=(), dtype="fp32", device="cuda", value=1.):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype, self.device, self.value = dtype, device, value

    def to(self, dtype):
        return Tensor(self.shape, dtype, self.device, self.value)

    def float(self):
        return self.to("fp32")

    def contiguous(self):
        return self

    def norm(self):
        return self.value

    def all(self):
        return True

    def __sub__(self, other):
        require(self.shape == other.shape, "shape mismatch")
        return Tensor(self.shape, self.dtype, value=0.)

    def __mul__(self, other):
        return Tensor(self.shape, self.dtype)

    def copy_(self, other):
        require(self.shape == other.shape, "copy shape")
        return self

    def fill_(self, value):
        return self


class Mod:
    def __init__(self):
        self.calls = []

    def sbb1_run(self, branch, stream, out, ns, router, pw, qw, al, bt, pe, qe, isc):
        require(branch.dtype == "bf16" and stream.dtype == "fp32", "call dtypes")
        require(branch.shape == stream.shape == out.shape == ns.shape, "call shapes")
        require(len(branch.shape) == 2 and branch.shape[1] == 8192, "call hidden")
        require(isc in (0, 8), "call input scale")
        self.calls.append(branch.shape[0])


class Stream:
    def wait_stream(self, other):
        pass


def fake_torch():
    fake = types.ModuleType("torch")
    fake.bfloat16, fake.float32 = "bf16", "fp32"
    fake.empty = lambda shape, dtype="fp32", device="cuda": Tensor(shape, dtype, device)
    fake.full = lambda shape, value, dtype="fp32", device="cuda": Tensor(shape, dtype, device)
    fake.randn = lambda shape, generator=None, device="cuda": Tensor(shape, "fp32", device)
    fake.isfinite = lambda value: Tensor(())
    fake.equal = lambda a, b: a.shape == b.shape and a.dtype == b.dtype
    fake.Generator = lambda device: types.SimpleNamespace(manual_seed=lambda seed: object())
    fake.cuda = types.SimpleNamespace(synchronize=lambda: None, Stream=Stream, current_stream=Stream,
        stream=lambda side: contextlib.nullcontext(), graph=lambda graph: contextlib.nullcontext(),
        CUDAGraph=lambda: types.SimpleNamespace(replay=lambda: None))
    fake.utils = types.ModuleType("torch.utils")
    fake.utils.cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    fake.utils.cpp_extension.load_inline = lambda **spec: Mod()
    return fake


def dry_run(base, source):
    fake = fake_torch()
    env = {"torch": fake, "hashlib": hashlib, "time": types.SimpleNamespace(time=lambda: 0.),
           "_pt1_wait": lambda name: None, "HIDDEN": 8192}
    names = ("_sbb1_prepare", "_sbb1_call", "SBB1_ROUTE", "SBB1_ENABLE", "SBB1_PREPARE_BY_S",
             "_SBB1", "K2A_M_MAX", "K2A_BUCKETS", "LAT_SBB_TRIG", "ROUTER_FP32_MAX_TOKENS",
             "_k2a_bucket_of", "_nb_sbb", "NB64_USE")
    extract(base, names, env)
    env["_sbb1_build"] = lambda: Mod()

    class Boundary:
        _cc_post_weight = Tensor(8192, "bf16")
        _cc_pre_weight = Tensor(8192, "bf16")
        _cc_alpha, _cc_beta = Tensor(8192), Tensor(8192)
        _post_norm = types.SimpleNamespace(has_weight=True, eps=1e-6)
        _pre_norm = types.SimpleNamespace(has_weight=True, eps=1e-6)
        _use_residual_gate, _post_norm_is_input_scale = True, False

        def _ensure_cached_weights(self, dtype):
            require(dtype == "bf16", "cache dtype")

        def __call__(self, stream, branch, branch_dtype, router_dtype):
            return branch, stream, stream if router_dtype is not None else None

    state = {"attn_boundary": Boundary(), "router": Tensor((256, 8192))}
    # Execute the original state construction AND its original Python self-test loops.
    with contextlib.redirect_stdout(io.StringIO()):
        ship = env["_sbb1_prepare"](state, 0.)
    require(ship is not None and ship["ok"][16] and ship["ok"][32], "real base prepare did not make scored routes")
    extract(source, ["SBB1_CU", "SBB1_HOST", "SBB1_DECL", "GLUE32_KERNEL", "GLUE32_HOST", "GLUE32_ON", "_GLUE32", "_sbb1"]
            + [n.name for n in ast.parse(SUPPORT).body if isinstance(n, ast.FunctionDef)], env)
    replacements = {"torch": fake, "torch.utils": fake.utils, "torch.utils.cpp_extension": fake.utils.cpp_extension}
    saved = {key: sys.modules.get(key) for key in replacements}
    try:
        sys.modules.update(replacements)
        env["_glue32_build"]()
    finally:
        for key, old in saved.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old
    require(set(env["_GLUE32"]["mods"]) == {1, 2}, "both arm builds exercised")
    with contextlib.redirect_stdout(io.StringIO()):
        env["_glue32_prepare"](state, 0.)
    bank = env["_GLUE32"]["bank"]
    require(set(bank) == {0, 1, 2}, "bank construction")
    require(bank[0] is ship and bank[1]["w"] is ship["w"] and bank[2]["e"] is ship["e"], "base state identity")
    for arm in (0, 1, 2, 3):
        env["GLUE32_ON"] = arm
        for t in (1, 8, 9, 16, 17, 32, 33, 128, 256):
            for rd in (None, "fp32"):
                selected = env["_glue32_pick"](t, rd)
                require((selected is not None) == (arm in (1, 2) and t in (16, 32) and rd is None), "arm selection")
                before = {a: len(h["mod"].calls) for a, h in bank.items()}
                actual = env["_sbb1"](Tensor((t, 8192)), Tensor((t, 8192), "bf16"), rd, state, t)
                chosen = arm if selected is not None else 0
                live = env["_k2a_bucket_of"](t) in ship["ok"]
                require((actual is not None) == live, "actual SBB route/fallback")
                for a, h in bank.items():
                    require(len(h["mod"].calls) == before[a] + int(live and a == chosen), "actual module dispatch")
    env["GLUE32_ON"] = 1
    bank[1]["ok"][16] = False
    require(env["_glue32_pick"](16, None) is None, "failed correctness gate must fall back")
    env["_SBB1"]["st"] = None
    env["_glue32_prepare"](state, 0.)
    require(env["_GLUE32"]["bank"] == {}, "missing base fallback")
    env["_SBB1"]["st"] = dict(ship, w=ship["w"][:4] + (1,) + ship["w"][5:])
    env["_glue32_prepare"](state, 0.)
    require(env["_GLUE32"]["bank"] == {}, "input-scale fallback")
    env["_SBB1"]["st"] = ship
    env["_GLUE32"]["mods"] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        env["_glue32_prepare"](state, 0.)
    require(set(env["_GLUE32"]["bank"]) == {0}, "build-failure fallback")
    env["_GLUE32"]["mods"] = {1: Mod(), 2: Mod()}
    saved_test = env["_glue32_selftest"]
    def fail_test(*args, **kwargs):
        raise RuntimeError("injected correctness failure")
    env["_glue32_selftest"] = fail_test
    with contextlib.redirect_stdout(io.StringIO()):
        env["_glue32_prepare"](state, 0.)
    require(not any(env["_GLUE32"]["bank"][1]["ok"].values()), "self-test exception fallback")
    env["_glue32_selftest"] = saved_test
    env["_SBB1"]["st"] = dict(ship, ok={16: False, 32: False})
    with contextlib.redirect_stdout(io.StringIO()):
        env["_glue32_prepare"](state, 0.)
    require(env["_GLUE32"]["bank"][1]["ok"] == {}, "base correctness gate respected")
    env["_GLUE32"]["mods"] = {}
    old_loader = fake.utils.cpp_extension.load_inline
    fake.utils.cpp_extension.load_inline = fail_test
    saved = {key: sys.modules.get(key) for key in replacements}
    try:
        sys.modules.update(replacements)
        env["_glue32_build"]()
    finally:
        fake.utils.cpp_extension.load_inline = old_loader
        for key, old in saved.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old
    require(env["_GLUE32"]["mods"] == {} and env["_GLUE32"]["err"], "compile exception recorded")
    return {"status": "PASS", "real_base_prepare": True, "actual_dispatch": True, "arms": [0, 1, 2],
            "compile_and_correctness_failure_paths": True, "tensor_double_only": True}


def reduction_gate():
    import numpy as np
    rng = np.random.default_rng(3111)
    cases = [np.zeros((1024, 4), dtype=np.float32),
             np.tile(np.array([-0., 0., -0., 0.], dtype=np.float32), (1024, 1))]
    for _ in range(32):
        cases.append((rng.normal(size=(1024, 4)) * 2. ** rng.integers(-30, 31, size=(1024, 4))).astype(np.float32))

    def warp(a):
        a = a.copy()
        for offset in (16, 8, 4, 2, 1):
            a = (a + a[np.arange(32) ^ offset]).astype(np.float32)
        return a[0]

    for values in cases:
        ranks = []
        for rank in range(16):
            part = np.zeros(4, dtype=np.float32)
            for w in range(2):
                part = part + warp(values[rank * 64 + w * 32:rank * 64 + (w + 1) * 32])
            ranks.append(part)
        expected = np.zeros(4, dtype=np.float32)
        for part in ranks:
            expected = expected + part
        for nv in (1, 2):
            nt = 1024 // nv
            coverage = [tid + pack * nt for pack in range(nv) for tid in range(nt)]
            require(sorted(coverage) == list(range(1024)), "virtual lane coverage")
            sm = np.full((32, 4), np.nan, dtype=np.float32)
            for pack in range(nv):
                for w in range(nt // 32):
                    lo = pack * nt + w * 32
                    sm[lo // 32] = warp(values[lo:lo + 32])
            parts = np.full((16, 4), np.nan, dtype=np.float32)
            for tid in range(16):
                mine = np.zeros(4, dtype=np.float32)
                for w in range(2):
                    mine = mine + sm[2 * tid + w]
                parts[tid] = mine
            got = np.zeros(4, dtype=np.float32)
            for q in range(16):
                got = got + parts[q]
            require(np.array_equal(expected.view(np.uint32), got.view(np.uint32)), "bitwise reduction order")
    return {"status": "PASS", "cases": len(cases) * 2, "arbitrary_FP32_moments": True}


def cpu_gate(write=False):
    base_bytes = BASE.read_bytes()
    base = base_bytes.decode("ascii").replace("\r\n", "\n")
    result = {"base_sha256": hashlib.sha256(base_bytes).hexdigest(), "reduction": reduction_gate(), "outputs": []}
    for version, path in enumerate(OUTS, 1):
        source = build(base, version)
        dry = dry_run(base, source)
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.encode("ascii"))
        elif path.exists():
            require(path.read_bytes() == source.encode("ascii"), f"regeneration mismatch {path.name}")
        result["outputs"].append({"path": str(path.relative_to(ROOT)), "bytes": len(source.encode()), "dry_run": dry,
                                  "sha256": hashlib.sha256(source.encode()).hexdigest(), "selector_default": 0})
    require(BASE.read_bytes() == base_bytes, "immutable row changed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="CPU gates only; no writes")
    args = parser.parse_args()
    print(json.dumps(cpu_gate(write=not args.check), indent=2))


if __name__ == "__main__":
    main()

"""TPW: default-off, in-file T=1 PREWAIT / NOLOAD-X lab on k3_moexr_c2.

Only this generator's named output is written. --check and --dry-run never write.
CUDA compilation is separate: bash tools/local/tpw_compile.py under cudaenv.sh.
The original _HYBLO_CUDA and TINY2 remain untouched. This is NOT a submission.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c2.py"
OUT = ROOT / "kernels/lab/k3_tpw_ab.py"


def once(s, old, new):
    assert s.count(old) == 1, ("anchor must occur once", old[:100], s.count(old))
    return s.replace(old, new, 1)


def literal(tree, name):
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


TOP8 = r'''
// TPW: the tail's arithmetic order, float key transform, sorting network and
// lower-expert tie-break; no tail output is read here. Only token zero exists.
__device__ __forceinline__ bool ft_better(unsigned k1, int j1, unsigned k2, int j2) { return k1 > k2 || (k1 == k2 && j1 < j2); }
template<int NSMAX>
__device__ __forceinline__ void tpw_top8(const Params& p, Smem& S) {
const int lane = threadIdx.x & 31;
unsigned hk[8];
int hj[8] = {0, 1, 2, 3, 4, 5, 6, 7};
float aa[8];
#pragma unroll
for (int q = 0; q < NSMAX; ++q) {
float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
if (q < p.tpw_ns) {
const float* src = p.tpw_la + q * p.tpw_stride + lane * 8;
v0 = __ldcg(reinterpret_cast<const float4*>(src));
v1 = __ldcg(reinterpret_cast<const float4*>(src + 4));
}
const float v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
for (int i = 0; i < 8; ++i) aa[i] = q == 0 ? v[i] : aa[i] + v[i];
}
float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;
if (p.tpw_lb) {
qb0 = __ldcg(reinterpret_cast<const float4*>(p.tpw_lb + lane * 8));
qb1 = __ldcg(reinterpret_cast<const float4*>(p.tpw_lb + lane * 8 + 4));
}
const float qb[8] = {qb0.x, qb0.y, qb0.z, qb0.w, qb1.x, qb1.y, qb1.z, qb1.w};
const float4 bs0 = *reinterpret_cast<const float4*>(p.tpw_bias + lane * 8);
const float4 bs1 = *reinterpret_cast<const float4*>(p.tpw_bias + lane * 8 + 4);
const float bs[8] = {bs0.x, bs0.y, bs0.z, bs0.w, bs1.x, bs1.y, bs1.z, bs1.w};
#pragma unroll
for (int i = 0; i < 8; ++i) {
float a = aa[i];
a += qb[i];
const float s = 1.f / (1.f + expf(-a * p.tpw_mult));
const unsigned u = __float_as_uint((s + bs[i]) + 0.f);
hk[i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
constexpr int NWK[19][2] = {{0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {0, 1}, {2, 3},
{4, 5}, {6, 7}, {2, 4}, {3, 5}, {1, 4}, {3, 6}, {1, 2}, {3, 4}, {5, 6}};
#pragma unroll
for (int x = 0; x < 19; ++x) {
const int i = NWK[x][0], k = NWK[x][1];
const bool sw = ft_better(hk[k], hj[k], hk[i], hj[i]);
const unsigned tk = hk[i]; const int tj = hj[i];
hk[i] = sw ? hk[k] : tk; hj[i] = sw ? hj[k] : tj;
hk[k] = sw ? tk : hk[k]; hj[k] = sw ? tj : hj[k];
}
const int ls = __ldcg(p.tpw_lstart);
#pragma unroll
for (int k = 0; k < 8; ++k) {
const unsigned mx = __reduce_max_sync(0xffffffffu, hk[0]);
const int owner = __ffs(__ballot_sync(0xffffffffu, hk[0] == mx)) - 1;
const int mid = __shfl_sync(0xffffffffu, lane * 8 + hj[0], owner);
const int rel = ((mid - ls) % 256 + 256) % 256;
if (lane == k) {
S.tpw_ids[k] = rel < 32 ? rel : -1;
if (p.tpw_audit) p.tpw_audit[blockIdx.x * 8 + k] = S.tpw_ids[k];
}
const bool me = lane == owner;
#pragma unroll
for (int j = 0; j < 7; ++j) {
hk[j] = me ? hk[j + 1] : hk[j]; hj[j] = me ? hj[j + 1] : hj[j];
}
hk[7] = me ? 0u : hk[7];
}
}
'''

EXTRA_ARGS = ("torch::Tensor lg, torch::Tensor bias, torch::Tensor lstart, torch::Tensor audit, "
              "int64_t la, int64_t lb, int64_t ns, int64_t sstride, double mult")


def cuda_edits(cu):
    """Record unique, reversible anchors; the runtime source factory reasserts each."""
    edits = []

    def edit(a, b):
        nonlocal cu
        cu = once(cu, a, b)
        edits.append((a, b))

    edit("#include <stdint.h>\n", "#include <stdint.h>\n#include <math.h>\n#ifndef OKC_TPW\n#define OKC_TPW 0\n#endif\n")
    edit("unsigned long long* dbg;\n};", "unsigned long long* dbg;\n#if OKC_TPW == 1\n"
         "const float *tpw_la, *tpw_lb, *tpw_bias;\nconst int* tpw_lstart;\nint* tpw_audit;\n"
         "int tpw_ns; long tpw_stride; float tpw_mult;\n#endif\n};")
    edit("signed char eid[EMAX];", "#if OKC_TPW == 1\nint tpw_ids[8];\n#endif\nsigned char eid[EMAX];")
    anchor = "__device__ void route(const Params& p, Smem& S) {"
    edit(anchor, "#if OKC_TPW == 1\n" + TOP8 + "#endif\n"
         "__device__ __forceinline__ int tpw_id(const Params& p, const Smem& S, int i) {\n"
         "#if OKC_TPW == 1\nreturn S.tpw_ids[i];\n#else\nreturn __ldg(p.topk_ids + i);\n#endif\n}\n" + anchor)
    # Individual full lines: no global, count-tolerant replacement of a pointer read.
    for a in (
        "int e = __ldg(p.topk_ids + i) - p.lo;",
        "if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }",
        "const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;",
        "const int e = __ldg(p.topk_ids + i) - p.lo;",
    ):
        b = a.replace("__ldg(p.topk_ids + i)", "tpw_id(p, S, i)")
        b = b.replace("__ldg(p.topk_ids + ln)", "tpw_id(p, S, ln)")
        b = b.replace("__ldg(p.topk_ids + ln + 32)", "tpw_id(p, S, ln + 32)")
        # Include line boundaries: 'int e' is also a substring of 'const int e'.
        edit("\n" + a + "\n", "\n" + b + "\n")
    edit('DBG(0);\nasm volatile("griddepcontrol.wait;\\n" ::: "memory");',
         'DBG(0);\n#if OKC_TPW == 1\nif (tid < 32) {\n'
         'if (p.tpw_ns > 1) tpw_top8<8>(p, S); else tpw_top8<1>(p, S);\n}\n'
         '__syncthreads(); // all 288 threads, before the helper branches\n#else\n'
         'asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#endif')
    edit("const int n = min(OKC_FAST1, v1 - v0);",
         "const int n = min(OKC_TPW == 1 ? OKC_NS : OKC_FAST1, v1 - v0);")
    pf = ('for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)\n'
          'asm volatile("prefetch.global.L2 [%0];\\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));\n')
    gen = ('const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);\n'
           'S.gen = gen;\nS.ssp = (gen & 1) ? p.ss1 : p.ss;\n')
    zero = ('{\nfloat* other = (S.gen & 1) ? p.ss : p.ss1;\n'
            'for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;\n'
            'if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n}\n')
    for s in (pf, gen, zero):
        edit(s, "#if OKC_TPW != 1\n" + s + "#endif\n")
    edit("WS_JOIN();\nrun_phase<false, BIG>(p, S, xbuf);",
         'WS_JOIN();\n#if OKC_TPW == 1\nWS_JOIN(); // full initial ring has now been issued\n'
         'asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n'
         + pf + "if (tid == 0) {\n" + gen + "}\nCTA_SYNC();\n" + zero
         + "#endif\nrun_phase<false, BIG>(p, S, xbuf);")
    anchor = "for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);"
    edit(anchor, anchor + '\n#if OKC_TPW == 1\nWS_JOIN();\n'
         'asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#endif')
    edit("load_x_regs<PB>(p, S, j, lane, warp, x0);",
         "#if OKC_TPW == 2\n// TIMING ONLY: keep doneA polls / ring / MMA; delete both x and inter loads.\n"
         "#pragma unroll\nfor (int c = 0; c < 2 * NCH; ++c) x0[c] = make_uint4(0, 0, 0, 0);\n"
         "#else\nload_x_regs<PB>(p, S, j, lane, warp, x0);\n#endif")
    # The original fat-job trap is unreachable for M=1; new builds enforce M=1 on the host.
    edit("if (nF > 0) __trap();", "// TPW host enforces M=1, hence nF=0; no device trap.")
    edit("torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {",
         "torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid, " + EXTRA_ARGS + ") {")
    edit("okc::Params p;\n", 'TORCH_CHECK(x.size(0) == 1 && lo == 0 && nloc == 32 && grid >= 4, "TPW is T=1 only");\n'
         "okc::Params p;\n#if OKC_TPW == 1\n"
         'TORCH_CHECK(ns >= 1 && ns <= 8, "TPW ns");\n'
         'TORCH_CHECK(lg.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat, "TPW logits precision");\n'
         'TORCH_CHECK(la >= 0 && lg.numel() >= la + (ns - 1) * sstride + 256, "TPW logits bounds");\n'
         'TORCH_CHECK(lb < 0 || lg.numel() >= lb + 256, "TPW second buffer bounds");\n'
         'TORCH_CHECK(audit.numel() == 0 || audit.numel() >= grid * 8, "TPW audit bounds");\n'
         'TORCH_CHECK(la % 4 == 0 && sstride % 4 == 0 && (lb < 0 || lb % 4 == 0), "TPW float4 alignment");\n'
         "p.tpw_la = lg.data_ptr<float>() + la;\np.tpw_lb = lb < 0 ? nullptr : lg.data_ptr<float>() + lb;\n"
         "p.tpw_bias = bias.data_ptr<float>(); p.tpw_lstart = lstart.data_ptr<int>();\n"
         "p.tpw_audit = audit.numel() ? audit.data_ptr<int>() : nullptr;\n"
         "p.tpw_ns = (int)ns; p.tpw_stride = (long)sstride; p.tpw_mult = (float)mult;\n#endif\n")
    # Configure both kernels in prepare via okc_set_pdl, including NX (never
    # invoked by correctness). No first-use attribute initialization in capture.
    config = '''static bool attr = false;
size_t sm = okc_smem_bytes();
if (!attr) {
cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
(void)e0; (void)e1;
attr = true;
}
'''
    edit(config, "tpw_configure();\nsize_t sm = okc_smem_bytes();\n")
    edit("void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }",
         "static void tpw_configure() {\n" + config.replace("(void)e0; (void)e1;",
          'TORCH_CHECK(e0 == cudaSuccess && e1 == cudaSuccess, "TPW shared-memory attributes");')
         + "}\nvoid okc_set_pdl(int64_t on) { tpw_configure(); g_okc_pdl = on ? 1 : 0; }")
    return edits


GLUE = '''
# TPW lab: 0 ship; 1 PREWAIT; 2 NOLOAD-X (TIMING ONLY, wrong output).
TPW=0
TNSM=0 # T=1 override only; the T=8 control retains its original grid.
TPW_TIMING_ONLY=False # explicit capture-time opt-in required for arm 2
_TPW={k:{"mod":None,"ok":False,"sc":None,"full":None,"pdl":-1,"err":None,"cases":0} for k in (1,2)}

def _tpw_src(k):
 assert k in (1,2)
 s=_HYBLO_CUDA
 for a,b in _TPW_EDITS:
  assert s.count(a)==1,a[:100]
  s=s.replace(a,b,1)
 fl=list(_HYBLO_FLAGS)+list(_HYBLO_KEYS[0][2])+[f"-DOKC_TPW={k}"]
 return s,fl

def _tpw_load(**kw):
 from torch.utils.cpp_extension import load_inline
 return load_inline(**kw)

def _tpw_build():
 for k,H in _TPW.items():
  if H["mod"] is not None:
   continue
  try:
   cu,fl=_tpw_src(k)
   tag=hashlib.sha1((_TPW_CPP+cu+" ".join(fl)).encode()).hexdigest()[:12]
   H["mod"]=_tpw_load(name=f"tpw_{k}_{tag}",cpp_sources=_TPW_CPP,cuda_sources=cu,
    functions=["okc_run","okc_set_pdl"],extra_cuda_cflags=fl,verbose=False)
  except Exception as ex:
   H["err"]=repr(ex)

def _tpw_tail_args(y,logits,state,data):
 # Exact layout expressions from _v8_tail; tensors, never prepare-time pointers.
 v8=state["v8"]
 ns,sstride=1,0
 if logits is None:
  lg,la,lb,l_ld=y,D_EXPERT,D_EXPERT+N_EXPERTS,y.stride(0)
 elif logits.dim()==3:
  lg,la,lb,l_ld=logits,0,-1,logits.stride(1)
  ns,sstride=logits.shape[0],logits.stride(0)
 else:
  lg,la,lb,l_ld=logits,0,-1,logits.stride(0)
 return (lg,v8["bias"],data["local_start"],la,lb,ns,sstride,0.1767766952966369)

def _tpw_run(H,x,ids,wts,ctx,grid,audit=None):
 ids,wts=ids.to(torch.int32).contiguous(),wts.to(torch.float32).contiguous()
 out=torch.empty((x.shape[0],4096),dtype=torch.bfloat16,device=x.device)
 pdl=1 if HYBLO_PDL else 0
 if H["pdl"]!=pdl:
  H["mod"].okc_set_pdl(pdl)
  H["pdl"]=pdl
 if ctx is None:
  ctx=(H["empty_f"],H["empty_f"],H["empty_i"],0,-1,1,0,0.1767766952966369)
 lg,bias,ls,la,lb,ns,ss,mult=ctx
 H["mod"].okc_run(_HYBLO["w"],H["sc"],x,ids,wts,out,0,32,grid,
  lg,bias,ls,H["empty_i"] if audit is None else audit,la,lb,ns,ss,mult)
 return out

def _tpw_select(m,ctx):
 if m!=1 or TPW==0:
  return None
 if TPW not in (1,2):
  raise ValueError("TPW must be 0, 1 or 2")
 if TPW==2 and not TPW_TIMING_ONLY:
  raise RuntimeError("NOLOAD-X is TIMING ONLY; excluded from correctness and submission")
 H=_TPW[TPW]
 if not H["ok"]:
  raise RuntimeError(f"TPW {TPW} OFF: {H['err']}")
 if TPW==1 and ctx is None:
  raise RuntimeError("PREWAIT requires the event-joined T=1 side-router route")
 return H

def _tpw_call(x,ids,wts,ctx):
 H=_tpw_select(x.shape[0],ctx)
 if H is None:
  return None
 return _tpw_run(H,x,ids,wts,ctx,TNSM if TNSM else _HYB["nsm"])

def _tpw_cases(v8,dev):
 # 16 inputs: ns=1/2/3/8, optional second buffer, ties, rounding, extremes,
 # wrapped local ranges, all-local and no-local routing. Fresh inputs every prepare.
 g=torch.Generator(device=dev).manual_seed(712031)
 for i in range(16):
  ns=(1,2,3,8)[i%4]
  part=torch.randn((ns,1,256),generator=g,device=dev,dtype=torch.float32)*3.0
  y=torch.randn((1,4096),generator=g,device=dev).to(torch.bfloat16)
  bias=v8["bias"]
  if i in (4,5):
   part.zero_(); bias=torch.zeros_like(bias)
  elif i in (6,7):
   part=part.round(); bias=(bias*4).round()/4
  elif i in (8,9):
   part=part*100
  elif i in (10,11):
   part.zero_(); bias=torch.zeros_like(bias)
   bias[:8]=10.0
  ls=(0,224,250,-3)[i%4]
  if i==11:
   ls=64
  data={"local_start":torch.tensor([ls],dtype=torch.int32,device=dev)}
  vv={"v8":dict(v8,bias=bias)}
  if i>=12:
   # Actual _v8_tail logits=None path: first + optional second buffer in y32.
   y=torch.cat((y.float(),part[0],torch.randn((1,256),generator=g,device=dev)),dim=1)
   logits=None
  elif ns==1:
   logits=part[0]
  else:
   logits=part
  yield i,y,logits,vv,data

def _tpw_prepare(state):
 for H in _TPW.values():
  H["ok"]=False
  H["cases"]=0
 if not (_HYBLO["ok"].get("tiny") and state.get("v8") is not None):
  for H in _TPW.values():
   H["err"]="base TINY / tail is OFF"
  return
 dev=_HYBLO["w"][0].device
 _tpw_build()
 for k,H in _TPW.items():
  if H["mod"] is None:
   continue
  try:
   H["sc"],H["full"]=_hyb_scratch(dev)
   H["empty_f"]=torch.empty((0,),dtype=torch.float32,device=dev)
   H["empty_i"]=torch.empty((0,),dtype=torch.int32,device=dev)
   H["mod"].okc_set_pdl(1 if HYBLO_PDL else 0)
   H["pdl"]=1 if HYBLO_PDL else 0
   if k==2:
    # No correctness claim or comparison for the deliberate output-null.
    H["ok"]=True
    H["err"]=None
    continue
   for i,y,lg,vv,data in _tpw_cases(state["v8"],dev):
    ctx=_tpw_tail_args(y,lg,vv,data)
    x,tw,tid,lid=_v8_tail(y,lg,vv,data,ftf=8)
    ref=_hyblo_run("tiny",x,lid,tw).clone()
    for grid in (_HYB["nsm"],128):
     audit=torch.full((grid,8),-777,dtype=torch.int32,device=dev)
     got=_tpw_run(H,x,lid,tw,ctx,grid,audit)
     torch.cuda.synchronize()
     assert torch.equal(audit,lid.expand(grid,8)),f"PREWAIT expert IDs differ: input {i}, grid {grid}"
     assert bool(torch.isfinite(got.float()).all()),f"PREWAIT nonfinite input {i}"
     ev,totrow=_k2a_rel_err(got,ref)
     assert ev<1e-2 and totrow<5e-2,("PREWAIT output",i,grid,ev,totrow)
     H["cases"]+=1
   assert H["cases"]>=16
   assert all(bool((f[n:].view(vt)==sv).all())for f,n,vt,sv in H["full"])
   H["ok"]=True
   H["err"]=None
  except Exception as ex:
   H["ok"]=False
   H["err"]=repr(ex)
 print("[tpw]",{k:{n:H[n] for n in ("ok","cases","err")} for k,H in _TPW.items()},flush=True)
'''


def generate(base):
    tree = ast.parse(base)
    tail = literal(tree, "V8_TAIL_CUDA_SRC")
    start = "__global__ void __launch_bounds__(NT + 32 * RW) tail_kernel3("
    assert tail.count(start) == 1
    tail3 = tail.split(start, 1)[1].split("template <int MT>", 1)[0]
    # Refuse regeneration if the reference's arithmetic/order contract drifts.
    for anchor in (
        "for (int q = 1; q < NSMAX; ++q) a += qa[q][i];",
        "a += qb[i];",
        "const float s = 1.f / (1.f + expf(-a * mult));",
        "const unsigned u = __float_as_uint((s + bs[i]) + 0.f);",
        "skey[e0 + i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);",
        "const bool sw = ft_better(hk[k], hj[k], hk[i], hj[i]);",
        "const int rel = ((mid - ls) % NE + NE) % NE;",
        "lid[t * TK + lane] = rel < 32 ? rel : -1;",
    ):
        assert tail3.count(anchor) == 1, ("tail contract changed", anchor)
    predicate = "bool ft_better(unsigned k1, int j1, unsigned k2, int j2) { return k1 > k2 || (k1 == k2 && j1 < j2); }"
    assert tail.count(predicate) == 1 and TOP8.count(predicate) == 1
    network = TOP8.split("constexpr int NWK[19][2] = ", 1)[1].split(";", 1)[0]
    assert tail3.count("constexpr int NWK[19][2] = " + network + ";") == 1
    cu = literal(tree, "_HYBLO_CUDA")
    edits = cuda_edits(cu)
    cpp = once(literal(tree, "_HYB_MOE_CPP"), "int64_t grid);", "int64_t grid, " + EXTRA_ARGS + ");")
    block = "\n_TPW_EDITS=" + repr(edits) + "\n_TPW_CPP=" + repr(cpp) + "\n" + GLUE
    out = once(base, "def _hyblo_run(k,x,ids,wts):", block + "\ndef _hyblo_run(k,x,ids,wts):")
    out = once(out, 'mod.okc_run(_HYBLO["w"],_HYBLO["sc"],x,ids,wts,out,0,32,_HYB["nsm"])',
               'mod.okc_run(_HYBLO["w"],_HYBLO["sc"],x,ids,wts,out,0,32,(TNSM if TNSM and x.shape[0]==1 else _HYB["nsm"]))')
    # Same late-prepare position as padz; no new thread / prefetch / external module surgery.
    anchor = ' _rq_prepare(state)\n'
    out = once(out, anchor, anchor + ' _tpw_prepare(state)\n')
    out = once(out, ' tail=V8_TAIL_ROUTE.get(_k2a_bucket_of(tokens))if state.get("v8")is not None else None',
               ' tpw_ctx=None\n tail=V8_TAIL_ROUTE.get(_k2a_bucket_of(tokens))if state.get("v8")is not None else None')
    anchor = '   expert_x,topk_weights,topk_ids,local_ids=_v8_tail(expert_input,logits,state,data)\n'
    out = once(out, anchor, anchor + '   if tokens==1 and TPW==1 and vgr is None and place!="main" and mode=="f32":\n'
               '    tpw_ctx=_tpw_tail_args(expert_input,logits,state,data)\n')
    out = once(out, '  expert_output=_v8_experts(expert_x,local_ids,topk_weights,state,cfg,tokens)',
               '  expert_output=_tpw_call(expert_x,local_ids,topk_weights,tpw_ctx)\n'
               '  if expert_output is None:\n'
               '   expert_output=_v8_experts(expert_x,local_ids,topk_weights,state,cfg,tokens)')
    compile(out, str(OUT), "exec")
    ot = ast.parse(out)
    assert literal(ot, "_HYBLO_CUDA") == cu
    assert literal(ot, "_HYBLO_FLAGS") == literal(tree, "_HYBLO_FLAGS")
    assert literal(ot, "_HYBLO_KEYS") == literal(tree, "_HYBLO_KEYS")
    # Every original function outside the three explicitly modified hooks is AST-identical.
    allowed = {"prepare", "kernel_fn", "_hyblo_run"}
    newf = {n.name:n for n in ot.body if isinstance(n, ast.FunctionDef)}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name not in allowed:
            assert ast.dump(n) == ast.dump(newf[n.name]), n.name
    return out


VARIANTS = [
    {"name":"ship", "set":{"TPW":0,"TNSM":0,"TPW_TIMING_ONLY":False}},
    {"name":"pw", "set":{"TPW":1,"TNSM":0,"TPW_TIMING_ONLY":False}},
    {"name":"mid", "set":{"TPW":0,"TNSM":0,"TPW_TIMING_ONLY":False}},
    {"name":"nx", "set":{"TPW":2,"TNSM":0,"TPW_TIMING_ONLY":True}},
    {"name":"pw128", "set":{"TPW":1,"TNSM":128,"TPW_TIMING_ONLY":False}},
    {"name":"ship2", "set":{"TPW":0,"TNSM":0,"TPW_TIMING_ONLY":False}},
]


def main():
    base = BASE.read_text(encoding="utf-8")
    text = generate(base)
    if "--variants" in sys.argv or "--variants-correct" in sys.argv:
        arms = [v for v in VARIANTS if v["name"] != "nx"] if "--variants-correct" in sys.argv else VARIANTS
        print(json.dumps(arms, indent=2))
        return
    if "--check" in sys.argv or "--dry-run" in sys.argv:
        assert OUT.read_text(encoding="utf-8") == text, "generated output stale"
    else:
        OUT.write_bytes(text.encode("utf-8"))
    result = {"generator":"PASS", "bytes":len(text.encode()), "base_sha256":hashlib.sha256(base.encode()).hexdigest(),
              "ship_cuda_unchanged":True, "tiny2_unchanged":True}
    if "--dry-run" in sys.argv:
        # The polyglot gate is imported as Python, without bytecode writes.
        ns = {"__name__":"tpw_gate", "__file__":str(ROOT / "tools/local/tpw_compile.py")}
        exec(compile((ROOT / "tools/local/tpw_compile.py").read_text(), ns["__file__"], "exec"), ns)
        result["dry_run"] = ns["cpu_gate"](text, base)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

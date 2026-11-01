"""VMLFIN: exact-anchor, T=32-only last-arriver finalization experiment.

Only writes kernels/lab/k3_vmlfin_ab.py. --check is read-only. No CUDA runs.
The runtime bank follows tools/padz_patches.py's VML_LAB lifecycle/accessors.
CPU gates and the real prepare-path dry run live in local/vmlfin_compile.py.
"""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/lab/k3_vmlfin_ab.py"


def sub(s, old, new, count=1):
    assert s.count(old) == count, ("exact anchor", old[:100], s.count(old), count)
    return s.replace(old, new)


# Kept as top-level Python source in the generated submission, not module surgery.
SOURCE = r'''
def _vmlfin_sub(s, a, b, n=1):
 assert s.count(a)==n, (a[:80],s.count(a),n)
 return s.replace(a,b)

def _vmlfin_src():
 s=_vml_CUDA
 # Capture the actual base finalizer, retaining operation and route order.
 fa="__global__ void __launch_bounds__(256) vm_fin(const Args A) {\n"
 fb="\n}\n}\nvoid vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V) {"
 assert s.count(fa)==1 and s.count(fb)==1
 f=s[s.index(fa):s.index(fb)+2]
 f=_vmlfin_sub(f,fa,"__device__ __noinline__ void vmlfin_token(const Args& A, const Tab& T, int t, FinScratch& F) {\n")
 f=_vmlfin_sub(f,"__shared__ int s_r[8], s_r8[8];\n__shared__ float s_w[8], s_w8[8];\n__shared__ int s_n;\n__shared__ float s_ss[2][8][4];\n",
 "int (&s_r)[8]=F.r, (&s_r8)[8]=F.r8;\nfloat (&s_w)[8]=F.w, (&s_w8)[8]=F.w8;\nint& s_n=F.n;\nfloat (&s_ss)[2][8][4]=F.ss;\n")
 f=_vmlfin_sub(f,"const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;",
 "const int tid = threadIdx.x & 255, warp = tid >> 5, lane = tid & 31;")
 f=_vmlfin_sub(f,'#if VM_FT\nasm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n#endif\n',"")
 f=_vmlfin_sub(f,'asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n',"")
 ca="if (t == 0) {\nfor (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;"
 cb="#if VM_FTR\nif (warp == 0) {"
 assert f.count(ca)==1 and f.count(cb)==1
 f=f[:f.index(ca)]+f[f.index(cb):]
 # Local map avoids racing CTA 0's global rmap publication (including empty tokens).
 f=_vmlfin_sub(f,"__ldcg(A.rmap + t * 8 + lane)","T.frmap[t * 8 + lane]")
 f=_vmlfin_sub(f,"__ldcg(A.rmap + t * 8 + tid)","T.frmap[t * 8 + tid]")
 f=f.replace("__syncthreads();","bar_sync(11 + (threadIdx.x >> 8), 256);")
 # One route at a time: same per-route lane reduction and same route accumulation,
 # but only 16 live y values instead of 64. Necessary at vm_main's register cap.
 f=_vmlfin_sub(f,"c0 += 4, par ^= 1","++c0, par ^= 1")
 f=f.replace("jj < 4; ++jj","jj < 1; ++jj")
 f=_vmlfin_sub(f,"const int nv = min(nloc - c0, 4);","const int nv = 1;")
 f=f[:-1]+"bar_sync(11 + (threadIdx.x >> 8), 256);\n}"  # scratch reuse within each group
 s=_vmlfin_sub(s,"struct Tab {\n", "#ifndef VM_VMLFIN\n#define VM_VMLFIN 0\n#endif\n"
 "#if VM_VMLFIN\nstatic_assert(VM_BRED && VM_RT && VM_FTR && VM_JN == 64 && !VM_WG5 && !VM_DP && !VM_SC && !VM_AD && !VM_LD && VM_MKW == 32, \"VMLFIN bf2 only\");\n#endif\nstruct Tab {\n")
 s=_vmlfin_sub(s,"int nj, R, Rpad, pad0;\n", "int nj, R, Rpad, pad0;\n#if VM_VMLFIN\nint frmap[256], fready[32], flast;\n#endif\n")
 s=_vmlfin_sub(s,"constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);\n",
 "constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);\n#if VM_VMLFIN\nstatic_assert(SMEM <= 227328, \"VMLFIN shared memory admission\");\n#endif\n")
 a="if (blockIdx.x == 0 && e < e1) A.rmap[e] = r;\n"
 s=_vmlfin_sub(s,a,a+"#if VM_VMLFIN\nif (e < e1) T.frmap[e] = r;\n#endif\n",2)
 ma="__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {\n"
 s=_vmlfin_sub(s,ma,"#if VM_VMLFIN\n__device__ __forceinline__ void vmlfin_body(const Args A) {\n#else\n"+ma+"#endif\n")
 tail=_VMLFIN_TAIL.replace("@TOKEN_HELPER@",f)
 s=_vmlfin_sub(s,fb,"\n}\n"+tail+"\n}\nvoid vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V) {")
 s=_vmlfin_sub(s,"const int64_t M = V[0], G = V[2], mask = V[4];\n",
 "const int64_t M = V[0], G = V[2], mask = V[4];\n#if VM_VMLFIN\n"
 "TORCH_CHECK(M == 32 && mask == 3 && T[19].numel() == 33, \"VMLFIN needs T32, mask3, own 33-word gbar\");\n#endif\n")
 s=_vmlfin_sub(s,"if ((mask & 2) && M > 0) {\n","#if !VM_VMLFIN\nif ((mask & 2) && M > 0) {\n")
 s=_vmlfin_sub(s,"C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));\n}\n}",
 "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));\n}\n#endif\n}")
 return s
'''

TAIL = r'''
#if VM_VMLFIN
struct FinScratch { int r[8], r8[8]; float w[8], w8[8]; int n; float ss[2][8][4]; };
@TOKEN_HELPER@
__global__ void __launch_bounds__(NTH, 1) vm_main(const __grid_constant__ Args A) {
extern __shared__ __align__(1024) uint8_t smem[];
// All early returns in the original body return here, including both producers.
vmlfin_body(A);
const int tid = threadIdx.x;
Tab& T = *reinterpret_cast<Tab*>(smem + TAB_OFF);
// Drain every consumer's y stores / BRED atomics before the elected publishers.
__threadfence();
__syncthreads();
if (tid == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
if (tid < 32) {
const int t = tid, b0 = T.bB[blockIdx.x], b1 = T.bB[blockIdx.x + 1];
int credit = 0, need = 0;
#pragma unroll 1
for (int k = 0; k < 8; ++k) {
const int r = T.frmap[8 * t + k];
if (r < 0) continue;
need += 256;
int lo = 0, hi = T.nj;
while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (T.jS[mid] <= r) lo = mid; else hi = mid; }
credit += max(0, min(b1, (lo + 1) * 256) - max(b0, lo * 256));
}
// Sum stage credits, not CTA counts: all four WGs have returned. BRED partials
// and whole-unit stores are covered, without a per-unit hot-loop publication.
T.fready[t] = need == 0 ? blockIdx.x == (unsigned)t % gridDim.x :
 (credit > 0 && atom_acqrel_add(A.gbar + 1 + t, credit) + credit == need);
}
__syncthreads();  // propagate the winning thread's acquire to both finisher groups
if (tid < 512) {
const int group = tid >> 8;
FinScratch& F = reinterpret_cast<FinScratch*>(smem)[group]; // weight ring is dead
for (int t = group; t < 32; t += 2) if (T.fready[t]) vmlfin_token(A, T, t, F);
}
// Nobody waits for another CTA. Elect one cleanup owner only after finalizers,
// consumers and producers on every CTA have finished all scratch accesses.
__threadfence();
__syncthreads();
if (tid == 0) T.flast = atom_acqrel_add(A.gbar, 1) + 1 == (int)gridDim.x;
__syncthreads();
if (T.flast && tid < 256) {
for (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;
for (int i = tid; i < JMAX * 16; i += 256) A.doneA[i] = 0;
for (int i = tid; i < 2 * SLOTSD * 4; i += 256) A.cnt[i] = 0;
if (tid < 33) A.gbar[tid] = 0;
}
}
#endif
'''

BANK = r'''
VMLFIN=False
_VMLFIN={"mod":None,"ok":False,"sc":None,"build_s":0.0,"err":None,"checks":[]}

def _vmlfin_build():
 from torch.utils.cpp_extension import load_inline
 try:
  s=_vmlfin_src()
  fl=_vml_FLAGS+list(_VML_DEFS["bf2"])+["-DVM_VMLFIN=1"]
  tag=hashlib.sha1((_vml_CPP+s+" ".join(fl)).encode()).hexdigest()[:10]
  t0=time.time()
  _VMLFIN["mod"]=load_inline(name=f"vml_moe_{tag}",cpp_sources=_vml_CPP,cuda_sources=s,functions=["vml_run","vml_cfg"],extra_cuda_cflags=fl,verbose=False)
  _VMLFIN["build_s"]=time.time()-t0
 except Exception as ex:
  _VMLFIN["err"]=repr(ex)[-600:]

def _vmlfin_prepare(st,w,dev):
 global VMLFIN
 selected=VMLFIN
 VMLFIN=False
 _VMLFIN["ok"]=False
 _VMLFIN["checks"]=[]
 try:
  _pt1_wait("vmlfin")
  if _VMLFIN["mod"]is None and _VMLFIN["err"]is None:
   _vmlfin_build()
  if _VMLFIN["mod"]is None or not _VML["ok"].get(32):
   return
  sms=torch.cuda.get_device_properties(dev).multi_processor_count
  _VMLFIN["sc"]=dict(_vml_alloc(dev,sms),err=_VML["sc"]["err"],gbar=torch.zeros((33,),dtype=torch.int32,device=dev))
  good=True
  for seed,dist in ((49,"skewed"),(54,"normal"),(61,"none"),(62,"local"),(63,"duplicate"),(64,"normal")):
   d=_k2a_self_test_inputs(32,seed,dev,dist)
   if dist=="none": d["topk_ids"].fill_(-1)
   if dist=="local": d["topk_ids"].copy_(torch.arange(8,dtype=torch.int32,device=dev).expand(32,8))
   if dist=="duplicate": d["topk_ids"].fill_(0)
   d={"hidden_states":d["hidden_states"].contiguous(),"topk_ids":d["topk_ids"].to(torch.int32).contiguous(),
      "topk_weights":d["topk_weights"].float().contiguous(),"config":{"local_start":0,"local_count":K2A_E_LOC}}
   ref=_k2a_ref_fp32(d,w)
   _VML["sc"]["err"].zero_()
   VMLFIN=False
   o0=_vml_route(st,d).clone()
   o1=_vml_route(st,d).clone()
   VMLFIN,_VMLFIN["ok"]=True,True
   oa=_vml_route(st,d).clone()
   ob=_vml_route(st,d).clone()
   VMLFIN,_VMLFIN["ok"]=False,False
   torch.cuda.synchronize()
   errf=int(_VML["sc"]["err"][0])
   tot,row=_k2a_rel_err(oa,ref)
   ev=_k2a_rel_err(oa,o0)[0]
   rp=_k2a_rel_err(ob,oa)[0]
   dd=float((oa.float()-o0.float()).abs().max())
   rep0=float((o1.float()-o0.float()).abs().max())
   fin=bool(torch.isfinite(oa.float()).all()and torch.isfinite(ob.float()).all())
   clean=all(bool((_VMLFIN["sc"][key]==0).all())for key in("cnt","xready","doneA","gbar"))
   ok=fin and clean and not errf and tot<1e-2 and row<5e-2 and ev<2e-3 and rp<1e-3
   _VMLFIN["checks"].append(dict(seed=seed,dist=dist,ok=bool(ok),fp32=tot,row=row,vs_base=ev,repeat=rp,max_abs=dd,base_repeat=rep0,clean=clean))
   print(f"[vmlfin] {seed} {dist}: fp32={tot:.2e} row={row:.2e} base={ev:.2e} max={dd:.2e} repeat={rp:.2e} clean={clean} err={errf} ok={ok}",flush=True)
   good &= ok
  _VMLFIN["ok"]=bool(good)
 except Exception as ex:
  _VMLFIN["ok"]=False
  _VMLFIN["err"]=repr(ex)[-600:]
 finally:
  VMLFIN=selected
'''


def generate(base):
    compile(base, str(BASE), "exec")
    # Preserve the base CUDA literal and builder byte for byte in the off arm.
    s = sub(base, 'def _vml_route(st,data):\n',
            '\n# VMLFIN BEGIN (tools/vmlfin_patches.py)\n' + SOURCE +
            '\n_VMLFIN_TAIL=' + repr(TAIL) + '\n' + BANK +
            '\n# VMLFIN END\ndef _vml_route(st,data):\n')
    anchor = ' return _vml_call(_VML["mods"].get(var,_VML["mod"]),st,scx,x,data["topk_ids"].to(torch.int32).contiguous(),\n'
    s = sub(s, anchor, ' mz=_VML["mods"].get(var,_VML["mod"])\n'
            ' if VMLFIN and x.shape[0]==32 and var=="bf2":\n'
            '  assert _VMLFIN["ok"] and _VMLFIN["mod"]is not None,"VMLFIN arm unavailable; inspect _VMLFIN"\n'
            '  mz,scx=_VMLFIN["mod"],_VMLFIN["sc"]\n'
            ' return _vml_call(mz,st,scx,x,data["topk_ids"].to(torch.int32).contiguous(),\n')
    s = sub(s, '  _m32e_vartest(st,w,dev)\n', '  _m32e_vartest(st,w,dev)\n  _vmlfin_prepare(st,w,dev)\n')
    s = sub(s, '  jobs.append(("vml",_vml_build_job))\n',
            '  jobs.append(("vml",_vml_build_job))\n  jobs.append(("vmlfin",_vmlfin_build))\n')
    compile(s, str(OUT), "exec")
    assert s.isascii() and "\r" not in s and len(s.encode()) < 1_000_000
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    base = BASE.read_text(encoding="utf-8")
    s = generate(base)
    if args.check:
        assert OUT.read_bytes() == s.encode("ascii"), "generated file drift"
    else:
        OUT.write_bytes(s.encode("ascii"))
    print({"generated": str(OUT.relative_to(ROOT)), "bytes": len(s),
           "sha256": hashlib.sha256(s.encode()).hexdigest(), "check_only": args.check})


if __name__ == "__main__":
    main()

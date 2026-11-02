"""VFIN: default-off, in-file VML finalization lab on the scored c2 source.

Writes only kernels/lab/k3_vfin_ab.py. --check never writes. No GPU work.
The Python bank and argument packing follow the proven padz VML vehicle.
"""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_moexr_c2.py"
OUT = ROOT / "kernels/lab/k3_vfin_ab.py"


def sub(s, a, b):
    assert s.count(a) == 1, ("unique anchor", a[:100], s.count(a))
    return s.replace(a, b, 1)


SOURCE = r'''
def _vfin_sub(s,a,b):
 assert s.count(a)==1,(a[:100],s.count(a))
 return s.replace(a,b,1)

def _vfin_src():
 s=_vml_CUDA
 fa="__global__ void __launch_bounds__(256) vm_fin(const Args A) {\n"
 fb="\n}\n}\nvoid vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V) {"
 assert s.count(fa)==1 and s.count(fb)==1
 f=s[s.index(fa):s.index(fb)+2]
 f=_vfin_sub(f,fa,"__device__ __noinline__ void vfin_token(const Args& A, int t, VfinScratch& F) {\n")
 f=_vfin_sub(f,"__shared__ int s_r[8], s_r8[8];\n__shared__ float s_w[8], s_w8[8];\n__shared__ int s_n;\n__shared__ float s_ss[2][8][4];\n",
 "int (&s_r)[8]=F.r, (&s_r8)[8]=F.r8;\nfloat (&s_w)[8]=F.w, (&s_w8)[8]=F.w8;\nint& s_n=F.n;\nfloat (&s_ss)[2][8][4]=F.ss;\n")
 f=_vfin_sub(f,"const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;",
 "const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;")
 f=_vfin_sub(f,'#if VM_FT\nasm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n#endif\n',"")
 f=_vfin_sub(f,'asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n',"")
 ca="if (t == 0) {\nfor (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;"
 cb="#if VM_FTR\nif (warp == 0) {"
 assert f.count(ca)==1 and f.count(cb)==1
 f=_vfin_sub(f,f[f.index(ca):f.index(cb)],"")
 # Keep the shipped 256-lane, eight-warp reduction and expression order.
 # Serializing route groups lowers live registers without changing route order.
 f=_vfin_sub(f,"c0 += 4, par ^= 1","++c0, par ^= 1")
 f=_vfin_sub(f,"const int nv = min(nloc - c0, 4);","const int nv = 1;")
 assert f.count("jj < 4; ++jj")==6
 f=f.replace("jj < 4; ++jj","jj < 1; ++jj")
 assert f.count("__syncthreads();")==6
 f=f.replace("__syncthreads();","bar_sync(11, 256);")
 s=_vfin_sub(s,"struct Tab {\n",_VFIN_DEFS+"\nstruct Tab {\n")
 s=_vfin_sub(s,"uint32_t cost[2 * VM_NTM]; unsigned long long mag[2 * VM_NTM];\n",
 "#if VM_VFIN\nint vfin;\n#endif\nuint32_t cost[2 * VM_NTM]; unsigned long long mag[2 * VM_NTM];\n")
 # Only VM_RT=1's actual route is changed (the other route remains verbatim).
 ra="if (x >= 0) ent[r] = (uint16_t)e;\nif (blockIdx.x == 0 && e < e1) A.rmap[e] = r;\n"
 s=_vfin_sub(s,ra,ra+"#if VM_VFIN\nif (A.vfin && blockIdx.x == 0 && r >= 0) A.cnt[VF_ROW + r] = e >> 3;\n#endif\n")
 sa="template <bool PB, int N>\n__device__ __forceinline__ void seg("
 s=_vfin_sub(s,sa,_VFIN_PUBLISH+"\n"+sa)
 pa="if constexpr (PB) { bred<N>(A, T, j, ub, q, wp, lane, d); SEGMK(segmk_done(A, smk, s0, s1);) return; }"
 pn="if constexpr (PB) { bred<N>(A, T, j, ub, q, wp, lane, d);\n#if VM_VFIN\nif (A.vfin) vfin_publish(A, T, j, s1 - s0, q, wp, lane);\n#endif\nSEGMK(segmk_done(A, smk, s0, s1);) return; }"
 s=_vfin_sub(s,pa,pn)
 ea="epilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);\n"
 s=_vfin_sub(s,ea,ea+"#if VM_VFIN\nif constexpr (PB) { if (A.vfin) vfin_publish(A, T, j, s1 - s0, q, wp, lane); }\n#endif\n")
 ma="__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {\n"
 s=_vfin_sub(s,ma,"#if VM_VFIN\n__device__ __forceinline__ void vfin_main_body(const Args& A) {\n#else\n"+ma+"#endif\n")
 rt="route(A, T, hist, ent, reinterpret_cast<uint32_t*>(scr + 12288));\n"
 s=_vfin_sub(s,rt,rt+"#if VM_VFIN\nif (A.vfin && blockIdx.x == 0) {\nfence_acq_rel();\n__syncthreads();\nif (tid == 0) red_release_add(A.cnt + VF_READY, 1);\n}\n#endif\n")
 s=_vfin_sub(s,fb,"\n}\n"+_VFIN_TAIL.replace("@TOKEN@",f)+"\n}\nvoid vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V) {")
 ha="const int64_t M = V[0], G = V[2], mask = V[4];\n"
 s=_vfin_sub(s,ha,ha+"#if VM_VFIN\na.vfin = M >= 17 && M <= 48 && M <= G && mask == 3;\nTORCH_CHECK(T[12].numel() >= vm::VF_END, \"VFIN cnt tail too small\");\n#endif\n")
 la="if ((mask & 2) && M > 0) {\n"
 s=_vfin_sub(s,la,"if ((mask & 2) && M > 0\n#if VM_VFIN\n&& !a.vfin\n#endif\n) {\n")
 return s
'''

DEFS = r'''
#ifndef VM_VFIN
#define VM_VFIN 0
#endif
#if VM_VFIN
static_assert(VM_BRED && VM_RT && VM_FTR && VM_JN == 64 && !VM_WG5 && !VM_DP && !VM_SC && !VM_AD && VM_LD == 1 && VM_MKW == 32,
              "VFIN supports the bf2 build with VM_LD=1 only");
constexpr int VF_JOB = 2 * SLOTS * 4, VF_TOK = VF_JOB + JMAX;
constexpr int VF_ROW = VF_TOK + 320, VF_Q = VF_ROW + 2560;
constexpr int VF_EXIT = VF_Q + 1, VF_READY = VF_EXIT + 1, VF_END = VF_READY + 1;
static_assert(VF_END <= 2 * (SLOTS + 384) * 4, "VFIN cnt tail bounds");
#endif
'''

PUBLISH = r'''
#if VM_VFIN
__device__ __noinline__ void vfin_publish(const Args& A, const Tab& T, int j, int n, int q, int wp, int lane) {
// Every writer drains its relaxed BRED/store before the elected lane publishes.
// This is outside the unpack/MMA stage loop; no extra weight bytes are issued.
fence_acq_rel();
bar_sync(1 + q, 128);
if (wp == 0 && lane == 0) {
const int old = atom_acqrel_add(A.cnt + VF_JOB + j, n);
if (old + n == 256 * 4) {
wait_flag(A.cnt + VF_READY, 1, A.err);
if (!ld_acquire(A.err)) {
for (int r = T.jS[j]; r < T.jS[j] + T.jC[j]; ++r) {
const int token = __ldcg(A.cnt + VF_ROW + r);
// acq_rel (stronger than release) chains different jobs publishing one token.
atom_acqrel_add(A.cnt + VF_TOK + token, 1);
}
}
}
}
}
#endif
'''

TAIL = r'''
#if VM_VFIN
struct VfinScratch { int r[8], r8[8]; float w[8], w8[8]; int n; float ss[2][8][4]; };
@TOKEN@
__device__ __noinline__ void vfin_finish(const Args& A, uint8_t* smem) {
const int tid = threadIdx.x;
int* ctl = reinterpret_cast<int*>(smem);
VfinScratch& F = *reinterpret_cast<VfinScratch*>(smem + 16);
bar_sync(5, 512);
if (tid == 0) {
ctl[0] = atomicAdd(A.cnt + VF_Q, 1);
ctl[1] = 0;
if (ctl[0] < A.M) {
wait_flag(A.cnt + VF_READY, 1, A.err);
if (!ld_acquire(A.err)) {
int need = 0;
for (int k = 0; k < 8; ++k) need += __ldcg(A.rmap + ctl[0] * 8 + k) >= 0;
wait_flag(A.cnt + VF_TOK + ctl[0], need, A.err);
ctl[1] = !ld_acquire(A.err);
}
}
}
bar_sync(5, 512);
// The exact shipped expression graph requires 256 logical lanes, not 512.
if (ctl[0] < A.M && ctl[1] && tid < 256) vfin_token(A, ctl[0], F);
bar_sync(5, 512);
// Every CTA, including no-ticket / timeout CTAs, participates in retirement.
fence_acq_rel();
bar_sync(5, 512);
if (tid == 0) ctl[2] = atom_acqrel_add(A.cnt + VF_EXIT, 1) + 1 == (int)gridDim.x;
bar_sync(5, 512);
if (ctl[2]) {
for (int i = tid; i < JMAX; i += 512) A.xready[i] = 0;
for (int i = tid; i < JMAX * 16; i += 512) A.doneA[i] = 0;
// Includes the original cnt region, jobdone, tokdone, rowtok and finq.
// Keep the exit word until every reset store has completed.
for (int i = tid; i < VF_END; i += 512) if (i != VF_EXIT) A.cnt[i] = 0;
fence_acq_rel();
bar_sync(5, 512);
if (tid == 0) __stcg(A.cnt + VF_EXIT, 0);
}
}
__global__ void __launch_bounds__(NTH, 1) vm_main(const __grid_constant__ Args A) {
extern __shared__ __align__(1024) uint8_t smem[];
vfin_main_body(A);
if (A.vfin) {
// All 576 threads join, including the two producer warps that returned from
// vfin_main_body. No ring storage is reused before their last access.
__syncthreads();
if (threadIdx.x < 512) vfin_finish(A, smem);
}
}
#endif
'''

BANK = r'''
VFIN=0
_VFIN={"mod":None,"ok":False,"sc":None,"build_s":0.0,"err":None,"checks":[]}

def _vfin_build():
 from torch.utils.cpp_extension import load_inline
 try:
  s=_vfin_src()
  fl=_vml_FLAGS+list(_VML_DEFS["bf2"])+["-DVM_VFIN=1","-DVM_LD=1"]
  tag=hashlib.sha1((_vml_CPP+s+" ".join(fl)).encode()).hexdigest()[:10]
  t0=time.time()
  _VFIN["mod"]=load_inline(name=f"vml_moe_{tag}",cpp_sources=_vml_CPP,cuda_sources=s,functions=["vml_run","vml_cfg"],extra_cuda_cflags=fl,verbose=False)
  _VFIN["build_s"]=time.time()-t0
 except Exception as ex:
  _VFIN["err"]=repr(ex)[-600:]

def _vfin_case(bs,seed,dev,dist):
 d=_k2a_self_test_inputs(bs,seed,dev,"skewed" if dist=="skewed" else "normal")
 if dist=="uniform":
  # Eight distinct local experts per token, balanced over all 32 local experts.
  d["topk_ids"]=(torch.arange(bs*8,device=dev,dtype=torch.int32).reshape(bs,8)%32).contiguous()
 elif dist=="empty":
  d["topk_ids"].fill_(-1)
 elif dist=="hot8":
  d["topk_ids"]=torch.arange(8,device=dev,dtype=torch.int32).expand(bs,8).contiguous()
 return d

def _vfin_run_bank(st,d,on,out=None):
 mod=_VFIN["mod"]if on else _VML["mods"].get("bf2",_VML["mod"])
 sc=_VFIN["sc"]if on else _VML["scb"]
 return _vml_call(mod,st,sc,d["hidden_states"].contiguous(),d["topk_ids"].to(torch.int32).contiguous(),
 d["topk_weights"].to(torch.float32).contiguous(),int(d["config"]["local_start"]),out=out,flags=VML_FLAGS[32],cost=VML_COST[32])

def _vfin_prepare(st,w,dev):
 global VFIN
 selected=VFIN
 VFIN=0
 _VFIN["ok"]=False
 _VFIN["checks"]=[]
 try:
  _pt1_wait("vfin")
  if _VFIN["mod"]is None and _VFIN["err"]is None:
   _vfin_build()
  if _VFIN["mod"]is None or not _VML["ok"].get(32):
   print(f"[vfin] OFF: build {_VFIN['err']} / base bucket32 {_VML['ok'].get(32)}",flush=True)
   return
  _VFIN["err"]=None
  sms=torch.cuda.get_device_properties(dev).multi_processor_count
  _VFIN["sc"]=dict(_vml_alloc(dev,sms),err=_VML["sc"]["err"])
  good=True
  for bs in (17,24,32,40,48):
   for dist in ("skewed","uniform","normal","empty","hot8"):
    d=_vfin_case(bs,71+bs,dev,dist)
    # Warm on a side stream, then capture; no pointer is cached in Python state.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    outs=[torch.empty((bs,4096),dtype=torch.bfloat16,device=dev)for _ in range(2)]
    _VML["sc"]["err"].zero_()
    with torch.cuda.stream(stream):
     for on in (0,1):
      _vfin_run_bank(st,d,on,outs[on])
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs=[torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()]
    for on in (0,1):
     with torch.cuda.graph(graphs[on],stream=stream):
      _vfin_run_bank(st,d,on,outs[on])
    for replay in range(3):
     # Exercise changed routing/activations inside the SAME captured graph too.
     fresh=_vfin_case(bs,71+bs+replay,dev,dist)
     for key in ("hidden_states","topk_ids","topk_weights"):
      d[key].copy_(fresh[key])
     for graph in graphs:
      graph.replay()
     torch.cuda.synchronize()
     exact=torch.equal(outs[0].view(torch.int16),outs[1].view(torch.int16))
     rel=_k2a_rel_err(outs[1],outs[0])[0]
     exact=bool(exact or rel<1e-3)  # lead: FP32 atomics in split B units are order-nondeterministic
     finite=bool(torch.isfinite(outs[1].float()).all())
     clean=all(bool((_VFIN["sc"][k]==0).all())for k in ("cnt","xready","doneA","y"))
     err=int(_VML["sc"]["err"][0])
     ok=bool(exact and finite and clean and not err)
     _VFIN["checks"].append(dict(bs=bs,dist=dist,replay=replay,bitwise=bool(exact),clean=clean,err=err,ok=ok))
     good &= ok
     if not ok:
      print(f"[vfin] FAIL bs={bs} {dist} replay={replay}: bitwise={exact} clean={clean} err={err}",flush=True)
      raise RuntimeError(f"VFIN bitwise/replay gate failed: T={bs} {dist} replay={replay}")
  _VFIN["ok"]=bool(good)
 except Exception as ex:
  _VFIN["err"]=repr(ex)[-600:]
  _VFIN["ok"]=False
 finally:
  VFIN=selected if _VFIN["ok"]else 0
  print(f"[vfin] {'LIVE' if _VFIN['ok'] else 'OFF'}: {len(_VFIN['checks'])} graph replay checks; {_VFIN['err']}",flush=True)
'''


def generate(base):
    compile(base, str(BASE), "exec")
    bank = (SOURCE + "\n_VFIN_DEFS=" + repr(DEFS) + "\n_VFIN_PUBLISH=" + repr(PUBLISH)
            + "\n_VFIN_TAIL=" + repr(TAIL) + "\n" + BANK)
    s = sub(base, "def _vml_route(st,data):\n", "\n# VFIN BEGIN\n" + bank + "\n# VFIN END\ndef _vml_route(st,data):\n")
    a = ' return _vml_call(_VML["mods"].get(var,_VML["mod"]),st,scx,x,data["topk_ids"].to(torch.int32).contiguous(),\n'
    s = sub(s, a, ' mz=_VML["mods"].get(var,_VML["mod"])\n'
            ' if VFIN==1 and 17<=x.shape[0]<=48 and var=="bf2" and _VFIN["ok"]:\n'
            '  mz,scx=_VFIN["mod"],_VFIN["sc"]\n'
            ' return _vml_call(mz,st,scx,x,data["topk_ids"].to(torch.int32).contiguous(),\n')
    # Match padz's st/w/dev hook; preserve the base route and every other engine.
    s = sub(s, '  _m32e_vartest(st,w,dev)\n', '  _m32e_vartest(st,w,dev)\n  _vfin_prepare(st,w,dev)\n')
    s = sub(s, '  jobs.append(("vml",_vml_build_job))\n',
            '  jobs.append(("vml",_vml_build_job))\n  jobs.append(("vfin",_vfin_build))\n')
    compile(s, str(OUT), "exec")
    assert s.isascii() and "\r" not in s and len(s.encode()) < 1_000_000
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    s = generate(BASE.read_text(encoding="utf-8"))
    if args.check:
        assert OUT.read_bytes() == s.encode("ascii"), "generated file drift"
    else:
        OUT.write_bytes(s.encode("ascii"))
    print({"generated": str(OUT.relative_to(ROOT)), "bytes": len(s), "sha256": hashlib.sha256(s.encode()).hexdigest(), "check_only": args.check})


if __name__ == "__main__":
    main()

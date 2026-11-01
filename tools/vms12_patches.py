"""VMS12: default-off, separately built hi module; no GPU work in this generator.

Only writes kernels/lab/k3_vms12_ab.py. Run with python -B -I.
The compiler/CPU gate is tools/local/vms12_compile.py (also a bash entry point).
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/lab/k3_vms12_ab.py"


def once(s, old, new):
    assert s.count(old) == 1, ("anchor must occur exactly once", old[:140], s.count(old))
    return s.replace(old, new, 1)


def literal(s, name):
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


def cuda12(src):
    """Separate source; the original CUDA string and both original builds survive verbatim."""
    s = src
    s = once(s, "namespace vms {", """#ifndef VMS_NW
#define VMS_NW 12
#endif
#if VMS_NW != 12 || !VMS_T16 || !VMS_RO || !VMS_ER || VMS_T16R || VMS_T16E || VMS_MK2
#error VMS12_requires_the_unmarked_hi_ER_shared_x_build
#endif
namespace vms {""")
    s = once(s, "constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 128, MAXJ = 64;",
             "constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 192, MAXJ = 64;")
    s = once(s, "constexpr int NW = 8, NT = NW * 32, NTL = NT + 32, NRB = VMS_NRB, NS = VMS_NS, G = VMS_G;",
             "constexpr int NW = VMS_NW, NT = NW * 32, NTL = NT + 32, NRB = VMS_NRB, NS = VMS_NS, G = VMS_G;")
    s = once(s, "  uint64_t pempty[NRB];", "  uint64_t pempty[NRB];\n  uint64_t pairfull[NRB][4];")
    s = once(s, "int base[4][NE];", "int base[6][NE];")
    # Keep eight helper inputs. Donors temporarily occupy slots 4..7.
    s = once(s, "float4 red[NRB][NW][32];", "float4 red[NRB][8][32];")
    s = once(s, "float4 red2[NRB][NW][32];", "float4 red2[NRB][8][32];")
    s = once(s, "  uint4 x2s[NW][16][32];       // R-vt16: tokens 8-15 x, one private uint4 column per lane",
             "  uint4 x2s[128][32]; // Exactly 64 blocks * two uint4 fragments, unchanged 64 KiB")
    s = once(s, "mbar_init(&S.pfull[i], NW); mbar_init(&S.pempty[i], 1);",
             "mbar_init(&S.pfull[i], 8); mbar_init(&S.pempty[i], 1);\n"
             "for (int z = 0; z < 4; ++z) mbar_init(&S.pairfull[i][z], 1);")
    # Expand only the new module's route. MAXP includes all 24*8 local records.
    ra = s.index("if (warp == 0) {\nconst int m8 = p.m * 8;")
    rz = s.index("\n__syncthreads();", ra)
    original_route = s[ra:rz]
    route = original_route
    route = once(route, "int idv[4]; unsigned mm[4];", "int idv[6]; unsigned mm[6];")
    route = once(route, "int cr[4], tot = 0;", "int cr[6], tot = 0;")
    # Assert each full line independently (each includes a different loop body).
    lines = route.splitlines()
    for i, line in enumerate(lines):
        if "for (int r = 0; r < 4; ++r)" in line:
            anchor = line + "\n" + lines[i + 1]
            route = once(route, anchor, anchor.replace("r < 4", "r < 6"))
    s = once(s, original_route, route)
    start = s.index("if (warp < NW) {\n#if !((VMS_FX & 4)")
    end = s.index("} else {\nconst int g = lane >> 2, t = lane & 3;\nfloat* ssc", start)
    body = s[start + len("if (warp < NW) {\n"):end]
    # Specialize the entire consumer, with compile-time xr[12] / xr[10].
    body = once(body, "uint4 xr[16];", "uint4 xr[2 * NC];")
    body = once(body, "uint4 wv[8];", "uint4 wv[NC];")
    body = once(body, "    uint4* x2p = &S.x2s[warp][0][lane];   // R-vt16: this lane's tokens 8-15 x",
                "    uint4* x2p = &S.x2s[2 * block0][lane];")
    body = once(body, "src += warp * 64 + 2 * t;", "src += block0 * 8 + 2 * t;")
    body = once(body, "src2 += warp * 64 + 2 * t;", "src2 += block0 * 8 + 2 * t;")
    # Full exact loop regions disambiguate otherwise repeated cc-loop anchors.
    body = once(body, "for (int cc = 0; cc < 8; ++cc) {\nxr[2 * cc]", "for (int cc = 0; cc < NC; ++cc) {\nxr[2 * cc]")
    body = once(body, "for (int cc = 0; cc < 8; ++cc) {\nT16X2(2 * cc)", "for (int cc = 0; cc < NC; ++cc) {\nT16X2(2 * cc)")
    body = once(body, "for (int cc = 0; cc < 16; ++cc) xr[cc]", "for (int cc = 0; cc < 2 * NC; ++cc) xr[cc]")
    body = once(body, "for (int cc = 0; cc < 16; ++cc) T16X2(cc)", "for (int cc = 0; cc < 2 * NC; ++cc) T16X2(cc)")
    body = once(body, "const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16);",
                "uint32_t scales[NC];\n#pragma unroll\nfor (int cc = 0; cc < NC; ++cc)\n"
                "scales[cc] = *reinterpret_cast<const uint16_t*>(slot + UC + (block0 + cc) * 64 + lane * 2);")
    body = once(body, "for (int cc = 0; cc < 8; ++cc) wv[cc] = *reinterpret_cast<const uint4*>(slot + (warp * 8 + cc) * 512 + lane * 16);",
                "for (int cc = 0; cc < NC; ++cc) wv[cc] = *reinterpret_cast<const uint4*>(slot + (block0 + cc) * 512 + lane * 16);")
    # Patch each of the three complete arithmetic loops once, then join them.
    for head, tail in (("if (two) {\n#pragma unroll\n", "\no2.x ="),
                       ("#endif\n#endif\n#pragma unroll\nfor (int cc", "\n#if VMS_T16 && VMS_T16E"),
                       ("#if VMS_T16 && VMS_T16E\nelse {\n#pragma unroll\n", "\no2.x =")):
        ia = body.index(head)
        iz = body.index(tail, ia)
        old = body[ia:iz]
        new = once(old, "cc < 8", "cc < NC")
        new = once(new, "(warp * 8 + cc) * 512", "(block0 + cc) * 512")
        new = once(new, "const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);",
                   "const uint32_t sw = scales[cc];")
        new = once(new, "const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;", "const uint32_t sel = 0x4040u;")
        new = once(new, "scd(sw, (cc & 1) ? 0x4342u : 0x4140u, k16)", "scd(sw, 0x4140u, k16)")
        body = once(body, old, new)
    pub = """S.red[bq][warp][lane] = o;
#if VMS_T16
if (two) S.red2[bq][warp][lane] = o2;
#endif
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pfull[bq]);"""
    body = once(body, pub, """// Pair (4+s,8+s) on SMSP s; warp s stays independent.
if (warp >= 8) {
  S.red[bq][warp - 4][lane] = o;
  if (two) S.red2[bq][warp - 4][lane] = o2;
  __syncwarp();
  if (lane == 0) mbar_arrive(&S.pairfull[bq][warp - 8]);
} else {
  if (warp >= 4) {
    mbar_wait(&S.pairfull[bq][warp - 4], (uint32_t)((q / NRB) & 1));
    const float4 u = S.red[bq][warp][lane];
    o.x += u.x; o.y += u.y; o.z += u.z; o.w += u.w;
    if (two) {
      const float4 u2 = S.red2[bq][warp][lane];
      o2.x += u2.x; o2.y += u2.y; o2.z += u2.z; o2.w += u2.w;
    }
  }
  S.red[bq][warp][lane] = o;
  if (two) S.red2[bq][warp][lane] = o2;
  __syncwarp();
  if (lane == 0) mbar_arrive(&S.pfull[bq]);
}""")
    fn = """template<int NC>
__device__ __forceinline__ void vms12_compute(const Params& p, Smem& S, unsigned char* ring,
                                            int a0, int a1, int b0, int nA, int N) {
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
const int block0 = (warp & 3) * 16 + (NC == 6 ? 0 : (warp < 8 ? 6 : 11));
const bool mk = p.marks != nullptr;
long long* mrow = p.marks + (size_t)blockIdx.x * 16;
""" + body + "}\n"
    s = once(s, s[start:end], """if (warp < NW) {
if (warp < 4) vms12_compute<6>(p, S, ring, a0, a1, b0, nA, N);
else vms12_compute<5>(p, S, ring, a0, a1, b0, nA, N);
""")
    s = once(s, "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {", fn + "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {")
    s = once(s, "for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane];",
             "for (int w = 1; w < 8; ++w) { const float4 u = S.red[bq][w][lane];")
    s = once(s, "for (int w = 0; w < NW; ++w) { const float4 u = S.red2[bq][w][lane];",
             "for (int w = 0; w < 8; ++w) { const float4 u = S.red2[bq][w][lane];")
    # The final combine is a 256-thread map, irrespective of compute warp count.
    s = once(s, "if (tid >= NT) break;", "if (tid >= 256) break;")
    s = once(s, 'TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");',
             'TORCH_CHECK(m >= 9 && m <= 24, "vms12: 9 <= m <= 24");')
    assert "__trap" not in s and "uint4 xr[2 * NC]" in s
    return s


GLUE = '''
# VMS12 experimental bank: default off, chosen at call/capture time.
VMS12 = False
VMS12_FLAGS = ["-O3", "-DVMS_G=132", "-DVMS_NS=2", "-DVMS_RO=1", "-DVMS_T16=1", "-DVMS_WT2=30", "-DVMS_ER=1", "-DVMS_NW=12", "--maxrregcount=128", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]

def _vms12_build():
 from torch.utils.cpp_extension import load_inline
 assert torch.cuda.get_device_properties(0).multi_processor_count == 132
 tag=hashlib.sha1((VMS12_CUDA_SRC+" ".join(VMS12_FLAGS)).encode()).hexdigest()[:10]
 return load_inline(name=f"k3_vms12_{tag}",cpp_sources=VMS_CPP_SRC,cuda_sources=VMS12_CUDA_SRC,
 functions=["run","attrs"],extra_cuda_cflags=VMS12_FLAGS)

def _vms12_repack(src):
 dst=src.clone()
 e,u,_=src.shape
 dst[:,:,32768:]=src[:,:,32768:].view(e,u,8,32,8,2).permute(0,1,2,4,3,5).contiguous().view(e,u,4096)
 return dst

def _vms12_bank(vs,mod,dev):
 return {"mod":mod,"ok":False,"error":None,"checks":[],
 "wA":_vms12_repack(vs["wA"]),"wB":_vms12_repack(vs["wB"]),
 "gsA":vs["gsA"],"gsB":vs["gsB"],"nw":vs["nw"],
 "inter":torch.zeros((192,K2A_D_EXPERT),dtype=torch.bfloat16,device=dev),
 "ybuf":torch.zeros((192,K2A_D_HIDDEN),dtype=torch.float32,device=dev),
 "ss":torch.zeros((384,),dtype=torch.float32,device=dev),
 "ctr":torch.zeros((72,),dtype=torch.int32,device=dev),"nomarks":vs["nomarks"]}

def _vms12_run(vs,md,x,local_ids,topk_weights):
 ids=local_ids.to(torch.int32).contiguous()
 tw=topk_weights.to(torch.float32).contiguous()
 m=x.shape[0]
 out=torch.empty((m,K2A_D_HIDDEN),dtype=torch.bfloat16,device=x.device)
 md.run(x,ids,tw,vs["wA"],vs["wB"],vs["gsA"],vs["gsB"],vs["nw"],vs["inter"],vs["ybuf"],vs["ss"],vs["ctr"],
 out,vs["nomarks"],m,K2A_POST_NORM_EPS,1,0)
 return out

def _vms12_route(st,x,local_ids,topk_weights):
 b=st.get("vms12")
 if VMS12 and VMS_ENABLE and 9<=x.shape[0]<=24 and b is not None and b["ok"]:
  return _vms12_run(b,b["mod"],x,local_ids,topk_weights)
 return None

def _vms12_refs(d,w):
 # Same FP32 arithmetic as _k2a_ref_fp32; retain pre-normalization intermediates.
 x=d["hidden_states"].float(); ids=d["topk_ids"].long()
 out=torch.zeros(x.shape,dtype=torch.float32,device=x.device)
 ir=[]; yr=[]
 for e in torch.unique(ids).tolist():
  if not 0<=e<K2A_E_LOC: continue
  tok,slot=(ids==e).nonzero(as_tuple=True)
  h=x[tok]@_k2a_dequant(w["w13"][e],w["w13_sf"][e],w["w13_gs"][e]).T
  inter=torch.nn.functional.silu(h[:,:K2A_D_EXPERT])*h[:,K2A_D_EXPERT:]
  y=(inter@_k2a_dequant(w["w2"][e],w["w2_sf"][e],w["w2_gs"][e]).T)*(1.0+w["post_norm_w"].float())
  ir.append(inter); yr.append(y)
  yn=y*torch.rsqrt(y.pow(2).mean(-1,keepdim=True)+K2A_POST_NORM_EPS)
  out.index_add_(0,tok,yn*d["topk_weights"][tok,slot].float().unsqueeze(1))
 return out.to(torch.bfloat16), (torch.cat(ir) if ir else None), (torch.cat(yr) if yr else None)

def _vms12_check(o,b,refs,label,baseline=None):
 ref,ri,ry=refs
 checks={"out":_k2a_rel_err(o,ref)}
 if ri is not None:
  checks["inter"]=_k2a_rel_err(b["inter"][:ri.shape[0]],ri)
  checks["ybuf"]=_k2a_rel_err(b["ybuf"][:ry.shape[0]],ry)
 if baseline is not None: checks["base"]=_k2a_rel_err(o,baseline)
 good=bool(torch.isfinite(o.float()).all()) and not int(b["ctr"][2])
 for key,(total,row) in checks.items():
  good=good and total<(8e-3 if key=="base" else 1e-2) and row<5e-2
 print(f"[vms12] {label} rel(total,row)={checks} err={int(b['ctr'][2])} -> {good}",flush=True)
 b["checks"].append({"label":label,"rel":checks,"ok":bool(good)})
 if not good: raise RuntimeError(f"VMS12 self-test failed: {label}")

def _vms12_self_test(st,w,dev):
 b=st["vms12"]; vs=st["vms"]
 cases=((9,"normal"),(12,"normal"),(13,"hot13"),(16,"normal"),(16,"hot13"),(24,"all_local"),(24,"empty"))
 for ci,(m,dist) in enumerate(cases):
  d=_k2a_self_test_inputs(m,731+ci,dev,"normal")
  if dist=="hot13":
   d["topk_ids"]=torch.full((m,8),-1,dtype=torch.int32,device=dev)
   d["topk_ids"][:13,0]=0
  elif dist=="all_local":
   d["topk_ids"]=torch.arange(8,device=dev,dtype=torch.int32).expand(m,8).contiguous()
  elif dist=="empty":
   d["topk_ids"]=torch.full((m,8),-1,dtype=torch.int32,device=dev)
  x=d["hidden_states"].contiguous(); ids=d["topk_ids"]; tw=d["topk_weights"]
  refs=_vms12_refs(d,w)
  baseline=None
  if m<=16:
   baseline=_vms12_run(vs,vs["mod_hi"],x,ids,tw).clone()
   torch.cuda.synchronize()
   rt,rr=_k2a_rel_err(baseline,refs[0])
   if not(rt<1e-2 and rr<5e-2 and not int(vs["ctr"][2])): raise RuntimeError("base self-test failed")
  for replay in range(2):
   b["inter"].fill_(float("nan")); b["ybuf"].fill_(float("nan"))
   o=_vms12_run(b,b["mod"],x,ids,tw)
   torch.cuda.synchronize()
   _vms12_check(o,b,refs,f"T={m} {dist} eager={replay}",baseline)
  # Warmed kernel attributes and allocations; only _vms12_run is captured.
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph): o=_vms12_run(b,b["mod"],x,ids,tw)
  for replay in range(2):
   b["inter"].fill_(float("nan")); b["ybuf"].fill_(float("nan"))
   graph.replay(); torch.cuda.synchronize()
   _vms12_check(o,b,refs,f"T={m} {dist} graph={replay}",baseline)

def _vms12_prepare(st,w,dev):
 st["vms12"]={"ok":False,"error":None,"checks":[]}
 if not VMS_ENABLE or not st.get("vms_ok"): return
 vs=st["vms"]
 if vs.get("mod_hi") is None: return
 try:
  mod=_vms12_build()
  reg,local,smem,g,ns=map(int,mod.attrs())
  # Four resident warps on SMSP0: 4*32*round_up_8(R)<=16384 -> R<=128.
  if not(0<reg<=128 and local==0 and smem<=227328 and g==132 and ns==2):
   raise RuntimeError(f"resource gate: {reg,local,smem,g,ns}")
  st["vms12"]=_vms12_bank(vs,mod,dev)
  _vms12_self_test(st,w,dev)
  st["vms12"]["ok"]=True
  print(f"[vms12] enabled bank; selector VMS12={VMS12}; attrs={reg,local,smem,g,ns}",flush=True)
 except Exception as ex:
  st["vms12"]["ok"]=False
  st["vms12"]["error"]=repr(ex)
  print(f"[vms12] disabled: {ex!r}",flush=True)

'''


def generate(base):
    cu = cuda12(literal(base, "VMS_CUDA_SRC"))
    extra = "\nVMS12_CUDA_SRC=r'''" + cu + "'''\n" + GLUE
    result = once(base, "VMS_SELF_TEST_CASES=", extra + "VMS_SELF_TEST_CASES=")
    # Like the proven late HYBLO hook: do not consume any earlier prepare guard's budget.
    result = once(result, " _rq_prepare(state)\n", " _rq_prepare(state)\n"
                  " if state.get(\"k2a33\")is not None:\n"
                  "  _vms12_prepare(state[\"k2a33\"][\"st\"],{k:weights[k]for k in(\"w13\",\"w13_sf\",\"w13_gs\",\"w2\",\"w2_sf\",\"w2_gs\")}|{\"post_norm_w\":weights[\"post_expert_norm\"]},weights[\"w13\"].device)\n")
    anchor = ' data={"hidden_states":x,"topk_ids":local_ids,"topk_weights":topk_weights,"config":{"local_start":0}}\n if HYBLO_T'
    result = once(result, anchor, anchor.replace("\n if HYBLO_T", "\n if VMS12:\n  o=_vms12_route(st,x,local_ids,topk_weights)\n  if o is not None: return o\n if HYBLO_T"))
    compile(result, str(OUT), "exec")
    assert len(result.encode()) < 1_000_000
    return result


def main():
    result = generate(BASE.read_text(encoding="utf-8"))
    OUT.write_bytes(result.encode("utf-8"))
    print(f"VMS12 generated {OUT.name}: {len(result.encode())} bytes; default False")


if __name__ == "__main__":
    main()

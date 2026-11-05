"""S7-UF: anchored, default-off T=1/8 TINYS + unpooler fusion.

Only this generator, box_out/s7/k3_uf_c1.py and tools/local/s7_uf_gate.sh
(and its adjacent JSON report) are task outputs. No device execution here.
WEIGHT TRAFFIC GROWS: lossless BF16 staging replaces GD's P12+escapes.
Run python -B tools/s7_uf_patches.py, then --check for read-only CPU gates.
"""
import argparse
import ast
import builtins
import difflib
import hashlib
import json
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUTPUT = ROOT / "box_out/s7/k3_uf_c1.py"


def sub(s, a, b):
    assert s.count(a) == 1, ("anchor", s.count(a), a[:100])
    return s.replace(a, b)


def node(src, name):
    found = [n for n in ast.parse(src).body if
             isinstance(n, ast.FunctionDef) and n.name == name or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(found) == 1, (name, len(found))
    return found[0]


def literal(src, name):
    return ast.literal_eval(node(src, name).value)


# No Smem layout change: bars[11] is initialized by ship and otherwise unused.
# The fat-x reserve starts after all three 36,864-byte expert ring slots.
UF_HELPERS = r'''
// UF bit 1/2: exact T=1/8 GD arithmetic, raw BF16 weight staging.
static_assert(OKC_UF_T == 1 || OKC_UF_T == 8, "UF exact shapes");
static_assert(OKC_TINYS == 7 && OKC_NS == NSF && OKC_NW == 8, "UF ship protocol");
constexpr int UF_XP = HD + 8; // avoid eight token rows mapping to the same LDS banks
static_assert(65536 <= 2 * FT_T * XP_FT * 2, "UF fits unused fat-x reserve");
static_assert(OKC_UF_T * UF_XP * 2 + 16384 <= OKC_NS * UNIT_BYTES, "UF post-join scratch");
__device__ __forceinline__ unsigned char* uf_weights(__nv_bfloat16* xbuf) {
return reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES;
}
__device__ __forceinline__ void uf_prefetch(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
if (TIDX == 0 && blockIdx.x < 128) {
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 65536;\n"
             :: "r"(smem_u32(S.bars + 11)) : "memory");
asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 65536, [%2];\n"
             :: "r"(smem_u32(uf_weights(xbuf))), "l"(p.uf_w + (size_t)blockIdx.x * 65536),
                "r"(smem_u32(S.bars + 11)) : "memory");
}
}
__device__ __forceinline__ void uf_finish(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
const int tid = TIDX, lane = tid & 31, warp = tid >> 5;
// ALL 132 CTAs already reset ctr[0:2] and doneA before entering here.
// Every output suffix is overwritten on every replay (no self-zeroing atomics).
for (int i = blockIdx.x * NT + tid; i < OKC_UF_T * 8192; i += gridDim.x * NT)
if (i % 8192 >= 1024) p.out[i] = __float2bfloat16_rn(0.f);
if (blockIdx.x >= 128) return;
// Helper joined and returned; the expert ring can now hold the combined BF16 x.
// Preserve ship's eight ordered FP32 weighted sums, then its BF16 rounding.
for (int i = tid; i < OKC_UF_T * (HD / 4); i += NT) {
const int tk = i / (HD / 4), col = (i % (HD / 4)) * 4;
float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
for (int s = 0; s < 8; ++s) {
const int q = S.pos[tk * 8 + s];
if (q >= 0) {
const float r = __ldg(p.topk_w + tk * 8 + s) * rsqrtf(__ldcg(S.ssp + q) * (1.f / HD) + 1e-8f);
const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;
}
}
const __nv_bfloat162 a = __floats2bfloat162_rn(acc.x, acc.y), b = __floats2bfloat162_rn(acc.z, acc.w);
*reinterpret_cast<uint2*>(xbuf + tk * UF_XP + col) = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}
// One elected waiter; no consumer can read the bulk destination on timeout.
if (tid == 0) {
S.unit_next2 = 0;
for (unsigned spin = 0; spin < TINYS_SPIN_LIMIT; ++spin) {
unsigned ready;
asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.shared::cta.b64 q, [%1], 0; selp.u32 %0, 1, 0, q; }\n"
             : "=r"(ready) : "r"(smem_u32(S.bars + 11)) : "memory");
if (ready) { S.unit_next2 = 1; break; }
}
if (!S.unit_next2) tinys_fail(p, 16);
}
CTA_SYNC();
if (!S.unit_next2) {
for (int i = tid; i < OKC_UF_T * 8; i += NT)
p.out[(i / 8) * 8192 + blockIdx.x * 8 + i % 8] = __float2bfloat16_rn(__int_as_float(0x7fc00000));
return;
}
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
// Two serial batches reproduce T=1's 16 GD warps; four reproduce T=8's
// four clusters of eight GD warps. NC=2 (even/odd q) is unchanged.
constexpr int KU = OKC_UF_T == 1 ? 4 : 2, BATCH = OKC_UF_T == 1 ? 2 : 4;
float* red = reinterpret_cast<float*>(xbuf + OKC_UF_T * UF_XP);
const uint2* uw = reinterpret_cast<const uint2*>(uf_weights(xbuf));
#pragma unroll
for (int h = 0; h < BATCH; ++h) {
float a0[4] = {}, a1[4] = {};
#pragma unroll
for (int s = 0; s < KU; ++s) {
const int u = (h * 8 + warp) * KU + s;
const int tk = OKC_UF_T == 1 ? 0 : lane >> 2;
const uint4* xp = reinterpret_cast<const uint4*>(xbuf + tk * UF_XP + u * 64 + (lane & 3) * 16);
const uint4 x0 = xp[0], x1 = xp[1];
#pragma unroll
for (int q = 0; q < 4; ++q) {
const uint2 w = uw[(u * 4 + q) * 32 + lane];
const uint4 xv = (q >> 1) ? x1 : x0;
const uint32_t b0 = (q & 1) ? xv.z : xv.x, b1 = (q & 1) ? xv.w : xv.y;
// Second eight A rows are unused, so zero them; first eight are exact GD bits.
mma_bf16((q & 1) ? a1 : a0, w.x, 0u, w.y, 0u, b0, b1);
}
}
red[((h * 8 + warp) * 2) * 32 + lane] = a0[0] + a1[0];
red[((h * 8 + warp) * 2 + 1) * 32 + lane] = a0[1] + a1[1];
}
CTA_SYNC();
if (tid < 64) {
const int l = tid & 31, c = tid >> 5, tk = 2 * (l & 3) + c;
if (tk < OKC_UF_T) {
float v = 0.f;
if constexpr (OKC_UF_T == 1) {
#pragma unroll
for (int k = 0; k < 16; ++k) v += red[(k * 2 + c) * 32 + l];
} else {
#pragma unroll
for (int h = 0; h < 4; ++h) {
float part = 0.f;
#pragma unroll
for (int k = 0; k < 8; ++k) part += red[((h * 8 + k) * 2 + c) * 32 + l];
// GD initializes own[] from cluster rank 0, then adds ranks 1,2,3.
if (h == 0) v = part; else v += part;
}
}
p.out[tk * 8192 + blockIdx.x * 8 + (l >> 2)] = __float2bfloat16_rn(v);
}
}
// No empty-barrier arrive: this single-use reserve is never reused in a call.
}
'''

EDITS = (
    ("unsigned long long* dbg;\n};", "unsigned long long* dbg;\nconst unsigned char* uf_w; // UF static weights, bound inside timed okc_run\n};"),
    ("template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused", UF_HELPERS + "\ntemplate <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused"),
    ("run_phase<false, BIG>(p, S, xbuf);\nDBG(2);", "run_phase<false, BIG>(p, S, xbuf);\nuf_prefetch(p, S, xbuf); // UF: issue during A->B gap\nDBG(2);"),
    ("constexpr int FSEG = HD / (NT * 4);\nint it0 = blockIdx.x;", "uf_finish(p, S, xbuf); // UF: after every ship cleanup, all 256 consumer threads\nreturn;\nconstexpr int FSEG = HD / (NT * 4);\nint it0 = blockIdx.x;"),
    ("p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;", "p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\nTORCH_CHECK(p.M == OKC_UF_T && grid == 132 && w.size() == 15, \"UF shape/grid/weights\");\nTORCH_CHECK(out.size(0) == p.M && out.size(1) == 8192 && out.is_contiguous(), \"UF output\");\nTORCH_CHECK(w[14].nbytes() == 128 * 65536 && w[14].is_contiguous(), \"UF weights\");\np.uf_w = (const unsigned char*)w[14].data_ptr();"),
)

GLUE = '''# S7-UF: default-off selectors; builds/self-tests are opt-in too.
# UF_ON=(1,) / (8,) / (1,8) BEFORE prepare builds just those shapes.
# For in-process A/B, set UF_BUILD=(1,8) before prepare, then switch UF_ON.
UF_ON=()
UF_BUILD=()
_UF_MODS={}
def _uf_src():
 s=_TINYS_CUDA
 for a,b in _UF_EDITS:
  if s.count(a)!=1:
   raise RuntimeError(("UF source anchor",a[:80],s.count(a)))
  s=s.replace(a,b)
 return s
def _uf_shapes(values):
 if not isinstance(values,(tuple,list,set,frozenset)) or any(type(t)is not int or t not in(1,8) for t in values):
  raise ValueError(("UF exact shapes",values))
 return tuple(sorted(set(values)))
def _uf_pack(w):
 # [tile8,row8,unit64,t4,q4,j4] -> [tile8,unit64,q4,lane32,j4].
 # Original BF16 bits are rearranged, never quantized; GD escapes are implicit.
 if w.shape!=(1024,4096) or w.dtype!=torch.bfloat16:
  raise ValueError("UF requires the real BF16 unpooler slice")
 return w.contiguous().view(128,8,64,4,4,4).permute(0,2,4,1,3,5).contiguous().view(-1)
def _uf_run(H,x,ids,wts,out=None,pdl=None):
 ids,wts=ids.to(torch.int32).contiguous(),wts.to(torch.float32).contiguous()
 x=x.contiguous()
 if x.shape[0]!=H["T"]:
  raise ValueError("UF run shape")
 if out is None:
  out=torch.empty((H["T"],HIDDEN),dtype=torch.bfloat16,device=x.device)
 pdl=int(HYBLO_PDL) if pdl is None else int(pdl)
 if H["pdl"]!=pdl:
  H["mod"].okc_set_pdl(pdl)
  H["pdl"]=pdl
 H["mod"].okc_run(H["w"],H["sc"],x,ids,wts,out,0,32,_HYB["nsm"])
 return out
def _uf_selftest(state,H):
 m=H["T"]; dev=state["dense"]["unpooler"].device
 ds=[_hyblo_inputs(m,seed,dev,dist) for seed,dist in ((81,"normal"),(82,"skewed"),(92,"conc"),(94,"none"))]
 x=ds[0]["hidden_states"].contiguous().clone()
 ids=ds[0]["topk_ids"].to(torch.int32).contiguous().clone()
 tw=ds[0]["topk_weights"].float().contiguous().clone()
 graph_out=_uf_run(H,x,ids,tw,pdl=0)
 torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):
  _uf_run(H,x,ids,tw,out=graph_out,pdl=0)
 worst=0.; nbit=0
 for d in ds+ds[::-1]:
  x.copy_(d["hidden_states"]); ids.copy_(d["topk_ids"]); tw.copy_(d["topk_weights"])
  ref_moe=_hyblo_run(_hyblo_key(m),x,ids,tw)
  ref=_gd_unpool(state,ref_moe,m)
  if ref is None:
   raise RuntimeError("UF needs shipped GD unpooler for validation")
  ref=ref.clone()
  eager=_uf_run(H,x,ids,tw,pdl=1)
  graph.replay(); graph.replay()
  torch.cuda.synchronize()
  for out in(eager,graph_out):
   total,row=_k2a_rel_err(out,ref)
   worst=max(worst,total)
   nbit+=int(torch.equal(out,ref))
   if not(total<2e-3 and row<1e-2 and bool(torch.isfinite(out.float()).all()) and bool((out[:,1024:]==0).all())):
    raise RuntimeError(("UF numeric self-test",m,total,row))
  if int(H["sc"][0][7]) or not all(bool((f[n:].view(vt)==sv).all()) for f,n,vt,sv in H["full"]):
   raise RuntimeError("UF sticky timeout/guard failure")
 print(f"[uf] T={m} self-test eager/graph/stale PASS worst {worst:.3e} bitwise {nbit}/16",flush=True)
 return {"worst":worst,"bitwise":nbit,"cases":16}
def _uf_prepare(state):
 from torch.utils.cpp_extension import load_inline
 wanted=tuple(sorted(set(_uf_shapes(UF_BUILD))|set(_uf_shapes(UF_ON))))
 state["s7_uf"]={}
 if not wanted:
  return
 if not HYBLO_ENABLE or HYBLO_K!="h" or _HYB["nsm"]!=132 or state.get("k2a33")is None:
  raise RuntimeError("UF requires the shipped H200 TINYS route")
 # Copy the exact base accessors; no VMS or loaded-module mutation.
 st=state["k2a33"]["st"]
 if st is None:
  raise RuntimeError("UF missing real K2A33 state")
 w=state["dense"]["unpooler"][:UNPOOL_SLICE]
 uw=_uf_pack(w)
 src=_uf_src()
 for m in wanted:
  k=_hyblo_key(m)
  if m not in HYBLO_T or not _HYBLO["ok"].get(k) or _gd_live(state,"unpooler",m)is None:
   raise RuntimeError(("UF shipped route did not pass prepare",m))
  expected=(1,1,16,4,2,1,6) if m==1 else(1,1,8,2,2,4,6)
  if GD_ROUTE["unpooler",m]!=expected:
   raise RuntimeError("UF arithmetic depends on the exact GD route")
  var=next(v for key,_,v in _HYBLO_KEYS if key==k)
  fl=list(_HYBLO_FLAGS)+list(var)+["-DOKC_TINYS=7",f"-DOKC_UF_T={m}","-std=c++20","-Xptxas=-v"]
  tag=hashlib.sha1((_HYB_MOE_CPP+src+" ".join(fl)).encode()).hexdigest()[:12]
  if tag not in _UF_MODS:
   _UF_MODS[tag]=load_inline(name=f"s7uf_{m}_{tag}",cpp_sources=_HYB_MOE_CPP,cuda_sources=src,
    functions=["okc_run","okc_set_pdl"],extra_cflags=["-std=c++20"],extra_cuda_cflags=fl,verbose=False)
  sc,full=_hyb_scratch(w.device)
  H={"T":m,"mod":_UF_MODS[tag],"w":_HYBLO["w"]+[uw],"sc":sc,"full":full,"pdl":None,"ok":False}
  H["test"]=_uf_selftest(state,H)
  H["ok"]=True
  state["s7_uf"][m]=H
def _uf_fused(state,x,ids,wts,cfg,tokens):
 if tokens not in _uf_shapes(UF_ON):
  return None
 if not(tokens==x.shape[0] and cfg["local_experts"]==K2A_E_LOC and cfg["top_k"]==K2A_TOP_K and cfg["n_experts"]==N_EXPERTS):
  raise ValueError("UF input/config mismatch")
 H=state.get("s7_uf",{}).get(tokens)
 if H is None or not H["ok"]:
  raise RuntimeError("UF requested arm was not prepared and validated")
 return _uf_run(H,x,ids,wts)

'''

LAYER_ANCHOR = ' _SU1_FLAG["k3trig"]=_su1_k3trig(state,tokens)\n'
LAYER_HOOK = ''' uf_output=_uf_fused(state,expert_x,local_ids,topk_weights,cfg,tokens)
 if uf_output is not None:
  next_attn_input,stream,_=_v9_bnd("ffn",stream,uf_output,None,state,tokens)
  return next_attn_input,stream,topk_ids
'''
PREP_ANCHOR = ' _tinys_prepare()\n'
BLOCK_ANCHOR = '_HYBLO_REG_HJ=[(0,0),(1,0),(0,2),(1,2)]\n'


def generate(base):
    # Entire original CUDA strings remain byte-identical. Only new modules use edits.
    assert "_UF_EDITS=" not in base
    assert "\r" not in base
    cu = literal(base, "_TINYS_CUDA")
    for a, b in EDITS:
        cu = sub(cu, a, b)
    out = sub(base, BLOCK_ANCHOR, "_UF_EDITS=" + repr(EDITS) + "\n" + GLUE + BLOCK_ANCHOR)
    out = sub(out, PREP_ANCHOR, PREP_ANCHOR + ' _uf_prepare(state)\n')
    out = sub(out, LAYER_ANCHOR, LAYER_HOOK + LAYER_ANCHOR)
    lines = out.splitlines(keepends=True)
    lines[1] = "# nonce uf-c1 (S7-UF default-off T=1/8 TINYS + BF16 unpooler; weight traffic grows; tools/s7_uf_patches.py)\n"
    out = "".join(lines)
    assert len(out.encode()) < 1_000_000, len(out.encode())
    compile(out, str(OUTPUT), "exec")
    return out


def identity_gate(base, generated):
    restored = sub(generated, "_UF_EDITS=" + repr(EDITS) + "\n" + GLUE + BLOCK_ANCHOR, BLOCK_ANCHOR)
    restored = sub(restored, PREP_ANCHOR + ' _uf_prepare(state)\n', PREP_ANCHOR)
    restored = sub(restored, LAYER_HOOK + LAYER_ANCHOR, LAYER_ANCHOR)
    a, b = restored.splitlines(keepends=True), base.splitlines(keepends=True)
    a[1] = b[1]
    assert "".join(a) == base, "unrelated bytes changed"
    assert literal(base, "_TINYS_CUDA") == literal(generated, "_TINYS_CUDA")
    return {"base_restored_except_named_insertions": True, "shipped_cuda_byte_identical": True,
            "default_selectors": [literal(generated, "UF_ON"), literal(generated, "UF_BUILD")]}


def geometry_gate():
    # Exhaustive ownership for all post-wait/reset/return paths.
    for m in (1, 8):
        output = set()
        suffix = set()
        done = set()
        other = set()
        ctr = set()
        for c in range(132):
            for t in range(256):
                i0 = c * 256 + t
                if i0 < 2:
                    ctr.add(i0)
                done.update(range(i0, 88, 132 * 256))
                other.update(range(i0, 3072, 132 * 256))
                suffix.update(i for i in range(i0, m * 8192, 132 * 256) if i % 8192 >= 1024)
                if c < 128 and t < 64:
                    l, v = t % 32, t // 32
                    token = 2 * (l % 4) + v
                    if token < m:
                        i = token * 8192 + c * 8 + l // 4
                        assert i not in output
                        output.add(i)
        assert done == set(range(88)) and other == set(range(3072)) and ctr == {0, 1}
        assert output.isdisjoint(suffix) and len(output | suffix) == m * 8192
        assert max(m * 4104 * 2 + (16 if m == 1 else 32) * 2 * 32 * 4, 0) <= 3 * 36864
        assert 3 * 36864 + 65536 <= 178176
    # GD's 64 K units, q parity and grouping are preserved for every output.
    for m, wk, ku, splits in ((1, 16, 4, 1), (8, 8, 2, 4)):
        gd = [[[(s * wk + k) * ku + u for u in range(ku)] for k in range(wk)] for s in range(splits)]
        uf = [[[(h * 8 + w) * ku + u for u in range(ku)] for w in range(8)] for h in range(2 if m == 1 else 4)]
        assert (gd[0] == uf[0] + uf[1]) if m == 1 else (gd == uf)
    return {"post_wait_coverage": "PASS", "smem_nonoverlap": "PASS", "GD_reduction_order": "PASS",
            "cleanup": ["ctr[0:2]: CTA0 consumer tid0..1", "doneA[0:88]: CTA0 consumer tid0..87",
                        "other ss[0:3072]: CTAs0..11 consumer tid0..255 before phases",
                        "next-generation ctr: CTA0 consumer tid0 before phases",
                        "UF suffix: all132 CTAs consumer tid0..255 before any UF return",
                        "UF combine/MMA: CTAs0..127 consumer tid0..255; output tid0..63",
                        "helper logical tid256..287: unchanged WS_JOIN, then return",
                        "single-use bar11: initialized each call, no empty/reuse arrive"]}


def dry_run(base, generated):
    """Real new prepare functions, actual base state expressions; fake CUDA only.

    Numerical GPU self-tests execute their Python control flow on symbolic tensors.
    Separate exhaustive NumPy pack/fragment tests verify real index/bit mappings.
    """
    import numpy as np
    calls, builds = [], []

    class Tensor:
        device = "cpu-dry"
        dtype = "bf16"
        def __init__(self, shape=(1, 4096), data=None):
            self.shape, self.data = tuple(shape), data
        def contiguous(self): return self
        def clone(self): return Tensor(self.shape, self.data)
        def to(self, *_): return self
        def float(self): return self
        def copy_(self, _): return self
        def all(self): return True
        def __bool__(self): return True
        def __int__(self): return 0
        def __eq__(self, _): return self
        def __getitem__(self, key):
            if self.data is not None:
                d = self.data[key]
                return Tensor(d.shape, d)
            if isinstance(key, slice):
                a, b, step = key.indices(self.shape[0])
                return Tensor((len(range(a, b, step)),) + self.shape[1:])
            return Tensor(self.shape[1:] or (1,))
        def view(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], str): return self
            if self.data is not None:
                d = self.data.reshape(*shape)
                return Tensor(d.shape, d)
            if shape == (-1,): shape = (int(np.prod(self.shape)),)
            return Tensor(shape)
        def permute(self, *axes):
            return Tensor(tuple(self.shape[a] for a in axes), None if self.data is None else self.data.transpose(*axes))

    class Graph:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def replay(self): calls.append("replay")

    class Module:
        def okc_set_pdl(self, pdl): calls.append(("pdl", pdl))
        def okc_run(self, *args):
            assert len(args[0]) == 15 and len(args[1]) == 3
            assert args[5].shape == (args[2].shape[0], 8192) and args[-3:] == (0, 32, 132)
            calls.append(("run", args[2].shape[0]))

    def loader(**kw):
        assert kw["cpp_sources"] == literal(base, "_HYB_MOE_CPP")
        assert kw["cuda_sources"] == env["_uf_src"]()
        builds.append(kw)
        return Module()

    real_import = builtins.__import__
    def mock_import(name, *a, **kw):
        return types.SimpleNamespace(load_inline=loader) if name == "torch.utils.cpp_extension" else real_import(name, *a, **kw)

    torch = types.SimpleNamespace(bfloat16="bf16", int32="i32", float32="f32",
                                  empty=lambda shape, **kw: Tensor(shape), equal=lambda a, b: True,
                                  isfinite=lambda a: a, cuda=types.SimpleNamespace(
                                      synchronize=lambda: None, CUDAGraph=Graph, graph=lambda g: g))
    env = {"__builtins__": dict(vars(builtins), __import__=mock_import, print=lambda *a, **kw: calls.append(("symbolic_log", a))), "hashlib": hashlib, "torch": torch,
           "HIDDEN": 8192, "D_EXPERT": 4096, "UNPOOL_SLICE": 1024, "N_EXPERTS": 256,
           "K2A_E_LOC": 32, "K2A_TOP_K": 8, "_HYB": {"nsm": 132},
           "_hyb_scratch": lambda d: ([Tensor((4096,)) for _ in range(3)], [(Tensor((10,)), 8, "i32", 123)]),
           "_gd_live": lambda state, proj, m: state["gd"]["route"].get((proj, m)),
           "_k2a_rel_err": lambda a, b: (0., 0.),
           "_gd_unpool": lambda state, x, m: Tensor((m, 8192)),
           "_hyblo_run": lambda k, x, ids, tw: Tensor(x.shape),
           "_hyblo_inputs": lambda m, seed, dev, dist: {"hidden_states": Tensor((m, 4096)),
                                                      "topk_ids": Tensor((m, 8)), "topk_weights": Tensor((m, 8))}}
    for name in ("_UF_EDITS", "UF_ON", "UF_BUILD", "_UF_MODS", "_TINYS_CUDA", "_HYB_MOE_CPP", "_HYBLO_KEYS",
                 "_HYBLO_FLAGS", "_HYBLO", "HYBLO_ENABLE", "HYBLO_K", "HYBLO_T", "HYBLO_PDL", "GD_ROUTE"):
        env[name] = literal(generated, name)
    funcs = [n for n in ast.parse(generated).body if isinstance(n, ast.FunctionDef) and
             (n.name.startswith("_uf_") or n.name == "_hyblo_key")]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "uf-real-python", "exec"), env)
    # Evaluate actual state dict/return expressions, without importing the layer.
    state_assign = next(n for n in ast.walk(ast.parse(base)) if isinstance(n, ast.Assign) and
                        any(isinstance(t, ast.Name) and t.id == "state" for t in n.targets) and
                        isinstance(n.value, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "dense" for k in n.value.keys))
    dense_expr = next(v for k, v in zip(state_assign.value.keys, state_assign.value.values) if isinstance(k, ast.Constant) and k.value == "dense")
    weights = {k: Tensor((8192, 4096)) for k in ("qkv_gate", "o", "pooler", "unpooler")}
    dense = eval(compile(ast.Expression(dense_expr), "base-dense-layout", "eval"), {"weights": weights})
    prep = node(base, "_v11_k2a33_prepare")
    ret = next(n.value for n in prep.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict))
    marker = object()
    k33 = eval(compile(ast.Expression(ret), "base-k2a33-layout", "eval"), {"st": marker, "ok": {1: True, 8: True}})
    state = {"dense": dense, "k2a33": k33, "gd": {"route": {("unpooler", m): object() for m in (1, 8)}}}
    env["_HYBLO"].update(ok={"tiny": True, "tiny2": True}, w=[object() for _ in range(14)])
    # Execute the actual inserted prepare hook, including default-off no-build.
    env["state"] = state
    hook = next(n for n in node(generated, "prepare").body if isinstance(n, ast.Expr) and
                isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_uf_prepare")
    exec(compile(ast.Module(body=[hook], type_ignores=[]), "uf-prepare-hook", "exec"), env)
    assert not builds and state["s7_uf"] == {}
    env["UF_BUILD"] = (1, 8)
    env["_uf_prepare"](state)
    assert len(builds) == 2 and set(state["s7_uf"]) == {1, 8}
    env["_uf_prepare"](state)
    assert len(builds) == 2, "cache not honored"
    cfg = {"local_experts": 32, "top_k": 8, "n_experts": 256}
    for selected in ((), (1,), (8,), (1, 8)):
        env["UF_ON"] = selected
        for m in (1, 2, 4, 5, 8, 16, 32, 128, 256):
            out = env["_uf_fused"](state, Tensor((m, 4096)), Tensor((m, 8)), Tensor((m, 8)), cfg, m)
            assert (out is not None) == (m in selected)
    for bad in ((2,), (True,), (1, "8"), 1):
        try: env["_uf_shapes"](bad)
        except ValueError: pass
        else: raise AssertionError(("bad selector accepted", bad))
    env["UF_ON"], env["UF_BUILD"] = (1,), ()
    before = len(builds)
    env["_uf_prepare"](state)
    assert set(state["s7_uf"]) == {1} and len(builds) == before
    env["UF_ON"] = (8,)
    try: env["_uf_fused"](state, Tensor((8, 4096)), Tensor(), Tensor(), cfg, 8)
    except RuntimeError: pass
    else: raise AssertionError("unprepared arm accepted")
    # Missing actual nested st must fail locally, not on a paid device.
    env["UF_ON"] = (1,)
    broken = dict(state, k2a33={"ok": {1: True}})
    try: env["_uf_prepare"](broken)
    except KeyError as ex: assert ex.args == ("st",)
    else: raise AssertionError("missing real state key was ignored")
    # Real pack code on all 4,194,304 distinct coordinate IDs (not mocked ops).
    ids = np.arange(1024 * 4096, dtype=np.uint32).reshape(1024, 4096)
    packed = env["_uf_pack"](Tensor(ids.shape, ids)).data.reshape(128, 64, 4, 32, 4)
    for q in range(4):
        for ln in range(32):
            for j in range(4):
                expected = (np.arange(128)[:, None] * 8 + ln // 4) * 4096 + np.arange(64)[None, :] * 64 + 16 * (ln % 4) + 4 * q + j
                assert np.array_equal(packed[:, :, q, ln, j], expected)
    return {"prepare_state_layout": "PASS: actual base expressions", "pack_coordinates_checked": ids.size,
            "prepare_paths": "default/build/cache/select/invalid/missing-key/selftest/graph-control-flow PASS",
            "build_count": len(builds), "calls": len(calls), "numerical_GPU_test": "NOT RUN: symbolic CPU only"}


def cpu_gate(base, generated):
    for s in (base, generated): compile(s, "layer", "exec")
    cuda = literal(base, "_TINYS_CUDA")
    for a, b in EDITS: cuda = sub(cuda, a, b)
    assert cuda.count("__trap()") == literal(base, "_TINYS_CUDA").count("__trap()")
    assert "S.bars + 11" not in literal(base, "_TINYS_CUDA")
    assert 'state["k2a33"]["st"]' in GLUE
    # Diff lists every changed line, including the serialized C++ edit expressions.
    sm = difflib.SequenceMatcher(a=base.splitlines(), b=generated.splitlines(), autojunk=False)
    changes = [{"base": [i+1,j], "candidate": [k+1,l], "kind": op} for op,i,j,k,l in sm.get_opcodes() if op != "equal"]
    return {"identity": identity_gate(base, generated), "geometry": geometry_gate(), "dry_run": dry_run(base, generated),
            "bytes": len(generated.encode()), "sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "per_bit_changed_lines": {"1_T1_and_2_T8_shared": changes},
            "accounting_EST": {
                "expert_weight_stage_before_after": [36864, 36864],
                "expert_consumer_instructions_before_after": "unchanged source: 9 LDS128,32 dq4,128 HMUL2,32 MMA,12 adds,1 STS128,1 CTA barrier per unit",
                "unpooler_weights_T1_before_after": [6291456+131072, 8388608],
                "unpooler_weights_T8_before_after": [6291456+262144, 8388608],
                "unpooler_stage_half_tile_before_after": [768, 1024],
                "UF_consumer_per_K64_unit": "per thread 4 LDS64 weight + 2 LDS128 x + 4 MMA; GD full16-row:3 LDG128+2 LDG128 x+decode+escape+4 MMA",
                "UF_consumer_total_per_thread": "both T:32 LDS64 weight+16 LDS128 x+32 MMA; T1 4 partial stores, T8 8 partial stores",
                "combine_per_thread": "4*T float4 outputs; per live route:1 LDG128 y,1 LDG32 ss,1 LDG32 w,1 rsqrt,2 MUL,4 FMA; 2 BF16x2 CVT+1 STS64 per float4",
                "combine_L2_y_bytes": "before 16384*sum(local_routes); after 128 times that (repeated L2-resident reads)",
                "UF_shared_partial_bytes_T1_T8": [4096, 8192],
                "UF_new_zero_tail_bytes_T1_T8": [14336, 114688],
                "weight_growth_floor_us_T1_T8": [(8388608-6422528)/4.64e6, (8388608-6553600)/4.64e6],
                "speedup": "UNMEASURED; no claimed saving; supplied 9.4us label is qkv, true unpooler d1s=4.1us"},
            "nvcc_spills_SASS_GPU": "NOT RUN"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    raw = BASE.read_bytes()
    assert b"\r" not in raw, "base unexpectedly not LF"
    base = raw.decode()
    generated = generate(base)
    result = cpu_gate(base, generated)
    if args.check:
        assert OUTPUT.read_bytes() == generated.encode(), "stale candidate"
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(generated.encode())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

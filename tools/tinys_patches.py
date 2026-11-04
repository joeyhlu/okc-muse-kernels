"""TINYS: isolated, default-off HYBLO tiny/tiny2 latency experiments.

Only k3_tinys_ab.py is generated. CPU mode imports no submission / torch / CUDA.
  python -B tools/tinys_patches.py
  python -B tools/tinys_patches.py --check

Kill reasons (unmeasured; this is the explicitly requested build):
1: helper backlog can dominate publication; one poller can delay discovery.
2: fences may already drain behind WS_JOIN; last arriver may not combine.
4: route may wait on warp 7 anyway; compiler/placement can hide the load move.
8: waiting for job 2 early can delay job 1; extra live addresses may spill.

No weight-stream growth: 32768 code + 4096 scale B/unit before and after.
The MMA/dequant/reduction loop is verbatim. Source-level instruction accounting:
1: helper all-lane fence + warp sync + lane-0 RED -> warp sync + release RED;
   B switch: 8 lane-0 pollers/CTA -> 1, then bar.sync 3,256 (all consumers).
2: helper fence removed; tid-0 fence+ATOM+poll -> acq_rel ATOM+non-last poll.
4: same single 4-B gen load, same stores, owner tid 0 -> 64.
8: each valid consumer thread replaces 16 global uint4 loads by 16 shared
   uint4 loads at job 2 (256 B/thread; 8192*cnt B/CTA). tid 0 adds cnt bulk
   copies, one expect_tx, two proxy fences; all consumers add a bounded wait.
   SMEM writes add 8192*cnt B; SMEM reads add 8192*cnt B; global x bytes do
   not grow. A additionally reads cnt 2-B ent entries in the issuer. Counters
   are 4-B L2 traffic; retry counts and lowered address/control ops need SASS.

Protocol proof: there are NO new CTA/thread exits. The helper (logical tids
256..287) reaches both WS_JOINs then returns exactly as ship. Consumers 0..255
finish both phases and the grid barrier, including empty CTAs. CTA 0 tids 0,1
reset ctr[0:2]; CTA 0 tids 0..87 reset doneA[0:88]. The existing all-CTA
strided loop clears the other ss bank (3072 floats), and CTA0/tid0 clears the
next generation counter before work. Combine covers tiles 0..4*M-1 using
all 256 consumer tids (4 columns/thread); the strided fallback is unchanged.
The sticky error at ctr[7] is never reset by the kernel. A failed wait reports
failure and continues participation, like the ship HYA sticky-error style;
its outputs must be rejected. There is no trap on a timeout.

Bit 2 memory argument: helper global stores precede WS_JOIN (bar.sync 2,288),
which orders them before consumer tid 0. WSBLK stores precede CTA_SYNC. The
acq_rel GPU-scope RMW publishes this transitive history and acquires preceding
RMWs on the same counter. The last arriver already owns the complete history;
other CTAs acquire the final count, then CTA_SYNC publishes it to consumers.
Thus the helper's final fence and the counter thread's SC fence are redundant.
Bit 1 analogously publishes all helper lanes after __syncwarp via release RED;
the consuming acquire is followed by a counted consumer-only CTA barrier.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import difflib
import hashlib
import json
from pathlib import Path
import time
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_pure_c1.py"
OUTPUT = ROOT / "kernels/lab/k3_tinys_ab.py"
BUILD_BITS = (1, 2, 3, 4, 7, 8, 15)


def sub(s, old, new):
    assert s.count(old) == 1, ("anchor count", s.count(old), old[:100])
    return s.replace(old, new, 1)


def literal(s, name):
    nodes = [n.value for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, (name, len(nodes))
    return ast.literal_eval(nodes[0])


DEFS = r'''
// TINYS_COMMON: scratch[0][7] is unused by ship and never reset in-kernel.
#ifndef OKC_TINYS
#define OKC_TINYS 0
#endif
static_assert((OKC_TINYS & ~15) == 0, "TINYS bits");
static_assert(OKC_NW == 8 && OKC_NS == 3 && OKC_SPLIT == 0 && OKC_TINY,
              "TINYS is only the shipped tiny/tiny2 specialization");
constexpr unsigned TINYS_SPIN_LIMIT = 1u << 24; // ship HYA M1 iteration bound
__device__ __forceinline__ void tinys_fail(const Params& p, int why) {
atomicOr(p.ctr + 7, why); // sticky; deliberately not cleared on replay
}
__device__ __forceinline__ int tinys_acquire(const int* a) {
int v;
asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(a) : "memory");
return v;
}
template <bool WithSleep = false>
__device__ __forceinline__ void tinys_wait_flag(const Params& p, const int* a, int n, int why) {
for (unsigned spin = 0; spin < TINYS_SPIN_LIMIT; ++spin) {
if (tinys_acquire(a) >= n) return;
if constexpr (WithSleep) __nanosleep(64);
}
tinys_fail(p, why);
}
#define TINYS_CONSUMERS() asm volatile("bar.sync 3, %0;\n" :: "n"(NT) : "memory")
'''

STAGING = r'''
// TINYS_8: one phase-local use of bars[4+PB], parity 0; inits stay after PDL.
#if OKC_TINYS & 8
constexpr int TINYS_ROW_BYTES = HD * 2, TINYS_ROW_PITCH = HD * 2 + 16;
static_assert(8 * TINYS_ROW_PITCH <= 2 * FT_T * XP_FT * 2, "unused tail fits");
static_assert(NSF == OKC_NS, "tail is exactly the unused fat-x allocation");
static_assert(HD == ID && NCH == 8, "copy/load map");
__device__ __forceinline__ void tinys_wait_stage(const Params& p, uint64_t* bar) {
for (unsigned spin = 0; spin < TINYS_SPIN_LIMIT; ++spin) {
unsigned ready;
asm volatile("{ .reg .pred q; mbarrier.try_wait.parity.shared::cta.b64 q, [%1], 0; "
             "selp.u32 %0, 1, 0, q; }\n" : "=r"(ready) : "r"(smem_u32(bar)) : "memory");
if (ready) return;
}
tinys_fail(p, 4);
}
template <bool PB>
__device__ __forceinline__ void tinys_stage(const Params& p, const Smem& S, int j,
                                          unsigned char* dst, uint64_t* bar) {
const int cnt = S.jC[j], st = S.jS[j];
asm volatile("fence.proxy.async.global;\nfence.proxy.async.shared::cta;\n" ::: "memory");
asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
             :: "r"(smem_u32(bar)), "r"(cnt * TINYS_ROW_BYTES) : "memory");
for (int r = 0; r < cnt; ++r) {
const __nv_bfloat16* src = PB ? p.inter + (size_t)(st + r) * ID
                            : p.x + (size_t)(S.ent[st + r] >> 3) * HD;
asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
             :: "r"(smem_u32(dst + r * TINYS_ROW_PITCH)), "l"(src),
                "r"(TINYS_ROW_BYTES), "r"(smem_u32(bar)) : "memory");
}
}
__device__ __forceinline__ void tinys_load_stage(const Smem& S, int j, int lane, int warp,
                                               const unsigned char* src, uint4 (&x)[2 * NCH]) {
const int g = lane >> 2, t = lane & 3;
if (g < S.jC[j]) {
const unsigned char* row = src + g * TINYS_ROW_PITCH + warp * 1024 + 32 * t;
#pragma unroll
for (int c = 0; c < NCH; ++c) {
x[2 * c] = *reinterpret_cast<const uint4*>(row + 128 * c);
x[2 * c + 1] = *reinterpret_cast<const uint4*>(row + 128 * c + 16);
}
} else {
#pragma unroll
for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
}
}
#endif
'''

STAGE_START = r'''
#if OKC_TINYS & 8
unsigned char* stg = ring + OKC_NS * UNIT_BYTES;
const int jA = njF + v0 / rbs;
const int jB = nU > 0 ? njF + (v1 - 1) / rbs : jA;
const bool strad = nU > 0 && jB != jA;
if (PB && nU > 0) {
// Bit 8 alone keeps eight sleeping first-job pollers; bit 1 elects tid 0, no sleep.
// The stage issuer must acquire the SECOND job before its async global read.
if ((OKC_TINYS & 1) ? tid == 0 : lane == 0)
tinys_wait_flag<!(OKC_TINYS & 1)>(p, p.doneA + jA, RB_A, 1);
if (strad && tid == 0) tinys_wait_flag(p, p.doneA + jB, RB_A, 1);
TINYS_CONSUMERS();
}
if (strad && tid == 0) tinys_stage<PB>(p, S, jB, stg, S.bars + 4 + (int)PB);
#endif
'''

SWITCH = r'''// TINYS_1_8: every consumer reaches the same switch/barrier; helper excluded.
#if OKC_TINYS & 8
if (strad && j == jB) {
tinys_wait_stage(p, S.bars + 4 + (int)PB);
tinys_load_stage(S, j, lane, warp, stg, x0);
} else {
load_x_regs<PB>(p, S, j, lane, warp, x0);
}
#else
if (PB) {
#if OKC_TINYS & 1
if (tid == 0) tinys_wait_flag(p, p.doneA + j, RB_A, 1);
TINYS_CONSUMERS();
#else
if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
__syncwarp();
#endif
}
load_x_regs<PB>(p, S, j, lane, warp, x0);
#endif'''


def patch_cuda(cu):
    cu = sub(cu, "__device__ __forceinline__ int ld_acquire(const int* p) {", DEFS + "\n__device__ __forceinline__ int ld_acquire(const int* p) {")
    cu = sub(cu, 'asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p));',
             '#if OKC_TINYS & 2\nasm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p) : "memory");\n#else\n'
             'asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p));\n#endif')
    cu = sub(cu, "template <bool PB>\n__device__ void skinny_ring_ws", STAGING + "\ntemplate <bool PB>\n__device__ void skinny_ring_ws")
    cu = sub(cu, "const int seq0 = seq;\nuint4 x0[2 * NCH];", "const int seq0 = seq;\n" + STAGE_START + "\nuint4 x0[2 * NCH];")
    cu = sub(cu, '''if (PB) {
if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
__syncwarp();
}
load_x_regs<PB>(p, S, j, lane, warp, x0);''', SWITCH)
    cu = sub(cu, "if (q == nA - 1) {\n__threadfence();\n__syncwarp();", "// TINYS_1: warp release publication (all helper stores precede sync).\nif (q == nA - 1) {\n#if !(OKC_TINYS & 1)\n__threadfence();\n#endif\n__syncwarp();")
    cu = sub(cu, "atomicAdd(p.doneA + jj, b - a);", '''#if OKC_TINYS & 1
asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" :: "l"(p.doneA + jj), "r"(b - a) : "memory");
#else
atomicAdd(p.doneA + jj, b - a);
#endif''')
    cu = sub(cu, "__threadfence();\nWS_JOIN();\n}", "// TINYS_2: WS_JOIN -> consumer CTA_SYNC -> acq_rel RMW publishes helper stores.\n#if !(OKC_TINYS & 2)\n__threadfence();\n#endif\nWS_JOIN();\n}")
    cu = sub(cu, "if (tid == 0) {\nS.seq = 0;", "// TINYS_4: warp 2 does not load topk at M <= 8. No mbarrier init moves.\nif (tid == ((OKC_TINYS & 4) ? 64 : 0)) {\nS.seq = 0;")
    cu = sub(cu, '''__threadfence();
const int old = atomicAdd(bc, 1);
if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);''', '''// TINYS_2: acq_rel chain collects all CTA publications; last already acquired.
#if OKC_TINYS & 2
int old;
asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
else tinys_wait_flag(p, bc, (int)gridDim.x, 2);
#else
__threadfence();
const int old = atomicAdd(bc, 1);
if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
#endif''')
    cu = sub(cu, "p.ctr = ib; p.doneA = ib + 8;", '''// TINYS_1: start doneA on a separate 128-B line; reset loop uses p.doneA.
static_assert(32 + okc::JMAX <= 8 + 1024, "doneA must end before ss1");
p.ctr = ib; p.doneA = ib + ((OKC_TINYS & 1) ? 32 : 8);
TORCH_CHECK(p.M >= 1 && p.M <= 8, "TINYS only supports M <= 8");
#if OKC_TINYS & 8
// <=32 skinny jobs / >=32 CTAs gives <=one job boundary in each CTA interval.
TORCH_CHECK(grid >= 32, "TINYS staging requires at least 32 CTAs");
#endif''')
    return cu


GLUE = '''
# TINYS Python bank: built after every shipped prepare step; ship sources untouched.
TINYS = {}
TINYS_BUILD_BITS = (1, 2, 3, 4, 7, 8, 15)
_TINYS_BANK = {}
_TINYS_SHIP_PREPARE = False
def _tinys_ship_prepare(st,w,dev):
 global _TINYS_SHIP_PREPARE
 _TINYS_SHIP_PREPARE=True
 try:
  return _hyblo_prepare(st,w,dev)
 finally:
  _TINYS_SHIP_PREPARE=False
def _tinys_prepare():
 from torch.utils.cpp_extension import load_inline
 if not HYBLO_ENABLE:
  return
 if not all(_HYBLO["ok"].get(k) for k,_,_ in _HYBLO_KEYS):
  raise RuntimeError("TINYS: shipped tiny/tiny2 did not pass prepare")
 for k,_,var in _HYBLO_KEYS:
  for bits in TINYS_BUILD_BITS:
   if not isinstance(bits,int) or not 1<=bits<=15:
    raise ValueError(("TINYS_BUILD_BITS",bits))
   key=(k,bits)
   H=_TINYS_BANK.setdefault(key,{"mod":None,"pdl":None})
   fl=list(_HYBLO_FLAGS)+list(var)+[f"-DOKC_TINYS={bits}","-std=c++20","-Xptxas=-v"]
   if H["mod"] is None:
    tag=hashlib.sha1((_HYB_MOE_CPP+_TINYS_CUDA+" ".join(fl)).encode()).hexdigest()[:12]
    H["mod"]=load_inline(name=f"tinys_{k}_{bits}_{tag}",cpp_sources=_HYB_MOE_CPP,cuda_sources=_TINYS_CUDA,
                         functions=["okc_run","okc_set_pdl"],extra_cflags=["-std=c++20"],extra_cuda_cflags=fl,verbose=False)
   # Exact ship accessors; all arms share weights and the generation-owned scratch.
   H["w"],H["sc"],H["full"]=_HYBLO["w"],_HYBLO["sc"],_HYBLO["full"]
def _tinys_select(k,m):
 if _TINYS_SHIP_PREPARE:
  return None
 if k not in ("tiny","tiny2") or not 1<=m<=8:
  return None
 bits=TINYS.get(1 if k=="tiny" else 8,0)
 if not isinstance(bits,int) or not 0<=bits<=15:
  raise ValueError(("TINYS",bits))
 if not bits:
  return None
 if (k,bits) not in _TINYS_BANK or _TINYS_BANK[k,bits]["mod"] is None:
  raise RuntimeError(f"TINYS: requested {k}/{bits} was not built in prepare")
 return _TINYS_BANK[k,bits]
def _tinys_run(H,x,ids,wts):
 mod=H["mod"]
 out=torch.empty((x.shape[0],4096),dtype=torch.bfloat16,device=x.device)
 pdl=1 if HYBLO_PDL else 0
 if H["pdl"]!=pdl:
  mod.okc_set_pdl(pdl)
  H["pdl"]=pdl
 mod.okc_run(H["w"],H["sc"],x,ids,wts,out,0,32,_HYB["nsm"])
 return out
'''


def generate(base):
    # Insertion, never replacement of _HYBLO_CUDA / _HYBLO_FLAGS / ship builds.
    src = sub(base, '_HYBLO_REG_HJ=', '_TINYS_CUDA=r"""' + patch_cuda(literal(base, "_HYBLO_CUDA")) + '"""\n' + GLUE + '\n_HYBLO_REG_HJ=')
    src = sub(src, ' mod=_HYBLO["mods"][k]\n', ' H=_tinys_select(k,x.shape[0])\n if H is not None:\n  return _tinys_run(H,x,ids,wts)\n mod=_HYBLO["mods"][k]\n')
    src = sub(src, '  _hyblo_prepare(state["k2a33"]["st"],', '  _tinys_ship_prepare(state["k2a33"]["st"],')
    # After all existing prepare work, so extra compiles cannot trip shipped guards.
    src = sub(src, ' _rq_prepare(state)\n builds=', ' _rq_prepare(state)\n _tinys_prepare()\n builds=')
    compile(src, str(OUTPUT), "exec")
    return src


def default_identity(base, generated):
    tree0, tree1 = ast.parse(base), ast.parse(generated)
    for name in ("_HYBLO_CUDA", "_HYB_MOE_CPP", "_HYBLO_FLAGS", "_HYBLO_KEYS"):
        assert literal(base, name) == literal(generated, name), name
    strip = []
    for n in tree1.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_tinys_"):
            continue
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                                             ("TINYS", "TINYS_BUILD_BITS", "_TINYS_BANK", "_TINYS_CUDA", "_TINYS_SHIP_PREPARE") for t in n.targets):
            continue
        if isinstance(n, ast.FunctionDef) and n.name == "prepare":
            for call in ast.walk(n):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=="_tinys_ship_prepare":
                    call.func.id="_hyblo_prepare"
            n.body = [s for s in n.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                      and isinstance(s.value.func, ast.Name) and s.value.func.id == "_tinys_prepare")]
        if isinstance(n, ast.FunctionDef) and n.name == "_hyblo_run":
            i = next(i for i,s in enumerate(n.body) if isinstance(s, ast.Assign) and
                     isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) and s.value.func.id == "_tinys_select")
            del n.body[i:i+2]
        strip.append(n)
    assert ast.dump(tree0, include_attributes=False) == ast.dump(ast.Module(body=strip, type_ignores=[]), include_attributes=False)
    assert literal(generated, "TINYS") == {}
    return {"status": "PASS", "ship_cuda_sha256": hashlib.sha256(literal(base,"_HYBLO_CUDA").encode()).hexdigest(),
            "all_other_base_AST_unchanged": True}


def geometry_gate():
    # Model every grid interval for all possible tiny/tiny2 local-job counts.
    intervals = 0
    for G in (32, 64, 96, 132):
        for nj in range(33):
            for rbs in (512, 256):
                v = [nj*rbs*c//G for c in range(G+1)]
                assert v[0] == 0 and v[-1] == nj*rbs
                for a,b in zip(v,v[1:]):
                    if a < b:
                        assert (b-1)//rbs - a//rbs in (0,1)
                    intervals += 1
        assert {i for c in range(G) for t in range(256) for i in range(c*256+t,88,G*256)} == set(range(88))
        assert {i for c in range(G) for t in range(256) for i in range(c*256+t,3072,G*256)} == set(range(3072))
        for M in (1,5,8):
            cover = [(it//4,(it%4)*1024+t*4+k) for c in range(G) for it in range(c,4*M,G)
                     for t in range(256) for k in range(4)]
            assert len(cover) == len(set(cover)) == M*4096
    for cnt in range(1,9):
        offsets = []
        for tid in range(256):
            lane, warp = tid%32, tid//32
            g,t = lane//4,lane%4
            if g >= cnt:
                continue
            for c in range(8):
                for h in range(2):
                    old = (warp*(64*8)+16*t)*2+c*128+h*16
                    new = warp*1024+32*t+128*c+16*h
                    assert old == new and new % 16 == 0 and new+16 <= 8192
                    offsets.extend(g*8208+new+b for b in range(16))
        assert len(offsets) == len(set(offsets)) == cnt*8192
    assert 8*8208 <= 67584 and 32+88 <= 1032
    assert 7 not in set(range(2)) | set(range(8,8+88)) | set(range(32,32+88)) | {3,4,5}
    return {"status":"PASS", "intervals":intervals, "row_counts":8, "reset_and_combine_coverage":True,
            "weight_bytes_per_unit_EST":[36864,36864], "smem_tail_bytes_EST":67584,
            "max_stage_bytes_EST":65664, "global_x_bytes_per_straddler_EST":[65536,65536]}


def dry_run(base, generated):
    """Execute the actual hook, bank construction, flags, cached build, selections.

    Only legacy CUDA/tensor operations are mocks. The state return dictionary is
    taken from _v11_k2a33_prepare, not reconstructed from a guessed accessor.
    """
    class Tensor:
        device = "cpu-dry-run"
        def __init__(self, shape=(1,4096)):
            self.shape = shape
        def to(self, *_): return self
        def contiguous(self): return self
    calls, builds = [], []
    class Module:
        def okc_set_pdl(self, pdl): calls.append(("pdl",pdl))
        def okc_run(self,*args): calls.append(("run",args))
    def loader(**kwargs):
        assert kwargs["cpp_sources"] == literal(base,"_HYB_MOE_CPP")
        assert kwargs["cuda_sources"] == literal(generated,"_TINYS_CUDA")
        assert kwargs["extra_cflags"] == ["-std=c++20"]
        builds.append(kwargs)
        return Module()
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "torch.utils.cpp_extension":
            return types.SimpleNamespace(load_inline=loader)
        return real_import(name,*args,**kwargs)
    env = {"__builtins__": dict(vars(builtins), __import__=mock_import), "hashlib":hashlib,
           "torch":types.SimpleNamespace(int32="i32",float32="f32",bfloat16="bf16",
                                         empty=lambda shape,**kw:Tensor(shape)),
           "HYBLO_ENABLE":True,"HYBLO_PDL":True,"_HYB":{"nsm":132}, "time":time}
    names = ("_HYBLO", "_HYBLO_KEYS", "_HYBLO_FLAGS", "_HYB_MOE_CPP", "_TINYS_CUDA", "TINYS", "TINYS_BUILD_BITS", "_TINYS_BANK", "_TINYS_SHIP_PREPARE")
    for name in names: env[name] = literal(generated,name)
    funcs = [n for n in ast.parse(generated).body if isinstance(n,ast.FunctionDef) and
             (n.name.startswith("_tinys_") or n.name in ("_hyblo_key","_hyblo_run"))]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),"tinys-dry-functions","exec"),env)
    H = env["_HYBLO"]
    H.update(mods={k:Module() for k,_,_ in env["_HYBLO_KEYS"]},ok={k:True for k,_,_ in env["_HYBLO_KEYS"]},
             w=object(), sc=[object() for _ in range(3)], full=object())
    prep0 = next(n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef) and n.name=="_v11_k2a33_prepare")
    ret = next(n.value for n in prep0.body if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict))
    marker = object()
    layout = eval(compile(ast.Expression(ret),"real-state-layout","eval"),{"st":marker,"ok":{}})
    prep = next(n for n in ast.parse(generated).body if isinstance(n,ast.FunctionDef) and n.name=="prepare")
    hook = next(n for n in prep.body if isinstance(n,ast.If) and 'state.get("k2a33")' in ast.get_source_segment(generated,n).splitlines()[0])
    def ship_prepare(st,w,dev):
        assert st is marker and w["post_norm_w"] is env["weights"]["post_expert_norm"] and dev==Tensor.device
        assert env["_TINYS_SHIP_PREPARE"]
        for m in (1,5,8):
            # A pre-set call-time selector must not hijack the ship self-tests.
            assert env["_tinys_select"](env["_hyblo_key"](m),m) is None
            env["_hyblo_run"](env["_hyblo_key"](m),Tensor((m,4096)),Tensor(),Tensor())
    env.update(state={"k2a":None,"k2a33":layout}, weights={k:Tensor() for k in
               ("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs","post_expert_norm")}, _hyblo_prepare=ship_prepare)
    env["TINYS"]={1:15,8:15}
    exec(compile(ast.Module(body=[hook],type_ignores=[]),"real-prepare-hook","exec"),env)
    assert not env["_TINYS_SHIP_PREPARE"] and env["TINYS"]=={1:15,8:15}
    def broken_prepare(*args): raise RuntimeError("mock ship failure")
    env["_hyblo_prepare"]=broken_prepare
    try: env["_tinys_ship_prepare"](None,None,None)
    except RuntimeError: pass
    else: raise AssertionError("exception must propagate")
    assert not env["_TINYS_SHIP_PREPARE"]
    bank_hook = next(n for n in prep.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                     and isinstance(n.value.func,ast.Name) and n.value.func.id=="_tinys_prepare")
    exec(compile(ast.Module(body=[bank_hook],type_ignores=[]),"new-prepare-hook","exec"),env)
    assert len(builds)==2*len(BUILD_BITS)
    env["_tinys_prepare"]()  # cached builds, scratch rebound to current ship state
    assert len(builds)==2*len(BUILD_BITS)
    for b in builds:
        k = "tiny2" if b["name"].startswith("tinys_tiny2_") else "tiny"
        flags = b["extra_cuda_cflags"]
        var = next(v for key,_,v in env["_HYBLO_KEYS"] if key==k)
        assert flags[:len(env["_HYBLO_FLAGS"])+len(var)] == list(env["_HYBLO_FLAGS"])+list(var)
    routes = 0
    for bits in (0,)+BUILD_BITS:
        env["TINYS"] = {} if not bits else {1:bits,8:bits}
        for m in (1,2,4,5,8):
            k = env["_hyblo_key"](m)
            assert k == ("tiny" if m<=4 else "tiny2")
            for pdl in (False,True):
                env["HYBLO_PDL"] = pdl
                out = env["_hyblo_run"](k,Tensor((m,4096)),Tensor(),Tensor())
                assert out.shape == (m,4096)
                args=calls[-1][1]
                assert args[0] is H["w"] and args[1] is H["sc"] and args[-1]==132
                routes += 1
    env["TINYS"]={1:1}
    assert env["_tinys_select"]("tiny2",5) is None
    assert env["_tinys_select"]("w",1) is None and env["_tinys_select"]("tiny2",16) is None
    for bad in (-1,16,"1",5):  # bit 5 is legal but deliberately not built by default
        env["TINYS"]={1:bad}
        try: env["_tinys_select"]("tiny",1)
        except (ValueError,RuntimeError): pass
        else: raise AssertionError(("selector must fail closed",bad))
    env["TINYS_BUILD_BITS"]=(16,)
    try: env["_tinys_prepare"]()
    except ValueError: pass
    else: raise AssertionError("invalid build bits accepted")
    env["HYBLO_ENABLE"]=False
    env["_tinys_prepare"]()
    env["HYBLO_ENABLE"]=True
    H["ok"]={}
    try: env["_tinys_prepare"]()
    except RuntimeError: pass
    else: raise AssertionError("missing ship prepare accepted")
    return {"status":"PASS","builds":len(builds),"run_paths":routes,"real_state_layout":ast.unparse(ret),
            "cached_build_and_missing_arm_gates":True,"limitation":"tensor/compiler calls mocked; no CUDA executed"}


def cpu_gate(base, generated):
    cu = literal(generated,"_TINYS_CUDA")
    original = literal(base,"_HYBLO_CUDA")
    def section(s,a,b): return s[s.index(a):s.index(b,s.index(a))]
    assert section(original,"__device__ void route(","__device__ __forceinline__ float probe_q") == section(cu,"__device__ void route(","__device__ __forceinline__ float probe_q")
    assert section(original,"const unsigned char* slot = ring + sl * UNIT_BYTES;","if (PB && warp < OKC_WSBLK && warp < nU)") == section(cu,"const unsigned char* slot = ring + sl * UNIT_BYTES;","if (PB && warp < OKC_WSBLK && warp < nU)")
    assert section(original,"constexpr int FSEG = HD / (NT * 4);","static int g_okc_pdl") == section(cu,"constexpr int FSEG = HD / (NT * 4);","static int g_okc_pdl")
    assert cu.count("__trap()") == original.count("__trap()")
    assert 'bar.sync 3' not in original, "consumer barrier id must be free"
    assert section(original,'asm volatile("griddepcontrol.wait;',"if (tid == 0) {\nS.seq = 0;") == section(cu,'asm volatile("griddepcontrol.wait;',"// TINYS_4:"), "PDL/init region changed"
    lines = {}
    for i,line in enumerate(generated.splitlines(),1):
        if "TINYS_" in line and ("// TINYS_" in line or "# TINYS" in line): lines[str(i)] = line.strip()
    def span(text):
        assert generated.count(text)==1,text[:80]
        a=generated.index(text)
        start=generated.count("\n",0,a)+1
        return [start,start+text.rstrip("\n").count("\n")]
    common=span(DEFS)
    sites={
        "1":[common,span(SWITCH),span("// TINYS_1: warp release publication (all helper stores precede sync).\nif (q == nA - 1) {\n#if !(OKC_TINYS & 1)\n__threadfence();\n#endif"),
             span('#if OKC_TINYS & 1\nasm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" :: "l"(p.doneA + jj), "r"(b - a) : "memory");\n#else\natomicAdd(p.doneA + jj, b - a);\n#endif'),
             span('// TINYS_1: start doneA on a separate 128-B line; reset loop uses p.doneA.\nstatic_assert(32 + okc::JMAX <= 8 + 1024, "doneA must end before ss1");\np.ctr = ib; p.doneA = ib + ((OKC_TINYS & 1) ? 32 : 8);')],
        "2":[common,span('#if OKC_TINYS & 2\nasm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p) : "memory");\n#else\nasm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p));\n#endif'),
             span('// TINYS_2: WS_JOIN -> consumer CTA_SYNC -> acq_rel RMW publishes helper stores.\n#if !(OKC_TINYS & 2)\n__threadfence();\n#endif\nWS_JOIN();'),
             span(section(cu,'// TINYS_2: acq_rel chain','\n}\nCTA_SYNC();\nDBG(4);'))],
        "4":[span('// TINYS_4: warp 2 does not load topk at M <= 8. No mbarrier init moves.\nif (tid == ((OKC_TINYS & 4) ? 64 : 0)) {')],
        "8":[common,span(STAGING),span(STAGE_START),span(SWITCH),
             span('TORCH_CHECK(p.M >= 1 && p.M <= 8, "TINYS only supports M <= 8");\n#if OKC_TINYS & 8\n// <=32 skinny jobs / >=32 CTAs gives <=one job boundary in each CTA interval.\nTORCH_CHECK(grid >= 32, "TINYS staging requires at least 32 CTAs");\n#endif')],
        "shared_python":[span(GLUE),span(' H=_tinys_select(k,x.shape[0])\n if H is not None:\n  return _tinys_run(H,x,ids,wts)'),
                         span('  _tinys_ship_prepare(state["k2a33"]["st"],{k:weights[k]for k in("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")}|{"post_norm_w":weights["post_expert_norm"]},weights["w13"].device)'),
                         span(' _tinys_prepare()\n builds=')],
    }
    stage={"weight_LDS128":9,"dq4_calls":32,"BF16x2_multiply":128,"MMA_sync":32,
           "FP32_sum_add":12,"red_STS128":1,"consumer_CTA_barrier":1}
    accounting={"label":"EST: source operations per consumer thread/unit; NOT measured SASS or timings",
                "hot_stage_before":stage,"hot_stage_after":stage,
                "weight_global_bytes_before_after":[36864,36864],
                "second_job_valid_thread_loads_before":{"global_uint4":16,"shared_uint4":0},
                "second_job_valid_thread_loads_after":{"global_uint4":0,"shared_uint4":16},
                "stage_issuer_extra":{"bulk_copies":"cnt","expect_tx":1,"proxy_fences":2,"A_ent_bytes":"2*cnt"},
                "new_scratch_traffic":{"global_x_bytes":"unchanged 8192*cnt","shared_writes":"8192*cnt","shared_reads":"8192*cnt"},
                "flag_and_barrier_bytes":"4 B per atomic/poll; wait retry counts unmeasured",
                "spills":"UNMEASURED: mandatory nvcc gate"}
    return {"identity":default_identity(base,generated),"geometry":geometry_gate(),"dry_run":dry_run(base,generated),
            "accounting":accounting,"per_bit_generated_lines":sites,
            "changed_markers":lines,"cuda_diff":list(difflib.unified_diff(original.splitlines(),cu.splitlines(),n=1)),
            "nvcc_spills_sass":"NOT_RUN", "GPU_correctness_timing":"NOT_RUN"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check",action="store_true",help="read-only regeneration, CPU gates, dry run")
    args=ap.parse_args()
    base=BASE.read_text(encoding="utf-8")
    generated=generate(base)
    if args.check:
        assert OUTPUT.read_bytes()==generated.encode(), "generated file is stale"
    else:
        OUTPUT.write_bytes(generated.encode("utf-8"))
    result=cpu_gate(base,generated)
    result.pop("cuda_diff")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

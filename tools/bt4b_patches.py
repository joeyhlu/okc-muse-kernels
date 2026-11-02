#!/usr/bin/env python3
"""BT4b timing-only ladder; copied BT4 source-bank / reversible-edit machinery.

Kill risks FIRST (these are bounds, not proposed production optimizations):
AB1: helper work may remain the tail; skipping the hop can expose a ring wait.
AB2: helper may not pace; uniform helper guards may increase register pressure.
COMBSKIP: combine may already overlap; sumsq removal may perturb helper pacing.
FIN: stragglers may dominate; parity/reset overhead may eat the barrier saving.
No new timing claims. MEASURED pools: bt4_sonnet C2/C5, lab R-tinymk-1/1b:
A->B 2.0-2.6 us; barrier+combine 1.38/3.06 us T1/T8. VMS pool unmeasured.
EST serial barrier ceiling 1*1.09 us (R-pdle-0), not a net saving prediction.

Protocol / cleanup ownership (all indices are source facts, not timings):
TINY TIDX=(threadIdx.x+256)%288: consumers logical 0..255 = physical 32..287;
helper logical 256..287 = physical 0..31. Every CTA 0..G-1 keeps both WS_JOIN,
all ring waits, wsrf/wsre arrivals and the helper fence/publication. AB1/AB2
ONLY waive the global doneA data-validity poll (explicit stale-inter probe).
These exact TINY builds set xready=nullptr; VMS uses ctr[8+j] as doneA and
has no xready consumer. No generic HYB/VML xready protocol is modified.
No address, route, loop trip count or synchronization count depends on y/inter.
AB2 retains all 8 consumer partial stores and every helper protocol operation;
it reads only warp 0's partial, including the last B units owned by WSBLK warps.
COMBSKIP keeps every original barrier, removes ss arithmetic/atomics and writes
the first VALID local pair directly from B to out (no write for empty tokens),
keeping normal final resets. Before the existing WS_JOIN, tiny tid0..M-1 marks
one row/token in S.ent's low 3 bits; VMS route warp0 does this before its existing
CTA join, with one extra warp sync. Every A-reader uses S.ent[row] >> 3 ONLY;
the token index is unchanged. Low bits become 0 for first pair, 1 otherwise.
FIN exits after the final WS_JOIN / VMS __syncthreads, before the grid barrier.
TINY all consumer CTAs logical tid0 increment ctr[4+par]; last increments ctr[3].
Entry CTA0 logical tid0..87 clears other doneA bank (JMAX=88), tid0..1 ctr[0:2];
all CTAs logical tid0..255 still clear other ss via i=CTA*256+tid, stride G*256
(EMAX=3072: CTA0..11 cover all). CTA0/tid0 resets the other exit counter.
VMS all CTAs physical tid0 increment ctr[4+par]; last increments ctr[0]. Entry
CTA0 tid0..63 clears other doneA (MAXJ=64), tid0..127 clears other ss (MAXP=128),
tid0 clears other exit counter. ctr[1] is unused by FIN; error ctr[2] never reset.
Both parity banks start zero, each mask/shape owns scratch, replay is on ONE
ordinary ordered stream, never PDL. Last exit means every CTA read generation;
kernel completion orders all reset/data stores before the next replay. Current
and other parity never alias. No reset launch; net timing includes the prologue.
Non-FIN TINY retains ctr[0:2], doneA clears above and token CTAs it=CTA+k*G,
tid0..255 (4 columns each); VMS retains CTA0 tid0..63 doneA, tid0..127 other ss,
tid0 generation increment, token it=CTA+k*G<m*4, tid0..255. Helper-only exits
are unchanged; FIN removes the entire unused token/combine loop intentionally.

Hot-loop accounting, EST source operations (actual SASS in Linux gate JSON):
All A/B weight stages: 36,864 bytes before=after; all MMA/DQ/copy instructions
unchanged. No added bytes per streamed stage. AB1 removes lane0's variable
4-byte global loads/nanosleeps at job transitions, keeps warp synchronization.
AB2 per helper lane/unit/tile: partial LDS 8*16=128 -> 16 bytes; 32 -> 4 FADD
in TINY / VMS tile2; VMS tile1 7*4=28 -> 0 FADD (initial LDS retained).
Across 32 helper lanes: 4096 -> 512 shared bytes/tile; consumer STS unchanged.
Per valid A result: scale multiplies unchanged, div/exp/add -> multiply-only;
exact instruction lowering compiler-dependent. All inter stores unchanged.
COMBSKIP per B lane/h: two squares+add, 3 shuffle/add, selected ss atomic -> 0;
y stores 2*4 bytes -> 2*2 output bytes for the first pair, 0 otherwise; two
BF16 conversions are added to that first-pair lane. Per token final read
16,384*local_pairs -> 0 bytes; output 8192 bytes for nonempty tokens unchanged;
4 weighted FMAs + rsqrt/weight per final lane/pair -> 0. Helper adds 1 shared
index load and a low-bit test per valid B lane/h, and skips norm-weight loads.
Per B lane/unit/tile (2 h), y+norm+new-index traffic excluding the removed ss
atomics: tiny 16+4=20 -> at most 4+8=12 bytes; VMS 16+8=24 -> at most 8+8=16.
For VMS TWO tiles, norm loads are shared: 32+8=40 -> at most 16+16=32 bytes.
Shared partial LDS is unchanged unless AB2 is set; no stage byte growth.
Prologue per token with L local pairs: 16 shared loads + L stores (tiny 32+2L B,
VMS 64+4L B), 8 compare/select + 8 valid tests + L encode operations;
included in net timing. FIN keeps
the B loop entirely (including sumsq) and removes final reads/output entirely.
FIN moves 88*4+8=360 TINY / (64+128)*4=768 VMS reset bytes to entry; adds 4
bytes other-exit reset in VMS; TINY already reset it. Replaces fence/atomic/
poll + two CTA barriers by atomic/last-generation increment after existing join.
Parity adds uniform integer indexing at job publication/read, no weight traffic.
"""
import argparse
import ast
import difflib
import hashlib
import json
import pprint
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/muse_moe_w4a16/mb_layer.py'
DEST = ROOT / 'kernels/muse_moe_w4a16/mb_layer_bt4b.py'


def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:100])
    return text.replace(old, new, 1)


def make_edits(src, tiny):
    """Each recorded edit has an exact-once anchor, a bit and reversible text."""
    edits = []
    def edit(bit, old, new):
        nonlocal src
        src = replace_once(src, old, new)
        edits.append((bit, old, new))
    def wrap(bit, start, end, prefix, suffix):
        assert src.count(start) == src.count(end) == 1, (start, end)
        old = src[src.index(start):src.index(end)]
        edit(bit, old, prefix + old + suffix)

    edit('ABI', 'struct Params {\n', 'struct Params {\nint bt4b_mask; // BT4b captured host snapshot\n')
    host = 'static int g_bt4b_mask = 0;\nvoid bt4b_set_mask(int64_t mask) { g_bt4b_mask = (int)mask; }\n'
    if tiny:
        edit('ABI', 'static int g_okc_pdl = 0;\n', host + 'static int g_okc_pdl = 0;\n')
        edit('ABI', 'p.zero = 0;\n', 'p.zero = 0;\np.bt4b_mask = g_bt4b_mask;\n')
        edit('ABI', 'cudaStream_t st = at::cuda::getCurrentCUDAStream();\n',
             'TORCH_CHECK(!g_bt4b_mask || !g_okc_pdl, "BT4b probes require ordered replay");\n'
             'static_assert(2 * okc::JMAX <= 1024, "BT4b doneA banks fit before ss1");\n'
             'cudaStream_t st = at::cuda::getCurrentCUDAStream();\n')
        edit('FIN', 'p.doneA + j)', 'p.doneA + ((p.bt4b_mask & 8) ? (S.gen & 1) * JMAX : 0) + j)')
        edit('FIN', 'p.doneA + jj,', 'p.doneA + ((p.bt4b_mask & 8) ? (S.gen & 1) * JMAX : 0) + jj,')
        edit('AB1', 'if (lane == 0) { while (ld_acquire(p.doneA',
             'if (!(p.bt4b_mask & 3) && lane == 0) { while (ld_acquire(p.doneA')
        # Both helper and WSBLK-owned B tail reductions: retain constant unroll.
        for tail in ('const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];', '__syncwarp();\nif (lane == 0) mbar_arrive1(S.wsre + bb);'):
            old = ('for (int w = 0; w < NW; ++w) {\n'
                   'const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n'
                   'sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n}\n' + tail)
            new = old.replace('++w) {\n', '++w) { if (w == 0 || !(p.bt4b_mask & 2)) {\n', 1).replace('\n}\n' + tail, '\n} }\n' + tail)
            edit('AB2', old, new)
        edit('AB2', 'SILU_DIV(gt, 1.f + __expf(-gt)) * up',
             '((p.bt4b_mask & 2) ? gt * up : SILU_DIV(gt, 1.f + __expf(-gt)) * up)')
        edit('FIN', 'DBG(1);\n{\nfloat* other',
             'DBG(1);\nif (p.bt4b_mask & 8) {\n'
             'const int i0 = blockIdx.x * NT + tid;\n'
             'if (i0 < 2) p.ctr[i0] = 0;\n'
             'for (int i = i0; i < JMAX; i += gridDim.x * NT) p.doneA[((S.gen + 1) & 1) * JMAX + i] = 0;\n'
             '}\n{\nfloat* other')
        edit('FIN', 'DBG(3);\nWS_JOIN();\n',
             'DBG(3);\nWS_JOIN();\nif (p.bt4b_mask & 8) {\n'
             'if (tid == 0) { const int old = atomicAdd(p.ctr + 4 + (S.gen & 1), 1);\n'
             'if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1); }\nreturn;\n}\n')
        # Direct B output; all normal final barriers and resets remain.
        edit('COMBSKIP', 'DBG(1);\n', 'DBG(1);\n' + first_pair_setup(True))
        edit('COMBSKIP', 'qvp[s] = hasp ?', 'qvp[s] = (hasp && !(p.bt4b_mask & 4)) ?')
        edit('COMBSKIP', 'constexpr int FSEG = HD / (NT * 4);\n',
             'if (p.bt4b_mask & 4) return; // BT4b: B already wrote the first pair\nconstexpr int FSEG = HD / (NT * 4);\n')
        for expr in ('(uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l)',
                     '(uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8)',
                     '1.f + __bfloat162float(p.normw[n0])', '1.f + __bfloat162float(p.normw[n0 + 8])'):
            edit('COMBSKIP', expr, f'((p.bt4b_mask & 4) ? {"0u" if expr.startswith("(uint32_t)") else "1.f"} : {expr})')
    else:
        edit('ABI', 'void run(torch::Tensor x,', host + 'void run(torch::Tensor x,')
        edit('ABI', 'vms::Params p;\n', 'vms::Params p;\np.bt4b_mask = g_bt4b_mask;\n'
             'TORCH_CHECK(!g_bt4b_mask || !pdl, "BT4b probes require ordered replay");\n'
             'TORCH_CHECK(!(g_bt4b_mask & 8) || ctr.numel() >= 8 + 2 * vms::MAXJ, "BT4b parity scratch");\n')
        # All acquire variants and helper publications share the same parity.
        for expr in ('ld_acq(p.ctr + 8 + jj)', 'ld_rlx(p.ctr + 8 + jj)', 'atomicAdd(p.ctr + 8 + j,'):
            new = expr.replace('p.ctr + 8 + ', 'p.ctr + 8 + ((p.bt4b_mask & 8) ? (S.gen & 1) * MAXJ : 0) + ')
            # ld_acq occurs in both FX variants; wrap the full poll block once.
            count = src.count(expr)
            assert count == (2 if expr.startswith('ld_acq') else 1)
            if count == 2:
                old = src[src.index('#if VMS_FX & 1\n'):src.index('if (mk && tid == 0 && mrow[4]')]
                edit('FIN', old, old.replace(expr, new))
            else:
                edit('FIN', expr, new)
        start, end = '#if VMS_FX & 1\n', 'if (mk && tid == 0 && mrow[4]'
        poll = src[src.index(start):src.index(end)]
        edit('AB1', poll, poll.replace('if (lane == 0) {', 'if (!(p.bt4b_mask & 3) && lane == 0) {')
             .replace('(void)ld_acq(', 'if (!(p.bt4b_mask & 3)) (void)ld_acq('))
        edit('AB2', 'for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }',
             'for (int w = 1; w < NW; ++w) { if (!(p.bt4b_mask & 2)) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; } }')
        edit('AB2', 'for (int w = 0; w < NW; ++w) { const float4 u = S.red2[bq][w][lane]; sm2.x += u.x; sm2.y += u.y; sm2.z += u.z; sm2.w += u.w; }',
             'for (int w = 0; w < NW; ++w) { if (w == 0 || !(p.bt4b_mask & 2)) { const float4 u = S.red2[bq][w][lane]; sm2.x += u.x; sm2.y += u.y; sm2.z += u.z; sm2.w += u.w; } }')
        for var in ('sm', 'sm2'):
            old = f'const float ga = (h ? {var}.y : {var}.x) * gs, up = (h ? {var}.w : {var}.z) * gs;\nconst float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;'
            edit('AB2', old, old.replace('__fdividef(ga, 1.f + __expf(-ga)) * up', '(p.bt4b_mask & 2) ? ga * up : __fdividef(ga, 1.f + __expf(-ga)) * up'))
        edit('FIN', '__syncthreads();\n#if VMS_T16 && !(VMS_T16X & 4)',
             '__syncthreads();\nif ((p.bt4b_mask & 8) && c == 0) {\n'
             'if (tid < MAXJ) p.ctr[8 + ((S.gen + 1) & 1) * MAXJ + tid] = 0;\n'
             'if (tid < MAXP) p.ss[((S.gen + 1) & 1) * MAXP + tid] = 0.f;\n'
             'if (tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n}\n'
             '#if VMS_T16 && !(VMS_T16X & 4)')
        edit('FIN', '__syncthreads();\n#if __CUDA_ARCH__ >= 900\nasm volatile("griddepcontrol.launch_dependents;',
             '__syncthreads();\nif (p.bt4b_mask & 8) {\n'
             'if (tid == 0) { const int old = atomicAdd(p.ctr + 4 + (S.gen & 1), 1);\n'
             'if (old == G - 1) atomicAdd(p.ctr, 1); }\nreturn;\n}\n'
             '#if __CUDA_ARCH__ >= 900\nasm volatile("griddepcontrol.launch_dependents;')
        # VMS route is solely warp0. Sync its last pos writes before marking
        # ent[row]'s unused low bits; ORIGINAL __syncthreads publishes to helpers.
        edit('COMBSKIP', 'else S.pos[i] = -1;\n}\n}\n}\n__syncthreads();',
             'else S.pos[i] = -1;\n}\n}\nif (p.bt4b_mask & 4) {\n__syncwarp();\n' +
             first_pair_setup(False) + '}\n}\n__syncthreads();')
        edit('COMBSKIP', 'const float* ssc = p.ss + (gen & 1) * MAXP;\n',
             'if (p.bt4b_mask & 4) return; // BT4b: B already wrote the first pair\nconst float* ssc = p.ss + (gen & 1) * MAXP;\n')
        for expr in ('__ldg(p.nw + n0)', '__ldg(p.nw + n0 + 8)'):
            edit('COMBSKIP', expr, f'((p.bt4b_mask & 4) ? 1.f : {expr})')
    # B writes its selected pair straight to output: no final readback. Each
    # row-block/column is still owned exactly once, including WSBLK's tail.
    if tiny:
        stores=('st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n'
                'st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);')
        x,y='v0f','v1f'
    else:
        stores=('p.ybuf[(size_t)(st + tk) * HD + n0] = y0;\n'
                'p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;')
        x,y='y0','y1'
    assert src.count(stores)==2
    for occurrence in range(2):
        begin=src.index(stores); end=begin+len(stores)
        anchor_begin=begin
        while src.count(src[anchor_begin:end])!=1:
            anchor_begin=src.rfind('\n',0,anchor_begin-1)+1
        old=src[anchor_begin:end]
        direct=(f'if (p.bt4b_mask & 4) {{ // BT4b direct B output {occurrence}\nconst int entry = S.ent[st + tk], token = entry >> 3;\n'
                'if ((entry & 7) == 0) {\n'
                f'p.out[(size_t)token * HD + n0] = __float2bfloat16({x});\n'
                f'p.out[(size_t)token * HD + n0 + 8] = __float2bfloat16({y});\n'
                '}\n} else {\n' + stores.replace('\n','\n/* BT4b kept-y */ ',1) + '\n}')
        edit('COMBSKIP',old,old[:begin-anchor_begin]+direct)
    # Sumsq lives in two B epilogues (helper + WSBLK, or two VMS tiles).
    a, b, group = ('v0f', 'v1f', 'gg') if tiny else ('y0', 'y1', 'g')
    sq = f'float sq = {a} * {a} + {b} * {b};\n'
    reduction = ('sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n'
                 'sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n'
                 'sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n' +
                 (f'if ({group} == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n' if tiny else
                  f'if ({group} == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);\n'))
    assert src.count(sq) == src.count(reduction) == 2
    # Include unique surrounding stores in each exact-once edit.
    for _ in range(2):
        begin = src.index('\n' + sq) + 1
        end = src.index(reduction, begin) + len(reduction)
        old = src[begin:end]
        new = old[len(sq):].replace(reduction, 'if (!(p.bt4b_mask & 4)) {\n' + sq + reduction + '}\n')
        # Both blocks can be identical; disambiguate by preceding result line.
        begin = src.rfind('\n', 0, begin - 1) + 1
        old2 = src[begin:end]
        while src.count(old2) != 1:
            begin = src.rfind('\n', 0, begin - 1) + 1
            old2 = src[begin:end]
        edit('COMBSKIP', old2, old2[:len(old2)-len(old)] + new)
        # Next iteration must skip the newly guarded square declaration.
        if _ == 0:
            src = src.replace('if (!(p.bt4b_mask & 4)) {\n' + sq, 'if (!(p.bt4b_mask & 4)) {\n/*BT4b_SS*/ ' + sq, 1)
            bit, before, after = edits[-1]
            edits[-1] = (bit, before, after.replace('if (!(p.bt4b_mask & 4)) {\n' + sq, 'if (!(p.bt4b_mask & 4)) {\n/*BT4b_SS*/ ' + sq))
    return edits


def first_pair_setup(tiny):
    """Mark unused slot bits; every A-reader's token index stays unchanged."""
    m='p.M' if tiny else 'p.m'
    return f'''if ((p.bt4b_mask & 4) && tid < {m}) {{
int pr = -1;
#pragma unroll
for (int s = 0; s < 8; ++s) {{ const int q = S.pos[tid * 8 + s]; if (pr < 0 && q >= 0) pr = q; }}
#pragma unroll
for (int s = 0; s < 8; ++s) {{ const int q = S.pos[tid * 8 + s]; if (q >= 0) S.ent[q] = (tid << 3) | (q == pr ? 0 : 1); }}
}}
'''


EXTENSION = r'''

# BT4B_BEGIN -- GENERATED timing-only support; no probe is a model.
_BT4B_EDITS = @EDITS@
_bt4b_original_builds = _mbl_builds

def bt4b_sources(engine, instrument=True, nsm=132):
 if engine in ("tiny", "tiny2"):
  src, cpp = _HYBLO_CUDA, _HYB_MOE_CPP
  flags = list(_HYBLO_FLAGS) + list(next(v for k,m,v in _HYBLO_KEYS if k == engine))
 elif engine == "vms_hi":
  src, cpp = VMS_CUDA_SRC, VMS_CPP_SRC
  flags = ["-O3",f"-DVMS_G={nsm}","-DVMS_RO=1","-DVMS_T16=1","-DVMS_WT2=30","-DVMS_NS=2","-DVMS_ER=1",
           "-Xptxas=-warn-spills","-gencode=arch=compute_90a,code=sm_90a"]
 else:
  raise ValueError(engine)
 if instrument:
  for bit,old,new in _BT4B_EDITS["tiny" if engine.startswith("tiny") else "vms_hi"]:
   assert src.count(old) == 1, (bit,old[:80])
   src = src.replace(old,new,1)
  cpp += "\nvoid bt4b_set_mask(int64_t mask);\n"
 return src,cpp,flags

def _mbl_builds(deadline):
 mods = _bt4b_original_builds(deadline)
 nsm = torch.cuda.get_device_properties(0).multi_processor_count
 for engine in ("tiny","tiny2","vms_hi"):
  _mbl_deadline(deadline,"BT4b build")
  src,cpp,flags = bt4b_sources(engine,nsm=nsm)
  tag = hashlib.sha1((cpp+src+" ".join(flags)).encode()).hexdigest()[:12]
  fn = ["run","attrs"] if engine == "vms_hi" else ["okc_run","okc_set_pdl"]
  mod = _mbl_load_inline(name="bt4b_"+engine+"_"+tag,cpp_sources=cpp,cuda_sources=src,
                         functions=fn+["bt4b_set_mask"],extra_cuda_cflags=flags,verbose=False)
  mod.bt4b_set_mask(0)
  if engine.startswith("tiny"): mod.okc_set_pdl(int(HYBLO_PDL))
  mods[engine] = mod
 return mods

def _bt4b_init(state,dev):
 state["bt4b_flag"] = torch.zeros(1,dtype=torch.int32,device=dev)
 state["bt4b_arms"] = {}

def bt4b_arm(state,mask,m):
 # One mutable device word, read back outside capture; each launch snapshots
 # the verified value as Params.bt4b_mask. Graphs cannot inherit later flags.
 if mask not in (0,1,2,4,8,5,10) or m not in (1,8,16):
  raise ValueError((mask,m))
 state["bt4b_flag"].fill_(mask)
 actual = int(state["bt4b_flag"].item())
 assert actual == mask
 print(f"[bt4b] flag readback={actual} M={m}",flush=True)
 for name in ("tiny","tiny2","vms_hi"):
  state["mods"][name].bt4b_set_mask(actual)
  if name.startswith("tiny"): state["mods"][name].okc_set_pdl(0)
 key = (mask,m)
 if key not in state["bt4b_arms"]:
  arm = dict(state)
  dev = state["out"][m].device
  if m <= 8:
   arm["h_sc"],arm["h_full"] = _hyb_scratch(dev)
  else:
   # Accessors are copied EXACTLY from _mbl_call / _mbl_banks.
   lo = 0
   st = state["banks"][lo]["k2a33"]["st"]
   vs = dict(st["vms"])
   for key2 in ("inter","ybuf","ss"):
    vs[key2] = torch.zeros_like(vs[key2])
   vs["ctr"] = torch.zeros((8+2*64,),dtype=torch.int32,device=dev)
   arm["banks"] = dict(state["banks"])
   arm["banks"][lo] = dict(state["banks"][lo])
   arm["banks"][lo]["k2a33"] = dict(state["banks"][lo]["k2a33"])
   arm["banks"][lo]["k2a33"]["st"] = dict(st,vms=vs)
  state["bt4b_arms"][key] = arm
 return state["bt4b_arms"][key]
# BT4B_END
'''


def environment(source):
    """Literal globals / CUDA renderers, copied from bt4_patches; no torch."""
    tree = ast.parse(source)
    env = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            safe = (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.Add,
                    ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod, ast.LShift,
                    ast.RShift, ast.USub, ast.UAdd, ast.Tuple, ast.List)
            if not all(isinstance(n, safe) for n in ast.walk(node.value)):
                continue
            try:
                value = eval(compile(ast.Expression(node.value), '<constant>', 'eval'), {'__builtins__': {}}, env)
            except (NameError, TypeError):
                continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                env[t.id] = value
            elif isinstance(t, ast.Tuple) and all(isinstance(x, ast.Name) for x in t.elts):
                env.update((x.id, v) for x, v in zip(t.elts, value))
    for name in ('_vml_wg_spec', '_vml_wg_spec64'):
        n = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module([n], type_ignores=[]), '<CUDA renderer>', 'exec'), env)
    n = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_vml_CUDA' for t in n.targets))
    exec(compile(ast.Module([n], type_ignores=[]), '<CUDA substitution>', 'exec'), env)
    return env


def generate(base):
    env = environment(base)
    edits = {k: make_edits(env[s], k == 'tiny') for k, s in (('tiny', '_HYBLO_CUDA'), ('vms_hi', 'VMS_CUDA_SRC'))}
    text = replace_once(base, ' _mbl_validate(state,weights)\n', ' _bt4b_init(state,dev)\n _mbl_validate(state,weights)\n')
    text = text.rstrip() + '\n' + EXTENSION.replace('@EDITS@', pprint.pformat(edits,width=110,sort_dicts=False))
    ast.parse(text)
    return text


def extension_env(source):
    env = environment(source)
    env['_mbl_builds'] = lambda deadline: {}
    exec(compile(source[source.index('# BT4B_BEGIN'):], '<BT4b support>', 'exec'), env)
    return env


def line_spans(values):
    spans=[]
    for v in sorted(set(values)):
        if spans and spans[-1][1]+1==v: spans[-1][1]=v
        else: spans.append([v,v])
    return spans


def bit_line_map(original, edits, offset):
    """Track literal changed characters through overlapping reversible edits."""
    source=original
    tags=[frozenset() for _ in source]
    origins=[]; line=offset+1
    for ch in source:
        origins.append(line)
        line+=ch=='\n'
    base_lines={}; insertion_lines={}
    for bit,old,new in edits:
        at=source.index(old)
        nt=[]; no=[]
        for tag,i,j,k,l in difflib.SequenceMatcher(a=old,b=new,autojunk=False).get_opcodes():
            if tag=='equal':
                nt.extend(tags[at+i:at+j]); no.extend(origins[at+i:at+j])
            else:
                base_lines.setdefault(bit,set()).update(x for x in origins[at+i:at+j] if x is not None)
                if i==j:
                    near=next((x for x in origins[at+i:] if x is not None),offset+original.count('\n')+1)
                    insertion_lines.setdefault(bit,set()).add(near)
                inherited=frozenset({bit}).union(*(tags[at+i:at+j]))
                nt.extend([inherited]*(l-k)); no.extend([None]*(l-k))
        source=source[:at]+new+source[at+len(old):]
        tags[at:at+len(old)]=nt; origins[at:at+len(old)]=no
    final_lines={}; line=1
    for ch,tag in zip(source,tags):
        for bit in tag: final_lines.setdefault(bit,set()).add(line)
        line+=ch=='\n'
    return {bit:{'base_file_changed_lines':line_spans(base_lines.get(bit,())),
                 'base_insert_before_lines':line_spans(insertion_lines.get(bit,())),
                 'final_cuda_changed_lines':line_spans(final_lines.get(bit,()))}
            for bit in sorted(base_lines.keys()|final_lines.keys()|insertion_lines.keys())}


def change_report(base, variant):
    env = extension_env(variant)
    constants={t.id:n.lineno for n in ast.parse(base).body if isinstance(n,ast.Assign)
               for t in n.targets if isinstance(t,ast.Name)}
    result = {}
    for engine in ('tiny', 'tiny2', 'vms_hi'):
        src = env['bt4b_sources'](engine, False)[0]
        edits = env['_BT4B_EDITS']['tiny' if engine.startswith('tiny') else engine]
        steps = []
        for bit, old, new in edits:
            line = src[:src.index(old)].count('\n') + 1
            steps.append({'bit': bit, 'edit_sequence_line': line, 'before': old, 'after': new})
            src = replace_once(src, old, new)
        a = env['bt4b_sources'](engine, False)[0].splitlines()
        b = src.splitlines()
        offset=constants['_HYBLO_CUDA' if engine.startswith('tiny') else 'VMS_CUDA_SRC']-1
        result[engine] = {'per_bit_exact_edits': steps,
                         'per_bit_lines':bit_line_map(env['bt4b_sources'](engine,False)[0],edits,offset),
                         'final_diff': [
            {'kind':tag,'base_cuda_lines': [i+1, j] if i<j else [],
             'base_file_lines':[offset+i+1,offset+j] if i<j else [],
             'base_insert_before':offset+i+1 if i==j else None,
             'probe_cuda_lines': [k+1, l] if k<l else [], 'before': a[i:j], 'after': b[k:l]}
            for tag,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes() if tag != 'equal']}
    return result


def cpu_gates(base, variant):
    assert variant == generate(base), 'generated drift'
    a, b = environment(base), extension_env(variant)
    strings = ('VMS_CUDA_SRC', 'VMS_CPP_SRC', '_HYBLO_CUDA', '_HYB_MOE_CUDA', '_HYB_MOE_CPP', '_vml_CUDA', '_vml_CPP')
    for name in strings:
        assert a[name] == b[name], name
    for engine in ('tiny', 'tiny2', 'vms_hi'):
        old = b['bt4b_sources'](engine, False)[0]
        new = b['bt4b_sources'](engine)[0]
        ent_lines=[line for line in old.splitlines() if 'S.ent[' in line]
        assert len(ent_lines)==(3 if engine=='vms_hi' else 2)
        assert all('>> 3' in line for line in ent_lines if 'S.ent[pr] =' not in line and 'S.ent[q] =' not in line)
        # All shared-data producer/consumer waits and rendezvous remain literal.
        import re
        for call in ('mbar_wait','mbar_arrive','mbar_arrive1','CTA_SYNC','WS_JOIN','__syncthreads','__syncwarp'):
            pattern=rf'\b{call}\([^;]*;'
            before,after=re.findall(pattern,old),re.findall(pattern,new)
            if engine=='vms_hi' and call=='__syncwarp':
                assert after==before+['__syncwarp();'],(engine,call)
            else:
                assert before==after,(engine,call)
        for bit, x, y in reversed(b['_BT4B_EDITS']['tiny' if engine.startswith('tiny') else engine]):
            new = replace_once(new, y, x)
        assert new == old
    # Unchanged routing, call ABI, prepare validation: flag0 exercises these.
    def funcs(s):
        return {n.name: ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    fa, fb = funcs(base), funcs(variant)
    for name in ('_mbl_call','_mbl_candidates','_mbl_validate','kernel_fn','_mbl_banks'):
        assert fa[name] == fb[name], name
    for n in (64, 88, 128, 3072):
        coverage = [c*256+t for c in range(132) for t in range(256) if c*256+t<n]
        assert coverage == list(range(n))
    # COMBSKIP first-pair ownership and output coordinates, including empty,
    # sparse top8, VMS tile2 and TINY's WSBLK-owned last B units.
    for present in range(256):
        pos=[100+s if present & (1<<s) else -1 for s in range(8)]
        first=next((q for q in pos if q>=0),-1)
        for token in (0,7,15):
            ent={q:(token<<3)|(0 if q==first else 1) for q in pos if q>=0}
            assert all((entry>>3)==token for entry in ent.values())
            writers=[q for q,entry in ent.items() if (entry&7)==0]
            assert writers==([first] if present else [])
    for tile in (0,8):
        for token in range(tile,tile+8):
            cols=[16*rb+(lane>>2)+delta for rb in range(256) for lane in range(32)
                  for h in range(2) if tile+2*(lane&3)+h==token for delta in (0,8)]
            assert sorted(cols)==list(range(4096))
    for nB in range(260):
        for wsblk in (1,2):
            tail=min(wsblk,nB)
            helper=list(range(nB-tail))
            consumers=[nB-1-w for w in range(tail)]
            assert sorted(helper+consumers)==list(range(nB))
    assert sorted((physical+256)%288 for physical in range(32,288))==list(range(256))
    # Multi-replay parity model includes arbitrary CTA completion order and nj=0.
    for G in (1, 4, 132):
        done, exits, gen = [[0]*88, [0]*88], [0,0], 0
        for replay in range(10):
            par, other = gen&1, (gen+1)&1
            assert all(v == 0 for v in done[par]) and exits[par] == 0
            done[other] = [0]*88; exits[other] = 0
            for c in reversed(range(G)):
                done[par][c%88] += 1
                exits[par] += 1
                if exits[par] == G: gen += 1
            assert gen == replay+1
    return {'status':'PASS','original_cuda_globals_equal':list(strings),'untouched_routes_equal':True,
            'runtime_source_erasure_equal':True,'parity_coverage':'PASS; GPU replay remains required',
            'combine_output_and_wsblk_coverage':'PASS; 256 top8 masks, both VMS tiles, WSBLK 1/2',
            'ROUTE0':'not implemented (optional)','sha256':hashlib.sha256(variant.encode()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true'); p.add_argument('--changes',action='store_true')
    args = p.parse_args()
    base = BASE.read_text(encoding='utf-8'); variant = generate(base)
    if args.check:
        assert DEST.read_bytes() == variant.encode(), 'regenerate variant'
    else:
        DEST.write_bytes(variant.encode())
    result = cpu_gates(base,variant)
    if args.changes: result['changes'] = change_report(base,variant)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

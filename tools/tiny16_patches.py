"""TINY16: generate the isolated HYBLO t16 experiment; no GPU or git actions.

Only kernels/lab/k3_tiny16_ab.py is written. Every replacement is unique within
its explicitly selected source/function. The original HYBLO CUDA/flags remain
verbatim for tiny/tiny2. Run with python -B -I tools/tiny16_patches.py.
"""
import ast
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/lab/k3_tiny16_ab.py"


def sub(s, old, new):
    assert s.count(old) == 1, ("non-unique anchor", old[:100], s.count(old))
    return s.replace(old, new, 1)


def section(s, first, last):
    assert s.count(first) == s.count(last) == 1, (first, last)
    a, b = s.index(first), s.index(last)
    assert a < b
    return s[a:b]


def literals(text):
    names = {"_HYBLO_CUDA", "_TINY16_CUDA", "_HYBLO_KEYS", "_HYBLO_FLAGS",
             "HYBLO_SELF_TEST", "HYBLO_T", "_HYBLO_NOTICE", "_HYB_MOE_CPP"}
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    assert len(nodes) == len({n.targets[0].id for n in nodes}), 'duplicate literal anchor'
    return {n.targets[0].id: ast.literal_eval(n.value) for n in nodes}


UNIT = r'''
__device__ __forceinline__ uint4 t16_ldcg(const uint4* p) {
uint4 v;
asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
: "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
return v;
}

// One full-K 16-row weight unit; TWO is uniform across all eight compute warps.
// x0 stays in registers across units of an expert. Only one x2 K chunk is ahead.
template <bool TWO>
__device__ __forceinline__ void t16_unit(const unsigned char* slot,
const uint4 (&x0)[2 * NCH], const uint4* src2, bool valid2,
int warp, int lane, uint32_t k16, float4& out0, float4& out1) {
const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
float acc[4][4] = {}, acc2[4][4] = {};
uint4 next0 = make_uint4(0, 0, 0, 0), next1 = next0;
if constexpr (TWO) {
if (valid2) { next0 = t16_ldcg(src2); next1 = t16_ldcg(src2 + 1); }
}
#pragma unroll
for (int c = 0; c < NCH; ++c) {
const uint4 y0 = next0, y1 = next1;
// Issue chunk c+1 before the weight decode and both tiles' MMAs for chunk c.
if constexpr (TWO) {
if (c + 1 < NCH && valid2) {
next0 = t16_ldcg(src2 + (c + 1) * 8);
next1 = t16_ldcg(src2 + (c + 1) * 8 + 1);
}
}
const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
const uint32_t b2[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
for (int st = 0; st < 4; ++st) {
uint32_t r0, r1, r2, r3;
dq4(ww[st], r0, r1, r2, r3);
const uint32_t a0 = bmul2(r0, sg), a1 = bmul2(r1, sg8);
const uint32_t a2 = bmul2(r2, sg), a3 = bmul2(r3, sg8);
mma_bf16(acc[st], a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
if constexpr (TWO) mma_bf16(acc2[st], a0, a1, a2, a3, b2[2 * st], b2[2 * st + 1]);
}
}
// Keep TINY2's four-accumulator grouping and FP32 addition order.
#pragma unroll
for (int k = 0; k < 4; ++k) {
acc[0][k] += acc[1][k] + acc[2][k] + acc[3][k];
if constexpr (TWO) acc2[0][k] += acc2[1][k] + acc2[2][k] + acc2[3][k];
}
out0 = make_float4(acc[0][0], acc[0][1], acc[0][2], acc[0][3]);
if constexpr (TWO) out1 = make_float4(acc2[0][0], acc2[0][1], acc2[0][2], acc2[0][3]);
}

template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
const uint32_t k16 = 16u + p.zero;
const int rbs = PB ? RB_B : RB_A, nU = v1 - v0, seq0 = seq;
uint4 x0[2 * NCH];
int curj = -1;
bool two = false, valid2 = false;
const uint4* src2 = nullptr;
for (int i = 0; i < nU; ++i) {
const int v = v0 + i, j = S.njF + v / rbs;
if (j != curj) {
curj = j;
if (PB) {
if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
__syncwarp();
}
// Both activation tiles are read only after this expert-ready acquire (PB).
load_x_regs<PB>(p, S, j, lane, warp, x0);
two = __all_sync(0xffffffffu, S.jC[j] > 8);
const int r = 8 + (lane >> 2), st = S.jS[j];
valid2 = r < S.jC[j];
src2 = nullptr;
if (valid2) {
const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + r) * ID
: p.x + (size_t)(S.ent[st + r] >> 3) * HD;
src2 = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * (lane & 3));
}
}
const int q = seq0 + i, sl = q % OKC_NS;
mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
const unsigned char* slot = ring + sl * UNIT_BYTES;
float4 a, b;
if (two) t16_unit<true>(slot, x0, src2, valid2, warp, lane, k16, a, b);
else t16_unit<false>(slot, x0, src2, valid2, warp, lane, k16, a, b);
const int bb = q % NRB;
// Both helper leaders acknowledge every unit, including a one-tile job.
mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));
*reinterpret_cast<float4*>(&S.red[bb][0][warp][lane][0]) = a;
if (two) *reinterpret_cast<float4*>(&S.red[bb][1][warp][lane][0]) = b;
CTA_SYNC();
if (tid == 0) {
const int nA = S.wsr[1] - S.wsr[0], N = nA + S.wsr[3] - S.wsr[2];
if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
mbar_arrive1(S.wsrf + bb);
}
}
seq = seq0 + nU;
}

'''


def cuda_t16(base):
    """Derive a new source; never edit the tiny/tiny2 CUDA literal."""
    s = sub(base, '#define OKC_WSNR 4', '#define OKC_WSNR 2')
    s = sub(s, 'constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);',
            'constexpr int NTL = NT + 128; // physical compute 0..7; helpers 8..11')
    s = sub(s, '#define TIDX ((int)((threadIdx.x + NT) % NTL))', '#define TIDX ((int)threadIdx.x)')
    s = sub(s, '#define OKC_SKT 8', '#define OKC_SKT 16')
    s = sub(s, 'static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");',
            'static_assert(OKC_SKT == 16 && OKC_WS && OKC_NS == 3 && OKC_NW == 8 && OKC_WSBLK == 0 && !OKC_SPLIT && OKC_PRMTSC && OKC_DQ7, "TINY16 geometry/precision");')
    s = sub(s, 'constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;',
            'constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32; // keep scratch ABI\n'
            'constexpr int SM_EMAX = 32 * 8; // compact route metadata; host enforces M <= 16')
    sm = section(s, 'struct Smem {', '__device__ __forceinline__ int muldiv_g')
    ns = sub(sm, 'uint32_t lut[256];', 'uint32_t lut[4]; // PRMTSC: unused; retain 16-byte red alignment')
    ns = sub(ns, 'float red[NRB][NW][32][RED_W];',
             'float red[NRB][2][NW][32][4]; // tile-major: retain contiguous float4 lanes')
    ns = sub(ns, 'uint64_t wse[8], wsrf[8], wsre[8];', 'uint64_t wse[OKC_NS], wsrf[NRB], wsre[NRB];')
    for ty, name in [('signed char', 'eid'), ('short', 'ent'), ('short', 'pos')]:
        ns = sub(ns, f'{ty} {name}[EMAX];', f'{ty} {name}[SM_EMAX];')
    s = sub(s, sm, ns)
    old = section(s, 'template <bool PB>\n__device__ void skinny_ring_ws',
                  '__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {')
    s = sub(s, old, UNIT)
    helper = section(s, '__device__ void ws_helper(', 'template <bool BIG, bool PB>')
    h = sub(helper, 'const int lane = TIDX & 31;\nWS_JOIN();',
            'const int lane = TIDX & 31, tile = (TIDX - NT) >> 5;\nWS_JOIN();\n'
            'if (tile >= 2) { WS_JOIN(); return; } // helper WG stays converged at both joins')
    h = sub(h, 'if (lane == 0)\nfor (int q = S.preissued;',
            'if (tile == 0 && lane == 0)\nfor (int q = S.preissued;')
    h = sub(h, 'const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;', 'const int Nh = N;')
    h = sub(h, 'float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll',
            'const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n'
            'const bool active = tile == 0 || cnt > 8;\n'
            'float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\nif (active) {\n#pragma unroll')
    h = sub(h, '&S.red[bb][w][lane][0]', '&S.red[bb][tile][w][lane][0]')
    h = sub(h, '}\n__syncwarp();\nif (lane == 0) mbar_arrive1(S.wsre + bb);\nconst int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];',
            '}\n}\n__syncwarp();\nif (lane == 0) mbar_arrive1(S.wsre + bb);')
    # Exact separate phase anchors; arithmetic and all store guards are retained.
    for first, last in [('if (!PB) {', '} else {\nconst float gsc = p.gsB[ce]'),
                        ('} else {\nconst float gsc = p.gsB[ce]', '\n}\n}\n__threadfence();\nWS_JOIN();')]:
        block = section(h, first, last)
        nb = sub(block, 'const int tk = 2 * tt + h;', 'const int tk = 8 * tile + 2 * tt + h;')
        if first == 'if (!PB) {':
            nb = sub(nb, 'if (!PB) {\nconst float gsc', 'if (!PB) {\nif (active) {\nconst float gsc')
            nb = sub(nb, 'if (q == nA - 1) {', '}\nif (q == nA - 1) {')
            nb = sub(nb, '__threadfence();\n__syncwarp();\nif (lane == 0) {',
                     '__threadfence();\n'
                     'asm volatile("bar.sync 3, 64;\\n" ::: "memory"); // both tiles released\n'
                     'if (tile == 0 && lane == 0) {')
        else:
            nb = sub(nb, '} else {\nconst float gsc', '} else if (active) {\nconst float gsc')
        h = sub(h, block, nb)
    s = sub(s, helper, h)
    # No unsupported fat job can reach this kernel: exact M bound is checked on host.
    s = sub(s, 'if (nF > 0) __trap();', '// Host M <= 16 and unique top-8 IDs imply nF == 0.')
    entry = section(s, 'template <bool BIG>\n__global__ void __launch_bounds__', 'static size_t okc_smem_bytes() {')
    e = sub(entry, 'if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }',
            'if (tid >= NT) {\n'
            'asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\\n" ::: "memory");\n'
            'ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return;\n}\n'
            'asm volatile("setmaxnreg.inc.sync.aligned.u32 192;\\n" ::: "memory");')
    # Initial empty-slot barrier must collect both helper leaders, not just helper 8.
    for first, last in [('#elif OKC_F1EARLY\n', 'if (ln < E8) {'),
                        ('#if OKC_SPLIT || !OKC_F1EARLY\n', '\n#if OKC_SPLIT\nconst int v0 = v01')]:
        block = section(e, first, last)
        nb = sub(block, 'mbar_init(S.wsre + k, 1);', 'mbar_init(S.wsre + k, 2);')
        e = sub(e, block, nb)
    # All compute CTAs reach the grid barrier and cleanup, even with no expert work.
    e = sub(e, 'WS_JOIN();\nasm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");',
            'WS_JOIN();\n'
            'asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\\n" ::: "memory");\n'
            'asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");')
    start = 'constexpr int FSEGP = HD / (NT * 4);'
    stop = 'CTA_SYNC();\nif (tid == 0) {\nint* bc = p.ctr + 4 + (S.gen & 1);'
    e = sub(e, section(e, start, stop), '// Sequential combine below: no eight-vector prefetch live range.\n')
    start = 'int it0 = blockIdx.x;\nif (hasp) {'
    stop = 'for (int it = it0; it < p.M * FSEG; it += gridDim.x) {'
    e = sub(e, section(e, start, stop), 'const int it0 = blockIdx.x;\n')
    s = sub(s, entry, e)
    smem = section(s, 'static size_t okc_smem_bytes() {', 'static int g_okc_pdl = 0;')
    s = sub(s, smem, 'static size_t okc_smem_bytes() {\n'
            'return ((sizeof(okc::Smem) + 127) / 128) * 128 + OKC_NS * okc::UNIT_BYTES;\n}\n'
            'static_assert(sizeof(okc::Smem) == 20240, "T16 metadata ABI");\n'
            'static_assert(((sizeof(okc::Smem) + 127) / 128) * 128 + OKC_NS * okc::UNIT_BYTES <= 140 * 1024, "T16 smem budget");\n\n')
    s = sub(s, 'okc::Params p;\np.wA',
            'TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 16 && nloc == 32, "TINY16 shape");\n'
            'okc::Params p;\np.wA')
    return s


VALIDATE = '''
# padz_run.py graph/replay pattern; the real VMS state is the st passed by prepare.
# A/B uses one prepared layer: ship=(1,8), t16=[1,8,16]. No module/state copying.
TINY16_ARMS=({"name":"ship","set":{"HYBLO_T":(1,8)}},
 {"name":"t16","set":{"HYBLO_T":[1,8,16]}},
 {"name":"mid1","set":{"HYBLO_T":(1,8)}},
 {"name":"t16b","set":{"HYBLO_T":[1,8,16]}},
 {"name":"mid2","set":{"HYBLO_T":(1,8)}})
def _tiny16_validate(st,w,dev,res):
 try:
  if "t16" not in _HYBLO["mods"]:
   res["t16"]=False
   print("[tiny16] OFF: build missing",flush=True)
   return
  assert st.get("vms_ok") and VMS_HI_ON and st["vms"].get("mod_hi") is not None, "VMS T16 reference unavailable"
  for bs in (9,12,13,16):
   for dist in ("skewed","conc","none"):
    d=_hyblo_inputs(bs,1100+bs,dev,dist)
    x,ids,tw=d["hidden_states"].contiguous(),d["topk_ids"].to(torch.int32).contiguous(),d["topk_weights"].float().contiguous()
    # These are comparisons to VMS itself, including empty-local routes.
    ship=_vms_call(st,x,ids,tw).clone()
    rf=_k2a_ref_fp32(d,w)
    a=_hyblo_run("t16",x,ids,tw).clone()
    torch.cuda.synchronize()
    sx,si,sw=x.clone(),ids.clone(),tw.clone()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
     go=_hyblo_run("t16",sx,si,sw)
    g.replay()
    torch.cuda.synchronize()
    gr=_k2a_rel_err(go,a)[0]
    # Reuse captured pointers while changing x, IDs, weights AND active-tile count.
    d2=_hyblo_inputs(bs,2200+bs,dev,"conc" if dist=="none" else "none" if dist=="conc" else "normal")
    xb,ib,wb=d2["hidden_states"].contiguous(),d2["topk_ids"].to(torch.int32).contiguous(),d2["topk_weights"].float().contiguous()
    sx.copy_(xb); si.copy_(ib); sw.copy_(wb)
    g.replay()
    torch.cuda.synchronize()
    eager=_hyblo_run("t16",xb,ib,wb).clone()
    ship2=_vms_call(st,xb,ib,wb).clone()
    rf2=_k2a_ref_fp32(d2,w)
    torch.cuda.synchronize()
    stale=_k2a_rel_err(go,eager)[0]
    ev=max(_k2a_rel_err(a,ship)[0],_k2a_rel_err(go,ship2)[0])
    e0,e1=_k2a_rel_err(a,rf),_k2a_rel_err(go,rf2)
    tot,row=max(e0[0],e1[0]),max(e0[1],e1[1])
    # Restore the first input to exercise both ping-pong ss generations repeatedly.
    sx.copy_(x); si.copy_(ids); sw.copy_(tw)
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    back=_k2a_rel_err(go,a)[0]
    finite=bool(torch.isfinite(a.float()).all() and torch.isfinite(go.float()).all() and torch.isfinite(eager.float()).all())
    ok=finite and ev<1e-2 and tot<1e-2 and row<5e-2 and max(gr,stale,back)<1e-3
    res["t16"]=res.get("t16",False) and ok
    print(f"[tiny16] M={bs} {dist} vs VMS={ev:.3e} FP32={tot:.3e}/{row:.3e} graph={gr:.3e} stale={stale:.3e} restore={back:.3e} -> {'ok' if ok else 'FAIL'}",flush=True)
    del g
 except Exception as ex:
  res["t16"]=False
  _HYBLO["err"]["t16"]=repr(ex)[:300]
  print(f"[tiny16] OFF: validation raised {ex!r}",flush=True)

'''


def generate(text):
    b = literals(text)
    cuda = cuda_t16(b['_HYBLO_CUDA'])
    text = sub(text, 'HYBLO_ENABLE=True\n',
               '_TINY16_CUDA=r"""' + cuda + '"""\n' + VALIDATE + 'HYBLO_ENABLE=True\n')
    # Keep the bundle default and the original two flag tuples exactly.
    key = ('t16', 16, ('-DOKC_L2H=1', '-DOKC_CPF=0', '-DOKC_WSBLK=0', '-DOKC_SPLIT=0'))
    line = next(x for x in text.splitlines() if x.startswith('_HYBLO_KEYS='))
    text = sub(text, line, '_HYBLO_KEYS=' + repr(b['_HYBLO_KEYS'] + (key,)))
    line = next(x for x in text.splitlines() if x.startswith('HYBLO_SELF_TEST='))
    cases = tuple((m, 100 + m + j, d) for m in (9, 12, 13, 16) for j, d in enumerate(('normal', 'skewed', 'conc', 'none')))
    text = sub(text, line, 'HYBLO_SELF_TEST=' + repr(b['HYBLO_SELF_TEST'] + cases))
    text = sub(text, '  fl=list(_HYBLO_FLAGS)+list(var)\n  t0=time.time()',
               '  fl=list(_HYBLO_FLAGS)+list(var)\n  src=_TINY16_CUDA if k=="t16" else _HYBLO_CUDA\n  t0=time.time()')
    text = sub(text, 'tag=hashlib.sha1((_HYB_MOE_CPP+_HYBLO_CUDA+" ".join(fl)).encode())',
               'tag=hashlib.sha1((_HYB_MOE_CPP+src+" ".join(fl)).encode())')
    text = sub(text, 'cpp_sources=_HYB_MOE_CPP,cuda_sources=_HYBLO_CUDA,',
               'cpp_sources=_HYB_MOE_CPP,cuda_sources=src,')
    text = sub(text, '  gok=_hyb_guards_ok()and all(bool((f[n:].view(vt)==sv).all())for f,n,vt,sv in _HYBLO["full"])',
               '  _tiny16_validate(st,w,dev,res)\n'
               '  gok=_hyb_guards_ok()and all(bool((f[n:].view(vt)==sv).all())for f,n,vt,sv in _HYBLO["full"])')
    # Separate additive notice; original Apache attribution remains byte-identical.
    text = sub(text, 'THIRD_PARTY_NOTICES+=_HYBLO_NOTICE\n',
               'THIRD_PARTY_NOTICES+=_HYBLO_NOTICE\n'
               'THIRD_PARTY_NOTICES+="\\nTINY16 modifications by team joeyhlu (2026-11-01): two token tiles, L2 x2 pipeline, two helpers, compact smem; derived from alygg77 helper0, Apache-2.0.\\n"\n')
    return text


def gates(base, generated):
    ast.parse(generated)
    compile(generated, str(OUT), 'exec')  # In-memory: no pyc files.
    assert generated.isascii() and '\r' not in generated
    assert len(generated.encode()) < 1_000_000
    b, g = literals(base), literals(generated)
    for name in ('_HYBLO_CUDA', '_HYBLO_FLAGS', '_HYBLO_NOTICE', '_HYB_MOE_CPP', 'HYBLO_T'):
        assert b[name] == g[name], name
    assert g['_HYBLO_KEYS'][:2] == b['_HYBLO_KEYS']
    assert g['_TINY16_CUDA'] == cuda_t16(b['_HYBLO_CUDA'])
    for first, last in [
        ('__device__ __forceinline__ void ring_issue(', 'template <bool PB>\n__device__ __forceinline__ void load_x_regs'),
        ('__device__ __forceinline__ void dq4(', '__device__ void early_issue(const Params& p, Smem& S);'),
        ('__device__ __forceinline__ void mma_bf16(', '__device__ __forceinline__ unsigned long long gtimer()')]:
        assert section(b['_HYBLO_CUDA'], first, last) == section(g['_TINY16_CUDA'], first, last)
    assert '__trap()' not in g['_TINY16_CUDA']
    assert 'x2s' not in g['_TINY16_CUDA']
    # No edits outside the narrowly enumerated Python nodes / inserted source.
    old = ast.parse(base).body
    new = ast.parse(generated).body
    changed = {'_HYBLO_KEYS', 'HYBLO_SELF_TEST', '_hyblo_build', '_hyblo_prepare'}
    added = {'_TINY16_CUDA', 'TINY16_ARMS', '_tiny16_validate'}
    def name(n):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)): return n.name
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name): return n.targets[0].id
        return None
    def keep(n):
        return name(n) not in changed | added and not (isinstance(n, ast.AugAssign)
            and isinstance(n.target, ast.Name) and n.target.id == 'THIRD_PARTY_NOTICES'
            and isinstance(n.value, ast.Constant) and 'TINY16 modifications' in str(n.value.value))
    assert [ast.dump(n) for n in old if keep(n)] == [ast.dump(n) for n in new if keep(n)]
    return {'bytes': len(generated.encode()), 'tiny_tiny2_source_flags': 'IDENTICAL',
            'weight_packet_dq_mma': 'IDENTICAL', 'base_sha256': hashlib.sha256(base.encode()).hexdigest(),
            'untouched_ast': 'PASS', 'sha256': hashlib.sha256(generated.encode()).hexdigest()}


def main():
    base = BASE.read_text(encoding='ascii')
    new = generate(base)
    result = gates(base, new)
    OUT.write_bytes(new.encode('ascii'))
    print(result)


if __name__ == '__main__':
    main()

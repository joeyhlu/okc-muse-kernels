"""AST2 independent riders; pure source construction, no layer import or GPU at import.

python -B -I tools/ast2_patches.py --ship --bits=31 --base=... --out=...
Without --ship, emit kernels/lab/k3_ast2_ab.py (lab-only capture selector).
CUDA compilation, resources, numerical and replay acceptance remain GPU gates.
"""
import argparse
import ast
import functools
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
LIMIT = 1_000_000
BITS = {1: "SK2ESC", 2: "RETIRE", 4: "ROUTE1", 8: "HYAQ", 16: "SK2TRIG"}
DROPPED = {}
SHAPES = {1: (4, 5), 2: (0, 1, 2, 3, 6, 7), 4: (0, 1, 6, 7),
          8: (3, 5, 7), 16: (4, 5)}


def check(ok, message):
    if not ok:
        raise AssertionError("[ast2] " + message)


def effective_bits(bits):
    check(isinstance(bits, int) and 0 <= bits <= 31, "bits must be 0..31")
    return bits & ~sum(DROPPED)


def shapes(bits):
    bits = effective_bits(bits)
    return tuple(sorted({s for b, ss in SHAPES.items() if b & bits for s in ss}))


def sub(s, old, new, what="anchor"):
    check(s.count(old) == 1, f"{what}: expected one anchor, got {s.count(old)}: {old[:90]!r}")
    return s.replace(old, new, 1)


def scoped(s, start, end, fn):
    check(s.count(start) == s.count(end) == 1, "unique scope " + start[:60])
    a, b = s.index(start), s.index(end)
    check(a < b, "scope order")
    return s[:a] + fn(s[a:b]) + s[b:]


@functools.lru_cache(maxsize=4)
def assignments(s):
    out = {}
    for n in ast.parse(s).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out.setdefault(n.targets[0].id, []).append(n)
    return out


def literal(s, name):
    ns = assignments(s).get(name, [])
    check(len(ns) == 1, "assignment " + name)
    return ast.literal_eval(ns[0].value)


def cuda_edit(s, name, fn):
    ns = assignments(s).get(name, [])
    check(len(ns) == 1, "CUDA assignment " + name)
    old = ast.get_source_segment(s, ns[0])
    m = re.match(r"\w+ = r([\"']{3})", old)
    check(m is not None, "raw literal " + name)
    return sub(s, old, old[:m.end()] + fn(literal(s, name)) + old[-3:], name)


def sk2esc(cu):
    # 32 exact-BF16 escape records per 8x64 unit, in the existing weight ring.
    # Each lane loads one record BEFORE releasing its slot; shuffles then patch
    # only statically indexed fragment registers. Over-cap weights use GD.
    check(cu.count("constexpr int TU = COMP ? 768 :") == 2, "sk2 TU")
    cu = cu.replace("constexpr int TU = COMP ? 768 :", "constexpr int TU = COMP ? 896 :")
    cu = sub(cu, "(NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32", "(NWR * WSK + 1) * 32")
    a = "const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;"
    b = "const int Mb = PK ? M : TOKR;"
    cu = scoped(cu, a, b, lambda _: """const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE;
const uint32_t zrow = bars + 16 * NSTAGE;
""")
    cu = scoped(cu, "if (COMP) {\nif constexpr (CL == 1)", "if constexpr (CL == 1) {\nmbar_init(empty0", lambda _: "")
    cu = cu.replace("NW + (COMP ? NESC : 0)", "NW")
    cu = sub(cu, "if (!COMP) mbar_init(pbar, 1);\nmbar_init(cbar, 32 * NESC);\nmbar_init(qbar, 1);\n", "")
    cu = sub(cu, "if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;",
             "if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(gbase + NSTAGE * SST + 16 * NSTAGE)[tid - 32] = 0u;")
    cu = scoped(cu, "if (warp >= NW + 1) {", "const int wr = warp / WSK", lambda _: "")
    cu = sub(cu, "uint2 cw[NT];", "uint2 cw[NT];\nuint32_t er[NT]; // AST2 SK2ESC: exact bits; no correction accumulators")
    anchor = 'asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));'
    cu = sub(cu, anchor, anchor + '\nasm volatile("ld.shared.u32 %0, [%1];" : "=r"(er[nt]) : "r"(wb + nt * TU + 768 + lane * 4));')
    anchor = "decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);"
    cu = sub(cu, anchor, anchor + """
const unsigned live = __ballot_sync(0xffffffffu, er[nt] != 0xffffffffu);
for (int z = 0; z < __popc(live); ++z) {
const uint32_t e = __shfl_sync(0xffffffffu, er[nt], z);
const int owner = (e >> 16) & 31, v = (e >> 21) & 15;
const uint32_t shift = (v & 1) * 16, mask = 0xffffu << shift;
#pragma unroll
for (int r = 0; r < 8; ++r)
if (lane == owner && (v >> 1) == r) b[nt][r] = (b[nt][r] & ~mask) | ((e & 0xffffu) << shift);
}
""")
    cu = sub(cu, "if constexpr (COMP) mbar_wait(cbar, 0);\n", "")
    cu = scoped(cu, "if constexpr (COMP) {\n#pragma unroll\nfor (int c = 0; c < NESC;", "#pragma unroll\nfor (int w = 0; w < WSK;", lambda _: "")
    cu = scoped(cu, "constexpr int NESC = COMP ?", 'TORCH_CHECK(M == TOKR,', lambda _: "const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 128;\n")
    check(cu.count("(NW + 1 + NESC) * 32") == 2, "sk2 launch threads")
    cu = cu.replace("(NW + 1 + NESC) * 32", "(NW + 1) * 32")
    check(not any(x in cu for x in ("corr[", "float* corr", "NESC", "ptab_s", "cbar")), "correction removed")
    return cu


def sk2esc_python(s):
    start = "    idx = esc.nonzero()\n"
    end = "\ndef _sk2_stream("
    # Avoid computing FP32 deltas: all 16 BF16 bits, including sign/zero/NaN payload,
    # are restored. Sentinel cannot alias a record (top six bits are always zero).
    body = '''    counts = esc.sum(dim=(-1, -2))
    if int(counts.max()) > 32:
        return None  # bounded per-unit cap; existing prepare leaves this projection on GD
    idx = esc.nonzero()
    unit = idx[:, 0] * U + idx[:, 1]
    flatcounts = counts.reshape(-1)
    starts = torch.cumsum(flatcounts, 0) - flatcounts
    rank = torch.arange(idx.shape[0], device=w.device) - starts[unit]
    records = torch.full((T * U, 32), -1, dtype=torch.int32, device=w.device)
    records[unit, rank] = (bits[esc] | (idx[:, 2] << 16) | (idx[:, 3] << 21)).to(torch.int32)
    restored = ((sign << 7 | base + o) << 8 | low)
    restored[esc] = bits[esc]
    if not bool((restored == bits).all()):
        return None
    packed = torch.cat((packed, records.view(torch.uint8).reshape(T, U, 128)), dim=-1).contiguous()
    return {'packed': packed, 'base4': base * 16843009}
'''
    s = scoped(s, start, end, lambda _: body)
    return scoped(s, "def _sk2_table(comp, n_cfg, R):", "def _sk2_build():", lambda _: '''def _sk2_table(comp, n_cfg, R):
    # ABI retained; kernel has no external correction table or correction scratch.
    return torch.empty(0, dtype=torch.int32, device=comp['packed'].device), 0

''')


def retire_hyb(cu):
    # doneA cleanup is distributed over CTA*NT+tid, not solely CTA zero.
    cu = sub(cu, "const bool hasp = itp < p.M * FSEGP;", """const bool hasp = itp < p.M * FSEGP;
const bool ast2_keep = hasp || blockIdx.x == 0 || blockIdx.x * NT < JMAX * 16; // doneAs cleanup spans JMAX * 16
""")
    cu = sub(cu, "while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);", "if (ast2_keep) while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);")
    return sub(cu, "}\nCTA_SYNC();\nDBG(4);", """}
// AST2 RETIRE: WS_JOIN + CTA_SYNC precede the unchanged release/arrival.
// The last arriver must still advance gen, even when this CTA retires.
if (!ast2_keep) return;
CTA_SYNC();
DBG(4);""")


def retire_vms(cu):
    cu = sub(cu, "const int gen = S.gen;\nif (tid == 0)", "const int gen = S.gen;\nconst bool ast2_keep = c == 0 || c < p.m * 4;\nif (tid == 0)")
    cu = sub(cu, "while (ld_acq(p.ctr + 1) < target) {", "while (ast2_keep && ld_acq(p.ctr + 1) < target) {")
    return sub(cu, "if (mk) mrow[8] = gt();\n}\n__syncthreads();", """if (mk) mrow[8] = gt();
}
// AST2 RETIRE: all threads joined before publication; CTA zero owns resets.
if (!ast2_keep) return;
__syncthreads();""")


def route1(cu):
    check("__device__ void early_issue(const Params& p, Smem& S) {\nreturn;" in cu, "ROUTE1 requires no-op route early_issue; FAST1 stays in moe_fused")
    code = r'''// AST2 ROUTE1: one warp builds; njF=-1 publishes a uniform fallback.
// FAST1 remains at its original issue point, before route().
#if OKC_TINY && !OKC_SPLIT
if ((p.M == 1 || p.M == 8) && p.nloc >= 0 && p.nloc <= 32) {
if (warp == 0) {
const int id0 = lane < E ? __ldg(p.topk_ids + lane) : -1;
const int id1 = lane + 32 < E ? __ldg(p.topk_ids + lane + 32) : -1;
const int v0 = id0 - p.lo, v1 = id1 - p.lo;
const int e0 = lane < E && v0 >= 0 && v0 < p.nloc ? v0 : -1;
const int e1 = lane + 32 < E && v1 >= 0 && v1 < p.nloc ? v1 : -1;
const unsigned dup = __match_any_sync(0xffffffffu, id0);
const bool unique1 = !__any_sync(0xffffffffu, lane < 8 && __popc(dup & 255u) != 1);
int c0 = 0, c1 = 0;
if (p.M == 1) {
const unsigned mask = __reduce_or_sync(0xffffffffu, e0 >= 0 ? (1u << e0) : 0u);
c0 = (mask >> lane) & 1;
} else {
#pragma unroll
for (int i = 0; i < 32; ++i) {
c0 += __shfl_sync(0xffffffffu, e0, i) == lane;
c1 += __shfl_sync(0xffffffffu, e1, i) == lane;
}
}
const int tot = c0 + c1;
const bool small = __all_sync(0xffffffffu, tot <= SK_T);
if (small && (p.M != 1 || unique1)) {
int incl = tot;
#pragma unroll
for (int d = 1; d < 32; d <<= 1) {
const int v = __shfl_up_sync(0xffffffffu, incl, d);
if (lane >= d) incl += v;
}
const int off = incl - tot;
const unsigned active = __ballot_sync(0xffffffffu, tot > 0);
const unsigned lt = (1u << lane) - 1u;
const int job = __popc(active & lt);
S.cnt[lane] = tot; S.off[lane] = off;
#pragma unroll
for (int w = 0; w < NW; ++w) S.wc[w][lane] = off + (w == 0 ? c0 : tot);
if (tot) { S.jE[job] = lane; S.jS[job] = off; S.jC[job] = tot; }
if (lane == 0) { S.njF = 0; S.njS = __popc(active); }
const unsigned peers0 = __match_any_sync(0xffffffffu, e0);
const unsigned peers1 = __match_any_sync(0xffffffffu, e1);
const int q0 = __shfl_sync(0xffffffffu, off, e0 < 0 ? 0 : e0) + __popc(peers0 & lt);
const int q1 = __shfl_sync(0xffffffffu, off + c0, e1 < 0 ? 0 : e1) + __popc(peers1 & lt);
if (lane < E) { S.eid[lane] = e0; S.pos[lane] = e0 < 0 ? -1 : q0; }
if (e0 >= 0) S.ent[q0] = lane;
if (lane + 32 < E) { S.eid[lane + 32] = e1; S.pos[lane + 32] = e1 < 0 ? -1 : q1; }
if (e1 >= 0) S.ent[q1] = lane + 32;
} else {
if (lane == 0) S.njF = -1;
}
}
CTA_SYNC(); // sole table-publication barrier, also joins original FAST1 issuer
if (S.njF == 0) {
early_issue(p, S);
return;
}
}
#endif
'''
    return sub(cu, "const int E = p.M * 8;\nconst int CH =", "const int E = p.M * 8;\n" + code + "const int CH =")


def hyaq(cu):
    helper = r'''// AST2 HYAQ: volatile coherent loads cannot cross the PDL memory clobber.
__device__ __forceinline__ uint4 ast2_qload(const __nv_bfloat16* p) {
uint4 v;
asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
return v;
}
'''
    cu = sub(cu, "#include <cuda_bf16.h>", "#include <cuda_bf16.h>\n" + helper)
    anchor = "tfor<L * 16 * 8, NT>(tidL, [&](int e) {"
    a = cu.index(anchor)
    b = cu.index("if (has_new) {", a)
    old = cu[a:b]
    # Restrict to L1 and the nonflat B1/L8 path; retain flat verify staging.
    cu = sub(cu, old, "if constexpr (!(L == 1 || (L == 8 && !FLAT))) {\n" + old + "}\n")
    for name, offset in (("qa", "q4 * 16"), ("qb", "q4 * 16 + 8"), ("qv", "partL * 8")):
        old = f"const uint4 {name} = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + {offset}]);"
        new = f"""const uint4 {name} = (L == 1 || (L == 8 && !FLAT))
? ast2_qload(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + {offset})
: *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + {offset}]);"""
        cu = sub(cu, old, new)
    # Deliberately keep all cp_commit/wait, metadata and sense/epoch accesses in
    # place: deleting an empty copy group changes wait_group's age semantics.
    return cu


def sk2trig(cu):
    guard = "(MT == 2 && TOKR == 32 && N == 8192 && K == 4096)"
    anchor = 'if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");\nif (warp == NW)'
    cu = sub(cu, anchor, 'if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");\n'
             '// AST2 SK2TRIG: all CTA threads trigger, after original two-stage prefetch and wait.\n'
             f'if (pdl && {guard}) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\nif (warp == NW)')
    return sub(cu, 'if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");',
               f'if (pdl && !{guard}) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");')


def sk2trig_python(s):
    # SBB's existing post-wait-stream instantiation accepts arbitrary T. No
    # sbb_kernel or host change, including when stacked on AST1 SBBPRE.
    s = sub(s, "VF_SBB2 = {1: True, 8: True, 16: False}", "VF_SBB2 = {1: True, 8: True, 16: False, 32: True}")
    a = '    if which == "ffn" and router_dtype is None and VF_SBB2.get(_k2a_bucket_of(tokens)):'
    return sub(s, a, a[:-1] + ''' and (
            _k2a_bucket_of(tokens) != 32 or (tokens == 32 and SK2_ON.get("unpooler")
            and _SK2["ok"].get("unpooler32_r144"))):''')


def apply_ship(s, bits, nonce=True):
    requested, bits = bits, effective_bits(bits)
    if bits & 1:
        s = cuda_edit(s, "_SK2_CUDA", sk2esc)
        s = sk2esc_python(s)
    if bits & 2:
        s = cuda_edit(s, "_HYBLO_CUDA", retire_hyb)
        s = cuda_edit(s, "VMS_CUDA_SRC", retire_vms)
    if bits & 4:
        s = cuda_edit(s, "_HYBLO_CUDA", route1)
    if bits & 8:
        s = cuda_edit(s, "HYA_CUDA_SRC", hyaq)
    if bits & 16:
        s = cuda_edit(s, "_SK2_CUDA", sk2trig)
        s = sk2trig_python(s)
    if bits and nonce:
        old = s.splitlines()[1]
        check(old.startswith("# nonce "), "nonce")
        s = sub(s, old + "\n", old + f" + AST2 requested={requested} effective={bits}\n")
    return s


def compile_jobs(base_text, bits):
    """Accept every subset; source capture only, never import the layer."""
    s = apply_ship(base_text, bits)
    bits = effective_bits(bits)
    flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
    jobs = {}
    if not bits or bits & (1 | 16):
        jobs['sk2'] = (literal(s, '_SK2_CUDA'), list(literal(s, '_SK2_FLAGS')))
    if not bits or bits & (2 | 4):
        for k, _, fl in literal(s, '_HYBLO_KEYS'):
            jobs['hyblo_' + k] = (literal(s, '_HYBLO_CUDA'), list(literal(s, '_HYBLO_FLAGS')) + list(fl))
    if not bits or bits & 2:
        vf = flags + ['-DVMS_G=132', '-DVMS_NS=3', '-DVMS_RO=1', '-Xptxas=-warn-spills']
        jobs['vms'] = (literal(s, 'VMS_CUDA_SRC'), vf)
        jobs['vms_hi'] = (literal(s, 'VMS_CUDA_SRC'), [f for f in vf if not f.startswith('-DVMS_NS=')]
                          + ['-DVMS_T16=1', '-DVMS_WT2=30', '-DVMS_NS=2', '-DVMS_ER=1'])
    if not bits or bits & 8:
        jobs['hya'] = (literal(s, 'HYA_CUDA_SRC'), flags + ['--expt-relaxed-constexpr'])
    if not bits or bits & 16:
        jobs['sbb'] = (literal(s, 'SBB1_CU') + literal(s, 'SBB1_HOST'), flags)
    return jobs


def gates(s, ship=True):
    compile(s, '<ast2>', 'exec')
    check(s.isascii() and '\r' not in s, 'ASCII LF output')
    n = len(s.encode('ascii'))
    check(not ship or n < LIMIT, f'ship size {n} >= {LIMIT}')
    return n


LAB = r'''
# AST2 lab only: separately compiled components and private mutable scratch.
# Never included in ship output. A failed build/gate is unavailable, not stock timing.
AST2_BITS = 0
AST2_KEYS = (1, 2, 4, 8, 16, 31)
AST2_LIVE = {0: True}
AST2_ERRORS = {}
AST2_DROPPED = {}
AST2_BUILD_S = {}
AST2_EFFECTIVE = {k: k for k in range(32)}
_AST2_BANK = {}
_AST2_MODULES = {}
_AST2_HITS = set()
_ast2_prepare0 = prepare
_ast2_kernel0 = kernel_fn
_ast2_sk20 = _sk2
_ast2_hroute0 = _hyblo_route
_ast2_vms0 = _vms_call
_ast2_hya0 = _hya_try
_ast2_sbb0 = _vf_sbb2

def _sk2(state, proj, x):
    out = _ast2_sk20(state, proj, x)
    if out is not None: _AST2_HITS.add(proj)
    return out

def _hyblo_route(*args, **kwargs):
    out = _ast2_hroute0(*args, **kwargs)
    if out is not None: _AST2_HITS.add('hyblo')
    return out

def _vms_call(*args, **kwargs):
    out = _ast2_vms0(*args, **kwargs)
    _AST2_HITS.add('vms')
    return out

def _hya_try(*args, **kwargs):
    out = _ast2_hya0(*args, **kwargs)
    if out is not None: _AST2_HITS.add('hya')
    return out

def _vf_sbb2(*args, **kwargs):
    out = _ast2_sbb0(*args, **kwargs)
    if out is not None: _AST2_HITS.add('sbb')
    return out

def _ast2_src(name, bits):
    src = globals()[name]
    for bit, changes in _AST2_CUDA_EDITS.items():
        if bits & bit and name in changes: src = globals()[changes[name]](src)
    return src

def _ast2_load(key, cpp, cu, fns, flags):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((cpp + cu + repr(flags)).encode()).hexdigest()[:16]
    if tag not in _AST2_MODULES:
        _AST2_MODULES[tag] = load_inline(name='ast2_' + key + '_' + tag, cpp_sources=cpp,
            cuda_sources=cu, functions=fns, extra_cuda_cflags=flags, verbose=False)
    return _AST2_MODULES[tag]

def _ast2_activate(bank):
    old = {k: globals()[k] for k in bank['globals']}
    globals().update(bank['globals'])
    return old

def _ast2_bank(state, bits, dev):
    flags = ['-O3', '-gencode=arch=compute_90a,code=sm_90a']
    bank = {'state': dict(state), 'globals': {}}
    st, gs = bank['state'], bank['globals']
    if bits & 6:
        assert HYBLO_ENABLE and HYBLO_K != 'w' and HYBLO_T == (1, 8)
        assert all(_HYBLO['ok'].get(k) for k in ('tiny','tiny2')), 'baseline HYBLO missing'
        h = dict(_HYBLO, mods={}, pdl={}, calls={}, err={}, ok=dict(_HYBLO['ok']))
        h['sc'], h['full'] = _hyb_scratch(dev)
        for k, _, fl in _HYBLO_KEYS:
            h['mods'][k] = _ast2_load('hyblo_'+k, _HYB_MOE_CPP, _ast2_src('_HYBLO_CUDA',bits),
                ['okc_run','okc_set_pdl'], list(_HYBLO_FLAGS)+list(fl))
        gs['_HYBLO'] = h
    if bits & 2:
        assert state.get('k2a33') and state['k2a33']['st'].get('vms_ok'), 'baseline VMS missing'
        vs = dict(state['k2a33']['st']['vms'])
        assert vs.get('mod_hi') is not None, 'baseline T16 VMS missing'
        for key in ('inter','ybuf','ss','ctr'): vs[key] = torch.zeros_like(vs[key])
        fl = flags + ['-DVMS_G=132','-DVMS_NS=3','-DVMS_RO=1','-Xptxas=-warn-spills']
        assert torch.cuda.get_device_properties(dev).multi_processor_count == 132, 'H200 132-SM build only'
        vs['mod'] = _ast2_load('vms',VMS_CPP_SRC,_ast2_src('VMS_CUDA_SRC',bits),['run','attrs'],fl)
        fl = [f for f in fl if not f.startswith('-DVMS_NS=')] + ['-DVMS_T16=1','-DVMS_WT2=30','-DVMS_NS=2','-DVMS_ER=1']
        vs['mod_hi'] = _ast2_load('vms_hi',VMS_CPP_SRC,_ast2_src('VMS_CUDA_SRC',bits),['run','attrs'],fl)
        st['k2a33'] = dict(state['k2a33'],st=dict(state['k2a33']['st'],vms=vs))
    if bits & 17:
        assert all(_SK2['ok'].get(k) for k in _SK2_CFGS), 'baseline sk2 missing'
        sk = dict(_SK2,ok=dict(_SK2['ok']),w=dict(_SK2['w']),err=None)
        sk['mod'] = _ast2_load('sk2',_SK2_CPP,_ast2_src('_SK2_CUDA',bits),['sk2','sk2_set_pdl'],list(_SK2_FLAGS))
        sk['dummy'] = torch.zeros_like(_SK2['dummy'])
        gs['_SK2'] = sk
        if bits & 1:
            gs.update(_sk2_compress=_ast2_compress, _sk2_table=_ast2_table)
    if bits & 8:
        assert state.get('hya'), 'baseline HYA missing'
        h = dict(_HYA,ws={k:tuple(torch.zeros_like(x) for x in v) for k,v in _HYA['ws'].items()},seen=set())
        h['mod'] = _ast2_load('hya',HYA_CPP_SRC,_ast2_src('HYA_CUDA_SRC',bits),['run'],flags+['--expt-relaxed-constexpr'])
        gs['_HYA'] = h
    if bits & 16:
        assert _VF['st2'] is not None, 'SBB2 missing'
        gs['_VF'] = dict(_VF,st2=dict(_VF['st2'],ok=dict(_VF['st2']['ok'])),seen=set())
        gs['VF_SBB2'] = dict(VF_SBB2)
        gs['VF_SBB2'][32] = True
        gs['_v9_bnd'] = _ast2_bnd
    return bank

def _ast2_equal(ref, rep, got, what):
    assert bool(torch.isfinite(got.float()).all()), (what,'nonfinite')
    if torch.equal(ref,rep):
        assert torch.equal(ref,got), (what,'repeat-stable baseline differs')
    else:
        den = ref.float().norm().clamp_min(1e-30)
        err = float((got.float()-ref.float()).norm()/den)
        noise = float((rep.float()-ref.float()).norm()/den)
        assert err <= max(2e-3, 2*noise), (what,'exceeds stock repeat',err,noise)

def _ast2_moe_stress(bank, state, bits, dev):
    # Capture distinct shapes sharing one engine scratch; replay alternating M,
    # including zero-local routes and changing x/ids/wts on every invocation.
    for family, ts in [('hyblo',(1,8)), ('vms',(1,8,16))]:
        if family == 'vms' and not bits & 2: continue
        graphs = {}
        for t in ts:
            d = _hyblo_inputs(t,815+t,dev,'normal')
            x,ids,tw = (d[n].clone() for n in ('hidden_states','topk_ids','topk_weights'))
            pairs = []
            for candidate in (False,True):
                old = _ast2_activate(bank) if candidate else {}
                engine_state = bank['state'] if candidate else state
                try:
                    def call():
                        if family == 'hyblo': return _hyblo_run(_hyblo_key(t),x,ids,tw)
                        return _vms_call(engine_state['k2a33']['st'],x,ids,tw)
                    call()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): out = call()
                    pairs.append((g,out))
                finally: globals().update(old)
            graphs[t] = (x,ids,tw,pairs)
        for i in range(200):
            t = ts[i % len(ts)]
            d = _hyblo_inputs(t,1400+i,dev,('normal','none','skewed','conc')[(i//len(ts))%4])
            x,ids,tw,pairs = graphs[t]
            x.copy_(d['hidden_states']); ids.copy_(d['topk_ids']); tw.copy_(d['topk_weights'])
            (ag,ao),(bg,bo) = pairs
            ag.replay(); ref = ao.clone()
            ag.replay(); rep = ao.clone()
            bg.replay(); got = bo.clone()
            torch.cuda.synchronize()
            _ast2_equal(ref,rep,got,family)
            if family == 'vms': assert int(bank['state']['k2a33']['st']['vms']['ctr'][2]) == 0, 'VMS sticky error'
        if family == 'hyblo':
            assert all(bool((f[n:].view(vt)==sv).all()) for f,n,vt,sv in bank['globals']['_HYBLO']['full']), 'HYBLO guards'

def _ast2_gates(bank, state, bits, dev):
    old = _ast2_activate(bank)
    try:
        if bits & 1:
            _sk2_prepare(bank['state'],time.time())
            assert all(_SK2['ok'].get(k) for k in _SK2_CFGS), 'SK2ESC pack/numerics fallback'
        if bits & 8:
            _hya_self_test(dev)
        if bits & 16:
            st = _VF['st2']; pw,qw,al,bt,isc,pe,qe = st['w']
            gen = torch.Generator(device=dev).manual_seed(19541)
            for i in range(4):
                stream = torch.randn((32,HIDDEN),generator=gen,device=dev)
                br = torch.randn((32,HIDDEN),generator=gen,device=dev).to(torch.bfloat16)
                if i == 0: br.zero_()
                ref = state['ffn_boundary'](stream,br,branch_dtype=torch.bfloat16,router_dtype=None)
                out,ns = torch.empty_like(br),torch.empty_like(stream)
                st['mod'].sbb1_run(br,stream,out,ns,st['e'],pw,qw,al,bt,pe,qe,isc|2)
                torch.cuda.synchronize()
                assert bool(torch.isfinite(ns).all()) and bool(torch.isfinite(out.float()).all())
                assert float((out.float()-ref[0].float()).norm()/ref[0].float().norm().clamp_min(1e-30)) < 1e-2
                assert float((ns-ref[1]).norm()/ref[1].norm().clamp_min(1e-30)) < 1e-4
            st['ok'][32] = True
    finally: globals().update(old)
    if bits & 6: _ast2_moe_stress(bank,state,bits,dev)

def prepare(weights):
    state = _ast2_prepare0(weights)
    dev = weights['qkv_gate'].device
    _AST2_BANK.clear(); AST2_LIVE.clear(); AST2_LIVE[0] = True
    AST2_ERRORS.clear(); AST2_BUILD_S.clear()
    for bits in AST2_KEYS:
        assert isinstance(bits,int) and 0 < bits <= 31, ('bad AST2 subset',bits)
        AST2_LIVE[bits] = False
        t0 = time.time()
        try:
            bank = _ast2_bank(state,bits,dev)
            _ast2_gates(bank,state,bits,dev)
            _AST2_BANK[bits] = bank
            AST2_LIVE[bits] = True
        except Exception as ex: AST2_ERRORS[bits] = repr(ex)[-1600:]
        AST2_BUILD_S[bits] = time.time()-t0
        print('[ast2] arm',bits,'live',AST2_LIVE[bits],'seconds',AST2_BUILD_S[bits],'error',AST2_ERRORS.get(bits),flush=True)
    return state

def kernel_fn(data,state):
    if AST2_BITS == 0: return _ast2_kernel0(data,state)
    assert AST2_LIVE.get(AST2_BITS), ('AST2 arm unavailable',AST2_BITS,AST2_ERRORS)
    bank = _AST2_BANK[AST2_BITS]
    old = _ast2_activate(bank)
    try:
        _AST2_HITS.clear()
        out = _ast2_kernel0(data,bank['state'])
        bits = AST2_BITS; cfg = data['config']; t = cfg['tokens']
        need = set()
        if t == 32 and bits & 17: need.update(('qkv','unpooler'))
        if t in (1,8) and bits & 6: need.add('hyblo')
        if t == 16 and bits & 2: need.add('vms')
        if bits & 8 and cfg['window'] == 0 and (cfg['batch'],cfg['q_len']) in ((16,1),(32,1),(1,8)): need.add('hya')
        if t == 32 and bits & 16: need.add('sbb')
        assert need <= _AST2_HITS, ('AST2 fallback is not a timing arm',need,_AST2_HITS)
        return out
    finally: globals().update(old)
'''


def function_source(s, name):
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == name]
    check(len(nodes) == 1, 'function source ' + name)
    return ast.get_source_segment(s, nodes[0]) + '\n'


def apply_lab(s, base_name='k3_hh_5_sk2_pz'):
    own = pathlib.Path(__file__).read_text(encoding='ascii')
    names = ('check','sub','scoped','sk2esc','retire_hyb','retire_vms','route1','hyaq','sk2trig')
    helpers = '\n'.join(function_source(own,n) for n in names)
    changed = apply_ship(s,17)
    for name, new in (('_sk2_compress','_ast2_compress'),('_sk2_table','_ast2_table'),('_v9_bnd','_ast2_bnd')):
        helpers += '\n' + function_source(changed,name).replace('def '+name+'(', 'def '+new+'(',1)
    edits = {1:{'_SK2_CUDA':'sk2esc'},2:{'_HYBLO_CUDA':'retire_hyb','VMS_CUDA_SRC':'retire_vms'},
             4:{'_HYBLO_CUDA':'route1'},8:{'HYA_CUDA_SRC':'hyaq'},16:{'_SK2_CUDA':'sk2trig'}}
    nonce = s.splitlines()[1]
    s = sub(s,nonce+'\n',f'# nonce AST2 lab on {base_name}; AST2_BITS=0..31\n')
    return s + '\n' + helpers + f'\n_AST2_CUDA_EDITS = {edits!r}\n' + LAB


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ship',action='store_true')
    p.add_argument('--bits',type=lambda x:int(x,0),default=31)
    p.add_argument('--base',type=pathlib.Path,default=BASE)
    p.add_argument('--out',type=pathlib.Path)
    a = p.parse_args()
    effective_bits(a.bits)
    check(not a.ship or a.out is not None,'--ship requires --out')
    out = a.out or ROOT / 'kernels/lab/k3_ast2_ab.py'
    check(out.resolve() != a.base.resolve(),'output must differ from base')
    s = a.base.read_text(encoding='ascii')
    new = apply_ship(s,a.bits) if a.ship else apply_lab(s,a.base.stem)
    n = gates(new,a.ship)
    out.write_bytes(new.encode('ascii'))
    print(f'[ast2] {out}: {n:,} bytes ({n-len(s):+,}); requested={a.bits} effective={effective_bits(a.bits)}')


if __name__ == '__main__':
    main()

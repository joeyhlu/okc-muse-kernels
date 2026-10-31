"""MG: independent riders on the final decode layer, forked from AST1.

python -B -I tools/mg_patches.py --ship --bits=88 --base=... --out=...
python -B -I tools/mg_patches.py                       # lab vehicle

8 KSMERGE is non-bitwise. 16 HBINIT and 64 RTR2 preserve arithmetic.
The two AST1 edit functions are verbatim, including their provenance comments.
No layer imports, CUDA calls or writes at import time. GPU acceptance is pending.
"""
import argparse
import ast
import functools
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
LIMIT = 1_000_000
BITS = {8: "KSMERGE", 16: "HBINIT", 64: "RTR2", 128: "RTR3"}
MG_KEYS = (8, 16, 64, 72, 88, 128)
SHAPES = {8: (0, 6, 2), 16: (0, 1, 6, 7), 64: (6, 7), 72: (0, 1, 2, 6, 7), 88: (0, 1, 2, 6, 7), 128: (6, 7)}


def effective_bits(bits):
    check(type(bits) is int and bits >= 0 and bits & ~216 == 0 and bits & 192 != 192, "unsupported --bits; allowed flags: 8,16,64|128 (RTR2 xor RTR3)")
    return bits


def check(ok, message):
    if not ok:
        raise AssertionError("[mg] " + message)


def sub(text, old, new, what="anchor"):
    n = text.count(old)
    check(n == 1, f"{what}: found {n}, expected exactly one: {old[:100]!r}")
    return text.replace(old, new, 1)


def scoped(text, start, end, fn):
    check(text.count(start) == text.count(end) == 1, "scope " + start[:70])
    a, b = text.index(start), text.index(end)
    check(a < b, "scope order")
    old = text[a:b]
    return sub(text, old, fn(old), start[:60])


@functools.lru_cache(maxsize=4)
def assignments(text):
    out = {}
    for n in ast.parse(text).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out.setdefault(n.targets[0].id, []).append(n)
    return out


def assignment(text, name):
    nodes = assignments(text).get(name, [])
    check(len(nodes) == 1, "assignment " + name)
    return nodes[0]


def literal(text, name):
    return ast.literal_eval(assignment(text, name).value)


def cuda_edit(text, name, fn):
    node = assignment(text, name)
    old = ast.get_source_segment(text, node)
    m = re.match(r"\w+ = r([\"']{3})", old)
    check(m is not None, "raw CUDA literal " + name)
    cu = old[m.end():-3]
    check(cu == ast.literal_eval(node.value), "literal contents " + name)
    return sub(text, old, old[:m.end()] + fn(cu) + old[-3:], name)


def hbinit(cu):
    init = """for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
"""
    def edit(s):
        # Separate exact anchors cover both preprocessor branches. Never leave a
        # second initializer that could reset a barrier with an in-flight copy.
        s = sub(s, "if (ln == 0) {\n" + init + "}\n", "")
        s = sub(s, "#if OKC_SPLIT || !OKC_F1EARLY\n" + init + "#endif\n", "")
        before = """// AST1 HBINIT: shipped skinny WS only. bars[0..NS), wsrf/wsre[0..NRB).
// No wse user with WSB=1; early_issue returns before its dead bar accesses.
static_assert(OKC_TINY && OKC_WS && OKC_WSB && !OKC_WGS && OKC_NS == 3 && NRB == 4
              && OKC_NW == 8 && !OKC_SPLIT && OKC_FAST1 <= OKC_NS && OKC_WSBLK <= NRB,
              "AST1 HBINIT requires the audited skinny specialization");
if (tid == NT - 32) {
for (int k = 0; k < OKC_NS; ++k) mbar_init(S.bars + k, 1);
for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, 1); mbar_init(S.wsre + k, 1); }
asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
}
// Initializer remains the first-copy issuer; existing route CTA_SYNC and
// WS_JOIN publish initialization to consumers/helper. No input/copy before wait.
"""
        return sub(s, "DBG(0);\nasm volatile(\"griddepcontrol.wait;\\n\" ::: \"memory\");", "DBG(0);\n" + before + 'asm volatile("griddepcontrol.wait;\\n" ::: "memory");')
    return scoped(cu, "template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused", 'static int g_okc_pdl', edit)


def ksmerge(cu):
    def edit(s, ks3):
        mark = "KS3P(4);" if ks3 else "KS1P(5);"
        test = "(m == neg_inf() || (m >= -32.f && m <= 32.f))" if ks3 else "((m[0] == neg_inf() || (m[0] >= -32.f && m[0] <= 32.f)) && (m[1] == neg_inf() || (m[1] >= -32.f && m[1] <= 32.f)))"
        mg = "MG == 1" if ks3 else "(MG & 3) == 1"
        # FP8 |V| <=448, <=2048 keys, relative partials <=2^20. These bounds
        # leave ample headroom at [-32,32]. Invalid exponents use the old frame.
        scale = """const float f = m == neg_inf() ? 0.f : ex2(m);
l *= f;
#pragma unroll
for (int n = 0; n < 8; ++n) { o[n][0] *= f; o[n][1] *= f; }
""" if ks3 else """const float f0 = m[0] == neg_inf() ? 0.f : ex2(m[0]), f1 = m[1] == neg_inf() ? 0.f : ex2(m[1]);
ls[0] *= f0; ls[1] *= f1;
#pragma unroll
for (int n = 0; n < 8; ++n) { o[n][0] *= f0; o[n][1] *= f0; o[n][2] *= f1; o[n][3] *= f1; }
"""
        s = sub(s, mark + "\n__syncthreads();\nfloat* ow", mark + "\n" + """// AST1 KSMERGE: one CTA-wide vote replaces the existing barrier. Every
// warp uses the same frame; empty (-inf) warps contribute zero. FP8 partials
// are bounded by 448*2048; outside [-32,32] retain the original clamp/frame.
bool absframe = false;
if constexpr (""" + mg + ") {\nabsframe = __syncthreads_and(" + test + ");\nif (absframe) {\n" + scale + "}\n} else __syncthreads();\nfloat* ow")
        if ks3:
            a = "for (int k = 0; k < 4; ++k) M = fmaxf(M, mw[(2 * k + rh) * 8 + rg]);"
            s = sub(s, a, "if (!absframe) {\n#pragma unroll\n" + a + "\n}")
            s = sub(s, "const float f = ex2(mw[w * 8 + rg] - Mu);", "const float f = absframe ? 1.f : ex2(mw[w * 8 + rg] - Mu);")
            for dst, x in (("lt", "lw[w * 8 + rg]"), ("acc.x", "xv.x"), ("acc.y", "xv.y"), ("acc.z", "xv.z"), ("acc.w", "xv.w")):
                s = sub(s, f"{dst} = fmaf(f, {x}, {dst});", f"{dst} = absframe ? {dst} + {x} : fmaf(f, {x}, {dst});")
        else:
            a = "for (int w = 0; w < NWARP; ++w) M = fmaxf(M, mw[w * 16 + mr]);"
            s = sub(s, a, "if (!absframe) {\n#pragma unroll\n" + a + "\n}")
            s = sub(s, "const float f = ex2(mw[w * 16 + mr] - Mu);", "const float f = absframe ? 1.f : ex2(mw[w * 16 + mr] - Mu);")
            s = sub(s, "lt = fmaf(f, lw[w * 16 + mr], lt);", "lt = absframe ? lt + lw[w * 16 + mr] : fmaf(f, lw[w * 16 + mr], lt);")
            # Whole if/else anchor keeps the repeated acc[0]/acc[1] edits exact.
            old = """if constexpr (DPT == 4) {
const float4 x = *reinterpret_cast<const float4*>(src);
acc[0] = fmaf(f, x.x, acc[0]); acc[1] = fmaf(f, x.y, acc[1]); acc[DPT - 2] = fmaf(f, x.z, acc[DPT - 2]);
acc[DPT - 1] = fmaf(f, x.w, acc[DPT - 1]);
} else {
const float2 x = *reinterpret_cast<const float2*>(src);
acc[0] = fmaf(f, x.x, acc[0]); acc[1] = fmaf(f, x.y, acc[1]);
}"""
            new = re.sub(r"(acc\[[^\]]+\]) = fmaf\(f, (x\.[xyzw]), \1\);", r"\1 = absframe ? \1 + \2 : fmaf(f, \2, \1);", old)
            s = sub(s, old, new)
        s = sub(s, "const float f = (M == neg_inf()) ? 0.f : ex2(fminf(M, 126.f));", "const float f = absframe ? 1.f : (M == neg_inf()) ? 0.f : ex2(fminf(M, 126.f));")
        # Remove pragma immediately before the new if: nvcc requires a loop.
        s = sub(s, "#pragma unroll\nif (!absframe)", "if (!absframe)")
        return s
    cu = scoped(cu, "KS1P(5);", "KS1P(9);\nreturn;", lambda s: edit(s, False))
    return scoped(cu, "KS3P(4);", "KS3P(7);\nreturn;", lambda s: edit(s, True))


def diff_anchor(a, b):
    """One contiguous replacement per source/bit; exact once, with stable context."""
    al, bl = a.splitlines(keepends=True), b.splitlines(keepends=True)
    i = 0
    while i < min(len(al),len(bl)) and al[i] == bl[i]:
        i += 1
    j = 0
    while j < min(len(al),len(bl))-i and al[-1-j] == bl[-1-j]:
        j += 1
    i = max(0,i-2); j = max(0,j-2)
    old, new = ''.join(al[i:len(al)-j]), ''.join(bl[i:len(bl)-j])
    while a.count(old) != 1:
        check(i > 0 or j > 0, "unique diff anchor")
        i = max(0,i-1); j = max(0,j-1)
        old, new = ''.join(al[i:len(al)-j]), ''.join(bl[i:len(bl)-j])
    check(sub(a,old,new) == b, "diff reconstruction")
    return old,new


RTR2_START = "    if constexpr (VF == 2) {   // router32_kernel<8>'s arithmetic for row vf_bx * 8 + warp, K slice vf_sl\n"
RTR2_END = "    if constexpr (VF >= 1) {\n      if (A.marks) {"
RTR2_OLD = r'''if (warp < 8) {
const float4* wsm = gd_vf_dsm + warp * 256;
const float4* xs = gd_vf_dsm + 8 * 256;
float racc[8];
#pragma unroll
for (int j = 0; j < 8; ++j) racc[j] = 0.f;
#pragma unroll
for (int i = 0; i < 8; ++i) {
const int c = i * 32 + lane;
const float4 wv = wsm[c];
#pragma unroll
for (int j = 0; j < 8; ++j) {
const float4 xv = xs[j * 256 + c];
racc[j] = fmaf(wv.x, xv.x, racc[j]);
racc[j] = fmaf(wv.y, xv.y, racc[j]);
racc[j] = fmaf(wv.z, xv.z, racc[j]);
racc[j] = fmaf(wv.w, xv.w, racc[j]);
}
}
#pragma unroll
for (int j = 0; j < 8; ++j) {
float a = racc[j];
#pragma unroll
for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
if (lane == 0 && j < A.M) A.part[((long)vf_sl * A.M + j) * 256 + vf_bx * 8 + warp] = a;
}
}
'''
RTR2_NEW = r'''if (warp < 8) {
// MG RTR2: two rows x four tokens; each output retains its ordered FP32 path.
const int row_pair = warp >> 1, token_half = warp & 1;
const float4* wsm0 = gd_vf_dsm + (2 * row_pair) * 256;
const float4* wsm1 = wsm0 + 256;
const float4* xs = gd_vf_dsm + 8 * 256 + token_half * 4 * 256;
float racc0[4], racc1[4];
#pragma unroll
for (int j = 0; j < 4; ++j) { racc0[j] = 0.f; racc1[j] = 0.f; }
#pragma unroll
for (int i = 0; i < 8; ++i) {
const int c = i * 32 + lane;
const float4 wv0 = wsm0[c], wv1 = wsm1[c];
#pragma unroll
for (int j = 0; j < 4; ++j) {
const float4 xv = xs[j * 256 + c];
racc0[j] = fmaf(wv0.x, xv.x, racc0[j]);
racc0[j] = fmaf(wv0.y, xv.y, racc0[j]);
racc0[j] = fmaf(wv0.z, xv.z, racc0[j]);
racc0[j] = fmaf(wv0.w, xv.w, racc0[j]);
racc1[j] = fmaf(wv1.x, xv.x, racc1[j]);
racc1[j] = fmaf(wv1.y, xv.y, racc1[j]);
racc1[j] = fmaf(wv1.z, xv.z, racc1[j]);
racc1[j] = fmaf(wv1.w, xv.w, racc1[j]);
}
}
#pragma unroll
for (int j = 0; j < 4; ++j) {
float a0 = racc0[j], a1 = racc1[j];
#pragma unroll
for (int o = 16; o > 0; o >>= 1) {
a0 += __shfl_xor_sync(0xffffffffu, a0, o);
a1 += __shfl_xor_sync(0xffffffffu, a1, o);
}
const int token = 4 * token_half + j;
if (lane == 0 && token < A.M) {
A.part[((long)vf_sl * A.M + token) * 256 + vf_bx * 8 + 2 * row_pair] = a0;
A.part[((long)vf_sl * A.M + token) * 256 + vf_bx * 8 + 2 * row_pair + 1] = a1;
}
}
}
'''


RTR3_NEW = r'''if (warp < 4) {
// MG RTR3: four rows x four tokens on warps 0..3 (warps 4..7 idle here); each output retains its ordered FP32 path.
const int row_quad = warp >> 1, token_half = warp & 1;
const float4* wsm0 = gd_vf_dsm + (4 * row_quad) * 256;
const float4* xs = gd_vf_dsm + 8 * 256 + token_half * 4 * 256;
float racc[4][4];
#pragma unroll
for (int r = 0; r < 4; ++r) {
#pragma unroll
for (int j = 0; j < 4; ++j) racc[r][j] = 0.f;
}
#pragma unroll
for (int i = 0; i < 8; ++i) {
const int c = i * 32 + lane;
float4 wv[4];
#pragma unroll
for (int r = 0; r < 4; ++r) wv[r] = wsm0[r * 256 + c];
#pragma unroll
for (int j = 0; j < 4; ++j) {
const float4 xv = xs[j * 256 + c];
#pragma unroll
for (int r = 0; r < 4; ++r) {
racc[r][j] = fmaf(wv[r].x, xv.x, racc[r][j]);
racc[r][j] = fmaf(wv[r].y, xv.y, racc[r][j]);
racc[r][j] = fmaf(wv[r].z, xv.z, racc[r][j]);
racc[r][j] = fmaf(wv[r].w, xv.w, racc[r][j]);
}
}
}
#pragma unroll
for (int j = 0; j < 4; ++j) {
const int token = 4 * token_half + j;
#pragma unroll
for (int r = 0; r < 4; ++r) {
float a = racc[r][j];
#pragma unroll
for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
if (lane == 0 && token < A.M) A.part[((long)vf_sl * A.M + token) * 256 + vf_bx * 8 + 4 * row_quad + r] = a;
}
}
}
'''


def rtr3(cu):
    return scoped(cu, RTR2_START, RTR2_END,
                  lambda s: sub(s, RTR2_OLD, RTR3_NEW, "RTR3 arithmetic"))


def rtr2(cu):
    # Only the interior of the suffix VF==2 block changes. In particular its
    # async drain/CTA barrier and the following VF>=1 marks block are untouched.
    return scoped(cu, RTR2_START, RTR2_END,
                  lambda s: sub(s, RTR2_OLD, RTR2_NEW, "RTR2 arithmetic"))


def apply_ship(s, bits):
    bits = effective_bits(bits)
    for bit, name, edit in ((8, "KS1_CUDA_SRC", ksmerge), (16, "_HYBLO_CUDA", hbinit),
                            (64, "GD_CUDA_SRC", rtr2), (128, "GD_CUDA_SRC", rtr3)):
        if bits & bit:
            s = cuda_edit(s, name, edit)
    if bits:
        nonce = s.splitlines()[1]
        check(nonce.startswith("# nonce "), "nonce")
        s = sub(s, nonce + "\n", nonce + f" + MG bits={bits}\n", "nonce")
    return s


def gd_keys(text):
    """Execute only the base's pure key builders and their scalar/table inputs."""
    constants = ("HIDDEN", "N_EXPERTS", "D_EXPERT", "SLICED_UNPOOL_MAX_TOKENS", "UNPOOL_SLICE",
                 "GD_SMS", "GD_ROUTE", "GD_DIMS", "GD_CAT_ROUTE", "GD_CAT_UNITS")
    names = ("_gd_nk", "_gd_key", "_gd_keys", "_gd_cat_key", "_gd_cat_keys")
    nodes = [assignment(text, n) for n in constants]
    tree = ast.parse(text)
    for name in names:
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        check(len(found) == 1, "GD key function " + name)
        nodes.extend(found)
    env = {}
    exec(compile(ast.Module(nodes, []), "<mg-gd-keys>", "exec"), env)
    return env["_gd_keys"]()


def gd_expand(cu, keys):
    # @INST@ occurs twice, so anchor each host entry point separately.
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    for start, end in (("int gd_run_raw(", "int gdc_run_raw("),
                       ("int gdc_run_raw(", "int gd_unpack_cat_raw(")):
        cu = scoped(cu, start, end, lambda s: sub(s, "@INST@", inst))
    return sub(cu, "@INFO@", info)


def compile_jobs(base_text, bits):
    """Exact CUDA translation units, without importing the layer or CUDA."""
    s = apply_ship(base_text, bits)
    flags = ["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"]
    jobs = {}
    if bits & 16 or bits == 0:
        for k, _, fl in literal(s, "_HYBLO_KEYS"):
            jobs["hyblo_" + k] = (literal(s, "_HYBLO_CUDA"), list(literal(s, "_HYBLO_FLAGS")) + list(fl))
    if bits & 8 or bits == 0:
        cu = literal(s, "KS1_CUDA_SRC")
        for anchor, name, macro in (("@INSTS@", "KS1_INSTS", "KS1_X"), ("@MINSTS@", "VSL_KS1_INSTS", "KS1_XM"),
                                    ("@M3INSTS@", "VSL_KS3_INSTS", "KS3_XM")):
            cu = sub(cu, anchor, " ".join(macro + "(" + ", ".join(map(str, row)) + ")" for row in literal(s, name)), anchor)
        jobs["ks"] = (cu, flags)
    if bits & 192 or bits == 0:
        jobs["gd"] = (gd_expand(literal(s, "GD_CUDA_SRC"), gd_keys(s)), [f for f in flags if f != "-lineinfo"])
    return jobs


def gates(s, ship=True):
    ast.parse(s)
    compile(s, "<mg>", "exec")
    check(s.isascii() and "\r" not in s, "ASCII LF output")
    n = len(s.encode("ascii"))
    check(not ship or n < LIMIT, f"ship size {n} >= {LIMIT}")
    if ship:
        check("MG_BITS" not in s and "_MG_BANK" not in s and "_mg_load" not in s, "lab code leaked into ship")
        check("importlib" not in s and "sys.modules" not in s, "dynamic module creation in ship")
    return n


LAB = r'''
# MG lab-only capture-time bank. Build/gate failure raises; no stock timing fallback.
MG_BITS = 0
MG_KEYS = (8, 16, 64, 72, 88, 128)
MG_LIVE = {0: True}
MG_ERRORS = {}
MG_BUILD_S = {}
_MG_BANK = {}
_MG_MODULES = {}
_MG_GATE_OK = set()
_MG_HITS = set()
_mg_original_prepare = prepare
_mg_original_kernel = kernel_fn
_mg_original_hroute = _hyblo_route
_mg_original_ksrun = _ks1_run
_mg_original_gdr = _vfr_gdr

def _hyblo_route(*args, **kwargs):
    out = _mg_original_hroute(*args, **kwargs)
    if out is not None:
        _MG_HITS.add('hyblo')
    return out

def _ks1_run(qkv, data, state):
    variant = _vsl_pick(data['config'], state)
    out = _mg_original_ksrun(qkv, data, state)
    _MG_HITS.add('mg1' if variant is not None and variant[-1] == 1 else 'stableks')
    return out   # R-mg-6: the fork dropped this return, KS1 'raised' (None.view) and every lab run fell back to HYA at the sliding shapes

def _vfr_gdr(state, x, z, vf):
    r = _gd_live(state, 'pooler', x.shape[0])
    if MG_BITS & 192 and vf == 2:
        assert r is not None and r['mod'] is _GD['mod'], 'RTR2 route still references stock'
        assert _GD['mod'] is _MG_BANK[MG_BITS]['globals']['_GD']['mod'], 'RTR2 wrong active module'
    out = _mg_original_gdr(state, x, z, vf)
    if out is not None and vf == 2 and x.shape[0] == 8:
        _MG_HITS.add('rtr2')
    return out

def _mg_src(name, bits):
    src = globals()[name]
    changes = _MG_EDITS[bits]
    if name in changes:
        old, new = changes[name]
        assert src.count(old) == 1, ('MG source anchor', bits, name)
        src = src.replace(old, new, 1)
    return src

def _mg_gd_expand(cu, keys):
    inst = '\n'.join('    GD_INST(%d, %s)' % (i, ', '.join(str(v) for v in k)) for i, k in enumerate(keys))
    info = '\n'.join('    GD_INFO(%d, %s)' % (i, ', '.join(str(v) for v in k)) for i, k in enumerate(keys))
    for start, end in (('int gd_run_raw(', 'int gdc_run_raw('), ('int gdc_run_raw(', 'int gd_unpack_cat_raw(')):
        assert cu.count(start) == cu.count(end) == 1
        a, b = cu.index(start), cu.index(end)
        assert a < b
        old = cu[a:b]
        assert old.count('@INST@') == 1 and cu.count(old) == 1
        cu = cu.replace(old, old.replace('@INST@', inst, 1), 1)
    assert cu.count('@INFO@') == 1
    return cu.replace('@INFO@', info, 1)

def _mg_load(key, cpp, cu, fns, flags):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((cpp + cu + repr(flags)).encode()).hexdigest()[:16]
    if tag not in _MG_MODULES:
        _MG_MODULES[tag] = load_inline(name='mg_' + key + '_' + tag, cpp_sources=cpp, cuda_sources=cu,
                                       functions=fns, extra_cuda_cflags=flags, verbose=False)
    return _MG_MODULES[tag], tag

def _mg_activate(bank):
    old = {k: globals()[k] for k in bank['globals']}
    globals().update(bank['globals'])
    return old

def _mg_prerequisites(state, bits):
    if bits & 16:
        assert HYBLO_ENABLE and HYBLO_K != 'w' and HYBLO_T == (1, 8), 'HYBLO route changed'
        assert all(_HYBLO['ok'].get(k) for k in ('tiny', 'tiny2')), 'baseline HYBLO not live'
    if bits & 8:
        assert all(tuple(v) in state.get('vsl', ()) for v in VSL_ROUTE.values()), 'MG1 route not live'
    if bits & 192:
        assert GD_ENABLE and VF_GDR and _VFR['ok'], 'baseline VF=2 route not live'
        assert _GD['mod'] is not None and _GD['keys'] == _gd_keys() == _MG_GD_KEYS, 'GD instantiations changed'
        r = _gd_live(state, 'pooler', 8)
        assert r is not None and r['mod'] is _GD['mod'] and r['nbx'] == 256, 'baseline GD pooler unavailable'
        assert GD_ROUTE['pooler', 8] == (1, 1, 8, 16, 2, 1, 1), 'VF=2 pooler geometry changed'

def _mg_bank(state, bits, dev):
    _mg_prerequisites(state, bits)
    flags = ['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a']
    bank = {'state': dict(state), 'globals': {}, 'tags': {}}
    st, gs, tags = bank['state'], bank['globals'], bank['tags']
    if bits & 16:
        h = dict(_HYBLO, mods={}, pdl={}, calls={}, err={})
        h['sc'], h['full'] = _hyb_scratch(dev)
        tags['hyblo'] = []
        for k, _, fl in _HYBLO_KEYS:
            h['mods'][k], tag = _mg_load('hyblo_' + k, _HYB_MOE_CPP, _mg_src('_HYBLO_CUDA', bits),
                                          ['okc_run', 'okc_set_pdl'], list(_HYBLO_FLAGS) + list(fl))
            tags['hyblo'].append(tag)
        tags['hyblo'] = tuple(tags['hyblo'])
        gs['_HYBLO'] = h
    if bits & 8:
        cu = _mg_src('KS1_CUDA_SRC', bits)
        for a, macro, rows in (('@INSTS@', 'KS1_X', KS1_INSTS), ('@MINSTS@', 'KS1_XM', VSL_KS1_INSTS),
                               ('@M3INSTS@', 'KS3_XM', VSL_KS3_INSTS)):
            assert cu.count(a) == 1
            cu = cu.replace(a, ' '.join(macro + '(' + ', '.join(map(str, r)) + ')' for r in rows), 1)
        md, tags['ks'] = _mg_load('ks', KS1_CPP_SRC, cu, ['ks1_forward', 'ks1_info', 'ks3_forward', 'ks3_info'], flags)
        gs['_KS1'] = dict(_KS1, mod=md, cnt={str(dev): torch.zeros(1024, dtype=torch.int32, device=dev)})
        gs['_VSL'] = {'ws': {str(dev): (torch.zeros(VSL_WS_FLOATS, dtype=torch.float32, device=dev),
                                      torch.zeros(VSL_WS_FLOATS // 8, dtype=torch.float32, device=dev))}, 'seen': set()}
    if bits & 192:
        cu = _mg_gd_expand(_mg_src('GD_CUDA_SRC', bits), _GD['keys'])
        md, tags['gd'] = _mg_load('gd', GD_CPP_SRC, cu,
            ['gd_run', 'gd_info', 'gd_unpack', 'gdr_run', 'gdr_info', 'gdc_run', 'gd_unpack_cat'],
            [f for f in flags if f != '-lineinfo'])
        gs['_GD'] = dict(_GD, mod=md)
        # This base caches module handles in route records. Rebind those too;
        # weights/escape tensors are immutable and shared, never repacked here.
        st['gd'] = dict(state['gd'], mod=md,
                        route={k: dict(r, mod=md) for k, r in state['gd']['route'].items()})
        if state.get('gd_cat') is not None:
            st['gd_cat'] = {k: dict(r, mod=md) for k, r in state['gd_cat'].items()}
        gs['_VFR'] = dict(_VFR, seen=set(), marks=torch.zeros_like(_VFR['marks']))
    return bank

def _mg_gd_gate(bank, stock, dev):
    candidate = _GD['mod']
    # GD info layout differs from KS: [registers, local bytes, resident CTAs, static smem].
    infos = [('gd_info', i) for i in range(len(_GD['keys']))] + [('gdr_info', 1), ('gdr_info', 2)]
    for name, inst in infos:
        a, b = list(getattr(stock, name)(inst)), list(getattr(candidate, name)(inst))
        print('[mg] GD resource', name, inst, 'stock', a, 'candidate', b, flush=True)
        assert len(a) == len(b) == 4
        assert (b[0] <= a[0] or (b[0] <= 128 and inst in (3, 4, 2) and name != 'x')) and b[1] <= a[1] and b[2] >= a[2] and b[3] <= a[3], (
            'RTR2 resource regression: regs/local/CTAs/smem', name, inst, a, b)
        if (name, inst) == ('gdr_info', 2):
            assert b[2] >= 2 and b[1] <= 64, 'RTR2 fails baseline VF=2 admission'
    st = bank['state']
    r = _gd_live(st, 'pooler', 8)
    assert r['mod'] is candidate
    gen = torch.Generator(device=dev).manual_seed(6408282)
    for t in (8, 2, 5):
        for rep in range(3):
            z = torch.randn((t, HIDDEN), generator=gen, device=dev, dtype=torch.float32).contiguous()
            x = torch.randn((t, HIDDEN), generator=gen, device=dev).to(torch.bfloat16).contiguous()
            reference = torch.full((8, t, N_EXPERTS), float('nan'), device=dev)
            st['v8']['mod'].v8_router32(z, st['router'], reference)
            # Exercise the extra marks barrier as well as the production no-marks path,
            # and both non-PDL and PDL admission. No inputs are read before the wait.
            for marked, pdl in ((False, 0), (True, 0), (False, 3), (True, 3)):
                outputs = []
                for mod in (stock, candidate):
                    y = torch.full((t, r['N']), float('nan'), dtype=torch.bfloat16, device=dev)
                    part = torch.full_like(reference, float('nan'))
                    marks = torch.full((4*r['nbx'] if marked else 0,), -1, dtype=torch.int64, device=dev)
                    mod.gdr_run(2, r['w'], r['esc'], x, y, t, r['K'], r['N'], r['tb'], r['nbx'], pdl,
                                z, st['router'], part, marks)
                    outputs.append((y, part, marks))
                torch.cuda.synchronize()
                (ya, pa, ma), (yb, pb, mb) = outputs
                assert torch.equal(ya, yb) and torch.equal(ya.view(torch.int16), yb.view(torch.int16)), 'RTR2 pooler bits differ'
                assert torch.equal(pa, pb) and torch.equal(pa.view(torch.int32), pb.view(torch.int32)), 'RTR2 A.part bits differ'
                assert torch.equal(reference.view(torch.int32), pb.view(torch.int32)), 'RTR2 differs from router32'
                assert bool(torch.isfinite(yb).all()) and bool(torch.isfinite(pb).all()), 'RTR2 unwritten/nonfinite output'
                if marked:
                    for marks in (ma, mb):
                        rows = marks.view(-1, 4)
                        assert bool((rows >= 0).all()) and bool((rows[:, 3] >= rows[:, 2]).all()), 'RTR2 marks incomplete'
        print('[mg] RTR2 bitwise gate PASS T=', t, '(3 random inputs, marks off/on, PDL off/on)', flush=True)

def _mg_component_gates(bank, bits, dev):
    global _HYBLO, KS3_SELF_TEST_CASES
    tags = bank['tags']
    old = _mg_activate(bank)
    try:
        if 'hyblo' in tags and tags['hyblo'] not in _MG_GATE_OK:
            candidate, stock = _HYBLO, old['_HYBLO']
            # Both generations, 200 alternating T1/T8 calls, empty routes and
            # changing inputs. Keep compute-owned terminal atomics untouched.
            cases = [_hyblo_inputs(t, 810+i, dev, dist) for i, (t,dist) in enumerate(
                ((1,'normal'), (8,'normal'), (1,'none'), (8,'none'), (1,'skewed'), (8,'conc')))]
            try:
                graphs = {}
                for t in (1,8):
                    d = next(d for d in cases if d['hidden_states'].shape[0] == t)
                    x, ids, tw = (d[n].clone() for n in ('hidden_states','topk_ids','topk_weights'))
                    k = _hyblo_key(t)
                    pairs = []
                    for engine in (stock,candidate):
                        _HYBLO = engine
                        _hyblo_run(k,x,ids,tw)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = _hyblo_run(k,x,ids,tw)
                        pairs.append((graph,output))
                    graphs[t] = (x,ids,tw,pairs)
                for i in range(200):
                    d = cases[i % len(cases)]
                    x,ids,tw,pairs = graphs[d['hidden_states'].shape[0]]
                    x.copy_(d['hidden_states']); ids.copy_(d['topk_ids']); tw.copy_(d['topk_weights'])
                    (ag,ao),(bg,bo) = pairs
                    ag.replay(); ref = ao.clone()
                    ag.replay(); rep = ao.clone()
                    bg.replay(); got = bo.clone()
                    torch.cuda.synchronize()
                    # Existing norm atomics are not deterministic in general.
                    # Do not mislabel a noisy stock case as a bitwise proof.
                    if torch.equal(ref, rep):
                        assert torch.equal(ref, got), 'HBINIT bitwise mismatch on repeat-stable stock'
                    else:
                        err = (got.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-30)
                        assert float(err) < 2e-3, 'HBINIT exceeds stock tolerance'
                assert all(bool((f[n:].view(vt) == sv).all()) for f,n,vt,sv in candidate['full']), 'HYBLO guards'
            finally:
                _HYBLO = candidate
            _MG_GATE_OK.add(tags['hyblo'])
        if 'ks' in tags and tags['ks'] not in _MG_GATE_OK:
            saved = KS3_SELF_TEST_CASES
            KS3_SELF_TEST_CASES = tuple(saved) + tuple((3,c,1400+i,True) for i,c in enumerate((1,2047,2048,2049)))
            try:
                # Existing reference/cache/clean-counter/graph gates, augmented
                # with short contexts and cache row zero.
                vsl = _vsl_self_test(bank['state']['cos_sin'], dev)
                assert all(tuple(v) in vsl for v in VSL_ROUTE.values()), 'KS MG1 numerical/cache gate failed'
                # Runtime API provides registers, local bytes and resident CTAs.
                # Compare occupancy and local-memory usage to the stock module.
                for f,args in [('ks1_info',(l,n,g+100*m)) for l,n,g,m in VSL_KS1_INSTS] + [
                               ('ks3_info',(n+100*m,)) for n,m in VSL_KS3_INSTS] + [('ks3_info',(8,))]:
                    a = list(getattr(old['_KS1']['mod'],f)(*args))
                    b = list(getattr(_KS1['mod'],f)(*args))
                    assert b[1] <= a[1] and b[3] >= a[3], ('KS spill/occupancy regression',f,args,a,b)
            finally:
                KS3_SELF_TEST_CASES = saved
            _MG_GATE_OK.add(tags['ks'])
        if 'gd' in tags and tags['gd'] not in _MG_GATE_OK:
            _mg_gd_gate(bank, old['_GD']['mod'], dev)
            _MG_GATE_OK.add(tags['gd'])
    finally:
        globals().update(old)

def prepare(weights):
    state = _mg_original_prepare(weights)
    dev = weights['qkv_gate'].device
    _MG_BANK.clear()
    MG_LIVE.clear(); MG_LIVE[0] = True
    MG_ERRORS.clear(); MG_BUILD_S.clear()
    _MG_GATE_OK.clear()
    keys = tuple(int(x) for x in __import__('os').environ['MG_KEYS'].split(',')) if __import__('os').environ.get('MG_KEYS') else MG_KEYS
    assert keys and all(b in _MG_EDITS for b in keys), 'invalid MG_KEYS'
    for bits in keys:
        MG_LIVE[bits] = False
        t0 = time.time()
        try:
            bank = _mg_bank(state, bits, dev)
            _mg_component_gates(bank, bits, dev)
            _MG_BANK[bits] = bank
            MG_LIVE[bits] = True
        except Exception as ex:
            MG_ERRORS[bits] = repr(ex)[-1200:]
            raise RuntimeError(('MG arm gate failed', bits, MG_ERRORS[bits])) from ex
        finally:
            MG_BUILD_S[bits] = time.time() - t0
            print('[mg] arm', bits, 'live', MG_LIVE[bits], 'seconds', MG_BUILD_S[bits],
                  'error', MG_ERRORS.get(bits), flush=True)
    return state

def kernel_fn(data, state):
    if MG_BITS == 0:
        return _mg_original_kernel(data, state)
    assert MG_LIVE.get(MG_BITS), ('MG arm unavailable', MG_BITS, MG_ERRORS)
    bank = _MG_BANK[MG_BITS]
    old = _mg_activate(bank)
    try:
        _MG_HITS.clear()
        out = _mg_original_kernel(data, bank['state'])
        cfg = data['config']; t = cfg['tokens']
        required = set()
        if MG_BITS & 16 and t in (1, 8): required.add('hyblo')
        if MG_BITS & 192 and t == 8: required.add('rtr2')
        if MG_BITS & 8 and cfg['window'] == 2048 and (cfg['batch'], cfg['q_len']) in ((1,1), (1,8), (16,1)):
            required.add('mg1')
        assert required <= _MG_HITS, ('MG fallback is not a timing arm', required, _MG_HITS)
        return out
    finally:
        globals().update(old)
'''


def lab_edits(s):
    names = ("KS1_CUDA_SRC", "_HYBLO_CUDA", "GD_CUDA_SRC")
    plans = {}
    for bits in MG_KEYS:
        changed = apply_ship(s, bits)
        plans[bits] = {name: diff_anchor(literal(s, name), literal(changed, name)) for name in names
                      if literal(s, name) != literal(changed, name)}
    return plans


def apply_lab(s, base_name="k3_hh_5_sk2_pz"):
    edits, keys = lab_edits(s), gd_keys(s)
    nonce = s.splitlines()[1]
    check(nonce.startswith("# nonce "), "lab nonce")
    s = sub(s, nonce + "\n", f"# nonce MG lab on {base_name}; MG_BITS 8/16/64/88\n", "lab nonce")
    return s + f"\n_MG_EDITS = {edits!r}\n_MG_GD_KEYS = {keys!r}\n" + LAB


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ship", action="store_true")
    p.add_argument("--bits", type=int, default=88)
    p.add_argument("--base", type=pathlib.Path, default=BASE)
    p.add_argument("--out", type=pathlib.Path)
    a = p.parse_args()
    effective_bits(a.bits)
    check(not a.ship or a.out is not None, "--ship requires --out")
    out = a.out or ROOT / "kernels/lab/k3_mg_ab.py"
    check(out.resolve() != a.base.resolve(), "output must differ from base")
    s = a.base.read_text(encoding="ascii")
    new = apply_ship(s, a.bits) if a.ship else apply_lab(s, a.base.stem)
    n = gates(new, a.ship)
    out.write_bytes(new.encode("ascii"))
    print(f"[mg] {out.name}: {n:,} bytes ({n-len(s):+,}); {'ship' if a.ship else 'lab'}")


if __name__ == "__main__":
    main()

"""hybst_patches.py (R-hybst): HYB (alygg77 WGS MoE engine, T 49..320) unit-end / tail riders behind -DOKC_HYBST bits.
All three bits keep HYB's arithmetic exactly (same expressions, same order): outputs are bitwise the base build's.

  python -I tools/hybst_patches.py [--base=kernels/lab/k3_hh_3k.py] [--out=kernels/lab/k3_hybst_1.py] [--bits=7]
                                   [--builds=0,7] [--st=0] [--final] [--check]
  python -I tools/lead_gate.py file kernels/lab/k3_hybst_1.py

Diagnosis: notes/study/hyb_unit_end.md. HYB already stitches its rings (one flat weight-issue loop and one flat x-issue
loop over the CTA's whole A + B stage list, parity carried in `seq`, releases deferred behind commit group 3 of the next
segment), so the ext_muse_moe_field.md s4.1 "stitch" is already in the base. What is still serial at a unit end:
  bit 1 EPF  whole-unit epilogue: wgs_epi_tile re-loads gsA/gsB[e] (+ normw[n0], normw[n0+8] in phase B) and S.jC/jS/jrow
             for EVERY n-tile, each load after the previous tile's asm-volatile "memory" stores (SASS: 1 LDG per tile in A,
             3 per tile in B, serialized by the store chain), and phase B's normw lines are an L1 miss (new 512 B per
             B unit) -> an L2 round trip with every WG stalled. Fix: prefetch.global.L1 of the gs / normw lines at the
             segment start (no register held across the stage loop) + the constants computed ONCE after wg_wait<0>
             (wgs_epi_consts / wgs_epi_tile_c: same expressions).
  bit 2 FIN  the phase-end flush (B: the designated finisher = the CTA's LAST segment, on the CTA tail; A: the last
             segment's count / release): read-back of ntt float4 per thread two at a time (ceil(ntt/2) serial L2 round
             trips + per-tile constant reloads). Fix: wgs_flush_end = wgs_flush with all ntt loads in flight at once and
             the constants loaded alongside (no accumulators are live at the phase end).
  bit 4 FP2  the in-launch final pass: pass 2 re-reads every y row (__ldcg, L2) and topk_w after two CTA barriers. Fix:
             pass 1 keeps its y float4s and the topk_w values in registers for pass 2.
Bit 8 (lab only): -DOKC_DEBUG=1 = alygg77's own per-CTA globaltimer marks (DBG 0..7) into _HYB["dbg"] (marks builds are
compared with their base like any other build). Lab file: HYB modules (HYB_ST_BUILDS, parallel nvcc), the call uses _HYB["mods"][HYB_ST] (an arm sets HYB_ST); each
extra build is checked against the default build's output on every HYB self-test case (dropped on any mismatch beyond the
default build's own repeat spread). --final: HYB_ST = bits, one build.
Edits (each anchor occurs exactly once): nonce; OKC_HYBST default 0; wgs_epi_consts / wgs_epi_tile_c / wgs_flush_end
after wgs_flush; EPF prefetch before the first stage; EPF hoisted whole-unit epilogue; FIN phase-end flush; FP2 final
pass (2 places); HYB notice line; HYB_ST / HYB_ST_BUILDS; _HYB dict; _hyb_build; _hyb_call; self-test cross-check; LIVE print.
--check: no write; --out must equal apply(--base) byte for byte, plus the $0 gates (AST, ASCII, < 1,000,000 B, pyslim
decorated defs re-parse, inserted anchors present).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000


def check(c, msg):
    if not c:
        raise SystemExit(f"[hybst_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


# ---------------------------------------------------------------- CUDA (inside _HYB_MOE_CUDA; raw text, \n stays literal)
A_DEF = "#define OKC_NOLOAD 0\n"
N_DEF = "#define OKC_NOLOAD 0\n#ifndef OKC_HYBST\n#define OKC_HYBST 0\n#endif\n#ifndef OKC_HYBFB\n#define OKC_HYBFB 4\n#endif\n"

A_SEGT = "template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg("
N_SEGT = r'''struct WgsEpi { float c0, c1; int cnt, jst, jrow; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_consts(const Params& p, const Smem& S, int j, int r16, WgsEpi& c) {
const int g = (threadIdx.x & 31) >> 2, e = S.jE[j];
c.cnt = S.jC[j]; c.jst = S.jS[j]; c.jrow = S.jrow[j];
if (!PB) {
c.c0 = p.gsA[e] * TWO62; c.c1 = 0.f;
} else {
const int n0 = r16 * 16 + g;
const float gsc = p.gsB[e] * TWO62;
c.c0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc; c.c1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
}
}
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile_c(const Params& p, const WgsEpi& c, int r16, int nt, const float* d4) {
const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
const int cnt = c.cnt;
if (!PB) {
const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
const float gsc = c.c0;
__nv_bfloat16* xb = p.xB + (size_t)c.jrow * HD + (col & 7);
#pragma unroll
for (int h = 0; h < 2; ++h) {
const int tk = nt * 8 + 2 * t + h;
if (tk < cnt) {
const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;
const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
__float2bfloat16(probe_q(v)));
}
}
} else {
const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
const float w0 = c.c0, w1 = c.c1;
#pragma unroll
for (int h = 0; h < 2; ++h) {
const int tk = tok0 + h;
if (tk < cnt) {
st_keep_f32(p.ybuf + ((size_t)(c.jst + tk) * HD + n0), d4[h] * w0);
st_keep_f32(p.ybuf + ((size_t)(c.jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
}
}
}
}
template <bool PB>
__device__ __forceinline__ void wgs_flush_end(const Params& p, Smem& S, WgsPend& pd) {
if (pd.kind == 0) return;
const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
constexpr int UBL = PB ? 4 : 5;
const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
if (pd.kind == 1) {
if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
pd.kind = 0;
return;
}
int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;
if constexpr (DF) {
if (pd.kind == 2) {
if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
pd.kind = 0;
return;
}
if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
pd.kind = 0;
} else {
if (wt == 0) {
int old;
asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
S.wglast[wg] = old + pd.nst == 16;
}
asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
const bool last = S.wglast[wg];
pd.kind = 0;
if (!last) return;
}
asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
WgsEpi c;
#pragma unroll 1
for (int nt0 = 0; nt0 < ntt; nt0 += OKC_HYBFB) {
float4 v[OKC_HYBFB];
#pragma unroll
for (int k = 0; k < OKC_HYBFB; ++k)
if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
if (nt0 == 0) wgs_epi_consts<PB>(p, S, j, r16, c);
#pragma unroll
for (int k = 0; k < OKC_HYBFB; ++k)
if (nt0 + k < ntt) {
*reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
wgs_epi_tile_c<PB>(p, c, r16, nt0 + k, d4);
}
}
if (wt == 0) *fc = 0;
if (!PB) {
asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
}
}
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg('''

A_FIRST = "stage(std::true_type{});\n"
N_FIRST = r'''#if OKC_HYBST & 1
if (slot < 0) {
const int je = S.jE[u >> UBL];
asm volatile("prefetch.global.L1 [%0];\n" ::"l"((PB ? p.gsB : p.gsA) + je));
if (PB) asm volatile("prefetch.global.L1 [%0];\n" ::"l"(p.normw + ((u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp) * 16 + g));
}
#endif
stage(std::true_type{});
'''

A_EPI = ("const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n#pragma unroll\n"
         "for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);\n")
N_EPI = ("const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n"
         "#if OKC_HYBST & 1\n{\nWgsEpi ec;\nwgs_epi_consts<PB>(p, S, j, r16, ec);\n#pragma unroll\n"
         "for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile_c<PB>(p, ec, r16, nt, d[nt]);\n}\n#else\n#pragma unroll\n"
         "for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);\n#endif\n")

A_FIN = "gs += s1 - s0;\n}\nwgs_flush<PB>(p, S, pd);\n}\n"
N_FIN = ("gs += s1 - s0;\n}\n#if OKC_HYBST & 2\nif constexpr (PB) wgs_flush_end<PB>(p, S, pd);\nelse wgs_flush<PB>(p, S, pd);\n"
         "#else\nwgs_flush<PB>(p, S, pd);\n#endif\n}\n")

A_P1 = ("float ps[8];\n#pragma unroll\nfor (int sl = 0; sl < 8; ++sl) {\nps[sl] = 0.f;\nif (qv[sl] >= 0) {\n"
        "const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n"
        "const float4 a = __ldcg(src), b = __ldcg(src + 1);\n")
N_P1 = ("float ps[8];\n#if OKC_HYBST & 4\nfloat4 ya[8], yb[8];\n"
        + r'asm volatile("prefetch.global.L1 [%0];\n" ::"l"(p.topk_w + tk * 8));' + "\n#endif\n#pragma unroll\n"
        "for (int sl = 0; sl < 8; ++sl) {\nps[sl] = 0.f;\nif (qv[sl] >= 0) {\n"
        "const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n"
        "const float4 a = __ldcg(src), b = __ldcg(src + 1);\n"
        "#if OKC_HYBST & 4\nya[sl] = a; yb[sl] = b;\n#endif\n")

A_P2 = ("const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);\n"
        "const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n"
        "const float4 a = __ldcg(src), b = __ldcg(src + 1);\n")
N_P2 = ("const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);\n"
        "#if OKC_HYBST & 4\nconst float4 a = ya[sl], b = yb[sl];\n#else\n"
        "const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n"
        "const float4 a = __ldcg(src), b = __ldcg(src + 1);\n#endif\n")

# ---------------------------------------------------------------- Python glue
A_NOTE = 'helpers.\n"""\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n'
N_NOTE = ('helpers.\nModified by team joeyhlu (2026-10-31, -DOKC_HYBST, tools/hybst_patches.py): unit-end epilogue constants\n'
          'prefetched to L1 and computed once per unit, the phase-end split read-back issues all n-tiles at once, and the final\n'
          'pass reuses its first-pass loads; HYB\'s arithmetic is unchanged.\n"""\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n')

A_STATE = ('_HYB = {"mod": None, "ok": False, "w": None, "sc": None, "full": None, "nsm": 0, "err": None, "build_s": 0.0, '
           '"pdl": -1}\n')
N_STATE = ('_HYB = {"mod": None, "ok": False, "w": None, "sc": None, "full": None, "nsm": 0, "err": None, "build_s": 0.0, '
           '"pdl": -1, "mods": {}, "pdls": {}}\n')

A_BUILD = '''def _hyb_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((_HYB_MOE_CPP + _HYB_MOE_CUDA + " ".join(HYB_FLAGS)).encode()).hexdigest()[:12]
    t0 = time.time()
    _HYB["mod"] = load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYB_MOE_CUDA,
                              functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=list(HYB_FLAGS), verbose=False)
    _HYB["build_s"] = time.time() - t0
    return _HYB["mod"]
'''
N_BUILD = '''def _hyb_build_one(bits):
    from torch.utils.cpp_extension import load_inline
    fl = list(HYB_FLAGS) + [f"-DOKC_HYBST={int(bits) & 7}"] + (["-DOKC_DEBUG=1"] if bits & 8 else [])
    tag = hashlib.sha1((_HYB_MOE_CPP + _HYB_MOE_CUDA + " ".join(fl)).encode()).hexdigest()[:12]
    _HYB["mods"][bits] = load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=_HYB_MOE_CUDA,
                                     functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
def _hyb_build():
    t0 = time.time()
    builds = tuple(dict.fromkeys((HYB_ST,) + tuple(HYB_ST_BUILDS)))
    ths = [threading.Thread(target=_hyb_build_one, args=(b,), name=f"hyb-st{b}", daemon=True) for b in builds[1:]]
    for th in ths:
        th.start()
    _hyb_build_one(builds[0])
    for th in ths:
        th.join()
    _HYB["mod"] = _HYB["mods"][HYB_ST]
    for b in builds[1:]:
        if b not in _HYB["mods"]:
            print(f"[hyb] HYBST build {b} failed: dropped", flush=True)
    _HYB["build_s"] = time.time() - t0
    return _HYB["mod"]
'''

A_CALL = '''    if _HYB["pdl"] != pdl:
        _HYB["mod"].okc_set_pdl(pdl)
        _HYB["pdl"] = pdl
    _HYB["mod"].okc_run(_HYB["w"], _HYB["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
'''
N_CALL = '''    md = _HYB["mods"].get(HYB_ST) or _HYB["mod"]
    if _HYB["pdls"].get(id(md)) != pdl:
        md.okc_set_pdl(pdl)
        _HYB["pdls"][id(md)] = pdl
    sc = _HYB["sc"] + [_HYB["dbg"]] if (HYB_ST & 8 and _HYB.get("dbg") is not None) else _HYB["sc"]
    md.okc_run(_HYB["w"], sc, x, ids, wts, out, 0, 32, _HYB["nsm"])
'''

A_ON = "def _hyb_on(m):\n"
N_ON = '''def _hyb_st_check(d, base_o, rep_d):
    global HYB_ST
    for b in [k for k, m in _HYB["mods"].items() if m is not _HYB["mod"]]:
        keep, HYB_ST = HYB_ST, b
        try:
            o = _hyb_call(d["hidden_states"], d["topk_ids"], d["topk_weights"])
            torch.cuda.synchronize()
            dd = float((o.float() - base_o.float()).abs().max())
            ok = bool(torch.isfinite(o.float()).all()) and (dd == 0.0 or _k2a_rel_err(o, base_o)[0] <= max(2.0 * rep_d, 1e-6))
        except Exception as ex:
            ok, dd = False, float("nan")
            print(f"[hyb] st{b} check raised {ex!r}"[:300], flush=True)
        finally:
            HYB_ST = keep
        print(f"[hyb] st{b} vs st{keep} bs={d['hidden_states'].shape[0]}: max abs diff {dd:.3e} (base repeat {rep_d:.1e}) -> "
              f"{'ok' if ok else 'FAIL: build dropped'}", flush=True)
        if not ok:
            _HYB["mods"].pop(b, None)
def _hyb_on(m):
'''

A_ST = "                prev = o.clone()\n        gok = _hyb_guards_ok()\n"
N_ST = "                prev = o.clone()\n            _hyb_st_check(d, prev, rep_d)\n        gok = _hyb_guards_ok()\n"

A_SCR = '        _HYB["sc"], _HYB["full"] = _hyb_scratch(dev)\n'
N_SCR = ('        _HYB["sc"], _HYB["full"] = _hyb_scratch(dev)\n'
         '        _HYB["dbg"] = torch.zeros(_HYB_SLOTS * 8, dtype=torch.int64, device=dev) if any(b & 8 for b in HYB_ST_BUILDS) else None\n')
A_DBG = "#define OKC_DEBUG 0\n"
N_DBG = "#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif\n"

A_LIVE = '''f"pdl {int(HYB_PDL)}, nvcc {_HYB['build_s']:.0f}s, prepare {time.time() - t1:.1f}s; "'''
N_LIVE = '''f"pdl {int(HYB_PDL)}, st {HYB_ST} builds {sorted(_HYB['mods'])}, nvcc {_HYB['build_s']:.0f}s, prepare {time.time() - t1:.1f}s; "'''


def apply(s, bits, builds, st):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = (f"# nonce k3-hybst (k3_hh_3k + R-hybst: HYB unit-end / tail riders OKC_HYBST, st {st} builds "
                f"{','.join(map(str, builds))}; tools/hybst_patches.py)")
    s = NL.join(lines)
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    cu = s[i0:i1]
    for a, n, w in ((A_DBG, N_DBG, "OKC_DEBUG default"), (A_DEF, N_DEF, "OKC_HYBST default"), (A_SEGT, N_SEGT, "wgs_seg template head"),
                    (A_FIRST, N_FIRST, "first stage call"), (A_EPI, N_EPI, "whole-unit epilogue"),
                    (A_FIN, N_FIN, "phase-end flush"), (A_P1, N_P1, "final pass 1"), (A_P2, N_P2, "final pass 2")):
        check(s.count(a) == 1 and cu.count(a) == 1, f"CUDA anchor {w} not unique inside _HYB_MOE_CUDA")
        cu = cu.replace(a, n)
    s = s[:i0] + cu + s[i1:]
    a_flags = next(ln for ln in s.split(NL) if ln.startswith("HYB_FLAGS = ("))
    s = sub(s, a_flags + NL, a_flags + NL + f"HYB_ST = {st}" + NL + f"HYB_ST_BUILDS = {tuple(builds)!r}" + NL, "HYB_FLAGS")
    s = sub(s, A_NOTE, N_NOTE, "HYB notice end")
    s = sub(s, A_STATE, N_STATE, "_HYB dict")
    s = sub(s, A_BUILD, N_BUILD, "_hyb_build")
    s = sub(s, A_CALL, N_CALL, "_hyb_call")
    s = sub(s, A_ON, N_ON, "_hyb_on")
    s = sub(s, A_ST, N_ST, "self-test loop end")
    s = sub(s, A_LIVE, N_LIVE, "LIVE print")
    s = sub(s, A_SCR, N_SCR, "HYB scratch")
    return s


def gates(path, s):
    import pyslim
    n = len(s.encode())
    check(n < LIMIT, f"size {n} B >= {LIMIT}")
    check(s.isascii(), "non-ASCII byte")
    ast.parse(s)
    d = pyslim.decorated_sources(s)
    bad = [x for x in d if x.startswith("UNPARSABLE")]
    check(not bad, f"decorated defs that do not re-parse from their own text: {bad[:2]}")
    for a in ("wgs_flush_end<PB>(p, S, pd);", "wgs_epi_tile_c<PB>(p, ec, r16, nt, d[nt]);", "ya[sl] = a; yb[sl] = b;",
              "#define OKC_HYBST 0", "def _hyb_st_check(", "HYB_ST_BUILDS = ", '"mods": {}, "pdls": {}}', "-DOKC_HYBST="):
        check(s.count(a) >= 1, f"missing {a!r}")
    print(f"hybst_patches: gates OK on {path.name}: {n:,} B ({LIMIT - n:,} B under the cap), {len(d)} decorated defs re-parse")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3k.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybst_1.py")
    bits = int(ARGS.get("bits", "7"))
    check(0 <= bits <= 15, "--bits in 0..15 (8 = alygg77's DBG marks build, lab only)")
    if "final" in FLAGS:
        builds, st = (bits,), bits
    else:
        builds = tuple(int(b) for b in ARGS.get("builds", f"0,{bits}").split(","))
        st = int(ARGS.get("st", str(builds[0])))
    new = apply(base.read_bytes().decode("ascii"), bits, builds, st)
    if "check" in FLAGS:
        check(out.exists(), f"{out} missing")
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        gates(out, new)
        print(f"hybst_patches: --check OK ({out.name} == apply({base.name}))")
        return
    ast.parse(new)
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"hybst_patches: {base.name} -> {out.name} ({len(new):,} B, +{len(new) - base.stat().st_size:,} B; st {st}, "
          f"builds {builds})")
    gates(out, new)


if __name__ == "__main__":
    main()

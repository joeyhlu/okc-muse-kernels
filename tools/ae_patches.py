"""ae_patches.py: ASYNC EPILOGUE (VM_AE) for the VML MoE at T >= 32 (notes/handoff_ae.md, lab R-ae-1..).

  python tools/ae_patches.py arms                                   -> kernels/lab/k3_ae_0.py (control: base, new nonce)
                                                                       kernels/lab/k3_ae_1.py (AE: bf2a + w72a routed)
  python tools/ae_patches.py rider --base=kernels/muse_decode_layer/k3_gc_20.py --out=<file>   (asserted anchors, any base)

Design (bf2 at T 32, w72 at T 128 / 256; compile-time VM_AE, no runtime flag branch in the consumer):
  * consumer, phase A, split segment: red.add.v4 of the partials into fix (fire-and-forget), __syncwarp, lane 0 arrives on the
    smem mbarrier aeb[k & 3] (count NCW) and the warp starts the next segment at once (no bar.sync / counter / read-back).
  * consumer, phase A, full unit: SwiGLU + xB stores + fence.proxy.async (as shipped), __syncwarp, lane 0 arrives.
  * worker: 2 spare warps (w72: the producer WG's gather warps NCW+2 / NCW+3 after gather_w5; bf2: NTH 576 -> 640, two extra
    warps, still 96 regs). Worker warp w owns WGs q = 2w, 2w+1. It walks the CTA's phase-A segment list (same split_point
    rotation as phase()), waits aeb[k & 3] (all 16 consumer warps), then: full unit -> red.release.gpu doneA += 2; split ->
    lanes 0 / 1 atom.acq_rel.gpu on cnt[slot][q], if last arriver: fence.acq_rel, read back + zero fix, SwiGLU, xB stores
    (bit-identical math and layout to epilogue<false, N> VCP_PK), cnt = 0, fence.proxy.async, red.release.gpu doneA += #q.
    Then arrives on aeb[4 + (k & 3)] (count 2): consumers wait on it before re-using a slot (4 deep: never blocks in practice).
  * memory model: notes/assumptions.md A29 (mbarrier arrive = release.cta / try_wait = acquire.cta replaces the WG bar.sync of
    the ship's red.add -> counter and store -> release chains; the gpu-scope release stays on the worker side).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_gc_20.py"


def _sub(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, n, old[:100])
    return s.replace(old, new)


WORKER = r"""#if VM_AE
static_assert(!VM_AD && !VM_DP && !VM_SC && !VM_DEFER2 && !VM_GSM && VM_WG5 != 1 && VCP_PK, "VM_AE: bf2 / w72 configurations only");
__device__ __forceinline__ void ae_arrive(const Args& A, const Tab& T, Pend& pd, int lane) {   // AE: this warp's segment pd.k is issued
const int k = pd.k++;
__syncwarp();
if (lane == 0) {
if (k >= 4) mbar_wait(sa(&T.aeb[4 + (k & 3)]), ((k >> 2) - 1) & 1, A.err);   // slot free (worker done with segment k - 4)
mbar_arrive(sa(&T.aeb[k & 3]));
}
}
template <int B>
__device__ __forceinline__ int ae_split(const Args& A, const Tab& T, int j, int ub, int u, int scnt, int w, int lane) {
const int slot = cta_of(T, 0, u << 4, gridDim.x);
int last = 0;
if (lane < 2) last = (atom_acqrel_add(A.cnt + slot * 4 + 2 * w + lane, scnt) + scnt == 16);
__syncwarp();
const unsigned m = __ballot_sync(0xffffffffu, last);
if (m == 0u) return 0;
fence_acq_rel();
const int cn = T.jC[j], npad = (cn + 7) & ~7, ntc = npad >> 3;
const float gs = __ldg(A.gsc13 + T.jE[j]);
const int g = lane >> 2, t = lane & 3;
__nv_bfloat16* const xb0 = A.xB + (size_t)T.jrow[j] * KDIM;
float* const f0 = A.fix + (size_t)(slot * 4 + 2 * w) * FIXW + lane * 4;
const int nit = 8 * ntc;   // item i: wp = i & 3, q = 2w + ((i >> 2) & 1), nt = i >> 3; items of a non-last q are skipped
#pragma unroll 1
for (int i0 = 0; i0 < nit; i0 += B) {
float4 v[B];
#pragma unroll
for (int b = 0; b < B; ++b) {
const int i = i0 + b, wp = i & 3, qi = (i >> 2) & 1, nt = i >> 3;
if (i < nit && ((m >> qi) & 1u)) v[b] = __ldcg(reinterpret_cast<const float4*>(f0 + (size_t)qi * FIXW + nt * 512 + wp * 128));
}
#pragma unroll
for (int b = 0; b < B; ++b) {
const int i = i0 + b, wp = i & 3, qi = (i >> 2) & 1, q = 2 * w + qi, nt = i >> 3;
if (i < nit && ((m >> qi) & 1u)) {   // warp-uniform (shfl below)
*reinterpret_cast<float4*>(f0 + (size_t)qi * FIXW + nt * 512 + wp * 128) = make_float4(0.f, 0.f, 0.f, 0.f);
const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;
const int col = 8 * tg + g, s2 = col >> 8, kk = col & 255;
const int ii = kk >> 6, tt = (kk >> 4) & 3, qq = (kk >> 1) & 7, bb = kk & 1;
const __nv_bfloat162 v2 = __floats2bfloat162_rn(swiglu(v[b].x * gs, v[b].z * gs), swiglu(v[b].y * gs, v[b].w * gs));
const uint32_t vs = *reinterpret_cast<const uint32_t*>(&v2);
const uint32_t vo = __shfl_xor_sync(0xffffffffu, vs, 4);
const uint32_t w2 = bb ? __byte_perm(vo, vs, 0x7632) : __byte_perm(vs, vo, 0x5410);
const int n = 8 * nt + 2 * t + bb;
if (n < cn) *reinterpret_cast<uint32_t*>(xb0 + (size_t)s2 * npad * 256 + ((((nt * 4 + ii) * 8 + qq) * 8 + (2 * t + bb)) * 4 + tt) * 2) = w2;
}
}
}
if (lane < 2 && ((m >> lane) & 1u)) A.cnt[slot * 4 + 2 * w + lane] = 0;
fence_proxy_async();
__syncwarp();
return __popc(m);
}
template <int B>
__device__ __forceinline__ void ae_worker(const Args& A, const Tab& T, int w, int lane) {   // AE worker warp w: WGs 2w, 2w + 1
const int c = blockIdx.x;
const int g0 = T.bA[c], g1 = T.bA[c + 1], sp = split_point(A, 0, g0, g1);
int k = 0;
#pragma unroll 1
for (int gs = sp, ge = g1; ;) {
if (gs >= ge) {
if (ge == g1 && sp > g0) { gs = g0; ge = sp; } else break;
}
const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs)), j = u >> 5, ub = u & 31;
mbar_wait(sa(&T.aeb[k & 3]), (k >> 2) & 1, A.err);
const int nrel = (s0 == 0 && s1 == 16) ? 2 : ae_split<B>(A, T, j, ub, u, s1 - s0, w, lane);
if (lane == 0 && nrel) red_release_add(A.doneA + j * 16 + (ub >> 1), nrel);
__syncwarp();
if (lane == 0) mbar_arrive(sa(&T.aeb[4 + (k & 3)]));
++k;
gs += s1 - s0;
}
}
#endif
"""


def ae(s):
    """VM_AE device code (default off) + the bf2a / w72a variants routed at 32 / 128 / 256."""
    s = _sub(s, "constexpr int NTH = 32 * NCW + (VM_WG5 ? 128 : 64);",
             "#ifndef VM_AE\n#define VM_AE 0   // async epilogue (tools/ae_patches.py)\n#endif\n#ifndef VM_AEB\n#define VM_AEB 8   // AE worker read-back batch\n#endif\n"
             "constexpr int NTH = 32 * NCW + ((VM_WG5 || VM_AE) ? 128 : 64);")
    s = _sub(s, "unsigned long long bperm[JMAX];   // bit 9: per job, 16 x 4 bits: B order position -> stage\n",
             "unsigned long long bperm[JMAX];   // bit 9: per job, 16 x 4 bits: B order position -> stage\n"
             "#if VM_AE\nunsigned long long aeb[8];         // AE: 0-3 segment issued (count NCW), 4-7 worker done (count 2)\n#endif\n")
    s = _sub(s, "#else\nstruct Pend { int i; };\n#endif\n",
             "#elif VM_AE\nstruct Pend { int i; int k; };\n#else\nstruct Pend { int i; };\n#endif\n")
    # forward decl: epilogue / seg (above) call ae_arrive, defined with the worker below seg
    s = _sub(s, "template <bool PB, int N>\n__device__ __forceinline__ void epilogue(",
             "__device__ __forceinline__ void ae_arrive(const Args& A, const Tab& T, Pend& pd, int lane);   // AE (defined below seg)\n"
             "template <bool PB, int N>\n__device__ __forceinline__ void epilogue(")
    s = _sub(s, "if constexpr (VM_DEFER || VM_DEFER2) {\npd.i = j * 16 + (ub >> 1);\n} else {",
             "if constexpr (VM_AE) {\nfence_proxy_async();\nae_arrive(A, T, pd, lane);   // the worker releases doneA\n"
             "} else if constexpr (VM_DEFER || VM_DEFER2) {\npd.i = j * 16 + (ub >> 1);\n} else {")
    s = _sub(s, "for (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);\n#if VM_AD\n",
             "for (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);\n"
             "#if VM_AE\nif constexpr (!PB) { ae_arrive(A, T, pd, lane); SEGMK(segmk_done(A, smk, s0, s1);) return; }   // counter + resolve: worker\n#endif\n"
             "#if VM_AD\n")
    s = _sub(s, "template <bool PB>\n#if VM_DP\n__device__ __forceinline__ void phase(",
             WORKER + "template <bool PB>\n#if VM_DP\n__device__ __forceinline__ void phase(")
    s = _sub(s, "{ mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }\n",
             "{ mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }\n"
             "#if VM_AE\nfor (int i = 0; i < 4; ++i) { mbar_init(sa(&T.aeb[i]), NCW); mbar_init(sa(&T.aeb[4 + i]), 2); }\n#endif\n")
    s = _sub(s, "if (warp > NCW + 1) { if (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1); if (VM_W5B6) bar_arrive(6, VM_B6); return; }",
             "if (warp > NCW + 1) {\nif (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1);\nif (VM_W5B6) bar_arrive(6, VM_B6);\n"
             "#if VM_AE\nae_worker<VM_AEB>(A, T, warp - NCW - 2, lane);\n#endif\nreturn;\n}")
    s = _sub(s, "#if VM_WG5 < 2\nif (warp >= NCW) {\n#if VM_WG5\n",
             "#if VM_WG5 < 2\nif (warp >= NCW) {\n#if VM_AE\nif (warp > NCW + 1) { ae_worker<VM_AEB>(A, T, warp - NCW - 2, lane); return; }\n#endif\n#if VM_WG5\n")
    # host side: the AE variants (same scratch family as bf2 / w72: BRED y buffer)
    s = defs(s, {"bf2a": ("bf2", ("-DVM_AE=1", "-DVM_AEB=16")), "w72a": ("w72", ("-DVM_AE=1", "-DVM_AEB=4"))})
    s = _sub(s, 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]',
             'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72", "bf2a", "w72a", "bf2r", "w72r", "bf2ar", "w72ar") else _VML["sc"]')
    return s


def defs(s, new):
    """add variants to the _VML_DEFS dict literal (kept a literal: tools/pdl_sass.py reads it from the AST)."""
    a = s.index("\n_VML_DEFS = {") + 1
    b = s.index("\n_VML = {", a)
    blk = s[a:b]
    assert blk.endswith(")}"), blk[-20:]
    add = ""
    for v, (src, extra) in new.items():
        i = blk.index(f'"{src}": (') + len(f'"{src}": (')
        k = blk.index(")", i)
        add += f',\n             "{v}": (' + blk[i:k] + "".join(f', "{d}"' for d in extra) + ")"
    return s[:a] + blk[:-1] + add + "}" + s[b:]


RLOG_N = 80 * 16 + 3 * 136   # JMAX x 16 slice releases + per CTA: start, consumer A end, end (GMAX rows each)


def rlog(s):
    """lab-only release log (VM_RLOG, marks files): a.marks off; T[18] of exactly RLOG_N int64 -> A.rlog:
    [j * 16 + s] = latest doneA release issue time (atomicMax globaltimer ns), [1280 + c] start, [1280 + 136 + c] consumer A end
    (q0 warp 0), [1280 + 272 + c] end (max over WGs). Every release site (ship pend_flush / inline, AE worker) is logged."""
    s = _sub(s, "#ifndef VM_DEFER2\n#define VM_DEFER2 0\n#endif\n",
             "#ifndef VM_DEFER2\n#define VM_DEFER2 0\n#endif\n#ifndef VM_RLOG\n#define VM_RLOG 0   // lab: doneA release log (tools/ae_patches.py)\n#endif\n"
             "#if VM_RLOG\n#define RLOG_REL(i) do { if (A.rlog) atomicMax(reinterpret_cast<unsigned long long*>(A.rlog) + (i), "
             "(unsigned long long)gtime()); } while (0)\n#else\n#define RLOG_REL(i) do { } while (0)\n#endif\n")
    s = _sub(s, "__nv_bfloat16* out; int* err; long long* marks; unsigned long long magG; int flags;\n",
             "__nv_bfloat16* out; int* err; long long* marks; unsigned long long magG; int flags;\n#if VM_RLOG\nlong long* rlog;\n#endif\n")
    s = _sub(s, "#if VM_MKW == 64\na.G0 = (int)G;\n#endif\n",
             "#if VM_MKW == 64\na.G0 = (int)G;\n#endif\n#if VM_RLOG\n"
             f"a.rlog = (T[18].numel() == {RLOG_N}) ? reinterpret_cast<long long*>(T[18].data_ptr()) : nullptr;\na.marks = nullptr;\n#endif\n")
    s = _sub(s, "if (((wp << 5) | lane) == 0) red_release_add(A.doneA + pd.i, 1);",
             "if (((wp << 5) | lane) == 0) { red_release_add(A.doneA + pd.i, 1); RLOG_REL(pd.i); }")
    s = _sub(s, "if (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);",
             "if (wt == 0) { red_release_add(A.doneA + j * 16 + (ub >> 1), 1); RLOG_REL(j * 16 + (ub >> 1)); }")
    if "if (lane == 0 && nrel) red_release_add(" in s:
        s = _sub(s, "if (lane == 0 && nrel) red_release_add(A.doneA + j * 16 + (ub >> 1), nrel);",
                 "if (lane == 0 && nrel) { red_release_add(A.doneA + j * 16 + (ub >> 1), nrel); RLOG_REL(j * 16 + (ub >> 1)); }")
    s = _sub(s, "if (A.marks) {   // v11: SM id",
             "#if VM_RLOG\nif (A.rlog) A.rlog[JMAX * 16 + blockIdx.x] = gtime();\n#endif\nif (A.marks) {   // v11: SM id")
    s = _sub(s, "if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();\n",
             "if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();\n"
             "#if VM_RLOG\nif (wp == 0 && lane == 0 && q == 0 && A.rlog) A.rlog[JMAX * 16 + GMAX + blockIdx.x] = gtime();\n#endif\n")
    s = _sub(s, "if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();\n",
             "if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();\n"
             "#if VM_RLOG\nif (wp == 0 && lane == 0 && A.rlog) atomicMax(reinterpret_cast<unsigned long long*>(A.rlog) + JMAX * 16 + 2 * GMAX + "
             "blockIdx.x, (unsigned long long)gtime());\n#endif\n")
    # host: hook the marks tensor at every VML bucket; RLOG on all four variants (unused ones are not built)
    s = _sub(s, "VML_PDEL = {}\n", "VML_PDEL = {}\nVML_MARKS = None   # AE lab: int64 tensor of RLOG_N -> release log (VM_RLOG builds)\n")
    s = _sub(s, "flags=fl, cost=VML_COST.get(rep), pool=", "flags=fl, marks=VML_MARKS, cost=VML_COST.get(rep), pool=")
    return defs(s, {v + "r": (v, ("-DVM_RLOG=1",)) for v in ("bf2", "w72", "bf2a", "w72a")})   # bf2r w72r bf2ar w72ar


def route(s, var="bf2a", bvar="w72a"):
    s = _sub(s, "\nVML_VAR = 'bf2'\n", f"\nVML_VAR = '{var}'\n")
    s = _sub(s, "\nVML_BVAR = {256: 'w72', 128: 'w72'}\n", f"\nVML_BVAR = {{256: '{bvar}', 128: '{bvar}'}}\n")
    return s


def nonce(s, tag):
    lines = s.split("\n", 2)
    assert lines[1].startswith("# nonce "), lines[1][:60]
    lines[1] = f"# nonce {tag}"
    return "\n".join(lines)


def build(name, s):
    p = ROOT / "kernels/lab" / name
    p.write_text(s, encoding="utf-8", newline="\n")
    print(f"wrote {p.relative_to(ROOT)} ({len(s)} B)")


def main():
    mode = sys.argv[1]
    base_p = next((ROOT / a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--base=")), BASE)
    out_p = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), None)
    s0 = base_p.read_text(encoding="utf-8")
    if mode == "arms":
        build("k3_ae_0.py", nonce(s0, f"k3-ae-0 (control: {base_p.name} unchanged; tools/ae_patches.py)"))
        build("k3_ae_1.py", nonce(route(ae(s0)), f"k3-ae-1 ({base_p.name} + VM_AE async epilogue: bf2a / w72a; tools/ae_patches.py)"))
        lab = rlog(route(ae(s0)))   # one file, every arm by globals: ship bf2 / w72, AE bf2a / w72a, + RLOG twins (r suffix)
        lab = _sub(lab, "\nVML_ABVARS = ()\n", "\nVML_ABVARS = ('bf2', 'w72', 'bf2r', 'w72r', 'bf2ar', 'w72ar')\n")
        build("k3_ae_L.py", nonce(lab, f"k3-ae-L ({base_p.name} + VM_AE + VM_RLOG; all arms in one file via VML_VAR / VML_BVAR; lab)"))
    elif mode == "rider":
        assert out_p, "--out=kernels/...py required"
        s = route(ae(s0))
        Path(ROOT / out_p).write_text(s, encoding="utf-8", newline="\n")
        print(f"wrote {out_p} (AE rider on {base_p.name})")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

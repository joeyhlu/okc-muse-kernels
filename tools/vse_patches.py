"""vse_patches.py: VML phase-A segment-end cost (R-vse): sub-step marks + VSE_HOLD (slot hold through the end-of-segment fences).

Mechanism (R-dsm-0b/0c): a GPU-scope fence / release / acq_rel atomic waits for the SM's in-flight bulk copies. At a phase-A
segment end the consumers have just freed the last stage's weight slot (mid-stage, codes in registers) and its x slot (after the
drain), so the producers' next copies are in flight exactly when thread 0 runs the split count atom.acq_rel (WG stalls at the
2nd bar.sync), the resolver fence.acq_rel and the next segment's pend_flush red.release (wgmma is WG-collective: one warp's
MEMBAR stalls the WG). B segment ends (BRED: red.add only) have no GPU fence; that is the A / B segment-end gap (R-lat32-5).
VSE_HOLD (compile-time, default 0 = the base source):
  bit 1: split phase-A shares keep their last stage's weight slot and x slot until the count atomic returned (non-last shares:
         released after the 2nd bar.sync; the resolver: after its fence.acq_rel unless bit 2 carries them).
  bit 2: unsplit phase-A ends (and bit-1 resolvers) keep them until the next pend_flush's red.release (carried in Pend).
  The ring still holds the next 2 stages (NS = NX = 3), so the consumer never waits on a held slot; only the producers' next copy
  starts later (~1-2 us, inside the ring's slack). Arithmetic and the store / atomic order are untouched (bitwise == base).
Marks (VM_MKW 64 builds only, SEGMK; thread 0 of WG 0, per-CTA sums, phase A only; cols 17-23 / 33-39 are the unused stage-count
columns of NTM <= 9): 17 first stage (incl. pend_flush), 36 rest stages, 22 drain, 23 end work, 35 split shares, 39 resolvers,
33 split red.add issue, 38 count atom MEMBAR (issue returns), 34 count atom incl. its result, 18 pend_flush total, 19 its
fence.proxy.async, 20 its bar.sync, 21 its red.release, 37 pend_flush count. Existing: 53 / 52 A epilogue time / count, 55
resolver fence + read-back, 62 bar + atom + bar, 56-61 both phases, 49 / 50 segments A / B, 48 first weight, 6 A end (WG 0).

VSE_PS (R-vse-2): head / tail split A shares __stcg their partial into a private region (head = A fix region, tail = B fix region,
unused under VM_BRED), middle shares red.add into a third region; the count atom carries region bits; the last arriver adds the
other region(s) to its registers (2 shares: bitwise == the red.add sum) and zeroes only region 2.
    python -I tools/vse_patches.py <file.py> [--ps=1] [--hold=<bits>] [--variants=bf2,w72] [--out=<new file>]
    (tools/vse_local_test.py for SASS / regs; layer files: --ps=1 on BOTH bf2 and w72, they share one VML scratch)
"""
import pathlib
import sys


def _rep(s, old, new, tag):
    n = s.count(old)
    assert n == 1, f"vse anchor '{tag}' found {n} times"
    return s.replace(old, new)


DEFS = r'''#ifndef VSE_HOLD
#define VSE_HOLD 0   // R-vse: bit 1 = split A shares hold their last weight + x slot through the count atomic; bit 2 = unsplit A ends through the next pend_flush
#endif
#ifndef VSE_PS
#define VSE_PS 0   // R-vse: 1 = head / tail split A shares plain-store private partials (no red.add, no resolver zeroing)
#endif
#if VM_MKW == 64 && VM_NTM > 9
#error "R-vse marks use the stage-count columns 17-23 / 33-39 (VM_NTM <= 9)"
#endif
'''
CHECK = r'''#if VSE_PS && (!VM_BRED || VM_DP || VM_AD)
#error "VSE_PS uses the B fix region (unused under VM_BRED) for tail shares"
#endif
#if VSE_HOLD && (!VM_TRICKS || VCP_PRE || VCP_EA || !VM_DEFER || !VM_BRED || VM_DP || VM_AD || VM_DEFER2 || !SFX || VM_RH)
#error "VSE_HOLD supports the shipped bf2 / w72 consumer shape only"
#endif
'''
HELP = r'''
#if VSE_HOLD
static_assert(NS == NX, "R-vse: one held index for the weight and the x ring");
__device__ __forceinline__ void vse_rel(int h, int lane) {   // R-vse: release the held weight + x slot h (lane 0 of each consumer warp)
extern __shared__ __align__(1024) uint8_t smem[];
if (lane == 0) {
const uint32_t bars = sa(smem + BAR_OFF);
mbar_arrive(bars + 8 * (NS + h));
mbar_arrive(bars + 8 * (2 * NS + NX + h));
}
}
#endif
#if VSE_PS
template <int N, int M>   // R-vse PS: M 0 d += region, 1 d += region + zero it, 2 d = region + zero it
__device__ __forceinline__ void vse_take(float* p, float* d) {
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(p + nt * 512));
if (M) *reinterpret_cast<float4*>(p + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
if (M == 2) { d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w; }
else { d[4 * nt] += v.x; d[4 * nt + 1] += v.y; d[4 * nt + 2] += v.z; d[4 * nt + 3] += v.w; }
}
}
#endif
#if VM_MKW == 64
__device__ __forceinline__ long long vse_gt(int dep) {   // R-vse marks: globaltimer read ordered after dep is available
long long t;
asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t) : "r"(dep));
return t;
}
#endif
'''


def apply(s):
    s = _rep(s, "#ifndef VM_DEFER\n#define VM_DEFER 0\n#endif\n", "#ifndef VM_DEFER\n#define VM_DEFER 0\n#endif\n" + DEFS, "defs")
    s = _rep(s, "template <int N> struct VcpCfg {", CHECK + "template <int N> struct VcpCfg {", "check")
    # Pend: carried held slots
    s = _rep(s, "struct Pend { int i; };\n#endif\n", "struct Pend { int i; };\n#endif\n" + HELP, "pend")
    # pend_flush: marks + carried release
    s = _rep(s, "if (pd.i >= 0) {\nfence_proxy_async();\nbar_sync(1 + q, 128);\n",
             "if (pd.i >= 0) {\n#if VSE_HOLD\nconst int vh = (pd.i >> 28) - 1;   // R-vse: held slot + 1 carried in bits 28-29\npd.i &= 0x0fffffff;\n#endif\n"
             "SEGMK(const bool vmk = A.marks != nullptr && q == 0 && ((wp << 5) | lane) == 0; "
             "const long long vt0 = vmk ? gtime() : 0;)\nfence_proxy_async();\nSEGMK(const long long vt1 = vmk ? gtime() : 0;)\n"
             "bar_sync(1 + q, 128);\nSEGMK(const long long vt2 = vmk ? gtime() : 0;)\n", "pf head")
    s = _rep(s, "pd.i = -1;\n}\n}\n#ifndef VCP_NB",
             "SEGMK(if (vmk) { const long long vt3 = gtime(); long long* vm = A.marks + (size_t)blockIdx.x * MKW; vm[18] += vt3 - vt0; "
             "vm[19] += vt1 - vt0; vm[20] += vt2 - vt1; vm[21] += vt3 - vt2; vm[37] += 1; })\n"
             "#if VSE_HOLD\nif (vh >= 0) vse_rel(vh, lane);   // R-vse: after the release's MEMBAR\n#endif\n"
             "pd.i = -1;\n}\n}\n#ifndef VCP_NB", "pf tail")
    # stage / stage_m: hold parameter
    s = _rep(s, "uint32_t (&hb)[VcpCfg<N>::NB][8], int q, int wp, int lane, bool first, Pend& pd) {",
             "#if VSE_HOLD\nuint32_t (&hb)[VcpCfg<N>::NB][8], int q, int wp, int lane, bool first, Pend& pd, bool hold) {\n#else\n"
             "uint32_t (&hb)[VcpCfg<N>::NB][8], int q, int wp, int lane, bool first, Pend& pd) {\n#endif", "stage sig")
    old = "if (VC::EA == 0 && i == 3 && hf == 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // all codes in registers"
    s = _rep(s, old, "#if VSE_HOLD\nif (VC::EA == 0 && i == 3 && hf == 0) { __syncwarp(); if (lane == 0 && !hold) mbar_arrive(bars + 8 * (NS + ws)); }"
             "   // R-vse: a held slot is released at the segment end\n#else\n" + old + "\n#endif", "stage wrel")
    s = _rep(s, "float (&d2)[2][N / 2], int q,\nint wp, int lane) {",
             "float (&d2)[2][N / 2], int q,\n#if VSE_HOLD\nint wp, int lane, bool hold) {\n#else\nint wp, int lane) {\n#endif", "stage_m sig")
    old = "if (lane == 0) mbar_arrive(bars + 8 * (NS + ws) + ((c[0].x ^ c[1].x ^ c[2].x ^ c[3].x ^ sw0 ^ sw1) & zom));"
    s = _rep(s, old, "#if VSE_HOLD\n" + old.replace("if (lane == 0)", "if (lane == 0 && !hold)") + "\n#else\n" + old + "\n#endif", "stage_m wrel")
    old = "if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs) + (xdep & zom));"
    s = _rep(s, old, "#if VSE_HOLD\n" + old.replace("if (lane == 0)", "if (lane == 0 && !hold)") + "\n#else\n" + old + "\n#endif", "stage_m xrel")
    # seg: hold predicate, calls, releases, marks
    old = "SEGMK(SegMk smk{0, 0, 0, 0, A.marks != nullptr && q == 0 && wp == 0 && lane == 0}; if (smk.on) smk.t0 = gtime();)"
    s = _rep(s, old, old + "\nSEGMK(smk.pb = PB;)\n#if VSE_HOLD\n"
             "const bool hseg = !PB && ((s0 == 0 && s1 == 16) ? (VSE_HOLD & 2) != 0 : (VSE_HOLD & 1) != 0);\nint hws = -1;\n#endif", "seg head")
    old = "stage_m<N>(A, smem, bars, it, d2, q, wp, lane);\nSEGMK(if (smk.on) smk.t1 = gtime();)"
    s = _rep(s, old, "#if VSE_HOLD\nstage_m<N>(A, smem, bars, it, d2, q, wp, lane, hseg && s1 == s0 + 1);\n#else\n"
             "stage_m<N>(A, smem, bars, it, d2, q, wp, lane);\n#endif\nSEGMK(if (smk.on) smk.t1 = gtime();)", "seg sm first")
    old = "for (int s = s0 + 1; s < s1; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane);"
    s = _rep(s, old, "#if VSE_HOLD\nfor (int s = s0 + 1; s < s1; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane, hseg && s == s1 - 1);\n"
             "if (hseg) hws = (it - 1) % NS;\n#else\n" + old + "\n#endif", "seg sm loop")
    old = "stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd);"
    s = _rep(s, old, "#if VSE_HOLD\nstage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd, hseg && s1 == s0 + 1);\n#else\n"
             + old + "\n#endif", "seg first")
    old = "for (int s = s0 + 1; s < s1; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd);"
    s = _rep(s, old, "#if VSE_HOLD\nfor (int s = s0 + 1; s < s1; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd, "
             "hseg && s == s1 - 1);\n#else\n" + old + "\n#endif", "seg loop")
    old = "wg_wait<0>();\nSEGMK(if (smk.on) smk.t3 = gtime();)\nif (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));"
    s = _rep(s, old, "wg_wait<0>();\nSEGMK(if (smk.on) smk.t3 = gtime();)\n#if VSE_HOLD\n"
             "if (hseg) hws = prevx; else if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));\n#else\n"
             "if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));\n#endif", "seg xrel")
    old = "SEGMK(long long t_at = smk.on ? gtime() : 0;)\nbar_sync(1 + q, 128);\n"
    s = _rep(s, old, "SEGMK(long long t_at = smk.on ? gtime() : 0;)\nSEGMK(if (smk.on && !PB) A.marks[(size_t)blockIdx.x * MKW + 33] += t_at - smk.t3;)\n"
             "bar_sync(1 + q, 128);\n", "seg red mark")
    old = "if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);"
    s = _rep(s, old, "#if VM_MKW == 64 || VSE_PS\nif (wt == 0) {\nSEGMK(const long long t_x0 = smk.on ? gtime() : 0;)\n"
             "#if VSE_PS\nconst int r_x = atom_acqrel_add(fc, (s1 - s0) | (256 << vr));   // R-vse PS: + region bit (head 256, tail 512, middles 1024 each)\n"
             "#else\nconst int r_x = atom_acqrel_add(fc, s1 - s0);\n#endif\n"
             "SEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 38] += gtime() - t_x0;)\n"
             "#if VSE_PS\nT.wlast[q] = ((r_x & 255) + (s1 - s0) == 16) ? (r_x | (256 << vr)) : 0;   // last: the regions written\n"
             "#else\nT.wlast[q] = (r_x + (s1 - s0) == 16);\n#endif\n"
             "SEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 34] += vse_gt(r_x) - t_x0;)\n}\n#else\n" + old + "\n#endif", "seg atom")
    # VSE_PS: head / tail split shares plain-store their partial into a private region (0 / 1; the B fix region is unused under
    # BRED), middle shares red.add into region 2; the resolver adds the other regions to its own registers (2 shares: own + other =
    # the red.add sum a + b bitwise), zeroes only region 2, resets the counter.
    old = "#pragma unroll\nfor (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);"
    s = _rep(s, old, "#if VSE_PS\nconst int vr = (s0 == 0) ? 0 : (s1 == 16 ? 1 : 2);\n"
             "float* fv = A.fix + ((size_t)((vr * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;\n#pragma unroll\nfor (int nt = 0; nt < N / 8; ++nt) {\n"
             "if (vr < 2) __stcg(reinterpret_cast<float4*>(fv + nt * 512), make_float4(d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]));\n"
             "else red_add_v4(fv + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);\n}\n#else\n" + old + "\n#endif", "ps store")
    old = ("#pragma unroll\nfor (int nt = 0; nt < N / 8; ++nt) {\nconst float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));\n"
           "*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);\n"
           "d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;\n}\nif (wt == 0) *fc = 0;\n"
           "SEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 55]")
    s = _rep(s, old, "#if VSE_PS\n{\nconst int lw = T.wlast[q];\nfloat* f2 = A.fix + ((size_t)((2 * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;\n"
             "if (vr == 2) vse_take<N, 2>(f2, d);   // own partial is inside region 2\n"
             "if (vr != 0 && (lw & 256)) vse_take<N, 0>(A.fix + ((size_t)(slot * 4 + q)) * FIXW + wt * 4, d);\n"
             "if (vr != 1 && (lw & 512)) vse_take<N, 0>(A.fix + ((size_t)((SLOTS + slot) * 4 + q)) * FIXW + wt * 4, d);\n"
             "if (vr != 2 && (lw >> 10)) vse_take<N, 1>(f2, d);\n}\nif (wt == 0) *fc = 0;\n#else\n"
             + old[:old.index("SEGMK(")] + "#endif\nSEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 55]", "ps resolve")
    old = "if (!T.wlast[q]) { SEGMK(segmk_done(A, smk, s0, s1);) return; }"
    s = _rep(s, old, "#if VSE_HOLD\nif (!T.wlast[q]) { if (hseg) vse_rel(hws, lane); SEGMK(segmk_done(A, smk, s0, s1);) return; }\n#else\n"
             + old + "\n#endif\nSEGMK(if (smk.on && !PB) A.marks[(size_t)blockIdx.x * MKW + 39] += 1;)", "seg nonlast")
    old = "SEGMK(const long long t_rb = smk.on ? gtime() : 0;)\nfence_acq_rel();\n"
    s = _rep(s, old, old + "#if VSE_HOLD\nif (hseg && !(VSE_HOLD & 2)) vse_rel(hws, lane);   // R-vse: after the resolver's MEMBAR\n#endif\n",
             "seg resolver")
    old = "epilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);\nSEGMK(segmk_done(A, smk, s0, s1);)"
    s = _rep(s, old, "epilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);\n#if VSE_HOLD\n"
             "if (hseg && (VSE_HOLD & 2)) pd.i |= (hws + 1) << 28;   // R-vse: released by the next pend_flush (slot + 1 in bits 28-29)\n#endif\n"
             "SEGMK(segmk_done(A, smk, s0, s1);)", "seg carry")
    # segment marks: phase-A split of 56-61
    s = _rep(s, "struct SegMk { long long t0, t1, t2, t3; bool on; };", "struct SegMk { long long t0, t1, t2, t3; bool on; int pb; };", "segmk")
    old = "mk[56] += m.t1 - m.t0; mk[57] += m.t2 - m.t1; mk[58] += m.t3 - m.t2; mk[59] += t4 - m.t3; mk[60] += 1; mk[61] += s1 - s0 - 1;"
    s = _rep(s, old, old + "\nif (!m.pb) { mk[17] += m.t1 - m.t0; mk[36] += m.t2 - m.t1; mk[22] += m.t3 - m.t2; mk[23] += t4 - m.t3; "
             "if (!(s0 == 0 && s1 == 16)) mk[35] += 1; }   // R-vse: phase A only", "segmk_done")
    return s


def set_flag(s, macro, value, variants=("bf2", "w72")):
    """layer file: append -D<macro>=<value> to the _VML_DEFS entries of the given VML variants (each must end in -DSFX=1).
    VSE_PS changes the fix-buffer protocol and the layer shares one VML scratch (_VML["sc"]) across variants: set it on EVERY
    variant that can run (VML_VAR + VML_BVAR values, both BRED), never on one alone."""
    for v in variants:
        i = s.index(chr(10) + '_VML_DEFS = {')
        k = s.index(f'"{v}": (', i)
        e = s.index('"-DSFX=1")', k)
        assert macro not in s[k:e], v
        s = s[:e] + f'"-DSFX=1", "-D{macro}={value}")' + s[e + len('"-DSFX=1")'):]
    return s


if __name__ == "__main__":
    opt = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    p = apply(t)
    var = tuple(opt.get("variants", "bf2,w72").split(","))
    for key, macro in (("hold", "VSE_HOLD"), ("ps", "VSE_PS")):
        if key in opt:
            p = set_flag(p, macro, int(opt[key]), var)
    print(f"[vse] {sys.argv[1]}: +{len(p) - len(t)} B, all anchors applied once; flags " + " ".join(f"{k}={opt[k]}" for k in ("hold", "ps") if k in opt))
    if "out" in opt:
        o = pathlib.Path(opt["out"])
        assert not o.exists(), f"refusing to overwrite {o}"
        o.write_text(p, encoding="utf-8", newline=chr(10))
        print(f"[vse] wrote {o}")

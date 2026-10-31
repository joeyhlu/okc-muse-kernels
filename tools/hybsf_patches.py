"""hybsf_patches.py (R-hybsf, HYB-SF builder): HYB-SF = VML's bit-6 structure ported to alygg77's WGS MoE engine (HYB, T 49..320), as
proposed in notes/lab.md R-hybgf-0 after the R-hybpro-1 / R-hybew-1 prologue marks (first consumed stage 10-12 us after the wait,
5-6.7 us above the floor; owner = wgs_gather, which gates both the weight issue (WS_JOIN) and the x issue (xready)).

  python -I tools/hybsf_patches.py [--base=kernels/lab/k3_hh_5.py] [--out=kernels/lab/k3_hybsf_ab.py] [--check]      lab file
  python -I tools/hybsf_patches.py --ship --base=<final> --out=<new> [--nf=2] [--gw=4]                                 ship form
  python -I tools/lead_gate.py file <out>

HYB-SF (-DOKC_HYBSF=1; the CUDA text preprocesses to the ship text with OKC_HYBSF 0 and OKC_HYB_MK 0):
  * NTL 576 -> 512 + 32 + 32 * GW (GW = OKC_HYBSF_GW gather warps, default 3 -> 640 threads = 20 warps = 5 per SM quadrant, the
    last count that keeps the 96-register cap: 21 warps put 6 on one quadrant -> ptxas caps at 80 registers and spills 132 B).
    Warp 16 = weight producer, warps 17..19 = gather warps, warp 17 also the x producer. Smem unchanged.
  * Consumers: route -> warp 0 does the jrow scan AND the job-cost prefix + both ranges (wgs_ranges_warp: bitwise the ship's
    wgs_prefix / wgs_range, R-hybew-1) -> bar.arrive 7 (count NTL) releases every producer-side warp at route end -> ss zero ->
    CTA_SYNC -> self-fill the first nf = min(p.sfnf, A stages) x slots straight from x (the ship gather's permuted 16 B chunks,
    4 x LDG.32 each, written to the ring slot in the wgmma layout) -> fence.proxy.async.shared::cta -> CTA_SYNC -> tid 0 plain
    mbarrier arrive on BX_F[0..nf) -> first stage. No WS_JOIN.
  * Weight producer: bar.sync 7 -> issues its WGS_NS copies at route end (ranges already in smem).
  * Gather warps: bar.sync 7 -> each gathers 1 / GW of this CTA's padded-row slice into xA (one row = 16 K-stages x 16 B per lane,
    all 64 loads in flight before the stores) -> fence.proxy.async.global -> lane 0 releases xready per job (red.release.gpu).
    Warp 17 then issues x from stage nf on (slot nf % WGS_NX, parity (nf / WGS_NX) & 1); the pad rows stay unwritten (ship).
  * nf is a run-time value in the lab (okc_set_sfnf; HYBP_NF 0..3) and -DOKC_HYBSF_NF=<n> in the ship form (the same code path).
Arithmetic unchanged: the same x bytes reach the same smem slots, the same ranges, the same wgmma order -> bitwise == ship on
deterministic cases (self-tested in prepare; conc69's atomics order is the only non-bitwise case).
Lab keys (HYBP_ST; the ship module stays the default): 64 = SF (ship flags), 71 = SF + HYB_F7 (st7 + hn; used at T >= HYB_T7 when
HYBP_ST = 64, like the ship's mod7), 80 = SF + marks (-DOKC_HYB_MK=1; 64 u64 per CTA, slot k = globaltimer, 32 + k = clock64):
  0 pre-wait  1 wait done  2 route + jrow + ranges done (bar.arrive 7)  3 self-fill arrived (ship: gather end)  4 consumers past the
  post-route CTA_SYNC (ship: WS_JOIN passed)  5 ranges done (warp 0 lane 0; ship: x producer)  6 first weight issue  7 first x issue
  (stage nf)  8 first weight stage landed  9 BX_F slot 0 phase 0 complete (= self-fill)  10 slot 0 freed (4th weight issue)
  11 A end  12 B end  13 grid barrier  14 end  15-18 ship gather sub-marks  19.. gather part ends (warps 17..)  23 self-fill
  loads + stores done (before the fence)  31 info (A stages | B stages << 16 | nj << 32 | rows R << 48).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import hybpro_patches as HP  # noqa: E402  (marks constants + check / sub)
sys.argv = _argv
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000
check, sub = HP.check, HP.sub

# ---------------- CUDA edits (raw text: asm "\n" stays two characters) ------------------------------------------------------
DEF_A = HP.MK_DEF_A
DEF_N = DEF_A + ("#ifndef OKC_HYB_MK\n#define OKC_HYB_MK 0\n#endif\n"
                 "#define MK(k) do { if (OKC_HYB_MK && p.dbg) { const unsigned long long t_ = gtimer(); "
                 "p.dbg[blockIdx.x * 64 + (k)] = t_; p.dbg[blockIdx.x * 64 + 32 + (k)] = (unsigned long long)clock64(); } } while (0)\n"
                 "#define MKB(k) do { asm volatile(\"\" ::: \"memory\"); MK(k); } while (0)\n")
NTL_A = "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);\n"
NTL_N = ("#ifndef OKC_HYBSF\n#define OKC_HYBSF 0\n#endif\n#ifndef OKC_HYBSF_GW\n#define OKC_HYBSF_GW 3\n#endif\n"
         "#ifndef OKC_HYBSF_NF\n#define OKC_HYBSF_NF 2\n#endif\n"
         "constexpr int NTL = NT + (OKC_WGS ? (OKC_HYBSF ? 32 + 32 * OKC_HYBSF_GW : 64) : OKC_WS ? 32 : 0);\n")
PAR_A = "__nv_bfloat16* out;\nunsigned long long* dbg;\nfloat* sspart;\n};\n"
PAR_N = "__nv_bfloat16* out;\nunsigned long long* dbg;\nfloat* sspart;\n#if OKC_HYBSF\nint sfnf;\n#endif\n};\n"
# weight producer: released at route end (ranges are already in smem: consumer warp 0 computed them before bar.arrive 7)
WP_A = ("__device__ void wgs_ranges(const Params& p, Smem& S);\n"
        "__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\nWS_JOIN();\nif ((threadIdx.x & 31) != 0) return;\n")
WP_N = r'''__device__ void wgs_ranges(const Params& p, Smem& S);
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_HYBSF
asm volatile("bar.sync 7, %0;\n" ::"n"(NTL) : "memory");
#else
WS_JOIN();
#endif
#if OKC_HYB_MK
if ((threadIdx.x & 31) == 1 && p.dbg && (S.wgr[1] > S.wgr[0] || S.wgr[3] > S.wgr[2])) { mbar_wait(S.bars + BW_F, 0u); MK(8); mbar_wait(S.bars + BX_F, 0u); MK(9); }
#endif
if ((threadIdx.x & 31) != 0) return;
'''
# gather warps (HYB layout; one call site, inlined) + the x producer starting at stage nf
GW_A = "constexpr int WGS_XROWS = EMAX + 8 * JMAX;\n"
GW_N = GW_A + r'''#if OKC_HYBSF
static_assert(OKC_HYBSF_GW >= 1 && OKC_HYBSF_GW <= 3, "gather warps: 20 warps keep 96 registers, 21 drop to 80 (6 warps on one SM quadrant)");
__device__ __forceinline__ void wgs_gather_warp(const Params& p, Smem& S, int part) {
const int lane = threadIdx.x & 31, nj = S.njF + S.njS;
const int R = S.jrow[nj];
const float rG = __fdividef(1.f, (float)gridDim.x);
const int qa = muldiv_g(R, blockIdx.x, gridDim.x, rG), qb = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
const int nq = qb - qa, q0 = qa + (nq * part) / OKC_HYBSF_GW, q1 = qa + (nq * (part + 1)) / OKC_HYBSF_GW;
if (q0 >= q1) return;
int j0 = 0;
for (int b = 0; b < nj; b += 32) { const int jj = b + lane; j0 += __popc(__ballot_sync(0xffffffffu, jj < nj && S.jrow[jj + 1] <= q0)); }
const int perm = 8 * ((lane >> 1) & 3) + 2 * (lane >> 3) + (lane & 1);
int j = j0;
#pragma unroll 1
for (int q = q0; q < q1; ++q) {
while (S.jrow[j + 1] <= q) ++j;
const int r = q - S.jrow[j], cnt = S.jC[j];
if (r >= cnt) continue;
const int ntt = (cnt + 7) >> 3;
const unsigned* src = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + perm;
uint4* dst = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((r >> 3) * 32 + lane) * 8) + (r & 7));
uint4 v[16];
#pragma unroll
for (int m = 0; m < 16; ++m) { const unsigned* s = src + m * 128; v[m] = make_uint4(__ldg(s), __ldg(s + 32), __ldg(s + 64), __ldg(s + 96)); }
#pragma unroll
for (int m = 0; m < 16; ++m) dst[(size_t)m * ntt * 256] = v[m];
}
asm volatile("fence.proxy.async.global;\n" ::: "memory");
__syncwarp();
if (lane == 0) {
int jj = j0;
for (int q = q0; q < q1;) {
while (S.jrow[jj + 1] <= q) ++jj;
const int e = min(q1, S.jrow[jj + 1]);
asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
q = e;
}
}
}
#endif
'''
XP_A = r'''__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
__syncwarp();
WS_JOIN();
if ((threadIdx.x & 31) != 0) return;
unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
int xs = 0, n = 0;
uint32_t xph = 0;
#pragma unroll 1
for (int pb = 0; pb < 2; ++pb) {
const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
const int ubl = pb ? 4 : 5;
const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
int curj = -1, npad = 8;
const __nv_bfloat16* jb = xsrc;
#pragma unroll 1
for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
if (gs == g1) gs = g0;
'''
XP_N = r'''__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_HYBSF
if ((threadIdx.x & 31) != 0) return;
const int nf0 = min(p.sfnf, S.wgr[1] - S.wgr[0]);
#else
asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
if ((threadIdx.x & 31) == 0) { wgs_ranges(p, S); MK(5); }
__syncwarp();
WS_JOIN();
if ((threadIdx.x & 31) != 0) return;
const int nf0 = 0;
#endif
unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
int xs = nf0 % WGS_NX, n = nf0;
uint32_t xph = (uint32_t)(nf0 / WGS_NX) & 1u;
#pragma unroll 1
for (int pb = 0; pb < 2; ++pb) {
const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
const int ubl = pb ? 4 : 5;
const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
int curj = -1, npad = 8;
const __nv_bfloat16* jb = xsrc;
#if OKC_HYBSF
const int i0 = pb ? 0 : nf0;
int gs0 = sp + i0;
if (gs0 >= g1) gs0 -= g1 - g0;
#else
const int i0 = 0, gs0 = sp;
#endif
#pragma unroll 1
for (int i = i0, gs = gs0; i < g1 - g0; ++i, ++gs, ++n) {
if (gs == g1) gs = g0;
'''
XI_A = HP.MK_XI_A
XI_N = XI_A.replace("if (++xs", "if (n == nf0) MK(7);\nif (++xs")
WI_A = HP.MK_WI_A.replace(" L2H_ARG ", " HN_ARG ")   # k3_hh_5 carries hyb193's hn operand on the weight copy
WI_N = HP.MK_WI_N.replace(" L2H_ARG ", " HN_ARG ")
RW_A = ("__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {\nconst int nj = S.njF + S.njS;\nwgs_prefix(S, nj);\n"
        "wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);\nwgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n}\n")
RW_N = RW_A + r'''#if OKC_HYBSF
__device__ __forceinline__ void wgs_ranges_warp(Smem& S) {
const int lane = threadIdx.x & 31, nj = S.njF + S.njS;
uint32_t carry = 0, carryB = 0;
for (int j0 = 0; j0 < nj; j0 += 32) {
const int j = j0 + lane;
const uint32_t v = j < nj ? (uint32_t)wgs_cost(0, S.jC[j]) << 9 : 0u, vB = j < nj ? (uint32_t)wgs_cost(1, S.jC[j]) << 8 : 0u;
uint32_t in = v, inB = vB;
#pragma unroll
for (int d = 1; d < 32; d <<= 1) {
const uint32_t t = __shfl_up_sync(0xffffffffu, in, d), tB = __shfl_up_sync(0xffffffffu, inB, d);
if (lane >= d) { in += t; inB += tB; }
}
if (j < nj) { S.wcp[j] = (int)(carry + in - v); S.wcpB[j] = (int)(carryB + inB - vB); }
carry += __shfl_sync(0xffffffffu, in, 31);
carryB += __shfl_sync(0xffffffffu, inB, 31);
}
if (lane == 0) { S.wcp[nj] = (int)carry; S.wcpB[nj] = (int)carryB; S.wtot = carry; S.wtotB = carryB; }
__syncwarp();
if (lane < 2) wgs_range(S, nj, lane, blockIdx.x, gridDim.x, S.wgr[2 * lane], S.wgr[2 * lane + 1]);
__syncwarp();
}
#endif
'''
# producer-side dispatch in moe_fused
DISP_A = ("if (tid >= NT) {\nif (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n"
          "else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\nreturn;\n}\n")
DISP_N = r'''if (tid >= NT) {
if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
#if OKC_HYBSF
else {
asm volatile("bar.sync 7, %0;\n" ::"n"(NTL) : "memory");
wgs_gather_warp(p, S, (tid - NT - 32) >> 5);
if ((tid & 31) == 0) MK(19 + ((tid - NT - 32) >> 5));
if (tid < NT + 64) wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
}
#else
else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
#endif
return;
}
'''
JROW = r'''if (tid < 32) {
const int nj = S.njF + S.njS;
int carry = 0;
for (int j0 = 0; j0 < nj; j0 += 32) {
const int j = j0 + tid;
const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
int incl = v;
#pragma unroll
for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }
if (j < nj) S.jrow[j] = carry + incl - v;
carry += __shfl_sync(0xffffffffu, incl, 31);
}
if (tid == 0) S.jrow[nj] = carry;
}
'''
SSZ = r'''{
float* other = (S.gen & 1) ? p.ss : p.ss1;
for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
}
'''
INFO = ('if (OKC_HYB_MK && p.dbg) p.dbg[blockIdx.x * 64 + 31] = (unsigned long long)(S.wgr[1] - S.wgr[0]) '
        '| ((unsigned long long)(S.wgr[3] - S.wgr[2]) << 16) | ((unsigned long long)(S.njF + S.njS) << 32) '
        '| ((unsigned long long)S.jrow[S.njF + S.njS] << 48);')
PRO_A = ('route(p, S);\n__threadfence_block();\nasm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 32) : "memory");\nDBG(1);\n' + SSZ + JROW
         + "CTA_SYNC();\nwgs_gather(p, S);\nDBG(7);\nWS_JOIN();\nDBG(6);\n")
PRO_N = ('route(p, S);\n#if OKC_HYBSF\n' + JROW.replace("if (tid == 0) S.jrow[nj] = carry;\n}\n",
                                                        "if (tid == 0) S.jrow[nj] = carry;\nwgs_ranges_warp(S);\nif (tid == 0) MK(5);\n}\n")
         + '#endif\n__threadfence_block();\nasm volatile("bar.arrive 7, %0;\\n" ::"n"(OKC_HYBSF ? NTL : NT + 32) : "memory");\nDBG(1);\n'
         'if (tid == 0) MK(2);\n' + SSZ + '#if !OKC_HYBSF\n' + JROW
         + "CTA_SYNC();\nif (tid == 0) MK(18);\nwgs_gather(p, S);\nDBG(7);\nif (tid == 0) MK(3);\nWS_JOIN();\nDBG(6);\n"
         "if (tid == 0) { MK(4); " + INFO + " }\n#else\nCTA_SYNC();\nif (tid == 0) MK(4);\n{\n"
         r'''const int lane = tid & 31, g0 = S.wgr[0], g1 = S.wgr[1], sp = wgs_split(0, g0, g1), nf = min(p.sfnf, g1 - g0);
unsigned char* xring = reinterpret_cast<unsigned char*>(xbuf) + WGS_NS * UNIT_BYTES;
const int perm = 8 * ((lane >> 1) & 3) + 2 * (lane >> 3) + (lane & 1);
#pragma unroll 1
for (int f = 0; f < nf; ++f) {
int gs = sp + f;
if (gs >= g1) gs -= g1 - g0;
const int u = gs >> 4, st = gs & 15, j = u >> 5, cnt = S.jC[j], jst = S.jS[j];
uint4* slot = reinterpret_cast<uint4*>(xring + f * WGS_XS);
#pragma unroll 1
for (int w0 = tid; w0 < cnt * 32; w0 += NT * 4) {
uint4 v[4];
#pragma unroll
for (int k = 0; k < 4; ++k) {
const int w = w0 + k * NT;
if (w < cnt * 32) {
const unsigned* s = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[jst + (w >> 5)] >> 3) * HD + st * 256) + perm;
v[k] = make_uint4(__ldg(s), __ldg(s + 32), __ldg(s + 64), __ldg(s + 96));
}
}
#pragma unroll
for (int k = 0; k < 4; ++k) {
const int w = w0 + k * NT, r = w >> 5;
if (w < cnt * 32) slot[((r >> 3) * 32 + lane) * 8 + (r & 7)] = v[k];
}
}
}
if (tid == 0) MKB(23);
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
CTA_SYNC();
if (tid == 0) {
for (int f = 0; f < nf; ++f) mbar_arrive1(S.bars + BX_F + f);
MK(3);
''' + INFO + "\n}\n}\n#endif\n")
# host side: run-time nf (lab) / compile-time default (ship form)
HOST_A = "static int g_okc_pdl = 0;\nvoid okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n"
HOST_N = HOST_A + "#if OKC_HYBSF\nstatic int g_okc_sfnf = OKC_HYBSF_NF;\nvoid okc_set_sfnf(int64_t n) { g_okc_sfnf = (int)n; }\n#endif\n"
HOSTP_A = "p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n"
HOSTP_N = HOSTP_A + "#if OKC_HYBSF\np.sfnf = g_okc_sfnf < 0 ? 0 : g_okc_sfnf > okc::WGS_NX ? okc::WGS_NX : g_okc_sfnf;\n#endif\n"
EDITS = [(DEF_A, DEF_N), (NTL_A, NTL_N), (PAR_A, PAR_N), (HP.MK_WAIT_A, HP.MK_WAIT_N), (WP_A, WP_N), (GW_A, GW_N), (XP_A, XP_N),
         (XI_A, XI_N), (WI_A, WI_N), (RW_A, RW_N), (DISP_A, DISP_N), (PRO_A, PRO_N), (HOST_A, HOST_N), (HOSTP_A, HOSTP_N)] + HP.MK_END

# ---------------- Python glue (k3_hh_5 has the hyb193 structure: _HYB["mod"] + _HYB["mod7"] at T >= HYB_T7) ----------------------
CALL_A = '''def _hyb_call(x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    md, k = (_HYB["mod7"], "pdl7") if (m >= HYB_T7 and _HYB["mod7"] is not None) else (_HYB["mod"], "pdl")
'''
CALL_N = '''def _hyb_call(x, ids, wts):
    m = x.shape[0]
    if HYBP_ST:
        H = _HYBP.get(71 if (HYBP_ST == 64 and m >= HYB_T7) else HYBP_ST)
        if H is not None and H["ok"]:
            return _hybp_run(H, x, ids, wts)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    md, k = (_HYB["mod7"], "pdl7") if (m >= HYB_T7 and _HYB["mod7"] is not None) else (_HYB["mod"], "pdl")
'''
HYB_LAB = '''HYBP_ST = 0
HYBP_NF = 2
_HYBP_KEYS = (64, 71, 80)
_HYBP = {k: {"mod": None, "ok": False, "sc": None, "full": None, "pdl": -1, "nf": -1, "build_s": 0.0, "err": None, "dbg": None}
         for k in _HYBP_KEYS}
_HYBP_CPP = _HYB_MOE_CPP + "void okc_set_sfnf(int64_t n);\\n"
def _hybp_src(k):
    s = _HYB_MOE_CUDA
    for a, n in _HYBP_EDITS:
        assert s.count(a) == 1, a[:60]
        s = s.replace(a, n)
    fl = list(HYB_FLAGS) + (list(HYB_F7) if (k & 7) == 7 else []) + (["-DOKC_HYB_MK=1"] if k & 16 else []) + (["-DOKC_HYBSF=1"] if k & 64 else [])
    return s, fl
def _hybp_build():
    from torch.utils.cpp_extension import load_inline
    def one(k):
        H = _HYBP[k]
        try:
            src, fl = _hybp_src(k)
            tag = hashlib.sha1((_HYBP_CPP + src + " ".join(fl)).encode()).hexdigest()[:12]
            t0 = time.time()
            H["mod"] = load_inline(name=f"hyb_moe_{tag}", cpp_sources=_HYBP_CPP, cuda_sources=src,
                                   functions=["okc_run", "okc_set_pdl", "okc_set_sfnf"], extra_cuda_cflags=fl, verbose=False)
            H["build_s"] = time.time() - t0
        except Exception as ex:
            H["err"] = repr(ex)[-600:]
    ths = [threading.Thread(target=one, args=(k,)) for k in _HYBP_KEYS]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
def _hybp_run(H, x, ids, wts):
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYB_PDL else 0
    if H["pdl"] != pdl:
        H["mod"].okc_set_pdl(pdl)
        H["pdl"] = pdl
    if H["nf"] != HYBP_NF:
        H["mod"].okc_set_sfnf(int(HYBP_NF))
        H["nf"] = int(HYBP_NF)
    sc = H["sc"] + [H["dbg"]] if H.get("dbg") is not None else H["sc"]
    H["mod"].okc_run(_HYB["w"], sc, x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
def _hybp_conc(T, dev, seed=777):
    ids = torch.full((T, 8), -1, dtype=torch.int32)
    for t in range(69):
        ids[t, 0] = (0, 7, 19)[t % 3]
    g = torch.Generator(device="cpu").manual_seed(4241 * seed + T)
    tw = torch.rand((T, 8), generator=g) + 0.05
    return (torch.randn((T, 4096), generator=g).to(torch.bfloat16).to(dev).contiguous(), ids.to(dev).contiguous(),
            (tw / tw.sum(-1, keepdim=True)).float().to(dev).contiguous())
def _hybp_ship(m, x, ids, tw, k):
    md = _HYB["mod7"] if ((k & 7) == 7 and _HYB["mod7"] is not None) else _HYB["mod"]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    md.okc_run(_HYB["w"], _HYB["sc"], x, ids, tw, out, 0, 32, _HYB["nsm"])
    return out
def _hybp_prepare(st, w, dev):
    global HYBP_NF
    _pt1_wait("hybp")
    cases = []
    for bs, seed, dist in HYB_SELF_TEST:
        d = _k2a_self_test_inputs(bs, seed, dev, dist)
        cases.append((f"{bs}{dist[0]}", d["hidden_states"].contiguous(), d["topk_ids"].to(torch.int32).contiguous(),
                      d["topk_weights"].float().contiguous()))
    cases.append(("conc69",) + _hybp_conc(128, dev))
    refs = {}
    for name, x, ids, tw in cases:
        d = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw, "config": {"local_start": 0, "local_count": K2A_E_LOC}}
        refs[name] = _k2a_ref_fp32(d, w)
    nf_save = HYBP_NF
    for k, H in _HYBP.items():
        H["ok"] = False
        if H["mod"] is None:
            print(f"[hybp] HYB st{k}: build failed {H['err']}", flush=True)
            continue
        try:
            H["sc"], H["full"] = _hyb_scratch(dev)
            H["dbg"] = torch.zeros(_HYB_SLOTS * 64, dtype=torch.int64, device=dev) if k & 16 else None
            torch.cuda.synchronize()
            good, worst, nbit, ncase = True, [0.0, 0.0, 0.0, 0.0], 0, 0
            for nf in (2, 0, 3, 1):
                HYBP_NF = nf
                for name, x, ids, tw in cases:
                    if nf != 2 and name not in ("128s", "256n", "conc69"):
                        continue
                    m = x.shape[0]
                    o3 = _hybp_ship(m, x, ids, tw, k).clone()
                    o4 = _hybp_ship(m, x, ids, tw, k).clone()
                    oa = _hybp_run(H, x, ids, tw).clone()
                    ob = _hybp_run(H, x, ids, tw).clone()
                    torch.cuda.synchronize()
                    rep3 = float((o4.float() - o3.float()).abs().max())
                    tot, row = _k2a_rel_err(oa, refs[name])
                    ev = _k2a_rel_err(oa, o3)[0]
                    dd = float((oa.float() - o3.float()).abs().max())
                    rp = _k2a_rel_err(ob, oa)[0]
                    fin = bool(torch.isfinite(oa.float()).all() and torch.isfinite(ob.float()).all())
                    ok = fin and tot < 1e-2 and row < 5e-2 and ev < 2e-3 and rp < 1e-3
                    nbit += dd == 0.0
                    ncase += 1
                    worst = [max(worst[0], tot), max(worst[1], row), max(worst[2], ev), max(worst[3], rp)]
                    if not ok or dd != 0.0:
                        print(f"[hybp] HYB st{k} nf{nf} self-test {name}: vs FP32 {tot:.2e} row {row:.2e} vs ship {ev:.2e} (max abs {dd:.2e}, "
                              f"ship repeat {rep3:.1e}) repeat {rp:.1e} finite {fin} -> {'ok' if ok else 'FAIL'}", flush=True)
                    good &= ok
            gok = all(bool((f[n:].view(vt) == sv).all()) for f, n, vt, sv in H["full"])
            H["ok"] = bool(good and gok)
            print(f"[hybp] HYB st{k} {'LIVE' if H['ok'] else 'OFF'}: {ncase} cases ({nbit} bitwise == ship) worst vs FP32 "
                  f"{worst[0]:.2e} row {worst[1]:.2e} vs ship {worst[2]:.2e} repeat {worst[3]:.1e}, guards {gok}, nvcc {H['build_s']:.0f}s",
                  flush=True)
        except Exception as ex:
            H["ok"] = False
            print(f"[hybp] HYB st{k} prepare raised {ex!r} -> OFF", flush=True)
    HYBP_NF = nf_save
'''
PREP_END = HP.HYB_PREP_END
PREP_END_N = HP.HYB_PREP_END_NEW
PT1_HYB, PT1_HYB_N = HP.PT1_HYB, HP.PT1_HYB_NEW
NOTE_A = '"""\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n'
NOTE_LAB = ("Modified by team joeyhlu (2026-10-31, tools/hybsf_patches.py, lab only): optional builds with per-CTA timing marks\n"
            "(-DOKC_HYB_MK) and with HYB-SF (-DOKC_HYBSF: the consumers fill the first x slots from x themselves, gather warps\n"
            "write xA, the producers start at route end); HYB's arithmetic and ranges are unchanged.\n" + NOTE_A)
NOTE_SHIP = ("Modified by team joeyhlu (2026-10-31, tools/hybsf_patches.py, -DOKC_HYBSF): the consumers fill the first x slots from x\n"
             "themselves (VML's bit-6 structure), dedicated gather warps write xA, and both producers start at route end;\n"
             "HYB's arithmetic and ranges are unchanged.\n" + NOTE_A)


def apply_cuda(cu):
    for a, n in EDITS:
        check(cu.count(a) == 1, f"CUDA anchor not unique ({cu.count(a)}x): {a[:70]!r}")
        cu = cu.replace(a, n)
    check(cu.count("MK(") + cu.count("MKB(") >= 22 and "wgs_gather_warp(p, S," in cu and "okc_set_sfnf" in cu, "SF inserted")
    return cu


def apply_lab(s):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = "# nonce k3-hybsf-ab (k3_hh_5 + R-hybsf lab arms: HYBP_ST 64 = HYB-SF (71 at T >= 193), 80 = SF + marks; tools/hybsf_patches.py)"
    s = NL.join(lines)
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    apply_cuda(s[i0:i1])   # dry run: the file keeps the ship CUDA text; _hybp_src applies _HYBP_EDITS at run time
    s = sub(s, CALL_A, CALL_N, "_hyb_call")
    s = sub(s, "def _hyb_on(m):\n", HYB_LAB + f"_HYBP_EDITS = {EDITS!r}\n" + "def _hyb_on(m):\n", "HYBP lab block")
    s = sub(s, PREP_END, PREP_END_N, "_hyb_prepare end")
    s = sub(s, PT1_HYB, PT1_HYB_N, "PT1 hyb job")
    s = sub(s, NOTE_A, NOTE_LAB, "HYB notice end")
    return s


import re

MK_LINE = re.compile(r"^(if \(tid == 0\) |if \(\(threadIdx\.x & 31\) == 0\) |if \(\(tid & 31\) == 0\) |if \(n == nf0\) |if \(n == 0\) |else if \(n == WGS_NS\) )?MKB?\(\d+( \+ [^;]*)?\);$")
INFO_LINE = re.compile(r"^if \(OKC_HYB_MK && p\.dbg\) p\.dbg\[.*;$")
PREFIX_FN = ("__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {\nuint32_t acc = 0, accB = 0;\nfor (int j = 0; j < nj; ++j) {\n"
             "S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;\nacc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;\naccB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;\n}\n"
             "S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;\nS.wtot = acc; S.wtotB = accB;\n}\n")
WS_JOIN_DEF = '#define WS_JOIN() asm volatile("bar.sync 2, %0;\\n" ::"n"(NTL) : "memory")\n'


def slim_sf(cu):
    """Ship form: resolve the OKC_HYBSF / OKC_HYB_MK blocks (SF on, marks off), drop the marks macros / statements and the functions
    the SF path no longer calls (wgs_gather, wgs_ranges, wgs_prefix, WS_JOIN). Same preprocessed text as -DOKC_HYBSF=1 -DOKC_HYB_MK=0."""
    out, stack = [], []   # stack entries: None (foreign #if, kept verbatim) or bool (emit state of one of our blocks)
    for ln in cu.split(NL):
        t = ln.strip()
        if t in ("#if OKC_HYBSF", "#if !OKC_HYBSF", "#if OKC_HYB_MK"):
            stack.append(t == "#if OKC_HYBSF")
            continue
        if t.startswith("#if") and stack and stack[-1] is not None and not all(x for x in stack if x is not None):
            check(False, f"nested foreign #if inside a dropped SF block: {t}")
        if t.startswith(("#if", "#ifdef", "#ifndef")):
            stack.append(None)
            out.append(ln)
            continue
        if t == "#else" and stack and stack[-1] is not None:
            stack[-1] = not stack[-1]
            continue
        if t == "#endif" and stack and stack[-1] is not None:
            stack.pop()
            continue
        if t in ("#else", "#endif"):
            if t == "#endif":
                stack.pop()
            out.append(ln)
            continue
        if all(x for x in stack if x is not None):
            out.append(ln)
    check(not stack, "unbalanced #if in the HYB CUDA text")
    cu = NL.join(out)
    cu = NL.join(ln for ln in cu.split(NL) if not (MK_LINE.match(ln.strip()) or INFO_LINE.match(ln.strip()) or ln.startswith(("#define MK(k)", "#define MKB(k)"))))
    cu = sub(cu, "#ifndef OKC_HYB_MK\n#define OKC_HYB_MK 0\n#endif\n", "", "OKC_HYB_MK default")
    cu = sub(cu, "#ifndef OKC_HYBSF\n#define OKC_HYBSF 0\n#endif\n", "#ifndef OKC_HYBSF\n#define OKC_HYBSF 1\n#endif\n", "OKC_HYBSF default")
    g0, g1 = cu.index("__device__ void wgs_gather(const Params& p, Smem& S) {\n"), cu.index("struct WgsPend { int kind, u, slot, nst; };\n")
    check(0 < g0 < g1 and cu.count("wgs_gather(p, S);") == 0, "wgs_gather still called")
    cu = cu[:g0] + cu[g1:]
    cu = sub(cu, RW_A, "", "wgs_ranges definition")
    cu = sub(cu, "__device__ void wgs_ranges(const Params& p, Smem& S);\n", "", "wgs_ranges declaration")
    cu = sub(cu, PREFIX_FN, "", "wgs_prefix definition")
    cu = sub(cu, WS_JOIN_DEF, "", "WS_JOIN macro")
    for w in ("MK(", "MKB(", "WS_JOIN", "wgs_gather(", "wgs_ranges(", "wgs_prefix(", "OKC_HYB_MK", "#if !OKC_HYBSF"):
        check(w not in cu, f"ship form still contains {w!r}")
    return cu


def apply_ship(s, nf, gw):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] += f" + R-hybsf HYB-SF nf {nf} gw {gw} [tools/hybsf_patches.py]"
    s = NL.join(lines)
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    s = s[:i0] + slim_sf(apply_cuda(s[i0:i1])) + s[i1:]
    a_flags = next(ln for ln in s.split(NL) if ln.startswith("HYB_FLAGS = ("))
    check(a_flags.endswith(")") and s.count(a_flags + NL) == 1, "HYB_FLAGS line")
    n_flags = a_flags[:-1] + f", '-DOKC_HYBSF=1', '-DOKC_HYBSF_NF={nf}', '-DOKC_HYBSF_GW={gw}')"
    s = sub(s, a_flags + NL, n_flags + NL, "HYB_FLAGS")
    s = sub(s, NOTE_A, NOTE_SHIP, "HYB notice end")
    return s


def gates(path, s):
    import pyslim
    n = len(s.encode())
    check(n < LIMIT or "_ab" in path.name or "lab" in path.name, f"size {n} B >= {LIMIT} (lab files may exceed the cap; a ship form may not)")
    check(s.isascii(), "non-ASCII byte")
    ast.parse(s)
    d = pyslim.decorated_sources(s)
    bad = [x for x in d if x.startswith("UNPARSABLE")]
    check(not bad, f"decorated defs that do not re-parse from their own text: {bad[:2]}")
    print(f"hybsf_patches: gates OK on {path.name}: {n:,} B ({LIMIT - n:,} B under the cap{'; LAB ONLY' if n >= LIMIT else ''}), {len(d)} decorated defs re-parse")


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_5.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybsf_ab.py")
    src = base.read_bytes().decode("ascii")
    if "ship" in FLAGS:
        nf, gw = int(ARGS.get("nf", "2")), int(ARGS.get("gw", "3"))
        check(0 <= nf <= 3 and 1 <= gw <= 3, "--nf 0..3, --gw 1..3 (4 gather warps = 21 warps -> 6 on one SM quadrant -> 80 registers + spills)")
        new = apply_ship(src, nf, gw)
    else:
        new = apply_lab(src)
    if "check" in FLAGS:
        check(out.exists(), f"{out} missing")
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        gates(out, new)
        print(f"hybsf_patches: --check OK ({out.name} == apply({base.name}))")
        return
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"hybsf_patches: {base.name} -> {out.name} ({len(new):,} B, +{len(new) - len(src):,} B)")
    gates(out, new)


if __name__ == "__main__":
    main()

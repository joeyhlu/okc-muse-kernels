"""dr_patches.py: VM_DR = the T32 double-row variant of the VML MoE kernel (side_moe_20261028/RESEARCH.md, "T32 double-row
design"), as compile-time patches of k3_gc_6's `_vml_CUDA_T` (every change behind `#if VM_DR`; VM_DR 0 compiles to the
same code, checked by `python -I tools/dr_patches.py check` with the C preprocessor).

Geometry (T <= 32 bucket only, jobs of <= 32 tokens = -DVM_JN=32 -DVM_XSN=32 on top of bf2's switches):
  logical unit = 512 weight rows = incumbent units 2U, 2U + 1 (phase A: 16 units x 16 stages per job, B: 8 x 16);
  one weight slot per 256-K stage = the four K2A 128-row blocks 4U .. 4U + 3 (4 x 18,432 = 73,728 B incl. scales), NS = 2;
  one x slot per stage (<= 32 tokens x 512 B = 16 KB), NX = 3, read by both tiles of every warpgroup;
  WG q owns block 4U + q: tile t = its rows 64t .. 64t + 63 (warp-tiles 4t + wp), accumulator sets d0 / d1.
  Rings: 2 x 73,728 + 3 x 16,384 = 196,608 B (+ barriers + tables = ~200.6 KB of the 227 KB per block).
Per tile the k16 order, zero-init, mma.sync chain split and FP32 accumulation are the incumbent's, so each tile's sums
are bitwise the incumbent's for the same rows / tokens / K range. Phase A releases doneA[job][U] += 2 per WG after
both tiles' SwiGLU stores (4 WGs x 2 = the incumbent's 8 credits per B stage; target and vm_fin reset unchanged).
Phase-B split shares red.add into y (VM_BRED, required); phase-A split shares use the incumbent fixup slots (2 x N/8
float4 blocks per thread <= FIXW). Weights: VM_DRP 0 = the K2A layout, 4 bulk copies per stage; VM_DRP 1 = the
lossless dr_repack() copy ([E][nrb/4][16][4][SW], built in prepare), one 73,728 B bulk copy per stage.
VM_DRALL (8): mma.sync stages with N <= VM_DRALL load all code words before the first mma and free the weight slot at
once (incumbent stage_m order); larger N load codes one 4-step group ahead and free the slot at group 3 (register cap).

    python -I tools/dr_patches.py lab      # writes kernels/lab/dr_moe.py (lab module: build / alloc / call / repack)
    python -I tools/dr_patches.py check    # anchors unique + VM_DR=0 preprocesses to the unpatched text (needs cpp)
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_gc_6.py"
LAB_DST = ROOT / "kernels/lab/dr_moe.py"
NL = chr(10)

BF2 = ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1")
DR = BF2 + ("-DVM_JN=32", "-DVM_XSN=32", "-DVM_DR=1")
DRP = DR + ("-DVM_DRP=1",)
C104 = ("-DVM_WG5=2", "-DVM_PREG=64", "-DVM_CREG=104", "-DVM_GV=2")   # setmaxnreg split: 128 x 64 + 512 x 104 = 640 x 96
DEFS = {"bf2": BF2, "dr": DR, "drp": DRP, "bf2c": BF2 + C104, "drc": DR + C104, "drpc": DRP + C104}


def drif(old, new):
    """(anchor, `#if VM_DR` new `#else` anchor `#endif`)."""
    return old, "#if VM_DR" + NL + new + "#else" + NL + old + "#endif" + NL


# ---------------------------------------------------------------- the DR consumer (inserted before phase<PB>)
DR_FN = r"""#if VM_DR
static_assert(VM_JN == 32 && VM_XSN == 32 && VM_NTM == 4 && VM_BRED && VM_SYNC7 && VM_NOB9 && VM_TRICKS && !VM_UD, "VM_DR config");
static_assert(!VM_DP && !VM_SC && !VM_AD && !VM_PS && !VM_DEFER && !VM_DEFER2 && VM_MKW == 32, "VM_DR: unsupported switch");
// RS-wgmma stage (N 24 / 32): commit groups alternate tiles (i, hf, t), so hb[t] is tile t's A buffer and wait_group 1
// retires the other tile's previous group; the weight slot is freed once both tiles' last code words are in registers.
template <int N>
__device__ __forceinline__ void stage_dr(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d0, float* d1,
uint32_t (&hb)[2][8], int q, int wp, int lane, bool first) {
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + q * SW;
const uint4* cp0 = reinterpret_cast<const uint4*>(st + wp * 2048) + lane;
const uint4* cp1 = cp0 + 512;
const uint32_t* sp0 = reinterpret_cast<const uint32_t*>(st + 16384 + wp * 256) + lane;
const uint32_t* sp1 = sp0 + 256;
uint4 c0[4], c1[4];
c0[0] = cp0[0]; c1[0] = cp1[0];
const uint32_t s00 = sp0[0], s01 = sp0[32], s10 = sp1[0], s11 = sp1[32];
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
const uint32_t zo = (uint32_t)A.flags >> 30;
const uint32_t k16 = 16u + zo, k64 = VM_TRICKS_DQ ? 64u + zo : 64u, k4 = VM_TRICKS_DQ ? 4u + zo : 4u;
const uint64_t dd = wg_desc_sa(sa(smem + NS * US), 4096u);
const uint32_t dlo = (uint32_t)dd + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(dd >> 32);
#pragma unroll
for (int i = 0; i < 4; ++i) {
const uint32_t par = i & 1;
if (i < 3) { c0[i + 1] = cp0[32 * (i + 1)]; c1[i + 1] = cp1[32 * (i + 1)]; }
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
for (int t = 0; t < 2; ++t) {
uint32_t* hh = hb[t];
const uint4 cc = t ? c1[i] : c0[i];
const uint32_t sw2 = t ? ((i < 2) ? s10 : s11) : ((i < 2) ? s00 : s01);
dec_half2(hf ? cc.y : cc.x, hf ? cc.w : cc.z, sw2, par, hh, k16, k64, k4);
if (i == 3 && hf == 0 && t == 1) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }
wg_fence();
if (first && i == 0 && hf == 0) mma2<N, true>(t ? d1 : d0, hh, dlo + 64 * i + 32 * hf, dhi);
else mma2<N, false>(t ? d1 : d0, hh, dlo + 64 * i + 32 * hf, dhi);
wg_commit();
wg_wait<1>();
if (i == 0 && hf == 0 && t == 0 && prevx >= 0) {
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
prevx = -1;
}
}
}
}
prevx = xs;
++it;
}
// mma.sync stage (N 8 / 16): per (i, hf) both tiles' A fragments, then ONE ldmatrix.x4 per 8-token tile feeds 4 mma.sync
// (two chains per tile, the incumbent's split).
template <int N>
__device__ __forceinline__ void stage_mdr(const Args& A, uint8_t* smem, uint32_t bars, int& it, float (&e0)[2][N / 2],
float (&e1)[2][N / 2], int q, int wp, int lane) {
constexpr bool ALL = N <= VM_DRALL;
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + q * SW;
const uint4* cp0 = reinterpret_cast<const uint4*>(st + wp * 2048) + lane;
const uint4* cp1 = cp0 + 512;
const uint32_t* sp0 = reinterpret_cast<const uint32_t*>(st + 16384 + wp * 256) + lane;
const uint32_t* sp1 = sp0 + 256;
uint4 c0[4], c1[4];
if constexpr (ALL) {
#pragma unroll
for (int i = 0; i < 4; ++i) { c0[i] = cp0[32 * i]; c1[i] = cp1[32 * i]; }
} else {
c0[0] = cp0[0]; c1[0] = cp1[0];
}
const uint32_t s00 = sp0[0], s01 = sp0[32], s10 = sp1[0], s11 = sp1[32];
if constexpr (ALL) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
const uint32_t xb = sa(smem + NS * US + xs * XS) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
const uint32_t k16q = 16u + ((uint32_t)A.flags >> 30);
#pragma unroll
for (int i = 0; i < 4; ++i) {
const uint32_t par = i & 1;
if constexpr (!ALL) { if (i < 3) { c0[i + 1] = cp0[32 * (i + 1)]; c1[i + 1] = cp1[32 * (i + 1)]; } }
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t h0[8], h1[8];
dec_half2(hf ? c0[i].y : c0[i].x, hf ? c0[i].w : c0[i].z, (i < 2) ? s00 : s01, par, h0, k16q, 64u, 4u);
dec_half2(hf ? c1[i].y : c1[i].x, hf ? c1[i].w : c1[i].z, (i < 2) ? s10 : s11, par, h1, k16q, 64u, 4u);
if constexpr (!ALL) { if (i == 3 && hf == 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); } }
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
uint32_t b0, b1, b2, b3;
ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
mma16816(&e0[0][4 * nt], h0[0], h0[4], h0[1], h0[5], b0, b1);
mma16816(&e0[1][4 * nt], h0[2], h0[6], h0[3], h0[7], b2, b3);
mma16816(&e1[0][4 * nt], h1[0], h1[4], h1[1], h1[5], b0, b1);
mma16816(&e1[1][4 * nt], h1[2], h1[6], h1[3], h1[7], b2, b3);
}
}
}
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
++it;
}
// both tiles of WG q in unit ub (512 rows): tile t = global warp-tile (4 ub + q) * 8 + 4 t + wp (the incumbent's tg of
// unit 2 ub + (q >> 1), WG 2 (q & 1) + t). Phase A: one release of 2 credits per WG after both tiles' stores.
template <bool PB, int N>
__device__ __forceinline__ void epilogue_dr(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d0,
const float* d1) {
const int g = lane >> 2, t = lane & 3, wt = (wp << 5) | lane;
const int e = T.jE[j], cn = T.jC[j];
if constexpr (!PB) {
#if VM_GSM
const float gs = T.gs13[e];
#else
const float gs = __ldg(A.gsc13 + e);
#endif
const int npad = (cn + 7) & ~7;
__nv_bfloat16* xb = A.xB + (size_t)T.jrow[j] * KDIM + (size_t)ub * npad * 256;
#pragma unroll
for (int tl = 0; tl < 2; ++tl) {
const float* d = tl ? d1 : d0;
const int kk = 8 * (((4 * ub + q) * 8 + 4 * tl + wp) & 31) + g;
const int i = kk >> 6, tt = (kk >> 4) & 3, qq = (kk >> 1) & 7, b = kk & 1;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn)
xb[((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + hh)) * 4 + tt) * 2 + b] =
__float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
}
}
fence_proxy_async();
bar_sync(1 + q, 128);
if (wt == 0) red_release_add(A.doneA + j * 16 + ub, 2);
} else {
#if VM_GSM
const float gs = T.gs2[e];
#else
const float gs = __ldg(A.gsc2 + e);
#endif
float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
for (int tl = 0; tl < 2; ++tl) {
const float* d = tl ? d1 : d0;
const int h0 = 16 * ((4 * ub + q) * 8 + 4 * tl + wp) + g;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn) {
yb[(size_t)n * KDIM + h0] = d[4 * nt + hh] * gs;
yb[(size_t)n * KDIM + h0 + 8] = d[4 * nt + 2 + hh] * gs;
}
}
}
}
}
template <int N>
__device__ __forceinline__ void bred_dr(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d0,
const float* d1) {
const int g = lane >> 2, t = lane & 3;
const int e = T.jE[j], cn = T.jC[j];
#if VM_GSM
const float gs = T.gs2[e];
#else
const float gs = __ldg(A.gsc2 + e);
#endif
float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
for (int tl = 0; tl < 2; ++tl) {
const float* d = tl ? d1 : d0;
const int h0 = 16 * ((4 * ub + q) * 8 + 4 * tl + wp) + g;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn) {
asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(yb + (size_t)n * KDIM + h0), "f"(d[4 * nt + hh] * gs) : "memory");
asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(yb + (size_t)n * KDIM + h0 + 8), "f"(d[4 * nt + 2 + hh] * gs) : "memory");
}
}
}
}
// stages [s0, s1) of one DR unit into the two tiles' FP32 sums (also the dr_probe consumer)
template <int N>
__device__ __forceinline__ void seg_dr_acc(const Args& A, uint8_t* smem, uint32_t bars, int& it, int s0, int s1, float* d0,
float* d1, int q, int wp, int lane) {
if constexpr (N <= 16) {
float e0[2][N / 2], e1[2][N / 2];
#pragma unroll
for (int k = 0; k < N / 2; ++k) { e0[0][k] = 0.f; e0[1][k] = 0.f; e1[0][k] = 0.f; e1[1][k] = 0.f; }
#pragma unroll 1
for (int s = s0; s < s1; ++s) stage_mdr<N>(A, smem, bars, it, e0, e1, q, wp, lane);
#pragma unroll
for (int k = 0; k < N / 2; ++k) { d0[k] = e0[0][k] + e0[1][k]; d1[k] = e1[0][k] + e1[1][k]; }
} else {
uint32_t hb[2][8];
int prevx = -1;
stage_dr<N>(A, smem, bars, it, prevx, d0, d1, hb, q, wp, lane, true);
#pragma unroll 1
for (int s = s0 + 1; s < s1; ++s) stage_dr<N>(A, smem, bars, it, prevx, d0, d1, hb, q, wp, lane, false);
wg_wait<0>();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
}
}
// one segment = stages [s0, s1) of DR unit u (A: u = job x 16 + ub, B: job x 8 + ub)
template <bool PB, int N>
__device__ __forceinline__ void seg_dr(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int u, int s0, int s1,
int q, int wp, int lane) {
float d0[N / 2], d1[N / 2];
seg_dr_acc<N>(A, smem, bars, it, s0, s1, d0, d1, q, wp, lane);
constexpr int UBL = PB ? 3 : 4;
const int j = u >> UBL, ub = u & ((1 << UBL) - 1);
if (!(s0 == 0 && s1 == 16)) {
if constexpr (PB) { bred_dr<N>(A, T, j, ub, q, wp, lane, d0, d1); return; }
const int wt = (wp << 5) | lane;
const int slot = cta_of(T, PB, u << 4, gridDim.x);
float* fb = A.fix + ((size_t)((PB * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + (PB * SLOTS + slot) * 4 + q;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
red_add_v4(fb + nt * 512, d0[4 * nt], d0[4 * nt + 1], d0[4 * nt + 2], d0[4 * nt + 3]);
red_add_v4(fb + (N / 8 + nt) * 512, d1[4 * nt], d1[4 * nt + 1], d1[4 * nt + 2], d1[4 * nt + 3]);
}
bar_sync(1 + q, 128);
if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);
bar_sync(1 + q, 128);
if (!T.wlast[q]) return;
fence_acq_rel();
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
const float4 w = __ldcg(reinterpret_cast<const float4*>(fb + (N / 8 + nt) * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
*reinterpret_cast<float4*>(fb + (N / 8 + nt) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d0[4 * nt] = v.x; d0[4 * nt + 1] = v.y; d0[4 * nt + 2] = v.z; d0[4 * nt + 3] = v.w;
d1[4 * nt] = w.x; d1[4 * nt + 1] = w.y; d1[4 * nt + 2] = w.z; d1[4 * nt + 3] = w.w;
}
if (wt == 0) *fc = 0;
}
epilogue_dr<PB, N>(A, T, j, ub, q, wp, lane, d0, d1);
}
#endif
"""

PHASE_ANCHOR = "template <bool PB>\n#if VM_DP\n__device__ __forceinline__ void phase("
_SEG_OLD = "seg<PB, 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n"
_DR_SWITCH = ("switch ((T.jC[j] + 7) >> 3) {\n"
              "case 1: seg_dr<PB, 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;\n"
              "case 2: seg_dr<PB, 16>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;\n"
              "case 3: seg_dr<PB, 24>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;\n"
              "default: seg_dr<PB, 32>(A, T, smem, bars, it, u, s0, s1, q, wp, lane); break;\n"
              "}\n(void)pd;\n")

CUDA_PATCHES = [
    ("constexpr int SW = 18432;\nconstexpr int US = 2 * SW;\n",
     "constexpr int SW = 18432;\n#ifndef VM_DR\n#define VM_DR 0\n#endif\n#ifndef VM_DRP\n#define VM_DRP 0\n#endif\n"
     "#ifndef VM_DRALL\n#define VM_DRALL 8\n#endif\n"
     "#if VM_DR\nconstexpr int US = 4 * SW;   // VM_DR: one 512-row stage (K2A blocks 4U .. 4U + 3) per weight slot\n#else\n"
     "constexpr int US = 2 * SW;\n#endif\n"),
    drif("constexpr int NS = 3, NX = VM_NX;\n", "constexpr int NS = 2, NX = VM_NX;\n"),
    ("constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);\n",
     "constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);\n#if VM_DR\nstatic_assert(SMEM <= 232448, \"VM_DR: rings + tables exceed 227 KB\");\n#endif\n"),
    # stream-K plan: a job has 16 x 16 (A) / 8 x 16 (B) DR stages instead of 32 x 16 / 16 x 16
    drif("if (c >= G) return nj << (9 - pb);\n", "if (c >= G) return nj << (8 - pb);\n"),
    drif("const uint32_t ns = 512u >> pb;\n", "const uint32_t ns = 256u >> pb;\n"),
    drif("return (lo << (9 - pb)) + (int)st;\n", "return (lo << (8 - pb)) + (int)st;\n"),
    drif("if (j < T.nj) { T.cpA[j] = accA + 512u * (uint32_t)sc; T.cpB[j] = accB + 256u * (uint32_t)sd; }\n"
         "accA += 512u * (uint32_t)__shfl_sync(0xffffffffu, sc + ca, 31);\n"
         "accB += 256u * (uint32_t)__shfl_sync(0xffffffffu, sd + cb, 31);\n",
         "if (j < T.nj) { T.cpA[j] = accA + 256u * (uint32_t)sc; T.cpB[j] = accB + 128u * (uint32_t)sd; }\n"
         "accA += 256u * (uint32_t)__shfl_sync(0xffffffffu, sc + ca, 31);\n"
         "accB += 128u * (uint32_t)__shfl_sync(0xffffffffu, sd + cb, 31);\n"),
    drif("const int j = gs >> 9, st = gs & 15, cn = T.jC[j];\n", "const int j = gs >> 8, st = gs & 15, cn = T.jC[j];\n"),
    # weight producer: the four 128-row blocks of DR unit ub, stage s
    drif("const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);\n"
         "const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);\nconst int e = T.jE[j];\n"
         "const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW\n"
         ": A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;\n",
         "const int j = pb ? (u >> 3) : (u >> 4), ub = pb ? (u & 7) : (u & 15);\nconst int s = gs & 15;\nconst int e = T.jE[j];\n"
         "#if VM_DRP\n"
         "const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 8 + ub) * 16 + s) * US : A.ws13 + ((size_t)(e * 16 + ub) * 16 + s) * US;\n"
         "#else\n"
         "const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 4 * ub) * 16 + s) * SW : A.ws13 + ((size_t)(e * 64 + 4 * ub) * 16 + s) * SW;\n"
         "#endif\n"),
    ("mbar_expect_tx(fb, US);\nbulk_g2s_ef(dst, src, SW, fb, pol);\nbulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);\n}\n"
     "#if VM_DP\n{\nTab& TT",
     "mbar_expect_tx(fb, US);\n#if VM_DR && VM_DRP\nbulk_g2s_ef(dst, src, US, fb, pol);\n#elif VM_DR\n#pragma unroll\n"
     "for (int b = 0; b < 4; ++b) bulk_g2s_ef(dst + b * SW, src + (size_t)(16 * b) * SW, SW, fb, pol);\n#else\n"
     "bulk_g2s_ef(dst, src, SW, fb, pol);\nbulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);\n#endif\n}\n"
     "#if VM_DP\n{\nTab& TT"),
    # x producer: same per-stage x slice (job, stage s); only the job index of a stage changes
    drif("const int j = pb ? (u >> 4) : (u >> 5);\nconst int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);\n",
         "const int j = pb ? (u >> 3) : (u >> 4);\nconst int s = gs & 15;\n"),
    (PHASE_ANCHOR, DR_FN + PHASE_ANCHOR),
    drif("const int j = PB ? (u >> 4) : (u >> 5);\n", "const int j = PB ? (u >> 3) : (u >> 4);\n"),
    ("switch ((T.jC[j] + 7) >> 3) {\ncase 1: " + _SEG_OLD,
     "#if VM_DR\n" + _DR_SWITCH + "#else\nswitch ((T.jC[j] + 7) >> 3) {\ncase 1: " + _SEG_OLD),
    ("default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n}\n#if VM_AD\n"
     "if constexpr (!PB) { if (pd.son && pd.stag == tag0)",
     "default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n}\n#endif\n#if VM_AD\n"
     "if constexpr (!PB) { if (pd.son && pd.stag == tag0)"),
    ("TORCH_CHECK((int64_t)V.size() >= 6 + 2 * VM_NTM, \"vml_run: cost table too short\");\n",
     "TORCH_CHECK((int64_t)V.size() >= 6 + 2 * VM_NTM, \"vml_run: cost table too short\");\n"
     "#if VM_DR\nTORCH_CHECK(M <= 192, \"vml_run: VM_DR (32-token jobs, JMAX jobs) supports M <= 192\");\n#endif\n"),
    drif("std::vector<int64_t> vml_cfg() { return {VM_JN, VM_NTM, VM_NX, VM_XSN, VM_MKW, (int64_t)vm::SMEM, VM_DP, VM_SC, VM_BRED, VM_DEFER2, VM_PS}; }\n",
         "std::vector<int64_t> vml_cfg() { return {VM_JN, VM_NTM, VM_NX, VM_XSN, VM_MKW, (int64_t)vm::SMEM, VM_DP, VM_SC, VM_BRED, VM_DEFER2, VM_PS, VM_DR, VM_DRP}; }\n"),
]

# ---------------------------------------------------------------- Python glue in a layer (names prefixed _vml_)
REPACK = '''

def _vml_dr_repack(ws):
    """VM_DRP lossless repack: K2A stage layout [E][nrb][16][SW] -> [E][nrb / 4][16][4][SW] (a 512-row stage contiguous)."""
    e, nrb, ns, sw = ws.shape
    return ws.view(e, nrb // 4, 4, ns, sw).permute(0, 1, 3, 2, 4).contiguous()
'''
DRP_VARS = ("drp", "drpc")   # variants reading the VM_DRP weight copy
_DEFLINES = "".join(f'             "{k}": {DEFS[k]!r},\n' for k in ("dr", "drp", "drc", "drpc", "bf2c"))
PY_PATCHES = [
    ("def _vml_cost_ext(mod, cost):\n    ntm = _vml_cfg(mod)[1]\n",
     "def _vml_cost_ext(mod, cost):\n    ntm = _vml_cfg(mod)[1]\n"
     "    if ntm == 4 and len(cost) == 16:   # VM_DR (32-token jobs): n-tiles 1..4 of each phase's 8-entry table\n"
     "        return list(cost[:4]) + list(cost[8:12])\n"),
    ('"-DVM_JN=72", "-DVM_XSN=72")}\n', '"-DVM_JN=72", "-DVM_XSN=72"),\n' + _DEFLINES.rstrip(",\n") + "}\n"),
    ('    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n',
     '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72", "dr", "drp", "drc", "drpc", "bf2c") else _VML["sc"]\n'
     f'    if var in {DRP_VARS!r}:\n        st = dict(st, ws13=st["dr_ws13"], ws2=st["dr_ws2"])\n'),
    ('        _VML["scb"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]))\n',
     '        _VML["scb"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]))\n'
     f'        if any(v in {DRP_VARS!r} for v in (VML_VAR,) + tuple(VML_ABVARS) + tuple(VML_BVAR.values())) and "dr_ws13" not in st:\n'
     '            st["dr_ws13"], st["dr_ws2"] = _vml_dr_repack(st["ws13"]), _vml_dr_repack(st["ws2"])\n'),
    ("\n\n_vml_FLAGS_DEFAULT = 1\n", REPACK + "\n\n_vml_FLAGS_DEFAULT = 1\n"),
]


def sub1(text, old, new, what):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[dr_patches] anchor {what!r} found {n} times (want 1)")
    return text.replace(old, new)


def cuda_span(text, name="_vml_CUDA_T"):
    a = text.index(f'{name} = r"""') + len(f'{name} = r"""')
    return a, text.index('"""', a)


def patch_cuda(cuda):
    for i, (old, new) in enumerate(CUDA_PATCHES):
        cuda = sub1(cuda, old, new, f"cuda {i}: {old[:50]}")
    return cuda


def patch_layer(text, bvar=None):
    """k3_gc_6 text -> the same file with the VM_DR switches, variants dr / drp and (optionally) VML_BVAR replaced."""
    a, b = cuda_span(text)
    text = text[:a] + patch_cuda(text[a:b]) + text[b:]
    for i, (old, new) in enumerate(PY_PATCHES):
        text = sub1(text, old, new, f"py {i}")
    if bvar is not None:
        old = text[text.index("\nVML_BVAR = "):]
        old = old[:old.index(NL, 1) + 1]
        text = sub1(text, old, f"\nVML_BVAR = {bvar!r}\n", "VML_BVAR")
    return text


def vml_block(text):
    """the layer's VML Python block: constants .. _vml_call (inclusive), as module source."""
    a = text.index("_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096\n")
    b = text.index("\nVML_ENABLE = True\n")
    return text[a:b]


LAB_HEAD = '''"""dr_moe.py (GENERATED by tools/dr_patches.py lab from kernels/muse_decode_layer/k3_gc_6.py; do not edit): k3_gc_6's VML
module (`_vml_` names, its CUDA = the shipped VML + the tools/dr_patches.py VM_DR switches) for lab use:
  build(var)                variant module: "bf2" = k3_gc_6's T <= 48 VML, "dr" = VM_DR, "drp" = VM_DR + VM_DRP; "...c" =
                            the same + setmaxnreg split (WG5 2, producers 64 / consumers 104 registers)
  alloc(dev, G)             scratch (y zeroed: BRED variants need the "scb" copy semantics of the layer)
  call(mod, var, k, sc, x, ids, wts, ...)   one MoE call (k: ws13 / ws2 / gsc13 / gsc2 / nw; drp / drpc read k["dr_ws13"])
  dr_repack(ws)             the lossless VM_DRP weight repack
  CUDA / DEFS / COST32      sources, compile switches, k3_gc_6's bucket-32 cost table
"""
import hashlib
import time

import torch

'''
LAB_TAIL = '''

CUDA = _vml_CUDA
DEFS = @DEFS@
COST32 = @COST32@
FLAGS32 = 193


def build(var="dr", verbose=False, extra=()):
    return _vml_build(verbose=verbose, defines=tuple(DEFS[var]) + tuple(extra))


def alloc(dev, G):
    return _vml_alloc(dev, G)


def dr_repack(ws):
    return _vml_dr_repack(ws)


def call(mod, var, k, sc, x, ids, wts, ls=0, marks=None, out=None, flags=FLAGS32, cost=None):
    if var in ("drp", "drpc"):
        k = dict(k, ws13=k["dr_ws13"], ws2=k["dr_ws2"])
    return _vml_call(mod, k, sc, x, ids, wts, ls=ls, marks=marks, out=out, flags=flags, cost=COST32 if cost is None else cost)
'''


def make_lab(text=None):
    t = patch_layer(BASE.read_text(encoding="utf-8") if text is None else text)
    blk = vml_block(t)
    cost = t[t.index("\nVML_COST = "):]
    cost = eval(cost[len("\nVML_COST = "):cost.index(NL, 1)])[32]   # noqa: S307 (our own generated literal)
    src = LAB_HEAD + blk + REPACK + LAB_TAIL.replace("@DEFS@", repr(DEFS)).replace("@COST32@", repr(list(cost)))
    compile(src, str(LAB_DST), "exec")
    LAB_DST.write_text(src, encoding="utf-8", newline=NL)
    print(f"[dr_patches] wrote {LAB_DST.relative_to(ROOT)} ({len(src.encode())} B)")
    return src


def _cpp(src, defines):
    """C preprocessor output of the CUDA text (includes dropped), comments and blank lines removed."""
    body = NL.join(ln for ln in src.split(NL) if not ln.startswith("#include"))
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
        f.write(body)
        p = f.name
    try:
        r = subprocess.run(["cpp", "-P", "-w", *defines, p], capture_output=True, text=True, check=True)
    finally:
        pathlib.Path(p).unlink()
    return NL.join(ln.strip() for ln in r.stdout.split(NL) if ln.strip())


def check():
    t = BASE.read_text(encoding="utf-8")
    a, b = cuda_span(t)
    old, new = t[a:b], patch_cuda(t[a:b])
    variants = {"bf2": BF2, "w72": BF2 + ("-DVM_WG5=2", "-DVM_PREG=32", "-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72"),
                "default": ()}
    for nm, d in variants.items():
        o, n = _cpp(old, d), _cpp(new, d)
        if o != n:
            raise SystemExit(f"[dr_patches] VM_DR=0 changes the {nm} preprocessed source")
        print(f"[dr_patches] {nm}: VM_DR=0 preprocessed source identical ({len(o)} chars)")
    patch_layer(t, {256: "w72", 128: "w72", 32: "dr"})
    print(f"[dr_patches] {len(CUDA_PATCHES)} CUDA + {len(PY_PATCHES)} Python anchors unique in {BASE.name}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "lab":
        make_lab()
    elif cmd == "check":
        check()
    else:
        raise SystemExit(__doc__)

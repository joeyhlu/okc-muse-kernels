"""xcf_probe.py: XCF phase-B cost probe (R-xcf-0; notes/handoff_xcf.md, notes/study/xc_moe.md s1-s6) on one H200, gated timer.

    python3 -u kernels/lab/xcf_probe.py [--ntoks=8,16,32,64] [--kbs=32,64,128,256,512,1024,4096] [--modes=cold]
                                        [--rounds=15] [--units=3] [--ey=8] [--jobs=4] [--compile-only] [--ptxas]
                                        [--out=box_out/xcf/probe.json]

Question (the XCF kill risk, R-xs-3): XCF's phase B runs W2[:, D] x h for all 4096 rows with K = the CTA's own dims |D|
(~248 at T32, ~540 T128, ~620 T256; less for hot experts) and ONE red.add per 256-row tile. What does that cost per 36 KB of
weights at a tile K extent KB, against the incumbent full-K B stage?
Arms (one module per N; the consumer is k3_gc_16's bf2 VML code verbatim: stage<N> RS-wgmma for N >= 24, stage_m<N>
mma.sync for N <= 16, dec_half2, the K2A stage layout + UNP L2 unpack; weights = MoE-board W13 as a stand-in, same bytes):
  inc  (kb 0) = dr_probe's drp_kernel (bf2 seg: 16 stages = K 4096 per 256-row tile, FP32 store x gs)
  kbXX        = the same stream, plus every XX of K a drain (wgmma wait_group 0; stage_m: its 2 chains summed) and ONE
                red.add.v4.f32 of each warp's 16 rows x N into y (XX < 256: inside the stage at k32-group ends; XX >= 256:
                at the end of every XX / 256 stages; scale pairs stay per k64 = 4 MMA, h / x stays the smem B operand).
                Rows are permuted per 16-row warp tile, phys(r) = 4 (g >> 1) + 2 (r >= 8) + (g & 1), so lanes g, g ^ 1
                swap one value per pair (2 shfl.xor 4) and each issues 1 red.v4 per n8 tile (vs 4 scalar red in VML bred).
                kb4096 = VML's B (one red per full-K tile).
  awR (kb -R) = the A-tail question: the inc stage loop with only R of the 4 consumer WGs working (the producer copies
                only their 64-row pieces, 2 copies of 8 KB + 1 KB per WG; idle WGs only pass the barriers). aw1 ~ inc / 4
                means a partial last row group costs only its bytes; aw1 ~ inc means idle WGs waste the stage (K-split
                needed). aw4 = inc's work with 8 copies per slot.
y = [ey][npad][4096] FP32. CTA c adds into block c * ey / G (ey 8: ~16.5 CTAs per block = XCF at T32's CTAs per expert),
its tile (unit slot k, half h) into rows 256 * ((2k + h + c) & 15) (staggered start, as XCF would).
Checks: err flag; y == index_add of inc's outputs over the same row map (FP32 sum order only: rel < 1e-5).
Timing (as xc_probe): cold = 256 MB L2 flush, then timer.gate() (~300 us, the judge's), then the graph replay; rotating arm
order; median of --rounds. stage_us = per CTA per 36 KB stage. y is not re-zeroed between timed replays (values only).
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools", ROOT / "tools/judge", ROOT / "kernels/muse_moe_w4a16", ROOT / "kernels/lab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dr_probe as DRP  # noqa: E402

BASE = ROOT / "kernels/muse_decode_layer/k3_gc_16.py"
KBS = (32, 64, 128, 256, 512, 1024, 4096)
AWS = (1, 2, 3, 4)   # aw arms: A stages with R active consumer WGs (kb = -R)

XCF_DEV = r"""
namespace vm {
struct XP { float* y; int ey; int npad; };   // y [ey][npad][KDIM] FP32
struct YS { float* yr; float gs; int pad; };   // per consumer warp, in the (unused) Tab region: off the register budget
extern __shared__ __align__(1024) uint8_t xcf_smem[];
// one warp's 16-row accumulator tile (mma rows g, g + 8; cols 8 nt + 2 t + {0, 1}) -> y rows phys(r) of its 16-row block:
// lanes g (even) / g + 1 swap one value per pair, then each holds 4 consecutive rows of one column -> 1 red.add.v4 per nt
template <int N>
__device__ __forceinline__ void xcf_red(int lane, const float* d) {
const YS ys = reinterpret_cast<const YS*>(xcf_smem + TAB_OFF)[threadIdx.x >> 5];
const float gs = ys.gs;
const int g = lane >> 2, t = lane & 3, od = g & 1;
float* p = ys.yr + 4 * (g >> 1);
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float s0 = od ? d[4 * nt] : d[4 * nt + 1], s1 = od ? d[4 * nt + 2] : d[4 * nt + 3];
const float r0 = __shfl_xor_sync(0xffffffffu, s0, 4), r1 = __shfl_xor_sync(0xffffffffu, s1, 4);
const int n = 8 * nt + 2 * t + od;
const float v0 = od ? r0 : d[4 * nt], v1 = od ? d[4 * nt + 1] : r0, v2 = od ? r1 : d[4 * nt + 2], v3 = od ? d[4 * nt + 3] : r1;
red_add_v4(p + (size_t)n * KDIM, v0 * gs, v1 * gs, v2 * gs, v3 * gs);
}
}
// stage<N> (bf2: VM_TRICKS, VcpCfg defaults NB 2 / W 1, no G2 / PRE / EA) + a drain and red.add every KB of K.
// KB < 256: chunks of KB / 32 k32 groups inside the stage; KB >= 256: FIRST stages open, END stages close a chunk.
template <int N, int KB, bool END, bool FIRST>
__device__ __forceinline__ void xcf_stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
uint32_t (&hb)[2][8], int q, int wp, int lane) {
constexpr int KG = (KB < 256) ? KB / 32 : 8;
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + (q >> 1) * SW;
const int tile = (q & 1) * 4 + wp;
const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
uint4 c[4];
c[0] = cp[0];
const uint32_t sw0 = sp[0], sw1 = sp[32];
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
const uint32_t zo = (uint32_t)A.flags >> 30;   // 0 (flags < 2^30), opaque to the compiler
const uint32_t k16 = 16u + zo, k64 = VM_TRICKS_DQ ? 64u + zo : 64u, k4 = VM_TRICKS_DQ ? 4u + zo : 4u;
const uint64_t d0 = wg_desc_sa(sa(smem + NS * US), 4096u);
const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);
#pragma unroll
for (int i = 0; i < 4; ++i) {
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
if (i < 3) c[i + 1] = cp[32 * (i + 1)];
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
const int gi = 2 * i + hf;
uint32_t* hh = hb[gi & 1];
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16, k64, k4);
if (i == 3 && hf == 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // all codes in registers
wg_fence();
const bool z = (KB < 256) ? (gi % KG == 0) : (FIRST && gi == 0);
if (z) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);
else mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
wg_commit();
const bool e = (KB < 256) ? (gi % KG == KG - 1) : (END && gi == 7);
if (e) { wg_wait<0>(); xcf_red<N>(lane, d); }
else wg_wait<1>();
if (gi == 0 && prevx >= 0) {   // the previous stage's last group is complete
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
prevx = -1;
}
}
}
prevx = xs;
++it;
}
template <int N>
__device__ __forceinline__ void xcf_flush_m(float (&d2)[2][N / 2], int lane) {
float dd[N / 2];
#pragma unroll
for (int k = 0; k < N / 2; ++k) { dd[k] = d2[0][k] + d2[1][k]; d2[0][k] = 0.f; d2[1][k] = 0.f; }
xcf_red<N>(lane, dd);
}
// stage_m<N> (bf2, VCP_LDA / VCP_M1 off) + a red.add every KB of K
template <int N, int KB, bool END>
__device__ __forceinline__ void xcf_stage_m(const Args& A, uint8_t* smem, uint32_t bars, int& it, float (&d2)[2][N / 2], int q,
int wp, int lane) {
constexpr int KG = (KB < 256) ? KB / 32 : 8;
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + (q >> 1) * SW;
const int tile = (q & 1) * 4 + wp;
const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
uint4 c[4];
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
const uint32_t sw0 = sp[0], sw1 = sp[32];
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
const uint32_t xb = sa(smem + NS * US + xs * XS) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
#pragma unroll
for (int i = 0; i < 4; ++i) {
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
const uint32_t k16q = 16u + ((uint32_t)A.flags >> 30);
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t hh[8];
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16q, 64u, 4u);
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
uint32_t b0, b1, b2, b3;
ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);
mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);
}
if (KB < 256 && ((2 * i + hf) % KG) == KG - 1) xcf_flush_m<N>(d2, lane);
}
}
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
if (KB >= 256 && END) xcf_flush_m<N>(d2, lane);
++it;
}
// one 256-row x 4096-K tile (16 stages) of this warp's 16 rows, red.add every KB of K
template <int N, int KB>
__device__ __forceinline__ void xcf_seg(const Args& A, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane) {
constexpr int SPC = (KB >= 256) ? KB / 256 : 1;
static_assert(SPC >= 1 && 16 % SPC == 0, "KB: 32 .. 4096, powers of two");
if constexpr (N <= 16) {
float d2[2][N / 2];
#pragma unroll
for (int k = 0; k < N / 2; ++k) { d2[0][k] = 0.f; d2[1][k] = 0.f; }
#pragma unroll 1
for (int sc = 0; sc < 16 / SPC; ++sc) {
#pragma unroll 1
for (int k = 0; k < SPC - 1; ++k) xcf_stage_m<N, KB, false>(A, smem, bars, it, d2, q, wp, lane);
xcf_stage_m<N, KB, true>(A, smem, bars, it, d2, q, wp, lane);
}
} else {
uint32_t hb[2][8];
float d[N / 2];
int prevx = -1;
#pragma unroll 1
for (int sc = 0; sc < 16 / SPC; ++sc) {
if constexpr (SPC == 1) {
xcf_stage<N, KB, true, true>(A, smem, bars, it, prevx, d, hb, q, wp, lane);
} else {   // FIRST / END compile-time per call site (a runtime first around wgmma serializes it: C7519)
xcf_stage<N, KB, false, true>(A, smem, bars, it, prevx, d, hb, q, wp, lane);
#pragma unroll 1
for (int k = 1; k < SPC - 1; ++k) xcf_stage<N, KB, false, false>(A, smem, bars, it, prevx, d, hb, q, wp, lane);
xcf_stage<N, KB, true, false>(A, smem, bars, it, prevx, d, hb, q, wp, lane);
}
}
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));   // drained by the last END stage
}
}
template <int N, int KB>
__global__ void __launch_bounds__(NTH, 1) xcf_kernel(const Args A, const PArgs P, const XP X) {
extern __shared__ __align__(1024) uint8_t smem[];
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const uint32_t bars = sa(smem + BAR_OFF);
if (tid == 0) {
for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__syncthreads();
const int2 rg = P.cta[blockIdx.x];
if (warp >= NCW) {
if (lane == 0) {
if (warp == NCW) drp_prod_w(A, P, sa(smem), bars, rg.x, rg.y);
else if (warp == NCW + 1) drp_prod_x(A, P, sa(smem + NS * US), bars, rg.x, rg.y);
}
return;
}
const int q = warp >> 2, wp = warp & 3, cta = blockIdx.x;
long long* mk = P.marks ? P.marks + (size_t)cta * 8 : nullptr;
if (mk && tid == 0) { mk[0] = gtime(); mk[5] = clock64(); }
const int ye = (int)(((long long)cta * X.ey) / gridDim.x);
float* yc = X.y + (size_t)ye * X.npad * KDIM + 128 * (q >> 1) + 16 * ((q & 1) * 4 + wp);
int it = 0;
#pragma unroll 1
for (int u = rg.x; u < rg.y; ++u) {
const float gs = __ldg(P.gsc + P.unit[u].x);
#pragma unroll 1
for (int h = 0; h < 2; ++h) {
const int tt = (2 * (u - rg.x) + h + cta) & 15;
if (lane == 0) reinterpret_cast<YS*>(smem + TAB_OFF)[warp] = YS{yc + 256 * tt, gs, 0};
__syncwarp();
xcf_seg<N, KB>(A, smem, bars, it, q, wp, lane);
}
}
if (mk && wp == 0 && lane == 0) { mk[1 + q] = gtime(); if (q == 0) mk[6] = clock64(); }
}
// ---- A-tail question: the incumbent A stage loop with only R of the 4 consumer WGs active (the producer copies only the
// active WGs' 64-row pieces: codes 8 KB + scales 1 KB each; idle WGs only pass the slot / x barriers). R = 4: the same work
// as inc with 8 copies per slot instead of 2.
template <int R>
__device__ void aw_prod_w(const Args& A, const PArgs& P, uint32_t ring, uint32_t bars, int u0, int u1) {
const uint64_t pol = policy_evict_first();
int it = 0;
for (int u = u0; u < u1; ++u) {
const int4 un = P.unit[u];
for (int h = 0; h < 2; ++h)
for (int s = 0; s < 16; ++s, ++it) {
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, R * 9216);
#pragma unroll
for (int q = 0; q < R; ++q) {
const uint8_t* src = P.ws + ((size_t)(un.x * 64 + 4 * un.y + 2 * h + (q >> 1)) * 16 + s) * SW;
const uint32_t db = dst + (q >> 1) * SW;
bulk_g2s_ef(db + (q & 1) * 8192, src + (q & 1) * 8192, 8192, fb, pol);
bulk_g2s_ef(db + 16384 + (q & 1) * 1024, src + 16384 + (q & 1) * 1024, 1024, fb, pol);
}
}
}
}
template <int N, int R>
__global__ void __launch_bounds__(NTH, 1) xcf_aw(const Args A, const PArgs P) {
extern __shared__ __align__(1024) uint8_t smem[];
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const uint32_t bars = sa(smem + BAR_OFF);
if (tid == 0) {
for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__syncthreads();
const int2 rg = P.cta[blockIdx.x];
if (warp >= NCW) {
if (lane == 0) {
if (warp == NCW) aw_prod_w<R>(A, P, sa(smem), bars, rg.x, rg.y);
else if (warp == NCW + 1) drp_prod_x(A, P, sa(smem + NS * US), bars, rg.x, rg.y);
}
return;
}
const int q = warp >> 2, wp = warp & 3;
long long* mk = P.marks ? P.marks + (size_t)blockIdx.x * 8 : nullptr;
if (mk && tid == 0) { mk[0] = gtime(); mk[5] = clock64(); }
int it = 0;
if (q < R) {
#pragma unroll 1
for (int u = rg.x; u < rg.y; ++u) {
const float gs = __ldg(P.gsc + P.unit[u].x);
#pragma unroll 1
for (int h = 0; h < 2; ++h) {
float d[N / 2];
drp_seg_inc<N>(A, smem, bars, it, d, q, wp, lane);
drp_store<N>(P, u, (2 * h + (q >> 1)) * 8 + (q & 1) * 4 + wp, lane, d, gs);
}
}
} else {
const int ns = (rg.y - rg.x) * 32;
#pragma unroll 1
for (int k = 0; k < ns; ++k, ++it) {
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
}
}
if (mk && wp == 0 && lane == 0) { mk[1 + q] = gtime(); if (q == 0) mk[6] = clock64(); }
}
}  // namespace vm
"""

XCF_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> xcf_attrs(int64_t kb);
void xcf_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,
             torch::Tensor y, torch::Tensor err, torch::Tensor marks, int64_t kb, int64_t ey, int64_t G);
"""


def xcf_host(n, kbs):
    """host functions of the module for N = n: kb 0 = drp_kernel<n> (inc), kb > 0 = xcf_kernel<n, kb>, kb = -R: xcf_aw<n, R>."""
    ks = (0,) + tuple(kbs) + tuple(-r for r in AWS)
    cases_a = "".join(f"    case {k}: return xcf_attr<{k}>();\n" for k in ks)
    cases_r = "".join(f"    case {k}: xcf_launch<{k}>(a, p, x, (int)G); break;\n" for k in ks)
    return r"""
template <int KB>
static void xcf_launch(const vm::Args& a, const vm::PArgs& p, const vm::XP& x, int G) {
  if constexpr (KB == 0) {
    cudaFuncSetAttribute(vm::drp_kernel<@N@>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
    vm::drp_kernel<@N@><<<G, vm::NTH, vm::SMEM, at::cuda::getCurrentCUDAStream()>>>(a, p);
  } else if constexpr (KB < 0) {
    cudaFuncSetAttribute(vm::xcf_aw<@N@, -KB>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
    vm::xcf_aw<@N@, -KB><<<G, vm::NTH, vm::SMEM, at::cuda::getCurrentCUDAStream()>>>(a, p);
  } else {
    cudaFuncSetAttribute(vm::xcf_kernel<@N@, KB>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
    vm::xcf_kernel<@N@, KB><<<G, vm::NTH, vm::SMEM, at::cuda::getCurrentCUDAStream()>>>(a, p, x);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int KB>
static std::vector<int64_t> xcf_attr() {
  cudaFuncAttributes fa;
  if constexpr (KB == 0) {
    cudaFuncSetAttribute(vm::drp_kernel<@N@>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
    cudaFuncGetAttributes(&fa, vm::drp_kernel<@N@>);
  } else if constexpr (KB < 0) {
    cudaFuncSetAttribute(vm::xcf_aw<@N@, -KB>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
    cudaFuncGetAttributes(&fa, vm::xcf_aw<@N@, -KB>);
  } else {
    cudaFuncSetAttribute(vm::xcf_kernel<@N@, KB>, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
    cudaFuncGetAttributes(&fa, vm::xcf_kernel<@N@, KB>);
  }
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vm::SMEM, (int64_t)vm::NTH};
}
std::vector<int64_t> xcf_attrs(int64_t kb) {
  switch (kb) {
@CA@    default: return {-1};
  }
}
void xcf_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,
             torch::Tensor y, torch::Tensor err, torch::Tensor marks, int64_t kb, int64_t ey, int64_t G) {
  vm::Args a{};
  a.err = err.data_ptr<int>();
  a.flags = 193;
  vm::PArgs p{};
  p.ws = reinterpret_cast<const uint8_t*>(ws.data_ptr()); p.gsc = gsc.data_ptr<float>();
  p.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr()); p.unit = reinterpret_cast<const int4*>(unit.data_ptr());
  p.cta = reinterpret_cast<const int2*>(cta.data_ptr()); p.out = out.data_ptr<float>();
  p.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.npad = @N@;
  vm::XP x{y.data_ptr<float>(), (int)ey, @N@};
  TORCH_CHECK(y.numel() >= ey * @N@ * 4096, "xcf: y too small");
  switch (kb) {
@CR@    default: TORCH_CHECK(false, "xcf: kb not instantiated");
  }
}
""".replace("@N@", str(n)).replace("@CA@", cases_a).replace("@CR@", cases_r)


def gc16_vml():
    """k3_gc_16's VML CUDA (up to its host entry point) and its bf2 defines."""
    import ast
    t = BASE.read_text(encoding="utf-8")
    ns = {}
    exec(compile("import hashlib, time\n" + t[t.index("_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096\n"):t.index("\nVML_ENABLE = True\n")],
                 "k3_gc_16:_vml_", "exec"), ns)   # noqa: S102 (our own source)
    cuda = ns["_vml_CUDA"]
    i = t.index('"bf2": (', t.index("\n_VML_DEFS = {"))
    bf2 = tuple(ast.literal_eval(t[i + 7:t.index(")", i) + 1]))
    return cuda[:cuda.index("void vml_run(")], bf2


def probe_dev():
    """dr_probe's device code (PArgs, producers, drp_seg_inc, drp_store, drp_kernel) without its explicit instantiations."""
    s = DRP.PROBE_CU[:DRP.PROBE_CU.index(DRP.HOST_MARK)]
    lines = [ln for ln in s.split("\n") if not ln.startswith("template __global__ void drp_kernel<")]
    return "\n".join(lines)


def module_src(n, kbs):
    head, bf2 = gc16_vml()
    return head + probe_dev() + XCF_DEV + xcf_host(n, kbs), bf2


def build(n, kbs, verbose=False, ptxas=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    src, bf2 = module_src(n, kbs)
    fl = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"] + list(bf2) + [
        "-DVM_DR=0", "-DVM_DRP=0"]
    if ptxas:
        fl.append("-Xptxas=-v")
    tag = hashlib.sha1((XCF_CPP + src + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"xcf_probe_n{n}_{tag}", cpp_sources=XCF_CPP, cuda_sources=src, functions=["xcf_run", "xcf_attrs"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


def yref_index(P, G, K, ey, dev):
    """flat y index of every inc output element out[u, row, n] (row map of xcf_kernel)."""
    import torch
    n_u, npad = P["n"], P["npad"]
    u = torch.arange(n_u, device=dev).view(-1, 1, 1)
    row = torch.arange(512, device=dev).view(1, -1, 1)
    n = torch.arange(npad, device=dev).view(1, 1, -1)
    c = u // K
    k = u % K
    ye = (c * ey) // G
    h, r = row // 256, row % 256
    tt = (2 * k + h + c) & 15
    r16 = r % 16
    phys = 4 * ((r16 & 7) >> 1) + 2 * (r16 >= 8).long() + (r16 & 1)
    yrow = 256 * tt + 16 * (r // 16) + phys
    return ((ye * npad + n) * 4096 + yrow).reshape(-1)


def main(argv=None):
    import concurrent.futures as cf
    import json
    import statistics
    import time

    argv = sys.argv[1:] if argv is None else argv
    log = lambda s: print(s, flush=True)  # noqa: E731
    ntoks = tuple(int(v) for v in str(DRP.arg(argv, "ntoks", "8,16,32,64")).split(","))
    kbs = tuple(int(v) for v in str(DRP.arg(argv, "kbs", ",".join(map(str, KBS)))).split(","))
    assert all(n % 8 == 0 and 8 <= n <= 64 for n in ntoks) and all(k in KBS for k in kbs)
    modes = tuple(str(DRP.arg(argv, "modes", "cold")).split(","))
    rounds, K, warm_set = int(DRP.arg(argv, "rounds", 15)), int(DRP.arg(argv, "units", 3)), int(DRP.arg(argv, "warm-set", 8))
    ey, jobs = int(DRP.arg(argv, "ey", 8)), int(DRP.arg(argv, "jobs", 4))
    aws = tuple(int(v) for v in str(DRP.arg(argv, "aws", ",".join(map(str, AWS)))).split(",") if v)
    assert all(r in AWS for r in aws)
    out_p = ROOT / str(DRP.arg(argv, "out", "box_out/xcf/probe.json"))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    co, px = "--compile-only" in argv, "--ptxas" in argv
    t0 = time.time()
    mods = {}
    with cf.ThreadPoolExecutor(max(1, jobs)) as ex:
        futs = {n: ex.submit(build, n, kbs, co or px, px) for n in ntoks}
        for n, f in futs.items():
            try:
                mods[n], dt = f.result()
                log(f"[xcf] built N{n} in {dt:.0f} s")
            except Exception as e:  # noqa: BLE001
                log(f"[xcf] build N{n} FAILED: {e!r}"[-3000:])
    for n in mods:
        log(f"[xcf] attrs N{n} (regs, local B, smem, threads): " + " ".join(f"kb{k} {list(mods[n].xcf_attrs(k))}" for k in (0,) + kbs + tuple(-r for r in aws)))
    if co:
        return 0 if len(mods) == len(ntoks) else 1

    import torch
    import judge as J
    import k2a_v37 as D
    import unp_patches as UNP
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    prob = J.load_problem(ROOT / "problems/muse_moe_w4a16")
    ref = prob.ref
    w = ref.generate_weights(ref.WEIGHTS_SEED, device=dev)
    log(f"[xcf] UNP L2 decode self-test: {UNP.selftest(20000)} words bit-exact vs BITPOS3")
    ws13 = torch.empty((32, 64, 16, D.SW3), dtype=torch.uint8, device=dev)
    for e in range(32):
        w13q, s13q = D.prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=UNP.L2POS)
        ws13[e] = D.stage_repack(w13q, s13q)
    gsc13 = (w["w13_gs"][:32].float() * D.W_SHIFT).contiguous()
    xsrc = torch.randn((512, 4096), generator=torch.Generator(device="cpu").manual_seed(5), dtype=torch.float32).to(torch.bfloat16).to(dev)
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    marks = torch.zeros((G * 8,), dtype=torch.int64, device=dev)
    log(f"[xcf] setup {time.time() - t0:.0f} s, {G} SMs, replay overhead {ov:.2f} us, gate {timer.gate_us} us, kbs {kbs}, ey {ey}, "
        f"rounds {rounds}, {K} units (= {32 * K} stages of 36 KB) per CTA")

    def timed(g, flush, warm):
        with torch.cuda.stream(timer.stream):
            if warm:
                g.replay()
            if flush:
                timer.flush()
            timer.gate()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    res = {"G": G, "overhead_us": ov, "gate_us": timer.gate_us, "kbs": kbs, "ey": ey, "rounds": rounds, "units_per_cta": K,
           "attrs": {n: {k: list(mods[n].xcf_attrs(k)) for k in (0,) + kbs + tuple(-r for r in aws)} for n in mods}, "rows": []}
    arms = (0,) + kbs + tuple(-r for r in aws)
    for N in ntoks:
        if N not in mods:
            continue
        mod = mods[N]
        for mode in modes:
            P = DRP.plan(N, mode, G, K, warm_set, xsrc, dev)
            npad = P["npad"]
            out = torch.full((P["n"], 512, npad), float("nan"), dtype=torch.float32, device=dev)
            y = torch.zeros((ey * npad * 4096,), dtype=torch.float32, device=dev)

            def fn(kb, mk=nomark, P=P, out=out, y=y, mod=mod):
                return lambda: mod.xcf_run(ws13, gsc13, P["xp"], P["unit"], P["cta"], out, y, err, mk, kb, ey, G)
            chk = {}
            err.zero_()
            fn(0)()
            torch.cuda.synchronize()
            chk[0] = {"err": int(err[0]), "finite": bool(torch.isfinite(out).all())}
            chk[0]["ok"] = chk[0]["err"] == 0 and chk[0]["finite"]
            idx = yref_index(P, G, K, ey, dev)
            yref = torch.zeros_like(y, dtype=torch.float64).index_add_(0, idx, out.reshape(-1).double())
            nref = float(yref.norm())
            for kb in kbs:
                y.zero_()
                err.zero_()
                fn(kb)()
                torch.cuda.synchronize()
                d = (y.double() - yref)
                rel = float(d.norm() / max(nref, 1e-300))
                mx = float(d.abs().max() / yref.abs().max().clamp(min=1e-300))
                chk[kb] = {"err": int(err[0]), "rel": rel, "maxrel": mx, "ok": int(err[0]) == 0 and rel < 1e-5 and bool(torch.isfinite(y).all())}
            out_inc = out.clone()
            for r in aws:   # active WG q covers rows 256 h + 64 q + [0, 64): bitwise = inc there
                out.fill_(float("nan"))
                err.zero_()
                fn(-r)()
                torch.cuda.synchronize()
                o, oi = out.view(P["n"], 2, 4, 64, npad)[:, :, :r], out_inc.view(P["n"], 2, 4, 64, npad)[:, :, :r]
                chk[-r] = {"err": int(err[0]), "rel": 0.0, "bitwise_inc": bool(torch.equal(o, oi))}
                chk[-r]["ok"] = chk[-r]["err"] == 0 and chk[-r]["bitwise_inc"]
            gr = {}
            for kb in arms:
                f = fn(kb)
                with torch.cuda.stream(timer.stream):
                    f()
                    torch.cuda.synchronize()
                    g_ = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_, stream=timer.stream):
                        f()
                torch.cuda.synchronize()
                gr[kb] = g_
            ts = {kb: [] for kb in arms}
            for rd in range(rounds):
                k = rd % len(arms)
                for kb in arms[k:] + arms[:k]:
                    ts[kb].append(max(timed(gr[kb], mode == "cold", mode == "warm") - ov, 1e-3))
            us = {kb: statistics.median(v) for kb, v in ts.items()}
            span, ghz = {}, {}
            for kb in arms:
                marks.zero_()
                with torch.cuda.stream(timer.stream):
                    if mode == "cold":
                        timer.flush()
                    timer.gate()
                    fn(kb, mk=marks)()
                torch.cuda.synchronize()
                m = marks.view(G, 8).cpu().double()
                span[kb] = float(((m[:, 1:5].max(dim=1).values - m[:, 0]) / 1e3).median())
                cyc, nsx = m[:, 6] - m[:, 5], m[:, 1] - m[:, 0]
                okm = (cyc > 0) & (nsx > 0)
                ghz[kb] = float((cyc[okm] / nsx[okm]).median()) if bool(okm.any()) else None
            stage = {kb: us[kb] / (32 * K) for kb in arms}
            rel_inc = {kb: stage[kb] / stage[0] - 1.0 for kb in arms if kb != 0}
            red_kb = {kb: G * K * 2 * (4096 // kb) * 256 * npad * 4 / 1e6 for kb in kbs}   # MB of red.add per call
            row = {"N": N, "mode": mode, "us": us, "stage_us": stage, "rel_inc": rel_inc, "span_us": span, "ghz": ghz,
                   "red_MB": red_kb, "check": chk, "spread_us": {kb: (min(v), max(v)) for kb, v in ts.items()}}
            res["rows"].append(row)
            log(f"[xcf] N={N:3d} {mode}: stage us inc {stage[0]:.3f} | " + " ".join(f"kb{kb} {stage[kb]:.3f} ({100 * rel_inc[kb]:+.1f}%)" for kb in kbs)
                + " | aw " + " ".join(f"{-kb}:{stage[kb]:.3f} (x{stage[kb] / stage[0]:.2f})" for kb in arms if kb < 0)
                + " | span " + " ".join(f"{kb} {span[kb]:.1f}" for kb in arms)
                + " | GHz " + " ".join(f"{ghz[kb]:.2f}" for kb in arms if ghz[kb])
                + " | ok " + " ".join(f"{kb}:{int(chk[kb]['ok'])}" for kb in arms)
                + " | rel " + " ".join(f"{kb}:{chk[kb]['rel']:.1e}" for kb in kbs))
            del gr, out, out_inc, y, P, yref, idx
            torch.cuda.empty_cache()
    out_p.write_text(json.dumps(res, indent=1, default=str))
    log(f"[xcf] wrote {out_p} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

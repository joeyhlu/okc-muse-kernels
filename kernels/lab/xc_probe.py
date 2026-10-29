"""xc_probe.py: step-0 probe of the XC MoE engine (notes/study/xc_moe.md) on one H200, gated timing (judge GpuTimer).

    python3 -u kernels/lab/xc_probe.py [--arms=inc,w72,ctl,pp,ppt,ppr16,ppr1] [--ntoks=8,16,32,56,64,112,128]
                                       [--rounds=15] [--units=3] [--modes=cold,warm] [--gbar] [--redbw]
                                       [--compile-only] [--out=box_out/xc/probe.json]

What is compared (phase-A / W13 stages of 256 rows x 256 K, the K2A stage layout + UNP-L2 bit layout + exact unpack, the
same units, tokens and activations for every arm; outputs FP32 sums x global scale per (row, token), comparable bit for bit):
  inc  = bf2 (k3_gc_6 T <= 48 VML consumer: 576 threads, 96 regs, NS 3 x 36 KB, NX 3; mma.sync stage_m for N <= 16),
         compiled from dr_probe's drp_kernel (kernels/lab/dr_probe.py) with more N instantiations
  w72  = k3_gc_6 w72 (the T >= 128 consumer: WG5 640 threads, setmaxnreg 32 / 112, JN / XSN 72), N 24..72
  ctl  = inc built again (identical code): arm-order / run-to-run noise
  pp   = XC PP consumer: 384 threads (WG0 producers, setmaxnreg.dec 40; WG1-2 consumers, setmaxnreg.inc 232), each
         consumer WG one 128-row K2A block as two m64 tiles, ordered ping-pong of k128 bursts (16 HGMMA per commit,
         wait_group 0, the other WG dequantizes meanwhile); N up to 128 in one pass
  ppt  = pp without the dequant (raw code words as A): the stream + LDS + wgmma floor of the PP loop
  ppr16 / ppr1 = pp whose tile epilogue red.adds (v2 f32) the accumulators into an L2-resident y every 16 / every 1
         stage (XC phase-B pattern: ppr1 = one red.add of 256 rows x N per 36 KB stage, the worst case)
Every timed replay: (cold) 256 MB L2 flush, then timer.gate() (~300 us sleep, as the judge), then the graph replay.
--gbar: software grid barrier latency (132 CTAs, red.release arrive + ld.acquire poll, bounded) over 200 rounds.
--redbw: pure red.add.v4.f32 throughput into a 4 / 16 MB L2-resident buffer.
Checks: error flags (bounded mbarrier waits), FP64 reference on sampled units, padded columns zero, bitwise vs inc.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools", ROOT / "tools/judge", ROOT / "kernels/muse_moe_w4a16", ROOT / "kernels/lab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dr_probe as DRP  # noqa: E402

ARMS = ("inc", "w72", "ctl", "pp", "ppt", "ppr16", "ppr1")
PP_ARMS = {"pp": 0, "ppt": 1, "ppr16": 2, "ppr1": 3}
INC_N = (8, 16, 24, 32, 40, 48, 56, 64)
W72_N = (24, 32, 40, 48, 56, 64, 72)
PP_N = (8, 16, 24, 32, 48, 56, 64, 96, 112, 128)
W72_DEFS = ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_WG5=2", "-DVM_PREG=32",
            "-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72")   # k3_gc_6 _VML_DEFS["w72"]

XC_CU = r"""
namespace xc {
using namespace vm;
constexpr int PNTH = 384;   // WG0: producers (warp 0 weights, warp 1 x), WG1 / WG2: consumers p = 0 / 1
template <int N> struct PC {
static constexpr int NS = (N > 112) ? 2 : 3;      // weight ring slots of US (36 KB: two 128-row K2A blocks x 256 K)
static constexpr int XSB = N * 512;                 // x slice of one 256-K stage (N = padded tokens)
static constexpr int NX = (N <= 64) ? 3 : 2;
static constexpr int XOFF = NS * US;
static constexpr int BOFF = XOFF + NX * XSB;
static constexpr int SMEM = BOFF + 8 * 2 * (NS + NX) + 64;
};
static_assert(PC<128>::SMEM <= 232448, "PP rings exceed 227 KB");
struct XArgs {
const uint8_t* ws; const float* gsc; const uint8_t* xp; const int4* unit; const int2* cta; float* out; float* y;
long long* marks; int* err; int npad; int mode;
};
template <int N>
__device__ __forceinline__ void pp_prod_w(const XArgs& P, uint32_t ring, uint32_t bars, int u0, int u1) {
constexpr int PNS = PC<N>::NS;
const uint64_t pol = policy_evict_first();
int it = 0;
for (int u = u0; u < u1; ++u) {
const int4 un = P.unit[u];
for (int h = 0; h < 2; ++h)
for (int s = 0; s < 16; ++s, ++it) {
const int sl = it % PNS;
if (it >= PNS) mbar_wait(bars + 8 * (PNS + sl), ((it / PNS) & 1) ^ 1, P.err);
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, US);
const int blk = un.x * 64 + 4 * un.y + 2 * h;
bulk_g2s_ef(dst, P.ws + ((size_t)blk * 16 + s) * SW, SW, fb, pol);
bulk_g2s_ef(dst + SW, P.ws + ((size_t)(blk + 1) * 16 + s) * SW, SW, fb, pol);
}
}
}
template <int N>
__device__ __forceinline__ void pp_prod_x(const XArgs& P, uint32_t xring, uint32_t bars, int u0, int u1) {
using C = PC<N>;
constexpr int PNS = C::NS;
const uint32_t bytes = (uint32_t)C::XSB;
int it = 0;
for (int u = u0; u < u1; ++u) {
const uint8_t* src = P.xp + (size_t)P.unit[u].z;
for (int h = 0; h < 2; ++h)
for (int s = 0; s < 16; ++s, ++it) {
const int sl = it % C::NX;
if (it >= C::NX) mbar_wait(bars + 8 * (2 * PNS + C::NX + sl), ((it / C::NX) & 1) ^ 1, P.err);
const uint32_t fb = bars + 8 * (2 * PNS + sl);
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * C::XSB, src + (size_t)s * bytes, bytes, fb);
}
}
}
// one consumer warp's codes / scales for chunk c (k128 = K2A groups 2c, 2c + 1) of its two warp-tiles wp and 4 + wp
__device__ __forceinline__ void pp_load(const uint8_t* bk, int wp, int lane, int c, uint4 (&cc)[2][2], uint32_t (&sc)[2]) {
#pragma unroll
for (int t = 0; t < 2; ++t) {
const int tile = 4 * t + wp;
const uint4* cp = reinterpret_cast<const uint4*>(bk + tile * 2048) + lane;
cc[t][0] = cp[32 * (2 * c)];
cc[t][1] = cp[32 * (2 * c + 1)];
sc[t] = reinterpret_cast<const uint32_t*>(bk + 16384 + tile * 256)[lane + 32 * c];
}
}
template <int N>
__device__ __forceinline__ void pp_store(const XArgs& P, int u, int tl, int lane, const float* d, float gs) {
const int g = lane >> 2, t = lane & 3;
float* o = P.out + ((size_t)u * 512 + 16 * tl + g) * P.npad;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
o[n] = d[4 * nt + hh] * gs;
o[(size_t)8 * P.npad + n] = d[4 * nt + 2 + hh] * gs;
}
}
__device__ __forceinline__ void red_v2(float* p, float a, float b) {
asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" :: "l"(p), "f"(a), "f"(b) : "memory");
}
template <int N>
__device__ __forceinline__ void pp_red(const XArgs& P, int slot, int tl, int lane, const float* d) {
const int g = lane >> 2, t = lane & 3;
float* o = P.y + ((size_t)(slot & 63) * 512 + 16 * tl + g) * N;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
red_v2(o + 8 * nt + 2 * t, d[4 * nt], d[4 * nt + 1]);
red_v2(o + 8 * N + 8 * nt + 2 * t, d[4 * nt + 2], d[4 * nt + 3]);
}
}
// MODE 0 pp, 1 ppt (no dequant), 2 ppr16 (red.add epilogue per 16 stages), 3 ppr1 (red.add per stage)
template <int N, int MODE>
__device__ __forceinline__ void pp_consume(const XArgs& P, uint8_t* smem, uint32_t bars, int p, int wp, int lane, int u0, int u1) {
using C = PC<N>;
constexpr int PNS = C::NS;
constexpr int RT = (MODE == 3) ? 1 : 16;
float d0[N / 2], d1[N / 2];
const uint32_t zo = (uint32_t)P.mode >> 30;   // 0, opaque to the compiler (VML's one PRMT + IMAD scale pair)
const uint32_t k16 = 16u + zo;
const uint64_t xd = wg_desc_sa(sa(smem + C::XOFF), 4096u);
const uint32_t xlo = (uint32_t)xd, xhi = (uint32_t)(xd >> 32);
const int nk = (u1 - u0) * 64;                 // chunks: units x 2 halves x 16 stages x 2
if (nk <= 0) return;
uint4 cc[2][2];
uint32_t sc[2];
mbar_wait(bars, 0, P.err);
pp_load(smem + p * SW, wp, lane, 0, cc, sc);
bool first = (p == 0);
#pragma unroll 1
for (int k = 0; k < nk; ++k) {
const int s = k >> 1, c = k & 1;
const int xs = s % C::NX;
uint32_t a[2][2][2][8];
#pragma unroll
for (int t = 0; t < 2; ++t)
#pragma unroll
for (int gi = 0; gi < 2; ++gi)
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
const uint32_t w0 = hf ? cc[t][gi].y : cc[t][gi].x, w8 = hf ? cc[t][gi].w : cc[t][gi].z;
if constexpr (MODE == 1) {
#pragma unroll
for (int j = 0; j < 4; ++j) { a[t][gi][hf][j] = w0 & 0x3F803F80u; a[t][gi][hf][4 + j] = w8 & 0x3F803F80u; }
} else {
dec_half2(w0, w8, sc[t], (uint32_t)gi, a[t][gi][hf], k16, 64u, 4u);
}
}
if (c == 0) mbar_wait(bars + 8 * (2 * PNS + xs), (s / C::NX) & 1, P.err);
if (!first) bar_sync(1 + p, 256);   // my turn: the other WG has issued its burst
first = false;
wg_fence();
const bool z = ((s % RT) == 0) && (c == 0);
const uint32_t xl = xlo + (uint32_t)xs * (uint32_t)(C::XSB >> 4);
#pragma unroll
for (int gi = 0; gi < 2; ++gi)
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
const uint32_t dl = xl + 64u * (uint32_t)(2 * c + gi) + 32u * (uint32_t)hf;
if (gi == 0 && hf == 0 && z) {
mma2<N, true>(d0, a[0][gi][hf], dl, xhi);
mma2<N, true>(d1, a[1][gi][hf], dl, xhi);
} else {
mma2<N, false>(d0, a[0][gi][hf], dl, xhi);
mma2<N, false>(d1, a[1][gi][hf], dl, xhi);
}
}
wg_commit();
bar_arrive(2 - p, 256);             // the other WG's turn
if (k + 1 < nk) {                   // next chunk's codes during my burst
const int s2 = (k + 1) >> 1, c2 = (k + 1) & 1, ws2 = s2 % PNS;
if (c2 == 0) mbar_wait(bars + 8 * ws2, (s2 / PNS) & 1, P.err);
pp_load(smem + ws2 * US + p * SW, wp, lane, c2, cc, sc);
if (c2 == 1) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (PNS + ws2)); }   // all of s2's codes in registers
}
wg_wait<0>();
if (c == 1) {
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (2 * PNS + C::NX + xs));
if ((s % RT) == RT - 1) {
const int u = u0 + (s >> 5), h = (s >> 4) & 1;
const int tl0 = (2 * h + p) * 8 + wp, tl1 = tl0 + 4;
if constexpr (MODE >= 2) {
pp_red<N>(P, u * 32 + (s & 31), tl0, lane, d0);
pp_red<N>(P, u * 32 + (s & 31), tl1, lane, d1);
} else {
const float gs = __ldg(P.gsc + P.unit[u].x);
pp_store<N>(P, u, tl0, lane, d0, gs);
pp_store<N>(P, u, tl1, lane, d1, gs);
}
}
}
}
if (p == 0) bar_sync(1, 256);          // the other WG's last arrive
}
template <int N, int MODE>
__global__ void __launch_bounds__(PNTH, 1) xc_pp(const XArgs P) {
extern __shared__ __align__(1024) uint8_t smem[];
using C = PC<N>;
if ((threadIdx.x >> 7) == 0) {
asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
const uint32_t bars = sa(smem + C::BOFF);
bar_sync(3, PNTH);                    // consumer thread 128 has initialized the barriers
if (lane == 0) {
const int2 rg = P.cta[blockIdx.x];
if (warp == 0) pp_prod_w<N>(P, sa(smem), bars, rg.x, rg.y);
else if (warp == 1) pp_prod_x<N>(P, sa(smem + C::XOFF), bars, rg.x, rg.y);
}
return;
}
asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const uint32_t bars = sa(smem + C::BOFF);
if (tid == 128) {
constexpr int PNS = C::NS;
for (int i = 0; i < PNS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (PNS + i), 8); }
for (int i = 0; i < C::NX; ++i) { mbar_init(bars + 8 * (2 * PNS + i), 1); mbar_init(bars + 8 * (2 * PNS + C::NX + i), 8); }
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
bar_sync(3, PNTH);
const int p = (warp - 4) >> 2, wp = warp & 3;
const int2 rg = P.cta[blockIdx.x];
long long* mk = P.marks ? P.marks + (size_t)blockIdx.x * 8 : nullptr;
if (mk && p == 0 && wp == 0 && lane == 0) { mk[0] = gtime(); mk[5] = clock64(); }
pp_consume<N, MODE>(P, smem, bars, p, wp, lane, rg.x, rg.y);
if (mk && wp == 0 && lane == 0) { mk[1 + p] = gtime(); if (p == 0) mk[6] = clock64(); }
}
// ---- software grid barrier latency: every CTA's thread 0 arrives (red.release) and polls (ld.acquire), bounded
__global__ void xc_gbar(int* cnt, long long* marks, int rounds, int* err) {
if (threadIdx.x != 0) return;
const int G = gridDim.x;
for (int r = 0; r < rounds; ++r) {
const long long ta = gtime();
red_release_add(cnt, 1);
const int target = G * (r + 1);
long long n = 0;
while (ld_acquire(cnt) < target) {
if (++n > (1ll << 26)) { atomicExch(err, 1); return; }
}
const long long tb = gtime();
marks[((size_t)r * G + blockIdx.x) * 2] = ta;
marks[((size_t)r * G + blockIdx.x) * 2 + 1] = tb;
}
}
// ---- pure red.add.v4.f32 throughput: each thread iters x one 16 B red.add, a warp covers 512 contiguous bytes
__global__ void __launch_bounds__(512) xc_redbw(float* y, int n16, int iters) {
const int gt = blockIdx.x * blockDim.x + threadIdx.x, gs = gridDim.x * blockDim.x;
for (int i = 0; i < iters; ++i) {
const int idx = (int)(((long long)gt + (long long)i * gs) % n16);
red_add_v4(y + 4 * (size_t)idx, 1.f, 1.f, 1.f, 1.f);
}
}
}  // namespace xc
""" + DRP.HOST_MARK + r"""
template <int N, int MODE>
static void xc_launch(const xc::XArgs& p, int G) {
  cudaFuncSetAttribute(xc::xc_pp<N, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, xc::PC<N>::SMEM);
  xc::xc_pp<N, MODE><<<G, xc::PNTH, xc::PC<N>::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int N, int MODE>
static std::vector<int64_t> xc_attr() {
  cudaFuncSetAttribute(xc::xc_pp<N, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, xc::PC<N>::SMEM);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, xc::xc_pp<N, MODE>);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, xc::xc_pp<N, MODE>, xc::PNTH, xc::PC<N>::SMEM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)xc::PC<N>::SMEM, (int64_t)xc::PNTH, occ, xc::PC<N>::NS, xc::PC<N>::NX};
}
#define XC_SWITCH(F, ...) \
  switch (npad) { \
    case 8: F<8, XC_MODE>(__VA_ARGS__); break; case 16: F<16, XC_MODE>(__VA_ARGS__); break; \
    case 24: F<24, XC_MODE>(__VA_ARGS__); break; case 32: F<32, XC_MODE>(__VA_ARGS__); break; \
    case 48: F<48, XC_MODE>(__VA_ARGS__); break; case 56: F<56, XC_MODE>(__VA_ARGS__); break; \
    case 64: F<64, XC_MODE>(__VA_ARGS__); break; case 96: F<96, XC_MODE>(__VA_ARGS__); break; \
    case 112: F<112, XC_MODE>(__VA_ARGS__); break; case 128: F<128, XC_MODE>(__VA_ARGS__); break; \
    default: TORCH_CHECK(false, "xc: npad not instantiated"); }
std::vector<int64_t> xc_attrs(int64_t npad) {
  std::vector<int64_t> r{-1};
  switch (npad) {
    case 8: r = xc_attr<8, XC_MODE>(); break; case 16: r = xc_attr<16, XC_MODE>(); break;
    case 24: r = xc_attr<24, XC_MODE>(); break; case 32: r = xc_attr<32, XC_MODE>(); break;
    case 48: r = xc_attr<48, XC_MODE>(); break; case 56: r = xc_attr<56, XC_MODE>(); break;
    case 64: r = xc_attr<64, XC_MODE>(); break; case 96: r = xc_attr<96, XC_MODE>(); break;
    case 112: r = xc_attr<112, XC_MODE>(); break; case 128: r = xc_attr<128, XC_MODE>(); break;
    default: break;
  }
  return r;
}
void xc_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,
            torch::Tensor y, torch::Tensor err, torch::Tensor marks, int64_t npad, int64_t G) {
  xc::XArgs p{};
  p.ws = reinterpret_cast<const uint8_t*>(ws.data_ptr()); p.gsc = gsc.data_ptr<float>();
  p.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr()); p.unit = reinterpret_cast<const int4*>(unit.data_ptr());
  p.cta = reinterpret_cast<const int2*>(cta.data_ptr()); p.out = out.data_ptr<float>(); p.y = y.data_ptr<float>();
  p.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.err = err.data_ptr<int>(); p.npad = (int)npad; p.mode = 0;
  XC_SWITCH(xc_launch, p, (int)G)
}
void xc_gbar_run(torch::Tensor cnt, torch::Tensor marks, int64_t rounds, torch::Tensor err, int64_t G) {
  xc::xc_gbar<<<(int)G, 32, 0, at::cuda::getCurrentCUDAStream()>>>(cnt.data_ptr<int>(), reinterpret_cast<long long*>(marks.data_ptr()),
                                                                    (int)rounds, err.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void xc_redbw_run(torch::Tensor y, int64_t iters, int64_t G) {
  xc::xc_redbw<<<(int)G, 512, 0, at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<float>(), (int)(y.numel() / 4), (int)iters);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
XC_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> xc_attrs(int64_t npad);
void xc_run(torch::Tensor ws, torch::Tensor gsc, torch::Tensor xp, torch::Tensor unit, torch::Tensor cta, torch::Tensor out,
            torch::Tensor y, torch::Tensor err, torch::Tensor marks, int64_t npad, int64_t G);
void xc_gbar_run(torch::Tensor cnt, torch::Tensor marks, int64_t rounds, torch::Tensor err, int64_t G);
void xc_redbw_run(torch::Tensor y, int64_t iters, int64_t G);
"""


def inc_probe_cu():
    """dr_probe's probe kernels with the drp_kernel N list widened to 40..72 (bf2: XSN 64 -> N <= 64; w72: XSN 72)."""
    s = DRP.PROBE_CU
    inst = "template __global__ void drp_kernel<32>(const Args A, const PArgs P);\n"
    assert s.count(inst) == 1
    s = s.replace(inst, inst + "".join(f"#if VM_XSN >= {n}\ntemplate __global__ void drp_kernel<{n}>(const Args A, const PArgs P);\n#endif\n"
                                       for n in (40, 48, 56, 64, 72)))
    for old, fmt in (("    case 32: return drp_attr<32>();\n", "    case {n}: return drp_attr<{n}>();\n"),
                     ("    case 32: drp_launch<32>(a, p, (int)G); break;\n", "    case {n}: drp_launch<{n}>(a, p, (int)G); break;\n")):
        assert s.count(old) == 1
        s = s.replace(old, old + "".join(f"#if VM_XSN >= {n}\n" + fmt.format(n=n) + "#endif\n" for n in (40, 48, 56, 64, 72)))
    return s


def build(arm, cuda, verbose=False, ptxas=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    fl = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    if ptxas:
        fl.append("-Xptxas=-v")
    head = cuda[:cuda.index("void vml_run(")]
    if arm in PP_ARMS:
        fl += list(DRP.arm_defines("inc")) + [f"-DXC_MODE={PP_ARMS[arm]}"]
        src, cpp, fns = head + XC_CU, XC_CPP, ["xc_run", "xc_attrs", "xc_gbar_run", "xc_redbw_run"]
    else:
        fl += list(W72_DEFS) if arm == "w72" else list(DRP.arm_defines("ctl" if arm == "ctl" else "inc"))
        src, cpp, fns = head + inc_probe_cu(), DRP.PROBE_CPP, ["drp_run", "drp_attrs"]
    tag = hashlib.sha1((cpp + src + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"xc_probe_{arm}_{tag}", cpp_sources=cpp, cuda_sources=src, functions=fns, extra_cuda_cflags=fl,
                      verbose=verbose)
    return mod, time.time() - t0


def arm_ns(arm):
    return INC_N if arm in ("inc", "ctl") else W72_N if arm == "w72" else PP_N


def main(argv=None):
    import concurrent.futures as cf
    import json
    import statistics
    import time

    import torch
    import judge as J
    import k2a_v37 as D
    import unp_patches as UNP

    argv = sys.argv[1:] if argv is None else argv
    log = lambda s: print(s, flush=True)  # noqa: E731
    arms = [a for a in str(DRP.arg(argv, "arms", ",".join(ARMS))).split(",") if a in ARMS]
    ntoks = tuple(int(v) for v in str(DRP.arg(argv, "ntoks", "8,16,32,56,64,112,128")).split(","))
    modes = tuple(str(DRP.arg(argv, "modes", "cold,warm")).split(","))
    rounds, K, warm_set = int(DRP.arg(argv, "rounds", 15)), int(DRP.arg(argv, "units", 3)), int(DRP.arg(argv, "warm-set", 8))
    out_p = ROOT / str(DRP.arg(argv, "out", "box_out/xc/probe.json"))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ns = DRP.vml_namespace({"torch": torch})
    cuda = ns["_vml_CUDA"]
    mods = {}
    co = "--compile-only" in argv
    with cf.ThreadPoolExecutor(len(arms)) as ex:
        futs = {a: ex.submit(build, a, cuda, co, co) for a in arms}
        for a, f in futs.items():
            try:
                mods[a], dt = f.result()
                log(f"[xc] built {a} in {dt:.0f} s")
            except Exception as e:  # noqa: BLE001
                log(f"[xc] build {a} FAILED: {e!r}"[:3000])
    if co:
        return 0
    arms = [a for a in arms if a in mods]
    for a in arms:
        for n in arm_ns(a):
            at = (mods[a].xc_attrs(n) if a in PP_ARMS else mods[a].drp_attrs(n))
            log(f"[xc] attrs {a} N{n}: {list(at)}")
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    prob = J.load_problem(ROOT / "problems/muse_moe_w4a16")
    ref = prob.ref
    w = ref.generate_weights(ref.WEIGHTS_SEED, device=dev)
    log(f"[xc] UNP L2 decode self-test: {UNP.selftest(20000)} words bit-exact vs BITPOS3")
    ws13 = torch.empty((32, 64, 16, D.SW3), dtype=torch.uint8, device=dev)
    for e in range(32):
        w13q, s13q = D.prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=UNP.L2POS)
        ws13[e] = D.stage_repack(w13q, s13q)
    gsc13 = (w["w13_gs"][:32].float() * D.W_SHIFT).contiguous()
    perm = D.tile_row_perm(8192, True, device=dev)
    xsrc = torch.randn((512, 4096), generator=torch.Generator(device="cpu").manual_seed(5), dtype=torch.float32).to(torch.bfloat16).to(dev)
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    marks = torch.zeros((G * 8,), dtype=torch.int64, device=dev)
    log(f"[xc] setup {time.time() - t0:.0f} s, {G} SMs, replay overhead {ov:.2f} us, gate {timer.gate_us} us, arms {arms}, "
        f"rounds {rounds}, {K} units (= {32 * K} 256-row stages) per CTA")

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

    res = {"G": G, "overhead_us": ov, "gate_us": timer.gate_us, "arms": arms, "rounds": rounds, "units_per_cta": K, "rows": []}
    for N in ntoks:
        npad = -(-N // 8) * 8
        live = [a for a in arms if npad in arm_ns(a)]
        if not live:
            continue
        for mode in modes:
            P = DRP.plan(N, mode, G, K, warm_set, xsrc, dev)
            ybuf = torch.zeros((64 * 512 * npad,), dtype=torch.float32, device=dev)
            outs = {a: torch.full((P["n"], 512, npad), float("nan"), dtype=torch.float32, device=dev) for a in live}

            def fn(a, mk=nomark, P=P, outs=outs, ybuf=ybuf):
                if a in PP_ARMS:
                    return lambda: mods[a].xc_run(ws13, gsc13, P["xp"], P["unit"], P["cta"], outs[a], ybuf, err, mk, P["npad"], G)
                return lambda: mods[a].drp_run(ws13, gsc13, P["xp"], P["unit"], P["cta"], outs[a], err, mk, P["npad"], G)
            chk, refs = {}, {}
            samp = sorted(set([0, P["n"] // 3, P["n"] // 2, (2 * P["n"]) // 3, P["n"] - 1]))
            for i in samp:
                e, U = P["units"][i]
                rows = perm[512 * U:512 * U + 512]
                W = DRP.dequant_rows64(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e], rows)
                xt = xsrc[torch.as_tensor(P["toks"][e], dtype=torch.long, device=dev)].double()
                refs[i] = W @ xt.T
            base = next((b for b in ("inc", "w72", "pp") if b in live), live[0])
            for a in live:
                err.zero_()
                fn(a)()
                torch.cuda.synchronize()
                o = outs[a]
                if a in ("ppt", "ppr16", "ppr1"):
                    chk[a] = {"err": int(err[0]), "ok": int(err[0]) == 0, "timing_only": True}
                    continue
                rel, mx = 0.0, 0.0
                for i, r in refs.items():
                    got = o[i, :, :N].double()
                    rel = max(rel, float((got - r).norm() / r.norm().clamp(min=1e-300)))
                    mx = max(mx, float((got - r).abs().max() / r.abs().max().clamp(min=1e-300)))
                padz = bool((o[:, :, N:] == 0).all()) if npad > N else True
                same = bool(torch.equal(o, outs[base])) if a != base else True
                chk[a] = {"err": int(err[0]), "rel_fp64": rel, "maxrel_fp64": mx, "pad_zero": padz, "finite": bool(torch.isfinite(o).all()),
                          "bitwise_" + base: same, "ok": int(err[0]) == 0 and rel < 1e-4 and padz and bool(torch.isfinite(o).all())}
            gr = {}
            for a in live:
                f = fn(a)
                with torch.cuda.stream(timer.stream):
                    f()
                    torch.cuda.synchronize()
                    g_ = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_, stream=timer.stream):
                        f()
                torch.cuda.synchronize()
                gr[a] = g_
            ts = {a: [] for a in live}
            for rd in range(rounds):
                k = rd % len(live)
                for a in live[k:] + live[:k]:
                    ts[a].append(max(timed(gr[a], mode == "cold", mode == "warm") - ov, 1e-3))
            us = {a: statistics.median(v) for a, v in ts.items()}
            span, ghz = {}, {}
            for a in live:
                marks.zero_()
                with torch.cuda.stream(timer.stream):
                    if mode == "cold":
                        timer.flush()
                    timer.gate()
                    fn(a, mk=marks)()
                torch.cuda.synchronize()
                m = marks.view(G, 8).cpu().double()
                ends = m[:, 1:3] if a in PP_ARMS else m[:, 1:5]
                span[a] = float(((ends.max(dim=1).values - m[:, 0]) / 1e3).median())
                cyc, nsx = m[:, 6] - m[:, 5], m[:, 1] - m[:, 0]
                okm = (cyc > 0) & (nsx > 0)
                ghz[a] = float((cyc[okm] / nsx[okm]).median()) if bool(okm.any()) else None
            stage = {a: us[a] / (32 * K) for a in live}
            wb = P["n"] * 32 * 2 * D.SW3
            row = {"N": N, "mode": mode, "us": us, "stage_us": stage, "span_us": span, "ghz": ghz,
                   "w_TBs": {a: wb / us[a] / 1e6 for a in live}, "check": chk}
            res["rows"].append(row)
            log(f"[xc] N={N:3d} {mode}: stage us " + " ".join(f"{a} {stage[a]:.3f}" for a in live)
                + " | TB/s " + " ".join(f"{a} {row['w_TBs'][a]:.2f}" for a in live)
                + " | span " + " ".join(f"{a} {span[a]:.1f}" for a in live)
                + " | GHz " + " ".join(f"{a} {ghz[a]:.2f}" for a in live if ghz[a])
                + " | ok " + " ".join(f"{a} {chk[a]['ok']}" for a in live)
                + " | ==" + base + " " + " ".join(f"{a} {chk[a].get('bitwise_' + base)}" for a in live if a != base and 'bitwise_' + base in chk[a]))
            del gr, outs, P, ybuf
            torch.cuda.empty_cache()
    if "--gbar" in argv and any(a in PP_ARMS for a in arms):
        mod = mods[next(a for a in arms if a in PP_ARMS)]
        R = 200
        cnt = torch.zeros((1,), dtype=torch.int32, device=dev)
        mk = torch.zeros((R * G * 2,), dtype=torch.int64, device=dev)
        err.zero_()
        mod.xc_gbar_run(cnt, mk, R, err, G)
        torch.cuda.synchronize()
        m = mk.view(R, G, 2).cpu().double()
        lat = (m[:, :, 1].max(dim=1).values - m[:, :, 0].max(dim=1).values) / 1e3        # last arrive -> last release
        lat0 = (m[:, :, 1].min(dim=1).values - m[:, :, 0].max(dim=1).values) / 1e3       # last arrive -> first release
        per = (m[1:, :, 1].max(dim=1).values - m[:-1, :, 1].max(dim=1).values) / 1e3     # round period
        res["gbar"] = {"err": int(err[0]), "lat_us_med": float(lat[5:].median()), "lat0_us_med": float(lat0[5:].median()),
                       "period_us_med": float(per[5:].median())}
        log(f"[xc] gbar: {res['gbar']}")
    if "--redbw" in argv and any(a in PP_ARMS for a in arms):
        mod = mods[next(a for a in arms if a in PP_ARMS)]
        res["redbw"] = {}
        for mb in (4, 16):
            y = torch.zeros((mb * 262144,), dtype=torch.float32, device=dev)
            iters = 64
            f = lambda y=y, iters=iters: mod.xc_redbw_run(y, iters, G)  # noqa: E731
            with torch.cuda.stream(timer.stream):
                f()
                torch.cuda.synchronize()
                g_ = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_, stream=timer.stream):
                    f()
            torch.cuda.synchronize()
            t = statistics.median(max(timed(g_, False, True) - ov, 1e-3) for _ in range(rounds))
            byt = G * 512 * iters * 16
            res["redbw"][mb] = {"us": t, "TBs": byt / t / 1e6}
            log(f"[xc] redbw {mb} MB: {t:.1f} us for {byt / 1e6:.1f} MB of red.add.v4.f32 = {byt / t / 1e6:.2f} TB/s")
    out_p.write_text(json.dumps(res, indent=1, default=str))
    log(f"[xc] wrote {out_p} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

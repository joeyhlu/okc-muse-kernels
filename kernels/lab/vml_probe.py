"""VML probe (step 2 of the VM-large brief, notes/study/vm_large.md): one-launch W13 consumer of the "WGS" structure on
K2A's ws13 stage layout, at a fixed token count per expert (N = 8 / 16 / 32 / 48 / 64), streaming from HBM.

Structure (our implementation of the public vjeux idea, no code taken): 1 CTA/SM, 576 threads = 4 consumer warpgroups
(16 warps) + 1 weight producer warp + 1 x producer warp. A unit = (expert job, 256-row block = two K2A 128-row blocks)
x 16 stages of 256 K. Per stage ONE 36,864 B weight slot (two 18,432 B bulk copies, evict_first) shared by the 4 WGs
(WG q: K2A block q >> 1, tiles (q & 1) * 4 + warp) and ONE x slot (N_pad x 512 B, K2A xw_u32 core-matrix layout) shared
too. The weight slot is released as soon as the stage's codes / scales are in registers (before any wgmma); the x slot
when the stage's last wgmma group retired (at the next stage's first wait_group 1): no drain at stage ends. RS-wgmma
m64nNk16, A = K2A hot_decode (BITPOS3) in registers, double-buffered, GS k16 steps per commit group, wait_group 1;
first wgmma of a unit with scale-d = 0.

mode (template): 0 full, 1 null consumer (wait / release only), 2 no wgmma (decode only), 4 late release (weight slot
released after a wait_group 0 at the stage end: the AW0 / HUC0 / K2A-hot ring discipline).
Work: unit list dealt as contiguous equal ranges (the probe uses a multiple of 132 units: no stream-K needed).
"""
import hashlib
import statistics
import time

import torch


def _wg_spec(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
    body = ("{\\n.reg .b64 dd;\\nmov.b64 dd, {%%%d, %%%d};\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 "
            "{%s}, {%s}, dd, %d, 1, 1, 0;\\n}\\n")
    s = ""
    for nm, sd, cons in (("wgmma_acc", 1, plus), ("wgmma_zero", 0, eq)):
        s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
              "uint32_t a3, uint32_t lo, uint32_t hi) {\n  asm volatile(\"" % (nm, n)
              + body % (na + 4, na + 5, n, outs, a, sd)
              + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}\n" % cons)
    return s


VP_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace vp {
constexpr int KDIM = 4096;
constexpr int SW = 18432;                 // K2A ws13 stage of one 128-row block: [tile 8][kb 4][lane][16 B] + scales 2 KB
constexpr int US = 2 * SW;                // one 256-row unit stage
#ifndef VP_NS
#define VP_NS 3
#endif
#ifndef VP_NX
#define VP_NX 3
#endif
constexpr int NS = VP_NS;                 // weight slots
constexpr int NX = VP_NX;                 // x slots
constexpr int XS = 64 * 512;              // x slot: <= 64 tokens x 256 K bf16
constexpr int NCW = 16;                   // consumer warps (4 warpgroups)
constexpr int NTH = 32 * NCW + 64;        // + weight producer warp + x producer warp
constexpr int BAR_OFF = NS * US + NX * XS;
constexpr int SMEM = BAR_OFF + 8 * 2 * (NS + NX);
constexpr long long WAIT_BUDGET = 1ll << 31;

struct Args {
  const uint8_t* ws13; const float* gsc13;
  const int4* job;     // expert, ntok, npad, x byte offset
  const int* jbase;    // first inter row of the job
  const int2* unit;    // (job, 256-row block 0..31)
  const int2* cta;     // [u0, u1) per CTA
  const uint8_t* xp; __nv_bfloat16* inter; int* err; long long* marks;
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // K2A BITPOS3 decode
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  int n = 0;
  while (!mbar_try(b, par)) {
    if ((++n & 255) == 0) {
      if (*reinterpret_cast<volatile int*>(err) != 0) return;
      if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); return; }
    }
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_acc(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <int N> __device__ __forceinline__ void wgmma_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
@WGSPECS@

// ---------------- producers (lane 0 of warps 16 / 17) ----------------
__device__ __forceinline__ void prod_w(const Args& A, uint32_t ring, uint32_t bars, int u0, int u1) {
  const uint64_t pol = policy_evict_first();
  int it = 0;
  for (int u = u0; u < u1; ++u) {
    const int2 un = A.unit[u];
    const int e = A.job[un.x].x;
    const uint8_t* wb = A.ws13 + ((size_t)e * 64 + 2 * un.y) * 16 * SW;
    for (int s = 0; s < 16; ++s, ++it) {
      const int sl = it % NS;
      if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
      const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
      mbar_expect_tx(fb, US);
      bulk_g2s_ef(dst, wb + (size_t)s * SW, SW, fb, pol);
      bulk_g2s_ef(dst + SW, wb + (size_t)(16 + s) * SW, SW, fb, pol);
    }
  }
}
__device__ __forceinline__ void prod_x(const Args& A, uint32_t xring, uint32_t bars, int u0, int u1) {
  int it = 0;
  for (int u = u0; u < u1; ++u) {
    const int4 jb = A.job[A.unit[u].x];
    const uint32_t bytes = (uint32_t)jb.z * 512u;
    const uint8_t* xs = A.xp + (size_t)jb.w;
    for (int s = 0; s < 16; ++s, ++it) {
      const int sl = it % NX;
      if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
      const uint32_t fb = bars + 8 * (2 * NS + sl);
      mbar_expect_tx(fb, bytes);
      bulk_g2s(xring + sl * XS, xs + (size_t)s * bytes, bytes, fb);
    }
  }
}

// ---------------- consumer ----------------
// one half group = 2 k16 steps: words (x | y) of row g and (z | w) of row g + 8, scales applied
__device__ __forceinline__ void dec_half(uint32_t wg_, uint32_t w8, uint32_t Sg, uint32_t S8, uint32_t* hh) {
  deq4(wg_, hh); deq4(w8, hh + 4);
#pragma unroll
  for (int j = 0; j < 4; ++j) { hh[j] = bmul2(hh[j], Sg); hh[4 + j] = bmul2(hh[4 + j], S8); }
}
template <int N, bool Z>
__device__ __forceinline__ void mma2(float* d, const uint32_t* hh, uint32_t dlo, uint32_t dhi) {
  if constexpr (Z) wgmma_zero<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  else wgmma_acc<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + 16, dhi);
}

template <int N, int MODE>
__device__ __forceinline__ void stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
                                      uint32_t (&hb)[2][8], int q, int wp, int lane, bool first, uint32_t& sink) {
  const int ws = it % NS, xs = it % NX;
  mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  if constexpr (MODE == 1) {
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
    mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
    ++it;
    return;
  }
  const uint8_t* st = smem + ws * US + (q >> 1) * SW;
  const int tile = (q & 1) * 4 + wp;
  const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
  uint4 c[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
  const uint32_t sw0 = sp[0], sw1 = sp[32];
  if constexpr (MODE != 4) {
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));     // codes + scales in registers: weight slot free
  }
  mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
  __syncwarp();
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);
  const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
    const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
    const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      uint32_t* hh = hb[hf];
      dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
      if constexpr (MODE == 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) sink ^= hh[j];
      } else {
        wg_fence();
        if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);
        else mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
        wg_commit();
        wg_wait<1>();
        if (i == 0 && hf == 0 && prevx >= 0) {   // the previous stage's last group retired: its x slot is free
          if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
          prevx = -1;
        }
      }
    }
  }
  if constexpr (MODE == 2) {
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
  } else {
    prevx = xs;
  }
  if constexpr (MODE == 4) {
    wg_wait<0>();
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
  }
  ++it;
}

template <int N, int MODE>
__device__ __forceinline__ void unit_run(const Args& A, uint8_t* smem, uint32_t bars, int& it, int u, int q, int wp, int lane,
                                      uint32_t& sink) {
  float d[N / 2];
  uint32_t hb[2][8];
  int prevx = -1;
  stage<N, MODE>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, sink);
#pragma unroll 1
  for (int s = 1; s < 16; ++s) stage<N, MODE>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, sink);
  if constexpr (MODE == 0 || MODE == 4) {
    wg_wait<0>();
    if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
    const int2 un = A.unit[u];
    const int4 jb = A.job[un.x];
    const float gs = __ldg(A.gsc13 + jb.x);
    const int g = lane >> 2, t = lane & 3;
    const int col = 8 * ((2 * un.y + (q >> 1)) * 8 + (q & 1) * 4 + wp) + g;
    const int rb = A.jbase[un.x];
#pragma unroll
    for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = 8 * nt + 2 * t + hh;
        if (n < jb.y)
          A.inter[(size_t)(rb + n) * KDIM + col] = __float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
      }
  }
}

template <int MODE>
__global__ void __launch_bounds__(NTH, 1) vp_kernel(const Args A) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = sa(smem + BAR_OFF);
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
    for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (A.marks) A.marks[(size_t)blockIdx.x * 8] = gtime();
  }
  __syncthreads();
  const int2 rg = A.cta[blockIdx.x];
  if (warp >= NCW) {
    if (lane == 0) {
      if (warp == NCW) prod_w(A, sa(smem), bars, rg.x, rg.y);
      else prod_x(A, sa(smem + NS * US), bars, rg.x, rg.y);
    }
    return;
  }
  const int q = warp >> 2, wp = warp & 3;
  int it = 0;
  uint32_t sink = 0;
  for (int u = rg.x; u < rg.y; ++u) {
    const int np = A.job[A.unit[u].x].z;
    switch (np >> 3) {
      case 1: unit_run<8, MODE>(A, smem, bars, it, u, q, wp, lane, sink); break;
      case 2: unit_run<16, MODE>(A, smem, bars, it, u, q, wp, lane, sink); break;
      case 3: unit_run<24, MODE>(A, smem, bars, it, u, q, wp, lane, sink); break;
      case 4: unit_run<32, MODE>(A, smem, bars, it, u, q, wp, lane, sink); break;
      case 5: unit_run<40, MODE>(A, smem, bars, it, u, q, wp, lane, sink); break;
      case 6: unit_run<48, MODE>(A, smem, bars, it, u, q, wp, lane, sink); break;
      case 7: unit_run<56, MODE>(A, smem, bars, it, u, q, wp, lane, sink); break;
      default: unit_run<64, MODE>(A, smem, bars, it, u, q, wp, lane, sink); break;
    }
  }
  if (sink == 0x9E3779B9u) A.err[3] = 1;
  if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 8 + 1 + q] = gtime();
}

template <int MODE>
void launch(const Args& a, int G) {
  cudaFuncSetAttribute(vp_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  vp_kernel<MODE><<<G, NTH, SMEM, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int MODE>
std::vector<int64_t> attrs() {
  cudaFuncSetAttribute(vp_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, vp_kernel<MODE>);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vp_kernel<MODE>, NTH, SMEM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SMEM, occ};
}
}  // namespace vp

std::vector<int64_t> vp_attrs(int64_t mode) {
  switch (mode) {
    case 0: return vp::attrs<0>();
    case 1: return vp::attrs<1>();
    case 2: return vp::attrs<2>();
    case 4: return vp::attrs<4>();
    default: return {-1};
  }
}

void vp_run(int64_t mode, torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor job, torch::Tensor jbase,
            torch::Tensor unit, torch::Tensor cta, torch::Tensor xp, torch::Tensor inter, torch::Tensor err,
            torch::Tensor marks, int64_t G) {
  vp::Args a;
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>();
  a.job = reinterpret_cast<const int4*>(job.data_ptr());
  a.jbase = jbase.data_ptr<int>();
  a.unit = reinterpret_cast<const int2*>(unit.data_ptr());
  a.cta = reinterpret_cast<const int2*>(cta.data_ptr());
  a.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr());
  a.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  a.err = err.data_ptr<int>();
  a.marks = marks.numel() >= G * 8 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  switch (mode) {
    case 0: vp::launch<0>(a, (int)G); break;
    case 1: vp::launch<1>(a, (int)G); break;
    case 2: vp::launch<2>(a, (int)G); break;
    case 4: vp::launch<4>(a, (int)G); break;
    default: TORCH_CHECK(false, "vp: bad mode");
  }
}
""".replace("@WGSPECS@", "".join(_wg_spec(n) for n in (8, 16, 24, 32, 40, 48, 56, 64)))

VP_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> vp_attrs(int64_t mode);
void vp_run(int64_t mode, torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor job, torch::Tensor jbase,
            torch::Tensor unit, torch::Tensor cta, torch::Tensor xp, torch::Tensor inter, torch::Tensor err,
            torch::Tensor marks, int64_t G);
"""
MODES = (0, 1, 2, 4)
NAMES = {0: "F", 1: "Z", 2: "D", 4: "L"}
FLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


FLAGS_EXTRA = []   # e.g. ["-DVP_NS=4", "-DVP_NX=2"]


def build(verbose=False):
    from torch.utils.cpp_extension import load_inline
    flags = FLAGS + list(FLAGS_EXTRA)
    tag = hashlib.sha1((VP_CPP + VP_CUDA + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vml_probe_{tag}", cpp_sources=VP_CPP, cuda_sources=VP_CUDA, functions=["vp_run", "vp_attrs"],
                      extra_cuda_cflags=flags, verbose=verbose)
    return mod, time.time() - t0


def pack_x(x, toks, npad):
    """x rows (zero-padded to npad) in K2A's xw_u32 wgmma B layout per 256-k stage:
    [s][n8][i 4][q 8][r 8][t 4][b 2] for k = 256 s + 64 i + 16 t + 2 q + b -> uint8 [16 * npad * 512]."""
    X = torch.zeros((npad, 4096), dtype=torch.bfloat16, device=x.device)
    X[:len(toks)] = x[torch.as_tensor(toks, dtype=torch.long, device=x.device)]
    X = X.view(npad // 8, 8, 16, 4, 4, 8, 2).permute(2, 0, 3, 5, 1, 4, 6).contiguous()
    return X.view(torch.uint8).reshape(-1)


def plan(ntok, n_units, G, dev, seed=7, n_exp=32):
    """n_units units (multiple of G) over experts 0.. in order, one job of ntok tokens per expert (random token rows of a
    random 256-row x), contiguous equal unit ranges per CTA."""
    assert n_units % G == 0
    ne = -(-n_units // 32)
    assert ne <= n_exp
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((256, 4096), generator=g).to(torch.bfloat16).to(dev)
    npad = -(-ntok // 8) * 8
    jobs, jbase, xparts, toks, xo = [], [], [], [], 0
    for e in range(ne):
        tk = torch.randperm(256, generator=g)[:ntok].tolist()
        xparts.append(pack_x(x, tk, npad))
        jobs.append((e, ntok, npad, xo))
        jbase.append(e * ntok)
        toks.append(tk)
        xo += 16 * npad * 512
    units = [(u // 32, u % 32) for u in range(n_units)]
    per = n_units // G
    cta = [(c * per, (c + 1) * per) for c in range(G)]
    t = lambda v: torch.tensor(v, dtype=torch.int32).to(dev).contiguous()  # noqa: E731
    return {"x": x, "toks": toks, "job": t(jobs), "jbase": t(jbase), "unit": t(units), "cta": t(cta),
            "xp": torch.cat(xparts).contiguous(), "rows": ne * ntok, "ne": ne, "units": n_units,
            "wbytes": n_units * 16 * 36864, "xbytes": n_units * 16 * npad * 512}


def ref_inter(P, w, e, dequant):
    x = P["x"].float()[torch.as_tensor(P["toks"][e], dtype=torch.long, device=P["x"].device)]
    W = dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e])
    h = x @ W.T
    return torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]


def probe(st, w, dequant, timer, ov, log=print, ntoks=(8, 16, 32, 48, 64), rounds=16, n_units=924):
    """Arms F / Z / D / L per N; checks F and L against FP32 for fully covered experts; per-CTA end marks for F."""
    dev = st["ws13"].device
    mod, dt = build()
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    at = {m: list(mod.vp_attrs(m)) for m in MODES}
    log(f"[vmlp] built in {dt:.0f} s, {G} SMs; attrs (regs, local, smem, occ): " +
        " ".join(f"{NAMES[m]} {a}" for m, a in at.items()))
    if n_units % G:
        n_units = (n_units // G) * G
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    marks = torch.zeros((G * 8,), dtype=torch.int64, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    rows = []
    for N in ntoks:
        P = plan(N, n_units, G, dev)
        inter = torch.zeros((P["rows"] + 8, 4096), dtype=torch.bfloat16, device=dev)

        def f(mode, mk=False, P=P, inter=inter):
            return lambda: mod.vp_run(mode, st["ws13"], st["gsc13"], P["job"], P["jbase"], P["unit"], P["cta"], P["xp"],
                                      inter, err, marks if mk else nomark, G)
        rel = {}
        full_e = [e for e in range(P["ne"]) if (e + 1) * 32 <= P["units"]]
        for m in (0, 4):
            err.zero_()
            inter.zero_()
            f(m)()
            torch.cuda.synchronize()
            worst = 0.0
            for e in (full_e[0], full_e[len(full_e) // 2], full_e[-1]):
                r = ref_inter(P, w, e, dequant)
                got = inter[e * N:(e + 1) * N].float()
                worst = max(worst, float((got - r).norm() / r.norm().clamp(min=1e-30)))
            rel[NAMES[m]] = worst if int(err[0]) == 0 and bool(torch.isfinite(inter.float()).all()) else float("inf")
        gr = {}
        for m in MODES:
            fn = f(m)
            with torch.cuda.stream(timer.stream):
                fn()
                torch.cuda.synchronize()
                g_ = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_, stream=timer.stream):
                    fn()
            torch.cuda.synchronize()
            gr[NAMES[m]] = g_
        names = list(gr)
        ts = {nm: [] for nm in names}
        for rd in range(rounds):
            k = rd % len(names)
            for nm in names[k:] + names[:k]:
                ts[nm].append(max(timer.time_replay(gr[nm]) - ov, 1e-3))
        us = {nm: statistics.median(v) for nm, v in ts.items()}
        marks.zero_()
        with torch.cuda.stream(timer.stream):
            timer.flush()
            f(0, mk=True)()
        torch.cuda.synchronize()
        mv = marks.view(G, 8).tolist()
        t00 = min(m_[0] for m_ in mv)
        ends = sorted((max(m_[1:5]) - t00) / 1e3 for m_ in mv)
        e1 = int(err[0])
        B = P["wbytes"]
        tb = {nm: B / us[nm] / 1e6 for nm in names}
        row = {"N": N, "us": us, "TBs": tb, "rel": rel, "err": e1, "end_p50": statistics.median(ends), "end_max": ends[-1],
               "end_min": ends[0], "wbytes": B, "xbytes": P["xbytes"], "stage_us": {nm: us[nm] * G / (P["units"] * 16) for nm in names}}
        rows.append(row)
        log(f"[vmlp] N={N:2d}: us " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items()) +
            " | W TB/s " + " ".join(f"{nm} {v:.2f}" for nm, v in tb.items()) +
            f" | F/Z {us['Z'] / us['F']:.3f} | stage us F {row['stage_us']['F']:.3f} | CTA end min/p50/max "
            f"{ends[0]:.1f}/{row['end_p50']:.1f}/{ends[-1]:.1f} | rel F {rel['F']:.2e} L {rel['L']:.2e} err {e1}")
        err.zero_()
        del gr, inter, P
    return rows

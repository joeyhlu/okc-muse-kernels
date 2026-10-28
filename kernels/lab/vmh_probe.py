"""vmh_probe: step-0 of indep_t32 design D1 (VMH = VMS cold units + whole 128-row hot super-units on RS-wgmma, one launch).

Question: inside a VMS-shaped CTA (288 threads = 8 compute warps + 1 helper warp, 1 CTA / SM, ring of 36 KB slots), can a
hot path stream the SAME VMS-layout weights (16-row x full-K units, already in the layer as VMS's wA / wB) through RS-wgmma
m64nNk16 at N = 16 / 24 / 32 near W2's measured stage cost (vml_probe R-vml-6: 1.41 / ~1.60 / 1.80 us per 36 KB), and
does the cold mma.sync path (VMS's consumer, x in registers) keep its ~1.10 us per unit when both paths live in one kernel
(X2S measured a 3 % codegen penalty for a second path)?

Hot stage = slice ws (512 K) of 8 consecutive VMS units (128 rows): 8 x 4096 B codes + 8 x 512 B scales, 16 bulk copies
into one 36,864 B ring slot. WG j (warps 4j..4j+3) takes units 4j..4j+3; each warp dequants its unit's slice with VMS's
exact arithmetic (8 chunks x 4 k16 steps), and the 4 resulting A registers per step are the RS-wgmma A fragment of its
16 rows (the m16n8k16 A layout = the wgmma RS A layout per warp). B = the stage's x slice [NH tokens x 512 K] in a separate
3-slot x ring, laid out as K-major no-swizzle core matrices with VMS's k permutation folded in (pack_xperm), LBO 128 B,
SBO 8192 B, one bulk copy per stage (in VMH the hot A epilogue / the prologue write this layout). A super-unit = the 8
slices of one 128-row group, accumulated in registers (m64nNH f32: NH / 2 per thread), stored after slice 7.
x slots are released one stage late (the wgmmas reading them are async): at the end of hot item q, item q - 1's slot is
free (every warp passed a wait_group 1 behind a later commit), and its next x copy is issued.

Arms (kernel<HOT, NH>, mode): cold0 = <0, -> cold units only (VMS's loop), cold1 = <1, 32> cold units only (penalty of the
hot path's presence), hotN = <1, N> hot super-units only, mix = <1, 32> 2 super-units then K cold units per CTA.
Correctness: the hot super-unit's 128 x NH outputs vs the cold path's outputs for the same units with the same tokens
(cold path = VMS's arithmetic; FP32 sums in another order: rel <= 1e-4 expected).
"""
from __future__ import annotations

import hashlib
import statistics
import time

import torch

UB, UC = 36864, 32768

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace vmh {
constexpr int HD = 4096, UB = 36864, UC = 32768, NW = 8, NT = NW * 32, NTL = NT + 32, NS = 3, NX = 3, NRB = 4;
constexpr int XSL = 32768;                                    // x slot: <= 32 tokens x 512 K bf16
constexpr long long WAIT_BUDGET = 1ll << 31;

struct P {
  const unsigned char* w; const __nv_bfloat16* x; const unsigned char* xp;
  float* yc; float* yh; long long* marks; int* err;
  int K, S, mode, xtok;
};
struct __align__(16) Smem {
  uint64_t full[NS], xfull[NX], pfull[NRB], pempty[NRB];
  float4 red[NRB][NW][32];
};
constexpr int RING_OFF = (int)((sizeof(Smem) + 127) / 128 * 128);
constexpr int XR_OFF = RING_OFF + NS * UB;
constexpr int SMEM_BYTES = XR_OFF + NX * XSL;

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) { asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(su32(b)), "r"(n)); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(su32(b)) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(su32(b)), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t par, int* err) {   // bounded (R9), no trap
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); return; }
  }
}
__device__ __forceinline__ void expect(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk(void* dst, const void* src, uint32_t bytes, uint64_t* bar, uint64_t pol, bool hint) {
  if (hint)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                 :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(bar)), "l"(pol) : "memory");
  else
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(bar)) : "memory");
}
__device__ __forceinline__ void cta_sync_compute() { asm volatile("bar.sync 1, %0;\n" :: "n"(NT) : "memory"); }
__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) { return __byte_perm(w, 0u, sel) * k16 + 0x5C005C00u; }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// VMS's dequant of one code word (8 e2m1 codes of rows g / g + 8, 4 k) into the 4 A registers of one k16 step
__device__ __forceinline__ void deq(uint32_t qw, uint32_t sg, uint32_t sg8, uint32_t (&a)[4]) {
  const uint32_t M = 0x81C081C0u;
  a[0] = bmul(qw & M, sg);
  a[1] = bmul((qw * 16u) & M, sg8);
  a[2] = bmul(__funnelshift_l(qw, qw, 10) & M, sg);
  a[3] = bmul(((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u), sg8);
}

// ---- RS-wgmma (A from registers, B from a K-major no-swizzle smem descriptor: LBO 128 B, SBO 8192 B)
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ uint64_t bdesc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(8192u >> 4) << 32);
}
template <int N, int Z> __device__ __forceinline__ void wgmma(float* d, const uint32_t (&a)[4], uint64_t db);
@WGSPECS@

template <int NH>
__device__ __forceinline__ void hot_stage(const unsigned char* slot, uint32_t xs_addr, float* d, int warp, int lane) {
  const int u = warp;                       // WG j = warps 4j..4j+3 -> units 4j..4j+3 of the super-unit (rows 16 u ..)
  const uint32_t k16 = 16u;
  const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + u * 512 + lane * 16);
  uint32_t hb[2][2][4];
#pragma unroll
  for (int cc = 0; cc < 8; ++cc) {
    const uint4 w = *reinterpret_cast<const uint4*>(slot + u * 4096 + cc * 512 + lane * 16);
    const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
    const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
    const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int gp = 0; gp < 2; ++gp) {        // commit group = 2 k16 steps; A double-buffered across groups
      const int b = (cc * 2 + gp) & 1;
      deq(ww[2 * gp], sg, sg8, hb[b][0]);
      deq(ww[2 * gp + 1], sg, sg8, hb[b][1]);
      wg_fence();
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s = 2 * gp + h;
        const uint64_t db = bdesc(xs_addr + (uint32_t)((cc * 4 + s) * 256));
        wgmma<NH, 0>(d, hb[b][h], db);       // always accumulate: a conditional zero-variant wgmma makes ptxas serialize (C7520)
      }
      wg_commit();
      wg_wait<1>();                          // group g - 1 done: its A buffer and (lagged) its x slot are free
    }
  }
}

template <int HOT, int NH>
__global__ void __launch_bounds__(NTL, 1) vmh_kernel(const P p) {
  extern __shared__ __align__(128) unsigned char smem[];
  Smem& S = *reinterpret_cast<Smem*>(smem);
  unsigned char* ring = smem + RING_OFF;
  unsigned char* xr = smem + XR_OFF;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x, G = gridDim.x;
  const int nh = HOT ? (p.mode == 0 ? 0 : 8 * p.S) : 0;           // hot items first
  const int N = nh + (p.mode == 1 ? 0 : p.K);
  const int hot0 = c * p.S * 8;                                  // first unit of this CTA's super-units
  const int cold0 = (p.mode == 2 ? G * p.S * 8 : 0) + c * p.K;   // first cold unit
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);
    for (int i = 0; i < NX; ++i) mbar_init(&S.xfull[i], 1);
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], 1); mbar_init(&S.pempty[i], 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  auto issue_w = [&](int q) {
    if (q >= N) return;
    uint64_t* bar = &S.full[q % NS];
    unsigned char* dst = ring + (q % NS) * UB;
    expect(bar, UB);
    if (HOT && q < nh) {
      const int su = q >> 3, ws = q & 7;
      const unsigned char* u0 = p.w + (size_t)(hot0 + 8 * su) * UB;
#pragma unroll 1
      for (int u = 0; u < 8; ++u) {
        bulk(dst + u * 4096, u0 + (size_t)u * UB + ws * 4096, 4096, bar, pol, true);
        bulk(dst + UC + u * 512, u0 + (size_t)u * UB + UC + ws * 512, 512, bar, pol, true);
      }
    } else {
      bulk(dst, p.w + (size_t)(cold0 + (q - nh)) * UB, UB, bar, pol, true);
    }
  };
  auto issue_x = [&](int q) {   // hot item q's x slice (shared by every super-unit: L2-resident, no evict hint)
    if (!HOT || q >= nh) return;
    uint64_t* bar = &S.xfull[q % NX];
    expect(bar, NH * 1024);
    bulk(xr + (q % NX) * XSL, p.xp + (size_t)(q & 7) * NH * 1024, NH * 1024, bar, pol, false);
  };
  long long* mrow = p.marks + (size_t)c * 4;
  if (tid == 0) {
    mrow[0] = gt();
    for (int q = 0; q < NS; ++q) issue_w(q);
    for (int q = 0; q < NX; ++q) issue_x(q);
  }
  if (warp < NW) {
    const int g = lane >> 2, t = lane & 3;
    uint4 xr16[16];
    if (N > nh) {   // the cold job's 8 tokens: VMS's per-warp x registers (k slice 512 warp .. + 511)
      const uint4* src = reinterpret_cast<const uint4*>(p.x + (size_t)(p.xtok + g) * HD) + warp * 64 + 2 * t;
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) { xr16[2 * cc] = __ldg(src + cc * 8); xr16[2 * cc + 1] = __ldg(src + cc * 8 + 1); }
    }
    const uint32_t k16 = 16u + (uint32_t)(p.K >> 30);   // opaque 16 (VMS: keeps the scale build an IMAD)
    float d[HOT ? NH / 2 : 1];
#pragma unroll
    for (int i = 0; i < (HOT ? NH / 2 : 1); ++i) d[i] = 0.f;
    if constexpr (HOT) {                         // hot super-units: straight-line wgmma / accumulator code (C7518 / C7520)
      for (int su = 0; su < nh / 8; ++su) {
#pragma unroll
        for (int i = 0; i < NH / 2; ++i) d[i] = 0.f;
        for (int ws = 0; ws < 8; ++ws) {
          const int q = su * 8 + ws;
          mbar_wait(&S.full[q % NS], (uint32_t)((q / NS) & 1), p.err);
          mbar_wait(&S.xfull[q % NX], (uint32_t)((q / NX) & 1), p.err);
          hot_stage<NH>(ring + (q % NS) * UB, su32(xr + (q % NX) * XSL), d, warp, lane);
          cta_sync_compute();
          if (tid == 0) {
            issue_w(q + NS);
            if (q >= 1) issue_x(q - 1 + NX);     // item q - 1's x slot: every warp passed a wait_group 1 after it
          }
        }
        wg_wait<0>();                            // super-unit done: store 128 rows x NH tokens (unit-local rows)
        float* y = p.yh + ((size_t)(c * p.S + su) * 128 + warp * 16) * NH;
#pragma unroll
        for (int i = 0; i < NH / 2; ++i) {
          const int rr = g + 8 * ((i >> 1) & 1), n = 8 * (i >> 2) + 2 * t + (i & 1);
          y[rr * NH + n] = d[i];
        }
      }
    }
    int ncold = 0;
    for (int q = nh; q < N; ++q) {               // VMS's cold unit (8 tokens, x in registers)
      const int sl = q % NS;
      mbar_wait(&S.full[sl], (uint32_t)((q / NS) & 1), p.err);
      const unsigned char* slot = ring + sl * UB;
      {
        const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16);
        float acc[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
#pragma unroll
        for (int cc = 0; cc < 8; ++cc) {
          const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * 8 + cc) * 512 + lane * 16);
          const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
          const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
          const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
          const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
          const uint32_t bb[8] = {xr16[2 * cc].x, xr16[2 * cc].y, xr16[2 * cc].z, xr16[2 * cc].w,
                                  xr16[2 * cc + 1].x, xr16[2 * cc + 1].y, xr16[2 * cc + 1].z, xr16[2 * cc + 1].w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            uint32_t a[4];
            deq(ww[s], sg, sg8, a);
            mma(acc[s], a[0], a[1], a[2], a[3], bb[2 * s], bb[2 * s + 1]);
          }
        }
        float4 o;
        o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
        o.y = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
        o.z = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
        o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
        const int bq = ncold & (NRB - 1);
        mbar_wait(&S.pempty[bq], (uint32_t)(((ncold / NRB) & 1) ^ 1), p.err);
        S.red[bq][warp][lane] = o;
        cta_sync_compute();
        if (tid == 0) { issue_w(q + NS); mbar_arrive(&S.pfull[bq]); }
        ++ncold;
      }
    }
  } else {                                       // helper warp: VMS's cold epilogue (sum the 8 K-slice partials)
    int ncold = 0;
    for (int q = nh; q < N; ++q, ++ncold) {
      const int bq = ncold & (NRB - 1);
      mbar_wait(&S.pfull[bq], (uint32_t)((ncold / NRB) & 1), p.err);
      float4 sm = S.red[bq][0][lane];
#pragma unroll
      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pempty[bq]);
      const int g = lane >> 2, t = lane & 3;
      float* y = p.yc + (size_t)(cold0 + (q - nh)) * 128;        // [16 unit rows][8 tokens]
      y[g * 8 + 2 * t] = sm.x; y[g * 8 + 2 * t + 1] = sm.y; y[(g + 8) * 8 + 2 * t] = sm.z; y[(g + 8) * 8 + 2 * t + 1] = sm.w;
    }
  }
  __syncthreads();
  if (tid == 0) mrow[1] = gt();
}
}  // namespace vmh

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

template <int HOT, int NH>
static void launch(const vmh::P& p, int G) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vmh::vmh_kernel<HOT, NH>, cudaFuncAttributeMaxDynamicSharedMemorySize, vmh::SMEM_BYTES));
    attr = true;
  }
  vmh::vmh_kernel<HOT, NH><<<G, vmh::NTL, vmh::SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(int64_t arm, torch::Tensor w, torch::Tensor x, torch::Tensor xp, torch::Tensor yc, torch::Tensor yh,
         torch::Tensor marks, torch::Tensor err, int64_t K, int64_t S, int64_t mode, int64_t xtok, int64_t G) {
  vmh::P p;
  p.w = w.data_ptr<uint8_t>(); p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.xp = xp.data_ptr<uint8_t>();
  p.yc = yc.data_ptr<float>(); p.yh = yh.data_ptr<float>(); p.marks = reinterpret_cast<long long*>(marks.data_ptr());
  p.err = err.data_ptr<int>(); p.K = (int)K; p.S = (int)S; p.mode = (int)mode; p.xtok = (int)xtok;
  switch (arm) {
    case 0: launch<0, 16>(p, (int)G); break;
    case 1: launch<1, 16>(p, (int)G); break;
    case 2: launch<1, 24>(p, (int)G); break;
    case 3: launch<1, 32>(p, (int)G); break;
    default: TORCH_CHECK(false, "vmh: arm 0..3");
  }
}

std::vector<int64_t> attrs(int64_t arm) {
  cudaFuncAttributes a;
  const void* f = arm == 0 ? (const void*)vmh::vmh_kernel<0, 16> : arm == 1 ? (const void*)vmh::vmh_kernel<1, 16>
                : arm == 2 ? (const void*)vmh::vmh_kernel<1, 24> : (const void*)vmh::vmh_kernel<1, 32>;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, f));
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)vmh::SMEM_BYTES};
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void run(int64_t arm, torch::Tensor w, torch::Tensor x, torch::Tensor xp, torch::Tensor yc, torch::Tensor yh,
         torch::Tensor marks, torch::Tensor err, int64_t K, int64_t S, int64_t mode, int64_t xtok, int64_t G);
std::vector<int64_t> attrs(int64_t arm);
"""

ARM_NH = {0: 16, 1: 16, 2: 24, 3: 32}   # arm 0 = cold-only kernel (HOT 0); NH unused there


def wg_specs():
    """wgmma<N, Z> specializations: m64nNk16 f32 += bf16 (A 4 registers) x bf16 (B desc); Z = 1 zeroes D (scale-d 0)."""
    out = []
    for n in (16, 24, 32):
        na = n // 2
        regs = ",".join(f"%{i}" for i in range(na))
        for z in (0, 1):
            cons = ", ".join(f'"{"=" if z else "+"}f"(d[{i}])' for i in range(na))
            out.append(
                f"template <> __device__ __forceinline__ void wgmma<{n}, {z}>(float* d, const uint32_t (&a)[4], uint64_t db) {{\n"
                f"  asm volatile(\"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, "
                f"{{%{na},%{na + 1},%{na + 2},%{na + 3}}}, %{na + 4}, {0 if z else 1}, 1, 1, 0;\\n\"\n"
                f"    : {cons}\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(db));\n}}")
    return "\n".join(out)


def cuda_src():
    return CUDA_SRC.replace("@WGSPECS@", wg_specs())


def build(verbose=False):
    from torch.utils.cpp_extension import load_inline
    src = cuda_src()
    flags = ["-O3", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((src + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vmh_probe_{tag}", cpp_sources=CPP_SRC, cuda_sources=src, functions=["run", "attrs"],
                      extra_cuda_cflags=flags, verbose=verbose)
    return mod, time.time() - t0


def make_units(n_units, dev, seed=11):
    """VMS-layout units (36,864 B): random code words, scale bytes 0x50..0x7E (in the PRMT decode's range)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(0, 256, (n_units, UB), dtype=torch.uint8, device=dev, generator=g)
    w[:, UC:] = torch.randint(0x50, 0x7F, (n_units, UB - UC), dtype=torch.uint8, device=dev, generator=g)
    return w


def pack_xperm(x, nh):
    """x [n <= nh, 4096] bf16 -> [8 slices][nh / 8][64 chunks][8 tokens][8] bf16: core matrix (G, c = (4 cc + s) 2 + half),
    token r, element e holds x[8 G + r][512 ws + 64 cc + 16 (e >> 1) + 4 s + 2 half + (e & 1)] (VMS's k order)."""
    dev = x.device
    xx = torch.zeros((nh, 4096), dtype=torch.bfloat16, device=dev)
    xx[:min(nh, x.shape[0])] = x[:nh]
    ws = torch.arange(8, device=dev).view(8, 1, 1, 1, 1, 1, 1, 1)
    G = torch.arange(nh // 8, device=dev).view(1, -1, 1, 1, 1, 1, 1, 1)
    cc = torch.arange(8, device=dev).view(1, 1, 8, 1, 1, 1, 1, 1)
    s = torch.arange(4, device=dev).view(1, 1, 1, 4, 1, 1, 1, 1)
    half = torch.arange(2, device=dev).view(1, 1, 1, 1, 2, 1, 1, 1)
    r = torch.arange(8, device=dev).view(1, 1, 1, 1, 1, 8, 1, 1)
    tp = torch.arange(4, device=dev).view(1, 1, 1, 1, 1, 1, 4, 1)
    j = torch.arange(2, device=dev).view(1, 1, 1, 1, 1, 1, 1, 2)
    k = 512 * ws + 64 * cc + 16 * tp + 4 * s + 2 * half + j
    n = 8 * G + r
    out = xx[n.expand(8, nh // 8, 8, 4, 2, 8, 4, 2), k.expand(8, nh // 8, 8, 4, 2, 8, 4, 2)]
    return out.contiguous().view(torch.uint8).view(-1)


def probe(timer, ov, log=print, K=(32, 64), S=(2, 4), rounds=12):
    dev = torch.device("cuda")
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    mod, dt = build()
    log(f"[vmh] built in {dt:.0f} s, {G} SMs; attrs (regs, local, smem): " +
        " ".join(f"arm{a} {list(mod.attrs(a))}" for a in range(4)))
    n_units = G * max(S) * 8 + G * max(K)
    w = make_units(n_units, dev)
    log(f"[vmh] {n_units} units = {n_units * UB / 1e6:.0f} MB")
    x = (torch.randn((32, 4096), device=dev) * 0.5).to(torch.bfloat16)
    xps = {nh: pack_xperm(x, nh) for nh in (16, 24, 32)}
    yc = torch.zeros((n_units * 128,), dtype=torch.float32, device=dev)
    yh = torch.zeros((G * max(S) * 128 * 32,), dtype=torch.float32, device=dev)
    marks = torch.zeros((G * 4,), dtype=torch.int64, device=dev)
    err = torch.zeros((4,), dtype=torch.int32, device=dev)

    # correctness: hot super-units (mode 1) vs the cold path on the same units (mode 0, 8 tokens at a time)
    s0 = min(S)
    chk = {}
    for arm in (1, 2, 3):
        nh = ARM_NH[arm]
        yh.zero_(); err.zero_()
        mod.run(arm, w, x, xps[nh], yc, yh, marks, err, 0, s0, 1, 0, G)
        torch.cuda.synchronize()
        hot = yh[:G * s0 * 128 * nh].view(G * s0, 8, 16, nh).clone()
        worst = 0.0
        for tb in range(0, nh, 8):
            yc.zero_()
            mod.run(0, w, x, xps[16], yc, yh, marks, err, G * s0 * 8 // G, 0, 0, tb, G)   # cold units 0 .. G s0 8 - 1
            torch.cuda.synchronize()
            cold = yc[:G * s0 * 8 * 128].view(G * s0, 8, 16, 8)
            ref = cold[..., : min(8, nh - tb)]
            got = hot[..., tb:tb + ref.shape[-1]]
            worst = max(worst, float((got - ref).norm() / ref.norm().clamp(min=1e-30)))
        chk[nh] = (worst, int(err[0]))
        log(f"[vmh] check hot N{nh} vs cold path: rel {worst:.2e} err {int(err[0])}")

    def graph(fn):
        with torch.cuda.stream(timer.stream):
            fn()
            torch.cuda.synchronize()
            g_ = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_, stream=timer.stream):
                fn()
        torch.cuda.synchronize()
        return g_

    arms = {}
    for k in K:
        arms[f"cold0_K{k}"] = (0, k, 0, 0)
        arms[f"cold1_K{k}"] = (3, k, 0, 0)
    for s in S:
        for arm in (1, 2, 3):
            arms[f"hot{ARM_NH[arm]}_S{s}"] = (arm, 0, s, 1)
    arms[f"mix_S{min(S)}_K{max(K)}"] = (3, max(K), min(S), 2)
    gr = {nm: graph(lambda a=a: mod.run(a[0], w, x, xps[ARM_NH[a[0]]], yc, yh, marks, err, a[1], a[2], a[3], 0, G))
          for nm, a in arms.items()}
    ts = {nm: [] for nm in gr}
    names = list(gr)
    for rd in range(rounds):
        kk = rd % len(names)
        for nm in names[kk:] + names[:kk]:
            ts[nm].append(max(timer.time_replay(gr[nm]) - ov, 1e-3))
    us = {nm: statistics.median(v) for nm, v in ts.items()}
    kl, kh = min(K), max(K)
    sl, sh = min(S), max(S)
    res = {"us": us, "check": chk, "err": int(err[0])}
    res["cold0_unit"] = (us[f"cold0_K{kh}"] - us[f"cold0_K{kl}"]) / (kh - kl)
    res["cold1_unit"] = (us[f"cold1_K{kh}"] - us[f"cold1_K{kl}"]) / (kh - kl)
    for nh in (16, 24, 32):
        res[f"hot{nh}_stage"] = (us[f"hot{nh}_S{sh}"] - us[f"hot{nh}_S{sl}"]) / (8 * (sh - sl))
    log("[vmh] us: " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items()))
    log(f"[vmh] per 36 KB: cold0 {res['cold0_unit']:.3f} cold1 {res['cold1_unit']:.3f} (x{res['cold1_unit'] / res['cold0_unit']:.3f}) "
        f"hot16 {res['hot16_stage']:.3f} hot24 {res['hot24_stage']:.3f} hot32 {res['hot32_stage']:.3f}")
    return res

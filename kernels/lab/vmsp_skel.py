"""VMSP step-0 skeleton (lead lever L8, 2026-10-29): ONE persistent launch for router + pooler + routing + pre-expert norm +
experts at T <= 8, on VMS's own ring (k3_gc_20's vms_kernel is the base of the expert part; that file is never edited).

Per CTA (G = SM count, 1 CTA / SM, 320 threads = 8 compute warps + helper warp 8 + planner warp 9 [+ observer warp 10]):
  unit sequence = nR router tiles (FP32, 8 rows x 1024 K = 32 KB each, tile i = slice (i >> 5), rows 8 (i & 31) ..)
                + nP pooler units (P12, 24 GD warp-units of 1536 B = 36 KB; warp w owns GD warp wk = w of each of the CTA's
                  nPT row tiles, 3 warp-units per unit, so a warp's K order and its 8-warp reduction match GD's)
                + VMS's A / B expert units (plan from the in-kernel routing).
  Ring: VMS's NS x 36,864 B slots, one cp.async.bulk per unit (first NS issued BEFORE griddepcontrol.wait: weights only),
  RO protocol (compute-warp bar.sync, tid 0 issues q + NS, per-warp red post, helper pfull / pempty).
Cross-CTA counters (ctr[3] router tiles done, ctr[4] tokens routed, ctr[5] pooler row tiles done; reset by CTA 0 after the
end barrier, like VMS's doneA):
  helper: after the CTA's last router unit -> red.release ctr[3] += nR; after each pooler row tile's last unit: bf16 y row
          tile + per-token sum-of-squares partial ssp[tile][t] -> red.release ctr[5] += 1.
  planner (warp 9): routing CTAs ((t * G) >> 3 = c for token t) wait ctr[3] = 256, run tail_kernel2's routing warp on the
          8 partial slices -> tw / tid / lid, red.release ctr[4] += 1; every CTA waits ctr[4] = m, runs VMS's route() on lid,
          publishes the plan (planbar); then waits ctr[5] = 256 and reduces ssp in a fixed order -> rstd[t] (rstdbar).
  compute: tid 0 waits planbar before its first expert issue; the first A x stage polls ctr[5], loads y (.cg) and the norm
          weight, waits rstdbar, xn = bf16((y * rstd) * w) (tail2's arithmetic up to the sum order) into VMS's x registers.
  End: VMS's launch_dependents, grid barrier, combine (tw via __ldcg).
Step-0 stand-ins (VMSP_MATH): 0 = null consumers everywhere (one LDS touch per unit + the real activation loads: router x
  slice, pooler h fragments, the real x stage, B inter loads); 1 = VMS's real expert consumer, null router / pooler.
  Router partials are injected (fpart: real FP32 partials from the host) by the helper -> real routing on "fake" logits;
  the pooler epilogue writes y = h[:, :4096] (bf16 copy) so the x stage and the experts see finite data.
VMSP_OBS 1: an observer warp records every unit's landing (globaltimer) into lmarks (marks builds only).
"""
from __future__ import annotations

import hashlib
import time

import torch

HD = 4096
KH = 8192
RT = 256
PT = 256
WU = 1536
UB = 36864
RTB = 32768
PTB = 16 * 8 * WU
NMK = 32
NLM = 96
MAXP = 128
MAXJ = 64
MULT = 0.1767766952966369
# k3_gc_20's VMS_POS (UNP L2 layout); the expert consumer below is k3_gc_20's
VMS_POS = (((6, 7, 8, 15), (22, 23, 24, 31)), ((28, 29, 30, 5), (12, 13, 14, 21)),
           ((25, 26, 27, 2), (9, 10, 11, 18)), ((4, 0, 1, 3), (20, 16, 17, 19)))

CUDA_SRC = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {
  uint32_t r;
  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));
  return r;
}
#ifndef VMSP_NS
#define VMSP_NS 3
#endif
#ifndef VMSP_G
#define VMSP_G 132
#endif
#ifndef VMSP_MATH
#define VMSP_MATH 0
#endif
#ifndef VMSP_OBS
#define VMSP_OBS 0
#endif
#if __CUDA_ARCH__ >= 900
#define VMSP_SM90 1
#else
#define VMSP_SM90 0
#endif

namespace vmsp {
constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 128, MAXJ = 64, KH = 8192;
constexpr int UB = 36864, UC = 32768, RBA = 512, RBB = 256;
constexpr int NW = 8, NT = NW * 32, NTL = NT + 64 + (VMSP_OBS ? 32 : 0);
constexpr int NRB = 4, NS = VMSP_NS, G = VMSP_G;
constexpr int RT = 256, PT = 256, WU = 1536, RTB = 32768, PTB = 16 * 8 * WU;
constexpr int NMK = 32, NLM = 96;
constexpr long long TMO = 20000000LL;   // 20 ms (globaltimer ns) bound on every cross-CTA wait -> sticky ctr[2]

struct Params {
  const float* rw; const float* rx; float* part; const float* fpart; const float* bias; const int* lstart;
  float* tw; int* tid; int* lid;
  const unsigned char* pw; const uint32_t* pesc; const __nv_bfloat16* h; __nv_bfloat16* y; float* ssp;
  const __nv_bfloat16* nwp; uint32_t tb;
  const unsigned char* wA; const unsigned char* wB; const float* gsA; const float* gsB; const float* nw;
  __nv_bfloat16* inter; float* ybuf; float* ss; int* ctr; __nv_bfloat16* out;
  long long* marks; long long* lmarks;
  int m; float eps; float peps; float mult; float rscale; float deps;
};

struct __align__(16) Smem {
  uint64_t full[NS];
  uint64_t pfull[NRB];
  uint64_t pempty[NRB];
  uint64_t planbar, rstdbar;
  int njobs, gen, nA, NEc, a0, b0, pad0, pad1;
  int base[4][NE];
  int jE[MAXJ], jS[MAXJ], jC[MAXJ];
  int ent[MAXP], pos[MAXP];
  float rstd[8];
  float4 red[NRB][NW][32];
};
constexpr int RING_OFF = (int)((sizeof(Smem) + 127) / 128 * 128);
constexpr int SMEM_BYTES = RING_OFF + NS * UB;

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(su32(b)), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(su32(b)) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint64_t* b, uint32_t par) {
  uint32_t ok;
#if VMSP_SM90
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(su32(b)), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(su32(b)), "r"(par) : "memory");
#endif
  return ok != 0u;
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t par) {
#if VMSP_SM90
  asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
               :: "r"(su32(b)), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
               :: "r"(su32(b)), "r"(par) : "memory");
#endif
}
__device__ __forceinline__ int ld_acq(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ int ld_rlx(const int* p) {
  int v; asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void red_rel(int* p, int v) {
  asm volatile("fence.acq_rel.gpu;\n red.release.gpu.global.add.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
// predecessor data (written before griddepcontrol.wait returned): plain coherent loads, never .nc (R-rpo)
__device__ __forceinline__ uint4 ldp4(const void* p) {
  uint4 r;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p) : "memory");
  return r;
}
__device__ __forceinline__ float4 ldpf4(const void* p) {
  const uint4 u = ldp4(p);
  return make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w));
}
__device__ __forceinline__ unsigned short ldp16(const void* p) {
  unsigned short r; asm volatile("ld.global.u16 %0, [%1];\n" : "=h"(r) : "l"(p) : "memory"); return r;
}
__device__ __forceinline__ bool poll_ge(const int* p, int target, int* err, int code) {
  const long long t0 = gt();
  int n = 0;
  while (ld_rlx(p) < target) {
    if ((++n & 31) == 0 && gt() - t0 > TMO) { atomicExch(err, code); return false; }
    __nanosleep(32);
  }
  return true;
}
__device__ __forceinline__ void cta_sync_compute() { asm volatile("bar.sync 1, %0;\n" :: "n"(NT) : "memory"); }

__device__ __forceinline__ void issue(const unsigned char* src, unsigned char* dst, uint64_t* bar, uint32_t bytes, int tid) {
#if VMSP_SM90
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(bar)), "l"(pol) : "memory");
#else
  for (int i = tid; i < (int)(bytes / 16); i += NT)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst + i * 16)), "l"(src + i * 16) : "memory");
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(su32(bar)) : "memory");
#endif
}

__device__ __forceinline__ void prefix_src(const Params& p, int u, int rt0, int nR, int pt0, int nPT,
                                           const unsigned char*& src, uint32_t& bytes) {
  if (u < nR) {
    src = reinterpret_cast<const unsigned char*>(p.rw) + (size_t)(rt0 + u) * RTB;
    bytes = RTB;
  } else {
    const int k = u - nR, wu = min(3, 16 * nPT - 3 * k);
    src = p.pw + (size_t)pt0 * PTB + (size_t)k * UB;
    bytes = (uint32_t)(8 * wu * WU);
  }
}
__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {
  if (q < nA) { const int v = a0 + q; return p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB; }
  const int v = b0 + (q - nA); return p.wB + ((size_t)S.jE[v >> 8] * RBB + (v & 255)) * UB;
}

__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {
  return __byte_perm(w, 0u, sel) * k16 + 0x5C005C00u;
}
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// xn = bf16((v * rstd) * w) on 8 bf16 (tail_kernel2's arithmetic)
__device__ __forceinline__ uint32_t xn2(uint32_t yv, uint32_t wv, float r) {
  const float a = (__uint_as_float(yv << 16) * r) * __uint_as_float(wv << 16);
  const float b = (__uint_as_float(yv & 0xFFFF0000u) * r) * __uint_as_float(wv & 0xFFFF0000u);
  __nv_bfloat162 o = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&o);
}
__device__ __forceinline__ uint4 xn8(uint4 yv, uint4 wv, float r) {
  return make_uint4(xn2(yv.x, wv.x, r), xn2(yv.y, wv.y, r), xn2(yv.z, wv.z, r), xn2(yv.w, wv.w, r));
}

__global__ void __launch_bounds__(NTL, 1) vmsp_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  Smem& S = *reinterpret_cast<Smem*>(smem);
  unsigned char* ring = smem + RING_OFF;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x;
  const bool mk = p.marks != nullptr;
  long long* mrow = p.marks + (size_t)c * NMK;
  const int m = p.m;
  const int rt0 = c * RT / G, nR = (c + 1) * RT / G - rt0;
  const int pt0 = c * PT / G, nPT = (c + 1) * PT / G - pt0;
  const int nP = (16 * nPT + 2) / 3, nRP = nR + nP;
  if (tid == 0) {
    if (mk) mrow[0] = gt();
#if VMSP_SM90
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);
#else
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], NT);
#endif
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], NW); mbar_init(&S.pempty[i], 1); }
    mbar_init(&S.planbar, 1);
    mbar_init(&S.rstdbar, 1);
#if VMSP_SM90
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
  __syncthreads();
  if (warp < NW) {   // pre-wait: the first NS weight units (constants only)
    for (int q = 0; q < NS && q < nRP; ++q) {
#if VMSP_SM90
      if (tid == 0)
#endif
      {
        const unsigned char* src; uint32_t bytes;
        prefix_src(p, q, rt0, nR, pt0, nPT, src, bytes);
        issue(src, ring + q * UB, &S.full[q], bytes, tid);
      }
    }
  }
#if VMSP_SM90
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  if (mk && tid == 0) { mrow[1] = gt(); unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); mrow[25] = sm;
                        mrow[27] = nR | (nP << 8) | (nPT << 16); }

  if (warp < NW) {
    // ================================ compute warps ================================
    const int g = lane >> 2, t = lane & 3;
    uint32_t touch = 0u;
    int sl = 0; uint32_t par = 0;
    int NEc = 0, nA = 0, a0 = 0, b0 = 0;
    bool planseen = false;
    float4 rxr[8];
#pragma unroll
    for (int k8 = 0; k8 < 8; ++k8) rxr[k8] = make_float4(0.f, 0.f, 0.f, 0.f);
    int rsl = -1;
    auto post = [&](int q, float4 o) {
      const int bq = q & (NRB - 1);
      cta_sync_compute();
      const int u = q + NS;
#if VMSP_SM90
      if (tid == 0)
#endif
      {
        if (u < nRP) {
          const unsigned char* src; uint32_t bytes;
          prefix_src(p, u, rt0, nR, pt0, nPT, src, bytes);
          issue(src, ring + sl * UB, &S.full[sl], bytes, tid);
        } else {
          if (!planseen) {
            mbar_wait(&S.planbar, 0u);
            planseen = true;
            NEc = S.NEc; nA = S.nA; a0 = S.a0; b0 = S.b0;
          }
          const int ue = u - nRP;
          if (ue < NEc) {
            if (mk && tid == 0 && ue == 0) mrow[18] = gt();
            issue(unit_src(p, S, ue, nA, a0, b0), ring + sl * UB, &S.full[sl], (uint32_t)UB, tid);
          }
        }
      }
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
      S.red[bq][warp][lane] = o;
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pfull[bq]);
      if (++sl == NS) { sl = 0; par ^= 1u; }
    };
    // ---------------- prefix: router tiles, then pooler units
    for (int q = 0; q < nRP; ++q) {
      uint4 hx[3][2];
      int wu = 0;
      if (q < nR) {
        const int slc = (rt0 + q) >> 5;
        if (slc != rsl) {   // the router input slice of this warp's token (T=1: every warp = one row of token 0)
          rsl = slc;
          const int tk = min(warp, m - 1);
          const float* src = p.rx + (size_t)tk * KH + slc * 1024 + lane * 4;
#pragma unroll
          for (int k8 = 0; k8 < 8; ++k8) rxr[k8] = ldpf4(src + k8 * 128);
        }
      } else {
        const int k = q - nR;
        wu = min(3, 16 * nPT - 3 * k);
        const int tk = min(g, m - 1);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          if (j < wu) {
            const int ch = (3 * k + j) & 15;
            const __nv_bfloat16* src = p.h + (size_t)tk * KH + (warp * 16 + ch) * 64 + 16 * t;
            hx[j][0] = ldp4(src);
            hx[j][1] = ldp4(src + 8);
          } else {
            hx[j][0] = make_uint4(0u, 0u, 0u, 0u); hx[j][1] = hx[j][0];
          }
        }
      }
      mbar_wait(&S.full[sl], par);
      if (mk && tid == 0) {
        if (q == 0) mrow[2] = gt();
        if (q == nR) mrow[4] = gt();
        if (q == nRP - 1) mrow[5] = gt();
      }
      const unsigned char* slot = ring + sl * UB;
      if (q < nR) {
        const uint4 w = *reinterpret_cast<const uint4*>(slot + tid * 16);
        uint32_t a = 0u;
#pragma unroll
        for (int k8 = 0; k8 < 8; ++k8) a ^= __float_as_uint(rxr[k8].x) ^ __float_as_uint(rxr[k8].w);
        touch ^= w.x ^ a;
      } else {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          if (j < wu) {
            const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * wu + j) * WU + lane * 16);
            touch ^= w.x ^ hx[j][0].x ^ hx[j][1].w;
          }
        }
      }
      post(q, make_float4(0.f, 0.f, 0.f, 0.f));
      if (mk && tid == 0 && q == nR - 1) mrow[3] = gt();
    }
    if (mk && tid == 0) mrow[6] = gt();
    mbar_wait(&S.planbar, 0u);
    planseen = true;
    NEc = S.NEc; nA = S.nA; a0 = S.a0; b0 = S.b0;
    // ---------------- experts: VMS's loop (k3_gc_20 vms_kernel, RO) with the in-kernel x stage
    const uint32_t k16 = 16u + (uint32_t)(m >> 30);
    uint4 xr[16];
    int curj = -1;
    bool firstA = true;
    for (int qe = 0; qe < NEc; ++qe) {
      const int q = nRP + qe;
      const bool PB = qe >= nA;
      const int v = PB ? b0 + (qe - nA) : a0 + qe;
      const int j = PB ? (v >> 8) + 4096 : (v >> 9);
      if (j != curj) {
        curj = j;
        const int jj = PB ? j - 4096 : j;
        const int cnt = S.jC[jj], st = S.jS[jj];
        if (PB) {
          if (mk && tid == 0 && mrow[28] == 0) mrow[28] = gt();
          if (lane == 0) poll_ge(p.ctr + 8 + jj, RBA, p.ctr + 2, 1);
          __syncwarp();
          (void)ld_acq(p.ctr + 8 + jj);
          if (mk && tid == 0 && mrow[20] == 0) mrow[20] = gt();
        } else if (firstA) {
          firstA = false;
          if (lane == 0) poll_ge(p.ctr + 5, PT, p.ctr + 2, 6);
          __syncwarp();
          (void)ld_acq(p.ctr + 5);
          if (mk && tid == 0) mrow[15] = gt();
        }
        if (g < cnt) {
          if (PB) {
            const uint4* src = reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID) + warp * 64 + 2 * t;
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) { xr[2 * cc] = __ldcg(src + cc * 8); xr[2 * cc + 1] = __ldcg(src + cc * 8 + 1); }
          } else {
            const int tok = S.ent[st + g] >> 3;
            const uint4* src = reinterpret_cast<const uint4*>(p.y + (size_t)tok * HD) + warp * 64 + 2 * t;
            const uint4* wsrc = reinterpret_cast<const uint4*>(p.nwp) + warp * 64 + 2 * t;
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) { xr[2 * cc] = __ldcg(src + cc * 8); xr[2 * cc + 1] = __ldcg(src + cc * 8 + 1); }
            mbar_wait(&S.rstdbar, 0u);
            const float r = S.rstd[tok];
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) {
              xr[2 * cc] = xn8(xr[2 * cc], __ldg(wsrc + cc * 8), r);
              xr[2 * cc + 1] = xn8(xr[2 * cc + 1], __ldg(wsrc + cc * 8 + 1), r);
            }
          }
        } else {
#pragma unroll
          for (int cc = 0; cc < 16; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);
        }
        if (!PB && mk && tid == 0 && mrow[16] == 0) mrow[16] = gt();
      }
      mbar_wait(&S.full[sl], par);
      if (mk && tid == 0) {
        if (qe == 0) mrow[17] = gt();
        if (qe == nA - 1) mrow[19] = gt();
      }
      const unsigned char* slot = ring + sl * UB;
      float4 o;
#if VMSP_MATH >= 1
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
          const uint32_t bb[8] = {xr[2 * cc].x, xr[2 * cc].y, xr[2 * cc].z, xr[2 * cc].w,
                                  xr[2 * cc + 1].x, xr[2 * cc + 1].y, xr[2 * cc + 1].z, xr[2 * cc + 1].w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const uint32_t qw = ww[s];
            const uint32_t M = 0x81C081C0u;
            const uint32_t r0 = qw & M;
            const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;
            const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;
            const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
            mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);
          }
        }
        o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
        o.y = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
        o.z = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
        o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
      }
#else
      {
        const uint4 w = *reinterpret_cast<const uint4*>(slot + tid * 16);
        touch ^= w.x ^ xr[0].x ^ xr[15].w;
        o = make_float4(0.f, 0.f, 0.f, 0.f);
      }
#endif
      post(q, o);
    }
    if (mk && tid == 0) mrow[21] = gt();
    if (m < 0) p.ctr[6] = (int)touch;   // never: keeps the stand-in loads alive
  } else if (warp == NW) {
    // ================================ helper warp: epilogues ================================
    const int g = lane >> 2, t4 = lane & 3;
    for (int q = 0; q < nRP; ++q) {
      const int bq = q & (NRB - 1);
      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
      float4 sm = S.red[bq][0][lane];
#pragma unroll
      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pempty[bq]);
      if (q < nR) {
        if (q == nR - 1) {
          if (p.fpart != nullptr) {   // step 0: the injected FP32 partials of this CTA's tiles
            for (int i = 0; i < nR; ++i) {
              const int ti = rt0 + i, slc = ti >> 5, bx = ti & 31;
              for (int e = lane; e < m * 8; e += 32) {
                const size_t o = ((size_t)slc * m + (e >> 3)) * 256 + bx * 8 + (e & 7);
                p.part[o] = __ldg(p.fpart + o);
              }
            }
          }
          __syncwarp();
          if (lane == 0) red_rel(p.ctr + 3, nR);
          if (mk && lane == 0) mrow[7] = gt();
        }
      } else {
        const int k = q - nR;
        for (int i = 0; i < nPT; ++i) {
          if ((16 * i + 15) / 3 != k) continue;
          const int tile = pt0 + i;
          float v[4] = {sm.x, sm.y, sm.z, sm.w};
          float sq0 = 0.f, sq1 = 0.f;
#pragma unroll
          for (int cq = 0; cq < 4; ++cq) {
            const int row = tile * 16 + g + 8 * (cq >> 1), tok = 2 * t4 + (cq & 1);
            if (tok < m) {
#if VMSP_MATH < 2
              { const unsigned short hb = ldp16(p.h + (size_t)tok * KH + row); v[cq] = __uint_as_float((uint32_t)hb << 16); }
#endif
              const __nv_bfloat16 b = __float2bfloat16_rn(v[cq]);
              p.y[(size_t)tok * HD + row] = b;
              const float f = __bfloat162float(b);
              if (cq & 1) sq1 += f * f; else sq0 += f * f;
            }
          }
#pragma unroll
          for (int o = 4; o < 32; o <<= 1) { sq0 += __shfl_xor_sync(0xffffffffu, sq0, o); sq1 += __shfl_xor_sync(0xffffffffu, sq1, o); }
          if (g == 0) {
            if (2 * t4 < m) p.ssp[tile * 8 + 2 * t4] = sq0;
            if (2 * t4 + 1 < m) p.ssp[tile * 8 + 2 * t4 + 1] = sq1;
          }
          __syncwarp();
          if (lane == 0) red_rel(p.ctr + 5, 1);
          if (mk && lane == 0) mrow[8] = gt();
        }
      }
    }
    mbar_wait(&S.planbar, 0u);
    const int NEc = S.NEc, nA = S.nA, a0 = S.a0, b0 = S.b0, a1 = a0 + nA;
    float* ssc = p.ss + (S.gen & 1) * MAXP;
    for (int qe = 0; qe < NEc; ++qe) {
      const int q = nRP + qe;
      const int bq = q & (NRB - 1);
      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
      float4 sm = S.red[bq][0][lane];
#pragma unroll
      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pempty[bq]);
      if (qe < nA) {
        const int v = a0 + qe, j = v >> 9, rb = v & 511;
        const int cnt = S.jC[j], st = S.jS[j];
        const float gs = __ldg(p.gsA + S.jE[j]);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int tk = 2 * t4 + hh;
          if (tk < cnt) {
            const float ga = (hh ? sm.y : sm.x) * gs, up = (hh ? sm.w : sm.z) * gs;
            const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;
            p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);
          }
        }
        if (qe == nA - 1 || ((v + 1) >> 9) != j) {
          __threadfence();
          __syncwarp();
          if (lane == 0) atomicAdd(p.ctr + 8 + j, min(a1, (j + 1) * RBA) - max(a0, j * RBA));
        }
      } else {
        const int v = b0 + (qe - nA), j = v >> 8, rb = v & 255;
        const int cnt = S.jC[j], st = S.jS[j];
        const float gs = __ldg(p.gsB + S.jE[j]);
        const int n0 = rb * 16 + g;
        const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int tk = 2 * t4 + hh;
          const float y0 = (hh ? sm.y : sm.x) * w0, y1 = (hh ? sm.w : sm.z) * w1;
          float sq = y0 * y0 + y1 * y1;
          if (tk < cnt) {
            p.ybuf[(size_t)(st + tk) * HD + n0] = y0;
            p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;
          }
          sq += __shfl_xor_sync(0xffffffffu, sq, 4);
          sq += __shfl_xor_sync(0xffffffffu, sq, 8);
          sq += __shfl_xor_sync(0xffffffffu, sq, 16);
          if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);
        }
      }
    }
    __threadfence();
    if (mk && lane == 0) mrow[22] = gt();
  } else if (warp == NW + 1) {
    // ================================ planner warp: routing, plan, rstd ================================
    const float4* pbias = reinterpret_cast<const float4*>(p.bias) + lane * 2;
    const float4 c0 = __ldg(pbias), c1 = __ldg(pbias + 1);
    int rtok = -1;
#pragma unroll
    for (int tt = 0; tt < 8; ++tt) if (tt < m && ((tt * G) >> 3) == c) rtok = tt;
    if (rtok >= 0) {   // tail_kernel2's routing warp for token rtok (bitwise its arithmetic)
      if (lane == 0) poll_ge(p.ctr + 3, RT, p.ctr + 2, 3);
      __syncwarp();
      (void)ld_acq(p.ctr + 3);
      if (mk && lane == 0) mrow[9] = gt();
      float4 qa[8][2];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4* pq = reinterpret_cast<const float4*>(p.part + ((size_t)q * m + rtok) * 256) + lane * 2;
        qa[q][0] = __ldcg(pq);
        qa[q][1] = __ldcg(pq + 1);
      }
      float s[8], b[8];
      float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
      for (int q = 1; q < 8; ++q) {
        a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
        a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
      }
      const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        s[jj] = 1.f / (1.f + expf(-lg[jj] * p.mult));
        b[jj] = s[jj] + bs[jj];
      }
      int sel_id[8];
      float sel_s[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        float bv = b[0], sv = s[0];
        int bj = 0;
#pragma unroll
        for (int jj = 1; jj < 8; ++jj)
          if (b[jj] > bv) { bv = b[jj]; sv = s[jj]; bj = jj; }
        const unsigned u = __float_as_uint(bv + 0.f);
        const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        const unsigned mx = __reduce_max_sync(0xffffffffu, key);
        const int owner = __ffs(__ballot_sync(0xffffffffu, key == mx)) - 1;
        sel_id[k] = __shfl_sync(0xffffffffu, lane * 8 + bj, owner);
        sel_s[k] = __shfl_sync(0xffffffffu, sv, owner);
        if (lane == owner) {
#pragma unroll
          for (int jj = 0; jj < 8; ++jj)
            if (jj == bj) b[jj] = -INFINITY;
        }
      }
      if (lane == 0) {
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) sum += sel_s[i];
        const float inv = p.rscale / (sum + p.deps);
        const int ls = __ldg(p.lstart);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          p.tw[rtok * 8 + k] = sel_s[k] * inv;
          p.tid[rtok * 8 + k] = sel_id[k];
          const int rel = ((sel_id[k] - ls) % 256 + 256) % 256;
          p.lid[rtok * 8 + k] = rel < 32 ? rel : -1;
        }
        red_rel(p.ctr + 4, 1);
        if (mk) mrow[10] = gt();
      }
    }
    if (lane == 0) poll_ge(p.ctr + 4, m, p.ctr + 2, 4);
    __syncwarp();
    (void)ld_acq(p.ctr + 4);
    if (mk && lane == 0) mrow[11] = gt();
    {   // VMS's route(): jobs of <= 8 pairs per local expert, ent / pos tables
      const int m8 = m * 8;
      int idv[4]; unsigned mm[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? (__ldcg(p.lid + i) & 255) : 255; }
      if (lane == 0) S.gen = ld_rlx(p.ctr);
#pragma unroll
      for (int r = 0; r < 4; ++r) { if (idv[r] >= NE || idv[r] < 0) idv[r] = -1; S.base[r][lane] = 0; }
      __syncwarp();
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        mm[r] = __match_any_sync(0xffffffffu, idv[r]);
        if (idv[r] >= 0 && lane == __ffs(mm[r]) - 1) S.base[r][idv[r]] = __popc(mm[r]);
      }
      __syncwarp();
      int cr[4], tot = 0;
#pragma unroll
      for (int r = 0; r < 4; ++r) { cr[r] = S.base[r][lane]; tot += cr[r]; }
      int inc = tot, nj = (tot + 7) >> 3, jinc = nj;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int a = __shfl_up_sync(0xffffffffu, inc, d), bb = __shfl_up_sync(0xffffffffu, jinc, d);
        if (lane >= d) { inc += a; jinc += bb; }
      }
      const int off = inc - tot, joff = jinc - nj;
      for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + 8 * k; S.jC[joff + k] = min(8, tot - 8 * k); }
      int run = off;
#pragma unroll
      for (int r = 0; r < 4; ++r) { S.base[r][lane] = run; run += cr[r]; }
      const int njobs = __shfl_sync(0xffffffffu, jinc, 31);
      __syncwarp();
      const unsigned lt = (1u << lane) - 1u;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
        if (i < m8) {
          if (idv[r] >= 0) { const int pr = S.base[r][idv[r]] + __popc(mm[r] & lt); S.ent[pr] = i; S.pos[i] = pr; }
          else S.pos[i] = -1;
        }
      }
      if (lane == 0) {
        const int NA = njobs * RBA, NN = njobs * (RBA + RBB);
        const int a0 = c * NA / G, a1 = (c + 1) * NA / G, t0 = c * NN / G, t1 = (c + 1) * NN / G;
        S.njobs = njobs; S.a0 = a0; S.nA = a1 - a0; S.b0 = t0 - a0; S.NEc = t1 - t0;
        if (mk) mrow[26] = (long long)(t1 - t0) | ((long long)(a1 - a0) << 16) | ((long long)njobs << 32);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.planbar);
      if (mk && lane == 0) mrow[12] = gt();
    }
    if (lane == 0) poll_ge(p.ctr + 5, PT, p.ctr + 2, 5);
    __syncwarp();
    (void)ld_acq(p.ctr + 5);
    if (mk && lane == 0) mrow[13] = gt();
    {   // rstd[t] from the 256 row-tile partials, fixed order (lane = 8 consecutive tiles, then a butterfly)
      float acc[8];
#pragma unroll
      for (int tt = 0; tt < 8; ++tt) acc[tt] = 0.f;
      const float4* sp = reinterpret_cast<const float4*>(p.ssp) + lane * 16;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float4 v = __ldcg(sp + i);
        const int b4 = (i & 1) * 4;
        acc[b4 + 0] += v.x; acc[b4 + 1] += v.y; acc[b4 + 2] += v.z; acc[b4 + 3] += v.w;
      }
      float mine = 0.f;
#pragma unroll
      for (int tt = 0; tt < 8; ++tt) {
        float a = acc[tt];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == tt) mine = a;
      }
      if (lane < 8 && lane < m) S.rstd[lane] = rsqrtf(mine / (float)HD + p.peps);
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.rstdbar);
      if (mk && lane == 0) mrow[14] = gt();
    }
  }
#if VMSP_OBS
  else if (warp == NW + 2) {   // observer: landing time of every unit (marks builds only)
    if (lane == 0 && p.lmarks != nullptr) {
      long long* lm = p.lmarks + (size_t)c * NLM;
      int sl = 0; uint32_t par = 0, ntot = nRP;
      for (int q = 0;; ++q) {
        if (q == nRP) { mbar_wait(&S.planbar, 0u); ntot = nRP + S.NEc; }
        if (q >= (int)ntot || q >= NLM) break;
        const long long t0 = gt();
        bool ok = true;
        while (!mbar_test(&S.full[sl], par)) { if (gt() - t0 > 2000000LL) { ok = false; break; } }
        lm[q] = ok ? gt() : -1;
        if (!ok) break;
        if (++sl == NS) { sl = 0; par ^= 1u; }
      }
    }
  }
#endif
  __syncthreads();
#if VMSP_SM90
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  const int gen = S.gen;
  if (tid == 0) {
    __threadfence();
    atomicAdd(p.ctr + 1, 1);
    const int target = (gen + 1) * G;
    const long long t0 = gt();
    int n = 0;
    while (ld_acq(p.ctr + 1) < target) {
      if ((++n & 31) == 0 && gt() - t0 > TMO) { atomicExch(p.ctr + 2, 2); break; }
      __nanosleep(32);
    }
    if (mk) mrow[23] = gt();
  }
  __syncthreads();
  if (c == 0) {   // reset for the next launch: every CTA has passed all of its waits
    if (tid < MAXJ) p.ctr[8 + tid] = 0;
    if (tid < MAXP) p.ss[((gen + 1) & 1) * MAXP + tid] = 0.f;
    if (tid == 0) { p.ctr[3] = 0; p.ctr[4] = 0; p.ctr[5] = 0; atomicAdd(p.ctr, 1); }
  }
  const float* ssc = p.ss + (gen & 1) * MAXP;
  for (int it = c; it < m * 4; it += G) {
    const int tk = it >> 2, col = (it & 3) * 1024 + (tid & 255) * 4;
    if (tid >= NT) break;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int pr = S.pos[tk * 8 + s];
      if (pr >= 0) {
        const float cf = __ldcg(p.tw + tk * 8 + s) * rsqrtf(__ldcg(ssc + pr) * (1.f / HD) + p.eps);
        const float4 y = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)pr * HD + col));
        a.x += cf * y.x; a.y += cf * y.y; a.z += cf * y.z; a.w += cf * y.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x, a.y), o1 = __floats2bfloat162_rn(a.z, a.w);
    uint2 pk; pk.x = *reinterpret_cast<uint32_t*>(&o0); pk.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = pk;
  }
  if (mk && tid == 0) mrow[24] = gt();
}
}  // namespace vmsp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void run(torch::Tensor rw, torch::Tensor rx, torch::Tensor part, torch::Tensor fpart, torch::Tensor bias, torch::Tensor lstart,
         torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, torch::Tensor pw, torch::Tensor pesc, torch::Tensor h,
         torch::Tensor y, torch::Tensor ssp, torch::Tensor nwp, int64_t tb,
         torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA, torch::Tensor gsB, torch::Tensor nw,
         torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr, torch::Tensor out,
         torch::Tensor marks, torch::Tensor lmarks, int64_t m, double eps, double peps, int64_t pdl) {
  TORCH_CHECK(m >= 1 && m <= 8, "vmsp: 1 <= m <= 8");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vmsp::vmsp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vmsp::SMEM_BYTES));
    attr = true;
  }
  vmsp::Params p;
  p.rw = rw.data_ptr<float>(); p.rx = rx.data_ptr<float>(); p.part = part.data_ptr<float>();
  p.fpart = fpart.numel() ? fpart.data_ptr<float>() : nullptr;
  p.bias = bias.data_ptr<float>(); p.lstart = lstart.data_ptr<int>();
  p.tw = tw.data_ptr<float>(); p.tid = tid.data_ptr<int>(); p.lid = lid.data_ptr<int>();
  p.pw = pw.data_ptr<uint8_t>(); p.pesc = reinterpret_cast<const uint32_t*>(pesc.data_ptr());
  p.h = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr()); p.y = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  p.ssp = ssp.data_ptr<float>(); p.nwp = reinterpret_cast<const __nv_bfloat16*>(nwp.data_ptr()); p.tb = (uint32_t)tb;
  p.wA = wA.data_ptr<uint8_t>(); p.wB = wB.data_ptr<uint8_t>(); p.gsA = gsA.data_ptr<float>(); p.gsB = gsB.data_ptr<float>();
  p.nw = nw.data_ptr<float>(); p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); p.ybuf = ybuf.data_ptr<float>();
  p.ss = ss.data_ptr<float>(); p.ctr = ctr.data_ptr<int>(); p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.lmarks = lmarks.numel() ? reinterpret_cast<long long*>(lmarks.data_ptr()) : nullptr;
  p.m = (int)m; p.eps = (float)eps; p.peps = (float)peps; p.mult = 0.1767766952966369f; p.rscale = 1.0f; p.deps = 1e-15f;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(vmsp::G); lc.blockDim = dim3(vmsp::NTL); lc.dynamicSmemBytes = vmsp::SMEM_BYTES;
  lc.stream = at::cuda::getCurrentCUDAStream(); lc.attrs = at; lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, vmsp::vmsp_kernel, p));
}

std::vector<int64_t> attrs() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, vmsp::vmsp_kernel));
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)vmsp::SMEM_BYTES, vmsp::G, vmsp::NS, vmsp::NTL};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
void run(torch::Tensor rw, torch::Tensor rx, torch::Tensor part, torch::Tensor fpart, torch::Tensor bias, torch::Tensor lstart,
         torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, torch::Tensor pw, torch::Tensor pesc, torch::Tensor h,
         torch::Tensor y, torch::Tensor ssp, torch::Tensor nwp, int64_t tb,
         torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA, torch::Tensor gsB, torch::Tensor nw,
         torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr, torch::Tensor out,
         torch::Tensor marks, torch::Tensor lmarks, int64_t m, double eps, double peps, int64_t pdl);
std::vector<int64_t> attrs();
'''


def flags(G, ns=3, math=0, obs=0, arch="sm_90a"):
    gen = "-gencode=arch=compute_90a,code=sm_90a" if arch == "sm_90a" else "-gencode=arch=compute_86,code=sm_86"
    return ["-O3", f"-DVMSP_G={G}", f"-DVMSP_NS={ns}", f"-DVMSP_MATH={math}", f"-DVMSP_OBS={obs}", "-Xptxas=-warn-spills", gen]


def build(G, ns=3, math=0, obs=0, arch="sm_90a", name="vmsp"):
    from torch.utils.cpp_extension import load_inline
    fl = flags(G, ns, math, obs, arch)
    tag = hashlib.sha1((CUDA_SRC + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=fl)
    return mod, time.time() - t0


def cta_prefix(c, G):
    rt0, rt1 = c * RT // G, (c + 1) * RT // G
    pt0, pt1 = c * PT // G, (c + 1) * PT // G
    nPT = pt1 - pt0
    return rt1 - rt0, (16 * nPT + 2) // 3, nPT


def router_tiles(w32):
    """FP32 router [256, 8192] -> tile-contiguous [256 tiles][8 rows][1024] (tile i: K slice i >> 5, rows 8 (i & 31) ..)"""
    return w32.float().view(32, 8, 8, 1024).permute(2, 0, 1, 3).contiguous()


def prepare(router_w, router_bias, pre_norm_w, vpk, dev, pooler_bytes=None, seed=7):
    """step-0 pack: router tiles (real), pooler stream (random bytes of the real size unless given), VMS expert units"""
    g = torch.Generator(device=dev).manual_seed(seed)
    pw = pooler_bytes if pooler_bytes is not None else torch.randint(0, 256, (PT * PTB,), generator=g, device=dev,
                                                                      dtype=torch.int32).to(torch.uint8)
    return {"rw": router_tiles(router_w).to(dev), "bias": router_bias.float().contiguous().to(dev), "pw": pw.contiguous(),
            "pesc": torch.full((PT * 8 * 32,), -1, dtype=torch.int32, device=dev),
            "nwp": (1.0 + pre_norm_w.float()).to(torch.bfloat16).contiguous().to(dev), "tb": 0, **vpk}


def scratch(dev, G):
    return {"part": torch.zeros((8 * 8 * 256,), dtype=torch.float32, device=dev),
            "y": torch.zeros((8, HD), dtype=torch.bfloat16, device=dev),
            "ssp": torch.zeros((PT * 8,), dtype=torch.float32, device=dev),
            "inter": torch.zeros((MAXP, HD), dtype=torch.bfloat16, device=dev),
            "ybuf": torch.zeros((MAXP, HD), dtype=torch.float32, device=dev),
            "ss": torch.zeros((2 * MAXP,), dtype=torch.float32, device=dev),
            "ctr": torch.zeros((8 + MAXJ,), dtype=torch.int32, device=dev),
            "marks": torch.zeros((G * NMK,), dtype=torch.int64, device=dev),
            "lmarks": torch.zeros((G * NLM,), dtype=torch.int64, device=dev),
            "none64": torch.empty((0,), dtype=torch.int64, device=dev),
            "nonef": torch.empty((0,), dtype=torch.float32, device=dev)}


def call(mod, pk, sc, h, rx, lstart, fpart=None, marks=False, lmarks=False, pdl=True, outs=None):
    """h bf16 [m, 8192], rx fp32 [m, 8192], lstart int32 [1]; returns (out bf16 [m, 4096], tw, tid, lid)"""
    m = h.shape[0]
    dev = h.device
    if outs is None:
        outs = (torch.empty((m, HD), dtype=torch.bfloat16, device=dev), torch.empty((m, 8), dtype=torch.float32, device=dev),
                torch.empty((m, 8), dtype=torch.int32, device=dev), torch.empty((m, 8), dtype=torch.int32, device=dev))
    out, tw, tid, lid = outs
    mod.run(pk["rw"], rx, sc["part"], sc["nonef"] if fpart is None else fpart, pk["bias"], lstart, tw, tid, lid,
            pk["pw"], pk["pesc"], h, sc["y"], sc["ssp"], pk["nwp"], pk["tb"], pk["wA"], pk["wB"], pk["gsA"], pk["gsB"],
            pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"], out, sc["marks"] if marks else sc["none64"],
            sc["lmarks"] if lmarks else sc["none64"], m, 1e-8, 1e-5, 1 if pdl else 0)
    return out, tw, tid, lid


def fake_partials(rx, router_w):
    """real FP32 router partials [8, m, 256] (K slices of 1024), computed on the host side"""
    m = rx.shape[0]
    w = router_w.float()
    return torch.stack([rx[:, s * 1024:(s + 1) * 1024].float() @ w[:, s * 1024:(s + 1) * 1024].T for s in range(8)]).contiguous()


def ref_routing(fpart, bias, ls):
    lg = fpart[0].clone()
    for s in range(1, 8):
        lg = lg + fpart[s]
    sc = torch.sigmoid(lg * MULT)
    bsc = sc + bias.float()
    top = bsc.topk(8, dim=-1)
    sel = sc.gather(1, top.indices)
    tw = sel / (sel.sum(-1, keepdim=True) + 1e-15)
    rel = (top.indices - ls) % 256
    lid = torch.where(rel < 32, rel, torch.full_like(rel, -1))
    return top.indices.to(torch.int32), tw, lid.to(torch.int32), bsc


def ref_xn(h, nwp, eps=1e-5):
    """step-0 stand-in pooler: y = bf16(h[:, :4096]); xn = bf16((y * rstd) * w)"""
    y = h[:, :HD].float()
    rstd = torch.rsqrt(y.pow(2).sum(-1, keepdim=True) / HD + eps)
    return ((y * rstd) * nwp.float()).to(torch.bfloat16)


def analyze(marks, lmarks, G, njobs, m):
    """marks [G, 32], lmarks [G, 96] (int64 lists) -> hop / rate / gate summary (us, from the earliest wait-return)"""
    import statistics
    rows = marks
    t0 = min(r[1] for r in rows if r[1] > 0)
    us = lambda v: (v - t0) / 1e3  # noqa: E731

    def q3(v):
        v = sorted(x for x in v if x is not None)
        return [round(v[0], 3), round(v[len(v) // 2], 3), round(v[-1], 3)] if v else None
    pre = [cta_prefix(c, G) for c in range(G)]
    res = {"njobs": njobs, "m": m}
    col = lambda i: [us(r[i]) for r in rows if r[i] > 0]  # noqa: E731
    names = {0: "entry", 1: "wait", 2: "q0_landed", 3: "router_consumed", 4: "pool_first_landed", 5: "pool_last_landed",
             6: "pool_last_consumed", 7: "router_release", 8: "pool_release_last", 9: "rdone_seen", 10: "routing_released",
             11: "routed_seen", 12: "plan_ready", 13: "pdone_seen_planner", 14: "rstd_ready", 15: "pdone_seen_compute",
             16: "x_stage_done", 17: "e0_consumed_start", 18: "e0_issued", 19: "lastA_landed", 20: "B_go", 21: "compute_end",
             22: "helper_end", 23: "barrier", 24: "end", 28: "B_first"}
    res["marks"] = {nm: q3(col(i)) for i, nm in names.items()}
    pool_end = max(col(8)) if col(8) else None
    res["pool_end"] = pool_end
    res["router_done"] = max(col(7)) if col(7) else None
    res["routing_done"] = max(col(10)) if col(10) else None
    # gate G2: first expert compute (x stage done) vs pooler end (last pooler-tile release anywhere)
    xs = [us(r[16]) for r in rows if r[16] > 0]
    res["g2_x_minus_pool_end"] = q3([x - pool_end for x in xs]) if xs and pool_end is not None else None
    # gate G3: plan ready vs this CTA's last pooler unit consumed
    res["g3_slack"] = q3([(r[6] - r[12]) / 1e3 for r in rows if r[6] > 0 and r[12] > 0])
    # G1 from the observer: last pooler unit landed -> first expert unit landed (per CTA)
    g1, land = [], []
    if lmarks is not None:
        for c, (r, lm) in enumerate(zip(rows, lmarks)):
            nR, nP, nPT = pre[c]
            nRP = nR + nP
            NEc = r[26] & 0xFFFF
            nA = (r[26] >> 16) & 0xFFFF
            if NEc > 0 and nRP < NLM and lm[nRP] > 0 and lm[nRP - 1] > 0:
                g1.append((lm[nRP] - lm[nRP - 1]) / 1e3)
            for q in range(min(NLM, nRP + NEc)):
                if lm[q] <= 0:
                    continue
                if q < nR:
                    kind, b = "R", RTB
                elif q < nRP:
                    k = q - nR
                    b, kind = 8 * min(3, 16 * nPT - 3 * k) * WU, "P"
                else:
                    kind, b = ("A" if q - nRP < nA else "B"), UB
                land.append((us(lm[q]), b, kind))
    res["g1_pool_to_expert_landing"] = q3(g1)
    rates = {}
    if land:
        land.sort()
        for kind in "RPAB":
            ts = sorted(t for t, _, k in land if k == kind)
            if len(ts) < 20:
                continue
            a, b = ts[len(ts) // 10], ts[(9 * len(ts)) // 10]
            if b - a < 0.3:
                continue
            nb = sum(bb for t, bb, _ in land if a <= t <= b)
            rates[kind] = {"TBs": round(nb / ((b - a) * 1e-6) / 1e12, 3), "win": [round(a, 2), round(b, 2)]}
        tt = [t for t, _, _ in land]
        rates["all"] = {"TBs": round(sum(b for _, b, _ in land) / ((max(tt) - min(tt)) * 1e-6) / 1e12, 3),
                        "win": [round(min(tt), 2), round(max(tt), 2)]}
    res["rates"] = rates
    ends = col(24)
    res["span_marks"] = round(max(ends), 3) if ends else None
    return res

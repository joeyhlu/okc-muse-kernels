"""vms_moe_lab.py: VM-small (VMS) one-launch MoE for T <= 16 tokens (lab kernel, notes/study/vm_small.md).

Our own implementation of the structure described in notes/study/competitor_scan_oct27.md s3 (ideas only):
  - one persistent launch, grid = SM count, 1 CTA/SM; routing done redundantly by every CTA (match_any, one warp);
  - unit = (job of <= 8 tokens, 16-row block) x full K = 36,864 B contiguous (32 KB codes + 4 KB scales) = one bulk copy;
  - 8 compute warps split K (512 each) and reduce through smem; a helper warp runs every unit's epilogue;
  - x fragments in registers for the whole job; static contiguous unit ranges per CTA, total-balanced over A + B;
  - the weight ring runs over the CTA's phase-A units and then its phase-B units without a break, so W2 weights stream
    while the warps wait on the per-job doneA counters (credited per job as soon as the CTA's last unit of the job ends);
  - dequant: E2M1 code bits stored at bf16 field positions (4 masked copies of the word, 7 ALU + 2 IMAD per 8 weights),
    block scale -> bf16(scale * 2^64) with one PRMT + one IMAD; fp32 global scale * 2^62 in the epilogue (exact products);
  - W13 unit rows = 8 gate rows then their 8 up rows, so SwiGLU forms from one lane's accumulators;
  - post-expert norm + weighted sum after a software grid barrier in the same launch (FP32 y rows, sum of squares
    by atomics, double-buffered by a generation counter); PDL: griddepcontrol.wait at entry, launch_dependents before
    the grid barrier (all CTAs resident by then, so the dependent cannot take an SM a waiting CTA needs).
sm_86 build (local RTX 3070 correctness): cp.async by all compute threads + cp.async.mbarrier.arrive.noinc instead of
the bulk copy, NS = 2 slots, no griddepcontrol.

API: build(G, ns, arch) -> module; prepack(w) -> dict; call(mod, pk, sc, x, ids, tw, out=None, marks=None, pdl=False).
"""
from __future__ import annotations

import hashlib
import time

import torch

HD = 4096
ID = 4096
NE = 32
UB = 36864
MAXT = 16
MAXP = MAXT * 8
MAXJ = 64

CUDA_SRC = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef VMS_NS
#define VMS_NS 3
#endif
#ifndef VMS_G
#define VMS_G 132
#endif

namespace vms {
constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 128, MAXJ = 64;
constexpr int UB = 36864, UC = 32768, RBA = 512, RBB = 256;
constexpr int NW = 8, NT = NW * 32, NTL = NT + 32, NRB = 4, NS = VMS_NS, G = VMS_G;
constexpr int WAIT_MAX = 1 << 22;

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* tw;
  const unsigned char* wA; const unsigned char* wB;
  const float* gsA; const float* gsB; const float* nw;
  __nv_bfloat16* inter; float* ybuf; float* ss; int* ctr;
  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;
  int fl;   // 1 probe: weights from 8 L2-resident units, 2 null consumer, 4 L1 prefetch of the next A job's x, 8 same for B (inter)
};

struct __align__(16) Smem {
  uint64_t full[NS];
  uint64_t pfull[NRB];
  uint64_t pempty[NRB];
  int njobs, gen;
  int base[4][NE];
  int jE[MAXJ], jS[MAXJ], jC[MAXJ];
  int ent[MAXP], pos[MAXP];
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
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t par) {
#if __CUDA_ARCH__ >= 900
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
__device__ __forceinline__ void cta_sync_compute() { asm volatile("bar.sync 1, %0;\n" :: "n"(NT) : "memory"); }

// unit q of this CTA's sequence (phase-A units a0.., then phase-B units b0..) -> global source
__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {
  if (p.fl & 1) return p.wA + (size_t)(blockIdx.x & 7) * UB;
  if (q < nA) { const int v = a0 + q; return p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB; }
  const int v = b0 + (q - nA); return p.wB + ((size_t)S.jE[v >> 8] * RBB + (v & 255)) * UB;
}

// sm_90: thread 0 only (bulk copy + expect_tx); sm_86: every compute thread (cp.async + arrive.noinc, count NT)
__device__ __forceinline__ void issue(const unsigned char* src, unsigned char* dst, uint64_t* bar, int tid) {
#if __CUDA_ARCH__ >= 900
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(UB) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(su32(dst)), "l"(src), "r"(UB), "r"(su32(bar)), "l"(pol) : "memory");
#else
  for (int i = tid; i < UB / 16; i += NT)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst + i * 16)), "l"(src + i * 16) : "memory");
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(su32(bar)) : "memory");
#endif
}

// e4m3 byte b (0..3) of w -> bf16x2(scale * 2^64) in both halves; exact for bytes 0x08..0x7E (prepare checks)
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

__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  Smem& S = *reinterpret_cast<Smem*>(smem);
  unsigned char* ring = smem + RING_OFF;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x;
  const bool mk = p.marks != nullptr;
  long long* mrow = p.marks + (size_t)c * 16;
  if (tid == 0) {
    if (mk) mrow[0] = gt();
#if __CUDA_ARCH__ >= 900
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);
#else
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], NT);
#endif
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], 1); mbar_init(&S.pempty[i], 1); }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  // ---- routing (warp 0, redundant per CTA): pairs sorted by expert, jobs of <= 8 pairs
  if (warp == 0) {
    const int m8 = p.m * 8;
    int idv[4]; unsigned mm[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; }
    if (tid == 0) S.gen = ld_rlx(p.ctr);
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
      const int a = __shfl_up_sync(0xffffffffu, inc, d), b = __shfl_up_sync(0xffffffffu, jinc, d);
      if (lane >= d) { inc += a; jinc += b; }
    }
    const int off = inc - tot, joff = jinc - nj;
    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + 8 * k; S.jC[joff + k] = min(8, tot - 8 * k); }
    int run = off;
#pragma unroll
    for (int r = 0; r < 4; ++r) { S.base[r][lane] = run; run += cr[r]; }
    if (lane == 31) S.njobs = jinc;
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
  }
  __syncthreads();
  const int njobs = S.njobs;
  const int NA = njobs * RBA, NN = njobs * (RBA + RBB);
  const int a0 = c * NA / G, a1 = (c + 1) * NA / G;
  const int t0 = c * NN / G, t1 = (c + 1) * NN / G;
  const int b0 = t0 - a0;
  const int nA = a1 - a0, N = t1 - t0;
  if (mk && tid == 0) { mrow[1] = gt(); mrow[10] = N; }
  if (warp < NW) {
    // ================= compute warps
    for (int q = 0; q < NS && q < N; ++q) {
#if __CUDA_ARCH__ >= 900
      if (tid == 0)
#endif
      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);
    }
    const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);   // opaque 16: the scale build's multiply stays an IMAD
    const int g = lane >> 2, t = lane & 3;
    uint4 xr[16];
    int curj = -1;
    int sl = 0; uint32_t par = 0;
    for (int q = 0; q < N; ++q) {
      const bool PB = q >= nA;
      const int v = PB ? b0 + (q - nA) : a0 + q;
      const int j = PB ? (v >> 8) + 4096 : (v >> 9);
      if (j != curj) {
        curj = j;
        const int jj = PB ? j - 4096 : j;
        const int cnt = S.jC[jj], st = S.jS[jj];
        if (PB) {
          if (mk && tid == 0 && mrow[3] == 0) mrow[3] = gt();
          if (lane == 0) {
            int n = 0;
            while (ld_rlx(p.ctr + 8 + jj) < RBA) {
              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
              __nanosleep(64);
            }
          }
          __syncwarp();
          (void)ld_acq(p.ctr + 8 + jj);
          if (mk && tid == 0 && mrow[4] == 0) mrow[4] = gt();
        }
        if (g < cnt) {
          const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID)
                                : reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + g] >> 3) * HD);
          src += warp * 64 + 2 * t;
#pragma unroll
          for (int cc = 0; cc < 8; ++cc) {
            xr[2 * cc] = PB ? ((p.fl & 8) ? __ldca(src + cc * 8) : __ldcg(src + cc * 8)) : __ldg(src + cc * 8);
            xr[2 * cc + 1] = PB ? ((p.fl & 8) ? __ldca(src + cc * 8 + 1) : __ldcg(src + cc * 8 + 1)) : __ldg(src + cc * 8 + 1);
          }
        } else {
#pragma unroll
          for (int cc = 0; cc < 16; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
      if ((p.fl & 12) && q + 1 < N) {   // the next unit starts a new job: pull its x / inter fragments into L1 now
        const bool PB1 = q + 1 >= nA;
        const int v1 = PB1 ? b0 + (q + 1 - nA) : a0 + q + 1;
        const int j1 = PB1 ? (v1 >> 8) + 4096 : (v1 >> 9);
        if (j1 != curj) {
          const int jj1 = PB1 ? j1 - 4096 : j1;
          bool okp = g < S.jC[jj1];
          if (PB1) okp = okp && (p.fl & 8) && ld_acq(p.ctr + 8 + jj1) >= RBA;
          else okp = okp && (p.fl & 4);
          if (okp) {
            const unsigned char* row = PB1 ? reinterpret_cast<const unsigned char*>(p.inter + (size_t)(S.jS[jj1] + g) * ID)
                                           : reinterpret_cast<const unsigned char*>(p.x + (size_t)(S.ent[S.jS[jj1] + g] >> 3) * HD);
            row += warp * 1024 + t * 256;
            asm volatile("prefetch.global.L1 [%0];\n" :: "l"(row));
            asm volatile("prefetch.global.L1 [%0];\n" :: "l"(row + 128));
          }
        }
      }
      mbar_wait(&S.full[sl], par);
      if (mk && tid == 0 && q == 0) mrow[2] = gt();
      const unsigned char* slot = ring + sl * UB;
      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16);
      float acc[4][4];
#pragma unroll
      for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
      if (!(p.fl & 2))
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
          const uint32_t r1 = (qw * 16u) & M;
          const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;
          const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
          mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);
        }
      }
      float4 o;
      o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
      o.y = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
      o.z = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
      o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
      const int bq = q & (NRB - 1);
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
      S.red[bq][warp][lane] = o;
      cta_sync_compute();
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (tid == 0)
#endif
        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);
      }
      if (tid == 0) mbar_arrive(&S.pfull[bq]);
      if (++sl == NS) { sl = 0; par ^= 1u; }
    }
    if (mk && tid == 0) mrow[5] = gt();
  } else {
    // ================= helper warp: per-unit K reduction + epilogue
    const int g = lane >> 2, t = lane & 3;
    float* ssc = p.ss + (S.gen & 1) * MAXP;
    for (int q = 0; q < N; ++q) {
      const int bq = q & (NRB - 1);
      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
      float4 sm = S.red[bq][0][lane];
#pragma unroll
      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pempty[bq]);
      if (q < nA) {
        const int v = a0 + q, j = v >> 9, rb = v & 511;
        const int cnt = S.jC[j], st = S.jS[j];
        const float gs = __ldg(p.gsA + S.jE[j]);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 2 * t + h;
          if (tk < cnt) {
            const float ga = (h ? sm.y : sm.x) * gs, up = (h ? sm.w : sm.z) * gs;
            const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;
            p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);
          }
        }
        if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units
          __threadfence();
          __syncwarp();
          if (lane == 0) atomicAdd(p.ctr + 8 + j, min(a1, (j + 1) * RBA) - max(a0, j * RBA));
          if (mk && lane == 0 && q == nA - 1) mrow[6] = gt();
        }
      } else {
        const int v = b0 + (q - nA), j = v >> 8, rb = v & 255;
        const int cnt = S.jC[j], st = S.jS[j];
        const float gs = __ldg(p.gsB + S.jE[j]);
        const int n0 = rb * 16 + g;
        const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 2 * t + h;
          const float y0 = (h ? sm.y : sm.x) * w0, y1 = (h ? sm.w : sm.z) * w1;
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
    if (mk && lane == 0) mrow[7] = gt();
  }
  __syncthreads();
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  const int gen = S.gen;
  if (tid == 0) {
    __threadfence();
    atomicAdd(p.ctr + 1, 1);
    const int target = (gen + 1) * G;
    int n = 0;
    while (ld_acq(p.ctr + 1) < target) {
      if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 2); break; }
      __nanosleep(32);
    }
    if (mk) mrow[8] = gt();
  }
  __syncthreads();
  if (c == 0) {   // reset for the next launch: every CTA has passed its doneA waits and read gen
    if (tid < MAXJ) p.ctr[8 + tid] = 0;
    if (tid < MAXP) p.ss[((gen + 1) & 1) * MAXP + tid] = 0.f;
    if (tid == 0) atomicAdd(p.ctr, 1);
  }
  // ---- final pass: out[t][col] = sum_s w * rsqrt(ss / HD + eps) * y ; item = (token, quarter of the columns)
  const float* ssc = p.ss + (gen & 1) * MAXP;
  for (int it = c; it < p.m * 4; it += G) {
    const int tk = it >> 2, col = (it & 3) * 1024 + (tid & 255) * 4;
    if (tid >= NT) break;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int pr = S.pos[tk * 8 + s];
      if (pr >= 0) {
        const float cf = __ldg(p.tw + tk * 8 + s) * rsqrtf(__ldcg(ssc + pr) * (1.f / HD) + p.eps);
        const float4 y = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)pr * HD + col));
        a.x += cf * y.x; a.y += cf * y.y; a.z += cf * y.z; a.w += cf * y.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x, a.y), o1 = __floats2bfloat162_rn(a.z, a.w);
    uint2 pk; pk.x = *reinterpret_cast<uint32_t*>(&o0); pk.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = pk;
  }
  if (mk && tid == 0) mrow[9] = gt();
}
}  // namespace vms

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo, int64_t fl) {
  TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vms::vms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vms::SMEM_BYTES));
    attr = true;
  }
  vms::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.ids = ids.data_ptr<int>(); p.tw = tw.data_ptr<float>();
  p.wA = wA.data_ptr<uint8_t>(); p.wB = wB.data_ptr<uint8_t>(); p.gsA = gsA.data_ptr<float>(); p.gsB = gsB.data_ptr<float>();
  p.nw = nw.data_ptr<float>(); p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); p.ybuf = ybuf.data_ptr<float>();
  p.ss = ss.data_ptr<float>(); p.ctr = ctr.data_ptr<int>(); p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo; p.fl = (int)fl;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(vms::G); lc.blockDim = dim3(vms::NTL); lc.dynamicSmemBytes = vms::SMEM_BYTES;
  lc.stream = at::cuda::getCurrentCUDAStream(); lc.attrs = at; lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, vms::vms_kernel, p));
}

std::vector<int64_t> attrs() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, vms::vms_kernel));
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)vms::SMEM_BYTES, vms::G, vms::NS};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo, int64_t fl);
std::vector<int64_t> attrs();
'''


def build(G, ns=3, arch="sm_90a", name="vms_lab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    gen = f"-gencode=arch=compute_{arch[3:].rstrip('a')}{'a' if arch.endswith('a') else ''},code={arch}"
    flags = ["-O3", f"-DVMS_G={G}", f"-DVMS_NS={ns}", "-Xptxas=-warn-spills", gen]
    tag = hashlib.sha1((CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags, verbose=verbose)
    return mod, time.time() - t0


# ---------------------------------------------------------------- weight prepack (bit/byte permutation only)
# word bit position of (reg r, half h, code bit b) where b = 0 m, 1 e0, 2 e1, 3 sign; decode in the kernel:
# r0 = q & M, r1 = (q << 4) & M, r2 = rotl(q, 10) & M, r3 = ((q << 6) & 0x80C080C0) | (rotl(q, 14) & 0x01000100)
POS = (((6, 7, 8, 15), (22, 23, 24, 31)),
       ((2, 3, 4, 11), (18, 19, 20, 27)),
       ((28, 29, 30, 5), (12, 13, 14, 21)),
       ((0, 1, 26, 9), (16, 17, 10, 25)))


def _codes(w):
    """packed uint8 [n, k/2] -> codes uint8 [n, k] (element 2i = low nibble)"""
    return torch.stack((w & 15, w >> 4), dim=-1).reshape(w.shape[0], -1)


def _pack_unit_words(cu):
    """cu: codes [nu, 16 rows, 4096 k] (unit-local rows: 0-7 = 'g', 8-15 = 'g + 8') -> int32 [nu, 8192] unit words
    K = 512 w + 64 c + 16 t + 4 s + 2 qq + h; reg r = 2 qq + hi8 (r0 row g k+0/1, r1 row g+8, r2 row g k+2/3, r3 row g+8)"""
    nu = cu.shape[0]
    x = cu.view(nu, 2, 8, 8, 8, 4, 4, 2, 2)                     # nu, hi8, g, w, c, t, s, qq, h
    x = x.permute(0, 3, 4, 2, 5, 6, 7, 1, 8).contiguous()        # nu, w, c, g, t, s, qq, hi8, h
    x = x.view(nu, 8, 8, 8, 4, 4, 4, 2).to(torch.int64)         # ..., r (= 2 qq + hi8), h
    word = torch.zeros(x.shape[:6], dtype=torch.int64, device=cu.device)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << POS[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return word.view(nu, 8192)


def _pack_unit_scales(su):
    """su: scale bytes [nu, 16 rows, 256 blocks] -> uint8 [nu, 4096]: [w][lane = 4 g + t][c][hi8], block = 32 w + 4 c + t"""
    nu = su.shape[0]
    x = su.view(nu, 2, 8, 8, 8, 4)                               # nu, hi8, g, w, c, t
    return x.permute(0, 3, 2, 5, 4, 1).contiguous().view(nu, 4096)


def prepack_expert(w, sf, gate_up):
    """one expert: w uint8 [n, 2048] packed codes, sf e4m3 [n, 256] -> uint8 [n / 16, 36864] units"""
    n = w.shape[0]
    rows = torch.arange(n, device=w.device)
    if gate_up:                                                  # unit rb: gate rows 8 rb + i, then up rows ID + 8 rb + i
        rb, i = rows // 16, rows % 16
        rows = torch.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    cu = _codes(w[rows]).view(n // 16, 16, 4096)
    sb = sf.view(torch.uint8)[rows].view(n // 16, 16, 256)
    words = _pack_unit_words(cu).view(torch.uint8).view(n // 16, 32768)
    scl = _pack_unit_scales(sb)
    return torch.cat((words, scl), dim=1).contiguous()


def scales_ok(sf):
    b = sf.view(torch.uint8)
    return bool(((b >= 0x08) & (b <= 0x7E)).all())


def prepack(w, log=print):
    """w: dict w13 [E, 8192, 2048] u8, w13_sf [E, 8192, 256] e4m3, w13_gs [E], w2 [E, 4096, 2048], w2_sf, w2_gs,
    post_norm_w [4096] bf16 -> dict of device tensors (None if the scale bytes do not fit the PRMT decode)"""
    t0 = time.time()
    if not (scales_ok(w["w13_sf"]) and scales_ok(w["w2_sf"])):
        log("[vms] scale bytes outside 0x08..0x7E: VMS route off")
        return None
    E = w["w13"].shape[0]
    dev = w["w13"].device
    wA = torch.empty((E, 512, UB), dtype=torch.uint8, device=dev)
    wB = torch.empty((E, 256, UB), dtype=torch.uint8, device=dev)
    for e in range(E):
        wA[e] = prepack_expert(w["w13"][e], w["w13_sf"][e], True)
        wB[e] = prepack_expert(w["w2"][e], w["w2_sf"][e], False)
    two62 = float(2 ** 62)
    pk = {"wA": wA, "wB": wB, "gsA": (w["w13_gs"].float().reshape(E) * two62).contiguous(),
          "gsB": (w["w2_gs"].float().reshape(E) * two62).contiguous(),
          "nw": (1.0 + w["post_norm_w"].float()).contiguous()}
    torch.cuda.synchronize()
    log(f"[vms] prepack {E} experts in {time.time() - t0:.1f} s")
    return pk


def scratch(dev):
    return {"inter": torch.zeros((MAXP, ID), dtype=torch.bfloat16, device=dev),
            "ybuf": torch.zeros((MAXP, HD), dtype=torch.float32, device=dev),
            "ss": torch.zeros((2 * MAXP,), dtype=torch.float32, device=dev),
            "ctr": torch.zeros((8 + MAXJ,), dtype=torch.int32, device=dev),
            "nomarks": torch.empty((0,), dtype=torch.int64, device=dev)}


def call(mod, pk, sc, x, ids, tw, out=None, marks=None, pdl=False, eps=1e-8, lo=0, fl=0):
    m = x.shape[0]
    if out is None:
        out = torch.empty((m, HD), dtype=torch.bfloat16, device=x.device)
    mod.run(x, ids, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"],
            out, sc["nomarks"] if marks is None else marks, m, eps, 1 if pdl else 0, lo, fl)
    return out


# ---------------------------------------------------------------- references
def dequant(w, sf, gs):
    c = _codes(w).long()
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=w.device)
    v = grid[c & 7] * torch.where(c >= 8, -1.0, 1.0)
    return (v.view(w.shape[0], -1, 16) * sf.float().unsqueeze(-1)).view(w.shape[0], -1) * float(gs)


def ref_fp32(x, ids, tw, w, eps=1e-8):
    """FP32 reference of the layer's MoE (K2A's _k2a_ref_fp32 semantics; ids local or -1)"""
    xf = x.float()
    out = torch.zeros(xf.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not 0 <= e < NE:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = xf[tok] @ dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :ID]) * h[:, ID:]
        y = (inter @ dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
        out.index_add_(0, tok, y * tw[tok, slot].float().unsqueeze(1))
    return out.to(torch.bfloat16)


def rel_err(a, b):
    a, b = a.float(), b.float()
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row

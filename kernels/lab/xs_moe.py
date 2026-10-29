"""xs_moe.py: XS (XC-small), a one-launch MoE for T <= 16 without the cross-CTA W13 -> W2 dependency (notes/study/xc_moe.md).

Base: VMS (kernels/lab/vms_moe_lab.py; routing, phase-A units, x in registers, mma.sync consumer, helper warp, grid barrier
and the VMS weight layout wA are VMS's code). What changes:
  - Each CTA owns a contiguous range of phase-A units (16 W13 rows = 8 intermediate dims each); its phase B is W2 rows x
    ONLY ITS OWN dims (all 4096 output rows, K = its dims). h never leaves the CTA: the helper's SwiGLU epilogue writes it
    to shared memory, the compute warps read the B fragments from there. No doneA counters, no cross-CTA wait.
  - W2 in a new k16-granular layout wB [E][256 row groups][256 k16 steps][144 B] (32 code words + 16 scale bytes): a
    CTA's K range is contiguous per 16-row group; one scale block per word. Dims outside the CTA's range inside a boundary
    k16 step multiply zero h (h smem is zeroed at entry).
  - Phase-B tiles of R rows (R = 512 / 256 / 128 by K depth, <= 36 KB per ring slot), rows split over the 8 compute warps,
    each warp accumulates its rows over the whole K and red.adds its 16-row partials into y (FP32, x nw x gs2).
  - The helper warp issues every ring refill (VMS_DW mode) and does only phase-A epilogues.
  - Final pass after the grid barrier: one item per token, the RMS norm computed from the completed y, y re-zeroed.
Numerics as VMS: FP32 accumulation, h bf16, y FP32 x (1 + post_norm_w), out = sum_s w_s rsqrt(ss_s / 4096 + eps) y_s.
Local sm_86 path (cp.async ring) for correctness tests on any GPU >= sm_80: tools/xs_local_test.py.
"""
import hashlib
import time

import torch

import vms_moe_lab as VMS

HD = 4096
ID = 4096
NE = 32
UB = 36864
MAXP = 128
BS = 144

CUDA_SRC = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef XS_NS
#define XS_NS 3
#endif
#ifndef XS_G
#define XS_G 132
#endif
#ifndef XS_L2
#define XS_L2 0   // 1: UNP L2 bit layout (tools/unp_patches.py; the layer's VMS_POS), 0: the lab VMS layout (VMS.POS)
#endif
#if XS_L2
__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {   // c, opaque to NVVM: the 3-way gather stays one IMAD
  uint32_t r;
  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));
  return r;
}
#define XS_DEC123(qw)   const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;   const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;   const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
#else
#define XS_DEC123(qw)   const uint32_t r1 = (qw * 16u) & M;   const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;   const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
#endif

namespace xs {
constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 128, MAXJ = 64;
constexpr int UB = 36864, UC = 32768, RBA = 512;
constexpr int NW = 8, NCH = 8, NT = NW * 32, NTL = NT + 32, NRB = 4, NS = XS_NS, G = XS_G;
constexpr int WAIT_MAX = 1 << 22;
constexpr int HSD = 1024;   // max own dims (incl. boundary padding) per job segment
constexpr int BS = 144;     // XS W2 layout: bytes per (16-row group, k16 step)
constexpr int MAXB = 4;     // phase-B sub-segments (<= 32 k16 steps each)

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* tw;
  const unsigned char* wA; const unsigned char* wB;
  const float* gsA; const float* gsB; const float* nw;
  float* ybuf; int* ctr; __nv_bfloat16* out; long long* marks; int m; float eps; int lo;
};
struct Seg { int j, r0, r1, kk0, qlast; };              // job, A units [r0, r1) of the job, first k16 step, last A unit q
struct BSeg { int s, kk0, nk, rgw, nb, q0; };           // segment, k16 steps [kk0, kk0 + nk), row groups / warp, slots, first q

struct __align__(16) Smem {
  uint64_t full[NS];
  uint64_t pfull[NRB];
  uint64_t pempty[NRB];
  uint64_t hready[2];
  int njobs, gen, nseg, nbs, nA, nU;
  Seg seg[2];
  BSeg bs[MAXB];
  int base[4][NE];
  int jE[MAXJ], jS[MAXJ], jC[MAXJ];
  int ent[MAXP], pos[MAXP];
  float4 red[NRB][NW][32];
  __nv_bfloat16 hS[2][8][HSD + 8];   // + 8: token rows 4 banks apart
  float fred[NW];
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
__device__ __forceinline__ void red_add(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ void bar_fin() { asm volatile("bar.sync 2, %0;\n" :: "n"(NT) : "memory"); }

// ring refill of unit q by the helper warp (all 32 lanes call it). sm_90: bulk copies + one expect_tx (lane 0);
// sm_86: cp.async 16-B pieces + arrive.noinc per lane (full barrier count 32).
__device__ __forceinline__ void issue(const Params& p, const Smem& S, int q, int a0, unsigned char* dst, uint64_t* bar, int lane) {
#if __CUDA_ARCH__ >= 900
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
#endif
  if (q < S.nA) {
    const int v = a0 + q;
    const unsigned char* src = p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB;
#if __CUDA_ARCH__ >= 900
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(UB) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                   :: "r"(su32(dst)), "l"(src), "r"(UB), "r"(su32(bar)), "l"(pol) : "memory");
    }
#else
    for (int i = lane; i < UB / 16; i += 32)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst + i * 16)), "l"(src + i * 16) : "memory");
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(su32(bar)) : "memory");
#endif
    return;
  }
}

// e4m3 byte (selected by sel) -> bf16x2(scale * 2^64) in both halves; exact for bytes 0x08..0x7E (prepare checks)
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

// one warp's phase-B sub-segment: row groups [32 warp, 32 warp + 32) of W2, k16 steps [B.kk0, B.kk0 + B.nk), D in flight
__device__ __forceinline__ void cp16(unsigned char* dst, const unsigned char* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst)), "l"(src) : "memory");
}
template <int D>
__device__ __forceinline__ void phase_b(const Params& p, const Smem& S, const BSeg& B, const Seg& sgm, const uint32_t* hrow,
                                        unsigned char* wring, int warp, int lane, uint32_t k16, bool hw) {
  const int g = lane >> 2, t = lane & 3, nk = B.nk, bytes = nk * BS, np = bytes / 16;
  const int e = S.jE[sgm.j], cnt = S.jC[sgm.j], st = S.jS[sgm.j];
  const unsigned char* src0 = p.wB + (((size_t)e * 256 + warp * 32) * 256 + B.kk0) * BS;
  const float gs = __ldg(p.gsB + e);
#pragma unroll
  for (int i = 0; i < D - 1; ++i) {
    for (int k = lane; k < np; k += 32) cp16(wring + i * bytes + k * 16, src0 + (size_t)i * 256 * BS + k * 16);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  if (hw) mbar_wait(const_cast<uint64_t*>(&S.hready[B.s]), 0u);
#pragma unroll 1
  for (int r = 0; r < 32; ++r) {
    if (r + D - 1 < 32) {
      const int rn = r + D - 1;
      for (int k = lane; k < np; k += 32) cp16(wring + (rn % D) * bytes + k * 16, src0 + (size_t)rn * 256 * BS + k * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group %0;\n" :: "n"(D - 1) : "memory");
    __syncwarp();
    const unsigned char* gp = wring + (r % D) * bytes;
    float acc[4][4];
#pragma unroll
    for (int c = 0; c < 4; ++c) { acc[c][0] = 0.f; acc[c][1] = 0.f; acc[c][2] = 0.f; acc[c][3] = 0.f; }
#pragma unroll 1
    for (int s0 = 0; s0 < nk; s0 += 4) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int s = s0 + c;
        if (s < nk) {
          const uint32_t b0 = hrow[8 * s + t], b1 = hrow[8 * s + 4 + t];
          const uint32_t qw = *reinterpret_cast<const uint32_t*>(gp + s * BS + lane * 4);
          const uint32_t sw = *reinterpret_cast<const uint16_t*>(gp + s * BS + 128 + 2 * g);
          const uint32_t sg = scd(sw, 0x4040u, k16), sg8 = scd(sw, 0x4141u, k16);
          const uint32_t M = 0x81C081C0u;
          const uint32_t r0 = qw & M;
          XS_DEC123(qw)
          mma(acc[c], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), b0, b1);
        }
      }
    }
    __syncwarp();   // this slot's codes are read: a later copy may overwrite it
    const int row0 = (warp * 32 + r) * 16 + g;
    const float w0 = __ldg(p.nw + row0) * gs, w1 = __ldg(p.nw + row0 + 8) * gs;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * t + h;
      if (tk < cnt) {
        float* yr = p.ybuf + (size_t)(st + tk) * HD + row0;
        red_add(yr, (acc[0][h] + acc[1][h] + acc[2][h] + acc[3][h]) * w0);
        red_add(yr + 8, (acc[0][2 + h] + acc[1][2 + h] + acc[2][2 + h] + acc[3][2 + h]) * w1);
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncwarp();
}

__global__ void __launch_bounds__(NTL, 1) xs_kernel(Params p) {
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
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 32);
#endif
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], NW); mbar_init(&S.pempty[i], 1); }
    mbar_init(&S.hready[0], 1); mbar_init(&S.hready[1], 1);
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
  {   // h smem = 0: dims of boundary k16 steps that this CTA does not own must multiply zero
    uint4* hz = reinterpret_cast<uint4*>(&S.hS[0][0][0]);
    for (int i = tid; i < (int)(sizeof(S.hS) / 16); i += NTL) hz[i] = make_uint4(0u, 0u, 0u, 0u);
  }
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  // ---- routing (warp 0, redundant per CTA; VMS's): pairs sorted by expert, jobs of <= 8 pairs
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
    int inc = tot, nj = (tot + 7) / 8, jinc = nj;
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
    __syncwarp();
    // ---- plan (lane 0): equal A-unit ranges; phase B = W2 rows x this CTA's own dims, per job segment
    if (lane == 0) {
      const int NA = S.njobs * RBA, a0 = c * NA / G, a1 = (c + 1) * NA / G;
      int ns = 0, nb = 0, q = a1 - a0;
      for (int v = a0; v < a1 && ns < 2;) {
        const int j = v >> 9, ve = min(a1, (j + 1) << 9);
        Seg s; s.j = j; s.r0 = v & 511; s.r1 = s.r0 + (ve - v); s.kk0 = s.r0 >> 1; s.qlast = ve - 1 - a0;
        const int kk1 = (s.r1 + 1) >> 1;
        if ((kk1 - s.kk0) * 16 > HSD) atomicExch(p.ctr + 2, 3);   // too many dims for one CTA (host routes such loads elsewhere)
        for (int k0 = s.kk0; k0 < kk1 && nb < MAXB; k0 += 32) {
          BSeg B; B.s = ns; B.kk0 = k0; B.nk = min(32, kk1 - k0);
          int R = 512;
          while (R > 128 && (R / 16) * B.nk * BS > UB) R >>= 1;
          B.rgw = R / 128; B.nb = HD / R; B.q0 = q;
          S.bs[nb++] = B;
        }
        S.seg[ns++] = s; v = ve;
      }
      S.nseg = ns; S.nbs = nb; S.nA = a1 - a0; S.nU = q;
    }
  }
  __syncthreads();
  const int NA = S.njobs * RBA, a0 = c * NA / G;
  const int nA = S.nA, nU = S.nU;
  if (mk && tid == 0) { mrow[1] = gt(); mrow[10] = nU; mrow[11] = nA; }
  const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);   // opaque 16: the scale build's multiply stays an IMAD
  if (warp < NW) {
    // ================= compute warps
    const int g = lane >> 2, t = lane & 3;
    uint4 xr[2 * NCH];
    int curj = -1, sl = 0, hwait = -1;
    uint32_t par = 0;
    for (int q = 0; q < nU; ++q) {
      const unsigned char* slot = ring + sl * UB;
      if (q < nA) {
        // ---- phase-A unit (VMS's consumer)
        const int v = a0 + q, j = v >> 9;
        if (j != curj) {
          curj = j;
          const int cnt = S.jC[j], st = S.jS[j];
          if (g < cnt) {
            const uint4* src = reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + g] >> 3) * HD) + warp * (8 * NCH) + 2 * t;
#pragma unroll
            for (int cc = 0; cc < NCH; ++cc) { xr[2 * cc] = __ldg(src + cc * 8); xr[2 * cc + 1] = __ldg(src + cc * 8 + 1); }
          } else {
#pragma unroll
            for (int cc = 0; cc < 2 * NCH; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);
          }
        }
        mbar_wait(&S.full[sl], par);
        if (mk && tid == 0 && q == 0) mrow[2] = gt();
        const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16);
        float acc[4][4];
#pragma unroll
        for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
#pragma unroll
        for (int cc = 0; cc < NCH; ++cc) {
          const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + cc) * 512 + lane * 16);
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
            XS_DEC123(qw)
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
        __syncwarp();
        if (lane == 0) mbar_arrive(&S.pfull[bq]);
      }
      if (++sl == NS) { sl = 0; par ^= 1u; }
    }
    // ---- phase B: each warp streams its own 32 W2 row groups (512 output rows) x this CTA's dims through a private
    // cp.async pipeline in the (now idle) ring, prefetching before h is ready; red.add of 16-row partials into y
    asm volatile("bar.sync 1, %0;\n" :: "n"(NT) : "memory");   // every warp is done with the ring's A units
    unsigned char* wring = ring + warp * (NS * UB / NW);
    for (int b = 0; b < S.nbs; ++b) {
      const BSeg B = S.bs[b];
      const Seg sgm = S.seg[B.s];
      const bool hw = (B.s != hwait);
      const uint32_t* hrow = reinterpret_cast<const uint32_t*>(&S.hS[B.s][g][0]) + (B.kk0 - sgm.kk0) * 8;
      constexpr int WB = NS * UB / NW;   // private pipeline bytes per warp
      const int rb = B.nk * BS;
      if (6 * rb <= WB) phase_b<6>(p, S, B, sgm, hrow, wring, warp, lane, k16, hw);
      else if (3 * rb <= WB) phase_b<3>(p, S, B, sgm, hrow, wring, warp, lane, k16, hw);
      else if (2 * rb <= WB) phase_b<2>(p, S, B, sgm, hrow, wring, warp, lane, k16, hw);
      else phase_b<1>(p, S, B, sgm, hrow, wring, warp, lane, k16, hw);
      if (hw) hwait = B.s;
    }
    if (mk && tid == 0) mrow[5] = gt();
  } else {
    // ================= helper warp: every ring refill + phase-A K reduction and SwiGLU epilogue into h smem
    const int g = lane >> 2, t = lane & 3;
    for (int q = 0; q < NS && q < nU; ++q) issue(p, S, q, a0, ring + q * UB, &S.full[q], lane);
    int hsl = 0, sgi = 0;
    for (int q = 0; q < nU; ++q) {
      const int bq = q & (NRB - 1);
      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
      if (q + NS < nU) issue(p, S, q + NS, a0, ring + hsl * UB, &S.full[hsl], lane);
      if (++hsl == NS) hsl = 0;
      if (q < nA) {
        float4 sm = S.red[bq][0][lane];
#pragma unroll
        for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
        __syncwarp();
        if (lane == 0) mbar_arrive(&S.pempty[bq]);
        const int v = a0 + q, j = v >> 9, rb = v & 511;
        while (sgi + 1 < S.nseg && q > S.seg[sgi].qlast) ++sgi;
        const int cnt = S.jC[j];
        const float gs = __ldg(p.gsA + S.jE[j]);
        const int dl = rb * 8 + g - 16 * S.seg[sgi].kk0;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 2 * t + h;
          if (tk < cnt && dl < HSD) {
            const float ga = (h ? sm.y : sm.x) * gs, up = (h ? sm.w : sm.z) * gs;
            S.hS[sgi][tk][dl] = __float2bfloat16(__fdividef(ga, 1.f + __expf(-ga)) * up);
          }
        }
        if (q == S.seg[sgi].qlast) {   // this CTA's last unit of the segment: its h is complete
          __syncwarp();
          if (lane == 0) mbar_arrive(&S.hready[sgi]);
        }
      }
    }
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
  if (c == 0 && tid == 0) atomicAdd(p.ctr, 1);
  // ---- final pass: one item per token; per local pair the RMS norm of the finished y row; y re-zeroed
  if (tid < NT) {
    for (int tk = c; tk < p.m; tk += G) {
      float a[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) a[k] = 0.f;
#pragma unroll 1
      for (int s = 0; s < 8; ++s) {
        const int pr = S.pos[tk * 8 + s];   // block-uniform
        if (pr < 0) continue;
        float4* yr = reinterpret_cast<float4*>(p.ybuf + (size_t)pr * HD) + tid * 4;
        float4 y[4];
        float sq = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          y[k] = __ldcg(yr + k);
          sq += y[k].x * y[k].x + y[k].y * y[k].y + y[k].z * y[k].z + y[k].w * y[k].w;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) __stcg(yr + k, make_float4(0.f, 0.f, 0.f, 0.f));
#pragma unroll
        for (int d = 16; d; d >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, d);
        if (lane == 0) S.fred[warp] = sq;
        bar_fin();
        float ss = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) ss += S.fred[w];
        bar_fin();
        const float cf = __ldg(p.tw + tk * 8 + s) * rsqrtf(ss * (1.f / HD) + p.eps);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          a[4 * k] += cf * y[k].x; a[4 * k + 1] += cf * y[k].y; a[4 * k + 2] += cf * y[k].z; a[4 * k + 3] += cf * y[k].w;
        }
      }
      uint4 pk[2];
      uint32_t* pw = reinterpret_cast<uint32_t*>(pk);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        __nv_bfloat162 v2 = __floats2bfloat162_rn(a[2 * k], a[2 * k + 1]);
        pw[k] = *reinterpret_cast<uint32_t*>(&v2);
      }
      uint4* op = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD) + tid * 2;
      op[0] = pk[0]; op[1] = pk[1];
    }
  }
  if (mk && tid == 0) mrow[9] = gt();
}
}  // namespace xs

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor ybuf, torch::Tensor ctr, torch::Tensor out, torch::Tensor marks,
         int64_t m, double eps, int64_t pdl, int64_t lo) {
  TORCH_CHECK(m >= 1 && m <= 16, "xs: 1 <= m <= 16");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(xs::xs_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, xs::SMEM_BYTES));
    attr = true;
  }
  xs::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.ids = ids.data_ptr<int>(); p.tw = tw.data_ptr<float>();
  p.wA = wA.data_ptr<uint8_t>(); p.wB = wB.data_ptr<uint8_t>(); p.gsA = gsA.data_ptr<float>(); p.gsB = gsB.data_ptr<float>();
  p.nw = nw.data_ptr<float>(); p.ybuf = ybuf.data_ptr<float>(); p.ctr = ctr.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(xs::G); lc.blockDim = dim3(xs::NTL); lc.dynamicSmemBytes = xs::SMEM_BYTES;
  lc.stream = at::cuda::getCurrentCUDAStream(); lc.attrs = at; lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, xs::xs_kernel, p));
}

std::vector<int64_t> attrs() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, xs::xs_kernel));
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)xs::SMEM_BYTES, xs::G, xs::NS, (int64_t)sizeof(xs::Smem)};
}
'''

CPP_SRC = r'''
#include <torch/extension.h>
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor ybuf, torch::Tensor ctr, torch::Tensor out, torch::Tensor marks,
         int64_t m, double eps, int64_t pdl, int64_t lo);
std::vector<int64_t> attrs();
'''


def build(G, ns=3, arch="sm_90a", name="xs_lab", verbose=False, l2=False):
    from torch.utils.cpp_extension import load_inline
    gen = f"-gencode=arch=compute_{arch[3:].rstrip('a')}{'a' if arch.endswith('a') else ''},code={arch}"
    flags = ["-O3", f"-DXS_G={G}", f"-DXS_NS={ns}", f"-DXS_L2={1 if l2 else 0}", "-Xptxas=-warn-spills", gen]
    if verbose:
        flags.append("-Xptxas=-v")
    tag = hashlib.sha1((CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags, verbose=verbose)
    return mod, time.time() - t0


# ---------------------------------------------------------------- XS W2 prepack (bit / byte permutation only)
def prepack_w2(w, sf, pos=None):
    """one expert's W2: w uint8 [4096, 2048] packed codes, sf e4m3 [4096, 256] -> uint8 [256 groups, 256 k16 steps, 144]:
    per (16-row group gr, k16 step s): 32 lane words (lane = 4 g + t; word r/h slots per VMS.POS: r0 = row g, K 16 s + 2 t + h;
    r1 = row g + 8, same K; r2 = row g, K + 8; r3 = row g + 8, K + 8) then 16 scale bytes [g][row g, row g + 8]."""
    cu = VMS._codes(w).view(256, 2, 8, 256, 2, 4, 2).to(torch.int64)    # gr, hi8, g, s, kh, t, h   (K = 16 s + 8 kh + 2 t + h)
    cu = cu.permute(0, 3, 2, 5, 4, 1, 6).contiguous()                    # gr, s, g, t, kh, hi8, h ; reg r = 2 kh + hi8
    cu = cu.view(256, 256, 8, 4, 4, 2)
    word = torch.zeros(cu.shape[:4], dtype=torch.int64, device=w.device)
    for r in range(4):
        for h in range(2):
            cd = cu[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << (pos or VMS.POS)[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32).view(256, 256, 32)
    sb = sf.view(torch.uint8).view(256, 2, 8, 256).permute(0, 3, 2, 1).contiguous().view(256, 256, 16)   # gr, s, g, hi8
    return torch.cat((word.view(torch.uint8).view(256, 256, 128), sb), dim=2).contiguous()


# UNP L2 positions of the layer's VMS (k3_gc_13 VMS_POS): with build(l2=True) XS shares the layer's VMS wA
POS_L2 = (((6, 7, 8, 15), (22, 23, 24, 31)), ((28, 29, 30, 5), (12, 13, 14, 21)),
          ((25, 26, 27, 2), (9, 10, 11, 18)), ((4, 0, 1, 3), (20, 16, 17, 19)))


def prepack_a(w, sf, pos):
    """VMS phase-A units with an explicit bit-position table (VMS.prepack_expert with pos instead of VMS.POS)"""
    n = w.shape[0]
    rows = torch.arange(n, device=w.device)
    rb, i = rows // 16, rows % 16
    rows = torch.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    cu = VMS._codes(w[rows]).view(n // 16, 16, 4096)
    nu = cu.shape[0]
    x = cu.view(nu, 2, 8, 8, 8, 4, 4, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8).contiguous().view(nu, 8, 8, 8, 4, 4, 4, 2).to(torch.int64)
    word = torch.zeros(x.shape[:6], dtype=torch.int64, device=w.device)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << pos[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32).view(torch.uint8).view(nu, 32768)
    sb = VMS._pack_unit_scales(sf.view(torch.uint8)[rows].view(nu, 16, 256))
    return torch.cat((word, sb), dim=1).contiguous()


def prepack(w, log=print, l2=False):
    t0 = time.time()
    if not (VMS.scales_ok(w["w13_sf"]) and VMS.scales_ok(w["w2_sf"])):
        log("[xs] scale bytes outside 0x08..0x7E: XS route off")
        return None
    E = w["w13"].shape[0]
    dev = w["w13"].device
    wA = torch.empty((E, 512, UB), dtype=torch.uint8, device=dev)
    wB = torch.empty((E, 256, 256, BS), dtype=torch.uint8, device=dev)
    for e in range(E):
        wA[e] = prepack_a(w["w13"][e], w["w13_sf"][e], POS_L2) if l2 else VMS.prepack_expert(w["w13"][e], w["w13_sf"][e], True)
        wB[e] = prepack_w2(w["w2"][e], w["w2_sf"][e], POS_L2 if l2 else None)
    two62 = float(2 ** 62)
    pk = {"wA": wA, "wB": wB, "gsA": (w["w13_gs"].float().reshape(E) * two62).contiguous(),
          "gsB": (w["w2_gs"].float().reshape(E) * two62).contiguous(), "nw": (1.0 + w["post_norm_w"].float()).contiguous()}
    torch.cuda.synchronize()
    log(f"[xs] prepack {E} experts in {time.time() - t0:.1f} s")
    return pk


def scratch(dev):
    return {"ybuf": torch.zeros((MAXP, HD), dtype=torch.float32, device=dev),
            "ctr": torch.zeros((8,), dtype=torch.int32, device=dev),
            "nomarks": torch.empty((0,), dtype=torch.int64, device=dev)}


def call(mod, pk, sc, x, ids, tw, out=None, marks=None, pdl=False, eps=1e-8, lo=0):
    m = x.shape[0]
    if out is None:
        out = torch.empty((m, HD), dtype=torch.bfloat16, device=x.device)
    mod.run(x, ids, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["ybuf"], sc["ctr"], out,
            sc["nomarks"] if marks is None else marks, m, eps, 1 if pdl else 0, lo)
    return out

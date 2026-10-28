"""vks_probe.py: step-0 probe for VMS-KS (notes/study/indep_mid.md s3), K-split wide jobs in the VMS consumer.

Question. Keep the VMS consumer's x fragments in the same 64 registers per thread, and trade K-range for token width:
  J0 = VMS today: 16 rows x 4096 K per unit, 8 tokens, warp K-slice 512, 1 m16n8k16 per decoded fragment;
  J1 = 32 rows x 2048 K (K-half h), 16 tokens, warp K-slice 256, 2 mma per decoded fragment;
  J2 = 32 rows x 1024 K (K-quarter q), 32 tokens, warp K-slice 128, 4 mma per decoded fragment.
Every weight is then decoded once for up to 32 tokens; the 2 / 4 CTAs that hold the other K-slices add their partials.
Does a J1 / J2 slot (36,864 B of weights) cost close to a J0 slot? If yes, the hot experts of decode16 / decode32
(9..32 tokens) stream once instead of ceil(n / 8) times, with no register or x-smem growth (the X2S / K16 walls).

Structure (1 CTA/SM, 256 threads = 8 compute warps; weights = the VMS W13 units of kernels/lab/vms_moe_lab.py prepack, so
the probe needs no new prepack; J1 / J2 slots are built from K-slices of two adjacent VMS units by bulk copies):
- ring NS = 3 slots x 36,864 B; thread 0 refills a slot right after that slot's last per-unit barrier (the VMS protocol);
  J0 slot = 1 VMS unit (1 copy), J1 slot = K-half h of VMS units 2P, 2P + 1 (2 x 16 KB codes + 2 x 2 KB scales),
  J2 slot = 2 units of 18,432 B, each K-quarter q of VMS units 2U, 2U + 1 (8 copies);
- x fragments loaded once per CTA (16 uint4 in every mode); CTA c uses K-half c & 1 (J1) / K-quarter c & 3 (J2);
- per unit: VMS decode (bit-position masks + bf16 scale multiply), mma.sync m16n8k16 per (m-tile, n-tile); partials to a
  double-buffered smem area, one bar.sync per unit, the 8 warps sum the 8 K-slice partials and red.add.v4 into a
  fragment-native fp32 output (out[16-row tile][n-tile][lane][4]). The design's real epilogue (SwiGLU / y rows after the
  last K-slice arrives) is not here; its cost is one 2 KB fixup per J1 unit, priced on paper.
MODE (compile-time, one __global__ each, so ptxas allocates registers per arm): 0 J0 only, 1 J1 only, 2 J2 only,
3 fused (all three paths in one kernel, runtime kind = the codegen tax a real VMS-KS kernel pays), 4 null consumer
(waits + barriers + refills only: the stream reference).
Slot time = (T(spc_b) - T(spc_a)) / (spc_b - spc_a): launch, ramp and drain cancel.
"""
from __future__ import annotations

import hashlib
import statistics
import time

DEV_SRC = r"""
#include <cuda_bf16.h>
#include <stdint.h>

namespace vks {
constexpr int UB = 36864, UC = 32768, NS = 3, NW = 8, NT = 256;
constexpr int RF4 = 8;                                   // float4 partials per lane per unit (max: J2)
constexpr int RED_OFF = 128;                             // barriers first
constexpr int RING_OFF = RED_OFF + 2 * NW * RF4 * 32 * 16;
constexpr int SMEM = RING_OFF + NS * UB;                 // 176,256 B
constexpr long long WAIT_MAX = 1ll << 24;

struct Params {
  const unsigned char* wA;     // VMS W13 units [E][512][36864]
  const __nv_bfloat16* x;      // [32][4096]
  float* out;                  // [E * 512 tiles][4 n-tiles][32 lanes][4]
  int* err;
  long long* marks;            // [G][4] globaltimer / clock64 at entry and exit (nullptr: off)
  int spc;                     // slots per CTA
  int kind;                    // MODE 3: 0 / 1 / 2
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {   // bounded (no trap): R9
  if (mbar_try(b, par)) return;
  long long n = 0;
  while (!mbar_try(b, par)) {
    if (++n > WAIT_MAX) { atomicExch(err, 1); return; }
  }
}
__device__ __forceinline__ void expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t pol_ef() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void bar_compute() { asm volatile("bar.sync 1, 256;\n" ::: "memory"); }
__device__ __forceinline__ void red_v4(float* p, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
// e4m3 byte -> bf16x2(scale * 2^64) in both halves (VMS scd; exact for bytes 0x08..0x7E)
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
// VMS bit-position decode of one code word (8 weights: rows g / g + 8, 4 k each) + block scales
struct Frag { uint32_t a0, a1, a2, a3; };
__device__ __forceinline__ Frag dq(uint32_t qw, uint32_t sg, uint32_t sg8) {
  const uint32_t M = 0x81C081C0u;
  const uint32_t r0 = qw & M;
  const uint32_t r1 = (qw * 16u) & M;
  const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;
  const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
  return Frag{bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8)};
}
// B fragment of k16 step s (0..3) of one chunk from its two x uint4 (K = ... + 16 t + 4 s + {0,1 | 2,3})
__device__ __forceinline__ uint32_t xb0(const uint4& X0, const uint4& X1, int s) {
  return s == 0 ? X0.x : s == 1 ? X0.z : s == 2 ? X1.x : X1.z;
}
__device__ __forceinline__ uint32_t xb1(const uint4& X0, const uint4& X1, int s) {
  return s == 0 ? X0.y : s == 1 ? X0.w : s == 2 ? X1.y : X1.w;
}

// ---- x fragments, once per CTA (16 uint4 in every kind)
template <int K>
__device__ __forceinline__ void load_x(const __nv_bfloat16* x, uint4 (&xr)[16], int w, int lane, int part) {
  const int g = lane >> 2, t = lane & 3;
  constexpr int NTL = K == 0 ? 1 : K == 1 ? 2 : 4, NCH = 8 / NTL;
#pragma unroll
  for (int nt = 0; nt < NTL; ++nt) {
    const int k0 = K == 0 ? 512 * w : K == 1 ? 2048 * part + 256 * w : 1024 * part + 128 * w;
    const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)(8 * nt + g) * 4096 + k0 + 16 * t);
#pragma unroll
    for (int cc = 0; cc < NCH; ++cc) {
      xr[nt * 2 * NCH + 2 * cc] = __ldg(src + 8 * cc);
      xr[nt * 2 * NCH + 2 * cc + 1] = __ldg(src + 8 * cc + 1);
    }
  }
}

// ---- consumers: one unit -> per-lane partials acc[f4][4]
// J0: VMS unit (16 rows x 4096 K); 4 accumulator chains by step, summed at the end (VMS)
__device__ __forceinline__ void consume0(const unsigned char* u, const uint4 (&xr)[16], float (&acc)[8][4], uint32_t k16,
                                         int w, int lane) {
  float a[4][4];
#pragma unroll
  for (int s = 0; s < 4; ++s) { a[s][0] = 0.f; a[s][1] = 0.f; a[s][2] = 0.f; a[s][3] = 0.f; }
  const uint4 sc = *reinterpret_cast<const uint4*>(u + UC + w * 512 + lane * 16);
#pragma unroll
  for (int cc = 0; cc < 8; ++cc) {
    const uint4 q = *reinterpret_cast<const uint4*>(u + (w * 8 + cc) * 512 + lane * 16);
    const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
    const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
    const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
    const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const Frag f = dq(qq[s], sg, sg8);
      mma(a[s], f.a0, f.a1, f.a2, f.a3, xb0(xr[2 * cc], xr[2 * cc + 1], s), xb1(xr[2 * cc], xr[2 * cc + 1], s));
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[0][i] = a[0][i] + a[1][i] + a[2][i] + a[3][i];
}
// J1: 32 rows (m-tiles = VMS units 2P, 2P+1) x K-half; warp K-slice 256 = 4 chunks; 2 n-tiles; chains acc[mt][nt]
__device__ __forceinline__ void consume1(const unsigned char* u, const uint4 (&xr)[16], float (&acc)[8][4], uint32_t k16,
                                         int w, int lane) {
#pragma unroll
  for (int f = 0; f < 4; ++f) { acc[f][0] = 0.f; acc[f][1] = 0.f; acc[f][2] = 0.f; acc[f][3] = 0.f; }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const uint2 sc = *reinterpret_cast<const uint2*>(u + UC + mt * 2048 + (w >> 1) * 512 + lane * 16 + 8 * (w & 1));
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      const uint4 q = *reinterpret_cast<const uint4*>(u + mt * 16384 + ((w >> 1) * 8 + 4 * (w & 1) + cc) * 512 + lane * 16);
      const uint32_t sw = cc < 2 ? sc.x : sc.y;
      const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
      const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
      const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const Frag f = dq(qq[s], sg, sg8);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
          mma(acc[mt * 2 + nt], f.a0, f.a1, f.a2, f.a3, xb0(xr[nt * 8 + 2 * cc], xr[nt * 8 + 2 * cc + 1], s),
              xb1(xr[nt * 8 + 2 * cc], xr[nt * 8 + 2 * cc + 1], s));
      }
    }
  }
}
// J2: one 18,432 B unit = 32 rows x K-quarter; warp K-slice 128 = 2 chunks; 4 n-tiles; chains acc[mt][nt]
__device__ __forceinline__ void consume2(const unsigned char* u, const uint4 (&xr)[16], float (&acc)[8][4], uint32_t k16,
                                         int w, int lane) {
#pragma unroll
  for (int f = 0; f < 8; ++f) { acc[f][0] = 0.f; acc[f][1] = 0.f; acc[f][2] = 0.f; acc[f][3] = 0.f; }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    const uint32_t sw = *reinterpret_cast<const uint32_t*>(u + 16384 + mt * 1024 + (w >> 2) * 512 + lane * 16 + 4 * (w & 3));
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const uint4 q = *reinterpret_cast<const uint4*>(u + mt * 8192 + ((w >> 2) * 8 + 2 * (w & 3) + cc) * 512 + lane * 16);
      const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
      const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
      const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const Frag f = dq(qq[s], sg, sg8);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
          mma(acc[mt * 4 + nt], f.a0, f.a1, f.a2, f.a3, xb0(xr[nt * 4 + 2 * cc], xr[nt * 4 + 2 * cc + 1], s),
              xb1(xr[nt * 4 + 2 * cc], xr[nt * 4 + 2 * cc + 1], s));
      }
    }
  }
}

// ---- partials -> smem, one barrier, K-slice sum by all 8 warps, red.add.v4 into the fragment-native output
template <int F4>
__device__ __forceinline__ void reduce_out(const float (&acc)[8][4], float4* red, int b, int w, int lane, int tid, float* out,
                                           int tile0, int ntl) {
#pragma unroll
  for (int f = 0; f < F4; ++f) red[((b * NW + w) * RF4 + f) * 32 + lane] = make_float4(acc[f][0], acc[f][1], acc[f][2], acc[f][3]);
  bar_compute();
  if (tid < 32 * F4) {
    const int ln = tid & 31, f = tid >> 5;
    float4 s = red[((b * NW) * RF4 + f) * 32 + ln];
#pragma unroll
    for (int v = 1; v < NW; ++v) {
      const float4 o = red[((b * NW + v) * RF4 + f) * 32 + ln];
      s.x += o.x; s.y += o.y; s.z += o.z; s.w += o.w;
    }
    const int mt = f / ntl, nt = f % ntl;
    red_v4(out + ((size_t)((tile0 + mt) * 4 + nt) * 32 + ln) * 4, s);
  }
}

// ---- slot fill (thread 0): source bytes of slot index s of CTA c for kind K
template <int K>
__device__ __forceinline__ void fill(const Params& p, uint32_t dst, uint32_t bar, int c, int s, uint64_t pol) {
  expect_tx(bar, UB);
  if (K == 0) {
    bulk(dst, p.wA + (size_t)(c * p.spc + s) * UB, UB, bar, pol);
  } else if (K == 1) {
    const int h = c & 1, P = (c >> 1) * p.spc + s;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const unsigned char* v = p.wA + (size_t)(2 * P + mt) * UB;
      bulk(dst + mt * 16384, v + 16384 * h, 16384, bar, pol);
      bulk(dst + UC + mt * 2048, v + UC + 2048 * h, 2048, bar, pol);
    }
  } else {
    const int q = c & 3;
#pragma unroll
    for (int ub = 0; ub < 2; ++ub) {
      const int U = (c >> 2) * 2 * p.spc + 2 * s + ub;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const unsigned char* v = p.wA + (size_t)(2 * U + mt) * UB;
        bulk(dst + ub * 18432 + mt * 8192, v + 8192 * q, 8192, bar, pol);
        bulk(dst + ub * 18432 + 16384 + mt * 1024, v + UC + 1024 * q, 1024, bar, pol);
      }
    }
  }
}

template <int K>
__device__ __forceinline__ void run_kind(const Params& p, unsigned char* smem, uint32_t bars, int c, int tid, int w, int lane) {
  float4* red = reinterpret_cast<float4*>(smem + RED_OFF);
  const unsigned char* ring = smem + RING_OFF;
  const uint32_t k16 = 16u + (uint32_t)(p.spc >> 30);                // opaque 16 (VMS)
  uint4 xr[16];
  load_x<K>(p.x, xr, w, lane, K == 1 ? (c & 1) : (c & 3));
  uint64_t pol = 0;
  if (tid == 0) {
    pol = pol_ef();
    for (int s = 0; s < NS && s < p.spc; ++s) fill<K>(p, su32(ring + s * UB), bars + 8 * s, c, s, pol);
  }
  int b = 0;
  for (int s = 0; s < p.spc; ++s) {
    const int sl = s % NS;
    mbar_wait(bars + 8 * sl, (uint32_t)((s / NS) & 1), p.err);
    const unsigned char* slot = ring + sl * UB;
    float acc[8][4];
    if (K == 0) {
      consume0(slot, xr, acc, k16, w, lane);
      reduce_out<1>(acc, red, b, w, lane, tid, p.out, c * p.spc + s, 1);
      b ^= 1;
    } else if (K == 1) {
      consume1(slot, xr, acc, k16, w, lane);
      reduce_out<4>(acc, red, b, w, lane, tid, p.out, 2 * ((c >> 1) * p.spc + s), 2);
      b ^= 1;
    } else {
#pragma unroll 1
      for (int ub = 0; ub < 2; ++ub) {
        consume2(slot + ub * 18432, xr, acc, k16, w, lane);
        reduce_out<8>(acc, red, b, w, lane, tid, p.out, 2 * ((c >> 2) * 2 * p.spc + 2 * s + ub), 4);
        b ^= 1;
      }
    }
    if (tid == 0 && s + NS < p.spc) fill<K>(p, su32(ring + sl * UB), bars + 8 * sl, c, s + NS, pol);
  }
}

__device__ __forceinline__ void run_null(const Params& p, unsigned char* smem, uint32_t bars, int c, int tid) {
  const unsigned char* ring = smem + RING_OFF;
  uint64_t pol = 0;
  if (tid == 0) {
    pol = pol_ef();
    for (int s = 0; s < NS && s < p.spc; ++s) fill<0>(p, su32(ring + s * UB), bars + 8 * s, c, s, pol);
  }
  for (int s = 0; s < p.spc; ++s) {
    const int sl = s % NS;
    mbar_wait(bars + 8 * sl, (uint32_t)((s / NS) & 1), p.err);
    bar_compute();
    if (tid == 0 && s + NS < p.spc) fill<0>(p, su32(ring + sl * UB), bars + 8 * sl, c, s + NS, pol);
  }
}

template <int MODE>
__global__ void __launch_bounds__(NT, 1) vks_kernel(const Params p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, c = blockIdx.x;
  const uint32_t bars = su32(smem);
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) mbar_init(bars + 8 * i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (tid == 0 && p.marks) { p.marks[c * 4] = gtime(); p.marks[c * 4 + 1] = clock64(); }
  if (MODE == 0) run_kind<0>(p, smem, bars, c, tid, w, lane);
  else if (MODE == 1) run_kind<1>(p, smem, bars, c, tid, w, lane);
  else if (MODE == 2) run_kind<2>(p, smem, bars, c, tid, w, lane);
  else if (MODE == 3) {
    if (p.kind == 0) run_kind<0>(p, smem, bars, c, tid, w, lane);
    else if (p.kind == 1) run_kind<1>(p, smem, bars, c, tid, w, lane);
    else run_kind<2>(p, smem, bars, c, tid, w, lane);
  } else run_null(p, smem, bars, c, tid);
  if (p.marks) {
    bar_compute();
    if (tid == 0) { p.marks[c * 4 + 2] = gtime(); p.marks[c * 4 + 3] = clock64(); }
  }
}
}  // namespace vks
"""

HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

template <int MODE>
static void vks_launch(const vks::Params& p, int G) {
  cudaFuncSetAttribute(vks::vks_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, vks::SMEM);
  vks::vks_kernel<MODE><<<G, vks::NT, vks::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int MODE>
static std::vector<int64_t> vks_attr() {
  cudaFuncSetAttribute(vks::vks_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, vks::SMEM);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, vks::vks_kernel<MODE>);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vks::vks_kernel<MODE>, vks::NT, vks::SMEM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vks::SMEM, occ};
}

std::vector<int64_t> vks_attrs(int64_t mode) {
  switch (mode) {
    case 0: return vks_attr<0>(); case 1: return vks_attr<1>(); case 2: return vks_attr<2>();
    case 3: return vks_attr<3>(); default: return vks_attr<4>();
  }
}

void vks_run(int64_t mode, int64_t kind, torch::Tensor wA, torch::Tensor x, torch::Tensor out, torch::Tensor err,
             torch::Tensor marks, int64_t spc, int64_t G) {
  vks::Params p;
  p.marks = marks.numel() >= G * 4 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.wA = reinterpret_cast<const unsigned char*>(wA.data_ptr());
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.out = out.data_ptr<float>();
  p.err = err.data_ptr<int>();
  p.spc = (int)spc;
  p.kind = (int)kind;
  switch (mode) {
    case 0: vks_launch<0>(p, (int)G); break;
    case 1: vks_launch<1>(p, (int)G); break;
    case 2: vks_launch<2>(p, (int)G); break;
    case 3: vks_launch<3>(p, (int)G); break;
    default: vks_launch<4>(p, (int)G); break;
  }
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> vks_attrs(int64_t mode);
void vks_run(int64_t mode, int64_t kind, torch::Tensor wA, torch::Tensor x, torch::Tensor out, torch::Tensor err,
             torch::Tensor marks, int64_t spc, int64_t G);
"""

FLAGS = ["-O3", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
# arm -> (MODE, runtime kind, tokens, slot kind); slot kind decides the bytes / tiles touched
ARMS = {"J0": (0, 0, 8, 0), "J1": (1, 1, 16, 1), "J2": (2, 2, 32, 2),
        "J0f": (3, 0, 8, 0), "J1f": (3, 1, 16, 1), "J2f": (3, 2, 32, 2), "Z": (4, 0, 8, 0)}


def build(verbose=False):
    from torch.utils.cpp_extension import load_inline
    src = DEV_SRC + HOST_SRC
    tag = hashlib.sha1((CPP_SRC + src + " ".join(FLAGS)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vks_probe_{tag}", cpp_sources=CPP_SRC, cuda_sources=src, functions=["vks_run", "vks_attrs"],
                      extra_cuda_cflags=FLAGS, verbose=verbose)
    return mod, time.time() - t0


def tiles_touched(kind, spc, G):
    """16-row tiles (expert-major VMS unit index) every K-slice of which the arm sums, all equal to [0, G * spc)."""
    return G * spc


def ref_tiles(w, x, ntile, ntok, dequant):
    """FP32 reference in the fragment-native layout [ntile][4][32][4] for tokens 0..ntok-1 (zeros elsewhere).
    VMS W13 unit rb of expert e: unit-local row i -> gate row 8 rb + i (i < 8) or up row 4096 + 8 rb + i - 8."""
    import torch
    dev = x.device
    E = -(-ntile // 512)
    ref = torch.zeros((ntile, 4, 32, 4), dtype=torch.float32, device=dev)
    xf = x[:ntok].float()
    lane = torch.arange(32, device=dev)
    g, t = lane // 4, lane % 4
    for e in range(E):
        W = dequant(w["w13"][e], w["w13_sf"][e], 1.0)                 # [8192, 4096] fp32 (global scale 1: the kernel
        h = W @ xf.T                                                  # output carries 2^-62 x gs13 per row)
        n0, n1 = e * 512, min(ntile, (e + 1) * 512)
        rb = torch.arange(n1 - n0, device=dev)
        for c in range(4):
            i = g + 8 * (c >> 1)                                      # unit-local row of fragment element c
            rows = torch.where(i < 8, 8 * rb[:, None] + i[None, :], 4096 + 8 * rb[:, None] + (i[None, :] - 8))
            for nt in range(4):
                tok = 8 * nt + 2 * t + (c & 1)
                ok = tok < ntok
                v = h[rows, torch.clamp(tok, max=ntok - 1)[None, :].expand_as(rows)]
                ref[n0:n1, nt, :, c] = torch.where(ok[None, :], v, torch.zeros_like(v))
    return ref * float(2.0 ** -62)


def probe(wA, w, x, dequant, timer, ov, G, log=print, spcs=(24, 48), rounds=12, arms=None):
    """wA: VMS W13 units [E, 512, 36864] (E >= G * max(spcs) / 512); w: the moe problem weights (FP32 check);
    x [32, 4096] bf16. Returns per-arm slot us and checks."""
    import torch
    mod, dt = build()
    arms = [a for a in (arms or list(ARMS)) if a in ARMS]
    log(f"[vks] built in {dt:.0f} s; attrs (regs, local, smem, occ): " +
        " ".join(f"{m} {list(mod.vks_attrs(m))}" for m in range(5)))
    assert G % 4 == 0, "J2 needs G % 4 == 0"
    dev = x.device
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    smax = max(spcs)
    ntile = G * smax
    assert wA.shape[0] * 512 >= ntile, "prepack more experts"
    out = torch.zeros((ntile, 4, 32, 4), dtype=torch.float32, device=dev)
    flat = wA.reshape(-1)
    marks = torch.zeros((G * 4,), dtype=torch.int64, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)

    def fn(a, spc, mk=False):
        mode, kind = ARMS[a][0], ARMS[a][1]
        return lambda: mod.vks_run(mode, kind, flat, x, out, err, marks if mk else nomark, spc, G)
    # correctness at the largest spc (out zeroed: every arm red.adds its K-slices into it)
    rel = {}
    refs = {}
    for a in arms:
        if a == "Z":
            continue
        ntok = ARMS[a][2]
        if ntok not in refs:
            refs[ntok] = ref_tiles(w, x, ntile, ntok, dequant)
        out.zero_()
        err.zero_()
        fn(a, smax)()
        torch.cuda.synchronize()
        r = refs[ntok]
        rel[a] = float((out - r).norm() / r.norm().clamp(min=1e-30)) if int(err[0]) == 0 else float("inf")
    graphs = {}
    for a in arms:
        for spc in spcs:
            f = fn(a, spc)
            with torch.cuda.stream(timer.stream):
                f()
                torch.cuda.synchronize()
                g_ = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_, stream=timer.stream):
                    f()
            torch.cuda.synchronize()
            graphs[(a, spc)] = g_
    keys = list(graphs)
    ts = {k: [] for k in keys}
    for rd in range(rounds):
        k0 = rd % len(keys)
        for k in keys[k0:] + keys[:k0]:
            ts[k].append(max(timer.time_replay(graphs[k]) - ov, 1e-3))
    us = {k: statistics.median(v) for k, v in ts.items()}
    lo, hi = min(spcs), max(spcs)
    slot = {a: (us[(a, hi)] - us[(a, lo)]) / (hi - lo) for a in arms}
    clk = {}
    for a in arms:   # one marked cold-L2 run per arm: effective SM clock (clock64 / globaltimer) and CTA end spread
        marks.zero_()
        with torch.cuda.stream(timer.stream):
            timer.flush()
            fn(a, hi, mk=True)()
        torch.cuda.synchronize()
        m = marks.view(G, 4).double()
        dt = (m[:, 2] - m[:, 0]).clamp(min=1.0)
        ghz = (m[:, 3] - m[:, 1]) / dt
        ends = (m[:, 2] - m[:, 0].min()) / 1e3
        clk[a] = {"ghz_p50": float(ghz.median()), "ghz_min": float(ghz.min()), "end_p50": float(ends.median()),
                  "end_max": float(ends.max())}
    for a in arms:
        log(f"[vks] {a:4s}: us spc{lo} {us[(a, lo)]:.1f} spc{hi} {us[(a, hi)]:.1f} | slot us {slot[a]:.3f} | "
            f"SM clock {clk[a]['ghz_p50']:.2f} GHz (min {clk[a]['ghz_min']:.2f}) | CTA end p50 / max "
            f"{clk[a]['end_p50']:.1f} / {clk[a]['end_max']:.1f} us | rel {rel.get(a, float('nan')):.1e} err {int(err[0])}")
    return {"slot_us": slot, "us": {f"{a}:{s}": v for (a, s), v in us.items()}, "rel": rel, "err": int(err[0]),
            "clock": clk}


def decide(slot, rel, log=print):
    """Pre-registered rule (notes/study/indep_mid.md s3.4)."""
    ok = all(rel.get(a, 9) <= 1e-2 for a in ("J0", "J1", "J2", "J0f", "J1f", "J2f") if a in rel)
    if not ok:
        log("[vks] INVALID: an arm failed the FP32 check")
        return "INVALID"
    t0s, t0, t1, t2 = slot["J0"], slot["J0f"], slot["J1f"], slot["J2f"]
    tax, r1, r2 = t0 / t0s, t1 / t0, t2 / t0
    log(f"[vks] codegen tax J0f/J0 {tax:.3f}; per 36 KB slot J1f/J0f {r1:.3f}, J2f/J0f {r2:.3f}; Z {slot.get('Z', 0):.3f} us")
    if tax > 1.05 or r1 > 1.45:
        verdict = "KILL"
    elif tax <= 1.02 and r1 <= 1.20 and r2 <= 1.40:
        verdict = "GO (T=16 and T=32)"
    elif tax <= 1.02 and r1 <= 1.20:
        verdict = "GO (T=16; J2 above 1.40: T=32 caps jobs at 16 tokens)"
    else:
        verdict = "PARK (T=32 only if J2f/J0f <= 1.40)"
    log(f"[vks] verdict: {verdict}")
    return verdict

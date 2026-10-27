"""AW0: all-RS-wgmma W13 consumers, step-0 lab (notes/study/moe_hot_fresh.md, request R-hot-1). Timing + correctness.

HUC0 (R-hot-0b, killed) measured one unserialized RS-wgmma warpgroup per SM streaming ~2.4 TB/s of NVFP4 weights alone
(~4.6 GB/s per warp, plus ~40 % X from L2), i.e. ~3x the bytes per warp of K2A's mma.sync consumer. Question here: do 2-3
such warpgroups per SM, each with its own ring and every expert as its consumer (cold experts at wgmma N = 8 / 16, hot ones at
N <= 64 per unit, all tokens of a unit in one weight pass, whole units, no split-K, no role split), stream the W13 phase at
>= 4 TB/s and beat K2A's K1 at bs 256 / 128?

One launch, 1 CTA/SM, 128 x (NWG + 1) threads, setmaxnreg: WG0 = NWG producer warps (one per ring, 40 regs), WG1..NWG =
consumer warpgroups (RC regs). A unit = one 128-row block (two m64 tiles: K2A's prepacked stage, warp w decodes tiles w and
w + 4) or, for N <= HALF_LE, one 64-row half block (tiles 4h..4h+3, half the stage copied), x full K x one token group of
one expert. Units are sorted by modelled cost and dealt serpentine over the G x NWG warpgroups (closed form). Plans and the
X pack (K2A's xw_u32 core-matrix layout) are host-built (lab only, outside timing): the gate adds 3 us for an on-device
plan + a PDL X-pack pre-kernel. mode bit 0 = null consumers.
cfg 0: NWG 2, NMAX 64, NST 2, RC 232 (384 thr) | cfg 1: NWG 3, NMAX 32, NST 2, RC 152 (512) | cfg 2: NWG 2, NMAX 32, NST 3, RC 232
"""
import hashlib
import statistics
import time

import torch


def _aw0_wg_spec(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    cons = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    return ("template <> __device__ __forceinline__ void wgmma_rs2<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
            "uint32_t a3, uint32_t lo, uint32_t hi) {\n  asm volatile(\"{\\n.reg .pred p;\\n.reg .b64 dd;\\nmov.b64 dd, "
            "{%%%d, %%%d};\\nsetp.ne.b32 p, 1, 0;\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, dd, p, 1, "
            "1, 0;\\n}\\n\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}\n"
            % (n, na + 4, na + 5, n, outs, a, cons))


AW0_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace aw {
constexpr int KDIM = 4096;
constexpr int SW = 18432;
constexpr long long WAIT_BUDGET = 1ll << 30;    // clock64 cycles (~0.5 s) per wait -> error flag, no trap

struct Args {
  const __nv_bfloat16* x; const uint8_t* ws13; const float* gsc13;
  const int* plan; const int4* hu; const int* hr; const uint8_t* xp; __nv_bfloat16* inter; int* err; long long* marks;
  int mode;
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout (K2A v3)
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
      if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }
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
template <int N> __device__ __forceinline__ void wgmma_rs2(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
@WGSPECS@

// ---------------- producer (one warp per ring) ----------------
template <int NST, int SLOT>
__device__ __forceinline__ void producer(const Args& A, uint32_t ring_a, uint32_t full_a, uint32_t empty_a, int b, int e,
                                         int lane) {
  const uint64_t pol = policy_evict_first();
  int it = 0;
  for (int u = b; u < e; ++u) {
    const int4 h = A.hu[u];   // expert, row block | half << 8 (2 = both halves), X byte offset, tokens | npad << 16
    const int rb = h.y & 255, half = (h.y >> 8) & 3;
    const uint32_t xb = (uint32_t)(h.w >> 16) * 512u;
    const uint8_t* wb = A.ws13 + (size_t)(h.x * 64 + rb) * 16 * SW;
    const uint8_t* xs = A.xp + (size_t)h.z;
    for (int s = 0; s < 16; ++s, ++it) {
      const int slot = it % NST;
      mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, A.err);
      if (lane == 0) {
        const uint32_t dst = ring_a + slot * SLOT, fb = full_a + 8 * slot;
        const uint8_t* ws = wb + (size_t)s * SW;
        if (half == 2) {
          mbar_expect_tx(fb, SW + xb);
          bulk_g2s_ef(dst, ws, SW, fb, pol);
        } else {   // tiles 4h .. 4h + 3: 8 KB of codes + 1 KB of scales, placed where tiles 0..3 live
          mbar_expect_tx(fb, 9216 + xb);
          bulk_g2s_ef(dst, ws + half * 8192, 8192, fb, pol);
          bulk_g2s_ef(dst + 16384, ws + 16384 + half * 1024, 1024, fb, pol);
        }
        bulk_g2s(dst + SW, xs + (size_t)s * xb, xb, fb);
      }
      __syncwarp();
    }
  }
}

__device__ __forceinline__ void hot_decode(const uint4 w, const uint32_t sw2, int par, uint32_t* h) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  deq4(w.x, h); deq4(w.y, h + 4); deq4(w.z, h + 8); deq4(w.w, h + 12);
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) { h[jj] = bmul2(h[jj], Sg); h[8 + jj] = bmul2(h[8 + jj], S8); }
}
template <int N>
__device__ __forceinline__ void issue4(float (&acc)[N / 2], const uint32_t* h, uint32_t dlo, uint32_t dhi, int i) {
  wgmma_rs2<N>(acc, h[0], h[8], h[1], h[9], dlo + 64 * i, dhi);
  wgmma_rs2<N>(acc, h[2], h[10], h[3], h[11], dlo + 64 * i + 16, dhi);
  wgmma_rs2<N>(acc, h[4], h[12], h[5], h[13], dlo + 64 * i + 32, dhi);
  wgmma_rs2<N>(acc, h[6], h[14], h[7], h[15], dlo + 64 * i + 48, dhi);
}
template <int N>
__device__ __forceinline__ void flush(const Args& A, const float (&acc)[N / 2], int tile, int rbase, int ntok, float gs,
                                      int g, int t) {
  const int col = 8 * tile + g;
#pragma unroll
  for (int q = 0; q < N / 8; ++q)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int n = 8 * q + 2 * t + hh;
      if (n < ntok)
        A.inter[(size_t)(rbase + n) * KDIM + col] =
            __float2bfloat16_rn(swiglu(acc[4 * q + hh] * gs, acc[4 * q + 2 + hh] * gs));
    }
}
// full unit: tiles hw and hw + 4 (two m64 accumulators); per kb a tile-A group then a tile-B group, wait_group 1 before
// a buffer is rewritten, wait_group 0 before the slot is released (HUC0's validated loop)
template <int N, int NST, int SLOT>
__device__ __forceinline__ void unit2(const Args& A, const uint8_t* ring, uint32_t full_a, uint32_t empty_a, int& it,
                                      const int4 h, int rbase, int hw, int lane, bool nullc) {
  const int g = lane >> 2, t = lane & 3;
  float acc0[N / 2], acc1[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) { acc0[i] = 0.f; acc1[i] = 0.f; }
  uint32_t hA[16], hB[16];
  for (int s = 0; s < 16; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, A.err);
    if (!nullc) {
      const uint8_t* st = ring + slot * SLOT;
      const uint4* wpA = reinterpret_cast<const uint4*>(st + hw * 2048) + lane;
      const uint4* wpB = reinterpret_cast<const uint4*>(st + (hw + 4) * 2048) + lane;
      const uint32_t* spA = reinterpret_cast<const uint32_t*>(st + 16384 + hw * 256) + lane;
      const uint32_t* spB = reinterpret_cast<const uint32_t*>(st + 16384 + (hw + 4) * 256) + lane;
      const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);
      const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 wa = wpA[i * 32], wb = wpB[i * 32];
        const uint32_t sa2 = spA[(i >> 1) * 32], sb2 = spB[(i >> 1) * 32];
        wg_wait<1>();
        hot_decode(wa, sa2, i & 1, hA);
        wg_fence();
        issue4<N>(acc0, hA, dlo, dhi, i);
        wg_commit();
        wg_wait<1>();
        hot_decode(wb, sb2, i & 1, hB);
        wg_fence();
        issue4<N>(acc1, hB, dlo, dhi, i);
        wg_commit();
      }
      wg_wait<0>();
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  if (!nullc) {
    const float gs = __ldg(A.gsc13 + h.x);
    const int ntok = h.w & 0xffff, rb = h.y & 255;
    flush<N>(A, acc0, rb * 8 + hw, rbase, ntok, gs, g, t);
    flush<N>(A, acc1, rb * 8 + hw + 4, rbase, ntok, gs, g, t);
  }
}
// half unit: one m64 tile (warp hw decodes tile hw of the half stage); kb buffers alternate (K2A hot_item's scheme)
template <int N, int NST, int SLOT>
__device__ __forceinline__ void unit1(const Args& A, const uint8_t* ring, uint32_t full_a, uint32_t empty_a, int& it,
                                      const int4 h, int rbase, int hw, int lane, bool nullc) {
  const int g = lane >> 2, t = lane & 3;
  float acc0[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc0[i] = 0.f;
  uint32_t hb[2][16];
  for (int s = 0; s < 16; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, A.err);
    if (!nullc) {
      const uint8_t* st = ring + slot * SLOT;
      const uint4* wp = reinterpret_cast<const uint4*>(st + hw * 2048) + lane;
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + hw * 256) + lane;
      const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);
      const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 w = wp[i * 32];
        const uint32_t sw2 = sp[(i >> 1) * 32];
        if (i >= 2) wg_wait<1>();
        hot_decode(w, sw2, i & 1, hb[i & 1]);
        wg_fence();
        issue4<N>(acc0, hb[i & 1], dlo, dhi, i);
        wg_commit();
      }
      wg_wait<0>();
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  if (!nullc) {
    const float gs = __ldg(A.gsc13 + h.x);
    const int ntok = h.w & 0xffff, rb = h.y & 255, half = (h.y >> 8) & 3;
    flush<N>(A, acc0, rb * 8 + 4 * half + hw, rbase, ntok, gs, g, t);
  }
}

template <int NWG, int NMAX, int NST, int RC, int PR>
__global__ void __launch_bounds__(128 * (NWG + 1), 1) aw_kernel(const Args A) {
  constexpr int SLOT = SW + NMAX * 512;
  constexpr int RING = NST * SLOT;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NWG * RING);   // ring r: full [2 NST r ..], empty [.. + NST]
  if (tid == 0) {
    for (int r = 0; r < NWG; ++r)
      for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + 2 * NST * r + i), 1); mbar_init(sa(bars + 2 * NST * r + NST + i), 4); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (A.marks) A.marks[(size_t)blockIdx.x * 4 + 0] = gtime();
  }
  __syncthreads();
  const bool nullc = (A.mode & 1) != 0;
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(PR) : "memory");
    if (warp < NWG) {
      const int* pl = A.plan + ((size_t)blockIdx.x * NWG + warp) * 2;
      producer<NST, SLOT>(A, sa(smem + warp * RING), sa(bars + 2 * NST * warp), sa(bars + 2 * NST * warp + NST), pl[0], pl[1],
                          lane);
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(RC) : "memory");
    const int cw = wg - 1, hw = warp & 3;
    const int* pl = A.plan + ((size_t)blockIdx.x * NWG + cw) * 2;
    const uint8_t* ring = smem + cw * RING;
    const uint32_t full_a = sa(bars + 2 * NST * cw), empty_a = sa(bars + 2 * NST * cw + NST);
    int it = 0;
    for (int u = pl[0]; u < pl[1]; ++u) {
      const int4 h = A.hu[u];
      const int np = h.w >> 16, rbase = A.hr[u], half = (h.y >> 8) & 3;
      if (half == 2) {
        if (np <= 8) unit2<8, NST, SLOT>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
        else if (np <= 16) unit2<16, NST, SLOT>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
        else if (np <= 32) unit2<32, NST, SLOT>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
        else if constexpr (NMAX >= 48) {
          if (np <= 48) unit2<48, NST, SLOT>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
          else unit2<64, NST, SLOT>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
        }
      } else {
        if (np <= 8) unit1<8, NST, SLOT>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
        else unit1<16, NST, SLOT>(A, ring, full_a, empty_a, it, h, rbase, hw, lane, nullc);
      }
    }
    if (hw == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 4 + 1 + cw] = gtime();
  }
}

template <int NWG, int NMAX, int NST>
constexpr int smem_bytes() { return NWG * NST * (SW + NMAX * 512) + 16 * NWG * NST; }

template <int NWG, int NMAX, int NST, int RC, int PR>
void launch(const Args& a, int G) {
  constexpr int SM = smem_bytes<NWG, NMAX, NST>();
  cudaFuncSetAttribute(aw_kernel<NWG, NMAX, NST, RC, PR>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
  aw_kernel<NWG, NMAX, NST, RC, PR><<<G, 128 * (NWG + 1), SM, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int NWG, int NMAX, int NST, int RC, int PR>
std::vector<int64_t> attrs() {
  constexpr int SM = smem_bytes<NWG, NMAX, NST>();
  cudaFuncSetAttribute(aw_kernel<NWG, NMAX, NST, RC, PR>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, aw_kernel<NWG, NMAX, NST, RC, PR>);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, aw_kernel<NWG, NMAX, NST, RC, PR>, 128 * (NWG + 1), SM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM), occ};
}
}  // namespace aw

std::vector<int64_t> aw0_attrs(int64_t cfg) {
  switch (cfg) {
    case 0: return aw::attrs<2, 64, 2, 232, 40>();
    case 1: return aw::attrs<3, 32, 2, 152, 56>();
    case 2: return aw::attrs<2, 32, 3, 232, 40>();
    default: return {-1};
  }
}

void aw0_run(int64_t cfg, torch::Tensor x, torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor plan, torch::Tensor hu,
             torch::Tensor hr, torch::Tensor xp, torch::Tensor inter, torch::Tensor err, torch::Tensor marks, int64_t G,
             int64_t mode) {
  aw::Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>();
  a.plan = plan.data_ptr<int>();
  a.hu = reinterpret_cast<const int4*>(hu.data_ptr());
  a.hr = hr.data_ptr<int>();
  a.xp = reinterpret_cast<const uint8_t*>(xp.data_ptr());
  a.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  a.err = err.data_ptr<int>();
  a.marks = marks.numel() >= G * 4 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  a.mode = (int)mode;
  switch (cfg) {
    case 0: aw::launch<2, 64, 2, 232, 40>(a, (int)G); break;
    case 1: aw::launch<3, 32, 2, 152, 56>(a, (int)G); break;
    case 2: aw::launch<2, 32, 3, 232, 40>(a, (int)G); break;
    default: TORCH_CHECK(false, "aw0: bad cfg");
  }
}
""".replace("@WGSPECS@", "".join(_aw0_wg_spec(n) for n in (8, 16, 32, 48, 64)))

AW0_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> aw0_attrs(int64_t cfg);
void aw0_run(int64_t cfg, torch::Tensor x, torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor plan, torch::Tensor hu,
             torch::Tensor hr, torch::Tensor xp, torch::Tensor inter, torch::Tensor err, torch::Tensor marks, int64_t G,
             int64_t mode);
"""

AW0_CFGS = {0: (2, 64, 2, 168), 1: (3, 32, 2, 128), 2: (2, 32, 3, 168)}   # NWG, NMAX, NST, expected launch regs
AW0_EXPERT_W13 = 64 * 16 * 18432


def aw0_build(verbose=False):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((AW0_CPP + AW0_CUDA).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"aw0_{tag}", cpp_sources=AW0_CPP, cuda_sources=AW0_CUDA, functions=["aw0_run", "aw0_attrs"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"], verbose=verbose)
    return mod, time.time() - t0


def aw0_case(bs, seed, dev):
    """generate_input's normal / ep8 / busiest routing math (our own copy, seeds disjoint from the judge's)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def aw0_pack_x(x, toks, npad):
    """x rows of toks, zero-padded to npad, in K2A's wgmma B layout per 256-k stage: [s][n8][i 4][q 8][r 8][t 4][b 2]
    for k = 256 s + 64 i + 16 t + 2 q + b (xw_u32 order) -> uint8 [16 * npad * 512]."""
    X = torch.zeros((npad, 4096), dtype=torch.bfloat16, device=x.device)
    X[:len(toks)] = x[torch.tensor(toks, dtype=torch.long, device=x.device)]
    X = X.view(npad // 8, 8, 16, 4, 4, 8, 2).permute(2, 0, 3, 5, 1, 4, 6).contiguous()
    return X.view(torch.uint8).reshape(-1)


def aw0_plan(d, G, nwg, nmax, dev, half_le=16):
    """Every touched expert -> token groups of <= nmax (npad in 8 / 16 / 32 / 48 / 64) -> units (128-row blocks, or
    64-row halves when npad <= half_le), sorted by modelled cost, dealt serpentine over the G x nwg warpgroups."""
    ids = d["topk_ids"].long().cpu()
    M = ids.shape[0]
    toks = {e: [] for e in range(32)}
    for t in range(M):
        for j in range(8):
            e = int(ids[t, j]) & 255
            if e < 32:
                toks[e].append(t)
    cnt = {e: len(v) for e, v in toks.items()}
    exps = sorted([e for e in range(32) if cnt[e] > 0], key=lambda e: (-cnt[e], e))
    pads = [p for p in (8, 16, 32, 48, 64) if p <= nmax]
    route, xoff, order, macs = 0, 0, 0, 0
    route_tok, xp_parts, units = {}, [], []
    for e in exps:
        tl = toks[e]
        N = len(tl)
        ng = -(-N // nmax)
        base = -(-(-(-N // ng)) // 8) * 8
        for k in range(ng):
            tk = tl[k * base:(k + 1) * base]
            if not tk:
                continue
            nt = len(tk)
            npad = next(p for p in pads if p >= nt)
            xp_parts.append(aw0_pack_x(d["hidden_states"], tk, npad))
            for n, t in enumerate(tk):
                route_tok[route + n] = (t, e)
            xb = 16 * npad * 512
            if npad <= half_le:
                cost = max(64 * 4096 * npad / 700.0, (9216 * 16 + xb) / 14.0)
                for rb in range(64):
                    for hf in (0, 1):
                        units.append((cost, order, e, rb | (hf << 8), xoff, nt | (npad << 16), route))
                        order += 1
            else:
                cost = max(128 * 4096 * npad / 700.0, (18432 * 16 + xb) / 14.0)
                for rb in range(64):
                    units.append((cost, order, e, rb | (2 << 8), xoff, nt | (npad << 16), route))
                    order += 1
            macs += 64 * 128 * 4096 * nt
            xoff += xb
            route += nt
    V = G * nwg
    units.sort(key=lambda u: (-u[0], u[1]))
    per = [[] for _ in range(V)]
    for p, u in enumerate(units):
        r, i = divmod(p, V)
        per[i if r % 2 == 0 else V - 1 - i].append(u)
    hu, hr, plan, vcost = [], [], torch.zeros((V, 2), dtype=torch.int32), []
    for v in range(V):
        plan[v, 0] = len(hu)
        for (c, _, e, rbh, xo, w, rbase) in per[v]:
            hu.append((e, rbh, xo, w))
            hr.append(rbase)
        plan[v, 1] = len(hu)
        vcost.append(sum(c for c, *_ in per[v]))
    P = {
        "plan": plan.to(dev).contiguous(),
        "hu": torch.tensor(hu, dtype=torch.int32).reshape(-1, 4).to(dev).contiguous(),
        "hr": torch.tensor(hr, dtype=torch.int32).to(dev).contiguous(),
        "xp": torch.cat(xp_parts).contiguous(), "R": route, "route_tok": route_tok, "cnt": cnt, "experts": exps,
        "bytes": len(exps) * AW0_EXPERT_W13, "macs": macs, "units": len(units),
        "cost_ratio": max(vcost) / (sum(vcost) / V),
    }
    return P


def aw0_ref(d, w, P, dequant):
    """FP32 reference of inter for every planned route: silu(x W_gate^T) * (x W_up^T)."""
    x = d["hidden_states"].float()
    ref = torch.zeros((max(P["R"], 1), 4096), dtype=torch.float32, device=x.device)
    by_e = {}
    for r, (t, e) in P["route_tok"].items():
        by_e.setdefault(e, []).append((r, t))
    for e, lst in by_e.items():
        rr = torch.tensor([a for a, _ in lst], dtype=torch.long, device=x.device)
        tt = torch.tensor([b for _, b in lst], dtype=torch.long, device=x.device)
        W = dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e])
        h = x[tt] @ W.T
        ref[rr] = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        del W
    return ref


class AwTimer:
    """The judge clone's timer: 256 MB read before every replay (cold L2), events around graph.replay()."""

    def __init__(self):
        self.flush_buf = torch.ones((256 << 20) // 4, dtype=torch.float32, device="cuda")
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")
        self.stream = torch.cuda.Stream()

    def capture(self, fn):
        with torch.cuda.stream(self.stream):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                fn()
        torch.cuda.synchronize()
        return gr

    def time(self, gr):
        with torch.cuda.stream(self.stream):
            torch.sum(self.flush_buf, dim=0, out=self.sink)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    def overhead(self, n=60):
        tiny = torch.zeros(1, device="cuda")
        with torch.cuda.stream(self.stream):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                tiny.add_(0)
        torch.cuda.synchronize()
        return statistics.median(self.time(gr) for _ in range(n))


def aw0_lab(st, k_full, k1_only, k_reset, w, dequant, log=print, rounds=24, cases=((256, 6), (128, 4), (32, 2)),
            seed0=880001):
    """Arms per routing: K (shipped K2A, full call), S (cfg 0, null consumers), A0 / A1 / A2 (cfg 0 / 1 / 2), A0f (cfg 0
    without half units); then K1 (shipped K2A W13 kernel(s) alone, separate block, state reset after)."""
    dev = st["ws13"].device
    mod, dt = aw0_build()
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    okc = {}
    for c, (nwg, nmax, nst, regs) in AW0_CFGS.items():
        a = list(mod.aw0_attrs(c))
        okc[c] = a[0] == regs and a[1] <= 64 and a[3] >= 1   # cfg 1: one 8 B pointer reload across setmaxnreg
        log(f"[aw0] cfg {c} (NWG {nwg}, NMAX {nmax}, NST {nst}) attrs (regs, local, smem, occ) {a} -> "
            f"{'ok' if okc[c] else 'DISABLED (launch regs / spill / occupancy)'}")
    log(f"[aw0] built in {dt:.0f}s; {sms} SMs")
    if not okc[0]:
        log("[aw0] ABORT: cfg 0 unusable")
        return []
    G = sms
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    marks = torch.zeros((G * 4,), dtype=torch.int64, device=dev)
    nomark = torch.zeros((0,), dtype=torch.int64, device=dev)
    timer = AwTimer()
    ov = timer.overhead()
    log(f"[aw0] graph overhead {ov:.2f} us")
    rows = []
    for bs, ncase in cases:
        for r in range(ncase):
            d = aw0_case(bs, seed0 + 1000 * bs + r, dev)
            t0 = time.time()
            plans = {"A0": (0, aw0_plan(d, G, 2, 64, dev)), "A1": (1, aw0_plan(d, G, 3, 32, dev)),
                     "A2": (2, aw0_plan(d, G, 2, 32, dev)), "A0f": (0, aw0_plan(d, G, 2, 64, dev, half_le=0))}
            plans = {nm: v for nm, v in plans.items() if okc[v[0]]}
            P0 = plans["A0"][1]
            R = P0["R"]
            inter = torch.zeros((max(R, 1), 4096), dtype=torch.bfloat16, device=dev)
            top = sorted((P0["cnt"][e] for e in P0["experts"]), reverse=True)[:6]
            log(f"[aw0] bs={bs} r={r}: {len(P0['experts'])} experts (top {top}), {R} routes, W13 {P0['bytes'] / 1e6:.0f} MB, "
                f"{P0['macs'] / 1e9:.2f} GMAC; units/plan cost max:mean " +
                " ".join(f"{nm} {p['units']}/{p['cost_ratio']:.3f}" for nm, (_, p) in plans.items()) +
                f"; plans {time.time() - t0:.1f}s")

            def af(nm, mode=0, mk=False, d=d, inter=inter):
                cfg, P = plans[nm]
                def f():
                    mod.aw0_run(cfg, d["hidden_states"], st["ws13"], st["gsc13"], P["plan"], P["hu"], P["hr"], P["xp"],
                                inter, err, marks if mk else nomark, G, mode)
                return f
            rel = {}
            if r == 0:
                ref = aw0_ref(d, w, P0, dequant)
                for nm in plans:
                    if plans[nm][1]["R"] != R:
                        continue
                    err.zero_()
                    inter.zero_()
                    af(nm)()
                    torch.cuda.synchronize()
                    got = inter.float()
                    rel[nm] = float((got[:R] - ref[:R]).norm() / ref[:R].norm().clamp(min=1e-30))
                    if int(err[0]):
                        rel[nm] = float("inf")
                del ref
                log("[aw0]   check rel(inter): " + " ".join(f"{nm} {v:.2e}" for nm, v in rel.items()))
            err.zero_()
            af("A0")()
            torch.cuda.synchronize()
            if int(err[0]):
                log(f"[aw0]   error flag {err.tolist()}: timing skipped")
                rows.append({"bs": bs, "bad": True})
                err.zero_()
                continue
            arms = {"K": lambda d=d: k_full(d), "S": af("A0", mode=1)}
            arms.update({nm: af(nm) for nm in plans})
            graphs = {nm: timer.capture(fn) for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            del graphs
            try:
                g1 = timer.capture(lambda d=d: k1_only(d))
                us["K1"] = statistics.median(max(timer.time(g1) - ov, 1e-3) for _ in range(max(8, rounds // 2)))
                del g1
            except Exception as ex:  # noqa: BLE001
                log(f"[aw0]   K1 arm failed: {ex!r}")
                us["K1"] = float("nan")
            k_reset()
            ends = {}
            for nm in ("A0", "A1"):
                if nm not in plans:
                    continue
                nwg = AW0_CFGS[plans[nm][0]][0]
                marks.zero_()
                with torch.cuda.stream(timer.stream):
                    torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                    af(nm, mk=True)()
                torch.cuda.synchronize()
                mv = marks.view(G, 4).tolist()
                t00 = min(m[0] for m in mv)
                e_ = sorted((m[1 + c] - t00) / 1e3 for m in mv for c in range(nwg) if m[1 + c] > 0)
                ends[nm] = (statistics.median(e_), e_[-1]) if e_ else (float("nan"), float("nan"))
            e1 = [int(v) for v in err.tolist()]
            B = P0["bytes"]
            best = min((nm for nm in plans), key=lambda nm: us[nm])
            row = {"bs": bs, "us": us, "B": B, "macs": P0["macs"], "rel": rel, "err": e1, "best": best, "ends": ends}
            rows.append(row)
            log("[aw0]   us: " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items()) +
                f" | W13 TB/s: S {B / us['S'] / 1e6:.2f} " + " ".join(f"{nm} {B / us[nm] / 1e6:.2f}" for nm in plans) +
                f" | best {best} ({us[best] + 3.0:.1f} +3) / K1 {(us[best] + 3.0) / us['K1']:.3f} err {e1}")
            log("[aw0]   WG end p50/max: " + " ".join(f"{nm} {a:.1f}/{b:.1f}" for nm, (a, b) in ends.items()))
            err.zero_()
            del inter, plans, P0
    log("[aw0] GATE (R-hot-1, medians over routings, best of A0/A1/A2/A0f per routing): G (best+3)/K1 <= 0.75 at bs256 "
        "and <= 0.90 at bs128 -> GO; rel <= 1e-2 and err 0 required; bs32 is a diagnostic only")
    g = {}
    valid = True
    for bs in (256, 128, 32):
        rr = [x for x in rows if x["bs"] == bs and not x.get("bad")]
        if not rr:
            continue
        if any(x.get("bad") for x in rows if x["bs"] == bs):
            valid = False
        rels = [v for x in rr for v in x["rel"].values()]
        if any(v > 1e-2 for v in rels):
            valid = False
        g[bs] = statistics.median((x["us"][x["best"]] + 3.0) / x["us"]["K1"] for x in rr)
        per_arm = {nm: statistics.median(x["us"][nm] for x in rr if nm in x["us"]) for nm in rr[0]["us"]}
        log(f"[aw0]   bs={bs}: G {g[bs]:.3f} | medians " + " ".join(f"{nm} {v:.1f}" for nm, v in per_arm.items()) +
            f" | W13 TB/s best {statistics.median(x['B'] / x['us'][x['best']] / 1e6 for x in rr):.2f}, S "
            f"{statistics.median(x['B'] / x['us']['S'] / 1e6 for x in rr):.2f} | rel {['%.1e' % v for v in rels]}")
    if not valid or 256 not in g or 128 not in g:
        dec = "INVALID (fix, rerun under the same rule)"
    elif g[256] <= 0.75 and g[128] <= 0.90:
        dec = "GO (AW: build the W13 + W2 launches for bs128 / 256)"
    elif g[256] <= 0.75:
        dec = "PARTIAL (AW at bs256 only)"
    elif g[256] <= 0.85:
        dec = "REVISE (one revision from S / marks / arms)"
    else:
        dec = "KILL (AW structure; add trap)"
    log(f"[aw0] DECISION {dec}")
    del timer
    torch.cuda.empty_cache()
    return rows

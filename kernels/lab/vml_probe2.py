"""VML probe 2 (L1 step-0, notes/study/beyond_vjeux.md s3 L1): can the RS-wgmma consumer overlap tensor time with dequant
issue and the weight stream? Same structure as kernels/lab/vml_probe.py (1 CTA/SM, 4 consumer WGs sharing one 36 KB
weight stage + one x stage, 2 producer warps), with compile-time arms:
  PG_GS  k16 steps per commit group (2 = R-vml-1 / public, 4)
  PG_NP  wgmma groups kept in flight (wait_group NP; NP + 1 A-fragment buffers; 1 = R-vml-1, 3)
  PG_RES 0 stream from HBM, 1 resident (weights + x slot 0 loaded once, no ring waits: consumer-only time),
         2 resident + tensor only (constant A fragments, no LDS / dequant: wgmma time alone)
  PG_PP  1: WGs 2 / 3 start half a stage after WGs 0 / 1 (named-barrier handshake on the first stage)
Stage time = kernel time x G / (units x 16).
"""
import hashlib
import time

import torch

import vml_probe as V1

HEAD = V1.VP_CUDA[:V1.VP_CUDA.index("// ---------------- consumer ----------------")]

BODY = r"""
// ---------------- consumer (probe 2) ----------------
#ifndef PG_GS
#define PG_GS 2
#endif
#ifndef PG_NP
#define PG_NP 1
#endif
#ifndef PG_RES
#define PG_RES 0
#endif
#ifndef PG_PP
#define PG_PP 0
#endif
#ifndef PG_RG
#define PG_RG 0
#endif
#ifndef PG_RC
#define PG_RC 112
#endif
#ifndef PG_W2
#define PG_W2 0
#endif
#ifndef PG_W2NP
#define PG_W2NP 2
#endif
constexpr int NCW2 = PG_W2 ? 8 : NCW;   // W2: 2 consumer WGs x 128 rows (two m64 accumulator chains per WG)
constexpr int NTH2 = PG_W2 ? 32 * 8 + 64 : (PG_RG ? 32 * NCW + 128 : NTH);
constexpr int GS = PG_GS, NPD = PG_NP, NB = PG_NP + 1, GPS = 16 / PG_GS, RES = PG_RES;
static_assert(GPS % NB == 0, "groups per stage must be a multiple of the A-buffer count");
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }

__device__ __forceinline__ void dec_half(uint32_t w0, uint32_t w8, uint32_t Sg, uint32_t S8, uint32_t* hh) {
  deq4(w0, hh); deq4(w8, hh + 4);
#pragma unroll
  for (int j = 0; j < 4; ++j) { hh[j] = bmul2(hh[j], Sg); hh[4 + j] = bmul2(hh[4 + j], S8); }
}
template <int N, bool Z>
__device__ __forceinline__ void mma2(float* d, const uint32_t* hh, uint32_t dlo, uint32_t dhi) {
  if constexpr (Z) wgmma_zero<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  else wgmma_acc<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
  wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + 16, dhi);
}

template <int N>
__device__ __forceinline__ void stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
                                      uint32_t (&hb)[NB][4 * GS], int q, int wp, int lane, bool first, bool pp_arrive) {
  const int ws = RES ? 0 : it % NS, xs = RES ? 0 : it % NX;
  if constexpr (!RES) mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  const uint8_t* st = smem + ws * US + (q >> 1) * SW;
  const int tile = (q & 1) * 4 + wp;
  uint4 c[4];
  uint32_t sw0 = 0, sw1 = 0;
  if constexpr (RES < 2) {
    const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
#pragma unroll
    for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
    sw0 = sp[0]; sw1 = sp[32];
  }
  if constexpr (!RES) {
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
    mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
    __syncwarp();
  }
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);
  const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
  for (int gi = 0; gi < GPS; ++gi) {
    uint32_t* hh = hb[gi % NB];
    if constexpr (RES < 2) {
      const int i = (GS == 2) ? (gi >> 1) : gi;
      const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
      if constexpr (GS == 2) {
        const int hf = gi & 1;
        dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
      } else {
        dec_half(c[i].x, c[i].z, Sg, S8, hh);
        dec_half(c[i].y, c[i].w, Sg, S8, hh + 8);
      }
    } else {
#pragma unroll
      for (int k = 0; k < 4 * GS; ++k) hh[k] = 0x3C003C00u ^ (uint32_t)(lane + it + gi + k);
    }
    wg_fence();
#pragma unroll
    for (int h2 = 0; h2 < GS / 2; ++h2) {
      const uint32_t off = 16u * (uint32_t)(gi * GS + 2 * h2);
      if (first && gi == 0 && h2 == 0) mma2<N, true>(d, hh + 8 * h2, dlo + off, dhi);
      else mma2<N, false>(d, hh + 8 * h2, dlo + off, dhi);
    }
    wg_commit();
    wg_wait<NPD>();
    if constexpr (!RES) {
      if (gi == NPD - 1 && prevx >= 0) {   // the previous stage's last group retired: its x slot is free
        if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
        prevx = -1;
      }
    }
    if constexpr (PG_PP) {
      if (pp_arrive && gi == GPS / 2 - 1) bar_arrive_n(9, 512);
    }
  }
  if constexpr (!RES) prevx = xs;
  ++it;
}

template <int N>
__device__ __forceinline__ void unit_run(const Args& A, uint8_t* smem, uint32_t bars, int& it, int u, int q, int wp, int lane,
                                         bool firstunit) {
  float d[N / 2];
  uint32_t hb[NB][4 * GS];
  int prevx = -1;
  if constexpr (PG_PP) {
    if (firstunit && q >= 2) bar_sync_n(9, 512);
  }
  stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, PG_PP && firstunit && q < 2);
#pragma unroll 1
  for (int s = 1; s < 16; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, false);
  wg_wait<0>();
  if constexpr (!RES) {
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
  } else {
    if (d[0] == 1234.5f) A.err[3] = 1;
  }
}

// ---------------- W2: 2 consumer WGs, each 128 rows of the 256-row stage as two independent m64 chains ----------------
template <int N>
__device__ __forceinline__ void stage_w2(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* dA, float* dB,
                                         uint32_t (&hA)[2][8], uint32_t (&hB)[2][8], int q, int wp, int lane, bool first) {
  const int ws = RES ? 0 : it % NS, xs = RES ? 0 : it % NX;
  if constexpr (!RES) mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
  const uint8_t* st = smem + ws * US + q * SW;
  uint4 ca[4], cb[4];
  uint32_t sa0 = 0, sa1 = 0, sb0 = 0, sb1 = 0;
  if constexpr (RES < 2) {
    const uint4* pa = reinterpret_cast<const uint4*>(st + wp * 2048) + lane;
    const uint4* pb = reinterpret_cast<const uint4*>(st + (wp + 4) * 2048) + lane;
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(st + 16384 + wp * 256) + lane;
    const uint32_t* qb = reinterpret_cast<const uint32_t*>(st + 16384 + (wp + 4) * 256) + lane;
#pragma unroll
    for (int i = 0; i < 4; ++i) { ca[i] = pa[32 * i]; cb[i] = pb[32 * i]; }
    sa0 = qa[0]; sa1 = qa[32]; sb0 = qb[0]; sb1 = qb[32];
  }
  if constexpr (!RES) {
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
    mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
    __syncwarp();
  }
  const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);
  const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#pragma unroll
  for (int gi = 0; gi < 8; ++gi) {
    const int i = gi >> 1, hf = gi & 1, par = i & 1;
    const uint32_t off = 16u * (uint32_t)(2 * gi);
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      uint32_t* hh = ch ? hB[gi & 1] : hA[gi & 1];
      float* d = ch ? dB : dA;
      if constexpr (RES < 2) {
        const uint32_t sw2 = ch ? ((i < 2) ? sb0 : sb1) : ((i < 2) ? sa0 : sa1);
        const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
        const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
        const uint4 cc = ch ? cb[i] : ca[i];
        dec_half(hf ? cc.y : cc.x, hf ? cc.w : cc.z, Sg, S8, hh);
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) hh[k] = 0x3C003C00u ^ (uint32_t)(lane + it + gi + k + 8 * ch);
      }
      wg_fence();
      if (first && gi == 0) mma2<N, true>(d, hh, dlo + off, dhi);
      else mma2<N, false>(d, hh, dlo + off, dhi);
      wg_commit();
    }
    wg_wait<PG_W2NP>();
    if constexpr (!RES) {
      if (gi == 0 && prevx >= 0) {
        if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
        prevx = -1;
      }
    }
  }
  if constexpr (!RES) prevx = xs;
  ++it;
}
template <int N>
__device__ __forceinline__ void unit_run_w2(const Args& A, uint8_t* smem, uint32_t bars, int& it, int u, int q, int wp, int lane) {
  float dA[N / 2], dB[N / 2];
  uint32_t hA[2][8], hB[2][8];
  int prevx = -1;
  stage_w2<N>(A, smem, bars, it, prevx, dA, dB, hA, hB, q, wp, lane, true);
#pragma unroll 1
  for (int s = 1; s < 16; ++s) stage_w2<N>(A, smem, bars, it, prevx, dA, dB, hA, hB, q, wp, lane, false);
  wg_wait<0>();
  if constexpr (!RES) {
    if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
    const int2 un = A.unit[u];
    const int4 jb = A.job[un.x];
    const float gs = __ldg(A.gsc13 + jb.x);
    const int g = lane >> 2, t = lane & 3;
    const int rb = A.jbase[un.x];
#pragma unroll
    for (int ch = 0; ch < 2; ++ch) {
      const float* d = ch ? dB : dA;
      const int col = 8 * ((2 * un.y + q) * 8 + wp + 4 * ch) + g;
#pragma unroll
      for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int n = 8 * nt + 2 * t + hh;
          if (n < jb.y)
            A.inter[(size_t)(rb + n) * KDIM + col] = __float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
        }
    }
  } else {
    if (dA[0] == 1234.5f || dB[0] == 1234.5f) A.err[3] = 1;
  }
}

__global__ void __launch_bounds__(NTH2, 1) vp2_kernel(const Args A) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t bars = sa(smem + BAR_OFF);
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW2); }
    for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW2); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (A.marks) A.marks[(size_t)blockIdx.x * 8] = gtime();
  }
  __syncthreads();
  const int2 rg = A.cta[blockIdx.x];
  if (warp >= NCW2) {
    if constexpr (PG_RG && !PG_W2) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
      if (warp >= NCW + 2) return;
    }
    if (lane == 0) {
      if constexpr (RES) {   // one weight stage + one x stage, loaded once
        const int2 un = A.unit[rg.x];
        const int4 jb = A.job[un.x];
        if (warp == NCW2) {
          const uint8_t* wb = A.ws13 + ((size_t)jb.x * 64 + 2 * un.y) * 16 * SW;
          mbar_expect_tx(bars, US);
          bulk_g2s(sa(smem), wb, SW, bars);
          bulk_g2s(sa(smem) + SW, wb + (size_t)16 * SW, SW, bars);
        } else {
          const uint32_t bytes = (uint32_t)jb.z * 512u;
          mbar_expect_tx(bars + 8 * (2 * NS), bytes);
          bulk_g2s(sa(smem + NS * US), A.xp + (size_t)jb.w, bytes, bars + 8 * (2 * NS));
        }
      } else {
        if (warp == NCW2) prod_w(A, sa(smem), bars, rg.x, rg.y);
        else prod_x(A, sa(smem + NS * US), bars, rg.x, rg.y);
      }
    }
    return;
  }
  if constexpr (PG_RG && !PG_W2) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(PG_RC) : "memory");
  if constexpr (RES) {
    mbar_wait(bars, 0, A.err);
    mbar_wait(bars + 8 * (2 * NS), 0, A.err);
    __syncwarp();
  }
  const int q = warp >> 2, wp = warp & 3;
  int it = 0;
  if constexpr (PG_W2) {
    for (int u = rg.x; u < rg.y; ++u) {
      switch (A.job[A.unit[u].x].z >> 3) {
        case 1: unit_run_w2<8>(A, smem, bars, it, u, q, wp, lane); break;
        case 2: unit_run_w2<16>(A, smem, bars, it, u, q, wp, lane); break;
        case 4: unit_run_w2<32>(A, smem, bars, it, u, q, wp, lane); break;
        case 6: unit_run_w2<48>(A, smem, bars, it, u, q, wp, lane); break;
        default: unit_run_w2<64>(A, smem, bars, it, u, q, wp, lane); break;
      }
    }
  } else
  for (int u = rg.x; u < rg.y; ++u) {
    const int np = A.job[A.unit[u].x].z;
    const bool fu = u == rg.x;
    switch (np >> 3) {
      case 1: unit_run<8>(A, smem, bars, it, u, q, wp, lane, fu); break;
      case 2: unit_run<16>(A, smem, bars, it, u, q, wp, lane, fu); break;
      case 4: unit_run<32>(A, smem, bars, it, u, q, wp, lane, fu); break;
      case 6: unit_run<48>(A, smem, bars, it, u, q, wp, lane, fu); break;
      default: unit_run<64>(A, smem, bars, it, u, q, wp, lane, fu); break;
    }
  }
  if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 8 + 1 + q] = gtime();
}
}  // namespace vp

std::vector<int64_t> vp2_attrs() {
  cudaFuncSetAttribute(vp::vp2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vp::SMEM);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, vp::vp2_kernel);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vp::vp2_kernel, vp::NTH2, vp::SMEM);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vp::SMEM, occ};
}

void vp2_run(torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor job, torch::Tensor jbase, torch::Tensor unit,
             torch::Tensor cta, torch::Tensor xp, torch::Tensor inter, torch::Tensor err, torch::Tensor marks, int64_t G) {
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
  cudaFuncSetAttribute(vp::vp2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vp::SMEM);
  vp::vp2_kernel<<<(int)G, vp::NTH2, vp::SMEM, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
CUDA = HEAD + BODY
CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> vp2_attrs();
void vp2_run(torch::Tensor ws13, torch::Tensor gsc13, torch::Tensor job, torch::Tensor jbase, torch::Tensor unit,
             torch::Tensor cta, torch::Tensor xp, torch::Tensor inter, torch::Tensor err, torch::Tensor marks, int64_t G);
"""
ARMS = {"F": {}, "PP": {"PG_PP": 1}, "C": {"PG_RES": 1},
        "W2": {"PG_W2": 1, "PG_W2NP": 2}, "W2a": {"PG_W2": 1, "PG_W2NP": 1}}


def flags_of(arm):
    return V1.FLAGS + [f"-D{k}={v}" for k, v in ARMS[arm].items()]


def build(arm, verbose=False):
    from torch.utils.cpp_extension import load_inline
    fl = flags_of(arm)
    tag = hashlib.sha1((CPP + CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vml_probe2_{arm}_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["vp2_run", "vp2_attrs"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0

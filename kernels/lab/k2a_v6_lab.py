"""K2A v6 "slice-chained" MoE lab. LAB FILE: box runs only (tools/box/k2a_v6.py). Prototype: S = 128, T <= 16 (step 0 killed S = 32 / 64 at T = 32: FP32 red throughput).

Unit = (local expert e, 64-column intermediate slice sl): one CTA streams the unit's W13 row block (K2A's ws13 layout,
16 stages of 18,432 B: gate + up rows of the slice's 64 columns, the n_e tokens' X rows staged per stage), applies SwiGLU
in K2A's whole-item precision (FP32 accumulators x gs13 -> swiglu -> BF16) into shared memory, then streams the W2 slice
(8 stages of 512 rows x 64 k, ws2s layout below) and red.adds (x gs2) its FP32 partials into y[route][4096]. There is no
W13 -> W2 grid dependency and no K2 launch; v6_fin (PDL secondary, one CTA per token) applies K3's post-norm + weighted
combine and zeroes the y rows it read (y is all-zero between calls).

  * probe (step 0): structured stream of (e, slice) units through a bulk ring with null consumers, optional red.v4
    traffic with the real address pattern (mode bits: 1 weight copies, 2 reds, 4 W2 stage stagger, 8 dynamic tickets).
  * v6_main / v6_fin (step 1): the prototype. Source = K2A33_KERNEL_SRC's helpers + route<> from k3_v29.py (verbatim,
    extracted at build time) + V6_DEV. cold_item / hc_item inner loops are reused verbatim for the W13 phase.

ws2s layout (lossless repack of K2A's v2.2 W2 prepack): [32 e][64 sl][8 j][18432 B]; stage (sl, j) holds output rows
512 j .. 512 j + 511 at k = 64 sl .. 64 sl + 63: codes [warp 8][m 4][lane 32][16 B] = wq[tile 32 j + 4 warp + m][kb sl],
scales [warp 8][mp 2][lane 32][4 B], byte 2 (m & 1) + hi = sq[tile(m = 2 mp + (m & 1))][sl >> 1][lane][2 (sl & 1) + hi].
"""
from __future__ import annotations

import hashlib
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAYER = ROOT / "kernels/muse_decode_layer/k3_v29.py"
SW = 18432
CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]

# ---------------------------------------------------------------------------------------------------------------------
# step 0: structured stream + red probe (standalone source)
# ---------------------------------------------------------------------------------------------------------------------
PROBE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef V6_RED_PTX
#define V6_RED_PTX 0
#endif

namespace v6p {
constexpr int SW = 18432;
constexpr int KD = 4096;
constexpr unsigned FULL = 0xffffffffu;
constexpr long long WAIT_BUDGET = 1ll << 31;

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
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); return; }
  }
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
__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void red4(float* p, float a, float b, float c, float d) {
#if V6_RED_PTX
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
#else
  atomicAdd(reinterpret_cast<float4*>(p), make_float4(a, b, c, d));
#endif
}

struct PP {
  const uint8_t* w13; const uint8_t* w2;            // stage (e, sl, k): w13 + (e*1024 + sl*nW13 + k)*SW, w2 + (e*512 + sl*nW2 + k)*SW
  const int* ue; const int* usl;                    // per unit (in order): expert, slice
  const int* ncnt; const int* noff;                 // per expert: tokens, first route row
  float* y; int* err; int* tick; long long* marks;
  int U, S, mode;                                   // mode: 1 copies, 2 reds, 4 W2 stagger, 8 dynamic tickets
};

template <int NST>
__global__ void __launch_bounds__(288, 2) probe(const PP P) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * SW);
  __shared__ int s_desc[NST];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0 && P.marks != nullptr) P.marks[(size_t)blockIdx.x * 4] = gt();
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + i), 1); mbar_init(sa(bars + NST + i), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int nW13 = P.S / 4, nW2 = P.S / 8, NSU = nW13 + nW2;
  const bool dyn = (P.mode & 8) != 0;
  const int G = (int)gridDim.x;
  const int ubeg = (int)((long long)P.U * blockIdx.x / G), uend = (int)((long long)P.U * (blockIdx.x + 1) / G);
  const uint32_t ring_a = sa(smem), full_a = sa(bars), empty_a = sa(bars + NST);
  if (warp == 8) {
    if (lane != 0) return;
    const uint64_t pol = policy_evict_first();
    int it = 0;
    int u = dyn ? atomicAdd(P.tick, 1) : ubeg;
    while (dyn ? (u < P.U) : (u < uend)) {
      if (*reinterpret_cast<volatile int*>(P.err)) break;
      const int un = dyn ? atomicAdd(P.tick, 1) : u + 1;
      const int e = __ldg(P.ue + u), sl = __ldg(P.usl + u);
      for (int k = 0; k < NSU; ++k, ++it) {
        const int slot = it % NST;
        mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
        s_desc[slot] = u * 64 + k;
        const uint32_t fb = full_a + 8 * slot;
        if (P.mode & 1) {
          const uint8_t* src = (k < nW13) ? P.w13 + ((size_t)e * 1024 + (size_t)sl * nW13 + k) * SW
                                          : P.w2 + ((size_t)e * 512 + (size_t)sl * nW2 + (k - nW13)) * SW;
          mbar_expect_tx(fb, SW);
          bulk_g2s_ef(ring_a + slot * SW, src, SW, fb, pol);
        } else {
          mbar_arrive(fb);
        }
      }
      u = un;
    }
    const int slot = it % NST;
    mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
    s_desc[slot] = -1;
    mbar_arrive(full_a + 8 * slot);
    if (dyn) {
      const int old = atomicAdd(P.tick + 1, 1);
      if (old == G - 1) { atomicExch(P.tick, 0); atomicExch(P.tick + 1, 0); }
    }
    return;
  }
  if (warp > 8) return;
  const int g = lane >> 2, t = lane & 3, b = g & 3;
  const int tokl = 2 * t + (b & 1), rofs = 8 * (b >> 1) + 4 * (g >> 2);
  for (int it = 0;; ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    const int d = s_desc[slot];
    if (d < 0 || *reinterpret_cast<volatile int*>(P.err)) break;
    const int k = d & 63;
    if ((P.mode & 2) && k >= nW13) {
      const int u = d >> 6;
      const int e = __ldg(P.ue + u), sl = __ldg(P.usl + u);
      int j = k - nW13;
      if (P.mode & 4) j = (j + sl) % nW2;
      const int RPS = KD / nW2, TPW = RPS / 128;
      const int ne = __ldg(P.ncnt + e), off = __ldg(P.noff + e);
      for (int m = 0; m < TPW; ++m) {
        float* yr = P.y + (size_t)off * KD + j * RPS + 16 * (TPW * warp + m) + rofs;
        for (int q = 0; 8 * q < ne; ++q) {
          const int tok = 8 * q + tokl;
          if (tok < ne) red4(yr + (size_t)tok * KD, 1.f, 1.f, 1.f, 1.f);
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  if (tid == 0 && P.marks != nullptr) P.marks[(size_t)blockIdx.x * 4 + 2] = gt();
}
}  // namespace v6p

template <int NST>
static void probe_launch(const v6p::PP& p, int G) {
  constexpr int SM = NST * v6p::SW + 2 * NST * 8;
  cudaFuncSetAttribute(v6p::probe<NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
  cudaFuncSetAttribute(v6p::probe<NST>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  v6p::probe<NST><<<G, 288, SM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v6p_run(torch::Tensor w13, torch::Tensor w2, torch::Tensor ue, torch::Tensor usl, torch::Tensor ncnt,
             torch::Tensor noff, torch::Tensor y, torch::Tensor err, torch::Tensor tick, torch::Tensor marks,
             int64_t U, int64_t S, int64_t mode, int64_t G, int64_t nst) {
  TORCH_CHECK(S == 32 || S == 64 || S == 128, "v6p: S must be 32, 64 or 128");
  v6p::PP p;
  p.w13 = reinterpret_cast<const uint8_t*>(w13.data_ptr());
  p.w2 = reinterpret_cast<const uint8_t*>(w2.data_ptr());
  p.ue = ue.data_ptr<int>(); p.usl = usl.data_ptr<int>();
  p.ncnt = ncnt.data_ptr<int>(); p.noff = noff.data_ptr<int>();
  p.y = y.data_ptr<float>(); p.err = err.data_ptr<int>(); p.tick = tick.data_ptr<int>();
  p.marks = marks.numel() >= G * 4 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.U = (int)U; p.S = (int)S; p.mode = (int)mode;
  if (nst == 3) probe_launch<3>(p, (int)G);
  else if (nst == 4) probe_launch<4>(p, (int)G);
  else probe_launch<5>(p, (int)G);
}
"""
PROBE_CPP = ("#include <torch/extension.h>\n"
             "void v6p_run(torch::Tensor w13, torch::Tensor w2, torch::Tensor ue, torch::Tensor usl, torch::Tensor ncnt, "
             "torch::Tensor noff, torch::Tensor y, torch::Tensor err, torch::Tensor tick, torch::Tensor marks, int64_t U, "
             "int64_t S, int64_t mode, int64_t G, int64_t nst);\n")


def build_probe(name="k2a_v6_probe", red_ptx=0, verbose=False):
    from torch.utils.cpp_extension import load_inline
    src = PROBE_SRC.replace("#define V6_RED_PTX 0", f"#define V6_RED_PTX {int(red_ptx)}")
    tag = hashlib.sha1((PROBE_CPP + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=PROBE_CPP, cuda_sources=src, functions=["v6p_run"],
                      extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0


# ---------------------------------------------------------------------------------------------------------------------
# step 1: the prototype (device code goes into K2A33's namespace k3)
# ---------------------------------------------------------------------------------------------------------------------
V6_DEV = r"""
// ======================================================================================================================
// K2A v6 slice-chained (T <= 16; tools/k2a_v60_patches.py / kernels/lab/k2a_v6_lab.py). Unit = (expert e, 128-column
// intermediate slice sl): the CTA streams the unit's two W13 row blocks 2 sl, 2 sl + 1 (ws13: 32 contiguous stages, the
// n_e <= 16 tokens' X rows staged per stage) with v3.3's cold_item (n_e <= 8) / v3.6's hc_item<2> inner loops, applies
// K2A's whole-item SwiGLU (FP32 x gs13 -> swiglu -> BF16) into s_inter, then streams the W2 slice (ws2s: 16 stages of
// 256 rows x 128 k) and red.adds the FP32 partials (x gs2) into y[route][4096]: no W13 -> W2 grid dependency, no K2,
// one tail. v6_fin (PDL secondary, one CTA per token) applies K3's post-norm + weighted combine (per route: the norm is
// per (token, expert), so y stays per route) and re-zeroes the y rows it read (y is all-zero between calls).
// mode: bit 0 interleaved unit order (else expert-major, experts by token count desc), bit 1 dynamic unit tickets
// (tick[0] ticket, tick[1] exit count; the last CTA resets both), bit 2 W2 stage stagger (stage j -> (j + sl) & 15).
// Static: U <= G -> CTA c takes unit c (blockIdx 0..131 are one per SM first); U > G -> proportional ranges.
// ======================================================================================================================
constexpr int V6_MT = 16;                  // tokens served (T <= 16: every expert has <= 16 tokens -> NQ <= 2, one pass)
constexpr int V6_XS2 = 136;                // bf16 per s_inter row: 128 + 8 pad (conflict-free LDS.128)
constexpr int V6_NSL = 32;                 // slices per expert
constexpr int V6_STG = SW + V6_MT * XROW;  // ring slot: weight stage + up to 16 staged X rows

struct V6P {
  Params p;               // K2A's: x, ids, wts, ws13, gsc13, gsc2, nw, y (FP32 [8 M3][4096], zero between calls), meta,
                          // err, out, M, lo; hot_t = 32 (route<> bookkeeping only)
  const uint8_t* ws2s;    // [32 e][32 sl][16 j][SW]
  int* tick;
  long long* marks;       // optional [G][4]: entry, (unused), consumer end
  int mode;
};

__device__ __forceinline__ long long v6_gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void v6_red4(float* p, float a, float b, float c, float d) {
  atomicAdd(reinterpret_cast<float4*>(p), make_float4(a, b, c, d));
}
__device__ __forceinline__ bool v6_err(const Params& P) { return *reinterpret_cast<volatile int*>(P.err) != 0; }
__device__ __forceinline__ void v6_unit(int u, int nact, int mode, int& rank, int& sl) {
  if (mode & 1) { rank = u % nact; sl = u / nact; } else { rank = u / V6_NSL; sl = u - (u / V6_NSL) * V6_NSL; }
}

// One kb of a W2-slice tile: consume_kb's decode and mma order with B = s_inter rows (xb = s_inter + g * XS2 + 16 t + 64 kb).
template <int NQ>
__device__ __forceinline__ void v6_kb2(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xb, float (*acc)[4]) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
#pragma unroll
  for (int q = 0; q < NQ; ++q) {
    const uint4* xp = reinterpret_cast<const uint4*>(xb + q * 8 * V6_XS2);
    const uint4 x0 = xp[0], x1 = xp[1];
    mma16816(acc[q], ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
    mma16816(acc[q], ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
    mma16816(acc[q], ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
    mma16816(acc[q], ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
  }
}

// 4x4 transpose across lanes b = g & 3 (lane ^ 4, lane ^ 8) of a C fragment v = (row g, tok 2t), (g, 2t+1), (g+8, 2t),
// (g+8, 2t+1) -> z[s] = rows 4 (g >> 2) + 8 bh + s (s = 0..3) of token 2t + bo (bo = b & 1, bh = b >> 1).
__device__ __forceinline__ void v6_t4(float v0, float v1, float v2, float v3, bool bo, bool bh,
                                      float& z0, float& z1, float& z2, float& z3) {
  const float s0 = bo ? v0 : v1, s1 = bo ? v2 : v3;
  const float r0 = __shfl_xor_sync(FULL, s0, 4), r1 = __shfl_xor_sync(FULL, s1, 4);
  const float y0 = bo ? r0 : v0, y1 = bo ? v1 : r0, y2 = bo ? r1 : v2, y3 = bo ? v3 : r1;
  const float t0 = bh ? y0 : y2, t1 = bh ? y1 : y3;
  const float q0 = __shfl_xor_sync(FULL, t0, 8), q1 = __shfl_xor_sync(FULL, t1, 8);
  z0 = bh ? q0 : y0; z1 = bh ? q1 : y1; z2 = bh ? y2 : q0; z3 = bh ? y3 : q1;
}

// 16 W13 stages of one row block (v3.3 cold_item / v3.6 hc_item consumer loops, verbatim), accumulators returned.
template <int NQ, int NST>
__device__ __forceinline__ void v6_w13(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a, int& it,
                                       float (*acc)[4]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int q = 0; q < NQ; ++q)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[q][i] = 0.f;
  if constexpr (NQ == 1) {
    float acc2[1][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc2[0][i] = 0.f;
    for (int s = 0; s < 16; ++s, ++it) {
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
      const uint8_t* st = ring + slot * V6_STG;
      const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
      const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;
      uint4 wv[4];
      uint32_t sv[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
      sv[0] = sp[0]; sv[1] = sp[32];
#pragma unroll
      for (int i = 0; i < 4; ++i) consume_kb<1>(wv[i], sv[i >> 1], i & 1, xrow, i * 64, 1, t, (i < 2) ? acc : acc2);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[0][i] += acc2[0][i];
  } else {
    for (int s = 0; s < 16; ++s, ++it) {
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
      const uint8_t* st = ring + slot * V6_STG;
      const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
      const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST + 16 * t;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 w = wp[i * 32];
        const uint32_t sw2 = sp[(i >> 1) * 32];
        const int par = i & 1;
        const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
        const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          uint32_t a[4], b[4];
          deq4(hf ? w.y : w.x, a);
          deq4(hf ? w.w : w.z, b);
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) { a[jj] = bmul2(a[jj], Sg); b[jj] = bmul2(b[jj], S8); }
#pragma unroll
          for (int q = 0; q < NQ; ++q) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xrow + q * 8 * XST + i * 64 + 8 * hf);
            mma16816(acc[q], a[0], b[0], a[1], b[1], xv.x, xv.y);
            mma16816(acc[q], a[2], b[2], a[3], b[3], xv.z, xv.w);
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
  }
}

// Consumers' work on one unit (48 ring stages: 2 x 16 W13 + 16 W2).
template <int NQ, int NST>
__device__ __forceinline__ void v6_body(const V6P& Q, const uint8_t* ring, uint32_t full_a, uint32_t empty_a, int& it_ref,
                                        int e, int sl, int n, int c0, __nv_bfloat16* s_inter) {
  const Params& P = Q.p;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  const float gs1 = __ldg(P.gsc13 + e);
#pragma unroll 1
  for (int h = 0; h < 2; ++h) {   // row block 2 sl + h -> s_inter columns 64 h .. 64 h + 63
    float acc[NQ][4];
    v6_w13<NQ, NST>(P, ring, full_a, empty_a, it, acc);
    if (h == 0) asm volatile("bar.sync 1, 256;\n" ::: "memory");   // the previous unit's W2 reads of s_inter are done
#pragma unroll
    for (int q = 0; q < NQ; ++q)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int nl = q * 8 + 2 * t + hh;
        if (nl < n)
          s_inter[nl * V6_XS2 + 64 * h + 8 * warp + g] = __float2bfloat16_rn(swiglu(acc[q][hh] * gs1, acc[q][2 + hh] * gs1));
      }
  }
  asm volatile("bar.sync 1, 256;\n" ::: "memory");
  // ---------------- W2 slice: 16 stages of 256 rows x 128 k -> red.v4 into y rows c0 .. c0 + n - 1 ----------------
  {
    const float gs = __ldg(P.gsc2 + e);
    const __nv_bfloat16* xb = s_inter + g * V6_XS2 + 16 * t;
    const bool bo = ((lane >> 2) & 1) != 0, bh = ((lane >> 3) & 1) != 0;
    const int tl = 2 * t + (bo ? 1 : 0);
    float* yb = P.y + (size_t)c0 * KDIM + 32 * warp + (bh ? 8 : 0) + 4 * (lane >> 4);
    for (int j = 0; j < 16; ++j, ++it) {
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
      const uint8_t* st = ring + slot * V6_STG;
      const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
      uint4 wv[4];
      uint32_t sv[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
      sv[0] = sp[0]; sv[1] = sp[32];
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);   // the stage is in registers: release the slot early
      const int jj = (Q.mode & 4) ? ((j + sl) & 15) : j;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        float c[NQ][4];
#pragma unroll
        for (int q = 0; q < NQ; ++q)
#pragma unroll
          for (int i = 0; i < 4; ++i) c[q][i] = 0.f;
        v6_kb2<NQ>(wv[2 * m], sv[m], 0, xb, c);
        v6_kb2<NQ>(wv[2 * m + 1], sv[m], 1, xb + 64, c);
        float* yr = yb + 256 * jj + 16 * m;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
          float z0, z1, z2, z3;
          v6_t4(c[q][0] * gs, c[q][1] * gs, c[q][2] * gs, c[q][3] * gs, bo, bh, z0, z1, z2, z3);
          const int tk = 8 * q + tl;
          if (tk < n) v6_red4(yr + (size_t)tk * KDIM, z0, z1, z2, z3);
        }
      }
    }
  }
  it_ref = it;
}

template <int NST>
__global__ void __launch_bounds__(288, 2) v6_main(const V6P Q) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int16_t s_list[8 * V6_MT];
  __shared__ Route S;
  __shared__ int s_eord[32];
  __shared__ int s_nact;
  __shared__ int s_desc[NST];
  __shared__ __align__(16) __nv_bfloat16 s_inter[V6_MT * V6_XS2];
  const Params& P = Q.p;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * V6_STG);
  if (tid == 0) {
    if (Q.marks != nullptr) Q.marks[(size_t)blockIdx.x * 4] = v6_gt();
    for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + NST + i), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  for (int i = tid; i < V6_MT * V6_XS2 / 2; i += blockDim.x) reinterpret_cast<uint32_t*>(s_inter)[i] = 0u;
  pdl_wait();   // in-layer: PDL secondary of the routing tail (ids / weights / x); no global reads or writes before
  route<1, 32, true>(P, S, nullptr, s_list, 64, blockIdx.x == 0);
  if (warp == 0) {   // active experts by token count (desc), ties by index
    const int c = S.cnt[lane];
    int rank = 0;
#pragma unroll 8
    for (int k = 0; k < 32; ++k) {
      const int ck = __shfl_sync(FULL, c, k);
      rank += (ck > c || (ck == c && k < lane)) ? 1 : 0;
    }
    const unsigned act = __ballot_sync(FULL, c > 0);
    if (c > 0) s_eord[rank] = lane;
    if (lane == 0) s_nact = __popc(act);
  }
  __syncthreads();
  const int nact = s_nact, U = nact * V6_NSL;
  const int G = (int)gridDim.x, c = (int)blockIdx.x;
  int ubeg, uend;
  if (U <= G) { ubeg = c; uend = (c < U) ? c + 1 : c; }
  else { ubeg = (int)((long long)U * c / G); uend = (int)((long long)U * (c + 1) / G); }
  const bool dyn = (Q.mode & 2) != 0;
  const uint32_t ring_a = sa(smem), full_a = sa(bars), empty_a = sa(bars + NST);
  if (warp == 8) {
    // ---------------- producer: per unit 32 W13 stages (+ X rows) then 16 W2 stages; lane (r + 1) & 31 copies X row r ----
    const uint64_t pol = policy_evict_first();
    const int r = (lane + 31) & 31;
    int it = 0, u = ubeg;
    if (dyn) { int t0 = 0; if (lane == 0) t0 = atomicAdd(Q.tick, 1); u = __shfl_sync(FULL, t0, 0); }
    for (;;) {
      if (dyn ? (u >= U) : (u >= uend)) break;
      if (v6_err(P)) break;
      int un = u + 1;
      if (dyn) { int t1 = 0; if (lane == 0) t1 = atomicAdd(Q.tick, 1); un = __shfl_sync(FULL, t1, 0); }
      int rank, sl;
      v6_unit(u, nact, Q.mode, rank, sl);
      const int e = s_eord[rank], n = S.cnt[e], c0 = S.off[e];
      int tok = 0;
      if (r < n) tok = ((int)s_list[c0 + r]) >> 3;
      const uint8_t* w13 = P.ws13 + ((size_t)(e * 64 + 2 * sl) * 16) * SW;   // row blocks 2 sl, 2 sl + 1: contiguous
      const uint8_t* w2 = Q.ws2s + ((size_t)(e * V6_NSL + sl) * 16) * SW;
      const uint32_t xbytes = (uint32_t)n * 512u;
      for (int s = 0; s < 48; ++s, ++it) {
        const int slot = it % NST;
        mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
        const uint32_t dst = ring_a + slot * V6_STG, fb = full_a + 8 * slot;
        if (lane == 0) {
          s_desc[slot] = u;
          if (s < 32) {
            mbar_expect_tx(fb, SW + xbytes);
            bulk_g2s_ef(dst, w13 + (size_t)s * SW, SW, fb, pol);
          } else {
            const int j = s - 32, jj = (Q.mode & 4) ? ((j + sl) & 15) : j;
            mbar_expect_tx(fb, SW);
            bulk_g2s_ef(dst, w2 + (size_t)jj * SW, SW, fb, pol);
          }
        }
        mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals
        __syncwarp();
        if (s < 32 && r < n) bulk_g2s(dst + SW + r * XROW, P.x + (size_t)tok * KDIM + (s & 15) * 256, 512, fb);
      }
      u = un;
    }
    if (dyn) {   // end marker for the consumers (33 arrivals, no bytes), then the ticket reset by the last CTA
      const int slot = it % NST;
      mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
      if (lane == 0) { s_desc[slot] = -1; mbar_arrive(full_a + 8 * slot); }
      __syncwarp();
      mbar_arrive(full_a + 8 * slot);
      if (lane == 0) {
        const int old = atomicAdd(Q.tick + 1, 1);
        if (old == G - 1) { atomicExch(Q.tick, 0); atomicExch(Q.tick + 1, 0); }
      }
    }
    return;
  }
  // ---------------- consumers (warps 0-7; warp w = tile w of a W13 row block, tiles 2w, 2w+1 of a W2 stage) ----------------
  int it = 0, u = ubeg;
  for (;;) {
    int uu;
    if (dyn) {
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
      uu = s_desc[slot];
      if (uu < 0) break;
    } else {
      if (u >= uend) break;
      uu = u++;
    }
    if (v6_err(P)) break;
    int rank, sl;
    v6_unit(uu, nact, Q.mode, rank, sl);
    const int e = s_eord[rank], n = S.cnt[e], c0 = S.off[e];
    if (n <= 8) v6_body<1, NST>(Q, smem, full_a, empty_a, it, e, sl, n, c0, s_inter);
    else v6_body<2, NST>(Q, smem, full_a, empty_a, it, e, sl, n, c0, s_inter);
  }
  if (tid == 0 && Q.marks != nullptr) Q.marks[(size_t)blockIdx.x * 4 + 2] = v6_gt();
}

// Finalize (PDL secondary of v6_main): one CTA per token, thread i = columns 8i .. 8i+7. K3's formulas: per route
// ss = sum (bf16(y) * w)^2, r = rsqrt(ss / 4096 + 1e-8) * topk weight, out = sum_s bf16(y_s) * w * r_s (slot order).
__global__ void __launch_bounds__(512) v6_fin(const V6P Q) {
  const Params& P = Q.p;
  __shared__ int s_c[8];
  __shared__ float s_w[8], s_r[8];
  __shared__ float s_red[16][8];
  pdl_wait();
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < 8) { s_c[tid] = __ldcg(P.meta + MH + tok * 8 + tid); s_w[tid] = __ldg(P.wts + tok * 8 + tid); }
  __syncthreads();
  const int col = 8 * tid;
  const float4 w0 = __ldg(reinterpret_cast<const float4*>(P.nw + col)), w1 = __ldg(reinterpret_cast<const float4*>(P.nw + col + 4));
  float4 v[8][2];
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int c = s_c[s];
    const float4* yp = reinterpret_cast<const float4*>(P.y + (size_t)(c < 0 ? 0 : c) * KDIM + col);
    v[s][0] = (c >= 0) ? __ldcg(yp) : z4;
    v[s][1] = (c >= 0) ? __ldcg(yp + 1) : z4;
  }
  float pz[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float z, a = 0.f;
    z = bf16r(v[s][0].x) * w0.x; a += z * z;  z = bf16r(v[s][0].y) * w0.y; a += z * z;
    z = bf16r(v[s][0].z) * w0.z; a += z * z;  z = bf16r(v[s][0].w) * w0.w; a += z * z;
    z = bf16r(v[s][1].x) * w1.x; a += z * z;  z = bf16r(v[s][1].y) * w1.y; a += z * z;
    z = bf16r(v[s][1].z) * w1.z; a += z * z;  z = bf16r(v[s][1].w) * w1.w; a += z * z;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(FULL, a, o);
    pz[s] = a;
  }
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < 8; ++s) s_red[warp][s] = pz[s];
  }
  __syncthreads();
  if (tid < 8) {
    float ss = 0.f;
#pragma unroll
    for (int w = 0; w < 16; ++w) ss += s_red[w][tid];
    s_r[tid] = (s_c[tid] >= 0) ? rsqrtf(ss * (1.f / 4096.f) + 1e-8f) * s_w[tid] : 0.f;
  }
  __syncthreads();
  float a[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) a[k] = 0.f;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const float r = s_r[s];
    a[0] += bf16r(v[s][0].x) * w0.x * r; a[1] += bf16r(v[s][0].y) * w0.y * r;
    a[2] += bf16r(v[s][0].z) * w0.z * r; a[3] += bf16r(v[s][0].w) * w0.w * r;
    a[4] += bf16r(v[s][1].x) * w1.x * r; a[5] += bf16r(v[s][1].y) * w1.y * r;
    a[6] += bf16r(v[s][1].z) * w1.z * r; a[7] += bf16r(v[s][1].w) * w1.w * r;
  }
  uint4 o;
  o.x = pack_bf2(a[0], a[1]); o.y = pack_bf2(a[2], a[3]); o.z = pack_bf2(a[4], a[5]); o.w = pack_bf2(a[6], a[7]);
  *reinterpret_cast<uint4*>(P.out + (size_t)tok * KDIM + col) = o;
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int c = s_c[s];
    if (c >= 0) {
      float4* yp = reinterpret_cast<float4*>(P.y + (size_t)c * KDIM + col);
      __stcg(yp, z4);
      __stcg(yp + 1, z4);
    }
  }
}
"""

V6_HOST = r"""
template <int NST>
static constexpr int v6_smem() { return NST * k3::V6_STG + 2 * NST * 8; }

template <int NST>
static void v6_launch(const k3::V6P& q, int G, int M, int flags) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = v6_smem<NST>();
  cudaFuncSetAttribute(k3::v6_main<NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
  cudaFuncSetAttribute(k3::v6_main<NST>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = flags & 1;   // in-layer: main kernel after the routing tail
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cfg.gridDim = dim3(G); cfg.blockDim = dim3(288); cfg.dynamicSmemBytes = SM;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::v6_main<NST>, q));
  if (!(flags & 2)) {
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = dim3(M); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::v6_fin, q));
  }
}

void v6_run(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2s,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor y, torch::Tensor meta,
            torch::Tensor err, torch::Tensor out, torch::Tensor tick, torch::Tensor marks,
            int64_t M, int64_t lo, int64_t G, int64_t mode, int64_t nst, int64_t flags) {
  TORCH_CHECK(M >= 1 && M <= k3::V6_MT, "k2a v6: 1 <= M <= 16");
  k3::V6P q{};
  k3::Params& p = q.p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ids = ids.data_ptr<int>();
  p.wts = wts.data_ptr<float>();
  p.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  p.gsc13 = gsc13.data_ptr<float>();
  p.gsc2 = gsc2.data_ptr<float>();
  p.nw = nw.data_ptr<float>();
  p.y = y.data_ptr<float>();
  p.meta = meta.data_ptr<int>();
  p.err = err.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.hot_t = 32; p.hot_w4 = 8; p.hot_mode = 0;
  p.M = (int)M; p.lo = (int)lo; p.probe = 0;
  q.ws2s = reinterpret_cast<const uint8_t*>(ws2s.data_ptr());
  q.tick = tick.data_ptr<int>();
  q.marks = marks.numel() >= G * 4 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  q.mode = (int)mode;
  if (nst == 4) v6_launch<4>(q, (int)G, (int)M, (int)flags);
  else v6_launch<3>(q, (int)G, (int)M, (int)flags);
}

std::vector<int64_t> v6_attrs(int64_t nst) {
  std::vector<int64_t> r;
  auto one = [&](const void* f, int nt, int sm) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, sm) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + sm), (int64_t)occ});
  };
  if (nst == 4) one((const void*)k3::v6_main<4>, 288, v6_smem<4>());
  else one((const void*)k3::v6_main<3>, 288, v6_smem<3>());
  one((const void*)k3::v6_fin, 512, 0);
  return r;
}
"""
V6_CPP = ("#include <torch/extension.h>\n#include <vector>\n"
          "void v6_run(torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2s, "
          "torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor y, torch::Tensor meta, torch::Tensor err, "
          "torch::Tensor out, torch::Tensor tick, torch::Tensor marks, int64_t M, int64_t lo, int64_t G, int64_t mode, "
          "int64_t nst, int64_t flags);\n"
          "std::vector<int64_t> v6_attrs(int64_t nst);\n")


def k2a33_head(layer=LAYER):
    """K2A33_KERNEL_SRC of the layer file up to (excluding) the lab-probe consumer: includes, namespace k3 {, constants,
    Params, Route, decode / mma / mbarrier / bulk / PDL helpers, route<>, my_range*, tail_first, consume_kb (verbatim)."""
    text = pathlib.Path(layer).read_text(encoding="utf-8")
    key = 'K2A33_KERNEL_SRC = r"""'
    if text.count(key) != 1:
        raise RuntimeError("k2a_v6: K2A33_KERNEL_SRC not found once")
    a = text.index(key) + len(key)
    src = text[a:text.index('"""', a)]
    cut = "// Lab probes: B operand from registers (no X smem reads) / decode only."
    if src.count(cut) != 1:
        raise RuntimeError("k2a_v6: K2A33 cut marker not found once")
    return src[:src.index(cut)]


def proto_source(layer=LAYER):
    return k2a33_head(layer) + V6_DEV + "\n}  // namespace k3\n" + V6_HOST


def build_proto(name="k2a_v6_proto", verbose=False):
    from torch.utils.cpp_extension import load_inline
    src = proto_source()
    tag = hashlib.sha1((V6_CPP + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=V6_CPP, cuda_sources=src, functions=["v6_run", "v6_attrs"],
                      extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0


def w2_slices(wq, sq):
    """K2A v2.2 W2 prepack (wq int32 [256 tiles, 64 kb, 32, 4], sq uint8 [256, 32 kp, 32, 4]) -> uint8 [32 sl, 16 j, 18432]:
    stage (sl, j) = output rows 256 j .. 256 j + 255 at k = 128 sl .. 128 sl + 127; codes [warp 8][m 2][kbl 2][lane][16 B] =
    wq[tile 16 j + 2 warp + m][kb 2 sl + kbl], scales [warp 8][m 2][lane][4 B] = sq[tile][kp sl][lane] (bytes 2 kbl + hi).
    Lossless: a permutation of the same bytes."""
    import torch
    assert tuple(wq.shape) == (256, 64, 32, 4) and tuple(sq.shape) == (256, 32, 32, 4), (wq.shape, sq.shape)
    wb = wq.contiguous().view(torch.uint8).view(16, 8, 2, 32, 2, 512).permute(3, 0, 1, 2, 4, 5).reshape(32, 16, 16384)
    sb = sq.contiguous().view(16, 8, 2, 32, 32, 4).permute(3, 0, 1, 2, 4, 5).reshape(32, 16, 2048)
    return torch.cat((wb, sb), dim=-1).contiguous()


def w2_slices_check(wq, sq, ws2s_e):
    """Inverse spot check of w2_slices on a few (sl, j, warp, m, kbl) positions (any device)."""
    import torch
    wb = wq.contiguous().view(torch.uint8).view(256, 64, 512)
    for sl, j, w, m, kbl in ((0, 0, 0, 0, 0), (31, 15, 7, 1, 1), (17, 3, 5, 0, 1), (9, 12, 1, 1, 0)):
        tile = 16 * j + 2 * w + m
        o = w * 2048 + m * 1024 + kbl * 512
        if not torch.equal(ws2s_e[sl, j, o:o + 512], wb[tile, 2 * sl + kbl]):
            return False
        for lane in (0, 13, 31):
            o2 = 16384 + w * 256 + m * 128 + lane * 4
            if not torch.equal(ws2s_e[sl, j, o2:o2 + 4], sq[tile, sl, lane]):
                return False
    return True


if __name__ == "__main__":
    import sys
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "k2a_v6_proto_gen.cu")
    out.write_text(proto_source(), encoding="utf-8")
    print(f"wrote {out}")

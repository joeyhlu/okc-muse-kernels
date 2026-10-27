"""vg_bnd_lab.py: VG-B layer-boundary kernel lab (VG glue builder, T >= 32), notes/study/vg_glue.md.

The layer boundary (post-norm + gated residual + pre-norm) at T = 32 / 128 / 256 after the real predecessor GEMM
(o-projection for the attention boundary, unpooler for the FFN boundary; cuBLAS like the layer at T >= 128), timed
judge-style (CUDA graph replay after a 256 MB L2 flush) as the increment over the GEMM alone, for:
  sgl          SGLang's FusedPostNormResidualPreNorm module (the layer's path at T >= 128)
  sgl_c<C>     SGLang fused_norm_residual cluster kernel, C CTAs per row (the layer's T <= 32 FFN boundary is C = 16)
  sgl_t<N><p>  SGLang single-CTA kernel with N threads, weight prefetch p (1) or not (0)
  vgb_<N>_<C>_<m><s>  VG-B (below): N threads x C CTAs per row, m = 'm' one 4-moment reduction (SBB1's arithmetic) or
               's' two sequential reductions (SGLang's arithmetic), s = 'p' stream loaded before griddepcontrol.wait
               (attention boundary only: the stream is a layer input) or 'n' after it.
VG-B loads every predecessor-independent input (alpha, beta, post weight, and the stream where legal) before the PDL
wait, keeps v and alpha*s in registers, and reloads beta / pre weight after the reduction (L1/L2-hot) to stay at
64 registers (2 CTAs/SM at 512 threads: one wave at T = 256).

    python3 -u kernels/lab/vg_bnd_lab.py [--ts=32,128,256] [--rounds=30] [--arms=...] [--out=box_out/vg2/bnd.json]
    python3 -u kernels/lab/vg_bnd_lab.py --local     # RTX 3070: correctness of the C = 1 builds (no clusters / PDL)
"""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = 8192

VGB_CU = r'''
#include <cuda_bf16.h>
#include <stdint.h>
#define VG_H 8192
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define VG_SM90 1
#else
#define VG_SM90 0
#endif

struct VgbParams {
  const __nv_bfloat16* branch;   // [T][H] bf16 (predecessor output)
  const float* stream;           // [T][H] fp32
  __nv_bfloat16* out;            // [T][H] bf16: pre-norm of the new stream
  float* new_stream;             // [T][H] fp32
  const __nv_bfloat16* post_w;   // [H] bf16 (post-norm input scale) or nullptr
  const __nv_bfloat16* pre_w;    // [H] bf16 or nullptr
  const float* alpha;            // [H] fp32 residual gate
  const float* beta;             // [H] fp32
  float post_eps, pre_eps;
};

__device__ __forceinline__ float vg_rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void vg_ld8bf(const __nv_bfloat16* p, float (&v)[8]) {
  const uint4 q = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) { v[2 * i] = __uint_as_float(w[i] << 16); v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u); }
}
__device__ __forceinline__ void vg_ld8f(const float* p, float (&v)[8]) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void vg_st8f(float* p, const float (&v)[8]) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ void vg_pdl_wait() {
#if VG_SM90
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void vg_pdl_trigger() {
#if VG_SM90
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Sum of 4 values over the CTA (and over the C-CTA cluster through DSMEM); every thread gets the totals.
template <int NT, int C>
__device__ __forceinline__ float4 vg_sum4(float4 m, float (*wred)[4], float (*part)[4], uint32_t bar_a, uint32_t rank) {
  constexpr int NW = NT / 32;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    m.x += __shfl_xor_sync(0xffffffffu, m.x, o);
    m.y += __shfl_xor_sync(0xffffffffu, m.y, o);
    m.z += __shfl_xor_sync(0xffffffffu, m.z, o);
    m.w += __shfl_xor_sync(0xffffffffu, m.w, o);
  }
  if (lane == 0) { wred[warp][0] = m.x; wred[warp][1] = m.y; wred[warp][2] = m.z; wred[warp][3] = m.w; }
  __syncthreads();
  float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int w = 0; w < NW; ++w) { t.x += wred[w][0]; t.y += wred[w][1]; t.z += wred[w][2]; t.w += wred[w][3]; }
  if constexpr (C > 1) {
#if VG_SM90
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    const int tid = threadIdx.x;
    if (tid < C) {
      if (tid == (int)rank) {
        *reinterpret_cast<float4*>(&part[rank][0]) = t;
      } else {
        const uint32_t dst_l = (uint32_t)__cvta_generic_to_shared(&part[rank][0]);
        uint32_t dst, rbar;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dst) : "r"(dst_l), "r"(tid));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(bar_a), "r"(tid));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                     ::"r"(dst), "f"(t.x), "f"(t.y), "f"(t.z), "f"(t.w), "r"(rbar) : "memory");
      }
    }
    uint32_t done = 0;
    while (!done)
      asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                   " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bar_a) : "memory");
    __syncthreads();
    t = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int q = 0; q < C; ++q) { t.x += part[q][0]; t.y += part[q][1]; t.z += part[q][2]; t.w += part[q][3]; }
#endif
  }
  return t;
}

// One boundary row per C-CTA cluster; thread tid of cluster rank r owns 8-element vectors k = 0..K-1 at columns
// ((r * K + k) * NT + tid) * 8 (coalesced per k). MOM: one 4-moment reduction (SBB1's arithmetic: the pre-norm sum
// from sum v^2, sum (a s)^2, sum (a s)(b v), sum (b v)^2); else two sequential reductions (SGLang's arithmetic).
// PRE: the stream is loaded before griddepcontrol.wait (legal only where it is not written by the predecessor chain).
// PW: post weight applied as an input scale before the post-norm (the FFN boundary); else no post weight.
template <int NT, int C, bool MOM, bool PRE, bool PW>
__global__ void __launch_bounds__(NT, (VG_H / (C * NT * 8) >= 4) ? ((512 / NT) > 0 ? 512 / NT : 1) : ((1024 / NT) > 0 ? 1024 / NT : 1))
vgb_kernel(const VgbParams p) {
  constexpr int K = VG_H / (C * NT * 8);
  static_assert(K >= 1 && VG_H % (C * NT * 8) == 0, "vgb geometry");
  static_assert(MOM || C == 1, "sequential reductions: one CTA per row");
  __shared__ float wred[2][NT / 32][4];
  __shared__ __align__(16) float part[C][4];
  __shared__ __align__(8) uint64_t bar;
  const int tid = threadIdx.x;
  uint32_t rank = 0, bar_a = 0;
  if constexpr (C > 1) {
#if VG_SM90
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    bar_a = (uint32_t)__cvta_generic_to_shared(&bar);
    if (tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar_a));
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_a), "r"((C - 1) * 16));
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
#endif
  }
  const size_t rb = (size_t)(blockIdx.x / C) * VG_H;
  float v[K][8], r[K][8];
  constexpr bool EARLY = (K == 1);   // constants before the wait only where registers allow (K = 1: <= 64 regs, no spill)
  float al[EARLY ? K : 1][8], pw[(PW && EARLY) ? K : 1][8];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int e = (((int)rank * K + k) * NT + tid) * 8;
    if constexpr (EARLY) {
      vg_ld8f(p.alpha + e, al[k]);
      if constexpr (PW) vg_ld8bf(p.post_w + e, pw[k]);
    }
    if constexpr (PRE) vg_ld8f(p.stream + rb + e, r[k]);
  }
  vg_pdl_wait();
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int e = (((int)rank * K + k) * NT + tid) * 8;
    vg_ld8bf(p.branch + rb + e, v[k]);
    if constexpr (!PRE) vg_ld8f(p.stream + rb + e, r[k]);
  }
  vg_pdl_trigger();
  if constexpr (MOM) {
    float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
    float bt[K][8];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int e = (((int)rank * K + k) * NT + tid) * 8;
      vg_ld8f(p.beta + e, bt[k]);
      float al_[8], pw_[8];
      if constexpr (EARLY) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { al_[i] = al[0][i]; pw_[i] = (PW ? pw[0][i] : 1.f); }
      } else {
        vg_ld8f(p.alpha + e, al_);
        if constexpr (PW) vg_ld8bf(p.post_w + e, pw_);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float x = v[k][i];
        if constexpr (PW) x = vg_rbf(x * pw_[i]);
        v[k][i] = x;
        const float a = al_[i] * r[k][i];
        r[k][i] = a;
        const float b = bt[k][i] * x;
        m.x = fmaf(x, x, m.x);
        m.y = fmaf(a, a, m.y);
        m.z = fmaf(a, b, m.z);
        m.w = fmaf(b, b, m.w);
      }
    }
    const float4 t = vg_sum4<NT, C>(m, wred[0], part, bar_a, rank);
    const float post_rstd = rsqrtf(t.x * (1.f / (float)VG_H) + p.post_eps);
    const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, t.w, 2.f * t.z), t.y);
    const float pre_rstd = rsqrtf(fmaxf(pre_sum, 0.f) * (1.f / (float)VG_H) + p.pre_eps);
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int e = (((int)rank * K + k) * NT + tid) * 8;
      float bt2[8], qw[8], u[8];
      vg_ld8f(p.beta + e, bt2);
      if (p.pre_w) vg_ld8bf(p.pre_w + e, qw);
      __align__(16) __nv_bfloat16 ob[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float n = vg_rbf(v[k][i] * post_rstd);
        u[i] = fmaf(bt2[i], n, r[k][i]);
        float z = u[i] * pre_rstd;
        if (p.pre_w) z *= qw[i];
        ob[i] = __float2bfloat16_rn(z);
      }
      *reinterpret_cast<uint4*>(p.out + rb + e) = *reinterpret_cast<const uint4*>(ob);
      vg_st8f(p.new_stream + rb + e, u);
    }
  } else {
    float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int e = (((int)rank * K + k) * NT + tid) * 8;
      float al_[8], pw_[8];
      if constexpr (EARLY) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { al_[i] = al[0][i]; pw_[i] = (PW ? pw[0][i] : 1.f); }
      } else {
        vg_ld8f(p.alpha + e, al_);
        if constexpr (PW) vg_ld8bf(p.post_w + e, pw_);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float x = v[k][i];
        if constexpr (PW) x = vg_rbf(x * pw_[i]);
        v[k][i] = x;
        r[k][i] *= al_[i];
        m.x = fmaf(x, x, m.x);
      }
    }
    const float4 t1 = vg_sum4<NT, 1>(m, wred[0], part, bar_a, rank);
    const float post_rstd = rsqrtf(t1.x * (1.f / (float)VG_H) + p.post_eps);
    m = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int e = (((int)rank * K + k) * NT + tid) * 8;
      float bt[8];
      vg_ld8f(p.beta + e, bt);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float n = vg_rbf(v[k][i] * post_rstd);
        const float u = fmaf(bt[i], n, r[k][i]);
        r[k][i] = u;
        m.x = fmaf(u, u, m.x);
      }
      vg_st8f(p.new_stream + rb + e, r[k]);
    }
    const float4 t2 = vg_sum4<NT, 1>(m, wred[1], part, bar_a, rank);
    const float pre_rstd = rsqrtf(t2.x * (1.f / (float)VG_H) + p.pre_eps);
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int e = (((int)rank * K + k) * NT + tid) * 8;
      float qw[8];
      if (p.pre_w) vg_ld8bf(p.pre_w + e, qw);
      __align__(16) __nv_bfloat16 ob[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float z = r[k][i] * pre_rstd;
        if (p.pre_w) z *= qw[i];
        ob[i] = __float2bfloat16_rn(z);
      }
      *reinterpret_cast<uint4*>(p.out + rb + e) = *reinterpret_cast<const uint4*>(ob);
    }
  }
}
'''

VGB_HOST = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
template <int NT, int C, bool MOM, bool PRE, bool PW>
static void vgb_go(const VgbParams& p, int T, cudaStream_t s) {
  auto fn = vgb_kernel<NT, C, MOM, PRE, PW>;
  if constexpr (C > 1) {
    static bool init = false;
    if (!init) { cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); init = true; }
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(T * C); cfg.blockDim = dim3(NT); cfg.dynamicSmemBytes = 0; cfg.stream = s;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (C > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = C; at[na].val.clusterDim.y = 1; at[na].val.clusterDim.z = 1; ++na;
  }
#if VGB_PDL
  at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[na].val.programmaticStreamSerializationAllowed = 1; ++na;
#endif
  cfg.attrs = at; cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fn, p));
}
template <int NT, int C, bool MOM>
static bool vgb_pick(const VgbParams& p, int T, cudaStream_t s, bool pre, bool pw) {
  if (pre && pw) vgb_go<NT, C, MOM, true, true>(p, T, s);
  else if (pre) vgb_go<NT, C, MOM, true, false>(p, T, s);
  else if (pw) vgb_go<NT, C, MOM, false, true>(p, T, s);
  else vgb_go<NT, C, MOM, false, false>(p, T, s);
  return true;
}
template <int NT, int C>
static bool vgb_dispatch(const VgbParams& p, int T, cudaStream_t s, bool mom, bool pre, bool pw) {
  if (mom) return vgb_pick<NT, C, true>(p, T, s, pre, pw);
  if constexpr (C == 1) return vgb_pick<NT, 1, false>(p, T, s, pre, pw);
  return false;
}
// geo: NT | C << 12 | MOM << 20 | PRE << 21. Returns false for a geometry this build does not carry.
bool vgb_run(torch::Tensor branch, torch::Tensor stream, torch::Tensor out, torch::Tensor new_stream, torch::Tensor post_w,
             torch::Tensor pre_w, torch::Tensor alpha, torch::Tensor beta, double post_eps, double pre_eps, int64_t geo) {
  const int T = (int)branch.size(0);
  TORCH_CHECK(branch.size(1) == VG_H && stream.size(1) == VG_H && branch.is_contiguous() && stream.is_contiguous()
              && out.is_contiguous() && new_stream.is_contiguous(), "vgb: layout");
  TORCH_CHECK(alpha.numel() == VG_H && beta.numel() == VG_H, "vgb: channel-wise residual gate required");
  VgbParams p = {};
  p.branch = (const __nv_bfloat16*)branch.data_ptr(); p.stream = stream.data_ptr<float>();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.new_stream = new_stream.data_ptr<float>();
  p.post_w = post_w.numel() ? (const __nv_bfloat16*)post_w.data_ptr() : nullptr;
  p.pre_w = pre_w.numel() ? (const __nv_bfloat16*)pre_w.data_ptr() : nullptr;
  p.alpha = alpha.data_ptr<float>(); p.beta = beta.data_ptr<float>();
  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;
  if (T == 0) return true;
  auto s = c10::cuda::getCurrentCUDAStream();
  const int nt = (int)(geo & 0xfff), c = (int)((geo >> 12) & 0xff);
  const bool mom = (geo >> 20) & 1, pre = (geo >> 21) & 1, pw = p.post_w != nullptr;
#define VG_CASE(N_, C_) if (nt == N_ && c == C_) return vgb_dispatch<N_, C_>(p, T, s, mom, pre, pw);
  VGB_CASES
#undef VG_CASE
  return false;
}
'''
VGB_DECL = ("bool vgb_run(torch::Tensor branch, torch::Tensor stream, torch::Tensor out, torch::Tensor new_stream, "
            "torch::Tensor post_w, torch::Tensor pre_w, torch::Tensor alpha, torch::Tensor beta, double post_eps, "
            "double pre_eps, int64_t geo);")
VGB_GEOS_SM90 = ((64, 16), (128, 8), (256, 4), (128, 4), (256, 2), (512, 1), (256, 1), (1024, 1))
VGB_GEOS_LOCAL = ((512, 1), (256, 1), (1024, 1))

# VG-F: the VML1 finalize (vm_fin's contract and arithmetic: post-expert norm of each local route's y row, routing
# weight, sum over the token's routes in slot order; counter resets for the next VML call) with every y load of a chunk
# of up to 4 local routes in flight at once (vm_fin's loads sit behind per-slot branches: one L2 round trip per slot and
# column block). One CTA per token, 256 threads, thread tid owns columns 1024 m + 4 tid (m = 0..3) as vm_fin, so the
# per-thread, warp and CTA summation orders are vm_fin's (bitwise-equal output expected; prepare() checks).
VGF_CU = r'''
#include <cuda_bf16.h>
#include <stdint.h>
#define VF_KDIM 4096
__device__ __forceinline__ float vf_bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__global__ void __launch_bounds__(256) vgf_kernel(const float* __restrict__ y, const int* __restrict__ rmap,
    const float* __restrict__ wts, const float* __restrict__ nw, __nv_bfloat16* __restrict__ out, int* xready, int* doneA,
    int* cnt, int n_xready, int n_doneA, int n_cnt) {
  __shared__ int s_r[8], s_r8[8];
  __shared__ float s_w[8], s_w8[8];
  __shared__ int s_n;
  __shared__ float s_ss[2][8][4];
  const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float4 w4[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(nw + 1024 * m + 4 * tid));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  if (t == 0) {
    for (int i = tid; i < n_xready; i += 256) xready[i] = 0;
    for (int i = tid; i < n_doneA; i += 256) doneA[i] = 0;
    for (int i = tid; i < n_cnt; i += 256) cnt[i] = 0;
  }
  if (tid < 8) {   // the slot loads in parallel (one L2 round trip), then thread 0 compacts the local slots in order
    s_r8[tid] = __ldcg(rmap + t * 8 + tid);
    s_w8[tid] = __ldg(wts + t * 8 + tid);
  }
  __syncthreads();
  if (tid == 0) {
    int n = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = s_r8[k];
      if (r >= 0) { s_r[n] = r; s_w[n] = s_w8[k]; ++n; }
    }
    s_n = n;
  }
  __syncthreads();
  const int nloc = s_n;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  float a[4][4];
#pragma unroll
  for (int m = 0; m < 4; ++m) { a[m][0] = 0.f; a[m][1] = 0.f; a[m][2] = 0.f; a[m][3] = 0.f; }
  int par = 0;
  for (int c0 = 0; c0 < nloc; c0 += 4, par ^= 1) {
    float4 v[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = (c0 + j < nloc) ? s_r[c0 + j] : -1;
#pragma unroll
      for (int m = 0; m < 4; ++m)
        v[j][m] = (r >= 0) ? __ldcg(reinterpret_cast<const float4*>(y + (size_t)r * VF_KDIM + 1024 * m + 4 * tid)) : z4;
    }
    float ss[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ss[j] = 0.f;
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        float f;
        f = vf_bf16r(v[j][m].x) * w4[m].x; ss[j] += f * f; f = vf_bf16r(v[j][m].y) * w4[m].y; ss[j] += f * f;
        f = vf_bf16r(v[j][m].z) * w4[m].z; ss[j] += f * f; f = vf_bf16r(v[j][m].w) * w4[m].w; ss[j] += f * f;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss[j] += __shfl_xor_sync(0xffffffffu, ss[j], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int j = 0; j < 4; ++j) s_ss[par][warp][j] = ss[j];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (c0 + j < nloc) {
        float s = 0.f;
        for (int q = 0; q < 8; ++q) s += s_ss[par][q][j];
        const float fr = rsqrtf(s * (1.f / 4096.f) + 1e-8f) * s_w[c0 + j];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          a[m][0] += vf_bf16r(v[j][m].x) * w4[m].x * fr; a[m][1] += vf_bf16r(v[j][m].y) * w4[m].y * fr;
          a[m][2] += vf_bf16r(v[j][m].z) * w4[m].z * fr; a[m][3] += vf_bf16r(v[j][m].w) * w4[m].w * fr;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a[m][0], a[m][1]), o1 = __floats2bfloat162_rn(a[m][2], a[m][3]);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0); o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(out + (size_t)t * VF_KDIM + 1024 * m + 4 * tid) = o;
  }
}
'''
VGF_HOST = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
void vgf_run(torch::Tensor y, torch::Tensor rmap, torch::Tensor wts, torch::Tensor nw, torch::Tensor out, torch::Tensor xready,
             torch::Tensor doneA, torch::Tensor cnt, int64_t M, int64_t pdl) {
  TORCH_CHECK(y.scalar_type() == at::kFloat && rmap.scalar_type() == at::kInt && wts.scalar_type() == at::kFloat
              && nw.scalar_type() == at::kFloat && out.scalar_type() == at::kBFloat16 && nw.numel() == VF_KDIM
              && out.size(1) == VF_KDIM && out.is_contiguous() && wts.is_contiguous() && rmap.numel() >= 8 * M
              && out.size(0) >= M, "vgf: tensors");
  if (M == 0) return;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)M); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vgf_kernel, (const float*)y.data_ptr(), (const int*)rmap.data_ptr(),
      (const float*)wts.data_ptr(), (const float*)nw.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (int*)xready.data_ptr(),
      (int*)doneA.data_ptr(), (int*)cnt.data_ptr(), (int)xready.numel(), (int)doneA.numel(), (int)cnt.numel()));
}
'''
VGF_DECL = ("void vgf_run(torch::Tensor y, torch::Tensor rmap, torch::Tensor wts, torch::Tensor nw, torch::Tensor out, "
            "torch::Tensor xready, torch::Tensor doneA, torch::Tensor cnt, int64_t M, int64_t pdl);")


def vgb_src(geos, pdl):
    cases = " ".join(f"VG_CASE({n}, {c})" for n, c in geos)
    return VGB_CU + VGB_HOST.replace("VGB_CASES", cases).replace("#if VGB_PDL", "#if 1" if pdl else "#if 0")


def vgb_geo(nt, c, mom, pre):
    return nt | (c << 12) | (int(mom) << 20) | (int(pre) << 21)


def vgb_build(geos, pdl, arch_flags, name):
    import torch
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name=name, cpp_sources=VGB_DECL, cuda_sources=vgb_src(geos, pdl), functions=["vgb_run"],
                      extra_cuda_cflags=["-O3", "-Xptxas=-warn-spills"] + arch_flags, verbose=False)
    print(f"[vgb] built {name} ({len(geos)} geometries) in {time.time() - t0:.1f}s", flush=True)
    return mod


def op_weights(op):
    """(post_w, pre_w, alpha, beta, input_scale, post_eps, pre_eps) of an SGLang FusedPostNormResidualPreNorm."""
    import torch
    op._ensure_cached_weights(torch.bfloat16)
    dev = op._cc_alpha.device
    e = torch.empty(0, device=dev, dtype=torch.bfloat16)
    pw = op._cc_post_weight.to(torch.bfloat16).contiguous() if op._post_norm.has_weight else e
    qw = op._cc_pre_weight.to(torch.bfloat16).contiguous() if op._pre_norm.has_weight else e
    return (pw, qw, op._cc_alpha.float().contiguous(), op._cc_beta.float().contiguous(),
            bool(op._post_norm_is_input_scale), float(op._post_norm.eps), float(op._pre_norm.eps))


def vgb_call(mod, w, stream, branch, geo):
    import torch
    pw, qw, al, bt, isc, pe, qe = w
    if pw.numel() and not isc:
        raise ValueError("vgb: a post weight applied after the norm is not supported")
    T = branch.shape[0]
    out = torch.empty((T, H), dtype=torch.bfloat16, device=branch.device)
    ns = torch.empty((T, H), dtype=torch.float32, device=branch.device)
    if not mod.vgb_run(branch, stream, out, ns, pw, qw, al, bt, pe, qe, geo):
        raise ValueError(f"vgb: geometry {geo:#x} not built")
    return out, ns


def rel(a, b):
    import torch
    return float((a.float() - b.float()).norm() / b.float().norm())


def main_local():
    """RTX 3070 (sm_86): correctness of the C = 1 builds vs SGLang's forward_native-equivalent math in torch."""
    import torch
    dev = "cuda"
    mod = vgb_build(VGB_GEOS_LOCAL, False, ["-gencode=arch=compute_86,code=sm_86"], "vgb_local_v1")
    g = torch.Generator(device=dev).manual_seed(7)
    gate = (0.3 * torch.randn(H, generator=g, device=dev)).to(torch.bfloat16).float() / 0.3
    beta = torch.sigmoid(gate)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-gate) * (1 + beta), min=1e-3))
    bad = 0
    for kind in ("attn", "ffn"):
        pw = ((1 + 0.1 * torch.randn(H, generator=g, device=dev))).to(torch.bfloat16) if kind == "ffn" else \
            torch.empty(0, device=dev, dtype=torch.bfloat16)
        qw = (1 + 0.1 * torch.randn(H, generator=g, device=dev)).to(torch.bfloat16)
        w = (pw, qw, alpha.contiguous(), beta.contiguous(), kind == "ffn", 1e-8, 1e-5)
        for T in (1, 5, 32, 128, 256):
            stream = torch.randn((T, H), generator=g, device=dev) * 3
            branch = torch.randn((T, H), generator=g, device=dev).to(torch.bfloat16)
            v = branch.float()
            if kind == "ffn":
                v = (v * pw.float()).to(torch.bfloat16).float()
            n = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-8)).to(torch.bfloat16).float()
            ns_ref = alpha * stream + beta * n
            out_ref = (ns_ref * torch.rsqrt(ns_ref.pow(2).mean(-1, keepdim=True) + 1e-5) * qw.float()).to(torch.bfloat16)
            for nt, c in VGB_GEOS_LOCAL:
                for mom in (True, False):
                    for pre in (False, True):
                        out, ns = vgb_call(mod, w, stream, branch, vgb_geo(nt, c, mom, pre))
                        torch.cuda.synchronize()
                        e0, e1 = rel(out, out_ref), rel(ns, ns_ref)
                        ok = e0 < 4e-3 and e1 < 1e-4
                        bad += not ok
                        if not ok or T == 256:
                            print(f"[local] {kind} T={T} vgb {nt}/{c} {'mom' if mom else 'seq'} pre={int(pre)}: out {e0:.2e} "
                                  f"stream {e1:.2e} -> {'ok' if ok else 'FAIL'}", flush=True)
    print(f"[local] {'ALL OK' if not bad else f'{bad} FAILURES'}", flush=True)
    return 1 if bad else 0


def main():
    import torch
    sys.path.insert(0, str(ROOT / "tools" / "judge"))
    import judge as J
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    ts = [int(x) for x in args.get("ts", "32,128,256").split(",")]
    rounds = int(args.get("rounds", 30))
    outp = Path(args.get("out", "box_out/vg2/bnd.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    prob = J.load_problem(Path("problems/muse_decode_layer"))
    ref = prob.ref
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")
    base = J.load_kernel(prob.dir / "baseline.py", "okc_baseline", prob)
    st = base.prepare(J.clone_data(weights))
    from sglang.kernels.ops.muse_spark_v12.norm_residual import fused_norm_residual as FNR
    mod = vgb_build(VGB_GEOS_SM90, True, ["-gencode=arch=compute_90a,code=sm_90a"], "vgb_lab_v1")
    timer = J.GpuTimer()
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(77)
    res = {"rounds": rounds, "shapes": {}}
    for kind in ("attn", "ffn"):
        op = st["attn_boundary" if kind == "attn" else "ffn_boundary"]
        w = op_weights(op)
        wgemm = weights["o"] if kind == "attn" else weights["unpooler"]
        for T in ts:
            K = wgemm.shape[1]
            xin = (torch.randn((T, K), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            stream = torch.randn((T, H), generator=gen, device=dev) * 2
            branch0 = torch.mm(xin, wgemm.t())
            ref_out = op(stream, branch0, branch_dtype=torch.bfloat16, router_dtype=None)
            arms = {"none": None, "sgl": ("mod",)}
            for c in (2, 4, 8, 16):
                arms[f"sgl_c{c}"] = ("fnr", c, None)
            for n, pf in ((1024, 0), (512, 1)):   # SGLang's own T >= 192 / T < 192 shapes are 512-0 / 256-1 (= sgl)
                arms[f"sgl_t{n}{pf}"] = ("fnr", 1, (n, bool(pf)))
            for n, c in VGB_GEOS_SM90:
                for mom in (True, False):
                    if not mom and c > 1:
                        continue
                    for pre in ((False, True) if kind == "attn" else (False,)):
                        arms[f"vgb_{n}_{c}_{'m' if mom else 's'}{'p' if pre else 'n'}"] = ("vgb", vgb_geo(n, c, mom, pre))
            if "arms" in args:
                keep = set(args["arms"].split(",")) | {"none", "sgl"}
                arms = {k: v for k, v in arms.items() if k in keep}

            def bnd(a, branch):
                if a is None:
                    return None
                if a[0] == "mod":
                    return op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=None)[:2]
                if a[0] == "fnr":
                    pw, qw, al, bt, isc, pe, qe = w
                    o = FNR(branch, stream, pe, qe, post_norm_is_input_scale=isc, post_weight=pw if pw.numel() else None,
                            pre_weight=qw if qw.numel() else None, alpha=al, beta=bt, want_router=False,
                            launch_config=a[2], cluster_size=a[1])
                    return o[:2]
                return vgb_call(mod, w, stream, branch, a[1])

            graphs, errs = {}, {}
            for name, a in list(arms.items()):
                try:
                    got = bnd(a, branch0)
                    torch.cuda.synchronize()
                    if got is not None:
                        errs[name] = (rel(got[0], ref_out[0]), rel(got[1], ref_out[1]))
                        if not (errs[name][0] < 1e-2 and errs[name][1] < 1e-4):
                            print(f"[lab] {kind} T={T} {name}: MISMATCH out {errs[name][0]:.2e} stream {errs[name][1]:.2e}",
                                  flush=True)
                            arms.pop(name)
                            continue

                    def fn(_d, a=a):
                        br = torch.mm(xin, wgemm.t())
                        return bnd(a, br)
                    graphs[name] = timer.capture(fn, {})[0]
                except Exception as ex:
                    print(f"[lab] {kind} T={T} {name}: skipped ({repr(ex)[:160]})", flush=True)
                    arms.pop(name, None)
            names = list(graphs)
            ts_ = {n: [] for n in names}
            for rd in range(rounds):
                order = names[rd % len(names):] + names[:rd % len(names)]
                for n in order:
                    ts_[n].append(timer.time_replay(graphs[n]))
            med = {n: statistics.median(v) for n, v in ts_.items()}
            d0 = med["none"]
            rows = sorted(((med[n] - d0, n) for n in names if n != "none"))
            print(f"[lab] {kind} T={T}: GEMM alone {d0:.2f} us; boundary increment (us, median of {rounds}):", flush=True)
            for inc, n in rows:
                e = errs.get(n, (0, 0))
                print(f"   {n:16s} {inc:7.2f}   (out {e[0]:.1e} stream {e[1]:.1e})", flush=True)
            res["shapes"][f"{kind}_{T}"] = {"gemm_us": d0, "inc_us": {n: med[n] - d0 for n in names}, "errs": errs,
                                           "raw": ts_}
            outp.write_text(json.dumps(res, indent=1))
            del graphs
    print(f"[lab] wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main_local() if "--local" in sys.argv else main())

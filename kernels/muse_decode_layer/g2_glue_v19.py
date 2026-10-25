"""G2 glue riders for the decode layer, k3_v19 subset: canonical source of the G2 section of k3_v19.py (tools/gen_k3_v19.py slices
every top-level node below verbatim into the layer; this file is not imported on its own: the nodes use the layer's globals).
Derived from kernels/muse_decode_layer/g2_glue.py (k3_g2's source) by REMOVING G2-C (the unpooler + FFN boundary kernel), the
router-first ('main') and register-weight ('fusedr') variants and every T = 8 instantiation, and by restricting G2-A to T = 1;
the retained code (prt_kernel, stream_h, launch / info, the G2-B self-test) is g2_glue.py's text. Design and measurements:
kernels/muse_decode_layer/g2_design.md.

  G2-A (G2_PRT): pooler (P1 packed BF16 GEMV) + T = 1 FP32 router + the routing / pre-expert-norm tail in ONE kernel
       (measured +1.1..1.5% at decode batch 1; at T = 8 it lost, so T = 8 is not built here).
  G2-B (G2_P1_PDL): the P1 GEMV launches with programmatic stream serialization, per projection (o, unpooler on).
  The third rider of k3_v19 (router place 'side_first' at the T = 8 bucket) is a V8_ROUTER_PLACE value patched by the generator.
"""
import time

import torch

# ---------------------------------------------------------------- G2: glue fusion at T <= 8
G2_ENABLE = True    # master switch (read at call time; False at prepare also skips the G2 build and self-tests)
G2_PRT = True       # G2-A: pooler + FP32 router + tail in one kernel at T = 1 (read at call time)
G2_PRT_PDL = True   # G2-A kernel launched with programmatic stream serialization (read at call time)
G2_PRT_MODE = {1: 'fused'}   # G2-A per bucket (read at call time): 'fused' = the router inside the pooler kernel (weights via smem); a bucket that is absent or not 'fused' (T = 8) keeps k3_v18's path (box A/B g2_ab3: fused +1.1..1.6% at T = 1, -0.3% at T = 8, so only T = 1 is built)
G2_P1_PDL = {'qkv': False, 'o': True, 'pooler': False, 'unpooler': True}   # G2-B: P1 launches with PDL, per projection (read at call time; the pooler only matters with G2_PRT off, where PDL would let it take the SMs before the side-stream router)
G2_PREPARE_BY_S = 190.0   # no G2 build starts after this many seconds of prepare (limit 300 s; k3_g2's cold build with 6 prt + 2 ffn instantiations took 44 s, self-tests ~5 s)
G2_SELF_TEST_CASES = ((1, 0, 701), (1, 64, 702), (1, 224, 703), (1, 32, 704), (1, 160, 705), (1, 96, 706))   # (T, local_start, seed): T = 1 only
_G2_LIVE = {"p1pdl": False}   # set by _g2_prepare: G2-B's self-test passed (G2_P1_PDL is honoured only then)

G2_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void g2_prt(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff, torch::Tensor esc,
            torch::Tensor fmt, torch::Tensor y, torch::Tensor x32, torch::Tensor w32, torch::Tensor part, torch::Tensor nw,
            torch::Tensor bias, torch::Tensor lstart, torch::Tensor xn, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid,
            torch::Tensor tcnt, double eps, double mult, double rscale, double deps, int64_t pdl, int64_t rt);
std::vector<int64_t> g2_info(int64_t mt, int64_t rt);
"""

# Appended to P1_CUDA_SRC (verbatim) in one load_inline; @STREAM_H@ = p1::stream with one hook call after its PDL wait
# (derived from P1_CUDA_SRC by _g2_stream_h at build time, never hand-copied).
G2_CUDA_SRC = r"""
namespace g2 {
using namespace p1;
constexpr int PD = 4096;     // pooler / expert width
constexpr int NE = 256;      // experts
constexpr int TK = 8;        // top-k
constexpr int NT = 512;      // threads per CTA (16 warps): the P1 pooler geometry (16, 1, 1) and v8::tail_kernel's NT
constexpr int RK = 8192;     // router K
constexpr int RKS4 = 256;    // float4 per router K slice row (1024 floats)
constexpr int NRT = 256;     // pooler row tiles = CTAs = router32's (32 row groups x 8 K slices)

struct PArgs {
  const float* x32;            // router input FP32 [M, 8192] (the boundary's FP32 copy)
  const float* w32;            // router weight FP32 [256, 8192]
  float* part;                 // router32's partials [8, M, 256]
  const __nv_bfloat16* nw;     // pre-expert norm effective weight [4096]
  const float* bias;           // router bias [256]
  const int* lstart;           // [1]
  __nv_bfloat16* xn;           // [M, 4096]
  float* tw; int* tid; int* lid;   // [M, 8] each
  int* tcnt;                   // [1] ticket counter: zero between calls (the last CTA resets it)
  float eps, mult, rscale, deps;
};

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

@STREAM_H@

// grid = 256 CTAs (CTA c = pooler row tile c AND, with a router mode, router32's CTA (bx, sl) = (c % 32, c / 32)), block = 512.
// RM 0: no router here (its partials come from router32 launched just before on the same stream; this kernel's PDL wait covers
// them), dynamic smem 0. RM 1: router weights cp.async'd to smem before the PDL wait, dynamic smem (8 + MT) * 4 KB. RM 2: router
// weights prefetched to L2 before the wait and loaded to registers after the pooler stream, dynamic smem MT * 4 KB (keeps L1
// for the pooler's x operand). Values are computed by the expressions of p1s_kernel<1, 0> (S = 1), router32_kernel<MT> and
// tail_kernel<false, 8> in their order, so the outputs equal the three-kernel path's.
template <int MT, int RM>
__global__ void __launch_bounds__(512, 2) prt_kernel(const __grid_constant__ Args A, const __grid_constant__ PArgs G) {
  constexpr bool RT = RM != 0;
  extern __shared__ float4 dsm[];
  float4* const wsm = dsm;                                // RM 1: router weight rows bx*8 .. bx*8+7, K slice sl: [8][256] float4
  float4* const xsm = dsm + (RM == 1 ? 8 * RKS4 : 0);     // router input K slice sl: [MT][256] float4 (rows >= M zero)
  __shared__ float red[MAXW][128];
  __shared__ float tred[8][NT / 32];
  __shared__ int s_last;
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = (int)(blockDim.x >> 5);
  const int rt = (int)blockIdx.x, m = A.M;
  const int bx = rt & 31, sl = rt >> 5, k0 = sl * 1024;
  // 0. router weights (prepare-time data) -> smem, issued before the PDL wait: this 8.4 MB stream overlaps the predecessor
  if constexpr (RM == 1) {
    for (int idx = tid; idx < 8 * RKS4; idx += (int)blockDim.x) {
      const int r = idx >> 8, c = idx & 255;
      cp16(wsm + idx, reinterpret_cast<const float4*>(G.w32 + (size_t)(bx * 8 + r) * RK + k0) + c);
    }
    cp_commit();
  } else if constexpr (RM == 2) {
    if (tid < 256) {   // 8 rows x 4 KB = 256 lines of 128 B
      const float* pl = G.w32 + (size_t)(bx * 8 + (tid >> 5)) * RK + k0 + (tid & 31) * 32;
      asm volatile("prefetch.global.L2 [%0];" :: "l"(pl));
    }
  }
  // 1. pooler: p1s_kernel<1, 0> with S = 1; right after stream's PDL wait, the router input slice -> smem
  const int c0 = 0, c1 = A.nsb_total;
  const int sb_lo = c0 + (int)((long long)warp * (c1 - c0) / nw);
  const int sb_hi = c0 + (int)((long long)(warp + 1) * (c1 - c0) / nw);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  auto hook = [&]() {
    if constexpr (RT) {
      for (int idx = tid; idx < MT * RKS4; idx += (int)blockDim.x) {
        const int j = idx >> 8, c = idx & 255;
        if (j < m) cp16(xsm + idx, reinterpret_cast<const float4*>(G.x32 + (size_t)j * RK + k0) + c);
        else xsm[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      cp_commit();
    }
  };
  stream_h<1, 0>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc, hook);
  {
    const int g = lane >> 2, t = lane & 3;
    float* rw = red[warp];
    rw[g * 8 + 2 * t] = acc[0];
    rw[g * 8 + 2 * t + 1] = acc[1];
    rw[(g + 8) * 8 + 2 * t] = acc[2];
    rw[(g + 8) * 8 + 2 * t + 1] = acc[3];
  }
  if constexpr (RT) cp_wait_all();
  __syncthreads();
  {
    const int row = rt * 16 + (tid & 15), tok = tid >> 4;
    if (tid < 16 * m) {
      float v = 0.f;
      for (int q = 0; q < nw; ++q) v += red[q][(tid & 15) * 8 + tok];
      v *= A.scale[rt];
      A.out[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
    }
  }
  // 2. router32_kernel<MT>'s CTA (bx, sl): warp w < 8 = router row bx*8 + w over K slice sl
  if (RT && warp < 8) {
    const int row = bx * 8 + warp;
    float4 wr[8];
    if constexpr (RM == 2) {
      const float4* wp = reinterpret_cast<const float4*>(G.w32 + (long)row * RK + k0);
#pragma unroll
      for (int i = 0; i < 8; ++i) wr[i] = __ldcs(wp + i * 32 + lane);
    }
    float racc[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) racc[j] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = i * 32 + lane;
      const float4 wv = RM == 1 ? wsm[warp * RKS4 + c] : wr[i];
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        const float4 xv = xsm[j * RKS4 + c];
        racc[j] = fmaf(wv.x, xv.x, racc[j]);
        racc[j] = fmaf(wv.y, xv.y, racc[j]);
        racc[j] = fmaf(wv.z, xv.z, racc[j]);
        racc[j] = fmaf(wv.w, xv.w, racc[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      float a = racc[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0 && j < m) G.part[((long)sl * m + j) * NE + row] = a;
    }
  }
  // 3. the last CTA (ticket; no spin) runs the tail for every token
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(G.tcnt, 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid == 0) atomicExch(G.tcnt, 0);
  const uint4 wq = *(reinterpret_cast<const uint4*>(G.nw) + tid);
  uint4 yq[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t)
    yq[t] = t < m ? __ldcg(reinterpret_cast<const uint4*>(A.out + (size_t)t * A.ldo) + tid) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    if (t < m) {
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&yq[t]);
      float v[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) tred[t][warp] = ss;
    }
  }
  __syncthreads();
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    if (t < m) {
      float tot = 0.f;
#pragma unroll
      for (int i = 0; i < NT / 32; ++i) tot += tred[t][i];
      const float rstd = rsqrtf(tot / (float)PD + G.eps);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&yq[t]);
      const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq);
      __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((__bfloat162float(h[i]) * rstd) * __bfloat162float(w[i]));
      *(reinterpret_cast<uint4*>(G.xn + (long)t * PD) + tid) = *reinterpret_cast<const uint4*>(o8);
    }
  }
  if (warp >= m) return;
  // tail_kernel<false, 8>'s warp-0 routing for token t = warp (lane l holds experts 8l .. 8l+7)
  const int t = warp;
  const float4* pbias = reinterpret_cast<const float4*>(G.bias) + lane * 2;
  const float4 c0b = pbias[0], c1b = pbias[1];
  const long sstride = (long)m * NE;
  const float4* pq0 = reinterpret_cast<const float4*>(G.part + (long)t * NE) + lane * 2;
  float4 a0 = __ldcg(pq0), a1 = __ldcg(pq0 + 1);
#pragma unroll
  for (int q = 1; q < 8; ++q) {   // FP32 split-K partials summed in a fixed order
    const float4* pq = reinterpret_cast<const float4*>(G.part + q * sstride + (long)t * NE) + lane * 2;
    const float4 b0 = __ldcg(pq), b1 = __ldcg(pq + 1);
    a0.x += b0.x; a0.y += b0.y; a0.z += b0.z; a0.w += b0.w;
    a1.x += b1.x; a1.y += b1.y; a1.z += b1.z; a1.w += b1.w;
  }
  const float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;   // tail_kernel's lb == nullptr term
  a0.x += qb0.x; a0.y += qb0.y; a0.z += qb0.z; a0.w += qb0.w; a1.x += qb1.x; a1.y += qb1.y; a1.z += qb1.z; a1.w += qb1.w;
  const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  const float bs[8] = {c0b.x, c0b.y, c0b.z, c0b.w, c1b.x, c1b.y, c1b.z, c1b.w};
  float s[8], b[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    s[j] = 1.f / (1.f + expf(-lg[j] * G.mult));
    b[j] = s[j] + bs[j];
  }
  // top-8 by biased score, ties -> lowest expert id: the selection of tail_kernel's shuffle loop, one redux.sync per pick. The key
  // is the order-preserving uint image of b (b = sigmoid + bias is never -0.0 / NaN: sigmoid > 0); 0 marks a picked slot.
  unsigned key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const unsigned u = __float_as_uint(b[j]);
    key[j] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  }
  int sel_id[TK];
  float sel_s[TK];
#pragma unroll
  for (int k = 0; k < TK; ++k) {
    unsigned hv = key[0];
    int hj = 0;
#pragma unroll
    for (int j = 1; j < 8; ++j)
      if (key[j] > hv) { hv = key[j]; hj = j; }
    float hs = s[0];
#pragma unroll
    for (int j = 1; j < 8; ++j)
      if (j == hj) hs = s[j];
    const unsigned mx = __reduce_max_sync(0xffffffffu, hv);
    const int wl = __ffs(__ballot_sync(0xffffffffu, hv == mx)) - 1;   // lowest lane holding the max = lowest expert id
    const int wj = __shfl_sync(0xffffffffu, hj, wl);
    sel_id[k] = wl * 8 + wj;
    sel_s[k] = __shfl_sync(0xffffffffu, hs, wl);
    if (lane == wl) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (j == hj) key[j] = 0u;
    }
  }
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < TK; ++i) sum += sel_s[i];
    const float inv = G.rscale / (sum + G.deps);
    const int ls = G.lstart[0];
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      G.tw[t * TK + k] = sel_s[k] * inv;
      G.tid[t * TK + k] = sel_id[k];
      const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
      G.lid[t * TK + k] = rel < 32 ? rel : -1;
    }
  }
}

template <int MT, int RM>
static void prt_launch(const Args& A, const PArgs& G, int pdl) {
  const int smem = RM == 1 ? (8 + MT) * RKS4 * 16 : RM == 2 ? MT * RKS4 * 16 : 0;
  static bool attr_set = false;
  if (!attr_set) {
    if (smem) C10_CUDA_CHECK(cudaFuncSetAttribute(prt_kernel<MT, RM>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr_set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NRT, 1, 1);
  cfg.blockDim = dim3((unsigned)NT, 1, 1);
  cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, prt_kernel<MT, RM>, A, G));
}

template <int MT, int RM>
static std::vector<int64_t> prt_info() {
  const int smem = RM == 1 ? (8 + MT) * RKS4 * 16 : RM == 2 ? MT * RKS4 * 16 : 0;
  if (smem) C10_CUDA_CHECK(cudaFuncSetAttribute(prt_kernel<MT, RM>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, prt_kernel<MT, RM>));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, prt_kernel<MT, RM>, NT, smem));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ, (int64_t)smem};
}

}  // namespace g2

void g2_prt(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff, torch::Tensor esc,
            torch::Tensor fmt, torch::Tensor y, torch::Tensor x32, torch::Tensor w32, torch::Tensor part, torch::Tensor nw,
            torch::Tensor bias, torch::Tensor lstart, torch::Tensor xn, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid,
            torch::Tensor tcnt, double eps, double mult, double rscale, double deps, int64_t pdl, int64_t rt) {
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == g2::RK &&
              M >= 1 && M <= 8 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "g2_prt: x bf16 [M <= 8, 8192] contiguous");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt && fmt.scalar_type() == at::kByte &&
              toff.numel() >= g2::NRT && scale.numel() >= g2::NRT && eoff.numel() >= g2::NRT + 1, "g2_prt: packed pooler");
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.is_contiguous() && y.size(0) == M && y.size(1) == g2::PD, "g2_prt: y");
  TORCH_CHECK(x32.scalar_type() == at::kFloat && x32.is_contiguous() && x32.size(0) == M && x32.size(1) == g2::RK &&
              reinterpret_cast<uintptr_t>(x32.data_ptr()) % 16 == 0, "g2_prt: x32 FP32 [M, 8192]");
  TORCH_CHECK(w32.scalar_type() == at::kFloat && w32.is_contiguous() && w32.size(0) == g2::NE && w32.size(1) == g2::RK &&
              reinterpret_cast<uintptr_t>(w32.data_ptr()) % 16 == 0, "g2_prt: w32 FP32 [256, 8192]");
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() == 8 * M * g2::NE &&
              reinterpret_cast<uintptr_t>(part.data_ptr()) % 16 == 0, "g2_prt: part [8, M, 256]");
  TORCH_CHECK(nw.scalar_type() == at::kBFloat16 && nw.numel() == g2::PD && reinterpret_cast<uintptr_t>(nw.data_ptr()) % 16 == 0 &&
              bias.scalar_type() == at::kFloat && bias.numel() == g2::NE && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0 &&
              lstart.scalar_type() == at::kInt, "g2_prt: nw / bias / lstart");
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous() && xn.size(0) == M && xn.size(1) == g2::PD &&
              reinterpret_cast<uintptr_t>(xn.data_ptr()) % 16 == 0 && tw.scalar_type() == at::kFloat && tw.is_contiguous() &&
              tw.numel() == M * g2::TK && tid.scalar_type() == at::kInt && tid.is_contiguous() && tid.numel() == M * g2::TK &&
              lid.scalar_type() == at::kInt && lid.is_contiguous() && lid.numel() == M * g2::TK &&
              tcnt.scalar_type() == at::kInt && tcnt.numel() >= 1, "g2_prt: outputs");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1::Args A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = g2::RK;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = reinterpret_cast<__nv_bfloat16*>(y.data_ptr()); A.ldo = g2::PD; A.N = g2::PD; A.nsb_total = g2::RK / 64;
  A.ws = nullptr; A.cnt = nullptr; A.red_mode = 0;
  g2::PArgs G;
  G.x32 = x32.data_ptr<float>(); G.w32 = w32.data_ptr<float>(); G.part = part.data_ptr<float>();
  G.nw = reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr()); G.bias = bias.data_ptr<float>(); G.lstart = lstart.data_ptr<int>();
  G.xn = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr()); G.tw = tw.data_ptr<float>(); G.tid = tid.data_ptr<int>();
  G.lid = lid.data_ptr<int>(); G.tcnt = tcnt.data_ptr<int>();
  G.eps = (float)eps; G.mult = (float)mult; G.rscale = (float)rscale; G.deps = (float)deps;
  TORCH_CHECK(M == 1 && rt == 1, "g2_prt: k3_v19 builds only the T = 1 kernel with the router inside the pooler kernel");
  g2::prt_launch<1, 1>(A, G, (int)pdl);
}

std::vector<int64_t> g2_info(int64_t mt, int64_t rt) {
  TORCH_CHECK(mt == 1 && rt == 1, "g2_info: k3_v19 builds only prt_kernel<1, 1>");
  return g2::prt_info<1, 1>();
}

"""


def _g2_stream_h(p1_src):
    """p1::stream from P1_CUDA_SRC with one change: a `hook` functor called right after its griddepcontrol.wait (the G2 kernel
    stages the router input there). Everything else of the function text is P1's, byte for byte (asserted)."""
    head = "template <int D, int KM>\n__device__ __forceinline__ void stream(const Args& A, int rt, int lane, int sb_lo, int nsb, float (&acc)[4]) {\n"
    i = p1_src.index(head)
    j = p1_src.index("// D = 1: <= 64 registers", i)
    body = p1_src[i + len(head):j]
    wait = '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
    assert body.count(wait) == 1 and body.count("griddepcontrol") == 1, "p1::stream: one PDL wait expected"
    new_head = ("template <int D, int KM, typename H>\n__device__ __forceinline__ void stream_h(const Args& A, int rt, int lane, int sb_lo, "
                "int nsb, float (&acc)[4], H&& hook) {\n")
    return new_head + body.replace(wait, wait + "  hook();\n").rstrip("\n") + "\n"


def _g2_build():
    """P1_CUDA_SRC verbatim + the G2-A kernel (T = 1 instantiation only) in one load_inline (layer-only P1 instantiation). Prepare-time only."""
    from torch.utils.cpp_extension import load_inline
    cuda = P1_CUDA_SRC + "\n" + G2_CUDA_SRC.replace("@STREAM_H@", _g2_stream_h(P1_CUDA_SRC))
    flags = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a", "-DP1_LAYER_ONLY=1"]
    tag = hashlib.sha1((G2_CPP_SRC + cuda + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"k3_g2_{tag}", cpp_sources=G2_CPP_SRC, cuda_sources=cuda, functions=["g2_prt", "g2_info"],
                      extra_cuda_cflags=flags, verbose=False)
    print(f"[g2] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


def _g2_p1_pdl(proj):
    """1 when this projection's P1 GEMV launches with programmatic stream serialization (G2-B, read at call time)."""
    return 1 if (G2_ENABLE and _G2_LIVE["p1pdl"] and G2_P1_PDL.get(proj)) else 0


def _g2_prt_run(g2, P, ffn_input, router_input, state, data, pdl):
    """One G2-A call: (expert_x BF16 [T, 4096], topk_weights FP32 [T, 8], topk_ids int32 [T, 8], local_ids int32 [T, 8]).
    The router runs inside the kernel (T = 1). Shape-only allocation, no sync."""
    tokens = ffn_input.shape[0]
    dev = ffn_input.device
    y = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=dev)
    xn = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    tw = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    lid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    v8 = state["v8"]
    g2["mod"].g2_prt(ffn_input, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], y, router_input, state["router"], part,
                     v8["nw"], v8["bias"], data["local_start"], xn, tw, tid, lid, g2["tcnt"], RMS_EPS, 0.1767766952966369, 1.0, 1e-15,
                     pdl, 1)
    return xn, tw, tid, lid


def _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail):
    """G2-A where it applies (T = 1 on the FP32 router + the separate tail, G2_ENABLE / G2_PRT on, its prepare self-test passed),
    else None -> the caller keeps k3_v18's router / pooler / tail. Shape-only branches; a host exception drops G2-A for the run
    (once, with a line; g2_prt raises before it launches anything)."""
    g2 = state.get("g2")
    if not (G2_ENABLE and G2_PRT and g2 is not None and g2.get("prt")) or tokens != 1:
        return None
    if mode != "f32" or tail != "sep" or router_input is None or state.get("v8") is None:
        return None
    p1 = state.get("p1")
    if p1 is None or "pooler" not in p1["proj"] or not p1["route"].get(("pooler", p1_bucket(tokens))):
        return None
    if not (_v13_p1_ok(ffn_input, HIDDEN) and router_input.dtype == torch.float32 and router_input.is_contiguous()
            and router_input.shape == ffn_input.shape and router_input.data_ptr() % 16 == 0):
        return None
    try:
        if G2_PRT_MODE.get(p1_bucket(tokens)) != "fused":
            return None
        return _g2_prt_run(g2, p1["proj"]["pooler"], ffn_input, router_input, state, data, 1 if G2_PRT_PDL else 0)
    except Exception as ex:  # noqa: BLE001
        g2["prt"] = False
        print(f"[g2] G2-A raised at call time, k3_v18 path from now on: {ex!r}", flush=True)
        return None


def _g2_selftest_prt(state, g2):
    """G2-A vs the k3_v18 path it replaces (router32 partials + P1 pooler + v8 tail) on FP32 router inputs for
    G2_SELF_TEST_CASES (T = 1): xn / topk weights max abs diff, ids and local ids equal, the ticket counter back at zero, a repeat
    bit-identical, PDL on and off. Returns (ok, worst xn rel diff, worst weight diff). Prepare-time only."""
    dev = state["router"].device
    P = state["p1"]["proj"]["pooler"]
    worst_x = worst_w = 0.0
    ok = True
    for T, ls, seed in G2_SELF_TEST_CASES:
        gen = torch.Generator(device=dev).manual_seed(seed)
        h = torch.randn((T, HIDDEN), generator=gen, device=dev)
        h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
        hb = h.to(torch.bfloat16).contiguous()
        d = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
        part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=dev)
        state["v8"]["mod"].v8_router32(h, state["router"], part)
        pool = p1_gemv(state["p1"], "pooler", hb)
        ref = _v8_tail(pool, part, state, d)
        pdls = (1, 0, 1, 0)
        outs = [_g2_prt_run(g2, P, hb, h, state, d, pdl) for pdl in pdls]
        torch.cuda.synchronize()
        cnt = int(g2["tcnt"][0])
        for k, got in enumerate(outs):
            ex = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
            ew = float((got[1] - ref[1]).abs().max())
            ids = bool(torch.equal(got[2], ref[2])) and bool(torch.equal(got[3], ref[3]))
            worst_x, worst_w = max(worst_x, ex), max(worst_w, ew)
            good = ex < 1e-3 and ew < 1e-6 and ids and bool(torch.isfinite(got[0].float()).all()) and cnt == 0
            if k >= 2:
                good &= all(bool(torch.equal(a, b)) for a, b in zip(outs[k - 2], outs[k]))
            print(f"[g2] G2-A self-test T={T} ls={ls} call {k} (pdl {pdls[k]}): xn rel {ex:.2e} (bitwise {bool(torch.equal(got[0], ref[0]))}) "
                  f"w maxdiff {ew:.1e} ids/lids equal {ids} ticket {cnt} -> {'ok' if good else 'FAIL'}", flush=True)
            ok &= good
    return ok, worst_x, worst_w


def _g2_selftest_p1pdl(state):
    """G2-B: each P1 projection routed at T = 1 / 8 gives the same bits with pdl = 1 as with pdl = 0 (eager; the hazard
    side is audited on the source by tools/gen_k3_v19.py). Prepare-time only."""
    st = state["p1"]
    ok = True
    gen = torch.Generator(device=state["router"].device).manual_seed(717)
    for proj in sorted(st["proj"]):
        for T in (1, 8):
            geo = p1_route(st, proj, T)
            if geo is None:
                continue
            P = st["proj"][proj]
            x = torch.randn((T, P["K"]), generator=gen, device=state["router"].device).to(torch.bfloat16)
            n_rows = P["rows_t8"]
            a = torch.empty((T, n_rows), dtype=torch.bfloat16, device=x.device)
            b = torch.empty_like(a)
            mode = P["mode_t8"]
            p1_call(st["mod"], P, x, a, n_rows, a.stride(0), geo, pdl=0, mode=mode)
            p1_call(st["mod"], P, x, b, n_rows, b.stride(0), geo, pdl=1, mode=mode)
            torch.cuda.synchronize()
            same = bool(torch.equal(a, b))
            print(f"[g2] G2-B self-test {proj} T={T}: pdl 1 == pdl 0 bitwise {same}", flush=True)
            ok &= same
    return ok


def _g2_prepare(state, t0):
    """Build G2 (P1's source + the G2-A kernel, T = 1 instantiation only), check occupancy (2 CTAs/SM so the 256 CTAs are one
    wave), self-test G2-A and G2-B against k3_v18's path. state["g2"] = None (k3_v18 everywhere) when G2_ENABLE is off, prepare is late, the build
    or a prerequisite (P1 pooler pack + route, the v8 tail, the FP32 router route) is missing; g2["prt"] / G2-B's
    state["g2_p1pdl"] record each self-test. Prepare-time only."""
    state["g2"] = None
    state["g2_p1pdl"] = False
    _G2_LIVE["p1pdl"] = False
    print(f"[g2] levers: G2_ENABLE={G2_ENABLE} G2_PRT={G2_PRT} G2_PRT_PDL={G2_PRT_PDL} G2_P1_PDL={G2_P1_PDL} "
          f"G2_PRT_MODE={G2_PRT_MODE}", flush=True)
    if not G2_ENABLE:
        return None
    if time.time() - t0 > G2_PREPARE_BY_S:
        print(f"[g2] prepare at {time.time() - t0:.0f}s > {G2_PREPARE_BY_S:.0f}s: G2 not built (k3_v18 path)", flush=True)
        return None
    t1 = time.time()
    if state.get("p1") is not None:
        try:
            state["g2_p1pdl"] = _g2_selftest_p1pdl(state)
        except Exception as ex:  # noqa: BLE001
            print(f"[g2] G2-B self-test raised {ex!r}", flush=True)
            state["g2_p1pdl"] = False
    _G2_LIVE["p1pdl"] = bool(state["g2_p1pdl"])
    print(f"[g2] G2-B {'live' if _G2_LIVE['p1pdl'] else 'off (no P1 or self-test failed)'}", flush=True)
    p1 = state.get("p1")
    if (p1 is None or "pooler" not in p1["proj"] or p1["proj"]["pooler"]["mode_t8"] != 0 or state.get("v8") is None
            or not p1["route"].get(("pooler", 1))):
        print("[g2] G2-A prerequisites missing (P1 pooler pack / route at T = 1, v8 tail): k3_v18 path", flush=True)
        return None
    try:
        mod = _g2_build()
        info = {(1, 1): mod.g2_info(1, 1)}
        print(f"[g2] prt_kernel regs / local bytes / CTAs per SM / dyn smem: {info}", flush=True)
        g2 = {"mod": mod, "tcnt": torch.zeros((4,), dtype=torch.int32, device=state["router"].device), "prt": False}
        if any(v[2] < 2 for v in info.values()):
            print("[g2] G2-A below 2 CTAs/SM (the 256 CTAs would not be one wave): k3_v18 path", flush=True)
        else:
            ok, ex, ew = _g2_selftest_prt(state, g2)
            g2["prt"] = ok
            g2["selftest"] = (ex, ew)
            print(f"[g2] G2-A {'live' if ok else 'FAILED: k3_v18 path'} (worst xn rel {ex:.2e}, weight diff {ew:.1e})", flush=True)
        state["g2"] = g2
    except Exception as ex:  # noqa: BLE001
        print(f"[g2] build / self-test FAILED -> k3_v18 path: {ex!r}", flush=True)
        state["g2"] = None
    print(f"[g2] ready in {time.time() - t1:.1f}s (prepare so far {time.time() - t0:.1f}s)", flush=True)
    return state["g2"]

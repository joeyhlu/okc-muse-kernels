# okc: problem=muse_attention_decode
# nonce ax3_3 (vfa_gc16as + AX3 4-warp pipelined sliding decode B16/32, tools/ax3_patches.py; route {(32, 1): (8, 10)}, insts ((8, 2), (8, 10), (16, 10), (8, 0)), sweep)

from typing import TypedDict

import hashlib

import threading

import time

import types

import torch

import torch.nn.functional as F

import triton

from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sgl_kernel.scalar_type import scalar_types

from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul

from sglang.kernels.ops.kvcache.kvcache import store_cache

from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm

from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting

from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate

from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm

from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)

from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm

from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)

from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

try:
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        _post_expert_norm_weighted_sum_kernel as _POSTNORM_KERNEL,
    )
except ImportError:
    _POSTNORM_KERNEL = None

V10_COMBINED_BUILD = True

USE_K1 = True

K1_ROUTE_MIN_BATCH = {('sliding', 1): 1, ('full', 1): 1, ('full', 8): 16}

K1_FOLD_MIN_BATCH = {("sliding", 1): 1, ("sliding", 8): 1, ("full", 1): 8, ("full", 8): 8}

K1_MIN_CHUNK_T = {("sliding", 1): 2, ("sliding", 8): 4, ("full", 1): 2, ("full", 8): 2}

K1_CPS_T = {("sliding", 1): 1, ("sliding", 8): 2, ("full", 1): 2, ("full", 8): 2}

K1_STAGES_T = {("sliding", 1): 4, ("sliding", 8): 3, ("full", 1): 4, ("full", 8): 3}

K1_SELF_TEST = True

R1_QK_FUSED_FP8_STAGE = True

NUM_SPLITS_OVERRIDE = {}

STAGE_LOC_MAX = 1 << 16

RMS_EPS = 1e-5

FP8 = torch.float8_e4m3fn

ROPE_POSITIONS = 1 << 21

ROPE_THETA = 500_000.0

K1_STAGES = 4

K1_COUNTERS = 1 << 16

def _k1_nsg(B):
    return 32 if B == 1 else (8 if B <= 4 else 4)

K1_SELF_TEST_CASES = (
    (2, 3000, 1, True, 1, 1.0, 101, (16, 2)),
    (3, 2047, 8, True, 1, 1.0, 102, (16, 2)),
    (2, 3000, 8, False, 64, 1.0, 103, (16, 2)),
    (2, 2500, 1, False, 64, 1.0 / 64, 104, (16, 2)),
    (2, 1800, 8, True, 1, 1.0 / 64, 105, (16, 2)),
    (2, 20, 8, False, 1, 1.0, 106, (16, 2)),
    (1, 9, 1, True, 64, 1.0, 107, (16, 2)),
    (2, 600, 8, False, 64, 1.0, 108, (128, 16)),
    (3, 1500, 1, False, 64, 1.0, 109, (128, 16)),
    (1, 9000, 1, False, 1, 1.0, 110, (16, 2)),
    (1, 5000, 8, False, 1, 1.0, 111, (16, 2)),
)

K1_CPP_SRC = r"""
#include <torch/extension.h>
void k1_forward(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor page_table,
                torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
                torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor out,
                torch::Tensor counters, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t W,
                int64_t page_shift, int64_t Gx, int64_t min_chunk, int64_t use_rope, double qk_sm, int64_t fold,
                int64_t nsg, int64_t stages);
int64_t k1_occupancy(int64_t L, int64_t ST);
torch::Tensor k1_prof_read();
"""

K1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#define K1_ARCH_OK 1
#else
#define K1_ARCH_OK 0
#endif

namespace k1 {

constexpr int TILE = 64;                       // keys per pipeline stage
constexpr int NEWK = 16;                       // keys in the new-token tile
constexpr int ROWB = 128;                      // smem bytes per key per K (or V): 2 heads x 64 dims x 1 B
constexpr int NWARP = 8;
constexpr int NTHR = NWARP * 32;
constexpr int STAGE_BYTES = 2 * TILE * ROWB;   // K tile then V tile
constexpr int VH_BYTES = TILE * 2 * 64 * 2;    // L=8: one tile of V as fp16 [key][head][64] (16 KB)
constexpr int NEW_BYTES = 2 * NEWK * ROWB;
constexpr int AUX_BYTES = 2048;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float P_OFFSET = 8.0f;               // p = exp2(s - m + 8) in (0, 256]
constexpr unsigned FULL = 0xffffffffu;
static_assert(NEW_BYTES == NTHR * 16, "new-tile zeroing assumes one uint4 per thread");

constexpr int VT_SBO = 272;                    // V^T: row-group stride (bank spread, k1_wg_emulate)
constexpr int VT_REGION = 2448;                // V^T: per (s, h) region of 9 row groups (8 dims + 1 ones group), == 16 mod 128
constexpr int VT_BYTES = 4 * VT_REGION;
constexpr int NEW8_BYTES = 8192 + VT_BYTES;
template <int L> constexpr int qs_bytes() { return (L == 1) ? 2 * 8 * 64 * 2 : 2 * 64 * 64; }
template <int L> constexpr int cs_bytes() { return L * 32 * 8; }   // (cos, sin) per (token, pair)
template <int L> constexpr int new_bytes() { return (L == 1) ? NEW_BYTES : NEW8_BYTES; }
template <int L, int ST> constexpr int ring_bytes() { return ST * STAGE_BYTES; }
template <int L, int ST> constexpr int off_vh() { return ring_bytes<L, ST>(); }
template <int L, int ST> constexpr int off_new() { return off_vh<L, ST>() + ((L == 8) ? 3 * VT_BYTES : 0); }   // v6: 3 V^T buffers
template <int L, int ST> constexpr int off_qs() { return off_new<L, ST>() + new_bytes<L>(); }
template <int L, int ST> constexpr int off_cs() { return off_qs<L, ST>() + qs_bytes<L>(); }
template <int L, int ST> constexpr int off_aux() { return off_cs<L, ST>() + cs_bytes<L>(); }
template <int L, int ST> constexpr int off_plan() { return off_aux<L, ST>() + ((L == 8) ? 0 : AUX_BYTES); }
static_assert(NEW8_BYTES % 16 == 0 && VT_REGION % 16 == 0, "16-B aligned wgmma operands");
template <int L, int ST> constexpr int smem_bytes() { return off_plan<L, ST>() + 64; }

template <int L> __device__ __forceinline__ int kswz(int x, int ch) {
  return (L == 1) ? (ch ^ (4 * (x & 1))) : (ch ^ (x & 1));
}
template <int L> __device__ __forceinline__ int vswz(int x, int ch) {
  return (L == 1) ? (ch ^ (2 * ((x >> 1) & 3))) : (ch ^ (x & 1));
}

struct Params {
  const __nv_bfloat16* qkv;   // [B*L, (2Hq+2Hkv)*64]
  const int* seq_lens;        // [B]
  const int64_t* cache_loc;   // [B*L]
  const int* page_table;      // [B, pt_stride]
  uint8_t* k_cache;           // e4m3 bytes, rows of Hkv*64
  uint8_t* v_cache;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;      // [32] (RoPE only)
  float* part_o;              // [Hkv/2, Gx, 16L, 64]
  float* part_lse;            // [Hkv/2, Gx, 16L]
  __nv_bfloat16* out;         // [B*L, Hq*64]
  int* counters;              // [B * Hkv/2] arrival counters (fold mode); zero between calls (self-resetting)
  int B, Hq, Hkv, W, page_shift, pt_stride, Gx, min_chunk, use_rope, fold;
  float qk_sm;                // qk_scale_factor * sm_scale
};

#ifndef K1_PROF
#define K1_PROF 0                                    // 1: per-CTA %globaltimer phase stamps into k1_prof (lab only)
#endif
#if K1_PROF
__device__ unsigned long long k1_prof[4096 * 16];
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define K1P(slot) do { if (threadIdx.x == 0 && blockIdx.y * gridDim.x + blockIdx.x < 4096) \
    k1_prof[(blockIdx.y * gridDim.x + blockIdx.x) * 16 + (slot)] = gtime(); } while (0)
#else
#define K1P(slot) do {} while (0)
#endif

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(FULL, v, d);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, d));
  return v;
}

__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t v) {
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xFFFFu), __NV_E4M3);
  return (uint32_t)r.x | ((uint32_t)r.y << 16);
}

__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
#if K1_ARCH_OK
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma16832_e4m3(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1) {
#if K1_ARCH_OK
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
#if K1_ARCH_OK
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gptr), "r"(src_bytes) : "memory");
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if K1_ARCH_OK
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

__device__ __forceinline__ uint64_t desc_il(const void* ptr, int lbo, int sbo) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(ptr);
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ int il_off(int r, int k, int lbo, int sbo) {
  return (r & 7) * 16 + (r >> 3) * sbo + (k & 15) + (k >> 4) * lbo;
}
#define K1_D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define K1_R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
__device__ __forceinline__ void wg_ss(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
#if K1_ARCH_OK
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " K1_R32 ", %32, %33, p, 1, 1;\n}\n"
               : K1_D32(d) : "l"(da), "l"(db), "r"(scale_d));
#endif
}
__device__ __forceinline__ void wg_rs(float (&d)[32], const uint32_t (&a)[4], uint64_t db) {
#if K1_ARCH_OK
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " K1_R32 ", {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
               : K1_D32(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
#endif
}
#define K1_D36(d) K1_D32(d),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35])
#define K1_R36 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}"
__device__ __forceinline__ void wg_rs72(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {
#if K1_ARCH_OK
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 " K1_R36 ", {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : K1_D36(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
#endif
}
__device__ __forceinline__ uint32_t pk4f(float a, float b, float c, float d) {
  uint32_t r;
  asm("{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\ncvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n}\n" : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t pk4_e4m3(float a, float b, float c, float d) {
  const uint32_t lo = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}
__device__ __forceinline__ int vt_off(int h, int s, int n, int kl) { return (2 * s + h) * VT_REGION + il_off(n, kl, 128, VT_SBO); }
__device__ __forceinline__ void key_to_sk(int x, int& s, int& kl) {
  const int q = x >> 4, b = (x >> 3) & 1, t = (x >> 1) & 3, e = x & 1;
  s = q >> 1;
  kl = 16 * (q & 1) + 4 * t + 2 * b + e;
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if K1_ARCH_OK
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

__device__ __forceinline__ float2 rope_tab(float2 x, float2 cs) {
  return make_float2(x.x * cs.x - x.y * cs.y, x.x * cs.y + x.y * cs.x);
}

__device__ __forceinline__ float ldf(const float* a, bool cg) { return cg ? __ldcg(a) : *a; }
__device__ __forceinline__ float4 ldf4(const float* a, bool cg) {
  return cg ? __ldcg(reinterpret_cast<const float4*>(a)) : *reinterpret_cast<const float4*>(a);
}

__device__ __forceinline__ float sigmoidf_(float g) { return 1.f / (1.f + expf(-g)); }

__device__ __forceinline__ int old_tiles(int seq, int W) {
  const int n_old = W ? min(seq, W - 1) : seq;
  return (max(n_old, 0) + TILE - 1) / TILE;
}

__device__ __forceinline__ int split_count(int nt, int C) { return max(1, (nt + C - 1) / C); }

__device__ __forceinline__ int warp_sum_i(int v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(FULL, v, d);
  return v;
}

__device__ int plan_C(const int* __restrict__ seq_lens, int B, int W, int Gx, int min_chunk) {
  const int lane = threadIdx.x & 31;
  int T = 0;
  for (int b0 = 0; b0 < B; b0 += 32) {
    const int b = b0 + lane;
    if (b < B) T += old_tiles(__ldg(seq_lens + b), W);
  }
  T = warp_sum_i(T);
  int lo = max(min_chunk, (T + Gx - 1) / Gx);
  const int ge = max(1, Gx - B);
  int hi = max(lo, (T + ge - 1) / ge);
  while (lo < hi) {                                  // warp-uniform
    const int mid = (lo + hi) >> 1;
    int cnt = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      if (b < B) cnt += split_count(old_tiles(__ldg(seq_lens + b), W), mid);
    }
    cnt = warp_sum_i(cnt);
    if (cnt <= Gx) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

__device__ void plan_find(const int* __restrict__ seq_lens, int B, int W, int C, int key, bool by_cta,
                          int (&res)[5]) {
  const int lane = threadIdx.x & 31;
  int carry = 0;
  res[0] = -1; res[1] = 0; res[2] = 0; res[3] = 0; res[4] = 0;
  for (int b0 = 0; b0 < B; b0 += 32) {
    const int b = b0 + lane;
    const int nt = (b < B) ? old_tiles(__ldg(seq_lens + b), W) : 0;
    const int S = (b < B) ? split_count(nt, C) : 0;
    int incl = S;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int v = __shfl_up_sync(FULL, incl, d);
      if (lane >= d) incl += v;
    }
    const int off = carry + incl - S;
    const bool hit = (b < B) && (by_cta ? (off <= key && key < off + S) : (b == key));
    const unsigned m = __ballot_sync(FULL, hit);
    if (m) {   // warp-uniform
      const int src = __ffs(m) - 1;
      res[0] = __shfl_sync(FULL, b, src);
      res[1] = __shfl_sync(FULL, key - off, src);
      res[2] = __shfl_sync(FULL, S, src);
      res[3] = __shfl_sync(FULL, off, src);
      res[4] = __shfl_sync(FULL, nt, src);
    }
    carry += __shfl_sync(FULL, incl, 31);
  }
}

__device__ __forceinline__ uint2 vld(const uint8_t* Vbuf, int x, int h, int g) {
  return *reinterpret_cast<const uint2*>(Vbuf + x * ROWB + 16 * ((4 * h + (g >> 1)) ^ (2 * ((x >> 1) & 3))) +
                                         8 * (g & 1));
}

__device__ __forceinline__ void v_frags(const uint8_t* Vbuf, int x0, int h, int g, uint32_t (&w01)[4],
                                        uint32_t (&w23)[4]) {
  const uint2 va = vld(Vbuf, x0, h, g);
  const uint2 vb = vld(Vbuf, x0 + 1, h, g);
  const uint2 vc = vld(Vbuf, x0 + 8, h, g);
  const uint2 vd = vld(Vbuf, x0 + 9, h, g);
  w01[0] = __byte_perm(va.x, vb.x, 0x5140); w01[1] = __byte_perm(va.x, vb.x, 0x7362);
  w01[2] = __byte_perm(va.y, vb.y, 0x5140); w01[3] = __byte_perm(va.y, vb.y, 0x7362);
  w23[0] = __byte_perm(vc.x, vd.x, 0x5140); w23[1] = __byte_perm(vc.x, vd.x, 0x7362);
  w23[2] = __byte_perm(vc.y, vd.y, 0x5140); w23[3] = __byte_perm(vc.y, vd.y, 0x7362);
}

template <int NT>
__device__ __forceinline__ void tile_compute1(const uint8_t* __restrict__ Kbuf, const uint8_t* __restrict__ Vbuf,
                                              const int kw, const int h, const int g, const int t,
                                              const uint32_t (&qf)[4][4], float (&o)[8][4], float& mA, float& lA,
                                              const float cl2, const bool need, const int tile_abs,
                                              const int key_lim, const int seq, const int W) {
  float sc[NT][4];
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) {
    const int key = kw + 8 * jn + g;
    const uint4 kv = *reinterpret_cast<const uint4*>(Kbuf + key * ROWB + 16 * ((4 * h + t) ^ (4 * (key & 1))));
    const uint32_t wv[4] = {kv.x, kv.y, kv.z, kv.w};
    sc[jn][0] = 0.f; sc[jn][1] = 0.f; sc[jn][2] = 0.f; sc[jn][3] = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s)
      mma16816(sc[jn], qf[s][0], 0u, qf[s][2], 0u, e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
  }
  if (need) {
#pragma unroll
    for (int jn = 0; jn < NT; ++jn) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ak = tile_abs + kw + 8 * jn + 2 * t + e;
        const bool ok = (ak < key_lim) && (ak <= seq) && (W == 0 || ak > seq - W);
        if (!ok) sc[jn][e] = neg_inf();
      }
    }
  }
  float xA = neg_inf();
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) xA = fmaxf(xA, fmaxf(sc[jn][0], sc[jn][1]));
  xA = fmaxf(xA, __shfl_xor_sync(FULL, xA, 1));
  xA = fmaxf(xA, __shfl_xor_sync(FULL, xA, 2));
  const float nA = fmaxf(mA, xA * cl2);
  const float uA = (nA == neg_inf()) ? 0.f : nA;
  const float aA = ex2(mA - uA);
  const float bA = uA - P_OFFSET;
  mA = nA;
  uint32_t pa[NT / 2][2];   // reg0 / reg2 of the A operand (rows g); rows g+8 are zero
  float psA = 0.f;
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk) {
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int jn = 2 * kk + hf;
      const float p0 = ex2(fmaf(sc[jn][0], cl2, -bA));
      const float p1 = ex2(fmaf(sc[jn][1], cl2, -bA));
      psA += p0 + p1;
      pa[kk][hf] = pack_h2(p0, p1);
    }
  }
  lA = lA * aA + psA;
  if (__any_sync(FULL, aA != 1.f)) {
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] *= aA; o[jn][1] *= aA; }
  }
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk) {
    uint32_t w01[4], w23[4];
    v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01, w23);
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      const uint32_t s01 = w01[jn >> 1], s23 = w23[jn >> 1];
      const uint32_t b0 = e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01);
      const uint32_t b1 = e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23);
      mma16816(o[jn], pa[kk][0], 0u, pa[kk][1], 0u, b0, b1);
    }
  }
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const uint8_t* p) {
#if K1_ARCH_OK
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
#endif
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const uint8_t* p) {
#if K1_ARCH_OK
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
#endif
}

__device__ __forceinline__ int vh_off(int x, int hh, int c16) { return x * 256 + hh * 128 + 16 * (c16 ^ (x & 7) ^ (2 * hh)); }

__device__ __forceinline__ void conv_chunk(const uint4 w, uint8_t* dst, int x, int ch) {
  uint4 a, c;
  a.x = e4m3x2_to_h2(w.x); a.y = e4m3x2_to_h2(w.x >> 16); a.z = e4m3x2_to_h2(w.y); a.w = e4m3x2_to_h2(w.y >> 16);
  c.x = e4m3x2_to_h2(w.z); c.y = e4m3x2_to_h2(w.z >> 16); c.z = e4m3x2_to_h2(w.w); c.w = e4m3x2_to_h2(w.w >> 16);
  const int hh = ch >> 2, c0 = 2 * (ch & 3);
  *reinterpret_cast<uint4*>(dst + vh_off(x, hh, c0)) = a;
  *reinterpret_cast<uint4*>(dst + vh_off(x, hh, c0 + 1)) = c;
}

__device__ __forceinline__ void conv_kv(const uint8_t* ksrc, const uint8_t* vsrc, uint8_t* kh, uint8_t* vh, int x,
                                        int part) {
  uint4 wk[2], wv[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int ch = 2 * part + u;
    wk[u] = *reinterpret_cast<const uint4*>(ksrc + x * ROWB + 16 * (ch ^ (x & 1)));
    wv[u] = *reinterpret_cast<const uint4*>(vsrc + x * ROWB + 16 * (ch ^ (x & 1)));
  }
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    conv_chunk(wk[u], kh, x, 2 * part + u);
    conv_chunk(wv[u], vh, x, 2 * part + u);
  }
}

template <int NT>
__device__ __forceinline__ void tile_compute8(const uint8_t* __restrict__ Kh, const uint8_t* __restrict__ Vh,
                                              const int h, const int g, const int t, const int lane,
                                              const uint32_t (&qf)[4][4], float (&o)[8][4], float (&m)[2],
                                              float (&l)[2], const float cl2, const bool need, const int tile_abs,
                                              const int key_lim, const int seq, const int W, const int j0,
                                              const int kb0) {
  float sc[NT][4];
  const int kx = lane & 7, kq = lane >> 3;
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) {
    sc[jn][0] = 0.f; sc[jn][1] = 0.f; sc[jn][2] = 0.f; sc[jn][3] = 0.f;
#pragma unroll
    for (int hs = 0; hs < 2; ++hs) {
      uint32_t kb[4];
      ldsm_x4(kb, Kh + vh_off(kb0 + 8 * jn + kx, h, 4 * hs + kq));
      mma16816(sc[jn], qf[2 * hs][0], qf[2 * hs][1], qf[2 * hs][2], qf[2 * hs][3], kb[0], kb[1]);
      mma16816(sc[jn], qf[2 * hs + 1][0], qf[2 * hs + 1][1], qf[2 * hs + 1][2], qf[2 * hs + 1][3], kb[2], kb[3]);
    }
  }
  if (need) {
#pragma unroll
    for (int jn = 0; jn < NT; ++jn) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ak = tile_abs + kb0 + 8 * jn + 2 * t + e;
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int j = j0 + rb;
          const bool ok = (ak < key_lim) && (ak <= seq + j) && (W == 0 || ak > seq + j - W);
          if (!ok) sc[jn][2 * rb + e] = neg_inf();
        }
      }
    }
  }
  float a[2];
  bool resc = false;
  uint32_t pa[NT / 2][4];
#pragma unroll
  for (int rb = 0; rb < 2; ++rb) {
    float x = neg_inf();
#pragma unroll
    for (int jn = 0; jn < NT; ++jn) x = fmaxf(x, fmaxf(sc[jn][2 * rb], sc[jn][2 * rb + 1]));
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
    const float n = fmaxf(m[rb], x * cl2);
    const float u = (n == neg_inf()) ? 0.f : n;
    a[rb] = ex2(m[rb] - u);
    resc |= (a[rb] != 1.f);
    const float bo = u - P_OFFSET;
    m[rb] = n;
    float ps = 0.f;
#pragma unroll
    for (int kk = 0; kk < NT / 2; ++kk) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int jn = 2 * kk + hf;
        const float p0 = ex2(fmaf(sc[jn][2 * rb], cl2, -bo));
        const float p1 = ex2(fmaf(sc[jn][2 * rb + 1], cl2, -bo));
        ps += p0 + p1;
        pa[kk][2 * hf + rb] = pack_h2(p0, p1);   // a0 row g keys 2t, a1 row g+8, a2/a3: keys 8+2t
      }
    }
    l[rb] = l[rb] * a[rb] + ps;
  }
  if (__any_sync(FULL, resc)) {
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      o[jn][0] *= a[0]; o[jn][1] *= a[0];
      o[jn][2] *= a[1]; o[jn][3] *= a[1];
    }
  }
  const int lx = (lane & 7) + 8 * ((lane >> 3) & 1), lc = lane >> 4;
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk) {
#pragma unroll
    for (int jp = 0; jp < 4; ++jp) {
      uint32_t bv[4];
      ldsm_x4_t(bv, Vh + vh_off(kb0 + 16 * kk + lx, h, 2 * jp + lc));
      mma16816(o[2 * jp], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], bv[0], bv[1]);
      mma16816(o[2 * jp + 1], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], bv[2], bv[3]);
    }
  }
}

#ifndef K1_DIAG6
#define K1_DIAG6 0                                   // lab: 1 no V^T transpose, 2 no softmax math, 3 no wgmma
#endif
#ifndef K1WG_QK_ONLY
#define K1WG_QK_ONLY 0                               // lab (stage B): 1 = PV wgmma stubbed
#endif
__device__ __forceinline__ void tile_wg(float (&S)[32], float (&O)[32], float (&m)[2], float (&l)[2], const uint64_t dq,
                                        const uint64_t dk, const uint8_t* __restrict__ vt, const int h, const int t,
                                        const float cl2, const bool need, const int tile_abs, const int key_lim,
                                        const int seq, const int W, const int j0) {
  wg_fence();
  wg_ss(S, dq, dk, 0);
  wg_ss(S, dq + 16, dk + 16, 1);                      // +256 B: head dims 32..63
  wg_commit();
  wg_wait0();
  if (need) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ak = tile_abs + 8 * j + 2 * t + e;
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int jj = j0 + rb;
          const bool ok = (ak < key_lim) && (ak <= seq + jj) && (W == 0 || ak > seq + jj - W);
          if (!ok) S[4 * j + 2 * rb + e] = neg_inf();
        }
      }
    }
  }
  float a[2];
  bool resc = false;
#pragma unroll
  for (int rb = 0; rb < 2; ++rb) {
    float x = neg_inf();
#pragma unroll
    for (int j = 0; j < 8; ++j) x = fmaxf(x, fmaxf(S[4 * j + 2 * rb], S[4 * j + 2 * rb + 1]));
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
    const float n = fmaxf(m[rb], x * cl2);
    const float u = (n == neg_inf()) ? 0.f : n;
    a[rb] = ex2(m[rb] - u);
    resc |= (a[rb] != 1.f);
    const float bo = u - P_OFFSET;
    m[rb] = n;
    float ps = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float pv = ex2(fmaf(S[4 * j + 2 * rb + e], cl2, -bo));
        S[4 * j + 2 * rb + e] = pv;
        ps += pv;
      }
    }
    l[rb] = l[rb] * a[rb] + ps;
  }
  if (__any_sync(FULL, resc)) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      O[4 * j + 0] *= a[0]; O[4 * j + 1] *= a[0];
      O[4 * j + 2] *= a[1]; O[4 * j + 3] *= a[1];
    }
  }
  uint32_t pa[2][4];
#pragma unroll
  for (int st = 0; st < 2; ++st) {
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const int q = 2 * st + h2, ja = 2 * q, jb = 2 * q + 1;
      pa[st][2 * h2 + 0] = pk4_e4m3(S[4 * ja + 0], S[4 * ja + 1], S[4 * jb + 0], S[4 * jb + 1]);   // row g
      pa[st][2 * h2 + 1] = pk4_e4m3(S[4 * ja + 2], S[4 * ja + 3], S[4 * jb + 2], S[4 * jb + 3]);   // row g+8
    }
  }
#if K1WG_QK_ONLY
  O[0] += __uint_as_float((pa[0][0] ^ pa[1][3]) & 0x007fffffu);   // keep P live; PV stubbed
#else
  wg_fence();
  wg_rs(O, pa[0], desc_il(vt + (0 + h) * VT_REGION, 128, VT_SBO));
  wg_rs(O, pa[1], desc_il(vt + (2 + h) * VT_REGION, 128, VT_SBO));
  wg_commit();
  wg_wait0();
#endif
}

#ifndef K1_POLYJ
#define K1_POLYJ 0                                   // of the 8 key blocks per row and tile, how many use ex2_poly
#endif
__device__ __forceinline__ float ex2_poly(float x) {
  x = fmaxf(x, -126.f);
  const float t = __fadd_rn(x, 12582912.f);
  const float jf = __fadd_rn(t, -12582912.f);
  const float f = __fadd_rn(x, -jf);
  const float pp = fmaf(fmaf(fmaf(0.0550424707f, f, 0.242289932f), f, 0.693254439f), f, 0.999950127f);
  const int ji = __float_as_int(t) - 0x4B400000;
  return __int_as_float(__float_as_int(pp) + (ji << 23));
}
__device__ __forceinline__ void qk_issue(float (&S)[32], const uint64_t dq, const uint64_t dk) {
  if (K1_DIAG6 == 3) { S[0] = __uint_as_float((uint32_t)dk & 0x3f0fffffu); return; }
  wg_fence();
  wg_ss(S, dq, dk, 0);
  wg_ss(S, dq + 16, dk + 16, 1);
  wg_commit();
}
__device__ __forceinline__ void softmax_pack(float (&S)[32], float (&O)[32], float (&m)[2], float (&l)[2],
                                             uint32_t (&pa)[2][4], const int t, const float cl2, const bool need,
                                             const int tile_abs, const int key_lim, const int seq, const int W,
                                             const int j0) {
  if (K1_DIAG6 == 2) {
#pragma unroll
    for (int i = 0; i < 32; ++i) S[i] = fminf(S[i], 4.f);
    l[0] += S[0]; l[1] += S[2]; m[0] = 0.f; m[1] = 0.f;
#pragma unroll
    for (int st = 0; st < 2; ++st)
#pragma unroll
      for (int k4 = 0; k4 < 4; ++k4) pa[st][k4] = pk4_e4m3(S[16 * st + 4 * k4], S[16 * st + 4 * k4 + 1], S[16 * st + 4 * k4 + 2], S[16 * st + 4 * k4 + 3]);
    return;
  }
  if (need) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ak = tile_abs + 8 * j + 2 * t + e;
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int jj = j0 + rb;
          const bool ok = (ak < key_lim) && (ak <= seq + jj) && (W == 0 || ak > seq + jj - W);
          if (!ok) S[4 * j + 2 * rb + e] = neg_inf();
        }
      }
    }
  }
  float a[2];
  bool resc = false;
#pragma unroll
  for (int rb = 0; rb < 2; ++rb) {
    float x = neg_inf();
#pragma unroll
    for (int j = 0; j < 8; ++j) x = fmaxf(x, fmaxf(S[4 * j + 2 * rb], S[4 * j + 2 * rb + 1]));
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
    const float n = fmaxf(m[rb], x * cl2);
    const float u = (n == neg_inf()) ? 0.f : n;
    a[rb] = ex2(m[rb] - u);
    resc |= (a[rb] != 1.f);
    const float bo = u - P_OFFSET;
    m[rb] = n;
    float ps = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float xa = fmaf(S[4 * j + 2 * rb + e], cl2, -bo);
        const float pv = (j < K1_POLYJ) ? ex2_poly(xa) : ex2(xa);
        S[4 * j + 2 * rb + e] = pv;
        ps += pv;
      }
    }
    l[rb] = l[rb] * a[rb] + ps;
  }
  if (__any_sync(FULL, resc)) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      O[4 * j + 0] *= a[0]; O[4 * j + 1] *= a[0];
      O[4 * j + 2] *= a[1]; O[4 * j + 3] *= a[1];
    }
  }
#pragma unroll
  for (int st = 0; st < 2; ++st) {
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const int q = 2 * st + h2, ja = 2 * q, jb = 2 * q + 1;
      pa[st][2 * h2 + 0] = pk4_e4m3(S[4 * ja + 0], S[4 * ja + 1], S[4 * jb + 0], S[4 * jb + 1]);
      pa[st][2 * h2 + 1] = pk4_e4m3(S[4 * ja + 2], S[4 * ja + 3], S[4 * jb + 2], S[4 * jb + 3]);
    }
  }
}
__device__ __forceinline__ void pv_issue(float (&O)[32], const uint32_t (&pa)[2][4], const uint8_t* vt, const int h) {
  if (K1_DIAG6 == 3) { O[1] += __uint_as_float(pa[0][1] & 0x007fffffu); return; }
  wg_fence();
#if K1WG_QK_ONLY
  O[0] += __uint_as_float((pa[0][0] ^ pa[1][3]) & 0x007fffffu);
#else
  wg_rs(O, pa[0], desc_il(vt + (0 + h) * VT_REGION, 128, VT_SBO));
  wg_rs(O, pa[1], desc_il(vt + (2 + h) * VT_REGION, 128, VT_SBO));
#endif
  wg_commit();
}
__device__ __forceinline__ void softmax_pack7(float (&S)[32], float (&O)[36], float (&m)[2], uint32_t (&pa)[2][4],
                                              const int t, const float cl2, const bool need, const int tile_abs,
                                              const int key_lim, const int seq, const int W, const int j0) {
  if (need) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ak = tile_abs + 8 * j + 2 * t + e;
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int jj = j0 + rb;
          const bool ok = (ak < key_lim) && (ak <= seq + jj) && (W == 0 || ak > seq + jj - W);
          if (!ok) S[4 * j + 2 * rb + e] = neg_inf();
        }
      }
    }
  }
  float a[2];
  bool resc = false;
#pragma unroll
  for (int rb = 0; rb < 2; ++rb) {
    float v8[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v8[j] = fmaxf(S[4 * j + 2 * rb], S[4 * j + 2 * rb + 1]);
#pragma unroll
    for (int w = 4; w >= 1; w >>= 1)
#pragma unroll
      for (int j = 0; j < w; ++j) v8[j] = fmaxf(v8[j], v8[j + w]);
    float x = v8[0];
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
    x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
    const float n = fmaxf(m[rb], x * cl2);
    const float u = (n == neg_inf()) ? 0.f : n;
    a[rb] = ex2(m[rb] - u);
    resc |= (a[rb] != 1.f);
    const float bo = u - P_OFFSET;
    m[rb] = n;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) S[4 * j + 2 * rb + e] = ex2(fmaf(S[4 * j + 2 * rb + e], cl2, -bo));
    }
  }
  if (__any_sync(FULL, resc)) {
#pragma unroll
    for (int j = 0; j < 9; ++j) {
      O[4 * j + 0] *= a[0]; O[4 * j + 1] *= a[0];
      O[4 * j + 2] *= a[1]; O[4 * j + 3] *= a[1];
    }
  }
#pragma unroll
  for (int st = 0; st < 2; ++st) {
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const int q = 2 * st + h2, ja = 2 * q, jb = 2 * q + 1;
      pa[st][2 * h2 + 0] = pk4f(S[4 * ja + 0], S[4 * ja + 1], S[4 * jb + 0], S[4 * jb + 1]);
      pa[st][2 * h2 + 1] = pk4f(S[4 * ja + 2], S[4 * ja + 3], S[4 * jb + 2], S[4 * jb + 3]);
    }
  }
}
__device__ __forceinline__ void pv_issue7(float (&O)[36], const uint32_t (&pa)[2][4], const uint8_t* vt, const int h) {
  wg_fence();
  wg_rs72(O, pa[0], desc_il(vt + (0 + h) * VT_REGION, 128, VT_SBO));
  wg_rs72(O, pa[1], desc_il(vt + (2 + h) * VT_REGION, 128, VT_SBO));
  wg_commit();
}
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }

template <int L>
__device__ __forceinline__ void merge_finalize(const Params& p, const float* po, const float* pl, const int S, const int b,
                                               const int hp, const bool cg) {
  constexpr int ROWS = 16 * L, R8 = 8 * L, ITEMS = ROWS * 16, K = ITEMS / NTHR;
  constexpr int KG = 1, UN = (L == 1) ? 16 : 8;
  static_assert(ITEMS % NTHR == 0 && K % KG == 0, "uniform item loop (half-warp shuffles)");
  const float vsc = __ldg(p.v_scale);
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  const int r0 = threadIdx.x >> 4, u = threadIdx.x & 15;
  uint2 graw[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int r = r0 + 16 * k, hh = r / R8, rr = r - hh * R8, j = rr >> 3, gi = rr & 7;
    graw[k] = *reinterpret_cast<const uint2*>(p.qkv + ((size_t)b * L + j) * stride + (size_t)(p.Hq + 2 * p.Hkv) * 64 +
                                              (size_t)((2 * hp + hh) * 8 + gi) * 64 + 4 * u);
  }
#pragma unroll
  for (int k0 = 0; k0 < K; k0 += KG) {
    float M[KG], ws[KG];
    float4 acc[KG];
#pragma unroll
    for (int kg = 0; kg < KG; ++kg) { M[kg] = neg_inf(); ws[kg] = 0.f; acc[kg] = make_float4(0.f, 0.f, 0.f, 0.f); }
    for (int s0 = 0; s0 < S; s0 += UN) {
      float lv[KG][UN];
      float4 ov[KG][UN];
#pragma unroll
      for (int kg = 0; kg < KG; ++kg) {
        const int r = r0 + 16 * (k0 + kg);
#pragma unroll
        for (int q = 0; q < UN; ++q) {
          const int s = s0 + q;
          if (s < S) {
            lv[kg][q] = ldf(pl + (size_t)s * ROWS + r, cg);
            ov[kg][q] = ldf4(po + ((size_t)s * ROWS + r) * 64 + 4 * u, cg);
          } else {
            lv[kg][q] = neg_inf();
            ov[kg][q] = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
      }
#pragma unroll
      for (int kg = 0; kg < KG; ++kg) {
        float bm = M[kg];
#pragma unroll
        for (int q = 0; q < UN; ++q) bm = fmaxf(bm, lv[kg][q]);
        const float Mu = (bm == neg_inf()) ? 0.f : bm;
        const float sc = ex2(M[kg] - Mu);             // 0 when M was -inf
        acc[kg].x *= sc; acc[kg].y *= sc; acc[kg].z *= sc; acc[kg].w *= sc;
        ws[kg] *= sc;
#pragma unroll
        for (int q = 0; q < UN; ++q) {
          const float w = ex2(lv[kg][q] - Mu);
          ws[kg] += w;
          acc[kg].x = fmaf(w, ov[kg][q].x, acc[kg].x); acc[kg].y = fmaf(w, ov[kg][q].y, acc[kg].y);
          acc[kg].z = fmaf(w, ov[kg][q].z, acc[kg].z); acc[kg].w = fmaf(w, ov[kg][q].w, acc[kg].w);
        }
        M[kg] = bm;
      }
    }
#pragma unroll
    for (int kg = 0; kg < KG; ++kg) {
      const int k = k0 + kg, r = r0 + 16 * k;
      float4 a4 = acc[kg];
      const float scl = (ws[kg] > 0.f) ? vsc / ws[kg] : 0.f;
      a4.x *= scl; a4.y *= scl; a4.z *= scl; a4.w *= scl;
      float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
      ss += __shfl_xor_sync(FULL, ss, 8);
      ss += __shfl_xor_sync(FULL, ss, 4);
      ss += __shfl_xor_sync(FULL, ss, 2);
      ss += __shfl_xor_sync(FULL, ss, 1);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const int hh = r / R8, rr = r - hh * R8, j = rr >> 3, gi = rr & 7;
      const int qh = (2 * hp + hh) * 8 + gi;
      const size_t tok = (size_t)b * L + j;
      const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw[k].x));
      const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw[k].y));
      const __nv_bfloat162 o01 = __floats2bfloat162_rn(a4.x * rs * sigmoidf_(g01.x), a4.y * rs * sigmoidf_(g01.y));
      const __nv_bfloat162 o23 = __floats2bfloat162_rn(a4.z * rs * sigmoidf_(g23.x), a4.w * rs * sigmoidf_(g23.y));
      uint2 ow;
      ow.x = *reinterpret_cast<const uint32_t*>(&o01);
      ow.y = *reinterpret_cast<const uint32_t*>(&o23);
      *reinterpret_cast<uint2*>(p.out + tok * (size_t)p.Hq * 64 + (size_t)qh * 64 + 4 * u) = ow;
    }
  }
}

#ifndef K1_L8_MINB
#define K1_L8_MINB 2
#endif

__device__ __forceinline__ uint32_t va_sa(const void* x) { return (uint32_t)__cvta_generic_to_shared(x); }
__device__ __forceinline__ void va_wait(uint64_t* b, int par) {
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra W;\n}\n" ::"r"(va_sa(b)), "r"(par) : "memory");
#else
  uint32_t ok = 0;
  do asm volatile("{\n.reg .pred P;\nmbarrier.test_wait.parity.shared.b64 P, [%1], %2;\nselp.u32 %0, 1, 0, P;\n}\n"
                  : "=r"(ok) : "r"(va_sa(b)), "r"(par) : "memory");
  while (!ok);
#endif
}
__device__ __forceinline__ void va_red4(float* a, float f, float x, float y, float z, float w) {
#if __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(x * f), "f"(y * f), "f"(z * f), "f"(w * f) : "memory");
#else
  atomicAdd(a, x * f); atomicAdd(a + 1, y * f); atomicAdd(a + 2, z * f); atomicAdd(a + 3, w * f);
#endif
}
#define VA_CSYNC asm volatile("bar.sync 1, 128;\n" ::: "memory")
template <int ST> constexpr int va_smem() { return ST * STAGE_BYTES + NEW_BYTES + 2048 + 16 * ST + 4 * 1032; }

template <int ST>
__global__ void __launch_bounds__(256, 2) va1_main(const Params p) {
#if K1_ARCH_OK
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* newk = smem + ST * STAGE_BYTES;
  uint8_t* newv = newk + NEWK * ROWB;
  uint8_t* qs = newk + NEW_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(qs + 2048);
  uint64_t* empt = full + ST;
  int* flag = reinterpret_cast<int*>(empt + ST);
  int* sV = flag + 4;
  int* sN = sV + 513;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int hp = blockIdx.y, HP = gridDim.y, B = p.B, segk = p.min_chunk;
  reinterpret_cast<uint4*>(newk)[tid] = make_uint4(0u, 0u, 0u, 0u);
  if (tid < ST) {
    asm volatile("mbarrier.init.shared.b64 [%0], 128;\n" ::"r"(va_sa(full + tid)) : "memory");
    asm volatile("mbarrier.init.shared.b64 [%0], 4;\n" ::"r"(va_sa(empt + tid)) : "memory");
  }
  if (warp == 0) {
    int carry = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      const int n = (b < B) ? __ldg(p.seq_lens + b) + 1 : 0, w = (b < B) ? n + segk : 0;
      int incl = w;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int v = __shfl_up_sync(FULL, incl, d);
        if (lane >= d) incl += v;
      }
      if (b < B) { sV[b] = carry + incl - w; sN[b] = n; }
      carry += __shfl_sync(FULL, incl, 31);
    }
    if (lane == 0) sV[B] = carry;
  }
  __syncthreads();
#if __CUDA_ARCH__ >= 900
  if (p.use_rope) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // use_rope = PDL: o proj may start
#endif
  const long long T = (long long)sV[B] - segk;
  const int v0 = (int)((T * blockIdx.x) / gridDim.x), v1 = (int)((T * (blockIdx.x + 1)) / gridDim.x);
  int bf = 0;
  for (int hi = B - 1; bf < hi;) {
    const int mid = (bf + hi) >> 1;
    if (sV[mid] + sN[mid] > v0) hi = mid;
    else bf = mid + 1;
  }
  const size_t rowB = (size_t)p.Hkv * 64;
  int n = 0;   // ring tile counter, continuous across segments
  if (warp >= 4) {
    const int q = tid - 128, ch = q & 7, r0 = q >> 3, psh = p.page_shift, pm = (1 << psh) - 1;
    const uint8_t* kc = p.k_cache + (size_t)hp * 128 + 16 * ch;
    const uint8_t* vc = p.v_cache + (size_t)hp * 128 + 16 * ch;
    for (int b = bf; b < B && sV[b] < v1; ++b) {
      const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]), cend = min(k1, sN[b] - 1);
      if (k1 <= k0 || cend <= k0) continue;
      const int nt = (cend - k0 + TILE - 1) / TILE;
      const int* pt = p.page_table + (size_t)b * p.pt_stride;
      int pg[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) pg[k] = (k0 + r0 + 16 * k < cend) ? __ldg(pt + ((k0 + r0 + 16 * k) >> psh)) : -1;
      for (int x = 0; x < nt; ++x, ++n) {
        int pgn[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int pos = k0 + (x + 1) * TILE + r0 + 16 * k;
          pgn[k] = (pos < cend) ? __ldg(pt + (pos >> psh)) : -1;
        }
        const int st = n % ST;
        if (n >= ST) va_wait(empt + st, ((n / ST) - 1) & 1);
        uint8_t* stg = smem + st * STAGE_BYTES;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int r = r0 + 16 * k, pos = k0 + x * TILE + r;
          const size_t go = pg[k] >= 0 ? (size_t)((pg[k] << psh) | (pos & pm)) * rowB : (size_t)0;
          cp_async16(stg + r * ROWB + 16 * kswz<1>(r, ch), kc + go, pg[k] >= 0 ? 16 : 0);
          cp_async16(stg + TILE * ROWB + r * ROWB + 16 * vswz<1>(r, ch), vc + go, pg[k] >= 0 ? 16 : 0);
          pg[k] = pgn[k];
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(va_sa(full + st)) : "memory");
      }
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    return;
  }
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");   // PDL launches: qkv is written by the previous kernel
#endif
  const int h = warp & 1, kw = 32 * (warp >> 1), g = lane >> 2, t = lane & 3, nc = max(1, p.W);
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale), cl2 = p.qk_sm * ksc * LOG2E;
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  for (int b = bf; b < B && sV[b] < v1; ++b) {
    const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]);
    if (k1 <= k0) continue;
    const int seq = sN[b] - 1, cend = min(k1, seq);
    const int nt = (cend > k0) ? (cend - k0 + TILE - 1) / TILE : 0;
    const __nv_bfloat16* qrow = p.qkv + (size_t)b * stride;
#pragma unroll
    for (int k = 0; k < 4; ++k) {                    // Q row R = warp + 4k = KV head hh x query head gi, fp16
      const int R = warp + 4 * k;
      float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qrow + (size_t)((2 * hp + (R >> 3)) * 8 + (R & 7)) * 64 + 2 * lane));
      const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
      reinterpret_cast<uint32_t*>(qs)[R * 32 + lane] = pack_h2(x.x * rs, x.y * rs);
    }
    if (k1 == sN[b]) {                               // the new token: warp = kind (K, V) x KV head
      const int kind = warp >> 1, hh = warp & 1, head = 2 * hp + hh;
      float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(qrow + (size_t)(p.Hq + kind * p.Hkv + head) * 64 + 2 * lane));
      if (kind == 0) {
        const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
        x.x *= rs; x.y *= rs;
      }
      const float sc = kind ? vsc : ksc;
      const uint16_t q8 = (uint16_t)__nv_cvt_float2_to_fp8x2(make_float2(x.x / sc, x.y / sc), __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)p.cache_loc[b] * rowB + (size_t)head * 64 + 2 * lane) = q8;
      const int bb = 64 * hh + 2 * lane;
      *reinterpret_cast<uint16_t*>((kind ? newv : newk) + 16 * (kind ? vswz<1>(0, bb >> 4) : kswz<1>(0, bb >> 4)) + (bb & 15)) = q8;
    }
    VA_CSYNC;
    uint32_t qf[4][4];
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs) + (h * 8 + g) * 32;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      qf[s][0] = qa[8 * t + 2 * s]; qf[s][2] = qa[8 * t + 2 * s + 1]; qf[s][1] = 0u; qf[s][3] = 0u;
    }
    float o[8][4];
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] = 0.f; o[jn][1] = 0.f; o[jn][2] = 0.f; o[jn][3] = 0.f; }
    float mA = neg_inf(), lA = 0.f;
    for (int i = 0; i < nt; ++i, ++n) {
      const int st = n % ST, ts = k0 + i * TILE;
      va_wait(full + st, (n / ST) & 1);
      const uint8_t* Kb = smem + st * STAGE_BYTES;
      tile_compute1<4>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, ts + TILE > cend, ts, cend, seq, 0);
      __syncwarp();
      if (lane == 0) asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(va_sa(empt + st)) : "memory");
    }
    if (k1 == sN[b] && kw == 0) tile_compute1<2>(newk, newv, 0, h, g, t, qf, o, mA, lA, cl2, true, seq, seq + 1, seq, 0);
    lA += __shfl_xor_sync(FULL, lA, 1);
    lA += __shfl_xor_sync(FULL, lA, 2);
    const float f = (mA == neg_inf()) ? 0.f : ex2(fminf(mA - P_OFFSET, 126.f));   // to the absolute frame 2^0
    const size_t r0 = (size_t)(b * HP + hp) * nc * 16, rb = r0 + (blockIdx.x % nc) * 16;   // nc copies (W = nc)
    float* wa = p.part_o + (rb + h * 8 + g) * 64 + 16 * t;   // dims 16t + 8e + jn
    va_red4(wa, f, o[0][0], o[1][0], o[2][0], o[3][0]);
    va_red4(wa + 4, f, o[4][0], o[5][0], o[6][0], o[7][0]);
    va_red4(wa + 8, f, o[0][1], o[1][1], o[2][1], o[3][1]);
    va_red4(wa + 12, f, o[4][1], o[5][1], o[6][1], o[7][1]);
    if (t == 0) atomicAdd(p.part_lse + rb + h * 8 + g, lA * f);
    __threadfence();
    VA_CSYNC;
    if (tid == 0) flag[0] = (atomicAdd(p.counters + b * HP + hp, k1 - k0) + k1 - k0 == sN[b]) ? 1 : 0;
    VA_CSYNC;
    if (flag[0]) {                                   // last share of (b, hp): finalize, zero what was read
      __threadfence();
      const int r = tid >> 3, u = tid & 7, qh = (2 * hp + (r >> 3)) * 8 + (r & 7);
      float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0;
      float lt = 0.f;
      for (int c = 0; c < nc; ++c) {
        float* ar = p.part_o + (r0 + 16 * c + r) * 64 + 8 * u;
        const float4 x0 = __ldcg(reinterpret_cast<const float4*>(ar)), x1 = __ldcg(reinterpret_cast<const float4*>(ar + 4));
        a0.x += x0.x; a0.y += x0.y; a0.z += x0.z; a0.w += x0.w; a1.x += x1.x; a1.y += x1.y; a1.z += x1.z; a1.w += x1.w;
        lt += __ldcg(p.part_lse + r0 + 16 * c + r);
        *reinterpret_cast<float4*>(ar) = make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(ar + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (u == 0) p.part_lse[r0 + 16 * c + r] = 0.f;
      }
      const float sc = (lt > 0.f) ? vsc / lt : 0.f;
      const uint4 gr = *reinterpret_cast<const uint4*>(qrow + (size_t)(p.Hq + 2 * p.Hkv + qh) * 64 + 8 * u);
      const float y[8] = {a0.x * sc, a0.y * sc, a0.z * sc, a0.w * sc, a1.x * sc, a1.y * sc, a1.z * sc, a1.w * sc};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
      ss += __shfl_xor_sync(FULL, ss, 1);
      ss += __shfl_xor_sync(FULL, ss, 2);
      ss += __shfl_xor_sync(FULL, ss, 4);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      uint32_t ow[4] = {gr.x, gr.y, gr.z, gr.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 g2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ow[e]));
        const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * e] * rs * sigmoidf_(g2.x), y[2 * e + 1] * rs * sigmoidf_(g2.y));
        ow[e] = *reinterpret_cast<const uint32_t*>(&o2);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)b * p.Hq * 64 + (size_t)qh * 64 + 8 * u) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
      if (tid == 0) p.counters[b * HP + hp] = 0;
    }
  }
#endif
}

template <int L, int ST>
__global__ void __launch_bounds__(NTHR, (L == 1) ? 2 : K1_L8_MINB) k1_main(const Params p) {
#if K1_ARCH_OK
  constexpr int R8 = 8 * L;          // rows per KV head (row r = token*8 + head-in-group)
  constexpr int ROWS = 16 * L;       // rows per CTA (2 KV heads)
  constexpr int RPW = 2 * R8 / NWARP;   // Q rows prepared per warp
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* newk = smem + off_new<L, ST>();
  uint8_t* newv = newk + NEWK * ROWB;
  uint8_t* qs = smem + off_qs<L, ST>();
  float2* cst = reinterpret_cast<float2*>(smem + off_cs<L, ST>());
  int* plan_s = reinterpret_cast<int*>(smem + off_plan<L, ST>());

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x, hp = blockIdx.y;
  const int HP = gridDim.y;
  K1P(0);

  if constexpr (L == 1) {
    reinterpret_cast<uint4*>(newk)[tid] = make_uint4(0u, 0u, 0u, 0u);   // zero newk + newv
  } else {
    for (int i = tid; i < NEW8_BYTES / 16; i += NTHR) reinterpret_cast<uint4*>(newk)[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  if (warp == 0) {
    const int C = plan_C(p.seq_lens, p.B, p.W, p.Gx, p.min_chunk);
    int res[5];
    plan_find(p.seq_lens, p.B, p.W, C, c, true, res);
    if (lane == 0) {
      plan_s[0] = res[0]; plan_s[1] = res[1]; plan_s[2] = res[2]; plan_s[3] = res[4]; plan_s[4] = res[3];
    }
  }
  __syncthreads();
  K1P(1);
  const int b = plan_s[0];
  if (b < 0) return;                                  // idle CTA (uniform)
  if constexpr (L == 8) {
    const int buf = tid >> 6, rg = (tid >> 4) & 3, ch = tid & 15;   // 4 buffers x 4 regions x 16 chunks of 16 B
    uint8_t* base = (buf < 3) ? (smem + off_vh<L, ST>() + buf * VT_BYTES) : (newk + 8192);
    *reinterpret_cast<uint4*>(base + rg * VT_REGION + 8 * VT_SBO + (ch & 7) * 16 + (ch >> 3) * 128) =
        make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }
  const int s_idx = plan_s[1], S_b = plan_s[2], nt_b = plan_s[3], off_b = plan_s[4];
  const int t0 = (int)(((long long)s_idx * nt_b) / S_b);
  const int ntl = (int)(((long long)(s_idx + 1) * nt_b) / S_b) - t0;
  const bool last = (s_idx == S_b - 1);
  const int seq = __ldg(p.seq_lens + b);
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  if (L == 8 && s_idx == 0 && tid < L * 16) {        // warm L2 with the gate rows the merge will read (1 line each)
    const int j = tid >> 4, qh = 16 * hp + (tid & 15);
    const __nv_bfloat16* ga = p.qkv + ((size_t)b * L + j) * stride + (size_t)(p.Hq + 2 * p.Hkv) * 64 + (size_t)qh * 64;
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(ga));
  }
  float2 xv[RPW];                                     // Q rows, loaded first (independent of the KV chain)
#pragma unroll
  for (int k = 0; k < RPW; ++k) {
    const int R = warp + k * NWARP;
    const int hh = R / R8, r = R - hh * R8, j = r >> 3, gi = r & 7;
    const int qh = (2 * hp + hh) * 8 + gi;
    xv[k] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
        p.qkv + ((size_t)b * L + j) * stride + (size_t)qh * 64 + 2 * lane));
  }
  const int W = p.W;
  const int kstart = W ? max(0, seq - W + 1) : 0;
  const size_t rowB = (size_t)p.Hkv * 64;
  const uint8_t* kc = p.k_cache + (size_t)hp * 128;
  const uint8_t* vc = p.v_cache + (size_t)hp * 128;
  const int* pt = p.page_table + (size_t)b * p.pt_stride;
  const int pshift = p.page_shift;
  const int pmask = (1 << pshift) - 1;

  const int lk = tid >> 2, part = tid & 3;
  const int lq = tid >> 6, lb = (tid >> 5) & 1, ltp = (tid >> 3) & 3, lc = tid & 7;
  const int lk0 = 16 * lq + 8 * lb + 2 * ltp;
  auto pe_load = [&](int i) -> int2 {
    int2 pe = make_int2(-1, -1);
    if constexpr (L == 1) {
      const int absk = kstart + (t0 + i) * TILE + lk;
      if (i < ntl && absk < seq) pe.x = __ldg(pt + (absk >> pshift));
    } else {
      const int absk = kstart + (t0 + i) * TILE + lk0;
      if (i < ntl && absk < seq) pe.x = __ldg(pt + (absk >> pshift));
      if (i < ntl && absk + 1 < seq) pe.y = __ldg(pt + ((absk + 1) >> pshift));
    }
    return pe;
  };
  auto issue = [&](int i, int2 pe2) {
    uint8_t* stg = ring + (i % ST) * STAGE_BYTES;
    if constexpr (L == 1) {
      const int absk = kstart + (t0 + i) * TILE + lk;
      const int pe = pe2.x;
      const bool valid = pe >= 0;
      const size_t goff = valid ? (size_t)((pe << pshift) | (absk & pmask)) * rowB : (size_t)0;
      const int nb = valid ? 16 : 0;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int ch = part + 4 * u;   // v24: sector-complete 16-B requests
        cp_async16(stg + lk * ROWB + 16 * kswz<L>(lk, ch), kc + goff + 16 * ch, nb);
        cp_async16(stg + TILE * ROWB + lk * ROWB + 16 * vswz<L>(lk, ch), vc + goff + 16 * ch, nb);
      }
    } else {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int x = lk0 + e;
        const int absk = kstart + (t0 + i) * TILE + x;
        const int pe = e ? pe2.y : pe2.x;
        const bool valid = pe >= 0;
        const size_t goff = valid ? (size_t)((pe << pshift) | (absk & pmask)) * rowB : (size_t)0;
        const int nb = valid ? 16 : 0;
        cp_async16(stg + (lc >> 2) * 4096 + (x >> 3) * 512 + (lc & 3) * 128 + (x & 7) * 16, kc + goff + 16 * lc, nb);
        cp_async16(stg + 8192 + e * 4096 + tid * 16, vc + goff + 16 * lc, nb);
      }
    }
  };
  {
    int2 pe0[ST - 1];
#pragma unroll
    for (int i = 0; i < ST - 1; ++i) pe0[i] = pe_load(i);
#pragma unroll
    for (int i = 0; i < ST - 1; ++i) {
      if (i < ntl) issue(i, pe0[i]);
      cp_async_commit();
    }
  }
  int2 pfA = pe_load(ST - 1);
  int2 pfB = pe_load(ST);

  if (p.use_rope && tid < L * 32) {
    const int j = tid >> 5, i = tid & 31;
    const float ang = __fmul_rn((float)(seq + j), __ldg(p.inv_freq + i));
    float sn, cs;
    sincosf(ang, &sn, &cs);
    cst[tid] = make_float2(cs, sn);
  }
  __syncthreads();

  {
#pragma unroll
    for (int k = 0; k < RPW; ++k) {
      const int R = warp + k * NWARP;
      const int hh = R / R8, r = R - hh * R8, j = r >> 3;
      float2 x = xv[k];
      const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
      x.x *= rs; x.y *= rs;
      if (p.use_rope) x = rope_tab(x, cst[j * 32 + lane]);
      if constexpr (L == 1) {
        reinterpret_cast<uint32_t*>(qs)[(hh * R8 + r) * 32 + lane] = pack_h2(x.x, x.y);   // fp16
      } else {                                      // e4m3, wgmma K-major layout (rows r of head hh)
        *reinterpret_cast<uint16_t*>(qs + hh * 4096 + il_off(r, 2 * lane, 128, 512)) =
            (uint16_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
      }
    }
  }
  if (last) {
    const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
    for (int R = warp; R < 4 * L; R += NWARP) {
      const int kind = R / (2 * L);                // 0 = K, 1 = V (warp-uniform)
      const int rr = R - kind * 2 * L;
      const int hh = rr / L, j = rr - hh * L;
      const size_t tok = (size_t)b * L + j;
      const int head = 2 * hp + hh;
      const int colh = (kind == 0) ? (p.Hq + head) : (p.Hq + p.Hkv + head);
      float2 x = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(p.qkv + tok * stride + (size_t)colh * 64 + 2 * lane));
      float sc;
      if (kind == 0) {
        const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
        x.x *= rs; x.y *= rs;
        if (p.use_rope) x = rope_tab(x, cst[j * 32 + lane]);
        sc = ksc;
      } else {
        sc = vsc;
      }
      const __nv_fp8x2_storage_t q8 =
          __nv_cvt_float2_to_fp8x2(make_float2(x.x / sc, x.y / sc), __NV_SATFINITE, __NV_E4M3);
      const int64_t row = p.cache_loc[tok];
      uint8_t* dst = ((kind == 0) ? p.k_cache : p.v_cache) + (size_t)row * rowB + (size_t)head * 64 + 2 * lane;
      *reinterpret_cast<uint16_t*>(dst) = (uint16_t)q8;
      if constexpr (L == 1) {
        const int bb = 64 * hh + 2 * lane;         // logical byte in the 128-B row
        const int ch = bb >> 4;
        const int pc = (kind == 0) ? kswz<L>(j, ch) : vswz<L>(j, ch);
        *reinterpret_cast<uint16_t*>(((kind == 0) ? newk : newv) + j * ROWB + 16 * pc + (bb & 15)) = (uint16_t)q8;
      } else if (kind == 0) {                      // new K tile (wgmma layout), key j
        *reinterpret_cast<uint16_t*>(newk + hh * 4096 + il_off(j, 2 * lane, 128, 512)) = (uint16_t)q8;
      } else {                                     // new V^T tile: dims 2lane, 2lane+1 of key j
        int st, kl;
        key_to_sk(j, st, kl);
        uint8_t* nvt = newk + 8192;
        nvt[vt_off(hh, st, 2 * lane, kl)] = (uint8_t)(q8 & 0xFF);
        nvt[vt_off(hh, st, 2 * lane + 1, kl)] = (uint8_t)(q8 >> 8);
      }
    }
  }
  if constexpr (L == 8) fence_async_smem();          // Q, new K / V^T tiles -> async proxy (wgmma)
  __syncthreads();
  K1P(2);

  const int g = lane >> 2, t = lane & 3;
  const float cl2 = p.qk_sm * __ldg(p.k_scale) * LOG2E;
  float* ost;     // [ROWS][64] normalized partial O
  float* lse_s;   // [ROWS]

  if constexpr (L == 1) {
    const int h = warp & 1, q4 = warp >> 1, kw = 16 * q4;
    uint32_t qf[4][4];
    {
      const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs) + (h * R8 + g) * 32;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int d0 = (16 * t + 4 * s) >> 1;      // u32 holding dims 16t+4s, +1
        qf[s][0] = qa[d0];
        qf[s][2] = qa[d0 + 1];
        qf[s][1] = 0u;
        qf[s][3] = 0u;
      }
    }
    float o[8][4];
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] = 0.f; o[jn][1] = 0.f; o[jn][2] = 0.f; o[jn][3] = 0.f; }
    float mA = neg_inf(), lA = 0.f;
    for (int i = 0; i < ntl; ++i) {
      cp_async_wait<ST - 2>();
      asm volatile("bar.sync %0, 64;\n" :: "r"(1 + q4) : "memory");   // v24: warp-pair barrier
      {
        const int nx = i + ST - 1;
        if (nx < ntl) issue(nx, pfA);
        cp_async_commit();
        pfA = pfB;
        pfB = pe_load(i + ST + 1);
      }
      const uint8_t* Kb = ring + (i % ST) * STAGE_BYTES;
      const int ts = kstart + (t0 + i) * TILE;
      const bool need = (ts + TILE > seq) || (W > 0 && ts < seq + L - W);
      tile_compute1<2>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, need, ts, seq, seq, W);
    }
    cp_async_wait<0>();
    __syncthreads();
    K1P(3);
    if (last && q4 == 0) tile_compute1<2>(newk, newv, 0, h, g, t, qf, o, mA, lA, cl2, true, seq, seq + 1, seq, W);
    lA += __shfl_xor_sync(FULL, lA, 1);
    lA += __shfl_xor_sync(FULL, lA, 2);
    float* ow = reinterpret_cast<float*>(ring);   // [8 warps][8 rows][64]
    float* mw = ow + NWARP * 8 * 64;              // [8][8]
    float* lw = mw + NWARP * 8;                   // [8][8]
    ost = lw + NWARP * 8;                         // [16][64]
    lse_s = ost + ROWS * 64;                      // [16]
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
      for (int e = 0; e < 2; ++e) ow[(warp * 8 + g) * 64 + ((8 * (2 * t + e) + jn) ^ g ^ ((t >> 1) << 3))] = o[jn][e];   // v20: XOR swizzle, 32 banks
    }
    if (t == 0) { mw[warp * 8 + g] = mA; lw[warp * 8 + g] = lA; }
    __syncthreads();
    for (int idx = tid; idx < 2 * 8 * 64; idx += NTHR) {
      const int d = idx & 63, r = (idx >> 6) & 7, hh = idx >> 9;
      float M = neg_inf();
#pragma unroll
      for (int q = 0; q < 4; ++q) M = fmaxf(M, mw[((q << 1) | hh) * 8 + r]);
      const float Mu = (M == neg_inf()) ? 0.f : M;
      float lt = 0.f, acc = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int w = (q << 1) | hh;
        const float f = ex2(mw[w * 8 + r] - Mu);
        lt += f * lw[w * 8 + r];
        acc += f * ow[(w * 8 + r) * 64 + (d ^ r ^ (((d >> 5) & 1) << 3))];
      }
      ost[(hh * 8 + r) * 64 + d] = (lt > 0.f) ? acc / lt : 0.f;
      if (d == 0) lse_s[hh * 8 + r] = (lt > 0.f) ? M + log2f(lt) - P_OFFSET : neg_inf();
    }
  } else {
    const int h = warp >> 2, wq = warp & 3, j0 = 2 * wq;
    uint8_t* vtb = smem + off_vh<L, ST>();            // V^T double buffer
    const uint64_t dq = desc_il(qs + h * 4096, 128, 512);
    float S[32], O[36];
#pragma unroll
    for (int i = 0; i < 36; ++i) O[i] = 0.f;
    float m[2] = {neg_inf(), neg_inf()};
    float l[2] = {0.f, 0.f};
    const int vt_own = vt_off(lc >> 2, lq >> 1, 16 * (lc & 3), 16 * (lq & 1) + 4 * ltp + 2 * lb);
    auto transpose_v = [&](int x) {
      if (K1_DIAG6 == 1) return;
      const uint8_t* stg = ring + (x % ST) * STAGE_BYTES + 8192;
      const uint4 va = *reinterpret_cast<const uint4*>(stg + tid * 16);
      const uint4 vb = *reinterpret_cast<const uint4*>(stg + 4096 + tid * 16);
      uint8_t* dst = vtb + (x % 3) * VT_BYTES + vt_own;
      const uint32_t wa[4] = {va.x, va.y, va.z, va.w}, wb[4] = {vb.x, vb.y, vb.z, vb.w};
#pragma unroll
      for (int w4 = 0; w4 < 4; ++w4) {
        const uint32_t lo = __byte_perm(wa[w4], wb[w4], 0x5140);
        const uint32_t hi = __byte_perm(wa[w4], wb[w4], 0x7362);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int d = 4 * w4 + u;
          const uint32_t hw = (u < 2) ? ((u == 0) ? (lo & 0xFFFFu) : (lo >> 16)) : ((u == 2) ? (hi & 0xFFFFu) : (hi >> 16));
          *reinterpret_cast<uint16_t*>(dst + (d & 7) * 16 + (d >> 3) * VT_SBO) = (uint16_t)hw;
        }
      }
    };
    uint32_t pa[2][4];
    if (ntl > 0) {
      cp_async_wait<ST - 2>();
      transpose_v(0);
      fence_async_smem();
    }
    for (int i = 0; i < ntl; ++i) {
      __syncthreads();
      {
        const int nx = i + ST - 1;                    // slot (i-1)%ST: QK(i-1) waited by all; raw V read by its owner
        if (nx < ntl) issue(nx, pfA);
        cp_async_commit();
        pfA = pfB;
        pfB = pe_load(i + ST + 1);
      }
      qk_issue(S, dq, desc_il(ring + (i % ST) * STAGE_BYTES + h * 4096, 128, 512));
      if (i + 1 < ntl) {
        cp_async_wait<ST - 2>();                      // this thread's chunks of tile i+1
        transpose_v(i + 1);
        fence_async_smem();
      }
      wg_wait<0>();                                   // QK(i) and PV(i-1)
      const int ts = kstart + (t0 + i) * TILE;
      const bool need = (ts + TILE > seq) || (W > 0 && ts < seq + L - W);
      softmax_pack7(S, O, m, pa, t, cl2, need, ts, seq, seq, W, j0);
      pv_issue7(O, pa, vtb + (i % 3) * VT_BYTES, h);
    }
    wg_wait<0>();
    cp_async_wait<0>();
    __syncthreads();
    K1P(3);
    if (last)                                         // new-token tiles (written + fenced in the prologue)
    {
      qk_issue(S, dq, desc_il(newk + h * 4096, 128, 512));
      wg_wait<0>();
      softmax_pack7(S, O, m, pa, t, cl2, true, seq, seq + L, seq, W, j0);
      pv_issue7(O, pa, newk + 8192, h);
      wg_wait<0>();
    }
    ost = reinterpret_cast<float*>(ring);            // [128][64] (ring drained; the last QK read waited)
    lse_s = ost + ROWS * 64;                          // [128]
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      const float lt = O[32 + 2 * rb];                // row sum from the ones columns (same in O[33 + 2rb])
      const float inv = (lt > 0.f) ? 1.f / lt : 0.f;
      const int row = h * R8 + 16 * wq + 8 * rb + g;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        *reinterpret_cast<float2*>(ost + row * 64 + 8 * j + 2 * t) =
            make_float2(O[4 * j + 2 * rb] * inv, O[4 * j + 2 * rb + 1] * inv);
      }
      if (t == 0) lse_s[row] = (lt > 0.f) ? m[rb] + log2f(lt) - P_OFFSET : neg_inf();
    }
  }
  __syncthreads();
  K1P(4);

  if (p.fold && S_b == 1) {
    merge_finalize<L>(p, ost, lse_s, 1, b, hp, false);
    K1P(5);
    return;
  }
  {
    const size_t pbase = (size_t)hp * p.Gx + c;
    float4* po = reinterpret_cast<float4*>(p.part_o + pbase * ROWS * 64);
    const float4* so = reinterpret_cast<const float4*>(ost);
    for (int idx = tid; idx < ROWS * 16; idx += NTHR) po[idx] = so[idx];
    for (int idx = tid; idx < ROWS; idx += NTHR) p.part_lse[pbase * ROWS + idx] = lse_s[idx];
  }
  if (p.fold) {
    __threadfence();                                  // this thread's partial stores visible device-wide
    __syncthreads();                                  // ... for every thread of the CTA
    int* cnt = p.counters + (size_t)b * HP + hp;
    if (tid == 0) plan_s[5] = (atomicAdd(cnt, 1) == S_b - 1) ? 1 : 0;
    __syncthreads();
    K1P(5);
    if (plan_s[5]) {                                  // last CTA of (b, hp): every partial is written
      __threadfence();
      K1P(7);
      const size_t base = (size_t)hp * p.Gx + off_b;
      merge_finalize<L>(p, p.part_o + base * ROWS * 64, p.part_lse + base * ROWS, S_b, b, hp, true);
      __syncthreads();
      if (tid == 0) *cnt = 0;                         // self-reset for the next call (after all reads)
      K1P(6);
    }
  } else {
    K1P(5);
  }
#endif
}

__global__ void __launch_bounds__(512) k1_combine(const Params p, const int L) {
#if K1_ARCH_OK
  __shared__ float4 sacc[32][16];
  __shared__ float sws[32];
  __shared__ float sred[16];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nwarp = blockDim.x >> 5;
  const int nsg = blockDim.x >> 4, sg = tid >> 4, u = tid & 15;
  const int qh = blockIdx.x % p.Hq;
  const int bj = blockIdx.x / p.Hq;
  const int b = bj / L, j = bj - b * L;
  const int hp = qh >> 4, h = (qh >> 3) & 1, gi = qh & 7;
  const int R8 = 8 * L, ROWS = 16 * L;
  const int row = h * R8 + j * 8 + gi;
  const int C = plan_C(p.seq_lens, p.B, p.W, p.Gx, p.min_chunk);
  int res[5];
  plan_find(p.seq_lens, p.B, p.W, C, b, false, res);
  const int S = res[2], off = res[3];
  const float* pl = p.part_lse + ((size_t)hp * p.Gx + off) * ROWS + row;
  const float* po = p.part_o + (((size_t)hp * p.Gx + off) * ROWS + row) * 64 + 4 * u;
  float mx = neg_inf();
  for (int s = tid; s < S; s += blockDim.x) mx = fmaxf(mx, pl[(size_t)s * ROWS]);
  mx = warp_max(mx);
  if (lane == 0) sred[wid] = mx;
  __syncthreads();
  mx = neg_inf();
  for (int w = 0; w < nwarp; ++w) mx = fmaxf(mx, sred[w]);
  const float Mu = (mx == neg_inf()) ? 0.f : mx;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  float ws = 0.f;
#pragma unroll 4
  for (int s = sg; s < S; s += nsg) {
    const float w = ex2(pl[(size_t)s * ROWS] - Mu);
    const float4 v = *reinterpret_cast<const float4*>(po + (size_t)s * ROWS * 64);
    ws += w;
    acc.x = fmaf(w, v.x, acc.x); acc.y = fmaf(w, v.y, acc.y);
    acc.z = fmaf(w, v.z, acc.z); acc.w = fmaf(w, v.w, acc.w);
  }
  sacc[sg][u] = acc;
  if (u == 0) sws[sg] = ws;
  __syncthreads();
  if (tid < 16) {
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    float wt = 0.f;
    for (int q = 0; q < nsg; ++q) {
      const float4 a = sacc[q][u];
      o.x += a.x; o.y += a.y; o.z += a.z; o.w += a.w;
      wt += sws[q];
    }
    const float sc = (wt > 0.f) ? __ldg(p.v_scale) / wt : 0.f;
    o.x *= sc; o.y *= sc; o.z *= sc; o.w *= sc;
    float ss = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
    ss += __shfl_xor_sync(0xFFFFu, ss, 8);
    ss += __shfl_xor_sync(0xFFFFu, ss, 4);
    ss += __shfl_xor_sync(0xFFFFu, ss, 2);
    ss += __shfl_xor_sync(0xFFFFu, ss, 1);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const size_t tok = (size_t)b * L + j;
    const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
    const uint2 graw = *reinterpret_cast<const uint2*>(p.qkv + tok * stride + (size_t)(p.Hq + 2 * p.Hkv) * 64 +
                                                       (size_t)qh * 64 + 4 * u);
    const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
    const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
    const __nv_bfloat162 o01 = __floats2bfloat162_rn(o.x * rs * sigmoidf_(g01.x), o.y * rs * sigmoidf_(g01.y));
    const __nv_bfloat162 o23 = __floats2bfloat162_rn(o.z * rs * sigmoidf_(g23.x), o.w * rs * sigmoidf_(g23.y));
    uint2 ow;
    ow.x = *reinterpret_cast<const uint32_t*>(&o01);
    ow.y = *reinterpret_cast<const uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(p.out + tok * (size_t)p.Hq * 64 + (size_t)qh * 64 + 4 * u) = ow;
  }
#endif
}

}  // namespace k1

template <int L, int ST>
static void ensure_smem_attr() {
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::k1_main<L, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        k1::smem_bytes<L, ST>()));
    attr_set = true;
  }
}

template <int L, int ST>
static void launch_main(const k1::Params& p, dim3 grid, cudaStream_t st) {
  ensure_smem_attr<L, ST>();
  k1::k1_main<L, ST><<<grid, k1::NTHR, k1::smem_bytes<L, ST>(), st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int L, int ST>
static int occ_of() {
  int nb = 0;
  ensure_smem_attr<L, ST>();
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::k1_main<L, ST>, k1::NTHR,
                                                               k1::smem_bytes<L, ST>()));
  return nb;
}

static bool stages_ok(int64_t L, int64_t ST) { return (L == 1) ? (ST == 4 || ST == 6 || ST == 203 || ST == 204 || ST == 213 || ST == 214) : (ST == 3 || ST == 4); }

template <int ST>
static int va_go(const k1::Params* p, dim3 grid, cudaStream_t s, bool pdl = false) {   // VA1 (tools/va_patches.py): launch, or occupancy
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::va1_main<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, k1::va_smem<ST>()));
    attr = true;
  }
  int nb = 0;
  if (p) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid; cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = k1::va_smem<ST>(); cfg.stream = s;
    cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k1::va1_main<ST>, *p));
  } else C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::va1_main<ST>, 256, k1::va_smem<ST>()));
  return nb;
}

int64_t k1_occupancy(int64_t L, int64_t ST) {
  TORCH_CHECK(stages_ok(L, ST), "k1: (L, stages) not instantiated");
  if (ST > 200) return (int64_t)(ST % 10 == 3 ? va_go<3>(nullptr, dim3(1), 0) : va_go<4>(nullptr, dim3(1), 0));
  if (L == 1) return (int64_t)(ST == 4 ? occ_of<1, 4>() : occ_of<1, 6>());
  return (int64_t)(ST == 3 ? occ_of<8, 3>() : occ_of<8, 4>());
}

torch::Tensor k1_prof_read() {
#if K1_PROF
  auto t = torch::empty({4096 * 16}, torch::dtype(torch::kInt64));
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(t.data_ptr<int64_t>(), k1::k1_prof, sizeof(unsigned long long) * 4096 * 16));
  static unsigned long long zeros[4096 * 16];
  C10_CUDA_CHECK(cudaMemcpyToSymbol(k1::k1_prof, zeros, sizeof(zeros)));
  return t;
#else
  return torch::empty({0}, torch::dtype(torch::kInt64));
#endif
}

void k1_forward(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor page_table,
                torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
                torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor out,
                torch::Tensor counters, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t W,
                int64_t page_shift, int64_t Gx, int64_t min_chunk, int64_t use_rope, double qk_sm, int64_t fold,
                int64_t nsg, int64_t stages) {
  TORCH_CHECK(L == 1 || L == 8, "k1: q_len must be 1 or 8");
  TORCH_CHECK(stages_ok(L, stages), "k1: (L, stages) not instantiated");
  TORCH_CHECK(Hq == 8 * Hkv && Hkv % 2 == 0 && Hkv > 0, "k1: need Hq == 8*Hkv and even Hkv");
  TORCH_CHECK(Gx > B, "k1: Gx must exceed B");
  TORCH_CHECK(min_chunk >= 1, "k1: min_chunk");
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.is_contiguous(), "k1: qkv");
  TORCH_CHECK(qkv.numel() == B * L * (2 * Hq + 2 * Hkv) * 64, "k1: qkv shape");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous() && seq_lens.numel() == B, "k1: seq_lens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.is_contiguous() && cache_loc.numel() == B * L,
              "k1: cache_loc");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.is_contiguous() && page_table.dim() == 2 &&
                  page_table.size(0) == B, "k1: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && k_cache.is_contiguous(), "k1: k_cache");
  TORCH_CHECK(v_cache.scalar_type() == at::kByte && v_cache.is_contiguous(), "k1: v_cache");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat && v_scale.scalar_type() == at::kFloat, "k1: scales");
  TORCH_CHECK(inv_freq.scalar_type() == at::kFloat && inv_freq.numel() >= 32, "k1: inv_freq");
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_o.numel() >= (Hkv / 2) * Gx * 16 * L * 64, "k1: part_o");
  TORCH_CHECK(part_lse.scalar_type() == at::kFloat && part_lse.numel() >= (Hkv / 2) * Gx * 16 * L, "k1: part_lse");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * L * Hq * 64, "k1: out");
  TORCH_CHECK(counters.scalar_type() == at::kInt && (fold == 0 || counters.numel() >= B * (Hkv / 2)), "k1: counters");
  TORCH_CHECK(fold != 0 || (nsg == 2 || nsg == 4 || nsg == 8 || nsg == 16 || nsg == 32), "k1: nsg");
  const c10::cuda::OptionalCUDAGuard guard(qkv.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  k1::Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.page_table = page_table.data_ptr<int>();
  p.k_cache = k_cache.data_ptr<uint8_t>();
  p.v_cache = v_cache.data_ptr<uint8_t>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.part_o = part_o.data_ptr<float>();
  p.part_lse = part_lse.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.counters = counters.data_ptr<int>();
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.W = (int)W; p.page_shift = (int)page_shift;
  p.pt_stride = (int)page_table.size(1); p.Gx = (int)Gx; p.min_chunk = (int)min_chunk; p.use_rope = (int)use_rope;
  p.fold = (int)fold;
  p.qk_sm = (float)qk_sm;
  const dim3 grid((unsigned)Gx, (unsigned)(Hkv / 2));
  if (stages > 200) {   // VA1: needs L 1, fold 1, B <= 512, part_o >= B*Hkv/2*16 rows of 64 (checked by Python)
    TORCH_CHECK(L == 1 && fold == 1 && B <= 512, "va1: L 1, fold, B <= 512");
    if (stages % 10 == 3) va_go<3>(&p, grid, st, stages > 210);
    else va_go<4>(&p, grid, st, stages > 210);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (L == 1) {
    if (stages == 4) launch_main<1, 4>(p, grid, st);
    else launch_main<1, 6>(p, grid, st);
  } else {
    if (stages == 3) launch_main<8, 3>(p, grid, st);
    else launch_main<8, 4>(p, grid, st);
  }
  if (!fold) {
    k1::k1_combine<<<(unsigned)(B * L * Hq), (unsigned)(16 * nsg), 0, st>>>(p, (int)L);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
"""

def _k1_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((K1_CPP_SRC + K1_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"k1_v7_{tag}",
        cpp_sources=K1_CPP_SRC,
        cuda_sources=K1_CUDA_SRC,
        functions=["k1_forward", "k1_occupancy", "k1_prof_read"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a"],
        verbose=True,
    )
    print(f"[k1] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod

_K1_MOD = None

_K3_BUILD_S = {}

if USE_K1 and not V10_COMBINED_BUILD:
    _t_build = time.time()
    try:
        _K1_MOD = _k1_build()
        _K3_BUILD_S["k1"] = time.time() - _t_build
    except Exception as _ex:
        print("[k3] K1 CUDA build FAILED: attention stays on the r1 FA3 path. Compiler output tail:\n"
              + str(_ex)[-12000:], flush=True)
        _K1_MOD = None

_K1_STATE = {}

_K1_SMS = {}

_K1_OCC = {}

_K1_INV_FREQ = {}

_K1_CNT = {}

_K3_WARNED = set()

def _k1_sms(dev):
    key = str(dev)
    if key not in _K1_SMS:
        _K1_SMS[key] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _K1_SMS[key]

def _k1_occupancy(L, st, dev):
    key = (L, st, str(dev))
    if key not in _K1_OCC:
        _K1_OCC[key] = max(1, int(_K1_MOD.k1_occupancy(L, st)))
        print(f"[k1] k1_main<{L},{st}> occupancy {_K1_OCC[key]} CTAs/SM", flush=True)
    return _K1_OCC[key]

def _k1_counters(dev):
    key = str(dev)
    if key not in _K1_CNT:
        if torch.cuda.is_current_stream_capturing():
            if not _K1_STATE.get("warned_cnt"):
                _K1_STATE["warned_cnt"] = True
                print("[k3] K1 fold counters first needed under graph capture (prepare allocates them): r1 path", flush=True)
            return None
        _K1_CNT[key] = torch.zeros(K1_COUNTERS, dtype=torch.int32, device=dev)
    return _K1_CNT[key]

def _k1_inv_freq(theta, dev):
    key = (float(theta), str(dev))
    if key not in _K1_INV_FREQ:
        if theta:
            inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev).float() / 64))
        else:
            inv = torch.zeros(32, dtype=torch.float32, device=dev)
        _K1_INV_FREQ[key] = inv.contiguous()
    return _K1_INV_FREQ[key]

def _k1_kind(cfg):
    return "sliding" if cfg["window"] else "full"

def _k1_supported(data, cfg):
    hq, hkv, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"]
    kind = "sliding" if cfg["window"] else "full"
    return (cfg["batch"] >= K1_ROUTE_MIN_BATCH.get((kind, cfg["q_len"]), 1 << 30) and cfg["head_dim"] == 64 and hkv > 0 and hq == 8 * hkv
            and hkv % 2 == 0 and cfg["q_len"] in (1, 8) and page > 0 and (page & (page - 1)) == 0
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["k_scale"].dtype == torch.float32 and data["v_scale"].dtype == torch.float32
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2
            and data["seq_lens"].dtype == torch.int32 and data["cache_loc"].dtype == torch.int64
            and data["seq_lens"].is_contiguous() and data["cache_loc"].is_contiguous()
            and data["page_table"].is_contiguous())

def _k1_ready(cfg, dev):
    hit = ((cfg["q_len"], K1_STAGES_T.get((_k1_kind(cfg), cfg["q_len"]), K1_STAGES), str(dev)) in _K1_OCC and (float(cfg["rope_theta"]), str(dev)) in _K1_INV_FREQ
           and str(dev) in _K1_SMS and str(dev) in _K1_CNT)
    return hit or not torch.cuda.is_current_stream_capturing()

def _k1_run(qkv, data, cfg, fold=None, min_chunk=None, cps=None, nsg=None, stages=None):
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    key = (_k1_kind(cfg), L)
    dev = qkv.device
    hp = hkv // 2
    if fold is None:
        fold = B >= K1_FOLD_MIN_BATCH.get(key, 1 << 30)
    if min_chunk is None:
        min_chunk = K1_MIN_CHUNK_T.get(key, 2)
    if cps is None:
        cps = K1_CPS_T.get(key, 2)
    if nsg is None:
        nsg = _k1_nsg(B)
    if stages is None:
        stages = K1_STAGES_T.get(key, K1_STAGES)
    cnt = _k1_counters(dev)
    if cnt is None:
        raise RuntimeError("k1: fold counters missing under graph capture")
    if fold and B * hp > cnt.numel():
        fold = False
    cps_eff = max(1, min(int(cps), _k1_occupancy(L, int(stages), dev)))
    gx = max(max(1, (_k1_sms(dev) * cps_eff) // hp), B + 1)
    part_o = torch.empty((hp, gx, 16 * L, 64), dtype=torch.float32, device=dev)
    part_lse = torch.empty((hp, gx, 16 * L), dtype=torch.float32, device=dev)
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    theta = cfg["rope_theta"]
    _K1_MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                    data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], _k1_inv_freq(theta, dev),
                    part_o, part_lse, out, cnt, B, L, hq, hkv, cfg["window"], cfg["page_size"].bit_length() - 1, gx,
                    int(min_chunk), 1 if theta else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]),
                    1 if fold else 0, int(nsg), int(stages))
    return out

VA1_ENABLE = True

VA1_MIN_B = 2

VA1_STAGES = 4

VA1_SEGK = 400

VA1_PDL = True

VA1_B1 = False

VA1_NC1 = 8

VA1_ROWS = 16 * 1024

VA1_CASES = ((2, 3000, 1, 1.0, 301), (3, 1500, 64, 1.0, 302), (16, 700, 1, 1.0 / 64, 303), (5, 9000, 1, 1.0, 304),
             (33, 300, 1, 1.0, 305), (32, 2500, 1, 1.0, 307), (1, 9000, 1, 1.0, 308), (1, 2000, 64, 1.0 / 64, 309))

_VA1_WS = {}

def _va1_wanted(data, cfg, state):
    B = cfg["batch"]
    return (VA1_ENABLE and state.get("va1_ok") and cfg["q_len"] == 1 and not cfg["window"] and not cfg["rope_theta"]
            and (1 if VA1_B1 else VA1_MIN_B) <= B <= 256 and B * cfg["num_kv_heads"] * 8 * VA1_NC1 <= VA1_ROWS
            and str(data["k_cache"].device) in _VA1_WS
            and _k1_supported(data, cfg))

def _va1_b1(cfg, state):
    return VA1_B1 and VA1_ENABLE and state.get("va1_ok") and cfg["batch"] == 1 and cfg["q_len"] == 1 and not cfg["window"]

def _va1_run(qkv, data, cfg, stages=None, segk=None, pdl=None):
    B, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    dev = qkv.device
    st = 200 + int(stages or VA1_STAGES)
    G = max(B + 1, _k1_sms(dev) * max(1, min(2, _K1_OCC.get((1, st, str(dev)), 1))) // (hkv // 2))
    pdl = VA1_PDL if pdl is None else pdl
    st += 10 if pdl else 0
    ws = _VA1_WS[str(dev)]
    out = torch.empty((B, 1, hq * 64), dtype=torch.bfloat16, device=dev)
    _K1_MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                       data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], ws[2], ws[0], ws[1], out, ws[3],
                       B, 1, hq, hkv, VA1_NC1 if B == 1 else 1, cfg["page_size"].bit_length() - 1, G, max(1, int(VA1_SEGK if segk is None else segk)),
                       1 if pdl else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]), 1, 4, st)
    return out

def _va1_try(qkv, data, cfg, state):
    if _va1_wanted(data, cfg, state):
        try:
            return _va1_run(qkv, data, cfg)
        except Exception as ex:
            _k3_disable(state, "va1_ok", False, "VA1", ex)
    return None

def _va1_prepare(state):
    state["va1_ok"] = False
    dev = torch.empty(0, device="cuda").device
    try:
        if not state.get("k1_ok"):
            raise RuntimeError("K1 module not live")
        for st in (203, 204):
            _k1_occupancy(1, st, dev)
        z = lambda *s: torch.zeros(s, dtype=torch.float32, device=dev)
        _VA1_WS[str(dev)] = ws = (z(VA1_ROWS, 64), z(VA1_ROWS), _k1_inv_freq(0.0, dev),
                                  torch.zeros(VA1_ROWS // 16, dtype=torch.int32, device=dev))
        for (B, ctx, page, kvs, seed) in VA1_CASES:
            for rep in range(2):
                d = _k1_mk_case(dev, B, ctx, 1, False, page, kvs, seed)
                out = _va1_run(d["qkv"], d, d["config"], pdl=bool(rep))
                pos = d["seq_lens"].long()[:, None]
                slots = torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page
                kn, vn = (c.view(torch.uint8).view(-1, 2, 64)[slots] for c in (d["k_cache"], d["v_cache"]))
                tot, row = _k1_rel_rows(out, _k1_ref_attention(d, kn, vn))
                kref, vref = _k1_ref_new_kv(d)
                kt, vt = _k1_rel_rows(kn.view(FP8).float(), kref)[0], _k1_rel_rows(vn.view(FP8).float(), vref)[0]
                d2 = _k1_mk_case(dev, B, ctx, 1, False, page, kvs, seed)
                vk = _k1_rel_rows(out, _k1_run(d2["qkv"], d2, d2["config"]))[0]
                clean = not (int(ws[3].abs().sum()) or float(ws[0].abs().sum()) or float(ws[1].abs().sum()))
                print(f"[va1] self-test B={B} ctx={ctx} page={page} kvs={kvs:g} #{rep}: rel {tot:.2e} row {row:.2e} vs K1 "
                      f"{vk:.2e} new K {kt:.1e} V {vt:.1e} ws {'clean' if clean else 'DIRTY'}", flush=True)
                if not (bool(torch.isfinite(out.float()).all()) and clean and tot < 2e-2 and row < 6e-2 and vk < 1e-2
                        and kt < 3e-2 and vt < 3e-2):
                    raise RuntimeError(f"self-test mismatch B={B} ctx={ctx}")
        state["va1_ok"] = True
        print(f"[va1] LIVE: full q_len 1 B >= {VA1_MIN_B}, stages {VA1_STAGES}, segk {VA1_SEGK}, enable {VA1_ENABLE}", flush=True)
    except Exception as ex:
        print(f"[va1] DISABLED, full decode attention stays on K1: {ex!r}", flush=True)

def _k1_rms_t(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def _k1_rope_t(x, pos, theta):
    inv = _k1_inv_freq(theta, x.device)
    ang = pos.float()[..., None] * inv
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    ev, od = x[..., 0::2], x[..., 1::2]
    return torch.stack([ev * cos - od * sin, ev * sin + od * cos], -1).flatten(-2)

def _k1_mk_case(dev, B, ctx, L, sliding, page, kv_scale, seed, hq=16, hkv=2):
    g = torch.Generator().manual_seed(seed)
    seq = [max(1, int(ctx * (0.6 + 0.8 * float(torch.rand(1, generator=g))))) for _ in range(B)]
    max_pages = (max(seq) + L + page - 1) // page + 1
    total = B * max_pages
    perm = torch.randperm(total, generator=g).to(torch.int32)
    pt = torch.full((B, max_pages), -1, dtype=torch.int32)
    at = 0
    for b in range(B):
        n = (seq[b] + L + page - 1) // page
        pt[b, :n] = perm[at:at + n]
        at += n
    u = torch.randn((hkv, 64), generator=g)
    kf = 0.7 * u + 0.7 * torch.randn((total * page, hkv, 64), generator=g)
    vf = torch.randn((total * page, hkv, 64), generator=g)
    k_cache = (kf / kv_scale).clamp(-448, 448).to(FP8).view(total, page, hkv, 64)
    v_cache = (vf / kv_scale).clamp(-448, 448).to(FP8).view(total, page, hkv, 64)
    q = 2 * torch.randn((B, L, hq * 64), generator=g)
    kn = 2 * torch.randn((B, L, hkv * 64), generator=g)
    vn = torch.randn((B, L, hkv * 64), generator=g)
    gate = torch.randn((B, L, hq * 64), generator=g)
    qkv = torch.cat([q, kn, vn, gate], -1).to(torch.bfloat16)
    seq_t = torch.tensor(seq, dtype=torch.int64)
    pos = seq_t[:, None] + torch.arange(L)
    cache_loc = (torch.gather(pt.long(), 1, pos // page) * page + pos % page).reshape(-1)
    window = 2048 if sliding else 0
    data = {
        "qkv": qkv, "positions": pos.reshape(-1), "cache_loc": cache_loc,
        "cache_seqlens": (seq_t + L).to(torch.int32),
        "cu_seqlens_q": torch.arange(0, B * L + 1, L, dtype=torch.int32), "scheduler_metadata": None,
        "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32), "v_scale": torch.tensor(kv_scale, dtype=torch.float32),
        "page_table": pt, "seq_lens": seq_t.to(torch.int32),
    }
    data = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    data["config"] = {"batch": B, "q_len": L, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": 64,
                      "page_size": page, "window": window, "rope_theta": 500000.0 if sliding else 0.0,
                      "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": kv_scale}
    return data

def _k1_ref_attention(data, k_new, v_new, q_e4m3=False):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, W, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["window"], cfg["page_size"]
    qkv = data["qkv"].float()
    q = _k1_rms_t(qkv[..., :hq * 64].reshape(B, L, hq, 64))
    seq_l = data["seq_lens"].long()
    pos = seq_l[:, None] + torch.arange(L, device=q.device)
    if cfg["rope_theta"]:
        q = _k1_rope_t(q, pos, cfg["rope_theta"])
    if q_e4m3:
        q = q.clamp(-448.0, 448.0).to(FP8).float()
    q = q * cfg["qk_scale_factor"]
    ks, vs = data["k_scale"].float(), data["v_scale"].float()
    kf = data["k_cache"].view(torch.uint8).view(-1, hkv, 64)
    vf = data["v_cache"].view(torch.uint8).view(-1, hkv, 64)
    out = torch.empty((B, L, hq, 64), device=q.device)
    for b in range(B):
        seq = int(seq_l[b])
        idx = torch.arange(seq, device=q.device)
        rows = data["page_table"][b].long()[idx // page] * page + idx % page
        keys = torch.cat([kf[rows].view(FP8).float(), k_new[b].view(FP8).float()]) * ks
        vals = torch.cat([vf[rows].view(FP8).float(), v_new[b].view(FP8).float()]) * vs
        for j in range(L):
            p = seq + j
            st = max(0, p - W + 1) if W else 0
            att = torch.einsum("kgd,tkd->kgt", q[b, j].view(hkv, hq // hkv, 64), keys[st:p + 1]) * cfg["sm_scale"]
            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vals[st:p + 1]).reshape(hq, 64)
    gate = qkv[..., (hq + 2 * hkv) * 64:].reshape(B, L, hq, 64)
    out = _k1_rms_t(out) * torch.sigmoid(gate)
    return out.reshape(B, L, hq * 64)

def _k1_ref_new_kv(data):
    cfg = data["config"]
    B, L, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"].float()
    k = _k1_rms_t(qkv[..., hq * 64:(hq + hkv) * 64].reshape(B, L, hkv, 64))
    if cfg["rope_theta"]:
        k = _k1_rope_t(k, data["seq_lens"].long()[:, None] + torch.arange(L, device=k.device), cfg["rope_theta"])
    v = qkv[..., (hq + hkv) * 64:(hq + 2 * hkv) * 64].reshape(B, L, hkv, 64)
    q8 = [(x / sc).clamp(-448.0, 448.0).to(FP8).float() for x, sc in ((k, data["k_scale"]), (v, data["v_scale"]))]
    return q8[0], q8[1]

def _k1_rel_rows(a, b):
    a, b = a.float().reshape(-1, a.shape[-1]), b.float().reshape(-1, b.shape[-1])
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row

def _k1_self_test(dev):
    t0 = time.time()
    cases = K1_SELF_TEST_CASES
    try:
        cnt = _k1_counters(dev)
        for (B, ctx, L, sliding, page, kvs, seed, heads) in cases:
            for mode in ("fold", "sep", "fold"):
                d = _k1_mk_case(dev, B, ctx, L, sliding, page, kvs, seed, heads[0], heads[1])
                out = _k1_run(d["qkv"], d, d["config"], fold=(mode == "fold"))
                torch.cuda.synchronize(dev)
                hkv = heads[1]
                pos = d["seq_lens"].long()[:, None] + torch.arange(L, device=dev)
                pg = d["config"]["page_size"]
                slots = torch.gather(d["page_table"].long(), 1, pos // pg) * pg + pos % pg
                k_new = d["k_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
                v_new = d["v_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
                ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=(L == 8))
                tot, row = _k1_rel_rows(out, ref)
                kref, vref = _k1_ref_new_kv(d)
                kt, _ = _k1_rel_rows(k_new.view(FP8).float(), kref)
                vt, _ = _k1_rel_rows(v_new.view(FP8).float(), vref)
                finite = bool(torch.isfinite(out.float()).all())
                cnt_ok = cnt is None or int(cnt.abs().sum()) == 0
                print(f"[k1] self-test {mode:4s} B={B} ctx={ctx} L={L} {'sliding' if sliding else 'full'} page={page}"
                      f" kv_scale={kvs:g} heads={heads}: out rel {tot:.3e} worst row {row:.3e}; new K {kt:.3e}"
                      f" V {vt:.3e}; counters {'0' if cnt_ok else 'NOT RESET'}", flush=True)
                if not (finite and tot < 2e-2 and row < 6e-2 and kt < 3e-2 and vt < 3e-2 and cnt_ok):
                    raise RuntimeError("self-test mismatch")
        print(f"[k1] self-test passed in {time.time() - t0:.1f}s", flush=True)
        return True
    except Exception as ex:
        print(f"[k3] K1 self-test FAILED: attention stays on the r1 FA3 path: {ex!r}", flush=True)
        return False

def _k1_prepare(dev):
    if not K1_SELF_TEST:
        print("[k3] K1_SELF_TEST=False: K1 used untested", flush=True)
    elif not _k1_self_test(dev):
        return False
    try:
        _k1_sms(dev)
        for (kind, L), st in K1_STAGES_T.items():
            _k1_occupancy(L, int(st), dev)
        for theta in (ROPE_THETA, 0.0):
            _k1_inv_freq(theta, dev)
        if _k1_counters(dev) is None:
            raise RuntimeError("fold counters not allocated")
    except Exception as ex:
        print(f"[k3] K1 constants FAILED: attention stays on the r1 FA3 path: {ex!r}", flush=True)
        return False
    return True

V10_QK_DIRECT = True

import triton.language as tl

@triton.jit
def _v10_qk_fwd_kernel(
    QKV, Q_OUT, K_OUT, POS, COS_SIN, K_CACHE, V_CACHE, CACHE_LOC, eps, stride_qkv, n_rows, kv_capacity,
    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BLOCK_M: tl.constexpr,
    HAS_ROPE: tl.constexpr, HAS_KV_STORE: tl.constexpr, FP8_KV_STORE: tl.constexpr, USE_PDL: tl.constexpr = False,
    FP8_MAX: tl.constexpr = 448.0,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, BD)
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    if HAS_KV_STORE:
        HEADS: tl.constexpr = NUM_Q_HEADS + 2 * NUM_KV_HEADS
    else:
        HEADS: tl.constexpr = NUM_Q_HEADS + NUM_KV_HEADS
    token = rows // HEADS
    head = rows % HEADS
    is_q = head < NUM_Q_HEADS
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS
    src = token * stride_qkv + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    x = raw.to(tl.float32)
    var = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D
    rstd = tl.rsqrt(var + eps)
    x = x * rstd[:, None]
    if HAS_ROPE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
        partner_cols = cols ^ 1
        xp = tl.load(QKV + src[:, None] + partner_cols[None, :], mask=mask, other=0.0).to(tl.float32)
        xp = (xp * rstd[:, None]).to(QKV.dtype.element_ty).to(tl.float32)
        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cs_mask = row_mask[:, None] & (cols < D)[None, :]
        cos = tl.load(COS_SIN + pos[:, None] * D + freq[None, :], mask=cs_mask, other=1.0)
        sin = tl.load(COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=cs_mask, other=0.0)
        is_even = (cols % 2 == 0)[None, :]
        x = tl.where(is_even, x * cos - xp * sin, xp * sin + x * cos)
    if FP8_KV_STORE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
    if HAS_KV_STORE:
        x = tl.where(is_v[:, None], raw.to(tl.float32), x)
        if FP8_KV_STORE:
            x = tl.minimum(tl.maximum(x, -FP8_MAX), FP8_MAX)
        loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)
        kv_mask = mask
        kv_head = tl.where(is_v, head - NUM_Q_HEADS - NUM_KV_HEADS, head - NUM_Q_HEADS)
        kv_dst = loc * (NUM_KV_HEADS * D) + kv_head * D
        tl.store(K_CACHE + kv_dst[:, None] + cols[None, :], x, mask=kv_mask & is_k[:, None])
        tl.store(V_CACHE + kv_dst[:, None] + cols[None, :], x, mask=kv_mask & is_v[:, None])
    q_dst = tl.where(is_q, (token * NUM_Q_HEADS + head) * D, 0)
    k_dst = tl.where(is_k, (token * NUM_KV_HEADS + head - NUM_Q_HEADS) * D, 0)
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])
    tl.store(K_OUT + k_dst[:, None] + cols[None, :], x, mask=mask & is_k[:, None])

def _v10_qk_store(qkv, positions, cos_sin, k_rows, v_rows, cache_loc, hq, hkv, d):
    tokens = qkv.shape[0]
    q_out = torch.empty((tokens, hq * d), dtype=FP8, device=qkv.device)
    k_out = torch.empty((tokens, hkv * d), dtype=qkv.dtype, device=qkv.device)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = 4 if n_rows < 1024 else 8 if n_rows < 8192 else 16 if n_rows < 65536 else 32
    pdl = {"USE_PDL": True, "launch_pdl": True} if _V10_PDL_OK else {}
    has_rope = positions is not None
    _v10_qk_fwd_kernel[(triton.cdiv(n_rows, block_m),)](
        qkv, q_out, k_out, positions if has_rope else qkv, cos_sin if has_rope else qkv, k_rows, v_rows,
        cache_loc, RMS_EPS, qkv.stride(0), n_rows, k_rows.shape[0], hq, hkv, d, triton.next_power_of_2(d), block_m,
        HAS_ROPE=has_rope, HAS_KV_STORE=True, FP8_KV_STORE=True, num_warps=4, **pdl)
    return q_out

_V10_PDL_OK = False

def _v10_qk_prepare(state):
    global _V10_PDL_OK
    if not V10_QK_DIRECT:
        return False
    try:
        from sglang.kernels.jit.utils import is_arch_support_pdl
        _V10_PDL_OK = bool(is_arch_support_pdl())
    except Exception:
        _V10_PDL_OK = False
    dev = state["router"].device
    gen = torch.Generator(device=dev).manual_seed(1010)
    try:
        for T, rope in ((8, True), (8, False), (128, True), (256, False), (5, True)):
            rows = 4096
            qkv = torch.randn((T, 2304), generator=gen, device=dev).to(torch.bfloat16)
            pos = torch.randint(0, 100000, (T,), generator=gen, device=dev)
            loc = torch.cat((torch.zeros(1, dtype=torch.int64, device=dev),
                             torch.randperm(rows - 1, generator=gen, device=dev)[:T - 1].to(torch.int64) + 1))
            kc = (torch.randn((rows, 128), generator=gen, device=dev)).to(FP8)
            vc = (torch.randn((rows, 128), generator=gen, device=dev)).to(FP8)
            kc2, vc2 = kc.clone(), vc.clone()
            rp, cs = (pos, state["cos_sin"]) if rope else (None, None)
            k_stage = torch.empty((T + 1, 128), dtype=FP8, device=dev)
            v_stage = torch.empty((T + 1, 128), dtype=FP8, device=dev)
            sl = torch.arange(1, T + 1, dtype=torch.int64, device=dev)
            q_ref, _ = state["qk_norm"].forward_triton(qkv, rp, cs, (k_stage, v_stage, sl))
            store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8), kc.view(torch.uint8),
                        vc.view(torch.uint8), loc, reserved_skip_index=-1)
            q_got = _v10_qk_store(qkv, rp, cs, kc2, vc2, loc, 16, 2, 64)
            torch.cuda.synchronize()
            same = (torch.equal(q_ref.view(torch.uint8), q_got.view(torch.uint8))
                    and torch.equal(kc.view(torch.uint8), kc2.view(torch.uint8))
                    and torch.equal(vc.view(torch.uint8), vc2.view(torch.uint8)))
            print(f"[v10] qk direct self-test T={T} rope={rope}: bit-identical {same}", flush=True)
            if not same:
                raise RuntimeError(f"qk direct mismatch T={T}")
    except Exception as ex:
        print(f"[v10] qk direct self-test FAILED -> staging path: {ex!r}", flush=True)
        return False
    return True

_vml_JMAX = 80

_vml_SLOTS = 136

_vml_FIXW = 8 * 128 * 4

def _vml_wg_spec64(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
    body = "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, %d, 1, 1, 0;\\n"
    s = ""
    for nm, sd, cons in (("wgmma_acc64", 1, plus), ("wgmma_zero64", 0, eq)):
        s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
              "uint32_t a3, uint64_t desc) {\n  asm volatile(\"" % (nm, n)
              + body % (n, outs, a, na + 4, sd)
              + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(desc));\n}\n" % cons)
    return s

def _vml_wg_spec(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
    body = ("{\\n.reg .b64 dd;\\nmov.b64 dd, {%%%d, %%%d};\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 "
            "{%s}, {%s}, dd, %d, 1, 1, 0;\\n}\\n")
    s = ""
    for nm, sd, cons in (("wgmma_acc", 1, plus), ("wgmma_zero", 0, eq)):
        s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
              "uint32_t a3, uint32_t lo, uint32_t hi) {\n  asm volatile(\"" % (nm, n)
              + body % (na + 4, na + 5, n, outs, a, sd)
              + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}\n" % cons)
    return s

_vml_CUDA_T = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace vm {
constexpr int KDIM = 4096, E_LOC = 32, TOPK = 8;
constexpr int SW = 18432;
constexpr int US = 2 * SW;
#ifndef VM_JN
#define VM_JN 64
#endif
#ifndef VM_NTM
#define VM_NTM (VM_JN / 8)
#endif
#ifndef VM_NX
#define VM_NX 3
#endif
#ifndef VM_XSN
#define VM_XSN 64
#endif
#ifndef VM_MKW
#define VM_MKW 32
#endif
static_assert(VM_NTM * 8 == VM_JN && VM_JN <= VM_XSN && VM_NTM <= 16 && (VM_MKW == 32 || VM_MKW == 64), "m32e config");
constexpr int NS = 3, NX = VM_NX;
constexpr int XS = VM_XSN * 512;
constexpr int MKW = VM_MKW, MKB = (VM_MKW == 64) ? 24 : 16, MKX = (VM_MKW == 64) ? 40 : 24, MKP = (VM_MKW == 64) ? 45 : 29;
constexpr int NCW = 16;
#ifndef VM_WG5
#define VM_WG5 0
#endif
#ifndef VM_PREG
#define VM_PREG 56
#endif
#ifndef VM_CREG
#define VM_CREG 112
#endif
#define VM_S2(x) #x
#define VM_S(x) VM_S2(x)
constexpr int NTH = 32 * NCW + (VM_WG5 ? 128 : 64);
#if VM_WG5 >= 2
#define VM_RSYNC() bar_sync(8, 32 * NCW)
#define VM_RNT (32 * NCW)
#else
#define VM_RSYNC() __syncthreads()
#define VM_RNT NTH
#endif
#ifndef VM_W5B6
#define VM_W5B6 1   // 0 = the R-m32e-14..17 barrier-6 count (race on ent; stress-test arm only)
#endif
#define VM_B6 ((VM_WG5 >= 2 && VM_W5B6) ? 32 * NCW + 96 : 32 * NCW + 32)
constexpr int JMAX = @JMAX@, GMAX = 136, SLOTS = @SLOTS@, FIXW = (@FIXW@ > VM_NTM * 512) ? @FIXW@ : VM_NTM * 512;
#ifndef VM_DP
#define VM_DP 0
#endif
#ifndef VM_DPI
#define VM_DPI 4
#endif
constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);
#ifndef VM_BRED
#define VM_BRED 0
#endif
#ifndef VM_DEFER2
#define VM_DEFER2 0
#endif
#ifndef VM_PS
#define VM_PS 0
#endif
#ifndef VM_AD
#define VM_AD 0
#endif
#ifndef VM_GSM
#define VM_GSM 0
#endif
#ifndef VM_UD
#define VM_UD 0
#endif
#ifndef VM_SYNC7
#define VM_SYNC7 0
#endif
#ifndef VM_NOB9
#define VM_NOB9 0
#endif
#ifndef VM_LD
#define VM_LD 0
#endif
#ifndef VM_FT
#define VM_FT 0
#endif
#ifndef VM_FTR
#define VM_FTR 0
#endif
#ifndef VM_TA
#define VM_TA 0
#endif
#if VM_NOB9
#define VM_F9(a) (0)
#else
#define VM_F9(a) ((a).flags & 512)
#endif
#ifndef VM_FSW
#define VM_FSW 0
#endif
#if VM_MKW == 64
#define SEGMK(...) __VA_ARGS__
#else
#define SEGMK(...)
#endif
static_assert(!(VM_BRED && VM_DP), "m32e: VM_BRED and VM_DP use different split paths");
#ifndef VM_SC
#define VM_SC 0
#endif
static_assert(!(VM_SC && VM_DP), "m32e: VM_SC and VM_DP are separate plans");
constexpr int BAR_OFF = NS * US + NX * XS;
constexpr int TAB_OFF = BAR_OFF + 8 * 2 * (NS + NX);
constexpr long long WAIT_BUDGET = 1ll << 31;

struct Tab {
int nj, R, Rpad, pad0;
int wlast[4];
int jE[JMAX], jC[JMAX], jS[JMAX + 1], jrow[JMAX + 1];
uint32_t cpA[JMAX + 1], cpB[JMAX + 1];
int bA[GMAX + 1], bB[GMAX + 1];
int est[E_LOC];
#if VM_GSM
float gs13[E_LOC], gs2[E_LOC];
#endif
int bready, pad1;                 // bit 9: B stage orders below are written (x-producer lanes 1-31, during phase A)
uint32_t brd[16];                 // bit 9 scratch: predicted readiness of one job's 16 B stages
uint8_t bpos[16];                 // bit 9 scratch: order position -> stage
#if VM_DP
int sA, sB, qpub, qpad;
uint32_t cA, cB;
int q[16];
#endif
unsigned long long bperm[JMAX];   // bit 9: per job, 16 x 4 bits: B order position -> stage
};
constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);

struct Args {
const __nv_bfloat16* x; const int* ids; const float* wts; int M; int ls;
const uint8_t* ws13; const uint8_t* ws2; const float* gsc13; const float* gsc2; const float* nw;
__nv_bfloat16* xA; __nv_bfloat16* xB; float* y; float* fix; int* cnt; int* xready; int* doneA; int* rmap; int* gbar;
__nv_bfloat16* out; int* err; long long* marks; unsigned long long magG; int flags;
uint32_t cost[2 * VM_NTM]; unsigned long long mag[2 * VM_NTM];
#if VM_DP
int poolA, poolB;
#endif
#if VM_SC
int segA, segB;
#endif
#if VM_PS
int pdel;
#endif
#if VM_MKW == 64
int G0;
#endif
};
__device__ __forceinline__ int split_point(const Args& A, int pb, int g0, int g1) {
const int ue = ((g1 - 1) >> 4) << 4;
return ((A.flags & 1) && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}

__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {   // c, opaque to NVVM: the 3-way gather stays one IMAD
  uint32_t r;
  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));
  return r;
}
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // UNP L2 layout (tools/unp_patches.py)
o[0] = w & 0x81C081C0u;
o[1] = __funnelshift_l(w, w, 10) & 0x81C081C0u;
o[2] = __funnelshift_l(w, w, 13) & 0x81C081C0u;
o[3] = ((w & 0x001B001Bu) * unp_ok32(0x1084u)) & 0x81C081C0u;   // UNP L2: carry-free bit gather
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
uint32_t d;
asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
return d;
}
#if VM_FSW
__device__ __forceinline__ float swiglu(float g, float u) { return __fdividef(g, 1.f + __expf(-g)) * u; }
#else
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
#endif
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
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
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); return; }
}
}
}
__device__ __forceinline__ int ld_acquire(const int* p) {
int v;
asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
return v;
}
__device__ __forceinline__ void wait_flag(const int* p, int target, int* err) {
if (ld_acquire(p) >= target) return;
const long long t0 = clock64();
while (ld_acquire(p) < target) {
__nanosleep(64);
if (*reinterpret_cast<volatile int*>(err) != 0) return;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 2); return; }
}
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int atom_acqrel_add(int* p, int v) {
int o;
asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(o) : "l"(p), "r"(v) : "memory");
return o;
}
__device__ __forceinline__ void red_add_v4(float* p, float a, float b, float c, float d) {
asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.global;\n" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
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
__device__ __noinline__ long long lat_probe(const uint8_t* p, int flags) {   // marks only: cycles per dependent L2 load
const unsigned z = (unsigned)flags >> 30;   // 0, opaque to the compiler
unsigned v;
asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
const long long t0 = clock64();
#pragma unroll 1
for (int k = 0; k < 8; ++k) asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p + (v & z)) : "memory");
asm volatile("add.u32 %0, %0, 0;\n" : "+r"(v));
return (clock64() - t0 + (long long)(v & z)) / 7;
}
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_acc(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <int N> __device__ __forceinline__ void wgmma_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <int N> __device__ __forceinline__ void wgmma_acc64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);
template <int N> __device__ __forceinline__ void wgmma_zero64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);
@WGSPECS@
__device__ __forceinline__ uint32_t mdiv(uint32_t a, unsigned long long m) {
const unsigned long long lo = (unsigned long long)a * (m & 0xFFFFFFFFull);
const unsigned long long hi = (unsigned long long)a * (m >> 32);
return (uint32_t)((hi + (lo >> 32)) >> 8);
}
__device__ __forceinline__ int excl_scan(int v, int lane) {
int s = v;
#pragma unroll
for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, s, o); if (lane >= o) s += n; }
return s - v;
}

__device__ __forceinline__ int stage_at(const Args& A, const Tab& T, int pb, int c, int G, unsigned long long magG) {
const int nj = T.nj;
if (nj == 0) return 0;
#if VM_DP
if (c >= G) return pb ? T.sB : T.sA;
const uint32_t* cp = pb ? T.cpB : T.cpA;
const uint32_t x = mdiv((pb ? T.cB : T.cA) * (uint32_t)c, magG);
#else
if (c >= G) return nj << (9 - pb);
const uint32_t* cp = pb ? T.cpB : T.cpA;
const uint32_t x = mdiv(cp[nj] * (uint32_t)c, magG);
#endif
int lo = 0, hi = nj;
while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (cp[mid] <= x) lo = mid; else hi = mid; }
const int nt = (T.jC[lo] + 7) >> 3;
#if VM_SC
const uint32_t xr = x - cp[lo], cs = A.cost[VM_NTM * pb + nt - 1], sg = (uint32_t)(pb ? A.segB : A.segA);
const uint32_t uc = 16u * cs + sg, uu = xr / uc, rr = xr - uu * uc;
uint32_t st = 16u * uu + (rr > sg ? (rr - sg) / cs : 0u);
#else
uint32_t st = mdiv(x - cp[lo], A.mag[VM_NTM * pb + nt - 1]);
#endif
const uint32_t ns = 512u >> pb;
if (st > ns - 1) st = ns - 1;
#if VM_SC
if (A.flags & 16384) {
const uint32_t ks = (pb ? (uint32_t)A.segB : (uint32_t)A.segA) / A.cost[VM_NTM * pb + nt - 1], w = st & 15u;
if (w <= ks) st -= w;
else if (16u - w <= ks) st += 16u - w;
}
#endif
return (lo << (9 - pb)) + (int)st;
}
__device__ __forceinline__ int cta_of(const Tab& T, int pb, int s, int G) {
const int* b = pb ? T.bB : T.bA;
int lo = 0, hi = G;
while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (b[mid] <= s) lo = mid; else hi = mid; }
return lo;
}
#ifndef VM_RT
#define VM_RT 0
#endif
#if VM_RT
static_assert(!VM_DP && !VM_SC && NCW >= 11 && GMAX + 1 <= 160 && 2 * VM_NTM <= 32, "VM_RT: static plan, warps 0-4 / 5-9 cover bA / bB");
__device__ __forceinline__ unsigned rt_eq(unsigned v, const unsigned* bb, int x) {   // lanes whose 5-bit id equals x
unsigned m = v;
#pragma unroll
for (int b = 0; b < 5; ++b) m &= ((x >> b) & 1) ? bb[b] : ~bb[b];
return m;
}
__device__ void route(const Args& A, Tab& T, int* hist, uint16_t* ent, uint32_t* fin) {   // R-rt-1: two barriers
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const int E8 = A.M * TOPK;
const int C = (((E8 + 15) >> 4) + 31) & ~31;
const int e0 = warp * C, e1 = min(E8, e0 + C);
int xv[5];
#pragma unroll
for (int k = 0; k < 5; ++k) xv[k] = __ldg(A.ids + min(e0 + 32 * k + lane, E8 - 1));   // unpredicated: all 5 loads in flight at once
#pragma unroll
for (int k = 0; k < 5; ++k) {
const int e = e0 + 32 * k + lane, id = xv[k] - A.ls;
xv[k] = (warp < NCW && 32 * k < C && e < e1 && (unsigned)id < (unsigned)E_LOC) ? id : -1;
}
uint32_t* scost = reinterpret_cast<uint32_t*>(hist + 768);   // cost / mag tables in smem: no indexed param (LDC) chains
unsigned long long* smag = reinterpret_cast<unsigned long long*>(hist + 832);
const int tq = min(tid, 2 * VM_NTM - 1);   // issued with the ids loads, stored after the ballots (LDC latency overlapped)
const uint32_t cq = A.cost[tq];
const unsigned long long mq = A.mag[tq];
int kb[5], kr[5];   // per chunk: entries of expert `lane` in this warp's earlier chunks; rank of this lane's entry in the chunk
if (warp < NCW) {
int run = 0;
#pragma unroll
for (int k = 0; k < 5; ++k) {
kb[k] = run;
kr[k] = 0;
if (e0 + 32 * k < e1) {
const int x = xv[k];
unsigned bb[5];
const unsigned v = __ballot_sync(0xffffffffu, x >= 0);
#pragma unroll
for (int b = 0; b < 5; ++b) bb[b] = __ballot_sync(0xffffffffu, (x >> b) & 1);
kr[k] = __popc(rt_eq(v, bb, x) & ((1u << lane) - 1u));
run += __popc(rt_eq(v, bb, lane));
}
}
hist[warp * 33 + lane] = run;
}
if (tid < 2 * VM_NTM) { scost[tid] = cq; smag[tid] = mq; }
VM_RSYNC();
if (warp < NCW) {
int tot = 0, pre = 0;
#pragma unroll
for (int w = 0; w < NCW; ++w) { const int h = hist[w * 33 + lane]; pre += (w < warp) ? h : 0; tot += h; }
const int est = excl_scan(tot, lane);
#pragma unroll
for (int k = 0; k < 5; ++k) {   // route rows: r = est[x] + earlier warps' x entries + this warp's earlier chunks + chunk rank
if (e0 + 32 * k >= e1) break;
const int e = e0 + 32 * k + lane, x = xv[k];
const int rb = __shfl_sync(0xffffffffu, est + pre + kb[k], x & 31);
const int r = (x >= 0) ? rb + kr[k] : -1;
if (x >= 0) ent[r] = (uint16_t)e;
if (blockIdx.x == 0 && e < e1) A.rmap[e] = r;
}
int* pl = hist + 528;   // plan handoff (last warp -> warps 0-9): 6 x 32 per-expert values + 3 scalars
if (warp == NCW - 1) {   // the plan, once: job split (the same closed form), padded rows, per-expert costs and their scans
#if VM_JN == 64
const int nj = (tot + 63) >> 6;
int base = 64;
#if VM_TA
if (nj) base = ((((tot + 7) >> 3) + nj - 1) / nj) << 3;   // VTR: tile-aligned split
#else
if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));
#endif
#else
const int nj = (tot + VM_JN - 1) / VM_JN;
#if VM_TA
const int base = nj ? ((((tot + 7) >> 3) + nj - 1) / nj) << 3 : VM_JN;   // VTR: tile-aligned split
#else
const int base = nj ? (tot + nj - 1) / nj : VM_JN;
#endif
#endif
int prow = 0, ca = 0, cb = 0;
for (int k = 0; k < nj; ++k) {
const int sz = min(base, tot - k * base), nt = ((sz + 7) >> 3) - 1;
prow += (sz + 7) & ~7;
ca += (int)scost[nt];
cb += (int)scost[VM_NTM + nt];
}
const int pjr = excl_scan(nj | (prow << 16), lane);   // jobs <= 80, padded rows <= 3200: two 16-bit fields, no carry
const int jst = pjr & 0xFFFF, rst = pjr >> 16, sca = excl_scan(ca, lane), scb = excl_scan(cb, lane);
const uint32_t csA = 512u * (uint32_t)sca, csB = 256u * (uint32_t)scb;
const int njT = __shfl_sync(0xffffffffu, jst + nj, 31);
const uint32_t totA = __shfl_sync(0xffffffffu, csA + 512u * (uint32_t)ca, 31), totB = __shfl_sync(0xffffffffu, csB + 256u * (uint32_t)cb, 31);
pl[lane] = (int)csA; pl[32 + lane] = (int)csB; pl[64 + lane] = nj; pl[96 + lane] = base; pl[128 + lane] = tot; pl[160 + lane] = jst;
if (lane == 0) { pl[192] = njT; pl[193] = (int)totA; pl[194] = (int)totB; }
bar_arrive(10, 32 * 11);
int r = rst;
uint32_t pa = csA, pq = csB;
for (int k = 0; k < nj; ++k) {
const int j = jst + k, sz = min(base, tot - k * base), nt = ((sz + 7) >> 3) - 1;
T.jE[j] = lane; T.jC[j] = sz; T.jS[j] = est + k * base; T.jrow[j] = r; T.cpA[j] = pa; T.cpB[j] = pq;
r += (sz + 7) & ~7;
pa += 512u * scost[nt];
pq += 256u * scost[VM_NTM + nt];
}
T.est[lane] = est;
#if VM_GSM
T.gs13[lane] = __ldg(A.gsc13 + lane); T.gs2[lane] = __ldg(A.gsc2 + lane);
#endif
if (lane == 31) {
T.nj = njT; T.R = est + tot; T.Rpad = rst + prow;
T.jS[njT] = est + tot; T.jrow[njT] = rst + prow; T.cpA[njT] = totA; T.cpB[njT] = totB;
}
} else if (warp < 10) {   // stage_at(pb, c) for c = 0..G: warps 0-4 phase A, 5-9 phase B
bar_sync(10, 32 * 11);
const int G = gridDim.x, pb = (warp >= 5) ? 1 : 0, c = tid - 160 * pb;
const uint32_t cs = (uint32_t)pl[32 * pb + lane];
const int nj = pl[64 + lane], base = pl[96 + lane], tq2 = pl[128 + lane], jst = pl[160 + lane];
const int njT = pl[192];
const uint32_t totP = (uint32_t)pl[193 + pb];
const int lastne = 31 - __clz(__ballot_sync(0xffffffffu, nj > 0));   // the last expert with jobs (-1: none)
const uint32_t x = mdiv(totP * (uint32_t)c, A.magG);
int lo = 0;
#pragma unroll
for (int s = 16; s >= 1; s >>= 1) { const uint32_t v = __shfl_sync(0xffffffffu, cs, (lo + s) & 31); if (lo + s <= lastne && v <= x) lo += s; }
const int njE = __shfl_sync(0xffffffffu, nj, lo), bE = __shfl_sync(0xffffffffu, base, lo), tE = __shfl_sync(0xffffffffu, tq2, lo);
const int jE0 = __shfl_sync(0xffffffffu, jst, lo);
uint32_t cp = __shfl_sync(0xffffffffu, cs, lo);
const uint32_t sc = 512u >> pb;
int k = 0, sz = min(bE, tE);
while (k + 1 < njE) {
const uint32_t w = sc * scost[VM_NTM * pb + max(1, (sz + 7) >> 3) - 1];
if (cp + w > x) break;
cp += w;
++k;
sz = min(bE, tE - k * bE);
}
const int nt = max(1, (sz + 7) >> 3);
uint32_t st = mdiv(x - cp, smag[VM_NTM * pb + nt - 1]);
const uint32_t ns = 512u >> pb;
if (st > ns - 1) st = ns - 1;
const int res = (njT == 0) ? 0 : (c >= G) ? (njT << (9 - pb)) : (((jE0 + k) << (9 - pb)) + (int)st);
if (c <= G) { if (pb) T.bB[c] = res; else T.bA[c] = res; }
}
}
if (tid == 0) T.bready = 0;
VM_RSYNC();
}
#else
__device__ void route(const Args& A, Tab& T, int* hist, uint16_t* ent, uint32_t* fin) {
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const int E8 = A.M * TOPK;
const int C = (((E8 + 15) >> 4) + 31) & ~31;
const int e0 = warp * C, e1 = min(E8, e0 + C);
int xv[5];
#pragma unroll
for (int k = 0; k < 5; ++k) {
const int e = e0 + 32 * k + lane;
xv[k] = -1;
if (warp < NCW && 32 * k < C && e < e1) { const int id = __ldg(A.ids + e) - A.ls; xv[k] = ((unsigned)id < (unsigned)E_LOC) ? id : -1; }
}
for (int i = tid; i < 16 * 33; i += VM_RNT) hist[i] = 0;
VM_RSYNC();
if (warp < NCW) {
#pragma unroll
for (int k = 0; k < 5; ++k) {
if (e0 + 32 * k >= e1) break;
const int x = xv[k];
const unsigned m = __match_any_sync(0xffffffffu, x);
if (x >= 0 && lane == __ffs(m) - 1) hist[warp * 33 + x] += __popc(m);
__syncwarp();
}
}
VM_RSYNC();
if (warp == 0) {
int tot = 0;
for (int w = 0; w < NCW; ++w) { const int h = hist[w * 33 + lane]; hist[w * 33 + lane] = tot; tot += h; }
const int est = excl_scan(tot, lane);
#if VM_JN == 64 && !VM_DP
const int nj = (tot + 63) >> 6;
int base = 64;
if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));   // == the countdown loop for every (base, nj, tot) (tools/gen_k3_mb.py asserts it); mb bit 13
#else
const int nj = (tot + VM_JN - 1) / VM_JN;
#if VM_TA
const int base = nj ? ((((tot + 7) >> 3) + nj - 1) / nj) << 3 : VM_JN;   // VTR: tile-aligned split
#else
const int base = nj ? (tot + nj - 1) / nj : VM_JN;
#endif
#endif
int prow = 0;
for (int k = 0; k < nj; ++k) { const int sz = min(base, tot - k * base); prow += (sz + 7) & ~7; }
#if VM_DP
int rank = lane;
if (A.flags & 8192) { rank = 0; for (int k = 0; k < 32; ++k) { const int t2 = __shfl_sync(0xffffffffu, tot, k); rank += (t2 > tot) || (t2 == tot && k < lane); } }
fin[rank] = (uint32_t)nj; fin[32 + rank] = (uint32_t)prow;
__syncwarp();
const int jsr = excl_scan((int)fin[lane], lane), rsr = excl_scan((int)fin[32 + lane], lane);
fin[64 + lane] = (uint32_t)jsr; fin[96 + lane] = (uint32_t)rsr;
__syncwarp();
const int jst = (int)fin[64 + rank], rst = (int)fin[96 + rank];
#else
const int jst = excl_scan(nj, lane), rst = excl_scan(prow, lane);
#endif
int r = rst;
for (int k = 0; k < nj; ++k) {
const int j = jst + k, sz = min(base, tot - k * base);
T.jE[j] = lane; T.jC[j] = sz; T.jS[j] = est + k * base; T.jrow[j] = r;
r += (sz + 7) & ~7;
}
T.est[lane] = est;
#if VM_GSM
T.gs13[lane] = __ldg(A.gsc13 + lane); T.gs2[lane] = __ldg(A.gsc2 + lane);
#endif
#if VM_DP
{
const int njT = __reduce_add_sync(0xffffffffu, nj), RT = __reduce_add_sync(0xffffffffu, tot), RpT = __reduce_add_sync(0xffffffffu, prow);
if (lane == 0) { T.nj = njT; T.R = RT; T.Rpad = RpT; T.jS[njT] = RT; T.jrow[njT] = RpT; }
}
#else
if (lane == 31) {
T.nj = jst + nj; T.R = est + tot; T.Rpad = rst + prow;
T.jS[jst + nj] = est + tot; T.jrow[jst + nj] = rst + prow;
}
#endif
}
VM_RSYNC();
if (warp < NCW) {
#pragma unroll
for (int k = 0; k < 5; ++k) {
if (e0 + 32 * k >= e1) break;
const int e = e0 + 32 * k + lane, x = xv[k];
const unsigned m = __match_any_sync(0xffffffffu, x);
const int rank = __popc(m & ((1u << lane) - 1u));
int r = -1;
if (x >= 0) { r = T.est[x] + hist[warp * 33 + x] + rank; ent[r] = (uint16_t)e; }
if (blockIdx.x == 0 && e < e1) A.rmap[e] = r;
__syncwarp();
if (x >= 0 && lane == __ffs(m) - 1) hist[warp * 33 + x] += __popc(m);
__syncwarp();
}
}
VM_RSYNC();
if (warp == 0) {
uint32_t accA = 0, accB = 0;
for (int j0 = 0; j0 < T.nj; j0 += 32) {
const int j = j0 + lane;
const int nt = (j < T.nj) ? ((T.jC[j] + 7) >> 3) - 1 : 0;
const int ca = (j < T.nj) ? (int)A.cost[nt] : 0, cb = (j < T.nj) ? (int)A.cost[VM_NTM + nt] : 0;
#if VM_SC
const int ja = (j < T.nj) ? 512 * ca + 32 * A.segA : 0, jb = (j < T.nj) ? 256 * cb + 16 * A.segB : 0;
const int sc = excl_scan(ja, lane), sd = excl_scan(jb, lane);
if (j < T.nj) { T.cpA[j] = accA + (uint32_t)sc; T.cpB[j] = accB + (uint32_t)sd; }
accA += (uint32_t)__shfl_sync(0xffffffffu, sc + ja, 31);
accB += (uint32_t)__shfl_sync(0xffffffffu, sd + jb, 31);
#else
const int sc = excl_scan(ca, lane), sd = excl_scan(cb, lane);
if (j < T.nj) { T.cpA[j] = accA + 512u * (uint32_t)sc; T.cpB[j] = accB + 256u * (uint32_t)sd; }
accA += 512u * (uint32_t)__shfl_sync(0xffffffffu, sc + ca, 31);
accB += 256u * (uint32_t)__shfl_sync(0xffffffffu, sd + cb, 31);
#endif
}
if (lane == 0) { T.cpA[T.nj] = accA; T.cpB[T.nj] = accB; }
#if VM_DP
if (lane == 0) {
const int SA = T.nj << 9, SB = T.nj << 8;
const int pa = min(DPMAX, (int)(((long long)(SA >> 4) * A.poolA) >> 8)), pq = min(DPMAX, (int)(((long long)(SB >> 4) * A.poolB) >> 8));
T.sA = SA - (pa << 4); T.sB = SB - (pq << 4);
const int ja = T.sA >> 9, jb = T.sB >> 8;
T.cA = (ja < T.nj) ? T.cpA[ja] + (uint32_t)(T.sA - (ja << 9)) * A.cost[((T.jC[ja] + 7) >> 3) - 1] : T.cpA[T.nj];
T.cB = (jb < T.nj) ? T.cpB[jb] + (uint32_t)(T.sB - (jb << 8)) * A.cost[VM_NTM + ((T.jC[jb] + 7) >> 3) - 1] : T.cpB[T.nj];
T.qpub = 0;
}
#endif
}
VM_RSYNC();
const int G = gridDim.x;
if (tid <= G) { T.bA[tid] = stage_at(A, T, 0, tid, G, A.magG); T.bB[tid] = stage_at(A, T, 1, tid, G, A.magG); }
if (tid == 0) T.bready = 0;
VM_RSYNC();
}

#endif   // VM_RT
__device__ void gather(const Args& A, const Tab& T, const uint16_t* ent) {
const int tid = threadIdx.x, G = gridDim.x, c = blockIdx.x;
const int q0 = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
const int q1 = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
const int s = tid >> 5, kk = (tid & 31) * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
int j = 0;
constexpr int GB = 8;
for (int qq = q0; qq < q1; qq += GB) {
uint4 v[GB];
int off[GB];
#pragma unroll
for (int k = 0; k < GB; ++k) {
const int q = qq + k;
off[k] = -1;
if (q < q1) {
while (T.jrow[j + 1] <= q) ++j;
const int r = q - T.jrow[j], cn = T.jC[j];
if (r < cn) {
const int npad = (cn + 7) & ~7;
off[k] = T.jrow[j] * 2048 + s * npad * 128 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
v[k] = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + tid);
}
}
}
#pragma unroll
for (int k = 0; k < GB; ++k)
if (off[k] >= 0) {
uint32_t* d = xa + off[k];
d[0] = v[k].x; d[32] = v[k].y; d[64] = v[k].z; d[96] = v[k].w;
}
}
fence_proxy_async();
bar_sync(5, 32 * NCW);
if (tid == 0) {
int jj = 0;
for (int q = q0; q < q1;) {
while (T.jrow[jj + 1] <= q) ++jj;
const int e = min(q1, T.jrow[jj + 1]);
red_release_add(A.xready + jj, e - q);
q = e;
}
}
}

__device__ void gather_warp(const Args& A, const Tab& T, const uint16_t* ent, int lane) {
const int G = gridDim.x, c = blockIdx.x;
const int q0 = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
const int q1 = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
const int kk = lane * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
int j = 0;
for (int q = q0; q < q1; ++q) {
while (T.jrow[j + 1] <= q) ++j;
const int r = q - T.jrow[j], cn = T.jC[j];
if (r >= cn) continue;
const int npad = (cn + 7) & ~7;
const uint4* src = reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + lane;
uint4 v[16];
#pragma unroll
for (int m = 0; m < 16; ++m) v[m] = __ldg(src + 32 * m);
uint32_t* d = xa + T.jrow[j] * 2048 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
#pragma unroll
for (int m = 0; m < 16; ++m) {
uint32_t* dm = d + m * npad * 128;
dm[0] = v[m].x; dm[32] = v[m].y; dm[64] = v[m].z; dm[96] = v[m].w;
}
}
fence_proxy_async();
__syncwarp();
if (lane == 0) {
int jj = 0;
for (int q = q0; q < q1;) {
while (T.jrow[jj + 1] <= q) ++jj;
const int e = min(q1, T.jrow[jj + 1]);
red_release_add(A.xready + jj, e - q);
q = e;
}
}
}
#if VM_WG5
#ifndef VM_GV
#define VM_GV 8
#endif
__device__ void gather_w5(const Args& A, const Tab& T, const uint16_t* ent, int lane, int part) {
const int G = gridDim.x, c = blockIdx.x;
const int q0c = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
const int q1c = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
const int n = q1c - q0c, q0 = q0c + (n * part) / 3, q1 = q0c + (n * (part + 1)) / 3;
const int kk = lane * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
int j = 0;
for (int q = q0; q < q1; ++q) {
while (T.jrow[j + 1] <= q) ++j;
const int r = q - T.jrow[j], cn = T.jC[j];
if (r >= cn) continue;
const int npad = (cn + 7) & ~7;
const uint4* src = reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + lane;
uint32_t* d = xa + T.jrow[j] * 2048 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
#pragma unroll
for (int h = 0; h < 16 / VM_GV; ++h) {
uint4 v[VM_GV];
#pragma unroll
for (int m = 0; m < VM_GV; ++m) v[m] = __ldg(src + 32 * (VM_GV * h + m));
#pragma unroll
for (int m = 0; m < VM_GV; ++m) {
uint32_t* dm = d + (VM_GV * h + m) * npad * 128;
dm[0] = v[m].x; dm[32] = v[m].y; dm[64] = v[m].z; dm[96] = v[m].w;
}
}
}
fence_proxy_async();
__syncwarp();
if (lane == 0) {
int jj = 0;
for (int q = q0; q < q1;) {
while (T.jrow[jj + 1] <= q) ++jj;
const int e = min(q1, T.jrow[jj + 1]);
red_release_add(A.xready + jj, e - q);
q = e;
}
}
}
#endif
__device__ __forceinline__ int nfill_of(const Args& A, const Tab& T) {
return (A.flags & 64) ? min(NX - 1, T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) : 0;
}
__device__ void self_fill(const Args& A, const Tab& T, uint8_t* smem, uint32_t bars, const uint16_t* ent, int nf) {
const int tid = threadIdx.x, c = blockIdx.x;
const int g0 = T.bA[c], g1 = T.bA[c + 1], sp = split_point(A, 0, g0, g1);
for (int f = 0; f < nf; ++f) {
const int gs = (sp + f < g1) ? sp + f : sp + f - (g1 - g0);
const int j = gs >> 9, st = gs & 15, cn = T.jC[j];
uint32_t* xs = reinterpret_cast<uint32_t*>(smem + NS * US + f * XS);
for (int w = tid; w < cn * 32; w += 32 * NCW) {
const int r = w >> 5, ch = w & 31, i = ch >> 3, t = (ch >> 1) & 3, qb = (ch & 1) * 4;
const uint4 v = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM + 256 * st) + ch);
uint32_t* d = xs + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
d[0] = v.x; d[32] = v.y; d[64] = v.z; d[96] = v.w;
}
}
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
bar_sync(5, 32 * NCW);
if (tid == 0)
for (int f = 0; f < nf; ++f) mbar_arrive(bars + 8 * (2 * NS + f));
}

__device__ __forceinline__ uint32_t afin(const Args& A, const Tab& T, int U, int G) {
const int a = U << 4, b = a + 16;
int lo = 0, hi = G - 1;
while (lo < hi) { const int m = (lo + hi) >> 1; if (T.bA[m + 1] > a) hi = m; else lo = m + 1; }
uint32_t f = 0;
for (int c = lo; c < G && T.bA[c] < b; ++c) {
const int g0 = T.bA[c], g1 = T.bA[c + 1];
if (g1 <= g0) continue;
const int sp = split_point(A, 0, g0, g1), l2 = max(a, g0), h2 = min(b, g1);
if (h2 <= l2) continue;
const int pos = (l2 < sp) ? (g1 - sp) + (min(h2, sp) - 1 - g0) : h2 - 1 - sp;
const uint32_t fc = (uint32_t)(((unsigned long long)(pos + 1) << 16) / (unsigned)(g1 - g0));
f = fc > f ? fc : f;
}
return f;
}
__device__ void bperm_lanes(const Args& A, Tab& T, int lane) {
const unsigned mask = 0xfffffffeu;
const int c = blockIdx.x, G = gridDim.x, g0 = T.bB[c], g1 = T.bB[c + 1];
if (g1 > g0) {
for (int j = g0 >> 8; j <= (g1 - 1) >> 8; ++j) {
for (int s = lane - 1; s < 16; s += 31) {
const uint32_t f0 = afin(A, T, 32 * j + 2 * s, G), f1 = afin(A, T, 32 * j + 2 * s + 1, G);
T.brd[s] = f0 > f1 ? f0 : f1;
}
__syncwarp(mask);
for (int s = lane - 1; s < 16; s += 31) {
const uint32_t r = T.brd[s];
int rank = 0;
for (int t = 0; t < 16; ++t) { const uint32_t q = T.brd[t]; rank += (q < r) || (q == r && t < s); }
T.bpos[rank] = (uint8_t)s;
}
__syncwarp(mask);
if (lane == 1) {
unsigned long long pm = 0;
for (int r = 0; r < 16; ++r) pm |= (unsigned long long)T.bpos[r] << (4 * r);
T.bperm[j] = pm;
}
__syncwarp(mask);
}
}
if (lane == 1) { __threadfence_block(); *reinterpret_cast<volatile int*>(&T.bready) = 1; }
}
__device__ __forceinline__ void bperm_wait(const Args& A, const Tab& T) {
if (*reinterpret_cast<const volatile int*>(&T.bready) != 0) { __threadfence_block(); return; }
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.bready) == 0) {
__nanosleep(64);
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 3); break; }
}
__threadfence_block();
}
__device__ __forceinline__ int bstage(const Args& A, const Tab& T, int j, int s) {
return VM_F9(A) ? (int)((T.bperm[j] >> (4 * s)) & 15ull) : s;
}

__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {
const uint64_t pol = policy_evict_first();
#if VM_PS
const long long tps = gtime();
#endif
const int c = blockIdx.x;
int it = 0;
for (int pb = 0; pb < 2; ++pb) {
const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
if (pb && VM_F9(A)) bperm_wait(A, T);
for (int i = 0; i < g1 - g0; ++i, ++it) {
const int gs = (sp + i < g1) ? sp + i : sp + i - (g1 - g0);
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);
const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);
const int e = T.jE[j];
const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW
: A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
#if VM_PS
else if (it > 0 && A.pdel > 0) { const long long tw = tps + (long long)A.pdel * it; while (gtime() < tw) __nanosleep(32); }
#endif
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, US);
bulk_g2s_ef(dst, src, SW, fb, pol);
bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);
}
#if VM_DP
{
Tab& TT = const_cast<Tab&>(T);
const int S0 = pb ? T.sB : T.sA, nit = ((pb ? (T.nj << 8) : (T.nj << 9)) - S0) / VM_DPI;
for (;;) {
const int i0 = (nit > 0) ? atomicAdd(A.gbar + pb, 1) : 0;
const int v = (i0 < nit) ? i0 : -1;
const int k = TT.qpub;
TT.q[k & 15] = v;
__threadfence_block();
*reinterpret_cast<volatile int*>(&TT.qpub) = k + 1;
if (v < 0) break;
for (int s2 = 0; s2 < VM_DPI; ++s2, ++it) {
const int gs = S0 + v * VM_DPI + s2;
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);
const int s = gs & 15;
const int e = T.jE[j];
const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW
: A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, US);
bulk_g2s_ef(dst, src, SW, fb, pol);
bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);
}
}
}
#endif
}
}
__device__ void prod_x(const Args& A, const Tab& T, uint32_t xring, uint32_t bars) {
const int c = blockIdx.x, nf = nfill_of(A, T);
int it = 0;
long long wA = 0, wB = 0, tB0 = 0, tB1 = 0;   // marks only (v9): readiness waits of this CTA's x producer
int nB = 0;
#if VM_DP
int kq = 0;
#endif
for (int pb = 0; pb < 2; ++pb) {
const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
if (pb && VM_F9(A)) bperm_wait(A, T);
int lastj = -1;
for (int i = 0; i < g1 - g0; ++i, ++it) {
const int gs = (sp + i < g1) ? sp + i : sp + i - (g1 - g0);
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5);
const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);
const int npad = (T.jC[j] + 7) & ~7;
if (!pb && i < nf) continue;
if (!pb) {
if (j != lastj) {
if (A.marks && ld_acquire(A.xready + j) < npad) {
const long long t = gtime();
wait_flag(A.xready + j, npad, A.err);
wA += gtime() - t;
} else {
wait_flag(A.xready + j, npad, A.err);
}
fence_proxy_async(); lastj = j;
}
} else {
if (A.marks && ld_acquire(A.doneA + j * 16 + s) < 8) {
const long long t = gtime();
wait_flag(A.doneA + j * 16 + s, 8, A.err);
wB += gtime() - t; ++nB;
} else {
wait_flag(A.doneA + j * 16 + s, 8, A.err);
}
fence_proxy_async();
if (A.marks) { tB1 = gtime(); if (tB0 == 0) tB0 = tB1; }
}
const uint32_t bytes = (uint32_t)npad * 512u;
const __nv_bfloat16* src = (pb ? A.xB : A.xA) + (size_t)T.jrow[j] * KDIM + (size_t)s * npad * 256;
const int sl = it % NX;
if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * (2 * NS + sl);
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, src, bytes, fb);
}
#if VM_DP
{
const int S0 = pb ? T.sB : T.sA;
for (;;) {
if (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
__nanosleep(32);
if (*reinterpret_cast<volatile int*>(A.err) != 0) break;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 4); break; }
}
}
__threadfence_block();
const int v = T.q[kq & 15];
++kq;
if (v < 0 || *reinterpret_cast<volatile int*>(A.err) != 0) break;
for (int s2 = 0; s2 < VM_DPI; ++s2, ++it) {
const int gs = S0 + v * VM_DPI + s2;
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5);
const int s = gs & 15;
const int npad = (T.jC[j] + 7) & ~7;
if (!pb) wait_flag(A.xready + j, npad, A.err);
else wait_flag(A.doneA + j * 16 + s, 8, A.err);
fence_proxy_async();
const uint32_t bytes = (uint32_t)npad * 512u;
const __nv_bfloat16* src = (pb ? A.xB : A.xA) + (size_t)T.jrow[j] * KDIM + (size_t)s * npad * 256;
const int sl = it % NX;
if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * (2 * NS + sl);
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, src, bytes, fb);
}
}
}
#endif
}
if (A.marks) {   // cols 24..28: B doneA wait ns, B stages that waited, A xready wait ns, first / last B x acquire (gtime)
long long* m = A.marks + (size_t)c * MKW;
m[MKX] = wB; m[MKX + 1] = nB; m[MKX + 2] = wA; m[MKX + 3] = tB0; m[MKX + 4] = tB1;
}
}

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
#if VM_UD
template <int N, bool Z>
__device__ __forceinline__ void mma2u(float* d, const uint32_t* hh, uint64_t desc) {
if constexpr (Z) wgmma_zero64<N>(d, hh[0], hh[4], hh[1], hh[5], desc);
else wgmma_acc64<N>(d, hh[0], hh[4], hh[1], hh[5], desc);
wgmma_acc64<N>(d, hh[2], hh[6], hh[3], hh[7], desc + 16);
}
#endif
#ifndef VM_DEFER
#define VM_DEFER 0
#endif
#ifndef VM_TRICKS
#define VM_TRICKS 1
#endif
#ifndef VM_TRICKS_DQ
#define VM_TRICKS_DQ 1   // 0: VM_TRICKS without the opaque-multiply dequant (2 registers fewer)
#endif
__device__ __forceinline__ void deq4_dq(uint32_t w, uint32_t* o, uint32_t k64, uint32_t k4) {   // UNP L2: == deq4
(void)k64; (void)k4;
deq4(w, o);
}
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {   // a x (b.lo, b.lo): HMUL2 .H0_H0
__nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {   // a x (b.hi, b.hi): HMUL2 .H1_H1
__nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void dec_half2(uint32_t w0, uint32_t w8, uint32_t sw2, uint32_t par, uint32_t* hh, uint32_t k16,
uint32_t k64, uint32_t k4) {
#if VM_TRICKS_DQ
deq4_dq(w0, hh, k64, k4); deq4_dq(w8, hh + 4, k64, k4);
#else
deq4(w0, hh); deq4(w8, hh + 4);
#endif
const uint32_t sb = 2u * par;
const uint32_t sgp = __byte_perm(sw2, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16 + 0x75807580u;
#pragma unroll
for (int j = 0; j < 4; ++j) { hh[j] = bmul2l(hh[j], sgp); hh[4 + j] = bmul2h(hh[4 + j], sgp); }
}
#if VM_AD
struct Pend { int i; int su, sslot, snt, scnt, son, stag; };
#else
struct Pend { int i; };
#endif
__device__ __forceinline__ void pend_flush(const Args& A, Pend& pd, int q, int wp, int lane) {
if (pd.i >= 0) {
fence_proxy_async();
bar_sync(1 + q, 128);
if (((wp << 5) | lane) == 0) red_release_add(A.doneA + pd.i, 1);
pd.i = -1;
}
}
#ifndef VCP_NB
#define VCP_NB 2
#endif
#ifndef VCP_W
#define VCP_W 1
#endif
#ifndef VCP_G2
#define VCP_G2 0
#endif
#ifndef VCP_NMIN
#define VCP_NMIN 24
#endif
#ifndef VCP_NMAX
#define VCP_NMAX 128
#endif
#ifndef VCP_PK
#define VCP_PK 0
#endif
#ifndef VCP_PRE
#define VCP_PRE 0
#endif
#ifndef VCP_EA
#define VCP_EA 0
#endif
#ifndef VCP_EAN
#define VCP_EAN 48
#endif
#ifndef VCP_M1
#define VCP_M1 0
#endif
#ifndef VCP_LDA
#define VCP_LDA 0
#endif
#ifndef VCP_DPI
#define VCP_DPI 1
#endif
#ifndef VCP_DPH
#define VCP_DPH 1
#endif
template <int N> struct VcpCfg {   // R-vcp: stage pipeline shape per N (defaults = the shipped 2 buffers, wait 1, 8 groups)
static constexpr bool on = N >= VCP_NMIN && N <= VCP_NMAX;
static constexpr int NB = on ? VCP_NB : 2;
static constexpr int W = on ? VCP_W : 1;
static constexpr bool G2 = on && VCP_G2;
static constexpr bool PRE = on && VCP_PRE;
static_assert(!PRE || (NB == 4 && W == 1 && !G2), "VCP_PRE needs VCP_NB 4, VCP_W 1");
static constexpr int EA = VCP_EA ? (N <= VCP_EAN ? 1 : 2) : 0;   // 1: up-front codes, 2: c[1..3] after group 0
static_assert(NB == 2 || NB == 4, "VCP_NB");
static_assert(W >= 1 && W <= (G2 ? NB / 2 - 1 : NB - 1), "VCP_W too deep for VCP_NB");
};
template <int N>
__device__ __forceinline__ void stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
uint32_t (&hb)[VcpCfg<N>::NB][8], int q, int wp, int lane, bool first, Pend& pd) {
using VC = VcpCfg<N>;
#if VM_UD
static_assert(VC::NB == 2 && VC::W == 1 && !VC::G2, "VCP needs VM_UD 0");
#endif
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + (q >> 1) * SW;
const int tile = (q & 1) * 4 + wp;
const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
uint4 c[4];
#if VM_TRICKS
c[0] = cp[0];
const uint32_t sw0 = sp[0], sw1 = sp[32];
if constexpr (VC::EA == 1) {   // VCP (a): every code word up front, the weight slot freed before the x wait
#pragma unroll
for (int i = 1; i < 4; ++i) c[i] = cp[32 * i];
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
}
#else
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
const uint32_t sw0 = sp[0], sw1 = sp[32];
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
#endif
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
#if VM_TRICKS
const uint32_t zo = (uint32_t)A.flags >> 30;   // 0 (flags < 2^30), opaque to the compiler
const uint32_t k16 = 16u + zo, k64 = VM_TRICKS_DQ ? 64u + zo : 64u, k4 = VM_TRICKS_DQ ? 4u + zo : 4u;
const uint64_t d0 = wg_desc_sa(sa(smem + NS * US), 4096u);
const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);
#if VM_UD
const uint64_t dsc = d0 + (uint64_t)((uint32_t)xs * (uint32_t)(XS >> 4));
#endif
#else
const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);
const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#if VM_UD
const uint64_t dsc = d0;
#endif
#endif
#if VM_TRICKS
if constexpr (VC::PRE) {   // VCP (2'): group g + 1 unpacked between commit(g) and wait(g - 1): 2 groups in flight during it
#pragma unroll
for (int g = 0; g < 8; ++g) {
const int i = g >> 1, hf = g & 1;
if (g == 0) {
c[1] = cp[32];
dec_half2(c[0].x, c[0].z, sw0, 0u, hb[0], k16, k64, k4);
}
wg_fence();
if (first && g == 0) mma2<N, true>(d, hb[g % VC::NB], dlo + 64 * i + 32 * hf, dhi);
else mma2<N, false>(d, hb[g % VC::NB], dlo + 64 * i + 32 * hf, dhi);
wg_commit();
if (g < 7) {
const int g1 = g + 1, i1 = g1 >> 1, h1 = g1 & 1;
if (h1 == 0 && i1 < 3) c[i1 + 1] = cp[32 * (i1 + 1)];   // codes one k64 step ahead of their unpack
dec_half2(h1 ? c[i1].y : c[i1].x, h1 ? c[i1].w : c[i1].z, (i1 < 2) ? sw0 : sw1, (uint32_t)(i1 & 1), hb[g1 % VC::NB], k16, k64, k4);
if (g1 == 6) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // all codes in registers
}
wg_wait<1>();
if (g == 0 && prevx >= 0) {
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
prevx = -1;
}
if constexpr (VM_DEFER) {
if (first && i == VCP_DPI && hf == VCP_DPH) pend_flush(A, pd, q, wp, lane);
}
}
} else
#endif
{
#pragma unroll
for (int i = 0; i < 4; ++i) {
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
#if VM_TRICKS
if (VC::EA == 0 && i < 3) c[i + 1] = cp[32 * (i + 1)];   // the next 4-step group's codes, one group ahead
#else
const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#endif
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t* hh = hb[(2 * i + hf) % VC::NB];
#if VM_TRICKS
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16, k64, k4);
if (VC::EA == 0 && i == 3 && hf == 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // all codes in registers
if (VC::EA == 2 && i == 0 && hf == 1) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // VCP (a), N > VCP_EAN
#else
dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
#endif
if (!VC::G2 || hf == 1) {
wg_fence();
if constexpr (VC::G2) {   // VCP (3): one commit group per k64 step (both k32 halves)
uint32_t* h0 = hb[(2 * i) % VC::NB];
if (first && i == 0) mma2<N, true>(d, h0, dlo + 64 * i, dhi);
else mma2<N, false>(d, h0, dlo + 64 * i, dhi);
mma2<N, false>(d, hh, dlo + 64 * i + 32, dhi);
} else {
if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);
else mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
}
wg_commit();
if (VC::EA == 2 && i == 0 && hf == 0) {
#pragma unroll
for (int k = 1; k < 4; ++k) c[k] = cp[32 * k];
}
wg_wait<VC::W>();
if ((VC::G2 ? i : 2 * i + hf) == VC::W - 1 && prevx >= 0) {   // the previous stage's last group is complete
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
prevx = -1;
}
if constexpr (VM_DEFER) {
if (first && i == VCP_DPI && (VC::G2 || hf == VCP_DPH)) pend_flush(A, pd, q, wp, lane);   // shipped point: behind commit group 3 (G2: the only commit of step i)
}
}
}
}
}
prevx = xs;
++it;
}
__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
: "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
template <int N>
__device__ __forceinline__ void stage_m(const Args& A, uint8_t* smem, uint32_t bars, int& it, float (&d2)[2][N / 2], int q,
int wp, int lane) {
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + (q >> 1) * SW;
const int tile = (q & 1) * 4 + wp;
const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
uint4 c[4];
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
const uint32_t sw0 = sp[0], sw1 = sp[32];
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
const uint32_t xb = sa(smem + NS * US + xs * XS) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
#pragma unroll
for (int i = 0; i < 4; ++i) {
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
const uint32_t k16q = 16u + ((uint32_t)A.flags >> 30);   // UNP SMQ: opaque 16 (flags < 2^30), as stage<N>
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t hh[8];
#if VCP_LDA || VCP_M1
uint32_t bb[N / 8][4];
#endif
#if VCP_LDA   // VCP (b): x fragments in flight during the unpack
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), bb[nt][0], bb[nt][1], bb[nt][2], bb[nt][3]);
#endif
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16q, 64u, 4u);   // UNP SMQ (tools/unp_patches.py)
#if VCP_LDA || VCP_M1
#if !VCP_LDA
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), bb[nt][0], bb[nt][1], bb[nt][2], bb[nt][3]);
#endif
if constexpr (VCP_M1 && N >= 16) {   // VCP (b): one accumulator, the nt chains interleaved
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], bb[nt][0], bb[nt][1]);
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) mma16816(&d2[0][4 * nt], hh[2], hh[6], hh[3], hh[7], bb[nt][2], bb[nt][3]);
} else {
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], bb[nt][0], bb[nt][1]);
mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], bb[nt][2], bb[nt][3]);
}
}
#else
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
uint32_t b0, b1, b2, b3;
ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);
mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);
}
#endif
}
}
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
++it;
}
template <bool PB, int N>
__device__ __forceinline__ void epilogue(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d,
Pend& pd) {
const int g = lane >> 2, t = lane & 3, wt = (wp << 5) | lane;
const int e = T.jE[j], cn = T.jC[j];
const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;
if constexpr (!PB) {
SEGMK(const long long t_a0 = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)
#if VM_GSM
const float gs = T.gs13[e];
#else
const float gs = __ldg(A.gsc13 + e);
#endif
const int npad = (cn + 7) & ~7;
const int col = 8 * tg + g, s2 = col >> 8, kk = col & 255;
const int i = kk >> 6, tt = (kk >> 4) & 3, qq = (kk >> 1) & 7, b = kk & 1;
__nv_bfloat16* xb = A.xB + (size_t)T.jrow[j] * KDIM + (size_t)s2 * npad * 256;
#if VCP_PK   // VCP: lanes g, g ^ 1 (lane ^ 4) hold adjacent bf16 of xB: one 4-B store per pair (same values)
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const __nv_bfloat162 v2 = __floats2bfloat162_rn(swiglu(d[4 * nt] * gs, d[4 * nt + 2] * gs), swiglu(d[4 * nt + 1] * gs, d[4 * nt + 3] * gs));
const uint32_t vs = *reinterpret_cast<const uint32_t*>(&v2);
const uint32_t vo = __shfl_xor_sync(0xffffffffu, vs, 4);
const uint32_t w2 = b ? __byte_perm(vo, vs, 0x7632) : __byte_perm(vs, vo, 0x5410);
const int n = 8 * nt + 2 * t + b;
if (n < cn) *reinterpret_cast<uint32_t*>(xb + ((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + b)) * 4 + tt) * 2) = w2;
}
if (false)
#endif
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn)
xb[((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + hh)) * 4 + tt) * 2 + b] =
__float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
}
SEGMK(if (A.marks && wt == 0 && q == 0) { A.marks[(size_t)blockIdx.x * MKW + 52] += 1; A.marks[(size_t)blockIdx.x * MKW + 53] += gtime() - t_a0; })
if constexpr (VM_DEFER || VM_DEFER2) {
pd.i = j * 16 + (ub >> 1);
} else {
SEGMK(const long long t_f = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)
fence_proxy_async();
bar_sync(1 + q, 128);
if (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);
SEGMK(if (A.marks && wt == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 63] += gtime() - t_f;)
}
} else {
SEGMK(const long long t_b0 = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)
#if VM_GSM
const float gs = T.gs2[e];
#else
const float gs = __ldg(A.gsc2 + e);
#endif
const int h0 = 16 * tg + g;
float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn) {
yb[(size_t)n * KDIM + h0] = d[4 * nt + hh] * gs;
yb[(size_t)n * KDIM + h0 + 8] = d[4 * nt + 2 + hh] * gs;
}
}
SEGMK(if (A.marks && wt == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 54] += gtime() - t_b0;)
}
}
#if VM_BRED
template <int N>
__device__ __forceinline__ void bred(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d) {
const int g = lane >> 2, t = lane & 3;
const int e = T.jE[j], cn = T.jC[j];
const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;
SEGMK(const long long t_r0 = (A.marks && wp == 0 && lane == 0 && q == 0) ? gtime() : 0;)
#if VM_GSM
const float gs = T.gs2[e];
#else
const float gs = __ldg(A.gsc2 + e);
#endif
const int h0 = 16 * tg + g;
float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn) {
asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(yb + (size_t)n * KDIM + h0), "f"(d[4 * nt + hh] * gs) : "memory");
asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(yb + (size_t)n * KDIM + h0 + 8), "f"(d[4 * nt + 2 + hh] * gs) : "memory");
}
}
SEGMK(if (A.marks && wp == 0 && lane == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 54] += gtime() - t_r0;)
}
#endif
#if VM_MKW == 64
struct SegMk { long long t0, t1, t2, t3; bool on; };
__device__ __forceinline__ void segmk_done(const Args& A, SegMk& m, int s0, int s1) {
if (m.on) {
const long long t4 = gtime();
long long* mk = A.marks + (size_t)blockIdx.x * MKW;
mk[56] += m.t1 - m.t0; mk[57] += m.t2 - m.t1; mk[58] += m.t3 - m.t2; mk[59] += t4 - m.t3; mk[60] += 1; mk[61] += s1 - s0 - 1;
}
}
#endif
#if VM_AD
template <int N>
__device__ __forceinline__ void m32e_finish(const Args& A, Tab& T, Pend& pd, int q, int wp, int lane) {
const int wt = (wp << 5) | lane;
float* fb = A.fix + ((size_t)(pd.sslot * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + pd.sslot * 4 + q;
float d[N / 2];
fence_acq_rel();
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
if (wt == 0) *fc = 0;
epilogue<false, N>(A, T, pd.su >> 5, pd.su & 31, q, wp, lane, d, pd);
}
__device__ __noinline__ void m32e_resolve(const Args& A, Tab& T, Pend& pd, int q, int wp, int lane) {
const int wt = (wp << 5) | lane;
int* fc = A.cnt + pd.sslot * 4 + q;
bar_sync(1 + q, 128);
if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, pd.scnt) + pd.scnt == 16);
bar_sync(1 + q, 128);
pd.son = 0;
if (!T.wlast[q]) return;
switch (pd.snt) {
case 1: m32e_finish<8>(A, T, pd, q, wp, lane); break;
case 2: m32e_finish<16>(A, T, pd, q, wp, lane); break;
case 3: m32e_finish<24>(A, T, pd, q, wp, lane); break;
case 4: m32e_finish<32>(A, T, pd, q, wp, lane); break;
case 5: m32e_finish<40>(A, T, pd, q, wp, lane); break;
case 6: m32e_finish<48>(A, T, pd, q, wp, lane); break;
case 7: m32e_finish<56>(A, T, pd, q, wp, lane); break;
default: m32e_finish<64>(A, T, pd, q, wp, lane); break;
}
}
#endif
template <bool PB, int N>
__device__ __forceinline__ void seg(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int u, int s0, int s1,
int q, int wp, int lane, Pend& pd) {
float d[N / 2];
SEGMK(SegMk smk{0, 0, 0, 0, A.marks != nullptr && q == 0 && wp == 0 && lane == 0}; if (smk.on) smk.t0 = gtime();)
bool sync_path = false;
#if VM_SYNC7
if constexpr (N <= 16) sync_path = true;
#else
if constexpr (N <= 16) sync_path = (A.flags & 128) != 0;
#endif
if (sync_path) {
float d2[2][N / 2];
#pragma unroll
for (int k = 0; k < N / 2; ++k) { d2[0][k] = 0.f; d2[1][k] = 0.f; }
stage_m<N>(A, smem, bars, it, d2, q, wp, lane);
SEGMK(if (smk.on) smk.t1 = gtime();)
if constexpr (VM_DEFER) pend_flush(A, pd, q, wp, lane);
#pragma unroll 1
for (int s = s0 + 1; s < s1; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane);
SEGMK(if (smk.on) smk.t2 = smk.t3 = gtime();)
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = d2[0][k] + d2[1][k];
} else {
uint32_t hb[VcpCfg<N>::NB][8];
int prevx = -1;
stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd);
SEGMK(if (smk.on) smk.t1 = gtime();)
#pragma unroll 1
for (int s = s0 + 1; s < s1; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd);
SEGMK(if (smk.on) smk.t2 = gtime();)
wg_wait<0>();
SEGMK(if (smk.on) smk.t3 = gtime();)
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
}
if constexpr (VM_DEFER2) pend_flush(A, pd, q, wp, lane);
constexpr int UBL = PB ? 4 : 5;
const int j = u >> UBL, ub = u & ((1 << UBL) - 1);
if (!(s0 == 0 && s1 == 16)) {
#if VM_BRED
if constexpr (PB) { bred<N>(A, T, j, ub, q, wp, lane, d); SEGMK(segmk_done(A, smk, s0, s1);) return; }
#endif
const int wt = (wp << 5) | lane;
#if VM_DP
const int s0u = (PB ? T.sB : T.sA) >> 4;
const int slot = (u >= s0u) ? (int)gridDim.x + (u - s0u) : cta_of(T, PB, u << 4, gridDim.x);
float* fb = A.fix + ((size_t)((PB * SLOTSD + slot) * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + (PB * SLOTSD + slot) * 4 + q;
#else
const int slot = cta_of(T, PB, u << 4, gridDim.x);
float* fb = A.fix + ((size_t)((PB * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + (PB * SLOTS + slot) * 4 + q;
#endif
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);
#if VM_AD
if constexpr (!PB) {
if (A.flags & 32768) {
if (pd.son) m32e_resolve(A, T, pd, q, wp, lane);
pd.son = 1; pd.su = u; pd.sslot = slot; pd.snt = N / 8; pd.scnt = s1 - s0; ++pd.stag;
SEGMK(segmk_done(A, smk, s0, s1);)
return;
}
}
#endif
SEGMK(long long t_at = smk.on ? gtime() : 0;)
bar_sync(1 + q, 128);
if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);
bar_sync(1 + q, 128);
SEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 62] += gtime() - t_at;)
if (!T.wlast[q]) { SEGMK(segmk_done(A, smk, s0, s1);) return; }
SEGMK(const long long t_rb = smk.on ? gtime() : 0;)
fence_acq_rel();
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
if (wt == 0) *fc = 0;
SEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 55] += gtime() - t_rb;)
}
epilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);
SEGMK(segmk_done(A, smk, s0, s1);)
}
template <bool PB>
#if VM_DP
__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane, int& kq) {
#else
__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane) {
#endif
const int c = blockIdx.x;
const int g0 = PB ? T.bB[c] : T.bA[c], g1 = PB ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, PB, g0, g1);
Pend pd{-1};
#if VM_DP
bool pool = false;
const int S0 = PB ? T.sB : T.sA;
#endif
#pragma unroll 1
for (int gs = sp, ge = g1; ;) {
#if VM_DP
int u, s0, s1;
if (!pool && gs >= ge) {
if (ge == g1 && sp > g0) { gs = g0; ge = sp; }
else {
pool = true;
#if VM_MKW == 64
if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * MKW + 54 + PB] = gtime();
#endif
}
}
if (pool) {
if (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
__nanosleep(32);
if (*reinterpret_cast<volatile int*>(A.err) != 0) break;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 5); break; }
}
}
__threadfence_block();
const int v = T.q[kq & 15];
++kq;
if (v < 0 || *reinterpret_cast<volatile int*>(A.err) != 0) break;
const int g2 = S0 + v * VM_DPI;
u = g2 >> 4; s0 = g2 & 15; s1 = s0 + VM_DPI;
#if VM_MKW == 64
if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * MKW + 52 + PB] += 1;
#endif
} else {
u = gs >> 4; s0 = gs & 15; s1 = min(16, s0 + (ge - gs));
}
#else
if (gs >= ge) {
if (ge == g1 && sp > g0) { gs = g0; ge = sp; } else break;
}
const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
#endif
const int j = PB ? (u >> 4) : (u >> 5);
#if VM_MKW == 64
if (A.marks && q == 0 && wp == 0 && lane == 0) {
long long* mk = A.marks + (size_t)c * MKW;
mk[8 + 16 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;
mk[49 + PB] += 1;
if (!(s0 == 0 && s1 == 16)) mk[51] += 1;
}
#else
if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * 32 + 8 + 8 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;
#endif
#if VM_AD
const int tag0 = pd.stag;
#endif
switch ((T.jC[j] + 7) >> 3) {
case 1: seg<PB, 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 2: seg<PB, 16>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 3: seg<PB, 24>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 4: seg<PB, 32>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 5: seg<PB, 40>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 6: seg<PB, 48>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 7: seg<PB, 56>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#if VM_NTM > 8
case 8: seg<PB, 64>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 9
case 9: seg<PB, 72>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 10
case 10: seg<PB, 80>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 11
case 11: seg<PB, 88>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 12
case 12: seg<PB, 96>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 13
case 13: seg<PB, 104>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 14
case 14: seg<PB, 112>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 15
case 15: seg<PB, 120>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
}
#if VM_AD
if constexpr (!PB) { if (pd.son && pd.stag == tag0) m32e_resolve(A, T, pd, q, wp, lane); }
#endif
gs += s1 - s0;
}
#if VM_AD
if constexpr (!PB) { if (pd.son) m32e_resolve(A, T, pd, q, wp, lane); }
#endif
if constexpr ((VM_DEFER || VM_DEFER2) && !PB) pend_flush(A, pd, q, wp, lane);   // before phase B: its x waits on these releases
}

__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {
extern __shared__ __align__(1024) uint8_t smem[];
Tab& T = *reinterpret_cast<Tab*>(smem + TAB_OFF);
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const uint32_t bars = sa(smem + BAR_OFF);
#if VM_WG5 >= 2
if (warp >= NCW) {   // producer WG: diverge before route / gather (no shared live range crosses the cap change)
asm volatile("setmaxnreg.dec.sync.aligned.u32 " VM_S(VM_PREG) ";\n" ::: "memory");
if (A.flags & 256) asm volatile("griddepcontrol.wait;\n" ::: "memory");
bar_sync(9, NTH);   // the consumers arrive after the mbarrier init + route()
const uint16_t* ent = reinterpret_cast<const uint16_t*>(smem + NS * US + (NX - 1) * XS + 4096);
if (warp > NCW + 1) { if (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1); if (VM_W5B6) bar_arrive(6, VM_B6); return; }
if (warp == NCW) {
if (A.flags & 4) bar_sync(7, 32 * NCW + 32);
if (lane == 0) prod_w(A, T, sa(smem), bars);
} else {
if (A.flags & 64) gather_w5(A, T, ent, lane, 0);
bar_sync(6, VM_B6);
if (lane == 0) prod_x(A, T, sa(smem + NS * US), bars);
else if VM_F9(A) bperm_lanes(A, T, lane);
}
return;
}
#if VM_WG5 == 2
asm volatile("setmaxnreg.inc.sync.aligned.u32 " VM_S(VM_CREG) ";\n" ::: "memory");
#endif
#endif
if (tid == 0) {
for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
if (A.marks) {   // v11: SM id + two L2 latency probes (4 KB apart: H200 partition class), then the start mark
long long* m = A.marks + (size_t)blockIdx.x * MKW;
unsigned smid;
asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
m[MKP] = smid;
m[MKP + 1] = lat_probe(A.ws13, A.flags);
m[MKP + 2] = lat_probe(A.ws13 + 4096, A.flags);
m[0] = gtime();
}
} else if (tid == 32 && (A.flags & 2)) {
const uint32_t tot = (uint32_t)A.M * (KDIM * 2), S = (mdiv(tot + gridDim.x - 1, A.magG) + 15u) & ~15u, off = blockIdx.x * S;
if (off < tot)
asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(reinterpret_cast<const uint8_t*>(A.x) + off),
"r"(min(S, tot - off)) : "memory");
}
if (A.flags & 256) asm volatile("griddepcontrol.wait;\n" ::: "memory");   // PDL: ids / x come from the previous kernel
#if VM_LD == 2
if (tid == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // VTR: vm_fin may launch now (after the wait: its pre-wait wts load is safe)
#endif
uint8_t* scr = smem + NS * US + (NX - 1) * XS;
int* hist = reinterpret_cast<int*>(scr);
uint16_t* ent = reinterpret_cast<uint16_t*>(scr + 4096);
route(A, T, hist, ent, reinterpret_cast<uint32_t*>(scr + 12288));
#if VM_WG5 >= 2
bar_arrive(9, NTH);
#endif
if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 7] = gtime();
#if VM_WG5 < 2
if (warp >= NCW) {
#if VM_WG5
asm volatile("setmaxnreg.dec.sync.aligned.u32 " VM_S(VM_PREG) ";\n" ::: "memory");
if (warp > NCW + 1) { if (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1); return; }
#endif
if (warp == NCW) {
if (A.flags & 4) bar_sync(7, 32 * NCW + 32);
if (lane == 0) prod_w(A, T, sa(smem), bars);
} else {
#if VM_WG5
if (A.flags & 64) gather_w5(A, T, ent, lane, 0);
#else
if (A.flags & 64) gather_warp(A, T, ent, lane);
#endif
bar_sync(6, 32 * NCW + 32);
if (lane == 0) prod_x(A, T, sa(smem + NS * US), bars);
else if VM_F9(A) bperm_lanes(A, T, lane);
}
return;
}
#endif
if (A.flags & 64) self_fill(A, T, smem, bars, ent, nfill_of(A, T));
else gather(A, T, ent);
bar_arrive(6, VM_B6);
if (A.flags & 4) bar_arrive(7, 32 * NCW + 32);
#if VM_WG5 == 1 || VM_WG5 == 3
asm volatile("setmaxnreg.inc.sync.aligned.u32 " VM_S(VM_CREG) ";\n" ::: "memory");
#endif
if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 5] = gtime();
#if VM_MKW == 64
if (tid == 0 && A.marks && (T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) + (T.bB[blockIdx.x + 1] - T.bB[blockIdx.x]) > 0) {
mbar_wait(bars, 0, A.err);
A.marks[(size_t)blockIdx.x * MKW + 48] = gtime();
}
#endif
const int q = warp >> 2, wp = warp & 3;
int it = 0;
#if VM_DP
int kq = 0;
phase<false>(A, T, smem, bars, it, q, wp, lane, kq);
#else
phase<false>(A, T, smem, bars, it, q, wp, lane);
#endif
if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();
#if VM_LD == 1
if (tid == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
#if VM_DP
phase<true>(A, T, smem, bars, it, q, wp, lane, kq);
#else
phase<true>(A, T, smem, bars, it, q, wp, lane);
#endif
if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();
}

__global__ void __launch_bounds__(256) vm_fin(const Args A) {
__shared__ int s_r[8], s_r8[8];
__shared__ float s_w[8], s_w8[8];
__shared__ int s_n;
__shared__ float s_ss[2][8][4];
const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#if VM_FT
asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // VTR: the next kernel lands + preloads weights now (it still waits for our completion)
#endif
#if VM_FTR
const float wv = (tid < 8) ? __ldg(A.wts + t * 8 + tid) : 0.f;   // vm_main's input: ready before vm_main ran
#endif
float4 w4[4];
#pragma unroll
for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * tid));
asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if VM_MKW == 64
if (tid == 0 && A.marks) atomicMin(reinterpret_cast<unsigned long long*>(A.marks + (size_t)A.G0 * MKW), (unsigned long long)gtime());
#endif
if (t == 0) {
for (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;
for (int i = tid; i < JMAX * 16; i += 256) A.doneA[i] = 0;
for (int i = tid; i < 2 * SLOTSD * 4; i += 256) A.cnt[i] = 0;
#if VM_DP
if (tid == 0) { A.gbar[0] = 0; A.gbar[1] = 0; }
#endif
}
#if VM_FTR
if (warp == 0) {   // VTR: ballot compaction, order kept (== the serial loop)
const int r = (lane < 8) ? __ldcg(A.rmap + t * 8 + lane) : -1;
const unsigned bm = __ballot_sync(0xffffffffu, r >= 0);
if (r >= 0) { const int pos = __popc(bm & ((1u << lane) - 1u)); s_r[pos] = r; s_w[pos] = wv; }
if (lane == 0) s_n = __popc(bm);
}
__syncthreads();
#else
if (tid < 8) { s_r8[tid] = __ldcg(A.rmap + t * 8 + tid); s_w8[tid] = __ldg(A.wts + t * 8 + tid); }
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
#endif
const int nloc = s_n;
const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
float a[4][4];
#pragma unroll
for (int m = 0; m < 4; ++m) { a[m][0] = 0.f; a[m][1] = 0.f; a[m][2] = 0.f; a[m][3] = 0.f; }
int par = 0;
for (int c0 = 0; c0 < nloc; c0 += 4, par ^= 1) {
float4 v[4][4];
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;
#pragma unroll
for (int m = 0; m < 4; ++m)
v[jj][m] = (r >= 0) ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * tid)) : z4;
}
#if VM_FTR
#pragma unroll
for (int jj = 0; jj < 4; ++jj)
#pragma unroll
for (int m = 0; m < 4; ++m) {   // VTR: products once
v[jj][m].x = bf16r(v[jj][m].x) * w4[m].x; v[jj][m].y = bf16r(v[jj][m].y) * w4[m].y;
v[jj][m].z = bf16r(v[jj][m].z) * w4[m].z; v[jj][m].w = bf16r(v[jj][m].w) * w4[m].w;
}
#endif
float ss[4];
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
ss[jj] = 0.f;
#pragma unroll
for (int m = 0; m < 4; ++m) {
float f;
#if VM_FTR
f = v[jj][m].x; ss[jj] += f * f; f = v[jj][m].y; ss[jj] += f * f;
f = v[jj][m].z; ss[jj] += f * f; f = v[jj][m].w; ss[jj] += f * f;
#else
f = bf16r(v[jj][m].x) * w4[m].x; ss[jj] += f * f; f = bf16r(v[jj][m].y) * w4[m].y; ss[jj] += f * f;
f = bf16r(v[jj][m].z) * w4[m].z; ss[jj] += f * f; f = bf16r(v[jj][m].w) * w4[m].w; ss[jj] += f * f;
#endif
}
#pragma unroll
for (int o = 16; o > 0; o >>= 1) ss[jj] += __shfl_xor_sync(0xffffffffu, ss[jj], o);
}
if (lane == 0) {
#pragma unroll
for (int jj = 0; jj < 4; ++jj) s_ss[par][warp][jj] = ss[jj];
}
__syncthreads();
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
if (c0 + jj < nloc) {
float sm = 0.f;
for (int w = 0; w < 8; ++w) sm += s_ss[par][w][jj];
const float fr = rsqrtf(sm * (1.f / 4096.f) + 1e-8f) * s_w[c0 + jj];
#pragma unroll
for (int m = 0; m < 4; ++m) {
#if VM_FTR
a[m][0] += v[jj][m].x * fr; a[m][1] += v[jj][m].y * fr;
a[m][2] += v[jj][m].z * fr; a[m][3] += v[jj][m].w * fr;
#else
a[m][0] += bf16r(v[jj][m].x) * w4[m].x * fr; a[m][1] += bf16r(v[jj][m].y) * w4[m].y * fr;
a[m][2] += bf16r(v[jj][m].z) * w4[m].z * fr; a[m][3] += bf16r(v[jj][m].w) * w4[m].w * fr;
#endif
}
}
}
#if VM_BRED
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;
if (r >= 0) {
#pragma unroll
for (int m = 0; m < 4; ++m) __stcg(reinterpret_cast<float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * tid), z4);
}
}
#endif
}
#pragma unroll
for (int m = 0; m < 4; ++m) {
__nv_bfloat162 o0 = __floats2bfloat162_rn(a[m][0], a[m][1]), o1 = __floats2bfloat162_rn(a[m][2], a[m][3]);
uint2 o;
o.x = *reinterpret_cast<uint32_t*>(&o0); o.y = *reinterpret_cast<uint32_t*>(&o1);
*reinterpret_cast<uint2*>(A.out + (size_t)t * KDIM + 1024 * m + 4 * tid) = o;
}
#if VM_MKW == 64
__syncthreads();
if (tid == 0 && A.marks) atomicMax(reinterpret_cast<unsigned long long*>(A.marks + (size_t)A.G0 * MKW + 1), (unsigned long long)gtime());
#endif
}
}

void vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V) {
vm::Args a;
const int64_t M = V[0], G = V[2], mask = V[4];
a.x = reinterpret_cast<const __nv_bfloat16*>(T[0].data_ptr()); a.ids = T[1].data_ptr<int>(); a.wts = T[2].data_ptr<float>();
a.M = (int)M; a.ls = (int)V[1];
a.ws13 = reinterpret_cast<const uint8_t*>(T[3].data_ptr()); a.ws2 = reinterpret_cast<const uint8_t*>(T[4].data_ptr());
a.gsc13 = T[5].data_ptr<float>(); a.gsc2 = T[6].data_ptr<float>(); a.nw = T[7].data_ptr<float>();
a.xA = reinterpret_cast<__nv_bfloat16*>(T[8].data_ptr()); a.xB = reinterpret_cast<__nv_bfloat16*>(T[9].data_ptr());
a.y = T[10].data_ptr<float>(); a.fix = T[11].data_ptr<float>(); a.cnt = T[12].data_ptr<int>();
a.xready = T[13].data_ptr<int>(); a.doneA = T[14].data_ptr<int>(); a.rmap = T[15].data_ptr<int>(); a.gbar = T[19].data_ptr<int>();
a.out = reinterpret_cast<__nv_bfloat16*>(T[16].data_ptr()); a.err = T[17].data_ptr<int>();
a.marks = T[18].numel() >= (G + (vm::MKW == 64 ? 1 : 0)) * vm::MKW ? reinterpret_cast<long long*>(T[18].data_ptr()) : nullptr;
#if VM_MKW == 64
a.G0 = (int)G;
#endif
TORCH_CHECK((int64_t)V.size() >= 6 + 2 * VM_NTM, "vml_run: cost table too short");
#if VM_DP
TORCH_CHECK((int64_t)V.size() >= 8 + 2 * VM_NTM, "vml_run: pool fractions missing");
a.poolA = (int)V[6 + 2 * VM_NTM]; a.poolB = (int)V[7 + 2 * VM_NTM];
#endif
#if VM_SC
TORCH_CHECK((int64_t)V.size() >= 8 + 2 * VM_NTM, "vml_run: segment costs missing");
a.segA = (int)V[6 + 2 * VM_NTM]; a.segB = (int)V[7 + 2 * VM_NTM];
#endif
#if VM_PS
TORCH_CHECK((int64_t)V.size() >= 7 + 2 * VM_NTM, "vml_run: producer delay missing");
a.pdel = (int)V[6 + 2 * VM_NTM];
#endif
a.magG = (unsigned long long)V[3]; a.flags = (int)V[5];
for (int k = 0; k < 2 * VM_NTM; ++k) { a.cost[k] = (uint32_t)V[6 + k]; a.mag[k] = ((1ull << 40) + V[6 + k] - 1) / V[6 + k]; }
auto stream = at::cuda::getCurrentCUDAStream();
if (mask & 1) {
cudaFuncSetAttribute(vm::vm_main, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
cudaLaunchConfig_t cm = {};
cm.gridDim = dim3((unsigned)G); cm.blockDim = dim3(vm::NTH); cm.dynamicSmemBytes = vm::SMEM; cm.stream = stream;
cudaLaunchAttribute am[1];
am[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
am[0].val.programmaticStreamSerializationAllowed = 1;
cm.attrs = am; cm.numAttrs = (V[5] & 256) ? 1 : 0;
C10_CUDA_CHECK(cudaLaunchKernelEx(&cm, vm::vm_main, a));
}
if ((mask & 2) && M > 0) {
cudaLaunchConfig_t cfg = {};
cfg.gridDim = dim3((unsigned)M); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0; cfg.stream = stream;
cudaLaunchAttribute at[1];
at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
at[0].val.programmaticStreamSerializationAllowed = 1;
cfg.attrs = at; cfg.numAttrs = 1;
C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));
}
}
std::vector<int64_t> vml_cfg() { return {VM_JN, VM_NTM, VM_NX, VM_XSN, VM_MKW, (int64_t)vm::SMEM, VM_DP, VM_SC, VM_BRED, VM_DEFER2, VM_PS}; }
"""

_vml_CUDA = (_vml_CUDA_T.replace("@WGSPECS@", "".join(_vml_wg_spec(n) + _vml_wg_spec64(n) for n in range(8, 129, 8)))
        .replace("@JMAX@", str(_vml_JMAX)).replace("@SLOTS@", str(_vml_SLOTS)).replace("@FIXW@", str(_vml_FIXW))
        )

_P1_MOD = None

_PT1 = {"threads": {}, "errors": {}, "secs": {}, "g2_mod": None}

def _pt1_target(name, fn):
    def run():
        t1 = time.time()
        try:
            fn()
        except Exception as ex:
            _PT1["errors"][name] = repr(ex)[:300]
        _PT1["secs"][name] = time.time() - t1
    return run

def _pt1_wait(name):
    th = _PT1["threads"].get(name)
    if th is not None and th is not threading.current_thread():
        th.join()

from functools import partial as k1fa4_partial

from types import SimpleNamespace as k1fa4_SimpleNamespace

from typing import Callable as k1fa4_Callable

from typing import Literal as k1fa4_Literal

from typing import Optional as k1fa4_Optional

import cuda.bindings.driver as k1fa4_cuda

import cutlass

import cutlass.cute as cute

import cutlass.utils.hopper_helpers as k1fa4_sm90_utils_basic

from cutlass import Float32

from cutlass import Int32

from cutlass import const_expr

from cutlass import pipeline as k1fa4_pipeline

from cutlass.base_dsl.arch import Arch as k1fa4_Arch

from cutlass.cute import FastDivmodDivisor as k1fa4_FastDivmodDivisor

from cutlass.cute.nvgpu import cpasync as k1fa4_cpasync

from cutlass.cute.nvgpu import warpgroup as k1fa4_warpgroup

from cutlass.pipeline import pipeline_init_arrive as k1fa4_pipeline_init_arrive

from cutlass.pipeline import pipeline_init_wait as k1fa4_pipeline_init_wait

from cutlass.utils import LayoutEnum as k1fa4_LayoutEnum

from quack import copy_utils as k1fa4_copy_utils

from quack import layout_utils as k1fa4_layout_utils

from quack import sm90_utils as k1fa4_sm90_utils

from quack.cute_dsl_utils import ParamsBase as k1fa4_ParamsBase

from sglang.kernels.ops.attention.flash_attn.cute import pipeline as k1fa4_pipeline_custom

from sglang.kernels.ops.attention.flash_attn.cute import utils as k1fa4_utils

from sglang.kernels.ops.attention.flash_attn.cute.block_info import BlockInfo as k1fa4_BlockInfo

from sglang.kernels.ops.attention.flash_attn.cute.block_sparse_utils import consume_block_sparse_loads as k1fa4_consume_block_sparse_loads

from sglang.kernels.ops.attention.flash_attn.cute.block_sparse_utils import produce_block_sparse_loads as k1fa4_produce_block_sparse_loads

from sglang.kernels.ops.attention.flash_attn.cute.block_sparsity import BlockSparseTensors as k1fa4_BlockSparseTensors

from sglang.kernels.ops.attention.flash_attn.cute.cute_dsl_utils import assume_tensor_aligned as k1fa4_assume_tensor_aligned

from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd import FlashAttentionForwardBase as k1fa4_FlashAttentionForwardBase

from sglang.kernels.ops.attention.flash_attn.cute.mask import AttentionMask as k1fa4_AttentionMask

from sglang.kernels.ops.attention.flash_attn.cute.named_barrier import NamedBarrierFwd as k1fa4_NamedBarrierFwd

from sglang.kernels.ops.attention.flash_attn.cute.pack_gqa import PackGQA as k1fa4_PackGQA

from sglang.kernels.ops.attention.flash_attn.cute.pack_gqa import make_packgqa_tiled_tma_atom as k1fa4_make_packgqa_tiled_tma_atom

from sglang.kernels.ops.attention.flash_attn.cute.pack_gqa import pack_gqa_layout as k1fa4_pack_gqa_layout

from sglang.kernels.ops.attention.flash_attn.cute.paged_kv import PagedKVManager as k1fa4_PagedKVManager

from sglang.kernels.ops.attention.flash_attn.cute.seqlen_info import SeqlenInfoQK as k1fa4_SeqlenInfoQK

from sglang.kernels.ops.attention.flash_attn.cute.softmax import Softmax as k1fa4_Softmax

from sglang.kernels.ops.attention.flash_attn.cute.softmax import apply_score_mod_inner as k1fa4_apply_score_mod_inner

from sglang.kernels.ops.attention.flash_attn.cute.tile_scheduler import SingleTileLPTScheduler as k1fa4_SingleTileLPTScheduler

from sglang.kernels.ops.attention.flash_attn.cute.tile_scheduler import SingleTileScheduler as k1fa4_SingleTileScheduler

from sglang.kernels.ops.attention.flash_attn.cute.tile_scheduler import SingleTileVarlenScheduler as k1fa4_SingleTileVarlenScheduler

from sglang.kernels.ops.attention.flash_attn.cute.tile_scheduler import TileSchedulerArguments as k1fa4_TileSchedulerArguments

from sglang.kernels.ops.attention.flash_attn.cute.utils import AuxData as k1fa4_AuxData

k1fa4_K = {"minb": 2, "regs1": (200, 56), "stages": 2, "pdl": True, "trig": False, "pv8": False, "ex2": 0}

def k1fa4_skip_check(*_a, **_k):
    return None

class k1fa4_FlashAttentionForwardSm90(k1fa4_FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
        has_bias: bool = False,
        bias_block_size: int = 128,
        rel_extent_padded: int = 128,
        **kwargs,
    ):
        kwargs["num_stages"] = k1fa4_K["stages"]
        super().__init__(*args, **kwargs)
        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.buffer_align_bytes = 1024
        self.use_tma_KV = not paged_kv_non_tma
        self.has_bias = has_bias
        self.rel_extent_padded = rel_extent_padded
        if has_bias:
            assert rel_extent_padded % self.tile_n == 0
        self.bias_n_max = rel_extent_padded // self.tile_n if has_bias else 0
        self.bias_block_size = bias_block_size
        assert self.use_tma_KV or not (
            self.check_hdim_oob or self.check_hdim_v_oob
        ), "Paged KV does not support irregular head dim"
        self.cluster_shape_mn = (1, 1)
        assert self.arch.is_family_of(k1fa4_Arch.sm_90a), "Only SM 9.x is supported"

    def _get_smem_layout_atom(self):
        sQ_layout_atom = k1fa4_warpgroup.make_smem_layout_atom(
            k1fa4_sm90_utils_basic.get_smem_layout_atom(
                k1fa4_LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim
            ),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = k1fa4_warpgroup.make_smem_layout_atom(
            k1fa4_sm90_utils_basic.get_smem_layout_atom(
                k1fa4_LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = k1fa4_warpgroup.make_smem_layout_atom(
                k1fa4_sm90_utils_basic.get_smem_layout_atom(
                    k1fa4_LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        tiled_mma_qk = k1fa4_sm90_utils_basic.make_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            k1fa4_warpgroup.OperandMajorMode.K,
            k1fa4_warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        tiled_mma_pv = k1fa4_sm90_utils_basic.make_trivial_tiled_mma(
            k1fa4_pv_dtype(self.dtype),
            k1fa4_pv_dtype(self.dtype),
            k1fa4_warpgroup.OperandMajorMode.K,
            k1fa4_pv_major_b(k1fa4_warpgroup.OperandMajorMode.MN),
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                1,
                1,
            ),
            tiler_mn=(64, self.tile_hdimv),
            a_source=(
                k1fa4_warpgroup.OperandSource.RMEM
                if self.mma_pv_is_rs
                else k1fa4_warpgroup.OperandSource.SMEM
            ),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)],
                self.buffer_align_bytes,
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]
        cosize_sP = (
            cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        )
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_Bias_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        sBias_struct = cute.struct.Align[
            cute.struct.MemRange[
                self.bias_dtype if const_expr(self.has_bias) else cutlass.BFloat16,
                cute.cosize(self.sBias_layout) if const_expr(self.has_bias) else 0,
            ],
            1024,
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageQKVBias:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_Bias: mbar_ptr_Bias_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sBias: sBias_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQVBias:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_Bias: mbar_ptr_Bias_struct
            sQ: sQV_struct
            sK: sK_struct
            sBias: sBias_struct
            sP: sP_struct

        if const_expr(self.has_bias):
            return (
                SharedStorageQKVBias
                if const_expr(not self.Q_in_regs)
                else SharedStorageSharedQVBias
            )
        return (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: k1fa4_Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: k1fa4_Optional[cute.Tensor] = None,
        mCuSeqlensK: k1fa4_Optional[cute.Tensor] = None,
        mSeqUsedQ: k1fa4_Optional[cute.Tensor] = None,
        mSeqUsedK: k1fa4_Optional[cute.Tensor] = None,
        mPageTable: k1fa4_Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: k1fa4_Optional[cute.Tensor] = None,
        blocksparse_tensors: k1fa4_Optional[k1fa4_BlockSparseTensors] = None,
        aux_data: k1fa4_AuxData = k1fa4_AuxData(),
        mBias: k1fa4_Optional[
            cute.Tensor
        ] = None,
        stream: k1fa4_cuda.CUstream = None,
    ):

        k1fa4_skip_check(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        self.bias_dtype = (
            mBias.element_type if const_expr(mBias is not None) else self.dtype
        )
        mQ, mK, mV, mO = [k1fa4_assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        if const_expr(mBias is not None):
            mBias = k1fa4_assume_tensor_aligned(mBias)
        Q_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ = k1fa4_layout_utils.select(mQ, Q_layout_transpose)
        if const_expr(mBias is not None):
            mBias = k1fa4_layout_utils.select(mBias, Q_layout_transpose)
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [k1fa4_layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        if const_expr(self.is_split_kv):
            num_splits = mO.shape[0]
            O_layout_transpose = (
                [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            )
            LSE_layout_transpose = (
                [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            )
        else:
            num_splits = Int32(1)
            O_layout_transpose = (
                [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            )
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
        mO = k1fa4_layout_utils.select(mO, O_layout_transpose)
        mLSE = (
            k1fa4_layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {
            1: k1fa4_K["regs1"],
            2: (240, 24),
            3: (160, 32),
        }[self.num_wg_mma]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and self.has_bias):
            raise NotImplementedError(
                "Block sparsity + sheared bias is not supported on SM90"
            )

        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= k1fa4_Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = self.use_tma_Q and not self.is_split_kv
        if const_expr(
            self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)
        ):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap
        self._setup_attributes()
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            k1fa4_sm90_utils.make_smem_layout(elem_type, k1fa4_LayoutEnum.ROW_MAJOR, shape, stage)
            for elem_type, shape, stage in [
                (mQ.element_type, (self.tile_m, self.tile_hdim), None),
                (mK.element_type, (self.tile_n, self.tile_hdim), self.num_stages),
                k1fa4_sv_spec(self),
                (self.dtype, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = k1fa4_sm90_utils.make_smem_layout(
                mV.element_type, k1fa4_LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )
        if const_expr(mBias is not None):
            self.sBias_layout = k1fa4_sm90_utils.make_smem_layout(
                self.bias_dtype,
                k1fa4_LayoutEnum.ROW_MAJOR,
                (self.bias_block_size, self.tile_n),
                self.num_stages,
            )
        else:
            self.sBias_layout = None

        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = k1fa4_pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = k1fa4_pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mBias is not None):
                mBias = k1fa4_pack_gqa_layout(
                    mBias, self.qhead_per_kvhead, nheads_kv, head_idx=2
                )
            if const_expr(mLSE is not None):
                mLSE = k1fa4_pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        gmem_tiled_copy_Q = k1fa4_cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = k1fa4_cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = k1fa4_cpasync.CopyBulkTensorTileS2GOp()
        gmem_tiled_copy_Bias = k1fa4_cpasync.CopyBulkTensorTileG2SOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        if const_expr(mBias is not None):
            self.tma_copy_bytes["Bias"] = cute.size_in_bytes(
                self.bias_dtype, cute.select(self.sBias_layout, mode=[0, 1])
            )
        make_tiled_tma_atom_fn = (
            k1fa4_partial(
                k1fa4_make_packgqa_tiled_tma_atom,
                qhead_per_kvhead=self.qhead_per_kvhead,
                head_idx=2,
            )
            if const_expr(self.pack_gqa)
            else k1fa4_cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = k1fa4_cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V, tma_tensor_V = k1fa4_cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,
            )
        tma_atom_Bias, tma_tensor_Bias = None, None
        if const_expr(mBias is not None):
            tma_atom_Bias, tma_tensor_Bias = k1fa4_cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_Bias,
                mBias,
                cute.select(self.sBias_layout, mode=[0, 1]),
                (self.bias_block_size, self.tile_n),
                1,
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = k1fa4_copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = k1fa4_SingleTileVarlenScheduler
        else:
            TileScheduler = (
                k1fa4_SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else k1fa4_SingleTileLPTScheduler
            )
        tile_sched_args = k1fa4_TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            (
                cute.size(mQ.shape[3])
                if const_expr(mCuSeqlensQ is None)
                else cute.size(mCuSeqlensQ.shape[0] - 1)
            ),
            num_splits,
            (
                cute.size(mK.shape[0])
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mQ.shape[1],
            mV.shape[1],
            total_q=(
                cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        if const_expr(self.has_bias):
            base_softmax_scale = softmax_scale
            softmax_scale_log2, softmax_scale = k1fa4_utils.LOG2_E, None
        else:
            base_softmax_scale = None
            softmax_scale_log2, softmax_scale = k1fa4_utils.compute_softmax_scale_log2(
                softmax_scale, self.score_mod
            )
        window_size_left = (
            Int32(window_size_left) if window_size_left is not None else None
        )
        window_size_right = (
            Int32(window_size_right) if window_size_right is not None else None
        )
        fastdiv_mods = k1fa4_utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors, mPageTable
        )

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            tma_tensor_Bias,
            tma_atom_Bias,
            softmax_scale_log2,
            softmax_scale,
            base_softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.sBias_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            num_splits,
            aux_data,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=k1fa4_K["minb"], use_pdl=k1fa4_K["pdl"],
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: k1fa4_Optional[cute.Tensor],
        mCuSeqlensQ: k1fa4_Optional[cute.Tensor],
        mCuSeqlensK: k1fa4_Optional[cute.Tensor],
        mSeqUsedQ: k1fa4_Optional[cute.Tensor],
        mSeqUsedK: k1fa4_Optional[cute.Tensor],
        mPageTable: k1fa4_Optional[cute.Tensor],
        tma_atom_Q: k1fa4_Optional[cute.CopyAtom],
        tma_atom_K: k1fa4_Optional[cute.CopyAtom],
        tma_atom_V: k1fa4_Optional[cute.CopyAtom],
        tma_atom_O: k1fa4_Optional[cute.CopyAtom],
        mBias: k1fa4_Optional[cute.Tensor],
        tma_atom_Bias: k1fa4_Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: k1fa4_Optional[Float32],
        base_softmax_scale: k1fa4_Optional[Float32],
        window_size_left: k1fa4_Optional[Int32],
        window_size_right: k1fa4_Optional[Int32],
        learnable_sink: k1fa4_Optional[cute.Tensor],
        blocksparse_tensors: k1fa4_Optional[k1fa4_BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        sBias_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: k1fa4_ParamsBase,
        TileScheduler: cutlass.Constexpr[k1fa4_Callable],
        SharedStorage: cutlass.Constexpr[k1fa4_Callable],
        num_splits: Int32 = 1,
        aux_data: k1fa4_AuxData = k1fa4_AuxData(),
        fastdiv_mods=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_O,
                tma_atom_Bias,
            ):
                if const_expr(tma_atom is not None):
                    k1fa4_cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        self._k1_storage = storage

        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = k1fa4_partial(
            k1fa4_pipeline.CooperativeGroup, k1fa4_pipeline.Agent.Thread
        )
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = k1fa4_pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = k1fa4_pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = k1fa4_pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = k1fa4_pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = k1fa4_pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = k1fa4_pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
        if const_expr(self.has_bias):
            pipeline_bias = k1fa4_pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_Bias.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Bias"],
                defer_sync=True,
            )
        else:
            pipeline_bias = None

        k1fa4_pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        sVt = k1fa4_vt_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        if const_expr(self.has_bias):
            sBias = storage.sBias.get_tensor(
                sBias_layout.outer, swizzle=sBias_layout.inner
            )
        else:
            sBias = None
        sO = storage.sQ.get_tensor(
            sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype
        )

        block_info = k1fa4_BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        SeqlenInfoCls = k1fa4_partial(
            k1fa4_SeqlenInfoQK.create,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=(
                mK.shape[0]
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks
                if blocksparse_tensors is not None
                else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
        )
        AttentionMaskCls = k1fa4_partial(
            k1fa4_AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        TileSchedulerCls = k1fa4_partial(TileScheduler.create, tile_sched_params)

        k1fa4_pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                mBias,
                sBias,
                tma_atom_Bias,
                pipeline_k,
                pipeline_v,
                pipeline_bias,
                pipeline_q,
                gmem_tiled_copy_Q,
                mPageTable,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                num_splits,
            )

        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                base_softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                num_splits,
                aux_data,
                fastdiv_mods,
                sBias,
                pipeline_bias,
            )




    @cute.jit
    def apply_bias(
        self,
        acc_S: cute.Tensor,
        thr_mma_qk: cute.ThrMma,
        sBias: cute.Tensor,
        pipeline_bias: k1fa4_pipeline.PipelineAsync,
        bias_softmax_scale: Float32,
        consumer_state: k1fa4_pipeline.PipelineState,
        apply_bias: bool,
    ):
        if apply_bias:
            pipeline_bias.consumer_wait(
                consumer_state, pipeline_bias.consumer_try_wait(consumer_state)
            )
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS = thr_mma_qk.partition_C(cS)
            for i in cutlass.range_constexpr(cute.size(acc_S.shape)):
                row = tScS[i][0]
                col = tScS[i][1]
                if (
                    const_expr(self.bias_block_size == self.tile_m)
                    or row < self.bias_block_size
                ):
                    acc_S[i] = acc_S[i] * bias_softmax_scale
                    acc_S[i] = acc_S[i] + sBias[row, col, consumer_state.index].to(
                        self.qk_acc_dtype
                    )
                else:
                    acc_S[i] = acc_S[i] * bias_softmax_scale
            pipeline_bias.consumer_release(consumer_state)
        elif const_expr(self.has_bias):
            for i in cutlass.range_constexpr(cute.size(acc_S.shape)):
                acc_S[i] = acc_S[i] * bias_softmax_scale

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: k1fa4_Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: k1fa4_Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: k1fa4_Optional[cute.Tensor],
        pipeline_k: k1fa4_pipeline.PipelineAsync,
        pipeline_v: k1fa4_pipeline.PipelineAsync,
        pipeline_q: k1fa4_pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: k1fa4_Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: k1fa4_Optional[Float32],
        base_softmax_scale: k1fa4_Optional[Float32],
        block_info: k1fa4_BlockInfo,
        SeqlenInfoCls: k1fa4_Callable,
        AttentionMaskCls: k1fa4_Callable,
        TileSchedulerCls: k1fa4_Callable,
        blocksparse_tensors: k1fa4_Optional[k1fa4_BlockSparseTensors],
        num_splits: Int32 = 1,
        aux_data: k1fa4_AuxData = k1fa4_AuxData(),
        fastdiv_mods=None,
        sBias: k1fa4_Optional[cute.Tensor] = None,
        pipeline_bias: k1fa4_Optional[k1fa4_pipeline.PipelineAsync] = None,
    ):
        aux_tensors = aux_data.tensors
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = k1fa4_sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = k1fa4_partial(
            k1fa4_sm90_utils.gemm_zero_init,
            tiled_mma_qk,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
        )
        acc_O, tOrP, tOrVt = k1fa4_sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = k1fa4_partial(k1fa4_sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        smem_copy_atom_P = k1fa4_utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_P = cute.make_tiled_copy_C(
            smem_copy_atom_P, tiled_mma_qk
        ).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = k1fa4_SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        self.mma_init()

        q_consumer_phase = Int32(0)
        kv_consumer_state = k1fa4_pipeline.make_pipeline_state(
            k1fa4_pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = k1fa4_softmax_cls(k1fa4_Softmax).create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        mma_one_n_block_all = k1fa4_partial(
            (
                self.mma_one_n_block_intrawg_overlap
                if const_expr(self.intra_wg_overlap)
                else self.mma_one_n_block
            ),
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
            thr_mma_qk=thr_mma_qk,
            sBias=sBias,
            pipeline_bias=pipeline_bias,
            bias_softmax_scale=base_softmax_scale,
        )

        process_first_half_block = k1fa4_partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
            thr_mma_qk=thr_mma_qk,
            sBias=sBias,
            pipeline_bias=pipeline_bias,
            bias_softmax_scale=base_softmax_scale,
        )
        process_last_half_block = k1fa4_partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        while work_tile.is_valid_tile:

            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    (
                        seqlen_q_divmod
                        if not recompute_fastdiv_mods_q
                        else k1fa4_FastDivmodDivisor(seqlen.seqlen_q)
                    ),
                    (
                        seqlen_k_divmod
                        if not recompute_fastdiv_mods_k
                        else k1fa4_FastDivmodDivisor(seqlen.seqlen_k)
                    ),
                )

            mask = AttentionMaskCls(seqlen)
            mask_fn = k1fa4_partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = k1fa4_partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=1.0 if const_expr(self.has_bias) else softmax_scale,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                )
            mma_one_n_block = k1fa4_partial(
                mma_one_n_block_all,
                seqlen=seqlen,
                softmax=softmax,
                score_mod_fn=score_mod_fn,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            num_bias_loads = Int32(0)
            if const_expr(self.has_bias):
                _, n_block_max_abs = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits, absolute=True
                )
                bias_idx_offset = n_block_max_abs - n_block_max
                num_bias_loads = min(
                    self.bias_n_max - bias_idx_offset, n_block_max - n_block_min
                )
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
            O_should_accumulate = False
            has_work = True
            if const_expr(self.is_split_kv):
                has_work = n_block_min < n_block_max

            if const_expr(not self.use_block_sparsity):
                if has_work:
                    if const_expr(self.intra_wg_overlap):
                        kv_consumer_state = process_first_half_block(
                            n_block=n_block_max - 1,
                            seqlen=seqlen,
                            kv_consumer_state=kv_consumer_state,
                            mask_fn=k1fa4_partial(mask_fn, mask_mod=self.mask_mod),
                            score_mod_fn=score_mod_fn,
                            is_first_block=True,
                            apply_bias=const_expr(self.has_bias) and num_bias_loads > 0,
                        )
                    else:
                        self.warp_scheduler_barrier_sync()
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1,
                            seqlen=seqlen,
                            mma_pv_fn=k1fa4_partial(mma_pv_fn, zero_init=True),
                            is_first_n_block=True,
                            mask_fn=k1fa4_partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=True
                            ),
                            apply_bias=const_expr(self.has_bias) and num_bias_loads > 0,
                        )
                        O_should_accumulate = True
                    n_block_max -= 1
                    if const_expr(self.is_causal or self.is_local or self.has_bias):
                        if const_expr(self.has_bias):
                            n_block_min_causal_local_mask = max(
                                n_block_max + 1 - num_bias_loads, n_block_min
                            )
                        else:
                            n_block_min_causal_local_mask = (
                                block_info.get_n_block_min_causal_local_mask(
                                    seqlen, m_block, n_block_min
                                )
                            )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                seqlen=seqlen,
                                mma_pv_fn=k1fa4_partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                mask_fn=(
                                    None
                                    if const_expr(self.has_bias)
                                    else k1fa4_partial(
                                        mask_fn,
                                        mask_mod=self.mask_mod,
                                        mask_seqlen=False,
                                    )
                                ),
                                apply_bias=self.has_bias,
                            )
                            O_should_accumulate = True
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen,
                            mma_pv_fn=k1fa4_partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            mask_fn=k1fa4_partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                            ),
                        )
                        O_should_accumulate = True
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min, unroll=1
                        ):
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                seqlen=seqlen,
                                mma_pv_fn=k1fa4_partial(
                                    mma_pv_fn, zero_init=not O_should_accumulate
                                ),
                                mask_fn=k1fa4_partial(
                                    mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                                ),
                            )
                            O_should_accumulate = True
                pipeline_q.consumer_release_w_index(0)
                if has_work:
                    if const_expr(self.intra_wg_overlap):
                        kv_consumer_state = process_last_half_block(
                            kv_consumer_state=kv_consumer_state,
                            zero_init=not O_should_accumulate,
                        )
                        O_should_accumulate = True
                    else:
                        self.warp_scheduler_barrier_arrive()
                else:
                    softmax.reset()
                    acc_O.fill(0.0)

            else:
                kv_consumer_state, O_should_accumulate, processed_any = (
                    k1fa4_consume_block_sparse_loads(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        m_block,
                        seqlen,
                        kv_consumer_state,
                        mma_pv_fn,
                        mma_one_n_block,
                        process_first_half_block,
                        process_last_half_block,
                        mask_fn,
                        score_mod_fn,
                        O_should_accumulate,
                        self.mask_mod,
                        fastdiv_mods,
                        self.intra_wg_overlap,
                        self.warp_scheduler_barrier_sync,
                        self.warp_scheduler_barrier_arrive,
                        self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                        (
                            self.q_subtile_factor
                            if self.q_subtile_factor is not None
                            else 1
                        ),
                        split_idx,
                        num_splits,
                    )
                )

                pipeline_q.consumer_release_w_index(0)

                if not processed_any:
                    softmax.reset()
                    acc_O.fill(0.0)

            q_consumer_phase ^= 1

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:
                    sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = k1fa4_layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        q_head_idx = (
                            row % self.qhead_per_kvhead
                            + head_idx * self.qhead_per_kvhead
                        )
                        sink_val[r] = Float32(learnable_sink[q_head_idx])
            if const_expr(self.is_split_kv and learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = sink_val if split_idx == 0 else -Float32.inf
                else:
                    if split_idx != 0:
                        sink_val.fill(-Float32.inf)

            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)

            if const_expr(k1fa4_K['trig'] == 'epi'):
                cute.arch.griddepcontrol_launch_dependents()
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: k1fa4_Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: k1fa4_SimpleNamespace,
        softmax: k1fa4_Softmax,
        seqlen: k1fa4_SeqlenInfoQK,
        scores_scale: k1fa4_Optional[cute.Tensor] = None,
        acc_O: k1fa4_Optional[cute.Tensor] = None,
        mask_fn: k1fa4_Callable = None,
        score_mod_fn: k1fa4_Optional[k1fa4_Callable] = None,
        is_first_block: bool = False,
        thr_mma_qk: k1fa4_Optional[cute.ThrMma] = None,
        sBias: k1fa4_Optional[cute.Tensor] = None,
        pipeline_bias: k1fa4_Optional[k1fa4_pipeline.PipelineAsync] = None,
        bias_softmax_scale: k1fa4_Optional[Float32] = None,
        apply_bias: bool = False,
    ):

        pipeline_k.consumer_wait(
            kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
        )
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        if const_expr(self.has_bias):
            self.apply_bias(
                acc_S,
                thr_mma_qk,
                sBias,
                pipeline_bias,
                bias_softmax_scale,
                kv_consumer_state,
                apply_bias,
            )

        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

        tOrP_acc = k1fa4_layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: k1fa4_Callable,
        zero_init: bool,
        scores_scale: k1fa4_Optional[cute.Tensor] = None,
        softmax: k1fa4_Optional[k1fa4_Softmax] = None,
        acc_O: k1fa4_Optional[cute.Tensor] = None,
    ):

        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        pipeline_v.consumer_wait(
            kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state)
        )
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: k1fa4_pipeline.PipelineState | k1fa4_pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: k1fa4_Callable,
        mma_pv_fn: k1fa4_Callable,
        pipeline_k: k1fa4_pipeline.PipelineAsync,
        pipeline_v: k1fa4_pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: k1fa4_SimpleNamespace,
        softmax: k1fa4_Softmax,
        seqlen: k1fa4_SeqlenInfoQK,
        scores_scale: k1fa4_Optional[cute.Tensor] = None,
        score_mod_fn: k1fa4_Optional[k1fa4_Callable] = None,
        mask_fn: k1fa4_Optional[k1fa4_Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        thr_mma_qk: k1fa4_Optional[cute.ThrMma] = None,
        sBias: k1fa4_Optional[cute.Tensor] = None,
        pipeline_bias: k1fa4_Optional[k1fa4_pipeline.PipelineAsync] = None,
        bias_softmax_scale: k1fa4_Optional[Float32] = None,
        apply_bias: bool = False,
    ):
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        k1fa4_warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(self.has_bias):
            self.apply_bias(
                acc_S,
                thr_mma_qk,
                sBias,
                pipeline_bias,
                bias_softmax_scale,
                smem_pipe_read,
                apply_bias,
            )

        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        tOrP_acc = k1fa4_layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        k1fa4_cvt_p(acc_S, tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: k1fa4_pipeline.PipelineState | k1fa4_pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: k1fa4_Callable,
        mma_pv_fn: k1fa4_Callable,
        pipeline_k: k1fa4_pipeline.PipelineAsync,
        pipeline_v: k1fa4_pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: k1fa4_SimpleNamespace,
        softmax: k1fa4_Softmax,
        seqlen: k1fa4_SeqlenInfoQK,
        scores_scale: k1fa4_Optional[cute.Tensor] = None,
        score_mod_fn: k1fa4_Optional[k1fa4_Callable] = None,
        mask_fn: k1fa4_Optional[k1fa4_Callable] = None,
        check_inf: cutlass.Constexpr = True,
        thr_mma_qk: k1fa4_Optional[cute.ThrMma] = None,
        sBias: k1fa4_Optional[cute.Tensor] = None,
        pipeline_bias: k1fa4_Optional[k1fa4_pipeline.PipelineAsync] = None,
        bias_softmax_scale: k1fa4_Optional[Float32] = None,
        apply_bias: bool = False,
    ):
        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)
        pipeline_v.consumer_wait(
            smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v)
        )
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        k1fa4_warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        if const_expr(self.has_bias):
            self.apply_bias(
                acc_S,
                thr_mma_qk,
                sBias,
                pipeline_bias,
                bias_softmax_scale,
                smem_pipe_read,
                apply_bias,
            )

        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        k1fa4_warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)
        tOrP_acc = k1fa4_layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        k1fa4_cvt_p(acc_S, tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = k1fa4_utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(k1fa4_NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_data: k1fa4_AuxData = k1fa4_AuxData(),
        fastdiv_mods=None,
    ):
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        k1fa4_apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(k1fa4_NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + k1fa4_utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = k1fa4_utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(k1fa4_NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

import torch

import triton

import triton.language as tl

from sglang.kernels.ops.attention.flash_attn.cute.cute_dsl_utils import to_cute_tensor as k1fa4_to_cute_tensor

K1FA4_RAW_STAGES = 2

K1FA4_TILE = (64, 128)

K1FA4_MAX_SPLITS = 128

K1FA4_STAGES = {"full": 3, "sliding": 2}

K1FA4_PV8 = {("full", 1): True, ("full", 8): True, ("sliding", 1): True, ("sliding", 8): True}

K1FA4_REGS1 = {False: (200, 56), True: (184, 72)}

K1FA4_P8_OFFSET = 8.0

K1FA4_EX2 = {"mode": 0, "freq": 8, "res": 1, "deg": 3}

K1FA4_L2PF = {"k": False, "v": False, "vh": False}

K1FA4_PREWAIT = False

K1FA4_COMB_HB = -1

K1FA4_COMB_ELEMS = 4096

K1FA4_COMB_CAP = {False: 4, True: 8}

K1FA4_COMB_WARPS = 4

K1FA4_PRO_BM = 0

K1FA4_PRO_WARPS = 4

K1FA4_TRIG = {"full": "epi", "sliding": "epi"}

K1FA4_SPLITS = {("sliding", 1): 32}

K1FA4_RMS_EPS = 1e-5

K1FA4_ROPE_POSITIONS = 1 << 21

K1FA4_STATE = {"ready": None}

ATT1_PRO_TF = True

ATT1_PW = {('full', 1): True}

ATT1_TRIG = {}

ATT1_COMB_NW = 0

_k1fa4_compiled = {}

_k1fa4_rope = {}

class k1fa4_Sm90Fwd(k1fa4_FlashAttentionForwardSm90):
    k1fa4_pv8 = False

    def __init__(self, dtype, *args, **kwargs):
        kind = "sliding" if kwargs.get("is_local") else "full"
        k1fa4_K["stages"] = K1FA4_STAGES[kind]
        k1fa4_K["trig"] = k1fa4_K.get("trig_o") or K1FA4_TRIG[kind]
        k1fa4_K["pv8"] = self.k1fa4_pv8
        k1fa4_K["regs1"] = K1FA4_REGS1[self.k1fa4_pv8]
        k1fa4_K["ex2"] = K1FA4_EX2["mode"]
        super().__init__(cutlass.Float16, *args, **kwargs)

    def _get_shared_storage_cls(self):
        q_n = max(cute.cosize(self.sQ_layout), 2 * cute.cosize(self.sO_layout))
        sQ_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, q_n], 1024]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(self.sK_layout)], 1024]
        sV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sV_layout)], 1024]
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, 0], 1024]
        sVr_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E4M3FN, K1FA4_RAW_STAGES * self.tile_n * self.tile_hdimv], 1024]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageK1fa4:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            sVr: sVr_struct

        return SharedStorageK1fa4

    @cute.jit
    def _k1fa4_upcast_own(self, pkm, tiled_r2s, tidx, sXr, sX):
        tRs = pkm.gmem_thr_copy_KV.partition_D(cute.group_modes(sXr, 0, 1))
        tXs = tiled_r2s.get_slice(tidx).partition_D(cute.group_modes(sX, 0, 1))
        for m in cutlass.range_constexpr(cute.size(tRs.shape[1])):
            r8 = cute.make_fragment_like(tRs[None, m, None])
            cute.autovec_copy(tRs[None, m, None], r8)
            r16 = cute.make_fragment_like(tXs[None, m, None])
            r16v = cute.make_tensor(r16.iterator, r8.layout)
            r16v.store(r8.load().to(self.dtype))
            cute.copy(tiled_r2s, r16, tXs[None, m, None])

    @cute.jit
    def load(self, mQ, mK, mV, sQ, sK, sV, tma_atom_Q, tma_atom_K, tma_atom_V, mBias, sBias, tma_atom_Bias,
             pipeline_k, pipeline_v, pipeline_bias, pipeline_q, gmem_tiled_copy_Q, mPageTable,
             blocksparse_tensors, block_info, SeqlenInfoCls, TileSchedulerCls, num_splits: Int32 = 1):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        nthr = self.num_threads_per_warp_group
        if const_expr(k1fa4_K["pdl"]):
            cute.arch.griddepcontrol_wait()
            if const_expr(k1fa4_K["trig"] == "start"):
                cute.arch.griddepcontrol_launch_dependents()
        storage = self._k1_storage
        sVr = storage.sVr.get_tensor(cute.make_layout(
            (self.tile_n, self.tile_hdimv, K1FA4_RAW_STAGES),
            stride=(self.tile_hdimv, 1, self.tile_n * self.tile_hdimv)))
        tiled_r2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128),
            cute.make_ordered_layout((nthr // 4, 4), order=(1, 0)), cute.make_layout((1, 16)))

        q_producer_phase = Int32(1)
        k_state = k1fa4_pipeline.make_pipeline_state(k1fa4_pipeline.PipelineUserType.Producer, self.num_stages)
        v_state = k1fa4_pipeline.make_pipeline_state(k1fa4_pipeline.PipelineUserType.Producer, self.num_stages)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            load_Q, _, _ = k1fa4_copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ,
                                                            single_stage=True)
            pkm = k1fa4_PagedKVManager.create(mPageTable, mK, mV, k1fa4_FastDivmodDivisor(mK.shape[0]), batch_idx,
                                              head_idx, tidx, seqlen.seqlen_k, 0, self.tile_n, self.tile_hdim,
                                              self.tile_hdimv, nthr, mK.element_type, arch=90)
            if warp_idx_in_wg == 0:
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            q_producer_phase ^= 1
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            nblk = n_block_max - n_block_min
            if nblk > 0:
                for it in cutlass.range(nblk, unroll=1):
                    n_block = n_block_max - 1 - it
                    pipeline_k.producer_acquire(k_state)
                    pkm.load_page_table(n_block)
                    pkm.load_KV(n_block, sK[None, None, k_state.index], "K")
                    pipeline_k.producer_commit(k_state)
                    k_state.advance()
                    pkm.load_KV(n_block, sVr[None, None, it % K1FA4_RAW_STAGES], "V")
                    cute.arch.cp_async_commit_group()
                    if it > 0:
                        cute.arch.cp_async_wait_group(1)
                        pipeline_v.producer_acquire(v_state)
                        self._k1fa4_upcast_own(pkm, tiled_r2s, tidx, sVr[None, None, (it - 1) % K1FA4_RAW_STAGES],
                                               sV[None, None, v_state.index])
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))
                        v_state.advance()
                cute.arch.cp_async_wait_group(0)
                pipeline_v.producer_acquire(v_state)
                self._k1fa4_upcast_own(pkm, tiled_r2s, tidx, sVr[None, None, (nblk - 1) % K1FA4_RAW_STAGES],
                                       sV[None, None, v_state.index])
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))
                v_state.advance()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        pipeline_v.producer_tail(v_state)

from cutlass._mlir.dialects import llvm as k1fa4_llvm

from cutlass.cutlass_dsl import T as k1fa4_T

from cutlass.cutlass_dsl import dsl_user_op as k1fa4_dsl_user_op

from dataclasses import dataclass as k1fa4_dataclass

def k1fa4_sv_spec(fa):
    if k1fa4_K["pv8"]:
        return (cutlass.Float8E4M3FN, (fa.tile_hdimv, fa.tile_n), fa.num_stages)
    return (fa.dtype, (fa.tile_n, fa.tile_hdimv), fa.num_stages)

def k1fa4_vt_view(sV):
    return sV if k1fa4_K["pv8"] else k1fa4_layout_utils.transpose_view(sV)

def k1fa4_pv_dtype(dtype):
    return cutlass.Float8E4M3FN if k1fa4_K["pv8"] else dtype

def k1fa4_pv_major_b(mode):
    return k1fa4_warpgroup.OperandMajorMode.K if k1fa4_K["pv8"] else mode

@k1fa4_dsl_user_op
def k1fa4_cvt_e4m3x4(a, b, c, d, *, loc=None, ip=None):
    return cutlass.Uint32(k1fa4_llvm.inline_asm(
        k1fa4_T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
         Float32(c).ir_value(loc=loc, ip=ip), Float32(d).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\ncvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
        "mov.b32 $0, {lo, hi};\n}",
        "=r,f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=k1fa4_llvm.AsmDialect.AD_ATT))

@cute.jit
def k1fa4_cvt_p8(acc_S, tOrP):
    nkb = cute.size(acc_S) // 16
    words = cute.make_tensor(cute.recast_ptr(tOrP.iterator, dtype=cutlass.Uint32), cute.make_layout((nkb * 4,)))
    for kb in cutlass.range_constexpr(nkb):
        for reg in cutlass.range_constexpr(4):
            rs, kh = reg % 2, reg // 2
            base = 2 * rs + 4 * (4 * kb + 2 * kh)
            words[4 * kb + reg] = k1fa4_cvt_e4m3x4(acc_S[base], acc_S[base + 1], acc_S[base + 4], acc_S[base + 5])

def k1fa4_cvt_p(acc_S, tOrP_acc, tOrP):
    if k1fa4_K["pv8"]:
        k1fa4_cvt_p8(acc_S, tOrP)
    else:
        k1fa4_utils.cvt_f16(tOrP_acc, tOrP)

@k1fa4_dataclass
class k1fa4_SoftmaxX(k1fa4_Softmax):

    @staticmethod
    def create(scale_log2, num_rows, arch=80, softmax_scale=None):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return k1fa4_SoftmaxX(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    @cute.jit
    def online_softmax(self, acc_S: cute.Tensor, is_first: cutlass.Constexpr[bool] = False,
                       check_inf: cutlass.Constexpr[bool] = True) -> cute.Tensor:
        mode = K1FA4_EX2["mode"]
        freq = K1FA4_EX2["freq"]
        res = K1FA4_EX2["res"]
        deg = K1FA4_EX2["deg"]
        off = K1FA4_P8_OFFSET if k1fa4_K["pv8"] else 0.0
        acc_S_mn = k1fa4_layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_fragment_like(self.row_max, Float32)
        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = k1fa4_utils.fmax_reduce(
                acc_S_row, init_val=row_max[r] if cutlass.const_expr(not is_first) else None, arch=arch)
            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur
            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur
            row_max_cur_scaled = row_max_cur * scale_log2
            if cutlass.const_expr(mode == 0):
                acc_S_row_exp = cute.math.exp2(acc_S_row * scale_log2 - (row_max_cur_scaled - off), fastmath=True)
            else:
                xs = cute.make_rmem_tensor(acc_S_row.shape, Float32)
                xs.store(acc_S_row * scale_log2 - (row_max_cur_scaled - off))
                if cutlass.const_expr(mode == 2):
                    for i in cutlass.range_constexpr(cute.size(xs)):
                        if cutlass.const_expr(i % freq < res):
                            xs[i] = k1fa4_utils.ex2_emulation(xs[i], poly_degree=deg)
                        else:
                            xs[i] = cute.math.exp2(xs[i], fastmath=True)
                acc_S_row_exp = xs.load()
            if cutlass.const_expr(is_first):
                acc_S_row_sum = k1fa4_utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
                row_scale[r] = 1.0
            else:
                row_scale[r] = cute.math.exp2((row_max_prev - row_max_cur) * scale_log2, fastmath=True)
                acc_S_row_sum = k1fa4_utils.fadd_reduce(acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch)
            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None].store(acc_S_row_exp)
        return row_scale

@k1fa4_dsl_user_op
def k1fa4_cp_async16_l2_128(dst_smem, src_gmem, *, loc=None, ip=None):
    k1fa4_llvm.inline_asm(
        None, [Int32(dst_smem).ir_value(loc=loc, ip=ip), cutlass.Int64(src_gmem).ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16;", "r,l", has_side_effects=True, is_align_stack=False,
        asm_dialect=k1fa4_llvm.AsmDialect.AD_ATT)

@k1fa4_dsl_user_op
def k1fa4_prefetch_l2_128(src_gmem, *, loc=None, ip=None):
    k1fa4_llvm.inline_asm(
        None, [cutlass.Int64(src_gmem).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], 128;", "l", has_side_effects=True, is_align_stack=False,
        asm_dialect=k1fa4_llvm.AsmDialect.AD_ATT)

def k1fa4_softmax_cls(default):
    return k1fa4_SoftmaxX if (k1fa4_K["ex2"] or (k1fa4_K["pv8"] and K1FA4_P8_OFFSET)) else default

class k1fa4_Sm90FwdPV8(k1fa4_Sm90Fwd):
    k1fa4_pv8 = True

    def _get_shared_storage_cls(self):
        q_n = max(cute.cosize(self.sQ_layout), 2 * cute.cosize(self.sO_layout))
        f8 = cutlass.Float8E4M3FN
        sQ_struct = cute.struct.Align[cute.struct.MemRange[f8, q_n], 1024]
        sK_struct = cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(self.sK_layout)], 1024]
        sV_struct = cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(self.sV_layout)], 1024]
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, 0], 1024]
        sVr_struct = cute.struct.Align[
            cute.struct.MemRange[f8, K1FA4_RAW_STAGES * self.tile_n * self.tile_hdimv], 1024]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageK1fa4PV8:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            sVr: sVr_struct

        return SharedStorageK1fa4PV8

    @cute.jit
    def _k1fa4_v_pages(self, mPageTable, batch_idx, seqlen_k, n_block, rbase, pg):
        for i in cutlass.range_constexpr(4):
            key = n_block * self.tile_n + rbase + (0, 1, 8, 9)[i]
            pg[i] = mPageTable[batch_idx, cutlass.min(key, seqlen_k - 1)]

    @cute.jit
    def _k1fa4_gather_v(self, mV_h, pg, c, tidx, sVr, slot, atom):
        for i in cutlass.range_constexpr(4):
            p = k1fa4_utils.elem_pointer(mV_h, (0, 16 * c, pg[i]))
            if const_expr(K1FA4_L2PF["vh"]):
                k1fa4_cp_async16_l2_128(sVr[None, tidx, i, slot].iterator.toint(), p.toint())
            else:
                src = cute.make_tensor(cute.make_ptr(cutlass.Float8E4M3FN, p.toint(), cute.AddressSpace.gmem,
                                                     assumed_align=16), cute.make_layout((16,)))
                cute.copy(atom, src, sVr[None, tidx, i, slot])

    @cute.jit
    def _k1fa4_prefetch_lines(self, mK_h, mV_h, pg, c):
        if const_expr(K1FA4_L2PF["k"]):
            if c == 0:
                for i in cutlass.range_constexpr(4):
                    a = k1fa4_utils.elem_pointer(mK_h, (0, 0, pg[i])).toint()
                    k1fa4_prefetch_l2_128(cutlass.Int64(a) & cutlass.Int64(-128))
        if const_expr(K1FA4_L2PF["v"]):
            if c == 1:
                for i in cutlass.range_constexpr(4):
                    a = k1fa4_utils.elem_pointer(mV_h, (0, 0, pg[i])).toint()
                    k1fa4_prefetch_l2_128(cutlass.Int64(a) & cutlass.Int64(-128))

    @cute.jit
    def _k1fa4_transpose_v(self, sVr32, tidx, slot, vt_base, c, g, rot):
        prmt = k1fa4_utils.prmt_b32
        rr = []
        for i in cutlass.range_constexpr(4):
            r = cute.make_rmem_tensor((4,), cutlass.Uint32)
            cute.autovec_copy(sVr32[None, tidx, i, slot], r)
            rr.append(r)
        for w in cutlass.range_constexpr(4):
            x0 = prmt(rr[0][w], rr[0][w], rot)
            x1 = prmt(rr[1][w], rr[1][w], rot)
            x2 = prmt(rr[2][w], rr[2][w], rot)
            x3 = prmt(rr[3][w], rr[3][w], rot)
            t0 = prmt(x0, x1, cutlass.Uint32(0x5140))
            t1 = prmt(x0, x1, cutlass.Uint32(0x7362))
            t2 = prmt(x2, x3, cutlass.Uint32(0x5140))
            t3 = prmt(x2, x3, cutlass.Uint32(0x7362))
            outs = (prmt(t0, t2, cutlass.Uint32(0x5410)), prmt(t0, t2, cutlass.Uint32(0x7632)),
                    prmt(t1, t3, cutlass.Uint32(0x5410)), prmt(t1, t3, cutlass.Uint32(0x7632)))
            for e in cutlass.range_constexpr(4):
                d = 16 * c + 4 * w + ((e + c) % 4)
                off = (d * self.tile_n + 4 * g) ^ ((d % 8) * 16)
                cute.make_tensor(cute.recast_ptr(vt_base + off, dtype=cutlass.Uint32), cute.make_layout((1,)))[0] = outs[e]

    @cute.jit
    def load(self, mQ, mK, mV, sQ, sK, sV, tma_atom_Q, tma_atom_K, tma_atom_V, mBias, sBias, tma_atom_Bias,
             pipeline_k, pipeline_v, pipeline_bias, pipeline_q, gmem_tiled_copy_Q, mPageTable,
             blocksparse_tensors, block_info, SeqlenInfoCls, TileSchedulerCls, num_splits: Int32 = 1):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        nthr = self.num_threads_per_warp_group
        pre = k1fa4_K["pdl"] and (K1FA4_PREWAIT or k1fa4_K.get("pw", False))
        if const_expr(k1fa4_K["pdl"] and not pre):
            cute.arch.griddepcontrol_wait()
        if const_expr(k1fa4_K["pdl"] and k1fa4_K["trig"] == "start"):
            cute.arch.griddepcontrol_launch_dependents()
        storage = self._k1_storage
        f8 = cutlass.Float8E4M3FN
        nraw = K1FA4_RAW_STAGES
        lag = nraw - 1
        sVr = storage.sVr.get_tensor(cute.make_layout((16, nthr, 4, nraw),
                                                      stride=(1, 16, 16 * nthr, 64 * nthr)))
        sVr32 = cute.make_tensor(cute.recast_ptr(sVr.iterator, dtype=cutlass.Uint32),
                                 cute.make_layout((4, nthr, 4, nraw), stride=(1, 4, 4 * nthr, 16 * nthr)))
        vt_ptr = storage.sV.data_ptr()
        vt_stage_bytes = self.tile_hdimv * self.tile_n
        atom = cute.make_copy_atom(k1fa4_cpasync.CopyG2SOp(cache_mode=k1fa4_cpasync.LoadCacheMode.GLOBAL), f8,
                                   num_bits_per_copy=128)
        lane = tidx % 32
        wv = tidx // 32
        c = lane % 4
        gq = lane // 4
        g = (gq % 2) + 2 * ((gq // 2) % 2) + 4 * wv + 16 * (gq // 4)
        rot = (cutlass.Uint32(0x32103210) >> (4 * c)) & cutlass.Uint32(0xFFFF)
        rbase = 32 * (g // 8) + 16 * ((g % 8) // 4) + 2 * (g % 4)

        q_producer_phase = Int32(1)
        k_state = k1fa4_pipeline.make_pipeline_state(k1fa4_pipeline.PipelineUserType.Producer, self.num_stages)
        v_state = k1fa4_pipeline.make_pipeline_state(k1fa4_pipeline.PipelineUserType.Producer, self.num_stages)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            load_Q, _, _ = k1fa4_copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ,
                                                            single_stage=True)
            pkm = k1fa4_PagedKVManager.create(mPageTable, mK, mV, k1fa4_FastDivmodDivisor(mK.shape[0]), batch_idx,
                                              head_idx, tidx, seqlen.seqlen_k, 0, self.tile_n, self.tile_hdim,
                                              self.tile_hdimv, nthr, mK.element_type, arch=90)
            mV_h = mV[None, None, head_idx, None]
            mK_h = mK[None, None, head_idx, None]
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            nblk = n_block_max - n_block_min
            if const_expr(not pre):
                if warp_idx_in_wg == 0:
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            else:
                if nblk <= 0:
                    cute.arch.griddepcontrol_wait()
                    if warp_idx_in_wg == 0:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            if nblk > 0:
                pg = cute.make_rmem_tensor((4,), Int32)
                pkm.load_page_table(n_block_max - 1)
                self._k1fa4_v_pages(mPageTable, batch_idx, seqlen.seqlen_k, n_block_max - 1, rbase, pg)
                if const_expr(pre):
                    if n_block_max * self.tile_n > seqlen.seqlen_k - seqlen.seqlen_q:
                        cute.arch.griddepcontrol_wait()
                for it in cutlass.range(nblk, unroll=1):
                    n_block = n_block_max - 1 - it
                    pipeline_k.producer_acquire(k_state)
                    pkm.load_KV(n_block, sK[None, None, k_state.index], "K")
                    pipeline_k.producer_commit(k_state)
                    k_state.advance()
                    self._k1fa4_gather_v(mV_h, pg, c, tidx, sVr, it % nraw, atom)
                    cute.arch.cp_async_commit_group()
                    if const_expr(pre):
                        if it == 0:
                            cute.arch.griddepcontrol_wait()
                            if warp_idx_in_wg == 0:
                                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                    if it + 1 < nblk:
                        pkm.load_page_table(n_block - 1)
                        self._k1fa4_v_pages(mPageTable, batch_idx, seqlen.seqlen_k, n_block - 1, rbase, pg)
                        if const_expr(K1FA4_L2PF["k"] or K1FA4_L2PF["v"]):
                            self._k1fa4_prefetch_lines(mK_h, mV_h, pg, c)
                    if it >= lag:
                        cute.arch.cp_async_wait_group(lag)
                        pipeline_v.producer_acquire(v_state)
                        self._k1fa4_transpose_v(sVr32, tidx, (it - lag) % nraw,
                                                vt_ptr + v_state.index * vt_stage_bytes, c, g, rot)
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))
                        v_state.advance()
                cute.arch.cp_async_wait_group(0)
                for j in cutlass.range(cutlass.max(nblk - lag, 0), nblk, unroll=1):
                    pipeline_v.producer_acquire(v_state)
                    self._k1fa4_transpose_v(sVr32, tidx, j % nraw, vt_ptr + v_state.index * vt_stage_bytes, c, g,
                                            rot)
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))
                    v_state.advance()
            q_producer_phase ^= 1
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        pipeline_v.producer_tail(v_state)

def k1fa4_comb_hb(h, ns_p2, d, length):
    if K1FA4_COMB_HB == 0:
        return 1
    if K1FA4_COMB_HB > 0:
        return min(K1FA4_COMB_HB, h)
    hb = min(h, K1FA4_COMB_CAP[length > 1])
    while hb > 1 and ns_p2 * hb * d > K1FA4_COMB_ELEMS:
        hb //= 2
    return hb

@triton.jit
def _k1fa4_combine_gate_hb(OP, LSEP, QKV, OUT, stride_qkv, gate_off, n_rows, NS, B, L, eps,
                           H: tl.constexpr, D: tl.constexpr, NS_P2: tl.constexpr, HB: tl.constexpr,
                           USE_PDL: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // (H // HB)
    hh = (pid % (H // HB)) * HB + tl.arange(0, HB)
    b = t // L
    l = t % L
    d = tl.arange(0, D)
    g = tl.load(QKV + t * stride_qkv + gate_off + hh[:, None] * D + d[None, :]).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    s = tl.arange(0, NS_P2)
    smask = s < NS
    lrow = (b * H + hh) * L + l
    lse = tl.load(LSEP + s[:, None] * (B * H * L) + lrow[None, :], mask=smask[:, None], other=float("-inf"))
    r = t * H + hh
    o = tl.load(OP + s[:, None, None] * (n_rows * D) + r[None, :, None] * D + d[None, None, :],
                mask=smask[:, None, None], other=0.0)
    m = tl.max(lse, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(lse - m[None, :])
    den = tl.sum(w, axis=0)
    acc = tl.sum(w[:, :, None] * o, axis=0) / tl.where(den > 0, den, 1.0)[:, None]
    acc = acc.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(acc * acc, axis=1) / D
    y = acc * tl.rsqrt(var + eps)[:, None]
    y = y * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(OUT + t * (H * D) + hh[:, None] * D + d[None, :], y.to(tl.bfloat16))

def k1fa4_num_splits_heuristic(total_mblocks, num_sms, num_n_blocks, max_splits):
    if num_n_blocks <= 4 or total_mblocks == 0:
        return 1
    return min(num_sms // total_mblocks, max_splits, num_n_blocks)

def k1fa4_num_splits_for(batch, hkv, keys, num_sms=132, ctas_per_sm=2):
    nnb = (keys + K1FA4_TILE[1] - 1) // K1FA4_TILE[1]
    return max(2, min(K1FA4_MAX_SPLITS, k1fa4_num_splits_heuristic(batch * hkv, num_sms, nnb, 128) * ctas_per_sm))

def k1fa4_fwd_partials(q8, k8, v8, page_table, seqused_k, softmax_scale, num_splits, window=0):
    assert q8.dtype == k8.dtype == v8.dtype == torch.float8_e4m3fn and k8.shape[1] == 1 and num_splits >= 2
    B, L, hq, d = q8.shape
    hkv = k8.shape[2]
    assert d == 64 and hq == 8 * hkv
    local = bool(window)
    wl, wr = (window - 1, 0) if local else (None, None)
    o_part = torch.empty((num_splits, B, L, hq, d), dtype=torch.float32, device=q8.device)
    lse_part = torch.empty((num_splits, B, hq, L), dtype=torch.float32, device=q8.device)
    kind = "sliding" if local else "full"
    pv8 = bool(K1FA4_PV8.get((kind, q8.shape[1]), False))
    k1fa4_K["pw"] = bool(ATT1_PW.get((kind, B), False))
    k1fa4_K["trig_o"] = ATT1_TRIG.get((kind, B))
    key = (local, hq // hkv, d, K1FA4_STAGES[kind], K1FA4_TRIG[kind], k1fa4_K["pdl"], k1fa4_K["minb"],
           K1FA4_REGS1[pv8], pv8, K1FA4_P8_OFFSET, tuple(sorted(K1FA4_EX2.items())), tuple(sorted(K1FA4_L2PF.items())), K1FA4_PREWAIT, k1fa4_K["pw"], k1fa4_K["trig_o"])
    fn = _k1fa4_compiled.get(key)
    if fn is None:
        fa = (k1fa4_Sm90FwdPV8 if pv8 else k1fa4_Sm90Fwd)(cutlass.Float8E4M3FN, d, d, hq // hkv, is_causal=not local, is_local=local,
                           is_split_kv=True, pack_gqa=True, tile_m=K1FA4_TILE[0], tile_n=K1FA4_TILE[1],
                           num_stages=2, num_threads=384, Q_in_regs=False, intra_wg_overlap=False, mma_pv_is_rs=True,
                           mask_mod=None, score_mod=None, has_aux_tensors=False, q_subtile_factor=None,
                           paged_kv_non_tma=True, has_bias=False, bias_block_size=K1FA4_TILE[0],
                           rel_extent_padded=0)
        tc = k1fa4_to_cute_tensor
        fn = cute.compile(
            fa, tc(q8), tc(k8), tc(v8), tc(o_part), tc(lse_part, assumed_align=4), softmax_scale, None, None, None,
            tc(seqused_k, assumed_align=4, leading_dim=0), tc(page_table, assumed_align=4, leading_dim=1), wl, wr,
            None, None, k1fa4_AuxData(None, None), None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        _k1fa4_compiled[key] = fn
    fn(q8.view(torch.uint8), k8.view(torch.uint8), v8.view(torch.uint8), o_part, lse_part, softmax_scale, None, None,
       None, seqused_k, page_table, wl, wr, None, None, k1fa4_AuxData(None, None), None)
    return o_part, lse_part

@triton.jit
def _k1fa4_prologue(QKV, Q_OUT, POS, COS_SIN, K_CACHE, V_CACHE, CACHE_LOC, eps, stride_qkv, n_rows,
                    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                    BLOCK_M: tl.constexpr, HAS_ROPE: tl.constexpr, USE_PDL: tl.constexpr,
                    FP8_MAX_C: tl.constexpr = 448.0, TF: tl.constexpr = False):
    if USE_PDL:
        if TF:
            tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
        if not TF:
            tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, BD)
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    HEADS: tl.constexpr = NUM_Q_HEADS + 2 * NUM_KV_HEADS
    token = rows // HEADS
    head = rows % HEADS
    is_q = head < NUM_Q_HEADS
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS
    src = token * stride_qkv + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    x = raw.to(tl.float32)
    var = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D
    rstd = tl.rsqrt(var + eps)
    x = x * rstd[:, None]
    if HAS_ROPE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
        partner_cols = cols ^ 1
        xp = tl.load(QKV + src[:, None] + partner_cols[None, :], mask=mask, other=0.0).to(tl.float32)
        xp = (xp * rstd[:, None]).to(QKV.dtype.element_ty).to(tl.float32)
        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cs_mask = row_mask[:, None] & (cols < D)[None, :]
        cos = tl.load(COS_SIN + pos[:, None] * D + freq[None, :], mask=cs_mask, other=1.0)
        sin = tl.load(COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=cs_mask, other=0.0)
        is_even = (cols % 2 == 0)[None, :]
        x = tl.where(is_even, x * cos - xp * sin, xp * sin + x * cos)
    x = x.to(QKV.dtype.element_ty).to(tl.float32)
    x = tl.where(is_v[:, None], raw.to(tl.float32), x)
    x = tl.minimum(tl.maximum(x, -FP8_MAX_C), FP8_MAX_C)
    loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)
    kv_head = tl.where(is_v, head - NUM_Q_HEADS - NUM_KV_HEADS, head - NUM_Q_HEADS)
    kv_dst = loc * (NUM_KV_HEADS * D) + kv_head * D
    tl.store(K_CACHE + kv_dst[:, None] + cols[None, :], x, mask=mask & is_k[:, None])
    tl.store(V_CACHE + kv_dst[:, None] + cols[None, :], x, mask=mask & is_v[:, None])
    q_dst = tl.where(is_q, (token * NUM_Q_HEADS + head) * D, 0)
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])

@triton.jit
def _k1fa4_combine_gate(OP, LSEP, QKV, OUT, stride_qkv, gate_off, n_rows, NS, B, L, eps,
                        H: tl.constexpr, D: tl.constexpr, NS_P2: tl.constexpr, USE_PDL: tl.constexpr):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    r = tl.program_id(0)
    t = r // H
    h = r % H
    b = t // L
    l = t % L
    lrow = (b * H + h) * L + l
    s = tl.arange(0, NS_P2)
    smask = s < NS
    lse = tl.load(LSEP + s * (B * H * L) + lrow, mask=smask, other=float("-inf"))
    m = tl.max(lse, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(lse - m)
    den = tl.sum(w, axis=0)
    d = tl.arange(0, D)
    o = tl.load(OP + s[:, None] * (n_rows * D) + r * D + d[None, :], mask=smask[:, None], other=0.0)
    acc = tl.sum(w[:, None] * o, axis=0) / tl.where(den > 0, den, 1.0)
    acc = acc.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(acc * acc, axis=0) / D
    y = acc * tl.rsqrt(var + eps)
    g = tl.load(QKV + t * stride_qkv + gate_off + h * D + d).to(tl.float32)
    y = y * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(OUT + t * (H * D) + h * D + d, y.to(tl.bfloat16))

def k1fa4_cos_sin(theta, d, device):
    key = (theta, str(device))
    cache = _k1fa4_rope.get(key)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(K1FA4_ROPE_POSITIONS, dtype=torch.float, device=device),
                             inv_freq)
        cache = _k1fa4_rope[key] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache

def k1fa4_supported(cfg):
    return (cfg["q_len"] in (1, 8) and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"])

def k1fa4_route_ok(cfg):
    if not k1fa4_supported(cfg):
        return False
    return bool(cfg["window"]) or cfg["q_len"] == 8 or cfg["batch"] <= 1

def k1fa4_run(data, splits=None, pdl=True):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    window = cfg["window"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    q8 = torch.empty((tokens, hq * d), dtype=torch.float8_e4m3fn, device=device)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = K1FA4_PRO_BM or (4 if n_rows < 1024 else (8 if n_rows < 8192 else 16))
    theta = cfg["rope_theta"]
    cos_sin = k1fa4_cos_sin(theta, d, device) if theta else qkv
    pdl_kw = {"launch_pdl": True} if pdl else {}
    _k1fa4_prologue[(triton.cdiv(n_rows, block_m),)](
        qkv, q8, data["positions"] if theta else qkv, cos_sin, k_cache.view(-1, hkv * d), v_cache.view(-1, hkv * d),
        data["cache_loc"], K1FA4_RMS_EPS, qkv.stride(0), n_rows, hq, hkv, d, triton.next_power_of_2(d), block_m,
        HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=K1FA4_PRO_WARPS,
        TF=bool(pdl and ATT1_PRO_TF and batch == 1 and not window), **pdl_kw)
    max_k = data["page_table"].shape[1]
    keys = min(max_k, window + length) if window else max_k
    ns = splits or K1FA4_SPLITS.get(("sliding" if window else "full", batch)) or k1fa4_num_splits_for(batch, hkv, keys)
    k1fa4_K["pdl"] = bool(pdl)
    o_part, lse_part = k1fa4_fwd_partials(q8.view(batch, length, hq, d), k_cache, v_cache, data["page_table"],
                                          data["cache_seqlens"], cfg["sm_scale"] * cfg["qk_scale_factor"], ns,
                                          window=window)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    rows = tokens * hq
    hb = k1fa4_comb_hb(hq, triton.next_power_of_2(ns), d, length)
    if hb > 1:
        _k1fa4_combine_gate_hb[(tokens * (hq // hb),)](
            o_part, lse_part, qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, K1FA4_RMS_EPS,
            H=hq, D=d, NS_P2=triton.next_power_of_2(ns), HB=hb, USE_PDL=pdl, num_warps=K1FA4_COMB_WARPS,
            **pdl_kw)
    else:
        _k1fa4_combine_gate[(rows,)](
            o_part, lse_part, qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, K1FA4_RMS_EPS,
            H=hq, D=d, NS_P2=triton.next_power_of_2(ns), USE_PDL=pdl, num_warps=2 if ns <= 8 else (ATT1_COMB_NW or 4), **pdl_kw)
    return out

def _k1fa4_case(batch, ctx, length, window, seed, row0, device):
    g = torch.Generator(device=device).manual_seed(seed)
    hq, hkv, d = 16, 2, 64
    seq = torch.randint(max(1, ctx // 2), ctx + 1, (batch,), generator=g, device=device)
    seq[0] = ctx
    max_pages = ctx + length
    total = batch * max_pages
    perm = torch.randperm(total, generator=g, device=device)
    if row0:
        j = int((perm == 0).nonzero()[0, 0])
        perm[j], perm[ctx] = perm[ctx].clone(), 0
    pt = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    off = 0
    for b in range(batch):
        n = int(seq[b]) + length
        pt[b, :n] = perm[off:off + n].to(torch.int32)
        off += n
    f8 = torch.float8_e4m3fn
    k_cache = torch.randn((total, 1, hkv, d), generator=g, device=device).to(f8)
    v_cache = torch.randn((total, 1, hkv, d), generator=g, device=device).to(f8)
    q = (2 * torch.randn((batch, length, hq, d), generator=g, device=device)).to(torch.bfloat16)
    k = (2 * torch.randn((batch, length, hkv, d), generator=g, device=device)).to(torch.bfloat16)
    v = torch.randn((batch, length, hkv, d), generator=g, device=device).to(torch.bfloat16)
    gate = torch.randn((batch, length, hq * d), generator=g, device=device).to(torch.bfloat16)
    qkv = torch.cat([q.flatten(2), k.flatten(2), v.flatten(2), gate], -1)
    positions = (seq[:, None] + torch.arange(length, device=device)).reshape(-1)
    cache_loc = torch.gather(pt.long(), 1, positions.view(batch, length)).reshape(-1)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    return {
        "qkv": qkv, "positions": positions, "cache_loc": cache_loc,
        "cache_seqlens": (seq + length).to(torch.int32),
        "cu_seqlens_q": torch.arange(0, batch * length + 1, length, dtype=torch.int32, device=device),
        "scheduler_metadata": None, "k_cache": k_cache, "v_cache": v_cache, "k_scale": one, "v_scale": one.clone(),
        "page_table": pt, "seq_lens": seq.to(torch.int32),
        "config": {"batch": batch, "q_len": length, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": d,
                   "page_size": 1, "window": window, "rope_theta": 500000.0 if window else 0.0,
                   "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": 1.0},
    }

def _k1fa4_reference(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    x = data["qkv"].view(batch, length, -1)
    q = x[..., :hq * d].reshape(batch, length, hq, d).float()
    k = x[..., hq * d:(hq + hkv) * d].reshape(batch, length, hkv, d).float()
    v = x[..., (hq + hkv) * d:(hq + 2 * hkv) * d].reshape(batch, length, hkv, d)
    gate = x[..., (hq + 2 * hkv) * d:].float()

    def rms(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + K1FA4_RMS_EPS)

    def rope(t, pos, theta):
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=t.device).float() / d))
        ang = pos.float()[..., None] * inv
        cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
        even, odd = t[..., 0::2], t[..., 1::2]
        return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)

    pos = data["seq_lens"].long()[:, None] + torch.arange(length, device=q.device)
    qn, kn = rms(q) * cfg["qk_scale_factor"], rms(k)
    if cfg["rope_theta"]:
        qn, kn = rope(qn, pos, cfg["rope_theta"]), rope(kn, pos, cfg["rope_theta"])
    qn = qn.to(torch.bfloat16).float()
    k_new, v_new = kn.to(torch.float8_e4m3fn).float(), v.float().to(torch.float8_e4m3fn).float()
    kf, vf = data["k_cache"].view(-1, hkv, d), data["v_cache"].view(-1, hkv, d)
    out = torch.empty((batch, length, hq, d), device=q.device)
    for b in range(batch):
        seq = int(data["seq_lens"][b])
        rows = data["page_table"][b, :seq].long()
        keys = torch.cat([kf[rows].float(), k_new[b]])
        vals = torch.cat([vf[rows].float(), v_new[b]])
        for j in range(length):
            p = seq + j
            start = max(0, p - cfg["window"] + 1) if cfg["window"] else 0
            att = torch.einsum("kgd,tkd->kgt", qn[b, j].view(hkv, hq // hkv, d), keys[start:p + 1]) * cfg["sm_scale"]
            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vals[start:p + 1]).reshape(hq, d)
    out = rms(out) * torch.sigmoid(gate.view(batch, length, hq, d))
    return out.reshape(batch, length, hq * d)

def _k1fa4_r1_path(data):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache as k1fa4_fa3
    from sglang.kernels.ops.kvcache.kvcache import store_cache as k1fa4_store_cache
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm as k1fa4_FusedQKNorm
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate as k1fa4_rms_norm_gate
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    width = hkv * d
    f8 = torch.float8_e4m3fn
    k_stage = torch.empty((tokens + 1, width), dtype=f8, device=device)
    v_stage = torch.empty((tokens + 1, width), dtype=f8, device=device)
    kv_store = (k_stage, v_stage, torch.arange(1, tokens + 1, dtype=torch.int64, device=device))
    norm = k1fa4_FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=K1FA4_RMS_EPS)
    if cfg["rope_theta"]:
        q8, _ = norm.forward_triton(qkv, data["positions"], k1fa4_cos_sin(cfg["rope_theta"], d, device), kv_store)
    else:
        q8, _ = norm.forward_triton(qkv, None, None, kv_store)
    k1fa4_store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8),
                      data["k_cache"].view(-1, width).view(torch.uint8), data["v_cache"].view(-1, width).view(torch.uint8),
                      data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    out = k1fa4_fa3(q=q8.view(tokens, hq, d), k_cache=data["k_cache"], v_cache=data["v_cache"],
                    page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
                    cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
                    softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                    window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=None)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = k1fa4_rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d,
                            eps=K1FA4_RMS_EPS, has_norm=True)
    return o.view(batch, length, hq * d)

K1FA4_SELF_TEST = [
    (3, 2100, 1, 2048, 101, False), (4, 6000, 8, 2048, 102, False), (2, 9000, 8, 0, 103, False),
    (1, 3000, 1, 0, 104, False), (3, 100, 8, 0, 107, False), (1, 10, 1, 2048, 108, False),
    (5, 3000, 1, 2048, 105, True), (2, 4000, 8, 0, 106, True),
]

def k1fa4_self_test(device=None, cases=None, verbose=True):
    device = device or torch.device("cuda")
    ok_all = True
    try:
        for batch, ctx, length, window, seed, row0 in (cases or K1FA4_SELF_TEST):
            ref = _k1fa4_reference(_k1fa4_case(batch, ctx, length, window, seed, row0, device))
            d1 = _k1fa4_case(batch, ctx, length, window, seed, row0, device)
            o1 = _k1fa4_r1_path(d1)
            d6 = _k1fa4_case(batch, ctx, length, window, seed, row0, device)
            o6 = k1fa4_run(d6)
            torch.cuda.synchronize()
            e1 = float((o1.float() - ref).norm() / ref.norm())
            e6 = float((o6.float() - ref).norm() / ref.norm())
            rows = d6["cache_loc"]
            same = (torch.equal(d6["k_cache"].view(-1, 128).view(torch.uint8)[rows],
                                d1["k_cache"].view(-1, 128).view(torch.uint8)[rows])
                    and torch.equal(d6["v_cache"].view(-1, 128).view(torch.uint8)[rows],
                                    d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]))
            ok = bool(torch.isfinite(o6).all()) and e6 <= 1.25 * e1 + 5e-3 and same
            ok_all &= ok
            if verbose:
                print(f"[k1fa4] self-test B{batch} ctx{ctx} L{length} w{window}{' row0' if row0 else ''}: "
                      f"{'PASS' if ok else 'FAIL'} err {e6:.4e} (r1 {e1:.4e}) cache==r1 {same}", flush=True)
    except Exception as ex:
        print(f"[k1fa4] self-test ERROR {repr(ex)[:400]}", flush=True)
        ok_all = False
    K1FA4_STATE["ready"] = ok_all
    return ok_all

K1FA4V6_ENABLE = True

K1FA4V6_DECODE_SLIDING = True

K1FA4V6_DECODE_FULL_B1 = True

K1FA4V6_VERIFY_SLIDING = True

K1FA4V6_VERIFY_FULL = True

K1FA4V6_VERIFY_FULL_MIN_BATCH = 16

K1FA4V6_PREPARE_BY_S = 200.0

K1FA4V6_WARM_CASES = ((16, 3000, 1, 2048, 401, False), (32, 3000, 1, 2048, 402, False), (16, 3000, 8, 2048, 403, False), (32, 3000, 8, 2048, 404, False), (1, 20000, 1, 0, 405, False), (1, 20000, 8, 0, 406, False), (16, 2500, 8, 0, 407, True), (32, 1500, 8, 0, 408, False), (1, 4000, 8, 2048, 409, False))

K1FA4V6_GRAPH_CASES = ((4, 6000, 8, 2048, 411), (1, 3000, 1, 0, 412))

def _v18_fa4_wanted(qkv, data, cfg, state):
    if not state.get("fa4v6") or not K1FA4_STATE["ready"] or not K1FA4V6_ENABLE:
        return False
    if not k1fa4_route_ok(cfg):
        return False
    if cfg["window"]:
        live = K1FA4V6_VERIFY_SLIDING if cfg["q_len"] == 8 else K1FA4V6_DECODE_SLIDING
    elif cfg["q_len"] == 8:
        live = K1FA4V6_VERIFY_FULL and (cfg["batch"] == 1 or cfg["batch"] >= K1FA4V6_VERIFY_FULL_MIN_BATCH)
    else:
        live = K1FA4V6_DECODE_FULL_B1
    return (bool(live) and cfg["rope_theta"] in (0.0, ROPE_THETA)
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2 and data["page_table"].is_contiguous()
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].is_contiguous()
            and data["cache_loc"].is_contiguous() and data["positions"].is_contiguous())

def _v18_fa4_core(qkv, data, pdl=True):
    return k1fa4_run(dict(data, qkv=qkv), pdl=pdl)

def _v18_graph_check(device):
    for batch, ctx, length, window, seed in K1FA4V6_GRAPH_CASES:
        d = _k1fa4_case(batch, ctx, length, window, seed, False, device)
        qkv = d["qkv"].view(batch * length, -1)
        o1 = _v18_fa4_core(qkv, d).clone()
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            og = _v18_fa4_core(qkv, d)
        g.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1))
        print(f"[fa4v6] graph check B{batch} ctx{ctx} L{length} w{window}: replay identical {same}", flush=True)
        del g, og, o1, d
        if not same:
            return False
    return True

def _v18_fa4_prepare(state, t0):
    state["fa4v6"] = False
    K1FA4_STATE["ready"] = False
    print(f"[v18] levers: K1FA4V6_ENABLE={K1FA4V6_ENABLE} decode_sliding={K1FA4V6_DECODE_SLIDING} decode_full_b1={K1FA4V6_DECODE_FULL_B1} "
          f"verify_sliding={K1FA4V6_VERIFY_SLIDING} verify_full={K1FA4V6_VERIFY_FULL} (batch 1 or >= {K1FA4V6_VERIFY_FULL_MIN_BATCH}) "
          f"stages {K1FA4_STAGES} trig {K1FA4_TRIG} splits {K1FA4_SPLITS} knobs {k1fa4_K}", flush=True)
    if not K1FA4V6_ENABLE:
        return
    if time.time() - t0 > K1FA4V6_PREPARE_BY_S:
        print(f"[v18] FA4 v6 skipped: prepare already took {time.time() - t0:.1f}s (> {K1FA4V6_PREPARE_BY_S}s): k3_v14's attention routes stay", flush=True)
        return
    device = state["router"].device
    try:
        from sglang.kernels.jit.utils import is_arch_support_pdl
        pdl_ok = bool(is_arch_support_pdl())
    except Exception:
        pdl_ok = False
    if not pdl_ok:
        print("[v18] FA4 v6 skipped: no PDL support on this arch (v6 is built and tested with PDL): k3_v14's attention routes stay", flush=True)
        return
    t1 = time.time()
    _k1fa4_rope[(ROPE_THETA, str(state["cos_sin"].device))] = state["cos_sin"]
    ok = bool(k1fa4_self_test(device, cases=list(K1FA4_SELF_TEST) + list(K1FA4V6_WARM_CASES)))
    t2 = time.time()
    if ok:
        try:
            ok = _v18_graph_check(device)
        except Exception as ex:
            print(f"[v18] FA4 v6 graph check raised {ex!r}", flush=True)
            ok = False
    state["fa4v6"] = bool(ok and K1FA4_STATE["ready"])
    print(f"[v18] FA4 v6 route {'live' if state['fa4v6'] else 'FAILED: k3_v14 attention routes stay'} (self-test incl. JIT {t2 - t1:.1f}s, graph check "
          f"{time.time() - t2:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)

def _r1_attention(qkv, data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    rope = (data["positions"], state["cos_sin"]) if cfg["rope_theta"] else (None, None)
    if R1_QK_FUSED_FP8_STAGE and not scaled and V10_QK_DIRECT and state.get("v10qk"):
        q8 = _v10_qk_store(qkv, rope[0], rope[1], k_cache.view(-1, hkv * d), v_cache.view(-1, hkv * d),
                           data["cache_loc"], hq, hkv, d)
        q_fp8 = q8.view(tokens, hq, d)
    elif R1_QK_FUSED_FP8_STAGE and not scaled:
        width = hkv * d
        k_stage = torch.empty((tokens + 1, width), dtype=FP8, device=qkv.device)
        v_stage = torch.empty((tokens + 1, width), dtype=FP8, device=qkv.device)
        stage_loc = state["stage_loc"]
        loc = (stage_loc[:tokens] if tokens <= stage_loc.shape[0]
               else torch.arange(1, tokens + 1, dtype=torch.int64, device=qkv.device))
        q8, _ = state["qk_norm"].forward_triton(qkv, rope[0], rope[1], (k_stage, v_stage, loc))
        store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8),
                    k_cache.view(-1, width).view(torch.uint8), v_cache.view(-1, width).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q_fp8 = q8.view(tokens, hq, d)
    else:
        if cfg["rope_theta"]:
            q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
        else:
            q, k = state["qk_norm"](qkv)
        v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
        k = k.reshape(tokens, hkv * d)
        if scaled:
            k, v = k / data["k_scale"], v / data["v_scale"]
        store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                    k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q_fp8 = q.view(tokens, hq, d).to(FP8)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    sched = data["scheduler_metadata"]
    splits = NUM_SPLITS_OVERRIDE.get(("sliding" if window else "full", batch, length), 0)
    if splits:
        descale["num_splits"] = splits
        sched = None
    out = flash_attn_with_kvcache(
        q=q_fp8,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=sched,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    return state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)

KS1_CPP_SRC = r"""
#include <torch/extension.h>
void ks1_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out,
                 torch::Tensor prof, int64_t L, int64_t NS, int64_t NTG, double qk_sm, double eps, int64_t trig,
                 int64_t pdl, int64_t sync, int64_t phalf, int64_t fold);
std::vector<int64_t> ks1_info(int64_t L, int64_t NS, int64_t NTG);
void ks3_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope);
std::vector<int64_t> ks3_info(int64_t NS);
void ax3_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor ws_o, torch::Tensor ws_l, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, int64_t VAR, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope);
std::vector<int64_t> ax3_info(int64_t NS, int64_t VAR);
"""

KS1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define KS1_ARCH_OK 1
#else
#define KS1_ARCH_OK 0
#endif

namespace ks1 {

constexpr int HQ = 16, HKV = 2, WIN = 2048;    // tp8 shard of Muse Spark: 16 q heads, 2 KV heads, head dim 64, window 2048
constexpr int NWARP = 8, NTHR = NWARP * 32;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float FP8_MAX = 448.0f;
constexpr unsigned FULL = 0xffffffffu;

struct Params {
  const __nv_bfloat16* qkv;   // [L, stride] q | k | v | gate
  const int* cache_seqlens;   // [1] old length + L
  const int64_t* cache_loc;   // [L]
  const int64_t* positions;   // [L]
  const int* page_table;      // [1, pt_cols]
  uint8_t* k_cache;           // e4m3 rows of HKV * 64 B (page size 1)
  uint8_t* v_cache;
  const float* cos_sin;       // [P, 64] cos | sin
  float* part_o;              // [2 NTG][NS][ROWS][64]
  float2* part_ml;            // [2 NTG][NS][ROWS] (max, sum) in the exp2 domain
  int* counters;              // [2 NTG] arrival tickets, zero between calls (self-resetting)
  __nv_bfloat16* out;         // [L, HQ * 64]
  long long* prof;            // lab: [CTAs][8] %globaltimer stamps, or nullptr
  int stride_qkv;             // elements
  int pt_cols;
  int B;                      // KS2: requests (grid z = B x NTG)
  int fold;                   // KS2: 1 = new keys on split NS - 1 (NS CTAs per group), 0 = dedicated new-key CTA s = NS
  int trig;                   // 0: implicit trigger at exit; 1: launch_dependents right after the wait; 2: at entry
  int phalf;                  // 1: partials as normalized fp16 O (else unnormalized fp32)
  int sync;                   // ticket protocol: 0 every thread fence.sc + relaxed atomic (+ fence.sc in the merger);
  float qk_sm;                // sm_scale * qk_scale_factor
  float eps;
};

template <int ROWS, int NSRC> struct CLB {             // VSL cluster merge buffer in each owner CTA (MG 2)
  static constexpr int RPO = (ROWS + NSRC - 1) / NSRC;     // rows one owner finalizes (owner of row r = r % NSRC)
  static constexpr int OFF_ML = RPO * NSRC * 64 * 4;       // [RPO][NSRC][64] fp32 partial o, then [RPO][NSRC] (max, sum)
  static constexpr int BYTES = OFF_ML + RPO * NSRC * 8;
};

template <int L, int NS, int NTG> struct Geo {
  static constexpr int KPS = WIN / NS;                        // old-key slots per split (NS * KPS = 2048 >= WIN - 1)
  static constexpr int TPG = L / NTG;                         // tokens per group (rows = 8 heads x TPG)
  static constexpr int ROWS = 8 * TPG;
  static constexpr int KPW = KPS / NWARP;                     // key slots per warp
  static constexpr int NT = KPW / 8;                          // n-tiles of 8 keys per warp
  static constexpr int TPK = (NTHR >= KPS) ? NTHR / KPS : 1;  // loader threads per key slot
  static constexpr int KPT = (KPS > NTHR) ? KPS / NTHR : 1;   // key slots per loader thread
  static constexpr int CPT = 4 / TPK;                         // 16-B chunks of the 64-B K (and V) head row per thread and slot
  static constexpr int OFF_VH = KPS * 128;
  static constexpr int OFF_KN = 2 * KPS * 128;
  static constexpr int OFF_VN = OFF_KN + 2048;
  static constexpr int OFF_Q = OFF_VN + 2048;
  static constexpr int OFF_ML = OFF_Q + 2048;
  static constexpr int OFF_MISC = OFF_ML + 2 * NWARP * 16 * 4;
  static constexpr int SMEM = OFF_MISC + 128;
  static constexpr int OFF_CB = SMEM;
  static constexpr int SMEM_CL = OFF_CB + CLB<ROWS, (NS < 16 ? NS : 16)>::BYTES;
  static_assert(KPW % 16 == 0 && TPK <= 4 && (4 % TPK) == 0, "key geometry");
  static_assert(NWARP * 16 * 64 * 4 <= 2 * KPS * 128, "merge scratch aliases the K / V tiles");
  static_assert(L % NTG == 0 && (TPG == 1 || TPG == 2), "token groups of 1 or 2 tokens");
};

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(FULL, v, d);
  return v;
}
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t v) {   // low 16 bits = two e4m3 -> f16x2 (exact)
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xFFFFu), __NV_E4M3);
  return (uint32_t)r.x | ((uint32_t)r.y << 16);
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 bf2f(uint32_t v) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if KS1_ARCH_OK
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const uint8_t* p) {
#if KS1_ARCH_OK
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
#endif
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const uint8_t* p) {
#if KS1_ARCH_OK
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
#endif
}
__device__ __forceinline__ uint4 ldg_nc16(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void pdl_wait() {
#if KS1_ARCH_OK
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if KS1_ARCH_OK
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
__device__ __forceinline__ int atom_add_acqrel(int* a, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void fence_acqrel() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void vsl_red4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void vsl_red2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" ::"l"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void vsl_red1(float* a, float x) { asm volatile("red.global.add.f32 [%0], %1;\n" ::"l"(a), "f"(x) : "memory"); }
__device__ __forceinline__ uint32_t vsl_sa(const void* x) { return (uint32_t)__cvta_generic_to_shared(x); }
__device__ __forceinline__ uint32_t vsl_mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void vsl_st4(uint32_t a, float x, float y, float z, float w) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void vsl_st2(uint32_t a, float x, float y) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};\n" ::"r"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void vsl_carrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void vsl_carrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void vsl_cwait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

template <int DPT, int UPR>
__device__ __forceinline__ void vsl_fin1(const float (&a)[DPT], float den, const uint32_t (&graw)[DPT / 2], float eps, __nv_bfloat16* op) {
  const float dd = (den > 0.f) ? den : 1.f;
  float y[DPT];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < DPT; ++i) { y[i] = bfr(a[i] / dd); ss = fmaf(y[i], y[i], ss); }
#pragma unroll
  for (int d = UPR / 2; d > 0; d >>= 1) ss += __shfl_xor_sync(FULL, ss, d);
  const float rs = rsqrtf(ss * (1.f / 64.f) + eps);
  uint32_t ow2[DPT / 2];
#pragma unroll
  for (int i = 0; i < DPT / 2; ++i) {
    const float2 gg = bf2f(graw[i]);
    const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * i] * rs * (1.f / (1.f + __expf(-gg.x))),
                                                    y[2 * i + 1] * rs * (1.f / (1.f + __expf(-gg.y))));
    ow2[i] = *reinterpret_cast<const uint32_t*>(&o2);
  }
  if (op == nullptr) return;
  if constexpr (DPT == 4) *reinterpret_cast<uint2*>(op) = make_uint2(ow2[0], ow2[DPT / 2 - 1]);
  else *reinterpret_cast<uint32_t*>(op) = ow2[0];
}

__device__ __forceinline__ void vsl_fin3(const float4 a, float den, const uint2 graw, float eps, __nv_bfloat16* op) {
  const float dd = (den > 0.f) ? den : 1.f;
  const float y0 = bfr(a.x / dd), y1 = bfr(a.y / dd), y2 = bfr(a.z / dd), y3 = bfr(a.w / dd);
  float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
  ss += __shfl_xor_sync(FULL, ss, 8);
  ss += __shfl_xor_sync(FULL, ss, 4);
  ss += __shfl_xor_sync(FULL, ss, 2);
  ss += __shfl_xor_sync(FULL, ss, 1);
  const float rs = rsqrtf(ss * (1.f / 64.f) + eps);
  const float2 g01 = bf2f(graw.x), g23 = bf2f(graw.y);
  const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0 * rs * (1.f / (1.f + __expf(-g01.x))), y1 * rs * (1.f / (1.f + __expf(-g01.y))));
  const __nv_bfloat162 o23 = __floats2bfloat162_rn(y2 * rs * (1.f / (1.f + __expf(-g23.x))), y3 * rs * (1.f / (1.f + __expf(-g23.y))));
  if (op != nullptr)
    *reinterpret_cast<uint2*>(op) = make_uint2(*reinterpret_cast<const uint32_t*>(&o01), *reinterpret_cast<const uint32_t*>(&o23));
}

#define KS1P(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[cta * 16 + (k)] = gtime(); } while (0)

__device__ __forceinline__ int swz(int x, int ch) { return x * 128 + 16 * (ch ^ (x & 7)); }

template <int NT, typename F>
__device__ __forceinline__ void segment(const uint8_t* kb, const uint8_t* vb, const int k0, const uint32_t (&qa)[4][4],
                                        float (&o)[8][4], float (&m)[2], float (&ls)[2], const float cl2, F valid) {
  const int lane = threadIdx.x & 31, t = lane & 3;
  float sc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    sc[j][0] = 0.f; sc[j][1] = 0.f; sc[j][2] = 0.f; sc[j][3] = 0.f;
    const int key = k0 + 8 * j + (lane & 7);
    uint32_t b[4], c[4];
    ldsm_x4(b, kb + swz(key, lane >> 3));
    ldsm_x4(c, kb + swz(key, 4 + (lane >> 3)));
    mma16816(sc[j], qa[0], b[0], b[1]);
    mma16816(sc[j], qa[1], b[2], b[3]);
    mma16816(sc[j], qa[2], c[0], c[1]);
    mma16816(sc[j], qa[3], c[2], c[3]);
  }
  float x0 = neg_inf(), x1 = neg_inf();
#pragma unroll
  for (int j = 0; j < NT; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int k = k0 + 8 * j + 2 * t + e;
      sc[j][e] = valid(k, 0) ? sc[j][e] * cl2 : neg_inf();
      sc[j][2 + e] = valid(k, 1) ? sc[j][2 + e] * cl2 : neg_inf();
      x0 = fmaxf(x0, sc[j][e]);
      x1 = fmaxf(x1, sc[j][2 + e]);
    }
  }
  x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, 1));
  x0 = fmaxf(x0, __shfl_xor_sync(FULL, x0, 2));
  x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, 1));
  x1 = fmaxf(x1, __shfl_xor_sync(FULL, x1, 2));
  const float n0 = fmaxf(m[0], x0), n1 = fmaxf(m[1], x1);
  const float u0 = (n0 == neg_inf()) ? 0.f : n0, u1 = (n1 == neg_inf()) ? 0.f : n1;
  const float a0 = ex2(m[0] - u0), a1 = ex2(m[1] - u1);
  m[0] = n0; m[1] = n1;
  ls[0] *= a0; ls[1] *= a1;
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk) {
    uint32_t pa[4];
    {
      const float p00 = ex2(sc[2 * kk][0] - u0), p01 = ex2(sc[2 * kk][1] - u0);
      const float p02 = ex2(sc[2 * kk][2] - u1), p03 = ex2(sc[2 * kk][3] - u1);
      const float p10 = ex2(sc[2 * kk + 1][0] - u0), p11 = ex2(sc[2 * kk + 1][1] - u0);
      const float p12 = ex2(sc[2 * kk + 1][2] - u1), p13 = ex2(sc[2 * kk + 1][3] - u1);
      ls[0] += (p00 + p01) + (p10 + p11);
      ls[1] += (p02 + p03) + (p12 + p13);
      pa[0] = pack_h2(p00, p01); pa[1] = pack_h2(p02, p03); pa[2] = pack_h2(p10, p11); pa[3] = pack_h2(p12, p13);
    }
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      const int key = k0 + 16 * kk + 8 * ((lane >> 3) & 1) + (lane & 7);
      uint32_t b[4];
      ldsm_x4_t(b, vb + swz(key, 2 * np + (lane >> 4)));
      mma16816(o[2 * np], pa, b[0], b[1]);
      mma16816(o[2 * np + 1], pa, b[2], b[3]);
    }
  }
}

template <int L, int NS, int NTG, int MG = 0>
__global__ void __launch_bounds__(NTHR, 2) ks1_kernel(const __grid_constant__ Params p) {
#if KS1_ARCH_OK
  using G = Geo<L, NS, NTG>;
  constexpr int TPG = G::TPG, ROWS = G::ROWS, KPS = G::KPS, KPW = G::KPW;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* kh = smem;
  uint8_t* vh = smem + G::OFF_VH;
  uint8_t* kn = smem + G::OFF_KN;
  uint8_t* vn = smem + G::OFF_VN;
  uint8_t* qs = smem + G::OFF_Q;
  float* mw = reinterpret_cast<float*>(smem + G::OFF_ML);
  float* lw = mw + NWARP * 16;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, h = blockIdx.y, b = (int)blockIdx.z / NTG, tg = (int)blockIdx.z % NTG;   // KS2: z = request x group
  const int grp = (b * HKV + h) * NTG + tg;
  const int cta = ((int)blockIdx.z * HKV + h) * (NS + 1) + s;
  const int tb = b * L;                                         // KS2: first token of request b
  const bool tail = p.fold ? (s == NS - 1) : (s == NS);         // the CTA that attends (and stores) the new keys
  const bool oldc = (s < NS);                                   // holds old-key slots
  const int NSP = NS + (p.fold ? 0 : 1);                        // partials per group
  if constexpr ((MG & 3) == 2) vsl_carrive_relaxed();
  if (p.trig == 2) pdl_trigger();
  KS1P(0);

  const int seq = max(__ldg(p.cache_seqlens + b) - L, 0);         // cached length before this block
  const int n_old = min(seq, WIN - 1);                          // old keys in the window of the block's first token
  const int kstart = seq - n_old;
  const int part = tid % G::TPK;
  uint4 kr[G::KPT][G::CPT], vr[G::KPT][G::CPT];
  {
    int prow[G::KPT];
#pragma unroll
    for (int kk = 0; kk < G::KPT; ++kk) {
      const int slot = tid / G::TPK + kk * (NTHR / G::TPK);
      const int gs = s * KPS + slot;
      prow[kk] = (oldc && gs < n_old && kstart + gs < p.pt_cols) ? __ldg(p.page_table + (size_t)b * p.pt_cols + kstart + gs) : -1;
    }
#pragma unroll
    for (int kk = 0; kk < G::KPT; ++kk) {
      const bool ok = prow[kk] >= 0;
      const uint8_t* kp = p.k_cache + (size_t)(ok ? prow[kk] : 0) * (HKV * 64) + h * 64 + 16 * part * G::CPT;
      const uint8_t* vp = p.v_cache + (size_t)(ok ? prow[kk] : 0) * (HKV * 64) + h * 64 + 16 * part * G::CPT;
#pragma unroll
      for (int c = 0; c < G::CPT; ++c) {
        kr[kk][c] = ok ? ldg_nc16(kp + 16 * c) : make_uint4(0u, 0u, 0u, 0u);
        vr[kk][c] = ok ? ldg_nc16(vp + 16 * c) : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
  float2 csq[TPG];
#pragma unroll
  for (int j = 0; j < TPG; ++j) {
    const long long pos = __ldg(p.positions + tb + tg * TPG + j);
    csq[j] = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
  }
  float2 csk = make_float2(1.f, 0.f);
  long long loc = 0;
  if (tail && warp < L) {
    const long long pos = __ldg(p.positions + tb + warp);
    csk = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
    loc = __ldg(p.cache_loc + tb + warp);
  }
  KS1P(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  KS1P(2);

  const size_t sq = (size_t)p.stride_qkv;
  uint32_t qraw[TPG];
#pragma unroll
  for (int j = 0; j < TPG; ++j)
    qraw[j] = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + (size_t)(tb + tg * TPG + j) * sq + (size_t)(h * 8 + warp) * 64) + lane);
  uint32_t kraw = 0u, vraw = 0u;
  if (tail && warp < L) {
    kraw = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + (size_t)(tb + warp) * sq + (size_t)(HQ + h) * 64) + lane);
    vraw = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + (size_t)(tb + warp) * sq + (size_t)(HQ + HKV + h) * 64) + lane);
  }
  constexpr int DPT = (ROWS == 8) ? 2 : 4;                      // merge item (one per thread): row mr, dims DPT mu .. + DPT - 1
  constexpr int UPR = 64 / DPT;                                 // threads per row
  static_assert(ROWS * UPR == NTHR, "one merge item per thread");
  const int mr = tid / UPR, mu = tid % UPR;
  uint32_t graw[DPT / 2];
  {
    const __nv_bfloat16* gp = p.qkv + (size_t)(tb + tg * TPG + (mr >> 3)) * sq + (size_t)(HQ + 2 * HKV) * 64 +
                              (size_t)(h * 8 + (mr & 7)) * 64 + DPT * mu;
    if constexpr (DPT == 4) {
      const uint2 v = __ldcg(reinterpret_cast<const uint2*>(gp));
      graw[0] = v.x;
      graw[DPT / 2 - 1] = v.y;
    } else {
      graw[0] = __ldcg(reinterpret_cast<const unsigned int*>(gp));
    }
  }
#pragma unroll
  for (int kk = 0; kk < (oldc ? G::KPT : 0); ++kk) {
    const int slot = tid / G::TPK + kk * (NTHR / G::TPK);
#pragma unroll
    for (int c = 0; c < G::CPT; ++c) {
      const int ch = part * G::CPT + c;                           // fp8 chunk: dims 16 ch .. 16 ch + 15
      const uint4 a = kr[kk][c], b = vr[kk][c];
      *reinterpret_cast<uint4*>(kh + swz(slot, 2 * ch)) =
          make_uint4(e4m3x2_to_h2(a.x), e4m3x2_to_h2(a.x >> 16), e4m3x2_to_h2(a.y), e4m3x2_to_h2(a.y >> 16));
      *reinterpret_cast<uint4*>(kh + swz(slot, 2 * ch + 1)) =
          make_uint4(e4m3x2_to_h2(a.z), e4m3x2_to_h2(a.z >> 16), e4m3x2_to_h2(a.w), e4m3x2_to_h2(a.w >> 16));
      *reinterpret_cast<uint4*>(vh + swz(slot, 2 * ch)) =
          make_uint4(e4m3x2_to_h2(b.x), e4m3x2_to_h2(b.x >> 16), e4m3x2_to_h2(b.y), e4m3x2_to_h2(b.y >> 16));
      *reinterpret_cast<uint4*>(vh + swz(slot, 2 * ch + 1)) =
          make_uint4(e4m3x2_to_h2(b.z), e4m3x2_to_h2(b.z >> 16), e4m3x2_to_h2(b.w), e4m3x2_to_h2(b.w >> 16));
    }
  }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int r = warp + 8 * j;
    uint32_t hv = 0u;
    if (j < TPG) {
      const float2 x = bf2f(qraw[j < TPG ? j : 0]);
      const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
      const float rs = rsqrtf(var + p.eps);
      const float e = bfr(x.x * rs), o = bfr(x.y * rs);
      const float2 c = csq[j < TPG ? j : 0];
      const float e2 = bfr(__fmaf_rn(e, c.x, -__fmul_rn(o, c.y)));
      const float o2 = bfr(__fmaf_rn(e, c.y, __fmul_rn(o, c.x)));
      hv = pack_h2(e2, o2);
    }
    *reinterpret_cast<uint32_t*>(qs + r * 128 + 16 * ((lane >> 2) ^ (r & 7)) + 4 * (lane & 3)) = hv;
  }
  if (tail) {   // new keys: rows warp, warp + 8 of the 16-slot tile (rows >= L stay zero)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int r = warp + 8 * j;
      uint32_t kh2 = 0u, vh2 = 0u;
      if (r < L) {
        const float2 x = bf2f(kraw);
        const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
        const float rs = rsqrtf(var + p.eps);
        const float e = bfr(x.x * rs), o = bfr(x.y * rs);
        float e2 = bfr(__fmaf_rn(e, csk.x, -__fmul_rn(o, csk.y)));
        float o2 = bfr(__fmaf_rn(e, csk.y, __fmul_rn(o, csk.x)));
        e2 = fminf(fmaxf(e2, -FP8_MAX), FP8_MAX);
        o2 = fminf(fmaxf(o2, -FP8_MAX), FP8_MAX);
        float2 v = bf2f(vraw);
        v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
        v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
        const uint32_t k8 = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(e2, o2), __NV_SATFINITE, __NV_E4M3);
        const uint32_t v8 = (uint32_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
        if (tg == 0) {
          *reinterpret_cast<uint16_t*>(p.k_cache + (size_t)loc * (HKV * 64) + h * 64 + 2 * lane) = (uint16_t)k8;
          *reinterpret_cast<uint16_t*>(p.v_cache + (size_t)loc * (HKV * 64) + h * 64 + 2 * lane) = (uint16_t)v8;
        }
        kh2 = e4m3x2_to_h2(k8);
        vh2 = e4m3x2_to_h2(v8);
      }
      *reinterpret_cast<uint32_t*>(kn + r * 128 + 16 * ((lane >> 2) ^ (r & 7)) + 4 * (lane & 3)) = kh2;
      *reinterpret_cast<uint32_t*>(vn + r * 128 + 16 * ((lane >> 2) ^ (r & 7)) + 4 * (lane & 3)) = vh2;
    }
  }
  KS1P(3);
  __syncthreads();
  KS1P(4);

  const float cl2 = p.qk_sm * LOG2E;
  uint32_t qa[4][4];
  {
    const int qr = (lane & 7) + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) ldsm_x4(qa[ks], qs + swz(qr, 2 * ks + (lane >> 4)));
  }
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m[2] = {neg_inf(), neg_inf()}, ls[2] = {0.f, 0.f};
  const int gbase = s * KPS;                                     // window slot of key 0 of this CTA's tile
  const int l0 = tg * TPG;                                       // token of rows g (rows g + 8: l0 + 1 when TPG = 2)
  if (oldc) {
    segment<G::NT>(kh, vh, warp * KPW, qa, o, m, ls, cl2, [&](int k, int j) {
      const int gs = gbase + k;                                  // key at absolute position kstart + gs
      const int l = l0 + ((TPG == 2) ? j : 0);
      return gs < n_old && gs >= n_old + l + 1 - WIN;            // > position(l) - WIN (window of token l)
    });
  }
  if (tail && warp == (p.fold ? NWARP - 1 : 0)) {
    segment<2>(kn, vn, 0, qa, o, m, ls, cl2, [&](int k, int j) {
      const int l = l0 + ((TPG == 2) ? j : 0);
      return k < L && k <= l;                                    // causal inside the block
    });
  }
  ls[0] += __shfl_xor_sync(FULL, ls[0], 1);
  ls[0] += __shfl_xor_sync(FULL, ls[0], 2);
  ls[1] += __shfl_xor_sync(FULL, ls[1], 1);
  ls[1] += __shfl_xor_sync(FULL, ls[1], 2);

  KS1P(5);
  __syncthreads();
  float* ow = reinterpret_cast<float*>(smem);                   // [NWARP][16][64], dims XOR 8 (row & 7)
  if (t == 0) {
    mw[warp * 16 + g] = m[0]; lw[warp * 16 + g] = ls[0];
    mw[warp * 16 + g + 8] = m[1]; lw[warp * 16 + g + 8] = ls[1];
  }
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = (8 * n + 2 * t) ^ (8 * g);
    *reinterpret_cast<float2*>(ow + (warp * 16 + g) * 64 + d) = make_float2(o[n][0], o[n][1]);
    if (ROWS == 16) *reinterpret_cast<float2*>(ow + (warp * 16 + g + 8) * 64 + d) = make_float2(o[n][2], o[n][3]);
  }
  __syncthreads();
  KS1P(6);
  uint16_t* ph = reinterpret_cast<uint16_t*>(p.part_o);       // phalf: normalized partial O in fp16 (half the merge bytes)
  {
    float M = neg_inf();
#pragma unroll
    for (int w = 0; w < NWARP; ++w) M = fmaxf(M, mw[w * 16 + mr]);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
    float lt = 0.f;
#pragma unroll
    for (int w = 0; w < NWARP; ++w) {
      const float f = ex2(mw[w * 16 + mr] - Mu);
      lt = fmaf(f, lw[w * 16 + mr], lt);
      const float* src = ow + (w * 16 + mr) * 64 + ((DPT * mu) ^ (8 * (mr & 7)));
      if constexpr (DPT == 4) {
        const float4 x = *reinterpret_cast<const float4*>(src);
        acc[0] = fmaf(f, x.x, acc[0]); acc[1] = fmaf(f, x.y, acc[1]); acc[DPT - 2] = fmaf(f, x.z, acc[DPT - 2]);
        acc[DPT - 1] = fmaf(f, x.w, acc[DPT - 1]);
      } else {
        const float2 x = *reinterpret_cast<const float2*>(src);
        acc[0] = fmaf(f, x.x, acc[0]); acc[1] = fmaf(f, x.y, acc[1]);
      }
    }
    if constexpr ((MG & 3) == 1) {   // VSL red merge: o 2^M, l 2^M into the zeroed workspace, ticket, the last CTA finalizes one row set
      const float f = (M == neg_inf()) ? 0.f : ex2(fminf(M, 126.f));
      float* wa = p.part_o + ((size_t)grp * ROWS + mr) * 64 + DPT * mu;
      float* wl = reinterpret_cast<float*>(p.part_ml) + (size_t)grp * ROWS + mr;
      if constexpr (DPT == 4) vsl_red4(wa, acc[0] * f, acc[1] * f, acc[DPT - 2] * f, acc[DPT - 1] * f);
      else vsl_red2(wa, acc[0] * f, acc[1] * f);
      if (mu == 0) vsl_red1(wl, lt * f);
      KS1P(7);
      __syncthreads();
      if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + grp, 1) == NSP - 1) ? 1 : 0;
      __syncthreads();
      KS1P(8);
      if (misc[0] == 0) return;
      float a[DPT];
      float den = (mu == 0) ? __ldcg(wl) : 0.f;
      if constexpr (DPT == 4) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(wa));
        a[0] = v.x; a[1] = v.y; a[DPT - 2] = v.z; a[DPT - 1] = v.w;
        __stcg(reinterpret_cast<float4*>(wa), make_float4(0.f, 0.f, 0.f, 0.f));
      } else {
        const float2 v = __ldcg(reinterpret_cast<const float2*>(wa));
        a[0] = v.x; a[1] = v.y;
        __stcg(reinterpret_cast<float2*>(wa), make_float2(0.f, 0.f));
      }
      if (mu == 0) __stcg(wl, 0.f);
      den = __shfl_sync(FULL, den, lane & ~(UPR - 1));
      vsl_fin1<DPT, UPR>(a, den, graw, p.eps, p.out + (size_t)(tb + tg * TPG + (mr >> 3)) * (HQ * 64) + (size_t)(h * 8 + (mr & 7)) * 64 + DPT * mu);
      if (tid == 0) p.counters[grp] = 0;
      KS1P(9);
      return;
    }
    const size_t prow = ((size_t)grp * (NS + 1) + s) * ROWS + mr;
    if (p.phalf) {
      const float inv = (lt > 0.f) ? 1.f / lt : 0.f;
      if constexpr (DPT == 4)
        __stcg(reinterpret_cast<uint2*>(ph + prow * 64 + DPT * mu),
               make_uint2(pack_h2(acc[0] * inv, acc[1] * inv), pack_h2(acc[DPT - 2] * inv, acc[DPT - 1] * inv)));
      else
        __stcg(reinterpret_cast<unsigned int*>(ph + prow * 64 + DPT * mu), pack_h2(acc[0] * inv, acc[1] * inv));
    } else {
      if constexpr (DPT == 4)
        __stcg(reinterpret_cast<float4*>(p.part_o + prow * 64 + DPT * mu), make_float4(acc[0], acc[1], acc[DPT - 2], acc[DPT - 1]));
      else
        __stcg(reinterpret_cast<float2*>(p.part_o + prow * 64 + DPT * mu), make_float2(acc[0], acc[1]));
    }
    if (mu == 0) __stcg(p.part_ml + prow, make_float2(M, lt));
  }
  KS1P(7);
  if (p.sync == 0) __threadfence();
  __syncthreads();
  if (tid == 0) {
    int tk;
    if (p.sync == 1) {
      tk = atom_add_acqrel(p.counters + grp, 1);
    } else {
      if (p.sync == 2) fence_acqrel();
      tk = atomicAdd(p.counters + grp, 1);
      if (p.sync == 2 && tk == NSP - 1) fence_acqrel();
    }
    misc[0] = (tk == NSP - 1) ? 1 : 0;                                // NS + 1 arrivals per group
  }
  __syncthreads();
  KS1P(8);
  if (misc[0] == 0) return;

  if (p.sync == 0) __threadfence();
  {
    float2 ml[NS + 1];
    float ov[NS + 1][DPT];
#pragma unroll
    for (int q = 0; q <= NS; ++q) {
      const size_t prow = ((size_t)grp * (NS + 1) + q) * ROWS + mr;
      if (q >= NSP) {                                            // KS2 fold: no slot NS
        ml[q] = make_float2(neg_inf(), 0.f);
#pragma unroll
        for (int i = 0; i < DPT; ++i) ov[q][i] = 0.f;
        continue;
      }
      ml[q] = __ldcg(p.part_ml + prow);
      if (p.phalf) {
        if constexpr (DPT == 4) {
          const uint2 v = __ldcg(reinterpret_cast<const uint2*>(ph + prow * 64 + DPT * mu));
          const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
          const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
          ov[q][0] = a.x; ov[q][1] = a.y; ov[q][DPT - 2] = b.x; ov[q][DPT - 1] = b.y;
        } else {
          const unsigned int v = __ldcg(reinterpret_cast<const unsigned int*>(ph + prow * 64 + DPT * mu));
          const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v));
          ov[q][0] = a.x; ov[q][1] = a.y;
        }
      } else {
        if constexpr (DPT == 4) {
          const float4 v = __ldcg(reinterpret_cast<const float4*>(p.part_o + prow * 64 + DPT * mu));
          ov[q][0] = v.x; ov[q][1] = v.y; ov[q][DPT - 2] = v.z; ov[q][DPT - 1] = v.w;
        } else {
          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.part_o + prow * 64 + DPT * mu));
          ov[q][0] = v.x; ov[q][1] = v.y;
        }
      }
    }
    float M = neg_inf();
#pragma unroll
    for (int q = 0; q <= NS; ++q) M = fmaxf(M, ml[q].x);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float den = 0.f;
    float a[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) a[i] = 0.f;
#pragma unroll
    for (int q = 0; q <= NS; ++q) {
      const float w = ex2(ml[q].x - Mu);
      const float wl = w * ml[q].y;                              // split weight: e^(m_q - M) l_q
      const float wa = p.phalf ? wl : w;                         // fp16 partials are normalized, fp32 ones are not
      den += wl;
#pragma unroll
      for (int i = 0; i < DPT; ++i) a[i] = fmaf(wa, ov[q][i], a[i]);
    }
    const float dd = (den > 0.f) ? den : 1.f;
    float y[DPT];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < DPT; ++i) { y[i] = bfr(a[i] / dd); ss = fmaf(y[i], y[i], ss); }
#pragma unroll
    for (int d = UPR / 2; d > 0; d >>= 1) ss += __shfl_xor_sync(FULL, ss, d);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    uint32_t ow2[DPT / 2];
#pragma unroll
    for (int i = 0; i < DPT / 2; ++i) {
      const float2 gg = bf2f(graw[i]);
      const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * i] * rs * (1.f / (1.f + __expf(-gg.x))),
                                                      y[2 * i + 1] * rs * (1.f / (1.f + __expf(-gg.y))));
      ow2[i] = *reinterpret_cast<const uint32_t*>(&o2);
    }
    __nv_bfloat16* op = p.out + (size_t)(tb + tg * TPG + (mr >> 3)) * (HQ * 64) + (size_t)(h * 8 + (mr & 7)) * 64 + DPT * mu;
    if constexpr (DPT == 4) *reinterpret_cast<uint2*>(op) = make_uint2(ow2[0], ow2[DPT / 2 - 1]);
    else *reinterpret_cast<uint32_t*>(op) = ow2[0];
  }
  if (tid == 0) p.counters[grp] = 0;                             // self-reset (every split of the group has arrived)
  KS1P(9);
#endif
}

template <int L, int NS, int NTG, int MG = 0>
static void launch(const Params& P, bool pdl, cudaStream_t st) {
  using G = Geo<L, NS, NTG>;
  constexpr int SM = ((MG & 3) == 2) ? G::SMEM_CL : G::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    if ((MG & 3) == 2 && NS > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG, MG>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(NS + (P.fold ? 0 : 1)), (unsigned)HKV, (unsigned)(NTG * P.B));
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  unsigned na = 0;
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  if ((MG & 3) == 2) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = (unsigned)NS;
    at[na].val.clusterDim.y = 1;
    at[na].val.clusterDim.z = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks1_kernel<L, NS, NTG, MG>, P));
}

template <int L, int NS, int NTG, int MG = 0>
static std::vector<int64_t> info_t() {
  using G = Geo<L, NS, NTG>;
  constexpr int SM = ((MG & 3) == 2) ? G::SMEM_CL : G::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks1_kernel<L, NS, NTG, MG>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks1_kernel<L, NS, NTG, MG>, NTHR, SM));
  if (MG == 0) return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
  int ncl = 0;
  if ((MG & 3) == 2) {
    if (NS > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG, MG>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)NS, (unsigned)HKV, (unsigned)(NTG * 64));
    cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
    cfg.dynamicSmemBytes = SM;
    cudaLaunchAttribute a1;
    a1.id = cudaLaunchAttributeClusterDimension;
    a1.val.clusterDim.x = (unsigned)NS;
    a1.val.clusterDim.y = 1;
    a1.val.clusterDim.z = 1;
    cfg.attrs = &a1;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&ncl, ks1_kernel<L, NS, NTG, MG>, &cfg));
  }
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ, (int64_t)ncl};
}

}  // namespace ks1

namespace ks3 {
using namespace ks1;

struct Params3 {
  const __nv_bfloat16* qkv;   // [B, stride] q | k | v | gate (q_len 1)
  const int* cache_seqlens;   // [B]
  const int64_t* cache_loc;   // [B]
  const int64_t* positions;   // [B]
  const int* page_table;      // [B, pt_cols]
  uint8_t* k_cache;           // e4m3 rows of 2 x 64 B (both KV heads)
  uint8_t* v_cache;
  const float* cos_sin;
  float* part_o;              // [B][NS][16][64]
  float2* part_ml;            // [B][NS][16]
  int* counters;              // [B]
  __nv_bfloat16* out;         // [B, 1024]
  long long* prof;            // [B NS][8] or nullptr
  int stride_qkv, pt_cols, B, trig, rope;
  float qk_sm, eps;
};

template <int NS> struct G3 {
  static constexpr int KMAX = WIN / NS;                 // old-key slots per split
  static constexpr int KPW = KMAX / 4;                  // slots per warp (4 key parts per KV head)
  static constexpr int NCH = KPW / 32;                  // 32-key chunks per warp
  static constexpr int OFF_V = KMAX * 128;
  static constexpr int OFF_KN = 2 * KMAX * 128;         // new-key tiles: 32 rows each (rows >= 1 zero)
  static constexpr int OFF_VN = OFF_KN + 32 * 128;
  static constexpr int OFF_Q = OFF_VN + 32 * 128;       // q fp16 [16 heads][64]
  static constexpr int OFF_ML = OFF_Q + 16 * 128;       // per warp (max, sum) of its 8 rows
  static constexpr int OFF_MISC = OFF_ML + 2 * NWARP * 8 * 4;
  static constexpr int SMEM = OFF_MISC + 64;
  static constexpr int OFF_CB = SMEM;
  static constexpr int SMEM_CL = OFF_CB + CLB<16, NS>::BYTES;
  static_assert(KPW % 32 == 0 && 2 * KMAX * 128 >= NWARP * 8 * 64 * 4 && (KMAX * 8) % NTHR == 0, "KS3 geometry");
};

__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gptr), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ int kswz(int x, int ch) { return ch ^ (4 * (x & 1)); }
__device__ __forceinline__ int vswz(int x, int ch) { return ch ^ (2 * ((x >> 1) & 3)); }
__device__ __forceinline__ uint2 vld(const uint8_t* Vs, int x, int h, int g) {
  return *reinterpret_cast<const uint2*>(Vs + x * 128 + 16 * vswz(x, 4 * h + (g >> 1)) + 8 * (g & 1));
}
#define KS3P(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[cta * 8 + (k)] = gtime(); } while (0)

__device__ __forceinline__ float2 norm_rope(uint32_t raw, float2 cs, int rope, float eps) {
  const float2 x = bf2f(raw);
  const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
  const float rs = rsqrtf(var + eps);
  const float e = bfr(x.x * rs), o = bfr(x.y * rs);
  if (!rope) return make_float2(e, o);
  return make_float2(bfr(__fmaf_rn(e, cs.x, -__fmul_rn(o, cs.y))), bfr(__fmaf_rn(e, cs.y, __fmul_rn(o, cs.x))));
}

template <typename F>
__device__ __forceinline__ void chunk32(const uint8_t* Ks, const uint8_t* Vs, const int x0, const int h, const uint32_t (&qf)[4][2],
                                        float (&o)[8][4], float& m, float& l, const float cl2, F valid) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  float sc[4][2];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int key = x0 + 8 * j + g;
    const uint4 kv = *reinterpret_cast<const uint4*>(Ks + key * 128 + 16 * kswz(key, 4 * h + t));
    const uint32_t wv[4] = {kv.x, kv.y, kv.z, kv.w};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t a[4] = {qf[s][0], 0u, qf[s][1], 0u};
      mma16816(c, a, e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
    }
    sc[j][0] = c[0];
    sc[j][1] = c[1];
  }
  float x = neg_inf();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      sc[j][e] = valid(x0 + 8 * j + 2 * t + e) ? sc[j][e] * cl2 : neg_inf();
      x = fmaxf(x, sc[j][e]);
    }
  }
  x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
  x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
  const float nm = fmaxf(m, x);
  const float u = (nm == neg_inf()) ? 0.f : nm;
  const float al = ex2(m - u);
  m = nm;
  l *= al;
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] *= al; o[n][1] *= al; }
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    const float p00 = ex2(sc[2 * kk][0] - u), p01 = ex2(sc[2 * kk][1] - u);
    const float p10 = ex2(sc[2 * kk + 1][0] - u), p11 = ex2(sc[2 * kk + 1][1] - u);
    l += (p00 + p01) + (p10 + p11);
    const uint32_t pa[4] = {pack_h2(p00, p01), 0u, pack_h2(p10, p11), 0u};
    const int xa = x0 + 16 * kk + 2 * t;
    const uint2 va = vld(Vs, xa, h, g), vb = vld(Vs, xa + 1, h, g), vc = vld(Vs, xa + 8, h, g), vd = vld(Vs, xa + 9, h, g);
    const uint32_t w01[4] = {__byte_perm(va.x, vb.x, 0x5140), __byte_perm(va.x, vb.x, 0x7362),
                             __byte_perm(va.y, vb.y, 0x5140), __byte_perm(va.y, vb.y, 0x7362)};
    const uint32_t w23[4] = {__byte_perm(vc.x, vd.x, 0x5140), __byte_perm(vc.x, vd.x, 0x7362),
                             __byte_perm(vc.y, vd.y, 0x5140), __byte_perm(vc.y, vd.y, 0x7362)};
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      const uint32_t s01 = w01[jn >> 1], s23 = w23[jn >> 1];
      mma16816(o[jn], pa, e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01), e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23));
    }
  }
}

template <int NS, int MG = 0>
__global__ void __launch_bounds__(NTHR, 2) ks3_kernel(const __grid_constant__ Params3 p) {
#if KS1_ARCH_OK
  using G = G3<NS>;
  constexpr int KMAX = G::KMAX, KPW = G::KPW;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* Ks = smem;
  uint8_t* Vs = smem + G::OFF_V;
  uint8_t* kn = smem + G::OFF_KN;
  uint8_t* vn = smem + G::OFF_VN;
  uint8_t* qs = smem + G::OFF_Q;
  float* mw = reinterpret_cast<float*>(smem + G::OFF_ML);
  float* lw = mw + NWARP * 8;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, b = blockIdx.y;
  const int cta = b * NS + s;
  const bool tail = (s == NS - 1);                              // also attends the new key
  const int hw = warp & 1, q4 = warp >> 1;                      // KV head, key part
  if constexpr (MG == 2) vsl_carrive_relaxed();
  if (p.trig == 2) pdl_trigger();
  KS3P(0);
  if constexpr ((MG & 4) != 0) {                         // AX3 rider: red / ticket lines toward L2 at entry
    if (threadIdx.x < 34) {
      const void* a = threadIdx.x < 32 ? (const void*)(p.part_o + (size_t)b * 1024 + 32 * threadIdx.x)
                      : (threadIdx.x == 32 ? (const void*)(reinterpret_cast<float*>(p.part_ml) + (size_t)b * 16)
                                           : (const void*)(p.counters + b));
      asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
    }
  }

  const int seq = max(__ldg(p.cache_seqlens + b) - 1, 0);
  const int n_old = min(seq, WIN - 1);
  const int kstart = seq - n_old;
  {
    constexpr int PER = KMAX * 8 / NTHR;                        // (key, 16-B chunk) pairs per thread
    const int* pt = p.page_table + (size_t)b * p.pt_cols;
    int rows[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int key = (tid + NTHR * j) >> 3;
      const int gs = s * KMAX + key;
      rows[j] = (gs < n_old && kstart + gs < p.pt_cols) ? __ldg(pt + kstart + gs) : -1;
    }
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int r = tid + NTHR * j, key = r >> 3, ch = r & 7;
      const bool ok = rows[j] >= 0;
      const size_t go = (size_t)(ok ? rows[j] : 0) * 128 + 16 * ch;
      cp_async16(Ks + key * 128 + 16 * kswz(key, ch), p.k_cache + go, ok ? 16 : 0);
      cp_async16(Vs + key * 128 + 16 * vswz(key, ch), p.v_cache + go, ok ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  float2 cs = make_float2(1.f, 0.f);
  if (p.rope) {
    const long long pos = __ldg(p.positions + b);
    cs = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
  }
  long long loc = 0;
  if (tail) {
    loc = __ldg(p.cache_loc + b);
    reinterpret_cast<uint4*>(kn)[tid] = make_uint4(0u, 0u, 0u, 0u);          // kn | vn: 8 KB, zero (rows >= 1 stay zero)
    reinterpret_cast<uint4*>(kn)[tid + NTHR] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
  }
  KS3P(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  KS3P(2);

  const __nv_bfloat16* xr = p.qkv + (size_t)b * p.stride_qkv;
  const uint32_t qr0 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)warp * 64) + lane);
  const uint32_t qr1 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(warp + 8) * 64) + lane);
  uint32_t nr = 0u;
  if (tail && warp < 4) nr = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + warp) * 64) + lane);   // k0 k1 v0 v1
  const int mr = tid >> 4, mu = tid & 15;                         // merge item: q head mr, dims 4 mu .. 4 mu + 3
  const uint2 graw = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 4 * mu));
  {
    const float2 a = norm_rope(qr0, cs, p.rope, p.eps), c = norm_rope(qr1, cs, p.rope, p.eps);
    reinterpret_cast<uint32_t*>(qs + warp * 128)[lane] = pack_h2(a.x, a.y);
    reinterpret_cast<uint32_t*>(qs + (warp + 8) * 128)[lane] = pack_h2(c.x, c.y);
  }
  if (tail && warp < 4) {                                        // warp 0 / 1: K head 0 / 1, warp 2 / 3: V head 0 / 1
    const int kind = warp >> 1, head = warp & 1;
    float2 v = kind ? bf2f(nr) : norm_rope(nr, cs, p.rope, p.eps);
    v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
    v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
    const uint32_t q8 = (uint32_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)loc * 128 + head * 64 + 2 * lane) = (uint16_t)q8;
    const int bb = 64 * head + 2 * lane;                          // key 0 of the new tile: chunk bb >> 4 (no swizzle at row 0)
    *reinterpret_cast<uint16_t*>((kind ? vn : kn) + 16 * (bb >> 4) + (bb & 15)) = (uint16_t)q8;
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  KS3P(3);

  uint32_t qf[4][2];
  {
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs + (hw * 8 + g) * 128);
#pragma unroll
    for (int k = 0; k < 4; ++k) { qf[k][0] = qa[8 * t + 2 * k]; qf[k][1] = qa[8 * t + 2 * k + 1]; }
  }
  const float cl2 = p.qk_sm * LOG2E;
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m = neg_inf(), l = 0.f;
  const int base = s * KMAX;
#pragma unroll
  for (int c = 0; c < G::NCH; ++c)
    chunk32(Ks, Vs, q4 * KPW + 32 * c, hw, qf, o, m, l, cl2, [&](int k) { return base + k < n_old; });
  if (tail && q4 == 3) chunk32(kn, vn, 0, hw, qf, o, m, l, cl2, [&](int k) { return k == 0; });
  l += __shfl_xor_sync(FULL, l, 1);
  l += __shfl_xor_sync(FULL, l, 2);
  KS3P(4);

  __syncthreads();
  float* ow = reinterpret_cast<float*>(smem);                   // [8 warps][8 rows][64], dims XOR 4 g
  if (t == 0) { mw[warp * 8 + g] = m; lw[warp * 8 + g] = l; }
#pragma unroll
  for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
    for (int e = 0; e < 2; ++e) ow[(warp * 8 + g) * 64 + ((8 * (2 * t + e) + jn) ^ (4 * g))] = o[jn][e];
  }
  __syncthreads();
  {
    const int rh = mr >> 3, rg = mr & 7;
    float M = neg_inf();
#pragma unroll
    for (int k = 0; k < 4; ++k) M = fmaxf(M, mw[(2 * k + rh) * 8 + rg]);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float lt = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int w = 2 * k + rh;
      const float f = ex2(mw[w * 8 + rg] - Mu);
      lt = fmaf(f, lw[w * 8 + rg], lt);
      const float4 xv = *reinterpret_cast<const float4*>(ow + (w * 8 + rg) * 64 + ((4 * mu) ^ (4 * rg)));
      acc.x = fmaf(f, xv.x, acc.x); acc.y = fmaf(f, xv.y, acc.y); acc.z = fmaf(f, xv.z, acc.z); acc.w = fmaf(f, xv.w, acc.w);
    }
    if constexpr ((MG & 3) == 1) {   // VSL red merge (see KS1; MG 5 = + the AX3 entry prefetch)
      const float f = (M == neg_inf()) ? 0.f : ex2(fminf(M, 126.f));
      float* wa = p.part_o + ((size_t)b * 16 + mr) * 64 + 4 * mu;
      float* wl = reinterpret_cast<float*>(p.part_ml) + (size_t)b * 16 + mr;
      vsl_red4(wa, acc.x * f, acc.y * f, acc.z * f, acc.w * f);
      if (mu == 0) vsl_red1(wl, lt * f);
      KS3P(5);
      __syncthreads();
      if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
      __syncthreads();
      KS3P(6);
      if (misc[0] == 0) return;
      float den = (mu == 0) ? __ldcg(wl) : 0.f;
      const float4 v = __ldcg(reinterpret_cast<const float4*>(wa));
      __stcg(reinterpret_cast<float4*>(wa), make_float4(0.f, 0.f, 0.f, 0.f));
      if (mu == 0) __stcg(wl, 0.f);
      den = __shfl_sync(FULL, den, lane & ~15);
      vsl_fin3(v, den, graw, p.eps, p.out + (size_t)b * (HQ * 64) + (size_t)mr * 64 + 4 * mu);
      if (tid == 0) p.counters[b] = 0;
      KS3P(7);
      return;
    }
    const size_t prow = ((size_t)b * NS + s) * 16 + mr;
    __stcg(reinterpret_cast<float4*>(p.part_o + prow * 64 + 4 * mu), acc);
    if (mu == 0) __stcg(p.part_ml + prow, make_float2(M, lt));
  }
  KS3P(5);
  __syncthreads();
  if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
  __syncthreads();
  KS3P(6);
  if (misc[0] == 0) return;

  {
    float2 ml[NS];
    float4 ov[NS];
#pragma unroll
    for (int q = 0; q < NS; ++q) {
      const size_t prow = ((size_t)b * NS + q) * 16 + mr;
      ml[q] = __ldcg(p.part_ml + prow);
      ov[q] = __ldcg(reinterpret_cast<const float4*>(p.part_o + prow * 64 + 4 * mu));
    }
    float M = neg_inf();
#pragma unroll
    for (int q = 0; q < NS; ++q) M = fmaxf(M, ml[q].x);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float den = 0.f;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int q = 0; q < NS; ++q) {
      const float w = ex2(ml[q].x - Mu);
      den += w * ml[q].y;
      a.x = fmaf(w, ov[q].x, a.x); a.y = fmaf(w, ov[q].y, a.y); a.z = fmaf(w, ov[q].z, a.z); a.w = fmaf(w, ov[q].w, a.w);
    }
    const float dd = (den > 0.f) ? den : 1.f;
    const float y0 = bfr(a.x / dd), y1 = bfr(a.y / dd), y2 = bfr(a.z / dd), y3 = bfr(a.w / dd);
    float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    ss += __shfl_xor_sync(FULL, ss, 8);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 1);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    const float2 g01 = bf2f(graw.x), g23 = bf2f(graw.y);
    const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0 * rs * (1.f / (1.f + __expf(-g01.x))), y1 * rs * (1.f / (1.f + __expf(-g01.y))));
    const __nv_bfloat162 o23 = __floats2bfloat162_rn(y2 * rs * (1.f / (1.f + __expf(-g23.x))), y3 * rs * (1.f / (1.f + __expf(-g23.y))));
    *reinterpret_cast<uint2*>(p.out + (size_t)b * (HQ * 64) + (size_t)mr * 64 + 4 * mu) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&o01), *reinterpret_cast<const uint32_t*>(&o23));
  }
  if (tid == 0) p.counters[b] = 0;                               // self-reset (every split of request b has arrived)
  KS3P(7);
#endif
}

template <int NS, int MG = 0>
static void launch3(const Params3& P, bool pdl, cudaStream_t st) {
  constexpr int SM = (MG == 2) ? G3<NS>::SMEM_CL : G3<NS>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    if (MG == 2 && NS > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS, MG>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  unsigned na = 0;
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  if (MG == 2) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = (unsigned)NS;
    at[na].val.clusterDim.y = 1;
    at[na].val.clusterDim.z = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks3_kernel<NS, MG>, P));
}

template <int NS, int MG = 0>
static std::vector<int64_t> info3() {
  constexpr int SM = (MG == 2) ? G3<NS>::SMEM_CL : G3<NS>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks3_kernel<NS, MG>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks3_kernel<NS, MG>, NTHR, SM));
  if (MG == 0) return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
  int ncl = 0;
  if (MG == 2) {
    if (NS > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS, MG>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)NS, 64u, 1u);
    cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
    cfg.dynamicSmemBytes = SM;
    cudaLaunchAttribute a1;
    a1.id = cudaLaunchAttributeClusterDimension;
    a1.val.clusterDim.x = (unsigned)NS;
    a1.val.clusterDim.y = 1;
    a1.val.clusterDim.z = 1;
    cfg.attrs = &a1;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&ncl, ks3_kernel<NS, MG>, &cfg));
  }
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ, (int64_t)ncl};
}

}  // namespace ks3

#define KS1_INSTS @INSTS@
#define KS1_MINSTS @MINSTS@
#define KS3_MINSTS @M3INSTS@

void ks1_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out,
                 torch::Tensor prof, int64_t L, int64_t NS, int64_t NTG, double qk_sm, double eps, int64_t trig,
                 int64_t pdl, int64_t sync, int64_t phalf, int64_t fold) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1, "ks1: qkv");
  const int64_t B = cache_seqlens.numel();
  TORCH_CHECK(B >= 1 && qkv.size(0) == B * L && qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64, "ks1: qkv shape");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous(), "ks1: int32 cache_seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == B * L && cache_loc.is_contiguous(), "ks1: cache_loc");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == B * L && positions.is_contiguous(), "ks1: positions");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == B &&
              page_table.is_contiguous(), "ks1: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "ks1: caches");
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous() && cos_sin.dim() == 2 && cos_sin.size(1) == 64,
              "ks1: cos_sin");
  const int64_t rows = 8 * (L / NTG);
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_o.numel() >= B * 2 * NTG * (NS + 1) * rows * 64, "ks1: part_o");
  TORCH_CHECK(part_ml.scalar_type() == at::kFloat && part_ml.numel() >= B * 2 * NTG * (NS + 1) * rows * 2, "ks1: part_ml");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= B * 2 * NTG, "ks1: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * L * ks1::HQ * 64, "ks1: out");
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 16 * B * 2 * NTG * (NS + 1)), "ks1: prof");
  const at::cuda::CUDAGuard guard(qkv.device());
  ks1::Params P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.cache_seqlens = cache_seqlens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.positions = positions.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = k_cache.data_ptr<uint8_t>();
  P.v_cache = v_cache.data_ptr<uint8_t>();
  P.cos_sin = cos_sin.data_ptr<float>();
  P.part_o = part_o.data_ptr<float>();
  P.part_ml = reinterpret_cast<float2*>(part_ml.data_ptr<float>());
  P.counters = counters.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.prof = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  P.stride_qkv = (int)qkv.stride(0);
  P.pt_cols = (int)page_table.size(1);
  P.B = (int)B;
  const int vmg = (sync >= 3) ? (int)(sync - 2) : 0;   // VSL: sync 3 = red merge (MG 1), 4 = cluster merge (MG 2, folded)
  P.fold = ((vmg & 3) == 2) ? 1 : ((B > 1 && fold) ? 1 : 0);
  P.trig = (int)trig;
  P.sync = vmg ? 1 : (int)sync;
  P.phalf = (int)phalf;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (vmg) {
#define KS1_XM(L_, NS_, NTG_, MG_) if (vmg == MG_ && L == L_ && NS == NS_ && NTG == NTG_) { ks1::launch<L_, NS_, NTG_, MG_>(P, pdl != 0, st); return; }
    KS1_MINSTS
#undef KS1_XM
    TORCH_CHECK(false, "ks1: VSL variant not built");
  }
#define KS1_X(L_, NS_, NTG_) if (L == L_ && NS == NS_ && NTG == NTG_) { ks1::launch<L_, NS_, NTG_>(P, pdl != 0, st); return; }
  KS1_INSTS
#undef KS1_X
  TORCH_CHECK(false, "ks1: geometry not built");
}

std::vector<int64_t> ks1_info(int64_t L, int64_t NS, int64_t NTG) {
  if (NTG >= 100) {
    const int64_t vmg = NTG / 100;
    NTG %= 100;
#define KS1_XM(L_, NS_, NTG_, MG_) if (vmg == MG_ && L == L_ && NS == NS_ && NTG == NTG_) return ks1::info_t<L_, NS_, NTG_, MG_>();
    KS1_MINSTS
#undef KS1_XM
    return {};
  }
#define KS1_X(L_, NS_, NTG_) if (L == L_ && NS == NS_ && NTG == NTG_) return ks1::info_t<L_, NS_, NTG_>();
  KS1_INSTS
#undef KS1_X
  return {};
}

void ks3_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {
  const int vmg3 = (int)(NS / 100);   // VSL: NS + 100 MG
  NS %= 100;
  const int64_t B = cache_seqlens.numel();
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == B &&
              qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64, "ks3: qkv");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous() && B >= 1, "ks3: cache_seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == B && cache_loc.is_contiguous(), "ks3: cache_loc");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == B && positions.is_contiguous(), "ks3: positions");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == B && page_table.is_contiguous(),
              "ks3: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "ks3: caches");
  TORCH_CHECK(!rope || (cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous() && cos_sin.size(1) == 64), "ks3: cos_sin");
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_o.numel() >= B * NS * 16 * 64, "ks3: part_o");
  TORCH_CHECK(part_ml.scalar_type() == at::kFloat && part_ml.numel() >= B * NS * 16 * 2, "ks3: part_ml");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= B, "ks3: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * ks1::HQ * 64, "ks3: out");
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 8 * B * NS), "ks3: prof");
  const at::cuda::CUDAGuard guard(qkv.device());
  ks3::Params3 P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.cache_seqlens = cache_seqlens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.positions = positions.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = k_cache.data_ptr<uint8_t>();
  P.v_cache = v_cache.data_ptr<uint8_t>();
  P.cos_sin = cos_sin.data_ptr<float>();
  P.part_o = part_o.data_ptr<float>();
  P.part_ml = reinterpret_cast<float2*>(part_ml.data_ptr<float>());
  P.counters = counters.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.prof = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  P.stride_qkv = (int)qkv.stride(0);
  P.pt_cols = (int)page_table.size(1);
  P.B = (int)B;
  P.trig = (int)trig;
  P.rope = (int)rope;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (vmg3) {
#define KS3_XM(NS_, MG_) if (vmg3 == MG_ && NS == NS_) { ks3::launch3<NS_, MG_>(P, pdl != 0, st); return; }
    KS3_MINSTS
#undef KS3_XM
    TORCH_CHECK(false, "ks3: VSL variant not built");
  }
  if (NS == 4) { ks3::launch3<4>(P, pdl != 0, st); return; }
  if (NS == 8) { ks3::launch3<8>(P, pdl != 0, st); return; }
  if (NS == 16) { ks3::launch3<16>(P, pdl != 0, st); return; }
  TORCH_CHECK(false, "ks3: NS must be 4, 8 or 16");
}

std::vector<int64_t> ks3_info(int64_t NS) {
  if (NS >= 100) {
    const int64_t vmg = NS / 100;
    NS %= 100;
#define KS3_XM(NS_, MG_) if (vmg == MG_ && NS == NS_) return ks3::info3<NS_, MG_>();
    KS3_MINSTS
#undef KS3_XM
    return {};
  }
  if (NS == 4) return ks3::info3<4>();
  if (NS == 8) return ks3::info3<8>();
  if (NS == 16) return ks3::info3<16>();
  return {};
}

// ---- AX3 (tools/ax3_patches.py): decode SLIDING attention, B >= 2, q_len 1; 4-warp CTAs over 64-key tiles ----
namespace ax3 {
using namespace ks1;

constexpr int NTA = 128;   // warp w: KV head (w & 1), key half (w >> 1) of every 64-key tile

struct PA {
  const __nv_bfloat16* qkv;   // [B, stride] q | k | v | gate
  const int* cache_seqlens;   // [B] (old length + 1)
  const int64_t* cache_loc;   // [B]
  const int64_t* positions;   // [B]
  const int* page_table;      // [B, pt_cols]
  uint8_t* k_cache;           // e4m3 rows of 2 x 64 B (both KV heads), page size 1
  uint8_t* v_cache;
  const float* cos_sin;       // [P, 64] cos | sin
  float* ws_o;                // [B][16][64] fp32, zero between calls (the request's last CTA re-zeroes its rows)
  float* ws_l;                // [B][16]
  int* counters;              // [B] arrival tickets, zero between calls
  __nv_bfloat16* out;         // [B, 1024]
  long long* prof;            // [B NS][8] %globaltimer marks, or nullptr
  int stride_qkv, pt_cols, B, trig, rope;
  float qk_sm, eps;
};

template <int NS, int AH> struct GA {
  static constexpr int KMAX = WIN / NS;           // window slots per split (2047 old keys + the new one = 2048)
  static constexpr int NTL = KMAX / 64;           // 64-key tiles per split
  static constexpr int NST = AH == 0 ? NTL : (AH + 1 < NTL ? AH + 1 : NTL);   // K / V stages (a ring when AH > 0)
  static constexpr int OFF_V = NST * 64 * 128;
  static constexpr int OFF_Q = 2 * NST * 64 * 128;  // q fp16 [16 heads][64]
  static constexpr int OFF_ML = OFF_Q + 16 * 128;   // [4 warps][8 rows] max, then sum
  static constexpr int OFF_MISC = OFF_ML + 2 * 4 * 8 * 4;
  static constexpr int SMEM = OFF_MISC + 16;
  static_assert(NTL >= 1 && NTL <= 8 && (AH > 0 || NTL <= 4) && 2 * NST * 64 * 128 >= 4 * 8 * 64 * 4, "AX3 geometry");
};

template <int N> __device__ __forceinline__ void cpwait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cpwait_rt(int n) {   // n = groups still allowed in flight (0..3)
  if (n <= 0) cpwait<0>();
  else if (n == 1) cpwait<1>();
  else if (n == 2) cpwait<2>();
  else cpwait<3>();
}
#define AXP(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[cta * 8 + (k)] = gtime(); } while (0)

// VAR bit 0 (lab): whole split landed before the first tile (KS3's order); bit 1 / bit 2: a ring with AH = 2 / 3 tiles in
// flight, the loop issues tile x + AH right after the barrier of tile x (VAR 0: every tile of the split in flight at once);
// bit 3: L2 prefetch of the merge workspace rows, row sums and ticket at entry (cold lines: the reds / ticket miss to DRAM).
template <int NS, int VAR>
__global__ void __launch_bounds__(NTA, 4) ax3_kernel(const __grid_constant__ PA p) {
#if KS1_ARCH_OK
  constexpr int AH = (VAR & 2) ? 2 : ((VAR & 4) ? 3 : 0);
  using G = GA<NS, AH>;
  constexpr int KMAX = G::KMAX, NTL = G::NTL, NST = G::NST;
  constexpr bool WALL = AH == 0 && (VAR & 1) != 0;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* Ks = smem;
  uint8_t* Vs = smem + G::OFF_V;
  uint8_t* qs = smem + G::OFF_Q;
  float* mw = reinterpret_cast<float*>(smem + G::OFF_ML);
  float* lw = mw + 32;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, b = blockIdx.y, cta = b * NS + s;
  const int hw = warp & 1, kh = warp >> 1;
  if (p.trig == 2) pdl_trigger();
  AXP(0);
  if constexpr ((VAR & 8) != 0) {                            // the red / ticket lines toward L2 now, not at the merge
    if (tid < 34) {
      const void* a = tid < 32 ? (const void*)(p.ws_o + (size_t)b * 1024 + 32 * tid)
                               : (tid == 32 ? (const void*)(p.ws_l + (size_t)b * 16) : (const void*)(p.counters + b));
      asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
    }
  }

  const int seq = max(__ldg(p.cache_seqlens + b) - 1, 0);   // old keys; the new key sits at position seq
  const int n_old = min(seq, WIN - 1);
  const int kstart = seq - n_old;
  const int base = s * KMAX;                                 // first window slot of this split
  const int nk = min(KMAX, n_old + 1 - base);                // slots with a key here (old ones + the new one); <= 0: none
  const int ntl = nk > 0 ? (nk + 63) >> 6 : 0;
  const int jn = n_old - base;                               // the new key's slot (this CTA owns it if 0 <= jn < KMAX)
  const bool owner = jn >= 0 && jn < KMAX;
  const int xn = jn >> 6;                                    // (owner) the new key's tile
  const int jrow = (xn % NST) * 64 + (jn & 63);              // (owner) its row in the stage buffers
  // thread: 16-B chunk c of rows r0 + 16 k of a tile; every page entry of the split first (one round trip, registers)
  const int c = tid & 7, r0 = tid >> 3;
  int rows[NTL][4];
  {
    const int* pt = p.page_table + (size_t)b * p.pt_cols;
#pragma unroll
    for (int x = 0; x < NTL; ++x)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int j = base + 64 * x + r0 + 16 * k;
        rows[x][k] = (j < n_old && kstart + j < p.pt_cols) ? __ldg(pt + kstart + j) : -1;
      }
  }
  auto issue = [&](const int x, const int (&rw)[4]) {      // tile x -> stage x % NST (the new key's slot is skipped)
    const int sb = (x % NST) * 64;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = r0 + 16 * k, rs = sb + r;
      if (64 * x + r != jn) {
        const bool ok = rw[k] >= 0;                          // slots past the window end: zero fill (V must be finite)
        const size_t go = (size_t)(ok ? rw[k] : 0) * 128 + 16 * c;
        ks3::cp_async16(Ks + rs * 128 + 16 * ks3::kswz(rs, c), p.k_cache + go, ok ? 16 : 0);
        ks3::cp_async16(Vs + rs * 128 + 16 * ks3::vswz(rs, c), p.v_cache + go, ok ? 16 : 0);
      }
    }
  };
  constexpr int PRE = AH == 0 ? NTL : AH;                    // tiles issued before the PDL wait (one commit group each)
#pragma unroll
  for (int x = 0; x < PRE; ++x) {
    if (x < ntl) issue(x, rows[x]);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  float2 cs = make_float2(1.f, 0.f);
  if (p.rope) {
    const long long pos = __ldg(p.positions + b);
    cs = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
  }
  const long long loc = owner ? __ldg(p.cache_loc + b) : 0;
  AXP(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();

  const __nv_bfloat16* xr = p.qkv + (size_t)b * p.stride_qkv;
  const int mr = tid >> 3, mp = tid & 7;                     // merge / finalize item: q head mr, dims 8 mp .. 8 mp + 7
  const uint4 graw = __ldcg(reinterpret_cast<const uint4*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 8 * mp));
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m = neg_inf(), l = 0.f;
  if (nk > 0) {
    uint32_t qr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qr[i] = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(warp + 4 * i) * 64) + lane);
    uint32_t nr = 0u;
    if (owner) nr = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + warp) * 64) + lane);   // k0 k1 v0 v1
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 a = ks3::norm_rope(qr[i], cs, p.rope, p.eps);
      reinterpret_cast<uint32_t*>(qs + (warp + 4 * i) * 128)[lane] = pack_h2(a.x, a.y);
    }
    uint16_t nq8 = 0;
    uint8_t* ndst = smem;
    if (owner) {                                             // warp 0 / 1: K head 0 / 1, warp 2 / 3: V head 0 / 1
      const int kind = warp >> 1, head = warp & 1;
      float2 v = kind ? bf2f(nr) : ks3::norm_rope(nr, cs, p.rope, p.eps);
      v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
      v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
      nq8 = (uint16_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)loc * 128 + head * 64 + 2 * lane) = nq8;
      const int bb = 64 * head + 2 * lane, cc = bb >> 4;
      ndst = (kind ? (Vs + jrow * 128 + 16 * ks3::vswz(jrow, cc)) : (Ks + jrow * 128 + 16 * ks3::kswz(jrow, cc))) + (bb & 15);
      if (xn < PRE) *reinterpret_cast<uint16_t*>(ndst) = nq8;   // its tile is issued already (else: right after its issue)
    }
    AXP(2);
    if constexpr (WALL) cpwait<0>();
    __syncthreads();                                         // q of every warp, the new key's slot
    uint32_t qf[4][2];
    {
      const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs + (hw * 8 + g) * 128);
#pragma unroll
      for (int k = 0; k < 4; ++k) { qf[k][0] = qa[8 * t + 2 * k]; qf[k][1] = qa[8 * t + 2 * k + 1]; }
    }
    const float cl2 = p.qk_sm * LOG2E;
#pragma unroll 1
    for (int x = 0; x < ntl; ++x) {
      if constexpr (AH == 0) {
        if constexpr (!WALL) {
          cpwait_rt(NTL - 1 - x);
          __syncthreads();                                   // tile x landed (every thread's copies)
        }
      } else {
        cpwait<AH - 1>();
        __syncthreads();                                     // tile x landed; every warp is done with tile x - 1's stage
        const int y = x + AH;
        if (y < ntl) {
          int rw[4] = {-1, -1, -1, -1};
#pragma unroll
          for (int yy = AH; yy < NTL; ++yy)
            if (yy == y) { rw[0] = rows[yy][0]; rw[1] = rows[yy][1]; rw[2] = rows[yy][2]; rw[3] = rows[yy][3]; }
          issue(y, rw);
          if (owner && xn == y) *reinterpret_cast<uint16_t*>(ndst) = nq8;
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
      }
      const int sb = (x % NST) * 64;
      ks3::chunk32(Ks, Vs, sb + 32 * kh, hw, qf, o, m, l, cl2, [&](int k) { return 64 * x + (k - sb) < nk; });
    }
    l += __shfl_xor_sync(FULL, l, 1);
    l += __shfl_xor_sync(FULL, l, 2);
  }
  AXP(3);

  __syncthreads();                                           // K / V reads done: the merge scratch aliases them
  float* ow = reinterpret_cast<float*>(smem);                // [4 warps][8 rows][64], dims XOR 4 g
  if (nk > 0) {
    if (t == 0) { mw[warp * 8 + g] = m; lw[warp * 8 + g] = l; }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ow[(warp * 8 + g) * 64 + ((16 * t + j) ^ (4 * g))] = o[j][0];
      ow[(warp * 8 + g) * 64 + ((16 * t + 8 + j) ^ (4 * g))] = o[j][1];
    }
  }
  __syncthreads();
  if (nk > 0) {                                              // two key halves of row mr -> absolute exp2 frame -> red.add
    const int rh = mr >> 3, rg = mr & 7, w0 = rh, w1 = rh + 2;
    const float m0 = mw[w0 * 8 + rg], m1 = mw[w1 * 8 + rg];
    const float M = fmaxf(m0, m1);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    const float fa = (M == neg_inf()) ? 0.f : ex2(fminf(M, 126.f));
    const float f0 = ex2(m0 - Mu) * fa, f1 = ex2(m1 - Mu) * fa;
    const float lt = f0 * lw[w0 * 8 + rg] + f1 * lw[w1 * 8 + rg];
    const float* r0p = ow + (w0 * 8 + rg) * 64;
    const float* r1p = ow + (w1 * 8 + rg) * 64;
    const int ia = (8 * mp) ^ (4 * rg), ib = (8 * mp + 4) ^ (4 * rg);
    const float4 x0 = *reinterpret_cast<const float4*>(r0p + ia), y0 = *reinterpret_cast<const float4*>(r1p + ia);
    const float4 x1 = *reinterpret_cast<const float4*>(r0p + ib), y1 = *reinterpret_cast<const float4*>(r1p + ib);
    float* wa = p.ws_o + ((size_t)b * 16 + mr) * 64 + 8 * mp;
    vsl_red4(wa, f0 * x0.x + f1 * y0.x, f0 * x0.y + f1 * y0.y, f0 * x0.z + f1 * y0.z, f0 * x0.w + f1 * y0.w);
    vsl_red4(wa + 4, f0 * x1.x + f1 * y1.x, f0 * x1.y + f1 * y1.y, f0 * x1.z + f1 * y1.z, f0 * x1.w + f1 * y1.w);
    if (mp == 0) vsl_red1(p.ws_l + (size_t)b * 16 + mr, lt);
  }
  AXP(4);
  __syncthreads();                                           // every thread's reds before the release (bar.sync cumulativity)
  if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
  __syncthreads();
  AXP(5);
  if (misc[0] == 0) return;
  {                                                          // last CTA of request b: 16 rows x 64 from the workspace
    float* wa = p.ws_o + ((size_t)b * 16 + mr) * 64 + 8 * mp;
    float* wl = p.ws_l + (size_t)b * 16 + mr;
    float den = (mp == 0) ? __ldcg(wl) : 0.f;
    const float4 v0 = __ldcg(reinterpret_cast<const float4*>(wa));
    const float4 v1 = __ldcg(reinterpret_cast<const float4*>(wa + 4));
    __stcg(reinterpret_cast<float4*>(wa), make_float4(0.f, 0.f, 0.f, 0.f));
    __stcg(reinterpret_cast<float4*>(wa + 4), make_float4(0.f, 0.f, 0.f, 0.f));
    if (mp == 0) __stcg(wl, 0.f);
    den = __shfl_sync(FULL, den, lane & ~7);
    const float dd = (den > 0.f) ? den : 1.f;
    const float y[8] = {bfr(v0.x / dd), bfr(v0.y / dd), bfr(v0.z / dd), bfr(v0.w / dd),
                        bfr(v1.x / dd), bfr(v1.y / dd), bfr(v1.z / dd), bfr(v1.w / dd)};
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = fmaf(y[i], y[i], ss);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 1);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    const uint32_t gw[4] = {graw.x, graw.y, graw.z, graw.w};
    uint32_t ov[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 gg = bf2f(gw[i]);
      const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * i] * rs * (1.f / (1.f + __expf(-gg.x))),
                                                      y[2 * i + 1] * rs * (1.f / (1.f + __expf(-gg.y))));
      ov[i] = *reinterpret_cast<const uint32_t*>(&o2);
    }
    *reinterpret_cast<uint4*>(p.out + (size_t)b * (HQ * 64) + (size_t)mr * 64 + 8 * mp) = make_uint4(ov[0], ov[1], ov[2], ov[3]);
  }
  if (tid == 0) p.counters[b] = 0;                           // self-reset (every split of request b has arrived)
  AXP(6);
#endif
}
#undef AXP

template <int NS, int VAR>
static void launch_a(const PA& P, bool pdl, cudaStream_t st) {
  constexpr int SM = GA<NS, (VAR & 2) ? 2 : ((VAR & 4) ? 3 : 0)>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ax3_kernel<NS, VAR>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)NTA, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ax3_kernel<NS, VAR>, P));
}

template <int NS, int VAR>
static std::vector<int64_t> info_a() {
  constexpr int SM = GA<NS, (VAR & 2) ? 2 : ((VAR & 4) ? 3 : 0)>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ax3_kernel<NS, VAR>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ax3_kernel<NS, VAR>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ax3_kernel<NS, VAR>, NTA, SM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
}

}  // namespace ax3

#define AX3_INSTS AX3_X(8, 2) AX3_X(8, 10) AX3_X(16, 10) AX3_X(8, 0)

void ax3_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor ws_o, torch::Tensor ws_l, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, int64_t VAR, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {
  const int64_t B = cache_seqlens.numel();
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == B &&
              qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64 && qkv.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0, "ax3: qkv");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous() && B >= 1, "ax3: cache_seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == B && cache_loc.is_contiguous(), "ax3: cache_loc");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == B && positions.is_contiguous(), "ax3: positions");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == B && page_table.is_contiguous(),
              "ax3: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "ax3: caches");
  TORCH_CHECK(!rope || (cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous() && cos_sin.size(1) == 64), "ax3: cos_sin");
  TORCH_CHECK(ws_o.scalar_type() == at::kFloat && ws_o.numel() >= B * 16 * 64 && ws_l.scalar_type() == at::kFloat &&
              ws_l.numel() >= B * 16, "ax3: workspace");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= B, "ax3: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * ks1::HQ * 64, "ax3: out");
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 8 * B * NS), "ax3: prof");
  const at::cuda::CUDAGuard guard(qkv.device());
  ax3::PA P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.cache_seqlens = cache_seqlens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.positions = positions.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = k_cache.data_ptr<uint8_t>();
  P.v_cache = v_cache.data_ptr<uint8_t>();
  P.cos_sin = cos_sin.data_ptr<float>();
  P.ws_o = ws_o.data_ptr<float>();
  P.ws_l = ws_l.data_ptr<float>();
  P.counters = counters.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.prof = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  P.stride_qkv = (int)qkv.stride(0);
  P.pt_cols = (int)page_table.size(1);
  P.B = (int)B;
  P.trig = (int)trig;
  P.rope = (int)rope;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
#define AX3_X(NS_, VAR_) if (NS == NS_ && VAR == VAR_) { ax3::launch_a<NS_, VAR_>(P, pdl != 0, st); return; }
  AX3_INSTS
#undef AX3_X
  TORCH_CHECK(false, "ax3: variant not built");
}

std::vector<int64_t> ax3_info(int64_t NS, int64_t VAR) {
#define AX3_X(NS_, VAR_) if (NS == NS_ && VAR == VAR_) return ax3::info_a<NS_, VAR_>();
  AX3_INSTS
#undef AX3_X
  return {};
}
"""

KS1_ENABLE = True

KS1_ROUTE = {1: True, 8: True}

KS1_GEOM = {1: (16, 1), 8: (16, 4)}

KS1_TRIG = {1: 1, 8: 1}

KS1_PDL = True

KS1_SYNC = {1: 1, 8: 1}

KS1_PHALF = {1: False, 8: False}

KS1_P1_CARVEOUT = -1

KS1_INSTS = ((1, 16, 1), (8, 16, 4), (1, 4, 1))

KS2_ENABLE = True

KS2_ROUTE = {(16, 1): True, (32, 1): False, (16, 8): False, (32, 8): False}

KS2_GEOM = {(16, 1): (4, 1), (32, 1): (4, 1), (16, 8): (16, 4), (32, 8): (16, 4)}

KS2_TRIG = {(16, 1): 1, (32, 1): 1, (16, 8): 1, (32, 8): 1}

KS2_FOLD = {(16, 1): True, (32, 1): True, (16, 8): True, (32, 8): True}

KS2_SELF_TEST_CASES = ((16, 3000, 1, 941, False), (32, 2600, 1, 942, True), (3, 2100, 1, 943, False))

KS3_ENABLE = True

KS3_ROUTE = {(32, 1): True, (16, 1): True}

KS3_NS = {(32, 1): 8, (16, 1): 8}

KS3_TRIG = {(32, 1): 1, (16, 1): 1}

KS3_INSTS = (8,)

KS3_SELF_TEST_CASES = ((32, 2600, 961, True), (16, 3000, 962, False), (3, 2100, 963, False))

KS1_PREPARE_BY_S = 230.0

VSL_ENABLE = True

VSL_ROUTE = {(1, 1): ('ks1', 1, 16, 1, 1), (1, 8): ('ks1', 8, 16, 4, 1), (16, 1): ('ks3', 16, 5)}

VSL_KS1_INSTS = ((1, 16, 1, 1), (8, 16, 4, 1))

VSL_KS3_INSTS = ((16, 1), (16, 5), (8, 1), (8, 5))

VSL_WS_FLOATS = 1 << 20

VSL_PREPARE_BY_S = 240.0

_VSL = {"ws": {}, "seen": set()}

KS1_PROF_BUF = None

KS1_EPS = 1e-5

KS1_SELF_TEST_CASES = ((3000, 1, 901, False), (3000, 8, 902, False), (2047, 1, 903, False), (2047, 8, 904, False),
                       (2048, 1, 905, False), (2040, 8, 906, False), (2049, 8, 907, False), (100, 1, 908, False),
                       (5, 8, 909, False), (1, 1, 910, False), (4000, 1, 911, True), (4000, 8, 912, True))

_KS1 = {"mod": None, "cnt": {}, "build_s": 0.0}

def _ks1_build():
    _pt1_wait("ks1")
    if _KS1["mod"] is not None:
        return _KS1["mod"]
    from torch.utils.cpp_extension import load_inline
    insts = " ".join(f"KS1_X({l}, {n}, {g})" for l, n, g in KS1_INSTS)
    minsts = " ".join(f"KS1_XM({l}, {n}, {g}, {m})" for l, n, g, m in VSL_KS1_INSTS) if VSL_ENABLE else ""
    m3insts = " ".join(f"KS3_XM({n}, {m})" for n, m in VSL_KS3_INSTS) if VSL_ENABLE else ""
    cuda = (KS1_CUDA_SRC.replace("@INSTS@", insts).replace("@MINSTS@", minsts).replace("@M3INSTS@", m3insts)
            + _dg1_fold("cuda"))
    tag = hashlib.sha1((KS1_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    _KS1["mod"] = load_inline(name=f"ks1_{tag}", cpp_sources=KS1_CPP_SRC + _dg1_fold("cpp"), cuda_sources=cuda,
                              functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info", "ax3_forward", "ax3_info"] + _dg1_fold("fns"),
                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                              verbose=False)
    _KS1["build_s"] = time.time() - t0
    _KS1["dg1_insts"] = _dg1_insts() if _dg1_fold_on() else None
    return _KS1["mod"]

def _ks1_counters(dev):
    key = str(dev)
    if key not in _KS1["cnt"]:
        _KS1["cnt"][key] = torch.zeros(1024, dtype=torch.int32, device=dev)
    return _KS1["cnt"][key]

def _ks1_call(qkv, data, cos_sin, geom=None, trig=None, pdl=None, sync=None, phalf=None, fold=None):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    ns, ntg = geom if geom is not None else (KS1_GEOM[L] if B == 1 else KS2_GEOM[(B, L)])
    if trig is None:
        trig = KS1_TRIG.get(L, 0) if B == 1 else KS2_TRIG.get((B, L), 0)
    if fold is None:
        fold = False if B == 1 else KS2_FOLD.get((B, L), False)
    pdl = KS1_PDL if pdl is None else pdl
    sync = KS1_SYNC.get(L, 1) if sync is None else sync
    phalf = KS1_PHALF.get(L, False) if phalf is None else phalf
    dev = qkv.device
    rows = 8 * (L // ntg)
    po = torch.empty((B * 2 * ntg * (ns + 1) * rows * 64,), dtype=torch.float32, device=dev)
    pml = torch.empty((B * 2 * ntg * (ns + 1) * rows * 2,), dtype=torch.float32, device=dev)
    out = torch.empty((B * L, 16 * 64), dtype=torch.bfloat16, device=dev)
    prof = KS1_PROF_BUF if KS1_PROF_BUF is not None else torch.empty((0,), dtype=torch.int64, device=dev)
    _KS1["mod"].ks1_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                            _ks1_counters(dev), out, prof, L, ns, ntg, float(cfg["sm_scale"] * cfg["qk_scale_factor"]),
                            KS1_EPS, int(trig), 1 if pdl else 0, int(sync), 1 if phalf else 0, 1 if fold else 0)
    return out

def _ks1_wanted(qkv, data, cfg, state):
    if not (KS1_ENABLE and state.get("ks1") and _KS1["mod"] is not None):
        return False
    B, L = cfg["batch"], cfg["q_len"]
    if B == 1:
        geo, passed = (KS1_GEOM.get(L) if KS1_ROUTE.get(L) else None), state.get("ks1") or set()
    else:
        geo, passed = (KS2_GEOM.get((B, L)) if (KS2_ENABLE and KS2_ROUTE.get((B, L))) else None), state.get("ks2") or set()
        if _ks3_routed((B, L), state):
            geo, passed = (0, 0), {(L, 0, 0)}
    if not (((geo is not None and (L, *geo) in passed) or _vsl_pick(cfg, state) is not None or _ax3_pick(cfg, state) is not None) and cfg["window"] == 2048
            and cfg["rope_theta"] == ROPE_THETA
            and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2):
        return False
    cs = state.get("cos_sin")
    return (cs is not None and cs.dtype == torch.float32 and cs.is_contiguous() and cs.dim() == 2 and cs.shape[1] == 64
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.shape[0] == B * L
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2 and data["page_table"].is_contiguous()
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].numel() == B
            and data["cache_loc"].dtype == torch.int64 and data["cache_loc"].is_contiguous()
            and data["positions"].dtype == torch.int64 and data["positions"].is_contiguous())

def _ks3_routed(key, state):
    return bool(KS3_ENABLE and key[1] == 1 and KS3_ROUTE.get(key) and KS3_NS.get(key) in (state.get("ks3") or set()))

def _ks3_call(qkv, data, cos_sin, ns=None, trig=None, pdl=None):
    cfg = data["config"]
    B = cfg["batch"]
    ns = KS3_NS[(B, 1)] if ns is None else ns
    trig = KS3_TRIG.get((B, 1), 0) if trig is None else trig
    pdl = KS1_PDL if pdl is None else pdl
    dev = qkv.device
    po = torch.empty((B * ns * 16 * 64,), dtype=torch.float32, device=dev)
    pml = torch.empty((B * ns * 16 * 2,), dtype=torch.float32, device=dev)
    out = torch.empty((B, 16 * 64), dtype=torch.bfloat16, device=dev)
    prof = KS1_PROF_BUF if KS1_PROF_BUF is not None else torch.empty((0,), dtype=torch.int64, device=dev)
    _KS1["mod"].ks3_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                            _ks1_counters(dev), out, prof, int(ns), float(cfg["sm_scale"] * cfg["qk_scale_factor"]), KS1_EPS,
                            int(trig), 1 if pdl else 0, 1 if cfg["rope_theta"] else 0)
    return out

def _ks3_self_test(cos_sin, device, verbose=False):
    cnt = _ks1_counters(device)
    bad, seen = set(), set()
    for batch, ctx, seed, row0 in KS3_SELF_TEST_CASES:
        ref = _k1fa4_reference(_k1fa4_case(batch, ctx, 1, 2048, seed, row0, device)).view(batch, -1)
        d1 = _k1fa4_case(batch, ctx, 1, 2048, seed, row0, device)
        o1 = _k1fa4_r1_path(d1).float().view(batch, -1)
        torch.cuda.synchronize(device)
        e1 = float((o1 - ref).norm() / ref.norm())
        rows = d1["cache_loc"]
        k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
        v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
        for ns in KS3_INSTS:
            if ns in bad:
                continue
            dk = _k1fa4_case(batch, ctx, 1, 2048, seed, row0, device)
            qkv = dk["qkv"].view(batch, -1)
            o_a = _ks3_call(qkv, dk, cos_sin, ns, 0).clone()
            o_b = _ks3_call(qkv, dk, cos_sin, ns, 1)
            torch.cuda.synchronize(device)
            ek = float((o_a.float() - ref).norm() / ref.norm())
            kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
            vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
            nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
            tick = int(cnt.abs().sum())
            same = bool(torch.equal(o_a, o_b))
            ok = bool(torch.isfinite(o_a).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and tick == 0 and same
            seen.add(ns)
            if not ok:
                bad.add(ns)
            if verbose or not ok:
                print(f"[ks3] self-test B{batch} NS{ns} ctx{ctx}{' row0' if row0 else ''}: {'PASS' if ok else 'FAIL'} err {ek:.4e} "
                      f"(r1 {e1:.4e}) cache bytes != r1 {nbad} tickets {tick} repeat identical {same}", flush=True)
            del dk, o_a, o_b
    ok_ns = {n for n in seen if n not in bad}
    ns = KS3_NS.get((32, 1))
    if ns in ok_ns:
        d = _k1fa4_case(32, 3000, 1, 2048, 970, False, device)
        qkv = d["qkv"].view(32, -1)
        o1 = _ks3_call(qkv, d, cos_sin, ns).clone()
        torch.cuda.synchronize(device)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            og = _ks3_call(qkv, d, cos_sin, ns)
        gr.replay()
        gr.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1)) and int(cnt.abs().sum()) == 0
        print(f"[ks3] graph check B32 NS{ns}: replay identical {same}", flush=True)
        del gr, og
        if not same:
            ok_ns.discard(ns)
    print(f"[ks3] NS ok: {sorted(ok_ns)} (failed {sorted(bad)})", flush=True)
    return ok_ns

AX3_ENABLE = True

AX3_INSTS = ((8, 2), (8, 10), (16, 10), (8, 0))

AX3_ROUTE = {(32, 1): (8, 10)}

AX3_TRIG = {(16, 1): 1, (32, 1): 1}

AX3_SELF_TEST_CASES = ((32, 2600, 961, True), (16, 3000, 962, False), (3, 2100, 963, False), (5, 100, 965, False),
                       (4, 1, 966, False), (2, 2048, 967, False), (2, 2049, 968, True), (32, 2047, 969, False),
                       (16, 80000, 964, False))

AX3_PREPARE_BY_S = 245.0

AX3_SWEEP = True

_AX3 = {"ws": {}, "seen": set()}


def _ax3_ws(dev):
    key = str(dev)
    if key not in _AX3["ws"]:
        _AX3["ws"][key] = (torch.zeros(512 * 16 * 64, dtype=torch.float32, device=dev),
                           torch.zeros(512 * 16, dtype=torch.float32, device=dev),
                           torch.zeros(512, dtype=torch.int32, device=dev))
    return _AX3["ws"][key]


def _ax3_pick(cfg, state):
    if not AX3_ENABLE or cfg["q_len"] != 1 or cfg["batch"] > 512:
        return None
    v = AX3_ROUTE.get((cfg["batch"], 1))
    if v is None:
        return None
    v = tuple(v)
    return v if v in (state.get("ax3") or ()) else None


def _ax3_call(qkv, data, cos_sin, v, trig=None, pdl=None, prof=None):
    cfg = data["config"]
    B = cfg["batch"]
    ns, var = v
    dev = qkv.device
    wo, wl, cnt = _ax3_ws(dev)
    out = torch.empty((B, 16 * 64), dtype=torch.bfloat16, device=dev)
    if prof is None:
        prof = torch.empty((0,), dtype=torch.int64, device=dev)
    trig = AX3_TRIG.get((B, 1), 1) if trig is None else trig
    pdl = KS1_PDL if pdl is None else pdl
    _KS1["mod"].ax3_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, wo, wl, cnt, out,
                            prof, int(ns), int(var), float(cfg["sm_scale"] * cfg["qk_scale_factor"]), KS1_EPS, int(trig),
                            1 if pdl else 0, 1 if cfg["rope_theta"] else 0)
    if v not in _AX3["seen"]:
        _AX3["seen"].add(v)
        print(f"[ax3] first call {v} B={B}", flush=True)
    return out


def _ax3_self_test(cos_sin, device):
    wo, wl, cnt = _ax3_ws(device)
    ok = set()
    for v in AX3_INSTS:
        v = tuple(v)
        good, worst = True, 0.0
        cnt.zero_()
        wo.zero_()
        wl.zero_()
        try:
            for B, ctx, seed, row0 in AX3_SELF_TEST_CASES:
                ref = _k1fa4_reference(_k1fa4_case(B, ctx, 1, 2048, seed, row0, device)).view(B, -1)
                d1 = _k1fa4_case(B, ctx, 1, 2048, seed, row0, device)
                o1 = _k1fa4_r1_path(d1).float().view(B, -1)
                torch.cuda.synchronize(device)
                e1 = float((o1 - ref).norm() / ref.norm())
                rows = d1["cache_loc"]
                k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                dk = _k1fa4_case(B, ctx, 1, 2048, seed, row0, device)
                qkv = dk["qkv"].view(B, -1)
                o_a = _ax3_call(qkv, dk, cos_sin, v).clone()
                o_b = _ax3_call(qkv, dk, cos_sin, v).clone()
                torch.cuda.synchronize(device)
                ek = float((o_a.float() - ref).norm() / ref.norm())
                kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
                clean = int(cnt.abs().sum()) == 0 and float(wo.abs().sum()) == 0.0 and float(wl.abs().sum()) == 0.0
                rep = float((o_a.float() - o_b.float()).abs().max()) <= 0.02 * float(o_a.float().abs().max()) + 1e-3
                c_ok = bool(torch.isfinite(o_a.float()).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and clean and rep
                worst = max(worst, ek)
                if not c_ok:
                    good = False
                    print(f"[ax3] self-test {v} B{B} ctx{ctx}{' row0' if row0 else ''}: FAIL err {ek:.4e} (r1 {e1:.4e}) "
                          f"cache bytes != r1 {nbad} clean {clean} repeat {rep}", flush=True)
                    cnt.zero_()
                    wo.zero_()
                    wl.zero_()
                del dk, o_a, o_b, d1, o1, ref
                if not good:
                    break
            if good:
                d = _k1fa4_case(32, 3000, 1, 2048, 977, False, device)
                qkv = d["qkv"].view(32, -1)
                o1 = _ax3_call(qkv, d, cos_sin, v).clone()
                torch.cuda.synchronize(device)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    og = _ax3_call(qkv, d, cos_sin, v)
                gr.replay()
                gr.replay()
                torch.cuda.synchronize(device)
                g_ok = (float((og.float() - o1.float()).abs().max()) <= 0.02 * float(o1.float().abs().max()) + 1e-3
                        and int(cnt.abs().sum()) == 0 and float(wo.abs().sum()) == 0.0)
                del gr, og, d
                if not g_ok:
                    good = False
                    print(f"[ax3] graph check {v}: FAIL", flush=True)
        except Exception as ex:  # noqa: BLE001
            good = False
            print(f"[ax3] self-test {v} raised {repr(ex)[-600:]}", flush=True)
            cnt.zero_()
            wo.zero_()
            wl.zero_()
        print(f"[ax3] variant {v}: {'ok' if good else 'FAILED'} (worst err {worst:.3e})", flush=True)
        if good:
            ok.add(v)
    return ok


def _ax3_prepare(state, device, t0):
    state["ax3"] = set()
    if not AX3_ENABLE or _KS1["mod"] is None or not hasattr(_KS1["mod"], "ax3_forward"):
        return
    if time.time() - t0 > AX3_PREPARE_BY_S:
        print(f"[ax3] skipped: prepare already took {time.time() - t0:.1f}s: KS3 stays", flush=True)
        return
    t1 = time.time()
    try:
        for ns, var in AX3_INSTS:
            print(f"[ax3] NS{ns} VAR{var}: regs / local / smem / CTAs per SM {list(_KS1['mod'].ax3_info(ns, var))}", flush=True)
        state["ax3"] = _ax3_self_test(state["cos_sin"], device)
    except Exception as ex:  # noqa: BLE001
        print(f"[ax3] self-test FAILED {repr(ex)[-1000:]}: KS3 stays", flush=True)
        state["ax3"] = set()
    live = {k: tuple(v) for k, v in AX3_ROUTE.items() if tuple(v) in state["ax3"]}
    print(f"[ax3] route live {live} ({time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
    if AX3_SWEEP:
        try:
            _ax3_sweep(state, device)
        except Exception as ex:  # noqa: BLE001
            print(f"[ax3] sweep FAILED {repr(ex)[-1000:]}", flush=True)
        torch.cuda.synchronize(device)


AX3_SWEEP_REPS = 40


def _ax3_board_case(B, seed, device):
    # board-like decode sliding input: ctx 80000 +- 7.5 % (agentic), page 1, scattered rows (as the reference builds them)
    g = torch.Generator(device=device).manual_seed(seed)
    ctx = 80000
    seq = (ctx * (1 + 0.075 * (2 * torch.rand((B,), generator=g, device=device) - 1))).long().clamp(1, 86000)
    max_pages = 86000 + 1
    total = B * max_pages
    perm = torch.randperm(total, generator=g, device=device).to(torch.int32)
    pt = torch.full((B, max_pages), -1, dtype=torch.int32, device=device)
    off = 0
    for b in range(B):
        n = int(seq[b]) + 1
        pt[b, :n] = perm[off:off + n]
        off += n
    f8 = torch.float8_e4m3fn
    k_cache = torch.empty((total, 1, 2, 64), dtype=f8, device=device)
    v_cache = torch.empty((total, 1, 2, 64), dtype=f8, device=device)
    for c in (k_cache, v_cache):
        cv = c.view(-1, 128)
        for i in range(0, cv.shape[0], 1 << 20):
            cv[i:i + (1 << 20)] = torch.randn((min(1 << 20, cv.shape[0] - i), 128), generator=g, device=device).to(f8)
    q = (2 * torch.randn((B, 1, 16, 64), generator=g, device=device)).to(torch.bfloat16)
    k = (2 * torch.randn((B, 1, 2, 64), generator=g, device=device)).to(torch.bfloat16)
    v = torch.randn((B, 1, 2, 64), generator=g, device=device).to(torch.bfloat16)
    gate = torch.randn((B, 1, 1024), generator=g, device=device).to(torch.bfloat16)
    qkv = torch.cat([q.flatten(2), k.flatten(2), v.flatten(2), gate], -1)
    positions = seq.clone()
    cache_loc = torch.gather(pt.long(), 1, positions.view(B, 1)).reshape(-1)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    return {"qkv": qkv, "positions": positions, "cache_loc": cache_loc, "cache_seqlens": (seq + 1).to(torch.int32),
            "k_cache": k_cache, "v_cache": v_cache, "k_scale": one, "v_scale": one.clone(), "page_table": pt,
            "seq_lens": seq.to(torch.int32),
            "config": {"batch": B, "q_len": 1, "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64, "page_size": 1,
                       "window": 2048, "rope_theta": 500000.0, "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125,
                       "kv_scale": 1.0}}


def _ax3_sweep(state, device):
    import statistics
    cs = state["cos_sin"]
    flush = torch.ones((64 << 20,), dtype=torch.float32, device=device)   # 256 MB, read before every timed replay
    sink = torch.empty((), dtype=torch.float32, device=device)
    stream = torch.cuda.Stream(device=device)
    cyc = int(300 * 1980)
    for _ in range(3):   # calibrate a ~300 us spin gate (as the judge's)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        torch.cuda._sleep(cyc)
        b.record()
        b.synchronize()
        cyc = int(cyc * 300.0 / max(a.elapsed_time(b) * 1e3, 1.0))

    def capture(fn):
        with torch.cuda.stream(stream):
            fn()
            torch.cuda.synchronize(device)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=stream):
                out = fn()
        torch.cuda.synchronize(device)
        return gr, out

    tiny = torch.zeros(1, device=device)
    g_empty, _ = capture(lambda: tiny.add_(1.0))
    for B in (16, 32):
        d = _ax3_board_case(B, 7100 + B, device)
        qkv = d["qkv"].view(B, -1)
        arms = []
        if B == 16 and ("ks3", 16, 1) in (state.get("vsl") or ()):
            arms.append(("KS3 ship NS16 MG1", lambda: _vsl_call(qkv, d, cs, ("ks3", 16, 1)), None))
        elif _ks3_routed((B, 1), state):
            arms.append((f"KS3 ship NS{KS3_NS.get((B, 1))} MG0", lambda: _ks3_call(qkv, d, cs), None))
        for vv in (state.get("vsl") or ()):
            if vv[0] == "ks3" and not (B == 16 and tuple(vv) == ("ks3", 16, 1)):
                arms.append((f"KS3 NS{vv[1]} MG{vv[2]}", lambda vv=vv: _vsl_call(qkv, d, cs, vv), None))
        for v in AX3_INSTS:
            v = tuple(v)
            if v in state["ax3"]:
                arms.append((f"AX3 NS{v[0]} VAR{v[1]}", lambda v=v: _ax3_call(qkv, d, cs, v), v))
        graphs = []
        for name, fn, v in arms:
            gr, out = capture(fn)
            graphs.append((name, gr, out, v))   # (the graph's output buffer: read after the replays)
        ts = {name: [] for name, _, _, _ in graphs}
        te = []
        for _ in range(AX3_SWEEP_REPS):
            for name, gr, _, _ in graphs + [("empty", g_empty, None, None)]:
                with torch.cuda.stream(stream):
                    torch.sum(flush, dim=0, out=sink)
                    torch.cuda._sleep(cyc)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    gr.replay()
                    e1.record()
                e1.synchronize()
                (te if name == "empty" else ts[name]).append(e0.elapsed_time(e1) * 1e3)
        ovh = statistics.median(te)
        ref = graphs[0][2].float()
        base = statistics.median(ts[graphs[0][0]])
        print(f"[ax3] sweep B{B}: replay overhead {ovh:.2f} us; raw median us (net = raw - overhead):", flush=True)
        for name, gr, out, v in graphs:
            med = statistics.median(ts[name])
            err = float((out.float() - ref).norm() / ref.norm())
            line = f"[ax3]   {name:18s} raw {med:6.2f} net {med - ovh:6.2f} (vs first {med - base:+.2f}) err vs first {err:.2e}"
            if v is not None:
                ns = v[0]
                prof = torch.zeros(8 * B * ns, dtype=torch.int64, device=device)
                g2, _ = capture(lambda v=v, prof=prof: _ax3_call(qkv, d, cs, v, prof=prof))
                reps = []
                for _ in range(8):
                    prof.zero_()
                    with torch.cuda.stream(stream):
                        torch.sum(flush, dim=0, out=sink)
                        torch.cuda._sleep(cyc)
                        g2.replay()
                    torch.cuda.synchronize(device)
                    pm = prof.view(-1, 8)[:, :7].double().cpu()
                    t0 = pm[:, 0].min()
                    reps.append(torch.where(pm > 0, (pm - t0) / 1000.0, torch.full_like(pm, float("nan"))))
                st = torch.stack(reps)
                med_m = st.nanmedian(dim=1).values.median(dim=0).values
                mx = torch.stack([torch.tensor([float(r[:, k][~torch.isnan(r[:, k])].max()) if (~torch.isnan(r[:, k])).any()
                                                else float("nan") for k in range(7)]) for r in reps]).median(dim=0).values
                line += " | marks " + " ".join(f"{k}:{med_m[k]:.2f}/{mx[k]:.2f}" for k in range(7))
                del g2, prof
            print(line, flush=True)
        del graphs, d, qkv
        torch.cuda.synchronize(device)
    del flush


def _ks1_run(qkv, data, state):
    cfg = data["config"]
    a3 = _ax3_pick(cfg, state) if (qkv.stride(0) % 8 == 0 and qkv.data_ptr() % 16 == 0) else None
    if a3 is not None:
        return _ax3_call(qkv, data, state["cos_sin"], a3)
    v = _vsl_pick(cfg, state)
    if v is not None:
        return _vsl_call(qkv, data, state["cos_sin"], v)
    if cfg["batch"] > 1 and _ks3_routed((cfg["batch"], cfg["q_len"]), state):
        return _ks3_call(qkv, data, state["cos_sin"])
    return _ks1_call(qkv, data, state["cos_sin"])

def _vsl_ws(dev):
    key = str(dev)
    if key not in _VSL["ws"]:
        _VSL["ws"][key] = (torch.zeros(VSL_WS_FLOATS, dtype=torch.float32, device=dev),
                           torch.zeros(VSL_WS_FLOATS // 8, dtype=torch.float32, device=dev))
    return _VSL["ws"][key]

def _vsl_pick(cfg, state):
    if not VSL_ENABLE:
        return None
    v = VSL_ROUTE.get((cfg["batch"], cfg["q_len"]))
    if v is None:
        return None
    v = tuple(v)
    return v if v in (state.get("vsl") or ()) else None

def _vsl_call(qkv, data, cos_sin, v):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    dev = qkv.device
    po, pml = _vsl_ws(dev)
    prof = KS1_PROF_BUF if KS1_PROF_BUF is not None else torch.empty((0,), dtype=torch.int64, device=dev)
    qk = float(cfg["sm_scale"] * cfg["qk_scale_factor"])
    if v[0] == "ks1":
        _, l, ns, ntg, mg = v
        out = torch.empty((B * L, 16 * 64), dtype=torch.bfloat16, device=dev)
        _KS1["mod"].ks1_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                                data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                                _ks1_counters(dev), out, prof, L, ns, ntg, qk, KS1_EPS, int(KS1_TRIG.get(L, 0)),
                                1 if KS1_PDL else 0, 2 + mg, 0, 1 if (mg & 3) == 2 else 0)
    else:
        _, ns, mg = v
        out = torch.empty((B, 16 * 64), dtype=torch.bfloat16, device=dev)
        _KS1["mod"].ks3_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                                data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                                _ks1_counters(dev), out, prof, int(ns + 100 * mg), qk, KS1_EPS,
                                int(KS3_TRIG.get((B, 1), 1)), 1 if KS1_PDL else 0, 1 if cfg["rope_theta"] else 0)
    if v not in _VSL["seen"]:
        _VSL["seen"].add(v)
        print(f"[vsl] first call {v} B={B} L={L}", flush=True)
    return out

def _vsl_variants():
    return ([("ks1", l, n, g, m) for l, n, g, m in VSL_KS1_INSTS] + [("ks3", n, m) for n, m in VSL_KS3_INSTS])

def _vsl_cases(v):
    if v[0] == "ks1":
        return [(1, ctx, length, seed + 50, row0) for ctx, length, seed, row0 in KS1_SELF_TEST_CASES if length == v[1]]
    return ([(b, ctx, 1, seed + 50, row0) for b, ctx, seed, row0 in KS3_SELF_TEST_CASES]
            + [(1, 3000, 1, 991, False), (1, 2047, 1, 992, False), (1, 100, 1, 993, False), (1, 1, 1, 994, False),
               (1, 4000, 1, 995, True), (16, 2049, 1, 996, False)])

def _vsl_self_test(cos_sin, device):
    cnt = _ks1_counters(device)
    po, pml = _vsl_ws(device)
    ok = set()
    for v in _vsl_variants():
        good, worst, nbit, nb = True, 0.0, 0, 0
        cnt.zero_()
        po.zero_()
        pml.zero_()
        try:
            for B, ctx, L, seed, row0 in _vsl_cases(v):
                T = B * L
                ref = _k1fa4_reference(_k1fa4_case(B, ctx, L, 2048, seed, row0, device)).view(T, -1)
                d1 = _k1fa4_case(B, ctx, L, 2048, seed, row0, device)
                o1 = _k1fa4_r1_path(d1).float().view(T, -1)
                torch.cuda.synchronize(device)
                e1 = float((o1 - ref).norm() / ref.norm())
                rows = d1["cache_loc"]
                k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                dk = _k1fa4_case(B, ctx, L, 2048, seed, row0, device)
                qkv = dk["qkv"].view(T, -1)
                o_a = _vsl_call(qkv, dk, cos_sin, v).clone()
                o_b = _vsl_call(qkv, dk, cos_sin, v).clone()
                torch.cuda.synchronize(device)
                ek = float((o_a.float() - ref).norm() / ref.norm())
                kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
                clean = int(cnt.abs().sum()) == 0 and float(po.abs().sum()) == 0.0 and float(pml.abs().sum()) == 0.0
                if (v[-1] & 3) == 2:
                    rep = bool(torch.equal(o_a, o_b))
                else:
                    rep = float((o_a.float() - o_b.float()).abs().max()) <= 0.02 * float(o_a.float().abs().max()) + 1e-3
                if v[0] == "ks3" and v[-1] == 2:
                    ds = _k1fa4_case(B, ctx, L, 2048, seed, row0, device)
                    o_s = _ks3_call(ds["qkv"].view(T, -1), ds, cos_sin, v[1], None, None)
                    torch.cuda.synchronize(device)
                    nb += 1
                    nbit += int(torch.equal(o_s, o_a))
                    del ds, o_s
                if v[0] == "ks1" and (v[-1] & 4):
                    dm = _k1fa4_case(B, ctx, L, 2048, seed, row0, device)
                    pt = dm["page_table"]
                    dm["page_table"] = torch.cat([pt, torch.full((B, 37), -1, dtype=pt.dtype, device=device)], 1).contiguous()
                    o_m = _vsl_call(dm["qkv"].view(T, -1), dm, cos_sin, v)
                    torch.cuda.synchronize(device)
                    em = float((o_m.float() - ref).norm() / ref.norm())
                    km = dm["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                    vm = dm["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                    nbad += int((km != k1b).sum()) + int((vm != v1b).sum())
                    ek = max(ek, em)
                    clean = clean and int(cnt.abs().sum()) == 0 and float(po.abs().sum()) == 0.0
                    del dm, o_m
                c_ok = (bool(torch.isfinite(o_a).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and clean and rep)
                worst = max(worst, ek)
                if not c_ok:
                    good = False
                    print(f"[vsl] self-test {v} B{B} ctx{ctx} L{L}{' row0' if row0 else ''}: FAIL err {ek:.4e} (r1 {e1:.4e}) "
                          f"cache bytes != r1 {nbad} clean {clean} repeat {rep}", flush=True)
                    cnt.zero_()
                    po.zero_()
                    pml.zero_()
                del dk, o_a, o_b, d1, o1, ref
                if not good:
                    break
            if good:
                B, ctx, L = (1, 3000, v[1]) if v[0] == "ks1" else (32, 3000, 1)
                d = _k1fa4_case(B, ctx, L, 2048, 977, False, device)
                qkv = d["qkv"].view(B * L, -1)
                o1 = _vsl_call(qkv, d, cos_sin, v).clone()
                torch.cuda.synchronize(device)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    og = _vsl_call(qkv, d, cos_sin, v)
                gr.replay()
                gr.replay()
                torch.cuda.synchronize(device)
                same = (bool(torch.equal(og, o1)) if (v[-1] & 3) == 2 else
                        float((og.float() - o1.float()).abs().max()) <= 0.02 * float(o1.float().abs().max()) + 1e-3)
                clean = int(cnt.abs().sum()) == 0 and float(po.abs().sum()) == 0.0
                if not (same and clean):
                    good = False
                    print(f"[vsl] graph check {v}: FAIL same {same} clean {clean}", flush=True)
                del gr, og, d
        except Exception as ex:
            good = False
            print(f"[vsl] self-test {v} raised {repr(ex)[-600:]}", flush=True)
            torch.cuda.synchronize(device)
            cnt.zero_()
            po.zero_()
            pml.zero_()
        if good:
            ok.add(v)
        bits = f", bitwise = KS3 {nbit}/{nb}" if nb else ""
        print(f"[vsl] variant {v}: {'ok' if good else 'FAILED'} (worst err {worst:.3e}{bits})", flush=True)
    return ok

def _vsl_prepare(state, device, t0):
    state["vsl"] = set()
    if not (VSL_ENABLE and _KS1["mod"] is not None):
        return
    if time.time() - t0 > VSL_PREPARE_BY_S:
        print(f"[vsl] skipped: prepare already took {time.time() - t0:.1f}s: KS1 / KS3 stay", flush=True)
        return
    t1 = time.time()
    try:
        mod = _KS1["mod"]
        for l, n, g, m in VSL_KS1_INSTS:
            print(f"[vsl] ks1 L{l} NS{n} NTG{g} MG{m}: regs / local / smem / CTAs per SM / clusters {list(mod.ks1_info(l, n, g + 100 * m))}",
                  flush=True)
        for n, m in VSL_KS3_INSTS:
            print(f"[vsl] ks3 NS{n} MG{m}: regs / local / smem / CTAs per SM / clusters {list(mod.ks3_info(n + 100 * m))}", flush=True)
        state["vsl"] = _vsl_self_test(state["cos_sin"], device)
    except Exception as ex:
        print(f"[vsl] prepare FAILED {repr(ex)[-1500:]}: KS1 / KS3 stay", flush=True)
        state["vsl"] = set()
    live = {k: tuple(v) for k, v in VSL_ROUTE.items() if tuple(v) in state["vsl"]}
    print(f"[vsl] variants ok {sorted(state['vsl'])}; route {live or 'none (KS1 / KS3 / FA4 as shipped)'} "
          f"({time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)

def _ks1_case(ctx, length, seed, row0, device):
    return _k1fa4_case(1, ctx, length, 2048, seed, row0, device)

def _ks1_self_test(cos_sin, device, geoms, verbose=True):
    cnt = _ks1_counters(device)
    bad = set()
    worst = {}
    for ctx, length, seed, row0 in KS1_SELF_TEST_CASES:
        mine = [(l, n, g) for l, n, g in geoms if l == length and (l, n, g) not in bad]
        if not mine:
            continue
        ref = _k1fa4_reference(_ks1_case(ctx, length, seed, row0, device)).view(length, -1)
        d1 = _ks1_case(ctx, length, seed, row0, device)
        o1 = _k1fa4_r1_path(d1).float().view(length, -1)
        torch.cuda.synchronize(device)
        e1 = float((o1 - ref).norm() / ref.norm())
        rows = d1["cache_loc"]
        k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
        v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
        for geo in mine:
            dk = _ks1_case(ctx, length, seed, row0, device)
            qkv = dk["qkv"].view(length, -1)
            o_a = _ks1_call(qkv, dk, cos_sin, geo[1:], 0, KS1_PDL).clone()
            o_b = _ks1_call(qkv, dk, cos_sin, geo[1:], 1, KS1_PDL, 0)
            o_c = _ks1_call(qkv, dk, cos_sin, geo[1:], 0, KS1_PDL, 2, not KS1_PHALF.get(length, False))
            torch.cuda.synchronize(device)
            ek = float((o_a.float() - ref).norm() / ref.norm())
            ec = float((o_c.float() - ref).norm() / ref.norm())
            kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
            vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
            nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
            tick = int(cnt.abs().sum())
            same = bool(torch.equal(o_a, o_b))
            ok = (bool(torch.isfinite(o_a).all()) and bool(torch.isfinite(o_c).all()) and max(ek, ec) <= 1.25 * e1 + 5e-3
                  and nbad == 0 and tick == 0 and same)
            worst[geo] = max(worst.get(geo, (0.0, 0.0)), (ek, e1))
            if not ok:
                bad.add(geo)
            if verbose or not ok:
                print(f"[ks1] self-test L{length} NS{geo[1]} NTG{geo[2]} ctx{ctx}{' row0' if row0 else ''}: {'PASS' if ok else 'FAIL'} "
                      f"err {ek:.4e} / other partials {ec:.4e} (r1 {e1:.4e}) cache bytes != r1 {nbad} tickets {tick} repeat identical {same}", flush=True)
            del dk, o_a, o_b, o_c
    ok_geo = {geo for geo in geoms if geo not in bad and geo in worst}
    for geo in sorted(set(geoms)):
        w = worst.get(geo, (float("nan"), float("nan")))
        print(f"[ks1] geometry L{geo[0]} NS{geo[1]} NTG{geo[2]}: {'ok' if geo in ok_geo else 'FAILED'} (worst err {w[0]:.3e} vs r1 "
              f"{w[1]:.3e})", flush=True)
    for length in sorted({l for l, _, _ in ok_geo}):
        geo = KS1_GEOM.get(length)
        if geo is None or (length, *geo) not in ok_geo:
            continue
        d = _ks1_case(3000, length, 920 + length, False, device)
        qkv = d["qkv"].view(length, -1)
        o1 = _ks1_call(qkv, d, cos_sin, geo).clone()
        torch.cuda.synchronize(device)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            og = _ks1_call(qkv, d, cos_sin, geo)
        gr.replay()
        gr.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1)) and int(cnt.abs().sum()) == 0
        print(f"[ks1] graph check L{length} NS{geo[0]} NTG{geo[1]}: replay identical {same}", flush=True)
        del gr, og
        if not same:
            ok_geo = {x for x in ok_geo if x[0] != length}
    return ok_geo

def _ks2_self_test(cos_sin, device, geoms, verbose=False):
    cnt = _ks1_counters(device)
    bad, seen = set(), set()
    for batch, ctx, length, seed, row0 in KS2_SELF_TEST_CASES:
        mine = [g for g in geoms if g[0] == length and g not in bad]
        if not mine:
            continue
        T = batch * length
        ref = _k1fa4_reference(_k1fa4_case(batch, ctx, length, 2048, seed, row0, device)).view(T, -1)
        d1 = _k1fa4_case(batch, ctx, length, 2048, seed, row0, device)
        o1 = _k1fa4_r1_path(d1).float().view(T, -1)
        torch.cuda.synchronize(device)
        e1 = float((o1 - ref).norm() / ref.norm())
        rows = d1["cache_loc"]
        k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
        v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
        for geo in mine:
            dk = _k1fa4_case(batch, ctx, length, 2048, seed, row0, device)
            qkv = dk["qkv"].view(T, -1)
            o_a = _ks1_call(qkv, dk, cos_sin, geo[1:], 0, KS1_PDL).clone()
            o_b = _ks1_call(qkv, dk, cos_sin, geo[1:], 1, KS1_PDL, 0)
            o_c = _ks1_call(qkv, dk, cos_sin, geo[1:], 0, KS1_PDL, fold=not KS2_FOLD.get((batch, length), False)).clone()
            torch.cuda.synchronize(device)
            ek = max(float((o_a.float() - ref).norm() / ref.norm()), float((o_c.float() - ref).norm() / ref.norm()))
            kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
            vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
            nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
            tick = int(cnt.abs().sum())
            same = bool(torch.equal(o_a, o_b))
            ok = bool(torch.isfinite(o_a).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and tick == 0 and same
            seen.add(geo)
            if not ok:
                bad.add(geo)
            if verbose or not ok:
                print(f"[ks2] self-test B{batch} L{length} NS{geo[1]} NTG{geo[2]} ctx{ctx}{' row0' if row0 else ''}: "
                      f"{'PASS' if ok else 'FAIL'} err {ek:.4e} (r1 {e1:.4e}) cache bytes != r1 {nbad} tickets {tick} "
                      f"repeat identical {same}", flush=True)
            del dk, o_a, o_b, o_c
    ok_geo = {g for g in seen if g not in bad}
    for length in sorted({g[0] for g in ok_geo}):
        geo = KS2_GEOM.get((16, length))
        if geo is None or (length, *geo) not in ok_geo:
            continue
        d = _k1fa4_case(16, 3000, length, 2048, 950 + length, False, device)
        qkv = d["qkv"].view(16 * length, -1)
        o1 = _ks1_call(qkv, d, cos_sin, geo).clone()
        torch.cuda.synchronize(device)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            og = _ks1_call(qkv, d, cos_sin, geo)
        gr.replay()
        gr.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1)) and int(cnt.abs().sum()) == 0
        print(f"[ks2] graph check B16 L{length} NS{geo[0]} NTG{geo[1]}: replay identical {same}", flush=True)
        del gr, og
        if not same:
            ok_geo = {x for x in ok_geo if x[0] != length}
    print(f"[ks2] geometries ok on B > 1: {sorted(ok_geo)} (failed {sorted(bad)})", flush=True)
    return ok_geo

def _ks1_prepare(state, t0):
    state["ks1"] = set()
    state["ks2"] = set()
    state["ks3"] = set()
    print(f"[ks1] levers: KS1_ENABLE={KS1_ENABLE} route {KS1_ROUTE} geom {KS1_GEOM} trig {KS1_TRIG} pdl {KS1_PDL} insts {KS1_INSTS}",
          flush=True)
    if not KS1_ENABLE:
        return
    if time.time() - t0 > KS1_PREPARE_BY_S:
        print(f"[ks1] skipped: prepare already took {time.time() - t0:.1f}s (> {KS1_PREPARE_BY_S}s): FA4 v8 stays", flush=True)
        return
    device = state["router"].device
    t1 = time.time()
    try:
        mod = _ks1_build()
        for l, n, g in KS1_INSTS:
            regs, local, smem_b, occ = (int(x) for x in mod.ks1_info(l, n, g))
            print(f"[ks1] L{l} NS{n} NTG{g}: {regs} regs, {local} B local, {smem_b} B smem, {occ} CTAs/SM", flush=True)
        ok_geo = _ks1_self_test(state["cos_sin"], device, KS1_INSTS, verbose=False)
        if KS2_ENABLE:
            state["ks2"] = _ks2_self_test(state["cos_sin"], device, [g for g in KS1_INSTS if g in ok_geo])
        if KS3_ENABLE:
            for n in KS3_INSTS:
                regs, local, smem_b, occ = (int(x) for x in mod.ks3_info(n))
                print(f"[ks3] NS{n}: {regs} regs, {local} B local, {smem_b} B smem, {occ} CTAs/SM", flush=True)
            state["ks3"] = _ks3_self_test(state["cos_sin"], device)
    except Exception as ex:
        print(f"[ks1] build / self-test FAILED {repr(ex)[-2000:]}: FA4 v8 stays", flush=True)
        ok_geo = set()
    state["ks1"] = set(ok_geo)
    _vsl_prepare(state, device, t0)
    _ax3_prepare(state, device, t0)
    if ok_geo and KS1_P1_CARVEOUT >= 0 and _P1_MOD is not None and hasattr(_P1_MOD, "p1s_carveout"):
        _P1_MOD.p1s_carveout(int(KS1_P1_CARVEOUT))
        print(f"[ks1] P1 preferred smem carveout {KS1_P1_CARVEOUT}%", flush=True)
    live = {l: tuple(KS1_GEOM[l]) for l in KS1_ROUTE if (l, *KS1_GEOM.get(l, (0, 0))) in ok_geo}
    print(f"[ks1] route {'live ' + str(live) if live else 'FAILED: FA4 v8 stays'} (build {_KS1['build_s']:.1f}s, total "
          f"{time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)

DG1_ENABLE = True

DG1_ROUTE = {('cat', 32): True, ('qkv', 32): True, ('o', 32): True}

DG1_GEOMETRY = {('cat', 32): (32, 2, 1, 3), ('qkv', 32): (32, 2, 1, 4), ('o', 32): (32, 1, 1, 1)}

DG1_INSTS = [(32, 1, 1), (32, 2, 1), (32, 2, 2), (128, 1, 1), (128, 2, 1), (128, 2, 2), (256, 1, 1), (256, 2, 1)]

DG1_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t dg1_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor err, int64_t S, int64_t pdl,
                int64_t trig, int64_t ldy, int64_t flags);
std::vector<int64_t> dg1_info(int64_t kid, int64_t S);
"""

DG1_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace dg1 {

constexpr long long WAIT_BUDGET = 2000000000LL;   // ~1 s at 1.98 GHz: a hang becomes err[0] = 1, never a trap

struct Params {
  const uint8_t* w;   // packed weights [K/64][N/8][8 kc][8 rows][16 B]
  void* y;            // [T, ldy] bf16 (f32 = 0) or fp32 (f32 = 1)
  int* err;           // err[0] = 1 on a bounded-wait timeout
  int N, nk, T, ldy, f32, trig;
  int flags;          // lab: bit0 exit at entry, bit1 skip wgmma, bit2 skip x loads, bit3 no L2 hint; bit8 tile-major W layout
};

template <int T, int NWG, int MT>
struct Cfg {
  static constexpr int BM = 64 * MT * NWG;                 // weight rows per CTA
  static constexpr int XB = T * 128;                        // x tile bytes (T rows x 64 bf16), SW128
  static constexpr int WB = BM * 128;                       // W tile bytes (BM rows x 64 bf16), core matrices
  static constexpr int STAGE = XB + WB;
  static constexpr int ST0 = (220 * 1024) / STAGE;
  static constexpr int ST = ST0 > 10 ? 10 : ST0;            // ring depth
  static constexpr int RS = BM + 4;                         // partial row stride (floats): conflict-free fragment stores
  static constexpr int SMEM = 1024 + ST * STAGE + 16 * ST;  // align slack + ring + full/empty barriers
  static constexpr int THREADS = 128 * NWG + 32;
  static_assert(T % 8 == 0 && T >= 8 && T <= 256, "T");
  static_assert(T * RS * 4 <= ST * STAGE, "partials fit the ring");
  static_assert(ST >= 2, "ring");
};

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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, bool& dead) {
  if (dead || mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); dead = true; return; }
  }
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_x(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t cl_n() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_nctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t r) {
  uint32_t o;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(o) : "r"(a), "r"(r));
  return o;
}
__device__ __forceinline__ float4 ld_cl4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t desc_il(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(1024u >> 4) << 32);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t a) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024u >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db);
@WGSPECS@
template <int NR> __device__ __forceinline__ void fence_regs(float* d) {
#pragma unroll
  for (int i = 0; i < NR; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}

template <int T, int NWG, int MT>
__global__ void __launch_bounds__(Cfg<T, NWG, MT>::THREADS, 1)
dg1_kernel(const __grid_constant__ CUtensorMap tmx, const Params P) {
  using C = Cfg<T, NWG, MT>;
  constexpr int ST = C::ST, STAGE = C::STAGE, XB = C::XB, BM = C::BM, RS = C::RS;
  extern __shared__ uint8_t dsm[];
  const uint32_t raw = sa(dsm);
  const uint32_t base = (raw + 1023u) & ~1023u;
  float* red = reinterpret_cast<float*>(dsm + (base - raw));
  const uint32_t bfull = base + ST * STAGE, bempty = bfull + 8 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = (int)cl_rank(), nrank = (int)cl_n();
  const int tile = blockIdx.y;
  const int kc0 = (P.nk * rank) / nrank, kc1 = (P.nk * (rank + 1)) / nrank, nloc = kc1 - kc0;
  if (P.trig) pdl_launch();
  if (P.flags & 1) return;
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, 4 * NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  bool dead = false;
  float acc[MT][T / 2];
  if (warp == 4 * NWG) {
    if (lane == 0) {
      const uint64_t pol = policy_evict_first();
      const bool tl = (P.flags >> 8) & 1, nox = (P.flags >> 2) & 1, noh = (P.flags >> 3) & 1;
      const uint32_t sbytes = nox ? (uint32_t)C::WB : (uint32_t)STAGE;
      const size_t slab = tl ? (size_t)C::WB : (size_t)P.N * 128;
      const uint8_t* wsrc = tl ? P.w + ((size_t)tile * P.nk + kc0) * C::WB : P.w + (size_t)kc0 * slab + (size_t)tile * (BM * 128);
      const int pre = nloc < ST ? nloc : ST;
      for (int i = 0; i < pre; ++i) {   // weights first: never written by a predecessor
        mbar_expect_tx(bfull + 8 * i, sbytes);
        if (noh) bulk_g2s(base + i * STAGE + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * i);
        else bulk_g2s_ef(base + i * STAGE + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * i, pol);
      }
      pdl_wait();
      if (!nox)
        for (int i = 0; i < pre; ++i) tma_x(base + i * STAGE, &tmx, (kc0 + i) * 64, 0, bfull + 8 * i);
      for (int i = ST; i < nloc; ++i) {
        const int s = i % ST;
        mbar_wait(bempty + 8 * s, (uint32_t)(((i / ST) - 1) & 1), P.err, dead);
        const uint32_t st = base + s * STAGE;
        mbar_expect_tx(bfull + 8 * s, sbytes);
        if (noh) bulk_g2s(st + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * s);
        else bulk_g2s_ef(st + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * s, pol);
        if (!nox) tma_x(st, &tmx, (kc0 + i) * 64, 0, bfull + 8 * s);
      }
    }
  } else {
    const int g = warp >> 2;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int q = 0; q < T / 2; ++q) acc[mt][q] = 0.f;
    for (int i = 0; i < nloc; ++i) {
      const int s = i % ST;
      mbar_wait(bfull + 8 * s, (uint32_t)((i / ST) & 1), P.err, dead);
      const uint32_t xs = base + s * STAGE, ws = xs + XB + g * (MT * 8192);
      wg_fence();
      if (!(P.flags & 2)) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int j = 0; j < 4; ++j) wgmma_ss<T>(acc[mt], desc_il(ws + mt * 8192 + j * 256), desc_sw128(xs + j * 32));
      }
      wg_commit();
      wg_wait<1>();
      if (i > 0 && lane == 0) mbar_arrive(bempty + 8 * ((i - 1) % ST));
    }
    wg_wait<0>();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) fence_regs<T / 2>(acc[mt]);
  }
  pdl_wait();
  __syncthreads();   // every warpgroup is done reading the ring before it is overwritten
  if (warp < 4 * NWG) {
    const int g = warp >> 2, w4 = warp & 3;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int r0 = g * MT * 64 + mt * 64 + w4 * 16 + (lane >> 2);
#pragma unroll
      for (int j = 0; j < T / 8; ++j) {
        const int t0 = 8 * j + 2 * (lane & 3);
        red[t0 * RS + r0] = acc[mt][4 * j];
        red[(t0 + 1) * RS + r0] = acc[mt][4 * j + 1];
        red[t0 * RS + r0 + 8] = acc[mt][4 * j + 2];
        red[(t0 + 1) * RS + r0 + 8] = acc[mt][4 * j + 3];
      }
    }
  }
  if (nrank > 1) cl_sync(); else __syncthreads();
  constexpr int R4 = BM / 4, Q4 = T * R4;
  const int q0 = (Q4 * rank) / nrank, q1 = (Q4 * (rank + 1)) / nrank;
  const uint32_t redA = sa(red);
  for (int q = q0 + tid; q < q1; q += C::THREADS) {
    const int t = q / R4, r4 = q - t * R4;
    float4 sum;
    if (nrank == 1) {
      sum = *reinterpret_cast<const float4*>(red + t * RS + 4 * r4);
    } else {
      const uint32_t off = redA + (uint32_t)(t * RS + 4 * r4) * 4u;
      sum = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int c = 0; c < nrank; ++c) {   // rank order: deterministic
        const float4 v = ld_cl4(mapa(off, (uint32_t)c));
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
      }
    }
    if (t < P.T) {
      const size_t o = (size_t)t * P.ldy + (size_t)tile * BM + 4 * r4;
      if (P.f32) {
        *reinterpret_cast<float4*>(reinterpret_cast<float*>(P.y) + o) = sum;
      } else {
        __nv_bfloat162 a = __floats2bfloat162_rn(sum.x, sum.y), b = __floats2bfloat162_rn(sum.z, sum.w);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(P.y) + o) = u;
      }
    }
  }
  if (nrank > 1) cl_sync();   // keep this CTA's partials alive until every rank has read them
}

}  // namespace dg1

typedef CUresult (*dg1_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                  const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static dg1_encode_fn dg1_get_encode() {
  static dg1_encode_fn fn = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) cudaGetLastError();
    else fn = reinterpret_cast<dg1_encode_fn>(p);
  }
  return fn;
}

template <int T, int NWG, int MT>
static int64_t dg1_launch_t(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& y, const torch::Tensor& err,
                            int64_t S, int64_t pdl, int64_t trig, int64_t ldy, int64_t flags) {
  using C = dg1::Cfg<T, NWG, MT>;
  auto kern = dg1::dg1_kernel<T, NWG, MT>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr = true;
  }
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "dg1: x bf16 [T, K] contiguous");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.dim() == 5 && w.is_contiguous(), "dg1: packed w [K/64, N/8, 8, 8, 8]");
  const int64_t Tact = x.size(0), K = x.size(1), N = w.size(1) * 8;
  TORCH_CHECK(w.size(0) * 64 == K && w.size(2) == 8 && w.size(3) == 8 && w.size(4) == 8, "dg1: packed w shape");
  TORCH_CHECK(Tact >= 1 && Tact <= T && K % 64 == 0 && N % C::BM == 0, "dg1: shape not supported by this instantiation");
  TORCH_CHECK(S >= 1 && S <= 16 && S <= K / 64, "dg1: split");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "dg1: align");
  const bool f32 = y.scalar_type() == at::kFloat;
  TORCH_CHECK(y.is_cuda() && (f32 || y.scalar_type() == at::kBFloat16) && y.dim() == 2 && y.stride(1) == 1 && y.stride(0) == ldy
              && y.size(0) >= Tact && y.size(1) >= N && ldy % 4 == 0 && reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0, "dg1: y");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, "dg1: err");
  dg1_encode_fn enc = dg1_get_encode();
  TORCH_CHECK(enc != nullptr, "dg1: cuTensorMapEncodeTiled unavailable");
  CUtensorMap tm;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)Tact};
  cuuint64_t strides[1] = {(cuuint64_t)(K * 2)};
  cuuint32_t box[2] = {64u, (cuuint32_t)T};
  cuuint32_t es[2] = {1u, 1u};
  const CUresult r = enc(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dg1: tensor map encode failed ", (int)r);
  dg1::Params P;
  P.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  P.y = y.data_ptr();
  P.err = err.data_ptr<int>();
  P.N = (int)N; P.nk = (int)(K / 64); P.T = (int)Tact; P.ldy = (int)ldy; P.f32 = f32 ? 1 : 0; P.trig = (int)trig;
  P.flags = (int)flags;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)S, (unsigned)(N / C::BM), 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute la[2];
  int na = 0;
  if (S > 1) {
    la[na].id = cudaLaunchAttributeClusterDimension;
    la[na].val.clusterDim.x = (unsigned)S; la[na].val.clusterDim.y = 1; la[na].val.clusterDim.z = 1;
    ++na;
  }
  la[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  la[na].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  ++na;
  cfg.attrs = la;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tm, P));
  return 0;
}

template <int T, int NWG, int MT>
static std::vector<int64_t> dg1_info_t(int64_t S) {
  using C = dg1::Cfg<T, NWG, MT>;
  auto kern = dg1::dg1_kernel<T, NWG, MT>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)S, 1, 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cudaLaunchAttribute la[1];
  la[0].id = cudaLaunchAttributeClusterDimension;
  la[0].val.clusterDim.x = (unsigned)S; la[0].val.clusterDim.y = 1; la[0].val.clusterDim.z = 1;
  cfg.attrs = la;
  cfg.numAttrs = 1;
  int ncl = -1;
  if (cudaOccupancyMaxActiveClusters(&ncl, kern, &cfg) != cudaSuccess) { cudaGetLastError(); ncl = -1; }
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  return {(int64_t)ncl, (int64_t)C::SMEM, (int64_t)C::THREADS, (int64_t)C::BM, (int64_t)C::ST, (int64_t)fa.numRegs,
          (int64_t)fa.localSizeBytes};
}

int64_t dg1_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor err, int64_t S, int64_t pdl,
                int64_t trig, int64_t ldy, int64_t flags) {
  switch (kid) {
@RUNCASES@
  }
  TORCH_CHECK(false, "dg1: unknown kid ", kid);
  return -2;
}

std::vector<int64_t> dg1_info(int64_t kid, int64_t S) {
  switch (kid) {
@INFOCASES@
  }
  TORCH_CHECK(false, "dg1: unknown kid ", kid);
  return {};
}
"""

def dg1_wg_spec(n):
    nr = n // 2
    regs = ",".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f"template <> __device__ __forceinline__ void wgmma_ss<{n}>(float* d, uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f"               : {outs}\n               : \"l\"(da), \"l\"(db));\n}}\n")

def dg1_cuda_src(insts=DG1_INSTS):
    ns = sorted({t for t, _, _ in insts})
    run = "\n".join(f"    case {k}: return dg1_launch_t<{t}, {g}, {m}>(x, w, y, err, S, pdl, trig, ldy, flags);"
                    for k, (t, g, m) in enumerate(insts))
    info = "\n".join(f"    case {k}: return dg1_info_t<{t}, {g}, {m}>(S);" for k, (t, g, m) in enumerate(insts))
    return (DG1_CUDA_SRC.replace("@WGSPECS@", "".join(dg1_wg_spec(n) for n in ns))
            .replace("@RUNCASES@", run).replace("@INFOCASES@", info))

DG1_FOLD_KS1 = True

def _dg1_insts():
    return tuple(sorted({DG1_GEOMETRY[k][:3] for k in DG1_ROUTE if k in DG1_GEOMETRY}))

def _dg1_fold_on():
    return bool(DG1_ENABLE and DG1_FOLD_KS1 and _dg1_insts())

def _dg1_fold(part):
    if not _dg1_fold_on():
        return [] if part == "fns" else ""
    return {"cuda": "\n" + dg1_cuda_src(list(_dg1_insts())), "cpp": DG1_CPP_SRC, "fns": ["dg1_run", "dg1_info"]}[part]

def _k3_disable(state, key, value, piece, ex):
    state[key] = value
    if piece not in _K3_WARNED:
        _K3_WARNED.add(piece)
        print(f"[k3] {piece} raised at call time, r1 path from now on: {ex!r}", flush=True)

VFA_ENABLE = True

VFA_MIN_B = 2

VFA_MAX_B = 256

VFA_STAGES = 4

VFA_SEGK = 300

VFA_PDL = True

VFA_DIAG = 0

VFA_SELF_TEST = True

VFA_CASES = ((2, 3000, 1, 1.0, 401), (3, 1500, 64, 1.0, 402), (16, 700, 1, 1.0 / 64, 403), (5, 9000, 1, 1.0, 404),
             (33, 300, 1, 1.0, 405), (32, 2500, 1, 1.0, 406), (2, 20, 1, 1.0, 407), (17, 100, 1, 1.0, 408), (4, 70, 64, 1.0, 409))

_VFA = {"mod": None, "ws": {}, "sms": {}, "occ": {}}

VFA_CPP_SRC = r"""
#include <torch/extension.h>
void vfa_forward(torch::Tensor q8, torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor page_table, torch::Tensor k_cache,
                 torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor ws_o, torch::Tensor ws_l,
                 torch::Tensor tickets, torch::Tensor out, int64_t B, int64_t Hq, int64_t Hkv, int64_t page_shift, int64_t G,
                 int64_t segk, int64_t pdl, double qk_sm, int64_t stages, int64_t diag);
int64_t vfa_occupancy(int64_t stages);
"""

VFA_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define VFA_SM90 1
#else
#define VFA_SM90 0
#endif

namespace vfa {

constexpr int TILE = 64;                        // keys per stage
constexpr int ROWB = 128;                       // bytes per key row in a stage: 2 KV heads x 64 dims e4m3
constexpr int STAGE_BYTES = TILE * ROWB;        // 8 KB per K stage (same for V)
constexpr int VT_BYTES = 72 * 64;               // V^T per (head, buffer): 72 rows (64 dims + 8 ones) x 64 keys, 64B swizzle
constexpr int Q_BYTES = 64 * 64;                // e4m3 Q tile per head: 64 rows (8 tokens x 8 heads) x 64 dims, 64B swizzle
constexpr int MAXB = 256;
constexpr int MAXSEG = 32;
constexpr int NCONS = 256;
constexpr int NPROD = 128;
constexpr int NTHR = NCONS + NPROD;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float POFF = 0.0f;                    // offset = a segment's first-tile row max: P = 2^(s - off) <= 1 there, later
constexpr uint32_t PCHK = 0x07070707u;          // + 7 per byte sets bit 7 for bytes > 0x78: P > 256 -> move the offset
constexpr unsigned FULL = 0xffffffffu;

template <int ST> constexpr int off_v() { return ST * STAGE_BYTES; }
template <int ST> constexpr int off_vt() { return 2 * ST * STAGE_BYTES; }
template <int ST> constexpr int off_q() { return off_vt<ST>() + 4 * VT_BYTES; }
template <int ST> constexpr int off_bar() { return off_q<ST>() + 2 * Q_BYTES; }
template <int ST> constexpr int off_cum() { return off_bar<ST>() + 128; }
template <int ST> constexpr int off_seg() { return off_cum<ST>() + (2 * MAXB + 4) * 4; }
template <int ST> constexpr int off_misc() { return off_seg<ST>() + MAXSEG * 16; }
template <int ST> constexpr int off_adr() { return off_misc<ST>() + 64; }
template <int ST> constexpr int smem_bytes() { return off_adr<ST>() + 4096 + 1024; }   // FA16-VSM slots, 1 KB align slack last

struct Params {
  const uint8_t* q8;            // [B*8, Hq*64] e4m3 unit-RMS queries (FA4 prologue)
  const __nv_bfloat16* qkv;     // [B*8, ld_qkv] bf16 (gate columns)
  const int* seq_lens;          // [B] cached keys before the block; the block's 8 keys follow in the cache
  const int* page_table;        // [B, pt_stride]
  const uint8_t* k_cache;       // rows of Hkv*64 e4m3
  const uint8_t* v_cache;
  const float* k_scale;
  const float* v_scale;
  float* ws_o;                  // [B*pairs*128, 64] fp32 absolute-frame numerators (zero between calls)
  float* ws_l;                  // [B*pairs*128] fp32 denominators
  int* tickets;                 // [B*pairs] key-count tickets (zero between calls)
  __nv_bfloat16* out;           // [B*8, Hq*64]
  int B, Hq, Hkv, ld_qkv, page_shift, pt_stride, segk, pdl;
  float qk_sm;                  // qk_scale_factor * sm_scale
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float ex2p(float x) {   // FA16-PEX: 2^x on the FMA pipe (x in [-125, 126])
  x = fminf(fmaxf(x, -125.f), 126.f);
  const float t = x + 12582912.f;                     // 1.5 x 2^23: round(x) in the low mantissa bits
  const float f = x - (t - 12582912.f);               // [-0.5, 0.5]
  const float p = fmaf(fmaf(fmaf(0.05550411f, f, 0.24022652f), f, 0.69314718f), f, 1.0f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ uint32_t pk4(float a, float b, float c, float d) {   // 4 x e4m3 (satfinite), a in the low byte
  uint32_t r;
  asm("{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\ncvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\nmov.b32 %0, {lo, hi};\n}"
      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
__device__ __forceinline__ uint32_t pk4h(float a, float b, float c, float d) {   // FA16-H16: 4 x e4m3(2^x) via f16x2, a low byte
  uint32_t r;
  asm("{\n.reg .b32 h0, h1;\n.reg .b16 lo, hi;\ncvt.rn.f16x2.f32 h0, %2, %1;\ncvt.rn.f16x2.f32 h1, %4, %3;\n"
      "ex2.approx.f16x2 h0, h0;\nex2.approx.f16x2 h1, h1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 lo, h0;\ncvt.rn.satfinite.e4m3x2.f16x2 hi, h1;\nmov.b32 %0, {lo, hi};\n}"
      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(FULL, v, 1));
  return fmaxf(v, __shfl_xor_sync(FULL, v, 2));
}
#define VFA_RF(x) asm volatile("" : "+f"(x))
#define VFA_RR(x) asm volatile("" : "+r"(x))

#if VFA_SM90
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nVFAW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra VFAW_%=;\n}\n"
               ::"r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void sts32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory"); }
__device__ __forceinline__ void red4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void red1(float* a, float x) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_named(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
#define VFA_D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define VFA_R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define VFA_R36 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}"
__device__ __forceinline__ void wg_qk(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " VFA_R32 ", %32, %33, p, 1, 1;\n}\n"
               : VFA_D32(d) : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wg_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 " VFA_R36 ", {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : VFA_D32(d), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
template <int DIAG> constexpr bool pvt_on() { return (DIAG >= 18 && DIAG <= 20) || (DIAG == 0 && true); }   // VFA3-PVT
template <int DIAG> constexpr int pvt_lag() { return DIAG == 19 ? 1 : (DIAG == 20 ? 3 : (DIAG == 18 ? 2 : 2)); }
#define VFA_PVT_PREG "32"
#define VFA_PVT_CREG "104"

template <int ST, bool O32, int DIAG>
__device__ __forceinline__ void producer(const Params& p, uint8_t* smem, const int* seg, int nseg, uint64_t* full, uint64_t* empt,
                                         int q, int pair) {
  const int ch = q & 7, r0 = q >> 3;
  const int lp = p.page_shift, pm = (1 << lp) - 1;
  const uint32_t rowb = (uint32_t)p.Hkv * 64u;
  const uint8_t* kc = p.k_cache + pair * 128 + ch * 16;
  const uint8_t* vc = p.v_cache + pair * 128 + ch * 16;
  const uint32_t so = sa(smem) + r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
  int n = 0;
  constexpr bool PVT = pvt_on<DIAG>();   // VFA3-PVT: this warp's 16 keys of both heads -> V^T (consumer's mapping, w = q >> 5)
  constexpr int LAG = pvt_lag<DIAG>();
  uint64_t* const vtf = empt + ST;
  uint64_t* const vte = vtf + 2;
  const int vlane = q & 31, vw = q >> 5, vg = vlane >> 2;
  const uint32_t vld0 = sa(smem) + off_v<ST>() + (16 * vw + 8 * ((vlane >> 3) & 1) + (vlane & 7)) * ROWB + (((vlane >> 4) ^ (vlane & 7)) << 4);
  const uint32_t vsta = sa(smem) + off_vt<ST>() + 128 * vg + ((vw ^ (vg & 3)) << 4) + 4 * (vlane & 3) + ((vg >= 4) ? 64u : 0u);
  const uint32_t vsa = (vg >= 4) ? 0x7531u : 0x6420u, vsb = vsa ^ 0x1111u;
  auto vt_tile = [&](int j) {
    const int st = j % ST, bf = j & 1;
    mbar_wait(full + st, (uint32_t)(j / ST) & 1u);
    if (j >= 2) mbar_wait(vte + bf, (uint32_t)((j >> 1) - 1) & 1u);
    const uint32_t vs0 = vld0 + (uint32_t)(st * STAGE_BYTES), d0 = vsta + (uint32_t)(bf * VT_BYTES);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const uint32_t vsrc = vs0 ^ (64u * hh), da = d0 + (uint32_t)(hh * 2 * VT_BYTES), db = da ^ 64u;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t r[4];
        ldsm_x4_t(r, u ? (vsrc ^ 32u) : vsrc);
        sts32(da + 1024 * (2 * u), prmt(r[0], r[1], vsa));
        sts32(db + 1024 * (2 * u), prmt(r[0], r[1], vsb));
        sts32(da + 1024 * (2 * u + 1), prmt(r[2], r[3], vsa));
        sts32(db + 1024 * (2 * u + 1), prmt(r[2], r[3], vsb));
      }
    }
    mbar_arrive(empt + st);   // the V half of stage st read
    fence_proxy_async();      // V^T (st.shared) -> async proxy (wgmma)
    mbar_arrive(vtf + bf);
  };
  (void)vt_tile; (void)vtf; (void)vte;
  for (int s = 0; s < nseg; ++s) {
    const int b = seg[4 * s], k0 = seg[4 * s + 1], k1 = seg[4 * s + 2];
    const int nt = (k1 - k0 + TILE - 1) / TILE;
    const int* pt = p.page_table + (long long)b * p.pt_stride;
    int pg[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int pos = k0 + r0 + 16 * k;
      pg[k] = pos < k1 ? __ldg(pt + (pos >> lp)) : -1;
    }
    for (int x = 0; x < nt; ++x, ++n) {
      int pgn[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {   // next tile's pages: in flight while this tile's copies issue
        const int pos = k0 + (x + 1) * TILE + r0 + 16 * k;
        pgn[k] = pos < k1 ? __ldg(pt + (pos >> lp)) : -1;
      }
      const int st = n % ST;
      if (n >= ST) mbar_wait(empt + st, (uint32_t)((n / ST) - 1) & 1u);
      const uint32_t kb = so + st * STAGE_BYTES, vb = kb + off_v<ST>();
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int pos = k0 + x * TILE + r0 + 16 * k;
        if (pg[k] >= 0) {
          if (O32) {
            const uint32_t off = (((uint32_t)pg[k] << lp) + (uint32_t)(pos & pm)) * rowb;
            cp_async16(kb + k * 16 * ROWB, kc + off, 16);
            cp_async16(vb + k * 16 * ROWB, vc + off, 16);
          } else {
            const long long off = (((long long)pg[k] << lp) + (pos & pm)) * (long long)rowb;
            cp_async16(kb + k * 16 * ROWB, kc + off, 16);
            cp_async16(vb + k * 16 * ROWB, vc + off, 16);
          }
        } else {   // past the segment: zero-filled rows (masked keys must not meet NaN bytes in V)
          cp_async16(kb + k * 16 * ROWB, kc, 0);
          cp_async16(vb + k * 16 * ROWB, vc, 0);
        }
      }
      cp_async_mbar_arrive(full + st);
      if constexpr (PVT) { if (n >= LAG) vt_tile(n - LAG); }   // VFA3-PVT: tile n - LAG landed long ago
#pragma unroll
      for (int k = 0; k < 4; ++k) pg[k] = pgn[k];
    }
  }
  if constexpr (PVT) { for (int j = (n > LAG ? n - LAG : 0); j < n; ++j) vt_tile(j); }   // VFA3-PVT: drain
}

template <int ST, int DIAG>
__device__ __forceinline__ void consumer(const Params& p, uint8_t* smem, const int* seg, const int* sN, int nseg, uint64_t* full,
                                         uint64_t* empt, int* misc, int tid, int pair, int pairs) {
  const int lane = tid & 31, warp = __shfl_sync(FULL, tid >> 5, 0);
  const int h = warp >> 2, w = warp & 3;
  const int g = lane >> 2, tig = lane & 3;
  const int hq = p.Hq;
  constexpr bool DEFER = (DIAG == 5);   // PV(i) waited at tile i + 1 (with QK(i + 1)) instead of at the end of tile i
  constexpr float POF = (DIAG == 6) ? 8.f : POFF;              // DIAG 6 (lab): offset = max - 8 with the 448 threshold (R-vfa-2..4)
  constexpr uint32_t PBK = (DIAG == 6) ? 0x02020202u : PCHK;
  constexpr bool NOVT = (DIAG == 17);   // VFA3 lab bound: no V^T at all (wrong output)
  constexpr bool PVT = pvt_on<DIAG>();   // VFA3-PVT: V^T written by the producer warps
  uint64_t* const vtf = empt + ST;
  uint64_t* const vte = vtf + 2;
  constexpr int PEX = (DIAG >= 8 && DIAG <= 11) ? 4 * (DIAG - 7) : 0;   // FA16-PEX (lab 8..11)
  constexpr bool H16 = (DIAG == 12 || DIAG == 14) || (DIAG == 0 && false);   // FA16-H16
  constexpr bool PVS = (DIAG == 13 || DIAG == 14) || (DIAG == 0 && false);   // FA16-PVS
  const float vs = __ldg(p.v_scale);
  const float ssc = p.qk_sm * __ldg(p.k_scale) * LOG2E;   // raw e4m3 x e4m3 score -> log2 units
  const uint32_t qlo = __shfl_sync(FULL, ((sa(smem) + off_q<ST>() + h * Q_BYTES) >> 4) | 0x10000u, 0);
  const uint32_t klo = __shfl_sync(FULL, ((sa(smem) + h * 64) >> 4) | 0x10000u, 0);
  const uint32_t vtlo = __shfl_sync(FULL, ((sa(smem) + off_vt<ST>() + h * 2 * VT_BYTES) >> 4) | 0x10000u, 0);
  constexpr uint64_t HI64 = ((uint64_t)(512 >> 4) | (2ull << 30)) << 32;
  constexpr uint64_t HI128 = ((uint64_t)(1024 >> 4) | (1ull << 30)) << 32;
  const int vkey = 16 * w + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld0 = sa(smem) + off_v<ST>() + vkey * ROWB + (((4 * h + (lane >> 4)) ^ (lane & 7)) << 4);
  const uint32_t vst = sa(smem) + off_vt<ST>() + h * 2 * VT_BYTES + 128 * g + ((w ^ (g & 3)) << 4) + 4 * tig;
  const uint32_t vsel_a = (g >= 4) ? 0x7531u : 0x6420u, vsel_b = (g >= 4) ? 0x6420u : 0x7531u;
  const uint32_t vst_a = vst + ((g >= 4) ? 64u : 0u), vst_b = vst + ((g >= 4) ? 0u : 64u);
  constexpr bool VSM = (DIAG == 16 || (DIAG == 0 && false));
  const uint32_t vslot = sa(smem) + off_adr<ST>() + (uint32_t)tid * 16u;
  if (VSM) asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(vslot), "r"(vld0), "r"(vst_a), "r"(vsel_a), "r"(vsel_b) : "memory");
  const int tl = tid & 127;
  int n = 0, st_c = 0, buf = 0;
  constexpr bool PP = (DIAG == 15 || (DIAG == 0 && false));
  if (PP && h == 1) bar_arrive(4, 256);   // FA16-PP: warpgroup 0 exponentiates first
  uint32_t ph_c = 0;
  for (int s = 0; s < nseg; ++s) {
    const int b = seg[4 * s], k0 = seg[4 * s + 1], k1 = seg[4 * s + 2];
    const int seq = __ldg(p.seq_lens + b);                     // the block's keys sit at seq .. seq + 7
    const int nt = (k1 - k0 + TILE - 1) / TILE;
    const int lim = min(k1, seq + 1);                            // a tile is mask-free iff t0 + 63 < lim
    const int bp = b * pairs + pair;
    if (DIAG != 1) {   // this head's e4m3 Q tile: row r = 8 j + hh at r * 64, 16-B chunk c at ((c ^ ((r >> 1) & 3)) << 4)
      const int r = tl & 63, c0 = 2 * (tl >> 6);
      const uint4* src = reinterpret_cast<const uint4*>(p.q8 + ((long long)(b * 8 + (r >> 3)) * hq + pair * 16 + 8 * h + (r & 7)) * 64 + c0 * 16);
      const uint4 a = __ldg(src), c = __ldg(src + 1);
      uint8_t* qd = smem + off_q<ST>() + h * Q_BYTES + r * 64;
      *reinterpret_cast<uint4*>(qd + ((c0 ^ ((r >> 1) & 3)) << 4)) = a;
      *reinterpret_cast<uint4*>(qd + (((c0 + 1) ^ ((r >> 1) & 3)) << 4)) = c;
      fence_proxy_async();
      bar_named(1 + h, 128);
    }
    float acc[36];
#pragma unroll
    for (int e = 0; e < 36; ++e) { acc[e] = 0.f; VFA_RF(acc[e]); }
    float cr0 = 0.f, cr1 = 0.f;
    bool first = true;
    uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};   // P fragments (loop-carried: live until the PV wait)
    for (int i = 0; i < nt; ++i, ++n) {
      mbar_wait(full + st_c, ph_c);
      if (DIAG == 1) {
        mbar_arrive(empt + st_c);
      } else {
        float sc[32];
        const uint32_t kl = klo + (uint32_t)(st_c * (STAGE_BYTES >> 4));
        if (DIAG != 3) {
          wg_fence();
          wg_qk(sc, HI64 | qlo, HI128 | kl, 0);
          wg_qk(sc, HI64 | (qlo + 2u), HI128 | (kl + 2u), 1);
          wg_commit();
        } else {
#pragma unroll
          for (int e = 0; e < 32; ++e) sc[e] = 0.f;
        }
        if (DIAG != 2 && !NOVT && !PVT) {   // this tile's V -> V^T (this head, buffer buf) while the QK wgmma runs
          if constexpr (VSM) {   // FA16-VSM: constants from this thread's smem slot (db = da ^ 64, selector b from the slot; the old a ^ 0x1311 was wrong, R-vfa2-1)
            uint32_t x0, x1, x2, x3;
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(vslot) : "memory");
            const uint32_t vsrc = x0 + (uint32_t)(st_c * STAGE_BYTES);
            const uint32_t da = x1 + (uint32_t)(buf * VT_BYTES), db = da ^ 64u, sA = x2, sB = x3;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
              uint32_t r[4];
              ldsm_x4_t(r, u ? (vsrc ^ 32u) : vsrc);
              sts32(da + 1024 * (2 * u), prmt(r[0], r[1], sA));
              sts32(db + 1024 * (2 * u), prmt(r[0], r[1], sB));
              sts32(da + 1024 * (2 * u + 1), prmt(r[2], r[3], sA));
              sts32(db + 1024 * (2 * u + 1), prmt(r[2], r[3], sB));
            }
          } else {
          const uint32_t vsrc = vld0 + (uint32_t)(st_c * STAGE_BYTES);
          const uint32_t da = vst_a + (uint32_t)(buf * VT_BYTES), db = vst_b + (uint32_t)(buf * VT_BYTES);
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            uint32_t r[4];
            ldsm_x4_t(r, u ? (vsrc ^ 32u) : vsrc);
            sts32(da + 1024 * (2 * u), prmt(r[0], r[1], vsel_a));
            sts32(db + 1024 * (2 * u), prmt(r[0], r[1], vsel_b));
            sts32(da + 1024 * (2 * u + 1), prmt(r[2], r[3], vsel_a));
            sts32(db + 1024 * (2 * u + 1), prmt(r[2], r[3], vsel_b));
          }
          }
        }
        if (DIAG != 3) wg_wait0();                               // QK(i) (and, deferred, PV(i - 1))
#pragma unroll
        for (int e = 0; e < 32; ++e) VFA_RF(sc[e]);
        if (DEFER) {
#pragma unroll
          for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) VFA_RR(pa[e >> 2][e & 3]);
        }
        mbar_arrive(empt + st_c);                                // K (wgmma) and V (ldmatrix) of the stage consumed
        const int t0 = k0 + i * TILE;
        if (t0 + TILE - 1 >= lim) {                              // edge tile: range + causal mask
          const int hi0 = min(k1 - 1, seq + 2 * w), hi1 = min(k1 - 1, seq + 2 * w + 1);
#pragma unroll
          for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int pos = t0 + 8 * nt8 + 2 * tig + e;
              if (pos > hi0) sc[4 * nt8 + e] = neg_inf();
              if (pos > hi1) sc[4 * nt8 + 2 + e] = neg_inf();
            }
        }
        if (first || DIAG == 4) {                                // per-row offsets = this tile's row maxima (log2 units)
          float m0 = neg_inf(), m1 = neg_inf();
#pragma unroll
          for (int nt8 = 0; nt8 < 8; ++nt8) {
            m0 = fmaxf(m0, fmaxf(sc[4 * nt8], sc[4 * nt8 + 1]));
            m1 = fmaxf(m1, fmaxf(sc[4 * nt8 + 2], sc[4 * nt8 + 3]));
          }
          m0 = quad_max(m0) * ssc;
          m1 = quad_max(m1) * ssc;
          if (first) {
            cr0 = (m0 == neg_inf()) ? 0.f : m0 - POF;
            cr1 = (m1 == neg_inf()) ? 0.f : m1 - POF;
          } else {
            const float n0 = fmaxf(m0 - POF, cr0), n1 = fmaxf(m1 - POF, cr1);
            const float a0 = ex2(cr0 - n0), a1 = ex2(cr1 - n1);
#pragma unroll
            for (int t = 0; t < 9; ++t) { acc[4 * t] *= a0; acc[4 * t + 1] *= a0; acc[4 * t + 2] *= a1; acc[4 * t + 3] *= a1; }
            cr0 = n0; cr1 = n1;
          }
        }
        auto e2 = [&](float x, int k) {   // FA16-PEX: element k of 32 on the FMA pipe iff (k * PEX) % 32 < PEX
          if (DIAG == 7) return x;
          return (((k * PEX) & 31) < PEX) ? ex2p(x) : ex2(x);
        };
        auto q4 = [&](float a, float b, float c, float d, int k) -> uint32_t {   // FA16-H16: exps as f16x2, same e4m3 P
          if constexpr (H16) return pk4h(a, b, c, d);
          else return pk4(e2(a, k), e2(b, k + 1), e2(c, k + 2), e2(d, k + 3));
        };
        auto packh = [&](int kk) {   // P = e4m3(ex2(s * ssc - offset)) of keys 32 kk .. +31, permuted into the PV A-fragment order
          const float* sv = sc + 16 * kk;
          const int q = 16 * kk;
          pa[kk][0] = q4(fmaf(sv[0], ssc, -cr0), fmaf(sv[1], ssc, -cr0), fmaf(sv[4], ssc, -cr0), fmaf(sv[5], ssc, -cr0), q);
          pa[kk][1] = q4(fmaf(sv[2], ssc, -cr1), fmaf(sv[3], ssc, -cr1), fmaf(sv[6], ssc, -cr1), fmaf(sv[7], ssc, -cr1), q + 4);
          pa[kk][2] = q4(fmaf(sv[8], ssc, -cr0), fmaf(sv[9], ssc, -cr0), fmaf(sv[12], ssc, -cr0), fmaf(sv[13], ssc, -cr0), q + 8);
          pa[kk][3] = q4(fmaf(sv[10], ssc, -cr1), fmaf(sv[11], ssc, -cr1), fmaf(sv[14], ssc, -cr1), fmaf(sv[15], ssc, -cr1), q + 12);
        };
        auto pack = [&]() { packh(0); packh(1); };
        if (PP) bar_named(4 + h, 256);   // FA16-PP: my turn
        const bool was_first = first;
        auto bump = [&]() {   // FA16-PVS: offsets up to this tile's row maxima, state rescaled (rare)
          float m0 = neg_inf(), m1 = neg_inf();
#pragma unroll
          for (int nt8 = 0; nt8 < 8; ++nt8) {
            m0 = fmaxf(m0, fmaxf(sc[4 * nt8], sc[4 * nt8 + 1]));
            m1 = fmaxf(m1, fmaxf(sc[4 * nt8 + 2], sc[4 * nt8 + 3]));
          }
          m0 = quad_max(m0) * ssc;
          m1 = quad_max(m1) * ssc;
          const float n0 = fmaxf(m0 - POF, cr0), n1 = fmaxf(m1 - POF, cr1);
          const float a0 = ex2(cr0 - n0), a1 = ex2(cr1 - n1);
#pragma unroll
          for (int t = 0; t < 9; ++t) { acc[4 * t] *= a0; acc[4 * t + 1] *= a0; acc[4 * t + 2] *= a1; acc[4 * t + 3] *= a1; }
          cr0 = n0; cr1 = n1;
        };
        auto ovf = [&](int kk) -> bool {
          constexpr uint32_t PB = PBK;
          const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
          return __any_sync(FULL, ov != 0u);
        };
        if constexpr (!PVS) {
        pack();
        if (!first && DIAG != 4) {   // lazy offset: an e4m3 byte >= 0x7E (P >= 448, saturation) sets bit 7 of (byte + 2)
          constexpr uint32_t PB = PBK;
          const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |
                               (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
          if (__any_sync(FULL, ov != 0u)) {   // rare: offsets up to this tile's row maxima, state rescaled, repacked
            float m0 = neg_inf(), m1 = neg_inf();
#pragma unroll
            for (int nt8 = 0; nt8 < 8; ++nt8) {
              m0 = fmaxf(m0, fmaxf(sc[4 * nt8], sc[4 * nt8 + 1]));
              m1 = fmaxf(m1, fmaxf(sc[4 * nt8 + 2], sc[4 * nt8 + 3]));
            }
            m0 = quad_max(m0) * ssc;
            m1 = quad_max(m1) * ssc;
            const float n0 = fmaxf(m0 - POF, cr0), n1 = fmaxf(m1 - POF, cr1);
            const float a0 = ex2(cr0 - n0), a1 = ex2(cr1 - n1);
#pragma unroll
            for (int t = 0; t < 9; ++t) { acc[4 * t] *= a0; acc[4 * t + 1] *= a0; acc[4 * t + 2] *= a1; acc[4 * t + 3] *= a1; }
            cr0 = n0; cr1 = n1;
            pack();
          }
        }
        if (PP) bar_arrive(5 - h, 256);   // FA16-PP: the other warpgroup's turn
        first = false;
        if (DIAG != 2) {
#pragma unroll
          for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) VFA_RR(pa[e >> 2][e & 3]);
          if constexpr (PVT) {   // VFA3-PVT: the producer's V^T of this tile (fence by the writers)
            mbar_wait(vtf + buf, ((uint32_t)n >> 1) & 1u);
          } else if constexpr (!NOVT) {
            fence_proxy_async();                                   // V^T (st.shared) -> async proxy
            bar_named(1 + h, 128);                                 // all 4 warps' V^T rows written
          }
          const uint32_t vl = vtlo + (uint32_t)(buf * (VT_BYTES >> 4));
          wg_fence();
          wg_pv(acc, pa[0], HI64 | vl);
          wg_pv(acc, pa[1], HI64 | (vl + 2u));
          wg_commit();
          if (!DEFER) {
            wg_wait0();                                          // nothing in flight across the loop edge
#pragma unroll
            for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
#pragma unroll
            for (int e = 0; e < 8; ++e) VFA_RR(pa[e >> 2][e & 3]);
          }
        }
        } else {   // FA16-PVS: PV of keys 0..31 in flight while keys 32..63 are exponentiated
          packh(0);
          if (!was_first && DIAG != 4 && ovf(0)) { bump(); packh(0); }
          first = false;
#pragma unroll
          for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) VFA_RR(pa[0][e]);
          if constexpr (PVT) {   // VFA3-PVT: the producer's V^T of this tile (fence by the writers)
            mbar_wait(vtf + buf, ((uint32_t)n >> 1) & 1u);
          } else if constexpr (!NOVT) {
            fence_proxy_async();                                   // V^T (st.shared) -> async proxy
            bar_named(1 + h, 128);                                 // all 4 warps' V^T rows written
          }
          const uint32_t vl = vtlo + (uint32_t)(buf * (VT_BYTES >> 4));
          wg_fence();
          wg_pv(acc, pa[0], HI64 | vl);
          wg_commit();
          packh(1);
          if (!was_first && DIAG != 4 && ovf(1)) {   // rare: PV(keys 0..31) done, then rescale and repack keys 32..63
            wg_wait0();
#pragma unroll
            for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
            bump();
            packh(1);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) VFA_RR(pa[1][e]);
          wg_fence();
          wg_pv(acc, pa[1], HI64 | (vl + 2u));
          wg_commit();
          wg_wait0();                                            // nothing in flight across the loop edge
#pragma unroll
          for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) VFA_RR(pa[e >> 2][e & 3]);
        }
      }
      if constexpr (PVT) mbar_arrive(vte + buf);   // VFA3-PVT: PV(n) done (waited), V^T buffer free
      buf ^= 1;
      if (++st_c == ST) { st_c = 0; ph_c ^= 1u; }
    }
    if (DIAG == 1) continue;
    if (DEFER) {   // the last tile's PV
      wg_wait0();
#pragma unroll
      for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) VFA_RR(pa[e >> 2][e & 3]);
    }
    const float f0 = ex2(fminf(cr0, 126.f)), f1 = ex2(fminf(cr1, 126.f));
    const int R0 = bp * 128 + (2 * w) * 16 + 8 * h + g, R1 = R0 + 16;
    float* a0p = p.ws_o + (long long)R0 * 64;
    float* a1p = p.ws_o + (long long)R1 * 64;
    const bool odd = tig & 1;
#pragma unroll
    for (int t = 0; t < 8; t += 2) {   // lanes (tig, tig ^ 1) swap so each holds 4 consecutive dims (one 16-B red)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float f = r ? f1 : f0;
        const float x0 = acc[4 * t + 2 * r] * f, x1 = acc[4 * t + 2 * r + 1] * f;
        const float y0 = acc[4 * (t + 1) + 2 * r] * f, y1 = acc[4 * (t + 1) + 2 * r + 1] * f;
        const float s0 = odd ? x0 : y0, s1 = odd ? x1 : y1;
        const float q0 = __shfl_xor_sync(FULL, s0, 1), q1 = __shfl_xor_sync(FULL, s1, 1);
        float* ap = (r ? a1p : a0p) + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1);
        red4(ap, odd ? q0 : x0, odd ? q1 : x1, odd ? y0 : q0, odd ? y1 : q1);
      }
    }
    if (tig == 0) {
      red1(p.ws_l + R0, acc[32] * f0);
      red1(p.ws_l + R1, acc[34] * f1);
    }
    bar_named(3, NCONS);
    if (tid == 0) {
      __threadfence();
      const int mine = k1 - k0;
      const int old = atomicAdd(p.tickets + bp, mine);
      misc[1] = (old + mine == sN[b]) ? 1 : 0;
    }
    bar_named(3, NCONS);
    if (misc[1]) {   // last segment of request b: normalize, rmsnorm, gate, bf16; zero what was read
      __threadfence();
      const int rr = tid >> 1, hf = tid & 1;
      const long long R = (long long)bp * 128 + rr;
      float4* src = reinterpret_cast<float4*>(p.ws_o + R * 64 + hf * 32);
      float4 v[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) v[e] = __ldcg(src + e);
      const float l = __ldcg(p.ws_l + R);
      const float inv = l > 0.f ? __fdividef(vs, l) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        v[e].x *= inv; v[e].y *= inv; v[e].z *= inv; v[e].w *= inv;
        ss += v[e].x * v[e].x + v[e].y * v[e].y + v[e].z * v[e].z + v[e].w * v[e].w;
      }
      ss += __shfl_xor_sync(FULL, ss, 1);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const int j = rr >> 4, hh = pair * 16 + (rr & 15);
      const long long tok = (long long)b * 8 + j;
      const uint4* gp = reinterpret_cast<const uint4*>(p.qkv + tok * p.ld_qkv + (hq + 2 * p.Hkv) * 64 + hh * 64 + hf * 32);
      uint4* op = reinterpret_cast<uint4*>(p.out + tok * hq * 64 + hh * 64 + hf * 32);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const uint4 gr = __ldg(gp + e);
        const uint32_t gw[4] = {gr.x, gr.y, gr.z, gr.w};
        uint32_t ow[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(&gw[c]);
          const float ga = __bfloat162float(g2.x), gb = __bfloat162float(g2.y);
          const float ya = (c & 1 ? v[2 * e + (c >> 1)].z : v[2 * e + (c >> 1)].x) * rn * __fdividef(1.f, 1.f + __expf(-ga));
          const float yb = (c & 1 ? v[2 * e + (c >> 1)].w : v[2 * e + (c >> 1)].y) * rn * __fdividef(1.f, 1.f + __expf(-gb));
          const __nv_bfloat162 o2 = __floats2bfloat162_rn(ya, yb);
          ow[c] = *reinterpret_cast<const uint32_t*>(&o2);
        }
        op[e] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) src[e] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (hf == 0) p.ws_l[R] = 0.f;
      if (tid == 0) p.tickets[bp] = 0;
    }
  }
}
#endif

template <int ST, int DIAG, bool O32>
__global__ void __launch_bounds__(NTHR, 2) vfa_main(const __grid_constant__ Params p) {
#if VFA_SM90
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024u - (sa(smem_raw) & 1023u)) & 1023u);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + off_bar<ST>());
  uint64_t* empt = full + ST;
  int* sV = reinterpret_cast<int*>(smem + off_cum<ST>());
  int* sN = sV + MAXB + 1;
  int* seg = reinterpret_cast<int*>(smem + off_seg<ST>());
  int* misc = reinterpret_cast<int*>(smem + off_misc<ST>());
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = __shfl_sync(FULL, tid >> 5, 0);
  const int pair = blockIdx.y, pairs = p.Hkv >> 1, B = p.B, segk = p.segk;
  if (tid < ST) {
    mbar_init(full + tid, NPROD);
    mbar_init(empt + tid, NCONS + (pvt_on<DIAG>() ? NPROD : 0));
  }
  if (tid < 2) {   // VFA3-PVT V^T barriers: vtf[2] (producer writes), vte[2] (consumer PV done)
    mbar_init(full + 2 * ST + tid, NPROD);
    mbar_init(full + 2 * ST + 2 + tid, NCONS);
  }
  if (tid >= NCONS) {   // rows 64..71 of the four V^T buffers: e4m3 ones (0x38); every byte equal, so no swizzle
    const int q = tid - NCONS;
    *reinterpret_cast<uint4*>(smem + off_vt<ST>() + (q >> 5) * VT_BYTES + 4096 + (q & 31) * 16) =
        make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  }
  if (warp == 0) {   // virtual key line: request b = n_b = seq_b + 8 keys then a dead gap of segk keys
    int carry = 0;
    for (int b0 = 0; b0 < B; b0 += 32) {
      const int b = b0 + lane;
      const int nb = (b < B) ? __ldg(p.seq_lens + b) + 8 : 0, wv = (b < B) ? nb + segk : 0;
      int incl = wv;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int v = __shfl_up_sync(FULL, incl, d);
        if (lane >= d) incl += v;
      }
      if (b < B) { sV[b] = carry + incl - wv; sN[b] = nb; }
      carry += __shfl_sync(FULL, incl, 31);
    }
    if (lane == 0) sV[B] = carry;
  }
  mbar_fence_init();
  __syncthreads();
  const long long T = (long long)sV[B] - segk;
  const int v0 = (int)((T * blockIdx.x) / gridDim.x), v1 = (int)((T * (blockIdx.x + 1)) / gridDim.x);
  if (tid == 0) {   // segment list (b, k0, k1) of this CTA's virtual range [v0, v1)
    int lo = 0, hi = B - 1;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (sV[mid] + sN[mid] > v0) hi = mid; else lo = mid + 1;
    }
    int ns = 0;
    for (int b = lo; b < B && sV[b] < v1 && ns < MAXSEG; ++b) {
      const int k0 = max(v0 - sV[b], 0), k1 = min(v1 - sV[b], sN[b]);
      if (k1 > k0) { seg[4 * ns] = b; seg[4 * ns + 1] = k0; seg[4 * ns + 2] = k1; ++ns; }
    }
    misc[0] = ns;
    misc[1] = 0;
  }
  __syncthreads();
  const int nseg = misc[0];
  if (p.pdl) pdl_wait();   // q8 and the block's cache rows come from the FA4 prologue
  if (tid >= NCONS) {
    if constexpr (pvt_on<DIAG>()) asm volatile("setmaxnreg.dec.sync.aligned.u32 " VFA_PVT_PREG ";\n" ::: "memory");
    else asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");
    producer<ST, O32, DIAG>(p, smem, seg, nseg, full, empt, tid - NCONS, pair);
    if (p.pdl) pdl_trigger();
  } else {
    if constexpr (pvt_on<DIAG>()) asm volatile("setmaxnreg.inc.sync.aligned.u32 " VFA_PVT_CREG ";\n" ::: "memory");
    else asm volatile("setmaxnreg.inc.sync.aligned.u32 104;\n" ::: "memory");
    consumer<ST, DIAG>(p, smem, seg, sN, nseg, full, empt, misc, tid, pair, pairs);
    if ((DIAG == 15 || (DIAG == 0 && false)) && tid < 128) bar_named(4, 256);   // FA16-PP: consume warpgroup 1's last hand-over
    if (p.pdl) pdl_trigger();
  }
#endif
}

template <int ST, int DIAG, bool O32>
static void vfa_go(const Params& p, dim3 grid, cudaStream_t stream, bool pdl) {
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vfa_main<ST, DIAG, O32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<ST>()));
    C10_CUDA_CHECK(cudaFuncSetAttribute(vfa_main<ST, DIAG, O32>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    init = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(NTHR);
  cfg.dynamicSmemBytes = smem_bytes<ST>();
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vfa_main<ST, DIAG, O32>, p));
}

template <int ST, bool O32>
static void vfa_diag(const Params& p, dim3 grid, cudaStream_t stream, bool pdl, int diag) {
#if VFA_LAB
  if constexpr (O32) {   // lab builds only: probe variants (32-bit offsets, the box's test caches)
    switch (diag) {
      case 1: vfa_go<ST, 1, true>(p, grid, stream, pdl); return;
      case 2: vfa_go<ST, 2, true>(p, grid, stream, pdl); return;
      case 3: vfa_go<ST, 3, true>(p, grid, stream, pdl); return;
      case 4: vfa_go<ST, 4, true>(p, grid, stream, pdl); return;
      case 5: vfa_go<ST, 5, true>(p, grid, stream, pdl); return;
      case 6: vfa_go<ST, 6, true>(p, grid, stream, pdl); return;
      default: break;
    }
  }
#endif
  (void)diag;
  vfa_go<ST, 0, O32>(p, grid, stream, pdl);
}

}  // namespace vfa

int64_t vfa_occupancy(int64_t stages) {
  int nb = 0;
  if (stages == 5) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vfa::vfa_main<5, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, vfa::smem_bytes<5>()));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, vfa::vfa_main<5, 0, true>, vfa::NTHR, vfa::smem_bytes<5>()));
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vfa::vfa_main<4, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, vfa::smem_bytes<4>()));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, vfa::vfa_main<4, 0, true>, vfa::NTHR, vfa::smem_bytes<4>()));
  }
  return nb;
}

void vfa_forward(torch::Tensor q8, torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor page_table, torch::Tensor k_cache,
                 torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor ws_o, torch::Tensor ws_l,
                 torch::Tensor tickets, torch::Tensor out, int64_t B, int64_t Hq, int64_t Hkv, int64_t page_shift, int64_t G,
                 int64_t segk, int64_t pdl, double qk_sm, int64_t stages, int64_t diag) {
  TORCH_CHECK(stages == 4 || stages == 5, "vfa: stages 4 or 5");
  TORCH_CHECK(Hq == 8 * Hkv && Hkv % 2 == 0 && Hkv > 0, "vfa: need Hq == 8*Hkv and even Hkv");
  TORCH_CHECK(B >= 1 && B <= vfa::MAXB, "vfa: B");
  TORCH_CHECK(G >= 1 && segk >= 0, "vfa: G / segk");
  TORCH_CHECK(q8.is_cuda() && q8.scalar_type() == at::kByte && q8.is_contiguous() && q8.numel() == B * 8 * Hq * 64, "vfa: q8");
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == B * 8 &&
                  qkv.size(1) >= (2 * Hq + 2 * Hkv) * 64 && qkv.stride(0) % 8 == 0, "vfa: qkv");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous() && seq_lens.numel() == B, "vfa: seq_lens");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.is_contiguous() && page_table.dim() == 2 && page_table.size(0) == B,
              "vfa: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && k_cache.is_contiguous() && v_cache.scalar_type() == at::kByte && v_cache.is_contiguous(),
              "vfa: caches");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat && v_scale.scalar_type() == at::kFloat, "vfa: scales");
  TORCH_CHECK(ws_o.scalar_type() == at::kFloat && ws_o.is_contiguous() && ws_o.numel() >= B * (Hkv / 2) * 128 * 64, "vfa: ws_o");
  TORCH_CHECK(ws_l.scalar_type() == at::kFloat && ws_l.is_contiguous() && ws_l.numel() >= B * (Hkv / 2) * 128, "vfa: ws_l");
  TORCH_CHECK(tickets.scalar_type() == at::kInt && tickets.is_contiguous() && tickets.numel() >= B * (Hkv / 2), "vfa: tickets");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * 8 * Hq * 64, "vfa: out");
  const c10::cuda::OptionalCUDAGuard guard(q8.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vfa::Params p;
  p.q8 = q8.data_ptr<uint8_t>();
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.page_table = page_table.data_ptr<int>();
  p.k_cache = k_cache.data_ptr<uint8_t>();
  p.v_cache = v_cache.data_ptr<uint8_t>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.ws_o = ws_o.data_ptr<float>();
  p.ws_l = ws_l.data_ptr<float>();
  p.tickets = tickets.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.ld_qkv = (int)qkv.stride(0); p.page_shift = (int)page_shift;
  p.pt_stride = (int)page_table.size(1); p.segk = (int)segk; p.pdl = (int)pdl;
  p.qk_sm = (float)qk_sm;
  const bool o32 = (k_cache.numel() < (1ll << 32)) && (v_cache.numel() < (1ll << 32));
  const dim3 grid((unsigned)G, (unsigned)(Hkv / 2));
  if (stages == 5) {
    if (o32) vfa::vfa_diag<5, true>(p, grid, stream, pdl != 0, (int)diag); else vfa::vfa_diag<5, false>(p, grid, stream, pdl != 0, (int)diag);
  } else {
    if (o32) vfa::vfa_diag<4, true>(p, grid, stream, pdl != 0, (int)diag); else vfa::vfa_diag<4, false>(p, grid, stream, pdl != 0, (int)diag);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

def _vfa_build(lab=False):
    if _VFA["mod"] is not None:
        return _VFA["mod"]
    from torch.utils.cpp_extension import load_inline
    src = VFA_CUDA_SRC.replace("#if VFA_LAB", "#if 1" if lab else "#if 0")
    tag = hashlib.sha1((VFA_CPP_SRC + src).encode()).hexdigest()[:10]
    t0 = time.time()
    _VFA["mod"] = load_inline(
        name=f"vfa_{tag}", cpp_sources=VFA_CPP_SRC, cuda_sources=src, functions=["vfa_forward", "vfa_occupancy"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _K3_BUILD_S["vfa"] = time.time() - t0
    return _VFA["mod"]

def _vfa_wanted(qkv, data, cfg, state):
    B = cfg["batch"]
    return (VFA_ENABLE and state.get("vfa_ok") and cfg["q_len"] == 8 and not cfg["window"] and cfg["rope_theta"] in (0.0, ROPE_THETA)
            and VFA_MIN_B <= B <= VFA_MAX_B and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
            and cfg["head_dim"] == 64 and str(qkv.device) in _VFA["ws"] and qkv.dtype == torch.bfloat16 and qkv.dim() == 2
            and qkv.stride(1) == 1 and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2 and data["page_table"].is_contiguous()
            and data["seq_lens"].dtype == torch.int32 and data["seq_lens"].is_contiguous()
            and data["cache_loc"].is_contiguous() and data["positions"].is_contiguous())

def _vfa_run(qkv, data, cfg, stages=None, segk=None, pdl=None, diag=None):
    B, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = B * 8
    dev = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    pdl = VFA_PDL if pdl is None else bool(pdl)
    q8 = torch.empty((tokens, hq * d), dtype=FP8, device=dev)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = K1FA4_PRO_BM or (4 if n_rows < 1024 else (8 if n_rows < 8192 else 16))
    theta = cfg["rope_theta"]
    cos_sin = k1fa4_cos_sin(theta, d, dev) if theta else qkv
    pdl_kw = {"launch_pdl": True} if pdl else {}
    _k1fa4_prologue[(triton.cdiv(n_rows, block_m),)](
        qkv, q8, data["positions"] if theta else qkv, cos_sin, k_cache.view(-1, hkv * d), v_cache.view(-1, hkv * d),
        data["cache_loc"], K1FA4_RMS_EPS, qkv.stride(0), n_rows, hq, hkv, d, triton.next_power_of_2(d), block_m,
        HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=K1FA4_PRO_WARPS, **pdl_kw)
    ws = _VFA["ws"][str(dev)]
    st = int(stages or VFA_STAGES)
    G = _VFA["sms"][str(dev)] * max(1, min(2, _VFA["occ"].get((st, str(dev)), 2)))
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=dev)
    _VFA["mod"].vfa_forward(q8.view(torch.uint8), qkv, data["seq_lens"], data["page_table"], k_cache.view(torch.uint8),
                            v_cache.view(torch.uint8), data["k_scale"], data["v_scale"], ws[0], ws[1], ws[2], out, B, hq, hkv,
                            cfg["page_size"].bit_length() - 1, G, int(VFA_SEGK if segk is None else segk), 1 if pdl else 0,
                            float(cfg["qk_scale_factor"] * cfg["sm_scale"]), st, int(VFA_DIAG if diag is None else diag))
    return out

def _vfa_try(qkv, data, cfg, state):
    if _vfa_wanted(qkv, data, cfg, state):
        try:
            return _vfa_run(qkv, data, cfg)
        except Exception as ex:
            _k3_disable(state, "vfa_ok", False, "VFA", ex)
    return None

def _vfa_ref_attention(data, k_new, v_new):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"]
    qkv = data["qkv"].float()
    q = _k1_rms_t(qkv[..., :hq * 64].reshape(B, L, hq, 64)).to(torch.bfloat16).float().clamp(-448.0, 448.0).to(FP8).float()
    q = q * cfg["qk_scale_factor"]
    ks, vs = data["k_scale"].float(), data["v_scale"].float()
    kf = data["k_cache"].view(torch.uint8).view(-1, hkv, 64)
    vf = data["v_cache"].view(torch.uint8).view(-1, hkv, 64)
    seq_l = data["seq_lens"].long()
    out = torch.empty((B, L, hq, 64), device=q.device)
    for b in range(B):
        seq = int(seq_l[b])
        idx = torch.arange(seq, device=q.device)
        rows = data["page_table"][b].long()[idx // page] * page + idx % page
        keys = torch.cat([kf[rows].view(FP8).float(), k_new[b].view(FP8).float()]) * ks
        vals = torch.cat([vf[rows].view(FP8).float(), v_new[b].view(FP8).float()]) * vs
        for j in range(L):
            att = torch.einsum("kgd,tkd->kgt", q[b, j].view(hkv, hq // hkv, 64), keys[:seq + j + 1]) * cfg["sm_scale"]
            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vals[:seq + j + 1]).reshape(hq, 64)
    gate = qkv[..., (hq + 2 * hkv) * 64:].reshape(B, L, hq, 64)
    return (_k1_rms_t(out) * torch.sigmoid(gate)).reshape(B, L, hq * 64)

def _vfa_self_test(dev):
    ws = _VFA["ws"][str(dev)]
    for (B, ctx, page, kvs, seed) in VFA_CASES:
        for rep in range(2):
            d = _k1_mk_case(dev, B, ctx, 8, False, page, kvs, seed)
            out = _vfa_run(d["qkv"].view(B * 8, -1), d, d["config"], pdl=bool(rep), stages=4 + rep if _VFA["occ"].get((5, str(dev)), 0) >= 2 else 4)
            torch.cuda.synchronize(dev)
            pos = d["seq_lens"].long()[:, None] + torch.arange(8, device=dev)
            slots = torch.gather(d["page_table"].long(), 1, pos // page) * page + pos % page
            kn, vn = (c.view(torch.uint8).view(-1, d["config"]["num_kv_heads"], 64)[slots] for c in (d["k_cache"], d["v_cache"]))
            tot, row = _k1_rel_rows(out, _vfa_ref_attention(d, kn, vn))
            vk = vp = -1.0
            if kvs == 1.0 and _K1_MOD is not None:
                d2 = _k1_mk_case(dev, B, ctx, 8, False, page, kvs, seed)
                vk = _k1_rel_rows(out, _k1_run(d2["qkv"].view(B * 8, -1), d2, d2["config"]))[0]
            pr = globals().get("p206_run")
            if pr is not None and kvs == 1.0 and B >= 16 and page == 1:
                try:
                    d3 = _k1_mk_case(dev, B, ctx, 8, False, page, kvs, seed)
                    vp = _k1_rel_rows(out, pr(dict(d3, qkv=d3["qkv"].view(B * 8, -1)), pdl=False))[0]
                except Exception as ex:
                    print(f"[vfa] (P206 comparison unavailable: {ex!r}"[:200] + ")", flush=True)
            clean = not (int(ws[2].abs().sum()) or float(ws[0].abs().sum()) or float(ws[1].abs().sum()))
            print(f"[vfa] self-test B={B} ctx={ctx} page={page} kvs={kvs:g} #{rep}: rel {tot:.2e} row {row:.2e} vs K1 {vk:.2e} "
                  f"vs P206 {vp:.2e} ws {'clean' if clean else 'DIRTY'}", flush=True)
            if not (bool(torch.isfinite(out.float()).all()) and clean and tot < 2e-2 and row < 6e-2 and vk < 6e-2 and vp < 2e-2):
                raise RuntimeError(f"self-test mismatch B={B} ctx={ctx} page={page}")
    d = _k1_mk_case(dev, 16, 1200, 8, False, 1, 1.0, 411)
    o1 = _vfa_run(d["qkv"].view(128, -1), d, d["config"]).clone()
    torch.cuda.synchronize(dev)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        og = _vfa_run(d["qkv"].view(128, -1), d, d["config"])
    gph.replay()
    torch.cuda.synchronize(dev)
    same = _k1_rel_rows(og, o1)[0] < 2e-3
    f = _k1_mk_case(dev, 16, 1200, 8, False, 1, 1.0, 411)
    gen = torch.Generator(device=dev).manual_seed(412)
    f["qkv"] = (torch.randn(f["qkv"].shape, generator=gen, device=dev) * 1.5).to(torch.bfloat16)
    for c in ("k_cache", "v_cache"):
        f[c] = torch.randn(f[c].shape, generator=gen, device=dev).to(FP8)
    f2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in f.items()}
    o2 = _vfa_run(f["qkv"].view(128, -1), f, f["config"]).clone()
    for k, v in f2.items():
        if isinstance(v, torch.Tensor):
            d[k].copy_(v)
    og.fill_(float("nan"))
    gph.replay()
    torch.cuda.synchronize(dev)
    same2 = _k1_rel_rows(og, o2)[0] < 2e-3 and bool(torch.isfinite(og).all())
    print(f"[vfa] graph check: replay identical {same}, after new inputs {same2}", flush=True)
    if not (same and same2):
        raise RuntimeError("graph check failed")

def _vfa_prepare(state, t0):
    state["vfa_ok"] = False
    if not VFA_ENABLE:
        return
    dev = torch.empty(0, device="cuda").device
    try:
        _pt1_wait("vfa")
        if _PT1["errors"].get("vfa"):
            raise RuntimeError(f"build failed: {_PT1['errors']['vfa']}")
        mod = _vfa_build()
        _VFA["sms"][str(dev)] = torch.cuda.get_device_properties(dev).multi_processor_count
        for st in (4, 5):
            _VFA["occ"][(st, str(dev))] = int(mod.vfa_occupancy(st))
        rows = VFA_MAX_B * 128
        _VFA["ws"][str(dev)] = (torch.zeros((rows, 64), dtype=torch.float32, device=dev), torch.zeros(rows, dtype=torch.float32, device=dev),
                                torch.zeros(VFA_MAX_B, dtype=torch.int32, device=dev))
        if VFA_SELF_TEST:
            _vfa_self_test(dev)
        state["vfa_ok"] = True
        print(f"[vfa] LIVE: verify full {VFA_MIN_B} <= B <= {VFA_MAX_B}, stages {VFA_STAGES} (occupancy {_VFA['occ']}), segk {VFA_SEGK}, "
              f"pdl {VFA_PDL}, enable {VFA_ENABLE}, build {_K3_BUILD_S.get('vfa', 0):.0f}s, prepare so far {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:
        print(f"[vfa] FAILED, the FA4 / K1 verify-full routes stay: {ex!r}"[:1500], flush=True)
        state["vfa_ok"] = False

from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate as _ad_rms_norm_gate  # noqa: E402

_ad_RMS_EPS = 1e-5

_ad_FP8 = torch.float8_e4m3fn

_ad_NORMS = {}

_ad_ROPE = {}

_ad_ROPE_POSITIONS = 1 << 21

_ad_R1_QK_FUSED_FP8_STAGE = True

_ad_NUM_SPLITS_OVERRIDE = {}

_ad_STAGE_LOC_MAX = 1 << 16

_ad_STAGE_LOC = {}  # device -> int64 arange(1, _ad_STAGE_LOC_MAX + 1): a constant, never input-derived

print(f"[r1] attention levers: qk_fused={_ad_R1_QK_FUSED_FP8_STAGE} splits_override={_ad_NUM_SPLITS_OVERRIDE}",
      flush=True)

def _ad_stage_loc(tokens, device):
    """Staging cache rows 1..tokens (int64). Built once, outside graph capture, like _ad_cos_sin."""
    cache = _ad_STAGE_LOC.get(device)
    if cache is None and not torch.cuda.is_current_stream_capturing():
        cache = _ad_STAGE_LOC[device] = torch.arange(1, _ad_STAGE_LOC_MAX + 1, dtype=torch.int64, device=device)
    if cache is not None and tokens <= cache.shape[0]:
        return cache[:tokens]
    return torch.arange(1, tokens + 1, dtype=torch.int64, device=device)

def _ad_qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _ad_NORMS:
        _ad_NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=_ad_RMS_EPS)
    return _ad_NORMS[key]

def _ad_cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ad_ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(_ad_ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ad_ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache

def _ad_r1_kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    if _ad_R1_QK_FUSED_FP8_STAGE and not scaled:
        # FusedQKNorm's Triton kernel with its _ad_FP8 KV-store epilogue (fused_qk_norm.py:105-121):
        # _ad_FP8 q out; normed/roped k (bf16-rounded) and raw v as _ad_FP8 rows of the "cache" it is
        # given. It skips cache row 0 (SGLang's padding slot), live in this problem, so it writes
        # staging rows 1..T and store_cache moves them exactly as the baseline does. Values equal
        # the baseline's bf16 -> fp8 casts (its +-448 clamp cannot fire: |k| <= 8 after the norm).
        width = hkv * d
        k_stage = torch.empty((tokens + 1, width), dtype=_ad_FP8, device=device)
        v_stage = torch.empty((tokens + 1, width), dtype=_ad_FP8, device=device)
        kv_store = (k_stage, v_stage, _ad_stage_loc(tokens, device))
        if cfg["rope_theta"]:
            q8, _ = _ad_qk_norm(hq, hkv, d).forward_triton(
                qkv, data["positions"], _ad_cos_sin(cfg["rope_theta"], d, device), kv_store)
        else:
            q8, _ = _ad_qk_norm(hq, hkv, d).forward_triton(qkv, None, None, kv_store)
        store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8),
                    k_cache.view(-1, width).view(torch.uint8), v_cache.view(-1, width).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q_fp8 = q8.view(tokens, hq, d)
    else:
        if cfg["rope_theta"]:
            q, k = _ad_qk_norm(hq, hkv, d)(qkv, data["positions"], _ad_cos_sin(cfg["rope_theta"], d, device))
        else:
            q, k = _ad_qk_norm(hq, hkv, d)(qkv)
        v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
        k = k.reshape(tokens, hkv * d)
        if scaled:
            k, v = k / data["k_scale"], v / data["v_scale"]
        # The pool keeps _ad_FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
        store_cache(k.to(_ad_FP8).view(torch.uint8), v.to(_ad_FP8).view(torch.uint8),
                    k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q_fp8 = q.view(tokens, hq, d).to(_ad_FP8)

    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    sched = data["scheduler_metadata"]
    splits = _ad_NUM_SPLITS_OVERRIDE.get(("sliding" if window else "full", batch, length), 0)
    if splits:
        descale["num_splits"] = splits  # (descale doubles as the extra-kwargs dict)
        sched = None  # the judge's metadata was built for num_splits=0; let FA3 rebuild it
    out = flash_attn_with_kvcache(
        q=q_fp8,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=sched,
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = _ad_rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=_ad_RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)

_AD = {"state": None}

def _ad_watchdog(limit_s):
    import os
    import threading
    done = threading.Event()
    def run():
        if not done.wait(limit_s):
            print(f"[vfa] prepare watchdog: no progress in {limit_s} s, exiting", flush=True)
            os._exit(17)
    threading.Thread(target=run, daemon=True).start()
    return done

def _p206_prepare(state, t0):
    state["p206"] = False   # the PR206 route is not in this base


def _ad_prepare(device):
    global _K1_MOD
    done = _ad_watchdog(280)
    t0 = time.time()
    th = threading.Thread(target=_pt1_target("vfa", _vfa_build), name="pt1-vfa", daemon=True)
    _PT1["threads"]["vfa"] = th
    th.start()
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    state = {
        "router": torch.empty((1,), dtype=torch.float32, device=device),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "stage_loc": torch.arange(1, STAGE_LOC_MAX + 1, dtype=torch.int64, device=device),
        "k1_ok": False,
    }
    del freqs
    if USE_K1:
        try:
            if _K1_MOD is None:
                _K1_MOD = _k1_build()
            state["k1_ok"] = _k1_prepare(device)
        except Exception as ex:  # noqa: BLE001
            print(f"[ad] K1 build / prepare FAILED, K1 off: {ex!r}", flush=True)
    state["v10qk"] = _v10_qk_prepare(state)
    _v18_fa4_prepare(state, t0)
    _ks1_prepare(state, t0)
    _p206_prepare(state, t0)
    _va1_prepare(state)
    _vfa_prepare(state, t0)
    torch.cuda.synchronize()
    done.set()
    print(f"[vfa] attention prepare {time.time() - t0:.1f}s: k1 {state['k1_ok']} fa4v6 {bool(state.get('fa4v6'))} "
          f"ks1 {bool(state.get('ks1'))} p206 {bool(state.get('p206'))} va1 {state.get('va1_ok')} vfa {state.get('vfa_ok')} "
          f"(stages {VFA_STAGES}, segk {VFA_SEGK}, pdl {VFA_PDL})", flush=True)
    return state

def _ad_attention(qkv, data, cfg, state):
    tokens = cfg["tokens"]
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    o = None
    ks1_ran = False
    if _ks1_wanted(qkv, data, cfg, state):
        try:
            o = _ks1_run(qkv, data, state).view(tokens, hq * d)
            ks1_ran = True
        except Exception as ex:
            _k3_disable(state, "ks1", set(), "KS1", ex)
            o = None
    if o is None and _vfa_wanted(qkv, data, cfg, state):
        o = _vfa_try(qkv, data, cfg, state)
        o = o.view(tokens, hq * d) if o is not None else None
    if o is None and not _va1_b1(cfg, state) and _v18_fa4_wanted(qkv, data, cfg, state):
        try:
            o = _v18_fa4_core(qkv, data).view(tokens, hq * d)
        except Exception as ex:
            _k3_disable(state, "fa4v6", False, "FA4 v6 route", ex)
            o = None
    if o is None and state["k1_ok"] and _k1_supported(data, cfg) and _k1_ready(cfg, qkv.device):
        try:
            o = _va1_try(qkv, data, cfg, state)
            o = (o if o is not None else _k1_run(qkv, data, cfg)).view(tokens, hq * d)
        except Exception as ex:
            _k3_disable(state, "k1_ok", False, "K1", ex)
            o = None
    if o is None:
        o = _r1_attention(qkv, data, state)
    return o

def kernel_fn(data):
    cfg = data["config"]
    if (cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]) != (16, 2, 64):
        return _ad_r1_kernel_fn(data)
    if _AD["state"] is None:
        if torch.cuda.is_current_stream_capturing():
            return _ad_r1_kernel_fn(data)
        _AD["state"] = _ad_prepare(data["qkv"].device)
    batch, length = cfg["batch"], cfg["q_len"]
    tokens = batch * length
    lcfg = dict(cfg)
    lcfg["tokens"] = tokens
    ldata = dict(data)
    ldata["config"] = lcfg
    o = _ad_attention(data["qkv"].reshape(tokens, -1), ldata, lcfg, _AD["state"])
    return o.view(batch, length, cfg["num_q_heads"] * cfg["head_dim"])


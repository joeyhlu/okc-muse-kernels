# okc: problem=muse_decode_layer
# nonce k3-vms2-1 (k3_gc_14 + VMS -DVMS_RO=1; tools/gen_k3_vms2.py)

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
USE_K2A = True
K1_ROUTE_MIN_BATCH = {('sliding', 1): 1, ('full', 1): 1, ('full', 8): 16}
K1_FOLD_MIN_BATCH = {("sliding", 1): 1, ("sliding", 8): 1, ("full", 1): 8, ("full", 8): 8}
K1_MIN_CHUNK_T = {("sliding", 1): 2, ("sliding", 8): 4, ("full", 1): 2, ("full", 8): 2}
K1_CPS_T = {("sliding", 1): 1, ("sliding", 8): 2, ("full", 1): 2, ("full", 8): 2}
K1_STAGES_T = {("sliding", 1): 4, ("sliding", 8): 3, ("full", 1): 4, ("full", 8): 3}
K2A_ROUTE = {1: 'v2', 8: 'v2', 16: 'v2', 32: 'marlin', 128: 'v2', 256: 'v2'}
K1_SELF_TEST = True

R1_QK_FUSED_FP8_STAGE = True
R1_LOCAL_ID_TABLE = True
R1_ALIGN_LOCAL_BUCKETS = True
R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS = 9
R1_POSTNORM_TRITON_ALL_T = True
R1_UNPOOL_INTO_ZEROS = True
R1_ROUTER_SPLIT_ALL_T = False
R1_HOPPER_GEMV_T1 = False
BLOCK_SIZE_M_OVERRIDE = {}
BLOCK_SIZE_M_RANGES = ((49, 1 << 30, 16),)


def _bm_override(num_tokens, default):
    if num_tokens in BLOCK_SIZE_M_OVERRIDE:
        return BLOCK_SIZE_M_OVERRIDE[num_tokens]
    for lo, hi, bm in BLOCK_SIZE_M_RANGES:
        if lo <= num_tokens <= hi:
            return bm
    return default
NUM_SPLITS_OVERRIDE = {}
STAGE_LOC_MAX = 1 << 16

print(f"[r1] layer levers: qk_fused={R1_QK_FUSED_FP8_STAGE} local_table={R1_LOCAL_ID_TABLE} "
      f"align_local={R1_ALIGN_LOCAL_BUCKETS} postnorm_triton={R1_POSTNORM_TRITON_ALL_T and _POSTNORM_KERNEL is not None} "
      f"unpool_zeros={R1_UNPOOL_INTO_ZEROS} router_split_all={R1_ROUTER_SPLIT_ALL_T} gemv_t1={R1_HOPPER_GEMV_T1} "
      f"bm_ranges={BLOCK_SIZE_M_RANGES} splits_override={NUM_SPLITS_OVERRIDE}", flush=True)
print(f"[k3] switches: USE_K1={USE_K1} USE_K2A={USE_K2A} K1=v7 route={K1_ROUTE_MIN_BATCH} fold={K1_FOLD_MIN_BATCH} min_chunk={K1_MIN_CHUNK_T} cps={K1_CPS_T} K2A=v2.2 route={K2A_ROUTE}",
      flush=True)


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    attn_residual_gate: torch.Tensor
    pre_ffn_norm: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    pooler: torch.Tensor
    pre_expert_norm: torch.Tensor
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_expert_norm: torch.Tensor
    unpooler: torch.Tensor
    post_ffn_norm: torch.Tensor
    ffn_residual_gate: torch.Tensor
    next_pre_attn_norm: torch.Tensor


class Config(TypedDict):
    batch: int
    q_len: int
    tokens: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float
    n_experts: int
    top_k: int
    local_experts: int


class Inputs(TypedDict):
    stream: torch.Tensor
    attn_input: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    local_start: torch.Tensor
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0



K1_TILE = 64
K1_STAGES = 4
K1_NEWK = 16
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



K2A_E_LOC = 32
K2A_D_HIDDEN = 4096
K2A_D_EXPERT = 4096
K2A_TOP_K = 8
K2A_POST_NORM_EPS = 1e-8
K2A_M_MAX = 320
K2A_META_HDR = 128
K2A_BITPOS = ((6, 7, 8, 15), (22, 23, 24, 31), (9, 10, 11, 18), (25, 26, 27, 2),
          (12, 13, 14, 21), (28, 29, 30, 5), (0, 1, 3, 4), (16, 17, 19, 20))
K2A_W_SHIFT = 256.0





K2A_BUCKETS = ((1, 4, 1), (5, 12, 8), (13, 24, 16), (25, 48, 32), (49, 192, 128), (193, K2A_M_MAX, 256))
K2A_V2_PLAN = {1: (0, 2, 1, 0), 8: (0, 1, 2, 0), 16: (0, 1, 1, 0), 32: (0, 2, 2, 0), 128: (1, 2, 2, 0), 256: (1, 2, 4, 0)}



def _k2a_reencode_scales(sf_u8: torch.Tensor):
    b = sf_u8.to(torch.int32)
    sign, ex, m = b >> 7, (b >> 3) & 15, b & 7
    nan = (ex == 15) & (m == 7)
    zero = (ex == 0) & (m == 0)
    sub = (ex == 0) & (m != 0)
    lut_e5 = torch.tensor([0, 1, 2, 2, 3, 3, 3, 3], dtype=torch.int32, device=b.device)
    lut_f = torch.tensor([0, 0, 0, 4, 0, 2, 4, 6], dtype=torch.int32, device=b.device)
    bp = ((ex + 3) << 3) | m
    bp = torch.where(sub, (lut_e5[m.long()] << 3) | lut_f[m.long()], bp)
    bp = torch.where(zero, torch.full_like(bp, 24), bp)
    return bp.to(torch.uint8), (sign == 1) & ~zero, zero, bool(nan.any())




def _k2a_tile_row_perm(n_rows: int, gate_up: bool, device=None) -> torch.Tensor:
    m = torch.arange(n_rows, device=device)
    if not gate_up:
        return m
    tile, rho = m // 16, m % 16
    return (rho // 8) * (n_rows // 2) + 8 * tile + (rho % 8)




def _k2a_bucket_of(m: int):
    for lo, hi, rep in K2A_BUCKETS:
        if lo <= m <= hi:
            return rep
    return None












K2A_PH_ROUTE, K2A_PH_G1, K2A_PH_G2, K2A_PH_FIN = 1, 2, 4, 8
K2A_HOT_LAUNCH_COMPILED = False
K2A_FUSE_MASKS = {0: (K2A_PH_ROUTE | K2A_PH_G1, K2A_PH_G2, K2A_PH_FIN), 1: (K2A_PH_ROUTE | K2A_PH_G1, K2A_PH_G2 | K2A_PH_FIN),
              2: (K2A_PH_ROUTE | K2A_PH_G1 | K2A_PH_G2 | K2A_PH_FIN,)}


def _k2a_v2_call(st, data, vid, fuse=1, ks_target=2, asg=0, force_atom=0, probe=0, hot_vid=-1, hot_t=0):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    lo = data["config"]["local_start"]
    if hot_t > 0 and hot_vid >= 0 and K2A_HOT_LAUNCH_COMPILED:
        launches = ((vid, K2A_PH_ROUTE | K2A_PH_G1, 0), (hot_vid, K2A_PH_ROUTE | K2A_PH_G1 | K2A_PH_G2, 1), (vid, K2A_PH_G2 | K2A_PH_FIN, 0))
    else:
        launches = tuple((vid, mk, 0) for mk in K2A_FUSE_MASKS[fuse])
    for v, mask, hm in launches:
        st["mod"].run(v, mask, x, ids, wts, st["w13q"], st["s13q"], st["w2q"], st["s2q"], st["gsc13"],
                      st["gsc2"], st["nw"], st["h"], st["inter"], st["y"], st["meta"], st["bar"], st["err"], out,
                      st["trace"], m, lo, st["grid"][v], ks_target, asg, force_atom, probe, hot_t, hm)
    return out






_K2A_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _k2a_dequant(packed, sf, gs):
    grid = torch.tensor(_K2A_GRID, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)
    return vals.view(*vals.shape[:-2], k)


def _k2a_ref_fp32(data, w):
    cfg = data["config"]
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    lo, hi = cfg["local_start"], cfg["local_start"] + cfg["local_count"]
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not lo <= e < hi:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _k2a_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :K2A_D_EXPERT]) * h[:, K2A_D_EXPERT:]
        y = (inter @ _k2a_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + K2A_POST_NORM_EPS)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].float().unsqueeze(1))
    return out.to(torch.bfloat16)


def _k2a_rel_err(a, b):
    a, b = a.float(), b.float()
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row


def _k2a_synth_inputs(bs, seed, device, dist="normal"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, K2A_D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=device)
    if dist == "normal":
        z = torch.randn((256,), generator=gen, device=device)
        u = torch.rand((256,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    else:
        scores = scores + torch.rand((256,), generator=gen, device=device).pow(6) * 4
    top = scores.topk(K2A_TOP_K, dim=-1)
    ids = top.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": K2A_D_HIDDEN, "d_expert": K2A_D_EXPERT, "n_experts": 256,
                   "local_start": 0, "local_count": K2A_E_LOC, "top_k": K2A_TOP_K, "sharding": "ep8"},
    }


def _k2a_self_test_inputs(bs, seed, device, dist="normal"):
    d = _k2a_synth_inputs(bs, seed, device, dist)
    if R1_LOCAL_ID_TABLE:
        ids = d["topk_ids"]
        d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
    return d


def _k2a_experts(x, local_ids, topk_weights, st):
    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    rep = _k2a_bucket_of(x.shape[0])
    vid, fuse, kst, asg, hv, ht = (tuple(K2A_V2_PLAN[rep]) + (-1, 0))[:6]
    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}
    return _k2a_v2_call(st, data, vid, fuse, kst, asg, 0, 0, hv, ht)


def _k2a_prepare(weights, t0):
    print("[kl1] K2A v2.2 is not in this file (legacy strip): a bucket K2A v3.3 cannot serve falls back to r1 Marlin", flush=True)
    return None


def _k2a_supported(cfg, tokens):
    rep = _k2a_bucket_of(tokens)
    return (rep is not None and K2A_ROUTE.get(rep) == "v2" and cfg["local_experts"] == K2A_E_LOC
            and cfg["top_k"] == K2A_TOP_K and cfg["n_experts"] == N_EXPERTS)

V8_ROUTER_ROUTE = {1: 'f32', 8: 'f32', 16: 'gemv', 32: 'stock', 128: 'kcat', 256: 'kcat'}
V8_ROUTER_PLACE = {1: 'side', 8: 'side_first', 16: 'side', 32: 'side', 128: 'side', 256: 'main'}
V8_DENSE_GEMV_T1 = True
V8_R1B = True
V8_TAIL_ROUTE = {1: 'sep', 8: 'sep', 16: 'cat', 32: 'cat', 128: 'cat', 256: 'cat'}
V9_TAIL_PDL = True
V8_TAIL = True

R1B_FIXUP = True
R1B_ROUTER_GEMV_MAX_M = 16
R1B_GEMV_ROUTES = {
    ("qkv_gate", 1): (1, 1, 1, 1),
    ("pooler", 1): (1, 1, 1, 1),
    ("unpooler", 1): (1, 1, 4, 1),
    ("router", 8): (8, 1, 1, 8),
    ("router", 32): (8, 1, 1, 8),
}
R1B_NWARP = 8
R1B_UNROLL = 4
R1B_SMEM_MAX = 40 * 1024
R1B_COUNTERS = 16384


def _r1b_bucket(layer: str, m: int):
    if m == 1 and layer != "router":
        return 1
    if m <= 8:
        return 8
    if layer == "router" and m <= R1B_ROUTER_GEMV_MAX_M:
        return 32
    return None


def _r1b_vec_of(w_dtype) -> int:
    return 4 if w_dtype == torch.float32 else 8


def _r1b_geometry(m: int, n: int, k: int, vec: int, route):
    mt, rpw, wk, split = route
    if R1B_NWARP % wk or k % split or m < 1:
        return None
    kslice = k // split
    chunk = 32 * vec * R1B_UNROLL
    rows_cta = (R1B_NWARP // wk) * rpw
    smem = mt * kslice * 2
    if kslice % chunk or n % rows_cta or n % 4 or smem > R1B_SMEM_MAX:
        return None
    gz = (m + mt - 1) // mt
    if gz > 65535 or (n // rows_cta) * gz > R1B_COUNTERS:
        return None
    return {"grid": (n // rows_cta, split, gz), "kslice": kslice, "chunk": chunk, "nchunk": kslice // chunk,
            "rows_cta": rows_cta, "smem": smem}


def _r1b_split_router(w32: torch.Tensor):
    hi = w32.to(torch.bfloat16)
    lo = (w32 - hi.float()).to(torch.bfloat16)
    return hi, lo


R1B_CPP_SRC = r"""
#include <torch/extension.h>
void r1b_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor part, torch::Tensor counters,
          int64_t M, int64_t N, int64_t K, int64_t mt, int64_t rpw, int64_t wk, int64_t split, int64_t fixup);
void r1b_reduce(torch::Tensor part, torch::Tensor out, int64_t split);
"""

R1B_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace r1b {

constexpr int NT = 256;       // threads per CTA
constexpr int NW = 8;         // warps per CTA
constexpr int U = 4;          // 16-B weight vectors per lane per row per chunk

template <typename WT> struct Vec;
template <> struct Vec<__nv_bfloat16> { static constexpr int N = 8; };
template <> struct Vec<float> { static constexpr int N = 4; };

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__device__ __forceinline__ void unpack_w(const float4 v, float (&f)[8], __nv_bfloat16*) {
  const uint32_t a = __float_as_uint(v.x), b = __float_as_uint(v.y), c = __float_as_uint(v.z), d = __float_as_uint(v.w);
  f[0] = bf_lo(a); f[1] = bf_hi(a); f[2] = bf_lo(b); f[3] = bf_hi(b);
  f[4] = bf_lo(c); f[5] = bf_hi(c); f[6] = bf_lo(d); f[7] = bf_hi(d);
}
__device__ __forceinline__ void unpack_w(const float4 v, float (&f)[4], float*) {
  f[0] = v.x; f[1] = v.y; f[2] = v.z; f[3] = v.w;
}
__device__ __forceinline__ void load_x(const __nv_bfloat16* p, float (&f)[8]) {
  const uint4 v = *reinterpret_cast<const uint4*>(p);
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
  f[4] = bf_lo(v.z); f[5] = bf_hi(v.z); f[6] = bf_lo(v.w); f[7] = bf_hi(v.w);
}
__device__ __forceinline__ void load_x(const __nv_bfloat16* p, float (&f)[4]) {
  const uint2 v = *reinterpret_cast<const uint2*>(p);
  f[0] = bf_lo(v.x); f[1] = bf_hi(v.x); f[2] = bf_lo(v.y); f[3] = bf_hi(v.y);
}
__device__ __forceinline__ void store_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }

template <typename WT, int MT, int RPW>
__global__ void __launch_bounds__(NT, 2) gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const WT* __restrict__ w, WT* __restrict__ out,
    float* __restrict__ part, int* __restrict__ counters,
    int M, int N, int K, int kslice, int WK, int fixup) {
  constexpr int VEC = Vec<WT>::N;
  constexpr int CH = 32 * VEC * U;                 // K elements per warp chunk (per row)
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // [MT][kslice]
  __shared__ float red[NW * RPW * MT];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wk = warp % WK, wr = warp / WK;
  const int rows_cta = (NW / WK) * RPW;
  const int row0 = blockIdx.x * rows_cta + wr * RPW;
  const int kbase = blockIdx.y * kslice;
  const int t0 = blockIdx.z * MT;
  const int mloc = min(MT, M - t0);
  const int nch = kslice / CH;

  float4 wv[RPW][U];
  int c = wk;
  if (c < nch) {
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const WT* p = w + (size_t)(row0 + r) * K + kbase + c * CH + lane * VEC;
#pragma unroll
      for (int u = 0; u < U; ++u) wv[r][u] = __ldcs(reinterpret_cast<const float4*>(p + u * 32 * VEC));
    }
  }

  const int vpr = kslice >> 3;                     // 16-B vectors per staged token row
  for (int i = tid; i < MT * vpr; i += NT) {
    const int t = i / vpr, v = i - t * vpr;
    uint4 val = make_uint4(0u, 0u, 0u, 0u);
    if (t < mloc) val = *reinterpret_cast<const uint4*>(x + (size_t)(t0 + t) * K + kbase + v * 8);
    reinterpret_cast<uint4*>(xs)[i] = val;
  }
  __syncthreads();

  float acc[RPW][MT];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[r][t] = 0.f;

  while (c < nch) {
    float4 cur[RPW][U];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int u = 0; u < U; ++u) cur[r][u] = wv[r][u];
    const int cn = c + WK;
    if (cn < nch) {                                // prefetch the next chunk
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const WT* p = w + (size_t)(row0 + r) * K + kbase + cn * CH + lane * VEC;
#pragma unroll
        for (int u = 0; u < U; ++u) wv[r][u] = __ldcs(reinterpret_cast<const float4*>(p + u * 32 * VEC));
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int kk = c * CH + u * 32 * VEC + lane * VEC;   // offset inside the K slice
      float wf[RPW][VEC];
#pragma unroll
      for (int r = 0; r < RPW; ++r) unpack_w(cur[r][u], wf[r], (WT*)nullptr);
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        float xf[VEC];
        load_x(xs + t * kslice + kk, xf);
#pragma unroll
        for (int r = 0; r < RPW; ++r)
#pragma unroll
          for (int j = 0; j < VEC; ++j) acc[r][t] = fmaf(wf[r][j], xf[j], acc[r][t]);
      }
    }
    c = cn;
  }

#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
      acc[r][t] = v;
    }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int t = 0; t < MT; ++t) red[(warp * RPW + r) * MT + t] = acc[r][t];
  }
  __syncthreads();

  const int split = gridDim.y;
  for (int i = tid; i < rows_cta * MT; i += NT) {
    const int t = i / rows_cta, rr = i - t * rows_cta;
    if (t >= mloc) continue;
    const int wr2 = rr / RPW, r2 = rr - wr2 * RPW;
    float s = 0.f;
    for (int q = 0; q < WK; ++q) s += red[((wr2 * WK + q) * RPW + r2) * MT + t];
    const size_t o = (size_t)(t0 + t) * N + (size_t)blockIdx.x * rows_cta + rr;
    if (split == 1) store_out(out + o, s);
    else part[(size_t)blockIdx.y * M * N + o] = s;
  }
  if (split == 1 || !fixup) return;

  __threadfence();
  __syncthreads();
  const int tile = blockIdx.z * gridDim.x + blockIdx.x;
  if (tid == 0) s_last = (atomicAdd(counters + tile, 1) == split - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int i = tid; i < rows_cta * MT; i += NT) {
    const int t = i / rows_cta, rr = i - t * rows_cta;
    if (t >= mloc) continue;
    const size_t o = (size_t)(t0 + t) * N + (size_t)blockIdx.x * rows_cta + rr;
    float s = 0.f;
    for (int q = 0; q < split; ++q) s += __ldcg(part + (size_t)q * M * N + o);
    store_out(out + o, s);
  }
  if (tid == 0) counters[tile] = 0;                // self-reset for the next launch / graph replay
}

template <typename OT>
__global__ void __launch_bounds__(256) reduce_kernel(const float* __restrict__ part, OT* __restrict__ out,
                                                     int MN, int split) {
  const int i = (blockIdx.x * 256 + threadIdx.x) * 4;
  if (i >= MN) return;
  float4 s = *reinterpret_cast<const float4*>(part + i);
  for (int q = 1; q < split; ++q) {
    const float4 v = *reinterpret_cast<const float4*>(part + (size_t)q * MN + i);
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  store_out(out + i, s.x); store_out(out + i + 1, s.y); store_out(out + i + 2, s.z); store_out(out + i + 3, s.w);
}

template <typename WT, int MT, int RPW>
static void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, torch::Tensor& part,
                   torch::Tensor& counters, int M, int N, int K, int wk, int split, int fixup) {
  const int kslice = K / split;
  const int rows_cta = (NW / wk) * RPW;
  const dim3 grid(N / rows_cta, split, (M + MT - 1) / MT);
  const size_t smem = (size_t)MT * kslice * 2;
  auto st = at::cuda::getCurrentCUDAStream();
  gemv_kernel<WT, MT, RPW><<<grid, NT, smem, st>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const WT*>(w.data_ptr()),
      reinterpret_cast<WT*>(out.data_ptr()), split > 1 ? part.data_ptr<float>() : nullptr,
      counters.data_ptr<int>(), M, N, K, kslice, wk, fixup);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace r1b

void r1b_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor part, torch::Tensor counters,
          int64_t M, int64_t N, int64_t K, int64_t mt, int64_t rpw, int64_t wk, int64_t split, int64_t fixup) {
  const bool f32 = w.scalar_type() == at::kFloat;
  const int vec = f32 ? 4 : 8;
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(0) == M && x.size(1) == K, "x");
  TORCH_CHECK(w.is_contiguous() && w.size(1) == K && w.size(0) >= N, "w");
  TORCH_CHECK(f32 || w.scalar_type() == at::kBFloat16, "w dtype");
  TORCH_CHECK(out.scalar_type() == w.scalar_type() && out.is_contiguous() && out.size(0) == M && out.size(1) == N, "out");
  TORCH_CHECK(r1b::NW % wk == 0 && split >= 1 && K % split == 0, "wk/split");
  const int64_t kslice = K / split, rows_cta = (r1b::NW / wk) * rpw;
  TORCH_CHECK(kslice % (32 * vec * r1b::U) == 0 && N % rows_cta == 0 && N % 4 == 0, "geometry");
  TORCH_CHECK(mt * kslice * 2 <= 40 * 1024, "smem");
  TORCH_CHECK(split == 1 || (part.scalar_type() == at::kFloat && part.numel() >= split * M * N), "part");
  TORCH_CHECK(counters.numel() >= (N / rows_cta) * ((M + mt - 1) / mt), "counters");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "align");
  const int m = (int)M, n = (int)N, k = (int)K, wk_ = (int)wk, sp = (int)split, fx = (int)fixup;
  if (f32) {
    TORCH_CHECK(mt == 8 && (rpw == 1 || rpw == 2), "fp32 variant");
    if (rpw == 1) r1b::launch<float, 8, 1>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
    else r1b::launch<float, 8, 2>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
  } else {
    TORCH_CHECK((mt == 1 || mt == 8) && (rpw == 1 || rpw == 2), "bf16 variant");
    if (mt == 1 && rpw == 1) r1b::launch<__nv_bfloat16, 1, 1>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
    else if (mt == 1) r1b::launch<__nv_bfloat16, 1, 2>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
    else if (rpw == 1) r1b::launch<__nv_bfloat16, 8, 1>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
    else r1b::launch<__nv_bfloat16, 8, 2>(x, w, out, part, counters, m, n, k, wk_, sp, fx);
  }
}

void r1b_reduce(torch::Tensor part, torch::Tensor out, int64_t split) {
  const int MN = (int)out.numel();
  TORCH_CHECK(MN % 4 == 0 && part.numel() >= split * MN && out.is_contiguous(), "reduce");
  auto st = at::cuda::getCurrentCUDAStream();
  const int blocks = (MN / 4 + 255) / 256;
  if (out.scalar_type() == at::kFloat)
    r1b::reduce_kernel<float><<<blocks, 256, 0, st>>>(part.data_ptr<float>(), out.data_ptr<float>(), MN, (int)split);
  else
    r1b::reduce_kernel<__nv_bfloat16><<<blocks, 256, 0, st>>>(
        part.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), MN, (int)split);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _r1b_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((R1B_CPP_SRC + R1B_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"r1b_gemv_{tag}",
        cpp_sources=R1B_CPP_SRC,
        cuda_sources=R1B_CUDA_SRC,
        functions=["r1b_gemv", "r1b_reduce"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
        verbose=True,
    )
    print(f"[r1b] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


def _r1b_router_split(x, state, mode):
    if mode == "kcat":
        return torch.mm(torch.cat((x, x), dim=1), state["router_cat"].t(), out_dtype=torch.float32)
    if mode == "ncat":
        n = state["router_ncat"].shape[0] // 2
        y = torch.mm(x, state["router_ncat"].t(), out_dtype=torch.float32)
        return y[:, :n] + y[:, n:]
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _r1b_gemv(state, x, w, n, route):
    mod = state["mod"]
    m, k = x.shape
    mt, rpw, wk, split = route
    out = torch.empty((m, n), dtype=w.dtype, device=x.device)
    part = state["dummy"] if split == 1 else torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    fix = 1 if R1B_FIXUP else 0
    mod.r1b_gemv(x, w, out, part, state["counters"], m, n, k, mt, rpw, wk, split, fix)
    if split > 1 and not fix:
        mod.r1b_reduce(part, out, split)
    return out


def _r1b_gemv_route(state, layer, m, n, k):
    if state.get("mod") is None:
        return None
    b = _r1b_bucket(layer, m)
    if b is None:
        return None
    route = state["routes"].get((layer, b))
    if route is None:
        return None
    w = state[layer]
    if _r1b_geometry(m, n, k, _r1b_vec_of(w.dtype), route) is None:
        return None
    return route


def _r1b_aligned(x):
    return x.is_contiguous() and x.data_ptr() % 16 == 0


def _r1b_test_input(m, k, seed, device, outliers=False, probe="force"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device=device)
    if outliers:
        idx = torch.randperm(k, generator=gen, device=device)[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device=device) < 1 / 8
    if probe == "force" or (probe == "judge" and m >= 8):
        probes[torch.randint(m, (1,), generator=gen, device=device)] = True
    if probe == "none":
        probes[:] = False
    cols = torch.randint(k, (m,), generator=gen, device=device)
    signs = torch.randint(2, (m,), generator=gen, device=device) * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device=device)
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16), probes


def _r1b_errors(out, x, w, probes):
    ref = x.double() @ w.double().T
    ref = ref.to(out.dtype).double()
    d = out.double() - ref
    tot = float(d.norm() / ref.norm().clamp(min=1e-30))
    rn = ref.norm(dim=-1)
    row = float((d.norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
    pr = 0.0
    if bool(probes.any()):
        pr = float((d[probes].norm(dim=-1) / rn[probes].clamp(min=1e-30)).max())
    return tot, row, pr


def _r1b_self_test(state, weights):
    dev = weights["router"].device
    ok_routes = {}
    for (layer, b), route in state["routes"].items():
        n_full, k = weights[layer].shape
        ms = {1: (1,), 8: (8, 5, 1), 32: (9, 16, 32)}[b]
        good = True
        for m in ms:
            n = 1024 if (layer == "unpooler" and m <= 8) else n_full
            if _r1b_geometry(m, n, k, _r1b_vec_of(weights[layer].dtype), route) is None:
                print(f"[r1b] route {layer}/{b} {route}: _r1b_geometry rejects m={m} n={n} k={k}", flush=True)
                good = False
                break
            for outl, probe in ((False, "none"), (False, "force"), (True, "none"), (True, "judge")):
                x, probes = _r1b_test_input(m, k, 1000 + 7 * m + (1 if outl else 0), dev, outl, probe)
                try:
                    tmp = dict(state)
                    tmp["routes"] = {(layer, b): route}
                    got = _r1b_gemv(tmp, x, state[layer], n, route)
                    got2 = _r1b_gemv(tmp, x, state[layer], n, route)
                    torch.cuda.synchronize()
                    tot, row, pr = _r1b_errors(got, x, weights[layer][:n], probes)
                    same = bool(torch.equal(got, got2))
                except Exception as ex:
                    print(f"[r1b] route {layer}/{b} m={m}: EXC {ex!r}", flush=True)
                    good = False
                    break
                tol = 1e-5 if layer == "router" else 4e-3
                ptol = 1e-6 if layer == "router" else 1e-6
                fin = bool(torch.isfinite(got.float()).all())
                print(f"[r1b] self-test {layer}/{b} {route} m={m} outl={outl} probe={probe}: rel {tot:.2e} row {row:.2e} "
                      f"probe {pr:.1e} repeat_eq={same} finite={fin}", flush=True)
                if not (fin and same and tot < tol and row < 5 * tol and pr <= ptol):
                    good = False
            if not good:
                break
        if good:
            ok_routes[(layer, b)] = route
        else:
            print(f"[r1b] route {layer}/{b} DROPPED -> cuBLAS", flush=True)
    return ok_routes


def _r1b_self_test_router_split(state, weights):
    w = weights["router"]
    dev = w.device
    for m in (1, 8, 9, 33, 128):
        for probe in ("none", "force"):
            x, probes = _r1b_test_input(m, w.shape[1], 2000 + m, dev, outliers=(m == 9), probe=probe)
            got = _r1b_router_split(x, state, state["router_mode"])
            torch.cuda.synchronize()
            tot, row, pr = _r1b_errors(got, x, w, probes)
            print(f"[r1b] self-test router_split/{state['router_mode']} m={m} probe={probe}: rel {tot:.2e} "
                  f"row {row:.2e} probe {pr:.1e}", flush=True)
            if not (bool(torch.isfinite(got).all()) and tot < 2e-5 and row < 1e-4 and pr < 1e-4):
                return False
    return True




def _v8_prepare(weights):
    if not V8_R1B:
        return None
    t0 = time.time()
    dev = weights["router"].device
    st = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler", "router")}
    hi, lo = _r1b_split_router(st["router"])
    st["router_cat"] = torch.cat((hi, lo), dim=1).contiguous()
    st["router_mode"] = "kcat"
    try:
        if not _r1b_self_test_router_split(st, weights):
            raise RuntimeError("tolerance")
    except Exception as ex:
        print(f"[v8] router kcat FAILED -> stock router: {ex!r}", flush=True)
        st["router_mode"] = None
    st["mod"] = None
    st["routes"] = {}
    st["counters"] = torch.zeros((R1B_COUNTERS,), dtype=torch.int32, device=dev)
    st["dummy"] = torch.zeros((4,), dtype=torch.float32, device=dev)
    try:
        st["mod"] = _V10_MOD if _V10_MOD is not None else _r1b_build()
        st["routes"] = dict(R1B_GEMV_ROUTES)
    except Exception as ex:
        print("[v8] R1B GEMV build FAILED -> cuBLAS. Tail:\n" + str(ex)[-8000:], flush=True)
        st["mod"] = None
    if st["mod"] is not None:
        try:
            st["routes"] = _r1b_self_test(st, weights)
        except Exception as ex:
            print(f"[v8] R1B self-test crashed -> cuBLAS: {ex!r}", flush=True)
            st["mod"] = None
            st["routes"] = {}
        st["counters"].zero_()
    torch.cuda.synchronize()
    print(f"[v8] R1B ready in {time.time() - t0:.1f}s: router_mode={st['router_mode']} gemv routes "
          f"{sorted(f'{a}/{b}' for a, b in st['routes'])}; router route {V8_ROUTER_ROUTE} place {V8_ROUTER_PLACE};"
          f" dense gemv T1 {V8_DENSE_GEMV_T1}", flush=True)
    return st


def _v8_gemm(x, layer, w, state):
    st = state.get("r1b")
    if V8_DENSE_GEMV_T1 and st is not None and x.shape[0] == 1:
        n, k = w.shape
        route = _r1b_gemv_route(st, layer, 1, n, k)
        if route is not None and _r1b_aligned(x):
            return _r1b_gemv(st, x, w, n, route)
    return _gemm(x, w)


def _v8_router_mode(tokens, state):
    st = state.get("r1b")
    rep = _k2a_bucket_of(tokens)
    mode = V8_ROUTER_ROUTE.get(rep, "stock") if st is not None else "stock"
    if mode == "f32":
        if tokens > ROUTER_FP32_MAX_TOKENS:
            mode = "gemv"
        elif state.get("v8") is None or V8_TAIL_ROUTE.get(rep) != "sep":
            mode = "stock"
    if tokens <= ROUTER_FP32_MAX_TOKENS and mode in ("gemv", "kcat"):
        mode = "stock"
    if mode == "gemv" and _r1b_gemv_route(st, "router", tokens, N_EXPERTS, HIDDEN) is None:
        mode = "kcat"
    if mode == "kcat" and not st["router_mode"]:
        mode = "stock"
    place = V8_ROUTER_PLACE.get(rep, "side") if mode != "stock" else "side"
    return mode, place


def _v8_logits(mode, ffn_input, router_input, state):
    st = state.get("r1b")
    if mode == "f32":
        part = torch.empty((8, router_input.shape[0], N_EXPERTS), dtype=torch.float32, device=router_input.device)
        kind = VF_ROUTER.get(1 if router_input.shape[0] == 1 else 8, "r32")
        if kind != "r32" and _VF["r8"] and _SBB1["mod"] is not None:
            _SBB1["mod"].vf_router32(router_input, state["router"], part, {"lean4": 1, "lean5": 2}.get(kind, 0))
        else:
            state["v8"]["mod"].v8_router32(router_input, state["router"], part)
        return part
    if mode == "gemv" and _r1b_aligned(ffn_input):
        route = _r1b_gemv_route(st, "router", ffn_input.shape[0], N_EXPERTS, HIDDEN)
        return _r1b_gemv(st, ffn_input, st["router"], N_EXPERTS, route)
    if mode in ("gemv", "kcat") and st["router_mode"]:
        return _r1b_router_split(ffn_input, st, "kcat")
    if router_input is not None:
        return F.linear(router_input, state["router"])
    logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
    logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
    return logits


V8_TAIL_CPP_SRC = r"""
#include <torch/extension.h>
void v8_tail(torch::Tensor y, torch::Tensor lg, int64_t la_off, int64_t lb_off, int64_t l_ld, int64_t ns,
             int64_t sstride, torch::Tensor nw, torch::Tensor bias, torch::Tensor lstart, torch::Tensor xn,
             torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double eps, double mult, double rscale, double deps,
             int64_t pdl);
void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part);
"""

V8_TAIL_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math.h>

namespace v8 {
constexpr int D = 4096;      // pooler / expert width
constexpr int NE = 256;      // experts
constexpr int TK = 8;        // top-k
constexpr int NT = 512;      // threads: 8 elements of the row each; threads 0..255 also own one expert

// One CTA per token (warp 0 also does the routing). y: [T, y_ld] (FP32 -> rounded to BF16 first, or BF16), columns 0..4095 = pooler output.
// (F.rms_norm on the BF16 pooler output with the BF16 effective weight), tw/tid = FusedMuseSparkRouting's top-8
template <bool Y32, int NSMAX>   // NSMAX: logits partial slices held (8: T <= 8 FP32 router, else 1)
__global__ void __launch_bounds__(NT) tail_kernel(
    const void* __restrict__ y, long y_ld, const float* __restrict__ la, const float* __restrict__ lb, long l_ld,
    int ns, long sstride, const __nv_bfloat16* __restrict__ nw, const float* __restrict__ bias, const int* __restrict__ lstart,
    __nv_bfloat16* __restrict__ xn, float* __restrict__ tw, int* __restrict__ tid, int* __restrict__ lid,
    float eps, float mult, float rscale, float deps) {
  const int t = blockIdx.x, tx = threadIdx.x, lane = tx & 31, warp = tx >> 5;
  __shared__ float red[NT / 32];
  const uint4 wq = *(reinterpret_cast<const uint4*>(nw) + tx);
  float4 c0 = make_float4(0.f, 0.f, 0.f, 0.f), c1 = c0;
  if (warp == 0) {
    const float4* pbias = reinterpret_cast<const float4*>(bias) + lane * 2;
    c0 = pbias[0];
    c1 = pbias[1];
  }
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // v4.7: K2A's K1 may launch now
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  float4 qa[NSMAX][2];
  float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;
  if (warp == 0) {
#pragma unroll
    for (int q = 0; q < NSMAX; ++q) {
      if (q < ns) {
        const float4* pq = reinterpret_cast<const float4*>(la + q * sstride + (long)t * l_ld) + lane * 2;
        qa[q][0] = pq[0];
        qa[q][1] = pq[1];
      } else {
        qa[q][0] = make_float4(0.f, 0.f, 0.f, 0.f);
        qa[q][1] = qa[q][0];
      }
    }
    if (lb != nullptr) {
      const float4* pb = reinterpret_cast<const float4*>(lb + (long)t * l_ld) + lane * 2;
      qb0 = pb[0];
      qb1 = pb[1];
    }
  }
  float v[8];
  if (Y32) {
    const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(y) + (long)t * y_ld) + tx * 2;
    const float4 a = p[0], c = p[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16(v[i]));
  } else {
    const uint4 q = *(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
  }
  float s[8], b[8];
  if (warp == 0) {
    float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
    for (int q = 1; q < NSMAX; ++q) {   // FP32 split-K partials summed in a fixed order (zeros past ns)
      a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
      a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
    }
    a0.x += qb0.x; a0.y += qb0.y; a0.z += qb0.z; a0.w += qb0.w; a1.x += qb1.x; a1.y += qb1.y; a1.z += qb1.z; a1.w += qb1.w;
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * mult));
      b[j] = s[j] + bs[j];
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / (float)D + eps);
  {
    const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq);
    __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((v[i] * rstd) * __bfloat162float(w[i]));
    *(reinterpret_cast<uint4*>(xn + (long)t * D) + tx) = *reinterpret_cast<const uint4*>(o8);
  }
  if (warp != 0) return;
  int sel_id[TK];
  float sel_s[TK];
#pragma unroll
  for (int k = 0; k < TK; ++k) {
    float bv = b[0], sv = s[0];
    int bj = 0;
#pragma unroll
    for (int j = 1; j < 8; ++j)
      if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
    int bi = lane * 8 + bj;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      const float os = __shfl_xor_sync(0xffffffffu, sv, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; sv = os; }
    }
    sel_id[k] = bi;
    sel_s[k] = sv;
    if ((bi >> 3) == lane) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (j == (bi & 7)) b[j] = -INFINITY;
    }
  }
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < TK; ++i) sum += sel_s[i];
    const float inv = rscale / (sum + deps);
    const int ls = lstart[0];
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      tw[t * TK + k] = sel_s[k] * inv;
      tid[t * TK + k] = sel_id[k];
      const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
      lid[t * TK + k] = rel < 32 ? rel : -1;
    }
  }
}

template <bool Y32, int NSMAX>
__global__ void __launch_bounds__(NT + 32) tail_kernel2(
    const void* __restrict__ y, long y_ld, const float* __restrict__ la, const float* __restrict__ lb, long l_ld,
    int ns, long sstride, const __nv_bfloat16* __restrict__ nw, const float* __restrict__ bias, const int* __restrict__ lstart,
    __nv_bfloat16* __restrict__ xn, float* __restrict__ tw, int* __restrict__ tid, int* __restrict__ lid,
    float eps, float mult, float rscale, float deps) {
  const int t = blockIdx.x, tx = threadIdx.x, lane = tx & 31, warp = tx >> 5;
  __shared__ float red[NT / 32];
  if (warp == NT / 32) {
    const float4* pbias = reinterpret_cast<const float4*>(bias) + lane * 2;
    const float4 c0 = pbias[0], c1 = pbias[1];
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // v4.7: K2A's K1 may launch now
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    float4 qa[NSMAX][2];
    float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;
#pragma unroll
    for (int q = 0; q < NSMAX; ++q) {
      if (q < ns) {
        const float4* pq = reinterpret_cast<const float4*>(la + q * sstride + (long)t * l_ld) + lane * 2;
        qa[q][0] = pq[0];
        qa[q][1] = pq[1];
      } else {
        qa[q][0] = make_float4(0.f, 0.f, 0.f, 0.f);
        qa[q][1] = qa[q][0];
      }
    }
    if (lb != nullptr) {
      const float4* pb = reinterpret_cast<const float4*>(lb + (long)t * l_ld) + lane * 2;
      qb0 = pb[0];
      qb1 = pb[1];
    }
    float s[8], b[8];
    float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
    for (int q = 1; q < NSMAX; ++q) {   // FP32 split-K partials summed in a fixed order (zeros past ns)
      a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
      a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
    }
    a0.x += qb0.x; a0.y += qb0.y; a0.z += qb0.z; a0.w += qb0.w; a1.x += qb1.x; a1.y += qb1.y; a1.z += qb1.z; a1.w += qb1.w;
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * mult));
      b[j] = s[j] + bs[j];
    }
    int sel_id[TK];
    float sel_s[TK];
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      float bv = b[0], sv = s[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 8; ++j)
        if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
      const unsigned u = __float_as_uint(bv + 0.f);   // -0 -> +0: equal floats get equal keys
      const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
      const unsigned mx = __reduce_max_sync(0xffffffffu, key);
      const int owner = __ffs(__ballot_sync(0xffffffffu, key == mx)) - 1;
      sel_id[k] = __shfl_sync(0xffffffffu, lane * 8 + bj, owner);
      sel_s[k] = __shfl_sync(0xffffffffu, sv, owner);
      if (lane == owner) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (j == bj) b[j] = -INFINITY;
      }
    }
    if (lane == 0) {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < TK; ++i) sum += sel_s[i];
      const float inv = rscale / (sum + deps);
      const int ls = lstart[0];
#pragma unroll
      for (int k = 0; k < TK; ++k) {
        tw[t * TK + k] = sel_s[k] * inv;
        tid[t * TK + k] = sel_id[k];
        const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
        lid[t * TK + k] = rel < 32 ? rel : -1;
      }
    }
    return;
  }
  const uint4 wq = *(reinterpret_cast<const uint4*>(nw) + tx);
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  float v[8];
  if (Y32) {
    const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(y) + (long)t * y_ld) + tx * 2;
    const float4 a = p[0], c = p[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16(v[i]));
  } else {
    const uint4 q = *(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  asm volatile("bar.sync 1, %0;\n" ::"r"(NT) : "memory");   // the 16 norm warps only
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / (float)D + eps);
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq);
  __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((v[i] * rstd) * __bfloat162float(w[i]));
  *(reinterpret_cast<uint4*>(xn + (long)t * D) + tx) = *reinterpret_cast<const uint4*>(o8);
}

template <int MT>
__global__ void __launch_bounds__(256) router32_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                       float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024, KS4 = KS / 4;
  __shared__ float4 xs[MT * KS4];
  const int sl = blockIdx.y, row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const int k0 = sl * KS;
  float4 wv[8];
  const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
#pragma unroll
  for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + i * 32 + lane);
  for (int idx = threadIdx.x; idx < MT * KS4; idx += 256) {
    const int j = idx / KS4, c = idx - j * KS4;
    xs[idx] = (j < m) ? reinterpret_cast<const float4*>(x + (long)j * K + k0)[c] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  __syncthreads();
  float acc[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = i * 32 + lane;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      const float4 xv = xs[j * KS4 + c];
      acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
      acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
      acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
      acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    float a = acc[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0 && j < m) part[((long)sl * m + j) * 256 + row] = a;
  }
}
}  // namespace v8

void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part) {
  const int m = (int)x.size(0);
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && part.scalar_type() == at::kFloat,
              "v8_router32: dtypes");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && part.is_contiguous(), "v8_router32: layout");
  TORCH_CHECK(x.size(1) == 8192 && w.size(0) == 256 && w.size(1) == 8192 && m >= 1 && m <= 8
              && part.numel() == 8L * m * 256, "v8_router32: shapes");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "v8_router32: alignment");
  auto st = at::cuda::getCurrentCUDAStream();
  if (m == 1)
    v8::router32_kernel<1><<<dim3(32, 8), 256, 0, st>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else
    v8::router32_kernel<8><<<dim3(32, 8), 256, 0, st>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v8_tail(torch::Tensor y, torch::Tensor lg, int64_t la_off, int64_t lb_off, int64_t l_ld, int64_t ns,
             int64_t sstride, torch::Tensor nw, torch::Tensor bias, torch::Tensor lstart, torch::Tensor xn,
             torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double eps, double mult, double rscale, double deps,
             int64_t pdl) {
  const int T = (int)y.size(0);
  TORCH_CHECK(y.stride(1) == 1 && lg.stride(lg.dim() - 1) == 1 && y.size(1) >= v8::D && ns >= 1 && sstride % 4 == 0,
              "v8_tail: layout");
  TORCH_CHECK(lg.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat, "v8_tail: dtypes");
  TORCH_CHECK(nw.scalar_type() == at::kBFloat16 && nw.numel() == v8::D, "v8_tail: norm weight");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(nw.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(xn.data_ptr()) % 16 == 0, "v8_tail: alignment");
  if (T == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  const float* base = lg.data_ptr<float>();
  const float* la = base + la_off;
  const float* lb = lb_off >= 0 ? base + lb_off : nullptr;
  TORCH_CHECK(ns <= 8, "v8_tail: at most 8 partial slices");
  TORCH_CHECK(l_ld % 4 == 0 && la_off % 4 == 0 && (lb_off < 0 || lb_off % 4 == 0)
              && reinterpret_cast<uintptr_t>(base) % 16 == 0 && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0,
              "v8_tail: logits / bias alignment");
  const long y_ld = y.stride(0);
  auto* xp = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr());
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = (pdl & 1) ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(T);
  lc.blockDim = dim3((pdl & 2) ? v8::NT + 32 : v8::NT);   // TL1
  lc.dynamicSmemBytes = 0;
  lc.stream = st;
  lc.attrs = at;
  lc.numAttrs = 1;
  if (y.scalar_type() == at::kFloat)
    { if (ns > 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, ((pdl & 2) ? v8::tail_kernel2<true, 8> : v8::tail_kernel<true, 8>), y.data_ptr(), y_ld, la, lb, (long)l_ld, (int)ns,
        (long)sstride, wp, (const float*)bias.data_ptr<float>(), (const int*)lstart.data_ptr<int>(), xp,
        tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(), (float)eps, (float)mult, (float)rscale,
        (float)deps));
      else C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, ((pdl & 2) ? v8::tail_kernel2<true, 1> : v8::tail_kernel<true, 1>), y.data_ptr(), y_ld, la, lb, (long)l_ld, (int)ns,
        (long)sstride, wp, (const float*)bias.data_ptr<float>(), (const int*)lstart.data_ptr<int>(), xp,
        tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(), (float)eps, (float)mult, (float)rscale,
        (float)deps)); }
  else
    { if (ns > 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, ((pdl & 2) ? v8::tail_kernel2<false, 8> : v8::tail_kernel<false, 8>), y.data_ptr(), y_ld, la, lb, (long)l_ld, (int)ns,
        (long)sstride, wp, (const float*)bias.data_ptr<float>(), (const int*)lstart.data_ptr<int>(), xp,
        tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(), (float)eps, (float)mult, (float)rscale,
        (float)deps));
      else C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, ((pdl & 2) ? v8::tail_kernel2<false, 1> : v8::tail_kernel<false, 1>), y.data_ptr(), y_ld, la, lb, (long)l_ld, (int)ns,
        (long)sstride, wp, (const float*)bias.data_ptr<float>(), (const int*)lstart.data_ptr<int>(), xp,
        tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(), (float)eps, (float)mult, (float)rscale,
        (float)deps)); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _v8_tail_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((V8_TAIL_CPP_SRC + V8_TAIL_CUDA_SRC).encode()).hexdigest()[:10]
    return load_inline(name=f"k3_v8_tail_{tag}", cpp_sources=V8_TAIL_CPP_SRC, cuda_sources=V8_TAIL_CUDA_SRC,
                       functions=["v8_tail", "v8_router32"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def _v8_tail(y, logits, state, data):
    v8 = state["v8"]
    tokens = y.shape[0]
    dev = y.device
    xn = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    tw = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    lid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    ns, sstride = 1, 0
    if logits is None:
        lg, la, lb, l_ld = y, D_EXPERT, D_EXPERT + N_EXPERTS, y.stride(0)
    elif logits.dim() == 3:
        lg, la, lb, l_ld = logits, 0, -1, logits.stride(1)
        ns, sstride = logits.shape[0], logits.stride(0)
    else:
        lg, la, lb, l_ld = logits, 0, -1, logits.stride(0)
    v8["mod"].v8_tail(y, lg, la, lb, l_ld, ns, sstride, v8["nw"], v8["bias"], data["local_start"], xn, tw, tid, lid,
                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15, (1 if V9_TAIL_PDL else 0) | (2 if TL1_TAIL2 else 0))
    return xn, tw, tid, lid


TIE1_GAP = 2e-5


def _tie1_compare(tid, tw, lid, ref_tid, ref_tw, ref_lid, l64, bias, lsel, table, ls):
    s64 = torch.sigmoid(l64.double() * 0.1767766952966369) + bias.double()
    T, E = s64.shape
    tid, rt = tid.long(), ref_tid.long()
    if tid.shape != rt.shape or not bool(((tid >= 0) & (tid < E)).all()):
        return False, float("inf"), False
    chosen = torch.zeros((T, E), dtype=torch.bool, device=s64.device).scatter_(1, tid, True)
    inf = torch.full((), float("inf"), dtype=s64.dtype, device=s64.device)
    cmin = torch.where(chosen, s64, inf).min(1).values
    umax = torch.where(chosen, -inf, s64).max(1).values
    ok = bool((chosen.sum(1) == tid.shape[1]).all()) and bool((cmin >= umax - TIE1_GAP).all())
    v = s64.topk(tid.shape[1] + 1, dim=-1).values
    dec = (v[:, -2] - v[:, -1]) > TIE1_GAP
    same = (torch.sort(tid, 1).values == torch.sort(rt, 1).values).all(1)
    ok = ok and bool(same[dec].all())
    ds = (torch.sigmoid(lsel.double() * 0.1767766952966369) + bias.double() - s64).abs().max()
    ok = ok and float(ds) < TIE1_GAP / 4
    werr = 0.0
    if bool(same.any()):
        o_g, o_r = torch.argsort(tid[same], 1), torch.argsort(rt[same], 1)
        werr = float((tw[same].float().gather(1, o_g) - ref_tw[same].float().gather(1, o_r)).abs().max())
    lid_ok = True
    if lid is not None:
        own = table[(tid + N_EXPERTS * ls).reshape(-1)].view(T, -1)
        lid_ok = bool(torch.equal(torch.sort(lid.long(), 1).values, torch.sort(own.long(), 1).values))
        lid_ok = lid_ok and bool(torch.equal(torch.sort(lid[same].long(), 1).values,
                                             torch.sort(ref_lid[same].long(), 1).values))
    return ok, werr, lid_ok


def _v8_tail_prepare(weights, state):
    if not V8_TAIL:
        return None
    t0 = time.time()
    dev = weights["pooler"].device
    try:
        mod = _V10_MOD if _V10_MOD is not None else _v8_tail_build()
    except Exception as ex:
        print("[v8] tail build FAILED -> k3_v07 path. Tail:\\n" + str(ex)[-8000:], flush=True)
        return None
    router = weights["router"].contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    v8 = {"mod": mod, "nw": state["pre_expert_norm"]._get_effective_weight().detach().to(torch.bfloat16).contiguous(),
          "bias": weights["router_bias"].float().contiguous(),
          "wcat": torch.cat((weights["pooler"], hi, lo), dim=0).contiguous()}
    try:
        gen = torch.Generator(device=dev).manual_seed(4242)
        for T, ls in ((1, 0), (8, 64), (16, 224), (33, 32), (128, 96), (256, 160)):
            x = (torch.randn((T, HIDDEN), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            lstart = torch.tensor([ls], dtype=torch.int32, device=dev)
            d = {"local_start": lstart}
            ref_l64 = x.double() @ router.double().T
            ref_logits = ref_l64.float()
            ref_pool = _gemm(x, weights["pooler"])
            ref_xn = state["pre_expert_norm"](ref_pool)
            ref_tw, ref_tid = state["routing"](None, ref_logits.clone(), TOP_K, True)
            ref_lid = state["local_id_table"][(ref_tid.long() + N_EXPERTS * ls).reshape(-1)].view(T, TOP_K)
            for kind in ("cat", "sep"):
                if kind == "cat":
                    y = torch.mm(x, v8["wcat"].t(), out_dtype=torch.float32)
                    lg = (y[:, D_EXPERT:D_EXPERT + N_EXPERTS] + y[:, D_EXPERT + N_EXPERTS:]).contiguous()
                    lerr = float((lg - ref_logits).norm() / ref_logits.norm())
                    got = _v8_tail(y, None, {"v8": v8}, d)
                else:
                    lerr = 0.0
                    got = _v8_tail(ref_pool, ref_logits, {"v8": v8}, d)
                torch.cuda.synchronize()
                xn, tw, tid, lid = got
                xerr = float((xn.float() - ref_xn.float()).norm() / ref_xn.float().norm())
                ids_eq, werr, lid_eq = _tie1_compare(tid, tw, lid, ref_tid, ref_tw, ref_lid, ref_l64, v8["bias"],
                                                     lg if kind == "cat" else ref_logits,
                                                     state["local_id_table"], ls)
                print(f"[v8] tail self-test {kind} T={T} ls={ls}: logits rel {lerr:.2e} xn rel {xerr:.2e} ids_eq {ids_eq}"
                      f" w maxerr {werr:.1e} lid_eq {lid_eq}", flush=True)
                if not (lerr < 1e-4 and xerr < 1e-2 and ids_eq and werr < 1e-5 and lid_eq
                        and bool(torch.isfinite(xn.float()).all())):
                    raise RuntimeError(f"tail self-test mismatch ({kind}, T={T})")
        for T, ls in ((1, 0), (5, 128), (8, 32)):
            h = torch.randn((T, HIDDEN), generator=gen, device=dev)
            h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
            lstart = torch.tensor([ls], dtype=torch.int32, device=dev)
            ref_l64 = h.double() @ router.double().T
            ref_logits = ref_l64.float()
            part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=dev)
            mod.v8_router32(h, state["router"], part)
            pool = _gemm(h.to(torch.bfloat16), weights["pooler"])
            got = _v8_tail(pool, part, {"v8": v8}, {"local_start": lstart})
            ref_tw, ref_tid = state["routing"](None, ref_logits.clone(), TOP_K, True)
            torch.cuda.synchronize()
            lerr = float((part.sum(0) - ref_logits).norm() / ref_logits.norm())
            ids_eq, werr, _ = _tie1_compare(got[2], got[1], None, ref_tid, ref_tw, None, ref_l64, v8["bias"],
                                            part.sum(0), None, ls)
            print(f"[v8] router32 self-test T={T}: logits rel {lerr:.2e} ids_eq {ids_eq} w maxerr {werr:.1e}", flush=True)
            if not (lerr < 1e-5 and ids_eq and werr < 1e-5):
                raise RuntimeError(f"router32 self-test mismatch (T={T})")
    except Exception as ex:
        print(f"[v8] tail self-test FAILED -> k3_v07 path: {ex!r}", flush=True)
        return None
    global TL1_TAIL2
    if TL1_TAIL2:
        try:
            gen2 = torch.Generator(device=dev).manual_seed(4343)
            same = True
            for T, ls in ((1, 0), (8, 64), (16, 224), (33, 32), (128, 96), (256, 160)):
                x = (torch.randn((T, HIDDEN), generator=gen2, device=dev) * 0.5).to(torch.bfloat16)
                d = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
                y = torch.mm(x, v8["wcat"].t(), out_dtype=torch.float32)
                lg2 = (y[:, D_EXPERT:D_EXPERT + N_EXPERTS] + y[:, D_EXPERT + N_EXPERTS:]).contiguous()
                part2 = (lg2.unsqueeze(0).repeat(8, 1, 1) * 0.125).contiguous()
                pool2 = y[:, :D_EXPERT].to(torch.bfloat16).contiguous()
                for yy, ll in ((y, None), (y, part2), (pool2, lg2), (pool2, part2)):
                    outs = []
                    for flag in (True, False):
                        TL1_TAIL2 = flag
                        outs.append([a.clone() for a in _v8_tail(yy, ll, {"v8": v8}, d)])
                    torch.cuda.synchronize()
                    same = same and all(bool(torch.equal(a, b)) for a, b in zip(outs[0], outs[1]))
            TL1_TAIL2 = same
            print(f"[tl1] tail_kernel2 == tail_kernel bitwise on 6 shapes x 4 variants: {same} -> "
                  f"{'tail_kernel2' if same else 'tail_kernel'}", flush=True)
        except Exception as ex:
            TL1_TAIL2 = False
            print(f"[tl1] tail_kernel2 check raised -> tail_kernel: {ex!r}", flush=True)
    print(f"[v8] tail ready in {time.time() - t0:.1f}s; route {V8_TAIL_ROUTE}", flush=True)
    return v8


V8_K2A3_ROUTE = {1: 'v3', 8: 'v3', 16: 'v3', 32: 'v3', 128: 'v3', 256: 'v3'}
V8_BUILD_V22_ALWAYS = False
K2A3_M3 = 320



K2A3_PLAN = {1: 0, 8: 0, 16: 0, 32: 0, 128: 1, 256: 1}






K2A3_HINT = 16

def _k2a3_call(st, data, vid, probe=K2A3_HINT, mask=7):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"],
                   st["inter"], st["y"], st["meta"], st["cnt"], st["err"], out, st["sink"], m,
                   data["config"]["local_start"], st["grid"][vid], st["sms"] * 2, probe)
    return out





def _v8_k2a3_prepare(weights, t0):
    print("[kl1] K2A v3 is not in this file (legacy strip): a bucket K2A v3.3 cannot serve falls back to r1 Marlin", flush=True)
    return None


def _v8_k2a3_call(x, local_ids, topk_weights, st):
    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}
    return _k2a3_call(st, data, K2A3_PLAN[_k2a_bucket_of(x.shape[0])])


def _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens):
    rep = _k2a_bucket_of(tokens)
    k33 = state.get("k2a33")
    if (k33 is not None and rep is not None and V11_K2A33_ROUTE.get(rep) == "v33" and k33["ok"].get(rep)
            and tokens <= K2A33_M3 and cfg["local_experts"] == K2A_E_LOC and cfg["top_k"] == K2A_TOP_K
            and cfg["n_experts"] == N_EXPERTS):
        try:
            return _v11_k2a33_call(expert_x, local_ids, topk_weights, k33["st"])
        except Exception as ex:
            _k3_disable(state, "k2a33", None, "K2A v3.3", ex)
    k31 = state.get("k2a31")
    if (k31 is not None and rep is not None and V9_K2A31_ROUTE.get(rep) == "v31" and k31["ok"].get(rep)
            and tokens <= K2A31_M3 and cfg["local_experts"] == K2A_E_LOC and cfg["top_k"] == K2A_TOP_K
            and cfg["n_experts"] == N_EXPERTS):
        try:
            return _v9_k2a31_call(expert_x, local_ids, topk_weights, k31["st"])
        except Exception as ex:
            _k3_disable(state, "k2a31", None, "K2A v3.1", ex)
    k3s = state.get("k2a3")
    if (k3s is not None and rep is not None and V8_K2A3_ROUTE.get(rep) == "v3" and k3s["ok"].get(rep)
            and tokens <= K2A3_M3 and cfg["local_experts"] == K2A_E_LOC and cfg["top_k"] == K2A_TOP_K
            and cfg["n_experts"] == N_EXPERTS):
        try:
            return _v8_k2a3_call(expert_x, local_ids, topk_weights, k3s["st"])
        except Exception as ex:
            _k3_disable(state, "k2a3", None, "K2A v3", ex)
    if state["k2a"] is not None and _k2a_supported(cfg, tokens):
        try:
            return _k2a_experts(expert_x, local_ids, topk_weights, state["k2a"])
        except Exception as ex:
            _k3_disable(state, "k2a", None, "K2A", ex)
    return None


def _v8_unpool_t1(expert_output, state):
    st = state.get("r1b")
    w = state["dense"]["unpooler"][:UNPOOL_SLICE]
    if not (V8_DENSE_GEMV_T1 and st is not None and _r1b_aligned(expert_output)):
        return None
    route = _r1b_gemv_route(st, "unpooler", 1, UNPOOL_SLICE, D_EXPERT)
    if route is None or route[3] != 1:
        return None
    zb = state.get("v12_zbuf")
    out = zb[:1] if (V12_ZBUF and zb is not None) else torch.zeros((1, HIDDEN), dtype=expert_output.dtype,
                                                                     device=expert_output.device)
    mt, rpw, wk, split = route
    st["mod"].r1b_gemv(expert_output, w, out[:, :UNPOOL_SLICE], st["dummy"], st["counters"], 1, UNPOOL_SLICE,
                       D_EXPERT, mt, rpw, wk, split, 1 if R1B_FIXUP else 0)
    return out


V9_K2A31_ROUTE = {1: 'v31', 8: 'v31', 16: 'v31', 32: 'v31', 128: 'v31', 256: 'v31'}
V9_BUILD_V3_ALWAYS = False
K2A31_BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (28, 29, 30, 5), (12, 13, 14, 21),
           (25, 26, 27, 2), (9, 10, 11, 18), (4, 0, 1, 3), (20, 16, 17, 19))

def _k2a31_pack_words(codes: torch.Tensor, bitpos=K2A_BITPOS) -> torch.Tensor:
    c = codes.to(torch.int64)
    w = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for i, (p0, p1, p2, ps) in enumerate(bitpos):
        ci = c[..., i]
        w |= ((ci & 1) << p0) | (((ci >> 1) & 1) << p1) | (((ci >> 2) & 1) << p2) | (((ci >> 3) & 1) << ps)
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w)
    return w.to(torch.int32)

def _k2a31_prepack_expert(w: torch.Tensor, sf: torch.Tensor, gate_up: bool, bitpos=K2A_BITPOS):
    n, k2 = w.shape
    k = 2 * k2
    rt, nkb = n // 16, k // 64
    sfb = sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf
    bp, neg, zero, bad = _k2a_reencode_scales(sfb)
    if bad:
        raise ValueError("NaN block scale in NVFP4 weights")
    codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(n, k)
    if bool(neg.any()) or bool(zero.any()):
        cb = codes.view(n, k // 16, 16)
        cb = torch.where(neg.unsqueeze(-1), cb ^ 8, cb)
        cb = torch.where(zero.unsqueeze(-1), torch.zeros_like(cb), cb)
        codes = cb.reshape(n, k)
    perm = _k2a_tile_row_perm(n, gate_up, device=w.device)
    codes = codes[perm]
    bp = bp[perm]
    c = codes.view(rt, 2, 8, nkb, 4, 2, 8).permute(0, 3, 2, 4, 1, 5, 6).reshape(rt, nkb, 32, 4, 8)
    wq = _k2a31_pack_words(c, bitpos)
    sq = bp.view(rt, 2, 8, nkb // 2, 2, 4).permute(0, 3, 2, 5, 4, 1).reshape(rt, nkb // 2, 32, 4)
    return wq.contiguous(), sq.contiguous()

K2A31_M3 = 320



K2A31_PLAN = {1: 0, 8: 0, 16: 0, 32: 0, 128: 1, 256: 1}








K2A31_HINT = 16

def _k2a31_call(st, data, vid, probe=K2A31_HINT, mask=7):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"],
                   st["inter"], st["y"], st["meta"], st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"], m,
                   data["config"]["local_start"], st["grid"][vid], st["sms"] * 2, probe)
    return out




def _v9_k2a31_prepare(weights, t0):
    print("[kl1] K2A v3.1 is not in this file (legacy strip): a bucket K2A v3.3 cannot serve falls back to r1 Marlin", flush=True)
    return None


def _v9_k2a31_call(x, local_ids, topk_weights, st):
    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}
    return _k2a31_call(st, data, K2A31_PLAN[_k2a_bucket_of(x.shape[0])])




V9_BND_ROUTE = {('attn', 1): (16, None), ('attn', 8): (16, None), ('attn', 16): (16, None), ('attn', 32): (16, None), ('ffn', 1): (16, None), ('ffn', 8): (16, None), ('ffn', 16): (16, None), ('ffn', 32): (16, None)}
_V9_FNR = None


def _v9_bnd(which, stream, branch, router_dtype, state, tokens):
    if which == "ffn" and router_dtype is None and VF_SBB2.get(_k2a_bucket_of(tokens)):
        r = _vf_sbb2(stream, branch, tokens)
        if r is not None:
            return r
    vgr = _vg_bnd(which, stream, branch, router_dtype, state, tokens)
    if vgr is not None:
        return vgr
    if which == "attn" and SBB1_ROUTE.get(_k2a_bucket_of(tokens)):
        sbr = _sbb1(stream, branch, router_dtype, state, tokens)
        if sbr is not None:
            return sbr
    """A layer boundary (post-norm + gated residual + pre-norm [+ FP32 router input]) on SGLang's own JIT
    fused_norm_residual with this bucket's measured launch config (V9_BND_ROUTE); k3_v08's module call otherwise."""
    op = state["attn_boundary" if which == "attn" else "ffn_boundary"]
    cfgb = V9_BND_ROUTE.get((which, _k2a_bucket_of(tokens))) if state.get("v9bnd") else None
    if cfgb is None or _V9_FNR is None:
        return op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=router_dtype)
    cs, lc = cfgb
    return _V9_FNR(branch, stream, op._post_norm.eps, op._pre_norm.eps,
                   post_norm_is_input_scale=op._post_norm_is_input_scale,
                   post_weight=op._cc_post_weight if op._post_norm.has_weight else None,
                   pre_weight=op._cc_pre_weight if op._pre_norm.has_weight else None,
                   alpha=op._cc_alpha if op._use_residual_gate else None,
                   beta=op._cc_beta if op._use_residual_gate else None,
                   want_router=router_dtype is not None, launch_config=lc, cluster_size=cs)


def _v9_bnd_prepare(state):
    global _V9_FNR
    try:
        from sglang.kernels.ops.muse_spark_v12.norm_residual import fused_norm_residual
        _V9_FNR = fused_norm_residual
    except Exception as ex:
        print(f"[v9] fused_norm_residual unavailable -> module boundaries: {ex!r}", flush=True)
        return False
    dev = state["router"].device
    gen = torch.Generator(device=dev).manual_seed(909)
    try:
        for name in ("attn_boundary", "ffn_boundary"):
            state[name]._ensure_cached_weights(torch.bfloat16)
        for (which, rep), cfgb in sorted(V9_BND_ROUTE.items(), key=str):
            lo, hi = [(a, b) for a, b, r in K2A_BUCKETS if r == rep][0]
            for T in sorted({lo, rep, hi if hi <= 320 else rep}):
                op = state["attn_boundary" if which == "attn" else "ffn_boundary"]
                stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
                branch = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
                rd = torch.float32 if (which == "attn" and T <= ROUTER_FP32_MAX_TOKENS) else None
                ref = op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=rd)
                state["v9bnd"] = True
                got = _v9_bnd(which, stream, branch, rd, state, T)
                torch.cuda.synchronize()
                e0 = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                e1 = float((got[1] - ref[1]).norm() / ref[1].norm())
                e2 = float((got[2] - ref[2]).norm() / ref[2].norm()) if rd is not None else 0.0
                ok = e0 < 1e-2 and e1 < 1e-4 and e2 < 1e-2 and bool(torch.isfinite(got[1]).all())
                print(f"[v9] boundary self-test {which} T={T} cfg {cfgb}: out {e0:.1e} stream {e1:.1e} router {e2:.1e}"
                      f" -> {'ok' if ok else 'FAIL'}", flush=True)
                if not ok:
                    raise RuntimeError(f"boundary {which} T={T}")
    except Exception as ex:
        print(f"[v9] boundary self-test FAILED -> module boundaries: {ex!r}", flush=True)
        state["v9bnd"] = False
        return False
    state["v9bnd"] = True
    print(f"[v9] boundaries routed: {V9_BND_ROUTE}", flush=True)
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



_V10_MOD = None


def _v10_build_all():
    global _V10_MOD
    _pt1_wait("v10")
    if _V10_MOD is not None:
        return _V10_MOD
    from torch.utils.cpp_extension import load_inline
    cases, attrs = _k2a36_variant_cases(K2A33_VARIANTS)
    k33 = (K2A33_KERNEL_SRC.replace("@CASES3@", cases).replace("@ATTRS3@", attrs)
           .replace("@WGSPECS@", _k2a33_wg_specs()))
    cpp = "\n".join((K1_CPP_SRC, K2A33_CPP_SRC, R1B_CPP_SRC, V8_TAIL_CPP_SRC))
    cuda = "\n".join((K1_CUDA_SRC, k33, R1B_CUDA_SRC, V8_TAIL_CUDA_SRC))
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    try:
        _V10_MOD = load_inline(
            name=f"k3_v10_all_{tag}", cpp_sources=cpp, cuda_sources=cuda,
            functions=["k1_forward", "k1_occupancy", "k1_prof_read", "run3", "attrs3v", "r1b_gemv", "r1b_reduce",
                       "v8_tail", "v8_router32"],
            extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a"],
            verbose=False)
        _K3_BUILD_S["all"] = time.time() - t0
        print(f"[v10] combined build (K1 + K2A v3.7 [v3.3 = vids 0-2, v3.6 = vid 3, v3.7 = vids 4-5] + R1B + tail) in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:
        print("[v10] combined build FAILED -> separate builds. Tail:\n" + str(ex)[-8000:], flush=True)
        _V10_MOD = None
    return _V10_MOD




def _v11_k2a33_build(name, variants, verbose=False):
    if _V10_MOD is None or list(variants) != list(K2A33_VARIANTS):
        return _K2A33_OWN_BUILD(name, variants, verbose)
    return _V10_MOD, 0.0


V11_K2A33_ROUTE = {1: 'v33', 8: 'v33', 16: 'v33', 32: 'v33', 128: 'v33', 256: 'v33'}
V11_BUILD_V31_ALWAYS = False
K2A33_HOT_T = 16

K2A33_HOT_W4 = 8

K2A33_M3 = 320

K2A33_SW3 = 18432

K2A33_VARIANTS = [(1, 3, 2, 0), (2, 3, 2, 0), (2, 2, 2, 1), (1, 3, 2, 2, 4), (1, 3, 2, 3, 1), (1, 3, 2, 3, 4), (2, 2, 2, 5)]
SK1_ENABLE = True
SK1_VID = 6

K2A33_PLAN = {1: 0, 8: 0, 16: 0, 32: 1, 128: 2, 256: 2}

K2A33_HOT_T_BUCKET = {128: 16, 256: 16}

def _k2a33_stage_repack(wq: torch.Tensor, sq: torch.Tensor) -> torch.Tensor:
    rt = wq.shape[0]
    nrb = rt // 8
    wb = wq.contiguous().view(torch.uint8).view(nrb, 8, 16, 4, 512).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 16384)
    sb = sq.contiguous().view(nrb, 8, 16, 2, 128).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 2048)
    return torch.cat((wb, sb), dim=-1).contiguous()

K2A33_KERNEL_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace k3 {

constexpr int KDIM = 4096;
constexpr int M3 = 320;
constexpr int P3 = 8 * M3;
constexpr int SW = 18432;
constexpr int XROW = 528;            // bytes per staged X row (512 + 16 pad: conflict-free LDS.128)
constexpr int XST = XROW / 2;        // bf16 elements per staged X row
constexpr int MH = 128;
constexpr unsigned FULL = 0xffffffffu;
#ifndef K3_NW_DISPATCH
#define K3_NW_DISPATCH 1
#endif
#ifndef V3_EARLY
#define V3_EARLY 0
#endif
#ifndef K36_HALF_FENCE
#define K36_HALF_FENCE 0   // v3.6 hc_item: compiler barrier between the two kb halves (no effect on ptxas; kept off)
#endif
#ifndef K36_KB_ROLL
#define K36_KB_ROLL 0      // v3.6 hc_item: 1 = kb loop not unrolled (register pressure fallback)
#endif
#ifndef K36_LAUNDER
#define K36_LAUNDER 1      // v3.6 flush_item: opaque zero offset on the output bases (register pressure knob)
#endif
constexpr long long WAIT_BUDGET = 1ll << 31;   // clock64 cycles (~1 s) per mbarrier wait -> error flag
constexpr int HO_BUDGET = 10;   // v3.6: whole K1 hot items handed to the epilogue warp while npart < 10 (ids <= 15)

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint8_t* ws13; const uint8_t* ws2;
  const float* gsc13; const float* gsc2; const float* nw;
  float* h; __nv_bfloat16* inter; float* y; int* meta; int* cnt; int* err;
  __nv_bfloat16* out; unsigned* sink;
  float* ss; int* cnt2;   // v3.1: per-route sum of squares of bf16(y)*(1+w) (K2), W2 split-item counters
  uint8_t* xw; uint8_t* xw2;   // v3.2: hot passes' W13 / W2 B operands in wgmma's core-matrix layout
  int hot_t;                   // v3.3: expert is warm (wgmma, 64-token passes) if cnt > hot_t (> 0); 0 = off
  int hot_w4;                  // v3.3: warm-work scale in the CTA role split's byte model, x8; v3.6 role split: per-chunk
  int hot_mode;                // v3.6 (HK2): bit 0 dual range, bit 1 second-wave CTAs cold part first; 0 = role split
  int M, lo, probe;
  int* rdy; long long* pxmk; int pxd;   // M32 PX
};
constexpr int PXB = 28;   // M32 PX probe bits (PXW / PXV never set: the v1 producer, as measured)
constexpr int PXW = 29;
constexpr int PXV = 30;


struct Route {
  int cnt[32], off[32], np[32], hp[32];
  int sbh[33], sbc[33];   // v3.3: stage prefixes over warm experts only / cold experts only
  int sb[33];
  int wc[8][32];
  int R;
};

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
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
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
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }
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
__device__ __forceinline__ void cp_async16_ef(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

template <int NC, int TPH = 64, bool HGE = false, bool LIST = true>
__device__ void route(const Params& P, Route& S, int8_t* s_ide, int16_t* s_list, int nrb, bool write_meta) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int NP = 8 * P.M;
  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  const int L = ((NP + 255) / 256) * 32;
  const int p0 = warp * L;
  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) {
        const int v = P.ids[p] - P.lo;
        e = (v >= 0 && v < 32) ? v : -1;
      }
      if (e >= 0) atomicAdd(&S.wc[warp][e], 1);
    }
  }
  __syncthreads();
  if (warp == 0) {
    int c = 0;
    for (int w = 0; w < 8; ++w) { const int v = S.wc[w][lane]; S.wc[w][lane] = c; c += v; }
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    const bool hot = HGE ? (c > P.hot_t) : (P.hot_t > 0 && c > P.hot_t);
    const int np = hot ? (c + TPH - 1) / TPH : (c + 8 * NC - 1) / (8 * NC);
    int hv = hot ? np : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, hv, d); if (lane >= d) hv += nb; }
    S.hp[lane] = hv - (hot ? np : 0);
    {
      int cv = hot ? 0 : np;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, cv, d); if (lane >= d) cv += nb; }
      S.sbh[lane] = 16 * nrb * (hv - (hot ? np : 0));
      S.sbc[lane] = 16 * nrb * (cv - (hot ? 0 : np));
      if (lane == 31) { S.sbh[32] = 16 * nrb * hv; S.sbc[32] = 16 * nrb * cv; }
    }
    int sv = np;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, sv, d); if (lane >= d) sv += nb; }
    S.cnt[lane] = c; S.off[lane] = off; S.np[lane] = np;
    S.sb[lane] = 16 * nrb * (sv - np);
    if (lane == 31) { S.sb[32] = 16 * nrb * sv; S.R = v; }
    for (int w = 0; w < 8; ++w) S.wc[w][lane] += off;
  }
  __syncthreads();
  if (!LIST) return;
  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) { const int v = __ldg(P.ids + p) - P.lo; e = (v >= 0 && v < 32) ? v : -1; }
      const unsigned m = __match_any_sync(FULL, e);
      const int rank = __popc(m & ((1u << lane) - 1u));
      const int b = (e >= 0) ? S.wc[warp][e] : 0;
      __syncwarp();
      if (e >= 0) {
        const int c = b + rank;
        s_list[c] = (int16_t)p;
        if (write_meta) P.meta[MH + p] = c;
        if (rank == 0) S.wc[warp][e] = b + __popc(m);
      } else if (write_meta && p < NP) {
        P.meta[MH + p] = -1;
      }
      __syncwarp();
    }
  }
  __syncthreads();
}
template <int NC, int TPH = 64, bool HGE = false, bool LIST = true>
__device__ void route_fr(const Params& P, Route& S, int8_t* s_ide, int16_t* s_list, int nrb, bool write_meta) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int NP = 8 * P.M;
  const int L = ((NP + 255) / 256) * 32;
  const int p0 = warp * L;
  constexpr int RQ = (P3 + 255) / 256;   // SK1: a lane's ids, loaded once (independent loads) for both passes
  int ev[RQ];
#pragma unroll
  for (int q = 0; q < RQ; ++q) {
    const int p = p0 + 32 * q + lane;
    int e = -1;
    if (warp < 8 && 32 * q < L && p < NP) {
      const int v = __ldg(P.ids + p) - P.lo;
      e = (v >= 0 && v < 32) ? v : -1;
    }
    ev[q] = e;
  }
  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  if (warp < 8) {
#pragma unroll
    for (int q = 0; q < RQ; ++q)
      if (ev[q] >= 0) atomicAdd(&S.wc[warp][ev[q]], 1);
  }
  __syncthreads();
  if (warp == 0) {
    int c = 0;
    for (int w = 0; w < 8; ++w) { const int v = S.wc[w][lane]; S.wc[w][lane] = c; c += v; }
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    const bool hot = HGE ? (c > P.hot_t) : (P.hot_t > 0 && c > P.hot_t);
    const int np = hot ? (c + TPH - 1) / TPH : (c + 8 * NC - 1) / (8 * NC);
    int hv = hot ? np : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, hv, d); if (lane >= d) hv += nb; }
    S.hp[lane] = hv - (hot ? np : 0);
    {
      int cv = hot ? 0 : np;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, cv, d); if (lane >= d) cv += nb; }
      S.sbh[lane] = 16 * nrb * (hv - (hot ? np : 0));
      S.sbc[lane] = 16 * nrb * (cv - (hot ? 0 : np));
      if (lane == 31) { S.sbh[32] = 16 * nrb * hv; S.sbc[32] = 16 * nrb * cv; }
    }
    int sv = np;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, sv, d); if (lane >= d) sv += nb; }
    S.cnt[lane] = c; S.off[lane] = off; S.np[lane] = np;
    S.sb[lane] = 16 * nrb * (sv - np);
    if (lane == 31) { S.sb[32] = 16 * nrb * sv; S.R = v; }
    for (int w = 0; w < 8; ++w) S.wc[w][lane] += off;
  }
  __syncthreads();
  if (!LIST) return;
  if (warp < 8) {
#pragma unroll
    for (int q = 0; q < RQ; ++q) {
      if (32 * q >= L) break;   // warp-uniform
      const int p = p0 + 32 * q + lane;
      const int e = ev[q];
      const unsigned m = __match_any_sync(FULL, e);
      const int rank = __popc(m & ((1u << lane) - 1u));
      const int b = (e >= 0) ? S.wc[warp][e] : 0;
      __syncwarp();
      if (e >= 0) {
        const int c = b + rank;
        s_list[c] = (int16_t)p;
        if (write_meta) P.meta[MH + p] = c;
        if (rank == 0) S.wc[warp][e] = b + __popc(m);
      } else if (write_meta && p < NP) {
        P.meta[MH + p] = -1;
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void my_range2(long long S, int cta, int ncta, int& beg, int& end) {
  beg = (int)(S * (long long)cta / ncta);
  end = (int)(S * (long long)(cta + 1) / ncta);
}
__device__ __forceinline__ void my_range(long long S, int& beg, int& end) {
  const long long G = gridDim.x;
  beg = (int)(S * (long long)blockIdx.x / G);
  end = (int)(S * (long long)(blockIdx.x + 1) / G);
}

__device__ __forceinline__ int tail_first(const Route& S, int beg, int end, int rs1 = 0) {   // rs1: full warp
  if (end <= beg) return beg;
  int e = 0;
  if (rs1) e = __popc(__ballot_sync(0xffffffffu, S.sb[(threadIdx.x & 31) + 1] <= end - 1));   // RS1: = the scan
  else while (S.sb[e + 1] <= end - 1) ++e;
  const int m = S.sb[e] + ((end - 1 - S.sb[e]) & ~15);
  return max(beg, m);
}

template <int NC>
__device__ __forceinline__ void consume_kb(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow,
                                           int kloc, int nch, int t, float (*acc)[4]) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
#pragma unroll
  for (int q = 0; q < NC; ++q) {
    if (q < nch) {
      const uint4* xp = reinterpret_cast<const uint4*>(xrow + q * 8 * XST + kloc + 16 * t);
      const uint4 x0 = xp[0], x1 = xp[1];
      mma16816(acc[q], ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
      mma16816(acc[q], ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
      mma16816(acc[q], ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
      mma16816(acc[q], ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
    }
  }
}

template <int NC>
__device__ __forceinline__ void consume_kb_breg(const uint4 w, const uint32_t sw2, int par, uint4 x0, uint4 x1,
                                                int nch, float (*acc)[4], uint32_t& sink, bool mma) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  if (!mma) {
#pragma unroll
    for (int j = 0; j < 8; ++j) sink ^= ag[j] ^ a8[j];
    return;
  }
#pragma unroll
  for (int q = 0; q < NC; ++q) {
    if (q < nch) {
      mma16816(acc[q], ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
      mma16816(acc[q], ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
      mma16816(acc[q], ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
      mma16816(acc[q], ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
    }
  }
}

__device__ __forceinline__ size_t xw_u32(int pid, int NW, int s2, int n, int i, int t, int q) {
  return (size_t)(pid * 16 + s2) * (NW * 128) + (n >> 3) * 1024 + 256 * i + 32 * q + 4 * (n & 7) + t;
}
__device__ __forceinline__ size_t xw_elem(int pid, int NW, int n, int col) {
  const int kk = col & 255;
  return 2 * xw_u32(pid, NW, col >> 8, n, kk >> 6, (kk >> 4) & 3, (kk >> 1) & 7) + (kk & 1);
}
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
#define K3_WG_HEAD "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned."
template <int N> __device__ __forceinline__ void wgmma_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                          uint64_t db);
template <> __device__ __forceinline__ void wgmma_rs<64>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                         uint64_t db) {
  asm volatile(K3_WG_HEAD "m64n64k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
@WGSPECS@
#undef K3_WG_HEAD
template <int N> __device__ __forceinline__ void wgmma_rs2(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <> __device__ __forceinline__ void wgmma_rs2<16>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%12, %13};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
template <> __device__ __forceinline__ void wgmma_rs2<32>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%20, %21};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
template <> __device__ __forceinline__ void wgmma_rs2<48>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%28, %29};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
template <> __device__ __forceinline__ void wgmma_rs2<64>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%36, %37};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
__device__ __forceinline__ void wgmma_nw(int nw, float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
  if (nw <= 16) wgmma_rs<16>(d, a0, a1, a2, a3, db);
  else if (nw <= 32) wgmma_rs<32>(d, a0, a1, a2, a3, db);
  else if (nw <= 48) wgmma_rs<48>(d, a0, a1, a2, a3, db);
  else wgmma_rs<64>(d, a0, a1, a2, a3, db);
}

template <bool HOT, int NC>
__device__ __forceinline__ long long wsk_wpass(int c, int jj) {
  constexpr int TP = HOT ? 64 : 8 * NC;
  const int nt = min(TP, c - jj * TP);
  return HOT ? 72 + ((nt + 7) & ~7) : 36 + nt;   // v4.3: hot rows weigh half as much (lab sweep)
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ int wsk_pos(const Route& S, long long T) {
  for (int e = 0; e < 32; ++e) {
    if (S.sb[e + 1] == S.sb[e]) continue;
    const int np = S.np[e], c = S.cnt[e];
    long long wrb = 0;
    for (int jj = 0; jj < np; ++jj) wrb += 16 * wsk_wpass<HOT, NC>(c, jj);
    if (T >= wrb * NRB) { T -= wrb * NRB; continue; }
    const int rb = (int)(T / wrb);
    T -= rb * wrb;
    for (int jj = 0; jj < np; ++jj) {
      const long long w = wsk_wpass<HOT, NC>(c, jj);
      if (T >= 16 * w) { T -= 16 * w; continue; }
      return S.sb[e] + (rb * np + jj) * 16 + (int)(T / w);
    }
    return S.sb[e] + (rb + 1) * np * 16;
  }
  return S.sb[32];
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ void wsk_range(const Route& S, int cta, int ncta, int& beg, int& end) {
  long long W = 0;
  for (int e = 0; e < 32; ++e) {
    if (S.sb[e + 1] == S.sb[e]) continue;
    for (int jj = 0; jj < S.np[e]; ++jj) W += 16LL * NRB * wsk_wpass<HOT, NC>(S.cnt[e], jj);
  }
  beg = wsk_pos<HOT, NC, NRB>(S, W * cta / ncta);
  end = (cta + 1 == ncta) ? S.sb[32] : wsk_pos<HOT, NC, NRB>(S, W * (cta + 1) / ncta);
}

template <int N, int NST, int STG, int MODE>
__device__ __forceinline__ void hot_item(const Params& P, const Route& S, const uint8_t* ring, uint32_t full_a,
                                      uint32_t empty_a, int& it_ref, int s0, int s1, int e, int j, int rb, int c0,
                                      int ntok, int npart) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  float hacc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) hacc[i] = 0.f;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    const uint8_t* st = ring + slot * STG;
    const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
    const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);
    const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);   // v4.0 LEAN: the low word's add never carries
    uint32_t hb[2][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 w = wp[i * 32];
      const uint32_t sw2 = sp[(i >> 1) * 32];
      const int par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
      if (i >= 2) wg_wait<1>();
      uint32_t* h = hb[i & 1];
      deq4(w.x, h); deq4(w.y, h + 4); deq4(w.z, h + 8); deq4(w.w, h + 12);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) { h[jj] = bmul2(h[jj], Sg); h[8 + jj] = bmul2(h[8 + jj], S8); }
      wg_fence();
      wgmma_rs2<N>(hacc, h[0], h[8], h[1], h[9], dlo + 64 * i, dhi);
      wgmma_rs2<N>(hacc, h[2], h[10], h[3], h[11], dlo + 64 * i + 16, dhi);
      wgmma_rs2<N>(hacc, h[4], h[12], h[5], h[13], dlo + 64 * i + 32, dhi);
      wgmma_rs2<N>(hacc, h[6], h[14], h[7], h[15], dlo + 64 * i + 48, dhi);
      wg_commit();
    }
    wg_wait<0>();
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  it_ref = it;
  const int nch = (ntok + 7) >> 3;
  const int tile = rb * 8 + warp;
  const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
  const bool whole = (s0 == 0 && s1 == 16);
  if constexpr (MODE == 1) {
    const int col = 8 * tile + g;
    if (whole) {   // xw_elem(pid, 64, n, col) = xw_elem(pid, 64, 2t, col) + 2048 (n >> 3) + 8 hh for n = 8q + 2t + hh
      __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(P.xw2) + xw_elem(S.hp[e] + j, 64, 2 * t, col);
#pragma unroll
      for (int q = 0; q < N / 8; ++q)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          if (q * 8 + 2 * t + hh < ntok)
            xb[2048 * q + 8 * hh] = __float2bfloat16_rn(swiglu(hacc[4 * q + hh] * gs, hacc[4 * q + 2 + hh] * gs));
    } else {
      float* hb = P.h + (size_t)(c0 + 2 * t) * (2 * KDIM) + col;
#pragma unroll
      for (int q = 0; q < N / 8; ++q)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          if (q * 8 + 2 * t + hh < ntok) {
            atomicAdd(hb + (8 * q + hh) * (2 * KDIM), hacc[4 * q + hh] * gs);
            atomicAdd(hb + (8 * q + hh) * (2 * KDIM) + KDIM, hacc[4 * q + 2 + hh] * gs);
          }
    }
  } else {
    float* yb = P.y + (size_t)(c0 + 2 * t) * KDIM + 16 * tile + g;
#pragma unroll
    for (int q = 0; q < N / 8; ++q)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
        if (q * 8 + 2 * t + hh < ntok) {
          const float v0 = hacc[4 * q + hh] * gs, v1 = hacc[4 * q + 2 + hh] * gs;
          float* yp = yb + (8 * q + hh) * KDIM;
          if (whole) { yp[0] = v0; yp[8] = v1; }
          else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
        }
  }
  if (MODE == 2 && whole) {
    const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
    for (int q = 0; q < N / 8; ++q) {
      if (q < nch) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const float z0 = bf16r(hacc[4 * q + hh] * gs) * w0, z1 = bf16r(hacc[4 * q + 2 + hh] * gs) * w1;
          float pz = z0 * z0 + z1 * z1;
          pz += __shfl_xor_sync(FULL, pz, 4);
          pz += __shfl_xor_sync(FULL, pz, 8);
          pz += __shfl_xor_sync(FULL, pz, 16);
          const int n = q * 8 + 2 * t + hh;
          if (g == 0 && n < ntok) atomicAdd(P.ss + c0 + n, pz);
        }
      }
    }
  }
  if (!whole) asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
}

template <int NC, int NST, int MODE, int HM, int STG, bool SKF = false>
__device__ void gemm(const Params& P, Route& S, const int16_t* s_list, uint8_t* ring, uint64_t* bars, bool pdl,
                     int cta, int ncta) {
  constexpr bool HOTCAP = (HM == 1);
  constexpr int QMAX = HOTCAP ? 8 : NC;   // accumulator chunks (8 tokens each)
  constexpr int NRB = (MODE == 1) ? 64 : 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long long Stot = S.sb[32];
  int beg, end;
  if constexpr (NC == 2) {   // v4.2 WSK: NC 2 builds only (HOTCAP roles; vid 0's NC 1 code stays byte-identical)
    if (P.hot_t > 0) wsk_range<HOTCAP, NC, NRB>(S, cta, ncta, beg, end);
    else my_range2(Stot, cta, ncta, beg, end);
  } else {
    my_range2(Stot, cta, ncta, beg, end);
  }
  const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;
  const uint32_t ring_a = sa(ring);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);

  if (warp == 8) {
    // until griddepcontrol.wait, which is taken lazily: only when a deferred slot would otherwise be reused, or at the
    int it = 0, e = 0;
    bool waited = !pdl;
    int nd = 0;
    const int pmode = P.probe & 15;
    const bool hint = (P.probe & 16) != 0;
    const uint64_t pol = policy_evict_first();
    uint32_t dDst[NST], dBar[NST];
    const __nv_bfloat16* dSrc[NST];
    uint32_t dB[NST];
    const int mid = tail_first(S, beg, end, (P.probe >> 7) & 1);
    for (int seg = 0; seg < 2; ++seg) {
    const int xb = seg ? beg : mid, xe = seg ? mid : end;
    e = 0;
    for (int x = xb; x < xe;) {
      while (S.sb[e + 1] <= x) ++e;
      const int np = S.np[e];
      constexpr bool hot = HOTCAP;
      const int TP = hot ? 64 : 8 * NC;
      const int l = x - S.sb[e];
      const int item = l >> 4;
      const int s0 = l & 15;
      const int s1 = min(16, s0 + (xe - x));
      const int rb = item / np, j = item - rb * np;
      const int ntok = min(TP, S.cnt[e] - j * TP);
      const int c0 = S.off[e] + j * TP;
      const uint8_t* wb = wsrc + ((size_t)(e * NRB + rb) * 16) * SW;
      const int ntx = (pmode == 5) ? 0 : ntok;
      int tok = 0;
      if (MODE == 1 && NC <= 2 && !hot && lane >= 1 && lane <= ntx) tok = ((int)s_list[c0 + lane - 1]) >> 3;
      const uint32_t xbytes = hot ? ((ntok + 7) >> 3) * 4096 : (MODE == 1) ? ntx * 512 : ntok * XROW;
      for (int s = s0; s < s1; ++s, ++it) {
        const int slot = it % NST;
        const uint32_t par = ((it / NST) & 1) ^ 1;
        if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now
          pdl_wait();
          asm volatile("fence.proxy.async;\n" ::: "memory");
          waited = true;
          if (lane == 0) {
#pragma unroll
            for (int d = 0; d < NST; ++d)
              if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
          }
          nd = 0;
        }
        mbar_wait(empty_a + 8 * slot, par, P.err);
        const uint32_t dst = ring_a + slot * STG;
        const uint32_t fb = full_a + 8 * slot;
        if (lane == 0) {
          mbar_expect_tx(fb, SW + xbytes);
          const uint8_t* src = wb + (size_t)s * SW;
          if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);
          else bulk_g2s(dst, src, SW, fb);
        }
        mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals
        __syncwarp();
        if (MODE == 1 && !hot) {
          if constexpr (NC <= 2) {
            if (lane >= 1 && lane <= ntx)
              bulk_g2s(dst + SW + (lane - 1) * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);
          } else {   // up to 8 NC rows: lane i copies rows i, i + 32
            for (int i = lane; i < ntx; i += 32)
              bulk_g2s(dst + SW + i * XROW, P.x + (size_t)(((int)s_list[c0 + i]) >> 3) * KDIM + s * 256, 512, fb);
          }
        } else {
          const int pid = S.hp[e] + j;
          const __nv_bfloat16* src = hot
              ? reinterpret_cast<const __nv_bfloat16*>(((MODE == 1) ? P.xw : P.xw2) + (size_t)(pid * 16 + s) * (64 * 512))
              : P.inter + ((size_t)s * (8 * M3) + c0) * XST;
          if (waited) {
            if (lane == 0) bulk_g2s(dst + SW, src, xbytes, fb);
          } else {
#pragma unroll
            for (int d = 0; d < NST; ++d)
              if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = src; dB[d] = xbytes; }
            ++nd;
          }
        }
      }
      x += s1 - s0;
    }
    }
    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain
    if (!waited) {
      pdl_wait();
      asm volatile("fence.proxy.async;\n" ::: "memory");
      if (lane == 0) {
#pragma unroll
        for (int d = 0; d < NST; ++d)
          if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
      }
    }
    return;
  }

  if (warp == 9) {
    {
      if ((P.probe & 15) != 0) return;   // probes: consumers never hand over
      int npart = 0, e = 0;
      const int mid = tail_first(S, beg, end, (P.probe >> 7) & 1);
      for (int seg = 0; seg < 2; ++seg) {
        const int xb = seg ? beg : mid, xe = seg ? mid : end;
        e = 0;
        for (int x = xb; x < xe;) {
          while (S.sb[e + 1] <= x) ++e;
          const int np = S.np[e];
          const int l = x - S.sb[e];
          const int item = l >> 4;
          const int s0 = l & 15;
          const int s1 = min(16, s0 + (xe - x));
          x += s1 - s0;
          if (s0 == 0 && s1 == 16) continue;
          constexpr bool hot = HOTCAP;
          const int TP = hot ? 64 : 8 * NC;
          const int rb = item / np, j = item - rb * np;
          const int ntok = min(TP, S.cnt[e] - j * TP);
          const int c0 = S.off[e] + j * TP;
          asm volatile("bar.sync %0, 288;\n" :: "r"(2 + npart) : "memory");
          ++npart;
          __threadfence();   // cumulative over the consumers' h atomics (ordered before us by the barrier)
          int last = 0;
          if (lane == 0) {
            int* cp = ((MODE == 1) ? P.cnt : P.cnt2) + (e * NRB + rb) * 64 + j;
            const int old = atomicAdd(cp, s1 - s0);
            last = (old + (s1 - s0) == 16);
            if (last) atomicExch(cp, 0);
          }
          last = __shfl_sync(FULL, last, 0);
          if (MODE == 2 && last) {
            __threadfence();
            const int col = rb * 128 + 4 * lane;
            const float4 wv = __ldg(reinterpret_cast<const float4*>(P.nw + col));
            constexpr int FBX = SKF ? 8 : 4;   // SK1: hot split-item finalize, 8 tokens per L2 trip
            for (int n0 = 0; n0 < ntok; n0 += FBX) {
              float4 yv[FBX];
#pragma unroll
              for (int u = 0; u < FBX; ++u)
                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
              for (int u = 0; u < FBX; ++u) {
                float pz = 0.f;
                if (n0 + u < ntok) {
                  float z;
                  z = bf16r(yv[u].x) * wv.x; pz += z * z;  z = bf16r(yv[u].y) * wv.y; pz += z * z;
                  z = bf16r(yv[u].z) * wv.z; pz += z * z;  z = bf16r(yv[u].w) * wv.w; pz += z * z;
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pz += __shfl_xor_sync(FULL, pz, o);
                if (lane == 0 && n0 + u < ntok) atomicAdd(P.ss + c0 + n0 + u, pz);
              }
            }
          }
          if (MODE == 1 && last) {
            __threadfence();
            const int cl = 2 * lane;
            const int col = rb * 64 + cl;
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float2 gv[4], uv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const float* hp = P.h + (size_t)(c0 + n0 + u) * (2 * KDIM) + col;
                  gv[u] = __ldcg(reinterpret_cast<const float2*>(hp));
                  uv[u] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
                }
              }
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const int c = c0 + n0 + u;
                  float* hp = P.h + (size_t)c * (2 * KDIM) + col;
                  const uint32_t pk = pack_bf2(swiglu(gv[u].x, uv[u].x), swiglu(gv[u].y, uv[u].y));
                  if (hot) {
                    *reinterpret_cast<uint32_t*>(P.xw2 + 2 * xw_elem(S.hp[e] + j, 64, n0 + u, col)) = pk;
                  } else {
                    *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
                  }
                  *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
                  *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
                }
              }
            }
          }
        }
      }
    }
    return;
  }

  if (MODE == 1 && ((P.probe & 15) == 0 || (P.probe & 15) >= 6)) {
    const size_t n4 = (size_t)S.R * KDIM / 4;
    float4* yp = reinterpret_cast<float4*>(P.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)blockIdx.x * 256 + tid; i < n4; i += (size_t)gridDim.x * 256) yp[i] = z;
    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;
  }
  const int g = lane >> 2, t = lane & 3;
  int it = 0, e = 0;
  uint32_t sink = 0u;
  int npart = 0;
  const int mid = tail_first(S, beg, end, (P.probe >> 7) & 1);
  for (int seg = 0; seg < 2; ++seg) {
  const int xb = seg ? beg : mid, xe = seg ? mid : end;
  e = 0;
  for (int x = xb; x < xe;) {
    while (S.sb[e + 1] <= x) ++e;
    const int np = S.np[e];
    const int l = x - S.sb[e];
    const int item = l >> 4;
    const int s0 = l & 15;
    const int s1 = min(16, s0 + (xe - x));
    constexpr bool hot = HOTCAP;
    const int TP = hot ? 64 : 8 * NC;
    const int rb = item / np, j = item - rb * np;
    const int ntok = min(TP, S.cnt[e] - j * TP);
    const int nch = (ntok + 7) >> 3;
    const int c0 = S.off[e] + j * TP;
    const int nw = ((ntok + 15) >> 4) << 4;   // hot: wgmma N in {16, 32, 48, 64}
    float acc[QMAX][4], acc2[NC][4];   // acc2: second mma chain (NC == 1 only)
    if constexpr (HOTCAP) {
      if ((P.probe & 15) == 0) {
        if (nw <= 16) hot_item<16, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else if (nw <= 32) hot_item<32, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else if (nw <= 48) hot_item<48, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else hot_item<64, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
      }
      if (!(s0 == 0 && s1 == 16)) ++npart;
      x += s1 - s0;
      continue;
    } else {
#pragma unroll
    for (int q = 0; q < QMAX; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[q][i] = 0.f;
#pragma unroll
    for (int q = 0; q < NC; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc2[q][i] = 0.f;
    for (int s = s0; s < s1; ++s, ++it) {
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
      const uint8_t* st = ring + slot * STG;
      const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
      const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;
      if ((P.probe & 15) == 1 || (P.probe & 15) == 5) {
        const uint4 a = wp[0], b = wp[96];
        sink ^= a.x ^ b.w ^ sp[0];
      } else if ((P.probe & 15) == 9 || (P.probe & 15) == 10) {
        uint4 wv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
        const uint32_t s0v = sp[0], s1v = sp[32];
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_a + 8 * slot);   // operands in registers: release the slot first
        const uint4 xb0 = make_uint4(s0v, s1v, s0v ^ 1u, s1v ^ 1u), xb1 = make_uint4(s1v, s0v, s1v ^ 2u, s0v ^ 2u);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          consume_kb_breg<NC>(wv[i], (i < 2) ? s0v : s1v, i & 1, xb0, xb1, nch, (i < 2 || NC > 1) ? acc : acc2, sink,
                              (P.probe & 15) == 9);
        continue;
      } else if (NC == 1 && V3_EARLY) {
        uint4 wv[4], x0[4], x1[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          wv[i] = wp[i * 32];
          const uint4* xp = reinterpret_cast<const uint4*>(xrow + i * 64 + 16 * t);
          x0[i] = xp[0]; x1[i] = xp[1];
        }
        const uint32_t s0v = sp[0], s1v = sp[32];
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_a + 8 * slot);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          consume_kb_breg<NC>(wv[i], (i < 2) ? s0v : s1v, i & 1, x0[i], x1[i], nch, (i < 2 || NC > 1) ? acc : acc2, sink, true);
        continue;
      } else {
        uint4 wv[4];
        uint32_t sv[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
        sv[0] = sp[0]; sv[1] = sp[32];
#pragma unroll
        for (int i = 0; i < 4; ++i) consume_kb<NC>(wv[i], sv[i >> 1], i & 1, xrow, i * 64, nch, t, (i < 2 || NC > 1) ? acc : acc2);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
    }
#pragma unroll
    for (int q = 0; q < NC; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) if (NC == 1) acc[q][i] += acc2[q][i];
    const int fmode = P.probe & 15;   // 6: partial items stored like whole ones (timing only); 7: no finalize
    if (fmode != 0 && fmode < 6) {
#pragma unroll
      for (int q = 0; q < QMAX; ++q) sink ^= __float_as_uint(acc[q][0] + acc[q][1] + acc[q][2] + acc[q][3]);
    } else {
      const int tile = rb * 8 + warp;
      const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
      const bool whole = (s0 == 0 && s1 == 16) || fmode == 6;
      if constexpr (MODE == 1) {
#pragma unroll
        for (int q = 0; q < QMAX; ++q) {
          if (q < nch) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int n = q * 8 + 2 * t + hh;
              if (n < ntok) {
                const int c = c0 + n;
                const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
                const int col = 8 * tile + g;
                if (whole && hot) {
                  reinterpret_cast<__nv_bfloat16*>(P.xw2)[xw_elem(S.hp[e] + j, 64, n, col)] = __float2bfloat16_rn(swiglu(v0, v1));
                } else if (whole) {
                  P.inter[((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));
                } else {
                  atomicAdd(P.h + (size_t)c * (2 * KDIM) + col, v0);
                  atomicAdd(P.h + (size_t)c * (2 * KDIM) + KDIM + col, v1);
                }
              }
            }
          }
        }
        if (!whole && fmode != 7) {
          asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
        }
      } else {
#pragma unroll
        for (int q = 0; q < QMAX; ++q) {
          if (q < nch) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int n = q * 8 + 2 * t + hh;
              if (n < ntok) {
                float* yp = P.y + (size_t)(c0 + n) * KDIM + 16 * tile + g;
                const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
                if (whole) { yp[0] = v0; yp[8] = v1; }
                else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
              }
            }
          }
        }
        if (whole) {
          const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
          for (int q = 0; q < QMAX; ++q) {
            if (q < nch) {
#pragma unroll
              for (int hh = 0; hh < 2; ++hh) {
                const float z0 = bf16r(acc[q][hh] * gs) * w0, z1 = bf16r(acc[q][2 + hh] * gs) * w1;
                float pz = z0 * z0 + z1 * z1;
                pz += __shfl_xor_sync(FULL, pz, 4);
                pz += __shfl_xor_sync(FULL, pz, 8);
                pz += __shfl_xor_sync(FULL, pz, 16);
                const int n = q * 8 + 2 * t + hh;
                if (g == 0 && n < ntok) atomicAdd(P.ss + c0 + n, pz);
              }
            }
          }
        } else if (fmode != 7) {
          asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
        }
      }
    }
    if (!(s0 == 0 && s1 == 16)) ++npart;
    x += s1 - s0;
  }
  }
  if (sink == 0x9E3779B9u) atomicAdd(P.sink, 1u);   // keeps probe work alive; never true in practice
}

template <int NC, int NST>
__device__ __forceinline__ void setup_bars(uint64_t* bars) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + NST + i), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
}

template <int NC>
__device__ __forceinline__ bool pick_role(Route& S, int P_hot_w4, int hot_t, int& cta, int& ncta) {
  const long long Sh = S.sbh[32], Sc = S.sbc[32];
  const int G = (int)gridDim.x;
  int Gh = 0;
  if (Sh > 0) {
    long long Wh = 0, Wc = 0;
    for (int e = 0; e < 32; ++e) {
      const int c = S.cnt[e];
      if (c == 0) continue;
      if (c > hot_t) {
        const int full = c >> 6, rem = c & 63;
        Wh += (long long)full * (36 + 64) + (rem ? 36 + ((rem + 7) & ~7) : 0);
      } else {
        const int tp = 8 * NC, full = c / tp, rem = c - full * tp;
        Wc += (long long)full * (36 + tp) + (rem ? 36 + rem : 0);
      }
    }
    if (P_hot_w4 >= 50) { Wh *= P_hot_w4 - 50; Wc *= 32; } else { Wh *= P_hot_w4; Wc *= 8; }   // v4.0 fine weight
    Gh = (int)((Wh * G + (Wh + Wc) / 2) / (Wh + Wc));
    Gh = max(1, min(Gh, Sc > 0 ? G - 1 : G));
  }
  const bool hr = (int)blockIdx.x < Gh;
  cta = hr ? (int)blockIdx.x : (int)blockIdx.x - Gh;
  ncta = hr ? Gh : G - Gh;
  __syncthreads();
  if (threadIdx.x < 33) S.sb[threadIdx.x] = hr ? S.sbh[threadIdx.x] : S.sbc[threadIdx.x];
  __syncthreads();
  return hr;
}

__device__ __forceinline__ int tail_first2(const int* sb, int beg, int end, int rs1 = 0) {   // rs1: full warp
  if (end <= beg) return beg;
  int e = 0;
  if (rs1) e = __popc(__ballot_sync(0xffffffffu, sb[(threadIdx.x & 31) + 1] <= end - 1));   // RS1: = the scan
  else while (sb[e + 1] <= end - 1) ++e;
  const int m = sb[e] + ((end - 1 - sb[e]) & ~15);
  return max(beg, m);
}

template <int NQ, int MODE, bool NOSWI = false>
__device__ __forceinline__ void flush_item(const Params& P, float (*acc)[4], bool whole, int e, int rb, int c0, int ntok,
                                           int npart, bool arrive) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tile = rb * 8 + warp;
  const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
  int z = 0;
#if K36_LAUNDER
  asm volatile("" : "+r"(z));
#endif
  float* hb = P.h + z;
  __nv_bfloat16* ib = P.inter + z;
  float* yb = P.y + z;
  float* ssb = P.ss + z;
  if constexpr (MODE == 1) {
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = q * 8 + 2 * t + hh;
        if (n < ntok) {
          const int c = c0 + n;
          const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
          const int col = 8 * tile + g;
          if (whole && !NOSWI) {
            ib[((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));
          } else {
            atomicAdd(hb + (size_t)c * (2 * KDIM) + col, v0);
            atomicAdd(hb + (size_t)c * (2 * KDIM) + KDIM + col, v1);
          }
        }
      }
    }
    if (arrive) asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");   // split item -> epilogue warp
  } else {
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = q * 8 + 2 * t + hh;
        if (n < ntok) {
          float* yp = yb + (size_t)(c0 + n) * KDIM + 16 * tile + g;
          const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
          if (whole) { yp[0] = v0; yp[8] = v1; }
          else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
        }
      }
    }
    if (whole) {
      const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
      for (int q = 0; q < NQ; ++q) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const float z0 = bf16r(acc[q][hh] * gs) * w0, z1 = bf16r(acc[q][2 + hh] * gs) * w1;
          float pz = z0 * z0 + z1 * z1;
          pz += __shfl_xor_sync(FULL, pz, 4);
          pz += __shfl_xor_sync(FULL, pz, 8);
          pz += __shfl_xor_sync(FULL, pz, 16);
          const int n = q * 8 + 2 * t + hh;
          if (g == 0 && n < ntok) atomicAdd(ssb + c0 + n, pz);
        }
      }
    } else if (arrive) {
      asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
    }
  }
}

template <int NST, int STG, int MODE, bool MB = false>
__device__ __forceinline__ void cold_item(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                          int& it_ref, int s0, int s1, int e, int rb, int c0, int ntok, int npart) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  float acc[1][4], acc2[1][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { acc[0][i] = 0.f; acc2[0][i] = 0.f; }
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    const uint8_t* st = ring + slot * STG;
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
  it_ref = it;
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[0][i] += acc2[0][i];
  const bool whole = s0 == 0 && s1 == 16;
  flush_item<1, MODE>(P, acc, whole, e, rb, c0, ntok, npart, MB ? false : !whole);
}

template <int NQ, int NST, int STG, int MODE, bool MB = false>
__device__ __forceinline__ void hc_item(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                        int& it_ref, int s0, int s1, int e, int rb, int c0, int ntok, int npart, bool ho) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  float acc[NQ][4];
#pragma unroll
  for (int q = 0; q < NQ; ++q)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[q][i] = 0.f;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    const uint8_t* st = ring + slot * STG;
    const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
    const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST + 16 * t;
#if K36_KB_ROLL
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int i = 0; i < 4; ++i) {
      const uint4 w = wp[i * 32];
      const uint32_t sw2 = sp[(i >> 1) * 32];
      const int par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
#if K36_HALF_FENCE
        if (hf) asm volatile("" ::: "memory");   // keep the second half's X loads after the first half's mma
#endif
        uint32_t a[4], b[4];
        deq4(hf ? w.y : w.x, a);   // gate rows g: consume_kb's ag[4 hf .. 4 hf + 3]
        deq4(hf ? w.w : w.z, b);   // rows g + 8:  consume_kb's a8[4 hf .. 4 hf + 3]
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) { a[jj] = bmul2(a[jj], Sg); b[jj] = bmul2(b[jj], S8); }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
          const uint4 xv = *reinterpret_cast<const uint4*>(xrow + q * 8 * XST + i * 64 + 8 * hf);   // x0 / x1
          mma16816(acc[q], a[0], b[0], a[1], b[1], xv.x, xv.y);
          mma16816(acc[q], a[2], b[2], a[3], b[3], xv.z, xv.w);
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  it_ref = it;
  constexpr bool NOSWI = (MODE == 1 && NQ >= 3);
  const bool whole = s0 == 0 && s1 == 16;
  if (!MB && NOSWI && whole && !ho) { atomicExch(P.err, 1); atomicExch(P.err + 3, 1); }   // barrier-id budget exhausted
  flush_item<NQ, MODE, NOSWI>(P, acc, whole, e, rb, c0, ntok, npart, MB ? false : (NOSWI ? (ho || !whole) : !whole));
}

template <int NCH, int NST, int MODE>
__device__ void gemm36(const Params& P, Route& S, const int16_t* s_list, uint8_t* ring, uint64_t* bars, bool pdl,
                       const int* pr) {
  constexpr int STG = SW + 8 * NCH * XROW;
  constexpr int NRB = (MODE == 1) ? 64 : 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;
  const uint32_t ring_a = sa(ring);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);

  if (warp == 8) {
    // K2's X copies (interS from K1) are deferred until griddepcontrol.wait, taken lazily as in v3.3; weight copies
    int it = 0;
    bool waited = !pdl;
    int nd = 0;
    const bool hint = (P.probe & 16) != 0;
    const uint64_t pol = policy_evict_first();
    uint32_t dDst[NST], dBar[NST];
    const __nv_bfloat16* dSrc[NST];
    uint32_t dB[NST];
    const int r = (lane + 31) & 31;
    for (int sg = 0; sg < 4; ++sg) {
      const int xb = pr[3 * sg], xe = pr[3 * sg + 1];
      if (xe <= xb) continue;
      const bool hk = pr[3 * sg + 2] != 0;
      const int* sb = hk ? S.sbh : S.sbc;
      const int TP = hk ? 8 * NCH : 8;
      {
        int e = 0;
        for (int x = xb; x < xe;) {
          while (sb[e + 1] <= x) ++e;
          const int np = S.np[e];
          const int l = x - sb[e];
          const int item = l >> 4;
          const int s0 = l & 15;
          const int s1 = min(16, s0 + (xe - x));
          const int rb = item / np, j = item - rb * np;
          const int ntok = min(TP, S.cnt[e] - j * TP);
          const int c0 = S.off[e] + j * TP;
          const uint8_t* wb = wsrc + ((size_t)(e * NRB + rb) * 16) * SW;
          int tok = 0;
          if (MODE == 1 && r < ntok) tok = ((int)s_list[c0 + r]) >> 3;
          const uint32_t xbytes = (MODE == 1) ? ntok * 512 : ntok * XROW;
          for (int s = s0; s < s1; ++s, ++it) {
            const int slot = it % NST;
            const uint32_t par = ((it / NST) & 1) ^ 1;
            if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now
              pdl_wait();
              asm volatile("fence.proxy.async;\n" ::: "memory");
              waited = true;
              if (lane == 0) {
#pragma unroll
                for (int d = 0; d < NST; ++d)
                  if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
              }
              nd = 0;
            }
            mbar_wait(empty_a + 8 * slot, par, P.err);
            const uint32_t dst = ring_a + slot * STG;
            const uint32_t fb = full_a + 8 * slot;
            if (lane == 0) {
              mbar_expect_tx(fb, SW + xbytes);
              const uint8_t* src = wb + (size_t)s * SW;
              if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);
              else bulk_g2s(dst, src, SW, fb);
            }
            mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals
            __syncwarp();
            if (MODE == 1) {
              if (r < ntok) bulk_g2s(dst + SW + r * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);
            } else {
              const __nv_bfloat16* src = P.inter + ((size_t)s * (8 * M3) + c0) * XST;
              if (waited) {
                if (lane == 0) bulk_g2s(dst + SW, src, xbytes, fb);
              } else {
#pragma unroll
                for (int d = 0; d < NST; ++d)
                  if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = src; dB[d] = xbytes; }
                ++nd;
              }
            }
          }
          x += s1 - s0;
        }
      }
    }
    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain
    if (!waited) {
      pdl_wait();
      asm volatile("fence.proxy.async;\n" ::: "memory");
      if (lane == 0) {
#pragma unroll
        for (int d = 0; d < NST; ++d)
          if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
      }
    }
    return;
  }

  if (warp == 9) {
    int npart = 0;
    for (int sg = 0; sg < 4; ++sg) {
      const int xb = pr[3 * sg], xe = pr[3 * sg + 1];
      if (xe <= xb) continue;
      const bool hk = pr[3 * sg + 2] != 0;
      const int* sb = hk ? S.sbh : S.sbc;
      const int TP = hk ? 8 * NCH : 8;
      {
        int e = 0;
        for (int x = xb; x < xe;) {
          while (sb[e + 1] <= x) ++e;
          const int np = S.np[e];
          const int l = x - sb[e];
          const int item = l >> 4;
          const int s0 = l & 15;
          const int s1 = min(16, s0 + (xe - x));
          x += s1 - s0;
          const int rb = item / np, j = item - rb * np;
          const int ntok = min(TP, S.cnt[e] - j * TP);
          const int c0 = S.off[e] + j * TP;
          const bool ho = (MODE == 1) && hk && ((ntok + 7) >> 3) >= 3 && npart < HO_BUDGET;   // = the consumers' ho
          if (s0 == 0 && s1 == 16 && !ho) continue;
          asm volatile("bar.sync %0, 288;\n" :: "r"(2 + npart) : "memory");
          ++npart;
          __threadfence();   // cumulative over the consumers' h / y atomics (ordered before us by the barrier)
          int last = 0;
          if (lane == 0) {
            int* cp = ((MODE == 1) ? P.cnt : P.cnt2) + (e * NRB + rb) * 64 + j;
            const int old = atomicAdd(cp, s1 - s0);
            last = (old + (s1 - s0) == 16);
            if (last) atomicExch(cp, 0);
          }
          last = __shfl_sync(FULL, last, 0);
          if (MODE == 2 && last) {
            __threadfence();
            const int col = rb * 128 + 4 * lane;
            const float4 wv = __ldg(reinterpret_cast<const float4*>(P.nw + col));
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float4 yv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u)
                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                float pz = 0.f;
                if (n0 + u < ntok) {
                  float z;
                  z = bf16r(yv[u].x) * wv.x; pz += z * z;  z = bf16r(yv[u].y) * wv.y; pz += z * z;
                  z = bf16r(yv[u].z) * wv.z; pz += z * z;  z = bf16r(yv[u].w) * wv.w; pz += z * z;
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pz += __shfl_xor_sync(FULL, pz, o);
                if (lane == 0 && n0 + u < ntok) atomicAdd(P.ss + c0 + n0 + u, pz);
              }
            }
          }
          if (MODE == 1 && last) {
            __threadfence();
            const int cl = 2 * lane;
            const int col = rb * 64 + cl;
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float2 gv[4], uv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const float* hp = P.h + (size_t)(c0 + n0 + u) * (2 * KDIM) + col;
                  gv[u] = __ldcg(reinterpret_cast<const float2*>(hp));
                  uv[u] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
                }
              }
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const int c = c0 + n0 + u;
                  float* hp = P.h + (size_t)c * (2 * KDIM) + col;
                  const uint32_t pk = pack_bf2(swiglu(gv[u].x, uv[u].x), swiglu(gv[u].y, uv[u].y));
                  *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
                  *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
                  *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
                }
              }
            }
          }
        }
      }
    }
    return;
  }

  if (MODE == 1) {
    const size_t n4 = ((P.probe >> PXB) & 1) ? (size_t)0 : (size_t)S.R * KDIM / 4;   // PXB is never set (PX removed)
    float4* yp = reinterpret_cast<float4*>(P.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)blockIdx.x * 256 + tid; i < n4; i += (size_t)gridDim.x * 256) yp[i] = z;
    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;
  }
  int it = 0, npart = 0;
  for (int sg = 0; sg < 4; ++sg) {
    const int xb = pr[3 * sg], xe = pr[3 * sg + 1];
    if (xe <= xb) continue;
    const bool hk = pr[3 * sg + 2] != 0;
    const int* sb = hk ? S.sbh : S.sbc;
    const int TP = hk ? 8 * NCH : 8;
    {
      int e = 0;
      for (int x = xb; x < xe;) {
        while (sb[e + 1] <= x) ++e;
        const int np = S.np[e];
        const int l = x - sb[e];
        const int item = l >> 4;
        const int s0 = l & 15;
        const int s1 = min(16, s0 + (xe - x));
        const int rb = item / np, j = item - rb * np;
        const int ntok = min(TP, S.cnt[e] - j * TP);
        const int nch = (ntok + 7) >> 3;
        const int c0 = S.off[e] + j * TP;
        const bool ho = (MODE == 1) && hk && nch >= 3 && npart < HO_BUDGET;
        if (nch <= 1) {
          cold_item<NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart);
        } else if constexpr (NCH >= 4) {
          if (nch >= 4) hc_item<4, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, ho);
          else if (nch == 3) hc_item<3, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, ho);
          else hc_item<2, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, false);
        } else {
          hc_item<2, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, false);
        }
        if (!(s0 == 0 && s1 == 16) || ho) ++npart;
        x += s1 - s0;
      }
    }
  }
}

__device__ __forceinline__ int m32_pos(const Route& S, int T, int W, int lane, int incl, int excl, int ws, int np) {
  if (T >= W) return S.sbh[32];
  const unsigned bal = __ballot_sync(0xffffffffu, incl > T);
  const int e = __ffs(bal) - 1;
  const int p = (lane == e) ? S.sbh[lane] + (int)(((long long)(T - excl) * np) / ws) : 0;
  return __shfl_sync(0xffffffffu, p, e);
}
__device__ __forceinline__ void m32_cut(const Params& P, const Route& S, int c, int G, int tp, int& beg, int& end) {
  const int lane = threadIdx.x & 31;
  const int span = S.sbh[lane + 1] - S.sbh[lane];
  int ws = 16, np = 1, cost = 0;
  if (span > 0) {
    np = S.np[lane];
    const int n = S.cnt[lane];
    ws = 0;
    for (int j = 0; j < np; ++j) { const int nt = min(tp, n - j * tp); ws += 16 + P.hot_w4 * (((nt + 7) >> 3) - 1); }
    cost = span / np * ws;
  }
  int incl = cost;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
  const int W = __shfl_sync(0xffffffffu, incl, 31), excl = incl - cost;
  beg = (c == 0) ? 0 : m32_pos(S, (int)((long long)W * c / G), W, lane, incl, excl, ws, np);
  end = (c + 1 == G) ? S.sbh[32] : m32_pos(S, (int)((long long)W * (c + 1) / G), W, lane, incl, excl, ws, np);
}

template <int NCH>
__device__ __forceinline__ void ranges36(const Params& P, const Route& S, int& b0, int& e0, int& k0, int& b1, int& e1,
                                         int& k1) {
  const int G = (int)gridDim.x, c = (int)blockIdx.x;
  const long long Sh = S.sbh[32], Sc = S.sbc[32];
  if (P.hot_mode & 1) {
    int hb, he, cb, ce;
    my_range2(Sh, c, G, hb, he);
    my_range2(Sc, c, G, cb, ce);
    const bool cold_first = (P.hot_mode & 2) && (2 * c >= G);
    k0 = cold_first ? 0 : 1; b0 = cold_first ? cb : hb; e0 = cold_first ? ce : he;
    k1 = 1 - k0;             b1 = cold_first ? hb : cb; e1 = cold_first ? he : ce;
    return;
  }
  int Gh = 0;
  if (Sh > 0) {
    long long Wh = 0, Wc = 0;
    if ((P.probe >> 7) & 1) {   // RS1: warp 0, lane = expert, exact integer butterfly sum (v3.7 segs37's scheme)
      const int n = S.cnt[threadIdx.x & 31];
      if (n > 0) {
        if (n > P.hot_t) {
          const int tp = 8 * NCH, full = n / tp, rem = n - full * tp;
          Wh = (long long)full * (16 + P.hot_w4 * (NCH - 1)) + (rem ? 16 + P.hot_w4 * (((rem + 7) >> 3) - 1) : 0);
        } else {
          Wc = 16ll * ((n + 7) >> 3);
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) { Wh += __shfl_xor_sync(0xffffffffu, Wh, o); Wc += __shfl_xor_sync(0xffffffffu, Wc, o); }
    } else
    for (int x = 0; x < 32; ++x) {
      const int n = S.cnt[x];
      if (n == 0) continue;
      if (n > P.hot_t) {
        const int tp = 8 * NCH, full = n / tp, rem = n - full * tp;
        Wh += (long long)full * (16 + P.hot_w4 * (NCH - 1)) + (rem ? 16 + P.hot_w4 * (((rem + 7) >> 3) - 1) : 0);
      } else {
        Wc += 16ll * ((n + 7) >> 3);
      }
    }
    Gh = (Sc > 0) ? (int)((Wh * G + (Wh + Wc) / 2) / (Wh + Wc)) : G;
    Gh = max(1, min(Gh, Sc > 0 ? G - 1 : G));
  }
  const bool hr = c < Gh;
  k0 = hr ? 1 : 0;
  if (hr) {
    if (((P.probe >> 27) & 1) && ((P.probe >> 7) & 1)) m32_cut(P, S, c, Gh, 8 * NCH, b0, e0);   // M32 weighted hot cut
    else my_range2(Sh, c, Gh, b0, e0);
  } else {
    my_range2(Sc, c - Gh, G - Gh, b0, e0);
  }
  k1 = 0; b1 = 0; e1 = 0;
}

template <int NCH>
__device__ __forceinline__ void segs36(const Params& P, const Route& S, int* sg) {
  int b[2], e[2], k[2];
  ranges36<NCH>(P, S, b[0], e[0], k[0], b[1], e[1], k[1]);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const int mid = tail_first2(k[part] ? S.sbh : S.sbc, b[part], e[part], (P.probe >> 7) & 1);
    if ((threadIdx.x & 31) == 0) {   // RS1: segs36 may run on warp 0
      sg[6 * part + 0] = mid;     sg[6 * part + 1] = e[part]; sg[6 * part + 2] = k[part];
      sg[6 * part + 3] = b[part]; sg[6 * part + 4] = mid;     sg[6 * part + 5] = k[part];
    }
  }
}


template <int NCH, int NST>
__global__ void __launch_bounds__(320, 2) k1_kernel36(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NCH * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<1, NST>(bars);
  pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 3)
  route<1, 8 * NCH, true>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  __shared__ int s_pr[12];
  if (threadIdx.x < (((P.probe >> 7) & 1) ? 32 : 1)) segs36<NCH>(P, S, s_pr);   // RS1: warp 0
  __syncthreads();
  gemm36<NCH, NST, 1>(P, S, s_list, smem, bars, false, s_pr);
}

template <int NCH, int NST>
__global__ void __launch_bounds__(320, 2) k2_kernel36(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NCH * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<1, NST>(bars);
  route<1, 8 * NCH, true>(P, S, nullptr, s_list, 32, false);
  __shared__ int s_pr[12];
  if (threadIdx.x < (((P.probe >> 7) & 1) ? 32 : 1)) segs36<NCH>(P, S, s_pr);   // RS1: warp 0
  __syncthreads();
  gemm36<NCH, NST, 2>(P, S, s_list, smem, bars, true, s_pr);
}

constexpr int NE37 = 8;
constexpr int DQ37 = 120;

template <int NC, int NST, int MINB, bool HOTCAP, bool SK1 = false>
__global__ void __launch_bounds__(320, MINB) k1_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = HOTCAP ? SW + 32768 : SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  if constexpr (!HOTCAP) setup_bars<NC, NST>(bars);
  if constexpr (!HOTCAP) pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 0 / 1)
  if constexpr (SK1) route_fr<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0); else route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  if constexpr (HOTCAP) {
    constexpr int STGC = SW + 8 * NC * XROW, NSTC = (NST * STG) / STGC;
    int cta, ncta;
    const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);
    if (threadIdx.x == 0) {
      const int n = hr ? NST : NSTC;
      for (int i = 0; i < n; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + n + i), 8); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (hr) gemm<NC, NST, 1, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);
    else gemm<NC, NSTC, 1, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);
  } else {
    gemm<NC, NST, 1, 0, STG>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);
  }
}

// K2: route + W2. PDL: weights of the first NST stages are requested before griddepcontrol.wait.
template <int NC, int NST, int MINB, bool HOTCAP, bool SK1 = false>
__global__ void __launch_bounds__(320, MINB) k2_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = HOTCAP ? SW + 32768 : SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  if constexpr (!HOTCAP) setup_bars<NC, NST>(bars);
  if constexpr (SK1) route_fr<NC>(P, S, nullptr, s_list, 32, false); else route<NC>(P, S, nullptr, s_list, 32, false);
  if constexpr (HOTCAP) {
    constexpr int STGC = SW + 8 * NC * XROW, NSTC = (NST * STG) / STGC;
    int cta, ncta;
    const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);
    if (threadIdx.x == 0) {
      const int n = hr ? NST : NSTC;
      for (int i = 0; i < n; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + n + i), 8); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (hr) gemm<NC, NST, 2, 1, STG, SK1>(P, S, s_list, smem, bars, true, cta, ncta);
    else gemm<NC, NSTC, 2, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);
  } else {
    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);
  }
}

template <int NC, bool SK1 = false>
__global__ void __launch_bounds__(256) p0_kernel(const Params P) {
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  if constexpr (SK1) route_fr<NC>(P, S, nullptr, s_list, 64, false); else route<NC>(P, S, nullptr, s_list, 64, false);
  int tot = 0;
  for (int e = 0; e < 32; ++e) if (S.cnt[e] > P.hot_t) tot += S.cnt[e];
  uint32_t* xw = reinterpret_cast<uint32_t*>(P.xw);
  for (int idx = blockIdx.x * 256 + threadIdx.x; idx < tot * 256; idx += gridDim.x * 256) {
    int r = idx >> 8;
    const int piece = idx & 255, s2 = piece >> 4, i = (piece >> 2) & 3, t = piece & 3;
    int e = 0;
    for (; e < 32; ++e) {
      if (S.cnt[e] <= P.hot_t) continue;
      if (r < S.cnt[e]) break;
      r -= S.cnt[e];
    }
    const int j = r >> 6, n = r & 63;
    const int tok = ((int)s_list[S.off[e] + r]) >> 3;
    const uint4* src = reinterpret_cast<const uint4*>(P.x + (size_t)tok * KDIM + s2 * 256 + 64 * i + 16 * t);
    const uint4 a = __ldg(src), b = __ldg(src + 1);
    const uint32_t v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    const size_t u0 = xw_u32(S.hp[e] + j, 64, s2, n, i, t, 0);
#pragma unroll
    for (int q = 0; q < 8; ++q) xw[u0 + 32 * q] = v[q];
  }
}

__global__ void __launch_bounds__(256) k3_kernel(const Params P) {
  __shared__ int s_fc[8];
  __shared__ float s_fr[8];
  if ((P.probe >> 25) & 1) pdl_launch();   // SU1: the staged unpooler may launch now (its weights do not depend on K3)
  pdl_wait();
  const int tid = threadIdx.x;
  const int items = 4 * P.M;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int tok = it >> 2;
    if (tid < 8) {
      const int c = __ldcg(P.meta + MH + tok * 8 + tid);
      const float wt = __ldg(P.wts + tok * 8 + tid);
      s_fc[tid] = c;
      s_fr[tid] = (c >= 0) ? rsqrtf(__ldcg(P.ss + c) * (1.f / 4096.f) + 1e-8f) * wt : 0.f;
    }
    __syncthreads();
    const int col = (it & 3) * 1024 + 4 * tid;
    float4 v[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int c = s_fc[s];
      v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
    }
    const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const float r = s_fr[s];
      a0 += bf16r(v[s].x) * w.x * r; a1 += bf16r(v[s].y) * w.y * r;
      a2 += bf16r(v[s].z) * w.z * r; a3 += bf16r(v[s].w) * w.w * r;
    }
    uint2 o;
    o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
    *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;
    __syncthreads();
  }
}

__global__ void __launch_bounds__(256) k3_kernel_b(const Params P) {
  __shared__ int s_fc[8];
  __shared__ float s_fw[8];
  __shared__ float s_fr[8];
  if ((P.probe >> 25) & 1) pdl_launch();   // SU1: the staged unpooler may launch now (its weights do not depend on K3)
  pdl_wait();
  const int tid = threadIdx.x;
  const int items = 4 * P.M;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int tok = it >> 2;
    if (tid < 8) {
      s_fc[tid] = __ldcg(P.meta + MH + tok * 8 + tid);
      s_fw[tid] = __ldg(P.wts + tok * 8 + tid);
    }
    __syncthreads();
    const int col = (it & 3) * 1024 + 4 * tid;
    float ssv = 0.f;
    if (tid < 8 && s_fc[tid] >= 0) ssv = __ldcg(P.ss + s_fc[tid]);   // issued, consumed after the y loads
    float4 v[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int c = s_fc[s];
      v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
    }
    const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
    if (tid < 8) s_fr[tid] = (s_fc[tid] >= 0) ? rsqrtf(ssv * (1.f / 4096.f) + 1e-8f) * s_fw[tid] : 0.f;
    __syncthreads();
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const float r = s_fr[s];
      a0 += bf16r(v[s].x) * w.x * r; a1 += bf16r(v[s].y) * w.y * r;
      a2 += bf16r(v[s].z) * w.z * r; a3 += bf16r(v[s].w) * w.w * r;
    }
    uint2 o;
    o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
    *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;
    __syncthreads();
  }
}

}  // namespace k3

template <int NC, int NST, bool HOT = false>
static constexpr int smem_bytes() { return NST * (k3::SW + 8 * NC * (HOT ? 512 : k3::XROW)) + 2 * NST * 8; }

template <int NC, int NST, int MINB, bool HOTCAP>
static constexpr int smem_bytes3() {
  return HOTCAP ? NST * (k3::SW + 32768) + 2 * 8 * (NST * (k3::SW + 32768) / (k3::SW + 8 * NC * k3::XROW))
                : NST * (k3::SW + 8 * NC * k3::XROW) + 2 * NST * 8;
}

template <int NC, int NST, int MINB, int HOTCAP>
static void launch3(const k3::Params& p, int G, int G3, int mask) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes3<NC, NST, MINB, (bool)HOTCAP>();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if (mask & 1) {
    if (HOTCAP) {
      k3::p0_kernel<NC, HOTCAP == 5><<<G3, 256, 0, st>>>(p);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP, HOTCAP == 5>, p));
    } else {
      cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
      at[0].val.programmaticStreamSerializationAllowed = (int)((p.probe >> 24) & 1);   // v4.7: K1 after the tail
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP, HOTCAP == 5>, p));
      at[0].val.programmaticStreamSerializationAllowed = 1;
    }
  }
  if (mask & 2) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP, HOTCAP == 5>, p));
  }
  if (mask & 4) {
    cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
    if ((p.probe >> 31) & 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel_b, p));   /* K3P */ else C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));
  }
}

template <int NC, int NST, int MINB, int HOTCAP>
static std::vector<int64_t> attrs3() {
  constexpr int SM = smem_bytes3<NC, NST, MINB, (bool)HOTCAP>();
  std::vector<int64_t> r;
  auto one = [&](const void* f, int nt) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, SM) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM), (int64_t)occ});
  };
  one((const void*)k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP, HOTCAP == 5>, 320);
  one((const void*)k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP, HOTCAP == 5>, 320);
  return r;
}

template <int NCH, int NST>
static constexpr int smem_bytes36() { return NST * (k3::SW + 8 * NCH * k3::XROW) + 2 * NST * 8; }

template <int NCH, int NST>
static void launch36(const k3::Params& p, int G, int G3, int mask) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes36<NCH, NST>();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if (mask & 1) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    at[0].val.programmaticStreamSerializationAllowed = (int)((p.probe >> 24) & 1);   // v4.7: K1 after the tail
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel36<NCH, NST>, p));
    at[0].val.programmaticStreamSerializationAllowed = 1;
  }
  if (mask & 2) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel36<NCH, NST>, p));
  }
  if (mask & 4) {
    cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
    if ((p.probe >> 31) & 1) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel_b, p));   /* K3P */ else C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));
  }
}

template <int NCH, int NST>
static std::vector<int64_t> attrs36() {
  constexpr int SM = smem_bytes36<NCH, NST>();
  std::vector<int64_t> r;
  auto one = [&](const void* f, int nt) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, SM) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM), (int64_t)occ});
  };
  one((const void*)k3::k1_kernel36<NCH, NST>, 320);
  one((const void*)k3::k2_kernel36<NCH, NST>, 320);
  return r;
}

template <int NCH, int NST>
static void launch37(const k3::Params& p, int G, int G3, int mask) {
  TORCH_CHECK(false, "K2A v3.7 removed (VML1 serves bucket 128)");
}

template <int NCH, int NST>
static std::vector<int64_t> attrs37() { return {0, 0, 0, 2, 0, 0, 0, 2}; }


void run3(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
          torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
          torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
          torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw,
          torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe) {
  k3::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ids = ids.data_ptr<int>();
  p.wts = wts.data_ptr<float>();
  p.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  p.ws2 = reinterpret_cast<const uint8_t*>(ws2.data_ptr());
  p.gsc13 = gsc13.data_ptr<float>();
  p.gsc2 = gsc2.data_ptr<float>();
  p.nw = nw.data_ptr<float>();
  p.h = h.data_ptr<float>();
  p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  p.y = y.data_ptr<float>();
  p.meta = meta.data_ptr<int>();
  p.cnt = cnt.data_ptr<int>();
  p.err = err.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.sink = reinterpret_cast<unsigned*>(sink.data_ptr());
  p.ss = ss.data_ptr<float>();
  p.cnt2 = cnt2.data_ptr<int>();
  p.xw = reinterpret_cast<uint8_t*>(xw.data_ptr());
  p.xw2 = reinterpret_cast<uint8_t*>(xw2.data_ptr());
  p.hot_t = (int)(hot_t % 1000);
  p.hot_w4 = (hot_t >= 1000) ? (int)((hot_t / 1000) % 100) : 8;   // x8 (v3.3) / sixteenths (v3.6 role split)
  p.hot_mode = (int)(hot_t / 100000);                             // v3.6 balance mode (HK2 only)
  p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;
  p.rdy = nullptr; p.pxmk = nullptr; p.pxd = 4;
  switch (vid) {
@CASES3@
    default: TORCH_CHECK(false, "k3: unknown variant");
  }
}

std::vector<int64_t> attrs3v(int64_t vid) {
  switch (vid) {
@ATTRS3@
    default: return std::vector<int64_t>{-1};
  }
}
"""

K2A33_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void run3(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
          torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
          torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
          torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw,
          torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe);
std::vector<int64_t> attrs3v(int64_t vid);
"""

def _k2a33_wg_specs():
    out = []
    for n in (16, 32, 48):
        k = n // 2
        regs = ",".join("%%%d" % i for i in range(k))
        dl = ", ".join('"+f"(d[%d])' % i for i in range(k))
        out.append(
            "template <> __device__ __forceinline__ void wgmma_rs<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
            "uint32_t a3, uint64_t db) {\n  asm volatile(K3_WG_HEAD \"m64n%dk16.f32.bf16.bf16 {%s}, {%%%d,%%%d,%%%d,%%%d}, "
            "%%%d, p, 1, 1, 0;\\n}\\n\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(db));\n}"
            % (n, n, regs, k, k + 1, k + 2, k + 3, k + 4, dl))
    return "\n".join(out)

def _k2a33_build_ext(name, variants, verbose=False):
    from torch.utils.cpp_extension import load_inline
    cases, attrs = _k2a36_variant_cases(variants)
    if list(variants) == list(K2A33_VARIANTS):
        assert (cases, attrs) == (_k2a36_gate_cases(), _k2a36_gate_attrs()), "gate cases out of date"
    src = K2A33_KERNEL_SRC.replace("@CASES3@", cases).replace("@ATTRS3@", attrs).replace("@WGSPECS@", _k2a33_wg_specs())
    tag = hashlib.sha1((K2A33_CPP_SRC + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=K2A33_CPP_SRC, cuda_sources=src, functions=["run3", "attrs3v"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"], verbose=verbose)
    return mod, time.time() - t0

_K2A33_OWN_BUILD = _k2a33_build_ext


def _k2a33_prepare_st(weights, st2=None, variants=K2A33_VARIANTS, name="k2a_v37", log=print, share=None):
    dev = weights["w13"].device
    t0 = time.time()
    if share is not None:
        ws13, ws2 = share["ws13"], share["ws2"]
    else:
        ws13 = torch.empty((K2A_E_LOC, 64, 16, K2A33_SW3), dtype=torch.uint8, device=dev)
        ws2 = torch.empty((K2A_E_LOC, 32, 16, K2A33_SW3), dtype=torch.uint8, device=dev)
        for e in range(K2A_E_LOC):
            w13q, s13q = _k2a31_prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True, bitpos=K2A31_BITPOS3)
            w2q, s2q = _k2a31_prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False, bitpos=K2A31_BITPOS3)
            ws13[e] = _k2a33_stage_repack(w13q, s13q)
            ws2[e] = _k2a33_stage_repack(w2q, s2q)
    st = dict(
        ws13=ws13, ws2=ws2,
        gsc13=(weights["w13_gs"][:K2A_E_LOC].float() * K2A_W_SHIFT).contiguous(),
        gsc2=(weights["w2_gs"][:K2A_E_LOC].float() * K2A_W_SHIFT).contiguous(),
        nw=(weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous(),
        h=torch.zeros((8 * K2A33_M3, 2 * K2A_D_EXPERT), dtype=torch.float32, device=dev),
        inter=torch.zeros((16 * 8 * K2A33_M3 * 264,), dtype=torch.bfloat16, device=dev),
        y=torch.zeros((8 * K2A33_M3, K2A_D_HIDDEN), dtype=torch.float32, device=dev),
        meta=torch.zeros((K2A_META_HDR + 8 * K2A33_M3,), dtype=torch.int32, device=dev),
        cnt=torch.zeros((K2A_E_LOC * 64 * 64,), dtype=torch.int32, device=dev),
        err=torch.zeros((4,), dtype=torch.int32, device=dev),
        sink=torch.zeros((4,), dtype=torch.int32, device=dev),
        ss=torch.zeros((8 * K2A33_M3,), dtype=torch.float32, device=dev),
        cnt2=torch.zeros((K2A_E_LOC * 32 * 64,), dtype=torch.int32, device=dev),
        xw=torch.zeros(((8 * K2A33_M3 // (K2A33_HOT_T + 1) + 32) * 16 * 64 * 512,), dtype=torch.uint8, device=dev),
        xw2=torch.zeros(((8 * K2A33_M3 // (K2A33_HOT_T + 1) + 32) * 16 * 64 * 512,), dtype=torch.uint8, device=dev),
    )
    torch.cuda.synchronize()
    log(f"[k2a3] stage repack in {time.time() - t0:.1f}s")
    mod, dt = _k2a33_build_ext(name, variants)
    log(f"[k2a3] compiled {len(variants)} variants in {dt:.1f}s")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = []
    for i, v in enumerate(variants):
        a = mod.attrs3v(i)
        occ = max(1, min(a[3], a[7], v[2]))
        grid.append(sms * occ)
        log(f"[k2a3] v{i} {tuple(v)}: K1 regs {a[0]} spill {a[1]} smem {a[2]} occ {a[3]} | K2 regs {a[4]} "
            f"spill {a[5]} smem {a[6]} occ {a[7]} -> grid {grid[-1]}"
            + ("" if min(a[3], a[7]) >= v[2] else f"  <-- WARNING: occupancy below {v[2]} CTAs/SM"))
    st.update(mod=mod, grid=grid, sms=sms, variants=variants)
    return st

K2A47_K1_PDL = True
K2A47_K1_PDL_T128 = False
K2A33_HINT = 16

def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):
    v = st["variants"][vid]
    hot_t = (K2A33_HOT_T if hot_t is None else hot_t) if (len(v) >= 4 and v[3]) else 0
    hot_t += 1000 * (K2A33_HOT_W4 if hot_w4 is None else hot_w4)
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"],
                   st["inter"], st["y"], st["meta"], st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"],
                   st["xw"], st["xw2"], hot_t, m,
                   data["config"]["local_start"], st["grid"][vid], _k3p_g3(st, m, _su1_g3(st, m)), probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0) | _k3p_bits(m) | ((1 << 7) if RS1_ENABLE else 0))
    return out

def _k2a33_reset(st):
    st["h"].zero_()
    st["cnt"].zero_()
    st["cnt2"].zero_()
    torch.cuda.synchronize()

K2A33_SELF_TEST_CASES = ((1, 11, "normal"), (5, 12, "normal"), (16, 17, "normal"), (33, 13, "normal"),
                      (100, 14, "normal"), (128, 15, "skewed"), (257, 16, "normal"), (320, 18, "normal"))


def _v11_k2a33_prepare(weights, t0):
    if not any(v == "v33" for v in V11_K2A33_ROUTE.values()):
        return None
    dev = weights["w13"].device
    w = {"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
         "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
         "post_norm_w": weights["post_expert_norm"]}
    t1 = time.time()
    try:
        global _k2a33_build_ext
        _k2a33_build_ext = _v11_k2a33_build
        st = _k2a33_prepare_st(w, log=lambda s: print(s, flush=True))
        _K3_BUILD_S["k2a33"] = time.time() - t1
    except Exception as ex:
        print("[v8] K2A v3.3 repack/build FAILED: experts stay on K2A v3.1 / v3 / v2.2 / Marlin. Tail:\n" + str(ex)[-8000:], flush=True)
        return None
    ok = {rep: True for _, _, rep in K2A_BUCKETS}
    for bs, seed, dist in K2A33_SELF_TEST_CASES:
        rep = _k2a_bucket_of(bs)
        try:
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            for k in range(2):
                st["err"].zero_()
                got = _v11_k2a33_call(d["hidden_states"], d["topk_ids"], d["topk_weights"], st)
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(got, ref)
                errf = int(st["err"][0])
                good = tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(got.float()).all())
                print(f"[k2a33] layer self-test bs={bs} {dist} v{K2A33_PLAN[rep]} call {k}: rel {tot:.3e} worst row "
                      f"{row:.3e} err={errf} -> {'ok' if good else 'FAIL'}", flush=True)
                if not good:
                    ok[rep] = False
        except Exception as ex:
            print(f"[v8] K2A v3.3 self-test bs={bs} raised {ex!r}", flush=True)
            ok[rep] = False
        _k2a33_reset(st)
    _v14_k2a36_prepare(st, w, dev)
    _v17_k2a37_prepare(st, w, dev)
    _sk1_prepare(st, w, dev)
    _vms_prepare(st, w, dev)
    _vml_prepare(st, w, dev)
    print(f"[v8] K2A v3.3 buckets {ok} in {time.time() - t1:.1f}s; route {V11_K2A33_ROUTE}", flush=True)
    return {"st": st, "ok": ok}


def _sk1_prepare(st, w, dev):
    st["sk1_ok"] = False
    if not SK1_ENABLE:
        return
    try:
        if len(st["variants"]) <= SK1_VID or st["grid"][SK1_VID] != st["grid"][K2A33_PLAN[256]]:
            print("[sk1] vid 6 missing or its grid differs from vid 2: bucket 256 stays on vid 2", flush=True)
            return
        good = True
        for bs, seed, dist in ((200, 31, "skewed"), (256, 32, "normal"), (257, 16, "normal"), (320, 18, "normal")):
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            x = d["hidden_states"].contiguous()
            data = {"hidden_states": x, "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"], "config": {"local_start": 0}}
            st["err"].zero_()
            o2 = _k2a33_call(st, data, K2A33_PLAN[256], hot_t=K2A33_HOT_T_BUCKET.get(256), hot_w4=76).clone()
            for k in range(2):
                o6 = _k2a33_call(st, data, SK1_VID, hot_t=K2A33_HOT_T_BUCKET.get(256), hot_w4=76)
                torch.cuda.synchronize()
                e2, _ = _k2a_rel_err(o6, o2)
                tot, row = _k2a_rel_err(o6, ref)
                errf = int(st["err"][0])
                ok = e2 < 2e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o6.float()).all())
                print(f"[sk1] self-test bs={bs} {dist} call {k}: rel vs vid 2 {e2:.2e} vs FP32 {tot:.3e} worst row {row:.3e} "
                      f"err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            _k2a33_reset(st)
        st["sk1_ok"] = good
        print(f"[sk1] SK1 {'LIVE at bucket 256 (vid 6)' if good else 'OFF: bucket 256 stays on vid 2'}", flush=True)
    except Exception as ex:
        st["sk1_ok"] = False
        print(f"[sk1] self-test raised {ex!r}: bucket 256 stays on vid 2", flush=True)
        try:
            _k2a33_reset(st)
        except Exception:
            pass


VMS_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {   // c, opaque to NVVM: the 3-way gather stays one IMAD
  uint32_t r;
  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));
  return r;
}

#ifndef VMS_NS
#define VMS_NS 3
#endif
#ifndef VMS_MK2
#define VMS_MK2 0
#endif
#ifndef VMS_RO
#define VMS_RO 0
#endif
#ifndef VMS_NRB
#define VMS_NRB 4
#endif
#ifndef VMS_FX
#define VMS_FX 0
#endif
#ifndef VMS_G
#define VMS_G 132
#endif

namespace vms {
constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 128, MAXJ = 64;
constexpr int UB = 36864, UC = 32768, RBA = 512, RBB = 256;
constexpr int NW = 8, NT = NW * 32, NTL = NT + 32, NRB = VMS_NRB, NS = VMS_NS, G = VMS_G;
constexpr int WAIT_MAX = 1 << 22;

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* tw;
  const unsigned char* wA; const unsigned char* wB;
  const float* gsA; const float* gsB; const float* nw;
  __nv_bfloat16* inter; float* ybuf; float* ss; int* ctr;
  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;
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

__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {
  if (q < nA) { const int v = a0 + q; return p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB; }
  const int v = b0 + (q - nA); return p.wB + ((size_t)S.jE[v >> 8] * RBB + (v & 255)) * UB;
}

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
#if VMS_MK2
  long long* mrow = p.marks + (size_t)c * 32;
  long long cF = 0, cE = 0, cS = 0, cC = 0, cW = 0, cP = 0, cU = 0, nE = 0, ck;
  if (mk && tid == 0) { mrow[16] = clock64(); unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); mrow[21] = sm; }
#else
  long long* mrow = p.marks + (size_t)c * 16;
#endif
  if (tid == 0) {
    if (mk) mrow[0] = gt();
#if __CUDA_ARCH__ >= 900
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);
#else
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], NT);
#endif
#if VMS_RO
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], NW); mbar_init(&S.pempty[i], 1); }
#else
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], 1); mbar_init(&S.pempty[i], 1); }
#endif
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
#if (VMS_FX & 4) && __CUDA_ARCH__ >= 900
  if (warp == 0) asm volatile("" :: "l"(p.ids), "l"(p.ctr), "l"(p.wA), "l"(p.wB), "r"(p.m), "r"(p.lo));
#endif
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
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
#if (VMS_FX & 4) && __CUDA_ARCH__ >= 900
    {
      const int nj0 = __shfl_sync(0xffffffffu, jinc, 31);
      if (lane == 0) {
        const int NA0 = nj0 * RBA, NN0 = nj0 * (RBA + RBB);
        const int a00 = c * NA0 / G, nA0 = (c + 1) * NA0 / G - a00;
        const int t00 = c * NN0 / G, N0 = (c + 1) * NN0 / G - t00;
        for (int q = 0; q < NS && q < N0; ++q) issue(unit_src(p, S, q, nA0, a00, t00 - a00), ring + q * UB, &S.full[q], 0);
      }
    }
#endif
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
  if (mk && tid == 0) mrow[1] = gt();
  if (warp < NW) {
#if !((VMS_FX & 4) && __CUDA_ARCH__ >= 900)
    for (int q = 0; q < NS && q < N; ++q) {
#if __CUDA_ARCH__ >= 900
      if (tid == 0)
#endif
      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);
    }
#endif
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
#if VMS_MK2
          ck = clock64();
#endif
#if VMS_FX & 1
          if (lane == 0) {
            int n = 0;
            while (ld_acq(p.ctr + 8 + jj) < RBA) {
              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
              __nanosleep(64);
            }
          }
          __syncwarp();
#else
          if (lane == 0) {
            int n = 0;
            while (ld_rlx(p.ctr + 8 + jj) < RBA) {
              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
              __nanosleep(64);
            }
          }
          __syncwarp();
          (void)ld_acq(p.ctr + 8 + jj);
#endif
          if (mk && tid == 0 && mrow[4] == 0) mrow[4] = gt();
#if VMS_MK2
          cW += clock64() - ck;
#endif
        }
        if (g < cnt) {
          const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID)
                                : reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + g] >> 3) * HD);
          src += warp * 64 + 2 * t;
#pragma unroll
          for (int cc = 0; cc < 8; ++cc) {
            xr[2 * cc] = PB ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);
            xr[2 * cc + 1] = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);
          }
        } else {
#pragma unroll
          for (int cc = 0; cc < 16; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
#if VMS_MK2
      ck = clock64();
#endif
      mbar_wait(&S.full[sl], par);
      if (mk && tid == 0 && q == 0) mrow[2] = gt();
#if VMS_MK2
      { const long long c2 = clock64(); if (q > 0) cF += c2 - ck; ck = c2; }
      if (mk && tid == 0 && q == nA - 1) mrow[15] = gt();
      if (mk && tid == 0 && q == N - 1) mrow[22] = gt();
#endif
      const unsigned char* slot = ring + sl * UB;
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
          const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;   // UNP L2 layout (tools/unp_patches.py)
          const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;
          const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
          mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);
        }
      }
      float4 o;
      o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
      o.y = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
      o.z = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
      o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
#if VMS_RO
      const int bq = q & (NRB - 1);
#if VMS_MK2
      { const long long c2 = clock64(); cC += c2 - ck; ck = c2; }
#endif
      cta_sync_compute();
#if VMS_MK2
      { const long long c2 = clock64(); cS += c2 - ck; ck = c2; }
#endif
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (tid == 0)
#endif
        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);
      }
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
#if VMS_MK2
      { const long long c2 = clock64(); cE += c2 - ck; if (c2 - ck > 200) ++nE; ck = c2; }
#endif
      S.red[bq][warp][lane] = o;
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pfull[bq]);
#else
      const int bq = q & (NRB - 1);
#if VMS_MK2
      { const long long c2 = clock64(); cC += c2 - ck; ck = c2; }
#endif
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
#if VMS_MK2
      { const long long c2 = clock64(); cE += c2 - ck; if (c2 - ck > 200) ++nE; ck = c2; }
#endif
      S.red[bq][warp][lane] = o;
      cta_sync_compute();
#if VMS_MK2
      { const long long c2 = clock64(); cS += c2 - ck; ck = c2; }
#endif
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (tid == 0)
#endif
        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);
      }
      if (tid == 0) mbar_arrive(&S.pfull[bq]);
#endif
      if (++sl == NS) { sl = 0; par ^= 1u; }
    }
    if (mk && tid == 0) mrow[5] = gt();
#if VMS_MK2
    if (mk && tid == 0) {
      mrow[10] = cF; mrow[11] = cE; mrow[12] = cS; mrow[19] = cC; mrow[18] = cW; mrow[20] = nE;
      mrow[13] = (long long)N | ((long long)nA << 16) | ((long long)njobs << 32);
    }
#endif
  } else {
    const int g = lane >> 2, t = lane & 3;
    float* ssc = p.ss + (S.gen & 1) * MAXP;
    for (int q = 0; q < N; ++q) {
      const int bq = q & (NRB - 1);
#if VMS_MK2
      ck = clock64(); if (q > 0) cU += ck - cP;
#endif
      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
#if VMS_MK2
      { const long long c2 = clock64(); cF += c2 - ck; cP = c2; }
#endif
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
#if VMS_FX & 2
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
#else
          __threadfence();
#endif
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
#if VMS_FX & 2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
#else
    __threadfence();
#endif
    if (mk && lane == 0) mrow[7] = gt();
#if VMS_MK2
    if (mk && lane == 0) { mrow[14] = cF; mrow[23] = cU + (clock64() - cP); }
#endif
  }
  __syncthreads();
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  const int gen = S.gen;
  if (tid == 0) {
#if VMS_FX & 2
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p.ctr + 1) : "memory");
#else
    __threadfence();
    atomicAdd(p.ctr + 1, 1);
#endif
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
#if VMS_MK2
  if (mk && tid == 0) mrow[17] = clock64();
#endif
}
}  // namespace vms

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo) {
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
  p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;
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
"""
VMS_CPP_SRC = r"""
#include <torch/extension.h>
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo);
std::vector<int64_t> attrs();
"""

VMS_ENABLE = True
VMS_MAXT = 16
VMS_UB = 36864
VMS_POS = (((6, 7, 8, 15), (22, 23, 24, 31)), ((28, 29, 30, 5), (12, 13, 14, 21)),
           ((25, 26, 27, 2), (9, 10, 11, 18)), ((4, 0, 1, 3), (20, 16, 17, 19)))
_VMS = {"mod": None, "G": None}


def _vms_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vms_{tag}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags)
    _VMS["G"] = G
    _VMS["mod"] = mod
    _K3_BUILD_S["vms"] = time.time() - t1
    return mod


def _vms_pack_words(cu):
    nu = cu.shape[0]
    x = cu.view(nu, 2, 8, 8, 8, 4, 4, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8).contiguous()
    x = x.view(nu, 8, 8, 8, 4, 4, 4, 2).to(torch.int64)
    word = torch.zeros(x.shape[:6], dtype=torch.int64, device=cu.device)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << VMS_POS[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return word.view(nu, 8192)


def _vms_prepack_expert(w, sf, gate_up):
    n = w.shape[0]
    rows = torch.arange(n, device=w.device)
    if gate_up:
        rb, i = rows // 16, rows % 16
        rows = torch.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    wr = w[rows]
    cu = torch.stack((wr & 15, wr >> 4), dim=-1).reshape(n // 16, 16, 4096)
    sb = sf.view(torch.uint8)[rows].view(n // 16, 2, 8, 8, 8, 4).permute(0, 3, 2, 5, 4, 1).contiguous().view(n // 16, 4096)
    words = _vms_pack_words(cu).view(torch.uint8).view(n // 16, 32768)
    return torch.cat((words, sb), dim=1).contiguous()


def _vms_call(st, x, local_ids, topk_weights):
    vs = st["vms"]
    ids = local_ids.to(torch.int32).contiguous()
    tw = topk_weights.to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    vs["mod"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],
                  out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0)
    return out


def _vms_prepare(st, w, dev):
    st["vms_ok"] = False
    if not VMS_ENABLE:
        return
    t1 = time.time()
    try:
        _pt1_wait("vms")
        mod = _VMS["mod"] if _VMS["mod"] is not None else _vms_build()
        if _VMS["G"] != torch.cuda.get_device_properties(0).multi_processor_count:
            print("[vms] built for a different SM count: VMS OFF (K2A at every T)", flush=True)
            return
        b13, b2 = w["w13_sf"].view(torch.uint8), w["w2_sf"].view(torch.uint8)
        if not (bool(((b13 >= 8) & (b13 <= 0x7E)).all()) and bool(((b2 >= 8) & (b2 <= 0x7E)).all())):
            print("[vms] block-scale bytes outside 0x08..0x7E (PRMT scale decode inexact): VMS OFF (K2A at every T)", flush=True)
            return
        E = w["w13"].shape[0]
        if E != K2A_E_LOC or tuple(w["w13"].shape[1:]) != (2 * K2A_D_EXPERT, K2A_D_HIDDEN // 2):
            print("[vms] expert shapes differ: VMS OFF", flush=True)
            return
        wA = torch.empty((E, 512, VMS_UB), dtype=torch.uint8, device=dev)
        wB = torch.empty((E, 256, VMS_UB), dtype=torch.uint8, device=dev)
        for e in range(E):
            wA[e] = _vms_prepack_expert(w["w13"][e], w["w13_sf"][e], True)
            wB[e] = _vms_prepack_expert(w["w2"][e], w["w2_sf"][e], False)
        two62 = float(2 ** 62)
        st["vms"] = {"mod": mod, "wA": wA, "wB": wB, "gsA": (w["w13_gs"].float().reshape(E) * two62).contiguous(),
                     "gsB": (w["w2_gs"].float().reshape(E) * two62).contiguous(),
                     "nw": (1.0 + w["post_norm_w"].float()).contiguous(),
                     "inter": torch.zeros((128, K2A_D_EXPERT), dtype=torch.bfloat16, device=dev),
                     "ybuf": torch.zeros((128, K2A_D_HIDDEN), dtype=torch.float32, device=dev),
                     "ss": torch.zeros((256,), dtype=torch.float32, device=dev),
                     "ctr": torch.zeros((72,), dtype=torch.int32, device=dev),
                     "nomarks": torch.empty((0,), dtype=torch.int64, device=dev)}
        good = True
        for bs, seed, dist in VMS_SELF_TEST_CASES:
            if bs > VMS_MAXT:
                continue
            if dist == "hot8":
                d = _k2a36_synth_inputs(bs, seed, dev, "hot8")
                ids = d["topk_ids"]
                d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
            else:
                d = _k2a_self_test_inputs(bs, seed, dev, dist)
            x = d["hidden_states"].contiguous()
            ref = _k2a_ref_fp32(d, w)
            st["err"].zero_()
            ok_k = _v11_k2a33_call(x, d["topk_ids"], d["topk_weights"], st).clone()
            torch.cuda.synchronize()
            for k in range(2):
                o = _vms_call(st, x, d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                e2, _ = _k2a_rel_err(o, ok_k)
                tot, row = _k2a_rel_err(o, ref)
                errf = int(st["vms"]["ctr"][2])
                ok = e2 < 8e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o.float()).all())
                if not ok or k == 0:
                    print(f"[vms] self-test bs={bs} {dist} call {k}: rel vs K2A {e2:.2e} vs FP32 {tot:.3e} worst row {row:.3e} "
                          f"err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            _k2a33_reset(st)
        st["vms_ok"] = good
        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "
              f"(regs/local/smem/G/NS {list(mod.attrs())}, {time.time() - t1:.1f}s)", flush=True)
    except Exception as ex:
        st["vms_ok"] = False
        print(f"[vms] prepare raised {ex!r}: VMS OFF (K2A at every T)", flush=True)
        try:
            _k2a33_reset(st)
        except Exception:
            pass


VMS_SELF_TEST_CASES = ((1, 41, "normal"), (1, 42, "skewed"), (3, 43, "normal"), (5, 44, "normal"), (8, 45, "normal"),
                       (8, 46, "skewed"), (12, 47, "skewed"), (16, 48, "normal"), (16, 49, "hot8"), (16, 50, "skewed"))


def _v11_k2a33_call(x, local_ids, topk_weights, st):
    x = x.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}
    if VMS_ENABLE and st.get("vms_ok") and x.shape[0] <= VMS_MAXT:
        try:
            return _vms_call(st, x, local_ids, topk_weights)
        except Exception as ex:
            print(f"[k3] VMS route raised ({ex!r}): K2A from here on", flush=True)
            st["vms_ok"] = False
    rep = _k2a_bucket_of(x.shape[0])
    if VML_ENABLE and VML_BUCKETS.get(rep) and _VML["ok"].get(rep) and x.shape[0] <= _vml_MMAX:
        try:
            return _vml_route(st, data)
        except Exception as ex:
            print(f"[k3] VML1 route raised ({ex!r}): K2A from here on", flush=True)
            _VML["ok"] = {}
    r37 = st["r37"].get(rep) if (K2A37_ENABLE and st.get("r37")) else None
    if r37 is not None:
        try:
            return _k2a37_call(st, data, r37[0], r37[1], r37[2])
        except Exception as ex:
            print(f"[k3] K2A v3.7 route raised ({ex!r}): v3.6 / v3.3 routes from here on", flush=True)
            st["r37"].clear()
    r36 = st["r36"].get(rep) if (K2A36_ENABLE and K2A36_BUCKETS.get(rep) and st.get("r36")) else None
    if r36 is not None:
        try:
            return _k2a36_call(st, data, r36[0], r36[1])
        except Exception as ex:
            print(f"[k3] K2A v3.6 route raised ({ex!r}): v3.3's plan from here on", flush=True)
            st["r36"].clear()
    vid = SK1_VID if (SK1_ENABLE and rep == 256 and st.get("sk1_ok")) else K2A33_PLAN[rep]
    return _k2a33_call(st, data, vid, hot_t=K2A33_HOT_T_BUCKET.get(rep),
                       hot_w4=(76 if rep == 256 else None))





_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096
_vml_MMAX = 320
_vml_JMAX = 80
_vml_RMAX = _vml_MMAX * _vml_TOPK
_vml_SLOTS = 136
_vml_FIXW = 8 * 128 * 4
_vml_COST = (128, 135, 147, 160, 171, 182, 193, 204)


def _vml_mag(d):
    return -(-(1 << 40) // d)


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

_vml_CPP = r"""
#include <torch/extension.h>
#include <vector>
void vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V);
std::vector<int64_t> vml_cfg();
"""
_vml_FLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def _vml_build(verbose=False, defines=()):
    from torch.utils.cpp_extension import load_inline
    fl = _vml_FLAGS + list(defines)
    tag = hashlib.sha1((_vml_CPP + _vml_CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vml_moe_{tag}", cpp_sources=_vml_CPP, cuda_sources=_vml_CUDA, functions=["vml_run", "vml_cfg"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


def _vml_alloc(dev, G):
    z = lambda n, dt=torch.int32: torch.zeros(n, dtype=dt, device=dev)
    xrows = _vml_RMAX + 8 * _vml_JMAX
    assert G <= _vml_SLOTS and G <= 136
    return {"xA": z((xrows * _vml_KDIM,), torch.bfloat16), "xB": z((xrows * _vml_KDIM,), torch.bfloat16),
            "y": z((_vml_RMAX * _vml_KDIM,), torch.float32), "fix": z((2 * (_vml_SLOTS + 384) * 4 * max(_vml_FIXW, 10 * 512),), torch.float32),
            "cnt": z((2 * (_vml_SLOTS + 384) * 4,)), "xready": z((_vml_JMAX,)), "doneA": z((_vml_JMAX * 16,)), "rmap": z((_vml_RMAX,)),
            "err": z((4,)), "gbar": z((2,)), "nomark": z((0,), torch.int64), "G": G, "magG": _vml_mag(G)}


def _vml_cost_ext(mod, cost):
    ntm = _vml_cfg(mod)[1]
    assert 8 <= ntm <= 10, "m32e scratch (fix regions of max(FIXW, 5120) floats) supports NTM 8..10 builds"
    c = list(cost)
    if len(c) == 2 * ntm:
        return c
    assert len(c) == 16 and ntm >= 8, (len(c), ntm)
    out = []
    for pb in (0, 1):
        h = c[8 * pb:8 * pb + 8]
        step = max(1, h[7] - h[6])
        out += h + [h[7] + step * (k - 8) for k in range(9, ntm + 1)]
    return out


def _vml_pool_ext(mod, pool):
    return [int(pool[0]), int(pool[1])] if _vml_cfg(mod)[6] else []


def _vml_seg_ext(mod, seg):
    return [int(seg[0]), int(seg[1])] if _vml_cfg(mod)[7] else []


def _vml_pdel_ext(mod, pdel):
    return [int(pdel)] if _vml_cfg(mod)[10] else []


def _vml_cfg(mod):
    c = getattr(mod, "_m32e_cfg", None)
    if c is None:
        c = tuple(int(v) for v in mod.vml_cfg())
        try:
            mod._m32e_cfg = c
        except Exception:
            pass
    return c


_vml_FLAGS_DEFAULT = 1


def _vml_call(mod, k, sc, x, ids, wts, ls=0, marks=None, mask=3, out=None, flags=None, cost=None, pool=(0, 0), seg=(0, 0), pdel=0):
    m = x.shape[0]
    assert m <= _vml_MMAX
    flags = _vml_FLAGS_DEFAULT if flags is None else flags
    if out is None:
        out = torch.empty((m, _vml_KDIM), dtype=torch.bfloat16, device=x.device)
    mod.vml_run([x, ids, wts, k["ws13"], k["ws2"], k["gsc13"], k["gsc2"], k["nw"], sc["xA"], sc["xB"], sc["y"], sc["fix"],
                 sc["cnt"], sc["xready"], sc["doneA"], sc["rmap"], out, sc["err"], sc["nomark"] if marks is None else marks, sc["gbar"]],
                [m, ls, sc["G"], sc["magG"], mask, flags] + _vml_cost_ext(mod, _vml_COST + _vml_COST if cost is None else cost) + _vml_pool_ext(mod, pool) + _vml_seg_ext(mod, seg) + _vml_pdel_ext(mod, pdel))
    return out


VML_ENABLE = True
VML_BUCKETS = {128: True, 256: True, 32: True}
VML_FLAGS = {128: 193, 256: 193, 32: 193}
VML_COST = {256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236], 128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267], 32: [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204]}
VML_VAR = 'bf2'
VML_ABVARS = ()
VML_BVAR = {256: 'w72', 128: 'w72'}
VML_POOL = {}
VML_DPFLAGS = 8192
VML_SEG = {}
VML_SCFLAGS = 0
VML_PDEL = {}
_VML_DEFS = {"": (), "o": ("-DVM_TRICKS=0",), "s": ("-DVM_TRICKS_DQ=0",), "d": ("-DVM_TRICKS_DQ=0", "-DVM_DP=1"),
             "d2": ("-DVM_TRICKS_DQ=0", "-DVM_DP=1", "-DVM_DPI=2"), "c": ("-DVM_TRICKS_DQ=0", "-DVM_SC=1"),
             "b": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1"), "bd": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_DEFER2=1"),
             "e": ("-DVM_TRICKS_DQ=0", "-DVM_DEFER2=1"), "p": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_PS=1"),
             "bf": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1"), "bg": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_GSM=1", "-DVM_FSW=1"),
             "bf2": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_RT=1", "-DVM_FT=1", "-DVM_FTR=1", "-DVM_TA=1"),
             "w72": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1", "-DVM_WG5=2", "-DVM_PREG=32", "-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72", "-DVM_RT=1", "-DVM_FTR=1", "-DVM_LD=2", "-DVM_TA=1", "-DVM_DEFER=1", "-DVCP_DPI=0", "-DVCP_DPH=0", "-DVCP_PK=1")}
_VML = {"mod": None, "mods": {}, "sc": None, "scb": None, "ok": {}, "err": None, "okv": {}, "force": None}


def _vml_build_job():
    import threading
    vs = (VML_VAR,) + tuple(dict.fromkeys(a for a in tuple(VML_ABVARS) + tuple(VML_BVAR.values()) if a != VML_VAR))
    errs = {}

    def one(v):
        try:
            _VML["mods"][v], _ = _vml_build(defines=_VML_DEFS[v])
        except Exception as ex:
            errs[v] = ex
    ths = [threading.Thread(target=one, args=(v,)) for v in vs[1:]]
    for th in ths:
        th.start()
    one(vs[0])
    for th in ths:
        th.join()
    if VML_VAR in errs:
        raise errs[VML_VAR]
    _VML["mod"] = _VML["mods"][VML_VAR]


def _vml_route(st, data):
    x = data["hidden_states"].contiguous()
    rep = _k2a_bucket_of(x.shape[0])
    var = _VML["force"] if _VML["force"] is not None else VML_BVAR.get(rep, VML_VAR)
    if _VML["force"] is None and var != VML_VAR and not _VML["okv"].get((rep, var)):
        var = VML_VAR
    dp, sc = var in ("d", "d2"), var == "c"
    fl = VML_FLAGS.get(rep)
    fl = (_vml_FLAGS_DEFAULT if fl is None else fl) | (VML_DPFLAGS if dp else 0) | (VML_SCFLAGS if sc else 0)
    sg = int(VML_SEG.get(rep, 0)) if sc else 0
    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]
    pdl = int(VML_PDEL.get(rep, 0)) if var == "p" else 0
    return _vml_call(_VML["mods"].get(var, _VML["mod"]), st, scx, x, data["topk_ids"].to(torch.int32).contiguous(),
                     data["topk_weights"].to(torch.float32).contiguous(), int(data["config"]["local_start"]),
                     flags=fl, cost=VML_COST.get(rep), pool=tuple(VML_POOL.get(rep, (0, 0))) if dp else (0, 0), seg=(sg, sg), pdel=pdl)


def _vml_prepare(st, w, dev):
    _VML["ok"] = {}
    if not VML_ENABLE:
        return
    try:
        _pt1_wait("vml")
        _VML["force"] = VML_VAR
        if _VML["mod"] is None:
            _vml_build_job()
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        if sms > _vml_SLOTS:
            print(f"[vml] OFF: {sms} SMs unsupported", flush=True)
            return
        _VML["sc"] = _vml_alloc(dev, sms)
        _VML["scb"] = dict(_VML["sc"], y=torch.zeros_like(_VML["sc"]["y"]))
        cases = {128: ((100, 41, "normal"), (128, 42, "skewed"), (160, 43, "skewed")),
                 256: ((200, 44, "skewed"), (256, 45, "normal"), (257, 46, "skewed"), (320, 47, "normal")),
                 32: ((25, 48, "normal"), (32, 49, "skewed"), (48, 50, "skewed"))}
        for rep, on in VML_BUCKETS.items():
            if not on:
                continue
            good = True
            for bs, seed, dist in cases[rep]:
                d = _k2a_self_test_inputs(bs, seed, dev, dist)
                d = {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                     "topk_weights": d["topk_weights"].float().contiguous(), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
                ref = _k2a_ref_fp32(d, w)
                ok_k = dict(_VML["ok"])
                _VML["ok"] = {}
                o_k = _v11_k2a33_call(d["hidden_states"], d["topk_ids"], d["topk_weights"], st).clone()
                _VML["ok"] = ok_k
                for k in range(2):
                    _VML["sc"]["err"].zero_()
                    o = _vml_route(st, d)
                    torch.cuda.synchronize()
                    ek, _ = _k2a_rel_err(o, o_k)
                    tot, row = _k2a_rel_err(o, ref)
                    errf = int(_VML["sc"]["err"][0])
                    ok = ek < 2e-3 and tot < 1e-2 and row < 5e-2 and not errf and bool(torch.isfinite(o.float()).all())
                    print(f"[vml] self-test bucket {rep} bs={bs} {dist} call {k}: rel vs K2A {ek:.2e} vs FP32 {tot:.3e} "
                          f"worst row {row:.3e} err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                    good &= ok
                _k2a33_reset(st)
            _VML["ok"][rep] = good
        print(f"[vml] VML1 live buckets {[r for r, v in _VML['ok'].items() if v]} (self-test {_VML['ok']})", flush=True)
        _VML["force"] = None
        _m32e_vartest(st, w, dev)
    except Exception as ex:
        _VML["force"] = None
        _VML["ok"] = {}
        _VML["err"] = repr(ex)[:500]
        print(f"[vml] prepare raised {ex!r}: every bucket stays on K2A", flush=True)
        try:
            _k2a33_reset(st)
        except Exception:
            pass


def _m32e_vartest(st, w, dev):
    _VML["okv"] = {}
    cases = {128: ((100, 41, "normal"), (128, 42, "skewed"), (160, 43, "skewed")),
             256: ((200, 44, "skewed"), (256, 45, "normal"), (257, 46, "skewed"), (320, 47, "normal")),
             32: ((25, 48, "normal"), (32, 49, "skewed"), (48, 50, "skewed"))}
    pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS))]
    for rep, var in pairs:
        if var == VML_VAR or not _VML["ok"].get(rep) or var not in _VML["mods"]:
            continue
        good = True
        try:
            for bs, seed, dist in cases[rep]:
                d = _k2a_self_test_inputs(bs, seed, dev, dist)
                d = {"hidden_states": d["hidden_states"].contiguous(), "topk_ids": d["topk_ids"].to(torch.int32).contiguous(),
                     "topk_weights": d["topk_weights"].float().contiguous(), "config": {"local_start": 0, "local_count": K2A_E_LOC}}
                ref = _k2a_ref_fp32(d, w)
                _VML["force"] = VML_VAR
                o_b = _vml_route(st, d).clone()
                for k in range(2):
                    _VML["sc"]["err"].zero_()
                    _VML["force"] = var
                    o = _vml_route(st, d)
                    _VML["force"] = None
                    torch.cuda.synchronize()
                    eb, _ = _k2a_rel_err(o, o_b)
                    tot, row = _k2a_rel_err(o, ref)
                    errf = int(_VML["sc"]["err"][0])
                    ok = eb < 2e-3 and tot < 1e-2 and row < 5e-2 and not errf and bool(torch.isfinite(o.float()).all())
                    print(f"[m32e] self-test bucket {rep} var {var} bs={bs} {dist} call {k}: rel vs {VML_VAR} {eb:.2e} vs FP32 "
                          f"{tot:.3e} worst row {row:.3e} err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                    good &= ok
        except Exception as ex:
            good = False
            print(f"[m32e] self-test bucket {rep} var {var} raised {ex!r}", flush=True)
        finally:
            _VML["force"] = None
        _VML["okv"][(rep, var)] = good
    print(f"[m32e] DP variants live {[k for k, v in _VML['okv'].items() if v]} (self-test {_VML['okv']})", flush=True)

K2A36_ENABLE = True
K2A36_BUCKETS = {16: True, 32: True}
K2A36_PLAN = {16: (3, (8, 0, 8)), 32: (3, (8, 0, 12))}
M32_WSK = True
M32_WSK_W4 = 8
M32_PX = False
M32_PX_D = 4

K2A36_HOT_DEFAULT = (8, 3, 8)

K2A36_SELF_TEST_CASES = ((1, 11, "normal"), (5, 12, "normal"), (12, 23, "skewed"), (16, 17, "normal"), (16, 19, "skewed"),
                      (20, 25, "hot8"), (24, 22, "skewed"), (32, 21, "skewed"), (32, 26, "hot8"), (33, 13, "normal"),
                      (100, 14, "normal"), (128, 15, "skewed"), (257, 16, "normal"), (320, 18, "normal"))

def _k2a36_synth_inputs(bs, seed, device, dist="normal"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, K2A_D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=device)
    if dist == "normal":
        z = torch.randn((256,), generator=gen, device=device)
        u = torch.rand((256,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    elif dist == "hot8":
        scores[:, :8] += 10.0
    else:
        scores = scores + torch.rand((256,), generator=gen, device=device).pow(6) * 4
    top = scores.topk(K2A_TOP_K, dim=-1)
    ids = top.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": K2A_D_HIDDEN, "d_expert": K2A_D_EXPERT, "n_experts": 256,
                   "local_start": 0, "local_count": K2A_E_LOC, "top_k": K2A_TOP_K, "sharding": "ep8"},
    }

def _k2a36_variant_cases(variants):
    cases, attrs = [], []
    for i, v in enumerate(variants):
        nc, nst, minb, hk, nch = (tuple(v) + (0,) * (5 - len(v)))[:5]
        if hk == 2:
            assert nc == 1 and minb == 2 and nch in (2, 4), v
            cases.append("    case %d: launch36<%d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, nch, nst))
            attrs.append("    case %d: return attrs36<%d, %d>();" % (i, nch, nst))
        elif hk == 3:
            assert nc == 1 and minb == 2 and nch in (1, 2, 4), v
            cases.append("    case %d: launch37<%d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, nch, nst))
            attrs.append("    case %d: return attrs37<%d, %d>();" % (i, nch, nst))
        else:
            cases.append("    case %d: launch3<%d, %d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, nc, nst, minb, hk))
            attrs.append("    case %d: return attrs3<%d, %d, %d, %d>();" % (i, nc, nst, minb, hk))
    return "\n".join(cases), "\n".join(attrs)

def _k2a36_gate_cases():
    return ("    case 0: launch3<1, 3, 2, 0>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 1: launch3<2, 3, 2, 0>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 2: launch3<2, 2, 2, 1>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 3: launch36<4, 3>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 4: launch37<1, 3>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 5: launch37<4, 3>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 6: launch3<2, 2, 2, 5>(p, (int)G, (int)G3, (int)mask); break;")

def _k2a36_gate_attrs():
    return ("    case 0: return attrs3<1, 3, 2, 0>();\n"
            "    case 1: return attrs3<2, 3, 2, 0>();\n"
            "    case 2: return attrs3<2, 2, 2, 1>();\n"
            "    case 3: return attrs36<4, 3>();\n"
            "    case 4: return attrs37<1, 3>();\n"
            "    case 5: return attrs37<4, 3>();\n"
            "    case 6: return attrs3<2, 2, 2, 5>();")

def _k2a36_hot_code(variant, hot=None):
    hk = (tuple(variant) + (0, 0))[3]
    if hk in (2, 3):
        ht, mode, w4 = K2A36_HOT_DEFAULT if hot is None else hot
        assert 0 <= ht < 1000 and 1 <= w4 < 100 and 0 <= mode < 4, hot
        return ht + 1000 * w4 + 100000 * mode
    if hk == 1:
        return (K2A33_HOT_T if hot is None else hot[0]) + 1000 * K2A33_HOT_W4
    return 1000 * K2A33_HOT_W4

def _k2a36_call(st, data, vid, hot=None, probe=K2A33_HINT, mask=7):
    x = data["hidden_states"].contiguous()
    if M32_WSK and RS1_ENABLE and _k2a_bucket_of(x.shape[0]) == 32 and (tuple(st["variants"][vid]) + (0, 0))[3] == 2:
        h0 = K2A36_HOT_DEFAULT if hot is None else hot
        hot, probe = (h0[0], h0[1], M32_WSK_W4), probe | (1 << 27)
    if (tuple(st["variants"][vid]) + (0, 0))[3] == 2 and x.shape[0] > 48:
        raise ValueError("k2a_v36: HK2 variants serve M <= 48")
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"],
                   st["inter"], st["y"], st["meta"], st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"],
                   st["xw"], st["xw2"], _k2a36_hot_code(st["variants"][vid], hot), m,
                   data["config"]["local_start"], st["grid"][vid], _k3p_g3(st, m, _su1_g3(st, m)), probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0) | _k3p_bits(m) | ((1 << 7) if RS1_ENABLE else 0))
    return out



def _v14_self_test_inputs(bs, seed, device, dist="normal"):
    d = _k2a36_synth_inputs(bs, seed, device, dist)
    if R1_LOCAL_ID_TABLE:
        ids = d["topk_ids"]
        d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
    return d


def _v14_k2a36_prepare(st, w, dev):
    st["r36"] = {}
    print(f"[v14] levers: K2A36_ENABLE={K2A36_ENABLE} K2A36_BUCKETS={K2A36_BUCKETS} K2A36_PLAN={K2A36_PLAN} "
          f"K2A33_PLAN={K2A33_PLAN} K2A33 variants={len(K2A33_VARIANTS)}", flush=True)
    if not K2A36_ENABLE:
        return
    t2 = time.time()
    for rep, (vid, hot) in sorted(K2A36_PLAN.items()):
        if vid >= len(st["variants"]) or st["grid"][vid] < st["sms"] * st["variants"][vid][2]:
            print(f"[k2a36] bucket {rep}: v{vid} not built or below its CTAs/SM target -> v3.3's plan", flush=True)
            continue
        good_all = True
        for bs, seed, dist in K2A36_SELF_TEST_CASES:
            if _k2a_bucket_of(bs) != rep:
                continue
            try:
                d = _v14_self_test_inputs(bs, seed, dev, dist)
                ref = _k2a_ref_fp32(d, w)
                data = {"hidden_states": d["hidden_states"], "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
                        "config": {"local_start": 0}}
                for k in range(2):
                    st["err"].zero_()
                    got = _k2a36_call(st, data, vid, hot)
                    torch.cuda.synchronize()
                    tot, row = _k2a_rel_err(got, ref)
                    errf = int(st["err"][0])
                    good = tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(got.float()).all())
                    print(f"[k2a36] layer self-test bs={bs} {dist} v{vid} hot={hot} call {k}: rel {tot:.3e} worst row "
                          f"{row:.3e} err={errf} -> {'ok' if good else 'FAIL'}", flush=True)
                    good_all &= good
            except Exception as ex:
                print(f"[k2a36] self-test bs={bs} v{vid} raised {ex!r}", flush=True)
                good_all = False
            _k2a33_reset(st)
            st["err"].zero_()
            if not good_all:
                break
        if good_all:
            st["r36"][rep] = (vid, tuple(hot))
        print(f"[k2a36] bucket {rep}: v{vid} hot {hot} -> {'route live' if good_all else 'FAILED: v3.3 plan'}", flush=True)
    print(f"[v14] K2A v3.6 routes live {st['r36']} (self-tests {time.time() - t2:.1f}s)", flush=True)



K2A37_ENABLE = False
K2A37_PLAN = {128: (5, (8, 0, 12), (8, 0, 3))}
K2A37_DQ = 120

def _k2a37_dyn_code(dyn):
    if dyn is None:
        return 0
    f1, f2, dsh, cold_only = (tuple(dyn) + (0,))[:4]
    assert 0 <= f1 < 64 and 0 <= f2 < 64 and 0 <= dsh <= 4, dyn
    return (f1 << 8) | (f2 << 14) | (dsh << 20) | ((1 << 23) if cold_only else 0)

KW1_K2_W4 = 5

def _k2a37_call(st, data, vid, hot=None, dyn=None, probe=K2A33_HINT, mask=7):
    raise RuntimeError("K2A v3.7 removed (VML1 serves bucket 128)")


def _v17_k2a37_prepare(st, w, dev):
    st["r37"] = {}
    print(f"[v17] levers: K2A37_ENABLE={K2A37_ENABLE} K2A37_PLAN={K2A37_PLAN} K2A36_PLAN={K2A36_PLAN} K2A33_PLAN={K2A33_PLAN} "
          f"K2A33 variants={len(K2A33_VARIANTS)}", flush=True)
    if not K2A37_ENABLE:
        return


V12_LT_ROUTE = {}
V12_ZBUF = True


def _v12_gemm(x, w, name, tokens):
    if V12_LT_ROUTE.get((name, _k2a_bucket_of(tokens))):
        b = _V12_ZERO_BIAS.get((w.shape[0], w.dtype, str(w.device)))
        if b is not None:
            return torch.addmm(b, x, w.t())
    return _gemm(x, w)


_V12_ZERO_BIAS = {}


def _v12_prepare(state):
    dense = state["dense"]
    dev = dense["qkv_gate"].device
    for w in (dense["qkv_gate"], dense["o"], dense["unpooler"]):
        _V12_ZERO_BIAS[(w.shape[0], w.dtype, str(w.device))] = torch.zeros((w.shape[0],), dtype=w.dtype, device=dev)
    state["v12_zbuf"] = torch.zeros((SLICED_UNPOOL_MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=dev) if V12_ZBUF else None
    gen = torch.Generator(device=dev).manual_seed(1212)
    try:
        for (name, rep), on in sorted(V12_LT_ROUTE.items(), key=str):
            if not on:
                continue
            w = {"qkv": dense["qkv_gate"], "o": dense["o"], "unpool": dense["unpooler"]}[name]
            x = torch.randn((rep, w.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            ref, got = _gemm(x, w), _v12_gemm(x, w, name, rep)
            torch.cuda.synchronize()
            err = float((got.float() - ref.float()).norm() / ref.float().norm())
            print(f"[v12] Lt self-test {name} T={rep}: rel {err:.1e}", flush=True)
            if not err < 1e-2:
                raise RuntimeError(f"Lt {name} T={rep}")
    except Exception as ex:
        print(f"[v12] Lt self-test FAILED -> _gemm everywhere: {ex!r}", flush=True)
        V12_LT_ROUTE.clear()
    return True


P1_ENABLE = True
P1_PROJ = {'qkv': True, 'o': True, 'pooler': True, 'unpooler': True}

P1_C = 0x70
P1_ESC_MAX = 32
P1_RAW_GROUP = 16
P1_MAXT = 8

P1_GEOMETRY = {
    ("qkv", 1): (16, 1, 1), ("qkv", 8): (4, 8, 1),
    ("o", 1): (8, 1, 1), ("o", 8): (8, 1, 1),
    ("pooler", 1): (16, 1, 1), ("pooler", 8): (16, 1, 1),
    ("unpooler", 1): (16, 1, 1), ("unpooler", 8): (16, 1, 1),
}
P1_ROUTE = {
    ("qkv", 1): True, ("qkv", 8): True,
    ("o", 1): True, ("o", 8): True,
    ("pooler", 1): True, ("pooler", 8): True,
    ("unpooler", 1): False, ("unpooler", 8): True,
}
P1_ROUTE_MIN = 1.03
P1_ROWS_T8 = {"unpooler": 1024}

P1_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void p1s_gemv(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mode, int64_t pdl,
              int64_t red_mode);
std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps);
void p1s_carveout(int64_t pct);
void p1_unpack(torch::Tensor pk, torch::Tensor toff, torch::Tensor base, torch::Tensor fmt, torch::Tensor eoff,
               torch::Tensor esc, torch::Tensor out, int64_t N, int64_t K, int64_t use_esc);
"""

P1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

#ifndef P1_LAYER_ONLY
#define P1_LAYER_ONLY 0
#endif

namespace p1 {

constexpr uint32_t KC4 = 0x38383838u;     // hi byte of every scaled bf16: exponent bits e4..e7 of 0x70
constexpr uint32_t CMASK = 0x87008700u;   // code bits per half: e1 e2 e3 (bits 8-10) and the sign (bit 15)
constexpr int SBP = 1536;                 // packed super-block bytes (16 rows x 64 k)
constexpr int SBR = 2048;                 // raw super-block bytes
constexpr int BIG = 0x7fffffff;
constexpr int MAXW = 16;                  // warps per CTA (<= 512 threads)

struct Args {
  const __nv_bfloat16* x; int M; int ldx;             // x [M, K] bf16 row-major, 1 <= M <= 8
  const uint8_t* pk; const int64_t* toff; const float* scale; const int* eoff; const int2* esc; const uint8_t* fmt;
  __nv_bfloat16* out; int ldo; int N; int nsb_total;  // out [M, ldo] bf16, columns [0, N)
  float* ws; int* cnt;                                 // split > 1: FP32 workspace [9][8][N] + counters [N/16]
  int red_mode;                                        // 0: deterministic slots [s][tok][N], summed q = 0..S-1 by the
  int trig;                                            // V20: 1 = griddepcontrol.launch_dependents at entry (one-wave grids)
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int r) { return __funnelshift_r(x, x, r); }

__device__ __forceinline__ void dec_kt(const uint4 (&p)[3], const int kt, uint32_t (&a)[4]) {
  const uint4 bc = p[kt >> 1];
  const uint32_t w0 = (kt & 1) ? bc.z : bc.x;
  const uint32_t w1 = (kt & 1) ? bc.w : bc.y;
  const uint32_t cw = kt == 0 ? p[2].x : kt == 1 ? p[2].y : kt == 2 ? p[2].z : p[2].w;
  a[0] = __byte_perm(w0, KC4, 0x4140) | (cw & CMASK);
  a[1] = __byte_perm(w0, KC4, 0x4342) | (rotr(cw, 4) & CMASK);
  a[2] = __byte_perm(w1, KC4, 0x4140) | (rotr(cw, 8) & CMASK);
  a[3] = __byte_perm(w1, KC4, 0x4342) | (rotr(cw, 12) & CMASK);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldw(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldx(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ int esc_key(uint32_t m, int sb_lo) { return ((int)(m & 0xFFFFu) - sb_lo) * 4 + (int)((m >> 23) & 3u); }
__device__ __forceinline__ void esc_patch4(uint32_t (&a)[4], int key, int sb_lo, int lane, uint32_t em, uint32_t ev,
                                           int& cur, int end, int& nkey) {
  while (cur < end) {
    const uint32_t m = __shfl_sync(0xffffffffu, em, cur);
    if (esc_key(m, sb_lo) != key) break;
    const uint32_t v = __shfl_sync(0xffffffffu, ev, cur);
    if (lane == (int)((m >> 16) & 31u)) {
      const int i = (int)((m >> 21) & 3u), h = (int)((m >> 25) & 1u);
      const uint32_t sc = v & 0xFFFFu;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (q == i) a[q] = h ? ((a[q] & 0x0000FFFFu) | (sc << 16)) : ((a[q] & 0xFFFF0000u) | sc);
    }
    ++cur;
  }
  const uint32_t mn = __shfl_sync(0xffffffffu, em, cur & 31);
  nkey = cur < end ? esc_key(mn, sb_lo) : BIG;
}

template <int D, int KM>
__device__ __forceinline__ void stream(const Args& A, int rt, int lane, int sb_lo, int nsb, float (&acc)[4]) {
  constexpr bool RAW = KM == 1;
  constexpr bool PROBE = KM == 3;
  constexpr int NCH = RAW ? 4 : 3;
  constexpr int SBB = RAW ? SBR : SBP;
  const int g = lane >> 2, t = lane & 3;
  const bool xon = g < A.M;
  const uint8_t* wp = A.pk + A.toff[rt] + (long long)sb_lo * SBB + lane * 16;
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(A.x + (size_t)(xon ? g : 0) * A.ldx + (size_t)sb_lo * 64 + 2 * t);
  uint4 w[D][NCH];
  uint32_t b[D][8];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < nsb) {
#pragma unroll
      for (int c = 0; c < NCH; ++c) w[d][c] = ldw(wp + d * SBB + c * 512);
    }
  uint32_t em = 0u, ev = 0u;
  int cur = 32, end = 0, nkey = BIG;
  if constexpr (KM == 0) {
    const int o0 = A.eoff[rt], o1 = A.eoff[rt + 1];
    const int idx = o0 + lane;
    int2 e = make_int2(0, 0);
    if (idx < o1) e = A.esc[idx];
    em = (uint32_t)e.x;
    ev = (uint32_t)e.y;
    const int esb = (int)(em & 0xFFFFu);
    const bool in = idx < o1 && esb >= sb_lo && esb < sb_lo + nsb;
    const uint32_t msk = __ballot_sync(0xffffffffu, in);
    cur = msk ? __ffs(msk) - 1 : 32;
    end = msk ? 32 - __clz(msk) : 0;
    const uint32_t mf = __shfl_sync(0xffffffffu, em, cur & 31);
    nkey = msk ? esc_key(mf, sb_lo) : BIG;
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < nsb) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t v = ldx(xw + d * 32 + 4 * j);
        b[d][j] = xon ? v : 0u;
      }
    }
  uint32_t px = 0u;
  for (int s0 = 0; s0 < nsb; s0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int s = s0 + d;
      if (s < nsb) {
        if constexpr (PROBE) {
#pragma unroll
          for (int c = 0; c < NCH; ++c) px ^= w[d][c].x ^ w[d][c].y ^ w[d][c].z ^ w[d][c].w;
#pragma unroll
          for (int j = 0; j < 8; ++j) px ^= b[d][j];
        } else {
#pragma unroll
          for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[4];
            if constexpr (RAW) {
              a[0] = w[d][kt].x; a[1] = w[d][kt].y; a[2] = w[d][kt].z; a[3] = w[d][kt].w;
            } else {
              dec_kt(w[d], kt, a);
              const int key = s * 4 + kt;
              if (key == nkey) esc_patch4(a, key, sb_lo, lane, em, ev, cur, end, nkey);
            }
            mma16816(acc, a, b[d][2 * kt], b[d][2 * kt + 1]);
          }
        }
        if (s + D < nsb) {
#pragma unroll
          for (int c = 0; c < NCH; ++c) w[d][c] = ldw(wp + (s + D) * SBB + c * 512);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t v = ldx(xw + (s + D) * 32 + 4 * j);
            b[d][j] = xon ? v : 0u;
          }
        }
      }
    }
  }
  if constexpr (PROBE) acc[0] = __uint_as_float(px & 0x007FFFFFu);   // finite, deterministic, keeps the loads live
}

// D = 1: <= 64 registers (2 x 512 or 4 x 256 threads per SM); D = 2: <= 128 (no spills; the sweep picks W / S so the
template <int D, int MODE> struct Bounds { static constexpr int minb = D == 1 ? 2 : 1; };

template <int D, int MODE>
__global__ void __launch_bounds__(512, Bounds<D, MODE>::minb) p1s_kernel(const __grid_constant__ Args A) {
  __shared__ float red[MAXW][128];      // [warp][row * 8 + token]
  __shared__ int s_last;
  if (A.trig) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // V20: early trigger (host: one-wave grids)
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = (int)(blockDim.x >> 5);
  const int rt = (int)blockIdx.x, S = (int)gridDim.y, sidx = (int)blockIdx.y;
  const int c0 = (int)((long long)sidx * A.nsb_total / S), c1 = (int)((long long)(sidx + 1) * A.nsb_total / S);
  const int sb_lo = c0 + (int)((long long)warp * (c1 - c0) / nw);
  const int sb_hi = c0 + (int)((long long)(warp + 1) * (c1 - c0) / nw);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (MODE == 0) {
    stream<D, 0>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  } else if constexpr (MODE == 1) {
    stream<D, 1>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  } else if constexpr (MODE == 3) {
    stream<D, 3>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  } else {
    if (A.fmt[rt]) stream<D, 1>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
    else stream<D, 0>(A, rt, lane, sb_lo, sb_hi - sb_lo, acc);
  }
  {
    const int g = lane >> 2, t = lane & 3;
    float* rw = red[warp];
    rw[g * 8 + 2 * t] = acc[0];
    rw[g * 8 + 2 * t + 1] = acc[1];
    rw[(g + 8) * 8 + 2 * t] = acc[2];
    rw[(g + 8) * 8 + 2 * t + 1] = acc[3];
  }
  __syncthreads();
  const int row = rt * 16 + (tid & 15), tok = tid >> 4;
  const bool mine = tid < 16 * A.M;
  if (mine) {
    float v = 0.f;
    for (int q = 0; q < nw; ++q) v += red[q][(tid & 15) * 8 + tok];
    v *= A.scale[rt];
    if (S == 1) A.out[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
    else if (A.red_mode == 0) __stcg(A.ws + ((size_t)sidx * 8 + tok) * A.N + row, v);   // this slice's slot
    else atomicAdd(A.ws + ((size_t)64 + tok) * A.N + row, v);   // control: order-dependent FP32 sum (slot 8)
  }
  if (S == 1) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(A.cnt + rt, 1) == S - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (mine) {
    float sum = 0.f;
    if (A.red_mode == 0) {
      for (int q = 0; q < S; ++q) sum += __ldcg(A.ws + ((size_t)q * 8 + tok) * A.N + row);
    } else {
      float* p = A.ws + ((size_t)64 + tok) * A.N + row;
      sum = __ldcg(p);
      *p = 0.f;                                                 // the atomic slot must start at zero next call
    }
    A.out[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(sum);
  }
  if (tid == 0) A.cnt[rt] = 0;
}

__global__ void p1_unpack_kernel(const uint8_t* __restrict__ pk, const int64_t* __restrict__ toff,
                                 const int* __restrict__ base, const uint8_t* __restrict__ fmt,
                                 const int* __restrict__ eoff, const int2* __restrict__ esc,
                                 uint16_t* __restrict__ out, int K, int use_esc) {
  const int sb = (int)blockIdx.x, rt = (int)blockIdx.y, lane = (int)threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  uint32_t v[16];
  if (fmt[rt]) {
    const uint8_t* p = pk + toff[rt] + (long long)sb * SBR + lane * 16;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 q = ldw(p + c * 512);
      v[4 * c] = q.x; v[4 * c + 1] = q.y; v[4 * c + 2] = q.z; v[4 * c + 3] = q.w;
    }
  } else {
    const uint8_t* p = pk + toff[rt] + (long long)sb * SBP + lane * 16;
    uint4 q[3] = {ldw(p), ldw(p + 512), ldw(p + 1024)};
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      uint32_t t4[4];
      dec_kt(q, kt, t4);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t wv = t4[i], o = 0u;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t x = (wv >> (16 * h)) & 0xFFFFu;
          const uint32_t n = (x >> 7) & 15u;
          const uint32_t y = (x & 0x807Fu) | (((uint32_t)(base[rt] + (int)n) & 0xFFu) << 7);
          o |= y << (16 * h);
        }
        v[4 * kt + i] = o;
      }
    }
    if (use_esc) {
      for (int j = eoff[rt]; j < eoff[rt + 1]; ++j) {
        const int2 e = esc[j];
        const uint32_t m = (uint32_t)e.x;
        if ((int)(m & 0xFFFFu) != sb || (int)((m >> 16) & 31u) != lane) continue;
        const int o = (int)((m >> 21) & 15u), h = (int)((m >> 25) & 1u);
        const uint32_t rw = ((uint32_t)e.y >> 16) & 0xFFFFu;
#pragma unroll
        for (int q2 = 0; q2 < 16; ++q2)
          if (q2 == o) v[q2] = h ? ((v[q2] & 0xFFFFu) | (rw << 16)) : ((v[q2] & 0xFFFF0000u) | rw);
      }
    }
  }
#pragma unroll
  for (int o = 0; o < 16; ++o) {
    const int kt = o >> 2, i = o & 3;
    const int row = rt * 16 + g + 8 * (i & 1);
    const int k = sb * 64 + 16 * kt + 2 * t + 8 * (i >> 1);
    *reinterpret_cast<uint32_t*>(out + (size_t)row * K + k) = v[o];   // (k, k+1) = (lo, hi)
  }
}

template <int D, int MODE>
static void launch(const Args& A, int warps, int S, int nrt, int pdl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)nrt, (unsigned)S, 1);
  cfg.blockDim = dim3((unsigned)(warps * 32), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, p1s_kernel<D, MODE>, A));
}

template <int D, int MODE>
static std::vector<int64_t> info_t(int warps) {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, p1s_kernel<D, MODE>));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p1s_kernel<D, MODE>, warps * 32, 0));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ};
}

}  // namespace p1

#define P1S_DISPATCH(D_, M_, CALL) if (depth == D_ && mode == M_) { CALL; }

void p1s_gemv(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mode, int64_t pdl,
              int64_t red_mode) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "p1s: x bf16 contiguous");
  TORCH_CHECK(M >= 1 && M <= 8 && x.size(0) == M && x.size(1) == K, "p1s: 1 <= M <= 8 and x [M, K]");
  TORCH_CHECK(K % 64 == 0 && N % 16 == 0 && N >= 16, "p1s: K % 64, N % 16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "p1s: x align");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt && fmt.scalar_type() == at::kByte,
              "p1s: packed dtypes");
  const int nrt = (int)(N / 16), nsb = (int)(K / 64);
  TORCH_CHECK(toff.numel() >= nrt && scale.numel() >= nrt && eoff.numel() >= nrt + 1 && fmt.numel() >= nrt, "p1s: meta");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1 && out.stride(0) == ldo && ldo >= N &&
              out.size(0) >= M && out.size(1) >= N, "p1s: out bf16 [M, >= N] with row stride ldo");
  TORCH_CHECK((warps == 4 || warps == 8 || warps == 16) && depth == 1 && mode >= 0 && mode <= 3,
              "p1s: warps in {4, 8, 16}, depth 1 (the round-2b winners), mode in 0..3");
  TORCH_CHECK(split >= 1 && split <= 8 && split * warps <= nsb,
              "p1s: 1 <= split <= 8 and every warp needs >= 1 super-block (split * warps <= K / 64)");
  TORCH_CHECK(red_mode == 0 || red_mode == 1, "p1s: red_mode 0 (deterministic) or 1 (atomic control)");
  if (split > 1)
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= 72 * N && cnt.scalar_type() == at::kInt &&
                cnt.numel() >= nrt, "p1s: split-K workspace [9 * 8 * N] fp32 (slot 8 zeroed) + counters [N / 16] int32 (zeroed)");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1::Args A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); A.ldo = (int)ldo; A.N = (int)N; A.nsb_total = nsb;
  A.ws = split > 1 ? ws.data_ptr<float>() : nullptr; A.cnt = split > 1 ? cnt.data_ptr<int>() : nullptr;
  A.red_mode = (int)red_mode;
  A.trig = (int)((pdl >> 1) & 1);   // V20: pdl bit 1 = early launch_dependents
  const int w = (int)warps, S = (int)split, pd = (int)(pdl & 1);
  P1S_DISPATCH(1, 0, (p1::launch<1, 0>(A, w, S, nrt, pd)); return)
#if !P1_LAYER_ONLY
  P1S_DISPATCH(1, 1, (p1::launch<1, 1>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(1, 2, (p1::launch<1, 2>(A, w, S, nrt, pd)); return)
  P1S_DISPATCH(1, 3, (p1::launch<1, 3>(A, w, S, nrt, pd)); return)
#endif
  TORCH_CHECK(false, "p1s: no instantiation");
}

std::vector<int64_t> p1s_info(int64_t depth, int64_t mode, int64_t warps) {
  const int w = (int)warps;
  P1S_DISPATCH(1, 0, return (p1::info_t<1, 0>(w)))
#if !P1_LAYER_ONLY
  P1S_DISPATCH(1, 1, return (p1::info_t<1, 1>(w)))
  P1S_DISPATCH(1, 2, return (p1::info_t<1, 2>(w)))
  P1S_DISPATCH(1, 3, return (p1::info_t<1, 3>(w)))
#endif
  return {0, 0, 0};
}

void p1s_carveout(int64_t pct) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(p1::p1s_kernel<1, 0>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)pct));
}

void p1_unpack(torch::Tensor pk, torch::Tensor toff, torch::Tensor base, torch::Tensor fmt, torch::Tensor eoff,
               torch::Tensor esc, torch::Tensor out, int64_t N, int64_t K, int64_t use_esc) {
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == N && out.size(1) == K, "unpack: out");
  TORCH_CHECK(K % 64 == 0 && N % 16 == 0, "unpack: shape");
  const c10::cuda::OptionalCUDAGuard guard(out.device());
  const dim3 grid((unsigned)(K / 64), (unsigned)(N / 16));
  p1::p1_unpack_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      pk.data_ptr<uint8_t>(), toff.data_ptr<int64_t>(), base.data_ptr<int>(), fmt.data_ptr<uint8_t>(),
      eoff.data_ptr<int>(), reinterpret_cast<const int2*>(esc.data_ptr<int>()),
      reinterpret_cast<uint16_t*>(out.data_ptr()), (int)K, (int)use_esc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_P1_MOD = None


def p1_build(verbose=False, layer_only=False):
    global _P1_MOD
    _pt1_wait("p1")
    if _P1_MOD is not None:
        return _P1_MOD
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a",
             f"-DP1_LAYER_ONLY={1 if layer_only else 0}"]
    cpp, cuda, fns = P1_CPP_SRC, P1_CUDA_SRC, ["p1s_gemv", "p1s_info", "p1_unpack", "p1s_carveout"]
    if V20_P1M_ENABLE and V20_P1M_ROUTE:
        cpp, cuda, fns = cpp + P1M_CPP_ADD, cuda + p1m_cuda_src(_v20_inst(), ()), fns + ["p1m_gemm", "p1m_info"]
    tag = hashlib.sha1((cpp + cuda + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    _P1_MOD = load_inline(
        name=f"p1s_mod_{tag}", cpp_sources=cpp, cuda_sources=cuda,
        functions=fns, extra_cuda_cflags=flags, verbose=verbose)
    print(f"[p1] compiled in {time.time() - t0:.1f}s", flush=True)
    return _P1_MOD


def p1_lane_maps(device):
    L = torch.arange(32, device=device).view(32, 1, 1)
    o = torch.arange(16, device=device).view(1, 16, 1)
    h = torch.arange(2, device=device).view(1, 1, 2)
    g, t, kt, i = L // 4, L % 4, o // 4, o % 4
    row = (g + 8 * (i % 2) + 0 * h).expand(32, 16, 2)
    kk = (16 * kt + 2 * t + 8 * (i // 2) + h).expand(32, 16, 2)
    return row, kk


def p1_esc_columns(esc):
    e = esc[:, 0].long()
    sb, L, o, h = e & 0xFFFF, (e >> 16) & 31, (e >> 21) & 15, (e >> 25) & 1
    return 64 * sb + 16 * (o >> 2) + 2 * (L & 3) + 8 * ((o & 3) >> 1) + h


def _wrap32(v64):
    return torch.where(v64 >= 2 ** 31, v64 - 2 ** 32, v64).to(torch.int32)


def p1_pack(w, esc_max=P1_ESC_MAX, raw_group=P1_RAW_GROUP, name="w"):
    t0 = time.time()
    assert w.dtype == torch.bfloat16 and w.is_cuda and w.dim() == 2
    w = w.contiguous()
    dev = w.device
    N, K = w.shape
    assert N % 16 == 0 and K % 64 == 0
    nrt, nsb = N // 16, K // 64
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    hist = torch.zeros((nrt, 256), dtype=torch.int32, device=dev)
    hist.scatter_add_(1, e.view(nrt, 16 * K).long(), torch.ones((nrt, 16 * K), dtype=torch.int32, device=dev))
    cs = torch.cat([torch.zeros((nrt, 1), dtype=torch.int64, device=dev), hist.long().cumsum(1)], 1)
    win = cs[:, 17:256] - cs[:, 1:240]
    base = (win.argmax(1) + 1).to(torch.int32)
    row, kk = p1_lane_maps(dev)
    fr = bits.view(nrt, 16, nsb, 64)[:, row, :, kk].permute(3, 4, 0, 1, 2).contiguous()
    del bits, e
    s = (fr >> 15) & 1
    ee = (fr >> 7) & 0xFF
    m = fr & 0x7F
    n = ee - base.view(nrt, 1, 1, 1, 1)
    inw = (n >= 0) & (n <= 15)
    nn = torch.where(inw, n, torch.zeros_like(n))
    sv = torch.where(inw, s, torch.zeros_like(s))
    byte = torch.where(inw, ((nn & 1) << 7) | m, torch.zeros_like(m))
    bp = byte.view(nrt, nsb, 32, 2, 8, 2).permute(0, 1, 3, 2, 4, 5).reshape(nrt, nsb, 2, 32, 16).to(torch.uint8)
    ii = torch.arange(4, device=dev).view(4, 1)
    hh = torch.arange(2, device=dev).view(1, 2)
    cw = torch.zeros((nrt, nsb, 32, 4), dtype=torch.int64, device=dev)
    for f, tgt in (((nn >> 1) & 1, 8), ((nn >> 2) & 1, 9), ((nn >> 3) & 1, 10), (sv, 15)):
        pos = (tgt + 16 * hh + 4 * ii) % 32
        cw += (f.view(nrt, nsb, 32, 4, 4, 2).to(torch.int64) << pos).sum(dim=(-1, -2))
    cw8 = _wrap32(cw).contiguous().view(torch.uint8).view(nrt, nsb, 1, 32, 16)
    packed_t = torch.cat([bp, cw8], dim=2).reshape(nrt, nsb * 1536)
    del byte, bp, cw, cw8, nn, sv, s, m
    fr16 = torch.where(fr >= 32768, fr - 65536, fr).to(torch.int16)
    raw_t = fr16.view(nrt, nsb, 32, 4, 4, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(torch.uint8).reshape(nrt, nsb * 2048)
    idx = (~inw).nonzero()
    if idx.shape[0]:
        key = (idx[:, 0] * nsb + idx[:, 1]) * 4 + (idx[:, 3] >> 2)
        idx = idx[torch.argsort(key, stable=True)]
    rt_i, sb_i, L_i, o_i, h_i = idx.unbind(1)
    raw16 = fr[rt_i, sb_i, L_i, o_i, h_i]
    rv = torch.where(raw16 >= 32768, raw16 - 65536, raw16).to(torch.int16).view(torch.bfloat16).double()
    j = (P1_C - base[rt_i]).double()
    sd = rv * torch.pow(torch.full_like(j, 2.0), j)
    sb16 = sd.to(torch.bfloat16)
    nan = torch.isnan(rv)
    exact = nan | (sb16.double() == sd)
    sc_bits = sb16.view(torch.int16).to(torch.int32) & 0xFFFF
    sc_bits = torch.where(nan, torch.full_like(sc_bits, 0x7FC0), sc_bits)
    cnt = torch.bincount(rt_i, minlength=nrt)
    bad = (torch.bincount(rt_i[~exact], minlength=nrt) > 0 if bool((~exact).any())
           else torch.zeros(nrt, dtype=torch.bool, device=dev))
    need_raw = (cnt > esc_max) | bad
    ng = (nrt + raw_group - 1) // raw_group
    pad = torch.zeros(ng * raw_group, dtype=torch.bool, device=dev)
    pad[:nrt] = need_raw
    raw_tile = pad.view(ng, raw_group).any(1, keepdim=True).expand(ng, raw_group).reshape(-1)[:nrt]
    keep = ~raw_tile[rt_i]
    meta = (sb_i | (L_i << 16) | (o_i << 21) | (h_i << 25))[keep]
    vals = (sc_bits.long() | (raw16.long() << 16))[keep]
    esc = torch.stack([_wrap32(meta.long()), _wrap32(vals)], 1).contiguous()
    if esc.numel() == 0:
        esc = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    cnt_k = torch.bincount(rt_i[keep], minlength=nrt)
    eoff = torch.zeros(nrt + 1, dtype=torch.int32, device=dev)
    eoff[1:] = cnt_k.cumsum(0).to(torch.int32)
    base_f = torch.where(raw_tile, torch.full_like(base, P1_C), base)
    scale = torch.pow(2.0, (base_f - P1_C).double()).float()
    fmt = raw_tile.to(torch.uint8)
    sizes = torch.where(raw_tile, torch.full_like(base, nsb * 2048), torch.full_like(base, nsb * 1536)).long()
    toff = torch.zeros(nrt, dtype=torch.int64, device=dev)
    toff[1:] = sizes.cumsum(0)[:-1]
    total = int(sizes.sum())
    buf = torch.empty(total + 64, dtype=torch.uint8, device=dev)
    toff_c, raw_c = toff.cpu().tolist(), raw_tile.cpu().tolist()
    for r in range(nrt):
        src = raw_t[r] if raw_c[r] else packed_t[r]
        buf[toff_c[r]:toff_c[r] + src.numel()] = src
    return {"name": name, "N": N, "K": K, "nrt": nrt, "nsb": nsb, "buf": buf, "toff": toff, "scale": scale, "fmt": fmt,
            "fmt_cpu": raw_c, "ws": torch.zeros(72 * N, dtype=torch.float32, device=dev),
            "cnt": torch.zeros(nrt, dtype=torch.int32, device=dev),
            "eoff": eoff, "esc": esc, "base": base_f.contiguous(), "bytes": total + nrt * 17 + 8 * int(keep.sum()),
            "n_esc": int(keep.sum()), "n_esc_all": int(idx.shape[0]), "n_raw_tiles": int(raw_tile.sum()),
            "max_esc_tile": int(cnt.max()) if nrt else 0, "unscalable": int((~exact).sum()),
            "pack_s": time.time() - t0}


def p1_describe(P):
    nw = P["N"] * P["K"]
    return (f"{P['name']} [{P['N']}x{P['K']}]: {P['bytes'] / 1e6:.2f} MB ({8 * P['bytes'] / nw:.3f} bits/w, "
            f"{P['bytes'] / (2 * nw):.4f} of BF16); escapes {P['n_esc_all']} ({P['n_esc_all'] / nw:.2e}/w, max/tile "
            f"{P['max_esc_tile']}, unscalable {P['unscalable']}); RAW tiles {P['n_raw_tiles']}/{P['nrt']}; "
            f"packed in {P['pack_s']:.2f}s")


def p1_unpack(mod, P, use_esc=True):
    out = torch.empty((P["N"], P["K"]), dtype=torch.bfloat16, device=P["buf"].device)
    mod.p1_unpack(P["buf"], P["toff"], P["base"], P["fmt"], P["eoff"], P["esc"], out, P["N"], P["K"], 1 if use_esc else 0)
    return out


def p1_mode_of(P, n_rows):
    f = P["fmt_cpu"][: n_rows // 16]
    return 1 if all(f) else (0 if not any(f) else 2)


def p1_call(mod, P, x, out, n_rows, ldo, geo, pdl=0, red_mode=0, mode=None):
    warps, split, depth = geo
    if mode is None:
        mode = p1_mode_of(P, n_rows)
    mod.p1s_gemv(x, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], out, P["ws"], P["cnt"],
                 x.shape[0], n_rows, P["K"], ldo, warps, split, depth, mode, pdl, red_mode)
    return out


def p1_verify(mod, P, w):
    u = p1_unpack(mod, P)
    mism = int((u.view(torch.int16) != w.view(torch.int16)).sum())
    neg = int((p1_unpack(mod, P, use_esc=False).view(torch.int16) != w.view(torch.int16)).sum())
    return mism, neg


def p1_bucket(T):
    return 1 if T == 1 else 8


def p1_prepare(weights, geometry=None, route=None, build=True, layer_only=True):
    try:
        mod = p1_build(layer_only=layer_only) if build else _P1_MOD
    except Exception as ex:
        print(f"[p1] build FAILED -> today's path everywhere: {ex!r}", flush=True)
        return None
    st = {"mod": mod, "geo": dict(geometry or P1_GEOMETRY), "route": dict(route or P1_ROUTE), "proj": {}}
    for name, w in weights.items():
        try:
            P = p1_pack(w, name=name)
            mism, neg = p1_verify(mod, P, w)
            torch.cuda.synchronize()
            ok = mism == 0 and (neg > 0 or P["n_esc"] == 0)
            print(f"[p1] {p1_describe(P)} | unpack mismatches {mism}, without patches {neg} -> "
                  f"{'ok' if ok else 'DROPPED'}", flush=True)
            if ok:
                for b in (1, 8):
                    geo = st["geo"].get((name, b))
                    n_rows = P1_ROWS_T8.get(name, P["N"])
                    if geo is not None and not (geo[1] * geo[0] <= P["nsb"] and 1 <= geo[1] <= 8):
                        st["route"][(name, b)] = False
                P["rows_t8"] = P1_ROWS_T8.get(name, P["N"])
                P["mode_t8"] = p1_mode_of(P, P["rows_t8"])
                st["proj"][name] = P
        except Exception as ex:
            print(f"[p1] {name}: pack/verify FAILED -> today's path: {ex!r}", flush=True)
    return st


def p1_route(st, proj, T):
    if st is None or T < 1 or T > P1_MAXT or proj not in st["proj"]:
        return None
    b = p1_bucket(T)
    if not st["route"].get((proj, b)):
        return None
    return st["geo"].get((proj, b))


def p1_gemv(st, proj, x, out=None, n_rows=None, ldo=None, geo=None):
    P = st["proj"][proj]
    T = x.shape[0]
    if n_rows is None:
        n_rows = P["rows_t8"]
    if out is None:
        out = torch.empty((T, n_rows), dtype=torch.bfloat16, device=x.device)
    if ldo is None:
        ldo = out.stride(0)
    if geo is None:
        geo = st["geo"][(proj, p1_bucket(T))]
    mode = P["mode_t8"] if n_rows == P["rows_t8"] else p1_mode_of(P, n_rows)
    p1_call(st["mod"], P, x, out, n_rows, ldo, geo, pdl=_g2_p1_pdl(proj) | _v20_p1_trig(st, proj, geo, mode), mode=mode)
    return out


def _v13_p1_ok(x, k):
    return x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == k and x.is_contiguous() and x.data_ptr() % 16 == 0


def _v13_p1(state, proj, x, out=None, **kw):
    st = state.get("p1")
    if not P1_ENABLE or st is None or not P1_PROJ.get(proj):
        return None
    if p1_route(st, proj, x.shape[0]) is None or not _v13_p1_ok(x, st["proj"][proj]["K"]):
        return None
    try:
        return p1_gemv(st, proj, x, out, **kw)
    except Exception as ex:
        _k3_disable(state, "p1", None, "P1", ex)
        return None


def _v13_pooler(ffn_input, dense, state):
    y = _vfr_marked_pooler(state, ffn_input) if VF_GDM else None
    if y is None:
        y = _gd(state, "pooler", ffn_input)
    if y is None:
        y = _v13_p1(state, "pooler", ffn_input)
    return y if y is not None else _v8_gemm(ffn_input, "pooler", dense["pooler"], state)


def _v13_unpool(expert_output, state, tokens):
    zb = state.get("v12_zbuf")
    if not (V12_ZBUF and zb is not None and R1_UNPOOL_INTO_ZEROS and tokens <= SLICED_UNPOOL_MAX_TOKENS
            and expert_output.dtype == zb.dtype):
        return None
    ffn_output = zb[:tokens]
    y = _v13_p1(state, "unpooler", expert_output, ffn_output, n_rows=UNPOOL_SLICE, ldo=HIDDEN)
    return ffn_output if y is not None else None


def _v13_today(proj, x, state, dense):
    tokens = x.shape[0]
    if proj == "qkv":
        return (_v8_gemm(x, "qkv_gate", dense["qkv_gate"], state) if tokens == 1
                else _v12_gemm(x, dense["qkv_gate"], "qkv", tokens))
    if proj == "o":
        return _v12_gemm(x, dense["o"], "o", tokens)
    if proj == "pooler":
        return _v8_gemm(x, "pooler", dense["pooler"], state)
    return torch.mm(x, dense["unpooler"][:UNPOOL_SLICE].t())


def _v13_p1_selftest(st, state, proj, T, gen):
    dense = state["dense"]
    P = st["proj"][proj]
    w = dense["qkv_gate" if proj == "qkv" else proj]
    n_rows, K, dev = P["rows_t8"], P["K"], w.device
    wr = w[:n_rows]
    fin = torch.isfinite(wr.float()).all(1)
    esc = sorted(set(p1_esc_columns(P["esc"]).tolist())) if P["n_esc"] else []
    esc = esc[::max(1, len(esc) // 48)][:48]
    rnd = torch.randint(0, K, (64,), generator=gen, device=dev).tolist()
    cols = esc + rnd[:16]
    cols += rnd[16:16 + (-len(cols)) % 8]
    chunks = [cols[i:i + T] for i in range(0, len(cols), T)] if T > 1 else [[c] for c in cols[:6] + cols[-2:]]
    bad = flushed = 0
    for cc in chunks:
        x = torch.zeros((T, K), dtype=torch.bfloat16, device=dev)
        ci = torch.tensor(cc, device=dev)
        x[torch.arange(T, device=dev), ci] = 1.0
        got = p1_gemv(st, proj, x).float()
        ref = wr[:, ci].t().float()
        mis = (got != ref) & ~(torch.isnan(got) & torch.isnan(ref)) & fin[None, :]
        sub = mis & (got == 0) & (ref.abs() < 2.0 ** -126)
        flushed += int(sub.sum())
        bad += int((mis & ~sub).sum())
    x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
    got = p1_gemv(st, proj, x)
    again = p1_gemv(st, proj, x)
    ref64 = x.double() @ wr.double().t()
    today = _v13_today(proj, x, state, dense)
    torch.cuda.synchronize()
    e64 = float((got.double() - ref64).norm() / ref64.norm())
    etoday = float((got.float() - today.float()).norm() / today.float().norm())
    same = bool(torch.equal(got, again))
    ok = bad == 0 and e64 < 4e-3 and etoday < 1e-2 and same
    return ok, (f"{len(cols)} one-hot cols ({len(esc)} escape) bad {bad} (subnormal flush {flushed}); random x rel vs fp64 "
                f"{e64:.2e} vs today {etoday:.2e}; repeat identical {same}")


def _v13_prepare(state):
    state["p1"] = None
    print(f"[v13] levers: P1_ENABLE={P1_ENABLE} P1_PROJ={P1_PROJ} K1 route={K1_ROUTE_MIN_BATCH} K2A33_PLAN={K2A33_PLAN}",
          flush=True)
    if not P1_ENABLE:
        return None
    t0 = time.time()
    dense = state["dense"]
    try:
        st = p1_prepare({"qkv": dense["qkv_gate"], "o": dense["o"], "pooler": dense["pooler"], "unpooler": dense["unpooler"]})
        if st is None:
            return None
        gen = torch.Generator(device=dense["qkv_gate"].device).manual_seed(1313)
        for proj in sorted(st["proj"]):
            P = st["proj"][proj]
            if P["mode_t8"] != 0:
                for b in (1, 8):
                    st["route"][(proj, b)] = False
                print(f"[p1] {proj}: RAW row tiles in rows [0, {P['rows_t8']}) (mode {P['mode_t8']}), the layer build has "
                      f"mode 0 only -> today's path", flush=True)
                continue
            for T in (1, 8):
                b = p1_bucket(T)
                if not st["route"].get((proj, b)) or p1_route(st, proj, T) is None:
                    continue
                ok, msg = _v13_p1_selftest(st, state, proj, T, gen)
                print(f"[p1] self-test {proj} T={T}: {msg} -> {'ok' if ok else 'FAIL: route off'}", flush=True)
                if not ok:
                    st["route"][(proj, b)] = False
    except Exception as ex:
        print(f"[v13] P1 prepare / self-test FAILED -> today's path everywhere: {ex!r}", flush=True)
        return None
    routed = sorted(f"{p}/{b}" for (p, b), on in st["route"].items() if on and p in st["proj"])
    print(f"[v13] P1 routes live {routed} (prepare {time.time() - t0:.1f}s)", flush=True)
    state["p1"] = st if routed else None
    return state["p1"]



_GEMV = None


def _gemm(x, w):
    if _GEMV is not None and x.shape[0] == 1 and _GEMV[0](1, w.shape[0], w.shape[1]):
        return _GEMV[1](x.view(1, -1), w)
    return _bf16_gemm_dispatch_impl(x, w, None)


def _local_id_table(device):
    s = torch.arange(N_EXPERTS, dtype=torch.int64, device=device).view(-1, 1)
    g = torch.arange(N_EXPERTS, dtype=torch.int64, device=device).view(1, -1)
    rel = (g - s) % N_EXPERTS
    return torch.where(rel < LOCAL_EXPERTS, rel, torch.full_like(rel, -1)).to(torch.int32).reshape(-1).contiguous()


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _r1_prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
        "local_id_table": _local_id_table(device),
        "stage_loc": torch.arange(1, STAGE_LOC_MAX + 1, dtype=torch.int64, device=device),
    }
    if R1_HOPPER_GEMV_T1:
        global _GEMV
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv, use_hopper_bf16_gemv

        _GEMV = (use_hopper_bf16_gemv, hopper_bf16_gemv)
    del freqs
    torch.cuda.synchronize()
    return state


PT1_PREFETCH = True
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


def _pt1_start():
    if not PT1_PREFETCH:
        return
    jobs = [("v10", _v10_build_all)] if V10_COMBINED_BUILD else []
    if P1_ENABLE:
        jobs.append(("p1", lambda: p1_build(layer_only=True)))
    if G2_ENABLE:
        jobs.append(("g2", _g2_build))
    if KS1_ENABLE:
        jobs.append(("ks1", _ks1_build))
    if SU1_ENABLE:
        jobs.append(("su1", _su1_build))
    if SBB1_ENABLE:
        jobs.append(("sbb1", _sbb1_build))
    if VG_ENABLE and (VGF_ENABLE or any(r[0] == "vgb" for r in list(VG_BND_ROUTE.values()) + [x for v in VG_BND_CAND.values() for x in v])):
        jobs.append(("vg", _vg_build))
    if GD_BUILD:
        jobs.append(("gd", _gd_build))
    if VML_ENABLE:
        jobs.append(("vml", _vml_build_job))
    if VMS_ENABLE:
        jobs.append(("vms", _vms_build))
    if VFA_ENABLE:
        jobs.append(("vfa", _vfa_build))
    if VD_ENABLE and VD_ROUTE:
        jobs.append(("vd", _vd_build))
    if LT_ENABLE and LT_PIN:
        jobs.append(("lt", _lt_build))
    for name, fn in jobs:
        th = threading.Thread(target=_pt1_target(name, fn), name=f"pt1-{name}", daemon=True)
        _PT1["threads"][name] = th
        th.start()
    print(f"[pt1] prefetching builds {[n for n, _ in jobs]}", flush=True)


def _pt1_wait(name):
    th = _PT1["threads"].get(name)
    if th is not None and th is not threading.current_thread():
        th.join()


def _pt1_join_all():
    for th in list(_PT1["threads"].values()):
        th.join()
    if _PT1["threads"]:
        secs = " ".join(f"{k}={v:.0f}s" for k, v in _PT1["secs"].items())
        print(f"[pt1] prefetch builds done: {secs}; errors {_PT1['errors'] or 'none'}", flush=True)


def prepare(weights):
    t0 = time.time()
    _pt1_start()
    state = _r1_prepare(weights)
    device = weights["qkv_gate"].device
    state["k1_ok"] = False
    state["k2a"] = None
    if V10_COMBINED_BUILD:
        global _K1_MOD
        mod = _v10_build_all()
        if USE_K1 and _K1_MOD is None:
            if mod is not None:
                _K1_MOD = mod
            else:
                try:
                    _K1_MOD = _k1_build()
                except Exception as ex:
                    print(f"[k3] K1 CUDA build FAILED: attention stays on the r1 FA3 path: {ex!r}", flush=True)
    if USE_K1:
        if _K1_MOD is None:
            print("[k3] K1 not built: attention stays on the r1 FA3 path", flush=True)
        else:
            state["k1_ok"] = _k1_prepare(device)
    state["k2a3"] = None
    state["k2a31"] = None
    state["k2a33"] = None
    if USE_K2A:
        state["k2a33"] = _v11_k2a33_prepare(weights, t0)
        k33ok = state["k2a33"]["ok"] if state["k2a33"] is not None else {}
        need_v31 = [rep for rep in V9_K2A31_ROUTE if not (V11_K2A33_ROUTE.get(rep) == "v33" and k33ok.get(rep))]
        state["k2a31"] = _v9_k2a31_prepare(weights, t0) if (need_v31 or V11_BUILD_V31_ALWAYS) else None
        k31ok = dict(state["k2a31"]["ok"]) if state["k2a31"] is not None else {}
        for rep, ok in k33ok.items():
            if ok and V11_K2A33_ROUTE.get(rep) == "v33":
                k31ok[rep] = True
        need_v3 = [rep for rep in V8_K2A3_ROUTE if not (V9_K2A31_ROUTE.get(rep) == "v31" and k31ok.get(rep))]
        state["k2a3"] = _v8_k2a3_prepare(weights, t0) if (need_v3 or V9_BUILD_V3_ALWAYS) else None
        k3ok = dict(state["k2a3"]["ok"]) if state["k2a3"] is not None else {}
        for rep, ok in k31ok.items():
            if ok and V9_K2A31_ROUTE.get(rep) == "v31":
                k3ok[rep] = True
        need_v2 = [rep for rep, r in K2A_ROUTE.items() if r == "v2" and not (V8_K2A3_ROUTE.get(rep) == "v3" and k3ok.get(rep))]
        state["k2a"] = _k2a_prepare(weights, t0) if (need_v2 or V8_BUILD_V22_ALWAYS) else None
        print(f"[v8] experts: v3 buckets {sorted(r for r, o in k3ok.items() if o and V8_K2A3_ROUTE.get(r) == 'v3')},"
              f" v2.2 built for {need_v2}", flush=True)
    state["r1b"] = _v8_prepare(weights)
    state["v8"] = _v8_tail_prepare(weights, state)
    state["v9bnd"] = _v9_bnd_prepare(state)
    state["v10qk"] = _v10_qk_prepare(state)
    _v12_prepare(state)
    _v13_prepare(state)
    _v20_prepare(state)
    _v18_fa4_prepare(state, t0)
    _g2_prepare(state, t0)
    _ks1_prepare(state, t0)
    _dg1_prepare(state, t0)
    _lt_prepare(state)
    _vd_prepare(state, t0)
    _sbb1_prepare(state, t0)
    _vf_prepare(state, t0)
    _vg_prepare(state, t0)
    _su1_prepare(state, t0)
    _gd_prepare(state, t0)
    _gd_cat_prepare(state, t0)
    _vfr_prepare(state)
    _va1_prepare(state)
    _vfa_prepare(state, t0)
    torch.cuda.synchronize()
    _pt1_join_all()
    occ = "/".join(str(v) for _, v in sorted(_K1_OCC.items(), key=lambda kv: str(kv[0]))) or "-"
    builds = " ".join(f"{k}={v:.0f}s" for k, v in _K3_BUILD_S.items()) or "none"
    print(f"[k3] attention={'K1' if state['k1_ok'] else 'r1-FA3'} experts={'K2A v3.3' if state.get('k2a33') is not None else 'K2A v3.1' if state.get('k2a31') is not None else 'K2A v3' if state.get('k2a3') is not None else ('K2A' if state['k2a'] is not None else 'r1-Marlin')}"
          f" (USE_K1={USE_K1} USE_K2A={USE_K2A}); nvcc {builds}; k1 v7 occupancy {occ} CTAs/SM,"
          f" route {K1_ROUTE_MIN_BATCH}, cps {K1_CPS_T}, fold counters {'ok' if _K1_CNT else 'none'};"
          f" k2a v2.2 route {K2A_ROUTE} plan {K2A_V2_PLAN};"
          f" prepare took {time.time() - t0:.1f}s", flush=True)
    return state


def _post_expert_norm_weighted_sum(y, topk_weights, norm_w, slot_ids):
    if not (R1_POSTNORM_TRITON_ALL_T and _POSTNORM_KERNEL is not None):
        return fused_post_expert_norm_weighted_sum(
            y, topk_weights, norm_weight=norm_w, eps=POST_NORM_EPS, weight_first=True,
            slot_expert_ids=slot_ids)
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = triton.next_power_of_2(hidden)
    _POSTNORM_KERNEL[(tokens,)](
        y, norm_w, topk_weights, out, slot_ids,
        TOPK=topk, H=hidden, eps=POST_NORM_EPS,
        APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True,
        BLOCK_H=block_h, num_warps=8 if block_h >= 4096 else 4, num_stages=2,
    )
    return out


def _experts(x, topk_weights, topk_ids, state, ids_are_local=False):
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = (topk_ids if ids_are_local else bank["expert_map"][topk_ids]).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    block_size_m = _bm_override(num_tokens, block_size_m)
    align_experts = N_EXPERTS
    if R1_ALIGN_LOCAL_BUCKETS and num_tokens >= R1_ALIGN_LOCAL_BUCKETS_MIN_TOKENS:
        align_experts = LOCAL_EXPERTS
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, align_experts)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _post_expert_norm_weighted_sum(y, topk_weights, state["post_expert_w"], local_ids)


# ---------------------------------------------------------------- V18: FA4 attention, vendored (K1-FA4 v6)
# kernels/muse_attention_decode/k1_fa4v6.py, pasted below VERBATIM as plain top-level code (tools/gen_k3_v18.py asserts byte-identity). It was generated
# by tools/gen_k1_fa4v6.py from the image's FA4 sm90 forward; nothing is loaded, written or patched at run time. Every name it binds starts with k1fa4_ /
# K1FA4_ / _k1fa4_, except torch / triton / tl (the same module objects this file imports) and cutlass / cute / const_expr / Int32 / Float32 (not bound
# anywhere else in this file).
# ====================================================================================================================
# K1-FA4 v8 attention (team joeyhlu, LAB). Generated by tools/gen_k1_fa4v8.py; do not edit by hand.
# Part 1 vendors FlashAttention-4's CuTe-DSL Hopper forward (FlashAttentionForwardSm90) from
#   sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py (lmsysorg/sglang:dev-muse-spark-open)
#   sha256 16815521b74bfc9e5fc16411e503da6ff5b4146a0d40e0563073bafcc90d61c1
# with these source changes, applied at generation time: launch min_blocks_per_mp / use_pdl and the
# 1-warpgroup setmaxnreg pair from k1fa4_K; K/V pipeline depth from k1fa4_K; V smem stage, Vt view, PV MMA
# types, P conversion and softmax class through k1fa4_* helpers (FP8 PV core when k1fa4_K['pv8']); no 16-bit
# type check; E4M3 QK tiled MMA; the shared-storage handle kept on self for the producer; an optional PDL
# trigger before the epilogue. The producer methods load / load_KV / load_Bias are removed (the k1fa4_Sm90Fwd /
# k1fa4_Sm90FwdPV8 load methods replace them). Module-level names carry a k1fa4_ prefix.
# FlashAttention-4 license (BSD 3-Clause):
#   BSD 3-Clause License
#
#   Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ====================================================================================================================
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


G2_ENABLE = True


G2_PRT = True


G2_PRT_PDL = True


G2_PRT_MODE = {1: 'fused'}


G2_P1_PDL = {'qkv': False, 'o': True, 'pooler': False, 'unpooler': True}


G2_PREPARE_BY_S = 190.0


G2_SELF_TEST_CASES = ((1, 0, 701), (1, 64, 702), (1, 224, 703), (1, 32, 704), (1, 160, 705), (1, 96, 706))


_G2_LIVE = {"p1pdl": False}


G2_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void g2_prt(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff, torch::Tensor esc,
            torch::Tensor fmt, torch::Tensor y, torch::Tensor x32, torch::Tensor w32, torch::Tensor part, torch::Tensor nw,
            torch::Tensor bias, torch::Tensor lstart, torch::Tensor xn, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid,
            torch::Tensor tcnt, double eps, double mult, double rscale, double deps, int64_t pdl, int64_t rt);
std::vector<int64_t> g2_info(int64_t mt, int64_t rt);
"""


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
  int vf;   // VF_PRT tail
};

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

@STREAM_H@

// grid = 256 CTAs (CTA c = pooler row tile c AND, with a router mode, router32's CTA (bx, sl) = (c % 32, c / 32)), block = 512.
// weights prefetched to L2 before the wait and loaded to registers after the pooler stream, dynamic smem MT * 4 KB (keeps L1
// for the pooler's x operand). Values are computed by the expressions of p1s_kernel<1, 0> (S = 1), router32_kernel<MT> and
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
  if (G.vf && tid < 2) {   // VF_PRT: warm the last CTA's constants in L2 (nw: 64 lines, bias: 8 lines)
    const char* pl = tid == 0 ? reinterpret_cast<const char*>(G.nw) + (rt & 63) * 128
                              : reinterpret_cast<const char*>(G.bias) + (rt & 7) * 128;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(pl));
  }
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
  float4 vb0 = make_float4(0.f, 0.f, 0.f, 0.f), vb1 = vb0;
  if (MT == 1 && G.vf) {   // VF_PRT: the routing operands ride with the pooler-row loads (red[][] is dead here)
    if (warp < 8) {
      const float4* pq = reinterpret_cast<const float4*>(G.part + (long)warp * m * NE) + lane * 2;
      const float4 u0 = __ldcg(pq), u1 = __ldcg(pq + 1);
      float4* r4 = reinterpret_cast<float4*>(&red[0][0]) + warp * 64 + lane * 2;
      r4[0] = u0;
      r4[1] = u1;
      if (warp == 0) {   // the routing warp (t = 0) keeps the bias in registers
        const float4* pb = reinterpret_cast<const float4*>(G.bias) + lane * 2;
        vb0 = pb[0];
        vb1 = pb[1];
      }
    }
  }
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
  const int t = warp;
  const bool vfs = MT == 1 && G.vf;
  const float4* rsm = reinterpret_cast<const float4*>(&red[0][0]);
  const float4* pbias = reinterpret_cast<const float4*>(G.bias) + lane * 2;
  const float4 c0b = vfs ? vb0 : pbias[0], c1b = vfs ? vb1 : pbias[1];
  const long sstride = (long)m * NE;
  const float4* pq0 = reinterpret_cast<const float4*>(G.part + (long)t * NE) + lane * 2;
  float4 a0 = vfs ? rsm[lane * 2] : __ldcg(pq0), a1 = vfs ? rsm[lane * 2 + 1] : __ldcg(pq0 + 1);
#pragma unroll
  for (int q = 1; q < 8; ++q) {   // FP32 split-K partials summed in a fixed order
    const float4* pq = reinterpret_cast<const float4*>(G.part + q * sstride + (long)t * NE) + lane * 2;
    const float4 b0 = vfs ? rsm[q * 64 + lane * 2] : __ldcg(pq), b1 = vfs ? rsm[q * 64 + lane * 2 + 1] : __ldcg(pq + 1);
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
  G.vf = (int)((pdl >> 1) & 1);
  TORCH_CHECK(M == 1 && rt == 1, "g2_prt: k3_v19 builds only the T = 1 kernel with the router inside the pooler kernel");
  g2::prt_launch<1, 1>(A, G, (int)(pdl & 1));
}

std::vector<int64_t> g2_info(int64_t mt, int64_t rt) {
  TORCH_CHECK(mt == 1 && rt == 1, "g2_info: k3_v19 builds only prt_kernel<1, 1>");
  return g2::prt_info<1, 1>();
}

"""


def _g2_stream_h(p1_src):
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
    _pt1_wait("g2")
    if _PT1["g2_mod"] is not None:
        return _PT1["g2_mod"]
    from torch.utils.cpp_extension import load_inline
    cuda = P1_CUDA_SRC + "\n" + G2_CUDA_SRC.replace("@STREAM_H@", _g2_stream_h(P1_CUDA_SRC))
    flags = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a", "-DP1_LAYER_ONLY=1"]
    tag = hashlib.sha1((G2_CPP_SRC + cuda + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"k3_g2_{tag}", cpp_sources=G2_CPP_SRC, cuda_sources=cuda, functions=["g2_prt", "g2_info"],
                      extra_cuda_cflags=flags, verbose=False)
    print(f"[g2] compiled in {time.time() - t0:.1f}s", flush=True)
    _PT1["g2_mod"] = mod
    return mod


def _g2_p1_pdl(proj):
    return 1 if (G2_ENABLE and _G2_LIVE["p1pdl"] and G2_P1_PDL.get(proj)) else 0


def _g2_prt_run(g2, P, ffn_input, router_input, state, data, pdl):
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
    g2 = state.get("g2")
    if not (G2_ENABLE and G2_PRT and g2 is not None and g2.get("prt")) or tokens != 1:
        return None
    if _vf_t1_live(state):
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
        return _g2_prt_run(g2, p1["proj"]["pooler"], ffn_input, router_input, state, data,
                           (1 if G2_PRT_PDL else 0) | (2 if (VF_PRT and _VF["prt"]) else 0))
    except Exception as ex:
        g2["prt"] = False
        print(f"[g2] G2-A raised at call time, k3_v18 path from now on: {ex!r}", flush=True)
        return None


def _g2_selftest_prt(state, g2):
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
        except Exception as ex:
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
    except Exception as ex:
        print(f"[g2] build / self-test FAILED -> k3_v18 path: {ex!r}", flush=True)
        state["g2"] = None
    print(f"[g2] ready in {time.time() - t1:.1f}s (prepare so far {time.time() - t0:.1f}s)", flush=True)
    return state["g2"]


P1M_INST = ((1, 1, 1), (11, 1, 1), (21, 1, 1), (2, 1, 1), (12, 1, 1), (22, 1, 1), (13, 1, 1), (23, 1, 1), (1, 1, 2), (12, 1, 2),
            (1, 2, 1), (2, 2, 1), (11, 2, 1), (12, 2, 1), (21, 2, 1), (22, 2, 1), (13, 2, 1), (23, 2, 1), (1, 2, 2), (12, 2, 2),
            (1, 4, 1), (11, 4, 1), (12, 4, 1), (21, 4, 1), (22, 4, 1), (13, 4, 1), (1, 4, 2), (11, 4, 2), (12, 4, 2))


P1K_INST = ((1, 1, 1), (12, 1, 1), (1, 2, 1), (12, 2, 1), (1, 4, 1), (12, 4, 1))


P1M_CPP_ADD = r"""
void p1m_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mt, int64_t rt,
              int64_t pdl, torch::Tensor ilbuf, int64_t il_step);
std::vector<int64_t> p1m_info(int64_t depth, int64_t mt, int64_t rt, int64_t warps);
void p1k_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t ctas, int64_t depth, int64_t mt, int64_t rt,
              int64_t kmax, int64_t pdl);
std::vector<int64_t> p1k_info(int64_t depth, int64_t mt, int64_t rt, int64_t warps);
"""


def _inst_lines(inst, fmt):
    return "\n".join(fmt.format(d=d, m=m, r=r) for d, m, r in inst)


def p1m_cuda_src(inst=P1M_INST, kinst=None):
    kinst = inst if kinst is None else kinst
    disp = _inst_lines(inst, "  P1M_DISPATCH({d}, {m}, {r}, (p1m::launch<{d}, {m}, {r}>(A, w, S, pd)); return)")
    info = _inst_lines(inst, "  P1M_DISPATCH({d}, {m}, {r}, return (p1m::info_t<{d}, {m}, {r}>(w)))")
    kdisp = _inst_lines(kinst, "  P1M_DISPATCH({d}, {m}, {r}, (p1m::launch_k<{d}, {m}, {r}>(A, Kp, w, C, pd)); return)")
    kinfo = _inst_lines(kinst, "  P1M_DISPATCH({d}, {m}, {r}, return (p1m::info_k<{d}, {m}, {r}>(w)))")
    return (P1M_CUDA_ADD.replace("@DISPATCH@", disp).replace("@INFO@", info).replace("@KDISPATCH@", kdisp)
            .replace("@KINFO@", kinfo))


P1M_CUDA_ADD = r"""
namespace p1m {
using p1::SBP; using p1::SBR; using p1::BIG; using p1::dec_kt; using p1::mma16816; using p1::ldw;
using p1::esc_key; using p1::esc_patch4;

__device__ __forceinline__ uint4 ldx4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void cpa16(void* sdst, const void* g) {
  const uint32_t sa = (uint32_t)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

__device__ __forceinline__ void ldb(uint32_t (&b)[8], const uint4* p) {
  const uint4 q0 = ldx4(p), q1 = ldx4(p + 1);
  b[0] = q0.x; b[1] = q0.y; b[2] = q0.z; b[3] = q0.w; b[4] = q1.x; b[5] = q1.y; b[6] = q1.z; b[7] = q1.w;
}

template <int MT, int RT> struct MaxW { static constexpr int v = RT == 1 ? 16 : 8; };

struct MArgs {
  const __nv_bfloat16* x; int M; int ldx;             // x [M, K] bf16 row-major, 1 <= M <= 8 MT
  const uint8_t* pk; const int64_t* toff; const float* scale; const int* eoff; const int2* esc; const uint8_t* fmt;
  void* out; int ldo; int N; int nsb_total; int f32;  // out [M, ldo] bf16 (f32 = 0) or fp32 (f32 = 1), columns [0, N)
  float* ws; int* cnt;                                 // split > 1: FP32 slots [S][M][N] + counters [N / (16 RT)]
  int probe;                                           // 1: stream probe (loads consumed by XOR, no decode / mma; timing only)
  int trig;                                            // 1: griddepcontrol.launch_dependents at entry (one-wave grids only)
  long long il_step;                                   // 0: P1 layout (toff + sb * SB bytes); > 0: grid-interleaved stream
};

extern __shared__ uint4 p1m_dsm[];   // CPA rings (dynamic shared memory; 0 bytes for the other depth codes)

template <int DD, int MT, int RT, int KM>
__device__ __forceinline__ void mstream(const MArgs& A, int rt0, int gw, int lane, int sb_lo, int nsb, float (&acc)[RT][MT][4]) {
  constexpr int D = DD % 10;
  constexpr bool XAU = DD >= 10;
  constexpr bool CPA = DD >= 30;      // weights through the per-warp cp.async smem ring (x at use)
  constexpr int WD = CPA ? 1 : D;     // weight register slots
  constexpr bool RAW = KM == 1;
  constexpr int NCH = RAW ? 4 : 3;
  constexpr int SBB = RAW ? SBR : SBP;
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* wp[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r)
    wp[r] = A.il_step ? A.pk + ((long long)gw * RT + r) * SBP + lane * 16
                      : A.pk + A.toff[rt0 + r] + (long long)sb_lo * SBB + lane * 16;
  const long long step = A.il_step ? A.il_step : SBB;
  const uint4* xq[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    const int tok = min(g + 8 * m, A.M - 1);
    xq[m] = reinterpret_cast<const uint4*>(A.x + (size_t)tok * A.ldx + (size_t)sb_lo * 64 + 16 * t);
  }
  uint4 w[WD][RT][NCH];
  uint4* ring = p1m_dsm + (size_t)(threadIdx.x >> 5) * (D * RT * 4 * 32);   // CPA: this warp's [D][RT][4 chunks][32 lanes]
  constexpr int XD = XAU ? 1 : D;   // x ring: D deep with the weights (refilled together), or one slot loaded at use
  uint32_t xb[XD][MT][8];
  if constexpr (CPA) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      if (d < nsb) {
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int c = 0; c < NCH; ++c) cpa16(&ring[((d * RT + r) * 4 + c) * 32 + lane], wp[r] + d * step + c * 512);
      }
      cpa_commit();   // one group per slot, empty or not: the wait count stays uniform
    }
  } else {
#pragma unroll
    for (int d = 0; d < D; ++d)
      if (d < nsb) {
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int c = 0; c < NCH; ++c) w[d][r][c] = ldw(wp[r] + d * step + c * 512);
      }
  }
  uint32_t em[RT], ev[RT];
  int cur[RT], end[RT], nkey[RT];
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    em[r] = 0u; ev[r] = 0u; cur[r] = 32; end[r] = 0; nkey[r] = BIG;
    if constexpr (!RAW) {
      const int o0 = A.eoff[rt0 + r], o1 = A.eoff[rt0 + r + 1];
      const int idx = o0 + lane;
      int2 e = make_int2(0, 0);
      if (idx < o1) e = A.esc[idx];
      em[r] = (uint32_t)e.x;
      ev[r] = (uint32_t)e.y;
      const int esb = (int)(em[r] & 0xFFFFu);
      const bool in = idx < o1 && esb >= sb_lo && esb < sb_lo + nsb;
      const uint32_t msk = __ballot_sync(0xffffffffu, in);
      cur[r] = msk ? __ffs(msk) - 1 : 32;
      end[r] = msk ? 32 - __clz(msk) : 0;
      const uint32_t mf = __shfl_sync(0xffffffffu, em[r], cur[r] & 31);
      nkey[r] = msk ? esc_key(mf, sb_lo) : BIG;
    }
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if constexpr (!XAU) {
#pragma unroll
    for (int d = 0; d < D; ++d)
      if (d < nsb) {
#pragma unroll
        for (int m = 0; m < MT; ++m) ldb(xb[d][m], xq[m] + d * 8);
      }
  }
  for (int s0 = 0; s0 < nsb; s0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int s = s0 + d;
      if (s < nsb) {
        if constexpr (XAU) {
#pragma unroll
          for (int m = 0; m < MT; ++m) ldb(xb[0][m], xq[m] + s * 8);
        }
        uint32_t (&b)[MT][8] = xb[XAU ? 0 : d];
        if constexpr (CPA) {
          cpa_wait<D - 1>();   // this lane's copies of SB s have landed
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int c = 0; c < NCH; ++c) w[0][r][c] = ring[((d * RT + r) * 4 + c) * 32 + lane];
        }
        uint4 (&wc)[RT][NCH] = w[CPA ? 0 : d];
        if constexpr (KM == 3) {
          uint32_t px = 0u;
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int c = 0; c < NCH; ++c) px ^= wc[r][c].x ^ wc[r][c].y ^ wc[r][c].z ^ wc[r][c].w;
#pragma unroll
          for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int j = 0; j < 8; ++j) px ^= b[m][j];
          acc[0][0][0] += __uint_as_float(px & 0x007FFFFFu);   // finite, keeps the loads live
        } else {
#pragma unroll
          for (int kt = 0; kt < 4; ++kt) {
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              uint32_t a[4];
              if constexpr (RAW) {
                a[0] = wc[r][kt].x; a[1] = wc[r][kt].y; a[2] = wc[r][kt].z; a[3] = wc[r][kt].w;
              } else {
                dec_kt(wc[r], kt, a);
                const int key = s * 4 + kt;
                if (key == nkey[r]) esc_patch4(a, key, sb_lo, lane, em[r], ev[r], cur[r], end[r], nkey[r]);
              }
#pragma unroll
              for (int m = 0; m < MT; ++m) mma16816(acc[r][m], a, b[m][2 * kt], b[m][2 * kt + 1]);
            }
          }
        }
        if constexpr (CPA) {
          if (s + D < nsb) {
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
              for (int c = 0; c < NCH; ++c) cpa16(&ring[((d * RT + r) * 4 + c) * 32 + lane], wp[r] + (s + D) * step + c * 512);
          }
          cpa_commit();
        } else if (s + D < nsb) {
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int c = 0; c < NCH; ++c) w[d][r][c] = ldw(wp[r] + (s + D) * step + c * 512);
          if constexpr (!XAU) {
#pragma unroll
            for (int m = 0; m < MT; ++m) ldb(xb[d][m], xq[m] + (s + D) * 8);
          }
        }
      }
    }
  }
}

__device__ __forceinline__ void mstore(const MArgs& A, int tok, int row, float v) {
  if (A.f32) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + row] = v;
  else reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
}

template <int D, int MT, int RT>
__global__ void __launch_bounds__(512, (D >= 20 ? 2 : 1)) p1m_kernel(const __grid_constant__ MArgs A) {
  constexpr int TT = 8 * MT, RS = TT + 1, ROWS = 16 * RT;
  __shared__ float red[MaxW<MT, RT>::v][ROWS * RS];   // [warp][row * RS + token]
  __shared__ int s_last;
  // tail (their griddepcontrol.wait still waits for this whole grid); the host sets it only when the grid is one wave
  if (A.trig) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = (int)(blockDim.x >> 5);
  const int rt0 = (int)blockIdx.x * RT, S = (int)gridDim.y, sidx = (int)blockIdx.y;
  const int c0 = (int)((long long)sidx * A.nsb_total / S), c1 = (int)((long long)(sidx + 1) * A.nsb_total / S);
  const int sb_lo = c0 + (int)((long long)warp * (c1 - c0) / nw);
  const int sb_hi = c0 + (int)((long long)(warp + 1) * (c1 - c0) / nw);
  float acc[RT][MT][4];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) { acc[r][m][0] = 0.f; acc[r][m][1] = 0.f; acc[r][m][2] = 0.f; acc[r][m][3] = 0.f; }
  const int gw = ((int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x) * nw + warp;   // global warp (interleaved layout)
  if (A.probe) mstream<D, MT, RT, 3>(A, rt0, gw, lane, sb_lo, sb_hi - sb_lo, acc);
  else if (A.fmt[rt0]) mstream<D, MT, RT, 1>(A, rt0, gw, lane, sb_lo, sb_hi - sb_lo, acc);   // RAW group (CTA-uniform)
  else mstream<D, MT, RT, 0>(A, rt0, gw, lane, sb_lo, sb_hi - sb_lo, acc);
  {
    const int g = lane >> 2, t = lane & 3;
    float* rw = red[warp];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        rw[(16 * r + g) * RS + 8 * m + 2 * t] = acc[r][m][0];
        rw[(16 * r + g) * RS + 8 * m + 2 * t + 1] = acc[r][m][1];
        rw[(16 * r + g + 8) * RS + 8 * m + 2 * t] = acc[r][m][2];
        rw[(16 * r + g + 8) * RS + 8 * m + 2 * t + 1] = acc[r][m][3];
      }
  }
  __syncthreads();
  const int nout = ROWS * A.M;
  for (int idx = tid; idx < nout; idx += (int)blockDim.x) {
    const int rl = idx % ROWS, tok = idx / ROWS;
    float v = 0.f;
    for (int q = 0; q < nw; ++q) v += red[q][rl * RS + tok];
    v *= A.scale[rt0 + (rl >> 4)];
    const int row = rt0 * 16 + rl;
    if (S == 1) mstore(A, tok, row, v);
    else __stcg(A.ws + ((size_t)sidx * A.M + tok) * A.N + row, v);   // this slice's slot
  }
  if (S == 1) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(A.cnt + blockIdx.x, 1) == S - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int idx = tid; idx < nout; idx += (int)blockDim.x) {
    const int rl = idx % ROWS, tok = idx / ROWS, row = rt0 * 16 + rl;
    float sum = 0.f;
    for (int q = 0; q < S; ++q) sum += __ldcg(A.ws + ((size_t)q * A.M + tok) * A.N + row);
    mstore(A, tok, row, sum);
  }
  if (tid == 0) A.cnt[blockIdx.x] = 0;
}

template <int D, int MT, int RT>
static size_t dyn_smem(int warps) { return D >= 30 ? (size_t)warps * (D % 10) * RT * 4 * 512 : 0; }

template <int D, int MT, int RT>
static void launch(const MArgs& A, int warps, int S, int pdl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(A.N / (16 * RT)), (unsigned)S, 1);
  cfg.blockDim = dim3((unsigned)(warps * 32), 1, 1);
  cfg.dynamicSmemBytes = dyn_smem<D, MT, RT>(warps);
  if (cfg.dynamicSmemBytes > 0)
    C10_CUDA_CHECK(cudaFuncSetAttribute(p1m_kernel<D, MT, RT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cfg.dynamicSmemBytes));
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, p1m_kernel<D, MT, RT>, A));
}

template <int D, int MT, int RT>
static std::vector<int64_t> info_t(int warps) {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, p1m_kernel<D, MT, RT>));
  int occ = 0;
  const size_t dyn = dyn_smem<D, MT, RT>(warps);
  if (dyn > 0)
    C10_CUDA_CHECK(cudaFuncSetAttribute(p1m_kernel<D, MT, RT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p1m_kernel<D, MT, RT>, warps * 32, dyn));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ};
}

struct KArgs { float* ws; int* cnt; int kmax; };

__device__ __forceinline__ int k_owner(long long X, long long U, int G) { return (int)(((X + 1) * G + U - 1) / U) - 1; }

template <int D, int MT, int RT>
__global__ void __launch_bounds__(512, (D >= 20 ? 2 : 1)) p1k_kernel(const __grid_constant__ MArgs A, const __grid_constant__ KArgs Kp) {
  constexpr int ROWS = 16 * RT;
  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5, nw = (int)(blockDim.x >> 5);
  const int G = (int)gridDim.x * nw, gw = (int)blockIdx.x * nw + warp;
  const int nsb = A.nsb_total, ng = A.N / ROWS, M = A.M;
  const long long U = (long long)ng * nsb;
  long long u = (long long)gw * U / G;
  const long long u1 = (long long)(gw + 1) * U / G;
  const int g8 = lane >> 2, t = lane & 3;
  while (u < u1) {
    const int grp = (int)(u / nsb), sb_lo = (int)(u - (long long)grp * nsb);
    const int len = (int)min((long long)(nsb - sb_lo), u1 - u);
    const int rt0 = grp * RT;
    float acc[RT][MT][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int m = 0; m < MT; ++m) { acc[r][m][0] = 0.f; acc[r][m][1] = 0.f; acc[r][m][2] = 0.f; acc[r][m][3] = 0.f; }
    if (A.probe) mstream<D, MT, RT, 3>(A, rt0, 0, lane, sb_lo, len, acc);
    else if (A.fmt[rt0]) mstream<D, MT, RT, 1>(A, rt0, 0, lane, sb_lo, len, acc);
    else mstream<D, MT, RT, 0>(A, rt0, 0, lane, sb_lo, len, acc);
    const int fw = k_owner((long long)grp * nsb, U, G), lw = k_owner((long long)(grp + 1) * nsb - 1, U, G);
    const int n = lw - fw + 1;
    if (n == 1) {
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const float sc = A.scale[rt0 + r];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int row = 16 * r + g8 + 8 * (e >> 1), tok = 8 * m + 2 * t + (e & 1);
            if (tok < M) mstore(A, tok, rt0 * 16 + row, acc[r][m][e] * sc);
          }
      }
    } else {
      float* slot = Kp.ws + ((size_t)grp * Kp.kmax + (gw - fw)) * (size_t)(ROWS * M);   // [row][tok], unscaled
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int row = 16 * r + g8 + 8 * (e >> 1), tok = 8 * m + 2 * t + (e & 1);
            if (tok < M) __stcg(slot + row * M + tok, acc[r][m][e]);
          }
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) last = atomicAdd(Kp.cnt + grp, 1) == n - 1;
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        __threadfence();
        const float* base = Kp.ws + (size_t)grp * Kp.kmax * (size_t)(ROWS * M);
        for (int idx = lane; idx < ROWS * M; idx += 32) {
          float s = 0.f;
          for (int k = 0; k < n; ++k) s += __ldcg(base + (size_t)k * (ROWS * M) + idx);
          const int row = idx / M, tok = idx - row * M;
          mstore(A, tok, rt0 * 16 + row, s * A.scale[rt0 + (row >> 4)]);   // scales are powers of two: exact
        }
        if (lane == 0) Kp.cnt[grp] = 0;
      }
    }
    u += len;
  }
}

template <int D, int MT, int RT>
static void launch_k(const MArgs& A, const KArgs& Kp, int warps, int ctas, int pdl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ctas, 1, 1);
  cfg.blockDim = dim3((unsigned)(warps * 32), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, p1k_kernel<D, MT, RT>, A, Kp));
}

template <int D, int MT, int RT>
static std::vector<int64_t> info_k(int warps) {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, p1k_kernel<D, MT, RT>));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, p1k_kernel<D, MT, RT>, warps * 32, 0));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ};
}

}  // namespace p1m

#define P1M_DISPATCH(D_, M_, R_, CALL) if (depth == D_ && mt == M_ && rt == R_) { CALL; }

void p1m_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t split, int64_t depth, int64_t mt, int64_t rt,
              int64_t pdl, torch::Tensor ilbuf, int64_t il_step) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "p1m: x bf16 contiguous");
  TORCH_CHECK(mt >= 1 && M >= 1 && M <= 8 * mt && x.size(0) == M && x.size(1) == K, "p1m: 1 <= M <= 8 mt and x [M, K]");
  TORCH_CHECK(rt >= 1 && K % 64 == 0 && N % (16 * rt) == 0 && N >= 16 * rt, "p1m: K % 64, N % (16 rt)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "p1m: x align");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt && fmt.scalar_type() == at::kByte,
              "p1m: packed dtypes");
  const int nrt = (int)(N / 16), nsb = (int)(K / 64);
  TORCH_CHECK(toff.numel() >= nrt && scale.numel() >= nrt && eoff.numel() >= nrt + 1 && fmt.numel() >= nrt, "p1m: meta");
  const bool f32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK((f32 || out.scalar_type() == at::kBFloat16) && out.stride(1) == 1 && out.stride(0) == ldo && ldo >= N &&
              out.size(0) >= M && out.size(1) >= N, "p1m: out bf16 / fp32 [M, >= N] with row stride ldo");
  TORCH_CHECK((warps == 4 || warps == 8 || (warps == 16 && rt == 1)) && split >= 1 && split <= 8 && split * warps <= nsb,
              "p1m: warps in {4, 8} (16 at rt 1), 1 <= split <= 8, split * warps <= K / 64");
  if (split > 1)
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= split * M * N && cnt.scalar_type() == at::kInt &&
                cnt.numel() >= N / (16 * rt), "p1m: split-K workspace [S * M * N] fp32 + counters (zeroed)");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1m::MArgs A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = out.data_ptr(); A.ldo = (int)ldo; A.N = (int)N; A.nsb_total = nsb; A.f32 = f32 ? 1 : 0;
  A.ws = split > 1 ? ws.data_ptr<float>() : nullptr; A.cnt = split > 1 ? cnt.data_ptr<int>() : nullptr;
  A.il_step = il_step;
  A.probe = (int)((pdl >> 1) & 1);   // pdl bit 1: stream probe
  A.trig = (int)((pdl >> 2) & 1);    // pdl bit 2: early launch_dependents
  pdl &= 1;
  if (il_step > 0) {
    TORCH_CHECK(ilbuf.scalar_type() == at::kByte && nsb % (split * warps) == 0 && il_step == (int64_t)(N / 16) * split * warps * 1536 &&
                ilbuf.numel() >= (int64_t)(N / 16) * nsb * 1536, "p1m: interleaved layout built for this (N, split, warps)");
    A.pk = ilbuf.data_ptr<uint8_t>();
  }
  const int w = (int)warps, S = (int)split, pd = (int)pdl;
@DISPATCH@
  TORCH_CHECK(false, "p1m: no instantiation for this (depth, mt, rt)");
}

std::vector<int64_t> p1m_info(int64_t depth, int64_t mt, int64_t rt, int64_t warps) {
  const int w = (int)warps;
@INFO@
  return {0, 0, 0};
}

void p1k_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
              torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t M,
              int64_t N, int64_t K, int64_t ldo, int64_t warps, int64_t ctas, int64_t depth, int64_t mt, int64_t rt,
              int64_t kmax, int64_t pdl) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "p1k: x bf16 contiguous");
  TORCH_CHECK(mt >= 1 && M >= 1 && M <= 8 * mt && x.size(0) == M && x.size(1) == K, "p1k: 1 <= M <= 8 mt and x [M, K]");
  TORCH_CHECK(rt >= 1 && K % 64 == 0 && N % (16 * rt) == 0 && N >= 16 * rt, "p1k: K % 64, N % (16 rt)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "p1k: x align");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt && fmt.scalar_type() == at::kByte,
              "p1k: packed dtypes");
  const int nrt = (int)(N / 16), nsb = (int)(K / 64), ng = (int)(N / (16 * rt));
  TORCH_CHECK(toff.numel() >= nrt && scale.numel() >= nrt && eoff.numel() >= nrt + 1 && fmt.numel() >= nrt, "p1k: meta");
  const bool f32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK((f32 || out.scalar_type() == at::kBFloat16) && out.stride(1) == 1 && out.stride(0) == ldo && ldo >= N &&
              out.size(0) >= M && out.size(1) >= N, "p1k: out bf16 / fp32 [M, >= N] with row stride ldo");
  TORCH_CHECK((warps == 4 || warps == 8 || warps == 16) && ctas >= 1 && ctas * warps <= (int64_t)ng * nsb,
              "p1k: warps in {4, 8, 16}, every warp needs >= 1 unit");
  TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= (int64_t)ng * kmax * 16 * rt * M && cnt.scalar_type() == at::kInt &&
              cnt.numel() >= ng && kmax >= 1, "p1k: workspace [groups * kmax * 16 rt * M] fp32 + counters [groups] (zeroed)");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1m::MArgs A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = out.data_ptr(); A.ldo = (int)ldo; A.N = (int)N; A.nsb_total = nsb; A.f32 = f32 ? 1 : 0;
  A.ws = nullptr; A.cnt = nullptr; A.il_step = 0;
  A.probe = (int)((pdl >> 1) & 1);   // pdl bit 1: stream probe
  A.trig = 0;
  pdl &= 1;
  p1m::KArgs Kp;
  Kp.ws = ws.data_ptr<float>(); Kp.cnt = cnt.data_ptr<int>(); Kp.kmax = (int)kmax;
  const int w = (int)warps, C = (int)ctas, pd = (int)pdl;
@KDISPATCH@
  TORCH_CHECK(false, "p1k: no instantiation for this (depth, mt, rt)");
}

std::vector<int64_t> p1k_info(int64_t depth, int64_t mt, int64_t rt, int64_t warps) {
  const int w = (int)warps;
@KINFO@
  return {0, 0, 0};
}
"""


def p1m_ws(P, split, M):
    need = split * M * P["N"]
    if P.get("m_ws") is None or P["m_ws"].numel() < need:
        P["m_ws"] = torch.zeros(need, dtype=torch.float32, device=P["buf"].device)
    if P.get("m_cnt") is None:
        P["m_cnt"] = torch.zeros(P["nrt"], dtype=torch.int32, device=P["buf"].device)


def p1m_lane_maps(device):
    L = torch.arange(32, device=device).view(32, 1, 1)
    o = torch.arange(16, device=device).view(1, 16, 1)
    h = torch.arange(2, device=device).view(1, 1, 2)
    g, t, kt, i = L // 4, L % 4, o // 4, o % 4
    row = (g + 8 * (i % 2) + 0 * h).expand(32, 16, 2)
    kk = (16 * t + 4 * kt + 2 * (i // 2) + h).expand(32, 16, 2)
    return row, kk


P1M_KPERM = [0] * 64


for _kt in range(4):
    for _t in range(4):
        for _a in range(2):
            for _h in range(2):
                P1M_KPERM[16 * _kt + 2 * _t + 8 * _a + _h] = 16 * _t + 4 * _kt + 2 * _a + _h


def p1m_interleave(P, warps, split, rt):
    nrt, nsb = P["nrt"], P["nsb"]
    if any(P["fmt_cpu"]) or nsb % (split * warps) or nrt % rt:
        return None
    per = nsb // (split * warps)
    G = (nrt // rt) * split * warps
    src = P["buf"][: nrt * nsb * 1536].view(nrt, nsb, 1536)
    bx = torch.arange(nrt // rt, device=src.device)
    sx = torch.arange(split, device=src.device)
    w = torch.arange(warps, device=src.device)
    r = torch.arange(rt, device=src.device)
    st = torch.arange(per, device=src.device)
    rtile = (bx.view(1, 1, -1, 1, 1) * rt + r.view(1, 1, 1, 1, -1)).expand(per, split, nrt // rt, warps, rt)
    sb = ((sx.view(1, -1, 1, 1, 1) * warps + w.view(1, 1, 1, -1, 1)) * per + st.view(-1, 1, 1, 1, 1)).expand(per, split, nrt // rt, warps, rt)
    il = src[rtile.reshape(-1), sb.reshape(-1)].reshape(-1).contiguous()
    assert il.numel() == nrt * nsb * 1536 and G * rt * per * 1536 == il.numel()
    return {"buf": il, "step": G * rt * 1536, "key": (warps, split, rt)}


def p1m_call(mod, P, x, out, geo, n_rows=None, ldo=None, pdl=0, il=None):
    warps, split, depth, mt, rt = geo
    n_rows = P["N"] if n_rows is None else n_rows
    ldo = out.stride(0) if ldo is None else ldo
    if split > 1:
        p1m_ws(P, split, x.shape[0])
    ws = P.get("m_ws") if split > 1 else P["ws"]
    cnt = P.get("m_cnt") if split > 1 else P["cnt"]
    if il is not None:
        assert il["key"] == (warps, split, rt) and n_rows == P["N"]
    mod.p1m_gemm(x, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], out, ws, cnt, x.shape[0], n_rows,
                 P["K"], ldo, warps, split, depth, mt, rt, pdl, il["buf"] if il is not None else P["buf"],
                 il["step"] if il is not None else 0)
    return out


def p1m_esc_columns(esc):
    e = esc[:, 0].long()
    sb, L, o, h = e & 0xFFFF, (e >> 16) & 31, (e >> 21) & 15, (e >> 25) & 1
    return 64 * sb + 16 * (L & 3) + 4 * (o >> 2) + 2 * ((o & 3) >> 1) + h


def _v20_p1_pack(w, esc_max=P1_ESC_MAX, raw_group=P1_RAW_GROUP, name="w"):
    t0 = time.time()
    assert w.dtype == torch.bfloat16 and w.is_cuda and w.dim() == 2
    w = w.contiguous()
    dev = w.device
    N, K = w.shape
    assert N % 16 == 0 and K % 64 == 0
    nrt, nsb = N // 16, K // 64
    bits = w.view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    hist = torch.zeros((nrt, 256), dtype=torch.int32, device=dev)
    hist.scatter_add_(1, e.view(nrt, 16 * K).long(), torch.ones((nrt, 16 * K), dtype=torch.int32, device=dev))
    cs = torch.cat([torch.zeros((nrt, 1), dtype=torch.int64, device=dev), hist.long().cumsum(1)], 1)
    win = cs[:, 17:256] - cs[:, 1:240]
    base = (win.argmax(1) + 1).to(torch.int32)
    row, kk = p1m_lane_maps(dev)
    fr = bits.view(nrt, 16, nsb, 64)[:, row, :, kk].permute(3, 4, 0, 1, 2).contiguous()
    del bits, e
    s = (fr >> 15) & 1
    ee = (fr >> 7) & 0xFF
    m = fr & 0x7F
    n = ee - base.view(nrt, 1, 1, 1, 1)
    inw = (n >= 0) & (n <= 15)
    nn = torch.where(inw, n, torch.zeros_like(n))
    sv = torch.where(inw, s, torch.zeros_like(s))
    byte = torch.where(inw, ((nn & 1) << 7) | m, torch.zeros_like(m))
    bp = byte.view(nrt, nsb, 32, 2, 8, 2).permute(0, 1, 3, 2, 4, 5).reshape(nrt, nsb, 2, 32, 16).to(torch.uint8)
    ii = torch.arange(4, device=dev).view(4, 1)
    hh = torch.arange(2, device=dev).view(1, 2)
    cw = torch.zeros((nrt, nsb, 32, 4), dtype=torch.int64, device=dev)
    for f, tgt in (((nn >> 1) & 1, 8), ((nn >> 2) & 1, 9), ((nn >> 3) & 1, 10), (sv, 15)):
        pos = (tgt + 16 * hh + 4 * ii) % 32
        cw += (f.view(nrt, nsb, 32, 4, 4, 2).to(torch.int64) << pos).sum(dim=(-1, -2))
    cw8 = _wrap32(cw).contiguous().view(torch.uint8).view(nrt, nsb, 1, 32, 16)
    packed_t = torch.cat([bp, cw8], dim=2).reshape(nrt, nsb * 1536)
    del byte, bp, cw, cw8, nn, sv, s, m
    fr16 = torch.where(fr >= 32768, fr - 65536, fr).to(torch.int16)
    raw_t = fr16.view(nrt, nsb, 32, 4, 4, 2).permute(0, 1, 3, 2, 4, 5).contiguous().view(torch.uint8).reshape(nrt, nsb * 2048)
    idx = (~inw).nonzero()
    if idx.shape[0]:
        key = (idx[:, 0] * nsb + idx[:, 1]) * 4 + (idx[:, 3] >> 2)
        idx = idx[torch.argsort(key, stable=True)]
    rt_i, sb_i, L_i, o_i, h_i = idx.unbind(1)
    raw16 = fr[rt_i, sb_i, L_i, o_i, h_i]
    rv = torch.where(raw16 >= 32768, raw16 - 65536, raw16).to(torch.int16).view(torch.bfloat16).double()
    j = (P1_C - base[rt_i]).double()
    sd = rv * torch.pow(torch.full_like(j, 2.0), j)
    sb16 = sd.to(torch.bfloat16)
    nan = torch.isnan(rv)
    exact = nan | (sb16.double() == sd)
    sc_bits = sb16.view(torch.int16).to(torch.int32) & 0xFFFF
    sc_bits = torch.where(nan, torch.full_like(sc_bits, 0x7FC0), sc_bits)
    cnt = torch.bincount(rt_i, minlength=nrt)
    bad = (torch.bincount(rt_i[~exact], minlength=nrt) > 0 if bool((~exact).any())
           else torch.zeros(nrt, dtype=torch.bool, device=dev))
    need_raw = (cnt > esc_max) | bad
    ng = (nrt + raw_group - 1) // raw_group
    pad = torch.zeros(ng * raw_group, dtype=torch.bool, device=dev)
    pad[:nrt] = need_raw
    raw_tile = pad.view(ng, raw_group).any(1, keepdim=True).expand(ng, raw_group).reshape(-1)[:nrt]
    keep = ~raw_tile[rt_i]
    meta = (sb_i | (L_i << 16) | (o_i << 21) | (h_i << 25))[keep]
    vals = (sc_bits.long() | (raw16.long() << 16))[keep]
    esc = torch.stack([_wrap32(meta.long()), _wrap32(vals)], 1).contiguous()
    if esc.numel() == 0:
        esc = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    cnt_k = torch.bincount(rt_i[keep], minlength=nrt)
    eoff = torch.zeros(nrt + 1, dtype=torch.int32, device=dev)
    eoff[1:] = cnt_k.cumsum(0).to(torch.int32)
    base_f = torch.where(raw_tile, torch.full_like(base, P1_C), base)
    scale = torch.pow(2.0, (base_f - P1_C).double()).float()
    fmt = raw_tile.to(torch.uint8)
    sizes = torch.where(raw_tile, torch.full_like(base, nsb * 2048), torch.full_like(base, nsb * 1536)).long()
    toff = torch.zeros(nrt, dtype=torch.int64, device=dev)
    toff[1:] = sizes.cumsum(0)[:-1]
    total = int(sizes.sum())
    buf = torch.empty(total + 64, dtype=torch.uint8, device=dev)
    toff_c, raw_c = toff.cpu().tolist(), raw_tile.cpu().tolist()
    for r in range(nrt):
        src = raw_t[r] if raw_c[r] else packed_t[r]
        buf[toff_c[r]:toff_c[r] + src.numel()] = src
    return {"name": name, "N": N, "K": K, "nrt": nrt, "nsb": nsb, "buf": buf, "toff": toff, "scale": scale, "fmt": fmt,
            "fmt_cpu": raw_c, "ws": torch.zeros(72 * N, dtype=torch.float32, device=dev),
            "cnt": torch.zeros(nrt, dtype=torch.int32, device=dev),
            "eoff": eoff, "esc": esc, "base": base_f.contiguous(), "bytes": total + nrt * 17 + 8 * int(keep.sum()),
            "n_esc": int(keep.sum()), "n_esc_all": int(idx.shape[0]), "n_raw_tiles": int(raw_tile.sum()),
            "max_esc_tile": int(cnt.max()) if nrt else 0, "unscalable": int((~exact).sum()),
            "pack_s": time.time() - t0}


V20_P1M_ENABLE = True


V20_P1M_ROUTE = {('qkv', 16): (8, 2, 1, 2, 1, 0), ('o', 16): (8, 1, 1, 2, 2, 1), ('cat', 16): (8, 1, 1, 2, 1, 0), ('unpooler', 16): (8, 1, 1, 2, 2, 1), ('unpooler', 32): (8, 1, 1, 4, 2, 1)}


V20_P1M_TOL = (4e-3, 1e-2)


V20_P1M_PDL = True


V20_P1M_TRIG = True


V20_P1_TRIG = {'qkv': True, 'o': True, 'pooler': True, 'unpooler': True}


def _v20_inst():
    return tuple(sorted({(g[2], g[3], g[4]) for g in V20_P1M_ROUTE.values()}))


def _v20_ok(x, k):
    return x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == k and x.is_contiguous() and x.data_ptr() % 16 == 0


def _v20_p1m(state, proj, x):
    st = state.get("p1m")
    if not V20_P1M_ENABLE or st is None:
        return None
    T = x.shape[0]
    if T <= 8 or T > 32:
        return None
    key = (proj, _k2a_bucket_of(T))
    if key not in V20_P1M_ROUTE:
        return None
    geo = st["route"].get(key)
    P = st["proj"].get(proj)
    if geo is None or P is None or T > 8 * geo[3] or not _v20_ok(x, P["K"]):
        return None
    try:
        out = torch.empty((T, P["N"]), dtype=torch.float32 if proj == "cat" else torch.bfloat16, device=x.device)
        pdl = (1 if V20_P1M_PDL else 0) | (4 if V20_P1M_TRIG and st["wave1"].get(key) else 0)
        p1m_call(st["mod"], P, x, out, geo[:5], il=st["il"].get(key), pdl=pdl)
        return out
    except Exception as ex:
        _k3_disable(state, "p1m", None, "P1M", ex)
        return None


def _v20_weight(state, proj):
    return state["v8"]["wcat"] if proj == "cat" else state["dense"]["qkv_gate" if proj == "qkv" else proj]


def _v20_selftest(st, state, proj, key, gen):
    P = st["proj"][proj]
    geo = st["route_try"][key]
    w = _v20_weight(state, proj)
    N, K, dev, T = P["N"], P["K"], w.device, key[1]
    f32 = proj == "cat"

    def run(x):
        out = torch.empty((T, N), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
        p1m_call(st["mod"], P, x, out, geo[:5], il=st["il"].get(key))
        return out
    fin = torch.isfinite(w.float()).all(1)
    esc = sorted(set(p1m_esc_columns(P["esc"]).tolist())) if P["n_esc"] else []
    esc = esc[::max(1, len(esc) // 48)][:48]
    rnd = torch.randint(0, K, (64,), generator=gen, device=dev).tolist()
    cols = esc + rnd[:16]
    cols += rnd[16:16 + (-len(cols)) % T]
    bad = flushed = 0
    for i in range(0, len(cols), T):
        cc = cols[i:i + T]
        x = torch.zeros((T, K), dtype=torch.bfloat16, device=dev)
        ci = torch.tensor(cc, device=dev)
        x[torch.arange(T, device=dev), ci] = 1.0
        got = run(x).float()
        ref = w[:, ci].t().float()
        mis = (got != ref) & ~(torch.isnan(got) & torch.isnan(ref)) & fin[None, :]
        sub = mis & (got == 0) & (ref.abs() < 2.0 ** -126)
        flushed += int(sub.sum())
        bad += int((mis & ~sub).sum())
    x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
    got = run(x)
    again = run(x)
    ref64 = x.double() @ w.double().t()
    today = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
    torch.cuda.synchronize()
    e64 = float((got.double() - ref64).norm() / ref64.norm())
    etoday = float((got.float() - today.float()).norm() / today.float().norm())
    same = bool(torch.equal(got, again))
    ok = bad == 0 and e64 < V20_P1M_TOL[0] and etoday < V20_P1M_TOL[1] and same
    return ok, (f"{len(cols)} one-hot cols ({len(esc)} escape) bad {bad} (subnormal flush {flushed}); random x rel vs fp64 "
                f"{e64:.2e} vs today {etoday:.2e}; repeat identical {same}")


def _v20_prepare(state):
    state["p1m"] = None
    if not V20_P1M_ENABLE or not V20_P1M_ROUTE:
        return None
    st1 = state.get("p1")
    mod = st1["mod"] if st1 is not None else None
    if mod is None or not hasattr(mod, "p1m_gemm"):
        print("[v20] P1M: no P1 build with p1m_gemm -> today's path at T = 9..32", flush=True)
        return None
    t0 = time.time()
    st = {"mod": mod, "proj": {}, "route": {}, "route_try": dict(V20_P1M_ROUTE), "il": {}, "wave1": {}}
    try:
        gen = torch.Generator(device=state["dense"]["qkv_gate"].device).manual_seed(2020)
        for proj in sorted({k[0] for k in V20_P1M_ROUTE}):
            if proj == "cat" and state.get("v8") is None:
                continue
            w = _v20_weight(state, proj)
            P = _v20_p1_pack(w, name=proj)
            mism = p1m_verify(mod, P, w)
            torch.cuda.synchronize()
            print(f"[v20] P1M {p1_describe(P)} | k-permuted unpack mismatches {mism} -> {'ok' if mism == 0 else 'DROPPED'}",
                  flush=True)
            if mism:
                continue
            st["proj"][proj] = P
            for key in sorted(k for k in V20_P1M_ROUTE if k[0] == proj):
                geo = V20_P1M_ROUTE[key]
                if geo[5]:
                    il = p1m_interleave(P, geo[0], geo[1], geo[4])
                    if il is None:
                        print(f"[v20] P1M {key}: no interleaved layout for {geo} -> route off", flush=True)
                        continue
                    st["il"][key] = il
                ok, msg = _v20_selftest(st, state, proj, key, gen)
                print(f"[v20] P1M self-test {proj} T={key[1]} geo {geo}: {msg} -> {'ok' if ok else 'FAIL: route off'}", flush=True)
                if ok:
                    st["route"][key] = geo
                    occ = int(mod.p1m_info(geo[2], geo[3], geo[4], geo[0])[2])
                    sms = torch.cuda.get_device_properties(w.device).multi_processor_count
                    st["wave1"][key] = (P["N"] // (16 * geo[4])) * geo[1] <= occ * sms
                else:
                    st["il"].pop(key, None)
    except Exception as ex:
        print(f"[v20] P1M prepare / self-test FAILED -> today's path at T = 9..32: {ex!r}", flush=True)
        return None
    print(f"[v20] P1M routes live {sorted(st['route'])} (prepare {time.time() - t0:.1f}s)", flush=True)
    state["p1m"] = st if st["route"] else None
    return state["p1m"]


def p1m_verify(mod, P, w):
    u = p1_unpack(mod, P)
    K = w.shape[1]
    perm = torch.tensor(P1M_KPERM, device=w.device)
    cols = (torch.arange(K // 64, device=w.device).view(-1, 1) * 64 + perm.view(1, -1)).reshape(-1)
    return int((u.view(torch.int16) != w[:, cols].view(torch.int16)).sum())


_V20_WAVE1 = {}


def _v20_p1_trig(st, proj, geo, mode):
    warps, split, depth = geo
    P = st["proj"][proj]
    key = (proj, warps, split, depth, mode, P["rows_t8"])
    if key not in _V20_WAVE1:
        occ = int(st["mod"].p1s_info(depth, mode, warps)[2])
        sms = torch.cuda.get_device_properties(P["buf"].device).multi_processor_count
        _V20_WAVE1[key] = (P["rows_t8"] // 16) * split <= occ * sms
    return 2 if (V20_P1_TRIG.get(proj) and _V20_WAVE1[key]) else 0


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
    if constexpr (MG == 1) {   // VSL red merge (see KS1)
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
VSL_ROUTE = {(1, 1): ('ks1', 1, 16, 1, 1), (1, 8): ('ks1', 8, 16, 4, 1), (16, 1): ('ks3', 16, 1)}
VSL_KS1_INSTS = ((1, 16, 1, 1), (8, 16, 4, 1))
VSL_KS3_INSTS = ((16, 1),)
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
                              functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"] + _dg1_fold("fns"),
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
    if not (((geo is not None and (L, *geo) in passed) or _vsl_pick(cfg, state) is not None) and cfg["window"] == 2048
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


def _ks1_run(qkv, data, state):
    cfg = data["config"]
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
    if ok_geo and KS1_P1_CARVEOUT >= 0 and _P1_MOD is not None and hasattr(_P1_MOD, "p1s_carveout"):
        _P1_MOD.p1s_carveout(int(KS1_P1_CARVEOUT))
        print(f"[ks1] P1 preferred smem carveout {KS1_P1_CARVEOUT}%", flush=True)
    live = {l: tuple(KS1_GEOM[l]) for l in KS1_ROUTE if (l, *KS1_GEOM.get(l, (0, 0))) in ok_geo}
    print(f"[ks1] route {'live ' + str(live) if live else 'FAILED: FA4 v8 stays'} (build {_KS1['build_s']:.1f}s, total "
          f"{time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)


DG1_ENABLE = True
DG1_ROUTE = {('cat', 32): True, ('qkv', 32): True, ('o', 32): True}
DG1_GEOMETRY = {('cat', 32): (32, 2, 1, 3), ('qkv', 32): (32, 2, 1, 4), ('o', 32): (32, 1, 1, 1)}
DG1_PDL = True
DG1_TRIG = True
DG1_TOL = (4e-3, 1e-4, 1e-2)

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


def dg1_pack(w):
    n, k = w.shape
    assert n % 8 == 0 and k % 64 == 0
    return w.contiguous().view(n // 8, 8, k // 64, 8, 8).permute(2, 0, 3, 1, 4).contiguous()


DG1_FOLD_KS1 = True


def _dg1_insts():
    return tuple(sorted({DG1_GEOMETRY[k][:3] for k in DG1_ROUTE if k in DG1_GEOMETRY}))


def _dg1_fold_on():
    return bool(DG1_ENABLE and DG1_FOLD_KS1 and _dg1_insts())


def _dg1_fold(part):
    if not _dg1_fold_on():
        return [] if part == "fns" else ""
    return {"cuda": "\n" + dg1_cuda_src(list(_dg1_insts())), "cpp": DG1_CPP_SRC, "fns": ["dg1_run", "dg1_info"]}[part]


_DG1 = {"mod": None, "insts": None, "build_s": 0.0}


def _dg1_build(insts):
    if _DG1["mod"] is not None and _DG1["insts"] == insts:
        return _DG1["mod"]
    from torch.utils.cpp_extension import load_inline
    cuda = dg1_cuda_src(list(insts))
    tag = hashlib.sha1((DG1_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    _DG1["mod"] = load_inline(name=f"dg1_{tag}", cpp_sources=DG1_CPP_SRC, cuda_sources=cuda, functions=["dg1_run", "dg1_info"],
                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _DG1["insts"], _DG1["build_s"] = insts, time.time() - t0
    return _DG1["mod"]


def _dg1_weight(state, proj):
    if proj == "cat":
        return state["v8"]["wcat"] if state.get("v8") is not None else None
    return state["dense"]["qkv_gate" if proj == "qkv" else proj]


VD_ENABLE = True
VD_BUCKETS = [128, 256]
VD_PROJS = ["qkv", "o", "cat", "unpooler"]
VD_PDL = True
VD_TRIG = True
VD_PF = 2
VD_ROT = 0
VD_FLAGS = 8
VD_TOL = (1.0e-2, 1.0e-4, 1.0e-2)
VD_INSTS = [(2, 128, 5, 3, 1, 1)]
VD_ROUTE = {('cat', 128): 0}
VD_TT = {('unpooler', 128): True}
VD_TT_PREP = ['unpooler']
VD_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t vd_run(int64_t kid, torch::Tensor x, torch::Tensor wpk, torch::Tensor wraw, torch::Tensor y, torch::Tensor err,
               torch::Tensor stamps, int64_t nrb, int64_t w_rb, int64_t w_s, int64_t flags, int64_t pdl, int64_t pf);
"""
VD_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstring>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace vd {

constexpr long long WAIT_BUDGET = 2000000000LL;   // ~1 s of clock64: a hang becomes err[0] = 1 (R9)
enum : int { F_EXIT = 1, F_NOMMA = 2, F_NOX = 4, F_EFW = 8, F_ELX = 16, F_STAMP = 32, F_WMAP = 64, F_TRIG = 128, F_WAIT0 = 256,
             F_DYN = 512 };

struct Params {
  const uint8_t* w;          // packed tiles (bulk mode): tile (rb, s) at w + rb * w_rb + s * w_s (bytes)
  void* y;                   // [T, ldy] bf16 or f32
  int* err;
  unsigned long long* stamps;   // [grid][4] (flag 32)
  long long w_rb, w_s;
  int nks, T, ldy, flags, pf;   // pf: weight stages issued before griddepcontrol.wait (the rest go x-first)
  int rot;                      // K-start rotation per row block (host arg pf >> 8)
};

template <int NWG, int BN, int NST, int CL, int F32, int RBD>
struct Cfg {
  static constexpr int BM = 64 * NWG;
  static constexpr int WB = BM * 128;              // weight tile bytes (BM rows x 64 bf16)
  static constexpr int XB = BN * 128;              // x tile bytes (BN tokens x 64 bf16)
  static constexpr int SB = WB + XB;               // stage bytes (multiple of 1024)
  static constexpr int NTH = NWG * 128;            // consumer threads
  static constexpr int THREADS = NTH + 32;         // + one producer warp
  static constexpr int NACC = BN / 2;              // fp32 accumulators per consumer thread
  static constexpr int NB = BN / 8;                // 8-token blocks
  static constexpr int JMAX = (NB + CL - 1) / CL;  // most token blocks a rank owns
  static constexpr int RB_BYTES = CL > 1 ? (CL - 1) * JMAX * NTH * 16 : 0;   // [src slot][block][thread] float4
  static constexpr int RB_OFF = RBD ? NST * SB : 0;
  static constexpr int BAR_OFF = NST * SB + (RBD ? RB_BYTES : 0);
  static constexpr int SMEM = 1024 + BAR_OFF + 16 * NST + 64;
  static constexpr int CTR_OFF = BAR_OFF + 16 * NST + 8;    // dynamic split: stage counter (rank 0's is used)
  static constexpr int ENDF_OFF = CTR_OFF + 4;               // dynamic split: ring index of this rank's end marker
  static constexpr int OB = F32 ? 4 : 2;           // output element bytes
  static constexpr int SROW = F32 ? (BM + 4) : (BM + 8);   // staging row pitch (elements), 16-B multiple
  static constexpr int S_OFF = RBD ? 0 : RB_BYTES;  // staging tile [owned tokens][BM] in the drained ring
  static constexpr int S_BYTES = JMAX * 8 * SROW * OB;
  static_assert(SB % 1024 == 0, "stage alignment");
  static_assert(S_OFF + S_BYTES <= NST * SB, "epilogue buffers alias the ring");
  static_assert(SMEM <= 232448, "smem");
  static_assert(CL >= 1 && CL <= NB, "cluster split");
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
__device__ __forceinline__ uint64_t pol_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_h(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t mapa(uint32_t local_addr, uint32_t rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(local_addr), "r"(rank));
  return ra;
}
__device__ __forceinline__ void st_async_f4(uint32_t raddr, float a, float b, float c, float d, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n"
               :: "r"(raddr), "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)), "r"(__float_as_uint(d)),
                  "r"(rbar) : "memory");
}
__device__ __forceinline__ int atom_add_cl(uint32_t raddr, int v) {
  int old;
  asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0, [%1], %2;\n" : "=r"(old) : "r"(raddr), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ bool mbar_try_cl(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait_cl(uint32_t b, uint32_t par, int* err, bool& dead) {
  if (dead || mbar_try_cl(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try_cl(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 2); dead = true; return; }
  }
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db);
template <> __device__ __forceinline__ void wgmma_ss<128>(float* d, uint64_t da, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
               : "l"(da), "l"(db));
}

template <int NWG, int BN, int NST, int CL, int F32, int RBD>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
vd_kernel(const __grid_constant__ CUtensorMap tmx, const __grid_constant__ CUtensorMap tmw, const Params P) {
  using C = Cfg<NWG, BN, NST, CL, F32, RBD>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t s_raw = sa(smem_raw);
  const uint32_t sbase = (s_raw + 1023u) & ~1023u;
  uint8_t* gbase = smem_raw + (sbase - s_raw);
  const uint32_t full0 = sbase + C::BAR_OFF, empty0 = full0 + 8 * NST, rbar = full0 + 16 * NST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int flags = P.flags;
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int s0 = ((int)crank * P.nks) / CL, s1 = (((int)crank + 1) * P.nks) / CL;
  const int nk = s1 - s0;
  const int jlo = CL > 1 ? ((int)crank * C::NB) / CL : 0;          // token blocks this rank finishes
  const int jhi = CL > 1 ? (((int)crank + 1) * C::NB) / CL : C::NB;
  bool dead = false;

  if (tid == 0) {
#pragma unroll 1
    for (int i = 0; i < NST; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    if constexpr (CL > 1) {
      mbar_init(rbar, 1);
      mbar_expect_tx(rbar, (uint32_t)((CL - 1) * (jhi - jlo) * C::NTH * 16));   // the peers' pushes of my blocks
    }
    if (CL > 1 && (flags & F_DYN) != 0) {   // dynamic split only: a cluster-scope fence costs ~0.5 us at entry (R-vd-3)
      *reinterpret_cast<volatile int*>(gbase + C::CTR_OFF) = 0;
      *reinterpret_cast<volatile int*>(gbase + C::ENDF_OFF) = 0x7fffffff;
      asm volatile("fence.acq_rel.cluster;\n" ::: "memory");   // the counter init is visible before the cluster arrive
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if constexpr (CL > 1) cluster_arrive_relaxed();   // phase 0: "my barriers are initialized" (waited before any push)

  const bool dyn = CL > 1 && (flags & F_DYN) != 0;
  if (warp == NWG * 4) {
    bool pw0 = false;   // lane 0 already did its phase-0 cluster wait (dynamic split)
    if (lane == 0) {
      const bool efw = (flags & F_EFW) != 0;
      const uint64_t pw = efw ? pol_first() : 0ull;
      uint32_t txb = (uint32_t)(C::WB + C::XB);
      const uint8_t* wbase = P.w + (long long)rb * P.w_rb;
      asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmx)) : "memory");
      const int off = P.rot > 0 ? (int)(((long long)rb * P.rot) % nk) : 0;
      auto kst = [&](int i) { int j = i + off; if (j >= nk) j -= nk; return s0 + j; };
      auto issue_w = [&](int i, uint32_t dst, uint32_t bar) {
        const int s = kst(i);
        const void* src = wbase + (long long)s * P.w_s;
        if (efw) bulk_g2s_h(dst, src, C::WB, bar, pw); else bulk_g2s(dst, src, C::WB, bar);
      };
      auto issue_x = [&](int i, uint32_t dst, uint32_t bar) {
        const int s = kst(i);
        tma2d(dst + C::WB, &tmx, s * 64, 0, bar);
      };
      if (!dyn) {
      const int npro = nk < NST ? nk : NST;
      const int pf = P.pf < npro ? P.pf : npro;
#pragma unroll 1
      for (int i = 0; i < pf; ++i) {   // a few weight stages before the PDL wait (they do not depend on the previous kernel)
        mbar_expect_tx(full0 + 8 * i, txb);
        issue_w(i, sbase + i * C::SB, full0 + 8 * i);
      }
      pdl_wait();
#pragma unroll 1
      for (int i = 0; i < pf; ++i) issue_x(i, sbase + i * C::SB, full0 + 8 * i);
#pragma unroll 1
      for (int i = pf; i < npro; ++i) {   // x first: stage 0 must not queue behind NST weight tiles
        mbar_expect_tx(full0 + 8 * i, txb);
        issue_x(i, sbase + i * C::SB, full0 + 8 * i);
        issue_w(i, sbase + i * C::SB, full0 + 8 * i);
      }
#pragma unroll 1
      for (int i = NST; i < nk; ++i) {
        const int slot = i % NST;
        const uint32_t dst = sbase + slot * C::SB, bar = full0 + 8 * slot;
        mbar_wait(empty0 + 8 * slot, (uint32_t)(((i / NST) - 1) & 1), P.err, dead);
        mbar_expect_tx(bar, txb);
        issue_x(i, dst, bar);
        issue_w(i, dst, bar);
      }
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {
      if (!(lane == 0 && pw0)) asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");   // phase 0
      if constexpr (!RBD) {   // phase 1: rings drained
        asm volatile("barrier.cluster.arrive.release;\n" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");
      }
    }
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[C::NACC];
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) acc[q] = 0.f;
  bool nomma = false;
  const bool w0 = (flags & F_WAIT0) != 0;
#pragma unroll 1
  for (int i = 0; dyn || i < nk; ++i) {
    const int slot = i % NST;
    mbar_wait(full0 + 8 * slot, (uint32_t)((i / NST) & 1), P.err, dead);
    if (dyn && (i == *reinterpret_cast<volatile int*>(gbase + C::ENDF_OFF) || dead)) break;   // this rank's end marker
    if (!nomma) {
      const uint32_t st = sbase + slot * C::SB;
      const uint64_t da = desc_sw128(st + wg * 8192), db = desc_sw128(st + C::WB);
#pragma unroll
      for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wgmma_ss<BN>(acc, da + 2 * kk, db + 2 * kk);
      wg_commit();
      if (w0) wg_wait<0>(); else wg_wait<1>();
#pragma unroll
      for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    }
    if (w0) { if (t == 0) mbar_arrive(empty0 + 8 * slot); }
    else if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NST));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
  if (flags & F_TRIG) pdl_trigger();

  named_sync(1, C::NTH);   // every warpgroup is done with the ring
  const int g = lane >> 2, t4 = lane & 3;
  const int r0 = wg * 64 + (warp & 3) * 16 + g;     // this thread's first row within the tile (second: r0 + 8)
  uint8_t* stg = gbase + C::S_OFF;
  if constexpr (CL > 1) {
    cluster_wait();                                     // phase 0: every rank's rbar is initialized
    if constexpr (!RBD) { cluster_arrive(); cluster_wait(); }   // phase 1: every rank's ring is drained (buffer aliases it)
    const uint32_t rbl = sbase + C::RB_OFF;
#pragma unroll
    for (int p = 0; p < CL; ++p) {
      if (p == (int)crank) continue;
      const int plo = (p * C::NB) / CL, phi = ((p + 1) * C::NB) / CL;
      const int slot = (int)crank < p ? (int)crank : (int)crank - 1;
      const uint32_t rbuf = mapa(rbl, (uint32_t)p), rbr = mapa(rbar, (uint32_t)p);
#pragma unroll
      for (int j = 0; j < C::NB; ++j) {
        if (j < plo || j >= phi) continue;
        st_async_f4(rbuf + (uint32_t)(((slot * C::JMAX + (j - plo)) * C::NTH + tid) * 16),
                    acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3], rbr);
      }
    }
    mbar_wait_cl(rbar, 0, P.err, dead);                 // the other ranks' copies of my blocks have landed
    const float4* RB = reinterpret_cast<const float4*>(gbase + C::RB_OFF);
#pragma unroll
    for (int j = 0; j < C::NB; ++j) {
      if (j < jlo || j >= jhi) continue;
      float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
      for (int s = 0; s < CL - 1; ++s) {
        const float4 q = RB[(s * C::JMAX + (j - jlo)) * C::NTH + tid];
        v0 += q.x; v1 += q.y; v2 += q.z; v3 += q.w;
      }
      acc[4 * j] = v0; acc[4 * j + 1] = v1; acc[4 * j + 2] = v2; acc[4 * j + 3] = v3;
    }
  }
#pragma unroll
  for (int j = 0; j < C::NB; ++j) {
    if (j < jlo || j >= jhi) continue;
    const int tt = 8 * (j - jlo) + 2 * t4;
    if constexpr (F32) {
      float* s = reinterpret_cast<float*>(stg);
      s[tt * C::SROW + r0] = acc[4 * j];
      s[(tt + 1) * C::SROW + r0] = acc[4 * j + 1];
      s[tt * C::SROW + r0 + 8] = acc[4 * j + 2];
      s[(tt + 1) * C::SROW + r0 + 8] = acc[4 * j + 3];
    } else {
      __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(stg);
      s[tt * C::SROW + r0] = __float2bfloat16_rn(acc[4 * j]);
      s[(tt + 1) * C::SROW + r0] = __float2bfloat16_rn(acc[4 * j + 1]);
      s[tt * C::SROW + r0 + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
      s[(tt + 1) * C::SROW + r0 + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
  named_sync(1, C::NTH);
  {
    constexpr int CH = C::BM * C::OB / 16;            // 16-B chunks per token row
    const int tlo = 8 * jlo;
    int thi = 8 * jhi;
    if (thi > P.T) thi = P.T;
    const int ntok = thi > tlo ? thi - tlo : 0;
    uint8_t* yb = reinterpret_cast<uint8_t*>(P.y) + (size_t)rb * C::BM * C::OB;
#pragma unroll 1
    for (int idx = tid; idx < ntok * CH; idx += C::NTH) {
      const int tt = idx / CH, c = idx - tt * CH;
      const uint4 v = *reinterpret_cast<const uint4*>(stg + (size_t)tt * C::SROW * C::OB + c * 16);
      *reinterpret_cast<uint4*>(yb + ((size_t)(tlo + tt) * P.ldy) * C::OB + c * 16) = v;
    }
  }
}


}  // namespace vd

typedef CUresult (*vd_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static vd_encode_fn vd_get_encode() {
  static vd_encode_fn fn = nullptr;
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
    else fn = reinterpret_cast<vd_encode_fn>(p);
  }
  return fn;
}

static void vd_map(CUtensorMap* tm, const void* ptr, int64_t rows, int64_t cols, int box_rows) {
  vd_encode_fn enc = vd_get_encode();
  TORCH_CHECK(enc != nullptr, "vd: cuTensorMapEncodeTiled unavailable");
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)(cols * 2)};
  cuuint32_t box[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1u, 1u};
  const CUresult r = enc(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "vd: tensor map encode failed ", (int)r);
}

template <int NWG, int BN, int NST, int CL, int F32, int RBD>
static int64_t vd_launch_t(const torch::Tensor& x, const torch::Tensor& wpk, const torch::Tensor& wraw, const torch::Tensor& y,
                           const torch::Tensor& err, const torch::Tensor& stamps, int64_t nrb, int64_t w_rb, int64_t w_s,
                           int64_t flags, int64_t pdl, int64_t pf) {
  using C = vd::Cfg<NWG, BN, NST, CL, F32, RBD>;
  auto kern = vd::vd_kernel<NWG, BN, NST, CL, F32, RBD>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    attr = true;
  }
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "vd: x bf16 [T, K]");
  const int64_t T = x.size(0), K = x.size(1);
  TORCH_CHECK(T >= 1 && T <= BN && K % 64 == 0 && x.data_ptr() != nullptr && ((uintptr_t)x.data_ptr() % 16) == 0, "vd: x shape");
  TORCH_CHECK(y.is_cuda() && y.dim() == 2 && y.stride(1) == 1 && y.size(0) >= T && y.size(1) >= nrb * C::BM, "vd: y");
  TORCH_CHECK(y.scalar_type() == (F32 ? at::kFloat : at::kBFloat16), "vd: y dtype");
  TORCH_CHECK((y.stride(0) * C::OB) % 16 == 0 && ((uintptr_t)y.data_ptr() % 16) == 0, "vd: y alignment");
  CUtensorMap tmx, tmw;
  memset(&tmw, 0, sizeof(tmw));
  vd_map(&tmx, x.data_ptr(), T, K, BN);
  TORCH_CHECK(wpk.is_cuda() && ((uintptr_t)wpk.data_ptr() % 16) == 0, "vd: wpk");
  vd::Params P;
  P.w = reinterpret_cast<const uint8_t*>(wpk.data_ptr());
  P.y = y.data_ptr();
  P.err = err.data_ptr<int>();
  P.stamps = reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>());
  P.w_rb = w_rb; P.w_s = w_s;
  P.nks = (int)(K / 64); P.T = (int)T; P.ldy = (int)y.stride(0); P.flags = (int)flags; P.pf = (int)(pf & 255); P.rot = (int)(pf >> 8);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(nrb * CL), 1, 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute at_[2];
  int na = 0;
  if (CL > 1) {
    at_[na].id = cudaLaunchAttributeClusterDimension;
    at_[na].val.clusterDim.x = CL; at_[na].val.clusterDim.y = 1; at_[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = na ? at_ : nullptr;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tmx, tmw, P));
  return 0;
}


int64_t vd_run(int64_t kid, torch::Tensor x, torch::Tensor wpk, torch::Tensor wraw, torch::Tensor y, torch::Tensor err,
               torch::Tensor stamps, int64_t nrb, int64_t w_rb, int64_t w_s, int64_t flags, int64_t pdl, int64_t pf) {
  switch (kid) {
    case 0: return vd_launch_t<2, 128, 5, 3, 1, 1>(x, wpk, wraw, y, err, stamps, nrb, w_rb, w_s, flags, pdl, pf);
  }
  TORCH_CHECK(false, "vd: unknown kid ", kid);
  return -2;
}

"""
_VD = {"mod": None, "build_s": 0.0}


def _vd_build():
    if _VD["mod"] is not None:
        return _VD["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((VD_CPP_SRC + VD_CUDA_SRC).encode()).hexdigest()[:10]
    t1 = time.time()
    _VD["mod"] = load_inline(name=f"vd_{tag}", cpp_sources=VD_CPP_SRC, cuda_sources=VD_CUDA_SRC, functions=["vd_run"],
                             extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _VD["build_s"] = time.time() - t1
    return _VD["mod"]


def _vd_pack(w, bm):
    n, k = w.shape
    nrb, nks = n // bm, k // 64
    t = w.contiguous().view(nrb, bm, nks, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bm, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7)).view(1, 1, bm, 8, 1).expand(nrb, nks, bm, 8, 8)
    t = torch.gather(t, 3, idx).permute(1, 0, 2, 3, 4).contiguous()
    back = torch.gather(t, 3, idx.permute(1, 0, 2, 3, 4)).permute(1, 2, 0, 3, 4).reshape(n, k)
    if not torch.equal(back.view(torch.int16), w.contiguous().view(torch.int16)):
        raise RuntimeError("VD pack is not bit-exact")
    return t.view(torch.uint8).view(-1), bm * 128, nrb * bm * 128


def _vd_call(st, r, x, out):
    flat, w_rb, w_s = st["w"][(r["proj"], r["bm"])]
    st["mod"].vd_run(r["kid"], x, flat, flat, out, st["err"], st["stamps"], r["nrb"], w_rb, w_s,
                     (128 if VD_TRIG else 0) | VD_FLAGS, 1 if VD_PDL else 0, VD_PF | (VD_ROT << 8))


def _vd_selftest(st, r, w, T, gen):
    f32 = r["f32"]
    dev = w.device
    ok, parts = True, []
    for t in sorted({T, T - 3, max(1, T // 2 + 1)}, reverse=True):
        x = torch.randn((t, r["K"]), generator=gen, device=dev).to(torch.bfloat16)
        outs = []
        for _ in range(2):
            out = torch.full((t, r["N"]), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
            _vd_call(st, r, x, out)
            outs.append(out)
        ref64 = x.double() @ w.double().t()
        today = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        torch.cuda.synchronize()
        got = outs[0]
        fin = bool(torch.isfinite(got).all())
        e64 = float((got.double() - ref64).norm() / ref64.norm()) if fin else float("inf")
        etoday = float((got.float() - today.float()).norm() / today.float().norm()) if fin else float("inf")
        same = bool(torch.equal(outs[0], outs[1]))
        errf = int(st["err"][0])
        good = fin and same and errf == 0 and e64 < VD_TOL[1 if f32 else 0] and etoday < VD_TOL[2]
        ok &= good
        parts.append(f"T={t}: rel vs fp64 {e64:.2e} vs today {etoday:.2e} finite {fin} repeat {same} err {errf}")
        st["err"].zero_()
    return ok, "; ".join(parts)


def _vd_tt_prepare(state, st):
    gen = torch.Generator(device=st["err"].device).manual_seed(77)
    for proj in VD_TT_PREP:
        w = _dg1_weight(state, proj)
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or proj in st["wt"]:
            continue
        f32, parts, good = proj == "cat", [], True
        try:
            wt = w.t().contiguous()
            for t in (128, 256):
                x = torch.randn((t, w.shape[1]), generator=gen, device=gen.device).to(torch.bfloat16)
                got = torch.mm(x, wt, out_dtype=torch.float32) if f32 else torch.mm(x, wt)
                ref = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
                e = float((got.float() - ref.float()).norm() / ref.float().norm())
                good = good and bool(torch.isfinite(got).all()) and e < VD_TOL[2]
                parts.append(f"T={t} rel vs today {e:.2e}")
        except Exception as ex:
            good, parts = False, [repr(ex)[:200]]
        print(f"[vd] torchT {proj}: {'; '.join(parts)} -> {'LIVE' if good else 'FAILED: today path'}", flush=True)
        if good:
            st["wt"][proj] = wt


def _vd_routes_prepare(state, st):
    gen = torch.Generator(device=st["err"].device).manual_seed(4242)
    for (proj, b), kid in sorted(VD_ROUTE.items(), key=lambda kv: kv[0]):
        if kid is None:
            continue
        nwg, bn, nst, cl, f32, rbd = VD_INSTS[kid]
        w = _dg1_weight(state, proj)
        if w is None or w.dtype != torch.bfloat16 or w.shape[0] % (64 * nwg) or w.shape[1] % 64:
            print(f"[vd] {proj}/{b}: no weight -> today's path", flush=True)
            continue
        bm = 64 * nwg
        try:
            if (proj, bm) not in st["w"]:
                st["w"][(proj, bm)] = _vd_pack(w, bm)
            r = {"kid": kid, "proj": proj, "bm": bm, "nrb": w.shape[0] // bm, "N": w.shape[0], "K": w.shape[1], "T": bn,
                 "f32": bool(f32)}
            good, msg = _vd_selftest(st, r, w, min(bn, b), gen)
        except Exception as ex:
            good, msg = False, repr(ex)[:300]
        print(f"[vd] {proj}/{b} inst {VD_INSTS[kid]}: {msg} -> {'LIVE' if good else 'FAILED: today path'}", flush=True)
        if good:
            st["route"][(proj, b)] = r


def _vd_prepare(state, t0):
    state["vd"] = None
    if not VD_ENABLE or not (VD_ROUTE or VD_TT_PREP):
        return None
    dev = state["dense"]["qkv_gate"].device
    t1 = time.time()
    st = {"mod": None, "err": torch.zeros((4,), dtype=torch.int32, device=dev),
          "stamps": torch.zeros((4,), dtype=torch.int64, device=dev), "w": {}, "route": {}, "wt": {}}
    _vd_tt_prepare(state, st)
    if VD_ROUTE and torch.cuda.get_device_capability(dev)[0] < 9:
        print("[vd] not sm_90: VD routes off", flush=True)
    elif VD_ROUTE:
        _pt1_wait("vd")
        try:
            st["mod"] = _vd_build()
        except Exception as ex:
            print(f"[vd] build FAILED -> VD routes on today's path: {ex!r}", flush=True)
        if st["mod"] is not None:
            _vd_routes_prepare(state, st)
    state["vd"] = st if (st["route"] or st["wt"]) else None
    print(f"[vd] routes live {sorted(st['route'])}; torchT live {sorted(st['wt'])} (build {_VD['build_s']:.0f}s, "
          f"prepare {time.time() - t1:.1f}s)", flush=True)
    return state["vd"]


def _vd_live(state, kind, proj, b=128):
    st = state.get("vd")
    return st is not None and (proj in st["wt"] if kind == "tt" else (proj, b) in st["route"])


def _vd(state, proj, x):
    st = state.get("vd")
    if not VD_ENABLE or st is None or x.dtype != torch.bfloat16 or x.dim() != 2:
        return None
    T = x.shape[0]
    b = _k2a_bucket_of(T)
    if VD_TT.get((proj, b)):
        wt = st["wt"].get(proj)
        if wt is not None and x.shape[1] == wt.shape[0]:
            return torch.mm(x, wt, out_dtype=torch.float32) if proj == "cat" else torch.mm(x, wt)
    if b not in VD_BUCKETS or proj not in VD_PROJS or VD_ROUTE.get((proj, b)) is None:
        return None
    r = st["route"].get((proj, b))
    if r is None or T > r["T"] or x.shape[1] != r["K"] or not x.is_contiguous() or x.data_ptr() % 16 != 0:
        return None
    try:
        out = torch.empty((T, r["N"]), dtype=torch.float32 if r["f32"] else torch.bfloat16, device=x.device)
        _vd_call(st, r, x, out)
        return out
    except Exception as ex:
        _k3_disable(state, "vd", None, "VD", ex)
        return None



def _dg1(state, proj, x):
    lt_out = _lt(state, proj, x)
    if lt_out is not None:
        return lt_out
    vd_out = _vd(state, proj, x)
    if vd_out is not None:
        return vd_out
    st = state.get("dg1")
    if not DG1_ENABLE or st is None:
        return None
    T = x.shape[0]
    if T <= 16:
        return None
    key = (proj, _k2a_bucket_of(T))
    if not DG1_ROUTE.get(key):
        return None
    r = st["route"].get(key)
    if (r is None or T > r["T"] or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != r["K"] or not x.is_contiguous()
            or x.data_ptr() % 16 != 0):
        return None
    try:
        out = torch.empty((T, r["N"]), dtype=torch.float32 if proj == "cat" else torch.bfloat16, device=x.device)
        st["mod"].dg1_run(r["kid"], x, st["w"][proj], out, st["err"], r["S"], 1 if DG1_PDL else 0,
                          1 if (DG1_TRIG and r["wave1"]) else 0, r["N"], 0)
        return out
    except Exception as ex:
        _k3_disable(state, "dg1", None, "DG1", ex)
        return None


def _dg1_selftest(st, r, proj, w, T, gen):
    f32 = proj == "cat"
    dev = w.device
    ok, parts = True, []
    for t in (T, T - 3):
        x = torch.randn((t, r["K"]), generator=gen, device=dev).to(torch.bfloat16)
        outs = []
        for _ in range(2):
            out = torch.full((t, r["N"]), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev)
            st["mod"].dg1_run(r["kid"], x, st["w"][proj], out, st["err"], r["S"], 1, 0, r["N"], 0)
            outs.append(out)
        ref64 = x.double() @ w.double().t()
        today = torch.mm(x, w.t(), out_dtype=torch.float32) if f32 else torch.mm(x, w.t())
        torch.cuda.synchronize()
        got = outs[0]
        fin = bool(torch.isfinite(got).all())
        e64 = float((got.double() - ref64).norm() / ref64.norm()) if fin else float('inf')
        etoday = float((got.float() - today.float()).norm() / today.float().norm()) if fin else float('inf')
        same = bool(torch.equal(outs[0], outs[1]))
        errf = int(st["err"][0])
        good = fin and same and errf == 0 and e64 < DG1_TOL[1 if f32 else 0] and etoday < DG1_TOL[2]
        ok &= good
        parts.append(f"T={t}: rel vs fp64 {e64:.2e} vs today {etoday:.2e} finite {fin} repeat {same} err {errf}")
        st["err"].zero_()
    return ok, "; ".join(parts)


def _dg1_prepare(state, t0):
    state["dg1"] = None
    keys = sorted(k for k in DG1_ROUTE if k in DG1_GEOMETRY)
    if not DG1_ENABLE or not keys:
        return None
    t1 = time.time()
    insts = tuple(sorted({DG1_GEOMETRY[k][:3] for k in keys}))
    try:
        ks = _KS1.get("mod")
        folded = ks is not None and _KS1.get("dg1_insts") == insts and hasattr(ks, "dg1_run")
        mod = ks if folded else _dg1_build(insts)
    except Exception as ex:
        print(f"[dg1] build FAILED -> today's path: {ex!r}", flush=True)
        return None
    dev = state["dense"]["qkv_gate"].device
    st = {"mod": mod, "w": {}, "route": {}, "err": torch.zeros(4, dtype=torch.int32, device=dev)}
    gen = torch.Generator(device=dev).manual_seed(4242)
    for key in keys:
        proj, rep = key
        T, nwg, mt, s = DG1_GEOMETRY[key]
        w = _dg1_weight(state, proj)
        if w is None:
            print(f"[dg1] {key}: no weight (v8 tail off) -> route off", flush=True)
            continue
        try:
            if proj not in st["w"]:
                st["w"][proj] = dg1_pack(w)
            kid = insts.index((T, nwg, mt))
            info = [int(v) for v in mod.dg1_info(kid, s)]
            n, k = w.shape
            if n % info[3] or rep > T or k % 64:
                print(f"[dg1] {key}: geometry {DG1_GEOMETRY[key]} does not fit W {tuple(w.shape)} -> route off", flush=True)
                continue
            r = {"kid": kid, "S": s, "T": T, "N": n, "K": k, "wave1": info[0] >= n // info[3]}
            ok, msg = _dg1_selftest(st, r, proj, w, rep, gen)
            print(f"[dg1] self-test {proj} bucket {rep} geo {DG1_GEOMETRY[key]} (BM {info[3]} ST {info[4]} regs {info[5]} clusters {info[0]}"
                  f"{' one wave' if r['wave1'] else ''}): {msg} -> {'ok' if ok else 'FAIL: route off'}", flush=True)
            if ok:
                st["route"][key] = r
        except Exception as ex:
            print(f"[dg1] {key}: self-test raised {ex!r} -> route off", flush=True)
    for proj in list(st["w"]):
        if not any(k[0] == proj for k in st["route"]):
            del st["w"][proj]
    state["dg1"] = st if st["route"] else None
    print(f"[dg1] routes live {sorted(st['route'])} (build {_DG1['build_s']:.1f}s, total {time.time() - t1:.1f}s; prepare so far"
          f" {time.time() - t0:.1f}s)", flush=True)
    return state["dg1"]


SU1_ENABLE = True
SU1_ROUTE = {("unpooler", 1): True, ("unpooler", 8): False, ("o", 1): False, ("o", 8): False, ("o", 16): False, ("o", 32): False}
SU1_GEOM = {"unpooler": (4096, 8, 1024), "o": (1024, 64, 8192)}
SU1_K3_TRIG = True
SU1_TRIG = 2
SU1_LATE = {}
SU1_PDL = True
SU1_SMEM = {"unpooler": 120 * 1024}
SU1_PROF = None
SU1_TOL = (4e-3, 1e-2)
SU1_PREPARE_BY_S = 250.0
SU1_SELF_TEST_T = {"unpooler": (1, 2, 5, 8), "o": (1, 8)}
SU1_MAX_T = {"unpooler": 8, "o": 32}

SU1_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void su1_run(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t T, int64_t K, int64_t R,
             int64_t ldo, int64_t pdl, int64_t trig, torch::Tensor prof, int64_t smem);
std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T);
"""

SU1_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define SU1_ARCH_OK 1
#else
#define SU1_ARCH_OK 0
#endif

namespace su1 {
constexpr int NTHR = 256;   // 8 warps
constexpr int CH = 16384;   // bytes per cp.async.bulk

__device__ __forceinline__ uint32_t sm_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}

template <int K, int R, int NT>
__global__ void __launch_bounds__(NTHR) su1_kernel(const uint8_t* __restrict__ wp, const __nv_bfloat16* __restrict__ x, int T,
                                                   __nv_bfloat16* __restrict__ out, long ldo, int* __restrict__ err, int trig,
                                                   long long* __restrict__ prof) {
#if SU1_ARCH_OK
  if (prof != nullptr && threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
    prof[8 * blockIdx.x + 0] = gtimer();
    prof[8 * blockIdx.x + 4] = (long long)smid;
  }
  constexpr int KT = K / 16;                  // k-tiles
  constexpr int KTW = KT / 8;                 // k-tiles per warp
  constexpr int KP = K / 32;                  // k-pairs
  constexpr int MT = (R >= 16) ? R / 16 : 1;  // m-tiles
  constexpr int NTOK = 8 * NT;                // token columns (n8 tiles)
  constexpr int BYTES = R * K * 2;
  static_assert(BYTES % CH == 0 && KTW % 2 == 0 && (R == 8 || R % 16 == 0) && 8 * R * NTOK * 4 <= BYTES, "su1 geometry");
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) unsigned long long bar;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sb = sm_u32(&bar);
  const int tmode = trig & 3;           // dependents: 0 implicit at exit, 1 right after the wait, 2 at entry
  const bool late = (trig >> 4) & 1;    // lab: stage the weights only after the wait (no overlap with the predecessor)
  if (tmode == 2) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  const uint8_t* src = wp + (size_t)blockIdx.x * BYTES;
  if (tid == 0) {   // prepare-time data only before the PDL wait: this CTA's weight block -> smem
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(sb) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sb), "r"(BYTES) : "memory");
    if (!late) {
#pragma unroll 1
      for (int o = 0; o < BYTES; o += CH)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(sm_u32(smem + o)), "l"((unsigned long long)(src + o)), "r"(CH), "r"(sb) : "memory");
    }
  }
  __syncthreads();   // the mbarrier is initialized before any thread waits on it
  asm volatile("griddepcontrol.wait;\n" ::: "memory");   // x (the predecessor's output) is visible from here on
  if (tmode == 1) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (late && tid == 0) {
#pragma unroll 1
    for (int o = 0; o < BYTES; o += CH)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(sm_u32(smem + o)), "l"((unsigned long long)(src + o)), "r"(CH), "r"(sb) : "memory");
  }
  if (prof != nullptr && tid == 0) prof[8 * blockIdx.x + 1] = gtimer();
  const int g = lane >> 2, t4 = lane & 3;
  uint32_t b[KTW][NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int tok = 8 * nt + g;
    const bool live = tok < T;
    const unsigned int* xr = reinterpret_cast<const unsigned int*>(x + (size_t)(live ? tok : 0) * K + 16 * warp * KTW + 2 * t4);
#pragma unroll
    for (int i = 0; i < KTW; ++i) {
      b[i][nt][0] = live ? __ldcg(xr + 8 * i) : 0u;
      b[i][nt][1] = live ? __ldcg(xr + 8 * i + 4) : 0u;
    }
  }
  {   // the staged weights: bounded wait (flag on timeout, never a trap)
    uint32_t done = 0;
#pragma unroll 1
    for (int it = 0; it < (1 << 20) && !done; ++it)
      asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                   : "=r"(done) : "r"(sb), "r"(0u) : "memory");
    if (!done && lane == 0) atomicOr(err, 1);
  }
  if (prof != nullptr && tid == 0) prof[8 * blockIdx.x + 2] = gtimer();
  float c[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) c[m][nt][0] = c[m][nt][1] = c[m][nt][2] = c[m][nt][3] = 0.f;
  const uint4* ws = reinterpret_cast<const uint4*>(smem);
  const int kp0 = warp * (KTW / 2);
#pragma unroll
  for (int i = 0; i < KTW / 2; ++i) {
    if constexpr (R == 8) {
      const uint4 a = ws[(kp0 + i) * 32 + lane];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        mma16816(c[0][nt], a.x, 0u, a.y, 0u, b[2 * i][nt][0], b[2 * i][nt][1]);
        mma16816(c[0][nt], a.z, 0u, a.w, 0u, b[2 * i + 1][nt][0], b[2 * i + 1][nt][1]);
      }
    } else {
#pragma unroll
      for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint4 a = ws[((m * KP + kp0 + i) * 2 + h) * 32 + lane];
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) mma16816(c[m][nt], a.x, a.y, a.z, a.w, b[2 * i + h][nt][0], b[2 * i + h][nt][1]);
        }
      }
    }
  }
  __syncthreads();   // every warp is done with the staged weights: the block becomes the [8 warps][R][NTOK] reduction scratch
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      float* q = red + (warp * R + 16 * m + g) * NTOK + 8 * nt + 2 * t4;
      q[0] = c[m][nt][0];
      q[1] = c[m][nt][1];
      if constexpr (R >= 16) {
        q[8 * NTOK] = c[m][nt][2];
        q[8 * NTOK + 1] = c[m][nt][3];
      }
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int o = tid; o < R * NTOK; o += NTHR) {
    const int r = o / NTOK, n = o % NTOK;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w * R * NTOK + o];   // fixed order: deterministic
    if (n < T) out[(size_t)n * ldo + (size_t)R * blockIdx.x + r] = __float2bfloat16(s);
  }
  if (prof != nullptr) {
    __syncthreads();
    if (tid == 0) prof[8 * blockIdx.x + 3] = gtimer();
  }
#endif
}

template <int K, int R, int NT>
static void launch(const uint8_t* wp, const __nv_bfloat16* x, int T, __nv_bfloat16* out, long ldo, int* err, int ncta, bool pdl,
                   int trig, long long* prof, int smem, cudaStream_t st) {
  constexpr int BYTES = R * K * 2;
  const int dyn = smem > BYTES ? smem : BYTES;
  static int attr = 0;
  if (attr < dyn) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(su1_kernel<K, R, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
    attr = dyn;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ncta, 1, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = dyn;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, su1_kernel<K, R, NT>, wp, x, T, out, ldo, err, trig, prof));
}

template <int K, int R, int NT>
static std::vector<int64_t> info_t() {
  constexpr int BYTES = R * K * 2;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, su1_kernel<K, R, NT>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(su1_kernel<K, R, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, su1_kernel<K, R, NT>, NTHR, BYTES));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + BYTES), (int64_t)occ};
}
}  // namespace su1

static int su1_nt(int64_t T) { return T <= 8 ? 1 : (T <= 16 ? 2 : 4); }

void su1_run(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t T, int64_t K, int64_t R,
             int64_t ldo, int64_t pdl, int64_t trig, torch::Tensor prof, int64_t smem) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous() && x.size(1) == K, "su1: x");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1, "su1: out");
  TORCH_CHECK(wp.is_contiguous() && err.scalar_type() == at::kInt, "su1: wp / err");
  TORCH_CHECK(T >= 1 && T <= 32 && x.size(0) >= T, "su1: 1 <= T <= 32");
  TORCH_CHECK(((uintptr_t)wp.data_ptr() % 16) == 0 && ((uintptr_t)x.data_ptr() % 16) == 0, "su1: 16 B alignment");
  const int64_t blk = R * K * 2;
  const int64_t nbytes = wp.numel() * wp.element_size();
  TORCH_CHECK(nbytes % blk == 0, "su1: wp size");
  const int ncta = (int)(nbytes / blk);
  TORCH_CHECK(out.size(0) >= T && out.size(1) >= (int64_t)ncta * R && ldo >= (int64_t)ncta * R, "su1: out shape");
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* w = reinterpret_cast<const uint8_t*>(wp.data_ptr());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  int* ep = err.data_ptr<int>();
  long long* pp = prof.numel() >= 8 * (int64_t)ncta ? reinterpret_cast<long long*>(prof.data_ptr()) : nullptr;
  const int nt = su1_nt(T);
  if (K == 4096 && R == 8 && nt == 1) su1::launch<4096, 8, 1>(w, xp, (int)T, op, (long)ldo, ep, ncta, pdl != 0, (int)trig, pp, (int)smem, st);
  else if (K == 1024 && R == 64 && nt == 1) su1::launch<1024, 64, 1>(w, xp, (int)T, op, (long)ldo, ep, ncta, pdl != 0, (int)trig, pp, (int)smem, st);
  else if (K == 1024 && R == 64 && nt == 2) su1::launch<1024, 64, 2>(w, xp, (int)T, op, (long)ldo, ep, ncta, pdl != 0, (int)trig, pp, (int)smem, st);
  else if (K == 1024 && R == 64 && nt == 4) su1::launch<1024, 64, 4>(w, xp, (int)T, op, (long)ldo, ep, ncta, pdl != 0, (int)trig, pp, (int)smem, st);
  else TORCH_CHECK(false, "su1: (K, R, T) not built");
}

std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T) {
  const int nt = su1_nt(T);
  if (K == 4096 && R == 8 && nt == 1) return su1::info_t<4096, 8, 1>();
  if (K == 1024 && R == 64 && nt == 1) return su1::info_t<1024, 64, 1>();
  if (K == 1024 && R == 64 && nt == 2) return su1::info_t<1024, 64, 2>();
  if (K == 1024 && R == 64 && nt == 4) return su1::info_t<1024, 64, 4>();
  TORCH_CHECK(false, "su1: (K, R, T) not built");
  return {};
}
"""

_SU1_FLAG = {"k3trig": False}
_SU1_SEEN = set()


_SU1 = {"mod": None, "build_s": 0.0}


def _su1_build():
    _pt1_wait("su1")
    if _SU1["mod"] is not None:
        return _SU1["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((SU1_CPP_SRC + SU1_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    _SU1["mod"] = load_inline(name=f"su1_{tag}", cpp_sources=SU1_CPP_SRC, cuda_sources=SU1_CUDA_SRC, functions=["su1_run", "su1_info"],
                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _SU1["build_s"] = time.time() - t0
    return _SU1["mod"]


def su1_pack(w, R):
    n, k = w.shape
    c, kp = n // R, k // 32
    w = w.contiguous()
    if R == 8:
        p = w.view(c, 8, kp, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6)
    else:
        p = w.view(c, R // 16, 2, 8, kp, 2, 2, 4, 2).permute(0, 1, 4, 5, 3, 7, 6, 2, 8)
    return p.contiguous().view(-1)


def _su1_weight(state, proj):
    K, R, rows = SU1_GEOM[proj]
    w = state["dense"]["unpooler" if proj == "unpooler" else "o"]
    return w[:rows] if proj == "unpooler" else w


def _su1_live(state, proj, tokens):
    st = state.get("su1")
    return bool(SU1_ENABLE and st is not None and proj in st["w"] and 1 <= tokens <= SU1_MAX_T[proj]
                and SU1_ROUTE.get((proj, _k2a_bucket_of(tokens))))


def _su1(state, proj, x, tokens):
    if not _su1_live(state, proj, tokens):
        return None
    st = state["su1"]
    K, R, rows = SU1_GEOM[proj]
    if not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[0] == tokens and x.shape[1] == K and x.is_contiguous()
            and x.data_ptr() % 16 == 0):
        return None
    try:
        if proj == "unpooler":
            zb = state.get("v12_zbuf")
            if not (V12_ZBUF and zb is not None and R1_UNPOOL_INTO_ZEROS and tokens <= SLICED_UNPOOL_MAX_TOKENS):
                return None
            out = zb[:tokens]
        else:
            out = torch.empty((tokens, rows), dtype=torch.bfloat16, device=x.device)
        st["mod"].su1_run(st["w"][proj], x, out, st["err"], tokens, K, R, HIDDEN, 1 if SU1_PDL else 0,
                          int(SU1_TRIG) | (16 if SU1_LATE.get(proj) else 0),
                          SU1_PROF if SU1_PROF is not None else st["noprof"], int(SU1_SMEM.get(proj, 0)))
        if (proj, tokens) not in _SU1_SEEN:
            _SU1_SEEN.add((proj, tokens))
            print(f"[su1] first call {proj} T={tokens} (k3trig {_SU1_FLAG['k3trig'] or SU1_K3_TRIG})", flush=True)
        return out
    except Exception as ex:
        _k3_disable(state, "su1", None, "SU1", ex)
        return None


def _su1_selftest(st, proj, w, gen):
    K, R, rows = SU1_GEOM[proj]
    dev = w.device
    ok, parts = True, []
    for t in SU1_SELF_TEST_T[proj]:
        x = torch.randn((t, K), generator=gen, device=dev).to(torch.bfloat16)
        outs = []
        for _ in range(2):
            out = torch.full((t, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
            st["mod"].su1_run(st["w"][proj], x, out, st["err"], t, K, R, HIDDEN, 1, 1, st["noprof"], 0)
            outs.append(out)
        ref64 = x.double() @ w.double().t()
        today = torch.mm(x, w.t())
        torch.cuda.synchronize()
        got = outs[0][:, :rows]
        fin = bool(torch.isfinite(got).all())
        rest = bool(torch.isnan(outs[0][:, rows:]).all()) if rows < HIDDEN else True
        e64 = float((got.double() - ref64).norm() / ref64.norm()) if fin else float("inf")
        etd = float((got.float() - today.float()).norm() / today.float().norm()) if fin else float("inf")
        same = bool(torch.equal(outs[0][:, :rows], outs[1][:, :rows]))
        errf = int(st["err"][0])
        good = fin and rest and same and errf == 0 and e64 < SU1_TOL[0] and etd < SU1_TOL[1]
        ok &= good
        parts.append(f"T={t}: fp64 {e64:.2e} mm {etd:.2e} fin {fin} cols {rest} rep {same} err {errf}")
        st["err"].zero_()
    return ok, "; ".join(parts)


SU1_K3_GRID = True
SU1_K3_TRIG_AT = {1: "su1", 8: "p1"}


def _su1_k3trig(state, tokens):
    g = _gd_k3trig(state, tokens)
    if g is not None:
        return g
    if not (SU1_ENABLE and SU1_K3_TRIG):
        return False
    rep = _k2a_bucket_of(tokens)
    mode = SU1_K3_TRIG_AT.get(rep)
    if mode == "su1":
        return _su1_live(state, "unpooler", tokens)
    if mode == "p1m":
        st = state.get("p1m")
        return bool(V20_P1M_ENABLE and V20_P1M_PDL and st is not None and ("unpooler", rep) in V20_P1M_ROUTE
                    and st["route"].get(("unpooler", rep)) is not None and 8 < tokens <= 32)
    if mode == "p1":
        return bool(1 < tokens <= 8 and not _su1_live(state, "unpooler", tokens) and P1_ENABLE and P1_PROJ.get("unpooler")
                    and P1_ROUTE.get(("unpooler", 8)) and _G2_LIVE.get("p1pdl") and G2_P1_PDL.get("unpooler"))
    return False


def _su1_g3(st, m):
    return max(4 * m, 1) if (SU1_K3_GRID and _SU1_FLAG["k3trig"]) else st["sms"] * 2


def _su1_prepare(state, t0):
    state["su1"] = None
    if not SU1_ENABLE:
        return None
    if time.time() - t0 > SU1_PREPARE_BY_S:
        print(f"[su1] skipped: prepare already took {time.time() - t0:.1f}s (> {SU1_PREPARE_BY_S}s): today's paths", flush=True)
        return None
    t1 = time.time()
    try:
        ks = _su1_build()
    except Exception as ex:
        print(f"[su1] build FAILED -> today's paths: {repr(ex)[-3000:]}", flush=True)
        return None
    dev = state["dense"]["o"].device
    st = {"mod": ks, "w": {}, "err": torch.zeros(4, dtype=torch.int32, device=dev), "noprof": torch.empty((0,), dtype=torch.int64, device=dev)}
    gen = torch.Generator(device=dev).manual_seed(5151)
    for proj in sorted({p for p, _ in SU1_ROUTE}):
        K, R, rows = SU1_GEOM[proj]
        try:
            w = _su1_weight(state, proj)
            nts = sorted({1 if t <= 8 else (9 if t <= 16 else 17) for t in SU1_SELF_TEST_T[proj]})
            info = ["/".join(str(int(ks.su1_info(K, R, t)[j])) for t in nts) for j in range(4)]
            st["w"][proj] = su1_pack(w, R)
            ok, msg = _su1_selftest(st, proj, w, gen)
            print(f"[su1] self-test {proj} (K {K} R {R}: {info[0]} regs, {info[1]} B local, {info[2]} B smem, {info[3]} CTAs/SM): {msg}"
                  f" -> {'ok' if ok else 'FAIL: today path'}", flush=True)
            if not ok:
                del st["w"][proj]
        except Exception as ex:
            print(f"[su1] {proj}: self-test raised {ex!r} -> today's path", flush=True)
            st["w"].pop(proj, None)
    state["su1"] = st if st["w"] else None
    print(f"[su1] live {sorted(st['w'])} route {SU1_ROUTE} k3trig {SU1_K3_TRIG} (build {_SU1['build_s']:.1f}s, total "
          f"{time.time() - t1:.1f}s; prepare so far "
          f"{time.time() - t0:.1f}s)", flush=True)
    return state["su1"]


def _k3_disable(state, key, value, piece, ex):
    state[key] = value
    if piece not in _K3_WARNED:
        _K3_WARNED.add(piece)
        print(f"[k3] {piece} raised at call time, r1 path from now on: {ex!r}", flush=True)



TL1_TAIL2 = True



K3P_K3B = {1: True, 8: True, 16: True, 32: True, 128: True, 256: True}
K3P_K3_GRID = {128: True, 256: True}


def _k3p_bits(m):
    return (1 << 31) if K3P_K3B.get(_k2a_bucket_of(m)) else 0


def _k3p_g3(st, m, default):
    return max(4 * m, 1) if K3P_K3_GRID.get(_k2a_bucket_of(m)) else default



RS1_ENABLE = True


SBB1_CU = r'''
// arithmetic, but every input that does not depend on the predecessor is loaded before griddepcontrol.wait.
#include <cuda_bf16.h>
#include <stdint.h>

#define SB_H 8192
#define SB_CS 16
#define SB_NT (SB_H / (SB_CS * 8))   // 64 threads per CTA

struct SbbParams {
  const __nv_bfloat16* branch;   // [T][H] bf16
  const float* stream;           // [T][H] fp32
  __nv_bfloat16* out;            // [T][H] bf16 (pre-normed)
  float* new_stream;             // [T][H] fp32
  float* router;                 // [T][H] fp32 or nullptr
  const __nv_bfloat16* post_w;   // [H] bf16 or nullptr
  const __nv_bfloat16* pre_w;    // [H] bf16 or nullptr
  const float* alpha;            // [H] or [1] fp32, or nullptr (no residual gate)
  const float* beta;
  float post_eps, pre_eps;
  int gate_channel_wise;
  int trig_early;   // LAT: launch_dependents at entry
};

__device__ __forceinline__ float sb_rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void sb_ld8bf(const __nv_bfloat16* p, float (&v)[8]) {
  const uint4 q = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) { v[2 * i] = __uint_as_float(w[i] << 16); v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u); }
}
__device__ __forceinline__ void sb_ld8f(const float* p, float (&v)[8]) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void sb_st8f(float* p, const float (&v)[8]) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}

template <bool kPreStream, bool kRouter, bool kInputScale>
__global__ void __launch_bounds__(SB_NT) sbb_kernel(const SbbParams p) {
  __shared__ __align__(16) float partials[SB_CS][4];
  __shared__ float warp_red[SB_NT / 32][4];
  __shared__ __align__(8) uint64_t bar;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (p.trig_early) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int row = blockIdx.x / SB_CS;
  const int e0 = (int)(rank * SB_NT + tid) * 8;
  const uint32_t bar_a = (uint32_t)__cvta_generic_to_shared(&bar);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar_a));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar_a), "r"((SB_CS - 1) * 16));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  float pw[8], qw[8], al[8], bt[8], st[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { pw[i] = 1.f; qw[i] = 1.f; al[i] = 1.f; bt[i] = 1.f; }
  if (p.post_w) sb_ld8bf(p.post_w + e0, pw);
  if (p.pre_w) sb_ld8bf(p.pre_w + e0, qw);
  if (p.alpha) {
    if (p.gate_channel_wise) { sb_ld8f(p.alpha + e0, al); sb_ld8f(p.beta + e0, bt); }
    else {
#pragma unroll
      for (int i = 0; i < 8; ++i) { al[i] = p.alpha[0]; bt[i] = p.beta[0]; }
    }
  }
  if (kPreStream) sb_ld8f(p.stream + (size_t)row * SB_H + e0, st);
  else if ((tid & 3) == 0) asm volatile("prefetch.global.L2 [%0];" :: "l"(p.stream + (size_t)row * SB_H + e0));
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float br[8];
  sb_ld8bf(p.branch + (size_t)row * SB_H + e0, br);
  if (!kPreStream) sb_ld8f(p.stream + (size_t)row * SB_H + e0, st);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  float val[8], res[8];
  float bsum = 0.f, rsum = 0.f, csum = 0.f, ssum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float v = br[i];
    if (p.post_w && kInputScale) v = sb_rbf(v * pw[i]);
    float mv = v;
    if (p.post_w && !kInputScale) mv *= pw[i];
    const float r = al[i] * st[i];
    const float sb = bt[i] * mv;
    val[i] = v;
    res[i] = r;
    bsum = fmaf(v, v, bsum);
    rsum = fmaf(r, r, rsum);
    csum = fmaf(r, sb, csum);
    ssum = fmaf(sb, sb, ssum);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    bsum += __shfl_xor_sync(0xffffffffu, bsum, o);
    rsum += __shfl_xor_sync(0xffffffffu, rsum, o);
    csum += __shfl_xor_sync(0xffffffffu, csum, o);
    ssum += __shfl_xor_sync(0xffffffffu, ssum, o);
  }
  if (lane == 0) { warp_red[warp][0] = bsum; warp_red[warp][1] = rsum; warp_red[warp][2] = csum; warp_red[warp][3] = ssum; }
  __syncthreads();
  float4 mine = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int w = 0; w < SB_NT / 32; ++w) {
    mine.x += warp_red[w][0]; mine.y += warp_red[w][1]; mine.z += warp_red[w][2]; mine.w += warp_red[w][3];
  }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (tid < SB_CS) {
    if (tid == (int)rank) {
      *reinterpret_cast<float4*>(&partials[rank][0]) = mine;
    } else {
      const uint32_t dst_l = (uint32_t)__cvta_generic_to_shared(&partials[rank][0]);
      uint32_t dst, rbar;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dst) : "r"(dst_l), "r"(tid));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(bar_a), "r"(tid));
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
                   ::"r"(dst), "f"(mine.x), "f"(mine.y), "f"(mine.z), "f"(mine.w), "r"(rbar) : "memory");
    }
  }
  {
    uint32_t done = 0;
    while (!done)
      asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                   " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bar_a) : "memory");
  }
  __syncthreads();
  float tb = 0.f, tr = 0.f, tc = 0.f, ts = 0.f;
#pragma unroll
  for (int q = 0; q < SB_CS; ++q) { tb += partials[q][0]; tr += partials[q][1]; tc += partials[q][2]; ts += partials[q][3]; }
  const float post_rstd = rsqrtf(tb * (1.f / (float)SB_H) + p.post_eps);
  const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, ts, 2.f * tc), tr);
  const float pre_rstd = rsqrtf(fmaxf(pre_sum, 0.f) * (1.f / (float)SB_H) + p.pre_eps);
  float ns[8], nz[8];
  __align__(16) __nv_bfloat16 ob[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float n = val[i] * post_rstd;
    if (p.post_w && !kInputScale) n *= pw[i];
    n = sb_rbf(n);
    const float u = fmaf(bt[i], n, res[i]);
    ns[i] = u;
    float z = u * pre_rstd;
    if (p.pre_w) z *= qw[i];
    nz[i] = z;
    ob[i] = __float2bfloat16_rn(z);
  }
  *reinterpret_cast<uint4*>(p.out + (size_t)row * SB_H + e0) = *reinterpret_cast<const uint4*>(ob);
  sb_st8f(p.new_stream + (size_t)row * SB_H + e0, ns);
  if (kRouter) sb_st8f(p.router + (size_t)row * SB_H + e0, nz);
}
'''
SBB1_HOST = r'''

// one CTA fits beside two GD pooler CTAs of an SM. Warp w of CTA (g, s) handles rows 16 g + w and 16 g + 8 + w.
template <int MT>
__global__ void __launch_bounds__(256, 6) vf_router32_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                             float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024;
  const int sl = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = sl * KS;
#pragma unroll 1
  for (int h = 0; h < 2; ++h) {
    const int row = blockIdx.x * 16 + h * 8 + warp;
    const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
    float acc[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
      float4 wv[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) wv[i] = __ldcs(wp + (hh * 2 + i) * 32 + lane);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int c = (hh * 2 + i) * 32 + lane;
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          if (j < m) {
            const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (long)j * K + k0) + c);
            acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
            acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
            acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
            acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      float a = acc[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0 && j < m) part[((long)sl * m + j) * 256 + row] = a;
    }
  }
}

// x through L1, <= 80 regs (10,240 per CTA: fits beside two GD pooler CTAs), grid (64, 8). Same arithmetic: bitwise.
template <int MT>
__global__ void __launch_bounds__(128, 6) vf_router32b_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                              float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024;
  const int sl = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = sl * KS, row = blockIdx.x * 4 + warp;
  const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
  float4 wv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + i * 32 + lane);
  float acc[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = i * 32 + lane;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      if (j < m) {
        const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (long)j * K + k0) + c);
        acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
        acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
        acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
        acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    float a = acc[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0 && j < m) part[((long)sl * m + j) * 256 + row] = a;
  }
}

template <int MT>
__global__ void __launch_bounds__(128, 6) vf_router32c_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                              float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024;
  const int sl = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = sl * KS;
#pragma unroll
  for (int q = 1; q < 4; ++q) {
    const float* pl = w + (long)(blockIdx.x * 16 + q * 4 + warp) * K + k0 + lane * 32;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(pl));
  }
#pragma unroll 1
  for (int q = 0; q < 4; ++q) {
    const int row = blockIdx.x * 16 + q * 4 + warp;
    const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
    float4 wv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + i * 32 + lane);
    float acc[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = i * 32 + lane;
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        if (j < m) {
          const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (long)j * K + k0) + c);
          acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
          acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
          acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
          acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      float a = acc[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0 && j < m) part[((long)sl * m + j) * 256 + row] = a;
    }
  }
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
template <bool A, bool B, bool C>
static void sbb_go(const SbbParams& p, int T, cudaStream_t s) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(sbb_kernel<A, B, C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    init = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(T * SB_CS); cfg.blockDim = dim3(SB_NT); cfg.dynamicSmemBytes = 0; cfg.stream = s;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = SB_CS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 2;
  cudaLaunchKernelEx(&cfg, sbb_kernel<A, B, C>, p);
}
void sbb1_run(torch::Tensor branch, torch::Tensor stream, torch::Tensor out, torch::Tensor new_stream, torch::Tensor router,
              torch::Tensor post_w, torch::Tensor pre_w, torch::Tensor alpha, torch::Tensor beta, double post_eps,
              double pre_eps, int64_t input_scale) {
  const int T = branch.size(0);
  TORCH_CHECK(branch.size(1) == SB_H && stream.size(1) == SB_H && branch.is_contiguous() && stream.is_contiguous(), "sbb1: layout");
  SbbParams p = {};
  p.branch = (const __nv_bfloat16*)branch.data_ptr(); p.stream = stream.data_ptr<float>();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.new_stream = new_stream.data_ptr<float>();
  p.router = router.numel() ? router.data_ptr<float>() : nullptr;
  p.post_w = post_w.numel() ? (const __nv_bfloat16*)post_w.data_ptr() : nullptr;
  p.pre_w = pre_w.numel() ? (const __nv_bfloat16*)pre_w.data_ptr() : nullptr;
  p.alpha = alpha.numel() ? alpha.data_ptr<float>() : nullptr;
  p.beta = beta.numel() ? beta.data_ptr<float>() : nullptr;
  p.gate_channel_wise = alpha.numel() == SB_H ? 1 : 0;
  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;
  p.trig_early = (input_scale & 8) ? 1 : 0;
  auto s = c10::cuda::getCurrentCUDAStream();
  const bool r = p.router != nullptr, post = (input_scale & 2) != 0, isc = (input_scale & 1) != 0;
  TORCH_CHECK(!(post && r), "sbb1: the post-wait stream mode has no router output");
  if (post && isc) sbb_go<false, false, true>(p, T, s);
  else if (post) sbb_go<false, false, false>(p, T, s);
  else if (r && isc) sbb_go<true, true, true>(p, T, s);
  else if (r) sbb_go<true, true, false>(p, T, s);
  else if (isc) sbb_go<true, false, true>(p, T, s);
  else sbb_go<true, false, false>(p, T, s);
}

void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind) {
  const int m = (int)x.size(0);
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && part.scalar_type() == at::kFloat &&
              x.is_contiguous() && w.is_contiguous() && part.is_contiguous() && x.size(1) == 8192 && w.size(0) == 256 &&
              w.size(1) == 8192 && m >= 1 && m <= 8 && part.numel() == 8L * m * 256 &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "vf_router32: x f32 [m <= 8, 8192], w f32 [256, 8192], part [8, m, 256]");
  auto s = c10::cuda::getCurrentCUDAStream();
  if (kind == 2)
    vf_router32c_kernel<8><<<dim3(16, 8), 128, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else if (kind == 1)
    vf_router32b_kernel<8><<<dim3(64, 8), 128, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else if (m == 1)
    vf_router32_kernel<1><<<dim3(16, 8), 256, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else
    vf_router32_kernel<8><<<dim3(16, 8), 256, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "vf_router32: launch failed");
}
'''
SBB1_DECL = 'void sbb1_run(torch::Tensor branch, torch::Tensor stream, torch::Tensor out, torch::Tensor new_stream, torch::Tensor router, torch::Tensor post_w, torch::Tensor pre_w, torch::Tensor alpha, torch::Tensor beta, double post_eps, double pre_eps, int64_t input_scale); void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind);'


VF_ROUTER = {1: 'lean', 8: 'r32'}
VF_PRT = False
VF_SBB2 = {1: True, 8: True, 16: False}
VF_T1 = True
VF_SELF_TEST_SEEDS = (731, 732, 733)
_VF = {"r8": False, "prt": False, "st2": None, "seen": set()}


def _vf_sbb2(stream, branch, tokens):
    st = _VF["st2"]
    if st is None or not st["ok"].get(_k2a_bucket_of(tokens)) or branch.dtype != torch.bfloat16 or stream.dtype != torch.float32:
        return None
    dev = branch.device
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    ns = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    pw, qw, al, bt, isc, pe, qe = st["w"]
    st["mod"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, st["e"], pw, qw, al, bt, pe, qe, isc | 2)
    if ("sbb2", tokens) not in _VF["seen"]:
        _VF["seen"].add(("sbb2", tokens))
        print(f"[vf] first call sbb2 T={tokens}", flush=True)
    return out, ns, None


def _vf_t1_live(state):
    if not VF_T1:
        return False
    ok = (VF_ROUTER.get(1) == "lean" and _VF["r8"] and _SBB1["mod"] is not None and state.get("v8") is not None
          and _gd_live(state, "pooler", 1) is not None)
    if not ok and "t1off" not in _VF["seen"]:
        _VF["seen"].add("t1off")
        print("[vf] FALLBACK t1: lean router / GD pooler T=1 not live -> G2 prt at T = 1", flush=True)
    return ok


def _vf_test_r8(state, mod):
    dev = state["router"].device
    ok = True
    for m, seed in ((1, 731), (3, 732), (8, 733), (8, 734)):
        gen = torch.Generator(device=dev).manual_seed(seed)
        z = torch.randn((m, HIDDEN), generator=gen, device=dev)
        z = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).contiguous()
        ref = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        got = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        state["v8"]["mod"].v8_router32(z, state["router"], ref)
        for kind in ((0,) if m == 1 else (0, 1, 2)):
            got.fill_(float("nan"))
            mod.vf_router32(z, state["router"], got, kind)
            torch.cuda.synchronize()
            eq = bool(torch.equal(ref, got))
            print(f"[vf] r8 self-test m={m} kind {kind}: router partials bitwise {eq}", flush=True)
            ok &= eq
    return ok


def _vf_test_prt(state, g2):
    P = state["p1"]["proj"]["pooler"]
    dev = state["router"].device
    ok = True
    for T, ls, seed in G2_SELF_TEST_CASES[:3]:
        gen = torch.Generator(device=dev).manual_seed(seed + 7)
        h = torch.randn((T, HIDDEN), generator=gen, device=dev)
        h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
        hb = h.to(torch.bfloat16).contiguous()
        d = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
        a = _g2_prt_run(g2, P, hb, h, state, d, 1)
        b = _g2_prt_run(g2, P, hb, h, state, d, 3)
        c = _g2_prt_run(g2, P, hb, h, state, d, 2)
        torch.cuda.synchronize()
        eq = all(bool(torch.equal(x, y)) for x, y in zip(a, b)) and all(bool(torch.equal(x, y)) for x, y in zip(a, c))
        cnt = int(g2["tcnt"][0])
        print(f"[vf] prt tail self-test T={T} ls={ls}: xn / w / ids / lids bitwise {eq}, ticket {cnt}", flush=True)
        ok &= eq and cnt == 0
    return ok


def _vf_prepare(state, t0):
    t1 = time.time()
    st1 = _SBB1["st"]
    mod = _SBB1["mod"]
    _VF["r8"] = False
    _VF["prt"] = False
    _VF["st2"] = None
    if mod is None or st1 is None:
        print("[vf] SBB1 module missing -> VF riders off", flush=True)
        return
    try:
        if state.get("v8") is not None and state.get("router") is not None:
            _VF["r8"] = _vf_test_r8(state, mod)
    except Exception as ex:
        print(f"[vf] r8 FAILED -> router32 stays: {ex!r}", flush=True)
        _VF["r8"] = False
    try:
        g2 = state.get("g2")
        if g2 is not None and g2.get("prt"):
            _VF["prt"] = _vf_test_prt(state, g2)
    except Exception as ex:
        print(f"[vf] prt tail FAILED -> G2 tail stays: {ex!r}", flush=True)
        _VF["prt"] = False
    try:
        op = state["ffn_boundary"]
        op._ensure_cached_weights(torch.bfloat16)
        dev = state["router"].device
        e = torch.empty(0, device=dev)
        w = (op._cc_post_weight.to(torch.bfloat16).contiguous() if op._post_norm.has_weight else e,
             op._cc_pre_weight.to(torch.bfloat16).contiguous() if op._pre_norm.has_weight else e,
             op._cc_alpha.float().contiguous() if op._use_residual_gate else e,
             op._cc_beta.float().contiguous() if op._use_residual_gate else e,
             1 if op._post_norm_is_input_scale else 0, float(op._post_norm.eps), float(op._pre_norm.eps))
        st2 = {"mod": mod, "w": w, "e": e, "ok": {}}
        gen = torch.Generator(device=dev).manual_seed(5254)
        for rep in sorted(VF_SBB2):
            lo, hi = [(a, b) for a, b, r in K2A_BUCKETS if r == rep][0]
            good = True
            for T in sorted({lo, rep, hi}):
                stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
                branch = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
                if T <= SLICED_UNPOOL_MAX_TOKENS:
                    branch[:, UNPOOL_SLICE:] = 0
                ref = op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=None)
                pw, qw, al, bt, isc, pe, qe = w
                out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
                ns = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
                mod.sbb1_run(branch, stream, out, ns, e, pw, qw, al, bt, pe, qe, isc | 2)
                torch.cuda.synchronize()
                e0 = float((out.float() - ref[0].float()).norm() / ref[0].float().norm())
                e1 = float((ns - ref[1]).norm() / ref[1].norm())
                bw = bool(torch.equal(out, ref[0])) and bool(torch.equal(ns, ref[1]))
                good &= e0 < 1e-2 and e1 < 1e-4 and bool(torch.isfinite(ns).all())
                print(f"[vf] sbb2 self-test ffn T={T}: out {e0:.1e} stream {e1:.1e} bitwise {bw} -> {'ok' if good else 'FAIL'}",
                      flush=True)
            st2["ok"][rep] = good
        _VF["st2"] = st2
    except Exception as ex:
        print(f"[vf] sbb2 FAILED -> SGLang FFN boundary stays: {ex!r}", flush=True)
        _VF["st2"] = None
    print(f"[vf] riders router {'LIVE' if _VF['r8'] else 'OFF'} prt {'LIVE' if _VF['prt'] else 'OFF'} sbb2 "
          f"{sorted(k for k, v in (_VF['st2'] or {'ok': {}})['ok'].items() if v)} (switches ROUTER {VF_ROUTER} PRT {VF_PRT} "
          f"SBB2 {VF_SBB2}; {time.time() - t1:.1f}s, prepare so far {time.time() - t0:.1f}s)", flush=True)


SBB1_ENABLE = True
LAT_SBB_TRIG = {1: True, 8: True}
SBB1_ROUTE = {1: True, 8: True, 16: True, 32: True}
SBB1_PREPARE_BY_S = 250.0
_SBB1 = {"mod": None, "st": None}


def _sbb1_build():
    _pt1_wait("sbb1")
    if _SBB1["mod"] is not None:
        return _SBB1["mod"]
    from torch.utils.cpp_extension import load_inline
    t1 = time.time()
    _SBB1["mod"] = load_inline("sbb1_k3t", cpp_sources=SBB1_DECL, cuda_sources=SBB1_CU + SBB1_HOST, functions=["sbb1_run", "vf_router32"],
                               extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"[sbb1] compiled in {time.time() - t1:.1f}s", flush=True)
    return _SBB1["mod"]


def _sbb1_call(st, stream, branch, router_dtype, tokens):
    dev = branch.device
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    ns = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    rt = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev) if router_dtype is not None else st["e"]
    pw, qw, al, bt, isc, pe, qe = st["w"]
    st["mod"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, rt, pw, qw, al, bt, pe, qe,
                       isc | (8 if LAT_SBB_TRIG.get(_k2a_bucket_of(tokens)) else 0))
    return out, ns, (rt if router_dtype is not None else None)


def _sbb1(stream, branch, router_dtype, state, tokens):
    st = _SBB1["st"]
    if not SBB1_ENABLE or st is None or not st["ok"].get(_k2a_bucket_of(tokens)):
        return None
    if branch.dtype != torch.bfloat16 or stream.dtype != torch.float32:
        return None
    return _sbb1_call(st, stream, branch, router_dtype, tokens)


def _sbb1_prepare(state, t0):
    _SBB1["st"] = None
    if not SBB1_ENABLE or not SBB1_ROUTE:
        return None
    if time.time() - t0 > SBB1_PREPARE_BY_S:
        print(f"[sbb1] skipped: prepare already at {time.time() - t0:.0f}s", flush=True)
        return None
    t1 = time.time()
    try:
        mod = _sbb1_build()
        op = state["attn_boundary"]
        op._ensure_cached_weights(torch.bfloat16)
        dev = state["router"].device
        e = torch.empty(0, device=dev)
        w = (op._cc_post_weight.to(torch.bfloat16).contiguous() if op._post_norm.has_weight else e,
             op._cc_pre_weight.to(torch.bfloat16).contiguous() if op._pre_norm.has_weight else e,
             op._cc_alpha.float().contiguous() if op._use_residual_gate else e,
             op._cc_beta.float().contiguous() if op._use_residual_gate else e,
             1 if op._post_norm_is_input_scale else 0, float(op._post_norm.eps), float(op._pre_norm.eps))
        st = {"mod": mod, "w": w, "e": e, "ok": {}}
        gen = torch.Generator(device=dev).manual_seed(5253)
        for rep in sorted(SBB1_ROUTE):
            lo, hi = [(a, b) for a, b, r in K2A_BUCKETS if r == rep][0]
            good = True
            for T in sorted({lo, rep, hi}):
                stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
                branch = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
                rd = torch.float32 if T <= ROUTER_FP32_MAX_TOKENS else None
                ref = op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=rd)
                got = _sbb1_call(st, stream, branch, rd, T)
                torch.cuda.synchronize()
                e0 = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                e1 = float((got[1] - ref[1]).norm() / ref[1].norm())
                e2 = float((got[2] - ref[2]).norm() / ref[2].norm()) if rd is not None else 0.0
                good &= e0 < 1e-2 and e1 < 1e-4 and e2 < 1e-2 and bool(torch.isfinite(got[1]).all())
                print(f"[sbb1] self-test attn T={T}: out {e0:.1e} stream {e1:.1e} router {e2:.1e} -> {'ok' if good else 'FAIL'}",
                      flush=True)
            st["ok"][rep] = good
        _SBB1["st"] = st
    except Exception as ex:
        print(f"[sbb1] FAILED -> SGLang attention boundary stays: {ex!r}", flush=True)
        _SBB1["st"] = None
        return None
    print(f"[sbb1] buckets ok {sorted(k for k, v in st['ok'].items() if v)} ({time.time() - t1:.1f}s; prepare so far "
          f"{time.time() - t0:.1f}s)", flush=True)
    return st



VG_CU = ''
VG_HOST = ''
VG_DECL = ''
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
VGF_DECL = 'void vgf_run(torch::Tensor y, torch::Tensor rmap, torch::Tensor wts, torch::Tensor nw, torch::Tensor out, torch::Tensor xready, torch::Tensor doneA, torch::Tensor cnt, int64_t M, int64_t pdl);'

VG_ENABLE = True
VG_BND_ROUTE = {}
VG_BND_CAND = {}
VG_PREPARE_BY_S = 255.0
VGF_ENABLE = True
VGF_PDL = True
_VG = {"mod": None, "w": {}, "ok": set(), "build_s": 0.0, "vgf_ok": False}


def _vgf_on():
    return VGF_ENABLE and _VG["vgf_ok"] and _VG["mod"] is not None


def _vml_fin(st, sc, ids, wts, out, m):
    _VG["mod"].vgf_run(sc["y"], sc["rmap"], wts, st["nw"], out, sc["xready"], sc["doneA"], sc["cnt"], m, 1 if VGF_PDL else 0)
    return out


def _vgf_fc(rep):
    return globals().get("VML_FLAGS", {}).get(rep), globals().get("VML_COST", {}).get(rep)


def _vgf_prepare(state):
    _VG["vgf_ok"] = False
    if not VGF_ENABLE:
        return
    k33 = state.get("k2a33")
    if k33 is None or _VML.get("sc") is None or _VML.get("mod") is None or not any(_VML["ok"].values()):
        print("[vg] VG-F: VML1 not live -> vm_fin stays", flush=True)
        return
    try:
        _vg_build()
        st, sc = k33["st"], _VML["sc"]
        dev = st["nw"].device
        good, n = True, 0
        for bs, seed, dist in ((100, 41, "normal"), (128, 42, "skewed"), (160, 43, "skewed"), (200, 44, "skewed"),
                               (256, 45, "normal"), (320, 47, "normal")):
            if not _VML["ok"].get(_k2a_bucket_of(bs)):
                continue
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            x = d["hidden_states"].contiguous()
            ids = d["topk_ids"].to(torch.int32).contiguous()
            wts = d["topk_weights"].to(torch.float32).contiguous()
            fl, co = _vgf_fc(_k2a_bucket_of(bs))
            if (_vml_FLAGS_DEFAULT if fl is None else fl) & 8:
                continue
            prev = None
            for call in range(3):
                sc["err"].zero_()
                out = torch.full((bs, _vml_KDIM), 3.0, dtype=torch.bfloat16, device=dev)
                ref = torch.full((bs, _vml_KDIM), -3.0, dtype=torch.bfloat16, device=dev)
                _vml_call(_VML["mod"], st, sc, x, ids, wts, 0, mask=1, out=out, flags=fl, cost=co)
                _vml_fin(st, sc, ids, wts, out, bs)
                torch.cuda.synchronize()
                clean = int(sc["xready"].abs().sum()) + int(sc["doneA"].abs().sum()) + int(sc["cnt"].abs().sum())
                _vml_call(_VML["mod"], st, sc, x, ids, wts, 0, mask=2, out=ref, flags=fl, cost=co)
                torch.cuda.synchronize()
                eq = bool(torch.equal(out, ref))
                ok = eq and not int(sc["err"][0]) and clean == 0 and bool(torch.isfinite(out.float()).all())
                rep_eq = "-" if prev is None else str(bool(torch.equal(prev, ref)))
                prev = ref
                good &= ok
                n += 1
                if not ok or call in (0, 2):
                    print(f"[vg] VG-F self-test bs={bs} {dist} call {call}: bitwise vs vm_fin {eq} err {int(sc['err'][0])} "
                          f"counters {clean} (vm_main reproducible across calls: {rep_eq}) -> {'ok' if ok else 'FAIL'}",
                          flush=True)
        _VG["vgf_ok"] = bool(good and n)
    except Exception as ex:
        _VG["vgf_ok"] = False
        print(f"[vg] VG-F self-test raised {ex!r} -> vm_fin stays", flush=True)
    try:
        _k2a33_reset(state["k2a33"]["st"])
    except Exception:
        pass
    print(f"[vg] VG-F (VML1 finalize) {'LIVE' if _VG['vgf_ok'] else 'FALLBACK to vm_fin'}", flush=True)


def _vg_build():
    _pt1_wait("vg")
    if _VG["mod"] is not None:
        return _VG["mod"]
    from torch.utils.cpp_extension import load_inline
    t1 = time.time()
    _VG["mod"] = load_inline("vg_k3", cpp_sources=VG_DECL + chr(10) + VGF_DECL, cuda_sources=VG_CU + VG_HOST + VGF_CU + VGF_HOST,
                             functions=(["vgb_run"] if VG_DECL else []) + ["vgf_run"],
                             extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _VG["build_s"] = time.time() - t1
    print(f"[vg] VG-B / VG-F compiled in {_VG['build_s']:.1f}s", flush=True)
    return _VG["mod"]


def _vg_key(route):
    return tuple(_vg_key(x) if isinstance(x, (list, tuple)) else x for x in route)


def _vg_run(which, route, stream, branch):
    pw, qw, al, bt, isc, pe, qe = _VG["w"][which]
    T = branch.shape[0]
    if route[0] == "vgb":
        out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=branch.device)
        ns = torch.empty((T, HIDDEN), dtype=torch.float32, device=branch.device)
        if not _VG["mod"].vgb_run(branch.contiguous(), stream.contiguous(), out, ns, pw, qw, al, bt, pe, qe, int(route[1])):
            raise RuntimeError(f"vgb geometry {int(route[1]):#x} not built")
        return out, ns, None
    if route[0] == "fnr":
        lc = tuple(route[2]) if route[2] is not None else None
        o = _V9_FNR(branch.contiguous(), stream.contiguous(), pe, qe, post_norm_is_input_scale=isc,
                    post_weight=pw if pw.numel() else None, pre_weight=qw if qw.numel() else None, alpha=al, beta=bt,
                    want_router=False, launch_config=lc, cluster_size=int(route[1]))
        return o[0], o[1], None
    raise ValueError(f"vg route {route!r}")


def _vg_bnd(which, stream, branch, router_dtype, state, tokens):
    if not VG_ENABLE or router_dtype is not None or not VG_BND_ROUTE:
        return None
    rep = _k2a_bucket_of(tokens)
    route = VG_BND_ROUTE.get((which, rep))
    if route is None or (which, rep, _vg_key(route)) not in _VG["ok"]:
        return None
    if branch.dtype != torch.bfloat16 or stream.dtype != torch.float32:
        return None
    return _vg_run(which, route, stream, branch)


def _vg_prepare(state, t0):
    _VG["ok"] = set()
    if not VG_ENABLE:
        return
    if time.time() - t0 <= VG_PREPARE_BY_S:
        _vgf_prepare(state)
    else:
        print(f"[vg] VG-F SKIPPED (prepare already at {time.time() - t0:.0f}s): vm_fin stays", flush=True)
    routes = {}
    for k, r in list(VG_BND_ROUTE.items()) + [(k, r) for k, rs in VG_BND_CAND.items() for r in rs]:
        lst = routes.setdefault(k, [])
        if _vg_key(r) not in [_vg_key(x) for x in lst]:
            lst.append(r)
    if not routes:
        return
    if time.time() - t0 > VG_PREPARE_BY_S:
        print(f"[vg] SKIPPED (prepare already at {time.time() - t0:.0f}s): base boundaries stay", flush=True)
        return
    t1 = time.time()
    try:
        if any(r[0] == "vgb" for rs in routes.values() for r in rs):
            _vg_build()
        for which in ("attn", "ffn"):
            op = state["attn_boundary" if which == "attn" else "ffn_boundary"]
            op._ensure_cached_weights(torch.bfloat16)
            dev = state["router"].device
            e = torch.empty(0, device=dev, dtype=torch.bfloat16)
            pw = op._cc_post_weight.to(torch.bfloat16).contiguous() if op._post_norm.has_weight else e
            qw = op._cc_pre_weight.to(torch.bfloat16).contiguous() if op._pre_norm.has_weight else e
            if not op._use_residual_gate or op._cc_alpha.numel() != HIDDEN:
                raise RuntimeError(f"{which}: no channel-wise residual gate")
            isc = bool(op._post_norm_is_input_scale)
            if pw.numel() and not isc:
                raise RuntimeError(f"{which}: post weight after the norm is not supported")
            _VG["w"][which] = (pw, qw, op._cc_alpha.float().contiguous(), op._cc_beta.float().contiguous(), isc,
                               float(op._post_norm.eps), float(op._pre_norm.eps))
        gen = torch.Generator(device=state["router"].device).manual_seed(6363)
        for (which, rep), rs in sorted(routes.items(), key=str):
            lo, hi = [(a, b) for a, b, r in K2A_BUCKETS if r == rep][0]
            op = state["attn_boundary" if which == "attn" else "ffn_boundary"]
            for r in rs:
                if r[0] == "fnr" and _V9_FNR is None:
                    print(f"[vg] {which} bucket {rep} {r}: SGLang fused_norm_residual unavailable", flush=True)
                    continue
                if r[0] == "vgb" and which == "ffn" and (int(r[1]) >> 21) & 1:
                    print(f"[vg] {which} bucket {rep} {r}: stream before the PDL wait is illegal here -> dropped", flush=True)
                    continue
                good, worst = True, (0.0, 0.0)
                for T in sorted({lo, rep, min(hi, K2A_M_MAX)}):
                    stream = torch.randn((T, HIDDEN), generator=gen, device=state["router"].device) * 2.0
                    branch = torch.randn((T, HIDDEN), generator=gen, device=state["router"].device).to(torch.bfloat16)
                    ref = op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=None)
                    for _ in range(2):
                        got = _vg_run(which, r, stream, branch)
                        torch.cuda.synchronize()
                        e0 = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
                        e1 = float((got[1] - ref[1]).norm() / ref[1].norm())
                        worst = (max(worst[0], e0), max(worst[1], e1))
                        good &= e0 < 1e-2 and e1 < 1e-4 and bool(torch.isfinite(got[1]).all()) and bool(
                            torch.isfinite(got[0].float()).all())
                tag = "ok" if good else "FAIL"
                print(f"[vg] self-test {which} bucket {rep} {r}: out {worst[0]:.1e} stream {worst[1]:.1e} -> {tag}", flush=True)
                if good:
                    _VG["ok"].add((which, rep, _vg_key(r)))
        live = sorted(f"{w}/{b}" for (w, b), r in VG_BND_ROUTE.items() if (w, b, _vg_key(r)) in _VG["ok"])
        dead = sorted(f"{w}/{b}" for (w, b), r in VG_BND_ROUTE.items() if (w, b, _vg_key(r)) not in _VG["ok"])
        print(f"[vg] VG-B boundaries LIVE {live}{' FALLBACK (base boundary) ' + str(dead) if dead else ''} "
              f"({len(_VG['ok'])} routes ok, {time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
    except Exception as ex:
        _VG["ok"] = set()
        print(f"[vg] FAILED -> base boundaries everywhere: {ex!r}", flush=True)


GD_ENABLE = True
GD_BUILD = True
GD_REQUIRE = False
GD_PREPARE_BY_S = 230.0
GD_SMS = 132
GD_ESC_CAP = 32
GD_ROUTE = {("qkv", 1): (1, 1, 4, 8, 2, 4, 1), ("qkv", 8): (1, 1, 4, 8, 2, 4, 1), ("qkv", 16): (2, 1, 4, 8, 1, 4, 0),
            ("qkv", 32): (4, 1, 4, 8, 1, 4, 0),
            ("o", 1): (1, 2, 4, 4, 2, 1, 0), ("o", 8): (1, 1, 8, 2, 2, 1, 0), ("o", 16): (2, 2, 4, 4, 2, 1, 0),
            ("pooler", 1): (1, 1, 8, 16, 2, 1, 1), ("pooler", 8): (1, 1, 8, 16, 2, 1, 1),
            ("unpooler", 1): (1, 1, 16, 4, 2, 1, 0), ("unpooler", 8): (1, 1, 16, 4, 2, 1, 0),
            ("unpooler", 16): (2, 2, 4, 16, 2, 1, 0), ("unpooler", 32): (4, 2, 4, 16, 1, 1, 0)}
GD_PDL = {"qkv": False, "o": True, "pooler": True, "unpooler": True}
GD_TRIG = {"qkv": True, "o": True, "pooler": True, "unpooler": True}
GD_K3_TRIG = True
GD_DIMS = {"qkv": (2304, HIDDEN), "o": (HIDDEN, 1024), "pooler": (D_EXPERT, HIDDEN), "unpooler": (HIDDEN, D_EXPERT)}
GD_WEIGHT = {"qkv": "qkv_gate", "o": "o", "pooler": "pooler", "unpooler": "unpooler"}
GD_SELF_TEST_T = {1: (1,), 8: (8, 2, 5), 16: (16, 9), 32: (32, 17)}
GD_TOL = (2e-3, 1e-2)
GD_SELF_TEST_PDL = (0, 3, 3)
GD_ONE_WAVE = True
_GD = {"mod": None, "keys": None, "build_s": 0.0}
_GD_SEEN = set()
GD_CPP_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <vector>
int gd_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb,
               int nbx, int nz, int pdl, cudaStream_t st);
int gd_info_raw(int64_t inst, int64_t* res);
int gd_unpack_raw(const void* w, const void* esc, void* out, int K, int G, int nbx, int wr, int wk, int ku, uint32_t tb,
                  cudaStream_t st);
void gd_run(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, int64_t M, int64_t K,
            int64_t ldo, int64_t tb, int64_t nbx, int64_t nz, int64_t pdl) {
  const int e = gd_run_raw(inst, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), (int)M, (int)K, (int)ldo,
                           (uint32_t)tb, (int)nbx, (int)nz, (int)pdl, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gd_run: inst ", inst, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
int gdr_run_raw(int vf, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb, int nbx,
                int pdl, const float* x32, const float* w32, float* part, long long* marks, cudaStream_t st);
int gdr_info_raw(int vf, int64_t* res);
void gdr_run(int64_t vf, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, int64_t M, int64_t K,
             int64_t ldo, int64_t tb, int64_t nbx, int64_t pdl, torch::Tensor x32, torch::Tensor w32, torch::Tensor part,
             torch::Tensor marks) {
  if (vf == 2)
    TORCH_CHECK(x32.scalar_type() == at::kFloat && x32.is_contiguous() && x32.size(0) == M && x32.size(1) == 8192 &&
                w32.scalar_type() == at::kFloat && w32.is_contiguous() && w32.size(0) == 256 && w32.size(1) == 8192 &&
                part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() == 8 * M * 256 && M >= 1 && M <= 8 &&
                reinterpret_cast<uintptr_t>(x32.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w32.data_ptr()) % 16 == 0,
                "gdr_run: x32 f32 [M <= 8, 8192], w32 f32 [256, 8192], part [8, M, 256]");
  TORCH_CHECK(nbx == 256, "gdr_run: the T = 8 pooler grid (256 row tiles)");
  long long* mk = marks.numel() >= 4 * nbx ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  const int e = gdr_run_raw((int)vf, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), (int)M, (int)K, (int)ldo,
                            (uint32_t)tb, (int)nbx, (int)pdl, vf == 2 ? x32.data_ptr<float>() : nullptr,
                            vf == 2 ? w32.data_ptr<float>() : nullptr, vf == 2 ? part.data_ptr<float>() : nullptr, mk,
                            at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gdr_run: vf ", vf, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
std::vector<int64_t> gdr_info(int64_t vf) {
  std::vector<int64_t> r(4, 0);
  const int e = gdr_info_raw((int)vf, r.data());
  TORCH_CHECK(e == 0, "gdr_info: vf ", vf, " error ", e);
  return r;
}
std::vector<int64_t> gd_info(int64_t inst) {
  std::vector<int64_t> r(4, 0);
  const int e = gd_info_raw(inst, r.data());
  TORCH_CHECK(e == 0, "gd_info: inst ", inst, " error ", e);
  return r;
}
int gdc_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, void* out2, int M, int K, int ldo,
                int ldo2, uint32_t tb, uint32_t tbr, int nbx, int pdl, cudaStream_t st);
int gd_unpack_cat_raw(const void* w, const void* esc, void* out, int K, int G, int wk, int kup, uint32_t tb, uint32_t tbr,
                      cudaStream_t st);
void gdc_run(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, torch::Tensor out2,
             int64_t M, int64_t K, int64_t ldo, int64_t ldo2, int64_t tb, int64_t tbr, int64_t nbx, int64_t pdl) {
  TORCH_CHECK(out2.scalar_type() == at::kFloat && out2.is_contiguous() && out2.dim() == 3 && out2.size(0) == 8 &&
              out2.size(1) == M && out2.size(2) == ldo2, "gdc_run: router partials must be a contiguous [8, M, ldo2] f32 tensor");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == M, "gdc_run: pooler output");
  const int e = gdc_run_raw(inst, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), out2.data_ptr(), (int)M, (int)K,
                            (int)ldo, (int)ldo2, (uint32_t)tb, (uint32_t)tbr, (int)nbx, (int)pdl, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gdc_run: inst ", inst, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,
                   int64_t tb, int64_t tbr) {
  const int e = gd_unpack_cat_raw(w.data_ptr(), esc.data_ptr(), out.data_ptr(), (int)K, (int)G, (int)wk, (int)kup, (uint32_t)tb,
                                  (uint32_t)tbr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gd_unpack_cat: CUDA error ", e);
}
void gd_unpack(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t nbx, int64_t wr,
               int64_t wk, int64_t ku, int64_t tb) {
  const int e = gd_unpack_raw(w.data_ptr(), esc.data_ptr(), out.data_ptr(), (int)K, (int)G, (int)nbx, (int)wr, (int)wk,
                              (int)ku, (uint32_t)tb, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gd_unpack: CUDA error ", e);
}
'''
GD_CUDA_SRC = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stddef.h>

namespace gd {

struct Args {
  const uint4* w;          // packed stream: unit (step s, global warp gw), plane p, lane l at ((s*G + gw)*3 + p)*32 + l
  const uint32_t* esc;     // [G * 32] escape entries, sentinel 0xFFFFFFFF
  const uint16_t* x;       // [M, K] bf16 bits, row-major: the predecessor's output, read only after griddepcontrol.wait
  void* out;               // [M, ldo] bf16 (OUT 0) or f32 (OUT 1)
  int M, K, ldo;
  uint32_t tb;             // P12 window base per byte (bytes 0/1 -> rows g of the tile, bytes 2/3 -> rows g + 8)
  int trig;                // 1: griddepcontrol.launch_dependents at entry (host: one-wave grids only)
  const float* x32; const float* w32; float* part; long long* marks;   // VF (vfr_patches): router rows / marks
  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases
};

__device__ __forceinline__ uint4 ldw(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ldx(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t lde(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void decode(const uint4 (&p)[3], uint32_t tb, uint32_t (&a)[16]) {
  const uint32_t L[8] = {p[0].x, p[0].y, p[0].z, p[0].w, p[1].x, p[1].y, p[1].z, p[1].w};
  const uint32_t C[4] = {p[2].x, p[2].y, p[2].z, p[2].w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t h0 = (C[q] & 0x87878787u) + tb;
    const uint32_t h1 = (__funnelshift_r(C[q], C[q], 4) & 0x87878787u) + tb;
    a[4 * q + 0] = __byte_perm(L[2 * q], h0, 0x5140);
    a[4 * q + 1] = __byte_perm(L[2 * q], h0, 0x7362);
    a[4 * q + 2] = __byte_perm(L[2 * q + 1], h1, 0x5140);
    a[4 * q + 3] = __byte_perm(L[2 * q + 1], h1, 0x7362);
  }
}

__device__ __forceinline__ void patch(uint32_t (&a)[16], uint32_t e, int lane) {
  if (lane == (int)((e >> 15) & 31u)) {
    const int r = (int)((e >> 11) & 15u);
    const uint32_t v = (e & 127u) << (8 + (((e >> 10) & 1u) << 4));
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (i == r) a[i] ^= v;
  }
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t cl_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa0(uint32_t a) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;\n" : "=r"(r) : "r"(a));
  return r;
}
__device__ __forceinline__ void mb_init_expect(uint32_t mb, uint32_t bytes) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(mb) : "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(mb), "r"(bytes) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ bool mb_try(uint32_t mb) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(mb) : "memory");
  return ok != 0u;
}
__device__ __forceinline__ void st_async(uint32_t addr, float v, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n"
               :: "r"(addr), "r"(__float_as_uint(v)), "r"(mb) : "memory");
}
#define GD_SM90 1
#else
__device__ __forceinline__ void pdl_trigger() {}
__device__ __forceinline__ void pdl_wait() {}
#define GD_SM90 0
#endif


extern __shared__ __align__(16) float4 gd_vf_dsm[];   // VF = 2: router weight rows [8][256] float4 + x slice [8][256] float4
__device__ __forceinline__ void vf_cp16(void* sdst, const void* gsrc) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gsrc) : "memory");
}
__device__ __forceinline__ long long vf_gt() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int vf_smid() {
  int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

#define GC_XP(S_) (((S_) < KUP) ? (xb + (S_) * 8) : (xbr + ((S_) - KUP) * 8))
template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>
__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {
  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;
  constexpr int KUP = KU - CAT;   // GC: pooler units per warp; units KUP.. are the virtual router tile's
  static_assert(CAT == 0 || (S == 1 && WR == 1 && OUT == 0 && CAT == 2), "GC: cat routes are S 1, WR 1, bf16 pooler, 2 units");
  constexpr int NC = MT == 1 ? 2 : 1;
  constexpr int NO = WR * MT * 128;     // CTA output values
  constexpr int XB = XP ? 2 : 1;
  __shared__ float red[NW][MT * 4][32];
  __shared__ float redr[CAT ? NW : 1][CAT ? MT * 4 : 1][32];   // GC: the router partials of the CTA's warps
  __shared__ float pbuf[S > 1 ? S - 1 : 1][S > 1 ? NO : 1];
  __shared__ __align__(8) unsigned long long mbar;
  if (A.trig) pdl_trigger();
  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;
  const int wk = warp % WK;
  const int g = lane >> 2, t = lane & 3;
  const int split = S > 1 ? (int)blockIdx.y : 0;
  const int G = (int)(gridDim.x * gridDim.y) * NW;
  const int gw = ((int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x) * NW + warp;
  const int u0 = (split * WK + wk) * KUP;
  const int tok0 = (int)blockIdx.z * (8 * MT);
  const size_t wstep = (size_t)G * (NP * 32);
  const uint4* wp = A.w + (size_t)gw * (NP * 32) + lane;

  uint4 R[ST][NP];
#pragma unroll
  for (int i = 0; i < ST; ++i)
    if (i < KU) {
#pragma unroll
      for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)i * wstep + p * 32);
    }
  const uint32_t em = lde(A.esc + (size_t)gw * 32 + lane);
#if GD_SM90
  int crank = 0;   // reduction role: rank 0 of the cluster is the leader (the K range above uses blockIdx.y)
  if constexpr (S > 1) {
    crank = (int)cl_rank();
    if (crank == 0 && threadIdx.x == 0) mb_init_expect(smem_u32(&mbar), (uint32_t)((S - 1) * NO * 4));
    cl_arrive();
  }
#endif
  const int vf_c = (int)blockIdx.x, vf_bx = vf_c & 31, vf_sl = vf_c >> 5;
  if constexpr (VF >= 1) {
    if (A.marks && threadIdx.x == 0) { A.marks[vf_c * 4 + 0] = vf_smid(); A.marks[vf_c * 4 + 1] = vf_gt(); }
  }
  if constexpr (VF == 2) {   // router32 CTA (vf_bx, vf_sl): 8 rows x 1024 K of FP32 weights (no producer dependency)
    for (int idx = (int)threadIdx.x; idx < 8 * 256; idx += NT) {
      const int r = idx >> 8, c = idx & 255;
      vf_cp16(gd_vf_dsm + idx, reinterpret_cast<const float4*>(A.w32 + (size_t)(vf_bx * 8 + r) * 8192 + vf_sl * 1024) + c);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  pdl_wait();
  if constexpr (VF >= 1) {
    if (A.marks && threadIdx.x == 0) A.marks[vf_c * 4 + 2] = vf_gt();
  }
  if constexpr (VF == 2) {   // the router input slice (the boundary's FP32 output), tokens >= M zero
    float4* xs = gd_vf_dsm + 8 * 256;
    for (int idx = (int)threadIdx.x; idx < 8 * 256; idx += NT) {
      const int j = idx >> 8, c = idx & 255;
      if (j < A.M) vf_cp16(xs + idx, reinterpret_cast<const float4*>(A.x32 + (size_t)j * 8192 + vf_sl * 1024) + c);
      else xs[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  const uint4* xb = reinterpret_cast<const uint4*>(A.x + (size_t)u0 * 64 + 16 * t);
  const uint4* xbr = reinterpret_cast<const uint4*>(A.x + (size_t)(((int)blockIdx.x & 7) * 16 + 2 * wk) * 64 + 16 * t);   // GC
  int xo[MT];   // row offsets in uint4 (K / 8 per token)
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    int tok = tok0 + mt * 8 + g;
    tok = tok < A.M ? tok : A.M - 1;
    xo[mt] = tok * (A.K >> 3);
  }
  uint4 X[XB][MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    X[0][mt][0] = ldx(xb + xo[mt]);
    X[0][mt][1] = ldx(xb + xo[mt] + 1);
  }
  int ecur = 0;
  uint32_t enx = __shfl_sync(0xffffffffu, em, 0);

  float acc[NC][MT][4];
#pragma unroll
  for (int c = 0; c < NC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[c][mt][i] = 0.f;

#pragma unroll
  for (int s = 0; s < KU; ++s) {
    const int slot = s % ST;
    const int xs = XP ? (s & 1) : 0;
    if (XP) {
      if (s + 1 < KU) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          X[(s + 1) & (XB - 1)][mt][0] = ldx(GC_XP(s + 1) + xo[mt]);
          X[(s + 1) & (XB - 1)][mt][1] = ldx(GC_XP(s + 1) + xo[mt] + 1);
        }
      }
    } else if (s > 0) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        X[0][mt][0] = ldx(GC_XP(s) + xo[mt]);
        X[0][mt][1] = ldx(GC_XP(s) + xo[mt] + 1);
      }
    }
    uint32_t a[16];
    decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);
    if (s + ST < KU) {
#pragma unroll
      for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);
    }
    while ((int)(enx >> 20) == s) {
      patch(a, enx, lane);
      ++ecur;
      enx = ecur < 32 ? __shfl_sync(0xffffffffu, em, ecur & 31) : 0xFFFFFFFFu;
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const uint4 xv = X[xs][mt][q >> 1];
        const uint32_t b0 = (q & 1) ? xv.z : xv.x;
        const uint32_t b1 = (q & 1) ? xv.w : xv.y;
        mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);
      }
    }
    if constexpr (CAT > 0) {
      if (s == KUP - 1) {   // GC: pooler tile done; park its partials, reuse the accumulators for the router units
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float v = acc[0][mt][i];
            if constexpr (NC == 2) v += acc[1][mt][i];
            red[warp][mt * 4 + i][lane] = v;
            acc[0][mt][i] = 0.f;
            if constexpr (NC == 2) acc[1][mt][i] = 0.f;
          }
      }
    }
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = acc[0][mt][i];
      if constexpr (NC == 2) v += acc[1][mt][i];
      if constexpr (CAT > 0) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;
    }
  __syncthreads();
  auto cta_value = [&](int i) -> float {
    const int l = i & 31;
    const int c = (i >> 5) & 3, r = i >> 7, mt = r % MT, w2 = r / MT;
    float v = 0.f;
#pragma unroll
    for (int k2 = 0; k2 < WK; ++k2) v += red[w2 * WK + k2][mt * 4 + c][l];
    return v;
  };
  auto store = [&](int i, float v) {
    const int l = i & 31, gg = l >> 2, tt = l & 3;
    const int c = (i >> 5) & 3, r = i >> 7, mt = r % MT, w2 = r / MT;
    const int row = ((int)blockIdx.x * WR + w2) * 16 + gg + 8 * (c >> 1);
    const int tok = tok0 + mt * 8 + 2 * tt + (c & 1);
    if (tok < A.M) {
      if constexpr (OUT == 1) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + row] = v;
      else reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
    }
  };
  if constexpr (S == 1) {
    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));
    if constexpr (CAT > 0) {   // GC: this K slice's router partial: logit(8 t + gg) = tile rows gg (hi) + gg + 8 (lo)
      const int gc_t = (int)blockIdx.x >> 3, gc_ks = (int)blockIdx.x & 7;
      for (int i = (int)threadIdx.x; i < NO / 2; i += NT) {
        const int l = i & 31, gg = l >> 2, tt = l & 3;
        const int c = (i >> 5) & 1, mt = i >> 6;
        float v = 0.f;
#pragma unroll
        for (int k2 = 0; k2 < WK; ++k2) v += redr[k2][mt * 4 + c][l] + redr[k2][mt * 4 + c + 2][l];
        const int tok = tok0 + mt * 8 + 2 * tt + c;
        if (tok < A.M)
          reinterpret_cast<float*>(A.out2)[(size_t)gc_ks * A.sst2 + (size_t)tok * A.ldo2 + gc_t * 8 + gg] = v;
      }
    }
    if constexpr (VF == 2) {   // router32_kernel<8>'s arithmetic for row vf_bx * 8 + warp, K slice vf_sl
      asm volatile("cp.async.wait_all;\n" ::: "memory");
      __syncthreads();
      if (warp < 8) {
        const float4* wsm = gd_vf_dsm + warp * 256;
        const float4* xs = gd_vf_dsm + 8 * 256;
        float racc[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) racc[j] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int c = i * 32 + lane;
          const float4 wv = wsm[c];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float4 xv = xs[j * 256 + c];
            racc[j] = fmaf(wv.x, xv.x, racc[j]);
            racc[j] = fmaf(wv.y, xv.y, racc[j]);
            racc[j] = fmaf(wv.z, xv.z, racc[j]);
            racc[j] = fmaf(wv.w, xv.w, racc[j]);
          }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float a = racc[j];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
          if (lane == 0 && j < A.M) A.part[((long)vf_sl * A.M + j) * 256 + vf_bx * 8 + warp] = a;
        }
      }
    }
    if constexpr (VF >= 1) {
      if (A.marks) {
        __syncthreads();
        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();
      }
    }
  } else {
#if GD_SM90
    if (crank != 0) {
      cl_wait();   // the leader's mbarrier is initialised (its arrive came after the init + fence)
      const uint32_t rb = mapa0(smem_u32(&pbuf[crank - 1][0]));
      const uint32_t rm = mapa0(smem_u32(&mbar));
      for (int i = (int)threadIdx.x; i < NO; i += NT) st_async(rb + 4u * (uint32_t)i, cta_value(i), rm);
      return;
    }
    float own[(NO + NT - 1) / NT];
#pragma unroll
    for (int j = 0; j < (NO + NT - 1) / NT; ++j) {
      const int i = (int)threadIdx.x + j * NT;
      own[j] = i < NO ? cta_value(i) : 0.f;
    }
    const uint32_t mb = smem_u32(&mbar);
    bool ok = true;
    for (uint32_t it = 0; !mb_try(mb); ++it)
      if (it > (1u << 22)) {
        ok = false;
        break;
      }
#pragma unroll
    for (int j = 0; j < (NO + NT - 1) / NT; ++j) {
      const int i = (int)threadIdx.x + j * NT;
      if (i < NO) {
        float v = own[j];
#pragma unroll
        for (int r = 0; r < S - 1; ++r) v += pbuf[r][i];
        store(i, ok ? v : __int_as_float(0x7fc00000));
      }
    }
#endif
  }
}

template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>
static int launch(const Args& A, int nbx, int nz, int pdl, cudaStream_t st) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)nbx, (unsigned)S, (unsigned)nz);
  cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  unsigned na = 0;
  if (S > 1) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = 1;
    attr[na].val.clusterDim.y = (unsigned)S;
    attr[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl & 1) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  return (int)cudaLaunchKernelEx(&cfg, gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>, A);
}

template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>
static int info(int64_t* res) {
  cudaFuncAttributes fa;
  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>;
  int e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, 0);
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes;
  return e;
}

__global__ void unpack_kernel(const uint4* __restrict__ w, const uint32_t* __restrict__ esc, uint32_t* __restrict__ out,
                              int K, int G, int nbx, int WR, int WK, int KU, uint32_t tb) {
  const int gw = (int)blockIdx.x, s = (int)blockIdx.y, lane = (int)threadIdx.x;
  const int g = lane >> 2, t = lane & 3, NW = WR * WK;
  const int c = gw / NW, wi = gw % NW, wr = wi / WK, wk = wi % WK, bx = c % nbx, by = c / nbx;
  const int rt = bx * WR + wr, u = (by * WK + wk) * KU + s;
  uint4 p[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) p[i] = w[(((size_t)s * G + gw) * 3 + i) * 32 + lane];
  uint32_t a[16];
  decode(p, tb, a);
  for (int j = 0; j < 32; ++j) {
    const uint32_t e = esc[(size_t)gw * 32 + j];
    if ((int)(e >> 20) == s) patch(a, e, lane);
  }
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = rt * 16 + g + 8 * (i & 1);
      const int k = u * 64 + 16 * t + 4 * q + 2 * (i >> 1);
      out[((size_t)row * K + k) >> 1] = a[4 * q + i];
    }
}

__global__ void unpack_cat_kernel(const uint4* __restrict__ w, const uint32_t* __restrict__ esc, uint32_t* __restrict__ out,
                                  int K, int G, int WK, int KUP, uint32_t tb, uint32_t tbr) {
  const int gw = (int)blockIdx.x, s = (int)blockIdx.y, lane = (int)threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const int bx = gw / WK, wk = gw % WK;
  int rt, u;
  uint32_t tbs;
  if (s < KUP) { rt = bx; u = wk * KUP + s; tbs = tb; }
  else { rt = (G / WK) + (bx >> 3); u = (bx & 7) * 16 + 2 * wk + (s - KUP); tbs = tbr; }   // virtual router tile bx / 8
  uint4 p[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) p[i] = w[(((size_t)s * G + gw) * 3 + i) * 32 + lane];
  uint32_t a[16];
  decode(p, tbs, a);
  for (int j = 0; j < 32; ++j) {
    const uint32_t e = esc[(size_t)gw * 32 + j];
    if ((int)(e >> 20) == s) patch(a, e, lane);
  }
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = rt * 16 + g + 8 * (i & 1);
      const int k = u * 64 + 16 * t + 4 * q + 2 * (i >> 1);
      out[((size_t)row * K + k) >> 1] = a[4 * q + i];
    }
}
#undef GC_XP

}  // namespace gd

using namespace gd;

#define GD_INST(ID, ...) \
  case ID: return launch<__VA_ARGS__>(A, nbx, nz, pdl, st);
#define GD_INFO(ID, ...) \
  case ID: return info<__VA_ARGS__>(res);

int gd_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb,
               int nbx, int nz, int pdl, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;
  switch (inst) {
@INST@
    default: return -1;
  }
}


int gdc_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, void* out2, int M, int K, int ldo,
                int ldo2, uint32_t tb, uint32_t tbr, int nbx, int pdl, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;
  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;
  const int nz = 1;
  switch (inst) {
@INST@
    default: return -1;
  }
}

int gd_unpack_cat_raw(const void* w, const void* esc, void* out, int K, int G, int wk, int kup, uint32_t tb, uint32_t tbr,
                      cudaStream_t st) {
  unpack_cat_kernel<<<dim3((unsigned)G, (unsigned)(kup + 2)), 32, 0, st>>>(
      reinterpret_cast<const uint4*>(w), reinterpret_cast<const uint32_t*>(esc), reinterpret_cast<uint32_t*>(out), K, G, wk,
      kup, tb, tbr);
  return (int)cudaGetLastError();
}

static constexpr int GDR_DS = 2 * 8 * 256 * 16;   // 64 KB dynamic smem for VF = 2

int gdr_run_raw(int vf, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb, int nbx,
                int pdl, const float* x32, const float* w32, float* part, long long* marks, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = x32; A.w32 = w32; A.part = part; A.marks = marks;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)nbx, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  unsigned na = 0;
  if (pdl & 1) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    na = 1;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  if (vf == 2) {
    static bool set = false;
    if (!set) {
      const int e = (int)cudaFuncSetAttribute(gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, GDR_DS);
      if (e) return e;
      set = true;
    }
    cfg.dynamicSmemBytes = GDR_DS;
    return (int)cudaLaunchKernelEx(&cfg, gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2>, A);
  }
  cfg.dynamicSmemBytes = 0;
  return (int)cudaLaunchKernelEx(&cfg, gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 1>, A);
}

int gdr_info_raw(int vf, int64_t* res) {
  cudaFuncAttributes fa;
  auto fn = vf == 2 ? gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2> : gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 1>;
  const int ds = vf == 2 ? GDR_DS : 0;
  int e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ds > 0 ? ds : 0);
  if (e) return e;
  e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, ds);
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes;
  return e;
}

int gd_info_raw(int64_t inst, int64_t* res) {
  switch (inst) {
@INFO@
    default: return -1;
  }
}

int gd_unpack_raw(const void* w, const void* esc, void* out, int K, int G, int nbx, int wr, int wk, int ku, uint32_t tb,
                  cudaStream_t st) {
  unpack_kernel<<<dim3((unsigned)G, (unsigned)ku, 1), 32, 0, st>>>(reinterpret_cast<const uint4*>(w),
      reinterpret_cast<const uint32_t*>(esc), reinterpret_cast<uint32_t*>(out), K, G, nbx, wr, wk, ku, tb);
  return (int)cudaGetLastError();
}
'''


VF_GDR = True
VF_GDM = False
_VFR = {"ok": False, "marks": None, "seen": set()}


def _vfr_gdr(state, x, z, vf):
    T = x.shape[0]
    r = _gd_live(state, "pooler", T)
    if r is None or _gd_bucket(T) != 8:
        return None
    y = torch.empty((T, r["N"]), dtype=torch.bfloat16, device=x.device)
    part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=x.device)
    pdl = (1 if GD_PDL.get("pooler") else 0) | (2 if GD_TRIG.get("pooler") else 0)
    mk = _VFR["marks"] if VF_GDM else _VFR["nomk"]
    r["mod"].gdr_run(vf, r["w"], r["esc"], x, y, T, r["K"], r["N"], r["tb"], r["nbx"], pdl,
                     z if vf == 2 else _VFR["nomk"], state["router"], part, mk)
    if (vf, T) not in _VFR["seen"]:
        _VFR["seen"].add((vf, T))
        print(f"[vfr] first call vf {vf} T={T} pdl {pdl} marks {VF_GDM}", flush=True)
    return y, part


def _vfr_fused(state, ffn_input, router_input):
    if not (VF_GDR and _VFR["ok"]) or router_input is None or router_input.dtype != torch.float32:
        return None
    if not (2 <= ffn_input.shape[0] <= 8 and router_input.is_contiguous() and ffn_input.is_contiguous()):
        return None
    return _vfr_gdr(state, ffn_input, router_input, 2)


def _vfr_marked_pooler(state, x):
    if not (VF_GDM and _VFR["ok"]) or not 2 <= x.shape[0] <= 8 or not x.is_contiguous():
        return None
    out = _vfr_gdr(state, x, None, 1)
    return None if out is None else out[0]


def _vfr_prepare(state):
    _VFR["ok"] = False
    st = state.get("gd")
    r = st["route"].get(("pooler", 8)) if st is not None else None
    if r is None or not r["desc"].startswith("m1(1,8,16,1)s2x1") or state.get("v8") is None or _SBB1.get("mod") is None:
        print(f"[vfr] FALLBACK: GD pooler T=8 route {None if r is None else r['desc']} is not (1,8,16,1)s2x1 -> side-stream router32",
              flush=True)
        return
    dev = state["router"].device
    try:
        i1, i2 = r["mod"].gdr_info(1), r["mod"].gdr_info(2)
        print(f"[vfr] instances regs / local / occ / smem: vf1 {list(i1)} vf2 {list(i2)}", flush=True)
        if i2[2] < 2 or i2[1] > 64 or i1[2] < 2:
            print("[vfr] FALLBACK: below 2 CTAs/SM or spills -> side-stream router32", flush=True)
            return
        _VFR["marks"] = torch.zeros((4 * r["nbx"],), dtype=torch.int64, device=dev)
        _VFR["nomk"] = torch.empty((0,), dtype=torch.int64, device=dev)
        gen = torch.Generator(device=dev).manual_seed(8282)
        ok = True
        for m in (8, 5, 2):
            z = torch.randn((m, HIDDEN), generator=gen, device=dev)
            z = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).contiguous()
            h = (z * 0.7).to(torch.bfloat16).contiguous()
            y0 = _gd(state, "pooler", h)
            p0 = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
            state["v8"]["mod"].v8_router32(z, state["router"], p0)
            y1, p1 = _vfr_gdr(state, h, z, 2)
            y2, _ = _vfr_gdr(state, h, None, 1)
            torch.cuda.synchronize()
            e = (bool(torch.equal(y0, y1)), bool(torch.equal(p0, p1)), bool(torch.equal(y0, y2)))
            print(f"[vfr] self-test m={m}: fused pooler bitwise {e[0]}, router partials bitwise vs router32 {e[1]}, "
                  f"marked pooler bitwise {e[2]}", flush=True)
            ok &= all(e)
        _VFR["seen"].clear()
        _VFR["ok"] = ok
    except Exception as ex:
        print(f"[vfr] FALLBACK: {ex!r} -> side-stream router32", flush=True)
        _VFR["ok"] = False
    print(f"[vfr] fused T=8 pooler + router rows {'LIVE' if _VFR['ok'] else 'OFF'} (VF_GDR {VF_GDR})", flush=True)


def _gd_bucket(T):
    return 1 if T == 1 else 8 if T <= 8 else 16 if T <= 16 else 32 if T <= 32 else None


def _gd_nk(proj, b):
    n, k = GD_DIMS[proj]
    return (UNPOOL_SLICE if proj == "unpooler" and b <= SLICED_UNPOOL_MAX_TOKENS else n), k


def _gd_key(n, k, rt):
    mt, wr, wk, ku, st_, s, xp = rt
    nrt, nu, nt = n // 16, k // 64, wr * wk * 32
    if n % 16 or k % 64 or nrt % wr or nu != wk * ku * s or st_ > ku or s not in (1, 2, 4, 8) or nt > 1024 or mt not in (1, 2, 4):
        return None
    ctas = (nrt // wr) * s
    return (0, mt, wr, wk, ku, st_, s, max(1, min(2048 // nt, -(-ctas // GD_SMS))), xp)


def _gd_keys():
    keys = []
    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None
        if key is not None and key not in keys:
            keys.append(key)
    for key in _gd_cat_keys():
        if key not in keys:
            keys.append(key)
    return keys


def _gd_build():
    _pt1_wait("gd")
    keys = _gd_keys()
    if _GD["mod"] is not None and _GD["keys"] == keys:
        return _GD["mod"]
    from torch.utils.cpp_extension import load_inline
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    cuda = GD_CUDA_SRC.replace("@INST@", inst).replace("@INFO@", info)
    tag = hashlib.sha1((GD_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"gd_{tag}", cpp_sources=GD_CPP_SRC, cuda_sources=cuda, functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _GD.update(mod=mod, keys=keys, build_s=time.time() - t1)
    return mod


def _gd_i32(t64):
    return torch.where(t64 >= 2 ** 31, t64 - 2 ** 32, t64).to(torch.int32)


def _gd_frag(wv):
    n, k = wv.shape
    dev = wv.device
    w16 = (wv.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).reshape(-1)
    lane = torch.arange(32, device=dev)
    g = (lane >> 2).view(32, 1, 1)
    t = (lane & 3).view(32, 1, 1)
    q = torch.arange(4, device=dev).view(1, 4, 1)
    j = torch.arange(8, device=dev).view(1, 1, 8)
    row16 = (g + 8 * ((j >> 1) & 1)).expand(32, 4, 8)
    k64 = (16 * t + 4 * q + (j & 1) + 2 * (j >> 2)).expand(32, 4, 8)
    nrt, nu = n // 16, k // 64
    rows = torch.arange(nrt, device=dev).view(nrt, 1, 1, 1, 1) * 16 + row16.view(1, 1, 32, 4, 8)
    cols = torch.arange(nu, device=dev).view(1, nu, 1, 1, 1) * 64 + k64.view(1, 1, 32, 4, 8)
    return w16[rows * k + cols]


def _gd_window_base(wv):
    h7 = (wv.contiguous().view(torch.int16).to(torch.int32) >> 8) & 0x7F
    hist = torch.bincount(h7.reshape(-1), minlength=128).double()
    cs = torch.cat([torch.zeros(1, dtype=hist.dtype, device=hist.device), hist.cumsum(0)])
    return int((cs[8:129] - cs[0:121]).argmax())


def _gd_pack(wv, wr, wk, ku, s, base):
    n, k = wv.shape
    nrt, nu = n // 16, k // 64
    nbx = nrt // wr
    G = nbx * s * wr * wk
    dev = wv.device
    fr = _gd_frag(wv)
    lo = fr & 0xFF
    h7 = (fr >> 8) & 0x7F
    sg = (fr >> 15) & 1
    d = h7 - base
    isesc = (d < 0) | (d > 7)
    d = torch.where(isesc, torch.zeros_like(d), d)
    dA, dB, sA, sB = d[..., :4], d[..., 4:], sg[..., :4], sg[..., 4:]
    sBr = sB[..., [3, 0, 1, 2]]
    byte = (dA | (sBr << 3) | (dB << 4) | (sA << 7)).to(torch.int64)
    cw = _gd_i32(byte[..., 0] | (byte[..., 1] << 8) | (byte[..., 2] << 16) | (byte[..., 3] << 24))
    lo64 = lo.to(torch.int64)
    l0 = _gd_i32(lo64[..., 0] | (lo64[..., 1] << 8) | (lo64[..., 2] << 16) | (lo64[..., 3] << 24))
    l1 = _gd_i32(lo64[..., 4] | (lo64[..., 5] << 8) | (lo64[..., 6] << 16) | (lo64[..., 7] << 24))
    p0 = torch.stack([l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]], dim=-1)
    p1 = torch.stack([l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]], dim=-1)
    units = torch.stack([p0, p1, cw], dim=2)
    pos = isesc.nonzero()
    nesc = int(pos.shape[0])
    esc = torch.full((G * 32,), -1, dtype=torch.int32, device=dev)
    if nesc:
        rt, u, ln, qq, jv = (pos[:, i] for i in range(5))
        bx, wri = rt // wr, rt % wr
        by, wki, st_ = u // (wk * ku), (u // ku) % wk, u % ku
        gw = ((by * nbx + bx) * wr + wri) * wk + wki
        x7 = (h7[rt, u, ln, qq, jv] ^ base).to(torch.int64)
        ent = (st_ << 20) | (ln << 15) | ((4 * qq + (jv >> 1)) << 11) | ((jv & 1) << 10) | x7
        order = torch.argsort((gw * 4096 + st_) * 16 + 4 * qq + (jv >> 1))
        gws, ents = gw[order], ent[order]
        cnt = torch.bincount(gws, minlength=G)
        if int(cnt.max()) > GD_ESC_CAP:
            raise ValueError(f"escape cap: a warp has {int(cnt.max())} > {GD_ESC_CAP} escapes")
        rank = torch.arange(nesc, device=dev) - (torch.cumsum(cnt, 0) - cnt)[gws]
        esc[gws * 32 + rank] = _gd_i32(ents)
    v = units.view(nbx, wr, s, wk, ku, 3, 32, 4).permute(4, 2, 0, 1, 3, 5, 6, 7).contiguous().view(-1)
    return {"w": v, "esc": esc, "tb": base * 0x01010101, "G": G, "nbx": nbx, "nesc": nesc}


def _gd_test_input(m, k, seed, device, outliers=False):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device=device)
    if outliers:
        idx = torch.randperm(k, generator=gen, device=device)[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device=device) < 1 / 8
    probes[torch.randint(m, (1,), generator=gen, device=device)] = True
    if m >= 4:
        probes[0] = False
    cols = torch.randint(k, (m,), generator=gen, device=device)
    signs = torch.randint(2, (m,), generator=gen, device=device) * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device=device)
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16), probes


def _gd_selftest(r, w, b):
    n, k = w.shape
    dev = w.device
    ldo = HIDDEN if r["slice"] else n
    parts = []
    for T in GD_SELF_TEST_T[b]:
        for outl in (False, True):
            x, probes = _gd_test_input(T, k, 977 + 31 * T + int(outl), dev, outl)
            outs = []
            for pdl in GD_SELF_TEST_PDL:
                o = torch.full((T, ldo), float("nan"), dtype=torch.bfloat16, device=dev)
                r["mod"].gd_run(r["inst"], r["w"], r["esc"], x, o, T, k, ldo, r["tb"], r["nbx"], 1, pdl)
                outs.append(o)
            ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
            torch.cuda.synchronize()
            got = outs[0][:, :n].double()
            if not bool(torch.isfinite(got).all()):
                return False, f"T={T} outl={outl}: non-finite"
            dd = got - ref
            rn = ref.norm(dim=-1)
            tot = float(dd.norm() / ref.norm().clamp(min=1e-30))
            row = float((dd.norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
            pr = float((dd[probes].norm(dim=-1) / rn[probes].clamp(min=1e-30)).max()) if bool(probes.any()) else 0.0
            rep = all(bool(torch.equal(outs[0][:, :n], o[:, :n])) for o in outs[1:])
            rest = all(bool(torch.isnan(o[:, n:]).all()) for o in outs) if ldo > n else True
            if not (tot <= GD_TOL[0] and row <= GD_TOL[1] and pr == 0.0 and rep and rest):
                return False, f"T={T} outl={outl}: rel {tot:.1e} row {row:.1e} probe {pr:.1e} repeat {rep} cols {rest}"
            parts.append(f"T={T}{'o' if outl else ''} {tot:.1e}")
    return True, " ".join(parts)


def _gd_require(why):
    if GD_REQUIRE:
        raise RuntimeError(f"[gd] GD_REQUIRE: {why}")


def _gd_prepare(state, t0):
    state["gd"] = None
    if not GD_BUILD:
        return None
    if time.time() - t0 > GD_PREPARE_BY_S:
        print(f"[gd] skipped: prepare already took {time.time() - t0:.1f}s (> {GD_PREPARE_BY_S}s): today's routes", flush=True)
        return _gd_require("prepare budget")
    t1 = time.time()
    try:
        mod = _gd_build()
    except Exception as ex:
        print(f"[gd] build FAILED -> today's routes everywhere: {str(ex)[-2000:]}", flush=True)
        return _gd_require("build failed")
    dense = state["dense"]
    dev = dense["qkv_gate"].device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ids = {k: i for i, k in enumerate(_GD["keys"])}
    packs, routes, bases, live = {}, {}, {}, []
    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        if not rt:
            continue
        n, k = _gd_nk(proj, b)
        w = dense[GD_WEIGHT[proj]]
        why = None
        if w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] != k or w.shape[0] < n:
            why = f"weight {tuple(w.shape)} {w.dtype} vs ({n}, {k}) bf16"
        key = _gd_key(n, k, rt)
        if why is None and (key not in ids or 8 * rt[0] < b):
            why = "geometry does not fit its bucket / not built"
        if why is None:
            info = mod.gd_info(ids[key])
            ctas = (n // 16 // rt[1]) * rt[5]
            if info[1] > 64 or (GD_ONE_WAVE and info[2] * sms < ctas):
                why = f"regs {info[0]} spill {info[1]} occ {info[2]} for {ctas} CTAs (> one wave or spills)"
        if why is None:
            wv = w[:n].contiguous()
            pk_key = (proj, n, rt[1], rt[2], rt[3], rt[5])
            if pk_key not in packs:
                try:
                    if (proj, n) not in bases:
                        bases[(proj, n)] = _gd_window_base(wv)
                    pk = _gd_pack(wv, rt[1], rt[2], rt[3], rt[5], bases[(proj, n)])
                    un = torch.empty_like(wv)
                    mod.gd_unpack(pk["w"], pk["esc"], un, k, pk["G"], pk["nbx"], rt[1], rt[2], rt[3], pk["tb"])
                    torch.cuda.synchronize()
                    exact = bool(torch.equal(un.view(torch.int16), wv.view(torch.int16)))
                    print(f"[gd] pack {proj} {n}x{k} ({rt[1]},{rt[2]},{rt[3]},{rt[5]}): {pk['w'].numel() * 4 / 1e6:.1f} MB, "
                          f"base {bases[(proj, n)]}, {pk['nesc']} escapes, bit-exact unpack {exact}", flush=True)
                    packs[pk_key] = pk if exact else None
                    del un
                except Exception as ex:
                    print(f"[gd] pack {proj} ({rt[1]},{rt[2]},{rt[3]},{rt[5]}) failed: {ex!r}", flush=True)
                    packs[pk_key] = None
            pk = packs[pk_key]
            if pk is None:
                why = "pack / bit-exact unpack failed"
        if why is None:
            desc = f"m{rt[0]}({rt[1]},{rt[2]},{rt[3]},{rt[5]})s{rt[4]}x{rt[6]}r{info[0]}"
            r = {"mod": mod, "inst": ids[key], "w": pk["w"], "esc": pk["esc"], "tb": pk["tb"], "nbx": pk["nbx"], "N": n, "K": k,
                 "T": 8 * rt[0], "slice": proj == "unpooler" and n == UNPOOL_SLICE, "desc": desc}
            try:
                ok, msg = _gd_selftest(r, wv, b)
            except Exception as ex:
                ok, msg = False, f"EXC {ex!r}"
            if not ok:
                why = f"self-test: {msg}"
            else:
                routes[(proj, b)] = r
                live.append(f"{proj}/{b} {desc}")
                print(f"[gd] self-test {proj}/{b} {desc}: ok ({msg})", flush=True)
        if why is not None:
            print(f"[gd] FALLBACK {proj}/{b}: {why} -> today's route", flush=True)
    torch.cuda.empty_cache()
    state["gd"] = {"mod": mod, "route": routes} if routes else None
    print(f"[gd] live ({time.time() - t1:.1f}s, build {_GD['build_s']:.0f}s): {'; '.join(live) or 'none'}", flush=True)
    missing = [kk for kk, rt in GD_ROUTE.items() if rt and kk not in routes]
    if missing:
        _gd_require(f"routes not live: {missing}")
    return None


def _gd_live(state, proj, tokens):
    st = state.get("gd")
    if not GD_ENABLE or st is None:
        return None
    b = _gd_bucket(tokens)
    if b is None or not GD_ROUTE.get((proj, b)):
        return None
    r = st["route"].get((proj, b))
    if r is None or tokens > r["T"] or (proj == "unpooler" and r["slice"] != (tokens <= SLICED_UNPOOL_MAX_TOKENS)):
        return None
    return r


def _gd(state, proj, x, out=None, ldo=None):
    T = x.shape[0]
    r = _gd_live(state, proj, T)
    if r is None or not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == r["K"] and x.is_contiguous()
                         and x.data_ptr() % 16 == 0):
        return None
    try:
        if out is None:
            out = torch.empty((T, r["N"]), dtype=torch.bfloat16, device=x.device)
            ldo = r["N"]
        pdl = (1 if GD_PDL.get(proj) else 0) | (2 if GD_TRIG.get(proj) else 0)
        r["mod"].gd_run(r["inst"], r["w"], r["esc"], x, out, T, r["K"], ldo, r["tb"], r["nbx"], 1, pdl)
        if (proj, T) not in _GD_SEEN:
            _GD_SEEN.add((proj, T))
            print(f"[gd] first call {proj} T={T}: {r['desc']} pdl {pdl}", flush=True)
        return out
    except Exception as ex:
        _k3_disable(state, "gd", None, "GD", ex)
        return None


def _gd_unpool(state, x, tokens):
    r = _gd_live(state, "unpooler", tokens)
    if r is None:
        return None
    if not r["slice"]:
        return _gd(state, "unpooler", x)
    zb = state.get("v12_zbuf")
    if not (V12_ZBUF and zb is not None and R1_UNPOOL_INTO_ZEROS and x.dtype == zb.dtype):
        return None
    return _gd(state, "unpooler", x, zb[:tokens], HIDDEN)


def _gd_k3trig(state, tokens):
    if tokens > SLICED_UNPOOL_MAX_TOKENS or _gd_live(state, "unpooler", tokens) is None:
        return None
    return bool(GD_K3_TRIG and GD_PDL.get("unpooler") and _k2a_bucket_of(tokens) in SU1_K3_TRIG_AT)


GD_CAT = {16: True, 32: True}
GD_CAT_ROUTE = {16: (2, 1, 8, 18, 2, 1, 0), 32: (4, 1, 8, 18, 1, 1, 0)}
GD_CAT_UNITS = 2
GD_CAT_PDL = 3
GD_CAT_SELF_TEST_T = {16: (16, 9), 32: (32, 17)}
GD_CAT_TOL = (1e-5, 1e-4)
_GD_CAT_SEEN = set()


def _gd_cat_key(rt):
    mt, wr, wk, ku, st_, s, xp = rt
    if wr != 1 or s != 1 or mt not in (2, 4) or wk * (ku - GD_CAT_UNITS) != HIDDEN // 64 or st_ > ku or xp not in (0, 1):
        return None
    if (2 * N_EXPERTS) // 16 * 8 != D_EXPERT // 16 or wk * GD_CAT_UNITS != 16:
        return None
    nt = wk * 32
    ctas = D_EXPERT // 16
    return (0, mt, wr, wk, ku, st_, s, max(1, min(2048 // nt, -(-ctas // GD_SMS))), xp, GD_CAT_UNITS)


def _gd_cat_keys():
    keys = []
    for b, rt in sorted(GD_CAT_ROUTE.items()):
        key = _gd_cat_key(rt) if rt else None
        if key is not None and key not in keys:
            keys.append(key)
    return keys


def _gd_cat_virtual(hi, lo):
    nt = N_EXPERTS // 8
    return torch.stack((hi.view(nt, 8, HIDDEN), lo.view(nt, 8, HIDDEN)), dim=1).reshape(2 * N_EXPERTS, HIDDEN).contiguous()


def _gd_cat_pack(wp, wv, wk, kup, base_p, bh, bl):
    n, k = wp.shape
    nbx = n // 16
    G = nbx * wk
    dev = wp.device
    frp = _gd_frag(wp).view(nbx, wk, kup, 32, 4, 8).permute(2, 0, 1, 3, 4, 5)
    frv = _gd_frag(wv)
    bxs = torch.arange(nbx, device=dev)
    tiles = (bxs // 8)[:, None].expand(nbx, wk)
    uu = (bxs % 8)[:, None] * 16 + 2 * torch.arange(wk, device=dev)[None, :]
    frr = torch.stack([frv[tiles, uu + j] for j in range(GD_CAT_UNITS)], dim=0)
    fr = torch.cat([frp, frr], dim=0).contiguous()
    ku = kup + GD_CAT_UNITS
    half = ((torch.arange(8, device=dev) >> 1) & 1).view(1, 1, 1, 1, 1, 8)
    base = torch.full(fr.shape, base_p, dtype=torch.int32, device=dev)
    base[kup:] = torch.where(half.bool(), torch.full_like(half, bl), torch.full_like(half, bh)).to(torch.int32)
    lo = fr & 0xFF
    h7 = (fr >> 8) & 0x7F
    sg = (fr >> 15) & 1
    d = h7 - base
    isesc = (d < 0) | (d > 7)
    d = torch.where(isesc, torch.zeros_like(d), d)
    dA, dB, sA, sB = d[..., :4], d[..., 4:], sg[..., :4], sg[..., 4:]
    sBr = sB[..., [3, 0, 1, 2]]
    byte = (dA | (sBr << 3) | (dB << 4) | (sA << 7)).to(torch.int64)
    cw = _gd_i32(byte[..., 0] | (byte[..., 1] << 8) | (byte[..., 2] << 16) | (byte[..., 3] << 24))
    lo64 = lo.to(torch.int64)
    l0 = _gd_i32(lo64[..., 0] | (lo64[..., 1] << 8) | (lo64[..., 2] << 16) | (lo64[..., 3] << 24))
    l1 = _gd_i32(lo64[..., 4] | (lo64[..., 5] << 8) | (lo64[..., 6] << 16) | (lo64[..., 7] << 24))
    p0 = torch.stack([l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]], dim=-1)
    p1 = torch.stack([l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]], dim=-1)
    units = torch.stack([p0, p1, cw], dim=3)
    pos = isesc.nonzero()
    nesc = int(pos.shape[0])
    esc = torch.full((G * 32,), -1, dtype=torch.int32, device=dev)
    if nesc:
        s_, bx, wi, ln, qq, jv = (pos[:, i] for i in range(6))
        gw = bx * wk + wi
        x7 = (h7[s_, bx, wi, ln, qq, jv] ^ base[s_, bx, wi, ln, qq, jv]).to(torch.int64)
        ent = (s_ << 20) | (ln << 15) | ((4 * qq + (jv >> 1)) << 11) | ((jv & 1) << 10) | x7
        order = torch.argsort((gw * 4096 + s_) * 16 + 4 * qq + (jv >> 1))
        gws, ents = gw[order], ent[order]
        cnt = torch.bincount(gws, minlength=G)
        if int(cnt.max()) > GD_ESC_CAP:
            raise ValueError(f"escape cap: a warp has {int(cnt.max())} > {GD_ESC_CAP} escapes")
        rank = torch.arange(nesc, device=dev) - (torch.cumsum(cnt, 0) - cnt)[gws]
        esc[gws * 32 + rank] = _gd_i32(ents)
    tbr = bh | (bh << 8) | (bl << 16) | (bl << 24)
    return {"w": units.contiguous().view(-1), "esc": esc, "tb": base_p * 0x01010101, "tbr": tbr, "G": G, "nbx": nbx,
            "nesc": nesc, "ku": ku}


def _gd_cat_run(r, x, y, part, T, pdl):
    r["mod"].gdc_run(r["inst"], r["w"], r["esc"], x, y, part, T, HIDDEN, D_EXPERT, N_EXPERTS, r["tb"], r["tbr"], r["nbx"], pdl)


def _gd_cat_selftest(r, wp, hi, lo, w32, b):
    dev = wp.device
    parts = []
    ref_w = hi.double() + lo.double()
    for T in GD_CAT_SELF_TEST_T[b]:
        for outl in (False, True):
            x, probes = _gd_test_input(T, HIDDEN, 977 + 31 * T + int(outl), dev, outl)
            outs = []
            for pdl in GD_SELF_TEST_PDL:
                y = torch.full((T, D_EXPERT), float("nan"), dtype=torch.bfloat16, device=dev)
                part = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
                _gd_cat_run(r, x, y, part, T, pdl)
                outs.append((y, part))
            torch.cuda.synchronize()
            y, part = outs[0]
            if not (bool(torch.isfinite(y).all()) and bool(torch.isfinite(part).all())):
                return False, f"T={T} outl={outl}: non-finite"
            refp = (x.double() @ wp.double().T).to(torch.bfloat16).double()
            dd = y.double() - refp
            rn = refp.norm(dim=-1)
            tot = float(dd.norm() / refp.norm().clamp(min=1e-30))
            row = float((dd.norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
            pr = float((dd[probes].norm(dim=-1) / rn[probes].clamp(min=1e-30)).max()) if bool(probes.any()) else 0.0
            lg = part.double().sum(0)
            refl = x.double() @ ref_w.T
            ref32 = x.double() @ w32.double().T
            el = float((lg - refl).norm() / refl.norm().clamp(min=1e-30))
            e32 = float((lg - ref32).norm() / ref32.norm().clamp(min=1e-30))
            rep = all(bool(torch.equal(y, o[0])) and bool(torch.equal(part, o[1])) for o in outs[1:])
            if not (tot <= GD_TOL[0] and row <= GD_TOL[1] and pr == 0.0 and el <= GD_CAT_TOL[0] and e32 <= GD_CAT_TOL[1] and rep):
                return False, (f"T={T} outl={outl}: pooler rel {tot:.1e} row {row:.1e} probe {pr:.1e}; logits rel {el:.1e} "
                               f"(vs fp32 router {e32:.1e}) repeat {rep}")
            parts.append(f"T={T}{'o' if outl else ''} {tot:.1e}/{el:.1e}")
    return True, " ".join(parts)


def _gd_cat_prepare(state, t0):
    state["gd_cat"] = None
    st, v8 = state.get("gd"), state.get("v8")
    if st is None or v8 is None or not GD_CAT_ROUTE:
        print(f"[gc] FALLBACK: GD {'live' if st else 'not live'}, v8 {'live' if v8 else 'not live'} -> today's cat", flush=True)
        return None
    t1 = time.time()
    mod = st["mod"]
    ids = {k: i for i, k in enumerate(_GD["keys"])}
    dev = v8["wcat"].device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    wp = v8["wcat"][:D_EXPERT]
    hi = v8["wcat"][D_EXPERT:D_EXPERT + N_EXPERTS]
    lo = v8["wcat"][D_EXPERT + N_EXPERTS:D_EXPERT + 2 * N_EXPERTS]
    w32 = state["router"]
    try:
        wv = _gd_cat_virtual(hi, lo)
        base_p, bh, bl = _gd_window_base(wp), _gd_window_base(hi), _gd_window_base(lo)
        ref = torch.cat((wp, wv), dim=0)
    except Exception as ex:
        print(f"[gc] FALLBACK: virtual router / bases failed: {ex!r} -> today's cat", flush=True)
        return None
    packs, routes, live = {}, {}, []
    for b, rt in sorted(GD_CAT_ROUTE.items()):
        why = None
        key = _gd_cat_key(rt) if rt else None
        if key is None or key not in ids:
            why = "geometry invalid / not built"
        if why is None:
            info = mod.gd_info(ids[key])
            ctas = D_EXPERT // 16
            if info[1] > 64 or (GD_ONE_WAVE and info[2] * sms < ctas):
                why = f"regs {info[0]} spill {info[1]} occ {info[2]} for {ctas} CTAs (> one wave or spills)"
        if why is None:
            kup = rt[3] - GD_CAT_UNITS
            pk_key = (rt[2], kup)
            if pk_key not in packs:
                try:
                    pk = _gd_cat_pack(wp, wv, rt[2], kup, base_p, bh, bl)
                    un = torch.empty_like(ref)
                    mod.gd_unpack_cat(pk["w"], pk["esc"], un, HIDDEN, pk["G"], rt[2], kup, pk["tb"], pk["tbr"])
                    torch.cuda.synchronize()
                    exact = bool(torch.equal(un.view(torch.int16), ref.view(torch.int16)))
                    print(f"[gc] pack cat ({rt[2]},{kup}+{GD_CAT_UNITS}): {pk['w'].numel() * 4 / 1e6:.1f} MB, bases {base_p}/{bh}/{bl}, "
                          f"{pk['nesc']} escapes, bit-exact unpack {exact}", flush=True)
                    packs[pk_key] = pk if exact else None
                    del un
                except Exception as ex:
                    print(f"[gc] pack cat ({rt[2]},{kup}) failed: {ex!r}", flush=True)
                    packs[pk_key] = None
            pk = packs[pk_key]
            if pk is None:
                why = "pack / bit-exact unpack failed"
        if why is None:
            desc = f"gc m{rt[0]}({rt[1]},{rt[2]},{rt[3]},{rt[5]})s{rt[4]}x{rt[6]}r{info[0]}"
            r = {"mod": mod, "inst": ids[key], "w": pk["w"], "esc": pk["esc"], "tb": pk["tb"], "tbr": pk["tbr"], "nbx": pk["nbx"],
                 "T": 8 * rt[0], "desc": desc}
            try:
                ok, msg = _gd_cat_selftest(r, wp, hi, lo, w32, b)
            except Exception as ex:
                ok, msg = False, f"EXC {ex!r}"
            if not ok:
                why = f"self-test: {msg}"
            else:
                routes[b] = r
                live.append(f"cat/{b} {desc}")
                print(f"[gc] self-test cat/{b} {desc}: ok ({msg})", flush=True)
        if why is not None:
            print(f"[gc] FALLBACK cat/{b}: {why} -> today's cat", flush=True)
    torch.cuda.empty_cache()
    state["gd_cat"] = routes or None
    print(f"[gc] live ({time.time() - t1:.1f}s): {'; '.join(live) or 'none'}; GD_CAT {GD_CAT}", flush=True)
    return None


def _gd_cat_live(state, tokens):
    st = state.get("gd_cat")
    if not GD_ENABLE or st is None or state.get("gd_cat_ok") is False:
        return None
    b = _gd_bucket(tokens)
    if b is None or not GD_CAT.get(b):
        return None
    r = st.get(b)
    return r if r is not None and tokens <= r["T"] else None


def _gd_cat(state, x, data, tokens):
    r = _gd_cat_live(state, tokens)
    if r is None or not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[0] == tokens and x.shape[1] == HIDDEN
                         and x.is_contiguous() and x.data_ptr() % 16 == 0):
        return None
    try:
        y = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=x.device)
        part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=x.device)
        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)
        if tokens not in _GD_CAT_SEEN:
            _GD_CAT_SEEN.add(tokens)
            print(f"[gc] first call T={tokens}: {r['desc']} pdl {GD_CAT_PDL}", flush=True)
        return _v8_tail(y, part, state, data)
    except Exception as ex:
        state["gd_cat_ok"] = False
        print(f"[gc] call raised ({ex!r}): today's cat from here on", flush=True)
        return None



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

def kernel_fn(data, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    dense = state["dense"]
    qkv = _gd(state, "qkv", data["attn_input"])
    if qkv is None:
        qkv = _v20_p1m(state, "qkv", data["attn_input"])
    if qkv is None:
        qkv = _v13_p1(state, "qkv", data["attn_input"])
    if qkv is None:
        qkv = _dg1(state, "qkv", data["attn_input"])
    if qkv is None:
        qkv = (_v8_gemm(data["attn_input"], "qkv_gate", dense["qkv_gate"], state) if tokens == 1
               else _v12_gemm(data["attn_input"], dense["qkv_gate"], "qkv", tokens))
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
    attn_output = _gd(state, "o", o)
    if attn_output is None and ks1_ran:
        attn_output = _su1(state, "o", o, tokens)
    if attn_output is None:
        attn_output = _v20_p1m(state, "o", o)
    if attn_output is None:
        attn_output = _v13_p1(state, "o", o)
    if attn_output is None:
        attn_output = _dg1(state, "o", o)
    if attn_output is None:
        attn_output = _v12_gemm(o, dense["o"], "o", tokens)

    mode, place = _v8_router_mode(tokens, state)
    tail = V8_TAIL_ROUTE.get(_k2a_bucket_of(tokens)) if state.get("v8") is not None else None
    if tail == "cat" and tokens <= ROUTER_FP32_MAX_TOKENS:
        tail = "sep"
    split_router = tokens > ROUTER_FP32_MAX_TOKENS or R1_ROUTER_SPLIT_ALL_T or mode not in ("stock", "f32") or tail == "cat"
    ffn_input, stream, router_input = _v9_bnd("attn", data["stream"], attn_output,
                                              None if split_router else torch.float32, state, tokens)
    g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)
    if g2out is not None:
        expert_x, topk_weights, topk_ids, local_ids = g2out
    elif tail == "cat" and (gc := _gd_cat(state, ffn_input, data, tokens)) is not None:
        expert_x, topk_weights, topk_ids, local_ids = gc
    elif tail == "cat":
        y32 = _v20_p1m(state, "cat", ffn_input)
        if y32 is None:
            y32 = _dg1(state, "cat", ffn_input)
        if y32 is None:
            y32 = torch.mm(ffn_input, state["v8"]["wcat"].t(), out_dtype=torch.float32)
        expert_x, topk_weights, topk_ids, local_ids = _v8_tail(y32, None, state, data)
    else:
        vgr = _vfr_fused(state, ffn_input, router_input) if (tail is not None and mode == "f32") else None
        if vgr is not None:
            expert_input, logits = vgr
        elif place == "main":
            logits = _v8_logits(mode, ffn_input, router_input, state)
            if tail is None:
                topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
            expert_input = _v13_pooler(ffn_input, dense, state)
        else:
            main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
            side_stream.wait_stream(main_stream)
            if place != "side_first":
                expert_input = _v13_pooler(ffn_input, dense, state)
            with torch.cuda.stream(side_stream):
                logits = _v8_logits(mode, ffn_input, router_input, state)
                if tail is None:
                    topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
            if place == "side_first":
                expert_input = _v13_pooler(ffn_input, dense, state)
            main_stream.wait_stream(side_stream)
            if not torch.cuda.is_current_stream_capturing():
                for tensor in ((logits,) if tail is not None else (topk_weights, topk_ids)):
                    tensor.record_stream(main_stream)
        if tail is not None:
            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(expert_input, logits, state, data)
        else:
            if R1_LOCAL_ID_TABLE:
                idx = torch.add(topk_ids, data["local_start"], alpha=N_EXPERTS)
                local_ids = torch.index_select(state["local_id_table"], 0, idx.reshape(-1)).view(tokens, TOP_K)
            else:
                local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
            expert_x = state["pre_expert_norm"](expert_input)
    _SU1_FLAG["k3trig"] = _su1_k3trig(state, tokens)
    try:
        expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)
    finally:
        _SU1_FLAG["k3trig"] = False
    if expert_output is None:
        expert_output = _experts(expert_x, topk_weights, local_ids, state, ids_are_local=R1_LOCAL_ID_TABLE)
    ffn_output = _gd_unpool(state, expert_output, tokens)
    if ffn_output is None:
        ffn_output = _su1(state, "unpooler", expert_output, tokens)
    if ffn_output is None:
        ffn_output = _v13_unpool(expert_output, state, tokens)
    if ffn_output is None and tokens == 1:
        ffn_output = _v8_unpool_t1(expert_output, state)
    if ffn_output is not None:
        pass
    elif tokens <= SLICED_UNPOOL_MAX_TOKENS:
        if R1_UNPOOL_INTO_ZEROS:
            zb = state.get("v12_zbuf")
            if V12_ZBUF and zb is not None and expert_output.dtype == zb.dtype:
                ffn_output = zb[:tokens]
            else:
                ffn_output = torch.zeros((tokens, HIDDEN), dtype=expert_output.dtype, device=expert_output.device)
            torch.mm(expert_output, dense["unpooler"][:UNPOOL_SLICE].t(), out=ffn_output[:, :UNPOOL_SLICE])
        else:
            unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
            ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _v20_p1m(state, "unpooler", expert_output)
        if ffn_output is None:
            ffn_output = _dg1(state, "unpooler", expert_output)
        if ffn_output is None:
            ffn_output = _v12_gemm(expert_output, dense["unpooler"], "unpool", tokens)

    next_attn_input, stream, _ = _v9_bnd("ffn", stream, ffn_output, None, state, tokens)
    return next_attn_input, stream, topk_ids


LT_ENABLE = True
LT_PIN = {('cat', 256): (66, 125, 35, 1, 0, 0, 2, 0, 5), ('qkv', 128): (66, 80, 35, 1, 0, 0, 2, 0, 7), ('qkv', 256): (66, 134, 35, 1, 0, 0, 2, 0, 9)}
LT_WS = 32 << 20
LT_CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>
struct LtPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  cudaDataType_t ct = CUDA_R_16BF;
  std::vector<cublasLtMatmulHeuristicResult_t> res;
};
static cublasLtHandle_t g_h = nullptr;
static std::vector<LtPlan> g_plans;
static const cublasLtMatmulAlgoConfigAttributes_t kAttrs[9] = {
    CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
static const cublasLtMatmulAlgo_t* lt_algo(int64_t pid, int64_t ai) {
  TORCH_CHECK(pid >= 0 && (size_t)pid < g_plans.size() && ai >= 0 && (size_t)ai < g_plans[(size_t)pid].res.size(), "lt: index");
  return &g_plans[(size_t)pid].res[(size_t)ai].algo;
}
int64_t lt_plan(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t ws_bytes, int64_t max_algos) {
  if (!g_h) TORCH_CHECK(cublasLtCreate(&g_h) == CUBLAS_STATUS_SUCCESS, "lt: create");
  LtPlan p;
  TORCH_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "lt: desc");
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, n, k) == CUBLAS_STATUS_SUCCESS, "lt: la");
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, k, m, k) == CUBLAS_STATUS_SUCCESS, "lt: lb");
  p.ct = out_f32 ? CUDA_R_32F : CUDA_R_16BF;
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lc, p.ct, n, m, n) == CUBLAS_STATUS_SUCCESS, "lt: lc");
  cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS, "lt: pref");
  uint64_t ws = (uint64_t)ws_bytes;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p.res.resize((size_t)max_algos);
  int got = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(g_h, p.op, p.la, p.lb, p.lc, p.lc, pref, (int)max_algos,
                                                    p.res.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "lt: heuristic status ", (int)s);
  p.res.resize((size_t)got);
  g_plans.push_back(p);
  return ((int64_t)(g_plans.size() - 1) << 8) | (int64_t)got;
}
std::vector<int64_t> lt_cfg(int64_t pid, int64_t ai) {
  const cublasLtMatmulAlgo_t* a = lt_algo(pid, ai);
  std::vector<int64_t> v;
  for (auto t : kAttrs) {
    size_t sz = 0;
    uint64_t x = 0;
    bool ok = cublasLtMatmulAlgoConfigGetAttribute(a, t, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz > 0 &&
              sz <= sizeof(x) && cublasLtMatmulAlgoConfigGetAttribute(a, t, &x, sz, &sz) == CUBLAS_STATUS_SUCCESS;
    v.push_back(ok ? (int64_t)x : -1);
  }
  return v;
}
int64_t lt_make(int64_t pid, std::vector<int64_t> cfg, int64_t ws_bytes) {
  TORCH_CHECK(pid >= 0 && (size_t)pid < g_plans.size() && cfg.size() == 9, "lt: make");
  LtPlan& p = g_plans[(size_t)pid];
  cublasLtMatmulAlgo_t a;
  if (cublasLtMatmulAlgoInit(g_h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, p.ct, p.ct, (int)cfg[0], &a) !=
      CUBLAS_STATUS_SUCCESS)
    return -1;
  for (int i = 1; i < 9; ++i) {
    size_t sz = 0;
    uint64_t v = (uint64_t)cfg[(size_t)i];
    if (cublasLtMatmulAlgoConfigGetAttribute(&a, kAttrs[i], nullptr, 0, &sz) != CUBLAS_STATUS_SUCCESS || sz == 0 ||
        sz > sizeof(v) || cublasLtMatmulAlgoConfigSetAttribute(&a, kAttrs[i], &v, sz) != CUBLAS_STATUS_SUCCESS)
      return -1;
  }
  cublasLtMatmulHeuristicResult_t r = {};
  if (cublasLtMatmulAlgoCheck(g_h, p.op, p.la, p.lb, p.lc, p.lc, &a, &r) != CUBLAS_STATUS_SUCCESS ||
      r.workspaceSize > (size_t)ws_bytes)
    return -1;
  r.algo = a;
  p.res.push_back(r);
  return (int64_t)p.res.size() - 1;
}
void lt_run(int64_t pid, int64_t ai, torch::Tensor w, torch::Tensor x, torch::Tensor out, double beta, torch::Tensor ws) {
  const cublasLtMatmulAlgo_t* a = lt_algo(pid, ai);
  LtPlan& p = g_plans[(size_t)pid];
  const float alpha = 1.f, b = (float)beta;
  cublasStatus_t s = cublasLtMatmul(g_h, p.op, &alpha, w.data_ptr(), p.la, x.data_ptr(), p.lb, &b, out.data_ptr(), p.lc,
                                    out.data_ptr(), p.lc, a, ws.data_ptr(), (size_t)ws.numel(),
                                    at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "lt_run: status ", (int)s);
}
'''
_LT = {"mod": None, "err": None}


def _lt_flags():
    import glob
    import os
    from torch.utils.cpp_extension import CUDA_HOME
    dirs = [os.path.join(CUDA_HOME, d) for d in ("lib64", "lib", os.path.join("targets", "x86_64-linux", "lib"))] if CUDA_HOME else []
    tdir = os.path.dirname(torch.__file__)
    dirs += glob.glob(os.path.join(os.path.dirname(tdir), "nvidia", "*", "lib")) + [os.path.join(tdir, "lib")]
    for d in dirs:
        libs = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
        if libs:
            lib = "-lcublasLt" if any(x.endswith("libcublasLt.so") for x in libs) else "-l:" + os.path.basename(libs[-1])
            return ["-L" + d, lib, "-Wl,-rpath," + d]
    return ["-lcublasLt"]


def _lt_build():
    if _LT["mod"] is None and _LT["err"] is None:
        from torch.utils.cpp_extension import load_inline
        flags = _lt_flags()
        try:
            _LT["mod"] = load_inline(name="lt2_" + hashlib.sha1((LT_CPP + " ".join(flags)).encode()).hexdigest()[:10],
                                     cpp_sources=LT_CPP, functions=["lt_plan", "lt_cfg", "lt_make", "lt_run"], extra_ldflags=flags,
                                     with_cuda=True, verbose=False)
        except Exception as ex:
            _LT["err"] = f"{str(ex)[-600:]} (flags {flags})"
    return _LT["mod"]


def _lt_today(proj, x, w):
    return torch.mm(x, w.t(), out_dtype=torch.float32) if proj == "cat" else _gemm(x, w)


def _lt_prepare(state):
    state["lt"] = None
    if not (LT_ENABLE and LT_PIN):
        return
    t1 = time.time()
    _pt1_wait("lt")
    mod = _lt_build()
    if mod is None:
        print(f"[lt] build FAILED -> today's GEMMs: {str(_LT['err'])[-300:]}", flush=True)
        return
    dev = state["dense"]["qkv_gate"].device
    ws = torch.empty((LT_WS,), dtype=torch.uint8, device=dev)
    gen = torch.Generator(device=dev).manual_seed(4321)
    pins, notes = {}, []
    for (proj, T), cfg in sorted(LT_PIN.items()):
        try:
            w = _dg1_weight(state, proj)
            n, k = w.shape
            f32 = proj == "cat"
            code = int(mod.lt_plan(T, n, k, int(f32), LT_WS, 32))
            pid, cnt = code >> 8, code & 255
            ai = next((i for i in range(cnt) if tuple(mod.lt_cfg(pid, i)) == tuple(cfg)), None)
            if ai is None:
                ai = int(mod.lt_make(pid, list(cfg), LT_WS))
                if ai < 0 or tuple(mod.lt_cfg(pid, ai)) != tuple(cfg):
                    notes.append(f"{proj}/{T}: config {tuple(cfg)} rejected by cuBLASLt -> today")
                    continue
            x = torch.randn((T, k), generator=gen, device=dev).to(torch.bfloat16)
            y = [torch.full((T, n), float("nan"), dtype=torch.float32 if f32 else torch.bfloat16, device=dev) for _ in range(2)]
            for o in y:
                mod.lt_run(pid, ai, w, x, o, 0.0, ws)
            ref = x.double() @ w.double().t()
            today = _lt_today(proj, x, w).float()
            e64 = float((y[0].double() - ref).norm() / ref.norm())
            et = float((y[0].float() - today).norm() / today.norm())
            ok = (bool(torch.isfinite(y[0]).all()) and torch.equal(y[0], y[1]) and e64 < DG1_TOL[1 if f32 else 0]
                  and et < DG1_TOL[2])
            notes.append(f"{proj}/{T}: lt{ai}{'' if ai < cnt else ' (built)'} rel {e64:.1e}, vs today {et:.1e} -> "
                         f"{'pinned' if ok else 'FAIL: today'}")
            if ok:
                pins[(proj, T)] = (pid, ai, n, k, f32)
        except Exception as ex:
            notes.append(f"{proj}/{T}: FALLBACK {ex!r}"[:160])
    torch.cuda.synchronize()
    state["lt"] = {"mod": mod, "pins": pins, "ws": ws} if pins else None
    print(f"[lt] LT2 ({time.time() - t1:.1f}s): " + "; ".join(notes), flush=True)


def _lt(state, proj, x):
    st = state.get("lt")
    if not LT_ENABLE or st is None or (proj, x.shape[0]) not in LT_PIN:
        return None
    p = st["pins"].get((proj, x.shape[0]))
    ok = (p is not None and x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == p[3] and x.is_contiguous()
          and x.data_ptr() % 256 == 0)
    if (proj, x.shape[0]) not in _LT.setdefault("seen", set()):
        _LT["seen"].add((proj, x.shape[0]))
        print(f"[lt] first call {proj}/{x.shape[0]}: {'pinned lt' + str(p[1]) if ok else 'today'} (x ptr % 256 = "
              f"{x.data_ptr() % 256})", flush=True)
    if not ok:
        return None
    try:
        out = torch.empty((x.shape[0], p[2]), dtype=torch.float32 if p[4] else torch.bfloat16, device=x.device)
        st["mod"].lt_run(p[0], p[1], _dg1_weight(state, proj), x, out, 0.0, st["ws"])
        return out
    except Exception as ex:
        _k3_disable(state, "lt", None, "LT2", ex)
        return None

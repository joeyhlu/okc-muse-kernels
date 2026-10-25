# okc: problem=muse_attention_decode
# nonce k1-v9c-1 (route: as k1_v4; fold merge of the routed (kind, L) through a thread-block cluster (DSMEM), C1)
"""K1 v9c (C1, from k1_v4; design and pod plan: kernels/muse_attention_decode/k1_c1_design.md). Only the split/merge
path changes; everything else is k1_v4 byte for byte (tile loop, roles, swizzles, new-token path, self-test cases):
  * cluster merge (K1_CLUSTER_T entries, fold mode): the splits of one (request, KV-head pair) are the ranks of one
    thread-block cluster (cudaLaunchKernelEx + cudaLaunchAttributeClusterDimension, CS <= 16 CTAs, grid = B*CS x Hkv/2).
    The plan is cluster-shaped (request = cluster index, split = rank: no device plan search). Each CTA leaves its O
    partial + lse in its own smem, barrier.cluster.arrive.release / wait.acquire, then rank r merges rows
    [r*16L/S, (r+1)*16L/S) by reading the peers' smem (mapa + ld.shared::cluster) with merge_finalize's exact lse
    math (same split order, same ops), applies the norm-gate epilogue and stores. No global partials, no counter, no
    threadfence, no last-CTA pass. A second cluster barrier keeps every CTA resident until its peers' reads are done.
  * K1_CLUSTER (define, default 1) compiles the cluster path in; the v4 global merge stays in the same build and is
    selected per call (cluster=False / table entry absent), so one build A/Bs both. K1_CLUSTER=0 = k1_v4's device code.
  * Self-test: k1_v4's cases unchanged (fold, sep, fold), then cluster cases (CS 2..16, idle ranks, S_b = 1, Hkv 16).
    A cluster failure disables only the cluster route (v4 merge kept); a v4 failure falls back to FA3 as before.
  * No new spin waits (R9): barrier.cluster is a hardware barrier over co-scheduled CTAs; every CTA of a cluster runs
    exactly two arrive/wait pairs on every path. A cluster-shape mismatch sets err[0] and returns before any barrier.

K1 v4 notes (kept): routes as k1_v2/v3 (K1 for L=1, rung-1 FA3 path with L2 for L=8). Measured on the box H200
(kernels/muse_attention_decode/k1_v4.md):
  * fold merge rewritten (merge_finalize): one pass with an online max over batches of 16 (L=1) / 8 (L=8) splits, so
    that many partial loads are in flight per thread, gate values loaded up front. Lab k1_v4_lab_merge.py: one
    256-thread CTA reads freshly written L2 partials at 24 GB/s with 1 load in flight, 75-80 GB/s with 4-8 (190 GB/s
    at 1024-thread MLP); load flavour (default/cg/volatile/relaxed) and fences make no difference. Sliding L=1 bench
    A/B vs k1_v2: B1 13.87 -> 12.76 us, B16 16.59 -> 15.74, B32 20.08 -> 19.12 (last-CTA merge 4.1 -> 1.8 us).
  * L=8 kernel (unrouted): K and V converted ONCE per tile into fp16 smem tiles, QK and PV on m16n8k16 FP16 with
    ldmatrix B fragments, fp16 Q (no emulated e4m3 mma). Correct (self-test, judge checks, k1_v4_emulate.py) but
    B32/B16 full 442/236 us vs rung-1 382/197: killed per the brief (needs <= 330/170).

K1 v1 notes (kept): fused gated-attention decode/verify for Muse Spark Open (sm_90, mma.sync; one main kernel per call, plus
a split-combine kernel only where the host table asks for it; fixed grids, per-request lengths read on device).

Changes from v0 (measured on the judge, job ded828a1027f427a: v0 1.081x, L=8 compute-bound, B=1 full combine 19.5 us):
  * L=8 (verify): warp = (KV head, row half = 2 m-tiles, key half of every 64-key tile). Each K/V byte is loaded and
    converted by 2 warps instead of 4, and every V fragment feeds 2 m-tiles. QK^T runs on
    mma.sync.m16n8k32.e4m3 with Q rounded to E4M3 (the normalized, roped value, before x5.473 -- the rounding the
    judge's FP8-query reference uses) and the cache's K bytes as the B operand (no conversion). PV stays FP16
    (P fp16, V e4m3->f16 exact). The two key halves merge in smem at the end. O is rescaled only when a row max
    moved (warp vote).
  * RoPE: (cos, sin) computed once per (token, pair) into smem, shared by all Q rows and the new keys.
  * fold: the last-arriving CTA of each (request, head pair) merges the splits (threadfence + per-(b, hp) counter
    that it resets itself) and writes the gated, normalized BF16 output, so there is no combine launch; a request with
    a single split finalizes straight from smem. Used on sliding layers and full layers with B >= K1_FOLD_MIN_BATCH.
  * k1_combine (the non-fold path, e.g. B=1 full layers with ~250 splits): one CTA per (b, j, q head) with nsg split
    groups x 16 float4 lanes and an in-order smem reduce, instead of one warp looping over every split.
  * host tables (per (layer kind, q_len)): K1_ROUTE_MIN_BATCH, K1_FOLD_MIN_BATCH, K1_MIN_CHUNK_T, K1_CPS_T, _nsg().

k1_main  grid (Gx, Hkv/2), 256 threads; CTA = (split of one request's keys, KV-head pair): full 128 B K and V rows.
  split plan on device (k1_emulate.plan): C = smallest tile cap with sum_b max(1, ceil(tiles_b / C)) <= Gx,
  balanced boundaries; the last split also norms/ropes/quantizes the L new keys and values, stores them at
  cache_loc and attends them from a 16-key smem tile (the rows it writes are never read back in the same call).
  4-stage cp.async ring with zero-fill, page-table prefetch two tiles ahead; online softmax in log2 units with
  p = exp2(s - m + 8) in FP16. L=1: warp = (KV head, key quarter); QK and PV on m16n8k16 FP16 (Q fp16).

Every index formula (swizzles, fp16 and fp8 fragment maps, plan, masks, warp roles, both merge paths) is checked on
CPU by k1_emulate.py against a copy of the judge's reference. FALLBACK (or a failed build / self-test, or an
unsupported or un-routed shape) uses the rung-1 SGLang path (FA3).
"""

import hashlib
import time
from typing import TypedDict

import torch

FALLBACK = False               # True: always use the rung-1 FA3 path
SELF_TEST = True               # first eager call compares K1 against an FP32 reference; fallback on mismatch
R1_QK_FUSED_FP8_STAGE = True   # fallback = rung-1 path with L2 (GPU-verified 1.18-1.30x on verify shapes)

# Kernel constants (mirrored in CUDA_SRC and k1_emulate.py)
K1_TILE = 64
K1_STAGES = 4                  # default ring depth (per-shape: K1_STAGES_T)
K1_NEWK = 16

# Host tables, keyed by (layer kind, q_len). Expected wins (re-measure): K1 everywhere.
#   route: smallest batch that takes K1 (others: FA3). v0 measured L=1 wins at every B except B=1 full (combine,
#   fixed in v1); L=8 is predicted to win after the v1 redesign. Set an entry to 1 << 30 to send it to FA3.
K1_ROUTE_MIN_BATCH = {("sliding", 1): 1, ("full", 1): 1}  # L=8: v1 0.53-0.92x, v3 0.45-0.91x, v4 0.41-0.91x of rung-1 (box) -> FA3
#   fold (last-CTA merge, no combine launch) when batch >= this; else the separate k1_combine.
K1_FOLD_MIN_BATCH = {("sliding", 1): 1, ("sliding", 8): 1, ("full", 1): 8, ("full", 8): 8}
#   minimum tiles (64 keys) per split.
K1_MIN_CHUNK_T = {("sliding", 1): 2, ("sliding", 8): 4, ("full", 1): 2, ("full", 8): 2}
#   target CTAs per SM when sizing Gx (capped by the measured occupancy of k1_main<L>).
K1_CPS_T = {("sliding", 1): 1, ("sliding", 8): 2, ("full", 1): 2, ("full", 8): 2}  # lab: sliding L=1 1 CTA/SM -1.3/-0.6 us at B16/B32
#   cp.async ring depth (instantiated: L=1 {4, 6}, L=8 {3, 4}).
K1_STAGES_T = {("sliding", 1): 4, ("sliding", 8): 4, ("full", 1): 4, ("full", 8): 4}  # box: L=1 6 stages +0-1%
K1_COUNTERS = 1 << 16          # fold arrival counters (B * Hkv/2 must fit)
#   v9c (C1): fold merge through a thread-block cluster (DSMEM) for these (kind, q_len) entries; others keep the v4
#   global last-CTA merge. CS = min(K1_CLUSTER_MAX, SMs*cps / (B*Hkv/2), splits a sliding window can use), CS >= 2
#   (else v4 merge); one cluster per (request, head pair). L=8 / full: lab only (A/B first).
K1_CLUSTER_T = {("sliding", 1): True}
K1_CLUSTER_MAX = 16            # > 8 needs cudaFuncAttributeNonPortableClusterSizeAllowed (H100/H200 allow 16)
#   pad the dynamic smem so at most one CTA fits per SM (a cluster's CTAs then land on CS distinct SMs, as cps=1 does
#   for the v4 grid); if the padded clusters can't all be co-resident the host retries unpadded, then CS - 1.
K1_CLUSTER_PAD1_T = {("sliding", 1): True}
K1_PAD1_BYTES = 120 * 1024     # > half of the 228 KB SM shared memory
#   cluster barrier arrivals: 0 = release at both (PTX-ordered; each costs a MEMBAR.ALL.GPU in SASS), bit 0 = relaxed at
#   barrier 1 (after __syncthreads), bit 1 = relaxed at barrier 2 (after the DSMEM values are consumed). Lab A/B first.
K1_C1_RELAX = 0


def _nsg(B):
    """split groups per k1_combine CTA (16 float4 lanes each): many splits at B=1, few at large B."""
    return 32 if B == 1 else (8 if B <= 4 else 4)


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn

try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sglang.kernels.ops.kvcache.kvcache import store_cache
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
    from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
    _HAVE_SGL = True
except Exception as _ex:  # noqa: BLE001
    _HAVE_SGL = False
    print(f"[k1] SGLang/FA3 imports unavailable ({_ex!r}); fallback is the PyTorch path", flush=True)


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


# ----------------------------------------------------------------------------------------------
# CUDA
# ----------------------------------------------------------------------------------------------

CPP_SRC = r"""
#include <torch/extension.h>
void k1_forward(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor page_table,
                torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
                torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor out,
                torch::Tensor counters, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t W,
                int64_t page_shift, int64_t Gx, int64_t min_chunk, int64_t use_rope, double qk_sm, int64_t fold,
                int64_t nsg, int64_t stages, torch::Tensor err, int64_t cs, int64_t smem_pad, int64_t c1_flags);
int64_t k1_occupancy(int64_t L, int64_t ST);
int64_t k1_cluster_occupancy(int64_t L, int64_t ST, int64_t cs, int64_t smem_pad);
torch::Tensor k1_prof_read();
"""

CUDA_SRC = r"""
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

// Shared memory per (L, ST = cp.async ring depth). L=8 (v4) adds the fp16 K and V tiles (16 KB each).
// L=1: Q fp16 [2][8][64]; L=8: Q e4m3 [2][64][64]
template <int L> constexpr int qs_bytes() { return 2 * 8 * L * 64 * 2; }   // fp16 Q [2][8L][64]
template <int L> constexpr int cs_bytes() { return L * 32 * 8; }   // (cos, sin) per (token, pair)
template <int L, int ST> constexpr int ring_bytes() { return ST * STAGE_BYTES; }
template <int L, int ST> constexpr int off_vh() { return ring_bytes<L, ST>(); }
template <int L, int ST> constexpr int off_new() { return off_vh<L, ST>() + ((L == 8) ? 2 * VH_BYTES : 0); }
// L=8: the 16 KB fp16 Q staging aliases the kh/vh tiles (read into registers before the first conversion)
template <int L, int ST> constexpr int off_qs() { return (L == 8) ? off_vh<L, ST>() : off_new<L, ST>() + NEW_BYTES; }
template <int L, int ST> constexpr int off_cs() { return off_new<L, ST>() + NEW_BYTES + ((L == 8) ? 0 : qs_bytes<L>()); }
static_assert(2 * 8 * 8 * 64 * 2 <= 2 * VH_BYTES, "L=8 Q staging fits in kh/vh");
template <int L, int ST> constexpr int off_aux() { return off_cs<L, ST>() + cs_bytes<L>(); }
template <int L, int ST> constexpr int off_plan() { return off_aux<L, ST>() + ((L == 8) ? 0 : AUX_BYTES); }
template <int L, int ST> constexpr int smem_bytes() { return off_plan<L, ST>() + 64; }

// V ring swizzle (16-B chunk ch of key x). L=1: read by v_frags (all warps). L=8: read only by the loader thread
// that copied it (converted to fp16 right after its own cp.async group completes).
// K ring swizzle. L=1: read per lane by tile_compute1. L=8: read only by the loader thread (conv_kv).
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
  int* err;                   // v9c: [>= 1] error words, zero between calls; err[0] |= 1: cluster shape mismatch
  int B, Hq, Hkv, W, page_shift, pt_stride, Gx, min_chunk, use_rope, fold;
  int cs;                     // v9c: CTAs per cluster (= max splits per (request, head pair)); 0 = v4 global merge
  int c1_flags;               // v9c lab: bit 0 = relaxed arrive at cluster barrier 1, bit 1 = at barrier 2 (K1_C1_RELAX)
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

// low 16 bits = two e4m3 (low byte -> low half) -> f16x2, exact
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

// FP8 e4m3 x e4m3 -> f32 (sm_89+). A 16x32 row: a0 (row g, k 4t..), a1 (row g+8), a2 (row g, k 16+4t..), a3.
// B 32x8 col: b0 (k 4t..4t+3, n g), b1 (k 16+4t.., n g). Lowest k in the lowest byte.
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

template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if K1_ARCH_OK
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

// Interleaved RoPE of one (even, odd) pair with a (cos, sin) table entry.
__device__ __forceinline__ float2 rope_tab(float2 x, float2 cs) {
  return make_float2(x.x * cs.x - x.y * cs.y, x.x * cs.y + x.y * cs.x);
}

__device__ __forceinline__ float ldf(const float* a, bool cg) { return cg ? __ldcg(a) : *a; }
__device__ __forceinline__ float4 ldf4(const float* a, bool cg) {
  return cg ? __ldcg(reinterpret_cast<const float4*>(a)) : *reinterpret_cast<const float4*>(a);
}

__device__ __forceinline__ float sigmoidf_(float g) { return 1.f / (1.f + expf(-g)); }

// ---------------------------------------------------------------------------------------------
// v9c: thread-block cluster primitives (sm_90, PTX ISA 7.8+). The .aligned forms are always executed CTA-uniformly.
// ---------------------------------------------------------------------------------------------
#ifndef K1_CLUSTER
#define K1_CLUSTER 1                                 // 0: compile out the cluster merge (k1_v4 device code only)
#endif
#if K1_CLUSTER
__device__ __forceinline__ unsigned cl_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ unsigned cl_size() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_nctarank;\n" : "=r"(r));
  return r;
}
// release: this thread's prior smem writes (partials) / DSMEM reads are ordered before the arrival. SASS (nvcc 13.0,
// sm_90 and sm_90a): MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; UCGABAR_ARV -- a GPU-scope fence (there is no cluster-scope
// MEMBAR; fence.acq_rel.cluster lowers the same way).
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
// relaxed arrival (lab, K1_C1_RELAX): UCGABAR_ARV only. Barrier 1 is preceded by __syncthreads (the partials are in
// this SM's shared memory, which serves the peers' DSMEM reads); barrier 2 follows the consumption of every DSMEM
// load's value. Not ordered by the PTX memory model, hence off by default.
__device__ __forceinline__ void cl_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
// acquire: the peers' released writes are visible after the wait. SASS: UCGABAR_WAIT; CCTL.IVALL (L1 invalidated:
// global values the merge needs are loaded before barrier 1).
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_arrive_sel(bool relaxed) {
  if (relaxed) cl_arrive_relaxed();
  else cl_arrive();
}
// shared::cta address -> the same offset in CTA `rank` of this cluster (shared::cluster window)
__device__ __forceinline__ unsigned cl_map(unsigned saddr, unsigned rank) {
  unsigned r;
  asm("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(saddr), "r"(rank));   // pure address arithmetic
  return r;
}
__device__ __forceinline__ float cl_ld(unsigned a) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ float4 cl_ld4(unsigned a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
  return v;
}
#endif

// ---------------------------------------------------------------------------------------------
// Split plan (k1_emulate.plan): warp-cooperative, all 32 lanes.
// ---------------------------------------------------------------------------------------------
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

// Smallest per-split tile cap C with sum_b split_count(nt_b, C) <= Gx. Binary search inside
// [max(min_chunk, ceil(T/Gx)), ceil(T/(Gx-B))]; the upper end always fits (Gx > B).
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

// by_cta: find the request whose splits contain CTA `key`; else: request b = key.
// res = {b (-1 if none), split index, S_b, off_b, tiles_b}
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

// ---------------------------------------------------------------------------------------------
// Tile compute (k1_emulate.emulate_k1.process).
//   K stage: key row x at x*ROWB, 16-B chunk c (head c>>2) at physical chunk c ^ (4*(x&1)).
//   V stage: chunk c at c ^ (2*((x>>1)&3)).
//   PV (both L): k-step kk = keys kw+16kk..+15; lane loads 8 B (dims 8g..8g+7) of keys 2t,2t+1,2t+8,2t+9;
//       n-tile jn col n <-> output dim 8n + jn.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint2 vld(const uint8_t* Vbuf, int x, int h, int g) {
  return *reinterpret_cast<const uint2*>(Vbuf + x * ROWB + 16 * ((4 * h + (g >> 1)) ^ (2 * ((x >> 1) & 3))) +
                                         8 * (g & 1));
}

// V B-fragments of one PV k-step: w01/w23[i] hold (key a, key b) byte pairs of dims 8g+2i, 8g+2i+1.
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

// L=1: QK on m16n8k16 FP16 (Q fp16 frags qf[s][4]; dims 16t+4s+{0,1} / +{2,3}); 8 of 16 rows real.
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

// L=8 (v4): warp = (KV head h = warp>>2, m-tile mt = warp&3: rows 16mt..16mt+15 = tokens 2mt (rows g) and 2mt+1
// (rows g+8) x 8 q heads), ALL keys of the tile. Both K and V of every tile are converted ONCE (e4m3 -> fp16, exact)
// by the loader threads into the CTA's fp16 tiles Kh / Vh; QK and PV are m16n8k16 FP16 with B fragments from
// ldmatrix (K: non-trans, V: .trans). Q is fp16 (normalized, roped; finer than the E4M3 floor). (v3's QK on
// mma.sync e4m3 was emulated by ptxas: F2FP unpack + 2 HMMA per mma, the K bytes unpacked by all 4 m-tile warps;
// that was the whole L=8 cost: B32 full 433 us with QK, 222 us without.)
//   fp16 tile (Kh and Vh alike): key x row = 256 B: [head hh][8 chunks of 8 dims]; chunk c16 at physical
//   c16 ^ (x&7) ^ (2hh). PV n-tile jn, C col 2t+e <-> output dim 8jn+2t+e.
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

// one 16-B chunk of 16 e4m3 (dims 16(ch&3)..+15 of head ch>>2 of key x) -> 2 fp16 chunks of the tile
__device__ __forceinline__ void conv_chunk(const uint4 w, uint8_t* dst, int x, int ch) {
  uint4 a, c;
  a.x = e4m3x2_to_h2(w.x); a.y = e4m3x2_to_h2(w.x >> 16); a.z = e4m3x2_to_h2(w.y); a.w = e4m3x2_to_h2(w.y >> 16);
  c.x = e4m3x2_to_h2(w.z); c.y = e4m3x2_to_h2(w.z >> 16); c.z = e4m3x2_to_h2(w.w); c.w = e4m3x2_to_h2(w.w >> 16);
  const int hh = ch >> 2, c0 = 2 * (ch & 3);
  *reinterpret_cast<uint4*>(dst + vh_off(x, hh, c0)) = a;
  *reinterpret_cast<uint4*>(dst + vh_off(x, hh, c0 + 1)) = c;
}

// Thread (key x, part) converts its own two K chunks and two V chunks ch = 2part+u of one stage (K bytes at ksrc with
// the L=8 K ring swizzle ch ^ (x&1); V bytes at vsrc with the L=8 V swizzle ch ^ (x&1)) into Kh / Vh.
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
  // keys kb0 .. kb0 + 8NT - 1 of the fp16 tiles (tile_abs = absolute position of tile key 0)
  float sc[NT][4];
  // ldmatrix.x4 (non-trans) lane address: matrix q = lane>>3 -> dim chunk 4hs+q of key 8jn + (lane&7)
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
  // ldmatrix.x4.trans lane address: matrix q = lane>>3 -> keys +8*(q&1), dim chunk +(q>>1)
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

// ---------------------------------------------------------------------------------------------
// Split merge + finalize of all 16L rows of (b, hp) by one CTA (fold path; k1_emulate.merge_fold + finalize_rows).
// Items (row r_k = tid/16 + 16k, quad u = tid%16 = dims 4u..4u+3); the 16 quads of a row are one half-warp.
// po: [S][16L][64], pl: [S][16L] (global partials with cg = true, or the CTA's own smem staging, S = 1).
// v4: one pass with an online max over batches of UN splits, KG items at a time, so KG*UN float4 partial loads are
// in flight per thread (lab k1_v4_lab_merge: a single 256-thread CTA reads freshly written L2 partials at 24 GB/s
// with 1 load in flight and 75-80 GB/s with 4-8; load flavour (default/cg/volatile/relaxed) and fences don't matter).
// Gate values are loaded up front.
// ---------------------------------------------------------------------------------------------
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

#if K1_CLUSTER
// ---------------------------------------------------------------------------------------------
// v9c cluster merge (k1_c1_emulate.merge_cluster): rank rk of the S >= 2 splits of (b, hp) merges rows
// [rk*16L/S, (rk+1)*16L/S) of the 16L rows, reading split s's partial from CTA rank s of the cluster (its ost / lse_s
// at the same smem offsets: every CTA runs the same layout). Items (row r = r_lo + tid/16 + 16k, quad u = tid%16) as in
// merge_finalize, so the 16 quads of a row are one half-warp; a CTA holds <= 8L rows (S >= 2), KC = ceil(8L/16) items
// per thread, each merged and stored before the next (register budget at L=8). Per item the math is merge_finalize's,
// op for op (online max over batches of UN splits in order
// s = 0..S-1, ex2 weights, fmaf accumulation, vsc / ws, half-warp RMS, sigmoid gate), so the output is bit-identical to
// the v4 fold merge whenever the two plans give the same splits (all scored sliding shapes: k1_c1_emulate).
// Gate values come in through graw and v_scale through vsc (both loaded before the cluster barrier, whose acquire
// invalidates L1). The second cluster barrier is ARRIVED here, once this CTA's DSMEM reads are consumed; the caller
// waits on it before exiting.
// ---------------------------------------------------------------------------------------------
template <int L> constexpr int c1_kc() { return (8 * L + 15) / 16; }

__device__ __forceinline__ int c1_rlo(int rk, int S, int rows) { return (rk * rows) / S; }

// gate values of this thread's merge items (zero for items past the rank's rows)
template <int L>
__device__ __forceinline__ void c1_gates(const Params& p, const int rk, const int S, const int b, const int hp,
                                         uint2 (&graw)[c1_kc<L>()]) {
  constexpr int ROWS = 16 * L, R8 = 8 * L;
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  const int r_lo = c1_rlo(rk, S, ROWS), r_hi = c1_rlo(rk + 1, S, ROWS);
  const int r0 = threadIdx.x >> 4, u = threadIdx.x & 15;
#pragma unroll
  for (int k = 0; k < c1_kc<L>(); ++k) {
    const int r = r_lo + r0 + 16 * k;
    graw[k] = make_uint2(0u, 0u);
    if (r < r_hi) {
      const int hh = r / R8, rr = r - hh * R8, j = rr >> 3, gi = rr & 7;
      graw[k] = *reinterpret_cast<const uint2*>(p.qkv + ((size_t)b * L + j) * stride + (size_t)(p.Hq + 2 * p.Hkv) * 64 +
                                                (size_t)((2 * hp + hh) * 8 + gi) * 64 + 4 * u);
    }
  }
}

template <int L>
__device__ __forceinline__ void merge_cluster(const Params& p, const float* ost, const float* lse_s, const int S,
                                              const int rk, const int b, const int hp,
                                              const uint2 (&graw)[c1_kc<L>()], const float vsc) {
  constexpr int ROWS = 16 * L, R8 = 8 * L, KC = c1_kc<L>();
  constexpr int UN = (L == 1) ? 16 : 8;              // merge_finalize's batch
  const int r_lo = c1_rlo(rk, S, ROWS), r_hi = c1_rlo(rk + 1, S, ROWS);
  const int r0 = threadIdx.x >> 4, u = threadIdx.x & 15, wr = 2 * (threadIdx.x >> 5);
  const unsigned o_sa = (unsigned)__cvta_generic_to_shared(ost);
  const unsigned l_sa = (unsigned)__cvta_generic_to_shared(lse_s);
#pragma unroll
  for (int k = 0; k < KC; ++k) {
    const int r = r_lo + r0 + 16 * k;
    const bool mine = r < r_hi;
    const bool wk = r_lo + wr + 16 * k < r_hi;        // warp-uniform: the warp's first row is one of this rank's
    float M = neg_inf(), ws = 0.f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    if (wk) {
      for (int s0 = 0; s0 < S; s0 += UN) {
        float lv[UN];
        float4 ov[UN];
#pragma unroll
        for (int q = 0; q < UN; ++q) {
          const int s = s0 + q;
          if (mine && s < S) {
            lv[q] = cl_ld(cl_map(l_sa + 4u * (unsigned)r, (unsigned)s));
            ov[q] = cl_ld4(cl_map(o_sa + 4u * (unsigned)(r * 64 + 4 * u), (unsigned)s));
          } else {
            lv[q] = neg_inf();
            ov[q] = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        float bm = M;
#pragma unroll
        for (int q = 0; q < UN; ++q) bm = fmaxf(bm, lv[q]);
        const float Mu = (bm == neg_inf()) ? 0.f : bm;
        const float sc = ex2(M - Mu);                 // 0 when M was -inf
        acc.x *= sc; acc.y *= sc; acc.z *= sc; acc.w *= sc;
        ws *= sc;
#pragma unroll
        for (int q = 0; q < UN; ++q) {
          const float w = ex2(lv[q] - Mu);
          ws += w;
          acc.x = fmaf(w, ov[q].x, acc.x); acc.y = fmaf(w, ov[q].y, acc.y);
          acc.z = fmaf(w, ov[q].z, acc.z); acc.w = fmaf(w, ov[q].w, acc.w);
        }
        M = bm;
      }
    }
    // barrier 2, after the last item's DSMEM reads (KC = 1, i.e. L = 1: before the epilogue); CTA-uniform
    if (k == KC - 1) cl_arrive_sel(p.c1_flags & 2);
    if (wk) {                                         // half-warp shuffles below
      float4 a4 = acc;
      const float scl = (ws > 0.f) ? vsc / ws : 0.f;
      a4.x *= scl; a4.y *= scl; a4.z *= scl; a4.w *= scl;
      float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
      ss += __shfl_xor_sync(FULL, ss, 8);
      ss += __shfl_xor_sync(FULL, ss, 4);
      ss += __shfl_xor_sync(FULL, ss, 2);
      ss += __shfl_xor_sync(FULL, ss, 1);
      const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      if (mine) {
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
}
#endif

// ---------------------------------------------------------------------------------------------
// k1_main
// ---------------------------------------------------------------------------------------------
#ifndef K1_L8_MINB
#define K1_L8_MINB 2
#endif

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

  reinterpret_cast<uint4*>(newk)[tid] = make_uint4(0u, 0u, 0u, 0u);   // zero newk + newv
#if K1_CLUSTER
  if (p.cs) {
    // v9c cluster-shaped plan (k1_c1_emulate.cluster_plan): request b = cluster index c / CS, split = rank; b's
    // tiles split at >= min_chunk tiles per split into S_b = ceil(nt / C_b) <= CS splits (C_b = max(min_chunk,
    // ceil(nt / CS))), boundaries balanced as in the v4 plan. Ranks >= S_b are idle. A launch whose cluster shape
    // isn't (CS, 1, 1) flags err[0] and every CTA returns before any cluster barrier (cl_size is launch-uniform).
    if (tid == 0) {
      const int CS = p.cs, bq = c / CS, rk = c - bq * CS;
      const bool shape_ok = (cl_size() == (unsigned)CS);
      if (!shape_ok || cl_rank() != (unsigned)rk) atomicOr(p.err, 1);
      const int nt = old_tiles(__ldg(p.seq_lens + bq), p.W);
      const int C = max(p.min_chunk, (nt + CS - 1) / CS);
      plan_s[0] = shape_ok ? bq : -1; plan_s[1] = rk; plan_s[2] = split_count(nt, C); plan_s[3] = nt;
      plan_s[4] = bq * CS;
    }
  } else
#endif
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
  const int s_idx = plan_s[1], S_b = plan_s[2], nt_b = plan_s[3], off_b = plan_s[4];
#if K1_CLUSTER
  if (p.cs && s_idx >= S_b) {                         // idle rank (short request): no keys, both cluster barriers
    cl_arrive_sel(p.c1_flags & 1);
    cl_wait();
    cl_arrive_sel(p.c1_flags & 2);
    cl_wait();
    return;
  }
  // cluster merge items of this rank and their gate values: loaded now (L=1, 2 registers), consumed after the
  // barrier, so the gate's DRAM round trip is off the merge tail; L=8 (4 items) loads them after the tile loop.
  uint2 c1g[c1_kc<L>()];
  if (L == 1 && p.cs && S_b > 1) c1_gates<L>(p, s_idx, S_b, b, hp, c1g);
#endif
  // balanced split boundaries: tiles [s*nt/S, (s+1)*nt/S), each <= C tiles
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

  // ---- loader: thread -> key lk of the stage, 16-B chunks 2*part, 2*part+1 of K and of V
  const int lk = tid >> 2, part = tid & 3;
  auto pe_load = [&](int i) -> int {
    const int absk = kstart + (t0 + i) * TILE + lk;
    int pe = -1;
    if (i < ntl && absk < seq) pe = __ldg(pt + (absk >> pshift));
    return pe;
  };
  auto issue = [&](int i, int pe) {
    uint8_t* stg = ring + (i % ST) * STAGE_BYTES;
    const int absk = kstart + (t0 + i) * TILE + lk;
    const bool valid = pe >= 0;
    const size_t goff = valid ? (size_t)((pe << pshift) | (absk & pmask)) * rowB : (size_t)0;
    const int nb = valid ? 16 : 0;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int ch = 2 * part + u;
      cp_async16(stg + lk * ROWB + 16 * kswz<L>(lk, ch), kc + goff + 16 * ch, nb);
      cp_async16(stg + TILE * ROWB + lk * ROWB + 16 * vswz<L>(lk, ch), vc + goff + 16 * ch, nb);
    }
  };
  {
    int pe0[ST - 1];
#pragma unroll
    for (int i = 0; i < ST - 1; ++i) pe0[i] = pe_load(i);
#pragma unroll
    for (int i = 0; i < ST - 1; ++i) {
      if (i < ntl) issue(i, pe0[i]);
      cp_async_commit();
    }
  }
  int pfA = pe_load(ST - 1);
  int pfB = pe_load(ST);

  // ---- RoPE (cos, sin) table for the block's L positions: angle = fp32(seq + j) * inv_freq[i], rounded once
  if (p.use_rope && tid < L * 32) {
    const int j = tid >> 5, i = tid & 31;
    const float ang = __fmul_rn((float)(seq + j), __ldg(p.inv_freq + i));
    float sn, cs;
    sincosf(ang, &sn, &cs);
    cst[tid] = make_float2(cs, sn);
  }
  __syncthreads();

  // ---- Q prep: RMSNorm (+RoPE); fp16 (L=1) or e4m3 (L=8, normalized+roped value, before x5.473) rows in smem
  {
#pragma unroll
    for (int k = 0; k < RPW; ++k) {
      const int R = warp + k * NWARP;
      const int hh = R / R8, r = R - hh * R8, j = r >> 3;
      float2 x = xv[k];
      const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
      x.x *= rs; x.y *= rs;
      if (p.use_rope) x = rope_tab(x, cst[j * 32 + lane]);
      reinterpret_cast<uint32_t*>(qs)[(hh * R8 + r) * 32 + lane] = pack_h2(x.x, x.y);   // fp16 (both L)
    }
  }
  // ---- new tokens' K/V (last split only): cache rows at cache_loc + the 16-key smem tile
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
      const int bb = 64 * hh + 2 * lane;           // logical byte in the 128-B row
      const int ch = bb >> 4;
      const int pc = (kind == 0) ? kswz<L>(j, ch) : vswz<L>(j, ch);
      *reinterpret_cast<uint16_t*>(((kind == 0) ? newk : newv) + j * ROWB + 16 * pc + (bb & 15)) = (uint16_t)q8;
    }
  }
  __syncthreads();
  K1P(2);

  const int g = lane >> 2, t = lane & 3;
  const float cl2 = p.qk_sm * __ldg(p.k_scale) * LOG2E;
  float* ost;     // [ROWS][64] normalized partial O
  float* lse_s;   // [ROWS]

  if constexpr (L == 1) {
    // ---- L=1 roles: warp = (KV head h = warp&1, key quarter q4 = warp>>1)
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
      __syncthreads();
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
    // merge the 4 key quarters of each head (order q4 = 0..3)
    float* ow = reinterpret_cast<float*>(ring);   // [8 warps][8 rows][64]
    float* mw = ow + NWARP * 8 * 64;              // [8][8]
    float* lw = mw + NWARP * 8;                   // [8][8]
    ost = lw + NWARP * 8;                         // [16][64]
    lse_s = ost + ROWS * 64;                      // [16]
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
      for (int e = 0; e < 2; ++e) ow[(warp * 8 + g) * 64 + 8 * (2 * t + e) + jn] = o[jn][e];
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
        acc += f * ow[(w * 8 + r) * 64 + d];
      }
      ost[(hh * 8 + r) * 64 + d] = (lt > 0.f) ? acc / lt : 0.f;
      if (d == 0) lse_s[hh * 8 + r] = (lt > 0.f) ? M + log2f(lt) - P_OFFSET : neg_inf();
    }
  } else {
    // ---- L=8 (v4) roles: warp = (KV head h = warp>>2, m-tile mt = warp&3), all keys of every tile
    const int h = warp >> 2, mt = warp & 3, j0 = 2 * mt;
    uint8_t* kh = smem + off_vh<L, ST>();             // fp16 K tile (16 KB); the fp16 Q staging aliases kh/vh
    uint8_t* vh = kh + VH_BYTES;                      // fp16 V tile (16 KB)
    uint32_t qf[4][4];                                // fp16 Q A-fragments, k-step s: dims 16s + {2t, 2t+8}
    {
      const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs) + (h * R8 + 16 * mt + g) * 32;
#pragma unroll
      for (int s4 = 0; s4 < 4; ++s4) {
        qf[s4][0] = qa[8 * s4 + t];
        qf[s4][1] = qa[8 * 32 + 8 * s4 + t];
        qf[s4][2] = qa[8 * s4 + 4 + t];
        qf[s4][3] = qa[8 * 32 + 8 * s4 + 4 + t];
      }
    }
    float o[8][4];
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) { o[jn][0] = 0.f; o[jn][1] = 0.f; o[jn][2] = 0.f; o[jn][3] = 0.f; }
    float m[2] = {neg_inf(), neg_inf()};
    float l[2] = {0.f, 0.f};
    for (int i = 0; i < ntl; ++i) {
      cp_async_wait<ST - 2>();
      __syncthreads();                                // A: tile i-1 consumed (kh/vh free; i = 0: Q frags read)
      const uint8_t* stg = ring + (i % ST) * STAGE_BYTES;
      conv_kv(stg, stg + TILE * ROWB, kh, vh, lk, part);   // own chunks of stage i (complete for this thread)
      {
        // the stage refilled here, (i-1) % ST, was read only by its own loader threads (conversion of tile i-1)
        const int nx = i + ST - 1;
        if (nx < ntl) issue(nx, pfA);
        cp_async_commit();
        pfA = pfB;
        pfB = pe_load(i + ST + 1);
      }
      __syncthreads();                                // B: kh/vh hold tile i
      const int ts = kstart + (t0 + i) * TILE;
      const bool need = (ts + TILE > seq) || (W > 0 && ts < seq + L - W);
      // two 32-key halves (register budget: 2 CTAs/SM at <= 128 regs)
      tile_compute8<4>(kh, vh, h, g, t, lane, qf, o, m, l, cl2, need, ts, seq, seq, W, j0, 0);
      tile_compute8<4>(kh, vh, h, g, t, lane, qf, o, m, l, cl2, need, ts, seq, seq, W, j0, 32);
    }
    cp_async_wait<0>();
    __syncthreads();
    K1P(3);
    if (last) {                                       // CTA-uniform
      if (tid < NEWK * 4) conv_kv(newk, newv, kh, vh, tid >> 2, tid & 3);
      __syncthreads();
      tile_compute8<2>(kh, vh, h, g, t, lane, qf, o, m, l, cl2, true, seq, seq + L, seq, W, j0, 0);
    }
    ost = reinterpret_cast<float*>(ring);            // [128][64] (ring is drained; newk/vh are separate)
    lse_s = ost + ROWS * 64;                          // [128]
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      float lt = l[rb];
      lt += __shfl_xor_sync(FULL, lt, 1);
      lt += __shfl_xor_sync(FULL, lt, 2);
      const float inv = (lt > 0.f) ? 1.f / lt : 0.f;
      const int row = h * R8 + 16 * mt + 8 * rb + g;
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        *reinterpret_cast<float2*>(ost + row * 64 + 8 * jn + 2 * t) =
            make_float2(o[jn][2 * rb] * inv, o[jn][2 * rb + 1] * inv);
      }
      if (t == 0) lse_s[row] = (lt > 0.f) ? m[rb] + log2f(lt) - P_OFFSET : neg_inf();
    }
  }
  __syncthreads();
  K1P(4);

#if K1_CLUSTER
  // ---- v9c output: cluster merge. Barrier 1 publishes every split's smem partial to the cluster; each rank merges
  // its rows from the peers' smem (S_b == 1: rank 0 alone, straight from its own smem as v4); barrier 2 keeps every
  // CTA resident until its peers' DSMEM reads are done (arrived inside merge_cluster, waited here).
  if (p.cs) {
    const int rk = plan_s[1];                         // re-read (smem): no extra register live across the tile loop
    if (L == 8 && S_b > 1) c1_gates<L>(p, rk, S_b, b, hp, c1g);
    const float c1vsc = __ldg(p.v_scale);
    cl_arrive_sel(p.c1_flags & 1);
    cl_wait();
    K1P(5);
    if (S_b > 1) {
      merge_cluster<L>(p, ost, lse_s, S_b, rk, b, hp, c1g, c1vsc);
    } else {
      merge_finalize<L>(p, ost, lse_s, 1, b, hp, false);
      cl_arrive_sel(p.c1_flags & 2);
    }
    K1P(6);
    cl_wait();
    return;
  }
#endif
  // ---- output: fold (merge in the last-arriving CTA, or straight from smem when S_b == 1) or partials only
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

// ---------------------------------------------------------------------------------------------
// k1_combine (separate path): one CTA per (request b, token j, q head qh); thread = (split group sg = tid/16,
// quad u = tid%16); group sg sums splits s = sg, sg+nsg, ...; groups reduced in order (k1_emulate.merge_sep).
// ---------------------------------------------------------------------------------------------
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

// dynamic smem attribute raised to the largest size requested so far (v9c: the 1-CTA/SM padded cluster launch asks
// for more than smem_bytes); non-portable cluster sizes (> 8) enabled on first use.
template <int L, int ST>
static void ensure_smem_attr(int smem = k1::smem_bytes<L, ST>(), int cs = 0) {
  static int attr_smem = 0;
  static bool attr_np = false;
  if (smem > attr_smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::k1_main<L, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr_smem = smem;
  }
  if (cs > 8 && !attr_np) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::k1_main<L, ST>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr_np = true;
  }
}

// v9c: one launch config for the cluster path (grid x = B * cs, cluster (cs, 1, 1)); at[] must outlive the launch call
static void cluster_cfg(cudaLaunchConfig_t& cfg, cudaLaunchAttribute (&at)[1], dim3 grid, int smem, int cs,
                        cudaStream_t st) {
  cfg = cudaLaunchConfig_t{};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(k1::NTHR, 1, 1);
  cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = st;
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = (unsigned)cs;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
}

template <int L, int ST>
static void launch_main(const k1::Params& p, dim3 grid, cudaStream_t st, int smem = k1::smem_bytes<L, ST>()) {
  ensure_smem_attr<L, ST>(smem, p.cs);
  if (p.cs > 0) {
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute at[1];
    cluster_cfg(cfg, at, grid, smem, p.cs, st);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k1::k1_main<L, ST>, p));
  } else {
    k1::k1_main<L, ST><<<grid, k1::NTHR, smem, st>>>(p);
  }
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

// v9c: clusters of cs CTAs (dynamic smem max(smem_bytes, smem_pad)) that can be co-resident on the device; 0 if the
// shape can't be launched at all (the host then keeps the v4 merge).
template <int L, int ST>
static int cl_occ_of(int cs, int smem_pad) {
  const int smem = smem_pad > k1::smem_bytes<L, ST>() ? smem_pad : k1::smem_bytes<L, ST>();
  ensure_smem_attr<L, ST>(smem, cs);
  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute at[1];
  cluster_cfg(cfg, at, dim3((unsigned)cs, 1, 1), smem, cs, 0);
  int n = 0;
  const cudaError_t e = cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(&k1::k1_main<L, ST>), &cfg);
  if (e != cudaSuccess) {
    (void)cudaGetLastError();                        // clear the (non-sticky) config error
    return 0;
  }
  return n;
}

// instantiated (L, ring stages): L=1 {4, 6}; L=8 {3, 4}
static bool stages_ok(int64_t L, int64_t ST) { return (L == 1) ? (ST == 4 || ST == 6) : (ST == 3 || ST == 4); }

// Resident k1_main CTAs per SM (registers + smem), used to size Gx so the grid is one wave.
int64_t k1_occupancy(int64_t L, int64_t ST) {
  TORCH_CHECK(stages_ok(L, ST), "k1: (L, stages) not instantiated");
  if (L == 1) return (int64_t)(ST == 4 ? occ_of<1, 4>() : occ_of<1, 6>());
  return (int64_t)(ST == 3 ? occ_of<8, 3>() : occ_of<8, 4>());
}

int64_t k1_cluster_occupancy(int64_t L, int64_t ST, int64_t cs, int64_t smem_pad) {
  TORCH_CHECK(stages_ok(L, ST), "k1: (L, stages) not instantiated");
  TORCH_CHECK(cs >= 2 && cs <= 16, "k1: cluster size");
#if K1_CLUSTER
  if (L == 1) return (int64_t)(ST == 4 ? cl_occ_of<1, 4>((int)cs, (int)smem_pad) : cl_occ_of<1, 6>((int)cs, (int)smem_pad));
  return (int64_t)(ST == 3 ? cl_occ_of<8, 3>((int)cs, (int)smem_pad) : cl_occ_of<8, 4>((int)cs, (int)smem_pad));
#else
  return 0;
#endif
}

// lab only (K1_PROF builds): read and clear the per-CTA phase stamps; empty tensor otherwise
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
                int64_t nsg, int64_t stages, torch::Tensor err, int64_t cs, int64_t smem_pad, int64_t c1_flags) {
  TORCH_CHECK(L == 1 || L == 8, "k1: q_len must be 1 or 8");
  TORCH_CHECK(stages_ok(L, stages), "k1: (L, stages) not instantiated");
  TORCH_CHECK(Hq == 8 * Hkv && Hkv % 2 == 0 && Hkv > 0, "k1: need Hq == 8*Hkv and even Hkv");
  TORCH_CHECK(Gx > B, "k1: Gx must exceed B");
  // v9c cluster path: fold only, grid x = B * cs (one cluster per request), no global partials or counters
  TORCH_CHECK(cs == 0 || (K1_CLUSTER && fold != 0 && cs >= 2 && cs <= 16 && Gx == B * cs),
              "k1: cluster path needs fold, 2 <= cs <= 16 and Gx == B*cs (and a K1_CLUSTER build)");
  TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, "k1: err");
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
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && (cs > 0 || part_o.numel() >= (Hkv / 2) * Gx * 16 * L * 64),
              "k1: part_o");
  TORCH_CHECK(part_lse.scalar_type() == at::kFloat && (cs > 0 || part_lse.numel() >= (Hkv / 2) * Gx * 16 * L),
              "k1: part_lse");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * L * Hq * 64, "k1: out");
  TORCH_CHECK(counters.scalar_type() == at::kInt && (fold == 0 || cs > 0 || counters.numel() >= B * (Hkv / 2)),
              "k1: counters");
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
  p.err = err.data_ptr<int>();
  p.cs = (int)cs;
  p.c1_flags = (int)c1_flags;
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.W = (int)W; p.page_shift = (int)page_shift;
  p.pt_stride = (int)page_table.size(1); p.Gx = (int)Gx; p.min_chunk = (int)min_chunk; p.use_rope = (int)use_rope;
  p.fold = (int)fold;
  p.qk_sm = (float)qk_sm;
  const dim3 grid((unsigned)Gx, (unsigned)(Hkv / 2));
  if (cs > 0) {                                       // v9c cluster path (padded smem: 1 CTA per SM)
    if (L == 1) {
      if (stages == 4) launch_main<1, 4>(p, grid, st, std::max(k1::smem_bytes<1, 4>(), (int)smem_pad));
      else launch_main<1, 6>(p, grid, st, std::max(k1::smem_bytes<1, 6>(), (int)smem_pad));
    } else {
      if (stages == 3) launch_main<8, 3>(p, grid, st, std::max(k1::smem_bytes<8, 3>(), (int)smem_pad));
      else launch_main<8, 4>(p, grid, st, std::max(k1::smem_bytes<8, 4>(), (int)smem_pad));
    }
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


def _build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CPP_SRC + CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"k1_v3_{tag}",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["k1_forward", "k1_occupancy", "k1_cluster_occupancy", "k1_prof_read"],
        # pinning the arch also stops cpp_extension from following the image's TORCH_CUDA_ARCH_LIST
        # (any user flag containing "arch" disables its own arch flags): one sm_90 compile, sm_90 ptxas lines
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90,code=sm_90"],
        verbose=True,
    )
    print(f"[k1] compiled in {time.time() - t0:.1f}s", flush=True)
    return mod


_MOD = None
if not FALLBACK:
    try:
        _MOD = _build()
    except Exception as _ex:  # noqa: BLE001
        print("[k1] CUDA build FAILED, using the fallback path. Compiler output tail:\n" + str(_ex)[-12000:],
              flush=True)
        _MOD = None
print(f"[k1] v9c FALLBACK={FALLBACK} SELF_TEST={SELF_TEST} built={_MOD is not None} route={K1_ROUTE_MIN_BATCH}"
      f" fold={K1_FOLD_MIN_BATCH} min_chunk={K1_MIN_CHUNK_T} cps={K1_CPS_T} cluster={K1_CLUSTER_T}"
      f" pad1={K1_CLUSTER_PAD1_T}", flush=True)

_STATE = {"tested": False, "ok": False, "cluster_ok": {1: False, 8: False}}
_SMS = {}
_OCC = {}
_INV_FREQ = {}
_CNT = {}
_ERR = {}
_CL_OCC = {}   # (L, st, cs, pad, device) -> co-resident clusters
_CL_PICK = {}  # (kind, L, B, hp, cps, st, device) -> (cs, pad), logged once


def _sms(dev):
    key = str(dev)
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SMS[key]


def _occupancy(L, st, dev):
    """resident k1_main<L, st> CTAs per SM; queried once per (L, st, device), eagerly."""
    key = (L, st, str(dev))
    if key not in _OCC:
        _OCC[key] = max(1, int(_MOD.k1_occupancy(L, st)))
        print(f"[k1] k1_main<{L},{st}> occupancy {_OCC[key]} CTAs/SM", flush=True)
    return _OCC[key]


def _counters(dev):
    """Persistent zeroed int32 fold counters (self-resetting in the kernel). None if first needed under capture."""
    key = str(dev)
    if key not in _CNT:
        if torch.cuda.is_current_stream_capturing():
            if not _STATE.get("warned_cnt"):
                _STATE["warned_cnt"] = True
                print("[k1] fold counters first needed under graph capture: separate combine for this call", flush=True)
            return None
        _CNT[key] = torch.zeros(K1_COUNTERS, dtype=torch.int32, device=dev)
    return _CNT[key]


def _err(dev):
    """Persistent zeroed int32[4] error words (cluster path; nonzero = cluster shape mismatch). None under capture."""
    key = str(dev)
    if key not in _ERR:
        if torch.cuda.is_current_stream_capturing():
            return None
        _ERR[key] = torch.zeros(4, dtype=torch.int32, device=dev)
    return _ERR[key]


def _cluster_occ(L, st, cs, pad, dev):
    """co-resident clusters of cs CTAs (pad: dynamic smem K1_PAD1_BYTES); queried eagerly, cached. None if first
    needed under graph capture (the caller then keeps the v4 merge)."""
    key = (L, st, cs, pad, str(dev))
    if key not in _CL_OCC:
        if torch.cuda.is_current_stream_capturing():
            return None
        _CL_OCC[key] = int(_MOD.k1_cluster_occupancy(L, st, cs, K1_PAD1_BYTES if pad else 0))
    return _CL_OCC[key]


def _pick_cluster(cfg, L, B, hp, cps, st, min_chunk, pad, dev, cs_force=None):
    """(cs, pad) for the cluster merge, or (0, False) = v4 merge. cs = min(K1_CLUSTER_MAX, SMs*cps / (B*hp), useful
    splits of a sliding window), >= 2 (scored sliding L=1: B 1/16/32 -> 16/8/4, the v4 plan's splits). Every cluster
    must be co-resident (one wave): else unpadded, else cs - 1 (down to 2; a size the driver rejects reports 0)."""
    key = (_kind(cfg), L, B, hp, cps, st, min_chunk, pad, cs_force, str(dev))
    if key in _CL_PICK:
        return _CL_PICK[key]
    if cs_force:
        cap = int(cs_force)
    else:
        cap = min(K1_CLUSTER_MAX, (_sms(dev) * cps) // max(1, B * hp))
        if cfg["window"]:   # a sliding request has <= ceil((W-1)/TILE) tiles, so <= that / min_chunk useful splits
            nt_max = -(-(cfg["window"] - 1) // K1_TILE)
            cap = min(cap, -(-nt_max // min_chunk))
    cs = int(cap)
    need = 1 if cs_force else B * hp         # forced (self-test / lab): launchable is enough
    pick = (0, False)
    while cs >= 2:
        for pd in ((True, False) if pad else (False,)):
            occ = _cluster_occ(L, st, cs, pd, dev)
            if occ is None:
                return (0, False)            # not cached and capturing: v4 merge for this call, decide later
            if occ >= need:
                pick = (cs, pd)
                break
        if pick[0] or cs_force:
            break
        cs -= 1
    _CL_PICK[key] = pick
    print(f"[k1] cluster pick {_kind(cfg)} L={L} B={B} hp={hp} cps={cps} st={st}: cs={pick[0]} pad1={pick[1]}"
          f" (cap {cap})", flush=True)
    return pick


def _inv_freq(theta, dev):
    """[32] fp32 RoPE inverse frequencies, as the reference forms them (a per-theta constant)."""
    key = (float(theta), str(dev))
    if key not in _INV_FREQ:
        if theta:
            inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev).float() / 64))
        else:
            inv = torch.zeros(32, dtype=torch.float32, device=dev)
        _INV_FREQ[key] = inv.contiguous()
    return _INV_FREQ[key]


def _kind(cfg):
    return "sliding" if cfg["window"] else "full"


def _supported(data):
    cfg = data["config"]
    hq, hkv, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"]
    if not (cfg["head_dim"] == 64 and hq == 8 * hkv and hkv % 2 == 0 and cfg["q_len"] in (1, 8)
            and page > 0 and (page & (page - 1)) == 0):
        return False
    q, kc, vc = data["qkv"], data["k_cache"], data["v_cache"]
    if not (q.is_cuda and q.dtype == torch.bfloat16 and q.is_contiguous() and kc.dtype == FP8 and vc.dtype == FP8
            and kc.is_contiguous() and vc.is_contiguous()):
        return False
    if not (data["seq_lens"].dtype == torch.int32 and data["cache_loc"].dtype == torch.int64
            and data["page_table"].dtype == torch.int32):
        return False
    return True


def _routed(cfg):
    return cfg["batch"] >= K1_ROUTE_MIN_BATCH.get((_kind(cfg), cfg["q_len"]), 1 << 30)


def _k1(data, fold=None, min_chunk=None, cps=None, nsg=None, stages=None, cluster=None, cs=None, pad1=None,
        relax=None):
    """One K1 call. Overrides (lab / self-test): fold (bool), min_chunk, cps (target CTAs/SM), nsg, stages,
    cluster (bool: v9c cluster merge; default K1_CLUSTER_T and the self-test verdict), cs (force the cluster size),
    pad1 (bool: 1 CTA/SM smem padding; default K1_CLUSTER_PAD1_T), relax (cluster arrive flags; default K1_C1_RELAX)."""
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    key = (_kind(cfg), L)
    qkv = data["qkv"]
    dev = qkv.device
    hp = hkv // 2
    if fold is None:
        fold = B >= K1_FOLD_MIN_BATCH.get(key, 1 << 30)
    if min_chunk is None:
        min_chunk = K1_MIN_CHUNK_T.get(key, 2)
    if cps is None:
        cps = K1_CPS_T.get(key, 2)
    if nsg is None:
        nsg = _nsg(B)
    if stages is None:
        stages = K1_STAGES_T.get(key, K1_STAGES)
    if cluster is None:
        cluster = K1_CLUSTER_T.get(key, False) and _STATE["cluster_ok"].get(L, False)
    if pad1 is None:
        pad1 = K1_CLUSTER_PAD1_T.get(key, False)
    cps_eff = max(1, min(int(cps), _occupancy(L, int(stages), dev)))
    # v9c: cluster merge (fold semantics, no partials / counters); falls back to the v4 merge when no cluster shape fits
    ccs, cpad, err = 0, False, None
    if fold and cluster:
        err = _err(dev)
        if err is not None:
            ccs, cpad = _pick_cluster(cfg, L, B, hp, cps_eff, int(stages), int(min_chunk), bool(pad1), dev, cs)
    _STATE["last_pick"] = (ccs, cpad)
    if ccs:
        gx = B * ccs
        part_o = part_lse = torch.empty(1, dtype=torch.float32, device=dev)   # unused on the cluster path
        cnt = err
    else:
        cnt = _counters(dev) if fold else None
        if fold and (cnt is None or B * hp > cnt.numel()):
            fold = False
        if cnt is None:
            cnt = _CNT.get(str(dev))
            if cnt is None:
                cnt = torch.zeros(1, dtype=torch.int32, device=dev)   # unused when fold == 0
        gx = max(max(1, (_sms(dev) * cps_eff) // hp), B + 1)
        part_o = torch.empty((hp, gx, 16 * L, 64), dtype=torch.float32, device=dev)
        part_lse = torch.empty((hp, gx, 16 * L), dtype=torch.float32, device=dev)
    if err is None:
        err = cnt                                    # unused when cs == 0 (only its dtype/device are checked)
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    theta = cfg["rope_theta"]
    _MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                    data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], _inv_freq(theta, dev),
                    part_o, part_lse, out, cnt, B, L, hq, hkv, cfg["window"], cfg["page_size"].bit_length() - 1, gx,
                    int(min_chunk), 1 if theta else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]),
                    1 if fold else 0, int(nsg), int(stages), err, int(ccs), K1_PAD1_BYTES if cpad else 0,
                    int(K1_C1_RELAX if relax is None else relax))
    return out


# ----------------------------------------------------------------------------------------------
# Fallbacks: the rung-1 SGLang path (FA3), and plain PyTorch if SGLang is unavailable
# ----------------------------------------------------------------------------------------------

_NORMS = {}
_ROPE = {}
ROPE_POSITIONS = 1 << 21
STAGE_LOC_MAX = 1 << 16
_STAGE_LOC = {}


def _stage_loc(tokens, device):
    cache = _STAGE_LOC.get(device)
    if cache is None and not torch.cuda.is_current_stream_capturing():
        cache = _STAGE_LOC[device] = torch.arange(1, STAGE_LOC_MAX + 1, dtype=torch.int64, device=device)
    if cache is not None and tokens <= cache.shape[0]:
        return cache[:tokens]
    return torch.arange(1, tokens + 1, dtype=torch.int64, device=device)


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _fa3_path(data):
    """kernels/muse_attention_decode/r1_glue.py kernel_fn (rung 1), NUM_SPLITS_OVERRIDE empty."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    if R1_QK_FUSED_FP8_STAGE and not scaled:
        width = hkv * d
        k_stage = torch.empty((tokens + 1, width), dtype=FP8, device=device)
        v_stage = torch.empty((tokens + 1, width), dtype=FP8, device=device)
        kv_store = (k_stage, v_stage, _stage_loc(tokens, device))
        if cfg["rope_theta"]:
            q8, _ = _qk_norm(hq, hkv, d).forward_triton(
                qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device), kv_store)
        else:
            q8, _ = _qk_norm(hq, hkv, d).forward_triton(qkv, None, None, kv_store)
        store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8),
                    k_cache.view(-1, width).view(torch.uint8), v_cache.view(-1, width).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q_fp8 = q8.view(tokens, hq, d)
    else:
        if cfg["rope_theta"]:
            q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
        else:
            q, k = _qk_norm(hq, hkv, d)(qkv)
        v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
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
    out = flash_attn_with_kvcache(
        q=q_fp8, k_cache=k_cache, v_cache=v_cache, page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"], **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


def _rms_t(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope_t(x, pos, theta):
    """Interleaved RoPE on [..., heads, 64] with pos broadcast over the leading dims."""
    inv = _inv_freq(theta, x.device)
    ang = pos.float()[..., None] * inv
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    ev, od = x[..., 0::2], x[..., 1::2]
    return torch.stack([ev * cos - od * sin, ev * sin + od * cos], -1).flatten(-2)


def _torch_path(data):
    """Plain PyTorch (graph-capturable, slow): every position the page table can hold, masked."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    q_in, k_in, v_in, gate = data["qkv"].split([hq * d, hkv * d, hkv * d, hq * d], -1)
    pos = data["positions"].view(batch, length)
    q = _rms_t(q_in.reshape(batch, length, hq, d)) * cfg["qk_scale_factor"]
    k = _rms_t(k_in.reshape(batch, length, hkv, d))
    if cfg["rope_theta"]:
        q, k = _rope_t(q, pos, cfg["rope_theta"]), _rope_t(k, pos, cfg["rope_theta"])
    table = data["page_table"].long()
    slots = data["cache_loc"].view(batch, length)
    k_cache = data["k_cache"].view(-1, hkv, d)
    v_cache = data["v_cache"].view(-1, hkv, d)
    k_scale, v_scale = data["k_scale"], data["v_scale"]
    k_cache[slots] = (k / k_scale).to(FP8)
    v_cache[slots] = (v_in.reshape(batch, length, hkv, d).float() / v_scale).to(FP8)
    t = torch.arange(table.shape[1] * page, device=q.device)
    out = torch.empty((batch, length, hq, d), dtype=torch.float32, device=q.device)
    for b in range(batch):
        rows = table[b].clamp(min=0)[t // page] * page + t % page
        keys = k_cache[rows].float() * k_scale
        vals = v_cache[rows].float() * v_scale
        visible = t[None, :] <= pos[b][:, None]
        if cfg["window"]:
            visible &= t[None, :] > pos[b][:, None] - cfg["window"]
        att = torch.einsum("lkgd,skd->lkgs", q[b].view(length, hkv, hq // hkv, d), keys) * cfg["sm_scale"]
        att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
        out[b] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(length, hq, d)
    out = _rms_t(out) * torch.sigmoid(gate.float()).view(batch, length, hq, d)
    return out.reshape(batch, length, hq * d).to(torch.bfloat16)


def _fallback(data):
    return _fa3_path(data) if _HAVE_SGL else _torch_path(data)


# ----------------------------------------------------------------------------------------------
# Self-test (first eager call): K1 vs an FP32 reference that attends K1's own new entries
# ----------------------------------------------------------------------------------------------

def _mk_case(dev, B, ctx, L, sliding, page, kv_scale, seed, hq=16, hkv=2):
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


def _ref_attention(data, k_new, v_new, q_e4m3=False):
    """FP32 semantics of TASK.md, attending the given new entries (e4m3 bytes [B, L, Hkv, 64]). q_e4m3: queries
    rounded to E4M3 at the judge's FP8-query point (normalized, roped, before x qk_scale_factor)."""
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, W, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["window"], cfg["page_size"]
    qkv = data["qkv"].float()
    q = _rms_t(qkv[..., :hq * 64].reshape(B, L, hq, 64))
    seq_l = data["seq_lens"].long()
    pos = seq_l[:, None] + torch.arange(L, device=q.device)
    if cfg["rope_theta"]:
        q = _rope_t(q, pos, cfg["rope_theta"])
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
    out = _rms_t(out) * torch.sigmoid(gate)
    return out.reshape(B, L, hq * 64)


def _ref_new_kv(data):
    """The new entries as the reference quantizes them (fp32 -> e4m3 RNE; clamped like satfinite), as floats."""
    cfg = data["config"]
    B, L, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"].float()
    k = _rms_t(qkv[..., hq * 64:(hq + hkv) * 64].reshape(B, L, hkv, 64))
    if cfg["rope_theta"]:
        k = _rope_t(k, data["seq_lens"].long()[:, None] + torch.arange(L, device=k.device), cfg["rope_theta"])
    v = qkv[..., (hq + hkv) * 64:(hq + 2 * hkv) * 64].reshape(B, L, hkv, 64)
    q8 = [(x / sc).clamp(-448.0, 448.0).to(FP8).float() for x, sc in ((k, data["k_scale"]), (v, data["v_scale"]))]
    return q8[0], q8[1]


def _rel_rows(a, b):
    a, b = a.float().reshape(-1, a.shape[-1]), b.float().reshape(-1, b.shape[-1])
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row


def _case_errors(d, out, L, hkv):
    """(rel L2, worst row, new-K rel, new-V rel, finite) of one K1 output against the FP32 reference that attends the
    kernel's own new entries (fp16 Q at both L, as k1_v4)."""
    pos = d["seq_lens"].long()[:, None] + torch.arange(L, device=out.device)
    pg = d["config"]["page_size"]
    slots = torch.gather(d["page_table"].long(), 1, pos // pg) * pg + pos % pg
    k_new = d["k_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
    v_new = d["v_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
    ref = _ref_attention(d, k_new, v_new, q_e4m3=False)
    tot, row = _rel_rows(out, ref)
    kref, vref = _ref_new_kv(d)
    kt, _ = _rel_rows(k_new.view(FP8).float(), kref)
    vt, _ = _rel_rows(v_new.view(FP8).float(), vref)
    return tot, row, kt, vt, bool(torch.isfinite(out.float()).all())


def _self_test(dev):
    """Each case runs in both merge modes (fold: last-CTA merge; separate: k1_combine) and twice in fold mode (the
    counters must have reset themselves). L=1 is compared with the FP32 reference; L=8 (E4M3 queries) with the
    reference at the same query rounding. New K/V bytes are compared with the reference's quantization.
    v9c: the v4 cases above run on the v4 merge (cluster=False); then _self_test_cluster decides the cluster route."""
    t0 = time.time()
    cases = (  # (B, ctx, L, sliding, page, kv_scale, seed, heads)
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
    try:
        cnt = _counters(dev)
        for (B, ctx, L, sliding, page, kvs, seed, heads) in cases:
            for mode in ("fold", "sep", "fold"):
                d = _mk_case(dev, B, ctx, L, sliding, page, kvs, seed, heads[0], heads[1])
                out = _k1(d, fold=(mode == "fold"), cluster=False)
                torch.cuda.synchronize(dev)
                tot, row, kt, vt, finite = _case_errors(d, out, L, heads[1])
                cnt_ok = cnt is None or int(cnt.abs().sum()) == 0
                print(f"[k1] self-test {mode:4s} B={B} ctx={ctx} L={L} {'sliding' if sliding else 'full'} page={page}"
                      f" kv_scale={kvs:g} heads={heads}: out rel {tot:.3e} worst row {row:.3e}; new K {kt:.3e}"
                      f" V {vt:.3e}; counters {'0' if cnt_ok else 'NOT RESET'}", flush=True)
                if not (finite and tot < 2e-2 and row < 6e-2 and kt < 3e-2 and vt < 3e-2 and cnt_ok):
                    raise RuntimeError("self-test mismatch")
        print(f"[k1] self-test passed in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[k1] self-test FAILED, using the fallback path: {ex!r}", flush=True)
        return False
    _self_test_cluster(dev)
    return True


def _self_test_cluster(dev):
    """v9c cluster merge vs the FP32 reference (same thresholds as the v4 cases) and vs the v4 fold merge of the same
    inputs (max |diff| printed; 0 whenever both plans give the same splits). Covers every cluster size, the padded
    launch, non-power-of-two sizes, idle ranks (short requests), S_b = 1 (rank 0 alone), 8 head pairs (grid y = 8), full
    layers and L=8 (lab). A forced size this device can't launch is skipped (the host never picks one).
    Sets _STATE["cluster_ok"][L]; a failure only disables the cluster route for that L. Also caches the cluster
    occupancy of every (L, stages, cs, pad) the host can pick, so no occupancy query happens under graph capture."""
    t0 = time.time()
    cl_cases = (  # (B, ctx, L, sliding, page, kv_scale, seed, heads, cs (None = host pick), pad1)
        (2, 3000, 1, True, 1, 1.0, 101, (16, 2), None, True),
        (2, 3000, 1, True, 1, 1.0, 101, (16, 2), 8, True),
        (2, 3000, 1, True, 1, 1.0, 101, (16, 2), 4, False),
        (2, 3000, 1, True, 1, 1.0, 101, (16, 2), 2, False),
        (2, 3000, 1, True, 1, 1.0, 101, (16, 2), 3, False),        # non-power-of-two sizes (portable / not)
        (2, 3000, 1, True, 1, 1.0, 101, (16, 2), 12, True),
        (3, 300, 1, True, 1, 1.0, 113, (16, 2), 16, True),         # 3..7 tiles: S_b 2..4, ranks >= S_b idle
        (3, 2047, 1, True, 64, 1.0 / 64, 112, (16, 2), None, True),  # window edge, kv scale 1/64
        (1, 9, 1, True, 64, 1.0, 107, (16, 2), 16, True),          # S_b = 1: rank 0 alone, 15 idle ranks
        (3, 1500, 1, False, 64, 1.0, 109, (128, 16), 4, False),    # full layer, 8 head pairs
        (1, 9000, 1, False, 1, 1.0, 110, (16, 2), 16, False),      # full, ~9 tiles per split
        (3, 2047, 8, True, 1, 1.0, 102, (16, 2), None, False),     # L=8 (lab path): host pick cs 8
        (2, 1800, 8, True, 1, 1.0 / 64, 105, (16, 2), 4, False),
        (2, 600, 8, False, 64, 1.0, 108, (128, 16), 2, False),
    )
    ok = {1: True, 8: True}
    tried = {1: False, 8: False}
    err = _err(dev)
    for (B, ctx, L, sliding, page, kvs, seed, heads, cs, pad) in cl_cases:
        if not ok[L]:
            continue
        try:
            d = _mk_case(dev, B, ctx, L, sliding, page, kvs, seed, heads[0], heads[1])
            ref_out = _k1(d, fold=True, cluster=False)
            d = _mk_case(dev, B, ctx, L, sliding, page, kvs, seed, heads[0], heads[1])
            out = _k1(d, fold=True, cluster=True, cs=cs, pad1=pad)
            torch.cuda.synchronize(dev)
            pick = _STATE["last_pick"]
            if not pick[0]:   # the host never picks an unlaunchable shape, so this is not a failure
                print(f"[k1] self-test clus B={B} ctx={ctx} L={L}: cs={cs} pad1={pad} not launchable here, skipped",
                      flush=True)
                continue
            tried[L] = True
            tot, row, kt, vt, finite = _case_errors(d, out, L, heads[1])
            dmax = float((out.float() - ref_out.float()).abs().max())
            e_ok = int(err.abs().sum()) == 0
            print(f"[k1] self-test clus B={B} ctx={ctx} L={L} {'sliding' if sliding else 'full'} page={page}"
                  f" kv_scale={kvs:g} heads={heads} cs={pick[0]} pad1={pick[1]}: out rel {tot:.3e} worst row"
                  f" {row:.3e}; new K {kt:.3e} V {vt:.3e}; vs v4 merge max|d| {dmax:.3e}; err {'0' if e_ok else 'SET'}",
                  flush=True)
            if not (finite and tot < 2e-2 and row < 6e-2 and kt < 3e-2 and vt < 3e-2 and e_ok):
                raise RuntimeError("cluster self-test mismatch")
        except Exception as ex:  # noqa: BLE001
            ok[L] = False
            print(f"[k1] cluster self-test FAILED for L={L}, v4 merge kept: {ex!r}", flush=True)
    for L in (1, 8):
        _STATE["cluster_ok"][L] = ok[L] and tried[L]
    try:   # occupancy of every (L, stages, cs, pad) the host may pick, so none is queried under capture
        occ = []
        for L, st in sorted({(k[1], v) for k, v in K1_STAGES_T.items()}):
            for c in range(2, K1_CLUSTER_MAX + 1):
                for pd in (True, False):
                    occ.append(f"L{L}/st{st}/cs{c}/{'pad' if pd else 'nopad'}={_cluster_occ(L, st, c, pd, dev)}")
        print("[k1] cluster occupancy (co-resident clusters): " + " ".join(occ), flush=True)
    except Exception as ex:  # noqa: BLE001
        _STATE["cluster_ok"] = {1: False, 8: False}
        print(f"[k1] cluster occupancy query FAILED, v4 merge kept: {ex!r}", flush=True)
    print(f"[k1] cluster self-test done in {time.time() - t0:.1f}s: cluster_ok={_STATE['cluster_ok']}", flush=True)


# ----------------------------------------------------------------------------------------------
# kernel_fn
# ----------------------------------------------------------------------------------------------

def kernel_fn(data: Inputs) -> torch.Tensor:
    if _MOD is None or not _supported(data) or not _routed(data["config"]):
        return _fallback(data)
    if not _STATE["tested"]:
        if torch.cuda.is_current_stream_capturing():
            if not _STATE.get("warned_capture"):
                _STATE["warned_capture"] = True
                print("[k1] first call arrived under graph capture before any eager call: FA3 fallback for this capture",
                      flush=True)
            return _fallback(data)
        _STATE["ok"] = _self_test(data["qkv"].device) if SELF_TEST else True
        _STATE["tested"] = True
    if not _STATE["ok"]:
        return _fallback(data)
    return _k1(data)

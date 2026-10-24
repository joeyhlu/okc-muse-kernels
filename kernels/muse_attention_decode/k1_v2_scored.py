# okc: problem=muse_attention_decode
# nonce k1-v2-scored (k1_v1 kernel for L=1 at every B; L=8 -> rung-1 path with L2; sliding L=1 1 CTA/SM; from jobs d857095aa064476f, f7843f722b124e77, 01be153c3cdb4b1d)
"""K1 v1: fused gated-attention decode/verify for Muse Spark Open (sm_90, mma.sync; one main kernel per call, plus
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
K1_STAGES = 4
K1_NEWK = 16

# Host tables, keyed by (layer kind, q_len). Expected wins (re-measure): K1 everywhere.
#   route: smallest batch that takes K1 (others: FA3). v0 measured L=1 wins at every B except B=1 full (combine,
#   fixed in v1); L=8 is predicted to win after the v1 redesign. Set an entry to 1 << 30 to send it to FA3.
K1_ROUTE_MIN_BATCH = {("sliding", 1): 1, ("full", 1): 1}  # measured: K1 v1 L=8 loses to FA3 at every shape (0.53-0.92x)
#   fold (last-CTA merge, no combine launch) when batch >= this; else the separate k1_combine.
K1_FOLD_MIN_BATCH = {("sliding", 1): 1, ("sliding", 8): 1, ("full", 1): 8, ("full", 8): 8}
#   minimum tiles (64 keys) per split.
K1_MIN_CHUNK_T = {("sliding", 1): 2, ("sliding", 8): 4, ("full", 1): 2, ("full", 8): 2}
#   target CTAs per SM when sizing Gx (capped by the measured occupancy of k1_main<L>).
K1_CPS_T = {("sliding", 1): 1, ("sliding", 8): 2, ("full", 1): 2, ("full", 8): 2}  # lab: sliding L=1 1 CTA/SM -1.3/-0.6 us at B16/B32
K1_COUNTERS = 1 << 16          # fold arrival counters (B * Hkv/2 must fit)


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
                int64_t nsg);
int64_t k1_occupancy(int64_t L);
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
constexpr int STAGES = 4;                      // cp.async ring depth
constexpr int NEWK = 16;                       // keys in the new-token tile
constexpr int ROWB = 128;                      // smem bytes per key per K (or V): 2 heads x 64 dims x 1 B
constexpr int NWARP = 8;
constexpr int NTHR = NWARP * 32;
constexpr int STAGE_BYTES = 2 * TILE * ROWB;   // K tile then V tile
constexpr int RING_BYTES = STAGES * STAGE_BYTES;
constexpr int NEW_BYTES = 2 * NEWK * ROWB;
constexpr int AUX_BYTES = 2048;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float P_OFFSET = 8.0f;               // p = exp2(s - m + 8) in (0, 256]
constexpr unsigned FULL = 0xffffffffu;
static_assert(NEW_BYTES == NTHR * 16, "new-tile zeroing assumes one uint4 per thread");

// L=1: Q fp16 [2][8][64]; L=8: Q e4m3 [2][64][64]
template <int L> constexpr int qs_bytes() { return (L == 1) ? 2 * 8 * 64 * 2 : 2 * 64 * 64; }
template <int L> constexpr int cs_bytes() { return L * 32 * 8; }   // (cos, sin) per (token, pair)
template <int L> constexpr int off_qs() { return RING_BYTES + NEW_BYTES; }
template <int L> constexpr int off_cs() { return off_qs<L>() + qs_bytes<L>(); }
template <int L> constexpr int off_aux() { return off_cs<L>() + cs_bytes<L>(); }
template <int L> constexpr int off_plan() { return off_aux<L>() + AUX_BYTES; }
template <int L> constexpr int smem_bytes() { return off_plan<L>() + 64; }

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

// L=8 (v1): warp = (KV head h, row half rh: rows 32rh..+31 = m-tiles mt 0/1, key half: keys kw..kw+8NT-1).
// QK on m16n8k32 e4m3: Q e4m3 frags qf[mt][s][4]; lane (g,t) reads 16 B [16t,16t+16) of the K row (words w0..w3,
// step s uses w[2s], w[2s+1]); logical k 4t+i <-> dim 16t+8s+i, 16+4t+i <-> 16t+8s+4+i. PV on m16n8k16 FP16;
// every V fragment feeds both m-tiles. Thread rows: (mt, rb) -> row 32rh+16mt+8rb+g, token jr[mt][rb].
template <int NT>
__device__ __forceinline__ void tile_compute8(const uint8_t* __restrict__ Kbuf, const uint8_t* __restrict__ Vbuf,
                                              const int kw, const int h, const int g, const int t,
                                              const uint32_t (&qf)[2][2][4], float (&o)[2][8][4],
                                              float (&m)[2][2], float (&l)[2][2], const float cl2, const bool need,
                                              const int tile_abs, const int key_lim, const int seq, const int W,
                                              const int (&jr)[2][2]) {
  float sc[2][NT][4];
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) {
    const int key = kw + 8 * jn + g;
    const uint4 kv = *reinterpret_cast<const uint4*>(Kbuf + key * ROWB + 16 * ((4 * h + t) ^ (4 * (key & 1))));
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      sc[mt][jn][0] = 0.f; sc[mt][jn][1] = 0.f; sc[mt][jn][2] = 0.f; sc[mt][jn][3] = 0.f;
      mma16832_e4m3(sc[mt][jn], qf[mt][0][0], qf[mt][0][1], qf[mt][0][2], qf[mt][0][3], kv.x, kv.y);
      mma16832_e4m3(sc[mt][jn], qf[mt][1][0], qf[mt][1][1], qf[mt][1][2], qf[mt][1][3], kv.z, kv.w);
    }
  }
  if (need) {
#pragma unroll
    for (int jn = 0; jn < NT; ++jn) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ak = tile_abs + kw + 8 * jn + 2 * t + e;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int rb = 0; rb < 2; ++rb) {
            const int j = jr[mt][rb];
            const bool ok = (ak < key_lim) && (ak <= seq + j) && (W == 0 || ak > seq + j - W);
            if (!ok) sc[mt][jn][2 * rb + e] = neg_inf();
          }
        }
      }
    }
  }
  float a[2][2];
  bool resc = false;
  uint32_t pa[2][NT / 2][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
      float x = neg_inf();
#pragma unroll
      for (int jn = 0; jn < NT; ++jn) x = fmaxf(x, fmaxf(sc[mt][jn][2 * rb], sc[mt][jn][2 * rb + 1]));
      x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
      x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
      const float n = fmaxf(m[mt][rb], x * cl2);
      const float u = (n == neg_inf()) ? 0.f : n;
      a[mt][rb] = ex2(m[mt][rb] - u);
      resc |= (a[mt][rb] != 1.f);
      const float bo = u - P_OFFSET;
      m[mt][rb] = n;
      float ps = 0.f;
#pragma unroll
      for (int kk = 0; kk < NT / 2; ++kk) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int jn = 2 * kk + hf;
          const float p0 = ex2(fmaf(sc[mt][jn][2 * rb], cl2, -bo));
          const float p1 = ex2(fmaf(sc[mt][jn][2 * rb + 1], cl2, -bo));
          ps += p0 + p1;
          pa[mt][kk][2 * hf + rb] = pack_h2(p0, p1);   // reg0 row g n-tile 2kk, reg1 row g+8, reg2/3: 2kk+1
        }
      }
      l[mt][rb] = l[mt][rb] * a[mt][rb] + ps;
    }
  }
  if (__any_sync(FULL, resc)) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        o[mt][jn][0] *= a[mt][0]; o[mt][jn][1] *= a[mt][0];
        o[mt][jn][2] *= a[mt][1]; o[mt][jn][3] *= a[mt][1];
      }
    }
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
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
        mma16816(o[mt][jn], pa[mt][kk][0], pa[mt][kk][1], pa[mt][kk][2], pa[mt][kk][3], b0, b1);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Split merge + finalize of all 16L rows of (b, hp) by one CTA (fold path; k1_emulate.merge_fold +
// finalize_rows). Items (row r, quad u = dims 4u..4u+3); the 16 quads of a row are one half-warp.
// po: [S][16L][64], pl: [S][16L] (global partials with cg = true, or the CTA's own smem staging, S = 1).
// ---------------------------------------------------------------------------------------------
template <int L>
__device__ __forceinline__ void merge_finalize(const Params& p, const float* po, const float* pl, const int S, const int b,
                               const int hp, const bool cg) {
  constexpr int ROWS = 16 * L, R8 = 8 * L, ITEMS = ROWS * 16;
  static_assert(ITEMS % NTHR == 0, "uniform item loop (half-warp shuffles)");
  const float vsc = __ldg(p.v_scale);
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  for (int it = threadIdx.x; it < ITEMS; it += NTHR) {
    const int r = it >> 4, u = it & 15;
    float M = neg_inf();
    for (int s = 0; s < S; ++s) M = fmaxf(M, ldf(pl + (size_t)s * ROWS + r, cg));
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float ws = 0.f;
    for (int s = 0; s < S; ++s) {
      const float w = ex2(ldf(pl + (size_t)s * ROWS + r, cg) - Mu);
      const float4 v = ldf4(po + ((size_t)s * ROWS + r) * 64 + 4 * u, cg);
      ws += w;
      acc.x = fmaf(w, v.x, acc.x); acc.y = fmaf(w, v.y, acc.y);
      acc.z = fmaf(w, v.z, acc.z); acc.w = fmaf(w, v.w, acc.w);
    }
    const float sc = (ws > 0.f) ? vsc / ws : 0.f;
    acc.x *= sc; acc.y *= sc; acc.z *= sc; acc.w *= sc;
    float ss = acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w;
    ss += __shfl_xor_sync(FULL, ss, 8);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 1);
    const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const int hh = r / R8, rr = r - hh * R8, j = rr >> 3, gi = rr & 7;
    const int qh = (2 * hp + hh) * 8 + gi;
    const size_t tok = (size_t)b * L + j;
    const uint2 graw = *reinterpret_cast<const uint2*>(p.qkv + tok * stride + (size_t)(p.Hq + 2 * p.Hkv) * 64 +
                                                       (size_t)qh * 64 + 4 * u);
    const float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.x));
    const float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&graw.y));
    const __nv_bfloat162 o01 = __floats2bfloat162_rn(acc.x * rs * sigmoidf_(g01.x), acc.y * rs * sigmoidf_(g01.y));
    const __nv_bfloat162 o23 = __floats2bfloat162_rn(acc.z * rs * sigmoidf_(g23.x), acc.w * rs * sigmoidf_(g23.y));
    uint2 ow;
    ow.x = *reinterpret_cast<const uint32_t*>(&o01);
    ow.y = *reinterpret_cast<const uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(p.out + tok * (size_t)p.Hq * 64 + (size_t)qh * 64 + 4 * u) = ow;
  }
}

// ---------------------------------------------------------------------------------------------
// k1_main
// ---------------------------------------------------------------------------------------------
template <int L>
__global__ void __launch_bounds__(NTHR, (L == 1) ? 2 : 1) k1_main(const Params p) {
#if K1_ARCH_OK
  constexpr int R8 = 8 * L;          // rows per KV head (row r = token*8 + head-in-group)
  constexpr int ROWS = 16 * L;       // rows per CTA (2 KV heads)
  constexpr int RPW = 2 * R8 / NWARP;   // Q rows prepared per warp
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* newk = smem + RING_BYTES;
  uint8_t* newv = newk + NEWK * ROWB;
  uint8_t* qs = smem + off_qs<L>();
  float2* cst = reinterpret_cast<float2*>(smem + off_cs<L>());
  float* aux = reinterpret_cast<float*>(smem + off_aux<L>());
  int* plan_s = reinterpret_cast<int*>(smem + off_plan<L>());

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x, hp = blockIdx.y;
  const int HP = gridDim.y;

  reinterpret_cast<uint4*>(newk)[tid] = make_uint4(0u, 0u, 0u, 0u);   // zero newk + newv
  if (warp == 0) {
    const int C = plan_C(p.seq_lens, p.B, p.W, p.Gx, p.min_chunk);
    int res[5];
    plan_find(p.seq_lens, p.B, p.W, C, c, true, res);
    if (lane == 0) {
      plan_s[0] = res[0]; plan_s[1] = res[1]; plan_s[2] = res[2]; plan_s[3] = res[4]; plan_s[4] = res[3];
    }
  }
  __syncthreads();
  const int b = plan_s[0];
  if (b < 0) return;                                  // idle CTA (uniform)
  const int s_idx = plan_s[1], S_b = plan_s[2], nt_b = plan_s[3], off_b = plan_s[4];
  // balanced split boundaries: tiles [s*nt/S, (s+1)*nt/S), each <= C tiles
  const int t0 = (int)(((long long)s_idx * nt_b) / S_b);
  const int ntl = (int)(((long long)(s_idx + 1) * nt_b) / S_b) - t0;
  const bool last = (s_idx == S_b - 1);
  const int seq = __ldg(p.seq_lens + b);
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
    uint8_t* stg = ring + (i % STAGES) * STAGE_BYTES;
    const int absk = kstart + (t0 + i) * TILE + lk;
    const bool valid = pe >= 0;
    const size_t goff = valid ? (size_t)((pe << pshift) | (absk & pmask)) * rowB : (size_t)0;
    const int nb = valid ? 16 : 0;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int ch = 2 * part + u;
      cp_async16(stg + lk * ROWB + 16 * (ch ^ (4 * (lk & 1))), kc + goff + 16 * ch, nb);
      cp_async16(stg + TILE * ROWB + lk * ROWB + 16 * (ch ^ (2 * ((lk >> 1) & 3))), vc + goff + 16 * ch, nb);
    }
  };
  {
    int pe0[STAGES - 1];
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) pe0[i] = pe_load(i);
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
      if (i < ntl) issue(i, pe0[i]);
      cp_async_commit();
    }
  }
  int pfA = pe_load(STAGES - 1);
  int pfB = pe_load(STAGES);

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
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  {
    float2 xv[RPW];
#pragma unroll
    for (int k = 0; k < RPW; ++k) {
      const int R = warp + k * NWARP;
      const int hh = R / R8, r = R - hh * R8, j = r >> 3, gi = r & 7;
      const int qh = (2 * hp + hh) * 8 + gi;
      xv[k] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
          p.qkv + ((size_t)b * L + j) * stride + (size_t)qh * 64 + 2 * lane));
    }
#pragma unroll
    for (int k = 0; k < RPW; ++k) {
      const int R = warp + k * NWARP;
      const int hh = R / R8, r = R - hh * R8, j = r >> 3;
      float2 x = xv[k];
      const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
      x.x *= rs; x.y *= rs;
      if (p.use_rope) x = rope_tab(x, cst[j * 32 + lane]);
      if (L == 1) {
        reinterpret_cast<uint32_t*>(qs)[(hh * R8 + r) * 32 + lane] = pack_h2(x.x, x.y);
      } else {
        reinterpret_cast<uint16_t*>(qs)[(hh * R8 + r) * 32 + lane] =
            (uint16_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
      }
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
      const int pc = (kind == 0) ? (ch ^ (4 * (j & 1))) : (ch ^ (2 * ((j >> 1) & 3)));
      *reinterpret_cast<uint16_t*>(((kind == 0) ? newk : newv) + j * ROWB + 16 * pc + (bb & 15)) = (uint16_t)q8;
    }
  }
  __syncthreads();

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
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      {
        const int nx = i + STAGES - 1;
        if (nx < ntl) issue(nx, pfA);
        cp_async_commit();
        pfA = pfB;
        pfB = pe_load(i + STAGES + 1);
      }
      const uint8_t* Kb = ring + (i % STAGES) * STAGE_BYTES;
      const int ts = kstart + (t0 + i) * TILE;
      const bool need = (ts + TILE > seq) || (W > 0 && ts < seq + L - W);
      tile_compute1<2>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, need, ts, seq, seq, W);
    }
    cp_async_wait<0>();
    __syncthreads();
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
    // ---- L=8 roles: warp = (KV head h = warp>>2, row half rh = (warp>>1)&1, key half kh = warp&1)
    const int h = warp >> 2, rh = (warp >> 1) & 1, kh = warp & 1, kw = 32 * kh, grp = warp >> 1;
    uint32_t qf[2][2][4];
    int jr[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const uint8_t* qrow = qs + (size_t)(h * R8 + 32 * rh + 16 * mt + g) * 64 + 16 * t;
      const uint4 qa = *reinterpret_cast<const uint4*>(qrow);
      const uint4 qb = *reinterpret_cast<const uint4*>(qrow + 8 * 64);
      qf[mt][0][0] = qa.x; qf[mt][0][1] = qb.x; qf[mt][0][2] = qa.y; qf[mt][0][3] = qb.y;
      qf[mt][1][0] = qa.z; qf[mt][1][1] = qb.z; qf[mt][1][2] = qa.w; qf[mt][1][3] = qb.w;
      jr[mt][0] = 4 * rh + 2 * mt;
      jr[mt][1] = 4 * rh + 2 * mt + 1;
    }
    float o[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) { o[mt][jn][0] = 0.f; o[mt][jn][1] = 0.f; o[mt][jn][2] = 0.f; o[mt][jn][3] = 0.f; }
    }
    float m[2][2] = {{neg_inf(), neg_inf()}, {neg_inf(), neg_inf()}};
    float l[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
    for (int i = 0; i < ntl; ++i) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      {
        const int nx = i + STAGES - 1;
        if (nx < ntl) issue(nx, pfA);
        cp_async_commit();
        pfA = pfB;
        pfB = pe_load(i + STAGES + 1);
      }
      const uint8_t* Kb = ring + (i % STAGES) * STAGE_BYTES;
      const int ts = kstart + (t0 + i) * TILE;
      const bool need = (ts + TILE > seq) || (W > 0 && ts < seq + L - W);
      tile_compute8<4>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, m, l, cl2, need, ts, seq, seq, W, jr);
    }
    cp_async_wait<0>();
    __syncthreads();
    if (last && kh == 0) tile_compute8<2>(newk, newv, 0, h, g, t, qf, o, m, l, cl2, true, seq, seq + L, seq, W, jr);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int rb = 0; rb < 2; ++rb) {
        l[mt][rb] += __shfl_xor_sync(FULL, l[mt][rb], 1);
        l[mt][rb] += __shfl_xor_sync(FULL, l[mt][rb], 2);
      }
    }
    // merge the two key halves of each (h, rh): kh = 1 stages O/m/l, kh = 0 merges (order kh 0, 1)
    float* ow = reinterpret_cast<float*>(ring);   // [4 groups][32 rows][64]
    ost = ow + 4 * 32 * 64;                       // [128][64]
    float* mw = aux;                              // [4][32]
    float* lw = mw + 4 * 32;                      // [4][32]
    lse_s = lw + 4 * 32;                          // [128]
    if (kh == 1) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int d = 8 * (2 * t + e) + jn;
            ow[(grp * 32 + 16 * mt + g) * 64 + d] = o[mt][jn][e];
            ow[(grp * 32 + 16 * mt + 8 + g) * 64 + d] = o[mt][jn][2 + e];
          }
        }
      }
      if (t == 0) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int rb = 0; rb < 2; ++rb) {
            mw[grp * 32 + 16 * mt + 8 * rb + g] = m[mt][rb];
            lw[grp * 32 + 16 * mt + 8 * rb + g] = l[mt][rb];
          }
        }
      }
    }
    __syncthreads();
    if (kh == 0) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
          const int rl = 16 * mt + 8 * rb + g;
          const float mp = mw[grp * 32 + rl], lp = lw[grp * 32 + rl];
          const float M = fmaxf(m[mt][rb], mp);
          const float Mu = (M == neg_inf()) ? 0.f : M;
          const float f0 = ex2(m[mt][rb] - Mu), f1 = ex2(mp - Mu);
          const float lt = f0 * l[mt][rb] + f1 * lp;
          const float inv = (lt > 0.f) ? 1.f / lt : 0.f;
          const int row = h * R8 + 32 * rh + rl;
#pragma unroll
          for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int d = 8 * (2 * t + e) + jn;
              ost[row * 64 + d] = (f0 * o[mt][jn][2 * rb + e] + f1 * ow[(grp * 32 + rl) * 64 + d]) * inv;
            }
          }
          if (t == 0) lse_s[row] = (lt > 0.f) ? M + log2f(lt) - P_OFFSET : neg_inf();
        }
      }
    }
  }
  __syncthreads();

  // ---- output: fold (merge in the last-arriving CTA, or straight from smem when S_b == 1) or partials only
  if (p.fold && S_b == 1) {
    merge_finalize<L>(p, ost, lse_s, 1, b, hp, false);
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
    if (plan_s[5]) {                                  // last CTA of (b, hp): every partial is written
      __threadfence();
      const size_t base = (size_t)hp * p.Gx + off_b;
      merge_finalize<L>(p, p.part_o + base * ROWS * 64, p.part_lse + base * ROWS, S_b, b, hp, true);
      __syncthreads();
      if (tid == 0) *cnt = 0;                         // self-reset for the next call (after all reads)
    }
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

template <int L>
static void ensure_smem_attr() {
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(k1::k1_main<L>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        k1::smem_bytes<L>()));
    attr_set = true;
  }
}

template <int L>
static void launch_main(const k1::Params& p, dim3 grid, cudaStream_t st) {
  ensure_smem_attr<L>();
  k1::k1_main<L><<<grid, k1::NTHR, k1::smem_bytes<L>(), st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Resident k1_main CTAs per SM (registers + smem), used to size Gx so the grid is one wave.
int64_t k1_occupancy(int64_t L) {
  int nb = 0;
  if (L == 1) {
    ensure_smem_attr<1>();
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::k1_main<1>, k1::NTHR, k1::smem_bytes<1>()));
  } else {
    ensure_smem_attr<8>();
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k1::k1_main<8>, k1::NTHR, k1::smem_bytes<8>()));
  }
  return (int64_t)nb;
}

void k1_forward(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor page_table,
                torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale,
                torch::Tensor inv_freq, torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor out,
                torch::Tensor counters, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t W,
                int64_t page_shift, int64_t Gx, int64_t min_chunk, int64_t use_rope, double qk_sm, int64_t fold,
                int64_t nsg) {
  TORCH_CHECK(L == 1 || L == 8, "k1: q_len must be 1 or 8");
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
  if (L == 1) launch_main<1>(p, grid, st);
  else launch_main<8>(p, grid, st);
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
        name=f"k1_v1_{tag}",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["k1_forward", "k1_occupancy"],
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
print(f"[k1] v1 FALLBACK={FALLBACK} SELF_TEST={SELF_TEST} built={_MOD is not None} route={K1_ROUTE_MIN_BATCH}"
      f" fold={K1_FOLD_MIN_BATCH} min_chunk={K1_MIN_CHUNK_T} cps={K1_CPS_T}", flush=True)

_STATE = {"tested": False, "ok": False}
_SMS = {}
_OCC = {}
_INV_FREQ = {}
_CNT = {}


def _sms(dev):
    key = str(dev)
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SMS[key]


def _occupancy(L, dev):
    """resident k1_main<L> CTAs per SM; queried once per (L, device), eagerly."""
    key = (L, str(dev))
    if key not in _OCC:
        _OCC[key] = max(1, int(_MOD.k1_occupancy(L)))
        print(f"[k1] k1_main<{L}> occupancy {_OCC[key]} CTAs/SM", flush=True)
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


def _k1(data, fold=None, min_chunk=None, cps=None, nsg=None):
    """One K1 call. Overrides (lab / self-test): fold (bool), min_chunk, cps (target CTAs/SM), nsg."""
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
    cnt = _counters(dev) if fold else None
    if fold and (cnt is None or B * hp > cnt.numel()):
        fold = False
    if cnt is None:
        cnt = _CNT.get(str(dev))
        if cnt is None:
            cnt = torch.zeros(1, dtype=torch.int32, device=dev)   # unused when fold == 0
    cps_eff = max(1, min(int(cps), _occupancy(L, dev)))
    gx = max(max(1, (_sms(dev) * cps_eff) // hp), B + 1)
    part_o = torch.empty((hp, gx, 16 * L, 64), dtype=torch.float32, device=dev)
    part_lse = torch.empty((hp, gx, 16 * L), dtype=torch.float32, device=dev)
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    theta = cfg["rope_theta"]
    _MOD.k1_forward(qkv, data["seq_lens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                    data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], _inv_freq(theta, dev),
                    part_o, part_lse, out, cnt, B, L, hq, hkv, cfg["window"], cfg["page_size"].bit_length() - 1, gx,
                    int(min_chunk), 1 if theta else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]),
                    1 if fold else 0, int(nsg))
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


def _self_test(dev):
    """Each case runs in both merge modes (fold: last-CTA merge; separate: k1_combine) and twice in fold mode (the
    counters must have reset themselves). L=1 is compared with the FP32 reference; L=8 (E4M3 queries) with the
    reference at the same query rounding. New K/V bytes are compared with the reference's quantization."""
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
                out = _k1(d, fold=(mode == "fold"))
                torch.cuda.synchronize(dev)
                hkv = heads[1]
                pos = d["seq_lens"].long()[:, None] + torch.arange(L, device=dev)
                pg = d["config"]["page_size"]
                slots = torch.gather(d["page_table"].long(), 1, pos // pg) * pg + pos % pg
                k_new = d["k_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
                v_new = d["v_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
                ref = _ref_attention(d, k_new, v_new, q_e4m3=(L == 8))
                tot, row = _rel_rows(out, ref)
                kref, vref = _ref_new_kv(d)
                kt, _ = _rel_rows(k_new.view(FP8).float(), kref)
                vt, _ = _rel_rows(v_new.view(FP8).float(), vref)
                finite = bool(torch.isfinite(out.float()).all())
                cnt_ok = cnt is None or int(cnt.abs().sum()) == 0
                print(f"[k1] self-test {mode:4s} B={B} ctx={ctx} L={L} {'sliding' if sliding else 'full'} page={page}"
                      f" kv_scale={kvs:g} heads={heads}: out rel {tot:.3e} worst row {row:.3e}; new K {kt:.3e}"
                      f" V {vt:.3e}; counters {'0' if cnt_ok else 'NOT RESET'}", flush=True)
                if not (finite and tot < 2e-2 and row < 6e-2 and kt < 3e-2 and vt < 3e-2 and cnt_ok):
                    raise RuntimeError("self-test mismatch")
        print(f"[k1] self-test passed in {time.time() - t0:.1f}s", flush=True)
        return True
    except Exception as ex:  # noqa: BLE001
        print(f"[k1] self-test FAILED, using the fallback path: {ex!r}", flush=True)
        return False


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

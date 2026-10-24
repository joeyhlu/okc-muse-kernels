# okc: problem=muse_attention_decode
# nonce k1-v02-1 (K1 for L=1 sliding all B and full B>=9; rung-1 path WITH L2 elsewhere — r1 measured 1.115x)
"""K1 v0: fused gated-attention decode/verify for Muse Spark Open (sm_90, mma.sync m16n8k16 FP16, one main
kernel + one small combine kernel per call; fixed grids, all per-request lengths read on device).

k1_main  grid (Gx, Hkv/2), 256 threads. Each CTA = (split of one request's key range, one KV-head pair), so
         every page-size-1 key read is a full 128 B K row + 128 B V row (both KV heads).
           - split plan computed on device from seq_lens (identical integer math in k1_combine):
             tiles of 64 keys over the request's old keys [kstart, seq); C = the smallest tile cap with
             sum_b max(1, ceil(tiles_b / C)) <= Gx (short binary search), S_b = max(1, ceil(tiles_b / C)),
             split s of b covers tiles [s*tiles_b/S_b, (s+1)*tiles_b/S_b); Gx = SMs x resident CTAs/SM;
           - QK-norm (+ interleaved RoPE on sliding layers) of the request's 16 x L query rows, FP16 in smem;
           - the last split of each request also normalizes/ropes the L new keys, casts K/V to E4M3
             (IEEE divide by the cache scale, RNE, satfinite), stores them at cache_loc, and keeps them in a
             16-key smem tile that it attends last -- the cache rows it writes are never read back;
           - 4-stage cp.async ring (zero-fill for slots past the range), page-table entries prefetched two
             tiles ahead; K/V e4m3 -> f16 with cvt.rn.f16x2.e4m3x2 (exact), QK^T and PV on mma.sync with
             FP32 accumulate, online softmax in log2 units with p = exp2(s - m + 8) (FP16 P);
           - L=1: warp = (KV head, 16-key quarter of each tile), 8 of 16 MMA rows used, 4 warps merged in
             smem; L=8: warp = (KV head, 16 rows = 2 tokens x 8 heads), every row owned by one warp;
           - writes the split's partial (O/l in FP32, lse = m + log2 l - 8).
k1_combine one warp per (request, token, q head): merges the splits, x v_scale, RMSNorm, x sigmoid(gate), BF16.

Every index formula (swizzles, fragment <-> dim permutations, plan, masks) is checked on CPU by
k1_emulate.py, which emulates this decomposition against a copy of the judge's reference.
FALLBACK (or a failed build / self-test, or an unsupported shape) uses the rung-1 SGLang path (FA3).
"""

import hashlib
import time
from typing import TypedDict

import torch

FALLBACK = False               # True: always use the rung-1 FA3 path
SELF_TEST = True               # first eager call compares K1 against an FP32 reference; fallback on mismatch
# (layer kind, q_len) -> smallest batch that takes K1 (measured, job ded828a1027f427a); others use FA3.
K1_ROUTE_MIN_BATCH = {("sliding", 1): 1, ("full", 1): 9}
R1_QK_FUSED_FP8_STAGE = True   # fallback = rung-1 path with L2 (GPU-verified: jobs 01be153c3cdb4b1d, 3acdfa92bf4c4f04)

# Kernel constants (mirrored in CUDA_SRC and k1_emulate.py)
K1_TILE = 64
K1_STAGES = 4
K1_NEWK = 16
K1_MIN_CHUNK = 2
K1_CPS = 2                     # target CTAs per SM when sizing Gx

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
                int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t W, int64_t page_shift, int64_t Gx,
                int64_t min_chunk, int64_t use_rope, double qk_sm);
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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
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
constexpr float LOG2E = 1.4426950408889634f;
constexpr float P_OFFSET = 8.0f;               // p = exp2(s - m + 8) in (0, 256]
constexpr unsigned FULL = 0xffffffffu;
static_assert(2 * NEWK * ROWB == NTHR * 16, "new-tile zeroing assumes one uint4 per thread");

template <int L>
constexpr int smem_bytes() { return STAGES * STAGE_BYTES + 2 * NEWK * ROWB + 2 * 8 * L * 64 * 2 + 64; }

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
  int B, Hq, Hkv, W, page_shift, pt_stride, Gx, min_chunk, use_rope;
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

// Interleaved RoPE of one (even, odd) pair: angle = fp32(pos) * inv_freq rounded once, as torch does.
__device__ __forceinline__ float2 rope2(float2 x, int pos, float invf) {
  const float ang = __fmul_rn((float)pos, invf);
  float sn, cs;
  sincosf(ang, &sn, &cs);
  return make_float2(x.x * cs - x.y * sn, x.x * sn + x.y * cs);
}

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
// One tile of keys for one warp (k1_emulate.emulate_k1.process).
//   K stage: key row x at x*ROWB, 16-B chunk c (head c>>2) at physical chunk c ^ (4*(x&1)).
//   V stage: chunk c at c ^ (2*((x>>1)&3)).
//   QK: n-tile jn = keys kw+8jn..+7; lane (g,t) loads 16 B of key kw+8jn+g, head h, dims 16t..16t+15;
//       k-step s uses word s: logical k (2t,2t+1) <-> dims 16t+4s+{0,1}, (2t+8,2t+9) <-> 16t+4s+{2,3}.
//   PV: k-step kk = keys kw+16kk..+15; lane loads 8 B (dims 8g..8g+7) of keys 2t,2t+1,2t+8,2t+9;
//       n-tile jn col n <-> output dim 8n + jn.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint2 vld(const uint8_t* Vbuf, int x, int h, int g) {
  return *reinterpret_cast<const uint2*>(Vbuf + x * ROWB + 16 * ((4 * h + (g >> 1)) ^ (2 * ((x >> 1) & 3))) +
                                         8 * (g & 1));
}

template <int L, int NT>
__device__ __forceinline__ void tile_compute(const uint8_t* __restrict__ Kbuf, const uint8_t* __restrict__ Vbuf,
                                             const int kw, const int h, const int g, const int t,
                                             const uint32_t (&qf)[4][4], float (&o)[8][4], float& mA, float& mB,
                                             float& lA, float& lB, const float cl2, const bool need,
                                             const int tile_abs, const int key_lim, const int seq, const int W,
                                             const int jA, const int jB) {
  float sc[NT][4];
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) {
    const int key = kw + 8 * jn + g;
    const uint4 kv = *reinterpret_cast<const uint4*>(Kbuf + key * ROWB + 16 * ((4 * h + t) ^ (4 * (key & 1))));
    const uint32_t wv[4] = {kv.x, kv.y, kv.z, kv.w};
    sc[jn][0] = 0.f; sc[jn][1] = 0.f; sc[jn][2] = 0.f; sc[jn][3] = 0.f;
#pragma unroll
    for (int s = 0; s < 4; ++s)
      mma16816(sc[jn], qf[s][0], qf[s][1], qf[s][2], qf[s][3], e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
  }
  if (need) {
#pragma unroll
    for (int jn = 0; jn < NT; ++jn) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int ak = tile_abs + kw + 8 * jn + 2 * t + e;
        const bool okA = (ak < key_lim) && (ak <= seq + jA) && (W == 0 || ak > seq + jA - W);
        if (!okA) sc[jn][e] = neg_inf();
        if (L == 8) {
          const bool okB = (ak < key_lim) && (ak <= seq + jB) && (W == 0 || ak > seq + jB - W);
          if (!okB) sc[jn][2 + e] = neg_inf();
        }
      }
    }
  }
  // row maxima in raw dot units, reduced over the quad that shares the row
  float xA = neg_inf(), xB = neg_inf();
#pragma unroll
  for (int jn = 0; jn < NT; ++jn) {
    xA = fmaxf(xA, fmaxf(sc[jn][0], sc[jn][1]));
    if (L == 8) xB = fmaxf(xB, fmaxf(sc[jn][2], sc[jn][3]));
  }
  xA = fmaxf(xA, __shfl_xor_sync(FULL, xA, 1));
  xA = fmaxf(xA, __shfl_xor_sync(FULL, xA, 2));
  const float nA = fmaxf(mA, xA * cl2);
  const float uA = (nA == neg_inf()) ? 0.f : nA;
  const float aA = ex2(mA - uA);
  const float bA = uA - P_OFFSET;
  mA = nA;
  float aB = 1.f, bB = 0.f;
  if (L == 8) {
    xB = fmaxf(xB, __shfl_xor_sync(FULL, xB, 1));
    xB = fmaxf(xB, __shfl_xor_sync(FULL, xB, 2));
    const float nB = fmaxf(mB, xB * cl2);
    const float uB = (nB == neg_inf()) ? 0.f : nB;
    aB = ex2(mB - uB);
    bB = uB - P_OFFSET;
    mB = nB;
  }
  // P (fp16) as the A operand of the PV MMA: reg0 row g keys 2t.., reg1 row g+8, reg2 row g keys 2t+8.., reg3
  uint32_t pa[NT / 2][4];
  float psA = 0.f, psB = 0.f;
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk) {
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int jn = 2 * kk + hf;
      const float p0 = ex2(fmaf(sc[jn][0], cl2, -bA));
      const float p1 = ex2(fmaf(sc[jn][1], cl2, -bA));
      psA += p0 + p1;
      pa[kk][2 * hf] = pack_h2(p0, p1);
      if (L == 8) {
        const float p2 = ex2(fmaf(sc[jn][2], cl2, -bB));
        const float p3 = ex2(fmaf(sc[jn][3], cl2, -bB));
        psB += p2 + p3;
        pa[kk][2 * hf + 1] = pack_h2(p2, p3);
      } else {
        pa[kk][2 * hf + 1] = 0u;
      }
    }
  }
  lA = lA * aA + psA;
  if (L == 8) lB = lB * aB + psB;
#pragma unroll
  for (int jn = 0; jn < 8; ++jn) {
    o[jn][0] *= aA; o[jn][1] *= aA;
    if (L == 8) { o[jn][2] *= aB; o[jn][3] *= aB; }
  }
#pragma unroll
  for (int kk = 0; kk < NT / 2; ++kk) {
    const int x0 = kw + 16 * kk + 2 * t;
    const uint2 va = vld(Vbuf, x0, h, g);
    const uint2 vb = vld(Vbuf, x0 + 1, h, g);
    const uint2 vc = vld(Vbuf, x0 + 8, h, g);
    const uint2 vd = vld(Vbuf, x0 + 9, h, g);
    // (key a, key b) byte pairs per dim: index jn>>1 -> {dims 8g+0/1, 8g+2/3, 8g+4/5, 8g+6/7}
    const uint32_t w01[4] = {__byte_perm(va.x, vb.x, 0x5140), __byte_perm(va.x, vb.x, 0x7362),
                             __byte_perm(va.y, vb.y, 0x5140), __byte_perm(va.y, vb.y, 0x7362)};
    const uint32_t w23[4] = {__byte_perm(vc.x, vd.x, 0x5140), __byte_perm(vc.x, vd.x, 0x7362),
                             __byte_perm(vc.y, vd.y, 0x5140), __byte_perm(vc.y, vd.y, 0x7362)};
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      const uint32_t s01 = w01[jn >> 1], s23 = w23[jn >> 1];
      const uint32_t b0 = e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01);
      const uint32_t b1 = e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23);
      mma16816(o[jn], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b0, b1);
    }
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
  constexpr int NT = (L == 1) ? 2 : 8;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* ring = smem;
  uint8_t* newk = smem + STAGES * STAGE_BYTES;
  uint8_t* newv = newk + NEWK * ROWB;
  __half* qs = reinterpret_cast<__half*>(newv + NEWK * ROWB);   // [2][R8][64]
  int* plan_s = reinterpret_cast<int*>(qs + 2 * R8 * 64);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x, hp = blockIdx.y;

  reinterpret_cast<uint4*>(newk)[tid] = make_uint4(0u, 0u, 0u, 0u);   // zero newk + newv
  if (warp == 0) {
    const int C = plan_C(p.seq_lens, p.B, p.W, p.Gx, p.min_chunk);
    int res[5];
    plan_find(p.seq_lens, p.B, p.W, C, c, true, res);
    if (lane == 0) {
      plan_s[0] = res[0]; plan_s[1] = res[1]; plan_s[2] = res[2]; plan_s[3] = res[4];
    }
  }
  __syncthreads();
  const int b = plan_s[0];
  if (b < 0) return;                                  // idle CTA (uniform)
  const int s_idx = plan_s[1], S_b = plan_s[2], nt_b = plan_s[3];
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

  // ---- Q prep: RMSNorm (+RoPE), fp16 rows in smem
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  const float invf = p.use_rope ? __ldg(p.inv_freq + lane) : 0.f;
  for (int R = warp; R < 2 * R8; R += NWARP) {
    const int hh = R / R8, r = R - hh * R8, j = r >> 3, gi = r & 7;
    const int qh = (2 * hp + hh) * 8 + gi;
    const __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(
        p.qkv + ((size_t)b * L + j) * stride + (size_t)qh * 64 + 2 * lane);
    float2 x = __bfloat1622float2(v2);
    const float rs = rsqrtf(warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f) + 1e-5f);
    x.x *= rs; x.y *= rs;
    if (p.use_rope) x = rope2(x, seq + j, invf);
    reinterpret_cast<uint32_t*>(qs)[(hh * R8 + r) * 32 + lane] = pack_h2(x.x, x.y);
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
        if (p.use_rope) x = rope2(x, seq + j, invf);
        sc = ksc;
      } else {
        sc = vsc;
      }
      const __nv_fp8x2_storage_t q8 =
          __nv_cvt_float2_to_fp8x2(make_float2(x.x / sc, x.y / sc), __NV_SATFINITE, __NV_E4M3);
      const int64_t row = __ldg(p.cache_loc + tok);
      uint8_t* dst = ((kind == 0) ? p.k_cache : p.v_cache) + (size_t)row * rowB + (size_t)head * 64 + 2 * lane;
      *reinterpret_cast<uint16_t*>(dst) = (uint16_t)q8;
      const int bb = 64 * hh + 2 * lane;           // logical byte in the 128-B row
      const int ch = bb >> 4;
      const int pc = (kind == 0) ? (ch ^ (4 * (j & 1))) : (ch ^ (2 * ((j >> 1) & 3)));
      *reinterpret_cast<uint16_t*>(((kind == 0) ? newk : newv) + j * ROWB + 16 * pc + (bb & 15)) = (uint16_t)q8;
    }
  }
  __syncthreads();

  // ---- per-warp roles and Q fragments
  const int g = lane >> 2, t = lane & 3;
  const int h = (L == 1) ? (warp & 1) : (warp >> 2);
  const int kw = (L == 1) ? 16 * (warp >> 1) : 0;
  const int r0 = (L == 1) ? 0 : 16 * (warp & 3);
  const int jA = (r0 + g) >> 3, jB = (r0 + g + 8) >> 3;
  uint32_t qf[4][4];
  {
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs) + (h * R8 + r0 + g) * 32;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int d0 = (16 * t + 4 * s) >> 1;      // u32 holding dims 16t+4s, +1
      qf[s][0] = qa[d0];
      qf[s][2] = qa[d0 + 1];
      if (L == 8) {
        qf[s][1] = qa[8 * 32 + d0];
        qf[s][3] = qa[8 * 32 + d0 + 1];
      } else {
        qf[s][1] = 0u;
        qf[s][3] = 0u;
      }
    }
  }
  const float cl2 = p.qk_sm * __ldg(p.k_scale) * LOG2E;
  float o[8][4];
#pragma unroll
  for (int jn = 0; jn < 8; ++jn) { o[jn][0] = 0.f; o[jn][1] = 0.f; o[jn][2] = 0.f; o[jn][3] = 0.f; }
  float mA = neg_inf(), mB = neg_inf(), lA = 0.f, lB = 0.f;

  // ---- main loop over the split's old-key tiles
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
    tile_compute<L, NT>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, mB, lA, lB, cl2, need, ts, seq, seq, W,
                        jA, jB);
  }
  cp_async_wait<0>();
  __syncthreads();
  // ---- the new tokens (keys seq .. seq+L-1), causal inside the block
  if (last && (L == 8 || (warp >> 1) == 0)) {
    tile_compute<L, 2>(newk, newv, 0, h, g, t, qf, o, mA, mB, lA, lB, cl2, true, seq, seq + L, seq, W, jA, jB);
  }
  lA += __shfl_xor_sync(FULL, lA, 1);
  lA += __shfl_xor_sync(FULL, lA, 2);
  if (L == 8) {
    lB += __shfl_xor_sync(FULL, lB, 1);
    lB += __shfl_xor_sync(FULL, lB, 2);
  }

  // ---- epilogue: partial O/l (fp32) and lse = m + log2(l) - 8, staged in the (drained) ring
  float* ost;
  float* lse_s;
  if (L == 8) {
    ost = reinterpret_cast<float*>(ring);
    lse_s = ost + ROWS * 64;
    const int rowA = h * R8 + r0 + g, rowB = rowA + 8;
    const float iA = (lA > 0.f) ? 1.f / lA : 0.f;
    const float iB = (lB > 0.f) ? 1.f / lB : 0.f;
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int d = 8 * (2 * t + e) + jn;
        ost[rowA * 64 + d] = o[jn][e] * iA;
        ost[rowB * 64 + d] = o[jn][2 + e] * iB;
      }
    }
    if (t == 0) {
      lse_s[rowA] = (lA > 0.f) ? mA + log2f(lA) - P_OFFSET : neg_inf();
      lse_s[rowB] = (lB > 0.f) ? mB + log2f(lB) - P_OFFSET : neg_inf();
    }
  } else {
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
      for (int q4 = 0; q4 < 4; ++q4) M = fmaxf(M, mw[((q4 << 1) | hh) * 8 + r]);
      const float Mu = (M == neg_inf()) ? 0.f : M;
      float l = 0.f, acc = 0.f;
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const int w = (q4 << 1) | hh;
        const float f = ex2(mw[w * 8 + r] - Mu);
        l += f * lw[w * 8 + r];
        acc += f * ow[(w * 8 + r) * 64 + d];
      }
      ost[(hh * 8 + r) * 64 + d] = (l > 0.f) ? acc / l : 0.f;
      if (d == 0) lse_s[hh * 8 + r] = (l > 0.f) ? M + log2f(l) - P_OFFSET : neg_inf();
    }
  }
  __syncthreads();
  {
    const size_t pbase = (size_t)hp * p.Gx + c;
    float4* po = reinterpret_cast<float4*>(p.part_o + pbase * ROWS * 64);
    const float4* so = reinterpret_cast<const float4*>(ost);
    for (int idx = tid; idx < ROWS * 16; idx += NTHR) po[idx] = so[idx];
    for (int idx = tid; idx < ROWS; idx += NTHR) p.part_lse[pbase * ROWS + idx] = lse_s[idx];
  }
#endif
}

// ---------------------------------------------------------------------------------------------
// k1_combine: one warp per (request b, token j, q head qh)
// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(32) k1_combine(const Params p, const int L) {
#if K1_ARCH_OK
  const int lane = threadIdx.x;
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
  const float* po = p.part_o + (((size_t)hp * p.Gx + off) * ROWS + row) * 64 + 2 * lane;
  float mx = neg_inf();
  for (int s = lane; s < S; s += 32) mx = fmaxf(mx, pl[(size_t)s * ROWS]);
  mx = warp_max(mx);
  float o0 = 0.f, o1 = 0.f;
  if (mx != neg_inf()) {
    float ws = 0.f;
#pragma unroll 4
    for (int s = 0; s < S; ++s) {
      const float w = ex2(pl[(size_t)s * ROWS] - mx);
      const float2 v = *reinterpret_cast<const float2*>(po + (size_t)s * ROWS * 64);
      ws += w;
      o0 = fmaf(w, v.x, o0);
      o1 = fmaf(w, v.y, o1);
    }
    const float inv = __ldg(p.v_scale) / ws;
    o0 *= inv;
    o1 *= inv;
  }
  const float r = rsqrtf(warp_sum(o0 * o0 + o1 * o1) * (1.f / 64.f) + 1e-5f);
  const size_t tok = (size_t)b * L + j;
  const size_t stride = (size_t)(2 * p.Hq + 2 * p.Hkv) * 64;
  const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
      p.qkv + tok * stride + (size_t)(p.Hq + 2 * p.Hkv) * 64 + (size_t)qh * 64 + 2 * lane));
  const float s0 = 1.f / (1.f + expf(-gf.x));
  const float s1 = 1.f / (1.f + expf(-gf.y));
  *reinterpret_cast<__nv_bfloat162*>(p.out + tok * (size_t)p.Hq * 64 + (size_t)qh * 64 + 2 * lane) =
      __floats2bfloat162_rn(o0 * r * s0, o1 * r * s1);
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
                int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t W, int64_t page_shift, int64_t Gx,
                int64_t min_chunk, int64_t use_rope, double qk_sm) {
  TORCH_CHECK(L == 1 || L == 8, "k1: q_len must be 1 or 8");
  TORCH_CHECK(Hq == 8 * Hkv && Hkv % 2 == 0 && Hkv > 0, "k1: need Hq == 8*Hkv and even Hkv");
  TORCH_CHECK(Gx > B, "k1: Gx must exceed B");
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
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.W = (int)W; p.page_shift = (int)page_shift;
  p.pt_stride = (int)page_table.size(1); p.Gx = (int)Gx; p.min_chunk = (int)min_chunk; p.use_rope = (int)use_rope;
  p.qk_sm = (float)qk_sm;
  const dim3 grid((unsigned)Gx, (unsigned)(Hkv / 2));
  if (L == 1) launch_main<1>(p, grid, st);
  else launch_main<8>(p, grid, st);
  k1::k1_combine<<<(unsigned)(B * L * Hq), 32, 0, st>>>(p, (int)L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CPP_SRC + CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(
        name=f"k1_v0_{tag}",
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
print(f"[k1] FALLBACK={FALLBACK} SELF_TEST={SELF_TEST} built={_MOD is not None} tile={K1_TILE} stages={K1_STAGES}"
      f" min_chunk={K1_MIN_CHUNK} cps={K1_CPS}", flush=True)

_STATE = {"tested": False, "ok": False}
_SMS = {}
_OCC = {}
_INV_FREQ = {}


def _sms(dev):
    key = str(dev)
    if key not in _SMS:
        _SMS[key] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SMS[key]


def _ctas_per_sm(L, dev):
    """min(K1_CPS, resident k1_main<L> CTAs per SM); queried once per (L, device), eagerly."""
    key = (L, str(dev))
    if key not in _OCC:
        occ = int(_MOD.k1_occupancy(L))
        _OCC[key] = max(1, min(K1_CPS, occ))
        print(f"[k1] k1_main<{L}> occupancy {occ} CTAs/SM -> {_OCC[key]} CTAs/SM for Gx", flush=True)
    return _OCC[key]


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


def _supported(data):
    cfg = data["config"]
    hq, hkv, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"]
    kind = "sliding" if cfg["window"] else "full"
    return (cfg["batch"] >= K1_ROUTE_MIN_BATCH.get((kind, cfg["q_len"]), 1 << 30)
            and cfg["head_dim"] == 64 and hq == 8 * hkv and hkv % 2 == 0 and cfg["q_len"] in (1, 8)
            and page > 0 and (page & (page - 1)) == 0 and data["qkv"].dtype == torch.bfloat16
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8)


def _k1(data):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    dev = qkv.device
    hp = hkv // 2
    gx = max(max(1, (_sms(dev) * _ctas_per_sm(L, dev)) // hp), B + 1)
    part_o = torch.empty((hp, gx, 16 * L, 64), dtype=torch.float32, device=dev)
    part_lse = torch.empty((hp, gx, 16 * L), dtype=torch.float32, device=dev)
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    theta = cfg["rope_theta"]
    _MOD.k1_forward(qkv.contiguous(), data["seq_lens"].contiguous(), data["cache_loc"].contiguous(),
                    data["page_table"].contiguous(), data["k_cache"].view(torch.uint8),
                    data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], _inv_freq(theta, dev),
                    part_o, part_lse, out, B, L, hq, hkv, cfg["window"], cfg["page_size"].bit_length() - 1, gx,
                    K1_MIN_CHUNK, 1 if theta else 0, float(cfg["qk_scale_factor"] * cfg["sm_scale"]))
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


def _ref_attention(data, k_new, v_new):
    """FP32 semantics of TASK.md, attending the given new entries (e4m3 [B, L, Hkv, 64])."""
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, W, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["window"], cfg["page_size"]
    qkv = data["qkv"].float()
    q = _rms_t(qkv[..., :hq * 64].reshape(B, L, hq, 64))
    seq_l = data["seq_lens"].long()
    pos = seq_l[:, None] + torch.arange(L, device=q.device)
    if cfg["rope_theta"]:
        q = _rope_t(q, pos, cfg["rope_theta"])
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
    )
    try:
        for (B, ctx, L, sliding, page, kvs, seed, heads) in cases:
            d = _mk_case(dev, B, ctx, L, sliding, page, kvs, seed, heads[0], heads[1])
            out = _k1(d)
            torch.cuda.synchronize(dev)
            hkv = heads[1]
            pos = d["seq_lens"].long()[:, None] + torch.arange(L, device=dev)
            pg = d["config"]["page_size"]
            slots = torch.gather(d["page_table"].long(), 1, pos // pg) * pg + pos % pg
            k_new = d["k_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
            v_new = d["v_cache"].view(torch.uint8).view(-1, hkv, 64)[slots]
            ref = _ref_attention(d, k_new, v_new)
            tot, row = _rel_rows(out, ref)
            kref, vref = _ref_new_kv(d)
            kt, _ = _rel_rows(k_new.view(FP8).float(), kref)
            vt, _ = _rel_rows(v_new.view(FP8).float(), vref)
            finite = bool(torch.isfinite(out.float()).all())
            print(f"[k1] self-test B={B} ctx={ctx} L={L} {'sliding' if sliding else 'full'} page={page}"
                  f" kv_scale={kvs:g} heads={heads}: out rel {tot:.3e} worst row {row:.3e}; new K {kt:.3e}"
                  f" V {vt:.3e}", flush=True)
            if not (finite and tot < 2e-2 and row < 6e-2 and kt < 3e-2 and vt < 3e-2):
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
    if _MOD is None or not _supported(data):
        return _fallback(data)
    if not _STATE["tested"]:
        if torch.cuda.is_current_stream_capturing():
            if not _STATE.get("warned_capture"):
                _STATE["warned_capture"] = True
                print("[k1] first call arrived under graph capture before any eager call: FA3 fallback for this capture", flush=True)
            return _fallback(data)
        _STATE["ok"] = _self_test(data["qkv"].device) if SELF_TEST else True
        _STATE["tested"] = True
    if not _STATE["ok"]:
        return _fallback(data)
    return _k1(data)

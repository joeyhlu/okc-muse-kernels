# okc: problem=muse_bf16_proj
# nonce ps-lab-2 (v0.1)
"""PS v0 + KT1 (notes/study/dense_independent.md section 6). CHECK ONLY, never submit for score.

PS = packed-stream kernel for m <= 32: P12 lossless 12-bit weights (the P1 pack, verbatim), exact 16-cluster x 8-CTA
partition (cluster = rows, CTA rank = K slice), per-warp cp.async.bulk rings (or LDG register streaming), x staged once
per CTA in smem (ldmatrix B fragments), mma.sync m16n8k16 swap-AB, one cluster DSMEM reduction, BF16/FP32 out.
R32 = FP32 router for m <= 8 (FFMA, same partition). Router m 9..32 = PS over the [hi; lo] pack (hi P12, lo RAW),
hi + lo summed in the cluster reduction.

prepare(): baseline state + build + pack + self-tests (fp64 reference, probe rows, outliers) + the KT1 lab (CUDA-graph
replays after a 256 MB READ flush, median minus an empty-kernel graph) + route selection. kernel_fn(): PS / R32 where
the self-test passed, SGLang's path everywhere else.
"""

from typing import TypedDict

import hashlib
import statistics
import time

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]; up to 8 tokens, only rows 0-1023 (this GPU's slice)
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [M, K]
    config: Config


ROUTER_FP32_MAX_TOKENS = 8
LAB_DEADLINE_S = 230.0
FLUSH_FLOATS = (256 << 20) // 4
REPS = 41

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void ps_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
             torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor err, int64_t M, int64_t K, int64_t ldo,
             int64_t out_f32, int64_t R, int64_t Q, int64_t pair, int64_t nhalf, int64_t nclu, int64_t ng, int64_t mode,
             int64_t ld, int64_t dep);
std::vector<int64_t> ps_info(int64_t ng, int64_t mode, int64_t ld, int64_t dep, int64_t smem, int64_t nclu);
int64_t ps_smem(int64_t ng, int64_t mode, int64_t ld, int64_t dep, int64_t R, int64_t Q, int64_t slot_b);
void r32_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t M, int64_t K);
std::vector<int64_t> r32_info();
void ps_empty();
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

namespace cg = cooperative_groups;

namespace ps {

constexpr uint32_t KC4 = 0x38383838u;
constexpr uint32_t CMASK = 0x87008700u;
constexpr int NW = 8;      // consumer warps per CTA
constexpr int CL = 8;      // cluster size = K split
constexpr int BIG = 0x7fffffff;

struct Args {
  const __nv_bfloat16* x; int M; int K;
  const uint8_t* pk; const int64_t* toff; const float* scale; const int* eoff; const int2* esc; const uint8_t* fmt;
  void* out; int ldo; int out_f32;
  int R; int Q; int pair; int nhalf;
  int xs_w; int part_off; int ring_off; int slot_b;
  int* err;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t rotr(uint32_t x, int r) { return __funnelshift_r(x, x, r); }

__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ bool mb_try(uint64_t* b, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return ok != 0u;
}
// bounded wait: gives up after 2^20 polls (or as soon as another wait has failed) and raises the error flag (no trap)
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t parity, int* err) {
  for (int i = 0; i < (1 << 20); ++i) {
    if (mb_try(b, parity)) return;
    if ((i & 1023) == 1023 && *reinterpret_cast<volatile int*>(err)) return;
  }
  atomicExch(err, 1);
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ uint4 ldw(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float4 ldw4f(const float* p) {
  float4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];\n"
               : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldx(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 lds128(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(smem_u32(p)) : "memory");
  return r;
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// P12 k-tile decode (verbatim P1 round-2b): lane's 3 x 16 B of one SB -> 4 scaled bf16x2 A registers
__device__ __forceinline__ void dec_kt(const uint4 (&p)[4], const int kt, uint32_t (&a)[4]) {
  const uint4 bc = p[kt >> 1];
  const uint32_t w0 = (kt & 1) ? bc.z : bc.x;
  const uint32_t w1 = (kt & 1) ? bc.w : bc.y;
  const uint32_t cw = kt == 0 ? p[2].x : kt == 1 ? p[2].y : kt == 2 ? p[2].z : p[2].w;
  a[0] = __byte_perm(w0, KC4, 0x4140) | (cw & CMASK);
  a[1] = __byte_perm(w0, KC4, 0x4342) | (rotr(cw, 4) & CMASK);
  a[2] = __byte_perm(w1, KC4, 0x4140) | (rotr(cw, 8) & CMASK);
  a[3] = __byte_perm(w1, KC4, 0x4342) | (rotr(cw, 12) & CMASK);
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
__device__ __forceinline__ void esc_setup(const Args& A, int tile, int sb_lo, int nsb, int lane, uint32_t& em, uint32_t& ev,
                                          int& cur, int& end, int& nkey, bool skip) {
  em = 0u; ev = 0u; cur = 32; end = 0; nkey = BIG;
  if (skip) return;
  const int o0 = A.eoff[tile], o1 = A.eoff[tile + 1];
  const int idx = o0 + lane;
  int2 e = make_int2(0, 0);
  if (idx < o1) e = A.esc[idx];
  em = (uint32_t)e.x; ev = (uint32_t)e.y;
  const int esb = (int)(em & 0xFFFFu);
  const bool in = idx < o1 && esb >= sb_lo && esb < sb_lo + nsb;
  const uint32_t msk = __ballot_sync(0xffffffffu, in);
  cur = msk ? __ffs(msk) - 1 : 32;
  end = msk ? 32 - __clz(msk) : 0;
  const uint32_t mf = __shfl_sync(0xffffffffu, em, cur & 31);
  nkey = msk ? esc_key(mf, sb_lo) : BIG;
}
__device__ __forceinline__ int tile_of(const Args& A, int cid, int tl) {
  if (!A.pair) return cid * A.R + tl;
  const int h = A.R >> 1;
  return tl < h ? cid * h + tl : A.nhalf + cid * h + (tl - h);
}
__device__ __forceinline__ void sb_src(const Args& A, int cid, int crank, int s, const uint8_t*& src, int& bytes) {
  const int tl = s / A.Q, sbq = s - tl * A.Q;
  const int tile = tile_of(A, cid, tl);
  bytes = A.fmt[tile] ? 2048 : 1536;
  src = A.pk + A.toff[tile] + (long long)(crank * A.Q + sbq) * bytes;
}
__device__ __forceinline__ void issue_sb(const Args& A, int cid, int crank, int s, uint8_t* dst, uint64_t* bar) {
  const uint8_t* src; int bytes;
  sb_src(A, cid, crank, s, src, bytes);
  mb_expect(bar, (uint32_t)bytes);
  bulk_g2s(dst, src, (uint32_t)bytes, bar);
}
__device__ __forceinline__ void load_sb(const Args& A, int cid, int crank, int s, int lane, uint4 (&w)[4]) {
  const uint8_t* src; int bytes;
  sb_src(A, cid, crank, s, src, bytes);
  src += lane * 16;
  w[0] = ldw(src); w[1] = ldw(src + 512); w[2] = ldw(src + 1024);
  w[3] = bytes == 2048 ? ldw(src + 1536) : make_uint4(0u, 0u, 0u, 0u);
}
template <int NG, int MODE>
__device__ __forceinline__ void flush_acc(const Args& A, float* part, int tl, int tile, float (&acc)[NG][4], int lane, uint32_t& px) {
  constexpr int TP = NG * 8;
  const int g = lane >> 2, t4 = lane & 3;
  if (MODE == 3) { acc[0][0] += __uint_as_float(px & 0x007FFFFFu); px = 0u; }
  const float sc = A.scale[tile];
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) {
    float* p0 = part + ((size_t)tl * 16 + g) * TP + gi * 8 + 2 * t4;
    float* p1 = part + ((size_t)tl * 16 + g + 8) * TP + gi * 8 + 2 * t4;
    atomicAdd(p0, acc[gi][0] * sc); atomicAdd(p0 + 1, acc[gi][1] * sc);
    atomicAdd(p1, acc[gi][2] * sc); atomicAdd(p1 + 1, acc[gi][3] * sc);
    acc[gi][0] = 0.f; acc[gi][1] = 0.f; acc[gi][2] = 0.f; acc[gi][3] = 0.f;
  }
}

// out[t, row] = sum_k x[t, k] W[row, k] for t < M <= 8 NG. grid = 8 * nclu CTAs, cluster 8 (rank = K slice of 64 Q).
// MODE 0: full; 1: B fragments from global x per SB (P1-style per-token path); 3: stream probe (no decode / mma).
// LD 0: per-warp cp.async.bulk ring of DEP SBs; 1: LDG register streaming DEP SBs ahead.
template <int NG, int MODE, int LD, int DEP>
__global__ void __launch_bounds__(NW * 32, 1) ps_kernel(const __grid_constant__ Args A) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t bar_x;
  __shared__ __align__(8) uint64_t bars[NW][DEP];
  constexpr int TP = NG * 8;
  constexpr bool XS = MODE != 1;
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int crank = (int)cluster.block_rank();
  const int cid = (int)blockIdx.x / CL;
  const int R = A.R, Q = A.Q, M = A.M;
  float* part = reinterpret_cast<float*>(smem + A.part_off);
  uint8_t* xs = smem;
  uint8_t* ring = smem + A.ring_off;
  const int RQ = R * Q;
  const int s_lo = (int)((long long)warp * RQ / NW), s_hi = (int)((long long)(warp + 1) * RQ / NW);

  if (LD == 0 && lane == 0) {
#pragma unroll
    for (int d = 0; d < DEP; ++d) mb_init(&bars[warp][d], 1);
  }
  if (tid == 0) mb_init(&bar_x, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  if (XS && tid == 0) {
    const uint32_t rowb = (uint32_t)Q * 128u;
    mb_expect(&bar_x, rowb * (uint32_t)M);
    for (int t = 0; t < M; ++t)
      bulk_g2s(xs + (size_t)t * A.xs_w * 4, A.x + (size_t)t * A.K + (size_t)crank * Q * 64, rowb, &bar_x);
  }
  uint4 wr[LD == 1 ? DEP : 1][4];
  if (LD == 0) {
    if (lane == 0) {
#pragma unroll
      for (int d = 0; d < DEP; ++d)
        if (s_lo + d < s_hi) issue_sb(A, cid, crank, s_lo + d, ring + (size_t)(warp * DEP + d) * A.slot_b, &bars[warp][d]);
    }
  } else {
#pragma unroll
    for (int d = 0; d < DEP; ++d)
      if (s_lo + d < s_hi) load_sb(A, cid, crank, s_lo + d, lane, wr[d]);
  }
  for (int i = tid; i < R * 16 * TP; i += NW * 32) part[i] = 0.f;
  if (XS) {
    uint32_t* xw = reinterpret_cast<uint32_t*>(xs);
    for (int i = tid; i < (TP - M) * A.xs_w; i += NW * 32) xw[(size_t)M * A.xs_w + i] = 0u;
  }
  __syncthreads();
  if (XS) mb_wait(&bar_x, 0u, A.err);

  float acc[NG][4];
#pragma unroll
  for (int gi = 0; gi < NG; ++gi) { acc[gi][0] = 0.f; acc[gi][1] = 0.f; acc[gi][2] = 0.f; acc[gi][3] = 0.f; }
  int seg_tl = -1, seg_tile = 0, seg_fmt = 0, seg_sb_lo = 0;
  uint32_t em = 0u, ev = 0u, px = 0u;
  int cur = 32, end = 0, nkey = BIG;
  const int g = lane >> 2, t4 = lane & 3;
  const uint32_t xs_base = smem_u32(xs);
  const uint32_t lrow = (uint32_t)((((lane & 7) + 8 * ((lane >> 4) & 1)) * A.xs_w) * 4 + ((lane >> 3) & 1) * 16);
  for (int s0 = s_lo; s0 < s_hi; s0 += DEP) {
#pragma unroll
    for (int d = 0; d < DEP; ++d) {
      const int s = s0 + d;
      if (s < s_hi) {
        const int tl = s / Q, sbq = s - tl * Q;
        if (tl != seg_tl) {
          if (seg_tl >= 0) flush_acc<NG, MODE>(A, part, seg_tl, seg_tile, acc, lane, px);
          seg_tl = tl; seg_tile = tile_of(A, cid, tl); seg_fmt = A.fmt[seg_tile];
          seg_sb_lo = crank * Q + sbq;
          const int nsb = min(s_hi, (tl + 1) * Q) - s;
          esc_setup(A, seg_tile, seg_sb_lo, nsb, lane, em, ev, cur, end, nkey, MODE == 3 || seg_fmt != 0);
        }
        uint4 w[4];
        if (LD == 0) {
          mb_wait(&bars[warp][d], (uint32_t)(((s0 - s_lo) / DEP) & 1), A.err);
          const uint8_t* sp = ring + (size_t)(warp * DEP + d) * A.slot_b + lane * 16;
          w[0] = lds128(sp); w[1] = lds128(sp + 512); w[2] = lds128(sp + 1024);
          w[3] = seg_fmt ? lds128(sp + 1536) : make_uint4(0u, 0u, 0u, 0u);
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          __syncwarp();
          if (lane == 0 && s + DEP < s_hi)
            issue_sb(A, cid, crank, s + DEP, ring + (size_t)(warp * DEP + d) * A.slot_b, &bars[warp][d]);
        } else {
          w[0] = wr[d][0]; w[1] = wr[d][1]; w[2] = wr[d][2]; w[3] = wr[d][3];
          if (s + DEP < s_hi) load_sb(A, cid, crank, s + DEP, lane, wr[d]);
        }
        if (MODE == 3) {
          px ^= w[0].x ^ w[0].y ^ w[0].z ^ w[0].w ^ w[1].x ^ w[1].y ^ w[1].z ^ w[1].w;
          px ^= w[2].x ^ w[2].y ^ w[2].z ^ w[2].w ^ w[3].x ^ w[3].y ^ w[3].z ^ w[3].w;
        } else {
          const int u = crank * Q + sbq - seg_sb_lo;
#pragma unroll
          for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[4];
            if (seg_fmt) {
              a[0] = w[kt].x; a[1] = w[kt].y; a[2] = w[kt].z; a[3] = w[kt].w;
            } else {
              dec_kt(w, kt, a);
              const int key = u * 4 + kt;
              if (key == nkey) esc_patch4(a, key, seg_sb_lo, lane, em, ev, cur, end, nkey);
            }
            uint32_t b[NG][2];
            if (MODE == 1) {
              const int kg = (crank * Q + sbq) * 64 + 16 * kt + 2 * t4;
#pragma unroll
              for (int gi = 0; gi < NG; ++gi) {
                const int tok = gi * 8 + g;
                if (tok < M) {
                  const uint32_t* p = reinterpret_cast<const uint32_t*>(A.x + (size_t)tok * A.K + kg);
                  b[gi][0] = ldx(p); b[gi][1] = ldx(p + 4);
                } else {
                  b[gi][0] = 0u; b[gi][1] = 0u;
                }
              }
            } else {
              const uint32_t kofs = (uint32_t)((sbq * 64 + 16 * kt) * 2);
              if (NG == 1) {
                uint32_t r2[2];
                ldsm2(r2, xs_base + lrow + kofs);
                b[0][0] = r2[0]; b[0][1] = r2[1];
              } else {
#pragma unroll
                for (int gp = 0; gp < NG / 2; ++gp) {
                  uint32_t r[4];
                  ldsm4(r, xs_base + lrow + (uint32_t)(gp * 16 * A.xs_w * 4) + kofs);
                  b[2 * gp][0] = r[0]; b[2 * gp][1] = r[1]; b[2 * gp + 1][0] = r[2]; b[2 * gp + 1][1] = r[3];
                }
              }
            }
#pragma unroll
            for (int gi = 0; gi < NG; ++gi) mma16816(acc[gi], a, b[gi][0], b[gi][1]);
          }
        }
      }
    }
  }
  if (seg_tl >= 0) flush_acc<NG, MODE>(A, part, seg_tl, seg_tile, acc, lane, px);
  __syncthreads();
  cluster.sync();
  const int orows = A.pair ? 8 * R : 16 * R;
  const int rows_per = orows / CL;
  const int orow0 = cid * orows;
  const int nq = (M + 3) >> 2;                       // token quads: one float4 per (row, quad, rank)
  for (int i = tid; i < rows_per * nq; i += NW * 32) {
    const int rr = crank * rows_per + i % rows_per;
    const int tq = i / rows_per;
    float4 pv[CL], pl[CL];
    const size_t o0 = A.pair ? ((size_t)(rr >> 4) * 16 + (rr & 15)) * TP + 4 * tq : (size_t)rr * TP + 4 * tq;
    const size_t o1 = A.pair ? ((size_t)((rr >> 4) + (R >> 1)) * 16 + (rr & 15)) * TP + 4 * tq : 0;
#pragma unroll
    for (int q = 0; q < CL; ++q) {
      const float* rp = cluster.map_shared_rank(part, q);
      pv[q] = *reinterpret_cast<const float4*>(rp + o0);
      pl[q] = A.pair ? *reinterpret_cast<const float4*>(rp + o1) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < CL; ++q) {
      v[0] += pv[q].x; v[1] += pv[q].y; v[2] += pv[q].z; v[3] += pv[q].w;
      if (A.pair) { v[0] += pl[q].x; v[1] += pl[q].y; v[2] += pl[q].z; v[3] += pl[q].w; }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int tok = 4 * tq + j;
      if (tok < M) {
        if (A.out_f32) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + orow0 + rr] = v[j];
        else reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tok * A.ldo + orow0 + rr] = __float2bfloat16_rn(v[j]);
      }
    }
  }
  cluster.sync();
}

// FP32 router, M <= 8: cluster c = rows [16c, 16c + 16), rank j = k [1024 j, 1024 j + 1024); warp w = rows 2w, 2w + 1.
__global__ void __launch_bounds__(256, 1) r32_kernel(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                                     float* __restrict__ out, int M, int K) {
  __shared__ __align__(16) float xs[8][1024];
  __shared__ float part[16][8];
  cg::cluster_group cluster = cg::this_cluster();
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int crank = (int)cluster.block_rank(), cid = (int)blockIdx.x / 8;
  const int k0 = crank * 1024;
  const float* w0 = w + (size_t)(16 * cid + 2 * warp) * K + k0 + 4 * lane;
  const float* w1 = w0 + K;
  float4 a[8], b[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { a[i] = ldw4f(w0 + 128 * i); b[i] = ldw4f(w1 + 128 * i); }
  for (int i = tid; i < 8 * 512; i += 256) {
    const int t = i >> 9, k = (i & 511) * 2;
    float2 v = make_float2(0.f, 0.f);
    if (t < M) {
      const __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(x + (size_t)t * K + k0 + k);
      v = __bfloat1622float2(p);
    }
    xs[t][k] = v.x; xs[t][k + 1] = v.y;
  }
  __syncthreads();
  float s0[8], s1[8];
#pragma unroll
  for (int t = 0; t < 8; ++t) { s0[t] = 0.f; s1[t] = 0.f; }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const float4 xv = *reinterpret_cast<const float4*>(&xs[t][4 * lane + 128 * i]);
      s0[t] = fmaf(a[i].x, xv.x, s0[t]); s0[t] = fmaf(a[i].y, xv.y, s0[t]);
      s0[t] = fmaf(a[i].z, xv.z, s0[t]); s0[t] = fmaf(a[i].w, xv.w, s0[t]);
      s1[t] = fmaf(b[i].x, xv.x, s1[t]); s1[t] = fmaf(b[i].y, xv.y, s1[t]);
      s1[t] = fmaf(b[i].z, xv.z, s1[t]); s1[t] = fmaf(b[i].w, xv.w, s1[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      s0[t] += __shfl_xor_sync(0xffffffffu, s0[t], off);
      s1[t] += __shfl_xor_sync(0xffffffffu, s1[t], off);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < 8; ++t) { part[2 * warp][t] = s0[t]; part[2 * warp + 1][t] = s1[t]; }
  }
  __syncthreads();
  cluster.sync();
  if (tid < 2 * M) {
    const int r = 2 * crank + (tid & 1), t = tid >> 1;
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < 8; ++q) v += cluster.map_shared_rank(&part[0][0], q)[r * 8 + t];
    out[(size_t)t * 256 + 16 * cid + r] = v;
  }
  cluster.sync();
}

__global__ void empty_kernel() {}

}  // namespace ps

static inline int64_t al128(int64_t v) { return (v + 127) & ~(int64_t)127; }

static void smem_layout(int ng, int mode, int ld, int dep, int R, int Q, int slot_b, int& xs_w, int& part_off, int& ring_off,
                        int& total) {
  const int TP = ng * 8;
  xs_w = Q * 32 + 4;
  const int64_t xs_b = mode != 1 ? al128((int64_t)TP * xs_w * 4) : 0;
  const int64_t part_b = al128((int64_t)R * 16 * TP * 4);
  const int64_t ring_b = ld == 0 ? (int64_t)ps::NW * dep * slot_b : 0;
  part_off = (int)xs_b;
  ring_off = (int)(xs_b + part_b);
  total = (int)(xs_b + part_b + ring_b);
}

int64_t ps_smem(int64_t ng, int64_t mode, int64_t ld, int64_t dep, int64_t R, int64_t Q, int64_t slot_b) {
  int xs_w, po, ro, tot;
  smem_layout((int)ng, (int)mode, (int)ld, (int)dep, (int)R, (int)Q, (int)slot_b, xs_w, po, ro, tot);
  return tot;
}

template <int NG, int MODE, int LD, int DEP>
static void launch_t(const ps::Args& A, int nclu, int smem) {
  auto kern = ps::ps_kernel<NG, MODE, LD, DEP>;
  static int set = 0;
  if (set < smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024));
    set = 220 * 1024;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(nclu * ps::CL), 1, 1);
  cfg.blockDim = dim3(ps::NW * 32, 1, 1);
  cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = ps::CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, A));
}

template <int NG, int MODE, int LD, int DEP>
static std::vector<int64_t> info_t(int smem, int nclu) {
  auto kern = ps::ps_kernel<NG, MODE, LD, DEP>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(nclu * ps::CL), 1, 1);
  cfg.blockDim = dim3(ps::NW * 32, 1, 1);
  cfg.dynamicSmemBytes = (size_t)smem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = ps::CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int ncl = -1;
  if (cudaOccupancyMaxActiveClusters(&ncl, kern, &cfg) != cudaSuccess) { cudaGetLastError(); ncl = -1; }
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)ncl};
}

#define PS_INST(X) X(1, 0, 1, 4) X(1, 0, 1, 6) X(1, 0, 0, 4) X(1, 1, 1, 4) X(1, 3, 1, 4) X(2, 0, 1, 4) X(2, 1, 1, 4) \
                   X(4, 0, 1, 4) X(4, 1, 1, 4) X(4, 3, 1, 4)

void ps_gemm(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff,
             torch::Tensor esc, torch::Tensor fmt, torch::Tensor out, torch::Tensor err, int64_t M, int64_t K, int64_t ldo,
             int64_t out_f32, int64_t R, int64_t Q, int64_t pair, int64_t nhalf, int64_t nclu, int64_t ng, int64_t mode,
             int64_t ld, int64_t dep) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(0) == M && x.size(1) == K, "ps: x");
  TORCH_CHECK(M >= 1 && M <= ng * 8 && K % 512 == 0 && Q * 8 * 64 == K, "ps: M / K / Q");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(pk.data_ptr()) % 16 == 0, "ps: align");
  TORCH_CHECK(out.stride(1) == 1 && out.stride(0) == ldo, "ps: out stride");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  ps::Args A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.K = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = out.data_ptr(); A.ldo = (int)ldo; A.out_f32 = (int)out_f32;
  A.R = (int)R; A.Q = (int)Q; A.pair = (int)pair; A.nhalf = (int)nhalf;
  A.slot_b = pair ? 2048 : 1536;
  int total = 0;
  smem_layout((int)ng, (int)mode, (int)ld, (int)dep, (int)R, (int)Q, A.slot_b, A.xs_w, A.part_off, A.ring_off, total);
  TORCH_CHECK(total <= 220 * 1024, "ps: smem ", total);
  A.err = err.data_ptr<int>();
#define PS_LAUNCH(NG_, MO_, LD_, DE_) \
  if (ng == NG_ && mode == MO_ && ld == LD_ && dep == DE_) { launch_t<NG_, MO_, LD_, DE_>(A, (int)nclu, total); return; }
  PS_INST(PS_LAUNCH)
#undef PS_LAUNCH
  TORCH_CHECK(false, "ps: no instantiation for ng/mode/ld/dep");
}

std::vector<int64_t> ps_info(int64_t ng, int64_t mode, int64_t ld, int64_t dep, int64_t smem, int64_t nclu) {
#define PS_INFO(NG_, MO_, LD_, DE_) \
  if (ng == NG_ && mode == MO_ && ld == LD_ && dep == DE_) return info_t<NG_, MO_, LD_, DE_>((int)smem, (int)nclu);
  PS_INST(PS_INFO)
#undef PS_INFO
  return {0, 0, -1};
}

void r32_gemv(torch::Tensor w, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t M, int64_t K) {
  TORCH_CHECK(w.scalar_type() == at::kFloat && w.size(0) == 256 && w.size(1) == K && K == 8192, "r32: w");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && M >= 1 && M <= 8, "r32: x");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(128, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 8; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ps::r32_kernel, (const float*)w.data_ptr<float>(),
                                    (const __nv_bfloat16*)x.data_ptr(), out.data_ptr<float>(), (int)M, (int)K));
}

std::vector<int64_t> r32_info() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ps::r32_kernel));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes};
}

void ps_empty() {
  ps::empty_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

# ===================================================================== P12 packer (verbatim from p1_module.py)
P1_C = 0x70
P1_ESC_MAX = 32
P1_RAW_GROUP = 16


def p1_lane_maps(device):
    L = torch.arange(32, device=device).view(32, 1, 1)
    o = torch.arange(16, device=device).view(1, 16, 1)
    h = torch.arange(2, device=device).view(1, 1, 2)
    g, t, kt, i = L // 4, L % 4, o // 4, o % 4
    row = (g + 8 * (i % 2) + 0 * h).expand(32, 16, 2)
    kk = (16 * kt + 2 * t + 8 * (i // 2) + h).expand(32, 16, 2)
    return row, kk


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
            "fmt_cpu": raw_c, "eoff": eoff, "esc": esc, "bytes": total + nrt * 17 + 8 * int(keep.sum()),
            "n_esc": int(keep.sum()), "n_raw": int(raw_tile.sum()), "pack_s": time.time() - t0}


# ===================================================================== build / call
_MOD = None
# v0.1 (after check 04708abf5dd74996): LDG arms deeper (v0's D2 = 24 KB/SM in flight ran at 1.2 TB/s), the ring arm only at
# T <= 8 with D4 (v0's 123-197 KB smem allowed only 15 co-resident 8-CTA clusters -> a second wave), the probe and gx arms
# on the LDG engine, float4 DSMEM epilogue (v0's scalar remote reads cost ~1 us each: o T=32 91 us).
ARMS = {"ldg4": (0, 1, 4), "ldg6": (0, 1, 6), "ring": (0, 0, 4), "gx": (1, 1, 4), "probe": (3, 1, 4)}
INSTS = [(1, 0, 1, 4), (1, 0, 1, 6), (1, 0, 0, 4), (1, 1, 1, 4), (1, 3, 1, 4), (2, 0, 1, 4), (2, 1, 1, 4),
         (4, 0, 1, 4), (4, 1, 1, 4), (4, 3, 1, 4)]


def _build():
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((CPP_SRC + CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    _MOD = load_inline(name=f"ps_lab_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                       functions=["ps_gemm", "ps_info", "ps_smem", "r32_gemv", "r32_info", "ps_empty"],
                       extra_cuda_cflags=flags, verbose=False)
    print(f"PS build {time.time() - t0:.1f}s", flush=True)
    return _MOD


def _ng(m):
    return 1 if m <= 8 else (2 if m <= 16 else 4)


def _geo(P, n_out, pair=False, nclu=16):
    """nclu clusters x 8 CTAs (16 x 8 = 128 SMs by default): cluster = n_out/16/nclu row tiles (R; pair: R/2 hi + R/2 lo
    tiles), CTA rank = K/8. nclu = 8 (64 CTAs) is the per-SM bandwidth-cap probe of KT1."""
    out_tiles = n_out // 16
    if out_tiles % nclu or P["nsb"] % 8:
        return None
    R = (out_tiles // nclu) * (2 if pair else 1)
    return {"P": P, "R": R, "Q": P["nsb"] // 8, "pair": 1 if pair else 0, "nhalf": out_tiles if pair else 0,
            "nclu": nclu, "n_out": n_out, "f32": 1 if pair else 0}


def _ps(st, G, x, arm, out=None):
    mode, ld, dep = ARMS[arm]
    m = x.shape[0]
    P = G["P"]
    if out is None:
        out = torch.empty((m, G["n_out"]), dtype=torch.float32 if G["f32"] else torch.bfloat16, device=x.device)
    st["mod"].ps_gemm(x, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], out, st["err"], m, P["K"],
                      out.stride(0), G["f32"], G["R"], G["Q"], G["pair"], G["nhalf"], G["nclu"], _ng(m), mode, ld, dep)
    return out


def _r32(st, x, out=None):
    m = x.shape[0]
    if out is None:
        out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    st["mod"].r32_gemv(st["router"], x, out, st["err"], m, 8192)
    return out


def _baseline(state, x, layer, n):
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


# ===================================================================== self-test inputs (own generator, judge-like)
def _input(m, k, seed, outliers=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device="cuda")
    if outliers:
        idx = torch.randperm(k, generator=gen, device="cuda")[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device="cuda") < 1 / 8
    probes[torch.randint(m, (1,), generator=gen, device="cuda")] = True
    cols = torch.randint(k, (m,), generator=gen, device="cuda")
    signs = torch.randint(2, (m,), generator=gen, device="cuda") * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device="cuda")
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16).contiguous()


def _check(out, x, wref, tol):
    ref = x.double() @ wref.double().T
    o = out.double()
    if not torch.isfinite(o).all():
        return False, "nonfinite"
    rel = float((o - ref).norm() / ref.norm().clamp(min=1e-30))
    rows = ((x != 0).sum(-1) == 1).nonzero().flatten()
    pr = 0.0
    if len(rows):
        pr = float(((o[rows] - ref[rows]).norm(dim=-1) / ref[rows].norm(dim=-1).clamp(min=1e-30)).max())
    rowrel = (o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=float(ref.norm(dim=-1).mean()) * 0.1)
    ok = rel <= tol and pr <= 1e-3 and float(rowrel.max()) <= 5 * tol
    return ok, f"rel {rel:.2e} probe {pr:.1e} row {float(rowrel.max()):.1e}"


# ===================================================================== timing
def _graph(fn):
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _time(L, fn, reps=REPS):
    g = _graph(fn)
    ts = []
    for r in range(reps):
        L["flush"].sum()
        L["e0"].record()
        g.replay()
        L["e1"].record()
        torch.cuda.synchronize()
        if r:
            ts.append(L["e0"].elapsed_time(L["e1"]) * 1000.0)
    return statistics.median(ts), min(ts)


# ===================================================================== prepare / kernel_fn
DENSE = ("qkv_gate", "o", "pooler", "unpooler")


def prepare(weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["routes"] = {}
    try:
        _ps_prepare(state, weights, hi, lo, t0)
    except Exception as ex:  # noqa: BLE001
        print(f"PS prepare ERR {ex!r}"[:900], flush=True)
        state["routes"] = {}
    torch.cuda.synchronize()
    print(f"PS prepare done {time.time() - t0:.1f}s routes {sorted(state['routes'])}", flush=True)
    return state


def _ps_prepare(state, weights, hi, lo, t0):
    mod = _build()
    st = {"mod": mod, "err": torch.zeros(4, dtype=torch.int32, device="cuda"), "router": state["router"]}
    state["ps"] = st
    packs = {}
    for name in DENSE:
        P = p1_pack(weights[name], name=name)
        packs[name] = P
        print(f"PS pack {name} {P['bytes'] / 1e6:.2f} MB esc {P['n_esc']} raw {P['n_raw']} {P['pack_s']:.1f}s", flush=True)
    PR = p1_pack(torch.cat((hi, lo), 0).contiguous(), name="router_hilo")
    print(f"PS pack router hi|lo {PR['bytes'] / 1e6:.2f} MB esc {PR['n_esc']} raw {PR['n_raw']}/32", flush=True)
    G = {("qkv_gate", 0): _geo(packs["qkv_gate"], 2304), ("o", 0): _geo(packs["o"], 8192),
         ("pooler", 0): _geo(packs["pooler"], 4096), ("unpooler", 1): _geo(packs["unpooler"], 1024),
         ("unpooler", 0): _geo(packs["unpooler"], 8192), ("router", 0): _geo(PR, 256, pair=True)}
    st["geo"] = G
    wref = {"qkv_gate": weights["qkv_gate"], "o": weights["o"], "pooler": weights["pooler"],
            "unpooler": weights["unpooler"]}
    # ---- info
    for ng, mo, ld, dep in INSTS:
        sm = int(mod.ps_smem(ng, mo, ld, dep, 16, 16, 1536))
        r, lb, ncl = mod.ps_info(ng, mo, ld, dep, sm, 16)
        print(f"PS info ng{ng} m{mo} ld{ld} d{dep}: regs {r} local {lb} B smem(pooler) {sm} maxclusters {ncl}", flush=True)
    print(f"PS info r32 regs/local {list(mod.r32_info())}", flush=True)
    # ---- self-tests (every arm, every routed geometry, m covering the buckets and the judge tests)
    ok_arm = {}          # (layer, sl, arm) -> passed at every tested m; okng[(layer, sl, arm, ng)] per token bucket
    okng = {}
    inst = {(ng, mo, ld, dep) for ng, mo, ld, dep in INSTS}
    for (layer, sl), g in G.items():
        if g is None:
            print(f"PS geo {layer}/{sl} unsupported", flush=True)
            continue
        if layer == "router":
            ms = (9, 16, 32)
        elif sl == 1:
            ms = (1, 5, 8)
        else:
            ms = (1, 7, 8, 9, 16, 32) if layer != "unpooler" else (9, 16, 32)
        for arm in ARMS:
            if arm == "probe" or (layer == "router" and arm != "ldg4"):
                continue
            mo, ld, dep = ARMS[arm]
            msgs = []
            for i, m in enumerate(ms):
                ng = _ng(m)
                if (ng, mo, ld, dep) not in inst:
                    continue
                good = okng.get((layer, sl, arm, ng), True)
                for outl in (False, True):
                    if not good:
                        break
                    x = _input(m, g["P"]["K"], 1000 + 17 * i + (7 if outl else 0), outl)
                    if int(st["err"][0]):
                        ok, msg = False, "err flag set (bounded wait timeout)"
                        good = False
                        msgs.append(msg)
                        break
                    try:
                        out = _ps(st, g, x, arm)
                        torch.cuda.synchronize()
                        if layer == "router":
                            ok, msg = _check(out, x, state["router"], 2e-5)
                        else:
                            ok, msg = _check(out, x, wref[layer][: g["n_out"]], 4e-3)
                    except Exception as ex:  # noqa: BLE001
                        ok, msg = False, f"EXC {ex!r}"[:200]
                    if not ok:
                        good = False
                        msgs.append(f"m{m} outl{int(outl)} {msg}")
                    elif m == ms[-1] and not outl:
                        msgs.append(f"m{m} {msg}")
                okng[(layer, sl, arm, ng)] = good
            tested = [v for (l2, s2, a2, _), v in okng.items() if (l2, s2, a2) == (layer, sl, arm)]
            ok_arm[(layer, sl, arm)] = bool(tested) and all(tested)
            print(f"PS test {layer}/{sl} {arm}: {'PASS' if ok_arm[(layer, sl, arm)] else 'FAIL'} {'; '.join(msgs)[:300]}",
                  flush=True)
    r32ok = True
    try:
        for i, m in enumerate((1, 5, 8)):
            for outl in (False, True):
                x = _input(m, 8192, 3000 + i + (7 if outl else 0), outl)
                ok, msg = _check(_r32(st, x), x, state["router"], 2e-5)
                torch.cuda.synchronize()
                if not ok:
                    r32ok = False
                    print(f"PS test r32 FAIL m{m} {msg}", flush=True)
    except Exception as ex:  # noqa: BLE001
        r32ok = False
        print(f"PS test r32 EXC {ex!r}"[:300], flush=True)
    print(f"PS test r32: {'PASS' if r32ok else 'FAIL'}; err flag {int(st['err'][0])}", flush=True)
    _set_routes(state, G, okng, r32ok, {})
    if int(st["err"][0]):
        print("PS err flag set after self-tests -> no PS routes, no lab", flush=True)
        state["routes"] = {}
        return
    try:
        res = _kt1_lab(state, st, G, packs, ok_arm, r32ok, t0)
        _set_routes(state, G, okng, r32ok, res)
    except Exception as ex:  # noqa: BLE001
        print(f"KT1 lab ERR {ex!r}"[:600], flush=True)
    if int(st["err"][0]):
        print("PS err flag set -> no PS routes", flush=True)
        state["routes"] = {}


def _set_routes(state, G, okng, r32ok, res):
    """Best measured PS arm per (layer, slice, bucket) among the self-tested ones (ring when unmeasured)."""
    routes = {}
    for (layer, sl), g in G.items():
        if g is None or layer == "router":
            continue
        for ng, mrep in ((1, 8 if sl == 1 else 1), (2, 16), (4, 32)):
            if (sl == 1 and ng > 1) or (layer == "unpooler" and sl == 0 and ng == 1):
                continue
            cands = [(res.get((layer, mrep, a)), a) for a in ("ldg4", "ldg6", "ring") if okng.get((layer, sl, a, ng), False)]
            cands = [c for c in cands if c[0] is not None]
            arm = min(cands)[1] if cands else ("ldg4" if okng.get((layer, sl, "ldg4", ng), False) else None)
            if arm is not None:
                routes[(layer, sl, ng)] = (g, arm)
    if r32ok:
        routes[("router", 0, 1)] = ("r32", None)
    gR = G.get(("router", 0))
    for ng in (2, 4):
        if gR is not None and okng.get(("router", 0, "ldg4", ng), False):
            routes[("router", 0, ng)] = (gR, "ldg4")
    state["routes"] = routes
    print("PS routes " + ", ".join(f"{k[0]}/{k[1]}/ng{k[2]}:{v[1] or 'r32'}" for k, v in sorted(routes.items())),
          flush=True)


def _kt1_lab(state, st, G, packs, ok_arm, r32ok, t0):
    mod = st["mod"]
    L = {"flush": torch.ones(FLUSH_FLOATS, dtype=torch.float32, device="cuda"),
         "e0": torch.cuda.Event(enable_timing=True), "e1": torch.cuda.Event(enable_timing=True)}
    floor, _ = _time(L, lambda: mod.ps_empty())
    print(f"KT1 floor (empty-kernel graph) {floor:.2f} us", flush=True)
    res = {}

    def T(key, fn):
        if time.time() - t0 > LAB_DEADLINE_S or int(st["err"][0]):
            return None
        md, mn = _time(L, fn)
        res[key] = md - floor
        return md - floor

    xs = {}
    for m in (1, 8, 16, 32):
        for k in (8192, 1024, 4096):
            xs[(m, k)] = _input(m, k, 77 + m + k)
    A4 = ("ldg4", "ldg6", "ring", "gx", "probe")
    lab = [("qkv_gate", 0, (1, 32), A4), ("pooler", 0, (1, 32), A4), ("o", 0, (1, 32), A4), ("unpooler", 1, (8,), A4),
           ("qkv_gate", 0, (8, 16), ("ldg4", "gx")), ("pooler", 0, (8, 16), ("ldg4", "gx")),
           ("unpooler", 0, (16, 32), ("ldg4",)), ("o", 0, (8, 16), ("ldg4",))]
    for layer, sl, ms, arms in lab:
        g = G[(layer, sl)]
        if g is None:
            continue
        for m in ms:
            x = xs[(m, g["P"]["K"])]
            out = torch.empty((m, g["n_out"]), dtype=torch.bfloat16, device="cuda")
            T((layer, m, "cublas"), lambda x=x, n=g["n_out"], layer=layer: _bf16_gemm_dispatch_impl(x, state[layer][:n], None))
            for arm in arms:
                if (_ng(m),) + ARMS[arm] not in set(INSTS):
                    continue
                if arm != "probe" and not ok_arm.get((layer, sl, arm), False):
                    continue
                T((layer, m, arm), lambda x=x, g=g, arm=arm, out=out: _ps(st, g, x, arm, out))
    # per-SM bandwidth-cap probe (parent, fresh_review_v37): the same pooler pack on 64 CTAs (8 clusters) vs 128
    g64 = _geo(packs["pooler"], 4096, nclu=8)
    if g64 is not None:
        x = xs[(1, 8192)]
        o64 = torch.empty((1, 4096), dtype=torch.bfloat16, device="cuda")
        T(("pooler", 1, "probe64"), lambda x=x, o64=o64: _ps(st, g64, x, "probe", o64))
        if ok_arm.get(("pooler", 0, "ldg4"), False):
            T(("pooler", 1, "ldg64"), lambda x=x, o64=o64: _ps(st, g64, x, "ldg4", o64))
    gR = G[("router", 0)]
    for m in (1, 8, 16, 32):
        x = xs[(m, 8192)]
        T(("router", m, "sglang"), lambda x=x: _baseline(state, x, "router", 256))
        if m <= 8 and r32ok:
            o32 = torch.empty((m, 256), dtype=torch.float32, device="cuda")
            T(("router", m, "r32"), lambda x=x, o32=o32: _r32(st, x, o32))
        if m > 8 and gR is not None and ok_arm.get(("router", 0, "ldg4"), False):
            o32 = torch.empty((m, 256), dtype=torch.float32, device="cuda")
            T(("router", m, "ldg4"), lambda x=x, o32=o32: _ps(st, gR, x, "ldg4", o32))
    print(f"KT1 err flag {int(st['err'][0])} lab done {time.time() - t0:.1f}s", flush=True)

    def fmt(key):
        v = res.get(key)
        return f"{v:6.2f}" if v is not None else "   -  "
    for layer, sl, ms, arms in lab[:4] + lab[6:7]:
        for m in ms:
            print(f"KT1 {layer:9s} T={m:<2d} cublas {fmt((layer, m, 'cublas'))} | " +
                  " ".join(f"{a} {fmt((layer, m, a))}" for a in A4), flush=True)
    for layer in ("qkv_gate", "pooler", "o"):
        print(f"KT1 sweep {layer:9s} ldg4 " + " ".join(f"T{m}:{fmt((layer, m, 'ldg4'))}" for m in (1, 8, 16, 32)) +
              " | gx " + " ".join(f"T{m}:{fmt((layer, m, 'gx'))}" for m in (1, 8, 16, 32)), flush=True)
    print(f"KT1 per-SM cap: pooler T=1 probe 128 CTAs {fmt(('pooler', 1, 'probe'))} vs 64 CTAs {fmt(('pooler', 1, 'probe64'))}"
          f" | ldg4 {fmt(('pooler', 1, 'ldg4'))} vs {fmt(('pooler', 1, 'ldg64'))}", flush=True)
    print("KT1 router " + " ".join(f"T{m}: sglang {fmt(('router', m, 'sglang'))} ps {fmt(('router', m, 'r32' if m <= 8 else 'ldg4'))}"
                                   for m in (1, 8, 16, 32)), flush=True)
    return res


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    if m <= 32 and x.is_contiguous() and x.data_ptr() % 16 == 0:
        sl = 1 if (layer == "unpooler" and m <= 8) else 0
        r = state["routes"].get((layer, sl, _ng(m)))
        if r is not None:
            g, arm = r
            if g == "r32":
                return _r32(state["ps"], x)
            return _ps(state["ps"], g, x, arm)
    return _baseline(state, x, layer, n)

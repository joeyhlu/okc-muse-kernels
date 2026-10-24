# LAB FILE — check runs only, never submit for score
# ===== BEGIN microbench_cuda.py (inlined verbatim into kernels/lab/r0b_*.py) =====
# H200 microbenchmarks M1-M5 (notes/study/03-hopper-gemm-attn.md §A.8-A.9) in ONE small
# torch.utils.cpp_extension.load_inline extension. Public entry point: run_all(print_fn, parts, deadline).
#   dev : device properties (torch + cudaDeviceGetAttribute by raw enum value)
#   M1  : idle pointer-chase latency (1 thread, ld.global.cg, random cyclic permutation of 128 B slots)
#   M2  : the same chase while the other blocks stream-read a different 1 GiB region
#   M3  : streaming read bandwidth vs bytes in flight per SM (vector width x unroll x blocks/SM)
#   M4  : random-row gather bandwidth, 64 B vs 128 B rows (page_size-1 KV), index loads included
#   M5  : E2M1 -> BF16 conversion throughput (Marlin bit placement, PRMT LUT, smem LUT x2) + INT peaks
# Every kernel has fixed trip counts (no data-dependent loops except the bounded M2 flag), every
# index is in range by construction or checked. Nothing here touches the problem's harness.
import time as _mb_time

_MB_CUDA = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define MB_MAGIC 0x9E3779B9u

static inline cudaStream_t mb_cur_stream() { return at::cuda::getCurrentCUDAStream().stream(); }

__device__ __forceinline__ unsigned long long mb_gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---------------- M1: idle pointer chase (one thread) ----------------
__global__ void mb_chase_kernel(const unsigned* __restrict__ buf, unsigned start, int warm, int hops,
                                long long* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  unsigned p = start;
  for (int i = 0; i < warm; ++i) p = __ldcg(buf + p);
  const long long c0 = clock64();
  const unsigned long long t0 = mb_gtimer();
  for (int i = 0; i < hops; ++i) p = __ldcg(buf + p);
  const long long c1 = clock64();
  const unsigned long long t1 = mb_gtimer();
  out[0] = c1 - c0;
  out[1] = (long long)(t1 - t0);
  out[2] = (long long)p;
}

// ---------------- M2: chase in block 0 while blocks 1.. stream ----------------
template <int U>
__global__ void __launch_bounds__(256) mb_chase_load_kernel(
    const unsigned* __restrict__ cbuf, unsigned start, int hops,
    const uint4* __restrict__ sbuf, long long n16, int max_passes,
    int* __restrict__ flag, long long* __restrict__ out,
    unsigned long long* __restrict__ bytes, uint4* __restrict__ sink) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      unsigned p = start;
      for (int i = 0; i < 64; ++i) p = __ldcg(cbuf + p);
      const long long c0 = clock64();
      const unsigned long long t0 = mb_gtimer();
      for (int i = 0; i < hops; ++i) p = __ldcg(cbuf + p);
      const long long c1 = clock64();
      const unsigned long long t1 = mb_gtimer();
      out[0] = c1 - c0;
      out[1] = (long long)(t1 - t0);
      out[2] = (long long)p;
      __threadfence();
      atomicExch(flag, 1);
    }
    return;
  }
  const long long nthreads = (long long)(gridDim.x - 1) * blockDim.x;
  const long long tid = (long long)(blockIdx.x - 1) * blockDim.x + threadIdx.x;
  uint4 acc = make_uint4(0u, 0u, 0u, 0u);
  unsigned long long cnt = 0ull;
  for (int pass = 0; pass < max_passes; ++pass) {
    if (*((volatile int*)flag) != 0) break;
    for (long long i = tid; i + (long long)(U - 1) * nthreads < n16; i += (long long)U * nthreads) {
      uint4 v[U];
#pragma unroll
      for (int u = 0; u < U; ++u) v[u] = __ldcs(sbuf + i + (long long)u * nthreads);
#pragma unroll
      for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
      cnt += (unsigned long long)U;
    }
  }
  atomicAdd(bytes, cnt * 16ull);
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == MB_MAGIC) sink[0] = acc;
}

// ---------------- M3: streaming read, V-wide loads, U in flight per thread ----------------
__device__ __forceinline__ void mb_zero(unsigned& a) { a = 0u; }
__device__ __forceinline__ void mb_zero(uint2& a) { a = make_uint2(0u, 0u); }
__device__ __forceinline__ void mb_zero(uint4& a) { a = make_uint4(0u, 0u, 0u, 0u); }
__device__ __forceinline__ void mb_xor(unsigned& a, unsigned b) { a ^= b; }
__device__ __forceinline__ void mb_xor(uint2& a, uint2 b) { a.x ^= b.x; a.y ^= b.y; }
__device__ __forceinline__ void mb_xor(uint4& a, uint4 b) { a.x ^= b.x; a.y ^= b.y; a.z ^= b.z; a.w ^= b.w; }
__device__ __forceinline__ unsigned mb_fold(unsigned a) { return a; }
__device__ __forceinline__ unsigned mb_fold(uint2 a) { return a.x ^ a.y; }
__device__ __forceinline__ unsigned mb_fold(uint4 a) { return a.x ^ a.y ^ a.z ^ a.w; }

template <typename V, int U>
__global__ void __launch_bounds__(256) mb_stream_kernel(const V* __restrict__ src, long long n, V* __restrict__ sink) {
  const long long stride = (long long)gridDim.x * blockDim.x;
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  V acc;
  mb_zero(acc);
  for (; i + (long long)(U - 1) * stride < n; i += (long long)U * stride) {
    V v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = __ldg(src + i + (long long)u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) mb_xor(acc, v[u]);
  }
  for (; i < n; i += stride) mb_xor(acc, __ldg(src + i));
  if (mb_fold(acc) == MB_MAGIC) sink[0] = acc;
}

// ---------------- M4: random row gather (16 B per thread, RB/16 threads per row) ----------------
template <int RB, int U>
__global__ void __launch_bounds__(256) mb_gather_kernel(const uint4* __restrict__ table, long long nrows,
                                                        const int* __restrict__ idx, long long nidx,
                                                        uint4* __restrict__ sink) {
  constexpr int TPR = RB / 16;
  const long long gtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int sub = (int)(gtid % TPR);
  const long long grp = gtid / TPR;
  const long long ngrp = ((long long)gridDim.x * blockDim.x) / TPR;
  uint4 acc = make_uint4(0u, 0u, 0u, 0u);
  for (long long r = grp; r < nidx; r += ngrp * U) {
    int rows[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long rr = r + (long long)u * ngrp;
      rows[u] = (rr < nidx) ? __ldg(idx + rr) : -1;
    }
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int row = rows[u];
      v[u] = (row >= 0 && (long long)row < nrows) ? __ldg(table + (long long)row * TPR + sub)
                                                 : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == MB_MAGIC) sink[0] = acc;
}

// ---------------- M5: E2M1 -> BF16 conversion variants ----------------
__device__ __forceinline__ unsigned mb_prmt(unsigned a, unsigned b, unsigned c) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// VAR 1: exactly sglang marlin/dequant.h:393-407 (dequant<nv_bfloat162, kFE2M1f, true>) applied the way
// marlin_moe/marlin_template.h:1262-1276 does (b_quant_1 = q, b_quant_0 = q << 8). Output BF16 = e2m1 * 2^-126.
// Slots: o0 = (n0, n4), o1 = (n1, n5), o2 = (n2, n6), o3 = (n3, n7); n_i = nibble i of q (bits 4i..4i+3).
__device__ __forceinline__ void mb_cvt_marlin(unsigned q, unsigned& o0, unsigned& o1, unsigned& o2, unsigned& o3) {
  unsigned x = q << 8;
  const unsigned a1 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x <<= 4;
  const unsigned a2 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x = q;
  const unsigned b1 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  x <<= 4;
  const unsigned b2 = (x & 0x80008000u) | ((x & 0x70007000u) >> 6);
  o0 = a2; o1 = a1; o2 = b2; o3 = b1;
}

// VAR 2: PRMT byte-LUT. Layout (a lossless offline repack): nibble i bits 0..2 = |v_i| code, nibble i bit 3 =
// sign of v_{s(i)} with s = {0,2,4,6,1,3,5,7}. Output o_j = (v_2j, v_2j+1), BF16 = e2m1 exactly (unscaled).
__device__ __forceinline__ void mb_cvt_prmt(unsigned q, unsigned& o0, unsigned& o1, unsigned& o2, unsigned& o3) {
  const unsigned HA = 0x3F3F3F00u, HB = 0x40404040u;  // high bytes of bf16(0, .5, 1, 1.5) | (2, 3, 4, 6)
  const unsigned LA = 0xC0800000u, LB = 0xC0804000u;  // low bytes
  const unsigned m = q & 0x77777777u;
  const unsigned mh = m >> 16;
  const unsigned hl = mb_prmt(HA, HB, m), ll = mb_prmt(LA, LB, m);
  const unsigned hh = mb_prmt(HA, HB, mh), lh = mb_prmt(LA, LB, mh);
  const unsigned p0 = mb_prmt(ll, hl, 0x5140u);
  const unsigned p1 = mb_prmt(ll, hl, 0x7362u);
  const unsigned p2 = mb_prmt(lh, hh, 0x5140u);
  const unsigned p3 = mb_prmt(lh, hh, 0x7362u);
  o0 = p0 | ((q << 12) & 0x80008000u);
  o1 = p1 | ((q << 8) & 0x80008000u);
  o2 = p2 | ((q << 4) & 0x80008000u);
  o3 = p3 | (q & 0x80008000u);
}

__device__ __forceinline__ unsigned mb_e2m1_bits(unsigned nib) {
  const unsigned mag = nib & 7u;
  const unsigned b = (mag == 0u) ? 0u : ((mag == 1u) ? 0x3F00u : (0x3F80u + (mag - 2u) * 0x40u));
  return b | ((nib & 8u) << 12);
}
__device__ __forceinline__ unsigned mb_lut_entry(unsigned byte) {
  return mb_e2m1_bits(byte & 15u) | (mb_e2m1_bits((byte >> 4) & 15u) << 16);
}

template <int VAR>
__device__ __forceinline__ void mb_fill_lut(unsigned* lut) {
  if constexpr (VAR == 3) {
    for (int e = threadIdx.x; e < 256; e += blockDim.x) lut[e] = mb_lut_entry((unsigned)e);
  }
  if constexpr (VAR == 4) {
    for (int e = threadIdx.x; e < 256 * 32; e += blockDim.x) lut[e] = mb_lut_entry((unsigned)(e >> 5));
  }
  __syncthreads();
}

// VAR 3: 256-entry smem LUT (one bf16x2 per byte, natural order o_k = (n_2k, n_2k+1)); random banks.
// VAR 4: the same LUT replicated per lane (entry*32 + lane): conflict-free, 32 KiB per block.
template <int VAR>
__device__ __forceinline__ void mb_cvt_any(unsigned q, const unsigned* lut, unsigned lane,
                                           unsigned& o0, unsigned& o1, unsigned& o2, unsigned& o3) {
  if constexpr (VAR == 1) {
    mb_cvt_marlin(q, o0, o1, o2, o3);
  } else if constexpr (VAR == 2) {
    mb_cvt_prmt(q, o0, o1, o2, o3);
  } else if constexpr (VAR == 3) {
    o0 = lut[q & 0xFFu];
    o1 = lut[(q >> 8) & 0xFFu];
    o2 = lut[(q >> 16) & 0xFFu];
    o3 = lut[q >> 24];
  } else if constexpr (VAR == 4) {
    o0 = lut[((q & 0xFFu) << 5) | lane];
    o1 = lut[(((q >> 8) & 0xFFu) << 5) | lane];
    o2 = lut[(((q >> 16) & 0xFFu) << 5) | lane];
    o3 = lut[((q >> 24) << 5) | lane];
  } else {
    const unsigned t = q & 0x3FFF3FFFu;  // VAR 0: harness overhead only
    o0 = t; o1 = t; o2 = t; o3 = t;
  }
}

__device__ __forceinline__ __nv_bfloat162 mb_u2bf(unsigned u) { return *reinterpret_cast<const __nv_bfloat162*>(&u); }
__device__ __forceinline__ unsigned mb_bf2u(__nv_bfloat162 b) { return *reinterpret_cast<const unsigned*>(&b); }

template <int VAR>
__global__ void __launch_bounds__(256) mb_cvt_bench_kernel(int iters, unsigned seed, uint4* __restrict__ sink,
                                                           long long* __restrict__ tinfo) {
  extern __shared__ unsigned mb_lut[];
  mb_fill_lut<VAR>(mb_lut);
  const unsigned lane = threadIdx.x & 31u;
  unsigned q = seed ^ ((blockIdx.x * blockDim.x + threadIdx.x) * 2654435761u);
  unsigned r = q * 2246822519u + 3266489917u;
  __nv_bfloat162 a0 = __float2bfloat162_rn(0.0f);
  __nv_bfloat162 a1 = a0, a2 = a0, a3 = a0;
  const __nv_bfloat162 kk = __float2bfloat162_rn(1.0f);
  long long c0 = 0;
  unsigned long long t0 = 0ull;
  if (threadIdx.x == 0) { c0 = clock64(); t0 = mb_gtimer(); }
#pragma unroll 2
  for (int it = 0; it < iters; ++it) {
    q = q * 1664525u + 1013904223u;
    r = r * 22695477u + 1u;
    unsigned o0, o1, o2, o3, p0, p1, p2, p3;
    mb_cvt_any<VAR>(q, mb_lut, lane, o0, o1, o2, o3);
    mb_cvt_any<VAR>(r, mb_lut, lane, p0, p1, p2, p3);
    a0 = __hfma2(mb_u2bf(o0), kk, a0);
    a1 = __hfma2(mb_u2bf(o1), kk, a1);
    a2 = __hfma2(mb_u2bf(o2), kk, a2);
    a3 = __hfma2(mb_u2bf(o3), kk, a3);
    a0 = __hfma2(mb_u2bf(p0), kk, a0);
    a1 = __hfma2(mb_u2bf(p1), kk, a1);
    a2 = __hfma2(mb_u2bf(p2), kk, a2);
    a3 = __hfma2(mb_u2bf(p3), kk, a3);
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    tinfo[0] = clock64() - c0;
    tinfo[1] = (long long)(mb_gtimer() - t0);
  }
  const unsigned f = mb_bf2u(a0) ^ mb_bf2u(a1) ^ mb_bf2u(a2) ^ mb_bf2u(a3);
  if (f == MB_MAGIC) sink[0] = make_uint4(f, 0u, 0u, 0u);
}

template <int VAR>
__global__ void __launch_bounds__(256) mb_cvt_check_kernel(const unsigned* __restrict__ words, int n,
                                                           uint4* __restrict__ out) {
  extern __shared__ unsigned mb_lut[];
  mb_fill_lut<VAR>(mb_lut);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned o0, o1, o2, o3;
  mb_cvt_any<VAR>(words[i], mb_lut, threadIdx.x & 31u, o0, o1, o2, o3);
  out[i] = make_uint4(o0, o1, o2, o3);
}

// INT pipe peaks: 8 independent chains of LOP3 (KIND 0) or PRMT (KIND 1), 32 ops per thread per iteration.
template <int KIND>
__global__ void __launch_bounds__(256) mb_intpeak_kernel(int iters, unsigned seed, unsigned* __restrict__ sink) {
  unsigned a0 = seed ^ threadIdx.x, a1 = a0 * 3u, a2 = a0 * 5u, a3 = a0 * 7u;
  unsigned a4 = a0 * 11u, a5 = a0 * 13u, a6 = a0 * 17u, a7 = a0 * 19u;
  const unsigned b = seed * 2654435761u, c = (seed >> 3) | 0x5140u;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if constexpr (KIND == 0) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a0) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a1) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a2) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a3) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a4) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a5) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a6) : "r"(b), "r"(c));
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(a7) : "r"(b), "r"(c));
      } else {
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a0) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a1) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a2) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a3) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a4) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a5) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a6) : "r"(b), "r"(c));
        asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(a7) : "r"(b), "r"(c));
      }
    }
  }
  const unsigned f = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
  if (f == MB_MAGIC) sink[0] = f;
}

// ---------------- host wrappers (return 0 on success, cudaError code otherwise, -2 bad config) ----------------
std::vector<int64_t> mb_dev_attrs() {
  int dev = 0;
  cudaGetDevice(&dev);
  // raw cudaDeviceAttr values (stable ABI, avoids enum names removed in newer toolkits):
  // 13 clockRate kHz, 16 SMs, 36 memClockRate kHz, 37 bus width bits, 38 L2 bytes, 39 max threads/SM,
  // 81 max smem/SM, 82 max regs/SM, 8 smem/block, 97 smem/block opt-in
  const int ids[10] = {13, 16, 36, 37, 38, 39, 81, 82, 8, 97};
  std::vector<int64_t> r;
  for (int k = 0; k < 10; ++k) {
    int v = -1;
    if (cudaDeviceGetAttribute(&v, (cudaDeviceAttr)ids[k], dev) != cudaSuccess) {
      v = -1;
      cudaGetLastError();
    }
    r.push_back((int64_t)v);
  }
  return r;
}

int64_t mb_chase(torch::Tensor buf, int64_t start, int64_t warm, int64_t hops, torch::Tensor out) {
  mb_chase_kernel<<<1, 32, 0, mb_cur_stream()>>>((const unsigned*)buf.data_ptr(), (unsigned)start, (int)warm,
                                                 (int)hops, (long long*)out.data_ptr());
  return (int64_t)cudaGetLastError();
}

template <int U>
static int64_t mb_chase_load_launch(const unsigned* cb, unsigned start, int hops, const uint4* sb, long long n16,
                                    int max_passes, int sblocks, int* fl, long long* o, unsigned long long* by,
                                    uint4* sk) {
  mb_chase_load_kernel<U><<<1 + sblocks, 256, 0, mb_cur_stream()>>>(cb, start, hops, sb, n16, max_passes, fl, o, by, sk);
  return (int64_t)cudaGetLastError();
}

int64_t mb_chase_load(torch::Tensor cbuf, int64_t start, int64_t hops, torch::Tensor sbuf, int64_t sblocks,
                      int64_t unroll, int64_t max_passes, torch::Tensor flag, torch::Tensor out,
                      torch::Tensor bytes, torch::Tensor sink) {
  const unsigned* cb = (const unsigned*)cbuf.data_ptr();
  const uint4* sb = (const uint4*)sbuf.data_ptr();
  const long long n16 = (long long)(sbuf.numel() * sbuf.element_size()) / 16;
  int* fl = (int*)flag.data_ptr();
  long long* o = (long long*)out.data_ptr();
  unsigned long long* by = (unsigned long long*)bytes.data_ptr();
  uint4* sk = (uint4*)sink.data_ptr();
  const unsigned st = (unsigned)start;
  const int h = (int)hops, mp = (int)max_passes, nb = (int)sblocks;
  switch (unroll) {
    case 1: return mb_chase_load_launch<1>(cb, st, h, sb, n16, mp, nb, fl, o, by, sk);
    case 2: return mb_chase_load_launch<2>(cb, st, h, sb, n16, mp, nb, fl, o, by, sk);
    case 4: return mb_chase_load_launch<4>(cb, st, h, sb, n16, mp, nb, fl, o, by, sk);
    case 8: return mb_chase_load_launch<8>(cb, st, h, sb, n16, mp, nb, fl, o, by, sk);
    default: return -2;
  }
}

template <typename V, int U>
static int64_t mb_stream_launch(const void* src, long long nbytes, int blocks, void* sink, bool occ_only) {
  if (occ_only) {
    int nb = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mb_stream_kernel<V, U>, 256, 0) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    return (int64_t)nb;
  }
  const long long n = nbytes / (long long)sizeof(V);
  mb_stream_kernel<V, U><<<blocks, 256, 0, mb_cur_stream()>>>((const V*)src, n, (V*)sink);
  return (int64_t)cudaGetLastError();
}

int64_t mb_stream_bw(torch::Tensor buf, int64_t vec_bytes, int64_t unroll, int64_t blocks, torch::Tensor sink,
                     int64_t occ_only) {
  const void* src = buf.data_ptr();
  const long long nbytes = (long long)(buf.numel() * buf.element_size());
  void* sk = sink.data_ptr();
  const int b = (int)blocks;
  const bool oo = occ_only != 0;
#define MB_SCASE(VB, VT, UU) \
  if (vec_bytes == VB && unroll == UU) return mb_stream_launch<VT, UU>(src, nbytes, b, sk, oo);
  MB_SCASE(4, unsigned, 1) MB_SCASE(4, unsigned, 2) MB_SCASE(4, unsigned, 4) MB_SCASE(4, unsigned, 8)
  MB_SCASE(8, uint2, 1) MB_SCASE(8, uint2, 2) MB_SCASE(8, uint2, 4) MB_SCASE(8, uint2, 8)
  MB_SCASE(16, uint4, 1) MB_SCASE(16, uint4, 2) MB_SCASE(16, uint4, 4) MB_SCASE(16, uint4, 8)
#undef MB_SCASE
  return -2;
}

template <int RB, int U>
static int64_t mb_gather_launch(const uint4* t, long long nrows, const int* ix, long long nidx, int blocks, uint4* sk) {
  mb_gather_kernel<RB, U><<<blocks, 256, 0, mb_cur_stream()>>>(t, nrows, ix, nidx, sk);
  return (int64_t)cudaGetLastError();
}

int64_t mb_gather(torch::Tensor table, torch::Tensor idx, int64_t row_bytes, int64_t unroll, int64_t blocks,
                  torch::Tensor sink) {
  if (row_bytes != 64 && row_bytes != 128) return -2;
  const uint4* t = (const uint4*)table.data_ptr();
  const long long nrows = (long long)(table.numel() * table.element_size()) / (long long)row_bytes;
  const int* ix = (const int*)idx.data_ptr();
  const long long nidx = (long long)idx.numel();
  uint4* sk = (uint4*)sink.data_ptr();
  const int b = (int)blocks;
#define MB_GCASE(RB, UU) \
  if (row_bytes == RB && unroll == UU) return mb_gather_launch<RB, UU>(t, nrows, ix, nidx, b, sk);
  MB_GCASE(64, 1) MB_GCASE(64, 2) MB_GCASE(64, 4) MB_GCASE(64, 8)
  MB_GCASE(128, 1) MB_GCASE(128, 2) MB_GCASE(128, 4) MB_GCASE(128, 8)
#undef MB_GCASE
  return -2;
}

static size_t mb_cvt_smem(int var) { return var == 3 ? (size_t)(256 * 4) : (var == 4 ? (size_t)(256 * 32 * 4) : (size_t)0); }

template <int VAR>
static int64_t mb_cvt_launch(int blocks, int iters, unsigned seed, uint4* sk, long long* ti, bool occ_only) {
  const size_t sm = mb_cvt_smem(VAR);
  if (occ_only) {
    int nb = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mb_cvt_bench_kernel<VAR>, 256, sm) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    return (int64_t)nb;
  }
  mb_cvt_bench_kernel<VAR><<<blocks, 256, sm, mb_cur_stream()>>>(iters, seed, sk, ti);
  return (int64_t)cudaGetLastError();
}

int64_t mb_cvt(int64_t var, int64_t blocks, int64_t iters, int64_t seed, torch::Tensor sink, torch::Tensor tinfo,
               int64_t occ_only) {
  uint4* sk = (uint4*)sink.data_ptr();
  long long* ti = (long long*)tinfo.data_ptr();
  const int b = (int)blocks, it = (int)iters;
  const unsigned sd = (unsigned)seed;
  const bool oo = occ_only != 0;
  switch (var) {
    case 0: return mb_cvt_launch<0>(b, it, sd, sk, ti, oo);
    case 1: return mb_cvt_launch<1>(b, it, sd, sk, ti, oo);
    case 2: return mb_cvt_launch<2>(b, it, sd, sk, ti, oo);
    case 3: return mb_cvt_launch<3>(b, it, sd, sk, ti, oo);
    case 4: return mb_cvt_launch<4>(b, it, sd, sk, ti, oo);
    default: return -2;
  }
}

template <int VAR>
static int64_t mb_cvt_check_launch(const unsigned* w, int n, uint4* o) {
  const int blocks = (n + 255) / 256;
  mb_cvt_check_kernel<VAR><<<blocks, 256, mb_cvt_smem(VAR), mb_cur_stream()>>>(w, n, o);
  return (int64_t)cudaGetLastError();
}

int64_t mb_cvt_check(int64_t var, torch::Tensor words, torch::Tensor out) {
  const unsigned* w = (const unsigned*)words.data_ptr();
  const int n = (int)words.numel();
  uint4* o = (uint4*)out.data_ptr();
  if (n <= 0) return -2;
  switch (var) {
    case 0: return mb_cvt_check_launch<0>(w, n, o);
    case 1: return mb_cvt_check_launch<1>(w, n, o);
    case 2: return mb_cvt_check_launch<2>(w, n, o);
    case 3: return mb_cvt_check_launch<3>(w, n, o);
    case 4: return mb_cvt_check_launch<4>(w, n, o);
    default: return -2;
  }
}

template <int KIND>
static int64_t mb_intpeak_launch(int blocks, int iters, unsigned seed, unsigned* sk, bool occ_only) {
  if (occ_only) {
    int nb = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mb_intpeak_kernel<KIND>, 256, 0) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    return (int64_t)nb;
  }
  mb_intpeak_kernel<KIND><<<blocks, 256, 0, mb_cur_stream()>>>(iters, seed, sk);
  return (int64_t)cudaGetLastError();
}

int64_t mb_intpeak(int64_t kind, int64_t blocks, int64_t iters, int64_t seed, torch::Tensor sink, int64_t occ_only) {
  unsigned* sk = (unsigned*)sink.data_ptr();
  const bool oo = occ_only != 0;
  if (kind == 0) return mb_intpeak_launch<0>((int)blocks, (int)iters, (unsigned)seed, sk, oo);
  if (kind == 1) return mb_intpeak_launch<1>((int)blocks, (int)iters, (unsigned)seed, sk, oo);
  return -2;
}
"""

_MB_CPP = r"""
#include <vector>
std::vector<int64_t> mb_dev_attrs();
int64_t mb_chase(torch::Tensor buf, int64_t start, int64_t warm, int64_t hops, torch::Tensor out);
int64_t mb_chase_load(torch::Tensor cbuf, int64_t start, int64_t hops, torch::Tensor sbuf, int64_t sblocks,
                      int64_t unroll, int64_t max_passes, torch::Tensor flag, torch::Tensor out,
                      torch::Tensor bytes, torch::Tensor sink);
int64_t mb_stream_bw(torch::Tensor buf, int64_t vec_bytes, int64_t unroll, int64_t blocks, torch::Tensor sink,
                     int64_t occ_only);
int64_t mb_gather(torch::Tensor table, torch::Tensor idx, int64_t row_bytes, int64_t unroll, int64_t blocks,
                  torch::Tensor sink);
int64_t mb_cvt(int64_t var, int64_t blocks, int64_t iters, int64_t seed, torch::Tensor sink, torch::Tensor tinfo,
               int64_t occ_only);
int64_t mb_cvt_check(int64_t var, torch::Tensor words, torch::Tensor out);
int64_t mb_intpeak(int64_t kind, int64_t blocks, int64_t iters, int64_t seed, torch::Tensor sink, int64_t occ_only);
"""

_MB_FUNCS = ["mb_dev_attrs", "mb_chase", "mb_chase_load", "mb_stream_bw", "mb_gather", "mb_cvt", "mb_cvt_check",
             "mb_intpeak"]
_MB_STATE = {"mod": None}
_MB_GIB = 1 << 30
_MB_MIB = 1 << 20


def _mb_build(P):
    """Compile the extension once. No arch flags of our own: torch adds -gencode for the visible device
    (sm_90) when TORCH_CUDA_ARCH_LIST is unset; the arch flags it will use are printed first."""
    if _MB_STATE["mod"] is not None:
        return _MB_STATE["mod"]
    from torch.utils import cpp_extension as _ce
    try:
        P("MB arch flags: %s" % " ".join(_ce._get_cuda_arch_flags()))
    except Exception as e:
        P("MB arch flags n/a (%s)" % type(e).__name__)
    t = _mb_time.time()
    mod = _ce.load_inline(name="okc_lab_mb_v1", cpp_sources=[_MB_CPP], cuda_sources=[_MB_CUDA],
                          functions=_MB_FUNCS, extra_cflags=["-O2"], extra_cuda_cflags=["-O3"], verbose=False)
    P("MB compile %.1f s" % (_mb_time.time() - t))
    _MB_STATE["mod"] = mod
    return mod


def _mb_ok(rc, what):
    if rc != 0:
        raise RuntimeError("%s returned %d" % (what, rc))


def _mb_err(e, n=1500):
    s = "%s: %s" % (type(e).__name__, e)
    return s if len(s) <= n else s[:300] + " ... " + s[-(n - 300):]


def _mb_dev(P, torch, info):
    dev = torch.cuda.current_device()
    pr = torch.cuda.get_device_properties(dev)
    sms = int(pr.multi_processor_count)
    info["sms"] = sms
    extra = []
    for k in ("L2_cache_size", "clock_rate", "memory_clock_rate", "memory_bus_width", "regs_per_multiprocessor",
              "max_threads_per_multi_processor", "shared_memory_per_multiprocessor"):
        v = getattr(pr, k, None)
        if v is not None:
            extra.append("%s=%s" % (k, v))
    P("DEV %s cc%d.%d SMs=%d mem=%.1fGiB torch=%s cuda=%s %s" % (pr.name, pr.major, pr.minor, sms,
                                                                pr.total_memory / _MB_GIB, torch.__version__,
                                                                torch.version.cuda, " ".join(extra)))
    try:
        P("DEV nvml clock_rate MHz=%s" % torch.cuda.clock_rate(dev))
    except Exception as e:  # pynvml may be absent
        P("DEV nvml n/a (%s)" % type(e).__name__)
    return dev


def _mb_dev_attrs(P, mod):
    a = mod.mb_dev_attrs()
    names = ["clk_kHz", "SMs", "memclk_kHz", "bus_bits", "L2_B", "thr/SM", "smem/SM", "regs/SM", "smem/blk",
             "smem/blk_optin"]
    P("DEV attrs " + " ".join("%s=%d" % (n, v) for n, v in zip(names, a)))
    if a[2] > 0 and a[3] > 0:
        P("DEV nominal HBM BW = %.0f GB/s (2 x memclk x bus/8)" % (2.0 * a[2] * 1e3 * a[3] / 8 / 1e9))
    return a


def _mb_make_chase(torch, buf, ws_bytes, slot_bytes=128):
    """Random single-cycle permutation over the 128 B slots of buf[:ws_bytes]; every element of the
    region holds a valid element index (zeros elsewhere), so the chase can never leave the region."""
    se = slot_bytes // 4
    S = ws_bytes // slot_bytes
    region = buf[: S * se]
    region.zero_()
    perm = torch.randperm(S, device=buf.device)
    nxt = torch.empty_like(perm)
    nxt[perm] = torch.roll(perm, -1)
    region.view(S, se)[:, 0].copy_((nxt * se).to(torch.int32))
    start = int(perm[0].item()) * se
    del perm, nxt
    return start, S


def _mb_m1(P, torch, mod, buf, info):
    out = torch.zeros(4, dtype=torch.int64, device=buf.device)
    total = buf.numel() * 4
    rows = []
    for ws in (4 * _MB_MIB, 24 * _MB_MIB, 256 * _MB_MIB, _MB_GIB, 2 * _MB_GIB):
        if ws > total:
            continue
        start, S = _mb_make_chase(torch, buf, ws)
        hops = 20000
        warm = min(S, 300000) if ws <= 64 * _MB_MIB else 2000
        _mb_ok(mod.mb_chase(buf, start, warm, hops, out), "mb_chase")
        torch.cuda.synchronize()
        cyc, ns = int(out[0].item()), int(out[1].item())
        lat_ns, lat_cyc = ns / hops, cyc / hops
        ghz = cyc / max(ns, 1)
        rows.append((ws, lat_ns, lat_cyc, ghz))
        P("M1 ws=%6.0fMiB lat=%7.1f ns %7.1f cyc  (SM clk %.3f GHz, %d hops)" % (ws / _MB_MIB, lat_ns, lat_cyc, ghz,
                                                                               hops))
    info["m1"] = rows
    dram = [r for r in rows if r[0] >= 256 * _MB_MIB]
    if dram:
        info["lat_ns"] = min(r[1] for r in dram)
        info["clk_mem_ghz"] = dram[0][3]
    return rows


def _mb_m2(P, torch, mod, buf, info):
    sms = info["sms"]
    half = buf.numel() // 2
    cbuf, sbuf = buf[:half], buf[half:]
    start, _ = _mb_make_chase(torch, cbuf, half * 4)
    out = torch.zeros(4, dtype=torch.int64, device=buf.device)
    flag = torch.zeros(1, dtype=torch.int32, device=buf.device)
    nbytes = torch.zeros(1, dtype=torch.int64, device=buf.device)
    sink = torch.zeros(64, dtype=torch.int32, device=buf.device)
    hops = 8192
    res = []
    for bps, u in ((0, 1), (1, 1), (1, 2), (1, 4), (2, 4), (4, 4)):
        flag.zero_()
        nbytes.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        _mb_ok(mod.mb_chase_load(cbuf, start, hops, sbuf, bps * sms, u, 400, flag, out, nbytes, sink), "mb_chase_load")
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        cyc, ns = int(out[0].item()), int(out[1].item())
        gbs = int(nbytes.item()) / (ms * 1e-3) / 1e9 if bps else 0.0
        infl = bps * 256 * 16 * u
        lat = ns / hops
        res.append((infl, gbs, lat))
        need = gbs * 1e9 * lat * 1e-9 / sms / 1024 if bps else 0.0
        P("M2 stream %2dKiB/SM in flight: %6.0f GB/s  chase lat=%7.1f ns %6.1f cyc  -> BWxlat/SM=%5.1f KiB" % (
            infl // 1024, gbs, lat, cyc / hops, need))
    info["m2"] = res
    return res


def _mb_m3(P, torch, mod, buf, info):
    sms = info["sms"]
    sbuf = buf[: _MB_GIB // 4]
    nbytes = sbuf.numel() * 4
    sink = torch.zeros(64, dtype=torch.int32, device=buf.device)
    res = []
    for vb in (4, 8, 16):
        for u in (1, 2, 4, 8):
            occ = mod.mb_stream_bw(sbuf, vb, u, 0, sink, 1)
            cells = []
            for bps in (1, 2, 4, 8):
                eff = min(bps, occ) if occ > 0 else bps
                blocks = eff * sms
                _mb_ok(mod.mb_stream_bw(sbuf, vb, u, blocks, sink, 0), "mb_stream_bw warm")
                best = 1e30
                for _ in range(3):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    _mb_ok(mod.mb_stream_bw(sbuf, vb, u, blocks, sink, 0), "mb_stream_bw")
                    e1.record()
                    torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1))
                gbs = nbytes / (best * 1e-3) / 1e9
                infl = eff * 256 * vb * u
                res.append((vb, u, eff, infl, gbs))
                cells.append("b%d%s %5.0f(%3dK)" % (bps, "" if eff == bps else ">%d" % eff, gbs, infl // 1024))
            P("M3 v%-2d u%d occ%d | " % (vb, u, occ) + " | ".join(cells))
    mx = max(r[4] for r in res)
    info["bw_max"] = mx
    P("M3 max %.0f GB/s (%.0f GB/s/SM)" % (mx, mx / sms))
    info["m3_knee"] = {}
    for frac in (0.8, 0.9, 0.95):
        ok = [r for r in res if r[4] >= frac * mx]
        r = min(ok, key=lambda x: x[3])
        info["m3_knee"][frac] = (r[3] / 1024, r[3] * sms / r[4])
        P("M3 knee: >=%2.0f%% of max with %5.1f KiB/SM in flight (v%d u%d b%d) -> implied lat %.0f ns" % (
            frac * 100, r[3] / 1024, r[0], r[1], r[2], r[3] * sms / (r[4] * 1e9) * 1e9))
    lat = info.get("lat_ns")
    if lat:
        for tgt in (4.2e12, 0.9 * mx * 1e9):
            P("M3 Little: idle lat %.0f ns x %.2f TB/s / %d SMs = %.1f KiB/SM needed" % (
                lat, tgt / 1e12, sms, tgt * lat * 1e-9 / sms / 1024))
    return res


def _mb_m4(P, torch, mod, buf, info):
    sms = info["sms"]
    table = buf[: _MB_GIB // 4]
    sink = torch.zeros(64, dtype=torch.int32, device=buf.device)
    blocks = 4 * sms
    for rb in (64, 128):
        nrows = _MB_GIB // rb
        nidx = (256 * _MB_MIB) // rb
        idx = torch.randperm(nrows, device=buf.device)[:nidx].to(torch.int32)
        seq = torch.arange(nidx, device=buf.device, dtype=torch.int32)
        cells = []
        for name, ix, us in (("rand", idx, (1, 2, 4, 8)), ("seq", seq, (4,))):
            for u in us:
                _mb_ok(mod.mb_gather(table, ix, rb, u, blocks, sink), "mb_gather warm")
                best = 1e30
                for _ in range(3):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    _mb_ok(mod.mb_gather(table, ix, rb, u, blocks, sink), "mb_gather")
                    e1.record()
                    torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1))
                gbs = nidx * rb / (best * 1e-3) / 1e9
                m4 = info.setdefault("m4", {})
                m4[(rb, name)] = max(gbs, m4.get((rb, name), 0.0))
                cells.append("%s u%d %5.0f GB/s %4.0f Mrow/s" % (name, u, gbs, nidx / (best * 1e-3) / 1e6))
        P("M4 row%3dB (%d thr/row, 1024 thr/SM, +4 B idx/row) | " % (rb, rb // 16) + " | ".join(cells))
        del idx, seq
    if "bw_max" in info:
        P("M4 compare: M3 streaming max %.0f GB/s" % info["bw_max"])


def _mb_expected(torch, var, w64):
    """Expected BF16 bit patterns (int64 [n, 4]) for variant var, from words w64 (int64 in [0, 2^32))."""
    dev = w64.device
    t8 = torch.tensor([0, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0], dtype=torch.int64, device=dev)
    nib = [(w64 >> (4 * i)) & 15 for i in range(8)]

    def bits(n):
        return t8[n & 7] | ((n & 8) << 12)

    if var == 1:
        def mb(n):
            return ((n & 8) << 12) | ((n & 7) << 6)
        cols = [mb(nib[a]) | (mb(nib[b]) << 16) for a, b in ((0, 4), (1, 5), (2, 6), (3, 7))]
    elif var == 2:
        slot = (0, 4, 1, 5, 2, 6, 3, 7)  # sign of v_i lives in nibble slot[i]

        def val(i):
            return bits((nib[i] & 7) | (nib[slot[i]] & 8))
        cols = [val(2 * j) | (val(2 * j + 1) << 16) for j in range(4)]
    else:
        cols = [bits(nib[2 * j]) | (bits(nib[2 * j + 1]) << 16) for j in range(4)]
    return torch.stack(cols, 1)


def _mb_m5(P, torch, mod, info):
    sms = info["sms"]
    dev = torch.cuda.current_device()
    sink = torch.zeros(64, dtype=torch.int32, device=dev)
    tinfo = torch.zeros(4, dtype=torch.int64, device=dev)
    # correctness of each conversion on random words + every byte pattern
    n = 1 << 14
    w64 = torch.randint(0, 1 << 32, (n,), dtype=torch.int64, device=dev)
    w64[:256] = torch.arange(256, device=dev) * 0x01010101
    w32 = torch.where(w64 >= (1 << 31), w64 - (1 << 32), w64).to(torch.int32)
    names = {0: "overhead", 1: "marlin-lop3", 2: "prmt-lut", 3: "smem-lut256", 4: "smem-lut-lane"}
    status = {}
    for var in (1, 2, 3, 4):
        out = torch.zeros((n, 4), dtype=torch.int32, device=dev)
        _mb_ok(mod.mb_cvt_check(var, w32, out), "mb_cvt_check")
        torch.cuda.synchronize()
        got = out.to(torch.int64) & 0xFFFFFFFF
        bad = int((got != _mb_expected(torch, var, w64)).any(1).sum().item())
        status[var] = "ok" if bad == 0 else "MISMATCH %d/%d" % (bad, n)
    iters = 4096
    for var in (0, 1, 2, 3, 4):
        occ = mod.mb_cvt(var, 0, 0, 0, sink, tinfo, 1)
        blocks = max(occ, 1) * sms
        _mb_ok(mod.mb_cvt(var, blocks, 64, 1234, sink, tinfo, 0), "mb_cvt warm")
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        _mb_ok(mod.mb_cvt(var, blocks, iters, 99991, sink, tinfo, 0), "mb_cvt")
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        ghz = int(tinfo[0].item()) / max(int(tinfo[1].item()), 1)
        info["clk_alu_ghz"] = ghz
        vals = blocks * 256.0 * iters * 16
        vpc = vals / (ms * 1e6 * ghz) / sms
        info.setdefault("m5", {})[var] = vpc
        info.setdefault("m5_check", {})[var] = status.get(var, "-")
        P("M5 %-13s %6.1f values/clk/SM  (occ %d blk/SM, %.2f ms, clk %.3f GHz) check=%s" % (
            names[var], vpc, occ, ms, ghz, status.get(var, "-")))
    for kind, nm in ((0, "LOP3"), (1, "PRMT")):
        occ = mod.mb_intpeak(kind, 0, 0, 0, sink, 1)
        blocks = max(occ, 1) * sms
        _mb_ok(mod.mb_intpeak(kind, blocks, 64, 7, sink, 0), "mb_intpeak warm")
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        _mb_ok(mod.mb_intpeak(kind, blocks, iters, 7, sink, 0), "mb_intpeak")
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        ghz = info.get("clk_alu_ghz", 1.755)
        ops = blocks * 256.0 * iters * 32
        info.setdefault("m5_peak", {})[nm] = ops / (ms * 1e6 * ghz) / sms
        P("M5 peak %s %6.1f ops/clk/SM (at %.3f GHz from the cvt kernels; %.2f ms)" % (nm, ops / (ms * 1e6 * ghz) / sms,
                                                                                       ghz, ms))
    ghz = info.get("clk_alu_ghz", 1.755)
    for bw in (4.2e12, 4.8e12):
        need = bw / 0.5625 / sms / (ghz * 1e9)
        P("M5 need at %.1f TB/s of NVFP4 (0.5625 B/value): %.1f values/clk/SM at %.3f GHz" % (bw / 1e12, need, ghz))


def mb_summary_lines(info):
    """A few compact lines with the decision-relevant microbench numbers (printed last by the lab files)."""
    out = []
    if info.get("m1"):
        out.append("SUM M1 lat ns/cyc: " + " ".join("%dMiB=%.0f/%.0f" % (r[0] // _MB_MIB, r[1], r[2])
                                                   for r in info["m1"]))
    if info.get("m2"):
        out.append("SUM M2 (KiB/SM in flight: GB/s, lat ns): " + " ".join("%d:%.0f,%.0f" % (r[0] // 1024, r[1], r[2])
                                                                          for r in info["m2"]))
    if info.get("bw_max"):
        k = info.get("m3_knee", {})
        out.append("SUM M3 max %.0f GB/s; KiB/SM for 80/90/95%%: %s; idle lat %s ns" % (
            info["bw_max"], "/".join("%.1f" % k[f][0] for f in (0.8, 0.9, 0.95) if f in k),
            "%.0f" % info["lat_ns"] if info.get("lat_ns") else "n/a"))
    if info.get("m4"):
        out.append("SUM M4 best GB/s: " + " ".join("%s%dB=%.0f" % (n, rb, v) for (rb, n), v in sorted(info["m4"].items())))
    if info.get("m5"):
        nm = {0: "ovh", 1: "marlin", 2: "prmt", 3: "lut256", 4: "lutlane"}
        out.append("SUM M5 values/clk/SM: " + " ".join("%s=%.1f(%s)" % (nm[v], x, info.get("m5_check", {}).get(v, "-"))
                                                      for v, x in sorted(info["m5"].items()))
                   + " | peaks " + " ".join("%s=%.1f" % kv for kv in sorted(info.get("m5_peak", {}).items()))
                   + " | clk %.3f GHz" % info.get("clk_alu_ghz", 0.0))
    return out


def run_all(print_fn=print, parts=("dev", "M1", "M2", "M3", "M4", "M5"), deadline=None):
    """Run the selected microbenchmarks; each part is guarded so one failure never stops the rest.
    deadline: absolute time.time() after which remaining parts are skipped."""
    import torch
    P = print_fn
    t0 = _mb_time.time()
    info = {}
    try:
        _mb_dev(P, torch, info)
    except Exception as e:
        P("DEV ERR " + _mb_err(e))
        info.setdefault("sms", 132)
    try:
        mod = _mb_build(P)
    except Exception as e:
        P("MB BUILD FAILED " + _mb_err(e, 6000))
        return info
    try:
        _mb_dev_attrs(P, mod)
    except Exception as e:
        P("DEV attrs ERR " + _mb_err(e))
    need_big = any(p in parts for p in ("M1", "M2", "M4"))
    buf = None
    try:
        buf = torch.empty((2 if need_big else 1) * _MB_GIB // 4, dtype=torch.int32, device="cuda")
        buf.zero_()
    except Exception as e:
        P("MB alloc ERR " + _mb_err(e))
    steps = (("M1", _mb_m1, True), ("M2", _mb_m2, True), ("M3", _mb_m3, True), ("M4", _mb_m4, True),
             ("M5", None, False))
    for name, fn, needs_buf in steps:
        if name not in parts:
            continue
        if deadline is not None and _mb_time.time() > deadline:
            P("%s skipped (deadline)" % name)
            continue
        if needs_buf and buf is None:
            P("%s skipped (no buffer)" % name)
            continue
        t = _mb_time.time()
        try:
            if name == "M5":
                _mb_m5(P, torch, mod, info)
            else:
                fn(P, torch, mod, buf, info)
        except Exception as e:
            P("%s ERR %s" % (name, _mb_err(e)))
        try:
            torch.cuda.synchronize()
        except Exception as e:
            P("%s sync ERR %s" % (name, _mb_err(e)))
        P("%s done in %.1f s" % (name, _mb_time.time() - t))
    del buf
    torch.cuda.empty_cache()
    P("MB total %.1f s" % (_mb_time.time() - t0))
    return info
# ===== END microbench_cuda.py =====

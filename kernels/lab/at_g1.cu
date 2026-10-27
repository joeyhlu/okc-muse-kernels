// at_g1: gather probes for the full-attention B16/B32 shapes (notes/study/attn_full_fresh.md s2).
// Appended to a K1 source copy (k1w module) inside kernels/muse_attention_decode/at_1.py; also compiles standalone
// (nvcc -arch=sm_90a -c -DAT_STANDALONE).
// gp<MAP, HINT>: gather-only, every cached key's K row and V row (128 B each) through the page table exactly once,
// into a 4-stage x 16 KB cp.async ring per CTA (K1's ring, 256 threads, 2 CTAs/SM, grid 2 x SMs), balanced flat
// partition of all cached keys (equal keys per CTA across request boundaries), no barriers (each thread reads back only
// the chunks it copied), checksum = sum of the uint32 words of every row (K1-style zero fill past the end).
//   MAP 0 = K1's v24 map: thread -> key tid>>2, chunks part and part+4 (one 64-B half row per instruction per key,
//           2 instructions per row, 8 rows per warp instruction).
//   MAP 1 = whole-row map: thread -> keys 16(tid>>6) + ((tid>>3)&7) + 8u, chunk tid&7 (8 lanes per row, 4 whole
//           rows per warp instruction; the warp pair of key quarter q4 loads exactly that quarter, as K1 consumes it).
//   HINT 1 = cp.async ... .L2::128B (full-line L2 fetch on the first miss).
//   PLAN 1 = K1's own split plan (k1::plan_C / plan_find: one request per CTA, splits <= C tiles; module build only)
//            instead of the balanced flat partition.
//   BAR 1  = K1's per-stage warp-pair named barrier (bar.sync 1 + q4, 64) before the next issue, and every thread
//            reads chunks the OTHER warp of its pair copied (rows x ^ 8 for MAP 0, x ^ 4 for MAP 1), as K1 consumes.
#ifndef AT_STANDALONE
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#endif
#include <stdint.h>

namespace atg {
constexpr int NT = 256, TILE = 64, ST = 4, ROWB = 128, STAGE = 2 * TILE * ROWB;
constexpr int MAXB = 128;
// dynamic smem: the 64 KB ring + 16 KB pad, so at most 2 CTAs fit per SM (K1's residency) and grid 2 x SMs is one even wave
constexpr int SMEM_DYN = ST * STAGE + 16384;

struct GP {
  const int* seq_lens;
  const int* page_table;
  const uint8_t* kc;
  const uint8_t* vc;
  unsigned long long* csum;
  int B, pt_stride, min_chunk;
};

template <int HINT>
__device__ __forceinline__ void cpa(void* s, const void* g, int nb) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const unsigned sa = (unsigned)__cvta_generic_to_shared(s);
  if (HINT)
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(nb) : "memory");
  else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(nb) : "memory");
#endif
}
__device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void wait_g() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}
// K1's ring swizzles (L = 1), so smem write patterns match the shipped kernel
__device__ __forceinline__ int kswz(int x, int ch) { return ch ^ (4 * (x & 1)); }
__device__ __forceinline__ int vswz(int x, int ch) { return ch ^ (2 * ((x >> 1) & 3)); }

template <int MAP, int HINT, int PLAN, int BAR>
__global__ void __launch_bounds__(NT, 2) gp(const GP p) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ long long cum[MAXB + 1];
  __shared__ unsigned long long red[NT / 32];
  const int tid = threadIdx.x;
  if (tid == 0) {
    long long a = 0;
    cum[0] = 0;
    for (int b = 0; b < p.B; ++b) { a += __ldg(p.seq_lens + b); cum[b + 1] = a; }
  }
  __shared__ int plan_s[5];
#ifndef AT_STANDALONE
  if (PLAN && tid < 32) {
    const int C = k1::plan_C(p.seq_lens, p.B, 0, (int)gridDim.x, p.min_chunk);
    int res[5];
    k1::plan_find(p.seq_lens, p.B, 0, C, (int)blockIdx.x, true, res);
    if (tid == 0) { plan_s[0] = res[0]; plan_s[1] = res[1]; plan_s[2] = res[2]; plan_s[3] = res[4]; plan_s[4] = res[3]; }
  }
#endif
  __syncthreads();
  const long long N = cum[p.B];
  long long k0 = N * (long long)blockIdx.x / gridDim.x, k1 = N * (long long)(blockIdx.x + 1) / gridDim.x;
  int ntl = (int)((k1 - k0 + TILE - 1) / TILE);
  int pb = 0;
  if (PLAN) {   // K1: request pb, tiles [t0, t0 + ntl) of its cached keys
    pb = plan_s[0];
    if (pb < 0) { ntl = 0; k0 = 0; k1 = 0; pb = 0; }
    else {
      const int s_idx = plan_s[1], S_b = plan_s[2], nt_b = plan_s[3];
      const int t0 = (int)(((long long)s_idx * nt_b) / S_b);
      ntl = (int)(((long long)(s_idx + 1) * nt_b) / S_b) - t0;
      k0 = cum[pb] + (long long)t0 * TILE;
      k1 = cum[pb + 1];
    }
  }
  constexpr int NK = (MAP == 0) ? 1 : 2;
  int xk[2];
  if (MAP == 0) { xk[0] = tid >> 2; xk[1] = xk[0]; }
  else { xk[0] = 16 * (tid >> 6) + ((tid >> 3) & 7); xk[1] = xk[0] + 8; }
  const int c0 = (MAP == 0) ? (tid & 3) : (tid & 7);
  int bcur[2] = {0, 0};
  auto rowof = [&](int i, int e) -> int {   // physical row of this thread's e-th key of stage i, -1 past the range
    const long long g = k0 + (long long)i * TILE + xk[e];
    if (i >= ntl || g >= k1) return -1;
    if (PLAN) return __ldg(p.page_table + (size_t)pb * p.pt_stride + (int)(g - cum[pb]));
    int b = bcur[e];
    while (g >= cum[b + 1]) ++b;
    bcur[e] = b;
    return __ldg(p.page_table + (size_t)b * p.pt_stride + (int)(g - cum[b]));
  };
  auto issue = [&](int i, const int (&r)[2]) {
    uint8_t* stg = sm + (i % ST) * STAGE;
    if (MAP == 0) {
      const int x = xk[0];
      const bool ok = r[0] >= 0;
      const size_t off = ok ? (size_t)r[0] * ROWB : 0;
      const int nb = ok ? 16 : 0;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int ch = c0 + 4 * u;
        cpa<HINT>(stg + x * ROWB + 16 * kswz(x, ch), p.kc + off + 16 * ch, nb);
        cpa<HINT>(stg + TILE * ROWB + x * ROWB + 16 * vswz(x, ch), p.vc + off + 16 * ch, nb);
      }
    } else {
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int x = xk[u];
        const bool ok = r[u] >= 0;
        const size_t off = ok ? (size_t)r[u] * ROWB : 0;
        const int nb = ok ? 16 : 0;
        cpa<HINT>(stg + x * ROWB + 16 * kswz(x, c0), p.kc + off + 16 * c0, nb);
        cpa<HINT>(stg + TILE * ROWB + x * ROWB + 16 * vswz(x, c0), p.vc + off + 16 * c0, nb);
      }
    }
  };
  auto consume = [&](int i, unsigned long long& acc) {
    const uint8_t* stg = sm + (i % ST) * STAGE;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int x0 = (MAP == 0) ? xk[0] : xk[u];
      const int x = BAR ? (x0 ^ ((MAP == 0) ? 8 : 4)) : x0;   // BAR: the partner warp's row (same quarter)
      const int ch = (MAP == 0) ? (c0 + 4 * u) : c0;
      const uint4 a = *reinterpret_cast<const uint4*>(stg + x * ROWB + 16 * kswz(x, ch));
      const uint4 b = *reinterpret_cast<const uint4*>(stg + TILE * ROWB + x * ROWB + 16 * vswz(x, ch));
      acc += (unsigned long long)a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w;
    }
  };
  int pe0[ST - 1][2];
#pragma unroll
  for (int i = 0; i < ST - 1; ++i) {
    pe0[i][0] = rowof(i, 0);
    pe0[i][1] = (NK == 2) ? rowof(i, 1) : -1;
  }
#pragma unroll
  for (int i = 0; i < ST - 1; ++i) {
    if (i < ntl) issue(i, pe0[i]);
    commit();
  }
  int pfA[2], pfB[2];
  pfA[0] = rowof(ST - 1, 0); pfA[1] = (NK == 2) ? rowof(ST - 1, 1) : -1;
  pfB[0] = rowof(ST, 0); pfB[1] = (NK == 2) ? rowof(ST, 1) : -1;
  unsigned long long acc = 0;
  for (int i = 0; i < ntl; ++i) {
    wait_g<ST - 2>();
    if (BAR) {
      asm volatile("bar.sync %0, 64;\n" ::"r"(1 + (tid >> 6)) : "memory");
      const int nx = i + ST - 1;
      if (nx < ntl) issue(nx, pfA);
      commit();
      consume(i, acc);
    } else {
      consume(i, acc);
      const int nx = i + ST - 1;
      if (nx < ntl) issue(nx, pfA);
      commit();
    }
    pfA[0] = pfB[0]; pfA[1] = pfB[1];
    pfB[0] = rowof(i + ST + 1, 0); pfB[1] = (NK == 2) ? rowof(i + ST + 1, 1) : -1;
  }
  wait_g<0>();
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, d);
  if ((tid & 31) == 0) red[tid >> 5] = acc;
  __syncthreads();
  if (tid == 0) {
    unsigned long long s = 0;
    for (int w = 0; w < NT / 32; ++w) s += red[w];
    atomicAdd(p.csum, s);
  }
}

// L2 flush: read a buffer larger than L2 (60 MB) through L2
__global__ void fl(const uint4* __restrict__ buf, long long n, int* sink) {
  uint32_t a = 0;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const uint4 v = __ldcg(buf + i);
    a ^= v.x ^ v.w;
  }
  if (a == 0x9e3779b9u) sink[0] = (int)a;
}
}  // namespace atg

#ifndef AT_STANDALONE
// probe modes: 0 gp<0,0,0,0> K1 map | 1 gp<1,0,0,0> whole rows | 2 gp<1,0,1,0> + K1 plan | 3 gp<1,0,1,1> + pair barrier
//              4 gp<0,0,1,1> = K1's own loader skeleton (K1 map + K1 plan + pair barrier)
template <int MAP, int HINT, int PLAN, int BAR>
static void at_launch(const atg::GP& p, int grid, cudaStream_t st) {
  static bool set = false;
  if (!set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(atg::gp<MAP, HINT, PLAN, BAR>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        atg::SMEM_DYN));
    set = true;
  }
  atg::gp<MAP, HINT, PLAN, BAR><<<grid, atg::NT, atg::SMEM_DYN, st>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
using gp_fn = void (*)(const atg::GP);
static gp_fn gp_of(int mode) {
  switch (mode) {
    case 0: return atg::gp<0, 0, 0, 0>;
    case 1: return atg::gp<1, 0, 0, 0>;
    case 2: return atg::gp<1, 0, 1, 0>;
    case 3: return atg::gp<1, 0, 1, 1>;
    default: return atg::gp<0, 0, 1, 1>;
  }
}

void at_probe(int64_t mode, torch::Tensor seq_lens, torch::Tensor page_table, torch::Tensor kc, torch::Tensor vc,
              torch::Tensor csum, int64_t B, int64_t grid) {
  TORCH_CHECK(B >= 1 && B <= atg::MAXB, "at_probe: B");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && seq_lens.is_contiguous() && seq_lens.numel() == B, "at_probe: seq_lens");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.is_contiguous() && page_table.dim() == 2, "at_probe: pt");
  TORCH_CHECK(kc.scalar_type() == at::kByte && vc.scalar_type() == at::kByte && kc.is_contiguous() && vc.is_contiguous(),
              "at_probe: caches");
  TORCH_CHECK(csum.scalar_type() == at::kLong && csum.numel() >= 1, "at_probe: csum");
  const c10::cuda::OptionalCUDAGuard guard(kc.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(csum.data_ptr(), 0, 8, st));
  atg::GP p;
  p.seq_lens = seq_lens.data_ptr<int>();
  p.page_table = page_table.data_ptr<int>();
  p.kc = kc.data_ptr<uint8_t>();
  p.vc = vc.data_ptr<uint8_t>();
  p.csum = reinterpret_cast<unsigned long long*>(csum.data_ptr<int64_t>());
  p.B = (int)B;
  p.pt_stride = (int)page_table.size(1);
  p.min_chunk = 2;   // K1_MIN_CHUNK_T[("full", 1)]
  switch ((int)mode) {
    case 0: at_launch<0, 0, 0, 0>(p, (int)grid, st); break;
    case 1: at_launch<1, 0, 0, 0>(p, (int)grid, st); break;
    case 2: at_launch<1, 0, 1, 0>(p, (int)grid, st); break;
    case 3: at_launch<1, 0, 1, 1>(p, (int)grid, st); break;
    default: at_launch<0, 0, 1, 1>(p, (int)grid, st); break;
  }
}

int64_t at_probe_occ(int64_t mode) {
  int nb = 0;
  const gp_fn f = gp_of((int)mode);
  C10_CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, atg::SMEM_DYN));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, f, atg::NT, atg::SMEM_DYN));
  return nb;
}

void at_flush(torch::Tensor buf, torch::Tensor sink, int64_t grid) {
  const c10::cuda::OptionalCUDAGuard guard(buf.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  atg::fl<<<(unsigned)grid, 512, 0, st>>>(reinterpret_cast<const uint4*>(buf.data_ptr()), (long long)(buf.numel() / 16),
                                          sink.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

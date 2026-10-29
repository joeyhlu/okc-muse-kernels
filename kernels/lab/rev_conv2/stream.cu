// stream.cu - R-rev-13 (notes/study/independent_review_oct28.md s21): why do our weight streams plateau at 2.4-3.7 TB/s?
// Cold-L2 read of one buffer (the layer's sizes: 28 / 50 / 162 / 304 MB) by
//   ring  : 1 CTA per SM, lane 0 of a producer warp issues cp.async.bulk into an NS x SB smem ring, NCW consumer warps
//           release slots (optionally spinning a modelled consumer cost per KB). Chunk order:
//           0 static contiguous shares, 1 static interleaved front (chunk c -> CTA c % G), 2 dynamic claim (atomicAdd).
//   ldg   : GD-style plain loads (ld.global.nc.L1::no_allocate.v4), CPS CTAs per SM x 8 warps, U x 16 B in flight per
//           lane, interleaved front; order 2 = dynamic claim of 64 KB blocks per warp.
// Span = first CTA entry -> last CTA exit (%globaltimer), median of reps; one 256 MB flush read before every replay.
// Build: nvcc -O3 -arch=sm_90a -std=c++17 -o stream stream.cu      Run: ./stream --mb 162 --kind ring --order 2
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
constexpr unsigned long long TIMEOUT_NS = 20000000ull;

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ uint32_t su32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t b, unsigned n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, unsigned par) {
  unsigned ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, unsigned par, unsigned* err) {
  if (mbar_try(b, par)) return;
  const unsigned long long t0 = gtime();
  while (!mbar_try(b, par)) if (gtime() - t0 > TIMEOUT_NS) { atomicOr(err, 1u); return; }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, unsigned bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ uint4 ldw(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

template <int NS, int SB, int NCW>
__global__ void __launch_bounds__((NCW + 1) * 32, 1) ring_kernel(const unsigned char* buf, long long nchunk, int order,
                                                                  unsigned* ctr, int cyc_per_kb, unsigned long long* tspan,
                                                                  unsigned* err) {
  extern __shared__ __align__(1024) unsigned char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + NS * SB);
  uint64_t* empty = full + NS;
  volatile long long* tag = reinterpret_cast<volatile long long*>(empty + NS);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, G = gridDim.x, cta = blockIdx.x;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) { mbar_init(su32(&full[s]), 1); mbar_init(su32(&empty[s]), NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    atomicMin(&tspan[0], gtime());
  }
  __syncthreads();
  const long long per = (nchunk + G - 1) / G;
  if (warp == NCW) {
    if (lane == 0) {
      int slot = 0; unsigned par = 0;
      for (long long k = 0;; ++k) {
        long long c;
        if (order == 0) c = k < per ? (long long)cta * per + k : nchunk;
        else if (order == 1) c = (long long)cta + k * G;
        else c = (long long)atomicAdd(ctr, 1u);
        mbar_wait(su32(&empty[slot]), par ^ 1u, err);
        if (c >= nchunk) { tag[slot] = -1; mbar_arrive(su32(&full[slot])); break; }
        tag[slot] = c;
        mbar_expect_tx(su32(&full[slot]), SB);
        bulk_g2s(su32(smem + slot * SB), buf + c * SB, SB, su32(&full[slot]));
        if (++slot == NS) { slot = 0; par ^= 1u; }
      }
    }
  } else {
    int slot = 0; unsigned par = 0;
    for (;;) {
      if (lane == 0) mbar_wait(su32(&full[slot]), par, err);
      __syncwarp();
      if (tag[slot] < 0) break;
      if (cyc_per_kb > 0) { const long long t0 = clock64(); while (clock64() - t0 < (long long)cyc_per_kb * SB / 1024) { } }
      __syncwarp();
      if (lane == 0) mbar_arrive(su32(&empty[slot]));
      if (++slot == NS) { slot = 0; par ^= 1u; }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&tspan[1], gtime());
}

// GD-style: every warp streams 16 B x U per lane per step; front order or dynamic 64 KB blocks
template <int U>
__global__ void __launch_bounds__(256) ldg_kernel(const uint4* buf, long long n16, int order, unsigned* ctr,
                                                  unsigned long long* tspan, unsigned* sink) {
  if (threadIdx.x == 0) atomicMin(&tspan[0], gtime());
  const int lane = threadIdx.x & 31;
  const long long nw = (long long)gridDim.x * (blockDim.x >> 5);
  const long long gw = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const long long step = 32ll * U;                      // uint4 per warp step
  unsigned x = 0;
  if (order != 2) {
    for (long long base = gw * step; base < n16; base += nw * step) {
      uint4 r[U];
#pragma unroll
      for (int u = 0; u < U; ++u) { const long long i = base + u * 32 + lane; r[u] = i < n16 ? ldw(buf + i) : make_uint4(0, 0, 0, 0); }
#pragma unroll
      for (int u = 0; u < U; ++u) x ^= r[u].x ^ r[u].y ^ r[u].z ^ r[u].w;
    }
  } else {
    constexpr long long BLK = 4096;                     // uint4 per claimed block (64 KB)
    const long long nblk = (n16 + BLK - 1) / BLK;
    for (;;) {
      long long b = 0;
      if (lane == 0) b = (long long)atomicAdd(ctr, 1u);
      b = __shfl_sync(0xffffffffu, b, 0);
      if (b >= nblk) break;
      for (long long base = b * BLK; base < min((b + 1) * BLK, n16); base += step) {
        uint4 r[U];
#pragma unroll
        for (int u = 0; u < U; ++u) { const long long i = base + u * 32 + lane; r[u] = i < n16 ? ldw(buf + i) : make_uint4(0, 0, 0, 0); }
#pragma unroll
        for (int u = 0; u < U; ++u) x ^= r[u].x ^ r[u].y ^ r[u].z ^ r[u].w;
      }
    }
  }
  if (x == 0x9e3779b9u) *sink = x;
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&tspan[1], gtime());
}

__global__ void flush_kernel(const uint4* buf, long long n16, unsigned* sink) {
  unsigned x = 0;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) {
    uint4 v;   // normal-policy loads (like the judge's torch.sum flush); __ldcs (evict-first) left earlier lines in L2
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(buf + i));
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x9e3779b9u) *sink = x;
}

static int argi(int argc, char** argv, const char* k, int d) {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], k)) return atoi(argv[i + 1]);
  return d;
}
static std::string args(int argc, char** argv, const char* k, const char* d) {
  for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], k)) return argv[i + 1];
  return d;
}

int main(int argc, char** argv) {
  const int mb = argi(argc, argv, "--mb", 162), order = argi(argc, argv, "--order", 0), reps = argi(argc, argv, "--reps", 30);
  const int cfg = argi(argc, argv, "--cfg", 0), cons = argi(argc, argv, "--cons", 0), cps = argi(argc, argv, "--cps", 4);
  const std::string kind = args(argc, argv, "--kind", "ring");
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  const int G = pr.multiProcessorCount;
  int clk_khz = 0; CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  const long long bytes = (long long)mb * 1000000ll / 65536 * 65536;
  const long long FL = 256ll << 20;
  unsigned char* buf; uint4* fl; unsigned *ctr, *err, *sink; unsigned long long* tspan;
  CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc(&fl, FL)); CK(cudaMemset(buf, 1, bytes)); CK(cudaMemset(fl, 2, FL));
  CK(cudaMalloc(&ctr, 4)); CK(cudaMalloc(&err, 4)); CK(cudaMalloc(&sink, 4)); CK(cudaMalloc(&tspan, 16));
  CK(cudaMemset(err, 0, 4));
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  CK(cudaMemsetAsync(ctr, 0, 4, st));
  CK(cudaMemsetAsync(tspan, 0xFF, 8, st)); CK(cudaMemsetAsync(tspan + 1, 0, 8, st));
  flush_kernel<<<4 * G, 512, 0, st>>>(fl, FL / 16, sink);
  const int cpk = (int)(cons * (clk_khz / 1e6));
  std::string desc;
  if (kind == "ring") {
    auto launch = [&](auto kern, int ns, int sb, int ncw) {
      const size_t sm = (size_t)ns * sb + 2 * ns * 8 + ns * 8;
      CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
      kern<<<G, (ncw + 1) * 32, sm, st>>>(buf, bytes / sb, order, ctr, cpk, tspan, err);
      char d[64]; snprintf(d, sizeof d, "ring NS%d x %dK / %dw", ns, sb / 1024, ncw); desc = d;
    };
    switch (cfg) {
      case 0: launch(ring_kernel<6, 32768, 4>, 6, 32768, 4); break;
      case 1: launch(ring_kernel<12, 16384, 4>, 12, 16384, 4); break;
      case 2: launch(ring_kernel<4, 49152, 4>, 4, 49152, 4); break;
      default: launch(ring_kernel<24, 8192, 4>, 24, 8192, 4); break;
    }
  } else {
    const int u = cfg == 0 ? 8 : 4;
    if (u == 8) ldg_kernel<8><<<cps * G, 256, 0, st>>>(reinterpret_cast<const uint4*>(buf), bytes / 16, order, ctr, tspan, sink);
    else ldg_kernel<4><<<cps * G, 256, 0, st>>>(reinterpret_cast<const uint4*>(buf), bytes / 16, order, ctr, tspan, sink);
    char d[64]; snprintf(d, sizeof d, "ldg %d CTA/SM x 8w x U%d", cps, u); desc = d;
  }
  CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
  std::vector<double> sp;
  for (int r = 0; r < reps + 3; ++r) {
    CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
    unsigned long long hs[2]; CK(cudaMemcpy(hs, tspan, 16, cudaMemcpyDeviceToHost));
    if (r >= 3) sp.push_back((hs[1] - hs[0]) / 1e3);
  }
  unsigned herr = 0; CK(cudaMemcpy(&herr, err, 4, cudaMemcpyDeviceToHost));
  std::sort(sp.begin(), sp.end());
  const double med = sp[sp.size() / 2];
  printf("{\"kind\":\"%s\",\"desc\":\"%s\",\"order\":%d,\"cons\":%d,\"MB\":%.1f,\"span_us\":%.2f,\"p10\":%.2f,\"p90\":%.2f,"
         "\"TBps\":%.3f,\"err\":%u}\n", kind.c_str(), desc.c_str(), order, cons, bytes / 1e6, med, sp[sp.size() / 10],
         sp[sp.size() * 9 / 10], bytes / 1e6 / med, herr);
  return herr ? 2 : 0;
}

"""bgv_src.py: BGV = bulk-ring GEMV for T <= 8 (one CTA per SM, one producer lane issuing cp.async.bulk of contiguous P12 chunks
into an NS x SU-unit smem ring with mbarrier full / empty, NCW consumer warps decoding GD's P12 unit format from smem and running
m16n8k16 BF16 mma with FP32 accumulation). Lab source only (kernels/lab/bgv_lab.py times it vs GD); the layer integration copies it.

Work split: grid = NCL clusters x C CTAs. Row tiles (16 rows) go round-robin to clusters (cluster c: tiles c, c + NCL, ...); rank r of
a cluster owns K units [r * upi, (r + 1) * upi) of each of its tiles (upi = K / 64 / C). A CTA's items (tile, K slice) are one
contiguous byte range of the packed stream, so the producer copies it as consecutive SU-unit chunks (SU x 1.5 KB). Inside a slot,
consumer warp w takes the slot's units w, w + NCW, ... Per item: warps reduce in smem (fixed order), then C == 1 stores BF16, C > 1
ranks st.async their FP32 partial to rank 0's smem (mbarrier complete_tx); rank 0 sums the ranks in order at the end and stores.
The weights do not depend on the predecessor: the producer never waits on griddepcontrol; the consumers wait before reading x.
"""

BGV_CUDA_SRC = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace bgv {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define BGV_SM90 1
#else
#define BGV_SM90 0
#endif

constexpr int UB = 1536;    // one P12 unit (GD format): 16 rows x 64 K = 3 planes x 32 lanes x 16 B
constexpr int IMAX = 8;     // items per CTA (host checks)

struct Args {
  const uint8_t* w;         // packed stream [cluster][rank][item][unit][3][32] x 16 B
  const uint32_t* esc;      // [cta][NCW][32 * ER] entries (j << 20 | lane << 15 | reg << 11 | half << 10 | x7), 0xFFFFFFFF = end
  const uint16_t* x;        // [M, K] bf16 bits (the predecessor's output; read after griddepcontrol.wait)
  void* out;                // [M, ldo] bf16
  int M, K, ldo;
  uint32_t tb;              // P12 window base per byte
  int ntiles, upi, trig;
  int* err;                 // optional: set to 1 on a bounded-wait timeout (outputs NaN)
};

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint4 ldx(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 lds(uint32_t a) {
  uint4 r;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
  return r;
}
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void decode(const uint4 (&p)[3], uint32_t tb, uint32_t (&a)[16]) {   // GD's decode, verbatim
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
__device__ __forceinline__ void patch(uint32_t (&a)[16], uint32_t e, int lane) {   // GD's escape patch, verbatim
  if (lane == (int)((e >> 15) & 31u)) {
    const int r = (int)((e >> 11) & 15u);
    const uint32_t v = (e & 127u) << (8 + (((e >> 10) & 1u) << 4));
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (i == r) a[i] ^= v;
  }
}

__device__ __forceinline__ void mb_init(uint32_t mb, uint32_t cnt) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mb), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint32_t mb) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(mb) : "memory");
}
__device__ __forceinline__ bool mb_test(uint32_t mb, uint32_t par) {
  uint32_t ok;
#if BGV_SM90
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(mb), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(mb), "r"(par) : "memory");
#endif
  return ok != 0u;
}
__device__ __forceinline__ bool mb_wait(uint32_t mb, uint32_t par) {   // bounded (no __trap): false on timeout
  for (uint32_t it = 0; it < (1u << 24); ++it)
    if (mb_test(mb, par)) return true;
  return false;
}

#if BGV_SM90
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void mb_expect(uint32_t mb, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t mb) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(mb) : "memory");
}
__device__ __forceinline__ void fence_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t mapa0(uint32_t a) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, 0;\n" : "=r"(r) : "r"(a));
  return r;
}
__device__ __forceinline__ void st_async(uint32_t addr, float v, uint32_t mb) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n"
               :: "r"(addr), "r"(__float_as_uint(v)), "r"(mb) : "memory");
}
#else
__device__ __forceinline__ void pdl_trigger() {}
__device__ __forceinline__ void pdl_wait() {}
#endif

__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// MODE 0: full GEMV. MODE 1: stream only (consumers wait / release each slot, no decode / mma, outputs untouched).
template <int C, int SU, int NS, int NCW, int MODE, int ER>
__global__ void __launch_bounds__((NCW + 1) * 32, 1) bgv_kernel(const __grid_constant__ Args A) {
  constexpr int SB = SU * UB, UW = SU / NCW, CAP = 32 * ER;
  static_assert(SU % NCW == 0 && (C == 1 || BGV_SM90 || true), "SU must be a multiple of NCW");
  extern __shared__ __align__(1024) uint8_t smem[];
  // x slice [M][upi * 64] bf16 at row stride xrs = upi * 128 + 16 B (odd 16-B units: conflict-free B-fragment loads)
  const int xrs = A.upi * 128 + 16;
  const int xbytes = MODE == 0 ? ((A.M * xrs + 127) & ~127) : 0;
  float* red = reinterpret_cast<float*>(smem + NS * SB + xbytes);  // [2][NCW][4][32]
  float* obuf = red + 2 * NCW * 128;                                // [IMAX][128]      (C > 1, rank 0: own partials)
  float* cbuf = obuf + IMAX * 128;                                  // [IMAX][C - 1][128]
  uint64_t* bars = reinterpret_cast<uint64_t*>(cbuf + IMAX * (C > 1 ? C - 1 : 1) * 128);
  const uint32_t ring0 = su32(smem), full0 = su32(bars), empty0 = su32(bars + NS), cmb0 = su32(bars + 2 * NS);
  const int tid = (int)threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (A.trig) pdl_trigger();
  const int NCL = (int)gridDim.x / C, cta = (int)blockIdx.x, c = cta / C, r = cta % C;
  const int q = A.ntiles / NCL, rm = A.ntiles % NCL;
  const int nit = q + (c < rm ? 1 : 0);
  const int offc = c * q + (c < rm ? c : rm);
  const int nsl = nit * A.upi / SU;
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
      mb_init(full0 + 8 * s, BGV_SM90 ? 1 : 32);
      mb_init(empty0 + 8 * s, NCW * 32);
    }
#if BGV_SM90
    if (C > 1 && r == 0) {
#pragma unroll 1
      for (int i = 0; i < nit; ++i) {
        mb_init(cmb0 + 8 * i, 1);
        mb_expect(cmb0 + 8 * i, (uint32_t)((C - 1) * 128 * 4));
      }
    }
    fence_init();
#endif
  }
  __syncthreads();
#if BGV_SM90
  if constexpr (C > 1) cl_arrive();
#endif
  if (warp == NCW) {   // producer
    const uint8_t* src = A.w + ((size_t)offc * C + (size_t)r * nit) * (size_t)A.upi * UB;
#if BGV_SM90
    if (lane == 0) {
#pragma unroll 1
      for (int k = 0; k < nsl; ++k) {
        const int s = k % NS;
        if (k >= NS && !mb_wait(empty0 + 8 * s, (uint32_t)(((k / NS) - 1) & 1))) {
          if (A.err) atomicExch(A.err, 1);
          break;
        }
        mb_expect(full0 + 8 * s, (uint32_t)SB);
        bulk_g2s(ring0 + (uint32_t)(s * SB), src + (size_t)k * SB, (uint32_t)SB, full0 + 8 * s);
      }
    }
#else
#pragma unroll 1
    for (int k = 0; k < nsl; ++k) {
      const int s = k % NS;
      if (k >= NS && !mb_wait(empty0 + 8 * s, (uint32_t)(((k / NS) - 1) & 1))) {
        if (A.err) atomicExch(A.err, 1);
        break;
      }
      const uint8_t* g = src + (size_t)k * SB;
      const uint32_t d = ring0 + (uint32_t)(s * SB);
      for (int o = lane * 16; o < SB; o += 32 * 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d + (uint32_t)o), "l"(g + o) : "memory");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(full0 + 8 * s) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
    return;
  }

  // consumers (the escape lists are prepare-time data: loaded before the dependency wait)
  uint32_t em[ER];
#pragma unroll
  for (int e = 0; e < ER; ++e) em[e] = A.esc[((size_t)cta * NCW + warp) * CAP + e * 32 + lane];
  pdl_wait();
#if BGV_SM90
  if constexpr (C > 1) {
    if (r != 0) cl_wait();   // rank 0's mbarriers are initialised before its arrive
  }
#endif
  bool ok = true;
  const int g = lane >> 2, t = lane & 3;
  const int spi = A.upi / SU;   // slots per item
  int ecur = 0;
  uint32_t enx = __shfl_sync(0xffffffffu, em[0], 0);
  const int tok = g < A.M ? g : A.M - 1;
  const uint32_t xs0 = ring0 + (uint32_t)(NS * SB);
  if constexpr (MODE == 0) {   // stage this rank's x slice once (cp.async 16 B), consumers only
    const int per = A.upi * 8;   // 16-B chunks per token row
    const uint8_t* xg = reinterpret_cast<const uint8_t*>(A.x) + (size_t)r * A.upi * 128;
#pragma unroll 1
    for (int idx = tid; idx < A.M * per; idx += NCW * 32) {
      const int tk = idx / per, c16 = idx - tk * per;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                   :: "r"(xs0 + (uint32_t)(tk * xrs + c16 * 16)), "l"(xg + (size_t)tk * A.K * 2 + c16 * 16) : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
    asm volatile("bar.sync 1, %0;\n" :: "n"(NCW * 32) : "memory");
  }
  const uint32_t xl = xs0 + (uint32_t)(tok * xrs + 32 * t);
  float acc[2][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
  int kin = 0, it = 0;   // slot index inside the current item, item index
#pragma unroll 1
  for (int k = 0; k < nsl; ++k) {
    const int s = k % NS;
    const bool last = kin + 1 == spi;
    if (!mb_wait(full0 + 8 * s, (uint32_t)((k / NS) & 1))) ok = false;
    if constexpr (MODE == 0) {
      const uint32_t sb = ring0 + (uint32_t)(s * SB) + (uint32_t)lane * 16u;
#pragma unroll
      for (int i = 0; i < UW; ++i) {
        const uint32_t ub = sb + (uint32_t)((warp + i * NCW) * UB);
        uint4 P[3];
        P[0] = lds(ub);
        P[1] = lds(ub + 512);
        P[2] = lds(ub + 1024);
        const uint32_t xa = xl + (uint32_t)((kin * SU + warp + i * NCW) * 128);
        uint4 X[2];
        X[0] = lds(xa);
        X[1] = lds(xa + 16);
        uint32_t a[16];
        decode(P, A.tb, a);
        const int j = k * UW + i;
        while ((int)(enx >> 20) == j) {
          patch(a, enx, lane);
          ++ecur;
          uint32_t src = em[0];
#pragma unroll
          for (int e = 1; e < ER; ++e)
            if ((ecur >> 5) == e) src = em[e];
          enx = ecur < CAP ? __shfl_sync(0xffffffffu, src, ecur & 31) : 0xFFFFFFFFu;
        }
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const uint4 xv = X[qq >> 1];
          const uint32_t b0 = (qq & 1) ? xv.z : xv.x;
          const uint32_t b1 = (qq & 1) ? xv.w : xv.y;
          mma16816(acc[qq & 1], a[4 * qq], a[4 * qq + 1], a[4 * qq + 2], a[4 * qq + 3], b0, b1);
        }
      }
    }
    mb_arrive(empty0 + 8 * s);
    if constexpr (MODE == 0) {
      if (last) {   // item end
        const int rt = c + it * NCL;
        float* rb = red + (it & 1) * NCW * 128;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          rb[(warp * 4 + i) * 32 + lane] = acc[0][i] + acc[1][i];
          acc[0][i] = 0.f;
          acc[1][i] = 0.f;
        }
        asm volatile("bar.sync 1, %0;\n" :: "n"(NCW * 32) : "memory");
        if (tid < 128) {
          const int l = tid & 31, cc = tid >> 5;
          float v = 0.f;
#pragma unroll
          for (int w2 = 0; w2 < NCW; ++w2) v += rb[(w2 * 4 + cc) * 32 + l];
          if constexpr (C == 1) {
            const int row = rt * 16 + (l >> 2) + 8 * (cc >> 1), tk = 2 * (l & 3) + (cc & 1);
            if (tk < A.M)
              reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tk * A.ldo + row] = __float2bfloat16_rn(ok ? v : __int_as_float(0x7fc00000));
          } else {
#if BGV_SM90
            if (r == 0) obuf[it * 128 + tid] = ok ? v : __int_as_float(0x7fc00000);
            else st_async(mapa0(su32(cbuf + (it * (C - 1) + r - 1) * 128 + tid)), ok ? v : __int_as_float(0x7fc00000),
                          mapa0(cmb0 + 8 * it));
#endif
          }
        }
      }
    }
    kin = last ? 0 : kin + 1;
    it += last ? 1 : 0;
  }
#if BGV_SM90
  if constexpr (C > 1 && MODE == 0) {
    if (r == 0 && tid < 128) {
      const int l = tid & 31, cc = tid >> 5;
#pragma unroll 1
      for (it = 0; it < nit; ++it) {
        const bool okc = mb_wait(cmb0 + 8 * it, 0u);
        float v = obuf[it * 128 + tid];
#pragma unroll
        for (int rr = 0; rr < C - 1; ++rr) v += cbuf[(it * (C - 1) + rr) * 128 + tid];
        const int rt = c + it * NCL;
        const int row = rt * 16 + (l >> 2) + 8 * (cc >> 1), tk = 2 * (l & 3) + (cc & 1);
        if (tk < A.M)
          reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tk * A.ldo + row] = __float2bfloat16_rn(okc ? v : __int_as_float(0x7fc00000));
      }
    }
  }
#endif
  if (!ok && A.err && lane == 0) atomicExch(A.err, 1);
}

// latency-bound predecessor for the PDL arms: one CTA per SM spins ns nanoseconds; triggers its dependents at entry
__global__ void __launch_bounds__(128) bgv_pred(long long ns, int* sink) {
  pdl_trigger();
  const long long t0 = gtime();
  while (gtime() - t0 < ns) {}
  if (threadIdx.x == 0 && sink) sink[blockIdx.x] = 1;
}

#if BGV_LOCAL
constexpr int SMEM_MAX = 101376;
#else
constexpr int SMEM_MAX = 232448;
#endif
template <int C, int SU, int NS, int NCW, int MODE, int ER>
constexpr int smem_bytes() {
  return NS * SU * UB + (2 * NCW * 128 + IMAX * 128 + IMAX * (C > 1 ? C - 1 : 1) * 128) * 4 + (2 * NS + IMAX) * 8;
}

template <int C, int SU, int NS, int NCW, int MODE, int ER>
static int launch(const Args& A, int grid, int pdl, cudaStream_t st) {
  auto fn = bgv_kernel<C, SU, NS, NCW, MODE, ER>;
  const int SM = smem_bytes<C, SU, NS, NCW, MODE, ER>() + (MODE == 0 ? ((A.M * (A.upi * 128 + 16) + 127) & ~127) : 0);
  if (SM > SMEM_MAX) return -2;
  static bool set = false;
  if (!set) {
    int e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
    if (e) return e;
    set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid, 1, 1);
  cfg.blockDim = dim3((unsigned)((NCW + 1) * 32), 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute attr[2];
  unsigned na = 0;
  if (C > 1) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = (unsigned)C;
    attr[na].val.clusterDim.y = 1;
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
  return (int)cudaLaunchKernelEx(&cfg, fn, A);
}

template <int C, int SU, int NS, int NCW, int MODE, int ER>
static int info(int64_t* res) {
  auto fn = bgv_kernel<C, SU, NS, NCW, MODE, ER>;
  constexpr int SM = smem_bytes<C, SU, NS, NCW, MODE, ER>();
  cudaFuncAttributes fa;
  int e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, (NCW + 1) * 32, SM);
  if (e) return e;
  int ncl = -1;
  if (C > 1) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(C * 66), 1, 1);
    cfg.blockDim = dim3((unsigned)((NCW + 1) * 32), 1, 1);
    cfg.dynamicSmemBytes = SM;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = (unsigned)C;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    e = (int)cudaOccupancyMaxActiveClusters(&ncl, fn, &cfg);
    if (e) return e;
  }
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = SM; res[4] = ncl; res[5] = C; res[6] = SU;
  res[7] = NS; res[8] = NCW; res[9] = MODE; res[10] = ER;
  return 0;
}

}  // namespace bgv

int bgv_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb,
                int ntiles, int upi, int grid, int pdl, int* err, cudaStream_t st) {
  bgv::Args A;
  A.w = reinterpret_cast<const uint8_t*>(w); A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x); A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb; A.ntiles = ntiles; A.upi = upi; A.trig = (pdl >> 1) & 1; A.err = err;
  switch (inst) {
#define BGV_X(I, C, SU, NS, NCW, MODE, ER) case I: return bgv::launch<C, SU, NS, NCW, MODE, ER>(A, grid, pdl, st);
@INSTS@
#undef BGV_X
    default: return -1;
  }
}
int bgv_info_raw(int64_t inst, int64_t* res) {
  switch (inst) {
#define BGV_X(I, C, SU, NS, NCW, MODE, ER) case I: return bgv::info<C, SU, NS, NCW, MODE, ER>(res);
@INSTS@
#undef BGV_X
    default: return -1;
  }
}
int bgv_pred_raw(int64_t ns, int grid, int* sink, cudaStream_t st) {
  bgv::bgv_pred<<<grid, 128, 0, st>>>((long long)ns, sink);
  return (int)cudaGetLastError();
}
'''

BGV_CPP_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
int bgv_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb,
                int ntiles, int upi, int grid, int pdl, int* err, cudaStream_t st);
int bgv_info_raw(int64_t inst, int64_t* res);
int bgv_pred_raw(int64_t ns, int grid, int* sink, cudaStream_t st);
void bgv_run(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, int64_t M, int64_t K,
             int64_t ldo, int64_t tb, int64_t ntiles, int64_t upi, int64_t grid, int64_t pdl, torch::Tensor err) {
  TORCH_CHECK(M >= 1 && M <= 8 && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
              "bgv_run: M <= 8, bf16 x / out");
  int* ep = err.numel() > 0 ? err.data_ptr<int>() : nullptr;
  const int e = bgv_run_raw(inst, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), (int)M, (int)K, (int)ldo,
                            (uint32_t)tb, (int)ntiles, (int)upi, (int)grid, (int)pdl, ep, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "bgv_run: inst ", inst, " CUDA error ", e);
}
std::vector<int64_t> bgv_info(int64_t inst) {
  std::vector<int64_t> r(11, 0);
  const int e = bgv_info_raw(inst, r.data());
  TORCH_CHECK(e == 0, "bgv_info: inst ", inst, " error ", e);
  return r;
}
void bgv_pred(int64_t ns, int64_t grid, torch::Tensor sink) {
  const int e = bgv_pred_raw(ns, (int)grid, sink.data_ptr<int>(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "bgv_pred: CUDA error ", e);
}
'''

BGV_IMAX = 8
BGV_UB = 1536


def bgv_pack(ns, wv, C, NCL, NCW, SU, ER):
    """P12 pack (GD unit format and window base, GD helpers from ns) in BGV order + per-(CTA, consumer warp) escape lists."""
    import torch
    n, k = wv.shape
    nrt, nu = n // 16, k // 64
    assert n % 16 == 0 and k % 64 == 0 and nu % C == 0
    upi = nu // C
    assert upi % SU == 0 and SU % NCW == 0, (upi, SU, NCW)
    q, rm = divmod(nrt, NCL)
    nit = [q + (1 if c < rm else 0) for c in range(NCL)]
    assert C == 1 or max(nit) <= BGV_IMAX, f"items per CTA {max(nit)} > {BGV_IMAX}"   # the cluster buffers hold IMAX items
    dev = wv.device
    base = ns["_gd_window_base"](wv)
    i32 = ns["_gd_i32"]
    fr = ns["_gd_frag"](wv)                       # [nrt, nu, 32, 4, 8]
    lo = fr & 0xFF
    h7 = (fr >> 8) & 0x7F
    sg = (fr >> 15) & 1
    d = h7 - base
    isesc = (d < 0) | (d > 7)
    d = torch.where(isesc, torch.zeros_like(d), d)
    dA, dB, sA, sB = d[..., :4], d[..., 4:], sg[..., :4], sg[..., 4:]
    sBr = sB[..., [3, 0, 1, 2]]
    byte = (dA | (sBr << 3) | (dB << 4) | (sA << 7)).to(torch.int64)
    cw = i32(byte[..., 0] | (byte[..., 1] << 8) | (byte[..., 2] << 16) | (byte[..., 3] << 24))
    lo64 = lo.to(torch.int64)
    l0 = i32(lo64[..., 0] | (lo64[..., 1] << 8) | (lo64[..., 2] << 16) | (lo64[..., 3] << 24))
    l1 = i32(lo64[..., 4] | (lo64[..., 5] << 8) | (lo64[..., 6] << 16) | (lo64[..., 7] << 24))
    p0 = torch.stack([l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]], dim=-1)
    p1 = torch.stack([l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]], dim=-1)
    units = torch.stack([p0, p1, cw], dim=2)      # [nrt, nu, 3, 32, 4] int32 = 1536 B per unit
    rts, rks = [], []
    for c in range(NCL):
        for r in range(C):
            for i in range(nit[c]):
                rts.append(c + i * NCL)
                rks.append(r)
    rt_t = torch.tensor(rts, dtype=torch.int64, device=dev)
    rk_t = torch.tensor(rks, dtype=torch.int64, device=dev)
    u_idx = (rk_t[:, None] * upi + torch.arange(upi, device=dev)[None, :]).reshape(-1)
    r_idx = rt_t[:, None].expand(-1, upi).reshape(-1)
    v = units[r_idx, u_idx].contiguous().view(-1)
    ncta = NCL * C
    cap = 32 * ER
    esc = torch.full((ncta * NCW * cap,), -1, dtype=torch.int32, device=dev)
    pos = isesc.nonzero()
    nesc = int(pos.shape[0])
    maxe = 0
    if nesc:
        rt, u, ln, qq, jv = (pos[:, i] for i in range(5))
        c = rt % NCL
        it = rt // NCL
        rr = u // upi
        jj = u % upi
        cta = c * C + rr
        s = it * upi + jj
        lu = s % SU
        w = lu % NCW
        j = (s // SU) * (SU // NCW) + lu // NCW
        assert int(j.max()) < 4095
        x7 = (h7[rt, u, ln, qq, jv] ^ base).to(torch.int64)
        ent = (j << 20) | (ln << 15) | ((4 * qq + (jv >> 1)) << 11) | ((jv & 1) << 10) | x7
        key = cta * NCW + w
        order = torch.argsort(key * 4096 + j)
        ks, es = key[order], ent[order]
        cnt = torch.bincount(ks, minlength=ncta * NCW)
        maxe = int(cnt.max())
        if maxe > cap:
            raise ValueError(f"bgv escape cap: a warp has {maxe} > {cap} escapes")
        rank = torch.arange(nesc, device=dev) - (torch.cumsum(cnt, 0) - cnt)[ks]
        esc[ks * cap + rank] = i32(es)
    return {"w": v, "esc": esc, "tb": base * 0x01010101, "ntiles": nrt, "upi": upi, "grid": ncta, "nesc": nesc, "maxesc": maxe,
            "nit_max": max(nit), "nit_min": min(nit)}

"""gd_patches.py: GD = gv1's lossless-P12 weight-streaming GEMVs (kernels/lab/gv1_c1.py, the source of bf16_proj judge
check d62947a889f04059 = 1.388) inside the decode layer, on k3_v39 (tools/gen_k3_gd.py writes k3_gd_<n>.py).

What GD replaces (1:1, the kernel count is unchanged), per gd bucket (1 | 2-8 | 9-16 | 17-32 tokens):
  * qkv at every bucket (today P1 at T <= 8, P1M at 16, DG1 at 32);
  * o at every bucket (today P1 / P1M / DG1);
  * the pooler at bucket 8 (today P1 on the main stream next to the side-stream FP32 router, which is untouched);
  * the unpooler: this GPU's 1024-row slice at T <= 8 into the zero buffer (today SU1 at T = 1, P1 at T = 2-8), the full
    matrix at 9-32 (today P1M).
Not replaced: the T = 1 pooler (inside G2's fused pooler + FP32 router + tail kernel), the cat GEMM at T >= 9 (P1M / DG1),
the routers (FP32 input at T <= 8 stays), everything at T > 32.

Kernel = gv1's gv_kernel, P12 only, with three layer changes: (1) `griddepcontrol.launch_dependents` at entry when the host
sets trig (every GD route is one wave by construction: prepare drops a route whose CTAs exceed occupancy x SMs); (2) the
first ST units of weights, the warp's escape list and the cluster mbarrier init are issued before `griddepcontrol.wait`, x
after it; (3) an OUT 1 (FP32) store, unused by the shipped routes. Launch = cudaLaunchKernelEx with the cluster attribute
(S > 1) and programmatic stream serialization (pdl bit 0). The K3 early trigger (SU1_K3_TRIG) stays on whenever the GD
unpooler slice runs with PDL. Weights are packed in prepare (gv1's pack, verbatim), verified by a bit-exact unpack, then each
route is self-tested at every token count class of its bucket (judge recipe: FP64 reference rounded to BF16, worst row,
exact probe rows, outlier channels; NaN-filled outputs; launches with PDL + trigger after a fill kernel; bitwise repeat;
untouched columns beyond the slice). Every failure prints a FALLBACK line and leaves today's route in place.
GD_ENABLE is read at call time (A/B arms); GD_ENABLE False = k3_v39 exactly.

    python -I tools/gd_patches.py in.py out.py
"""
import pathlib
import py_compile
import sys

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
std::vector<int64_t> gd_info(int64_t inst) {
  std::vector<int64_t> r(4, 0);
  const int e = gd_info_raw(inst, r.data());
  TORCH_CHECK(e == 0, "gd_info: inst ", inst, " error ", e);
  return r;
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

// one 64-k unit (4 k-steps) -> 16 A-fragment registers a[4q + i] (P12: AND, IADD, SHF, AND, IADD, 4 PRMT per 8 weights)
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

// escape entry: step << 20 | lane << 15 | reg << 11 | half << 10 | xor7
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

// out[t, n] = sum_k x[t, k] W[n, k] for the CTA's WR row tiles, tokens tok0 .. tok0 + 8 MT - 1.
// grid = (row groups, S, token slices); cluster (1, S, 1) when S > 1.
template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP>
__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {
  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;
  constexpr int NC = MT == 1 ? 2 : 1;
  constexpr int NO = WR * MT * 128;     // CTA output values
  constexpr int XB = XP ? 2 : 1;
  __shared__ float red[NW][MT * 4][32];
  __shared__ float pbuf[S > 1 ? S - 1 : 1][S > 1 ? NO : 1];
  __shared__ __align__(8) unsigned long long mbar;
  if (A.trig) pdl_trigger();
  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;
  const int wk = warp % WK;
  const int g = lane >> 2, t = lane & 3;
  const int split = S > 1 ? (int)blockIdx.y : 0;
  const int G = (int)(gridDim.x * gridDim.y) * NW;
  const int gw = ((int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x) * NW + warp;
  const int u0 = (split * WK + wk) * KU;
  const int tok0 = (int)blockIdx.z * (8 * MT);
  const size_t wstep = (size_t)G * (NP * 32);
  const uint4* wp = A.w + (size_t)gw * (NP * 32) + lane;

  // 1. prepare()-time data before the PDL wait: the first ST units of weights, the warp's escape list, the cluster mbarrier
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
  // 2. x (and the output) belong to the predecessor kernel. A no-op without programmatic serialization.
  pdl_wait();
  // x: lane (g, t) reads token tok0 + 8 mt + g (clamped to M - 1), k = 64 u + 16 t .. + 15 (two 16-B loads)
  const uint4* xb = reinterpret_cast<const uint4*>(A.x + (size_t)u0 * 64 + 16 * t);
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

  // 3. the stream: decode slot -> refill it -> escapes -> MMAs
#pragma unroll
  for (int s = 0; s < KU; ++s) {
    const int slot = s % ST;
    const int xs = XP ? (s & 1) : 0;
    if (XP) {
      if (s + 1 < KU) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          X[(s + 1) & (XB - 1)][mt][0] = ldx(xb + xo[mt] + (s + 1) * 8);
          X[(s + 1) & (XB - 1)][mt][1] = ldx(xb + xo[mt] + (s + 1) * 8 + 1);
        }
      }
    } else if (s > 0) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        X[0][mt][0] = ldx(xb + xo[mt] + s * 8);
        X[0][mt][1] = ldx(xb + xo[mt] + s * 8 + 1);
      }
    }
    uint32_t a[16];
    decode(R[slot], A.tb, a);
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
  }

  // 4. warp partials -> smem -> sum over the WK warps of each row tile (-> cluster leader when S > 1)
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = acc[0][mt][i];
      if constexpr (NC == 2) v += acc[1][mt][i];
      red[warp][mt * 4 + i][lane] = v;
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
    // bounded wait (no trap): a protocol fault shows up as NaN outputs, which the prepare() self-test rejects
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

template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP>
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
  return (int)cudaLaunchKernelEx(&cfg, gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP>, A);
}

template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP>
static int info(int64_t* res) {
  cudaFuncAttributes fa;
  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP>;
  int e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, 0);
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes;
  return e;
}

// bit-exact unpack of a packed matrix back to [N, K] (verification); one warp per (global warp, step)
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

}  // namespace gd

using namespace gd;

#define GD_INST(ID, OUT, MT, WR, WK, KU, ST, S, MB, XP) \
  case ID: return launch<OUT, MT, WR, WK, KU, ST, S, MB, XP>(A, nbx, nz, pdl, st);
#define GD_INFO(ID, OUT, MT, WR, WK, KU, ST, S, MB, XP) \
  case ID: return info<OUT, MT, WR, WK, KU, ST, S, MB, XP>(res);

int gd_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb,
               int nbx, int nz, int pdl, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  switch (inst) {
@INST@
    default: return -1;
  }
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

PY = '''

# GD (tools/gd_patches.py): gv1's lossless-P12 weight-streaming GEMVs (kernels/lab/gv1_c1.py; bf16_proj check d62947a8)
# for qkv / o / the bucket-8 pooler / the unpooler (slice at T <= 8, full at 9-32). GD_ENABLE is read at call time.
GD_ENABLE = True
GD_BUILD = True          # prepare: build / pack / self-test GD (False: nothing, = k3_v39)
GD_REQUIRE = False       # prepare raises if a GD_ROUTE entry is not live (A/B validity)
GD_PREPARE_BY_S = 230.0
GD_SMS = 132
GD_ESC_CAP = 32
# (proj, gd bucket) -> (mt, wr, wk, ku, st, s, xp): the routes bf16_proj check d62947a8 autotuned on the judge H200
GD_ROUTE = {("qkv", 1): (1, 1, 4, 8, 2, 4, 1), ("qkv", 8): (1, 1, 4, 8, 2, 4, 1), ("qkv", 16): (2, 1, 4, 8, 1, 4, 0),
            ("qkv", 32): (4, 1, 4, 8, 1, 4, 0),
            ("o", 1): (1, 2, 4, 4, 2, 1, 0), ("o", 8): (1, 1, 8, 2, 2, 1, 0), ("o", 16): (2, 2, 4, 4, 2, 1, 0),
            ("o", 32): (4, 2, 4, 4, 1, 1, 0),
            ("pooler", 8): (1, 1, 8, 16, 2, 1, 1),
            ("unpooler", 1): (1, 1, 16, 4, 2, 1, 0), ("unpooler", 8): (1, 1, 16, 4, 2, 1, 0),
            ("unpooler", 16): (2, 2, 4, 16, 2, 1, 0), ("unpooler", 32): (4, 2, 4, 16, 1, 1, 0)}
GD_PDL = {"qkv": False, "o": True, "pooler": False, "unpooler": True}    # = P1's per-projection PDL policy (G2_P1_PDL)
GD_TRIG = {"qkv": True, "o": True, "pooler": True, "unpooler": True}    # early launch_dependents (every route is one wave)
GD_K3_TRIG = True        # keep K3's early trigger when the GD slice runs with PDL (= SU1 / P1 today)
GD_DIMS = {"qkv": (2304, HIDDEN), "o": (HIDDEN, 1024), "pooler": (D_EXPERT, HIDDEN), "unpooler": (HIDDEN, D_EXPERT)}
GD_WEIGHT = {"qkv": "qkv_gate", "o": "o", "pooler": "pooler", "unpooler": "unpooler"}
GD_SELF_TEST_T = {1: (1,), 8: (8, 2, 5), 16: (16, 9), 32: (32, 17)}
GD_TOL = (2e-3, 1e-2)    # rel L2 and worst row vs the FP64 reference rounded to BF16 (probe rows exact); gv1's judge recipe
GD_SELF_TEST_PDL = (0, 3, 3)   # plain; PDL + early trigger behind a fill kernel, twice (bitwise repeat)
GD_ONE_WAVE = True       # prepare drops a route whose CTAs exceed occupancy x SMs (trig assumes one wave)
_GD = {"mod": None, "keys": None, "build_s": 0.0}
_GD_SEEN = set()
GD_CPP_SRC = @CPP@
GD_CUDA_SRC = @CUDA@


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
    return keys


def _gd_build():
    _pt1_wait("gd")
    keys = _gd_keys()
    if _GD["mod"] is not None and _GD["keys"] == keys:
        return _GD["mod"]
    from torch.utils.cpp_extension import load_inline
    inst = "\\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    info = "\\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    cuda = GD_CUDA_SRC.replace("@INST@", inst).replace("@INFO@", info)
    tag = hashlib.sha1((GD_CPP_SRC + cuda).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"gd_{tag}", cpp_sources=GD_CPP_SRC, cuda_sources=cuda, functions=["gd_run", "gd_info", "gd_unpack"],
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

'''

# ---- PRT (k3_gd_2+, apply(prt=True)): the T = 1 pooler inside a G2-style fused kernel on gv1's stream --------------------
# G2's prt_kernel<1, 1> (k3_v39: pooler + FP32 router + routing tail in one launch; router weights cp.async'd to smem before
# the PDL wait; the last CTA by ticket runs the tail) with P1's pooler stream replaced by gv1's P12 stream of geometry
# (1, 16, 8, 1) ST 2 (judge 12.67 us standalone vs P1's 15.65): 256 CTAs x 512 threads = G2's grid, so CTA c is still pooler
# row tile c and router32's CTA (c % 32, c / 32). Router and tail expressions are G2's, verbatim.
PRT_CPP = r'''
int gd_prt_raw(const void* w, const void* esc, uint32_t tb, const void* x, void* y, const void* x32, const void* w32, void* part,
               const void* nw, const void* bias, const void* lstart, void* xn, void* tw, void* tid, void* lid, void* tcnt,
               float eps, float mult, float rscale, float deps, int pdl, cudaStream_t st);
int gd_prt_info_raw(int64_t* res);
void gd_prt(torch::Tensor w, torch::Tensor esc, int64_t tb, torch::Tensor x, torch::Tensor y, torch::Tensor x32,
            torch::Tensor w32, torch::Tensor part, torch::Tensor nw, torch::Tensor bias, torch::Tensor lstart, torch::Tensor xn,
            torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, torch::Tensor tcnt, double eps, double mult, double rscale,
            double deps, int64_t pdl) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(0) == 1 && x.size(1) == 8192 &&
              x32.scalar_type() == at::kFloat && x32.is_contiguous() && x32.numel() == 8192 && w32.scalar_type() == at::kFloat &&
              w32.is_contiguous() && w32.numel() == 256 * 8192 && y.numel() == 4096 && xn.numel() == 4096 && part.numel() == 8 * 256 &&
              tw.numel() == 8 && tid.numel() == 8 && lid.numel() == 8 && nw.numel() == 4096 && bias.numel() == 256,
              "gd_prt: T = 1 shapes");
  const int e = gd_prt_raw(w.data_ptr(), esc.data_ptr(), (uint32_t)tb, x.data_ptr(), y.data_ptr(), x32.data_ptr(), w32.data_ptr(),
                           part.data_ptr(), nw.data_ptr(), bias.data_ptr(), lstart.data_ptr(), xn.data_ptr(), tw.data_ptr(),
                           tid.data_ptr(), lid.data_ptr(), tcnt.data_ptr(), (float)eps, (float)mult, (float)rscale, (float)deps,
                           (int)pdl, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gd_prt: CUDA error ", e, " ", cudaGetErrorString((cudaError_t)e));
}
std::vector<int64_t> gd_prt_info() {
  std::vector<int64_t> r(4, 0);
  TORCH_CHECK(gd_prt_info_raw(r.data()) == 0, "gd_prt_info");
  return r;
}
'''

PRT_DEV = r'''
// GD-PRT (T = 1): G2's prt_kernel<1, 1> with gv1's P12 pooler stream (geometry (1, 16, KU, 1)) in place of P1's stream.
constexpr int PD = 4096, NE = 256, TK = 8, PNT = 512, RK = 8192, RKS4 = 256;

struct PArgs {
  const float* x32; const float* w32; float* part; const __nv_bfloat16* nw; const float* bias; const int* lstart;
  __nv_bfloat16* xn; float* tw; int* tid; int* lid; int* tcnt;
  float eps, mult, rscale, deps;
};

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

template <int KU, int ST>
__global__ void __launch_bounds__(512, 2) gprt_kernel(const __grid_constant__ Args A, const __grid_constant__ PArgs G) {
  constexpr int NW = 16, NP = 3;
  extern __shared__ float4 dsm[];
  float4* const wsm = dsm;               // router weight rows bx*8 .. bx*8+7, K slice sl: [8][256] float4
  float4* const xsm = dsm + 8 * RKS4;    // router input K slice sl: [256] float4 (token 0)
  __shared__ float red[NW][4][32];
  __shared__ float tred[NW];
  __shared__ int s_last;
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rt = (int)blockIdx.x, m = A.M;
  const int bx = rt & 31, sl = rt >> 5, k0 = sl * 1024;
  // 0. router weights (prepare-time data) -> smem, issued before the PDL wait
  for (int idx = tid; idx < 8 * RKS4; idx += PNT) {
    const int r = idx >> 8, c = idx & 255;
    cp16(wsm + idx, reinterpret_cast<const float4*>(G.w32 + (size_t)(bx * 8 + r) * RK + k0) + c);
  }
  cp_commit();
  // 1. pooler row tile rt, warp = K slice of KU units: the first ST units and the escapes before the PDL wait (gv1's stream)
  const int gw = rt * NW + warp;
  const size_t wstep = (size_t)((int)gridDim.x * NW) * (NP * 32);
  const uint4* wp = A.w + (size_t)gw * (NP * 32) + lane;
  uint4 R[ST][NP];
#pragma unroll
  for (int i = 0; i < ST; ++i)
#pragma unroll
    for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)i * wstep + p * 32);
  const uint32_t em = lde(A.esc + (size_t)gw * 32 + lane);
  pdl_wait();
  for (int idx = tid; idx < RKS4; idx += PNT) cp16(xsm + idx, reinterpret_cast<const float4*>(G.x32 + k0) + idx);
  cp_commit();
  const uint4* xb = reinterpret_cast<const uint4*>(A.x + (size_t)warp * KU * 64 + 16 * (lane & 3));
  int ecur = 0;
  uint32_t enx = __shfl_sync(0xffffffffu, em, 0);
  float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int s = 0; s < KU; ++s) {
    const uint4 x0 = ldx(xb + s * 8), x1 = ldx(xb + s * 8 + 1);
    uint32_t a[16];
    decode(R[s % ST], A.tb, a);
    if (s + ST < KU) {
#pragma unroll
      for (int p = 0; p < NP; ++p) R[s % ST][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);
    }
    while ((int)(enx >> 20) == s) {
      patch(a, enx, lane);
      ++ecur;
      enx = ecur < 32 ? __shfl_sync(0xffffffffu, em, ecur & 31) : 0xFFFFFFFFu;
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint4 xv = (q >> 1) ? x1 : x0;
      mma16816(acc[q & 1], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], (q & 1) ? xv.z : xv.x, (q & 1) ? xv.w : xv.y);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) red[warp][i][lane] = acc[0][i] + acc[1][i];
  cp_wait_all();
  __syncthreads();
  if (tid < 128) {
    const int l = tid & 31, c = tid >> 5;
    float v = 0.f;
#pragma unroll
    for (int k2 = 0; k2 < NW; ++k2) v += red[k2][c][l];
    const int row = rt * 16 + (l >> 2) + 8 * (c >> 1), tok = 2 * (l & 3) + (c & 1);
    if (tok < m) reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
  }
  // 2. router32_kernel<1>'s CTA (bx, sl): warp w < 8 = router row bx*8 + w over K slice sl (G2's RM 1 expressions)
  if (warp < 8) {
    const int row = bx * 8 + warp;
    float racc = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = i * 32 + lane;
      const float4 wv = wsm[warp * RKS4 + c];
      const float4 xv = xsm[c];
      racc = fmaf(wv.x, xv.x, racc);
      racc = fmaf(wv.y, xv.y, racc);
      racc = fmaf(wv.z, xv.z, racc);
      racc = fmaf(wv.w, xv.w, racc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) racc += __shfl_xor_sync(0xffffffffu, racc, o);
    if (lane == 0) G.part[(long)sl * NE + row] = racc;
  }
  // 3. the last CTA (ticket; no spin) runs the tail (G2's expressions for M = 1)
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(G.tcnt, 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid == 0) atomicExch(G.tcnt, 0);
  const uint4 wq = *(reinterpret_cast<const uint4*>(G.nw) + tid);
  const uint4 yq = __ldcg(reinterpret_cast<const uint4*>(A.out) + tid);
  {
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&yq);
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) tred[warp] = ss;
  }
  __syncthreads();
  {
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) tot += tred[i];
    const float rstd = rsqrtf(tot / (float)PD + G.eps);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&yq);
    const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq);
    __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((__bfloat162float(h[i]) * rstd) * __bfloat162float(w[i]));
    *(reinterpret_cast<uint4*>(G.xn) + tid) = *reinterpret_cast<const uint4*>(o8);
  }
  if (warp >= m) return;
  // tail_kernel<false, 8>'s warp-0 routing for token 0 (lane l holds experts 8l .. 8l+7)
  const float4* pbias = reinterpret_cast<const float4*>(G.bias) + lane * 2;
  const float4 c0b = pbias[0], c1b = pbias[1];
  const float4* pq0 = reinterpret_cast<const float4*>(G.part) + lane * 2;
  float4 a0 = __ldcg(pq0), a1 = __ldcg(pq0 + 1);
#pragma unroll
  for (int q = 1; q < 8; ++q) {   // FP32 split-K partials summed in a fixed order
    const float4* pq = reinterpret_cast<const float4*>(G.part + q * NE) + lane * 2;
    const float4 b0 = __ldcg(pq), b1 = __ldcg(pq + 1);
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
      G.tw[k] = sel_s[k] * inv;
      G.tid[k] = sel_id[k];
      const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
      G.lid[k] = rel < 32 ? rel : -1;
    }
  }
}
'''

PRT_HOST = r'''
int gd_prt_raw(const void* w, const void* esc, uint32_t tb, const void* x, void* y, const void* x32, const void* w32, void* part,
               const void* nw, const void* bias, const void* lstart, void* xn, void* tw, void* tid, void* lid, void* tcnt,
               float eps, float mult, float rscale, float deps, int pdl, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w); A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x); A.out = y; A.M = 1; A.K = RK; A.ldo = PD; A.tb = tb; A.trig = 0;
  PArgs G;
  G.x32 = reinterpret_cast<const float*>(x32); G.w32 = reinterpret_cast<const float*>(w32); G.part = reinterpret_cast<float*>(part);
  G.nw = reinterpret_cast<const __nv_bfloat16*>(nw); G.bias = reinterpret_cast<const float*>(bias);
  G.lstart = reinterpret_cast<const int*>(lstart); G.xn = reinterpret_cast<__nv_bfloat16*>(xn);
  G.tw = reinterpret_cast<float*>(tw); G.tid = reinterpret_cast<int*>(tid); G.lid = reinterpret_cast<int*>(lid);
  G.tcnt = reinterpret_cast<int*>(tcnt);
  G.eps = eps; G.mult = mult; G.rscale = rscale; G.deps = deps;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(256u, 1u, 1u);
  cfg.blockDim = dim3((unsigned)PNT, 1u, 1u);
  cfg.dynamicSmemBytes = (size_t)9 * RKS4 * 16;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  return (int)cudaLaunchKernelEx(&cfg, gprt_kernel<8, 2>, A, G);
}

int gd_prt_info_raw(int64_t* res) {
  cudaFuncAttributes fa;
  int e = (int)cudaFuncGetAttributes(&fa, gprt_kernel<8, 2>);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, gprt_kernel<8, 2>, PNT, 9 * RKS4 * 16);
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes;
  return e;
}
'''

PRT_PY = '''

# GD prt (tools/gd_patches.py, prt=True): the T = 1 pooler inside a G2-style fused pooler + FP32 router + tail kernel on gv1's
# P12 stream; GD_ENABLE and GD_PRT are read at call time (False = G2's prt).
GD_PRT = True
GD_PRT_PDL = True        # = G2_PRT_PDL
GD_PRT_GEO = (1, 16, 8, 1)   # (wr, wk, ku, s) of the pooler pack gprt_kernel<8, 2> streams


def _gd_prt_run(p, x, router_input, state, data, pdl):
    dev = x.device
    y = torch.empty((1, D_EXPERT), dtype=torch.bfloat16, device=dev)
    part = torch.empty((8, 1, N_EXPERTS), dtype=torch.float32, device=dev)
    xn = torch.empty((1, D_EXPERT), dtype=torch.bfloat16, device=dev)
    tw = torch.empty((1, TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((1, TOP_K), dtype=torch.int32, device=dev)
    lid = torch.empty((1, TOP_K), dtype=torch.int32, device=dev)
    v8 = state["v8"]
    p["mod"].gd_prt(p["w"], p["esc"], p["tb"], x, y, router_input, state["router"], part, v8["nw"], v8["bias"],
                    data["local_start"], xn, tw, tid, lid, p["tcnt"], RMS_EPS, 0.1767766952966369, 1.0, 1e-15, pdl)
    return xn, tw, tid, lid


def _gd_prt(ffn_input, router_input, state, data, tokens, mode, tail):
    st = state.get("gd")
    if not (GD_ENABLE and GD_PRT and st is not None and st.get("prt") is not None) or tokens != 1:
        return None
    if mode != "f32" or tail != "sep" or router_input is None or state.get("v8") is None:
        return None
    x = ffn_input
    if not (x.dtype == torch.bfloat16 and x.dim() == 2 and tuple(x.shape) == (1, HIDDEN) and x.is_contiguous()
            and x.data_ptr() % 16 == 0 and router_input.dtype == torch.float32 and router_input.is_contiguous()
            and router_input.shape == x.shape and router_input.data_ptr() % 16 == 0):
        return None
    try:
        out = _gd_prt_run(st["prt"], x, router_input, state, data, 1 if GD_PRT_PDL else 0)
        if ("prt", 1) not in _GD_SEEN:
            _GD_SEEN.add(("prt", 1))
            print(f"[gd] first call prt T=1: {st['prt']['desc']}", flush=True)
        return out
    except Exception as ex:
        st["prt"] = None
        print(f"[gd] prt raised at call time, G2's prt from now on: {ex!r}", flush=True)
        return None


def _gd_prt_selftest(state, p, w):
    dev = w.device
    parts = []
    for seed, ls in ((4243, 0), (4244, 37), (4245, 200)):
        gen = torch.Generator(device=dev).manual_seed(seed)
        h = torch.randn((1, HIDDEN), generator=gen, device=dev)
        h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
        hb = h.to(torch.bfloat16).contiguous()
        d = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
        part = torch.empty((8, 1, N_EXPERTS), dtype=torch.float32, device=dev)
        state["v8"]["mod"].v8_router32(h, state["router"], part)
        pool = (hb.double() @ w.double().T).to(torch.bfloat16)
        ref = _v8_tail(pool, part, state, d)
        outs = [_gd_prt_run(p, hb, h, state, d, pdl) for pdl in (0, 1, 1)]
        torch.cuda.synchronize()
        cnt = int(p["tcnt"][0])
        for k, got in enumerate(outs):
            ex = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
            ew = float((got[1] - ref[1]).abs().max())
            ids = bool(torch.equal(got[2], ref[2])) and bool(torch.equal(got[3], ref[3]))
            if not (ex < 2e-3 and ew < 1e-6 and ids and bool(torch.isfinite(got[0].float()).all()) and cnt == 0):
                return False, f"ls {ls} call {k}: xn rel {ex:.2e} w maxdiff {ew:.1e} ids {ids} ticket {cnt}"
        if not all(bool(torch.equal(a, b)) for a, b in zip(outs[1], outs[2])):
            return False, f"ls {ls}: repeat not bitwise"
        parts.append(f"ls {ls} xn {ex:.1e}")
    return True, " ".join(parts)


def _gd_prt_prepare(state, mod, live):
    if not GD_PRT:
        return None
    w = state["dense"]["pooler"]
    if state.get("v8") is None or state.get("router") is None:
        print("[gd] FALLBACK prt: no v8 tail / router state -> G2's prt", flush=True)
        return None
    why = None
    try:
        info = mod.gd_prt_info()
        sms = torch.cuda.get_device_properties(w.device).multi_processor_count
        if info[1] > 64 or (GD_ONE_WAVE and info[2] * sms < 256):
            why = f"regs {info[0]} spill {info[1]} occ {info[2]} for 256 CTAs"
        else:
            wr, wk, ku, s = GD_PRT_GEO
            wv = w.contiguous()
            pk = _gd_pack(wv, wr, wk, ku, s, _gd_window_base(wv))
            un = torch.empty_like(wv)
            mod.gd_unpack(pk["w"], pk["esc"], un, wv.shape[1], pk["G"], pk["nbx"], wr, wk, ku, pk["tb"])
            torch.cuda.synchronize()
            exact = bool(torch.equal(un.view(torch.int16), wv.view(torch.int16)))
            del un
            if not exact:
                why = "pack / bit-exact unpack failed"
            else:
                p = {"mod": mod, "w": pk["w"], "esc": pk["esc"], "tb": pk["tb"], "desc": f"prt(1,16,8,1)s2r{info[0]}",
                     "tcnt": torch.zeros((1,), dtype=torch.int32, device=w.device)}
                ok, msg = _gd_prt_selftest(state, p, wv)
                if not ok:
                    why = f"self-test: {msg}"
    except Exception as ex:
        why = f"EXC {ex!r}"
    if why is not None:
        print(f"[gd] FALLBACK prt: {why} -> G2's prt", flush=True)
        return None
    print(f"[gd] self-test prt {p['desc']}: ok ({msg}; {pk['nesc']} escapes)", flush=True)
    live.append(f"pooler/1 {p['desc']}")
    return p

'''

PRT_EDITS = [
    ("prt build functions",
     'functions=["gd_run", "gd_info", "gd_unpack"]',
     'functions=["gd_run", "gd_info", "gd_unpack", "gd_prt", "gd_prt_info"]'),
    ("prt prepare",
     '    torch.cuda.empty_cache()\n    state["gd"] = {"mod": mod, "route": routes} if routes else None\n',
     '    prt = _gd_prt_prepare(state, mod, live)\n    torch.cuda.empty_cache()\n'
     '    state["gd"] = {"mod": mod, "route": routes, "prt": prt} if (routes or prt is not None) else None\n'),
    ("prt route at T = 1",
     '    g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n',
     '    g2out = _gd_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n    if g2out is None:\n'
     '        g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n'),
]

PY_ANCHOR = "\ndef kernel_fn(data, state):\n"

EDITS = [
    ("PT1 build job",
     '    if SBB1_ENABLE:\n        jobs.append(("sbb1", _sbb1_build))\n',
     '    if SBB1_ENABLE:\n        jobs.append(("sbb1", _sbb1_build))\n'
     '    if GD_BUILD:\n        jobs.append(("gd", _gd_build))\n'),
    ("prepare: _gd_prepare after _su1_prepare",
     "    _su1_prepare(state, t0)\n    _p206_prepare(state, t0)\n",
     "    _su1_prepare(state, t0)\n    _gd_prepare(state, t0)\n    _p206_prepare(state, t0)\n"),
    ("pooler route",
     'def _v13_pooler(ffn_input, dense, state):\n    y = _v13_p1(state, "pooler", ffn_input)\n',
     'def _v13_pooler(ffn_input, dense, state):\n    y = _gd(state, "pooler", ffn_input)\n'
     '    if y is None:\n        y = _v13_p1(state, "pooler", ffn_input)\n'),
    ("K3 early trigger follows the GD slice",
     "def _su1_k3trig(state, tokens):\n",
     "def _su1_k3trig(state, tokens):\n    g = _gd_k3trig(state, tokens)\n    if g is not None:\n        return g\n"),
    ("qkv route",
     '    qkv = _v20_p1m(state, "qkv", data["attn_input"])\n',
     '    qkv = _gd(state, "qkv", data["attn_input"])\n    if qkv is None:\n'
     '        qkv = _v20_p1m(state, "qkv", data["attn_input"])\n'),
    ("o route",
     '    attn_output = _su1(state, "o", o, tokens) if ks1_ran else None\n',
     '    attn_output = _gd(state, "o", o)\n    if attn_output is None and ks1_ran:\n'
     '        attn_output = _su1(state, "o", o, tokens)\n'),
    ("unpooler route",
     '    ffn_output = _su1(state, "unpooler", expert_output, tokens)\n',
     '    ffn_output = _gd_unpool(state, expert_output, tokens)\n    if ffn_output is None:\n'
     '        ffn_output = _su1(state, "unpooler", expert_output, tokens)\n'),
]


def block(prt=False):
    """The GD Python block with the two kernel sources inlined as raw string literals (+ the T = 1 prt kernel if prt)."""
    cpp, cuda, py = GD_CPP_SRC, GD_CUDA_SRC, PY
    if prt:
        ns_end = "}  // namespace gd\n"
        if cuda.count(ns_end) != 1:
            raise RuntimeError("gd patch: namespace end anchor")
        cpp, cuda, py = cpp + PRT_CPP, cuda.replace(ns_end, PRT_DEV + "\n" + ns_end) + PRT_HOST, py + PRT_PY
    for src in (cpp, cuda):
        if "'''" in src or src.endswith("\\"):
            raise RuntimeError("gd patch: a kernel source holds ''' or ends in a backslash")
    return py.replace("@CPP@", "r'''" + cpp + "'''").replace("@CUDA@", "r'''" + cuda + "'''")


def apply(text, prt=False):
    for name in ("GD_ENABLE", "GD_CUDA_SRC", "def _gd(", "def _gd_prepare(", "_gd_build"):
        if name in text:
            raise RuntimeError(f"gd patch: {name!r} already in the base")
    for name, old, new in EDITS:
        if text.count(old) != 1:
            raise RuntimeError(f"gd patch {name!r}: anchor found {text.count(old)}x (want 1)")
        text = text.replace(old, new)
    if text.count(PY_ANCHOR) != 1:
        raise RuntimeError("gd patch: kernel_fn anchor")
    text = text.replace(PY_ANCHOR, block(prt) + PY_ANCHOR)
    if prt:
        for name, old, new in PRT_EDITS:
            if text.count(old) != 1:
                raise RuntimeError(f"gd patch {name!r}: anchor found {text.count(old)}x (want 1)")
            text = text.replace(old, new)
    return text


if __name__ == "__main__":
    s, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    t = s.read_text(encoding="utf-8")
    out = apply(t)
    d.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(d), doraise=True)
    print(f"gd: applied {s.name} -> {d} ({len(out) - len(t):+d} chars, compiles)")

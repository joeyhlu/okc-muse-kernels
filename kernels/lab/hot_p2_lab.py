"""P2 lab kernel (kernels/muse_moe_w4a16/moe_v4_study.md §10): base cost of a weight-stationary SS-wgmma expert GEMM at
decode shapes. Each touched expert is read once: one pass of N <= 128 tokens per expert (a second pass only above 128).

LAB FILE. Box runs only, through tools/box/hot_p2.py. Never submit it, and never import it from a scored file.

Kernel p2_gemm<MODE>: one persistent CTA per SM, 384 threads = 3 warpgroups.
  WG0  warp 0 lane 0 = producer: per 64-k stage, one 9216 B bulk copy of the expert's repacked NVFP4 stage
       (256 rows x 64 k: 8 KB E2M1 codes + 1 KB re-encoded E4M3 scales) and one N x 128 B bulk copy of the token
       tile (BF16, core-matrix layout). Both land in ring slot (NST 4); one expect_tx barrier.
       warps 1-3 = dequant: codes -> exact BF16 (K2A v3's deq4 / bmul2, the same bits as the shipped kernel) ->
       the 256 x 64 A tile in smem (no-swizzle K-major core matrices, K2B v1's ADDR_IL layout), double-buffered
       (NA 2), fence.proxy.async before the arrive.
       setmaxnreg.dec 56.
  WG1/2 consumers (setmaxnreg.inc 224): WG c owns rows 128c .. 128c + 127 = two m64 tiles. Per stage: 4 k16 x
       2 tiles of wgmma m64nNk16 SS, A and B both from smem descriptors (no LDS, no register operands). The
       accumulators are the only register inputs and are touched outside an open stage only. K2B v1's validated
       discipline: fence_acc + wg_fence, commit, wait_group 1, release the previous stage.
  MODE 0 full, 1 no dequant (A stale, timing only), 2 no wgmma (timing only).
Item = (pass, 256-row block), all of K. Items go to CTAs by host-side LPT on a byte/MMA cost model (lab only: a real
kernel plans on device). Output: FP32 Y[pass][row][N] (valid token columns only), times the expert's global scale.
Prep kernels:
  p2_xpack     token rows -> W13 B tiles;
  p2_swiglu    Y13 -> bf16(silu(gate) * up) -> W2 B tiles.
Bounded waits only (R9): mbar_wait returns false after WAIT_BUDGET clocks and sets err[0]; the role then returns.
No __trap. No raw pointers or descriptors kept across calls.
"""
import hashlib
import time

NCLS = (8, 16, 32, 48, 64, 96, 128)   # wgmma N classes (one compiled consumer item per class)
MODES = (0, 1, 2)
RB, KS, WST = 256, 64, 9216
EXPERT_W13_STAGES = 32 * 64            # 256-row blocks x 64-k stages of W13 (8192 x 4096)
EXPERT_W2_STAGES = 16 * 64             # W2 (4096 x 4096)


def _wgmma_ss(n):
    """wgmma.mma_async m64n{n}k16 f32 += bf16 x bf16, A and B from smem descriptors, both K-major (K2B v1's form,
    scale-d fixed to 1: accumulators are zeroed before each item, outside any open stage)."""
    nr = n // 2
    regs = ", ".join("%%%d" % i for i in range(nr))
    outs = ", ".join('"+f"(d[%d])' % i for i in range(nr))
    return ("template <> __device__ __forceinline__ void wgmma_ss<%d>(float* d, uint64_t da, uint64_t db) {\n"
            "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
            "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n\"\n"
            "    : %s\n    : \"l\"(da), \"l\"(db));\n}\n" % (n, n, regs, nr, nr + 1, outs))


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace p2 {
constexpr int RB = 256, KS = 64, WST = 9216, NMAX = 128, XSB = NMAX * 128, SLOT = WST + XSB;   // SLOT 25600
constexpr int NST = 4, NA = 2, ABYTES = RB * KS * 2;                                            // A tile 32768
constexpr int OFF_A = NST * SLOT;                                                               // 102400
constexpr int OFF_BAR = OFF_A + NA * ABYTES;                                                    // 167936
constexpr int SMEM = OFF_BAR + 8 * (2 * NST + 2 * NA);                                          // 168032
constexpr long long WAIT_BUDGET = 1ll << 31;   // clock64 cycles (~1 s) per wait -> err flag, role returns (R9)

struct Params {
  const uint8_t* w;        // [E][nrb][nks][WST]
  const uint8_t* xp;       // B tiles: pass p at byte xoff[p]: [nks][N/8][kc 8][row 8][16 B]
  const long long* xoff;
  float* y;                // pass p at float yoff[p]: [nrb * 256][N]
  const long long* yoff;
  const int* pe;           // pass -> local expert
  const int* pn;           // pass -> N class
  const int* pc;           // pass -> real tokens (<= N)
  const int* items;        // (pass << 8) | rb, CTA c owns items [cbeg[c], cbeg[c + 1])
  const int* cbeg;
  const float* gs;         // per local expert: global scale x W_SHIFT
  int* err;
  int nrb, nks;
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
__device__ __forceinline__ bool mbar_wait(uint32_t b, uint32_t par, int* err) {   // bounded (R9)
  if (mbar_try(b, par)) return true;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); return false; }
  }
  return true;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// K2A v3 decode (BITPOS3 word layout): 8 E2M1 codes -> 4 bf16x2 words (value * 2^-127 class), exact
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}
// wgmma smem descriptor, no swizzle (layout type 0): start >> 4, LBO 128 B (next core matrix along K), SBO 1024 B
// (next 8-row group); K2B v1's IL descriptor (validated on the H200)
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(1024 >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int NR>
__device__ __forceinline__ void fence_acc(float* d) {
#pragma unroll
  for (int i = 0; i < NR; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db);
@WGMMA@

// ---------------- producer (WG0 warp 0 lane 0) ----------------
__device__ __forceinline__ void producer(const Params& P, uint32_t sb, int beg, int end, uint32_t FULL0, uint32_t EMP0) {
  int it = 0;
  for (int q = beg; q < end; ++q) {
    const int item = __ldg(P.items + q);
    const int p = item >> 8, rb = item & 255;
    const int e = __ldg(P.pe + p), n = __ldg(P.pn + p);
    const uint32_t xb = (uint32_t)n * 128u;
    const uint8_t* ws = P.w + ((size_t)(e * P.nrb + rb) * P.nks) * WST;
    const uint8_t* xs = P.xp + __ldg(P.xoff + p);
    for (int ks = 0; ks < P.nks; ++ks, ++it) {
      const int slot = it % NST;
      if (!mbar_wait(EMP0 + 8 * slot, ((it / NST) & 1) ^ 1, P.err)) return;
      const uint32_t dst = sb + slot * SLOT;
      const uint32_t fb = FULL0 + 8 * slot;
      mbar_expect_tx(fb, (uint32_t)WST + xb);
      bulk_g2s(dst, ws + (size_t)ks * WST, (uint32_t)WST, fb);
      bulk_g2s(dst + WST, xs + (size_t)ks * xb, xb, fb);
    }
  }
}

// ---------------- dequant (WG0 warps 1-3, 96 threads) ----------------
template <int MODE>
__device__ __forceinline__ void dequant(const Params& P, uint8_t* smem, int beg, int end, uint32_t FULL0, uint32_t EMP0,
                                        uint32_t AF0, uint32_t AE0) {
  const int dt = threadIdx.x - 32, lane = threadIdx.x & 31;
  int it = 0;
  for (int q = beg; q < end; ++q) {
    for (int ks = 0; ks < P.nks; ++ks, ++it) {
      const int slot = it % NST, ab = it % NA;
      if (!mbar_wait(FULL0 + 8 * slot, (it / NST) & 1, P.err)) return;
      if (!mbar_wait(AE0 + 8 * ab, ((it / NA) & 1) ^ 1, P.err)) return;
      if constexpr (MODE != 1) {
        const uint8_t* st = smem + slot * SLOT;
        const uint32_t* cw = reinterpret_cast<const uint32_t*>(st);           // [kc 8][row 256] code words
        const uint32_t* scw = reinterpret_cast<const uint32_t*>(st + 8192);   // [row 256] 4 scale bytes (16-k groups)
        uint8_t* ad = smem + OFF_A + ab * ABYTES;
#pragma unroll 2
        for (int idx = dt; idx < 2048; idx += 96) {
          const int kc = idx >> 8, r = idx & 255;
          const uint32_t wv = cw[idx];
          const uint32_t sw = scw[r];
          const uint32_t S = __byte_perm(sw, 0u, 0x4040u + 0x0101u * (uint32_t)(kc >> 1)) * 16u + 0x75807580u;
          uint32_t o[4];
          deq4(wv, o);
          uint4 v;
          v.x = bmul2(o[0], S); v.y = bmul2(o[1], S); v.z = bmul2(o[2], S); v.w = bmul2(o[3], S);
          // K2B_ADDR_IL(r, kc): 8-row groups of 1024 B, 16 B chunk kc at 128 B, row r & 7 at 16 B
          *reinterpret_cast<uint4*>(ad + ((r >> 3) * 1024) + (kc * 128) + ((r & 7) * 16)) = v;
        }
      }
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");   // generic STS -> visible to wgmma (async proxy)
      __syncwarp();
      if (lane == 0) { mbar_arrive(AF0 + 8 * ab); mbar_arrive(EMP0 + 8 * slot); }
    }
  }
}

// ---------------- consumer item (one WG, rows 128 cw .. 128 cw + 127 of the item's 256-row block) ----------------
template <int MODE, int N>
__device__ __forceinline__ bool citem(const Params& P, uint32_t sb, int& it, int p, int rb, int cw, uint32_t FULL0,
                                      uint32_t EMP0, uint32_t AF0, uint32_t AE0) {
  constexpr int NR = N / 2;
  const int t = threadIdx.x & 127, w = t >> 5, lane = t & 31;
  float acc0[NR], acc1[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) { acc0[i] = 0.f; acc1[i] = 0.f; }
  bool ok = true;
  int prev = -1;
  for (int ks = 0; ks < P.nks; ++ks, ++it) {
    const int slot = it % NST, ab = it % NA;
    ok = mbar_wait(FULL0 + 8 * slot, (it / NST) & 1, P.err) && mbar_wait(AF0 + 8 * ab, (it / NA) & 1, P.err);
    if (!ok) break;
    if constexpr (MODE != 2) {
      const uint32_t a0 = sb + OFF_A + ab * ABYTES + cw * 16384;   // 8-row group 16 cw (1024 B per group)
      const uint64_t da0 = make_desc(a0), da1 = make_desc(a0 + 8192), db = make_desc(sb + slot * SLOT + WST);
      fence_acc<NR>(acc0);
      fence_acc<NR>(acc1);
      wg_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) {   // k16 step j: +256 B along K in both operands (desc units of 16 B)
        wgmma_ss<N>(acc0, da0 + (uint64_t)(16 * j), db + (uint64_t)(16 * j));
        wgmma_ss<N>(acc1, da1 + (uint64_t)(16 * j), db + (uint64_t)(16 * j));
      }
      wg_commit();
      fence_acc<NR>(acc0);
      fence_acc<NR>(acc1);
    }
    wg_wait<1>();   // the previous stage's wgmmas retired: its A buffer and ring slot are free
    if (prev >= 0 && t == 0) { mbar_arrive(AE0 + 8 * (prev % NA)); mbar_arrive(EMP0 + 8 * (prev % NST)); }
    prev = it;
  }
  wg_wait<0>();
  fence_acc<NR>(acc0);
  fence_acc<NR>(acc1);
  if (prev >= 0 && t == 0) { mbar_arrive(AE0 + 8 * (prev % NA)); mbar_arrive(EMP0 + 8 * (prev % NST)); }
  if (!ok) return false;
  const int cnt = __ldg(P.pc + p);
  const float gs = __ldg(P.gs + __ldg(P.pe + p));
  float* yb = P.y + __ldg(P.yoff + p);
  const int row0 = rb * 256 + cw * 128 + w * 16 + (lane >> 2);
#pragma unroll
  for (int v = 0; v < NR; ++v) {   // K2B_ACC_ROW / K2B_ACC_COL
    const int col = ((v >> 2) * 8) + ((lane & 3) * 2) + (v & 1);
    const int rr = ((v >> 1) & 1) * 8;
    if (col < cnt) {
      yb[(size_t)(row0 + rr) * N + col] = acc0[v] * gs;
      yb[(size_t)(row0 + 64 + rr) * N + col] = acc1[v] * gs;
    }
  }
  return true;
}

template <int MODE>
__device__ __forceinline__ void consumer(const Params& P, uint32_t sb, int beg, int end, int cw, uint32_t FULL0,
                                         uint32_t EMP0, uint32_t AF0, uint32_t AE0) {
  int it = 0;
  for (int q = beg; q < end; ++q) {
    const int item = __ldg(P.items + q);
    const int p = item >> 8, rb = item & 255;
    bool ok;
    switch (__ldg(P.pn + p)) {
      case 8: ok = citem<MODE, 8>(P, sb, it, p, rb, cw, FULL0, EMP0, AF0, AE0); break;
      case 16: ok = citem<MODE, 16>(P, sb, it, p, rb, cw, FULL0, EMP0, AF0, AE0); break;
      case 32: ok = citem<MODE, 32>(P, sb, it, p, rb, cw, FULL0, EMP0, AF0, AE0); break;
      case 48: ok = citem<MODE, 48>(P, sb, it, p, rb, cw, FULL0, EMP0, AF0, AE0); break;
      case 64: ok = citem<MODE, 64>(P, sb, it, p, rb, cw, FULL0, EMP0, AF0, AE0); break;
      case 96: ok = citem<MODE, 96>(P, sb, it, p, rb, cw, FULL0, EMP0, AF0, AE0); break;
      case 128: ok = citem<MODE, 128>(P, sb, it, p, rb, cw, FULL0, EMP0, AF0, AE0); break;
      default: ok = false; if (threadIdx.x == 128) atomicExch(P.err + 1, 1); break;
    }
    if (!ok) return;
  }
}

template <int MODE>
__global__ void __launch_bounds__(384, 1) p2_gemm(const Params P) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2;
  const uint32_t sb = sa(smem);
  const uint32_t FULL0 = sb + OFF_BAR, EMP0 = FULL0 + 8 * NST, AF0 = EMP0 + 8 * NST, AE0 = AF0 + 8 * NA;
  if (tid == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(FULL0 + 8 * i, 1); mbar_init(EMP0 + 8 * i, 5); }   // 3 dequant warps + 2 WGs
    for (int i = 0; i < NA; ++i) { mbar_init(AF0 + 8 * i, 3); mbar_init(AE0 + 8 * i, 2); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const int beg = P.cbeg[blockIdx.x], end = P.cbeg[blockIdx.x + 1];
  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;\n" ::: "memory");
    if (warp == 0) {
      if (lane == 0) producer(P, sb, beg, end, FULL0, EMP0);
      return;
    }
    dequant<MODE>(P, smem, beg, end, FULL0, EMP0, AF0, AE0);
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 224;\n" ::: "memory");
  consumer<MODE>(P, sb, beg, end, wg - 1, FULL0, EMP0, AF0, AE0);
}

// ---------------- prep: token rows -> W13 B tiles (16 B per thread, zero rows above the pass's tokens) ----------------
__global__ void __launch_bounds__(256) p2_xpack(const __nv_bfloat16* x, const int* ptok, const long long* tokoff,
                                                const long long* coff, const long long* xoff, const int* pn,
                                                const int* pc, int npass, uint8_t* xp) {
  const long long total = coff[npass];
  for (long long g = (long long)blockIdx.x * 256 + threadIdx.x; g < total; g += (long long)gridDim.x * 256) {
    int lo = 0, hi = npass - 1;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (coff[mid] <= g) lo = mid; else hi = mid - 1; }
    const int p = lo, n8 = pn[p] >> 3;
    const long long c = g - coff[p];
    const int k8 = (int)(c / pn[p]), n = (int)(c - (long long)k8 * pn[p]);   // n fastest
    const int ks = k8 >> 3, kc = k8 & 7;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (n < pc[p]) v = *reinterpret_cast<const uint4*>(x + (size_t)ptok[tokoff[p] + n] * 4096 + 8 * k8);
    *reinterpret_cast<uint4*>(xp + xoff[p] + (((size_t)ks * n8 + (n >> 3)) * 8 + kc) * 128 + (n & 7) * 16) = v;
  }
}

// ---------------- prep: Y13 (gate rows 0-4095, up rows 4096-8191) -> bf16(silu(g) * u) -> W2 B tiles ----------------
__global__ void __launch_bounds__(256) p2_swiglu(const float* y13, const long long* yoff, const long long* coff,
                                                 const long long* xoff, const int* pn, const int* pc, int npass,
                                                 uint8_t* xp) {
  const long long total = coff[npass];
  for (long long g = (long long)blockIdx.x * 256 + threadIdx.x; g < total; g += (long long)gridDim.x * 256) {
    int lo = 0, hi = npass - 1;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (coff[mid] <= g) lo = mid; else hi = mid - 1; }
    const int p = lo, N = pn[p], n8 = N >> 3;
    const long long c = g - coff[p];
    const int k8 = (int)(c / N), n = (int)(c - (long long)k8 * N);
    const int ks = k8 >> 3, kc = k8 & 7;
    uint32_t o[4] = {0u, 0u, 0u, 0u};
    if (n < pc[p]) {
      const float* yb = y13 + yoff[p];
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        float r2[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int i = 8 * k8 + 2 * h + u;
          const float gv = yb[(size_t)i * N + n], uv = yb[(size_t)(4096 + i) * N + n];
          r2[u] = gv / (1.f + __expf(-gv)) * uv;
        }
        __nv_bfloat162 b2 = __floats2bfloat162_rn(r2[0], r2[1]);
        o[h] = *reinterpret_cast<uint32_t*>(&b2);
      }
    }
    *reinterpret_cast<uint4*>(xp + xoff[p] + (((size_t)ks * n8 + (n >> 3)) * 8 + kc) * 128 + (n & 7) * 16) =
        make_uint4(o[0], o[1], o[2], o[3]);
  }
}
}  // namespace p2

template <int MODE>
static void gemm_launch(const p2::Params& p, int G) {
  p2::p2_gemm<MODE><<<G, 384, p2::SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// one call, stages by bit: 1 token gather (W13 B tiles), 2 W13 GEMM, 4 SwiGLU pack (W2 B tiles), 8 W2 GEMM
void p2run(int64_t mode, int64_t what, torch::Tensor x, torch::Tensor w13, torch::Tensor w2, torch::Tensor gs13,
           torch::Tensor gs2, torch::Tensor ptok, torch::Tensor meta64, torch::Tensor meta32, torch::Tensor items13,
           torch::Tensor cbeg13, torch::Tensor items2, torch::Tensor cbeg2, torch::Tensor xp13, torch::Tensor xp2,
           torch::Tensor y13, torch::Tensor y2, torch::Tensor err, int64_t npass, int64_t G) {
  // meta64 (int64): [tokoff npass | coff npass+1 | xoff npass | yoff13 npass | yoff2 npass]
  // meta32 (int32): [pe npass | pn npass | pc npass]
  const long long* m64 = reinterpret_cast<const long long*>(meta64.data_ptr<int64_t>());   // int64_t is long on LP64
  const int* m32 = meta32.data_ptr<int>();
  const int np = (int)npass;
  const long long *tokoff = m64, *coff = m64 + np, *xoff = coff + np + 1, *yoff13 = xoff + np, *yoff2 = yoff13 + np;
  const int *pe = m32, *pn = m32 + np, *pc = m32 + 2 * np;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  p2::Params p;
  p.xoff = xoff; p.pe = pe; p.pn = pn; p.pc = pc; p.err = err.data_ptr<int>(); p.nks = 64;
  auto gemm = [&](int nrb) {
    if (mode == 0) gemm_launch<0>(p, (int)G);
    else if (mode == 1) gemm_launch<1>(p, (int)G);
    else gemm_launch<2>(p, (int)G);
    (void)nrb;
  };
  if (what & 1) {
    p2::p2_xpack<<<4 * (int)G, 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), ptok.data_ptr<int>(),
                                             tokoff, coff, xoff, pn, pc, np, reinterpret_cast<uint8_t*>(xp13.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (what & 2) {
    p.w = reinterpret_cast<const uint8_t*>(w13.data_ptr()); p.xp = reinterpret_cast<const uint8_t*>(xp13.data_ptr());
    p.y = y13.data_ptr<float>(); p.yoff = yoff13; p.items = items13.data_ptr<int>(); p.cbeg = cbeg13.data_ptr<int>();
    p.gs = gs13.data_ptr<float>(); p.nrb = 32;
    gemm(32);
  }
  if (what & 4) {
    p2::p2_swiglu<<<4 * (int)G, 256, 0, st>>>(y13.data_ptr<float>(), yoff13, coff, xoff, pn, pc, np,
                                              reinterpret_cast<uint8_t*>(xp2.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (what & 8) {
    p.w = reinterpret_cast<const uint8_t*>(w2.data_ptr()); p.xp = reinterpret_cast<const uint8_t*>(xp2.data_ptr());
    p.y = y2.data_ptr<float>(); p.yoff = yoff2; p.items = items2.data_ptr<int>(); p.cbeg = cbeg2.data_ptr<int>();
    p.gs = gs2.data_ptr<float>(); p.nrb = 16;
    gemm(16);
  }
}

std::vector<int64_t> p2attrs() {   // per MODE: regs, local (spill) bytes, max dyn smem ok, occupancy
  std::vector<int64_t> r;
  auto one = [&](const void* f) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, p2::SMEM);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, 384, p2::SMEM) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ});
  };
  one((const void*)p2::p2_gemm<0>);
  one((const void*)p2::p2_gemm<1>);
  one((const void*)p2::p2_gemm<2>);
  return r;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void p2run(int64_t mode, int64_t what, torch::Tensor x, torch::Tensor w13, torch::Tensor w2, torch::Tensor gs13,
           torch::Tensor gs2, torch::Tensor ptok, torch::Tensor meta64, torch::Tensor meta32, torch::Tensor items13,
           torch::Tensor cbeg13, torch::Tensor items2, torch::Tensor cbeg2, torch::Tensor xp13, torch::Tensor xp2,
           torch::Tensor y13, torch::Tensor y2, torch::Tensor err, int64_t npass, int64_t G);
std::vector<int64_t> p2attrs();
"""

CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def cuda_source():
    return CUDA_SRC.replace("@WGMMA@", "".join(_wgmma_ss(n) for n in NCLS))


def _gate_wgmma():
    """tools/box/sass_gate.py --subst '@WGMMA@=call:_gate_wgmma' (zero arguments, no module globals: a copy of
    _wgmma_ss over NCLS; build() asserts the two agree)."""
    out = []
    for n in (8, 16, 32, 48, 64, 96, 128):
        nr = n // 2
        regs = ", ".join("%%%d" % i for i in range(nr))
        outs = ", ".join('"+f"(d[%d])' % i for i in range(nr))
        out.append("template <> __device__ __forceinline__ void wgmma_ss<%d>(float* d, uint64_t da, uint64_t db) {\n"
                   "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"
                   "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n\"\n"
                   "    : %s\n    : \"l\"(da), \"l\"(db));\n}\n" % (n, n, regs, nr, nr + 1, outs))
    return "".join(out)


def build(name="hot_p2_lab", verbose=True):
    """torch extension with p2run / p2attrs. verbose=True so ptxas advisories (C75xx, spills) reach the log."""
    from torch.utils.cpp_extension import load_inline
    src = cuda_source()
    assert src == CUDA_SRC.replace("@WGMMA@", _gate_wgmma()), "_gate_wgmma out of date"
    tag = hashlib.sha1((CPP_SRC + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC, cuda_sources=src, functions=["p2run", "p2attrs"],
                      extra_cuda_cflags=CFLAGS + ["-Xptxas=-v"], verbose=verbose)
    return mod, time.time() - t0


# ---------------------------------------------------------------------------------------------------------------
# host side (torch imported lazily: this module also loads without torch for offline planning tests)
# ---------------------------------------------------------------------------------------------------------------

def repack_matrix(D, w, sf):
    """Lossless repack of one expert matrix (w uint8 [R, K/2] E2M1 low nibble first, sf e4m3 [R, K/16]) into P2 stages
    [R/256][K/64][9216]: 8192 B of BITPOS3 code words [kc 8][row 256] (8 consecutive k per word, so deq4 yields
    natural k order) + 1024 B of K2A-re-encoded scale bytes [row 256][group 4]. Same codes, scales and sign/zero
    handling as k2a_v37.prepack_expert."""
    import torch
    R, K = w.shape[0], w.shape[1] * 2
    sfb = sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf
    bp, neg, zero, bad = D.reencode_scales(sfb)
    if bad:
        raise ValueError("NaN block scale")
    codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(R, K)
    if bool(neg.any()) or bool(zero.any()):
        cb = codes.view(R, K // 16, 16)
        cb = torch.where(neg.unsqueeze(-1), cb ^ 8, cb)
        cb = torch.where(zero.unsqueeze(-1), torch.zeros_like(cb), cb)
        codes = cb.reshape(R, K)
    nrb, nks = R // 256, K // 64
    words = D.pack_words(codes.view(nrb, 256, nks, 8, 8), D.BITPOS3)          # int32 [rb][r][ks][kc]
    words = words.permute(0, 2, 3, 1).contiguous()                             # [rb][ks][kc][r]
    sc = bp.view(nrb, 256, nks, 4).permute(0, 2, 1, 3).contiguous()           # [rb][ks][r][g]
    return torch.cat((words.view(torch.uint8).reshape(nrb, nks, 8192), sc.reshape(nrb, nks, 1024)), dim=-1).contiguous()


def stage_cost(n):
    """LPT cost of one 64-k stage (clocks): max(ingest at ~18.6 B/clk/SM, MMA at 60% of 2048 FMA/clk)."""
    return max(495 + 6.9 * n, 13.3 * n)


def plan(counts, tok_lists, G, only_hot=False, hot_t=8):
    """Passes (expert, <= 128 tokens, N class) and LPT item lists. counts[e] = tokens of local expert e, tok_lists[e] =
    their token indices. Returns a dict of python lists (the bench turns them into tensors)."""
    import heapq
    passes = []   # (e, n, N, tokens)
    for e, c in enumerate(counts):
        if c <= 0 or (only_hot and c <= hot_t):
            continue
        toks = tok_lists[e]
        for j in range(0, c, 128):
            sub = toks[j:j + 128]
            n = len(sub)
            N = next(x for x in NCLS if x >= n)
            passes.append((e, n, N, sub))
    out = {"passes": passes}
    for nm, nrb in (("13", 32), ("2", 16)):
        its = sorted(((stage_cost(N) * 64, p, rb) for p, (_, _, N, _) in enumerate(passes) for rb in range(nrb)),
                     reverse=True)
        heap = [(0.0, c) for c in range(G)]
        heapq.heapify(heap)
        lists = [[] for _ in range(G)]
        for cost, p, rb in its:
            load, c = heapq.heappop(heap)
            lists[c].append((p << 8) | rb)
            heapq.heappush(heap, (load + cost, c))
        items, cbeg = [], [0]
        for c in range(G):
            items += lists[c]
            cbeg.append(len(items))
        out["items" + nm], out["cbeg" + nm] = items, cbeg
        out["imbalance" + nm] = max(h[0] for h in heap) / (sum(h[0] for h in heap) / G) if passes else 1.0
    tokoff, ptok, coff, xoff, yoff13, yoff2 = [], [], [0], [], [], []
    xo = y13 = y2 = 0
    for e, n, N, sub in passes:
        tokoff.append(len(ptok))
        ptok += list(sub)
        coff.append(coff[-1] + N * 512)
        xoff.append(xo)
        xo += N * 128 * 64
        yoff13.append(y13)
        y13 += 8192 * N
        yoff2.append(y2)
        y2 += 4096 * N
    out.update(tokoff=tokoff, ptok=ptok or [0], coff=coff, xoff=xoff, yoff13=yoff13, yoff2=yoff2,
               pe=[p[0] for p in passes], pn=[p[2] for p in passes], pc=[p[1] for p in passes],
               xbytes=max(xo, 16), y13n=max(y13, 1), y2n=max(y2, 1))
    return out


def unpack_x(xp_bytes, off, N):
    """B tile bytes of one pass ([64 ks][N/8][8 kc][8 rows][8 bf16]) -> bf16 [N, 4096]."""
    import torch
    t = xp_bytes[off:off + N * 128 * 64].view(torch.bfloat16).view(64, N // 8, 8, 8, 8)   # ks, n8, kc, row, i
    return t.permute(1, 3, 0, 2, 4).reshape(N, 4096)


if __name__ == "__main__":   # write the generated CUDA source for review / an offline nvcc
    import pathlib
    import sys
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "hot_p2_lab_gen.cu")
    out.write_text(cuda_source(), encoding="utf-8")
    out.with_suffix(".cpp").write_text(CPP_SRC, encoding="utf-8")
    print(f"wrote {out}")

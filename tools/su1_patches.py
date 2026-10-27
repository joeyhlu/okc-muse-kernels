"""su1_patches.py: SU1 = staged small GEMV at T <= 8 (explorer session, 2026-10-26).

Idea: at T <= 8 the small dependent projections stream their weights only AFTER their predecessor ends, while that predecessor
leaves HBM (and most SMs) idle: the unpooler slice (1024 x 4096 BF16, 8.4 MB) waits for K2A's K3 (post-expert norm, a few CTAs
that read L2 only), and the o-projection (8192 x 1024, 16.8 MB) waits for KS1 (B = 1 sliding attention, 34 / 136 CTAs, 0.5 MB of
KV). v34's fresh nsys trace (box_out/fresh_v34_profile, shapes 0 / 6): unpooler 5.0 us (R1B, T = 1) / 6.9 us (P1, T = 8) after K3
ends, every byte on the critical path (K3 triggers its dependents only at exit).

SU1 kernel: 128 CTAs x 256 threads; CTA c owns R output rows (R = 8: unpooler slice, K = 4096, 64 KB; R = 64: o, K = 1024, 128 KB)
of a weight copy repacked in prepare (su1_pack: lossless permutation into mma.sync m16n8k16 A-fragment order, so a warp's LDS.128
reads 512 contiguous bytes). Before griddepcontrol.wait thread 0 stages the CTA's whole block into smem with cp.async.bulk (4 or 8
x 16 KB, one mbarrier with expect_tx; prepare-time data only, the pointer is a per-call argument). After the wait: x fragments
from L2 (ld.global.cg), bounded mbarrier wait (err flag, no trap), K / 8 per warp of mma.sync (BF16 in, FP32 accumulate), a fixed-
order cross-warp sum in smem, BF16 out (row stride ldo). Deterministic. K3 early trigger: K2A v3.7's k3_kernel executes
griddepcontrol.launch_dependents at entry when probe bit 25 is set (set by the layer only for calls whose unpooler is SU1), so the
SU1 grid can take SMs as K2's CTAs drain and stage during K2's tail + K3.

Layer glue (all read at call time, so layer_ab2 arms switch them without a new prepare): SU1_ENABLE, SU1_ROUTE {(proj, bucket)},
SU1_K3_TRIG, SU1_TRIG, SU1_PDL. The o route is taken only when the attention ran on KS1 (its PDL trigger right after its wait lets
SU1 launch during the attention). SU1 has its own small load_inline built last in prepare (a failure only turns SU1 off). Prepare
self-test per routed
projection at T = 1, 2, 5, 8: random x vs FP64 (< 4e-3) and torch.mm (< 1e-2), NaN-prefilled output written exactly in its
columns, repeat bit-identical, err flag clear; a failure turns that projection off (today's path).

    python -I tools/su1_patches.py [k3_v34.py] [out.py]
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v34.py"
OUT = ROOT / "kernels/lab/k3_v34_su1.py"

LAYER_PY = r'''# ---------------------------------------------------------------- SU1 (tools/su1_patches.py): staged small GEMV, T <= 8
SU1_ENABLE = True    # master switch (read at call time; False at prepare also skips the SU1 fold, packs and self-tests)
SU1_ROUTE = {("unpooler", 1): True, ("unpooler", 8): True, ("o", 1): False, ("o", 8): False, ("o", 16): False, ("o", 32): False}   # (projection, bucket) -> routed (read at call time)
SU1_GEOM = {"unpooler": (4096, 8, 1024), "o": (1024, 64, 8192)}   # projection -> (K, rows per CTA, rows used): 128 CTAs each
SU1_K3_TRIG = True   # K2A v3.7's K3 triggers its dependents at entry on calls whose unpooler is SU1 (probe bit 25; read at call time)
SU1_TRIG = 1         # SU1's own griddepcontrol.launch_dependents: 0 implicit at exit, 1 right after its wait, 2 at entry (read at call time)
SU1_LATE = {}        # lab: projection -> True stages the weights only after the PDL wait (no overlap; read at call time)
SU1_PDL = True       # programmatic stream serialization (the weight staging overlaps the predecessor; read at call time)
SU1_SMEM = {}        # projection -> dynamic smem bytes requested (> the block only to cap CTAs per SM; read at call time)
SU1_PROF = None      # lab only: int64 CUDA tensor [>= 8 x 128] -> per-CTA %globaltimer stamps of the next calls
SU1_TOL = (4e-3, 1e-2)   # self-test rel-L2: vs FP64, vs torch.mm
SU1_PREPARE_BY_S = 250.0   # no SU1 build starts after this many seconds of prepare (limit 300 s; cold build ~30-40 s)
SU1_SELF_TEST_T = {"unpooler": (1, 2, 5, 8), "o": (1, 2, 5, 8, 13, 16, 29, 32)}
SU1_MAX_T = {"unpooler": 8, "o": 32}   # the unpooler slice exists only up to 8 tokens

SU1_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void su1_run(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t T, int64_t K, int64_t R,
             int64_t ldo, int64_t pdl, int64_t trig, torch::Tensor prof, int64_t smem);
std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T);
"""

SU1_CUDA_SRC = r"""
// ============================================================================================ SU1 (tools/su1_patches.py)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define SU1_ARCH_OK 1
#else
#define SU1_ARCH_OK 0
#endif

namespace su1 {
constexpr int NTHR = 256;   // 8 warps
constexpr int CH = 16384;   // bytes per cp.async.bulk

__device__ __forceinline__ uint32_t sm_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// out[n * ldo + R * blockIdx.x + r] = bf16(sum_k W[R * blockIdx.x + r][k] * x[n][k]) for n < T <= 8 NT, r < R.
// wp: su1_pack's copy: CTA c's block is R * K * 2 contiguous bytes in A-fragment order (R = 8: [K/32][lane] x {a0, a2 of k-tile 2kp,
// a0, a2 of 2kp + 1}, rows 8..15 of the m16 tile are zero; R >= 16: [R/16][K/32][2][lane] x {a0, a1, a2, a3}).
__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}

// prof (lab only, nullptr in the layer): per CTA [entry, after the PDL wait, weights landed, done, smid] in %globaltimer ns.
template <int K, int R, int NT>
__global__ void __launch_bounds__(NTHR) su1_kernel(const uint8_t* __restrict__ wp, const __nv_bfloat16* __restrict__ x, int T,
                                                   __nv_bfloat16* __restrict__ out, long ldo, int* __restrict__ err, int trig,
                                                   long long* __restrict__ prof) {
#if SU1_ARCH_OK
  if (prof != nullptr && threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
    prof[8 * blockIdx.x + 0] = gtimer();
    prof[8 * blockIdx.x + 4] = (long long)smid;
  }
  constexpr int KT = K / 16;                  // k-tiles
  constexpr int KTW = KT / 8;                 // k-tiles per warp
  constexpr int KP = K / 32;                  // k-pairs
  constexpr int MT = (R >= 16) ? R / 16 : 1;  // m-tiles
  constexpr int NTOK = 8 * NT;                // token columns (n8 tiles)
  constexpr int BYTES = R * K * 2;
  static_assert(BYTES % CH == 0 && KTW % 2 == 0 && (R == 8 || R % 16 == 0) && 8 * R * NTOK * 4 <= BYTES, "su1 geometry");
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) unsigned long long bar;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint32_t sb = sm_u32(&bar);
  const int tmode = trig & 3;           // dependents: 0 implicit at exit, 1 right after the wait, 2 at entry
  const bool late = (trig >> 4) & 1;    // lab: stage the weights only after the wait (no overlap with the predecessor)
  if (tmode == 2) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  const uint8_t* src = wp + (size_t)blockIdx.x * BYTES;
  if (tid == 0) {   // prepare-time data only before the PDL wait: this CTA's weight block -> smem
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(sb) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(sb), "r"(BYTES) : "memory");
    if (!late) {
#pragma unroll 1
      for (int o = 0; o < BYTES; o += CH)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(sm_u32(smem + o)), "l"((unsigned long long)(src + o)), "r"(CH), "r"(sb) : "memory");
    }
  }
  __syncthreads();   // the mbarrier is initialized before any thread waits on it
  asm volatile("griddepcontrol.wait;\n" ::: "memory");   // x (the predecessor's output) is visible from here on
  if (tmode == 1) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (late && tid == 0) {
#pragma unroll 1
    for (int o = 0; o < BYTES; o += CH)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(sm_u32(smem + o)), "l"((unsigned long long)(src + o)), "r"(CH), "r"(sb) : "memory");
  }
  if (prof != nullptr && tid == 0) prof[8 * blockIdx.x + 1] = gtimer();
  // B fragments of x^T (k16 x n8): token 8 nt + (lane >> 2), k = 16 kt + 2 (lane & 3) (+ 8); tokens >= T are zero
  const int g = lane >> 2, t4 = lane & 3;
  uint32_t b[KTW][NT][2];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int tok = 8 * nt + g;
    const bool live = tok < T;
    const unsigned int* xr = reinterpret_cast<const unsigned int*>(x + (size_t)(live ? tok : 0) * K + 16 * warp * KTW + 2 * t4);
#pragma unroll
    for (int i = 0; i < KTW; ++i) {
      b[i][nt][0] = live ? __ldcg(xr + 8 * i) : 0u;
      b[i][nt][1] = live ? __ldcg(xr + 8 * i + 4) : 0u;
    }
  }
  {   // the staged weights: bounded wait (flag on timeout, never a trap)
    uint32_t done = 0;
#pragma unroll 1
    for (int it = 0; it < (1 << 20) && !done; ++it)
      asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                   : "=r"(done) : "r"(sb), "r"(0u) : "memory");
    if (!done && lane == 0) atomicOr(err, 1);
  }
  if (prof != nullptr && tid == 0) prof[8 * blockIdx.x + 2] = gtimer();
  float c[MT][NT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) c[m][nt][0] = c[m][nt][1] = c[m][nt][2] = c[m][nt][3] = 0.f;
  const uint4* ws = reinterpret_cast<const uint4*>(smem);
  const int kp0 = warp * (KTW / 2);
#pragma unroll
  for (int i = 0; i < KTW / 2; ++i) {
    if constexpr (R == 8) {
      const uint4 a = ws[(kp0 + i) * 32 + lane];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        mma16816(c[0][nt], a.x, 0u, a.y, 0u, b[2 * i][nt][0], b[2 * i][nt][1]);
        mma16816(c[0][nt], a.z, 0u, a.w, 0u, b[2 * i + 1][nt][0], b[2 * i + 1][nt][1]);
      }
    } else {
#pragma unroll
      for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint4 a = ws[((m * KP + kp0 + i) * 2 + h) * 32 + lane];
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) mma16816(c[m][nt], a.x, a.y, a.z, a.w, b[2 * i + h][nt][0], b[2 * i + h][nt][1]);
        }
      }
    }
  }
  __syncthreads();   // every warp is done with the staged weights: the block becomes the [8 warps][R][NTOK] reduction scratch
  float* red = reinterpret_cast<float*>(smem);
  // C fragment: c[m][nt][0..1] = rows 16 m + g, tokens 8 nt + 2 t4 (+1); c[m][nt][2..3] = rows 16 m + g + 8
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      float* q = red + (warp * R + 16 * m + g) * NTOK + 8 * nt + 2 * t4;
      q[0] = c[m][nt][0];
      q[1] = c[m][nt][1];
      if constexpr (R >= 16) {
        q[8 * NTOK] = c[m][nt][2];
        q[8 * NTOK + 1] = c[m][nt][3];
      }
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int o = tid; o < R * NTOK; o += NTHR) {
    const int r = o / NTOK, n = o % NTOK;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w * R * NTOK + o];   // fixed order: deterministic
    if (n < T) out[(size_t)n * ldo + (size_t)R * blockIdx.x + r] = __float2bfloat16(s);
  }
  if (prof != nullptr) {
    __syncthreads();
    if (tid == 0) prof[8 * blockIdx.x + 3] = gtimer();
  }
#endif
}

// smem: dynamic shared memory requested (>= R K 2; more only to cap CTAs per SM)
template <int K, int R, int NT>
static void launch(const uint8_t* wp, const __nv_bfloat16* x, int T, __nv_bfloat16* out, long ldo, int* err, int ncta, bool pdl,
                   int trig, long long* prof, int smem, cudaStream_t st) {
  constexpr int BYTES = R * K * 2;
  const int dyn = smem > BYTES ? smem : BYTES;
  static int attr = 0;
  if (attr < dyn) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(su1_kernel<K, R, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
    attr = dyn;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ncta, 1, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = dyn;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, su1_kernel<K, R, NT>, wp, x, T, out, ldo, err, trig, prof));
}

template <int K, int R, int NT>
static std::vector<int64_t> info_t() {
  constexpr int BYTES = R * K * 2;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, su1_kernel<K, R, NT>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(su1_kernel<K, R, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, su1_kernel<K, R, NT>, NTHR, BYTES));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + BYTES), (int64_t)occ};
}
}  // namespace su1

static int su1_nt(int64_t T) { return T <= 8 ? 1 : (T <= 16 ? 2 : 4); }

void su1_run(torch::Tensor wp, torch::Tensor x, torch::Tensor out, torch::Tensor err, int64_t T, int64_t K, int64_t R,
             int64_t ldo, int64_t pdl, int64_t trig, torch::Tensor prof, int64_t smem) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous() && x.size(1) == K, "su1: x");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(1) == 1, "su1: out");
  TORCH_CHECK(wp.is_contiguous() && err.scalar_type() == at::kInt, "su1: wp / err");
  TORCH_CHECK(T >= 1 && T <= 32 && x.size(0) >= T, "su1: 1 <= T <= 32");
  TORCH_CHECK(((uintptr_t)wp.data_ptr() % 16) == 0 && ((uintptr_t)x.data_ptr() % 16) == 0, "su1: 16 B alignment");
  const int64_t blk = R * K * 2;
  const int64_t nbytes = wp.numel() * wp.element_size();
  TORCH_CHECK(nbytes % blk == 0, "su1: wp size");
  const int ncta = (int)(nbytes / blk);
  TORCH_CHECK(out.size(0) >= T && out.size(1) >= (int64_t)ncta * R && ldo >= (int64_t)ncta * R, "su1: out shape");
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* w = reinterpret_cast<const uint8_t*>(wp.data_ptr());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  int* ep = err.data_ptr<int>();
  long long* pp = prof.numel() >= 8 * (int64_t)ncta ? reinterpret_cast<long long*>(prof.data_ptr()) : nullptr;
  const int nt = su1_nt(T);
  if (K == 4096 && R == 8 && nt == 1) su1::launch<4096, 8, 1>(w, xp, (int)T, op, (long)ldo, ep, ncta, pdl != 0, (int)trig, pp, (int)smem, st);
  else if (K == 1024 && R == 64 && nt == 1) su1::launch<1024, 64, 1>(w, xp, (int)T, op, (long)ldo, ep, ncta, pdl != 0, (int)trig, pp, (int)smem, st);
  else if (K == 1024 && R == 64 && nt == 2) su1::launch<1024, 64, 2>(w, xp, (int)T, op, (long)ldo, ep, ncta, pdl != 0, (int)trig, pp, (int)smem, st);
  else if (K == 1024 && R == 64 && nt == 4) su1::launch<1024, 64, 4>(w, xp, (int)T, op, (long)ldo, ep, ncta, pdl != 0, (int)trig, pp, (int)smem, st);
  else TORCH_CHECK(false, "su1: (K, R, T) not built");
}

std::vector<int64_t> su1_info(int64_t K, int64_t R, int64_t T) {
  const int nt = su1_nt(T);
  if (K == 4096 && R == 8 && nt == 1) return su1::info_t<4096, 8, 1>();
  if (K == 1024 && R == 64 && nt == 1) return su1::info_t<1024, 64, 1>();
  if (K == 1024 && R == 64 && nt == 2) return su1::info_t<1024, 64, 2>();
  if (K == 1024 && R == 64 && nt == 4) return su1::info_t<1024, 64, 4>();
  TORCH_CHECK(false, "su1: (K, R, T) not built");
  return {};
}
"""

_SU1_FLAG = {"k3trig": False}   # set by kernel_fn around the expert call: _k2a33_call adds probe bit 25 (K3 early trigger)
_SU1_SEEN = set()   # (projection, T) already logged once


_SU1 = {"mod": None, "build_s": 0.0}


def _su1_build():
    """SU1's own small load_inline (cached), built last in prepare so a failure only turns SU1 off. Prepare-time only."""
    if _SU1["mod"] is not None:
        return _SU1["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((SU1_CPP_SRC + SU1_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    _SU1["mod"] = load_inline(name=f"su1_{tag}", cpp_sources=SU1_CPP_SRC, cuda_sources=SU1_CUDA_SRC, functions=["su1_run", "su1_info"],
                              extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _SU1["build_s"] = time.time() - t0
    return _SU1["mod"]


def su1_pack(w, R):
    """Lossless permutation of BF16 W [N, K] into SU1's per-CTA A-fragment order (N % R == 0, K % 32 == 0). Prepare-time only."""
    n, k = w.shape
    c, kp = n // R, k // 32
    w = w.contiguous()
    if R == 8:   # W[c, g, kp, h16, hi8, t, pair] -> [c, kp, g, t, h16, hi8, pair]
        p = w.view(c, 8, kp, 2, 2, 4, 2).permute(0, 2, 1, 5, 3, 4, 6)
    else:        # W[c, m, hr, g, kp, h16, hi8, t, pair] -> [c, m, kp, h16, g, t, hi8, hr, pair]
        p = w.view(c, R // 16, 2, 8, kp, 2, 2, 4, 2).permute(0, 1, 4, 5, 3, 7, 6, 2, 8)
    return p.contiguous().view(-1)


def _su1_weight(state, proj):
    K, R, rows = SU1_GEOM[proj]
    w = state["dense"]["unpooler" if proj == "unpooler" else "o"]
    return w[:rows] if proj == "unpooler" else w


def _su1_live(state, proj, tokens):
    st = state.get("su1")
    return bool(SU1_ENABLE and st is not None and proj in st["w"] and 1 <= tokens <= SU1_MAX_T[proj]
                and SU1_ROUTE.get((proj, _k2a_bucket_of(tokens))))


def _su1(state, proj, x, tokens):
    """SU1 for (projection, T <= SU1_MAX_T) when routed and self-tested, else None -> the caller keeps today's path. unpooler: this GPU's
    1024 columns straight into the persistent pre-zeroed [8, 8192] buffer (row stride 8192, columns >= 1024 stay zero); o: a new
    [T, 8192] output. Shape-only branches (graph safe); a host exception disables SU1 for the run (once, with a line)."""
    if not _su1_live(state, proj, tokens):
        return None
    st = state["su1"]
    K, R, rows = SU1_GEOM[proj]
    if not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[0] == tokens and x.shape[1] == K and x.is_contiguous()
            and x.data_ptr() % 16 == 0):
        return None
    try:
        if proj == "unpooler":
            zb = state.get("v12_zbuf")
            if not (V12_ZBUF and zb is not None and R1_UNPOOL_INTO_ZEROS and tokens <= SLICED_UNPOOL_MAX_TOKENS):
                return None
            out = zb[:tokens]
        else:
            out = torch.empty((tokens, rows), dtype=torch.bfloat16, device=x.device)
        st["mod"].su1_run(st["w"][proj], x, out, st["err"], tokens, K, R, HIDDEN, 1 if SU1_PDL else 0,
                          int(SU1_TRIG) | (16 if SU1_LATE.get(proj) else 0),
                          SU1_PROF if SU1_PROF is not None else st["noprof"], int(SU1_SMEM.get(proj, 0)))
        if (proj, tokens) not in _SU1_SEEN:
            _SU1_SEEN.add((proj, tokens))
            print(f"[su1] first call {proj} T={tokens} (k3trig {_SU1_FLAG['k3trig'] or SU1_K3_TRIG})", flush=True)
        return out
    except Exception as ex:  # noqa: BLE001
        _k3_disable(state, "su1", None, "SU1", ex)
        return None


def _su1_selftest(st, proj, w, gen):
    """(ok, message) for one projection at T in SU1_SELF_TEST_T: random x vs FP64 and torch.mm, NaN-prefilled [T, 8192] output
    written exactly in its columns, repeat bit-identical, err flag clear. Prepare-time only."""
    K, R, rows = SU1_GEOM[proj]
    dev = w.device
    ok, parts = True, []
    for t in SU1_SELF_TEST_T[proj]:
        x = torch.randn((t, K), generator=gen, device=dev).to(torch.bfloat16)
        outs = []
        for _ in range(2):
            out = torch.full((t, HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev)
            st["mod"].su1_run(st["w"][proj], x, out, st["err"], t, K, R, HIDDEN, 1, 1, st["noprof"], 0)
            outs.append(out)
        ref64 = x.double() @ w.double().t()
        today = torch.mm(x, w.t())
        torch.cuda.synchronize()
        got = outs[0][:, :rows]
        fin = bool(torch.isfinite(got).all())
        rest = bool(torch.isnan(outs[0][:, rows:]).all()) if rows < HIDDEN else True
        e64 = float((got.double() - ref64).norm() / ref64.norm()) if fin else float("inf")
        etd = float((got.float() - today.float()).norm() / today.float().norm()) if fin else float("inf")
        same = bool(torch.equal(outs[0][:, :rows], outs[1][:, :rows]))
        errf = int(st["err"][0])
        good = fin and rest and same and errf == 0 and e64 < SU1_TOL[0] and etd < SU1_TOL[1]
        ok &= good
        parts.append(f"T={t}: fp64 {e64:.2e} mm {etd:.2e} fin {fin} cols {rest} rep {same} err {errf}")
        st["err"].zero_()
    return ok, "; ".join(parts)


def _su1_prepare(state, t0):
    """SU1 build + packs + per-projection self-tests (state["su1"] stays None -> today's paths when off, past SU1_PREPARE_BY_S, the
    build fails or nothing passes)."""
    state["su1"] = None
    if not SU1_ENABLE:
        return None
    if time.time() - t0 > SU1_PREPARE_BY_S:
        print(f"[su1] skipped: prepare already took {time.time() - t0:.1f}s (> {SU1_PREPARE_BY_S}s): today's paths", flush=True)
        return None
    t1 = time.time()
    try:
        ks = _su1_build()
    except Exception as ex:  # noqa: BLE001
        print(f"[su1] build FAILED -> today's paths: {repr(ex)[-3000:]}", flush=True)
        return None
    dev = state["dense"]["o"].device
    st = {"mod": ks, "w": {}, "err": torch.zeros(4, dtype=torch.int32, device=dev), "noprof": torch.empty((0,), dtype=torch.int64, device=dev)}
    gen = torch.Generator(device=dev).manual_seed(5151)
    for proj in sorted({p for p, _ in SU1_ROUTE}):
        K, R, rows = SU1_GEOM[proj]
        try:
            w = _su1_weight(state, proj)
            nts = sorted({1 if t <= 8 else (9 if t <= 16 else 17) for t in SU1_SELF_TEST_T[proj]})   # one T per instantiation (NT 1 / 2 / 4)
            info = ["/".join(str(int(ks.su1_info(K, R, t)[j])) for t in nts) for j in range(4)]
            st["w"][proj] = su1_pack(w, R)
            ok, msg = _su1_selftest(st, proj, w, gen)
            print(f"[su1] self-test {proj} (K {K} R {R}: {info[0]} regs, {info[1]} B local, {info[2]} B smem, {info[3]} CTAs/SM): {msg}"
                  f" -> {'ok' if ok else 'FAIL: today path'}", flush=True)
            if not ok:
                del st["w"][proj]
        except Exception as ex:  # noqa: BLE001
            print(f"[su1] {proj}: self-test raised {ex!r} -> today's path", flush=True)
            st["w"].pop(proj, None)
    state["su1"] = st if st["w"] else None
    print(f"[su1] live {sorted(st['w'])} route {SU1_ROUTE} k3trig {SU1_K3_TRIG} (build {_SU1['build_s']:.1f}s, total "
          f"{time.time() - t1:.1f}s; prepare so far "
          f"{time.time() - t0:.1f}s)", flush=True)
    return state["su1"]


'''

FN = ("def kernel_fn(data, state):", "\n    return next_attn_input")
PREP = ("def prepare(weights):", "\n    occ = ")
K2A33 = ('K2A33_KERNEL_SRC = r"""', 'K2A33_CPP_SRC = r"""')
K2A33CALL = ("def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):", "\ndef _k2a33_reset(st):")
DISABLE = "def _k3_disable(state, key, value, piece, ex):\n"


def patches():
    return [
        ("PY: SU1 block", (DISABLE, None), DISABLE, LAYER_PY + DISABLE),
        ("K2A v3.7 K3: early trigger on probe bit 25", K2A33,
         "__global__ void __launch_bounds__(256) k3_kernel(const Params P) {\n  __shared__ int s_fc[8];\n  __shared__ float s_fr[8];\n"
         "  pdl_wait();\n",
         "__global__ void __launch_bounds__(256) k3_kernel(const Params P) {\n  __shared__ int s_fc[8];\n  __shared__ float s_fr[8];\n"
         "  if ((P.probe >> 25) & 1) pdl_launch();   // SU1: the staged unpooler may launch now (its weights do not depend on K3)\n"
         "  pdl_wait();\n"),
        ("K2A call: probe bit 25", K2A33CALL,
         "probe | ((1 << 24) if K2A47_K1_PDL else 0))",
         "probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG[\"k3trig\"] else 0))"),
        ("PREP: SU1 prepare", PREP,
         "    _dg1_prepare(state, t0)   # DG1: swap-AB wgmma projections at T = 32 / 128 / 256 (today's cuBLAS stays on failure)\n",
         "    _dg1_prepare(state, t0)   # DG1: swap-AB wgmma projections at T = 32 / 128 / 256 (today's cuBLAS stays on failure)\n"
         "    _su1_prepare(state, t0)   # SU1: staged small GEMVs at T <= 8 (today's paths stay on failure)\n"),
        ("FN: KS1 ran flag", FN,
         "    o = None\n    if _ks1_wanted(qkv, data, cfg, state):\n",
         "    o = None\n    ks1_ran = False\n    if _ks1_wanted(qkv, data, cfg, state):\n"),
        ("FN: KS1 ran set", FN,
         "            o = _ks1_run(qkv, data, state).view(tokens, hq * d)\n",
         "            o = _ks1_run(qkv, data, state).view(tokens, hq * d)\n            ks1_ran = True\n"),
        ("FN: o via SU1", FN,
         "    attn_output = _v20_p1m(state, \"o\", o)   # V20: P1M (T = 9..32, routed), else None\n",
         "    attn_output = _su1(state, \"o\", o, tokens) if ks1_ran else None   # SU1: o staged during KS1 / KS3 (routed), else None\n"
         "    if attn_output is None:\n        attn_output = _v20_p1m(state, \"o\", o)   # V20: P1M (T = 9..32, routed), else None\n"),
        ("FN: K3 trigger flag", FN,
         "    expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)\n",
         "    _SU1_FLAG[\"k3trig\"] = bool(SU1_K3_TRIG and _su1_live(state, \"unpooler\", tokens))   # SU1: K3 triggers its dependents early\n"
         "    try:\n"
         "        expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)\n"
         "    finally:\n"
         "        _SU1_FLAG[\"k3trig\"] = False\n"),
        ("FN: unpooler via SU1", FN,
         "    ffn_output = _v13_unpool(expert_output, state, tokens)   # V13: P1 into the pre-zeroed buffer (T <= 8, routed), else None\n",
         "    ffn_output = _su1(state, \"unpooler\", expert_output, tokens)   # SU1: staged unpooler slice (T <= 8, routed), else None\n"
         "    if ffn_output is None:\n"
         "        ffn_output = _v13_unpool(expert_output, state, tokens)   # V13: P1 into the pre-zeroed buffer (T <= 8, routed), else None\n"),
    ]


def _scope(text, scope):
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"su1: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    a = text.index(scope[0])
    if scope[1] is None:
        return a, len(text)
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patch_list=None):
    if "SU1_ENABLE" in text:
        raise RuntimeError("su1: layer already carries SU1")
    for what, scope, old, new in (patch_list if patch_list is not None else patches()):
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"su1 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    text = src.read_text(encoding="utf-8")
    pl = patches()
    for what, scope, old, _ in pl:
        a, b = _scope(text, scope)
        print(f"  {text[a:b].count(old)}x  {what}")
    patched = apply(text, pl)
    out.write_bytes(patched.encode("utf-8"))   # LF as read
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(pl)} patches hit once in scope; {out} ({len(patched) - len(text):+d} chars, {len(patched.encode())} B) compiles")

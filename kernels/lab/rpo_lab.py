"""rpo_lab.py: step 0 of L3 RAW-PREFETCH o (MID builder, notes/study/lead_oct29.md s3, notes/memory/gpu_requests.md R-rpo-0).
    python3 -u kernels/lab/rpo_lab.py [--layer=kernels/muse_decode_layer/k3_gc_19.py] [--out=box_out/rpo0/lab.json] [--reps=30]
Question: once the o weights are local, is the post-release o time (release -> last output) bound by GD's P12 decode
(INT pipe) + MMA, so that raw BF16 weights (lossless, 16.8 MB) prefetched before griddepcontrol.wait and consumed with
mma.sync finish >= 1.5 us earlier? Arms per T = 1 / 8 / 16 (o = [8192 out x 1024 in]):
  GD routes (the base file's GD block by AST; judge weight recipe): the ship route, XP (x one-ahead) and full-register
  prefetch geometries (ST = KU: o4 = WR 2 / 256 CTAs, w1 = WR 1 / 512 CTAs of 4 warps, o8 = WK 8 / 512 CTAs of 8 warps).
  RPO (this file): raw BF16 [N, K] (no repack: a lane's A fragments of a 16 x 64 unit are two 32-B row chunks), 103 CTAs x
  20 warps (5 row tiles x 4 K quarters, 4 units per warp), weights issued before griddepcontrol.wait into registers (WS 0)
  or lane-private smem via cp.async (WS 1), x staged once in padded smem (XS 1) or loaded per unit from L2 (XS 0); the
  same k order / unit -> accumulator map / warp reduction as GD at WK 4 KU 4 (so bitwise = GD's T = 1 / 16 ship routes).
  Floors: Z 1 = no weights (release -> x -> reduce -> store), Z 2 = weights landed in smem, no math.
Timing (bgv_lab's judge-like harness): 256 MB read flush, 300 us sleep gate, one CUDA graph replay, median over reps (arms
interleaved), net of an empty graph. Configs: standalone (sa); PDL pair after a spinning predecessor that triggers at entry:
x18_<ns> = 18 CTAs with 227 KB smem (18 SMs blocked, the d1s KS1 footprint), t1_<ns> = 1 tiny CTA (all SMs free).
exposed = pair - predecessor alone (= hop + post-release work).
"""
import ast
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
LAYER = ROOT / ARGS.get("layer", "kernels/muse_decode_layer/k3_gc_19.py")
OUT = ROOT / ARGS.get("out", "box_out/rpo0/lab.json")
REPS = int(ARGS.get("reps", "30"))
RSEED = int(ARGS.get("rseed", "8677534258059892491"))
sys.path.insert(0, str(ROOT / "tools/judge"))

RPO_CPP_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <vector>
int rpo_run_raw(int inst, const void* w, const void* x, void* out, int M, int ntiles, int ldo, int pdl, cudaStream_t st);
int rpo_info_raw(int inst, int64_t* res);
int rpo_pred_raw(int64_t ns, int excl, int* sink, cudaStream_t st);
void rpo_run(int64_t inst, torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t M, int64_t ntiles, int64_t ldo,
             int64_t pdl) {
  const int e = rpo_run_raw((int)inst, w.data_ptr(), x.data_ptr(), out.data_ptr(), (int)M, (int)ntiles, (int)ldo, (int)pdl,
                            at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "rpo_run: inst ", inst, " CUDA error ", e);
}
std::vector<int64_t> rpo_info(int64_t inst) {
  std::vector<int64_t> r(5, 0);
  const int e = rpo_info_raw((int)inst, r.data());
  TORCH_CHECK(e == 0, "rpo_info: inst ", inst, " error ", e);
  return r;
}
void rpo_pred(int64_t ns, int64_t excl, torch::Tensor sink) {
  const int e = rpo_pred_raw(ns, (int)excl, sink.data_ptr<int>(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "rpo_pred: CUDA error ", e);
}
'''

RPO_CUDA_SRC = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stddef.h>

namespace rpo {
constexpr int K = 1024;     // o: 1024 inputs
constexpr int XST = 1032;   // staged x row stride in bf16 (2064 B: rows 16 B apart mod 128 -> conflict-free v4 reads)

struct RArgs {
  const uint16_t* w;     // [N, K] bf16 bits, row-major (raw o weight)
  const uint16_t* x;     // [M, K] bf16 bits: the predecessor's output, read only after griddepcontrol.wait
  __nv_bfloat16* out;    // [M, ldo]
  int M, ntiles, ldo, trig;
};

__device__ __forceinline__ uint4 ldw(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ldx(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t comp(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
#else
__device__ __forceinline__ void pdl_trigger() {}
__device__ __forceinline__ void pdl_wait() {}
#endif

// CTA = WR row tiles (16 rows) x WK K-quarters (KU units of 64 K per warp); one CTA per SM (launch bound 1).
// Unit (row tile rt, K block kb): lane (g, t) holds rows rt*16+g / +8, k [kb + 16t, kb + 16t + 16) = P[0..1] / P[2..3];
// MMA step q uses pairs 2q / 2q+1 of each chunk (GD's k order: a[4q+i] = row g + 8 (i & 1), k 16t + 4q + 2 (i >> 1)).
template <int MT, int WR, int WK, int KU, int WS, int XS, int Z>
__global__ void __launch_bounds__(WR * WK * 32, 1) rpo_kernel(const __grid_constant__ RArgs A) {
  constexpr int NW = WR * WK, NT = NW * 32, NC = MT == 1 ? 2 : 1, NO = WR * MT * 128;
  constexpr int XB = XS ? 1 : 2;
  static_assert(WK * KU * 64 == K, "a CTA covers the full K");
  extern __shared__ __align__(16) uint8_t dsm[];
  __shared__ float red[NW][MT * 4][32];
  if (A.trig) pdl_trigger();
  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;
  const int wr = warp / WK, wk = warp % WK, g = lane >> 2, t = lane & 3;
  const int rt0 = (int)blockIdx.x * WR + wr;
  const int rt = rt0 < A.ntiles ? rt0 : A.ntiles - 1;
  const int kb = wk * (KU * 64) + 16 * t;
  const uint16_t* w0 = A.w + (size_t)(rt * 16 + g) * K + kb;
  const uint16_t* w1 = w0 + 8 * K;
  uint4* wsm = reinterpret_cast<uint4*>(dsm) + (size_t)warp * (KU * 4 * 32) + lane;   // [warp][u][p][lane] x 16 B
  uint4 R[WS ? 1 : KU][4];
  if constexpr (Z != 1) {
#pragma unroll
    for (int u = 0; u < KU; ++u) {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint16_t* src = (p < 2 ? w0 : w1) + u * 64 + (p & 1) * 8;
        if constexpr (WS) cp16(smem_u32(wsm + (u * 4 + p) * 32), src);
        else R[u][p] = ldw(src);
      }
    }
    if constexpr (WS) asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  pdl_wait();
  uint16_t* xs = reinterpret_cast<uint16_t*>(dsm + (WS ? NW * KU * 4 * 512 : 0));
  if constexpr (XS) {
    for (int i = (int)threadIdx.x; i < A.M * (K / 8); i += NT) {
      const int r = i / (K / 8), c = i % (K / 8);
      *reinterpret_cast<uint4*>(xs + r * XST + c * 8) = ldx(A.x + (size_t)r * K + c * 8);
    }
    __syncthreads();
  }
  int xr[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int tok = mt * 8 + g;
    xr[mt] = tok < A.M ? tok : A.M - 1;
  }
  auto loadx = [&](int u, uint4 (&X)[MT][2]) {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if constexpr (XS) {
        const uint4* p = reinterpret_cast<const uint4*>(xs + xr[mt] * XST + kb + u * 64);
        X[mt][0] = p[0];
        X[mt][1] = p[1];
      } else {
        const uint16_t* p = A.x + (size_t)xr[mt] * K + kb + u * 64;
        X[mt][0] = ldx(p);
        X[mt][1] = ldx(p + 8);
      }
    }
  };
  float acc[NC][MT][4];
#pragma unroll
  for (int c = 0; c < NC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[c][mt][i] = 0.f;
  if constexpr (WS && Z != 1) asm volatile("cp.async.wait_all;\n" ::: "memory");   // the lane's own copies only
  uint4 X[XB][MT][2];
  loadx(0, X[0]);
#pragma unroll
  for (int u = 0; u < KU; ++u) {
    const int xb = XB == 2 ? (u & 1) : 0;
    if constexpr (XB == 2) {
      if (u + 1 < KU) loadx(u + 1, X[(u + 1) & (XB - 1)]);
    } else {
      if (u > 0) loadx(u, X[0]);
    }
    uint4 P[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      if constexpr (Z == 1) P[p] = make_uint4(0u, 0u, 0u, 0u);
      else if constexpr (WS) P[p] = wsm[(u * 4 + p) * 32];
      else P[p] = R[u][p];
    }
    if constexpr (Z == 0) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int hi = q >> 1, c0 = 2 * (q & 1);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          mma16816(acc[q % NC][mt], comp(P[hi], c0), comp(P[2 + hi], c0), comp(P[hi], c0 + 1), comp(P[2 + hi], c0 + 1),
                   comp(X[xb][mt][hi], c0), comp(X[xb][mt][hi], c0 + 1));
      }
    } else {   // floors: keep the x (and weight) loads live, no math
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
        acc[0][mt][0] += __uint_as_float((X[xb][mt][0].x ^ X[xb][mt][1].w ^ P[0].x ^ P[3].w) & 0x007fffffu);
    }
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float v = acc[0][mt][i];
      if constexpr (NC == 2) v += acc[1][mt][i];
      red[warp][mt * 4 + i][lane] = v;
    }
  __syncthreads();
  for (int i = (int)threadIdx.x; i < NO; i += NT) {
    const int l = i & 31, c = (i >> 5) & 3, r = i >> 7, mt = r % MT, w2 = r / MT;
    float v = 0.f;
#pragma unroll
    for (int k2 = 0; k2 < WK; ++k2) v += red[w2 * WK + k2][mt * 4 + c][l];
    const int gg = l >> 2, tt = l & 3;
    const int rtile = (int)blockIdx.x * WR + w2;
    const int row = rtile * 16 + gg + 8 * (c >> 1);
    const int tok = mt * 8 + 2 * tt + (c & 1);
    if (rtile < A.ntiles && tok < A.M) A.out[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
  }
}

template <int MT, int WR, int WK, int KU, int WS, int XS, int Z>
constexpr int dyn_bytes(int M) { return (WS ? WR * WK * KU * 4 * 512 : 0) + (XS ? M * XST * 2 : 0); }

template <int MT, int WR, int WK, int KU, int WS, int XS, int Z>
static int set_smem() {
  static int done = 0;
  if (done) return 0;
  const int dyn = dyn_bytes<MT, WR, WK, KU, WS, XS, Z>(16);
  // always set: static red[] (10-20 KB) + dynamic can pass the 48 KB default even when dyn alone does not (R-rpo-0 try 1)
  const int e = (int)cudaFuncSetAttribute(rpo_kernel<MT, WR, WK, KU, WS, XS, Z>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  if (!e) done = 1;
  else cudaGetLastError();
  return e;
}

template <int MT, int WR, int WK, int KU, int WS, int XS, int Z>
static int launch(const RArgs& A, int pdl, cudaStream_t st) {
  int e = set_smem<MT, WR, WK, KU, WS, XS, Z>();
  if (e) return e;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)((A.ntiles + WR - 1) / WR), 1, 1);
  cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);
  cfg.dynamicSmemBytes = (size_t)dyn_bytes<MT, WR, WK, KU, WS, XS, Z>(A.M);
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  unsigned na = 0;
  if (pdl & 1) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    na = 1;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  e = (int)cudaLaunchKernelEx(&cfg, rpo_kernel<MT, WR, WK, KU, WS, XS, Z>, A);
  if (e) cudaGetLastError();   // clear it: a failed arm must not poison torch's next call
  return e;
}

template <int MT, int WR, int WK, int KU, int WS, int XS, int Z>
static int info(int64_t* res) {
  auto fn = rpo_kernel<MT, WR, WK, KU, WS, XS, Z>;
  cudaFuncAttributes fa;
  int e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  e = set_smem<MT, WR, WK, KU, WS, XS, Z>();
  if (e) return e;
  const int dyn = dyn_bytes<MT, WR, WK, KU, WS, XS, Z>(16);
  if (dyn + (int)fa.sharedSizeBytes > 232448) return 2;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, dyn);
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes; res[4] = dyn;
  return e;
}

// predecessor: spins ns after launch_dependents at entry. excl > 0: excl CTAs with 227 KB dynamic smem (one per SM, nothing
// else fits beside it); excl == 0: one tiny CTA (all other SMs free).
__global__ void __launch_bounds__(32) rpo_pred_k(long long ns, int* sink) {
  pdl_trigger();
  const long long t0 = gtime();
  while (gtime() - t0 < ns) {}
  if (threadIdx.x == 0 && sink) sink[blockIdx.x] = 1;
}
}  // namespace rpo

#define RPO_X(ID, ...) \
  case ID: return rpo::launch<__VA_ARGS__>(A, pdl, st);
#define RPO_I(ID, ...) \
  case ID: return rpo::info<__VA_ARGS__>(res);

int rpo_run_raw(int inst, const void* w, const void* x, void* out, int M, int ntiles, int ldo, int pdl, cudaStream_t st) {
  rpo::RArgs A;
  A.w = reinterpret_cast<const uint16_t*>(w);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = reinterpret_cast<__nv_bfloat16*>(out);
  A.M = M; A.ntiles = ntiles; A.ldo = ldo; A.trig = (pdl >> 1) & 1;
  switch (inst) {
@INSTS@
    default: return -1;
  }
}
int rpo_info_raw(int inst, int64_t* res) {
  switch (inst) {
@INFOS@
    default: return -1;
  }
}
int rpo_pred_raw(int64_t ns, int excl, int* sink, cudaStream_t st) {
  static int done = 0;
  const int big = 232448;
  if (!done) {
    const int e = (int)cudaFuncSetAttribute(rpo::rpo_pred_k, cudaFuncAttributeMaxDynamicSharedMemorySize, big);
    if (e) return e;
    done = 1;
  }
  if (excl > 0) rpo::rpo_pred_k<<<excl, 32, big, st>>>((long long)ns, sink);
  else rpo::rpo_pred_k<<<1, 32, 0, st>>>((long long)ns, sink);
  return (int)cudaGetLastError();
}
'''

# RPO instances (MT, WR, WK, KU, WS, XS, Z) -> arm name per MT
RPO_INSTS = [
    ("r_reg", (1, 5, 4, 4, 0, 1, 0)),     # raw in registers, x staged in smem, 103 CTAs x 640 thr
    ("r_smem", (1, 5, 4, 4, 1, 1, 0)),    # raw in lane-private smem (160 KB / CTA)
    ("r_reg128", (1, 4, 4, 4, 0, 1, 0)),  # raw in registers, 128 CTAs x 512 thr (one wave only on >= 128 free SMs)
    ("r_reggx", (1, 5, 4, 4, 0, 0, 0)),   # raw in registers, x per unit from L2 (one-ahead)
    ("z_nox", (1, 5, 4, 4, 0, 1, 1)),     # floor: no weights: release -> x -> reduce -> store
    ("z_smem", (1, 5, 4, 4, 1, 1, 2)),    # floor: weights landed in smem, no math
    ("r_reg", (2, 5, 4, 4, 0, 1, 0)),
    ("r_smem", (2, 5, 4, 4, 1, 1, 0)),
    ("z_nox", (2, 5, 4, 4, 0, 1, 1)),
    ("z_smem", (2, 5, 4, 4, 1, 1, 2)),
]
GD_ARMS = {
    1: [("ship", None), ("shipx", (1, 2, 4, 4, 2, 1, 1)), ("o4", (1, 2, 4, 4, 4, 1, 0)), ("w1", (1, 1, 4, 4, 4, 1, 0)),
        ("w1x", (1, 1, 4, 4, 4, 1, 1)), ("o8", (1, 1, 8, 2, 2, 1, 0)), ("o8x", (1, 1, 8, 2, 2, 1, 1))],
    8: [("ship", None), ("shipx", (1, 1, 8, 2, 2, 1, 1)), ("w1", (1, 1, 4, 4, 4, 1, 0)), ("w1x", (1, 1, 4, 4, 4, 1, 1)),
        ("o1", (1, 2, 4, 4, 2, 1, 0))],
    16: [("ship", None), ("shipx", (2, 2, 4, 4, 2, 1, 1)), ("w1", (2, 1, 4, 4, 4, 1, 0)), ("o8", (2, 1, 8, 2, 2, 1, 0)),
         ("o8x", (2, 1, 8, 2, 2, 1, 1))],
}
# R-rpo-1: PF routes (XP & 2 = coherent pre-wait stages, tools/rpo_patches.py) when the layer's GD source has ldwp
GD_ARMS_PF = {
    1: [("p_ship", (1, 2, 4, 4, 2, 1, 2)), ("p_w1", (1, 1, 4, 4, 4, 1, 2)), ("p_o8", (1, 1, 8, 2, 2, 1, 2)),
        ("p_o4", (1, 2, 4, 4, 4, 1, 2))],
    8: [("p_ship", (1, 1, 8, 2, 2, 1, 2)), ("p_w1", (1, 1, 4, 4, 4, 1, 2))],
    16: [("p_ship", (2, 2, 4, 4, 2, 1, 2)), ("p_w1", (2, 1, 4, 4, 4, 1, 2)), ("p_o8", (2, 1, 8, 2, 2, 1, 2))],
}
CONFIGS = {1: [("x18", 2500), ("x18", 4500), ("x18", 7000), ("x18", 12000), ("t1", 12000)],
           8: [("x18", 4500), ("x18", 12000), ("t1", 12000)],
           16: [("x18", 4500), ("x18", 12000), ("t1", 12000)]}
TS = [int(t) for t in ARGS.get("ts", "1,8,16").split(",")]


def extract(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name and (name.startswith(("GD_", "_GD", "_gd")) or name in ("K2A_BUCKETS", "_k2a_bucket_of")):
            nodes.append(node)
    ns = {"torch": torch, "time": time, "hashlib": hashlib, "math": math, "HIDDEN": 8192, "D_EXPERT": 4096, "UNPOOL_SLICE": 1024,
          "SLICED_UNPOOL_MAX_TOKENS": 8, "V12_ZBUF": True, "R1_UNPOOL_INTO_ZEROS": True, "K2A_M_MAX": 4096, "N_EXPERTS": 256,
          "TOP_K": 8, "RMS_EPS": 1e-5, "_pt1_wait": lambda name: None, "DISABLED": [], "_k3_disable": lambda *a: None,
          "_v8_tail": None, "SU1_K3_TRIG_AT": set(), "_nb64": lambda T: False, "NB64_GD_B": {}, "NB64_USE": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)   # lab tool only
    return ns


def main():
    ns = extract(LAYER)
    dev = "cuda"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    N, K = 8192, 1024
    import judge as J  # noqa: E402  (tools/judge on the box)
    prob = J.load_problem(ROOT / "problems/muse_decode_layer")
    ref = prob.ref
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device=dev, router_seed=RSEED)
    W = weights[ns["GD_WEIGHT"]["o"]].contiguous()
    del weights
    torch.cuda.empty_cache()
    assert W.dtype == torch.bfloat16 and tuple(W.shape) == (N, K), (W.dtype, W.shape)
    from torch.utils.cpp_extension import load_inline
    gencode = ["-gencode=arch=compute_90a,code=sm_90a"]
    if "ldwp(" in ns["GD_CUDA_SRC"]:
        for T in TS:
            GD_ARMS[T] = GD_ARMS[T] + GD_ARMS_PF[T]
    routes = {}   # (T, name) -> rt
    for T in TS:
        for name, rt in GD_ARMS[T]:
            routes[(T, name)] = rt if rt is not None else ns["GD_ROUTE"][("o", T)]
    gkeys = []
    for rt in routes.values():
        key = ns["_gd_key"](N, K, rt)
        assert key is not None, rt
        if key not in gkeys:
            gkeys.append(key)
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in kk)) for i, kk in enumerate(gkeys))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in kk)) for i, kk in enumerate(gkeys))
    gsrc = ns["GD_CUDA_SRC"].replace("@INST@", inst).replace("@INFO@", info)
    rsrc = RPO_CUDA_SRC.replace("@INSTS@", "\n".join("    RPO_X(%d, %s)" % (i, ", ".join(map(str, v)))
                                                     for i, (_, v) in enumerate(RPO_INSTS)))
    rsrc = rsrc.replace("@INFOS@", "\n".join("    RPO_I(%d, %s)" % (i, ", ".join(map(str, v))) for i, (_, v) in enumerate(RPO_INSTS)))
    t0 = time.time()
    gmod = load_inline(name="rpogd_" + hashlib.sha1((ns["GD_CPP_SRC"] + gsrc).encode()).hexdigest()[:10],
                       cpp_sources=ns["GD_CPP_SRC"], cuda_sources=gsrc, functions=["gd_run", "gd_info"],
                       extra_cuda_cflags=["-O3"] + gencode, verbose=False)
    rmod = load_inline(name="rpo_" + hashlib.sha1((RPO_CPP_SRC + rsrc).encode()).hexdigest()[:10],
                       cpp_sources=RPO_CPP_SRC, cuda_sources=rsrc, functions=["rpo_run", "rpo_info", "rpo_pred"],
                       extra_cuda_cflags=["-O3", "-Xptxas=-v"] + gencode, verbose=False)
    print(f"[rpo] built GD ({len(gkeys)} inst) + RPO ({len(RPO_INSTS)} inst) in {time.time() - t0:.0f}s; sms {sms}", flush=True)
    ginfo, rinfo = {}, {}
    for i, kk in enumerate(gkeys):
        r = gmod.gd_info(i)
        ginfo[kk] = list(r)
        print(f"[rpo] gd key {kk}: regs {r[0]} lmem {r[1]} occ {r[2]} smem {r[3]}", flush=True)
    for i, (nm, v) in enumerate(RPO_INSTS):
        r = rmod.rpo_info(i)
        rinfo[i] = list(r)
        print(f"[rpo] rpo {i} {nm} {v}: regs {r[0]} lmem {r[1]} occ {r[2]} smem {r[3]} dyn {r[4]}", flush=True)

    stream = torch.cuda.Stream()
    flush_buf = torch.ones((256 << 20) // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((1,), dtype=torch.float32, device=dev)
    psink = torch.zeros((sms,), dtype=torch.int32, device=dev)
    gate_cyc = [int(300 * 1980)]
    with torch.cuda.stream(stream):
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            torch.cuda._sleep(gate_cyc[0])
            b.record()
            b.synchronize()
            gate_cyc[0] = int(gate_cyc[0] * 300 / max(a.elapsed_time(b) * 1e3, 1.0))
    graphs = []

    def capture(name, fn):
        with torch.cuda.stream(stream):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=stream):
                fn()
        torch.cuda.synchronize()
        graphs.append((name, gr))

    def pred_fn(kind, spin):
        return lambda: rmod.rpo_pred(spin, 18 if kind == "x18" else 0, psink)

    capture("empty", lambda: None)
    allcfg = sorted({c for T in TS for c in CONFIGS[T]})
    for kind, spin in allcfg:
        capture(f"pred/{kind}_{spin}", pred_fn(kind, spin))
    gen = torch.Generator(device=dev).manual_seed(5)
    keep, meta, packs = [W], {}, {}
    for T in TS:
        x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
        refo = x.double() @ W.double().T
        keep.append(x)
        arms = []   # (name, fn(pdl))
        outs = {}
        for name, _ in GD_ARMS[T]:
            rt = routes[(T, name)]
            key = ns["_gd_key"](N, K, rt)
            gi = gkeys.index(key)
            pkk = (rt[1], rt[2], rt[3], rt[5])
            if pkk not in packs:
                packs[pkk] = ns["_gd_pack"](W, rt[1], rt[2], rt[3], rt[5], ns["_gd_window_base"](W))
            pk = packs[pkk]
            y = torch.full((T, N), float("nan"), dtype=torch.bfloat16, device=dev)
            keep += [pk, y]

            def gd_fn(pdl, pk=pk, gi=gi, x=x, y=y, T=T):
                return lambda: gmod.gd_run(gi, pk["w"], pk["esc"], x, y, T, K, N, pk["tb"], pk["nbx"], 1, pdl)
            arms.append((f"gd_{name}", gd_fn, y, ginfo[key], rt))
        mt = 1 if T <= 8 else 2
        for i, (nm, v) in enumerate(RPO_INSTS):
            if v[0] != mt or (nm == "r_reg128" and T != 1) or (nm == "r_reggx" and T == 8):
                continue
            y = torch.full((T, N), float("nan"), dtype=torch.bfloat16, device=dev)
            keep.append(y)

            def r_fn(pdl, i=i, x=x, y=y, T=T):
                return lambda: rmod.rpo_run(i, W, x, y, T, N // 16, N, pdl)
            arms.append((nm, r_fn, y, rinfo[i], v))
        for name, fn, y, inf, geo in arms:
            full = f"{T}/{name}"
            try:
                fn(2)()
                torch.cuda.synchronize()
            except Exception as ex:
                print(f"[rpo] {full}: LAUNCH FAILED {ex!r}"[:300], flush=True)
                continue
            got = y.clone()
            fn(2)()
            torch.cuda.synchronize()
            rep = bool(torch.equal(got, y))
            rel = float((got.double() - refo).norm() / refo.norm()) if bool(torch.isfinite(got).all()) else float("nan")
            if name == "gd_ship":
                outs["ship"] = got
            vs = None
            if "ship" in outs:
                dd = (got.float() - outs["ship"].float()).abs()
                vs = (int((dd > 0).sum()), float(dd.max()))
            meta[full] = {"rel": rel, "repeat": rep, "vs_ship": vs, "info": inf, "geo": list(geo)}
            zero = name.startswith("z_")
            ok = zero or (rel < 2e-3 and rep)
            print(f"[rpo] check {full}: rel {rel:.2e} repeat {rep} vs_ship {vs} info {inf} geo {geo}" +
                  ("" if ok else " -> CHECK FAILED, not timed"), flush=True)
            if not ok:
                continue
            capture(f"{full}/sa", fn(2))
            for kind, spin in CONFIGS[T]:
                capture(f"{full}/{kind}_{spin}", lambda f=fn(3), p=pred_fn(kind, spin): (p(), f()))
    ts = {name: [] for name, _ in graphs}
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in graphs]
    t1 = time.time()
    for rep in range(REPS):
        for (name, gr), (a, b) in zip(graphs, ev):
            with torch.cuda.stream(stream):
                torch.sum(flush_buf, dim=0, out=sink)
                torch.cuda._sleep(gate_cyc[0])
                a.record()
                gr.replay()
                b.record()
            b.synchronize()
            ts[name].append(a.elapsed_time(b) * 1e3)
    empty = statistics.median(ts["empty"])
    res = {}
    for name, v in ts.items():
        s = sorted(v)
        res[name] = {"net_us": round(statistics.median(v) - empty, 3), "q25": round(s[len(s) // 4] - empty, 3),
                     "q75": round(s[(3 * len(s)) // 4] - empty, 3)}
    print(f"[rpo] timed {len(graphs)} graphs x {REPS} reps in {time.time() - t1:.0f}s; empty {empty:.2f} us", flush=True)
    for kind, spin in allcfg:
        print(f"[rpo] pred {kind}_{spin}: {res[f'pred/{kind}_{spin}']['net_us']:.2f} us", flush=True)
    summary = {}
    for T in TS:
        cfgs = CONFIGS[T]
        print(f"[rpo] T={T}: arm: sa | exposed " + " ".join(f"{k}_{s}" for k, s in cfgs), flush=True)
        for name in sorted({n.split("/")[1] for n in res if n.startswith(f"{T}/")}):
            pre = f"{T}/{name}/"
            if pre + "sa" not in res:
                continue
            row = {"sa": res[pre + "sa"]["net_us"]}
            for kind, spin in cfgs:
                c = f"{kind}_{spin}"
                row[c] = round(res[pre + c]["net_us"] - res[f"pred/{c}"]["net_us"], 3)
            summary[f"{T}/{name}"] = row
            print(f"[rpo]   {name:10s} sa {row['sa']:6.2f} | " + " ".join(f"{row[f'{k}_{s}']:6.2f}" for k, s in cfgs), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"empty_us": empty, "reps": REPS, "res": res, "summary": summary, "meta": meta, "sms": sms,
                               "gd_keys": [list(k) for k in gkeys]}, indent=1))
    print(f"[rpo] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

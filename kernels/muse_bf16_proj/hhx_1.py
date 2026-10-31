# okc: problem=muse_bf16_proj
# nonce hhx-1 (vd_1 + the layer's DG1 at m = 32 and the dpin cuBLASLt pins at m = 128 / 256 as extra search candidates; tools/gen_hhx_board.py)
"""vp_4 (= gv1-c + the R-vp-2 winners): lossless 12-bit weight-streaming skinny GEMMs for the dense projections (M <= 32),
cuBLAS (SGLang's own call, a pinned cuBLASLt algorithm or cuBLAS on a K-major weight copy) above that.

XP >= 2 (vp_2): each warp's x fragments of the next XP - 1 units are in flight in a per-warp shared-memory ring
(cp.async, 16 B per lane, no registers held). On the H200 clone (R-vp-2) it wins at m = 8 / 16 and loses at m = 32, so
only m = 8 / 16 routes use it. NR (row tiles per warp sharing x, vp_1) stays in the template with NR = 1 everywhere (R-vp-1:
slower at every shape).

Format P12 (per-matrix exponent window, lossless for every BF16 value):
  bf16 = s | e7..e0 | m6..m0. Low byte (e0 m6..m0) is stored as is. The high 7 exponent bits h7 = (w >> 8) & 127
  are coded as d = h7 - base in 3 bits (16 consecutive exponents), plus the sign: 4 bits. 12 bits per weight.
  Values outside the window are coded d = 0 and listed as escapes (per warp, <= 32): reg ^= (h7 ^ base) << (8+16h).
  Decode of 8 weights: AND, IADD, SHF, AND, IADD, 4 PRMT (exact; no scale, no rounding).
Unit = 16 rows x 64 k in mma.m16n8k16 A-fragment order with k permuted inside the unit (k = 16 t + 4 q + 2 a + b),
so lane t's B operand is 16 contiguous k of x: two 16-B loads straight from global memory.
Geometry (WR, WK, KU, S): a CTA owns WR row tiles; WK warps split each tile's K slice; each warp streams KU units;
S CTAs (one cluster) split K further and push FP32 partials into the leader over DSMEM (st.async + mbarrier).
Layout: grid-interleaved stream order [unit step][global warp][plane][lane][16 B], so at every step the whole grid
reads one contiguous front. Every route is verified in prepare(): bit-exact unpack of the packed matrix, then
outputs vs an FP64 reference (judge recipe incl. probe rows), then timed against SGLang's call with cold-L2 CUDA-graph
replays; a route is used only if it passes and wins. Every fallback is printed.
"""

import hashlib
import math
import threading
import time
from typing import TypedDict

import torch
import torch.nn.functional as F

GV_ENABLE = True
GV_AUTOTUNE = True          # time candidates in prepare() and keep the fastest that beats SGLang's call
GV_MARGIN = 0.97            # a route must be <= 0.97 x SGLang's time (net of the graph floor) to be used
GV_REPS = 25                # cold-L2 graph replays per candidate (paired with the baseline)
GV_THREADS = 6              # parallel load_inline builds
GV_PREPARE_BUDGET = 240.0   # seconds; candidates past it are skipped (loudly)
GV_FORCE_FMT = None         # test knob: keep only this format's candidates where a bucket has any
ESC_CAP = 32
MAX_GV_M = 32               # BF16 layers: GV up to 32 tokens; the router also runs token slices above that

# (layer, bucket) -> candidate routes, best guess first. A route:
#   (fmt, mt, wr, wk, ku, st, s, xp)  fmt 0 = P12 packed, 1 = RAW bf16 fragments; mt = 8-token tiles; xp = x one
#   unit ahead (1) or at use (0). The router runs on a virtual [512, 8192] matrix: per 16-row tile 8 hi + 8 lo rows.
CANDIDATES = {
    # pooler 4096 x 8192: 256 row tiles x 128 units
    ("pooler", 1): [(0, 1, 2, 16, 8, 1, 1, 0), (0, 1, 1, 8, 16, 2, 1, 1), (0, 1, 2, 8, 16, 2, 1, 0),
                    (0, 1, 1, 16, 8, 2, 1, 0), (1, 1, 2, 16, 8, 1, 1, 0), (2, 1, 2, 16, 8, 1, 1, 0),
                    (2, 1, 2, 8, 16, 2, 1, 0), (2, 1, 1, 8, 16, 2, 1, 1)],
    ("pooler", 8): [(0, 1, 2, 16, 8, 1, 1, 0), (0, 1, 1, 8, 16, 2, 1, 1), (0, 1, 2, 8, 16, 2, 1, 0),
                    (2, 1, 2, 16, 8, 1, 1, 0), (2, 1, 2, 8, 16, 2, 1, 0)],
    ("pooler", 16): [(0, 2, 1, 8, 16, 2, 1, 0), (0, 2, 2, 16, 8, 1, 1, 0), (0, 2, 2, 8, 16, 1, 1, 0)],
    ("pooler", 32): [(0, 4, 1, 8, 16, 1, 1, 0), (0, 4, 2, 8, 16, 1, 1, 0), (0, 4, 1, 16, 8, 1, 1, 0)],
    # qkv_gate 2304 x 8192: 144 row tiles x 128 units (cluster K split keeps one wave)
    ("qkv_gate", 1): [(0, 1, 1, 4, 8, 2, 4, 1), (0, 1, 1, 8, 16, 2, 1, 1), (0, 1, 1, 8, 8, 2, 2, 1),
                      (0, 1, 1, 4, 4, 2, 8, 1), (1, 1, 1, 4, 8, 2, 4, 1), (2, 1, 1, 4, 8, 2, 4, 1),
                      (2, 1, 1, 8, 8, 2, 2, 1)],
    ("qkv_gate", 8): [(0, 1, 1, 4, 8, 2, 4, 1), (0, 1, 1, 8, 8, 2, 2, 1), (0, 1, 1, 4, 4, 2, 8, 1),
                      (0, 1, 1, 8, 16, 2, 1, 1), (2, 1, 1, 4, 8, 2, 4, 1)],
    ("qkv_gate", 16): [(0, 2, 1, 4, 8, 1, 4, 0), (0, 2, 1, 4, 8, 2, 4, 0), (0, 2, 1, 8, 8, 1, 2, 0),
                       (0, 2, 1, 8, 16, 2, 1, 0)],
    ("qkv_gate", 32): [(0, 4, 1, 4, 8, 1, 4, 0), (0, 4, 1, 8, 8, 1, 2, 0), (0, 4, 1, 4, 4, 1, 8, 0),
                       (0, 4, 1, 8, 16, 1, 1, 0)],
    # o 8192 x 1024: 512 row tiles x 16 units
    ("o", 1): [(0, 1, 4, 8, 2, 2, 1, 0), (0, 1, 1, 8, 2, 2, 1, 0), (0, 1, 2, 8, 2, 2, 1, 0), (0, 1, 2, 4, 4, 2, 1, 0),
              (2, 1, 2, 4, 4, 2, 1, 0), (2, 1, 1, 8, 2, 2, 1, 0)],
    ("o", 8): [(0, 1, 1, 8, 2, 2, 1, 0), (0, 1, 4, 8, 2, 2, 1, 0), (0, 1, 2, 8, 2, 2, 1, 0), (2, 1, 1, 8, 2, 2, 1, 0)],
    ("o", 16): [(0, 2, 2, 4, 4, 2, 1, 0), (0, 2, 2, 8, 2, 2, 1, 0), (0, 2, 4, 4, 4, 2, 1, 0)],
    ("o", 32): [(0, 4, 2, 4, 4, 2, 1, 0), (0, 4, 2, 4, 4, 1, 1, 0), (0, 4, 4, 4, 4, 1, 1, 0), (0, 4, 1, 4, 4, 1, 1, 0)],
    # unpooler slice 1024 x 4096 (<= 8 tokens): 64 row tiles x 64 units; full 8192 x 4096 above 8
    ("unpooler", 1): [(0, 1, 1, 8, 2, 2, 4, 0), (0, 1, 1, 16, 2, 2, 2, 0), (0, 1, 1, 4, 4, 2, 4, 0),
                      (0, 1, 1, 4, 2, 2, 8, 0), (0, 1, 1, 16, 4, 2, 1, 0), (2, 1, 1, 16, 4, 2, 1, 0),
                      (2, 1, 1, 4, 4, 2, 4, 0)],
    ("unpooler", 8): [(0, 1, 1, 16, 2, 2, 2, 0), (0, 1, 1, 8, 2, 2, 4, 0), (0, 1, 1, 4, 2, 2, 8, 0),
                      (0, 1, 1, 16, 4, 2, 1, 0), (2, 1, 1, 16, 4, 2, 1, 0)],
    ("unpooler", 16): [(0, 2, 2, 4, 16, 2, 1, 0), (0, 2, 2, 8, 8, 2, 1, 0), (0, 2, 4, 8, 8, 1, 1, 0)],
    ("unpooler", 32): [(0, 4, 2, 4, 16, 1, 1, 0), (0, 4, 1, 4, 16, 1, 1, 0), (0, 4, 2, 8, 8, 1, 1, 0)],
    # router: virtual 512 x 8192 (hi / lo rows per tile), 32 row tiles x 128 units, FP32 out = hi + lo
    ("router", 1): [(0, 1, 1, 4, 4, 2, 8, 1), (0, 1, 1, 8, 4, 2, 4, 1), (0, 1, 1, 8, 2, 2, 8, 1),
                    (1, 1, 1, 4, 4, 2, 8, 1), (0, 1, 1, 8, 16, 2, 1, 1)],
    ("router", 8): [(0, 1, 1, 4, 4, 2, 8, 1), (0, 1, 1, 8, 4, 2, 4, 1), (0, 1, 1, 8, 2, 2, 8, 1),
                    (0, 1, 1, 8, 16, 2, 1, 1)],
    ("router", 16): [(0, 2, 1, 4, 4, 2, 8, 0), (0, 2, 1, 8, 4, 2, 4, 0), (0, 2, 1, 8, 2, 2, 8, 0),
                     (0, 2, 1, 8, 16, 2, 1, 0)],
    ("router", 32): [(0, 4, 1, 4, 4, 1, 8, 0), (0, 4, 1, 8, 4, 1, 4, 0), (0, 4, 1, 8, 2, 1, 8, 0),
                     (0, 4, 1, 8, 16, 1, 1, 0)],
    # token slices of 32 above 32 tokens (every slice re-reads the 6.3 MB pack from L2)
    ("router", 128): [(0, 4, 1, 4, 4, 1, 8, 0), (0, 4, 1, 8, 4, 1, 4, 0), (0, 4, 1, 8, 16, 1, 1, 0)],
    ("router", 256): [(0, 4, 1, 4, 4, 1, 8, 0), (0, 4, 1, 8, 4, 1, 4, 0), (0, 4, 1, 8, 16, 1, 1, 0)],
}
# vp_4 = the R-vp-2 (H200 judge clone, box_out/vp1/vp_2.log) winners only. XP >= 2 = x through a per-warp cp.async smem
# ring of XP units (smem NW x XP x MT KB per CTA): wins at m = 8 / 16 (o 4.00 / 4.29 vs 4.29 / 4.64 us, pooler m16 12.42 vs
# 12.77, unpooler m16 12.51 vs 12.96), loses at m = 32 (pooler 20.6 vs 14.9), so none at m32. NR (R-vp-1) is dropped.
# Router m256: 64-token slices (MT 8) 14.3 vs Lt 15.8. m128 / 256 BF16 layers also try torchT (cuBLAS on a K-major copy).
VP_CANDIDATES = {
    ("o", 8): [(0, 1, 1, 8, 2, 2, 1, 2)],
    ("o", 16): [(0, 2, 2, 8, 2, 2, 1, 2)],
    ("pooler", 16): [(0, 2, 1, 8, 16, 1, 1, 2), (0, 2, 1, 8, 16, 2, 1, 3)],
    ("unpooler", 16): [(0, 2, 2, 4, 16, 2, 1, 3), (0, 2, 2, 4, 16, 1, 1, 2)],
    ("router", 256): [(0, 8, 1, 8, 16, 1, 1, 0)],
}
VP_ENABLE = True
if VP_ENABLE:
    for _k, _v in VP_CANDIDATES.items():
        CANDIDATES[_k] = list(CANDIDATES.get(_k, [])) + _v
# P11 (fmt 2, 11-bit) is parked: R-dense-1 (H200 judge clone) measured it 19-87 % slower than P12 at m <= 8 (its escape
# correction tail costs more than the 4.6 % bytes it saves). The code stays for reference; no fmt-2 kernel is built.
P11_ENABLE = False
if not P11_ENABLE:
    CANDIDATES = {key: [r for r in v if r[0] != 2] for key, v in CANDIDATES.items()}
TIMED_M = {1: 1, 8: 8, 16: 16, 32: 32, 128: 128, 256: 256}   # bucket -> the judge's timed m (router 128/256: slices)
LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
          "router": (256, 8192)}
UNPOOL_SLICE = 1024
SLICED_MAX_TOKENS = 8
ROUTER_FP32_MAX_TOKENS = 8


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


# The torch-facing wrappers live in the C++ file; the .cu file includes only CUDA headers (so nvcc's host pass stays
# light: one module, compile time dominated by this small binding file, not by the kernels).
CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <vector>
int gv_run_raw(int64_t inst, const void* w, const void* esc, const void* eoff, const void* x, void* out, int M, int K,
               int ldo, uint32_t tb, int nbx, int nz, cudaStream_t st);
int gv_info_raw(int64_t inst, int64_t* res);
int gv_unpack_raw(const void* w, const void* esc, void* out, int fmt, int K, int G, int nbx, int wr, int wk, int ku,
                  uint32_t tb, int nr, cudaStream_t st);
void gv_run(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor eoff, torch::Tensor x, torch::Tensor out,
            int64_t M, int64_t K, int64_t ldo, int64_t tb, int64_t nbx, int64_t nz) {
  const int e = gv_run_raw(inst, w.data_ptr(), esc.data_ptr(), eoff.data_ptr(), x.data_ptr(), out.data_ptr(), (int)M,
                           (int)K, (int)ldo, (uint32_t)tb, (int)nbx, (int)nz, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gv_run: inst ", inst, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
std::vector<int64_t> gv_info(int64_t inst) {
  std::vector<int64_t> r(4, 0);
  const int e = gv_info_raw(inst, r.data());
  TORCH_CHECK(e == 0, "gv_info: inst ", inst, " error ", e);
  return r;
}
void gv_unpack(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t fmt, int64_t K, int64_t G, int64_t nbx,
               int64_t wr, int64_t wk, int64_t ku, int64_t tb, int64_t nr) {
  const int e = gv_unpack_raw(w.data_ptr(), esc.data_ptr(), out.data_ptr(), (int)fmt, (int)K, (int)G, (int)nbx, (int)wr,
                              (int)wk, (int)ku, (uint32_t)tb, (int)nr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gv_unpack: CUDA error ", e);
}
"""

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stddef.h>

namespace gv {

struct Args {
  const uint8_t* w;        // packed stream: unit (step s, global warp gw) at (s*G + gw) * UB bytes (planes inside)
  const uint32_t* esc;     // P12: [G * 32] escape XOR entries, sentinel 0xFFFFFFFF. P11: correction entries
  const int* eoff;         // P11: [G + 1] offsets of each warp's correction entries
  const uint16_t* x;       // [M, K] bf16 bits, row-major
  void* out;               // [M, ldo] bf16 (OUT 0) or f32 (OUT 2: router, out = hi + lo rows)
  int M, K, ldo;
  uint32_t tb;             // P12 / P11: base per byte (bytes 0/1 -> rows g of the tile, bytes 2/3 -> rows g + 8)
};

__device__ __forceinline__ uint4 ldw(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ldw2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldw1(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
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

// unit bytes: P12 3 x 512 (two low-byte planes + code plane), RAW 4 x 512 (A fragments), P11 512 + 512 + 256 + 128
template <int FMT> struct UB { static constexpr int v = FMT == 0 ? 1536 : (FMT == 1 ? 2048 : 1408); };

template <int FMT>
__device__ __forceinline__ void ldunit(const uint8_t* up, int lane, uint4 (&r)[4]) {
  if constexpr (FMT == 0) {
    r[0] = ldw(up + lane * 16); r[1] = ldw(up + 512 + lane * 16); r[2] = ldw(up + 1024 + lane * 16);
  } else if constexpr (FMT == 1) {
#pragma unroll
    for (int p = 0; p < 4; ++p) r[p] = ldw(up + 512 * p + lane * 16);
  } else {
    r[0] = ldw(up + lane * 16); r[1] = ldw(up + 512 + lane * 16);
    const uint2 c = ldw2(up + 1024 + lane * 8);
    r[2].x = c.x; r[2].y = c.y; r[2].z = ldw1(up + 1280 + lane * 4);
  }
}

// one 64-k unit (4 k-steps) -> 16 A-fragment registers a[4q + i]
template <int FMT>
__device__ __forceinline__ void decode(const uint4 (&p)[4], uint32_t tb, uint32_t (&a)[16]) {
  if constexpr (FMT == 2) {
    // P11: code words C0..C2 hold k-steps 0..2 at bit residues {0,1,3} mod 4 of each byte pattern; k-step 3 sits in
    // their free residue-2 bits and is rebuilt by three rotates + two bit-selects.
    const uint32_t L[8] = {p[0].x, p[0].y, p[0].z, p[0].w, p[1].x, p[1].y, p[1].z, p[1].w};
    uint32_t C[4];
    C[0] = p[2].x; C[1] = p[2].y; C[2] = p[2].z;
    const uint32_t r0 = __funnelshift_l(C[0], C[0], 2), r1 = __funnelshift_l(C[1], C[1], 3),
                   r2 = __funnelshift_l(C[2], C[2], 1);
    const uint32_t t01 = (r0 & 0x11111111u) | (r1 & ~0x11111111u);
    C[3] = (t01 & 0x33333333u) | (r2 & ~0x33333333u);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t h0 = (C[q] & 0x83838383u) + tb;
      const uint32_t h1 = (__funnelshift_r(C[q], C[q], 4) & 0x83838383u) + tb;
      a[4 * q + 0] = __byte_perm(L[2 * q], h0, 0x5140);
      a[4 * q + 1] = __byte_perm(L[2 * q], h0, 0x7362);
      a[4 * q + 2] = __byte_perm(L[2 * q + 1], h1, 0x5140);
      a[4 * q + 3] = __byte_perm(L[2 * q + 1], h1, 0x7362);
    }
  } else if constexpr (FMT == 0) {
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
  } else {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      a[4 * q + 0] = p[q].x; a[4 * q + 1] = p[q].y; a[4 * q + 2] = p[q].z; a[4 * q + 3] = p[q].w;
    }
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
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
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
#define GV_CLUSTER_OK 1
#else
#define GV_CLUSTER_OK 0
#endif

// out[t, n] = sum_k x[t, k] W[n, k] for the CTA's WR x NR row tiles, tokens tok0 .. tok0 + 8 MT - 1.
// grid = (row groups, S, token slices); cluster (1, S, 1) when S > 1. Warp (wr, wk) owns row tiles
// (bx WR + wr) NR + r, r < NR, and reuses every x fragment for its NR tiles. The leader's cluster buffer is dynamic smem.
template <int FMT, int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int NR>
__global__ void __launch_bounds__(WR * WK * 32, MB) gv_kernel(const __grid_constant__ Args A) {
  static_assert(FMT != 2 || NR == 1, "P11 correction lists are per real warp");
  constexpr int NW = WR * WK, NT = NW * 32;
  constexpr int NC = MT == 1 ? 2 : 1;
  constexpr int NO = WR * MT * (OUT == 2 ? 64 : 128);     // CTA output values per row-tile round (NR rounds)
  constexpr int XB = XP == 1 ? 2 : 1;
  constexpr int XR = XP >= 2 ? XP : 0;                     // x ring depth (units) per warp
  constexpr int UBV = UB<FMT>::v;
  __shared__ float red[NW][MT * 4][32];
  // dynamic smem: x rings [NW][XR][MT][2][32 lanes] uint4, then the cluster leader's [S - 1][NR * NO] partials
  extern __shared__ __align__(16) unsigned char dsm[];
  float* pbuf = reinterpret_cast<float*>(dsm + (size_t)NW * XR * MT * 2 * 32 * 16);
  __shared__ __align__(8) unsigned long long mbar;
  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;
  const int wk = warp % WK, wr = warp / WK;
  const int g = lane >> 2, t = lane & 3;
  const int split = S > 1 ? (int)blockIdx.y : 0;
  const int G = (int)(gridDim.x * gridDim.y) * NW;
  const int gw = ((int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x) * NW + warp;
  const int u0 = (split * WK + wk) * KU;
  const int tok0 = (int)blockIdx.z * (8 * MT);
  const size_t wstep = (size_t)G * NR * UBV;               // one step = every warp's NR units
  const uint8_t* wp = A.w + (size_t)gw * NR * UBV;         // this warp's unit r of step s: wp + s wstep + r UBV

  // 1. the first ST units of weights, x of the first unit(s), then the escape lists
  uint4 R[ST][NR][4];
#pragma unroll
  for (int i = 0; i < ST; ++i)
#pragma unroll
    for (int r = 0; r < NR; ++r)
      if (i < KU) ldunit<FMT>(wp + (size_t)i * wstep + r * UBV, lane, R[i][r]);
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
  uint4* xr = reinterpret_cast<uint4*>(dsm) + (size_t)(threadIdx.x >> 5) * XR * MT * 2 * 32 + (threadIdx.x & 31);
  auto xissue = [&](int su) {   // cp.async this lane's x of unit su into ring slot su % XR (one commit group per unit)
    if (su < KU) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          cpa16(smem_u32(xr + (((su % (XR ? XR : 1)) * MT + mt) * 2 + h) * 32), xb + xo[mt] + su * 8 + h);
    }
    cpa_commit();
  };
  if constexpr (XR > 0) {
#pragma unroll
    for (int i = 0; i < XR - 1; ++i) xissue(i);
  } else {
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      X[0][mt][0] = ldx(xb + xo[mt]);
      X[0][mt][1] = ldx(xb + xo[mt] + 1);
    }
  }
  uint32_t em[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    em[r] = 0xFFFFFFFFu;
    if constexpr (FMT == 0) em[r] = lde(A.esc + ((size_t)gw * NR + r) * 32 + lane);
  }
  int ce0 = 0, ce1 = 0;
  if constexpr (FMT == 2) {   // P11: this warp's correction entries, pulled into L1 now, used after the stream
    ce0 = A.eoff[gw];
    ce1 = A.eoff[gw + 1];
    const char* pe = reinterpret_cast<const char*>(A.esc + ce0) + lane * 128;
    if (pe < reinterpret_cast<const char*>(A.esc + ce1)) asm volatile("prefetch.global.L1 [%0];\n" :: "l"(pe));
  }
#if GV_CLUSTER_OK
  int crank = 0;   // reduction role: rank 0 of the cluster is the leader (the K range above uses blockIdx.y)
  if constexpr (S > 1) {
    crank = (int)cl_rank();
    if (crank == 0 && threadIdx.x == 0) mb_init_expect(smem_u32(&mbar), (uint32_t)((S - 1) * NR * NO * 4));
    cl_arrive();
  }
#endif
  int ecur[NR];
  uint32_t enx[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    ecur[r] = 0;
    enx[r] = 0xFFFFFFFFu;
    if constexpr (FMT == 0) enx[r] = __shfl_sync(0xffffffffu, em[r], 0);
  }

  float acc[NC][NR][MT][4];
#pragma unroll
  for (int c = 0; c < NC; ++c)
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[c][r][mt][i] = 0.f;

  // 2. the stream: decode slot -> refill it -> escapes -> MMAs
#pragma unroll
  for (int s = 0; s < KU; ++s) {
    const int slot = s % ST;
    const int xs = XP == 1 ? (s & 1) : 0;
    if constexpr (XR > 0) {
      xissue(s + XR - 1);
      cpa_wait<XR - 1>();   // unit s's group (each lane reads back only what it copied: no warp sync needed)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) X[0][mt][h] = xr[(((s % XR) * MT + mt) * 2 + h) * 32];
    } else if (XP == 1) {
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
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      uint32_t a[16];
      decode<FMT>(R[slot][r], A.tb, a);
      if (s + ST < KU) ldunit<FMT>(wp + (size_t)(s + ST) * wstep + r * UBV, lane, R[slot][r]);
      if constexpr (FMT == 0) {
        while ((int)(enx[r] >> 20) == s) {
          patch(a, enx[r], lane);
          ++ecur[r];
          enx[r] = ecur[r] < 32 ? __shfl_sync(0xffffffffu, em[r], ecur[r] & 31) : 0xFFFFFFFFu;
        }
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint4 xv = X[xs][mt][q >> 1];
          const uint32_t b0 = (q & 1) ? xv.z : xv.x;
          const uint32_t b1 = (q & 1) ? xv.w : xv.y;
          mma16816(acc[q % NC][r][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);
        }
      }
    }
  }

  // 3. warp partials -> smem -> sum over the WK warps of each row tile (-> cluster leader when S > 1), in NR rounds
  //    (round r: every warp's row tile r) so the reduction buffer stays [NW][MT x 4][32].
  auto cta_value = [&](int i) -> float {
    const int l = i & 31;
    if constexpr (OUT == 2) {
      const int c = (i >> 5) & 1, r = i >> 6, mt = r % MT, w2 = r / MT;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < WK; ++k2) v += red[w2 * WK + k2][mt * 4 + c][l] + red[w2 * WK + k2][mt * 4 + c + 2][l];
      return v;
    } else {
      const int c = (i >> 5) & 3, r = i >> 7, mt = r % MT, w2 = r / MT;
      float v = 0.f;
#pragma unroll
      for (int k2 = 0; k2 < WK; ++k2) v += red[w2 * WK + k2][mt * 4 + c][l];
      return v;
    }
  };
  auto store = [&](int rr, int i, float v) {
    const int l = i & 31, gg = l >> 2, tt = l & 3;
    if constexpr (OUT == 2) {
      const int c = (i >> 5) & 1, r = i >> 6, mt = r % MT, w2 = r / MT;
      const int row = (((int)blockIdx.x * WR + w2) * NR + rr) * 8 + gg;
      const int tok = tok0 + mt * 8 + 2 * tt + c;
      if (tok < A.M) reinterpret_cast<float*>(A.out)[(size_t)tok * A.ldo + row] = v;
    } else {
      const int c = (i >> 5) & 3, r = i >> 7, mt = r % MT, w2 = r / MT;
      const int row = (((int)blockIdx.x * WR + w2) * NR + rr) * 16 + gg + 8 * (c >> 1);
      const int tok = tok0 + mt * 8 + 2 * tt + (c & 1);
      if (tok < A.M) reinterpret_cast<__nv_bfloat16*>(A.out)[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);
    }
  };
  constexpr int NJ = (NO + NT - 1) / NT;
  float own[S > 1 ? NR * NJ : 1];
#if GV_CLUSTER_OK
  uint32_t rb = 0, rm = 0;
  if constexpr (S > 1) {
    if (crank != 0) {
      cl_wait();   // the leader's mbarrier is initialised (its arrive came after the init + fence)
      rb = mapa0(smem_u32(&pbuf[(crank - 1) * NR * NO]));
      rm = mapa0(smem_u32(&mbar));
    }
  }
#endif
#pragma unroll
  for (int rr = 0; rr < NR; ++rr) {
    if (rr > 0) __syncthreads();   // round rr - 1's reads of red are done
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float v = acc[0][rr][mt][i];
        if constexpr (NC == 2) v += acc[1][rr][mt][i];
        red[warp][mt * 4 + i][lane] = v;
      }
    if constexpr (FMT == 2) {
      // P11 escapes were streamed as w_dec (offset 0); add x * (w_true - w_dec) into this warp's partials.
      // entry = kk << 20 | row16 << 16 | w_true (kk: k offset inside the warp's K range). NR == 1 only.
      __syncwarp();
      for (int e = ce0 + lane; e < ce1; e += 32) {
        const uint32_t en = A.esc[e];
        const int kk = (int)(en >> 20), r16 = (int)((en >> 16) & 15u);
        const uint32_t wt = en & 0xFFFFu;
        const uint32_t bse = r16 < 8 ? (A.tb & 0xFFu) : ((A.tb >> 16) & 0xFFu);
        const uint32_t wd = (wt & 0x80FFu) | (bse << 8);
        const float delta = __uint_as_float(wt << 16) - __uint_as_float(wd << 16);
        const int k = u0 * 64 + kk;
#pragma unroll
        for (int j = 0; j < 8 * MT; ++j) {
          const int tok = tok0 + j;
          if (tok < A.M) {
            const float xv = __uint_as_float(((uint32_t)A.x[(size_t)tok * A.K + k]) << 16);
            atomicAdd(&red[warp][(j >> 3) * 4 + (r16 >> 3) * 2 + (j & 1)][(r16 & 7) * 4 + ((j & 7) >> 1)], xv * delta);
          }
        }
      }
    }
    __syncthreads();
    if constexpr (S == 1) {
      for (int i = (int)threadIdx.x; i < NO; i += NT) store(rr, i, cta_value(i));
    } else {
#if GV_CLUSTER_OK
      if (crank != 0) {
        for (int i = (int)threadIdx.x; i < NO; i += NT)
          st_async(rb + 4u * (uint32_t)(rr * NO + i), cta_value(i), rm);
      } else {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const int i = (int)threadIdx.x + j * NT;
          own[rr * NJ + j] = i < NO ? cta_value(i) : 0.f;
        }
      }
#endif
    }
  }
  if constexpr (S > 1) {
#if GV_CLUSTER_OK
    if (crank != 0) return;
    // bounded wait (no trap): a protocol fault shows up as NaN outputs, which the prepare() self-test rejects
    const uint32_t mb = smem_u32(&mbar);
    bool ok = true;
    for (uint32_t it = 0; !mb_try(mb); ++it)
      if (it > (1u << 22)) {
        ok = false;
        break;
      }
#pragma unroll
    for (int rr = 0; rr < NR; ++rr)
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const int i = (int)threadIdx.x + j * NT;
        if (i < NO) {
          float v = own[rr * NJ + j];
#pragma unroll
          for (int r = 0; r < S - 1; ++r) v += pbuf[r * NR * NO + rr * NO + i];
          store(rr, i, ok ? v : __int_as_float(0x7fc00000));
        }
      }
#endif
  }
}

// dynamic smem: x rings NW x XR x MT x 1 KB, then the cluster leader's partial buffer (S - 1) x NR x NO floats
template <int OUT, int MT, int WR, int WK, int S, int NR, int XP>
constexpr int dyn_bytes() {
  return WR * WK * (XP >= 2 ? XP : 0) * MT * 1024 + (S > 1 ? (S - 1) * NR * WR * MT * (OUT == 2 ? 64 : 128) * 4 : 0);
}

template <int FMT, int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int NR>
static int set_attr() {
  constexpr int dyn = dyn_bytes<OUT, MT, WR, WK, S, NR, XP>();
  static int done = 0;   // set once (first call is the prepare() self-test, outside any capture)
  if (dyn > 0 && !done) {
    const int e = (int)cudaFuncSetAttribute(gv_kernel<FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
    if (e) {
      (void)cudaGetLastError();   // do not leave the error pending for the next torch launch
      return e;
    }
  }
  done = 1;
  return 0;
}

template <int FMT, int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int NR>
static int launch(const Args& A, int nbx, int nz, cudaStream_t st) {
  const int e = set_attr<FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR>();
  if (e) return e;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)nbx, (unsigned)S, (unsigned)nz);
  cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);
  cfg.dynamicSmemBytes = (size_t)dyn_bytes<OUT, MT, WR, WK, S, NR, XP>();
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = (unsigned)S;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = S > 1 ? 1u : 0u;
  const int el = (int)cudaLaunchKernelEx(&cfg, gv_kernel<FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR>, A);
  if (el) (void)cudaGetLastError();
  return el;
}

template <int FMT, int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int NR>
static int info(int64_t* res) {
  cudaFuncAttributes fa;
  auto fn = gv_kernel<FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR>;
  int e = set_attr<FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR>();
  if (e) return e;
  e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, WR * WK * 32, (size_t)dyn_bytes<OUT, MT, WR, WK, S, NR, XP>());
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ;
  res[3] = (int64_t)fa.sharedSizeBytes + dyn_bytes<OUT, MT, WR, WK, S, NR, XP>();
  return e;
}

// bit-exact unpack of a packed matrix back to [N, K] (verification); one warp per (virtual warp, step).
// Virtual warp gv = real warp * NR + r streams row tile (bx WR + wr) NR + r.
template <int FMT>
__global__ void unpack_kernel(const uint8_t* __restrict__ w, const uint32_t* __restrict__ esc, uint32_t* __restrict__ out,
                              int K, int G, int nbx, int WR, int WK, int KU, uint32_t tb, int NR) {
  const int gv = (int)blockIdx.x, s = (int)blockIdx.y, lane = (int)threadIdx.x;
  const int g = lane >> 2, t = lane & 3, NW = WR * WK;
  const int gw = gv / NR, rr = gv % NR;
  const int c = gw / NW, wi = gw % NW, wr = wi / WK, wk = wi % WK, bx = c % nbx, by = c / nbx;
  const int rt = (bx * WR + wr) * NR + rr, u = (by * WK + wk) * KU + s;
  uint4 p[4];
  ldunit<FMT>(w + ((size_t)s * G + gv) * UB<FMT>::v, lane, p);
  uint32_t a[16];
  decode<FMT>(p, tb, a);
  if (FMT == 0) {
    for (int j = 0; j < 32; ++j) {
      const uint32_t e = esc[(size_t)gv * 32 + j];
      if ((int)(e >> 20) == s) patch(a, e, lane);
    }
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

}  // namespace gv

using namespace gv;

#define GV_INST(ID, FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR) \
  case ID: return launch<FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR>(A, nbx, nz, st);
#define GV_INFO(ID, FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR) \
  case ID: return info<FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR>(res);

int gv_run_raw(int64_t inst, const void* w, const void* esc, const void* eoff, const void* x, void* out, int M, int K,
               int ldo, uint32_t tb, int nbx, int nz, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint8_t*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.eoff = reinterpret_cast<const int*>(eoff);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  switch (inst) {
@INST@
    default: return -1;
  }
}

int gv_info_raw(int64_t inst, int64_t* res) {
  switch (inst) {
@INFO@
    default: return -1;
  }
}

int gv_unpack_raw(const void* w, const void* esc, void* out, int fmt, int K, int G, int nbx, int wr, int wk, int ku,
                  uint32_t tb, int nr, cudaStream_t st) {
  const dim3 grid((unsigned)G, (unsigned)ku, 1);
  const uint8_t* wb = reinterpret_cast<const uint8_t*>(w);
  const uint32_t* eb = reinterpret_cast<const uint32_t*>(esc);
  uint32_t* ob = reinterpret_cast<uint32_t*>(out);
  if (fmt == 0)
    unpack_kernel<0><<<grid, 32, 0, st>>>(wb, eb, ob, K, G, nbx, wr, wk, ku, tb, nr);
  else if (fmt == 1)
    unpack_kernel<1><<<grid, 32, 0, st>>>(wb, eb, ob, K, G, nbx, wr, wk, ku, tb, nr);
  else
    unpack_kernel<2><<<grid, 32, 0, st>>>(wb, eb, ob, K, G, nbx, wr, wk, ku, tb, nr);
  return (int)cudaGetLastError();
}
"""


# Pinned cuBLASLt (m >= 128): the heuristic's algorithm list for out[m, n] = x[m, k] @ w[n, k]^T (+ beta out), timed in
# prepare(); the fastest is replayed. A separate module (links cublasLt), so a link problem only costs these routes.
LT_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>
struct LtPlan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> res;
};
static cublasLtHandle_t g_h = nullptr;
static std::vector<LtPlan> g_plans;
// column-major view: C^T [n x m] = op_T(W) [n x k] * X^T [k x m]  (W row-major [n, k], x row-major [m, k])
int64_t lt_plan(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t ws_bytes, int64_t max_algos) {
  if (!g_h) TORCH_CHECK(cublasLtCreate(&g_h) == CUBLAS_STATUS_SUCCESS, "lt: create");
  LtPlan p;
  TORCH_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "lt: desc");
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, n, k) == CUBLAS_STATUS_SUCCESS, "lt: la");
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, k, m, k) == CUBLAS_STATUS_SUCCESS, "lt: lb");
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lc, out_f32 ? CUDA_R_32F : CUDA_R_16BF, n, m, n) == CUBLAS_STATUS_SUCCESS,
              "lt: lc");
  cublasLtMatmulPreference_t pref;
  TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS, "lt: pref");
  uint64_t ws = (uint64_t)ws_bytes;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p.res.resize((size_t)max_algos);
  int got = 0;
  cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(g_h, p.op, p.la, p.lb, p.lc, p.lc, pref, (int)max_algos,
                                                    p.res.data(), &got);
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "lt: heuristic status ", (int)s);
  p.res.resize((size_t)got);
  g_plans.push_back(p);
  return ((int64_t)(g_plans.size() - 1) << 8) | (int64_t)got;
}
void lt_run(int64_t pid, int64_t ai, torch::Tensor w, torch::Tensor x, torch::Tensor out, double beta, torch::Tensor ws) {
  LtPlan& p = g_plans[(size_t)pid];
  const float alpha = 1.f, b = (float)beta;
  cublasStatus_t s = cublasLtMatmul(g_h, p.op, &alpha, w.data_ptr(), p.la, x.data_ptr(), p.lb, &b, out.data_ptr(), p.lc,
                                    out.data_ptr(), p.lc, &p.res[(size_t)ai].algo, ws.data_ptr(), (size_t)ws.numel(),
                                    at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "lt_run: status ", (int)s);
}
"""
LT_ENABLE = True
LT_M = (128, 256)
LT_MAX_ALGOS = 24
LT_WS = 32 << 20


# ----------------------------------------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------------------------------------

def _vrows(layer, m):
    """Rows of the (virtual) matrix a route streams: the unpooler slice at <= 8 tokens, 2 x 256 for the router."""
    if layer == "router":
        return 512
    if layer == "unpooler" and m <= SLICED_MAX_TOKENS:
        return UNPOOL_SLICE
    return LAYERS[layer][0]


def _bucket(layer, m):
    if m == 1:
        return 1
    if m <= 8:
        return 8
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    if layer != "router":
        return None
    return 128 if m <= 128 else 256


def _r9(route):
    """Routes are (fmt, mt, wr, wk, ku, st, s, xp[, nr]); nr (row tiles per warp) defaults to 1 (= gv1)."""
    return tuple(route) + (1,) if len(route) == 8 else tuple(route)


def _inst_key(layer, b, route):
    """Template tuple (FMT, OUT, MT, WR, WK, KU, ST, S, MB, XP, NR) of a route, or None if the geometry does not fit."""
    fmt, mt, wr, wk, ku, st, s, xp, nr = _r9(route)
    n = _vrows(layer, TIMED_M[b])
    k = LAYERS[layer][1]
    nrt, nu = n // 16, k // 64
    if n % 16 or k % 64 or nrt % (wr * nr) or nu != wk * ku * s or st > ku or s not in (1, 2, 4, 8):
        return None
    if fmt == 2 and nr != 1:
        return None
    if b <= 32 and 8 * mt < b:   # a route covers all tokens of its bucket in one CTA (token slices above 32)
        return None
    if fmt == 2 and (layer == "router" or ku > 16):   # P11: BF16 layers; correction entries hold k in 10 bits
        return None
    nt = wr * wk * 32
    if nt > 1024:
        return None
    ctas = (nrt // (wr * nr)) * s
    mb = max(1, min(2048 // nt, -(-ctas // 132)))
    return (fmt, 2 if layer == "router" else 0, mt, wr, wk, ku, st, s, mb, xp, nr)


def _src_for(keys):
    inst = "\n".join("    GV_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    info = "\n".join("    GV_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    return CUDA_SRC.replace("@INST@", inst).replace("@INFO@", info)


def _build_one(name, keys, flags):
    from torch.utils.cpp_extension import load_inline
    src = _src_for(keys)
    tag = hashlib.sha1((CPP_SRC + src + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC, cuda_sources=src,
                      functions=["gv_run", "gv_info", "gv_unpack"], extra_cuda_cflags=flags, verbose=False)
    return mod, time.time() - t0


def _build_all(groups, flags):
    """groups: {module name: [template keys]} -> {module name: (module, {key: inst id})}, built in threads."""
    out, errs = {}, {}

    def work(name, keys):
        try:
            mod, dt = _build_one(name, keys, flags)
            out[name] = (mod, {k: i for i, k in enumerate(keys)})
            print(f"[gv1] built {name}: {len(keys)} kernels in {dt:.1f}s", flush=True)
        except Exception as ex2:  # noqa: BLE001
            errs[name] = ex2

    items = list(groups.items())
    for i in range(0, len(items), GV_THREADS):
        ths = [threading.Thread(target=work, args=it) for it in items[i:i + GV_THREADS]]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    for name, ex2 in errs.items():
        print(f"[gv1] BUILD FAILED {name} ({len(groups[name])} kernels) -> its routes fall back to SGLang: "
              f"{str(ex2)[-3000:]}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------
# packing (prepare-time, on the GPU)
# ----------------------------------------------------------------------------------------------------------------

def _i32(t64):
    """uint32 values held in int64 -> int32 with the same bits."""
    return torch.where(t64 >= 2 ** 31, t64 - 2 ** 32, t64).to(torch.int32)


def _frag(wv):
    """[N, K] bf16 -> F [NRT, NU, 32 lanes, 4 k-steps, 8] int32 (16-bit patterns) in fragment order:
    j = 0..7 -> (row g, k0), (g, k0+1), (g+8, k0), (g+8, k0+1), (g, k0+2), (g, k0+3), (g+8, k0+2), (g+8, k0+3),
    k0 = 16 t + 4 q (k permuted inside the 64-k unit so lane t's x operand is 16 contiguous k)."""
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


def _window_base(h7, width=8):
    """Start b (<= 127 - width) of the `width` consecutive h7 values holding the most weights (P12: 8, P11: 4)."""
    hist = torch.bincount(h7.reshape(-1), minlength=128).double()
    cs = torch.cat([torch.zeros(1, dtype=hist.dtype, device=hist.device), hist.cumsum(0)])
    nb = 128 - width
    cover = cs[width:width + nb] - cs[0:nb]
    return int(cover.argmax())


def _stream(units, wr, wk, ku, s, nr=1):
    """units [NRT, NU, UW] int32 (one unit's bytes in kernel order) -> grid-interleaved stream
    [KU, S, nbx, WR, WK, NR, UW] (row tile = (bx WR + wr) NR + r; a warp's NR units of a step are contiguous)."""
    nrt, nu, uw = units.shape
    nbx = nrt // (wr * nr)
    v = units.reshape(nbx, wr, nr, s, wk, ku, uw).permute(5, 3, 0, 1, 4, 2, 6).contiguous()
    return v.view(-1)


def _word(b):
    """[..., 4] bytes (int64) -> int64 little-endian uint32."""
    return b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)


def _rotr32(v, r):
    return ((v >> r) | (v << (32 - r))) & 0xFFFFFFFF


def _warp_of(rt, u, wr, wk, ku, nbx, nr=1):
    """(virtual warp = real warp x NR + r, step) of row tile rt, unit u."""
    rr, rt2 = rt % nr, rt // nr
    bx, wri = rt2 // wr, rt2 % wr
    by, wki, st_ = u // (wk * ku), (u // ku) % wk, u % ku
    return (((by * nbx + bx) * wr + wri) * wk + wki) * nr + rr, st_


def pack(wv, fmt, wr, wk, ku, s, bases, nr=1):
    """Pack the virtual matrix wv [N, K] bf16 for one geometry. bases = (base rows 0-7, base rows 8-15) of every
    16-row tile. fmt 0 = P12 (16-exponent window, <= ESC_CAP XOR escapes per warp; raises ValueError beyond),
    1 = RAW bf16 fragments, 2 = P11 (8-exponent window, escapes as an FP32 correction list per warp).
    Returns dict(w, esc, eoff, tb, G, nbx, nesc[, esc_check])."""
    n, k = wv.shape
    nrt, nu = n // 16, k // 64
    nbx = nrt // (wr * nr)
    G = nbx * s * wr * wk * nr     # virtual warps (stream units per step, escape lists)
    dev = wv.device
    fr = _frag(wv)
    dummy_off = torch.zeros((G + 1,), dtype=torch.int32, device=dev)
    if fmt == 1:
        lo16 = fr[..., 0::2].to(torch.int64)
        hi16 = fr[..., 1::2].to(torch.int64)
        words = _i32(lo16 | (hi16 << 16))                     # [NRT, NU, 32, 4 q, 4 i]
        units = words.permute(0, 1, 3, 2, 4).reshape(nrt, nu, 512)
        esc = torch.full((G * 32,), -1, dtype=torch.int32, device=dev)
        return {"w": _stream(units, wr, wk, ku, s, nr), "esc": esc, "eoff": dummy_off, "tb": 0, "G": G, "nbx": nbx,
                "nesc": 0}
    ba, bb = bases
    jj = torch.arange(8, device=dev)
    basej = torch.where(((jj >> 1) & 1) == 0, torch.tensor(ba, device=dev), torch.tensor(bb, device=dev)).to(torch.int32)
    lo = fr & 0xFF
    h7 = (fr >> 8) & 0x7F
    sg = (fr >> 15) & 1
    d = h7 - basej
    dmax = 7 if fmt == 0 else 3
    isesc = (d < 0) | (d > dmax)
    d = torch.where(isesc, torch.zeros_like(d), d)
    dA, dB, sA, sB = d[..., :4], d[..., 4:], sg[..., :4], sg[..., 4:]
    sBr = sB[..., [3, 0, 1, 2]]
    byte = (dA | (sBr << 3) | (dB << 4) | (sA << 7)).to(torch.int64)             # [NRT, NU, 32, 4 q, 4 bytes]
    cw = _word(byte)                                                              # [NRT, NU, 32, 4 q] int64
    lo64 = lo.to(torch.int64)
    l0 = _i32(_word(lo64[..., 0:4]))                                              # [NRT, NU, 32, 4 q]
    l1 = _i32(_word(lo64[..., 4:8]))
    p0 = torch.stack([l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]], dim=-1)    # [NRT, NU, 32, 4]
    p1 = torch.stack([l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]], dim=-1)
    tb = ba | (ba << 8) | (bb << 16) | (bb << 24)
    pos = isesc.nonzero()
    nesc = int(pos.shape[0])
    if fmt == 2:
        # k-step 3's code word lives in the free residue-2 bits (byte bits 2, 6) of k-steps 0..2's words
        c3 = cw[..., 3]
        c0 = cw[..., 0] | _rotr32(c3 & 0x11111111, 2)
        c1 = cw[..., 1] | _rotr32(c3 & 0x22222222, 3)
        c2 = cw[..., 2] | _rotr32(c3 & 0x88888888, 1)
        p2 = torch.stack([_i32(c0), _i32(c1)], dim=-1)                            # [NRT, NU, 32, 2]
        p3 = _i32(c2)                                                             # [NRT, NU, 32]
        units = torch.cat([p0.reshape(nrt, nu, 128), p1.reshape(nrt, nu, 128), p2.reshape(nrt, nu, 64),
                           p3.reshape(nrt, nu, 32)], dim=2)                       # [NRT, NU, 352] = 1408 B
        # correction entries (kk << 20 | row16 << 16 | w_true), grouped per warp
        esc = torch.zeros((max(nesc, 1),), dtype=torch.int32, device=dev)
        eoff = torch.zeros((G + 1,), dtype=torch.int32, device=dev)
        chk = None
        if nesc:
            rt, u, ln, qq, jv = (pos[:, i] for i in range(5))
            gw, st_ = _warp_of(rt, u, wr, wk, ku, nbx, nr)
            r16 = (ln >> 2) + 8 * ((jv >> 1) & 1)
            k64 = 16 * (ln & 3) + 4 * qq + (jv & 1) + 2 * (jv >> 2)
            wtrue = fr[rt, u, ln, qq, jv].to(torch.int64)
            ent = ((st_ * 64 + k64) << 20) | (r16 << 16) | wtrue
            order = torch.argsort(gw)
            esc = _i32(ent[order])
            cnt = torch.bincount(gw, minlength=G)
            eoff[1:] = torch.cumsum(cnt, 0).to(torch.int32)
            chk = ((rt * 16 + r16) * k + u * 64 + k64, wtrue)
        return {"w": _stream(units, wr, wk, ku, s, nr), "esc": esc, "eoff": eoff, "tb": tb, "G": G, "nbx": nbx,
                "nesc": nesc, "esc_check": chk}
    units = torch.stack([p0, p1, _i32(cw)], dim=2).reshape(nrt, nu, 384)           # [NRT, NU, 3 planes x 32 x 4]
    # escapes: one int32 per escaped weight, grouped per global warp, sorted by step
    esc = torch.full((G * 32,), -1, dtype=torch.int32, device=dev)
    if nesc:
        rt, u, ln, qq, jv = (pos[:, i] for i in range(5))
        gw, st_ = _warp_of(rt, u, wr, wk, ku, nbx, nr)
        x7 = (h7[rt, u, ln, qq, jv] ^ basej[jv]).to(torch.int64)
        ent = (st_ << 20) | (ln << 15) | ((4 * qq + (jv >> 1)) << 11) | ((jv & 1) << 10) | x7
        key = (gw * 4096 + st_) * 16 + 4 * qq + (jv >> 1)
        order = torch.argsort(key)
        gws, ents = gw[order], ent[order]
        cnt = torch.bincount(gws, minlength=G)
        if int(cnt.max()) > ESC_CAP:
            raise ValueError(f"escape cap: a warp has {int(cnt.max())} > {ESC_CAP} escapes")
        start = torch.cumsum(cnt, 0) - cnt
        rank = torch.arange(nesc, device=dev) - start[gws]
        esc[gws * 32 + rank] = _i32(ents)
    return {"w": _stream(units, wr, wk, ku, s, nr), "esc": esc, "eoff": dummy_off, "tb": tb, "G": G, "nbx": nbx,
            "nesc": nesc}


def split_router(w32):
    hi = w32.to(torch.bfloat16)
    lo = (w32 - hi.float()).to(torch.bfloat16)
    return hi, lo


def router_virtual(hi, lo):
    """[512, K]: tile r holds hi rows 8r..8r+7 (tile rows 0-7) and lo rows 8r..8r+7 (tile rows 8-15)."""
    n, k = hi.shape
    return torch.stack([hi.view(n // 8, 8, k), lo.view(n // 8, 8, k)], dim=1).reshape(2 * n, k).contiguous()


# ----------------------------------------------------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------------------------------------------------

_SGL = {"fn": None}
_SEEN = set()
# CUDA graphs captured in prepare() are never destroyed: destroying a graph captured on a stream after a newer graph
# was captured on that stream breaks the newer one (measured locally: cuBLAS state inside the older graph's pool). The
# graphs are small (their pools hold only output buffers), so they are parked here for the life of the process.
_KEEP_GRAPHS = []


def _bf16_gemm(x, w):
    fn = _SGL["fn"]
    return fn(x, w, None) if fn is not None else F.linear(x, w)


def _router_stock(x, state):
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _stock(x, layer, n, state):
    if layer == "router":
        return _router_stock(x, state)
    return _bf16_gemm(x, state[layer][:n])


def _gv_call(r, x, n_out, out_dtype):
    m, k = x.shape
    out = torch.empty((m, n_out), dtype=out_dtype, device=x.device)
    nz = -(-m // (8 * r["mt"])) if r["slices"] else 1
    r["mod"].gv_run(r["inst"], r["w"], r["esc"], r["eoff"], x, out, m, k, n_out, r["tb"], r["nbx"], nz)
    return out


def _lt_call(r, x, n_out):
    """Pinned cuBLASLt algorithm (router: hi then lo accumulated into the same FP32 output, beta = 1)."""
    m = x.shape[0]
    if r["router"]:
        out = torch.empty((m, n_out), dtype=torch.float32, device=x.device)
        r["mod"].lt_run(r["pid"], r["ai"], r["w"], x, out, 0.0, r["ws"])
        r["mod"].lt_run(r["pid"], r["ai"], r["w2"], x, out, 1.0, r["ws"])
        return out
    out = torch.empty((m, n_out), dtype=torch.bfloat16, device=x.device)
    r["mod"].lt_run(r["pid"], r["ai"], r["w"], x, out, 0.0, r["ws"])
    return out


def _route_call(r, x, n_out, layer):
    if r["kind"] == "gv":
        return _gv_call(r, x, n_out, torch.float32 if layer == "router" else torch.bfloat16)
    if r["kind"] == "lt":
        return _lt_call(r, x, n_out)
    if r["kind"] == "vd":
        out = torch.empty((x.shape[0], n_out), dtype=torch.bfloat16, device=x.device)
        _vd_call(r, x, out)
        return out
    if r["kind"] == "dg1":
        out = torch.empty((x.shape[0], n_out), dtype=torch.bfloat16, device=x.device)
        r["mod"].dg1_run(r["kid"], x, r["w"], out, r["err"], r["S"], 0, 0, n_out, 0)
        return out
    if r["kind"] == "mmT":
        return torch.mm(x, r["wt"])
    return torch.addmm(r["bias"], x, r["wt"])   # "addmm": cuBLASLt bias-epilogue kernels (zero bias)


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    m = x.shape[0]
    b = _bucket(layer, m)
    r = state["exact"].get((layer, m))
    key = (layer, m) if r is not None else (layer, b)
    if r is None and b is not None:
        r = state["routes"].get((layer, b))
    if key not in _SEEN:
        _SEEN.add(key)
        if r is not None:
            print(f"[gv1] kernel_fn {layer} m={m} -> {r['kind']} {r['desc']}", flush=True)
        else:
            why = state["why"].get(key, "above the GV range" if b is None else "no route")
            print(f"[gv1] kernel_fn {layer} m={m} -> SGLang path ({why})", flush=True)
    if r is not None and x.is_contiguous() and x.data_ptr() % r.get("align", 16) == 0:
        return _route_call(r, x, n, layer)
    return _stock(x, layer, n, state)


# ----------------------------------------------------------------------------------------------------------------
# prepare: build, pack, verify, autotune
# ----------------------------------------------------------------------------------------------------------------

def _test_input(m, k, seed, device, outliers=False):
    """The judge's recipe: normal x, optional 30x channels, ~1/8 probe rows (one power-of-two entry), >= 1 probe."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device=device)
    if outliers:
        idx = torch.randperm(k, generator=gen, device=device)[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device=device) < 1 / 8
    probes[torch.randint(m, (1,), generator=gen, device=device)] = True
    if m >= 4:
        probes[0] = False   # keep a dense row too
    cols = torch.randint(k, (m,), generator=gen, device=device)
    signs = torch.randint(2, (m,), generator=gen, device=device) * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device=device)
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16), probes


def _check(out, x, w, probes, router, ref=None, tol=None):
    """(ok, msg) with the judge's metric (rel L2 vs FP64 rounded to the output dtype, worst row, probe rows)."""
    if ref is None:
        ref = (x.double() @ w.double().T).to(out.dtype).double()
    o = out.double()
    if not bool(torch.isfinite(o).all()):
        return False, "non-finite"
    d = o - ref
    tot = float(d.norm() / ref.norm().clamp(min=1e-30))
    rn = ref.norm(dim=-1)
    row = float((d.norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
    pr = float((d[probes].norm(dim=-1) / rn[probes].clamp(min=1e-30)).max()) if bool(probes.any()) else 0.0
    t0, ptol = (2e-5, 2e-4) if router else (2e-3, 0.0)
    tol = t0 if tol is None else tol
    ok = tot <= tol and row <= 5 * tol and pr <= ptol
    return ok, f"rel {tot:.1e} row {row:.1e} probe {pr:.1e}"


class _Timer:
    def __init__(self, dev):
        self.flush = torch.empty((64 << 20,), dtype=torch.float32, device=dev)   # 256 MB > L2
        self.flush.fill_(1.0)
        self.floor = None
        # One private capture stream, warmed up eagerly before every capture: cuBLAS creates its per-stream state
        # (workspace) outside any capture, so no state of ours lives in a graph pool that is freed later, and none
        # of it is attached to torch's shared default capture stream that the judge's own captures use.
        self.cs = torch.cuda.Stream()
        self.keep = []

    def graph(self, fn):
        self.cs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.cs):
            for _ in range(2):
                fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.cs):
            fn()
        torch.cuda.synchronize()
        self.keep.append(g)
        return g

    def times(self, graphs, reps):
        """Cold-L2 replay times (us) of each graph, interleaved, order alternating per rep."""
        ev = []
        for r in range(reps):
            order = list(range(len(graphs)))
            if r % 2:
                order.reverse()
            for i in order:
                self.flush.sum()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graphs[i].replay()
                e1.record()
                ev.append((i, e0, e1))
        torch.cuda.synchronize()
        out = [[] for _ in graphs]
        for i, e0, e1 in ev:
            out[i].append(e0.elapsed_time(e1) * 1000.0)
        return [sorted(v)[len(v) // 2] for v in out]

    def release(self):
        torch.cuda.synchronize()
        _KEEP_GRAPHS.extend(self.keep)
        self.keep = []

    def graph_floor(self):
        if self.floor is None:
            z = torch.zeros((1,), device=self.flush.device)
            g = self.graph(lambda: z.add_(1.0))
            self.floor = self.times([g], 15)[0]
        return self.floor


def _build_lt(box):
    """Build the cuBLASLt module (links libcublasLt; tries the toolkit's lib dirs and torch's bundled copy)."""
    import glob
    import os
    from torch.utils.cpp_extension import load_inline, CUDA_HOME
    t0 = time.time()
    dirs = []
    if CUDA_HOME:
        dirs += [os.path.join(CUDA_HOME, d) for d in ("lib64", "lib", os.path.join("targets", "x86_64-linux", "lib"))]
    tdir = os.path.dirname(torch.__file__)
    dirs += glob.glob(os.path.join(os.path.dirname(tdir), "nvidia", "*", "lib"))
    dirs += [os.path.join(tdir, "lib")]
    flags = []
    for d in dirs:
        libs = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
        if libs:
            plain = [lb for lb in libs if lb.endswith("libcublasLt.so")]
            flags = ["-L" + d, "-lcublasLt"] if plain else ["-L" + d, "-l:" + os.path.basename(libs[-1])]
            flags.append("-Wl,-rpath," + d)
            break
    if not flags:
        flags = ["-lcublasLt"]
    tag = hashlib.sha1((LT_CPP + " ".join(flags)).encode()).hexdigest()[:10]
    try:
        box["mod"] = load_inline(name=f"gv1_lt_{tag}", cpp_sources=LT_CPP, functions=["lt_plan", "lt_run"],
                                 extra_ldflags=flags, with_cuda=True, verbose=False)
        print(f"[gv1] built cuBLASLt module in {time.time() - t0:.1f}s ({' '.join(flags)})", flush=True)
    except Exception as ex:  # noqa: BLE001
        box["err"] = f"{str(ex)[-1500:]} (flags {flags})"


VD_ENABLE = True
VD_FLAGS = 8   # L2 evict_first on the weight tiles (R-vd-2: >= neutral on every shape)
VD_INSTS = [(1, 128, 8, 2, 0, 1), (1, 128, 8, 3, 0, 1), (1, 128, 9, 1, 0, 0), (2, 128, 6, 2, 0, 1)]
VD_BOARD = {('o', 128): [(2, None, 0)], ('pooler', 128): [(0, None, 0)], ('qkv_gate', 128): [(1, None, 0)], ('unpooler', 128): [(3, None, 0)]}
VD_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t vd_run(int64_t kid, torch::Tensor x, torch::Tensor wpk, torch::Tensor wraw, torch::Tensor y, torch::Tensor err,
               torch::Tensor stamps, int64_t nrb, int64_t w_rb, int64_t w_s, int64_t flags, int64_t pdl, int64_t pf);
std::vector<int64_t> vd_info(int64_t kid);
"""
VD_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cstring>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace vd {

constexpr long long WAIT_BUDGET = 2000000000LL;   // ~1 s of clock64: a hang becomes err[0] = 1 (R9)
enum : int { F_EXIT = 1, F_NOMMA = 2, F_NOX = 4, F_EFW = 8, F_ELX = 16, F_STAMP = 32, F_WMAP = 64, F_TRIG = 128, F_WAIT0 = 256,
             F_DYN = 512 };

struct Params {
  const uint8_t* w;          // packed tiles (bulk mode): tile (rb, s) at w + rb * w_rb + s * w_s (bytes)
  void* y;                   // [T, ldy] bf16 or f32
  int* err;
  unsigned long long* stamps;   // [grid][4] (flag 32)
  long long w_rb, w_s;
  int nks, T, ldy, flags, pf;   // pf: weight stages issued before griddepcontrol.wait (the rest go x-first)
  int rot;                      // K-start rotation per row block (host arg pf >> 8)
};

// RBD 1: the DSMEM receive buffer is dedicated smem after the ring (no barrier after the main loop);
// RBD 0: it aliases the drained ring, one cluster barrier after the main loop
template <int NWG, int BN, int NST, int CL, int F32, int RBD>
struct Cfg {
  static constexpr int BM = 64 * NWG;
  static constexpr int WB = BM * 128;              // weight tile bytes (BM rows x 64 bf16)
  static constexpr int XB = BN * 128;              // x tile bytes (BN tokens x 64 bf16)
  static constexpr int SB = WB + XB;               // stage bytes (multiple of 1024)
  static constexpr int NTH = NWG * 128;            // consumer threads
  static constexpr int THREADS = NTH + 32;         // + one producer warp
  static constexpr int NACC = BN / 2;              // fp32 accumulators per consumer thread
  static constexpr int NB = BN / 8;                // 8-token blocks
  static constexpr int JMAX = (NB + CL - 1) / CL;  // most token blocks a rank owns
  static constexpr int RB_BYTES = CL > 1 ? (CL - 1) * JMAX * NTH * 16 : 0;   // [src slot][block][thread] float4
  static constexpr int RB_OFF = RBD ? NST * SB : 0;
  static constexpr int BAR_OFF = NST * SB + (RBD ? RB_BYTES : 0);
  static constexpr int SMEM = 1024 + BAR_OFF + 16 * NST + 64;
  static constexpr int CTR_OFF = BAR_OFF + 16 * NST + 8;    // dynamic split: stage counter (rank 0's is used)
  static constexpr int ENDF_OFF = CTR_OFF + 4;               // dynamic split: ring index of this rank's end marker
  static constexpr int OB = F32 ? 4 : 2;           // output element bytes
  static constexpr int SROW = F32 ? (BM + 4) : (BM + 8);   // staging row pitch (elements), 16-B multiple
  static constexpr int S_OFF = RBD ? 0 : RB_BYTES;  // staging tile [owned tokens][BM] in the drained ring
  static constexpr int S_BYTES = JMAX * 8 * SROW * OB;
  static_assert(SB % 1024 == 0, "stage alignment");
  static_assert(S_OFF + S_BYTES <= NST * SB, "epilogue buffers alias the ring");
  static_assert(SMEM <= 232448, "smem");
  static_assert(CL >= 1 && CL <= NB, "cluster split");
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// @lab{
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// @lab}
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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, bool& dead) {
  if (dead || mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); dead = true; return; }
  }
}
__device__ __forceinline__ uint64_t pol_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// @lab{
__device__ __forceinline__ uint64_t pol_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// @lab}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_h(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
// @lab{
__device__ __forceinline__ void tma2d_h(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;\n"
               :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
// @lab}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t mapa(uint32_t local_addr, uint32_t rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(local_addr), "r"(rank));
  return ra;
}
__device__ __forceinline__ void st_async_f4(uint32_t raddr, float a, float b, float c, float d, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n"
               :: "r"(raddr), "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)), "r"(__float_as_uint(d)),
                  "r"(rbar) : "memory");
}
__device__ __forceinline__ int atom_add_cl(uint32_t raddr, int v) {
  int old;
  asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0, [%1], %2;\n" : "=r"(old) : "r"(raddr), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ bool mbar_try_cl(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait_cl(uint32_t b, uint32_t par, int* err, bool& dead) {
  if (dead || mbar_try_cl(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try_cl(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 2); dead = true; return; }
  }
}
// SM90 smem matrix descriptor: K-major, 128-B swizzle, 8-row groups 1024 B apart (SBO), LBO unused (1)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db);
template <> __device__ __forceinline__ void wgmma_ss<128>(float* d, uint64_t da, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
               : "l"(da), "l"(db));
}

template <int NWG, int BN, int NST, int CL, int F32, int RBD>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
vd_kernel(const __grid_constant__ CUtensorMap tmx, const __grid_constant__ CUtensorMap tmw, const Params P) {
  using C = Cfg<NWG, BN, NST, CL, F32, RBD>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t s_raw = sa(smem_raw);
  const uint32_t sbase = (s_raw + 1023u) & ~1023u;
  uint8_t* gbase = smem_raw + (sbase - s_raw);
  const uint32_t full0 = sbase + C::BAR_OFF, empty0 = full0 + 8 * NST, rbar = full0 + 16 * NST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int flags = P.flags;
  // @lab{
  if (flags & F_EXIT) return;
  // @lab}
  uint32_t crank = 0;
  if constexpr (CL > 1) asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(crank));
  const int rb = blockIdx.x / CL;
  const int s0 = ((int)crank * P.nks) / CL, s1 = (((int)crank + 1) * P.nks) / CL;
  const int nk = s1 - s0;
  const int jlo = CL > 1 ? ((int)crank * C::NB) / CL : 0;          // token blocks this rank finishes
  const int jhi = CL > 1 ? (((int)crank + 1) * C::NB) / CL : C::NB;
  // @lab{
  const bool stamp = (flags & F_STAMP) != 0;
  unsigned long long t_entry = 0;
  if (stamp && tid == 0) t_entry = gtime();
  // @lab}
  bool dead = false;

  if (tid == 0) {
#pragma unroll 1
    for (int i = 0; i < NST; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    if constexpr (CL > 1) {
      mbar_init(rbar, 1);
      mbar_expect_tx(rbar, (uint32_t)((CL - 1) * (jhi - jlo) * C::NTH * 16));   // the peers' pushes of my blocks
    }
    if (CL > 1 && (flags & F_DYN) != 0) {   // dynamic split only: a cluster-scope fence costs ~0.5 us at entry (R-vd-3)
      *reinterpret_cast<volatile int*>(gbase + C::CTR_OFF) = 0;
      *reinterpret_cast<volatile int*>(gbase + C::ENDF_OFF) = 0x7fffffff;
      asm volatile("fence.acq_rel.cluster;\n" ::: "memory");   // the counter init is visible before the cluster arrive
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if constexpr (CL > 1) cluster_arrive_relaxed();   // phase 0: "my barriers are initialized" (waited before any push)

  const bool dyn = CL > 1 && (flags & F_DYN) != 0;
  if (warp == NWG * 4) {
    // ------------------------------------------------------------------ producer (one lane)
    bool pw0 = false;   // lane 0 already did its phase-0 cluster wait (dynamic split)
    if (lane == 0) {
      const bool efw = (flags & F_EFW) != 0;
      const uint64_t pw = efw ? pol_first() : 0ull;
      uint32_t txb = (uint32_t)(C::WB + C::XB);
      // @lab{
      const bool nox = (flags & F_NOX) != 0, wmap = (flags & F_WMAP) != 0, elx = (flags & F_ELX) != 0;
      const uint64_t px = elx ? pol_last() : 0ull;
      if (nox) txb = (uint32_t)C::WB;
      if (wmap) asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmw)) : "memory");
      // @lab}
      const uint8_t* wbase = P.w + (long long)rb * P.w_rb;
      asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(&tmx)) : "memory");
      // K rotation: row block rb walks its K range starting at stage (rb * rot) mod nk, so CTAs that run in lockstep
      // read different x tiles (128 SMs hitting the same L2 lines at once is a hotspot); 0 = no rotation
      const int off = P.rot > 0 ? (int)(((long long)rb * P.rot) % nk) : 0;
      auto kst = [&](int i) { int j = i + off; if (j >= nk) j -= nk; return s0 + j; };
      auto issue_w = [&](int i, uint32_t dst, uint32_t bar) {
        const int s = kst(i);
        // @lab{
        if (wmap) { if (efw) tma2d_h(dst, &tmw, s * 64, rb * C::BM, bar, pw); else tma2d(dst, &tmw, s * 64, rb * C::BM, bar); return; }
        // @lab}
        const void* src = wbase + (long long)s * P.w_s;
        if (efw) bulk_g2s_h(dst, src, C::WB, bar, pw); else bulk_g2s(dst, src, C::WB, bar);
      };
      auto issue_x = [&](int i, uint32_t dst, uint32_t bar) {
        const int s = kst(i);
        // @lab{
        if (nox) return;
        if (elx) { tma2d_h(dst + C::WB, &tmx, s * 64, 0, bar, px); return; }
        // @lab}
        tma2d(dst + C::WB, &tmx, s * 64, 0, bar);
      };
      if (!dyn) {
      const int npro = nk < NST ? nk : NST;
      const int pf = P.pf < npro ? P.pf : npro;
#pragma unroll 1
      for (int i = 0; i < pf; ++i) {   // a few weight stages before the PDL wait (they do not depend on the previous kernel)
        mbar_expect_tx(full0 + 8 * i, txb);
        issue_w(i, sbase + i * C::SB, full0 + 8 * i);
      }
      pdl_wait();
#pragma unroll 1
      for (int i = 0; i < pf; ++i) issue_x(i, sbase + i * C::SB, full0 + 8 * i);
#pragma unroll 1
      for (int i = pf; i < npro; ++i) {   // x first: stage 0 must not queue behind NST weight tiles
        mbar_expect_tx(full0 + 8 * i, txb);
        issue_x(i, sbase + i * C::SB, full0 + 8 * i);
        issue_w(i, sbase + i * C::SB, full0 + 8 * i);
      }
#pragma unroll 1
      for (int i = NST; i < nk; ++i) {
        const int slot = i % NST;
        const uint32_t dst = sbase + slot * C::SB, bar = full0 + 8 * slot;
        mbar_wait(empty0 + 8 * slot, (uint32_t)(((i / NST) - 1) & 1), P.err, dead);
        mbar_expect_tx(bar, txb);
        issue_x(i, dst, bar);
        issue_w(i, dst, bar);
      }
      // @dyn{
      } else if constexpr (CL > 1) {
        // dynamic split inside the cluster: rank r first takes stages r, r + CL, .. (NST of them, static), then claims the
        // rest from a counter in rank 0's smem (DSMEM atomic, one claim ahead), so the ranks finish together; an end
        // marker (ring index in smem + a plain arrive on that slot's full barrier) tells the consumers where the list ends
        const int nkt = P.nks;
        const int offd = P.rot > 0 ? (int)(((long long)rb * P.rot) % nkt) : 0;
        auto ks = [&](int c) { int j = c + offd; if (j >= nkt) j -= nkt; return j; };
        auto put_w = [&](int s, uint32_t dst, uint32_t bar) {
          // @lab{
          if (wmap) { if (efw) tma2d_h(dst, &tmw, s * 64, rb * C::BM, bar, pw); else tma2d(dst, &tmw, s * 64, rb * C::BM, bar); return; }
          // @lab}
          const void* src = wbase + (long long)s * P.w_s;
          if (efw) bulk_g2s_h(dst, src, C::WB, bar, pw); else bulk_g2s(dst, src, C::WB, bar);
        };
        auto put_x = [&](int s, uint32_t dst, uint32_t bar) {
          // @lab{
          if (nox) return;
          if (elx) { tma2d_h(dst + C::WB, &tmx, s * 64, 0, bar, px); return; }
          // @lab}
          tma2d(dst + C::WB, &tmx, s * 64, 0, bar);
        };
        int npre = 0;
        while (npre < NST && (int)crank + CL * npre < nkt) ++npre;
        const int pf = P.pf < npre ? P.pf : npre;
#pragma unroll 1
        for (int i = 0; i < pf; ++i) { mbar_expect_tx(full0 + 8 * i, txb); put_w(ks((int)crank + CL * i), sbase + i * C::SB, full0 + 8 * i); }
        pdl_wait();
#pragma unroll 1
        for (int i = 0; i < pf; ++i) put_x(ks((int)crank + CL * i), sbase + i * C::SB, full0 + 8 * i);
#pragma unroll 1
        for (int i = pf; i < npre; ++i) {
          mbar_expect_tx(full0 + 8 * i, txb);
          put_x(ks((int)crank + CL * i), sbase + i * C::SB, full0 + 8 * i);
          put_w(ks((int)crank + CL * i), sbase + i * C::SB, full0 + 8 * i);
        }
        int iend = npre;
        if (npre == NST) {
          asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");   // phase 0: rank 0's counter is initialized
          pw0 = true;
          const uint32_t ctr = mapa(sbase + C::CTR_OFF, 0u);
          int c = atom_add_cl(ctr, 1) + CL * NST;
#pragma unroll 1
          for (int i = NST; ; ++i) {
            const int slot = i % NST;
            mbar_wait(empty0 + 8 * slot, (uint32_t)(((i / NST) - 1) & 1), P.err, dead);
            if (c >= nkt || dead) { iend = i; break; }
            const int cn = atom_add_cl(ctr, 1) + CL * NST;   // claim ahead: its latency hides behind this stage's issue
            const uint32_t dst = sbase + slot * C::SB, bar = full0 + 8 * slot;
            mbar_expect_tx(bar, txb);
            put_x(ks(c), dst, bar);
            put_w(ks(c), dst, bar);
            c = cn;
          }
        }
        *reinterpret_cast<volatile int*>(gbase + C::ENDF_OFF) = iend;
        mbar_arrive(full0 + 8 * (iend % NST));                // release: the consumers read the marker after their wait
      // @dyn}
      }
    }
    __syncwarp();
    if constexpr (CL > 1) {
      if (!(lane == 0 && pw0)) asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");   // phase 0
      if constexpr (!RBD) {   // phase 1: rings drained
        asm volatile("barrier.cluster.arrive.release;\n" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");
      }
    }
    return;
  }

  // -------------------------------------------------------------------- consumers
  const int wg = warp >> 2, t = tid & 127;
  float acc[C::NACC];
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) acc[q] = 0.f;
  bool nomma = false;
  const bool w0 = (flags & F_WAIT0) != 0;
  // @lab{
  nomma = (flags & F_NOMMA) != 0;
  unsigned long long t_first = 0;
  // @lab}
#pragma unroll 1
  for (int i = 0; dyn || i < nk; ++i) {
    const int slot = i % NST;
    mbar_wait(full0 + 8 * slot, (uint32_t)((i / NST) & 1), P.err, dead);
    if (dyn && (i == *reinterpret_cast<volatile int*>(gbase + C::ENDF_OFF) || dead)) break;   // this rank's end marker
    // @lab{
    if (stamp && i == 0 && tid == 0) t_first = gtime();
    // @lab}
    if (!nomma) {
      const uint32_t st = sbase + slot * C::SB;
      const uint64_t da = desc_sw128(st + wg * 8192), db = desc_sw128(st + C::WB);
#pragma unroll
      for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) wgmma_ss<BN>(acc, da + 2 * kk, db + 2 * kk);
      wg_commit();
      if (w0) wg_wait<0>(); else wg_wait<1>();
#pragma unroll
      for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    }
    if (w0) { if (t == 0) mbar_arrive(empty0 + 8 * slot); }
    else if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NST));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < C::NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
  // @lab{
  unsigned long long t_loop = 0;
  if (stamp && tid == 0) t_loop = gtime();
  // @lab}
  if (flags & F_TRIG) pdl_trigger();

  // -------------------------------------------------------------------- epilogue
  named_sync(1, C::NTH);   // every warpgroup is done with the ring
  const int g = lane >> 2, t4 = lane & 3;
  const int r0 = wg * 64 + (warp & 3) * 16 + g;     // this thread's first row within the tile (second: r0 + 8)
  uint8_t* stg = gbase + C::S_OFF;
  if constexpr (CL > 1) {
    // push every token block another rank finishes straight from registers into that rank's receive buffer
    // ([sender slot][block - its jlo][thread] float4), fire-and-forget st.async completing the receiver's rbar
    cluster_wait();                                     // phase 0: every rank's rbar is initialized
    if constexpr (!RBD) { cluster_arrive(); cluster_wait(); }   // phase 1: every rank's ring is drained (buffer aliases it)
    const uint32_t rbl = sbase + C::RB_OFF;
#pragma unroll
    for (int p = 0; p < CL; ++p) {
      if (p == (int)crank) continue;
      const int plo = (p * C::NB) / CL, phi = ((p + 1) * C::NB) / CL;
      const int slot = (int)crank < p ? (int)crank : (int)crank - 1;
      const uint32_t rbuf = mapa(rbl, (uint32_t)p), rbr = mapa(rbar, (uint32_t)p);
#pragma unroll
      for (int j = 0; j < C::NB; ++j) {
        if (j < plo || j >= phi) continue;
        st_async_f4(rbuf + (uint32_t)(((slot * C::JMAX + (j - plo)) * C::NTH + tid) * 16),
                    acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3], rbr);
      }
    }
    mbar_wait_cl(rbar, 0, P.err, dead);                 // the other ranks' copies of my blocks have landed
    const float4* RB = reinterpret_cast<const float4*>(gbase + C::RB_OFF);
#pragma unroll
    for (int j = 0; j < C::NB; ++j) {
      if (j < jlo || j >= jhi) continue;
      float v0 = acc[4 * j], v1 = acc[4 * j + 1], v2 = acc[4 * j + 2], v3 = acc[4 * j + 3];
#pragma unroll
      for (int s = 0; s < CL - 1; ++s) {
        const float4 q = RB[(s * C::JMAX + (j - jlo)) * C::NTH + tid];
        v0 += q.x; v1 += q.y; v2 += q.z; v3 += q.w;
      }
      acc[4 * j] = v0; acc[4 * j + 1] = v1; acc[4 * j + 2] = v2; acc[4 * j + 3] = v3;
    }
  }
  // stage owned token blocks [jlo, jhi) as [token][BM rows] (pitch SROW)
#pragma unroll
  for (int j = 0; j < C::NB; ++j) {
    if (j < jlo || j >= jhi) continue;
    const int tt = 8 * (j - jlo) + 2 * t4;
    if constexpr (F32) {
      float* s = reinterpret_cast<float*>(stg);
      s[tt * C::SROW + r0] = acc[4 * j];
      s[(tt + 1) * C::SROW + r0] = acc[4 * j + 1];
      s[tt * C::SROW + r0 + 8] = acc[4 * j + 2];
      s[(tt + 1) * C::SROW + r0 + 8] = acc[4 * j + 3];
    } else {
      __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(stg);
      s[tt * C::SROW + r0] = __float2bfloat16_rn(acc[4 * j]);
      s[(tt + 1) * C::SROW + r0] = __float2bfloat16_rn(acc[4 * j + 1]);
      s[tt * C::SROW + r0 + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
      s[(tt + 1) * C::SROW + r0 + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
  named_sync(1, C::NTH);
  {
    constexpr int CH = C::BM * C::OB / 16;            // 16-B chunks per token row
    const int tlo = 8 * jlo;
    int thi = 8 * jhi;
    if (thi > P.T) thi = P.T;
    const int ntok = thi > tlo ? thi - tlo : 0;
    uint8_t* yb = reinterpret_cast<uint8_t*>(P.y) + (size_t)rb * C::BM * C::OB;
#pragma unroll 1
    for (int idx = tid; idx < ntok * CH; idx += C::NTH) {
      const int tt = idx / CH, c = idx - tt * CH;
      const uint4 v = *reinterpret_cast<const uint4*>(stg + (size_t)tt * C::SROW * C::OB + c * 16);
      *reinterpret_cast<uint4*>(yb + ((size_t)(tlo + tt) * P.ldy) * C::OB + c * 16) = v;
    }
  }
  // @lab{
  if (stamp && tid == 0) {
    unsigned long long* sp = P.stamps + 4 * (size_t)blockIdx.x;
    sp[0] = t_entry; sp[1] = t_first; sp[2] = t_loop; sp[3] = gtime();
  }
  // @lab}
}

// @lab{
__global__ void null_kernel(int* e) { if (threadIdx.x == 1023) e[3] = 0; }
// @lab}

}  // namespace vd

typedef CUresult (*vd_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static vd_encode_fn vd_get_encode() {
  static vd_encode_fn fn = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) cudaGetLastError();
    else fn = reinterpret_cast<vd_encode_fn>(p);
  }
  return fn;
}

static void vd_map(CUtensorMap* tm, const void* ptr, int64_t rows, int64_t cols, int box_rows) {
  vd_encode_fn enc = vd_get_encode();
  TORCH_CHECK(enc != nullptr, "vd: cuTensorMapEncodeTiled unavailable");
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)(cols * 2)};
  cuuint32_t box[2] = {64u, (cuuint32_t)box_rows};
  cuuint32_t es[2] = {1u, 1u};
  const CUresult r = enc(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, es,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "vd: tensor map encode failed ", (int)r);
}

template <int NWG, int BN, int NST, int CL, int F32, int RBD>
static int64_t vd_launch_t(const torch::Tensor& x, const torch::Tensor& wpk, const torch::Tensor& wraw, const torch::Tensor& y,
                           const torch::Tensor& err, const torch::Tensor& stamps, int64_t nrb, int64_t w_rb, int64_t w_s,
                           int64_t flags, int64_t pdl, int64_t pf) {
  using C = vd::Cfg<NWG, BN, NST, CL, F32, RBD>;
  auto kern = vd::vd_kernel<NWG, BN, NST, CL, F32, RBD>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    attr = true;
  }
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "vd: x bf16 [T, K]");
  const int64_t T = x.size(0), K = x.size(1);
  TORCH_CHECK(T >= 1 && T <= BN && K % 64 == 0 && x.data_ptr() != nullptr && ((uintptr_t)x.data_ptr() % 16) == 0, "vd: x shape");
  TORCH_CHECK(y.is_cuda() && y.dim() == 2 && y.stride(1) == 1 && y.size(0) >= T && y.size(1) >= nrb * C::BM, "vd: y");
  TORCH_CHECK(y.scalar_type() == (F32 ? at::kFloat : at::kBFloat16), "vd: y dtype");
  TORCH_CHECK((y.stride(0) * C::OB) % 16 == 0 && ((uintptr_t)y.data_ptr() % 16) == 0, "vd: y alignment");
  CUtensorMap tmx, tmw;
  memset(&tmw, 0, sizeof(tmw));
  vd_map(&tmx, x.data_ptr(), T, K, BN);
  TORCH_CHECK(wpk.is_cuda() && ((uintptr_t)wpk.data_ptr() % 16) == 0, "vd: wpk");
  // @lab{
  if ((flags & vd::F_WMAP) != 0) {
    TORCH_CHECK(wraw.dim() == 2 && wraw.size(1) == K && wraw.size(0) >= nrb * C::BM && wraw.is_contiguous(), "vd: wraw");
    vd_map(&tmw, wraw.data_ptr(), wraw.size(0), K, C::BM);
  }
  if (flags & vd::F_STAMP) TORCH_CHECK(stamps.numel() >= 4 * nrb * CL, "vd: stamps");
  // @lab}
  vd::Params P;
  P.w = reinterpret_cast<const uint8_t*>(wpk.data_ptr());
  P.y = y.data_ptr();
  P.err = err.data_ptr<int>();
  P.stamps = reinterpret_cast<unsigned long long*>(stamps.data_ptr<int64_t>());
  P.w_rb = w_rb; P.w_s = w_s;
  P.nks = (int)(K / 64); P.T = (int)T; P.ldy = (int)y.stride(0); P.flags = (int)flags; P.pf = (int)(pf & 255); P.rot = (int)(pf >> 8);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(nrb * CL), 1, 1);
  cfg.blockDim = dim3(C::THREADS, 1, 1);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream().stream();
  cudaLaunchAttribute at_[2];
  int na = 0;
  if (CL > 1) {
    at_[na].id = cudaLaunchAttributeClusterDimension;
    at_[na].val.clusterDim.x = CL; at_[na].val.clusterDim.y = 1; at_[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = na ? at_ : nullptr;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tmx, tmw, P));
  return 0;
}

// @lab{
template <int NWG, int BN, int NST, int CL, int F32, int RBD>
static std::vector<int64_t> vd_info_t() {
  using C = vd::Cfg<NWG, BN, NST, CL, F32, RBD>;
  auto kern = vd::vd_kernel<NWG, BN, NST, CL, F32, RBD>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  int ncl = -1;
  {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(132 * CL, 1, 1);
    cfg.blockDim = dim3(C::THREADS, 1, 1);
    cfg.dynamicSmemBytes = C::SMEM;
    cudaLaunchAttribute a;
    a.id = cudaLaunchAttributeClusterDimension;
    a.val.clusterDim.x = CL; a.val.clusterDim.y = 1; a.val.clusterDim.z = 1;
    cfg.attrs = &a; cfg.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&ncl, kern, &cfg) != cudaSuccess) { cudaGetLastError(); ncl = -1; }
  }
  return {(int64_t)C::SMEM, (int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)ncl, (int64_t)C::BM, (int64_t)BN,
          (int64_t)NST, (int64_t)CL, (int64_t)F32};
}
// @lab}

int64_t vd_run(int64_t kid, torch::Tensor x, torch::Tensor wpk, torch::Tensor wraw, torch::Tensor y, torch::Tensor err,
               torch::Tensor stamps, int64_t nrb, int64_t w_rb, int64_t w_s, int64_t flags, int64_t pdl, int64_t pf) {
  // @lab{
  if (kid < 0) {
    vd::null_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream().stream()>>>(err.data_ptr<int>());
    return 0;
  }
  // @lab}
  switch (kid) {
    case 0: return vd_launch_t<1, 128, 8, 2, 0, 1>(x, wpk, wraw, y, err, stamps, nrb, w_rb, w_s, flags, pdl, pf);
    case 1: return vd_launch_t<1, 128, 8, 3, 0, 1>(x, wpk, wraw, y, err, stamps, nrb, w_rb, w_s, flags, pdl, pf);
    case 2: return vd_launch_t<1, 128, 9, 1, 0, 0>(x, wpk, wraw, y, err, stamps, nrb, w_rb, w_s, flags, pdl, pf);
    case 3: return vd_launch_t<2, 128, 6, 2, 0, 1>(x, wpk, wraw, y, err, stamps, nrb, w_rb, w_s, flags, pdl, pf);
  }
  TORCH_CHECK(false, "vd: unknown kid ", kid);
  return -2;
}

// @lab{
std::vector<int64_t> vd_info(int64_t kid) {
  switch (kid) {
    case 0: return vd_info_t<1, 128, 8, 2, 0, 1>();
    case 1: return vd_info_t<1, 128, 8, 3, 0, 1>();
    case 2: return vd_info_t<1, 128, 9, 1, 0, 0>();
    case 3: return vd_info_t<2, 128, 6, 2, 0, 1>();
  }
  TORCH_CHECK(false, "vd: unknown kid ", kid);
  return {};
}
// @lab}
"""
_VD = {"mod": None, "err": None, "errt": None}


def _vd_build_job():
    try:
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1((VD_CPP_SRC + VD_CUDA_SRC).encode()).hexdigest()[:10]
        t1 = time.time()
        _VD["mod"] = load_inline(name=f"vd_{tag}", cpp_sources=VD_CPP_SRC, cuda_sources=VD_CUDA_SRC,
                                 functions=["vd_run", "vd_info"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
        print(f"[vd] built {len(VD_INSTS)} kernels in {time.time() - t1:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        _VD["err"] = repr(ex)[-1500:]


def _vd_pack(w, bm):
    """lossless repack: pre-swizzled (TMA SW128 image) [bm x 64] tiles in front-major order (tile (s, rb) at s nrb + rb)"""
    n, k = w.shape
    nrb, nks = n // bm, k // 64
    t = w.contiguous().view(nrb, bm, nks, 8, 8).permute(0, 2, 1, 3, 4)
    r = torch.arange(bm, device=w.device)
    c = torch.arange(8, device=w.device)
    idx = (c[None, :] ^ (r[:, None] & 7)).view(1, 1, bm, 8, 1).expand(nrb, nks, bm, 8, 8)
    t = torch.gather(t, 3, idx).permute(1, 0, 2, 3, 4).contiguous()
    back = torch.gather(t, 3, idx.permute(1, 0, 2, 3, 4)).permute(1, 2, 0, 3, 4).reshape(n, k)
    if not torch.equal(back.view(torch.int16), w.contiguous().view(torch.int16)):
        raise RuntimeError("VD pack is not bit-exact")
    return t.view(torch.uint8).view(-1), bm * 128, nrb * bm * 128


def _vd_call(r, x, out):
    r["mod"].vd_run(r["kid"], x, r["w"], r["w"], out, r["err"], r["stamps"], r["nrb"], r["w_rb"], r["w_s"], r["flags"], 0, r["pf"])


def _vd_cands(state, layer, m, w):
    """VD candidate routes for (layer, m), or [] (logged) when the module or the shape does not fit."""
    if not VD_ENABLE or (layer, m) not in VD_BOARD:
        return []
    th = _VD.get("thread")
    if th is not None:
        th.join()
    if _VD["mod"] is None:
        print(f"[vd] module unavailable -> no VD candidates: {_VD['err']}", flush=True)
        return []
    if _VD["errt"] is None:
        _VD["errt"] = torch.zeros((4,), dtype=torch.int32, device=w.device)
        _VD["stt"] = torch.zeros((4,), dtype=torch.int64, device=w.device)
    out = []
    for kid, fl, pf in VD_BOARD[(layer, m)]:
        nwg, bn, nst, cl, f32, rbd = VD_INSTS[kid]
        fl = VD_FLAGS if fl is None else fl
        bm = 64 * nwg
        if f32 or m > bn or w.shape[0] % bm or w.shape[1] % 64:
            continue
        packs = state.setdefault("vd_w", {})   # per prepare() (never a module-level cache of weights)
        try:
            if (layer, bm) not in packs:
                packs[(layer, bm)] = _vd_pack(w, bm)
        except Exception as ex:  # noqa: BLE001
            print(f"[vd] {layer} pack bm {bm} failed: {ex!r}", flush=True)
            continue
        flat, w_rb, w_s = packs[(layer, bm)]
        out.append({"kind": "vd", "mod": _VD["mod"], "kid": kid, "w": flat, "w_rb": w_rb, "w_s": w_s, "nrb": w.shape[0] // bm,
                    "err": _VD["errt"], "stamps": _VD["stt"], "flags": fl, "pf": pf,
                    "desc": f"vd{kid}(bm{bm},cl{cl},st{nst},rb{rbd},f{fl},p{pf})"})
    return out



def _lt_search(state, mod, timer, floor, dev, budget_end):
    """Pinned cuBLASLt (and addmm) routes at the exact timed m >= 128: time every heuristic algorithm, keep the best if it
    beats both SGLang's call and any GV route of that bucket. Returns summary strings."""
    summary = []
    kmaj = {}
    ws = torch.empty((LT_WS,), dtype=torch.uint8, device=dev)
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        for m in LT_M:
            if time.time() > budget_end:
                print(f"[gv1] LT {layer} m={m}: prepare budget exhausted -> skipped", flush=True)
                continue
            router = layer == "router"
            n = 256 if router else LAYERS[layer][0]
            k = LAYERS[layer][1]
            w = state["router_hi"] if router else state[layer]
            try:
                code = int(mod.lt_plan(m, n, k, 1 if router else 0, LT_WS, LT_MAX_ALGOS))
            except Exception as ex:  # noqa: BLE001
                print(f"[gv1] LT {layer} m={m}: plan failed {ex!r}", flush=True)
                continue
            pid, cnt = code >> 8, code & 255
            x, probes = _test_input(m, k, 31337 + m, dev)
            ref = (x.double() @ (state["router"] if router else w).double().T).to(
                torch.float32 if router else torch.bfloat16).double()
            cands = []
            for ai in range(cnt):
                r = {"kind": "lt", "mod": mod, "pid": pid, "ai": ai, "w": w, "w2": state["router_lo"] if router else None,
                     "ws": ws, "router": router, "desc": f"lt{ai}"}
                try:
                    got = _lt_call(r, x, n)
                    torch.cuda.synchronize()
                    ok, msg = _check(got, x, state["router"] if router else w, probes, router, ref,
                                     None if router else 5e-3)
                except Exception as ex:  # noqa: BLE001
                    ok, msg = False, repr(ex)[:120]
                if ok:
                    cands.append(r)
            if not router:
                bias = torch.zeros((n,), dtype=torch.bfloat16, device=dev)
                r = {"kind": "addmm", "bias": bias, "wt": w.t(), "desc": "addmm0"}
                got = _route_call(r, x, n, layer)
                ok, msg = _check(got, x, w, probes, False, ref, 5e-3)
                if ok:
                    cands.append(r)
                # "torchT": cuBLAS on a K-major copy of W ([K, N] contiguous, a lossless repack): other kernels than TN
                if layer not in kmaj:
                    kmaj[layer] = w.t().contiguous()
                r = {"kind": "mmT", "wt": kmaj[layer], "desc": "torchT"}
                got = _route_call(r, x, n, layer)
                ok, msg = _check(got, x, w, probes, False, ref, 5e-3)
                if ok:
                    cands.append(r)
                for r in _vd_cands(state, layer, m, w):
                    try:
                        got = _route_call(r, x, n, layer)
                        got2 = _route_call(r, x, n, layer)
                        torch.cuda.synchronize()
                        ok, msg = _check(got, x, w, probes, False, ref, 5e-3)
                        ok = ok and bool(torch.equal(got, got2)) and int(r["err"][0]) == 0
                        for tp in (m - 3, m // 2 + 1):
                            xp = x[:tp].contiguous()
                            gp = _route_call(r, xp, n, layer)
                            torch.cuda.synchronize()
                            okp, msgp = _check(gp, xp, w, probes[:tp], False, None, 5e-3)
                            ok, msg = ok and okp, msg + f"; m{tp} " + msgp
                    except Exception as ex:  # noqa: BLE001
                        ok, msg = False, repr(ex)[:160]
                    print(f"[vd] {layer}/m{m} {r['desc']}: {msg} -> {'candidate' if ok else 'REJECTED'}", flush=True)
                    if ok:
                        cands.append(r)
            for r in _hhx_pin_cands(state, layer, m, w, x, probes, ref, dev):
                cands.append(r)
            gvr = state["routes"].get((layer, _bucket(layer, m)))
            if gvr is not None:
                cands.append(gvr)
            if not cands:
                print(f"[gv1] LT {layer} m={m}: no algorithm passed ({cnt} tried)", flush=True)
                continue
            # first pass: every candidate, few reps; second pass: the best 3 vs SGLang
            try:
                gs = [timer.graph(lambda: _stock(x, layer, n, state))]
                for r in cands:
                    gs.append(timer.graph(lambda r=r: _route_call(r, x, n, layer)))
                tt = timer.times(gs, 7)
                order = sorted(range(len(cands)), key=lambda i: tt[i + 1])[:3]
                top = [cands[i] for i in order]
                tt2 = timer.times([gs[0]] + [gs[i + 1] for i in order], GV_REPS)
                timer.release()
            except Exception as ex:  # noqa: BLE001
                print(f"[gv1] LT {layer} m={m}: timing failed {ex!r}", flush=True)
                timer.release()
                continue
            base = max(tt2[0] - floor, 0.1)
            best_i = min(range(len(top)), key=lambda i: tt2[i + 1])
            best, bt = top[best_i], max(tt2[best_i + 1] - floor, 0.1)
            first = " ".join(f"{c['desc']}:{t - floor:.1f}" for c, t in sorted(zip(cands, tt[1:]), key=lambda z: z[1])[:6])
            print(f"[gv1] LT {layer} m={m}: sgl {base:.2f} | best {best['desc']} {bt:.2f} | first pass {first}", flush=True)
            if bt <= GV_MARGIN * base:
                state["exact"][(layer, m)] = best
                summary.append(f"{layer}/m{m}: {best['kind']} {best['desc']} x{base / bt:.3f}")
            else:
                state["why"][(layer, m)] = f"no Lt/GV route beats SGLang (best x{base / bt:.3f})"
                summary.append(f"{layer}/m{m}: SGLang (best x{base / bt:.3f})")
    return summary


def _route_ms(b):
    """Token counts a route is verified at."""
    return {1: (1,), 8: (8, 5, 7), 16: (16, 9), 32: (32, 17), 128: (128, 33, 100), 256: (256, 129, 300)}[b]

DG1_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t dg1_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor err, int64_t S, int64_t pdl,
int64_t trig, int64_t ldy, int64_t flags);
std::vector<int64_t> dg1_info(int64_t kid, int64_t S);
"""
DG1_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
namespace dg1 {
constexpr long long WAIT_BUDGET = 2000000000LL;
struct Params {
const uint8_t* w;
void* y;
int* err;
int N, nk, T, ldy, f32, trig;
int flags;
};
template <int T, int NWG, int MT>
struct Cfg {
static constexpr int BM = 64 * MT * NWG;
static constexpr int XB = T * 128;
static constexpr int WB = BM * 128;
static constexpr int STAGE = XB + WB;
static constexpr int ST0 = (220 * 1024) / STAGE;
static constexpr int ST = ST0 > 10 ? 10 : ST0;
  static constexpr int RS = BM + 4;                         // partial row stride (floats): conflict-free fragment stores
  static constexpr int SMEM = 1024 + ST * STAGE + 16 * ST;  // align slack + ring + full/empty barriers
  static constexpr int THREADS = 128 * NWG + 32;
static_assert(T % 8 == 0 && T >= 8 && T <= 256, "T");
static_assert(T * RS * 4 <= ST * STAGE, "partials fit the ring");
static_assert(ST >= 2, "ring");
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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, bool& dead) {
if (dead || mbar_try(b, par)) return;
const long long t0 = clock64();
while (!mbar_try(b, par)) {
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); dead = true; return; }
}
}
__device__ __forceinline__ uint64_t policy_evict_first() {
uint64_t pol;
asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
return pol;
}
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
:: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
:: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_x(uint32_t dst, const CUtensorMap* tm, int c0, int c1, uint32_t bar) {
asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
:: "r"(dst), "l"(reinterpret_cast<uint64_t>(tm)), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t cl_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ uint32_t cl_n() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_nctarank;\n" : "=r"(r)); return r; }
__device__ __forceinline__ void cl_sync() {
asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
}
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t r) {
uint32_t o;
asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(o) : "r"(a), "r"(r));
return o;
}
__device__ __forceinline__ float4 ld_cl4(uint32_t a) {
float4 v;
asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
return v;
}
__device__ __forceinline__ uint64_t desc_il(uint32_t a) {
return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(1024u >> 4) << 32);
}
__device__ __forceinline__ uint64_t desc_sw128(uint32_t a) {
return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024u >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_ss(float* d, uint64_t da, uint64_t db);
@WGSPECS@
template <int NR> __device__ __forceinline__ void fence_regs(float* d) {
#pragma unroll
for (int i = 0; i < NR; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <int T, int NWG, int MT>
__global__ void __launch_bounds__(Cfg<T, NWG, MT>::THREADS, 1)
dg1_kernel(const __grid_constant__ CUtensorMap tmx, const Params P) {
using C = Cfg<T, NWG, MT>;
constexpr int ST = C::ST, STAGE = C::STAGE, XB = C::XB, BM = C::BM, RS = C::RS;
extern __shared__ uint8_t dsm[];
const uint32_t raw = sa(dsm);
const uint32_t base = (raw + 1023u) & ~1023u;
  float* red = reinterpret_cast<float*>(dsm + (base - raw));
  const uint32_t bfull = base + ST * STAGE, bempty = bfull + 8 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const int rank = (int)cl_rank(), nrank = (int)cl_n();
const int tile = blockIdx.y;
const int kc0 = (P.nk * rank) / nrank, kc1 = (P.nk * (rank + 1)) / nrank, nloc = kc1 - kc0;
if (P.trig) pdl_launch();
if (P.flags & 1) return;
if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(bfull + 8 * s, 1);
      mbar_init(bempty + 8 * s, 4 * NWG);
}
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__syncthreads();
bool dead = false;
float acc[MT][T / 2];
if (warp == 4 * NWG) {
if (lane == 0) {
const uint64_t pol = policy_evict_first();
const bool tl = (P.flags >> 8) & 1, nox = (P.flags >> 2) & 1, noh = (P.flags >> 3) & 1;
const uint32_t sbytes = nox ? (uint32_t)C::WB : (uint32_t)STAGE;
const size_t slab = tl ? (size_t)C::WB : (size_t)P.N * 128;
const uint8_t* wsrc = tl ? P.w + ((size_t)tile * P.nk + kc0) * C::WB : P.w + (size_t)kc0 * slab + (size_t)tile * (BM * 128);
const int pre = nloc < ST ? nloc : ST;
      for (int i = 0; i < pre; ++i) {   // weights first: never written by a predecessor
        mbar_expect_tx(bfull + 8 * i, sbytes);
        if (noh) bulk_g2s(base + i * STAGE + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * i);
        else bulk_g2s_ef(base + i * STAGE + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * i, pol);
      }
pdl_wait();
      if (!nox)
        for (int i = 0; i < pre; ++i) tma_x(base + i * STAGE, &tmx, (kc0 + i) * 64, 0, bfull + 8 * i);
      for (int i = ST; i < nloc; ++i) {
const int s = i % ST;
mbar_wait(bempty + 8 * s, (uint32_t)(((i / ST) - 1) & 1), P.err, dead);
        const uint32_t st = base + s * STAGE;
        mbar_expect_tx(bfull + 8 * s, sbytes);
        if (noh) bulk_g2s(st + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * s);
        else bulk_g2s_ef(st + XB, wsrc + (size_t)i * slab, C::WB, bfull + 8 * s, pol);
        if (!nox) tma_x(st, &tmx, (kc0 + i) * 64, 0, bfull + 8 * s);
      }
}
} else {
const int g = warp >> 2;
#pragma unroll
for (int mt = 0; mt < MT; ++mt)
#pragma unroll
for (int q = 0; q < T / 2; ++q) acc[mt][q] = 0.f;
for (int i = 0; i < nloc; ++i) {
      const int s = i % ST;
      mbar_wait(bfull + 8 * s, (uint32_t)((i / ST) & 1), P.err, dead);
      const uint32_t xs = base + s * STAGE, ws = xs + XB + g * (MT * 8192);
wg_fence();
if (!(P.flags & 2)) {
#pragma unroll
for (int mt = 0; mt < MT; ++mt)
#pragma unroll
for (int j = 0; j < 4; ++j) wgmma_ss<T>(acc[mt], desc_il(ws + mt * 8192 + j * 256), desc_sw128(xs + j * 32));
}
wg_commit();
wg_wait<1>();
if (i > 0 && lane == 0) mbar_arrive(bempty + 8 * ((i - 1) % ST));
}
wg_wait<0>();
#pragma unroll
for (int mt = 0; mt < MT; ++mt) fence_regs<T / 2>(acc[mt]);
}
pdl_wait();
__syncthreads();
if (warp < 4 * NWG) {
const int g = warp >> 2, w4 = warp & 3;
#pragma unroll
for (int mt = 0; mt < MT; ++mt) {
const int r0 = g * MT * 64 + mt * 64 + w4 * 16 + (lane >> 2);
#pragma unroll
for (int j = 0; j < T / 8; ++j) {
const int t0 = 8 * j + 2 * (lane & 3);
red[t0 * RS + r0] = acc[mt][4 * j];
red[(t0 + 1) * RS + r0] = acc[mt][4 * j + 1];
red[t0 * RS + r0 + 8] = acc[mt][4 * j + 2];
red[(t0 + 1) * RS + r0 + 8] = acc[mt][4 * j + 3];
}
}
}
if (nrank > 1) cl_sync(); else __syncthreads();
constexpr int R4 = BM / 4, Q4 = T * R4;
const int q0 = (Q4 * rank) / nrank, q1 = (Q4 * (rank + 1)) / nrank;
const uint32_t redA = sa(red);
for (int q = q0 + tid; q < q1; q += C::THREADS) {
const int t = q / R4, r4 = q - t * R4;
float4 sum;
if (nrank == 1) {
sum = *reinterpret_cast<const float4*>(red + t * RS + 4 * r4);
} else {
const uint32_t off = redA + (uint32_t)(t * RS + 4 * r4) * 4u;
sum = make_float4(0.f, 0.f, 0.f, 0.f);
for (int c = 0; c < nrank; ++c) {
const float4 v = ld_cl4(mapa(off, (uint32_t)c));
sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
}
}
if (t < P.T) {
const size_t o = (size_t)t * P.ldy + (size_t)tile * BM + 4 * r4;
if (P.f32) {
*reinterpret_cast<float4*>(reinterpret_cast<float*>(P.y) + o) = sum;
} else {
__nv_bfloat162 a = __floats2bfloat162_rn(sum.x, sum.y), b = __floats2bfloat162_rn(sum.z, sum.w);
uint2 u;
u.x = *reinterpret_cast<uint32_t*>(&a);
u.y = *reinterpret_cast<uint32_t*>(&b);
*reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(P.y) + o) = u;
}
}
}
if (nrank > 1) cl_sync();
}
}
typedef CUresult (*dg1_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static dg1_encode_fn dg1_get_encode() {
static dg1_encode_fn fn = nullptr;
static bool tried = false;
if (!tried) {
tried = true;
void* p = nullptr;
cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
const cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
const cudaError_t e = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
#endif
if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) cudaGetLastError();
else fn = reinterpret_cast<dg1_encode_fn>(p);
}
return fn;
}
template <int T, int NWG, int MT>
static int64_t dg1_launch_t(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& y, const torch::Tensor& err,
int64_t S, int64_t pdl, int64_t trig, int64_t ldy, int64_t flags) {
using C = dg1::Cfg<T, NWG, MT>;
auto kern = dg1::dg1_kernel<T, NWG, MT>;
static bool attr = false;
if (!attr) {
C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
attr = true;
}
TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "dg1: x bf16 [T, K] contiguous");
TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.dim() == 5 && w.is_contiguous(), "dg1: packed w [K/64, N/8, 8, 8, 8]");
const int64_t Tact = x.size(0), K = x.size(1), N = w.size(1) * 8;
TORCH_CHECK(w.size(0) * 64 == K && w.size(2) == 8 && w.size(3) == 8 && w.size(4) == 8, "dg1: packed w shape");
TORCH_CHECK(Tact >= 1 && Tact <= T && K % 64 == 0 && N % C::BM == 0, "dg1: shape not supported by this instantiation");
TORCH_CHECK(S >= 1 && S <= 16 && S <= K / 64, "dg1: split");
TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "dg1: align");
const bool f32 = y.scalar_type() == at::kFloat;
TORCH_CHECK(y.is_cuda() && (f32 || y.scalar_type() == at::kBFloat16) && y.dim() == 2 && y.stride(1) == 1 && y.stride(0) == ldy
&& y.size(0) >= Tact && y.size(1) >= N && ldy % 4 == 0 && reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0, "dg1: y");
TORCH_CHECK(err.is_cuda() && err.scalar_type() == at::kInt && err.numel() >= 1, "dg1: err");
dg1_encode_fn enc = dg1_get_encode();
TORCH_CHECK(enc != nullptr, "dg1: cuTensorMapEncodeTiled unavailable");
CUtensorMap tm;
cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)Tact};
cuuint64_t strides[1] = {(cuuint64_t)(K * 2)};
cuuint32_t box[2] = {64u, (cuuint32_t)T};
cuuint32_t es[2] = {1u, 1u};
const CUresult r = enc(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
TORCH_CHECK(r == CUDA_SUCCESS, "dg1: tensor map encode failed ", (int)r);
dg1::Params P;
P.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
P.y = y.data_ptr();
P.err = err.data_ptr<int>();
P.N = (int)N; P.nk = (int)(K / 64); P.T = (int)Tact; P.ldy = (int)ldy; P.f32 = f32 ? 1 : 0; P.trig = (int)trig;
P.flags = (int)flags;
cudaLaunchConfig_t cfg = {};
cfg.gridDim = dim3((unsigned)S, (unsigned)(N / C::BM), 1);
cfg.blockDim = dim3(C::THREADS, 1, 1);
cfg.dynamicSmemBytes = C::SMEM;
cfg.stream = at::cuda::getCurrentCUDAStream().stream();
cudaLaunchAttribute la[2];
int na = 0;
if (S > 1) {
la[na].id = cudaLaunchAttributeClusterDimension;
la[na].val.clusterDim.x = (unsigned)S; la[na].val.clusterDim.y = 1; la[na].val.clusterDim.z = 1;
++na;
}
la[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
la[na].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
++na;
cfg.attrs = la;
cfg.numAttrs = na;
C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, tm, P));
return 0;
}
template <int T, int NWG, int MT>
static std::vector<int64_t> dg1_info_t(int64_t S) {
using C = dg1::Cfg<T, NWG, MT>;
auto kern = dg1::dg1_kernel<T, NWG, MT>;
C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
cudaLaunchConfig_t cfg = {};
cfg.gridDim = dim3((unsigned)S, 1, 1);
cfg.blockDim = dim3(C::THREADS, 1, 1);
cfg.dynamicSmemBytes = C::SMEM;
cudaLaunchAttribute la[1];
la[0].id = cudaLaunchAttributeClusterDimension;
la[0].val.clusterDim.x = (unsigned)S; la[0].val.clusterDim.y = 1; la[0].val.clusterDim.z = 1;
cfg.attrs = la;
cfg.numAttrs = 1;
int ncl = -1;
if (cudaOccupancyMaxActiveClusters(&ncl, kern, &cfg) != cudaSuccess) { cudaGetLastError(); ncl = -1; }
cudaFuncAttributes fa;
C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
return {(int64_t)ncl, (int64_t)C::SMEM, (int64_t)C::THREADS, (int64_t)C::BM, (int64_t)C::ST, (int64_t)fa.numRegs,
(int64_t)fa.localSizeBytes};
}
int64_t dg1_run(int64_t kid, torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor err, int64_t S, int64_t pdl,
int64_t trig, int64_t ldy, int64_t flags) {
switch (kid) {
@RUNCASES@
}
TORCH_CHECK(false, "dg1: unknown kid ", kid);
return -2;
}
std::vector<int64_t> dg1_info(int64_t kid, int64_t S) {
switch (kid) {
@INFOCASES@
}
TORCH_CHECK(false, "dg1: unknown kid ", kid);
return {};
}
"""
DG1_INSTS = [(32, 1, 1), (32, 2, 1), (32, 2, 2), (128, 1, 1), (128, 2, 1), (128, 2, 2), (256, 1, 1), (256, 2, 1)]
def dg1_wg_spec(n):
    nr = n // 2
    regs = ",".join(f"%{i}" for i in range(nr))
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(nr))
    return (f"template <> __device__ __forceinline__ void wgmma_ss<{n}>(float* d, uint64_t da, uint64_t db) {{\n"
            f'  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            f'               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nr}, %{nr + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f"               : {outs}\n               : \"l\"(da), \"l\"(db));\n}}\n")
def dg1_cuda_src(insts=DG1_INSTS):
    ns = sorted({t for t, _, _ in insts})
    run = "\n".join(f"    case {k}: return dg1_launch_t<{t}, {g}, {m}>(x, w, y, err, S, pdl, trig, ldy, flags);"
                    for k, (t, g, m) in enumerate(insts))
    info = "\n".join(f"    case {k}: return dg1_info_t<{t}, {g}, {m}>(S);" for k, (t, g, m) in enumerate(insts))
    return (DG1_CUDA_SRC.replace("@WGSPECS@", "".join(dg1_wg_spec(n) for n in ns))
            .replace("@RUNCASES@", run).replace("@INFOCASES@", info))
def dg1_pack(w):
    n, k = w.shape
    assert n % 8 == 0 and k % 64 == 0
    return w.contiguous().view(n // 8, 8, k // 64, 8, 8).permute(2, 0, 3, 1, 4).contiguous()
HHX_LT_CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <vector>
struct LtPlan {
cublasLtMatmulDesc_t op = nullptr;
cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
cudaDataType_t ct = CUDA_R_16BF;
std::vector<cublasLtMatmulHeuristicResult_t> res;
};
static cublasLtHandle_t g_h = nullptr;
static std::vector<LtPlan> g_plans;
static const cublasLtMatmulAlgoConfigAttributes_t kAttrs[9] = {
CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
static const cublasLtMatmulAlgo_t* lt_algo(int64_t pid, int64_t ai) {
TORCH_CHECK(pid >= 0 && (size_t)pid < g_plans.size() && ai >= 0 && (size_t)ai < g_plans[(size_t)pid].res.size(), "lt: index");
return &g_plans[(size_t)pid].res[(size_t)ai].algo;
}
static int64_t lt_plan_impl(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t nn, int64_t batch,
                              int64_t ws_bytes, int64_t max_algos) {
  TORCH_CHECK(batch >= 1 && n % batch == 0, "lt: batch");
  const int64_t nb = n / batch;
  if (!g_h) TORCH_CHECK(cublasLtCreate(&g_h) == CUBLAS_STATUS_SUCCESS, "lt: create");
LtPlan p;
TORCH_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "lt: desc");
cublasOperation_t ta = nn ? CUBLAS_OP_N : CUBLAS_OP_T, tb = CUBLAS_OP_N;
cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
TORCH_CHECK((nn ? cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, nb, k, n)
: cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, nb, k)) == CUBLAS_STATUS_SUCCESS, "lt: la");
TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, k, m, k) == CUBLAS_STATUS_SUCCESS, "lt: lb");
p.ct = out_f32 ? CUDA_R_32F : CUDA_R_16BF;
  TORCH_CHECK(cublasLtMatrixLayoutCreate(&p.lc, p.ct, nb, m, n) == CUBLAS_STATUS_SUCCESS, "lt: lc");
  if (batch > 1) {   // ULP: B slices of the output features; x is shared (batch offset 0)
    const int32_t bc = (int32_t)batch;
    cublasLtMatrixLayout_t ls[3] = {p.la, p.lb, p.lc};
const int64_t so[3] = {nn ? nb : k * nb, 0, nb};
for (int i = 0; i < 3; ++i) {
      TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)) ==
                      CUBLAS_STATUS_SUCCESS, "lt: batch count");
      TORCH_CHECK(cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so[i],
                                                   sizeof(int64_t)) == CUBLAS_STATUS_SUCCESS, "lt: batch offset");
    }
}
cublasLtMatmulPreference_t pref;
TORCH_CHECK(cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS, "lt: pref");
uint64_t ws = (uint64_t)ws_bytes;
cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
p.res.resize((size_t)max_algos);
int got = 0;
cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(g_h, p.op, p.la, p.lb, p.lc, p.lc, pref, (int)max_algos,
p.res.data(), &got);
cublasLtMatmulPreferenceDestroy(pref);
TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "lt: heuristic status ", (int)s);
p.res.resize((size_t)got);
g_plans.push_back(p);
return ((int64_t)(g_plans.size() - 1) << 8) | (int64_t)got;
}
int64_t lt_plan(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t ws_bytes, int64_t max_algos) {
return lt_plan_impl(m, n, k, out_f32, 0, 1, ws_bytes, max_algos);
}
int64_t lt_plan_b(int64_t m, int64_t n, int64_t k, int64_t out_f32, int64_t nn, int64_t batch, int64_t ws_bytes,
                  int64_t max_algos) {
  return lt_plan_impl(m, n, k, out_f32, nn, batch, ws_bytes, max_algos);
}
std::vector<int64_t> lt_cfg(int64_t pid, int64_t ai) {
const cublasLtMatmulAlgo_t* a = lt_algo(pid, ai);
std::vector<int64_t> v;
for (auto t : kAttrs) {
size_t sz = 0;
uint64_t x = 0;
bool ok = cublasLtMatmulAlgoConfigGetAttribute(a, t, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz > 0 &&
sz <= sizeof(x) && cublasLtMatmulAlgoConfigGetAttribute(a, t, &x, sz, &sz) == CUBLAS_STATUS_SUCCESS;
v.push_back(ok ? (int64_t)x : -1);
}
return v;
}
int64_t lt_make(int64_t pid, std::vector<int64_t> cfg, int64_t ws_bytes) {
TORCH_CHECK(pid >= 0 && (size_t)pid < g_plans.size() && cfg.size() == 9, "lt: make");
LtPlan& p = g_plans[(size_t)pid];
cublasLtMatmulAlgo_t a;
if (cublasLtMatmulAlgoInit(g_h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, p.ct, p.ct, (int)cfg[0], &a) !=
CUBLAS_STATUS_SUCCESS)
return -1;
for (int i = 1; i < 9; ++i) {
size_t sz = 0;
uint64_t v = (uint64_t)cfg[(size_t)i];
if (cublasLtMatmulAlgoConfigGetAttribute(&a, kAttrs[i], nullptr, 0, &sz) != CUBLAS_STATUS_SUCCESS || sz == 0 ||
sz > sizeof(v) || cublasLtMatmulAlgoConfigSetAttribute(&a, kAttrs[i], &v, sz) != CUBLAS_STATUS_SUCCESS)
return -1;
}
cublasLtMatmulHeuristicResult_t r = {};
if (cublasLtMatmulAlgoCheck(g_h, p.op, p.la, p.lb, p.lc, p.lc, &a, &r) != CUBLAS_STATUS_SUCCESS ||
r.workspaceSize > (size_t)ws_bytes)
return -1;
r.algo = a;
p.res.push_back(r);
return (int64_t)p.res.size() - 1;
}
void lt_run(int64_t pid, int64_t ai, torch::Tensor w, torch::Tensor x, torch::Tensor out, double beta, torch::Tensor ws) {
const cublasLtMatmulAlgo_t* a = lt_algo(pid, ai);
LtPlan& p = g_plans[(size_t)pid];
const float alpha = 1.f, b = (float)beta;
cublasStatus_t s = cublasLtMatmul(g_h, p.op, &alpha, w.data_ptr(), p.la, x.data_ptr(), p.lb, &b, out.data_ptr(), p.lc,
out.data_ptr(), p.lc, a, ws.data_ptr(), (size_t)ws.numel(),
at::cuda::getCurrentCUDAStream());
TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "lt_run: status ", (int)s);
}
'''


# ----------------------------------------------------------------------------------------------------------------
# HHX (R-dense-probe, 2026-10-31): the layer's dense engines that the Oct 28 board file (vd_1) never had, as extra candidates.
#   * dg1 : the layer's DG1 BF16 bulk-copy + wgmma skinny GEMM at m = 32 (qkv_gate, o), verbatim from kernels/lab/k3_hh_3tkv.py
#   * pins: the layer's cuBLASLt pins (dpin: algorithm id 66 with explicit tile / custom option / cluster, NN or batched
#           layouts, built by lt_plan_b / lt_make) at m = 128 / 256. The tile / custom / cluster words follow the Apache-2.0
#           public file of alygg77 (okc-muse-spark-kernels, commit 6cae49a6; their NOTICE: vjeux, Nathen Fernandes (MIT),
#           Matsvei); they are data (algorithm configuration words), not code.
# Every candidate is verified with the judge metric against FP64 (+ bitwise repeat) and timed cold-L2 in a CUDA graph against the
# route that vd_1's search picked; it replaces that route only if it is >= 3 % faster. Every skip is printed.
# ----------------------------------------------------------------------------------------------------------------

HHX_ENABLE = True
HHX_LT_WS = 32 << 20
HHX_DG1_INSTS = [(32, 1, 1), (32, 2, 1)]
# (layer, m) -> (T, nwg, mt, s) of the layer's DG1_GEOMETRY ('qkv', 32) and ('o', 32)
HHX_DG1 = {("qkv_gate", 32): (32, 2, 1, 4), ("o", 32): (32, 1, 1, 1)}
# (layer, m) -> [(9-word cuBLASLt config, nn, batch, tag)]; nn = 1: the weight is stored [K, N] (a lossless transpose)
HHX_PINS = {('o', 128): [((66, 18, 35, 1, 0, 0, 3, 0, 5), 1, 8, 'dpin128')], ('o', 256): [((66, 20, 35, 1, 0, 0, 6, 0, 5), 1, 2, 'alyo256')], ('qkv_gate', 128): [((66, 92, 35, 1, 0, 0, 2, 0, 5), 0, 16, 'dpin128')], ('qkv_gate', 256): [((66, 124, 35, 1, 0, 0, 0, 0, 2), 0, 1, 'dpin256')], ('unpooler', 128): [((66, 18, 35, 1, 0, 0, 3, 0, 5), 1, 2, 'alyun128')], ('unpooler', 256): [((66, 20, 35, 1, 0, 0, 7, 0, 5), 1, 2, 'dpin256')]}
_HHX = {"lt": None, "dg1": None, "lt_err": None, "dg1_err": None, "threads": [], "wt": {}, "dg1w": {}, "ws": None}


def _hhx_lt_flags():
    import glob
    import os
    from torch.utils.cpp_extension import CUDA_HOME
    dirs = [os.path.join(CUDA_HOME, d) for d in ("lib64", "lib", os.path.join("targets", "x86_64-linux", "lib"))] if CUDA_HOME else []
    tdir = os.path.dirname(torch.__file__)
    dirs += glob.glob(os.path.join(os.path.dirname(tdir), "nvidia", "*", "lib")) + [os.path.join(tdir, "lib")]
    for d in dirs:
        libs = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
        if libs:
            lib = "-lcublasLt" if any(x.endswith("libcublasLt.so") for x in libs) else "-l:" + os.path.basename(libs[-1])
            return ["-L" + d, lib, "-Wl,-rpath," + d]
    return ["-lcublasLt"]


def _hhx_build_lt():
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    flags = _hhx_lt_flags()
    try:
        _HHX["lt"] = load_inline(name="hhx_lt_" + hashlib.sha1((HHX_LT_CPP + " ".join(flags)).encode()).hexdigest()[:10],
                                 cpp_sources=HHX_LT_CPP, functions=["lt_plan", "lt_plan_b", "lt_cfg", "lt_make", "lt_run"],
                                 extra_ldflags=flags, with_cuda=True, verbose=False)
        print(f"[hhx] built the pin cuBLASLt module in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        _HHX["lt_err"] = f"{str(ex)[-600:]} (flags {flags})"


def _hhx_build_dg1():
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    try:
        cuda = dg1_cuda_src(list(HHX_DG1_INSTS))
        tag = hashlib.sha1((DG1_CPP_SRC + cuda).encode()).hexdigest()[:10]
        _HHX["dg1"] = load_inline(name=f"hhx_dg1_{tag}", cpp_sources=DG1_CPP_SRC, cuda_sources=cuda,
                                  functions=["dg1_run", "dg1_info"],
                                  extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
        print(f"[hhx] built DG1 ({len(HHX_DG1_INSTS)} instances) in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:  # noqa: BLE001
        _HHX["dg1_err"] = str(ex)[-800:]


def _hhx_start(dev):
    """Builds in background threads (Hopper only), joined where the candidates are used."""
    if not HHX_ENABLE or torch.cuda.get_device_capability(dev)[0] < 9:
        return
    for fn in (_hhx_build_lt, _hhx_build_dg1):
        th = threading.Thread(target=fn)
        th.start()
        _HHX["threads"].append(th)


def _hhx_join():
    for th in _HHX["threads"]:
        th.join()
    _HHX["threads"] = []


def _hhx_pin_cands(state, layer, m, w, x, probes, ref, dev):
    """Pinned cuBLASLt candidates of (layer, m): kind 'lt' routes of the pin module, verified; [] when none applies."""
    out = []
    if not HHX_ENABLE or (layer, m) not in HHX_PINS:
        return out
    _hhx_join()
    mod = _HHX["lt"]
    if mod is None:
        print(f"[hhx] pins {layer}/m{m}: module unavailable ({_HHX['lt_err']})", flush=True)
        return out
    if _HHX["ws"] is None:
        _HHX["ws"] = torch.empty((HHX_LT_WS,), dtype=torch.uint8, device=dev)
    n, k = LAYERS[layer]
    for cfg, nn, bt, tag in HHX_PINS[(layer, m)]:
        desc = f"pin{tag}{'nn' if nn else ''}b{bt}"
        try:
            wl = w
            if nn:
                if layer not in _HHX["wt"]:
                    _HHX["wt"][layer] = w.t().contiguous()
                wl = _HHX["wt"][layer]
            code = int(mod.lt_plan_b(m, n, k, 0, nn, bt, HHX_LT_WS, 32))
            pid, cnt = code >> 8, code & 255
            ai = next((i for i in range(cnt) if tuple(mod.lt_cfg(pid, i)) == tuple(cfg)), None)
            if ai is None:
                ai = int(mod.lt_make(pid, list(cfg), HHX_LT_WS))
                if ai < 0 or tuple(mod.lt_cfg(pid, ai)) != tuple(cfg):
                    print(f"[hhx] pins {layer}/m{m} {desc}: config {tuple(cfg)} rejected by cuBLASLt -> skipped", flush=True)
                    continue
            r = {"kind": "lt", "mod": mod, "pid": pid, "ai": ai, "w": wl, "w2": None, "ws": _HHX["ws"], "router": False,
                 "desc": desc, "align": 16 if nn else 256}
            if x.data_ptr() % r["align"]:
                print(f"[hhx] pins {layer}/m{m} {desc}: test x not {r['align']}-B aligned -> skipped", flush=True)
                continue
            got = _lt_call(r, x, n)
            got2 = _lt_call(r, x, n)
            torch.cuda.synchronize()
            ok, msg = _check(got, x, w, probes, False, ref, 5e-3)
            ok = ok and bool(torch.equal(got, got2))
            print(f"[hhx] pins {layer}/m{m} {desc} (lt{ai}{'' if ai < cnt else ' built'}): {msg} -> {'candidate' if ok else 'REJECTED'}", flush=True)
            if ok:
                out.append(r)
        except Exception as ex:  # noqa: BLE001
            print(f"[hhx] pins {layer}/m{m} {desc}: FALLBACK {repr(ex)[:200]}", flush=True)
    return out


def _hhx_dg1_route(layer, m, w, dev):
    mod = _HHX["dg1"]
    T, nwg, mt, s = HHX_DG1[(layer, m)]
    kid = HHX_DG1_INSTS.index((T, nwg, mt))
    info = [int(v) for v in mod.dg1_info(kid, s)]
    n, k = w.shape
    if n % info[3] or m > T or k % 64:
        print(f"[hhx] dg1 {layer}/m{m}: geometry {HHX_DG1[(layer, m)]} does not fit W {tuple(w.shape)} -> skipped", flush=True)
        return None
    if layer not in _HHX["dg1w"]:
        _HHX["dg1w"][layer] = dg1_pack(w)
    return {"kind": "dg1", "mod": mod, "kid": kid, "w": _HHX["dg1w"][layer], "err": torch.zeros(4, dtype=torch.int32, device=dev),
            "S": s, "T": T, "N": n, "K": k, "desc": f"dg1_{T}.{nwg}.{mt}s{s}(bm{info[3]},r{info[5]})"}


def _hhx_dg1_phase(state, timer, floor, dev):
    """DG1 at m = 32 against the route vd_1 picked (GV, or SGLang's call when none beat it)."""
    summary = []
    if not HHX_ENABLE or torch.cuda.get_device_capability(dev)[0] < 9:
        return summary
    _hhx_join()
    if _HHX["dg1"] is None:
        print(f"[hhx] dg1 module unavailable ({_HHX['dg1_err']}) -> no dg1 routes", flush=True)
        return summary
    for (layer, m) in sorted(HHX_DG1):
        n, k = LAYERS[layer]
        w = state[layer]
        try:
            r = _hhx_dg1_route(layer, m, w, dev)
            if r is None:
                continue
            good = True
            for outl in (False, True):
                for mm in (m, m - 3):
                    x, probes = _test_input(mm, k, 5150 + mm + int(outl), dev, outl)
                    got = _route_call(r, x, n, layer)
                    got2 = _route_call(r, x, n, layer)
                    torch.cuda.synchronize()
                    ok, msg = _check(got, x, w, probes, False, None, 5e-3)
                    ok = ok and bool(torch.equal(got, got2)) and int(r["err"][0]) == 0
                    if not ok:
                        print(f"[hhx] dg1 {layer}/m{m} {r['desc']} SELF-TEST FAIL m={mm} outl={outl}: {msg}", flush=True)
                        good = False
                        break
                if not good:
                    break
            if not good:
                continue
            x, _ = _test_input(m, k, 4242 + m, dev)
            inc = state["exact"].get((layer, m)) or state["routes"].get((layer, _bucket(layer, m)))
            gs = [timer.graph(lambda: _stock(x, layer, n, state))]
            if inc is not None:
                gs.append(timer.graph(lambda: _route_call(inc, x, n, layer)))
            gs.append(timer.graph(lambda: _route_call(r, x, n, layer)))
            tt = timer.times(gs, GV_REPS)
            timer.release()
            sgl = max(tt[0] - floor, 0.1)
            inc_t = max(tt[1] - floor, 0.1) if inc is not None else sgl
            dg_t = max(tt[-1] - floor, 0.1)
            print(f"[hhx] T {layer} m={m}: sgl {sgl:.2f} | vd_1 route {inc['desc'] if inc is not None else 'SGLang'} {inc_t:.2f} | "
                  f"{r['desc']} {dg_t:.2f}", flush=True)
            if dg_t <= 0.97 * inc_t:
                state["exact"][(layer, m)] = r
                summary.append(f"{layer}/m{m}: dg1 {r['desc']} x{sgl / dg_t:.3f}")
            else:
                summary.append(f"{layer}/m{m}: dg1 not adopted (x{sgl / dg_t:.3f} vs incumbent x{sgl / inc_t:.3f})")
        except Exception as ex:  # noqa: BLE001
            print(f"[hhx] dg1 {layer}/m{m}: FALLBACK {repr(ex)[:200]}", flush=True)
    return summary



def prepare(weights):
    t_start = time.time()
    dev = weights["router"].device
    state = {name: w.contiguous() for name, w in weights.items()}
    try:
        from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
        _SGL["fn"] = _bf16_gemm_dispatch_impl
    except Exception as ex:  # noqa: BLE001
        print(f"[gv1] SGLang dispatch unavailable, fallback is F.linear: {ex!r}", flush=True)
    hi, lo = split_router(state["router"])
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["routes"] = {}
    state["exact"] = {}
    state["why"] = {}
    _SEEN.clear()
    if not GV_ENABLE:
        print("[gv1] GV disabled: SGLang path everywhere", flush=True)
        return state
    lt_box = {}
    lt_thread = None
    if LT_ENABLE:
        lt_thread = threading.Thread(target=_build_lt, args=(lt_box,))
        lt_thread.start()
    _hhx_start(dev)
    if VD_ENABLE and VD_BOARD and _VD["mod"] is None and torch.cuda.get_device_capability(dev)[0] >= 9:
        _VD["thread"] = threading.Thread(target=_vd_build_job)
        _VD["thread"].start()
    cap = torch.cuda.get_device_capability(dev)
    has_cluster = cap[0] >= 9
    # matrices the routes stream
    mats = {"qkv_gate": state["qkv_gate"], "o": state["o"], "pooler": state["pooler"],
            "unpooler": state["unpooler"], "unpooler_slice": state["unpooler"][:UNPOOL_SLICE].contiguous(),
            "router": router_virtual(hi, lo)}

    def mat_of(layer, b):
        if layer == "unpooler" and b in (1, 8):
            return "unpooler_slice"
        return layer

    # bases (per matrix; the router: hi rows / lo rows)
    bases = {}
    for name, wv in mats.items():
        if name == "router":
            fr = _frag(wv)
            h7 = (fr >> 8) & 0x7F
            bases[name] = (_window_base(h7[..., [0, 1, 4, 5]]), _window_base(h7[..., [2, 3, 6, 7]]))
            bases[(name, 2)] = (_window_base(h7[..., [0, 1, 4, 5]], 4), _window_base(h7[..., [2, 3, 6, 7]], 4))
            del fr, h7
        else:
            h7w = (wv.view(torch.int16).to(torch.int32) >> 8) & 0x7F
            b0, b4 = _window_base(h7w), _window_base(h7w, 4)
            bases[name] = (b0, b0)
            bases[(name, 2)] = (b4, b4)
            del h7w
    print(f"[gv1] exponent windows (h7 base; P11 keyed (name, 2)): {bases}", flush=True)
    # candidate template keys
    cands = {}
    for (layer, b), routes in CANDIDATES.items():
        lst = []
        for rt_ in routes:
            rt_ = _r9(rt_)
            key = _inst_key(layer, b, rt_)
            if key is None:
                print(f"[gv1] candidate {layer}/{b} {rt_}: geometry does not fit -> skipped", flush=True)
                continue
            if key[7] > 1 and not has_cluster:
                continue
            lst.append((rt_, key))
        if GV_FORCE_FMT is not None and any(rt_[0] == GV_FORCE_FMT for rt_, _ in lst):
            lst = [(rt_, key) for rt_, key in lst if rt_[0] == GV_FORCE_FMT]
        cands[(layer, b)] = lst
    allkeys, groups = [], {}
    for (layer, b), lst in cands.items():
        for rt_, key in lst:
            if key not in allkeys:
                allkeys.append(key)
            if key not in groups.setdefault(f"gv1_{layer}", []):
                groups[f"gv1_{layer}"].append(key)
    flags = ["-O3"]
    one = _build_all({"gv1_all": allkeys}, flags)
    if "gv1_all" in one:
        built = {g: one["gv1_all"] for g in groups}
    else:   # a bad instantiation should only cost its own layer
        print("[gv1] single-module build failed -> per-layer modules", flush=True)
        built = _build_all(groups, flags)
    print(f"[gv1] builds done at {time.time() - t_start:.1f}s ({len(allkeys)} kernels)", flush=True)
    timer = _Timer(dev)
    floor = timer.graph_floor()
    print(f"[gv1] graph floor {floor:.2f} us (subtracted from both sides, as the judge does)", flush=True)
    packs = {}
    verified = {}
    summary = []
    for (layer, b), lst in sorted(cands.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if time.time() - t_start > GV_PREPARE_BUDGET:
            state["why"][(layer, b)] = "prepare budget exhausted"
            print(f"[gv1] FALLBACK {layer}/{b}: prepare budget exhausted -> SGLang path", flush=True)
            continue
        mname = mat_of(layer, b)
        wv = mats[mname]
        n_out = 256 if layer == "router" else wv.shape[0]
        k = wv.shape[1]
        ref_w = state["router"] if layer == "router" else wv
        live = []
        refs = {}
        for rt_, key in lst:
            fmt, mt, wr, wk, ku, st, s, xp, nr = rt_
            gname = f"gv1_{layer}"
            if gname not in built:
                state["why"][(layer, b)] = "build failed"
                continue
            mod, ids = built[gname]
            desc = f"f{fmt}m{mt}({wr},{wk},{ku},{s})s{st}x{xp}" + (f"n{nr}" if nr > 1 else "")
            try:
                info = mod.gv_info(ids[key])
            except Exception as ex:  # noqa: BLE001
                print(f"[gv1] {layer}/{b} {desc}: info failed {ex!r}", flush=True)
                continue
            ctas = (wv.shape[0] // 16 // (wr * nr)) * s
            if info[1] > 64 or info[2] * 132 < ctas:
                print(f"[gv1] {layer}/{b} {desc}: regs {info[0]} spill {info[1]} occ {info[2]} for {ctas} CTAs "
                      f"-> skipped (spills > 64 B or > 1 wave)", flush=True)
                continue
            pk_key = (mname, fmt, wr, wk, ku, s, nr)
            try:
                if pk_key not in packs:
                    packs[pk_key] = pack(wv, fmt, wr, wk, ku, s, bases[(mname, 2)] if fmt == 2 else bases[mname], nr)
                pk = packs[pk_key]
                if pk_key not in verified:
                    un = torch.empty_like(wv)
                    mod.gv_unpack(pk["w"], pk["esc"], un, fmt, k, pk["G"], pk["nbx"], wr, wk, ku, pk["tb"], nr)
                    torch.cuda.synchronize()
                    if pk.get("esc_check") is not None:   # P11: the correction list restores the escaped weights
                        idx, wt = pk.pop("esc_check")
                        un.view(torch.int16).view(-1)[idx] = torch.where(wt >= 32768, wt - 65536, wt).to(torch.int16)
                    pk.pop("esc_check", None)
                    verified[pk_key] = bool(torch.equal(un.view(torch.int16), wv.view(torch.int16)))
                    if not verified[pk_key] or len(verified) <= 3:
                        print(f"[gv1] pack {mname} f{fmt}({wr},{wk},{ku},{s})n{nr}: {pk['w'].numel() * 4 / 1e6:.1f} MB, "
                              f"{pk['nesc']} escapes, bit-exact unpack {verified[pk_key]}", flush=True)
                if not verified[pk_key]:
                    continue
            except Exception as ex:  # noqa: BLE001
                print(f"[gv1] {layer}/{b} {desc}: pack failed -> skipped: {ex!r}", flush=True)
                continue
            r = {"mod": mod, "inst": ids[key], "w": pk["w"], "esc": pk["esc"], "eoff": pk["eoff"], "tb": pk["tb"],
                 "nbx": pk["nbx"],
                 "mt": mt, "slices": b > 32, "desc": desc + f"r{info[0]}", "kind": "gv"}
            # outputs vs FP64 at every token count of the bucket (judge recipe, with and without outlier channels)
            good = True
            for m in _route_ms(b):
                for outl in (False, True):
                    x, probes = _test_input(m, k, 977 + 31 * m + int(outl), dev, outl)
                    try:
                        got = _gv_call(r, x, n_out, torch.float32 if layer == "router" else torch.bfloat16)
                        got2 = _gv_call(r, x, n_out, torch.float32 if layer == "router" else torch.bfloat16)
                        torch.cuda.synchronize()
                        if (m, outl) not in refs:
                            refs[(m, outl)] = (x.double() @ ref_w.double().T).to(got.dtype).double()
                        ok, msg = _check(got, x, ref_w, probes, layer == "router", refs[(m, outl)])
                        ok = ok and bool(torch.equal(got, got2))
                    except Exception as ex:  # noqa: BLE001
                        ok, msg = False, f"EXC {ex!r}"
                    if not ok:
                        print(f"[gv1] SELF-TEST FAIL {layer}/{b} {desc} m={m} outl={outl}: {msg}", flush=True)
                        good = False
                        break
                if not good:
                    break
            if good:
                live.append(r)
        if not live:
            state["why"][(layer, b)] = "no candidate passed"
            print(f"[gv1] FALLBACK {layer}/{b}: no candidate passed -> SGLang path", flush=True)
            continue
        # time: candidates vs SGLang's call, cold L2, graph replays
        tms = (TIMED_M[b],)
        best, best_ratio, rows = None, None, []
        try:
            res = {id(r): [] for r in live}
            for m in tms:
                x, _ = _test_input(m, k, 4242 + m, dev)
                n_call = 256 if layer == "router" else _vrows(layer, m)
                gs = [timer.graph(lambda: _stock(x, layer, n_call, state))]
                for r in live:
                    gs.append(timer.graph(lambda r=r: _gv_call(r, x, n_call, torch.float32 if layer == "router"
                                                               else torch.bfloat16)))
                tt = timer.times(gs, GV_REPS)
                timer.release()
                base_net = max(tt[0] - floor, 0.1)
                for r, tr in zip(live, tt[1:]):
                    res[id(r)].append((max(tr - floor, 0.1), base_net))
                rows.append((m, tt))
            for r in live:
                ratio = math.exp(sum(math.log(a / bb) for a, bb in res[id(r)]) / len(res[id(r)]))
                r["ratio"] = ratio
                if best_ratio is None or ratio < best_ratio:
                    best, best_ratio = r, ratio
        except Exception as ex:  # noqa: BLE001
            print(f"[gv1] FALLBACK {layer}/{b}: timing failed -> SGLang path: {ex!r}", flush=True)
            state["why"][(layer, b)] = "timing failed"
            continue
        for m, tt in rows:
            cand = " | ".join(f"{r['desc']} {tr - floor:.2f}" for r, tr in zip(live, tt[1:]))
            print(f"[gv1] T {layer} m={m}: sgl {tt[0] - floor:.2f} | {cand}", flush=True)
        if best_ratio <= GV_MARGIN or not GV_AUTOTUNE:
            state["routes"][(layer, b)] = best
            summary.append(f"{layer}/{b}: GV {best['desc']} x{1 / best_ratio:.3f}")
        else:
            state["why"][(layer, b)] = f"slower than SGLang (best x{1 / best_ratio:.3f})"
            summary.append(f"{layer}/{b}: SGLang (GV best x{1 / best_ratio:.3f})")
            print(f"[gv1] FALLBACK {layer}/{b}: best GV x{1 / best_ratio:.3f} vs SGLang -> SGLang path", flush=True)
    if lt_thread is not None:
        lt_thread.join()
        if "mod" in lt_box:
            summary += _lt_search(state, lt_box["mod"], timer, floor, dev, t_start + 270.0)
        else:
            print(f"[gv1] FALLBACK cuBLASLt routes off (module build failed): {lt_box.get('err')}", flush=True)
    try:
        summary += _hhx_dg1_phase(state, timer, floor, dev)
    except Exception as ex:  # noqa: BLE001
        print(f"[hhx] dg1 phase raised {ex!r}", flush=True)
    # free packs that no route uses
    used = {id(r["w"]) for r in list(state["routes"].values()) + list(state["exact"].values()) if r["kind"] == "gv"}
    for kk in list(packs):
        if id(packs[kk]["w"]) not in used:
            del packs[kk]
    timer.release()
    del timer
    print(f"[gv1] ROUTES ({time.time() - t_start:.1f}s): " + "; ".join(summary), flush=True)
    return state

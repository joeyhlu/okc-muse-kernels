# okc: problem=muse_bf16_proj
# nonce ps-ctl-v1-1
"""PS v1 CONTROL (notes/study/dense_independent.md section 6, re-opened control-first). CHECK ONLY, never submit for score.

Pure stream of P12-sized packs (1536 B per 16 x 64 super-block, per-tile contiguous) on the stream lab's geometry:
264 CTAs = 2 per SM x 16 warps (32 warps/SM), LDG register streaming D SBs per warp, x slice staged in smem ONCE per
CTA, K split S = 8 (CTA = (row group, K slice), 33 groups with 7-8 / 4-5 / 15-16 tiles), XOR consume (no decode, no
mma), per-CTA FP32 partial slots, optional deterministic last-arriver fixup (counter, rank-ordered sum, BF16 store).
kernel_fn is SGLang's path everywhere (this file only measures, in prepare()).
GO for the real v1 (pre-registered): pooler T=1 <= 13.3 us (judge units) and T=8 / T=32 increments <= 0.3 / 1.0 us.
"""

from typing import TypedDict

import hashlib
import statistics
import time

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


ROUTER_FP32_MAX_TOKENS = 8
FLUSH_FLOATS = (256 << 20) // 4
REPS = 41
LAB_DEADLINE_S = 200.0

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void ctl_run(torch::Tensor x, torch::Tensor pk, torch::Tensor ws, torch::Tensor cnt, torch::Tensor out, int64_t M,
             int64_t K, int64_t nt, int64_t S, int64_t G, int64_t fix, int64_t dep);
std::vector<int64_t> ctl_info(int64_t dep, int64_t smem);
void ctl_empty();
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <vector>

namespace ctl {
constexpr int NW = 16;
struct Args {
  const __nv_bfloat16* x; int M; int K;
  const uint8_t* pk; int nt; int nsb; int Q; int S; int G;
  float* ws; int* cnt; __nv_bfloat16* out; int fix;
};
__device__ __forceinline__ uint4 ldw(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
template <int D>
__global__ void __launch_bounds__(NW * 32, 2) ctl_kernel(const __grid_constant__ Args A) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ int s_last;
  const int tid = (int)threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int ngrp = A.G / A.S;
  const int grp = (int)blockIdx.x / A.S, ks = (int)blockIdx.x - grp * A.S;
  const int t0 = grp * A.nt / ngrp, t1 = (grp + 1) * A.nt / ngrp;
  const int R = t1 - t0, Q = A.Q, RQ = R * Q;
  const int s_lo = warp * RQ / NW, s_hi = (warp + 1) * RQ / NW;
  // 1. weights first: D SBs per warp in registers (tile-major contiguous run of this CTA's (tile, K-slice) SBs)
  uint4 w[D][3];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (s_lo + d < s_hi) {
      const int s = s_lo + d, tl = s / Q, sb = ks * Q + (s - tl * Q);
      const uint8_t* p = A.pk + ((long long)(t0 + tl) * A.nsb + sb) * 1536 + lane * 16;
      w[d][0] = ldw(p); w[d][1] = ldw(p + 512); w[d][2] = ldw(p + 1024);
    }
  // 2. x slice [M][64 Q] bf16 -> smem once (16 B per thread)
  const int rowv = Q * 8;   // uint4 per x row slice
  uint4* xs = reinterpret_cast<uint4*>(smem);
  for (int i = tid; i < A.M * rowv; i += NW * 32) {
    const int r = i / rowv, c = i - r * rowv;
    xs[i] = reinterpret_cast<const uint4*>(A.x + (size_t)r * A.K + (size_t)ks * Q * 64)[c];
  }
  __syncthreads();
  // 3. stream: XOR consume + one x word per SB (keeps the smem x live, like the B-fragment reads)
  uint32_t px = 0u;
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(smem);
  for (int s0 = s_lo; s0 < s_hi; s0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int s = s0 + d;
      if (s < s_hi) {
        px ^= w[d][0].x ^ w[d][0].y ^ w[d][0].z ^ w[d][0].w ^ w[d][1].x ^ w[d][1].y ^ w[d][1].z ^ w[d][1].w;
        px ^= w[d][2].x ^ w[d][2].y ^ w[d][2].z ^ w[d][2].w;
        px ^= xw[((s % Q) * 32 + lane) % (A.M * Q * 32)];
        const int sn = s + D;
        if (sn < s_hi) {
          const int tl = sn / Q, sb = ks * Q + (sn - tl * Q);
          const uint8_t* p = A.pk + ((long long)(t0 + tl) * A.nsb + sb) * 1536 + lane * 16;
          w[d][0] = ldw(p); w[d][1] = ldw(p + 512); w[d][2] = ldw(p + 1024);
        }
      }
    }
  }
  // 4. this CTA's FP32 partial slot ws[ks][row][tok] for its R x 16 rows (the real v1 writes the same bytes)
  const int N = A.nt * 16, rows = R * 16;
  const float v = __uint_as_float(px & 0x007FFFFFu);
  for (int i = tid; i < rows * A.M; i += NW * 32) {
    const int r = i / A.M, t = i - r * A.M;
    __stcg(A.ws + ((size_t)ks * N + t0 * 16 + r) * A.M + t, v);
  }
  if (!A.fix) return;
  // 5. deterministic fixup: the last CTA of the group sums the S slots in rank order and stores BF16
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(A.cnt + grp, 1) == A.S - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  for (int i = tid; i < rows * A.M; i += NW * 32) {
    const int r = i / A.M, t = i - r * A.M;
    float a = 0.f;
    for (int q = 0; q < A.S; ++q) a += __ldcg(A.ws + ((size_t)q * N + t0 * 16 + r) * A.M + t);
    A.out[(size_t)t * N + t0 * 16 + r] = __float2bfloat16_rn(a);
  }
  if (tid == 0) A.cnt[grp] = 0;
}
__global__ void empty_kernel() {}
}  // namespace ctl

template <int D>
static void launch_t(const ctl::Args& A, int smem) {
  auto kern = ctl::ctl_kernel<D>;
  static bool set = false;
  if (!set) { C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024)); set = true; }
  kern<<<A.G, ctl::NW * 32, smem, at::cuda::getCurrentCUDAStream()>>>(A);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ctl_run(torch::Tensor x, torch::Tensor pk, torch::Tensor ws, torch::Tensor cnt, torch::Tensor out, int64_t M,
             int64_t K, int64_t nt, int64_t S, int64_t G, int64_t fix, int64_t dep) {
  TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16 && K % (64 * S) == 0 && G % S == 0, "ctl: shapes");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  ctl::Args A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.K = (int)K;
  A.pk = pk.data_ptr<uint8_t>(); A.nt = (int)nt; A.nsb = (int)(K / 64); A.Q = (int)(K / 64 / S); A.S = (int)S; A.G = (int)G;
  A.ws = ws.data_ptr<float>(); A.cnt = cnt.data_ptr<int>(); A.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  A.fix = (int)fix;
  const int smem = (int)(M * A.Q * 128);
  TORCH_CHECK(smem <= 100 * 1024, "ctl: smem");
  if (dep == 1) launch_t<1>(A, smem);
  else if (dep == 2) launch_t<2>(A, smem);
  else if (dep == 3) launch_t<3>(A, smem);
  else TORCH_CHECK(false, "ctl: dep");
}

std::vector<int64_t> ctl_info(int64_t dep, int64_t smem) {
  auto kern = dep == 1 ? ctl::ctl_kernel<1> : dep == 2 ? ctl::ctl_kernel<2> : ctl::ctl_kernel<3>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, ctl::NW * 32, (size_t)smem));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ};
}

void ctl_empty() {
  ctl::empty_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_MOD = None


def _build():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
        tag = hashlib.sha1((CPP_SRC + CUDA_SRC).encode()).hexdigest()[:10]
        t0 = time.time()
        _MOD = load_inline(name=f"ps_ctl_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                           functions=["ctl_run", "ctl_info", "ctl_empty"], extra_cuda_cflags=flags, verbose=False)
        print(f"CTL build {time.time() - t0:.1f}s", flush=True)
    return _MOD


def _baseline(state, x, layer, n):
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


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


def _time(L, fn):
    g = _graph(fn)
    ts = []
    for r in range(REPS):
        L["flush"].sum()
        L["e0"].record()
        g.replay()
        L["e1"].record()
        torch.cuda.synchronize()
        if r:
            ts.append(L["e0"].elapsed_time(L["e1"]) * 1000.0)
    return statistics.median(ts)


def prepare(weights):
    t0 = time.time()
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    try:
        _lab(state, t0)
    except Exception as ex:  # noqa: BLE001
        print(f"CTL lab ERR {ex!r}"[:800], flush=True)
    torch.cuda.synchronize()
    return state


def _lab(state, t0):
    mod = _build()
    dev = "cuda"
    L = {"flush": torch.ones(FLUSH_FLOATS, dtype=torch.float32, device=dev),
         "e0": torch.cuda.Event(enable_timing=True), "e1": torch.cuda.Event(enable_timing=True)}
    floor = _time(L, lambda: mod.ctl_empty())
    print(f"CTL floor (empty-kernel graph) {floor:.2f} us", flush=True)
    for dep in (1, 2, 3):
        print(f"CTL info D{dep}: regs/local/CTAs-per-SM at 64 KB smem {list(mod.ctl_info(dep, 65536))}", flush=True)
    mats = {"pooler": (4096, 8192), "qkv_gate": (2304, 8192), "o": (8192, 1024)}
    res = {}
    for name, (N, K) in mats.items():
        nt = N // 16
        pk = torch.randint(0, 256, (nt * (K // 64) * 1536 + 64,), dtype=torch.uint8, device=dev)
        ws = torch.zeros(8 * N * 32, dtype=torch.float32, device=dev)
        cnt = torch.zeros(512, dtype=torch.int32, device=dev)
        for m in (1, 8, 16, 32):
            if time.time() - t0 > LAB_DEADLINE_S:
                break
            x = torch.randn((m, K), device=dev).to(torch.bfloat16)
            out = torch.empty((m, N), dtype=torch.bfloat16, device=dev)
            res[(name, m, "cublas")] = _time(L, lambda x=x, N=N, name=name: _bf16_gemm_dispatch_impl(x, state[name][:N], None)) - floor
            S = 8 if K >= 8192 else 1
            G = 264 if K >= 8192 else 256
            for dep, fix in ((2, 1), (2, 0), (1, 1), (3, 1)):
                key = (name, m, f"D{dep}{'fix' if fix else 'nofix'}")
                res[key] = _time(L, lambda x=x, pk=pk, ws=ws, cnt=cnt, out=out, m=m, dep=dep, fix=fix, S=S, G=G:
                                 mod.ctl_run(x, pk, ws, cnt, out, m, K, nt, S, G, fix, dep)) - floor
            print(f"CTL {name:9s} T={m:<2d} cublas {res[(name, m, 'cublas')]:6.2f} | " +
                  " ".join(f"{k[2]} {v:6.2f}" for k, v in res.items() if k[0] == name and k[1] == m and k[2] != "cublas"),
                  flush=True)
    for name in mats:
        for arm in ("D2fix", "D1fix", "D3fix", "D2nofix"):
            vals = [res.get((name, m, arm)) for m in (1, 8, 16, 32)]
            if None not in vals:
                print(f"CTL slope {name:9s} {arm:7s} T1 {vals[0]:6.2f}  +T8 {vals[1] - vals[0]:+5.2f}  +T16 {vals[2] - vals[0]:+5.2f}"
                      f"  +T32 {vals[3] - vals[0]:+5.2f}", flush=True)
    print(f"CTL done {time.time() - t0:.1f}s", flush=True)


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    return _baseline(state, x, cfg["layer"], cfg["n"])

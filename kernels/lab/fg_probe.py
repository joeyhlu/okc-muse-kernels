"""fg_probe.py: pure K+V gather microbench on the judge's real full-attention caches (R-fg-1, notes/memory/gpu_requests.md).

    python3 -u kernels/lab/fg_probe.py [--shapes=d16,d32,v16,v32] [--reps=7] [--out=box_out/fg/probe.json]

Inputs: reference.generate_input of the four scored full-layer B16/B32 loads (seeds 604/606/610/612, page 1, agentic,
coherent). Every arm reads every cached key's K row and V row (128 B each: 2 kv heads x 64 FP8) exactly once and returns
the u64 sum of their u32 words (checked against torch). Cold L2 before every sample (256 MB sum, as tools/judge).
Kernels (256 threads, balanced flat partition of the row list over the grid, one wave):
  ring ST/CPS : VA's loader skeleton: whole 128-B rows, 8 lanes per row, K and V of a key from the same thread, cp.async.cg
                16 B into an ST x 16 KB ring (TILE 64 keys), row indices prefetched one tile ahead, CPS CTAs per SM.
  ldg U       : register gather, ld.global.nc.L1::no_allocate.v4, double-buffered U rows per thread (2U..4U loads in flight).
  stream      : dense read of the whole K and V pools (incl. unused rows): the HBM streaming ceiling.
Row-list orders: judge (page-table order), t256 / t4096 (sorted inside 256 / 4096-key chunks), sreq (each request
sorted: concurrent CTAs of all requests sweep the same physical region = a "front"), srot (sorted, per-request random
rotation: same per-CTA locality, no front), glob (all rows in physical order: a stream with gaps).
Also times torch's segmented sort of the page table (the cost of building sreq inside the timed call).
"""
import argparse
import importlib.util
import json
import os
import pathlib
import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROB = ROOT / "problems" / "muse_attention_decode"
SHAPES = {"d16": (16, 1, 604), "d32": (32, 1, 606), "v16": (16, 8, 610), "v32": (32, 8, 612)}

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>

namespace fg {
typedef unsigned long long u64;
__device__ __forceinline__ void cpa16(uint32_t sa, const void* g, int sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(sz) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ u64 s4(uint4 v) { return (u64)v.x + (u64)v.y + (u64)v.z + (u64)v.w; }
__device__ __forceinline__ uint4 ldnc(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// CTA-wide u64 sum, one atomic per CTA (R-fg-1 v1 did one same-address atomic per THREAD: ~0.74 ns each, serialized,
// = +25..150 us per launch; v2 numbers are direct).
__device__ __forceinline__ void cta_add(u64* out, u64 acc) {
  __shared__ u64 red[8];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    u64 x = 0;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) x += red[w];
    atomicAdd(out, x);
  }
}
__device__ __forceinline__ void cpa4(uint32_t sa, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(sa), "l"(g) : "memory");
}

// ring<ST, D>: D = row-index prefetch distance in tiles (1 = VA / VFA today, 2 = two tiles ahead in registers,
// 0 = the CTA's whole index segment staged in smem by one 4-B cp.async burst before the ring starts).
template <int ST, int D>
__global__ void __launch_bounds__(256) ring_k(const int* __restrict__ rows, long long N, const uint8_t* __restrict__ kc,
                                             const uint8_t* __restrict__ vc, u64* out) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int t = threadIdx.x, ch = t & 7, r0 = t >> 3;
  const long long a = N * blockIdx.x / gridDim.x, e = N * (blockIdx.x + 1) / gridDim.x;
  const int nseg = (int)(e - a), nt = (nseg + 63) / 64;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  int* sidx = reinterpret_cast<int*>(sm + ST * 16384);
  if (D == 0) {
    const uint32_t sb = base + ST * 16384;
    for (int i = t; i < nseg; i += 256) cpa4(sb + 4u * i, rows + a + i);
    cpa_commit();
    cpa_wait<0>();
    __syncthreads();
  }
  const uint32_t so0 = (uint32_t)(r0 * 128 + 16 * ch), so1 = so0 + 4096u;
  u64 acc = 0;
  auto gidx = [&](int j, int h) -> int {
    const int k = j * 64 + r0 + 32 * h;
    if (j >= nt || k >= nseg) return -1;
    return D == 0 ? sidx[k] : __ldg(rows + a + k);
  };
  int c0 = gidx(0, 0), c1 = gidx(0, 1), n0 = -1, n1 = -1;
  if (D == 2) { n0 = gidx(1, 0); n1 = gidx(1, 1); }
  auto issue = [&](int j) {
    const uint32_t st = base + (uint32_t)((j % ST) * 16384);
    const size_t g0 = c0 >= 0 ? (size_t)c0 * 128 + 16 * ch : 0, g1 = c1 >= 0 ? (size_t)c1 * 128 + 16 * ch : 0;
    cpa16(st + so0, kc + g0, c0 >= 0 ? 16 : 0);
    cpa16(st + 8192u + so0, vc + g0, c0 >= 0 ? 16 : 0);
    cpa16(st + so1, kc + g1, c1 >= 0 ? 16 : 0);
    cpa16(st + 8192u + so1, vc + g1, c1 >= 0 ? 16 : 0);
    cpa_commit();
    if (D == 2) { c0 = n0; c1 = n1; n0 = gidx(j + 2, 0); n1 = gidx(j + 2, 1); }
    else { c0 = gidx(j + 1, 0); c1 = gidx(j + 1, 1); }
  };
#pragma unroll
  for (int j = 0; j < ST - 1; ++j) issue(j);
  for (int j = 0; j < nt; ++j) {
    issue(j + ST - 1);
    cpa_wait<ST - 1>();
    const uint8_t* st = sm + (j % ST) * 16384;
    acc += s4(*reinterpret_cast<const uint4*>(st + so0)) + s4(*reinterpret_cast<const uint4*>(st + 8192 + so0)) +
           s4(*reinterpret_cast<const uint4*>(st + so1)) + s4(*reinterpret_cast<const uint4*>(st + 8192 + so1));
  }
  cpa_wait<0>();
  cta_add(out, acc);
}

template <int U>
__global__ void __launch_bounds__(256) ldg_k(const int* __restrict__ rows, long long N, const uint8_t* __restrict__ kc,
                                            const uint8_t* __restrict__ vc, u64* out) {
  const int t = threadIdx.x, ch = t & 7, r0 = t >> 3;
  const long long a = N * blockIdx.x / gridDim.x, e = N * (blockIdx.x + 1) / gridDim.x;
  constexpr int SK = 32 * U;
  const int ns = (int)((e - a + SK - 1) / SK);
  u64 acc = 0;
  int ix[U];
  uint4 ka[U], va[U];
  auto ldidx = [&](int s) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long k = a + (long long)s * SK + r0 + 32 * u;
      ix[u] = (s < ns && k < e) ? __ldg(rows + k) : -1;
    }
  };
  auto ld = [&](uint4* kk, uint4* vv) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (ix[u] >= 0) {
        const size_t g = (size_t)ix[u] * 128 + 16 * ch;
        kk[u] = ldnc(kc + g);
        vv[u] = ldnc(vc + g);
      } else {
        kk[u] = make_uint4(0u, 0u, 0u, 0u);
        vv[u] = kk[u];
      }
    }
  };
  ldidx(0);
  ld(ka, va);
  ldidx(1);
  for (int s = 0; s < ns; ++s) {
    uint4 kb[U], vb[U];
    ld(kb, vb);
    ldidx(s + 2);
#pragma unroll
    for (int u = 0; u < U; ++u) acc += s4(ka[u]) + s4(va[u]);
#pragma unroll
    for (int u = 0; u < U; ++u) { ka[u] = kb[u]; va[u] = vb[u]; }
  }
  cta_add(out, acc);
}

__global__ void __launch_bounds__(256) stream_k(const uint4* __restrict__ k, const uint4* __restrict__ v, long long n, u64* out) {
  u64 acc = 0;
  const long long G = (long long)gridDim.x * 256;
  for (long long i = (long long)blockIdx.x * 256 + threadIdx.x; i < n; i += 4 * G) {
    uint4 x[8];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const long long j = i + u * G;
      x[2 * u] = j < n ? ldnc(k + j) : make_uint4(0u, 0u, 0u, 0u);
      x[2 * u + 1] = j < n ? ldnc(v + j) : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) acc += s4(x[u]);
  }
  cta_add(out, acc);
}

// bulk ST: one cp.async.bulk (TMA 1D, 128 B, mbarrier complete_tx) per K row and per V row, issued by warp 0 (2 keys
// per lane per tile), consumed by warps 1-7 (they sum only the tile's valid rows), ST x 16 KB ring, full/empty mbarriers.
__device__ __forceinline__ uint32_t sa_of(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_wait(uint32_t b, int par) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra W;\n}\n" ::"r"(b), "r"(par) : "memory");
}
template <int ST>
__global__ void __launch_bounds__(256) bulk_k(const int* __restrict__ rows, long long N, const uint8_t* __restrict__ kc,
                                             const uint8_t* __restrict__ vc, u64* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ __align__(8) unsigned long long full[ST], empt[ST];
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const long long a = N * blockIdx.x / gridDim.x, e = N * (blockIdx.x + 1) / gridDim.x;
  const int nt = (int)((e - a + 63) / 64);
  if (t < ST) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(sa_of(full + t)) : "memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 7;\n" ::"r"(sa_of(empt + t)) : "memory");
  }
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  __syncthreads();
  u64 acc = 0;
  if (warp == 0) {
    int r0 = (a + lane < e) ? __ldg(rows + a + lane) : 0, r1 = (a + lane + 32 < e) ? __ldg(rows + a + lane + 32) : 0;
    for (int j = 0; j < nt; ++j) {
      const int st = j % ST;
      if (j >= ST) mb_wait(sa_of(empt + st), ((j / ST) - 1) & 1);
      const long long k0 = a + (long long)j * 64;
      const int nv = (int)min((long long)64, e - k0);
      if (lane == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(sa_of(full + st)), "r"(nv * 256) : "memory");
      __syncwarp();
      const int c0 = r0, c1 = r1;
      const long long n0 = k0 + 64 + lane, n1 = n0 + 32;   // next tile's indices, one tile ahead
      r0 = (j + 1 < nt && n0 < e) ? __ldg(rows + n0) : 0;
      r1 = (j + 1 < nt && n1 < e) ? __ldg(rows + n1) : 0;
      const uint32_t sk = sa_of(sm + st * 16384), fb = sa_of(full + st);
      if (lane < nv) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];\n"
                     ::"r"(sk + lane * 128), "l"(kc + (size_t)c0 * 128), "r"(fb) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];\n"
                     ::"r"(sk + 8192 + lane * 128), "l"(vc + (size_t)c0 * 128), "r"(fb) : "memory");
      }
      if (lane + 32 < nv) {
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];\n"
                     ::"r"(sk + (lane + 32) * 128), "l"(kc + (size_t)c1 * 128), "r"(fb) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];\n"
                     ::"r"(sk + 8192 + (lane + 32) * 128), "l"(vc + (size_t)c1 * 128), "r"(fb) : "memory");
      }
    }
  } else {
    const int c = t - 32;
    for (int j = 0; j < nt; ++j) {
      const int st = j % ST;
      const int nv = (int)min((long long)64, e - (a + (long long)j * 64));
      mb_wait(sa_of(full + st), (j / ST) & 1);
      const uint8_t* s = sm + st * 16384;
      for (int q = c; q < 1024; q += 224) {
        if (((q & 511) >> 3) < nv) acc += s4(*reinterpret_cast<const uint4*>(s + q * 16));
      }
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(sa_of(empt + st)) : "memory");
    }
  }
  cta_add(out, acc);
#endif
}

template <typename F> int occ_of(F f, int smem) {
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int n = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, f, 256, smem);
  return n;
}
}  // namespace fg

#define FG_RING(S, D) case 10 * S + D: fg::occ_of(fg::ring_k<S, D>, smem); fg::ring_k<S, D><<<grid, 256, smem, s>>>(R, N, K, V, O); break;
#define FG_BULK(S) case S: fg::occ_of(fg::bulk_k<S>, smem); fg::bulk_k<S><<<grid, 256, smem, s>>>(R, N, K, V, O); break;
int fg_launch(int kind, int p, int grid, int smem, const int* R, long long N, const uint8_t* K, const uint8_t* V,
              unsigned long long* O, long long n16, cudaStream_t s) {
  if (kind == 0) { switch (p) { FG_RING(4, 1) FG_RING(4, 2) FG_RING(4, 0) FG_RING(8, 0) FG_RING(3, 0) FG_RING(2, 1) FG_RING(2, 2) FG_RING(2, 0) default: return -1; } }
  else if (kind == 1) { fg::occ_of(fg::ldg_k<2>, smem); fg::occ_of(fg::ldg_k<4>, smem); if (p == 2) fg::ldg_k<2><<<grid, 256, smem, s>>>(R, N, K, V, O); else if (p == 4) fg::ldg_k<4><<<grid, 256, smem, s>>>(R, N, K, V, O); else return -1; }
  else if (kind == 2) fg::stream_k<<<grid, 256, 0, s>>>((const uint4*)K, (const uint4*)V, n16, O);
  else if (kind == 3) { switch (p) { FG_BULK(4) FG_BULK(6) FG_BULK(8) default: return -1; } }
  return (int)cudaGetLastError();
}
int fg_occ(int kind, int p, int smem) {
  if (kind == 0) { switch (p) {
    case 41: return fg::occ_of(fg::ring_k<4, 1>, smem); case 42: return fg::occ_of(fg::ring_k<4, 2>, smem);
    case 40: return fg::occ_of(fg::ring_k<4, 0>, smem); case 80: return fg::occ_of(fg::ring_k<8, 0>, smem);
    case 30: return fg::occ_of(fg::ring_k<3, 0>, smem); case 21: return fg::occ_of(fg::ring_k<2, 1>, smem);
    case 22: return fg::occ_of(fg::ring_k<2, 2>, smem); case 20: return fg::occ_of(fg::ring_k<2, 0>, smem); } return -1; }
  if (kind == 1) return p == 2 ? fg::occ_of(fg::ldg_k<2>, smem) : fg::occ_of(fg::ldg_k<4>, smem);
  if (kind == 3) { switch (p) { case 4: return fg::occ_of(fg::bulk_k<4>, smem); case 6: return fg::occ_of(fg::bulk_k<6>, smem);
    case 8: return fg::occ_of(fg::bulk_k<8>, smem); } return -1; }
  return fg::occ_of(fg::stream_k, 0);
}
"""
CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>
int fg_launch(int kind, int p, int grid, int smem, const int* R, long long N, const uint8_t* K, const uint8_t* V,
              unsigned long long* O, long long n16, cudaStream_t s);
int fg_occ(int kind, int p, int smem);
void launch(int64_t kind, int64_t p, int64_t grid, int64_t smem, torch::Tensor rows, torch::Tensor kc, torch::Tensor vc,
            torch::Tensor out) {
  const int* R = rows.numel() ? rows.data_ptr<int>() : nullptr;
  const int rc = fg_launch((int)kind, (int)p, (int)grid, (int)smem, R, rows.numel(), (const uint8_t*)kc.data_ptr(),
                           (const uint8_t*)vc.data_ptr(), (unsigned long long*)out.data_ptr(), kc.numel() / 16,
                           at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(rc == 0, "fg_launch rc ", rc);
}
int64_t occ(int64_t kind, int64_t p, int64_t smem) { return fg_occ((int)kind, (int)p, (int)smem); }
"""
SMEM_FOR_CPS = {1: 150 * 1024, 2: 100 * 1024, 3: 70 * 1024, 4: 52 * 1024, 5: 40 * 1024, 6: 34 * 1024}
INT_MAX = 2 ** 31 - 1


def load_ref():
    sys.path.insert(0, str(PROB))
    spec = importlib.util.spec_from_file_location("fg_reference", str(PROB / "reference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build():
    cc = torch.cuda.get_device_capability()
    arch = os.environ.get("FG_ARCH") or f"-arch=sm_{cc[0]}{cc[1]}"
    t0 = time.time()
    m = load_inline("fg_probe_v2", CPP_SRC, cuda_sources=CUDA_SRC, functions=["launch", "occ"],
                    extra_cuda_cflags=["-O3", arch, "-lineinfo"], verbose=False)
    print(f"[fg] build {time.time() - t0:.1f}s {arch}", flush=True)
    return m


def orders(data, gen):
    pt, sl = data["page_table"], data["seq_lens"].long()
    B, maxp = pt.shape
    mask = torch.arange(maxp, device=pt.device)[None, :] < sl[:, None]
    judge = pt[mask].contiguous()
    pts = torch.where(mask, pt, torch.full_like(pt, INT_MAX)).sort(dim=1).values
    sreq = pts[mask].contiguous()
    parts = []
    for b in range(B):
        n = int(sl[b])
        parts.append(torch.roll(pts[b, :n], int(torch.randint(0, n, (1,), generator=gen))))
    srot = torch.cat(parts).contiguous()
    glob = judge.sort().values.contiguous()
    out = {"judge": judge, "sreq": sreq, "srot": srot, "glob": glob}
    n = judge.numel()
    for c in (256, 4096):
        m = (n + c - 1) // c * c
        x = torch.full((m,), INT_MAX, dtype=judge.dtype, device=judge.device)
        x[:n] = judge
        out[f"t{c}"] = x.view(-1, c).sort(dim=1).values.flatten()[:n].contiguous()
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="d16,d32,v16,v32")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="box_out/fg/probe.json")
    ap.add_argument("--ctx", type=int, default=80000)
    ap.add_argument("--bulk", action="store_true", help="sm_90 only: the per-row cp.async.bulk arms (separate process)")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = load_ref()
    m = build()
    if args.build_only:
        return
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    flush_buf = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    out = torch.zeros(1, dtype=torch.int64, device=dev)
    # arms: (name, kind, param, cps, order); ring param = 10 * ST + D (D: index prefetch distance, 0 = smem-staged)
    arms = [("ring4x2", 0, 41, 2, "judge"), ("ring4x2d2", 0, 42, 2, "judge"), ("ring4x2ds", 0, 40, 2, "judge"),
            ("ring4x1", 0, 41, 1, "judge"), ("ring4x1ds", 0, 40, 1, "judge"), ("ring8x1ds", 0, 80, 1, "judge"),
            ("ring3x3ds", 0, 30, 3, "judge"), ("ring2x4ds", 0, 20, 4, "judge"), ("ring2x4d2", 0, 22, 4, "judge"),
            ("ring2x6", 0, 21, 6, "judge"), ("ring2x6d2", 0, 22, 6, "judge"), ("ldg4x2", 1, 4, 2, "judge"),
            ("ldg4x3", 1, 4, 3, "judge"), ("ldg4x4", 1, 4, 4, "judge"), ("ldg2x4", 1, 2, 4, "judge"),
            ("ldg2x5", 1, 2, 5, "judge"), ("ldg4x4_t4096", 1, 4, 4, "t4096"), ("ldg4x4_sreq", 1, 4, 4, "sreq"),
            ("ring4x2ds_sreq", 0, 40, 2, "sreq"), ("stream", 2, 0, 6, None), ("ring4x2_b", 0, 41, 2, "judge")]
    if args.bulk:
        arms = [("ring4x2", 0, 41, 2, "judge"), ("bulk4x2", 3, 4, 2, "judge"), ("bulk6x2", 3, 6, 2, "judge"),
                ("bulk4x3", 3, 4, 3, "judge"), ("ring4x2_b", 0, 41, 2, "judge")]

    def plan(nkeys):
        """(occupancy, smem) per arm for this shape; residency is forced to cps by the smem request."""
        occs = {}
        for name, kind, p, cps, _ in arms:
            budget = min(227 * 1024, (228 * 1024) // cps - 1024)
            if kind == 2:
                smem = 0
            elif kind == 1:
                smem = SMEM_FOR_CPS[cps]
            else:
                st = p // 10 if kind == 0 else p
                req = st * 16384
                if kind == 0 and p % 10 == 0:
                    req += 16 * ((nkeys // (sms * cps) + 2) * 4 // 16 + 1)
                smem = max(req, SMEM_FOR_CPS[cps])
                if smem > budget:
                    smem = req
            if os.environ.get("FG_LOCAL") and kind in (0, 3):
                smem = (p // 10 if kind == 0 else p) * 16384 + (16 * ((nkeys // sms + 2) * 4 // 16 + 1) if kind == 0 and p % 10 == 0 else 0)
            o = int(m.occ(kind, p, smem)) if smem <= budget or os.environ.get("FG_LOCAL") else 0
            if not os.environ.get("FG_LOCAL") and kind != 2 and o != cps:
                o = 0
            occs[name] = (o, smem)
        return occs

    results = {"sms": sms, "device": torch.cuda.get_device_name(), "shapes": {}}
    gen = torch.Generator().manual_seed(1234)
    for sh in args.shapes.split(","):
        B, L, seed = SHAPES[sh]
        t0 = time.time()
        data = ref.generate_input(batch=B, ctx=args.ctx, heads="tp8", layer="full", seed=seed, q_len=L,
                                  spread="agentic", dist="coherent", page=1)
        kc = data["k_cache"].view(torch.uint8)
        vc = data["v_cache"].view(torch.uint8)
        total = kc.shape[0]
        lists, mask = orders(data, gen)
        nkeys = lists["judge"].numel()
        rk = (kc.reshape(total, -1).view(torch.int32).to(torch.int64) & 0xFFFFFFFF).sum(1)
        rv = (vc.reshape(total, -1).view(torch.int32).to(torch.int64) & 0xFFFFFFFF).sum(1)
        want = int(rk[lists["judge"].long()].sum() + rv[lists["judge"].long()].sum())
        want_all = int(rk.sum() + rv.sum())
        del rk, rv
        torch.cuda.synchronize()
        print(f"[fg] {sh}: B={B} L={L} keys={nkeys} pool={total} gen {time.time() - t0:.1f}s", flush=True)
        occs = plan(nkeys)
        skipped = [a[0] for a in arms if occs[a[0]][0] <= 0]
        if skipped:
            print(f"[fg] {sh} skipped (residency not met): {' '.join(skipped)}", flush=True)
        sarms = [a for a in arms if occs[a[0]][0] > 0]

        empty = torch.empty(0, dtype=torch.int32, device=dev)

        def launch(arm):
            name, kind, p, cps, order = arm
            o, smem = occs[name]
            m.launch(kind, p, sms * o, smem, lists[order] if order else empty, kc, vc, out)

        samples = {a[0]: [] for a in sarms}
        ok = {}
        for a in sarms:  # warm + checksum
            out.zero_()
            launch(a)
            torch.cuda.synchronize()
            got = int(out.item())
            ok[a[0]] = got == (want_all if a[1] == 2 else want)
        for _ in range(args.reps):
            for a in sarms:
                torch.sum(flush_buf, dim=0, out=sink)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                launch(a)
                e1.record()
                samples[a[0]].append((e0, e1))
        # the cost of building sreq inside the timed call: torch's segmented sort of the masked page table
        pt = data["page_table"]
        sort_s = []
        for _ in range(args.reps):
            torch.sum(flush_buf, dim=0, out=sink)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            torch.where(mask, pt, torch.full_like(pt, INT_MAX)).sort(dim=1)
            e1.record()
            sort_s.append((e0, e1))
        torch.cuda.synchronize()
        res = {}
        mb = nkeys * 256 / 1e6
        base = None
        for a in sarms:
            us = [x.elapsed_time(y) * 1e3 for x, y in samples[a[0]]]
            med = statistics.median(us)
            byt = total * 256 / 1e6 if a[1] == 2 else mb
            res[a[0]] = {"us": med, "min": min(us), "tbps": byt / med, "ok": ok[a[0]], "cps": occs[a[0]][0]}
            if base is None:
                base = med
        sort_us = statistics.median([x.elapsed_time(y) * 1e3 for x, y in sort_s])
        results["shapes"][sh] = {"keys": nkeys, "pool": total, "mb": mb, "arms": res, "sort_us": sort_us}
        print(f"[fg] {sh} keys {nkeys} {mb:.1f} MB; torch segmented sort {sort_us:.1f} us", flush=True)
        for k, v in res.items():
            print(f"[fg] {sh} {k:15s} {v['us']:7.1f} us (min {v['min']:7.1f}) {v['tbps']:5.2f} TB/s x{base / v['us']:.3f}"
                  f" cps {v['cps']} {'ok' if v['ok'] else 'CHECKSUM-MISMATCH'}", flush=True)
        del data, kc, vc, lists, mask, pt
        torch.cuda.empty_cache()
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=1))
    print(f"[fg] wrote {p}", flush=True)


if __name__ == "__main__":
    main()

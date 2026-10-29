"""MC pair probe (request R-mc-1, notes/memory/gpu_requests.md): a hot expert split into two jobs (today's VML w72 re-stream:
both jobs stream the expert from HBM, not in lockstep) vs a 2-CTA cluster where each weight stage is TMA-multicast to both
CTAs (each CTA issues one 18 KB half with .multicast::cluster, ctaMask 0b11) and each CTA computes its own half of the tokens.

Kernel: kernels/lab/rev_pair.py (rev_probe = vml_probe2 + shipped v13 consumer switches, PG_EVN) plus:
  PG_MC   1: launched as 2-CTA clusters (cudaLaunchKernelEx, cluster dim 2); weight ring slots are filled by multicast halves;
             weight-slot empty barriers count both CTAs' consumer warps (local + remote mbarrier arrive, release.cluster);
             producer waits the empty phase then fence.acq_rel.cluster; producer tail waits the last NS releases so no CTA
             exits while its partner can still arrive on its barriers; cluster barrier after mbarrier init.
  PG_CLU  1: launched as 2-CTA clusters, nothing else changed (the cluster-launch cost for non-hot work).
Plans: std (rev_pair / vml_probe), pairI / lagI = rev_pair.plan_pair with partners (2c, 2c + 1) instead of (c, c + G/2), so
cluster ranks 0 / 1 are the two jobs of one expert.
"""
import torch

import rev_pair as RP

P2 = RP.P2
V1 = RP.V1


def _sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


_HELP = r"""#ifndef PG_MC
#define PG_MC 0
#endif
#ifndef PG_CLU
#define PG_CLU 0
#endif
__device__ __forceinline__ uint32_t mc_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ void bulk_g2s_mc(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint16_t mask) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "h"(mask) : "memory");
}
__device__ __forceinline__ void mbar_arrive_w(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory");
#if PG_MC
  uint32_t rb;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rb) : "r"(b), "r"(mc_rank() ^ 1u));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" :: "r"(rb) : "memory");
#endif
}
__device__ __forceinline__ void prod_w("""

H = RP.HEAD
H = _sub(H, "__device__ __forceinline__ void prod_w(", _HELP)
# producer: empty wait + cluster fence, multicast halves, tail
H = _sub(H, """      if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
      const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;""",
         """      if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
#if PG_MC
      if (it >= NS) asm volatile("fence.acq_rel.cluster;\\n" ::: "memory");
#endif
      const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;""")
H = _sub(H, """#if PG_EVN
      bulk_g2s(dst, wb + (size_t)s * SW, SW, fb);""",
         """#if PG_MC
      {
        const uint32_t r = mc_rank();
        bulk_g2s_mc(dst + r * SW, wb + (size_t)(r ? 16 + s : s) * SW, SW, fb, (uint16_t)3);
      }
#elif PG_EVN
      bulk_g2s(dst, wb + (size_t)s * SW, SW, fb);""")
H = _sub(H, """#endif
    }
  }
}
__device__ __forceinline__ void prod_x(""",
         """#endif
    }
  }
#if PG_MC
  for (int k = (it > NS ? it - NS : 0); k < it; ++k) mbar_wait(bars + 8 * (NS + k % NS), (k / NS) & 1, A.err);
#endif
}
__device__ __forceinline__ void prod_x(""")
B = RP.R.BODY if hasattr(RP.R, "BODY") else ""
CUDA0 = H + B
# weight-slot releases (every consumer path) -> local + remote arrive under PG_MC
n1 = CUDA0.count("mbar_arrive(bars + 8 * (NS + ws))")
n2 = CUDA0.count("mbar_arrive(bars + 8 * (PG_WS ? WSB + 4 * NS + q * NS + ws : NS + ws))")
assert n1 + n2 >= 1, (n1, n2)
CUDA = CUDA0.replace("mbar_arrive(bars + 8 * (NS + ws))", "mbar_arrive_w(bars + 8 * (NS + ws))").replace(
    "mbar_arrive(bars + 8 * (PG_WS ? WSB + 4 * NS + q * NS + ws : NS + ws))",
    "mbar_arrive_w(bars + 8 * (PG_WS ? WSB + 4 * NS + q * NS + ws : NS + ws))")
CUDA = _sub(CUDA, "mbar_init(bars + 8 * (NS + i), NCW2); }", "mbar_init(bars + 8 * (NS + i), NCW2 * (PG_MC ? 2 : 1)); }")
# cluster barrier after the init block (all threads present), only in cluster builds
_K = "__global__ void __launch_bounds__(NTH2, 1) vp2_kernel(const Args A) {"
_k0 = CUDA.index(_K)
_pre, CUDA = CUDA[:_k0], CUDA[_k0:]
CUDA = _pre + CUDA.replace("""    if (A.marks) { A.marks[(size_t)blockIdx.x * 8] = gtime(); A.marks[(size_t)blockIdx.x * 8 + 5] = clock64(); }
  }
  __syncthreads();
  const int2 rg = A.cta[blockIdx.x];""",
            """    if (A.marks) { A.marks[(size_t)blockIdx.x * 8] = gtime(); A.marks[(size_t)blockIdx.x * 8 + 5] = clock64(); }
  }
  __syncthreads();
#if PG_MC || PG_CLU
  asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;\\n" ::: "memory");
#endif
  const int2 rg = A.cta[blockIdx.x];""", 1)
assert CUDA.count("barrier.cluster.arrive.release.aligned") == 1
# host: cluster launch + occupancy of 2-CTA clusters
CUDA = _sub(CUDA, """  vp::VP2K<<<(int)G, vp::VP2T, vp::SMEM2, at::cuda::getCurrentCUDAStream()>>>(a);""",
            """#if PG_MC || PG_CLU
  {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)G); cfg.blockDim = dim3(vp::VP2T); cfg.dynamicSmemBytes = vp::SMEM2;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, vp::VP2K, a);
  }
#else
  vp::VP2K<<<(int)G, vp::VP2T, vp::SMEM2, at::cuda::getCurrentCUDAStream()>>>(a);
#endif""")
CUDA = _sub(CUDA, """  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vp::SMEM2, occ};""",
            """  int ncl = -1;
  {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(132); cfg.blockDim = dim3(vp::VP2T); cfg.dynamicSmemBytes = vp::SMEM2;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&ncl, vp::VP2K, &cfg) != cudaSuccess) { ncl = -2; cudaGetLastError(); }
  }
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)vp::SMEM2, occ, ncl};""")
CPP = RP.CPP
_S = {"PG_SC2": 1, "PG_DS": 1, "PG_CL": 1}
BUILDS = {"S": dict(_S), "SC": dict(_S, PG_CLU=1), "MC": dict(_S, PG_MC=1)}


def flags_of(b):
    return P2.V1.FLAGS + [f"-D{k}={v}" for k, v in BUILDS[b].items()]


def build(b, verbose=False):
    import hashlib
    import time
    from torch.utils.cpp_extension import load_inline
    fl = flags_of(b)
    tag = hashlib.sha1((CPP + CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"mc_pair_{b}_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["vp2_run", "vp2_attrs"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


def plan_pair_i(ntok, G, dev, mode="pair", per=7, seed=7):
    """rev_pair.plan_pair with partners (2c, 2c + 1): CTA 2c runs job A of pair c, CTA 2c + 1 job B (cluster ranks 0 / 1)."""
    p = RP.plan_pair(ntok, G, dev, mode, per, seed)
    half = G // 2
    u = p["unit"].view(G, per, 2)
    order = torch.tensor([c // 2 + (c % 2) * half for c in range(G)], dtype=torch.long, device=u.device)
    p["unit"] = u[order].reshape(G * per, 2).contiguous()
    return p


ref_job = RP.ref_job

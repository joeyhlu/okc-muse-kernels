"""K1-WG unit test (box, 1 CTA = 1 warpgroup): the PTX facts the CPU emulator cannot confirm.
  python3 kernels/muse_attention_decode/k1_wg_lab_unit.py
  S = Q.K^T : 2 x wgmma m64n64k32 e4m3 SS, Q and K in K-major INTERLEAVE smem (LBO 128, SBO 512) as k1_wg_design 3.1/3.2
  O = P.V   : 2 x wgmma m64n64k32 e4m3 RS, P packed from the S-accumulator positions into A registers with the key
              permutation key(s,k) = 16q + 8(i>>1) + 2t + (i&1); V^T in smem (LBO 128, SBO 272, region 2192) as 3.3
Checks S against dec(Q) @ dec(K)^T and O against e4m3(P) @ dec(V) (P given), both through the accumulator layout
(row 16w+g(+8), col 8j+2t+e)."""
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <stdint.h>
#define VT_SBO 272
#define VT_REGION 2192
__device__ __forceinline__ uint64_t desc_il(const void* p, int lbo, int sbo) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ int il(int r, int k, int lbo, int sbo) { return (r & 7) * 16 + (r >> 3) * sbo + (k & 15) + (k >> 4) * lbo; }

#define D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"

__device__ __forceinline__ void wg_ss(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", %32, %33, p, 1, 1;\n}\n"
               : D32(d) : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wg_rs(float (&d)[32], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, p, 1, 1;\n}\n"
               : D32(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t pk4(float a, float b, float c, float d) {
  const uint32_t lo = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}

__global__ void __launch_bounds__(128) unit(const uint8_t* Q, const uint8_t* K, const uint8_t* V, const float* P,
                                             float* S_out, float* O_out) {
  __shared__ __align__(1024) uint8_t qs[4096];
  __shared__ __align__(1024) uint8_t ks[4096];
  __shared__ __align__(1024) uint8_t vt[4 * VT_REGION];
  const int tid = threadIdx.x;
  for (int i = tid; i < 4096; i += 128) {
    const int r = i >> 6, k = i & 63;
    qs[il(r, k, 128, 512)] = Q[i];
    ks[il(r, k, 128, 512)] = K[i];
  }
  for (int i = tid; i < 4096; i += 128) {       // V^T: element (n = dim, logical k) of step s = V[key(s,k)][n]
    const int key = i >> 6, n = i & 63;
    const int q = key >> 4, b = (key >> 3) & 1, t = (key >> 1) & 3, e = key & 1;
    const int s = q >> 1, kl = 16 * (q & 1) + 4 * t + 2 * b + e;
    vt[s * 2 * VT_REGION + il(n, kl, 128, VT_SBO)] = V[i];
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  float d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wg_fence();
  wg_ss(d, desc_il(qs, 128, 512), desc_il(ks, 128, 512), 0);
  wg_ss(d, desc_il(qs + 256, 128, 512), desc_il(ks + 256, 128, 512), 1);
  wg_commit();
  wg_wait0();
  const int w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      S_out[(16 * w + g) * 64 + 8 * j + 2 * t + e] = d[4 * j + e];
      S_out[(16 * w + g + 8) * 64 + 8 * j + 2 * t + e] = d[4 * j + 2 + e];
    }
  }
  // P from global into the accumulator positions, then packed to A registers (set q: j = 2q, 2q+1)
  float pr[32];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      pr[4 * j + e] = P[(16 * w + g) * 64 + 8 * j + 2 * t + e];
      pr[4 * j + 2 + e] = P[(16 * w + g + 8) * 64 + 8 * j + 2 * t + e];
    }
  }
  float o[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) o[i] = 0.f;
  wg_fence();
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    uint32_t a[4];
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {                // h2 = 0: set q = 2s (a0 row g, a1 row g+8); 1: q = 2s+1 (a2, a3)
      const int q = 2 * s + h2, j0 = 2 * q, j1 = 2 * q + 1;
      a[2 * h2 + 0] = pk4(pr[4 * j0 + 0], pr[4 * j0 + 1], pr[4 * j1 + 0], pr[4 * j1 + 1]);
      a[2 * h2 + 1] = pk4(pr[4 * j0 + 2], pr[4 * j0 + 3], pr[4 * j1 + 2], pr[4 * j1 + 3]);
    }
    wg_rs(o, a, desc_il(vt + s * 2 * VT_REGION, 128, VT_SBO));
  }
  wg_commit();
  wg_wait0();
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      O_out[(16 * w + g) * 64 + 8 * j + 2 * t + e] = o[4 * j + e];
      O_out[(16 * w + g + 8) * 64 + 8 * j + 2 * t + e] = o[4 * j + 2 + e];
    }
  }
}

std::vector<torch::Tensor> run(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor P) {
  auto S = torch::zeros({64, 64}, P.options());
  auto O = torch::zeros({64, 64}, P.options());
  unit<<<1, 128>>>(Q.data_ptr<uint8_t>(), K.data_ptr<uint8_t>(), V.data_ptr<uint8_t>(), P.data_ptr<float>(),
                   S.data_ptr<float>(), O.data_ptr<float>());
  return {S, O};
}
"""
mod = load_inline("k1wg_unit", "std::vector<torch::Tensor> run(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor P);",
                  cuda_sources=SRC, functions=["run"],
                  extra_cuda_cflags=["-O3", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a"])
torch.manual_seed(0)
F8 = torch.float8_e4m3fn
q = torch.randn(64, 64).to(F8)
k = torch.randn(64, 64).to(F8)
v = torch.randn(64, 64).to(F8)
p = torch.rand(64, 64) * 256
S, O = mod.run(q.view(torch.uint8).cuda().contiguous(), k.view(torch.uint8).cuda().contiguous(),
               v.view(torch.uint8).cuda().contiguous(), p.cuda().contiguous())
torch.cuda.synchronize()
S_ref = q.float() @ k.float().T
O_ref = p.to(F8).float() @ v.float()
es = float((S.cpu() - S_ref).abs().max() / S_ref.abs().max())
eo = float((O.cpu() - O_ref).abs().max() / O_ref.abs().max())
print(f"UNIT S (SS e4m3, INTERLEAVE LBO128/SBO512): max rel err {es:.3e}  {'OK' if es < 1e-3 else 'FAIL'} (FP8 wgmma accumulates at ~2^-13; a layout error is O(1))")
print(f"UNIT O (RS e4m3, packed P, V^T SBO272/region2192): max rel err {eo:.3e}  {'OK' if eo < 1e-3 else 'FAIL'}")

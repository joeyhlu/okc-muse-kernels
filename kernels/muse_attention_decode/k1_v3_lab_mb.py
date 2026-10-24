"""K1 v3 lab (R1): per-SM throughput of the instructions the L=8 tile loop is built from, on the box H200.
Run: python3 kernels/muse_attention_decode/k1_v3_lab_mb.py
Each kernel: one wave (132 x BPS blocks x 256 thr), 8 independent chains per thread; reports results per SM per clock
(clock64 cycles of the slowest block). 'vals' = values produced per clock (2 per lane for x2 ops)."""
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <stdint.h>
#include <cuda_fp16.h>
#define N 8
#define IT 512
__device__ __forceinline__ uint32_t f2u(float f) { return __float_as_uint(f); }

template <int OP>
__global__ void __launch_bounds__(256) mb(uint32_t* out, long long* cyc, uint32_t seed) {
  uint32_t r[N]; float f[N]; float acc[N][4];
#pragma unroll
  for (int i = 0; i < N; ++i) { r[i] = seed * (threadIdx.x + 7 * i + 1); f[i] = 0.001f * (i + threadIdx.x % 7);
    acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < IT; ++it) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (OP == 0) { asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[i])); }
      if (OP == 1) { asm volatile("ex2.approx.f16x2 %0, %0;" : "+r"(r[i])); }
      if (OP == 2) { uint32_t y; asm volatile("{.reg .b16 lo; mov.b32 {lo, _}, %1; cvt.rn.f16x2.e4m3x2 %0, lo;}" : "=r"(y) : "r"(r[i])); r[i] = y; }
      if (OP == 3) { uint32_t y; asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(y) : "f"(f[i]), "f"(f[(i + 1) % N])); f[i] = __uint_as_float(y); }
      if (OP == 4) { uint32_t y; asm volatile("{.reg .b16 lo; cvt.rn.satfinite.e4m3x2.f32 lo, %1, %2; mov.b32 %0, {lo, lo};}" : "=r"(y) : "f"(f[i]), "f"(f[(i + 1) % N])); f[i] = __uint_as_float(y & 0x3fff3fffu); }
      if (OP == 5) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3]) : "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5])); }
      if (OP == 6) { asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3]) : "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5])); }
      if (OP == 7) { asm volatile("fma.rn.f32 %0, %0, %0, %1;" : "+f"(f[i]) : "f"(f[(i + 3) % N])); }
      if (OP == 8) { asm volatile("max.f32 %0, %0, %1;" : "+f"(f[i]) : "f"(f[(i + 3) % N])); }
      if (OP == 9) { asm volatile("prmt.b32 %0, %0, %1, 0x5140;" : "+r"(r[i]) : "r"(r[(i + 3) % N])); }
      if (OP == 10) { asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                     : "+r"(r[i]), "+r"(r[(i + 4) % N]) : "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5])); }
      if (OP == 11) { asm volatile("fma.rn.f16x2 %0, %0, %0, %1;" : "+r"(r[i]) : "r"(r[(i + 3) % N])); }
    }
  }
  long long t1 = clock64();
  uint32_t s = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) s ^= r[i] ^ f2u(f[i]) ^ f2u(acc[i][0] + acc[i][3]);
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

torch::Tensor run(int64_t op, int64_t blocks) {
  auto out = torch::empty({blocks * 256}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto cyc = torch::empty({blocks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  uint32_t* o = (uint32_t*)out.data_ptr<int>(); long long* c = (long long*)cyc.data_ptr<int64_t>();
  switch (op) {
    case 0: mb<0><<<blocks, 256>>>(o, c, 3u); break; case 1: mb<1><<<blocks, 256>>>(o, c, 3u); break;
    case 2: mb<2><<<blocks, 256>>>(o, c, 3u); break; case 3: mb<3><<<blocks, 256>>>(o, c, 3u); break;
    case 4: mb<4><<<blocks, 256>>>(o, c, 3u); break; case 5: mb<5><<<blocks, 256>>>(o, c, 3u); break;
    case 6: mb<6><<<blocks, 256>>>(o, c, 3u); break; case 7: mb<7><<<blocks, 256>>>(o, c, 3u); break;
    case 8: mb<8><<<blocks, 256>>>(o, c, 3u); break; case 9: mb<9><<<blocks, 256>>>(o, c, 3u); break;
    case 10: mb<10><<<blocks, 256>>>(o, c, 3u); break; case 11: mb<11><<<blocks, 256>>>(o, c, 3u); break;
  }
  return cyc;
}
"""
CPP = "torch::Tensor run(int64_t op, int64_t blocks);"
mod = load_inline("k1v3_mb", CPP, cuda_sources=SRC, functions=["run"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"])
NAMES = ["ex2.f32", "ex2.f16x2", "cvt.f16x2.e4m3x2", "cvt.f16x2.f32 (pack)", "cvt.e4m3x2.f32",
         "mma m16n8k16 f16->f32", "mma m16n8k32 e4m3->f32", "ffma", "fmax", "prmt", "mma m16n8k16 f16->f16",
         "hfma2"]
sms = torch.cuda.get_device_properties(0).multi_processor_count
for bps in (4, 8):
    blocks = sms * bps
    for op, nm in enumerate(NAMES):
        mod.run(op, blocks)
        torch.cuda.synchronize()
        cyc = mod.run(op, blocks)
        torch.cuda.synchronize()
        cmax = float(cyc.max())
        # blocks are co-resident (bps x 256 thr per SM); per SM: bps blocks * 256 thr * N * IT ops in cmax clocks
        warp_ops = bps * 8 * 8 * 512 / cmax          # warp-instructions per SM per clock
        print(f"bps={bps} {nm:26s} {warp_ops:6.3f} warp-instr/clk/SM = {32 * warp_ops:6.1f} lane-ops/clk/SM", flush=True)

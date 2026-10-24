"""k2b_v1_unit.py: box unit test for K2b v1's SS wgmma operand layouts. Run this FIRST in the GPU burst (~1 min):
  python3 kernels/muse_prefill_layer/k2b_v1_unit.py
It compiles the kernel's own index macros and helpers (the K2B_KERNEL_SRC prefix of k2b_v1.py, up to the producer)
and, for LAYOUT in {SW128 (0), IL (1)} x N in {128, 256}:
  * stages A [128 x 64k] and B [N x 64k] (BF16, K-major) into two smem slots each with K2B_ADDR_SW / K2B_ADDR_IL,
    at 1024 B-aligned buffer offsets like the kernel's ring (slot 1 is a non-zero base);
  * 2 consumer warpgroups (M split: WG c reads A rows 64c..64c+63 at +8192 B) issue 4 x wgmma m64nNk16 per stage with
    make_desc(addr, LBO, SBO, LT) + k16 * (KADV >> 4) exactly as the kernel; 3 stages (slots 0, 1, 0);
  * stores the accumulators through K2B_ACC_ROW / K2B_ACC_COL and compares bitwise with A @ B^T (integer-valued
    inputs, exact in FP32).
Negative control: SW128-staged data read with the IL descriptor must mismatch (proves the test can fail).
CPA cases stage through the kernel's own path instead: cp_async16 (A row 5 zero-filled with src-size 0) +
cp.async.mbarrier.arrive.noinc (count 384) + bounded mbar_wait with the start-phase parity, err flag checked.
Prints one line per case and 'UNIT PASS' / 'UNIT FAIL'.
"""
import pathlib

import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve().parent
SRC = (HERE / "k2b_v1.py").read_text()
G = {"torch": torch}
_pure = SRC.split("# ===== K2B PURE BEGIN =====")[1].split("# ===== K2B PURE END =====")[0].split("\n", 1)[1]
exec(compile(_pure, "k2b_v1.py:pure", "exec"), G)
KS = G["K2B_KERNEL_SRC"]
PREFIX = KS.split("template <class C>\n__device__ __forceinline__ int next_valid")[0]
PREFIX = PREFIX.replace("@WGMMA@", G["k2b_wgmma_src"](128) + G["k2b_wgmma_src"](256))

UNIT = r"""
template <int N, int LAYOUT, int DLT, int CPA>
__global__ void __launch_bounds__(384, 1) unit_k(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int kst,
                                                 int* err) {
  extern __shared__ uint8_t smem_raw[];
  const uint32_t sraw = smem_u32(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  uint8_t* sp = smem_raw + (sbase - sraw);
  constexpr int A_BYTES = 128 * 128, B_BYTES = N * 128, OFF_B = 2 * A_BYTES;
  constexpr int LBO = DLT == 0 ? K2B_LBO_SW : K2B_LBO_IL;
  constexpr int LT = DLT == 0 ? K2B_LT_SW : K2B_LT_IL;
  constexpr int KADV = DLT == 0 ? K2B_KADV_SW : K2B_KADV_IL;
  const int tid = threadIdx.x, wg = tid >> 7, tw = tid & 127;
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  const int kdim = 64 * kst;
  const uint32_t bar = sbase + OFF_B + 2 * B_BYTES;   // 2 mbarriers (one per slot), count 384
  if (CPA && tid == 0) {
    mbar_init(bar, 384);
    mbar_init(bar + 8, 384);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  for (int ks = 0; ks < kst; ++ks) {
    const int slot = ks & 1;
    if (CPA) {   // the kernel's path: cp.async 16 B (A row 5 zero-filled via src-size 0) + arrive.noinc + bounded wait
      for (int idx = tid; idx < 128 * 8; idx += 384) {
        const int r = idx >> 3, c = idx & 7;
        const int off = LAYOUT == 0 ? K2B_ADDR_SW(r, c) : K2B_ADDR_IL(r, c);
        cp_async16(sbase + slot * A_BYTES + off, A + (size_t)r * kdim + ks * 64 + c * 8, r == 5 ? 0 : 16);
      }
      for (int idx = tid; idx < N * 8; idx += 384) {
        const int r = idx >> 3, c = idx & 7;
        const int off = LAYOUT == 0 ? K2B_ADDR_SW(r, c) : K2B_ADDR_IL(r, c);
        cp_async16(sbase + OFF_B + slot * B_BYTES + off, B + (size_t)r * kdim + ks * 64 + c * 8, 16);
      }
      cpasync_arrive_noinc(bar + 8 * slot);
      mbar_wait(bar + 8 * slot, (ks >> 1) & 1, err, 9);
    } else
    for (int idx = tid; idx < 128 * 8; idx += 384) {
      const int r = idx >> 3, c = idx & 7;
      const int off = LAYOUT == 0 ? K2B_ADDR_SW(r, c) : K2B_ADDR_IL(r, c);
      *reinterpret_cast<uint4*>(sp + slot * A_BYTES + off) =
          *reinterpret_cast<const uint4*>(A + (size_t)r * kdim + ks * 64 + c * 8);
    }
    if (!CPA)
    for (int idx = tid; idx < N * 8; idx += 384) {
      const int r = idx >> 3, c = idx & 7;
      const int off = LAYOUT == 0 ? K2B_ADDR_SW(r, c) : K2B_ADDR_IL(r, c);
      *reinterpret_cast<uint4*>(sp + OFF_B + slot * B_BYTES + off) =
          *reinterpret_cast<const uint4*>(B + (size_t)r * kdim + ks * 64 + c * 8);
    }
    fence_proxy_async();
    __syncthreads();
    if (wg >= 1) {
      const int c = wg - 1;
      const uint64_t da = make_desc(sbase + slot * A_BYTES + c * 8192, LBO, K2B_SBO, LT);
      const uint64_t db = make_desc(sbase + OFF_B + slot * B_BYTES, LBO, K2B_SBO, LT);
      fence_acc<N / 2>(acc);
      wg_fence();
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16)
        wgmma_ss<N>(acc, da + (uint64_t)(k16 * (KADV >> 4)), db + (uint64_t)(k16 * (KADV >> 4)), (ks | k16) != 0);
      wg_commit();
      wg_wait<0>();
      fence_acc<N / 2>(acc);
    }
    __syncthreads();
  }
  if (wg >= 1) {
    const int c = wg - 1, w = tw >> 5, lane = tw & 31;
#pragma unroll
    for (int v = 0; v < N / 2; ++v)
      C[(size_t)(64 * c + K2B_ACC_ROW(w, lane, v)) * N + K2B_ACC_COL(v, lane)] = acc[v];
  }
}

template <int N, int LAYOUT, int DLT, int CPA>
void launch_unit(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor E, int kst) {
  const int smem = 2 * 128 * 128 + 2 * N * 128 + 16 + 1024;
  TORCH_CHECK(cudaFuncSetAttribute(unit_k<N, LAYOUT, DLT, CPA>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
  unit_k<N, LAYOUT, DLT, CPA><<<1, 384, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
      C.data_ptr<float>(), kst, E.data_ptr<int>());
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
}  // namespace k2b1

void unit(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor E, int64_t n, int64_t layout, int64_t dlt,
          int64_t cpa, int64_t kst) {
  if (n == 128 && layout == 0 && dlt == 0 && !cpa) k2b1::launch_unit<128, 0, 0, 0>(A, B, C, E, (int)kst);
  else if (n == 128 && layout == 1 && dlt == 1 && !cpa) k2b1::launch_unit<128, 1, 1, 0>(A, B, C, E, (int)kst);
  else if (n == 256 && layout == 0 && dlt == 0 && !cpa) k2b1::launch_unit<256, 0, 0, 0>(A, B, C, E, (int)kst);
  else if (n == 256 && layout == 1 && dlt == 1 && !cpa) k2b1::launch_unit<256, 1, 1, 0>(A, B, C, E, (int)kst);
  else if (n == 128 && layout == 0 && dlt == 1 && !cpa) k2b1::launch_unit<128, 0, 1, 0>(A, B, C, E, (int)kst);
  else if (n == 256 && layout == 0 && dlt == 0 && cpa) k2b1::launch_unit<256, 0, 0, 1>(A, B, C, E, (int)kst);
  else if (n == 256 && layout == 1 && dlt == 1 && cpa) k2b1::launch_unit<256, 1, 1, 1>(A, B, C, E, (int)kst);
  else TORCH_CHECK(false, "unit: case not instantiated");
}
"""
CPP = ("#include <torch/extension.h>\nvoid unit(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor E, int64_t n, "
       "int64_t layout, int64_t dlt, int64_t cpa, int64_t kst);\n")


def main():
    import sys
    mod = load_inline("k2b_v1_unit", CPP, cuda_sources=PREFIX + UNIT, functions=["unit"], verbose="--build-only" in sys.argv,
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a",
                                         "-Xptxas=-v"])
    if "--build-only" in sys.argv:   # CPU-only warm-up of the build cache (box.sh run, outside the GPU lock)
        print("[unit] built", flush=True)
        return
    torch.manual_seed(0)
    kst = 3
    ok = True
    for n, layout, dlt, cpa, expect in ((128, 0, 0, 0, True), (256, 0, 0, 0, True), (128, 1, 1, 0, True),
                                        (256, 1, 1, 0, True), (128, 0, 1, 0, False), (256, 0, 0, 1, True),
                                        (256, 1, 1, 1, True)):
        A = (torch.randint(-8, 8, (128, 64 * kst), device="cuda") * 0.25).to(torch.bfloat16)
        B = (torch.randint(-8, 8, (n, 64 * kst), device="cuda") * 0.5).to(torch.bfloat16)
        C = torch.full((128, n), float("nan"), device="cuda")
        E = torch.zeros((4,), dtype=torch.int32, device="cuda")
        mod.unit(A, B, C, E, n, layout, dlt, cpa, kst)
        torch.cuda.synchronize()
        Az = A.float().clone()
        if cpa:
            Az[5] = 0.0          # cp.async src-size 0 zero-fills row 5
        ref = Az @ B.float().T
        exact = bool(torch.equal(C, ref)) and int(E[0]) == 0
        bad = int((C != ref).sum())
        tag = "PASS" if exact == expect else "FAIL"
        ok &= exact == expect
        print(f"[unit] N={n} staged={'SW128' if layout == 0 else 'IL'}{' via cp.async+mbarrier' if cpa else ''} "
              f"desc={'SW128' if dlt == 0 else 'IL'}: exact={exact} (mismatches {bad}/{C.numel()}, err {int(E[0])}), "
              f"expected exact={expect} -> {tag}", flush=True)
    print("UNIT PASS" if ok else "UNIT FAIL", flush=True)


if __name__ == "__main__":
    main()

"""k2b_v2_unit.py: box unit test for K2b v2's consumer stage (RS wgmma, K2A lane items as A, permuted SW128 B).
  python3 kernels/muse_prefill_layer/k2b_v2_unit.py [--build-only]
Compiles k2b_v2.py's own kernel helpers (the K2B_KERNEL_SRC prefix, including k2b_v2_maps.cuh) and drives the REAL
`k2b2::consumer_stage<C, N>` (mbarrier wait, LDS item + scale, deq4 + bmul2 into the A buffer, 4 x wgmma RS,
wait_group 1, release) on one CTA:
  * a random 128-row x 256-k NVFP4 matrix (incl. wide/zero/negative scales) repacked with K2A v2.2's prepack_expert;
  * WG0 stages each k64 stage (8 K2A tiles' codes + scale words, N activation rows of x_perm at K2B_ADDR_SW) into
    its own ring slot, fences and arrives on full[slot] exactly like producer_publish;
  * the 2 consumer WGs run consumer_stage for the 4 stages, then store acc through K2B2_ACC_WROW / K2B2_ACC_TOK.
Checks C [128 rows, N] == (W e2m1*sf * 2^-8) @ x^T (fp32 accumulate vs float64; integer-ish inputs) for
N in {16, 48, 128, 256} and that err stays 0. Prints 'UNIT PASS' / 'UNIT FAIL'.
"""
import pathlib
import sys

import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve().parent
SRC = (HERE / "k2b_v2.py").read_text()
G = {"torch": torch}
_pure = SRC.split("# ===== K2B PURE BEGIN =====")[1].split("# ===== K2B PURE END =====")[0].split("\n", 1)[1]
exec(compile(_pure, "k2b_v2.py:pure", "exec"), G)
KS = G["K2B_KERNEL_SRC"].replace("@MAPS@", G["K2B2_MAPS"])
PREFIX = KS.split("template <class C>\n__device__ __forceinline__ int next_valid")[0]
PREFIX = PREFIX.replace("@WGMMA@", "".join(G["k2b_wgmma_rs_src"](n) for n in (16, 48, 128, 256)))
# consumer_stage sits after next_valid in the source: append it (and its deps are all in PREFIX)
STAGE = KS[KS.index("template <class C, int N>\n__device__ __forceinline__ void consumer_stage"):
           KS.index("template <class C, int N>\n__device__ __forceinline__ void consumer_tile")]

UNIT = r"""
template <int N>
__global__ void __launch_bounds__(384, 1) unit_k(const uint4* wq, const uint32_t* sq, const __nv_bfloat16* xp,
                                                 float* out, int* err) {
  using C = Cfg<2, 4, 0>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sbase = smem_u32(smem_raw);
  uint8_t* sp = smem_raw;
  const int tid = threadIdx.x, wg = tid >> 7, tw = tid & 127, lane = tid & 31;
  Params P;
  P.err = err; P.probe = 0;
  if (tid == 0) {
    for (int s = 0; s < C::S; ++s) { mbar_init(sbase + C::OFF_BAR + 8 * s, 4); mbar_init(sbase + C::OFF_BAR + 8 * (C::S + s), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (wg == 0) {                       // "producer": stage all 4 k64 stages into slots 0..3, publish each
    asm volatile("setmaxnreg.dec.sync.aligned.u32 56;\n" ::: "memory");   // frees the pool the consumers grow into
    for (int kb = 0; kb < 4; ++kb) {
      for (int idx = tw; idx < 256; idx += 128)       // codes: item (tile i, lane) of stage kb
        *reinterpret_cast<uint4*>(sp + C::OFF_C + kb * C::C_BYTES + idx * 16) = wq[((idx >> 5) * 4 + kb) * 32 + (idx & 31)];
      for (int idx = tw; idx < 256; idx += 128)       // scale words (tile i, kb pair, lane)
        *reinterpret_cast<uint32_t*>(sp + C::OFF_SC + kb * C::SC_BYTES + idx * 4) = sq[((idx >> 5) * 2 + (kb >> 1)) * 32 + (idx & 31)];
      for (int idx = tw; idx < N * 8; idx += 128) {   // activation rows (logical k order), SW128
        const int r = idx >> 3, c = idx & 7;
        *reinterpret_cast<uint4*>(sp + C::OFF_A + kb * C::A_BYTES + K2B_ADDR_SW(r, c)) =
            *reinterpret_cast<const uint4*>(xp + (size_t)r * 256 + kb * 64 + c * 8);
      }
      fence_proxy_async();
      producer_bar();
      __syncwarp();
      if (lane == 0) mbar_arrive(sbase + C::OFF_BAR + 8 * kb);
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 224;\n" ::: "memory");
    const int c = wg - 1, w = tw >> 5, it = c * 4 + w;
    float acc[N / 2];
    uint32_t a0[16], a1[16];
    for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
    for (int i = 0; i < 16; ++i) { a0[i] = 0u; a1[i] = 0u; }
    for (int kb = 0; kb < 4; kb += 2) {
      consumer_stage<C, N>(P, sbase, kb, kb, it, lane, acc, a0);
      consumer_stage<C, N>(P, sbase, kb + 1, kb + 1, it, lane, acc, a1);
    }
    wg_wait<0>();
    fence_acc<N / 2>(acc);
    for (int v = 0; v < N / 2; ++v)
      out[(size_t)(64 * c + K2B2_ACC_WROW(w, lane, v)) * N + K2B2_ACC_TOK(v, lane)] = acc[v];
  }
}
}  // namespace k2b2

void unit(torch::Tensor wq, torch::Tensor sq, torch::Tensor xp, torch::Tensor out, torch::Tensor err, int64_t n) {
  using C = k2b2::Cfg<2, 4, 0>;
  auto s = at::cuda::getCurrentCUDAStream();
  auto W = reinterpret_cast<const uint4*>(wq.data_ptr());
  auto S = reinterpret_cast<const uint32_t*>(sq.data_ptr());
  auto X = reinterpret_cast<const __nv_bfloat16*>(xp.data_ptr());
#define K2B2_UNIT_CASE(NN) if (n == NN) { \
    TORCH_CHECK(cudaFuncSetAttribute(k2b2::unit_k<NN>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) == cudaSuccess); \
    k2b2::unit_k<NN><<<1, 384, C::SMEM, s>>>(W, S, X, out.data_ptr<float>(), err.data_ptr<int>()); }
  K2B2_UNIT_CASE(16) K2B2_UNIT_CASE(48) K2B2_UNIT_CASE(128) K2B2_UNIT_CASE(256)
  TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""
CPP = ("#include <torch/extension.h>\nvoid unit(torch::Tensor wq, torch::Tensor sq, torch::Tensor xp, torch::Tensor out, "
       "torch::Tensor err, int64_t n);\n")


def main():
    mod = load_inline("k2b_v2_unit", CPP, cuda_sources=PREFIX + STAGE + UNIT, functions=["unit"],
                      verbose="--build-only" in sys.argv,
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a",
                                         "-Xptxas=-v", "-Xptxas=-warn-spills"])
    if "--build-only" in sys.argv:
        print("[unit] built", flush=True)
        return
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    nrow, k = 128, 256
    w = torch.randint(0, 256, (nrow, k // 2), generator=g, dtype=torch.int64).to(torch.uint8)
    sf = torch.randint(0x30, 0x48, (nrow, k // 16), generator=g, dtype=torch.int64)
    sf[::7, ::3] = torch.randint(0, 0x7F, sf[::7, ::3].shape, generator=g)     # some wide / subnormal / zero scales
    sf[5, 2] = 0
    sf[9, 1] = 0xB8                                                              # one negative scale
    sf = sf.to(torch.uint8)
    wq, sq = G["prepack_expert"](w, sf, gate_up=False)
    codes = torch.stack((w & 15, w >> 4), -1).reshape(nrow, k).long()
    e2m1 = torch.tensor(G["_K2B_E2M1"], dtype=torch.float64)
    vals = torch.where((codes & 8) != 0, -e2m1[codes & 7], e2m1[codes & 7])
    sfv = sf.view(torch.float8_e4m3fn).to(torch.float64)
    wd = (vals.view(nrow, k // 16, 16) * sfv.unsqueeze(-1)).view(nrow, k) * 2.0 ** -8   # prepacked = natural (no gate/up)
    perm = torch.tensor([64 * (c // 64) + G["k2b2_perm64"](c % 64) for c in range(k)])
    ok = True
    for n in (16, 48, 128, 256):
        x = (torch.randint(-8, 8, (n, k), generator=g) * 0.25).to(torch.bfloat16)
        xp = x[:, perm].contiguous()
        out = torch.full((nrow, n), float("nan"), device="cuda")
        err = torch.zeros((4,), dtype=torch.int32, device="cuda")
        mod.unit(wq.cuda(), sq.cuda(), xp.cuda(), out, err, n)
        torch.cuda.synchronize()
        ref = wd @ x.double().T
        diff = (out.double().cpu() - ref).abs().max().item()
        rel = diff / ref.abs().max().item()
        passed = rel < 1e-5 and int(err[0]) == 0 and not bool(torch.isnan(out).any())
        ok &= passed
        print(f"[unit] N={n}: max rel {rel:.2e}, err {int(err[0])} -> {'PASS' if passed else 'FAIL'}", flush=True)
    print("UNIT PASS" if ok else "UNIT FAIL", flush=True)


if __name__ == "__main__":
    main()

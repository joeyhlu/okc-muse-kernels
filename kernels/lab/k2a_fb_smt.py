"""SMT helper text for kernels/lab/k2a_fb_lab.py (per-SM arrival ticket role assignment, lab only)."""
SMT_HELPERS = r'''// SMT (lab): logical CTA index from a per-SM arrival ticket: the first CTA on each SM takes the lowest free index
// (0, 1, ...), later ones count down from G - 1, so a role split "first Gh CTAs hot" puts at most one hot CTA per SM when
// Gh <= SMs and no SM with two cold CTAs when Gh >= SMs. Counters in P.xw (int view, unused by vid 0 / 3 / 5):
// [k * 1024 + smid] arrivals, [k * 1024 + 512 / 513] first / later index counters, [k * 1024 + 514] exits.
__shared__ int fb_lcta;
__device__ __forceinline__ void fb_sm_ticket(const Params& P, int k) {
  if (threadIdx.x == 0) {
    if constexpr (SMT != 0) {
      int* T = reinterpret_cast<int*>(P.xw) + k * 1024;
      unsigned smid;
      asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      const int t = atomicAdd(T + (smid & 511), 1);
      fb_lcta = (t == 0) ? atomicAdd(T + 512, 1) : (int)gridDim.x - 1 - atomicAdd(T + 513, 1);
    } else {
      fb_lcta = (int)blockIdx.x;
    }
  }
}
__device__ __forceinline__ void fb_sm_exit(const Params& P, int k) {   // thread 0 after its gemm
  if constexpr (SMT != 0) {
    if (threadIdx.x == 0) {
      int* T = reinterpret_cast<int*>(P.xw) + k * 1024;
      __threadfence();
      if (atomicAdd(T + 514, 1) == (int)gridDim.x - 1)
        for (int i = 0; i < 515; ++i) atomicExch(T + i, 0);
    }
  }
}
'''
